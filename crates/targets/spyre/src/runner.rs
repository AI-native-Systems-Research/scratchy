// SPDX-License-Identifier: Apache-2.0
//! Execute an emitted KTIR bundle on the `ktir_emulator` emulator — Spyre's
//! hardware-free run path. Behind the `runner` feature (it links `ktir_emulator`);
//! the emulator-free manifest/fill helpers live in [`crate::manifest`].
//!
//! The serving worker runs the macro-embedded bundle in-memory via
//! [`run_bundle_all_embedded`] — no disk. The disk [`run_bundle`] /
//! [`run_bundle_all`] below back the `#[cfg(test)]` numeric gate, which reads a
//! bundle dumped at emit time via `SCRATCHY_KTIR_DUMP=<dir>` (`manifest.json`
//! plus one `node{i}.mlir` per op). Each keeps one host buffer per tensor id and
//! threads them through the nodes in emit order (topological), returning the
//! `result` tensor.
//!
//! Buffers cross the host boundary as f32 ([`Arg::Tensor`]); the emulator
//! marshals them into the f16 the memrefs declare and reads them back as f32.

use crate::manifest::Manifest;
use anyhow::{Context, Result, anyhow};
use ktir_emulator::dtypes::DType;
use ktir_emulator::interpreter::{Arg, execute_function};
use ktir_emulator::ktir_optimizer::fusion::{Binding, NodeSpec, ProgramSpec};
use ktir_emulator::parser::parse_module;
use ktir_emulator::program::Session;
use scratchy_tensors::DType as SDType;
use std::collections::HashMap;
use std::path::Path;

/// Run every node in order, threading `bufs` (one buffer per tensor id) through
/// the emulator, fill the runtime length mask, and return ALL buffers.
///
/// `decode_pos` overrides `manifest.decode_position` — a generation loop passes
/// the live decode position so one bundle serves every step (the AttnDecode
/// prefix masks columns past it); `None` uses the bundle's baked value (the
/// self-check gate). Sources must already be filled in `bufs`; the mask buffer
/// (`manifest.attn_mask`) is filled here. Each node reads its input buffers
/// (f32 `Arg::Tensor`s the emulator widens to f16), executes, and its single
/// output buffer is overwritten with the read-back result. Returning every
/// buffer lets a worker lift each layer's roped-K / V into its KV cache.
pub fn run_bundle_all(
    dir: &Path,
    manifest: &Manifest,
    bufs: Vec<Vec<f32>>,
    decode_pos: Option<u32>,
) -> Result<Vec<Vec<f32>>> {
    // Read each node's MLIR off disk up front (the self-check / dump path), then
    // thread them through the shared executor. The serving worker uses the
    // embedded variant below instead — no disk.
    let texts: Vec<String> = manifest
        .nodes
        .iter()
        .map(|n| {
            std::fs::read_to_string(dir.join(&n.mlir)).with_context(|| format!("read {}", n.mlir))
        })
        .collect::<Result<_>>()?;
    run_nodes(|ni| texts[ni].as_str(), manifest, bufs, decode_pos)
}

/// Like [`run_bundle_all`] but the node MLIR comes from in-memory `&str`s
/// (`nodes[ni]` = `(func, mlir)` for `manifest.nodes[ni]`, index-aligned) — the
/// macro-embedded bundle path the serving worker takes. ZERO disk reads.
pub fn run_bundle_all_embedded(
    nodes: &[(&str, &str)],
    manifest: &Manifest,
    bufs: Vec<Vec<f32>>,
    decode_pos: Option<u32>,
) -> Result<Vec<Vec<f32>>> {
    run_nodes(|ni| nodes[ni].1, manifest, bufs, decode_pos)
}

/// Shared per-node executor. `node_text(ni)` returns the MLIR for
/// `manifest.nodes[ni]`; the disk and embedded entry points differ only in how
/// they source that text.
fn run_nodes<'a>(
    node_text: impl Fn(usize) -> &'a str,
    manifest: &Manifest,
    mut bufs: Vec<Vec<f32>>,
    decode_pos: Option<u32>,
) -> Result<Vec<Vec<f32>>> {
    if let Some(mask_id) = manifest.attn_mask {
        let cap = manifest.tensors[mask_id].cols;
        let dp = decode_pos.unwrap_or(manifest.decode_position) as usize;
        bufs[mask_id] = crate::manifest::attn_mask_fill(cap, dp);
    }
    let dbg = std::env::var_os("SPYRE_DEBUG").is_some();
    for (ni, node) in manifest.nodes.iter().enumerate() {
        let text = node_text(ni);
        let module =
            parse_module(text).map_err(|e| anyhow!("parse node{ni} ({}): {e}", node.func))?;
        let args: Vec<(&str, Arg)> = node
            .args
            .iter()
            .map(|a| {
                (
                    a.name.as_str(),
                    Arg::Tensor {
                        data: bufs[a.tensor].clone(),
                        shape: manifest.shape(a.tensor),
                        dtype: DType::F16,
                    },
                )
            })
            .collect();
        let outputs = execute_function(&module, &node.func, &args)
            .map_err(|e| anyhow!("execute {}: {e}", node.func))?;
        for a in &node.args {
            if a.is_output {
                let o = outputs.get(&a.name).ok_or_else(|| {
                    anyhow!("node {} produced no output for arg {}", node.func, a.name)
                })?;
                bufs[a.tensor] = o.data.clone();
                if dbg {
                    // Flag dead (all-zero) or non-finite outputs — the signals a
                    // broken forward leaves (e.g. an f16 overflow killing a
                    // layer). Quiet on healthy nodes.
                    let o = &bufs[a.tensor];
                    let nz = o.iter().filter(|x| **x != 0.0).count();
                    let nan = o.iter().filter(|x| !x.is_finite()).count();
                    if nz == 0 || nan > 0 {
                        let mx = o.iter().map(|x| x.abs()).fold(0f32, f32::max);
                        let ins: Vec<usize> = node
                            .args
                            .iter()
                            .filter(|x| !x.is_output)
                            .map(|x| x.tensor)
                            .collect();
                        eprintln!(
                            "[runner-dbg] node{ni} {} out_t={} nonzero={nz} nonfinite={nan} maxabs={mx:.3} <- in {ins:?}",
                            node.func, a.tensor
                        );
                    }
                }
            }
        }
    }
    Ok(bufs)
}

/// Run the bundle and return just the `result` tensor (logits). The mask is
/// filled from the bundle's baked `decode_position` (the self-check path).
pub fn run_bundle(dir: &Path, manifest: &Manifest, bufs: Vec<Vec<f32>>) -> Result<Vec<f32>> {
    let bufs = run_bundle_all(dir, manifest, bufs, None)?;
    Ok(bufs[manifest.result].clone())
}

/// Build a [`ProgramSpec`] from the manifest: node order + arg↔tensor bindings,
/// the source tensor set (every `is_source` id plus the runtime length mask),
/// and the result set. `extra_results` are intermediate op-output ids the caller
/// reads back (the per-layer new roped-K / new-V) — adding them to `results`
/// keeps fusion from inlining them away, so they stay HBM-materialized.
fn build_spec(manifest: &Manifest, extra_results: &[usize]) -> ProgramSpec {
    let nodes = manifest
        .nodes
        .iter()
        .map(|n| NodeSpec {
            func: n.func.clone(),
            bindings: n
                .args
                .iter()
                .map(|a| Binding {
                    // Manifest arg names carry no `%`; the fused-program binding
                    // wants the SSA form (matches ktir-emulator's own bundle loader).
                    arg: format!("%{}", a.name),
                    tensor: a.tensor as u64,
                    is_output: a.is_output,
                })
                .collect(),
        })
        .collect();
    let mut sources: std::collections::HashSet<u64> = manifest
        .tensors
        .iter()
        .filter(|t| t.is_source)
        .map(|t| t.id as u64)
        .collect();
    if let Some(m) = manifest.attn_mask {
        sources.insert(m as u64);
    }
    let mut results: std::collections::HashSet<u64> =
        std::collections::HashSet::from([manifest.result as u64]);
    results.extend(extra_results.iter().map(|&id| id as u64));
    ProgramSpec {
        nodes,
        sources,
        results,
    }
}

/// Build the zero/one-pass ktir ingest `Arg` for a typed-byte weight.
fn weight_arg(data: Vec<u8>, dt: SDType, shape: Vec<usize>) -> Arg {
    match dt {
        SDType::BF16 => Arg::TensorBf16 { data, shape },
        SDType::F16 => Arg::TensorBytes {
            data,
            shape,
            dtype: DType::F16,
        },
        _ => Arg::TensorBytes {
            data,
            shape,
            dtype: DType::F32,
        },
    }
}

/// A resident, fused execution session for ONE bundle — the production serving
/// path (the per-node [`run_bundle_all`] above is the parity oracle the numeric
/// gate checks against, not the serving path).
///
/// The whole bundle is parsed into one module and run through the optimized
/// driver: whole-program fusion, the K-loop GEMM / map-window GPU offloads, and
/// native head-parallel attention — none of which the per-node interpreter fires
/// (it runs each node in isolation at its native grid, where the single-core
/// offload gate never trips). Weights are uploaded to resident HBM ONCE at
/// construction; each [`run_step`](Self::run_step) reuses them and re-marshals
/// only the per-token dynamic sources (embedding, RoPE tables, prefix-KV, mask).
pub struct SpyreSession {
    session: Session,
    /// Per-program tensor shapes: `shapes[prog][id] = (rows, cols)`. Programs
    /// share weight ids but differ in activation shapes (decode m=1 vs prefill
    /// m=M), so the mask/dynamic marshaling is per-program.
    shapes: Vec<Vec<(usize, usize)>>,
    mask_id: Option<usize>,
}

impl SpyreSession {
    /// Build ONE resident session over multiple programs that SHARE weight
    /// tensor ids (decode + prefill: same graph, different m) — weights uploaded
    /// ONCE. `programs[i] = (nodes, manifest, extra_results)`; `weights` (shared,
    /// same ids across programs) bound once as typed bytes (f16 verbatim, bf16
    /// narrowed by ktir-emulator). Program `i` is addressed by index in `run_step`.
    ///
    /// The resident executor OWNS each parsed module, so the session owns its
    /// whole `Rc` graph and is soundly `Send`.
    #[allow(clippy::type_complexity)]
    pub fn new_multi(
        programs: &[(&[(&str, &str)], &Manifest, &[usize])],
        weights: Vec<(usize, Vec<u8>, SDType, Vec<usize>)>,
    ) -> Result<Self> {
        let mut modules: Vec<ktir_emulator::ir::IRModule> = Vec::with_capacity(programs.len());
        let mut specs: Vec<ProgramSpec> = Vec::with_capacity(programs.len());
        for (nodes, manifest, extra) in programs {
            let texts: Vec<&str> = nodes.iter().map(|(_, t)| *t).collect();
            let module = ktir_emulator::program::module_from_nodes(&texts)
                .map_err(|e| anyhow!("module_from_nodes: {e}"))?;
            modules.push(module);
            specs.push(build_spec(manifest, extra));
        }
        // Weights moved in once (no clone); shared across all programs by id.
        let keys: Vec<String> = weights.iter().map(|(id, ..)| format!("t{id}")).collect();
        let args: Vec<(&str, Arg)> = weights
            .into_iter()
            .enumerate()
            .map(|(i, (_id, data, dt, shape))| (keys[i].as_str(), weight_arg(data, dt, shape)))
            .collect();
        let prog_refs: Vec<(ktir_emulator::ir::IRModule, &ProgramSpec)> =
            modules.into_iter().zip(specs.iter()).collect();
        let session =
            Session::new_multi(prog_refs, &args).map_err(|e| anyhow!("Session::new_multi: {e}"))?;
        let shapes = programs
            .iter()
            .map(|(_, m, _)| m.tensors.iter().map(|t| (t.rows, t.cols)).collect())
            .collect();
        let mask_id = programs[0].1.attn_mask;
        Ok(Self {
            session,
            shapes,
            mask_id,
        })
    }

    /// Run one fused forward of program `prog`: overwrite this pass's dynamic
    /// sources (`dynamic[i] = (id, data)` — embedding / RoPE / prefix-KV), fill
    /// the length mask for `decode_pos`, run, and read back `output_ids`. Weights
    /// stay resident. Returns `id -> data` for each requested id.
    pub fn run_step(
        &mut self,
        prog: usize,
        dynamic: &[(usize, Vec<f32>)],
        decode_pos: u32,
        // Real (unpadded) row count. The KTIR runner fills exactly the bound
        // shapes (it does not pad to a symbolic bucket), so it ignores this; it
        // exists for parity with the sendnn paged runner (which pads s0).
        _real_len: usize,
        output_ids: &[usize],
    ) -> Result<HashMap<usize, Vec<f32>>> {
        let shapes = &self.shapes[prog];
        let mut owned: Vec<(String, Arg)> = Vec::with_capacity(dynamic.len() + 1);
        for (id, data) in dynamic {
            let (r, c) = shapes[*id];
            owned.push((
                format!("t{id}"),
                Arg::Tensor {
                    data: data.clone(),
                    shape: vec![r, c],
                    dtype: DType::F16,
                },
            ));
        }
        if let Some(mid) = self.mask_id {
            let (_, cap) = shapes[mid];
            owned.push((
                format!("t{mid}"),
                Arg::Tensor {
                    data: crate::manifest::attn_mask_fill(cap, decode_pos as usize),
                    shape: vec![1, cap],
                    dtype: DType::F16,
                },
            ));
        }
        let args: Vec<(&str, Arg)> = owned.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();
        self.session
            .set_sources(&args)
            .map_err(|e| anyhow!("set_sources: {e}"))?;
        let keys: Vec<String> = output_ids.iter().map(|id| format!("t{id}")).collect();
        let krefs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
        let out = self
            .session
            .run_program(prog, &krefs)
            .map_err(|e| anyhow!("run_program {prog}: {e}"))?;
        let mut res = HashMap::with_capacity(output_ids.len());
        for &id in output_ids {
            let o = out
                .get(&format!("t{id}"))
                .ok_or_else(|| anyhow!("program {prog} produced no output t{id}"))?;
            res.insert(id, o.data.clone());
        }
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::attn_mask_fill;
    use crate::manifest::{alloc_buffers, argmax};

    /// Semantics lock for the AttnDecode runtime length-mask the emitter emits
    /// (`lower_subtile_tape_to_ktir.rs`): the prefix scores `addf` a shared
    /// `[1, capacity]` mask tile the host fills via [`attn_mask_fill`] — 0 on
    /// valid columns, large-negative past `decode_position`. This drives the
    /// exact emitted op (`ktdp.load` the mask + `arith.addf`) through ktir_emulator
    /// and asserts masked columns collapse to f16 −inf (⇒ exp = 0 in softmax),
    /// independent of the full bundle. A dense `<[...]>` iota was the first
    /// design but is unexecutable from text in ktir_emulator (the parser yields a
    /// `FloatList` value yet never sets the `dense_list`/`is_tensor` attrs the
    /// constant handler needs) — the additive HBM mask sidesteps it.
    #[test]
    fn additive_length_mask_zeroes_masked_columns() {
        // scores[1,4] = [1,1,1,1]; decode_pos=2 ⇒ keep cols 0,1; mask cols 2,3.
        let mlir = r#"
module {
  func.func @mask_test(%scores_ptr: index, %mask_ptr: index, %out_ptr: index) attributes {grid = [1, 1]} {
    %c0 = arith.constant 0 : index
    %sview = ktdp.construct_memory_view %scores_ptr, sizes: [1, 4], strides: [4, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x4xf16>
    %sacc = ktdp.construct_access_tile %sview[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x4xf16> -> !ktdp.access_tile<1x4xindex>
    %sc = ktdp.load %sacc : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>
    %mview = ktdp.construct_memory_view %mask_ptr, sizes: [1, 4], strides: [4, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x4xf16>
    %macc = ktdp.construct_access_tile %mview[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x4xf16> -> !ktdp.access_tile<1x4xindex>
    %m = ktdp.load %macc : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>
    %masked = arith.addf %sc, %m : tensor<1x4xf16>
    %oview = ktdp.construct_memory_view %out_ptr, sizes: [1, 4], strides: [4, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x4xf16>
    %oacc = ktdp.construct_access_tile %oview[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x4xf16> -> !ktdp.access_tile<1x4xindex>
    ktdp.store %masked, %oacc : tensor<1x4xf16>, !ktdp.access_tile<1x4xindex>
    return
  }
}
"#;
        let module = parse_module(mlir).expect("mask snippet parses");
        let mask = attn_mask_fill(4, 2);
        assert_eq!(mask, vec![0.0, 0.0, -1.0e38, -1.0e38]);
        let args = [
            (
                "%scores_ptr",
                Arg::Tensor {
                    data: vec![1.0; 4],
                    shape: vec![1, 4],
                    dtype: DType::F16,
                },
            ),
            (
                "%mask_ptr",
                Arg::Tensor {
                    data: mask,
                    shape: vec![1, 4],
                    dtype: DType::F16,
                },
            ),
            (
                "%out_ptr",
                Arg::Tensor {
                    data: vec![0.0; 4],
                    shape: vec![1, 4],
                    dtype: DType::F16,
                },
            ),
        ];
        let out = execute_function(&module, "mask_test", &args).expect("mask snippet executes");
        let r = &out.get("%out_ptr").expect("out tensor").data;
        assert_eq!(r.len(), 4);
        // cols 0,1 unchanged (+0); cols 2,3 driven to f16 -inf.
        assert!((r[0] - 1.0).abs() < 1e-2, "col0 kept: {r:?}");
        assert!((r[1] - 1.0).abs() < 1e-2, "col1 kept: {r:?}");
        assert!(r[2] < -1e4, "col2 masked: {r:?}");
        assert!(r[3] < -1e4, "col3 masked: {r:?}");
    }

    fn bundle_dir() -> std::path::PathBuf {
        // Reads a bundle dumped via `SCRATCHY_KTIR_DUMP=<dir>` at emit time;
        // the tests skip when it's unset (the joined dir won't exist).
        std::path::PathBuf::from(std::env::var("SCRATCHY_KTIR_DUMP").unwrap_or_default())
            .join("smollm2-135m")
    }

    fn read_f32_bin(p: &std::path::Path) -> Vec<f32> {
        std::fs::read(p)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Every emitted node parses through the real `ktir_emulator` parser — the f16
    /// KTIR is syntactically valid. Green independent of execution.
    #[test]
    fn parses_all_emitted_nodes() {
        let dir = bundle_dir();
        if !dir.join("manifest.json").exists() {
            eprintln!("skip: no bundle at {dir:?}");
            return;
        }
        let manifest = Manifest::load(&dir).expect("load manifest");
        for node in &manifest.nodes {
            let text = std::fs::read_to_string(dir.join(&node.mlir)).unwrap();
            parse_module(&text).unwrap_or_else(|e| panic!("parse {} failed: {e}", node.mlir));
        }
        eprintln!("parsed {} nodes OK", manifest.nodes.len());
    }

    /// Mechanical end-to-end: all 452 nodes execute through `ktir_emulator` and
    /// thread to a `logits[49152]` vector. Synthetic sources (proves wiring).
    #[test]
    fn runs_emitted_smollm_bundle_end_to_end() {
        let dir = bundle_dir();
        if !dir.join("manifest.json").exists() {
            eprintln!("skip: no bundle — build -p scratchy-models --features spyre,arch-llama");
            return;
        }
        let manifest = Manifest::load(&dir).expect("load manifest");
        let mut bufs = alloc_buffers(&manifest);
        for (id, buf) in bufs
            .iter_mut()
            .enumerate()
            .take(manifest.num_sources as usize)
        {
            let n = manifest.tensor_len(id);
            *buf = (0..n)
                .map(|j| (((id * 131 + j * 7) % 197) as f32 / 197.0 - 0.5) * 0.1)
                .collect();
        }
        let logits = run_bundle(&dir, &manifest, bufs).expect("bundle executes");
        assert_eq!(
            logits.len(),
            manifest.tensor_len(manifest.result),
            "logits width"
        );
        assert!(logits.iter().all(|x| x.is_finite()), "logits all finite");
        eprintln!(
            "ran {} nodes → logits[{}]",
            manifest.nodes.len(),
            logits.len()
        );
    }

    /// Length-independence: ONE bundle, different `decode_pos` ⇒ different
    /// attention (more/fewer prefix positions in the softmax) ⇒ different
    /// logits. Proves the runtime mask knob is live end-to-end on the REAL
    /// bundle (correctness per position is the worker's MLX gate). `#[ignore]`
    /// — runs the full 452-node bundle several times. Needs capacity ≥ 3
    /// (`KTIR_PREFIX_LEN=4` at emit) so dp ∈ {0,1,2} are distinct.
    #[test]
    #[ignore = "slow: runs the full bundle 3x; needs KTIR_PREFIX_LEN>=4 emit"]
    fn decode_pos_override_changes_logits() {
        let dir = bundle_dir();
        if !dir.join("manifest.json").exists() {
            eprintln!("skip: no bundle");
            return;
        }
        let manifest = Manifest::load(&dir).expect("load manifest");
        let cap = manifest
            .attn_mask
            .map(|id| manifest.tensors[id].cols)
            .unwrap_or(0);
        assert!(cap >= 3, "re-emit with KTIR_PREFIX_LEN>=4 (capacity {cap})");
        let run = |dp: u32| {
            let mut bufs = alloc_buffers(&manifest);
            for (id, buf) in bufs
                .iter_mut()
                .enumerate()
                .take(manifest.num_sources as usize)
            {
                *buf = read_f32_bin(&dir.join(format!("t{id}.bin")));
            }
            run_bundle_all(&dir, &manifest, bufs, Some(dp)).unwrap()[manifest.result].clone()
        };
        let (a, b, c) = (run(0), run(1), run(2));
        assert!(
            a.iter().all(|x| x.is_finite()) && c.iter().all(|x| x.is_finite()),
            "finite"
        );
        let diff = |x: &[f32], y: &[f32]| x.iter().zip(y).any(|(p, q)| (p - q).abs() > 1e-3);
        assert!(diff(&a, &b), "decode_pos 0 vs 1 must differ");
        assert!(diff(&b, &c), "decode_pos 1 vs 2 must differ");
        eprintln!(
            "length-independence: argmax dp0={} dp1={} dp2={}",
            argmax(&a),
            argmax(&b),
            argmax(&c)
        );
    }

    /// ★ THE GATE: the emitted bundle, run on `ktir_emulator` with the emit-time
    /// synthetic sources, reproduces the in-tree f32 reference (`eval_dag`,
    /// dropped as `golden.bin`). This validates the ACTUAL macro-emitted bundle
    /// is numerically correct — pure Rust, no weights, no MLX. f16 storage vs
    /// f32 reference → expect close-but-not-exact; the backend is correct iff
    /// the two logit vectors are highly correlated AND the argmax agrees.
    #[test]
    fn bundle_matches_eval_dag_golden() {
        let dir = bundle_dir();
        if !dir.join("golden.bin").exists() {
            eprintln!("skip: no golden.bin (rebuild after the eval_dag golden landed)");
            return;
        }
        let manifest = Manifest::load(&dir).expect("load manifest");
        let mut bufs = alloc_buffers(&manifest);
        for (id, buf) in bufs
            .iter_mut()
            .enumerate()
            .take(manifest.num_sources as usize)
        {
            *buf = read_f32_bin(&dir.join(format!("t{id}.bin")));
            assert_eq!(buf.len(), manifest.tensor_len(id), "source {id} size");
        }
        let out = run_bundle(&dir, &manifest, bufs).expect("bundle executes");
        let golden = read_f32_bin(&dir.join("golden.bin"));
        assert_eq!(out.len(), golden.len(), "logits width vs golden");

        let dot: f64 = out
            .iter()
            .zip(&golden)
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let na: f64 = out.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = golden
            .iter()
            .map(|b| (*b as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let cosine = dot / (na * nb + 1e-12);
        let max_abs_err = out
            .iter()
            .zip(&golden)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let scale = golden.iter().map(|x| x.abs()).fold(0f32, f32::max).max(1.0);
        eprintln!(
            "GATE ktir_emulator vs eval_dag: cosine={cosine:.6}, max_abs_err={max_abs_err:.4} (scale {scale:.2}), argmax {} vs {}",
            argmax(&out),
            argmax(&golden)
        );
        // Backend-correct ⇒ near-collinear with the f32 reference despite f16
        // storage; a wrong computation gives cosine ≈ 0 / NaN.
        assert!(
            cosine > 0.99,
            "ktir_emulator output not collinear with eval_dag (cosine {cosine:.4})"
        );
    }
}

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

use anyhow::{Result, anyhow};
use ktir_core::ir::Ssa;
use ktir_emulator::dtypes::DType;
use ktir_emulator::interpreter::Arg;
use ktir_emulator::ktir_optimizer::fusion::{Binding, NodeSpec, ProgramSpec};
use ktir_emulator::program::Session;
use scratchy_spyre_bundle as bundle;
use scratchy_tensors::DType as SDType;
use std::collections::HashMap;

/// The program spec for one launch group: its functions in launch order, and which tensor each of
/// their parameters points at.
///
/// ⭐ THE BINDING IS THE ONE THE LOWERING RECORDED. `LaunchProgram::args` pairs each parameter's
/// `Ssa` with the `PlaceId` it carries, minted by the construction that made the parameter — so
/// this is a carry, not a join on a spelling. `PlaceId::Act` names a SubtileIR tensor.
///
/// ⛔ AND `is_output` IS READ OFF THE PROGRAM. An argument is written exactly when a `ktdp.store`
/// writes a tile of its view: `store`'s tile operand names a `construct_access_tile`, whose first
/// operand names a `construct_memory_view`, whose first operand IS the argument. A flag carried
/// beside the program could disagree with what the program does; this cannot.
fn build_spec(programs: &[bundle::LaunchProgram<'static>], results: &[u64]) -> ProgramSpec {
    let mut nodes = Vec::with_capacity(programs.len());
    let mut sources: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut written: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for p in programs {
        let writes = written_args(&p.func);
        let mut bindings = Vec::with_capacity(p.args.len());
        for (ssa, id) in p.args.iter() {
            let t = u64::from(id.tid());
            let is_output = writes.contains(ssa);
            // A tensor first seen as a READ comes from outside this group; one this group writes is
            // produced by it.
            if is_output {
                written.insert(t);
            } else if !written.contains(&t) {
                sources.insert(t);
            }
            bindings.push(Binding {
                arg: *ssa,
                tensor: t,
                is_output,
            });
        }
        nodes.push(NodeSpec {
            func: p.func.name.to_string(),
            bindings,
        });
    }
    ProgramSpec {
        nodes,
        sources,
        results: results.iter().copied().collect(),
    }
}

/// Which of a function's arguments it WRITES — read off the program, never declared beside it.
fn written_args(f: &ktir_core::ir::IRFunction<'static>) -> std::collections::HashSet<Ssa> {
    use ktir_core::opkind::OpKind;
    let producer = |v: Ssa| f.operations.iter().find(|o| o.result == Some(v));
    let mut out = std::collections::HashSet::new();
    for op in f.operations {
        if op.op_type != OpKind::KtdpStore {
            continue;
        }
        let Some(&tile) = op.operands.get(1) else {
            continue;
        };
        let Some(t) = producer(tile) else { continue };
        let Some(&view) = t.operands.first() else {
            continue;
        };
        let Some(v) = producer(view) else { continue };
        if let Some(&ptr) = v.operands.first() {
            out.insert(ptr);
        }
    }
    out
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
}

impl SpyreSession {
    /// ⭐⭐⭐ ONE RESIDENT SESSION OVER THE BUNDLE'S LAUNCH GROUPS.
    ///
    /// `programs[i] = (that program's launch groups, the tensor ids to read back)`. Decode and
    /// prefill share weight tensor ids — same graph, different `m` — so the weights are marshalled
    /// into resident HBM ONCE here and every pass chains its kernels against them with no
    /// re-marshal. Program `i` is addressed by index in [`SpyreSession::run_step`].
    ///
    /// ⛔ NOTHING IS PARSED. A launch group carries `IRFunction`s the lowering constructed and
    /// `#[forward]` baked; this hands those values straight to the executor. The `&str` MLIR this
    /// took before, and the module-from-text step behind it, have no counterpart — there is no text
    /// form of a program at any point.
    ///
    /// ⭐ AND IT IS THE FAST ENTRYPOINT. `program::Session` is the optimized path: whole-program
    /// fusion, resident weights, and the GPU offloads. The per-node interpreter — which
    /// `program.rs:12-14` calls "the slow parity-oracle path", where "the GPU offloads, gated on a
    /// single-core grid, never fire" — is not reachable from outside the emulator crate.
    #[allow(clippy::type_complexity)]
    pub fn new_multi(
        programs: &[(&[bundle::LaunchGroup<'static>], &[u64])],
        weights: Vec<(usize, Vec<u8>, SDType, Vec<usize>)>,
    ) -> Result<Self> {
        let mut funcs: Vec<Vec<ktir_core::ir::IRFunction<'static>>> =
            Vec::with_capacity(programs.len());
        let mut specs: Vec<ProgramSpec> = Vec::with_capacity(programs.len());
        for (groups, results) in programs {
            // A group IS a launch, and its programs run in order; the groups run in order too, so
            // the whole program is that sequence flattened.
            let progs: Vec<bundle::LaunchProgram<'static>> = groups
                .iter()
                .flat_map(|g| g.programs.iter().cloned())
                .collect();
            funcs.push(progs.iter().map(|p| p.func).collect());
            specs.push(build_spec(&progs, results));
        }
        // Weights moved in once (no clone); shared across all programs by tensor id, and
        // moved AGAIN into the session so they reach HBM without a copy.
        let args: Vec<(u64, Arg)> = weights
            .into_iter()
            .map(|(id, data, dt, shape)| (id as u64, weight_arg(data, dt, shape)))
            .collect();
        let prog_refs: Vec<(&[ktir_core::ir::IRFunction<'static>], &ProgramSpec)> = funcs
            .iter()
            .map(|f| f.as_slice())
            .zip(specs.iter())
            .collect();
        let session =
            Session::new_multi(prog_refs, args).map_err(|e| anyhow!("Session::new_multi: {e}"))?;
        Ok(Self { session })
    }

    /// Run one fused forward of program `prog`: overwrite this pass's changing sources, run, and
    /// read back `outputs`. Weights stay resident.
    ///
    /// ⛔ EVERY TENSOR IS NAMED BY IDENTITY. This keyed its arguments by `format!("t{id}")` and
    /// looked the results back up the same way — a join on a spelling both ends had to agree about.
    /// The executor's own key is the tensor id, so that is what crosses: there is no name.
    ///
    /// ⭐ A SOURCE CARRIES ITS SHAPE. The per-program `[(rows, cols)]` table this read came from the
    /// manifest; a caller that is handing over the bytes already knows their shape, and passing it
    /// is what removes the second copy of it. The attention length mask is a source like any other
    /// — it is filled by whoever knows the decode position, not here.
    /// ⛔ THE SOURCES ARE MOVED, NOT BORROWED-AND-CLONED. `Arg::Tensor` owns its buffer, so taking
    /// `&[..]` here meant a `data.clone()` per source: granite's decode binds its whole prefix KV
    /// every step — 40 layers x (K, V) x `[cap, kv_dim]` — so that was ~42 MB memcpy'd per forward
    /// to hand over buffers the caller builds fresh and drops immediately.
    pub fn run_step(
        &mut self,
        prog: usize,
        sources: Vec<(u64, Vec<f32>, Vec<usize>)>,
        // `(tensor id, elements wanted)` — `0` reads the whole tensor. See
        // `ResidentExec::run_program`: the session's tensors are resident at the WIDEST program's
        // row count, so a decode step must say how few rows it actually wrote.
        outputs: &[(u64, usize)],
    ) -> Result<HashMap<u64, Vec<f32>>> {
        let args: Vec<(u64, Arg)> = sources
            .into_iter()
            .map(|(id, data, shape)| {
                (
                    id,
                    Arg::Tensor {
                        data,
                        shape,
                        dtype: DType::F16,
                    },
                )
            })
            .collect();
        // A forward is THREE phases and only one of them is the program; `KTIR_SEG_PROF` reports
        // the segments, which left the rest unattributed.
        let prof = std::env::var_os("KTIR_SEG_PROF").is_some();
        let t0 = std::time::Instant::now();
        self.session
            .set_sources(&args)
            .map_err(|e| anyhow!("set_sources: {e}"))?;
        let t_bind = t0.elapsed();
        let t1 = std::time::Instant::now();
        let out = self
            .session
            .run_program(prog, outputs)
            .map_err(|e| anyhow!("run_program {prog}: {e}"))?;
        let t_run = t1.elapsed();
        let t2 = std::time::Instant::now();
        let mut res = HashMap::with_capacity(outputs.len());
        for &(id, _) in outputs {
            let o = out
                .get(&id)
                .ok_or_else(|| anyhow!("program {prog} produced no output for tensor {id}"))?;
            res.insert(id, o.data.clone());
        }
        if prof {
            eprintln!(
                "  [run_step] bind {:.1}ms ({} sources) | run {:.1}ms | readback {:.1}ms ({} outputs, elem sizes {})",
                t_bind.as_secs_f64() * 1e3,
                args.len(),
                t_run.as_secs_f64() * 1e3,
                t2.elapsed().as_secs_f64() * 1e3,
                outputs.len(),
                {
                    let mut sh: Vec<String> =
                        out.values().map(|o| format!("{:?}", o.shape)).collect();
                    sh.sort();
                    sh.dedup();
                    format!("{:?}", sh)
                },
            );
        }
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::attn_mask_fill;
    use crate::manifest::{alloc_buffers, argmax};

    /// Semantics lock for the attention runtime length-mask the emitter emits
    /// (`KtirFunc::attn`): the prefix scores `addf` a shared
    /// `[1, capacity]` mask tile the host fills via [`attn_mask_fill`] — 0 on
    /// valid columns, large-negative past `decode_position`. This drives the
    /// exact emitted op (`ktdp.load` the mask + `arith.addf`) through ktir_emulator
    /// and asserts masked columns collapse to f16 −inf (⇒ exp = 0 in softmax),
    /// independent of the full bundle. A dense `<[...]>` iota was the first
    /// design but is unexecutable in ktir_emulator (a `FloatList` value never sets the
    /// `dense_list`/`is_tensor` attrs the constant handler needs) — the additive HBM mask
    /// sidesteps it.
    /// The three tensors this snippet binds, by identity. A launch binds an ADDRESS per parameter
    /// and the executor threads one buffer per tensor id, so these ARE the argument names.
    const SCORES: u64 = 0;
    const MASK: u64 = 1;
    const OUT: u64 = 2;

    /// A `[1, 4]` HBM view over parameter `ptr`, its whole-tile access window, and the loaded tile —
    /// the same three ops `KtirFunc::view_of` / `tile` / `load_tile` emit, built here directly so
    /// the test drives CONSTRUCTED IR rather than a second, parsed copy of it.
    fn view_load(
        a: &'static ktir_core::arena::Arena,
        ops: &mut Vec<ktir_core::ir::Operation<'static>>,
        ptr: Ssa,
        c0: Ssa,
        next: &mut u32,
    ) -> Ssa {
        use ktir_core::attrkey::AttrKey;
        use ktir_core::ir::{Attr, Operation};
        use ktir_core::irtype::IrType;
        use ktir_core::opkind::OpKind;
        let dims = vec![1i64, 4];
        let mut fresh = || {
            let v = Ssa(*next);
            *next += 1;
            v
        };
        let (view, acc, val) = (fresh(), fresh(), fresh());
        let mut vop = Operation::new(a, Some(view), OpKind::KtdpConstructMemoryView, &[ptr])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())))
            .with_attr(a, AttrKey::Strides, Attr::IntList(a.ints(vec![4, 1])))
            .with_attr(a, AttrKey::MemorySpace, Attr::Str("HBM"))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16));
        vop.result_type = Some(IrType::MemRef {
            dims: a.ints(dims.clone()),
            elem: DType::F16,
        });
        ops.push(vop);
        let mut aop = Operation::new(
            a,
            Some(acc),
            OpKind::KtdpConstructAccessTile,
            &[view, c0, c0],
        )
        .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())));
        aop.result_type = Some(IrType::AccessTile {
            dims: a.ints(dims.clone()),
        });
        ops.push(aop);
        let mut lop = Operation::new(a, Some(val), OpKind::KtdpLoad, &[acc]);
        lop.result_type = Some(IrType::Tensor {
            dims: a.ints(dims),
            elem: DType::F16,
        });
        ops.push(lop);
        val
    }

    /// Semantics lock for the attention runtime length-mask the emitter emits
    /// (`KtirFunc::attn`): the prefix scores `addf` a shared `[1, capacity]` mask tile the host
    /// fills via [`attn_mask_fill`] — 0 on valid columns, large-negative past the decode position.
    /// This drives the exact emitted ops (`ktdp.load` the mask + `arith.addf`) through the REAL
    /// launch path — `SpyreSession::new_multi` / `run_step`, keyed by tensor id — and asserts
    /// masked columns collapse to f16 −inf (⇒ `exp` = 0 in softmax), independent of a full bundle.
    #[test]
    fn additive_length_mask_zeroes_masked_columns() {
        use ktir_core::attrkey::AttrKey;
        use ktir_core::ir::{Attr, Operation};
        use ktir_core::irtype::IrType;
        use ktir_core::opkind::OpKind;
        let a = ktir_core::arena::Arena::global();
        let mut ops: Vec<Operation<'static>> = Vec::new();
        let mut next = 3u32; // %0..%2 are the parameters
        let (scores_p, mask_p, out_p) = (Ssa(0), Ssa(1), Ssa(2));

        let c0 = Ssa(next);
        next += 1;
        let mut zero = Operation::new(a, Some(c0), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Int(0),
        );
        zero.result_type = Some(IrType::Index);
        ops.push(zero);

        let sc = view_load(a, &mut ops, scores_p, c0, &mut next);
        let m = view_load(a, &mut ops, mask_p, c0, &mut next);
        let dims = vec![1i64, 4];
        let masked = Ssa(next);
        next += 1;
        let mut add = Operation::new(a, Some(masked), OpKind::ArithAddf, &[sc, m]);
        add.result_type = Some(IrType::Tensor {
            dims: a.ints(dims.clone()),
            elem: DType::F16,
        });
        ops.push(add);

        // The store's own view + window over the output parameter.
        let (oview, oacc) = (Ssa(next), Ssa(next + 1));
        next += 2;
        let mut vop = Operation::new(a, Some(oview), OpKind::KtdpConstructMemoryView, &[out_p])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())))
            .with_attr(a, AttrKey::Strides, Attr::IntList(a.ints(vec![4, 1])))
            .with_attr(a, AttrKey::MemorySpace, Attr::Str("HBM"))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16));
        vop.result_type = Some(IrType::MemRef {
            dims: a.ints(dims.clone()),
            elem: DType::F16,
        });
        ops.push(vop);
        let mut aop = Operation::new(
            a,
            Some(oacc),
            OpKind::KtdpConstructAccessTile,
            &[oview, c0, c0],
        )
        .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())));
        aop.result_type = Some(IrType::AccessTile { dims: a.ints(dims) });
        ops.push(aop);
        ops.push(Operation::new(a, None, OpKind::KtdpStore, &[masked, oacc]));
        ops.push(Operation::new(a, None, OpKind::FuncReturn, &[]));

        let func = ktir_core::ir::IRFunction {
            name: "mask_test",
            arguments: a.args(vec![
                (scores_p, IrType::Index),
                (mask_p, IrType::Index),
                (out_p, IrType::Index),
            ]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        };
        let group = bundle::LaunchGroup {
            kv: Default::default(),
            programs: std::borrow::Cow::Owned(vec![bundle::LaunchProgram {
                func,
                args: std::borrow::Cow::Owned(vec![
                    (scores_p, bundle::PlaceId::Act(SCORES as u32)),
                    (mask_p, bundle::PlaceId::Act(MASK as u32)),
                    (out_p, bundle::PlaceId::Act(OUT as u32)),
                ]),
            }]),
            init_binary: std::borrow::Cow::Borrowed(&[]),
            job_bin_ptr: 0,
            correction: std::borrow::Cow::Borrowed(&[]),
        };

        let mask = attn_mask_fill(4, 2);
        assert_eq!(mask, vec![0.0, 0.0, -1.0e38, -1.0e38]);
        let mut session = SpyreSession::new_multi(&[(&[group], &[OUT])], Vec::new())
            .expect("build the one-program session");
        let out = session
            .run_step(
                0,
                vec![
                    (SCORES, vec![1.0f32; 4], vec![1, 4]),
                    (MASK, mask, vec![1, 4]),
                ],
                // Whole tensor: this fixture's output is one row already.
                &[(OUT, 0)],
            )
            .expect("run the mask snippet");
        let r = &out[&OUT];
        assert_eq!(r.len(), 4);
        // cols 0,1 unchanged (+0); cols 2,3 driven to f16 -inf.
        assert!((r[0] - 1.0).abs() < 1e-2, "col0 kept: {r:?}");
        assert!((r[1] - 1.0).abs() < 1e-2, "col1 kept: {r:?}");
        assert!(r[2] < -1e4, "col2 masked: {r:?}");
        assert!(r[3] < -1e4, "col3 masked: {r:?}");
    }
}

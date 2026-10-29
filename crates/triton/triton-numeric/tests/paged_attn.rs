// SPDX-License-Identifier: Apache-2.0

//! PAGED ATTENTION, EXECUTED: spyre-inference's serving step — the KV cache write
//! (`reshape_and_cache_head_major`) and the head-major page walk (`page_attn_head_major_decode` /
//! `_prefill`) — as Triton (`kernels/paged_kv_write.py`, `kernels/paged_attn.py`), through the
//! whole ladder, run by `ktir_emulator`.
//!
//! ```text
//!   triton_frontend      .py -> TTIR       (desc.scatter, the page-id read, the slab load)
//!   triton_ktir          TTIR -> KTIR      (indirect access tiles: a store and the page loads)
//!   to_ktir_emit         KTIR -> IRFunction
//!   ktir_emulator        the executed step
//! ```

use std::collections::HashMap;

use ktir_core::arena::Arena;
use ktir_core::ir::IRFunction;
use ktir_core::opkind::OpKind;
use triton_frontend::codegen::{self, ArgSpec, KernelSpec};
use triton_frontend::opt;
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

const ATTN: &str = include_str!("../../../targets/spyre/kernels/paged_attn.py");
const KV_WRITE: &str = include_str!("../../../targets/spyre/kernels/paged_kv_write.py");

/// A kernel through the whole ladder to the emulator's IR.
fn lower(src: &str, spec: &KernelSpec) -> IRFunction<'static> {
    let mut tt = codegen::compile(src, spec, Target::spyre())
        .unwrap_or_else(|e| panic!("{}: codegen refused the kernel: {e}", spec.file));
    opt::make_ttir(&mut tt).unwrap_or_else(|e| panic!("make_ttir refused the kernel: {e}"));
    let mut m = triton_ktir::from_ttir::convert(&tt)
        .unwrap_or_else(|e| panic!("from_ttir refused the kernel: {e}"));
    triton_ktir::make_ktir(&mut m, &[1])
        .unwrap_or_else(|e| panic!("make_ktir refused the kernel: {e}"));
    triton_ktir::passes::to_ktir::run(&mut m, &[1])
        .unwrap_or_else(|e| panic!("to_ktir refused the kernel: {e}"));
    triton_ktir::passes::to_ktir_emit::lower(&m, Arena::global())
        .unwrap_or_else(|e| panic!("to_ktir_emit refused the kernel: {e}"))
}

fn spec(file: &str, kernel: &str, pointers: &[(&str, &str)], ce: &[(&str, i128)]) -> KernelSpec {
    let mut signature = HashMap::new();
    let mut constexprs = HashMap::new();
    for (k, v) in pointers {
        signature.insert(k.to_string(), ArgSpec::parse(v).expect("pointer spec"));
    }
    for (k, v) in ce {
        signature.insert(k.to_string(), ArgSpec::Constexpr);
        constexprs.insert(k.to_string(), Val::Int(*v));
    }
    KernelSpec {
        kernel: kernel.to_string(),
        signature,
        constexprs,
        file: file.to_string(),
    }
}

fn kv_write_spec(t: i128, nkvh: i128, hd: i128, pool_rows: i128) -> KernelSpec {
    spec(
        "paged_kv_write.py",
        "paged_kv_write_fwd",
        &[
            ("desc_k", "*fp16"),
            ("desc_v", "*fp16"),
            ("desc_kp", "*fp16"),
            ("desc_vp", "*fp16"),
            ("desc_rows", "*i32"),
        ],
        &[
            ("T", t),
            ("NKVH", nkvh),
            ("HD", hd),
            ("POOL_ROWS", pool_rows),
        ],
    )
}

struct AttnShape {
    mq: i128,
    nqh: i128,
    nkvh: i128,
    hd: i128,
    block: i128,
    num_pages: i128,
    num_blocks: i128,
}

fn attn_spec(s: &AttnShape) -> KernelSpec {
    let mut k = spec(
        "paged_attn.py",
        "paged_attn_fwd",
        &[
            ("desc_q", "*fp16"),
            ("desc_o", "*fp16"),
            ("desc_kp", "*fp16"),
            ("desc_vp", "*fp16"),
            ("desc_pt", "*i32"),
            ("desc_mask", "*fp16"),
        ],
        &[
            ("MQ", s.mq),
            ("NQH", s.nqh),
            ("NKVH", s.nkvh),
            ("HD", s.hd),
            ("BLOCK", s.block),
            ("NUM_PAGES", s.num_pages),
            ("NUM_BLOCKS", s.num_blocks),
        ],
    );
    k.signature.insert("SCALE".to_string(), ArgSpec::Constexpr);
    k.constexprs
        .insert("SCALE".to_string(), Val::Float(1.0 / (s.hd as f64).sqrt()));
    k
}

fn count(f: &IRFunction, kind: OpKind) -> usize {
    f.ops_deep()
        .into_iter()
        .filter(|op| op.op_type == kind)
        .count()
}

/// The handoff census: every page read and every cache write reaches the emulator as an
/// INDIRECT access tile — a gather or a scatter the device performs, not a direct tile at a
/// computed corner.
#[test]
fn the_step_reaches_the_emulator_as_hardware_gathers_and_scatters() {
    let w = lower(KV_WRITE, &kv_write_spec(1, 8, 64, 8 * 8 * 64));
    assert_eq!(
        count(&w, OpKind::KtdpConstructIndirectAccessTile),
        2,
        "one scatter tile per pool"
    );
    // Under Triton's scatter floor of 8 rows (one token over 2 kv heads), each row is its own
    // store at the row the table names — still an indirect tile, one per row per pool.
    let w2 = lower(KV_WRITE, &kv_write_spec(1, 2, 64, 4 * 2 * 64));
    assert_eq!(
        count(&w2, OpKind::KtdpConstructIndirectAccessTile),
        2 * 2,
        "one indirect store tile per (row, pool)"
    );
    let decode = AttnShape {
        mq: 1,
        nqh: 32,
        nkvh: 8,
        hd: 64,
        block: 64,
        num_pages: 8,
        num_blocks: 3,
    };
    let a = lower(ATTN, &attn_spec(&decode));
    // ⭐ THE WALK IS A LOOP — `for_each_tile`'s ONE block body, not a copy per block: each kv head
    // walks its blocks in an `scf.for` whose body gathers one K page and one V page through the
    // trip's own page-table row.
    assert_eq!(
        count(&a, OpKind::ScfFor),
        8,
        "one page walk per kv head, standing as a loop"
    );
    assert_eq!(
        count(&a, OpKind::KtdpConstructIndirectAccessTile),
        2 * 8,
        "one K and one V page gather in each walk's one block body"
    );
    // And the program is the same size however long the walk: a 64-block walk is the 3-block
    // walk's ops with a different trip count.
    let long = lower(
        ATTN,
        &attn_spec(&AttnShape {
            num_pages: 64,
            num_blocks: 64,
            ..decode
        }),
    );
    assert_eq!(
        long.ops_deep().len(),
        a.ops_deep().len(),
        "a longer walk grew the program"
    );
    // The prompt walk gathers each page ONCE per kv head too: its four query heads ride one
    // walk as rows (spyre-inference's batched GQA), not one walk each.
    let prompt = AttnShape { mq: 32, ..decode };
    let p = lower(ATTN, &attn_spec(&prompt));
    assert_eq!(
        count(&p, OpKind::KtdpConstructIndirectAccessTile),
        2 * 8,
        "one K and one V page gather per kv head's block body, whatever the query group"
    );
    // ⛔ NO WIDEN SURVIVES. The kernel's `exp` islands (`x.to(f32)` → exp → `.to(f16)`) collapse
    // to the bare f16 op, so neither program carries an f16→f32 `arith.extf` — the convert whose
    // card form (`dl16tofp32`) permutes lanes, which a fused forward models.
    for (name, f) in [
        ("paged_kv_write", &w),
        ("paged_kv_write (2 rows)", &w2),
        ("paged_attn", &a),
        ("paged_attn (prompt)", &p),
    ] {
        assert_eq!(
            count(f, OpKind::ArithExtf),
            0,
            "{name}: an f16→f32 widen reached the emulator"
        );
    }
}

// ── THE EXECUTED STEP, against spyre-inference's own kernels ─────────────────────────────────

use ktir_core::dtypes::DType;
use ktir_emulator::interpreter::Arg;
use ktir_emulator::latency::HardwareConfig;
use triton_numeric::data;

/// One generated step (`test-numeric/gen_paged_attn_data.py`): its `meta.json`, and its files
/// read against their recorded shape, dtype and sha256.
struct Step {
    dir: std::path::PathBuf,
    meta: serde_json::Value,
    shas: HashMap<String, String>,
}

/// How far past the gate a negative control must land, in gates. Ten: the smallest defect a
/// control plants here — decode missing its one new token out of 151 — measured 40 gates out.
const CONTROL_MARGIN: f64 = 10.0;

impl Step {
    fn load(config: &str) -> Step {
        let dir = data::dir(config);
        let meta: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("meta.json"))
                .unwrap_or_else(|e| panic!("{config}: no meta.json: {e}")),
        )
        .expect("meta.json parses");
        let shas = std::fs::read_to_string(dir.join("sha256.txt"))
            .expect("sha256.txt")
            .lines()
            .filter_map(|l| l.split_once("  "))
            .map(|(h, f)| (f.to_string(), h.to_string()))
            .collect();
        Step { dir, meta, shas }
    }

    fn geom(&self, key: &str) -> i128 {
        self.meta["geometry"][key].as_i64().expect("geometry key") as i128
    }

    fn mq(&self) -> i128 {
        self.meta["mq"].as_i64().expect("mq") as i128
    }

    fn num_blocks(&self) -> i128 {
        self.meta["derived"]["num_blocks"]
            .as_i64()
            .expect("num_blocks") as i128
    }

    /// `(bytes, shape, dtype)` of a recorded buffer, its length and hash checked.
    fn file(&self, key: &str) -> (Vec<u8>, Vec<usize>, DType) {
        let e = &self.meta["files"][key];
        let shape: Vec<usize> = e["shape"]
            .as_array()
            .expect("shape")
            .iter()
            .map(|v| v.as_u64().expect("extent") as usize)
            .collect();
        let dtype = match e["dtype"].as_str().expect("dtype") {
            "f16" => DType::F16,
            "f32" => DType::F32,
            "i32" => DType::I32,
            other => panic!("{key}: dtype {other}"),
        };
        let name = e["file"].as_str().expect("file");
        let n: usize = shape.iter().product();
        let raw = data::read_bin(&self.dir.join(name), n * dtype.bytes_per_elem())
            .unwrap_or_else(|e| panic!("{key}: {e}"));
        assert_eq!(
            Some(&data::sha256_hex(&raw)),
            self.shas.get(name),
            "{key}: the bytes are not the ones the step was generated against"
        );
        (raw, shape, dtype)
    }

    fn values(&self, key: &str) -> Vec<f64> {
        let (raw, shape, dtype) = self.file(key);
        data::decode_f64(&raw, shape.iter().product(), dtype).expect("decodes")
    }

    fn arg(&self, key: &str, shape: Vec<usize>) -> Arg {
        let (data, recorded, dtype) = self.file(key);
        assert_eq!(
            shape.iter().product::<usize>(),
            recorded.iter().product::<usize>(),
            "{key}: bound at {shape:?}, recorded {recorded:?}"
        );
        Arg::TensorBytes { data, shape, dtype }
    }
}

fn f16_arg(values: &[f64], shape: Vec<usize>) -> Arg {
    Arg::TensorBytes {
        data: ktir_core::codec::encode(
            &values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            DType::F16,
        ),
        shape,
        dtype: DType::F16,
    }
}

/// Run `f` on `bound` (positional, declaration order); every argument's buffer, read back.
fn run(f: &IRFunction<'static>, bound: Vec<Arg>) -> Vec<Vec<f64>> {
    let params: Vec<ktir_core::ir::Ssa> = f.arguments.iter().map(|(s, _)| *s).collect();
    assert_eq!(
        params.len(),
        bound.len(),
        "one binding per kernel parameter"
    );
    let args: Vec<_> = params.iter().copied().zip(bound).collect();
    let (raw, _) = ktir_emulator::interpreter::execute_function_with_latency(
        f,
        &args,
        HardwareConfig::default(),
    )
    .unwrap_or_else(|e| panic!("the emulator refused the kernel: {e}"));
    params
        .iter()
        .map(|p| {
            raw.get(p)
                .map(|o| o.data.iter().map(|&v| v as f64).collect())
                .unwrap_or_default()
        })
        .collect()
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// The step's attention, on the emulator, over pools `kp`/`vp` with page table `pt`.
fn attend(s: &Step, kp: &[f64], vp: &[f64], pt: Arg) -> Vec<f64> {
    let (mq, nqh, nkvh, hd) = (s.mq(), s.geom("NQH"), s.geom("NKVH"), s.geom("HD"));
    let (block, pages, nb) = (s.geom("BLOCK"), s.geom("NUM_PAGES"), s.num_blocks());
    let shape = AttnShape {
        mq,
        nqh,
        nkvh,
        hd,
        block,
        num_pages: pages,
        num_blocks: nb,
    };
    let f = lower(ATTN, &attn_spec(&shape));
    let q_shape = vec![mq as usize, (nqh * hd) as usize];
    let pool = vec![pages as usize, nkvh as usize, block as usize, hd as usize];
    // spyre-inference's `[NUM_BLOCKS, MQ, BLOCK]` mask stack, staged at the score rows the way the
    // worker stages it (`paged_step::mask_stack`): row `m * GQA + g` of each tile is their row `m`.
    let gqa = (nqh / nkvh) as usize;
    let theirs = s.values("mask");
    let (b, rows) = (block as usize, mq as usize);
    let mask: Vec<f64> = (0..nb as usize * rows * gqa)
        .flat_map(|r| {
            let (tile, m) = (r / (rows * gqa), r % (rows * gqa) / gqa);
            theirs[(tile * rows + m) * b..(tile * rows + m + 1) * b].to_vec()
        })
        .collect();
    let out = run(
        &f,
        vec![
            s.arg("q", q_shape.clone()),
            f16_arg(&vec![0.0; (mq * nqh * hd) as usize], q_shape),
            f16_arg(kp, pool.clone()),
            f16_arg(vp, pool),
            pt,
            f16_arg(&mask, vec![nb as usize * rows * gqa, b]),
        ],
    );
    out[1].clone()
}

/// ⭐ ONE SERVING STEP: our scatter writes this step's K/V into the pools, then our page walk
/// gathers its pages out of THOSE pools — and both are held to spyre-inference's kernels.
fn the_step_matches_spyre_inference(config: &str) {
    let s = Step::load(config);
    let (mq, nkvh, hd) = (s.mq(), s.geom("NKVH"), s.geom("HD"));
    let pool_rows = s.geom("NUM_PAGES") * nkvh * s.geom("BLOCK");

    // THE WRITE — exact: a scatter moves bits.
    let w = lower(KV_WRITE, &kv_write_spec(mq, nkvh, hd, pool_rows));
    let rows = (mq * nkvh) as usize;
    let pool = vec![pool_rows as usize, hd as usize];
    let after = run(
        &w,
        vec![
            s.arg("k_new", vec![rows, hd as usize]),
            s.arg("v_new", vec![rows, hd as usize]),
            s.arg("kp_pre", pool.clone()),
            s.arg("vp_pre", pool),
            s.arg("write_rows", vec![rows]),
        ],
    );
    let (kp, vp) = (after[2].clone(), after[3].clone());
    let (want_k, want_v) = (s.values("kp_post"), s.values("vp_post"));
    assert!(
        kp == want_k,
        "{config}: the K pool after the scatter is not spyre-inference's"
    );
    assert!(
        vp == want_v,
        "{config}: the V pool after the scatter is not spyre-inference's"
    );
    // The write's own control: the step's write CHANGES the pools, so a scatter that wrote
    // nothing could not have produced `kp_post` from `kp_pre`.
    assert!(
        s.values("kp_pre") != want_k,
        "{config}: the step's write leaves the K pool unchanged, so it tests nothing"
    );

    // THE ATTENTION, over the pools our scatter wrote.
    let nb = s.num_blocks() as usize;
    let got = attend(&s, &kp, &vp, s.arg("page_table", vec![nb, 32]));
    let (exact, served) = (s.values("out_f32"), s.values("out_f16"));
    let err = max_abs(&got, &exact);
    let theirs = max_abs(&served, &exact);
    // THE GATE: spyre-inference's OWN f16 run misses its f32 run by `theirs` — the precision the
    // algorithm has at the model dtype. Ours is the same algorithm at the same dtype on a
    // different executor (f32 contractions, IEEE rounding), so it may land on either side of
    // `theirs`, but within a small multiple of it; a wrong page, a dropped rescale or a missing
    // token is far past it (the controls below measure exactly that, and must miss by at least
    // `CONTROL_MARGIN` gates — a control that lands near the gate proves nothing either way).
    let gate = 4.0 * theirs.max(5e-4);
    println!(
        "{config}: max|ours - f32| {err:.3e}, spyre-inference's own f16 {theirs:.3e}, \
         max|ours - their f16| {:.3e}, gate {gate:.3e}",
        max_abs(&got, &served)
    );
    assert!(
        err < gate,
        "{config}: {err:.3e} from spyre-inference's answer, gate {gate:.3e}"
    );

    // CONTROL 1 — the pages are read THROUGH the table: the same pools under a rotated table
    // must miss (a direct read of the pool, or the table ignored, would not).
    if nb > 1 {
        let (mut table, shape, _) = s.file("page_table");
        table.rotate_left(32 * 4);
        let rotated = attend(
            &s,
            &kp,
            &vp,
            Arg::TensorBytes {
                data: table,
                shape,
                dtype: DType::I32,
            },
        );
        let miss = max_abs(&rotated, &exact);
        println!("{config}: rotated page table misses by {miss:.3e}");
        assert!(
            miss > CONTROL_MARGIN * gate,
            "{config}: a rotated page table still matches ({miss:.3e})"
        );
    }
    // CONTROL 2 — the attention reads what the scatter wrote: over the PRE-write pools it must
    // miss, because this step's own keys are absent.
    let stale = attend(
        &s,
        &s.values("kp_pre"),
        &s.values("vp_pre"),
        s.arg("page_table", vec![nb, 32]),
    );
    let miss = max_abs(&stale, &exact);
    println!("{config}: attending the pre-write pools misses by {miss:.3e}");
    assert!(
        miss > CONTROL_MARGIN * gate,
        "{config}: the pre-write pools still match ({miss:.3e})"
    );
}

#[test]
fn decode_matches_spyre_inference() {
    the_step_matches_spyre_inference("paged_attn_decode");
}

/// Two kv heads: the write is two rows, under Triton's scatter floor, so it runs the kernel's
/// row-at-a-time arm — held to the same reference.
#[test]
fn decode_over_two_kv_heads_matches_spyre_inference() {
    the_step_matches_spyre_inference("paged_attn_decode_kv2");
}

#[test]
fn a_prompt_chunk_over_a_cached_prefix_matches_spyre_inference() {
    the_step_matches_spyre_inference("paged_attn_prefill");
}

#[test]
fn a_first_prompt_chunk_matches_spyre_inference() {
    the_step_matches_spyre_inference("paged_attn_prefill_first");
}

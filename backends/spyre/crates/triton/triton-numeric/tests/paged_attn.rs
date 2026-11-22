// SPDX-License-Identifier: Apache-2.0

//! PAGED ATTENTION, EXECUTED: spyre-inference's serving step — the KV cache write
//! (`reshape_and_cache_head_major`) and the head-major page walk (`page_attn_head_major_decode` /
//! `_prefill`) — as Triton, kernel for kernel (`kernels/reshape_and_cache_head_major.py`,
//! `kernels/page_attn_head_major_decode.py`, `kernels/page_attn_head_major_prefill.py`), through the
//! whole ladder, run by `ktir_emulator`, on spyre-inference's own operands.
//!
//! ```text
//!   triton_frontend      .py -> TTIR       (desc.scatter / gather, the page-id read, the page load)
//!   triton_ktir          TTIR -> KTIR      (indirect access tiles: the stores and the page loads)
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

const DECODE: &str = include_str!("../../../../kernels/page_attn_head_major_decode.py");
const PREFILL: &str = include_str!("../../../../kernels/page_attn_head_major_prefill.py");
const KV_WRITE: &str = include_str!("../../../../kernels/reshape_and_cache_head_major.py");
const BATCHED: &str = include_str!("../../../../kernels/batched_decode_head_major.py");
const DECODE_FP8: &str = include_str!("../../../../kernels/page_attn_head_major_decode_fp8.py");
const PREFILL_FP8: &str = include_str!("../../../../kernels/page_attn_head_major_prefill_fp8.py");
const KV_WRITE_FP8: &str = include_str!("../../../../kernels/reshape_and_cache_head_major_fp8.py");
const BATCHED_FP8: &str = include_str!("../../../../kernels/batched_decode_head_major_fp8.py");

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

fn kv_write_spec(t: i128, nkvh: i128, hd: i128, num_rows: i128) -> KernelSpec {
    spec(
        "reshape_and_cache_head_major.py",
        "reshape_and_cache_head_major_kernel",
        &[
            ("key", "*fp16"),
            ("value", "*fp16"),
            ("k_rows", "*fp16"),
            ("v_rows", "*fp16"),
            ("row_index", "*i32"),
        ],
        &[
            ("T", t),
            ("NUM_KV_HEADS", nkvh),
            ("HEAD_SIZE", hd),
            ("NUM_ROWS", num_rows),
        ],
    )
}

#[derive(Clone, Copy)]
struct AttnShape {
    padded_query_len: i128,
    num_heads: i128,
    num_kv_heads: i128,
    head_size: i128,
    block_size: i128,
    num_pages: i128,
    num_blocks: i128,
    /// The prefill walk's pages a trip — 1 for spyre-inference's own walk.
    blocks_per_trip: i128,
}

/// The kernel a sequence's step runs — spyre-inference's dispatch: the decode kernel at one query
/// row, the prefill kernel past it.
fn attn_kernel(s: &AttnShape) -> (&'static str, KernelSpec) {
    let decode = s.padded_query_len == 1;
    let pointers = [
        ("query", "*fp16"),
        ("query_row_index", "*i32"),
        ("k_pages", "*fp16"),
        ("v_pages", "*fp16"),
        ("page_index_table", "*i32"),
        ("mask_stack", "*fp16"),
        ("out", "*fp16"),
    ];
    let mut ce = vec![
        ("NUM_TOKENS", s.padded_query_len),
        ("NUM_HEADS", s.num_heads),
        ("NUM_KV_HEADS", s.num_kv_heads),
        ("HEAD_SIZE", s.head_size),
        ("BLOCK_SIZE", s.block_size),
        ("NUM_PAGES", s.num_pages),
        ("NUM_BLOCKS", s.num_blocks),
    ];
    let (src, file, kernel) = if decode {
        (
            DECODE,
            "page_attn_head_major_decode.py",
            "page_attn_head_major_decode_kernel",
        )
    } else {
        ce.push(("PADDED_QUERY_LEN", s.padded_query_len));
        ce.push(("BLOCKS_PER_TRIP", s.blocks_per_trip));
        (
            PREFILL,
            "page_attn_head_major_prefill.py",
            "page_attn_head_major_prefill_kernel",
        )
    };
    let mut k = spec(file, kernel, &pointers, &ce);
    k.signature.insert("SCALE".to_string(), ArgSpec::Constexpr);
    k.constexprs.insert(
        "SCALE".to_string(),
        Val::Float(1.0 / (s.head_size as f64).sqrt()),
    );
    (src, k)
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
    // A prompt chunk's write: one scatter per kv head per cache (their per-head `index_copy_`).
    let w = lower(KV_WRITE, &kv_write_spec(32, 8, 64, 8 * 8 * 64));
    assert_eq!(
        count(&w, OpKind::KtdpConstructIndirectAccessTile),
        2 * 8,
        "one scatter tile per (kv head, cache)"
    );
    // A decode step's one token is under Triton's scatter floor of 8 rows: each row is its own
    // store at the row its head's index names — still an indirect tile, one per (head, cache).
    let w1 = lower(KV_WRITE, &kv_write_spec(1, 8, 64, 8 * 8 * 64));
    assert_eq!(
        count(&w1, OpKind::KtdpConstructIndirectAccessTile),
        2 * 8,
        "one indirect store tile per (row, kv head, cache)"
    );
    let decode = AttnShape {
        padded_query_len: 1,
        num_heads: 32,
        num_kv_heads: 8,
        head_size: 64,
        block_size: 64,
        num_pages: 8,
        num_blocks: 3,
        blocks_per_trip: 1,
    };
    let (src, k) = attn_kernel(&decode);
    let a = lower(src, &k);
    // ⭐ THE WALK IS A LOOP — `walk_tiles`' ONE block body, not a copy per block.
    assert_eq!(
        count(&a, OpKind::ScfFor),
        1,
        "one page walk, standing as a loop"
    );
    // The query row and the output row (through `query_row_index`) and the block's K and V rows
    // (through its `kv_row_pool` row, itself a direct tile at the page the table names) — every
    // one read or written through memory.
    assert_eq!(
        count(&a, OpKind::KtdpConstructIndirectAccessTile),
        4,
        "query, K, V and out through their indices"
    );
    // And the program is the same size however long the walk.
    let (src, k) = attn_kernel(&AttnShape {
        num_pages: 64,
        num_blocks: 64,
        ..decode
    });
    assert_eq!(
        lower(src, &k).ops_deep().len(),
        a.ops_deep().len(),
        "a longer walk grew the program"
    );
    // The prompt kernel gathers each page ONCE, whole: every query head rides the one walk.
    let (src, k) = attn_kernel(&AttnShape {
        padded_query_len: 32,
        ..decode
    });
    let p = lower(src, &k);
    assert_eq!(
        count(&p, OpKind::ScfFor),
        1,
        "one page walk, standing as a loop"
    );
    assert_eq!(
        count(&p, OpKind::KtdpConstructIndirectAccessTile),
        4,
        "the query rows, the K page, the V page and the output rows through their indices"
    );
    // ⛔ NO WIDEN SURVIVES. The kernels' `exp` islands (`x.to(f32)` → exp → `.to(f16)`) collapse
    // to the bare f16 op, so no program carries an f16→f32 `arith.extf` — the convert whose
    // card form (`dl16tofp32`) permutes lanes, which a fused forward models.
    for (name, f) in [
        ("reshape_and_cache_head_major", &w),
        ("reshape_and_cache_head_major (1 token)", &w1),
        ("page_attn_head_major_decode", &a),
        ("page_attn_head_major_prefill", &p),
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

/// spyre-inference's `[NUM_BLOCKS, 32]` page table as recorded, its rows rotated left by `rotate`
/// (a control's wrong table).
fn page_table(s: &Step, rotate: usize) -> Arg {
    let (raw, shape, _) = s.file("page_table");
    let (nb, stick) = (shape[0], shape[1]);
    let row = stick * 4;
    let data: Vec<u8> = (0..nb)
        .flat_map(|b| {
            let from = (b + rotate) % nb * row;
            raw[from..from + row].to_vec()
        })
        .collect();
    Arg::TensorBytes {
        data,
        shape,
        dtype: DType::I32,
    }
}

fn i32_arg(words: &[i32], shape: Vec<usize>) -> Arg {
    Arg::TensorBytes {
        data: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        shape,
        dtype: DType::I32,
    }
}

/// The step's attention, on the emulator, over pools `kp`/`vp` with page table `pt` — the kernel
/// spyre-inference dispatches at this query width, on their operands.
fn attend(s: &Step, kp: &[f64], vp: &[f64], pt: Arg) -> Vec<f64> {
    let (rows, block) = (s.mq() as usize, s.geom("BLOCK") as usize);
    // `[num_blocks, padded_query_len, block_size]`, as recorded.
    let mask = s.arg("mask", vec![s.num_blocks() as usize * rows, block]);
    attend_walk(s, kp, vp, (pt, mask), (s.num_blocks(), 1))
}

/// [`attend`] over a walk of `num_blocks` blocks taken `blocks_per_trip` pages a trip, with its own
/// page table and mask stack.
fn attend_walk(
    s: &Step,
    kp: &[f64],
    vp: &[f64],
    (pt, mask): (Arg, Arg),
    (num_blocks, blocks_per_trip): (i128, i128),
) -> Vec<f64> {
    let (mq, nqh, nkvh, hd) = (s.mq(), s.geom("NQH"), s.geom("NKVH"), s.geom("HD"));
    let (block, pages) = (s.geom("BLOCK"), s.geom("NUM_PAGES"));
    let shape = AttnShape {
        padded_query_len: mq,
        num_heads: nqh,
        num_kv_heads: nkvh,
        head_size: hd,
        block_size: block,
        num_pages: pages,
        num_blocks,
        blocks_per_trip,
    };
    let (src, k) = attn_kernel(&shape);
    let f = lower(src, &k);
    let (rows, width) = (mq as usize, (nqh * hd) as usize);
    let q_shape = vec![rows, width];
    // The batch is this one sequence: its rows are the query's rows, in order.
    let row_index: Vec<i32> = (0..mq as i32).collect();
    // The unfolded cache `[pages, kv · block · D]`: one row per page.
    let unfolded = vec![pages as usize, (nkvh * block * hd) as usize];
    let bound = vec![
        s.arg("q", q_shape.clone()),
        i32_arg(&row_index, vec![rows]),
        f16_arg(kp, unfolded.clone()),
        f16_arg(vp, unfolded),
        pt,
        mask,
        f16_arg(&vec![0.0; rows * width], q_shape),
    ];
    let out = run(&f, bound);
    out.last().expect("the out operand").clone()
}

/// ⭐ ONE SERVING STEP: our scatter writes this step's K/V into the pools, then our page walk
/// gathers its pages out of THOSE pools — and both are held to spyre-inference's kernels.
fn the_step_matches_spyre_inference(config: &str) {
    let s = Step::load(config);
    let (mq, nkvh, hd) = (s.mq(), s.geom("NKVH"), s.geom("HD"));
    let pool_rows = s.geom("NUM_PAGES") * nkvh * s.geom("BLOCK");

    // THE WRITE — exact: a scatter moves bits.
    let w = lower(KV_WRITE, &kv_write_spec(mq, nkvh, hd, pool_rows));
    let (t, kvh) = (mq as usize, nkvh as usize);
    let pool = vec![pool_rows as usize, hd as usize];
    // `kv_write_index`'s one `[T]` index per kv head, one after another: the recording holds the
    // same rows `(token, head)`-major.
    let (raw, _, _) = s.file("write_rows");
    let token_major: Vec<i32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| i32::from_le_bytes(*w))
        .collect();
    let per_head: Vec<i32> = (0..kvh)
        .flat_map(|h| (0..t).map(move |tok| (h, tok)))
        .map(|(h, tok)| token_major[tok * kvh + h])
        .collect();
    let kv_shape = vec![t, (nkvh * hd) as usize];
    let after = run(
        &w,
        vec![
            s.arg("k_new", kv_shape.clone()),
            s.arg("v_new", kv_shape),
            s.arg("kp_pre", pool.clone()),
            s.arg("vp_pre", pool),
            i32_arg(&per_head, vec![t * kvh]),
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
    let got = attend(&s, &kp, &vp, page_table(&s, 0));
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
        let rotated = attend(&s, &kp, &vp, page_table(&s, 1));
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
        page_table(&s, 0),
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

/// ⭐ OUR TRIP WALK — the prefill kernel taking eight pages a trip (`BLOCKS_PER_TRIP`, the form a
/// walk past `MAX_WALK_TRIPS` trips bakes) answers what its walk page by page answers (held to
/// spyre-inference's own kernels above): one prompt chunk of eight rows after a cached prefix of
/// fifteen pages, its sixteen pages scattered over the pool. The page ids ride one word a block in
/// the trip form, one int32 stick a block in theirs.
#[test]
fn a_walk_of_eight_pages_a_trip_is_the_walk_page_by_page() {
    const TRIP: usize = 8;
    let (p, nqh, nkvh, hd, block, pages) = (8usize, 4usize, 2usize, 64usize, 64usize, 16usize);
    let walk = pages;
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut noise = |n: usize| -> Vec<f64> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed % 2001) as f64 / 1000.0 - 1.0
            })
            .collect()
    };
    let q = noise(p * nqh * hd);
    let page_elems = nkvh * block * hd;
    let (kp, vp) = (noise(pages * page_elems), noise(pages * page_elems));
    // Block b of the walk sits on page (5·b + 3) mod 16: every page once, none in order.
    let page_of = |b: usize| ((5 * b + 3) % pages) as i32;
    // The chunk's rows sit at positions 960.. after the prefix; a key past a row's position is
    // masked.
    let start = (walk - 1) * block;
    let mask: Vec<f64> = (0..walk)
        .flat_map(|b| {
            (0..p).flat_map(move |r| {
                (0..block).map(move |c| {
                    if b * block + c <= start + r {
                        0.0
                    } else {
                        -65504.0
                    }
                })
            })
        })
        .collect();
    let run_walk = |trip: usize, table: Arg| -> Vec<f64> {
        let shape = AttnShape {
            padded_query_len: p as i128,
            num_heads: nqh as i128,
            num_kv_heads: nkvh as i128,
            head_size: hd as i128,
            block_size: block as i128,
            num_pages: pages as i128,
            num_blocks: walk as i128,
            blocks_per_trip: trip as i128,
        };
        let (src, k) = attn_kernel(&shape);
        let f = lower(src, &k);
        let rows: Vec<i32> = (0..p as i32).collect();
        let bound = vec![
            f16_arg(&q, vec![p, nqh * hd]),
            i32_arg(&rows, vec![p]),
            f16_arg(&kp, vec![pages, page_elems]),
            f16_arg(&vp, vec![pages, page_elems]),
            table,
            f16_arg(&mask, vec![walk * p, block]),
            f16_arg(&vec![0.0; p * nqh * hd], vec![p, nqh * hd]),
        ];
        run(&f, bound).last().expect("the out operand").clone()
    };
    let stick = 32;
    let mut sticks = vec![0i32; walk * stick];
    for b in 0..walk {
        sticks[b * stick] = page_of(b);
    }
    let words: Vec<i32> = (0..walk).map(page_of).collect();
    let page_by_page = run_walk(1, i32_arg(&sticks, vec![walk, stick]));
    let by_trips = run_walk(TRIP, i32_arg(&words, vec![walk]));
    let err = max_abs(&by_trips, &page_by_page);
    // Both walks round every softmax step to f16; they differ only in where the running maximum
    // is taken.
    let gate = 2e-3;
    println!("trip walk vs page walk: {err:.3e}, gate {gate:.3e}");
    assert!(
        err < gate,
        "eight pages a trip answers {err:.3e} apart from page by page"
    );
    // The control: the trip walk over its pages read in another order must miss.
    let shifted: Vec<i32> = (0..walk).map(|b| page_of((b + 1) % walk)).collect();
    let miss = max_abs(
        &run_walk(TRIP, i32_arg(&shifted, vec![walk])),
        &page_by_page,
    );
    assert!(
        miss > CONTROL_MARGIN * gate,
        "the trip walk over shifted pages still matches ({miss:.3e})"
    );
}

/// The pools a batch reads: as recorded, or these values on fp16 head-major pages or fp8
/// token-major ones.
enum Pages<'a> {
    Recorded,
    Fp16(&'a [f64], &'a [f64]),
    Fp8(&'a [f64], &'a [f64]),
}

/// ⭐ SEVERAL SEQUENCES' DECODE IN ONE LAUNCH — `batched_decode_head_major_kernel` over the pools
/// and the tiled walk's metadata (`gen_paged_attn_data.py`'s `BATCHED_CONFIGS`), held to
/// spyre-inference's own kernel at its own chunking.
struct Batch {
    step: Step,
    seqs: i128,
    blocks_per_chunk: i128,
    num_chunks: i128,
    /// The index rows one page's gather takes: 1, or the kv heads (the tables folded by kv head).
    rows_per_page: i128,
}

impl Batch {
    fn load(config: &str) -> Batch {
        let step = Step::load(config);
        let t = &step.meta["derived"]["triton"];
        let get = |v: &serde_json::Value| v.as_i64().expect("an integer") as i128;
        Batch {
            seqs: get(&step.meta["seqs"]),
            blocks_per_chunk: get(&t["blocks_per_chunk"]),
            num_chunks: get(&t["num_chunks"]),
            rows_per_page: get(&t["rows_per_page"]),
            step,
        }
    }

    fn entries(&self) -> i128 {
        self.blocks_per_chunk * self.seqs
    }

    /// The kernel's answer for these query rows and page ids (each as recorded unless replaced).
    fn attend(&self, rep_row_ids: &[i32], page_ids: &[i32]) -> Vec<f64> {
        self.attend_on(rep_row_ids, page_ids, Pages::Recorded)
    }

    /// [`Self::attend`] over `pages`.
    fn attend_on(&self, rep_row_ids: &[i32], page_ids: &[i32], pages_of: Pages<'_>) -> Vec<f64> {
        let s = &self.step;
        let (nqh, nkvh, hd) = (s.geom("NQH"), s.geom("NKVH"), s.geom("HD"));
        let (block, pages) = (s.geom("BLOCK"), s.geom("NUM_PAGES"));
        let mut k = spec(
            "batched_decode_head_major.py",
            "batched_decode_head_major_kernel",
            &[
                ("query", "*fp16"),
                ("rep_row_ids", "*i32"),
                ("k_pages", "*fp16"),
                ("v_pages", "*fp16"),
                ("chunk_page_ids", "*i32"),
                ("mask_by_chunk", "*fp16"),
                ("out", "*fp16"),
            ],
            &[
                ("NUM_TOKENS", self.seqs),
                ("NUM_SEQS", self.seqs),
                ("BLOCKS_PER_CHUNK", self.blocks_per_chunk),
                ("NUM_CHUNKS", self.num_chunks),
                ("NUM_KV_HEADS", nkvh),
                ("NUM_QUERIES_PER_KV", nqh / nkvh),
                ("BLOCK_SIZE", block),
                ("HEAD_SIZE", hd),
                ("NUM_PAGES", pages),
                ("ROWS_PER_PAGE", self.rows_per_page),
            ],
        );
        k.signature.insert("SCALE".to_string(), ArgSpec::Constexpr);
        k.constexprs.insert(
            "SCALE".to_string(),
            Val::Float(s.meta["derived"]["scale"].as_f64().expect("scale")),
        );
        let (rows, width) = (self.seqs as usize, (nqh * hd) as usize);
        let e = self.entries() as usize;
        let index_rows = e * self.rows_per_page as usize;
        let page = vec![pages as usize, (nkvh * block * hd) as usize];
        let (k, kp, vp) = match pages_of {
            Pages::Recorded => ((BATCHED, k), s.arg("kp", page.clone()), s.arg("vp", page)),
            Pages::Fp16(kp, vp) => ((BATCHED, k), f16_arg(kp, page.clone()), f16_arg(vp, page)),
            Pages::Fp8(kp, vp) => (
                fp8_twin(k, ["k_pages", "v_pages"]),
                fp8_arg(kp, page.clone()),
                fp8_arg(vp, page),
            ),
        };
        let (src, k) = k;
        let f = lower(src, &k);
        let out = run(
            &f,
            vec![
                s.arg("q", vec![rows, width]),
                i32_arg(rep_row_ids, vec![index_rows]),
                kp,
                vp,
                i32_arg(page_ids, vec![self.num_chunks as usize * index_rows]),
                self.mask_rows(e, nkvh as usize, block as usize),
                f16_arg(&vec![0.0; rows * width], vec![rows, width]),
            ],
        );
        out.last().expect("the out operand").clone()
    }

    /// The recorded mask `[C·E·NKVH, BLOCK]` (spyre-inference's stack, its kv axis materialized) as
    /// the kernel reads it, `[C·E, BLOCK]`: one row per (block slot, sequence), every kv head reading
    /// that row through the kernel's broadcast. The recorded copies of a row are asserted equal, so
    /// taking the first loses nothing the reference states.
    fn mask_rows(&self, e: usize, nkvh: usize, block: usize) -> Arg {
        let (data, recorded, dtype) = self.step.file("mask");
        let rows = self.num_chunks as usize * e;
        assert_eq!(
            recorded.iter().product::<usize>(),
            rows * nkvh * block,
            "mask: {recorded:?}"
        );
        let width = block * 2;
        let mut out = Vec::with_capacity(rows * width);
        for r in 0..rows {
            let first = &data[(r * nkvh) * width..][..width];
            for h in 1..nkvh {
                assert_eq!(
                    &data[(r * nkvh + h) * width..][..width],
                    first,
                    "mask row {r}: kv head {h} differs from kv head 0"
                );
            }
            out.extend_from_slice(first);
        }
        Arg::TensorBytes {
            data: out,
            shape: vec![rows, block],
            dtype,
        }
    }

    fn words(&self, key: &str) -> Vec<i32> {
        let (raw, _, _) = self.step.file(key);
        raw.as_chunks::<4>()
            .0
            .iter()
            .map(|w| i32::from_le_bytes(*w))
            .collect()
    }
}

fn the_batch_matches_spyre_inference(config: &str) {
    let b = Batch::load(config);
    let (rep, ids) = (b.words("rep_row_ids"), b.words("chunk_page_ids"));
    let got = b.attend(&rep, &ids);
    let (exact, served) = (b.step.values("out_f32"), b.step.values("out_f16"));
    let err = max_abs(&got, &exact);
    let theirs = max_abs(&served, &exact);
    // The gate of `the_step_matches_spyre_inference`: a small multiple of spyre-inference's own
    // f16-vs-f32 miss.
    let gate = 4.0 * theirs.max(5e-4);
    println!(
        "{config}: B={} J={} C={}: max|ours - f32| {err:.3e}, spyre-inference's own f16 \
         {theirs:.3e}, max|ours - their f16| {:.3e}, gate {gate:.3e}",
        b.seqs,
        b.blocks_per_chunk,
        b.num_chunks,
        max_abs(&got, &served)
    );
    assert!(
        err < gate,
        "{config}: {err:.3e} from spyre-inference's answer, gate {gate:.3e}"
    );

    // Both tables hold `x · per + a` at entry `e`'s row `a` (`per` = 1: `x` itself).
    let per = b.rows_per_page as i32;
    // CONTROL 1 — each sequence attends its OWN query: rotating the sequences' query rows must
    // miss (a kernel reading one row for every entry would not).
    let seqs = b.seqs as i32;
    let rotated_rows: Vec<i32> = rep
        .iter()
        .map(|&r| ((r / per + 1) % seqs) * per + r % per)
        .collect();
    let miss = max_abs(&b.attend(&rotated_rows, &ids), &exact);
    println!("{config}: rotated query rows miss by {miss:.3e}");
    assert!(
        miss > CONTROL_MARGIN * gate,
        "{config}: rotated query rows still match ({miss:.3e})"
    );
    // CONTROL 2 — the pages are read THROUGH the ids: shifting every id by one page must miss.
    let pages = b.step.geom("NUM_PAGES") as i32;
    let shifted: Vec<i32> = ids
        .iter()
        .map(|&p| ((p / per + 1) % pages) * per + p % per)
        .collect();
    let miss = max_abs(&b.attend(&rep, &shifted), &exact);
    println!("{config}: shifted page ids miss by {miss:.3e}");
    assert!(
        miss > CONTROL_MARGIN * gate,
        "{config}: shifted page ids still match ({miss:.3e})"
    );
    // CONTROL 3 — a page's kv heads are gathered as its rows: rotating each page's heads must
    // miss.
    if per > 1 {
        let heads_rotated: Vec<i32> = ids
            .iter()
            .map(|&p| p - p % per + (p % per + 1) % per)
            .collect();
        let miss = max_abs(&b.attend(&rep, &heads_rotated), &exact);
        println!("{config}: rotated kv heads miss by {miss:.3e}");
        assert!(
            miss > CONTROL_MARGIN * gate,
            "{config}: rotated kv heads still match ({miss:.3e})"
        );
    }
}

/// Three sequences in the 4-sequence bucket, one chunk: the slots share one maximum per sequence.
#[test]
fn a_decode_batch_matches_spyre_inference() {
    the_batch_matches_spyre_inference("batched_decode");
}

/// Two chunks: each block slot keeps its own softmax, and the slots merge after the walk.
#[test]
fn a_decode_batch_over_two_chunks_matches_spyre_inference() {
    the_batch_matches_spyre_inference("batched_decode_chunks");
}

/// A chunk under Triton's gather floor in pages (two sequences of one block): gathered by kv head,
/// its tables folded, and at two kv heads those 4 rows padded to 8 with a masked block slot.
#[test]
fn a_decode_batch_under_the_gather_floor_matches_spyre_inference() {
    the_batch_matches_spyre_inference("batched_decode_floor");
}

// ── THE FP8 POOL: token-major pages, read back as the fp16 step reads head-major ones ───────────

/// A head-major pool `[pages, kv, block, hd]` laid token-major, `[pages, block, kv, hd]`.
fn token_major(head_major: &[f64], (nkvh, block, hd): (usize, usize, usize)) -> Vec<f64> {
    let page = nkvh * block * hd;
    let mut out = vec![0.0; head_major.len()];
    for (p, src) in head_major.chunks_exact(page).enumerate() {
        for h in 0..nkvh {
            for t in 0..block {
                let from = (h * block + t) * hd;
                let to = p * page + (t * nkvh + h) * hd;
                out[to..to + hd].copy_from_slice(&src[from..from + hd]);
            }
        }
    }
    out
}

/// [`token_major`]'s inverse.
fn head_major(token_major: &[f64], (nkvh, block, hd): (usize, usize, usize)) -> Vec<f64> {
    let page = nkvh * block * hd;
    let mut out = vec![0.0; token_major.len()];
    for (p, src) in token_major.chunks_exact(page).enumerate() {
        for h in 0..nkvh {
            for t in 0..block {
                let from = (t * nkvh + h) * hd;
                let to = p * page + (h * block + t) * hd;
                out[to..to + hd].copy_from_slice(&src[from..from + hd]);
            }
        }
    }
    out
}

fn fp8_bytes(values: &[f64]) -> Vec<u8> {
    ktir_core::codec::encode(
        &values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        DType::Fp8E4m3,
    )
}

fn fp8_arg(values: &[f64], shape: Vec<usize>) -> Arg {
    Arg::TensorBytes {
        data: fp8_bytes(values),
        shape,
        dtype: DType::Fp8E4m3,
    }
}

/// `values` rounded to fp8 E4M3 and read back.
fn fp8_round(values: &[f64]) -> Vec<f64> {
    ktir_core::codec::decode(&fp8_bytes(values), values.len(), DType::Fp8E4m3)
        .into_iter()
        .map(f64::from)
        .collect()
}

/// A paged kernel's fp8-pool twin: its `_fp8` file's `_fp8_kernel` entry, its pool pointers fp8,
/// and its source.
fn fp8_twin(mut k: KernelSpec, pools: [&str; 2]) -> (&'static str, KernelSpec) {
    k.kernel = k.kernel.replace("_kernel", "_fp8_kernel");
    k.file = k.file.replace(".py", "_fp8.py");
    let src = match k.file.as_str() {
        "page_attn_head_major_decode_fp8.py" => DECODE_FP8,
        "page_attn_head_major_prefill_fp8.py" => PREFILL_FP8,
        "reshape_and_cache_head_major_fp8.py" => KV_WRITE_FP8,
        "batched_decode_head_major_fp8.py" => BATCHED_FP8,
        other => panic!("{other}: no fp8 twin"),
    };
    for p in pools {
        k.signature.insert(
            p.to_string(),
            ArgSpec::parse("*fp8e4nv").expect("pointer spec"),
        );
    }
    (src, k)
}

/// The step's attention on the fp8 twin over token-major fp8 pools `kp`/`vp`.
fn attend_fp8(s: &Step, kp: &[f64], vp: &[f64]) -> Vec<f64> {
    let (mq, nqh, nkvh, hd) = (s.mq(), s.geom("NQH"), s.geom("NKVH"), s.geom("HD"));
    let (block, pages) = (s.geom("BLOCK"), s.geom("NUM_PAGES"));
    let rows = mq as usize;
    let shape = AttnShape {
        padded_query_len: mq,
        num_heads: nqh,
        num_kv_heads: nkvh,
        head_size: hd,
        block_size: block,
        num_pages: pages,
        num_blocks: s.num_blocks(),
        blocks_per_trip: 1,
    };
    let (_, k) = attn_kernel(&shape);
    let (src, k) = fp8_twin(k, ["k_pages", "v_pages"]);
    let f = lower(src, &k);
    let width = (nqh * hd) as usize;
    let q_shape = vec![rows, width];
    let row_index: Vec<i32> = (0..mq as i32).collect();
    let unfolded = vec![pages as usize, (nkvh * block * hd) as usize];
    let mask = s.arg("mask", vec![s.num_blocks() as usize * rows, block as usize]);
    let bound = vec![
        s.arg("q", q_shape.clone()),
        i32_arg(&row_index, vec![rows]),
        fp8_arg(kp, unfolded.clone()),
        fp8_arg(vp, unfolded),
        page_table(s, 0),
        mask,
        f16_arg(&vec![0.0; rows * width], q_shape),
    ];
    run(&f, bound).last().expect("the out operand").clone()
}

/// ⭐ ONE SERVING STEP OVER AN FP8 POOL. The fp8 write narrows this step's K/V into token-major
/// pages — exactly the pools spyre-inference's step leaves, rounded to fp8 and laid token-major —
/// and the fp8 walk over them answers what the fp16 walk answers over the same values laid
/// head-major. The control: the fp8 walk over the same bytes laid head-major misses.
fn the_fp8_step_is_the_fp16_step_on_its_values(config: &str) {
    let s = Step::load(config);
    let (mq, nkvh, hd, block) = (s.mq(), s.geom("NKVH"), s.geom("HD"), s.geom("BLOCK"));
    let geom = (nkvh as usize, block as usize, hd as usize);
    let pool_rows = s.geom("NUM_PAGES") * nkvh * block;
    let stick = <ktir_core::ddl_binds::Baked as ktir_core::ddl_binds::DeviceGen>::FLIT_BYTES;
    let (src, k) = fp8_twin(
        spec(
            "reshape_and_cache_head_major.py",
            "reshape_and_cache_head_major_kernel",
            &[
                ("key", "*fp16"),
                ("value", "*fp16"),
                ("k_rows", "*fp16"),
                ("v_rows", "*fp16"),
                ("row_index", "*i32"),
            ],
            &[
                ("T", mq),
                ("NUM_KV_HEADS", nkvh),
                ("HEAD_SIZE", hd),
                ("NUM_ROWS", pool_rows),
                ("STICK", i128::from(stick)),
            ],
        ),
        ["k_rows", "v_rows"],
    );
    let w = lower(src, &k);
    // Each token's row on a token-major page, from the recording's head-0 row
    // `(page · nkvh + 0) · block + slot`.
    let (raw, _, _) = s.file("write_rows");
    let recorded: Vec<i32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| i32::from_le_bytes(*w))
        .collect();
    let (t, kvh, blk) = (mq as usize, nkvh as usize, block as usize);
    let rows: Vec<i32> = (0..t)
        .map(|tok| {
            let r = recorded[tok * kvh] as usize;
            ((r / (kvh * blk)) * blk + r % blk) as i32
        })
        .collect();
    let pool = vec![(pool_rows / nkvh) as usize, (nkvh * hd) as usize];
    let kv_shape = vec![t, (nkvh * hd) as usize];
    let after = run(
        &w,
        vec![
            s.arg("k_new", kv_shape.clone()),
            s.arg("v_new", kv_shape),
            fp8_arg(&token_major(&s.values("kp_pre"), geom), pool.clone()),
            fp8_arg(&token_major(&s.values("vp_pre"), geom), pool),
            i32_arg(&rows, vec![t]),
        ],
    );
    let (kp, vp) = (after[2].clone(), after[3].clone());
    assert!(
        kp == fp8_round(&token_major(&s.values("kp_post"), geom)),
        "{config}: the fp8 K pool after the write is not spyre-inference's, rounded and token-major"
    );
    assert!(
        vp == fp8_round(&token_major(&s.values("vp_post"), geom)),
        "{config}: the fp8 V pool after the write is not spyre-inference's, rounded and token-major"
    );

    let got = attend_fp8(&s, &kp, &vp);
    let fp16 = attend(
        &s,
        &head_major(&kp, geom),
        &head_major(&vp, geom),
        page_table(&s, 0),
    );
    let err = max_abs(&got, &fp16);
    let theirs = max_abs(&s.values("out_f16"), &s.values("out_f32"));
    let gate = 4.0 * theirs.max(5e-4);
    println!(
        "{config}: fp8 walk vs fp16 walk on its values {err:.3e}, gate {gate:.3e}; \
         vs spyre-inference's f32 {:.3e}",
        max_abs(&got, &s.values("out_f32"))
    );
    assert!(
        err < gate,
        "{config}: the fp8 walk misses the fp16 walk on the same values by {err:.3e}, gate {gate:.3e}"
    );
    // CONTROL — the pages are read token-major: the same bytes laid head-major must miss.
    let wrong = attend_fp8(&s, &head_major(&kp, geom), &head_major(&vp, geom));
    let miss = max_abs(&wrong, &fp16);
    println!("{config}: the fp8 walk over head-major bytes misses by {miss:.3e}");
    assert!(
        miss > CONTROL_MARGIN * gate,
        "{config}: the fp8 walk over head-major bytes still matches ({miss:.3e})"
    );
}

/// One decode token: under the scatter's floor, its row stored as one block of whole sticks.
#[test]
fn an_fp8_decode_step_is_the_fp16_step_on_its_values() {
    the_fp8_step_is_the_fp16_step_on_its_values("paged_attn_decode");
}

/// A prompt chunk of 32 tokens: every row scattered at once.
#[test]
fn an_fp8_prompt_chunk_is_the_fp16_chunk_on_its_values() {
    the_fp8_step_is_the_fp16_step_on_its_values("paged_attn_prefill");
}

/// ⭐ A DECODE BATCH OVER AN FP8 POOL: the fp8 twin over the recorded pools rounded to fp8 and laid
/// token-major answers what the fp16 kernel answers over the same values laid head-major. The
/// control: the fp8 twin over the same bytes laid head-major misses.
fn the_fp8_batch_is_the_fp16_batch_on_its_values(config: &str) {
    let b = Batch::load(config);
    let s = &b.step;
    let geom = (
        s.geom("NKVH") as usize,
        s.geom("BLOCK") as usize,
        s.geom("HD") as usize,
    );
    let (rep, ids) = (b.words("rep_row_ids"), b.words("chunk_page_ids"));
    let (kp, vp) = (fp8_round(&s.values("kp")), fp8_round(&s.values("vp")));
    let (kt, vt) = (token_major(&kp, geom), token_major(&vp, geom));
    let got = b.attend_on(&rep, &ids, Pages::Fp8(&kt, &vt));
    let fp16 = b.attend_on(&rep, &ids, Pages::Fp16(&kp, &vp));
    let err = max_abs(&got, &fp16);
    let theirs = max_abs(&s.values("out_f16"), &s.values("out_f32"));
    let gate = 4.0 * theirs.max(5e-4);
    println!("{config}: fp8 batch vs fp16 batch on its values {err:.3e}, gate {gate:.3e}");
    assert!(
        err < gate,
        "{config}: the fp8 batch misses the fp16 batch on the same values by {err:.3e}, gate {gate:.3e}"
    );
    let wrong = b.attend_on(&rep, &ids, Pages::Fp8(&kp, &vp));
    let miss = max_abs(&wrong, &fp16);
    println!("{config}: the fp8 batch over head-major bytes misses by {miss:.3e}");
    assert!(
        miss > CONTROL_MARGIN * gate,
        "{config}: the fp8 batch over head-major bytes still matches ({miss:.3e})"
    );
}

#[test]
fn an_fp8_decode_batch_is_the_fp16_batch_on_its_values() {
    the_fp8_batch_is_the_fp16_batch_on_its_values("batched_decode");
}

#[test]
fn an_fp8_decode_batch_over_two_chunks_is_the_fp16_batch_on_its_values() {
    the_fp8_batch_is_the_fp16_batch_on_its_values("batched_decode_chunks");
}

#[test]
fn an_fp8_decode_batch_under_the_gather_floor_is_the_fp16_batch_on_its_values() {
    the_fp8_batch_is_the_fp16_batch_on_its_values("batched_decode_floor");
}

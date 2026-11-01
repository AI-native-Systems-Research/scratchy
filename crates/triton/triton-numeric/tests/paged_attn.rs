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

const DECODE: &str = include_str!("../../../targets/spyre/kernels/page_attn_head_major_decode.py");
const PREFILL: &str =
    include_str!("../../../targets/spyre/kernels/page_attn_head_major_prefill.py");
const KV_WRITE: &str =
    include_str!("../../../targets/spyre/kernels/reshape_and_cache_head_major.py");
const BATCHED: &str = include_str!("../../../targets/spyre/kernels/batched_decode_head_major.py");

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
}

/// The kernel a sequence's step runs — spyre-inference's dispatch: the decode kernel at one query
/// row, the prefill kernel past it.
fn attn_kernel(s: &AttnShape) -> (&'static str, KernelSpec) {
    let decode = s.padded_query_len == 1;
    let mut pointers = vec![
        ("query", "*fp16"),
        ("query_row_index", "*i32"),
        ("k_pages", "*fp16"),
        ("v_pages", "*fp16"),
        ("page_index_table", "*i32"),
    ];
    // Both one-sequence kernels read `kv_row_pool`: the prompt trip gathers its page as the pool's
    // `nkvh` rows, one kv head a core, as the decode trip does.
    pointers.push(("kv_row_pool", "*i32"));
    pointers.extend([("mask_stack", "*fp16"), ("out", "*fp16")]);
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
    let (mq, nqh, nkvh, hd) = (s.mq(), s.geom("NQH"), s.geom("NKVH"), s.geom("HD"));
    let (block, pages, nb) = (s.geom("BLOCK"), s.geom("NUM_PAGES"), s.num_blocks());
    let shape = AttnShape {
        padded_query_len: mq,
        num_heads: nqh,
        num_kv_heads: nkvh,
        head_size: hd,
        block_size: block,
        num_pages: pages,
        num_blocks: nb,
    };
    let (src, k) = attn_kernel(&shape);
    let f = lower(src, &k);
    let (rows, width) = (mq as usize, (nqh * hd) as usize);
    let q_shape = vec![rows, width];
    // The batch is this one sequence: its rows are the query's rows, in order.
    let row_index: Vec<i32> = (0..mq as i32).collect();
    let mut bound = vec![s.arg("q", q_shape.clone()), i32_arg(&row_index, vec![rows])];
    // BOTH one-sequence kernels read the FOLDED cache `[pages * kv, block * D]` through
    // `kv_row_pool` `[pages, kv, 1]` — the same bytes the unfolded `[pages, kv * block * D]` view
    // holds, gathered a kv head a core.
    let folded = vec![(pages * nkvh) as usize, (block * hd) as usize];
    bound.push(f16_arg(kp, folded.clone()));
    bound.push(f16_arg(vp, folded));
    bound.push(pt);
    let pool: Vec<i32> = (0..(pages * nkvh) as i32).collect();
    bound.push(i32_arg(&pool, vec![pages as usize, nkvh as usize]));
    // `[num_blocks, padded_query_len, block_size]`, as recorded.
    bound.push(s.arg("mask", vec![nb as usize * rows, block as usize]));
    bound.push(f16_arg(&vec![0.0; rows * width], q_shape));
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
        let f = lower(BATCHED, &k);
        let (rows, width) = (self.seqs as usize, (nqh * hd) as usize);
        let e = self.entries() as usize;
        let index_rows = e * self.rows_per_page as usize;
        let page = vec![pages as usize, (nkvh * block * hd) as usize];
        let out = run(
            &f,
            vec![
                s.arg("q", vec![rows, width]),
                i32_arg(rep_row_ids, vec![index_rows]),
                s.arg("kp", page.clone()),
                s.arg("vp", page),
                i32_arg(page_ids, vec![self.num_chunks as usize * index_rows]),
                s.arg(
                    "mask",
                    vec![self.num_chunks as usize * e * nkvh as usize, block as usize],
                ),
                f16_arg(&vec![0.0; rows * width], vec![rows, width]),
            ],
        );
        out.last().expect("the out operand").clone()
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

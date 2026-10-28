// SPDX-License-Identifier: Apache-2.0
// The model-independent body of the `cuda_hf_parity` tests, shared by every qwen3.5 model's
// binary (`cuda_hf_parity.rs` for qwen3.5-0.8b, `cuda_hf_parity_9b.rs` for qwen3.5-9b).
// Included with `include!` at the binary's root, not as a module, so every test keeps its name.
// The including binary supplies the model: its `CUDA_TAPES` and `CUDA_WEIGHTS`, and the
// constants `HIDDEN`, `WEIGHTS` (the weight buffers `load_weights` uploads), `UNTIED_LM_HEAD`
// (the lm_head's own tensor, `None` when it is tied to the embedding table), and the goldens'
// `MODEL` (`cuda_hf_reference.py --model`), `CHECKPOINT` and `GOLDENS` dir.
// qwen3.6-35b-a3b's `cuda_hf_parity_35b.rs` includes it too (AI-native-Systems-Research/
// scratchy#859): each binary states its P1 rule (`P1`, a `P1Rule`), and `load_checks` adds the
// 35B's MoE bundles' rows.
// gemma-4-26b-a4b-it's `cuda_hf_parity_gemma4.rs` includes it too (#933): each binary states its
// `VOCAB`, its GDN pool (`GDN_POOL`, `None` for a model with no GDN layer, whose forwards then pass
// no `GdnSequence` and whose `load_checks` has no GDN bundle to check) and its worker's allocations
// (`WORKER_ALLOCATIONS`: the arena, the KV pool and the GDN pool when there is one), and
// `load_checks` adds Gemma's expert banks' and routers' rows. Each binary also states
// `P3`, a `P3Rule`: `Strict` for every model but gemma-4, whose P3 accepts one top-1
// disagreement outside a near-tie per prompt on a route-explained row (the user's decision
// 2026-10-27, option (a)).

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

use half::bf16;
use scratchy_target_cuda::runtime::{
    CudaDevice, CudaWorker, DeviceBuffer, GdnPoolConfig, GdnSequence, Input, KvPoolConfig,
    KvSequence, load_weights,
};
use scratchy_target_cuda::step::{MoeRouteBf16, gdn_bundle};
use scratchy_target_cuda::weights::{CudaWeight, open_checkpoint};

// The goldens' routing dump and P9's rule, for the route-explained rows P1 and P3 print (#933).
include!("cuda_route_rule.rs");

/// The smallest bucket; every prompt and greedy sequence fits it.
const M: u32 = 512;
/// Blocks per seat: 8 · 64 = 512 tokens.
const KV_POOL: KvPoolConfig = KvPoolConfig { num_blocks: 16 };
const SEAT_A: (&[u32], u32) = (&[0, 1, 2, 3, 4, 5, 6, 7], 0);
const SEAT_B: (&[u32], u32) = (&[8, 9, 10, 11, 12, 13, 14, 15], 1);
/// P6's first chunk.
const CHUNK: usize = 128;

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

/// P1's rule (#859): a fixed floor on min cosine(CUDA, HF-f32), or at most `delta` below
/// HF-bf16's own min cosine on the same prompt, or (gemma-4) the distribution of a prompt's
/// per-row cosines rather than its single worst row: at each of `quantiles` (by
/// [`nearest_rank`]), CUDA's cosine at most HF-bf16's own measured scatter below HF-bf16's at the
/// same quantile, the scatter at most `cap` ([`hf_bf16_scatter`]; the user's decisions
/// 2026-10-27, option (a), and 2026-10-28, the distribution gate).
enum P1Rule {
    Floor(f64),
    BelowHfBf16(f64),
    ScatterQuantiles { cap: f64, quantiles: &'static [f64] },
}

/// The `q` quantile of `v` by nearest rank: its `ceil(q * n)`-th smallest value (1-based, at least
/// the first), so the 5th percentile of 15 rows is their minimum and of 159 rows their 8th
/// smallest.
fn nearest_rank(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let k = ((q * s.len() as f64).ceil() as usize).clamp(1, s.len());
    s[k - 1]
}

/// Whether a cosine `cos` against HF-f32 passes `rule`, `hf_bf16_cos` HF-bf16's own cosine on the
/// same rows: P1's and P8's cosine gate. A `ScatterQuantiles` rule gates by [`p1_margin`]'s
/// measured scatter (P1 applies it at each quantile, P8 to its one row).
fn p1_gate(rule: &P1Rule, cos: f64, hf_bf16_cos: f64) -> bool {
    match *rule {
        P1Rule::Floor(f) => cos >= f,
        P1Rule::BelowHfBf16(d) => cos >= hf_bf16_cos - d,
        P1Rule::ScatterQuantiles { .. } => cos >= hf_bf16_cos - p1_margin(),
    }
}

/// P3's rule for a top-1 disagreement outside a near-tie (#816): every one fails, or (gemma-4) at
/// most `per_prompt` of them pass when their rows are route-explained (the user's decision
/// 2026-10-27, option (a)).
enum P3Rule {
    Strict,
    RouteExplained { per_prompt: usize },
}

/// Whether P3 accepts one more top-1 disagreement outside a near-tie, on a row `explained` or not,
/// with `accepted` of them already accepted on this prompt.
fn p3_accepts(rule: &P3Rule, explained: bool, accepted: usize) -> bool {
    match *rule {
        P3Rule::Strict => false,
        P3Rule::RouteExplained { per_prompt } => explained && accepted < per_prompt,
    }
}

/// HF-bf16's own scatter on the goldens (the user's decision 2026-10-27, option (a)): over every
/// pair of prompts where one's ids are a prefix of the other's, the largest |cosine(HF-bf16 row,
/// HF-f32 row) of the one − the same of the other| over their shared rows. Causal attention makes
/// those rows the same computation, and HF-f32 agrees there; HF-bf16 moves with the batch shape
/// alone. Returns the scatter and where it is, `(value, shorter prompt, longer prompt, row)`, and
/// the largest |HF-f32 − HF-f32| over the same rows (which says the rows are the same). `None`
/// when no prompt is a prefix of another.
fn hf_bf16_scatter() -> Option<((f64, usize, usize, usize), f32)> {
    let ps = prompts();
    let mut best: Option<(f64, usize, usize, usize)> = None;
    let mut f32_diff = 0f32;
    for a in ps {
        for b in ps {
            let n = a.ids.len();
            if n >= b.ids.len() || b.ids[..n] != a.ids[..] {
                continue;
            }
            for row in 0..n {
                let at = |v: &[f32]| v[row * VOCAB..][..VOCAB].to_vec();
                let (fa, fb) = (at(&a.f32), at(&b.f32));
                let d = (cosine(&at(&a.bf16), &fa) - cosine(&at(&b.bf16), &fb)).abs();
                f32_diff = (fa.iter().zip(&fb)).fold(f32_diff, |m, (x, y)| m.max((x - y).abs()));
                if best.is_none_or(|(v, ..)| d > v) {
                    best = Some((d, a.p, b.p, row));
                }
            }
        }
    }
    best.map(|b| (b, f32_diff))
}

/// P1's margin below HF-bf16's min cosine under the binary's `P1`: its `delta`, or (a
/// `ScatterQuantiles` rule) HF-bf16's measured scatter, printed once with where it is, and
/// asserted at most the rule's `cap` so a regenerated golden cannot widen it silently. 0 for a
/// fixed floor.
fn p1_margin() -> f64 {
    static MARGIN: OnceLock<f64> = OnceLock::new();
    *MARGIN.get_or_init(|| match P1 {
        P1Rule::Floor(_) => 0.0,
        P1Rule::BelowHfBf16(d) => d,
        P1Rule::ScatterQuantiles { cap, .. } => {
            let ((scatter, short, long, row), f32_diff) = hf_bf16_scatter()
                .expect("P1's scatter rule needs a prompt that is a prefix of another");
            let ok = scatter <= cap;
            eprintln!(
                "P1 margin: HF-bf16's own scatter {scatter:.6} at row {row}, the largest \
                 |cosine(HF-bf16, f32)| difference between prompt {short} and prompt {long} over \
                 their shared prefix (max |HF-f32 - HF-f32| there {f32_diff:.4e}); need <= {cap} \
                 (the user's decision 2026-10-27, option (a)) {}",
                verdict(ok)
            );
            assert!(
                ok,
                "HF-bf16's measured scatter {scatter} is above the approved {cap}: the goldens \
                 changed; stop and report"
            );
            scatter
        }
    })
}

/// Both P1 rules gate as stated, on fixed numbers (no GPU, no goldens): the fixed floor on CUDA's
/// cosine alone, the HF-bf16 rule on CUDA's cosine against HF-bf16's minus delta. Each binary
/// runs one of the two (`P1`), so this is where the other is checked.
#[test]
fn p1_rules_gate_as_stated() {
    let floor = P1Rule::Floor(0.999);
    let below = P1Rule::BelowHfBf16(0.005);
    let cases = [
        (&floor, 0.9995, 0.9, true),
        (&floor, 0.9985, 1.0, false),
        (&below, 0.980474, 0.983452, true),
        (&below, 0.977, 0.983452, false),
    ];
    // `ScatterQuantiles` reads its margin from the goldens, so its gate is checked where they are
    // (`p1_to_p4_one_forward_per_prompt` prints it), not on fixed numbers; its cap and quantiles
    // are read here, and `nearest_rank` is checked on fixed numbers.
    let scatter = P1Rule::ScatterQuantiles {
        cap: 0.013,
        quantiles: &[0.05, 0.5],
    };
    assert!(matches!(
        scatter,
        P1Rule::ScatterQuantiles { cap, quantiles } if cap == 0.013 && quantiles == [0.05, 0.5]
    ));
    let fifteen: Vec<f64> = (1..=15).map(f64::from).rev().collect();
    let many: Vec<f64> = (1..=159).map(f64::from).collect();
    let mut pass = nearest_rank(&fifteen, 0.05) == 1.0
        && nearest_rank(&fifteen, 0.5) == 8.0
        && nearest_rank(&many, 0.05) == 8.0
        && nearest_rank(&many, 0.5) == 80.0
        && nearest_rank(&[3.0], 0.05) == 3.0;
    for (rule, cos, noise, want) in cases {
        let got = p1_gate(rule, cos, noise);
        pass &= got == want;
    }
    eprintln!("P1 rules on fixed numbers {}", verdict(pass));
    assert!(pass, "a P1 rule does not gate as stated");
}

/// Both P3 rules accept as stated (no GPU, no goldens): the strict rule nothing; the
/// route-explained rule one disagreement on a route-explained row per prompt, and none on a row
/// that is not, nor a second. Each binary runs one of the two (`P3`).
#[test]
fn p3_rules_accept_as_stated() {
    let strict = P3Rule::Strict;
    let one = P3Rule::RouteExplained { per_prompt: 1 };
    let cases = [
        (&strict, true, 0, false),
        (&strict, false, 0, false),
        (&one, true, 0, true),
        (&one, false, 0, false),
        (&one, true, 1, false),
    ];
    let pass = (cases.iter()).all(|&(rule, explained, accepted, want)| {
        p3_accepts(rule, explained, accepted) == want
    });
    eprintln!("P3 rules on fixed numbers {}", verdict(pass));
    assert!(pass, "a P3 rule does not accept as stated");
}

/// The H100's device memory a worker over every tape may hold after its forwards: 78.0 GiB of
/// the 79.6 GiB `cudaMemGetInfo` reports, about 1.5 GiB left for serve's KV budget (#859's
/// memory gate).
const PEAK_LIMIT: u64 = 78 << 30;

/// Bytes as GiB.
fn gib(bytes: u64) -> f64 {
    bytes as f64 / f64::from(1u32 << 30)
}

/// Little-endian byte pairs as the u16 halves `DeviceBuffer` holds.
fn le_halves(bytes: &[u8]) -> Vec<u16> {
    (bytes.as_chunks::<2>().0.iter())
        .map(|&p| u16::from_le_bytes(p))
        .collect()
}

fn goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(GOLDENS)
}

/// The goldens' `meta.json`.
fn meta() -> &'static serde_json::Value {
    static META: OnceLock<serde_json::Value> = OnceLock::new();
    META.get_or_init(|| {
        let path = goldens_dir().join("meta.json");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "{}: {e}. Make the goldens first: python3 -I \
                 crates/models/arch/tests/cuda_hf_reference.py --model {MODEL} --checkpoint \
                 <{CHECKPOINT} dir> --prompts crates/e2e/testdata/prompts.txt --out \
                 crates/models/arch/{GOLDENS}",
                path.display()
            )
        });
        serde_json::from_str(&text).expect("meta.json parses")
    })
}

fn checkpoint() -> PathBuf {
    PathBuf::from(
        meta()["checkpoint"]
            .as_str()
            .expect("meta.json names the checkpoint"),
    )
}

/// One prompt of the goldens: its number, ids, and the f32 and bf16 logits of every row.
struct Prompt {
    p: usize,
    ids: Vec<u32>,
    f32: Vec<f32>,
    bf16: Vec<f32>,
}

fn read_f32(name: &str, rows: usize) -> Vec<f32> {
    let path = goldens_dir().join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(bytes.len(), rows * VOCAB * 4, "{name}: [L, vocab] f32");
    (bytes.as_chunks::<4>().0.iter())
        .map(|&q| f32::from_le_bytes(q))
        .collect()
}

fn prompts() -> &'static [Prompt] {
    static PROMPTS: OnceLock<Vec<Prompt>> = OnceLock::new();
    PROMPTS.get_or_init(|| {
        (meta()["prompts"]
            .as_array()
            .expect("meta.json lists prompts")
            .iter())
        .map(|e| {
            let p = e["prompt"].as_u64().expect("a prompt number") as usize;
            let ids: Vec<u32> = (e["ids"].as_array().expect("ids").iter())
                .map(|v| v.as_u64().expect("an id") as u32)
                .collect();
            assert!(
                !ids.is_empty() && ids.len() + 8 <= M as usize,
                "prompt {p}: {} ids",
                ids.len()
            );
            Prompt {
                p,
                f32: read_f32(&format!("ref_f32_{p}.bin"), ids.len()),
                bf16: read_f32(&format!("ref_bf16_{p}.bin"), ids.len()),
                ids,
            }
        })
        .collect()
    })
}

/// The device, the loaded weights, the allocation count before loading, and the loader's time.
struct Rig {
    device: CudaDevice,
    weights: Vec<DeviceBuffer>,
    before: u64,
    load_secs: f64,
}

fn rig() -> Rig {
    let device = CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"));
    let before = device.allocations();
    let t = Instant::now();
    let weights = load_weights(&device, &checkpoint(), CUDA_TAPES, CUDA_WEIGHTS)
        .unwrap_or_else(|e| panic!("load_weights: {e}"));
    Rig {
        device,
        weights,
        before,
        load_secs: t.elapsed().as_secs_f64(),
    }
}

impl Rig {
    fn worker(&self) -> CudaWorker<'_> {
        CudaWorker::new(
            &self.device,
            CUDA_TAPES,
            &self.weights,
            Some(KV_POOL),
            GDN_POOL,
        )
        .unwrap_or_else(|e| panic!("CudaWorker::new: {e}"))
    }
}

fn bucket() -> usize {
    (CUDA_TAPES.iter().position(|t| t.bucket_m == M)).expect("the m=512 tape")
}

/// One forward of `ids` (the real rows) at `prefix_len` in `seat`: the logits of the real rows,
/// widened to f32.
fn forward(w: &mut CudaWorker<'_>, ids: &[u32], prefix_len: u32, seat: (&[u32], u32)) -> Vec<f32> {
    let tape = &CUDA_TAPES[bucket()];
    let q = ids.len() as u32;
    let mut padded = vec![0u32; M as usize];
    padded[..ids.len()].copy_from_slice(ids);
    let positions: Vec<u32> = (prefix_len..prefix_len + M).collect();
    let kv = KvSequence {
        block_table: seat.0,
        prefix_len,
        q_len: q,
    };
    let gdn = GDN_POOL.map(|_| GdnSequence {
        slot: seat.1,
        q_len: q,
        fresh: prefix_len == 0,
    });
    let mut out = vec![0u16; tape.result.bytes as usize / 2];
    w.forward(
        bucket(),
        Input::TokenIds(&padded),
        &positions,
        Some(&kv),
        gdn.as_ref(),
        &mut out,
    )
    .unwrap_or_else(|e| panic!("forward: {e}"));
    (out[..ids.len() * VOCAB].iter())
        .map(|&b| bf16::from_bits(b).to_f32())
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa.sqrt() * bb.sqrt())
}

/// RMS and max of `a − b`.
fn error(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut sq, mut max) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = (f64::from(x) - f64::from(y)).abs();
        sq += d * d;
        max = max.max(d);
    }
    ((sq / a.len() as f64).sqrt(), max)
}

/// The indices of `row`'s `k` highest values, highest first (lowest index on a tie).
fn top_k(row: &[f32], k: usize) -> Vec<usize> {
    let mut ix: Vec<usize> = (0..row.len()).collect();
    ix.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
    ix.truncate(k);
    ix
}

fn rows(v: &[f32]) -> impl Iterator<Item = &[f32]> {
    v.chunks(VOCAB)
}

/// The m=512 tape cut just after each `MoeRoute` step, its result that layer's `topk_idx`, the
/// MoE layers' picks per token (the route test's capture, `cuda_route_parity_body.rs`), and the
/// picks per token; empty for a dense model, whose tape has no routing.
fn route_cut_tapes() -> (&'static [scratchy_target_cuda::tape::CudaTape], usize) {
    static CUTS: OnceLock<(&'static [scratchy_target_cuda::tape::CudaTape], usize)> = OnceLock::new();
    *CUTS.get_or_init(|| {
        let full = &CUDA_TAPES[bucket()];
        let mut top_k = 0;
        let cuts: Vec<scratchy_target_cuda::tape::CudaTape> = (full.steps.iter().enumerate())
            .filter_map(|(i, s)| match *s {
                scratchy_target_cuda::step::CudaStep::MoeRoute(MoeRouteBf16 {
                    top_k: k, ..
                }) => {
                    let moe = full.moe.expect("a routing tape has a MoE scratch");
                    top_k = k as usize;
                    Some(scratchy_target_cuda::tape::CudaTape {
                        steps: Box::leak(full.steps[..=i].to_vec().into_boxed_slice()),
                        result: moe.topk_idx,
                        ..*full
                    })
                }
                _ => None,
            })
            .collect();
        (Box::leak(cuts.into_boxed_slice()), top_k)
    })
}

/// One row's expert picks against HF-f32's, walked over every layer under P9's rule: the layers
/// where CUDA's pick differs, and whether P9 accepts every one of those differences.
struct RowRoute {
    differs: Vec<usize>,
    accepted: bool,
}

impl RowRoute {
    /// Route-explained (printed only, never gated): CUDA's pick differs from
    /// HF-f32's at one or more layers, and P9's rule accepts every one of those differences.
    fn explained(&self) -> bool {
        !self.differs.is_empty() && self.accepted
    }
}

/// Every row of a prompt under P9's rule, from CUDA's picks per layer (`cuda[layer][row]`) and
/// the prompt's routing dump: per row, the layers where its pick differs and whether every such
/// difference is accepted. A difference is accepted exactly as `p9_expert_picks_match_hf_f32`
/// accepts it: inside a router tie (layers ≥ 1, and layer 0); outside one, at a layer ≥ 1, when
/// the same token flipped at an earlier layer or HF-bf16 flips at least as many tokens as CUDA in
/// that layer ([`layer_rule`]'s excuses, per row).
fn row_routes(cuda: &[Vec<BTreeSet<i32>>], route: &Route) -> Vec<RowRoute> {
    let n = route.rows;
    let mut rows: Vec<RowRoute> = (0..n)
        .map(|_| RowRoute {
            differs: Vec::new(),
            accepted: true,
        })
        .collect();
    let mut flipped = vec![false; n];
    for (layer, picks) in cuda.iter().enumerate() {
        let tau = route.tau(layer);
        let differs: Vec<bool> = (0..n)
            .map(|row| picks[row] != route.f32.set(n, layer, row))
            .collect();
        let flips = differs.iter().filter(|&&d| d).count();
        let bf16_flips = (0..n)
            .filter(|&row| route.bf16.set(n, layer, row) != route.f32.set(n, layer, row))
            .count();
        for row in (0..n).filter(|&row| differs[row]) {
            let (tie, _) = route.tie(layer, row, tau);
            // The row's own difference under `layer_rule`: inside a tie it is no outside flip;
            // outside one it is excused by its earlier flip, or else counted not excused.
            let outside = usize::from(!tie);
            let (excused, unexcused) = match (tie, flipped[row]) {
                (true, _) => (0, 0),
                (false, true) => (1, 0),
                (false, false) => (0, 1),
            };
            let (ok, ..) = layer_rule(layer, (outside, excused, unexcused), (flips, bf16_flips));
            rows[row].differs.push(layer);
            rows[row].accepted &= ok;
        }
        for (row, &d) in differs.iter().enumerate() {
            flipped[row] |= d;
        }
    }
    rows
}

/// Per prompt, every row under P9's rule (printed by P1 and P3, never gated): CUDA's
/// picks at each MoE layer from one fresh forward of the cut tape, against the prompt's routing
/// dump `route_<p>.bin`. Empty for a dense model, which routes nothing, so the rule changes
/// nothing there.
fn prompt_routes(r: &Rig) -> Vec<Vec<RowRoute>> {
    let (cuts, top_k) = route_cut_tapes();
    if cuts.is_empty() {
        return prompts().iter().map(|_| Vec::new()).collect();
    }
    let mut w = CudaWorker::new(&r.device, cuts, &r.weights, Some(KV_POOL), GDN_POOL)
        .unwrap_or_else(|e| panic!("CudaWorker::new over the {} route cuts: {e}", cuts.len()));
    (prompts().iter())
        .map(|p| {
            let n = p.ids.len();
            let route = read_route_of(&format!("route_{}.bin", p.p), n, (cuts.len(), top_k));
            let picks: Vec<Vec<BTreeSet<i32>>> = (0..cuts.len())
                .map(|layer| {
                    let out = run_cut(&mut w, (cuts, layer), &p.ids);
                    let idx: Vec<i32> = (out.as_chunks::<2>().0.iter())
                        .map(|&[lo, hi]| (u32::from(lo) | u32::from(hi) << 16) as i32)
                        .collect();
                    (idx.chunks(top_k).take(n))
                        .map(|row| row.iter().copied().collect())
                        .collect()
                })
                .collect();
            row_routes(&picks, &route)
        })
        .collect()
}

/// One fresh forward of `ids` through tape `b` of `tapes` (a cut worker's own list) in
/// [`SEAT_A`]: the whole result, as u16 halves.
fn run_cut(
    w: &mut CudaWorker<'_>,
    (tapes, b): (&[scratchy_target_cuda::tape::CudaTape], usize),
    ids: &[u32],
) -> Vec<u16> {
    let tape = &tapes[b];
    let q = ids.len() as u32;
    let mut padded = vec![0u32; M as usize];
    padded[..ids.len()].copy_from_slice(ids);
    let positions: Vec<u32> = (0..M).collect();
    let kv = KvSequence {
        block_table: SEAT_A.0,
        prefix_len: 0,
        q_len: q,
    };
    let gdn = GDN_POOL.map(|_| GdnSequence {
        slot: SEAT_A.1,
        q_len: q,
        fresh: true,
    });
    let mut out = vec![0u16; tape.result.bytes as usize / 2];
    w.forward(
        b,
        Input::TokenIds(&padded),
        &positions,
        Some(&kv),
        gdn.as_ref(),
        &mut out,
    )
    .unwrap_or_else(|e| panic!("the forward cut after MoE layer {b}'s routing: {e}"));
    out
}

/// The prompts' CUDA logits, one forward each from fresh state: computed once per test binary.
fn cuda_logits(r: &Rig) -> Vec<Vec<f32>> {
    let mut w = r.worker();
    (prompts().iter())
        .map(|p| forward(&mut w, &p.ids, 0, SEAT_A))
        .collect()
}

#[test]
fn load_checks() {
    let r = rig();
    let loaded = r.device.allocations() - r.before;
    let w = r.worker();
    let worker = r.device.allocations() - r.before - loaded;
    drop(w);
    let nonzero = (r.weights.iter()).filter(|b| b.bytes() > 0).count();
    let mut pass = loaded == WEIGHTS && worker == WORKER_ALLOCATIONS && nonzero == WEIGHTS as usize;
    eprintln!(
        "load: allocations weights={loaded} (expect {WEIGHTS}) worker={worker} (expect \
         {WORKER_ALLOCATIONS}) \
         nonzero_buffers={nonzero} load_time={:.2}s (from opening the checkpoint to the last \
         upload) {}",
        r.load_secs,
        verdict(pass)
    );
    let sizes_ok = (r.weights.iter().enumerate()).all(|(ix, b)| {
        let declared = CUDA_TAPES
            .iter()
            .map(|t| t.weight_bytes[ix])
            .max()
            .unwrap_or(0);
        b.bytes() == declared
    });
    eprintln!(
        "load: every buffer's bytes equal its weight_bytes {}",
        verdict(sizes_ok)
    );
    pass &= sizes_ok;
    // The checkpoint's own bytes, read through the same reader.
    let mut ck = open_checkpoint(&checkpoint()).unwrap_or_else(|e| panic!("{e}"));
    let Some(CudaWeight::Bf16 { tensor, .. }) = CUDA_WEIGHTS[0] else {
        panic!("weight 0 is the embedding table");
    };
    let (table, ..) = ck.take_to_cpu_bytes(tensor).expect("the embedding table");
    let device_table = r.weights[0].download().expect("download the table");
    for row in [0, VOCAB - 1] {
        let want: Vec<u16> = (table[row * HIDDEN * 2..][..HIDDEN * 2]
            .as_chunks::<2>()
            .0
            .iter())
        .map(|&p| u16::from_le_bytes(p))
        .collect();
        let ok = device_table[row * HIDDEN..][..HIDDEN] == want[..];
        eprintln!(
            "load: embed row {row} equals the checkpoint's {}",
            verdict(ok)
        );
        pass &= ok;
    }
    // An untied lm_head is its own `[vocab, hidden]` tensor: its first and last rows too.
    if let Some(lm_head) = UNTIED_LM_HEAD {
        let ix = (CUDA_WEIGHTS.iter())
            .position(|w| matches!(w, Some(CudaWeight::Bf16 { tensor, .. }) if *tensor == lm_head))
            .unwrap_or_else(|| panic!("no entry for {lm_head}"));
        let (table, ..) = ck.take_to_cpu_bytes(lm_head).expect("the lm_head");
        let device_table = r.weights[ix].download().expect("download the lm_head");
        for row in [0, VOCAB - 1] {
            let want: Vec<u16> = (table[row * HIDDEN * 2..][..HIDDEN * 2]
                .as_chunks::<2>()
                .0
                .iter())
            .map(|&p| u16::from_le_bytes(p))
            .collect();
            let ok = device_table[row * HIDDEN..][..HIDDEN] == want[..];
            eprintln!(
                "load: weight {ix} {lm_head} row {row} equals the checkpoint's {}",
                verdict(ok)
            );
            pass &= ok;
        }
    }
    // A model with GDN layers (`GDN_POOL` is `Some`): its first GDN bundle, unpacked. A model with
    // none has no bundle to check.
    if GDN_POOL.is_some() {
        let (ix, bundle) = (CUDA_WEIGHTS.iter().enumerate())
            .find_map(|(ix, w)| match *w {
                Some(b @ CudaWeight::GdnBundle { .. }) => Some((ix, b)),
                _ => None,
            })
            .expect("a GDN bundle");
        let CudaWeight::GdnBundle {
            conv1d,
            a_log,
            dt_bias,
            norm,
            conv_dim,
            conv_kernel,
            nv,
            head_v,
        } = bundle
        else {
            unreachable!("matched a bundle");
        };
        let layout = gdn_bundle(conv_dim, conv_kernel, nv, head_v).expect("the bundle fits");
        let halves = r.weights[ix].download().expect("download the bundle");
        let device: Vec<f32> = (halves.as_chunks::<2>().0.iter())
            .map(|&[lo, hi]| f32::from_bits(u32::from(lo) | u32::from(hi) << 16))
            .collect();
        for (tensor, at) in [
            (conv1d, layout.conv1d),
            (a_log, layout.a_log),
            (dt_bias, layout.dt_bias),
            (norm, layout.norm),
        ] {
            let want = ck
                .tensor_to_f32(tensor)
                .unwrap_or_else(|| panic!("{tensor}"));
            let got = &device[at as usize / 4..][..want.len()];
            let ok = (got.iter().zip(&want)).all(|(g, w)| g.to_bits() == w.to_bits());
            eprintln!(
                "load: weight {ix} {tensor} ({} values at {at}) equals the checkpoint's widened to f32 {}",
                want.len(),
                verdict(ok)
            );
            pass &= ok;
        }
    }
    // A MoE model's first and last MoE bundles (qwen3.6-35b-a3b's layers 0 and 39; none in a
    // dense model): the router whole at byte 0, experts 0 and `experts − 1`'s first `gate_up`
    // and `down` rows, and the layer's shared-expert gate, each against the checkpoint's bytes.
    let moe: Vec<(usize, CudaWeight)> = (CUDA_WEIGHTS.iter().enumerate())
        .filter_map(|(ix, w)| match *w {
            Some(b @ CudaWeight::MoeBundle { .. }) => Some((ix, b)),
            _ => None,
        })
        .collect();
    let ends = (moe.iter().enumerate()).filter(|&(n, _)| n == 0 || n + 1 == moe.len());
    for (_, &(ix, bundle)) in ends {
        let CudaWeight::MoeBundle {
            router,
            gate_up,
            down,
            experts,
            hidden,
            inter,
        } = bundle
        else {
            unreachable!("matched a MoE bundle");
        };
        let layout = scratchy_target_cuda::step::moe_bundle(experts, hidden, inter)
            .expect("the MoE bundle fits");
        let device = r.weights[ix].download().expect("download the MoE bundle");
        let (e, h, i) = (experts as usize, hidden as usize, inter as usize);
        let (want, ..) = ck.take_to_cpu_bytes(router).expect("the router");
        let at = layout.router as usize / 2;
        let ok = device[at..][..e * h] == le_halves(&want)[..];
        eprintln!(
            "load: weight {ix} {router} [{e}, {h}] at byte {} equals the checkpoint's {}",
            layout.router,
            verdict(ok)
        );
        pass &= ok;
        for (tensor, base, per_expert, row) in [
            (gate_up, layout.gate_up, 2 * i * h, h),
            (down, layout.down, h * i, i),
        ] {
            let (want, ..) = ck.take_to_cpu_bytes(tensor).expect("an expert tensor");
            for expert in [0, e - 1] {
                let from = expert * per_expert;
                let at = base as usize / 2 + from;
                let ok = device[at..][..row] == le_halves(&want[from * 2..][..row * 2])[..];
                eprintln!(
                    "load: weight {ix} {tensor} expert {expert} row 0 ({row} values) equals the \
                     checkpoint's {}",
                    verdict(ok)
                );
                pass &= ok;
            }
        }
        let gate = router.replace(".mlp.gate.weight", ".mlp.shared_expert_gate.weight");
        let gix = (CUDA_WEIGHTS.iter())
            .position(|w| matches!(w, Some(CudaWeight::Bf16 { tensor, .. }) if *tensor == gate))
            .unwrap_or_else(|| panic!("no entry for {gate}"));
        let (want, ..) = ck.take_to_cpu_bytes(&gate).expect("the shared-expert gate");
        let got = r.weights[gix].download().expect("download the shared-expert gate");
        let ok = got == le_halves(&want);
        eprintln!(
            "load: weight {gix} {gate} ({} values) equals the checkpoint's {}",
            got.len(),
            verdict(ok)
        );
        pass &= ok;
    }
    // Gemma's first and last expert banks (layers 0 and 29; none in a qwen model): experts 0 and
    // `experts − 1`'s first `gate_up` and `down` rows, and the layer's router (`router.proj`,
    // `router.scale`, `router.per_expert_scale`) and `layer_scalar`, each whole, against the
    // checkpoint's bytes (#933).
    let banks: Vec<(usize, CudaWeight)> = (CUDA_WEIGHTS.iter().enumerate())
        .filter_map(|(ix, w)| match *w {
            Some(b @ CudaWeight::ExpertBank { .. }) => Some((ix, b)),
            Some(
                CudaWeight::Bf16 { .. }
                | CudaWeight::GdnBundle { .. }
                | CudaWeight::MoeBundle { .. },
            )
            | None => None,
        })
        .collect();
    let ends = (banks.iter().enumerate()).filter(|&(n, _)| n == 0 || n + 1 == banks.len());
    for (_, &(ix, bank)) in ends {
        let CudaWeight::ExpertBank {
            gate_up,
            down,
            experts,
            hidden,
            inter,
        } = bank
        else {
            unreachable!("matched an expert bank");
        };
        let layout = scratchy_target_cuda::step::expert_bank(experts, hidden, inter)
            .expect("the expert bank fits");
        let device = r.weights[ix].download().expect("download the expert bank");
        let (e, h, i) = (experts as usize, hidden as usize, inter as usize);
        for (tensor, base, per_expert, row) in [
            (gate_up, layout.gate_up, 2 * i * h, h),
            (down, layout.down, h * i, i),
        ] {
            let (want, ..) = ck.take_to_cpu_bytes(tensor).expect("an expert tensor");
            for expert in [0, e - 1] {
                let from = expert * per_expert;
                let at = base as usize / 2 + from;
                let ok = device[at..][..row] == le_halves(&want[from * 2..][..row * 2])[..];
                eprintln!(
                    "load: weight {ix} {tensor} expert {expert} row 0 ({row} values) equals the \
                     checkpoint's {}",
                    verdict(ok)
                );
                pass &= ok;
            }
        }
        let layer = gate_up
            .strip_suffix(".experts.gate_up_proj")
            .unwrap_or_else(|| panic!("{gate_up} is a layer's experts.gate_up_proj"));
        for suffix in [
            ".router.proj.weight",
            ".router.scale",
            ".router.per_expert_scale",
            ".layer_scalar",
        ] {
            let name = format!("{layer}{suffix}");
            let gix = (CUDA_WEIGHTS.iter())
                .position(|w| matches!(w, Some(CudaWeight::Bf16 { tensor, .. }) if *tensor == name))
                .unwrap_or_else(|| panic!("no entry for {name}"));
            let (want, ..) = ck.take_to_cpu_bytes(&name).expect("a router tensor");
            let got = r.weights[gix].download().expect("download a router tensor");
            let ok = got == le_halves(&want);
            eprintln!(
                "load: weight {gix} {name} ({} values) equals the checkpoint's {}",
                got.len(),
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(pass, "a load check failed");
}

/// `nvidia-smi`'s memory line for the memory gate's readings, or why there is none. Printed beside
/// `cudaMemGetInfo`, information only.
fn nvidia_smi() -> String {
    std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader",
        ])
        .output()
        .map_or_else(
            |e| format!("nvidia-smi unavailable: {e}"),
            |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
        )
}

/// The memory gate (#859), run first and alone: `cudaMemGetInfo` and `nvidia-smi` after the
/// device opens, after `load_weights`, after a worker over every tape, and after one forward of
/// the largest bucket's every row and one decode. Pass: `WEIGHTS` + `WORKER_ALLOCATIONS`
/// allocations, every buffer its `weight_bytes`, every bucket in the worker, and a peak (total −
/// free after the forwards) at most `PEAK_LIMIT`. The KV pool is serve's cap, one sequence's
/// block table plus block 0 (`CUDA_MAX_BLOCKS_PER_SEQ` + 1); the GDN pool is the rig's
/// (`GDN_POOL`, none for a model with no GDN layer). Reads only the goldens' `checkpoint`.
#[test]
fn memory_fits() {
    let device = CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"));
    let reading = |when: &str| {
        let (free, total) = device.mem_info().unwrap_or_else(|e| panic!("mem_info: {e}"));
        let (free, total) = (free as u64, total as u64);
        eprintln!(
            "memory {when}: used {:.3} GiB free {:.3} GiB total {:.3} GiB (cudaMemGetInfo); \
             nvidia-smi used, total: {}",
            gib(total - free),
            gib(free),
            gib(total),
            nvidia_smi()
        );
        (free, total)
    };
    reading("after CudaDevice::open");
    let before = device.allocations();
    let t = Instant::now();
    let weights = load_weights(&device, &checkpoint(), CUDA_TAPES, CUDA_WEIGHTS)
        .unwrap_or_else(|e| panic!("load_weights: {e}"));
    let load_secs = t.elapsed().as_secs_f64();
    let loaded = device.allocations() - before;
    reading("after load_weights");
    let sizes_ok = (weights.iter().enumerate()).all(|(ix, b)| {
        b.bytes() == (CUDA_TAPES.iter().map(|t| t.weight_bytes[ix]).max()).unwrap_or(0)
    });
    let blocks = scratchy_target_cuda::step::CUDA_MAX_BLOCKS_PER_SEQ + 1;
    let mut w = CudaWorker::new(
        &device,
        CUDA_TAPES,
        &weights,
        Some(KvPoolConfig { num_blocks: blocks }),
        GDN_POOL,
    )
    .unwrap_or_else(|e| panic!("CudaWorker::new: {e}"));
    let worker = device.allocations() - before - loaded;
    reading("after CudaWorker::new");
    let ms: Vec<u32> = CUDA_TAPES.iter().map(|t| t.bucket_m).collect();
    let buckets_ok = ms == [1, 512, 1024, 2048, 4096];
    let (b, tape) = (CUDA_TAPES.iter().enumerate())
        .max_by_key(|(_, t)| t.bucket_m)
        .expect("a tape");
    let rows = tape.bucket_m;
    let ids: Vec<u32> = (0..rows).map(|i| (i * 7919) % VOCAB as u32).collect();
    let positions: Vec<u32> = (0..rows).collect();
    let table: Vec<u32> = (0..blocks - 1).collect();
    let kv = KvSequence {
        block_table: &table,
        prefix_len: 0,
        q_len: rows,
    };
    let gdn = GDN_POOL.map(|_| GdnSequence {
        slot: 0,
        q_len: rows,
        fresh: true,
    });
    let mut out = vec![0u16; tape.result.bytes as usize / 2];
    w.forward(
        b,
        Input::TokenIds(&ids),
        &positions,
        Some(&kv),
        gdn.as_ref(),
        &mut out,
    )
    .unwrap_or_else(|e| panic!("the m={rows} forward: {e}"));
    let row = decode(&mut w, 7, rows, (&table[..], 0));
    let non_finite = (out.iter().map(|&b| bf16::from_bits(b).to_f32()))
        .chain(row.iter().copied())
        .filter(|v| !v.is_finite())
        .count();
    let (free, total) = reading(&format!("after one m={rows} forward and one m=1 decode"));
    let peak = total - free;
    let ok = loaded == WEIGHTS
        && worker == WORKER_ALLOCATIONS
        && sizes_ok
        && buckets_ok
        && peak <= PEAK_LIMIT;
    eprintln!(
        "memory: allocations weights={loaded} (expect {WEIGHTS}) worker={worker} (expect \
         {WORKER_ALLOCATIONS}) \
         every buffer its weight_bytes={sizes_ok} buckets {ms:?} (expect all five) arena={} KV \
         pool={:?} ({blocks} blocks) GDN pool={:?} load_time={load_secs:.2}s non-finite \
         logits={non_finite} (information only)",
        w.arena_bytes(),
        w.kv_pool_bytes(),
        w.gdn_pool_bytes()
    );
    eprintln!(
        "memory: peak {:.3} GiB of {:.3} GiB (headroom {:.3} GiB), need <= {:.1} GiB (#859's \
         gate, about 1.5 GiB left for serve's KV budget) {}",
        gib(peak),
        gib(total),
        gib(total - peak),
        gib(PEAK_LIMIT),
        verdict(ok)
    );
    assert!(ok, "the memory gate failed: see the lines above");
}

#[test]
fn p1_to_p4_one_forward_per_prompt() {
    let r = rig();
    let cuda = cuda_logits(&r);
    let routes = prompt_routes(&r);
    let (mut p1, mut p2, mut p3, mut p4) = (true, true, true, true);
    let (mut agree, mut total) = (0usize, 0usize);
    for ((p, got), route) in prompts().iter().zip(&cuda).zip(&routes) {
        let n = p.ids.len();
        // A route-explained row: its CUDA picks differ from HF-f32's at some layer, and P9's rule
        // accepts every such difference. Printed only, never gated (#933's diagnosis; the
        // excusal 712dd08e made of it excused nearly every row and was backed out). Empty for a
        // dense model.
        let explained = |row: usize| route.get(row).is_some_and(RowRoute::explained);
        let excused: Vec<usize> = (0..n).filter(|&row| explained(row)).collect();
        if !route.is_empty() {
            eprintln!(
                "P1/P3 prompt {}: {} of {n} rows route-explained (picks differ from HF-f32's \
                 only where P9 accepts it) {:?} (information only, not gated)",
                p.p,
                excused.len(),
                excused
            );
        }
        for &row in &excused {
            let (c, f, b) = (
                &got[row * VOCAB..][..VOCAB],
                &p.f32[row * VOCAB..][..VOCAB],
                &p.bf16[row * VOCAB..][..VOCAB],
            );
            eprintln!(
                "P1/P3 prompt {} row {row}: route-explained, cosine(CUDA, f32)={:.6} \
                 cosine(HF-bf16, f32)={:.6}, picks differ at layers {:?} (information only: P1 and \
                 P3 gate it)",
                p.p,
                cosine(c, f),
                cosine(b, f),
                route[row].differs
            );
        }
        // P1.
        let min_cos = (rows(got).zip(rows(&p.f32)))
            .map(|(a, b)| cosine(a, b))
            .fold(f64::INFINITY, f64::min);
        let noise_cos = (rows(&p.bf16).zip(rows(&p.f32)))
            .map(|(a, b)| cosine(a, b))
            .fold(f64::INFINITY, f64::min);
        let ok1 = match P1 {
            P1Rule::Floor(f) => {
                let ok1 = p1_gate(&P1, min_cos, noise_cos);
                eprintln!(
                    "P1 prompt {}: {n} rows, min cosine(CUDA, f32) = {min_cos:.6} (HF-bf16 \
                     {noise_cos:.6}) (need >= {f}) {}",
                    p.p,
                    verdict(ok1)
                );
                if noise_cos < f {
                    eprintln!(
                        "P1 prompt {}: HF-bf16's own min cosine is below {f}: the floor is wrong, \
                         not the kernels. STOP AND REPORT",
                        p.p
                    );
                }
                ok1
            }
            P1Rule::BelowHfBf16(d) => {
                let ok1 = p1_gate(&P1, min_cos, noise_cos);
                eprintln!(
                    "P1 prompt {}: {n} rows, min cosine(CUDA, f32) = {min_cos:.6} HF-bf16 \
                     {noise_cos:.6} CUDA - HF-bf16 = {:+.6} (need >= HF-bf16 - {d}) {}",
                    p.p,
                    min_cos - noise_cos,
                    verdict(ok1)
                );
                ok1
            }
            P1Rule::ScatterQuantiles { quantiles, .. } => {
                let d = p1_margin();
                let cuda: Vec<f64> = (rows(got).zip(rows(&p.f32)))
                    .map(|(a, b)| cosine(a, b))
                    .collect();
                let hf: Vec<f64> = (rows(&p.bf16).zip(rows(&p.f32)))
                    .map(|(a, b)| cosine(a, b))
                    .collect();
                let mut ok1 = true;
                for &q in quantiles {
                    let (c, h) = (nearest_rank(&cuda, q), nearest_rank(&hf, q));
                    let ok = p1_gate(&P1, c, h);
                    ok1 &= ok;
                    eprintln!(
                        "P1 prompt {}: {n} rows, cosine(CUDA, f32) at the {:.0}th percentile = \
                         {c:.6} HF-bf16 {h:.6} CUDA - HF-bf16 = {:+.6} (need >= HF-bf16 - {d:.6}, \
                         HF-bf16's measured scatter) {}",
                        p.p,
                        q * 100.0,
                        c - h,
                        verdict(ok)
                    );
                }
                eprintln!(
                    "P1 prompt {}: min cosine(CUDA, f32) = {min_cos:.6} HF-bf16 {noise_cos:.6} \
                     (information only: the distribution gate gates the percentiles)",
                    p.p
                );
                ok1
            }
        };
        p1 &= ok1;
        // P2.
        let (rms, max) = error(got, &p.f32);
        let (noise_rms, noise_max) = error(&p.bf16, &p.f32);
        let ok2 = rms <= 2.0 * noise_rms && max <= 2.0 * noise_max;
        eprintln!(
            "P2 prompt {}: RMS(CUDA-f32)={rms:.6e} RMS(HF-bf16-f32)={noise_rms:.6e} \
             max|CUDA-f32|={max:.6e} max|HF-bf16-f32|={noise_max:.6e} (need <= 2x each) {}",
            p.p,
            verdict(ok2)
        );
        p2 &= ok2;
        // P3.
        // `ties`: the rows whose f32 margin is at most twice their noise, counted whether or not
        // CUDA agrees; `near`: those of them CUDA disagrees on.
        let (mut ties, mut near, mut wrong, mut last_ok) = (Vec::new(), 0usize, Vec::new(), true);
        // Top-1 disagreements outside near-ties on route-explained rows, printed; and those P3
        // accepts under the binary's `P3`, the rest in `wrong`, which gates.
        let (mut routed, mut accepted) = (Vec::new(), Vec::new());
        for (row, ((c, f), b)) in rows(got).zip(rows(&p.f32)).zip(rows(&p.bf16)).enumerate() {
            let top = top_k(f, 2);
            let margin = f64::from(f[top[0]] - f[top[1]]);
            let noise = error(b, f).1;
            let same = top_k(c, 1)[0] == top[0];
            total += 1;
            agree += usize::from(same);
            let tie = margin <= 2.0 * noise;
            if tie {
                ties.push(row);
            }
            if !same && tie {
                near += 1;
            }
            if !same && !tie {
                if explained(row) {
                    routed.push(row);
                }
                // At most the binary's `P3` per prompt, on a route-explained row (the user's
                // decision 2026-10-27, option (a)); every other one gates.
                if p3_accepts(&P3, explained(row), accepted.len()) {
                    accepted.push(row);
                } else {
                    wrong.push(row);
                }
            }
            if row == n - 1 && !same {
                last_ok = tie;
                eprintln!(
                    "P3 prompt {}: last row disagrees, f32 margin {margin:.4e} vs 2x noise \
                     {:.4e}: {}",
                    p.p,
                    2.0 * noise,
                    match (tie, explained(row)) {
                        (true, _) => "near-tie",
                        (false, true) => "NOT a near-tie, route-explained",
                        (false, false) => "NOT a near-tie",
                    }
                );
            }
        }
        let ok3 = wrong.is_empty() && last_ok;
        for &row in &accepted {
            eprintln!(
                "P3 prompt {}: row {row}'s top-1 disagreement outside a near-tie accepted, its row \
                 route-explained (at most one per prompt; the user's decision 2026-10-27, option \
                 (a))",
                p.p
            );
        }
        if !route.is_empty() {
            eprintln!(
                "P3 prompt {}: of the top-1 disagreements outside near-ties, route-explained={} \
                 {:?} (information only, not gated)",
                p.p,
                routed.len(),
                routed
            );
        }
        eprintln!(
            "P3 prompt {}: top-1 disagreements outside near-ties={} {:?}, near-tie rows={} {:?} \
             (of them CUDA disagrees on {near}) {}",
            p.p,
            wrong.len(),
            wrong,
            ties.len(),
            ties,
            verdict(ok3)
        );
        p3 &= ok3;
        // P4.
        let last = n - 1;
        let c5: BTreeSet<usize> = top_k(&got[last * VOCAB..][..VOCAB], 5)
            .into_iter()
            .collect();
        let f5: BTreeSet<usize> = top_k(&p.f32[last * VOCAB..][..VOCAB], 5)
            .into_iter()
            .collect();
        let shared = c5.intersection(&f5).count();
        let ok4 = shared >= 4;
        eprintln!(
            "P4 prompt {}: last-row top-5 CUDA {c5:?} f32 {f5:?}, shared {shared} (need >= 4) {}",
            p.p,
            verdict(ok4)
        );
        p4 &= ok4;
    }
    // Information only, not a gate: P3 is the per-prompt rule above, no top-1 disagreement
    // outside a near-tie. The 99% floor it replaced came from 0.8b runs; qwen3.5-9b agreed on
    // 245/249 = 0.9839 rows, every disagreement a near-tie (PR #12).
    eprintln!(
        "P3 all prompts: top-1 agreement {agree}/{total} = {:.4} (information only)",
        agree as f64 / total as f64
    );
    eprintln!(
        "P1-P4 rules and their numbers' sources: P1 {}; P2 <= 2x \
         HF-bf16's RMS and max per prompt (this run's HF-bf16, #816); P3 no top-1 disagreement \
         outside a near-tie, margin <= 2x the row's max |HF-bf16 - f32| (this run's HF-bf16, the \
         user's 2026-10-23 rule); P4 >= 4 of 5 shared on the last row (#816)",
        match P1 {
            P1Rule::Floor(f) => format!("min cosine >= {f}, gated (0.8b runs, #816)"),
            P1Rule::BelowHfBf16(d) => format!(
                "min cosine >= HF-bf16's - {d} per prompt, gated (the user's 2026-10-26 rule, \
                 #859)"
            ),
            P1Rule::ScatterQuantiles { cap, quantiles } => format!(
                "cosine at each percentile {:?} >= HF-bf16's at the same percentile - its \
                 measured scatter {:.6} (<= {cap}) per prompt, gated (the user's decisions \
                 2026-10-27, option (a), and 2026-10-28, the distribution gate)",
                quantiles.iter().map(|q| q * 100.0).collect::<Vec<_>>(),
                p1_margin()
            ),
        }
    );
    if routes.iter().any(|r| !r.is_empty()) {
        eprintln!(
            "P1/P3 route-explained rows (a row whose CUDA picks differ from HF-f32's at one or \
             more layers, every difference accepted by P9's rule) are printed only: P1 and P3 \
             gate every row"
        );
    }
    assert!(p1 && p2 && p3 && p4, "P1 {p1} P2 {p2} P3 {p3} P4 {p4}");
}

#[test]
fn p5_greedy_matches_hf_f32() {
    let r = rig();
    let mut w = r.worker();
    let mut near_ties = 0;
    let mut pass = true;
    for p in prompts().iter().take(5) {
        let path = goldens_dir().join(format!("greedy_{}.json", p.p));
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let golden: serde_json::Value = serde_json::from_str(&text).expect("greedy json");
        let steps = golden["steps"].as_array().expect("steps");
        let want: Vec<u32> = (steps.iter())
            .map(|s| s["token"].as_u64().expect("a token") as u32)
            .collect();
        let mut seq = p.ids.clone();
        let mut got = Vec::new();
        for _ in 0..want.len() {
            let logits = forward(&mut w, &seq, 0, SEAT_A);
            let last = &logits[(seq.len() - 1) * VOCAB..];
            let token = top_k(last, 1)[0] as u32;
            got.push(token);
            seq.push(token);
        }
        let first = got.iter().zip(&want).position(|(g, w)| g != w);
        let ok = match first {
            None => true,
            Some(step) => {
                let margin = steps[step]["margin"].as_f64().expect("a margin");
                let noise = steps[step]["bf16_max_err"].as_f64().expect("a noise");
                let tie = margin < 2.0 * noise;
                eprintln!(
                    "P5 prompt {}: first divergence at step {step}, f32 margin {margin:.4e} vs \
                     2x noise {:.4e}: {}",
                    p.p,
                    2.0 * noise,
                    if tie { "near-tie" } else { "NOT a near-tie" }
                );
                near_ties += usize::from(tie);
                tie
            }
        };
        eprintln!(
            "P5 prompt {}: CUDA {got:?} HF-f32 {want:?} {}",
            p.p,
            verdict(ok)
        );
        pass &= ok;
    }
    eprintln!(
        "P5 rule: every first divergence at a near-tie step, margin < 2x the step's bf16_max_err \
         (this run's HF-bf16, #816); near-tie prompts {near_ties} of 5, not capped (the user's \
         2026-10-26 rule, as PR #11's chat rule) {}",
        verdict(pass)
    );
    assert!(pass, "the greedy tokens differ from HF-f32's");
}

#[test]
fn p6_chunked_equals_one_shot_on_real_weights() {
    let r = rig();
    let mut w = r.worker();
    let p = &prompts()[5];
    assert!(p.ids.len() > CHUNK, "prompt 6 is {} tokens", p.ids.len());
    let one_shot = forward(&mut w, &p.ids, 0, SEAT_A);
    forward(&mut w, &p.ids[..CHUNK], 0, SEAT_B);
    let rest = forward(&mut w, &p.ids[CHUNK..], CHUNK as u32, SEAT_B);
    let tail = &one_shot[CHUNK * VOCAB..];
    let mismatched = (rest.iter().zip(tail))
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let ok = mismatched == 0;
    eprintln!(
        "P6 prompt 6 ({} tokens): {CHUNK} fresh then {} continuing against one forward, rows \
         {CHUNK}..: mismatched_bits={mismatched} {}",
        p.ids.len(),
        p.ids.len() - CHUNK,
        verdict(ok)
    );
    assert!(ok, "the chunked forward differs from the one-shot one");
}

/// Prompt 1's one-shot forward alone, for `compute-sanitizer --tool memcheck` and `--tool
/// initcheck` (`cuda_hf_parity prompt_1_one_shot --exact`): the load, one forward, finite logits.
#[test]
fn prompt_1_one_shot() {
    let r = rig();
    let mut w = r.worker();
    let p = &prompts()[0];
    let got = forward(&mut w, &p.ids, 0, SEAT_A);
    let non_finite = got.iter().filter(|v| !v.is_finite()).count();
    let ok = non_finite == 0;
    eprintln!(
        "prompt 1 one-shot: {} rows, non-finite logits={non_finite} {}",
        p.ids.len(),
        verdict(ok)
    );
    assert!(ok, "non-finite logits");
}

// ── Decode on real weights ─────────────────────────────────────────

/// One decode of `token` at position `prefix_len` in `seat` (the m=1 tape, one row): its logits,
/// widened to f32.
fn decode(w: &mut CudaWorker<'_>, token: u32, prefix_len: u32, seat: (&[u32], u32)) -> Vec<f32> {
    let b = (CUDA_TAPES.iter().position(|t| t.bucket_m == 1)).expect("the m=1 tape");
    let kv = KvSequence {
        block_table: seat.0,
        prefix_len,
        q_len: 1,
    };
    let gdn = GDN_POOL.map(|_| GdnSequence {
        slot: seat.1,
        q_len: 1,
        fresh: false,
    });
    let mut out = vec![0u16; VOCAB];
    w.forward(
        b,
        Input::TokenIds(&[token]),
        &[prefix_len],
        Some(&kv),
        gdn.as_ref(),
        &mut out,
    )
    .unwrap_or_else(|e| panic!("decode at {prefix_len}: {e}"));
    out.iter().map(|&b| bf16::from_bits(b).to_f32()).collect()
}

/// Greedy through decode: one fresh m=512 prefill of the prompt (its last row's argmax is token
/// 1), then `steps − 1` decodes, each of the token just chosen: the tokens.
fn greedy_by_decode(w: &mut CudaWorker<'_>, ids: &[u32], steps: usize) -> Vec<u32> {
    let logits = forward(w, ids, 0, SEAT_A);
    let mut token = top_k(&logits[(ids.len() - 1) * VOCAB..], 1)[0] as u32;
    let mut got = vec![token];
    for step in 1..steps {
        let p = (ids.len() + step - 1) as u32;
        token = top_k(&decode(w, token, p, SEAT_A), 1)[0] as u32;
        got.push(token);
    }
    got
}

/// P7: prompts 1-5, one fresh m=512 prefill then 7 decodes give HF-f32's 8 greedy tokens, under
/// P5's near-tie rule (every first divergence at a step whose f32 margin is under twice that
/// step's `bf16_max_err`; the number of such prompts is not capped). Both sides printed.
#[test]
fn p7_greedy_via_decode_matches_hf_f32() {
    let r = rig();
    let mut w = r.worker();
    let mut near_ties = 0;
    let mut pass = true;
    for p in prompts().iter().take(5) {
        let path = goldens_dir().join(format!("greedy_{}.json", p.p));
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let golden: serde_json::Value = serde_json::from_str(&text).expect("greedy json");
        let steps = golden["steps"].as_array().expect("steps");
        let want: Vec<u32> = (steps.iter())
            .map(|s| s["token"].as_u64().expect("a token") as u32)
            .collect();
        let got = greedy_by_decode(&mut w, &p.ids, want.len());
        let ok = match got.iter().zip(&want).position(|(g, w)| g != w) {
            None => true,
            Some(step) => {
                let margin = steps[step]["margin"].as_f64().expect("a margin");
                let noise = steps[step]["bf16_max_err"].as_f64().expect("a noise");
                let tie = margin < 2.0 * noise;
                eprintln!(
                    "P7 prompt {}: first divergence at step {step}, f32 margin {margin:.4e} vs \
                     2x noise {:.4e}: {}",
                    p.p,
                    2.0 * noise,
                    if tie { "near-tie" } else { "NOT a near-tie" }
                );
                near_ties += usize::from(tie);
                tie
            }
        };
        eprintln!(
            "P7 prompt {}: CUDA prefill + 7 decodes {got:?} HF-f32 {want:?} {}",
            p.p,
            verdict(ok)
        );
        pass &= ok;
    }
    eprintln!(
        "P7 rule: P5's (this run's HF-bf16, #830); near-tie prompts {near_ties} of 5, not capped \
         (the user's 2026-10-26 rule, as PR #11's chat rule) {}",
        verdict(pass)
    );
    assert!(pass, "the greedy tokens via decode differ from HF-f32's");
}

/// P8: for prompts 1-5, prefill the first L − 1 tokens, then decode token L − 1. That row against
/// HF-f32's last row: cosine under the model's `P1` rule, RMS and max error ≤ 2 × HF-bf16's on that row, the same
/// top-1 unless the f32 margin is at most twice the row's max noise, top-5 overlap ≥ 4. Against
/// CUDA's own one-shot last row: cosine ≥ 0.999, the same argmax unless near-tie, max |diff|
/// printed. A one-token prompt has no prefix to prefill, and is skipped, saying so.
#[test]
fn p8_the_decode_row_matches_hf_f32() {
    let r = rig();
    let mut w = r.worker();
    let mut pass = true;
    for p in prompts().iter().take(5) {
        let n = p.ids.len();
        if n < 2 {
            eprintln!(
                "P8 prompt {}: one token, no prefix to prefill: SKIPPED",
                p.p
            );
            continue;
        }
        let one_shot = forward(&mut w, &p.ids, 0, SEAT_B);
        let one_shot = &one_shot[(n - 1) * VOCAB..];
        forward(&mut w, &p.ids[..n - 1], 0, SEAT_A);
        let row = decode(&mut w, p.ids[n - 1], (n - 1) as u32, SEAT_A);
        let (f32_row, bf16_row) = (&p.f32[(n - 1) * VOCAB..], &p.bf16[(n - 1) * VOCAB..]);
        let cos = cosine(&row, f32_row);
        let (rms, max) = error(&row, f32_row);
        let (noise_rms, noise_max) = error(bf16_row, f32_row);
        let top = top_k(f32_row, 2);
        let margin = f64::from(f32_row[top[0]] - f32_row[top[1]]);
        let tie = margin <= 2.0 * noise_max;
        let same = top_k(&row, 1)[0] == top[0];
        let c5: BTreeSet<usize> = top_k(&row, 5).into_iter().collect();
        let f5: BTreeSet<usize> = top_k(f32_row, 5).into_iter().collect();
        let shared = c5.intersection(&f5).count();
        let noise_cos = cosine(bf16_row, f32_row);
        let cos_ok = p1_gate(&P1, cos, noise_cos);
        let cos_need = match P1 {
            P1Rule::Floor(f) => format!("{f}"),
            P1Rule::BelowHfBf16(d) => format!("HF-bf16's {noise_cos:.6} - {d}"),
            P1Rule::ScatterQuantiles { .. } => {
                format!("HF-bf16's {noise_cos:.6} - {:.6}", p1_margin())
            }
        };
        let ok_hf = cos_ok
            && rms <= 2.0 * noise_rms
            && max <= 2.0 * noise_max
            && (same || tie)
            && shared >= 4;
        eprintln!(
            "P8 prompt {} decode row {} vs HF-f32: cosine={cos:.6} (need >= {cos_need}) \
             RMS={rms:.6e} (HF-bf16 {noise_rms:.6e}) max={max:.6e} (HF-bf16 {noise_max:.6e}) \
             (need <= 2x each) top-1 same={same} (f32 margin {margin:.4e}, near-tie={tie}) \
             top-5 shared={shared} (need >= 4) {}",
            p.p,
            n - 1,
            verdict(ok_hf)
        );
        let cos_own = cosine(&row, one_shot);
        let (_, diff_own) = error(&row, one_shot);
        let same_own = top_k(&row, 1)[0] == top_k(one_shot, 1)[0];
        let ok_own = cos_own >= 0.999 && (same_own || tie);
        eprintln!(
            "P8 prompt {} decode row {} vs CUDA's one-shot last row: cosine={cos_own:.6} (need \
             >= 0.999) argmax same={same_own} (near-tie={tie}) max|diff|={diff_own:.6e} {}",
            p.p,
            n - 1,
            verdict(ok_own)
        );
        pass &= ok_hf && ok_own;
    }
    assert!(
        pass,
        "a decode row is out of agreement: see the lines above"
    );
}

/// Prompt 1's prefill then 7 decodes alone, for `compute-sanitizer --tool memcheck`
/// (`cuda_hf_parity prompt_1_decode --exact`): the load, the forwards, finite logits.
#[test]
fn prompt_1_decode() {
    let r = rig();
    let mut w = r.worker();
    let p = &prompts()[0];
    let logits = forward(&mut w, &p.ids, 0, SEAT_A);
    let mut token = top_k(&logits[(p.ids.len() - 1) * VOCAB..], 1)[0] as u32;
    let mut non_finite = 0;
    for step in 0..7 {
        let row = decode(&mut w, token, (p.ids.len() + step) as u32, SEAT_A);
        non_finite += row.iter().filter(|v| !v.is_finite()).count();
        token = top_k(&row, 1)[0] as u32;
    }
    let ok = non_finite == 0;
    eprintln!(
        "prompt 1 prefill + 7 decodes: non-finite decode logits={non_finite} {}",
        verdict(ok)
    );
    assert!(ok, "non-finite decode logits");
}

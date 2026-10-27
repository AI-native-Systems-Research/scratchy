// SPDX-License-Identifier: Apache-2.0
//! Gemma 4 26B-A4B's CUDA prefill and decode, run from the real `google/gemma-4-26B-A4B-it`
//! checkpoint on the H100, against HuggingFace transformers on the same token ids
//! (AI-native-Systems-Research/scratchy#933, step 13c).
//!
//! The same checks as qwen3.6-35b-a3b's `cuda_hf_parity_35b.rs` (the tests are
//! `common/cuda_hf_parity_body.rs`), at Gemma's constants, plus three decode checks on real
//! weights. The reference is `tests/cuda_hf_reference.py --model gemma-4-26b-a4b-it`'s goldens
//! (`tests/goldens/gemma4-moe-gemma-4-26b-a4b-it-hf/`), of the pinned revision
//! `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` (2 shards), its experts run as HF's per-expert
//! `eager` loop, every prompt's ids starting with `<bos>` (2). The weights load through
//! `load_weights` and `CUDA_WEIGHTS`; the forward is `CudaWorker` over `CUDA_TAPES`. Both sides'
//! logits are soft-capped at ±30, and the near-tie margins are of the capped values.
//!
//! - memory_fits, run first and alone: `cudaMemGetInfo` and `nvidia-smi` at four points, 627 + 2
//!   allocations (the arena and the KV pool; there is no GDN layer), all five buckets, the peak
//!   after an m=4096 forward and a decode at most 78.0 GiB of the 79.6 GiB, with the headroom.
//!   #933 predicts about 52.6 GiB.
//! - load: 627 weight buffers and 2 worker allocations, each its `weight_bytes`; embed rows 0 and
//!   262143 (the lm_head is tied), and layers 0 and 29's expert banks (experts 0 and 127's first
//!   `gate_up` and `down` rows), routers (`router.proj`, `router.scale`,
//!   `router.per_expert_scale`) and `layer_scalar` equal the checkpoint's; the load time.
//! - P1 cosine, gated: on each prompt, min cosine(CUDA, f32) ≥ min cosine(HF-bf16, f32) − 0.005
//!   (the user's 2026-10-26 rule for MoE models, #859's).
//! - P2, P3, P4, P6: the 9b's rules, gated. P5, P7: every first divergence a near-tie.
//! - P8, gated: the decode row against HF-f32 under P1's rule, and against CUDA's own one-shot row
//!   at cosine ≥ 0.999.
//! - (k), (n), (r): CUDA's decode rows against CUDA's own one-shot rows on real weights, below.
//!
//! The expert picks (P9) are `cuda_route_parity_gemma4.rs`'s. One rig holds about 52.6 GiB of the
//! H100's 79.6 GiB (50.5 GB of weights, the 3.5 GB m=4096 arena, the KV pool), so run this binary
//! with `--test-threads=1`. Built only under `nvcc` and `gemma-4-26b-a4b-it`
//! (`required-features`). Needs an sm_90 GPU and the goldens: a missing goldens dir fails with
//! the command that makes it, never skips. `memory_fits` and `load_checks` read only the goldens'
//! `checkpoint`. `prompt_1_one_shot` and `prompt_1_decode` run alone for the sanitizers. Every
//! check prints its line before a test fails.

use scratchy_models::gemma4_moe::gemma_4_26b_a4b_it::{CUDA_TAPES, CUDA_WEIGHTS};
use scratchy_target_cuda::step::CudaStep;
use scratchy_target_cuda::tape::CudaTape;

include!("common/cuda_hf_parity_body.rs");

const HIDDEN: usize = 2816;
const VOCAB: usize = 262_144;
/// The weight buffers `load_weights` uploads: 627 (30 expert banks, each two checkpoint tensors;
/// the tied lm_head reads the embedding table's).
const WEIGHTS: u64 = 627;
/// Tied (`tie_word_embeddings`): the lm_head is the embedding table, whose rows `load_checks`
/// checks already.
const UNTIED_LM_HEAD: Option<&str> = None;
/// The goldens: `cuda_hf_reference.py --model`, the checkpoint, and where they are written.
const MODEL: &str = "gemma-4-26b-a4b-it";
const CHECKPOINT: &str = "gemma-4-26B-A4B-it";
const GOLDENS: &str = "tests/goldens/gemma4-moe-gemma-4-26b-a4b-it-hf";
/// P1's rule (#933, the user's decision 2026-10-27, option (a)): min cosine(CUDA, f32) at most
/// HF-bf16's own measured scatter below HF-bf16's own min cosine on the same prompt, the scatter
/// measured on the goldens (prompt 1's rows are prompt 6's rows 0-20; the verifier measured 0.0126
/// on row 4) and asserted at most 0.013, the figure the user approved.
const P1: P1Rule = P1Rule::BelowHfBf16Scatter { cap: 0.013 };
/// P3 accepts at most one top-1 disagreement outside a near-tie per prompt, on a route-explained
/// row (the user's decision 2026-10-27, option (a)).
const P3: P3Rule = P3Rule::RouteExplained { per_prompt: 1 };
/// No GDN layer: no GDN pool, and no `GdnSequence` in a forward. The worker's allocations are the
/// arena and the KV pool.
const GDN_POOL: Option<GdnPoolConfig> = None;
const WORKER_ALLOCATIONS: u64 = 2;

// ── Decode on real weights: (k), (n) and (r) ───────────────────────

// These are #925's decode-against-one-shot checks, which its synthetic weights could not judge
// (their attention is nearly a hard argmax), rerun on the checkpoint. CUDA is compared with CUDA,
// so no HF logits are needed. Per row printed: cosine, max |diff|, whether the top-1 agrees and
// whether the one-shot row is a near-tie (its top-1 − top-2 margin at most twice the max |diff|),
// the first layer whose expert picks differ from the one-shot's and how many do, and finiteness.
// Gated: finiteness, and the top-1 on every row that is not a near-tie. Any cosine below
// `STOP_BELOW` fails the test as a stop-and-report (#933). The floors that will gate the cosines
// are the user's, proposed from this run's numbers; until then they are printed, not set.

/// The layers, each with its own routing.
const LAYERS: usize = 30;
/// The decode input's length: prompt 6's `<bos>`, then the rest of its ids repeated.
const DECODE_TOKENS: usize = 1100;
/// The beginning-of-sequence id every Gemma prompt starts with.
const BOS: u32 = 2;
/// A cosine below this stops the run for a report (#933): a decode-path defect, not bf16 noise.
/// Not a floor; the floors are the user's.
const STOP_BELOW: f64 = 0.99;
/// The decode rig's KV pool: 40 blocks, two seats of [`DECODE_SEAT_BLOCKS`] (1,088 tokens each,
/// for (r)'s 1,028), 576,716,800 bytes.
const DECODE_KV_POOL: KvPoolConfig = KvPoolConfig { num_blocks: 40 };
const DECODE_SEAT_BLOCKS: usize = 17;

const fn decode_seat(from: u32) -> [u32; DECODE_SEAT_BLOCKS] {
    let mut b = [0; DECODE_SEAT_BLOCKS];
    let mut i = 0;
    while i < DECODE_SEAT_BLOCKS {
        b[i] = from + i as u32;
        i += 1;
    }
    b
}

/// Blocks 0..17: the prefills and decodes.
const DECODE_SEAT_A: [u32; DECODE_SEAT_BLOCKS] = decode_seat(0);
/// Blocks 17..34: the one-shot forwards, no block of [`DECODE_SEAT_A`]'s.
const DECODE_SEAT_B: [u32; DECODE_SEAT_BLOCKS] = decode_seat(DECODE_SEAT_BLOCKS as u32);

/// The decode input: prompt 6's ids from `meta.json` (`<bos>` then all eight prompt lines), with
/// everything after the `<bos>` repeated until there are [`DECODE_TOKENS`]. Real text, with no
/// tokenizer call.
fn decode_ids() -> Vec<u32> {
    let p6: Vec<u32> = (meta()["prompts"][5]["ids"].as_array())
        .expect("meta.json lists prompt 6's ids")
        .iter()
        .map(|v| v.as_u64().expect("an id") as u32)
        .collect();
    assert_eq!(p6.first(), Some(&BOS), "prompt 6 starts with <bos>");
    assert!(p6.len() > 1, "prompt 6 has text after <bos>");
    let mut ids = vec![BOS];
    while ids.len() < DECODE_TOKENS {
        ids.extend_from_slice(&p6[1..]);
    }
    ids.truncate(DECODE_TOKENS);
    ids
}

/// [`CUDA_TAPES`], then each of `ms`' tapes cut just after each layer's `MoeRoute` step, its
/// result that layer's `topk_idx` (`cuda_prefill_gemma4`'s `route_cuts`): cut `l` of `ms[i]` is
/// tape [`pick_cut`]`(i, l)`. A `'static` list for one worker; every span stays in the same
/// arena, so each cut validates as the full tape does.
fn with_route_cuts(ms: &[u32]) -> &'static [CudaTape] {
    let mut list = CUDA_TAPES.to_vec();
    for &m in ms {
        let tape = (CUDA_TAPES.iter().find(|t| t.bucket_m == m)).expect("the bucket's tape");
        let moe = tape.moe.expect("a MoE scratch");
        let cuts: Vec<CudaTape> = (tape.steps.iter().enumerate())
            .filter(|(_, s)| matches!(s, CudaStep::MoeRoute(_)))
            .map(|(i, _)| CudaTape {
                steps: Box::leak(tape.steps[..=i].to_vec().into_boxed_slice()),
                result: moe.topk_idx,
                ..*tape
            })
            .collect();
        assert_eq!(cuts.len(), LAYERS, "m={m}: one MoeRoute step per layer");
        list.extend(cuts);
    }
    Box::leak(list.into_boxed_slice())
}

/// The tape index of cut `layer` of the `i`-th bucket a [`with_route_cuts`] list was made for.
fn pick_cut(i: usize, layer: usize) -> usize {
    CUDA_TAPES.len() + i * LAYERS + layer
}

/// The tape index of the m-row bucket: [`CUDA_TAPES`] lead every list here, in order.
fn bucket_of(m: u32) -> usize {
    (CUDA_TAPES.iter().position(|t| t.bucket_m == m)).expect("the bucket's tape")
}

/// One forward of `ids` at `prefix_len` in `blocks`, through tape `b` of `tapes` (the worker's
/// own list): the full result.
fn run_on(
    w: &mut CudaWorker<'_>,
    (tapes, b): (&[CudaTape], usize),
    ids: &[u32],
    prefix_len: u32,
    blocks: &[u32],
) -> Vec<u16> {
    let tape = &tapes[b];
    let mut padded = vec![0u32; tape.bucket_m as usize];
    padded[..ids.len()].copy_from_slice(ids);
    let positions: Vec<u32> = (prefix_len..prefix_len + tape.bucket_m).collect();
    let kv = KvSequence {
        block_table: blocks,
        prefix_len,
        q_len: ids.len() as u32,
    };
    let mut out = vec![0u16; tape.result.bytes as usize / 2];
    w.forward(
        b,
        Input::TokenIds(&padded),
        &positions,
        Some(&kv),
        None,
        &mut out,
    )
    .unwrap_or_else(|e| panic!("m={} at {prefix_len}: {e}", tape.bucket_m));
    out
}

/// i32 words from the u16 halves a picks result holds.
fn pick_words(halves: &[u16]) -> Vec<i32> {
    (halves.as_chunks::<2>().0.iter())
        .map(|&[lo, hi]| (u32::from(lo) | u32::from(hi) << 16) as i32)
        .collect()
}

/// Row `row`'s picks, as a set, of a `[rows, top_k]` result.
fn pick_set(idx: &[i32], row: usize, top_k: usize) -> BTreeSet<i32> {
    idx[row * top_k..][..top_k].iter().copied().collect()
}

/// One decode check: its worker over the tapes and the route cuts of the one-shot bucket and of
/// m=1, the one-shot rows as f32, and the one-shot's picks per layer.
struct DecodeCheck<'r> {
    worker: CudaWorker<'r>,
    tapes: &'static [CudaTape],
    one_shot: Vec<f32>,
    picks: Vec<Vec<i32>>,
    top_k: usize,
}

impl<'r> DecodeCheck<'r> {
    /// `ids` fresh in one forward of the `one_shot_m` bucket, in [`DECODE_SEAT_B`]: its logits,
    /// and its picks per layer from the route cuts (a cut forward rewrites the positions it runs
    /// with what the full one writes there).
    fn new(r: &'r Rig, one_shot_m: u32, ids: &[u32]) -> Self {
        let tapes = with_route_cuts(&[one_shot_m, 1]);
        let mut worker =
            CudaWorker::new(&r.device, tapes, &r.weights, Some(DECODE_KV_POOL), GDN_POOL)
                .unwrap_or_else(|e| {
                    panic!("CudaWorker::new over the tapes and the route cuts: {e}")
                });
        let full = run_on(
            &mut worker,
            (tapes, bucket_of(one_shot_m)),
            ids,
            0,
            &DECODE_SEAT_B,
        );
        let one_shot = (full[..ids.len() * VOCAB].iter())
            .map(|&b| bf16::from_bits(b).to_f32())
            .collect();
        let picks = (0..LAYERS)
            .map(|l| {
                pick_words(&run_on(
                    &mut worker,
                    (tapes, pick_cut(0, l)),
                    ids,
                    0,
                    &DECODE_SEAT_B,
                ))
            })
            .collect();
        let top_k = (CUDA_TAPES[bucket_of(1)].moe).expect("a MoE scratch").top_k as usize;
        Self {
            worker,
            tapes,
            one_shot,
            picks,
            top_k,
        }
    }

    /// A fresh prefill of `ids` in the m-row bucket, in [`DECODE_SEAT_A`].
    fn prefill(&mut self, m: u32, ids: &[u32]) {
        run_on(
            &mut self.worker,
            (self.tapes, bucket_of(m)),
            ids,
            0,
            &DECODE_SEAT_A,
        );
    }

    /// A decode of `token` at `p` in [`DECODE_SEAT_A`] on the cache as it stands, first cut after
    /// each layer's routing, then whole: the first layer whose picks are not the one-shot row
    /// `p`'s and how many layers' are not, and the decode's logits as f32. A cut decode writes
    /// position `p` of the layers it runs with what the full decode writes there.
    fn decode(&mut self, token: u32, p: usize) -> (Option<usize>, usize, Vec<f32>) {
        let (mut first, mut count) = (None, 0);
        for l in 0..LAYERS {
            let got = run_on(
                &mut self.worker,
                (self.tapes, pick_cut(1, l)),
                &[token],
                p as u32,
                &DECODE_SEAT_A,
            );
            if pick_set(&pick_words(&got), 0, self.top_k) != pick_set(&self.picks[l], p, self.top_k)
            {
                first = first.or(Some(l));
                count += 1;
            }
        }
        let row = run_on(
            &mut self.worker,
            (self.tapes, bucket_of(1)),
            &[token],
            p as u32,
            &DECODE_SEAT_A,
        );
        let row = (row[..VOCAB].iter())
            .map(|&b| bf16::from_bits(b).to_f32())
            .collect();
        (first, count, row)
    }

    /// The decode row at `p` against the one-shot row `p`, printed: finiteness, cosine, max
    /// |diff|, the top-1 and the near-tie, and the pick scan. Returns `(gated pass, cosine)`.
    fn line(&self, what: &str, p: usize, scan: (Option<usize>, usize), row: &[f32]) -> (bool, f64) {
        let want = &self.one_shot[p * VOCAB..][..VOCAB];
        let finite = row.iter().all(|v| v.is_finite());
        let cos = cosine(row, want);
        let (_, diff) = error(row, want);
        let top = top_k(want, 2);
        let margin = f64::from(want[top[0]] - want[top[1]]);
        let tie = margin <= 2.0 * diff;
        let got = top_k(row, 1)[0];
        let same = got == top[0];
        let ok = finite && (same || tie);
        eprintln!(
            "{what}: finite={finite} cosine={cos:.6} max|diff|={diff:.6e} top-1 \
             {got} one-shot {} same={same} (one-shot margin {margin:.4e}, near-tie when <= 2 x \
             max|diff| = {:.4e}: {tie}) first layer whose picks differ={:?}, layers whose picks \
             differ={} of {LAYERS} (gated: finite, top-1 unless near-tie) {}",
            top[0],
            2.0 * diff,
            scan.0,
            scan.1,
            verdict(ok)
        );
        (ok, cos)
    }
}

/// After a decode check's rows: the worst cosine, printed for the user's floor, and the stop rule.
/// Fails the test on a gated row, then on a cosine below [`STOP_BELOW`].
fn decode_verdict(what: &str, pass: bool, cosines: &[f64]) {
    let worst = cosines.iter().copied().fold(f64::INFINITY, f64::min);
    let stop = worst < STOP_BELOW;
    eprintln!(
        "{what}: {} rows, worst cosine {worst:.6} (the floor that will gate it is the user's, \
         proposed from this run, not set; below {STOP_BELOW} stops the run for a report, #933): \
         {}",
        cosines.len(),
        if stop {
            "STOP AND REPORT"
        } else {
            "above the stop rule"
        }
    );
    assert!(
        pass,
        "{what}: a decode row is non-finite or its top-1 disagrees outside a near-tie"
    );
    assert!(
        !stop,
        "{what}: a decode cosine is below {STOP_BELOW}: stop and report the table above"
    );
}

/// (k), teacher-forced: for each P = 300 … 307, a fresh prefill of P tokens in m=512, then one
/// decode of token P, against row P of one 308-token m=512 forward in disjoint blocks.
#[test]
fn k_real_weights_teacher_forced_decodes() {
    let r = rig();
    let ids = decode_ids();
    let mut c = DecodeCheck::new(&r, 512, &ids[..308]);
    let (mut pass, mut cosines) = (true, Vec::new());
    for (p, &token) in ids.iter().enumerate().take(308).skip(300) {
        c.prefill(512, &ids[..p]);
        let (first, count, row) = c.decode(token, p);
        let (ok, cos) = c.line(
            &format!(
                "(k) real weights, decode at P={p} after {p} tokens prefilled, against row {p} of \
                 one 308-token forward"
            ),
            p,
            (first, count),
            &row,
        );
        pass &= ok;
        cosines.push(cos);
    }
    decode_verdict("(k) real weights", pass, &cosines);
}

/// (n), the KV-block boundary: 319 tokens prefilled in m=512, then decodes at P = 319 (the last
/// slot of block 4) and P = 320 (the first of block 5), against one 321-token forward.
#[test]
fn n_real_weights_decodes_at_a_block_boundary() {
    let r = rig();
    let ids = decode_ids();
    let mut c = DecodeCheck::new(&r, 512, &ids[..321]);
    c.prefill(512, &ids[..319]);
    let (mut pass, mut cosines) = (true, Vec::new());
    for (p, &token) in ids.iter().enumerate().take(321).skip(319) {
        let (first, count, row) = c.decode(token, p);
        let (ok, cos) = c.line(
            &format!(
                "(n) real weights, decode at P={p} (kv_len {}, block {} slot {}), against row {p} \
                 of one 321-token forward",
                p + 1,
                p / 64,
                p % 64
            ),
            p,
            (first, count),
            &row,
        );
        pass &= ok;
        cosines.push(cos);
    }
    decode_verdict("(n) real weights", pass, &cosines);
}

/// (r), across the sliding window: 1,020 tokens prefilled in m=1024, then decodes at P = 1020 …
/// 1027 in turn (the sliding layers drop key P − 1024 from P = 1024), against one 1,028-token
/// forward on the m=2048 tape.
#[test]
fn r_real_weights_decodes_across_the_window() {
    let r = rig();
    let ids = decode_ids();
    let mut c = DecodeCheck::new(&r, 2048, &ids[..1028]);
    c.prefill(1024, &ids[..1020]);
    let (mut pass, mut cosines) = (true, Vec::new());
    for (p, &token) in ids.iter().enumerate().take(1028).skip(1020) {
        let (first, count, row) = c.decode(token, p);
        let (ok, cos) = c.line(
            &format!(
                "(r) real weights, decode at P={p} (window floor {}), against row {p} of one \
                 1028-token forward",
                p.saturating_sub(1023)
            ),
            p,
            (first, count),
            &row,
        );
        pass &= ok;
        cosines.push(cos);
    }
    decode_verdict("(r) real weights", pass, &cosines);
}

// ── #933's diagnosis prints: P1 prompt 6 and P3 prompt 3 ───────────

/// The P1 prompt the diagnosis reads per row, and the row bands its RMS ratio is split into
/// (inclusive; the first is the prefix prompt 6 shares with prompt 1, and 64 and 128 are key-tile
/// boundaries of the D=512 global attention).
const DIAG_P1_PROMPT: usize = 6;
const DIAG_P1_BANDS: [(usize, usize); 4] = [(0, 20), (21, 63), (64, 127), (128, 158)];
/// The rows printed as CUDA's worst per-row cosines.
const DIAG_P1_WORST: usize = 5;
/// The P3 prompt and row the diagnosis reads, and the two ids printed there: HF-f32's top-1 and
/// top-2 on that row in the goldens.
const DIAG_P3_PROMPT: usize = 3;
const DIAG_P3_ROW: usize = 9;
const DIAG_P3_IDS: [usize; 2] = [1852, 32990];

/// #933's diagnosis (`/tmp/v933-diag.md`), printed only, nothing gated: one fresh forward each
/// of prompts 6 and 3, as `p1_to_p4_one_forward_per_prompt` makes them.
/// - Prompt 6, every row: cosine(CUDA, HF-f32) and cosine(HF-bf16, HF-f32); CUDA's
///   [`DIAG_P1_WORST`] lowest rows; and per band of [`DIAG_P1_BANDS`], RMS(CUDA − f32),
///   RMS(HF-bf16 − f32) and their ratio, which says whether CUDA's error steps up at row 64 or
///   128.
/// - Prompt 3, row 9: CUDA's argmax and its logit, and the logits at [`DIAG_P3_IDS`] from CUDA,
///   HF-f32 and HF-bf16, with each side's argmax and the row's max |CUDA − f32| and
///   max |HF-bf16 − f32|.
#[test]
fn diag_p1_prompt_6_rows_and_p3_prompt_3_row_9() {
    let r = rig();
    let mut w = r.worker();
    let prompt = |n: usize| {
        (prompts().iter().find(|p| p.p == n)).unwrap_or_else(|| panic!("prompt {n} in meta.json"))
    };
    // Prompt 6, per row.
    let p = prompt(DIAG_P1_PROMPT);
    let n = p.ids.len();
    let got = forward(&mut w, &p.ids, 0, SEAT_A);
    let mut per_row: Vec<(usize, f64, f64)> = Vec::with_capacity(n);
    for (row, ((c, f), b)) in rows(&got).zip(rows(&p.f32)).zip(rows(&p.bf16)).enumerate() {
        let (cc, bc) = (cosine(c, f), cosine(b, f));
        let ((crms, cmax), (brms, bmax)) = (error(c, f), error(b, f));
        eprintln!(
            "diag P1 prompt {} row {row} (token {}): cosine(CUDA, f32)={cc:.6} \
             cosine(HF-bf16, f32)={bc:.6} CUDA - HF-bf16={:+.6} RMS(CUDA-f32)={crms:.4e} \
             RMS(HF-bf16-f32)={brms:.4e} max|CUDA-f32|={cmax:.4e} max|HF-bf16-f32|={bmax:.4e} \
             (information only)",
            p.p,
            p.ids[row],
            cc - bc
        );
        per_row.push((row, cc, bc));
    }
    let mut worst = per_row.clone();
    worst.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    for &(row, cc, bc) in worst.iter().take(DIAG_P1_WORST) {
        eprintln!(
            "diag P1 prompt {} worst CUDA row {row}: cosine(CUDA, f32)={cc:.6} \
             cosine(HF-bf16, f32)={bc:.6} (information only)",
            p.p
        );
    }
    for (lo, hi) in DIAG_P1_BANDS {
        if lo >= n {
            eprintln!(
                "diag P1 prompt {} rows {lo}-{hi}: none of its {n} rows",
                p.p
            );
            continue;
        }
        let hi = hi.min(n - 1);
        let span = lo * VOCAB..(hi + 1) * VOCAB;
        let (crms, _) = error(&got[span.clone()], &p.f32[span.clone()]);
        let (brms, _) = error(&p.bf16[span.clone()], &p.f32[span]);
        let mean = |pick: fn(&(usize, f64, f64)) -> f64| {
            per_row[lo..=hi].iter().map(pick).sum::<f64>() / (hi + 1 - lo) as f64
        };
        eprintln!(
            "diag P1 prompt {} rows {lo}-{hi}: RMS(CUDA-f32)={crms:.4e} RMS(HF-bf16-f32)={brms:.4e} \
             ratio CUDA/HF-bf16={:.3} mean cosine CUDA={:.6} HF-bf16={:.6} (information only)",
            p.p,
            crms / brms.max(f64::MIN_POSITIVE),
            mean(|r| r.1),
            mean(|r| r.2)
        );
    }
    // Prompt 3, row 9.
    let p = prompt(DIAG_P3_PROMPT);
    let row = DIAG_P3_ROW;
    assert!(row < p.ids.len(), "prompt {} has row {row}", p.p);
    let got = forward(&mut w, &p.ids, 0, SEAT_A);
    let at = |v: &[f32]| v[row * VOCAB..][..VOCAB].to_vec();
    let (c, f, b) = (at(&got), at(&p.f32), at(&p.bf16));
    for (side, logits) in [("CUDA", &c), ("HF-f32", &f), ("HF-bf16", &b)] {
        let top = top_k(logits, 2);
        eprintln!(
            "diag P3 prompt {} row {row} (token {}) {side}: argmax {} logit {:.4} (2nd {} logit \
             {:.4}), logit[{}]={:.4} logit[{}]={:.4} (information only)",
            p.p,
            p.ids[row],
            top[0],
            logits[top[0]],
            top[1],
            logits[top[1]],
            DIAG_P3_IDS[0],
            logits[DIAG_P3_IDS[0]],
            DIAG_P3_IDS[1],
            logits[DIAG_P3_IDS[1]]
        );
    }
    let ((crms, cmax), (brms, bmax)) = (error(&c, &f), error(&b, &f));
    eprintln!(
        "diag P3 prompt {} row {row}: cosine(CUDA, f32)={:.6} cosine(HF-bf16, f32)={:.6} \
         RMS(CUDA-f32)={crms:.4e} RMS(HF-bf16-f32)={brms:.4e} max|CUDA-f32|={cmax:.4e} \
         max|HF-bf16-f32|={bmax:.4e} (information only)",
        p.p,
        cosine(&c, &f),
        cosine(&b, &f)
    );
}

// ── #933's diagnosis prints, part 2: decode at the bad rows ────────

/// The (prompt, row) pairs the decode diagnosis runs a second way: prompt 6's rows whose CUDA
/// error is furthest above HF-bf16's (`/tmp/v933-diag2.md`), and prompt 3's row 9.
const DIAG_DECODE_ROWS: [(usize, usize); 7] = [
    (DIAG_P1_PROMPT, 25),
    (DIAG_P1_PROMPT, 30),
    (DIAG_P1_PROMPT, 37),
    (DIAG_P1_PROMPT, 41),
    (DIAG_P1_PROMPT, 71),
    (DIAG_P1_PROMPT, 148),
    (DIAG_P3_PROMPT, DIAG_P3_ROW),
];

/// #933's diagnosis, part 2, test A (`/tmp/v933-diag2.md`), printed only, nothing gated: each row
/// `r` of [`DIAG_DECODE_ROWS`] as a decode (a fresh prefill of the prompt's rows `0..r` in m=512,
/// then one decode of token `r` at P = `r`), beside CUDA's one-shot row `r` (one fresh m=512
/// forward of the whole prompt) and the goldens' row `r`. Per row: cosine, RMS and max |·| of
/// decode − HF-f32, one-shot − HF-f32 and HF-bf16 − HF-f32; cosine and max |·| of decode −
/// one-shot; each side's argmax; and the first layer whose decode picks differ from the
/// one-shot's, and how many do. The decode runs the decode attention and the gathered expert
/// GEMV; the one-shot the prefill attention, the sort and the grouped GEMM.
#[test]
fn diag_decode_at_the_bad_rows() {
    let r = rig();
    for prompt in [DIAG_P1_PROMPT, DIAG_P3_PROMPT] {
        let p = (prompts().iter().find(|p| p.p == prompt))
            .unwrap_or_else(|| panic!("prompt {prompt} in meta.json"));
        let mut c = DecodeCheck::new(&r, M, &p.ids);
        for &(_, row) in DIAG_DECODE_ROWS.iter().filter(|(q, _)| *q == prompt) {
            assert!(
                row > 0 && row < p.ids.len(),
                "prompt {prompt} has row {row} after a prefix"
            );
            c.prefill(M, &p.ids[..row]);
            let (first, count, decode) = c.decode(p.ids[row], row);
            let at = |v: &[f32]| v[row * VOCAB..][..VOCAB].to_vec();
            let (one_shot, f, b) = (at(&c.one_shot), at(&p.f32), at(&p.bf16));
            let ((drms, dmax), (orms, omax), (brms, bmax)) =
                (error(&decode, &f), error(&one_shot, &f), error(&b, &f));
            let (_, dvo) = error(&decode, &one_shot);
            let argmax = |v: &[f32]| top_k(v, 1)[0];
            eprintln!(
                "diag decode prompt {prompt} row {row} (token {}): decode vs HF-f32 cosine={:.6} \
                 RMS={drms:.4e} max={dmax:.4e}; one-shot vs HF-f32 cosine={:.6} RMS={orms:.4e} \
                 max={omax:.4e}; HF-bf16 vs HF-f32 cosine={:.6} RMS={brms:.4e} max={bmax:.4e}; \
                 decode vs one-shot cosine={:.6} max={dvo:.4e}; argmax decode {} one-shot {} \
                 HF-f32 {} HF-bf16 {}; first layer whose decode picks differ from the \
                 one-shot's={first:?}, layers that differ={count} of {LAYERS} (information only)",
                p.ids[row],
                cosine(&decode, &f),
                cosine(&one_shot, &f),
                cosine(&b, &f),
                cosine(&decode, &one_shot),
                argmax(&decode),
                argmax(&one_shot),
                argmax(&f),
                argmax(&b)
            );
        }
    }
}

// ── #933's diagnosis prints, part 3: P1 prompt 6 under option (a) ──

/// The bad rows test A reruns besides prompt 6's minimum row: prompt 6's rows 25, 30, 37, 41, 71
/// and 148, and prompt 3's row 9 (`/tmp/v933-diag2.md:73-87`).
const DIAG_A_ROWS: [(usize, usize); 7] = DIAG_DECODE_ROWS;

/// #933's diagnosis, test A (`/tmp/v933-diag2.md:73-87`), printed only, nothing gated. First,
/// prompt 6's row of CUDA's minimum cosine against HF-f32 (P1's line prints only the value), from
/// one fresh m=512 forward of the prompt. Then, for that row and each of [`DIAG_A_ROWS`], the
/// row `r` as a decode ((k)'s machinery, `DecodeCheck`: a fresh prefill of rows `0..r` in m=512,
/// then one decode of token `r` at P = `r`): cosine(decode, CUDA one-shot row r), cosine(decode,
/// HF-f32 row r), cosine(one-shot, HF-f32 row r), cosine(HF-bf16, HF-f32 row r), and the layers
/// whose decode picks differ from the one-shot's. Decode closer to HF-f32 than one-shot names a
/// prefill-only kernel; decode equal to one-shot clears them.
#[test]
fn diag_decode_vs_one_shot_at_bad_rows() {
    let r = rig();
    let prompt = |n: usize| {
        (prompts().iter().find(|p| p.p == n)).unwrap_or_else(|| panic!("prompt {n} in meta.json"))
    };
    // Prompt 6's minimum-cosine row, from one fresh forward.
    let p6 = prompt(DIAG_P1_PROMPT);
    let got = {
        let mut w = r.worker();
        forward(&mut w, &p6.ids, 0, SEAT_A)
    };
    let (min_row, min_cos) = (rows(&got).zip(rows(&p6.f32)).enumerate())
        .map(|(row, (c, f))| (row, cosine(c, f)))
        .fold((0, f64::INFINITY), |m, x| if x.1 < m.1 { x } else { m });
    let bf16_at_min = cosine(
        &p6.bf16[min_row * VOCAB..][..VOCAB],
        &p6.f32[min_row * VOCAB..][..VOCAB],
    );
    eprintln!(
        "diag A prompt {} minimum CUDA row: row {min_row} (token {}) cosine(CUDA, f32)={min_cos:.6} \
         cosine(HF-bf16, f32) on that row={bf16_at_min:.6} (information only)",
        p6.p, p6.ids[min_row]
    );
    let rows_of = |want: usize| -> Vec<usize> {
        let mut v: Vec<usize> = (DIAG_A_ROWS.iter())
            .filter(|(q, _)| *q == want)
            .map(|&(_, row)| row)
            .collect();
        if want == DIAG_P1_PROMPT && !v.contains(&min_row) {
            v.insert(0, min_row);
        }
        v
    };
    for want in [DIAG_P1_PROMPT, DIAG_P3_PROMPT] {
        let p = prompt(want);
        let picks = rows_of(want);
        let mut c = DecodeCheck::new(&r, M, &p.ids);
        for row in picks {
            if row == 0 || row >= p.ids.len() {
                eprintln!(
                    "diag A prompt {want} row {row}: no prefix to prefill before it, SKIPPED \
                     (information only)"
                );
                continue;
            }
            c.prefill(M, &p.ids[..row]);
            let (first, count, decode) = c.decode(p.ids[row], row);
            let at = |v: &[f32]| v[row * VOCAB..][..VOCAB].to_vec();
            let (one_shot, f, b) = (at(&c.one_shot), at(&p.f32), at(&p.bf16));
            eprintln!(
                "diag A prompt {want} row {row} (token {}){}: cosine(decode, one-shot)={:.6} \
                 cosine(decode, f32)={:.6} cosine(one-shot, f32)={:.6} cosine(HF-bf16, f32)={:.6}; \
                 first layer whose decode picks differ from the one-shot's={first:?}, layers that \
                 differ={count} of {LAYERS} (information only)",
                p.ids[row],
                if want == DIAG_P1_PROMPT && row == min_row {
                    " [CUDA's minimum row]"
                } else {
                    ""
                },
                cosine(&decode, &one_shot),
                cosine(&decode, &f),
                cosine(&one_shot, &f),
                cosine(&b, &f)
            );
        }
    }
}

// ── #933's diagnosis, test (ii): the residual stream layer by layer ─

/// Where `cuda_hf_reference.py --hidden-states 3,6` writes its per-layer dump: a directory of its
/// own beside the goldens, so the goldens' files and their shas stay as they are.
const HIDDEN_GOLDENS: &str = "tests/goldens/gemma4-moe-gemma-4-26b-a4b-it-hidden";
/// The prompts test (ii) reads, and the rows it prints on each (the bad rows of
/// `/tmp/v933-diag3.md`; the prompt's worst row is added, and rows past a prompt's end are left
/// out).
const HIDDEN_PROMPTS: [usize; 2] = [DIAG_P3_PROMPT, DIAG_P1_PROMPT];
const HIDDEN_ROWS: [usize; 7] = [9, 25, 30, 37, 41, 71, 148];

/// One prompt's per-layer residual dump: `[LAYERS + 1, L, HIDDEN]` f32 of the f32 run, then of the
/// bf16 run (`cuda_hf_reference.py --hidden-states`'s `hidden_<p>.bin`).
fn read_hidden(p: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(HIDDEN_GOLDENS);
    let path = dir.join(format!("hidden_{p}.bin"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}. Make it first: python3 -I crates/models/arch/tests/cuda_hf_reference.py \
             --model {MODEL} --checkpoint <{CHECKPOINT} dir> --prompts \
             crates/e2e/testdata/prompts.txt --hidden-states 3,6 --out \
             crates/models/arch/{HIDDEN_GOLDENS}",
            path.display()
        )
    });
    let one = (LAYERS + 1) * n * HIDDEN;
    assert_eq!(
        bytes.len(),
        2 * one * 4,
        "hidden_{p}.bin: two runs of [{}, {n}, {HIDDEN}] f32",
        LAYERS + 1
    );
    let all: Vec<f32> = (bytes.as_chunks::<4>().0.iter())
        .map(|&q| f32::from_le_bytes(q))
        .collect();
    let (f32s, bf16s) = all.split_at(one);
    (f32s.to_vec(), bf16s.to_vec())
}

/// [`CUDA_TAPES`], then the m=512 tape cut just after each layer's `LayerScale` (the layer's last
/// step), its result that step's `out`, the layer's residual `[m, HIDDEN]`; then the same tape cut
/// after each layer's routing, its result `topk_idx`. Cut `l` of each kind is tape
/// `CUDA_TAPES.len() + l` and `CUDA_TAPES.len() + LAYERS + l`.
fn hidden_and_route_cuts() -> &'static [CudaTape] {
    use scratchy_target_cuda::tape::ArenaSpan;
    let tape = &CUDA_TAPES[bucket_of(M)];
    let cut = |last: usize, result: ArenaSpan| CudaTape {
        steps: Box::leak(tape.steps[..=last].to_vec().into_boxed_slice()),
        result,
        ..*tape
    };
    let moe = tape.moe.expect("a MoE scratch");
    let rows = u64::from(tape.bucket_m);
    let residual: Vec<CudaTape> = (tape.steps.iter().enumerate())
        .filter_map(|(i, s)| match *s {
            CudaStep::LayerScale(l) => Some(cut(
                i,
                ArenaSpan {
                    offset: l.out,
                    bytes: rows * u64::from(l.cols) * 2,
                },
            )),
            _ => None,
        })
        .collect();
    let route: Vec<CudaTape> = (tape.steps.iter().enumerate())
        .filter(|(_, s)| matches!(s, CudaStep::MoeRoute(_)))
        .map(|(i, _)| cut(i, moe.topk_idx))
        .collect();
    assert_eq!(residual.len(), LAYERS, "one layer scale per layer");
    assert_eq!(route.len(), LAYERS, "one routing per layer");
    let mut list = CUDA_TAPES.to_vec();
    list.extend(residual);
    list.extend(route);
    Box::leak(list.into_boxed_slice())
}

/// #933's diagnosis, test (ii) (`/tmp/v933-diag3.md` §3(ii)), printed only, nothing gated: for
/// prompts 3 and 6, CUDA's residual after each layer (the m=512 tape cut after that layer's
/// `LayerScale`, read back as bf16) against HF-f32's same layer output from
/// `cuda_hf_reference.py --hidden-states`, beside HF-bf16's against HF-f32. Per layer and per row
/// of [`HIDDEN_ROWS`] and the prompt's worst row (by final-logits cosine): cosine and RMS of CUDA
/// against HF-f32, the same of HF-bf16, their RMS ratio, and whether that row's CUDA picks differ
/// from HF-f32's at that layer; per layer, how many rows' picks differ. The first layer where a
/// row's ratio jumps names the layer to cut inside next.
#[test]
fn diag_hidden_per_layer() {
    let r = rig();
    let tapes = hidden_and_route_cuts();
    let mut w = CudaWorker::new(&r.device, tapes, &r.weights, Some(DECODE_KV_POOL), GDN_POOL)
        .unwrap_or_else(|e| panic!("CudaWorker::new over the tapes and the cuts: {e}"));
    let base = CUDA_TAPES.len();
    let top_k = (CUDA_TAPES[bucket_of(M)].moe).expect("a MoE scratch").top_k as usize;
    for want in HIDDEN_PROMPTS {
        let p = (prompts().iter().find(|p| p.p == want))
            .unwrap_or_else(|| panic!("prompt {want} in meta.json"));
        let n = p.ids.len();
        let (hf32, hbf16) = read_hidden(want, n);
        // The prompt's worst row by its final logits against HF-f32.
        let logits = run_on(&mut w, (tapes, bucket_of(M)), &p.ids, 0, &DECODE_SEAT_B);
        let worst = (0..n)
            .map(|row| {
                let c: Vec<f32> = (logits[row * VOCAB..][..VOCAB].iter())
                    .map(|&b| bf16::from_bits(b).to_f32())
                    .collect();
                (row, cosine(&c, &p.f32[row * VOCAB..][..VOCAB]))
            })
            .fold((0, f64::INFINITY), |m, x| if x.1 < m.1 { x } else { m });
        let mut rows: Vec<usize> = HIDDEN_ROWS.iter().copied().filter(|&r| r < n).collect();
        if !rows.contains(&worst.0) {
            rows.push(worst.0);
        }
        eprintln!(
            "diag (ii) prompt {want}: {n} rows, worst final-logits row {} (cosine {:.6}); rows \
             printed {rows:?} (information only)",
            worst.0, worst.1
        );
        let route = read_route_of(&format!("route_{want}.bin"), n, (LAYERS, top_k));
        for layer in 0..LAYERS {
            let out = run_on(&mut w, (tapes, base + layer), &p.ids, 0, &DECODE_SEAT_B);
            let cuda: Vec<f32> = (out[..n * HIDDEN].iter())
                .map(|&b| bf16::from_bits(b).to_f32())
                .collect();
            let picks = pick_words(&run_on(
                &mut w,
                (tapes, base + LAYERS + layer),
                &p.ids,
                0,
                &DECODE_SEAT_B,
            ));
            let differs = |row: usize| pick_set(&picks, row, top_k) != route.f32.set(n, layer, row);
            let flips = (0..n).filter(|&row| differs(row)).count();
            // HF's entry `layer + 1` is decoder layer `layer`'s output.
            let at =
                |v: &[f32], row: usize| v[((layer + 1) * n + row) * HIDDEN..][..HIDDEN].to_vec();
            for &row in &rows {
                let (c, f, b) = (
                    &cuda[row * HIDDEN..][..HIDDEN],
                    at(&hf32, row),
                    at(&hbf16, row),
                );
                let ((crms, _), (brms, _)) = (error(c, &f), error(&b, &f));
                eprintln!(
                    "diag (ii) prompt {want} layer {layer} row {row}: CUDA vs f32 cosine={:.6} \
                     RMS={crms:.4e}; HF-bf16 vs f32 cosine={:.6} RMS={brms:.4e}; RMS ratio \
                     CUDA/HF-bf16={:.3}; picks differ from f32's here={} (information only)",
                    cosine(c, &f),
                    cosine(&b, &f),
                    crms / brms.max(f64::MIN_POSITIVE),
                    differs(row)
                );
            }
            eprintln!(
                "diag (ii) prompt {want} layer {layer}: {flips} of {n} rows' picks differ from \
                 HF-f32's (information only)"
            );
        }
    }
}

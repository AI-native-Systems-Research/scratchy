// SPDX-License-Identifier: Apache-2.0
//! Whole-model prefill and decode forwards of Qwen3.6-35B-A3B's emitted CUDA tapes on the H100.
//!
//! The same checks (a) to (o) as qwen3.5-9b's `cuda_prefill_9b.rs` (the tests are
//! `common/cuda_prefill_body.rs`), on the 35B's tapes (AI-native-Systems-Research/scratchy#859),
//! all 40 layers with their mixture-of-experts blocks. The tapes are `CUDA_TAPES`, exactly as
//! `#[forward]` emitted them; the weights are synthetic, each buffer sized from the tapes'
//! `weight_bytes` and filled by what reads it, as for the 9b, and each layer's MoE bundle as the
//! GEMMs that read it: the router `[256, 2048]` and every expert's gate and up rows uniform in
//! ±1/√2048, its down rows in ±1/√512. The lm_head is untied, its own `lm_head.weight`. The input
//! is token ids, uniform over the vocabulary and seeded; padding rows are id 0.
//!
//! The 35B's numbers: 523 weight buffers (69.3 GB, uploaded from the host, where they are built
//! once per binary), then 3 worker allocations (the arena, the KV pool, the GDN pool); the arena
//! is the m=4096 tape's 3,617,371,392 bytes, of which the MoE scratch is 737,939,712. One rig
//! holds about 73 GB of the H100's 80 GB, and the host copy another 69.3 GB of RAM, so run this
//! binary with `--test-threads=1`.
//!
//! (n)'s cosine is printed, not gated: the 35B's floor is set by the user from this run
//! (`boundary_floor_gates`); its finiteness still gates. Every other check is the 9b's, at the
//! 9b's thresholds.
//!
//! Built only under `nvcc` and `qwen3.6-35b-a3b` (`required-features`), and needs an sm_90 GPU:
//! no GPU fails rather than skips. Every check prints its line before a test fails, so one run
//! reports every number.

// `pub`: this binary uses part of `common`, and nothing in it is dead code.
#[path = "../../../targets/cuda/tests/common/numerics.rs"]
pub mod numerics;

use scratchy_models::qwen3_5_moe::qwen3_6_35b_a3b::CUDA_TAPES;

include!("common/cuda_prefill_body.rs");

const HIDDEN: usize = 2048;
/// The weight buffers: the 523 the tapes read (30 GDN bundles, 40 MoE bundles, the untied
/// lm_head's among them).
const WEIGHTS: u64 = 523;
/// The worker's arena: the largest tape's, m=4096 (`cuda_tapes_35b`'s `ARENAS`).
const ARENA: u64 = 3_617_371_392;
/// (n)'s line prints against the 9b's 0.998 for reference only: for this MoE model the floor is
/// the user's (`boundary_floor_gates` is false), set from this run's printed cosines.
const BOUNDARY_DECODE_COSINE: f64 = 0.998;

// ── Diagnostic: where a teacher-forced decode leaves the one-shot row (#859) ──

use scratchy_target_cuda::step::{AttnPrefillBf16, MoeRouteBf16};
use scratchy_target_cuda::tape::MoeTape;

/// The teacher-forced decode the diagnostic follows: (k)'s step 1, 301 tokens prefilled and token
/// 301 decoded, against row 301 of (k)'s one 308-token forward.
const DIAG_P: usize = 301;

/// `tape` cut just after step `last`, its result `out`: a `'static` copy, as
/// `cuda_narrow_diag_9b`'s (#846). Every span stays in the same arena, so it validates as the
/// full tape does.
fn cut(tape: &CudaTape, last: usize, out: ArenaSpan) -> CudaTape {
    CudaTape {
        steps: Box::leak(tape.steps[..=last].to_vec().into_boxed_slice()),
        result: out,
        ..*tape
    }
}

/// One tape's diagnostic cuts: just after its first attention step (model layer 3), its result
/// that step's output; and just after each MoE layer's routing, its result `topk_idx` (the
/// router logits stay in the MoE scratch, read back after).
fn diag_cuts(tape: &CudaTape) -> (CudaTape, Vec<CudaTape>) {
    let moe = tape.moe.expect("a MoE scratch");
    let rows = u64::from(tape.bucket_m);
    let attn = (tape.steps.iter().enumerate())
        .find_map(|(i, s)| match *s {
            CudaStep::AttnPrefill(a) | CudaStep::AttnDecode(a) => Some((i, a)),
            _ => None,
        })
        .map(|(i, a): (usize, AttnPrefillBf16)| {
            let bytes = rows * u64::from(a.q_heads * a.head_dim) * 2;
            cut(
                tape,
                i,
                ArenaSpan {
                    offset: a.out,
                    bytes,
                },
            )
        })
        .expect("an attention step");
    let routes: Vec<CudaTape> = (tape.steps.iter().enumerate())
        .filter(|(_, s)| matches!(s, CudaStep::MoeRoute(MoeRouteBf16 { .. })))
        .map(|(i, _)| cut(tape, i, moe.topk_idx))
        .collect();
    (attn, routes)
}

/// i32 words from the u16 halves a result holds.
fn words(halves: &[u16]) -> Vec<i32> {
    (halves.as_chunks::<2>().0.iter())
        .map(|&[lo, hi]| (u32::from(lo) | u32::from(hi) << 16) as i32)
        .collect()
}

/// How many bf16 steps apart two values are (signed-magnitude ladder: `+0` and `−0` are 0 apart).
fn ulps_apart(a: u16, b: u16) -> u32 {
    let ladder = |x: u16| {
        let m = i32::from(x & 0x7FFF);
        if x & 0x8000 != 0 { -m } else { m }
    };
    ladder(a).abs_diff(ladder(b))
}

/// The diagnosis of #859's (k)/(n) drop, printed only (AI-native-Systems-Research/scratchy#859):
/// one teacher-forced decode at P = 301 against row 301 of one 308-token forward, read on tapes
/// cut inside the forward. (a) model layer 3's attention output row, decode against one-shot:
/// mismatched bits and max ulps. (b) per MoE layer, the token's `topk_idx` (as sets) and router
/// logits, decode against one-shot, and the first layer whose picks differ, with the one-shot
/// row's rank-8 minus rank-9 router-logit gap beside the logits' max |diff| there. Before each cut
/// decode the 301 tokens are prefilled again (fresh), because a cut decode writes the cache and
/// GDN state of the layers it ran. Asserted: only that every forward ran.
#[test]
fn k_diag_where_the_teacher_forced_decode_leaves_the_one_shot_row() {
    let r = rig();
    let (prefill, decode) = (&CUDA_TAPES[bucket(512)], &CUDA_TAPES[bucket(1)]);
    let (attn_512, routes_512) = diag_cuts(prefill);
    let (attn_1, routes_1) = diag_cuts(decode);
    assert_eq!(routes_512.len(), routes_1.len(), "MoE layers per tape");
    let layers = routes_512.len();
    // Buckets: 0 the full m=512 tape, 1 the full m=1 tape, 2 and 3 the attention cuts, then
    // the route cuts, m=512 at `4 + ℓ` and m=1 at `4 + layers + ℓ`.
    let mut list = vec![*prefill, *decode, attn_512, attn_1];
    list.extend(routes_512);
    list.extend(routes_1);
    let tapes: &'static [CudaTape] = Box::leak(list.into_boxed_slice());
    let mut w = CudaWorker::new(&r.device, tapes, &r.table, Some(KV_POOL), Some(GDN_POOL))
        .unwrap_or_else(|e| panic!("CudaWorker::new over the diagnostic tapes: {e}"));
    let run = |w: &mut CudaWorker<'_>, b: usize, ids: &[u32], prefix_len: u32, seat: Seat| {
        let tape = &tapes[b];
        let q = ids.len() as u32;
        let kv = KvSequence {
            block_table: seat.blocks,
            prefix_len,
            q_len: q,
        };
        let gdn = GdnSequence {
            slot: seat.slot,
            q_len: q,
            fresh: prefix_len == 0,
        };
        let mut out = vec![0u16; tape.result.bytes as usize / 2];
        w.forward(
            b,
            Input::TokenIds(&padded(tape, ids)),
            &positions(tape, prefix_len),
            Some(&kv),
            Some(&gdn),
            &mut out,
        )
        .unwrap_or_else(|e| panic!("bucket {b} (m={}): {e}", tape.bucket_m));
        out
    };
    let rows = sequence_rows((Q + DECODES) as usize);
    let p = DIAG_P;
    // One teacher-forced decode against the one-shot row: the full tapes, as (k teacher-forced).
    let one_shot = run(&mut w, 0, &rows, 0, SEAT_B);
    run(&mut w, 0, &rows[..p], 0, SEAT_A);
    let row = run(&mut w, 1, &rows[p..=p], p as u32, SEAT_A);
    let want = &one_shot[p * VOCAB..][..VOCAB];
    eprintln!(
        "(k-diag P={p}) logits: decode against row {p} of one 308-token forward: cosine={:.6} \
         max|diff|={:e} mismatched_bits={}",
        cosine(&row, want),
        max_diff(&row, want),
        row.iter().zip(want).filter(|(a, b)| a != b).count()
    );
    // (a) model layer 3's attention output row.
    let width = tapes[3].result.bytes as usize / 2;
    let a_one = run(&mut w, 2, &rows, 0, SEAT_B);
    let a_one = &a_one[p * width..][..width];
    run(&mut w, 0, &rows[..p], 0, SEAT_A);
    let a_dec = run(&mut w, 3, &rows[p..=p], p as u32, SEAT_A);
    let ulps = (a_dec.iter().zip(a_one))
        .map(|(&a, &b)| ulps_apart(a, b))
        .max()
        .unwrap_or(0);
    eprintln!(
        "(k-diag P={p}) (a) layer 3 attention output row ({width} values), decode against \
         one-shot: mismatched_bits={} max_ulps={ulps} max|diff|={:e}",
        a_dec.iter().zip(a_one).filter(|(a, b)| a != b).count(),
        max_diff(&a_dec, a_one)
    );
    // (b) per MoE layer, the picks and the router logits.
    // Each tape's own MoE scratch: the m=512 and m=1 tapes place their logits apart.
    let (moe_512, moe_1) = (
        prefill.moe.expect("the m=512 MoE scratch"),
        decode.moe.expect("the m=1 MoE scratch"),
    );
    let experts = moe_512.experts as usize;
    let top_k = moe_512.top_k as usize;
    let read_logits = |w: &mut CudaWorker<'_>, moe: MoeTape, rows: usize| {
        let mut l = vec![0u16; rows * experts];
        let span = ArenaSpan {
            offset: moe.logits.offset,
            bytes: (rows * experts * 2) as u64,
        };
        w.read_arena(span, &mut l).expect("read the router logits");
        l
    };
    let mut first = None;
    for layer in 0..layers {
        let idx_one = words(&run(&mut w, 4 + layer, &rows, 0, SEAT_B));
        let logits_one = read_logits(&mut w, moe_512, p + 1);
        let logits_one = &logits_one[p * experts..][..experts];
        run(&mut w, 0, &rows[..p], 0, SEAT_A);
        let idx_dec = words(&run(
            &mut w,
            4 + layers + layer,
            &rows[p..=p],
            p as u32,
            SEAT_A,
        ));
        let logits_dec = read_logits(&mut w, moe_1, 1);
        let set = |v: &[i32]| {
            v.iter()
                .copied()
                .collect::<std::collections::BTreeSet<i32>>()
        };
        let (s_one, s_dec) = (set(&idx_one[p * top_k..][..top_k]), set(&idx_dec[..top_k]));
        let mut sorted: Vec<f32> = (logits_one.iter())
            .map(|&b| bf16::from_bits(b).to_f32())
            .collect();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let gap = sorted[top_k - 1] - sorted[top_k];
        let diff = max_diff(&logits_dec, logits_one);
        let same = s_one == s_dec;
        eprintln!(
            "(k-diag P={p}) (b) layer {layer}: picks same={same} one-shot {s_one:?} decode \
             {s_dec:?}; router logits mismatched_bits={} max|diff|={diff:e}; one-shot rank-8 \
             minus rank-9 logit gap={gap:e}",
            logits_dec
                .iter()
                .zip(logits_one)
                .filter(|(a, b)| a != b)
                .count()
        );
        if !same && first.is_none() {
            first = Some((layer, gap, diff));
        }
    }
    match first {
        Some((layer, gap, diff)) => eprintln!(
            "(k-diag P={p}) (b) first layer whose picks differ: {layer}; there the one-shot \
             rank-8 minus rank-9 router-logit gap={gap:e} against the router logits' \
             max|diff|={diff:e} (gap <= diff: {})",
            gap <= diff
        ),
        None => eprintln!("(k-diag P={p}) (b) the picks agree at every one of the {layers} layers"),
    }
}

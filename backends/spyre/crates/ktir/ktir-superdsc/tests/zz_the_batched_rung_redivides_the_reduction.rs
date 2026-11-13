// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ **THE DECODE BUNDLE'S ARITHMETIC IS A FUNCTION OF THE RUNG WIDTH**, and that is why a
//! batched row is not bit-identical to the same row decoded alone. This file pins the two halves of
//! that localization so a batched divergence is never re-read as a KV-addressing fault again.
//!
//! ⛔ THE CARD MEASUREMENT IT EXISTS TO EXPLAIN (granite-3.1-2b fp8, 2026-09-20, `SCRATCHY_TOP2`).
//! EIGHT **IDENTICAL** PROMPTS — so every row's KV history, prefix mask, write slot and hole are
//! identical to solo's, every row's tokens are identical, and a cross-row leak is unobservable
//! because there is nothing to leak. One row's top-2 logit gap at its FIRST batched step:
//!
//! | rung | top-2 gap at pos 46 | core division of `q_proj` |
//! |------|---------------------|---------------------------|
//! | m=1  | 8.156250            | `out=32`                  |
//! | m=2  | 7.937500            | `mb=2, out=16`            |
//! | m=4  | 7.937500            | `mb=4, out=8`             |
//! | m=8  | 7.812500            | `mb=8, out=4`             |
//!
//! Three distinct values, deterministic per rung, **identical for every launch slot** (rows 1..7 of
//! the rung are bit-identical to each other at all 420 positions) and reproducible run to run (two
//! of three runs bit-identical everywhere). So: launch-slot position is arithmetically neutral,
//! padding rows are neutral, the page gather is neutral, the masked hole is neutral — the only thing
//! that moves a value is `mb`.
//!
//! ⭐ AND THAT IS WHAT MAKES BATCHED OUTPUT LOOK NONDETERMINISTIC. A request's decode steps are split
//! across rungs as the batch composition changes — it decodes at m=1 while its batch-mates are still
//! prefilling, then joins m=2, m=4, m=8 as admissions land. Each rung computes slightly different
//! values, so the row's output depends on the SEQUENCE OF RUNGS IT EXPERIENCED, and that sequence
//! moves with scheduling timing. MEASURED: of three identical width-8 runs, the two whose ramp-up
//! matched are bit-identical at every position, and the third differs from the first position at
//! which one row entered the batch a step earlier. A fixed divergence COUNT with a moving IDENTITY is
//! what that looks like from a sampled-token log, and it was read as a race three times.
//!
//! ⛔ WHAT IT COSTS: "a batched row must equal its bs=1 output token for token" is UNREACHABLE while
//! this holds. On the 420-token granite-2b probe the surviving divergences sit on logit margins of
//! **0.000000**, 0.156250 and 1.343750 — the first is an EXACT tie, where argmax is settled by which
//! index `max_by` saw last and ANY one-ulp difference flips the token. Meeting that gate needs ONE
//! work division for every rung, which is a throughput decision (and must still clear the LX-fit
//! refusals the row-count-dependent division exists to avoid) — not a KV, mask, hole, padding-row,
//! gather or scheduler fix.

use ktir_superdsc::superdsc_opspec::{Df, ItDim};
use ktir_superdsc::work::distribute_cores;

/// granite-3.1-2b's rmsnorm mean: reduce `hidden = 2048` fp16 columns per row. `out` is the SUMMED
/// axis, `mb` the launch's row count, `y` the vestigial trailing dim the rank-3 form carries.
fn rmsnorm_mean_dims(rows: u32) -> [ItDim; 3] {
    [
        ItDim {
            name: "mb",
            size: rows,
            is_reduction: false,
            is_stick: false,
            df: Df::Fp16,
        },
        ItDim {
            name: "out",
            size: 2048,
            is_reduction: true,
            is_stick: true,
            df: Df::Fp16,
        },
        ItDim {
            name: "y",
            size: 1,
            is_reduction: false,
            is_stick: false,
            df: Df::Fp16,
        },
    ]
}

/// granite-3.1-2b's seven per-layer projections as `(name, n, k)`. `in` is the reduction (K).
const PROJ: [(&str, u32, u32); 7] = [
    ("q_proj", 2048, 2048),
    ("k_proj", 512, 2048),
    ("v_proj", 512, 2048),
    ("o_proj", 2048, 2048),
    ("gate_proj", 8192, 2048),
    ("up_proj", 8192, 2048),
    ("down_proj", 2048, 8192),
];

fn matmul_dims(rows: u32, n: u32, k: u32) -> [ItDim; 3] {
    [
        ItDim {
            name: "mb",
            size: rows,
            is_reduction: false,
            is_stick: false,
            df: Df::Fp16,
        },
        ItDim {
            name: "out",
            size: n,
            is_reduction: false,
            is_stick: true,
            df: Df::Fp16,
        },
        ItDim {
            name: "in",
            size: k,
            is_reduction: true,
            is_stick: true,
            df: Df::Fp8,
        },
    ]
}

/// The division as a sorted list, and the split of one named axis (absent ⇒ `1`, i.e. unsplit).
fn division(dims: &[ItDim]) -> Vec<(&'static str, u32)> {
    let mut v: Vec<_> = distribute_cores(dims, 32).into_iter().collect();
    v.sort();
    v
}
fn split_of(dims: &[ItDim], axis: &str) -> u32 {
    distribute_cores(dims, 32).get(axis).copied().unwrap_or(1)
}

/// ⭐ HALF ONE: **the summation order is NOT what moves.** Every decode rung contracts K on ONE core
/// — `in` is unsplit at m=1 and at every batched rung — so no output element is ever a sum of
/// per-core partials, and the natural first explanation for "batch ≠ solo" (a reduction split `r`
/// ways at one rung and `r'` at another gives a different addition order) is REFUTED, not assumed.
///
/// ⛔ RECORDED BECAUSE IT WAS PROPOSED AND WRONG. Reading `divide_and_time_tile_for_lx`'s doc — "the
/// `r` cores each contract a K-slice into a partial product dxp PSUM-accumulates" — plus its measured
/// note that the repair fires when "the activation slab grows" with the row count reads exactly like
/// a row-count-dependent reduction split. Evaluating the splitter says otherwise for every granite-2b
/// shape: that repair path is never taken here. A doc comment is not evidence; this is.
///
/// Fail-first: set `("down_proj", 2048, 8192)`'s `k` to a width whose one-stick weight slab does not
/// fit `USABLE_LX_BYTES` and the LX repair splits `in`, breaking this.
#[test]
fn no_decode_rung_splits_its_k_contraction_so_summation_order_is_not_the_variable() {
    for rung in [1u32, 2, 4, 8, 16, 32] {
        let d = rmsnorm_mean_dims(rung);
        assert_eq!(
            split_of(&d, "out"),
            1,
            "the rmsnorm mean over 2048 fp16 columns is split {} ways across cores at rows={rung} — \
             its partials are then summed in a different order than at another rung, and THAT would \
             explain a batched row differing from a solo row. Re-open the summation-order reading.",
            split_of(&d, "out"),
        );
        for (name, n, k) in PROJ {
            let d = matmul_dims(rung, n, k);
            assert_eq!(
                split_of(&d, "in"),
                1,
                "{name} (n={n} k={k}) splits its K contraction {} ways at rows={rung}: its output \
                 elements are PSUM sums of per-core partials, so this op's rounding is rung-dependent \
                 by summation order. That is a DIFFERENT (and fixable-by-pinning) cause than the one \
                 this file pins — re-localize before trusting the doc above.",
                split_of(&d, "in"),
            );
        }
    }
}

/// ⭐⭐⭐ HALF TWO: **the division that DOES move is `mb`/`out`.** This is the quantity the card
/// measurement in this file's header tracks: m=1 divides one row's output 32 ways; m=8 gives each of
/// 8 rows 4 output slices. Same K contraction, different per-core output tile — and three distinct
/// logit values on the card.
///
/// ⛔ SO THE LAW IS: **a decode row's numerics are a property of the rung it ran on.** Any gate that
/// compares a batched row against a bs=1 row is comparing two different divisions of the same
/// arithmetic, and the residual it reports is this, not a KV fault. Pinned as a test rather than a
/// comment because three separate hunts spent rounds attributing this residual to addressing.
///
/// Fail-first: make `distribute_cores` ignore `mb` (hand every rung the m=1 division) and this
/// breaks — which is also exactly the change that would make a batched row solo-exact, at whatever
/// the throughput cost of one division for every width turns out to be.
#[test]
fn the_output_division_of_every_projection_moves_with_the_rung_width() {
    let mut moved = Vec::new();
    for (name, n, k) in PROJ {
        let solo = division(&matmul_dims(1, n, k));
        let solo_out = split_of(&matmul_dims(1, n, k), "out");
        for rung in [2u32, 4, 8] {
            let d = matmul_dims(rung, n, k);
            eprintln!(
                "{name:10} n={n:5} k={k:5}  m=1 -> {solo:?}   m={rung} -> {:?}",
                division(&d)
            );
            if split_of(&d, "out") != solo_out || split_of(&d, "mb") != 1 {
                moved.push((name, rung));
            }
        }
    }
    assert_eq!(
        moved.len(),
        PROJ.len() * 3,
        "only {} of {} (projection, rung) pairs re-divide their output against the m=1 division. If a \
         projection now divides IDENTICALLY at m=1 and at a batched rung, that rung's arithmetic may \
         have become bit-identical to solo's — re-run the identical-prompt probe (`p_id_8` at \
         admit=1 vs admit=8 under `SCRATCHY_TOP2`) and update the header table rather than relaxing \
         this number.",
        moved.len(),
        PROJ.len() * 3,
    );
}

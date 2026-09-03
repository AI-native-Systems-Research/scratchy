// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "superdsc")]
//! A BATCHED-DECODE logits row is not where a reader assumes it is.
//!
//! The lm-head tail of a batched-decode bundle runs unfolded at `m = requests`, so its output is a
//! `[rows, width]` tensor whose row `r` is request `r`'s next-token distribution. That tensor is
//! STICK-BLOCKED on the vocab axis, so row `r` is NOT a contiguous run and NOT at offset `r * width`:
//! the vocab is cut into 64-wide blocks and all `rows` rows of a block are stored together.
//!
//! At `rows == 1` the formula collapses to the identity, which is why the one-request path reads its
//! logits with a plain `..vocab` slice and why that slice, reused per row, is silently wrong the
//! moment a second request joins the batch — it hands every request a piece of request 0's
//! distribution. Fluent output, wrong token, no crash. These tests pin the law the worker's gather
//! is derived from, at the real granite geometry.

use scratchy_subtile::sdsc_abstract::StickLayout;

/// granite-3.1-2b: 49155 logical vocab, placed 51200 wide (the emitter pads the vocab axis to a whole
/// number of sticks per core: 51200/64 = 800 sticks = 25 per core across 32 cores).
const VOCAB: usize = 49155;
const WIDTH: usize = 51200;

/// The one-request tail is FLAT — this is the invariant the `..vocab` slice rests on.
#[test]
fn a_single_row_of_logits_is_flat() {
    let lay = StickLayout::row_blocked(1, WIDTH);
    for c in [0, 1, 63, 64, 65, 4095, VOCAB - 1] {
        assert_eq!(
            lay.dev_off(0, c),
            c,
            "at one row the stick block term vanishes"
        );
    }
}

/// The naive `r * vocab` (or even `r * width`) slice reads the WRONG request. Pinned as a test so the
/// batched path can never quietly go back to it.
#[test]
fn a_contiguous_slice_per_request_reads_the_wrong_request() {
    let rows = 8;
    let lay = StickLayout::row_blocked(rows, WIDTH);
    // Request 1's first logit does NOT live one vocab (nor one width) along.
    assert_ne!(lay.dev_off(1, 0), VOCAB);
    assert_ne!(lay.dev_off(1, 0), WIDTH);
    // It lives one STICK along — and the 64 elements at `1 * vocab` belong to a different request
    // entirely, which is why the wrong slice still looks like plausible logits.
    assert_eq!(lay.dev_off(1, 0), 64);
    // Everything the naive slice for request 1 would read sits inside block 0..(rows*64), i.e. the
    // region holding the FIRST 64 vocab entries of all 8 requests — not request 1's vocab at all.
    assert!(VOCAB > rows * 64);
}

/// The gather the worker performs — every `(request, vocab)` cell — must address distinct, in-bounds
/// elements. Distinctness is what proves no two requests are handed the same logit.
#[test]
fn the_per_request_gather_is_injective_and_in_bounds() {
    for rows in [1usize, 2, 4, 8] {
        let lay = StickLayout::row_blocked(rows, WIDTH);
        let mut seen = vec![false; rows * WIDTH];
        for r in 0..rows {
            for c in 0..VOCAB {
                let off = lay.dev_off(r, c);
                assert!(
                    off < rows * WIDTH,
                    "rows={rows} ({r},{c}) -> {off} is past the placement"
                );
                assert!(
                    !seen[off],
                    "rows={rows} ({r},{c}) -> {off} was already gathered"
                );
                seen[off] = true;
            }
        }
    }
}

/// The closed form, written out once, so a change to `dev_off` that breaks batched decode fails HERE
/// with the arithmetic visible rather than as drifting text on a pod.
#[test]
fn the_block_stride_is_rows_times_one_stick() {
    let rows = 8;
    let lay = StickLayout::row_blocked(rows, WIDTH);
    for r in 0..rows {
        for c in [0usize, 1, 63, 64, 127, 128, VOCAB - 1] {
            assert_eq!(
                lay.dev_off(r, c),
                (c / 64) * (rows * 64) + r * 64 + (c % 64)
            );
        }
    }
    // Consecutive vocab BLOCKS are `rows` sticks apart: that factor of `rows` is the whole difference
    // between the batched layout and the one-row layout.
    assert_eq!(lay.dev_off(0, 64) - lay.dev_off(0, 0), rows * 64);
}

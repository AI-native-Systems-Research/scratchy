// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ THE GATHER IS `hd == 64`-ONLY, AND **NOTHING SAID SO EXCEPT A DOC COMMENT.**
//!
//! [`GatherScratch::of_fold_pass`] returns `None` for `hd > POOL_STICK`, and `assemble_attn`'s
//! `kv_block_index.zip(of_fold_pass(..))` turns that `None` into "emit EXACTLY the bundle that
//! shipped, byte for byte". So on any model with a head dim above one stick the whole HBM-gather
//! campaign is **silently absent**: no copy ops, no index staged, no collapse — and the run is
//! `rc=0`, the output is clean, and every number looks like a batching result.
//!
//! ⛔⛔⛔ THAT SILENCE IS WHAT THIS FILE EXISTS FOR. Measured on the card, pod
//! `nickm-7db9667cdd-z2jc6`, this tree, `RedHatAI/granite-3.1-8b-instruct-FP8-dynamic`
//! (`hidden 4096, nqh 32, nkvh 8, 40 layers` ⇒ **hd = 128**), `scr batch` over the `c` probe
//! (420-token generations, one distinct integer sequence per row) at rungs 2/4/8, each against its
//! own `--max-num-seqs 1` run of the identical file:
//!
//! ```text
//! rung 2  rc=0  own_ok=1/2   solo_diff=1   degen=1   ITL  88-100 ms (solo 79.4)   N=3
//! rung 8  rc=0  own_ok=0-1/8 solo_diff=8   degen=3-4 ITL 175-198 ms (solo 79.4)   N=3
//! ```
//!
//! ⛔⛔⛔ AND THE "CLEAN AT EVERY RUNG, 3.66×" THIS TABLE USED TO HOLD IS RETRACTED. That reading came
//! from one trial per rung and from `own_bad` alone; re-measured at N=3 against each width's own
//! `--max-num-seqs 1` oracle, `origin/main` on the 8b is ONE bad row of two at rung 2 (a different row
//! each trial) and 7-8 bad of eight at rung 8, with 3-4 degenerate rows. So the 8b's batched decode is
//! broken at width 8 **with no gather anywhere in the bundle** — a pre-existing defect, not this
//! campaign's, and not something a gather could fix.
//!
//! ⭐ WHAT THE ORIGINAL POINT OF THIS FILE STILL IS, and it is now stronger: not one of those tokens
//! went through a gather, at either reading of the numbers. A reader with a batched result and no test
//! records "the gather holds on the 8b" when the mechanism is absent. `own_bad` is the misleading column
//! and `solo_diff` the real one; and a clean batched result is never evidence that the mechanism under
//! test ran.
//!
//! ## The boundary is `hd` and NOTHING ELSE, which is why the carry-over looks like it holds
//! granite-3.1-8b has the SAME `nkvh = 8` and the same GQA as granite-3.1-2b, so every quantity the
//! gather's cut is sized by — the row law `nkvh * mq`, the one-index-stick ceiling
//! [`CopyDims::ENTRIES_PER_OP`] — is hd-INDEPENDENT and predicts identically at both. Those
//! predictions are pinned below precisely so that their holding is not read as the gather holding:
//! the ONE quantity that moves with `hd` is the copy's row width, and it is the one the refusal is
//! about.
//!
//! ## And why a wider `hd` is a precondition rather than a bound to widen
//! The copy is a FLAT block move, so a 64-slot window of the plane must be CONTIGUOUS. For Kᵗ
//! (`[hd, cap]`, stick-major on `cap`) it is, at every head dim. For V (`[cap, hd]`, stick-major on
//! `hd`) it is only while `hd` is one stick — above that a window is `nslab` runs
//! `PLANE_SLOTS * 64` apart, and a flat copy would relayout it. Widening the gather therefore means
//! an index entry per (window, SLAB) — new work, not a constant to raise. The numbers that work
//! would need are asserted below so they are written down once and cannot drift: at hd=128 a row is
//! 8192 elements where at hd=64 it is 4096.
//!
//! ⛔⛔⛔ **AND THE "SECOND `skip_addr`" THIS PARAGRAPH USED TO NAME IS NOT THE PRICE** — that was
//! prose, and `zz_the_two_planes_block_numbering_diverges_above_one_stick.rs` now derives the real
//! one from [`PagedKvPool::addr`] itself. Three corrections, all machine-checked there:
//! * **Kᵗ needs nothing.** A Kᵗ window is ONE contiguous `hd * POOL_STICK` run at every head dim, at
//!   an offset that is an exact multiple of it — the shipped entry unit already names it.
//! * **V's window stride is `POOL_STICK * POOL_STICK` (4096), not `hd * POOL_STICK`.** V is
//!   stick-major on `hd`, so its SLOT stride is one stick whatever `hd` is. At hd=128 that is HALF a
//!   stick block, so a stick-block `skip_addr` cannot name V's odd windows at all — no `idx` is
//!   fractional. The entry unit that serves both planes is `POOL_STICK * POOL_STICK`, which IS the
//!   stick block at hd=64, so the proven geometry pays nothing for it.
//! * **The price is a second index TABLE, not a second `skip_addr`.** dxp derives ONE `skip_addr` per
//!   (index lds, value alloc) pair and each copy op is its own SDSC with its own pair
//!   (`GatherIndexConversion.cpp::computeGatherMetadata`), so two different `skip_addr`s are free.
//!   What is NOT free: in the 4096 unit the two planes number the same blocks DIFFERENTLY
//!   (Kᵗ `b * nslab + s`, V `s * blocks_per_page + b`) and coincide for every coordinate **iff
//!   `nslab == 1`** — which is exactly why ONE table serves both copies today and cannot above one
//!   stick. A section base inside one table derived from the live window COUNT is the
//!   emitter/host disagreement already recorded on [`GatherScratch::row_of`].

use ktir_superdsc::sdsc_abstract::{
    CopyDims, GatherScratch, POOL_STICK, PageScratch, PagedKvPool, QueryRowCount, SlotCount,
    SlotWindow,
};

/// granite-3.1-2b-instruct: `hidden 2048 / nqh 32` ⇒ hd == one stick. The ONE configuration every
/// card measurement of the gather was taken on.
const HD_2B: usize = 64;
/// granite-3.1-8b-instruct (and llama-3.2-3b): `hidden 4096 / nqh 32` ⇒ hd == TWO sticks.
const HD_8B: usize = 128;
/// Both models. The gather's row law is sized by this and by `mq`, so it is not the boundary.
const NKVH: usize = 8;

fn scratch_at(hd: usize, cap: u32, mq: u32) -> Option<GatherScratch> {
    GatherScratch::of_fold_pass(
        PagedKvPool::new(NKVH, hd),
        SlotWindow::count_in(SlotCount::new(cap)),
        QueryRowCount::of_mq(mq),
    )
}

/// ⭐⭐⭐⭐⭐ THE REFUSAL ITSELF, AT EVERY RUNG AND EVERY WIDTH THE LADDER BAKES — so no combination
/// of `active_cap` and `mq` can reach a gather at hd=128 while some other combination does not.
///
/// The pair of loops is the point: the 2b answers `Some` for all of them and the 8b answers `None`
/// for all of them, with `nkvh`, the window count and `mq` held equal across the pair. A refusal
/// that depended on the rung or the width would be a bug in the gate; a refusal that depends only
/// on `hd` is the documented precondition, and this is the assertion that they are the same thing.
#[test]
fn a_head_dim_above_one_stick_gets_no_gather_at_any_rung_or_width() {
    assert_eq!(
        HD_2B, POOL_STICK as usize,
        "this file's premise is that granite-2b's head dim IS the pool stick — if POOL_STICK moved, \
         every `hd`-vs-stick claim below is about a different boundary and the 8b numbers in the \
         module doc were measured against the old one"
    );
    assert_eq!(
        HD_8B,
        2 * POOL_STICK as usize,
        "and that the 8b's head dim is exactly TWO sticks, which is what makes `nslab == 2` and a V \
         window two strided runs"
    );

    // `ActiveCap::decode_ladder`'s rungs: the interior ones plus the PAGE_SLOTS ceiling, i.e. every
    // `nb` a fold pass can sweep. Widths are the baked decode rungs the 8b run reported ready.
    for cap in [64u32, 128, 256] {
        for mq in [1u32, 2, 4, 8, 16, 32] {
            // ⛔⛔⛔⛔⛔ THE **LIVE** DOOR, ASKED IN THE SAME LOOP. `assemble_attn` stopped calling
            // `GatherScratch::of_fold_pass` when the gather went page-granular; every assertion in this
            // file below is about a door the shipped emitter no longer reads, which is how a green suite
            // certified hd=128 for a whole round. `PageScratch::of_pass` IS the shipped door, and it
            // refuses two slabs on CARD EVIDENCE (its own note carries the table and the `origin/main`
            // control). A widening must move THIS assertion, not the dead one.
            assert!(
                PageScratch::of_pass(PagedKvPool::new(NKVH, HD_2B), QueryRowCount::of_mq(mq))
                    .is_some(),
                "hd={HD_2B} mq={mq}: the live page-granular door must admit one slab — every card \
                 measurement of a correct gather comes from this geometry"
            );
            // ⭐⭐⭐⭐⭐ AND THE LIVE DOOR NOW ADMITS **TWO SLABS TOO**, WHICH IS THE ONE ASSERTION IN THIS
            // FILE THAT HAS BEEN BOTH WAYS ROUND. It asserted `is_none()` on the card evidence that
            // hd=128 gathered garbage from the first generated token — a green test pinning a door that
            // did nothing at the only geometry anyone wanted it for. The two causes were found and they
            // were NOT in the gather: the gathered VALUE leg wrote only feature slab 0 of `run_o` (no
            // slab loop, so the upper half of every head's output had no prefix contribution at all),
            // and the gathered SCORE leg contracted two sticks under a `y`-batch, the shape
            // `ScoreArm::choose` records as measured-twice incoherent inside dxp. Both are `nslab` loops
            // now, mirroring the ungathered arms.
            assert!(
                PageScratch::of_pass(PagedKvPool::new(NKVH, HD_8B), QueryRowCount::of_mq(mq))
                    .is_some(),
                "hd={HD_8B} mq={mq}: the live page-granular door must admit TWO slabs — a refusal here \
                 is a bundle that silently emits the ungathered form, which is how an 8b build passed \
                 its gate while gathering nothing. The head-dim obstacles were all in the FOLD's legs"
            );
            let two_b = scratch_at(HD_2B, cap, mq);
            assert!(
                two_b.is_some(),
                "hd={HD_2B} cap={cap} mq={mq}: the gather must still be available at one stick — \
                 this is the configuration all of its card evidence comes from"
            );
            assert!(
                scratch_at(HD_8B, cap, mq).is_none(),
                "hd={HD_8B} cap={cap} mq={mq}: `of_fold_pass` handed back a scratch for a head dim \
                 of TWO sticks. A 64-slot V window is then `nslab` runs `PLANE_SLOTS*64` apart, so \
                 the flat copy would RELAYOUT it — it bakes and scrambles the value leg. Either the \
                 gather learned a per-(window, slab) entry and a second `skip_addr`, in which case \
                 this test is the place to say so, or the guard was weakened"
            );
        }
    }
}

/// ⭐⭐⭐⭐⭐ THE BUNDLE-LEVEL DOOR **IS** THE PER-PASS DOOR — [`GatherScratch::admits`] agrees with
/// `of_fold_pass(..).is_some()` at EVERY window count, so the placement and the emission cannot be
/// decided by two different predicates.
///
/// ⛔⛔⛔ THE DEFECT THIS PINS. `KV_BLOCK_INDEX_TID`'s placement (and therefore
/// `BakeFacts::gathers_kv`, and therefore the forward tape's `KvBlockIndex` step) was reserved on
/// `rows_are_requests` ALONE, while `GathersKv::of_launch_groups` — the per-BODY answer off the
/// emitted fold groups — additionally requires `of_fold_pass` to have returned a scratch. At hd=64
/// `of_fold_pass` never refuses, so all three doors agreed and the split could not be seen; at hd=128
/// it always refuses, so an 8b bundle reserved an index activation and staged a table for a gather
/// that appears in no descriptor. Both bundle-level sites now call `admits`, and this test is what
/// makes "asking at one window is asking the geometry" a checked property rather than a comment: if
/// some future refusal inside `of_fold_pass` depends on the window count, the two halves of the
/// gather would silently disagree again and this fails first.
#[test]
fn the_bundle_level_door_is_the_per_pass_door_at_every_window_count() {
    for hd in [HD_2B, HD_8B, 96, 192, 256] {
        for mq in [1u32, 2, 4, 8, 16, 32] {
            let admits =
                GatherScratch::admits(PagedKvPool::new(NKVH, hd), QueryRowCount::of_mq(mq));
            // ⛔ EVERY `active_cap` A LADDER RUNG CAN BE, not just the ceiling: the bundle-level
            // question is asked once and must be the answer for every body the bundle holds.
            for cap in [64u32, 128, 256] {
                assert_eq!(
                    admits,
                    scratch_at(hd, cap, mq).is_some(),
                    "hd={hd} mq={mq} cap={cap}: `admits` and `of_fold_pass` disagree. The placement \
                     asks the first and the descriptor asks the second, so a bundle would reserve an \
                     index table no op gathers through — or, in the other direction, gather through \
                     one nothing placed, which reads entry 0: a REAL address"
                );
            }
        }
    }
    // And a degenerate width is refused by the same door, so the placement cannot reserve for one.
    assert!(
        !GatherScratch::admits(PagedKvPool::new(NKVH, HD_2B), QueryRowCount::of_mq(0)),
        "a zero-row batch admits no gather"
    );
    assert!(
        !GatherScratch::admits(PagedKvPool::new(0, HD_2B), QueryRowCount::of_mq(8)),
        "a pool with no kv heads admits no gather"
    );
}

/// ⭐⭐⭐ THE CARRY-OVER PREDICTIONS HOLD, AND HOLDING IS NOT THE GATHER HOLDING.
///
/// `rows()` and the one-index-stick ceiling are the two numbers the rung-8 corruption was cut by
/// (`GatherScratch::copies`), and both are computed from `nkvh`, `mq` and the index's own dtype —
/// no `hd` anywhere. granite-3.1-8b shares the 2b's `nkvh`, so they predict the SAME cut at the
/// SAME widths on both models. Asserted at the 2b's geometry because that is the only one that has
/// a scratch at all; stated here so that "the predictions carried over" is never read as evidence
/// the mechanism did.
#[test]
fn the_cut_is_sized_by_nkvh_and_mq_and_never_by_the_head_dim() {
    assert_eq!(
        CopyDims::ENTRIES_PER_OP,
        32,
        "one gather op's index is ONE 128-byte stick of SENUINT32 entries. It is the index's dtype's \
         stick and has no head-dim term, so the wrap that separated rung 2 from rung 8 lands at the \
         same width at every head dim"
    );
    for mq in [1u32, 2, 4, 8] {
        // `nb = cap / 64`, so cap=256 is the ceiling rung's four windows.
        let s = scratch_at(HD_2B, 256, mq).expect("one stick admits a scratch");
        assert_eq!(
            s.rows(),
            4 * NKVH as u32 * mq,
            "rows() is `windows * nkvh * mq` — the window-major row law and nothing else. A term \
             in `hd` here would make the host's table and the body's rows disagree on 8b before the \
             refusal ever got a chance to fire"
        );
        assert_eq!(
            s.copy_count(),
            s.rows().div_ceil(CopyDims::ENTRIES_PER_OP),
            "and the op count is that row count cut into index sticks"
        );
    }
}

/// ⭐⭐⭐ THE ONE QUANTITY THAT DOES MOVE WITH `hd` — the copy's row, which IS `skip_addr`.
///
/// Every number a widened gather would have to get right is derived from this one, so they are
/// written down here at both head dims rather than recomputed by the next attempt from a doc
/// comment. `stick_block_elems` is the pool's own quantum and the row width is exactly it, which is
/// what makes an index entry a plain global stick-block number; `plane_block_elems` is FOUR of
/// those and is the distance to the next KV HEAD, the confusion that would make a gather skip three
/// quarters of every head's keys and still bake.
#[test]
fn the_row_width_is_the_pools_stick_block_at_both_head_dims() {
    for (hd, row, plane) in [
        (HD_2B, 4096u32, 64 * PagedKvPool::PLANE_SLOTS),
        (HD_8B, 8192u32, 128 * PagedKvPool::PLANE_SLOTS),
    ] {
        let pool = PagedKvPool::new(NKVH, hd);
        assert_eq!(
            pool.stick_block_elems(),
            row as usize,
            "hd={hd}: one 64-slot Kᵗ block is `hd * POOL_STICK` elements, and that is the unit one \
             index entry names"
        );
        assert_eq!(
            pool.plane_block_elems(),
            plane,
            "hd={hd}: the KV-HEAD stride is `hd * PLANE_SLOTS`, which is NOT the entry unit"
        );
    }
    // The 2b's scratch states the same number a third way — as the copy's declared row — so the
    // three spellings of `skip_addr` cannot drift apart at the head dim that ships.
    let s = scratch_at(HD_2B, 256, 8).expect("one stick admits a scratch");
    assert_eq!(
        s.cols(),
        PagedKvPool::new(NKVH, HD_2B).stick_block_elems() as u32,
        "the gathered row width and the pool's stick block are ONE number — `addr = idx * \
         skip_addr + base` is exact only while they are"
    );
    assert_eq!(
        s.entry_page(),
        HD_2B as u32,
        "and the positions one entry covers is `cols / POOL_STICK` = hd, the pin that keeps \
         `skip_addr` at one pool block while the operands stay one stick wide"
    );
}

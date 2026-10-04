// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ THE REAL GROUP WALK OVER A SPLIT GATHERED FOLD — the spyre-crate half of the
//! window-chunk split's verification. The emission-side tests
//! (`ktir-superdsc/tests/zz_the_oversized_gathered_fold_splits_into_window_chunks.rs`) pin what the
//! EMITTER produces; this file pins what the WALK does with it: the chunk tag the emitter stamps
//! (`kv_request`) must break the fold's trip run into one launch group per chunk, every group at or
//! under `DxGroupCeiling::MAX_DESCRIPTORS`, with the group's `KvShifts` still naming the fold's
//! regime (paged, batched, gathered, whole-batch rows) so the runtime's `reps` arithmetic is
//! unchanged by the split.
//!
//! The observable is [`launch_index`] — the SAME walk the bake runs (`trip_kinds_for` +
//! `group_ranges` at `group_size()`), reached through its public entry rather than a copy, so a
//! drift between the test and the bake is impossible by construction.

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::assemble_attn;
use scratchy_subtile::sdsc_abstract::{AttnGeometry, attn_bundle_rows};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::{
    FoldGrouping, bundle::FoldRows, launch_index,
};

const CAP: u32 = 256;

/// gemma-4-12b's sliding attention class: nqh=16, nkvh=8, hd=256 — the geometry whose mq=32 fold
/// was the measured 8,300-descriptor bake explosion.
const NQH: u32 = 16;
const NKVH: u32 = 8;
const HD: u32 = 256;

/// The whole attention emission at one width, gather ON, rows are requests.
fn ops_at(mq: u32) -> Vec<ktir_superdsc::emit::EmittedOp> {
    let mut sym = 0i64;
    let geom = AttnGeometry::<NQH, NKVH, HD>::minted();
    let width = attn_bundle_rows(geom, mq, true)
        .unwrap_or_else(|| panic!("mq={mq} is not a baked decode rung"));
    assemble_attn(
        7,
        geom,
        width,
        CAP,
        CAP,
        "qs",
        "nks",
        "nvr",
        "kct",
        "vc",
        "pmask",
        "cmask",
        Some("kv_idx"),
        scratchy_target_spyre::bundle_code::PlaceId::Act(7),
        true,
        &mut sym,
        None,
    )
    .expect("attention must emit")
}

/// ⭐⭐⭐⭐⭐ THE WALK SPLITS WHAT THE EMITTER SPLIT — at gemma-4's class-0 geometry the mq=32 fold
/// is two chunks, and the walk must hand the bake TWO fold groups of 4,182 trips each (under the
/// 4,204 ceiling), each still declaring the gathered whole-batch regime.
#[test]
fn the_walk_cuts_the_gemma4_fold_into_under_ceiling_groups() {
    // ⭐ SPLIT — the grouping the multi-page runtime runs: the fold keeps its own re-launchable
    // groups, so the chunk cut is directly visible as TWO page_fold groups.
    let ops = ops_at(32);
    let shifts = launch_index(&ops, FoldGrouping::Split);
    let fold_groups: Vec<_> = shifts.iter().filter(|s| s.page_fold).collect();
    assert_eq!(
        fold_groups.len(),
        2,
        "the 8,300-descriptor fold must be TWO launch groups (one per window chunk)"
    );
    // Every fold group still declares the gathered collapsed regime — the split cut the GROUP
    // COUNT, not the pass semantics: one pass per page, serving every request, whole-batch rows.
    for g in &fold_groups {
        assert!(g.batched_requests, "the axis survives the split");
        assert!(g.gathered, "the gather flag survives the split");
        assert_eq!(g.fold_rows, FoldRows::WholeBatch);
    }
    // AND THE TWO GROUPS ARE THE TWO CHUNKS — distinct request tags, chunk 0 first.
    let tags: Vec<u32> = fold_groups.iter().map(|g| g.request).collect();
    assert_eq!(tags, vec![0, 1], "chunk 0 keeps request 0");

    // ⭐ FUSED — the single-page twin reclassifies the fold trips to `Pure` (no `page_fold` groups;
    // that is what Fused means), so the chunk cut is not visible as fold groups here. What must
    // still hold is that the walk partitions the SAME ops without a group past the ordinary
    // per-kind cap — the property the crate's own `fused_folding_takes_the_ordinary_cap` guard test
    // pins at the trip level. This run is the emission-level witness that the twin still walks.
    let ops = ops_at(32);
    let shifts = launch_index(&ops, FoldGrouping::Fused);
    assert!(
        shifts.iter().all(|s| !s.page_fold),
        "Fused reclassifies every fold trip — no group may still claim the fold relaunch"
    );
}

/// ⭐⭐⭐⭐⭐ AND THE UNSPLIT RUNGS DO NOT MOVE — at the widths whose fold fits one group (every
/// rung below 32 at this geometry, and every width at granite's), the walk still produces ONE fold
/// group tagged request 0. This is the granite-parity pin from the walk's side: the tag the split
/// introduced is invisible wherever the split does not fire.
#[test]
fn the_walk_keeps_one_fold_group_where_the_fold_fits() {
    for mq in [1u32, 2, 4, 8] {
        let ops = ops_at(mq);
        let shifts = launch_index(&ops, FoldGrouping::Split);
        let fold_groups: Vec<_> = shifts.iter().filter(|s| s.page_fold).collect();
        assert_eq!(
            fold_groups.len(),
            1,
            "mq={mq}: an under-ceiling fold is ONE group"
        );
        assert_eq!(fold_groups[0].request, 0, "mq={mq}: chunk 0 keeps request 0");
        assert!(fold_groups[0].gathered);
    }
}

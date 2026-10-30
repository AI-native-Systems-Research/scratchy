// SPDX-License-Identifier: Apache-2.0
//! granite's two-rank world, through the real `#[forward]` expansion.
//!
//! The spyre and subtile crates prove the sharding analysis and the collective emission on
//! hand-built graphs; this proves what granite's own bake ships: ONE sharded world, world 2, with
//! its own decode bundle and wiring, and the slice law each source loads by — derived by the
//! analysis, never declared. Granite's 40 layers each carry two K-split projections (o and down),
//! each followed by the all-reduce the analysis wove in, so exactly 80 sources load by ROWS; its
//! odd vocab (49155) does not split into whole sticks, so the lm head stays whole and no
//! all-gather is needed.

#![cfg(all(feature = "spyre", feature = "granite-3.1-2b-instruct"))]

use scratchy_target_spyre::wiring::SourceShard;

#[test]
fn granite_ships_one_two_rank_world() {
    let model = &scratchy_models::granite::granite_3_1_2b_instruct::SUPERDSC_TP;
    assert_eq!(model.len(), 1, "granite bakes exactly one sharded world");
    let tp = &model[0];
    assert_eq!(tp.world.get(), 2);
    assert!(!tp.decode.fp.is_empty(), "the two-rank decode bundle has a fingerprint");
    let one_rank = &scratchy_models::granite::granite_3_1_2b_instruct::KTIR_BUNDLE;
    assert!(
        one_rank.decode.iter().all(|d| d.fp != tp.decode.fp),
        "the two-rank bundle is its own, not the one-rank one"
    );
    assert!(
        tp.sources.len() as u32 <= tp.wirings.decode.num_sources,
        "every sliced source is a source of the two-rank wiring"
    );
}

/// The two-rank world ships whole: granite bakes a batched-prefill program in its one-rank world,
/// so its two-rank world carries one too — its own bundle and its own wiring.
#[test]
fn granites_two_rank_world_carries_its_prefill() {
    let tp = &scratchy_models::granite::granite_3_1_2b_instruct::SUPERDSC_TP[0];
    let one_rank = &scratchy_models::granite::granite_3_1_2b_instruct::KTIR_BUNDLE;
    let prefill = tp.prefill.as_ref().expect("the two-rank world bakes a prefill");
    assert!(tp.wirings.prefill.is_some(), "and its wiring");
    assert!(
        one_rank.prefill.as_ref().is_none_or(|p| p.fp != prefill.fp),
        "the two-rank prefill is its own bundle"
    );
}

/// The selection a loading session makes: a world of one runs the one-rank wiring, each rank of
/// the two-rank world runs the forward baked for it — the same one on both ranks.
#[test]
fn each_rank_selects_its_worlds_forward() {
    use scratchy_target_spyre::bundle_code::collective::TpWorld;
    use scratchy_target_spyre::wiring::{WorldForward, forward_for};
    let tp = scratchy_models::granite::granite_3_1_2b_instruct::SUPERDSC_TP;
    for id in TpWorld::of::<1>().ranks() {
        assert!(matches!(forward_for(tp, id), Ok(WorldForward::OneRank)));
    }
    for id in TpWorld::of::<2>().ranks() {
        match forward_for(tp, id) {
            Ok(WorldForward::Sharded(f)) => assert!(std::ptr::eq(f, &tp[0])),
            _ => panic!("rank {} of the two-rank world found no forward", id.rank()),
        }
    }
    // A model that baked no sharded world refuses a two-rank session by name.
    let none: &'static [scratchy_target_spyre::wiring::TpForward] = &[];
    let rank0 = TpWorld::of::<2>().ranks().next().unwrap();
    assert!(forward_for(none, rank0).is_err());
}

#[test]
fn granites_slice_laws_are_megatrons() {
    let tp = &scratchy_models::granite::granite_3_1_2b_instruct::SUPERDSC_TP[0];
    let count = |want: fn(&SourceShard) -> bool| tp.sources.iter().filter(|s| want(s)).count();
    assert_eq!(
        count(|s| matches!(s, SourceShard::Rows)),
        2 * 40,
        "one K-split projection (o, down) per all-reduce, two per layer"
    );
    assert!(
        count(|s| matches!(s, SourceShard::Cols)) >= 5 * 40,
        "q, k, v, gate and up split by columns in every layer"
    );
    assert!(
        count(|s| matches!(s, SourceShard::Whole)) > 0,
        "the norms and the lm head stay whole"
    );
}

// SPDX-License-Identifier: Apache-2.0
//! granite's sharded world, through the real `#[forward]` expansion — the world this build names
//! (`Build`, its `tp/N`).
//!
//! The spyre and subtile crates prove the sharding analysis and the collective emission on
//! hand-built graphs; this proves what granite's own bake ships: ONE sharded world, the build's,
//! with its own decode bundle and wiring, and the slice law each source loads by — derived by the
//! analysis, never declared. Granite's 40 layers each carry two K-split projections (o and down),
//! each followed by the all-reduce the analysis wove in, so exactly 80 sources load by ROWS; its
//! odd vocab (49155) does not split into whole sticks, so the lm head stays whole and no
//! all-gather is needed.

#![cfg(all(
    any(feature = "tp-2", feature = "tp-4", feature = "tp-8"),
    compiled_model = "granite::granite_3_1_2b_instruct"
))]

use scratchy_target_spyre::bundle_code::collective::{Build, World};
use scratchy_target_spyre::wiring::{SourceShard, TpForward};

/// The type `#[forward]` baked granite's sharded worlds as: the build's world.
const _: fn() -> &'static [TpForward<Build>] =
    || scratchy_target_spyre_models::granite::granite_3_1_2b_instruct::SUPERDSC_TP;

#[test]
fn granite_ships_one_sharded_world() {
    let model = &scratchy_target_spyre_models::granite::granite_3_1_2b_instruct::SUPERDSC_TP;
    assert_eq!(model.len(), 1, "granite bakes exactly one sharded world");
    let tp = &model[0];
    assert!(<Build as World>::SHARDED, "the build names a sharded world");
    assert!(
        !tp.decode.fp.is_empty(),
        "the sharded decode bundle has a fingerprint"
    );
    let one_rank = &scratchy_target_spyre_models::granite::granite_3_1_2b_instruct::KTIR_BUNDLE;
    assert!(
        one_rank.decode.iter().all(|d| d.fp != tp.decode.fp),
        "the sharded bundle is its own, not the one-rank one"
    );
    assert!(
        tp.sources.len() as u32 <= tp.wirings.decode.num_sources,
        "every sliced source is a source of the sharded wiring"
    );
}

/// The sharded world ships whole: granite bakes a batched-prefill program in its one-rank world,
/// so its sharded world carries one too — its own bundle and its own wiring.
#[test]
fn granites_sharded_world_carries_its_prefill() {
    let tp = &scratchy_target_spyre_models::granite::granite_3_1_2b_instruct::SUPERDSC_TP[0];
    let one_rank = &scratchy_target_spyre_models::granite::granite_3_1_2b_instruct::KTIR_BUNDLE;
    let prefill = tp
        .prefill
        .as_ref()
        .expect("the sharded world bakes a prefill");
    assert!(tp.wirings.prefill.is_some(), "and its wiring");
    assert!(
        one_rank.prefill.as_ref().is_none_or(|p| p.fp != prefill.fp),
        "the sharded prefill is its own bundle"
    );
}

#[test]
fn granites_slice_laws_are_megatrons() {
    let tp = &scratchy_target_spyre_models::granite::granite_3_1_2b_instruct::SUPERDSC_TP[0];
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

// SPDX-License-Identifier: Apache-2.0
//! THE FOLD-ROW REGIME TRAVELS ON THE BUNDLE, and the worker's intermediate-segment stride is a
//! DERIVATION from it — not a constant of the worker's own.
//!
//! What this pins: the emitter's fold ops declare the regime they were assembled under
//! (`EmittedOp::kv_fold_rows`, from `attn.rs`'s `BlockRows`); every entry of the baked launch index
//! carries it (`bundle::OpEntry::kv_fold_rows`); `BundleCode::fold_rows` is its one reader, and
//! refuses a bundle whose fold groups disagree; and `FoldRowRegime::int_rep_stride_bytes` maps it to
//! the stride `fold_plan`'s intermediate shift multiplies (`SessionKv::int_rep_stride_bytes`).
//! Without it the value lives twice — `per_req` in the emitter and a constant in the worker — with a
//! comment as the only tie.

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::assemble_attn;
use scratchy_subtile::sdsc_abstract::FoldRowRegime;
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::{
    FoldGrouping, bundle, bundle_fingerprint, launch_index,
};

/// A decode-batch attention emission (rows are requests), the shape whose fold groups the regime
/// describes. granite-3.1-8b's head geometry at hd=64, batch width 4.
fn batch_decode_ops() -> Vec<ktir_superdsc::emit::EmittedOp> {
    let mut sym = 0i64;
    let geom = scratchy_subtile::sdsc_abstract::AttnGeometry::<32, 8, 64>::minted();
    let width = scratchy_subtile::sdsc_abstract::attn_bundle_rows(geom, 4, true)
        .expect("4 is a baked decode rung");
    assemble_attn(
        7,
        geom,
        width,
        256,
        256,
        "qs",
        "nks",
        "nvr",
        "kct",
        "vc",
        "pmask",
        "cmask",
        None,
        scratchy_target_spyre::bundle_code::PlaceId::Act(7),
        true, // rows_are_requests: a decode batch
        &mut sym,
        None,
    )
    .expect("attention must emit")
}

/// THE STRIDE IS A DERIVATION. Whole-batch passes sweep every shared-buffer row, so there is no
/// per-request block to rebase onto and the stride is 0 — the value the worker used to assert as a
/// constant. Per-request, a request's block is its `nqh` head rows of one fp16 stick: the very
/// number `fold_plan`'s own per-request fixtures bake (`int_rep_stride_bytes = 32 * 64 * 2`).
#[test]
fn the_stride_derives_from_the_regime() {
    assert_eq!(FoldRowRegime::WholeBatch.int_rep_stride_bytes(32), 0);
    assert_eq!(
        FoldRowRegime::PerRequest.int_rep_stride_bytes(32),
        32 * 64 * 2
    );
    // The derivation scales with the head count — it is `nqh` rows, not a fixed block.
    assert_eq!(
        FoldRowRegime::PerRequest.int_rep_stride_bytes(8),
        8 * 64 * 2
    );
}

/// WHAT THE EMITTER BAKES TODAY: every fold op of a batch-decode emission declares WHOLE-BATCH rows.
/// This is the emitter-side half of the audited claim — the worker's stride-0 was the right VALUE,
/// and here is the fact it derives from (the fold's passes are assembled over `nqh*mq` rows, so no
/// pass may be rebased). If the emitter ever moves to per-request passes, this test is the reminder
/// that the manifest — and through it the worker's stride — moves with it, not a second constant.
#[test]
fn the_attention_emitter_declares_whole_batch_fold_rows() {
    let ops = batch_decode_ops();
    let folds: Vec<_> = ops.iter().filter(|o| o.kv_page_fold).collect();
    assert!(!folds.is_empty(), "a batch-decode emission has fold ops");
    for op in folds {
        assert_eq!(
            op.kv_fold_rows,
            FoldRowRegime::WholeBatch,
            "fold op `{}` declares a regime its blocks were not assembled under",
            op.op_name
        );
    }
}

/// THE LAUNCH INDEX CARRIES THE REGIME, both ways, and only on the FOLD groups — a non-fold group
/// has no pass to rebase, so its entry must read whole-batch whatever the ops declare.
#[test]
fn the_launch_index_carries_the_regime() {
    let ops = batch_decode_ops();
    let idx = launch_index(&ops, FoldGrouping::Split);
    let folds: Vec<_> = idx.iter().filter(|e| e.page_fold).collect();
    assert!(!folds.is_empty(), "a batch-decode emission has fold groups");
    for e in &folds {
        assert_eq!(e.fold_rows, bundle::FoldRows::WholeBatch);
    }
    assert_eq!(code_of(&idx).fold_rows(), Ok(bundle::FoldRows::WholeBatch));

    // Flip the ops' declaration and every fold entry must say so.
    let mut ops = batch_decode_ops();
    for op in ops.iter_mut().filter(|o| o.kv_page_fold) {
        op.kv_fold_rows = FoldRowRegime::PerRequest;
    }
    let idx = launch_index(&ops, FoldGrouping::Split);
    let folds: Vec<_> = idx.iter().filter(|e| e.page_fold).collect();
    assert!(!folds.is_empty());
    for e in &folds {
        assert_eq!(e.fold_rows, bundle::FoldRows::PerRequest);
    }
    assert_eq!(code_of(&idx).fold_rows(), Ok(bundle::FoldRows::PerRequest));
}

/// A bundle whose launches carry just these shifts — enough to ask it for its regime.
fn code_of(shifts: &[bundle::KvShifts]) -> bundle::BundleCode<'static> {
    bundle::BundleCode {
        fp: std::borrow::Cow::Borrowed("test"),
        layout: bundle::BundleLayout::default(),
        groups: std::borrow::Cow::Owned(
            shifts
                .iter()
                .map(|kv| bundle::LaunchGroup {
                    kv: *kv,
                    ..bundle::LaunchGroup::default()
                })
                .collect(),
        ),
        reroll: None,
    }
}

/// THE REGIME IS METADATA, NOT KERNEL CONTENT. The bundle fingerprint hashes the device ops and the
/// group partition — the emit gate's inputs — and the regime is in neither: flipping it changes no
/// device op byte and no group boundary, so two bundles that differ only in their declared regime
/// fingerprint identically. This is what licenses declaring it in the manifest at all.
#[test]
fn the_regime_is_metadata_not_kernel_content() {
    let ops = batch_decode_ops();
    let mut flipped = batch_decode_ops();
    for op in flipped.iter_mut().filter(|o| o.kv_page_fold) {
        op.kv_fold_rows = FoldRowRegime::PerRequest;
    }
    assert_eq!(
        bundle_fingerprint(&ops),
        bundle_fingerprint(&flipped),
        "the fold-row regime must not reach any device op byte or group boundary"
    );
}

/// THE READER REFUSES what the session cannot express. A session carries ONE intermediate stride, so
/// fold groups that disagree about their regime have no stride that is right for all of them.
#[test]
fn a_mixed_bundle_is_refused() {
    let fold = |rows| bundle::KvShifts {
        page_fold: true,
        fold_rows: rows,
        ..bundle::KvShifts::default()
    };
    assert!(
        code_of(&[
            fold(bundle::FoldRows::WholeBatch),
            fold(bundle::FoldRows::PerRequest)
        ])
        .fold_rows()
        .is_err(),
        "disagreeing fold groups must refuse"
    );
    // No fold groups at all: nothing re-launches, so nothing is rebased — whole-batch by derivation,
    // which keeps every unpaged bundle loading exactly as before.
    assert_eq!(
        code_of(&[bundle::KvShifts::default()]).fold_rows(),
        Ok(bundle::FoldRows::WholeBatch)
    );
}

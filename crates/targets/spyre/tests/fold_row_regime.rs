// SPDX-License-Identifier: Apache-2.0
//! THE FOLD-ROW REGIME TRAVELS ON THE BUNDLE, and the worker's intermediate-segment stride is a
//! DERIVATION from it — not a constant of the worker's own.
//!
//! What this pins: every entry of the baked launch index carries the regime
//! (`bundle::OpEntry::kv_fold_rows`); `BundleCode::fold_rows` is its one reader, and
//! refuses a bundle whose fold groups disagree; and `FoldRowRegime::int_rep_stride_bytes` maps it to
//! the stride `fold_plan`'s intermediate shift multiplies (`SessionKv::int_rep_stride_bytes`).
//! Without it the value lives twice — a declaration in the emitter and a constant in the worker —
//! with a comment as the only tie.
//!
//! (The emitter-side half — a batch-decode emission's fold ops declaring WholeBatch — was pinned
//! through the deleted hand-written attention emitter and went with it; when the live spliced fold
//! family needs the same pin, mint it from the splice, not from a copy of the old emitter.)

use scratchy_subtile::sdsc_abstract::FoldRowRegime;
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle;

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

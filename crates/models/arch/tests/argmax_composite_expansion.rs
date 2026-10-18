// SPDX-License-Identifier: Apache-2.0
//! The composite argmax in a REAL `#[forward]` expansion (#603 PR3a).
//!
//! The spyre-crate tests pin the walk on hand-built fixtures (the mint's facts,
//! the ladder's content-key law, the numeric parity); this file pins the MODEL
//! side, the way `spyre_weight_folds.rs` does: granite-3.3-2b-base — TIED, the
//! model the loop-closure front targets — mints the composite in its real
//! expansion, and an UNTIED granite (micro-g3.3-8b-instruct-1b, the 8b shape at
//! micro scale, `tie_word_embeddings: false`) compiles its whole ladder without
//! placing either constant.
//!
//! The observable is [`BakeFacts::argmax_vocab`] — the registry law's emission
//! half: read off the ramp's own placement in the bundle the expansion baked
//! (`BundleCode::layout`), the same lookup the worker's load performs, resolved
//! through the same `bundle(fp)` the runtime uses. A tied model whose tail the
//! walk cannot recognize is a LOUD BUILD ERROR — never a silent host fallback —
//! so the expansion compiling at all is the mint's negative half; this file is
//! the positive half. Every rung of the real decode ladder is checked (the
//! anchor `KTIR_BUNDLE::decode` names, plus every sibling body its `RerollMeta`
//! names), so "the composite mints" is a claim about the whole ladder, not one
//! bundle — the model-side echo of the walk test's content-key assert.
//!
//! granite-fp8 boundary (documented, never papered over): under a quant preset
//! that replaces the dense emission, granite's lm-head is W8A8 — arity 3 — and
//! the walk REFUSES loudly (the spyre-crate walk test pins that refusal and its
//! message). This file runs the model's own dense scope, where the lm-head is
//! Dense and the tied gather reads the table in place.

#![cfg(all(
    feature = "spyre",
    feature = "granite-3.3-2b-base",
    feature = "micro-g3.3-8b-instruct-1b"
))]

use scratchy_target_spyre::bundle_code::{BundleCode, bundle};
use scratchy_target_spyre::manifest::KtirBundle;
use scratchy_target_spyre::wiring::BakeFacts;

/// granite-3.3-2b-base's config `vocab_size` — the width the ramp must carry.
const GRANITE_VOCAB: usize = 49152;

/// Every decode bundle of one model's ladder: the anchor `KTIR_BUNDLE::decode`
/// names, plus every rung body its `RerollMeta` names — the same resolver the
/// runtime uses (`bundle(fp)` over the sibling fingerprints), so a rung that
/// fails to resolve here is a rung the runtime could not launch either.
fn decode_bundles(kb: &KtirBundle) -> Vec<&'static BundleCode<'static>> {
    let mut out = Vec::new();
    for d in kb.decode {
        let anchor = bundle(d.fp).unwrap_or_else(|| panic!("the decode anchor {} resolves", d.fp));
        if let Some(meta) = &anchor.reroll {
            for rung in meta.rungs.iter() {
                for fp in rung
                    .bodies
                    .iter()
                    .chain(rung.bodies_fused.iter())
                    .filter(|s| !s.is_none())
                {
                    out.push(
                        bundle(fp.as_str())
                            .unwrap_or_else(|| panic!("the rung body {} resolves", fp.as_str())),
                    );
                }
            }
        }
        out.push(anchor);
    }
    out
}

/// TIED granite mints the composite in the real expansion: every decode bundle
/// of the real ladder places the ramp at granite's own vocab, with the scalar-V
/// stick alongside it.
#[test]
fn tied_granite_mints_the_composite_in_the_real_expansion() {
    let kb = &scratchy_models::granite::granite_3_3_2b_base::KTIR_BUNDLE;
    let bundles = decode_bundles(kb);
    assert!(!bundles.is_empty(), "granite bakes a decode ladder");
    for code in &bundles {
        let facts = BakeFacts::of(&code.layout);
        assert_eq!(
            facts.argmax_vocab, GRANITE_VOCAB,
            "every decode rung places the ramp at granite's vocab ({})",
            code.fp
        );
        assert_eq!(
            code.layout
                .place_of_tid(scratchy_target_spyre::lower_subtile_tape_to_superdsc::argmax_v_tid())
                .map(|p| p.size),
            Some(128),
            "the scalar-V stick is [1, 32] fp32 alongside it ({})",
            code.fp
        );
    }
}

/// The PREFILL bundle (m > 1) keeps the host readback path: the one-request
/// gate means no composite, and the bundle says so by placing no ramp.
#[test]
fn granites_prefill_places_no_composite() {
    let kb = &scratchy_models::granite::granite_3_3_2b_base::KTIR_BUNDLE;
    let p = kb
        .prefill
        .as_ref()
        .expect("granite bakes a batched-prefill program; the bundle has none");
    let code = bundle(p.fp).expect("the prefill anchor resolves");
    assert_eq!(
        BakeFacts::of(&code.layout).argmax_vocab,
        0,
        "prefill is m > 1 — the composite is the one-request rung's alone"
    );
}

/// UNTIED granite stays untouched: its whole decode ladder compiles and places
/// NEITHER constant — the host readback path is the untied model's own, and the
/// tie token (`TiedLmHeadEmbedding::from_tied_config`) is the guard that keeps
/// it there.
#[test]
fn an_untied_model_places_no_composite() {
    let kb = &scratchy_models::granite::micro_g3_3_8b_instruct_1b::KTIR_BUNDLE;
    let bundles = decode_bundles(kb);
    assert!(
        !bundles.is_empty(),
        "the untied model bakes a decode ladder"
    );
    for code in &bundles {
        assert_eq!(
            BakeFacts::of(&code.layout).argmax_vocab,
            0,
            "untied keeps the host readback path ({})",
            code.fp
        );
        assert!(
            code.layout
                .place_of_tid(scratchy_target_spyre::lower_subtile_tape_to_superdsc::argmax_v_tid())
                .is_none(),
            "no scalar-V stick either ({})",
            code.fp
        );
    }
    if let Some(p) = kb.prefill.as_ref() {
        let code = bundle(p.fp).expect("the untied prefill anchor resolves");
        assert_eq!(BakeFacts::of(&code.layout).argmax_vocab, 0);
    }
}

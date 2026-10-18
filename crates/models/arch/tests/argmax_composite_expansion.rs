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
//! The mint-site gate, PINNED FROM BOTH SIDES: the expansion's gate
//! (`lm_head_admits_the_composite`, keyed on the lm-head's STRUCTURE, never on a
//! preset's name) mints the tie token only behind the form the first cut can SERVE —
//! an arity-2 DENSE lm-head at whole-fp16-stick vocab and hidden. The boundary this
//! file's models actually sit on is GEOMETRY: granite-3.1-2b-instruct — the card
//! gate's OWN cell model — carries vocab 49155, three lanes past the last fp16
//! stick (768·64 + 3), and the composite's `[1, vocab]` links run at the stick (the
//! dxp scheduler rejects sub-stick tiles), so it keeps the HOST argmax by
//! construction while stick-aligned tied granite-3.3-2b-base mints rung by rung.
//! BOTH the fp16 and fp8 card cells are host-argmax cells — by that GEOMETRY, not
//! by quant staging: granite's fp8 preset (`fp8-dynamic-per-channel`) IGNORES the
//! lm_head, so the dense tail persists under it and 3.3 keeps minting. That
//! scope-side law is pinned by `argmax_composite_expansion_scope.rs`, which runs
//! under the quant line too (preset builds RENAME the variant modules
//! `<stem>_<preset>`, so this file's module paths are the dense line's alone). A
//! preset that DOES stage the lm-head (fp8 W8A8: the scale rides as a THIRD
//! operand; MLX-affine: the table is packed) makes it arity-3/non-Dense and no
//! token is minted — that refusal is pinned on a hand-built W8A8 graph by the
//! spyre-crate walk test, as is the walk's own loud refusal for a hand-minted
//! token.

#![cfg(all(
    feature = "spyre",
    feature = "granite-3.3-2b-base",
    feature = "granite-3.1-2b-instruct",
    feature = "micro-g3.3-8b-instruct-1b"
))]

use scratchy_target_spyre::bundle_code::{BundleCode, bundle};
use scratchy_target_spyre::manifest::KtirBundle;
use scratchy_target_spyre::wiring::BakeFacts;

/// granite-3.3-2b-base's config `vocab_size` — the width the ramp must carry.
const GRANITE_VOCAB: usize = 49152;

/// granite-3.1-2b-instruct's config `vocab_size` — the CARD GATE's own cell model
/// (the 2b fp16 / 2b fp8 legs run this stem), and NOT a whole fp16 stick:
/// 49155 = 768·64 + 3, three lanes past the last one. The composite's `[1, vocab]`
/// links run at the stick, so this model keeps the host argmax in EVERY scope — the
/// geometry test below pins the boundary by its arithmetic, so a config edit that
/// moves the vocab moves the expectation with it.
const GRANITE_3_1_VOCAB: usize = 49155;

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

/// A model's decode-ladder expectation:
///
/// * [`Expect::Ramp`] — every rung places the composite's ramp at the model's own
///   vocab, with the scalar-V stick alongside it;
/// * [`Expect::HostPath`] — no rung mints the composite; the ladder keeps the HOST
///   argmax, for the boundary the variant carries (stated in every assert message).
#[derive(Clone, Copy)]
enum Expect {
    Ramp(usize),
    HostPath(&'static str),
}

/// granite-3.1's own boundary, the dense line's string: the geometry the first cut
/// cannot serve.
const NON_STICK_WHY: &str = "its vocab 49155 is three lanes past the last fp16 stick \
     (768·64 + 3), and the composite's [1, vocab] links run at the stick";

/// ⭐⭐⭐ THE MINT-SITE GATE'S MODEL-SIDE LAW — the composite follows the lm-head's
/// STRUCTURE, never a preset's name: granite-3.3-2b-base (tied, stick-aligned) mints
/// on every rung of its real ladder, while granite-3.1-2b-instruct (tied, non-stick
/// vocab) mints nothing and keeps the host argmax. The scope-side half of the law —
/// what survives a quant preset — is `argmax_composite_expansion_scope.rs`'s own;
/// this file runs on the DENSE gate line (`spyre,granite-3.3-2b-base,
/// granite-3.1-2b-instruct,micro-g3.3-8b-instruct-1b`), because preset builds rename
/// the variant modules and this file names them by path.
fn assert_composite_follows_lm_head_structure(kb: &KtirBundle, expect: Expect, what: &str) {
    let bundles = decode_bundles(kb);
    assert!(!bundles.is_empty(), "{what} bakes a decode ladder");
    for code in &bundles {
        let facts = BakeFacts::of(&code.layout);
        match expect {
            Expect::Ramp(vocab) => {
                assert_eq!(
                    facts.argmax_vocab, vocab,
                    "every decode rung places the ramp at {what}'s own vocab ({})",
                    code.fp
                );
                assert_eq!(
                    code.layout
                        .place_of_tid(
                            scratchy_target_spyre::lower_subtile_tape_to_superdsc::argmax_v_tid()
                        )
                        .map(|p| p.size),
                    Some(128),
                    "the scalar-V stick is [1, 32] fp32 alongside it ({})",
                    code.fp
                );
            }
            Expect::HostPath(why) => {
                assert_eq!(
                    facts.argmax_vocab, 0,
                    "{what}: {why}, so no tie token is minted and the rung keeps the \
                     host argmax ({})",
                    code.fp
                );
                assert!(
                    code.layout
                        .place_of_tid(
                            scratchy_target_spyre::lower_subtile_tape_to_superdsc::argmax_v_tid()
                        )
                        .is_none(),
                    "{what}: no scalar-V stick either ({})",
                    code.fp
                );
            }
        }
    }
}

/// TIED granite mints the composite in the real expansion — see
/// [`assert_composite_follows_lm_head_structure`] for the law and its two gate lines.
#[test]
fn tied_granite_mints_the_composite_in_the_real_expansion() {
    assert_composite_follows_lm_head_structure(
        &scratchy_models::granite::granite_3_3_2b_base::KTIR_BUNDLE,
        Expect::Ramp(GRANITE_VOCAB),
        "granite-3.3-2b-base",
    );
}

/// The CARD GATE's own cell model — granite-3.1-2b-instruct, the stem the 2b fp16 and
/// 2b fp8 legs run — sits on the gate's GEOMETRY boundary: its vocab is three lanes
/// past the last fp16 stick, so the expansion mints NO tie token and every rung of its
/// real ladder keeps the host argmax — in BOTH quant scopes (under the fp8 preset the
/// arity-3 boundary refuses it too). Both card cells are therefore host-argmax
/// controls by construction; the composite's card evidence needs a stick-aligned tied
/// granite (3.3-2b-base) or the pad-lane follow-up. The arithmetic is pinned FIRST, so
/// a config edit that moves granite-3.1's vocab moves this expectation with it.
#[test]
fn the_card_gates_own_granite_keeps_the_host_path_by_geometry() {
    assert_eq!(
        GRANITE_3_1_VOCAB,
        768 * 64 + 3,
        "the boundary this test pins is the non-stick vocab itself"
    );
    assert_composite_follows_lm_head_structure(
        &scratchy_models::granite::granite_3_1_2b_instruct::KTIR_BUNDLE,
        Expect::HostPath(NON_STICK_WHY),
        "granite-3.1-2b-instruct",
    );
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

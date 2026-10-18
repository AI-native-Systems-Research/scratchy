// SPDX-License-Identifier: Apache-2.0
//! The composite argmax in a REAL `#[forward]` expansion (#603 PR3a).
//!
//! The spyre-crate tests pin the walk on hand-built fixtures (the mint's facts,
//! the ladder's content-key law, the numeric parity); this file pins the MODEL
//! side, the way `spyre_weight_folds.rs` does: granite-3.3-2b-base — TIED, the
//! model the loop-closure front targets — mints the composite in its real
//! expansion, and an UNTIED granite (micro-g3.3-8b-instruct-1b, the 8b shape at
//! micro scale, `tie_word_embeddings: false`) compiles its whole ladder placing
//! none of the composite's constants.
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
//! preset's name) mints the tie token behind the two facts the composite needs —
//! an arity-2 DENSE lm-head and a whole-fp16-stick HIDDEN. The vocab needs no
//! conjunct of its own: a sub-stick vocab takes the PAD-LANE form, whose staged
//! fp16 bias (0 on the real lanes, the fp16 floor −65504 on the pads) owns the
//! lanes past the vocab so garbage there can never win the max. granite-3.1
//! — the card gate's OWN cell model, vocab 49155 = 768·64 + 3 — is that form's
//! own cell: it mints rung by rung exactly like stick-aligned granite-3.3, with
//! the bias placed alongside the ramp. BOTH the fp16 and fp8 card cells take the
//! composite by GEOMETRY, not by quant staging: granite's fp8 preset
//! (`fp8-dynamic-per-channel`) IGNORES the lm_head, so the dense tail persists
//! under it and both keep minting. That
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

/// The fp16 stick width (`Fp16::ELEMS_PER_STICK`, pinned against the constant
/// itself by the spyre-crate tests) — the staging width every `[1, vocab]` link of
/// the composite runs at, pad lanes included.
const FP16_STICK: usize = 64;

/// granite-3.1-2b-instruct's config `vocab_size` — the CARD GATE's own cell model
/// (the 2b fp16 / 2b fp8 legs run this stem), and NOT a whole fp16 stick:
/// 49155 = 768·64 + 3, three lanes past the last one — the PAD-LANE form's own
/// cell: the staged fp16 bias owns the 61 lanes from the vocab to the padded
/// stick, so the sub-stick vocab is served by GEOMETRY. The arithmetic test
/// below pins the shape, so a config edit that moves the vocab moves the
/// expectation with it.
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

/// A model's decode-ladder expectation: [`Expect::Ramp`] — every rung places the
/// composite's ramp at the model's own vocab with the scalar-V stick alongside
/// it, and, when that vocab is sub-stick, the fp16 pad-lane bias at the padded
/// stick (the PAD-LANE form's own footprint).
#[derive(Clone, Copy)]
enum Expect {
    Ramp(usize),
}

/// ⭐⭐⭐ THE MINT-SITE GATE'S MODEL-SIDE LAW — the composite follows the lm-head's
/// STRUCTURE, never a preset's name: granite-3.3-2b-base (tied, stick-aligned) and
/// granite-3.1-2b-instruct (tied, vocab three lanes past the last fp16 stick) BOTH
/// mint on every rung of their real ladders — the latter through the PAD-LANE
/// form, whose staged bias owns the lanes past the vocab. The scope-side half of
/// the law — what survives a quant preset — is `argmax_composite_expansion_scope.rs`'s
/// own; this file runs on the DENSE gate line (`spyre,granite-3.3-2b-base,
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
                let v_pad = (vocab as u64).next_multiple_of(FP16_STICK as u64);
                assert_eq!(
                    code.layout
                        .place_of_tid(
                            scratchy_target_spyre::lower_subtile_tape_to_superdsc::argmax_ramp_tid(
                            )
                        )
                        .map(|p| p.size),
                    Some(v_pad * 4),
                    "the ramp is fp32 at the stick-padded staging width {v_pad} ({}, {what})",
                    code.fp
                );
                let bias = code
                    .layout
                    .place_of_tid(
                        scratchy_target_spyre::lower_subtile_tape_to_superdsc::argmax_bias_tid(),
                    )
                    .map(|p| p.size);
                if v_pad > vocab as u64 {
                    let pads = v_pad - vocab as u64;
                    assert_eq!(
                        bias,
                        Some(v_pad * 2),
                        "a sub-stick vocab takes the PAD-LANE form: the fp16 bias (0 on the \
                         real lanes, the fp16 floor on the {pads} pads) is placed at the \
                         padded stick ({}, {what})",
                        code.fp
                    );
                } else {
                    assert!(
                        bias.is_none(),
                        "a whole-stick vocab needs no pad-lane bias ({}, {what})",
                        code.fp
                    );
                }
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
/// 2b fp8 legs run — is the PAD-LANE form's own cell: its vocab is three lanes past
/// the last fp16 stick, and the staged bias owns the 61 lanes from the vocab to the
/// padded stick, so every rung of its real ladder mints the composite exactly like
/// stick-aligned granite-3.3 — in BOTH quant scopes (granite's fp8 preset ignores the
/// lm_head, so the dense tail persists under it). Both card cells are therefore
/// composite cells by GEOMETRY. The arithmetic is pinned FIRST, so a config edit that
/// moves granite-3.1's vocab moves this expectation with it.
#[test]
fn the_card_gates_own_granite_takes_the_pad_lane_form() {
    assert_eq!(
        GRANITE_3_1_VOCAB,
        768 * 64 + 3,
        "the shape this test pins is the sub-stick vocab itself"
    );
    assert_eq!(
        GRANITE_3_1_VOCAB.next_multiple_of(64) - GRANITE_3_1_VOCAB,
        61,
        "61 pad lanes from the vocab to the padded stick"
    );
    assert_composite_follows_lm_head_structure(
        &scratchy_models::granite::granite_3_1_2b_instruct::KTIR_BUNDLE,
        Expect::Ramp(GRANITE_3_1_VOCAB),
        "granite-3.1-2b-instruct",
    );
}

/// The PREFILL bundle (m > 1) keeps the host readback path: the one-request
/// gate means no composite, and the bundle says so by placing none of the
/// composite's constants.
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
    assert!(
        code.layout
            .place_of_tid(scratchy_target_spyre::lower_subtile_tape_to_superdsc::argmax_ramp_tid())
            .is_none(),
        "no ramp placement either"
    );
    assert!(
        code.layout
            .place_of_tid(scratchy_target_spyre::lower_subtile_tape_to_superdsc::argmax_bias_tid())
            .is_none(),
        "no pad-lane bias placement either"
    );
}

/// UNTIED granite stays untouched: its whole decode ladder compiles placing NONE
/// of the composite's constants — the host readback path is the untied model's
/// own, and the tie token (`TiedLmHeadEmbedding::from_tied_config`) is the guard
/// that keeps it there.
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
        assert!(
            code.layout
                .place_of_tid(
                    scratchy_target_spyre::lower_subtile_tape_to_superdsc::argmax_bias_tid()
                )
                .is_none(),
            "no pad-lane bias either ({})",
            code.fp
        );
    }
    if let Some(p) = kb.prefill.as_ref() {
        let code = bundle(p.fp).expect("the untied prefill anchor resolves");
        assert_eq!(BakeFacts::of(&code.layout).argmax_vocab, 0);
    }
}

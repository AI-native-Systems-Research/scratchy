// SPDX-License-Identifier: Apache-2.0
//! THE COMPOSITE ARGMAX ACROSS QUANT SCOPES (#603 PR3a) — the registry-side law.
//!
//! `argmax_composite_expansion.rs` pins the ladder, rung by rung, through module
//! paths — which makes it the DENSE gate line's alone: a selected preset REPLACES a
//! model's dense emission and RENAMES its module (`<stem>_<preset>`), so under
//! `scratchy-quantizations/fp8-dynamic-per-channel` the paths this suite would name
//! do not exist. This file names NO module path: it reads the bundle REGISTRY this
//! build linked (`inventory`, the same `bundle(fp)` resolution the worker uses, but
//! walked whole) and the `granite.rs` emission this build produced (the
//! `gemma4_router_affine_fingerprint.rs` precedent) — so it runs under BOTH gate
//! lines, dense and quant, and the two lines' asserts are the SAME asserts.
//!
//! The law: the composite follows the lm-head's STRUCTURE, never a preset's name —
//! and granite's own fp8 preset (`fp8-dynamic-per-channel`) IGNORES the lm_head, so
//! the dense tail PERSISTS under it. Consequences this file pins:
//!
//!   • granite-3.3-2b-base (tied, stick-aligned 49152) MINTS in BOTH scopes — the
//!     registry holds minting bundles whose ramp width is 49152 under the quant line
//!     too, which is what "keys on structure, not preset name" means when the
//!     preset leaves the lm-head alone;
//!   • granite-3.1-2b-instruct (tied, NON-stick 49155 = 768·64 + 3) mints in BOTH
//!     scopes too — through the PAD-LANE form: the staged fp16 bias owns the 61
//!     lanes from the vocab to the padded stick, so the sub-stick vocab is served
//!     by GEOMETRY and the card gate's own cell model is a composite cell in
//!     every scope;
//!   • every minting bundle carries its scalar-V stick (the pairing law), its ramp
//!     at the stick-padded staging width, and — when the vocab is sub-stick — the
//!     fp16 pad-lane bias at that same padded width.
//!
//! Attribution note: this gate line compiles exactly the granite family
//! (3.3-2b-base, 3.1-2b-instruct, micro-g3.3-8b-instruct-1b), so registry-wide
//! claims ARE granite-family claims — and the emission half attributes per VARIANT
//! module by name, dense and preset spellings both.

#![cfg(all(
    feature = "spyre",
    feature = "granite-3.3-2b-base",
    feature = "granite-3.1-2b-instruct",
    feature = "micro-g3.3-8b-instruct-1b"
))]

use scratchy_target_spyre::bundle_code::{BundleCode, inventory};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::{
    argmax_bias_tid, argmax_ramp_tid, argmax_v_tid,
};
use scratchy_target_spyre::wiring::BakeFacts;

/// granite-3.3-2b-base's vocab — the one TIED, stick-aligned model this gate line
/// compiles, so the whole-stick half of the mint set the asserts expect.
const GRANITE_3_3_VOCAB: usize = 49152;

/// granite-3.1-2b-instruct's vocab — the card gate's own cell model, and NOT a whole
/// fp16 stick: 49155 = 768·64 + 3, three lanes past the last one — the PAD-LANE
/// form's own cell (61 pad lanes to the padded stick 49216).
const GRANITE_3_1_VOCAB: usize = 49155;

/// The fp16 stick width (`Fp16::ELEMS_PER_STICK`, pinned against the constant itself
/// by the spyre-crate tests) — the staging width the composite's `[1, vocab]` links
/// run at, pad lanes included.
const FP16_STICK: usize = 64;

/// Which quant scope this build carried — named in every message. Never branched on
/// for the LAW (the law is scope-invariant); the emission half uses it only to pick
/// which variant-module spelling to look for.
fn quant_scope() -> &'static str {
    if scratchy_quantizations::enabled_presets().contains(&"fp8-dynamic-per-channel") {
        "fp8-dynamic-per-channel"
    } else {
        "dense"
    }
}

/// The `granite.rs` emission this build produced (`scratchy-forwards.rs` writes
/// `$OUT_DIR/granite.rs`, which `src/arch/granite.rs` includes).
fn emission() -> String {
    std::fs::read_to_string(concat!(env!("OUT_DIR"), "/granite.rs"))
        .expect("scratchy-forwards.rs writes $OUT_DIR/granite.rs")
}

/// One variant module's body: from its `pub mod <name> {` to the next `pub mod`
/// (the trailing `{` keeps a stem from matching its preset-suffixed sibling).
/// `None` when this build's quant scope left the variant out.
fn variant_module<'a>(src: &'a str, variant: &str) -> Option<&'a str> {
    let head = format!("pub mod {variant} {{");
    let start = src.find(&head)? + head.len();
    let body = &src[start..];
    Some(match body.find("pub mod ") {
        Some(next) => &body[..next],
        None => body,
    })
}

/// ⚓ THE LINK ANCHOR — the registry this file asserts over is only as linked as
/// the binary's references: each variant module's `inventory::submit!`s ride along
/// with its `KTIR_BUNDLE` static, and a test that names NO variant module by path
/// (preset builds rename them — the whole point of this file) references nothing
/// the linker would keep, leaving the registry EMPTY (observed: 0 bundles). The
/// arch-level `Weights` ENUM is preset-stable, and its `ktir_bundle` override's
/// arms reference every compiled variant's `KTIR_BUNDLE` — so a fn pointer to it,
/// kept alive through `black_box`, links the variants in. The pointer is never
/// called; it is a reference, not a behavior.
fn link_anchor() {
    let ktir_bundle: fn(
        &scratchy_models::granite::Weights,
    ) -> Option<&'static (dyn std::any::Any + Send + Sync)> =
        scratchy_forward_compiler::ScratchyWeights::ktir_bundle;
    std::hint::black_box(ktir_bundle);
}

/// ⭐⭐⭐ THE SCOPE LAW, REGISTRY SIDE — walk every bundle this build linked (the
/// same `BundleCode`s `bundle(fp)` resolves, taken whole) and state the mint set:
/// granite-3.3 mints (in BOTH scopes — the fp8 preset ignores the lm_head, so the
/// dense tail persists and the gate admits it), granite-3.1 mints through the
/// PAD-LANE form (its sub-stick vocab is served, not refused), and every minting
/// bundle pairs its ramp with the scalar-V stick and carries the fp16 bias iff
/// its vocab is sub-stick.
#[test]
fn the_composite_follows_structure_across_quant_scopes() {
    link_anchor();
    let scope = quant_scope();

    let mut bundles = 0usize;
    let mut minted: Vec<usize> = Vec::new();
    for code in inventory::iter::<BundleCode<'static>>() {
        bundles += 1;
        let facts = BakeFacts::of(&code.layout);
        if facts.argmax_vocab != 0 {
            minted.push(facts.argmax_vocab);
            assert_eq!(
                code.layout.place_of_tid(argmax_v_tid()).map(|p| p.size),
                Some(128),
                "a minting bundle carries its scalar-V stick alongside the ramp ({}, \
                 {scope} scope)",
                code.fp
            );
            let vocab = facts.argmax_vocab;
            let v_pad = (vocab as u64).next_multiple_of(FP16_STICK as u64);
            assert_eq!(
                code.layout.place_of_tid(argmax_ramp_tid()).map(|p| p.size),
                Some(v_pad * 4),
                "a minting bundle's ramp is fp32 at the stick-padded staging width \
                 {v_pad} ({}, {scope} scope)",
                code.fp
            );
            let bias = code.layout.place_of_tid(argmax_bias_tid()).map(|p| p.size);
            if v_pad > vocab as u64 {
                assert_eq!(
                    bias,
                    Some(v_pad * 2),
                    "a sub-stick vocab takes the PAD-LANE form: the fp16 bias is placed \
                     at the padded stick ({}, {scope} scope)",
                    code.fp
                );
            } else {
                assert!(
                    bias.is_none(),
                    "a whole-stick vocab needs no pad-lane bias ({}, {scope} scope)",
                    code.fp
                );
            }
        }
    }
    assert!(
        bundles >= 6,
        "the registry carries this line's three models' ladders ({bundles} bundles; \
         the current shape bakes 12 — prefill plus decode rungs per model) — else \
         every assert below is vacuous"
    );
    assert!(
        minted.contains(&GRANITE_3_3_VOCAB),
        "granite-3.3-2b-base mints in the {scope} scope too — the fp8 preset IGNORES \
         the lm_head, so the dense tail persists and the gate admits it: the composite \
         follows STRUCTURE, never a preset's name"
    );
    assert!(
        minted.contains(&GRANITE_3_1_VOCAB),
        "granite-3.1-2b-instruct mints in the {scope} scope — its vocab 49155 is three \
         lanes past the last fp16 stick (768·64 + 3) and the PAD-LANE form's staged \
         bias owns them: the card gate's own cell model is a composite cell by \
         GEOMETRY, in every scope"
    );
    assert!(
        minted
            .iter()
            .all(|v| *v == GRANITE_3_3_VOCAB || *v == GRANITE_3_1_VOCAB),
        "every minting bundle is one of this line's two tied granites' own vocabs \
         (49152 stick-aligned, 49155 sub-stick) — minted set: {minted:?} ({scope} scope)"
    );
}

/// ⭐ THE SCOPE LAW, EMISSION SIDE — per VARIANT module, by name, in the emission
/// this build produced: the mint's own footprint (`argmax_o` op names — the walk
/// mints them, and only the walk) is present in granite-3.3's AND granite-3.1's
/// variants, under the dense spelling AND the preset spelling. This is
/// the attribution half (which model minted), and the non-vacuity half (the preset
/// build really carries the `<stem>_<preset>` variants the registry asserts ran
/// over).
#[test]
fn the_mint_follows_the_variant_not_the_scope() {
    let scope = quant_scope();
    let src = emission();

    // The variant spellings this scope carries: dense builds name the bare stem,
    // preset builds append the preset.
    let (g33, g31) = if scope == "dense" {
        ("granite_3_3_2b_base", "granite_3_1_2b_instruct")
    } else {
        (
            "granite_3_3_2b_base_fp8_dynamic_per_channel",
            "granite_3_1_2b_instruct_fp8_dynamic_per_channel",
        )
    };
    let mints = |variant: &str| {
        let module = variant_module(&src, variant)
            .unwrap_or_else(|| panic!("the {variant} variant module is in the emission"));
        module.matches("argmax_o").count()
    };
    assert!(
        mints(g33) > 0,
        "granite-3.3's variant carries the composite's op names ({scope} scope) — the \
         fp8 preset ignores the lm_head, so the mint survives it"
    );
    assert!(
        mints(g31) > 0,
        "granite-3.1's variant carries the composite's op names ({scope} scope) — the \
         PAD-LANE form serves its sub-stick vocab, and the fp8 preset ignores the \
         lm_head so the mint survives it there too"
    );
}

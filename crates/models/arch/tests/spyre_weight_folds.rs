// SPDX-License-Identifier: Apache-2.0
//! granite's foldable constants, pinned on the model they were built for.
//!
//! TWO families would commute into staged weights — the dsl scales every residual with
//! `residual_multiplier` (`add(oproj * scalar(s), h)`, `add(down * scalar(s), h)` per layer)
//! and torch-spyre's attention pre-multiplies both the query and the key by
//! `√attention_multiplier` — but granite's own multipliers (0.22 and √(1/64) = 0.125) are
//! DOWN-scales, and the f16 floor gate refuses exactly those: the landing `f16(ws·s)` cannot
//! be proven above f16's normal floor at bake (staged ws is checkpoint data, unbounded
//! below), and a subnormal landing sheds mantissa bits the device codec does not restore —
//! measured on this very model as a 0.29% o-side seed amplifying ~35× to 10.2% at the
//! logits and a different card answer. So the pin is the REFUSAL: granite folds NOTHING,
//! every multiplier reaches the device as a const a surviving program reads, and a
//! recognition that silently folded one thing on either side is a failure in either
//! direction.
//!
//! This is the MODEL-side half of the pin: the spyre-crate tests prove the gate's both
//! sides on hand-authored graphs (the up-multiplier fold and the down-multiplier refusal);
//! these prove granite's own graph, through the real `#[forward]` expansion, admits exactly
//! zero folds — and that the multipliers the folds would have removed occupy their registry
//! slots again.
//!
//! The tied lm_head/embedding table stays OUT (its `logits_scaling` rides the splice's own
//! lm-head-tail fold, and the shared table refuses a per-consumer scale), and the
//! `embedding_multiplier` stays in the registry (its producer is a gather, not a matmul) —
//! both asserted, so a recognition that silently widened or narrowed is a failure either
//! way.

#![cfg(all(feature = "spyre", feature = "granite-3.1-2b-instruct"))]

/// The phases a granite bake carries, with the wiring each one lowered — decode always, the
/// batched-prefill program when the bake made one.
fn phases() -> Vec<(&'static str, &'static scratchy_target_spyre::wiring::Wiring)> {
    let w = &scratchy_models::granite::granite_3_1_2b_instruct::SUPERDSC_WIRINGS;
    match &w.prefill {
        Some(p) => vec![("decode", &w.decode), ("prefill", p)],
        None => panic!("granite bakes a batched-prefill program; the wiring has none"),
    }
}

#[test]
fn granite_folds_nothing_and_every_multiplier_reaches_the_device() {
    let residual = 0.22f32; // granite-3.1-2b-instruct config `residual_multiplier`
    let embed = 12.0f32; // config `embedding_multiplier` — a GATHER's scale, never foldable
    let eps = 1e-5f32; // config `rms_norm_eps` — a survivor the registry must still serve
    for (phase, wiring) in phases() {
        // ⛔ EXACTLY EMPTY, not merely "no 0.22": the fold list is the union both constant
        // families feed, so a recognition that folded one thing on either side — residual,
        // attention, or anything else — is wrong in either direction.
        assert_eq!(
            wiring.weight_scale_folds.len(),
            0,
            "{phase}: granite's multipliers are all down-scales and the f16 floor gate \
             refuses every one: {:?}",
            wiring.weight_scale_folds
        );
        // The residual multiplier is BACK in the registry — the per-layer ScalarMul
        // programs survive on the tape and read their const. (The embedding multiplier was
        // never foldable and keeps its slot for its own reason.)
        assert!(
            wiring.scalarmul_scales.contains(&residual),
            "{phase}: the un-folded residual multiplier must occupy a registry slot: {:?}",
            wiring.scalarmul_scales
        );
        assert!(
            wiring.scalarmul_scales.contains(&embed),
            "{phase}: the embedding multiplier is a gather's scale and must stay in the \
             registry: {:?}",
            wiring.scalarmul_scales
        );
        assert!(
            wiring.scalarmul_scales.contains(&eps),
            "{phase}: the rms eps is additive and must stay in the registry: {:?}",
            wiring.scalarmul_scales
        );
    }
}

#[test]
fn granites_attention_scale_stays_on_the_tape_at_the_f16_floor() {
    let unsplit = 1.0f32 / 64.0; // the config's own `attention_multiplier` — the score-side mul
    let sqrt_scale = (1.0f32 / 64.0).sqrt(); // √(attention_multiplier) = 0.125, a down-scale
    for (phase, wiring) in phases() {
        // The attention fold refused: no entry at √scale (the residual family's absence is
        // pinned by the empty-list assertion above; this names the attention side so a
        // failure points at the family that regressed).
        assert!(
            !wiring
                .weight_scale_folds
                .iter()
                .any(|(_, s)| s.to_bits() == sqrt_scale.to_bits()),
            "{phase}: the attention scale is a down-scale and must not fold: {:?}",
            wiring.weight_scale_folds
        );
        // The attention's OWN registry slot is BACK — the census arm keeps it now that the
        // node keeps its scale and the lowering emits the query/key multiplies again. (The
        // √scale bits are 0.125, which lm_head's own `1/logits_scaling` holds a slot for
        // either way, so the un-split multiplier is the slot that discriminates.)
        assert!(
            wiring.scalarmul_scales.contains(&unsplit),
            "{phase}: the un-folded attention multiplier must occupy a registry slot: {:?}",
            wiring.scalarmul_scales
        );
    }
}

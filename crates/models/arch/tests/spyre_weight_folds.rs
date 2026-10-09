// SPDX-License-Identifier: Apache-2.0
//! granite's foldable constants, pinned on the model they were built for.
//!
//! TWO families commute into staged weights, and each is one test:
//!   * the dsl scales every residual with `residual_multiplier` — `add(oproj * scalar(s), h)`
//!     and `add(down * scalar(s), h)` per layer — and each of those multipliers commutes into
//!     its matmul's weight (`(x·W)·s == x·(W·s)`), taking the ScalarMul op off the tape;
//!   * torch-spyre's attention scales the scores by pre-multiplying BOTH the query and the key
//!     by `√attention_multiplier`, and `(q·√s)·(k·√s)ᵀ == (q·kᵀ)·s` — so each `√s` commutes
//!     into the projection weight that produced its side (through the rope, which is linear),
//!     and the AttnDecode node stays on the tape at `scale = 1.0`, emitting no multiplies.
//!
//! This is the MODEL-side half of the pin: the spyre-crate tests prove the recognition and the
//! walk; these prove granite's own graph, through the real `#[forward]` expansion, admits
//! exactly the folds that structure implies — and that the folded scales left the const
//! registry the surviving programs read.
//!
//! The tied lm_head/embedding table stays OUT (its `logits_scaling` rides the splice's own
//! lm-head-tail fold, and the shared table refuses a per-consumer scale), and the
//! `embedding_multiplier` stays IN the registry (its producer is a gather, not a matmul) — both
//! asserted, so a recognition that silently widened or narrowed is a failure either way.

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
fn granite_folds_every_residual_multiplier_into_its_matmul_weight() {
    let residual = 0.22f32; // granite-3.1-2b-instruct config `residual_multiplier`
    let embed = 12.0f32; // config `embedding_multiplier` — a GATHER's scale, never foldable
    for (phase, wiring) in phases() {
        // TWO per layer — o_proj's and down_proj's — at the config's own multiplier. The count
        // is BY VALUE: the fold list is the union both constant families feed, so counting
        // entries (not the list) is what pins this family fired exactly twice per layer.
        let want = 2 * wiring.geometry.layers as usize;
        let folded = wiring
            .weight_scale_folds
            .iter()
            .filter(|(_, s)| s.to_bits() == residual.to_bits())
            .count();
        assert_eq!(
            folded, want,
            "{phase}: every layer's o_proj and down_proj residual multiplier folds"
        );
        // The folded scale LEFT the registry — no `[1,1]` const exists for a program that no
        // longer is, and a stale entry would shift the tid of every scale after it.
        assert!(
            !wiring
                .scalarmul_scales
                .iter()
                .any(|s| s.to_bits() == residual.to_bits()),
            "{phase}: the folded multiplier must not occupy a registry slot: {:?}",
            wiring.scalarmul_scales
        );
        // The embedding multiplier STAYS — its producer is the embedding GATHER, which no
        // constant commutes into, so it must still reach the device as a bound const.
        assert!(
            wiring.scalarmul_scales.contains(&embed),
            "{phase}: the embedding multiplier is a gather's scale and must stay in the \
             registry: {:?}",
            wiring.scalarmul_scales
        );
    }
}

#[test]
fn granite_folds_the_attention_scale_into_both_projection_weights() {
    let residual = 0.22f32; // config `residual_multiplier` — the OTHER family on the same list
    let attn = (1.0f32 / 64.0).sqrt(); // √(config `attention_multiplier`), what each side folds
    let unsplit = 1.0f32 / 64.0; // the config's own `attention_multiplier` — the score-side mul
    let eps = 1e-5f32; // config `rms_norm_eps` — a survivor the registry must still serve
    for (phase, wiring) in phases() {
        let layers = wiring.geometry.layers as usize;
        // TWO per layer — W_q's and W_k's — at `√attention_multiplier`, and TOGETHER with the
        // residual family's two, exactly the list: a recognition that folded one more or one
        // fewer thing on either side is wrong in either direction.
        let folded = wiring
            .weight_scale_folds
            .iter()
            .filter(|(_, s)| s.to_bits() == attn.to_bits())
            .count();
        assert_eq!(
            folded,
            2 * layers,
            "{phase}: every layer's W_q and W_k fold the attention scale"
        );
        assert_eq!(
            wiring.weight_scale_folds.len(),
            4 * layers,
            "{phase}: the fold list is exactly the residual and attention families, nothing \
             else: {:?}",
            wiring.weight_scale_folds
        );
        assert!(
            wiring
                .weight_scale_folds
                .iter()
                .any(|(_, s)| s.to_bits() == residual.to_bits()),
            "{phase}: the residual family shares this list; its absence means this phase lost \
             the residual fold: {:?}",
            wiring.weight_scale_folds
        );
        // The attention's OWN registry slot — the un-split `attention_multiplier` the AttnDecode
        // census arm pushed — is GONE: the arm leaves the census with the node folded, and a
        // stale entry would shift the tid of every scale after it. (√scale's slot cannot be
        // pinned by absence on this model: lm_head's own `1/logits_scaling` is the same bits,
        // 0.125, and keeps a slot for its own reason.)
        assert!(
            !wiring
                .scalarmul_scales
                .iter()
                .any(|s| s.to_bits() == unsplit.to_bits()),
            "{phase}: the folded attention scale must not occupy a registry slot: {:?}",
            wiring.scalarmul_scales
        );
        // The survivors the registry still serves: the rms eps (additive — no fold exists) and
        // lm_head's own 0.125.
        assert!(
            wiring.scalarmul_scales.contains(&eps),
            "{phase}: the rms eps is additive and must stay in the registry: {:?}",
            wiring.scalarmul_scales
        );
        assert!(
            wiring.scalarmul_scales.contains(&attn),
            "{phase}: lm_head's 1/logits_scaling keeps the 0.125 slot the attention's √scale \
             bit-collides with: {:?}",
            wiring.scalarmul_scales
        );
    }
}

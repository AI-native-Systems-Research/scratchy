// SPDX-License-Identifier: Apache-2.0
//! The ScalarMul weight fold, pinned on the model it was built for.
//!
//! granite's dsl scales every residual with `residual_multiplier` — `add(oproj * scalar(s), h)`
//! and `add(down * scalar(s), h)` per layer — and each of those multipliers commutes into its
//! matmul's weight (`(x·W)·s == x·(W·s)`), so the fold takes all of them off the tape at once.
//! This is the MODEL-side half of the pin: the spyre-crate tests prove the recognition and the
//! walk (`the_residual_scalarmul_folds_into_the_weight_and_off_the_tape`); this one proves
//! granite's own graph, through the real `#[forward]` expansion, admits exactly the folds that
//! structure implies — two per layer, at the config's multiplier — and that the folded scale
//! left the const registry the surviving programs read.
//!
//! The tied lm_head/embedding table stays OUT (its `logits_scaling` rides the splice's own
//! lm-head-tail fold, and the shared table refuses a per-consumer scale), and the
//! `embedding_multiplier` stays IN the registry (its producer is a gather, not a matmul) — both
//! asserted, so a recognition that silently widened or narrowed is a failure either way.

#![cfg(all(feature = "spyre", feature = "granite-3.1-2b-instruct"))]

#[test]
fn granite_folds_every_residual_multiplier_into_its_matmul_weight() {
    let w = &scratchy_models::granite::granite_3_1_2b_instruct::SUPERDSC_WIRINGS;
    let residual = 0.22f32; // granite-3.1-2b-instruct config `residual_multiplier`
    let embed = 12.0f32; // config `embedding_multiplier` — a GATHER's scale, never foldable
    let phases: &[(&str, &scratchy_target_spyre::wiring::Wiring)] = match &w.prefill {
        Some(p) => &[("decode", &w.decode), ("prefill", p)],
        None => panic!("granite bakes a batched-prefill program; the wiring has none"),
    };
    for (phase, wiring) in phases {
        // TWO per layer — o_proj's and down_proj's — at the config's own multiplier, and
        // nothing else: a recognition that folded one more or one fewer thing is wrong in
        // either direction.
        let want = 2 * wiring.geometry.layers as usize;
        assert_eq!(
            wiring.weight_scale_folds.len(),
            want,
            "{phase}: every layer's o_proj and down_proj residual multiplier folds"
        );
        assert!(
            wiring
                .weight_scale_folds
                .iter()
                .all(|(_, s)| s.to_bits() == residual.to_bits()),
            "{phase}: every folded multiplier is the config residual_multiplier (0.22), got {:?}",
            wiring.weight_scale_folds
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

// SPDX-License-Identifier: Apache-2.0
//! End-to-end for the Gemma2 DSL body. Proves:
//!
//! 1. The `#[forward]` proc-macro accepts `if`/`else` in the body.
//! 2. Every Gemma2 config in `crates/models/arch/configs/gemma2/` compiles
//!    through parse → classify → shape-infer → CFG → unroll → solve
//!    × workloads → schedule × workloads → codegen.
//! 3. The solver finds an Impl for every tile (no UnclaimedTile on
//!    `Gelu`, `SlidingAttention`, or `TanhSoftCap` at any M).
//! 4. Per-layer tile count matches the hand-computed expectation,
//!    confirming the conditional emits exactly one attention-family
//!    tile per unrolled iteration.

use scratchy_forward_compiler_macro::forward;

// The full Gemma2 body, mirrored from `scratchy-models/src/gemma2.rs`.
// Duplicated here (not `mod`-included) because proc-macro attributes
// can only annotate items declared in the crate being compiled, not
// re-exported from another crate.
#[forward(
    workloads = [1, 8, 64, 512, 4096],
)]
fn gemma2() {
    hidden_states = embed(input_ids, embed_tokens) * sqrt(hidden_size);
    for layer in 0..num_hidden_layers {
        pre_attn_normed = rmsnorm(hidden_states, input_layernorm[layer] + 1.0);

        q = gemm(pre_attn_normed, self_attn.q_proj[layer]);
        k = gemm(pre_attn_normed, self_attn.k_proj[layer]);
        v = gemm(pre_attn_normed, self_attn.v_proj[layer]);
        (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
        if layer % sliding_window_pattern == 0 {
            attn = sliding_attention(q, k, v, kv_cache[layer], block_table);
        } else {
            attn = attention(q, k, v, kv_cache[layer], block_table);
        }
        oproj = gemm(attn, self_attn.o_proj[layer]);

        post_attn_normed = rmsnorm(oproj, post_attention_layernorm[layer] + 1.0);

        hidden_states = add(post_attn_normed, hidden_states);
        pre_ffwd_normed = rmsnorm(hidden_states, pre_feedforward_layernorm[layer] + 1.0);

        gate = gelu(gemm(pre_ffwd_normed, mlp.gate_proj[layer]));
        up = gemm(pre_ffwd_normed, mlp.up_proj[layer]);
        down = gemm(gate * up, mlp.down_proj[layer]);

        post_ffwd_normed = rmsnorm(down, post_feedforward_layernorm[layer] + 1.0);

        hidden_states = add(post_ffwd_normed, hidden_states);
    }
    normed = rmsnorm(hidden_states, norm + 1.0);
    logits = gemm(normed, lm_head);
    capped = tanh_softcap(logits);
}

/// Tiles per unrolled Gemma2 layer body:
///
/// - pre-attn rmsnorm: Add(w, 1.0) + RmsNorm = 2
/// - Q/K/V gemms (3) + rope_append (1)
/// - exactly one attention-family tile (the `if` picks one arm)
/// - o_proj gemm (1)
/// - post-attn rmsnorm: Add + RmsNorm = 2
/// - residual add (1)
/// - pre-ffwd rmsnorm: Add + RmsNorm = 2
/// - GELU MLP: gate_gemm + gelu + up_gemm + mul + down_gemm = 5
/// - post-ffwd rmsnorm: Add + RmsNorm = 2
/// - residual add (1)
///
/// Total: 21 tiles per layer. Plus 1 embed at the top and 4
/// post-loop tiles (Add + final rmsnorm + lm_head gemm + tanh_softcap).
const GEMMA2_TILES_PER_LAYER: usize = 21;
const GEMMA2_PRE_LOOP_TILES: usize = 2; // embed + ScalarMul (embed scale)
const GEMMA2_POST_LOOP_TILES: usize = 4; // Add + final norm + lm_head + softcap

fn expected_tiles(num_hidden_layers: usize) -> usize {
    GEMMA2_PRE_LOOP_TILES + num_hidden_layers * GEMMA2_TILES_PER_LAYER + GEMMA2_POST_LOOP_TILES
}

#[test]
fn gemma2_2b_tile_count_matches_expected() {
    // Gemma2-2B: 26 layers → 2 + 26 × 21 + 4 = 552.
    assert_eq!(gemma2_2b::NUM_TILES, expected_tiles(26));
}

#[test]
fn gemma2_9b_tile_count_matches_expected() {
    // Gemma2-9B: 42 layers → 2 + 42 × 21 + 4 = 888.
    assert_eq!(gemma2_9b::NUM_TILES, expected_tiles(42));
}

#[test]
fn gemma2_27b_tile_count_matches_expected() {
    // Gemma2-27B: 46 layers → 2 + 46 × 21 + 4 = 972.
    assert_eq!(gemma2_27b::NUM_TILES, expected_tiles(46));
}

#[test]
fn every_gemma2_config_was_compiled() {
    // If any Gemma2 JSON failed to compile, this file wouldn't build.
    // The references force observation of each compiled specialization.
    let _: usize = gemma2_2b::NUM_TILES;
    let _: usize = gemma2_9b::NUM_TILES;
    let _: usize = gemma2_27b::NUM_TILES;
}

// `every_workload_bucket_solved_for_gemma2_2b` and
// `gemma2_predicted_us_is_finite_and_monotonic_in_m` used to read
// per-bucket `m_<N>::PREDICTED_US`. The cost-monotonicity invariant
// (prefill ≫ decode) lives in `solver::tests::
// prefill_cost_dwarfs_decode_cost`, which calls `solve()` directly
// for `llama-3.2-1b` — gemma2 shares the same cost model, so the
// llama-side check is a sufficient regression guard for the silent-
// drop-to-zero bug class. The per-bucket `m_X[_sk_Y]` stub modules
// are no longer emitted (~9k lines saved workspace-wide).

#[test]
fn arch_level_weights_enum_and_dispatch_exist_for_gemma2() {
    // The compiler emitted the arch-level `Weights` enum, its
    // per-model variants, and the dispatching `forward` fn.
    #[cfg(feature = "cuda")]
    {
        fn _probe_2b(w: gemma2_2b::Weights) -> Weights {
            Weights::Gemma2_2b(w)
        }
        fn _probe_9b(w: gemma2_9b::Weights) -> Weights {
            Weights::Gemma2_9b(w)
        }
        fn _probe_27b(w: gemma2_27b::Weights) -> Weights {
            Weights::Gemma2_27b(w)
        }
        let _: fn(gemma2_2b::Weights) -> Weights = _probe_2b;
        let _: fn(gemma2_9b::Weights) -> Weights = _probe_9b;
        let _: fn(gemma2_27b::Weights) -> Weights = _probe_27b;
    }
}

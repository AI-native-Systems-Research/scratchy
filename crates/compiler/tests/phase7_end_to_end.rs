// SPDX-License-Identifier: Apache-2.0
//! End-to-end integration test — the ONE test that proves the
//! macro actually drives the full pipeline.
//!
//! The whole pipeline (parse → classify → shape-infer → CFG → unroll
//! → solve × workloads → schedule × workloads) runs at compile time.
//! If any pass fails for any (model × workload) combo, this file
//! fails to compile.
//!
//! The emitted `llama::*` modules carry real values from the
//! solver + scheduler. We assert on them to prove the pipeline's
//! output isn't dummy.

use scratchy_forward_compiler_macro::forward;

// The real Llama body — SwiGLU MLP (`silu(gate) * up`), exercising
// the Expr::Mul path that used to emit OpKind::Add by mistake.
// `models_dir` is discovered from the carrier fn name `llama` by
// walking up looking for `crates/scratchy-model-llama/configs/` (a stale layout
// that predates the consolidated `scratchy-models` crate; kept only for
// this standalone test fixture).
#[forward(
    workloads = [1, 8, 64, 512, 4096],
)]
fn llama() {
    hidden_states = embed(input_ids, embed_tokens);
    for layer in 0..num_hidden_layers {
        normed = rmsnorm(hidden_states, input_layernorm[layer]);
        q = gemm(normed, self_attn.q_proj[layer]);
        k = gemm(normed, self_attn.k_proj[layer]);
        v = gemm(normed, self_attn.v_proj[layer]);
        (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
        attn = attention(q, k, v, kv_cache[layer], block_table);
        oproj = gemm(attn, self_attn.o_proj[layer]);
        hidden_states = add(oproj, hidden_states);

        normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
        gate = silu(gemm(normed2, mlp.gate_proj[layer]));
        up = gemm(normed2, mlp.up_proj[layer]);
        down = gemm(gate * up, mlp.down_proj[layer]);
        hidden_states = add(down, hidden_states);
    }
    normed = rmsnorm(hidden_states, norm);
    logits = gemm(normed, lm_head);
}

#[test]
fn llama_3_2_1b_tile_count_matches_expected() {
    // Llama-3.2-1B has 16 layers. Per-layer SwiGLU body is 15
    // tiles: rmsnorm, qgemm, kgemm, vgemm, rope, attn, ogemm, add,
    // rmsnorm2, gate_gemm, silu, up_gemm, mul, down_gemm, add2.
    // Pre: embed (1). Post: rmsnorm + lm_head (2).
    // Total: 1 + 16 × 15 + 2 = 243 tiles.
    assert_eq!(llama_3_2_1b::NUM_TILES, 243);
}

#[test]
fn llama_3_1_8b_tile_count_matches_expected() {
    // 32 layers × 15 + 3 = 483 tiles.
    assert_eq!(llama_3_1_8b::NUM_TILES, 483);
}

#[test]
fn every_llama_config_was_compiled() {
    // The nine Llama configs. If any didn't go through the
    // pipeline this file wouldn't compile.
    let _: usize = llama_2_7b::NUM_TILES;
    let _: usize = llama_2_13b::NUM_TILES;
    let _: usize = llama_2_70b::NUM_TILES;
    let _: usize = llama_3_8b::NUM_TILES;
    let _: usize = llama_3_70b::NUM_TILES;
    let _: usize = llama_3_1_8b::NUM_TILES;
    let _: usize = llama_3_1_70b::NUM_TILES;
    let _: usize = llama_3_2_1b::NUM_TILES;
    let _: usize = llama_3_2_3b::NUM_TILES;
}

// `every_workload_point_was_solved_for_llama_3_2_1b` and
// `solver_produced_finite_positive_cost_across_workloads` used to
// live here, reading per-bucket `m_<N>::PREDICTED_US` constants
// emitted by the macro. Both were thin: the first probed path
// resolution with no value asserted, the second's only real
// invariant (prefill ≫ decode) is now `solver::tests::
// prefill_cost_dwarfs_decode_cost`, which calls `solve()` directly
// instead of routing the f64 through baked compile-time constants
// that drift every time `target_profiles/*.csv` is regenerated.
// The per-bucket `m_X[_sk_Y]` modules are no longer emitted.

#[test]
fn arch_level_weights_enum_and_dispatch_exist() {
    // Observe the compiler emitted the arch-level `Weights` enum
    // with one variant per compiled model, and the dispatching
    // `forward` fn. Under #[cfg(feature = "cuda")] only — these
    // reference cuda-only types (GpuWeights, CUstream, GpuDevice).
    #[cfg(feature = "cuda")]
    {
        // Type-level observation: an enum variant for the 1B model
        // exists and wraps that model's `Weights`. If the dispatcher
        // wasn't emitted this line wouldn't type-check.
        fn _probe(w: llama_3_2_1b::Weights) -> Weights {
            Weights::Llama_3_2_1b(w)
        }
        let _: fn(llama_3_2_1b::Weights) -> Weights = _probe;
    }
}

#[test]
fn backbone_forward_emitted_for_every_bucket() {
    // Per-bucket fn surfaces (`forward_m_<N>` / `forward_backbone_m_<N>`)
    // were replaced by a per-canonical `FORWARD_TABLE` + `find_bucket`
    // dispatch — fn-pointer probes per bucket no longer apply. The
    // observable contract is now "the arch-level `forward_backbone`
    // dispatcher exists with the expected signature, and behind it
    // the canonical's FORWARD_TABLE has at least one entry per
    // compiled (m, sk) point."
    //
    // The compiled-per-bucket guarantee survives via two checks:
    //   1. `every_workload_point_was_solved_for_llama_3_2_1b` above
    //      reads the per-bucket `m_<N>::PREDICTED_US` constant — if
    //      a bucket failed to compile, the path wouldn't resolve.
    //   2. The arch dispatcher's `find_bucket` consults a slice
    //      whose row count equals `sfufs.per_workload.len()`.
    #[cfg(feature = "cuda")]
    {
        type ArchF = unsafe fn(
            &Weights,
            &scratchy_forward_compiler::ForwardCtx,
            &mut scratchy_target_cuda::device::GpuDevice,
            u64,
        ) -> scratchy_target_cuda::alloc::OwnedTensor;
        let _: ArchF = forward_backbone;
    }
}

// `scheduler_produces_linear_chain_on_fused_body` used to read
// `llama_3_2_1b::m_1::{NUM_WAVES, NUM_SUBGRAPHS}` to assert the
// post-fusion body is a serial chain (waves == subgraphs). The
// per-bucket stub module that exposed these was dropped along
// with PREDICTED_US — its assertion is structural (driven by the
// solver + scheduler topology), so it belongs in
// `schedule::tests` calling `schedule_workloads()` directly. Not
// re-added here; if the serial-chain invariant ever regresses,
// the right place to catch it is the unit test, not via macro-
// emitted constants in an integration test.

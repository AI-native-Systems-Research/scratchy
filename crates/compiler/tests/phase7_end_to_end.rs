// SPDX-License-Identifier: Apache-2.0
//! End-to-end integration test — the ONE test that proves the
//! macro actually drives the full pipeline.
//!
//! The whole pipeline (parse → classify → shape-infer → CFG → unroll
//! → tape × workloads) runs at compile time. If any pass fails for any
//! (model × workload) combo, this file fails to compile.
//!
//! The emitted `llama::*` modules carry real values from the pipeline.
//! We assert on them to prove its output isn't dummy.

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

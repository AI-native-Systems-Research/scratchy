# The gemma4-moe forward, in torch-idiom Python. Parsed by scratchy
# (compiler/frontend/src/parse_python.rs), never executed by the
# compiler. Executed under torch only by the CI oracle, where the SAME
# text is the reference implementation and the spec at once.
import torch
import torch.nn.functional as F


# 16/32 decoder rungs opt in here, on the only arch where they were measured
# to win: MoE expert kernels pay real dead compute for padded rows, so a
# 9-15-row batch on the 64 rung wastes ~5x expert work (conc-12 TPOT
# 140.8 -> 85.1 ms on gemma-4-26b, M1 Max). Dense TQ archs lose low-m
# utilization instead (granite +8.6%, qwen2.5-7b +26% at batch 12) and stay
# on the flat 8 -> 64 default ladder.
@forward(workloads=[1, 2, 4, 8, 16, 32, 64, 512, 1024, 2048, 4096])
def gemma4_moe():
    hidden_states = embed(input_ids, embed_tokens) * sqrt(hidden_size)
    for layer in range(num_hidden_layers):
        pre_attn_normed = rmsnorm(hidden_states, input_layernorm[layer])

        # ── ATTENTION — identical to dense gemma4 (dual-class) ──
        if layer % sliding_window_pattern == sliding_window_global_remainder:
            # The q/k norms are per head: each normalizes the per-head view (`heads` rows of
            # `global_head_dim`), then the rows go back to one per token. k_eq_v: the V unit norm
            # reads the same per-head K projection.
            qg = gemm(pre_attn_normed, self_attn.q_proj_global[layer])
            qg_heads = reshape(qg, [num_tokens * num_attention_heads, global_head_dim])
            qg_heads = rmsnorm(qg_heads, self_attn.q_norm_global[layer])
            qg = reshape(qg_heads, [num_tokens, num_attention_heads * global_head_dim])
            kvg = gemm(pre_attn_normed, self_attn.k_proj_global[layer])
            kvg_heads = reshape(kvg, [num_tokens * num_global_key_value_heads, global_head_dim])
            kg_heads = rmsnorm(kvg_heads, self_attn.k_norm_global[layer])
            kg = reshape(kg_heads, [num_tokens, num_global_key_value_heads * global_head_dim])
            vg_heads = rmsnorm_unit(kvg_heads)
            vg = reshape(vg_heads, [num_tokens, num_global_key_value_heads * global_head_dim])
            (qg, kg, vg) = rope_append(qg, kg, vg, positions, rotary, kv_cache[layer])
            attng = attention(qg, kg, vg, kv_cache[layer], block_table)
            oproj = gemm(attng, self_attn.o_proj_global[layer])
        else:
            qs = gemm(pre_attn_normed, self_attn.q_proj[layer])
            qs_heads = reshape(qs, [num_tokens * num_attention_heads, head_dim])
            qs_heads = rmsnorm(qs_heads, self_attn.q_norm[layer])
            qs = reshape(qs_heads, [num_tokens, num_attention_heads * head_dim])
            ks = gemm(pre_attn_normed, self_attn.k_proj[layer])
            ks_heads = reshape(ks, [num_tokens * num_key_value_heads, head_dim])
            ks_heads = rmsnorm(ks_heads, self_attn.k_norm[layer])
            ks = reshape(ks_heads, [num_tokens, num_key_value_heads * head_dim])
            vs = rmsnorm_unit(gemm(pre_attn_normed, self_attn.v_proj[layer]))
            (qs, ks, vs) = rope_append(qs, ks, vs, positions, rotary_local, kv_cache[layer])
            attns = sliding_attention(qs, ks, vs, kv_cache[layer], block_table)
            oproj = gemm(attns, self_attn.o_proj[layer])

        post_attn_normed = rmsnorm(oproj, post_attention_layernorm[layer])
        hidden_states = add(post_attn_normed, hidden_states)

        # ── DENSE GeGLU MLP path (h1) ──
        h1 = rmsnorm(hidden_states, pre_feedforward_layernorm[layer])
        gate1 = gelu(gemm(h1, mlp.gate_proj[layer]))
        up1 = gemm(h1, mlp.up_proj[layer])
        h1 = gemm(gate1 * up1, mlp.down_proj[layer])
        h1 = rmsnorm(h1, post_feedforward_layernorm_dense[layer])

        # ── SPARSE SwitchGLU MoE path (h2) ──
        # `gemma_moe` fuses the whole sparse block: it routes off the
        # post-attention residual (input 0) — a folded-gain RMSNorm
        # (`router.scale` · hidden^-0.5) → `router.proj` → top-k →
        # softmax → × `router.per_expert_scale` — and runs the selected
        # GeGLU experts on the pre-FF2-normed input (input 1), summing
        # by the routed weights. Routing indices/weights stay internal
        # to the op (no u32/f32 tiles cross the bf16 op boundary).
        h2 = rmsnorm(hidden_states, pre_feedforward_layernorm_moe[layer])
        h2 = gemma_moe(hidden_states, h2, router[layer], experts.switch_glu[layer])
        h2 = rmsnorm(h2, post_feedforward_layernorm_moe[layer])

        # ── COMBINE ──
        combined = rmsnorm(add(h1, h2), post_feedforward_layernorm[layer])
        hidden_states = add(hidden_states, combined)
        hidden_states = scalar_weight_mul(hidden_states, layer_scalar[layer])
    normed = rmsnorm(hidden_states, norm)
    logits = tanh_softcap(gemm(normed, lm_head))
    return logits

# The glm4-moe forward, in torch-idiom Python. Parsed by scratchy
# (compiler/macros/src/parse_python.rs), never executed by the
# compiler. Executed under torch only by the CI oracle, where the SAME
# text is the reference implementation and the spec at once.
#
# GLM-4.5-Air (`glm4_moe`): Qwen-style q/k/v projections WITH learned
# biases (`attention_bias: true`), 0.5 partial rotary (NeoX halves, like
# every non-traditional mlx rope), DeepSeek-V3-style MoE routing
# (sigmoid + e_score_correction_bias, flat top-k at n_group=1) with a
# plain-add shared expert under `mlp.shared_experts` (PLURAL, unlike
# Qwen's `mlp.shared_expert` — renamed in arch.json), and a dense
# SwiGLU MLP for `layer < first_k_dense_replace`.
import torch
import torch.nn.functional as F


@forward
def glm4_moe():
    hidden_states = embed(input_ids, embed_tokens)
    for layer in range(num_hidden_layers):
        normed = rmsnorm(hidden_states, input_layernorm[layer])
        q = gemm(normed, self_attn.q_proj[layer])
        q = bias_add(q, self_attn.q_proj.bias[layer])
        k = gemm(normed, self_attn.k_proj[layer])
        k = bias_add(k, self_attn.k_proj.bias[layer])
        v = gemm(normed, self_attn.v_proj[layer])
        v = bias_add(v, self_attn.v_proj.bias[layer])
        (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer])
        attn = attention(q, k, v, kv_cache[layer], block_table)
        oproj = gemm(attn, self_attn.o_proj[layer])
        hidden_states = add(oproj, hidden_states)

        normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer])
        if layer < first_k_dense_replace:
            # Dense SwiGLU MLP (layer 0).
            mlp_out = gemm(
                silu(gemm(normed2, mlp.gate_proj[layer])) * gemm(normed2, mlp.up_proj[layer]),
                mlp.down_proj[layer],
            )
        else:
            # GLM MoE: sigmoid+bias-routed experts (inside moe_block) +
            # the always-on shared expert — a plain-add SwiGLU MLP,
            # quantized with everything else (no sigmoid gate, unlike
            # Qwen's shared_expert).
            routed = moe_block(normed2, mlp[layer])
            shared_y = gemm(
                silu(gemm(normed2, mlp.shared_expert.gate_proj[layer]))
                * gemm(normed2, mlp.shared_expert.up_proj[layer]),
                mlp.shared_expert.down_proj[layer],
            )
            mlp_out = add(routed, shared_y)
        hidden_states = add(mlp_out, hidden_states)
    normed = rmsnorm(hidden_states, norm)
    logits = gemm(normed, lm_head)
    return logits

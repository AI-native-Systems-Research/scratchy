# The glm4-moe-lite forward, in torch-idiom Python. Parsed by scratchy
# (compiler/frontend/src/parse_python.rs), never executed by the
# compiler. Unlike the other carriers, no CI oracle executes it yet: the
# torch oracle has no implementation of the MLA decoder state's ops, so this
# text is the spec but not yet a reference implementation.
#
# GLM-4.7-Flash (`glm4_moe_lite`): glm4-moe's MoE half under multi-head
# latent attention (the `mla` decoder state, crates/state/decoder/mla). The query
# goes through a Q-LoRA bottleneck (q_a_proj, its norm, q_b_proj); the
# keys and values are one compressed latent plus one rope key every head
# shares (kv_a_proj_with_mqa, split). The append ropes q's rope part and
# the rope key and caches the latent row; the attention decompresses the
# latent per head (kv_b_proj). The MLP is glm4-moe's:
# sigmoid + e_score_correction_bias routed experts with a plain-add
# shared expert under `mlp.shared_experts` (renamed in arch.json), and a
# dense SwiGLU MLP for `layer < first_k_dense_replace`.
import torch
import torch.nn.functional as F


@forward
def glm4_moe_lite():
    hidden_states = embed(input_ids, embed_tokens)
    for layer in range(num_hidden_layers):
        normed = rmsnorm(hidden_states, input_layernorm[layer])
        q_a = rmsnorm(gemm(normed, self_attn.q_a_proj[layer]), self_attn.q_a_layernorm[layer])
        q = gemm(q_a, self_attn.q_b_proj[layer])
        kv_a = gemm(normed, self_attn.kv_a_proj_with_mqa[layer])
        (kv_latent, k_pe) = mla_split(kv_a)
        kv_latent = rmsnorm(kv_latent, self_attn.kv_a_layernorm[layer])
        (q, kv_latent, k_pe) = mla_append(q, kv_latent, k_pe, positions, rotary, mla_cache[layer])
        attn = mla_attention(q, kv_latent, k_pe, self_attn.kv_b_proj[layer],
                             mla_cache[layer], block_table)
        hidden_states = add(gemm(attn, self_attn.o_proj[layer]), hidden_states)

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

# The glm4-moe-lite forward, in torch-idiom Python. Parsed by scratchy
# (compiler/frontend/src/parse_python.rs), never executed by the
# compiler. Executed under torch only by the CI oracle, where the SAME
# text is the reference implementation and the spec at once.
#
# GLM-4.7-Flash (`glm4_moe_lite`): glm4-moe's MoE half under multi-head
# latent attention. The query goes through a Q-LoRA bottleneck (q_a_proj,
# its norm, q_b_proj); the keys and values are one compressed latent plus
# one rope key every head shares (kv_a_proj_with_mqa, split). The
# attention is the absorbed form: `embed_q` folds each head's key
# decompression into its no-rope query, the cache holds one latent row per
# token, and `unembed_out` decompresses each head's attended latent into its
# value (the two halves of the HF `kv_b_proj`, per head, as the MLX
# checkpoints ship them). The MLP is glm4-moe's: sigmoid +
# e_score_correction_bias routed experts with a plain-add shared expert
# under `mlp.shared_experts` (renamed in arch.json), and a dense SwiGLU
# MLP for `layer < first_k_dense_replace`.
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
        (q, kv_latent, k_pe) = mla_append(q, kv_latent, k_pe, positions, rotary, kv_cache[layer])
        q_latent = head_matmul(q, self_attn.embed_q[layer])
        latent = mla_attention(q_latent, q, k_pe, kv_cache[layer], block_table)
        attn = head_matmul(latent, self_attn.unembed_out[layer])
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

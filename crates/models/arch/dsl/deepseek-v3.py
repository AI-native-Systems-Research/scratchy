# The deepseek-v3 forward, in torch-idiom Python. Parsed by scratchy
# (ruff_python_parser -> the identical Ast the Rust costume produced),
# never executed by the compiler. Executed under torch only by the CI
# oracle, where the SAME text is the reference implementation and the
# spec at once.
import torch
import torch.nn.functional as F


@forward
def deepseek_v3():
    hidden_states = embed(input_ids, embed_tokens)
    for layer in range(num_hidden_layers):
        # ── Attention ──────────────────────────────────────────────
        normed = rmsnorm(hidden_states, input_layernorm[layer])

        # Q path: q_a_proj → q_a_layernorm → q_b_proj (q_lora_rank path)
        q_a = gemm(normed, self_attn.q_a_proj[layer])
        q_a = rmsnorm(q_a, self_attn.q_a_layernorm[layer])
        q = gemm(q_a, self_attn.q_b_proj[layer])

        # KV path: kv_a_proj_with_mqa → mla_split → kv_a_layernorm → kv_b_proj
        kv_a = gemm(normed, self_attn.kv_a_proj_with_mqa[layer])
        (kv_latent, k_pe) = mla_split(kv_a)
        kv_latent = rmsnorm(kv_latent, self_attn.kv_a_layernorm[layer])
        kv_b = gemm(kv_latent, self_attn.kv_b_proj[layer])

        # MLA attention: assembles K/V, applies interleaved RoPE, writes cache
        attn = mla_attention(
            q,
            kv_b,
            k_pe,
            positions,
            rotary,
            kv_cache[layer],
            block_table,
        )
        oproj = gemm(attn, self_attn.o_proj[layer])
        hidden_states = add(oproj, hidden_states)

        # ── MLP / MoE ──────────────────────────────────────────────
        normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer])
        if layer < first_k_dense_replace:
            # Dense SwiGLU MLP (layer 0 only).
            mlp_out = gemm(
                silu(gemm(normed2, mlp.gate_proj[layer])) * gemm(normed2, mlp.up_proj[layer]),
                mlp.down_proj[layer],
            )
        else:
            # DeepSeek MoE: sigmoid-routed experts + shared expert
            mlp_out = moe_block(normed2, moe[layer])
        hidden_states = add(mlp_out, hidden_states)
    normed = rmsnorm(hidden_states, norm)
    logits = gemm(normed, lm_head)
    return logits

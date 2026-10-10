# The Qwen3.5/3.6 multi-token-prediction head, in torch-idiom Python. Parsed by scratchy
# (compiler/macros/src/parse_python.rs), never executed by the compiler.
#
# Drafts the token after next: row t reads the target model's final (post-norm) hidden state
# `target_hidden[t]` and the embedding of the token that follows it, normalizes each, fuses them
# with `fc`, and runs one full-attention decoder layer (its own KV cache, the target's positions)
# with the target's sparse-MoE MLP. `embed_tokens` / `lm_head` are the target's. The final
# `normed` is what the next draft depth reads as its own `target_hidden`.
import torch
import torch.nn.functional as F


@forward
def qwen3_5_mtp():
    embeds = rmsnorm(embed(input_ids, embed_tokens), pre_fc_norm_embedding)
    hidden = rmsnorm(target_hidden, pre_fc_norm_hidden)
    hidden_states = gemm(concat(embeds, hidden), fc)
    for layer in range(num_hidden_layers):
        normed = rmsnorm(hidden_states, input_layernorm[layer])

        # Full attention with sigmoid output gate, as the target's periodic layers: q_proj is
        # doubled and gate_split deinterleaves per-head [query | gate]; q/k get per-head
        # RMSNorm; partial RoPE.
        qg = gemm(normed, self_attn.q_proj[layer])
        (q, gate) = gate_split(qg)
        q = rmsnorm(q, self_attn.q_norm[layer])
        k = gemm(normed, self_attn.k_proj[layer])
        k = rmsnorm(k, self_attn.k_norm[layer])
        v = gemm(normed, self_attn.v_proj[layer])
        (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer])
        attn = attention(q, k, v, kv_cache[layer], block_table)
        attn = gate_apply(attn, gate)
        mixer_out = gemm(attn, self_attn.o_proj[layer])
        hidden_states = add(mixer_out, hidden_states)

        # Sparse MoE MLP plus the sigmoid-gated shared expert, as the target's.
        normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer])
        routed = moe_block(normed2, mlp[layer])
        shared_y = gemm(
            silu(gemm(normed2, mlp.shared_expert.gate_proj[layer]))
            * gemm(normed2, mlp.shared_expert.up_proj[layer]),
            mlp.shared_expert.down_proj[layer],
        )
        g = gemm(normed2, mlp.shared_expert_gate[layer])
        mlp_out = gate_scale(routed, shared_y, g)
        hidden_states = add(mlp_out, hidden_states)
    normed = rmsnorm(hidden_states, norm)
    logits = gemm(normed, lm_head)
    return logits

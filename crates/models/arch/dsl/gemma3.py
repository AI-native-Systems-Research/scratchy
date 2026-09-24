# The gemma3 forward, in torch-idiom Python. Parsed by scratchy
# (ruff_python_parser -> the identical Ast the Rust costume produced),
# never executed by the compiler. Executed under torch only by the CI
# oracle, where the SAME text is the reference implementation and the
# spec at once.
import torch
import torch.nn.functional as F


@forward
def gemma3():
    hidden_states = embed(input_ids, embed_tokens) * sqrt(hidden_size)
    for layer in range(num_hidden_layers):
        pre_attn_normed = rmsnorm(hidden_states, input_layernorm[layer] + 1.0)

        q = gemm(pre_attn_normed, self_attn.q_proj[layer])
        q = rmsnorm(q, self_attn.q_norm[layer] + 1.0)
        k = gemm(pre_attn_normed, self_attn.k_proj[layer])
        k = rmsnorm(k, self_attn.k_norm[layer] + 1.0)
        v = gemm(pre_attn_normed, self_attn.v_proj[layer])

        if layer % sliding_window_pattern == sliding_window_global_remainder:
            (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer])
            attn = attention(q, k, v, kv_cache[layer], block_table)
        else:
            (q, k, v) = rope_append(q, k, v, positions, rotary_local, kv_cache[layer])
            attn = sliding_attention(q, k, v, kv_cache[layer], block_table)

        oproj = gemm(attn, self_attn.o_proj[layer])
        post_attn_normed = rmsnorm(oproj, post_attention_layernorm[layer] + 1.0)
        hidden_states = add(post_attn_normed, hidden_states)

        pre_ffwd_normed = rmsnorm(hidden_states, pre_feedforward_layernorm[layer] + 1.0)
        gate = gelu(gemm(pre_ffwd_normed, mlp.gate_proj[layer]))
        up = gemm(pre_ffwd_normed, mlp.up_proj[layer])
        down = gemm(gate * up, mlp.down_proj[layer])
        post_ffwd_normed = rmsnorm(down, post_feedforward_layernorm[layer] + 1.0)
        hidden_states = add(post_ffwd_normed, hidden_states)
    normed = rmsnorm(hidden_states, norm + 1.0)
    logits = gemm(normed, lm_head)
    return logits

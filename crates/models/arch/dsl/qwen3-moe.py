# The qwen3-moe forward, in torch-idiom Python. Parsed by scratchy
# (compiler/frontend/src/parse_python.rs), never executed by the
# compiler. Executed under torch only by the CI oracle, where the SAME
# text is the reference implementation and the spec at once.
import torch
import torch.nn.functional as F


@forward
def qwen3_moe():
    hidden_states = embed(input_ids, embed_tokens)
    for layer in range(num_hidden_layers):
        normed = rmsnorm(hidden_states, input_layernorm[layer])
        q = gemm(normed, self_attn.q_proj[layer])
        q_heads = reshape(q, [num_tokens * num_attention_heads, head_dim])
        q_heads = rmsnorm(q_heads, self_attn.q_norm[layer])
        q = reshape(q_heads, [num_tokens, num_attention_heads * head_dim])
        k = gemm(normed, self_attn.k_proj[layer])
        k_heads = reshape(k, [num_tokens * num_key_value_heads, head_dim])
        k_heads = rmsnorm(k_heads, self_attn.k_norm[layer])
        k = reshape(k_heads, [num_tokens, num_key_value_heads * head_dim])
        v = gemm(normed, self_attn.v_proj[layer])
        (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer])
        attn = attention(q, k, v, kv_cache[layer], block_table)
        oproj = gemm(attn, self_attn.o_proj[layer])
        hidden_states = add(oproj, hidden_states)

        normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer])
        mlp_out = moe_block(normed2, mlp[layer])
        hidden_states = add(mlp_out, hidden_states)
    normed = rmsnorm(hidden_states, norm)
    logits = gemm(normed, lm_head)
    return logits

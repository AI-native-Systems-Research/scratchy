# The mixtral forward, in torch-idiom Python. Parsed by scratchy
# (compiler/macros/src/parse_python.rs), never executed by the
# compiler. Executed under torch only by the CI oracle, where the SAME
# text is the reference implementation and the spec at once.
import torch
import torch.nn.functional as F


@forward
def mixtral():
    hidden_states = embed(input_ids, embed_tokens)
    for layer in range(num_hidden_layers):
        normed = rmsnorm(hidden_states, input_layernorm[layer])
        q = gemm(normed, self_attn.q_proj[layer])
        k = gemm(normed, self_attn.k_proj[layer])
        v = gemm(normed, self_attn.v_proj[layer])
        (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer])
        attn = attention(q, k, v, kv_cache[layer], block_table)
        oproj = gemm(attn, self_attn.o_proj[layer])
        hidden_states = add(oproj, hidden_states)

        normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer])
        mlp_out = moe_block(normed2, block_sparse_moe[layer])
        hidden_states = add(mlp_out, hidden_states)
    normed = rmsnorm(hidden_states, norm)
    logits = gemm(normed, lm_head)
    return logits

# The qwen3 forward, in torch-idiom Python. Parsed by scratchy
# (compiler/frontend/src/parse_python.rs), never executed by the
# compiler. Executed under torch only by the CI oracle, where the SAME
# text is the reference implementation and the spec at once.
#
# Dialect: the carrier is the `@forward`-decorated `def` (the
# decorator carries the compile metadata).
# Statements are `name = expr`, `(a, b, c) = expr`, `for ivar in
# range(<bound>)`, `if <layer predicate>:` (with `else:`). Expressions
# are op calls, attribute chains (weight refs), `[layer]` indexing,
# `*` / `+`, and numeric literals. Anything else is a parse error.
import torch
import torch.nn.functional as F


@forward
def qwen3():
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
        gate = silu(gemm(normed2, mlp.gate_proj[layer]))
        up = gemm(normed2, mlp.up_proj[layer])
        down = gemm(gate * up, mlp.down_proj[layer])
        hidden_states = add(down, hidden_states)
    normed = rmsnorm(hidden_states, norm)
    logits = gemm(normed, lm_head)
    return logits

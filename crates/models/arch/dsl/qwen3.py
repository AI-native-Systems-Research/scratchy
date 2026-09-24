# The same forward, in torch-idiom Python — parsed by the same
# pipeline (ruff_python_parser -> the identical Ast parse.rs
# produces), never executed by the compiler. Executed under torch
# only by the CI oracle, where the SAME text is the reference
# implementation and the spec at once.
#
# Dialect: statements are `name = expr`, `(a, b, c) = expr`,
# `for ivar in range(<bound>)`, `if <layer predicate>:`. Expressions
# are op calls, attribute chains (weight refs), `[layer]` indexing,
# `*` / `+`, and numeric literals. Anything else is a parse error.
import torch
import torch.nn.functional as F


def forward_qwen3():
    hidden_states = embed(input_ids, embed_tokens)
    for layer in range(num_hidden_layers):
        normed = rmsnorm(hidden_states, input_layernorm[layer])
        q = gemm(normed, self_attn.q_proj[layer])
        q = rmsnorm(q, self_attn.q_norm[layer])
        k = gemm(normed, self_attn.k_proj[layer])
        k = rmsnorm(k, self_attn.k_norm[layer])
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

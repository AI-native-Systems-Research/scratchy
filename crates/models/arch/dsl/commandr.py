# The commandr forward, in torch-idiom Python. Parsed by scratchy
# (ruff_python_parser -> the identical Ast the Rust costume produced),
# never executed by the compiler. Executed under torch only by the CI
# oracle, where the SAME text is the reference implementation and the
# spec at once.
import torch
import torch.nn.functional as F


@forward
def commandr():
    hidden_states = embed(input_ids, embed_tokens)
    for layer in range(num_hidden_layers):
        # CohereLayerNorm expressed as plain math: subtract row mean,
        # then RMS-normalize. The `(mean, sub, rmsnorm)` trio is
        # claimed by `MeanSubRmsNormImpl` and emits one
        # `cohere_layer_norm` kernel call — same runtime path as the
        # retired `OpKind::LayerNorm` opcode.
        mu = mean(hidden_states)
        centered = sub(hidden_states, mu)
        normed = rmsnorm(centered, input_layernorm[layer])

        # Attention branch.
        q = gemm(normed, self_attn.q_proj[layer])
        k = gemm(normed, self_attn.k_proj[layer])
        v = gemm(normed, self_attn.v_proj[layer])
        (q, k, v) = rope_append_interleaved(q, k, v, positions, rotary, kv_cache[layer])
        attn = attention(q, k, v, kv_cache[layer], block_table)
        oproj = gemm(attn, self_attn.o_proj[layer])

        # MLP branch (SwiGLU, same `normed` input).
        gate = silu(gemm(normed, mlp.gate_proj[layer]))
        up = gemm(normed, mlp.up_proj[layer])
        down = gemm(gate * up, mlp.down_proj[layer])

        # Three-way residual: hidden += attn; hidden += mlp.
        hidden_states = add(oproj, hidden_states)
        hidden_states = add(down, hidden_states)
    # Final norm (same factorization) + lm_head + logit scale.
    mu = mean(hidden_states)
    centered = sub(hidden_states, mu)
    normed = rmsnorm(centered, norm)
    logits = gemm(normed, lm_head) * scalar(logit_scale)
    return logits

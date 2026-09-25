# The modernbert forward, in torch-idiom Python. Parsed by scratchy
# (compiler/macros/src/parse_python.rs), never executed by the
# compiler. Executed under torch only by the CI oracle, where the SAME
# text is the reference implementation and the spec at once.
import torch
import torch.nn.functional as F


@forward
def modernbert():
    # Embedding lookup + initial CohereLayerNorm.
    hidden_states = embed(input_ids, embeddings.tok_embeddings)
    hidden_states = rmsnorm(sub(hidden_states, mean(hidden_states)), embeddings.norm)

    for layer in range(num_hidden_layers):
        # Attn pre-norm. Layer 0 is identity (HF checkpoint has no
        # `attn_norm.weight` for layer 0; the embeddings LN already
        # did the work). The `* 1.0` passthrough satisfies the
        # if/else merge-carry rule (both branches must bind
        # `normed`) without needing a layer-0 weight load.
        if layer < 1:
            normed = hidden_states * 1.0
        else:
            normed = rmsnorm(sub(hidden_states, mean(hidden_states)), attn_norm[layer])

        q = gemm(normed, attn.q_proj[layer])
        k = gemm(normed, attn.k_proj[layer])
        v = gemm(normed, attn.v_proj[layer])

        # Dual rotary: every Nth layer rotates with the global
        # theta cache, the rest with the local-theta cache. The
        # 3-arg `attention(q, k, v)` form picks
        # `EncoderAttentionImpl` (bidirectional FA2, no causal mask,
        # no KV cache read) — the kv_cache extern threaded into
        # `rope_append` is the per-step write path; the encoder
        # never reads those stored K/V back.
        if layer % global_attn_every_n_layers == 0:
            (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer])
            attn_out = attention(q, k, v)
        else:
            (q, k, v) = rope_append(q, k, v, positions, rotary_local, kv_cache[layer])
            attn_out = attention(q, k, v)
        oproj = gemm(attn_out, attn.Wo[layer])
        hidden_states = add(oproj, hidden_states)

        # MLP pre-norm + GeGLU + residual.
        normed = rmsnorm(sub(hidden_states, mean(hidden_states)), mlp_norm[layer])
        gate = gelu(gemm(normed, mlp.gate_proj[layer]))
        up = gemm(normed, mlp.up_proj[layer])
        down = gemm(gate * up, mlp.Wo[layer])
        hidden_states = add(down, hidden_states)

    # Final encoder norm — backbone terminator (1d). The output of
    # this RmsNorm is `[num_tokens, hidden_size]`; the dispatch
    # layer reads it via `forward_backbone` (1f).
    hidden_states = rmsnorm(sub(hidden_states, mean(hidden_states)), final_norm)
    return hidden_states

# The qwen2-5-vl vision forward, in torch-idiom Python. Parsed by
# scratchy (compiler/macros/src/parse_python.rs), never executed by
# the compiler. Executed under torch only by the CI oracle, where the
# SAME text is the reference implementation and the spec at once.
import torch
import torch.nn.functional as F


@vision_forward(workloads=[256, 1024, 4096, 16384], processor=crate.PROCESSOR)
def qwen2_5_vl():
    hidden_states = gemm(pixels, patch_embed.proj)

    # Window-permute hidden_states at S² (= vision_merge_factor) row
    # granularity so each window contains a contiguous run.
    hidden_states = reshape(
        hidden_states,
        [
            num_tokens / vision_merge_factor,
            vision_merge_factor * vision_embed_dim,
        ],
    )
    hidden_states = embedding_gather(hidden_states, window_index)
    hidden_states = reshape(hidden_states, [num_tokens, vision_embed_dim])

    for layer in range(vision_depth):
        normed = rmsnorm(hidden_states, norm1[layer])
        q = gemm(normed, attn.q[layer])
        q = bias_add(q, attn.q.bias[layer])
        k = gemm(normed, attn.k[layer])
        k = bias_add(k, attn.k.bias[layer])
        v = gemm(normed, attn.v[layer])
        v = bias_add(v, attn.v.bias[layer])
        (q, k) = vision_rope(q, k, cos, sin)
        if layer in [7, 15, 23, 31]:
            attn_out = varlen_attention(q, k, v, cu_seqlens_full, max_seqlen_full)
        else:
            attn_out = varlen_attention(q, k, v, cu_seqlens_window, max_seqlen_window)
        oproj = gemm(attn_out, attn.proj[layer])
        oproj = bias_add(oproj, attn.proj.bias[layer])
        hidden_states = add(oproj, hidden_states)

        normed2 = rmsnorm(hidden_states, norm2[layer])
        # SwiGLU MLP. Explicit bias_add tiles after each gemm so the
        # (Gemm, BiasAdd) matcher routes through FusedGemmBias —
        # singleton Instruction::Gemm skips bias entirely. See lib
        # docstring for the full rationale.
        gate = silu(gemm(normed2, mlp.gate_proj[layer]))
        up = gemm(normed2, mlp.up_proj[layer])
        down = gemm(gate * up, mlp.down_proj[layer])
        down = bias_add(down, mlp.down_proj.bias[layer])
        hidden_states = add(down, hidden_states)

    merged = rmsnorm(hidden_states, merger.ln_q)
    merged = reshape(
        merged,
        [num_tokens / vision_merge_factor, vision_merge_hidden],
    )
    mlp0 = gemm(merged, merger.mlp_0)
    mlp0 = bias_add(mlp0, merger.mlp_0.bias)
    mlp0 = gelu_erf(mlp0)
    projected = gemm(mlp0, merger.mlp_2)
    projected = bias_add(projected, merger.mlp_2.bias)
    out = embedding_gather(projected, reverse_indices)
    return out

# The qwen2-vl vision forward, in torch-idiom Python. Parsed by
# scratchy (ruff_python_parser -> the identical Ast the Rust costume
# produced), never executed by the compiler. Executed under torch only
# by the CI oracle, where the SAME text is the reference
# implementation and the spec at once.
import torch
import torch.nn.functional as F


@vision_forward(workloads=[256, 1024, 4096, 16384], processor=crate.PROCESSOR)
def qwen2_vl():
    hidden_states = gemm(pixels, patch_embed.proj)

    for layer in range(vision_depth):
        # PyTorch `nn.LayerNorm` decomposes as
        # `(mean, sub, rmsnorm, bias_add)`. The
        # `MeanSubRmsNormBiasAddImpl` matcher claims the 4-tile chain
        # and lowers to one `kernels::layer_norm_bias` call; the
        # bias-side weight ref is structural (the loader pulls both
        # `<prefix>.weight` and `<prefix>.bias` from the rmsnorm's
        # weight ref via the `LayerNorm` wrapper).
        m1 = mean(hidden_states)
        c1 = sub(hidden_states, m1)
        n1 = rmsnorm(c1, norm1[layer])
        normed = bias_add(n1, norm1.bias[layer])
        q = gemm(normed, attn.q[layer])
        q = bias_add(q, attn.q.bias[layer])
        k = gemm(normed, attn.k[layer])
        k = bias_add(k, attn.k.bias[layer])
        v = gemm(normed, attn.v[layer])
        v = bias_add(v, attn.v.bias[layer])
        (q, k) = vision_rope(q, k, cos, sin)
        attn_out = varlen_attention(q, k, v, cu_seqlens, max_seqlen)
        oproj = gemm(attn_out, attn.proj[layer])
        oproj = bias_add(oproj, attn.proj.bias[layer])
        hidden_states = add(oproj, hidden_states)

        m2 = mean(hidden_states)
        c2 = sub(hidden_states, m2)
        n2 = rmsnorm(c2, norm2[layer])
        normed2 = bias_add(n2, norm2.bias[layer])
        fc1 = gemm(normed2, mlp.fc1[layer])
        fc1 = bias_add(fc1, mlp.fc1.bias[layer])
        fc1 = quick_gelu(fc1)
        fc2 = gemm(fc1, mlp.fc2[layer])
        fc2 = bias_add(fc2, mlp.fc2.bias[layer])
        hidden_states = add(fc2, hidden_states)

    mq = mean(hidden_states)
    cq = sub(hidden_states, mq)
    nq = rmsnorm(cq, merger.ln_q)
    merged = bias_add(nq, merger.ln_q.bias)
    merged = reshape(
        merged,
        [num_tokens / vision_merge_factor, vision_merge_hidden],
    )
    mlp0 = gemm(merged, merger.mlp_0)
    mlp0 = bias_add(mlp0, merger.mlp_0.bias)
    mlp0 = gelu_erf(mlp0)
    out = gemm(mlp0, merger.mlp_2)
    out = bias_add(out, merger.mlp_2.bias)
    return out

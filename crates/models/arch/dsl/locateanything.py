# The locateanything vision forward, in torch-idiom Python. Parsed by
# scratchy (ruff_python_parser -> the identical Ast the Rust costume
# produced), never executed by the compiler. Executed under torch only
# by the CI oracle, where the SAME text is the reference
# implementation and the spec at once.
import torch
import torch.nn.functional as F


@vision_forward(workloads=[256, 1024, 4096, 16384, 25600], processor=crate.PROCESSOR)
def locateanything():
    hidden_states = gemm(pixels, patch_embed.proj)
    hidden_states = bias_add(hidden_states, patch_embed.proj.bias)
    # Learned positional embedding (bicubic `Learnable2DInterpPosEmb`,
    # host-side; carried as the `pos_embeds` extern).
    hidden_states = add(pos_embeds, hidden_states)

    for layer in range(vision_depth):
        # LayerNorm-with-bias as the (mean, sub, rmsnorm, bias_add) 4-tile chain.
        m1 = mean(hidden_states)
        c1 = sub(hidden_states, m1)
        n1 = rmsnorm(c1, norm0[layer])
        normed = bias_add(n1, norm0.bias[layer])
        q = gemm(normed, attn.q[layer])
        q = bias_add(q, attn.q.bias[layer])
        k = gemm(normed, attn.k[layer])
        k = bias_add(k, attn.k.bias[layer])
        v = gemm(normed, attn.v[layer])
        v = bias_add(v, attn.v.bias[layer])
        (q, k) = vision_rope(q, k, cos, sin)
        attn_out = varlen_attention(q, k, v, cu_seqlens, max_seqlen)
        oproj = gemm(attn_out, attn.wo[layer])
        oproj = bias_add(oproj, attn.wo.bias[layer])
        hidden_states = add(oproj, hidden_states)

        m2 = mean(hidden_states)
        c2 = sub(hidden_states, m2)
        n2 = rmsnorm(c2, norm1[layer])
        normed2 = bias_add(n2, norm1.bias[layer])
        fc1 = gemm(normed2, mlp.fc0[layer])
        fc1 = bias_add(fc1, mlp.fc0.bias[layer])
        fc1 = gelu(fc1)
        fc2 = gemm(fc1, mlp.fc1[layer])
        fc2 = bias_add(fc2, mlp.fc1.bias[layer])
        hidden_states = add(fc2, hidden_states)

    # final LayerNorm, then the 2×2 patch merge — a PURE reshape under
    # window-major packing (mlx `patch_merger`'s reshape/transpose
    # collapses to row grouping; validated err 0.0 vs golden).
    mf = mean(hidden_states)
    cf = sub(hidden_states, mf)
    nf = rmsnorm(cf, final_layernorm)
    fl = bias_add(nf, final_layernorm.bias)
    merged = reshape(fl, [num_tokens / vision_merge_factor, vision_merge_hidden])

    # multi_modal_projector: LayerNorm(4608) → linear_1 → gelu_erf
    # (EXACT erf — the block MLPs above use the tanh approx; the two
    # flavors are numerically distinct) → linear_2 (→ d_model 2048).
    mp = mean(merged)
    cp = sub(merged, mp)
    pnorm = rmsnorm(cp, mm.layer_norm)
    pn = bias_add(pnorm, mm.layer_norm.bias)
    p1 = gemm(pn, mm.proj_in)
    p1 = bias_add(p1, mm.proj_in.bias)
    p1 = gelu_erf(p1)
    out = gemm(p1, mm.proj_out)
    out = bias_add(out, mm.proj_out.bias)
    return out

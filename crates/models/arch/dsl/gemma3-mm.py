# The gemma3-mm vision forward, in torch-idiom Python. Parsed by
# scratchy (compiler/macros/src/parse_python.rs), never executed by
# the compiler. Executed under torch only by the CI oracle, where the
# SAME text is the reference implementation and the spec at once.
import torch
import torch.nn.functional as F


@vision_forward(workloads=[256, 1024, 4096, 16384], processor=crate.PROCESSOR)
def gemma3_mm():
    # Patch embed: Conv2d(in=3, out=1152, k=14, s=14) flattened at
    # load time to a [1152, 588] linear. With bias.
    hidden_states = gemm(pixels, embeddings.patch_embedding)
    hidden_states = bias_add(hidden_states, embeddings.patch_embedding.bias)

    # Learned positional embedding lookup. `position_ids` is a vision-
    # prelude extern of `[num_tokens]` u32 the wrapper builds as
    # `[0..vision_num_positions, ...]` per image; the table is the
    # 2D `[vision_num_positions, vision_embed_dim]` learned weight.
    pos_emb = pos_embed(position_ids, embeddings.position_embedding)
    hidden_states = add(hidden_states, pos_emb)

    for layer in range(vision_depth):
        # Pre-norm self-attention. PyTorch `nn.LayerNorm` (γ+β)
        # decomposes as `(mean, sub, rmsnorm, bias_add)`; the
        # `MeanSubRmsNormBiasAddImpl` matcher claims the 4-tile chain
        # and lowers to one `kernels::layer_norm_bias` call.
        m1 = mean(hidden_states)
        c1 = sub(hidden_states, m1)
        n1 = rmsnorm(c1, layer_norm1[layer])
        normed = bias_add(n1, layer_norm1.bias[layer])

        q = gemm(normed, self_attn.q_proj[layer])
        q = bias_add(q, self_attn.q_proj.bias[layer])
        k = gemm(normed, self_attn.k_proj[layer])
        k = bias_add(k, self_attn.k_proj.bias[layer])
        v = gemm(normed, self_attn.v_proj[layer])
        v = bias_add(v, self_attn.v_proj.bias[layer])

        # SigLIP runs full attention per-image. Single cu_seqlens
        # segments the batched-flat tensor at image boundaries (one
        # segment per image, max_seqlen = vision_num_positions).
        attn_out = varlen_attention(q, k, v, cu_seqlens, max_seqlen)

        oproj = gemm(attn_out, self_attn.out_proj[layer])
        oproj = bias_add(oproj, self_attn.out_proj.bias[layer])
        hidden_states = add(oproj, hidden_states)

        # Pre-norm MLP (GELU-tanh, biases on both fc1 / fc2).
        m2 = mean(hidden_states)
        c2 = sub(hidden_states, m2)
        n2 = rmsnorm(c2, layer_norm2[layer])
        normed2 = bias_add(n2, layer_norm2.bias[layer])

        fc1 = gemm(normed2, mlp.fc1[layer])
        fc1 = bias_add(fc1, mlp.fc1.bias[layer])
        fc1 = gelu(fc1)
        fc2 = gemm(fc1, mlp.fc2[layer])
        fc2 = bias_add(fc2, mlp.fc2.bias[layer])
        hidden_states = add(fc2, hidden_states)

    # Post-encoder LayerNorm (γ+β) — same 4-tile decomposition.
    mp = mean(hidden_states)
    cp = sub(hidden_states, mp)
    np = rmsnorm(cp, post_layernorm)
    hidden_states = bias_add(np, post_layernorm.bias)

    # MM projector: AvgPool2d(k=4) collapses the 64×64 patch grid to
    # 16×16 = 256 tokens, RMSNorm (no bias) on the pooled features,
    # matmul-only `nn.Parameter` projection to text d_model.
    #
    # Gemma3RMSNorm uses `output * (1.0 + weight)`, NOT `output * weight`
    # (so zero-init weights are identity-equivalent). Same `+ 1.0` as
    # every RMSNorm site in the text-decoder body (the `gemma3` arch).
    # Without this, garbled output: the projector RMSNorm scales the
    # post-pool features by ~zero (initial weights are tiny floats), the
    # projection collapses, and the decoder receives near-noise embeds.
    pooled = avg_pool_2d(hidden_states)
    normed_pool = rmsnorm(pooled, mm.mm_soft_emb_norm + 1.0)
    out = gemm(normed_pool, mm.mm_input_projection_weight)
    return out

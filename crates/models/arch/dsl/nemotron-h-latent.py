# The nemotron-h-latent forward, in torch-idiom Python. Parsed by
# scratchy (compiler/macros/src/parse_python.rs), never executed by the
# compiler. Executed under torch only by the CI oracle, where the SAME
# text is the reference implementation and the spec at once.
#
# Nemotron-H with LatentMoE (Nemotron-3 Super; mlx-lm `nemotron_h.py`
# with `moe_latent_size` set): the nemotron-h forward, except that the
# routed experts run in a narrower latent space — `fc1_latent_proj`
# projects the block input down, the experts read that latent while the
# router still reads the full-width input, and the expert-weighted sum
# (linear, so it commutes with the projection) is projected back up by
# `fc2_latent_proj` before the full-width shared expert is added.
import torch
import torch.nn.functional as F


@forward
def nemotron_h_latent():
    hidden_states = embed(input_ids, embeddings)
    for layer in range(num_hidden_layers):
        normed = rmsnorm(hidden_states, norm[layer])
        if layer in mamba_layers:
            # ── Mamba-2: in_proj → [z | xBC | dt]; causal conv1d +
            # silu on xBC; selective scan with D skip; gate-before-norm
            # grouped RMSNorm; out_proj.
            zxbcdt = gemm(normed, mamba.in_proj[layer])
            y = mamba2(zxbcdt, mamba[layer])
            mixer_out = gemm(y, mamba.out_proj[layer])
        else:
            if layer in attention_layers:
                # ── GQA attention, no positional rotation: K/V go to
                # the paged cache as projected.
                q = gemm(normed, attn.q_proj[layer])
                k = gemm(normed, attn.k_proj[layer])
                v = gemm(normed, attn.v_proj[layer])
                (k, v) = kv_append(k, v, kv_cache[layer])
                a = attention(q, k, v, kv_cache[layer], block_table)
                mixer_out = gemm(a, attn.o_proj[layer])
            else:
                if layer in moe_layers:
                    # ── LatentMoE: route on the full-width input, run
                    # the relu2 experts on its latent projection, sum in
                    # latent space, project back; add the full-width
                    # relu2 shared expert unscaled.
                    latent = gemm(normed, moe.fc1_latent_proj[layer])
                    routed = moe_block(normed, latent, moe[layer])
                    routed = gemm(routed, moe.fc2_latent_proj[layer])
                    shared = gemm(
                        relu2(gemm(normed, moe.shared_experts.up_proj[layer])),
                        moe.shared_experts.down_proj[layer],
                    )
                    mixer_out = add(routed, shared)
                else:
                    # ── Dense relu2 MLP (`-`).
                    mixer_out = gemm(
                        relu2(gemm(normed, mlp.up_proj[layer])),
                        mlp.down_proj[layer],
                    )
        hidden_states = add(mixer_out, hidden_states)
    normed = rmsnorm(hidden_states, norm_f)
    logits = gemm(normed, lm_head)
    return logits

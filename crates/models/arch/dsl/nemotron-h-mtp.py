# The Nemotron-3 Super multi-token-prediction head, in torch-idiom Python. Parsed by scratchy
# (compiler/macros/src/parse_python.rs), never executed by the compiler.
#
# DeepSeek-V3-style MTP (`num_nextn_predict_layers: 1`): drafts the token after next. Row t
# reads the target model's final (post-norm) hidden state `target_hidden[t]` and the embedding
# of the token that follows it, RMSNorms each, fuses them with `eh_proj`, and runs its own two
# decoder layers — `mtp_hybrid_override_pattern`, an attention layer then a LatentMoE one
# (`*E`), the target's own mix in miniature. Its attention has NO positional rotation (as the
# target's) and its OWN KV cache (not the target's). `embed_tokens` / `lm_head` are the
# target's, lent by the loader; the logits span the full vocabulary (no prefix slice). The
# final `normed` is what the next draft depth reads as its own `target_hidden`.
import torch
import torch.nn.functional as F


@forward
def nemotron_h_mtp():
    embeds = rmsnorm(embed(input_ids, embed_tokens), enorm)
    hidden = rmsnorm(target_hidden, hnorm)
    hidden_states = gemm(concat(embeds, hidden), eh_proj)
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
    normed = rmsnorm(hidden_states, final_layernorm)
    logits = gemm(normed, lm_head)
    return logits

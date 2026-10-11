# The nemotron-h forward, in torch-idiom Python. Parsed by scratchy
# (compiler/macros/src/parse_python.rs), never executed by the
# compiler. Executed under torch only by the CI oracle, where the SAME
# text is the reference implementation and the spec at once.
#
# Nemotron-H (mlx-lm `nemotron_h.py`): every layer is ONE pre-norm
# block, `x = x + mixer(rmsnorm(x, norm[layer]))`, its mixer picked by
# the config's `hybrid_override_pattern` through the layer sets it
# derives — Mamba-2 (`M`), GQA attention with NO RoPE (`*`), the
# sigmoid-routed relu2 MoE with a relu2 shared expert (`E`), or a dense
# relu2 MLP (`-`). On disk every mixer is `backbone.layers.N.mixer`;
# arch.json renames the per-kind DSL names onto it.
import torch
import torch.nn.functional as F


@forward
def nemotron_h():
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
                    # ── MoE: sigmoid + e_score_correction_bias top-k
                    # routing, non-gated relu2 experts (inside
                    # moe_block), plus the unscaled relu2 shared expert.
                    routed = moe_block(normed, moe[layer])
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

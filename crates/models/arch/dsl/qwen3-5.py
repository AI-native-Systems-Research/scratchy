# The qwen3-5 forward, in torch-idiom Python. Parsed by scratchy
# (compiler/frontend/src/parse_python.rs), never executed by the
# compiler. Executed under torch only by the CI oracle, where the SAME
# text is the reference implementation and the spec at once.
import torch
import torch.nn.functional as F


# The 16 and 32 rungs hold a 2-4-lane DFlash2 verify (lanes x 8 rows) on decode-shaped kernels,
# as qwen3-5-moe's do: on the default ladder a 2-lane verify of Qwen3.8-27B ran the 64-row tape,
# 47.7 ms of GPU a forward (M5 Max) against 38.7 at a lane's 8 rows and Splash's 38.74 ms 2-lane
# cycle. A head shares its target's rungs.
@forward(workloads=[1, 2, 4, 8, 16, 32, 64, 512, 1024, 2048, 4096])
def qwen3_5():
    hidden_states = embed(input_ids, embed_tokens)
    for layer in range(num_hidden_layers):
        normed = rmsnorm(hidden_states, input_layernorm[layer])

        # Token mixer: GDN linear attention for `l % interval != 3`,
        # full attention (with output gate) for the periodic `l % interval == 3`.
        if layer % full_attention_interval != 3:
            # ── Gated-DeltaNet linear attention ─────────────────────
            qkv = gemm(normed, linear_attn.in_proj_qkv[layer])
            z = gemm(normed, linear_attn.in_proj_z[layer])
            a = gemm(normed, linear_attn.in_proj_a[layer])
            b = gemm(normed, linear_attn.in_proj_b[layer])
            core = gated_delta_net(qkv, z, a, b, linear_attn[layer])
            mixer_out = gemm(core, linear_attn.out_proj[layer])
        else:
            # ── Full attention with sigmoid output gate ─────────────
            # q_proj is doubled; gate_split deinterleaves per-head
            # [query | gate]. q/k get per-head RMSNorm; partial RoPE.
            qg = gemm(normed, self_attn.q_proj[layer])
            (q, gate) = gate_split(qg)
            q_heads = reshape(q, [num_tokens * num_attention_heads, head_dim])
            q_heads = rmsnorm(q_heads, self_attn.q_norm[layer])
            q = reshape(q_heads, [num_tokens, num_attention_heads * head_dim])
            k = gemm(normed, self_attn.k_proj[layer])
            k_heads = reshape(k, [num_tokens * num_key_value_heads, head_dim])
            k_heads = rmsnorm(k_heads, self_attn.k_norm[layer])
            k = reshape(k_heads, [num_tokens, num_key_value_heads * head_dim])
            v = gemm(normed, self_attn.v_proj[layer])
            (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer])
            attn = attention(q, k, v, kv_cache[layer], block_table)
            # Sigmoid output gate: attn * sigmoid(gate), fused (bare `*` /
            # `sigmoid` aren't DSL-callable — Silu/Mul are synthesis-only).
            attn = gate_apply(attn, gate)
            mixer_out = gemm(attn, self_attn.o_proj[layer])
        hidden_states = add(mixer_out, hidden_states)

        # Dense SwiGLU MLP.
        normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer])
        mlp_out = gemm(
            silu(gemm(normed2, mlp.gate_proj[layer])) * gemm(normed2, mlp.up_proj[layer]),
            mlp.down_proj[layer],
        )
        hidden_states = add(mlp_out, hidden_states)
    normed = rmsnorm(hidden_states, norm)
    logits = gemm(normed, lm_head)
    return logits

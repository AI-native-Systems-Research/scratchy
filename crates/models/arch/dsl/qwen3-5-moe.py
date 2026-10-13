# The qwen3-5-moe forward, in torch-idiom Python. Parsed by scratchy
# (compiler/macros/src/parse_python.rs), never executed by the
# compiler. Executed under torch only by the CI oracle, where the SAME
# text is the reference implementation and the spec at once.
import torch
import torch.nn.functional as F


# 16/32 decoder rungs, as gemma4-moe takes them: without them a 9-63-row step runs the 64 rung,
# whose expert matvecs run every padded row's pairs. Qwen3.6-35B-A3B with its MTP head, base M5,
# greedy batches of N decoding together: N=16 144.6 -> 177.2 output tok/s, N=12 133.5 -> 144.9,
# N=4 and 8 (rungs they already had) unchanged; a prompt on the 16 rung's first token 62.1 ->
# 57.9 ms, on the 32 rung 106.4 -> 106.0.
@forward(workloads=[1, 2, 4, 8, 16, 32, 64, 512, 1024, 2048, 4096])
def qwen3_5_moe():
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
            q = rmsnorm(q, self_attn.q_norm[layer])
            k = gemm(normed, self_attn.k_proj[layer])
            k = rmsnorm(k, self_attn.k_norm[layer])
            v = gemm(normed, self_attn.v_proj[layer])
            (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer])
            attn = attention(q, k, v, kv_cache[layer], block_table)
            # Sigmoid output gate: attn * sigmoid(gate), fused (bare `*` /
            # `sigmoid` aren't DSL-callable — Silu/Mul are synthesis-only).
            attn = gate_apply(attn, gate)
            mixer_out = gemm(attn, self_attn.o_proj[layer])
        hidden_states = add(mixer_out, hidden_states)

        # Sparse MoE MLP: routed experts (softmax → top-8 → renorm →
        # SwitchGLU → weighted sum, all inside moe_block) + the always-on
        # shared expert — a quantized SwiGLU MLP scaled by the per-token
        # sigmoid gate ([T, 1], row-broadcast inside gate_scale).
        normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer])
        routed = moe_block(normed2, mlp[layer])
        shared_y = gemm(
            silu(gemm(normed2, mlp.shared_expert.gate_proj[layer]))
            * gemm(normed2, mlp.shared_expert.up_proj[layer]),
            mlp.shared_expert.down_proj[layer],
        )
        g = gemm(normed2, mlp.shared_expert_gate[layer])
        mlp_out = gate_scale(routed, shared_y, g)
        hidden_states = add(mlp_out, hidden_states)
    normed = rmsnorm(hidden_states, norm)
    logits = gemm(normed, lm_head)
    return logits

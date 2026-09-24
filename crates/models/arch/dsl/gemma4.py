# The gemma4 forward, in torch-idiom Python. Parsed by scratchy
# (ruff_python_parser -> the identical Ast the Rust costume produced),
# never executed by the compiler. Executed under torch only by the CI
# oracle, where the SAME text is the reference implementation and the
# spec at once.
import torch
import torch.nn.functional as F


@forward
def gemma4():
    hidden_states = embed(input_ids, embed_tokens) * sqrt(hidden_size)
    for layer in range(num_hidden_layers):
        pre_attn_normed = rmsnorm(hidden_states, input_layernorm[layer])

        # Branch-local variable names are deliberately DISTINCT per
        # class — the if/else merge unifies same-named assignments
        # across branches, and the two classes' q/k/v widths differ
        # (sliding 16×256 vs global 16×512). The branches converge at
        # `oproj`, which is [.., hidden_size] in both.
        if layer % sliding_window_pattern == sliding_window_global_remainder:
            # ── GLOBAL class: head_dim 512, 1 kv head, k_eq_v ──
            qg = gemm(pre_attn_normed, self_attn.q_proj_global[layer])
            qg = rmsnorm(qg, self_attn.q_norm_global[layer])
            kvg = gemm(pre_attn_normed, self_attn.k_proj_global[layer])
            kg = rmsnorm(kvg, self_attn.k_norm_global[layer])
            vg = rmsnorm_unit(kvg)
            (qg, kg, vg) = rope_append(qg, kg, vg, positions, rotary, kv_cache[layer])
            attng = attention(qg, kg, vg, kv_cache[layer], block_table)
            oproj = gemm(attng, self_attn.o_proj_global[layer])
        else:
            # ── SLIDING class: head_dim 256, 8 kv heads, window 1024 ──
            qs = gemm(pre_attn_normed, self_attn.q_proj[layer])
            qs = rmsnorm(qs, self_attn.q_norm[layer])
            ks = gemm(pre_attn_normed, self_attn.k_proj[layer])
            ks = rmsnorm(ks, self_attn.k_norm[layer])
            vs = rmsnorm_unit(gemm(pre_attn_normed, self_attn.v_proj[layer]))
            (qs, ks, vs) = rope_append(qs, ks, vs, positions, rotary_local, kv_cache[layer])
            attns = sliding_attention(qs, ks, vs, kv_cache[layer], block_table)
            oproj = gemm(attns, self_attn.o_proj[layer])

        post_attn_normed = rmsnorm(oproj, post_attention_layernorm[layer])
        hidden_states = add(post_attn_normed, hidden_states)

        pre_ffwd_normed = rmsnorm(hidden_states, pre_feedforward_layernorm[layer])
        gate = gelu(gemm(pre_ffwd_normed, mlp.gate_proj[layer]))
        up = gemm(pre_ffwd_normed, mlp.up_proj[layer])
        down = gemm(gate * up, mlp.down_proj[layer])
        post_ffwd_normed = rmsnorm(down, post_feedforward_layernorm[layer])
        hidden_states = add(post_ffwd_normed, hidden_states)

        hidden_states = scalar_weight_mul(hidden_states, layer_scalar[layer])
    normed = rmsnorm(hidden_states, norm)
    logits = tanh_softcap(gemm(normed, lm_head))
    return logits

# The gemma2 forward, in torch-idiom Python. Parsed by scratchy
# (ruff_python_parser -> the identical Ast the Rust costume produced),
# never executed by the compiler. Executed under torch only by the CI
# oracle, where the SAME text is the reference implementation and the
# spec at once.
import torch
import torch.nn.functional as F


@forward
def gemma2():
    # Gemma scales embeddings by sqrt(hidden_size) — matches vllm
    # Python `hidden_states *= self.normalizer` (layernorm.py line
    # 304) and hand-written `kernels::scale_inplace(hidden_states,
    # embed_scale, ..)` in scratchy-serving-cuda's `Gemma2ForCausalLM::forward`.
    # The `sqrt(hidden_size)` folds to a compile-time f64 per-model
    # at CFG build; the `* scalar` triggers the ScalarMulImpl which
    # emits `scale_inplace` on the embed output.
    hidden_states = embed(input_ids, embed_tokens) * sqrt(hidden_size)
    for layer in range(num_hidden_layers):
        # Pre-attention norm. Gemma's `(1+w)` convention rides as
        # a scalar addition on the weight ref — the solver's
        # ScalarOffsetRmsNormImpl claims the `(Add(w, 1.0), rmsnorm)`
        # pair and emits a single rms_norm call with the 1.0 as
        # the kernel's `weight_offset`.
        pre_attn_normed = rmsnorm(hidden_states, input_layernorm[layer] + 1.0)

        # Attention block.
        q = gemm(pre_attn_normed, self_attn.q_proj[layer])
        k = gemm(pre_attn_normed, self_attn.k_proj[layer])
        v = gemm(pre_attn_normed, self_attn.v_proj[layer])
        (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer])
        # Gemma2 alternates: even layers are sliding, odd are full.
        # HF default `layer_is_sliding[i] = (i % sliding_window_pattern == 0)`;
        # matches scratchy-serving-cuda hand-written `i % 2 == 0` → sliding.
        if layer % sliding_window_pattern == 0:
            attn = sliding_attention(q, k, v, kv_cache[layer], block_table)
        else:
            attn = attention(q, k, v, kv_cache[layer], block_table)
        oproj = gemm(attn, self_attn.o_proj[layer])

        # Post-attention norm on the attention output, before residual.
        post_attn_normed = rmsnorm(oproj, post_attention_layernorm[layer] + 1.0)

        # Residual + pre-ffwd norm.
        hidden_states = add(post_attn_normed, hidden_states)
        pre_ffwd_normed = rmsnorm(hidden_states, pre_feedforward_layernorm[layer] + 1.0)

        # GELU MLP — solver claims `(gemm, gemm, gelu, mul)` as a
        # FusedGateUpGeluMul subgraph.
        gate = gelu(gemm(pre_ffwd_normed, mlp.gate_proj[layer]))
        up = gemm(pre_ffwd_normed, mlp.up_proj[layer])
        down = gemm(gate * up, mlp.down_proj[layer])

        # Post-ffwd norm on the MLP output, before residual.
        post_ffwd_normed = rmsnorm(down, post_feedforward_layernorm[layer] + 1.0)

        hidden_states = add(post_ffwd_normed, hidden_states)
    # Final norm + lm_head + logit softcap.
    normed = rmsnorm(hidden_states, norm + 1.0)
    logits = gemm(normed, lm_head)
    capped = tanh_softcap(logits)
    return capped

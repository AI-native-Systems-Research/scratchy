# The gpt-oss forward, in torch-idiom Python. Parsed by scratchy
# (compiler/macros/src/parse_python.rs), never executed by the
# compiler. Executed under torch only by the CI oracle, where the SAME
# text is the reference implementation and the spec at once.
#
# gpt-oss (20b/120b): Qwen-shaped q/k/v/o projections, ALL with
# learned biases (`attention_bias: true`), full NeoX rotary under YaRN
# (theta 150000, factor 32), and alternating layer classes —
# `layer_types` derives pattern 2, remainder 1: even layers sliding
# window 128, odd layers full. Every attention carries per-head
# learned SINK logits (`self_attn.sinks`, an extra unscaled softmax
# column that never touches the ·V accumulation). The MLP is gpt-oss
# MoE: a biased linear router over raw logits, top-4, softmax over
# exactly the four picks, SwiGLU-OAI experts with per-expert linear
# biases — all fused inside `gptoss_moe` (routing indices/weights stay
# internal to the op; the ±7 clamps are baked into the kernels).
import torch
import torch.nn.functional as F


@forward
def gpt_oss():
    hidden_states = embed(input_ids, embed_tokens)
    for layer in range(num_hidden_layers):
        normed = rmsnorm(hidden_states, input_layernorm[layer])
        q = gemm(normed, self_attn.q_proj[layer])
        q = bias_add(q, self_attn.q_proj.bias[layer])
        k = gemm(normed, self_attn.k_proj[layer])
        k = bias_add(k, self_attn.k_proj.bias[layer])
        v = gemm(normed, self_attn.v_proj[layer])
        v = bias_add(v, self_attn.v_proj.bias[layer])
        (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer])
        # layer_types ["sliding_attention", "full_attention", ...]:
        # even layers slide (window 128), odd layers attend fully.
        if layer % sliding_window_pattern == sliding_window_global_remainder:
            attn = sink_attention(q, k, v, self_attn.sinks[layer], kv_cache[layer], block_table)
        else:
            attn = sink_sliding_attention(q, k, v, self_attn.sinks[layer], kv_cache[layer], block_table)
        oproj = gemm(attn, self_attn.o_proj[layer])
        oproj = bias_add(oproj, self_attn.o_proj.bias[layer])
        hidden_states = add(oproj, hidden_states)

        normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer])
        # router(+bias) → top-4 → softmax over the four → SwiGLU-OAI
        # experts with per-expert linear biases, weighted sum — fused.
        mlp_out = gptoss_moe(normed2, mlp[layer])
        hidden_states = add(mlp_out, hidden_states)
    normed = rmsnorm(hidden_states, norm)
    logits = gemm(normed, lm_head)
    return logits

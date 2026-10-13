# The Gemma 4 assistant drafter (`Gemma4AssistantForCausalLM`), in torch-idiom Python. Parsed by
# scratchy (compiler/macros/src/parse_python.rs), never executed by the compiler.
#
# Drafts the token after next from a FIXED position: row t reads its target's final (post-norm)
# hidden state `target_hidden[t]` — or, past the first draft, the drafter's own projected one — and
# the target's embedding of the token after it, and runs four Gemma 4 decoder layers whose attention
# has no K/V of its own: every layer's query attends its TARGET's KV of the target's last layer of
# the same class (`shared_attention`, `shared_sliding_attention`), up to the row's own position.
# The query ropes at the row's position (`rope_query`). The logits are the drafter's own tied
# embedding's; the hidden state the next draft reads is `post_projection` of the final norm's.
import torch
import torch.nn.functional as F


@forward
def gemma4_assistant():
    embeds = embed(input_ids, target_embed_tokens) * sqrt(backbone_hidden_size)
    hidden_states = gemm(concat(embeds, target_hidden), pre_projection)
    for layer in range(num_hidden_layers):
        pre_attn_normed = rmsnorm(hidden_states, input_layernorm[layer])

        # Branch-local names per class, as the target's: the two classes' query widths differ
        # (sliding 16x256, global 16x512); they converge at `oproj`.
        if layer % sliding_window_pattern == sliding_window_global_remainder:
            qg = gemm(pre_attn_normed, self_attn.q_proj_global[layer])
            qg = rmsnorm(qg, self_attn.q_norm_global[layer])
            qg = rope_query(qg, positions, rotary)
            attng = shared_attention(qg, kv_cache[layer], block_table)
            oproj = gemm(attng, self_attn.o_proj_global[layer])
        else:
            qs = gemm(pre_attn_normed, self_attn.q_proj[layer])
            qs = rmsnorm(qs, self_attn.q_norm[layer])
            qs = rope_query(qs, positions, rotary_local)
            attns = shared_sliding_attention(qs, kv_cache[layer], block_table)
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
    projected = export_hidden(gemm(normed, post_projection))
    logits = gemm(normed, lm_head)
    return logits

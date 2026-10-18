# The P-EAGLE drafter (AWS's parallel-drafting EAGLE-3, `amazon/GPT-OSS-20B-P-EAGLE`), in
# torch-idiom Python. Parsed by scratchy (compiler/macros/src/parse_python.rs), never executed by
# the compiler.
#
# Row t reads its target's residual stream entering three of the target's layers, concatenated
# (`target_hidden[t]`: EAGLE-3's auxiliary hidden states), and the embedding of the token after
# it; a row it drafts past the newest reads its learned placeholders instead — the mask token's
# embedding and `mask_hidden` — so one forward drafts every token at once. `fc` fuses the three
# hidden states into one, the residual its first layer adds to. That layer (`midlayer`) attends
# over [its embedding's norm | the fused hidden state's norm]; three plain Llama decoder layers
# follow, each with its own KV. The logits are its target's lm_head's, which it borrows. Its
# attention projections carry biases where its config's `attention_bias` says
# (`attention_bias_layers`: 120B's drafter has them, 20B's not); without, the `* 1.0`
# passthrough (the if/else merge-carry rule) runs no command.
import torch
import torch.nn.functional as F


@forward
def p_eagle():
    hidden_states = gemm(target_hidden, fc)
    for layer in range(num_hidden_layers):
        if layer < 1:
            embeds = rmsnorm(embed(input_ids, embed_tokens), midlayer.input_layernorm)
            x = concat(embeds, rmsnorm(hidden_states, midlayer.hidden_norm))
            q = gemm(x, midlayer.self_attn.q_proj)
            k = gemm(x, midlayer.self_attn.k_proj)
            v = gemm(x, midlayer.self_attn.v_proj)
            if layer in attention_bias_layers:
                q = bias_add(q, midlayer.self_attn.q_proj.bias)
                k = bias_add(k, midlayer.self_attn.k_proj.bias)
                v = bias_add(v, midlayer.self_attn.v_proj.bias)
            else:
                q = q * 1.0
                k = k * 1.0
                v = v * 1.0
        else:
            normed = rmsnorm(hidden_states, input_layernorm[layer])
            q = gemm(normed, self_attn.q_proj[layer])
            k = gemm(normed, self_attn.k_proj[layer])
            v = gemm(normed, self_attn.v_proj[layer])
            if layer in attention_bias_layers:
                q = bias_add(q, self_attn.q_proj.bias[layer])
                k = bias_add(k, self_attn.k_proj.bias[layer])
                v = bias_add(v, self_attn.v_proj.bias[layer])
            else:
                q = q * 1.0
                k = k * 1.0
                v = v * 1.0
        (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer])
        attn = attention(q, k, v, kv_cache[layer], block_table)
        if layer < 1:
            oproj = gemm(attn, midlayer.self_attn.o_proj)
            if layer in attention_bias_layers:
                oproj = bias_add(oproj, midlayer.self_attn.o_proj.bias)
            else:
                oproj = oproj * 1.0
        else:
            oproj = gemm(attn, self_attn.o_proj[layer])
            if layer in attention_bias_layers:
                oproj = bias_add(oproj, self_attn.o_proj.bias[layer])
            else:
                oproj = oproj * 1.0
        hidden_states = add(oproj, hidden_states)

        if layer < 1:
            normed2 = rmsnorm(hidden_states, midlayer.post_attention_layernorm)
            gate = silu(gemm(normed2, midlayer.mlp.gate_proj))
            up = gemm(normed2, midlayer.mlp.up_proj)
            down = gemm(gate * up, midlayer.mlp.down_proj)
        else:
            normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer])
            gate = silu(gemm(normed2, mlp.gate_proj[layer]))
            up = gemm(normed2, mlp.up_proj[layer])
            down = gemm(gate * up, mlp.down_proj[layer])
        hidden_states = add(down, hidden_states)
    normed = rmsnorm(hidden_states, norm)
    logits = gemm(normed, lm_head)
    return logits

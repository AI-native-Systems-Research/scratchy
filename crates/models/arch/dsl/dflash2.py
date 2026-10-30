# Splash's DFlash2 block drafter (`DFlash2DraftModel`, incoai), in torch-idiom Python. Parsed by
# scratchy (compiler/macros/src/parse_python.rs), never executed by the compiler.
#
# Splash's two draft graphs (runtime/model/DFlashDraft.cpp), one tape: its plays run them apart.
#
# The context commit (`DFlashDraft::addContextCommit`): each row's captured target layer rows
# (`target_hidden`, the target's `target_layer_ids` outputs side by side) through `fc` and
# `hidden_norm`, then each layer's key and value rows of its fused QKV projection, normed, roped and
# written into the layer's ring at the row's position (`draft_context_kv`). Each ring op rotates
# its keys (and queries) at the row's position as Splash does, from the model's rotary base, in
# fp32: no rotary table. A row is in the ring once its target row is: every key a block attends is
# a committed context row. The context rows go into the lane's context window too
# (`draft_window_store`, Splash `DFlashContextWindow`), 4-bit: a prefix-cache state keeps the
# window, and a hit rebuilds the lane's rings from it (`DFlashDraft::addContextRebuild`), a play of
# the window's load (`draft_window_load`) and the layers' commits alone.
#
# The block draft (`DFlashDraft::addDecode`): eight rows a lane, its anchor and seven mask tokens,
# embedded by the target's table. Each layer runs Splash's dynamic convolutions around its attention
# and its MLP (`draft_conv`: a stage weighs a channel's row and the row before it by a base weight
# plus its group's dynamic weight; the second stage adds the residual), and its attention over the
# ring and the block's own eight rows, which every row sees (`draft_attention`). The final norm
# feeds the target's lm_head and the candidate selector (`draft_select`): the top-16 candidates of
# each proposal row, scored with the bigram codebooks from the previous choice, walked into the
# seven proposals, and the rows the target verifies.
import torch
import torch.nn.functional as F


@forward
def dflash2():
    context = rmsnorm(gemm(target_hidden, fc), hidden_norm)
    stored = draft_window_store(context, positions)
    context = draft_window_load(stored, positions)
    for layer in range(num_hidden_layers):
        context_kv = gemm(context, self_attn.kv_proj[layer])
        committed = draft_context_kv(context_kv, self_attn.k_norm[layer], positions)

    hidden_states = embed(input_ids, embed_tokens)
    for layer in range(num_hidden_layers):
        normed = rmsnorm(hidden_states, input_layernorm[layer])
        dynamic = gemm(normed, attention_conv.kernel_projection[layer])
        convolved = draft_conv(normed, dynamic, attention_conv.base_kernel[layer])
        qkv = gemm(convolved, self_attn.qkv_proj[layer])
        attn = draft_attention(
            qkv, self_attn.q_norm[layer], self_attn.k_norm[layer], positions
        )
        projected = gemm(attn, self_attn.o_proj[layer])
        residual = draft_conv(projected, dynamic, attention_conv.base_kernel[layer], hidden_states)

        normed2 = rmsnorm(residual, post_attention_layernorm[layer])
        dynamic2 = gemm(normed2, mlp_conv.kernel_projection[layer])
        convolved2 = draft_conv(normed2, dynamic2, mlp_conv.base_kernel[layer])
        gate = silu(gemm(convolved2, mlp.gate_proj[layer]))
        up = gemm(convolved2, mlp.up_proj[layer])
        down = gemm(gate * up, mlp.down_proj[layer])
        hidden_states = draft_conv(down, dynamic2, mlp_conv.base_kernel[layer], residual)
    normed = rmsnorm(hidden_states, norm)
    selector = gemm(normed, candidate_selector.hidden_projection)
    logits = gemm(normed, lm_head)
    tokens = draft_select(
        logits,
        selector,
        candidate_selector.predecessor_codebook,
        candidate_selector.successor_codebook,
    )
    return tokens

# SPDX-License-Identifier: Apache-2.0
"""spyre-inference's `batched_decode_head_major_kernel`
(`spyre_inference/v1/attention/ops/batched_decode_head_major.py`, spyre-inference 297fc48), in
Triton: decode attention for several sequences in one launch over a head-major KV cache, each index
selecting a whole page with all its kv heads.

Their kernel, statement for statement, in the form their tiled walk takes
(`SPYRE_ATTN_FOR_EACH_TILE=1`, their default: `split_index`). Each block below names the line it is.
The operands are theirs, in their order:

* `query` `[num_tokens, num_heads * head_size]`, the whole batch's query rows;
* `rep_row_ids` `[entries]` int32, entry `j * num_seqs + s` holding sequence `s`'s query row;
* `k_pages` / `v_pages` the cache `[num_pages_total, num_kv_heads, block_size, head_size]`;
* `chunk_page_ids` `[num_chunks, entries, 1]` int32, chunk `c`'s entry `j * num_seqs + s` holding
  the page of block slot `j` of sequence `s`;
* `mask_by_chunk` `[num_chunks, blocks_per_chunk, num_seqs, num_kv_heads, 1, block_size]`;
* `out`, whose first `num_seqs` rows the kernel stores (`out[:num_seqs].copy_(attn)`).

`entries = blocks_per_chunk * num_seqs`. The walk is `walk_tiles` over the `num_chunks` chunks, the
carry seeded `(-inf, 0, 0)`. With more than one chunk each block slot keeps its own running softmax
(`_entry_local_update`), and the slots merge once after the walk (`_merge_entry_local`); with one
chunk the slots share one maximum per sequence (`reduce_chunk`).

Where Triton spells a torch op differently, the spelling is Triton's and the op is theirs:

* `tl.dot` is at most rank 3, so `torch.matmul(q, k_page.transpose(-2, -1))` over
  `[entries, kv, group, D] x [entries, kv, D, block]` is the rank-3 dot over `[entries * kv, group,
  D]`, the same per-(entry, kv) products, and its result is viewed back at
  `[blocks_per_chunk, num_seqs, kv, group, block]`; the `pv` matmul likewise;
* a `keepdim` reduction is the reduction with its axis re-inserted at its use (`x[..., None]`), so
  the softmax state is `[slots, num_seqs, kv, group]`, their `[slots, num_seqs, kv, group, 1]`
  without the unit axis, and the one-chunk state drops their leading unit slot axis too;
* `tl.exp` takes f32, so each `exp` widens its f16 argument and narrows the result (the ladder folds
  the pair back to the f16 op);
* Triton's gather needs at least 8 indices, so the host sizes a chunk to at least 8 entries: below
  that, the chunk takes `ceil(8 / num_seqs)` block slots, the ones past a sequence's blocks masked
  and gathering page 0, which is their own padding law ("padding columns gather page 0 under an
  all -inf mask and contribute zero").
"""

import triton
import triton.language as tl


@triton.jit
def batched_decode_head_major_kernel(query, rep_row_ids, k_pages, v_pages,  #
                                     chunk_page_ids, mask_by_chunk, out,  #
                                     NUM_TOKENS: tl.constexpr, NUM_SEQS: tl.constexpr,  #
                                     BLOCKS_PER_CHUNK: tl.constexpr, NUM_CHUNKS: tl.constexpr,  #
                                     NUM_KV_HEADS: tl.constexpr,
                                     NUM_QUERIES_PER_KV: tl.constexpr,  #
                                     BLOCK_SIZE: tl.constexpr, HEAD_SIZE: tl.constexpr,  #
                                     NUM_PAGES: tl.constexpr, SCALE: tl.constexpr):
    B: tl.constexpr = NUM_SEQS
    J: tl.constexpr = BLOCKS_PER_CHUNK
    KV: tl.constexpr = NUM_KV_HEADS
    G: tl.constexpr = NUM_QUERIES_PER_KV
    D: tl.constexpr = HEAD_SIZE
    NUM_HEADS: tl.constexpr = KV * G
    ENTRIES: tl.constexpr = J * B
    PAGE: tl.constexpr = KV * BLOCK_SIZE * D

    q_rows_desc = tl.make_tensor_descriptor(query, shape=[NUM_TOKENS, NUM_HEADS * D],
                                            strides=[NUM_HEADS * D, 1],
                                            block_shape=[1, NUM_HEADS * D])
    rep_desc = tl.make_tensor_descriptor(rep_row_ids, shape=[ENTRIES], strides=[1],
                                         block_shape=[ENTRIES])
    k_whole = tl.make_tensor_descriptor(k_pages, shape=[NUM_PAGES, PAGE], strides=[PAGE, 1],
                                        block_shape=[1, PAGE])
    v_whole = tl.make_tensor_descriptor(v_pages, shape=[NUM_PAGES, PAGE], strides=[PAGE, 1],
                                        block_shape=[1, PAGE])
    # chunk_page_ids `[num_chunks, entries, 1]`, contiguous: chunk c's ids are the entries at c * E.
    ids_desc = tl.make_tensor_descriptor(chunk_page_ids, shape=[NUM_CHUNKS * ENTRIES], strides=[1],
                                         block_shape=[ENTRIES])
    # mask_by_chunk `[num_chunks, J, B, KV, 1, block]`: chunk c's tile is J * B * KV rows of a block.
    masks = tl.make_tensor_descriptor(mask_by_chunk, shape=[NUM_CHUNKS * ENTRIES * KV, BLOCK_SIZE],
                                      strides=[BLOCK_SIZE, 1],
                                      block_shape=[ENTRIES * KV, BLOCK_SIZE])
    out_rows = tl.make_tensor_descriptor(out, shape=[NUM_TOKENS, NUM_HEADS * D],
                                         strides=[NUM_HEADS * D, 1], block_shape=[B, NUM_HEADS * D])

    # q = query.index_select(0, rep_row_ids).reshape(entries, num_kv_heads, num_queries_per_kv, D)
    rep = rep_desc.load([0])
    q = tl.reshape(q_rows_desc.gather(rep, 0), [ENTRIES * KV, G, D])

    if NUM_CHUNKS > 1:
        # use_entry_local: state_shape (blocks_per_chunk, num_seqs, num_kv_heads, G, 1)
        tile_max = tl.zeros([J, B, KV, G], dtype=tl.float16) - float("inf")
        tile_sum = tl.zeros([J, B, KV, G], dtype=tl.float16)
        tile_output = tl.zeros([J, B, KV, G, D], dtype=tl.float16)
        for c in tl.range(0, NUM_CHUNKS, 1):
            # page_ids = page_ids[0]; k_page = k_pages[page_ids].reshape(entries, KV, block, D)
            page_ids = ids_desc.load([c * ENTRIES])
            k_page = tl.reshape(k_whole.gather(page_ids, 0), [ENTRIES * KV, BLOCK_SIZE, D])
            v_page = tl.reshape(v_whole.gather(page_ids, 0), [ENTRIES * KV, BLOCK_SIZE, D])
            # scores = torch.matmul(q, k_page.transpose(-2, -1)) * scale
            scores = tl.dot(q, tl.trans(k_page, (0, 2, 1)), out_dtype=tl.float16) * SCALE
            # sc = scores.reshape(J, B, KV, G, block) + mask_rows
            mask_rows = tl.reshape(masks.load([c * ENTRIES * KV, 0]), [J, B, KV, BLOCK_SIZE])
            sc = tl.reshape(scores, [J, B, KV, G, BLOCK_SIZE]) + mask_rows[:, :, :, None, :]
            # _entry_local_update: sc = torch.clamp(sc, min=torch.finfo(sc.dtype).min)
            sc = tl.maximum(sc, -65504.0)
            pos_max = tl.max(sc, 4)
            rescale = tl.exp((-tl.maximum(pos_max - tile_max, 0.0)).to(tl.float32)).to(tl.float16)
            new_max = tl.maximum(tile_max, pos_max)
            probs = tl.exp((sc - new_max[:, :, :, :, None]).to(tl.float32)).to(tl.float16)
            tile_sum = tile_sum * rescale + tl.sum(probs, 4).to(tl.float16)
            # pv(probs): torch.matmul(probs.reshape(entries, KV, G, block), v_page)
            pv = tl.dot(tl.reshape(probs, [ENTRIES * KV, G, BLOCK_SIZE]), v_page,
                        out_dtype=tl.float16)
            tile_output = tile_output * rescale[:, :, :, :, None] + tl.reshape(pv, [J, B, KV, G, D])
            tile_max = new_max
        # _merge_entry_local
        merged_max = tl.max(tile_max, 0)
        weight = tl.exp((tile_max - merged_max[None, :, :, :]).to(tl.float32)).to(tl.float16)
        merged_sum = tl.sum(weight * tile_sum, 0).to(tl.float16)
        merged_out = tl.sum(weight[:, :, :, :, None] * tile_output, 0).to(tl.float16)
        attn = tl.reshape(merged_out / merged_sum[:, :, :, None], [B, NUM_HEADS * D])
    else:
        # state_shape (1, num_seqs, num_kv_heads, G, 1)
        tile_max = tl.zeros([B, KV, G], dtype=tl.float16) - float("inf")
        tile_sum = tl.zeros([B, KV, G], dtype=tl.float16)
        tile_output = tl.zeros([B, KV, G, D], dtype=tl.float16)
        for c in tl.range(0, NUM_CHUNKS, 1):
            page_ids = ids_desc.load([c * ENTRIES])
            k_page = tl.reshape(k_whole.gather(page_ids, 0), [ENTRIES * KV, BLOCK_SIZE, D])
            v_page = tl.reshape(v_whole.gather(page_ids, 0), [ENTRIES * KV, BLOCK_SIZE, D])
            scores = tl.dot(q, tl.trans(k_page, (0, 2, 1)), out_dtype=tl.float16) * SCALE
            mask_rows = tl.reshape(masks.load([c * ENTRIES * KV, 0]), [J, B, KV, BLOCK_SIZE])
            sc = tl.reshape(scores, [J, B, KV, G, BLOCK_SIZE]) + mask_rows[:, :, :, None, :]
            # chunk_max = torch.amax(torch.amax(sc, dim=-1, keepdim=True), dim=0, keepdim=True)
            chunk_max = tl.max(tl.max(sc, 4), 0)
            # rescale = torch.exp(-torch.relu(chunk_max - tile_max))
            rescale = tl.exp((-tl.maximum(chunk_max - tile_max, 0.0)).to(tl.float32)).to(tl.float16)
            new_max = tl.maximum(tile_max, chunk_max)
            # reduce_chunk(torch.exp(sc - new_max), v_page)
            probs = tl.exp((sc - new_max[None, :, :, :, None]).to(tl.float32)).to(tl.float16)
            chunk_sum = tl.sum(tl.sum(probs, 4), 0).to(tl.float16)
            pv = tl.dot(tl.reshape(probs, [ENTRIES * KV, G, BLOCK_SIZE]), v_page,
                        out_dtype=tl.float16)
            chunk_out = tl.sum(tl.reshape(pv, [J, B, KV, G, D]), 0).to(tl.float16)
            tile_sum = tile_sum * rescale + chunk_sum
            tile_output = tile_output * rescale[:, :, :, None] + chunk_out
            tile_max = new_max
        # attn = (tile_output / tile_sum).reshape(num_seqs, num_heads, head_size)
        attn = tl.reshape(tile_output / tile_sum[:, :, :, None], [B, NUM_HEADS * D])
    # out[:num_seqs].copy_(attn)
    out_rows.store([0, 0], attn)

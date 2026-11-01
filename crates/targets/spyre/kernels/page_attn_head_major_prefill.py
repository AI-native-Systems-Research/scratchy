# SPDX-License-Identifier: Apache-2.0
"""spyre-inference's `page_attn_head_major_prefill_kernel`
(`spyre_inference/v1/attention/ops/page_attn_head_major_prefill.py`, spyre-inference 297fc48), in
Triton: paged attention over a head-major KV cache for a query wider than one token — batched GQA
over `[kv_head, group, query, D]`, one accumulator, the store fused.

Their kernel, statement for statement; each block below names the line it is. The operands are
theirs, in their order:

* `query` `[num_tokens, num_heads * head_size]`, the whole batch's query rows;
* `query_row_index` `[padded_query_len]` int32, this sequence's rows of it;
* `k_pages` / `v_pages` the UNFOLDED cache `[num_pages_total, num_kv_heads, block_size, head_size]`;
* `page_index_table` `[num_blocks, INT32_ELEMS_PER_STICK]` int32, row i holding the i-th active
  block's page id at column 0;
* `mask_stack` `[num_blocks, padded_query_len, block_size]`, tiled on dim 0;
* `out`, which the kernel stores into through `query_row_index` (`out.index_copy_`).

The walk is `walk_tiles` at `SPYRE_ATTN_FOR_EACH_TILE=1` (their default): ONE block body over
`num_blocks` trips, the carry seeded `(-inf, 0, 0)`.

Where Triton spells a torch op differently, the spelling is Triton's and the op is theirs:

* `tl.dot` is at most rank 3, so `torch.matmul(q, k_page.transpose(-2, -1))` over
  `[kv, group, query, D] x [kv, 1, D, block]` is the rank-3 dot over `[kv, group * query, D]` —
  the same per-(kv, group) products, `q`'s `[group, query]` rows contiguous either way — and its
  result is viewed back at `[kv, group, query, block]`;
* `tl.exp` takes f32, so each `exp` widens its f16 argument and narrows the result (the ladder folds
  the pair back to the f16 op);
* a row index of fewer than 8 entries is below Triton's gather and scatter floor, so under 8 query
  rows the rows are read and written as the tile at the first row `query_row_index` names, which
  is the rows it names when they are consecutive — as a one-sequence forward's always are.
"""

import triton
import triton.language as tl


@triton.jit
def page_attn_head_major_prefill_kernel(query, query_row_index, k_pages, v_pages,  #
                                        page_index_table, mask_stack, out,  #
                                        NUM_TOKENS: tl.constexpr, NUM_HEADS: tl.constexpr,  #
                                        NUM_KV_HEADS: tl.constexpr, HEAD_SIZE: tl.constexpr,  #
                                        BLOCK_SIZE: tl.constexpr, NUM_PAGES: tl.constexpr,  #
                                        NUM_BLOCKS: tl.constexpr,
                                        PADDED_QUERY_LEN: tl.constexpr,  #
                                        SCALE: tl.constexpr):
    NUM_QUERIES_PER_KV: tl.constexpr = NUM_HEADS // NUM_KV_HEADS
    INT32_ELEMS_PER_STICK: tl.constexpr = 32
    P: tl.constexpr = PADDED_QUERY_LEN
    G: tl.constexpr = NUM_QUERIES_PER_KV
    if P >= 8:
        q_rows_desc = tl.make_tensor_descriptor(query, shape=[NUM_TOKENS, NUM_HEADS * HEAD_SIZE],
                                                strides=[NUM_HEADS * HEAD_SIZE, 1],
                                                block_shape=[1, NUM_HEADS * HEAD_SIZE])
        out_rows = tl.make_tensor_descriptor(out, shape=[NUM_TOKENS, NUM_HEADS * HEAD_SIZE],
                                             strides=[NUM_HEADS * HEAD_SIZE, 1],
                                             block_shape=[1, NUM_HEADS * HEAD_SIZE])
        rows_desc = tl.make_tensor_descriptor(query_row_index, shape=[P], strides=[1],
                                              block_shape=[P])
        rows = rows_desc.load([0])
        # q_rows = query.index_select(0, query_row_index)
        q_rows = q_rows_desc.gather(rows, 0)
    else:
        q_tile = tl.make_tensor_descriptor(query, shape=[NUM_TOKENS, NUM_HEADS * HEAD_SIZE],
                                           strides=[NUM_HEADS * HEAD_SIZE, 1],
                                           block_shape=[P, NUM_HEADS * HEAD_SIZE])
        out_tile = tl.make_tensor_descriptor(out, shape=[NUM_TOKENS, NUM_HEADS * HEAD_SIZE],
                                             strides=[NUM_HEADS * HEAD_SIZE, 1],
                                             block_shape=[P, NUM_HEADS * HEAD_SIZE])
        first = tl.load(query_row_index)
        q_rows = q_tile.load([first, 0])
    # q = q_rows.unsqueeze(0).transpose(1, 2).reshape(num_kv_heads, num_queries_per_kv,
    #                                                 padded_query_len, head_size)
    q = tl.reshape(tl.trans(tl.reshape(q_rows, [P, NUM_HEADS, HEAD_SIZE]), (1, 0, 2)),
                   [NUM_KV_HEADS, G * P, HEAD_SIZE])

    k_unfolded = tl.make_tensor_descriptor(k_pages, shape=[NUM_PAGES,
                                                           NUM_KV_HEADS * BLOCK_SIZE * HEAD_SIZE],
                                           strides=[NUM_KV_HEADS * BLOCK_SIZE * HEAD_SIZE, 1],
                                           block_shape=[1, NUM_KV_HEADS * BLOCK_SIZE * HEAD_SIZE])
    v_unfolded = tl.make_tensor_descriptor(v_pages, shape=[NUM_PAGES,
                                                           NUM_KV_HEADS * BLOCK_SIZE * HEAD_SIZE],
                                           strides=[NUM_KV_HEADS * BLOCK_SIZE * HEAD_SIZE, 1],
                                           block_shape=[1, NUM_KV_HEADS * BLOCK_SIZE * HEAD_SIZE])
    masks = tl.make_tensor_descriptor(mask_stack, shape=[NUM_BLOCKS * P, BLOCK_SIZE],
                                      strides=[BLOCK_SIZE, 1], block_shape=[P, BLOCK_SIZE])

    # init=(full(-inf), zeros, zeros), state_shape (num_kv_heads, num_queries_per_kv,
    # padded_query_len, 1)
    tile_max = tl.zeros([NUM_KV_HEADS, G, P], dtype=tl.float16) - float("inf")
    tile_sum = tl.zeros([NUM_KV_HEADS, G, P], dtype=tl.float16)
    tile_output = tl.zeros([NUM_KV_HEADS, G, P, HEAD_SIZE], dtype=tl.float16)
    for i in tl.range(0, NUM_BLOCKS, 1):
        # page_index, mask_tile = page_index_table[i:i+1], mask_stack[i:i+1]
        # page_idx = page_index[0, 0:1]
        page = tl.load(page_index_table + i * INT32_ELEMS_PER_STICK)
        # k_page = k_pages.index_select(0, page_idx).squeeze(0).unsqueeze(1)
        k_page = tl.reshape(k_unfolded.load([page, 0]), [NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE])
        v_page = tl.reshape(v_unfolded.load([page, 0]), [NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE])
        # scores = torch.matmul(q, k_page.transpose(-2, -1)) * scale
        scores = tl.reshape(tl.dot(q, tl.trans(k_page, (0, 2, 1)), out_dtype=tl.float16) * SCALE,
                            [NUM_KV_HEADS, G, P, BLOCK_SIZE])
        # scores = scores + mask_tile[0]
        scores = scores + masks.load([i * P, 0])[None, None, :, :]
        # scores_max = torch.amax(scores, dim=-1, keepdim=True)
        scores_max = tl.max(scores, 3)
        # rescale = torch.exp(-torch.relu(scores_max - tile_max))
        rescale = tl.exp((-tl.maximum(scores_max - tile_max, 0.0)).to(tl.float32)).to(tl.float16)
        # new_max = torch.maximum(tile_max, scores_max)
        new_max = tl.maximum(tile_max, scores_max)
        # tile_probs = torch.exp(scores - new_max)
        tile_probs = tl.exp((scores - new_max[:, :, :, None]).to(tl.float32)).to(tl.float16)
        # new_sum = tile_sum * rescale + tile_probs.sum(dim=-1, keepdim=True)
        tile_sum = tile_sum * rescale + tl.sum(tile_probs, 3).to(tl.float16)
        # new_output = tile_output * rescale + torch.matmul(tile_probs, v_page)
        pv = tl.dot(tl.reshape(tile_probs, [NUM_KV_HEADS, G * P, BLOCK_SIZE]), v_page,
                    out_dtype=tl.float16)
        tile_output = tile_output * rescale[:, :, :, None] + tl.reshape(
            pv, [NUM_KV_HEADS, G, P, HEAD_SIZE])
        tile_max = new_max
    # attn = tile_output / tile_sum
    attn = tile_output / tile_sum[:, :, :, None]
    # attn = attn.reshape(1, num_heads, padded_query_len, head_size).transpose(1, 2)
    # attn = attn.reshape(padded_query_len, num_heads, head_size)
    attn = tl.reshape(tl.trans(tl.reshape(attn, [NUM_HEADS, P, HEAD_SIZE]), (1, 0, 2)),
                      [P, NUM_HEADS * HEAD_SIZE])
    # out.index_copy_(0, query_row_index, attn)
    if P >= 8:
        out_rows.scatter(attn, rows, 0)
    else:
        out_tile.store([first, 0], attn)

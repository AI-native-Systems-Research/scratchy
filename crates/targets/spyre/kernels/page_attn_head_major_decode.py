# SPDX-License-Identifier: Apache-2.0
"""spyre-inference's `page_attn_head_major_decode_kernel`
(`spyre_inference/v1/attention/ops/page_attn_head_major_decode.py`, spyre-inference 297fc48), in
Triton: per-sequence decode attention over a head-major KV cache, the page LX-resident.

Their kernel, statement for statement; each block below names the line it is. The operands are
theirs, in their order:

* `query` `[num_tokens, num_heads * head_size]`, the whole batch's query rows;
* `query_row_index` `[padded_query_len]` int32, this sequence's rows of it (here one);
* `k_pages` / `v_pages` the FOLDED cache `[num_pages_total * num_kv_heads, block_size, head_size]`;
* `page_index_table` `[num_blocks, INT32_ELEMS_PER_STICK]` int32, row i holding the i-th active
  block's page id at column 0;
* `kv_row_pool` `[num_pages_total, num_kv_heads, 1]` int32, every page's folded-cache rows
  `page * num_kv_heads + kv`;
* `mask_stack` `[num_blocks, padded_query_len, block_size]`, tiled on dim 0;
* `out`, which the kernel stores into through `query_row_index` (`out.index_copy_`).

The walk is `walk_tiles` at `SPYRE_ATTN_FOR_EACH_TILE=1` (their default): ONE block body over
`num_blocks` trips, the carry seeded `(-inf, 0, 0)`.

Where Triton spells a torch op differently, the spelling is Triton's and the op is theirs:
`tl.exp` takes f32, so each `exp` widens its f16 argument and narrows the result (the ladder folds
the pair back to the f16 op); `kv_row_pool.index_select(0, page)` is the `[num_kv_heads]` run of
the contiguous pool at `page * num_kv_heads`, its `[1, num_kv_heads, 1]` slab; and a subscript of
fewer than 8 rows is below Triton's gather floor, so under 8 kv heads the rows `kv_row_pool[page]`
names — consecutive by its construction — are read as the tile at the first of them.
"""

import triton
import triton.language as tl


@triton.jit
def page_attn_head_major_decode_kernel(query, query_row_index, k_pages, v_pages,  #
                                       page_index_table, kv_row_pool, mask_stack, out,  #
                                       NUM_TOKENS: tl.constexpr, NUM_HEADS: tl.constexpr,  #
                                       NUM_KV_HEADS: tl.constexpr, HEAD_SIZE: tl.constexpr,  #
                                       BLOCK_SIZE: tl.constexpr, NUM_PAGES: tl.constexpr,  #
                                       NUM_BLOCKS: tl.constexpr, SCALE: tl.constexpr):
    NUM_QUERIES_PER_KV: tl.constexpr = NUM_HEADS // NUM_KV_HEADS
    INT32_ELEMS_PER_STICK: tl.constexpr = 32
    q_rows = tl.make_tensor_descriptor(query, shape=[NUM_TOKENS, NUM_HEADS * HEAD_SIZE],
                                       strides=[NUM_HEADS * HEAD_SIZE, 1],
                                       block_shape=[1, NUM_HEADS * HEAD_SIZE])
    out_rows = tl.make_tensor_descriptor(out, shape=[NUM_TOKENS, NUM_HEADS * HEAD_SIZE],
                                         strides=[NUM_HEADS * HEAD_SIZE, 1],
                                         block_shape=[1, NUM_HEADS * HEAD_SIZE])
    k_folded = tl.make_tensor_descriptor(k_pages, shape=[NUM_PAGES * NUM_KV_HEADS,
                                                         BLOCK_SIZE * HEAD_SIZE],
                                         strides=[BLOCK_SIZE * HEAD_SIZE, 1],
                                         block_shape=[1, BLOCK_SIZE * HEAD_SIZE])
    v_folded = tl.make_tensor_descriptor(v_pages, shape=[NUM_PAGES * NUM_KV_HEADS,
                                                         BLOCK_SIZE * HEAD_SIZE],
                                         strides=[BLOCK_SIZE * HEAD_SIZE, 1],
                                         block_shape=[1, BLOCK_SIZE * HEAD_SIZE])
    # Under the gather floor: num_kv_heads consecutive rows of the folded cache.
    k_tiles = tl.make_tensor_descriptor(k_pages, shape=[NUM_PAGES * NUM_KV_HEADS,
                                                        BLOCK_SIZE * HEAD_SIZE],
                                        strides=[BLOCK_SIZE * HEAD_SIZE, 1],
                                        block_shape=[NUM_KV_HEADS, BLOCK_SIZE * HEAD_SIZE])
    v_tiles = tl.make_tensor_descriptor(v_pages, shape=[NUM_PAGES * NUM_KV_HEADS,
                                                        BLOCK_SIZE * HEAD_SIZE],
                                        strides=[BLOCK_SIZE * HEAD_SIZE, 1],
                                        block_shape=[NUM_KV_HEADS, BLOCK_SIZE * HEAD_SIZE])
    # kv_row_pool `[num_pages, num_kv_heads, 1]`, contiguous: page p's slab is the num_kv_heads
    # elements at p * num_kv_heads.
    pool = tl.make_tensor_descriptor(kv_row_pool, shape=[NUM_PAGES * NUM_KV_HEADS], strides=[1],
                                     block_shape=[NUM_KV_HEADS])
    masks = tl.make_tensor_descriptor(mask_stack, shape=[NUM_BLOCKS, BLOCK_SIZE],
                                      strides=[BLOCK_SIZE, 1], block_shape=[1, BLOCK_SIZE])

    # q = query.index_select(0, query_row_index).reshape(num_kv_heads, num_queries_per_kv, head_size)
    row = tl.load(query_row_index)
    q = tl.reshape(q_rows.load([row, 0]), [NUM_KV_HEADS, NUM_QUERIES_PER_KV, HEAD_SIZE])

    # init=(full(-inf), zeros, zeros), state_shape (num_kv_heads, num_queries_per_kv, 1)
    tile_max = tl.zeros([NUM_KV_HEADS, NUM_QUERIES_PER_KV], dtype=tl.float16) - float("inf")
    tile_sum = tl.zeros([NUM_KV_HEADS, NUM_QUERIES_PER_KV], dtype=tl.float16)
    tile_output = tl.zeros([NUM_KV_HEADS, NUM_QUERIES_PER_KV, HEAD_SIZE], dtype=tl.float16)
    for i in tl.range(0, NUM_BLOCKS, 1):
        # page_index, mask_tile = page_index_table[i:i+1], mask_stack[i:i+1]
        # kv_rows = kv_row_pool.index_select(0, page_index[0, 0:1])
        page = tl.load(page_index_table + i * INT32_ELEMS_PER_STICK)
        # k_page = k_pages[kv_rows].reshape(num_kv_heads, block_size, head_size)
        kv_rows = pool.load([page * NUM_KV_HEADS])
        if NUM_KV_HEADS >= 8:
            k_page = tl.reshape(k_folded.gather(kv_rows, 0), [NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE])
            v_page = tl.reshape(v_folded.gather(kv_rows, 0), [NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE])
        else:
            # Under the gather floor. The rows `kv_row_pool[page]` names ascend by one
            # (`page * num_kv_heads + kv`), so they are the num_kv_heads-row tile at the first.
            first = tl.min(kv_rows, 0)
            k_page = tl.reshape(k_tiles.load([first, 0]), [NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE])
            v_page = tl.reshape(v_tiles.load([first, 0]), [NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE])
        # scores = torch.matmul(q, k_page.permute(0, 2, 1)) * scale
        scores = tl.dot(q, tl.trans(k_page, (0, 2, 1)), out_dtype=tl.float16) * SCALE
        # scores = scores + mask_tile[0]
        scores = scores + masks.load([i, 0])[None, :, :]
        # scores_max = torch.amax(scores, dim=-1, keepdim=True)
        scores_max = tl.max(scores, 2)
        # rescale = torch.exp(-torch.relu(scores_max - tile_max))
        rescale = tl.exp((-tl.maximum(scores_max - tile_max, 0.0)).to(tl.float32)).to(tl.float16)
        # new_max = torch.maximum(tile_max, scores_max)
        new_max = tl.maximum(tile_max, scores_max)
        # probs = torch.exp(scores - new_max)
        probs = tl.exp((scores - new_max[:, :, None]).to(tl.float32)).to(tl.float16)
        # tile_sum * rescale + probs.sum(dim=-1, keepdim=True)
        tile_sum = tile_sum * rescale + tl.sum(probs, 2).to(tl.float16)
        # tile_output * rescale + torch.matmul(probs, v_page)
        tile_output = tile_output * rescale[:, :, None] + tl.dot(probs, v_page,
                                                                  out_dtype=tl.float16)
        tile_max = new_max
    # attn = (tile_output / tile_sum).reshape(1, num_heads, head_size)
    attn = tl.reshape(tile_output / tile_sum[:, :, None], [1, NUM_HEADS * HEAD_SIZE])
    # out.index_copy_(0, query_row_index, attn)
    out_rows.store([row, 0], attn)

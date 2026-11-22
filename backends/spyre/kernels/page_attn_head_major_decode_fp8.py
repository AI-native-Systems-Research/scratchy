# SPDX-License-Identifier: Apache-2.0
"""The fp8 KV pool's twin of `page_attn_head_major_decode.py`'s `page_attn_head_major_decode_kernel`.

An fp8 page is TOKEN-major, `[block_size, num_kv_heads, head_size]`: one token's keys are
`num_kv_heads * head_size` bytes, whole sticks at every head size, so the write lands whole
sticks. Every page is widened to f16 and read head-major through a transpose.
"""

import triton
import triton.language as tl


@triton.jit
def page_attn_head_major_decode_fp8_kernel(query, query_row_index, k_pages, v_pages,  #
                                       page_index_table, mask_stack, out,  #
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
    # One row per page: the page's num_kv_heads · block_size · head_size elements, contiguous.
    k_unfolded = tl.make_tensor_descriptor(k_pages, shape=[NUM_PAGES,
                                                           NUM_KV_HEADS * BLOCK_SIZE * HEAD_SIZE],
                                           strides=[NUM_KV_HEADS * BLOCK_SIZE * HEAD_SIZE, 1],
                                           block_shape=[1, NUM_KV_HEADS * BLOCK_SIZE * HEAD_SIZE])
    v_unfolded = tl.make_tensor_descriptor(v_pages, shape=[NUM_PAGES,
                                                           NUM_KV_HEADS * BLOCK_SIZE * HEAD_SIZE],
                                           strides=[NUM_KV_HEADS * BLOCK_SIZE * HEAD_SIZE, 1],
                                           block_shape=[1, NUM_KV_HEADS * BLOCK_SIZE * HEAD_SIZE])
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
        # k_page = k_pages[page].reshape(num_kv_heads, block_size, head_size): the page's one
        # row of the unfolded pool, read through the page index.
        k_page = tl.trans(tl.reshape(k_unfolded.load([page, 0]).to(tl.float16),
                                     [BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE]), (1, 0, 2))
        v_page = tl.trans(tl.reshape(v_unfolded.load([page, 0]).to(tl.float16),
                                     [BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE]), (1, 0, 2))
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

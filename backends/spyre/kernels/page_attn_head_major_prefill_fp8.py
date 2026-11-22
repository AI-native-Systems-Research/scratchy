# SPDX-License-Identifier: Apache-2.0
"""The fp8 KV pool's twin of `page_attn_head_major_prefill.py`'s `page_attn_head_major_prefill_kernel`.

An fp8 page is TOKEN-major, `[block_size, num_kv_heads, head_size]`: one token's keys are
`num_kv_heads * head_size` bytes, whole sticks at every head size, so the write lands whole
sticks. Every page is widened to f16 and read head-major through a transpose.
"""

import triton
import triton.language as tl


@triton.jit
def page_attn_head_major_prefill_fp8_kernel(query, query_row_index, k_pages, v_pages,  #
                                        page_index_table, mask_stack, out,  #
                                        NUM_TOKENS: tl.constexpr, NUM_HEADS: tl.constexpr,  #
                                        NUM_KV_HEADS: tl.constexpr, HEAD_SIZE: tl.constexpr,  #
                                        BLOCK_SIZE: tl.constexpr, NUM_PAGES: tl.constexpr,  #
                                        NUM_BLOCKS: tl.constexpr, BLOCKS_PER_TRIP: tl.constexpr,
                                        PADDED_QUERY_LEN: tl.constexpr,  #
                                        SCALE: tl.constexpr):
    NUM_QUERIES_PER_KV: tl.constexpr = NUM_HEADS // NUM_KV_HEADS
    INT32_ELEMS_PER_STICK: tl.constexpr = 32
    P: tl.constexpr = PADDED_QUERY_LEN
    G: tl.constexpr = NUM_QUERIES_PER_KV
    J: tl.constexpr = BLOCKS_PER_TRIP
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
    if J == 1:
        for i in tl.range(0, NUM_BLOCKS, 1):
            # page_index, mask_tile = page_index_table[i:i+1], mask_stack[i:i+1]
            # page_idx = page_index[0, 0:1]
            page = tl.load(page_index_table + i * INT32_ELEMS_PER_STICK)
            # k_page = k_pages.index_select(0, page_idx).squeeze(0).unsqueeze(1)
            k_page = tl.trans(tl.reshape(k_unfolded.load([page, 0]).to(tl.float16),
                                         [BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE]), (1, 0, 2))
            v_page = tl.trans(tl.reshape(v_unfolded.load([page, 0]).to(tl.float16),
                                         [BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE]), (1, 0, 2))
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
    else:
        # J PAGES A TRIP — their walk at a block J pages wide: the same running softmax, over
        # J * block_size keys at once. The trip's pages gather as J whole rows of the unfolded
        # cache, each kv head's keys then laid side by side across the J pages.
        # The page index table holds a walk of several pages a trip as `[num_blocks]`
        # contiguous int32, a trip's J pages the gather's index.
        trip_ids = tl.make_tensor_descriptor(page_index_table, shape=[NUM_BLOCKS], strides=[1],
                                             block_shape=[J])
        trip_masks = tl.make_tensor_descriptor(mask_stack, shape=[NUM_BLOCKS * P, BLOCK_SIZE],
                                               strides=[BLOCK_SIZE, 1],
                                               block_shape=[J * P, BLOCK_SIZE])
        for c in tl.range(0, NUM_BLOCKS // J, 1):
            pages = trip_ids.load([c * J])
            k_trip = tl.reshape(tl.trans(tl.reshape(k_unfolded.gather(pages, 0).to(tl.float16),
                                                    [J, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE]),
                                         (2, 0, 1, 3)),
                                [NUM_KV_HEADS, J * BLOCK_SIZE, HEAD_SIZE])
            v_trip = tl.reshape(tl.trans(tl.reshape(v_unfolded.gather(pages, 0).to(tl.float16),
                                                    [J, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE]),
                                         (2, 0, 1, 3)),
                                [NUM_KV_HEADS, J * BLOCK_SIZE, HEAD_SIZE])
            scores = tl.reshape(tl.dot(q, tl.trans(k_trip, (0, 2, 1)), out_dtype=tl.float16) * SCALE,
                                [NUM_KV_HEADS, G, P, J * BLOCK_SIZE])
            # the trip's J mask tiles, side by side per query row
            mask = tl.reshape(tl.trans(tl.reshape(trip_masks.load([c * J * P, 0]),
                                                  [J, P, BLOCK_SIZE]), (1, 0, 2)),
                              [P, J * BLOCK_SIZE])
            scores = scores + mask[None, None, :, :]
            scores_max = tl.max(scores, 3)
            rescale = tl.exp((-tl.maximum(scores_max - tile_max, 0.0)).to(tl.float32)).to(tl.float16)
            new_max = tl.maximum(tile_max, scores_max)
            tile_probs = tl.exp((scores - new_max[:, :, :, None]).to(tl.float32)).to(tl.float16)
            tile_sum = tile_sum * rescale + tl.sum(tile_probs, 3).to(tl.float16)
            pv = tl.dot(tl.reshape(tile_probs, [NUM_KV_HEADS, G * P, J * BLOCK_SIZE]), v_trip,
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

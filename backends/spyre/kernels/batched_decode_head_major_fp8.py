# SPDX-License-Identifier: Apache-2.0
"""The fp8 KV pool's twin of `batched_decode_head_major.py`'s `batched_decode_head_major_kernel`.

An fp8 page is TOKEN-major, `[block_size, num_kv_heads, head_size]`: one token's keys are
`num_kv_heads * head_size` bytes, whole sticks at every head size, so the write lands whole
sticks. Every gathered page is widened to f16 and read head-major through a transpose.
"""

import triton
import triton.language as tl


@triton.jit
def batched_decode_head_major_fp8_kernel(query, rep_row_ids, k_pages, v_pages,  #
                                         chunk_page_ids, mask_by_chunk, out,  #
                                         NUM_TOKENS: tl.constexpr, NUM_SEQS: tl.constexpr,  #
                                         BLOCKS_PER_CHUNK: tl.constexpr, NUM_CHUNKS: tl.constexpr,  #
                                         NUM_KV_HEADS: tl.constexpr,
                                         NUM_QUERIES_PER_KV: tl.constexpr,  #
                                         BLOCK_SIZE: tl.constexpr, HEAD_SIZE: tl.constexpr,  #
                                         NUM_PAGES: tl.constexpr, ROWS_PER_PAGE: tl.constexpr,  #
                                         SCALE: tl.constexpr):
    B: tl.constexpr = NUM_SEQS
    J: tl.constexpr = BLOCKS_PER_CHUNK
    KV: tl.constexpr = NUM_KV_HEADS
    G: tl.constexpr = NUM_QUERIES_PER_KV
    D: tl.constexpr = HEAD_SIZE
    NUM_HEADS: tl.constexpr = KV * G
    ENTRIES: tl.constexpr = J * B
    P: tl.constexpr = ROWS_PER_PAGE
    ROWS: tl.constexpr = ENTRIES * P
    PAGE: tl.constexpr = KV * BLOCK_SIZE * D

    # The query `[num_tokens * P, num_heads * D / P]`: token t's row r is row t * P + r.
    q_rows_desc = tl.make_tensor_descriptor(query, shape=[NUM_TOKENS * P, NUM_HEADS * D // P],
                                            strides=[NUM_HEADS * D // P, 1],
                                            block_shape=[1, NUM_HEADS * D // P])
    # rep_row_ids `[entries * P]`: entry e's row r at e * P + r, holding rep * P + r.
    rep_desc = tl.make_tensor_descriptor(rep_row_ids, shape=[ROWS], strides=[1],
                                         block_shape=[ROWS])
    # The cache `[num_pages * P, page / P]`: page p's row r is row p * P + r.
    k_rows = tl.make_tensor_descriptor(k_pages, shape=[NUM_PAGES * P, PAGE // P],
                                       strides=[PAGE // P, 1], block_shape=[1, PAGE // P])
    v_rows = tl.make_tensor_descriptor(v_pages, shape=[NUM_PAGES * P, PAGE // P],
                                       strides=[PAGE // P, 1], block_shape=[1, PAGE // P])
    # chunk_page_ids `[num_chunks, entries * P]`, contiguous: chunk c's rows are the ROWS at c * ROWS.
    ids_desc = tl.make_tensor_descriptor(chunk_page_ids, shape=[NUM_CHUNKS * ROWS], strides=[1],
                                         block_shape=[ROWS])
    # mask_by_chunk `[num_chunks, J, B, 1, 1, block]`: chunk c's tile is J * B rows of a block, one per
    # (block slot, sequence); every kv head reads its row through the broadcast below, as theirs does.
    masks = tl.make_tensor_descriptor(mask_by_chunk, shape=[NUM_CHUNKS * ENTRIES, BLOCK_SIZE],
                                      strides=[BLOCK_SIZE, 1],
                                      block_shape=[ENTRIES, BLOCK_SIZE])
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
            page_ids = ids_desc.load([c * ROWS])
            k_page = tl.reshape(tl.trans(tl.reshape(k_rows.gather(page_ids, 0).to(tl.float16),
                                                    [ENTRIES, BLOCK_SIZE, KV, D]), (0, 2, 1, 3)),
                                [ENTRIES * KV, BLOCK_SIZE, D])
            # scores = torch.matmul(q, k_page.transpose(-2, -1)) * scale
            scores = tl.dot(q, tl.trans(k_page, (0, 2, 1)), out_dtype=tl.float16) * SCALE
            # sc = scores.reshape(J, B, KV, G, block) + mask_rows
            # one mask row per (block slot, sequence), sprayed over its kv heads' query rows
            mask_rows = tl.reshape(masks.load([c * ENTRIES, 0]), [J, B, BLOCK_SIZE])
            sc = tl.reshape(tl.reshape(scores, [J, B, KV * G, BLOCK_SIZE]) + mask_rows[:, :, None, :],
                            [J, B, KV, G, BLOCK_SIZE])
            # _entry_local_update: sc = torch.clamp(sc, min=torch.finfo(sc.dtype).min)
            sc = tl.maximum(sc, -65504.0)
            pos_max = tl.max(sc, 4)
            rescale = tl.exp((-tl.maximum(pos_max - tile_max, 0.0)).to(tl.float32)).to(tl.float16)
            new_max = tl.maximum(tile_max, pos_max)
            probs = tl.exp((sc - new_max[:, :, :, :, None]).to(tl.float32)).to(tl.float16)
            tile_sum = tile_sum * rescale + tl.sum(probs, 4).to(tl.float16)
            # pv(probs): torch.matmul(probs.reshape(entries, KV, G, block), v_page) — V gathered
            # here, once K's widened copy is dead.
            v_page = tl.reshape(tl.trans(tl.reshape(v_rows.gather(page_ids, 0).to(tl.float16),
                                                    [ENTRIES, BLOCK_SIZE, KV, D]), (0, 2, 1, 3)),
                                [ENTRIES * KV, BLOCK_SIZE, D])
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
            page_ids = ids_desc.load([c * ROWS])
            k_page = tl.reshape(tl.trans(tl.reshape(k_rows.gather(page_ids, 0).to(tl.float16),
                                                    [ENTRIES, BLOCK_SIZE, KV, D]), (0, 2, 1, 3)),
                                [ENTRIES * KV, BLOCK_SIZE, D])
            scores = tl.dot(q, tl.trans(k_page, (0, 2, 1)), out_dtype=tl.float16) * SCALE
            # one mask row per (block slot, sequence), sprayed over its kv heads' query rows
            mask_rows = tl.reshape(masks.load([c * ENTRIES, 0]), [J, B, BLOCK_SIZE])
            sc = tl.reshape(tl.reshape(scores, [J, B, KV * G, BLOCK_SIZE]) + mask_rows[:, :, None, :],
                            [J, B, KV, G, BLOCK_SIZE])
            # A masked score plus a negative one overflows f16 to -inf: clamp at the mask's own
            # floor, as the per-entry branch does, so a fully masked row never subtracts -inf
            # from -inf.
            sc = tl.maximum(sc, -65504.0)
            # chunk_max = torch.amax(torch.amax(sc, dim=-1, keepdim=True), dim=0, keepdim=True)
            chunk_max = tl.max(tl.max(sc, 4), 0)
            # rescale = torch.exp(-torch.relu(chunk_max - tile_max))
            rescale = tl.exp((-tl.maximum(chunk_max - tile_max, 0.0)).to(tl.float32)).to(tl.float16)
            new_max = tl.maximum(tile_max, chunk_max)
            # reduce_chunk(torch.exp(sc - new_max), v_page)
            probs = tl.exp((sc - new_max[None, :, :, :, None]).to(tl.float32)).to(tl.float16)
            chunk_sum = tl.sum(tl.sum(probs, 4), 0).to(tl.float16)
            v_page = tl.reshape(tl.trans(tl.reshape(v_rows.gather(page_ids, 0).to(tl.float16),
                                                    [ENTRIES, BLOCK_SIZE, KV, D]), (0, 2, 1, 3)),
                                [ENTRIES * KV, BLOCK_SIZE, D])
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

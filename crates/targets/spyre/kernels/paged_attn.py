# SPDX-License-Identifier: Apache-2.0
"""PAGED ATTENTION — spyre-inference's head-major online-softmax page walk
(`spyre_inference/v1/attention/ops/page_attn_head_major_prefill.py`, spyre-inference 297fc48, the
decoder backend `SPYRE_ATTN_KV_LAYOUT=head_major` selects by default), in Triton, reading its pages
by HARDWARE GATHER.

THE OPERANDS are the backend's own:

* the K and V POOLS, `[NUM_PAGES, NKVH, BLOCK, HD]` each (`get_kv_cache_shape`), written by
  `paged_kv_write.py` before this kernel runs (`do_kv_cache_update` precedes the attention);
* the PAGE TABLE, `[NUM_BLOCKS, PT_PITCH]` int32 — `page_index_tables_cpu`: row `i` names the `i`-th
  active block's physical page (below for how). spyre-inference's rows are one int32 stick wide; these
  are one stick per score row (`PT_PITCH`, below);
* the MASK STACK, `[NUM_BLOCKS, MQ, BLOCK]` additive — `attention_mask_stacks`: 0 where a query row may
  attend, `finfo(f16).min` past the sequence (and above the causal diagonal for a prompt chunk),
  staged at the score tile's rows (below);
* the query rows `[MQ, NQH * HD]` and the output rows, the node's own.

ONE WALK, EVERY KV HEAD IN IT — the prefill kernel's shape: the walk visits each page once
(`walk_tiles` over `page_index_table[:num_blocks]` and `mask_stack[:num_blocks]`, tiled on dim 0), and
its block body attends every kv head at once, batched over `[kv_head, group, query, D]`:

* THE QUERY is `query.unsqueeze(0).transpose(1, 2).reshape(NKVH, GQA, MQ, HD)` — every head's `MQ`
  rows, kv-head-major. Here it is ONE descriptor read of the query rows viewed `[NQH, MQ, HD]` (head
  `n`'s row `m` is query row `m`'s columns `n * HD ..`), so the transpose is the view's strides, and
  the `[NKVH, GQA * MQ, HD]` the dots take is a row-major reshape of it: score row `g * MQ + m` of kv
  head `h` is query row `m` of head `h * GQA + g`. At `MQ == 1` it is the decode kernel's fold of the
  query group into the row axis.
* THE PAGE IS GATHERED BY THE DEVICE, ONE KV HEAD'S SLAB PER INDEX ENTRY — the decode kernel's own
  gather "on (page, kv_head) so the gather's split lands per kv head": the pools read as the folded
  cache `[NUM_PAGES * NKVH, BLOCK * HD]`, and block `i`'s `NKVH` rows `kv_row_pool[page]` =
  `page * NKVH + kv` gathered through the page table's row, which holds them
  (`kp_rows.gather(kv_rows, 0)`). An entry is one `[BLOCK, HD]` slab, so the card spreads a page over
  its kv heads' cores — a whole page as one entry is more than one core's scratchpad holds. Under
  Triton's gather floor of 8 rows (`NKVH < 8`, a page a quarter the size or less) the row holds the
  page id itself and the whole page is the one entry: `k_pages.index_select(0, page_idx)`,
  `kp_desc.load([page, 0, 0, 0])` with `page` read from memory. Either way the load lowers to a
  `ktdp.construct_indirect_access_tile` indexed through the page table's view — the gather's address
  is the table entry, never arithmetic in the kernel.
* EVERY HEAD OF A GROUP READS ITS QUERY ROW'S MASK (spyre-inference's `scores + mask_tile[0]`,
  broadcast over the head axes). The host stages the group's broadcast — the mask stack holds each
  block's tile at the score rows, row `g * MQ + m` repeating query row `m`'s — and the kernel
  broadcasts it over the kv heads.

THE RECURRENCE is the kernels' `block_body`, verbatim, at `walk_tiles`' seed `(-inf, 0, 0)` (the
default `SPYRE_ATTN_FOR_EACH_TILE=1` path):

    scores     = q @ k_page^T * scale + mask_tile
    scores_max = amax(scores, -1)
    rescale    = exp(min(tile_max - scores_max, 0))    # == exp(tile_max - new_max)
    new_max    = maximum(tile_max, scores_max)
    probs      = exp(scores - new_max)
    tile_sum   = tile_sum * rescale + sum(probs, -1)
    tile_out   = tile_out * rescale + probs @ v_page
    out        = tile_out / tile_sum

in the model dtype (f16) throughout, as the device runs it. The rescale is spelled as torch-spyre's own
SDPA decomposition spells its correction (`exp(min(m_old - m_blk, 0))`); spyre-inference writes
`exp(-relu(scores_max - tile_max))`, the same value bit for bit (IEEE negation and subtraction are
sign-symmetric, so `-max(x, 0) == min(-x, 0)` exactly). Both read `tile_max` before the maximum that
supersedes it.

THE WALK IS A LOOP — `walk_tiles`' one block body, `tl.range`, not a copy per block: the trip's
page-table row and mask tile are the only things that move with it. The card runs the body once per
trip, stepping the step's sources by ONE shift per trip, so the page table's row pitch is what one
trip's mask tile moves on the card — `ROWS` stick rows of the stick-major stack, one int32 stick per
score row (`PT_PITCH = ROWS * 32`).
"""

import triton
import triton.language as tl


@triton.jit
def paged_attn_fwd(desc_q, desc_o, desc_kp, desc_vp, desc_pt, desc_mask,  #
                   MQ: tl.constexpr, NQH: tl.constexpr, NKVH: tl.constexpr, HD: tl.constexpr,  #
                   BLOCK: tl.constexpr, NUM_PAGES: tl.constexpr, NUM_BLOCKS: tl.constexpr,  #
                   SCALE: tl.constexpr):
    GQA: tl.constexpr = NQH // NKVH
    ROWS: tl.constexpr = GQA * MQ
    # The page table's row pitch, in int32: one int32 stick per score row — what a trip's mask tile
    # moves on the card.
    PT_PITCH: tl.constexpr = ROWS * 32
    # The pools as the folded cache, one `(page, kv head)` slab per row, gathered NKVH rows a block.
    kp_rows = tl.make_tensor_descriptor(desc_kp, shape=[NUM_PAGES * NKVH, BLOCK * HD],
                                        strides=[BLOCK * HD, 1], block_shape=[1, BLOCK * HD])
    vp_rows = tl.make_tensor_descriptor(desc_vp, shape=[NUM_PAGES * NKVH, BLOCK * HD],
                                        strides=[BLOCK * HD, 1], block_shape=[1, BLOCK * HD])
    pt_rows = tl.make_tensor_descriptor(desc_pt, shape=[NUM_BLOCKS * PT_PITCH], strides=[1],
                                        block_shape=[NKVH])
    # Under the gather's floor: the pools head-major, a whole page per block.
    kp_desc = tl.make_tensor_descriptor(desc_kp, shape=[NUM_PAGES, NKVH, BLOCK, HD],
                                        strides=[NKVH * BLOCK * HD, BLOCK * HD, HD, 1],
                                        block_shape=[1, NKVH, BLOCK, HD])
    vp_desc = tl.make_tensor_descriptor(desc_vp, shape=[NUM_PAGES, NKVH, BLOCK, HD],
                                        strides=[NKVH * BLOCK * HD, BLOCK * HD, HD, 1],
                                        block_shape=[1, NKVH, BLOCK, HD])
    # The mask stack, one `[GQA * MQ, BLOCK]` tile per active block, at the score rows.
    mask_desc = tl.make_tensor_descriptor(desc_mask, shape=[NUM_BLOCKS * ROWS, BLOCK],
                                          strides=[BLOCK, 1], block_shape=[ROWS, BLOCK])
    # The query and output rows viewed head-major, `[NQH, MQ, HD]`: the transpose is the strides.
    q_desc = tl.make_tensor_descriptor(desc_q, shape=[NQH, MQ, HD], strides=[HD, NQH * HD, 1],
                                       block_shape=[NQH, MQ, HD])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[NQH, MQ, HD], strides=[HD, NQH * HD, 1],
                                       block_shape=[NQH, MQ, HD])
    q = tl.reshape(q_desc.load([0, 0, 0]), [NKVH, ROWS, HD])
    tile_max = tl.zeros([NKVH, ROWS], dtype=tl.float16) - float("inf")
    tile_sum = tl.zeros([NKVH, ROWS], dtype=tl.float16)
    tile_out = tl.zeros([NKVH, ROWS, HD], dtype=tl.float16)
    for i in tl.range(0, NUM_BLOCKS, 1):
        if NKVH >= 8:
            # `kv_row_pool[page]`: the block's NKVH folded-cache rows, `page * NKVH + kv`.
            kv_rows = pt_rows.load([i * PT_PITCH])
            k_page = tl.reshape(kp_rows.gather(kv_rows, 0), [NKVH, BLOCK, HD])
            v_page = tl.reshape(vp_rows.gather(kv_rows, 0), [NKVH, BLOCK, HD])
        else:
            # `page_index[0, 0:1]`: the block's page, column 0 of its table row.
            page = tl.load(desc_pt + i * PT_PITCH)
            k_page = tl.reshape(kp_desc.load([page, 0, 0, 0]), [NKVH, BLOCK, HD])
            v_page = tl.reshape(vp_desc.load([page, 0, 0, 0]), [NKVH, BLOCK, HD])
        scores = tl.dot(q, tl.trans(k_page, (0, 2, 1)), out_dtype=tl.float16) * SCALE
        # The block's mask tile, at the score rows, over every kv head.
        scores = scores + mask_desc.load([i * ROWS, 0])[None, :, :]
        scores_max = tl.max(scores, 2)
        rescale = tl.exp(tl.minimum(tile_max - scores_max, 0.0).to(tl.float32)).to(tl.float16)
        new_max = tl.maximum(tile_max, scores_max)
        probs = tl.exp((scores - new_max[:, :, None]).to(tl.float32)).to(tl.float16)
        tile_sum = tile_sum * rescale + tl.sum(probs, 2).to(tl.float16)
        tile_out = tile_out * rescale[:, :, None] + tl.dot(probs, v_page, out_dtype=tl.float16)
        tile_max = new_max
    o = tile_out / tile_sum[:, :, None]
    o_desc.store([0, 0, 0], tl.reshape(o, [NQH, MQ, HD]))

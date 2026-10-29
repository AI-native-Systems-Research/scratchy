# SPDX-License-Identifier: Apache-2.0
"""PAGED ATTENTION — spyre-inference's head-major online-softmax page walk
(`spyre_inference/v1/attention/ops/page_attn_head_major_prefill.py` and
`page_attn_head_major_decode.py`, spyre-inference 297fc48, the decoder backend
`SPYRE_ATTN_KV_LAYOUT=head_major` selects by default), in Triton, reading its pages by HARDWARE
GATHER.

THE OPERANDS are the backend's own:

* the K and V POOLS, `[NUM_PAGES, NKVH, BLOCK, HD]` each (`get_kv_cache_shape`), written by
  `paged_kv_write.py` before this kernel runs (`do_kv_cache_update` precedes the attention);
* the PAGE TABLE, `[NUM_BLOCKS, 32]` int32 — `page_index_tables_cpu`: row `i` holds the `i`-th active
  block's physical page at column 0, each row one int32 stick wide;
* the MASK STACK, `[NUM_BLOCKS, MQ * GQA, BLOCK]` additive — `attention_mask_stacks`: 0 where a query
  row may attend, `finfo(f16).min` past the sequence (and above the causal diagonal for a prompt
  chunk), staged at the score tile's rows (below);
* the query rows `[MQ, NQH * HD]` and the output rows, the node's own.

THE PAGE IS GATHERED BY THE DEVICE. Each block reads its page id from the page table
(`page_index[0, 0:1]`) and selects ONE `(page, kv head)` slab of the pool with it
(`k_pages.index_select(0, page_idx)`): `kp_desc.load([page, h, 0, 0])` with `page` read from memory
lowers to a `ktdp.construct_indirect_access_tile` whose page axis is indexed through the page table's
view — the gather's address is the table entry, never arithmetic in the kernel.

THE RECURRENCE is the kernels' `block_body`, verbatim, at `for_each_tile`'s seed `(-inf, 0, 0)` (the
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

ONE WALK PER KV HEAD, ITS WHOLE QUERY GROUP AT ONCE — both kernels' shape. A KV head's `GQA` query
heads read the same pages, so they ride ONE walk as rows: the `[MQ, GQA * HD]` block of the query
rows that holds the group is `[MQ * GQA, HD]` rows, row `m * GQA + g` being query row `m` of the
group's head `g` — a row-major reshape, nothing moved. Every page is then gathered once per KV head
and serves the whole group:

* DECODE (`MQ == 1`, `page_attn_head_major_decode`): "the query groups fold into the row axis" —
  `[GQA, HD]`;
* PROMPT (`MQ > 1`, `page_attn_head_major_prefill`): the batched GQA over `[kv_head, group, query,
  D]`, one KV head at a time — `[MQ * GQA, HD]`.

Every head of the group reads its query row's mask (spyre-inference's `scores + mask_tile[0]`,
broadcast over the head axes). The host stages that broadcast: the mask stack holds each block's
tile at the SCORE rows, row `m * GQA + g` repeating query row `m`'s, and the kernel adds it as is.
Broadcasting on the device would be a middle-axis broadcast (`[MQ, 1, BLOCK] -> [MQ, GQA, BLOCK]`),
an address that advances every `GQA` rows, which no pointwise operand of the device states.

Every loop is `tl.static_range` — the page walk included — so each block's page-table offset is a
compile-time constant, the one form the gather lowering states (a loaded value is the gather's index;
the table offset it is loaded from is not).
"""

import triton
import triton.language as tl


@triton.jit
def _page_walk(q, kp_desc, vp_desc, mask_desc, desc_pt, h: tl.constexpr,  #
               MQ: tl.constexpr, GQA: tl.constexpr, HD: tl.constexpr,  #
               BLOCK: tl.constexpr, NUM_BLOCKS: tl.constexpr, SCALE: tl.constexpr):
    """KV head `h`'s walk over the active pages for its group's `[MQ * GQA, HD]` query rows `q`."""
    tile_max = tl.zeros([MQ * GQA], dtype=tl.float16) - float("inf")
    tile_sum = tl.zeros([MQ * GQA], dtype=tl.float16)
    tile_out = tl.zeros([MQ * GQA, HD], dtype=tl.float16)
    for i in tl.range(0, NUM_BLOCKS, 1):
        # `page_index[0, 0:1]`: the block's page, column 0 of its stick-wide table row.
        page = tl.load(desc_pt + i * 32)
        # The gather: ONE (page, kv head) slab of each pool, selected by the page id.
        k_page = tl.reshape(kp_desc.load([page, h, 0, 0]), [BLOCK, HD])
        v_page = tl.reshape(vp_desc.load([page, h, 0, 0]), [BLOCK, HD])
        scores = tl.dot(q, k_page.T, out_dtype=tl.float16) * SCALE
        # The block's mask tile, at the score rows (each query row's mask on each of its heads).
        scores = scores + mask_desc.load([i * MQ * GQA, 0])
        scores_max = tl.max(scores, 1)
        rescale = tl.exp(tl.minimum(tile_max - scores_max, 0.0).to(tl.float32)).to(tl.float16)
        new_max = tl.maximum(tile_max, scores_max)
        probs = tl.exp((scores - new_max[:, None]).to(tl.float32)).to(tl.float16)
        tile_sum = tile_sum * rescale + tl.sum(probs, 1).to(tl.float16)
        tile_out = tile_out * rescale[:, None] + tl.dot(probs, v_page, out_dtype=tl.float16)
        tile_max = new_max
    return tile_out / tile_sum[:, None]


@triton.jit
def paged_attn_fwd(desc_q, desc_o, desc_kp, desc_vp, desc_pt, desc_mask,  #
                   MQ: tl.constexpr, NQH: tl.constexpr, NKVH: tl.constexpr, HD: tl.constexpr,  #
                   BLOCK: tl.constexpr, NUM_PAGES: tl.constexpr, NUM_BLOCKS: tl.constexpr,  #
                   SCALE: tl.constexpr):
    GQA: tl.constexpr = NQH // NKVH
    # The pools, head-major: one `[BLOCK, HD]` slab per (page, kv head).
    kp_desc = tl.make_tensor_descriptor(desc_kp, shape=[NUM_PAGES, NKVH, BLOCK, HD],
                                        strides=[NKVH * BLOCK * HD, BLOCK * HD, HD, 1],
                                        block_shape=[1, 1, BLOCK, HD])
    vp_desc = tl.make_tensor_descriptor(desc_vp, shape=[NUM_PAGES, NKVH, BLOCK, HD],
                                        strides=[NKVH * BLOCK * HD, BLOCK * HD, HD, 1],
                                        block_shape=[1, 1, BLOCK, HD])
    # The mask stack, one `[MQ * GQA, BLOCK]` tile per active block, at the score rows.
    mask_desc = tl.make_tensor_descriptor(desc_mask, shape=[NUM_BLOCKS * MQ * GQA, BLOCK],
                                          strides=[BLOCK, 1], block_shape=[MQ * GQA, BLOCK])
    # The query and output rows; a KV head's group is the `[MQ, GQA * HD]` block of them.
    q_desc = tl.make_tensor_descriptor(desc_q, shape=[MQ, NQH * HD], strides=[NQH * HD, 1],
                                       block_shape=[MQ, GQA * HD])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[MQ, NQH * HD], strides=[NQH * HD, 1],
                                       block_shape=[MQ, GQA * HD])
    for h in tl.static_range(NKVH):
        q = tl.reshape(q_desc.load([0, h * GQA * HD]), [MQ * GQA, HD])
        o = _page_walk(q, kp_desc, vp_desc, mask_desc, desc_pt, h, MQ, GQA, HD, BLOCK,
                       NUM_BLOCKS, SCALE)
        o_desc.store([0, h * GQA * HD], tl.reshape(o, [MQ, GQA * HD]))

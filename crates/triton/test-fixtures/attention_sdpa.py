# Copyright (c) 2026 IBM Corporation. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""attention_sdpa fixture: Granite SDPA, ONE algorithm for decode and prefill.

Supersedes the earlier `paged_attention_decode.py`, which read a block table
INSIDE the kernel. That could not lower, and the reason is architectural rather
than a missing feature -- see "why there is no block table here" below.

Conventions are taken from scratchy's hardware-proven Spyre attention
(`~/git/scratchy/crates/targets/spyre/src/ir/bridge/tiled_op_sdsc_op/attn.rs`),
itself a port of torch-spyre's `spyre__sdpa_overrideable`
(`decompositions.py:527`). Where this kernel and that implementation could
differ, this kernel follows that implementation, because it is the one with
device evidence behind it.

WHY THERE IS NO BLOCK TABLE HERE
--------------------------------
Established from the deeptools source: an HBM (L3LU/L3SU) transfer's base
address may be a runtime SSA value -- `constructImmutableAddress`
(`AgenToSentient/Helper.cpp:1217`) inserts the memory view's start address
directly with no constant check. But the primitive that PRODUCES an address by
reading memory is the LX Indirect-Base-Register chain (load index from LX ->
`LXVIRTUALIBR` -> `agen.indirect_vector_load`), and every stage of it is hard
gated to LX-family units (`Helper.cpp:403/474/488/508`, lowering refuses
off-unit at `3172/3221/3271/3317`). There is no L3/HBM equivalent. Staging the
KV cache into LX to gather it there is ruled out on capacity (~20070 usable LX
sticks; a 4096-token KV cache does not fit).

So a Spyre kernel cannot read a block table and use it to address an HBM fetch.
scratchy resolves this exactly the way this kernel assumes: the HOST walks the
block table and installs a page map before launch (`fold_plan.rs`:
`phys = block_tables[row][page]; Bytes(phys * page_stride_bytes)`), and a
request's pool row is its identity for its whole life SPECIFICALLY so that "its
KV base is affine in r, and therefore ... one launch [can] address the whole
batch".

HOST CONTRACT (the part that is not in this file)
-------------------------------------------------
This kernel addresses K/V as a logically contiguous `[rows, nqh, max_seq, d]`
view, affine in the row index. Paging is INVISIBLE to it. The caller must
present that view over physically scattered pages -- on Spyre via a paged
logical memory view whose per-page base addresses the host resolved. Cost of
that arrangement is measured, not free: scratchy records the fold as
`pages x requests` passes at a fixed cost per pass, ~2.05 ms per row at
batch <= 8, and calls those passes "the batched-decode scaling curve".

DELIBERATE DESIGN CHOICES, EACH WITH A REASON
---------------------------------------------
* GQA IS DEDUPED: the KV cache carries `nkvh` DISTINCT heads, and query head `qh`
  reads its group's plane at `gqa_kv_head(qh, gqa) = qh / gqa`. There is exactly
  ONE Spyre KV cache structure and this is it:
      pub struct PagedKvPool { pub nkvh: usize, pub hd: usize }
      // "DISTINCT kv heads -- a query head reads its group's plane via gqa_kv_head"
  (`scratchy/crates/compiler/subtile/src/sdsc_abstract.rs`), whose companion note
  spells out the reason: caches sized to the `nkvh` distinct heads give "4x less
  restickify traffic + 4x smaller resident KV", and warns that "passing a RAW
  `qh` here (the pre-dedup `qh*hd*cap`) indexes OOB / a wrong kv-head once the
  cache is `nkvh`-sized".
  The `nqh`-shaped `[nqh, cap, hd]` buffer that appears in attn.rs's
  `qh*cap*hd` addressing is a per-launch STAGED DESTINATION (`lower_subtile_tape_
  to_superdsc.rs`: "DEST `kc` is [nqh, cap, hd]"), not the pool. An earlier
  version of this fixture read that as a cache layout and stored K/V
  GQA-expanded, which would have read the wrong head's keys: right shape, wrong
  values, and no shape check catches it. scratchy makes an nqh/nkvh transposition
  a deliberate build error for exactly that reason.
* PAGE GEOMETRY: the Spyre pool's page is `PagedKvPool::PAGE_SLOTS = 256` tokens
  (not 16, not 64). 256 is a multiple of BLOCK_N=64, so the one-stick tiling
  divides a page exactly 4 ways -- no sub-stick addressing. Page size should come
  from the backend's `required_block_size()`, not from a constant chosen here.
* BLOCK_N is one stick (64) and that is a CORRECTNESS requirement, not tuning.
  A documented Spyre defect makes reduce-MAX over more than one stick of columns
  mis-combine partial per-tile maxima, so the online-softmax reduce must stay
  within a single stick per step.
* ONE path for decode and prefill, selected by query-row count MQ (1 = decode,
  >1 = chunked prefill) rather than two implementations -- as scratchy does.
* Scale is SPLIT across both operands, as the reference decomposition does
  (`matmul(q*s, (k*s).T)`), which keeps the fp16 intermediates smaller than
  folding the whole multiplier into one side. `s = sqrt(1/d_h)`, so `s*s`
  recovers Granite's `attention_multiplier = 1/d_h` exactly. NOTE: `s` is the
  PER-OPERAND scale, not the multiplier -- passing `1/d_h` here would scale
  scores by `1/d_h**2` and silently flatten the distribution (softmax is
  shift-invariant but NOT scale-invariant). The oracle below applies `1/d_h`
  ONCE to the product, derived from the HF Granite definition rather than from
  this kernel's structure, so it cannot share that class of error.
* Loop trip count is STATIC (`MAX_BLOCKS`) with masking for the live context
  length, because affine loop bounds cannot be runtime values. A runtime
  `context_len` is read only to build a MASK (a predicate), never an address.

NOTE ON HEAD BATCHING (not expressed here, deliberately)
--------------------------------------------------------
scratchy runs every reduce/pointwise op once per block across all `nqh*mq` rows
in a shared buffer; looping per head instead cost them a measured 31 -> 13.7
tok/s. That is an op-count concern in a hand-emitted path. Expressing it here
would mean hand-fusing heads into the tile, which is the emitter's job in a
Triton flow. Recorded so it is a conscious omission rather than an oversight:
if the emitted DFIR reissues the reduce chain per head, this is the known cause.
"""

import torch

import triton
import triton.language as tl

# Granite 3.3-8B. GQA is DEDUPED, so the cache carries NUM_KV_HEADS planes and a
# query head reads plane `head // GQA`.
NUM_SEQS = 4
NUM_HEADS = 32
NUM_KV_HEADS = 8
GQA = NUM_HEADS // NUM_KV_HEADS  # 4 query heads share one kv plane
HEAD_DIM = 128
MAX_SEQ = 4096
MQ_DECODE = 1

BLOCK_N = 64  # exactly one fp16 Spyre stick -- see docstring, this is correctness
ATTENTION_MULTIPLIER = 1.0 / HEAD_DIM  # Granite: 1/d_h, NOT 1/sqrt(d_h)
DTYPE = torch.float16


def operand_scale(head_dim: int = HEAD_DIM) -> float:
    """PER-OPERAND scale for the split form: applied to q AND k, so the product
    carries Granite's attention_multiplier = 1/head_dim exactly."""
    return (1.0 / head_dim) ** 0.5


@triton.jit
def attention_sdpa_kernel(
    out_ptr,  # [rows, nqh, mq, d]  viewed 2-D as [rows*nqh*mq, d]
    q_ptr,  # [rows, nqh, mq, d]
    k_ptr,  # [rows, NKVH, max_seq, d]  GQA-DEDUPED, logically contiguous
    v_ptr,  # [rows, NKVH, max_seq, d]
    context_lens_ptr,  # [rows] int32 -- live token count, used for MASKING only
    scale,
    n_qrows,  # rows*nqh*mq     -- descriptor extent for q/out
    n_krows,  # rows*NKVH*max_seq -- descriptor extent for k/v
    MQ: tl.constexpr,
    NQH: tl.constexpr,
    NKVH: tl.constexpr,
    GQA: tl.constexpr,
    MAX_SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)

    ctx = tl.load(context_lens_ptr + row)

    m_off = tl.arange(0, MQ)
    n_off = tl.arange(0, BLOCK_N)

    # DESCRIPTOR FORM (not raw tt.addptr). ConvertTTIRToKTDP only converts
    # descriptor ops -- DescriptorLoad/Store/Gather/Scatter -- and silently leaves
    # raw pointer arithmetic untouched, which starves PlanCorelets of the
    # ktdp.construct_access_tile it needs to derive the corelet partition. So the
    # descriptor form is what makes this kernel lowerable at all, not a style
    # choice. Every existing fixture (vector_add, mul, bias_add_f32) uses it.
    #
    # The 4-D logical tensors are addressed as 2-D [n_rows, HEAD_DIM] views: the
    # innermost dim is contiguous (stride 1, required for a descriptor) and the
    # leading dims collapse into one row index. Paging stays invisible -- the host
    # presents this contiguous view over scattered pages (see HOST CONTRACT).
    q_desc = tl.make_tensor_descriptor(q_ptr, shape=[n_qrows, HEAD_DIM],
                                      strides=[HEAD_DIM, 1],
                                      block_shape=[MQ, HEAD_DIM])
    o_desc = tl.make_tensor_descriptor(out_ptr, shape=[n_qrows, HEAD_DIM],
                                      strides=[HEAD_DIM, 1],
                                      block_shape=[MQ, HEAD_DIM])
    k_desc = tl.make_tensor_descriptor(k_ptr, shape=[n_krows, HEAD_DIM],
                                      strides=[HEAD_DIM, 1],
                                      block_shape=[BLOCK_N, HEAD_DIM])
    v_desc = tl.make_tensor_descriptor(v_ptr, shape=[n_krows, HEAD_DIM],
                                      strides=[HEAD_DIM, 1],
                                      block_shape=[BLOCK_N, HEAD_DIM])

    # Scale folded into Q up front; K is scaled per block below. Both operands
    # scaled, matching the reference decomposition.
    #
    # `scale` MUST arrive as f16 (signature "fp16"). If it comes in as f32, f16 *
    # f32 promotes and the entire score chain becomes genuine f32 -- not a
    # widening island -- which LegalizeTypes rightly refuses. Declaring it f16 is
    # cheaper than casting in-kernel, and works identically under the interpreter
    # (where it is a plain Python float and NEP-50 weak promotion keeps f16).
    q_row0 = (row * NQH + head) * MQ
    q = q_desc.load([q_row0, 0]) * scale

    # ACCUMULATE IN F16, matching scratchy's Df::Fp16 attention path throughout.
    # NOT an oversight and NOT a concession to the compiler: f32 accumulators
    # would halve the lane count (elemsPerStick("f32") == 32 vs 64 for f16, from
    # bytesPerStick=128), so a 64-wide score tile would span TWO sticks and hit
    # the documented reduce-MAX defect this kernel tiles to avoid. Spyre also
    # RED-stops genuine f32 compute in LegalizeTypes by design -- demoting it is
    # an owner decision, and this is that decision recorded at the source rather
    # than worked around in the pass.
    m_i = tl.full([MQ], float("-inf"), dtype=tl.float16)
    l_i = tl.zeros([MQ], dtype=tl.float16)
    acc = tl.zeros([MQ, HEAD_DIM], dtype=tl.float16)

    # gqa_kv_head(qh, gqa) = qh / gqa -- the shared kv plane this query head reads.
    # Using the raw `head` here would index a wrong kv plane (and run off the end
    # of an nkvh-sized cache), which is the exact mistake scratchy's note warns
    # about and which no shape check would catch.
    kv_row0 = (row * NKVH + head // GQA) * MAX_SEQ

    # Static trip count; the live length enters as a mask, never as a bound.
    #
    # The LOADS are unmasked. Every block is inside the descriptor's declared
    # extent (MAX_SEQ is a multiple of BLOCK_N), so a load past the live context
    # reads real allocated cache memory -- stale KV, but finite fp16. Those
    # columns are then masked to -inf in the scores, and their softmax weight is
    # pinned to 0, so stale values cannot reach the output. Dropping the load mask
    # is what lets this be a plain descriptor load instead of a predicated
    # pointer load.
    for blk in range(MAX_BLOCKS):
        s_off = blk * BLOCK_N + n_off
        live = s_off < ctx

        k = k_desc.load([kv_row0 + blk * BLOCK_N, 0]) * scale

        # [MQ, BLOCK_N] scores. One stick wide, so the reduce below stays within
        # a single stick -- required, see docstring.
        #
        # `.to(tl.float16)` on every tl.sum is load-bearing, not cosmetic. Triton
        # auto-widens reductions to f32 for stability, so without it the f32
        # leaks into the loop-carried accumulators and Triton's own frontend
        # rejects the type inconsistency ("Loop-carried variable m_i has initial
        # type fp16 but is re-assigned to fp32"). Truncating here is what makes
        # the widening a local extf/reduce/truncf ISLAND -- exactly the shape
        # LegalizeTypes collapses -- instead of genuine f32 compute it must
        # refuse. It also matches the hardware, which rounds to DL16 at every
        # MAC rather than keeping a wide accumulator.
        scores = tl.sum(q[:, None, :] * k[None, :, :], axis=2).to(tl.float16)

        keep = live[None, :]
        if CAUSAL:
            # Query row r attends key position s iff s <= (ctx - MQ + r).
            q_pos = ctx - MQ + m_off
            keep = keep & (s_off[None, :] <= q_pos[:, None])
        neg_inf = tl.full(scores.shape, float("-inf"), dtype=tl.float16)
        scores = tl.where(keep, scores, neg_inf)

        # Every reduction AND every transcendental is truncated back to f16.
        # Triton widens both, and any f32 that reaches a loop-carried variable is
        # a frontend type error before Spyre sees the IR at all.
        blk_max = tl.max(scores, axis=1).to(tl.float16)
        m_new = tl.maximum(m_i, blk_max)
        # An all-masked block leaves m_new at -inf; exp(-inf - -inf) is nan, so
        # pin the correction to 1 and the weights to 0 for that case.
        finite = m_new > tl.full([MQ], float("-inf"), dtype=tl.float16)
        m_safe = tl.where(finite, m_new, tl.zeros([MQ], dtype=tl.float16))
        # tl.exp requires f32 input, so widen for the exp and truncate straight
        # back: extf -> exp -> truncf is precisely the stability-widening ISLAND
        # LegalizeTypes is built to collapse, which is why it belongs here rather
        # than as an f32 accumulator the pass would have to refuse.
        correction = tl.where(finite,
                              tl.exp((m_i - m_safe).to(tl.float32)).to(tl.float16),
                              tl.full([MQ], 1.0, dtype=tl.float16))
        p = tl.where(keep & finite[:, None],
                     tl.exp((scores - m_safe[:, None]).to(tl.float32)).to(tl.float16),
                     tl.zeros(scores.shape, dtype=tl.float16))

        l_i = l_i * correction + tl.sum(p, axis=1).to(tl.float16)

        v = v_desc.load([kv_row0 + blk * BLOCK_N, 0])

        acc = acc * correction[:, None] + tl.sum(p[:, :, None] * v[None, :, :],
                                                 axis=1).to(tl.float16)
        m_i = m_new

    acc = acc / l_i[:, None]

    o_desc.store([q_row0, 0], acc)


def inputs(seed: int = 0, num_seqs: int = NUM_SEQS, mq: int = MQ_DECODE):
    """fp16 inputs. K/V carry NUM_KV_HEADS DISTINCT planes (GQA-deduped, matching
    PagedKvPool), logically contiguous -- paging is the host's business."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(num_seqs, NUM_HEADS, mq, HEAD_DIM, generator=g, dtype=DTYPE)
    k = torch.randn(num_seqs, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM, generator=g, dtype=DTYPE)
    v = torch.randn(num_seqs, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM, generator=g, dtype=DTYPE)
    ctx = torch.randint(mq, MAX_SEQ + 1, (num_seqs,), generator=g, dtype=torch.int32)
    ctx[0] = MAX_SEQ
    if num_seqs > 1:
        ctx[1] = BLOCK_N + 1  # partial tail block
    return q, k, v, ctx


def reference(q, k, v, ctx, causal: bool = False):
    """Native PyTorch oracle, written from the HF Granite definition:
        scores = (Q @ K^T) * attention_multiplier ; softmax ; @ V
    with GQA handled by broadcasting each kv plane to its group of query heads.
    The multiplier is applied ONCE to the product here -- deliberately NOT
    mirroring the kernel's split-scale form, so this oracle cannot share the
    kernel's scaling errors."""
    num_seqs, nqh, mq, d = q.shape
    nkvh = k.shape[1]
    gqa = nqh // nkvh
    mult = 1.0 / d
    out = torch.empty_like(q)
    for r in range(num_seqs):
        c = int(ctx[r].item())
        qs = q[r].to(torch.float32)                              # [nqh, mq, d]
        # Broadcast each distinct kv plane across its gqa query heads. Computed
        # here from the shapes rather than mirroring the kernel's `head // GQA`,
        # so a wrong grouping in the kernel still shows up as a mismatch.
        ks = k[r, :, :c, :].to(torch.float32).repeat_interleave(gqa, dim=0)
        vs = v[r, :, :c, :].to(torch.float32).repeat_interleave(gqa, dim=0)
        scores = torch.einsum("hmd,hsd->hms", qs, ks) * mult
        if causal:
            q_pos = c - mq + torch.arange(mq)
            mask = torch.arange(c)[None, :] <= q_pos[:, None]
            scores = scores.masked_fill(~mask[None, :, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[r] = torch.einsum("hms,hsd->hmd", probs, vs).to(DTYPE)
    return out

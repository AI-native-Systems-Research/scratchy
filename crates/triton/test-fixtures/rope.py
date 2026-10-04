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

"""Granite's RoPE: Hugging Face SPLIT-HALF `rotate_half`, NOT the interleaved form.

    q_embed = q * cos + rotate_half(q) * sin
    rotate_half(x) = cat(-x[..., d//2:], x[..., :d//2])

which, written per half (x1 = x[..., :d//2], x2 = x[..., d//2:]), is exactly

    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin

Granite-3.3 8B: head_dim = hidden_size / num_attention_heads = 4096 / 32 = 128, so each
half is 64 elements. `rope_theta = 1e7` and there is no rope scaling
(`attention_scaling = 1.0`, all 128 dims rotated) -- from
`crates/models/arch/configs/granite/granite-3.3-8b-instruct.json` and this repo's own
`journal/artifacts/granite-block/2026-07-10-granite-block-plan.md`. NONE of those constants
appear below: `cos` and `sin` are a TABLE the host builds, which is what makes the kernel
theta-independent and scaling-independent.

===============================================================================
THE SLICING FORM WAS MEASURED, NOT ASSUMED, AND IT IS REFUSED -- BY TRITON
===============================================================================

The obvious transcription of `rotate_half` is a partial slice, and it does not exist in
Triton at all. Compiled with real `make_ir`:

    x1 = x[:, :HALF]
    -> ValueError: unsupported tensor index: <triton.language.core.slice object ...>
       CompilationError: at 8:9

`tensor.__getitem__` (`python/triton/language/core.py`) accepts exactly two things: `None`
(which is an `expand_dims`) and a FULL `:` slice (which is a no-op). Anything with a `start`
or a `stop` raises. So the refusal is not this front end's boundary -- it is the language's,
and no amount of work on the Rust bridge would change it.

A SECOND FORM DOES COMPILE IN TRITON, and it was measured too, because "refused" would have
been the wrong answer if an expressible spelling existed:

    r = tl.reshape(x, [BLOCK_M, 2, HALF])   -> tt.reshape  : tensor<64x128xf16> -> tensor<64x2x64xf16>
    p = tl.permute(r, [0, 2, 1])            -> tt.trans {order = array<i32: 0, 2, 1>}
    a, b = tl.split(p)                      -> tt.split    : tensor<64x64x2xf16> -> two tensor<64x64xf16>

IT IS NOT USED HERE, for two reasons and the second is the load-bearing one:

  * it needs THREE constructs the Rust bridge does not have -- `tt.reshape`, `tt.split` (a
    TWO-result op, so a tuple-bound assignment), and a rank-3 tile, which no fixture in the
    tree has ever produced;
  * and it asks for a permutation ACROSS THE LANE AXIS. The only lane permutations attested
    on this device are the reduction fold's `NFWD0` / `NFWD2` within 8
    (`test/pod/scheduler_probes/README.md`, the reduction-FOLD entry), with two recorded
    traps around them -- a shuffle whose input is also an operand of the combine is SILENTLY
    dropped, and a shuffle feeding `vectorchain.binary` CORE DUMPS. An arbitrary interleave
    of 128 lanes into 64x2 is nothing like that shape, and nothing attests it.

===============================================================================
WHAT THIS KERNEL DOES INSTEAD: THE HALVES ARE STICKS, SO THE ROTATION IS FREE
===============================================================================

At head_dim 128 in f16, `arch.elemsPerStick("f16")` is 64, so **each half is EXACTLY ONE
STICK**. The two halves are therefore two ordinary descriptor loads of the same tensor at
column offsets 0 and HALF, and `rotate_half` becomes WHICH STICK EACH PRODUCT READS -- no
data movement, no permutation, no reshape. That is the whole reason this fixture is four
loads and two stores rather than a construct problem.

THE DELTA, and the justification for each item
==============================================

1. TWO LOADS AND TWO STORES INSTEAD OF ONE OF EACH (see above). There is no concatenate in
   this kernel because there is nothing to concatenate: the output halves are written where
   they belong. `attention_flash.py`'s delta 5 keeps that kernel to one store so the emitter
   sees one; here two stores is what the algorithm is, and they are to DISJOINT column
   ranges of the same descriptor.

2. `cos` AND `sin` ARE HALF-WIDTH TABLES, AND THAT IS EXACT RATHER THAN AN APPROXIMATION.
   ⛔ RETIRED BY DELTA 8 AS A STORAGE CLAIM -- the arithmetic below is still true and the
   BUFFER is now full width and head-replicated. Read delta 8 before proposing this again.
   HF's rotary embedding builds `emb = torch.cat((freqs, freqs), dim=-1)` and then
   `cos = emb.cos()`, so `cos[..., :d//2]` and `cos[..., d//2:]` are THE SAME NUMBERS. A
   full-width table would be two identical sticks per row and both halves of the kernel would
   read the same values from different addresses. Halving it is not a shortcut, it is
   deleting a duplicate -- and it means one stick per row instead of two.

3. `cos` / `sin` ARE INDEXED BY POSITION ONLY, NOT BY HEAD.
   ⛔ RETIRED BY DELTA 8 AS A DESCRIPTOR CLAIM -- the angles still depend on position alone, but
   the binding can no longer be `[N_TOK, HALF]`, and `offs_p` is gone. See delta 8. `apply_rotary_pos_emb`
   unsqueezes them across the head axis, so every head at a position reads the same row.
   That is why `offs_p` omits `off_h` while `offs_m` includes it -- the one place a reader
   should look twice, and the reason the two offsets are separate variables.

4. H / N_TOK / HEAD_DIM ARE constexpr (SPYRE-SPECIFIC), for `attention_flash.py`'s delta 7
   reason: affine bounds are static and the emitter derives its work division from the
   descriptor extents.

5. THE TABLE IS f16, LIKE EVERYTHING ELSE (delta 2 of `attention_flash.py`). HF computes
   `cos`/`sin` in f32 and casts to the activation dtype; the host does that cast once when it
   builds the table, so no widening island appears in this kernel. `reference()` below keeps
   HF's f32 table and casts at the end, which is what will show the difference if the f16
   table ever costs anything.

6. NOT DONE, AND DELIBERATELY: THE Q AND K PLANES ARE NOT FUSED. RoPE applies to q and to k,
   with the same table and different extents (Granite is GQA: 32 query heads, 8 kv heads).
   Two launches of this kernel do it, and a fused version would have to carry both extents
   and both descriptors for no structural gain. `H` is a constexpr, so a q launch and a k
   launch are two configurations of the same source -- which is exactly how `rope` and
   `rope_granite_kv` below differ.

7. THE ROWS ARE TOKEN-MAJOR: row = position * H + head. THIS FIXTURE USED TO BE HEAD-MAJOR
   (`offs_m = off_h * N_TOK + start_m * BLOCK_M` over an `[H * N_TOK, HEAD_DIM]` buffer) AND
   THAT WAS WRONG.

   ⛔ THE TWO ORDERS HAVE THE SAME SHAPE AND DIFFERENT BYTES, so nothing catches a mismatch.
   `[H * N_TOK, HEAD_DIM]` and `[N_TOK * H, HEAD_DIM]` are the same `[8192, 128]` view; the
   arrangement authority's `addr_eq` admits a same-total contiguous reshape and the footprint
   guard sees one buffer of the right size. A head-major buffer under a token-major consumer
   comes out as a wrong ANSWER, not as a build error -- which is why the backend now names the
   order out loud (`RopeRows` in `rust/triton-ktir-superdsc/src/lib.rs`) instead of inferring
   it, and why this delta exists rather than a silent edit.

   THE PRODUCER SETTLES IT. `KtirFunc::rope` (scratchy `lower_subtile_tape_to_ktir.rs:3527`),
   which the device emitter `rope_at` was written against, is five lines:

       let mh = rows * heads;
       let x_view = self.view_shaped(x_t, mh, hd);        # the [rows*heads, hd] view
       for ri in 0..rows {
           let rbase = self.idx(ri * heads);              # <- position ri's ROW BASE
           let xf_acc = self.tile(x_view, rbase, zero, heads, half);   # heads ROWS TALL
           let cos1 = self.load_1d(cos_t, rows * tbl_cols, ri * tbl_cols, half);
           let cosb = self.broadcast(cos1, vec![heads, half], 0);      # ONE row, over those heads

   Position `ri` occupies rows `ri*heads .. ri*heads + heads`, one row per head, and that
   position's SINGLE cos/sin row is broadcast across exactly those rows. So the row index is
   `token*heads + head`. Head-outermost rows would make that same tile `heads` CONSECUTIVE
   TOKENS of one head, every one of them rotated by position `ri`'s angle.

   AND IT IS THE SAME BYTES ATTENTION READS. Row-major `[mq, heads*hd]` puts element
   `(ri, h*hd + d)` at flat `ri*heads*hd + h*hd + d`, which IS row `ri*heads + h`, column `d`
   of the tall `[mq*heads, hd]` view -- the contiguous reshape. A head-outermost tall view is
   instead the reshape of `[heads, mq, hd]`, i.e. the TRANSPOSED plane, so a head-major RoPE
   would force a transpose between it and the attention that consumes it.

   WHAT IT COSTS HERE, and it is more than the offset expression. Under token-major, 64
   consecutive rows are NOT 64 tokens of one head -- they are a mix of tokens and heads -- so
   a `[BLOCK_M, HALF]` block can no longer pair each of its rows with a table row of the same
   position. A work item therefore covers ONE WHOLE TOKEN: the block is `[H, HALF]` at row
   base `pos * H`, and cos/sin are read ONE ROW at a time and broadcast across the H head
   rows. That is `rope_at`'s `broadcast(cos1, [heads, half], 0)` written in Triton, and it is
   why `BLOCK_M` is gone: there is nothing left to block, and the grid indexes positions.

   A BLOCK OF SEVERAL POSITIONS IS NOT EXPRESSIBLE HERE, which is why BLOCK_POS is not a
   parameter. `[BLOCK_POS, HALF]` -> `[BLOCK_POS * H, HALF]` needs each table row repeated H
   times, i.e. `expand_dims` to rank 3, a broadcast, then a `reshape` back to rank 2 -- the
   same three constructs the module docstring above records as absent from the Rust bridge
   (`tt.reshape`, and a rank-3 tile no fixture in this tree has produced). The size-1 row axis
   this kernel does use is an ordinary Triton broadcast and needs none of them.

   THE TABLES ARE STILL INDEXED BY POSITION ONLY, which is delta 3 -- and delta 3 is precisely
   why the row reorder works at all: no table row has ever carried a head term, so reordering
   the head axis cannot touch it. Their staged WIDTH is delta 8.

8. THE cos/sin TABLES ARE BOUND AT `x`'s EXTENT, `[N_TOK * H, HEAD_DIM]`, HEAD-REPLICATED --
   which RETIRES delta 2's storage claim and delta 3's descriptor claim. Both are kept above,
   not deleted: a future reader who proposes a half-width position-indexed table again should
   find why it was tried and why it cannot work here.

   MEASURED, and that is why this delta exists rather than an argument. With the half-width
   `[N_TOK, HALF]` tables the device emitter aborts:

       assemble_pointwise_broadcast rope_xc_o3: t1: access offset 0B + 2097152B exceeds its
       placement footprint 32768B (seg3) -- the op addresses PAST its own tensor and would
       alias whatever is placed next.

   32768B is `[256, 64]` f16, the half-width table this fixture used to bind. 2097152B is
   `256 * (32 * 128) * 2` -- `mq * total` -- the WHOLE roped plane, which is the extent
   `rope_at` reads a table at. `rope_kv8` gives the same wall at 524288B = `256 * (8 * 128) * 2`.
   So the guard is right and the fixture's table contract was wrong.

   WHY, AND IT IS ALGORITHMIC. At `mq > 1` (prefill) `rope_at` reads cos/sin with `In::full` --
   the whole tensor, not a row -- because it does the rotation as a P-matrix matmul plus a slab
   swap and then two FULL-WIDTH pointwise multiplies, one table row per (position, head). Its own
   comment: "cos/sin are the worker's [mq,total] per-position head-tiled table" and "The table is
   head-tiled (every head holds the same row), so head h's copy already sits at row h*mq+r".
   That whole-tensor read is what lets it emit `4 * heads` ops instead of `4 * mq * heads`
   (Q: 3968 -> 128, its own numbers). The staging buys that, so it is a layout the emitter
   REQUIRES.

   THE FIXTURE'S PER-HALF STICK ALGEBRA IS A DIFFERENT ALGORITHM FROM THE DEVICE BODY'S, and
   that is the honest statement of what deltas 2 and 3 got wrong. "Each half is exactly one
   stick, so rotate_half becomes which stick each product reads" is true of THIS kernel and
   false of `rope_at`, which rotates by matmul. Delta 2's arithmetic claim survives intact --
   `cos[..., :d//2]` and `cos[..., d//2:]` really are the same numbers, and every head at a
   position really does share them, so the staged table is PURE DUPLICATION: one `[N_TOK, HALF]`
   block of distinct angles, present `2 * H` times per position. What does not survive is the
   conclusion that the buffer may therefore be that small. Delta 3's fact also survives -- the
   angles depend on position only -- but the DESCRIPTOR can no longer say so, because a
   position-indexed `[N_TOK, HALF]` binding under a whole-plane reader is an out-of-bounds
   access, not a saving.

   `[N_TOK * H, HEAD_DIM]` RATHER THAN `[N_TOK, H * HEAD_DIM]`, and they are the same bytes:
   row-major, element (t, h, d) at flat `(t*H + h)*HEAD_DIM + d` either way. The tall form is
   bound because its row stride is `HEAD_DIM`, a plain constexpr, where the wide form needs
   `H * HEAD_DIM` -- an i32 multiply the descriptor's i64 stride sign-extends, and MEASURED:
   "ttir operation `arith.extsi` is not in this adapter's measured op set". The tall form also
   lets the tables be read at `x`'s own row base, which is what makes the H rows line up with
   `x1`/`x2` row for row with no broadcast at all.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def rope_fwd(desc_x, desc_cos, desc_sin, desc_o,  #
             H: tl.constexpr, N_TOK: tl.constexpr,  # SPYRE: delta 4
             HEAD_DIM: tl.constexpr,  #
             HALF: tl.constexpr,  #
             ):
    tl.static_assert(HALF + HALF == HEAD_DIM)
    # ONE WHOLE TOKEN PER WORK ITEM (delta 7). The grid indexes POSITIONS, not (block, head):
    # all H heads of a position live in one tile, because that is the only way the single
    # cos/sin row of that position can reach them.
    pos = tl.program_id(0)
    y_dim = N_TOK * H
    # THE BLOCK IS `[H, HALF]` -- `heads` ROWS TALL, one half wide (delta 1, delta 7). At
    # head_dim 128 in f16 each half is exactly one 64-lane stick, so the two halves are the
    # same descriptor at column offset 0 and at HALF.
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[y_dim, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[H, HALF])
    # THE TABLES ARE BOUND AT `x`'s OWN EXTENT and read at `x`'s own row base (delta 8): one
    # table row per (position, head), so the H rows this work item reads already hold H copies of
    # this position's angles. Still one half-stick wide per read (delta 2's arithmetic), and still
    # a function of the position alone (delta 3's independence) -- what changed is that the HOST
    # stages the replication instead of the kernel asking for a narrower buffer.
    #
    # THE SHAPE IS `[y_dim, HEAD_DIM]`, NOT `[N_TOK, H * HEAD_DIM]`. Those are the SAME BYTES --
    # row-major, element (t, h, d) at flat (t*H + h)*HEAD_DIM + d either way -- but the second
    # needs a `H * HEAD_DIM` row stride, which materialises an i32 multiply that the descriptor's
    # i64 stride sign-extends. MEASURED: "ttir operation `arith.extsi` is not in this adapter's
    # measured op set". Binding the tables exactly like `x` keeps every stride a plain constexpr.
    cos_desc = tl.make_tensor_descriptor(desc_cos, shape=[y_dim, HEAD_DIM],
                                         strides=[HEAD_DIM, 1],
                                         block_shape=[H, HALF])
    sin_desc = tl.make_tensor_descriptor(desc_sin, shape=[y_dim, HEAD_DIM],
                                         strides=[HEAD_DIM, 1],
                                         block_shape=[H, HALF])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[y_dim, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[H, HALF])

    # TOKEN-MAJOR ROWS (delta 7): row = position * H + head, so position `pos` owns the H
    # consecutive rows starting at `pos * H` -- one row per head. The table row is `pos`
    # itself, with no head term at all (delta 3).
    offs_m = pos * H

    x1 = x_desc.load([offs_m, 0])
    x2 = x_desc.load([offs_m, HALF])
    # The SAME row base as `x`: the table is head-replicated, so these H rows are H copies of
    # position `pos`'s angles and line up with `x1`/`x2` row for row with no broadcast.
    c = cos_desc.load([offs_m, 0])
    s = sin_desc.load([offs_m, 0])
    # rotate_half, per half: the MINUS is on the second half's contribution to the first,
    # which is `cat(-x2, x1)` written out.
    o_desc.store([offs_m, 0], x1 * c - x2 * s)
    o_desc.store([offs_m, HALF], x2 * c + x1 * s)


# --- the Spyre configuration ---------------------------------------------------
# Granite-3.3 8B, from its config.json: hidden_size 4096 / num_attention_heads 32 = head_dim
# 128, num_key_value_heads 8 (so GQA 4), rope_theta 1e7 (host-side, see the module docstring).
HEAD_DIM = 128
GRANITE_Q_HEADS = 32
GRANITE_KV_HEADS = 8

# NO `BLOCK_M`: a work item is exactly one position, H rows tall (delta 7). The launch grid is
# `[N_TOK]` -- positions -- and the head axis is no longer a grid axis, because all H heads of a
# position have to share that position's one cos/sin row.
SIGNATURE = {
    "desc_x": "*fp16", "desc_cos": "*fp16", "desc_sin": "*fp16", "desc_o": "*fp16",
    "H": "constexpr", "N_TOK": "constexpr", "HEAD_DIM": "constexpr",
    "HALF": "constexpr",
}


def grid(n_tok=256):
    """The launch grid: one work item per POSITION (delta 7).

    `H` no longer appears here. It still separates the q configuration from the kv one, but
    through the PLANE WIDTH `H * HEAD_DIM` and the block height `H`, not through a grid axis.
    """
    return (n_tok,)


def constexprs(h=4, n_tok=256, head_dim=HEAD_DIM):
    """One configuration. `H` is the head count of the plane being rotated -- 32 for Granite's
    queries, 8 for its keys (delta 6)."""
    return {"H": h, "N_TOK": n_tok, "HEAD_DIM": head_dim,
            "HALF": head_dim // 2}


def inputs(seed: int = 0, h=4, n_tok=256, head_dim=HEAD_DIM, theta=1e7):
    """An x plane laid out TOKEN-MAJOR -- `[N_TOK * H, HEAD_DIM]`, row = position * H + head
    (delta 7) -- plus the HALF-width cos/sin tables the host owes the kernel (delta 2), built
    the way HF builds them.

    The row order is BUILT rather than commented: the plane is generated as `[N_TOK, H,
    HEAD_DIM]` and reshaped, so the reshape is what states the nest. `[H * N_TOK, HEAD_DIM]`
    and `[N_TOK * H, HEAD_DIM]` are the same shape and the same element count, so a comment
    claiming an order is exactly the thing that cannot be checked.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_tok, h, head_dim, generator=g, dtype=torch.float16).reshape(
        n_tok * h, head_dim)
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    pos = torch.arange(n_tok, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    cos, sin = stage_table(freqs.cos(), h), stage_table(freqs.sin(), h)
    return x, cos, sin


def stage_table(angles: torch.Tensor, h: int) -> torch.Tensor:
    """The `[N_TOK * H, HEAD_DIM]` head-replicated table the device reader requires (delta 8).

    `angles` is the `[N_TOK, HALF]` block of DISTINCT values -- HF's `freqs.cos()`/`.sin()`.
    Staging is two duplications and no arithmetic:

      * `cat(a, a)` to the full head width, which is HF's own
        `emb = torch.cat((freqs, freqs), dim=-1)` left UN-halved;
      * `repeat_interleave(h, dim=0)`, so position `t`'s row appears at rows `t*h .. t*h + h`
        -- one per head, IN PLACE, matching `x`'s token-major nest (delta 7).

    `repeat_interleave`, not `repeat`: `repeat(h, 1)` would tile the whole table h times and put
    position `t`'s angles at row `t + k*n_tok`, which is the HEAD-major nest. Same shape, so only
    the values tell them apart.
    """
    full = torch.cat((angles, angles), dim=-1)              # [N_TOK, HEAD_DIM]
    return full.repeat_interleave(h, dim=0).to(torch.float16)  # [N_TOK * H, HEAD_DIM]


def reference(x: torch.Tensor, cos_staged: torch.Tensor, sin_staged: torch.Tensor,
              h: int, n_tok: int) -> torch.Tensor:
    """Native PyTorch reference: HF's `apply_rotary_pos_emb` on the FULL width.

    Deliberately NOT written per half. The kernel's two-load form is the thing under test, so
    the reference uses the staged full-width table and HF's `rotate_half`. If halving the READ
    were ever wrong, this is where it would show.

    The tables arrive staged `[N_TOK * H, HEAD_DIM]` and head-replicated (delta 8), already row
    for row with `x` -- so there is nothing to repeat here. That alignment is ASSERTED rather
    than assumed: "every head holds the same row" is the fact the device reader relies on, and a
    staging bug would otherwise surface only as wrong numbers on card.
    """
    dtype = x.dtype
    head_dim = x.shape[-1]
    for t, name in ((cos_staged, "cos"), (sin_staged, "sin")):
        assert t.shape == (n_tok * h, head_dim), (
            f"{name} must be staged [N_TOK * H, HEAD_DIM] = "
            f"[{n_tok * h}, {head_dim}], got {tuple(t.shape)}")
        # HEAD-REPLICATED: within a position, all h rows must be identical.
        per_tok = t.reshape(n_tok, h, head_dim)
        assert torch.equal(per_tok, per_tok[:, :1, :].expand_as(per_tok)), (
            f"{name} is not head-replicated: rows within a position differ, so the device "
            f"reader would give each head a different angle")
        # AND BOTH HALVES EQUAL, which is delta 2's arithmetic claim stated as a check.
        assert torch.equal(t[:, :head_dim // 2], t[:, head_dim // 2:]), (
            f"{name}'s two halves differ; HF builds `cat((freqs, freqs))`, so they must not")
    cos = cos_staged.to(torch.float32)
    sin = sin_staged.to(torch.float32)
    xf = x.to(torch.float32)
    d = head_dim // 2
    x1, x2 = xf[..., :d], xf[..., d:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (xf * cos + rotated * sin).to(dtype)

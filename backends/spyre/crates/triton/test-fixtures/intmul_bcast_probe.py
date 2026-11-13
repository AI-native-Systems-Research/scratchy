# SPDX-License-Identifier: Apache-2.0
"""PROBE-ONLY (front-b-noise, auditor order 2026-10-26, the wall-5 (B) ruling): the
INT-MUL-BCAST leg — ONE `arith.muli` over a `[1, 512]` dense i32 row and a `[1, 1]`
Col-broadcast i32 word, the exact card form the integer door's `Mul` broadcast
admission names as its measurement in flight (`int_binary`'s own doc).

WHY THIS KERNEL EXISTS: the wall-5 fold restages the hash's uniform multipliers
(`hc1`/`hc2`/`hc3`, the wrap-fold's `c256`) as `[1, 1]` one-word tables —
broadcast OPERANDS of `muli32toi32`, the same shared `elementwise_broadcast`
body at the same SENUINT32 stick basis the add leg measured. The add leg
(`int_bcast_probe.py`, 2026-10-26T22:31:15Z) measured the Col operand mode for
the ADD alone; the auditor's (B) ruling orders this one op card-legged before
the `Mul`+`Col` admission is cited. This kernel is the measurement's specimen —
the minimal program whose single multiply IS the form in question, nothing else
on the card between the loads and the store.

FIDELITY: identical to the add leg's — `a` is the dense row at the sampler's
own whole-row block shape (`[1, PAD_V]` at corner (0, 0), the shared-table
idiom); `b` is the `[1, 1]` word at the seed word's own shape (a `[1, 1]`
descriptor window of a `[1, 5]` view); the multiply is `a * b`, which Triton's
implicit promotion spells as the collapse + `linalg.generic` broadcast +
`arith.muli` chain the whole-function walk resolves. No constants, no splats,
no f32 — a pure i32 program, so `LegalizeTypes` has no island to demand.

PARAMETERS, IN ORDER: `a` (`[1, PAD_V]` i32), `b` (the `[1, 1]` i32 word,
staged at the SENUINT32 one-stick footprint of 32 words), `o` (the
`[1, PAD_V]` i32 product). CONSTEXPRS: `PAD_V` (512, the auditor's stated
width), `B` (the grid's width, always 1). GRID: `[1]`.

THE VERDICT is a host bit-for-bit compare: every lane `j` of `o` must equal
`(a[j] * b[0]) mod 2^32` — wrapping i32 is exact on both sides, so ANY
mismatch is an addressing defect in the Col operand mode, never numerics.
"""

import triton
import triton.language as tl


@triton.jit
def intmul_bcast_probe(a_ptr, b_ptr, o_ptr,  #
                       PAD_V: tl.constexpr, B: tl.constexpr):
    # THE DENSE ROW — the sampler's whole-row block at corner (0, 0).
    a_d = tl.make_tensor_descriptor(a_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    a = a_d.load([0, 0])
    # THE PER-ROW WORD — the production seed-word spelling verbatim
    # (`sample_fwd.py`'s own `s_desc`): a [1, 5] view read through a [1, 1]
    # tile at column 0, so the load is the seed word's own [1, 1] i32 shape.
    b_d = tl.make_tensor_descriptor(b_ptr, shape=[1, 5], strides=[5, 1],
                                    block_shape=[1, 1])
    b = b_d.load([0, 0])
    # THE OP UNDER MEASUREMENT — Triton's implicit promotion broadcasts the
    # [1, 1] word along the stick axis; the emitted chain is collapse ->
    # linalg.generic broadcast -> arith.muli, the Col-broadcast `muli32toi32`
    # the integer door's `Mul` admission routes through
    # `elementwise_broadcast`.
    o = a * b
    o_d = tl.make_tensor_descriptor(o_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    o_d.store([0, 0], o)

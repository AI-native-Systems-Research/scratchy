# SPDX-License-Identifier: Apache-2.0
"""PROBE-ONLY (front-b-noise, auditor order 2026-10-26, ruling (b)): the INT-BCAST
leg — ONE `arith.addi` over a `[1, 512]` dense i32 row and a `[1, 1]`
Col-broadcast i32 word, the exact card form the integer door's broadcast
admission names as its measurement in flight (`int_binary`'s own doc).

WHY THIS KERNEL EXISTS: the production fold's hash adds its per-row seed words
(`slo`/`shi`/`stp`/`srow`, each a `[rows, 1]` i32 window of the seed table)
against the `[rows, PAD_V]` hash tile — `h = h*C2 + shi + stp + srow` spells
four Col-broadcast `addi32toi32` ops. B1 measured the DENSE SENUINT32 form
only, so the door refused the broadcast by name (the fold's third wall), and
the auditor ruled (b): measure the form on card rather than move 786 KB/row of
host traffic to dodge it. This kernel is the measurement's specimen — the
minimal program whose single add IS the form in question, nothing else on the
card between the loads and the store.

FIDELITY: `a` is the dense row at the sampler's own whole-row block shape
(`[1, PAD_V]` at corner (0, 0), the shared-table idiom); `b` is the per-row
word at the seed word's own shape — a `[1, 1]` descriptor window, exactly the
production `%sgate_2`/`%slo_1` loads' shape; the add is `a + b`, which Triton's
implicit promotion spells as the collapse + `linalg.generic` broadcast +
`arith.addi` chain the whole-function walk resolves (the same chain the gate's
sitofp ruling traversed). No constants, no splats, no f32 — a pure i32 program,
so `LegalizeTypes` has no island to demand.

PARAMETERS, IN ORDER: `a` (`[1, PAD_V]` i32), `b` (the `[1, 1]` i32 word,
staged at the SENUINT32 one-stick footprint of 32 words), `o` (the
`[1, PAD_V]` i32 sum). CONSTEXPRS: `PAD_V` (512, the auditor's stated width),
`B` (the grid's width, always 1). GRID: `[1]`.

THE VERDICT is a host bit-for-bit compare: every lane `j` of `o` must equal
`(a[j] + b[0]) mod 2^32` — wrapping i32 is exact on both sides, so ANY
mismatch is an addressing defect in the Col operand mode, never numerics.
"""

import triton
import triton.language as tl


@triton.jit
def int_bcast_probe(a_ptr, b_ptr, o_ptr,  #
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
    # linalg.generic broadcast -> arith.addi, the Col-broadcast `addi32toi32`
    # the integer door's admission routes through `elementwise_broadcast`.
    o = a + b
    o_d = tl.make_tensor_descriptor(o_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    o_d.store([0, 0], o)

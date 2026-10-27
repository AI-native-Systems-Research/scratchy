# SPDX-License-Identifier: Apache-2.0
"""PROBE-ONLY (front-b-noise, auditor GO 2026-10-27): the INT MULTI-ROW legs — the
multi-row `Col` admission's measurement, four cells for attribution, all in the
PRODUCTION PID-CORNERED SPELLING (grid `[B]`, `[1, ·]` blocks, the per-row loads
at `[pid, 0]` — `sample_fwd.py`'s own grid law at `B = 2`):

  * `intmr_dense_add` / `intmr_dense_mul` — `[2, 512]` dense + `[2, 512]` dense,
    the MULTI-ROW DENSE CONTROL (itself unmeasured: B1 measured `[1, 512]`).
    Isolates the multi-row staging alone — if this cell fails, the defect is in
    the dense multi-row layout, not the broadcast operand mode.
  * `intmr_col_add` / `intmr_col_mul` — `[2, 512]` dense + `[2, 1]` `Col`, the
    form WALL 9 named: a band-typed rung mints its per-request seed words as
    `[rows, 1]` against a `[rows, cols]` tile at `rows > 1` (the granite-3.1-2b
    decode rerung's pod build, 2026-10-27T01:19Z). This cell measures the
    operand mode ON TOP of the control's staging.

WHY THE PER-ROW WORDS DIFFER: row 0 and row 1 carry DIFFERENT words (and the
dense control different rows), so a broadcast defect that sprays row 0's word
across row 1 — or reads either row at the wrong stick — is a mismatch, never a
coincidental pass. The staged inputs state this (`intmr_inputs.py`).

FIDELITY: `a` is the dense tile at the sampler's own whole-row block shape
(`[1, PAD_V]` at `[pid, 0]` of a `[B, PAD_V]` view); the `Col` cell's `b` is the
seed word's own idiom verbatim (`sample_fwd.py`'s `s_desc`): a `[B, 1]` table
read through a `[1, 1]` tile at `[pid, 0]`; the dense control's `b` is a second
`[B, PAD_V]` view read the same way as `a`. The ops are `a + b` / `a * b`,
which Triton's implicit promotion spells as the collapse + `linalg.generic`
broadcast + `arith.addi`/`arith.muli` chain the whole-function walk resolves —
at `B = 2` the walk types the regions at the view's own `[2, ·]`, the band
height, exactly the multi-row form the door's in-flight admission names. No
constants, no splats, no f32 — pure i32, so `LegalizeTypes` has no island to
demand.

PARAMETERS, IN ORDER (every cell): `a` (`[B, PAD_V]` i32), `b` (the cell's own
shape — `[B, PAD_V]` dense, or `[B, 1]` at the SENUINT32 one-stick footprint of
32 words per row), `o` (the `[B, PAD_V]` i32 result). CONSTEXPRS: `PAD_V` (512,
the auditor's stated width), `B` (2). GRID: `[2]`.

THE VERDICT is a host bit-for-bit compare, per row and lane: every `o[r][j]`
must equal `(a[r][j] OP b[r])` — `b[r][j]` in the dense control, the row's own
word in the `Col` cell — wrapping i32, exact on both sides, so ANY mismatch is
an addressing defect in the multi-row staging or the `Col` operand mode, never
numerics.
"""

import triton
import triton.language as tl


@triton.jit
def intmr_dense_add_probe(a_ptr, b_ptr, o_ptr,  #
                          PAD_V: tl.constexpr, B: tl.constexpr):
    # THE CONTROL — two dense [B, PAD_V] views, one program per row.
    pid = tl.program_id(0)
    a_d = tl.make_tensor_descriptor(a_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    a = a_d.load([pid, 0])
    b_d = tl.make_tensor_descriptor(b_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    b = b_d.load([pid, 0])
    o = a + b
    o_d = tl.make_tensor_descriptor(o_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    o_d.store([pid, 0], o)


@triton.jit
def intmr_col_add_probe(a_ptr, b_ptr, o_ptr,  #
                        PAD_V: tl.constexpr, B: tl.constexpr):
    # THE WALL-9 FORM — a [B, 1] per-row word table against the [B, PAD_V] tile.
    pid = tl.program_id(0)
    a_d = tl.make_tensor_descriptor(a_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    a = a_d.load([pid, 0])
    # The production seed-word spelling verbatim (`sample_fwd.py`'s own
    # `s_desc`): a [B, 1] table read through a [1, 1] tile at this program's
    # own row, so the load is the seed word's own [1, 1] i32 shape.
    b_d = tl.make_tensor_descriptor(b_ptr, shape=[B, 1], strides=[1, 1],
                                    block_shape=[1, 1])
    b = b_d.load([pid, 0])
    o = a + b
    o_d = tl.make_tensor_descriptor(o_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    o_d.store([pid, 0], o)


@triton.jit
def intmr_dense_mul_probe(a_ptr, b_ptr, o_ptr,  #
                          PAD_V: tl.constexpr, B: tl.constexpr):
    pid = tl.program_id(0)
    a_d = tl.make_tensor_descriptor(a_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    a = a_d.load([pid, 0])
    b_d = tl.make_tensor_descriptor(b_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    b = b_d.load([pid, 0])
    o = a * b
    o_d = tl.make_tensor_descriptor(o_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    o_d.store([pid, 0], o)


@triton.jit
def intmr_col_mul_probe(a_ptr, b_ptr, o_ptr,  #
                        PAD_V: tl.constexpr, B: tl.constexpr):
    pid = tl.program_id(0)
    a_d = tl.make_tensor_descriptor(a_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    a = a_d.load([pid, 0])
    b_d = tl.make_tensor_descriptor(b_ptr, shape=[B, 1], strides=[1, 1],
                                    block_shape=[1, 1])
    b = b_d.load([pid, 0])
    o = a * b
    o_d = tl.make_tensor_descriptor(o_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    o_d.store([pid, 0], o)


@triton.jit
def intmr_f32col_add_probe(a_ptr, b_ptr, o_ptr,  #
                           PAD_V: tl.constexpr, B: tl.constexpr):
    # THE FP32 TWIN of `intmr_col_add_probe`, for the wall-9 emission diff: the
    # same shapes, the same pid-cornered spelling, the same operand order —
    # fp32 alone differs. This is `invt`'s own production class (the score add
    # `x·inv_t + g·gate` reads inv_t as a `[B, 1]` fp32 Col operand), the form
    # the auditor states is card-proven at rows > 1 (batched greedy's equality
    # rides the same Col addressing). Baking this cell beside the int cell and
    # diffing the two Col-operand TensorArgs field by field isolates the
    # emission defect: everything but the dataFormat must agree.
    pid = tl.program_id(0)
    a_d = tl.make_tensor_descriptor(a_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    a = a_d.load([pid, 0])
    b_d = tl.make_tensor_descriptor(b_ptr, shape=[B, 1], strides=[1, 1],
                                    block_shape=[1, 1])
    b = b_d.load([pid, 0])
    o = a + b
    o_d = tl.make_tensor_descriptor(o_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    o_d.store([pid, 0], o)

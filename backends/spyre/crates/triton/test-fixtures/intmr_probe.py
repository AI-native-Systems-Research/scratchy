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
  * `intmb_add` / `intmb_mul` (WALL 10, auditor GO 2026-10-27) — `[2, 512]`
    dense + `[1, 512]` `Mb`, the ROW-VECTOR form the 32-row ceiling rung's
    bake refusal named: the whole-row `lane` table against the band-typed
    tile, the hash's own `h = lane * hc1 + slo` shape. The `[2, 65536]`
    twins run the same kernels at production width.
  * `intmb_col_add` / `intmb_col_mul` — `[1, 512]` `Mb` + `[2, 1]` `Col`, the
    PRODUCTION form of the hash's multiply: `lane * hc1` has NO dense operand
    (the per-lane table sprays against the per-row word), so these cells
    measure the two operand modes TOGETHER — on top of the dense+Mb cells
    that isolate the spray alone.

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
def intmb_add_probe(a_ptr, b_ptr, o_ptr,  #
                    PAD_V: tl.constexpr, B: tl.constexpr):
    # THE WALL-10 FORM — the whole-row `[1, PAD_V]` table against the band-typed
    # `[B, PAD_V]` tile: `b` is the `lane` table's own idiom verbatim
    # (`sample_fwd.py`'s `la_desc`: a `[1, PAD_V]` view read at `[0, 0]`, the
    # same row every program reads), and `a` is the sampler's dense tile at the
    # pid-cornered spelling. This is the exact shape of the hash's
    # `h = lane * hc1 + slo` at every rung above one row: the per-lane entropy
    # is a program-INVARIANT whole-row table (no `linalg.broadcast` anywhere —
    # Triton needs none, the shapes already agree per program), so the `Mb`
    # operand mode can only come from the band's own derivation, which is the
    # production path this cell measures.
    pid = tl.program_id(0)
    a_d = tl.make_tensor_descriptor(a_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    a = a_d.load([pid, 0])
    b_d = tl.make_tensor_descriptor(b_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    b = b_d.load([0, 0])
    o = a + b
    o_d = tl.make_tensor_descriptor(o_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    o_d.store([pid, 0], o)


@triton.jit
def intmb_mul_probe(a_ptr, b_ptr, o_ptr,  #
                    PAD_V: tl.constexpr, B: tl.constexpr):
    # The MUL twin — the hash's own op (`lane * hc1`): the multiplication of the
    # per-lane table by a per-lane-varying dense tile, wrapping i32, exact on
    # both sides of the compare.
    pid = tl.program_id(0)
    a_d = tl.make_tensor_descriptor(a_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    a = a_d.load([pid, 0])
    b_d = tl.make_tensor_descriptor(b_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    b = b_d.load([0, 0])
    o = a * b
    o_d = tl.make_tensor_descriptor(o_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    o_d.store([pid, 0], o)


@triton.jit
def intmb_col_add_probe(a_ptr, b_ptr, o_ptr,  #
                        PAD_V: tl.constexpr, B: tl.constexpr):
    # THE PRODUCTION HASH FORM — `Mb × Col`, NO dense operand: `a` is the
    # whole-row `lane` table (`[1, PAD_V]` at `[0, 0]`), `b` is a per-row word
    # table (`[B, 1]` at `[pid, 0]`, the seed words' own idiom). This is the
    # exact shape of `h = lane * hc1 + slo`'s multiply at every rung above one
    # row: the per-lane table sprays down the rows (Mb) against the per-row
    # word (Col) — the only op in the sampler whose EVERY operand is a
    # broadcast, so it measures the two operand modes TOGETHER, on top of the
    # dense+Mb cells that isolate the spray alone.
    pid = tl.program_id(0)
    a_d = tl.make_tensor_descriptor(a_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    a = a_d.load([0, 0])
    b_d = tl.make_tensor_descriptor(b_ptr, shape=[B, 1], strides=[1, 1],
                                    block_shape=[1, 1])
    b = b_d.load([pid, 0])
    o = a + b
    o_d = tl.make_tensor_descriptor(o_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    o_d.store([pid, 0], o)


@triton.jit
def intmb_col_mul_probe(a_ptr, b_ptr, o_ptr,  #
                        PAD_V: tl.constexpr, B: tl.constexpr):
    # The MUL twin — the hash's own op verbatim (`lane * hc1`): the whole-row
    # table times the per-row word, wrapping i32.
    pid = tl.program_id(0)
    a_d = tl.make_tensor_descriptor(a_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    a = a_d.load([0, 0])
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


@triton.jit
def intmr_f32maxadd_probe(a_ptr, o_ptr,  #
                          PAD_V: tl.constexpr, B: tl.constexpr):
    # THE REDUCE-CHAIN TWIN (auditor order, wall 9): the sampler's own max
    # spelling verbatim — `m = tl.max(a, axis=1)`; `mm = tl.expand_dims(m, 1)`
    # — with the result consumed as the add's Col operand, the EXACT chain of
    # `max_o17` → `equal_o18` in the batched greedy bundle. The wide twin
    # proved the HOST-staged [B,1] Col read is row-blind (stick s reads word
    # (s mod rows) for every row); this cell measures the DEVICE-written
    # side: the reduce's [B,1] output is written by the card, and the question
    # is whether the same Col read picks each row's own max (self-consistent,
    # the greedy's safety) or the same row-blind word (the equality passes by
    # luck — urgent). The bake's max-op output TensorArg against the add-op's
    # Col input TensorArg is the write-layout vs read-layout statement,
    # readable from the emission alone.
    #
    # THE FPTOSI SINK: the sampler's own f32 selection survives LegalizeTypes'
    # RED-stop only inside the exactness island terminating at `arith.fptosi`
    # (`fp32toint32_o12` in the bundle); a bare f32 `tt.reduce` is refused
    # (FP32_CARD_BINDS carries no reduce row). So the probe stores the sum as
    # i32 through the same `.to(tl.int32)` the sampler's tail spells — the
    # island then protects the reduce and its combiner exactly as in
    # production. All staged words and sums are exact integers < 2^24, so the
    # conversion is exact and the i32 compare stays a pure addressing test.
    pid = tl.program_id(0)
    a_d = tl.make_tensor_descriptor(a_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    a = a_d.load([pid, 0])
    m = tl.max(a, axis=1)
    mm = tl.expand_dims(m, 1)
    o = a + mm
    o_d = tl.make_tensor_descriptor(o_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                    block_shape=[1, PAD_V])
    o_d.store([pid, 0], o.to(tl.int32))

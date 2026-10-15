# SPDX-License-Identifier: Apache-2.0
"""THE WINDOWED-RMSNORM SPLICE'S KERNELS — the head-norm sandwich fused: the
per-head rmsnorm `out[h·D..(h+1)·D] = rms(x[h·D..(h+1)·D]) · gamma` over the
`[m, heads·D]` activation, ONE program where the graph carried a
reshape → rmsnorm → reshape sandwich.

BODY PROVENANCE: main's `lower_sandwich_node`/`KtirFunc` windowed form — per
head `h`, the `[M, D]` window at column `h·D` runs `rmsnorm.py`'s own plain
chain (its module doc is the law — the reduce's accumulator is the card's
DLF16, which the emulator models), times the head's `[D]` gamma slice when
the norm is gained. The head loop is constexpr, so `to_ktir`
unrolls it: one rsqrt per head, every trip the same `EPS` — the shape
`program_windowed_eps` walks (all roots agree).

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about these
kernels):

* PARAMETERS, IN ORDER: `windowedrmsnorm_gained_fwd` takes
  `desc_x, desc_gamma, desc_o`; `windowedrmsnorm_unit_fwd` takes `desc_x,
  desc_o`. ⭐ THE REGISTRY BINDS THE NODE'S OWN OPERAND ORDER, and the gained and
  unit sandwich nodes carry different operand lists — two entries, so neither
  program carries a parameter its node does not state.
* CONSTEXPRS: `M`, `D`, `HEADS`, `EPS`, `INV_D` — stated by the splice from the
  sandwich's own geometry (`M =` the activation's rows, `D =` the head window,
  `HEADS =` `full / D` over the activation's width, `EPS =` the tape's epsilon,
  `INV_D = 1/D` folded on the host).
* GRID: `[1]`.

THE DOOR'S OWN FACTS (`ktir_superdsc::windowed_rmsnorm`): `m =` the output's
rows, `full =` the output view's `v_cols`, `D =` the FIRST access tile's width
(one head's window), `heads = full / D` — so the descriptors state the FULL
`[M, HEADS·D]` storage with a `[M, D]` block at each head's column corner. A
nonzero GAIN OFFSET is stated to the door the same way the epsilon is: the
program's gamma leg adds it over the load, and the door resolves the value's
`[1,1]` registry slot and emits the add ahead of the gain multiply.
"""

import triton
import triton.language as tl


@triton.jit
def windowedrmsnorm_gained_fwd(desc_x, desc_gamma, desc_o,  #
                               M: tl.constexpr, D: tl.constexpr,  #
                               HEADS: tl.constexpr,  #
                               EPS: tl.constexpr,  #
                               INV_D: tl.constexpr,  #
                               ):
    full = HEADS * D
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, full], strides=[full, 1],
                                       block_shape=[M, D])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, full], strides=[full, 1],
                                        block_shape=[M, D])
    # HF's weight is 1D, so the gamma descriptor is too — `rmsnorm.py`'s own law.
    g_desc = tl.make_tensor_descriptor(desc_gamma, shape=[D], strides=[1],
                                        block_shape=[D])
    w = g_desc.load([0])
    for h in tl.range(0, HEADS, 1):
        offs_c = h * D
        x = x_desc.load([0, offs_c])
        # EXX2_ZEROMEAN over the head's window — rmsnorm.py's own plain chain
        # (its module doc's law).
        ms = tl.sum(x * x, 1) * INV_D
        r = tl.rsqrt((ms + EPS).to(tl.float32)).to(tl.float16)
        o_desc.store([0, offs_c], x * r[:, None] * w[None, :])


@triton.jit
def windowedrmsnorm_unit_fwd(desc_x, desc_o,  #
                             M: tl.constexpr, D: tl.constexpr,  #
                             HEADS: tl.constexpr,  #
                             EPS: tl.constexpr,  #
                             INV_D: tl.constexpr,  #
                             ):
    full = HEADS * D
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, full], strides=[full, 1],
                                       block_shape=[M, D])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, full], strides=[full, 1],
                                        block_shape=[M, D])
    for h in tl.range(0, HEADS, 1):
        offs_c = h * D
        x = x_desc.load([0, offs_c])
        ms = tl.sum(x * x, 1) * INV_D
        r = tl.rsqrt((ms + EPS).to(tl.float32)).to(tl.float16)
        o_desc.store([0, offs_c], x * r[:, None])


@triton.jit
def windowedrmsnorm_oneplus_fwd(desc_x, desc_gamma, desc_o,  #
                                M: tl.constexpr, D: tl.constexpr,  #
                                HEADS: tl.constexpr,  #
                                EPS: tl.constexpr,  #
                                INV_D: tl.constexpr,  #
                                GAIN_OFFSET: tl.constexpr,  #
                                ):
    # The `(1 + w)` convention's windowed form — the gained entry's own body with
    # the gain leg adding the offset over the load (`rmsnorm_oneplus_fwd`'s law,
    # stated once here because the gamma row is shared by every head).
    full = HEADS * D
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, full], strides=[full, 1],
                                        block_shape=[M, D])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, full], strides=[full, 1],
                                        block_shape=[M, D])
    g_desc = tl.make_tensor_descriptor(desc_gamma, shape=[D], strides=[1],
                                        block_shape=[D])
    w = g_desc.load([0]) + GAIN_OFFSET
    for h in tl.range(0, HEADS, 1):
        offs_c = h * D
        x = x_desc.load([0, offs_c])
        ms = tl.sum(x * x, 1) * INV_D
        r = tl.rsqrt((ms + EPS).to(tl.float32)).to(tl.float16)
        o_desc.store([0, offs_c], x * r[:, None] * w[None, :])

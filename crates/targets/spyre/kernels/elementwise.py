# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton elementwise kernels — Add / Mul / Sub / Silu / Gelu.

BODY PROVENANCE: derived from `crates/targets/spyre/kernels/silumul.py`'s form (which
is `crates/triton/test-fixtures/swiglu_mlp.py`'s activation body stated standalone) —
descriptor-loaded whole-region tiles, one arithmetic chain, one store. `SubOp::SiluMul`
is the FUSED gate-up form; these entries are the SPLIT `SubOp::Elementwise` kinds.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about these kernels):

* ONE ENTRY PER `EwKind` the total registry covers: `add_fwd` (Add AND BiasAdd at
  decode — both map to `Elementwise::Add` and the same `add_s{id}` name), `mul_fwd`
  (Mul), `sub_fwd` (Sub), `silu_fwd` (Silu), `gelu_fwd` (Gelu — the tanh form, the
  one `"gelu"` the DDL primitive approximates). The kinds the DEVICE has no primitive
  for (QuickGelu, GeluErf) have no entry by construction: the consumer's own
  `elementwise_op_func` refuses them by name, so no kernel spelling could lower.
* PARAMETERS, IN ORDER: the node's operand order then the output. Binary:
  `desc_a`, `desc_b`, `desc_o`; unary: `desc_x`, `desc_o`. The registry does not
  permute.
* CONSTEXPRS: `M`, `N`, `BLOCK_M`, `BLOCK_N`, `N_TOTAL`, `C_START`, `N_BLOCKS`,
  `TAIL_H` — stated by the splice from the node's own region (`M =` rows, `N =
  BLOCK_N =` the region's width), the same whole-region single-tile law the builder
  states when the region fits a core's LX, PLUS the STORAGE the region windows
  (`N_TOTAL =` the tensor's full width, `C_START =` the region's column corner). A
  whole-region node states `N_TOTAL = N`, `C_START = 0`. See `scalarmul.py`'s window
  law: the descriptor names the storage, the load names the window.
* GRID: `[1]`.

⭐ ROW BLOCKING, THE BUILDER'S OWN LAW (`lower_elementwise_node`): a whole
`[rows, cols]` region whose live set (`rows × cols × live-tiles`) exceeds the
`EW_LX_ELEMS` budget is ROW-BLOCKED, not refused — `N_BLOCKS` full `[BLOCK_M, N]`
tiles (BLOCK_M = the builder's own `rows_per_block(cols, live)`) plus a `[TAIL_H, N]`
tail tile when the region does not divide evenly. The trip count is constexpr, so
`to_ktir` unrolls it and the spliced program is straight-line with one
constant-corner tile per block — the door folds those through `Region::r_cover` and
ONE descriptor spans all of them, exactly as it spans the builder's row-blocked
program (`lower_elementwise_node_rows`).
"""

import triton
import triton.language as tl


@triton.jit
def add_fwd(desc_a, desc_b, desc_o,  #
            M: tl.constexpr, N: tl.constexpr,  #
            BLOCK_M: tl.constexpr,  #
            BLOCK_N: tl.constexpr,  #
            N_TOTAL: tl.constexpr,  #
            C_START: tl.constexpr,  #
            N_BLOCKS: tl.constexpr,  #
            TAIL_H: tl.constexpr,  #
            ):
    a_desc = tl.make_tensor_descriptor(desc_a, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    b_desc = tl.make_tensor_descriptor(desc_b, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    for blk in tl.range(0, N_BLOCKS, 1):
        offs_m = blk * BLOCK_M
        a = a_desc.load([offs_m, C_START])
        b = b_desc.load([offs_m, C_START])
        o_desc.store([offs_m, C_START], a + b)
    if TAIL_H > 0:
        t_a = tl.make_tensor_descriptor(desc_a, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        t_b = tl.make_tensor_descriptor(desc_b, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        t_o = tl.make_tensor_descriptor(desc_o, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        offs_m = N_BLOCKS * BLOCK_M
        a = t_a.load([offs_m, C_START])
        b = t_b.load([offs_m, C_START])
        t_o.store([offs_m, C_START], a + b)


@triton.jit
def mul_fwd(desc_a, desc_b, desc_o,  #
            M: tl.constexpr, N: tl.constexpr,  #
            BLOCK_M: tl.constexpr,  #
            BLOCK_N: tl.constexpr,  #
            N_TOTAL: tl.constexpr,  #
            C_START: tl.constexpr,  #
            N_BLOCKS: tl.constexpr,  #
            TAIL_H: tl.constexpr,  #
            ):
    a_desc = tl.make_tensor_descriptor(desc_a, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    b_desc = tl.make_tensor_descriptor(desc_b, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    for blk in tl.range(0, N_BLOCKS, 1):
        offs_m = blk * BLOCK_M
        a = a_desc.load([offs_m, C_START])
        b = b_desc.load([offs_m, C_START])
        o_desc.store([offs_m, C_START], a * b)
    if TAIL_H > 0:
        t_a = tl.make_tensor_descriptor(desc_a, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        t_b = tl.make_tensor_descriptor(desc_b, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        t_o = tl.make_tensor_descriptor(desc_o, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        offs_m = N_BLOCKS * BLOCK_M
        a = t_a.load([offs_m, C_START])
        b = t_b.load([offs_m, C_START])
        t_o.store([offs_m, C_START], a * b)


@triton.jit
def sub_fwd(desc_a, desc_b, desc_o,  #
            M: tl.constexpr, N: tl.constexpr,  #
            BLOCK_M: tl.constexpr,  #
            BLOCK_N: tl.constexpr,  #
            N_TOTAL: tl.constexpr,  #
            C_START: tl.constexpr,  #
            N_BLOCKS: tl.constexpr,  #
            TAIL_H: tl.constexpr,  #
            ):
    a_desc = tl.make_tensor_descriptor(desc_a, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    b_desc = tl.make_tensor_descriptor(desc_b, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    for blk in tl.range(0, N_BLOCKS, 1):
        offs_m = blk * BLOCK_M
        a = a_desc.load([offs_m, C_START])
        b = b_desc.load([offs_m, C_START])
        o_desc.store([offs_m, C_START], a - b)
    if TAIL_H > 0:
        t_a = tl.make_tensor_descriptor(desc_a, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        t_b = tl.make_tensor_descriptor(desc_b, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        t_o = tl.make_tensor_descriptor(desc_o, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        offs_m = N_BLOCKS * BLOCK_M
        a = t_a.load([offs_m, C_START])
        b = t_b.load([offs_m, C_START])
        t_o.store([offs_m, C_START], a - b)


@triton.jit
def silu_fwd(desc_x, desc_o,  #
             M: tl.constexpr, N: tl.constexpr,  #
             BLOCK_M: tl.constexpr,  #
             BLOCK_N: tl.constexpr,  #
             N_TOTAL: tl.constexpr,  #
             C_START: tl.constexpr,  #
             N_BLOCKS: tl.constexpr,  #
             TAIL_H: tl.constexpr,  #
             ):
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    for blk in tl.range(0, N_BLOCKS, 1):
        offs_m = blk * BLOCK_M
        x = x_desc.load([offs_m, C_START])
        # silu(x) = x / (1 + exp(-x)) — the same chain the builder's own arm writes
        # (`negate → math.exp → splat 1 → addf → divf`) and `silumul.py` spells for the
        # fused form: f16 in and out, the f32 island only around `exp` (the ladder's
        # LegalizeTypes collapses the widen/truncate), `tl.fdiv` not `/` (`/` upcasts
        # f16→f32).
        e = tl.exp((-x).to(tl.float32)).to(tl.float16)
        s = tl.fdiv(x, 1.0 + e)
        o_desc.store([offs_m, C_START], s)
    if TAIL_H > 0:
        t_x = tl.make_tensor_descriptor(desc_x, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        t_o = tl.make_tensor_descriptor(desc_o, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        offs_m = N_BLOCKS * BLOCK_M
        x = t_x.load([offs_m, C_START])
        e = tl.exp((-x).to(tl.float32)).to(tl.float16)
        s = tl.fdiv(x, 1.0 + e)
        t_o.store([offs_m, C_START], s)


@triton.jit
def gelu_fwd(desc_x, desc_o,  #
             M: tl.constexpr, N: tl.constexpr,  #
             BLOCK_M: tl.constexpr,  #
             BLOCK_N: tl.constexpr,  #
             N_TOTAL: tl.constexpr,  #
             C_START: tl.constexpr,  #
             N_BLOCKS: tl.constexpr,  #
             TAIL_H: tl.constexpr,  #
             ):
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    for blk in tl.range(0, N_BLOCKS, 1):
        offs_m = blk * BLOCK_M
        x = x_desc.load([offs_m, C_START])
        # The TANH approximation, the reference the device's `gelu` DDL primitive
        # approximates (`eval_dag`'s own comment): 0.5x(1 + tanh(v)),
        # v = sqrt(2/pi)(x + 0.044715 x^3). The frontend has no `tanh` (its tl.math
        # surface is exp/exp2/rsqrt only), so the tanh spells through the sigmoid
        # identity 1 + tanh(v) = 2/(1 + exp(-2v)) — giving gelu(x) = x/(1+exp(-2v)) —
        # the same exp-island shape `silumul.py` spells. The cubic is computed in the
        # f32 island: an f16 x^3 overflows at |x| >= 41, and a gemm output outlier is
        # enough to reach it.
        x32 = x.to(tl.float32)
        v = 1.5957691216057308 * (x32 + 0.044715 * x32 * x32 * x32)  # 2*sqrt(2/pi)*(...)
        e = tl.exp(-v).to(tl.float16)
        g = tl.fdiv(x, 1.0 + e)
        o_desc.store([offs_m, C_START], g)
    if TAIL_H > 0:
        t_x = tl.make_tensor_descriptor(desc_x, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        t_o = tl.make_tensor_descriptor(desc_o, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        offs_m = N_BLOCKS * BLOCK_M
        x = t_x.load([offs_m, C_START])
        x32 = x.to(tl.float32)
        v = 1.5957691216057308 * (x32 + 0.044715 * x32 * x32 * x32)
        e = tl.exp(-v).to(tl.float16)
        g = tl.fdiv(x, 1.0 + e)
        t_o.store([offs_m, C_START], g)

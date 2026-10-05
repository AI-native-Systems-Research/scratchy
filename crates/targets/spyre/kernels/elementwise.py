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
* CONSTEXPRS: `M`, `N`, `BLOCK_M`, `BLOCK_N` — stated by the splice from the node's
  own region (`M = BLOCK_M =` rows, `N = BLOCK_N =` cols), the same whole-region
  single-tile law the builder states when the region fits a core's LX.
* GRID: `[1]`.

⛔ NO ROW BLOCKING, and the SPLICE refuses the node loudly (a named Err naming the
missing block-shape kernel) when the builder would block: a whole `[rows, cols]`
region whose live set (`rows × cols × live-tiles`) exceeds the builder's own
`EW_LX_ELEMS` budget is a shape a one-tile kernel cannot spell. The splice mirrors
the builder's own guard so the two paths never disagree about which of them takes
the node.
"""

import triton
import triton.language as tl


@triton.jit
def add_fwd(desc_a, desc_b, desc_o,  #
            M: tl.constexpr, N: tl.constexpr,  #
            BLOCK_M: tl.constexpr,  #
            BLOCK_N: tl.constexpr,  #
            ):
    start_m = tl.program_id(0)
    a_desc = tl.make_tensor_descriptor(desc_a, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    b_desc = tl.make_tensor_descriptor(desc_b, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    offs_m = start_m * BLOCK_M
    a = a_desc.load([offs_m, 0])
    b = b_desc.load([offs_m, 0])
    o_desc.store([offs_m, 0], a + b)


@triton.jit
def mul_fwd(desc_a, desc_b, desc_o,  #
            M: tl.constexpr, N: tl.constexpr,  #
            BLOCK_M: tl.constexpr,  #
            BLOCK_N: tl.constexpr,  #
            ):
    start_m = tl.program_id(0)
    a_desc = tl.make_tensor_descriptor(desc_a, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    b_desc = tl.make_tensor_descriptor(desc_b, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    offs_m = start_m * BLOCK_M
    a = a_desc.load([offs_m, 0])
    b = b_desc.load([offs_m, 0])
    o_desc.store([offs_m, 0], a * b)


@triton.jit
def sub_fwd(desc_a, desc_b, desc_o,  #
            M: tl.constexpr, N: tl.constexpr,  #
            BLOCK_M: tl.constexpr,  #
            BLOCK_N: tl.constexpr,  #
            ):
    start_m = tl.program_id(0)
    a_desc = tl.make_tensor_descriptor(desc_a, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    b_desc = tl.make_tensor_descriptor(desc_b, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    offs_m = start_m * BLOCK_M
    a = a_desc.load([offs_m, 0])
    b = b_desc.load([offs_m, 0])
    o_desc.store([offs_m, 0], a - b)


@triton.jit
def silu_fwd(desc_x, desc_o,  #
             M: tl.constexpr, N: tl.constexpr,  #
             BLOCK_M: tl.constexpr,  #
             BLOCK_N: tl.constexpr,  #
             ):
    start_m = tl.program_id(0)
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    offs_m = start_m * BLOCK_M
    x = x_desc.load([offs_m, 0])
    # silu(x) = x / (1 + exp(-x)) — the same chain the builder's own arm writes
    # (`negate → math.exp → splat 1 → addf → divf`) and `silumul.py` spells for the
    # fused form: f16 in and out, the f32 island only around `exp` (the ladder's
    # LegalizeTypes collapses the widen/truncate), `tl.fdiv` not `/` (`/` upcasts
    # f16→f32).
    e = tl.exp((-x).to(tl.float32)).to(tl.float16)
    s = tl.fdiv(x, 1.0 + e)
    o_desc.store([offs_m, 0], s)


@triton.jit
def gelu_fwd(desc_x, desc_o,  #
             M: tl.constexpr, N: tl.constexpr,  #
             BLOCK_M: tl.constexpr,  #
             BLOCK_N: tl.constexpr,  #
             ):
    start_m = tl.program_id(0)
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    offs_m = start_m * BLOCK_M
    x = x_desc.load([offs_m, 0])
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
    o_desc.store([offs_m, 0], g)

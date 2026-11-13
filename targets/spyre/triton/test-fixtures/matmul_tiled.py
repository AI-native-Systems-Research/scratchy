# SPDX-License-Identifier: Apache-2.0
"""The grid-corner uniform twin: a tiled matmul, `o = a @ b`, one row band per program.

`mul_scale` is the address-provenance ladder's rung-2 twin in ONE dimension (`offset =
pid * BLOCK`, a 1-D window); this is the same rung in TWO -- the ROW corner of every
window is the per-core uniform `pid * BLOCK_M`, and the kernel exists to exercise the
family end to end: `uniform_grid_corners` (triton-ktir) rewrites the corner into the
canonical spelling, the fire gate exempts it, the launch keeps its grid, and the
whole-function door's uniform arm resolves the window at the buffer base -- no clone
protocol, no unroll. The geometry is chosen so the contract the door checks holds
EXACTLY: `M / BLOCK_M` programs, `stride == BLOCK_M == r_len`, `positions * stride ==
M == v_rows` -- the grid tiles the view, one row band per core.

The K axis is a single tile (`K == BLOCK_K`), deliberately: the acceptance is the
CORNER family, not the loop family, and a standing sweep (causal attention's shape)
is the sit-out condition by design. A multi-tile K would add a constant-trip loop
that `unroll_constant_trip_loops` splices before the corner pass runs, exercising
nothing this kernel does not.

The form is `swiglu_mlp.py`'s, reduced: descriptors not pointer blocks (its delta 3 --
the masked pointer form is refused by `spyre-dot-to-linalg`), f16 accumulators (its
delta 2 -- the device MAC rounds to DL16 at every step, so there is no wide
accumulator to keep), B n-major with `.T` at the load (its deltas 11 and 13), and
shapes as constexpr (its delta 4 -- one kernel per shape).
"""

import torch

import triton
import triton.language as tl

M = 512
N = 64
K = 64
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64
DTYPE = torch.float16


@triton.jit
def matmul_tiled_kernel(a_ptr, b_ptr, o_ptr,
                        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    # The rung-2 corner: each program's row band, a per-core uniform.
    start_m = pid * BLOCK_M
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1],
                                       block_shape=[BLOCK_M, BLOCK_K])
    # B n-major (`[N, K]` row-major), transposed at the load like `swiglu_mlp`'s
    # weights -- its row corner is the constant 0, shared by every program.
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K], strides=[K, 1],
                                       block_shape=[BLOCK_N, BLOCK_K])
    o_desc = tl.make_tensor_descriptor(o_ptr, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    # f16 accumulator, seeded by the caller like every shipped matmul here.
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)
    a = a_desc.load([start_m, 0])
    b = b_desc.load([0, 0]).T
    acc = tl.dot(a, b, acc)
    o_desc.store([start_m, 0], acc)


def inputs(seed: int = 0):
    """Randomized fp16 inputs (the pod gate runs fp16)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(M, K, generator=g, dtype=DTYPE)
    b = torch.randn(N, K, generator=g, dtype=DTYPE)
    return a, b


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Native PyTorch reference: o = a @ b.T."""
    return a @ b.T


SIGNATURE = {
    "a_ptr": "*fp16", "b_ptr": "*fp16", "o_ptr": "*fp16",
    "M": "constexpr", "N": "constexpr", "K": "constexpr",
    "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr",
}


def constexprs(m=M, n=N, k=K, bm=BLOCK_M, bn=BLOCK_N, bk=BLOCK_K):
    """One configuration: `m / bm` programs, each one `[bm, k]` row band of `a`."""
    return {"M": m, "N": n, "K": k, "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}

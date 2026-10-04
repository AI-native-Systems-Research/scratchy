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

"""SP-E2-07 fixture: bias_add_f32 -- the fp32-input kernel that forces
LegalizeTypes to do real work.

The kernel loads f16 operands, then does a fp32 stability-widened accumulate
with an fp32 splat bias (Triton's `tl.float32` upcast idiom): the f16 inputs are
widened to f32, the add + bias run in f32, and the result is truncated back to
f16 on store. That produces the arith.extf / f32 compute / arith.truncf and the
splat dense<> f32 constant in TTIR that:
  - LegalizeTypes strips/retypes (f32 -> f16, drop extf/truncf), and
  - DecomposeDenseConstants splits the (now-f16) splat bias.
Unlike vector_add/mul (pure f16), this fixture exercises the full four-pass
pipeline including non-trivial type legalization and constant decomposition.

`reference()` is a *native* PyTorch bias-add computed in fp32 then cast to fp16
-- the same numeric domain the widened kernel targets, NOT a re-implementation
of the kernel's tiling.
"""

import torch

import triton
import triton.language as tl

N = 1024
BLOCK = 64
BIAS = 1.0
DTYPE = torch.float16


@triton.jit
def bias_add_f32_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[n], strides=[1],
                                       block_shape=[BLOCK])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[n], strides=[1],
                                       block_shape=[BLOCK])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[n], strides=[1],
                                       block_shape=[BLOCK])
    # Load f16, widen to f32 for the accumulate (Triton stability widening).
    a = a_desc.load([offset]).to(tl.float32)
    b = b_desc.load([offset]).to(tl.float32)
    # Splat f32 bias constant (forces DecomposeDenseConstants).
    bias = tl.full([BLOCK], BIAS, tl.float32)
    c = a + b + bias
    # Truncate back to f16 on store.
    c_desc.store([offset], c.to(tl.float16))


def inputs(seed: int = 0):
    """Randomized fp16 inputs (the pod gate runs fp16)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(N, generator=g, dtype=DTYPE)
    b = torch.randn(N, generator=g, dtype=DTYPE)
    return a, b


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Native PyTorch reference: c = (a + b + bias), accumulated in fp32 (the
    widened domain) then cast to fp16."""
    acc = a.to(torch.float32) + b.to(torch.float32) + BIAS
    return acc.to(torch.float16)

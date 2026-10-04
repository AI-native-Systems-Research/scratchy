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

"""SP-E2-07 fixture: mul (elementwise c = a * b, f16 buffers).

Same descriptor + program_id shape as vector_add, but the compute is a multiply
-- a second elementwise op so the gate is not single-op. f16 buffers, so
LegalizeTypes is a no-op (see bias_add_f32 for real type work).

`reference()` is a native PyTorch multiply.
"""

import torch

import triton
import triton.language as tl

N = 1024
BLOCK = 64
DTYPE = torch.float16


@triton.jit
def mul_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[n], strides=[1],
                                       block_shape=[BLOCK])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[n], strides=[1],
                                       block_shape=[BLOCK])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[n], strides=[1],
                                       block_shape=[BLOCK])
    a = a_desc.load([offset])
    b = b_desc.load([offset])
    c_desc.store([offset], a * b)


def inputs(seed: int = 0):
    """Randomized fp16 inputs (the pod gate runs fp16)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(N, generator=g, dtype=DTYPE)
    b = torch.randn(N, generator=g, dtype=DTYPE)
    return a, b


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Native PyTorch reference: c = a * b."""
    return a * b

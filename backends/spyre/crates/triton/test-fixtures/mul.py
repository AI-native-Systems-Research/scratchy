# SPDX-License-Identifier: Apache-2.0
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

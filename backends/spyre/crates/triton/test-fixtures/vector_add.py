# SPDX-License-Identifier: Apache-2.0
"""SP-E2-07 fixture: vector_add (elementwise c = a + b, f16 buffers).

The descriptor-based kernel is the TTIR the Phase-2 passes lower:
- tl.make_tensor_descriptor + desc.load/store -> ConvertTTIRToKTDP
- tl.program_id                               -> DistributeWork
(f16 buffers -> LegalizeTypes is a no-op here; see bias_add_f32 for the kernel
that forces real type legalization.)

`reference()` is a *native* PyTorch add -- not a re-implementation of the
kernel's blocking. The deferred pod numeric gate compares the pod's emitted
output against this.
"""

import torch

import triton
import triton.language as tl

N = 1024
BLOCK = 64
DTYPE = torch.float16


@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
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
    c_desc.store([offset], a + b)


def inputs(seed: int = 0):
    """Randomized fp16 inputs (the pod gate runs fp16)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(N, generator=g, dtype=DTYPE)
    b = torch.randn(N, generator=g, dtype=DTYPE)
    return a, b


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Native PyTorch reference: c = a + b."""
    return a + b

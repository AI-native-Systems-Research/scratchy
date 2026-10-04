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

"""SP-E2-07 rung-3 twin: mul with a RUNTIME scale (c = a * scale, f16 buffers).

`mul` is the smallest elementwise kernel in the set; this is the SAME kernel
with the second operand replaced by a runtime fp32 scalar argument -- the
smallest thing that can exercise the address-provenance ladder's rung 3
(address-provenance ladder: rung 1 compile-time constant, rung 2 per-core
uniform, rung 3 runtime scalar from the launcher, rung 4 index read from
device memory). `mul_scale: tl.constexpr` would be rung 1; the argument
here is NOT constexpr, so the value must cross the whole chain as a launch
binding.

`reference()` is a native PyTorch multiply by the same scalar.
"""

import torch

import triton
import triton.language as tl

N = 1024
BLOCK = 64
DTYPE = torch.float16


@triton.jit
def mul_scale_kernel(a_ptr, c_ptr, scale, n, BLOCK: tl.constexpr):
    # `scale` arrives as f16 (SIGNATURE "fp16"), NOT f32, and that is a recorded
    # owner decision, not a convenience: an f32 scalar against an f16 tile promotes
    # the whole multiply to genuine f32 (`extf`/`truncf` around it in the ttir),
    # which LegalizeTypes refuses by design (see attention_sdpa.py's own note --
    # "declaring it f16 is cheaper than casting in-kernel"). It also matches the
    # device's scalarmul scale slots, which are [1,1] fp16.
    pid = tl.program_id(0)
    offset = pid * BLOCK
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[n], strides=[1],
                                       block_shape=[BLOCK])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[n], strides=[1],
                                       block_shape=[BLOCK])
    a = a_desc.load([offset])
    c_desc.store([offset], a * scale)


def inputs(seed: int = 0):
    """Randomized fp16 inputs (the pod gate runs fp16)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(N, generator=g, dtype=DTYPE)
    return a


def reference(a: torch.Tensor, scale: float) -> torch.Tensor:
    """Native PyTorch reference: c = a * scale."""
    return a * scale


SIGNATURE = {
    "a_ptr": "*fp16", "c_ptr": "*fp16", "scale": "fp16",
    "n": "constexpr", "BLOCK": "constexpr",
}


def constexprs(n=N, block=BLOCK):
    """One configuration. Shapes are constexpr (delta 2), so a shape change
    is a recompile."""
    return {"n": n, "BLOCK": block}

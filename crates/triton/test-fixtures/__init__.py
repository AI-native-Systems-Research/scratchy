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

"""SP-E2-07 Phase-2 fixtures: annotated @triton.jit kernels + native-PyTorch
reference functions.

These are build-independent: each module pairs a descriptor-based Triton kernel
(the TTIR the Phase-2 TTIR->KTIR passes consume) with a *native* PyTorch
reference (plain torch ops, NOT a re-implementation of the kernel's tiling).
The structural gate (`make verify-phase2`) runs the lit suite; the DEFERRED pod
numeric gate (`verify-phase2-numeric`, SP-E2-07 second half) will reuse the
`reference()` functions here to compare `torch.allclose(pod_run, reference)`.

Fixtures:
- vector_add  : elementwise c = a + b
- mul         : elementwise c = a * b
- bias_add_f32 : fp32 inputs + fp32 stability-widened accumulate, the kernel
                 that forces LegalizeTypes to do real f32->f16 work.
"""

FIXTURES = ("vector_add", "mul", "bias_add_f32")

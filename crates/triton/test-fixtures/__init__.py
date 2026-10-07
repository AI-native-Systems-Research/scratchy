# SPDX-License-Identifier: Apache-2.0
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

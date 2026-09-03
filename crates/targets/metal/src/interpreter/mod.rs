// SPDX-License-Identifier: Apache-2.0
//! Backend-specific interpreters for `Instruction<W>` tapes.
//!
//! - **CUDA** (in `scratchy_target_cuda::eval`): direct kernel dispatch
//!   via `Instruction::eval()` walking the tape per forward.
//! - **Metal** (this module): lowering pass + worker pool. The
//!   `Instruction<W>` tape is translated once into a buffer-pointer-free
//!   `LoweredMetalTape` per (model variant, bucket); a `MetalWorkerPool`
//!   bakes that lowered tape into a per-worker arena + dispatch and runs
//!   the MTL4 compute encoder per forward. Phase 5.A lands the lowering
//!   pass; subsequent phases land specialized pipelines (5.B), the
//!   worker (5.C), the pool (5.D), `forward()` (5.E), macro emission
//!   (5.F), and the cpu_golden / scratchy-e2e wiring (5.G).

pub mod metal;

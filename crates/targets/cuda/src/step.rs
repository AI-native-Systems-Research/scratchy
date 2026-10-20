// SPDX-License-Identifier: Apache-2.0
//! CUDA's step: one kernel launch of a lowered tape, the counterpart of metal's `MetalStep`.

/// One launch-emitting step of a lowered tape.
///
/// UNINHABITED UNTIL AN OPCODE LOWERS. It gains one variant per opcode CUDA lowers, each added by
/// the issue that writes that opcode's kernel; until then the compiler's opcode lowering refuses
/// every op, and no tape of this type can be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaStep {}

// SPDX-License-Identifier: Apache-2.0
//! THE PROGRAM IR'S OWN BOUNDS — `sys-arch-spec/progir/progir.h`.
//!
//! `ProgramAndStateInfo` fixes how much of a unit a program may occupy, and both the compiler that fills those
//! arrays and the model that executes them are bounded by the same two numbers.
//!
//! ⛔ THEY WERE WRITTEN OUT TWICE, VERBATIM — `256` and `128` in the compiler's island 4 and again in the model's
//! `prog_ir_graph`, with nothing tying the two together. A bound held in two places is a bound that can be raised
//! in one of them, and the symptom would be a compiler emitting programs the model silently truncates.

/// HOW MANY INSTRUCTIONS ANY ONE UNIT MAY HOLD — `kMaxCompIBuff` (`progir.h:507-508`).
pub const MAX_INSTRUCTIONS_PER_UNIT: usize = 256;

/// HOW MANY REGISTERS ANY ONE UNIT MAY INITIALISE — `kMaxCompRegs` (`progir.h:509-510`).
pub const MAX_REGISTERS_PER_UNIT: usize = 128;

/// Both bounds index a fixed-width array, so both are powers of two.
const _: () = assert!(
    MAX_INSTRUCTIONS_PER_UNIT.is_power_of_two() && MAX_REGISTERS_PER_UNIT.is_power_of_two(),
    "a `ProgramAndStateInfo` bound indexes a fixed-width array"
);

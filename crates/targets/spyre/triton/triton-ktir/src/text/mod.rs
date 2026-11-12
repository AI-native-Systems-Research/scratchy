// SPDX-License-Identifier: Apache-2.0

//! TEST INSTRUMENTS. Nothing in `crate::passes` may use this module.
//!
//! Text exists here for exactly two jobs, and both are measurement:
//!
//! * [`parse`] turns a golden `.mlir` file into an input VALUE, so bridge two can
//!   be built and tested before bridge one (Python -> ttir) exists. When bridge
//!   one lands, its output replaces this and the parser keeps its test-only role.
//! * [`print`] turns a KTIR value back into text so it can be diffed against what
//!   the C++ toolchain emits. Without a printer the golden diff has nothing to
//!   compare and the port has no oracle.
//! * [`diff`] is the comparison itself -- structural, field by field, and with a
//!   planted-difference control so a diff that compares nothing cannot pass.
//!
//! WHY THIS IS NOT A PIPELINE STAGE. scratchy's convention, quoted in the owner's
//! brief: "NOTHING IS SERIALIZED AND NOTHING IS PARSED... There is no text form of
//! a program at any point." A text boundary inside a compiler is a place where two
//! stages agree by accident; the value boundary makes them agree by type.

pub mod diff;
pub mod parse;
pub mod print;

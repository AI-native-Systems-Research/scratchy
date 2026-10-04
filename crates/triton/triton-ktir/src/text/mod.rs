// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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

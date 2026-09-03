// SPDX-License-Identifier: Apache-2.0
//! MOVED to `crate::tape::lowered` — the tape-construction
//! layer is neutral data the `#[forward]` macro constructs at expansion;
//! this shim keeps existing paths compiling during the transition.
pub use crate::tape::lowered::*;

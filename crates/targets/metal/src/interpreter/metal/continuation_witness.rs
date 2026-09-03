// SPDX-License-Identifier: Apache-2.0
//! MOVED to `crate::tape::continuation_witness` — the tape-construction
//! layer is neutral data the `#[forward]` macro constructs at expansion;
//! this shim keeps existing paths compiling during the transition.
pub use crate::tape::continuation_witness::*;

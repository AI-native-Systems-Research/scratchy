// SPDX-License-Identifier: Apache-2.0
//! Hand-written CUDA model forwards have been removed; scratchy-forward-compiler is
//! the sole model-forward path. This module survives only as a re-export
//! shim for `attention_helpers`, reached via `crate::model::attention_helpers`.

pub use crate::attention_helpers;

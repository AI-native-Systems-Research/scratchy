// SPDX-License-Identifier: Apache-2.0
//! Gated-DeltaNet's backend-neutral recurrent state, shared by Mamba-2.
//!
//! A GDN layer keeps a causal-conv ring and a recurrent ssm state per resident sequence, and
//! Mamba-2 runs on the same pool, so this crate serves both. It names no backend: the kernels, the
//! lowering and the snapshot traffic stay in each target.

pub mod geometry;
pub mod layout;
pub mod pool;
pub mod reference;
pub mod slots;

pub use pool::{GDN_CONV_AND_RECORD_DTYPE, GdnStatePool};

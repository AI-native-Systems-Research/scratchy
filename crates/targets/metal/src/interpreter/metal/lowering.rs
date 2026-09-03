// SPDX-License-Identifier: Apache-2.0
//! The runtime lowering path is GONE — the `#[forward]` macro bakes
//! every bucket's tape at expansion (`tape::lowering` runs there) and
//! the pool selects + materializes a baked variant at load. This shim
//! keeps the worker's spec-decode entry point at its old path.
pub use crate::tape::lowering::force_chunked_attention_addressing;

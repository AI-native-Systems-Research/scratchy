// SPDX-License-Identifier: Apache-2.0
//! Re-export of the backend-neutral [`GdnStatePool`](scratchy_layers::gdn_state::GdnStatePool).
//!
//! The pool is generic over `M: PoolMemory` and names no backend type, so it
//! lives in `scratchy-layers`. This module preserves the historical
//! `scratchy_target_cuda::gdn_state::…` path for existing callers.

pub use scratchy_layers::gdn_state::*;

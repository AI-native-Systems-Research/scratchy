// SPDX-License-Identifier: Apache-2.0
//! Re-export of the backend-neutral [`GdnStatePool`](scratchy_layers::gdn_state::GdnStatePool).
//!
//! The pool is generic over `M: PoolMemory` and names no backend type, so it
//! lives in `scratchy-layers`. This module preserves the
//! `scratchy_target_metal::gdn_state::…` path for the `ForwardCtx` field, the
//! worker, and the macro-emitted forward bodies.
pub use scratchy_layers::gdn_state::*;

use scratchy_tensors::DType;

/// The ssm state's element dtype the metal GDN kernels read and write: `device half*` in
/// `gdn_decode`, `gdn_scan_varlen` and `gdn_scan_pipelined`, cast at the buffer boundary — the
/// scan's arithmetic stays f32; the store's rounding is the state's per-step quantization. The
/// worker allocates the pool with this, and the shaders match it.
pub const GDN_SSM_STATE_DTYPE: DType = DType::F16;

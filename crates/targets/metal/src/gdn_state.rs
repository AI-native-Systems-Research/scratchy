// SPDX-License-Identifier: Apache-2.0
//! Re-export of the backend-neutral [`GdnStatePool`](scratchy_layers::gdn_state::GdnStatePool).
//!
//! The pool is generic over `M: PoolMemory` and names no backend type, so it
//! lives in `scratchy-layers`. This module preserves the
//! `scratchy_target_metal::gdn_state::…` path for the `ForwardCtx` field, the
//! worker, and the macro-emitted forward bodies.
pub use scratchy_layers::gdn_state::*;

use scratchy_tensors::DType;

/// The ssm state's element dtype the metal GDN kernels read and write: `gdn_ssm_t` in
/// `gdn_state.h` (`gdn_decode`, `gdn_verify`, `gdn_scan_varlen`, `gdn_scan_pipelined`), cast at
/// the buffer boundary — the scan's arithmetic is f32. The worker allocates the pool with this,
/// and the shaders match it. f32: an f16 state rounds every step's sums, which cost Qwen3.8-27B +
/// DFlash2 3% acceptance (3.48 -> 3.38 tokens a cycle, 16 ShareGPT prompts, greedy).
pub const GDN_SSM_STATE_DTYPE: DType = DType::F32;

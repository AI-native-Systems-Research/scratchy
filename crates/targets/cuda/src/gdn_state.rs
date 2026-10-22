// SPDX-License-Identifier: Apache-2.0
//! Re-export of the backend-neutral [`GdnStatePool`](scratchy_layers::gdn_state::GdnStatePool).
//!
//! The pool is generic over `M: PoolMemory` and names no backend type, so it
//! lives in `scratchy-layers`. This module preserves the historical
//! `scratchy_target_cuda::gdn_state::…` path for existing callers.

pub use scratchy_layers::gdn_state::*;

use scratchy_tensors::DType;

/// The ssm state's element dtype the cuda GDN kernels read and write (`device float*` — the
/// surviving `gdn_recurrent_*` kernels operate in f32).
pub const GDN_SSM_STATE_DTYPE: DType = DType::F32;

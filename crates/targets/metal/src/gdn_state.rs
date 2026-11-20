// SPDX-License-Identifier: Apache-2.0
//! The metal kernels' side of the recurrent state: the ssm element dtype they read and write. The
//! pool itself is backend-neutral (`scratchy_state_decoder_gdn::pool`).

use scratchy_tensors::DType;

/// The ssm state's element dtype the metal GDN kernels read and write: `gdn_ssm_t` in
/// `gdn_state.h` (`gdn_decode`, `gdn_verify`, `gdn_scan_varlen`, `gdn_scan_pipelined`), cast at
/// the buffer boundary — the scan's arithmetic is f32. The worker allocates the pool with this,
/// and the shaders match it. f32: an f16 state rounds every step's sums, which cost Qwen3.8-27B +
/// DFlash2 3% acceptance (3.48 -> 3.38 tokens a cycle, 16 ShareGPT prompts, greedy).
pub const GDN_SSM_STATE_DTYPE: DType = DType::F32;

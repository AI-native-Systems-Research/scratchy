// SPDX-License-Identifier: Apache-2.0
//! Canonical Gated-DeltaNet (GDN) recurrent-state layout for Qwen3.5 /
//! Qwen3-Next linear-attention layers.
//!
//! Unlike the paged KV cache (one block-paged buffer shared by all attention
//! layers), a GDN layer keeps **two** non-paged state buffers, sized **one
//! slot per active sequence** (`num_slots = max_num_seqs`), NOT by token
//! blocks. This mirrors the surviving CUDA kernels (`gdn_conv1d_*`,
//! `gdn_recurrent_fwd`) whose state args are:
//!
//! ```text
//!   conv_state : [num_slots, conv_dim, conv_state_len]   (conv_state_len = conv_kernel - 1)
//!   ssm_state  : [num_slots, num_v_heads, head_v_dim, head_k_dim]
//! ```
//!
//! * conv_state is the causal-conv1d ring of the last `conv_kernel-1` tokens
//!   per channel (`conv_dim = 2·key_dim + value_dim`).
//! * ssm_state is the recurrent delta-rule state matrix `S[head_v, head_k]`
//!   per value head.
//!
//! Like `PagedKvLayout` (now in `scratchy-target-metal` alongside the
//! `cpu_golden` oracle) this lives at the crate root (not feature-gated) so
//! the GPU pool sizing and the per-request slot allocator share one
//! definition.
//! The slot index is *not* part of the layout — it selects which slot of the
//! leading axis; the layout describes the within-slot addressing only.

/// Per-(linear-attention-)layer GDN state layout.
///
/// All dims come from the model config: `conv_dim = 2·(num_k_heads·head_k_dim)
/// + num_v_heads·head_v_dim`, `conv_kernel = linear_conv_kernel_dim`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GdnStateLayout {
    /// One slot per concurrently-resident sequence (= `max_num_seqs`).
    pub num_slots: u32,
    /// Channels of the causal conv1d = `2·key_dim + value_dim`.
    pub conv_dim: u32,
    /// Causal conv kernel width (`linear_conv_kernel_dim`, e.g. 4).
    pub conv_kernel: u32,
    pub num_v_heads: u32,
    pub head_v_dim: u32,
    pub head_k_dim: u32,
}

/// Runtime GDN config a hybrid arch reports to the worker so it can size and
/// allocate the `scratchy_target_cuda::gdn_state::GdnStatePool`. Backend-neutral:
/// carries the per-layer dims + the per-global-layer linear-attention mask. The
/// `num_slots` (= `max_num_seqs`) is supplied by the worker, not the model.
///
/// The proc-macro emits a per-arch `ScratchyWeights::gdn_runtime_config`
/// override returning `Some(_)` for arches whose forward body contains a
/// `gated_delta_net` op; non-hybrid arches use the trait default (`None`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GdnRuntimeConfig {
    pub conv_dim: u32,
    pub conv_kernel: u32,
    pub num_k_heads: u32,
    pub num_v_heads: u32,
    pub head_k_dim: u32,
    pub head_v_dim: u32,
    /// Per global-layer-index linear-attention mask, length `num_hidden_layers`.
    /// `true` = this layer is a GDN (linear-attention) layer.
    pub linear_layers: Vec<bool>,
}

impl GdnRuntimeConfig {
    /// Number of GDN (linear-attention) layers.
    pub fn num_linear_layers(&self) -> usize {
        self.linear_layers.iter().filter(|&&b| b).count()
    }
}

impl GdnStateLayout {
    /// Build a layout from the linear-attention config dims. `key_dim` and
    /// `value_dim` are derived (`num_k_heads·head_k_dim`,
    /// `num_v_heads·head_v_dim`); `conv_dim = 2·key_dim + value_dim`.
    pub fn new(
        num_slots: u32,
        num_k_heads: u32,
        num_v_heads: u32,
        head_k_dim: u32,
        head_v_dim: u32,
        conv_kernel: u32,
    ) -> Self {
        assert!(num_slots > 0, "GdnStateLayout: num_slots must be > 0");
        assert!(conv_kernel >= 1, "GdnStateLayout: conv_kernel must be >= 1");
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_dim = 2 * key_dim + value_dim;
        Self {
            num_slots,
            conv_dim,
            conv_kernel,
            num_v_heads,
            head_v_dim,
            head_k_dim,
        }
    }

    /// Conv ring length: `conv_kernel - 1` past tokens retained per channel.
    pub fn conv_state_len(&self) -> usize {
        (self.conv_kernel as usize) - 1
    }

    /// Elements in one slot of the conv-state buffer: `conv_dim · (conv_kernel-1)`.
    pub fn conv_elems_per_slot(&self) -> usize {
        (self.conv_dim as usize) * self.conv_state_len()
    }

    /// Total elements in the conv-state buffer: `num_slots · conv_elems_per_slot`.
    pub fn conv_buffer_elems(&self) -> usize {
        (self.num_slots as usize) * self.conv_elems_per_slot()
    }

    /// Elements in one slot of the recurrent (ssm) state buffer:
    /// `num_v_heads · head_v_dim · head_k_dim`.
    pub fn ssm_elems_per_slot(&self) -> usize {
        (self.num_v_heads as usize) * (self.head_v_dim as usize) * (self.head_k_dim as usize)
    }

    /// Total elements in the recurrent-state buffer.
    pub fn ssm_buffer_elems(&self) -> usize {
        (self.num_slots as usize) * self.ssm_elems_per_slot()
    }

    /// Element offset of `conv_state[slot, 0, 0]`.
    pub fn conv_slot_offset(&self, slot: u32) -> usize {
        (slot as usize) * self.conv_elems_per_slot()
    }

    /// Element offset of `ssm_state[slot, 0, 0, 0]`.
    pub fn ssm_slot_offset(&self, slot: u32) -> usize {
        (slot as usize) * self.ssm_elems_per_slot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen3.5-0.8B GDN dims: num_k_heads=16, num_v_heads=16, head_k=head_v=128,
    /// conv_kernel=4 ⇒ key_dim=value_dim=2048, conv_dim=6144, conv_state_len=3,
    /// ssm per slot = 16·128·128 = 262144.
    #[test]
    fn test_layout_qwen35_0p8b() {
        let l = GdnStateLayout::new(4, 16, 16, 128, 128, 4);
        assert_eq!(l.conv_dim, 6144);
        assert_eq!(l.conv_state_len(), 3);
        assert_eq!(l.conv_elems_per_slot(), 6144 * 3);
        assert_eq!(l.conv_buffer_elems(), 4 * 6144 * 3);
        assert_eq!(l.ssm_elems_per_slot(), 16 * 128 * 128);
        assert_eq!(l.ssm_buffer_elems(), 4 * 16 * 128 * 128);
    }

    /// GVA case (Qwen3.5-4B / 35B-A3B): num_k_heads=16, num_v_heads=32 ⇒
    /// key_dim=2048, value_dim=4096, conv_dim = 2·2048+4096 = 8192.
    #[test]
    fn test_layout_gva() {
        let l = GdnStateLayout::new(8, 16, 32, 128, 128, 4);
        assert_eq!(l.conv_dim, 8192);
        assert_eq!(l.ssm_elems_per_slot(), 32 * 128 * 128);
    }

    /// Slot offsets are contiguous, non-overlapping spans of one slot's size.
    #[test]
    fn test_slot_offsets() {
        let l = GdnStateLayout::new(3, 2, 4, 4, 4, 4);
        assert_eq!(l.conv_slot_offset(0), 0);
        assert_eq!(l.conv_slot_offset(1), l.conv_elems_per_slot());
        assert_eq!(l.conv_slot_offset(2), 2 * l.conv_elems_per_slot());
        assert_eq!(l.ssm_slot_offset(2), 2 * l.ssm_elems_per_slot());
        // last slot's span ends exactly at buffer end
        assert_eq!(
            l.ssm_slot_offset(2) + l.ssm_elems_per_slot(),
            l.ssm_buffer_elems()
        );
    }
}

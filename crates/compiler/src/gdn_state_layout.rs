// SPDX-License-Identifier: Apache-2.0
//! Canonical Gated-DeltaNet (GDN) recurrent-state layout for Qwen3.5 /
//! Qwen3-Next linear-attention layers.
//!
//! Unlike the paged KV cache (one block-paged buffer shared by all attention
//! layers), a GDN layer keeps **two** non-paged state buffers, sized **one
//! slot per active sequence** (`num_slots = max_num_seqs`), NOT by token
//! blocks. The `gdn_*` kernels' state args are:
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
//! The layout itself, `GdnStateLayout`, lives in `scratchy-subtile`
//! (`scratchy_subtile::gdn_state_layout::GdnStateLayout`, beside `PagedKvLayout`), so the GPU pool
//! sizing (the CUDA GDN state pool included) and the per-request slot allocator share one
//! definition. The slot index is *not* part of the layout — it selects which slot of the leading
//! axis; the layout describes the within-slot addressing only.

/// Runtime GDN config a hybrid arch reports to the worker so it can size and
/// allocate the `scratchy_layers::gdn_state::GdnStatePool`. Backend-neutral:
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
    /// The drafts each sequence of a verify step carries (`CanonicalParams::SPEC_DRAFTS`): what a
    /// slot keeps to undo them, as the GDN kernels bake it.
    pub spec_drafts: u8,
}

impl GdnRuntimeConfig {
    /// A slot's checkpoints: the model's drafts.
    pub fn checkpoint_rows(&self) -> scratchy_layers::gdn_state::CheckpointRows {
        scratchy_layers::gdn_state::CheckpointRows(self.spec_drafts)
    }

    /// Number of GDN (linear-attention) layers.
    pub fn num_linear_layers(&self) -> usize {
        self.linear_layers.iter().filter(|&&b| b).count()
    }

    /// The state geometry the GDN state pool allocates each linear layer.
    pub fn state_dims(&self) -> scratchy_layers::gdn_state::GdnStateDims {
        scratchy_layers::gdn_state::GdnStateDims {
            conv_dim: self.conv_dim as usize,
            conv_kernel: self.conv_kernel as usize,
            num_v_heads: self.num_v_heads as usize,
            head_v_dim: self.head_v_dim as usize,
            head_k_dim: self.head_k_dim as usize,
        }
    }
}

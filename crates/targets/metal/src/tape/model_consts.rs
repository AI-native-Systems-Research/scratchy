// SPDX-License-Identifier: Apache-2.0
//! The model facts the tape construction reads — as a VALUE.
//!
//! The construction fns were generic over `W: CanonicalParams` and read
//! associated consts. The `#[forward]` macro holds the same facts as
//! *values* at expansion (associated consts cannot be implemented from
//! them), so the fns take this struct instead: the runtime driver builds
//! it via [`MetalModelConsts::from_canonical`] (a const-for-const copy),
//! the macro builds it from `ModelParams` directly, and both call the
//! ONE construction implementation.
//!
//! Field names mirror the `CanonicalParams` consts one-for-one so the
//! `W::X → p.x` refactor is a mechanical rename with nothing to
//! re-derive or transpose.

use scratchy_tensors::{MetalDtype, ScaleDtype};

#[derive(Clone, Copy, Debug)]
pub struct MetalModelConsts {
    pub metal_dtype: MetalDtype,
    pub scale_dtype: ScaleDtype,
    pub head_dim: u32,
    pub global_head_dim: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub num_global_kv_heads: u32,
    pub q_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub attn_scale: f32,
    pub sliding_window: i32,
    pub final_logit_softcapping: f32,
    pub rms_norm_eps: f32,
    pub norm_weight_offset: f32,
    pub block_size: u32,
    pub global_block_size: u32,
    pub rot_dim: u32,
    pub global_rot_dim: u32,
    pub rope_on_read: bool,
    pub rope_proportional: bool,
    pub max_blocks_per_seq: u32,
    pub tq_kv_bits: u32,
    pub vision_num_heads: u32,
    pub vision_head_dim: u32,
    pub vision_q_size: usize,
    pub vision_in_features: usize,
    pub vision_rope_interleaved: bool,
    pub vision_attn_scale: f32,
    pub vision_patch_grid_side: u32,
    pub vision_pool_kernel: u32,
    pub gdn_num_k_heads: u32,
    pub gdn_num_v_heads: u32,
    pub gdn_head_k_dim: u32,
    pub gdn_head_v_dim: u32,
    pub gdn_conv_kernel: u32,
    pub gdn_conv_dim: usize,
}

impl MetalModelConsts {
    /// The runtime driver's constructor: every field is the same-named
    /// `CanonicalParams` const. A field this misses is a compile error
    /// here, not a wrong value downstream.
    pub fn from_canonical<W: scratchy_ir::CanonicalParams>() -> Self {
        Self {
            metal_dtype: W::METAL_DTYPE,
            scale_dtype: W::SCALE_DTYPE,
            head_dim: W::HEAD_DIM,
            global_head_dim: W::GLOBAL_HEAD_DIM,
            num_q_heads: W::NUM_Q_HEADS,
            num_kv_heads: W::NUM_KV_HEADS,
            num_global_kv_heads: W::NUM_GLOBAL_KV_HEADS,
            q_size: W::Q_SIZE,
            hidden_size: W::HIDDEN_SIZE,
            intermediate_size: W::INTERMEDIATE_SIZE,
            attn_scale: W::ATTN_SCALE,
            sliding_window: W::SLIDING_WINDOW,
            final_logit_softcapping: W::FINAL_LOGIT_SOFTCAPPING,
            rms_norm_eps: W::RMS_NORM_EPS,
            norm_weight_offset: W::NORM_WEIGHT_OFFSET,
            block_size: W::BLOCK_SIZE,
            global_block_size: W::GLOBAL_BLOCK_SIZE,
            rot_dim: W::ROT_DIM,
            global_rot_dim: W::GLOBAL_ROT_DIM,
            rope_on_read: W::ROPE_ON_READ,
            rope_proportional: W::ROPE_PROPORTIONAL,
            max_blocks_per_seq: W::MAX_BLOCKS_PER_SEQ,
            tq_kv_bits: W::TQ_KV_BITS,
            vision_num_heads: W::VISION_NUM_HEADS,
            vision_head_dim: W::VISION_HEAD_DIM,
            vision_q_size: W::VISION_Q_SIZE,
            vision_in_features: W::VISION_IN_FEATURES,
            vision_rope_interleaved: W::VISION_ROPE_INTERLEAVED,
            vision_attn_scale: W::VISION_ATTN_SCALE,
            vision_patch_grid_side: W::VISION_PATCH_GRID_SIDE,
            vision_pool_kernel: W::VISION_POOL_KERNEL,
            gdn_num_k_heads: W::GDN_NUM_K_HEADS,
            gdn_num_v_heads: W::GDN_NUM_V_HEADS,
            gdn_head_k_dim: W::GDN_HEAD_K_DIM,
            gdn_head_v_dim: W::GDN_HEAD_V_DIM,
            gdn_conv_kernel: W::GDN_CONV_KERNEL,
            gdn_conv_dim: W::GDN_CONV_DIM,
        }
    }
}

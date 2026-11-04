// SPDX-License-Identifier: Apache-2.0
//! The model facts the target lowerings read — as a VALUE, ONE struct for every target.
//!
//! This module is a PURE MOVE of metal's `tape::model_consts` (the struct metal's lowering was
//! already written once over) to this target-neutral crate, so that spyre reads the SAME struct
//! from the SAME fill site instead of carrying its own geometry beside it. The `#[forward]`
//! macro fills it once at expansion (see `codegen.rs`'s `emit_canonical_params_impl`); the
//! runtime driver builds it via [`MetalModelConsts::from_canonical`] (a const-for-const copy);
//! both targets' lowerings take it as a parameter and are written ONCE over whatever it holds.
//!
//! Field names mirror the `CanonicalParams` consts one-for-one so the
//! `W::X → p.x` refactor is a mechanical rename with nothing to
//! re-derive or transpose.
//!
//! The name keeps its metal origin on purpose: it is ONE struct under ONE name (no alias, no
//! spyre twin), and the name says where its lowering discipline came from.

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
    /// The FULL (causal/global) class's GQA group size — `num_q_heads / num_global_kv_heads`,
    /// divided ONCE at the fill, right after the law's const assert proves it exact. The splice's
    /// `NUM_QUERIES_PER_KV` constexpr reads it; no lowering re-derives the division.
    pub global_queries_per_kv: u32,
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
    /// `MROPE_SECTION.is_some()`: the rope reads the per-forward band-split cos/sin table.
    pub mrope: bool,
    pub kv_codec: scratchy_ir::KvCodec,
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
    /// The drafts each sequence of a verify step carries ([`scratchy_ir::CanonicalParams`]).
    pub spec_drafts: u32,
    /// A DFlash2 drafter's convolution channel group and selector rank; 0 for other models.
    pub draft_conv_group: u32,
    pub draft_selector_rank: u32,
    /// A DFlash2 drafter's mask token: what its block's proposal rows embed; 0 for other models.
    pub draft_mask_token: u32,
    /// The blocks of the rows a target captures for a drafter; 0 when none reads any.
    pub draft_capture_slots: u32,
    /// The width of a `target_hidden` row a head reads ([`scratchy_ir::CanonicalParams`]).
    pub target_hidden_width: usize,
    /// Whether a head's target routes each verify row through experts of its own
    /// ([`scratchy_ir::CanonicalParams`]).
    pub target_routes_experts: bool,
    /// The vocabulary: the lm_head's rows.
    pub vocab_size: usize,
}

/// ⭐ THE TWO KV GEOMETRIES A MODEL HOLDS — a hybrid-attention arch (gemma-4: sliding layers
/// beside every-6th global ones) states two `(nkvh, hd)` pairs, and every lowering that picks
/// one states WHICH. Not a `bool`: `local`/`global` are negations of each other and both are in
/// scope at every construction site — the exact shape that lets one land in the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvClass {
    /// The windowed (sliding) layers' — the model's BASE pair (`num_kv_heads`, `head_dim`).
    Base,
    /// The full causal layers' — the model's GLOBAL pair (`num_global_kv_heads`,
    /// `global_head_dim`).
    Global,
}

/// The class's own geometry, as [`MetalModelConsts::attn_geometry`] states it — the head dim,
/// kv heads, rot window and page a class owns, beside its query head count. Named fields, never
/// a tuple: a reader states which fact it takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvGeometry {
    /// The class's query head count (every class's own).
    pub nqh: u32,
    /// The class's kv head count — the GQA divisibility law at the fill proves it divides
    /// `nqh`, so it is at least one and no reader re-defends a zero.
    pub nkvh: u32,
    /// The class's head dim.
    pub hd: u32,
    /// The class's rot window.
    pub rot_dim: u32,
    /// The class's page size.
    pub block_size: u32,
}

impl From<bool> for KvClass {
    /// The class a site's own `is_global` bool names — the bridge for metal's arms that carry
    /// the class as a bool today. The mask-keyed [`KvClass::of`] is the preferred spelling; a
    /// site that has the mask never needs this.
    fn from(is_global: bool) -> Self {
        if is_global { Self::Global } else { Self::Base }
    }
}

impl KvClass {
    /// The class of a mask — the ONE statement of which pair a layer's attention takes: a
    /// windowed mask is a base-pair layer's, the causal mask a global-pair layer's. Both
    /// targets' selections go through here ([`MetalModelConsts::attn_geometry`]).
    pub const fn of(mask: crate::subtile_ir::AttnMask) -> Self {
        if mask.window().is_some() {
            Self::Base
        } else {
            Self::Global
        }
    }
}

impl MetalModelConsts {
    /// ⭐ THE CLASS'S ATTENTION GEOMETRY — every fact a class owns, the ONE statement of the
    /// two-geometry law every target's lowering reads: a windowed class (sliding layers,
    /// gemma-4's shape) takes the base facts, the full causal class the global ones; the query
    /// head count is every class's own. Stated here — on the one struct every lowering already
    /// reads — so no lowering re-derives it as an inline `if`, and spyre and metal cannot
    /// disagree about which facts a class takes. The class's head dim, kv heads, rot window and
    /// page travel TOGETHER: a site that has a [`KvClass`] in scope branches on nothing.
    pub const fn attn_geometry(&self, class: KvClass) -> KvGeometry {
        match class {
            KvClass::Base => KvGeometry {
                nqh: self.num_q_heads,
                nkvh: self.num_kv_heads,
                hd: self.head_dim,
                rot_dim: self.rot_dim,
                block_size: self.block_size,
            },
            KvClass::Global => KvGeometry {
                nqh: self.num_q_heads,
                nkvh: self.num_global_kv_heads,
                hd: self.global_head_dim,
                rot_dim: self.global_rot_dim,
                block_size: self.global_block_size,
            },
        }
    }

    /// The value that travels with a `compile_error!` token stream on an expansion-refused
    /// path: the emitted tokens never compile, so NO field of this value is ever read. It
    /// exists only so a refusal can return the same `(TokenStream, Self)` type as the success
    /// path — it is not a default, constructs nothing, and is unreachable in any build that
    /// does not already fail.
    pub const COMPILE_ERROR: Self = Self {
        metal_dtype: MetalDtype::Bf16,
        scale_dtype: ScaleDtype::Bf16,
        head_dim: 0,
        global_head_dim: 0,
        num_q_heads: 0,
        num_kv_heads: 0,
        num_global_kv_heads: 0,
        global_queries_per_kv: 0,
        q_size: 0,
        hidden_size: 0,
        intermediate_size: 0,
        attn_scale: 0.0,
        sliding_window: 0,
        final_logit_softcapping: 0.0,
        rms_norm_eps: 0.0,
        norm_weight_offset: 0.0,
        block_size: 0,
        global_block_size: 0,
        rot_dim: 0,
        global_rot_dim: 0,
        rope_on_read: false,
        rope_proportional: false,
        mrope: false,
        kv_codec: scratchy_ir::KvCodec::Dense,
        vision_num_heads: 0,
        vision_head_dim: 0,
        vision_q_size: 0,
        vision_in_features: 0,
        vision_rope_interleaved: false,
        vision_attn_scale: 0.0,
        vision_patch_grid_side: 0,
        vision_pool_kernel: 0,
        gdn_num_k_heads: 0,
        gdn_num_v_heads: 0,
        gdn_head_k_dim: 0,
        gdn_head_v_dim: 0,
        gdn_conv_kernel: 0,
        gdn_conv_dim: 0,
        spec_drafts: 0,
        draft_conv_group: 0,
        draft_selector_rank: 0,
        draft_mask_token: 0,
        draft_capture_slots: 0,
        target_hidden_width: 0,
        target_routes_experts: false,
        vocab_size: 0,
    };

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
            // A plain division: the GQA law's const assert (the one this fill's own doc names)
            // proves `num_global_kv_heads >= 1` for every attention body, so the division is
            // exact and safe. The no-attention body (`head_dim == 0`, vision) has no GQA group;
            // gating the field on that head-dim zero is the parked "absence as a type" item,
            // held here as an explicit `head_dim > 0` arm rather than a 0-queries sentinel.
            global_queries_per_kv: if W::HEAD_DIM > 0 {
                W::NUM_Q_HEADS / W::NUM_GLOBAL_KV_HEADS
            } else {
                0
            },
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
            mrope: W::MROPE_SECTION.is_some(),
            kv_codec: W::KV_CODEC,
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
            spec_drafts: W::SPEC_DRAFTS,
            draft_conv_group: W::DRAFT_CONV_GROUP,
            draft_selector_rank: W::DRAFT_SELECTOR_RANK,
            draft_mask_token: W::DRAFT_MASK_TOKEN,
            draft_capture_slots: W::DRAFT_CAPTURE_SLOTS,
            target_hidden_width: W::TARGET_HIDDEN_WIDTH,
            target_routes_experts: W::TARGET_ROUTES_EXPERTS,
            vocab_size: W::VOCAB_SIZE,
        }
    }
}

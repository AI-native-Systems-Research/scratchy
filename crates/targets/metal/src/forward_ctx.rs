// SPDX-License-Identifier: Apache-2.0
//! Ambient runtime args ([`ForwardCtx`]) the `#[forward]`-emitted code takes,
//! plus [`EmbedPatch`] (the multimodal embed-splice descriptor).
//!
//! `ForwardCtx` names the backend-coupled `KvCachePool` / `GdnStatePool`, so it
//! lives in the target crate rather than the cfg-free compiler. The compiler's
//! `ScratchyWeights` trait takes the neutral `scratchy_tensors::ForwardCtxHandle`
//! instead; the per-arch `impl ScratchyWeights` recovers `&ForwardCtx` from it.
//! The generated forward fns and the metal worker name
//! `::scratchy_target_metal::ForwardCtx` directly.
//!
//! This is the metal sibling of the cuda `ForwardCtx`: the vision fields are
//! unconditional (metal is the only configuration here), `has_spec_tokens` is
//! kept (the metal lm_head slice gating reads it), and the cuda-only nccl
//! `tp_group` field is absent (metal is single-device).

use crate::TensorView;
use crate::gdn_state::GdnStatePool;
use crate::kv_cache::KvCachePool;

/// Ambient runtime args the emitted forward fn needs. The caller builds a
/// `ForwardCtx` per forward call and passes it in. Fields are the union of what
/// any ported kernel needs at invocation time; new kernels can reference new
/// fields, which is the extension point. RoPE caches live on the emitted
/// per-arch `Weights` struct (built inside `Weights::load`), not threaded here.
pub struct ForwardCtx<'a> {
    pub input_ids: TensorView<'a>,
    /// Positions tensor. Text arches: `[n_tokens]` u32. MRoPE arches
    /// (Qwen2-VL etc.): `[3, n_tokens]` u32 — rows are (T, H, W) coordinates.
    /// Set up by whoever constructs the `ForwardCtx` (the metal worker).
    pub positions: TensorView<'a>,
    pub slot_mapping: TensorView<'a>,
    pub cu_seqlens_q: TensorView<'a>,
    pub seqused_k: TensorView<'a>,
    /// Per-TOKEN SPAN LABEL for block-diagonal span attention (`0` =
    /// shared/query — attends everything; `first_token+1` = a Relocatable span —
    /// attends only `[first_token, q_pos]`). One entry per token (NOT per block),
    /// so a span may begin/end mid-block with no cropping. Owned, built
    /// per-forward from the request's `block_annotations` via
    /// `span_ids_per_token`. `None` ⇒ no spans ⇒ the kernel's span mask is
    /// inert (byte-identical to the legacy path).
    pub span_ids: Option<Vec<u32>>,
    pub block_table: TensorView<'a>,
    /// Sliding KV-cache groups (gemma4 SWA): per SLIDING group, the
    /// slot_mapping + block table for that group's local layers. vLLM's
    /// group-shared layout has N sliding groups (gemma4: 5); group 0 (full)
    /// uses [`Self::slot_mapping`] / [`Self::block_table`]. Empty on non-SWA
    /// models (the macro emits a single-group `ForwardInputs`). Index `s` here
    /// is KV-cache group `s + 1`.
    pub sliding_slot_mappings: Vec<TensorView<'a>>,
    pub sliding_block_tables: Vec<TensorView<'a>>,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub kv_cache: &'a KvCachePool,
    /// Gated-DeltaNet recurrent-state pool for hybrid arches (Qwen3.5 /
    /// Qwen3-Next). The non-paged sibling of `kv_cache`: the
    /// `Instruction::GatedDeltaNet` eval arm reads/writes per-linear-layer
    /// `conv_state`/`ssm_state` here, indexed by [`Self::gdn_state_indices`].
    /// `None` for non-hybrid arches.
    pub gdn_state: Option<&'a GdnStatePool<crate::PoolMem>>,
    /// Per-sequence GDN state-slot ids, shape `[num_seqs]` i32. `None` for
    /// non-hybrid arches.
    pub gdn_state_indices: Option<TensorView<'a>>,
    /// Per-sequence GDN fresh flags, shape `[num_seqs]` u32 — 1 on a
    /// sequence's first (fresh) forward. `None` for non-hybrid arches.
    pub gdn_is_fresh: Option<TensorView<'a>>,
    /// `true` when at least one req in this forward carries `spec_token_ids`
    /// (= a spec-decode verify batch). Threaded through to `gate_matches` so
    /// the lm_head slice trio (gated `OnlyIfSingleSeqNoSpec`) skips and the
    /// full `M=bucket_m` lm_head fallback (gated `OnlyIfMultiSeqOrSpec`) fires.
    /// Caller sets `false` for prefill / decode / lockstep / draft chain.
    pub has_spec_tokens: bool,
    /// TurboQuant KV compression on (kv_cache_dtype == "turboquant"). When set
    /// (and head_dim is a supported power-of-2 ≤256), the metal RuntimeFactory
    /// provisions the packed stores + scratch and the per-layer tape ops fire.
    pub kv_turboquant: bool,
    /// Multimodal embed splice. `mm_embeds` carries the projected
    /// vision-encoder output `[total_mm_tokens, hidden]`; `embed_patches`
    /// names the destination ranges in the input-id sequence. Empty
    /// `embed_patches` = text-only batch, no splice. `mm_embeds = None` is
    /// only valid when `embed_patches` is empty.
    pub mm_embeds: Option<TensorView<'a>>,
    pub embed_patches: &'a [EmbedPatch],
    /// Vision-tower 2D RoPE cos table, `[total_L, head_dim/2]` bf16. Built
    /// host-side from `grid_thw` per vision-encoder call. `None` for text-side
    /// forwards.
    pub vision_rope_cos: Option<TensorView<'a>>,
    /// Vision-tower 2D RoPE sin table. Same shape / population as
    /// [`Self::vision_rope_cos`].
    pub vision_rope_sin: Option<TensorView<'a>>,
    /// Vision-tower 2D RoPE **angle** table (`freqs`, f32), `[total_L,
    /// vision_head_dim/2]`. The metal `vision_rope_2d` kernel reads raw
    /// `freqs` and computes cos/sin internally. `None` for text-side forwards.
    pub vision_rope_freqs: Option<TensorView<'a>>,
    /// Vision-tower input patches buffer, `[num_tokens, vision_in_features]`
    /// bf16. The `vision_forward` host wrapper packs per-image CHW pixels into
    /// this rank-2 layout. `None` for text-side forwards.
    pub pixels: Option<TensorView<'a>>,
    /// Qwen3.5-VL learned positional embedding, host-interpolated, `[num_tokens,
    /// vision_embed_dim]`. `None` for text-side calls / towers without one.
    pub pos_embeds: Option<TensorView<'a>>,
    /// Qwen2.5-VL: cu_seqlens for the per-image full-frame segmentation.
    /// `None` outside windowed-attention vision arches.
    pub vision_cu_seqlens_full: Option<TensorView<'a>>,
    /// Qwen2.5-VL: cu_seqlens for the per-window segmentation. Same `None`
    /// semantics as [`Self::vision_cu_seqlens_full`].
    pub vision_cu_seqlens_window: Option<TensorView<'a>>,
    /// Qwen2.5-VL: max segment length under [`Self::vision_cu_seqlens_full`].
    pub vision_max_seqlen_full: Option<usize>,
    /// Qwen2.5-VL: max segment length under [`Self::vision_cu_seqlens_window`].
    pub vision_max_seqlen_window: Option<usize>,
    /// Qwen2.5-VL: per-merged-cell natural→window-grouped permutation,
    /// `[L / spatial_merge_size²]` u32.
    pub vision_window_index: Option<TensorView<'a>>,
    /// Qwen2.5-VL: inverse of [`Self::vision_window_index`].
    pub vision_reverse_indices: Option<TensorView<'a>>,
    /// SigLIP-style learned positional embedding indices, `[num_tokens]` u32.
    /// Consumed by `Instruction::PosEmbed`. `None` for text-side forwards and
    /// vision arches without a positional embedding.
    pub vision_position_ids: Option<TensorView<'a>>,
    /// Per-sequence final-token row indices, `[num_seqs]` u32. `Some(_)` when
    /// `num_seqs < num_tokens` so the lm_head GEMM gathers before the matmul;
    /// `None` at decode or when not built by the worker.
    pub last_token_indices: Option<TensorView<'a>>,
}

// `EmbedPatch` (the multimodal embed-splice descriptor) is a cfg-free type in
// the compiler, re-exported here so `scratchy_target_metal::EmbedPatch` and the
// `ForwardCtx` field above keep resolving.
pub use scratchy_forward_compiler::EmbedPatch;

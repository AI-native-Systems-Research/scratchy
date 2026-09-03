// SPDX-License-Identifier: Apache-2.0
//! Ambient runtime args ([`ForwardCtx`]) the `#[forward]`-emitted code
//! takes, plus [`EmbedPatch`] (the multimodal embed-splice descriptor).
//!
//! `ForwardCtx` names the backend-coupled `KvCachePool` / `GdnStatePool`, so it
//! lives here (cross-backend cuda + metal) rather than in the cfg-free
//! compiler. The compiler's `ScratchyWeights` trait takes the neutral
//! `scratchy_tensors::ForwardCtxHandle` instead; the per-arch `impl
//! ScratchyWeights` recovers `&ForwardCtx` from it. The generated forward fns
//! and the worker name `::scratchy_target_cuda::ForwardCtx` directly.

use crate::gdn_state::GdnStatePool;
use crate::kv_cache::KvCachePool;
use crate::tensor::TensorView;

/// Ambient runtime args the emitted forward fn needs. The
/// caller builds a `ForwardCtx` per forward call and passes
/// it in. Fields are the union of what any ported kernel
/// needs at invocation time; new kernels can reference new
/// fields, which is the extension point. RoPE caches live on
/// the emitted per-arch `Weights` struct (both global `rotary`
/// and Gemma3's `rotary_local`), built inside `Weights::load`
/// from manifest-driven bounds/scalars/rope_scaling — not
/// threaded through here.
pub struct ForwardCtx<'a> {
    pub input_ids: TensorView<'a>,
    /// Positions tensor. Shape contract:
    /// - Text arches (`W::MROPE_SECTION == None`): `[n_tokens]` u32.
    /// - MRoPE arches (Qwen2-VL etc. with `W::MROPE_SECTION == Some(_)`):
    ///   `[3, n_tokens]` u32 — rows are (T, H, W) coordinates;
    ///   the rope kernel reads each rotary index from the row that
    ///   `mrope_section` assigns it to. Must be set up by whoever
    ///   constructs the `ForwardCtx` (currently the `Self::Scratchy`
    ///   arm in `scratchy-target-cuda::cuda_worker`); scratchy-forward-compiler
    ///   itself is shape-agnostic past the kernel boundary.
    pub positions: TensorView<'a>,
    pub slot_mapping: TensorView<'a>,
    pub cu_seqlens_q: TensorView<'a>,
    pub seqused_k: TensorView<'a>,
    pub block_table: TensorView<'a>,
    /// Per-KV-group block tables (gemma4 hybrid SWA): `block_tables[g]` is
    /// group `g`'s table (group 0 = full/global, 1.. = sliding). A layer picks
    /// its table via [`Self::block_table_for`]. EMPTY on uniform models — the
    /// helpers then fall back to the scalar [`Self::block_table`] (byte-
    /// identical to the pre-hybrid single-table path). Built by the worker from
    /// `AttentionMetadata::sliding_groups`.
    pub block_tables: Vec<TensorView<'a>>,
    /// Per-KV-group slot mappings, paired with [`Self::block_tables`]. Empty on
    /// uniform models → [`Self::slot_mapping_for`] falls back to the scalar
    /// [`Self::slot_mapping`]. A sliding layer's KV write must use its group's
    /// slot mapping (encoded with that group's block size).
    pub slot_mappings: Vec<TensorView<'a>>,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub kv_cache: &'a KvCachePool,
    /// Gated-DeltaNet recurrent-state pool for hybrid arches (Qwen3.5 /
    /// Qwen3-Next). The non-paged sibling of `kv_cache`: the
    /// `Instruction::GatedDeltaNet` eval arm reads/writes per-linear-layer
    /// `conv_state`/`ssm_state` here, indexed by the slot ids in
    /// [`Self::gdn_state_indices`]. `None` for non-hybrid arches (their
    /// codegen never emits a `GatedDeltaNet` op, so the field is never
    /// read) — same optional-resource contract as `tp_group`.
    pub gdn_state: Option<&'a GdnStatePool<crate::PoolMem>>,
    /// Per-sequence GDN state-slot ids, shape `[num_seqs]` i32 — one slot
    /// per batched sequence in `cu_seqlens_q` order. Built each step by the
    /// worker from its `GdnSlotAllocator`. `None` for non-hybrid arches.
    pub gdn_state_indices: Option<TensorView<'a>>,
    /// Per-sequence GDN fresh flags, shape `[num_seqs]` u32 — 1 when the
    /// sequence is on its first (fresh) forward, so the GDN conv1d/scan
    /// kernels treat the slot's recurrent state as zero (the
    /// "degeneration after N requests" guard). `None` for non-hybrid arches.
    pub gdn_is_fresh: Option<TensorView<'a>>,
    /// Multimodal embed splice. `mm_embeds` carries the projected
    /// vision-encoder output `[total_mm_tokens, hidden]` produced by
    /// the multimodal `vision_forward`; `embed_patches`
    /// names the destination ranges in the input-id sequence. After
    /// the `Instruction::Embed` arm gathers `embed_tokens`, it
    /// D2D-copies each patch's slice from `mm_embeds` into the
    /// gather output's corresponding rows. Empty `embed_patches` =
    /// text-only batch, no splice — byte-identical to pre-MM
    /// behavior. `mm_embeds = None` is only valid when
    /// `embed_patches` is empty.
    #[cfg(feature = "cuda")]
    pub mm_embeds: Option<TensorView<'a>>,
    #[cfg(feature = "cuda")]
    pub embed_patches: &'a [EmbedPatch],
    /// Vision-tower 2D RoPE cos table, shape `[total_L, head_dim/2]`,
    /// bf16. Built host-side from `grid_thw` per vision-encoder call;
    /// the caller (`vision_forward`) uploads it and sets the field
    /// before invoking the vision interpreter. `None` for text-side
    /// forward calls — the `Instruction::VisionRope` arm panics on
    /// `expect` if reached without these set, mirroring the
    /// `tp_group` contract for `Instruction::AllReduce` at tp>1.
    #[cfg(feature = "cuda")]
    pub vision_rope_cos: Option<TensorView<'a>>,
    /// Vision-tower 2D RoPE sin table. Same shape / population /
    /// invariants as [`Self::vision_rope_cos`].
    #[cfg(feature = "cuda")]
    pub vision_rope_sin: Option<TensorView<'a>>,
    /// Vision-tower 2D RoPE **angle** table (`freqs`, f32), shape
    /// `[total_L, vision_head_dim/2]`. METAL-ONLY consumer: the
    /// `vision_rope_2d` kernel reads raw `freqs` and computes
    /// cos/sin internally (the CUDA `vision_rope_apply` kernel reads
    /// the precomputed [`Self::vision_rope_cos`]/[`Self::vision_rope_sin`]
    /// instead). Built host-side from `grid_thw`, uploaded and set by
    /// the metal `vision_forward` wrapper before the interpreter runs.
    /// `None` on the cuda path and for text-side forwards.
    #[cfg(feature = "cuda")]
    pub vision_rope_freqs: Option<TensorView<'a>>,
    /// Vision-tower input patches buffer, shape `[num_tokens,
    /// vision_in_features]`, bf16. The vision encoder's
    /// `vision_forward` host wrapper packs per-image CHW pixels
    /// into this rank-2 layout (one row per patch, channels-times-
    /// patch-area columns), uploads it, and sets the field before
    /// invoking the vision interpreter. `None` for text-side
    /// forward calls — `Instruction::LoadPixels` panics on
    /// `expect` if reached without it set, mirroring the
    /// [`Self::vision_rope_cos`] contract.
    ///
    /// Synthesized by `vision_lowering::materialize_pixels` after
    /// `fuf::unroll`: every vision-prelude `pixels` extern in the
    /// DSL classifies into a `FufInput::Extern` and is rewritten
    /// to a `FufInput::Tile` whose producer is a single
    /// `OpKind::LoadPixels` node; that node's runtime
    /// counterpart copies this view into a tile-table OwnedTensor
    /// the rest of the encoder consumes. See
    /// `Instruction::LoadPixels` for the eval body.
    #[cfg(feature = "cuda")]
    pub pixels: Option<TensorView<'a>>,
    /// Qwen3.5-VL learned positional embedding, already interpolated
    /// host-side via `fast_pos_embed_interpolate` (4-corner bilinear
    /// over a 48×48 grid), shape `[num_tokens, vision_embed_dim]`,
    /// model dtype. The vision wrapper computes, uploads, and sets it
    /// before invoking the interpreter; the DSL adds it to the
    /// patch-embed output (`add(pos_embeds, hidden_states)`). `None`
    /// for text-side calls and towers without a learned positional
    /// embedding — `Instruction::LoadPosEmbeds` panics on `expect` if
    /// reached without it set, mirroring [`Self::pixels`]. Synthesized
    /// into a tile by `vision_lowering::materialize_pos_embeds`; see
    /// `Instruction::LoadPosEmbeds` for the eval body.
    #[cfg(feature = "cuda")]
    pub pos_embeds: Option<TensorView<'a>>,
    /// Qwen2.5-VL: cu_seqlens for the per-image **full-frame**
    /// segmentation. Populated by the vision wrapper for arches
    /// whose body calls `varlen_attention(..., cu_seqlens_full,
    /// max_seqlen_full)` at fullatt-layer indices; `None` for
    /// every text-side call and for vision arches that use a
    /// single `cu_seqlens_q` (Qwen2-VL).
    #[cfg(feature = "cuda")]
    pub vision_cu_seqlens_full: Option<TensorView<'a>>,
    /// Qwen2.5-VL: cu_seqlens for the per-window segmentation.
    /// Populated by the vision wrapper for windowed-attention
    /// layers; same `None` semantics as
    /// [`Self::vision_cu_seqlens_full`].
    #[cfg(feature = "cuda")]
    pub vision_cu_seqlens_window: Option<TensorView<'a>>,
    /// Qwen2.5-VL: max segment length under
    /// [`Self::vision_cu_seqlens_full`]. `None` when not in use.
    #[cfg(feature = "cuda")]
    pub vision_max_seqlen_full: Option<usize>,
    /// Qwen2.5-VL: max segment length under
    /// [`Self::vision_cu_seqlens_window`]. `None` when not in use.
    #[cfg(feature = "cuda")]
    pub vision_max_seqlen_window: Option<usize>,
    /// Qwen2.5-VL: per-merged-cell natural→window-grouped
    /// permutation `[L / spatial_merge_size²]` u32. Drives the
    /// entry-side `embedding_gather(x, window_index)` (and the
    /// matching `embedding_gather(cos/sin, window_index)`) so
    /// every windowed-attention layer reads contiguous segments.
    #[cfg(feature = "cuda")]
    pub vision_window_index: Option<TensorView<'a>>,
    /// Qwen2.5-VL: inverse of [`Self::vision_window_index`] —
    /// per-merged-cell window-grouped→natural permutation that
    /// undoes the entry permute on the merger output before
    /// splice into the language-model embedding stream.
    #[cfg(feature = "cuda")]
    pub vision_reverse_indices: Option<TensorView<'a>>,
    /// SigLIP-style learned positional embedding indices, shape
    /// `[num_tokens]` u32. Built host-side as `[0..num_pos,
    /// 0..num_pos, ...]` per image. Consumed by
    /// `Instruction::PosEmbed` via `kernels::embedding_gather_masked`
    /// (the same kernel `Instruction::Embed` calls). `None` for
    /// text-side forward calls and for vision arches that don't
    /// need a positional embedding (Qwen2-VL / Qwen2.5-VL use
    /// 2D RoPE via `vision_rope` instead).
    #[cfg(feature = "cuda")]
    pub vision_position_ids: Option<TensorView<'a>>,
    /// Per-sequence final-token row indices, shape `[num_seqs]` u32.
    /// At prefill, lm_head only needs the last token of each sequence
    /// (Python vLLM: `logits_indices = query_start_loc[1:] - 1`); the
    /// metal interpreter has a `GatherLastToken` lowering for the
    /// same reason. `Some(_)` when `num_seqs < num_tokens` so the
    /// lm_head GEMM gathers to `[num_seqs, hidden]` before the
    /// matmul, turning a wasteful `M=num_tokens × N=vocab × K=hidden`
    /// GEMM into `M=num_seqs`. `None` at decode (every row is a
    /// sample row) or when not built by the worker. Consumed by
    /// `Instruction::CutlassFusedAddRmsNormGemm` (and any future
    /// lm_head op) — see cuda_worker.rs for context.
    pub last_token_indices: Option<TensorView<'a>>,
    // The TP communicator the `Instruction::AllReduce` arm calls
    // into. `None` at tp=1 (the lowering pass emits no AllReduce
    // rows, so the field is never read). `Some(_)` only when
    // built with `--features nccl` AND the worker constructed an
    // NCCL group for this rank — see scratchy-target-cuda::cuda_worker.
    #[cfg(feature = "nccl")]
    pub tp_group: Option<&'a std::sync::Arc<crate::NcclGroup>>,
}

impl<'a> ForwardCtx<'a> {
    /// The block table a given layer's paged attention must read. On uniform
    /// models (`block_tables` empty, or a single group) this is the scalar
    /// [`Self::block_table`] — byte-identical to the pre-hybrid path. On
    /// gemma4 hybrid SWA it resolves the layer's KV-cache group via the pool's
    /// `layer_to_group` and returns that group's table. This is the ONLY way
    /// attention ops should reach the block table — reading the raw scalar
    /// field layer-agnostically is the bug (sliding layer reading the global
    /// group's blocks → garbage after token 1) the grouped path fixes.
    #[inline]
    pub fn block_table_for(&self, layer: usize) -> TensorView<'a> {
        if self.block_tables.len() <= 1 {
            return self.block_table;
        }
        let g = self.kv_cache.group_of_layer(layer);
        self.block_tables
            .get(g)
            .copied()
            .unwrap_or(self.block_table)
    }

    /// The slot mapping a given layer's KV-cache WRITE must use, paired with
    /// [`Self::block_table_for`]. Falls back to the scalar
    /// [`Self::slot_mapping`] on uniform models.
    #[inline]
    pub fn slot_mapping_for(&self, layer: usize) -> TensorView<'a> {
        if self.slot_mappings.len() <= 1 {
            return self.slot_mapping;
        }
        let g = self.kv_cache.group_of_layer(layer);
        self.slot_mappings
            .get(g)
            .copied()
            .unwrap_or(self.slot_mapping)
    }
}

// `EmbedPatch` (the multimodal embed-splice descriptor) is now a cfg-free type
// in the compiler, re-exported here so `scratchy_target_cuda::EmbedPatch` and
// the `ForwardCtx` fields above keep resolving.
pub use scratchy_forward_compiler::EmbedPatch;

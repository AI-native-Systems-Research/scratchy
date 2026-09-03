// SPDX-License-Identifier: Apache-2.0
//! Per-forward runtime buffers that the worker re-binds across forwards.
//!
//! Each [`Binding::Runtime`](super::lowered::Binding::Runtime) on the
//! lowered tape names one of these slots. The worker bakes the
//! buffer's pointer into the dispatch at recording time — so the buffers themselves
//! must outlive every call. The engine writes new content into each
//! buffer's `contents()` on every forward and re-runs
//! the MTL4 compute encoder; no dispatch re-recording happens on the
//! per-forward path.
//!
//! [`RuntimeBindings`] is the Metal analogue of CUDA's `ForwardCtx`:
//! one struct per worker, per-shape buffers sized for the worker's
//! largest bucket. KV cache buffers are split per layer because the
//! lowered tape carries the layer index on `KvCacheK`/`KvCacheV`
//! variants (each layer gets its own `Buffer`); other runtime tensors
//! are global per worker.

use crate::interpreter::metal::__re::Buffer;

use super::lowered::RuntimeBindingKind;

/// Per-worker runtime buffers. The worker reads `&Buffer` for each
/// runtime binding while baking its bucket dispatchs; the engine refills
/// `contents()` on every forward.
///
/// All `Vec<Buffer>` fields are indexed by layer id (the `layer`
/// payload on the matching [`RuntimeBindingKind`] variant). The
/// constructor walks model meta + bucket layout to size every field
/// for the worker's largest bucket; the worker itself only borrows.
/// TurboQuant per-forward buffers (packed canonical KV stores + codebook).
/// `packed_*`/`norms_*` are `[num_layers]`; codebook buffers are shared.
pub struct TqRuntimeBuffers {
    pub packed_k: Vec<Buffer>,
    pub packed_v: Vec<Buffer>,
    pub norms_k: Vec<Buffer>,
    pub norms_v: Vec<Buffer>,
    pub signs: Buffer,
    pub boundaries: Buffer,
    pub centroids: Buffer,
    /// The reused per-layer fp16 SCRATCH: chunk-table (bound at every layer's
    /// `kv_cache_k/v` slot — attention + rope + dequant + quantize all use it)
    /// and its backing data (kept alive + resident here).
    pub scratch_k_table: Buffer,
    pub scratch_v_table: Buffer,
    pub scratch_k_data: Buffer,
    pub scratch_v_data: Buffer,
}

pub struct RuntimeBindings {
    pub input_ids: Buffer,
    pub positions: Buffer,
    pub cu_seqlens_q: Buffer,
    pub seq_used_k: Buffer,
    /// Spans block-diagonal attention: `[num_logical_blocks]` u32 SPAN LABEL
    /// per block — `0` = shared/query (attends everything), `k+1` = the k-th
    /// Relocatable span (attends only itself). The prefill attention kernel
    /// reads it as ONE extra mask term (self-only). All-zero (the default and
    /// every non-spans forward) ⇒ the mask is a no-op ⇒ byte-identical.
    pub span_ids: Buffer,
    /// Per-KV-cache-group slot_mappings and block tables (vLLM hybrid layout).
    /// `[num_groups]`: one group for uniform models, `1 + N` for gemma4 SWA
    /// (full + N sliding). A layer binds its group's buffer via
    /// [`Self::layer_to_group`]. The worker fills each group's buffer from the
    /// scheduler's per-group block IDs (full-class slot_mapping encoded with
    /// the page-unified `GLOBAL_BLOCK_SIZE`, sliding with `BLOCK_SIZE`).
    pub slot_mappings: Vec<Buffer>,
    pub block_tables: Vec<Buffer>,
    /// `[num_layers]` u32 — each layer's KV-cache group index, indexing
    /// [`Self::slot_mappings`] / [`Self::block_tables`]. All-zero on uniform
    /// models (every layer → group 0 → byte-identical to the single-table
    /// path). Set from the model's `HybridKvLayout` on SWA arches.
    pub layer_to_group: Vec<u32>,
    /// Per-layer paged K cache buffers. Shape: `[num_layers]`.
    pub kv_cache_k: Vec<Buffer>,
    /// Per-layer paged V cache buffers. Shape: `[num_layers]`.
    pub kv_cache_v: Vec<Buffer>,
    /// Spans / rope-on-read: per-layer `[num_physical_blocks]` u8 mirror
    /// of the KV pool's `block_is_unrotated` flag (`1` = stored
    /// unrotated). Parallel to `kv_cache_k/v`, direct-indexed by layer.
    /// On `W::ROPE_ON_READ` arches the worker memcpys the flags here each
    /// forward; on every other arch it is a one-element placeholder that
    /// is never bound (so non-spans pays nothing).
    pub block_unrotated_flags: Vec<Buffer>,
    /// TurboQuant runtime buffers — `Some` only when kv_cache_dtype=turboquant.
    /// The packed code stores are the canonical KV (~4.7x smaller); the per-layer
    /// dequant fills the (reused) `kv_cache_k/v` scratch before each attention.
    pub tq: Option<TqRuntimeBuffers>,
    /// `[1]` u32 — actual `num_tokens` of the in-flight forward.
    /// The pool writes the current call's `num_tokens` into this
    /// 4-byte buffer at the start of every `forward()` so kernels
    /// that need M at runtime can read it without a function
    /// constant (M varies per call).
    pub num_tokens_u32: Buffer,
    /// `[1]` u32 — number of sample rows the lm_head slice trio
    /// gathers / GEMMs / scatters this forward. Equals the length
    /// of the in-flight `last_token_indices` slice (= `num_seqs` for
    /// non-spec prefill/decode, `K+1` for spec-decode verify, `0`
    /// when the caller doesn't pass indices). Read by the
    /// `gather_last_token`/`scatter_first_to_last_row` kernels.
    pub num_sample_rows_u32: Buffer,
    /// `[num_sample_rows]` u32 — per-sample-row source/destination
    /// index into the lm_head input/output tensor. Mirrors Python
    /// vLLM's `logits_indices` and the CUDA path's
    /// `ForwardCtx.last_token_indices`. Sized at construction for
    /// the worker's largest bucket's `bucket_m`; the pool writes
    /// the in-flight slice's content at the start of every forward.
    pub sample_indices: Buffer,
    /// Per-layer GDN conv-state ring buffers (persistent f32 pool),
    /// GLOBAL-layer-indexed like `gdn_state_ssm`. Non-linear (full-attn)
    /// layers hold a dummy buffer that is never bound (the
    /// `GatedDeltaNet` lowering only emits `GdnConvState` on linear
    /// layers). Empty for non-hybrid arches.
    pub gdn_state_conv: Vec<Buffer>,
    /// Per-layer GDN recurrent (ssm) state buffers. Same indexing /
    /// dummy / emptiness contract as [`Self::gdn_state_conv`].
    pub gdn_state_ssm: Vec<Buffer>,
    /// `[num_seqs]` i32 — GDN state-pool slot id per batched sequence.
    /// Shared storage; the pool overwrites `contents()` each forward.
    pub gdn_state_indices: Buffer,
    /// `[num_seqs]` u32 — per-sequence fresh flag. Shared storage.
    pub gdn_is_fresh: Buffer,
    /// `[total_L, vision_head_dim/2]` f32 — vision 2D-RoPE angle table
    /// (`freqs`). Shared storage; the pool overwrites `contents()` each
    /// forward. 16-byte placeholder on non-vision arches.
    pub vision_rope_freqs: Buffer,
    /// `[num_tokens, vision_in_features]` model-dtype — vision patch
    /// pixel rows. Shared storage; overwritten per forward. 16-byte
    /// placeholder on non-vision arches.
    pub pixels: Buffer,
    /// `[num_tokens, vision_embed_dim]` model-dtype — Qwen3.5-VL
    /// host-interpolated learned positional embedding. Shared storage;
    /// overwritten per forward. 16-byte placeholder on non-vision arches
    /// and on towers without a learned positional embedding.
    pub vision_pos_embeds: Buffer,
    /// `[max_m, hidden]` model-dtype — projected vision embeddings for
    /// the multimodal splice. Shared storage; overwritten per forward
    /// (text-only batches leave it untouched). 16-byte placeholder on
    /// arches without the splice.
    pub mm_embeds: Buffer,
    /// `[max_m]` u32 — per-`mm_embeds`-row destination text-embedding row
    /// (`u32::MAX` = skip). Shared storage; overwritten per forward.
    pub mm_dst_rows: Buffer,
    /// `[max_m, ROT_DIM]` model-dtype — per-token MRoPE cos/sin override
    /// table (Qwen3.5-VL text decoder). Shared storage; the pool
    /// overwrites `contents()` each forward with the band-split rows the
    /// macro forward builds. Bound at the rope kernel's cos/sin slot in
    /// place of the static `WeightBundleKind::CosSin` cache when
    /// `W::MROPE_SECTION.is_some()`. 16-byte placeholder on 1D-rope arches.
    pub mrope_cos_sin: Buffer,
    /// i32 cu_seqlens for Qwen2.5-VL full-attention layers
    /// (`cu_seqlens_kind = 1`). 16-byte placeholder elsewhere.
    pub vision_cu_seqlens_full: Buffer,
    /// i32 cu_seqlens for Qwen2.5-VL window-attention layers
    /// (`cu_seqlens_kind = 2`).
    pub vision_cu_seqlens_window: Buffer,
    /// u32 merged-row window permutation (`EmbeddingGather` kind 0).
    pub vision_window_index: Buffer,
    /// u32 inverse permutation (`EmbeddingGather` kind 1).
    pub vision_reverse_indices: Buffer,
    /// u32 SigLIP positional-embedding indices (`PosEmbed` gather).
    pub vision_position_ids: Buffer,
}

impl RuntimeBindings {
    /// A layer's KV-cache group index. Falls back to group 0 when
    /// `layer_to_group` is empty (placeholder/empty runtimes that never
    /// dispatch a paged-cache kernel) or shorter than `layer` — uniform
    /// single-group behavior.
    #[inline]
    fn group_of(&self, layer: super::ids::LayerId) -> usize {
        self.layer_to_group
            .get(layer.get() as usize)
            .map(|&g| g as usize)
            .unwrap_or(0)
    }

    /// Resolve a [`RuntimeBindingKind`] to the buffer the worker
    /// should bake into the dispatch. Panics if a `KvCache*` layer index
    /// exceeds the held `Vec` length — that's a model-meta bug, not a
    /// data-driven failure mode.
    pub fn buffer_for(&self, kind: RuntimeBindingKind) -> &Buffer {
        match kind {
            RuntimeBindingKind::InputIds => &self.input_ids,
            RuntimeBindingKind::Positions => &self.positions,
            RuntimeBindingKind::SlotMapping { layer } => &self.slot_mappings[self.group_of(layer)],
            RuntimeBindingKind::CuSeqlensQ => &self.cu_seqlens_q,
            RuntimeBindingKind::SeqUsedK => &self.seq_used_k,
            RuntimeBindingKind::SpanIds => &self.span_ids,
            RuntimeBindingKind::BlockTable { layer } => &self.block_tables[self.group_of(layer)],
            RuntimeBindingKind::KvCacheK { layer } => &self.kv_cache_k[layer.get() as usize],
            RuntimeBindingKind::KvCacheV { layer } => &self.kv_cache_v[layer.get() as usize],
            RuntimeBindingKind::BlockUnrotatedFlags { layer } => {
                &self.block_unrotated_flags[layer.get() as usize]
            }
            // The tq commands are injected into EVERY tape (gated
            // OnlyIfTurboquant) but the bake resolves all bindings regardless of
            // the gate. When tq is absent (fp16/bf16 cache), fall back to a
            // present buffer — the gated command never dispatches, so the bound
            // buffer is inert. (Never panic here: it breaks the non-tq path.)
            RuntimeBindingKind::TqPackedK { layer } => self
                .tq
                .as_ref()
                .and_then(|t| t.packed_k.get(layer.get() as usize))
                .unwrap_or(&self.input_ids),
            RuntimeBindingKind::TqPackedV { layer } => self
                .tq
                .as_ref()
                .and_then(|t| t.packed_v.get(layer.get() as usize))
                .unwrap_or(&self.input_ids),
            RuntimeBindingKind::TqNormsK { layer } => self
                .tq
                .as_ref()
                .and_then(|t| t.norms_k.get(layer.get() as usize))
                .unwrap_or(&self.input_ids),
            RuntimeBindingKind::TqNormsV { layer } => self
                .tq
                .as_ref()
                .and_then(|t| t.norms_v.get(layer.get() as usize))
                .unwrap_or(&self.input_ids),
            RuntimeBindingKind::TqSigns => self
                .tq
                .as_ref()
                .map(|t| &t.signs)
                .unwrap_or(&self.input_ids),
            RuntimeBindingKind::TqBoundaries => self
                .tq
                .as_ref()
                .map(|t| &t.boundaries)
                .unwrap_or(&self.input_ids),
            RuntimeBindingKind::TqCentroids => self
                .tq
                .as_ref()
                .map(|t| &t.centroids)
                .unwrap_or(&self.input_ids),
            RuntimeBindingKind::NumTokensU32 => &self.num_tokens_u32,
            RuntimeBindingKind::NumSeqsU32 => &self.num_sample_rows_u32,
            RuntimeBindingKind::SampleIndices => &self.sample_indices,
            RuntimeBindingKind::GdnConvState { layer } => {
                &self.gdn_state_conv[layer.get() as usize]
            }
            RuntimeBindingKind::GdnSsmState { layer } => &self.gdn_state_ssm[layer.get() as usize],
            RuntimeBindingKind::GdnStateIndices => &self.gdn_state_indices,
            RuntimeBindingKind::GdnIsFresh => &self.gdn_is_fresh,
            RuntimeBindingKind::VisionRopeFreqs => &self.vision_rope_freqs,
            RuntimeBindingKind::Pixels => &self.pixels,
            RuntimeBindingKind::VisionPosEmbeds => &self.vision_pos_embeds,
            RuntimeBindingKind::MmEmbeds => &self.mm_embeds,
            RuntimeBindingKind::MmDstRows => &self.mm_dst_rows,
            RuntimeBindingKind::MropeCosSin => &self.mrope_cos_sin,
            RuntimeBindingKind::VisionCuSeqlensFull => &self.vision_cu_seqlens_full,
            RuntimeBindingKind::VisionCuSeqlensWindow => &self.vision_cu_seqlens_window,
            RuntimeBindingKind::VisionWindowIndex => &self.vision_window_index,
            RuntimeBindingKind::VisionReverseIndices => &self.vision_reverse_indices,
            RuntimeBindingKind::VisionPositionIds => &self.vision_position_ids,
        }
    }
}

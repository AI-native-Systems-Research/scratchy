// SPDX-License-Identifier: Apache-2.0
//! Classified AST: every free variable reference is tagged as
//! [`ExternKind`], [`WeightId`], or [`LocalId`]. String identifiers
//! live only in the side tables ([`LocalTable`], [`WeightTable`]);
//! the program itself refers to everything by numeric ID.
//!
//! This is the last stage where the caller can still recover the
//! user's original identifiers (via the tables). Passes below here
//! work with IDs only.

use syn::Ident;

/// A local binding. Each assignment statement introduces a fresh
/// `LocalId` even if the target name was previously bound; reads
/// at a site resolve to the *most recent* LocalId for that name at
/// that site (straight-line SSA).
///
/// For-loop induction variables are LocalIds whose scope is the
/// loop body.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LocalId(pub u32);

/// A weight reference, identified by its dotted path. Two DSL
/// reads of `self_attn.q_proj[layer]` resolve to the same `WeightId`
/// (indexing is stored on the expression, not on the id).
pub use scratchy_subtile::handoff::WeightId;

pub use scratchy_subtile::handoff::UnrollIndex;

/// The fixed enum of non-weight parameters. Every model uses the
/// same names and shapes for these; the `#[forward]` /
/// `#[vision_forward]` macros know about them by name.
///
/// Two prelude families share the enum:
///   - **Decoder** (text-side `#[forward]`): `InputIds`, `Positions`,
///     `Rotary`, `RotaryLocal`, `BlockTable`, `KvCache`.
///   - **Vision** (image-side `#[vision_forward]`): `Pixels`,
///     `CuSeqlens`, `Cos`, `Sin`, `GridThw`, `MaxSeqlen`.
///
/// The two preludes are disjoint by design — a vision body can't
/// read `kv_cache` and a decoder body can't read `pixels`. Which
/// names a given DSL body can name is selected by the [`Prelude`]
/// passed into [`crate::classify::classify`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExternKind {
    InputIds,
    Positions,
    Rotary,
    /// Alternate rotary cache for architectures with dual RoPE bases
    /// (e.g. Gemma3's `rope_local_base_freq` for sliding-attention
    /// layers). Lives on the Weights struct, not ForwardCtx.
    RotaryLocal,
    BlockTable,
    KvCache,
    /// Vision-encoder pixel/patch tensor: per-row patch of shape
    /// `[total_l, in_chans * temporal_patch_size * patch_size²]`
    /// produced by the host-side `patches_from_normalized_chw`
    /// helper. Replaces the decoder's `InputIds` as the encoder's
    /// data input.
    Pixels,
    /// Variable-length attention cumulative-seqlens index, shape
    /// `[num_images + 1]` of i32. Vision encoders run varlen
    /// flash-attn over a concatenated multi-image batch — `CuSeqlens`
    /// is the ragged-batch boundary index, analogous to the paged
    /// `BlockTable` for decoder attention but flat (no paging).
    CuSeqlens,
    /// 2D RoPE cos table, shape `[total_l, head_dim/2]`. Built
    /// host-side from `grid_thw` and uploaded as bf16. Distinct from
    /// decoder `Rotary` because vision RoPE has no positional
    /// extern (positions live in `cos`/`sin` themselves) and is
    /// applied via `vision_rope` rather than `rope_append`.
    Cos,
    /// 2D RoPE sin table, shape mirrors [`Cos`].
    Sin,
    /// Per-image grid `(t, h, w)` triples, shape `[num_images, 3]`
    /// of u32. Drives host-side cu_seqlens / cos / sin builders
    /// (already lifted to `scratchy-vision`); kept as an extern so
    /// future ops like a `windowed_varlen_attention` can read it
    /// directly when the windowed-block dispatch is DSL-visible.
    GridThw,
    /// Scalar i32 — the maximum per-image post-patch token count in
    /// the current batch. Sized as a kernel input to varlen flash-
    /// attn for shared-mem tile sizing. Shape `[]` (rank-0 / scalar).
    MaxSeqlen,
    /// Variable-length attention cu_seqlens for the **full image-frame**
    /// segmentation. Qwen2.5-VL runs full-frame attention for layers
    /// in `fullatt_block_indexes = [7, 15, 23, 31]` — each image is one
    /// segment, so the full-frame cu_seqlens partitions the flat
    /// `[total_L]` tensor at image boundaries (same shape as `CuSeqlens`
    /// but materially different segmentation than the windowed variant).
    /// Shape opaque; rank-1 i32 in practice.
    CuSeqlensFull,
    /// Variable-length attention cu_seqlens for the **windowed**
    /// segmentation. Qwen2.5-VL bins post-merger cells into
    /// 112-px-edge spatial windows (`vit_merger_window_size = 4`
    /// merged cells per side) and emits one segment per window. Used
    /// by the 28-of-32 layers NOT in `fullatt_block_indexes`. Shape
    /// opaque; rank-1 i32.
    CuSeqlensWindow,
    /// Scalar usize — full-frame max segment length (max over image
    /// segments, post-spatial-merge). Mirrors [`MaxSeqlen`] but for
    /// the full-frame variant. Pairs with [`CuSeqlensFull`] in
    /// `varlen_attention` calls at fullatt-layer indices.
    MaxSeqlenFull,
    /// Scalar usize — windowed max segment length. Mirrors
    /// [`MaxSeqlenFull`] but for the windowed variant. Pairs with
    /// [`CuSeqlensWindow`] at non-fullatt layer indices.
    MaxSeqlenWindow,
    /// Per-merged-cell permutation `[L / spatial_merge_size²]` of u32:
    /// natural→window-grouped order. Drives the entry-side
    /// `embedding_gather(x, window_index)` that re-orders tokens so
    /// each varlen-attention window sees a contiguous segment. Built
    /// host-side per request from `grid_thw + window_size`. Opaque
    /// rank-1.
    WindowIndex,
    /// Inverse of [`WindowIndex`] — per-merged-cell permutation
    /// `[L / spatial_merge_size²]` of u32 mapping window-grouped order
    /// back to natural row order. Drives the post-merger-MLP
    /// `embedding_gather(merged, reverse_indices)` that unpermutes
    /// before the splice into the language-model embedding stream.
    /// Opaque rank-1.
    ReverseIndices,
    /// Per-row position index for the SigLIP-style learned position
    /// embedding lookup, shape `[num_tokens]` u32. Built host-side as
    /// `[0..num_pos, 0..num_pos, ...]` (num_pos = `vision_num_positions`)
    /// repeated for each image in the batch — fixed-size SigLIP at
    /// 896² always has `num_tokens % num_pos == 0`. Drives the
    /// vision-prelude `pos_embed(position_ids, embeddings.position_embedding)`
    /// that gathers positional rows from a learned table the same way
    /// `embed(input_ids, embed_tokens)` does on the decoder side. Used
    /// by Gemma3-MM (SigLIP) and any future tower with a learned
    /// positional embedding (LLaVA / InternVL); bypasses the
    /// `Embed` OpKind because that one anchors on `vocab_size` /
    /// `hidden_size` bounds that don't apply to a vision tower.
    PositionIds,
    /// Qwen3.5-VL learned positional embedding, already interpolated
    /// host-side, shape `[num_tokens, vision_embed_dim]` model-dtype.
    /// Qwen3.5-VL's pos-embed is a `fast_pos_embed_interpolate`
    /// (4-corner bilinear over a 48×48 learned grid) that is far
    /// cheaper to compute on the host than to express as a kernel —
    /// so unlike Gemma3's `PositionIds`→`pos_embed` row-gather, this
    /// arch ships the *result* as an extern and the DSL just
    /// `add(pos_embeds, hidden_states)` after patch_embed. Materialized
    /// into a tile by `vision_lowering::materialize_pos_embeds`
    /// (mirrors [`Pixels`]).
    PosEmbeds,
}

/// Which DSL prelude (extern + op name set) is in scope when a
/// classified body is built. Every `#[forward]` body uses
/// [`Prelude::Decoder`]; every `#[vision_forward]` body uses
/// [`Prelude::Vision`].
///
/// The two preludes are disjoint, but the downstream IR
/// ([`Program`], [`crate::fuf::Fuf`]) shares one type and one
/// pipeline — preludes only influence what NAMES the parser /
/// classifier accept on input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Prelude {
    /// Text decoder: input_ids / positions / rotary / kv_cache /
    /// block_table externs; full text-side OpKind set.
    #[default]
    Decoder,
    /// Vision encoder: pixels / cu_seqlens / cos / sin / grid_thw /
    /// max_seqlen externs; vision-extended OpKind set
    /// (VarlenAttention / VisionRope / QuickGelu / GeluErf
    /// reachable in addition to the shared core).
    Vision,
}

impl ExternKind {
    /// Prelude-aware extern lookup. Decoder bodies see the text-side
    /// extern set; vision bodies see the vision-side set. The two
    /// sets are disjoint — `pixels` is unrecognized in a decoder
    /// body and `input_ids` is unrecognized in a vision body, both
    /// raising classify-time errors per the strict-DSL rule.
    pub fn from_name_for(name: &str, prelude: Prelude) -> Option<Self> {
        match (prelude, name) {
            (Prelude::Decoder, "input_ids") => Some(Self::InputIds),
            (Prelude::Decoder, "positions") => Some(Self::Positions),
            (Prelude::Decoder, "rotary") => Some(Self::Rotary),
            (Prelude::Decoder, "rotary_local") => Some(Self::RotaryLocal),
            (Prelude::Decoder, "block_table") => Some(Self::BlockTable),
            (Prelude::Decoder, "kv_cache") => Some(Self::KvCache),
            (Prelude::Vision, "pixels") => Some(Self::Pixels),
            (Prelude::Vision, "cu_seqlens") => Some(Self::CuSeqlens),
            (Prelude::Vision, "cos") => Some(Self::Cos),
            (Prelude::Vision, "sin") => Some(Self::Sin),
            (Prelude::Vision, "grid_thw") => Some(Self::GridThw),
            (Prelude::Vision, "max_seqlen") => Some(Self::MaxSeqlen),
            (Prelude::Vision, "cu_seqlens_full") => Some(Self::CuSeqlensFull),
            (Prelude::Vision, "cu_seqlens_window") => Some(Self::CuSeqlensWindow),
            (Prelude::Vision, "max_seqlen_full") => Some(Self::MaxSeqlenFull),
            (Prelude::Vision, "max_seqlen_window") => Some(Self::MaxSeqlenWindow),
            (Prelude::Vision, "window_index") => Some(Self::WindowIndex),
            (Prelude::Vision, "reverse_indices") => Some(Self::ReverseIndices),
            (Prelude::Vision, "position_ids") => Some(Self::PositionIds),
            (Prelude::Vision, "pos_embeds") => Some(Self::PosEmbeds),
            _ => None,
        }
    }
}

/// The fixed enum of DSL op kinds. One variant per op, no
/// tile-kind sub-variants. Extending the DSL with a new op means
/// adding one variant here, one shape signature in Phase 4, and
/// one kernel implementation — nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpKind {
    Embed,
    RmsNorm,
    Gemm,
    RopeAppend,
    /// RoPE that pairs adjacent elements `(2i, 2i+1)` for rotation
    /// instead of NeoX's `(i, i + half_dim)`. Different element
    /// pairing → genuinely different math, hence its own variant
    /// rather than a flag on `RopeAppend`. Used by Cohere's CommandR
    /// family.
    RopeAppendInterleaved,
    Attention,
    /// Same signature as `Attention`; picked by the DSL body at
    /// sliding-window attention layers. The distinction is carried
    /// through the FUF so the solver can match distinct Impls
    /// (dense flash-attn vs. window-masked flash-attn).
    SlidingAttention,
    /// Variable-length attention used by vision encoders (Qwen2-VL,
    /// Qwen2.5-VL, ViT-style towers). Same q/k/v shape as `Attention`
    /// but with a `cu_seqlens` ragged-batch index instead of a paged
    /// `kv_cache + block_table`, and a `max_seqlen` scalar that the
    /// kernel uses to size shared-mem tiles. Inputs:
    /// `(q, k, v, cu_seqlens, max_seqlen)`. Shape-preserving on q.
    /// Distinct OpKind so vision-side Impls (FlashAttention varlen,
    /// windowed varlen) match without fighting text-side Attention's
    /// heads-layout anchoring.
    VarlenAttention,
    Silu,
    /// Gaussian-Error Linear Unit, tanh approximation:
    /// `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`.
    /// Distinct from `GeluErf` and `QuickGelu` — the three differ in
    /// numerics. Locked to tanh-form by its current consumers (the
    /// `(Gelu, Mul)` fusion patterns in Gemma2/3 MLPs).
    Gelu,
    /// Quick-GELU: `x * sigmoid(1.702 * x)`. Used by Qwen2-VL's vision
    /// MLP and CLIP/ViT-style towers. Unary elementwise, shape-preserving.
    /// Numerically distinct from `Gelu` (tanh) and `GeluErf` (erf).
    QuickGelu,
    /// Erf-form GELU: `0.5 * x * (1 + erf(x / sqrt(2)))`. Used by
    /// SigLIP, BERT, and other vision/text towers that ship their
    /// MLP weights calibrated for the exact-erf form. Unary
    /// elementwise, shape-preserving. Numerically distinct from
    /// tanh-form `Gelu` and `QuickGelu`.
    GeluErf,
    /// Tanh-based soft-cap: `y = cap * tanh(x / cap)`. Unary
    /// elementwise with an additional scalar argument; shape-
    /// preserving. Used at logit exit for architectures that cap
    /// large pre-softmax magnitudes.
    TanhSoftCap,
    /// Unit-gain RMSNorm — no learnable scale (mlx `RMSNormNoScale`).
    /// Gemma4 `v_norm`: V is rms-normalized per head before the cache
    /// write with NO weight on disk. DSL `rmsnorm_unit(x)`; the Impl
    /// bakes the per-head reduction width from the attention class
    /// (sliding vs global) at fan_out time.
    RmsNormUnit,
    /// Multiply by a loaded `[1]`-shaped weight (Gemma4 `layer_scalar`
    /// applied to the hidden state at the end of every layer). DSL
    /// `scalar_weight_mul(x, layer_scalar[layer])`. Distinct from
    /// `ScalarMul` (compile-time constant) and `Mul` (same-shape
    /// elementwise).
    ScalarWeightMul,
    Add,
    /// Tensor-tensor elementwise subtract: `sub(x, y) -> x - y`.
    /// Shape-preserving like `Add`. Exists as a math primitive so
    /// the LayerNorm pattern can be written as
    /// `(mean, sub, rmsnorm)` and claimed by a fusion Impl rather
    /// than the DSL hiding the centering math behind a `layer_norm`
    /// opcode. Like `Silu` / `Mul`, has no singleton Impl — every
    /// `Sub` tile must be claimed by a fusion Impl or the solver
    /// reports `UnclaimedTile`.
    Sub,
    /// Last-dim reduction: `mean(x: [..., D]) -> [..., D]`. The output
    /// is shape-preserving (rather than rank-reducing) because Mean
    /// only ever appears inside the `(mean, sub, rmsnorm)` fusion
    /// pattern claimed by the LayerNorm Impl — the produced tile is
    /// never actually materialized as a standalone kernel, so the
    /// shape system records identity to keep the downstream `sub(x,
    /// mean_x)` legal under same-shape binary elementwise unification.
    /// Like `Silu` / `Mul`, has no singleton Impl.
    Mean,
    /// Tensor-parallel all-reduce-sum across `tp_world_size` ranks,
    /// in place. Identity-shape: `(x: [...]) -> [...]`. Never appears
    /// in any per-arch DSL — produced exclusively by the lowering
    /// pass that inserts one of these after every `Gemm` whose weight
    /// has shard-kind `ShardDim1` (row-parallel: `o_proj`,
    /// `down_proj`). At `tp_world_size = 1` the pass is a no-op so no
    /// FUF carries this op kind. Lowers to `Instruction::AllReduce`
    /// (gated under the `nccl` feature on `scratchy-forward-compiler`).
    AllReduce,
    /// Tensor-parallel all-gather along the LAST dim of one input
    /// tile across `tp_world_size` ranks. Output shape is the input
    /// shape with the last dim multiplied by `tp_world_size`. Like
    /// `AllReduce`, never appears in any DSL — produced exclusively
    /// by the lowering pass at tp>1. Inserted after the `Gemm` whose
    /// weight is `lm_head` (vocab-parallel `ShardDim0`): the per-rank
    /// matmul produces partial logits `[N, vocab/tp]` and the
    /// AllGather reassembles `[N, vocab]` for the sampler. Mirrors
    /// Python vLLM's `tensor_model_parallel_all_gather` on the
    /// `LogitsProcessor` path. Lowers to `Instruction::AllGather`.
    AllGather,
    /// Multimodal embed splice: D2D-copies projected vision-encoder
    /// embeddings into the placeholder rows of the post-embed hidden
    /// states. Identity-shape in-place mutation: `(x: [...]) -> [...]`.
    /// Never appears in any per-arch DSL — produced exclusively by
    /// `tp_lowering::insert_mm_splices`, which walks the FUF post-
    /// `insert_all_reduces` and appends one `MmEmbedSplice` after
    /// every `OpKind::Embed` (or after the `AllReduce` the tp pass
    /// chained onto it). Runs on every rank; at runtime the op is a
    /// no-op for text-only batches (empty `ForwardCtx::embed_patches`).
    ///
    /// Lives at the TP pass's layer (not the DSL) so the splice runs
    /// AFTER the vocab-parallel Embed's AllReduce-sum at tp>1, where
    /// otherwise each rank would overwrite the patch rows and the
    /// AllReduce would multiply the splice by `tp_world_size`. At
    /// tp=1 no AllReduce fires, and the splice sits directly on the
    /// Embed output — same semantics as the pre-refactor
    /// `Instruction::Embed::eval` inline splice.
    MmEmbedSplice,
    /// Broadcast-add of a learned per-feature bias vector across the
    /// batch/token dimensions: `bias_add(x: [..., D], b: [D]) -> [..., D]`.
    /// Semantically distinct from `Add` — `Add` is same-shape
    /// elementwise (residual stream), `BiasAdd` is a vector
    /// broadcast-add (affine-transform completion). Kept as its own
    /// op so fusion patterns that fold bias into a GEMM epilog
    /// (`(Gemm, BiasAdd)` → cuBLAS `gemm_bias`) can match distinctly
    /// from patterns that consume residual `Add`.
    BiasAdd,
    /// Elementwise multiplication. Produced by the DSL's `*`
    /// operator (e.g. `gate * up` in the SwiGLU MLP). Not reachable
    /// from `from_name` because `*` is a binary operator at the
    /// parse level rather than a named call.
    Mul,
    /// Shape view — reinterprets a tensor under a different rank
    /// without moving or copying data. Element count is preserved;
    /// the codegen maps to `OwnedTensor::reshape` / `TensorView::reshape`
    /// (metadata-only). Today Reshape tiles are **synthesized by
    /// shape inference** when an op expects a factor of the
    /// producer's last dim (e.g. per-head rmsnorm expects `[D]` but
    /// upstream gemm produces `[..., heads * D]`). The target shape
    /// is stored in [`Program::reshape_targets`] keyed by the new
    /// local's id; inference re-reads it when typechecking the
    /// synthesized statement. Not yet exposed to DSL authors via
    /// `from_name` — add an entry there if a future pattern needs
    /// explicit user-written reshape.
    Reshape,
    /// Vision RoPE pair-rotation: `(q', k') = vision_rope(q, k, cos, sin)`.
    /// 2-target tuple-returning; both outputs are shape-preserving on
    /// their respective inputs. Distinct from `RopeAppend` because it
    /// has no `kv_cache` extern (vision encoders have no KV cache —
    /// the encoder runs as one-shot prefill) and no `positions`
    /// extern (vision RoPE indexes off `grid_thw`-derived row/col
    /// positions baked into `cos`/`sin`). Used by Qwen2-VL and
    /// Qwen2.5-VL vision towers; expected reuse by future ViT-RoPE
    /// architectures.
    VisionRope,
    /// MLA kv_a split: decomposes `[T, kv_lora_rank + qk_rope_head_dim]`
    /// into `(kv_latent: [T, kv_lora_rank], k_pe: [T, qk_rope_head_dim])`.
    /// DSL form: `(kv_latent, k_pe) = mla_split(kv_a)`. Tuple-returning;
    /// the 2-target binding is handled in `Stmt::AssignTuple`. Used
    /// exclusively by DeepSeek V2/V3.
    MlaSplit,
    /// MLA full attention: applies interleaved RoPE to q_pe / k_pe, writes
    /// compressed KV to paged cache, assembles full K and V from cached
    /// kv_b + k_pe, runs flash attention, then slices the output to
    /// `v_head_dim`. Consumes `q, kv_b, k_pe` plus externs
    /// `(positions, rotary, kv_cache[layer], block_table)`. Output:
    /// `[T, num_attention_heads * v_head_dim]`. Used by DeepSeek V2/V3.
    MlaAttention,
    /// MoE block — gate routing + top-K fused GEMM for routed experts
    /// plus an optional shared expert. The DSL form is the same across
    /// every MoE arch: `moe_out = moe_block(x, moe[layer])`. Mirrors
    /// HF Python naming (`MixtralSparseMoeBlock`,
    /// `Qwen2MoeSparseMoeBlock`, `DeepseekV2MoE`). The forward math
    /// (no shared / sigmoid-gated shared / scaled-plain-add shared)
    /// is encoded in the loaded layer struct (`FusedMoELayer` /
    /// `SharedFusedMoELayer` / `DeepSeekV2MoELayer` + their FP8/Ggml
    /// flavors), and each is claimed by a distinct `Implementation`
    /// keyed on the rust_type fingerprint of the `moe[layer]` weight
    /// accessor. Shape-preserving.
    Moe,
    /// Gemma-4 fused MoE block (`Gemma4ForConditionalGeneration` 26B-A4B).
    /// DSL form: `h2 = gemma_moe(router_in, expert_in, router[layer],
    /// experts.switch_glu[layer])`. Unlike [`OpKind::Moe`] it takes TWO
    /// activation inputs (the router runs on the post-attention residual
    /// `router_in`; the experts on the pre-FF2-normed `expert_in`) and TWO
    /// weight bundles (a `GemmaRouterLayer` at base `router` carrying the
    /// 8-bit dequant'd dense gate + per-expert score scale + RMSNorm gain,
    /// and a router-less `SwitchGluExpertsLayer` at base `experts.switch_glu`
    /// carrying the 4-bit GeGLU experts). Routing = folded-temperature
    /// softmax over the top-k scores × per-expert-scale; experts use
    /// GeGLU (not SiLU). No shared expert. Output = `expert_in` shape.
    /// Metal-only; claimed by `MetalGemmaMoeImpl`.
    GemmaMoe,
    /// Gated-DeltaNet linear attention (Qwen3.5 / Qwen3-Next). One coarse op
    /// orchestrating the surviving `gdn_*` kernels: causal conv1d(+SiLU) →
    /// split q/k/v → input-dependent gating → recurrent delta-rule scan →
    /// gated RMSNorm. DSL form:
    /// `core = gated_delta_net(qkv, z, a, b, linear_attn[layer])` where
    /// `qkv = in_proj_qkv` `[T, conv_dim]`, `z = in_proj_z` `[T, value_dim]`,
    /// `a/b = in_proj_{a,b}` `[T, num_v_heads]`, and `linear_attn[layer]`
    /// resolves to a `GatedDeltaNetLayer` (conv1d / A_log / dt_bias / norm).
    /// The recurrent + conv state is the ambient `ForwardCtx::gdn_state`
    /// (non-paged, one slot/seq) selected by `gdn_state_indices`, read at
    /// the `layer` carried on the `Instruction`. Output: `[T, value_dim]`
    /// (feeds `out_proj`). Used by Qwen3.5 / Qwen3-Next linear layers.
    GatedDeltaNet,
    /// Qwen3.5 attention output-gate split. Deinterleaves the DOUBLED
    /// `q_proj` output `[T, num_heads*2*head_dim]` (viewed `[T, nh, 2, hd]`)
    /// into `query` (`[:,:,0,:]`) and `gate` (`[:,:,1,:]`), each
    /// `[T, num_heads*head_dim]`. Tuple-returning (`(q, gate) = gate_split(qg)`);
    /// the 2-target binding is handled in `Stmt::AssignTuple` like `MlaSplit`.
    /// Used only by Qwen3.5 full-attention layers (`attn_output_gate=True`).
    GateSplit,
    /// Qwen3.5 attention output gate application: `out = attn * sigmoid(gate)`.
    /// Shape-preserving 2-input elementwise. A fused op because bare `*` and
    /// `sigmoid` are not DSL-callable (`Silu`/`Mul` are synthesis-only).
    GateApply,
    /// Qwen3.5-MoE shared-expert combine: `out = routed + shared_y *
    /// sigmoid(g)` with `g` `[T, 1]` broadcast across the hidden axis.
    /// 3-input elementwise-with-row-broadcast; a fused op because bare
    /// `*`/`sigmoid` aren't DSL-callable and `Mul`'s shape sig unifies
    /// dims, so `[T, 1] × [T, H]` cannot be expressed with `mul`.
    GateScale,
    /// Vision-prelude pixels materialization. Synthesized by
    /// `vision_lowering::materialize_pixels` between `fuf::unroll`
    /// and the solver: takes zero FUF inputs and produces a single
    /// rank-2 output `[num_tokens, vision_in_features]` whose runtime
    /// value is `ctx.fwd.pixels` wrapped into a tile.
    ///
    /// Mirrors the role `EmbedRefImpl` plays for `input_ids` on the
    /// decoder side: every downstream vision Impl
    /// (`VarlenAttention` / `VisionRope` / `QuickGelu` / `GeluErf`)
    /// reads its first input as `FufInput::Tile { id, slot }`, so the
    /// extern → tile transition has to happen exactly once,
    /// up-front, rather than being hand-unrolled into every per-Impl
    /// `fan_out`.
    ///
    /// No DSL surface — `from_name` deliberately omits it. The
    /// lowering pass writes `outputs[0]` directly (same pattern as
    /// `AllGather` and `MmEmbedSplice`), so `apply_signature` rejects
    /// this OpKind.
    LoadPixels,
    /// Vision-prelude `pos_embeds` extern → tile materialization
    /// (Qwen3.5-VL). The exact sibling of [`Self::LoadPixels`]:
    /// synthesized by `vision_lowering::materialize_pos_embeds`, takes
    /// zero FUF inputs, produces a single rank-2 output
    /// `[num_tokens, vision_embed_dim]` whose runtime value is the
    /// host-interpolated `ctx.fwd.pos_embeds` view. No DSL surface;
    /// `apply_signature` rejects it (output shape written directly).
    LoadPosEmbeds,
    /// Row-permutation gather: `out = embedding_gather(x, indices)`.
    /// `x` is a rank-2 tile `[L, N]` (any inner-dim factorization);
    /// `indices` is a vision-prelude extern of `[`[`ExternKind::WindowIndex`]`
    /// or [`ExternKind::ReverseIndices`]`]` (rank-1 u32 of length L). Output
    /// is `[L, N]` with row `i` set to `x[indices[i]]`. Distinct OpKind
    /// from `Embed` (which gathers off a learned embedding table) and
    /// from `Reshape` (which is metadata-only) — this is a real
    /// permutation kernel call. Used by Qwen2.5-VL's window-attention
    /// dispatch: tokens + cos/sin tables are gather-permuted into
    /// window order on encoder entry, and the merger output is
    /// gather-permuted back to natural order at exit.
    EmbeddingGather,
    /// 2-D non-overlapping average-pool over the patch grid:
    /// `out = avg_pool_2d(x: [L, e]) -> [L / vision_pool_factor, e]`.
    ///
    /// Models the Gemma3-MM SigLIP→text projector, which reshapes the
    /// flat patch grid `[ph², e]` to `[ph, ph, e]`, applies a stride-k
    /// k×k AvgPool2d (k = `vision_pool_kernel`), and flattens back to
    /// `[(ph/k)², e]`. The k² spatial cells contributing to each output
    /// row are NOT contiguous in row-major `[ph², e]` — they're spread
    /// across k different rows separated by `ph` row-strides — so the
    /// pool can't be expressed as `reshape + mean + gather` over
    /// existing ops; it needs its own kernel that walks the 2-D grid.
    ///
    /// Shape: leading dim divides by `vision_pool_factor = k * k`
    /// (`Dim::Div`); trailing dim is preserved. The kernel reads
    /// `vision_patch_grid_side` and `vision_pool_kernel` off
    /// `CanonicalParams` so it can convert flat-row index → (row, col)
    /// and walk the k² source cells per output row.
    AvgPool2d,
    /// CLIP-class CLS-token strip: `strip_cls(x) -> [L-1, e]`. Removes
    /// the leading CLS row that LLaVA-1.5's CLIP encoder produces.
    /// Claimed by [`crate::impl_lib::StripClsImpl`].
    StripCls,
    /// Vision-tower learned positional embedding lookup:
    /// `pos_embed(position_ids, weight) -> [num_tokens, vision_embed_dim]`.
    /// Mirror of [`Self::Embed`] but anchored on
    /// `vision_num_positions` / `vision_embed_dim` instead of
    /// `vocab_size` / `hidden_size`. Used by SigLIP-style encoders
    /// (Gemma3-MM today, LLaVA / InternVL on the horizon) where the
    /// patch-embed Conv2d output gets a learned per-position bias
    /// added before the encoder blocks. The `position_ids` arg is the
    /// vision-prelude [`ExternKind::PositionIds`] extern; the runtime
    /// reuses `kernels::embedding_gather_masked` (the same kernel
    /// `Instruction::Embed` calls) reading `ctx.fwd.position_ids`.
    PosEmbed,
}

impl OpKind {
    /// Map a DSL op-call ident to its `OpKind`. Binary operators
    /// (currently just `*`) do not flow through this path.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "embed" => Some(Self::Embed),
            "rmsnorm" => Some(Self::RmsNorm),
            "gemm" => Some(Self::Gemm),
            "rope_append" => Some(Self::RopeAppend),
            "rope_append_interleaved" => Some(Self::RopeAppendInterleaved),
            "attention" => Some(Self::Attention),
            "sliding_attention" => Some(Self::SlidingAttention),
            // Vision-tower ops. Unambiguous names: the four below are
            // unused on the decoder side, so the lookup is shared with
            // the decoder prelude — a `#[forward]` body that wrote
            // `varlen_attention(...)` would parse but get rejected by
            // the decoder-side externs (cu_seqlens / max_seqlen are
            // only resolvable under `Prelude::Vision`). Pairs with the
            // shape signatures in `shape::sig_varlen_attention`,
            // `sig_vision_rope`, and `sig_unary_elementwise` (the two
            // GELU variants).
            "varlen_attention" => Some(Self::VarlenAttention),
            "vision_rope" => Some(Self::VisionRope),
            "quick_gelu" => Some(Self::QuickGelu),
            "gelu_erf" => Some(Self::GeluErf),
            "silu" => Some(Self::Silu),
            "gelu" => Some(Self::Gelu),
            "tanh_softcap" => Some(Self::TanhSoftCap),
            "rmsnorm_unit" => Some(Self::RmsNormUnit),
            "scalar_weight_mul" => Some(Self::ScalarWeightMul),
            "add" => Some(Self::Add),
            "sub" => Some(Self::Sub),
            "mean" => Some(Self::Mean),
            "bias_add" => Some(Self::BiasAdd),
            // `Reshape` is synthesized by shape inference, not DSL-
            // writable today. Intentionally not listed in `from_name`;
            // add the arm if a future pattern needs explicit reshape.
            "mla_split" => Some(Self::MlaSplit),
            "mla_attention" => Some(Self::MlaAttention),
            "moe_block" => Some(Self::Moe),
            "gemma_moe" => Some(Self::GemmaMoe),
            "gated_delta_net" => Some(Self::GatedDeltaNet),
            "gate_split" => Some(Self::GateSplit),
            "gate_apply" => Some(Self::GateApply),
            "gate_scale" => Some(Self::GateScale),
            "embedding_gather" => Some(Self::EmbeddingGather),
            "avg_pool_2d" => Some(Self::AvgPool2d),
            "strip_cls" => Some(Self::StripCls),
            "pos_embed" => Some(Self::PosEmbed),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::RmsNorm => "rmsnorm",
            Self::Gemm => "gemm",
            Self::RopeAppend => "rope_append",
            Self::RopeAppendInterleaved => "rope_append_interleaved",
            Self::Attention => "attention",
            Self::SlidingAttention => "sliding_attention",
            Self::VarlenAttention => "varlen_attention",
            Self::Silu => "silu",
            Self::Gelu => "gelu",
            Self::QuickGelu => "quick_gelu",
            Self::GeluErf => "gelu_erf",
            Self::VisionRope => "vision_rope",
            Self::TanhSoftCap => "tanh_softcap",
            Self::RmsNormUnit => "rmsnorm_unit",
            Self::ScalarWeightMul => "scalar_weight_mul",
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mean => "mean",
            Self::BiasAdd => "bias_add",
            Self::Mul => "mul",
            Self::Reshape => "reshape",
            Self::MlaSplit => "mla_split",
            Self::MlaAttention => "mla_attention",
            Self::Moe => "moe_block",
            Self::GemmaMoe => "gemma_moe",
            Self::GatedDeltaNet => "gated_delta_net",
            Self::GateSplit => "gate_split",
            Self::GateApply => "gate_apply",
            Self::GateScale => "gate_scale",
            // No DSL surface — produced only by the post-FUF lowering
            // pass at tp>1. `from_name` deliberately omits it so a
            // user can't write `all_reduce(...)` in a `#[forward]`
            // body; the canonical path is the lowering pass.
            Self::AllReduce => "all_reduce",
            // Same DSL-omission story as AllReduce — only the
            // lowering pass produces this op kind.
            Self::AllGather => "all_gather",
            // Lowering-pass-only op kind — see `OpKind::MmEmbedSplice`
            // doc-comment. No DSL surface.
            Self::MmEmbedSplice => "mm_embed_splice",
            // Vision lowering-pass-only op kind — see
            // `OpKind::LoadPixels` doc-comment. No DSL surface.
            Self::LoadPixels => "load_pixels",
            Self::LoadPosEmbeds => "load_pos_embeds",
            Self::EmbeddingGather => "embedding_gather",
            Self::AvgPool2d => "avg_pool_2d",
            Self::StripCls => "strip_cls",
            Self::PosEmbed => "pos_embed",
        }
    }
}

/// A classified DSL program.
#[derive(Clone, Debug, Default)]
pub struct Program {
    pub statements: Vec<Stmt>,
    /// Ident for each LocalId (for diagnostics and codegen only).
    pub locals: LocalTable,
    /// Path segments for each WeightId (for diagnostics and
    /// runtime weight lookup).
    pub weights: WeightTable,
    /// Target shapes for synthesized `Reshape` statements. Keyed by
    /// the LocalId of the reshape's output (the newly-introduced
    /// fresh local). Empty for DSL programs with no shape-mismatch
    /// recoveries.
    ///
    /// Populated by `shape::infer` when a per-axis-factor mismatch
    /// between a weight's declared shape and its consumer's inferred
    /// shape is detected (e.g. Qwen3's per-head `q_norm` of shape
    /// `[head_dim]` applied to a gemm output of shape `[..., heads *
    /// head_dim]`). The synthesizer inserts a `Reshape` Stmt and
    /// records the target shape here; the second inference pass
    /// reads this map to typecheck the synthesized statement.
    pub reshape_targets: std::collections::HashMap<LocalId, Vec<crate::shape::Dim>>,
    /// Prelude this program was classified under. `Decoder` for
    /// text-side `#[forward]`, `Vision` for `#[vision_forward]`.
    /// Threaded into codegen for prelude-specific decisions —
    /// safetensors prefix conventions (`model.layers.<L>.<x>` vs
    /// `visual.blocks.<L>.<x>`), reshape `nt` source, etc.
    pub prelude: Prelude,
    /// Decoder-side safetensors prefix to prepend to every text-decoder
    /// safetensors key. Populated by `compile_carrier` from the arch's
    /// representative config (`decoder_safetensors_prefix` JSON field).
    /// `None` for text-only and Qwen-style VL arches; `Some("language_model")`
    /// for Gemma3-MM-style multimodal where HF nests the text decoder
    /// under `language_model.<...>`.
    pub decoder_safetensors_prefix: Option<String>,
    /// DSL-leaf → disk-leaf renames (see
    /// `ModelParams::weight_leaf_renames`). Applied by
    /// `codegen::safetensors_prefix` after segment translation.
    pub weight_leaf_renames: Vec<(String, String)>,
}

/// Side table: `LocalId` → debug ident.
#[derive(Clone, Debug, Default)]
pub struct LocalTable {
    entries: Vec<Ident>,
}

impl LocalTable {
    pub fn push(&mut self, name: Ident) -> LocalId {
        let id = LocalId(self.entries.len() as u32);
        self.entries.push(name);
        id
    }

    pub fn name(&self, id: LocalId) -> &Ident {
        &self.entries[id.0 as usize]
    }
}

/// Side table: `WeightId` → dotted path segments (plain strings).
///
/// Path segments are stringified at intern time — `proc_macro2::Ident`
/// wraps rustc's thread-local symbol bridge, so reading an Ident off
/// the main thread (e.g. via `Ident::to_string`) panics. Storing
/// `Vec<String>` keeps the classified program Send-safe for the
/// macro's per-model rayon loop.
#[derive(Clone, Debug, Default)]
pub struct WeightTable {
    /// Invariant: paths are unique (interning).
    entries: Vec<Vec<String>>,
}

impl WeightTable {
    /// Insert a path, returning the assigned `WeightId`. Idempotent:
    /// two calls with paths of equal string segments return the same id.
    /// Accepts `Vec<Ident>` and stringifies — call this from the main
    /// thread during classify, while the proc-macro bridge is live.
    pub fn intern(&mut self, path: Vec<Ident>) -> WeightId {
        let path: Vec<String> = path.iter().map(|i| i.to_string()).collect();
        self.intern_str(path)
    }

    pub fn intern_str(&mut self, path: Vec<String>) -> WeightId {
        for (i, existing) in self.entries.iter().enumerate() {
            if existing == &path {
                return WeightId(i as u32);
            }
        }
        let id = WeightId(self.entries.len() as u32);
        self.entries.push(path);
        id
    }

    pub fn path(&self, id: WeightId) -> &[String] {
        &self.entries[id.0 as usize]
    }

    /// Whether any interned path lies strictly under the given segment
    /// prefix (a proper subtree — not the prefix node itself). Drives
    /// the single-owner rule: a struct accessor (e.g. the MoE `mlp`)
    /// must not also load weight paths the DSL claims as ordinary
    /// leaves (e.g. `mlp.shared_expert.*` when the model expresses the
    /// shared expert as DSL gemms + `gate_scale`).
    pub fn has_subtree(&self, prefix: &[&str]) -> bool {
        self.entries
            .iter()
            .any(|p| p.len() > prefix.len() && p.iter().zip(prefix).all(|(s, t)| s == t))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Test helper: find a weight id by its path segments as strings.
    #[cfg(test)]
    pub fn path_for_test(&self, segments: &[&str]) -> Option<WeightId> {
        self.entries.iter().enumerate().find_map(|(i, p)| {
            if p.len() == segments.len() && p.iter().zip(segments).all(|(s, t)| s == t) {
                Some(WeightId(i as u32))
            } else {
                None
            }
        })
    }
}

/// A statement in the classified program.
#[derive(Clone, Debug)]
pub enum Stmt {
    /// `target = value` where `target` is a fresh LocalId.
    Assign { target: LocalId, value: Expr },
    /// `(t0, t1, ...) = value` where each `t_i` is a fresh LocalId.
    AssignTuple { targets: Vec<LocalId>, value: Expr },
    /// `for ivar in 0..<bound> { body }`. `ivar` is a fresh LocalId
    /// scoped to the body.
    ///
    /// `loop_carry` enumerates the names that are bound both
    /// *before* the loop and *inside* the body. Each entry is
    /// `(outer, inner)` where `outer` is the LocalId of the outer
    /// binding that body reads see initially, and `inner` is the
    /// LocalId of the body's *last* write to that name. After each
    /// iteration, the unroller re-binds `outer`'s tile to `inner`'s
    /// tile so the next iteration's reads see the iteration's
    /// previous output.
    For {
        ivar: LocalId,
        start: Bound,
        end: Bound,
        body: Vec<Stmt>,
        loop_carry: Vec<(LocalId, LocalId)>,
    },
    /// `if <predicate> { then_body } else { else_body }`. The
    /// predicate is a compile-time-evaluable function of a loop
    /// induction variable and config constants — evaluated at
    /// unroll time, each unrolled iteration descends into exactly
    /// one arm.
    ///
    /// Both arms must bind the same set of names. For each name
    /// bound in either arm, `merge_carry` has one entry
    /// `(merge_id, then_final, else_final)`: reads after the If
    /// resolve to `merge_id`; at unroll time the unroller sets
    /// `local_to_tile[merge_id]` to whichever arm ran. An arm that
    /// doesn't bind the name reuses its pre-If binding's LocalId
    /// as the arm's "final" id.
    If {
        cond: BoolPred,
        then_body: Vec<Stmt>,
        else_body: Vec<Stmt>,
        merge_carry: Vec<(LocalId, LocalId, LocalId)>,
    },
}

/// Loop bound: either a literal integer or a symbolic identifier
/// that names a per-model bound (e.g. `num_hidden_layers`). The
/// Ident is preserved here because bound resolution happens later,
/// in Phase 3 when config.json values are loaded.
#[derive(Clone, Debug)]
pub enum Bound {
    Lit(u64),
    Sym(Ident),
}

/// Boolean predicate used as an `if` condition. The predicate
/// enum is deliberately closed and narrow — it exists to express
/// layer-indexed dispatch patterns (Gemma2 alternating
/// sliding/full attention, DeepSeek-V3 "first N layers dense")
/// without extending the expression IR with booleans or general
/// binary arithmetic. Evaluated only at unroll time against
/// concrete loop-var values.
#[derive(Clone, Debug)]
pub enum BoolPred {
    /// `ivar % divisor == remainder`.
    Modulo {
        ivar: LocalId,
        divisor: Bound,
        remainder: Bound,
    },
    /// `ivar % divisor != remainder`.
    NotModulo {
        ivar: LocalId,
        divisor: Bound,
        remainder: Bound,
    },
    /// `ivar < bound`.
    Less { ivar: LocalId, bound: Bound },
    /// `members.contains(&ivar)` — set membership over a closed list
    /// of integer literals. See [`crate::ast::BoolExpr::In`].
    In { ivar: LocalId, members: Vec<u64> },
}

/// A value-producing expression.
#[derive(Clone, Debug)]
pub enum Expr {
    /// Read of a local binding.
    Local(LocalId),
    /// Read of a non-weight parameter, optionally indexed by a
    /// local (the loop variable).
    Extern {
        kind: ExternKind,
        index: Option<LocalId>,
    },
    /// Read of a weight, optionally indexed by a local.
    Weight {
        id: WeightId,
        index: Option<LocalId>,
    },
    /// Op call.
    Call { op: OpKind, args: Vec<Expr> },
    /// Multiplication (`gate * up`). Tensor × tensor.
    Mul { lhs: Box<Expr>, rhs: Box<Expr> },
    /// Addition (`w + 1.0`) — the classifier resolves this to a
    /// tile-level `OpKind::Add` call with the scalar captured as
    /// `ScalarLit` inside `args`. See [`classify_expr`].
    ///
    /// A numeric scalar literal. Used as an operand to elementwise
    /// ops that admit a scalar broadcast (e.g. Gemma's `w + 1.0`).
    ScalarLit(f64),
    /// `sqrt(<bound_name>)` — unresolved compile-time scalar. The
    /// CFG builder resolves this to `ScalarLit(bounds[name].sqrt())`
    /// per-model (so the value becomes concrete before the FUF).
    SqrtBound(Ident),
    /// `scalar(<name>)` / `recip_scalar(<name>)` — unresolved
    /// compile-time scalar read from `ModelParams.scalars`. CFG
    /// builder folds to `ScalarLit(scalars[name])` (or its
    /// reciprocal) per-model.
    ConfigScalar { name: Ident, recip: bool },
}

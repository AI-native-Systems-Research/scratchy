// SPDX-License-Identifier: Apache-2.0
//! Universal interpreter instruction set.
//!
//! `Instruction` is a closed enum over every kernel-call shape any
//! solver-picked `Implementation` produces. The match in
//! `Instruction::eval` lives ONCE in this module — not per canonical,
//! not per arch. Each arm calls existing kernels in `scratchy-target-cuda`
//! directly. Per-canonical specialization flows through method-level
//! generics:
//!
//! 1. `eval<W: CanonicalParams>` reads model constants via
//!    `W::HEAD_DIM` etc. and weight tensors via the
//!    [`WeightAccessors`] trait — `ctx.wm.<kind>_at(bucket, op_idx,
//!    slot, layer)`. The proc-macro emits a per-arch
//!    `WeightAccessors` impl whose body matches on
//!    `(bucket, op_idx, slot)` to resolve the named field on
//!    `Weights`. The variant itself carries no weight info.
//! 2. [`CanonicalParams`] supertype — trait implemented per
//!    canonical with associated `const`s for model-wide values
//!    (`HEAD_DIM`, `INTERMEDIATE_SIZE`, `ATTN_SCALE`, …).
//!
//! Per canonical the macro emits: a `Weights` struct,
//! `impl CanonicalParams for Weights { … }`,
//! `impl WeightAccessors for Weights { … }`, static
//! `&[Instruction]` slices for backbone + lm_head per bucket, and a
//! 1-line forward shim that delegates to [`run`].

// File is ungated at the module level so the proc-macro can use
// `Instruction` without a backend feature. The `Instruction` enum
// + `CanonicalParams` / `WeightAccessors` traits compile under any
// feature combination — only `GpuTensor` (always available) plus the
// layer struct *type names* (also always available; methods are
// cuda-gated inside `scratchy-target-cuda/src/layers.rs`) are referenced.
//
// `MAX_DIMS` is referenced only inside the cuda eval body (e.g.
// `let mut shape = [0usize; MAX_DIMS];`) now that the `Reshape`
// variant moved with the enum to `scratchy-ir` — gated to cuda.
#[cfg(feature = "cuda")]
use scratchy_tensors::tensor::MAX_DIMS;
// The `Instruction` enum + `WeightAccessors` trait live in the cfg-free
// `scratchy-ir` crate; pull them into scope so the cuda eval body
// + the free fns resolve them. The in-crate layer structs the eval body
// names are referenced exclusively through fully-qualified paths
// (`crate::tensor::GpuTensor`, `crate::layers::LinearLayer`, etc.), so
// no `use` import is needed for those. Both `Instruction` and
// `WeightAccessors` are named only by the cuda eval body + runtime free
// fns (the latter for the `ctx.wm.<kind>_at(...)` method calls —
// `CanonicalParams`, which had the `: WeightAccessors` supertrait, is
// the third) — gated to cuda.
#[cfg(feature = "cuda")]
use scratchy_ir::Instruction;
//
// Cuda-only imports (eval body, runtime entry points) — gated
// individually below so the metal-feature / no-backend builds only
// pull in the cross-backend pieces above.
#[cfg(feature = "cuda")]
use crate::alloc::OwnedTensor;
#[cfg(feature = "cuda")]
use crate::attention_helpers as ah;
#[cfg(feature = "cuda")]
use crate::cutlass;
#[cfg(feature = "cuda")]
use crate::device::GpuDevice;
#[cfg(feature = "cuda")]
use crate::flashinfer;
#[cfg(feature = "cuda")]
use crate::forward_ctx::ForwardCtx;
#[cfg(feature = "cuda")]
use crate::kernels;
#[cfg(feature = "cuda")]
use crate::tile_table::{TileEntry, take_owned, tile_ref, view};
// Runtime `forward` extension traits — the eval body calls `w.forward(...)`
// (method syntax) on `&LinearLayer` / `&Fp8AnyLinear`, whose `forward`
// relocated off the neutral defs into `*Ops` traits in targets/cuda.
#[cfg(feature = "cuda")]
use crate::layers::{Bnb4bitOps, Fp8AnyOps, LinearLayerOps, MarlinOps};
// MoE `forward` extension traits — the eval arms call `w.forward(...)` on the
// enum/wrapper MoE types, whose `forward` relocated into `*Ops` traits in
// targets/cuda.
#[cfg(feature = "cuda")]
use crate::layers_moe::{
    DeepSeekV2Fp8BlockMoEOps, DeepSeekV2GgmlMoEOps, DeepSeekV2MoEOps, FusedMoEOps,
    SharedFusedMoEOps,
};

// `CanonicalParams` moved to the cfg-free `scratchy-ir` crate
// (the 2 metal-only `METAL_DTYPE` / `SCALE_DTYPE` consts now name the
// relocated `scratchy_tensors::{MetalDtype, ScaleDtype}` enums, so the
// trait is fully cfg-neutral). Pulled back into scope here so the cuda
// eval body, `run` / `run_backbone`, and the per-arch consts the eval
// reads (`W::HEAD_DIM` etc.) resolve unchanged. Named only by the cuda
// eval / runtime free fns — gated to cuda to avoid an unused-import
// warning under the metal-feature / no-backend builds.
#[cfg(feature = "cuda")]
use scratchy_ir::CanonicalParams;

/// Runtime state passed by `&mut` into every `op.eval(&mut ctx)`.
/// Constants live on `W: CanonicalParams`, NOT here.
///
/// **Reshape / RopeAppend `out_slot != in_slot` invariant.** Both
/// instructions write `TileEntry::Reshaped { ref_slot, tensor }` at
/// `out_slot`. The codegen `colored_slot_map` guarantees `out_slot !=
/// in_slot` for every Impl that declares `output_alias_is_view = true`
/// (today: `ReshapeRefImpl`, `RopeAppendRefImpl`) — so the overwrite
/// at `tiles[out_slot]` never drops the upstream's `OwnedTensor` at
/// `tiles[in_slot]`. The G.7(e.tail.2) bug fix had a runtime safety
/// net (`pinned_owned: Vec<OwnedTensor>`) that pinned the prior Owned
/// past the overwrite; once the codegen invariant is in place, the
/// safety net is unnecessary and was deleted in this commit.
#[cfg(feature = "cuda")]
pub struct InterpreterCtx<'a, W> {
    pub wm: &'a W,
    pub tiles: &'a mut Vec<Option<TileEntry>>,
    pub fwd: &'a ForwardCtx<'a>,
    pub device: &'a mut GpuDevice,
    /// Iter index of the enclosing `Op::Loop`, else 0.
    pub layer_offset: u32,
    /// Absolute KV-cache layer of the most recent KV WRITE (RopeAppend etc.),
    /// i.e. `layer_offset + op_layer`. Set by the KV-write op so a following
    /// paged-attention op that lacks its own layer operand
    /// (AttentionPrefillContiguous / SlidingAttentionPrefillContiguous — used
    /// on the chunked-prefill CONTINUATION path) reads the RIGHT cache layer.
    /// On UNROLLED arches (gemma4: per-layer global/sliding branching)
    /// `layer_offset` is 0 and the layer lives only in the op operand, so those
    /// operand-less attention ops would otherwise read cache layer 0 → garbage.
    /// The KV write always precedes the attention within a decoder layer, so
    /// this is the correct layer by construction.
    pub kv_layer: u32,
    /// `OwnedTensor`s removed from `tiles` to make room for a
    /// `Reshaped` at the same slot index. Keeping them here pins
    /// the underlying GPU memory for the rest of the run, so any
    /// `Reshaped { ref_slot, tensor }` that aliases this storage
    /// continues to point at live memory.
    ///
    /// **Why this exists.** When `Instruction::Reshape(in_slot,
    /// out_slot, ...)` is emitted with `out_slot == in_slot`, the
    /// previous `TileEntry::Owned(OwnedTensor)` at that slot would
    /// be dropped on overwrite — freeing its block back to the
    /// caching allocator. The new `TileEntry::Reshaped` still holds
    /// the original GPU pointer in its `tensor` field, but that
    /// memory is now in the free pool; subsequent `alloc_tensor`
    /// calls in the same forward (e.g., a downstream attention
    /// kernel's output) hand the same address back, the kernel
    /// writes there while still reading the Reshaped, and the
    /// supposedly-still-live "input" tile sees torn writes (NaN /
    /// extreme magnitudes). Pinning here is the minimum-invasive
    /// fix: the codegen contract that `Reshaped::ref_slot` "pins
    /// the slot whose `OwnedTensor` actually owns the storage" only
    /// works when the OwnedTensor still lives at `ref_slot`; in the
    /// in-place case it has already been overwritten. Stash it.
    pub pinned_owned: Vec<OwnedTensor>,
}

/// Assert a runtime weight tensor's `[N, K]` shape matches the
/// codegen-time constants the `Instruction` was emitted with. The
/// constants come from the FUF's `eval_shape` at solve time; the
/// runtime tensor is whatever `WeightAccessors` resolved to from the loaded
/// safetensors. A mismatch means the loader produced a weight whose
/// shape disagrees with the model the solver compiled against —
/// silent shape drift here corrupts every output. Real `assert!`,
/// not `debug_assert!`, because release builds need to fail loud
/// rather than march on with a K-mismatch.
///
/// At tp>1 the codegen-time (n, k) reflects the *unsharded* model
/// because shape inference unifies `num_q_heads * head_dim` with
/// `hidden_size` (numerically equal in most arches), losing the
/// distinction between sharded-axis dims and replicated-axis dims.
/// The runtime tensor is per-rank-sharded by the loader, so the
/// numbers legitimately disagree at tp>1. Skip the check there;
/// the kernel itself uses the runtime tensor's shapes directly,
/// so the assertion is purely a sanity check that's only sound at
/// tp=1.
#[cfg(feature = "cuda")]
#[track_caller]
fn assert_weight_shape(
    op: &'static str,
    weight: crate::tensor::GpuTensor,
    n: u32,
    k: u32,
    tp_active: bool,
) {
    if tp_active {
        return;
    }
    let actual_n = weight.dim(0) as u32;
    let actual_k = weight.dim(1) as u32;
    assert_eq!(
        actual_n, n,
        "{op}: weight N (out_features) mismatch — runtime={actual_n} codegen={n}"
    );
    assert_eq!(
        actual_k, k,
        "{op}: weight K (in_features) mismatch — runtime={actual_k} codegen={k}"
    );
}

/// Whether a TP group is attached on this forward pass (i.e. tp>1).
/// Used by `assert_weight_shape` to skip its check at tp>1 where
/// runtime per-rank shapes legitimately disagree with the codegen's
/// unified-bounds shapes.
#[cfg(feature = "cuda")]
#[inline]
#[cfg(feature = "cuda")]
fn tp_active<W>(_ctx: &InterpreterCtx<'_, W>) -> bool {
    #[cfg(feature = "nccl")]
    {
        _ctx.fwd.tp_group.is_some()
    }
    #[cfg(not(feature = "nccl"))]
    {
        false
    }
}

/// Cold path for `Instruction::AttentionPrefillContiguous` when
/// `max_seqlen_q != max_seqlen_k` (chunked-prefill chunk 2+).
///
/// **Deliberately non-generic.** Callers pass HEAD_DIM / ATTN_SCALE /
/// ATTN_SOFTCAP as runtime args instead of `W`-typed const-generics, so
/// this compiles to ONE symbol regardless of how many model archs the
/// crate emits. With `#[cold]` + `#[inline(never)]` the symbol stays
/// out-of-line. eval<W>'s match arm reduces to a small forwarder; none
/// of the `flashinfer_attention` / `flash_attn_paged_ext` symbol
/// references end up in the per-arch eval<W> bodies, where they would
/// otherwise grow the function and disturb LLVM's code layout for the
/// hot fresh-prefill path.
///
/// Reads K/V from the paged cache (block_table + seqused_k); chunk 1's
/// writes plus this chunk's upstream `RopeAndCacheKV` mean the cache
/// spans the full sequence at this point, so q's q_len rows can attend
/// over the full seqlen_k.
#[cfg(feature = "cuda")]
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn attn_prefill_chunked_continuation(
    kv_cache: &crate::kv_cache::KvCachePool,
    cu_seqlens_q: crate::tensor::TensorView<'_>,
    seqused_k: crate::tensor::TensorView<'_>,
    block_table: crate::tensor::TensorView<'_>,
    q: crate::tensor::TensorView<'static>,
    max_q: usize,
    max_k: usize,
    layer: usize,
    head_dim: u32,
    attn_scale: f32,
    attn_softcap: f32,
    // `-1` = full causal (uniform / global-full continuation); `>= 0` = sliding
    // window (gemma4 SWA sliding layers). flashinfer has no window support, so
    // a windowed continuation skips it and uses the paged flash kernel (which
    // masks `kv_len-1-k >= window`) directly.
    window_size_left: i32,
    num_sm: i32,
    caching: &mut crate::CachingAllocator,
    stream: crate::CUstream,
    interleaved: bool,
) -> OwnedTensor {
    if window_size_left < 0 {
        let fi_cfg = flashinfer::FlashInferConfig {
            dtype: flashinfer::FiDType::Bf16,
            head_dim,
            use_logits_soft_cap: attn_softcap > 0.0,
        };
        let sk_bucket = ah::sk_bucket_for(max_k);
        let fi = unsafe {
            ah::flashinfer_attention(
                q,
                cu_seqlens_q,
                seqused_k,
                block_table,
                max_q,
                max_k,
                attn_scale,
                attn_softcap,
                kv_cache,
                layer,
                num_sm,
                fi_cfg,
                sk_bucket,
                caching,
                stream,
            )
        };
        if let Some(t) = fi {
            return t;
        }
    }
    unsafe {
        kernels::flash_attn_paged_ext(
            *q,
            *kv_cache.k_cache(layer),
            *kv_cache.v_cache(layer),
            *cu_seqlens_q,
            *seqused_k,
            *block_table,
            max_q,
            max_k,
            attn_scale,
            true,
            attn_softcap,
            window_size_left,
            kv_cache.block_size_for(layer),
            num_sm,
            caching,
            stream,
            ::std::ptr::null::<u8>(),
            0,
            interleaved,
            kv_cache.block_unrotated_gpu(),
        )
    }
}

/// Local extension trait carrying the CUDA interpreter's per-op
/// `eval`. Lives here (not on the foreign `Instruction` in
/// `scratchy-ir`) because `eval` reaches into the cuda-only
/// `InterpreterCtx` + kernel free fns; a local-trait-for-foreign-type
/// impl is orphan-legal.
#[cfg(feature = "cuda")]
pub trait InstructionEval {
    /// Evaluate one instruction. Closed match (no `_` arm).
    /// `Loop` is dispatched by [`run`] — never reaches here.
    ///
    /// `acc` is the weight accessor index for this op position in
    /// the [`Tape`]. Variants that don't consume a weight (e.g.
    /// `Add`, `BarrierSignal`, `Reshape`) ignore it; the
    /// proc-macro emits a sentinel value at the matching tape
    /// position so the parallel arrays stay aligned.
    ///
    /// # Safety
    /// Tile slot indices in range; weight accessors produce live
    /// GPU memory; `ctx.device.compute_stream` is live.
    #[allow(clippy::too_many_lines)]
    /// `bucket` and `op_idx` together identify the call site so the
    /// per-arch [`WeightAccessors`] impl can resolve which named
    /// weight to return. Variants that consume no weight ignore both.
    unsafe fn eval<W: CanonicalParams>(
        &self,
        ctx: &mut InterpreterCtx<'_, W>,
        bucket: u32,
        op_idx: u32,
    );
}

#[cfg(feature = "cuda")]
impl InstructionEval for Instruction {
    /// Evaluate one instruction. Closed match (no `_` arm).
    /// `Loop` is dispatched by [`run`] — never reaches here.
    ///
    /// `acc` is the weight accessor index for this op position in
    /// the [`Tape`]. Variants that don't consume a weight (e.g.
    /// `Add`, `BarrierSignal`, `Reshape`) ignore it; the
    /// proc-macro emits a sentinel value at the matching tape
    /// position so the parallel arrays stay aligned.
    ///
    /// # Safety
    /// Tile slot indices in range; weight accessors produce live
    /// GPU memory; `ctx.device.compute_stream` is live.
    #[allow(clippy::too_many_lines)]
    /// `bucket` and `op_idx` together identify the call site so the
    /// per-arch [`WeightAccessors`] impl can resolve which named
    /// weight to return. Variants that consume no weight ignore both.
    unsafe fn eval<W: CanonicalParams>(
        &self,
        ctx: &mut InterpreterCtx<'_, W>,
        bucket: u32,
        op_idx: u32,
    ) {
        let _ = (bucket, op_idx); // unused for variants that consume no weight
        match *self {
            // Metal-only fused lowerings — constructed only by the metal
            // interpreter, dispatched via the ICB tape, never the cuda eval.
            Instruction::SynthGateUpSiluMul(..) => {
                unreachable!("SynthGateUpSiluMul is metal-only; cuda never constructs it")
            }
            Instruction::AffineEmbed(..) => {
                unreachable!("AffineEmbed is metal-only; cuda never constructs it")
            }
            Instruction::Embed(out_slot) => unsafe {
                let weight = ctx.wm.embedding_at(bucket, op_idx, 0, 0u32).weight;
                // At tp=1 the embed weight covers the full vocab and
                // vocab_offset is 0 — the mask never trips. At tp>1
                // the weight is the per-rank `[vocab/tp, hidden]`
                // shard; vocab_offset = rank * vocab_per_rank gives
                // each rank a disjoint slice of the global vocab.
                // Per-Embed call follows with an AllReduce-sum
                // (injected by tp_lowering when shard-kind for the
                // embed weight is ShardDim0), and then a
                // `SpliceMmEmbeds` pass (also injected by tp_lowering)
                // that D2D-copies the vision-encoder embeddings into
                // the image-placeholder rows. The splice MUST run
                // after the AllReduce so its overwrite doesn't get
                // multiplied by `tp_world_size` on the sum.
                let vocab_per_rank = weight.dim(0) as u32;
                #[cfg(feature = "nccl")]
                let vocab_offset = ctx
                    .fwd
                    .tp_group
                    .map(|g| (g.rank() as u32) * vocab_per_rank)
                    .unwrap_or(0);
                #[cfg(not(feature = "nccl"))]
                let vocab_offset: u32 = 0;
                let out = kernels::embedding_gather_masked(
                    weight,
                    *ctx.fwd.input_ids,
                    vocab_offset,
                    vocab_per_rank,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::SpliceMmEmbeds(slot) => unsafe {
                // Text-only batches: skip. One branch per forward pass.
                if ctx.fwd.embed_patches.is_empty() {
                    return;
                }
                let out = tile_ref(ctx.tiles, slot).as_gpu_tensor(ctx.tiles);
                let mm = ctx
                    .fwd
                    .mm_embeds
                    .expect("mm_embeds required when embed_patches is non-empty");
                let hidden = out.dim(1);
                let elem_bytes = out.dtype().size_bytes();
                let row_bytes = hidden * elem_bytes;
                let dst_base = out.raw_ptr();
                let src_base = mm.raw_ptr();
                let mut src_row: usize = 0;
                for patch in ctx.fwd.embed_patches.iter() {
                    let length = patch.length as usize;
                    let dst = dst_base.add((patch.token_offset as usize) * row_bytes);
                    let src = src_base.add(src_row * row_bytes);
                    let _ = crate::driver::memcpy_dtod_async(
                        dst,
                        src,
                        length * row_bytes,
                        ctx.device.compute_stream,
                    );
                    src_row += length;
                }
            },
            Instruction::RmsNorm(in_slot, out_slot, layer, _hidden_size, _m_multiplier) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.rms_norm_at(bucket, op_idx, 0, layer);
                // `W::NORM_WEIGHT_OFFSET` folds the `(1 + w)` zero-centered
                // RMSNorm convention (Gemma / Qwen3.5) at kernel time. Metal
                // reads this through the same `CanonicalParams` const; cuda
                // matches by routing through `rms_norm_with_offset`. For
                // arches where the DSL already wrote `(Add(w, 1.0), rmsnorm)`
                // explicitly (Gemma2/3) the solver claims it as
                // `Instruction::ScalarOffsetRmsNorm` BEFORE this arm fires,
                // so the offset doesn't double-apply. Default 0.0 makes
                // standard arches (Llama / Qwen2 / Qwen3) byte-identical.
                let out = kernels::rms_norm_with_offset(
                    *v,
                    w.weight,
                    w.eps,
                    W::NORM_WEIGHT_OFFSET,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::MeanSubRmsNorm(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.rms_norm_at(bucket, op_idx, 0, layer);
                let out = kernels::cohere_layer_norm(
                    *v,
                    w.weight,
                    w.eps,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::MeanSubRmsNormBiasAdd(in_slot, out_slot, layer) => unsafe {
                // `LayerNorm`'s `bias` is `Option<GpuTensor>`; this Impl
                // is only emitted when the DSL has a downstream
                // `bias_add`, which means the loader is expected to
                // populate the bias. Empty bias is a loader/DSL
                // mismatch — surface it via `expect` rather than
                // silently dropping the bias term and producing wrong
                // numerics.
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let ln = ctx.wm.layer_norm_at(bucket, op_idx, 0, layer);
                let bias = ln
                    .bias
                    .expect("MeanSubRmsNormBiasAdd: LayerNorm.bias must be Some");
                let out = kernels::layer_norm_bias(
                    *v,
                    ln.weight,
                    bias,
                    ln.eps,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Reshape(in_slot, out_slot, dims_lit, dims_nt_pow, dims_div_lit, ndim) => {
                let upstream = tile_ref(ctx.tiles, in_slot).as_gpu_tensor(ctx.tiles);
                // `num_tokens` source: vision bodies set `pixels`
                // (and the input_ids ForwardCtx field is unused); the
                // decoder path keys off `input_ids`.
                let nt = match ctx.fwd.pixels {
                    Some(p) => (*p).dim(0),
                    None => (*ctx.fwd.input_ids).dim(0),
                };
                let mut shape = [0usize; MAX_DIMS];
                let nd = ndim as usize;
                for i in 0..nd {
                    let mut d = dims_lit[i] as usize;
                    for _ in 0..(dims_nt_pow[i] as usize) {
                        d *= nt;
                    }
                    let div = dims_div_lit[i] as usize;
                    debug_assert!(
                        div > 0 && d.is_multiple_of(div),
                        "Reshape: axis {i} numerator {d} not divisible by \
                         denominator {div} — codegen bug"
                    );
                    shape[i] = d / div;
                }
                let reshaped = upstream.reshape(&shape[..nd]);
                // Pin overwritten Owned: if `out_slot == in_slot`, the
                // existing `TileEntry::Owned` would otherwise drop here,
                // freeing the very memory the new `Reshaped` aliases.
                // See `pinned_owned` doc on `InterpreterCtx`. Take the
                // old entry out first so we can stash any Owned without
                // double-borrowing `ctx.tiles`.
                let prev = std::mem::take(&mut ctx.tiles[out_slot as usize]);
                if let Some(TileEntry::Owned(t)) = prev {
                    ctx.pinned_owned.push(t);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Reshaped {
                    ref_slot: in_slot,
                    tensor: reshaped,
                });
            }
            Instruction::Add(delta_slot, residual_slot) => unsafe {
                let delta = tile_ref(ctx.tiles, delta_slot).as_view(ctx.tiles);
                let residual = tile_ref(ctx.tiles, residual_slot).as_view(ctx.tiles);
                kernels::add_inplace(*residual, *delta, ctx.device.compute_stream);
            },
            #[cfg(feature = "nccl")]
            Instruction::AllReduce(slot) => unsafe {
                let group = ctx.fwd.tp_group.expect(
                    "Instruction::AllReduce reached eval but \
                     ForwardCtx::tp_group is None — caller must \
                     attach an NcclGroup at tp_world_size > 1",
                );
                let gt = tile_ref(ctx.tiles, slot).as_gpu_tensor(ctx.tiles);
                group
                    .all_reduce_inplace_promote(gt, &mut ctx.device.caching)
                    .expect("NCCL all_reduce_inplace_promote failed");
            },
            #[cfg(feature = "nccl")]
            Instruction::AllGather(in_slot, out_slot) => unsafe {
                let group = ctx.fwd.tp_group.expect(
                    "Instruction::AllGather reached eval but \
                     ForwardCtx::tp_group is None — caller must \
                     attach an NcclGroup at tp_world_size > 1",
                );
                let v = tile_ref(ctx.tiles, in_slot).as_gpu_tensor(ctx.tiles);
                let out = group.all_gather_last_dim(v, &mut ctx.device.caching);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::ScalarMul(in_slot, out_slot, scale) => {
                let owned = take_owned(ctx.tiles, in_slot);
                unsafe {
                    kernels::scale_inplace(*owned, scale, &ctx.device.cublas);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(owned));
            }
            Instruction::TanhSoftCap(in_slot, out_slot) => {
                let owned = take_owned(ctx.tiles, in_slot);
                unsafe {
                    kernels::tanh_softcap_inplace(
                        *owned,
                        W::FINAL_LOGIT_SOFTCAPPING,
                        ctx.device.compute_stream,
                    );
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(owned));
            }
            Instruction::FusedAddRmsNorm(
                delta_slot,
                residual_slot,
                layer,
                _hidden_size,
                _m_multiplier,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let delta = tile_ref(ctx.tiles, delta_slot).as_view(ctx.tiles);
                let residual = tile_ref(ctx.tiles, residual_slot).as_view(ctx.tiles);
                let w = ctx.wm.rms_norm_at(bucket, op_idx, 0, layer);
                // Same `(1 + w)` fold as the standalone RmsNorm arm above —
                // see that comment for the rationale and double-apply guard.
                let _ = kernels::fused_add_rms_norm_inplace_with_offset(
                    *delta,
                    *residual,
                    w.weight,
                    w.eps,
                    W::NORM_WEIGHT_OFFSET,
                    ctx.device.compute_stream,
                );
            },
            Instruction::FusedAddRmsNormWithOffset(delta_slot, residual_slot, layer, offset) => unsafe {
                let layer = ctx.layer_offset + layer;
                let delta = tile_ref(ctx.tiles, delta_slot).as_view(ctx.tiles);
                let residual = tile_ref(ctx.tiles, residual_slot).as_view(ctx.tiles);
                let w = ctx.wm.rms_norm_at(bucket, op_idx, 0, layer);
                let _ = kernels::fused_add_rms_norm_inplace_with_offset(
                    *delta,
                    *residual,
                    w.weight,
                    w.eps,
                    offset,
                    ctx.device.compute_stream,
                );
            },
            Instruction::ScalarOffsetRmsNorm(
                in_slot,
                out_slot,
                layer,
                offset,
                _hidden_size,
                _m_multiplier,
            ) => unsafe {
                // `hidden_size` / `m_multiplier` are metal-only (baked
                // fn-consts); the cuda kernel reads the tile shape.
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.rms_norm_at(bucket, op_idx, 0, layer);
                let out = kernels::rms_norm_with_offset(
                    *v,
                    w.weight,
                    w.eps,
                    offset,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassFusedRmsNormGemm(
                in_slot,
                out_slot,
                layer,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let nw = ctx.wm.rms_norm_at(bucket, op_idx, 0, layer);
                let gw = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                assert_weight_shape(
                    "CutlassFusedRmsNormGemm",
                    gw.dense_weight(),
                    n,
                    k,
                    tp_active(ctx),
                );
                // (1+w) zero-centered RMSNorm fold (Qwen3.5 / Gemma) —
                // see Instruction::RmsNorm arm for rationale.
                let normed = kernels::rms_norm_with_offset(
                    *v,
                    nw.weight,
                    nw.eps,
                    W::NORM_WEIGHT_OFFSET,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = cutlass::cutlass_gemm(
                    normed.as_gpu_tensor(),
                    gw.dense_weight(),
                    cutlass::CutlassTile::new(tile_m, tile_n, stages),
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassFusedMeanSubRmsNormGemm(
                in_slot,
                out_slot,
                layer,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let nw = ctx.wm.rms_norm_at(bucket, op_idx, 0, layer);
                let gw = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                assert_weight_shape(
                    "CutlassFusedMeanSubRmsNormGemm",
                    gw.dense_weight(),
                    n,
                    k,
                    tp_active(ctx),
                );
                let normed = kernels::cohere_layer_norm(
                    *v,
                    nw.weight,
                    nw.eps,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = cutlass::cutlass_gemm(
                    normed.as_gpu_tensor(),
                    gw.dense_weight(),
                    cutlass::CutlassTile::new(tile_m, tile_n, stages),
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassFusedAddRmsNormGemm(
                delta_slot,
                residual_slot,
                out_slot,
                layer,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let delta = tile_ref(ctx.tiles, delta_slot).as_view(ctx.tiles);
                let residual = tile_ref(ctx.tiles, residual_slot).as_view(ctx.tiles);
                let nw = ctx.wm.rms_norm_at(bucket, op_idx, 0, layer);
                let gw = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                assert_weight_shape(
                    "CutlassFusedAddRmsNormGemm",
                    gw.dense_weight(),
                    n,
                    k,
                    tp_active(ctx),
                );
                // After this kernel: delta buffer = normed output;
                // residual buffer = updated residual. The residual
                // alias is set up by the codegen prelude (TileEntry::View
                // on (add_id, 0) → residual upstream).
                // (1+w) zero-centered RMSNorm fold (Qwen3.5 / Gemma) —
                // see Instruction::RmsNorm arm for rationale.
                let (normed_view, _) = kernels::fused_add_rms_norm_inplace_with_offset(
                    *delta,
                    *residual,
                    nw.weight,
                    nw.eps,
                    W::NORM_WEIGHT_OFFSET,
                    ctx.device.compute_stream,
                );
                // Last-token-per-seq narrow before lm_head GEMM. At
                // prefill, lm_head only needs the row at
                // `query_start_loc[i+1]-1` for each sequence; running
                // the GEMM at full M=bucket_m wastes ~num_tokens/num_seqs×
                // compute. Mirrors metal's `GatherLastToken` lowering
                // and Python vLLM's
                // `logits_indices = query_start_loc[1:] - 1`.
                // The gather is gated on `idx.dim(0) < bucket_m` so the
                // decode path (M=BS already) is byte-identical.
                // `LMHEAD_NARROW=0` disables (debug-only A/B).
                let narrow_disabled = matches!(
                    std::env::var("LMHEAD_NARROW").as_deref(),
                    Ok("0") | Ok("off") | Ok("false")
                );
                let gathered_owned: Option<OwnedTensor>;
                let gemm_input = match ctx.fwd.last_token_indices {
                    Some(idx) if !narrow_disabled && idx.dim(0) < normed_view.dim(0) => {
                        let owned = kernels::embedding_gather(
                            normed_view,
                            *idx,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                        );
                        let view = owned.as_gpu_tensor();
                        gathered_owned = Some(owned);
                        view
                    }
                    _ => {
                        gathered_owned = None;
                        normed_view
                    }
                };
                let out = cutlass::cutlass_gemm(
                    gemm_input,
                    gw.dense_weight(),
                    cutlass::CutlassTile::new(tile_m, tile_n, stages),
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                drop(gathered_owned);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Gemm(in_slot, out_slot, layer, n, k) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                assert_weight_shape("Gemm", w.dense_weight(), n, k, tp_active(ctx));
                // Last-token-per-seq narrow before lm_head — same gate as
                // `CutlassFusedAddRmsNormGemm` (instr.rs above), but covers
                // the case where the cost solver picks plain cuBLAS Gemm
                // for the lm_head shape (e.g. Qwen2.5-7B M=65536 N=152064:
                // unfused full-M Gemm allocates 19.93 GiB output, OOMing
                // on first prefill at bench-latency bs=32 il=2048). Detected
                // via `n == vocab_size`. `LMHEAD_NARROW=0` disables for
                // debug A/B.
                let narrow_disabled = matches!(
                    std::env::var("LMHEAD_NARROW").as_deref(),
                    Ok("0") | Ok("off") | Ok("false")
                );
                // lm_head detection: prefer the unambiguous `n == VOCAB_SIZE`
                // when the canonical sets it (every modern arch). The
                // legacy `n > INTERMEDIATE_SIZE` fallback (when
                // VOCAB_SIZE is unset) mis-fires on MoE arches whose
                // INTERMEDIATE_SIZE is `moe_intermediate_size` and
                // therefore smaller than q_proj / linear_attn projections
                // (Qwen3.5-MoE: moe_inter=512 < q_gate_dim=8192) —
                // narrowing their input to last-token-per-seq at
                // multi-token prefill produces NaN activations.
                let is_lm_head = if W::VOCAB_SIZE > 0 {
                    (n as usize) == W::VOCAB_SIZE
                } else {
                    (k as usize) == W::HIDDEN_SIZE && (n as usize) > W::INTERMEDIATE_SIZE
                };
                let gathered_owned: Option<OwnedTensor>;
                let gemm_input = match ctx.fwd.last_token_indices {
                    Some(idx) if !narrow_disabled && is_lm_head && idx.dim(0) < (*v).dim(0) => {
                        let owned = kernels::embedding_gather(
                            *v,
                            *idx,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                        );
                        let view = owned.as_gpu_tensor();
                        gathered_owned = Some(owned);
                        view
                    }
                    _ => {
                        gathered_owned = None;
                        *v
                    }
                };
                let out =
                    ctx.device
                        .cublas
                        .gemm(gemm_input, w.dense_weight(), &mut ctx.device.caching);
                drop(gathered_owned);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::FusedCublasGemmAdd(in_slot, residual_slot, layer, n, k) => unsafe {
                // cuBLAS gemm(activation, weight) → delta; then
                // add_inplace folds delta into the residual buffer.
                // The residual upstream's OwnedTensor is aliased to
                // the Add tile's slot via the codegen prelude — same
                // as CutlassGemmAdd.
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let residual = tile_ref(ctx.tiles, residual_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                assert_weight_shape("FusedCublasGemmAdd", w.dense_weight(), n, k, tp_active(ctx));
                let delta = ctx
                    .device
                    .cublas
                    .gemm(*v, w.dense_weight(), &mut ctx.device.caching);
                kernels::add_inplace(*residual, delta.as_gpu_tensor(), ctx.device.compute_stream);
            },
            Instruction::FusedGemmBias(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                debug_assert!(
                    w.dense_bias().is_some(),
                    "FusedGemmBias: DSL `bias_add` claimed but \
                     LinearLayer has no bias — check safetensors path"
                );
                let out = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::FusedGateUpSiluMul(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                let gate_up = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = kernels::silu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::FusedGateUpGeluMul(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                let gate_up = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = kernels::gelu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::FusedQkvRopeCache(in_slot, out_slot, layer, biased, interleaved) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                if biased {
                    debug_assert!(
                        w.dense_bias().is_some(),
                        "FusedQkvRopeCache: DSL `bias_add` on QKV claimed but \
                         packed LinearLayer has no bias — check safetensors path"
                    );
                }
                let qkv_packed = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
                let out = if interleaved {
                    kernels::fused_qkv_interleaved_rope_cache(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                } else if ctx.fwd.kv_cache.is_fp8() {
                    kernels::fused_qkv_rope_cache_fp8(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        ctx.fwd.kv_cache.k_scale_ptr(layer as usize),
                        ctx.fwd.kv_cache.v_scale_ptr(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                } else {
                    kernels::fused_qkv_rope_cache(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        W::MROPE_SECTION,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::FusedQkvQkNormRopeCache(in_slot, out_slot, layer, q_offset, k_offset) => {
                let layer = ctx.layer_offset + layer;
                let mut q_out = unsafe {
                    let view_in = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    // FusedQkvQkNormRopeCache consumes 6 weights at this
                    // tape position: q_proj / k_proj / v_proj LinearLayers,
                    // q_norm / k_norm RmsNorms, and one CosSin. The per-
                    // arch impl returns each via its kind getter; the
                    // proc-macro records all 6 weight_slots in fan_out
                    // order so the per-(bucket, op_idx) match arms wire
                    // up to the right named accessors.
                    let qw = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                    let kw = ctx.wm.linear_at(bucket, op_idx, 1, layer);
                    let vw = ctx.wm.linear_at(bucket, op_idx, 2, layer);
                    let qnorm = ctx.wm.rms_norm_at(bucket, op_idx, 0, layer);
                    let knorm = ctx.wm.rms_norm_at(bucket, op_idx, 1, layer);
                    let nt = (*ctx.fwd.input_ids).dim(0);
                    let q = qw.forward(
                        view_in,
                        &mut ctx.device.cublas,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    let k = kw.forward(
                        view_in,
                        &mut ctx.device.cublas,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    let v_proj = vw.forward(
                        view_in,
                        &mut ctx.device.cublas,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    let q_view = (*q).reshape(&[nt, W::NUM_Q_HEADS as usize, W::HEAD_DIM as usize]);
                    let k_view =
                        (*k).reshape(&[nt, W::NUM_KV_HEADS as usize, W::HEAD_DIM as usize]);
                    let v_view =
                        (*v_proj).reshape(&[nt, W::NUM_KV_HEADS as usize, W::HEAD_DIM as usize]);
                    let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
                    kernels::qk_norm_rope_inplace(
                        q_view,
                        k_view,
                        qnorm.weight,
                        knorm.weight,
                        cos_sin,
                        *ctx.fwd.positions,
                        W::NUM_Q_HEADS as usize,
                        W::NUM_KV_HEADS as usize,
                        W::HEAD_DIM as usize,
                        qnorm.eps,
                        q_offset,
                        k_offset,
                        ctx.device.compute_stream,
                    );
                    kernels::reshape_and_cache(
                        k_view,
                        v_view,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        *ctx.fwd.slot_mapping,
                        ctx.fwd.kv_cache.block_size,
                        ctx.device.compute_stream,
                    );
                    q
                };
                unsafe {
                    let nt = (*q_out).dim(0);
                    let dt = (*q_out).dtype();
                    q_out.reshape(&[nt, W::NUM_Q_HEADS as usize, W::HEAD_DIM as usize], dt);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(q_out));
            }
            Instruction::FusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
                biased,
                interleaved,
            ) => {
                let layer = ctx.layer_offset + layer;
                let (q, k, v_out) = unsafe {
                    let view_in = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                    if biased {
                        debug_assert!(
                            w.dense_bias().is_some(),
                            "FusedQkvRopePrefill: DSL `bias_add` on QKV claimed but \
                             packed LinearLayer has no bias — check safetensors path"
                        );
                    }
                    let qkv_packed = w.forward(
                        view_in,
                        &mut ctx.device.cublas,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
                    if interleaved {
                        kernels::fused_qkv_interleaved_rope(
                            *qkv_packed,
                            *ctx.fwd.positions,
                            cos_sin,
                            W::Q_SIZE,
                            W::KV_SIZE,
                            W::NUM_Q_HEADS as usize,
                            W::NUM_KV_HEADS as usize,
                            W::HEAD_DIM as usize,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                        )
                    } else {
                        kernels::fused_qkv_rope(
                            *qkv_packed,
                            *ctx.fwd.positions,
                            cos_sin,
                            W::Q_SIZE,
                            W::KV_SIZE,
                            W::NUM_Q_HEADS as usize,
                            W::NUM_KV_HEADS as usize,
                            W::HEAD_DIM as usize,
                            W::MROPE_SECTION,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                        )
                    }
                };
                unsafe {
                    ah::write_kv_cache(
                        k.view(),
                        v_out.view(),
                        ctx.fwd.slot_mapping,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        ctx.device.compute_stream,
                    );
                }
                ctx.tiles[q_out_slot as usize] = Some(TileEntry::Owned(q));
                ctx.tiles[k_out_slot as usize] = Some(TileEntry::Owned(k));
                ctx.tiles[v_out_slot as usize] = Some(TileEntry::Owned(v_out));
            }
            Instruction::AttentionViaCache(in_slot, out_slot, layer, interleaved) => {
                let layer = ctx.layer_offset + layer;
                // Gemma-4 GLOBAL decode: head_dim 512 > flash cap → gather K/V
                // from the per-layer cache and run naive SDPA (q is already
                // RoPE'd by the global RopeAppend). Detect by the per-head width
                // (robust to a 2D `[nt, q_size]` or 3D `[nt, heads, hd]` q tile).
                let head_dim_q = {
                    let q = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    if (*q).ndim() >= 3 {
                        (*q).dim(2)
                    } else {
                        (*q).dim(1) / W::NUM_Q_HEADS as usize
                    }
                };
                let is_global_decode =
                    W::GLOBAL_HEAD_DIM != W::HEAD_DIM && head_dim_q == W::GLOBAL_HEAD_DIM as usize;
                if is_global_decode {
                    // GUARD (bs=1): the global gather + naive SDPA decode path
                    // reads a single block_table row and uses one scalar
                    // q_pos_base; >1 batched sequence would attend across
                    // sequence boundaries / read only request 0's KV. Fail loud.
                    let nseq = (*ctx.fwd.cu_seqlens_q).dim(0).saturating_sub(1);
                    assert_eq!(
                        nseq, 1,
                        "gemma4 GLOBAL decode SDPA is bs=1 only (got {nseq} \
                         sequences); batched global attention is not yet implemented."
                    );
                    let mut out = unsafe {
                        let q = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                        let hd = W::GLOBAL_HEAD_DIM as usize;
                        let kvh = W::NUM_GLOBAL_KV_HEADS as usize;
                        let nq = (*q).dim(0);
                        let q3 = q.reshape(&[nq, W::NUM_Q_HEADS as usize, hd]);
                        let total_kv = ctx.fwd.max_seqlen_k;
                        let k_g = ah::gather_global_kv_contiguous(
                            ctx.fwd.kv_cache,
                            *ctx.fwd.block_table_for(layer as usize),
                            layer as usize,
                            true,
                            total_kv,
                            kvh,
                            hd,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                        );
                        let v_g = ah::gather_global_kv_contiguous(
                            ctx.fwd.kv_cache,
                            *ctx.fwd.block_table_for(layer as usize),
                            layer as usize,
                            false,
                            total_kv,
                            kvh,
                            hd,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                        );
                        kernels::sdpa_naive(
                            *q3,
                            k_g.as_gpu_tensor(),
                            v_g.as_gpu_tensor(),
                            W::ATTN_SCALE,
                            W::ATTN_SOFTCAP,
                            -1,
                            (ctx.fwd.max_seqlen_k as i32) - (ctx.fwd.max_seqlen_q as i32),
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                        )
                    };
                    unsafe {
                        let nt = (*out).dim(0);
                        let dt = (*out).dtype();
                        out.reshape(&[nt, W::GLOBAL_Q_SIZE], dt);
                    }
                    ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
                    return;
                }
                let mut out = unsafe {
                    let q = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
                    let has_spans = !ctx.fwd.kv_cache.block_unrotated_gpu().is_null();
                    let (cos_sin_ptr, rotary_dim) = if has_spans {
                        (cos_sin.raw_ptr() as *const u8, cos_sin.dim(1))
                    } else {
                        (::std::ptr::null::<u8>(), 0)
                    };
                    ah::attention_decode_from_cache(
                        q,
                        ctx.fwd.cu_seqlens_q,
                        ctx.fwd.seqused_k,
                        ctx.fwd.block_table_for(layer as usize),
                        ctx.fwd.max_seqlen_q,
                        ctx.fwd.max_seqlen_k,
                        W::ATTN_SCALE,
                        W::ATTN_SOFTCAP,
                        -1,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        ctx.device.num_sm,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                        cos_sin_ptr,
                        rotary_dim,
                        interleaved,
                    )
                };
                unsafe {
                    let nt = (*out).dim(0);
                    let dt = (*out).dtype();
                    out.reshape(&[nt, W::Q_SIZE], dt);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            Instruction::AttentionPrefillContiguous(
                q_slot,
                k_slot,
                v_slot,
                out_slot,
                interleaved,
            ) => {
                let mut out = unsafe {
                    let q = tile_ref(ctx.tiles, q_slot).as_view(ctx.tiles);
                    let k = tile_ref(ctx.tiles, k_slot).as_view(ctx.tiles);
                    let v = tile_ref(ctx.tiles, v_slot).as_view(ctx.tiles);
                    // Gemma-4 GLOBAL attention: head_dim 512 > the flash-attn cap
                    // of 256, so route to the naive SDPA (any head_dim). Detect by
                    // the per-head width of the (already class-aware) q tile. Full
                    // causal attention (window -1); q_pos_base places the queries
                    // at the tail of the K/V (0 for fresh prefill, seqlen-1 for
                    // decode — both expressed as max_seqlen_k - max_seqlen_q, bs=1).
                    if (*q).dim(2) > 256 {
                        // Gemma-4 GLOBAL prefill (head_dim 512 > flash cap).
                        // GUARD (bs=1): the gather + naive SDPA below assume a
                        // single sequence (one block_table row, one q_pos_base).
                        // With >1 batched sequence the math silently attends
                        // across sequence boundaries — fail loudly instead.
                        let nseq = (*ctx.fwd.cu_seqlens_q).dim(0).saturating_sub(1);
                        assert_eq!(
                            nseq, 1,
                            "gemma4 GLOBAL prefill SDPA is bs=1 only (got {nseq} \
                             sequences); batched global attention is not yet \
                             implemented (would attend across sequence boundaries)."
                        );
                        if ctx.fwd.max_seqlen_q == ctx.fwd.max_seqlen_k {
                            // Fresh full prefill: the in-tile K/V IS the whole
                            // sequence, so attend over it directly (q_pos_base 0).
                            kernels::sdpa_naive(
                                *q,
                                *k,
                                *v,
                                W::ATTN_SCALE,
                                W::ATTN_SOFTCAP,
                                -1,
                                0,
                                &mut ctx.device.caching,
                                ctx.device.compute_stream,
                            )
                        } else {
                            // Chunked-prefill continuation (chunk 2+): the in-tile
                            // K/V holds ONLY the current chunk, but the queries
                            // must attend over the FULL prior context, which lives
                            // in the paged cache. Gather the whole K/V history
                            // (mirrors the global-decode path) and attend with
                            // q_pos_base = past_len. Without this the global layers
                            // silently lose all prior-chunk context on long prompts.
                            let hd = W::GLOBAL_HEAD_DIM as usize;
                            let kvh = W::NUM_GLOBAL_KV_HEADS as usize;
                            let total_kv = ctx.fwd.max_seqlen_k;
                            let gl = ctx.kv_layer as usize;
                            let k_g = ah::gather_global_kv_contiguous(
                                ctx.fwd.kv_cache,
                                *ctx.fwd.block_table_for(gl),
                                gl,
                                true,
                                total_kv,
                                kvh,
                                hd,
                                &mut ctx.device.caching,
                                ctx.device.compute_stream,
                            );
                            let v_g = ah::gather_global_kv_contiguous(
                                ctx.fwd.kv_cache,
                                *ctx.fwd.block_table_for(gl),
                                gl,
                                false,
                                total_kv,
                                kvh,
                                hd,
                                &mut ctx.device.caching,
                                ctx.device.compute_stream,
                            );
                            let qpb = (ctx.fwd.max_seqlen_k as i32) - (ctx.fwd.max_seqlen_q as i32);
                            kernels::sdpa_naive(
                                *q,
                                k_g.as_gpu_tensor(),
                                v_g.as_gpu_tensor(),
                                W::ATTN_SCALE,
                                W::ATTN_SOFTCAP,
                                -1,
                                qpb,
                                &mut ctx.device.caching,
                                ctx.device.compute_stream,
                            )
                        }
                    } else if ctx.fwd.max_seqlen_q <= 1
                        || ctx.fwd.max_seqlen_q == ctx.fwd.max_seqlen_k
                    {
                        // HOT path: fresh prefill (q == k) OR decode (q == 1).
                        // Byte-for-byte the pre-fix code; the cold branch's
                        // flashinfer/flash_attn_paged symbols live in a
                        // non-generic out-of-line helper, so eval<W>'s
                        // monomorphizations don't pull them into this body.
                        kernels::flash_attn_contiguous(
                            *q,
                            *k,
                            *v,
                            *ctx.fwd.cu_seqlens_q,
                            *ctx.fwd.cu_seqlens_q,
                            ctx.fwd.max_seqlen_q,
                            ctx.fwd.max_seqlen_k,
                            W::ATTN_SCALE,
                            true,
                            W::ATTN_SOFTCAP,
                            -1,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                            ::std::ptr::null::<u8>(),
                            0,
                            interleaved,
                        )
                    } else {
                        // COLD path: chunked-prefill continuation (chunk 2+).
                        attn_prefill_chunked_continuation(
                            ctx.fwd.kv_cache,
                            ctx.fwd.cu_seqlens_q,
                            ctx.fwd.seqused_k,
                            ctx.fwd.block_table_for(ctx.kv_layer as usize),
                            q,
                            ctx.fwd.max_seqlen_q,
                            ctx.fwd.max_seqlen_k,
                            ctx.kv_layer as usize,
                            W::HEAD_DIM,
                            W::ATTN_SCALE,
                            W::ATTN_SOFTCAP,
                            -1, // full causal (non-sliding class)
                            ctx.device.num_sm,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                            interleaved,
                        )
                    }
                };
                unsafe {
                    // `[nt, num_q_heads, head_dim]` → `[nt, num_q_heads*head_dim]`.
                    // Class-aware: global = GLOBAL_Q_SIZE (16*512), local = Q_SIZE.
                    let nt = (*out).dim(0);
                    let q_size = (*out).dim(1) * (*out).dim(2);
                    let dt = (*out).dtype();
                    out.reshape(&[nt, q_size], dt);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            Instruction::AttentionPrefillPaged(_q_slot, _out_slot, _layer, _interleaved) => {
                unimplemented!(
                    "AttentionPrefillPaged is metal-only — cuda routes prefill through \
                     `Instruction::AttentionPrefillContiguous` (flash_attn_contiguous)."
                );
            }
            Instruction::SlidingAttentionPrefillPaged(_q_slot, _out_slot, _layer, _interleaved) => {
                unimplemented!(
                    "SlidingAttentionPrefillPaged is metal-only — cuda routes sliding \
                     prefill through `Instruction::SlidingAttentionPrefillContiguous`."
                );
            }
            Instruction::EncoderAttention(q_slot, k_slot, v_slot, out_slot) => {
                // Encoder/bidirectional self-attention: same FA2 kernel
                // as the prefill prefill path but with `is_causal=false`
                // (every query attends to every key in its sequence). No
                // softcap (encoder models don't use it), no sliding
                // window (full attention), no fused RoPE (RoPE applied
                // upstream — null cos_sin pointer + rotary_dim=0 makes
                // flash-attn skip its fused rotary). cu_seqlens_q is
                // reused as cu_seqlens_k since K and V live alongside Q
                // for self-attention.
                let mut out = unsafe {
                    let q = tile_ref(ctx.tiles, q_slot).as_view(ctx.tiles);
                    let k = tile_ref(ctx.tiles, k_slot).as_view(ctx.tiles);
                    let v = tile_ref(ctx.tiles, v_slot).as_view(ctx.tiles);
                    kernels::flash_attn_contiguous(
                        *q,
                        *k,
                        *v,
                        *ctx.fwd.cu_seqlens_q,
                        *ctx.fwd.cu_seqlens_q,
                        ctx.fwd.max_seqlen_q,
                        ctx.fwd.max_seqlen_q,
                        W::ATTN_SCALE,
                        false,
                        0.0,
                        -1,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                        ::std::ptr::null::<u8>(),
                        0,
                        false,
                    )
                };
                unsafe {
                    let nt = (*out).dim(0);
                    let dt = (*out).dtype();
                    out.reshape(&[nt, W::Q_SIZE], dt);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            Instruction::SlidingAttentionViaCache(in_slot, out_slot, layer, interleaved) => {
                let layer = ctx.layer_offset + layer;
                let mut out = unsafe {
                    let q = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
                    let has_spans = !ctx.fwd.kv_cache.block_unrotated_gpu().is_null();
                    let (cos_sin_ptr, rotary_dim) = if has_spans {
                        (cos_sin.raw_ptr() as *const u8, cos_sin.dim(1))
                    } else {
                        (::std::ptr::null::<u8>(), 0)
                    };
                    ah::attention_decode_from_cache(
                        q,
                        ctx.fwd.cu_seqlens_q,
                        ctx.fwd.seqused_k,
                        ctx.fwd.block_table_for(layer as usize),
                        ctx.fwd.max_seqlen_q,
                        ctx.fwd.max_seqlen_k,
                        W::ATTN_SCALE,
                        W::ATTN_SOFTCAP,
                        W::SLIDING_WINDOW,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        ctx.device.num_sm,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                        cos_sin_ptr,
                        rotary_dim,
                        interleaved,
                    )
                };
                unsafe {
                    let nt = (*out).dim(0);
                    let dt = (*out).dtype();
                    out.reshape(&[nt, W::Q_SIZE], dt);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            Instruction::SlidingAttentionPrefillContiguous(
                q_slot,
                k_slot,
                v_slot,
                out_slot,
                interleaved,
            ) => {
                let layer = ctx.kv_layer as usize;
                // Chunked-prefill continuation (chunk 2+): the prior chunks'
                // windowed K/V live in the paged cache (RopeAppend wrote this
                // chunk's K/V before this op), so a fresh in-tile attention
                // would miss them. Attend q against the cached K/V with the
                // sliding window via the paged flash path. Fresh prefill
                // (q == k, no cached prefix) keeps the in-tile kernel.
                let is_continuation = ctx.fwd.max_seqlen_k > ctx.fwd.max_seqlen_q;
                let mut out = if is_continuation {
                    // Gather the sliding class's cached K/V (bs=1) and run the
                    // naive windowed SDPA — the SAME proven path the global
                    // class uses for its continuation (verified sane output),
                    // rather than flash_attn_paged's windowed varlen-prefill
                    // (q>1) path, which garbles here. sdpa_naive applies the
                    // sliding window via its `window` arg; O(T·window) effective.
                    unsafe {
                        let hd = W::HEAD_DIM as usize;
                        let kvh = W::NUM_KV_HEADS as usize;
                        let total_kv = ctx.fwd.max_seqlen_k;
                        let q = tile_ref(ctx.tiles, q_slot).as_view(ctx.tiles);
                        let nq = (*q).dim(0);
                        let q3 = q.reshape(&[nq, W::NUM_Q_HEADS as usize, hd]);
                        let k_g = ah::gather_global_kv_contiguous(
                            ctx.fwd.kv_cache,
                            *ctx.fwd.block_table_for(layer),
                            layer,
                            true,
                            total_kv,
                            kvh,
                            hd,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                        );
                        let v_g = ah::gather_global_kv_contiguous(
                            ctx.fwd.kv_cache,
                            *ctx.fwd.block_table_for(layer),
                            layer,
                            false,
                            total_kv,
                            kvh,
                            hd,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                        );
                        let qpb = (ctx.fwd.max_seqlen_k as i32) - (ctx.fwd.max_seqlen_q as i32);
                        kernels::sdpa_naive(
                            *q3,
                            k_g.as_gpu_tensor(),
                            v_g.as_gpu_tensor(),
                            W::ATTN_SCALE,
                            W::ATTN_SOFTCAP,
                            W::SLIDING_WINDOW,
                            qpb,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                        )
                    }
                } else {
                    unsafe {
                        let q = tile_ref(ctx.tiles, q_slot).as_view(ctx.tiles);
                        let k = tile_ref(ctx.tiles, k_slot).as_view(ctx.tiles);
                        let v = tile_ref(ctx.tiles, v_slot).as_view(ctx.tiles);
                        kernels::flash_attn_contiguous(
                            *q,
                            *k,
                            *v,
                            *ctx.fwd.cu_seqlens_q,
                            *ctx.fwd.cu_seqlens_q,
                            ctx.fwd.max_seqlen_q,
                            ctx.fwd.max_seqlen_k,
                            W::ATTN_SCALE,
                            true,
                            W::ATTN_SOFTCAP,
                            W::SLIDING_WINDOW,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                            ::std::ptr::null::<u8>(),
                            0,
                            interleaved,
                        )
                    }
                };
                unsafe {
                    let nt = (*out).dim(0);
                    let dt = (*out).dtype();
                    out.reshape(&[nt, W::Q_SIZE], dt);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            Instruction::VarlenAttention(q_slot, k_slot, v_slot, out_slot, cu_seqlens_kind) => {
                let (cu_view, max_seqlen) = match cu_seqlens_kind {
                    0 => (ctx.fwd.cu_seqlens_q, ctx.fwd.max_seqlen_q),
                    1 => {
                        let cu = ctx.fwd.vision_cu_seqlens_full.expect(
                            "Instruction::VarlenAttention(kind=1) reached eval but \
                             ForwardCtx::vision_cu_seqlens_full is None — caller must \
                             populate it before invoking the vision interpreter",
                        );
                        let m = ctx.fwd.vision_max_seqlen_full.expect(
                            "Instruction::VarlenAttention(kind=1) reached eval but \
                             ForwardCtx::vision_max_seqlen_full is None",
                        );
                        (cu, m)
                    }
                    2 => {
                        let cu = ctx.fwd.vision_cu_seqlens_window.expect(
                            "Instruction::VarlenAttention(kind=2) reached eval but \
                             ForwardCtx::vision_cu_seqlens_window is None — caller must \
                             populate it before invoking the vision interpreter",
                        );
                        let m = ctx.fwd.vision_max_seqlen_window.expect(
                            "Instruction::VarlenAttention(kind=2) reached eval but \
                             ForwardCtx::vision_max_seqlen_window is None",
                        );
                        (cu, m)
                    }
                    other => panic!(
                        "Instruction::VarlenAttention: cu_seqlens_kind must be 0 \
                         (default) | 1 (full) | 2 (window); got {other}"
                    ),
                };
                let mut out = unsafe {
                    let q_view = tile_ref(ctx.tiles, q_slot).as_view(ctx.tiles);
                    let k_view = tile_ref(ctx.tiles, k_slot).as_view(ctx.tiles);
                    let v_view = tile_ref(ctx.tiles, v_slot).as_view(ctx.tiles);
                    let total_l = (*q_view).dim(0);
                    let nh = W::VISION_NUM_HEADS as usize;
                    let hd = W::VISION_HEAD_DIM as usize;
                    let q3 = q_view.reshape(&[total_l, nh, hd]);
                    let k3 = k_view.reshape(&[total_l, nh, hd]);
                    let v3 = v_view.reshape(&[total_l, nh, hd]);
                    kernels::flash_attn_contiguous(
                        *q3,
                        *k3,
                        *v3,
                        *cu_view,
                        *cu_view,
                        max_seqlen,
                        max_seqlen,
                        W::VISION_ATTN_SCALE,
                        false,
                        0.0,
                        -1,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                        ::std::ptr::null::<u8>(),
                        0,
                        false,
                    )
                };
                unsafe {
                    let nt = (*out).dim(0);
                    let dt = (*out).dtype();
                    out.reshape(&[nt, W::VISION_Q_SIZE], dt);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            Instruction::VisionRope(q_slot, k_slot, q_out_slot, k_out_slot) => {
                let cos = ctx.fwd.vision_rope_cos.expect(
                    "Instruction::VisionRope reached eval but \
                     ForwardCtx::vision_rope_cos is None — caller must \
                     populate it before invoking the vision interpreter",
                );
                let sin = ctx.fwd.vision_rope_sin.expect(
                    "Instruction::VisionRope reached eval but \
                     ForwardCtx::vision_rope_sin is None — caller must \
                     populate it before invoking the vision interpreter",
                );
                let q_owned = take_owned(ctx.tiles, q_slot);
                let k_owned = take_owned(ctx.tiles, k_slot);
                let total_l = (*q_owned).dim(0);
                let nh = W::VISION_NUM_HEADS as usize;
                let hd = W::VISION_HEAD_DIM as usize;
                let q3 = (*q_owned).reshape(&[total_l, nh, hd]);
                let k3 = (*k_owned).reshape(&[total_l, nh, hd]);
                unsafe {
                    kernels::vision_rope_apply(q3, *cos, *sin, ctx.device.compute_stream);
                    kernels::vision_rope_apply(k3, *cos, *sin, ctx.device.compute_stream);
                }
                ctx.tiles[q_out_slot as usize] = Some(TileEntry::Owned(q_owned));
                ctx.tiles[k_out_slot as usize] = Some(TileEntry::Owned(k_owned));
            }
            Instruction::QuickGelu(in_slot, out_slot) => {
                let owned = take_owned(ctx.tiles, in_slot);
                unsafe {
                    kernels::quick_gelu_inplace(*owned, ctx.device.compute_stream);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(owned));
            }
            Instruction::Gelu(in_slot, out_slot) => {
                let owned = take_owned(ctx.tiles, in_slot);
                unsafe {
                    kernels::gelu_tanh_inplace(*owned, ctx.device.compute_stream);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(owned));
            }
            Instruction::PosEmbed(out_slot) => unsafe {
                let weight = ctx.wm.embedding_at(bucket, op_idx, 0, 0u32).weight;
                let position_ids = ctx.fwd.vision_position_ids.expect(
                    "Instruction::PosEmbed reached eval but \
                     ForwardCtx::vision_position_ids is None — caller \
                     (vision_forward host wrapper) must populate this \
                     view before driving the vision interpreter, \
                     mirroring the pixels / cu_seqlens / cos / sin \
                     contract",
                );
                let num_positions = weight.dim(0) as u32;
                // No tp sharding on the vision positional table —
                // vision is replicated per-rank; vocab_offset = 0 and
                // vocab_per_rank covers the full table so the mask
                // arm in `embedding_gather_masked` never trips.
                let out = kernels::embedding_gather_masked(
                    weight,
                    *position_ids,
                    0u32,
                    num_positions,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::GeluErf(in_slot, out_slot) => {
                let owned = take_owned(ctx.tiles, in_slot);
                unsafe {
                    kernels::gelu_erf_inplace(*owned, ctx.device.compute_stream);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(owned));
            }
            Instruction::EmbeddingGather(in_slot, out_slot, indices_kind) => {
                let indices_view = match indices_kind {
                    0 => ctx.fwd.vision_window_index.expect(
                        "Instruction::EmbeddingGather(kind=0) reached eval but \
                         ForwardCtx::vision_window_index is None — caller must \
                         populate it before invoking the vision interpreter",
                    ),
                    1 => ctx.fwd.vision_reverse_indices.expect(
                        "Instruction::EmbeddingGather(kind=1) reached eval but \
                         ForwardCtx::vision_reverse_indices is None — caller must \
                         populate it before invoking the vision interpreter",
                    ),
                    other => panic!(
                        "Instruction::EmbeddingGather: indices_kind must be 0 \
                         (window_index) or 1 (reverse_indices); got {other}"
                    ),
                };
                let owned = unsafe {
                    let in_view = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    kernels::embedding_gather(
                        *in_view,
                        *indices_view,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(owned));
            }
            Instruction::AvgPool2d(in_slot, out_slot) => {
                let ph = W::VISION_PATCH_GRID_SIDE;
                let k = W::VISION_POOL_KERNEL;
                if ph == 0 || k == 0 {
                    panic!(
                        "Instruction::AvgPool2d reached eval with \
                         VISION_PATCH_GRID_SIDE={ph} VISION_POOL_KERNEL={k} — \
                         the per-arch CanonicalParams bake must populate both \
                         (vision_patch_grid_side / vision_pool_kernel bounds in \
                         configs/<variant>.json)",
                    );
                }
                let owned = unsafe {
                    let in_view = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    kernels::avg_pool_2d(
                        *in_view,
                        ph,
                        k,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(owned));
            }
            Instruction::LoadPixels(out_slot) => unsafe {
                let view = ctx.fwd.pixels.expect(
                    "Instruction::LoadPixels invoked without ForwardCtx::pixels — \
                     caller (vision_forward host wrapper) must populate this view \
                     before driving the vision interpreter, mirroring the \
                     vision_rope_cos / vision_rope_sin contract",
                );
                let raw = view.as_raw();
                let shape: Vec<usize> = raw.shape().iter().map(|&d| d as usize).collect();
                let owned = ctx.device.caching.alloc_tensor(&shape, raw.dtype());
                let bytes = raw.size_bytes();
                crate::driver::memcpy_dtod_async(
                    (*owned).raw_ptr(),
                    raw.raw_ptr(),
                    bytes,
                    ctx.device.compute_stream,
                )
                .expect("LoadPixels D2D copy");
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(owned));
            },
            Instruction::LoadPosEmbeds(out_slot) => unsafe {
                let view = ctx.fwd.pos_embeds.expect(
                    "Instruction::LoadPosEmbeds invoked without ForwardCtx::pos_embeds — \
                     caller (vision_forward host wrapper) must populate this view \
                     (host-interpolated learned positional embedding) before driving \
                     the vision interpreter, mirroring the pixels contract",
                );
                let raw = view.as_raw();
                let shape: Vec<usize> = raw.shape().iter().map(|&d| d as usize).collect();
                let owned = ctx.device.caching.alloc_tensor(&shape, raw.dtype());
                let bytes = raw.size_bytes();
                crate::driver::memcpy_dtod_async(
                    (*owned).raw_ptr(),
                    raw.raw_ptr(),
                    bytes,
                    ctx.device.compute_stream,
                )
                .expect("LoadPosEmbeds D2D copy");
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(owned));
            },
            Instruction::FlashInferAttentionDecode(
                in_slot,
                out_slot,
                layer,
                head_dim,
                use_logits_soft_cap,
            ) => {
                let layer = ctx.layer_offset + layer;
                let mut out = unsafe {
                    let q = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let fi_cfg = flashinfer::FlashInferConfig {
                        dtype: flashinfer::FiDType::Bf16,
                        head_dim,
                        use_logits_soft_cap,
                    };
                    let sk_bucket = ah::sk_bucket_for(ctx.fwd.max_seqlen_k);
                    // The TP>1 path runs through `AttentionViaCacheImpl`
                    // (FA2) on sm<90 and `FlashAttention3DecodeImpl` (FA3)
                    // on sm>=90 — both graph-capture-safe. The solver
                    // doesn't emit `Instruction::FlashInferAttentionDecode`
                    // when `tp_world_size > 1` (gated in
                    // `FlashInferAttentionDecodeImpl::applies_to`), so any
                    // execution of this arm is a TP=1 path where FI's
                    // cooperative persistent kernel is the right choice.
                    let fi = ah::flashinfer_attention(
                        q,
                        ctx.fwd.cu_seqlens_q,
                        ctx.fwd.seqused_k,
                        ctx.fwd.block_table_for(layer as usize),
                        ctx.fwd.max_seqlen_q,
                        ctx.fwd.max_seqlen_k,
                        W::ATTN_SCALE,
                        W::ATTN_SOFTCAP,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        ctx.device.num_sm,
                        fi_cfg,
                        sk_bucket,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    match fi {
                        Some(t) => t,
                        None => {
                            let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
                            let has_spans = !ctx.fwd.kv_cache.block_unrotated_gpu().is_null();
                            let (cos_sin_ptr, rotary_dim) = if has_spans {
                                (cos_sin.raw_ptr() as *const u8, cos_sin.dim(1))
                            } else {
                                (::std::ptr::null::<u8>(), 0)
                            };
                            ah::attention_decode_from_cache(
                                q,
                                ctx.fwd.cu_seqlens_q,
                                ctx.fwd.seqused_k,
                                ctx.fwd.block_table_for(layer as usize),
                                ctx.fwd.max_seqlen_q,
                                ctx.fwd.max_seqlen_k,
                                W::ATTN_SCALE,
                                W::ATTN_SOFTCAP,
                                -1,
                                ctx.fwd.kv_cache,
                                layer as usize,
                                ctx.device.num_sm,
                                &mut ctx.device.caching,
                                ctx.device.compute_stream,
                                cos_sin_ptr,
                                rotary_dim,
                                false,
                            )
                        }
                    }
                };
                unsafe {
                    let nt = (*out).dim(0);
                    let dt = (*out).dtype();
                    out.reshape(&[nt, W::Q_SIZE], dt);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            #[cfg(fa3_built)]
            Instruction::FlashAttention3Decode(in_slot, out_slot, layer, _head_dim) => {
                // Solver-selected: this Instruction is emitted only when
                // `FlashAttention3DecodeImpl::target_compatible()` passed
                // (sm_90+, head_dim==128, no softcap, FA3 cost-CSV row
                // present). No fallback inside — if FA3 dispatch fails,
                // the bug is at the Impl/cost-CSV level, not runtime.
                let step_layer = layer as usize;
                let layer = ctx.layer_offset + layer;
                let mut out = unsafe {
                    let q = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    ah::flash_attn_3_decode(
                        q,
                        ctx.fwd.cu_seqlens_q,
                        ctx.fwd.seqused_k,
                        ctx.fwd.block_table_for(layer as usize),
                        ctx.fwd.max_seqlen_q,
                        ctx.fwd.max_seqlen_k,
                        W::ATTN_SCALE,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        step_layer,
                        ctx.device.num_sm,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                unsafe {
                    let nt = (*out).dim(0);
                    let dt = (*out).dtype();
                    out.reshape(&[nt, W::Q_SIZE], dt);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            Instruction::FlashInferAttentionPrefill(
                q_slot,
                k_slot,
                v_slot,
                out_slot,
                layer,
                head_dim,
                use_logits_soft_cap,
            ) => {
                let layer = ctx.layer_offset + layer;
                let mut out = unsafe {
                    let q = tile_ref(ctx.tiles, q_slot).as_view(ctx.tiles);
                    let k = tile_ref(ctx.tiles, k_slot).as_view(ctx.tiles);
                    let v = tile_ref(ctx.tiles, v_slot).as_view(ctx.tiles);
                    let fi_cfg = flashinfer::FlashInferConfig {
                        dtype: flashinfer::FiDType::Bf16,
                        head_dim,
                        use_logits_soft_cap,
                    };
                    let sk_bucket = ah::sk_bucket_for(ctx.fwd.max_seqlen_k);
                    // Solver-gated to TP=1 in
                    // `FlashInferAttentionPrefillImpl::applies_to`. See
                    // FlashInferAttentionDecode arm above for the full
                    // explanation.
                    let fi = ah::flashinfer_attention(
                        q,
                        ctx.fwd.cu_seqlens_q,
                        ctx.fwd.seqused_k,
                        ctx.fwd.block_table_for(layer as usize),
                        ctx.fwd.max_seqlen_q,
                        ctx.fwd.max_seqlen_k,
                        W::ATTN_SCALE,
                        W::ATTN_SOFTCAP,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        ctx.device.num_sm,
                        fi_cfg,
                        sk_bucket,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    match fi {
                        Some(t) => t,
                        None => kernels::flash_attn_contiguous(
                            *q,
                            *k,
                            *v,
                            *ctx.fwd.cu_seqlens_q,
                            *ctx.fwd.cu_seqlens_q,
                            ctx.fwd.max_seqlen_q,
                            ctx.fwd.max_seqlen_k,
                            W::ATTN_SCALE,
                            true,
                            W::ATTN_SOFTCAP,
                            -1,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                            ::std::ptr::null::<u8>(),
                            0,
                            false,
                        ),
                    }
                };
                unsafe {
                    let nt = (*out).dim(0);
                    let dt = (*out).dtype();
                    out.reshape(&[nt, W::Q_SIZE], dt);
                }
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            Instruction::RopeAppend(
                q_slot,
                k_slot,
                v_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
                interleaved,
                is_global,
            ) => unsafe {
                // Gemma-4 CLASS-AWARE geometry: the GLOBAL (full-attention) class
                // uses GLOBAL_HEAD_DIM / NUM_GLOBAL_KV_HEADS and PROPORTIONAL rope
                // (theta-1e6 cache, pair_off = head_dim/2, rot_dim < head_dim);
                // the local/sliding class uses the base consts + NeoX rope.
                // Identity on uniform-geometry models (GLOBAL_* == base, is_global
                // never carries a different shape).
                let layer = ctx.layer_offset + layer;
                // Record the absolute KV layer for the following paged-attention
                // op that lacks its own layer operand (chunked-prefill
                // continuation). See InterpreterCtx::kv_layer.
                ctx.kv_layer = layer;
                let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
                let q_view = tile_ref(ctx.tiles, q_slot).as_view(ctx.tiles);
                let k_view = tile_ref(ctx.tiles, k_slot).as_view(ctx.tiles);
                let v_view = tile_ref(ctx.tiles, v_slot).as_view(ctx.tiles);

                let (head_dim, num_kv_heads) = if is_global {
                    (W::GLOBAL_HEAD_DIM as usize, W::NUM_GLOBAL_KV_HEADS as usize)
                } else {
                    (W::HEAD_DIM as usize, W::NUM_KV_HEADS as usize)
                };
                let proportional = is_global && W::GLOBAL_ROT_DIM != W::GLOBAL_HEAD_DIM;

                if proportional {
                    kernels::rotary_embedding_proportional_inplace(
                        *q_view,
                        *k_view,
                        *ctx.fwd.positions,
                        cos_sin,
                        head_dim,
                        ctx.device.compute_stream,
                    );
                } else if interleaved {
                    kernels::rotary_embedding_interleaved_inplace(
                        *q_view,
                        *k_view,
                        *ctx.fwd.positions,
                        cos_sin,
                        head_dim,
                        ctx.device.compute_stream,
                    );
                } else {
                    kernels::rotary_embedding_inplace(
                        *q_view,
                        *k_view,
                        *ctx.fwd.positions,
                        cos_sin,
                        head_dim,
                        ctx.device.compute_stream,
                    );
                }
                let nt = (*k_view).dim(0);
                let k_3d = k_view.reshape(&[nt, num_kv_heads, head_dim]);
                let v_3d = v_view.reshape(&[nt, num_kv_heads, head_dim]);
                kernels::reshape_and_cache(
                    *k_3d,
                    *v_3d,
                    *ctx.fwd.kv_cache.k_cache(layer as usize),
                    *ctx.fwd.kv_cache.v_cache(layer as usize),
                    *ctx.fwd.slot_mapping_for(layer as usize),
                    ctx.fwd.kv_cache.block_size_for(layer as usize),
                    ctx.device.compute_stream,
                );
                let nt_q = (*q_view).dim(0);
                let q_3d = q_view.reshape(&[nt_q, W::NUM_Q_HEADS as usize, head_dim]);
                // Pin any Owned overwritten by these three Reshaped
                // writes. See InterpreterCtx::pinned_owned for the why.
                for slot in [q_out_slot, k_out_slot, v_out_slot] {
                    let prev = std::mem::take(&mut ctx.tiles[slot as usize]);
                    if let Some(TileEntry::Owned(t)) = prev {
                        ctx.pinned_owned.push(t);
                    }
                }
                ctx.tiles[q_out_slot as usize] = Some(TileEntry::Reshaped {
                    ref_slot: q_slot,
                    tensor: *q_3d,
                });
                ctx.tiles[k_out_slot as usize] = Some(TileEntry::Reshaped {
                    ref_slot: k_slot,
                    tensor: *k_3d,
                });
                ctx.tiles[v_out_slot as usize] = Some(TileEntry::Reshaped {
                    ref_slot: v_slot,
                    tensor: *v_3d,
                });
            },
            Instruction::MlaSplit(in_slot, kv_latent_slot, k_pe_slot) => unsafe {
                let kv_a_tv = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let nt = (*kv_a_tv).dim(0);
                let dt = (*kv_a_tv).dtype();
                let kv_latent = ctx.device.caching.alloc_tensor(&[nt, W::KV_LORA_RANK], dt);
                let k_pe = ctx
                    .device
                    .caching
                    .alloc_tensor(&[nt, W::QK_ROPE_HEAD_DIM], dt);
                kernels::mla_split_kv_a(
                    *kv_a_tv,
                    *kv_latent.view(),
                    *k_pe.view(),
                    W::KV_LORA_RANK,
                    W::QK_ROPE_HEAD_DIM,
                    ctx.device.compute_stream,
                );
                ctx.tiles[kv_latent_slot as usize] = Some(TileEntry::Owned(kv_latent));
                ctx.tiles[k_pe_slot as usize] = Some(TileEntry::Owned(k_pe));
            },
            Instruction::MlaAttention(q_slot, kv_b_slot, k_pe_slot, out_slot, layer) => {
                let layer = ctx.layer_offset + layer;
                let out = unsafe {
                    mla_attention_eval(ctx, q_slot, kv_b_slot, k_pe_slot, layer, bucket, op_idx)
                };
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            Instruction::GatedDeltaNet(qkv_slot, z_slot, a_slot, b_slot, out_slot, layer) => {
                let layer = ctx.layer_offset + layer;
                let out = unsafe {
                    gated_delta_net_eval(
                        ctx, qkv_slot, z_slot, a_slot, b_slot, layer, bucket, op_idx,
                    )
                };
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            }
            Instruction::GateApply(attn_slot, gate_slot, out_slot) => unsafe {
                // out = attn * sigmoid(gate)  (Qwen3.5 attention output gate).
                let attn_tv = tile_ref(ctx.tiles, attn_slot).as_view(ctx.tiles);
                let gate_tv = tile_ref(ctx.tiles, gate_slot).as_view(ctx.tiles);
                let nt = (*attn_tv).dim(0);
                let ncols = (*attn_tv).dim(1);
                let dt = (*attn_tv).dtype();
                let out = ctx.device.caching.alloc_tensor(&[nt, ncols], dt);
                crate::driver::memcpy_dtod_async(
                    (*out.view()).raw_ptr(),
                    (*attn_tv).as_ptr::<u8>(),
                    (*attn_tv).size_bytes(),
                    ctx.device.compute_stream,
                )
                .expect("GateApply: copy attn");
                kernels::sigmoid_mul_inplace(
                    *out.view(),
                    *gate_tv,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::GateScale(routed_slot, shared_slot, gate_slot, out_slot) => unsafe {
                // out = routed + shared_y * sigmoid(g)  (Qwen3.5-MoE shared
                // expert; g is [T, 1], row-broadcast across the hidden axis).
                // 4-buffer combine kernel reads routed directly — no
                // staging copy (mirrors the metal gate_scale binding).
                let routed_tv = tile_ref(ctx.tiles, routed_slot).as_view(ctx.tiles);
                let shared_tv = tile_ref(ctx.tiles, shared_slot).as_view(ctx.tiles);
                let gate_tv = tile_ref(ctx.tiles, gate_slot).as_view(ctx.tiles);
                let nt = (*routed_tv).dim(0);
                let ncols = (*routed_tv).dim(1);
                let dt = (*routed_tv).dtype();
                let out = ctx.device.caching.alloc_tensor(&[nt, ncols], dt);
                kernels::sigmoid_rowgate_combine(
                    *out.view(),
                    *routed_tv,
                    *shared_tv,
                    *gate_tv,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::GateSplit(qg_slot, q_slot, gate_slot) => unsafe {
                // Per-head deinterleave qg[T, nh*2*hd] → q,gate [T, nh*hd] via
                // gate_split.{metal,cu}. Each head's 2*head_dim block is
                // [query | gate]; the cuda kernel mirrors the metal source byte-for-byte.
                let qg_tv = tile_ref(ctx.tiles, qg_slot).as_view(ctx.tiles);
                let num_heads = W::NUM_Q_HEADS as usize;
                let head_dim = W::HEAD_DIM as usize;
                let (q_out, gate_out) = kernels::gate_split(
                    *qg_tv,
                    num_heads,
                    head_dim,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[q_slot as usize] = Some(TileEntry::Owned(q_out));
                ctx.tiles[gate_slot as usize] = Some(TileEntry::Owned(gate_out));
            },
            Instruction::DeepSeekMoe(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.deepseek_moe_at(bucket, op_idx, 0, layer);
                let out = w.forward(v, ctx.device);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::DeepSeekMoeFp8Block(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.deepseek_moe_fp8_at(bucket, op_idx, 0, layer);
                let out = w.forward(v, ctx.device);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::DeepSeekMoeGgml(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.deepseek_moe_ggml_at(bucket, op_idx, 0, layer);
                let out = w.forward(v, ctx.device);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::FusedMoe(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.fused_moe_at(bucket, op_idx, 0, layer);
                let out = w.forward(v, ctx.device);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::SharedFusedMoe(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.shared_fused_moe_at(bucket, op_idx, 0, layer);
                let out = w.forward(v, ctx.device);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassGemm(in_slot, out_slot, layer, tile_m, tile_n, stages, n, k) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                assert_weight_shape("CutlassGemm", w.dense_weight(), n, k, tp_active(ctx));
                let out = cutlass::cutlass_gemm(
                    *v,
                    w.dense_weight(),
                    cutlass::CutlassTile::new(tile_m, tile_n, stages),
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassGemmSplitK(
                in_slot,
                out_slot,
                layer,
                tile_m,
                tile_n,
                stages,
                split_k,
                n,
                k,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                assert_weight_shape("CutlassGemmSplitK", w.dense_weight(), n, k, tp_active(ctx));
                let out = cutlass::cutlass_gemm_splitk(
                    *v,
                    w.dense_weight(),
                    cutlass::CutlassSplitKTile::new(tile_m, tile_n, stages, split_k),
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassGemmAdd(
                in_slot,
                residual_slot,
                layer,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let residual = tile_ref(ctx.tiles, residual_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                assert_weight_shape("CutlassGemmAdd", w.dense_weight(), n, k, tp_active(ctx));
                cutlass::cutlass_gemm_add(
                    *v,
                    w.dense_weight(),
                    *residual,
                    cutlass::CutlassTile::new(tile_m, tile_n, stages),
                    ctx.device.compute_stream,
                );
            },
            Instruction::CutlassGemv(in_slot, out_slot, layer, n, k) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                assert_weight_shape("CutlassGemv", w.dense_weight(), n, k, tp_active(ctx));
                let out = cutlass::cutlass_gemv(
                    *v,
                    w.dense_weight(),
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassFusedGemmBias(
                in_slot,
                out_slot,
                layer,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                assert_weight_shape(
                    "CutlassFusedGemmBias",
                    w.dense_weight(),
                    n,
                    k,
                    tp_active(ctx),
                );
                let bias = w.dense_bias().expect(
                    "CutlassFusedGemmBias: LinearLayer has no bias — check safetensors path",
                );
                let out = cutlass::cutlass_gemm_bias(
                    *v,
                    w.dense_weight(),
                    bias,
                    cutlass::CutlassTile::new(tile_m, tile_n, stages),
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassFusedGateUpSiluMul(
                in_slot,
                out_slot,
                layer,
                tile_m,
                tile_n,
                stages,
            ) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                let packed = w.dense_weight();
                let gate_w = packed.narrow_dim0(0, W::INTERMEDIATE_SIZE);
                let up_w = packed.narrow_dim0(W::INTERMEDIATE_SIZE, W::INTERMEDIATE_SIZE);
                let up_out = cutlass::cutlass_gemm(
                    *v,
                    up_w,
                    cutlass::CutlassTile::new(tile_m, tile_n, stages),
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = cutlass::cutlass_gemm_silu_mul(
                    *v,
                    gate_w,
                    up_out,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassFusedGateUpGeluMul(
                in_slot,
                out_slot,
                layer,
                tile_m,
                tile_n,
                stages,
                packed_n,
                k,
            ) => unsafe {
                // Mirrors the cuBLAS-peer FusedGateUpGeluMul:
                //   1. ONE GEMM at packed (M, 2I, K) → [M, 2I] intermediate
                //   2. gelu_and_mul_fused over [M, 2I] → [M, I]
                // The GEMM here is a calibrated CUTLASS standalone tile
                // instead of cuBLAS; the elementwise step is identical.
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                assert_weight_shape(
                    "CutlassFusedGateUpGeluMul",
                    w.dense_weight(),
                    packed_n,
                    k,
                    tp_active(ctx),
                );
                let gate_up = cutlass::cutlass_gemm(
                    *v,
                    w.dense_weight(),
                    cutlass::CutlassTile::new(tile_m, tile_n, stages),
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = kernels::gelu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassFusedQkvRopeCache(
                in_slot,
                out_slot,
                layer,
                interleaved,
                tile_m,
                tile_n,
                stages,
                packed_n,
                k,
            ) => unsafe {
                // Mirrors the cuBLAS-peer FusedQkvRopeCache: ONE GEMM at
                // packed (M, q+2*kv, K) → fused_qkv_rope_cache* writing
                // K/V to the paged cache and returning rotated Q. The
                // GEMM here is a calibrated CUTLASS standalone tile
                // instead of cuBLAS; the rope+cache step is identical.
                //
                // Non-biased only — claim is gated to `biased=false` in
                // CutlassFusedQkvRopeCacheImpl::matches; qwen2's biased
                // QKV stays on the cuBLAS peer until the bias-zoo CSV
                // gains shape-swept rows.
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                assert_weight_shape(
                    "CutlassFusedQkvRopeCache",
                    w.dense_weight(),
                    packed_n,
                    k,
                    tp_active(ctx),
                );
                let qkv_packed = cutlass::cutlass_gemm(
                    *v,
                    w.dense_weight(),
                    cutlass::CutlassTile::new(tile_m, tile_n, stages),
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
                let out = if interleaved {
                    kernels::fused_qkv_interleaved_rope_cache(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                } else if ctx.fwd.kv_cache.is_fp8() {
                    kernels::fused_qkv_rope_cache_fp8(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        ctx.fwd.kv_cache.k_scale_ptr(layer as usize),
                        ctx.fwd.kv_cache.v_scale_ptr(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                } else {
                    kernels::fused_qkv_rope_cache(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        W::MROPE_SECTION,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::CutlassFusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
                interleaved,
                tile_m,
                tile_n,
                stages,
                packed_n,
                k_dim,
            ) => {
                let layer = ctx.layer_offset + layer;
                let (q, k_tensor, v_out) = unsafe {
                    let view_in = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                    assert_weight_shape(
                        "CutlassFusedQkvRopePrefill",
                        w.dense_weight(),
                        packed_n,
                        k_dim,
                        tp_active(ctx),
                    );
                    let qkv_packed = cutlass::cutlass_gemm(
                        *view_in,
                        w.dense_weight(),
                        cutlass::CutlassTile::new(tile_m, tile_n, stages),
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
                    if interleaved {
                        kernels::fused_qkv_interleaved_rope(
                            *qkv_packed,
                            *ctx.fwd.positions,
                            cos_sin,
                            W::Q_SIZE,
                            W::KV_SIZE,
                            W::NUM_Q_HEADS as usize,
                            W::NUM_KV_HEADS as usize,
                            W::HEAD_DIM as usize,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                        )
                    } else {
                        kernels::fused_qkv_rope(
                            *qkv_packed,
                            *ctx.fwd.positions,
                            cos_sin,
                            W::Q_SIZE,
                            W::KV_SIZE,
                            W::NUM_Q_HEADS as usize,
                            W::NUM_KV_HEADS as usize,
                            W::HEAD_DIM as usize,
                            W::MROPE_SECTION,
                            &mut ctx.device.caching,
                            ctx.device.compute_stream,
                        )
                    }
                };
                unsafe {
                    ah::write_kv_cache(
                        k_tensor.view(),
                        v_out.view(),
                        ctx.fwd.slot_mapping,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        ctx.device.compute_stream,
                    );
                }
                ctx.tiles[q_out_slot as usize] = Some(TileEntry::Owned(q));
                ctx.tiles[k_out_slot as usize] = Some(TileEntry::Owned(k_tensor));
                ctx.tiles[v_out_slot as usize] = Some(TileEntry::Owned(v_out));
            }
            Instruction::MarlinGemm(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.marlin_at(bucket, op_idx, 0, layer);
                let out = w.forward(v, &mut ctx.device.caching, ctx.device.compute_stream);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::MarlinFusedGateUpSiluMul(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.marlin_at(bucket, op_idx, 0, layer);
                let gate_up = w.forward(v, &mut ctx.device.caching, ctx.device.compute_stream);
                let out = kernels::silu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::MarlinFusedGateUpGeluMul(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.marlin_at(bucket, op_idx, 0, layer);
                let gate_up = w.forward(v, &mut ctx.device.caching, ctx.device.compute_stream);
                let out = kernels::gelu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::MarlinFusedQkvRopeCache(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.marlin_at(bucket, op_idx, 0, layer);
                let qkv_packed = w.forward(v, &mut ctx.device.caching, ctx.device.compute_stream);
                let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
                let out = if ctx.fwd.kv_cache.is_fp8() {
                    kernels::fused_qkv_rope_cache_fp8(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        ctx.fwd.kv_cache.k_scale_ptr(layer as usize),
                        ctx.fwd.kv_cache.v_scale_ptr(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                } else {
                    kernels::fused_qkv_rope_cache(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        W::MROPE_SECTION,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::MarlinFusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
            ) => {
                let layer = ctx.layer_offset + layer;
                let (q, k, v_out) = unsafe {
                    let view_in = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let w = ctx.wm.marlin_at(bucket, op_idx, 0, layer);
                    let qkv_packed =
                        w.forward(view_in, &mut ctx.device.caching, ctx.device.compute_stream);
                    let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
                    kernels::fused_qkv_rope(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::NUM_KV_HEADS as usize,
                        W::HEAD_DIM as usize,
                        W::MROPE_SECTION,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                unsafe {
                    ah::write_kv_cache(
                        k.view(),
                        v_out.view(),
                        ctx.fwd.slot_mapping,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        ctx.device.compute_stream,
                    );
                }
                ctx.tiles[q_out_slot as usize] = Some(TileEntry::Owned(q));
                ctx.tiles[k_out_slot as usize] = Some(TileEntry::Owned(k));
                ctx.tiles[v_out_slot as usize] = Some(TileEntry::Owned(v_out));
            }
            Instruction::GgmlGemm(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                let out = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::GgmlFusedGateUpSiluMul(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                let gate_up = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = kernels::silu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::GgmlFusedGateUpGeluMul(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                let gate_up = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = kernels::gelu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::GgmlFusedQkvRopeCache(in_slot, out_slot, layer, interleaved) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                let qkv_packed = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
                let out = if interleaved {
                    kernels::fused_qkv_interleaved_rope_cache(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                } else if ctx.fwd.kv_cache.is_fp8() {
                    kernels::fused_qkv_rope_cache_fp8(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        ctx.fwd.kv_cache.k_scale_ptr(layer as usize),
                        ctx.fwd.kv_cache.v_scale_ptr(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                } else {
                    kernels::fused_qkv_rope_cache(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        W::MROPE_SECTION,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::GgmlFusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
            ) => {
                let layer = ctx.layer_offset + layer;
                let (q, k, v_out) = unsafe {
                    let view_in = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let w = ctx.wm.linear_at(bucket, op_idx, 0, layer);
                    let qkv_packed = w.forward(
                        view_in,
                        &mut ctx.device.cublas,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
                    kernels::fused_qkv_rope(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::NUM_KV_HEADS as usize,
                        W::HEAD_DIM as usize,
                        W::MROPE_SECTION,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                unsafe {
                    ah::write_kv_cache(
                        k.view(),
                        v_out.view(),
                        ctx.fwd.slot_mapping,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        ctx.device.compute_stream,
                    );
                }
                ctx.tiles[q_out_slot as usize] = Some(TileEntry::Owned(q));
                ctx.tiles[k_out_slot as usize] = Some(TileEntry::Owned(k));
                ctx.tiles[v_out_slot as usize] = Some(TileEntry::Owned(v_out));
            }
            Instruction::Bnb4Gemm(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.bnb4_at(bucket, op_idx, 0, layer);
                let out = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Bnb4FusedGateUpSiluMul(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.bnb4_at(bucket, op_idx, 0, layer);
                let gate_up = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = kernels::silu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Bnb4FusedGateUpGeluMul(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.bnb4_at(bucket, op_idx, 0, layer);
                let gate_up = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = kernels::gelu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Bnb4FusedQkvRopeCache(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.bnb4_at(bucket, op_idx, 0, layer);
                let qkv_packed = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
                let out = if ctx.fwd.kv_cache.is_fp8() {
                    kernels::fused_qkv_rope_cache_fp8(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        ctx.fwd.kv_cache.k_scale_ptr(layer as usize),
                        ctx.fwd.kv_cache.v_scale_ptr(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                } else {
                    kernels::fused_qkv_rope_cache(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        W::MROPE_SECTION,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Bnb4FusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
            ) => {
                let layer = ctx.layer_offset + layer;
                let (q, k, v_out) = unsafe {
                    let view_in = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let w = ctx.wm.bnb4_at(bucket, op_idx, 0, layer);
                    let qkv_packed = w.forward(
                        view_in,
                        &mut ctx.device.cublas,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
                    kernels::fused_qkv_rope(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::NUM_KV_HEADS as usize,
                        W::HEAD_DIM as usize,
                        W::MROPE_SECTION,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                unsafe {
                    ah::write_kv_cache(
                        k.view(),
                        v_out.view(),
                        ctx.fwd.slot_mapping,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        ctx.device.compute_stream,
                    );
                }
                ctx.tiles[q_out_slot as usize] = Some(TileEntry::Owned(q));
                ctx.tiles[k_out_slot as usize] = Some(TileEntry::Owned(k));
                ctx.tiles[v_out_slot as usize] = Some(TileEntry::Owned(v_out));
            }
            Instruction::Fp8Gemm(in_slot, out_slot, layer)
            | Instruction::Fp8FusedGemmBias(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.fp8_at(bucket, op_idx, 0, layer);
                let out = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Fp8FusedGateUpSiluMul(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.fp8_at(bucket, op_idx, 0, layer);
                let gate_up = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = kernels::silu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Fp8FusedGateUpGeluMul(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.fp8_at(bucket, op_idx, 0, layer);
                let gate_up = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let out = kernels::gelu_and_mul_fused(
                    *gate_up,
                    W::INTERMEDIATE_SIZE,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Fp8FusedQkvRopeCache(in_slot, out_slot, layer) => unsafe {
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                let w = ctx.wm.fp8_at(bucket, op_idx, 0, layer);
                let qkv_packed = w.forward(
                    v,
                    &mut ctx.device.cublas,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
                let out = if ctx.fwd.kv_cache.is_fp8() {
                    kernels::fused_qkv_rope_cache_fp8(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        ctx.fwd.kv_cache.k_scale_ptr(layer as usize),
                        ctx.fwd.kv_cache.v_scale_ptr(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                } else {
                    kernels::fused_qkv_rope_cache(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        *ctx.fwd.slot_mapping,
                        *ctx.fwd.kv_cache.k_cache(layer as usize),
                        *ctx.fwd.kv_cache.v_cache(layer as usize),
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::HEAD_DIM as usize,
                        W::MROPE_SECTION,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::Fp8FusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
            ) => {
                let layer = ctx.layer_offset + layer;
                let (q, k, v_out) = unsafe {
                    let view_in = tile_ref(ctx.tiles, in_slot).as_view(ctx.tiles);
                    let w = ctx.wm.fp8_at(bucket, op_idx, 0, layer);
                    let qkv_packed = w.forward(
                        view_in,
                        &mut ctx.device.cublas,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    );
                    let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
                    kernels::fused_qkv_rope(
                        *qkv_packed,
                        *ctx.fwd.positions,
                        cos_sin,
                        W::Q_SIZE,
                        W::KV_SIZE,
                        W::NUM_Q_HEADS as usize,
                        W::NUM_KV_HEADS as usize,
                        W::HEAD_DIM as usize,
                        W::MROPE_SECTION,
                        &mut ctx.device.caching,
                        ctx.device.compute_stream,
                    )
                };
                unsafe {
                    ah::write_kv_cache(
                        k.view(),
                        v_out.view(),
                        ctx.fwd.slot_mapping,
                        ctx.fwd.kv_cache,
                        layer as usize,
                        ctx.device.compute_stream,
                    );
                }
                ctx.tiles[q_out_slot as usize] = Some(TileEntry::Owned(q));
                ctx.tiles[k_out_slot as usize] = Some(TileEntry::Owned(k));
                ctx.tiles[v_out_slot as usize] = Some(TileEntry::Owned(v_out));
            }
            Instruction::AffineQmm(..) => {
                unreachable!(
                    "Instruction::AffineQmm is metal-only — the macro must \
                     not emit it on the cuda forward (Affine weights stay \
                     in StorageFormat::Dense on cuda by the FUF downgrade)"
                );
            }
            Instruction::Nvfp4Qmm(..) => {
                unreachable!(
                    "Instruction::Nvfp4Qmm is metal-only — the macro must \
                     not emit it on the cuda forward (NVFP4 weights stay \
                     in StorageFormat::Dense on cuda by the FUF downgrade)"
                );
            }
            Instruction::SynthPreAttn(..) => {
                unreachable!(
                    "Instruction::SynthPreAttn is metal-only — emitted by the \
                     compiler-driven megakernel synthesis pass on the metal forward only"
                );
            }
            Instruction::SynthMlpPreDown(..) => {
                unreachable!(
                    "Instruction::SynthMlpPreDown is metal-only — emitted by the \
                     compiler-driven megakernel synthesis pass on the metal forward only"
                );
            }
            Instruction::SiluMul(..) => {
                unreachable!(
                    "Instruction::SiluMul is metal-only — emitted by the \
                     decomposed q-MLP path on Affine; cuda's q-MLP routes \
                     through Marlin/Bnb/Fp8/etc fused kernels"
                );
            }
            Instruction::GeluMul(..) => {
                unreachable!(
                    "Instruction::GeluMul is metal-only — emitted by the \
                     decomposed GeGLU q-MLP path on Affine (Gemma2/3/4); \
                     cuda routes GeGLU through its fused dense kernels"
                );
            }
            Instruction::RmsNormUnit(in_slot, out_slot, hidden, _m_multiplier) => unsafe {
                // Unit-gain RMSNorm (gain ≡ 1, NO learned weight — Gemma4 q/k/v
                // per-head norm). Treat the tile as `[rows, hidden]` (per-head
                // width = `hidden`; `rows = numel/hidden = M * m_multiplier`),
                // normalize each row by rsqrt(mean(x²)+eps), restore the shape.
                // Implemented via the `(w + offset)` fold: a zeros gain + offset
                // 1.0 ⇒ unit gain, reusing `rms_norm_with_offset` (no dedicated
                // unit kernel). eps is the model `RMS_NORM_EPS` const (metal's
                // unit kernel uses the same).
                let v = tile_ref(ctx.tiles, in_slot).as_gpu_tensor(ctx.tiles);
                let hid = hidden as usize;
                let rows = v.numel() / hid;
                let orig_shape: Vec<usize> = v.shape().iter().map(|&d| d as usize).collect();
                let in2d = v.reshape(&[rows, hid]);

                let zeros = ctx.device.caching.alloc_tensor(&[hid], v.dtype());
                crate::driver::memset_d8(
                    zeros.as_gpu_tensor().raw_ptr() as *mut u8,
                    0,
                    hid * v.dtype().size_bytes(),
                    ctx.device.compute_stream,
                )
                .expect("RmsNormUnit: zero the unit-gain weight");

                let mut out = kernels::rms_norm_with_offset(
                    in2d,
                    zeros.as_gpu_tensor(),
                    W::RMS_NORM_EPS,
                    1.0,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                // Restore the original tile shape (data is contiguous, identical
                // element count) so downstream ops see `[M, n_heads*head_dim]`.
                out.reshape(&orig_shape, v.dtype());
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::ScalarWeightMul(in_slot, out_slot, layer) => unsafe {
                // out = in * layer_scalar[layer][0] (Gemma4). The `[1]`-shaped
                // gain loads through the RmsNorm-kind accessor, same as metal.
                let layer = ctx.layer_offset + layer;
                let v = tile_ref(ctx.tiles, in_slot).as_gpu_tensor(ctx.tiles);
                let w = ctx.wm.rms_norm_at(bucket, op_idx, 0, layer);
                let out = kernels::scalar_weight_mul(
                    v,
                    w.weight,
                    &mut ctx.device.caching,
                    ctx.device.compute_stream,
                );
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::NormAddScalarMul(..) => {
                unimplemented!("NormAddScalarMul cuda eval is unwired — Gemma4 is metal-first");
            }
            Instruction::RopeAppendNormed(..) => {
                unimplemented!("RopeAppendNormed cuda eval is unwired — Gemma4 is metal-first");
            }
            Instruction::MetalBiasAdd(..) => {
                unreachable!(
                    "Instruction::MetalBiasAdd is metal-only — emitted by \
                     MetalBiasAddImpl for QKV biases on the singleton \
                     (non-synth) path. cuda folds biases into cuBLAS \
                     gemm_bias via FusedGemmBias instead"
                );
            }
            Instruction::MetalFusedMoe(..) | Instruction::MetalSharedFusedMoe(..) => {
                unreachable!(
                    "Instruction::Metal{{,Shared}}FusedMoe is metal-only — \
                     emitted by Metal{{,Shared}}FusedMoeImpl carrying macro-baked \
                     MoE shape for the Metal lowering arm. cuda's MoE path emits \
                     Instruction::{{,Shared}}FusedMoe and reads shape from the \
                     runtime FusedMoELayer / SharedFusedMoELayer struct"
                );
            }
            Instruction::GemmaMoe(
                router_in,
                expert_in,
                out_slot,
                layer,
                _num_experts,
                _top_k,
                _moe_inter,
                _hidden,
                _group_size,
                _bits,
            ) => unsafe {
                // The sparse SwitchGLU block: folded-gain router RMSNorm + dense
                // router GEMM + top-k + softmax(temp=hidden^-0.5) +
                // per_expert_scale, then the GeGLU experts and the weighted sum.
                // Shape lives in the runtime layer structs (the fat Instruction
                // shape fields are ignored, matching the FusedMoe arms). The two
                // activation tiles are router_in (post-attention residual) and
                // expert_in (ff2-normed). Gemma-4 rms_norm_eps = 1e-6.
                let layer = ctx.layer_offset + layer;
                let xr = tile_ref(ctx.tiles, router_in).as_view(ctx.tiles);
                let xe = tile_ref(ctx.tiles, expert_in).as_view(ctx.tiles);
                // `gemma_router_at` and `gemma_switch_glu_at` are SEPARATE
                // weight-accessor maps, each with its own slot numbering, so the
                // single expert bundle is slot 0 of the switch_glu map (NOT slot
                // 1 — that stale index, a holdover from a shared-slot design,
                // tripped the weight-accessor coverage check the first time
                // gemma4-moe ran on cuda).
                let router = ctx.wm.gemma_router_at(bucket, op_idx, 0, layer);
                let experts = ctx.wm.gemma_switch_glu_at(bucket, op_idx, 0, layer);
                let out =
                    crate::layers_moe::gemma_moe_forward(router, experts, xr, xe, 1e-6, ctx.device);
                ctx.tiles[out_slot as usize] = Some(TileEntry::Owned(out));
            },
            Instruction::StripCls(_, _) => {
                unimplemented!("Instruction::StripCls eval is unwired");
            }
            Instruction::Loop(..) => {
                unreachable!("Instruction::Loop should be handled by run(), not eval()");
            }
            Instruction::Alias(dst, src) => {
                ctx.tiles[dst as usize] = Some(view(src));
            }
            Instruction::Free(slot) => {
                ctx.tiles[slot as usize] = None;
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
unsafe fn mla_attention_eval<W: CanonicalParams>(
    ctx: &mut InterpreterCtx<'_, W>,
    q_slot: u32,
    kv_b_slot: u32,
    k_pe_slot: u32,
    layer: u32,
    bucket: u32,
    op_idx: u32,
) -> OwnedTensor {
    unsafe {
        let q_tv = tile_ref(ctx.tiles, q_slot).as_view(ctx.tiles);
        let kv_b_tv = tile_ref(ctx.tiles, kv_b_slot).as_view(ctx.tiles);
        let k_pe_tv = tile_ref(ctx.tiles, k_pe_slot).as_view(ctx.tiles);
        let cos_sin = ctx.wm.cos_sin_at(bucket, op_idx, 0, layer);
        let nt = (*q_tv).dim(0);
        let dt = (*q_tv).dtype();

        let q_pe = ctx
            .device
            .caching
            .alloc_tensor(&[nt, (W::NUM_Q_HEADS as usize) * W::QK_ROPE_HEAD_DIM], dt);
        kernels::mla_extract_q_pe(
            *q_tv,
            *q_pe
                .view()
                .reshape(&[nt, (W::NUM_Q_HEADS as usize) * W::QK_ROPE_HEAD_DIM]),
            W::NUM_Q_HEADS as usize,
            W::QK_HEAD_DIM,
            W::QK_NOPE_HEAD_DIM,
            W::QK_ROPE_HEAD_DIM,
            ctx.device.compute_stream,
        );

        kernels::rotary_embedding_interleaved_inplace(
            *q_pe
                .view()
                .reshape(&[nt, (W::NUM_Q_HEADS as usize) * W::QK_ROPE_HEAD_DIM]),
            *k_pe_tv,
            *ctx.fwd.positions,
            cos_sin,
            W::QK_ROPE_HEAD_DIM,
            ctx.device.compute_stream,
        );

        kernels::mla_write_q_pe(
            *q_pe
                .view()
                .reshape(&[nt, (W::NUM_Q_HEADS as usize) * W::QK_ROPE_HEAD_DIM]),
            *q_tv,
            W::NUM_Q_HEADS as usize,
            W::QK_HEAD_DIM,
            W::QK_NOPE_HEAD_DIM,
            W::QK_ROPE_HEAD_DIM,
            ctx.device.compute_stream,
        );
        drop(q_pe);

        let k = ctx
            .device
            .caching
            .alloc_tensor(&[nt, (W::NUM_Q_HEADS as usize) * W::QK_HEAD_DIM], dt);
        kernels::mla_assemble_k(
            *kv_b_tv,
            *k_pe_tv,
            *k.view()
                .reshape(&[nt, (W::NUM_Q_HEADS as usize) * W::QK_HEAD_DIM]),
            W::NUM_Q_HEADS as usize,
            W::QK_NOPE_HEAD_DIM,
            W::QK_ROPE_HEAD_DIM,
            W::V_HEAD_DIM,
            W::QK_HEAD_DIM,
            ctx.device.compute_stream,
        );

        let v = ctx
            .device
            .caching
            .alloc_tensor(&[nt, (W::NUM_Q_HEADS as usize) * W::QK_HEAD_DIM], dt);
        crate::driver::memset_d8(
            (*v.view()).raw_ptr(),
            0,
            (*v.view()).size_bytes(),
            ctx.device.compute_stream,
        )
        .expect("MLA: memset V");
        kernels::mla_assemble_v(
            *kv_b_tv,
            *v.view()
                .reshape(&[nt, (W::NUM_Q_HEADS as usize) * W::QK_HEAD_DIM]),
            W::NUM_Q_HEADS as usize,
            W::QK_NOPE_HEAD_DIM,
            W::V_HEAD_DIM,
            W::QK_HEAD_DIM,
            ctx.device.compute_stream,
        );

        let k_tv = k.view();
        let k_3d = k_tv.reshape(&[nt, W::NUM_Q_HEADS as usize, W::QK_HEAD_DIM]);
        let v_tv = v.view();
        let v_3d = v_tv.reshape(&[nt, W::NUM_Q_HEADS as usize, W::QK_HEAD_DIM]);
        ah::write_kv_cache(
            k_3d,
            v_3d,
            ctx.fwd.slot_mapping,
            ctx.fwd.kv_cache,
            layer as usize,
            ctx.device.compute_stream,
        );

        let q_3d = q_tv.reshape(&[nt, W::NUM_Q_HEADS as usize, W::QK_HEAD_DIM]);
        let attn = ah::attention_standard(
            q_3d,
            k_3d,
            v_3d,
            ctx.fwd.cu_seqlens_q,
            ctx.fwd.seqused_k,
            ctx.fwd.block_table_for(layer as usize),
            ctx.fwd.max_seqlen_q,
            ctx.fwd.max_seqlen_k,
            W::MLA_ATTN_SCALE,
            ctx.fwd.kv_cache,
            layer as usize,
            ctx.device.num_sm,
            &mut ctx.device.caching,
            ctx.device.compute_stream,
            ::std::ptr::null(),
            0,
            false,
        );
        drop(k);
        drop(v);

        let sliced = ctx
            .device
            .caching
            .alloc_tensor(&[nt, (W::NUM_Q_HEADS as usize) * W::V_HEAD_DIM], dt);
        let attn_tv = attn.view();
        let attn_flat = attn_tv.reshape(&[nt, (W::NUM_Q_HEADS as usize) * W::QK_HEAD_DIM]);
        kernels::mla_slice_attn_output(
            *attn_flat,
            *sliced.view(),
            W::NUM_Q_HEADS as usize,
            W::QK_HEAD_DIM,
            W::V_HEAD_DIM,
            ctx.device.compute_stream,
        );
        drop(attn);
        sliced
    }
}

/// Cast a model-dtype `GpuTensor` to a fresh f32 `OwnedTensor` of the given
/// shape (elementwise; `cast_bias_to_f32` handles F32/F16/BF16). The shape's
/// element count must equal `src.numel()`.
#[cfg(feature = "cuda")]
unsafe fn gdn_cast_to_f32(
    caching: &mut crate::alloc::CachingAllocator,
    src: crate::tensor::GpuTensor,
    shape: &[usize],
    stream: crate::CUstream,
) -> OwnedTensor {
    let dst = caching.alloc_tensor(shape, crate::dtype::DType::F32);
    unsafe { kernels::cast_bias_to_f32(src, *dst.view(), stream) };
    dst
}

/// Read `count` i32s from a device tensor to host (blocking). Used by the GDN
/// prefill path to slice per-sequence token ranges.
#[cfg(feature = "cuda")]
unsafe fn gdn_dtoh_i32(
    t: crate::tensor::TensorView<'_>,
    count: usize,
    stream: crate::CUstream,
) -> Vec<i32> {
    let mut host = vec![0i32; count];
    unsafe {
        crate::driver::memcpy_dtoh_async(
            host.as_mut_ptr() as *mut u8,
            (*t).as_ptr::<u8>(),
            count * 4,
            stream,
        )
        .expect("gdn_dtoh_i32: D2H copy failed");
        crate::driver::stream_synchronize(stream).expect("gdn_dtoh_i32: sync");
    }
    host
}

/// Sibling of [`gdn_dtoh_i32`] for u32 tensors — used by the GDN eval
/// to read the per-seq `is_fresh` mask before zeroing recycled-slot state.
#[cfg(feature = "cuda")]
unsafe fn gdn_dtoh_u32(
    t: crate::tensor::TensorView<'_>,
    count: usize,
    stream: crate::CUstream,
) -> Vec<u32> {
    let mut host = vec![0u32; count];
    unsafe {
        crate::driver::memcpy_dtoh_async(
            host.as_mut_ptr() as *mut u8,
            (*t).as_ptr::<u8>(),
            count * 4,
            stream,
        )
        .expect("gdn_dtoh_u32: D2H copy failed");
        crate::driver::stream_synchronize(stream).expect("gdn_dtoh_u32: sync");
    }
    host
}

/// Gated-DeltaNet eval — orchestrates the surviving `gdn_*` kernels for one
/// linear-attention layer. Validated op-by-op against the cpu_golden oracle
/// (`cpu_golden::gdn_*`):
///   1. cast qkv/z/a/b (model dtype) → f32
///   2. causal conv1d(+SiLU): decode = batched `gdn_conv1d_update` over
///      `state_indices`; prefill = per-sequence `gdn_conv1d_prefill`
///   3. `gdn_conv_split` → q/k/v
///   4. `gdn_gating` → g (= -exp(A_log)·softplus(a+dt_bias)), beta (= sigmoid(b))
///   5. `gdn_recurrent_fwd` (delta-rule scan; **scale = 1/sqrt(head_k)** —
///      the deleted Qwen3-Next path passed 1.0, which dropped the q-scale and
///      was a source of its flakiness)
///   6. `gdn_rms_norm_gated` (per value-head; **SiLU(z)** gate)
///   7. cast f32 core → model dtype (the DSL `out_proj` gemm follows)
///
/// Reads the `linear_attn[layer]` weight bundle via `gated_delta_net_at` and
/// the ambient conv/ssm state from `ForwardCtx::{gdn_state, gdn_state_indices}`.
/// The worker zero-inits a sequence's slot on its first (fresh) forward, so the
/// state read here is always valid (the "degeneration after N requests" guard).
/// NOTE: multi-sequence prefill in one batch (chunked/batched prefill) is the
/// P6 follow-up; bring-up exercises single-sequence prefill + batched decode.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
unsafe fn gated_delta_net_eval<W: CanonicalParams>(
    ctx: &mut InterpreterCtx<'_, W>,
    qkv_slot: u32,
    z_slot: u32,
    a_slot: u32,
    b_slot: u32,
    layer: u32,
    bucket: u32,
    op_idx: u32,
) -> OwnedTensor {
    unsafe {
        use crate::dtype::DType;
        use crate::tensor::GpuTensor;

        let stream = ctx.device.compute_stream;

        // Dims from the per-arch CanonicalParams consts.
        let nk = W::GDN_NUM_K_HEADS as usize;
        let nv = W::GDN_NUM_V_HEADS as usize;
        let hk = W::GDN_HEAD_K_DIM as usize;
        let hv = W::GDN_HEAD_V_DIM as usize;
        let kernel = W::GDN_CONV_KERNEL as usize;
        let conv_dim = W::GDN_CONV_DIM;
        let key_dim = nk * hk;
        let value_dim = nv * hv;

        // Input activation tiles (model dtype, from the in_proj_* gemms).
        let qkv_tv = tile_ref(ctx.tiles, qkv_slot).as_view(ctx.tiles);
        let z_tv = tile_ref(ctx.tiles, z_slot).as_view(ctx.tiles);
        let a_tv = tile_ref(ctx.tiles, a_slot).as_view(ctx.tiles);
        let b_tv = tile_ref(ctx.tiles, b_slot).as_view(ctx.tiles);
        let nt = (*qkv_tv).dim(0);
        let model_dt = (*qkv_tv).dtype();

        // Weight bundle (copy the Copy GpuTensors out so the &wm borrow ends).
        let (conv1d_w, a_log_w, dt_bias_w, norm_w) = {
            let w = ctx.wm.gated_delta_net_at(bucket, op_idx, 0, layer);
            (w.conv1d, w.a_log, w.dt_bias, w.norm)
        };

        // Ambient recurrent/conv state for this layer + the per-seq slot ids.
        let gdn_state = ctx
            .fwd
            .gdn_state
            .expect("GatedDeltaNet eval: ForwardCtx.gdn_state is None (worker did not build the GDN state pool)");
        let state_indices = ctx
            .fwd
            .gdn_state_indices
            .expect("GatedDeltaNet eval: ForwardCtx.gdn_state_indices is None");
        let is_fresh_view = ctx
            .fwd
            .gdn_is_fresh
            .expect("GatedDeltaNet eval: ForwardCtx.gdn_is_fresh is None");
        let conv_state = gdn_state.conv_state(layer as usize);
        let ssm_state = gdn_state.ssm_state(layer as usize);
        let num_seqs = (*state_indices).dim(0);

        // Honor the per-seq `is_fresh` flag by zeroing the recurrent
        // conv/ssm regions for any slot whose owning request is on its
        // first forward (slot just allocated, OR a recycled slot whose
        // prior owner finished). Without this the cuda gdn_conv1d /
        // gdn_recurrent_fwd kernels would continue from the prior
        // sequence's stale state — silent decode corruption that
        // surfaces as identical-token loops on the second prompt
        // through a recycled slot. The metal kernels gate the zero-
        // init INSIDE the kernel via the `is_fresh` buffer; cuda
        // doesn't (no fresh-aware branch in
        // `crates/scratchy-serving-cuda/csrc/gdn_*_kernels.cu`), so we pre-zero
        // here on the same compute stream.
        let fresh_host = gdn_dtoh_u32(is_fresh_view, num_seqs, stream);
        let slots_for_fresh = gdn_dtoh_i32(state_indices, num_seqs, stream);
        // conv_state shape: [num_slots, conv_dim, conv_state_len]
        // ssm_state shape:  [num_slots, num_v_heads, head_v_dim, head_k_dim]
        let conv_slot_bytes = conv_state.dim(1) * conv_state.dim(2) * DType::F32.size_bytes();
        let ssm_slot_bytes =
            ssm_state.dim(1) * ssm_state.dim(2) * ssm_state.dim(3) * DType::F32.size_bytes();
        for s in 0..num_seqs {
            if fresh_host[s] == 0 {
                continue;
            }
            let slot = slots_for_fresh[s] as usize;
            let conv_off = slot * conv_slot_bytes;
            let ssm_off = slot * ssm_slot_bytes;
            crate::driver::memset_d8(
                conv_state.raw_ptr().add(conv_off),
                0,
                conv_slot_bytes,
                stream,
            )
            .expect("GDN fresh: zero conv_state slot");
            crate::driver::memset_d8(ssm_state.raw_ptr().add(ssm_off), 0, ssm_slot_bytes, stream)
                .expect("GDN fresh: zero ssm_state slot");
        }

        // 1. Cast inputs + small weights to f32 (the gdn_* kernels are f32).
        let mixed_qkv = gdn_cast_to_f32(&mut ctx.device.caching, *qkv_tv, &[nt, conv_dim], stream);
        let z_f32 = gdn_cast_to_f32(&mut ctx.device.caching, *z_tv, &[nt, value_dim], stream);
        let a_f32 = gdn_cast_to_f32(&mut ctx.device.caching, *a_tv, &[nt, nv], stream);
        let b_f32 = gdn_cast_to_f32(&mut ctx.device.caching, *b_tv, &[nt, nv], stream);
        // conv1d.weight on-disk is [conv_dim, 1, kernel]; view it as [conv_dim, kernel].
        let conv_w_f32 = gdn_cast_to_f32(
            &mut ctx.device.caching,
            conv1d_w,
            &[conv_dim, kernel],
            stream,
        );
        let a_log_f32 = gdn_cast_to_f32(&mut ctx.device.caching, a_log_w, &[nv], stream);
        let dt_bias_f32 = gdn_cast_to_f32(&mut ctx.device.caching, dt_bias_w, &[nv], stream);
        let norm_w_f32 = gdn_cast_to_f32(&mut ctx.device.caching, norm_w, &[hv], stream);

        // 2. Causal conv1d (+SiLU), writing into conv_out and updating the ring.
        let conv_out = ctx.device.caching.alloc_tensor(&[nt, conv_dim], DType::F32);
        if num_seqs == nt {
            // Decode: one token per sequence — batched ring update.
            kernels::gdn_conv1d_update(
                *conv_state,
                *mixed_qkv.view(),
                *conv_w_f32.view(),
                *conv_out.view(),
                *state_indices,
                conv_dim,
                kernel,
                num_seqs,
                stream,
            );
        } else {
            // Prefill: per-sequence, seeding the ring from zero (worker-zeroed
            // on fresh). Slice the token axis via cu_seqlens_q.
            let cu = gdn_dtoh_i32(ctx.fwd.cu_seqlens_q, num_seqs + 1, stream);
            let slots = gdn_dtoh_i32(state_indices, num_seqs, stream);
            let f32_sz = DType::F32.size_bytes();
            for s in 0..num_seqs {
                let seq_start = cu[s] as usize;
                let seq_len = cu[s + 1] as usize - seq_start;
                if seq_len == 0 {
                    continue;
                }
                let slot_idx = slots[s] as usize;
                let byte_off = seq_start * conv_dim * f32_sz;
                let x_view = GpuTensor::new(
                    (*mixed_qkv.view()).raw_ptr().add(byte_off),
                    &[seq_len, conv_dim],
                    DType::F32,
                );
                let out_view = GpuTensor::new(
                    (*conv_out.view()).raw_ptr().add(byte_off),
                    &[seq_len, conv_dim],
                    DType::F32,
                );
                kernels::gdn_conv1d_prefill(
                    *conv_state,
                    x_view,
                    *conv_w_f32.view(),
                    out_view,
                    slot_idx,
                    conv_dim,
                    kernel,
                    seq_len,
                    stream,
                );
            }
        }
        drop(mixed_qkv);

        // 3. Split conv output into q/k/v (f32).
        let (q_owned, k_owned, v_owned) = kernels::gdn_conv_split(
            *conv_out.view(),
            nt,
            nk,
            nv,
            hk,
            hv,
            key_dim,
            value_dim,
            conv_dim,
            &mut ctx.device.caching,
            stream,
        );
        drop(conv_out);

        // 4. Input-dependent gating: g, beta.
        let g = ctx.device.caching.alloc_tensor(&[nt, nv], DType::F32);
        let beta = ctx.device.caching.alloc_tensor(&[nt, nv], DType::F32);
        kernels::gdn_gating(
            *g.view(),
            *beta.view(),
            *a_log_f32.view(),
            *a_f32.view(),
            *b_f32.view(),
            *dt_bias_f32.view(),
            nv,
            nt,
            stream,
        );

        // 5. Recurrent delta-rule scan. scale = 1/sqrt(head_k) (q-scale; the
        // kernel applies it after L2-normalization — see gdn_recurrent_kernels.cu).
        let o = ctx.device.caching.alloc_tensor(&[nt, nv, hv], DType::F32);
        let scale = 1.0f32 / (hk as f32).sqrt();
        kernels::gdn_recurrent_fwd(
            *q_owned.view(),
            *k_owned.view(),
            *v_owned.view(),
            *g.view(),
            *beta.view(),
            *o.view(),
            *ssm_state,
            *state_indices,
            *ctx.fwd.cu_seqlens_q,
            scale,
            num_seqs,
            nt,
            nk,
            nv,
            hk,
            hv,
            stream,
        );
        drop(q_owned);
        drop(k_owned);
        drop(v_owned);
        drop(g);
        drop(beta);

        // 6. Gated RMSNorm (per value-head over head_v; SiLU(z) gate).
        let total_rows = nt * nv;
        let o_flat = o.view().reshape(&[total_rows, hv]);
        let z_flat = z_f32.view().reshape(&[total_rows, hv]);
        let normed = ctx
            .device
            .caching
            .alloc_tensor(&[total_rows, hv], DType::F32);
        kernels::gdn_rms_norm_gated(
            *o_flat,
            *z_flat,
            *norm_w_f32.view(),
            *normed.view(),
            W::RMS_NORM_EPS,
            hv,
            total_rows,
            stream,
        );
        drop(o);

        // 7. Cast core back to model dtype as [nt, value_dim] for the out_proj
        // gemm the DSL applies next.
        let normed_flat = normed.view().reshape(&[nt, value_dim]);
        if model_dt == DType::F32 {
            normed
        } else {
            let out =
                kernels::cast_from_f32(*normed_flat, model_dt, &mut ctx.device.caching, stream);
            drop(normed);
            out
        }
    }
}

/// Walk one slice in-place against ctx. `Instruction::Loop(count,
/// body_len)` re-runs the next `body_len` instructions `count`
/// times with `ctx.layer_offset` set to the iter index.
///
/// `bucket` is the static bucket id (e.g. `BACKBONE`, `LM_HEAD`)
/// the proc-macro emits; it is forwarded together with each op's
/// flat position in the bucket to `Instruction::eval`, which
/// passes both to the per-arch [`WeightAccessors`] impl.
///
/// Loop bodies use their absolute position in the bucket (not the
/// position within the body), so the per-arch match table is keyed
/// uniformly on `(bucket, flat_position)` regardless of whether
/// the op sits inside a `Loop` body.
#[cfg(feature = "cuda")]
unsafe fn run_slice<W: CanonicalParams>(
    instructions: &[Instruction],
    bucket: u32,
    ctx: &mut InterpreterCtx<'_, W>,
) {
    let mut i = 0usize;
    while i < instructions.len() {
        match instructions[i] {
            Instruction::Loop(count, body_len, layer_stride) => {
                let body_start = i + 1;
                let body_end = body_start + body_len as usize;
                let body = &instructions[body_start..body_end];
                for l in 0..count {
                    // ⛔ SCALED BY THE STRIDE. A one-layer body advances the layer by one per
                    // iteration; a body spanning a multi-layer cell (gemma-4's `SSSSSG`) advances
                    // by the cell's height. Taking `l` raw would bind layer `l` for iteration `l`
                    // of a six-layer cell.
                    ctx.layer_offset = l * layer_stride;
                    for (j, instr) in body.iter().enumerate() {
                        let body_op_idx = (body_start + j) as u32;
                        unsafe {
                            instr.eval(ctx, bucket, body_op_idx);
                        }
                    }
                }
                ctx.layer_offset = 0;
                i = body_end;
            }
            instr => {
                unsafe {
                    instr.eval(ctx, bucket, i as u32);
                }
                i += 1;
            }
        }
    }
}

/// Run backbone followed by lm_head against one tile table.
///
/// # Safety
/// Both slices well-formed; tile slot indices in range; weight
/// accessor fns produce live GPU memory.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn run<W: CanonicalParams>(
    backbone: &[Instruction],
    backbone_bucket: u32,
    lm_head: &[Instruction],
    lm_head_bucket: u32,
    wm: &W,
    fwd: &ForwardCtx,
    device: &mut GpuDevice,
    num_slots: u32,
    terminal_slot: u32,
) -> OwnedTensor {
    let mut tiles: Vec<Option<TileEntry>> = (0..num_slots).map(|_| None).collect();
    let mut ctx = InterpreterCtx {
        wm,
        tiles: &mut tiles,
        fwd,
        device,
        layer_offset: 0,
        kv_layer: 0,
        pinned_owned: Vec::new(),
    };
    unsafe {
        run_slice(backbone, backbone_bucket, &mut ctx);
        run_slice(lm_head, lm_head_bucket, &mut ctx);
    }
    take_owned(&mut tiles, terminal_slot)
}

/// Backbone-only run: returns a memcpy'd backbone tile so it
/// outlives the per-call tile table.
///
/// # Safety
/// Same as [`run`].
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_backbone<W: CanonicalParams>(
    backbone: &[Instruction],
    backbone_bucket: u32,
    wm: &W,
    fwd: &ForwardCtx,
    device: &mut GpuDevice,
    num_slots: u32,
    backbone_slot: u32,
) -> OwnedTensor {
    let mut tiles: Vec<Option<TileEntry>> = (0..num_slots).map(|_| None).collect();
    let mut ctx = InterpreterCtx {
        wm,
        tiles: &mut tiles,
        fwd,
        device,
        layer_offset: 0,
        kv_layer: 0,
        pinned_owned: Vec::new(),
    };
    unsafe {
        run_slice(backbone, backbone_bucket, &mut ctx);
    }
    let bb_view = unsafe { tile_ref(&tiles, backbone_slot).as_view(&tiles) };
    let bb_shape_u32: &[u32] = bb_view.shape();
    let bb_shape: Vec<usize> = bb_shape_u32.iter().map(|&d| d as usize).collect();
    let bb_out = device.caching.alloc_tensor(&bb_shape, bb_view.dtype());
    unsafe {
        crate::driver::memcpy_dtod_async(
            bb_out.raw_ptr(),
            bb_view.raw_ptr() as *const u8,
            bb_view.size_bytes(),
            device.compute_stream,
        )
        .expect("run_backbone: DtoD memcpy of output");
    }
    bb_out
}

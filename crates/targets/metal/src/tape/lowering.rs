// SPDX-License-Identifier: Apache-2.0
//! Lowering pass: a bucket's [`MetalStepTape`] → `LoweredMetalTape`.
//!
//! Pure CPU code. No Metal device required, no kernel dispatch — just
//! a structural translation that:
//!
//! 1. Records each [`StepRow::Loop`] as a `TapeLoop` over its body, which
//!    is lowered ONCE: iteration `i` is the body with every layer advanced
//!    by `i * stride`, applied at load (`Binding::bump_layer`).
//! 2. Emits no command for the metadata-only `Reshape`.
//! 3. For each compute step, picks the right `KernelId`,
//!    derives a `DispatchShape` from `bucket_m` + the kernel's
//!    convention, and produces `Binding`s pointing at arena slots,
//!    typed weight thunks, or runtime buffers.
//!
//! The match over [`MetalStep`] is exhaustive: a new step kind fails to
//! compile here (E0004) until its arm exists.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::tape::ids::ArenaSlotIdx as Slot;
use crate::tape::ids::SourceIx;
use crate::tape::kernel_bindings::{CosSinTable, source};
use crate::tape::model_consts::MetalModelConsts;
use scratchy_ir::{KvCodec, TqBits};
use crate::tape::step::{
    AffineBits, AffineGroupSize, AffineMatmul, AttnMask, BiasStorage, CuSeqlens, ExpertProj,
    GainOffset, GatedAct, GatherIndices, HiddenSize, IntermediateSize, KDim, KvOffsets, KvOperand,
    LayerId, MetalStep, MetalStepTape, MoeBlock, MoeRegion, MoeRows, MoeScores, MoeStep, NDim,
    QmvBatchLimit, RopeFormTag, RotaryTables, RotatedRows, RouterInput, RowSource, RowsDivisor,
    RowsPerToken, SampleRowsStep, Scale, StepRow,
};
use scratchy_subtile::handoff::WeightKind;

/// Set by the worker when a draft model is loaded. The
/// `attention_via_cache_v2_*` kernel's BPC=0 fast path has an unresolved
/// interaction with the draft K-step chain on 3B-class+ models (symptom:
/// out-of-vocab token IDs from the draft proposer). Forcing the chunked
/// addressing path keeps spec-decode correct; non-spec-decode workloads
/// keep the fast path. One-shot at worker init; the lowering pass picks
/// up the value at first forward.
static FORCE_CHUNKED_ADDRESSING: AtomicBool = AtomicBool::new(false);

/// Disable the BPC=0 attention reader fast path for the lifetime of this
/// process. The worker calls this when a draft model is loaded.
pub fn force_chunked_attention_addressing() {
    FORCE_CHUNKED_ADDRESSING.store(true, Ordering::Relaxed);
}

/// Whether chunked addressing is currently forced (spec-decode). The
/// pool reads this at build time to select the matching baked tape
/// variant.
pub fn chunked_attention_addressing_forced() -> bool {
    FORCE_CHUNKED_ADDRESSING.load(Ordering::Relaxed)
}

/// Value passed as `ATTN_BLOCKS_PER_CHUNK` to the attention reader's
/// pipeline. `0` selects the direct-addressing fast path: the kernel
/// treats `k_cache[0]` as the layer base and computes
/// `physical_block * kv_blk_stride` directly (no per-block chunk-table
/// load, no modulo). Safe because the worker installs one MTLBuffer per
/// `(layer, K/V)` with `chunk_table[0]` filled to the layer base. Falls
/// back to the chunked addressing path when
/// [`force_chunked_attention_addressing`] has been called.
fn attention_blocks_per_chunk(chunked: bool) -> u32 {
    if chunked { crate::BLOCKS_PER_CHUNK } else { 0 }
}
use crate::quantized::{
    DequantDtype, QmmTKernel, QmvKernel, SMALL_M_TILE_COLS, ScaleDtype, SmallMTile, W4A8_TILE_ROWS,
    W4a8Rows, W4a8Tile, pick_qmm_t_kernel, pick_qmv_kernel, qmm_t_dispatch_shape,
    qmm_t_kernel_static_name, qmm_t_kernel_static_name_with_compute, qmm_w4a8_static_name,
    qmv_dispatch_shape, qmv_kernel_static_name, small_m_kernel_static_name,
    splitk_reduce_kernel_static_name, w4a8_quant_static_name, w4a8_scratch_bytes,
};
use crate::specialized_pipeline_cache::ConstantValue;

use crate::tape::lowered::{
    Binding, DispatchShape, GatedCommand, GemmDims, IntoBaked, KernelId, LoweredCommand,
    LoweredMetalTape, LoweringError, MetalDtype, RuntimeBindingKind, WeightTensor, baked,
};

/// Where one bucket's tape is baked: the bucket, its two halves' tape
/// indices, and the runtime inputs the bake varies over.
#[derive(Clone, Copy)]
pub struct BakePoint<'a> {
    /// Chunked-addressing attention variant (spec-decode). A PARAMETER,
    /// not process state: the macro bakes both variants from parallel
    /// threads, and a global here was a cross-model probe race.
    pub chunked: bool,
    pub bucket_m: u32,
    pub num_arena_slots: u32,
    /// The rotary table each attention class re-ropes cached K with (rope-on-read); `None`
    /// when the model has no rotary.
    pub rotary: Option<RotaryTables>,
    /// Runtime per-sequence block-table capacity (`KvCachePool::max_blocks_per_seq`).
    /// Replaces the compile-time `p.max_blocks_per_seq` at the `MaxBlocksPerSeq`
    /// function-constant sites + the rope-once `roped_k_scratch` sizing.
    pub block_cap: u32,
    pub profile: Option<&'a crate::targets::MetalTargetProfile>,
}

/// The weights a row can bind: its site's sources and the model's class rotary tables.
#[derive(Clone, Copy)]
struct RowSources<'a> {
    index: usize,
    site: &'a [RowSource],
    rotary: Option<RotaryTables>,
}

impl<'a> RowSources<'a> {
    fn of_row(sites: &'a [Vec<RowSource>], index: usize, at: &BakePoint<'_>) -> Self {
        let site = sites.get(index).map_or(&[][..], Vec::as_slice);
        let rotary = at.rotary;
        Self {
            index,
            site,
            rotary,
        }
    }

    /// The `slot`-th `kind` source of the row's site (`handoff::accessor_slot`'s rule).
    fn of(self, kind: WeightKind, slot: u32) -> Result<SourceIx, LoweringError> {
        let mut same = self.site.iter().filter(|s| s.kind == kind);
        let index = self.index;
        let ix = same.nth(slot as usize).map(|s| s.ix);
        ix.ok_or(LoweringError::NoSource { index, kind, slot })
    }

    /// The table the row's rope reads: its site's first cos/sin, or MRoPE's per-forward one.
    fn cos_sin(self, p: &MetalModelConsts) -> Result<CosSinTable, LoweringError> {
        match p.mrope {
            true => Ok(CosSinTable::Mrope),
            false => self.of(WeightKind::CosSin, 0).map(CosSinTable::Static),
        }
    }

    /// The rotary table a rope-on-read attention of layer class `is_global` re-ropes with.
    fn table(self, is_global: bool) -> Result<SourceIx, LoweringError> {
        let index = self.index;
        let tables = self.rotary.ok_or(LoweringError::NoRotaryTable { index })?;
        Ok(tables.of(is_global))
    }

    /// [`Self::table`] of `class` (`is_global`), when rope-on-read is on (`Some`).
    fn rotary(self, class: Option<bool>) -> Result<Option<SourceIx>, LoweringError> {
        class.map(|g| self.table(g)).transpose()
    }
}

/// Lower one bucket's step tape, `backbone ++ lm_head`.
///
/// Concatenation matches the cuda interpreter's effective behavior:
/// `forward()` runs backbone then lm_head in sequence for a given
/// bucket. Lowering them as one stream lets the worker bake both
/// halves into the bucket's single dispatch plan, with no extra mid-bucket
/// boundary the caller needs to manage. Loop bodies never span the two
/// halves; if one ever did, the malformed-loop check fires inside the
/// offending half and surfaces the per-half index.
pub fn lower_subtile_tape_to_metal(
    steps: &MetalStepTape,
    p: &MetalModelConsts,
    at: BakePoint<'_>,
) -> Result<LoweredMetalTape, LoweringError> {
    let half = |rows, barriers, sources| lower(p, rows, barriers, sources, &at);
    let bb = half(
        &steps.backbone,
        &steps.backbone_barriers,
        &steps.backbone_sources,
    )?;
    let lh = half(
        &steps.lm_head,
        &steps.lm_head_barriers,
        &steps.lm_head_sources,
    )?;
    let binds_kv_cache = |c: &GatedCommand| {
        c.command.bindings.iter().any(|b| {
            matches!(
                b,
                Binding::Runtime {
                    kind: RuntimeBindingKind::KvCacheK { .. } | RuntimeBindingKind::KvCacheV { .. },
                    ..
                }
            )
        })
    };
    if p.kv_codec.is_turboquant()
        && bb.commands.iter().any(binds_kv_cache)
        && !bb
            .commands
            .iter()
            .any(|c| c.command.kernel == KernelId::TqQuantizeToPacked)
    {
        return Err(LoweringError::TurboQuantCompressesNothing);
    }
    Ok(LoweredMetalTape {
        bucket_m: at.bucket_m,
        num_arena_slots: at.num_arena_slots,
        commands: baked([bb.commands, lh.commands].concat()),
        barrier_before: baked([bb.barrier_before, lh.barrier_before].concat()),
        // The backbone's commands lead this concatenation, so its loop's `start`/`period`
        // still address the same rows. The lm_head tail is appended after and has no loop.
        loops: bb.loops,
        splitk_scratch_bytes: bb.splitk_scratch_bytes.max(lh.splitk_scratch_bytes),
        moe_scratch_bytes: bb.moe_scratch_bytes.max(lh.moe_scratch_bytes),
        roped_k_scratch_bytes: bb.roped_k_scratch_bytes.max(lh.roped_k_scratch_bytes),
        attn_unfused_scratch_bytes: bb
            .attn_unfused_scratch_bytes
            .max(lh.attn_unfused_scratch_bytes),
    })
}

/// A sample-rows step at this bake point, from its matmul's plain lowering `plain` (with its
/// small-M twin). The sampler reads one logits row per sequence, so on a prefill bucket the
/// lm_head's full-M GEMM is mostly waste (~31× at 1024 tokens: ~325 ms of Llama-3.2-3B's TTFT on
/// M4). The rows slice when [`slices`] holds: the gather, the matmul's qmv at the sampled rows and
/// the scatter run on a step with rows to drop and no speculative tokens (`OnlyIfNoSpec`); a
/// speculative verify reads every row, so there the plain matmul runs over them (`OnlyIfSpec`).
/// The two gates never both hold, so the all-rows matmul reads the gathered buffer as the norm
/// left it. Otherwise the matmul runs plain and the rest emits nothing. Every other step's
/// commands pass through.
fn sample_rows(
    p: &MetalModelConsts,
    step: &MetalStep,
    plain: Vec<GatedCommand>,
    bucket_m: u32,
    w: RowSources<'_>,
    profile: Option<&crate::targets::MetalTargetProfile>,
) -> Result<Vec<GatedCommand>, LoweringError> {
    use crate::tape::lowered::RuntimeGate::{OnlyIfNoSpec, OnlyIfSpec};
    use SampleRowsStep as R;
    let MetalStep::SampleRows(g, s) = *step else {
        return Ok(plain);
    };
    let sampled = |c| vec![GatedCommand::gated(c, OnlyIfNoSpec)];
    Ok(match (slices(&plain, bucket_m), s) {
        (false, R::Matmul) => plain,
        (false, R::Gather | R::Scatter | R::AllRows) => Vec::new(),
        (true, R::Gather) => sampled(gather_last_token_command(p, g.input, g.k.get())),
        (true, R::Matmul) => {
            let x = Some(crate::tape::lowered::MScaleAxis::X);
            let ix = w.of(WeightKind::Linear, 0)?;
            let codes = super::kernel_constants::AffineCodes::of(profile, g.bits.get());
            sampled(affine_qmv_command(p, &g, 1, x, g.layer, ix, codes))
        }
        (true, R::Scatter) => sampled(scatter_first_to_last_row_command(p, g.output, g.n.get())),
        (true, R::AllRows) => plain
            .into_iter()
            .map(|c| GatedCommand::gated(c.command, OnlyIfSpec))
            .collect(),
    })
}

/// Whether the sampled rows slice at this bake point: a multi-row bucket whose matmul lowers to
/// one qmm_t command, or the W4A8 pre-pass + GEMM pair — not qmv (a small bucket), not SplitK's
/// pair, not a small-M twin.
fn slices(plain: &[GatedCommand], bucket_m: u32) -> bool {
    let kernels: Vec<KernelId> = plain.iter().map(|c| c.command.kernel).collect();
    bucket_m > 1
        && matches!(
            kernels.as_slice(),
            [KernelId::AffineQmmT | KernelId::AffineQmmTNax]
                | [KernelId::AffineW4a8Quant, KernelId::AffineQmmW4a8]
        )
}

/// An MLX-affine matmul's matvec (qmv) command over `rows` rows, bound to `layer`'s weight `ix`.
/// The kernel is MLX's `dispatch_qmv` pick for the shape (quad for K∈{64,128}, fast when N%8 = 0 ∧
/// K%512 = 0, else generic); the grid `(rows, ⌈N/bn⌉, 1)` scales along X with the live token
/// count, or — `seq_axis` — is SET to the live sequence count: the lm_head slice's qmv reads just
/// the gathered sampled rows, one threadgroup row per sequence, instead of sweeping the
/// `⌈bucket_m/32⌉` qmm_t tiles.
fn affine_qmv_command(
    p: &MetalModelConsts,
    g: &AffineMatmul,
    rows: u32,
    seq_axis: Option<crate::tape::lowered::MScaleAxis>,
    layer: LayerId,
    ix: SourceIx,
    codes: super::kernel_constants::AffineCodes,
) -> LoweredCommand {
    let (n, k, bits) = (g.n.get(), g.k.get(), g.bits.get());
    let kernel = pick_qmv_kernel(n, k, bits);
    let (tg, tpg) = qmv_dispatch_shape(kernel, rows, n, /*B=*/ 1);
    let kernel_id = match kernel {
        QmvKernel::Quad { .. } => KernelId::AffineQmvQuad,
        QmvKernel::Fast => KernelId::AffineQmvFast,
        QmvKernel::Generic => KernelId::AffineQmv,
    };
    let (dtype, scale_dtype) = (dequant_dtype_for(p), scale_dtype_for(p));
    let (x, y) = (g.input.get(), g.output.get());
    LoweredCommand {
        kernel: kernel_id,
        library: "quantized_qmv",
        function: qmv_kernel_static_name(kernel, dtype, scale_dtype, bits, g.group_size.get()),
        constants: super::kernel_constants::AffineQmvConstants {
            k: super::ids::KDimI32(k as i32),
            n: super::ids::NDimI32(n as i32),
            codes,
        }
        .into_baked(),
        dispatch: DispatchShape {
            threadgroups: tg,
            threads_per_threadgroup: tpg,
            m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                seq_axis,
                axis: crate::tape::lowered::MScaleAxis::X,
                bucket_m: super::ids::BucketM(rows),
            }),
        },
        bindings: baked(affine_qmm_bindings(x, y, layer, ix)),
        gemm_dims: None,
    }
}

/// The sampled rows' gather and scatter kernels, each with its f16 and bf16 symbols.
const GATHER: (KernelId, [&str; 2]) = (
    KernelId::GatherLastToken,
    [
        "gather_last_token_f16_specialized",
        "gather_last_token_bf16_specialized",
    ],
);
const SCATTER: (KernelId, [&str; 2]) = (
    KernelId::ScatterFirstToLastRow,
    [
        "scatter_first_to_last_row_f16_specialized",
        "scatter_first_to_last_row_bf16_specialized",
    ],
);

/// The lm_head slice's input: each sequence's last row of `slot` moved to row `i`, in place.
pub(crate) fn gather_last_token_command(
    p: &MetalModelConsts,
    slot: Slot,
    hidden_size: u32,
) -> LoweredCommand {
    sample_slice_command(p, slot, hidden_size, GATHER)
}

/// The lm_head slice's output: row `i` of `slot` moved back to sequence `i`'s last row, in place.
pub(crate) fn scatter_first_to_last_row_command(
    p: &MetalModelConsts,
    slot: Slot,
    vocab_size: u32,
) -> LoweredCommand {
    sample_slice_command(p, slot, vocab_size, SCATTER)
}

fn sample_slice_command(
    p: &MetalModelConsts,
    slot: Slot,
    row_stride: u32,
    (kernel, [f16_symbol, bf16_symbol]): (KernelId, [&'static str; 2]),
) -> LoweredCommand {
    // One thread per column, walking the sequences in order: the rows
    // move in place, so one thread per row would race (see the shader).
    const THREADS_PER_TG: u32 = 256;
    LoweredCommand {
        kernel,
        library: "gather_last_token",
        function: pick_specialized_symbol(f16_symbol, bf16_symbol, p.metal_dtype),
        constants: super::kernel_constants::GatherLastTokenConstants {
            row_stride: super::ids::HiddenSize(row_stride),
        }
        .into_baked(),
        dispatch: DispatchShape {
            threadgroups: (row_stride.div_ceil(THREADS_PER_TG), 1, 1),
            threads_per_threadgroup: (THREADS_PER_TG, 1, 1),
            m_scaling: None,
        },
        bindings: baked(vec![
            Binding::ArenaSlot {
                slot: slot.get(),
                binding_index: 0,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::CuSeqlensQ,
                binding_index: 1,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::NumSeqsU32,
                binding_index: 2,
            },
        ]),
        gemm_dims: None,
    }
}

/// A projection bias on the KV writer's weight site, and the writer's rotary table (K's bias
/// was rotated with the key).
#[derive(Clone, Copy)]
struct TqBias {
    which: WeightTensor,
    layer: super::ids::LayerId,
    bias: SourceIx,
    rotary: CosSinTable,
}

impl TqBias {
    /// The bias itself, bound at `index`.
    fn bias_binding(self, index: u8) -> Binding {
        source(self.bias, self.which, self.layer, index)
    }
}

/// One TurboQuant'd cache operand and its writer's projection bias, if any.
/// `IS_K` fixes the store it lives in AND how that bias is restored: K's was
/// rotated with the key by the writer's rotary table (the same site's
/// cos/sin source), V's is cached as-is.
#[derive(Clone, Copy)]
struct TqOperand<const IS_K: bool>(Option<TqBias>);

/// The K and V a KV writer caches, as the TurboQuant codec must see them: it
/// quantizes each vector relative to its own norm, so an additive offset it
/// does not remove sets its error (`scratchy_ir::KvOffset`). Built only from the
/// writer's declared `KvOffsets`; every codec command takes one.
#[derive(Clone, Copy)]
struct TqOperands {
    k: TqOperand<true>,
    v: TqOperand<false>,
}

impl TqOperands {
    /// The operands a KV writer with `offsets` caches, each bias bound from the row's site `w`
    /// — refused where a K bias has no rotary table to rotate it by.
    fn of(
        p: &MetalModelConsts,
        offsets: KvOffsets,
        layer: LayerId,
        w: RowSources<'_>,
    ) -> Result<Self, LoweringError> {
        let bias = |b: Option<(BiasStorage, u32)>| {
            b.map(|(storage, slot)| {
                Ok(TqBias {
                    which: match storage {
                        BiasStorage::Dense => WeightTensor::Bias,
                        BiasStorage::Affine => WeightTensor::AffineLinearBias,
                    },
                    layer,
                    bias: w.of(WeightKind::Linear, slot)?,
                    rotary: w.cos_sin(p)?,
                })
            })
            .transpose()
        };
        let [k, v] = crate::op_abi::rope_append_bias_slots(offsets);
        let ops = Self {
            k: TqOperand(bias(k)?),
            v: TqOperand(bias(v)?),
        };
        if ops.k.0.is_some() && !p.rope_on_read {
            return Err(LoweringError::TurboQuantOffsetUnbound { index: w.index });
        }
        Ok(ops)
    }
}

/// TurboQuant prefill: the per-layer staging command — the layer's K (or V)
/// for every sequence of the step, written into the fp16 scratch in the
/// codebook's rotated domain (`tq_stage_rotated`), its offset restored.
/// `is_global` is the attention's layer class (its rope-on-read cos_sin for
/// span blocks and a rotated K bias; V is never roped). The grid (block-table
/// width, num_kv_heads, num_seqs) is overridden by the worker per forward; the
/// baked shape is a placeholder.
fn tq_stage_command<const IS_K: bool>(
    p: &MetalModelConsts,
    layer: u32,
    operand: TqOperand<IS_K>,
    is_global: bool,
    table: Option<SourceIx>,
    bucket_m: u32,
    block_cap: u32,
    pass: super::kernel_constants::TqStagePass,
    bits: TqBits,
) -> LoweredCommand {
    let (ror_rd, ror_po, ror_on, _) = if IS_K {
        rope_on_read_params(p, is_global)
    } else {
        (None, None, None, None)
    };
    let mut bindings = Vec::from(super::kernel_bindings::TqStageBindingSet {
        kv_layer: super::ids::LayerId(layer),
        is_v: !IS_K,
        rope_on_read: table.filter(|_| IS_K),
    });
    bindings.extend(operand.0.map(|b| b.bias_binding(10)));
    LoweredCommand {
        kernel: KernelId::TqStageRotated,
        library: "attention",
        function: pick_specialized_symbol(
            "tq_stage_rotated_f16",
            "tq_stage_rotated_bf16",
            p.metal_dtype,
        ),
        constants: super::kernel_constants::TqStageConstants {
            head_dim: super::ids::HeadDim(p.global_head_dim),
            num_kv_heads: super::ids::NumKvHeads(p.num_global_kv_heads),
            block_size: super::ids::BlockSize(p.global_block_size),
            max_blocks: super::ids::MaxBlocksPerSeq(block_cap),
            blocks_per_chunk: super::ids::BlocksPerChunk(crate::BLOCKS_PER_CHUNK),
            bits: super::ids::TqCodeBits(bits.get()),
            rot_dim: ror_rd,
            pair_off: ror_po,
            rope_on_read: ror_on,
            k_bias: IS_K && operand.0.is_some(),
            v_bias: !IS_K && operand.0.is_some(),
            pass,
        }
        .into_baked(),
        dispatch: DispatchShape {
            threadgroups: (1, p.num_global_kv_heads, bucket_m),
            threads_per_threadgroup: (32, 1, 1),
            m_scaling: None,
        },
        bindings: baked(bindings),
        gemm_dims: None,
    }
}

/// TurboQuant prefill: rotate the attention's q rows into the codebook domain
/// in place (`inverse` false, R·q), or its output rows back (`inverse`, Rᵀ·o).
fn tq_rotate_command(
    p: &MetalModelConsts,
    rows: Slot,
    inverse: bool,
    bucket_m: u32,
) -> LoweredCommand {
    let function = if inverse {
        pick_specialized_symbol(
            "tq_unrotate_rows_f16",
            "tq_unrotate_rows_bf16",
            p.metal_dtype,
        )
    } else {
        pick_specialized_symbol("tq_rotate_rows_f16", "tq_rotate_rows_bf16", p.metal_dtype)
    };
    LoweredCommand {
        kernel: KernelId::TqRotateRows,
        library: "attention",
        function,
        constants: super::kernel_constants::TqRotateRowsConstants {
            head_dim: super::ids::HeadDim(p.global_head_dim),
            num_q_heads: super::ids::NumQHeads(p.num_q_heads),
        }
        .into_baked(),
        dispatch: DispatchShape {
            threadgroups: (bucket_m, p.num_q_heads, 1),
            threads_per_threadgroup: (32, 1, 1),
            m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                seq_axis: None,
                axis: crate::tape::lowered::MScaleAxis::X,
                bucket_m: super::ids::BucketM(bucket_m),
            }),
        },
        bindings: super::kernel_bindings::TqRotateRowsBindingSet { rows }.into_baked(),
        gemm_dims: None,
    }
}

/// `Binding::Runtime`s for `kinds`, bound from index `first`.
fn runtime_at(first: u8, kinds: impl IntoIterator<Item = RuntimeBindingKind>) -> Vec<Binding> {
    (first..)
        .zip(kinds)
        .map(|(binding_index, kind)| Binding::Runtime {
            kind,
            binding_index,
        })
        .collect()
}

/// `Binding::Inline`s for `values`, bound from index `first`.
fn inline_at(first: u8, values: impl IntoIterator<Item = u32>) -> Vec<Binding> {
    (first..)
        .zip(values)
        .map(|(binding_index, value)| Binding::Inline {
            binding_index,
            value,
        })
        .collect()
}

/// TurboQuant: build the per-layer quantize command (new tokens in the fp16
/// scratch → packed[L]), run right after the layer's KV writer, removing the
/// operand's offset first (bindings 18..=23: bias, rotary table, positions,
/// then `offset_mode` — 0 none, 1 the bias, 2 the bias rotated to the key's
/// position — `rot_dim`, `pair_off`). New-token count == num_tokens, so the grid
/// scales like RopeAppend (m_scaling on X). The TurboQuant layers are the
/// GLOBAL class (every layer on uniform arches, where GLOBAL_* == base).
fn tq_quantize_command<const IS_K: bool>(
    p: &MetalModelConsts,
    layer: u32,
    operand: TqOperand<IS_K>,
    bucket_m: u32,
    bits: TqBits,
) -> LoweredCommand {
    use RuntimeBindingKind as RB;
    let (hd, nkv, bs) = (
        p.global_head_dim,
        p.num_global_kv_heads,
        p.global_block_size,
    );
    let bits = bits.get();
    let vpw = 32 / bits;
    let scale = (1.0f32 / (hd as f32).sqrt()).to_bits();
    let li = super::ids::LayerId(layer);
    let (cache, packed, norms) = if IS_K {
        (
            RB::KvCacheK { layer: li },
            RB::TqPackedK { layer: li },
            RB::TqNormsK { layer: li },
        )
    } else {
        (
            RB::KvCacheV { layer: li },
            RB::TqPackedV { layer: li },
            RB::TqNormsV { layer: li },
        )
    };
    let slots = RB::SlotMapping { layer: li };
    let runtime = [cache, slots, RB::TqSigns, RB::TqBoundaries, RB::TqCentroids];
    let mut bindings = runtime_at(0, runtime.into_iter().chain([packed, norms]));
    let codec = [hd, bits, vpw, hd.div_ceil(vpw), 1 << bits, scale, nkv, bs];
    bindings.extend(inline_at(
        7,
        codec.into_iter().chain([crate::BLOCKS_PER_CHUNK]),
    ));
    bindings.extend(runtime_at(16, [slots]));
    // do_writeback 0: the quantize runs before attention, which reads the
    // step's new keys raw (decode) or rotates them itself (prefill staging).
    bindings.extend(inline_at(17, [0]));
    if let Some(b) = operand.0 {
        bindings.push(b.bias_binding(18));
        if IS_K {
            bindings.push(b.rotary.binding(b.layer, 19));
            bindings.extend(runtime_at(20, [RB::Positions]));
        }
    }
    let (rot_dim, pair_off, ..) = rope_on_read_params(p, true);
    let mode = operand.0.map_or(0, |_| 1 + u32::from(IS_K));
    let pairing = [
        rot_dim.map_or(0, |r| r.get()),
        pair_off.map_or(0, |po| po.get()),
    ];
    bindings.extend(inline_at(21, [mode, pairing[0], pairing[1]]));
    LoweredCommand {
        kernel: KernelId::TqQuantizeToPacked,
        library: "turboquant",
        function: pick_specialized_symbol(
            "tq_compress_paged",
            "tq_compress_paged_bf16",
            p.metal_dtype,
        ),
        constants: baked(vec![]),
        dispatch: DispatchShape {
            threadgroups: (bucket_m, nkv, 1),
            threads_per_threadgroup: (hd, 1, 1),
            m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                seq_axis: None,
                axis: crate::tape::lowered::MScaleAxis::X,
                bucket_m: super::ids::BucketM(bucket_m),
            }),
        },
        bindings: baked(bindings),
        gemm_dims: None,
    }
}

/// TurboQuant decode: the `AttentionViaCacheTq` twin of a decode
/// `AttentionViaCache` command — the same kernel, geometry and bindings plus
/// `ATTN_TQ_BITS` and the packed-store bindings, so it reads every key but the
/// one this step appended straight from the packed store, restoring each
/// operand's offset (a rotated K bias by the rope-on-read table and pairing).
/// One query head per threadgroup until [`serve_tq_decode_heads`] sets the
/// device's count at load.
fn tq_attention_command(
    attn: &LoweredCommand,
    layer: u32,
    ops: TqOperands,
    bits: TqBits,
) -> LoweredCommand {
    let mut constants = attn.constants.to_vec();
    constants.extend(Vec::from(
        super::kernel_constants::AttentionViaCacheTqConstants {
            bits: super::ids::TqCodeBits(bits.get()),
            k_bias: ops.k.0.is_some(),
            v_bias: ops.v.0.is_some(),
            heads: super::ids::TqDecodeHeads(1),
        },
    ));
    let mut bindings = attn.bindings.to_vec();
    bindings.extend(Vec::from(super::kernel_bindings::TqAttentionBindingSet {
        kv_layer: super::ids::LayerId(layer),
    }));
    bindings.extend(ops.k.0.map(|b| b.bias_binding(14)));
    bindings.extend(ops.v.0.map(|b| b.bias_binding(15)));
    LoweredCommand {
        kernel: KernelId::AttentionViaCacheTq,
        constants: baked(constants),
        bindings: baked(bindings),
        ..*attn
    }
}

/// The model's TurboQuant code width: a codec step lowers only on a model built with the codec.
fn tq_bits(p: &MetalModelConsts) -> Result<TqBits, LoweringError> {
    match p.kv_codec {
        KvCodec::TurboQuant(bits) => Ok(bits),
        KvCodec::Dense => Err(LoweringError::CodecStepOnDenseModel),
    }
}

/// Have a TurboQuant decode command serve `heads` query heads per
/// threadgroup: the pool's pick for the device it loads on, since it turns on
/// the GPU's core count and every tape is baked for a whole chip generation.
/// Every other command is left as it is.
pub(crate) fn serve_tq_decode_heads(
    command: &mut LoweredCommand,
    heads: super::ids::TqDecodeHeads,
) {
    if command.kernel != KernelId::AttentionViaCacheTq {
        return;
    }
    let slot = super::kernel_constants::AttentionViaCacheTqConstants::HEADS;
    let constants = command.constants.iter().map(|c| {
        if c.index == slot.get() {
            ConstantValue::uint(slot, heads.get())
        } else {
            *c
        }
    });
    command.constants = baked(constants.collect());
    command.dispatch.threadgroups.1 /= heads.get();
}

/// A command that computes batch row 0 only ([`SeqScope::RowZero`]: the
/// rope-once pair, whose scratch holds one sequence's keys, and the hd512
/// unfused attention) runs on single-sequence steps only. The step's per-row
/// paged attentions serve steps with several sequences: one reading the
/// cache's roped K as is takes the steps without unrotated span blocks, and
/// one binding cos_sin, re-roping those blocks as it reads, takes the rest (or
/// every such step, when it is alone). A step with a row-zero command and no
/// re-roping per-row twin is refused. Every other command is left as it is.
fn route_by_sequence_count(
    index: usize,
    cmds: Vec<GatedCommand>,
) -> Result<Vec<GatedCommand>, LoweringError> {
    use crate::tape::lowered::RuntimeGate::{
        self, OnlyIfOneSequence, OnlyIfUnrotatedBlocks, UnlessOneSequence, UnlessUnrotatedBlocks,
    };
    use crate::tape::lowered::SeqScope::{AllRows, RowZero};
    let Some(row_zero) = cmds.iter().find(|c| c.command.seq_scope() == RowZero) else {
        return Ok(cmds);
    };
    let per_row = |c: &LoweredCommand| {
        c.kernel == KernelId::AttentionPrefillSdpaPaged && c.seq_scope() == AllRows
    };
    // A paged attention's only model source is its rotary table (cos_sin).
    let reropes = |c: &LoweredCommand| {
        c.bindings
            .iter()
            .any(|b| matches!(b, Binding::Source { .. }))
    };
    let twins = || cmds.iter().map(|c| &c.command).filter(|c| per_row(c));
    if !twins().any(reropes) {
        return Err(LoweringError::RowZeroWithoutPerRowTwin {
            index,
            kernel: row_zero.command.kernel,
        });
    }
    let plain = twins().any(|c| !reropes(c));
    Ok(cmds
        .into_iter()
        .map(|c| {
            let only: &[RuntimeGate] = match c.command.seq_scope() {
                RowZero => &[OnlyIfOneSequence],
                AllRows if !per_row(&c.command) => return c,
                AllRows if !reropes(&c.command) => &[UnlessOneSequence, UnlessUnrotatedBlocks],
                AllRows if plain => &[UnlessOneSequence, OnlyIfUnrotatedBlocks],
                AllRows => &[UnlessOneSequence],
            };
            GatedCommand::gated(c.command, RuntimeGate::and(c.gate, only))
        })
        .collect())
}

/// On a NAX device, the small-M matrix-unit twin of an MLX-affine 4-bit
/// `AffineQmm` in a bucket that can see a `SMALL_M_TOKENS` step: the twin runs
/// on those steps (`OnlyIfSmallMTokens`) and the instruction's own GEMM on
/// every other (`UnlessSmallMTokens`).
fn route_small_m(
    p: &MetalModelConsts,
    step: &MetalStep,
    cmds: Vec<GatedCommand>,
    bucket_m: u32,
    w: RowSources<'_>,
    profile: Option<&crate::targets::MetalTargetProfile>,
) -> Result<Vec<GatedCommand>, LoweringError> {
    use crate::tape::lowered::RuntimeGate::{OnlyIfSmallMTokens, UnlessSmallMTokens};
    // A sample-rows step lowered its matmul, twin included.
    let (MetalStep::AffineQmm(g) | MetalStep::SampleRows(g, _)) = *step else {
        return Ok(cmds);
    };
    let AffineMatmul {
        input: Slot(in_slot),
        output: Slot(out_slot),
        layer,
        n: NDim(n),
        k: KDim(k),
        group_size: AffineGroupSize(group_size),
        bits: AffineBits(4),
        ..
    } = g
    else {
        return Ok(cmds);
    };
    // The small-M kernel reads the codes as stored, signed (offset-8).
    let codes = super::kernel_constants::AffineCodes::of(profile, 4);
    let tile = match SmallMTile::for_bucket(bucket_m) {
        Some(tile)
            if codes == super::kernel_constants::AffineCodes::Offset8
                && matches!(group_size, 32 | 64 | 128)
                && n.is_multiple_of(SMALL_M_TILE_COLS) =>
        {
            tile
        }
        _ => return Ok(cmds),
    };
    let small_m = LoweredCommand {
        kernel: KernelId::AffineQmmSmallM,
        library: "quantized_qmm_nax",
        function: small_m_kernel_static_name(
            dequant_dtype_for(p),
            scale_dtype_for(p),
            group_size,
            tile,
        ),
        constants: super::kernel_constants::AffineQmmTConstants {
            k: super::ids::KDimI32(k as i32),
            n: super::ids::NDimI32(n as i32),
            m: super::ids::MDimI32(bucket_m as i32),
            codes,
        }
        .into_baked(),
        dispatch: DispatchShape {
            threadgroups: (n / SMALL_M_TILE_COLS, bucket_m.div_ceil(tile.rows()), 1),
            threads_per_threadgroup: (32 * crate::quantized::SMALL_M_SIMDGROUPS, 1, 1),
            m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                seq_axis: None,
                axis: crate::tape::lowered::MScaleAxis::Y,
                bucket_m: super::ids::BucketM(bucket_m),
            }),
        },
        bindings: baked(affine_qmm_bindings(
            in_slot,
            out_slot,
            layer,
            w.of(WeightKind::Linear, 0)?,
        )),
        gemm_dims: None,
    };
    Ok(cmds
        .into_iter()
        .map(|c| GatedCommand::gated(c.command, UnlessSmallMTokens))
        .chain([GatedCommand::gated(small_m, OnlyIfSmallMTokens)])
        .collect())
}

/// A gated row's `gate` on every command of it. A command its realization already gated cannot
/// take a second: [`LoweringError::DoubleGate`].
fn row_gate(
    cmds: Vec<GatedCommand>,
    gate: Option<crate::tape::lowered::RuntimeGate>,
    index: usize,
) -> Result<Vec<GatedCommand>, LoweringError> {
    let Some(gate) = gate else { return Ok(cmds) };
    let gated = |c: GatedCommand| match c.gate {
        None => Ok(GatedCommand::gated(c.command, gate)),
        Some(_) => Err(LoweringError::DoubleGate { index }),
    };
    cmds.into_iter().map(gated).collect()
}

/// A loop span the walk has entered but not yet left.
struct OpenSpan {
    /// First command of the body, in the emitted command stream.
    cmd_start: usize,
    iters: u32,
    layer_stride: u32,
    /// The body's rows in the STEP stream, for the shape-state replay.
    body: std::ops::Range<usize>,
}

/// Close every span that ends at `at`, recording its `TapeLoop`.
///
/// ⛔ THE SHAPE STATE STILL ADVANCES FOR EVERY ITERATION. The commands are emitted once, but
/// `update_shape_state` threads the activation width along the tape — so the rows AFTER a loop
/// must see the width all of its iterations produce, not one. The body was already walked once
/// (that walk is what emitted it), so the replay is the remaining `iters - 1`.
fn close_spans(
    open: &mut Vec<OpenSpan>,
    at: usize,
    loops: &mut Vec<super::lowered::TapeLoop>,
    commands: &[GatedCommand],
    rows: &[StepRow],
    cur_width: &mut crate::tape::lowered::ActivationWidth,
    m_divisor: &mut u32,
) {
    while open.last().is_some_and(|s| s.body.end == at) {
        let s = open.pop().expect("checked by the guard");
        loops.push(super::lowered::TapeLoop {
            start: s.cmd_start as u32,
            period: (commands.len() - s.cmd_start) as u32,
            iters: s.iters,
            layer_stride: s.layer_stride,
        });
        advance_shape(
            &rows[s.body.clone()],
            s.iters.saturating_sub(1),
            cur_width,
            m_divisor,
        );
    }
}

/// Replay `update_shape_state` over `rows` `times` times, expanding any nested loops.
fn advance_shape(
    rows: &[StepRow],
    times: u32,
    cur_width: &mut crate::tape::lowered::ActivationWidth,
    m_divisor: &mut u32,
) {
    for _ in 0..times {
        let mut k = 0usize;
        while k < rows.len() {
            match &rows[k] {
                StepRow::Loop { iters, body, .. } => {
                    let body = k + 1..k + 1 + body.get() as usize;
                    advance_shape(&rows[body.clone()], iters.get(), cur_width, m_divisor);
                    k = body.end;
                }
                StepRow::Step(step, _) => {
                    update_shape_state(step, cur_width, m_divisor);
                    k += 1;
                }
            }
        }
    }
}

/// Lower one half's step rows.
///
/// `at.bucket_m` is the bucket point this tape is specialized for —
/// the worker's `forward()` only routes batches with `num_tokens`
/// matching this bucket's `[m_min, m_max_excl)` range here.
/// Dispatch shapes that depend on `M` (token-parallel kernels, GEMM
/// outer dim) are computed against `bucket_m`.
///
/// `at.num_arena_slots` is the colored slot count; the lowered tape
/// carries it through verbatim — the worker uses it to size its
/// per-shape-class arena. `sources` is each row's weight site.
fn lower(
    p: &MetalModelConsts,
    rows: &[StepRow],
    barriers_in: &[bool],
    sources: &[Vec<RowSource>],
    at: &BakePoint<'_>,
) -> Result<LoweredMetalTape, LoweringError> {
    let BakePoint {
        chunked,
        bucket_m,
        num_arena_slots,
        block_cap,
        profile,
        ..
    } = *at;
    let mut commands: Vec<GatedCommand> = Vec::with_capacity(rows.len());
    let mut barrier_before: Vec<bool> = Vec::with_capacity(rows.len());
    // The layer loop, if this tape has one. Set by the `StepRow::Loop` arm below,
    // which records the body instead of unrolling it.
    let mut loops: Vec<super::lowered::TapeLoop> = Vec::new();
    // Loop spans still open at the current walk position, innermost last.
    let mut open_spans: Vec<OpenSpan> = Vec::new();
    let mut splitk_scratch_bytes: u32 = 0;
    let mut moe_scratch_bytes: u32 = 0;
    let mut roped_k_scratch_bytes: u32 = 0;
    let mut attn_unfused_scratch_bytes: u32 = 0;
    // Shape state for the dimensionless vision arms (`Gelu`,
    // `MeanSubRmsNormBiasAdd`, vision `Add`) and the merger's row
    // reduction. `cur_width` tracks the activation row-width (set by
    // each `Gemm`'s output-N and the merger `Reshape`'s static col dim);
    // `m_divisor` tracks the row-count divisor in force (1 normally,
    // `vision_merge_factor` after the merger reshape). Updated by
    // `update_shape_state` after each step is lowered, so each
    // arm sees the state in force AT its position. Text arches never set
    // a non-1 divisor (their reshapes carry `rows_div = 1`), so this is
    // inert outside vision towers.
    // Activation row-width in force at each step (see `lower_one`).
    // Seeded to the residual-stream width (HIDDEN_SIZE) so the embed-time
    // scalar multiply (granite `* embedding_multiplier`), which precedes
    // the first Gemm that would publish a width, broadcasts over the full
    // residual row rather than a stale 0. Typed as `ActivationWidth` so an
    // elementwise broadcast can never be handed head geometry (Q_SIZE).
    let mut cur_width = crate::tape::lowered::ActivationWidth::residual_stream(p);
    let mut m_divisor: u32 = 1;
    let mut i = 0usize;
    // One barrier flag per row, rolled: the loop body's flags serve every
    // iteration (body equivalence is what let the loop roll, so each
    // iteration has the same hazard signature). A metadata-only row
    // (Reshape) emits no command, so its flag is unused. A step that
    // lowers to several commands (the SplitK matmul pair) gives its flag
    // to the first; the rest get `true` (intra-step scratch RAW).
    let flag_for = |idx: usize| -> bool { barriers_in.get(idx).copied().unwrap_or(true) };

    while i < rows.len() {
        close_spans(
            &mut open_spans,
            i,
            &mut loops,
            &commands,
            rows,
            &mut cur_width,
            &mut m_divisor,
        );
        match &rows[i] {
            StepRow::Loop {
                iters,
                body,
                stride,
            } => {
                // ⭐ RECORD THE SPAN, LOWER THE BODY ONCE, WALK ON. Every row is lowered exactly
                // once no matter how deep it sits, so nesting needs a STACK of open spans rather
                // than a special case: gemma-4 rolls an outer six-layer `SSSSSG` cell with an
                // inner loop over the five sliding layers inside it.
                //
                // `layer_offset` is consumed in exactly one pattern — `LayerId(*layer +
                // layer_offset)` — and a weight's source family is identical across
                // iterations (the roll proof checks it). So iteration `i` IS this body with every
                // `LayerId` advanced by `i * layer_stride`, which `Binding::bump_layer` applies
                // at load. Materialising the copies here is what made the baked tape 14.3M lines
                // of const literals and pinned rustc's front end for ~46s on one file.
                let body_start = i + 1;
                let body_end = body_start
                    .checked_add(body.get() as usize)
                    .filter(|end| *end <= rows.len())
                    .ok_or_else(|| LoweringError::MalformedLoop {
                        index: i,
                        count: iters.get(),
                        body_len: body.get(),
                        remaining: rows.len().saturating_sub(body_start),
                    })?;
                open_spans.push(OpenSpan {
                    cmd_start: commands.len(),
                    iters: iters.get(),
                    layer_stride: stride.get(),
                    body: body_start..body_end,
                });
                i = body_start;
            }
            StepRow::Step(step, gate) => {
                let w = RowSources::of_row(sources, i, at);
                let own = lower_one(
                    p,
                    chunked,
                    step,
                    w,
                    bucket_m,
                    0,
                    &mut splitk_scratch_bytes,
                    &mut moe_scratch_bytes,
                    &mut roped_k_scratch_bytes,
                    &mut attn_unfused_scratch_bytes,
                    block_cap,
                    profile,
                    cur_width,
                    m_divisor,
                )?;
                let own = own.into_iter().map(GatedCommand::ungated).collect();
                let cmds = route_small_m(p, step, own, bucket_m, w, profile)?;
                let cmds = sample_rows(p, step, cmds, bucket_m, w, profile)?;
                let cmds = row_gate(cmds, *gate, i)?;
                let cmds = route_by_sequence_count(i, cmds)?;
                update_shape_state(step, &mut cur_width, &mut m_divisor);
                let n_cmds = cmds.len();
                // The hazard flags track arena slots, not the shared scratch: a command that
                // writes it (a W4A8 pre-pass, a split-K partial) must wait for the previous reader.
                let writes_scratch = cmds.first().is_some_and(|c| {
                    c.command
                        .bindings
                        .iter()
                        .any(|b| matches!(b, Binding::Scratch { .. }))
                });
                commands.extend(cmds);
                if n_cmds >= 1 {
                    barrier_before.push(flag_for(i) || writes_scratch);
                    barrier_before.extend(std::iter::repeat_n(true, n_cmds - 1));
                }
                i += 1;
            }
        }
    }

    close_spans(
        &mut open_spans,
        rows.len(),
        &mut loops,
        &commands,
        rows,
        &mut cur_width,
        &mut m_divisor,
    );
    // ⛔ OUTERMOST FIRST. Spans close innermost-first, but the expander resolves a nest by
    // taking the first loop that starts at a position and recursing into the REST of the list —
    // so an inner loop ahead of its parent would be read as the parent.
    loops.sort_by_key(|l| (l.start, std::cmp::Reverse(l.period)));
    debug_assert_eq!(commands.len(), barrier_before.len());
    Ok(LoweredMetalTape {
        bucket_m,
        num_arena_slots,
        commands: baked(commands),
        barrier_before: baked(barrier_before),
        loops: baked(loops),
        splitk_scratch_bytes,
        moe_scratch_bytes,
        roped_k_scratch_bytes,
        attn_unfused_scratch_bytes,
    })
}

/// Abstract-interpret the activation shape across the step
/// stream so the dimensionless vision arms can recover the width / row
/// count the macro already solved — the info is read from the steps,
/// not re-derived.
///
/// - `cur_width` (row-width): set by each `Gemm`'s output-N and by the
///   merger `Reshape`'s static (num_tokens-independent) col dim. The
///   `MeanSubRmsNormBiasAdd` (LayerNorm HIDDEN) and `Gelu` (cols) arms
///   read it; the vision `Add` arm reads it as the residual width.
/// - `m_divisor` (row divisor): the merger `Reshape`
///   `[num_tokens / vision_merge_factor, vision_merge_hidden]` carries
///   `rows_div = vision_merge_factor`; every op after it
///   operates on `bucket_m / vision_merge_factor` rows. Text reshapes
///   carry `rows_div = 1`, so this never fires outside vision towers.
fn update_shape_state(
    step: &MetalStep,
    cur_width: &mut crate::tape::lowered::ActivationWidth,
    m_divisor: &mut u32,
) {
    use crate::tape::lowered::ActivationWidth;
    use MetalStep as I;
    match step {
        // A dense GEMM publishes a `[*, n]` activation.
        I::Gemm(_, _, _, NDim(n), _) => {
            *cur_width = ActivationWidth::from_gemm_n(*n);
        }
        // Quantized matmuls publish `[*, n]` exactly like dense GEMMs.
        // Without this arm, `cur_width` after an AffineQmm lm_head is
        // whatever the last Gemm/Reshape left (Gemma4: the global-attn
        // head reshape's 8192) — the trailing `TanhSoftCap` then caps
        // only the first `eff_m * 8192` logits, and a PARTIAL cap is
        // not argmax-invariant (uncapped raw values past the boundary
        // outrun capped ones; flipped Gemma4 greedy decode at margins
        // up to ~2).
        I::AffineQmm(AffineMatmul { n: NDim(n), .. })
        | I::SampleRows(
            AffineMatmul { n: NDim(n), .. },
            SampleRowsStep::Matmul | SampleRowsStep::AllRows,
        ) => {
            *cur_width = ActivationWidth::from_gemm_n(*n);
        }
        // The merger reshape `[num_tokens / div, width]`: the
        // num_tokens-scaling row axis carries the row divisor; the
        // static column axis is the new width.
        // The divisor is SET unconditionally (including back to 1):
        // Qwen2.5-VL reshapes `[L/S², S²·E]` for the window gather and
        // then BACK to `[L, E]` before the encoder blocks — a sticky
        // divisor would shrink every downstream op's `eff_m` by S².
        // (Qwen3.5-VL's single trailing merger reshape never exposed
        // this.)
        I::Reshape(_, _, _, RowsDivisor(rows_div), HiddenSize(cols)) => {
            *m_divisor = (*rows_div).max(1);
            *cur_width = ActivationWidth::from_reshape_width(*cols);
        }
        _ => {}
    }
}

/// Spans rope-on-read: per-attention-arm rotary params for the kernel
/// fn-consts (rot_dim/pair_off/rope_on_read) + the BindingSet's
/// `rope_on_read` flag. Returns all-`None` when `p.rope_on_read` is
/// false (non-spans → byte-identical lowering). `is_global` selects the
/// layer class: GLOBAL arms (`AttentionViaCache`/`AttentionPrefillPaged`)
/// use `GLOBAL_ROT_DIM` + proportional pairing; SLIDING arms use the
/// base `ROT_DIM` with full-NeoX `rot_dim/2` pairing. MUST match how
/// `rope_append` rotated that layer (lowering.rs ~2339).
#[allow(clippy::type_complexity)]
fn rope_on_read_params(
    p: &MetalModelConsts,
    is_global: bool,
) -> (
    Option<super::ids::RotDim>,
    Option<super::ids::RopePairOff>,
    Option<u32>,
    Option<bool>,
) {
    if !p.rope_on_read {
        return (None, None, None, None);
    }
    let (rd, hd) = if is_global {
        (p.global_rot_dim, p.global_head_dim)
    } else {
        (p.rot_dim, p.head_dim)
    };
    let pair_off = if is_global && p.rope_proportional {
        hd / 2
    } else {
        rd / 2
    };
    (
        Some(super::ids::RotDim(rd)),
        Some(super::ids::RopePairOff(pair_off)),
        Some(1),
        Some(is_global),
    )
}

/// The sdpa-paged prefill attention: one threadgroup per (q head, query
/// token). Each query token finds its sequence in `cu_seqlens_q` and reads K
/// through that sequence's own block-table row, so it is right for any number
/// of sequences; it leaves grid Z alone (see `MScaling::seq_axis`).
fn sdpa_paged_command(
    p: &MetalModelConsts,
    constants: super::kernel_constants::AttentionPrefillPagedConstants,
    bindings: super::kernel_bindings::AttentionPrefillPagedBindingSet,
    bucket_m: u32,
) -> LoweredCommand {
    use crate::tape::kernel_identity::{AttentionSdpaPagedBf16, AttentionSdpaPagedF16};
    let dispatch = DispatchShape {
        threadgroups: (p.num_q_heads, bucket_m, 1),
        threads_per_threadgroup: (1024, 1, 1),
        m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
            seq_axis: None,
            axis: crate::tape::lowered::MScaleAxis::Y,
            bucket_m: super::ids::BucketM(bucket_m),
        }),
    };
    match p.metal_dtype {
        crate::tape::lowered::MetalDtype::Bf16 => {
            LoweredCommand::for_kernel::<AttentionSdpaPagedBf16>(constants, bindings, dispatch)
        }
        _ => LoweredCommand::for_kernel::<AttentionSdpaPagedF16>(constants, bindings, dispatch),
    }
}

/// The steel/NAX paged prefill attention's grid: one threadgroup per
/// (BQ-block of queries, q head), and one Z-layer per sequence
/// (`tid.z = seq_idx`) so a BQ-block never straddles a sequence boundary.
fn steel_paged_dispatch(p: &MetalModelConsts, bucket_m: u32, bq: u32) -> DispatchShape {
    DispatchShape {
        threadgroups: (bucket_m.div_ceil(bq), p.num_q_heads, 1),
        threads_per_threadgroup: (128, 1, 1),
        m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
            seq_axis: Some(crate::interpreter::metal::lowered::MScaleAxis::Z),
            axis: crate::tape::lowered::MScaleAxis::X,
            bucket_m: super::ids::BucketM(bucket_m),
        }),
    }
}

/// Co-resident NeoX-pair lane layout for the decode rope-on-read path
/// (`AttentionViaCacheConstants::pair_coresident`, shader slot 12). Returns
/// `Some(1)` when rope-on-read is active AND the NeoX pairing offset is
/// `head_dim/2` so each lane's `{d, d+head_dim/2}` pair is co-resident —
/// making the on-read rope in-lane (no `simd_shuffle`). Admits BOTH full
/// NeoX (`rot_dim == head_dim`, `pair_off == head_dim/2`) AND proportional
/// rope (`rot_dim < head_dim`, `pair_off == head_dim/2`, gemma4 global
/// hd512/rot128): for proportional the co-resident layout already pairs
/// element `d` with `d + head_dim/2`, and the in-lane kernel skips first-side
/// indices outside the `rot_dim/2` rotary window. The single requirement is
/// `pair_off == head_dim/2` (so the lane's co-resident `+head_dim/2` partner
/// IS the rope pair). `SPANS_CORESIDENT=0` forces `None` (the old contiguous
/// shuffle path) for same-binary A/B. `None` whenever rope-on-read is off,
/// which is byte-identical to non-spans.
fn pair_coresident_param(
    _rot_dim: Option<super::ids::RotDim>,
    pair_off: Option<super::ids::RopePairOff>,
    rope_on_read: Option<u32>,
    head_dim: u32,
) -> Option<u32> {
    if rope_on_read != Some(1) {
        return None;
    }
    // The co-resident lane holds `{d, d+head_dim/2}`; that is the rope pair
    // iff `pair_off == head_dim/2` (true for full NeoX AND gemma4's
    // proportional rope, where pair_off is set to head_dim/2 in
    // `rope_on_read_params`). Any other pairing falls back to the shuffle path.
    if pair_off.map(|p| p.get()) != Some(head_dim / 2) {
        return None;
    }
    // The co-resident lane layout splits its `qk_per_thread =
    // head_dim / 32` elements into two EQUAL halves (`np =
    // qk_per_thread / 2` in `attn_elem_off`), one either side of
    // `head_dim/2`. That integer division truncates when
    // `qk_per_thread` is ODD, so a lane would claim `np` elements
    // from the first half and `qk_per_thread - np` from the second —
    // the union across lanes then reads some elements twice and
    // others never. head_dim 96 (phi-3) is the first shape to hit
    // it: 96/32 = 3. Every head_dim that works today is a multiple
    // of 64 (64/128/256/512 → 2/4/8/16), so this excludes nothing
    // that currently runs.
    if !(head_dim / 32).is_multiple_of(2) {
        return None;
    }
    if std::env::var("SPANS_CORESIDENT").as_deref() == Ok("0") {
        return None;
    }
    Some(1)
}

/// Lower one step to zero, one, or many `LoweredCommand`s. Most steps
/// produce exactly one; the metadata-only `Reshape` produces zero;
/// `MetalStep::AffineQmm` in the matmul branch produces two when the
/// dispatcher picks `QmmTKernel::SplitK` (the `affine_qmm_t_splitk`
/// kernel writes a `[split_k, M, N]` partial into a shared scratch
/// buffer, then `splitk_reduce_sum` reduces it into the AffineQmm's
/// arena slot).
///
/// `layer_offset` is added to each step's own layer at weight-binding
/// time (the cuda interpreter's `let layer = ctx.layer_offset + layer;`).
/// A rolled loop's body is lowered once at offset 0; its later
/// iterations advance the layer at load (`Binding::bump_layer`).
///
/// `splitk_scratch_bytes` is the running max of `split_k * M * N *
/// elem_size_bytes(dtype)` across every `AffineQmm` in this tape that
/// picked `SplitK`. The worker uses it to size the shared scratch
/// buffer that `Binding::Scratch` resolves against.
fn lower_one(
    p: &MetalModelConsts,
    chunked: bool,
    inst: &MetalStep,
    w: RowSources<'_>,
    bucket_m: u32,
    layer_offset: u32,
    splitk_scratch_bytes: &mut u32,
    moe_scratch_bytes: &mut u32,
    roped_k_scratch_bytes: &mut u32,
    attn_unfused_scratch_bytes: &mut u32,
    // Runtime per-sequence block-table capacity (`KvCachePool::max_blocks_per_seq`).
    // Replaces `p.max_blocks_per_seq` at the `MaxBlocksPerSeq` function-constant
    // sites (block-table row stride + scratch bound) and the rope-once
    // `roped_k_scratch` `num_pages` so long context isn't truncated.
    block_cap: u32,
    profile: Option<&crate::targets::MetalTargetProfile>,
    // Shape state threaded by `lower()` (see [`ShapeState`]). `cur_width`
    // is the current activation row-width (← each `Gemm`'s output-N, or
    // the merger `Reshape`'s static col dim) — the dimensionless vision
    // `Gelu` / LayerNorm arms read it. `m_divisor` is the row-count
    // divisor in force (1 normally; `vision_merge_factor` after the
    // merger reshape) so post-merge ops dispatch over `bucket_m /
    // m_divisor` rows.
    cur_width: crate::tape::lowered::ActivationWidth,
    m_divisor: u32,
) -> Result<Vec<LoweredCommand>, LoweringError> {
    // `eff_m` = the live row count this op operates on at the bucket
    // level. For everything before the merger reshape this is `bucket_m`;
    // for the merger's `[num_tokens / vision_merge_factor, ...]` ops it
    // shrinks by the merge factor. m_scaling's `bucket_m` field stays the
    // FULL bucket so the runtime `num_tokens` rescale is relative to the
    // whole bucket (correct for both partial-bucket prefill and the
    // num_tokens == bucket_m vision golden path).
    let eff_m = bucket_m / m_divisor.max(1);
    // The variable is threaded so the MoE lowering pass can grow the
    // bucket's scratch footprint as it stamps Binding::MoeScratch
    // offsets; the read keeps it from reading as unused here.
    let _ = &moe_scratch_bytes;
    use MetalStep as I;

    let cmd = match inst {
        // ── Token embedding ────────────────────────────────────────
        I::Embed(Slot(out_slot)) => LoweredCommand {
            kernel: KernelId::Embed,
            library: "embed",
            function: pick_specialized_symbol(
                "embed_f16_specialized",
                "embed_bf16_specialized",
                p.metal_dtype,
            ),
            constants: super::kernel_constants::EmbedConstants {
                bucket_m: super::ids::BucketM(bucket_m),
                // Embedding-table row stride = residual-stream width =
                // HIDDEN_SIZE (NOT Q_SIZE — they differ when head_dim !=
                // hidden/num_heads, e.g. Qwen3.5 head_dim=256).
                q_size: super::ids::QSize(p.hidden_size as u32),
            }
            .into_baked(),
            // 1D dispatch over the `bucket_m` tokens; one thread per
            // token gathers a row from `embed_tokens.weight`. Scales
            // proportionally with actual M at dispatch time.
            dispatch: {
                let mut d = DispatchShape::dispatch_1d(bucket_m, THREADS_PER_GROUP);
                d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::X,
                    bucket_m: super::ids::BucketM(bucket_m),
                });
                d
            },
            bindings: baked(vec![
                // out: arena[out_slot]
                Binding::ArenaSlot {
                    slot: *out_slot,
                    binding_index: 0,
                },
                // weight: embed_tokens.weight (layer 0; Embed is not layered)
                source(
                    w.of(WeightKind::Embedding, 0)?,
                    WeightTensor::Weight,
                    super::ids::LayerId(0),
                    1,
                ),
                // input_ids
                Binding::Runtime {
                    kind: RuntimeBindingKind::InputIds,
                    binding_index: 2,
                },
            ]),
            gemm_dims: None,
        },

        // ── Standalone RMSNorm ─────────────────────────────────────
        // The step is `RmsNorm(in_slot, out_slot, ...)`. An earlier
        // `(out_slot, in_slot, ...)`
        // pattern here silently swapped the names — the kernel read
        // from a fresh slot and overwrote the upstream tile.
        I::RmsNorm(
            Slot(in_slot),
            Slot(out_slot),
            LayerId(layer),
            HiddenSize(hidden_size),
            RowsPerToken(m_multiplier),
        ) => LoweredCommand {
            kernel: KernelId::RmsNorm,
            library: "rmsnorm",
            // Symbol names are `rmsnorm_<T_act>_s_<T_scale>_specialized`.
            // `scale_dtype_for(p)` flips between `_s_f16_` (Llama
            // family) and `_s_bf16_` (Qwen3 family) based on the
            // canonical's on-disk scale-storage convention.
            function: rmsnorm_kernel_static_name(p, scale_dtype_for(p)),
            constants: super::kernel_constants::RmsNormConstants {
                // Total row count = bucket_m * m_multiplier. For
                // standard residual-stream norms m_multiplier=1
                // (rows-per-token). For per-head q_norm/k_norm
                // (Qwen3) m_multiplier=num_q_heads/num_kv_heads —
                // the input is treated as `[T*heads, head_dim]` and
                // the kernel needs T*heads RMSNORM_M rows.
                bucket_m: super::ids::BucketM(bucket_m * *m_multiplier),
                // Per-instruction `hidden_size` — for the standard
                // residual-stream norm this is `p.hidden_size`; for
                // per-head q_norm/k_norm (Qwen3) it's `p.head_dim`.
                q_size: super::ids::QSize(*hidden_size),
                rms_norm_eps: super::ids::RmsNormEps(p.rms_norm_eps),
                weight_offset: p.norm_weight_offset,
            }
            .into_baked(),
            // Dispatch: `bucket_m * m_multiplier` threadgroups at
            // bake time; runtime scaling rule
            // (`worker::scale_tg_for_num_tokens`) computes
            // `scaled = baseline * n / s.bucket_m` where
            // `n = num_tokens`. Setting `baseline = bucket_m * m_mult`
            // and `s.bucket_m = bucket_m` makes `scaled = num_tokens *
            // m_mult` — exactly the per-head row count we need.
            dispatch: DispatchShape {
                threadgroups: (bucket_m * *m_multiplier, 1, 1),
                threads_per_threadgroup: (THREADS_PER_GROUP, 1, 1),
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::X,
                    bucket_m: super::ids::BucketM(bucket_m),
                }),
            },
            bindings: baked(vec![
                Binding::ArenaSlot {
                    slot: *out_slot,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: *in_slot,
                    binding_index: 1,
                },
                source(
                    w.of(WeightKind::RmsNorm, 0)?,
                    WeightTensor::Weight,
                    super::ids::LayerId(*layer + layer_offset),
                    2,
                ),
            ]),
            gemm_dims: None,
        },

        // ── Scalar-offset RMSNorm (Gemma `rmsnorm(x, weight + 1.0)`) ─
        // Identical to `RmsNorm` except the `(1 + w)` offset rides the
        // instruction's `offset` field (the DSL's explicit `+ 1.0`),
        // not the global `p.norm_weight_offset`. `hidden_size` /
        // `m_multiplier` are baked by `ScalarOffsetRmsNormImpl::fan_out`
        // (residual = HIDDEN_SIZE/1; per-head q/k = head_dim/num_heads).
        I::ScalarOffsetRmsNorm(
            Slot(in_slot),
            Slot(out_slot),
            LayerId(layer),
            GainOffset(offset),
            HiddenSize(hidden_size),
            RowsPerToken(m_multiplier),
        ) => LoweredCommand {
            kernel: KernelId::RmsNorm,
            library: "rmsnorm",
            function: rmsnorm_kernel_static_name(p, scale_dtype_for(p)),
            constants: super::kernel_constants::RmsNormConstants {
                bucket_m: super::ids::BucketM(bucket_m * *m_multiplier),
                q_size: super::ids::QSize(*hidden_size),
                rms_norm_eps: super::ids::RmsNormEps(p.rms_norm_eps),
                weight_offset: *offset,
            }
            .into_baked(),
            dispatch: DispatchShape {
                threadgroups: (bucket_m * *m_multiplier, 1, 1),
                threads_per_threadgroup: (THREADS_PER_GROUP, 1, 1),
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::X,
                    bucket_m: super::ids::BucketM(bucket_m),
                }),
            },
            bindings: baked(vec![
                Binding::ArenaSlot {
                    slot: *out_slot,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: *in_slot,
                    binding_index: 1,
                },
                source(
                    w.of(WeightKind::RmsNorm, 0)?,
                    WeightTensor::Weight,
                    super::ids::LayerId(*layer + layer_offset),
                    2,
                ),
            ]),
            gemm_dims: None,
        },

        // ── Unit-gain RMSNorm (mlx RMSNormNoScale — Gemma4 v_norm) ─
        // Same row math as `RmsNorm` with gain ≡ 1 and NO weight
        // binding. (hidden_size, m_multiplier) are per-instruction —
        // baked per attention class by `RmsNormUnitImpl::fan_out`
        // (Gemma4: sliding 256×8, global 512×1).
        I::RmsNormUnit(
            Slot(in_slot),
            Slot(out_slot),
            HiddenSize(hidden_size),
            RowsPerToken(m_multiplier),
        ) => LoweredCommand {
            kernel: KernelId::RmsNormUnit,
            library: "rmsnorm",
            function: pick_specialized_symbol(
                "rmsnorm_unit_f16_specialized",
                "rmsnorm_unit_bf16_specialized",
                p.metal_dtype,
            ),
            // The unit kernel declares fn-consts 0..2 only (no
            // WEIGHT_OFFSET slot 3); RmsNormConstants sets 0..3 and the
            // extra constant is tolerated.
            constants: super::kernel_constants::RmsNormConstants {
                bucket_m: super::ids::BucketM(bucket_m * *m_multiplier),
                q_size: super::ids::QSize(*hidden_size),
                rms_norm_eps: super::ids::RmsNormEps(p.rms_norm_eps),
                weight_offset: 0.0,
            }
            .into_baked(),
            dispatch: DispatchShape {
                threadgroups: (bucket_m * *m_multiplier, 1, 1),
                threads_per_threadgroup: (THREADS_PER_GROUP, 1, 1),
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::X,
                    bucket_m: super::ids::BucketM(bucket_m),
                }),
            },
            bindings: baked(vec![
                Binding::ArenaSlot {
                    slot: *out_slot,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: *in_slot,
                    binding_index: 1,
                },
            ]),
            gemm_dims: None,
        },

        // ── Multiply by a loaded [1]-shaped weight (Gemma4 layer_scalar) ─
        // out = in * w[0] over the [M, HIDDEN_SIZE] hidden state.
        // Exact-thread elementwise dispatch like `Add`; the weight
        // loads through the RmsNorm-kind accessor (a 1-element gain).
        I::ScalarWeightMul(Slot(in_slot), Slot(out_slot), LayerId(layer)) => LoweredCommand {
            kernel: KernelId::ScalarWeightMul,
            library: "elementwise",
            function: pick_specialized_symbol(
                "scalar_weight_mul_f16_specialized",
                "scalar_weight_mul_bf16_specialized",
                p.metal_dtype,
            ),
            constants: &[],
            dispatch: {
                let mut d =
                    DispatchShape::dispatch_1d(eff_m * (p.hidden_size as u32), THREADS_PER_GROUP);
                d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::X,
                    bucket_m: super::ids::BucketM(bucket_m),
                });
                d
            },
            bindings: baked(vec![
                Binding::ArenaSlot {
                    slot: *out_slot,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: *in_slot,
                    binding_index: 1,
                },
                source(
                    w.of(WeightKind::RmsNorm, 0)?,
                    WeightTensor::Weight,
                    super::ids::LayerId(*layer + layer_offset),
                    2,
                ),
            ]),
            gemm_dims: None,
        },

        // ── Fused residual-add + RMSNorm ───────────────────────────
        I::FusedAddRmsNorm(
            Slot(delta_slot),
            Slot(residual_slot),
            LayerId(layer),
            HiddenSize(hidden_size),
            _m_multiplier,
        ) => {
            LoweredCommand {
                kernel: KernelId::FusedAddRmsNorm,
                library: "fused_add_rmsnorm",
                function: fused_add_rmsnorm_kernel_static_name(p, scale_dtype_for(p)),
                constants: super::kernel_constants::RmsNormConstants {
                    bucket_m: super::ids::BucketM(bucket_m),
                    // Per-instruction hidden_size — see I::RmsNorm
                    // arm. FusedAddRmsNorm is always on the residual
                    // stream so it's always `p.hidden_size`, but
                    // we plumb it through the field for uniformity.
                    q_size: super::ids::QSize(*hidden_size),
                    rms_norm_eps: super::ids::RmsNormEps(p.rms_norm_eps),
                    weight_offset: p.norm_weight_offset,
                }
                .into_baked(),
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, 1, 1),
                    threads_per_threadgroup: (THREADS_PER_GROUP, 1, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    }),
                },
                bindings: baked(vec![
                    // residual: read+write (in-place add target,
                    // norm-input source)
                    Binding::ArenaSlot {
                        slot: *residual_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *delta_slot,
                        binding_index: 1,
                    },
                    source(
                        w.of(WeightKind::RmsNorm, 0)?,
                        WeightTensor::Weight,
                        super::ids::LayerId(*layer + layer_offset),
                        2,
                    ),
                ]),
                gemm_dims: None,
            }
        }

        // ── FusedAddRmsNorm with per-instruction offset (Gemma3) ───
        // `add(delta, residual)` then `rmsnorm(·, weight + offset)` —
        // gemma3's sandwich norms write the `+ 1.0` explicitly, so the
        // offset rides the instruction field instead of the global
        // `p.norm_weight_offset`. Always on the residual stream
        // (HIDDEN_SIZE; m=1) — residual adds are never per-head.
        I::FusedAddRmsNormWithOffset(
            Slot(delta_slot),
            Slot(residual_slot),
            LayerId(layer),
            GainOffset(offset),
        ) => LoweredCommand {
            kernel: KernelId::FusedAddRmsNorm,
            library: "fused_add_rmsnorm",
            function: fused_add_rmsnorm_kernel_static_name(p, scale_dtype_for(p)),
            constants: super::kernel_constants::RmsNormConstants {
                bucket_m: super::ids::BucketM(bucket_m),
                q_size: super::ids::QSize(p.hidden_size as u32),
                rms_norm_eps: super::ids::RmsNormEps(p.rms_norm_eps),
                weight_offset: *offset,
            }
            .into_baked(),
            dispatch: DispatchShape {
                threadgroups: (bucket_m, 1, 1),
                threads_per_threadgroup: (THREADS_PER_GROUP, 1, 1),
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::X,
                    bucket_m: super::ids::BucketM(bucket_m),
                }),
            },
            bindings: baked(vec![
                Binding::ArenaSlot {
                    slot: *residual_slot,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: *delta_slot,
                    binding_index: 1,
                },
                source(
                    w.of(WeightKind::RmsNorm, 0)?,
                    WeightTensor::Weight,
                    super::ids::LayerId(*layer + layer_offset),
                    2,
                ),
            ]),
            gemm_dims: None,
        },

        // ── Gemma4 post-FFN tail: rmsnorm → add → scalar_weight_mul ─
        I::NormAddScalarMul(
            Slot(delta_slot),
            Slot(residual_slot),
            Slot(out_slot),
            LayerId(layer),
            HiddenSize(hidden_size),
        ) => {
            LoweredCommand {
                kernel: KernelId::NormAddScalarMul,
                library: "fused_add_rmsnorm",
                function: norm_add_scalar_mul_kernel_static_name(p, scale_dtype_for(p)),
                // Same fn-const quartet as FusedAddRmsNorm (M, hidden,
                // eps, weight_offset) — the kernel reuses FUSED_ARN_*.
                constants: super::kernel_constants::RmsNormConstants {
                    bucket_m: super::ids::BucketM(bucket_m),
                    q_size: super::ids::QSize(*hidden_size),
                    rms_norm_eps: super::ids::RmsNormEps(p.rms_norm_eps),
                    weight_offset: p.norm_weight_offset,
                }
                .into_baked(),
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, 1, 1),
                    threads_per_threadgroup: (THREADS_PER_GROUP, 1, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    }),
                },
                bindings: baked(vec![
                    // 0: delta (down-proj output; norm input, read)
                    Binding::ArenaSlot {
                        slot: *delta_slot,
                        binding_index: 0,
                    },
                    // 1: residual (read)
                    Binding::ArenaSlot {
                        slot: *residual_slot,
                        binding_index: 1,
                    },
                    // 2: out (write — the new hidden_states)
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 2,
                    },
                    // 3: post-FFN norm gains [hidden] (RmsNorm sub-slot 0)
                    source(
                        w.of(WeightKind::RmsNorm, 0)?,
                        WeightTensor::Weight,
                        super::ids::LayerId(*layer + layer_offset),
                        3,
                    ),
                    // 4: layer_scalar [1] (RmsNorm sub-slot 1)
                    source(
                        w.of(WeightKind::RmsNorm, 1)?,
                        WeightTensor::Weight,
                        super::ids::LayerId(*layer + layer_offset),
                        4,
                    ),
                ]),
                gemm_dims: None,
            }
        }

        // ── Generic dense GEMM ─────────────────────────────────────
        I::Gemm(Slot(in_slot), Slot(out_slot), LayerId(layer), NDim(n), KDim(k)) => {
            // The worker reads `gemm_dims` and dispatches the
            // `gemm_{f16,bf16}_specialized` kernel (M/N/K baked into
            // function constants), so this tile-shape hint is a
            // placeholder; the GEMM bake picks its own grid.
            // `eff_m` shrinks by the merge factor for the merger's
            // post-reshape GEMMs (`[num_tokens / vision_merge_factor,
            // vision_merge_hidden]`); == bucket_m everywhere else. The
            // bf16 GEMM path bakes M from `gemm_dims.m` (m_scaling is
            // forced None for it in the worker), so the divisor MUST land
            // on `gemm_dims.m`, not just the placeholder dispatch.
            let tg_x = eff_m.div_ceil(GEMM_TILE_M);
            let tg_y = (*n).div_ceil(GEMM_TILE_N);
            LoweredCommand {
                kernel: KernelId::Gemm,
                // GEMM is opaque to `pipeline_for_command` — f16 takes the
                // MPS branch, bf16 has its own dims-keyed builder
                // (`pipeline_for_gemm_bf16`). Empty library/function +
                // empty constants signal the worker to route GEMM commands
                // through the special-case path instead.
                library: "",
                function: "",
                constants: &[],
                dispatch: DispatchShape {
                    threadgroups: (tg_x, tg_y, 1),
                    threads_per_threadgroup: (GEMM_TILE_M, GEMM_TILE_N, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    }),
                },
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *in_slot,
                        binding_index: 1,
                    },
                    source(
                        w.of(WeightKind::Linear, 0)?,
                        WeightTensor::Weight,
                        super::ids::LayerId(*layer + layer_offset),
                        2,
                    ),
                ]),
                gemm_dims: Some(GemmDims {
                    m: eff_m,
                    n: *n,
                    k: *k,
                }),
            }
        }

        // ── MLX-affine int4 matmul (qmv + qmm_t) ──────────────────
        //
        // `MetalStep::AffineQmm` covers every per-Linear shape on a
        // metal-quantized model. The dispatcher rule mirrors MLX
        // `quantized.cpp:1387 QuantizedMatmul::eval_gpu`:
        //   * `bucket_m < vector_limit` → matvec (qmv_quad / qmv_fast /
        //     qmv) per `dispatch_qmv` (`:1365`) — D∈{64,128} pow2-bits
        //     wins quad, then N%8==0 ∧ K%512==0 wins fast, else generic.
        //   * `bucket_m ≥ vector_limit` → matmul (qmm_t for transpose=true).
        //     SplitK is a sibling of qmm_t Standard for B==1; lands in
        //     C3 alongside its downstream sum-reduce.
        //
        // `vector_limit` (`QmvBatchLimit`) rides on the step, baked from
        // `get_qmv_batch_limit(K, N, arch_gen)`.
        //
        // Bindings match the kernel signatures in `quantized_qmv.metal`
        // and `quantized_qmm.metal`:
        //   buffer(0) packed weight   buffer(1) scales   buffer(2) biases
        //   buffer(3) x activations   buffer(4) y output
        // K / N (and M for qmm_t) ride as function constants 0/1(/2)
        // post the C1 refactor; the dispatcher never sets them as
        // setBytes — required for dispatch recording.
        //
        // Every sample-rows step lowers its matmul over every row; `sample_rows` realizes it.
        I::AffineQmm(g) | I::SampleRows(g, _) => {
            let AffineMatmul {
                input: Slot(in_slot),
                output: Slot(out_slot),
                layer: LayerId(layer),
                n: NDim(n),
                k: KDim(k),
                group_size: AffineGroupSize(group_size),
                bits: AffineBits(bits),
                vector_limit: QmvBatchLimit(vector_limit),
            } = g;
            let dtype = dequant_dtype_for(p);
            let scale_dtype = scale_dtype_for(p);
            let n_v = *n;
            let k_v = *k;
            let bits_v = *bits;
            let gs = *group_size;
            let vl = *vector_limit;

            if bucket_m < vl {
                // Matvec branch (decode-shape). The MLX-mirrored shape
                // heuristic (qmv_fast when N%8==0 && K%512==0, qmv_quad
                // when K∈{64,128}, else generic) — measured 0.28 ms /
                // 3% TPOT faster than the old cost-CSV pick on
                // single-stream M1 Max (the sweep's per-CB overhead
                // biased it toward generic).
                let layer = super::ids::LayerId(*layer + layer_offset);
                let codes = super::kernel_constants::AffineCodes::of(profile, g.bits.get());
                affine_qmv_command(p, g, bucket_m, None, layer, w.of(WeightKind::Linear, 0)?, codes)
            } else {
                // Matmul branch (prefill-shape). `pick_qmm_t_kernel`
                // mirrors MLX `quantized.cpp:1411-1424 + :788-805`:
                //   * `Standard` when split_k ≤ 1 (target ~512 tgs).
                //   * `SplitK` when n_tiles × m_tiles is sparse enough
                //     that splitting K into `split_k` partitions pushes
                //     the total threadgroup count up to roughly 512;
                //     fed by the `splitk_reduce_sum` kernel that
                //     collapses the `[split_k, M, N]` partial to
                //     `[M, N]` in the AffineQmm's out slot.
                // NAX hardware MMA (`affine_qmm_t_nax`, MPP `matmul2d`):
                // M5+/A19+ only — `is_nax_capable` gates on arch gen ≥ 17
                // (MLX `mlx/backend/metal/device.cpp:828`). M4 and earlier
                // lack the unit (M4's `matmul2d` emulates and produces a
                // wrong layout), so `is_nax_capable` is false there.
                // ~3× prefill GEMM speedup on M5.
                let is_nax = profile.is_some_and(|p| crate::targets::is_nax_capable(p.generation));
                // 8-bit weights (Gemma4 MLP projections): NAX has
                // `_b_8_` instantiations (byte-per-element W-loader,
                // same MMA) — the dominant Gemma4 prefill lever (the
                // b8 MLP was ~77% of prefill GPU time on the Standard
                // kernel). SplitK stays b4-only, so a b8 SplitK pick
                // downgrades to Standard.
                let kernel = match pick_qmm_t_kernel(bucket_m, n_v, k_v, /*B=*/ 1, gs, is_nax) {
                    QmmTKernel::SplitK { .. } if *bits == 8 => QmmTKernel::Standard,
                    k => k,
                };
                // NAX tile is 64×64 so align check uses 64; Standard/SplitK use 32.
                let aligned_n = match kernel {
                    QmmTKernel::Nax => n_v.is_multiple_of(64),
                    _ => n_v.is_multiple_of(32),
                };
                // M1 fast-path: bf16 simdgroup MMA is software emulation
                // (~1.7× slower than f16). Pick T_compute=F16 for the
                // qmm_t kernel — kernel reads bf16 from device memory,
                // casts to f16 on threadgroup-tile populate, runs MMA
                // in f16, casts back to bf16 on store. Output is bf16
                // so the residual stream is unchanged. M2+ has
                // hardware bf16 so we keep T_compute=T_act there.
                use crate::targets::bf16_simdgroup_is_slow_path;
                let f16_compute_eligible = profile
                    .map(|p| bf16_simdgroup_is_slow_path(p.generation))
                    .unwrap_or(false)
                    && matches!(dtype, DequantDtype::Bf16)
                    && !matches!(kernel, QmmTKernel::Nax)
                    // The mixed-compute (`_c_f16_`) qmm_t kernels (INST_QMM_T_C)
                    // are instantiated for bits=4 ONLY. Applying the M1 f16
                    // flip to an 8-bit weight would mis-resolve to a b4 symbol
                    // (the with-compute name builder omits `bits`) and decode
                    // 8-bit as 4-bit → silent garbage (OptiQ on M1). 8-bit
                    // keeps same-compute bf16 → the existing b8 kernel.
                    && bits_v == 4;
                let compute_dtype = if f16_compute_eligible {
                    DequantDtype::F16
                } else {
                    dtype
                };
                // W4A8 on the matrix unit's int8 lane: a pre-pass quantizes
                // the activations per (row, 64-chunk) into the shared scratch,
                // then the GEMM multiplies them against the offset-8 codes.
                let codes = super::kernel_constants::AffineCodes::of(profile, bits_v);
                let w4a8_tile = W4a8Tile::for_n(n_v).filter(|_| {
                    matches!(kernel, QmmTKernel::Nax)
                        && codes == super::kernel_constants::AffineCodes::Offset8
                        && matches!(gs, 64 | 128)
                        && k_v.is_multiple_of(64)
                        && bucket_m.is_multiple_of(W4A8_TILE_ROWS)
                });
                if let Some(tile) = w4a8_tile {
                    *splitk_scratch_bytes =
                        (*splitk_scratch_bytes).max(w4a8_scratch_bytes(bucket_m, k_v));
                    let constants = || {
                        super::kernel_constants::AffineQmmTConstants {
                            k: super::ids::KDimI32(k_v as i32),
                            n: super::ids::NDimI32(n_v as i32),
                            m: super::ids::MDimI32(bucket_m as i32),
                            codes,
                        }
                        .into_baked()
                    };
                    let rows = || {
                        Some(crate::interpreter::metal::lowered::MScaling {
                            seq_axis: None,
                            axis: crate::tape::lowered::MScaleAxis::Y,
                            bucket_m: super::ids::BucketM(bucket_m),
                        })
                    };
                    let quant = LoweredCommand {
                        kernel: KernelId::AffineW4a8Quant,
                        library: "quantized_qmm_nax",
                        function: w4a8_quant_static_name(W4a8Rows::Dense, dtype),
                        constants: constants(),
                        dispatch: DispatchShape {
                            threadgroups: ((k_v / 64).div_ceil(16), bucket_m, 1),
                            threads_per_threadgroup: (128, 1, 1),
                            m_scaling: rows(),
                        },
                        bindings: baked(vec![
                            Binding::ArenaSlot {
                                slot: *in_slot,
                                binding_index: 0,
                            },
                            Binding::Scratch { binding_index: 1 },
                        ]),
                        gemm_dims: None,
                    };
                    let mut bindings = affine_qmm_bindings(
                        *in_slot,
                        *out_slot,
                        super::ids::LayerId(*layer + layer_offset),
                        w.of(WeightKind::Linear, 0)?,
                    );
                    for b in bindings.iter_mut() {
                        if matches!(
                            b,
                            Binding::ArenaSlot {
                                binding_index: 3,
                                ..
                            }
                        ) {
                            *b = Binding::Scratch { binding_index: 3 };
                        }
                    }
                    let gemm = LoweredCommand {
                        kernel: KernelId::AffineQmmW4a8,
                        library: "quantized_qmm_nax",
                        function: qmm_w4a8_static_name(
                            W4a8Rows::Dense,
                            dtype,
                            scale_dtype,
                            gs,
                            tile,
                        ),
                        constants: constants(),
                        dispatch: DispatchShape {
                            threadgroups: (n_v / tile.cols(), bucket_m / W4A8_TILE_ROWS, 1),
                            threads_per_threadgroup: (32 * tile.simdgroups(), 1, 1),
                            m_scaling: rows(),
                        },
                        bindings: baked(bindings),
                        gemm_dims: None,
                    };
                    return Ok(vec![quant, gemm]);
                }
                match kernel {
                    QmmTKernel::Nax => {
                        let (tg, tpg) = qmm_t_dispatch_shape(kernel, bucket_m, n_v, /*B=*/ 1);
                        LoweredCommand {
                            kernel: KernelId::AffineQmmTNax,
                            library: "quantized_qmm_nax",
                            function: qmm_t_kernel_static_name(
                                kernel,
                                dtype,
                                scale_dtype,
                                bits_v,
                                gs,
                                aligned_n,
                            ),
                            constants: super::kernel_constants::AffineQmmTConstants {
                                k: super::ids::KDimI32(k_v as i32),
                                n: super::ids::NDimI32(n_v as i32),
                                m: super::ids::MDimI32(bucket_m as i32),
                                codes: super::kernel_constants::AffineCodes::of(profile, bits_v),
                            }
                            .into_baked(),
                            dispatch: DispatchShape {
                                threadgroups: tg,
                                threads_per_threadgroup: tpg,
                                // qmm_t NAX grid = (n_tiles, m_tiles=ceil(M/64), B)
                                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                                    seq_axis: None,
                                    axis: crate::tape::lowered::MScaleAxis::Y,
                                    bucket_m: super::ids::BucketM(bucket_m),
                                }),
                            },
                            bindings: baked(affine_qmm_bindings(
                                *in_slot,
                                *out_slot,
                                super::ids::LayerId(*layer + layer_offset),
                                w.of(WeightKind::Linear, 0)?,
                            )),
                            gemm_dims: None,
                        }
                    }
                    QmmTKernel::Standard => {
                        let (tg, tpg) = qmm_t_dispatch_shape(kernel, bucket_m, n_v, /*B=*/ 1);
                        LoweredCommand {
                            kernel: KernelId::AffineQmmT,
                            library: "quantized_qmm",
                            function: qmm_t_kernel_static_name_with_compute(
                                kernel,
                                dtype,
                                compute_dtype,
                                scale_dtype,
                                bits_v,
                                gs,
                                aligned_n,
                            ),
                            constants: super::kernel_constants::AffineQmmTConstants {
                                k: super::ids::KDimI32(k_v as i32),
                                n: super::ids::NDimI32(n_v as i32),
                                m: super::ids::MDimI32(bucket_m as i32),
                                codes: super::kernel_constants::AffineCodes::of(profile, bits_v),
                            }
                            .into_baked(),
                            dispatch: DispatchShape {
                                threadgroups: tg,
                                threads_per_threadgroup: tpg,
                                // qmm_t Standard grid = (n_tiles, m_tiles=ceil(M/32), B)
                                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                                    seq_axis: None,
                                    axis: crate::tape::lowered::MScaleAxis::Y,
                                    bucket_m: super::ids::BucketM(bucket_m),
                                }),
                            },
                            bindings: baked(affine_qmm_bindings(
                                *in_slot,
                                *out_slot,
                                super::ids::LayerId(*layer + layer_offset),
                                w.of(WeightKind::Linear, 0)?,
                            )),
                            gemm_dims: None,
                        }
                    }
                    QmmTKernel::SplitK {
                        split_k,
                        k_partition_size,
                    } => {
                        // Two commands:
                        //   (1) qmm_t_splitk writes the `[split_k, M, N]`
                        //       partial into `Binding::Scratch`.
                        //   (2) splitk_reduce_sum reads scratch and
                        //       reduces along axis 0 into the AffineQmm's
                        //       arena slot.
                        let elem_bytes = elem_size_bytes(dtype);
                        let scratch_bytes = split_k
                            .saturating_mul(bucket_m)
                            .saturating_mul(n_v)
                            .saturating_mul(elem_bytes);
                        *splitk_scratch_bytes = (*splitk_scratch_bytes).max(scratch_bytes);

                        let (tg, tpg) = qmm_t_dispatch_shape(kernel, bucket_m, n_v, /*B=*/ 1);
                        let qmm_t_cmd = LoweredCommand {
                            kernel: KernelId::AffineQmmTSplitK,
                            library: "quantized_qmm",
                            function: qmm_t_kernel_static_name_with_compute(
                                kernel,
                                dtype,
                                compute_dtype,
                                scale_dtype,
                                bits_v,
                                gs,
                                aligned_n,
                            ),
                            // SplitK needs FOUR function constants:
                            // (0=K, 1=N, 2=M, 3=k_partition_size) per
                            // `quantized_qmm.metal:80-83`. The
                            // standalone `MetalAffineQmmT::execute`
                            // (`quantized.rs:776-781`) emits the same
                            // four; missing `k_partition_size` (slot 3)
                            // leaves the partition stride undefined and
                            // every layer's prefill output is garbage.
                            constants: super::kernel_constants::AffineQmmTSplitKConstants {
                                k: super::ids::KDimI32(k_v as i32),
                                n: super::ids::NDimI32(n_v as i32),
                                m: super::ids::MDimI32(bucket_m as i32),
                                k_partition_size: super::ids::KPartitionSizeI32(
                                    k_partition_size as i32,
                                ),
                                codes: super::kernel_constants::AffineCodes::of(profile, bits_v),
                            }
                            .into_baked(),
                            dispatch: DispatchShape {
                                threadgroups: tg,
                                threads_per_threadgroup: tpg,
                                // qmm_t SplitK grid = (n_tiles, m_tiles=ceil(M/32), split_k)
                                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                                    seq_axis: None,
                                    axis: crate::tape::lowered::MScaleAxis::Y,
                                    bucket_m: super::ids::BucketM(bucket_m),
                                }),
                            },
                            bindings: baked(affine_qmm_splitk_bindings(
                                *in_slot,
                                super::ids::LayerId(*layer + layer_offset),
                                w.of(WeightKind::Linear, 0)?,
                            )),
                            gemm_dims: None,
                        };

                        // splitk_reduce_sum: bindings (0=output → out_slot,
                        // 1=intermediate → Scratch), function constants
                        // (0=M, 1=N, 2=split_k), 1D dispatch over M*N
                        // output elements.
                        let nthreads = bucket_m.saturating_mul(n_v);
                        let reduce_cmd = LoweredCommand {
                            kernel: KernelId::SplitKReduceSum,
                            library: "quantized_splitk_reduce",
                            function: splitk_reduce_kernel_static_name(dtype),
                            constants: super::kernel_constants::SplitKReduceSumConstants {
                                bucket_m: super::ids::BucketM(bucket_m),
                                n: super::ids::NDim(n_v),
                                split_k: super::ids::SplitK(split_k),
                            }
                            .into_baked(),
                            dispatch: {
                                let mut d = DispatchShape::dispatch_1d(nthreads, THREADS_PER_GROUP);
                                // groups = ceil(bucket_m * n_v / TPG) is linear
                                // in M; proportional scaling shrinks it for
                                // actual num_tokens.
                                d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                                    seq_axis: None,
                                    axis: crate::tape::lowered::MScaleAxis::X,
                                    bucket_m: super::ids::BucketM(bucket_m),
                                });
                                d
                            },
                            bindings: baked(vec![
                                Binding::ArenaSlot {
                                    slot: *out_slot,
                                    binding_index: 0,
                                },
                                Binding::Scratch { binding_index: 1 },
                            ]),
                            gemm_dims: None,
                        };
                        return Ok(vec![qmm_t_cmd, reduce_cmd]);
                    }
                }
            }
        }

        // ── Fused silu(gate) * up for the decomposed q-MLP path ───
        //
        // C4 will start emitting `(AffineQmm gate, AffineQmm up,
        // SiluMul)` from the macro when both gate_proj and up_proj
        // are MLX-affine quantized — `MetalFusedGateUpSiluMulImpl`
        // currently rejects non-Dense storage. SiluMul is the
        // elementwise tail of that decomposition; the gate / up
        // arena slots hold the two AffineQmm outputs and SiluMul
        // writes `silu(gate) * up` into out_slot. `width` (per-row
        // element count) rides on the instruction — baked at macro
        // time from the claim's solved gate/up Gemm N, NOT from
        // `p.intermediate_size`: one model can carry SwiGLU blocks
        // of different widths (Qwen3.5-MoE's 512-wide shared expert
        // vs a dense MLP), and a zero/mismatched global bound made
        // this dispatch a silent no-op.
        I::SiluMul(Slot(gate_slot), Slot(up_slot), Slot(out_slot), IntermediateSize(width)) => {
            let dtype = dequant_dtype_for(p);
            assert!(
                *width > 0,
                "SiluMul: width must be > 0 — a zero width dispatches no threads \
                 and silently passes raw gate_proj output downstream"
            );
            let n = bucket_m * width;
            LoweredCommand {
                kernel: KernelId::SiluMul,
                library: "silu_mul",
                function: silu_mul_static_name(dtype),
                constants: super::kernel_constants::SiluMulConstants {
                    n: super::ids::HiddenSize(n),
                }
                .into_baked(),
                // 1D dispatch over M * intermediate_size output elements,
                // one thread per element. Threadgroup width clamped to
                // the pipeline's max at execute time would be cleaner;
                // for now match the elementwise convention used by
                // `KernelId::Add` / `KernelId::ScalarMul`. Scales
                // proportionally with M at dispatch time.
                dispatch: {
                    let mut d = DispatchShape::dispatch_1d(n, THREADS_PER_GROUP);
                    d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    });
                    d
                },
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *gate_slot,
                        binding_index: 1,
                    },
                    Binding::ArenaSlot {
                        slot: *up_slot,
                        binding_index: 2,
                    },
                ]),
                gemm_dims: None,
            }
        }

        // ── Fused gelu_tanh(gate) * up — GeGLU q-MLP tail (Gemma) ──
        //
        // GELU sibling of the `SiluMul` arm above: the decomposed
        // GeGLU MLP emits `(AffineQmm gate, AffineQmm up, GeluMul)`
        // when gate/up are MLX-affine quantized (Gemma2/3/4). Same
        // shapes, same dispatch, gelu_tanh activation.
        I::GeluMul(Slot(gate_slot), Slot(up_slot), Slot(out_slot)) => {
            let dtype = dequant_dtype_for(p);
            let n = bucket_m * (p.intermediate_size as u32);
            LoweredCommand {
                kernel: KernelId::GeluMul,
                library: "silu_mul",
                function: gelu_mul_static_name(dtype),
                constants: super::kernel_constants::SiluMulConstants {
                    n: super::ids::HiddenSize(n),
                }
                .into_baked(),
                dispatch: {
                    let mut d = DispatchShape::dispatch_1d(n, THREADS_PER_GROUP);
                    d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    });
                    d
                },
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *gate_slot,
                        binding_index: 1,
                    },
                    Binding::ArenaSlot {
                        slot: *up_slot,
                        binding_index: 2,
                    },
                ]),
                gemm_dims: None,
            }
        }

        // ── Qwen3.5 attention output gate: out = attn * sigmoid(gate) ──
        I::GateApply(Slot(attn_slot), Slot(gate_slot), Slot(out_slot)) => {
            let dtype = dequant_dtype_for(p);
            // out is [M, num_heads * head_dim]; one thread per element,
            // m_scaling rescales the X axis to the runtime token count.
            let n = bucket_m * (p.num_q_heads * p.head_dim);
            LoweredCommand {
                kernel: KernelId::GateApply,
                library: "gate_apply",
                function: gate_apply_static_name(dtype),
                constants: super::kernel_constants::SiluMulConstants {
                    n: super::ids::HiddenSize(n),
                }
                .into_baked(),
                dispatch: {
                    let mut d = DispatchShape::dispatch_1d(n, THREADS_PER_GROUP);
                    d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    });
                    d
                },
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *attn_slot,
                        binding_index: 1,
                    },
                    Binding::ArenaSlot {
                        slot: *gate_slot,
                        binding_index: 2,
                    },
                ]),
                gemm_dims: None,
            }
        }

        // ── Qwen3.5-MoE shared-expert combine: out = routed + shared_y * sigmoid(g) ──
        I::GateScale(Slot(routed_slot), Slot(shared_slot), Slot(gate_slot), Slot(out_slot)) => {
            let dtype = dequant_dtype_for(p);
            // routed/shared_y/out are [M, hidden] residual-stream tiles
            // (the op sits post-MoE); g is [M, 1], row-broadcast via
            // `row = gid / cols` in the shader. One thread per output
            // element, m_scaling rescales X to the runtime token count
            // (cols stays hidden_size, so the row index is M-invariant).
            let cols = p.hidden_size as u32;
            let n = bucket_m * cols;
            LoweredCommand {
                kernel: KernelId::GateScale,
                library: "gate_scale",
                function: gate_scale_static_name(dtype),
                constants: super::kernel_constants::GateScaleConstants {
                    n: super::ids::HiddenSize(n),
                    cols,
                }
                .into_baked(),
                dispatch: {
                    let mut d = DispatchShape::dispatch_1d(n, THREADS_PER_GROUP);
                    d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    });
                    d
                },
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *routed_slot,
                        binding_index: 1,
                    },
                    Binding::ArenaSlot {
                        slot: *shared_slot,
                        binding_index: 2,
                    },
                    Binding::ArenaSlot {
                        slot: *gate_slot,
                        binding_index: 3,
                    },
                ]),
                gemm_dims: None,
            }
        }

        // ── Qwen3.5 attention output-gate split (per-head deinterleave) ──
        I::GateSplit(Slot(qg_slot), Slot(q_slot), Slot(gate_slot)) => {
            let dtype = dequant_dtype_for(p);
            // Each output is [M, num_heads * head_dim]; one thread per
            // output element writes both the query and gate halves.
            let n = bucket_m * (p.num_q_heads * p.head_dim);
            LoweredCommand {
                kernel: KernelId::GateSplit,
                library: "gate_split",
                function: gate_split_static_name(dtype),
                constants: super::kernel_constants::GateSplitConstants {
                    n: super::ids::HiddenSize(n),
                    head_dim: p.head_dim,
                    num_heads: p.num_q_heads,
                }
                .into_baked(),
                dispatch: {
                    let mut d = DispatchShape::dispatch_1d(n, THREADS_PER_GROUP);
                    d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    });
                    d
                },
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: *q_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *gate_slot,
                        binding_index: 1,
                    },
                    Binding::ArenaSlot {
                        slot: *qg_slot,
                        binding_index: 2,
                    },
                ]),
                gemm_dims: None,
            }
        }

        // ── MLX-affine int4 quantized embedding (P6) ─────────────
        //
        // `MetalStep::AffineEmbed` replaces `MetalStep::Embed`
        // when `model.embed_tokens` ships as a quantized triple
        // `(weight=U32, scales, biases)` — i.e. every
        // `mlx-community/*-4bit` checkpoint. The fused
        // `affine_embed_<dtype>_gs_<gs>_b_4` kernel reads
        // `vocab_idx = indices[token_row]` then dequants from row
        // `vocab_idx` of the packed weight + scales + biases in one
        // pass (faithful port of `nn.QuantizedEmbedding.__call__`).
        //
        // 2D dispatch:
        //   threadgroups = (ceil((Q_SIZE/2) / 256), bucket_m, 1)
        //   threads_per_threadgroup = (256, 1, 1)
        // The kernel bounds-checks `index.x * 2 >= hidden_size`,
        // which keeps the partial trailing threadgroup safe when
        // Q_SIZE/2 is not a multiple of 256 (Llama-3.2-3B's
        // Q_SIZE=3072 → bytes_per_row=1536 = 6 × 256, clean; Qwen2
        // 1.5B's Q_SIZE=1536 → 768 = 3 × 256, clean; but
        // e.g. Q_SIZE=2048 → 1024 = 4 × 256, no partial threads —
        // pessimistically still safe).
        //
        // Bindings (mirror `affine_qmm_bindings` ordering for
        // consistency with the standalone `MetalAffineEmbed::execute`):
        //   buffer(0) packed weight   buffer(1) scales   buffer(2) biases
        //   buffer(3) input_ids       buffer(4) out (arena[out_slot])
        // hidden_size rides as `[[function_constant(0)]]`.
        I::AffineEmbed(Slot(out_slot), AffineGroupSize(group_size), AffineBits(bits)) => {
            let dtype = dequant_dtype_for(p);
            let scale_dtype = scale_dtype_for(p);
            let bits_v = *bits;
            let gs = *group_size;
            assert!(
                matches!(bits_v, 4 | 8),
                "AffineEmbed: only bits ∈ {{4, 8}} is wired (4-bit default; \
                 8-bit for MLX-native mixed/dynamic quant like OptiQ whose \
                 embed_tokens is 8-bit); got bits={bits_v}"
            );
            assert!(
                matches!(gs, 32 | 64 | 128),
                "AffineEmbed: only group_size ∈ {{32, 64, 128}} is wired \
                 (mlx-community uses gs=64 for every Llama/Qwen/Gemma 4bit); \
                 got gs={gs}"
            );
            // AffineEmbed reads rows of `hidden_size` (residual-stream
            // width) — NOT `Q_SIZE` (= num_q_heads * head_dim).
            // Llama-3.x / Qwen2.5 have hidden==Q_SIZE so the
            // pre-existing `p.q_size` worked by coincidence;
            // Qwen3-30B-A3B has hidden=2048, Q_SIZE=4096 — the
            // wrong width yielded out-of-bounds `gindex` into scales
            // and garbage embed output (silent, no fault).
            let hidden_size = p.hidden_size as u32;
            // Packed bytes per token row = hidden * bits / 8: bits=4 packs
            // 2 codes/byte (hidden/2), bits=8 packs 1 code/byte (hidden).
            // One thread per packed byte.
            let bytes_per_row = hidden_size * bits_v / 8;
            let groups_x = bytes_per_row.div_ceil(THREADS_PER_GROUP);
            LoweredCommand {
                kernel: KernelId::AffineEmbed,
                library: "quantized_dequantize",
                function: affine_embed_kernel_static_name(dtype, scale_dtype, gs, bits_v),
                constants: super::kernel_constants::AffineEmbedConstants {
                    hidden_size: super::ids::HiddenSize(hidden_size),
                    codes: super::kernel_constants::AffineCodes::of(profile, bits_v),
                }
                .into_baked(),
                dispatch: DispatchShape {
                    threadgroups: (groups_x, bucket_m, 1),
                    threads_per_threadgroup: (THREADS_PER_GROUP, 1, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::Y,
                        bucket_m: super::ids::BucketM(bucket_m),
                    }),
                },
                bindings: baked(vec![
                    source(
                        w.of(WeightKind::AffineQuantEmbedding, 0)?,
                        WeightTensor::Weight,
                        super::ids::LayerId(0),
                        0,
                    ),
                    source(
                        w.of(WeightKind::AffineQuantEmbedding, 0)?,
                        WeightTensor::AffineScales,
                        super::ids::LayerId(0),
                        1,
                    ),
                    source(
                        w.of(WeightKind::AffineQuantEmbedding, 0)?,
                        WeightTensor::AffineBiases,
                        super::ids::LayerId(0),
                        2,
                    ),
                    Binding::Runtime {
                        kind: RuntimeBindingKind::InputIds,
                        binding_index: 3,
                    },
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 4,
                    },
                ]),
                gemm_dims: None,
            }
        }

        // ── Fused gate-up SwiGLU MLP (GEMM + SwiGLU in one dispatch) ──
        //
        // Two specialized variants behind `KernelId::FusedGateUpSiluMul`:
        //
        // - **Decode (`bucket_m == 1`)**: dispatches
        //   `fused_gate_up_silu_mul_decode_*_specialized`, which uses
        //   simd_sum dot products. 256 threads/group = 8 simdgroups
        //   × 32 lanes; one simdgroup owns one output. Threadgroup
        //   grid: `(ceil(N/4), 1, 1)` — `MLP_DECODE_BLOCK_M = 4`
        //   outputs per group.
        //
        // - **Prefill (`bucket_m >= 2`)**: dispatches
        //   `fused_gate_up_silu_mul_gemm_steel_*_specialized` (MLX-
        //   steel pattern, BM=BN=32, BK=16, WM=WN=2). 128
        //   threads/group = 4 simdgroups; each simdgroup carries
        //   2×2 = 4 `simdgroup_*8x8` accumulator frags per gate / up.
        //   Threadgroup grid: `(ceil(N/32), ceil(M/32), 1)`.
        //
        // Bindings are identical for both variants: (out, in, weight)
        // at indices 0/1/2. The kernel splits the packed `[gate|up]`
        // weight `[2*N, K]` internally — gate rows [0, N), up rows
        // [N, 2N). The decode-vs-steel symbol pick happens inline
        // below against `bucket_m`.
        I::FusedGateUpSiluMul(Slot(in_slot), Slot(out_slot), LayerId(layer)) => {
            fused_gate_up_mul_cmd(
                p,
                *in_slot,
                *out_slot,
                *layer,
                false,
                bucket_m,
                w.of(WeightKind::Linear, 0)?,
                layer_offset,
            )
        }
        // Dense GeGLU (Gemma3 text MLP) — same fused gate/up GEMM +
        // activation-mul kernel as SiLU; the `IS_GELU` fn-const flips
        // the epilogue to gelu_approx (tanh). The quant GeGLU path
        // decomposes to AffineQmm + GeluMul instead.
        I::FusedGateUpGeluMul(Slot(in_slot), Slot(out_slot), LayerId(layer)) => {
            fused_gate_up_mul_cmd(
                p,
                *in_slot,
                *out_slot,
                *layer,
                true,
                bucket_m,
                w.of(WeightKind::Linear, 0)?,
                layer_offset,
            )
        }

        // ── RoPE + KV cache append ─────────────────────────────────
        I::RopeAppend(
            _q_slot,
            _k_slot,
            _v_slot,
            Slot(q_out_slot),
            Slot(k_out_slot),
            Slot(v_out_slot),
            LayerId(layer),
            _pairing,
            class,
            _kv_offsets,
        ) => {
            // 2D dispatch: (M, num_heads) — one threadgroup per
            // (token, head) pair rotates the head's `head_dim` slice
            // and writes K/V to the layer's paged cache page.
            //
            // Geometry class (Gemma4): global tiles use GLOBAL_*
            // (512 head_dim / 1 kv head / rot 128 proportional);
            // sliding tiles use the base consts (256 / 8 / full).
            // Identity on uniform models.
            let is_global = *class == AttnMask::Causal;
            let (hd, n_kv, rd, bs) = if is_global {
                (
                    p.global_head_dim,
                    p.num_global_kv_heads,
                    p.global_rot_dim,
                    // Page-unified GLOBAL block size — the slot_mapping for a
                    // full layer is encoded with this bs (Gemma4: 32).
                    p.global_block_size,
                )
            } else {
                (p.head_dim, p.num_kv_heads, p.rot_dim, p.block_size)
            };
            // Rotation pairing: standard NeoX pairs lane i with
            // i + rot_dim/2 INSIDE the rot window (full rope and
            // HF-style partial rope, e.g. Qwen3.5 rot 64 of 256).
            // Gemma4's proportional rope (mlx `ProportionalRoPE`)
            // instead rotates the first rot_dim/2 lanes of EACH
            // head half — lane i pairs with i + head_dim/2.
            let pair_off = if is_global && p.rope_proportional {
                hd / 2
            } else {
                rd / 2
            };
            let n_q_heads = p.num_q_heads;
            LoweredCommand {
                kernel: KernelId::RopeAppend,
                library: "rope",
                function: pick_specialized_symbol(
                    "rope_append_f16_specialized",
                    "rope_append_bf16_specialized",
                    p.metal_dtype,
                ),
                constants: super::kernel_constants::RopeAppendConstants {
                    head_dim: super::ids::HeadDim(hd),
                    num_q_heads: super::ids::NumQHeads(p.num_q_heads),
                    num_kv_heads: super::ids::NumKvHeads(n_kv),
                    rot_dim: super::ids::RotDim(rd),
                    block_size: super::ids::BlockSize(bs),
                    blocks_per_chunk: super::ids::BlocksPerChunk(crate::BLOCKS_PER_CHUNK),
                    pair_off: super::ids::RopePairOff(pair_off),
                    // Spans: store relocatable K unrotated (skip K-rotation
                    // for flagged blocks). None → byte-identical when off.
                    rope_on_read: if p.rope_on_read { Some(1) } else { None },
                }
                .into_baked(),
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, n_q_heads, 1),
                    threads_per_threadgroup: (hd, 1, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    }),
                },
                bindings: super::kernel_bindings::RopeAppendBindingSet {
                    q_out: super::ids::ArenaSlotIdx(*q_out_slot),
                    k_out: super::ids::ArenaSlotIdx(*k_out_slot),
                    v_out: super::ids::ArenaSlotIdx(*v_out_slot),
                    cos_sin: w.cos_sin(p)?,
                    layer: super::ids::LayerId(*layer + layer_offset),
                }
                .into_baked(),
                gemm_dims: None,
            }
        }

        // ── Gemma4 norm-prologue rope: per-head q/k rmsnorm + v unit-
        // norm folded into the rope dispatch ────────────────────────
        I::RopeAppendNormed(
            _q_slot,
            Slot(k_slot),
            Slot(v_slot),
            Slot(q_out_slot),
            _k_out_slot,
            _v_out_slot,
            LayerId(layer),
            pairing,
            class,
        ) => {
            assert!(
                *pairing == RopeFormTag::NeoX,
                "metal lowering: RopeAppendNormed is NeoX-only (the matcher \
                 requires OpKind::RopeAppend)"
            );
            // Same geometry-class selection as the RopeAppend arm.
            let is_global = *class == AttnMask::Causal;
            let (hd, n_kv, rd, bs) = if is_global {
                (
                    p.global_head_dim,
                    p.num_global_kv_heads,
                    p.global_rot_dim,
                    p.global_block_size,
                )
            } else {
                (p.head_dim, p.num_kv_heads, p.rot_dim, p.block_size)
            };
            assert!(
                hd <= 512,
                "metal lowering: RopeAppendNormed threadgroup staging is sized \
                 for head_dim <= 512 (got {hd})"
            );
            let pair_off = if is_global && p.rope_proportional {
                hd / 2
            } else {
                rd / 2
            };
            let n_q_heads = p.num_q_heads;
            LoweredCommand {
                kernel: KernelId::RopeAppendNormed,
                library: "rope",
                function: rope_append_normed_kernel_static_name(p, scale_dtype_for(p)),
                constants: super::kernel_constants::RopeAppendNormedConstants {
                    head_dim: super::ids::HeadDim(hd),
                    num_q_heads: super::ids::NumQHeads(n_q_heads),
                    num_kv_heads: super::ids::NumKvHeads(n_kv),
                    rot_dim: super::ids::RotDim(rd),
                    block_size: super::ids::BlockSize(bs),
                    blocks_per_chunk: super::ids::BlocksPerChunk(crate::BLOCKS_PER_CHUNK),
                    pair_off: super::ids::RopePairOff(pair_off),
                    rms_norm_eps: super::ids::RmsNormEps(p.rms_norm_eps),
                    weight_offset: p.norm_weight_offset,
                    // Spans: store relocatable K NORMED-but-unrotated.
                    rope_on_read: if p.rope_on_read { Some(1) } else { None },
                }
                .into_baked(),
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, n_q_heads, 1),
                    threads_per_threadgroup: (hd, 1, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    }),
                },
                // q_out aliases the raw q storage (in-place norm+rotate,
                // staged through TG memory); k/v bind their RAW slots
                // read-only — the kernel writes K/V to the cache only.
                bindings: super::kernel_bindings::RopeAppendNormedBindingSet {
                    q_out: super::ids::ArenaSlotIdx(*q_out_slot),
                    k_in: super::ids::ArenaSlotIdx(*k_slot),
                    v_in: super::ids::ArenaSlotIdx(*v_slot),
                    cos_sin: w.cos_sin(p)?,
                    q_gains: w.of(WeightKind::RmsNorm, 0)?,
                    k_gains: w.of(WeightKind::RmsNorm, 1)?,
                    layer: super::ids::LayerId(*layer + layer_offset),
                }
                .into_baked(),
                gemm_dims: None,
            }
        }

        // ── Decode-bucket attention (single query token / seq) ─────
        I::AttentionViaCache(Slot(q_slot), Slot(out_slot), LayerId(layer), _pairing) => {
            // Spans rope-on-read (GLOBAL class). All-None when !ROPE_ON_READ.
            let (ror_rd, ror_po, ror_on, ror_bind) = rope_on_read_params(p, true);
            // 2D dispatch: (batch, num_q_heads). Each threadgroup
            // computes one head's attention output for one sequence.
            // `bucket_m == batch` for decode buckets.
            //
            // v2 kernel (paged-cache port of MLX sdpa_vector) uses
            // 1024 threads/group = 32 simdgroups × 32 lanes. Each
            // simdgroup processes 1/32 of the K axis with online
            // softmax; no per-token threadgroup_barrier in the K
            // loop. Production path now wires both f16 and bf16 to v2
            // (the v1 2-pass-softmax kernel accumulated bf16 rounding
            // error per layer on Llama-3.2 decode and produced
            // degenerate output after the first decode token).
            let n_q_heads = p.num_q_heads;
            LoweredCommand {
                kernel: KernelId::AttentionViaCache,
                library: "attention",
                function: pick_specialized_symbol(
                    "attention_via_cache_v2_f16_specialized",
                    "attention_via_cache_v2_bf16_specialized",
                    p.metal_dtype,
                ),
                constants: super::kernel_constants::AttentionViaCacheConstants {
                    // `attention()` tiles are the GLOBAL class on
                    // hybrid sliding/global arches (Gemma4: 512×1kv);
                    // GLOBAL_* default to the base values on uniform
                    // models, so this is identity for Llama/Qwen.
                    head_dim: super::ids::HeadDim(p.global_head_dim),
                    num_q_heads: super::ids::NumQHeads(p.num_q_heads),
                    num_kv_heads: super::ids::NumKvHeads(p.num_global_kv_heads),
                    attn_scale: super::ids::AttnScale(p.attn_scale),
                    // GLOBAL class block size — vLLM page-unifies the full
                    // class UP to the sliding page (Gemma4: 32 vs sliding 16).
                    // Defaults to BLOCK_SIZE on uniform arches.
                    block_size: super::ids::BlockSize(p.global_block_size),
                    max_blocks: super::ids::MaxBlocksPerSeq(block_cap),
                    blocks_per_chunk: super::ids::BlocksPerChunk(attention_blocks_per_chunk(
                        chunked,
                    )),
                    // Full attention: window disabled.
                    window: super::ids::AttnWindow(0),
                    rot_dim: ror_rd,
                    pair_off: ror_po,
                    rope_on_read: ror_on,
                    pair_coresident: pair_coresident_param(
                        ror_rd,
                        ror_po,
                        ror_on,
                        p.global_head_dim,
                    ),
                }
                .into_baked(),
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, n_q_heads, 1),
                    threads_per_threadgroup: (1024, 1, 1),
                    // AttentionViaCache (decode) — bucket_m == 1 here
                    // (decode bucket). Scaling is a no-op but kept
                    // for uniformity in case decode shares a bucket.
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    }),
                },
                bindings: super::kernel_bindings::AttentionViaCacheBindingSet {
                    output: super::ids::ArenaSlotIdx(*out_slot),
                    q: super::ids::ArenaSlotIdx(*q_slot),
                    kv_layer: super::ids::LayerId(*layer + layer_offset),
                    rope_on_read: w.rotary(ror_bind)?,
                }
                .into_baked(),
                gemm_dims: None,
            }
        }

        // ── Prefill-bucket attention reading from the paged KV cache ─
        // The paged variant of `attention_prefill_sdpa_v2_*`. The
        // upstream `RopeAppend` wrote rotated K + raw V into the
        // per-layer paged cache; this kernel reads them through
        // `block_table` indirection. K-axis covers the FULL
        // `seqused_k[seq]` (prefix + new tokens), so the kernel
        // serves chunked-prefill / prefix-cache-hit / multi-turn
        // scenarios that the contiguous prefill kernel cannot
        // (its K-axis = `cu_seqlens_q`, new tokens only).
        //
        // Bindings match the kernel's `set_buffer(i, …)` order:
        // 0 output, 1 Q, 2 cu_seqlens_q, 3 seq_used_k,
        // 4 block_table, 5 K cache (per-layer), 6 V cache (per-layer).
        // Dispatch matches `AttentionPrefillSdpa` (1 Q per
        // threadgroup, head on grid X, Q on grid Y, 1024 threads).
        I::AttentionPrefillPaged(Slot(q_slot), Slot(out_slot), LayerId(layer), _pairing) => {
            // ── hd512 UNFUSED attention (gemma4 GLOBAL class, head_dim 512) ──
            // The fused gqa_shared kernel is O(T²) and slow for head_dim 512
            // (it dominated 30k prefill). Replace it with mlx-style UNFUSED
            // attention in bf16: q_convert → gather K/V → per-head
            // (QKᵀ GEMM → causal softmax → PV GEMM) → o_convert. The two GEMMs
            // run on the NAX matrix accelerator (M5+) or the simdgroup steel
            // GEMM (pre-M5), chosen by `is_nax` — both far beat gqa_shared here.
            // Typed dispatch: no env var, on by default for the hd512 class.
            // 🛑 PR #79's hd512 unfused global attention is CORRECT for a single
            // contiguous prefill but emits GARBAGE on a chunked-prefill
            // CONTINUATION (num_computed_tokens>0 → lq != kv_len; the single-chunk
            // case lq == kv_len is what masked the bug). Bisect-confirmed:
            // gemma-4-12b at 3k is coherent at 2f6bc04d (global on gqa_shared) and
            // emits token-0/empty at be0079d3 (this unfused path). AttentionPrefillPaged
            // is the continuation path, so route hd512 global here to the correct
            // gqa_shared/SDPA paged path below (still present after this branch).
            // Re-enable once the lq!=kv_len defect in the unfused gather/QKᵀ/PV is
            // fixed (dump Kdense/scores to localize).
            // 🛑 COMPILE-TIME LOCK — flipping this to `true` FAILS THE BUILD.
            // Re-enabling the hd512 unfused path for the continuation arm is a
            // regression: it emits garbage on a chunked-prefill continuation
            // (lq != kv_len) → gemma-4 empty output past one 2048-token chunk.
            // To re-enable you MUST first fix the lq!=kv_len handling in the
            // unfused gather/QKᵀ/PV kernels AND re-validate gemma-4-12b coherence
            // at >2048 tokens, then update this lock with evidence.
            // Re-enabled: both bugs that broke continuations are fixed — (1) the steel
            // binding-order swap below, and (2) the (M,K,N) descriptor-order bug in
            // gemm_t_nax_impl that silently halved the NAX contraction (metal_nax.h).
            // Both are covered by tests/unfused_attn_compose_test.rs.
            const HD512_UNFUSED_CONTINUATION_OK: bool = true;
            const _: () = assert!(
                HD512_UNFUSED_CONTINUATION_OK,
                "hd512 unfused global attention is BROKEN on chunked-prefill \
                 continuation (lq != kv_len -> garbage logits -> gemma-4 emits \
                 token-0/empty past one 2048-token chunk). Do NOT re-enable for \
                 AttentionPrefillPaged without fixing the unfused gather/QKt/PV \
                 lq!=kv_len handling and re-validating gemma-4-12b at >2048 tokens. \
                 Bisect: 2f6bc04d good, be0079d3 broken."
            );
            // The unfused kernels read sequence 0 only (`seq_used[0]`,
            // `cu_seqlens_q[1] - cu_seqlens_q[0]`): they serve single-sequence
            // steps, and the paged attention below the rest.
            #[allow(clippy::overly_complex_bool_expr)]
            let hd512_unfused =
                HD512_UNFUSED_CONTINUATION_OK && p.global_head_dim > 256 && p.rope_on_read;
            let unfused = if hd512_unfused {
                use crate::specialized_pipeline_cache::ConstantValue as CV;
                let is_nax = profile.is_some_and(|p| crate::targets::is_nax_capable(p.generation));
                let (rd, po, _on, _bind) = rope_on_read_params(p, true);
                let nh = p.num_q_heads;
                let nkv = p.num_global_kv_heads.max(1);
                let gqa = nh / nkv;
                let hd = p.global_head_dim;
                let bs = p.global_block_size;
                let lq = bucket_m;
                let max_kv = block_cap * bs;
                let lid = super::ids::LayerId(*layer + layer_offset);
                let rot = rd.map(|r| r.get()).unwrap_or(hd);
                let pair = po.map(|p| p.get()).unwrap_or(0);
                let e: u32 = 2; // bf16
                let al = |x: u32| (x + 255) & !255u32;
                let q_head_off = 0u32;
                let kdense_off = al(q_head_off + nh * lq * hd * e);
                let vdense_off = al(kdense_off + nkv * max_kv * hd * e);
                let scores_off = al(vdense_off + nkv * hd * max_kv * e);
                let out_head_off = al(scores_off + lq * max_kv * e);
                let total = al(out_head_off + nh * lq * hd * e);
                *attn_unfused_scratch_bytes = (*attn_unfused_scratch_bytes).max(total);

                let scr = |off: u32, bi: u8| Binding::AttnUnfusedScratch {
                    offset: off,
                    binding_index: bi,
                };
                let rt = |kind, bi: u8| Binding::Runtime {
                    kind,
                    binding_index: bi,
                };
                let table = w.table(true)?;
                let cossin = |bi: u8| source(table, WeightTensor::Weight, lid, bi);
                let tg1 = |g: (u32, u32, u32)| DispatchShape {
                    threadgroups: (g.0.div_ceil(64), g.1, g.2),
                    threads_per_threadgroup: (64, 1, 1),
                    m_scaling: None,
                };
                let tg_gemm = |n: u32, m: u32| DispatchShape {
                    threadgroups: (n.div_ceil(64), m.div_ceil(64), 1),
                    threads_per_threadgroup: (32, 2, 2),
                    m_scaling: None,
                };
                // Steel simdgroup GEMM (pre-M5): blocked 32×32 output tile, 4
                // simdgroups (128 threads) per threadgroup.
                let tg_gemm_steel = |n: u32, m: u32| DispatchShape {
                    threadgroups: (n.div_ceil(32), m.div_ceil(32), 1),
                    threads_per_threadgroup: (128, 1, 1),
                    m_scaling: None,
                };

                let mut cmds: Vec<LoweredCommand> = Vec::new();
                // 1. Q [Lq,nh,hd] token-major → [nh,Lq,hd] head-major (bf16).
                cmds.push(LoweredCommand {
                    kernel: KernelId::AttnQConvert,
                    library: "attention_layout_convert",
                    function: "q_convert_prod_bf16",
                    constants: baked(vec![CV::uint(0, lq), CV::uint(1, nh), CV::uint(2, hd)]),
                    dispatch: tg1((lq, nh, 1)),
                    bindings: baked(vec![
                        scr(q_head_off, 0),
                        Binding::ArenaSlot {
                            slot: *q_slot,
                            binding_index: 1,
                        },
                    ]),
                    gemm_dims: None,
                });
                // 2. gather K (rope-on-read) → Kdense [nkv,max_kv,hd].
                cmds.push(LoweredCommand {
                    kernel: KernelId::AttnGatherKRope,
                    library: "attention_dense_gather",
                    function: "gather_dense_kvmajor_rope_bf16",
                    constants: baked(vec![
                        CV::uint(0, hd),
                        CV::uint(1, nkv),
                        CV::uint(2, bs),
                        CV::uint(3, crate::BLOCKS_PER_CHUNK),
                        CV::uint(4, rot),
                        CV::uint(5, pair),
                        CV::uint(6, max_kv),
                    ]),
                    dispatch: tg1((max_kv, nkv, 1)),
                    bindings: baked(vec![
                        scr(kdense_off, 0),
                        rt(RuntimeBindingKind::BlockTable { layer: lid }, 1),
                        rt(RuntimeBindingKind::KvCacheK { layer: lid }, 2),
                        rt(RuntimeBindingKind::SeqUsedK, 3),
                        cossin(4),
                    ]),
                    gemm_dims: None,
                });
                // 3. gather V (transposed copy) → Vdense_T [nkv,hd,max_kv].
                cmds.push(LoweredCommand {
                    kernel: KernelId::AttnGatherVCopyT,
                    library: "attention_dense_gather",
                    function: "gather_dense_kvmajor_copyT_bf16",
                    constants: baked(vec![
                        CV::uint(0, hd),
                        CV::uint(1, nkv),
                        CV::uint(2, bs),
                        CV::uint(3, crate::BLOCKS_PER_CHUNK),
                        CV::uint(6, max_kv),
                    ]),
                    dispatch: tg1((max_kv, nkv, 1)),
                    bindings: baked(vec![
                        scr(vdense_off, 0),
                        rt(RuntimeBindingKind::BlockTable { layer: lid }, 1),
                        rt(RuntimeBindingKind::KvCacheV { layer: lid }, 2),
                        rt(RuntimeBindingKind::SeqUsedK, 3),
                    ]),
                    gemm_dims: None,
                });
                // 4. per head: QKᵀ (NAX GEMM, N=kv_len@seq_used) → softmax → PV.
                for h in 0..nh {
                    let kv = h / gqa;
                    // QKᵀ: NAX matmul2d (M5+) or simdgroup steel (pre-M5). Both
                    // C=A@B^T with N=kv_len read from seq_used at runtime.
                    let (qk_lib, qk_fn, qk_consts, qk_disp) = if is_nax {
                        (
                            "quantized_qmm_nax",
                            "gemm_nax_bf16_qk",
                            vec![
                                CV::int(0, hd as i32),     // QMM_K = hd
                                CV::int(1, max_kv as i32), // QMM_N grid max (live N from seq_used)
                                CV::int(2, lq as i32),     // QMM_M = Lq
                                // QK_SPAN_BLOCK_NAX = span_ids block size (== the
                                // gather's `bs` = p.global_block_size), drives the
                                // block-diagonal bound. MUST match span_ids layout.
                                CV::uint(3, bs),
                            ],
                            tg_gemm(max_kv, lq),
                        )
                    } else {
                        (
                            "gemm",
                            "gemm_bf16_qk",
                            // GEMM_M=Lq, GEMM_K=hd; QK_SPAN_BLOCK = span_ids block
                            // size (== the gather's `bs` = p.global_block_size).
                            vec![CV::uint(0, lq), CV::uint(2, hd), CV::uint(3, bs)],
                            tg_gemm_steel(max_kv, lq),
                        )
                    };
                    // The NAX (gemm_nax_bf16_qk) and steel (gemm_bf16_qk) kernels use
                    // OPPOSITE buffer conventions: NAX = [input(Q)@0, weight(K)@1,
                    // output(scores)@2]; steel = [output(scores)@0, input(Q)@1,
                    // weight(K)@2]. Bind per the kernel actually dispatched, else the
                    // steel path writes scores into Q's slot and never fills out_head
                    // → o_convert reads zero → garbage on every prefill.
                    let q_b = scr(q_head_off + h * lq * hd * e, if is_nax { 0 } else { 1 });
                    let k_b = scr(
                        kdense_off + kv * max_kv * hd * e,
                        if is_nax { 1 } else { 2 },
                    );
                    let s_b = scr(scores_off, if is_nax { 2 } else { 0 });
                    cmds.push(LoweredCommand {
                        kernel: KernelId::AttnGemmQk,
                        library: qk_lib,
                        function: qk_fn,
                        constants: baked(qk_consts),
                        dispatch: qk_disp,
                        bindings: baked(vec![
                            q_b,
                            k_b,
                            s_b,
                            rt(RuntimeBindingKind::SeqUsedK, 3),
                            rt(RuntimeBindingKind::SpanIds, 4),
                            rt(RuntimeBindingKind::CuSeqlensQ, 5),
                        ]),
                        gemm_dims: None,
                    });
                    cmds.push(LoweredCommand {
                        kernel: KernelId::AttnCausalSoftmax,
                        library: "attention_causal_softmax",
                        function: "causal_softmax_prod_bf16",
                        // SOFT_SPAN_BLOCK = span_ids block size (gather's `bs`).
                        constants: baked(vec![CV::float(1, p.attn_scale), CV::uint(3, bs)]),
                        dispatch: tg1((lq, 1, 1)),
                        bindings: baked(vec![
                            scr(scores_off, 0),
                            rt(RuntimeBindingKind::SeqUsedK, 1),
                            rt(RuntimeBindingKind::CuSeqlensQ, 2),
                            rt(RuntimeBindingKind::SpanIds, 3),
                        ]),
                        gemm_dims: None,
                    });
                    // PV: NAX matmul2d (M5+) or simdgroup steel (pre-M5). Both
                    // C=A@B^T with K=kv_len read from seq_used at runtime.
                    let (pv_lib, pv_fn, pv_consts, pv_disp) = if is_nax {
                        (
                            "quantized_qmm_nax",
                            "gemm_nax_bf16_pv",
                            vec![
                                CV::int(0, max_kv as i32), // QMM_K unused (live K from seq_used)
                                CV::int(1, hd as i32),     // QMM_N = hd
                                CV::int(2, lq as i32),     // QMM_M = Lq
                                // QK_SPAN_BLOCK_NAX: bounds the PV contraction to
                                // the query-tile's span (== gather's `bs`).
                                CV::uint(3, bs),
                            ],
                            tg_gemm(hd, lq),
                        )
                    } else {
                        (
                            "gemm",
                            "gemm_bf16_pv",
                            // GEMM_M=Lq, GEMM_N=hd, GEMM_K=max_kv (V^T row stride;
                            // contraction kv_len read from seq_used).
                            vec![CV::uint(0, lq), CV::uint(1, hd), CV::uint(2, max_kv)],
                            tg_gemm_steel(hd, lq),
                        )
                    };
                    // Same NAX-vs-steel buffer-order split as QKᵀ: NAX =
                    // [input(probs)@0, weight(V)@1, output(out_head)@2]; steel =
                    // [output(out_head)@0, input(probs)@1, weight(V)@2].
                    let p_b = scr(scores_off, if is_nax { 0 } else { 1 });
                    let v_b = scr(
                        vdense_off + kv * hd * max_kv * e,
                        if is_nax { 1 } else { 2 },
                    );
                    let o_b = scr(out_head_off + h * lq * hd * e, if is_nax { 2 } else { 0 });
                    cmds.push(LoweredCommand {
                        kernel: KernelId::AttnGemmPv,
                        library: pv_lib,
                        function: pv_fn,
                        constants: baked(pv_consts),
                        dispatch: pv_disp,
                        bindings: baked(vec![
                            p_b,
                            v_b,
                            o_b,
                            rt(RuntimeBindingKind::SeqUsedK, 3),
                            rt(RuntimeBindingKind::SpanIds, 4),
                            rt(RuntimeBindingKind::CuSeqlensQ, 5),
                        ]),
                        gemm_dims: None,
                    });
                }
                // 5. O [nh,Lq,hd] head-major → [Lq,nh,hd] token-major into out_slot.
                cmds.push(LoweredCommand {
                    kernel: KernelId::AttnOConvert,
                    library: "attention_layout_convert",
                    function: "o_convert_prod_bf16",
                    constants: baked(vec![CV::uint(0, lq), CV::uint(1, nh), CV::uint(2, hd)]),
                    dispatch: tg1((lq, nh, 1)),
                    bindings: baked(vec![
                        Binding::ArenaSlot {
                            slot: *out_slot,
                            binding_index: 0,
                        },
                        scr(out_head_off, 1),
                    ]),
                    gemm_dims: None,
                });
                Some(cmds)
            } else {
                None
            };
            // Steel-attention paged kernel — MLX FA-2 algorithm with
            // simdgroup_matrix MMAs (BQ=32, BK=16, BD=128, WM=4). Wins
            // big over the sdpa_vector port for prefill, but at small
            // bucket_m a BQ=32 tile wastes most of its work, so we
            // default-route to SDPA there and steel for bucket_m≥32.
            //
            // Earlier comment claimed a "bisected coherence regression
            // (df84c658d)" — verified false: steel and SDPA emit
            // identical output at every M tested (incl. 2..64 + the
            // BQ=32 / BQ+1 boundary). The "garbage" cited in the bisect
            // was Llama-3.2-3B-Instruct degenerating on bare /v1/
            // completions prompts; same behavior on both kernels and
            // on mlx_lm.server.
            //
            // The four (kernel, dtype) combinations are typed ZSTs in
            // `super::kernel_identity`; routing through `for_kernel<K>`
            // means the (library, function, KERNEL_ID) trio comes from
            // one source. Bug class #8 — drift between the three
            // independent `&'static str` fields — can't recur.
            use crate::steel_paged::{nax_paged_symbol, steel_paged_symbol};
            // Spans rope-on-read (GLOBAL class). All-None when !ROPE_ON_READ.
            let (ror_rd, ror_po, ror_on, ror_bind) = rope_on_read_params(p, true);
            const BQ_STEEL: u32 = 32;
            // NAX kernel tiles queries in BQ=64 blocks (4 warps × 16-row
            // NAX Q-frags), vs the simdgroup steel kernel's BQ=32.
            const BQ_NAX: u32 = 64;
            // Steel attention paged needs an instantiation in
            // `attention_steel_paged.metal` for the model's HEAD_DIM
            // (BD template arg). The instantiation list is owned by
            // `scratchy-target-metal/build.rs::STEEL_PAGED_HEAD_DIMS`
            // and exposed here through `steel_paged_symbol()` —
            // `Some(symbol)` means the (dtype, head_dim) combo is
            // built; `None` means we must fall through to SDPA.
            //
            // Routing a HEAD_DIM that's NOT instantiated through
            // steel produces silently-wrong logits (MSL template-
            // instance lookup fails or, worse, links to the wrong
            // `_bd<X>_` symbol — verified on Llama-3.2-1B, HEAD_DIM=64,
            // before the lookup-driven gate landed).
            let steel_dtype_tag: &str = match p.metal_dtype {
                crate::tape::lowered::MetalDtype::Bf16 => "bf16",
                _ => "f16",
            };
            // NAX matrix-accelerator paged attention (M5+/A19+ only —
            // `is_nax_capable` gates on arch gen ≥ 17). The NAX kernel
            // (`attention_steel_nax_paged`, BQ64/BK32/BD128) drives the
            // Apple matrix accelerator via MPP `matmul2d` and runs ~3.56×
            // the simdgroup steel kernel on the Llama-3B prefill shape
            // (11.76 vs 3.3 TFLOP/s). Only instantiated for head_dim 128,
            // so it serves Llama-3.x (hd 128) prefill; everything else
            // falls through to the simdgroup steel path below.
            let is_nax = profile.is_some_and(|p| crate::targets::is_nax_capable(p.generation));
            let nax_symbol = if is_nax {
                nax_paged_symbol(steel_dtype_tag, p.global_head_dim)
            } else {
                None
            };
            // Class head_dim: 512 has no steel instantiation, so
            // Gemma4 global prefill auto-falls-back to SDPA-paged.
            let steel_symbol = steel_paged_symbol(steel_dtype_tag, p.global_head_dim);
            // Chunked-prefill long-context correctness (the launch-claude bug):
            // the steel/NAX/gqa_shared paged prefill reads pre-roped K from the
            // rope-once-to-scratch buffer (`p.rope_on_read` is the universal
            // default). That scratch + the `RopeOnce*` grid MUST cover the WHOLE
            // sequence a continuation chunk attends (computed prefix + new), not
            // just one prefill bucket — else the attention reads past the scratch
            // for prompts > one bucket (~4096 tok) → garbage K → `!!!!`. FIXED by
            // sizing the scratch to `block_cap` blocks (rope-once command
            // construction below) and overriding the `RopeOnce*` grid to the live
            // block-table width in the worker (worker.rs, the `tq_dequant_max_blocks`
            // pattern). Steel stays ON (the fast kernel).
            let use_steel =
                (steel_symbol.is_some() || nax_symbol.is_some()) && bucket_m >= BQ_STEEL;
            // Prefer the NAX kernel when its symbol is present AND steel is
            // selected. Its grid uses BQ=64 (vs steel's BQ=32); both kernels
            // share the same bindings/constants and a per-(BQ-block, q_head)
            // grid with seq on Z.
            let use_nax = use_steel && nax_symbol.is_some();
            let bq_steel = if use_nax { BQ_NAX } else { BQ_STEEL };
            // GQA-cooperative fallback selection (see the longer comment at the
            // dispatch site below). Computed early so `constants.k_scratch` (slot
            // 11) can be set when the gqa_shared kernel reads pre-roped K from
            // the rope-once scratch. head_dim 512 (gemma4 global) has no steel
            // instantiation → use_steel is false → this path is taken.
            let gqa = p.num_q_heads / p.num_global_kv_heads.max(1);
            let use_gqa_shared = !use_steel
                && (8..=32).contains(&gqa)
                && p.global_head_dim.is_multiple_of(32)
                && p.global_head_dim <= 512
                && p.global_block_size <= 64;
            // Spans rope-once-to-scratch on the gqa_shared path: K is roped ONCE
            // into the shared scratch by a preceding RopeOnceGqaShared command,
            // and the attention reads pre-roped K (slot 7 = scratch, ATTN_K_SCRATCH
            // set) with no per-tile smem rotation. The gqa_shared twin of
            // nax_spans/steel_spans below.
            // No rope-once pair where the unfused attention serves
            // single-sequence steps.
            let rope_once = p.rope_on_read && unfused.is_none();
            let gqa_shared_spans = use_gqa_shared && rope_once;
            let constants = super::kernel_constants::AttentionPrefillPagedConstants {
                // Paged prefill attends the whole cached sequence on a
                // continuation chunk → FullSeqUsed. Resolved once here; the
                // witness is a required field so no arm can skip it.
                geom: super::continuation_witness::KvGeometry::resolve(
                    super::continuation_witness::KvAxis::FullSeqUsed,
                ),
                // GLOBAL class on hybrid arches; identity on uniform
                // models (see the decode arm note).
                head_dim: super::ids::HeadDim(p.global_head_dim),
                num_q_heads: super::ids::NumQHeads(p.num_q_heads),
                num_kv_heads: super::ids::NumKvHeads(p.num_global_kv_heads),
                attn_scale: super::ids::AttnScale(p.attn_scale),
                // GLOBAL class block size (page-unified; Gemma4: 32).
                block_size: super::ids::BlockSize(p.global_block_size),
                max_blocks: super::ids::MaxBlocksPerSeq(block_cap),
                // Prefill kernels (steel + sdpa paged) stay at the standard
                // BPC; the steel loader (paged_loader.h) has its own chunk
                // arithmetic that hasn't been adapted to the BPC=0 fast
                // path. Decode reader (AttentionViaCache) is the only kernel
                // currently consulting `attention_blocks_per_chunk(chunked)`.
                blocks_per_chunk: super::ids::BlocksPerChunk(crate::BLOCKS_PER_CHUNK),
                // Full attention: window disabled (both kernels read
                // slot 7; 0 folds every window branch away).
                window: super::ids::AttnWindow(0),
                // Steel kernel reads slot 99; omitting it leaves Metal
                // undefined and the kernel can hit a diagnostic path
                // (the b3ddb3b46 regression). sdpa_vector ignores it.
                debug_mode: if use_steel {
                    Some(super::ids::AttnDebugMode(0))
                } else {
                    None
                },
                rot_dim: ror_rd,
                pair_off: ror_po,
                rope_on_read: ror_on,
                // Slot 11: gqa_shared reads pre-roped K from the scratch.
                k_scratch: if gqa_shared_spans { Some(1) } else { None },
                // This arm runs the span seek via ATTN_PAGED_ROR (ror_on above);
                // the decoupled self-only gate is only needed by the sliding
                // arm, which runs with ROR off.
                self_only: None,
            };
            // Spans (rope-once-to-scratch): when a steel-family kernel (NAX
            // matrix-accel OR simdgroup steel) OR the GQA-cooperative shared
            // kernel is selected AND rope-on-read is active, K is roped ONCE
            // into the shared scratch by a preceding RopeOnce{Nax,Steel,
            // GqaShared} command, and the attention reads pre-roped K from the
            // scratch at slot 7 (no per-tile rotation, no cos_sin in the
            // attention). The remaining sdpa-paged path keeps the in-kernel
            // cos_sin path.
            //
            // The scratch holds ONE sequence's keys (it has no batch
            // dimension: at `num_pages = block_cap` one sequence is already
            // ~268 MB), so the rope-once pair runs only on single-sequence
            // steps (`OnlyIfOneSequence`). A step with several sequences runs
            // a per-row twin instead (`UnlessOneSequence`): the same attention
            // reading K from the cache through each sequence's own block-table
            // row and re-roping flagged (bit-31) blocks in-kernel — sdpa-paged
            // for steel/NAX, gqa_shared without the scratch for gqa_shared.
            // `route_by_sequence_count` attaches the gates.
            let nax_spans = use_nax && rope_once;
            let steel_spans = use_steel && !use_nax && rope_once;
            // NAX, simdgroup steel, and gqa_shared all read pre-roped K from the
            // shared `Binding::RopedKScratch` at slot 7 (the rope-once-to-scratch
            // pattern); they share the scratch-source binding flag.
            let roped_k_scratch = nax_spans || steel_spans || gqa_shared_spans;
            // The attention's bindings: K from the rope-once scratch, or from
            // the cache, with cos_sin to re-rope unrotated span blocks when
            // `reropes`.
            let rotary = w.rotary(ror_bind)?;
            let bindings_for = |scratch: bool, reropes: bool| {
                super::kernel_bindings::AttentionPrefillPagedBindingSet {
                    output: super::ids::ArenaSlotIdx(*out_slot),
                    q: super::ids::ArenaSlotIdx(*q_slot),
                    kv_layer: super::ids::LayerId(*layer + layer_offset),
                    rope_on_read: if reropes { rotary } else { None },
                    nax_roped_k_scratch: scratch,
                }
            };
            let bindings = bindings_for(roped_k_scratch, !roped_k_scratch);
            // The per-row twins' constants, neither reading the scratch: the
            // plain twin reads the cache's roped K as is (rope-on-read off);
            // the re-roping twin is sdpa-paged or gqa_shared (no steel debug
            // slot), re-roping span blocks in-kernel.
            let plain_constants = super::kernel_constants::AttentionPrefillPagedConstants {
                rope_on_read: None,
                k_scratch: None,
                ..constants
            };
            let reroping_constants = super::kernel_constants::AttentionPrefillPagedConstants {
                debug_mode: None,
                k_scratch: None,
                ..constants
            };
            let sdpa_paged =
                |constants, bindings| sdpa_paged_command(p, constants, bindings, bucket_m);
            // GQA-cooperative fallback: when steel can't take the shape (head_dim
            // 512 has no steel instantiation — TG memory) AND the GQA ratio is
            // high, the per-(q_head, query) sdpa_vector kernel re-streams
            // identical K/V `gqa`× from device. The gqa_shared kernel stages each
            // paged K/V block through threadgroup memory once per query and fans
            // it out to all heads (one simdgroup per head). Gemma4 global layers
            // (512 hd, 16:1) went from ~350 ms/layer to bandwidth-proportional on
            // T=2930. Gated to gqa >= 8 so low-GQA arches keep the proven
            // sdpa_vector path. (`gqa` / `use_gqa_shared` / `gqa_shared_spans`
            // were computed above so `constants.k_scratch` could be set.)
            let attention = if use_steel {
                // Symbol came from the codegen'd table above
                // (`steel_symbol.is_some()` is the gate). Build the
                // command directly instead of going through
                // `for_kernel::<K>` — there's no typed ZST for steel
                // because BD lives in the symbol name; see the
                // comment in `kernel_identity.rs`.
                //
                // NAX (matrix-accelerator) wins when its symbol is present
                // (M5+, head_dim 128); it shares the simdgroup steel
                // kernel's bindings/constants and only swaps the
                // library/function pair. Falls back to the simdgroup
                // `attention_steel_paged` symbol otherwise.
                let (library, function) = if use_nax {
                    (
                        "attention_steel_nax_paged",
                        nax_symbol.expect("nax_symbol is Some when use_nax is true"),
                    )
                } else {
                    (
                        "attention_steel_paged",
                        steel_symbol.expect("steel_symbol is Some when use_steel is true"),
                    )
                };
                let attn_cmd = LoweredCommand {
                    kernel: KernelId::AttentionPrefillSdpaPaged,
                    library,
                    function,
                    constants: constants.into_baked(),
                    dispatch: steel_paged_dispatch(p, bucket_m, bq_steel),
                    bindings: bindings.into_baked(),
                    gemm_dims: None,
                };
                if roped_k_scratch {
                    // Three commands: (1) RopeOnce{Nax,Steel} ropes the cache's K
                    // into the shared scratch (sized per-layer below); (2) the
                    // steel/NAX attention reads pre-roped K from the scratch;
                    // (3) its sdpa-paged twin for steps with several sequences.
                    // Pick the rope-once kernel matching the selected attention
                    // kernel: NAX (hd128 only) → `rope_once_nax`; simdgroup steel
                    // (hd 64/96/128/256, incl. SmolLM hd64) → `rope_once_steel`.
                    let (rope_kernel, rope_library, rope_sym) = if use_nax {
                        (
                            KernelId::RopeOnceNax,
                            "attention_steel_nax_paged",
                            crate::steel_paged::rope_once_nax_symbol(
                                steel_dtype_tag,
                                p.global_head_dim,
                            )
                            .expect("rope_once_nax_symbol is Some when use_nax is true (hd128)"),
                        )
                    } else {
                        (
                            KernelId::RopeOnceSteel,
                            "attention_steel_paged",
                            crate::steel_paged::rope_once_steel_symbol(
                                steel_dtype_tag,
                                p.global_head_dim,
                            )
                            .expect(
                                "rope_once_steel_symbol is Some when use_steel is true \
                                 (steel head_dim instantiated)",
                            ),
                        )
                    };
                    let elem_bytes = match p.metal_dtype {
                        crate::tape::lowered::MetalDtype::Bf16 => 2u32,
                        _ => 2u32,
                    };
                    // Size the rope-once scratch + grid to the PER-SEQUENCE block
                    // capacity (block_cap), not one prefill bucket: a
                    // chunked-prefill continuation attends the whole sequence, so
                    // the scratch must hold every logical block the attention reads.
                    // The worker caps the live grid.y to this value.
                    let num_pages = block_cap.max(bucket_m.div_ceil(p.global_block_size));
                    let scratch_bytes = num_pages
                        .saturating_mul(p.num_global_kv_heads)
                        .saturating_mul(p.global_block_size)
                        .saturating_mul(p.global_head_dim)
                        .saturating_mul(elem_bytes);
                    *roped_k_scratch_bytes = (*roped_k_scratch_bytes).max(scratch_bytes);
                    // Grid: x = num_kv_heads * BLOCK_SIZE * (rot_dim/2),
                    // y = num_pages (one logical block per row).
                    let rot_half = ror_rd.map(|r| r.get() / 2).unwrap_or(0).max(1);
                    let rope_threads = p.num_global_kv_heads * p.global_block_size * rot_half;
                    let rope_cmd = LoweredCommand {
                        kernel: rope_kernel,
                        library: rope_library,
                        function: rope_sym,
                        constants: constants.into_baked(),
                        dispatch: DispatchShape {
                            threadgroups: (rope_threads.div_ceil(64), num_pages, 1),
                            threads_per_threadgroup: (64, 1, 1),
                            // num_pages scales with the live num_tokens.
                            m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                                seq_axis: None,
                                axis: crate::tape::lowered::MScaleAxis::Y,
                                bucket_m: super::ids::BucketM(num_pages),
                            }),
                        },
                        bindings: super::kernel_bindings::RopeOnceNaxBindingSet {
                            kv_layer: super::ids::LayerId(*layer + layer_offset),
                            table: w.table(true)?,
                        }
                        .into_baked(),
                        gemm_dims: None,
                    };
                    let plain = LoweredCommand {
                        constants: plain_constants.into_baked(),
                        bindings: bindings_for(false, false).into_baked(),
                        ..attn_cmd
                    };
                    let reroping = sdpa_paged(reroping_constants, bindings_for(false, true));
                    return Ok(vec![rope_cmd, attn_cmd, plain, reroping]);
                }
                attn_cmd
            } else if use_gqa_shared {
                let gqa_shared = |constants: super::kernel_constants::AttentionPrefillPagedConstants,
                                  bindings: super::kernel_bindings::AttentionPrefillPagedBindingSet| {
                    LoweredCommand {
                        kernel: KernelId::AttentionPrefillSdpaPaged,
                        library: "attention",
                        function: pick_specialized_symbol(
                            "attention_prefill_sdpa_gqa_shared_f16_specialized",
                            "attention_prefill_sdpa_gqa_shared_bf16_specialized",
                            p.metal_dtype,
                        ),
                        constants: constants.into_baked(),
                        dispatch: DispatchShape {
                            // One TG per (kv_head, query); `32 × gqa`
                            // threads = one simdgroup per q-head (gqa <=
                            // 32 keeps this within the 1024-thread cap).
                            threadgroups: (p.num_global_kv_heads, bucket_m, 1),
                            threads_per_threadgroup: (32 * gqa, 1, 1),
                            m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                                seq_axis: None,
                                axis: crate::tape::lowered::MScaleAxis::Y,
                                bucket_m: super::ids::BucketM(bucket_m),
                            }),
                        },
                        bindings: bindings.into_baked(),
                        gemm_dims: None,
                    }
                };
                let attn_cmd = gqa_shared(constants, bindings);
                if gqa_shared_spans {
                    // Three commands: (1) RopeOnceGqaShared ropes the cache's K
                    // into the shared scratch (dense, logical-block indexed);
                    // (2) the gqa_shared attention reads pre-roped K from the
                    // scratch (slot 7, ATTN_K_SCRATCH set) with no per-tile smem
                    // rotation; (3) its twin without the scratch, re-roping K
                    // in smem, for steps with several sequences. head_dim 512
                    // (gemma4 global) has no steel/NAX instantiation, so this is
                    // the path launch-claude takes.
                    let rope_sym = crate::steel_paged::rope_once_gqa_shared_symbol(steel_dtype_tag)
                        .expect("rope_once_gqa_shared_symbol is Some for f16/bf16");
                    // f16 and bf16 are both 2 B/elem.
                    let elem_bytes = 2u32;
                    // Size the rope-once scratch + grid to the PER-SEQUENCE block
                    // capacity (block_cap), not one prefill bucket: a
                    // chunked-prefill continuation attends the whole sequence, so
                    // the scratch must hold every logical block the attention reads.
                    // The worker caps the live grid.y to this value.
                    let num_pages = block_cap.max(bucket_m.div_ceil(p.global_block_size));
                    let scratch_bytes = num_pages
                        .saturating_mul(p.num_global_kv_heads)
                        .saturating_mul(p.global_block_size)
                        .saturating_mul(p.global_head_dim)
                        .saturating_mul(elem_bytes);
                    *roped_k_scratch_bytes = (*roped_k_scratch_bytes).max(scratch_bytes);
                    // Grid: x = num_kv_heads * BLOCK_SIZE * (rot_dim/2),
                    // y = num_pages (one logical block per row). Same decode as
                    // the rope_once_gqa_shared kernel's gid.x.
                    let rot_half = ror_rd.map(|r| r.get() / 2).unwrap_or(0).max(1);
                    let rope_threads = p.num_global_kv_heads * p.global_block_size * rot_half;
                    let rope_cmd = LoweredCommand {
                        kernel: KernelId::RopeOnceGqaShared,
                        library: "attention",
                        function: rope_sym,
                        constants: constants.into_baked(),
                        dispatch: DispatchShape {
                            threadgroups: (rope_threads.div_ceil(64), num_pages, 1),
                            threads_per_threadgroup: (64, 1, 1),
                            m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                                seq_axis: None,
                                axis: crate::tape::lowered::MScaleAxis::Y,
                                bucket_m: super::ids::BucketM(num_pages),
                            }),
                        },
                        // Same 5 bindings as RopeOnceNax (scratch out, block
                        // table, k_cache, seq_used_k, class-resolved cos_sin).
                        bindings: super::kernel_bindings::RopeOnceNaxBindingSet {
                            kv_layer: super::ids::LayerId(*layer + layer_offset),
                            table: w.table(true)?,
                        }
                        .into_baked(),
                        gemm_dims: None,
                    };
                    let plain = gqa_shared(plain_constants, bindings_for(false, false));
                    let reroping = gqa_shared(reroping_constants, bindings_for(false, true));
                    return Ok(vec![rope_cmd, attn_cmd, plain, reroping]);
                }
                attn_cmd
            } else {
                sdpa_paged(constants, bindings)
            };
            match unfused {
                Some(mut cmds) => {
                    cmds.push(attention);
                    return Ok(cmds);
                }
                None => attention,
            }
        }

        // ── Sliding-window decode attention (Gemma2/3/4 local layers) ─
        // Identical to `AttentionViaCache` except the kernel's
        // `ATTN_WINDOW` function constant carries `p.sliding_window`
        // (the decode kernel skips keys with `kv_len-1 - k >= window`).
        I::SlidingAttentionViaCache(Slot(q_slot), Slot(out_slot), LayerId(layer), _pairing) => {
            debug_assert!(
                p.sliding_window > 0,
                "SlidingAttentionViaCache lowered with SLIDING_WINDOW <= 0"
            );
            // Spans rope-on-read (SLIDING class). All-None when !ROPE_ON_READ.
            let (ror_rd, ror_po, ror_on, ror_bind) = rope_on_read_params(p, false);
            let n_q_heads = p.num_q_heads;
            LoweredCommand {
                kernel: KernelId::AttentionViaCache,
                library: "attention",
                function: pick_specialized_symbol(
                    "attention_via_cache_v2_f16_specialized",
                    "attention_via_cache_v2_bf16_specialized",
                    p.metal_dtype,
                ),
                constants: super::kernel_constants::AttentionViaCacheConstants {
                    head_dim: super::ids::HeadDim(p.head_dim),
                    num_q_heads: super::ids::NumQHeads(p.num_q_heads),
                    num_kv_heads: super::ids::NumKvHeads(p.num_kv_heads),
                    attn_scale: super::ids::AttnScale(p.attn_scale),
                    block_size: super::ids::BlockSize(p.block_size),
                    max_blocks: super::ids::MaxBlocksPerSeq(block_cap),
                    blocks_per_chunk: super::ids::BlocksPerChunk(attention_blocks_per_chunk(
                        chunked,
                    )),
                    window: super::ids::AttnWindow(p.sliding_window),
                    rot_dim: ror_rd,
                    pair_off: ror_po,
                    rope_on_read: ror_on,
                    pair_coresident: pair_coresident_param(ror_rd, ror_po, ror_on, p.head_dim),
                }
                .into_baked(),
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, n_q_heads, 1),
                    threads_per_threadgroup: (1024, 1, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    }),
                },
                bindings: super::kernel_bindings::AttentionViaCacheBindingSet {
                    output: super::ids::ArenaSlotIdx(*out_slot),
                    q: super::ids::ArenaSlotIdx(*q_slot),
                    kv_layer: super::ids::LayerId(*layer + layer_offset),
                    rope_on_read: w.rotary(ror_bind)?,
                }
                .into_baked(),
                gemm_dims: None,
            }
        }

        // ── Sliding-window paged prefill (Gemma2/3/4 local layers) ──
        // Same steel-vs-SDPA routing as `AttentionPrefillPaged`, with
        // the BASE-class geometry (sliding head_dim/kv heads) and
        // `ATTN_WINDOW = p.sliding_window` (fn-const slot 7, read by
        // both kernels). The steel kernel additionally SKIPS K-tiles
        // entirely older than the window (`kb_start`), making windowed
        // prefill O(T·window) — the dominant Gemma-family TTFT lever
        // (sliding layers are 40 of Gemma4-12B's 48).
        I::SlidingAttentionPrefillPaged(Slot(q_slot), Slot(out_slot), LayerId(layer), _pairing) => {
            debug_assert!(
                p.sliding_window > 0,
                "SlidingAttentionPrefillPaged lowered with SLIDING_WINDOW <= 0"
            );
            use crate::steel_paged::steel_paged_symbol;
            // Spans rope-on-read (SLIDING class). All-None when !ROPE_ON_READ.
            let (ror_rd, ror_po, ror_on, ror_bind) = rope_on_read_params(p, false);
            const BQ_STEEL: u32 = 32;
            let steel_dtype_tag: &str = match p.metal_dtype {
                crate::tape::lowered::MetalDtype::Bf16 => "bf16",
                _ => "f16",
            };
            // BASE class head_dim (sliding layers) — Gemma4: 256,
            // which IS instantiated, so sliding prefill gets steel
            // while the 512-wide global class falls back to SDPA.
            let steel_symbol = steel_paged_symbol(steel_dtype_tag, p.head_dim);
            let use_steel = steel_symbol.is_some() && bucket_m >= BQ_STEEL;
            // Spans rope-once-to-scratch (SLIDING class): when the sliding
            // steel kernel (hd256, never NAX/gqa_shared) is selected AND
            // rope-on-read is active, K is roped ONCE into the shared scratch
            // by a preceding RopeOnceSteel command (SLIDING geometry +
            // SLIDING cos_sin), and the attention reads pre-roped K from the
            // scratch (slot 7) with no in-kernel rotation. Mirrors the
            // `steel_spans` path in the global AttentionPrefillPaged arm.
            let steel_spans = use_steel && p.rope_on_read;
            let constants = super::kernel_constants::AttentionPrefillPagedConstants {
                // Paged prefill attends the whole cached sequence on a
                // continuation chunk → FullSeqUsed. Resolved once here; the
                // witness is a required field so no arm can skip it.
                geom: super::continuation_witness::KvGeometry::resolve(
                    super::continuation_witness::KvAxis::FullSeqUsed,
                ),
                head_dim: super::ids::HeadDim(p.head_dim),
                num_q_heads: super::ids::NumQHeads(p.num_q_heads),
                num_kv_heads: super::ids::NumKvHeads(p.num_kv_heads),
                attn_scale: super::ids::AttnScale(p.attn_scale),
                block_size: super::ids::BlockSize(p.block_size),
                max_blocks: super::ids::MaxBlocksPerSeq(block_cap),
                blocks_per_chunk: super::ids::BlocksPerChunk(crate::BLOCKS_PER_CHUNK),
                window: super::ids::AttnWindow(p.sliding_window),
                // Steel reads slot 99 (the b3ddb3b46 lesson);
                // sdpa_vector declares no slot 99.
                debug_mode: if use_steel {
                    Some(super::ids::AttnDebugMode(0))
                } else {
                    None
                },
                rot_dim: ror_rd,
                pair_off: ror_po,
                // Steel spans reads pre-roped K from the scratch, so the
                // in-kernel rope is OFF; the non-spans path keeps ror_on.
                rope_on_read: if steel_spans { None } else { ror_on },
                // Sliding prefill is the steel/sdpa path, not gqa_shared;
                // steel reads its pre-roped K via ATTN_PAGED_ROR (slot 7
                // scratch), not the gqa_shared ATTN_K_SCRATCH (slot 11), so
                // slot 11 stays unset (matches the global steel_spans arm).
                k_scratch: None,
                // Self-only span masking, decoupled from rope-on-read: this arm
                // sets `rope_on_read: None` (the K-source is the plain cache),
                // which also gated off the kb-loop span seek — so a Relocatable
                // span attended the preamble at every sliding layer and its K/V
                // was NOT a pure function of its bytes, breaking the spans
                // design's content-addressed reuse contract. Gate the seek on
                // its own constant so span isolation holds on all 30 layers.
                self_only: if steel_spans { Some(1) } else { None },
            };
            // Steel spans reads pre-roped K from the scratch (slot 7), so it
            // does NOT bind cos_sin there; the non-spans path keeps the
            // in-kernel cos_sin binding. Sliding prefill uses the simdgroup
            // steel kernel (hd256), never NAX (hd128 only).
            let rotary = w.rotary(ror_bind)?;
            let bindings_for = |scratch: bool, reropes: bool| {
                super::kernel_bindings::AttentionPrefillPagedBindingSet {
                    output: super::ids::ArenaSlotIdx(*out_slot),
                    q: super::ids::ArenaSlotIdx(*q_slot),
                    kv_layer: super::ids::LayerId(*layer + layer_offset),
                    rope_on_read: if reropes { rotary } else { None },
                    nax_roped_k_scratch: scratch,
                }
            };
            let bindings = bindings_for(steel_spans, !steel_spans);
            if use_steel {
                let function = steel_symbol.expect("steel_symbol is Some when use_steel is true");
                let attn_cmd = LoweredCommand {
                    kernel: KernelId::AttentionPrefillSdpaPaged,
                    library: "attention_steel_paged",
                    function,
                    constants: constants.into_baked(),
                    dispatch: steel_paged_dispatch(p, bucket_m, BQ_STEEL),
                    bindings: bindings.into_baked(),
                    gemm_dims: None,
                };
                if steel_spans {
                    // Three commands: (1) RopeOnceSteel ropes the cache's K into
                    // the shared scratch using SLIDING geometry (HEAD_DIM 256,
                    // NUM_KV_HEADS, BLOCK_SIZE) + the SLIDING-class cos_sin
                    // (is_global: false — gemma4 uses a different rope theta for
                    // local vs global layers); (2) the steel attention reads
                    // pre-roped K from the scratch (slot 7); (3) its sdpa-paged
                    // twin, for steps with several sequences — the scratch holds
                    // one sequence's keys. Mirrors the global
                    // AttentionPrefillPaged steel_spans path.
                    let rope_sym =
                        crate::steel_paged::rope_once_steel_symbol(steel_dtype_tag, p.head_dim)
                            .expect(
                                "rope_once_steel_symbol is Some when sliding use_steel is true \
                         (steel head_dim instantiated)",
                            );
                    // f16 and bf16 are both 2 B/elem.
                    let elem_bytes = 2u32;
                    // Per-sequence block capacity (see the GLOBAL site above).
                    let num_pages = block_cap.max(bucket_m.div_ceil(p.block_size));
                    let scratch_bytes = num_pages
                        .saturating_mul(p.num_kv_heads)
                        .saturating_mul(p.block_size)
                        .saturating_mul(p.head_dim)
                        .saturating_mul(elem_bytes);
                    *roped_k_scratch_bytes = (*roped_k_scratch_bytes).max(scratch_bytes);
                    // Grid: x = num_kv_heads * BLOCK_SIZE * (rot_dim/2),
                    // y = num_pages (one logical block per row).
                    let rot_half = ror_rd.map(|r| r.get() / 2).unwrap_or(0).max(1);
                    let rope_threads = p.num_kv_heads * p.block_size * rot_half;
                    let rope_cmd = LoweredCommand {
                        kernel: KernelId::RopeOnceSteel,
                        library: "attention_steel_paged",
                        function: rope_sym,
                        constants: constants.into_baked(),
                        dispatch: DispatchShape {
                            threadgroups: (rope_threads.div_ceil(64), num_pages, 1),
                            threads_per_threadgroup: (64, 1, 1),
                            m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                                seq_axis: None,
                                axis: crate::tape::lowered::MScaleAxis::Y,
                                bucket_m: super::ids::BucketM(num_pages),
                            }),
                        },
                        // SLIDING-class cos_sin (is_global: false).
                        bindings: super::kernel_bindings::RopeOnceNaxBindingSet {
                            kv_layer: super::ids::LayerId(*layer + layer_offset),
                            table: w.table(false)?,
                        }
                        .into_baked(),
                        gemm_dims: None,
                    };
                    // The plain twin reads the cache's roped K as is (no span
                    // gate); the re-roping twin is sdpa-paged (in-kernel rope, no
                    // steel debug slot).
                    let plain = LoweredCommand {
                        constants: super::kernel_constants::AttentionPrefillPagedConstants {
                            rope_on_read: None,
                            self_only: None,
                            ..constants
                        }
                        .into_baked(),
                        bindings: bindings_for(false, false).into_baked(),
                        ..attn_cmd
                    };
                    let reroping_constants =
                        super::kernel_constants::AttentionPrefillPagedConstants {
                            debug_mode: None,
                            rope_on_read: ror_on,
                            self_only: None,
                            ..constants
                        };
                    let reroping = sdpa_paged_command(
                        p,
                        reroping_constants,
                        bindings_for(false, true),
                        bucket_m,
                    );
                    return Ok(vec![rope_cmd, attn_cmd, plain, reroping]);
                }
                attn_cmd
            } else {
                sdpa_paged_command(p, constants, bindings, bucket_m)
            }
        }

        // ── Plain residual add ─────────────────────────────────────
        I::Add(Slot(delta_slot), Slot(residual_slot)) => LoweredCommand {
            kernel: KernelId::Add,
            library: "elementwise",
            function: pick_specialized_symbol(
                "residual_add_f16_specialized",
                "residual_add_bf16_specialized",
                p.metal_dtype,
            ),
            // Token-parallel kernel reads element count from dispatch
            // shape; no function constants.
            constants: &[],
            // Token-parallel; reduction is per-element so the dispatch
            // covers `M * width` elements. Width is `p.q_size` on the
            // text path (the residual stream); on vision towers
            // (`VISION_Q_SIZE > 0`) the text `Q_SIZE` is garbage (the
            // config lacks text fields), so use the shape-tracked
            // `cur_width` (the residual width = vision_embed_dim at every
            // residual add). `eff_m` is bucket_m for these block-residual
            // adds (they precede the merger reshape).
            dispatch: {
                let add_width: u32 = if p.vision_q_size > 0 {
                    cur_width.get()
                } else {
                    p.q_size as u32
                };
                let mut d = DispatchShape::dispatch_1d(eff_m * add_width, THREADS_PER_GROUP);
                d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::X,
                    bucket_m: super::ids::BucketM(bucket_m),
                });
                d
            },
            bindings: baked(vec![
                Binding::ArenaSlot {
                    slot: *residual_slot,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: *delta_slot,
                    binding_index: 1,
                },
            ]),
            gemm_dims: None,
        },

        // ── Scalar-multiply broadcast ──────────────────────────────
        I::ScalarMul(Slot(in_slot), Slot(out_slot), Scale(scale)) => LoweredCommand {
            kernel: KernelId::ScalarMul,
            library: "elementwise",
            function: pick_specialized_symbol(
                "scalar_mul_f16_specialized",
                "scalar_mul_bf16_specialized",
                p.metal_dtype,
            ),
            // Slot 2: elementwise.metal fn-const indices are file-scoped
            // (0 = BIAS_ADD_NUM_COLS, 1 = TANH_SOFTCAP_CAP).
            constants: baked(vec![ConstantValue::float(2, *scale)]),
            // A scalar-multiply must cover the FULL activation row.
            // `cur_width` is the shape-tracked activation width: vocab
            // after the lm_head Gemm for granite's
            // `logits * 1/logits_scaling`; HIDDEN on the residual-stream
            // `embed * embedding_multiplier` and `branch *
            // residual_multiplier` muls (the embed multiply precedes the
            // first Gemm, so `cur_width` is seeded to the residual-stream
            // width — see `lower()`).
            //
            // The previous `bucket_m * p.q_size` scaled only the first
            // `Q_SIZE` columns of each row. For granite that is 4096 of
            // vocab=49159 — ~89% of every logits row kept its raw
            // `logits_scaling`× magnitude, so argmax was dominated by the
            // unscaled tail: degenerate repetition on short prompts, and
            // on longer prompts argmax→token 0 (the shared bos=eos=pad id)
            // → an immediate EOS stop → empty output. A PARTIAL scalar-
            // multiply is NOT argmax-invariant. `activation_broadcast`
            // takes a typed `ActivationWidth`, so passing `p.q_size` here
            // is now a COMPILE error — the bug is unrepresentable, not
            // guarded at runtime.
            dispatch: DispatchShape::activation_broadcast(
                eff_m,
                cur_width,
                super::ids::BucketM(bucket_m),
            ),
            bindings: baked(vec![
                Binding::ArenaSlot {
                    slot: *out_slot,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: *in_slot,
                    binding_index: 1,
                },
                // The scalar `scale` is baked into a function
                // constant on the specialized pipeline (Phase 5.B);
                // no runtime binding needed.
            ]),
            gemm_dims: None,
        },

        // ── Final logit softcapping (Gemma2/Gemma4) ────────────────
        //
        // `out = cap * tanh(x / cap)` with cap =
        // `p.final_logit_softcapping` baked as function constant 0.
        // Runs on the logits (`cur_width` = vocab after the lm_head
        // GEMM); token-parallel exact-thread dispatch like `Add`.
        // Always full-M: a softcapped lm_head's result is this cap, not
        // the GEMM, so the GEMM stays in the backbone and the lm_head
        // narrow path (`lower_subtile_tape_to_metal`) never sees it.
        I::TanhSoftCap(Slot(in_slot), Slot(out_slot)) => {
            debug_assert!(
                p.final_logit_softcapping > 0.0,
                "TanhSoftCap lowered with FINAL_LOGIT_SOFTCAPPING <= 0"
            );
            LoweredCommand {
                kernel: KernelId::TanhSoftCap,
                library: "elementwise",
                function: pick_specialized_symbol(
                    "tanh_soft_cap_f16_specialized",
                    "tanh_soft_cap_bf16_specialized",
                    p.metal_dtype,
                ),
                // Slot 1: elementwise.metal's fn-const indices are
                // file-scoped (slot 0 = BIAS_ADD_NUM_COLS).
                constants: baked(vec![ConstantValue::float(1, p.final_logit_softcapping)]),
                // Final logit softcap runs over the logits row (`cur_width`
                // = vocab after the lm_head Gemm). Same typed broadcast as
                // the granite ScalarMul: passing head geometry is a compile
                // error.
                dispatch: DispatchShape::activation_broadcast(
                    eff_m,
                    cur_width,
                    super::ids::BucketM(bucket_m),
                ),
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: *in_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 1,
                    },
                ]),
                gemm_dims: None,
            }
        }

        // ── Per-row bias broadcast (singleton, non-synth path) ────
        //
        // Emitted by `MetalBiasAddImpl` for Qwen2-style QKV biases
        // when the synth pre-attn megakernel doesn't claim the chain
        // (today: M ≥ 2 prefill). One bias-add dispatch per BiasAdd
        // tile; the synth path will subsume these at M=1 once the
        // matcher absorbs biases (P2).
        //
        // Binding contract (matches `bias_add_<dtype>_specialized` in
        // `elementwise.metal`):
        //   buffer(0) = input  [M, N]            T_act r
        //   buffer(1) = bias   [N]               T_act r
        //   buffer(2) = output [M, N]            T_act w
        //
        // `WeightTensor::Bias` resolves dense `<prefix>.bias`;
        // `WeightTensor::AffineLinearBias` resolves MLX-affine's
        // `linear_bias` (Qwen2 4bit ships this). The `BiasStorage` on
        // the step picks which arm — the macro knows the
        // weight's StorageFormat at FUF construction time.
        I::MetalBiasAdd(Slot(in_slot), Slot(out_slot), LayerId(layer), NDim(n), storage) => {
            LoweredCommand {
                kernel: KernelId::BiasAdd,
                library: "elementwise",
                function: pick_specialized_symbol(
                    "bias_add_f16_specialized",
                    "bias_add_bf16_specialized",
                    p.metal_dtype,
                ),
                constants: baked(vec![ConstantValue::uint(0, *n)]),
                dispatch: {
                    // `eff_m * n` baseline (eff_m shrinks by the merge factor
                    // for the merger's post-reshape bias_adds); m_scaling
                    // keeps the FULL bucket_m so the runtime num_tokens
                    // rescale stays relative to the whole bucket.
                    let mut d = DispatchShape::dispatch_1d(eff_m * *n, THREADS_PER_GROUP);
                    d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    });
                    d
                },
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: *in_slot,
                        binding_index: 0,
                    },
                    source(
                        w.of(WeightKind::Linear, 0)?,
                        match storage {
                            BiasStorage::Affine => WeightTensor::AffineLinearBias,
                            BiasStorage::Dense => WeightTensor::Bias,
                        },
                        super::ids::LayerId(*layer + layer_offset),
                        1,
                    ),
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 2,
                    },
                ]),
                gemm_dims: None,
            }
        }

        // ── The KV codec (TurboQuant): its row's guard gates every command ──
        I::KvEncode(operand, layer, offsets) => {
            let bits = tq_bits(p)?;
            let ops = TqOperands::of(p, *offsets, *layer, w)?;
            let layer = layer.get() + layer_offset;
            return Ok(vec![match operand {
                KvOperand::K => tq_quantize_command(p, layer, ops.k, bucket_m, bits),
                KvOperand::V => tq_quantize_command(p, layer, ops.v, bucket_m, bits),
            }]);
        }
        I::KvStage(operand, layer, class, offsets) => {
            let bits = tq_bits(p)?;
            let ops = TqOperands::of(p, *offsets, *layer, w)?;
            let global = *class == AttnMask::Causal;
            // Span blocks of K re-rope with the attention class's table.
            let table = w.rotary(p.rope_on_read.then_some(global))?;
            let (layer, cap) = (layer.get() + layer_offset, block_cap);
            // A row new for one sequence can be a prefix hit for another in the same step: the
            // step's new rows stage first, the cached rows after.
            use super::kernel_constants::TqStagePass;
            return Ok([TqStagePass::New, TqStagePass::Cached]
                .map(|pass| match operand {
                    KvOperand::K => {
                        tq_stage_command(p, layer, ops.k, global, table, bucket_m, cap, pass, bits)
                    }
                    KvOperand::V => {
                        tq_stage_command(p, layer, ops.v, global, table, bucket_m, cap, pass, bits)
                    }
                })
                .to_vec());
        }
        I::RotateRows(rows, which) => {
            let inverse = *which == RotatedRows::Output;
            return Ok(vec![tq_rotate_command(p, *rows, inverse, bucket_m)]);
        }
        // The decode attention of its class, reading the packed store.
        I::AttnPackedKv(q, out, layer, pairing, class, offsets) => {
            let bits = tq_bits(p)?;
            let ops = TqOperands::of(p, *offsets, *layer, w)?;
            let decode = match class {
                AttnMask::Causal => I::AttentionViaCache(*q, *out, *layer, *pairing),
                AttnMask::SlidingWindow => I::SlidingAttentionViaCache(*q, *out, *layer, *pairing),
            };
            let via_cache = lower_one(
                p,
                chunked,
                &decode,
                w,
                bucket_m,
                layer_offset,
                splitk_scratch_bytes,
                moe_scratch_bytes,
                roped_k_scratch_bytes,
                attn_unfused_scratch_bytes,
                block_cap,
                profile,
                cur_width,
                m_divisor,
            )?;
            let layer = layer.get() + layer_offset;
            let twin = |c: &LoweredCommand| tq_attention_command(c, layer, ops, bits);
            return Ok(via_cache.first().map(twin).into_iter().collect());
        }
        I::Moe(block, step) => {
            let is_nax = profile.is_some_and(|p| crate::targets::is_nax_capable(p.generation));
            let at = MoeBake {
                bucket_m,
                layer_offset,
                is_nax,
                codes: super::kernel_constants::AffineCodesTarget::of(profile),
            };
            return lower_moe_step(p, block, *step, w, at, moe_scratch_bytes);
        }

        // ── Qwen3.5 Gated-DeltaNet: conv1d → gating → scan → gated-RMSNorm ──
        //
        // One coarse op lowers to FOUR compute commands that pass f32
        // intermediates through the bucket's `moe_scratch` (reused as
        // generic op-scratch). The final `core` lands in the model-dtype
        // arena `out_slot` (read by the out_proj gemm). Persistent
        // conv/ssm state binds from `GdnStatePool` via Runtime bindings
        // (the KV-cache pattern); per-forward slot indices + is_fresh
        // come from the worker. `lower` supplies barrier_before=true for
        // commands 1.. (intra-instruction RAW on conv_out/g/beta/o +
        // the in-place state rings), so each stage sees the prior one's
        // writes. Math is the golden-tested cpu_golden recurrence.
        I::GatedDeltaNet(
            Slot(qkv_slot),
            Slot(z_slot),
            Slot(a_slot),
            Slot(b_slot),
            Slot(out_slot),
            LayerId(layer),
        ) => {
            use crate::tape::ids::{BucketM, LayerId};
            use crate::tape::lowered::{MScaleAxis, MScaling};
            let global_layer = *layer + layer_offset;
            let layer_id = LayerId(global_layer);
            let dtype = dequant_dtype_for(p);
            let nk = p.gdn_num_k_heads;
            let nv = p.gdn_num_v_heads;
            let hk = p.gdn_head_k_dim;
            let hv = p.gdn_head_v_dim;
            let kernel = p.gdn_conv_kernel;
            let conv_dim = p.gdn_conv_dim as u32;
            let value_dim = nv * hv;
            let scale = (hk as f32).powf(-0.5);

            let layout = GdnScratchLayout::compute(bucket_m, conv_dim, nv, value_dim);
            *moe_scratch_bytes = (*moe_scratch_bytes).max(layout.total);

            // One source for the whole bundle; `which` picks the sub-tensor.
            let ix = w.of(WeightKind::GatedDeltaNet, 0)?;
            let weight = |which, binding_index| source(ix, which, layer_id, binding_index);
            let runtime = |kind: RuntimeBindingKind, binding_index: u8| Binding::Runtime {
                kind,
                binding_index,
            };

            let mut cmds = Vec::with_capacity(4);

            // 1. Causal depthwise conv1d (+SiLU), varlen + stateful.
            //    grid (num_seqs[set via seq_axis=X], ceil(conv_dim/tg_y), 1).
            cmds.push(LoweredCommand {
                kernel: KernelId::GatedDeltaNet,
                library: "gdn_conv1d_varlen",
                function: gdn_conv1d_varlen_static_name(dtype),
                constants: baked(vec![
                    ConstantValue::uint(0, conv_dim),
                    ConstantValue::uint(1, kernel),
                ]),
                dispatch: {
                    let tg_y = conv_dim.clamp(1, THREADS_PER_GROUP);
                    DispatchShape {
                        threadgroups: (1, conv_dim.div_ceil(tg_y), 1),
                        threads_per_threadgroup: (1, tg_y, 1),
                        // bucket_m=1 no-ops the token scale; seq_axis SETS X = num_seqs.
                        m_scaling: Some(MScaling {
                            axis: MScaleAxis::X,
                            bucket_m: BucketM(1),
                            seq_axis: Some(MScaleAxis::X),
                        }),
                    }
                },
                bindings: baked(vec![
                    Binding::MoeScratch {
                        binding_index: 0,
                        byte_offset: layout.conv_out,
                    },
                    Binding::ArenaSlot {
                        slot: *qkv_slot,
                        binding_index: 1,
                    },
                    weight(WeightTensor::GdnConv1d, 2),
                    runtime(RuntimeBindingKind::GdnConvState { layer: layer_id }, 3),
                    runtime(RuntimeBindingKind::CuSeqlensQ, 4),
                    runtime(RuntimeBindingKind::GdnStateIndices, 5),
                    runtime(RuntimeBindingKind::GdnIsFresh, 6),
                ]),
                gemm_dims: None,
            });

            // 2. Input-dependent gating → g, beta (f32 scratch).
            //    1D over [num_tokens · nv]; scales X with num_tokens.
            cmds.push(LoweredCommand {
                kernel: KernelId::GatedDeltaNet,
                library: "gdn_gating",
                function: gdn_gating_static_name(dtype),
                constants: baked(vec![
                    ConstantValue::uint(0, bucket_m * nv),
                    ConstantValue::uint(1, nv),
                ]),
                dispatch: {
                    let mut d = DispatchShape::dispatch_1d(bucket_m * nv, THREADS_PER_GROUP);
                    d.m_scaling = Some(MScaling {
                        axis: MScaleAxis::X,
                        bucket_m: BucketM(bucket_m),
                        seq_axis: None,
                    });
                    d
                },
                bindings: baked(vec![
                    Binding::MoeScratch {
                        binding_index: 0,
                        byte_offset: layout.g,
                    },
                    Binding::MoeScratch {
                        binding_index: 1,
                        byte_offset: layout.beta,
                    },
                    Binding::ArenaSlot {
                        slot: *a_slot,
                        binding_index: 2,
                    },
                    Binding::ArenaSlot {
                        slot: *b_slot,
                        binding_index: 3,
                    },
                    weight(WeightTensor::GdnALog, 4),
                    weight(WeightTensor::GdnDtBias, 5),
                ]),
                gemm_dims: None,
            });

            // 3. Recurrent gated delta-rule scan → o (f32 scratch).
            //    grid (ceil(hv/tgx), nv, num_seqs[set via seq_axis=Z]).
            //    Always the _f32 instantiation (all I/O is f32 scratch).
            cmds.push(LoweredCommand {
                kernel: KernelId::GatedDeltaNet,
                library: "gdn_scan_varlen",
                function: "gdn_scan_varlen_f32",
                constants: baked(vec![
                    ConstantValue::uint(0, nk),
                    ConstantValue::uint(1, nv),
                    ConstantValue::uint(2, hk),
                    ConstantValue::uint(3, hv),
                    ConstantValue::float(4, scale),
                ]),
                dispatch: {
                    let tgx = hv.clamp(1, THREADS_PER_GROUP);
                    DispatchShape {
                        threadgroups: (hv.div_ceil(tgx), nv, 1),
                        threads_per_threadgroup: (tgx, 1, 1),
                        m_scaling: Some(MScaling {
                            axis: MScaleAxis::Z,
                            bucket_m: BucketM(1),
                            seq_axis: Some(MScaleAxis::Z),
                        }),
                    }
                },
                bindings: baked(vec![
                    Binding::MoeScratch {
                        binding_index: 0,
                        byte_offset: layout.o,
                    },
                    Binding::MoeScratch {
                        binding_index: 1,
                        byte_offset: layout.conv_out,
                    },
                    Binding::MoeScratch {
                        binding_index: 2,
                        byte_offset: layout.g,
                    },
                    Binding::MoeScratch {
                        binding_index: 3,
                        byte_offset: layout.beta,
                    },
                    runtime(RuntimeBindingKind::GdnSsmState { layer: layer_id }, 4),
                    runtime(RuntimeBindingKind::CuSeqlensQ, 5),
                    runtime(RuntimeBindingKind::GdnStateIndices, 6),
                    runtime(RuntimeBindingKind::GdnIsFresh, 7),
                ]),
                gemm_dims: None,
            });

            // 4. Gated RMSNorm → core (model-dtype arena out_slot).
            //    One threadgroup per row [num_tokens · nv]; scales X.
            cmds.push(LoweredCommand {
                kernel: KernelId::GatedDeltaNet,
                library: "gdn_rms_norm_gated",
                function: gdn_rms_norm_gated_static_name(dtype),
                constants: baked(vec![
                    ConstantValue::uint(0, hv),
                    ConstantValue::uint(1, bucket_m * nv),
                    ConstantValue::float(2, p.rms_norm_eps),
                ]),
                dispatch: DispatchShape {
                    threadgroups: (bucket_m * nv, 1, 1),
                    threads_per_threadgroup: (THREADS_PER_GROUP, 1, 1),
                    m_scaling: Some(MScaling {
                        axis: MScaleAxis::X,
                        bucket_m: BucketM(bucket_m),
                        seq_axis: None,
                    }),
                },
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::MoeScratch {
                        binding_index: 1,
                        byte_offset: layout.o,
                    },
                    Binding::ArenaSlot {
                        slot: *z_slot,
                        binding_index: 2,
                    },
                    weight(WeightTensor::GdnNorm, 3),
                ]),
                gemm_dims: None,
            });

            return Ok(cmds);
        }

        // ── Vision LayerNorm-with-bias (Qwen3.5-VL ViT norm1/norm2/
        //    merger.norm) ────────────────────────────────────────────
        //
        // ── Weight-only LayerNorm (ModernBERT `norm_bias=False`) ─────
        //
        // Faithful to the cuda `Instruction::MeanSubRmsNorm` eval
        // (`kernels::cohere_layer_norm`): `(x-mean)*rsqrt(var+eps)*weight`
        // with NO bias term. Shares the `vision_layernorm` kernel; the
        // `LN_HAS_BIAS=0` function constant suppresses the bias read,
        // and binding index 3 points at the WEIGHT buffer so the
        // argument is a valid allocation the kernel never dereferences
        // (Metal requires the binding to exist even when unread).
        I::MeanSubRmsNorm(Slot(in_slot), Slot(out_slot), LayerId(layer)) => {
            let layer_id = super::ids::LayerId(*layer + layer_offset);
            let ix = w.of(WeightKind::RmsNorm, 0)?;
            LoweredCommand {
                kernel: KernelId::VisionLayerNorm,
                library: "vision_layernorm",
                function: vision_layernorm_static_name(
                    p.metal_dtype,
                    norm_gain_dtype(p, WeightKind::RmsNorm),
                ),
                constants: baked(vec![
                    ConstantValue::uint(0, eff_m),
                    ConstantValue::uint(1, cur_width.get()),
                    ConstantValue::float(2, p.rms_norm_eps),
                    ConstantValue::uint(3, 0),
                ]),
                dispatch: DispatchShape {
                    threadgroups: (eff_m, 1, 1),
                    threads_per_threadgroup: (THREADS_PER_GROUP, 1, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    }),
                },
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *in_slot,
                        binding_index: 1,
                    },
                    source(ix, WeightTensor::Weight, layer_id, 2),
                    source(ix, WeightTensor::Weight, layer_id, 3),
                ]),
                gemm_dims: None,
            }
        }

        // Faithful to the cuda `Instruction::MeanSubRmsNormBiasAdd` eval
        // (`instr.rs`): one fused `(x-mean)*rsqrt(var+eps)*weight + bias`
        // pass over the row-width. The `vision_layernorm` kernel does one
        // threadgroup per row; `LN_HIDDEN` is the shape-tracked residual
        // width (`cur_width` = vision_embed_dim at every norm site) and
        // `LN_M` / the grid are `eff_m` (= bucket_m here — all three
        // norms precede the merger reshape). Weight + bias resolve from
        // one `LayerNorm` source, mirroring the multi-tensor `GatedDeltaNet` binding pattern.
        I::MeanSubRmsNormBiasAdd(Slot(in_slot), Slot(out_slot), LayerId(layer)) => {
            let layer_id = super::ids::LayerId(*layer + layer_offset);
            let ix = w.of(WeightKind::LayerNorm, 0)?;
            LoweredCommand {
                kernel: KernelId::VisionLayerNorm,
                library: "vision_layernorm",
                function: vision_layernorm_static_name(
                    p.metal_dtype,
                    norm_gain_dtype(p, WeightKind::LayerNorm),
                ),
                constants: baked(vec![
                    ConstantValue::uint(0, eff_m),
                    ConstantValue::uint(1, cur_width.get()),
                    ConstantValue::float(2, p.rms_norm_eps),
                    ConstantValue::uint(3, 1),
                ]),
                dispatch: DispatchShape {
                    threadgroups: (eff_m, 1, 1),
                    threads_per_threadgroup: (THREADS_PER_GROUP, 1, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    }),
                },
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *in_slot,
                        binding_index: 1,
                    },
                    source(ix, WeightTensor::Weight, layer_id, 2),
                    source(ix, WeightTensor::Bias, layer_id, 3),
                ]),
                gemm_dims: None,
            }
        }

        // ── Vision standalone tanh-approx GELU (ViT MLP / merger MLP) ─
        //
        // Faithful to the cuda `Instruction::Gelu` eval
        // (`kernels::gelu_tanh_inplace`). Flat token-parallel; `n` =
        // bucket-level element count `eff_m * cur_width`, so the merger's
        // post-reshape gelu (`eff_m = bucket_m / vision_merge_factor`,
        // `cur_width = vision_merge_hidden`) shrinks rows by the merge
        // factor for free. The `gelu_tanh` kernel guards `gid >= n`, so
        // the m_scaling tail and any padding rows are no-ops. `n` is the
        // kernel's runtime `constant uint& n` at buffer(2) — bound as a
        // `setBytes` inline value (NOT a function constant).
        I::Gelu(Slot(in_slot), Slot(out_slot)) => {
            let n_elems = eff_m * cur_width.get();
            LoweredCommand {
                kernel: KernelId::VisionGelu,
                library: "activation",
                function: gelu_tanh_static_name(p.metal_dtype),
                constants: &[],
                dispatch: {
                    let mut d = DispatchShape::dispatch_1d(n_elems, THREADS_PER_GROUP);
                    d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    });
                    d
                },
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *in_slot,
                        binding_index: 1,
                    },
                    Binding::Inline {
                        binding_index: 2,
                        value: n_elems,
                    },
                ]),
                gemm_dims: None,
            }
        }

        // ── Vision standalone erf-exact GELU (LocateAnything projector) ─
        //
        // Same dispatch shape as `I::Gelu` above; only the kernel symbol
        // differs (`gelu_erf` = exact erf, NOT the tanh approximation —
        // the two flavors coexist in one model: MoonViT block MLPs use
        // tanh, the multimodal projector uses erf).
        I::GeluErf(Slot(in_slot), Slot(out_slot)) => {
            let n_elems = eff_m * cur_width.get();
            LoweredCommand {
                kernel: KernelId::VisionGelu,
                library: "activation",
                function: gelu_erf_static_name(p.metal_dtype),
                constants: &[],
                dispatch: {
                    let mut d = DispatchShape::dispatch_1d(n_elems, THREADS_PER_GROUP);
                    d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    });
                    d
                },
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *in_slot,
                        binding_index: 1,
                    },
                    Binding::Inline {
                        binding_index: 2,
                        value: n_elems,
                    },
                ]),
                gemm_dims: None,
            }
        }

        // Same dispatch shape again; QuickGELU = x·sigmoid(1.702x)
        // (Qwen2-VL tower blocks — its merger uses `gelu_erf`, so both
        // flavors appear in one tape).
        I::QuickGelu(Slot(in_slot), Slot(out_slot)) => {
            let n_elems = eff_m * cur_width.get();
            LoweredCommand {
                kernel: KernelId::VisionGelu,
                library: "activation",
                function: quick_gelu_static_name(p.metal_dtype),
                constants: &[],
                dispatch: {
                    let mut d = DispatchShape::dispatch_1d(n_elems, THREADS_PER_GROUP);
                    d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    });
                    d
                },
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *in_slot,
                        binding_index: 1,
                    },
                    Binding::Inline {
                        binding_index: 2,
                        value: n_elems,
                    },
                ]),
                gemm_dims: None,
            }
        }

        // ── Vision pixels materialization (Qwen3.5-VL ViT prelude) ──
        //
        // Faithful to the cuda `Instruction::LoadPixels` eval (a D2D
        // copy of `ForwardCtx::pixels` into a fresh tile). Here the
        // pixels live in the `Pixels` runtime extern (overwritten per
        // forward); `copy_rows` blits them into the arena `out_slot` the
        // patch_embed GEMM reads. `n` = bucket-level pixel count
        // `eff_m * VISION_IN_FEATURES` (LoadPixels is the prelude op, so
        // eff_m == bucket_m); m_scaling shrinks the grid to live
        // num_tokens and the kernel's `gid >= n` guard caps the tail.
        I::LoadPixels(Slot(out_slot)) => {
            let n_elems = eff_m * p.vision_in_features as u32;
            LoweredCommand {
                kernel: KernelId::VisionLoadPixels,
                library: "elementwise",
                function: copy_rows_static_name(p.metal_dtype),
                constants: &[],
                dispatch: {
                    let mut d = DispatchShape::dispatch_1d(n_elems, THREADS_PER_GROUP);
                    d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    });
                    d
                },
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::Pixels,
                        binding_index: 1,
                    },
                    Binding::Inline {
                        binding_index: 2,
                        value: n_elems,
                    },
                ]),
                gemm_dims: None,
            }
        }

        // ── Vision pos_embeds materialization (Qwen3.5-VL ViT) ──────
        //
        // The exact sibling of the `LoadPixels` arm above: `copy_rows`
        // blits the host-interpolated learned positional embedding from
        // the `VisionPosEmbeds` runtime extern into the arena `out_slot`
        // that the downstream `add(pos_embeds, hidden_states)` consumes.
        // `n` = `eff_m * VISION_Q_SIZE` — VISION_Q_SIZE (=
        // vision_num_heads * vision_head_dim) is the residual-stream
        // width (= vision_embed_dim), matching the patch_embed output
        // pos_embeds is added to. m_scaling shrinks the grid to live
        // num_tokens; the kernel's `gid >= n` guard caps the tail.
        I::LoadPosEmbeds(Slot(out_slot)) => {
            let n_elems = eff_m * p.vision_q_size as u32;
            LoweredCommand {
                kernel: KernelId::VisionLoadPixels,
                library: "elementwise",
                function: copy_rows_static_name(p.metal_dtype),
                constants: &[],
                dispatch: {
                    let mut d = DispatchShape::dispatch_1d(n_elems, THREADS_PER_GROUP);
                    d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    });
                    d
                },
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::VisionPosEmbeds,
                        binding_index: 1,
                    },
                    Binding::Inline {
                        binding_index: 2,
                        value: n_elems,
                    },
                ]),
                gemm_dims: None,
            }
        }

        // ── Vision 2D NeoX RoPE (Qwen3.5-VL ViT) ────────────────────
        //
        // `vision_rope_2d` rotates ONE tensor per dispatch, so q and k
        // get one command each. The kernel reads `freqs` (f32, the
        // VisionRopeFreqs runtime extern) and computes cos/sin
        // internally; output and input are DISTINCT arena slots
        // (VisionRopeImpl declares no `output_alias`, so liveness
        // coloring keeps them separate — required because the kernel
        // reads a rotate_half `partner` element, so in-place would race
        // across threadgroups). Faithful to the cuda
        // `Instruction::VisionRope` eval (rotate_half on `[L, H, D]`).
        I::VisionRope(Slot(q_slot), Slot(k_slot), Slot(q_out_slot), Slot(k_out_slot)) => {
            let hd = p.vision_head_dim;
            let nh = p.vision_num_heads;
            // N_ELEMS guard = L * H * D = eff_m * VISION_Q_SIZE (rope is
            // a per-block op, so eff_m == bucket_m). m_scaling rescales
            // the grid to the live num_tokens; the function-constant
            // guard caps it at the baked bucket level.
            let n_elems = eff_m * p.vision_q_size as u32;
            let consts = || {
                vec![
                    ConstantValue::uint(0, hd),
                    ConstantValue::uint(1, nh),
                    ConstantValue::uint(2, n_elems),
                ]
            };
            let dispatch = || {
                let mut d = DispatchShape::dispatch_1d(n_elems, THREADS_PER_GROUP);
                d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::X,
                    bucket_m: super::ids::BucketM(bucket_m),
                });
                d
            };
            let rope_cmd = |out_slot: u32, in_slot: u32| LoweredCommand {
                kernel: KernelId::VisionRope,
                library: "vision_rope_2d",
                function: vision_rope_2d_static_name(p.metal_dtype, p.vision_rope_interleaved),
                constants: baked(consts()),
                dispatch: dispatch(),
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: out_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: in_slot,
                        binding_index: 1,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::VisionRopeFreqs,
                        binding_index: 2,
                    },
                ]),
                gemm_dims: None,
            };
            return Ok(vec![
                rope_cmd(*q_out_slot, *q_slot),
                rope_cmd(*k_out_slot, *k_slot),
            ]);
        }

        // ── Vision bidirectional varlen attention (Qwen3.5-VL ViT) ──
        //
        // Faithful to the cuda `Instruction::VarlenAttention` eval
        // (cacheless, non-causal, per-segment SDPA, scale =
        // VISION_ATTN_SCALE). Qwen3.5-VL emits `CuSeqlens::Batch`, so
        // the segment boundaries come from the existing `CuSeqlensQ`
        // runtime binding (the vision wrapper writes the per-image
        // cu_seqlens into `ForwardCtx::cu_seqlens_q`). `VA_NUM_SEGS` is
        // baked to `bucket_m` — a safe upper bound: the kernel's
        // segment-search loop breaks at the first containing segment
        // (O(1) for the common full-image case), and zero-padded
        // cu_seqlens entries past the real segments form empty ranges
        // that never match. q/k/v/out are `[L, H, D]` token-major.
        // ── Encoder (bidirectional) self-attention — ModernBERT ──────
        //
        // Faithful to the cuda `Instruction::EncoderAttention` eval:
        // flash attention with `is_causal=false`, no softcap, no
        // sliding window, RoPE already applied upstream. That is the
        // SAME computation the vision tower's varlen attention does,
        // so it rides the same kernel: an encoder forward IS a
        // prefill, and `CuSeqlensQ` is exactly the prefill sequence
        // boundary buffer, so one segment per sequence falls out
        // without a new runtime binding. Geometry comes from the TEXT
        // consts (`head_dim`/`num_attention_heads`/`attn_scale`), not
        // the vision ones.
        I::EncoderAttention(Slot(q_slot), Slot(k_slot), Slot(v_slot), Slot(out_slot)) => {
            let hd = p.head_dim;
            let nh = p.num_q_heads;
            LoweredCommand {
                kernel: KernelId::VisionVarlenAttn,
                library: "vision_varlen_attn",
                function: vision_varlen_attn_static_name(p.metal_dtype),
                constants: baked(vec![
                    ConstantValue::uint(0, hd),
                    ConstantValue::uint(1, nh),
                    ConstantValue::uint(2, bucket_m),
                    ConstantValue::uint(3, eff_m),
                    ConstantValue::float(4, p.attn_scale),
                ]),
                dispatch: {
                    let mut d = DispatchShape::dispatch_1d(eff_m * nh, 64);
                    d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    });
                    d
                },
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *q_slot,
                        binding_index: 1,
                    },
                    Binding::ArenaSlot {
                        slot: *k_slot,
                        binding_index: 2,
                    },
                    Binding::ArenaSlot {
                        slot: *v_slot,
                        binding_index: 3,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::CuSeqlensQ,
                        binding_index: 4,
                    },
                ]),
                gemm_dims: None,
            }
        }

        I::VarlenAttention(Slot(q_slot), Slot(k_slot), Slot(v_slot), Slot(out_slot), segments) => {
            // `Batch`: single per-image segmentation in `cu_seqlens_q`
            // (Qwen3.5-VL / LocateAnything / Qwen2-VL). `VisionFull` /
            // `VisionWindow`: Qwen2.5-VL windowed attention — the full-attention layers
            // read per-image boundaries, the window layers per-window
            // boundaries; both arrive as dedicated runtime externs. The
            // kernel is identical across kinds (its segment-search walks
            // whatever cu_seqlens it's bound to; `VA_NUM_SEGS` stays the
            // bucket_m upper bound and zero-padded tail entries form
            // empty ranges).
            let cu_binding = match segments {
                CuSeqlens::Batch => RuntimeBindingKind::CuSeqlensQ,
                CuSeqlens::VisionFull => RuntimeBindingKind::VisionCuSeqlensFull,
                CuSeqlens::VisionWindow => RuntimeBindingKind::VisionCuSeqlensWindow,
            };
            let hd = p.vision_head_dim;
            let nh = p.vision_num_heads;
            LoweredCommand {
                kernel: KernelId::VisionVarlenAttn,
                library: "vision_varlen_attn",
                function: vision_varlen_attn_static_name(p.metal_dtype),
                constants: baked(vec![
                    ConstantValue::uint(0, hd),
                    ConstantValue::uint(1, nh),
                    // Safe upper bound on segment count (= max tokens);
                    // padded cu_seqlens entries are empty no-ops.
                    ConstantValue::uint(2, bucket_m),
                    // VA_N_TOKENS guard (= bucket_m; m_scaling shrinks the
                    // grid to live num_tokens, the guard caps the tail).
                    ConstantValue::uint(3, eff_m),
                    ConstantValue::float(4, p.vision_attn_scale),
                ]),
                dispatch: {
                    // 1 thread per (query token, head); 64 threads/tg.
                    let mut d = DispatchShape::dispatch_1d(eff_m * nh, 64);
                    d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    });
                    d
                },
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *q_slot,
                        binding_index: 1,
                    },
                    Binding::ArenaSlot {
                        slot: *k_slot,
                        binding_index: 2,
                    },
                    Binding::ArenaSlot {
                        slot: *v_slot,
                        binding_index: 3,
                    },
                    Binding::Runtime {
                        kind: cu_binding,
                        binding_index: 4,
                    },
                ]),
                gemm_dims: None,
            }
        }

        // ── Row gather by runtime index buffer (Qwen2.5-VL) ────────
        //
        // `out[i, :] = src[indices[i], :]` over the CURRENT logical
        // tile shape — the DSL reshapes to `[L/S², S²·E]` around the
        // `WindowIndex` gather so window permutation moves whole merge
        // groups, and runs `ReverseIndices` on the `[L/S², d_model]` merger
        // output. Mirrors the cuda `kernels::embedding_gather` row
        // semantics; indices come from the `vision_window_index` /
        // `vision_reverse_indices` runtime externs.
        I::EmbeddingGather(Slot(in_slot), Slot(out_slot), indices) => {
            let idx_binding = match indices {
                GatherIndices::WindowIndex => RuntimeBindingKind::VisionWindowIndex,
                GatherIndices::ReverseIndices => RuntimeBindingKind::VisionReverseIndices,
            };
            let n_elems = eff_m * cur_width.get();
            LoweredCommand {
                kernel: KernelId::EmbeddingGather,
                library: "embedding_gather",
                function: embedding_gather_static_name(p.metal_dtype),
                constants: &[],
                dispatch: {
                    let mut d = DispatchShape::dispatch_1d(n_elems, THREADS_PER_GROUP);
                    d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    });
                    d
                },
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *in_slot,
                        binding_index: 1,
                    },
                    Binding::Runtime {
                        kind: idx_binding,
                        binding_index: 2,
                    },
                    Binding::Inline {
                        binding_index: 3,
                        value: n_elems,
                    },
                    Binding::Inline {
                        binding_index: 4,
                        value: cur_width.get(),
                    },
                ]),
                gemm_dims: None,
            }
        }

        // ── Metadata-only: no Metal dispatch ───────────────────────
        I::Reshape(..) => {
            // A view: no device memory moves. The rows after it see its
            // shape through `update_shape_state`.
            return Ok(Vec::new());
        }

        // ── Multimodal embed splice (VL text decoder) ──────────────
        //
        // Scatter the projected vision embeddings into the text
        // embedding stream at the image-placeholder rows. Faithful to
        // the cuda `Instruction::SpliceMmEmbeds` eval (a per-patch D2D
        // memcpy), but expressed as one scatter kernel: each source row
        // `s` of `mm_embeds` is copied to text-embedding row
        // `dst_rows[s]` (or skipped when `dst_rows[s] == u32::MAX`,
        // which covers text-only batches AND the padding tail past
        // `total_mm`). `slot` is the Embed-output arena tile (in/out —
        // `MmEmbedSpliceImpl` aliases its output onto this input).
        // `n` = `bucket_m * HIDDEN_SIZE` (safe upper bound: total_mm <=
        // num_tokens <= bucket_m); m_scaling shrinks the grid to the
        // live num_tokens. HIDDEN_SIZE is the residual width (matches
        // the `Embed` arm; mm_embeds rows are HIDDEN_SIZE wide).
        I::SpliceMmEmbeds(Slot(slot)) => {
            let hidden = p.hidden_size as u32;
            let n_elems = bucket_m * hidden;
            LoweredCommand {
                kernel: KernelId::MmEmbedSplice,
                library: "elementwise",
                function: mm_embed_splice_static_name(p.metal_dtype),
                constants: &[],
                dispatch: {
                    let mut d = DispatchShape::dispatch_1d(n_elems, THREADS_PER_GROUP);
                    d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    });
                    d
                },
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: *slot,
                        binding_index: 0,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::MmEmbeds,
                        binding_index: 1,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::MmDstRows,
                        binding_index: 2,
                    },
                    Binding::Inline {
                        binding_index: 3,
                        value: hidden,
                    },
                ]),
                gemm_dims: None,
            }
        }
    };

    Ok(vec![cmd])
}

/// Default 1D threads-per-threadgroup. Matches Metal's preferred
/// width on Apple Silicon (32-wide simdgroups × 8 = 256). The Phase
/// 5.B specialized-pipeline cache may pick a different value per
/// kernel + bucket once measured.
const THREADS_PER_GROUP: u32 = 256;

/// 2D GEMM tile dims (M × N axes). Placeholder — kept narrow so the
/// dispatch math stays sensible at small buckets; will be replaced
/// per (model, bucket) by the SpecializedPipelineCache in Phase 5.B.
const GEMM_TILE_M: u32 = 16;
const GEMM_TILE_N: u32 = 16;

/// Output tile dim for the prefill matrix variant
/// (`fused_gate_up_silu_mul_gemm_steel_*_specialized`). 4 simdgroups
/// per threadgroup × WM*WN = 2*2 placement → 32×32 output tile.
/// Matches `STEEL_BM`/`STEEL_BN` in `shaders/fused_gate_up_silu_mul.metal`.
const MLP_STEEL_TILE: u32 = 32;

/// Threads per threadgroup for the steel matrix variant.
/// `WM * WN * 32 = 2 * 2 * 32 = 128`.
const MLP_STEEL_THREADS: u32 = 128;

/// Output rows per threadgroup for the M=1 decode variant
/// (`fused_gate_up_silu_mul_decode_*_specialized`). Matches the
/// MLX gemv port's `blockM = BM*SM*TM = 4`.
const MLP_DECODE_BLOCK_M: u32 = 4;

/// Pick the f16-or-bf16 specialization for a kernel that follows the
/// `<base>_<dtype>_specialized` naming convention. Centralizes the
/// `Int4`-not-yet-wired panic so each lowering arm spells out only
/// the two symbol names it owns.
fn pick_specialized_symbol(
    f16_symbol: &'static str,
    bf16_symbol: &'static str,
    dtype: MetalDtype,
) -> &'static str {
    match dtype {
        MetalDtype::F16 => f16_symbol,
        MetalDtype::Bf16 => bf16_symbol,
        // AWQ/GPTQ dequant kernels have a different binding contract
        // (packed u32 weights + group scales) so a single dtype
        // substitution can't model them — surfaced as a panic for
        // clarity since `p.metal_dtype = Int4` is unreachable today.
        MetalDtype::Int4 => {
            panic!("metal lowering: Int4 dtype not yet wired through kernel symbol picker")
        }
    }
}

/// Static `&'static str` for the `silu_mul_<dtype>` symbol exported
/// by `silu_mul.metal`. Same `&'static str` constraint as the qmv /
/// qmm name helpers — `LoweredCommand::function` can't allocate.
fn silu_mul_static_name(dtype: DequantDtype) -> &'static str {
    match dtype {
        DequantDtype::F16 => "silu_mul_f16",
        DequantDtype::Bf16 => "silu_mul_bf16",
    }
}

/// `gelu_mul_<dtype>` sibling (decomposed GeGLU tail) — same
/// `silu_mul.metal` library.
fn gelu_mul_static_name(dtype: DequantDtype) -> &'static str {
    match dtype {
        DequantDtype::F16 => "gelu_mul_f16",
        DequantDtype::Bf16 => "gelu_mul_bf16",
    }
}

fn gate_apply_static_name(dtype: DequantDtype) -> &'static str {
    match dtype {
        DequantDtype::F16 => "gate_apply_f16",
        DequantDtype::Bf16 => "gate_apply_bf16",
    }
}

fn gate_split_static_name(dtype: DequantDtype) -> &'static str {
    match dtype {
        DequantDtype::F16 => "gate_split_f16",
        DequantDtype::Bf16 => "gate_split_bf16",
    }
}

fn gate_scale_static_name(dtype: DequantDtype) -> &'static str {
    match dtype {
        DequantDtype::F16 => "gate_scale_f16",
        DequantDtype::Bf16 => "gate_scale_bf16",
    }
}

// ── Gated-DeltaNet kernel symbol pickers ──────────────────────────
// The conv1d / gating / gated-RMSNorm kernels read model-dtype inputs
// (`x`/`a`/`b`/`z`/weights) and write f32 scratch (or, for the final
// RMSNorm, the model-dtype arena `out_slot`). The scan reads ALL-f32
// scratch (conv_out/g/beta/ssm/o), so it is always the `_f32`
// instantiation regardless of the model dtype.
fn gdn_conv1d_varlen_static_name(dtype: DequantDtype) -> &'static str {
    match dtype {
        DequantDtype::F16 => "gdn_conv1d_varlen_f16",
        DequantDtype::Bf16 => "gdn_conv1d_varlen_bf16",
    }
}

fn gdn_gating_static_name(dtype: DequantDtype) -> &'static str {
    match dtype {
        DequantDtype::F16 => "gdn_gating_f16",
        DequantDtype::Bf16 => "gdn_gating_bf16",
    }
}

fn gdn_rms_norm_gated_static_name(dtype: DequantDtype) -> &'static str {
    match dtype {
        DequantDtype::F16 => "gdn_rms_norm_gated_f16",
        DequantDtype::Bf16 => "gdn_rms_norm_gated_bf16",
    }
}

/// Dtype the gain/bias buffers of a norm bundle actually hold on the
/// device — the LOADER decides it, so it follows the bundle kind:
///
/// * `RmsNorm` gains go through `RmsNormOps::load`'s `take_as_dtype`,
///   pinned to the canonical's `SCALE_DTYPE` (ModernBERT: f16 gains
///   under bf16 activations).
/// * `LayerNorm` gains/biases go through `layer_norm_load`'s plain
///   `take`, which casts to the model's activation dtype (every ViT
///   norm today).
///
/// Reading one as the other is a bit-reinterpretation, not a rounding
/// error: an f16 0.575 read as bf16 is 7.3e-05.
fn norm_gain_dtype(p: &MetalModelConsts, bundle: WeightKind) -> ScaleDtype {
    match bundle {
        WeightKind::RmsNorm => scale_dtype_for(p),
        WeightKind::LayerNorm => match p.metal_dtype {
            MetalDtype::F16 => ScaleDtype::F16,
            MetalDtype::Bf16 => ScaleDtype::Bf16,
            MetalDtype::Int4 => {
                panic!("layer_norm gains: Int4 activations unsupported")
            }
        },
        other => panic!("norm_gain_dtype: {other:?} is not a norm bundle"),
    }
}

/// `vision_layernorm.metal` host-name picker,
/// `vision_layernorm_<T_act>_s_<T_scale>`. Activations are unquantized
/// (Int4 is unreachable — there is no quantized vision tower, and the
/// text encoders that share this kernel are unquantized too); the gain
/// dtype comes from [`norm_gain_dtype`], NOT from the activation.
fn vision_layernorm_static_name(dtype: MetalDtype, scale: ScaleDtype) -> &'static str {
    match (dtype, scale) {
        (MetalDtype::F16, ScaleDtype::F16) => "vision_layernorm_f16_s_f16",
        (MetalDtype::F16, ScaleDtype::Bf16) => "vision_layernorm_f16_s_bf16",
        (MetalDtype::Bf16, ScaleDtype::F16) => "vision_layernorm_bf16_s_f16",
        (MetalDtype::Bf16, ScaleDtype::Bf16) => "vision_layernorm_bf16_s_bf16",
        (MetalDtype::Int4, _) => {
            panic!("vision_layernorm: Int4 unsupported (vision tower is unquantized bf16/f16)")
        }
    }
}

/// `activation.metal` `gelu_tanh` (tanh-approx GELU) host-name picker.
fn gelu_tanh_static_name(dtype: MetalDtype) -> &'static str {
    match dtype {
        MetalDtype::F16 => "gelu_tanh_f16",
        MetalDtype::Bf16 => "gelu_tanh_bf16",
        MetalDtype::Int4 => {
            panic!("gelu_tanh: Int4 unsupported (activations are bf16/f16)")
        }
    }
}

/// `activation.metal` `gelu_erf` host-name picker (erf-exact GELU).
fn gelu_erf_static_name(dtype: MetalDtype) -> &'static str {
    match dtype {
        MetalDtype::F16 => "gelu_erf_f16",
        MetalDtype::Bf16 => "gelu_erf_bf16",
        MetalDtype::Int4 => {
            panic!("gelu_erf: Int4 unsupported (activations are bf16/f16)")
        }
    }
}

/// `activation.metal` `quick_gelu` host-name picker
/// (x·sigmoid(1.702x) — Qwen2-VL tower blocks).
fn quick_gelu_static_name(dtype: MetalDtype) -> &'static str {
    match dtype {
        MetalDtype::F16 => "quick_gelu_f16",
        MetalDtype::Bf16 => "quick_gelu_bf16",
        MetalDtype::Int4 => {
            panic!("quick_gelu: Int4 unsupported (activations are bf16/f16)")
        }
    }
}

/// `embedding_gather.metal` host-name picker (row gather by a runtime
/// u32 index buffer — Qwen2.5-VL window permutation / inverse).
/// Lower a fused gate/up GEMM + activation-mul (`silu` or `gelu`) to
/// the `fused_gate_up_silu_mul` library. Shared by `FusedGateUpSiluMul`
/// (SwiGLU, `is_gelu = false`) and `FusedGateUpGeluMul` (dense GeGLU —
/// Gemma3 text, `is_gelu = true`); the `IS_GELU` fn-const (slot 9
/// decode / 10 prefill) flips the kernel epilogue. Dispatch / bindings
/// are identical across activations.
#[allow(clippy::too_many_arguments)]
fn fused_gate_up_mul_cmd(
    p: &MetalModelConsts,
    in_slot: u32,
    out_slot: u32,
    layer: u32,
    is_gelu: bool,
    bucket_m: u32,
    ix: SourceIx,
    layer_offset: u32,
) -> LoweredCommand {
    let inter = p.intermediate_size as u32;
    let (threadgroups, threads_per_threadgroup) = if bucket_m == 1 {
        // Decode (MLX gemv port): blockM = BM*SM*TM = 4 outputs per
        // threadgroup, 256 threads/group = BN*SN = 8 simdgroups × 32 lanes.
        ((inter.div_ceil(MLP_DECODE_BLOCK_M), 1, 1), (256, 1, 1))
    } else {
        // Prefill (MLX-steel matrix variant): 32×32 output tile, 128
        // threads = 4 simdgroups × 32 lanes.
        let tg_x = inter.div_ceil(MLP_STEEL_TILE);
        let tg_y = bucket_m.div_ceil(MLP_STEEL_TILE);
        ((tg_x, tg_y, 1), (MLP_STEEL_THREADS, 1, 1))
    };
    // Both activations share the `fused_gate_up_silu_mul` symbols; the
    // decode (slots 3/4/5/9) and steel (6/7/8/10) variants keep distinct
    // fn-const slots so they don't clash when the library is loaded.
    let function = if bucket_m == 1 {
        pick_specialized_symbol(
            "fused_gate_up_silu_mul_decode_f16_specialized",
            "fused_gate_up_silu_mul_decode_bf16_specialized",
            p.metal_dtype,
        )
    } else {
        pick_specialized_symbol(
            "fused_gate_up_silu_mul_gemm_steel_f16_specialized",
            "fused_gate_up_silu_mul_gemm_steel_bf16_specialized",
            p.metal_dtype,
        )
    };
    let constants: Vec<ConstantValue> = if bucket_m == 1 {
        super::kernel_constants::FusedGateUpSiluMulDecodeConstants {
            bucket_m: super::ids::BucketM(bucket_m),
            intermediate_size: super::ids::IntermediateSize(p.intermediate_size as u32),
            // gate/up gemm K-dim = MLP input width = HIDDEN_SIZE, not
            // Q_SIZE (differ when head_dim != hidden/heads, e.g. Qwen3.5).
            q_size: super::ids::QSize(p.hidden_size as u32),
            is_gelu,
        }
        .into()
    } else {
        super::kernel_constants::FusedGateUpSiluMulPrefillConstants {
            bucket_m: super::ids::BucketM(bucket_m),
            intermediate_size: super::ids::IntermediateSize(p.intermediate_size as u32),
            q_size: super::ids::QSize(p.hidden_size as u32),
            is_gelu,
        }
        .into()
    };
    LoweredCommand {
        kernel: KernelId::FusedGateUpSiluMul,
        library: "fused_gate_up_silu_mul",
        function,
        constants: baked(constants),
        dispatch: DispatchShape {
            threadgroups,
            threads_per_threadgroup,
            m_scaling: if bucket_m == 1 {
                None
            } else {
                Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::Y,
                    bucket_m: super::ids::BucketM(bucket_m),
                })
            },
        },
        bindings: baked(vec![
            Binding::ArenaSlot {
                slot: out_slot,
                binding_index: 0,
            },
            Binding::ArenaSlot {
                slot: in_slot,
                binding_index: 1,
            },
            source(
                ix,
                WeightTensor::Weight,
                super::ids::LayerId(layer + layer_offset),
                2,
            ),
        ]),
        gemm_dims: None,
    }
}

fn embedding_gather_static_name(dtype: MetalDtype) -> &'static str {
    match dtype {
        MetalDtype::F16 => "embedding_gather_rows_f16",
        MetalDtype::Bf16 => "embedding_gather_rows_bf16",
        MetalDtype::Int4 => {
            panic!("embedding_gather: Int4 unsupported (activations are bf16/f16)")
        }
    }
}

/// `vision_rope_2d.metal` host-name picker. `interleaved` selects the
/// adjacent-pair (GPT-J / MoonViT) entry points over the NeoX
/// rotate_half ones — driven by `p.vision_rope_interleaved` (the
/// `vision_rope_style` config key).
fn vision_rope_2d_static_name(dtype: MetalDtype, interleaved: bool) -> &'static str {
    match (dtype, interleaved) {
        (MetalDtype::F16, false) => "vision_rope_2d_f16",
        (MetalDtype::Bf16, false) => "vision_rope_2d_bf16",
        (MetalDtype::F16, true) => "vision_rope_2d_interleaved_f16",
        (MetalDtype::Bf16, true) => "vision_rope_2d_interleaved_bf16",
        (MetalDtype::Int4, _) => {
            panic!("vision_rope_2d: Int4 unsupported (vision tower is bf16/f16)")
        }
    }
}

/// `vision_varlen_attn.metal` host-name picker.
fn vision_varlen_attn_static_name(dtype: MetalDtype) -> &'static str {
    match dtype {
        MetalDtype::F16 => "vision_varlen_attn_f16",
        MetalDtype::Bf16 => "vision_varlen_attn_bf16",
        MetalDtype::Int4 => {
            panic!("vision_varlen_attn: Int4 unsupported (vision tower is bf16/f16)")
        }
    }
}

/// `elementwise.metal` `copy_rows` host-name picker (LoadPixels blit).
fn copy_rows_static_name(dtype: MetalDtype) -> &'static str {
    match dtype {
        MetalDtype::F16 => "copy_rows_f16",
        MetalDtype::Bf16 => "copy_rows_bf16",
        MetalDtype::Int4 => panic!("copy_rows: Int4 unsupported (pixels are bf16/f16)"),
    }
}

/// `elementwise.metal` `mm_embed_splice` host-name picker (MM splice).
fn mm_embed_splice_static_name(dtype: MetalDtype) -> &'static str {
    match dtype {
        MetalDtype::F16 => "mm_embed_splice_f16",
        MetalDtype::Bf16 => "mm_embed_splice_bf16",
        MetalDtype::Int4 => panic!("mm_embed_splice: Int4 unsupported (embeds are bf16/f16)"),
    }
}

/// Convert the lowering-side dtype enum to the kernel-dispatcher one.
/// `MetalDtype` is the lowering vocabulary; `DequantDtype` is what
/// `quantized.rs` speaks (and what `qmv_kernel_static_name` /
/// `qmm_t_kernel_static_name` consume). Same two cases either way —
/// the duplicate enum exists because the kernel-dispatcher crate
/// can't depend on lowering types.
fn dequant_dtype_for(p: &MetalModelConsts) -> DequantDtype {
    match p.metal_dtype {
        MetalDtype::F16 => DequantDtype::F16,
        MetalDtype::Bf16 => DequantDtype::Bf16,
        MetalDtype::Int4 => panic!(
            "metal lowering: MetalStep::AffineQmm requires p.metal_dtype \
             ∈ {{F16, Bf16}} (the activation dtype); got Int4"
        ),
    }
}

/// Scale-storage dtype the kernel reads `*.scales` / `*.biases` device
/// pointers as — the on-disk dtype for the affine quant per-group
/// params. Reads `p.scale_dtype`, populated from each arch's
/// quantization manifest (default F16; Qwen3 family overrides to BF16
/// because their mlx-community 4bit checkpoints ship BF16 scales).
fn scale_dtype_for(p: &MetalModelConsts) -> ScaleDtype {
    p.scale_dtype
}

/// Bindings shared by every `MetalStep::AffineQmm` lowering's qmv
/// and qmm_t Standard kernels — both bind buffers 0..4 in the same
/// order: (packed weight, scales, biases, x in, y out). Worker
/// resolves the `Affine*` `WeightTensor` arms via
/// `LinearLayer::AffineQuant` (`worker.rs:1414`).
fn affine_qmm_bindings(
    in_slot: u32,
    out_slot: u32,
    layer: super::ids::LayerId,
    ix: SourceIx,
) -> Vec<Binding> {
    let mut v = affine_weight_bindings(ix, layer).to_vec();
    v.extend([
        Binding::ArenaSlot {
            slot: in_slot,
            binding_index: 3,
        },
        Binding::ArenaSlot {
            slot: out_slot,
            binding_index: 4,
        },
    ]);
    v
}

/// An MLX-affine projection's packed weight, scales and biases at bindings 0..3.
fn affine_weight_bindings(ix: SourceIx, layer: super::ids::LayerId) -> [Binding; 3] {
    use WeightTensor as T;
    let at = |which, binding_index| source(ix, which, layer, binding_index);
    [
        at(T::Weight, 0),
        at(T::AffineScales, 1),
        at(T::AffineBiases, 2),
    ]
}

/// Bindings for `affine_qmm_t_splitk`: same first four bindings as
/// `affine_qmm_bindings` (packed weight, scales, biases, x in) but
/// the `y` output (binding 4) is the shared `Binding::Scratch`
/// buffer instead of an arena slot — the kernel writes the
/// `[split_k, M, N]` partial here, and the follow-up
/// `splitk_reduce_sum` reads it.
fn affine_qmm_splitk_bindings(
    in_slot: u32,
    layer: super::ids::LayerId,
    ix: SourceIx,
) -> Vec<Binding> {
    let mut v = affine_weight_bindings(ix, layer).to_vec();
    v.extend([
        Binding::ArenaSlot {
            slot: in_slot,
            binding_index: 3,
        },
        Binding::Scratch { binding_index: 4 },
    ]);
    v
}

/// Byte width of one element in the activation dtype the worker
/// allocates the SplitK scratch buffer against. F16 / Bf16 = 2 bytes.
fn elem_size_bytes(dtype: DequantDtype) -> u32 {
    match dtype {
        DequantDtype::F16 | DequantDtype::Bf16 => 2,
    }
}

/// Format the kernel symbol name for a `MetalStep::RmsNorm`
/// lowering. Matches the `INST_RMSNORM` instantiations in
/// `shaders/rmsnorm.metal` — `rmsnorm_<T_act>_s_<T_scale>_specialized`.
/// Mirrors P10b's in-register cast pattern: RMSNorm gains stay in their on-disk
/// dtype on the device, the kernel reads them through a `T_scale`
/// pointer and casts to `T_act` in registers.
fn rmsnorm_kernel_static_name(p: &MetalModelConsts, scale_dtype: ScaleDtype) -> &'static str {
    use ScaleDtype as S;
    match (p.metal_dtype, scale_dtype) {
        (MetalDtype::F16, S::F16) => "rmsnorm_f16_s_f16_specialized",
        (MetalDtype::Bf16, S::F16) => "rmsnorm_bf16_s_f16_specialized",
        (MetalDtype::F16, S::Bf16) => "rmsnorm_f16_s_bf16_specialized",
        (MetalDtype::Bf16, S::Bf16) => "rmsnorm_bf16_s_bf16_specialized",
        (dt, sdt) => unreachable!(
            "rmsnorm_kernel_static_name: (dtype={dt:?}, scale_dtype={sdt:?}) \
             not instantiated"
        ),
    }
}

/// As [`rmsnorm_kernel_static_name`] for `MetalStep::FusedAddRmsNorm`.
/// Symbol naming: `fused_add_rmsnorm_<T_act>_s_<T_scale>_specialized`,
/// matching the `INST_FUSED_ARN` instantiations in
/// `shaders/fused_add_rmsnorm.metal`.
fn fused_add_rmsnorm_kernel_static_name(
    p: &MetalModelConsts,
    scale_dtype: ScaleDtype,
) -> &'static str {
    use ScaleDtype as S;
    match (p.metal_dtype, scale_dtype) {
        (MetalDtype::F16, S::F16) => "fused_add_rmsnorm_f16_s_f16_specialized",
        (MetalDtype::Bf16, S::F16) => "fused_add_rmsnorm_bf16_s_f16_specialized",
        (MetalDtype::F16, S::Bf16) => "fused_add_rmsnorm_f16_s_bf16_specialized",
        (MetalDtype::Bf16, S::Bf16) => "fused_add_rmsnorm_bf16_s_bf16_specialized",
        (dt, sdt) => unreachable!(
            "fused_add_rmsnorm_kernel_static_name: (dtype={dt:?}, \
             scale_dtype={sdt:?}) not instantiated"
        ),
    }
}

/// `MetalStep::NormAddScalarMul` symbol — the norm-THEN-add mirror
/// of `fused_add_rmsnorm_kernel_static_name`; same dtype enumeration,
/// same `fused_add_rmsnorm` library.
fn norm_add_scalar_mul_kernel_static_name(
    p: &MetalModelConsts,
    scale_dtype: ScaleDtype,
) -> &'static str {
    use ScaleDtype as S;
    match (p.metal_dtype, scale_dtype) {
        (MetalDtype::F16, S::F16) => "norm_add_scalar_mul_f16_s_f16_specialized",
        (MetalDtype::Bf16, S::F16) => "norm_add_scalar_mul_bf16_s_f16_specialized",
        (MetalDtype::F16, S::Bf16) => "norm_add_scalar_mul_f16_s_bf16_specialized",
        (MetalDtype::Bf16, S::Bf16) => "norm_add_scalar_mul_bf16_s_bf16_specialized",
        (dt, sdt) => unreachable!(
            "norm_add_scalar_mul_kernel_static_name: (dtype={dt:?},              scale_dtype={sdt:?}) not instantiated"
        ),
    }
}

/// `MetalStep::RopeAppendNormed` symbol — same dtype enumeration as
/// the rmsnorm/fused_add_rmsnorm families (T_scale = on-disk gain
/// dtype), `rope` library.
fn rope_append_normed_kernel_static_name(
    p: &MetalModelConsts,
    scale_dtype: ScaleDtype,
) -> &'static str {
    use ScaleDtype as S;
    match (p.metal_dtype, scale_dtype) {
        (MetalDtype::F16, S::F16) => "rope_append_normed_f16_s_f16_specialized",
        (MetalDtype::Bf16, S::F16) => "rope_append_normed_bf16_s_f16_specialized",
        (MetalDtype::F16, S::Bf16) => "rope_append_normed_f16_s_bf16_specialized",
        (MetalDtype::Bf16, S::Bf16) => "rope_append_normed_bf16_s_bf16_specialized",
        (dt, sdt) => unreachable!(
            "rope_append_normed_kernel_static_name: (dtype={dt:?}, \
             scale_dtype={sdt:?}) not instantiated"
        ),
    }
}

/// Format the kernel symbol name for an `AffineEmbed` lowering.
/// Matches the `DEFINE_AFFINE_EMBED_B{4,8}` macro invocations in
/// `shaders/quantized_dequantize.metal`
/// (`affine_embed_<dtype>_s_<scale_dtype>_gs_<gs>_b_<bits>`). bits=8 is
/// for MLX-native mixed/dynamic quant (OptiQ) 8-bit embeddings.
fn affine_embed_kernel_static_name(
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    group_size: u32,
    bits: u32,
) -> &'static str {
    assert!(
        matches!(group_size, 32 | 64 | 128),
        "affine_embed_kernel_static_name: unsupported group_size={group_size} — only 32/64/128"
    );
    assert!(
        matches!(bits, 4 | 8),
        "affine_embed_kernel_static_name: unsupported bits={bits} — only 4/8 instantiated"
    );
    let (d, s) = (dequant_infix(dtype), scale_infix(scale_dtype));
    leak_symbol(format!("affine_embed_{d}_s_{s}_gs_{group_size}_b_{bits}"))
}

// ────────────────────────────────────────────────────────────────────
// Metal MoE lowering (`MetalStep::Moe`)
// ────────────────────────────────────────────────────────────────────

/// Per-bucket MoE scratch layout. Each region is 256-byte aligned
/// (Apple Silicon `MTLBuffer.offset` alignment for general buffer
/// bindings). Offsets are stamped into `Binding::MoeScratch` on the
/// emitted commands; the worker (§3a) allocates one shared
/// `moe_scratch` buffer of `total` bytes and binds at the offsets.
#[derive(Clone, Copy, Debug)]
struct MoeScratchLayout {
    router_logits: u32,
    sorted_full: u32,
    topk_inds: u32,
    topk_scores: u32,
    gate_out: u32,
    up_out: u32,
    down_out: u32,
    // Grouped-GEMM prefill regions (all 0 when `grouped == false` — then
    // the layout is byte-identical to the matvec path). `mpad_max` =
    // bucket_m*top_k + (BM-1)*num_experts is the static padded-row bound.
    grp_count: u32,       // [num_experts] u32   histogram
    grp_offset: u32,      // [num_experts] u32   padded exclusive-scan
    grp_total: u32,       // [1] u32             actual padded row count
    grp_fill: u32,        // [num_experts] u32   scatter running counters
    grp_pos: u32,         // [bucket_m*top_k] u32  pair → padded row
    grp_indices_pad: u32, // [mpad_max] u32      per-row expert (sentinel-filled)
    grp_x_pad: u32,       // [mpad_max, hidden]  gathered expert input
    grp_gate_pad: u32,    // [mpad_max, moe_inter] (act reuses this in place)
    grp_up_pad: u32,      // [mpad_max, moe_inter]
    grp_down_pad: u32,    // [mpad_max, hidden]
    mpad_max: u32,
    total: u32,
}

fn align_256(n: u32) -> u32 {
    (n + 255) & !255
}

/// The gemma4 MoE grouped expert GEMM (sort tokens by expert → batched
/// per-expert GEMM) is the prefill-speed path — mlx/vLLM both use it for
/// prefill. Enabled for prefill buckets (`bucket_m*top_k >= 64`,
/// mlx-lm's SwitchGLU `_gather_sort` threshold — below it the
/// per-(token,expert) matvec is fine, so decode keeps the matvec).
/// Validated on gemma-4-26b-a4b (correct output + 1.9-3.3x faster
/// prefill).
fn moe_grouped_enabled(bucket_m: u32, top_k: u32) -> bool {
    bucket_m * top_k >= 64
}

/// Per-bucket Gated-DeltaNet scratch layout. The four GDN sub-commands
/// pass f32 intermediates between each other through the bucket's
/// shared `moe_scratch` buffer (reused as generic op-scratch — a GDN
/// layer and a MoE layer never run concurrently in the serialized
/// tape, so the single buffer, sized to the max of both layouts, is
/// safe). conv_out / g / beta / o are all f32; the final `core` lands
/// in the model-dtype arena `out_slot`, not here. Regions are
/// 256-byte aligned (Apple Silicon `MTLBuffer.offset` alignment).
#[derive(Clone, Copy, Debug)]
struct GdnScratchLayout {
    conv_out: u32, // [bucket_m, conv_dim] f32  (conv1d out → scan in)
    g: u32,        // [bucket_m, nv]       f32  (gating out → scan in)
    beta: u32,     // [bucket_m, nv]       f32  (gating out → scan in)
    o: u32,        // [bucket_m, value_dim] f32 (scan out → rms in)
    total: u32,
}

impl GdnScratchLayout {
    fn compute(bucket_m: u32, conv_dim: u32, nv: u32, value_dim: u32) -> Self {
        const F32: u32 = 4;
        let mut off = 0u32;
        let conv_out = off;
        off = align_256(off + bucket_m * conv_dim * F32);
        let g = off;
        off = align_256(off + bucket_m * nv * F32);
        let beta = off;
        off = align_256(off + bucket_m * nv * F32);
        let o = off;
        off = align_256(off + bucket_m * value_dim * F32);
        Self {
            conv_out,
            g,
            beta,
            o,
            total: off,
        }
    }
}

impl MoeScratchLayout {
    fn compute(
        bucket_m: u32,
        num_experts: u32,
        top_k: u32,
        moe_inter: u32,
        hidden: u32,
        elem_size: u32,
        grouped: bool,
    ) -> Self {
        let mut off = 0u32;
        let router_logits = off;
        off = align_256(off + bucket_m * num_experts * elem_size);
        let sorted_full = off;
        off = align_256(off + bucket_m * num_experts * 4);
        let topk_inds = off;
        off = align_256(off + bucket_m * top_k * 4);
        let topk_scores = off;
        off = align_256(off + bucket_m * top_k * elem_size);
        let gate_out = off;
        off = align_256(off + bucket_m * top_k * moe_inter * elem_size);
        let up_out = off;
        off = align_256(off + bucket_m * top_k * moe_inter * elem_size);
        let down_out = off;
        off = align_256(off + bucket_m * top_k * hidden * elem_size);
        // Grouped-GEMM prefill regions (only when enabled). Pad to BM=64
        // (the NAX m-tile; also a multiple of the steel BM=32).
        const BM: u32 = 64;
        let mpad_max = if grouped {
            bucket_m * top_k + (BM - 1) * num_experts
        } else {
            0
        };
        let mut grp = [0u32; 10];
        if grouped {
            let sizes = [
                num_experts * 4,                  // count
                num_experts * 4,                  // offset
                4,                                // total
                num_experts * 4,                  // fill
                bucket_m * top_k * 4,             // pos
                mpad_max * 4,                     // indices_pad
                mpad_max * hidden * elem_size,    // x_pad
                mpad_max * moe_inter * elem_size, // gate_pad
                mpad_max * moe_inter * elem_size, // up_pad
                mpad_max * hidden * elem_size,    // down_pad
            ];
            for i in 0..10 {
                grp[i] = off;
                off = align_256(off + sizes[i]);
            }
        }
        Self {
            router_logits,
            sorted_full,
            topk_inds,
            topk_scores,
            gate_out,
            up_out,
            down_out,
            grp_count: grp[0],
            grp_offset: grp[1],
            grp_total: grp[2],
            grp_fill: grp[3],
            grp_pos: grp[4],
            grp_indices_pad: grp[5],
            grp_x_pad: grp[6],
            grp_gate_pad: grp[7],
            grp_up_pad: grp[8],
            grp_down_pad: grp[9],
            mpad_max,
            total: off,
        }
    }
}

fn softmax_precise_symbol(p: &MetalModelConsts) -> &'static str {
    match p.metal_dtype {
        MetalDtype::F16 => "block_softmax_precise_float16",
        MetalDtype::Bf16 => "block_softmax_precise_bfloat16",
        MetalDtype::Int4 => panic!(),
    }
}

fn take_along_axis_symbol(p: &MetalModelConsts) -> &'static str {
    match p.metal_dtype {
        MetalDtype::F16 => "take_along_axis_2d_contig_float16",
        MetalDtype::Bf16 => "take_along_axis_2d_contig_bfloat16",
        MetalDtype::Int4 => panic!(),
    }
}

fn moe_weighted_sum_symbol(p: &MetalModelConsts) -> &'static str {
    match p.metal_dtype {
        MetalDtype::F16 => "moe_weighted_sum_float16",
        MetalDtype::Bf16 => "moe_weighted_sum_bfloat16",
        MetalDtype::Int4 => panic!(),
    }
}

/// `c_arg_block_sort_<dtype>_uint32_bn<bn>_tn4` symbol picker for
/// MoE router argsort. `bn=32` (N_PER_BLOCK=128) covers Mixtral E=8,
/// Qwen2-MoE E=60, Qwen3-MoE E=128; Qwen3.5-MoE E=256 needs the
/// `bn=64` (N_PER_BLOCK=256) instantiation. The router input is
/// router-probs (Qwen) or router-logits (Mixtral), both in
/// p.metal_dtype.
fn argpartition_symbol(p: &MetalModelConsts, num_experts: u32) -> &'static str {
    match (p.metal_dtype, num_experts > 128) {
        (MetalDtype::F16, false) => "c_arg_block_sort_float16_uint32_bn32_tn4",
        (MetalDtype::Bf16, false) => "c_arg_block_sort_bfloat16_uint32_bn32_tn4",
        (MetalDtype::F16, true) => "c_arg_block_sort_float16_uint32_bn64_tn4",
        (MetalDtype::Bf16, true) => "c_arg_block_sort_bfloat16_uint32_bn64_tn4",
        (MetalDtype::Int4, _) => {
            panic!("argpartition_symbol: MoE router argsort over int4 dtype is nonsensical")
        }
    }
}

/// Memoize + leak a formatted kernel symbol as `&'static str` (the type
/// `LoweredCommand.function` requires). Distinct names are bounded by the
/// (dtype, scale, gs, bits, …) cross-product, so the leak is a handful of
/// small strings per process — the same convention as
/// `quantized::qmv_kernel_static_name`.
fn leak_symbol(name: String) -> &'static str {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().expect("leak_symbol cache poisoned");
    if let Some(&v) = guard.get(&name) {
        return v;
    }
    let leaked: &'static str = Box::leak(name.clone().into_boxed_str());
    guard.insert(name, leaked);
    leaked
}

fn dequant_infix(d: DequantDtype) -> &'static str {
    match d {
        DequantDtype::F16 => "f16",
        DequantDtype::Bf16 => "bf16",
    }
}

fn scale_infix(s: ScaleDtype) -> &'static str {
    match s {
        ScaleDtype::F16 => "f16",
        ScaleDtype::Bf16 => "bf16",
    }
}

fn affine_gather_qmv_symbol(
    n_out: u32,
    k_in: u32,
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    group_size: u32,
    bits: u32,
) -> &'static str {
    assert!(
        matches!(group_size, 32 | 64 | 128),
        "affine_gather_qmv_symbol: unsupported group_size={group_size} — only 32/64/128 instantiated"
    );
    assert!(
        matches!(bits, 4 | 8),
        "affine_gather_qmv_symbol: unsupported bits={bits} — only 4/8 instantiated"
    );
    // MLX-native mixed/dynamic quant (OptiQ) ships 8-bit experts on the
    // sensitive edge layers; the `_b_{bits}` suffix selects the matching
    // gather-qmv monomorphization.
    let fast = if n_out.is_multiple_of(8) && k_in.is_multiple_of(512) {
        "_fast"
    } else {
        ""
    };
    let (d, s) = (dequant_infix(dtype), scale_infix(scale_dtype));
    leak_symbol(format!(
        "affine_gather_qmv{fast}_{d}_s_{s}_gs_{group_size}_b_{bits}"
    ))
}

/// Symbol for the MoE grouped expert GEMM (`affine_gather_qmm_t_kernel`,
/// quantized_qmm.metallib). Gemma4 ships f16 scales; bf16/f16 activation.
fn affine_gather_qmm_t_symbol(
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    group_size: u32,
    aligned_n: bool,
    bits: u32,
) -> &'static str {
    assert!(
        matches!(group_size, 32 | 64 | 128),
        "affine_gather_qmm_t_symbol: unsupported group_size={group_size} — only 32/64/128"
    );
    assert!(
        matches!(bits, 4 | 8),
        "affine_gather_qmm_t_symbol: unsupported bits={bits} — only 4/8 instantiated"
    );
    let (d, s) = (dequant_infix(dtype), scale_infix(scale_dtype));
    let aln = if aligned_n { "true" } else { "false" };
    leak_symbol(format!(
        "affine_gather_qmm_t_{d}_s_{s}_gs_{group_size}_b_{bits}_alN_{aln}_batch_0"
    ))
}

/// Symbol for the MoE grouped expert GEMM on NAX (quantized_qmm_nax).
/// Only the aligned_N=true variants — NAX is dispatched only when
/// N % 64 == 0. gs ∈ {64, 128} (NAX has no gs=32).
fn affine_gather_qmm_t_nax_symbol(
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    group_size: u32,
    bits: u32,
) -> &'static str {
    assert!(
        matches!(group_size, 64 | 128),
        "affine_gather_qmm_t_nax_symbol: unsupported group_size={group_size} — NAX gather is gs 64/128 only"
    );
    assert!(
        matches!(bits, 4 | 8),
        "affine_gather_qmm_t_nax_symbol: unsupported bits={bits} — only 4/8 instantiated"
    );
    let (d, s) = (dequant_infix(dtype), scale_infix(scale_dtype));
    leak_symbol(format!(
        "affine_gather_qmm_t_nax_{d}_s_{s}_gs_{group_size}_b_{bits}_alN_true_batch_0"
    ))
}

/// Symbol for `moe_group_scatter_<dtype>` (moe_group.metallib).
fn moe_group_scatter_symbol(dtype: MetalDtype) -> &'static str {
    match dtype {
        MetalDtype::F16 => "moe_group_scatter_float16",
        MetalDtype::Bf16 => "moe_group_scatter_bfloat16",
        _ => "moe_group_scatter_float32",
    }
}

/// Symbol for `moe_group_gather_<dtype>` (moe_group.metallib).
fn moe_group_gather_symbol(dtype: MetalDtype) -> &'static str {
    match dtype {
        MetalDtype::F16 => "moe_group_gather_float16",
        MetalDtype::Bf16 => "moe_group_gather_bfloat16",
        _ => "moe_group_gather_float32",
    }
}

/// Build a `LoweredCommand` for one of the new MoE kernels with the
/// usual fields filled in. Callers populate `bindings` + `dispatch` +
/// `constants` then pass through.
fn make_moe_command(
    kernel: KernelId,
    library: &'static str,
    function: &'static str,
    constants: Vec<ConstantValue>,
    dispatch: DispatchShape,
    bindings: Vec<Binding>,
) -> LoweredCommand {
    LoweredCommand {
        kernel,
        library,
        function,
        constants: baked(constants),
        dispatch,
        bindings: baked(bindings),
        gemm_dims: None,
    }
}

/// The bake a MoE step lowers at.
#[derive(Clone, Copy)]
struct MoeBake {
    bucket_m: u32,
    layer_offset: u32,
    /// M5 matrix accelerator: a grouped projection takes the NAX GEMM (the dominant prefill
    /// lever; the steel grouped GEMM is ~5-10x slower per call).
    is_nax: bool,
    /// How this target stores the experts' codes.
    codes: super::kernel_constants::AffineCodesTarget,
}

/// Whether a grouped bake's expert GEMMs run W4A8 on the matrix unit's int8 lane: the experts'
/// codes are stored offset-8 and the shapes fit its tiles. Every expert's run starts on a 64-row
/// boundary, so each 32-row tile is one expert's. A function of the block and the bake only, so
/// every step of the block agrees.
fn moe_w4a8(b: &MoeBlock, at: MoeBake, s: &MoeScratch) -> bool {
    let (hidden, inter) = (b.hidden.0, b.inter.0);
    s.grouped
        && at.is_nax
        && at.codes.for_bits(b.quant.bits.0) == super::kernel_constants::AffineCodes::Offset8
        && matches!(b.quant.group_size.0, 64 | 128)
        && hidden.is_multiple_of(64)
        && inter.is_multiple_of(64)
        && W4a8Tile::for_n(hidden).is_some()
        && W4a8Tile::for_n(inter).is_some()
        && s.l.mpad_max.is_multiple_of(W4A8_TILE_ROWS)
}

/// A MoE block's scratch at one bake: the layout, the router pre-norm's region past it, and
/// whether the experts group (sort tokens by expert, batched per-expert GEMM) — the declared
/// `op_abi::METAL_GROUPED_EXPERTS` at a bucket `moe_grouped_enabled` admits.
struct MoeScratch {
    l: MoeScratchLayout,
    router_normed: u32,
    grouped: bool,
}

impl MoeScratch {
    fn of(b: &MoeBlock, bucket_m: u32, p: &MetalModelConsts, moe_scratch_bytes: &mut u32) -> Self {
        let elem = elem_size_bytes(dequant_dtype_for(p));
        let groups = crate::op_abi::METAL_GROUPED_EXPERTS.contains(&b.bundle);
        let grouped = groups && moe_grouped_enabled(bucket_m, b.top_k.0);
        let (e, k, i, h) = (b.experts.0, b.top_k.0, b.inter.0, b.hidden.0);
        let l = MoeScratchLayout::compute(bucket_m, e, k, i, h, elem, grouped);
        let router_normed = align_256(l.total);
        let total = match b.input {
            RouterInput::PreNormed => align_256(router_normed + bucket_m * h * elem),
            RouterInput::Raw => l.total,
        };
        *moe_scratch_bytes = (*moe_scratch_bytes).max(total);
        Self {
            l,
            router_normed,
            grouped,
        }
    }

    /// `region` as a scratch binding at `binding_index`.
    fn at(&self, binding_index: u8, region: MoeRegion) -> Binding {
        use MoeRegion as R;
        let (l, grouped) = (&self.l, self.grouped);
        let pair = |gathered: u32, sorted: u32| if grouped { sorted } else { gathered };
        let byte_offset = match region {
            R::RouterNormed => self.router_normed,
            R::RouterLogits => l.router_logits,
            R::SortedExperts => l.sorted_full,
            R::TopKIndices => l.topk_inds,
            R::TopKScores => l.topk_scores,
            R::SortedRows => l.grp_x_pad,
            R::ExpertGate => pair(l.gate_out, l.grp_gate_pad),
            R::ExpertUp => pair(l.up_out, l.grp_up_pad),
            R::ExpertDown => pair(l.down_out, l.grp_down_pad),
            R::TokenRows => l.down_out,
        };
        scratch_at(binding_index, byte_offset)
    }
}

fn scratch_at(binding_index: u8, byte_offset: u32) -> Binding {
    Binding::MoeScratch {
        binding_index,
        byte_offset,
    }
}

fn arena_at(binding_index: u8, slot: u32) -> Binding {
    Binding::ArenaSlot {
        slot,
        binding_index,
    }
}

/// Lower one step of a MoE block: its command(s) at the block's scratch layout for this bake.
/// A gathered bake's sort and unsort emit none — their readers bind what they view.
fn lower_moe_step(
    p: &MetalModelConsts,
    b: &MoeBlock,
    step: MoeStep,
    w: RowSources<'_>,
    at: MoeBake,
    moe_scratch_bytes: &mut u32,
) -> Result<Vec<LoweredCommand>, LoweringError> {
    use crate::tape::lowered::{MScaleAxis as A, MScaling};
    use ConstantValue as C;
    use MoeRegion as R;
    use MoeStep as S;
    let MoeBake { bucket_m, .. } = at;
    let (dtype, scale_dtype) = (dequant_dtype_for(p), scale_dtype_for(p));
    let layer = |l: &LayerId| LayerId(l.0 + at.layer_offset);
    let ms = |axis| {
        let bucket_m = super::ids::BucketM(bucket_m);
        let seq_axis = None;
        Some(MScaling {
            seq_axis,
            axis,
            bucket_m,
        })
    };
    let rows_1d = |threads: u32, per: u32, axis| {
        let mut d = DispatchShape::dispatch_1d(threads, per);
        d.m_scaling = ms(axis);
        d
    };
    let grid = |threadgroups, threads_per_threadgroup, m_scaling| DispatchShape {
        threadgroups,
        threads_per_threadgroup,
        m_scaling,
    };
    // Per token, `k` columns wide: the top-k kernels' shape.
    let per_k = |k: u32| {
        grid(
            (k.div_ceil(k.min(32)), bucket_m, 1),
            (k.min(32), 1, 1),
            ms(A::Y),
        )
    };
    let rows_of = |s: &MoeScratch, i: u8, rows: MoeRows| match rows {
        MoeRows::Tokens(Slot(slot)) => arena_at(i, slot),
        MoeRows::Scratch(r) => s.at(i, r),
    };
    let cmd = make_moe_command;
    let s = MoeScratch::of(b, bucket_m, p, moe_scratch_bytes);
    let (e, k, inter, hidden) = (b.experts.0, b.top_k.0, b.inter.0, b.hidden.0);
    let pairs = bucket_m * k;
    let router = || w.of(b.router.weight_kind(), 0);
    Ok(match step {
        // Plain RMSNorm by the router gain: Gemma's `(1 + w)` offset does not apply to it.
        S::RouterNorm(Slot(x), l, eps) => vec![LoweredCommand {
            kernel: KernelId::RmsNorm,
            library: "rmsnorm",
            function: rmsnorm_kernel_static_name(p, scale_dtype),
            constants: super::kernel_constants::RmsNormConstants {
                bucket_m: super::ids::BucketM(bucket_m),
                q_size: super::ids::QSize(hidden),
                rms_norm_eps: super::ids::RmsNormEps(eps.0),
                weight_offset: 0.0,
            }
            .into_baked(),
            dispatch: grid((bucket_m, 1, 1), (THREADS_PER_GROUP, 1, 1), ms(A::X)),
            bindings: baked(vec![
                s.at(0, R::RouterNormed),
                arena_at(1, x),
                source(router()?, WeightTensor::GemmaRouterScale, layer(&l), 2),
            ]),
            gemm_dims: None,
        }],
        S::RouterLogits(rows, l) => {
            let tiles = (bucket_m.div_ceil(GEMM_TILE_M), e.div_ceil(GEMM_TILE_N), 1);
            let gate = crate::op_abi::router_gate(b.router);
            vec![LoweredCommand {
                kernel: KernelId::Gemm,
                library: "",
                function: "",
                constants: &[],
                dispatch: grid(tiles, (GEMM_TILE_M, GEMM_TILE_N, 1), ms(A::X)),
                bindings: baked(vec![
                    s.at(0, R::RouterLogits),
                    rows_of(&s, 1, rows),
                    source(router()?, gate, layer(&l), 2),
                ]),
                gemm_dims: Some(GemmDims {
                    m: bucket_m,
                    n: e,
                    k: hidden,
                }),
            }]
        }
        S::Softmax(scores) => {
            let (region, width) = match scores {
                MoeScores::Router => (R::RouterLogits, e),
                MoeScores::TopK => (R::TopKScores, k),
            };
            let bindings = vec![s.at(0, region), s.at(1, region), inline_at(2, [width])[0]];
            let shape = grid((bucket_m, 1, 1), (256, 1, 1), ms(A::X));
            vec![cmd(
                KernelId::Softmax,
                "softmax",
                softmax_precise_symbol(p),
                vec![],
                shape,
                bindings,
            )]
        }
        // A full per-row sort; bn*tn = 128 covers E ≤ 128, E = 256 takes bn = 64.
        S::Argsort => {
            let bn = if e > 128 { 64 } else { 32 };
            let mut bindings = vec![s.at(0, R::RouterLogits), s.at(1, R::SortedExperts)];
            bindings.extend(inline_at(2, [e, 1, 1, e, e]));
            let shape = grid((1, bucket_m, 1), (bn, 1, 1), ms(A::Y));
            let symbol = argpartition_symbol(p, e);
            vec![cmd(
                KernelId::ArgPartitionTopK,
                "argpartition",
                symbol,
                vec![],
                shape,
                bindings,
            )]
        }
        S::TopK => {
            let mut bindings = vec![s.at(0, R::SortedExperts), s.at(1, R::TopKIndices)];
            bindings.extend(inline_at(2, [e, k]));
            let (kernel, library) = (KernelId::SliceTrailingColsU32, "slice_trailing_cols");
            vec![cmd(
                kernel,
                library,
                "slice_trailing_cols_u32",
                vec![],
                per_k(k),
                bindings,
            )]
        }
        S::GatherScores => {
            let (from, at_inds) = (s.at(0, R::RouterLogits), s.at(1, R::TopKIndices));
            let mut bindings = vec![from, at_inds, s.at(2, R::TopKScores)];
            bindings.extend(inline_at(3, [e, k]));
            let symbol = take_along_axis_symbol(p);
            let kernel = KernelId::TakeAlongAxis;
            vec![cmd(
                kernel,
                "take_along_axis",
                symbol,
                vec![],
                per_k(k),
                bindings,
            )]
        }
        S::Scale(scale) => {
            let (f16, bf16) = ("scalar_mul_f16_specialized", "scalar_mul_bf16_specialized");
            let symbol = pick_specialized_symbol(f16, bf16, p.metal_dtype);
            let bindings = vec![s.at(0, R::TopKScores), s.at(1, R::TopKScores)];
            let shape = rows_1d(pairs, THREADS_PER_GROUP, A::X);
            let constants = vec![C::float(2, scale.0)];
            vec![cmd(
                KernelId::ScalarMul,
                "elementwise",
                symbol,
                constants,
                shape,
                bindings,
            )]
        }
        S::Renorm => {
            let symbol = match p.metal_dtype {
                MetalDtype::F16 => "topk_renorm_float16",
                MetalDtype::Bf16 => "topk_renorm_bfloat16",
                MetalDtype::Int4 => panic!("topk_renorm: int4 unreachable"),
            };
            let (scores, width) = (R::TopKScores, inline_at(2, [k])[0]);
            let bindings = vec![s.at(0, scores), s.at(1, scores), width];
            let shape = grid((bucket_m, 1, 1), (256, 1, 1), ms(A::X));
            vec![cmd(
                KernelId::Softmax,
                "softmax",
                symbol,
                vec![],
                shape,
                bindings,
            )]
        }
        S::ExpertScale(l) => {
            let scale = source(router()?, WeightTensor::GemmaPerExpertScale, layer(&l), 2);
            let bindings = vec![s.at(0, R::TopKScores), s.at(1, R::TopKIndices), scale];
            let (kernel, library) = (KernelId::MoePerExpertScale, "moe_per_expert_scale");
            let shape = rows_1d(pairs, THREADS_PER_GROUP, A::X);
            let symbol = moe_per_expert_scale_symbol(p);
            vec![cmd(
                kernel,
                library,
                symbol,
                vec![C::uint(0, pairs)],
                shape,
                bindings,
            )]
        }
        // Grouped: histogram + padded-offset scan, sentinel fill, scatter each (token, expert)
        // row to its padded slot. Gathered: nothing — the projections read the token rows.
        S::Sort(Slot(x)) if s.grouped => {
            let (mpad, one) = (s.l.mpad_max as i32, (1, 1, 1));
            let offsets = vec![C::int(0, pairs as i32), C::int(1, e as i32)];
            let init = vec![C::int(1, e as i32), C::int(2, mpad)];
            let scatter = [0, 1, 2, 3, 4].into_iter().zip([pairs, e, s.l.mpad_max, k, hidden]);
            let scatter = scatter.map(|(i, v)| C::int(i, v as i32)).collect();
            let (l, lib) = (&s.l, "moe_group");
            // W4A8: the bucket's token rows are quantized once, into the `down_out` region
            // (unused until the unsort), and scattered as int8 rows + scales into `x_pad` — not
            // once per expert copy.
            let w4a8 = moe_w4a8(b, at, &s);
            let quant = w4a8.then(|| {
                cmd(
                    KernelId::AffineW4a8Quant,
                    "quantized_qmm_nax",
                    w4a8_quant_static_name(W4a8Rows::Dense, dtype),
                    vec![C::int(0, hidden as i32), C::int(2, bucket_m as i32)],
                    grid(((hidden / 64).div_ceil(16), bucket_m, 1), (128, 1, 1), ms(A::Y)),
                    vec![arena_at(0, x), scratch_at(1, l.down_out)],
                )
            });
            let (scatter_kernel, scatter_symbol, scatter_src, scatter_threads) = if w4a8 {
                (
                    KernelId::MoeGroupScatterQ8,
                    "moe_group_scatter_q8",
                    scratch_at(2, l.down_out),
                    (hidden / 16).min(256),
                )
            } else {
                (
                    KernelId::MoeGroupScatter,
                    moe_group_scatter_symbol(p.metal_dtype),
                    arena_at(2, x),
                    hidden.min(256),
                )
            };
            let at = |i, off| scratch_at(i, off);
            quant.into_iter().chain([
                cmd(
                    KernelId::MoeGroupOffsets,
                    lib,
                    "moe_group_offsets",
                    offsets,
                    grid(one, (256, 1, 1), None),
                    vec![
                        s.at(0, R::TopKIndices),
                        at(1, l.grp_count),
                        at(2, l.grp_offset),
                        at(3, l.grp_total),
                    ],
                ),
                cmd(
                    KernelId::MoeGroupInit,
                    lib,
                    "moe_group_init",
                    init,
                    grid((l.mpad_max.div_ceil(256), 1, 1), (256, 1, 1), None),
                    vec![at(0, l.grp_indices_pad), at(1, l.grp_fill)],
                ),
                cmd(
                    scatter_kernel,
                    lib,
                    scatter_symbol,
                    scatter,
                    grid((1, pairs, 1), (scatter_threads, 1, 1), ms(A::Y)),
                    vec![
                        s.at(0, R::TopKIndices),
                        at(1, l.grp_offset),
                        scatter_src,
                        at(3, l.grp_fill),
                        at(4, l.grp_pos),
                        at(5, l.grp_indices_pad),
                        s.at(6, R::SortedRows),
                    ],
                ),
            ])
            .collect()
        }
        S::Sort(_) => vec![],
        S::ExpertMatmul(rows, l, proj, AffineGroupSize(gs), width) => {
            let (n_out, k_in) = match proj {
                ExpertProj::Down => (hidden, inter),
                ExpertProj::Gate | ExpertProj::Up => (inter, hidden),
            };
            let out = match proj {
                ExpertProj::Gate => R::ExpertGate,
                ExpertProj::Up => R::ExpertUp,
                ExpertProj::Down => R::ExpertDown,
            };
            let ix = w.of(b.bundle.weight_kind(), 0)?;
            let [tw, ts, tb] = crate::op_abi::expert_tensors(proj);
            let lw = layer(&l);
            let weights = [
                source(ix, tw, lw, 0),
                source(ix, ts, lw, 1),
                source(ix, tb, lw, 2),
            ];
            let bits = width.bits().0;
            let codes = at.codes.for_bits(bits);
            let dims: Vec<ConstantValue> = [C::int(0, k_in as i32), C::int(1, n_out as i32)]
                .into_iter()
                .chain(codes.constant())
                .collect();
            let w4a8_tile = W4a8Tile::for_n(n_out).filter(|_| moe_w4a8(b, at, &s));
            if let Some(tile) = w4a8_tile {
                // y[Mpad, n_out] = gather_qmm_w4a8(x_pad as int8 rows + scales, W, indices_pad):
                // gate and up read the token rows the sort scattered as int8; down quantizes its
                // activation over the padded rows into `x_pad` first (sentinel rows skipped).
                let mpad = s.l.mpad_max;
                let x_pad = scratch_at(3, s.l.grp_x_pad);
                let quant = (proj == ExpertProj::Down).then(|| {
                    cmd(
                        KernelId::AffineGatherW4a8Quant,
                        "quantized_qmm_nax",
                        w4a8_quant_static_name(W4a8Rows::Grouped, dtype),
                        vec![
                            C::int(0, inter as i32),
                            C::int(2, mpad as i32),
                            C::int(4, e as i32),
                        ],
                        grid(((inter / 64).div_ceil(16), mpad, 1), (128, 1, 1), None),
                        vec![
                            s.at(0, R::ExpertGate),
                            scratch_at(1, s.l.grp_x_pad),
                            scratch_at(2, s.l.grp_indices_pad),
                        ],
                    )
                });
                let constants = [
                    C::int(0, k_in as i32),
                    C::int(1, n_out as i32),
                    C::int(2, mpad as i32),
                    C::int(4, e as i32),
                ]
                .into_iter()
                .chain(codes.constant())
                .collect();
                let mut bindings = weights.to_vec();
                bindings.extend([x_pad, s.at(4, out), scratch_at(5, s.l.grp_indices_pad)]);
                let gemm = cmd(
                    KernelId::AffineGatherQmmW4a8,
                    "quantized_qmm_nax",
                    qmm_w4a8_static_name(W4a8Rows::Grouped, dtype, scale_dtype, gs, tile),
                    constants,
                    grid(
                        (n_out / tile.cols(), mpad / W4A8_TILE_ROWS, 1),
                        (32 * tile.simdgroups(), 1, 1),
                        None,
                    ),
                    bindings,
                );
                return Ok(quant.into_iter().chain([gemm]).collect());
            }
            if s.grouped {
                // y[Mpad, n_out] = gather_qmm(x_pad, W, indices_pad); the host padded to BM = 64.
                let use_nax = at.is_nax && n_out.is_multiple_of(64) && matches!(gs, 64 | 128);
                let (kernel, library, symbol, tile) = if use_nax {
                    let symbol = affine_gather_qmm_t_nax_symbol(dtype, scale_dtype, gs, bits);
                    (
                        KernelId::AffineGatherQmmTNax,
                        "quantized_qmm_nax",
                        symbol,
                        64,
                    )
                } else {
                    let aligned = n_out.is_multiple_of(32);
                    let symbol = affine_gather_qmm_t_symbol(dtype, scale_dtype, gs, aligned, bits);
                    (KernelId::AffineGatherQmmT, "quantized_qmm", symbol, 32)
                };
                let mpad = s.l.mpad_max;
                let mut constants = dims;
                constants.extend([C::int(2, mpad as i32), C::int(4, e as i32)]);
                let x = match rows {
                    MoeRows::Tokens(_) => s.at(3, R::SortedRows),
                    MoeRows::Scratch(r) => s.at(3, r),
                };
                let mut bindings = weights.to_vec();
                let indices = scratch_at(5, s.l.grp_indices_pad);
                bindings.extend([x, s.at(4, out), indices]);
                let tiles = (n_out.div_ceil(tile), mpad.div_ceil(tile), 1);
                vec![cmd(
                    kernel,
                    library,
                    symbol,
                    constants,
                    grid(tiles, (32, 2, 2), None),
                    bindings,
                )]
            } else {
                // Token rows are read once per chosen expert (`k` pairs a row); pair rows once.
                let per_row = match rows {
                    MoeRows::Tokens(_) => k,
                    MoeRows::Scratch(_) => 1,
                };
                let symbol = affine_gather_qmv_symbol(n_out, k_in, dtype, scale_dtype, gs, bits);
                let kernel = match symbol.contains("_fast_") {
                    true => KernelId::AffineGatherQmvFast,
                    false => KernelId::AffineGatherQmv,
                };
                let mut bindings = weights.to_vec();
                let (x, indices) = (rows_of(&s, 3, rows), s.at(4, R::TopKIndices));
                bindings.extend([x, indices, s.at(5, out), inline_at(6, [per_row])[0]]);
                let shape = grid((1, n_out.div_ceil(8), pairs), (32, 2, 1), ms(A::Z));
                vec![cmd(kernel, "quantized_qmv", symbol, dims, shape, bindings)]
            }
        }
        // `out = act(gate) * up`, written over the gate rows; the grouped bake runs every
        // padded row.
        S::GatedAct(act) => {
            let (kernel, symbol) = match act {
                GatedAct::Silu => (KernelId::SiluMul, silu_mul_static_name(dtype)),
                GatedAct::Gelu => (KernelId::GeluMul, gelu_mul_static_name(dtype)),
            };
            let (rows, m_scaling) = match s.grouped {
                true => (s.l.mpad_max, None),
                false => (pairs, ms(A::X)),
            };
            let n = rows * inter;
            let shape = grid((n.div_ceil(256), 1, 1), (256, 1, 1), m_scaling);
            let gate = |i| s.at(i, R::ExpertGate);
            let bindings = vec![gate(0), gate(1), s.at(2, R::ExpertUp)];
            vec![cmd(
                kernel,
                "silu_mul",
                symbol,
                vec![C::uint(0, n)],
                shape,
                bindings,
            )]
        }
        S::Unsort if s.grouped => {
            let symbol = moe_group_gather_symbol(p.metal_dtype);
            let (from, pos) = (s.at(0, R::ExpertDown), scratch_at(1, s.l.grp_pos));
            let bindings = vec![from, pos, s.at(2, R::TokenRows)];
            let shape = grid((1, pairs, 1), (hidden.min(256), 1, 1), ms(A::Y));
            let kernel = KernelId::MoeGroupGather;
            vec![cmd(
                kernel,
                "moe_group",
                symbol,
                vec![C::int(5, hidden as i32)],
                shape,
                bindings,
            )]
        }
        S::Unsort => vec![],
        // `out[n, d] = Σ_k rows[n, k, d] · scores[n, k]`.
        S::Combine(Slot(out)) => {
            let threads = hidden.min(64);
            let shape = grid(
                (hidden.div_ceil(threads), bucket_m, 1),
                (threads, 1, 1),
                ms(A::Y),
            );
            let (rows, scores) = (s.at(0, R::TokenRows), s.at(1, R::TopKScores));
            let bindings = vec![rows, scores, arena_at(2, out)];
            let constants = vec![C::int(0, k as i32), C::int(1, hidden as i32)];
            let symbol = moe_weighted_sum_symbol(p);
            let kernel = KernelId::MoeWeightedSum;
            vec![cmd(
                kernel,
                "moe_weighted_sum",
                symbol,
                constants,
                shape,
                bindings,
            )]
        }
    })
}

/// `moe_per_expert_scale_{float16,bfloat16}` symbol picker.
fn moe_per_expert_scale_symbol(p: &MetalModelConsts) -> &'static str {
    match p.metal_dtype {
        MetalDtype::F16 => "moe_per_expert_scale_float16",
        MetalDtype::Bf16 => "moe_per_expert_scale_bfloat16",
        MetalDtype::Int4 => panic!("moe_per_expert_scale: int4 unreachable"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scratchy_ir::CanonicalParams;

    fn tp() -> MetalModelConsts {
        MetalModelConsts::from_canonical::<TestParams>()
    }

    /// The code width the TurboQuant fixture is built at.
    const TQ_BITS: TqBits = TqBits::new(4);

    /// `tp()` built with the `turboquant` feature: its KV codec is TurboQuant.
    fn tq_consts() -> MetalModelConsts {
        MetalModelConsts {
            kv_codec: KvCodec::TurboQuant(TQ_BITS),
            ..tp()
        }
    }
    use AttnMask::{Causal, SlidingWindow};
    use RopeFormTag::{Interleaved, NeoX};

    /// Minimal `CanonicalParams` impl for lowering-shape tests. No
    /// kernel actually runs — `lower_one` just inspects the variant
    /// fields and `p.metal_dtype`. Pinned to bf16 to match the
    /// canonical Llama-3.x configuration the macro emits.
    struct TestParams;
    impl CanonicalParams for TestParams {
        const HEAD_DIM: u32 = 64;
        const NUM_Q_HEADS: u32 = 32;
        const NUM_KV_HEADS: u32 = 4;
        const Q_SIZE: usize = 2048;
        // Residual-stream width. For this Llama-1B-like shape hidden==Q_SIZE.
        // AffineEmbed (and other residual-width arms) read `HIDDEN_SIZE`, so it
        // must be pinned here — the trait default of 0 would zero those dispatches.
        const HIDDEN_SIZE: usize = 2048;
        const KV_SIZE: usize = 256;
        const INTERMEDIATE_SIZE: usize = 8192;
        const ATTN_SCALE: f32 = 0.125;
        const ATTN_SOFTCAP: f32 = 0.0;
        const SLIDING_WINDOW: i32 = -1;
        const KV_LORA_RANK: usize = 0;
        const QK_NOPE_HEAD_DIM: usize = 0;
        const QK_ROPE_HEAD_DIM: usize = 0;
        const V_HEAD_DIM: usize = 0;
        const FINAL_LOGIT_SOFTCAPPING: f32 = 0.0;
        const QK_HEAD_DIM: usize = 0;
        const MLA_ATTN_SCALE: f32 = 0.0;
        const METAL_DTYPE: MetalDtype = MetalDtype::Bf16;
    }
    // `WeightAccessors` is a supertrait of `CanonicalParams`; every method
    // has a default `unreachable!` body, and the lowering-shape tests never
    // resolve a weight source (resolution is the pool's, at load), so the
    // empty impl suffices.
    impl scratchy_ir::WeightAccessors for TestParams {}

    /// A Llama layer's KV writer operands: bias-free projections.
    const LLAMA_KV: scratchy_ir::KvOffsets = scratchy_ir::KvOffsets {
        k: scratchy_ir::KvOffset::Centered,
        v: scratchy_ir::KvOffset::Centered,
    };

    /// A layer's KV writer whose K and V carry `offsets`.
    fn tq_writer(layer: u32, class: AttnMask, offsets: scratchy_ir::KvOffsets) -> MetalStep {
        let [q, k, v, q_out, k_out, v_out] = [0, 1, 2, 3, 4, 5].map(Slot);
        MetalStep::RopeAppend(
            q,
            k,
            v,
            q_out,
            k_out,
            v_out,
            LayerId(layer),
            NeoX,
            class,
            offsets,
        )
    }

    /// `layer`'s attention of `kind`, from q slot 3 into slot 6.
    fn attention(
        kind: fn(Slot, Slot, LayerId, RopeFormTag) -> MetalStep,
        layer: u32,
        pairing: RopeFormTag,
    ) -> MetalStep {
        kind(Slot(3), Slot(6), LayerId(layer), pairing)
    }

    /// The weight kinds a metal step binds.
    const KINDS: [WeightKind; 11] = [
        WeightKind::Embedding,
        WeightKind::AffineQuantEmbedding,
        WeightKind::RmsNorm,
        WeightKind::LayerNorm,
        WeightKind::Linear,
        WeightKind::CosSin,
        WeightKind::FusedMoe,
        WeightKind::SharedFusedMoe,
        WeightKind::GemmaRouter,
        WeightKind::GemmaSwitchGlu,
        WeightKind::GatedDeltaNet,
    ];

    /// The source every test row's site names as its `slot`-th `kind` weight: `10·k + slot`, so a
    /// bound `SourceIx` says which `(kind, slot)` the lowering asked for.
    fn src(kind: WeightKind, slot: u32) -> SourceIx {
        let k = KINDS.iter().position(|x| *x == kind).expect("a metal kind");
        SourceIx(10 * k as u32 + slot)
    }

    /// Every test row's site: two sources of each kind.
    static TEST_SITE: std::sync::LazyLock<Vec<RowSource>> = std::sync::LazyLock::new(|| {
        let two = |kind: &WeightKind| {
            let kind = kind.clone();
            (0..2).map(move |slot| RowSource {
                kind: kind.clone(),
                ix: src(kind.clone(), slot),
            })
        };
        KINDS.iter().flat_map(two).collect()
    });

    /// The tests' class rotary tables.
    const TEST_ROTARY: RotaryTables = RotaryTables {
        global: SourceIx(1000),
        sliding: SourceIx(1001),
    };

    /// A test row's sources.
    fn row() -> RowSources<'static> {
        RowSources {
            index: 0,
            site: &TEST_SITE,
            rotary: Some(TEST_ROTARY),
        }
    }

    /// `rows` as a step tape: every row fenced, every row's site [`TEST_SITE`], no lm_head.
    fn row_tape(rows: Vec<StepRow>) -> MetalStepTape {
        MetalStepTape {
            backbone_barriers: vec![true; rows.len()],
            backbone_sources: vec![TEST_SITE.clone(); rows.len()],
            backbone: rows,
            ..MetalStepTape::default()
        }
    }

    /// `steps` as rows that always run.
    fn plain(steps: &[MetalStep]) -> Vec<StepRow> {
        steps.iter().map(|s| StepRow::Step(*s, None)).collect()
    }

    /// The tests' bake point: 8 arena slots, the test rotary tables, block capacity 128.
    fn bake_point(
        bucket_m: u32,
        profile: Option<&crate::targets::MetalTargetProfile>,
    ) -> BakePoint<'_> {
        BakePoint {
            chunked: false,
            bucket_m,
            num_arena_slots: 8,
            rotary: Some(TEST_ROTARY),
            block_cap: 128,
            profile,
        }
    }

    /// A TurboQuant-coded layer's rows — `writer`, then `attention` — as the codec steps expand
    /// them from `op_abi::METAL_KV_CODEC`, each gated by its guard's `METAL_GUARD_GATES` gate: the
    /// writer's encodes after it, then the attention's decode or prefill form around it. Every
    /// codec row carries the writer's offsets, as the step records give it them.
    fn coded(writer: MetalStep, attention: MetalStep) -> Vec<StepRow> {
        use crate::op_abi::{METAL_GUARD_GATES, METAL_KV_CODEC};
        use scratchy_subtile::kv_codec::{After, Before};
        let MetalStep::RopeAppend(.., wl, _, _, offsets) = writer else {
            panic!("a coded writer: {writer:?}");
        };
        use MetalStep as S;
        let (decode, q, out, layer, pairing, class) = match attention {
            S::AttentionViaCache(q, o, l, p) => (true, q, o, l, p, Causal),
            S::SlidingAttentionViaCache(q, o, l, p) => (true, q, o, l, p, SlidingWindow),
            S::AttentionPrefillPaged(q, o, l, p) => (false, q, o, l, p, Causal),
            S::SlidingAttentionPrefillPaged(q, o, l, p) => (false, q, o, l, p, SlidingWindow),
            other => panic!("a coded attention: {other:?}"),
        };
        let gated = |step, guard| StepRow::Step(step, METAL_GUARD_GATES.gate(guard));
        let codec = METAL_KV_CODEC;
        let mut rows = plain(&[writer]);
        for g in codec.after_writer {
            rows.push(gated(S::KvEncode(g.step, wl, offsets), g.guard));
        }
        let around = if decode { codec.decode } else { codec.prefill };
        for g in around.before {
            let step = match g.step {
                Before::Stage(operand) => S::KvStage(operand, layer, class, offsets),
                Before::RotateQuery => S::RotateRows(q, RotatedRows::Query),
            };
            rows.push(gated(step, g.guard));
        }
        rows.push(gated(attention, around.anchor));
        for g in around.after {
            let step = match g.step {
                After::RotateOutput => S::RotateRows(out, RotatedRows::Output),
                After::PackedTwin => S::AttnPackedKv(q, out, layer, pairing, class, offsets),
            };
            rows.push(gated(step, g.guard));
        }
        rows
    }

    /// Lower one coded layer — the KV writer, then its attention — at `bucket_m`.
    fn lower_tq_layer(attention: MetalStep, bucket_m: u32) -> LoweredMetalTape {
        let rows = coded(tq_writer(0, Causal, LLAMA_KV), attention);
        lower_tq(&tq_consts(), rows, bucket_m)
    }

    fn lower_tq(p: &MetalModelConsts, rows: Vec<StepRow>, bucket_m: u32) -> LoweredMetalTape {
        try_lower_tq(p, rows, bucket_m).expect("lower_subtile_tape_to_metal")
    }

    fn try_lower_tq(
        p: &MetalModelConsts,
        rows: Vec<StepRow>,
        bucket_m: u32,
    ) -> Result<LoweredMetalTape, LoweringError> {
        lower_subtile_tape_to_metal(&row_tape(rows), p, bake_point(bucket_m, None))
    }

    /// A dense model's tape carries no TurboQuant command and gates nothing on
    /// the step shape: the codec pass runs only on a model built with it, so its
    /// layers lower as written. Pure-CPU lowering checks — never submit a Metal
    /// command buffer.
    #[test]
    fn dense_codec_lowers_no_turboquant_commands() {
        assert_eq!(tp().kv_codec, KvCodec::Dense);
        for (attention, bucket_m) in [
            (attention(MetalStep::AttentionViaCache, 0, Interleaved), 1),
            (attention(MetalStep::AttentionPrefillPaged, 0, NeoX), 64),
        ] {
            let rows = plain(&[tq_writer(0, Causal, LLAMA_KV), attention]);
            let tape = lower_tq(&tp(), rows, bucket_m);
            assert!(
                tape.commands.iter().all(|c| !matches!(
                    c.command.kernel,
                    KernelId::TqQuantizeToPacked
                        | KernelId::TqStageRotated
                        | KernelId::TqRotateRows
                        | KernelId::AttentionViaCacheTq
                )),
                "{attention:?}"
            );
            assert!(tape.commands.iter().all(|c| !matches!(
                c.gate,
                Some(RuntimeGate::OnlyIfDecodeStep | RuntimeGate::UnlessDecodeStep)
            )));
        }
        // A codec step reaching a dense model's lowering is refused, never lowered.
        let coded_rows = coded(
            tq_writer(0, Causal, LLAMA_KV),
            attention(MetalStep::AttentionViaCache, 0, Interleaved),
        );
        assert!(matches!(
            try_lower_tq(&tp(), coded_rows, 1),
            Err(LoweringError::CodecStepOnDenseModel)
        ));
    }

    /// A TurboQuant model whose backbone binds the KV cache but compresses
    /// none of it does not lower: a hybrid model's sliding layers alone stay
    /// fp16, so nothing would write the packed stores its factory provisions.
    #[test]
    fn turboquant_that_compresses_nothing_does_not_lower() {
        let p = MetalModelConsts {
            global_head_dim: 512,
            num_global_kv_heads: 1,
            global_block_size: 32,
            global_rot_dim: 128,
            sliding_window: 1024,
            ..tq_consts()
        };
        let rows = plain(&[
            tq_writer(1, SlidingWindow, LLAMA_KV),
            attention(MetalStep::SlidingAttentionViaCache, 1, Interleaved),
        ]);
        assert!(matches!(
            try_lower_tq(&p, rows, 1),
            Err(LoweringError::TurboQuantCompressesNothing)
        ));
    }

    /// Each TurboQuant codec command must carry the `OnlyIfTurboquant` gate
    /// so it is skipped on a non-turboquant (fp16) KV cache — otherwise the TQ
    /// kernels dispatch on fp16, read OOB from the unbound packed/norms fallback
    /// bindings, and silently WEDGE the GPU. The gate is fused onto each
    /// `GatedCommand`, so it can no longer be dropped by a parallel-vec mishap
    /// (that is now a compile error); these tests verify a codec row's guard
    /// reaches every command of it with the gate VALUE `op_abi::METAL_GUARD_GATES`
    /// declares, and `lower_subtile_tape_to_metal` carries it. Pure-CPU lowering
    /// checks — never submit a Metal command buffer.
    ///
    /// Decode: no dequant pass; the plain attention runs only off TurboQuant
    /// and its `AttentionViaCacheTq` twin — same geometry, plus the codebook
    /// width and the packed-store bindings — only on TurboQuant KV.
    #[test]
    fn decode_turboquant_reads_packed_store_without_dequant() {
        use crate::tape::lowered::RuntimeGate::{self, OnlyIfDecodeStep, UnlessDecodeStep};
        let tape = lower_tq_layer(attention(MetalStep::AttentionViaCache, 0, Interleaved), 1);
        let steps: Vec<(KernelId, Option<RuntimeGate>)> = tape
            .commands
            .iter()
            .map(|c| (c.command.kernel, c.gate))
            .collect();
        assert_eq!(
            steps,
            [
                (KernelId::RopeAppend, None),
                (KernelId::TqQuantizeToPacked, None),
                (KernelId::TqQuantizeToPacked, None),
                (KernelId::AttentionViaCache, Some(UnlessDecodeStep)),
                (KernelId::AttentionViaCacheTq, Some(OnlyIfDecodeStep)),
            ]
        );
        let (fp16, tq) = (&tape.commands[3].command, &tape.commands[4].command);
        assert_eq!(
            (fp16.library, fp16.function, fp16.dispatch),
            (tq.library, tq.function, tq.dispatch)
        );
        let bits = TQ_BITS.get();
        assert_eq!(tq.constants[..fp16.constants.len()], *fp16.constants);
        assert_eq!(
            tq.constants[fp16.constants.len()..],
            [ConstantValue::uint(13, bits), ConstantValue::uint(16, 1)]
        );
        assert_eq!(tq.bindings[..fp16.bindings.len()], *fp16.bindings);
        let layer = crate::tape::ids::LayerId(0);
        let tq_kinds: Vec<(u8, RuntimeBindingKind)> = tq.bindings[fp16.bindings.len()..]
            .iter()
            .map(|b| match b {
                Binding::Runtime {
                    kind,
                    binding_index,
                } => (*binding_index, *kind),
                other => panic!("TurboQuant attention binding {other:?} is not a runtime buffer"),
            })
            .collect();
        assert_eq!(
            tq_kinds,
            [
                (7, RuntimeBindingKind::TqPackedK { layer }),
                (8, RuntimeBindingKind::TqPackedV { layer }),
                (9, RuntimeBindingKind::TqNormsK { layer }),
                (10, RuntimeBindingKind::TqNormsV { layer }),
                (11, RuntimeBindingKind::TqSigns),
                (12, RuntimeBindingKind::TqCentroids),
                (13, RuntimeBindingKind::SlotMapping { layer }),
            ]
        );
    }

    /// The TurboQuant decode twin's query heads per threadgroup turn on the
    /// device's GPU core count, which no baked class knows: every class's
    /// profile lowers it at one head, and the pool serves the device's count
    /// at load. The test geometry's GQA group of 8 (head_dim 64) fits 8 heads;
    /// the most that leave 4/5 of the cores a threadgroup is 4 on an 8-core
    /// M1, 2 on a 16-core M1 Pro, and 1 on a 32-core M1 Max.
    #[test]
    fn decode_turboquant_heads_follow_the_device_not_the_baked_class() {
        use crate::tape::ids::{GpuCores, HeadDim, NumKvHeads, NumQHeads, TqDecodeHeads};
        let p = tq_consts();
        let rows = coded(
            tq_writer(0, Causal, LLAMA_KV),
            attention(MetalStep::AttentionViaCache, 0, Interleaved),
        );
        let find = |profile, k| {
            let tape = lower_subtile_tape_to_metal(
                &row_tape(rows.clone()),
                &p,
                bake_point(/*bucket_m=*/ 1, Some(profile)),
            )
            .expect("lower_subtile_tape_to_metal");
            tape.commands
                .iter()
                .find(|c| c.command.kernel == k)
                .expect("command")
                .command
        };
        let (fp16, tq) = (
            find(&crate::targets::M1_MAX, KernelId::AttentionViaCache),
            find(&crate::targets::M1_MAX, KernelId::AttentionViaCacheTq),
        );
        for class in [&crate::targets::M4_10CORE, &crate::targets::M5_10CORE] {
            assert!(
                find(class, KernelId::AttentionViaCacheTq) == tq,
                "one tape per class"
            );
        }
        assert_eq!(p.num_q_heads / p.num_kv_heads, 8);
        let (x, y, z) = fp16.dispatch.threadgroups;
        assert_eq!(y, p.num_q_heads);
        assert_eq!(tq.dispatch.threadgroups, (x, y, z));
        assert_eq!(tq.constants.last(), Some(&ConstantValue::uint(16, 1)));
        for (cores, heads) in [(8, 4), (16, 2), (32, 1)] {
            let served_heads = TqDecodeHeads::for_group(
                HeadDim(p.global_head_dim),
                NumQHeads(p.num_q_heads),
                NumKvHeads(p.num_global_kv_heads),
                GpuCores(cores),
            );
            assert_eq!(served_heads, TqDecodeHeads(heads), "{cores} cores");
            let (mut served, mut other) = (tq, fp16);
            serve_tq_decode_heads(&mut served, served_heads);
            serve_tq_decode_heads(&mut other, served_heads);
            assert_eq!(served.dispatch.threadgroups, (x, y / heads, z));
            assert_eq!(
                served.constants.last(),
                Some(&ConstantValue::uint(16, heads))
            );
            assert!(
                other == fp16,
                "only the TurboQuant decode command is served"
            );
        }
    }

    use crate::tape::lowered::RuntimeGate;

    /// The TurboQuant commands a multi-token tape wraps around one layer's
    /// paged attention: stage K and V and rotate q before it, rotate its output
    /// back after it (off decode steps), its decode twin (on decode steps).
    fn tq_attention_steps(
        prefill_attention: &[(KernelId, Option<RuntimeGate>)],
    ) -> Vec<(KernelId, Option<RuntimeGate>)> {
        use crate::tape::lowered::RuntimeGate::{OnlyIfDecodeStep, UnlessDecodeStep};
        // K and V staged twice: the new rows, then the cached ones.
        let mut steps = vec![
            (KernelId::TqStageRotated, Some(UnlessDecodeStep)),
            (KernelId::TqStageRotated, Some(UnlessDecodeStep)),
            (KernelId::TqStageRotated, Some(UnlessDecodeStep)),
            (KernelId::TqStageRotated, Some(UnlessDecodeStep)),
            (KernelId::TqRotateRows, Some(UnlessDecodeStep)),
        ];
        steps.extend(
            prefill_attention
                .iter()
                .map(|&(k, _)| (k, Some(UnlessDecodeStep))),
        );
        steps.push((KernelId::TqRotateRows, Some(UnlessDecodeStep)));
        steps.push((KernelId::AttentionViaCacheTq, Some(OnlyIfDecodeStep)));
        steps
    }

    /// The attention's own commands in `tape`: off decode steps, and not one
    /// of the codec's staging or rotation commands around it.
    fn own_attention_steps(tape: &LoweredMetalTape) -> Vec<(KernelId, Option<RuntimeGate>)> {
        gated_steps(tape)
            .into_iter()
            .filter(|&(k, g)| {
                g == Some(RuntimeGate::UnlessDecodeStep)
                    && !matches!(k, KernelId::TqStageRotated | KernelId::TqRotateRows)
            })
            .collect()
    }

    fn gated_steps(tape: &LoweredMetalTape) -> Vec<(KernelId, Option<RuntimeGate>)> {
        tape.commands
            .iter()
            .map(|c| (c.command.kernel, c.gate))
            .collect()
    }

    /// Hybrid arches (gemma-4) compress only the GLOBAL layers — the codec pass
    /// expands the declared class only (`kv_codec`'s
    /// `a_hybrid_arch_codes_its_declared_class_only`): the global layer's rows lower
    /// to its writer's quantize and the TurboQuant commands around its attention,
    /// while the sliding layer's rows stay plain fp16.
    #[test]
    fn hybrid_turboquant_compresses_global_layers_only() {
        use crate::tape::lowered::RuntimeGate::{OnlyIfDecodeStep, UnlessDecodeStep};
        let p = MetalModelConsts {
            global_head_dim: 512,
            num_global_kv_heads: 1,
            global_block_size: 32,
            global_rot_dim: 128,
            sliding_window: 1024,
            ..tq_consts()
        };
        let global_writer = tq_writer(0, Causal, LLAMA_KV);
        let sliding_writer = tq_writer(1, SlidingWindow, LLAMA_KV);
        let quantize = [
            (KernelId::TqQuantizeToPacked, None),
            (KernelId::TqQuantizeToPacked, None),
        ];
        let global = attention(MetalStep::AttentionViaCache, 0, Interleaved);
        let sliding = attention(MetalStep::SlidingAttentionViaCache, 1, Interleaved);
        let rows = [
            coded(global_writer, global),
            plain(&[sliding_writer, sliding]),
        ];
        let decode = lower_tq(&p, rows.concat(), 1);
        let mut want = vec![(KernelId::RopeAppend, None)];
        want.extend(quantize);
        want.extend([
            (KernelId::AttentionViaCache, Some(UnlessDecodeStep)),
            (KernelId::AttentionViaCacheTq, Some(OnlyIfDecodeStep)),
            (KernelId::RopeAppend, None),
            (KernelId::AttentionViaCache, None),
        ]);
        assert_eq!(gated_steps(&decode), want);

        let global_attention = attention(MetalStep::AttentionPrefillPaged, 0, NeoX);
        let sliding_attention = attention(MetalStep::SlidingAttentionPrefillPaged, 1, NeoX);
        // A sliding layer lowers as it would in a dense model; alone in a tape
        // it compresses nothing, which a TurboQuant model's tape may not do.
        let dense = MetalModelConsts {
            kv_codec: KvCodec::Dense,
            ..p
        };
        let fp16 = |i: MetalStep| gated_steps(&lower_tq(&dense, plain(&[i]), 64));
        // A TurboQuant'd attention compresses its layer's writer's operands, so
        // its own commands are read off a tape that has the writer.
        let own = |i: MetalStep| own_attention_steps(&lower_tq(&p, coded(global_writer, i), 64));
        let prefill = lower_tq(
            &p,
            [
                coded(global_writer, global_attention),
                plain(&[sliding_writer, sliding_attention]),
            ]
            .concat(),
            64,
        );
        let mut want = vec![(KernelId::RopeAppend, None)];
        want.extend(quantize);
        want.extend(tq_attention_steps(&own(global_attention)));
        want.push((KernelId::RopeAppend, None));
        want.extend(fp16(sliding_attention));
        assert_eq!(gated_steps(&prefill), want);
    }

    /// On NAX, an MLX-affine 4-bit GEMM in a bucket that can see a 4–16-token
    /// step gets its small-M twin — the 8-row tile in the 8-token bucket, the
    /// 16-row one in the 64-token bucket — gated against the GEMM it replaces,
    /// with the GEMM's own weights and slots. Batch 1, the prefill buckets,
    /// 8-bit weights and non-NAX devices are left as they were.
    #[test]
    fn small_m_twin_serves_decode_batches_on_nax() {
        use crate::tape::lowered::RuntimeGate::{OnlyIfSmallMTokens, UnlessSmallMTokens};
        let gemm = |bits| {
            MetalStep::AffineQmm(AffineMatmul {
                input: Slot(0),
                output: Slot(1),
                layer: LayerId(0),
                n: NDim(3072),
                k: KDim(8192),
                bits: AffineBits(bits),
                vector_limit: QmvBatchLimit(10),
                ..q_proj()
            })
        };
        let lower_at = |step, bucket_m, profile| {
            lower_subtile_tape_to_metal(
                &row_tape(plain(&[step])),
                &tp(),
                bake_point(bucket_m, profile),
            )
            .expect("lower_subtile_tape_to_metal")
        };
        let m5 = Some(&crate::targets::M5_10CORE);
        for (bucket_m, own, tile) in [
            (8, &[KernelId::AffineQmvFast][..], SmallMTile::Rows8),
            (
                64,
                &[KernelId::AffineW4a8Quant, KernelId::AffineQmmW4a8][..],
                SmallMTile::Rows16,
            ),
        ] {
            let tape = lower_at(gemm(4), bucket_m, m5);
            let mut want: Vec<_> = own.iter().map(|&k| (k, Some(UnlessSmallMTokens))).collect();
            want.push((KernelId::AffineQmmSmallM, Some(OnlyIfSmallMTokens)));
            assert_eq!(gated_steps(&tape), want);
            let (gemm_cmd, small) = (
                &tape.commands[own.len() - 1].command,
                &tape.commands[own.len()].command,
            );
            // Weights and output; the W4A8 GEMM reads its activation from
            // the pre-pass's scratch (binding 3).
            let but_input = |c: &LoweredCommand| -> Vec<Binding> {
                c.bindings
                    .iter()
                    .filter(|b| {
                        !matches!(
                            b,
                            Binding::ArenaSlot {
                                binding_index: 3,
                                ..
                            } | Binding::Scratch { binding_index: 3 }
                        )
                    })
                    .cloned()
                    .collect()
            };
            let p = tp();
            assert_eq!(
                small.function,
                small_m_kernel_static_name(dequant_dtype_for(&p), scale_dtype_for(&p), 64, tile)
            );
            assert_eq!(
                small.dispatch.threadgroups,
                (3072 / SMALL_M_TILE_COLS, bucket_m / tile.rows(), 1)
            );
            assert!(
                but_input(small) == but_input(gemm_cmd)
                    && small.bindings[..]
                        == affine_qmm_bindings(
                            0,
                            1,
                            crate::tape::ids::LayerId(0),
                            src(WeightKind::Linear, 0),
                        )[..],
                "same weights and slots"
            );
        }
        for (instruction, bucket_m, profile) in [
            (gemm(4), 1, m5),
            (gemm(4), 512, m5),
            (gemm(8), 8, m5),
            (gemm(4), 8, Some(&crate::targets::M1_8CORE)),
        ] {
            let tape = lower_at(instruction, bucket_m, profile);
            assert!(
                tape.commands
                    .iter()
                    .all(|c| c.gate.is_none() && c.command.kernel != KernelId::AffineQmmSmallM),
                "bucket {bucket_m}: no small-M twin"
            );
        }
    }

    /// Llama-1B's `q_proj` shape (N=K=2048, gs=64, 4-bit, qmv batch limit 18), slot 7 into slot 11
    /// at layer 3.
    fn q_proj() -> AffineMatmul {
        AffineMatmul {
            input: Slot(7),
            output: Slot(11),
            layer: LayerId(3),
            n: NDim(2048),
            k: KDim(2048),
            group_size: AffineGroupSize(64),
            bits: AffineBits(4),
            vector_limit: QmvBatchLimit(18),
        }
    }

    /// The lm_head half a sample-rows construct over [`q_proj`] lowers from: its four rows, each
    /// with its `flags` entry.
    fn sampled_rows(flags: [bool; 4]) -> MetalStepTape {
        use SampleRowsStep::{AllRows, Gather, Matmul, Scatter};
        let row = |s| StepRow::Step(MetalStep::SampleRows(q_proj(), s), None);
        MetalStepTape {
            lm_head: [Gather, Matmul, Scatter, AllRows].map(row).to_vec(),
            lm_head_barriers: flags.to_vec(),
            lm_head_sources: vec![TEST_SITE.clone(); 4],
            ..MetalStepTape::default()
        }
    }

    /// [`q_proj`] lowered on its own at `at`.
    fn plain_q_proj(at: BakePoint<'_>) -> LoweredMetalTape {
        let rows = row_tape(plain(&[MetalStep::AffineQmm(q_proj())]));
        lower_subtile_tape_to_metal(&rows, &tp(), at).expect("lowers")
    }

    /// Where the matmul lowers to one qmm_t command (bucket 512: Standard), the rows slice: the
    /// gather, the matmul's qmv at the sampled rows and the scatter run on a step with rows to
    /// drop and no speculative tokens; the plain matmul over every row on a speculative step.
    /// Each row's flag rides its command.
    #[test]
    fn sampled_rows_slice_where_the_matmul_is_one_qmm_t() {
        use crate::tape::lowered::MScaleAxis;
        use crate::tape::lowered::RuntimeGate::{OnlyIfNoSpec, OnlyIfSpec};
        let rows = sampled_rows([true, false, true, false]);
        let at = bake_point(512, None);
        let tape = lower_subtile_tape_to_metal(&rows, &tp(), at).expect("lowers");
        assert_eq!(
            gated_steps(&tape),
            [
                (KernelId::GatherLastToken, Some(OnlyIfNoSpec)),
                (KernelId::AffineQmvFast, Some(OnlyIfNoSpec)),
                (KernelId::ScatterFirstToLastRow, Some(OnlyIfNoSpec)),
                (KernelId::AffineQmmT, Some(OnlyIfSpec)),
            ]
        );
        assert_eq!(tape.barrier_before, [true, false, true, false]);
        // The all-rows command IS the plain matmul's.
        let plain = plain_q_proj(at);
        assert!(plain.commands.len() == 1 && tape.commands[3].command == plain.commands[0].command);
        // The gather over the matmul's input, `k` wide; the scatter over its output, `n` wide; one
        // thread per column, walking the sequences in order (the rows move in place).
        let [gather, qmv, scatter] = [0, 1, 2].map(|i| tape.commands[i].command);
        for (c, slot, width) in [(gather, 7, 2048), (scatter, 11, 2048)] {
            assert!(
                matches!(c.bindings[0], Binding::ArenaSlot { slot: s, binding_index: 0 } if s == slot)
            );
            assert_eq!(c.dispatch.threadgroups, (width / 256, 1, 1));
            assert_eq!(c.dispatch.m_scaling, None);
        }
        // The qmv: one row baked, the live sequence count at dispatch; the matmul's weight and
        // slots.
        assert_eq!(qmv.function, "affine_qmv_fast_bf16_s_f16_gs_64_b_4_batch_0");
        assert_eq!(qmv.dispatch.threadgroups, (1, 2048 / 8, 1));
        let ms = qmv.dispatch.m_scaling.expect("scales with the sequences");
        assert_eq!(ms.seq_axis, Some(MScaleAxis::X));
        let bindings = affine_qmm_bindings(7, 11, LayerId(3), src(WeightKind::Linear, 0));
        assert_eq!(qmv.bindings, bindings);
        assert_eq!(tape.splitk_scratch_bytes, 0);
    }

    /// Where the matmul does not lower to one qmm_t command it runs plain and ungated under its
    /// own row's flag, and the gather, the scatter and the all-rows matmul emit nothing: one row
    /// (bucket 1), qmv (bucket 8, under its batch limit), SplitK's pair (bucket 64), the small-M
    /// twin (bucket 64 on a NAX device).
    #[test]
    fn sampled_rows_run_plain_where_the_matmul_does_not_slice() {
        use KernelId as K;
        let m5 = Some(&crate::targets::M5_10CORE);
        let rows = sampled_rows([true, false, true, true]);
        for (bucket_m, profile, last) in [
            (1, None, K::AffineQmvFast),
            (8, None, K::AffineQmvFast),
            (64, None, K::SplitKReduceSum),
            (64, m5, K::AffineQmmSmallM),
        ] {
            let at = bake_point(bucket_m, profile);
            let tape = lower_subtile_tape_to_metal(&rows, &tp(), at).expect("lowers");
            let plain = plain_q_proj(at);
            assert!(tape.commands == plain.commands, "bucket {bucket_m}: plain");
            assert_eq!(tape.commands.last().map(|c| c.command.kernel), Some(last));
            // A first command that writes the shared scratch (split-K's partial, a W4A8
            // pre-pass) waits out its last reader whatever its row's flag.
            let writes_scratch = tape.commands.first().is_some_and(|c| {
                c.command
                    .bindings
                    .iter()
                    .any(|b| matches!(b, Binding::Scratch { .. }))
            });
            let flags: Vec<bool> = (0..tape.commands.len())
                .map(|c| c > 0 || writes_scratch)
                .collect();
            assert_eq!(tape.barrier_before, flags, "bucket {bucket_m}");
            assert_eq!(tape.splitk_scratch_bytes, plain.splitk_scratch_bytes);
        }
    }

    /// On an M5 target, an MLX-affine 4-bit GEMM in a prefill bucket runs
    /// W4A8: the pre-pass quantizes the GEMM's input into the shared scratch
    /// (behind a barrier), and the GEMM reads it there with the GEMM's own
    /// weights and output. Both declare the codes offset-8. 8-bit weights,
    /// a group size the kernel lacks and older devices keep their qmm_t; only
    /// the 4-bit codes on M5 are offset-8. The lm_head slice replaces the
    /// pair on non-spec steps and keeps it for spec verification.
    #[test]
    fn w4a8_serves_prefill_gemms_on_m5() {
        use crate::tape::kernel_constants::AffineCodes;
        use crate::tape::lowered::RuntimeGate::{OnlyIfNoSpec, OnlyIfSpec};
        let gemm = |gs, bits| AffineMatmul {
            input: Slot(0),
            output: Slot(1),
            layer: LayerId(0),
            n: NDim(3072),
            k: KDim(8192),
            group_size: AffineGroupSize(gs),
            bits: AffineBits(bits),
            vector_limit: QmvBatchLimit(10),
        };
        let lower_at = |rows: MetalStepTape, profile| {
            lower_subtile_tape_to_metal(&rows, &tp(), bake_point(512, profile)).expect("lowers")
        };
        let backbone = |g| row_tape(plain(&[MetalStep::AffineQmm(g)]));
        let codes = |tape: &LoweredMetalTape| -> Vec<AffineCodes> {
            tape.commands
                .iter()
                .map(|c| AffineCodes::of_constants(c.command.constants))
                .collect()
        };
        let m5 = Some(&crate::targets::M5_10CORE);

        let tape = lower_at(backbone(gemm(64, 4)), m5);
        assert_eq!(
            gated_steps(&tape),
            [
                (KernelId::AffineW4a8Quant, None),
                (KernelId::AffineQmmW4a8, None)
            ]
        );
        let (quant, qmm) = (&tape.commands[0].command, &tape.commands[1].command);
        assert_eq!(
            quant.bindings,
            [
                Binding::ArenaSlot {
                    slot: 0,
                    binding_index: 0
                },
                Binding::Scratch { binding_index: 1 },
            ]
        );
        assert_eq!(quant.dispatch.threadgroups, (8192 / 64 / 16, 512, 1));
        let p = tp();
        assert_eq!(
            qmm.function,
            qmm_w4a8_static_name(
                W4a8Rows::Dense,
                dequant_dtype_for(&p),
                scale_dtype_for(&p),
                64,
                W4a8Tile::Cols128
            )
        );
        assert_eq!(
            qmm.dispatch.threadgroups,
            (3072 / 128, 512 / W4A8_TILE_ROWS, 1)
        );
        assert!(
            qmm.bindings
                .contains(&Binding::Scratch { binding_index: 3 })
        );
        assert!(tape.splitk_scratch_bytes >= w4a8_scratch_bytes(512, 8192));
        assert!(
            tape.barrier_before[0],
            "the pre-pass waits out the scratch's last reader"
        );
        assert_eq!(codes(&tape), [AffineCodes::Offset8; 2]);

        for (g, profile, want) in [
            (gemm(64, 8), m5, AffineCodes::AsWritten),
            (gemm(32, 4), m5, AffineCodes::Offset8),
            (
                gemm(64, 4),
                Some(&crate::targets::M1_8CORE),
                AffineCodes::AsWritten,
            ),
        ] {
            let tape = lower_at(backbone(g), profile);
            assert!(
                tape.commands
                    .iter()
                    .all(|c| c.command.kernel != KernelId::AffineQmmW4a8),
                "{g:?}: no W4A8"
            );
            assert!(codes(&tape).iter().all(|&c| c == want), "{g:?}");
        }

        // The lm_head's sampled rows replace the pair on a non-spec step and keep it for spec.
        use SampleRowsStep::{AllRows, Gather, Matmul, Scatter};
        let row = |s| StepRow::Step(MetalStep::SampleRows(gemm(64, 4), s), None);
        let sampled = MetalStepTape {
            lm_head: [Gather, Matmul, Scatter, AllRows].map(row).to_vec(),
            lm_head_barriers: vec![true; 4],
            lm_head_sources: vec![TEST_SITE.clone(); 4],
            ..MetalStepTape::default()
        };
        let steps = gated_steps(&lower_at(sampled, m5));
        assert!(steps.contains(&(KernelId::GatherLastToken, Some(OnlyIfNoSpec))));
        assert!(steps.ends_with(&[
            (KernelId::AffineW4a8Quant, Some(OnlyIfSpec)),
            (KernelId::AffineQmmW4a8, Some(OnlyIfSpec)),
        ]));
    }

    /// A multi-token tape runs its own attention off decode steps, in the
    /// codebook's rotated domain: K/V staged and q rotated before it, the
    /// output rotated back after it. On a decode step — one token per
    /// sequence, e.g. a batch of decoding sequences — it yields to the
    /// `AttentionViaCacheTq` twin of its decode-kernel form. There is no
    /// dequant pass on any step.
    #[test]
    fn prefill_turboquant_attends_in_the_rotated_domain() {
        use crate::tape::lowered::RuntimeGate::{UnlessDecodeStep};
        let prefill = attention(MetalStep::AttentionPrefillPaged, 0, NeoX);
        let tape = lower_tq_layer(prefill, 64);
        let mut want = vec![
            (KernelId::RopeAppend, None),
            (KernelId::TqQuantizeToPacked, None),
            (KernelId::TqQuantizeToPacked, None),
        ];
        let own = own_attention_steps(&tape);
        assert!(!own.is_empty(), "the attention's own commands");
        want.extend(tq_attention_steps(&own));
        assert_eq!(gated_steps(&tape), want);

        // q rotated in, the output rotated back: the attention's own slots.
        let rotations: Vec<(&str, u32)> = tape
            .commands
            .iter()
            .filter(|c| c.command.kernel == KernelId::TqRotateRows)
            .map(|c| match c.command.bindings[0] {
                Binding::ArenaSlot { slot, .. } => (c.command.function, slot),
                other => panic!("rotated rows bound as {other:?}"),
            })
            .collect();
        assert_eq!(
            rotations,
            [("tq_rotate_rows_bf16", 3), ("tq_unrotate_rows_bf16", 6)]
        );
        // K then V staged, only K re-roping span blocks (cos_sin at slot 9),
        // each operand's new rows (pass 1, slot 17) before its cached ones (pass 2).
        let staged: Vec<(bool, Option<u32>)> = tape
            .commands
            .iter()
            .filter(|c| c.command.kernel == KernelId::TqStageRotated)
            .map(|c| {
                let reropes = c
                    .command
                    .bindings
                    .iter()
                    .any(|b| matches!(b, Binding::Source { .. }));
                let pass = c
                    .command
                    .constants
                    .iter()
                    .find(|k| k.index == 17)
                    .map(|k| k.bits);
                (reropes, pass)
            })
            .collect();
        let ror = tp().rope_on_read;
        assert_eq!(
            staged,
            [
                (ror, Some(1)),
                (ror, Some(2)),
                (false, Some(1)),
                (false, Some(2))
            ]
        );

        let twin = |tape: &LoweredMetalTape| {
            tape.commands
                .iter()
                .find(|c| c.command.kernel == KernelId::AttentionViaCacheTq)
                .map(|c| c.command)
                .expect("Tq twin")
        };
        let decode_form = lower_tq_layer(attention(MetalStep::AttentionViaCache, 0, NeoX), 64);
        assert!(
            twin(&tape) == twin(&decode_form),
            "the twin is the decode kernel's form of the same attention"
        );
    }

    /// The rope-once scratch holds one sequence's keys, so a prefill attention
    /// that reads it runs only on single-sequence steps. A step with several
    /// sequences runs a per-row twin reading K through each sequence's own
    /// block-table row: the same kernel reading the cache's roped K as is, or,
    /// when the step holds an unrotated span block, the sdpa-paged kernel
    /// re-roping it (cos_sin). All keep their TurboQuant gate. Checked for the
    /// full and sliding arms.
    #[test]
    fn prefill_attention_reads_the_rope_once_scratch_only_for_one_sequence() {
        use crate::tape::lowered::RuntimeGate::{OnlyIfOneSequence, OnlyIfUnrotatedBlocks, UnlessOneSequence, UnlessDecodeStep, UnlessUnrotatedBlocks};
        let p = MetalModelConsts {
            rope_on_read: true,
            sliding_window: 512,
            ..tq_consts()
        };
        let one = RuntimeGate::All(&[UnlessDecodeStep, OnlyIfOneSequence]);
        let plain = RuntimeGate::All(&[
            UnlessDecodeStep,
            UnlessOneSequence,
            UnlessUnrotatedBlocks,
        ]);
        let reroping = RuntimeGate::All(&[
            UnlessDecodeStep,
            UnlessOneSequence,
            OnlyIfUnrotatedBlocks,
        ]);
        for attention in [
            attention(MetalStep::AttentionPrefillPaged, 0, NeoX),
            attention(MetalStep::SlidingAttentionPrefillPaged, 0, NeoX),
        ] {
            let tape = lower_tq(&p, coded(tq_writer(0, Causal, LLAMA_KV), attention), 64);
            let own: Vec<&GatedCommand> = tape
                .commands
                .iter()
                .filter(|c| [one, plain, reroping].iter().any(|g| c.gate == Some(*g)))
                .collect();
            assert_eq!(
                own.iter()
                    .map(|c| (c.command.kernel, c.gate))
                    .collect::<Vec<_>>(),
                [
                    (KernelId::RopeOnceSteel, Some(one)),
                    (KernelId::AttentionPrefillSdpaPaged, Some(one)),
                    (KernelId::AttentionPrefillSdpaPaged, Some(plain)),
                    (KernelId::AttentionPrefillSdpaPaged, Some(reroping)),
                ],
                "{attention:?}"
            );
            let binds =
                |c: &GatedCommand, f: fn(&Binding) -> bool| c.command.bindings.iter().any(f);
            let scratch = |b: &Binding| matches!(b, Binding::RopedKScratch { .. });
            let cos_sin = |b: &Binding| matches!(b, Binding::Source { .. });
            assert!(binds(own[1], scratch) && !binds(own[1], cos_sin));
            assert!(!binds(own[2], scratch) && !binds(own[2], cos_sin));
            assert!(!binds(own[3], scratch) && binds(own[3], cos_sin));
            assert_eq!(
                own[2].command.function, own[1].command.function,
                "the plain twin is the scratch attention's own kernel"
            );
            assert_eq!(
                own[3].command.function,
                "attention_prefill_sdpa_v2_paged_bf16_specialized"
            );
        }
    }

    /// Gemma-4's hd512 global prefill runs the unfused attention, whose kernels
    /// read sequence 0 only, on single-sequence steps, and on the rest a paged
    /// attention re-roping span blocks as it reads: gqa_shared at a GQA ratio it
    /// takes, sdpa-paged otherwise. Neither needs a rope-once pair or scratch.
    #[test]
    fn hd512_unfused_prefill_runs_only_for_one_sequence() {
        use crate::tape::lowered::RuntimeGate::{
            OnlyIfOneSequence, UnlessDecodeStep, UnlessOneSequence,
        };
        use crate::tape::lowered::SeqScope;
        for ((codec, step_gate), (kv_heads, per_row)) in [
            (KvCodec::Dense, None),
            (KvCodec::TurboQuant(TQ_BITS), Some(UnlessDecodeStep)),
        ]
        .into_iter()
        .flat_map(|c| {
            [
                (2, "attention_prefill_sdpa_gqa_shared_bf16_specialized"),
                (16, "attention_prefill_sdpa_v2_paged_bf16_specialized"),
            ]
            .map(|h| (c, h))
        }) {
            let one = RuntimeGate::and(step_gate, &[OnlyIfOneSequence]);
            let rest = RuntimeGate::and(step_gate, &[UnlessOneSequence]);
            let p = MetalModelConsts {
                rope_on_read: true,
                global_head_dim: 512,
                num_global_kv_heads: kv_heads,
                global_block_size: 32,
                global_rot_dim: 128,
                kv_codec: codec,
                ..tp()
            };
            let attention = attention(MetalStep::AttentionPrefillPaged, 0, NeoX);
            let writer = tq_writer(0, Causal, LLAMA_KV);
            // A dense model's tape carries the layer as written; a coded one's, as the codec
            // pass expands it.
            let rows = match codec {
                KvCodec::Dense => plain(&[writer, attention]),
                KvCodec::TurboQuant(_) => coded(writer, attention),
            };
            let tape = lower_tq(&p, rows, 64);
            let gated = |g: RuntimeGate| {
                tape.commands
                    .iter()
                    .filter(move |c| c.gate == Some(g))
                    .map(|c| c.command)
            };
            assert!(gated(one).any(|c| c.kernel == KernelId::AttnGatherKRope));
            assert!(gated(one).all(|c| c.seq_scope() == SeqScope::RowZero));
            assert_eq!(
                gated(rest).map(|c| c.function).collect::<Vec<_>>(),
                [per_row],
                "{codec}: {kv_heads} kv heads"
            );
            assert_eq!(tape.roped_k_scratch_bytes, 0);
        }
    }

    /// A command computing batch row 0 only, in a step without a per-row
    /// twin that re-ropes span blocks, is refused: some step with several
    /// sequences would run no attention, or row 0's for every sequence.
    #[test]
    fn a_row_zero_command_without_a_reroping_per_row_twin_is_refused() {
        use crate::tape::lowered::SeqScope;
        let p = MetalModelConsts {
            rope_on_read: true,
            ..tq_consts()
        };
        let attention = attention(MetalStep::AttentionPrefillPaged, 0, NeoX);
        let tape = lower_tq(&p, coded(tq_writer(0, Causal, LLAMA_KV), attention), 64);
        let cmds: Vec<GatedCommand> = tape
            .commands
            .iter()
            .filter(|c| {
                matches!(
                    c.command.kernel,
                    KernelId::RopeOnceSteel | KernelId::AttentionPrefillSdpaPaged
                )
            })
            .map(|c| GatedCommand::ungated(c.command))
            .collect();
        let [rope_once, scratch, plain, reroping] = cmds[..] else {
            panic!("{} commands", cmds.len());
        };
        assert_eq!(rope_once.command.seq_scope(), SeqScope::RowZero);
        assert_eq!(scratch.command.seq_scope(), SeqScope::RowZero);
        for without_reroping in [vec![rope_once, scratch], vec![rope_once, scratch, plain]] {
            assert!(matches!(
                route_by_sequence_count(7, without_reroping),
                Err(LoweringError::RowZeroWithoutPerRowTwin {
                    index: 7,
                    kernel: KernelId::RopeOnceSteel
                })
            ));
        }
        assert!(route_by_sequence_count(7, vec![rope_once, scratch, reroping]).is_ok());
    }

    /// Every binding of `cmd` bound at `index` or later, in order.
    fn bound_from(cmd: &LoweredCommand, index: u8) -> Vec<Binding> {
        let at = |b: &Binding| match *b {
            Binding::Runtime { binding_index, .. }
            | Binding::Inline { binding_index, .. }
            | Binding::Source { binding_index, .. }
            | Binding::ArenaSlot { binding_index, .. } => binding_index,
            other => panic!("TurboQuant command binding {other:?}"),
        };
        cmd.bindings
            .iter()
            .filter(|b| at(b) >= index)
            .copied()
            .collect()
    }

    /// A Qwen2 writer's projection biases reach every codec command: the
    /// quantize removes them (K's rotated by the writer's rotary table at each
    /// token's position — mode 2 — V's as-is, mode 1), and the prefill staging
    /// and decode twin restore them, all bound at the writer's weight site in
    /// `op_abi::rope_append_bias_slots` order. The same layer with a centered
    /// writer binds none of it and runs mode 0: the offset is the difference.
    #[test]
    fn turboquant_restores_a_biased_writers_offsets() {
        use scratchy_ir::{BiasStorage, KvOffset, KvOffsets};
        let p = MetalModelConsts {
            rope_on_read: true,
            ..tq_consts()
        };
        let qwen2 = KvOffsets {
            k: KvOffset::LinearBias(BiasStorage::Affine),
            v: KvOffset::LinearBias(BiasStorage::Affine),
        };
        let layer = crate::tape::ids::LayerId(0);
        let linear = |slot| src(WeightKind::Linear, slot);
        let bias = |slot, bi| source(linear(slot), WeightTensor::AffineLinearBias, layer, bi);
        let cos_sin = source(src(WeightKind::CosSin, 0), WeightTensor::Weight, layer, 19);
        let positions = Binding::Runtime {
            kind: RuntimeBindingKind::Positions,
            binding_index: 20,
        };
        let mode = |mode| {
            [(21, mode), (22, p.rot_dim), (23, p.rot_dim / 2)].map(|(binding_index, value)| {
                Binding::Inline {
                    binding_index,
                    value,
                }
            })
        };
        let cat = |a: &[Binding], b: [Binding; 3]| [a, &b].concat();
        // Each codec command's offset bindings and bias constants, in tape order.
        let offsets = |offsets, attention, bucket_m| {
            let rows = coded(tq_writer(0, Causal, offsets), attention);
            let tape = lower_tq(&p, rows, bucket_m);
            tape.commands
                .iter()
                .filter_map(|c| {
                    let c = &c.command;
                    let from = match c.kernel {
                        KernelId::TqQuantizeToPacked => 18,
                        KernelId::TqStageRotated => 10,
                        KernelId::AttentionViaCacheTq => 14,
                        _ => return None,
                    };
                    let bias_consts: Vec<_> = c
                        .constants
                        .iter()
                        .filter(|k| matches!(k.index, 14 | 15))
                        .copied()
                        .collect();
                    Some((c.kernel, bound_from(c, from), bias_consts))
                })
                .collect::<Vec<_>>()
        };
        let (q, stage, twin) = (
            KernelId::TqQuantizeToPacked,
            KernelId::TqStageRotated,
            KernelId::AttentionViaCacheTq,
        );
        let (k_bias, v_bias) = (ConstantValue::uint(14, 1), ConstantValue::uint(15, 1));
        let decode = attention(MetalStep::AttentionViaCache, 0, Interleaved);
        let prefill = attention(MetalStep::AttentionPrefillPaged, 0, NeoX);

        let quantize = [
            (q, cat(&[bias(0, 18), cos_sin, positions], mode(2)), vec![]),
            (q, cat(&[bias(1, 18)], mode(1)), vec![]),
        ];
        let mut want = quantize.to_vec();
        want.push((twin, vec![bias(0, 14), bias(1, 15)], vec![k_bias, v_bias]));
        assert_eq!(offsets(qwen2, decode, 1), want);

        let mut want = quantize.to_vec();
        // Each operand staged twice: its new rows, then its cached ones.
        want.extend(std::iter::repeat_n(
            (stage, vec![bias(0, 10)], vec![k_bias]),
            2,
        ));
        want.extend(std::iter::repeat_n(
            (stage, vec![bias(1, 10)], vec![v_bias]),
            2,
        ));
        want.push((twin, vec![bias(0, 14), bias(1, 15)], vec![k_bias, v_bias]));
        assert_eq!(offsets(qwen2, prefill, 64), want);

        let centered = |kernel| (kernel, mode(0).to_vec(), vec![]);
        let mut want = vec![centered(q), centered(q)];
        want.extend(std::iter::repeat_n((stage, vec![], vec![]), 4));
        want.push((twin, vec![], vec![]));
        assert_eq!(offsets(LLAMA_KV, prefill, 64), want);
    }

    /// The quantize binds exactly the ABI `turboquant.metal` declares, in order
    /// — written out here, independently of the builder, for a centered layer
    /// (the offset tail is pinned above).
    #[test]
    fn turboquant_quantize_binds_the_kernel_abi() {
        use RuntimeBindingKind as RB;
        let p = tq_consts();
        let layer = crate::tape::ids::LayerId(0);
        let rt = |binding_index, kind| Binding::Runtime {
            kind,
            binding_index,
        };
        let il = |binding_index, value| Binding::Inline {
            binding_index,
            value,
        };
        let bits = TQ_BITS.get();
        let (hd, vpw) = (p.head_dim, 32 / bits);
        let scale = (1.0f32 / (hd as f32).sqrt()).to_bits();
        let tape = lower_tq_layer(attention(MetalStep::AttentionViaCache, 0, Interleaved), 1);
        assert_eq!(
            tape.commands[2].command.bindings, // the quantize of V
            [
                rt(0, RB::KvCacheV { layer }),
                rt(1, RB::SlotMapping { layer }),
                rt(2, RB::TqSigns),
                rt(3, RB::TqBoundaries),
                rt(4, RB::TqCentroids),
                rt(5, RB::TqPackedV { layer }),
                rt(6, RB::TqNormsV { layer }),
                il(7, hd),
                il(8, bits),
                il(9, vpw),
                il(10, hd.div_ceil(vpw)),
                il(11, 1 << bits),
                il(12, scale),
                il(13, p.num_kv_heads),
                il(14, p.block_size),
                il(15, crate::BLOCKS_PER_CHUNK),
                rt(16, RB::SlotMapping { layer }),
                il(17, 0),
                il(21, 0),
                il(22, 0),
                il(23, 0),
            ]
        );
    }

    /// A codec command with nothing to take its operands' offsets from does not
    /// lower: a K bias with no rotary table bound to rotate it to each key — the
    /// first codec row, the writer's K encode (row 1), is refused. (An attention
    /// with no KV writer before it has no offsets to carry and never becomes a
    /// codec row: the codec pass refuses it, `kv_codec`'s
    /// `an_attention_without_its_writer_does_not_expand`.)
    #[test]
    fn turboquant_without_its_writers_offsets_does_not_lower() {
        use scratchy_ir::{BiasStorage, KvOffset, KvOffsets};
        let decode = attention(MetalStep::AttentionViaCache, 0, Interleaved);
        let k_biased = KvOffsets {
            k: KvOffset::LinearBias(BiasStorage::Dense),
            v: KvOffset::Centered,
        };
        assert!(!tp().rope_on_read);
        assert!(matches!(
            try_lower_tq(&tq_consts(), coded(tq_writer(0, Causal, k_biased), decode), 1),
            Err(LoweringError::TurboQuantOffsetUnbound { index: 1 })
        ));
    }

    /// granite regression: the terminal `logits *= recip(logits_scaling)`
    /// ScalarMul must broadcast over the FULL logits row (vocab), not the
    /// head-projection width `Q_SIZE`. A `Gemm(n=vocab)` publishes the
    /// activation width the following ScalarMul covers; covering only
    /// `Q_SIZE` leaves each row's tail unscaled — a non-argmax-invariant
    /// partial multiply (granite gibberish). The fix is type-enforced via
    /// `ActivationWidth` (a `p.q_size` cannot reach `activation_broadcast`
    /// at all); this pins the resulting dispatch. Pure-CPU lowering check.
    #[test]
    fn scalar_mul_after_lm_head_gemm_broadcasts_over_vocab_not_qsize() {
        use crate::tape::lowered::ActivationWidth;
        const VOCAB: u32 = 49_159; // distinct from TestParams::Q_SIZE (2048)
        const BUCKET_M: u32 = 4;
        assert_ne!(
            VOCAB,
            TestParams::Q_SIZE as u32,
            "fixture needs vocab != Q_SIZE"
        );

        // Abstract-interpret the width the lm_head Gemm publishes, exactly
        // as `update_shape_state` does, then feed it to the ScalarMul.
        let mut width = ActivationWidth::residual_stream(&tp());
        let mut m_divisor = 1u32;
        let gemm = MetalStep::Gemm(Slot(0), Slot(1), LayerId(2), NDim(VOCAB), KDim(0));
        update_shape_state(&gemm, &mut width, &mut m_divisor);
        assert_eq!(
            width.get(),
            VOCAB,
            "Gemm output-N becomes the activation width"
        );

        let scalar_mul = MetalStep::ScalarMul(Slot(2), Slot(3), Scale(0.0625));
        let mut scratch = 0u32;
        let mut moe_scratch = 0u32;
        let mut roped_k = 0u32;
        let mut attn_unfused = 0u32;
        let cmds = lower_one(
            &tp(),
            /*chunked=*/ false,
            &scalar_mul,
            row(),
            BUCKET_M,
            0,
            &mut scratch,
            &mut moe_scratch,
            &mut roped_k,
            &mut attn_unfused,
            128,
            None,
            width,
            m_divisor,
        )
        .expect("lower scalar_mul");
        assert_eq!(cmds.len(), 1);
        let cmd = &cmds[0];
        assert_eq!(cmd.kernel, KernelId::ScalarMul);

        // `dispatch_1d` rounds `eff_m * width` up by THREADS_PER_GROUP (256).
        const TPG: u32 = 256;
        let want_groups = (BUCKET_M * VOCAB).div_ceil(TPG);
        let qsize_groups = (BUCKET_M * TestParams::Q_SIZE as u32).div_ceil(TPG);
        assert_eq!(
            cmd.dispatch.threadgroups.0, want_groups,
            "ScalarMul must cover bucket_m*vocab (the full logits row)"
        );
        assert_ne!(
            cmd.dispatch.threadgroups.0, qsize_groups,
            "the granite bug: bucket_m*Q_SIZE leaves the logits-row tail unscaled"
        );
        assert!(
            cmd.dispatch.m_scaling.is_some(),
            "ScalarMul carries X-axis m_scaling for the num_tokens rescale"
        );
    }

    /// At `bucket_m < vector_limit` we should land in the qmv branch
    /// and pick `qmv_fast` for the Llama-1B q_proj shape (N=2048,
    /// K=2048, gs=64, bits=4). N % 8 == 0 && K % 512 == 0 → fast,
    /// not generic; K ∉ {64,128} → not quad.
    #[test]
    fn affine_qmm_lowers_to_qmv_fast_at_decode_bucket() {
        let inst = MetalStep::AffineQmm(q_proj());
        let mut scratch = 0u32;
        let mut moe_scratch = 0u32;
        let mut roped_k = 0u32;
        let mut attn_unfused = 0u32;
        let cmds = lower_one(
            &tp(),
            /*chunked=*/ false,
            &inst,
            row(),
            /*bucket_m=*/ 1,
            /*layer_offset=*/ 5,
            &mut scratch,
            &mut moe_scratch,
            &mut roped_k,
            &mut attn_unfused,
            /*block_cap=*/ 128,
            /*profile=*/ None,
            /*cur_width=*/
            crate::interpreter::metal::lowered::ActivationWidth::residual_stream(&tp()),
            /*m_divisor=*/ 1,
        )
        .expect("lower");
        assert_eq!(cmds.len(), 1, "qmv branch emits exactly one command");
        assert_eq!(scratch, 0, "qmv branch never allocates splitk scratch");
        let cmd = &cmds[0];
        assert_eq!(cmd.kernel, KernelId::AffineQmvFast);
        assert_eq!(cmd.library, "quantized_qmv");
        assert_eq!(cmd.function, "affine_qmv_fast_bf16_s_f16_gs_64_b_4_batch_0");
        assert_eq!(
            cmd.constants,
            vec![ConstantValue::int(0, 2048), ConstantValue::int(1, 2048)],
        );
        // qmv_fast grid: (M, ceil(N/8), B); group: (32, 2, 1).
        assert_eq!(cmd.dispatch.threadgroups, (1, 2048 / 8, 1));
        assert_eq!(cmd.dispatch.threads_per_threadgroup, (32, 2, 1));
        // 5 bindings: weight (idx 0), scales (1), biases (2), in (3), out (4).
        assert_eq!(cmd.bindings.len(), 5);
        // The `layer` baked into the bindings = inst.layer + layer_offset.
        // The site's first Linear source, at inst.layer + layer_offset.
        let affine = affine_weight_bindings(src(WeightKind::Linear, 0), LayerId(8));
        assert_eq!(cmd.bindings[..3], affine);
        match &cmd.bindings[0] {
            Binding::Source {
                which,
                layer,
                binding_index,
                ..
            } => {
                assert_eq!(*which, WeightTensor::Weight);
                assert_eq!(layer.0, 8);
                assert_eq!(*binding_index, 0);
            }
            _ => panic!("bindings[0]: expected Source"),
        }
        match &cmd.bindings[1] {
            Binding::Source { which, .. } => assert_eq!(*which, WeightTensor::AffineScales),
            _ => panic!("bindings[1]: expected AffineScales Source"),
        }
        match &cmd.bindings[2] {
            Binding::Source { which, .. } => assert_eq!(*which, WeightTensor::AffineBiases),
            _ => panic!("bindings[2]: expected AffineBiases Source"),
        }
        match &cmd.bindings[3] {
            Binding::ArenaSlot {
                slot,
                binding_index,
            } => {
                assert_eq!(*slot, 7);
                assert_eq!(*binding_index, 3);
            }
            _ => panic!("bindings[3]: expected in_slot ArenaSlot"),
        }
        match &cmd.bindings[4] {
            Binding::ArenaSlot {
                slot,
                binding_index,
            } => {
                assert_eq!(*slot, 11);
                assert_eq!(*binding_index, 4);
            }
            _ => panic!("bindings[4]: expected out_slot ArenaSlot"),
        }
        assert!(cmd.gemm_dims.is_none());
    }

    /// At a bucket_m where `pick_qmm_t_kernel` returns `Standard`
    /// (current_tgs ≥ 512 → split_k=1), AffineQmm lowers to a single
    /// `KernelId::AffineQmmT` command. bucket_m=512 with N=K=2048,
    /// gs=64 gives n_tiles=64, m_tiles=16 → current_tgs=1024 ≥ 512.
    /// aligned_N=true since N=2048 % 32 == 0.
    #[test]
    fn affine_qmm_lowers_to_qmm_t_standard_when_grid_full() {
        let inst = MetalStep::AffineQmm(q_proj());
        let mut scratch = 0u32;
        let mut moe_scratch = 0u32;
        let mut roped_k = 0u32;
        let mut attn_unfused = 0u32;
        let cmds = lower_one(
            &tp(),
            /*chunked=*/ false,
            &inst,
            row(),
            /*bucket_m=*/ 512,
            /*layer_offset=*/ 0,
            &mut scratch,
            &mut moe_scratch,
            &mut roped_k,
            &mut attn_unfused,
            /*block_cap=*/ 128,
            /*profile=*/ None,
            /*cur_width=*/
            crate::interpreter::metal::lowered::ActivationWidth::residual_stream(&tp()),
            /*m_divisor=*/ 1,
        )
        .expect("lower");
        assert_eq!(cmds.len(), 1, "Standard path emits exactly one command");
        assert_eq!(scratch, 0, "Standard path never allocates splitk scratch");
        let cmd = &cmds[0];
        assert_eq!(cmd.kernel, KernelId::AffineQmmT);
        assert_eq!(cmd.library, "quantized_qmm");
        assert_eq!(
            cmd.function,
            "affine_qmm_t_bf16_s_f16_gs_64_b_4_alN_true_batch_0"
        );
        assert_eq!(
            cmd.constants,
            vec![
                ConstantValue::int(0, 2048),
                ConstantValue::int(1, 2048),
                ConstantValue::int(2, 512),
            ],
        );
        // qmm_t grid: (ceil(N/32), ceil(M/32), B); group: (32, 2, 2).
        assert_eq!(cmd.dispatch.threadgroups, (2048 / 32, 512 / 32, 1));
        assert_eq!(cmd.dispatch.threads_per_threadgroup, (32, 2, 2));
        assert_eq!(cmd.bindings.len(), 5);
        assert!(cmd.gemm_dims.is_none());
    }

    /// At a bucket_m where `pick_qmm_t_kernel` returns SplitK
    /// (current_tgs < 512), AffineQmm lowers to TWO commands:
    /// `AffineQmmTSplitK` writing to the shared scratch buffer +
    /// `SplitKReduceSum` reducing `[split_k, M, N]` → `[M, N]`.
    /// bucket_m=64 with N=K=2048, gs=64 → n_tiles=64, m_tiles=2,
    /// current_tgs=128, split_k=4 (gated to 2048 % (4*64) == 0).
    #[test]
    fn affine_qmm_lowers_to_qmm_t_splitk_pair_at_sparse_prefill() {
        let inst = MetalStep::AffineQmm(q_proj());
        let mut scratch = 0u32;
        let mut moe_scratch = 0u32;
        let mut roped_k = 0u32;
        let mut attn_unfused = 0u32;
        let cmds = lower_one(
            &tp(),
            /*chunked=*/ false,
            &inst,
            row(),
            /*bucket_m=*/ 64,
            /*layer_offset=*/ 0,
            &mut scratch,
            &mut moe_scratch,
            &mut roped_k,
            &mut attn_unfused,
            /*block_cap=*/ 128,
            /*profile=*/ None,
            /*cur_width=*/
            crate::interpreter::metal::lowered::ActivationWidth::residual_stream(&tp()),
            /*m_divisor=*/ 1,
        )
        .expect("lower");
        assert_eq!(cmds.len(), 2, "SplitK pair emits two commands");
        // Scratch sized to split_k * M * N * 2 bytes (bf16 = 2):
        // 4 * 64 * 2048 * 2 = 1_048_576.
        assert_eq!(scratch, 4 * 64 * 2048 * 2);

        let qmm_t = &cmds[0];
        assert_eq!(qmm_t.kernel, KernelId::AffineQmmTSplitK);
        assert_eq!(qmm_t.library, "quantized_qmm");
        assert_eq!(
            qmm_t.function,
            "affine_qmm_t_splitk_bf16_s_f16_gs_64_b_4_alN_true",
        );
        // qmm_t_splitk grid: (n_tiles, m_tiles, split_k); group same as qmm_t.
        assert_eq!(qmm_t.dispatch.threadgroups, (64, 2, 4));
        assert_eq!(qmm_t.dispatch.threads_per_threadgroup, (32, 2, 2));
        // Bindings: w/scales/biases as Weight (0/1/2), in as ArenaSlot (3),
        // y output as Scratch (4).
        assert_eq!(qmm_t.bindings.len(), 5);
        match &qmm_t.bindings[4] {
            Binding::Scratch { binding_index } => assert_eq!(*binding_index, 4),
            _ => panic!("qmm_t bindings[4]: expected Scratch"),
        }

        let reduce = &cmds[1];
        assert_eq!(reduce.kernel, KernelId::SplitKReduceSum);
        assert_eq!(reduce.library, "quantized_splitk_reduce");
        assert_eq!(reduce.function, "splitk_reduce_sum_bf16");
        // reduce constants: 0=M, 1=N, 2=split_k.
        assert_eq!(
            reduce.constants,
            vec![
                ConstantValue::uint(0, 64),
                ConstantValue::uint(1, 2048),
                ConstantValue::uint(2, 4),
            ],
        );
        // Bindings: 0 = output ArenaSlot, 1 = Scratch input.
        assert_eq!(reduce.bindings.len(), 2);
        match &reduce.bindings[0] {
            Binding::ArenaSlot {
                slot,
                binding_index,
            } => {
                assert_eq!(*slot, 11);
                assert_eq!(*binding_index, 0);
            }
            _ => panic!("reduce bindings[0]: expected out ArenaSlot"),
        }
        match &reduce.bindings[1] {
            Binding::Scratch { binding_index } => assert_eq!(*binding_index, 1),
            _ => panic!("reduce bindings[1]: expected Scratch"),
        }
    }

    /// SiluMul lowers to a 1D dispatch over `bucket_m * width`
    /// elements with three ArenaSlot bindings (out, gate, up). Function
    /// constant 0 holds the total element count; `width` rides on the
    /// instruction (macro-baked from the claim's gate/up Gemm N).
    #[test]
    fn silu_mul_lowers_with_three_arena_bindings_and_n_constant() {
        let inst = MetalStep::SiluMul(
            /*gate=*/ Slot(5),
            /*up=*/ Slot(6),
            /*out=*/ Slot(7),
            /*width=*/ IntermediateSize(11008),
        );
        let mut scratch = 0u32;
        let mut moe_scratch = 0u32;
        let mut roped_k = 0u32;
        let mut attn_unfused = 0u32;
        let cmds = lower_one(
            &tp(),
            /*chunked=*/ false,
            &inst,
            row(),
            /*bucket_m=*/ 64,
            /*layer_offset=*/ 0,
            &mut scratch,
            &mut moe_scratch,
            &mut roped_k,
            &mut attn_unfused,
            /*block_cap=*/ 128,
            /*profile=*/ None,
            /*cur_width=*/
            crate::interpreter::metal::lowered::ActivationWidth::residual_stream(&tp()),
            /*m_divisor=*/ 1,
        )
        .expect("lower");
        assert_eq!(cmds.len(), 1, "SiluMul emits exactly one command");
        let cmd = &cmds[0];
        assert_eq!(cmd.kernel, KernelId::SiluMul);
        assert_eq!(cmd.library, "silu_mul");
        assert_eq!(cmd.function, "silu_mul_bf16");
        // n = bucket_m * width = 64 * 11008 (the instruction-carried
        // width, independent of TestParams::INTERMEDIATE_SIZE).
        let n_expected = 64 * 11008_u32;
        assert_eq!(cmd.constants, vec![ConstantValue::uint(0, n_expected)]);
        assert_eq!(cmd.bindings.len(), 3);
        match &cmd.bindings[0] {
            Binding::ArenaSlot {
                slot,
                binding_index,
            } => {
                assert_eq!(*slot, 7);
                assert_eq!(*binding_index, 0);
            }
            _ => panic!("bindings[0]: expected out_slot ArenaSlot"),
        }
        match &cmd.bindings[1] {
            Binding::ArenaSlot {
                slot,
                binding_index,
            } => {
                assert_eq!(*slot, 5);
                assert_eq!(*binding_index, 1);
            }
            _ => panic!("bindings[1]: expected gate_slot ArenaSlot"),
        }
        match &cmd.bindings[2] {
            Binding::ArenaSlot {
                slot,
                binding_index,
            } => {
                assert_eq!(*slot, 6);
                assert_eq!(*binding_index, 2);
            }
            _ => panic!("bindings[2]: expected up_slot ArenaSlot"),
        }
        assert!(cmd.gemm_dims.is_none());
    }

    /// Unaligned-N Llama-1B-style lm_head (N=128256 — 128256 % 32 = 0
    /// so this is actually aligned). Use a synthetic shape for the
    /// unaligned branch: N=2050 → N % 32 = 2 → alN=false. Use
    /// bucket_m=512 so we stay on the Standard path
    /// (`pick_qmm_t_kernel` returns SplitK at small bucket_m).
    #[test]
    fn affine_qmm_qmm_t_unaligned_n_picks_unaligned_kernel() {
        let inst = MetalStep::AffineQmm(AffineMatmul {
            layer: LayerId(0),
            n: NDim(2050),
            ..q_proj()
        });
        let mut scratch = 0u32;
        let mut moe_scratch = 0u32;
        let mut roped_k = 0u32;
        let mut attn_unfused = 0u32;
        let cmds = lower_one(
            &tp(),
            /*chunked=*/ false,
            &inst,
            row(),
            /*bucket_m=*/ 512,
            /*layer_offset=*/ 0,
            &mut scratch,
            &mut moe_scratch,
            &mut roped_k,
            &mut attn_unfused,
            /*block_cap=*/ 128,
            /*profile=*/ None,
            /*cur_width=*/
            crate::interpreter::metal::lowered::ActivationWidth::residual_stream(&tp()),
            /*m_divisor=*/ 1,
        )
        .expect("lower");
        assert_eq!(cmds.len(), 1);
        let cmd = &cmds[0];
        assert_eq!(
            cmd.function,
            "affine_qmm_t_bf16_s_f16_gs_64_b_4_alN_false_batch_0"
        );
        // Ceil-div on N: 2050.div_ceil(32) = 65; M-tiles: 512/32 = 16.
        assert_eq!(cmd.dispatch.threadgroups, (65, 512 / 32, 1));
    }

    /// AffineEmbed lowers to a single command targeting
    /// `affine_embed_<dtype>_gs_<gs>_b_4` in `quantized_dequantize`,
    /// with hidden_size in function_constant(0) and a 2D dispatch
    /// (bytes_per_row in X, num_tokens in Y).
    #[test]
    fn affine_embed_lowers_to_single_command_with_2d_dispatch() {
        let inst = MetalStep::AffineEmbed(
            /*out_slot=*/ Slot(0),
            AffineGroupSize(64),
            AffineBits(4),
        );
        let bucket_m = 32u32;
        let mut scratch = 0u32;
        let mut moe_scratch = 0u32;
        let mut roped_k = 0u32;
        let mut attn_unfused = 0u32;
        let cmds = lower_one(
            &tp(),
            /*chunked=*/ false,
            &inst,
            row(),
            bucket_m,
            /*layer_offset=*/ 5,
            &mut scratch,
            &mut moe_scratch,
            &mut roped_k,
            &mut attn_unfused,
            /*block_cap=*/ 128,
            /*profile=*/ None,
            /*cur_width=*/
            crate::interpreter::metal::lowered::ActivationWidth::residual_stream(&tp()),
            /*m_divisor=*/ 1,
        )
        .expect("lower");
        assert_eq!(cmds.len(), 1, "AffineEmbed always emits a single command");
        assert_eq!(scratch, 0, "AffineEmbed never allocates splitk scratch");

        let cmd = &cmds[0];
        assert_eq!(cmd.kernel, KernelId::AffineEmbed);
        assert_eq!(cmd.library, "quantized_dequantize");
        // TestParams pins Bf16; group_size=64 → bf16 / s_f16 / gs_64 symbol.
        assert_eq!(cmd.function, "affine_embed_bf16_s_f16_gs_64_b_4");

        // function_constant(0) = hidden_size = p.hidden_size = 2048.
        assert_eq!(cmd.constants, vec![ConstantValue::uint(0, 2048)]);

        // 2D dispatch: (HIDDEN_SIZE/2 / THREADS_PER_GROUP, bucket_m, 1).
        // HIDDEN_SIZE=2048 → bytes_per_row=1024; THREADS_PER_GROUP=256.
        // groups_x = 1024.div_ceil(256) = 4.
        assert_eq!(cmd.dispatch.threadgroups, (4, bucket_m, 1));
        assert_eq!(
            cmd.dispatch.threads_per_threadgroup,
            (THREADS_PER_GROUP, 1, 1)
        );

        // 5 bindings: weight (0), scales (1), biases (2), input_ids (3), out (4).
        assert_eq!(cmd.bindings.len(), 5);
        let embedding = src(WeightKind::AffineQuantEmbedding, 0);
        match &cmd.bindings[0] {
            Binding::Source {
                ix,
                which,
                layer,
                binding_index,
            } => {
                assert_eq!(*ix, embedding);
                assert_eq!(*which, WeightTensor::Weight);
                // Embed is unlayered — `layer = 0` regardless of layer_offset.
                assert_eq!(layer.0, 0);
                assert_eq!(*binding_index, 0);
            }
            _ => panic!("bindings[0]: expected the AffineQuantEmbedding source"),
        }
        match &cmd.bindings[1] {
            Binding::Source { ix, which, .. } => {
                assert_eq!((*ix, *which), (embedding, WeightTensor::AffineScales))
            }
            _ => panic!("bindings[1]: expected AffineScales"),
        }
        match &cmd.bindings[2] {
            Binding::Source { ix, which, .. } => {
                assert_eq!((*ix, *which), (embedding, WeightTensor::AffineBiases))
            }
            _ => panic!("bindings[2]: expected AffineBiases"),
        }
        match &cmd.bindings[3] {
            Binding::Runtime {
                kind,
                binding_index,
            } => {
                assert_eq!(*kind, RuntimeBindingKind::InputIds);
                assert_eq!(*binding_index, 3);
            }
            _ => panic!("bindings[3]: expected InputIds runtime"),
        }
        match &cmd.bindings[4] {
            Binding::ArenaSlot {
                slot,
                binding_index,
            } => {
                assert_eq!(*slot, 0);
                assert_eq!(*binding_index, 4);
            }
            _ => panic!("bindings[4]: expected out_slot ArenaSlot"),
        }
        assert!(cmd.gemm_dims.is_none());
    }
}

// SPDX-License-Identifier: Apache-2.0
//! Lowering pass: `&[Instruction]` → `LoweredMetalTape`.
//!
//! Pure CPU code. No Metal device required, no kernel dispatch — just
//! a structural translation that:
//!
//! 1. Statically unrolls `Loop(count, body_len)` over the loop body —
//!    each iteration's `iter` index is added to per-instruction
//!    `layer` literals at weight-binding time (Metal mirror of the
//!    cuda interpreter's `ctx.layer_offset = iter` pattern).
//! 2. Drops metadata-only instructions (`Reshape`, `Alias`, `Free`)
//!    that don't emit a Metal dispatch.
//! 3. For each compute instruction, picks the right `KernelId`,
//!    derives a `DispatchShape` from `bucket_m` + the kernel's
//!    convention, and produces `Binding`s pointing at arena slots,
//!    typed weight thunks, or runtime buffers.
//!
//! Coverage (post-Phase B):
//! `Embed`, `RmsNorm`, `FusedAddRmsNorm`, `Gemm`, `FusedGateUpSiluMul`,
//! `RopeAppend`, `AttentionViaCache`, `AttentionPrefillPaged`,
//! `Add`, `ScalarMul`, plus `Loop`/`Reshape`/`Alias`/`Free` as
//! structural ops. `AttentionPrefillContiguous` is matched only as an
//! `unreachable!` arm — Phase B's metal macro adapter
//! (`metal/attention.rs::fan_out`) emits `AttentionPrefillPaged` for
//! every metal model, so the contiguous variant should never reach
//! lowering. Every other variant raises
//! [`LoweringError::UnsupportedVariant`] with the variant's type name
//! so the model author knows which arm to add next.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::tape::model_consts::MetalModelConsts;
use scratchy_ir::Instruction;

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
    DequantDtype, QmmTKernel, QmvKernel, SMALL_M_TILE_COLS, ScaleDtype, SmallMTile,
    pick_qmm_t_kernel, pick_qmv_kernel, qmm_t_dispatch_shape, qmm_t_kernel_static_name,
    qmm_t_kernel_static_name_with_compute, qmv_dispatch_shape, qmv_kernel_static_name,
    small_m_kernel_static_name, splitk_reduce_kernel_static_name,
};
use crate::specialized_pipeline_cache::ConstantValue;

use crate::tape::lowered::{
    Binding, DispatchShape, GatedCommand, GemmDims, IntoBaked, KernelId, LoweredCommand,
    LoweredMetalTape, LoweringError, MetalDtype, RuntimeBindingKind, WeightBundleKind,
    WeightLocator, WeightTensor, baked,
};

/// Lower one bucket's `(backbone ++ lm_head)` instruction stream.
///
/// Concatenation matches the cuda interpreter's effective behavior:
/// `forward()` runs backbone then lm_head in sequence for a given
/// bucket. Lowering them as one stream lets the worker bake both
/// halves into the bucket's single dispatch plan, with no extra mid-bucket
/// boundary the caller needs to manage.
///
/// Avoids requiring `Instruction: Clone` — the caller hands two
/// `&[Instruction]` slices and we walk them in place, unrolling
/// `Loop` per the same rules as the single-slice [`lower`]. Loop
/// bodies that span the backbone/lm_head boundary are not supported
/// (no model emits one — the cuda lowering pass partitions loops
/// strictly inside one half), but if one ever shows up the malformed-
/// loop check fires inside the offending half and surfaces the
/// per-half index.
#[allow(clippy::too_many_arguments)]
pub fn lower_pair(
    p: &MetalModelConsts,
    // Chunked-addressing attention variant (spec-decode). A PARAMETER,
    // not process state: the macro bakes both variants from parallel
    // threads, and a global here was a cross-model probe race.
    chunked: bool,
    backbone: &[Instruction],
    lm_head: &[Instruction],
    backbone_barriers: &[bool],
    lm_head_barriers: &[bool],
    bucket_m: u32,
    num_arena_slots: u32,
    backbone_tape_index: u32,
    lm_head_tape_index: u32,
    // Runtime per-sequence block-table capacity (`KvCachePool::max_blocks_per_seq`).
    // Replaces the compile-time `p.max_blocks_per_seq` at the `MaxBlocksPerSeq`
    // function-constant sites + the rope-once `roped_k_scratch` sizing.
    block_cap: u32,
    profile: Option<&crate::targets::MetalTargetProfile>,
) -> Result<LoweredMetalTape, LoweringError> {
    let bb = lower(
        p,
        chunked,
        backbone,
        backbone_barriers,
        bucket_m,
        num_arena_slots,
        backbone_tape_index,
        block_cap,
        profile,
    )?;
    let lh = lower(
        p,
        chunked,
        lm_head,
        lm_head_barriers,
        bucket_m,
        num_arena_slots,
        lm_head_tape_index,
        block_cap,
        profile,
    )?;
    let mut commands = bb.commands.to_vec();
    let mut barrier_before = bb.barrier_before.to_vec();
    // Backbone commands (`bb.commands`) carry their own gate on each
    // `GatedCommand`: `inject_tq` tagged the per-layer TurboQuant commands
    // (and the plain decode attention they replace). There is no separate gate vec to
    // re-initialize, so those gates cannot be dropped here — the lm_head
    // slice/fallback pushes below assign their own gate per command.

    // Sample-position slice — fast lm_head for prefill.
    //
    // The lm_head GEMM only needs the LAST token's hidden state per
    // sequence; the worker discards every other num_tokens-1 row of
    // logits host-side (`embedding_gather` keyed on
    // `last_token_indices`). At num_tokens=1024 + vocab=128256 +
    // K=hidden this is ~31× wasted GEMM work — ~325ms of the prefill
    // TTFT on M4 Llama-3.2-3B-4bit.
    //
    // To shrink it we wrap the lm_head AffineQmm with a pre-GEMM
    // gather and a post-GEMM scatter:
    //
    //   1. `GatherLastToken`     : in[0,:]  := in[num_tokens-1,:]
    //   2. lm_head AffineQmm     : m-tile dispatch overridden to 1 tile
    //                              (BM=32 rows of output computed; row
    //                              0 is the only one we care about)
    //   3. `ScatterFirstToLastRow`: out[num_tokens-1,:] := out[0,:]
    //
    // Post-step keeps the worker's downstream
    // `embedding_gather(logits, last_token_indices=[N-1])` correct
    // without it needing to know about the slice.
    //
    // Preconditions for applying the slice (else fall through to the
    // standard lm_head lowering):
    //   * `bucket_m > 1` — decode/batched-decode buckets need every
    //     row of logits intact.
    //   * `lm_head` is exactly one `Instruction::AffineQmm` (the
    //     common case across Llama / Qwen / Mistral). Tied
    //     embeddings and multi-step lm_heads fall through.
    //   * It lowered to exactly one `LoweredCommand` — i.e. Standard
    //     or Nax qmm_t, not SplitK (whose split partials and reduce
    //     would need their own row handling). At num_tokens=1024 the
    //     dispatcher always picks Standard, so this is the prefill
    //     path in practice.
    // lm_head only needs the LAST token's row at prefill (the worker's
    // post-pass `embedding_gather` keys on `last_token_indices=[N-1]`
    // and discards every other row). Without this slice, lm_head runs
    // a full M=bucket_m × N=vocab × K=hidden GEMM — at M=1024, vocab=
    // 128256, hidden=3072 that's ~190 ms of pure waste on M1 Max.
    //
    let slice_disabled = false;
    // The slice also tolerates a TRAILING `TanhSoftCap` (Gemma2/4
    // final logit softcapping): `lm_head = [AffineQmm, TanhSoftCap]`
    // lowers to gather → qmv → narrow softcap → scatter. Without this,
    // the softcap's presence forced the FULL `M = bucket_m × vocab`
    // lm_head GEMM at every prefill chunk — ~0.75 s of the Gemma4-12B
    // T=2930 prefill was the discarded lm_head rows.
    let qmm_cmd_ok = |idx: usize| {
        matches!(
            lh.commands.get(idx).map(|c| c.command.kernel),
            Some(KernelId::AffineQmmT | KernelId::AffineQmmTNax)
        )
    };
    let slice_info = if !slice_disabled && bucket_m > 1 {
        match (lm_head, lh.commands.len()) {
            // Plain lm_head (Llama / Qwen / Mistral).
            (
                [
                    Instruction::AffineQmm(
                        in_slot,
                        out_slot,
                        layer,
                        n,
                        k,
                        group_size,
                        bits,
                        _vector_limit,
                    ),
                ],
                1,
            ) if qmm_cmd_ok(0) => Some(LmHeadSliceInfo {
                in_slot: *in_slot,
                out_slot: *out_slot,
                layer: *layer,
                // The lm_head AffineQmm is the FIRST instruction in
                // the lm_head slice, so its op_idx is 0 inside the
                // lm_head tape.
                locator: WeightLocator {
                    bucket: lm_head_tape_index,
                    op_idx: 0,
                    slot: 0,
                },
                n: *n,
                k: *k,
                group_size: *group_size,
                bits: *bits,
                softcap_out_slot: None,
            }),
            // lm_head + final softcap (Gemma2/4). The softcap must
            // consume the qmm's output slot.
            (
                [
                    Instruction::AffineQmm(
                        in_slot,
                        out_slot,
                        layer,
                        n,
                        k,
                        group_size,
                        bits,
                        _vector_limit,
                    ),
                    Instruction::TanhSoftCap(sc_in, sc_out),
                ],
                2,
            ) if qmm_cmd_ok(0)
                && matches!(
                    lh.commands.get(1).map(|c| c.command.kernel),
                    Some(KernelId::TanhSoftCap)
                )
                && sc_in == out_slot =>
            {
                Some(LmHeadSliceInfo {
                    in_slot: *in_slot,
                    out_slot: *out_slot,
                    layer: *layer,
                    locator: WeightLocator {
                        bucket: lm_head_tape_index,
                        op_idx: 0,
                        slot: 0,
                    },
                    n: *n,
                    k: *k,
                    group_size: *group_size,
                    bits: *bits,
                    softcap_out_slot: Some(*sc_out),
                })
            }
            _ => None,
        }
    } else {
        None
    };

    use crate::tape::lowered::RuntimeGate;
    if let Some(info) = slice_info {
        // Non-spec path: index-driven gather → qmv-M=num_sample_rows
        // → index-driven scatter. Works for any num_seqs (single-seq
        // prefill / multi-seq prefill / decode) because the gather
        // kernel reads the per-seq sample row from `last_token_indices`
        // and the qmv runs at M = num_sample_rows (worker overrides
        // TG.X via seq_axis). Scatter writes back so the worker's
        // downstream argmax reads from the original sample positions.
        commands.push(GatedCommand::gated(
            gather_last_token_command(p, info.in_slot, info.k, bucket_m),
            RuntimeGate::OnlyIfNoSpec,
        ));
        barrier_before.push(true);
        commands.push(GatedCommand::gated(
            lm_head_qmv_command(p, &info, bucket_m),
            RuntimeGate::OnlyIfNoSpec,
        ));
        barrier_before.push(*lh.barrier_before.first().unwrap_or(&true));
        // Trailing softcap (Gemma2/4): cap the narrow qmv output
        // before the scatter so the scattered sample rows carry
        // CAPPED logits (a partial/absent cap is not argmax-invariant
        // against fully-capped reference values, and sampling reads
        // these magnitudes). Dispatch covers rows 0..num_tokens —
        // rows past num_sample_rows hold stale data whose capping is
        // harmless (the scatter only copies rows 0..num_sample_rows).
        if let Some(sc_out) = info.softcap_out_slot {
            commands.push(GatedCommand::gated(
                lm_head_softcap_command(p, info.out_slot, sc_out, info.n, bucket_m),
                RuntimeGate::OnlyIfNoSpec,
            ));
            barrier_before.push(true);
        }
        commands.push(GatedCommand::gated(
            scatter_first_to_last_row_command(
                p,
                info.softcap_out_slot.unwrap_or(info.out_slot),
                info.n,
                bucket_m,
            ),
            RuntimeGate::OnlyIfNoSpec,
        ));
        barrier_before.push(true);

        // Spec-decode verify fallback: rejection sampling needs every
        // query row's logits, not just the per-seq sample rows. The
        // full `M = bucket_m` qmm populates the entire logits buffer.
        // CUDA's path doesn't need this because cuMemGetAllocationGran
        // forces 2 MiB tiles anyway — but on metal the slice is
        // strictly smaller, so we keep the fallback gated for spec.
        for (i, cmd) in lh.commands.iter().enumerate() {
            commands.push(GatedCommand::gated(cmd.command, RuntimeGate::OnlyIfSpec));
            barrier_before.push(*lh.barrier_before.get(i).unwrap_or(&true));
        }
    } else {
        // No slice — slice precondition (single AffineQmm lm_head,
        // bucket_m > 1, etc.) didn't hold. Fall through to the
        // standard lm_head lowering, ungated.
        for (i, cmd) in lh.commands.iter().enumerate() {
            commands.push(*cmd);
            barrier_before.push(*lh.barrier_before.get(i).unwrap_or(&true));
        }
    }

    debug_assert_eq!(commands.len(), barrier_before.len());

    Ok(LoweredMetalTape {
        bucket_m,
        num_arena_slots,
        commands: baked(commands),
        barrier_before: baked(barrier_before),
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

/// Source-Instruction params captured at `lm_head` AffineQmm so the
/// slice path can synthesize a qmv command (kernel substitute) plus
/// the matching gather/scatter framing. Lives only inside `lower_pair`.
struct LmHeadSliceInfo {
    in_slot: u32,
    out_slot: u32,
    layer: u32,
    /// Tape locator for the lm_head LinearLayer accessor — the worker
    /// resolves through `WeightAccessors::linear_at`.
    locator: WeightLocator,
    n: u32,
    k: u32,
    group_size: u32,
    bits: u32,
    /// `Some(out_slot)` when the lm_head slice carries a trailing
    /// `TanhSoftCap` (Gemma2/4 final logit softcapping) — the narrow
    /// chain inserts a softcap between the qmv and the scatter, and
    /// the scatter reads/writes the softcap's output slot.
    softcap_out_slot: Option<u32>,
}

/// Lower the lm_head AffineQmm through a qmv matvec kernel
/// dispatched at `M = num_sample_rows` (= 1 for single-seq, num_seqs
/// for multi-seq, K+1 for spec-decode verify). The qmv kernel
/// natively supports M>1 via its X-axis threadgroup grid — one TG
/// per row, all reading the same 197 MB packed weight tile per
/// (BN, BK) sub-tile. Worker sets `threadgroups.X = num_sample_rows`
/// at dispatch time via the `seq_axis` MScaling override.
///
/// vs the qmm_t-1-tile BM=32 fallback this avoids the
/// `ceil(bucket_m/32) × 4008` tile sweep (32+ wasted m-rows per
/// tile at multi-seq prefill); the gathered inputs at rows
/// 0..num_sample_rows-1 mean the qmv reads exactly the rows it
/// needs.
fn lm_head_qmv_command(
    p: &MetalModelConsts,
    info: &LmHeadSliceInfo,
    _bucket_m: u32,
) -> LoweredCommand {
    let dtype = dequant_dtype_for(p);
    let scale_dtype = scale_dtype_for(p);
    let n_v = info.n;
    let k_v = info.k;
    let bits_v = info.bits;
    let gs = info.group_size;

    let kernel = pick_qmv_kernel(n_v, k_v, bits_v);
    // Bake X=1 baseline. Worker overrides X to live `num_seqs` at
    // dispatch via `seq_axis = Some(X)` — qmv runs at M=num_seqs.
    let (tg, tpg) = qmv_dispatch_shape(kernel, /* M = */ 1, n_v, /* B = */ 1);
    let kernel_id = match kernel {
        QmvKernel::Quad { .. } => KernelId::AffineQmvQuad,
        QmvKernel::Fast => KernelId::AffineQmvFast,
        QmvKernel::Generic => KernelId::AffineQmv,
    };

    LoweredCommand {
        kernel: kernel_id,
        library: "quantized_qmv",
        function: qmv_kernel_static_name(kernel, dtype, scale_dtype, bits_v, gs),
        constants: super::kernel_constants::AffineQmvConstants {
            k: super::ids::KDimI32(k_v as i32),
            n: super::ids::NDimI32(n_v as i32),
        }
        .into_baked(),
        dispatch: DispatchShape {
            threadgroups: tg,
            threads_per_threadgroup: tpg,
            // Baseline X=1. seq_axis=Some(X) SETS X to live num_seqs
            // at dispatch — qmv runs at M = num_seqs (one TG per row).
            m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                axis: crate::tape::lowered::MScaleAxis::X,
                bucket_m: super::ids::BucketM(1),
                seq_axis: Some(crate::tape::lowered::MScaleAxis::X),
            }),
        },
        bindings: baked(affine_qmm_bindings(
            info.in_slot,
            info.out_slot,
            super::ids::LayerId(info.layer),
            info.locator,
        )),
        gemm_dims: None,
    }
}

/// Narrow softcap for the lm_head slice: `out = cap * tanh(in / cap)`
/// over rows 0..num_tokens of the narrow qmv output (X-proportional
/// scaling — same shape math as the full `I::TanhSoftCap` arm, but
/// over the slice's `[rows, vocab]` region instead of the full-M
/// logits buffer). Rows past `num_sample_rows` are stale and their
/// capping is harmless; the scatter copies only the sample rows.
fn lm_head_softcap_command(
    p: &MetalModelConsts,
    in_slot: u32,
    out_slot: u32,
    vocab_size: u32,
    bucket_m: u32,
) -> LoweredCommand {
    debug_assert!(
        p.final_logit_softcapping > 0.0,
        "lm_head_softcap_command built with FINAL_LOGIT_SOFTCAPPING <= 0"
    );
    LoweredCommand {
        kernel: KernelId::TanhSoftCap,
        library: "elementwise",
        function: pick_specialized_symbol(
            "tanh_soft_cap_f16_specialized",
            "tanh_soft_cap_bf16_specialized",
            p.metal_dtype,
        ),
        // Slot 1: elementwise.metal's fn-const indices are file-scoped
        // (slot 0 = BIAS_ADD_NUM_COLS).
        constants: baked(vec![ConstantValue::float(1, p.final_logit_softcapping)]),
        dispatch: {
            let mut d = DispatchShape::dispatch_1d(bucket_m * vocab_size, THREADS_PER_GROUP);
            d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                seq_axis: None,
                axis: crate::tape::lowered::MScaleAxis::X,
                bucket_m: super::ids::BucketM(bucket_m),
            });
            d
        },
        bindings: baked(vec![
            Binding::ArenaSlot {
                slot: in_slot,
                binding_index: 0,
            },
            Binding::ArenaSlot {
                slot: out_slot,
                binding_index: 1,
            },
        ]),
        gemm_dims: None,
    }
}

fn gather_last_token_command(
    p: &MetalModelConsts,
    slot: u32,
    hidden_size: u32,
    bucket_m: u32,
) -> LoweredCommand {
    sample_slice_command(
        p,
        slot,
        hidden_size,
        bucket_m,
        KernelId::GatherLastToken,
        "gather_last_token_f16_specialized",
        "gather_last_token_bf16_specialized",
    )
}

fn scatter_first_to_last_row_command(
    p: &MetalModelConsts,
    slot: u32,
    vocab_size: u32,
    bucket_m: u32,
) -> LoweredCommand {
    sample_slice_command(
        p,
        slot,
        vocab_size,
        bucket_m,
        KernelId::ScatterFirstToLastRow,
        "scatter_first_to_last_row_f16_specialized",
        "scatter_first_to_last_row_bf16_specialized",
    )
}

fn sample_slice_command(
    p: &MetalModelConsts,
    slot: u32,
    row_stride: u32,
    bucket_m: u32,
    kernel: KernelId,
    f16_symbol: &'static str,
    bf16_symbol: &'static str,
) -> LoweredCommand {
    // 2D dispatch: tg.x covers the row-stride dim (one thread per
    // element), tg.y covers the sample-row dim (one TG row per seq).
    // The Y dim is baked to bucket_m worst-case; threads with
    // `tid.y >= num_seqs` early-out so the actual work is N rows.
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
            threadgroups: (row_stride.div_ceil(THREADS_PER_TG), bucket_m, 1),
            threads_per_threadgroup: (THREADS_PER_TG, 1, 1),
            // seq_axis=Y SETS tg.y = num_seqs at dispatch — only N TGs
            // along the sample-row dim instead of the bucket_m baseline.
            // axis here is a no-op (bucket_m=1 → X*num_tokens/1, but X
            // is the row-stride dim which doesn't scale with M).
            m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                axis: crate::tape::lowered::MScaleAxis::X,
                bucket_m: super::ids::BucketM(1),
                seq_axis: Some(crate::tape::lowered::MScaleAxis::Y),
            }),
        },
        bindings: baked(vec![
            Binding::ArenaSlot {
                slot,
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

/// TurboQuant prefill: the per-layer staging command — the layer's K (or V)
/// for every sequence of the step, written into the fp16 scratch in the
/// codebook's rotated domain (`tq_stage_rotated`). `is_global` is the
/// attention's layer class (its rope-on-read cos_sin for span blocks; V is
/// never roped). The grid (block-table width, num_kv_heads, num_seqs) is
/// overridden by the worker per forward; the baked shape is a placeholder.
fn tq_stage_command(
    p: &MetalModelConsts,
    layer: u32,
    is_v: bool,
    is_global: bool,
    bucket_m: u32,
    block_cap: u32,
) -> LoweredCommand {
    let (ror_rd, ror_po, ror_on, ror_bind) = if is_v {
        (None, None, None, None)
    } else {
        rope_on_read_params(p, is_global)
    };
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
            bits: super::ids::TqCodeBits(crate::turboquant::tq_bits(p.tq_kv_bits)),
            rot_dim: ror_rd,
            pair_off: ror_po,
            rope_on_read: ror_on,
        }
        .into_baked(),
        dispatch: DispatchShape {
            threadgroups: (1, p.num_global_kv_heads, bucket_m),
            threads_per_threadgroup: (32, 1, 1),
            m_scaling: None,
        },
        bindings: super::kernel_bindings::TqStageBindingSet {
            kv_layer: super::ids::LayerId(layer),
            is_v,
            rope_on_read: ror_bind,
        }
        .into_baked(),
        gemm_dims: None,
    }
}

/// TurboQuant prefill: rotate the attention's q rows into the codebook domain
/// in place (`inverse` false, R·q), or its output rows back (`inverse`, Rᵀ·o).
fn tq_rotate_command(
    p: &MetalModelConsts,
    rows: u32,
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
        bindings: super::kernel_bindings::TqRotateRowsBindingSet {
            rows: super::ids::ArenaSlotIdx(rows),
        }
        .into_baked(),
        gemm_dims: None,
    }
}

/// TurboQuant: build the per-layer quantize command (new tokens in the fp16
/// scratch → packed[L]), run right after the layer's KV writer. New-token
/// count == num_tokens, so the grid scales like RopeAppend (m_scaling on X).
/// `is_v` selects the V store/scratch. The TurboQuant layers are the GLOBAL
/// class (every layer on uniform arches, where GLOBAL_* == base).
fn tq_quantize_command(
    p: &MetalModelConsts,
    layer: u32,
    is_v: bool,
    bucket_m: u32,
) -> LoweredCommand {
    use crate::tape::lowered::{Binding, RuntimeBindingKind as RB};
    let (hd, nkv, bs) = (
        p.global_head_dim,
        p.num_global_kv_heads,
        p.global_block_size,
    );
    let bits = crate::turboquant::tq_bits(p.tq_kv_bits);
    let vpw = 32 / bits;
    let pdim = hd.div_ceil(vpw);
    let n_cent = 1u32 << bits;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let li = super::ids::LayerId(layer);
    let (cache, packed, norms) = if is_v {
        (
            RB::KvCacheV { layer: li },
            RB::TqPackedV { layer: li },
            RB::TqNormsV { layer: li },
        )
    } else {
        (
            RB::KvCacheK { layer: li },
            RB::TqPackedK { layer: li },
            RB::TqNormsK { layer: li },
        )
    };
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
        bindings: baked(vec![
            Binding::Runtime {
                kind: cache,
                binding_index: 0,
            },
            Binding::Runtime {
                kind: RB::SlotMapping { layer: li },
                binding_index: 1,
            },
            Binding::Runtime {
                kind: RB::TqSigns,
                binding_index: 2,
            },
            Binding::Runtime {
                kind: RB::TqBoundaries,
                binding_index: 3,
            },
            Binding::Runtime {
                kind: RB::TqCentroids,
                binding_index: 4,
            },
            Binding::Runtime {
                kind: packed,
                binding_index: 5,
            },
            Binding::Runtime {
                kind: norms,
                binding_index: 6,
            },
            Binding::Inline {
                binding_index: 7,
                value: hd,
            },
            Binding::Inline {
                binding_index: 8,
                value: bits,
            },
            Binding::Inline {
                binding_index: 9,
                value: vpw,
            },
            Binding::Inline {
                binding_index: 10,
                value: pdim,
            },
            Binding::Inline {
                binding_index: 11,
                value: n_cent,
            },
            Binding::Inline {
                binding_index: 12,
                value: scale.to_bits(),
            },
            Binding::Inline {
                binding_index: 13,
                value: nkv,
            },
            Binding::Inline {
                binding_index: 14,
                value: bs,
            },
            Binding::Inline {
                binding_index: 15,
                value: crate::BLOCKS_PER_CHUNK,
            },
            Binding::Runtime {
                kind: RB::SlotMapping { layer: li },
                binding_index: 16,
            },
            // do_writeback 0: the quantize runs before attention, which reads
            // the step's new keys raw (decode) or rotates them itself (prefill
            // staging).
            Binding::Inline {
                binding_index: 17,
                value: 0,
            },
        ]),
        gemm_dims: None,
    }
}

/// TurboQuant decode: the `AttentionViaCacheTq` twin of a decode
/// `AttentionViaCache` command — the same kernel, geometry and bindings plus
/// `ATTN_TQ_BITS` and the packed-store bindings, so it reads every key but the
/// one this step appended straight from the packed store.
fn tq_attention_command(p: &MetalModelConsts, attn: &LoweredCommand, layer: u32) -> LoweredCommand {
    let mut constants = attn.constants.to_vec();
    constants.extend(Vec::from(
        super::kernel_constants::AttentionViaCacheTqConstants {
            bits: super::ids::TqCodeBits(crate::turboquant::tq_bits(p.tq_kv_bits)),
        },
    ));
    let mut bindings = attn.bindings.to_vec();
    bindings.extend(Vec::from(super::kernel_bindings::TqAttentionBindingSet {
        kv_layer: super::ids::LayerId(layer),
    }));
    LoweredCommand {
        kernel: KernelId::AttentionViaCacheTq,
        constants: baked(constants),
        bindings: baked(bindings),
        ..*attn
    }
}

/// A paged attention instruction: its q and output arena slots, and its
/// decode-kernel form — one query per sequence, the form that reads the
/// TurboQuant packed store directly.
struct PagedAttention {
    q: u32,
    out: u32,
    decode_form: Instruction,
}

fn paged_attention(instruction: &Instruction) -> Option<PagedAttention> {
    match *instruction {
        Instruction::AttentionPrefillPaged(q, out, layer, il)
        | Instruction::AttentionViaCache(q, out, layer, il) => Some(PagedAttention {
            q,
            out,
            decode_form: Instruction::AttentionViaCache(q, out, layer, il),
        }),
        Instruction::SlidingAttentionPrefillPaged(q, out, layer, il)
        | Instruction::SlidingAttentionViaCache(q, out, layer, il) => Some(PagedAttention {
            q,
            out,
            decode_form: Instruction::SlidingAttentionViaCache(q, out, layer, il),
        }),
        _ => None,
    }
}

/// A paged attention lowered for `inject_tq`: its slots and the lowered
/// command of its decode-kernel form.
struct TqAttention {
    q: u32,
    out: u32,
    via_cache: LoweredCommand,
}

/// TurboQuant: inject the per-layer commands, gated on the TurboQuant runtime
/// gates so they're inert unless the worker provisions the tq buffers. No-op
/// unless head_dim is a power of 2 ≤ 512 (the only supported geometry).
///
/// - The KV writer is followed by the quantize of the step's new K/V.
/// - On a decode step — every sequence contributes one token, in any bucket —
///   the attention is the `AttentionViaCacheTq` twin of its decode-kernel
///   form, reading the packed store directly (`OnlyIfTurboquantDecode`).
/// - On any other step the instruction's own attention runs
///   (`UnlessTurboquantDecode`) in the codebook's rotated domain: the K/V
///   staging and q rotation before it and the output's rotation back after it
///   are `OnlyIfTurboquantNotDecode`. A decode tape (`decode`) has no other
///   steps and carries none of them.
fn inject_tq(
    p: &MetalModelConsts,
    instruction: &Instruction,
    cmds: Vec<LoweredCommand>,
    attention: Option<TqAttention>,
    decode: bool,
    bucket_m: u32,
    block_cap: u32,
) -> Vec<GatedCommand> {
    use crate::tape::lowered::RuntimeGate::{
        OnlyIfTurboquant, OnlyIfTurboquantDecode, OnlyIfTurboquantNotDecode, UnlessTurboquantDecode,
    };
    // Supported geometry = pow2 head_dim <= 512 (the kernels' widened threadgroup
    // arrays). Leave the tape byte-identical for anything else.
    // - UNIFORM arches (GLOBAL_HEAD_DIM == HEAD_DIM): quantize every layer at the
    //   base geometry, dequant-before-rope + quantize-after-attention (as before).
    // - HYBRID/SWA arches (gemma4: GLOBAL 512 != base 256): quantize ONLY the
    //   GLOBAL (full-context) group — the memory hog — and leave the sliding group
    //   fp16. Global layers are identified by the KV-writer's threadgroup width
    //   (RopeAppendNormed dispatches head_dim threads, so == GLOBAL_HEAD_DIM means
    //   a global layer). The attention kernel does NOT dispatch head_dim threads,
    //   so for hybrid we put BOTH dequant + quantize around the writer (one
    //   reliable detection point); its global attention is the
    //   `AttentionPrefillPaged` / `AttentionViaCache` instruction (sliding
    //   layers lower the `Sliding*` ones). MUST agree with the factory, which
    //   provisions GLOBAL-sized buffers for the global layers only.
    let hybrid = p.global_head_dim != p.head_dim;
    let base_ok = p.head_dim.is_power_of_two() && p.head_dim <= 512;
    let global_ok = p.global_head_dim.is_power_of_two() && p.global_head_dim <= 512;
    if !base_ok || (hybrid && !global_ok) {
        return cmds.into_iter().map(GatedCommand::ungated).collect();
    }
    // The actual KV layer comes from the rope/attention command's own KvCacheK
    // binding — NOT the loop `iter` (0 for unrolled tapes like Llama).
    use crate::tape::lowered::{Binding, RuntimeBindingKind as RB};
    fn cmd_kv_layer(cmd: &LoweredCommand) -> Option<u32> {
        cmd.bindings.iter().find_map(|b| match b {
            Binding::Runtime {
                kind: RB::KvCacheK { layer },
                ..
            } => Some(layer.get()),
            Binding::Runtime {
                kind: RB::KvCacheV { layer },
                ..
            } => Some(layer.get()),
            _ => None,
        })
    }
    let tq = |c| GatedCommand::gated(c, OnlyIfTurboquant);
    let prefill = |c| GatedCommand::gated(c, OnlyIfTurboquantNotDecode);
    let global_attention = matches!(
        instruction,
        Instruction::AttentionPrefillPaged(..) | Instruction::AttentionViaCache(..)
    );
    if let Some(a) = attention.filter(|_| !hybrid || global_attention) {
        let layer = cmd_kv_layer(&a.via_cache).unwrap_or(0);
        let mut out = Vec::with_capacity(cmds.len() + 5);
        if !decode {
            for is_v in [false, true] {
                out.push(prefill(tq_stage_command(
                    p,
                    layer,
                    is_v,
                    global_attention,
                    bucket_m,
                    block_cap,
                )));
            }
            out.push(prefill(tq_rotate_command(p, a.q, false, bucket_m)));
        }
        out.extend(
            cmds.into_iter()
                .map(|c| GatedCommand::gated(c, UnlessTurboquantDecode)),
        );
        if !decode {
            out.push(prefill(tq_rotate_command(p, a.out, true, bucket_m)));
        }
        out.push(GatedCommand::gated(
            tq_attention_command(p, &a.via_cache, layer),
            OnlyIfTurboquantDecode,
        ));
        return out;
    }
    let mut out = Vec::with_capacity(cmds.len() + 2);
    for cmd in cmds {
        let writer = matches!(
            cmd.kernel,
            KernelId::RopeAppend
                | KernelId::RopeAppendNormed
                | KernelId::FusedQkvRopeCache
                | KernelId::FusedAffineQkvRopeCache
        );
        let tq_layer = !hybrid || cmd.dispatch.threads_per_threadgroup.0 == p.global_head_dim;
        let layer = cmd_kv_layer(&cmd).unwrap_or(0);
        out.push(GatedCommand::ungated(cmd));
        if writer && tq_layer {
            out.push(tq(tq_quantize_command(p, layer, false, bucket_m)));
            out.push(tq(tq_quantize_command(p, layer, true, bucket_m)));
        }
    }
    out
}

/// On a NAX device, the small-M matrix-unit twin of an MLX-affine 4-bit
/// `AffineQmm` in a bucket that can see a `SMALL_M_TOKENS` step: the twin runs
/// on those steps (`OnlyIfSmallMTokens`) and the instruction's own GEMM on
/// every other (`UnlessSmallMTokens`).
fn route_small_m(
    p: &MetalModelConsts,
    instruction: &Instruction,
    cmds: Vec<GatedCommand>,
    bucket_m: u32,
    tape_index: u32,
    index: usize,
    profile: Option<&crate::targets::MetalTargetProfile>,
) -> Vec<GatedCommand> {
    use crate::tape::lowered::RuntimeGate::{OnlyIfSmallMTokens, UnlessSmallMTokens};
    let Instruction::AffineQmm(in_slot, out_slot, layer, n, k, group_size, 4, _) = *instruction
    else {
        return cmds;
    };
    let is_nax = profile.is_some_and(|t| crate::targets::is_nax_capable(t.generation));
    let tile = match SmallMTile::for_bucket(bucket_m) {
        Some(tile)
            if is_nax
                && matches!(group_size, 32 | 64 | 128)
                && n.is_multiple_of(SMALL_M_TILE_COLS) =>
        {
            tile
        }
        _ => return cmds,
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
        }
        .into_baked(),
        dispatch: DispatchShape {
            threadgroups: (n / SMALL_M_TILE_COLS, bucket_m.div_ceil(tile.rows()), 1),
            threads_per_threadgroup: (32, 1, 1),
            m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                seq_axis: None,
                axis: crate::tape::lowered::MScaleAxis::Y,
                bucket_m: super::ids::BucketM(bucket_m),
            }),
        },
        bindings: baked(affine_qmm_bindings(
            in_slot,
            out_slot,
            super::ids::LayerId(layer),
            WeightLocator {
                bucket: tape_index,
                op_idx: index as u32,
                slot: 0,
            },
        )),
        gemm_dims: None,
    };
    cmds.into_iter()
        .map(|c| GatedCommand::gated(c.command, UnlessSmallMTokens))
        .chain([GatedCommand::gated(small_m, OnlyIfSmallMTokens)])
        .collect()
}

/// Lower one bucket's `Instruction` tape.
///
/// `bucket_m` is the bucket point this tape is specialized for —
/// the worker's `forward()` only routes batches with `num_tokens`
/// matching this bucket's `[m_min, m_max_excl)` range here.
/// Dispatch shapes that depend on `M` (token-parallel kernels, GEMM
/// outer dim) are computed against `bucket_m`.
///
/// `num_arena_slots` is the colored slot count from
/// `colored_slot_map()` (in `scratchy-forward-compiler-macro`). The lowered
/// tape carries it through verbatim — the worker uses it to size its
/// per-shape-class arena.
/// A loop span the walk has entered but not yet left.
struct OpenSpan {
    /// First command of the body, in the emitted command stream.
    cmd_start: usize,
    iters: u32,
    layer_stride: u32,
    /// The body's rows in the INSTRUCTION stream, for the shape-state replay.
    body: std::ops::Range<usize>,
}

/// Close every span that ends at `at`, recording its `TapeLoop`.
///
/// ⛔ THE SHAPE STATE STILL ADVANCES FOR EVERY ITERATION. The commands are emitted once, but
/// `update_shape_state` threads the activation width along the tape — so the rows AFTER a loop
/// must see the width all of its iterations produce, not one. The body was already walked once
/// (that walk is what emitted it), so the replay is the remaining `iters - 1`.
#[allow(clippy::too_many_arguments)]
fn close_spans(
    open: &mut Vec<OpenSpan>,
    at: usize,
    loops: &mut Vec<super::lowered::TapeLoop>,
    commands: &[GatedCommand],
    instructions: &[Instruction],
    p: &MetalModelConsts,
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
            p,
            &instructions[s.body.clone()],
            s.iters.saturating_sub(1),
            cur_width,
            m_divisor,
        );
    }
}

/// Replay `update_shape_state` over `rows` `times` times, expanding any nested loops.
fn advance_shape(
    p: &MetalModelConsts,
    rows: &[Instruction],
    times: u32,
    cur_width: &mut crate::tape::lowered::ActivationWidth,
    m_divisor: &mut u32,
) {
    for _ in 0..times {
        let mut k = 0usize;
        while k < rows.len() {
            match rows[k] {
                Instruction::Loop(iters, period, _) => {
                    let body = k + 1..k + 1 + period as usize;
                    advance_shape(p, &rows[body.clone()], iters, cur_width, m_divisor);
                    k = body.end;
                }
                ref other => {
                    update_shape_state(p, other, cur_width, m_divisor);
                    k += 1;
                }
            }
        }
    }
}

pub fn lower(
    p: &MetalModelConsts,
    chunked: bool,
    instructions: &[Instruction],
    barriers_in: &[bool],
    bucket_m: u32,
    num_arena_slots: u32,
    tape_index: u32,
    // Runtime per-sequence block-table capacity (see [`lower_pair`]). Threaded
    // into `lower_one` where the `MaxBlocksPerSeq` function constant + the
    // rope-once scratch sizing read it (replacing `p.max_blocks_per_seq`).
    block_cap: u32,
    profile: Option<&crate::targets::MetalTargetProfile>,
) -> Result<LoweredMetalTape, LoweringError> {
    let mut commands: Vec<GatedCommand> = Vec::with_capacity(instructions.len());
    let mut barrier_before: Vec<bool> = Vec::with_capacity(instructions.len());
    // A decode tape (one query token per sequence) reads the TurboQuant packed
    // store directly in its attention — see `inject_tq`.
    let decode = instructions.iter().any(|i| {
        matches!(
            i,
            Instruction::AttentionViaCache(..) | Instruction::SlidingAttentionViaCache(..)
        )
    });
    // The layer loop, if this tape has one. Set by the `Instruction::Loop` arm below,
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
    // `update_shape_state` after each instruction is lowered, so each
    // arm sees the state in force AT its position. Text arches never set
    // a non-1 divisor (their reshapes carry `dims_div = 1`), so this is
    // inert outside vision towers.
    // Activation row-width in force at each instruction (see `lower_one`).
    // Seeded to the residual-stream width (HIDDEN_SIZE) so the embed-time
    // scalar multiply (granite `* embedding_multiplier`), which precedes
    // the first Gemm that would publish a width, broadcasts over the full
    // residual row rather than a stale 0. Typed as `ActivationWidth` so an
    // elementwise broadcast can never be handed head geometry (Q_SIZE).
    let mut cur_width = crate::tape::lowered::ActivationWidth::residual_stream(p);
    let mut m_divisor: u32 = 1;
    let mut i = 0usize;
    // Macro-static-aligned barrier accessor: the macro emits one bool
    // per `Instruction` (pre-loop-unrolling). Loop expansion at
    // runtime reuses the body's flags across every iteration —
    // body byte-equivalence (the precondition for the macro's loop
    // compression) guarantees the same hazard signature per iter.
    // For metadata-only instructions (Reshape/Alias/Free), no
    // LoweredCommand is emitted so the flag is unused. For the
    // SplitK matmul pair (2 LoweredCommands from 1 Instruction),
    // the first emit takes the macro flag; the second emit gets
    // `true` (intra-Instruction scratch RAW).
    let flag_for = |idx: usize| -> bool { barriers_in.get(idx).copied().unwrap_or(true) };

    while i < instructions.len() {
        close_spans(
            &mut open_spans,
            i,
            &mut loops,
            &commands,
            instructions,
            p,
            &mut cur_width,
            &mut m_divisor,
        );
        match &instructions[i] {
            Instruction::Loop(count, body_len, layer_stride) => {
                // ⭐ RECORD THE SPAN, LOWER THE BODY ONCE, WALK ON. Every row is lowered exactly
                // once no matter how deep it sits, so nesting needs a STACK of open spans rather
                // than a special case: gemma-4 rolls an outer six-layer `SSSSSG` cell with an
                // inner loop over the five sliding layers inside it.
                //
                // `layer_offset` is consumed in exactly one pattern — `LayerId(*layer +
                // layer_offset)` — and the weight locator's `op_idx` is the BODY position,
                // identical across iterations. So iteration `i` IS this body with every
                // `LayerId` advanced by `i * layer_stride`, which `Binding::bump_layer` applies
                // at load. Materialising the copies here is what made the baked tape 14.3M lines
                // of const literals and pinned rustc's front end for ~46s on one file.
                let body_start = i + 1;
                let body_end = body_start
                    .checked_add(*body_len as usize)
                    .filter(|end| *end <= instructions.len())
                    .ok_or_else(|| LoweringError::MalformedLoop {
                        index: i,
                        count: *count,
                        body_len: *body_len,
                        remaining: instructions.len().saturating_sub(body_start),
                    })?;
                open_spans.push(OpenSpan {
                    cmd_start: commands.len(),
                    iters: *count,
                    layer_stride: *layer_stride,
                    body: body_start..body_end,
                });
                i = body_start;
            }
            other => {
                let mut lower_at = |instruction: &Instruction| {
                    lower_one(
                        p,
                        chunked,
                        instruction,
                        i,
                        bucket_m,
                        0,
                        tape_index,
                        &mut splitk_scratch_bytes,
                        &mut moe_scratch_bytes,
                        &mut roped_k_scratch_bytes,
                        &mut attn_unfused_scratch_bytes,
                        block_cap,
                        profile,
                        cur_width,
                        m_divisor,
                    )
                };
                let own = lower_at(other)?;
                // The decode-kernel form of a paged attention: under TurboQuant,
                // decode steps run its `AttentionViaCacheTq` twin (`inject_tq`).
                let attention = match paged_attention(other) {
                    Some(a) => lower_at(&a.decode_form)?
                        .into_iter()
                        .next()
                        .map(|via_cache| TqAttention {
                            q: a.q,
                            out: a.out,
                            via_cache,
                        }),
                    None => None,
                };
                let cmds = route_small_m(
                    p,
                    other,
                    inject_tq(p, other, own, attention, decode, bucket_m, block_cap),
                    bucket_m,
                    tape_index,
                    i,
                    profile,
                );
                update_shape_state(p, other, &mut cur_width, &mut m_divisor);
                let n_cmds = cmds.len();
                commands.extend(cmds);
                if n_cmds >= 1 {
                    barrier_before.push(flag_for(i));
                    barrier_before.extend(std::iter::repeat_n(true, n_cmds - 1));
                }
                i += 1;
            }
        }
    }

    close_spans(
        &mut open_spans,
        instructions.len(),
        &mut loops,
        &commands,
        instructions,
        p,
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

/// Abstract-interpret the activation shape across the instruction
/// stream so the dimensionless vision arms can recover the width / row
/// count the macro already solved — the info is read from the IR,
/// not re-derived.
///
/// - `cur_width` (row-width): set by each `Gemm`'s output-N and by the
///   merger `Reshape`'s static (num_tokens-independent) col dim. The
///   `MeanSubRmsNormBiasAdd` (LayerNorm HIDDEN) and `Gelu` (cols) arms
///   read it; the vision `Add` arm reads it as the residual width.
/// - `m_divisor` (row divisor): the merger `Reshape`
///   `[num_tokens / vision_merge_factor, vision_merge_hidden]` carries
///   `dims_div_lit[row_axis] = vision_merge_factor`; every op after it
///   operates on `bucket_m / vision_merge_factor` rows. Text reshapes
///   carry `dims_div = 1`, so this never fires outside vision towers.
fn update_shape_state(
    p: &MetalModelConsts,
    inst: &Instruction,
    cur_width: &mut crate::tape::lowered::ActivationWidth,
    m_divisor: &mut u32,
) {
    use crate::tape::lowered::ActivationWidth;
    use Instruction as I;
    match inst {
        // A dense GEMM publishes a `[*, n]` activation.
        I::Gemm(_, _, _, n, _) => {
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
        I::AffineQmm(_, _, _, n, _, _, _, _) => {
            *cur_width = ActivationWidth::from_gemm_n(*n);
        }
        // The merger reshape `[num_tokens / div, width]`: the
        // num_tokens-scaling axis (`dims_nt_pow != 0`) carries the row
        // divisor; the static axis (`dims_nt_pow == 0`) is the new width.
        // The divisor is SET unconditionally (including back to 1):
        // Qwen2.5-VL reshapes `[L/S², S²·E]` for the window gather and
        // then BACK to `[L, E]` before the encoder blocks — a sticky
        // divisor would shrink every downstream op's `eff_m` by S².
        // (Qwen3.5-VL's single trailing merger reshape never exposed
        // this.)
        I::Reshape(_, _, dims_lit, dims_nt_pow, dims_div_lit, ndim) => {
            for ax in 0..(*ndim as usize).min(dims_lit.len()) {
                if dims_nt_pow[ax] != 0 {
                    *m_divisor = dims_div_lit[ax].max(1);
                } else {
                    *cur_width = ActivationWidth::from_reshape_width(dims_lit[ax]);
                }
            }
        }
        // AvgPool2d collapses the [ph², e] grid to [(ph/k)², e] — a
        // k²-fold ROW reduction (Gemma3-MM: 4096 patches → 256 tokens
        // at k=4). Fold it into the row divisor so the downstream
        // projector ops dispatch the post-pool row count, not the full
        // patch count (over-dispatch writes past the pooled tile → GPU
        // fault). Width (e) is unchanged.
        I::AvgPool2d(_, _) => {
            let k = p.vision_pool_kernel.max(1);
            *m_divisor *= k * k;
        }
        _ => {}
    }
}

/// Lower one non-`Loop` instruction to zero, one, or many
/// `LoweredCommand`s. Most instructions produce exactly one; metadata-
/// only instructions (`Reshape`/`Alias`/`Free`) produce zero;
/// `Instruction::AffineQmm` in the matmul branch produces two when the
/// dispatcher picks `QmmTKernel::SplitK` (the `affine_qmm_t_splitk`
/// kernel writes a `[split_k, M, N]` partial into a shared scratch
/// buffer, then `splitk_reduce_sum` reduces it into the AffineQmm's
/// arena slot).
///
/// `layer_offset` is the enclosing `Loop`'s iteration index (0 for
/// straight-line code). Combined with each variant's compile-time
/// `layer` literal at weight-binding time, this matches the cuda
/// interpreter's `let layer = ctx.layer_offset + layer;` pattern in
/// `instr.rs`. Without this, every iteration of a `Loop(N, body)`
/// emits identical commands and every per-layer weight lookup
/// resolves to layer 0 — which wedges the model on layer 0's norms,
/// projections, RoPE caches, and KV-cache slots, producing nonsense
/// logits at the end of the chain.
///
/// `splitk_scratch_bytes` is the running max of `split_k * M * N *
/// elem_size_bytes(dtype)` across every `AffineQmm` in this tape that
/// picked `SplitK`. The worker uses it to size the shared scratch
/// buffer that `Binding::Scratch` resolves against.
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

/// The one exit for an opcode metal has no kernel for. `family`
/// names WHICH set of ops was refused, which the error type cannot
/// carry (it holds a `&'static str` and used to be handed
/// `type_name_of_val`, i.e. the enum's own name — the same string for
/// every refusal). Every caller is an enumerated arm at the end of
/// [`lower_one`]'s match; there is no catch-all, so this function
/// cannot absorb a variant nobody considered.
fn no_metal_kernel(
    index: usize,
    family: &'static str,
) -> Result<Vec<LoweredCommand>, LoweringError> {
    Err(LoweringError::UnsupportedVariant {
        index,
        variant_type: family,
    })
}

#[allow(clippy::too_many_arguments)]
fn lower_one(
    p: &MetalModelConsts,
    chunked: bool,
    inst: &Instruction,
    index: usize,
    bucket_m: u32,
    layer_offset: u32,
    tape_index: u32,
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
    use Instruction as I;

    let cmd = match inst {
        // ── Token embedding ────────────────────────────────────────
        I::Embed(out_slot) => LoweredCommand {
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
                Binding::Weight {
                    kind: WeightBundleKind::Embedding,
                    which: WeightTensor::Weight,
                    layer: super::ids::LayerId(0),
                    locator: WeightLocator {
                        bucket: tape_index,
                        op_idx: index as u32,
                        slot: 0,
                    },
                    binding_index: 1,
                },
                // input_ids
                Binding::Runtime {
                    kind: RuntimeBindingKind::InputIds,
                    binding_index: 2,
                },
            ]),
            gemm_dims: None,
        },

        // ── SigLIP learned positional embedding (Gemma3-MM) ────────
        //
        // `pos_embed(position_ids, position_embedding)` — identical to
        // `Embed` (one thread per row gathers `table[idx[i]]`), but the
        // index buffer is the `vision_position_ids` extern instead of
        // `input_ids`, and the table is the learned positional
        // embedding (still `WeightBundleKind::Embedding`, op-local). Row
        // stride = `p.hidden_size` (= vision_embed_dim on the vision
        // Weights after the vision-const fix). Faithful to the cuda
        // `Instruction::PosEmbed` eval (`embedding_gather_masked`).
        I::PosEmbed(out_slot) => LoweredCommand {
            kernel: KernelId::Embed,
            library: "embed",
            function: pick_specialized_symbol(
                "embed_f16_specialized",
                "embed_bf16_specialized",
                p.metal_dtype,
            ),
            constants: super::kernel_constants::EmbedConstants {
                bucket_m: super::ids::BucketM(bucket_m),
                q_size: super::ids::QSize(p.hidden_size as u32),
            }
            .into_baked(),
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
                Binding::ArenaSlot {
                    slot: *out_slot,
                    binding_index: 0,
                },
                Binding::Weight {
                    kind: WeightBundleKind::Embedding,
                    which: WeightTensor::Weight,
                    layer: super::ids::LayerId(0),
                    locator: WeightLocator {
                        bucket: tape_index,
                        op_idx: index as u32,
                        slot: 0,
                    },
                    binding_index: 1,
                },
                // position_ids extern (in place of input_ids)
                Binding::Runtime {
                    kind: RuntimeBindingKind::VisionPositionIds,
                    binding_index: 2,
                },
            ]),
            gemm_dims: None,
        },

        // ── Standalone RMSNorm ─────────────────────────────────────
        // Macro/cuda emit `Instruction::RmsNorm(in_slot, out_slot, ...)`
        // (see `instr.rs:689`). An earlier `(out_slot, in_slot, ...)`
        // pattern here silently swapped the names — the kernel read
        // from a fresh slot and overwrote the upstream tile.
        I::RmsNorm(in_slot, out_slot, layer, hidden_size, m_multiplier) => LoweredCommand {
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
                Binding::Weight {
                    kind: WeightBundleKind::RmsNorm,
                    which: WeightTensor::Weight,
                    layer: super::ids::LayerId(*layer + layer_offset),
                    locator: WeightLocator {
                        bucket: tape_index,
                        op_idx: index as u32,
                        slot: 0,
                    },
                    binding_index: 2,
                },
            ]),
            gemm_dims: None,
        },

        // ── Scalar-offset RMSNorm (Gemma `rmsnorm(x, weight + 1.0)`) ─
        // Identical to `RmsNorm` except the `(1 + w)` offset rides the
        // instruction's `offset` field (the DSL's explicit `+ 1.0`),
        // not the global `p.norm_weight_offset`. `hidden_size` /
        // `m_multiplier` are baked by `ScalarOffsetRmsNormImpl::fan_out`
        // (residual = HIDDEN_SIZE/1; per-head q/k = head_dim/num_heads).
        I::ScalarOffsetRmsNorm(in_slot, out_slot, layer, offset, hidden_size, m_multiplier) => {
            LoweredCommand {
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
                    Binding::Weight {
                        kind: WeightBundleKind::RmsNorm,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 0,
                        },
                        binding_index: 2,
                    },
                ]),
                gemm_dims: None,
            }
        }

        // ── Unit-gain RMSNorm (mlx RMSNormNoScale — Gemma4 v_norm) ─
        // Same row math as `RmsNorm` with gain ≡ 1 and NO weight
        // binding. (hidden_size, m_multiplier) are per-instruction —
        // baked per attention class by `RmsNormUnitImpl::fan_out`
        // (Gemma4: sliding 256×8, global 512×1).
        I::RmsNormUnit(in_slot, out_slot, hidden_size, m_multiplier) => LoweredCommand {
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
        I::ScalarWeightMul(in_slot, out_slot, layer) => LoweredCommand {
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
                Binding::Weight {
                    kind: WeightBundleKind::RmsNorm,
                    which: WeightTensor::Weight,
                    layer: super::ids::LayerId(*layer + layer_offset),
                    locator: WeightLocator {
                        bucket: tape_index,
                        op_idx: index as u32,
                        slot: 0,
                    },
                    binding_index: 2,
                },
            ]),
            gemm_dims: None,
        },

        // ── Fused residual-add + RMSNorm ───────────────────────────
        I::FusedAddRmsNorm(delta_slot, residual_slot, layer, hidden_size, _m_multiplier) => {
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
                    Binding::Weight {
                        kind: WeightBundleKind::RmsNorm,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 0,
                        },
                        binding_index: 2,
                    },
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
        I::FusedAddRmsNormWithOffset(delta_slot, residual_slot, layer, offset) => LoweredCommand {
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
                Binding::Weight {
                    kind: WeightBundleKind::RmsNorm,
                    which: WeightTensor::Weight,
                    layer: super::ids::LayerId(*layer + layer_offset),
                    locator: WeightLocator {
                        bucket: tape_index,
                        op_idx: index as u32,
                        slot: 0,
                    },
                    binding_index: 2,
                },
            ]),
            gemm_dims: None,
        },

        // ── Gemma4 post-FFN tail: rmsnorm → add → scalar_weight_mul ─
        I::NormAddScalarMul(delta_slot, residual_slot, out_slot, layer, hidden_size) => {
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
                    Binding::Weight {
                        kind: WeightBundleKind::RmsNorm,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 0,
                        },
                        binding_index: 3,
                    },
                    // 4: layer_scalar [1] (RmsNorm sub-slot 1)
                    Binding::Weight {
                        kind: WeightBundleKind::RmsNorm,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 1,
                        },
                        binding_index: 4,
                    },
                ]),
                gemm_dims: None,
            }
        }

        // ── Generic dense GEMM ─────────────────────────────────────
        I::Gemm(in_slot, out_slot, layer, n, k) => {
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
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 0,
                        },
                        binding_index: 2,
                    },
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
        // `Instruction::AffineQmm` covers every per-Linear shape on a
        // metal-quantized model. The dispatcher rule mirrors MLX
        // `quantized.cpp:1387 QuantizedMatmul::eval_gpu`:
        //   * `bucket_m < vector_limit` → matvec (qmv_quad / qmv_fast /
        //     qmv) per `dispatch_qmv` (`:1365`) — D∈{64,128} pow2-bits
        //     wins quad, then N%8==0 ∧ K%512==0 wins fast, else generic.
        //   * `bucket_m ≥ vector_limit` → matmul (qmm_t for transpose=true).
        //     SplitK is a sibling of qmm_t Standard for B==1; lands in
        //     C3 alongside its downstream sum-reduce.
        //
        // `vector_limit` rides on the Instruction so the macro can bake
        // it from `get_qmv_batch_limit(K, N, arch_gen)` — see the
        // declaration on `Instruction::AffineQmm`.
        //
        // Bindings match the kernel signatures in `quantized_qmv.metal`
        // and `quantized_qmm.metal`:
        //   buffer(0) packed weight   buffer(1) scales   buffer(2) biases
        //   buffer(3) x activations   buffer(4) y output
        // K / N (and M for qmm_t) ride as function constants 0/1(/2)
        // post the C1 refactor; the dispatcher never sets them as
        // setBytes — required for dispatch recording.
        I::AffineQmm(in_slot, out_slot, layer, n, k, group_size, bits, vector_limit) => {
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
                let kernel = pick_qmv_kernel(n_v, k_v, bits_v);
                let (tg, tpg) = qmv_dispatch_shape(kernel, bucket_m, n_v, /*B=*/ 1);
                let kernel_id = match kernel {
                    QmvKernel::Quad { .. } => KernelId::AffineQmvQuad,
                    QmvKernel::Fast => KernelId::AffineQmvFast,
                    QmvKernel::Generic => KernelId::AffineQmv,
                };
                LoweredCommand {
                    kernel: kernel_id,
                    library: "quantized_qmv",
                    function: qmv_kernel_static_name(kernel, dtype, scale_dtype, bits_v, gs),
                    constants: super::kernel_constants::AffineQmvConstants {
                        k: super::ids::KDimI32(k_v as i32),
                        n: super::ids::NDimI32(n_v as i32),
                    }
                    .into_baked(),
                    dispatch: DispatchShape {
                        threadgroups: tg,
                        threads_per_threadgroup: tpg,
                        // qmv grid x-axis == M directly
                        // (`qmv_dispatch_shape` returns `(m, ceil(N/bn), B)`);
                        // shrinks linearly with actual num_tokens.
                        m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                            seq_axis: None,
                            axis: crate::tape::lowered::MScaleAxis::X,
                            bucket_m: super::ids::BucketM(bucket_m),
                        }),
                    },
                    bindings: baked(affine_qmm_bindings(
                        *in_slot,
                        *out_slot,
                        super::ids::LayerId(*layer + layer_offset),
                        WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 0,
                        },
                    )),
                    gemm_dims: None,
                }
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
                // wrong layout), so `is_nax_capable` is false there. The
                // NAX library is runtime-compiled (`newLibraryWithSource`)
                // because the offline metallib toolchain miscompiles MPP
                // cooperative tensors — see
                // `compile_nax_library_from_source`.
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
                                WeightLocator {
                                    bucket: tape_index,
                                    op_idx: index as u32,
                                    slot: 0,
                                },
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
                                WeightLocator {
                                    bucket: tape_index,
                                    op_idx: index as u32,
                                    slot: 0,
                                },
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
                                WeightLocator {
                                    bucket: tape_index,
                                    op_idx: index as u32,
                                    slot: 0,
                                },
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

        // ── NVFP4 int4 matmul (decode qmv / prefill qmm_t) ────────
        //
        // Mirrors the `AffineQmm` arm but with the NVFP4 kernels (E2M1
        // decode, no per-group bias, gs=16). First cut routes only to
        // the generic `nvfp4_qmv` (decode) and standard `nvfp4_qmm_t`
        // (prefill) — the fast/quad/nax/splitk perf variants are a
        // Phase-2 follow-on. Bindings are 4-buffer (no biases) via
        // `nvfp4_qmm_bindings`.
        I::Nvfp4Qmm(in_slot, out_slot, layer, n, k, group_size, _bits, vector_limit) => {
            let dtype = dequant_dtype_for(p);
            let n_v = *n;
            let k_v = *k;
            let gs = *group_size;
            let vl = *vector_limit;
            let locator = WeightLocator {
                bucket: tape_index,
                op_idx: index as u32,
                slot: 0,
            };
            debug_assert_eq!(gs, 16, "NVFP4 group_size is always 16");

            if bucket_m < vl {
                // Decode matvec — generic qmv only (first cut).
                let kernel = QmvKernel::Generic;
                let (tg, tpg) = qmv_dispatch_shape(kernel, bucket_m, n_v, /*B=*/ 1);
                LoweredCommand {
                    kernel: KernelId::Nvfp4Qmv,
                    library: "quantized_qmv",
                    function: nvfp4_qmv_name(dtype),
                    constants: super::kernel_constants::AffineQmvConstants {
                        k: super::ids::KDimI32(k_v as i32),
                        n: super::ids::NDimI32(n_v as i32),
                    }
                    .into_baked(),
                    dispatch: DispatchShape {
                        threadgroups: tg,
                        threads_per_threadgroup: tpg,
                        m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                            axis: crate::tape::lowered::MScaleAxis::X,
                            bucket_m: super::ids::BucketM(bucket_m),
                            seq_axis: None,
                        }),
                    },
                    bindings: baked(nvfp4_qmm_bindings(
                        *in_slot,
                        *out_slot,
                        super::ids::LayerId(*layer + layer_offset),
                        locator,
                    )),
                    gemm_dims: None,
                }
            } else {
                // Prefill matmul. NAX (Apple9 / M4+) when the chip is
                // NAX-capable and K % 64 == 0 (NAX tile BK=64); else the
                // standard 32×32 qmm_t. NAX is the prefill perf path; the
                // standard kernel is the non-NAX-hardware / K%64≠0 fallback.
                let is_nax = profile.is_some_and(|p| crate::targets::is_nax_capable(p.generation))
                    && k_v.is_multiple_of(64);
                if is_nax {
                    // NAX tile is 64×64 → N alignment check uses 64.
                    let aligned_n = n_v.is_multiple_of(64);
                    let (tg, tpg) =
                        qmm_t_dispatch_shape(QmmTKernel::Nax, bucket_m, n_v, /*B=*/ 1);
                    LoweredCommand {
                        kernel: KernelId::Nvfp4QmmTNax,
                        library: "quantized_qmm_nax",
                        function: nvfp4_qmm_t_nax_name(dtype, aligned_n),
                        constants: super::kernel_constants::AffineQmmTConstants {
                            k: super::ids::KDimI32(k_v as i32),
                            n: super::ids::NDimI32(n_v as i32),
                            m: super::ids::MDimI32(bucket_m as i32),
                        }
                        .into_baked(),
                        dispatch: DispatchShape {
                            threadgroups: tg,
                            threads_per_threadgroup: tpg,
                            // qmm_t NAX grid = (n_tiles, m_tiles=ceil(M/64), B)
                            m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                                axis: crate::tape::lowered::MScaleAxis::Y,
                                bucket_m: super::ids::BucketM(bucket_m),
                                seq_axis: None,
                            }),
                        },
                        bindings: baked(nvfp4_qmm_bindings(
                            *in_slot,
                            *out_slot,
                            super::ids::LayerId(*layer + layer_offset),
                            locator,
                        )),
                        gemm_dims: None,
                    }
                } else {
                    // Standard 32×32 qmm_t fallback. N alignment uses 32.
                    let kernel = QmmTKernel::Standard;
                    let aligned_n = n_v.is_multiple_of(32);
                    let (tg, tpg) = qmm_t_dispatch_shape(kernel, bucket_m, n_v, /*B=*/ 1);
                    LoweredCommand {
                        kernel: KernelId::Nvfp4QmmT,
                        library: "quantized_qmm",
                        function: nvfp4_qmm_t_name(dtype, aligned_n),
                        constants: super::kernel_constants::AffineQmmTConstants {
                            k: super::ids::KDimI32(k_v as i32),
                            n: super::ids::NDimI32(n_v as i32),
                            m: super::ids::MDimI32(bucket_m as i32),
                        }
                        .into_baked(),
                        dispatch: DispatchShape {
                            threadgroups: tg,
                            threads_per_threadgroup: tpg,
                            m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                                axis: crate::tape::lowered::MScaleAxis::Y,
                                bucket_m: super::ids::BucketM(bucket_m),
                                seq_axis: None,
                            }),
                        },
                        bindings: baked(nvfp4_qmm_bindings(
                            *in_slot,
                            *out_slot,
                            super::ids::LayerId(*layer + layer_offset),
                            locator,
                        )),
                        gemm_dims: None,
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
        I::SiluMul(gate_slot, up_slot, out_slot, width) => {
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
        I::GeluMul(gate_slot, up_slot, out_slot) => {
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
        I::GateApply(attn_slot, gate_slot, out_slot) => {
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
        I::GateScale(routed_slot, shared_slot, gate_slot, out_slot) => {
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
        I::GateSplit(qg_slot, q_slot, gate_slot) => {
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
        // `Instruction::AffineEmbed` replaces `Instruction::Embed`
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
        I::AffineEmbed(out_slot, group_size, bits) => {
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
                    Binding::Weight {
                        kind: WeightBundleKind::AffineQuantEmbedding,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(0),
                        locator: WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 0,
                        },
                        binding_index: 0,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::AffineQuantEmbedding,
                        which: WeightTensor::AffineScales,
                        layer: super::ids::LayerId(0),
                        locator: WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 0,
                        },
                        binding_index: 1,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::AffineQuantEmbedding,
                        which: WeightTensor::AffineBiases,
                        layer: super::ids::LayerId(0),
                        locator: WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 0,
                        },
                        binding_index: 2,
                    },
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
        I::FusedGateUpSiluMul(in_slot, out_slot, layer) => fused_gate_up_mul_cmd(
            p,
            *in_slot,
            *out_slot,
            *layer,
            false,
            bucket_m,
            tape_index,
            index,
            layer_offset,
        ),
        // Dense GeGLU (Gemma3 text MLP) — same fused gate/up GEMM +
        // activation-mul kernel as SiLU; the `IS_GELU` fn-const flips
        // the epilogue to gelu_approx (tanh). The quant GeGLU path
        // decomposes to AffineQmm + GeluMul instead.
        I::FusedGateUpGeluMul(in_slot, out_slot, layer) => fused_gate_up_mul_cmd(
            p,
            *in_slot,
            *out_slot,
            *layer,
            true,
            bucket_m,
            tape_index,
            index,
            layer_offset,
        ),

        // ── RoPE + KV cache append ─────────────────────────────────
        I::RopeAppend(
            _q_slot,
            _k_slot,
            _v_slot,
            q_out_slot,
            k_out_slot,
            v_out_slot,
            layer,
            _interleaved,
            is_global,
        ) => {
            // 2D dispatch: (M, num_heads) — one threadgroup per
            // (token, head) pair rotates the head's `head_dim` slice
            // and writes K/V to the layer's paged cache page.
            //
            // Geometry class (Gemma4): global tiles use GLOBAL_*
            // (512 head_dim / 1 kv head / rot 128 proportional);
            // sliding tiles use the base consts (256 / 8 / full).
            // Identity on uniform models.
            let (hd, n_kv, rd, bs) = if *is_global {
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
            let pair_off = if *is_global && p.rope_proportional {
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
                    cos_sin_locator: crate::tape::lowered::WeightLocator {
                        bucket: tape_index,
                        op_idx: index as u32,
                        slot: 0,
                    },
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
            k_slot,
            v_slot,
            q_out_slot,
            _k_out_slot,
            _v_out_slot,
            layer,
            interleaved,
            is_global,
        ) => {
            assert!(
                !*interleaved,
                "metal lowering: RopeAppendNormed is NeoX-only (the matcher \
                 requires OpKind::RopeAppend)"
            );
            // Same geometry-class selection as the RopeAppend arm.
            let (hd, n_kv, rd, bs) = if *is_global {
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
            let pair_off = if *is_global && p.rope_proportional {
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
                    cos_sin_locator: crate::tape::lowered::WeightLocator {
                        bucket: tape_index,
                        op_idx: index as u32,
                        slot: 0,
                    },
                    q_gains_locator: crate::tape::lowered::WeightLocator {
                        bucket: tape_index,
                        op_idx: index as u32,
                        slot: 0,
                    },
                    k_gains_locator: crate::tape::lowered::WeightLocator {
                        bucket: tape_index,
                        op_idx: index as u32,
                        slot: 1,
                    },
                    layer: super::ids::LayerId(*layer + layer_offset),
                }
                .into_baked(),
                gemm_dims: None,
            }
        }

        // ── Fused QKV matmul + RoPE + paged KV-cache write ─────────
        // Dense BF16/F16 path; Llama-style NeoX, no QKV bias. Qwen2
        // bias / Cohere interleaved variants land in follow-up
        // commits, gated at the matcher.
        I::FusedQkvRopeCache(in_slot, out_slot, layer, biased, interleaved) => {
            assert!(
                !*biased && !*interleaved,
                "metal lowering: FusedQkvRopeCache currently supports only \
                 NeoX-style RoPE without QKV bias (biased={biased} \
                 interleaved={interleaved}); the matcher should not have \
                 claimed this shape",
            );
            let n_q_heads = p.num_q_heads;
            let n_kv_heads = p.num_kv_heads;
            let num_heads_total = n_q_heads + 2 * n_kv_heads;
            LoweredCommand {
                kernel: KernelId::FusedQkvRopeCache,
                library: "fused_qkv_rope_cache",
                function: pick_specialized_symbol(
                    "fused_qkv_rope_cache_f16_specialized",
                    "fused_qkv_rope_cache_bf16_specialized",
                    p.metal_dtype,
                ),
                constants: super::kernel_constants::FusedQkvRopeCacheConstants {
                    q_size: super::ids::QSize(p.q_size as u32),
                    num_q_heads: super::ids::NumQHeads(p.num_q_heads),
                    num_kv_heads: super::ids::NumKvHeads(p.num_kv_heads),
                    head_dim: super::ids::HeadDim(p.head_dim),
                    rot_dim: super::ids::RotDim(p.rot_dim),
                    block_size: super::ids::BlockSize(p.block_size),
                    bucket_m: super::ids::BucketM(bucket_m),
                    blocks_per_chunk: super::ids::BlocksPerChunk(crate::BLOCKS_PER_CHUNK),
                    // Spans: store K unrotated for bit-31 (relocatable) blocks.
                    rope_on_read: if p.rope_on_read { Some(1) } else { None },
                }
                .into_baked(),
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, num_heads_total, 1),
                    threads_per_threadgroup: (p.head_dim, 1, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    }),
                },
                bindings: super::kernel_bindings::FusedQkvRopeCacheBindingSet {
                    q_out: super::ids::ArenaSlotIdx(*out_slot),
                    input: super::ids::ArenaSlotIdx(*in_slot),
                    qkv_locator: crate::tape::lowered::WeightLocator {
                        bucket: tape_index,
                        op_idx: index as u32,
                        slot: 0,
                    },
                    cos_sin_locator: crate::tape::lowered::WeightLocator {
                        bucket: tape_index,
                        op_idx: index as u32,
                        slot: 1,
                    },
                    layer: super::ids::LayerId(*layer + layer_offset),
                }
                .into_baked(),
                gemm_dims: None,
            }
        }

        // ── Compiler-synthesized pre-attention chunk ───────────────
        // Generated by scratchy-forward-compiler-macro::fuse_pass at macro time;
        // bound at runtime via the SpecializedPipelineCache's
        // source-library registry (populated at worker-pool init from
        // the macro-emitted per-arch SYNTHESIZED_KERNEL_SOURCES const).
        //
        // Same dispatch shape as the hand-written
        // fused_add_rmsnorm_affine_qkv_rope_cache kernel:
        //   threadgroups = (M, NUM_Q + 2*NUM_KV, 1)
        //   threads_per_tg = MK_SIMD_SIZE * HEAD_DIM / MK_ROWS_PER_SIMDGROUP
        I::SynthPreAttn(
            residual_slot,
            delta_slot,
            q_out_slot,
            residual_out_slot,
            layer,
            group_size,
            bits,
            symbol,
            has_linear_bias,
        ) => {
            assert_eq!(
                *bits, 4,
                "metal lowering: SynthPreAttn only wired for bits=4"
            );
            let _ = group_size; // baked into the symbol name; reserved for future per-gs dispatch tuning
            let n_q_heads = p.num_q_heads;
            let n_kv_heads = p.num_kv_heads;
            let num_heads_total = n_q_heads + 2 * n_kv_heads;
            let threads_per_tg = 32 * p.head_dim / 4;
            LoweredCommand {
                kernel: KernelId::SynthPreAttn,
                // Library name = the synthesized kernel's symbol; the
                // SpecializedPipelineCache stores one synthesized
                // library per symbol so `library == function` here.
                library: symbol,
                function: symbol,
                // Pre-attn synth kernel bakes HIDDEN / NUM_Q / NUM_KV /
                // HEAD_DIM / ROT_DIM / BLOCK_SIZE / EPS as MSL
                // `constant constexpr` literals at synth time. Only
                // `M` (active token count up to bucket capacity)
                // stays a function constant — varies per bucket.
                constants: super::kernel_constants::SynthMegakernelConstants {
                    bucket_m: super::ids::BucketM(bucket_m),
                }
                .into_baked(),
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, num_heads_total, 1),
                    threads_per_threadgroup: (threads_per_tg, 1, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    }),
                },
                bindings: {
                    let mut v: Vec<Binding> = vec![
                        // 0: q_out
                        Binding::ArenaSlot {
                            slot: *q_out_slot,
                            binding_index: 0,
                        },
                        // 1: residual_io (read+write)
                        Binding::ArenaSlot {
                            slot: *residual_slot,
                            binding_index: 1,
                        },
                        // 2: delta (read)
                        Binding::ArenaSlot {
                            slot: *delta_slot,
                            binding_index: 2,
                        },
                        // 3: rms_weight
                        Binding::Weight {
                            kind: WeightBundleKind::RmsNorm,
                            which: WeightTensor::Weight,
                            layer: super::ids::LayerId(*layer + layer_offset),
                            locator: WeightLocator {
                                bucket: tape_index,
                                op_idx: index as u32,
                                slot: 0,
                            },
                            binding_index: 3,
                        },
                        // 4..6: Q weight + scales + biases
                        Binding::Weight {
                            kind: WeightBundleKind::LinearLayer,
                            which: WeightTensor::Weight,
                            layer: super::ids::LayerId(*layer + layer_offset),
                            locator: WeightLocator {
                                bucket: tape_index,
                                op_idx: index as u32,
                                slot: 0,
                            },
                            binding_index: 4,
                        },
                        Binding::Weight {
                            kind: WeightBundleKind::LinearLayer,
                            which: WeightTensor::AffineScales,
                            layer: super::ids::LayerId(*layer + layer_offset),
                            locator: WeightLocator {
                                bucket: tape_index,
                                op_idx: index as u32,
                                slot: 0,
                            },
                            binding_index: 5,
                        },
                        Binding::Weight {
                            kind: WeightBundleKind::LinearLayer,
                            which: WeightTensor::AffineBiases,
                            layer: super::ids::LayerId(*layer + layer_offset),
                            locator: WeightLocator {
                                bucket: tape_index,
                                op_idx: index as u32,
                                slot: 0,
                            },
                            binding_index: 6,
                        },
                        // 7..9: K weight + scales + biases (LinearLayer sub-slot 1)
                        Binding::Weight {
                            kind: WeightBundleKind::LinearLayer,
                            which: WeightTensor::Weight,
                            layer: super::ids::LayerId(*layer + layer_offset),
                            locator: WeightLocator {
                                bucket: tape_index,
                                op_idx: index as u32,
                                slot: 1,
                            },
                            binding_index: 7,
                        },
                        Binding::Weight {
                            kind: WeightBundleKind::LinearLayer,
                            which: WeightTensor::AffineScales,
                            layer: super::ids::LayerId(*layer + layer_offset),
                            locator: WeightLocator {
                                bucket: tape_index,
                                op_idx: index as u32,
                                slot: 1,
                            },
                            binding_index: 8,
                        },
                        Binding::Weight {
                            kind: WeightBundleKind::LinearLayer,
                            which: WeightTensor::AffineBiases,
                            layer: super::ids::LayerId(*layer + layer_offset),
                            locator: WeightLocator {
                                bucket: tape_index,
                                op_idx: index as u32,
                                slot: 1,
                            },
                            binding_index: 9,
                        },
                        // 10..12: V weight + scales + biases (LinearLayer sub-slot 2)
                        Binding::Weight {
                            kind: WeightBundleKind::LinearLayer,
                            which: WeightTensor::Weight,
                            layer: super::ids::LayerId(*layer + layer_offset),
                            locator: WeightLocator {
                                bucket: tape_index,
                                op_idx: index as u32,
                                slot: 2,
                            },
                            binding_index: 10,
                        },
                        Binding::Weight {
                            kind: WeightBundleKind::LinearLayer,
                            which: WeightTensor::AffineScales,
                            layer: super::ids::LayerId(*layer + layer_offset),
                            locator: WeightLocator {
                                bucket: tape_index,
                                op_idx: index as u32,
                                slot: 2,
                            },
                            binding_index: 11,
                        },
                        Binding::Weight {
                            kind: WeightBundleKind::LinearLayer,
                            which: WeightTensor::AffineBiases,
                            layer: super::ids::LayerId(*layer + layer_offset),
                            locator: WeightLocator {
                                bucket: tape_index,
                                op_idx: index as u32,
                                slot: 2,
                            },
                            binding_index: 12,
                        },
                        // 13: cos_sin
                        Binding::Weight {
                            kind: WeightBundleKind::CosSin,
                            which: WeightTensor::Weight,
                            layer: super::ids::LayerId(*layer + layer_offset),
                            locator: WeightLocator {
                                bucket: tape_index,
                                op_idx: index as u32,
                                slot: 0,
                            },
                            binding_index: 13,
                        },
                        // 14: positions
                        Binding::Runtime {
                            kind: RuntimeBindingKind::Positions,
                            binding_index: 14,
                        },
                        // 15: slot_mapping (this layer's KV-cache group)
                        Binding::Runtime {
                            kind: RuntimeBindingKind::SlotMapping {
                                layer: super::ids::LayerId(*layer + layer_offset),
                            },
                            binding_index: 15,
                        },
                        // 16: kv_cache_k
                        Binding::Runtime {
                            kind: RuntimeBindingKind::KvCacheK {
                                layer: super::ids::LayerId(*layer + layer_offset),
                            },
                            binding_index: 16,
                        },
                        // 17: kv_cache_v
                        Binding::Runtime {
                            kind: RuntimeBindingKind::KvCacheV {
                                layer: super::ids::LayerId(*layer + layer_offset),
                            },
                            binding_index: 17,
                        },
                    ];
                    if *has_linear_bias {
                        // 18..20: Q / K / V linear bias (Qwen2-style
                        // per-row bias on each QKV LinearLayer). The
                        // worker resolves `AffineLinearBias` against
                        // `LinearLayer::AffineQuant.linear_bias` —
                        // load_affine_quant auto-detects `<prefix>.bias`
                        // in the safetensors. Bias-free arches never
                        // reach this branch.
                        v.push(Binding::Weight {
                            kind: WeightBundleKind::LinearLayer,
                            which: WeightTensor::AffineLinearBias,
                            layer: super::ids::LayerId(*layer + layer_offset),
                            locator: WeightLocator {
                                bucket: tape_index,
                                op_idx: index as u32,
                                slot: 0,
                            },
                            binding_index: 18,
                        });
                        v.push(Binding::Weight {
                            kind: WeightBundleKind::LinearLayer,
                            which: WeightTensor::AffineLinearBias,
                            layer: super::ids::LayerId(*layer + layer_offset),
                            locator: WeightLocator {
                                bucket: tape_index,
                                op_idx: index as u32,
                                slot: 1,
                            },
                            binding_index: 19,
                        });
                        v.push(Binding::Weight {
                            kind: WeightBundleKind::LinearLayer,
                            which: WeightTensor::AffineLinearBias,
                            layer: super::ids::LayerId(*layer + layer_offset),
                            locator: WeightLocator {
                                bucket: tape_index,
                                op_idx: index as u32,
                                slot: 2,
                            },
                            binding_index: 20,
                        });
                    }
                    // 21: residual_out (write target — distinct from
                    // residual_io at buffer 1). Fixed index past the
                    // optional 18..20 bias block so it's stable across
                    // the Llama (no-bias, sparse 18..20) and Qwen paths.
                    // The kernel reads buffer 1 and writes the updated
                    // residual here, never in place.
                    v.push(Binding::ArenaSlot {
                        slot: *residual_out_slot,
                        binding_index: 21,
                    });
                    baked(v)
                },
                gemm_dims: None,
            }
        }

        // ── Compiler-synthesized MLP pre-down megakernel ───────────
        // Mirrors SynthPreAttn but for the (FusedAddRmsNorm + gate qmv
        // + up qmv + SiluMul) chain. Output `silu_mul_out_slot` is a
        // device buffer of shape `[M, intermediate_size]` consumed by
        // the down-projection's standalone AffineQmm.
        //
        // Dispatch shape:
        //   threadgroups = (M, INTERMEDIATE_SIZE / HEAD_DIM, 1)
        //   threads_per_tg = MK_SIMD_SIZE * HEAD_DIM / MK_ROWS_PER_SIMDGROUP
        // The qmv atom is reused unchanged by aliasing kernel-scope
        // `__head_dim` = TILE_N and `__head` = tile index. TILE_N is
        // set to HEAD_DIM so the same `32 * HEAD_DIM / 4` thread count
        // and per-simdgroup row layout carries over from pre-attn.
        I::SynthMlpPreDown(
            residual_slot,
            delta_slot,
            silu_mul_out_slot,
            residual_out_slot,
            layer,
            group_size,
            bits,
            symbol,
        ) => {
            assert!(
                matches!(*bits, 4 | 8),
                "metal lowering: SynthMlpPreDown only wired for bits=4/8"
            );
            let _ = group_size;
            // Mirrors `synthesize_mlp_pre_down_chunk`'s TILE_N bake:
            // min(HEAD_DIM, 128) keeps `threads_per_tg = 32*TILE_N/4`
            // within Metal's 1024 ceiling (Gemma4 head_dim=256).
            let tile_n = p.head_dim.min(128);
            let intermediate = p.intermediate_size as u32;
            let num_tiles = intermediate / tile_n;
            assert!(
                intermediate.is_multiple_of(tile_n),
                "metal lowering: SynthMlpPreDown requires INTERMEDIATE_SIZE \
                 ({intermediate}) divisible by HEAD_DIM ({tile_n})"
            );
            let threads_per_tg = 32 * tile_n / 4;
            LoweredCommand {
                kernel: KernelId::SynthMlpPreDown,
                library: symbol,
                function: symbol,
                // MLP-pre-down synth kernel bakes HIDDEN / INTERMEDIATE /
                // TILE_N / EPS as MSL `constant constexpr` literals at
                // synth time. Only `M_FC` stays a function constant.
                constants: super::kernel_constants::SynthMegakernelConstants {
                    bucket_m: super::ids::BucketM(bucket_m),
                }
                .into_baked(),
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, num_tiles, 1),
                    threads_per_threadgroup: (threads_per_tg, 1, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(bucket_m),
                    }),
                },
                bindings: baked(vec![
                    // 0: silu_mul_out
                    Binding::ArenaSlot {
                        slot: *silu_mul_out_slot,
                        binding_index: 0,
                    },
                    // 1: residual_io (read+write)
                    Binding::ArenaSlot {
                        slot: *residual_slot,
                        binding_index: 1,
                    },
                    // 2: delta (read)
                    Binding::ArenaSlot {
                        slot: *delta_slot,
                        binding_index: 2,
                    },
                    // 3: rms_weight
                    Binding::Weight {
                        kind: WeightBundleKind::RmsNorm,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 0,
                        },
                        binding_index: 3,
                    },
                    // 4..6: gate weight + scales + biases
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 0,
                        },
                        binding_index: 4,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineScales,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 0,
                        },
                        binding_index: 5,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineBiases,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 0,
                        },
                        binding_index: 6,
                    },
                    // 7..9: up weight + scales + biases (LinearLayer sub-slot 1)
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 1,
                        },
                        binding_index: 7,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineScales,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 1,
                        },
                        binding_index: 8,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineBiases,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 1,
                        },
                        binding_index: 9,
                    },
                    // 10: residual_out (write target — distinct from
                    // residual_io at buffer 1; the kernel reads buffer 1
                    // and writes the updated residual here, never in
                    // place, to avoid the cross-threadgroup race).
                    Binding::ArenaSlot {
                        slot: *residual_out_slot,
                        binding_index: 10,
                    },
                ]),
                gemm_dims: None,
            }
        }

        // ── SynthGateUpSiluMul — fused gate+up GEMM + SiluMul (large-M) ─────
        I::SynthGateUpSiluMul(x_norm_slot, out_slot, layer, _group_size, _bits, symbol) => {
            let intermediate = p.intermediate_size as u32;
            let tg_n = 32u32;
            let tg_m = 32u32;
            LoweredCommand {
                kernel: KernelId::SynthGateUpSiluMul,
                library: symbol,
                function: symbol,
                constants: super::kernel_constants::SynthMegakernelConstants {
                    bucket_m: super::ids::BucketM(bucket_m),
                }
                .into_baked(),
                dispatch: DispatchShape {
                    threadgroups: (intermediate.div_ceil(tg_n), bucket_m.div_ceil(tg_m), 1),
                    threads_per_threadgroup: (128, 1, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::Y,
                        bucket_m: super::ids::BucketM(bucket_m),
                    }),
                },
                bindings: baked(vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *x_norm_slot,
                        binding_index: 1,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 0,
                        },
                        binding_index: 2,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineScales,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 0,
                        },
                        binding_index: 3,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineBiases,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 0,
                        },
                        binding_index: 4,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 1,
                        },
                        binding_index: 5,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineScales,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 1,
                        },
                        binding_index: 6,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineBiases,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 1,
                        },
                        binding_index: 7,
                    },
                ]),
                gemm_dims: None,
            }
        }

        // ── Decode-bucket attention (single query token / seq) ─────
        I::AttentionViaCache(q_slot, out_slot, layer, _is_decode) => {
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
                    rope_on_read: ror_bind,
                }
                .into_baked(),
                gemm_dims: None,
            }
        }

        // ── Plain prefill on metal goes through `AttentionPrefillPaged` ─
        // Phase B (`crates/scratchy-forward-compiler-macro/src/metal/attention.rs`
        // `(false, true)` arm) emits `Instruction::AttentionPrefillPaged`
        // for every metal Llama-arch model — the contiguous variant
        // is CUDA-only on metal builds. Hitting this arm means a
        // future macro change started emitting the wrong variant.
        I::AttentionPrefillContiguous(_, _, _, _, _) => {
            unreachable!(
                "metal lowering: AttentionPrefillContiguous is unreachable post-Phase B \
                 — metal::attention::fan_out emits AttentionPrefillPaged. \
                 If this fires, a macro adapter for a metal model is emitting the wrong variant."
            );
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
        I::AttentionPrefillPaged(q_slot, out_slot, layer, _interleaved) => {
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
            #[allow(clippy::overly_complex_bool_expr)]
            if HD512_UNFUSED_CONTINUATION_OK && p.global_head_dim > 256 && p.rope_on_read {
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
                let cossin = |bi: u8| Binding::Weight {
                    kind: WeightBundleKind::RopeOnReadCosSin { is_global: true },
                    which: WeightTensor::Weight,
                    layer: lid,
                    locator: WeightLocator {
                        bucket: 0,
                        op_idx: 0,
                        slot: 0,
                    },
                    binding_index: bi,
                };
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
                return Ok(cmds);
            }
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
            use crate::tape::kernel_identity::{AttentionSdpaPagedBf16, AttentionSdpaPagedF16};
            // Spans rope-on-read (GLOBAL class). All-None when !ROPE_ON_READ.
            let (ror_rd, ror_po, ror_on, ror_bind) = rope_on_read_params(p, true);
            let n_q_heads = p.num_q_heads;
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
            // falls through to the simdgroup steel path below. The library
            // is runtime-compiled at startup in `specialized_pipeline_cache`
            // (offline metallib miscompiles MPP).
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
            let gqa_shared_spans = use_gqa_shared && p.rope_on_read;
            let (tg_shape, threads_per_tg, m_scale_axis) = if use_steel {
                let nq_blocks = bucket_m.div_ceil(bq_steel);
                (
                    (nq_blocks, n_q_heads, 1),
                    (128u32, 1u32, 1u32),
                    crate::tape::lowered::MScaleAxis::X,
                )
            } else {
                (
                    (n_q_heads, bucket_m, 1),
                    (1024u32, 1u32, 1u32),
                    crate::tape::lowered::MScaleAxis::Y,
                )
            };
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
            // ⛔⛔⛔⭐⭐⭐⭐⭐ THE SCRATCH PATH IS WRONG FOR num_reqs > 1, AND THIS IS WHERE THE FIX GOES.
            // `RopeOnce{Nax,Steel,GqaShared}` hard-codes `seq_idx = 0` (attention.metal:1263 and the two
            // mlx_steel_attn paged headers) and the scratch has NO batch dimension, so it stages ONE K
            // image from batch row 0 and serves it to every row. MEASURED on metal 2026-08-11: 12 short
            // distinct prompts score 1/12, and the only correct one is the request that prefilled ALONE;
            // the same file with `--max-num-seqs 1` scores 4/4.
            //
            // ⭐ THE CORRECT PATH ALREADY EXISTS AND IS THE `else` OF THIS VERY FLAG: the sdpa-paged
            // kernel with the in-kernel cos_sin path reads `block_table + seq_idx * max_blocks`, i.e. each
            // request's OWN row (attention.metal ~854). So the fix is to route a multi-sequence step to
            // that path instead of building a batch-aware scratch — adding a batch dimension is
            // ~268 MB PER SEQUENCE at this config (see the sizing below, `num_pages = block_cap`).
            //
            // ⛔ WHAT MAKES IT MORE THAN A ONE-LINER: `use_nax`/`use_steel` are decided HERE, at lowering
            // time, and a command's kernel + bindings are fixed once lowered — while "how many sequences
            // are in this step" is a RUNTIME fact (a mixed step's `m` is total tokens, indistinguishable
            // from a single long prefill's `m`). So this needs either runtime kernel selection (two
            // lowered variants chosen per step) or a scratch packed per sequence with runtime offsets.
            // That is a real design decision with a single-sequence perf cost, not a guess to make blind.
            let nax_spans = use_nax && p.rope_on_read;
            let steel_spans = use_steel && !use_nax && p.rope_on_read;
            // NAX, simdgroup steel, and gqa_shared all read pre-roped K from the
            // shared `Binding::RopedKScratch` at slot 7 (the rope-once-to-scratch
            // pattern); they share the scratch-source binding flag.
            let roped_k_scratch = nax_spans || steel_spans || gqa_shared_spans;
            let bindings = super::kernel_bindings::AttentionPrefillPagedBindingSet {
                output: super::ids::ArenaSlotIdx(*out_slot),
                q: super::ids::ArenaSlotIdx(*q_slot),
                kv_layer: super::ids::LayerId(*layer + layer_offset),
                // Steel/NAX/gqa_shared read pre-roped K from the scratch, so
                // they do NOT bind cos_sin at slot 7 (the rope is done by
                // RopeOnce{Nax,Steel,GqaShared}). The sdpa-paged path keeps the
                // in-kernel cos_sin path.
                rope_on_read: if roped_k_scratch { None } else { ror_bind },
                nax_roped_k_scratch: roped_k_scratch,
            };
            let dispatch = DispatchShape {
                threadgroups: tg_shape,
                threads_per_threadgroup: threads_per_tg,
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    // Steel tiles queries in BQ-blocks that must not
                    // straddle a sequence boundary, so its grid needs one
                    // Z-layer per sequence (`tid.z = seq_idx`). SDPA is
                    // per-query-token and self-attributes, so it leaves Z
                    // alone. See MScaling::seq_axis.
                    seq_axis: if use_steel {
                        Some(crate::interpreter::metal::lowered::MScaleAxis::Z)
                    } else {
                        None
                    },
                    axis: m_scale_axis,
                    bucket_m: super::ids::BucketM(bucket_m),
                }),
            };
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
            if use_steel {
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
                    dispatch,
                    bindings: bindings.into_baked(),
                    gemm_dims: None,
                };
                if roped_k_scratch {
                    // Two commands: (1) RopeOnce{Nax,Steel} ropes the cache's K
                    // into the shared scratch (sized per-layer below); (2) the
                    // steel/NAX attention reads pre-roped K from the scratch.
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
                            is_global: true,
                        }
                        .into_baked(),
                        gemm_dims: None,
                    };
                    return Ok(vec![rope_cmd, attn_cmd]);
                }
                attn_cmd
            } else if use_gqa_shared {
                let attn_cmd = LoweredCommand {
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
                };
                if gqa_shared_spans {
                    // Two commands: (1) RopeOnceGqaShared ropes the cache's K
                    // into the shared scratch (dense, logical-block indexed);
                    // (2) the gqa_shared attention reads pre-roped K from the
                    // scratch (slot 7, ATTN_K_SCRATCH set) with no per-tile smem
                    // rotation. head_dim 512 (gemma4 global) has no steel/NAX
                    // instantiation, so this is the path launch-claude takes.
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
                            is_global: true,
                        }
                        .into_baked(),
                        gemm_dims: None,
                    };
                    return Ok(vec![rope_cmd, attn_cmd]);
                }
                attn_cmd
            } else {
                match p.metal_dtype {
                    crate::tape::lowered::MetalDtype::Bf16 => {
                        LoweredCommand::for_kernel::<AttentionSdpaPagedBf16>(
                            constants, bindings, dispatch,
                        )
                    }
                    _ => LoweredCommand::for_kernel::<AttentionSdpaPagedF16>(
                        constants, bindings, dispatch,
                    ),
                }
            }
        }

        // ── Sliding-window decode attention (Gemma2/3/4 local layers) ─
        // Identical to `AttentionViaCache` except the kernel's
        // `ATTN_WINDOW` function constant carries `p.sliding_window`
        // (the decode kernel skips keys with `kv_len-1 - k >= window`).
        I::SlidingAttentionViaCache(q_slot, out_slot, layer, _is_decode) => {
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
                    rope_on_read: ror_bind,
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
        I::SlidingAttentionPrefillPaged(q_slot, out_slot, layer, _interleaved) => {
            debug_assert!(
                p.sliding_window > 0,
                "SlidingAttentionPrefillPaged lowered with SLIDING_WINDOW <= 0"
            );
            use crate::steel_paged::steel_paged_symbol;
            use crate::tape::kernel_identity::{AttentionSdpaPagedBf16, AttentionSdpaPagedF16};
            // Spans rope-on-read (SLIDING class). All-None when !ROPE_ON_READ.
            let (ror_rd, ror_po, ror_on, ror_bind) = rope_on_read_params(p, false);
            let n_q_heads = p.num_q_heads;
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
            let (tg_shape, threads_per_tg, m_scale_axis) = if use_steel {
                let nq_blocks = bucket_m.div_ceil(BQ_STEEL);
                (
                    (nq_blocks, n_q_heads, 1),
                    (128u32, 1u32, 1u32),
                    crate::tape::lowered::MScaleAxis::X,
                )
            } else {
                (
                    (n_q_heads, bucket_m, 1),
                    (1024u32, 1u32, 1u32),
                    crate::tape::lowered::MScaleAxis::Y,
                )
            };
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
            let bindings = super::kernel_bindings::AttentionPrefillPagedBindingSet {
                output: super::ids::ArenaSlotIdx(*out_slot),
                q: super::ids::ArenaSlotIdx(*q_slot),
                kv_layer: super::ids::LayerId(*layer + layer_offset),
                // Steel spans reads pre-roped K from the scratch (slot 7), so
                // it does NOT bind cos_sin there; the non-spans path keeps the
                // in-kernel cos_sin binding.
                rope_on_read: if steel_spans { None } else { ror_bind },
                // Sliding prefill uses the simdgroup steel kernel (hd256),
                // never NAX (hd128 only). Spans → reads pre-roped K from the
                // shared scratch at slot 7 (RopeOnceSteel below).
                nax_roped_k_scratch: steel_spans,
            };
            let dispatch = DispatchShape {
                threadgroups: tg_shape,
                threads_per_threadgroup: threads_per_tg,
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    // Steel: one grid-Z layer per sequence so a BQ
                    // tile never straddles a sequence boundary (see
                    // the AttentionPrefillPaged arm).
                    seq_axis: if use_steel {
                        Some(crate::interpreter::metal::lowered::MScaleAxis::Z)
                    } else {
                        None
                    },
                    axis: m_scale_axis,
                    bucket_m: super::ids::BucketM(bucket_m),
                }),
            };
            if use_steel {
                let function = steel_symbol.expect("steel_symbol is Some when use_steel is true");
                let attn_cmd = LoweredCommand {
                    kernel: KernelId::AttentionPrefillSdpaPaged,
                    library: "attention_steel_paged",
                    function,
                    constants: constants.into_baked(),
                    dispatch,
                    bindings: bindings.into_baked(),
                    gemm_dims: None,
                };
                if steel_spans {
                    // Two commands: (1) RopeOnceSteel ropes the cache's K into
                    // the shared scratch using SLIDING geometry (HEAD_DIM 256,
                    // NUM_KV_HEADS, BLOCK_SIZE) + the SLIDING-class cos_sin
                    // (is_global: false — gemma4 uses a different rope theta for
                    // local vs global layers); (2) the steel attention reads
                    // pre-roped K from the scratch (slot 7). Mirrors the
                    // global AttentionPrefillPaged steel_spans path.
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
                            is_global: false,
                        }
                        .into_baked(),
                        gemm_dims: None,
                    };
                    return Ok(vec![rope_cmd, attn_cmd]);
                }
                attn_cmd
            } else {
                match p.metal_dtype {
                    crate::tape::lowered::MetalDtype::Bf16 => {
                        LoweredCommand::for_kernel::<AttentionSdpaPagedBf16>(
                            constants, bindings, dispatch,
                        )
                    }
                    _ => LoweredCommand::for_kernel::<AttentionSdpaPagedF16>(
                        constants, bindings, dispatch,
                    ),
                }
            }
        }

        // ── Plain residual add ─────────────────────────────────────
        I::Add(delta_slot, residual_slot) => LoweredCommand {
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
        I::ScalarMul(in_slot, out_slot, scale) => LoweredCommand {
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
        // NOTE: the lm_head narrow-path rewrite in `lower_pair` now
        // tolerates a trailing TanhSoftCap (`[AffineQmm, TanhSoftCap]`
        // slices get gather → qmv → narrow softcap → scatter); this
        // full-M arm remains the spec-decode fallback and the decode
        // (bucket_m == 1) path.
        I::TanhSoftCap(in_slot, out_slot) => {
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
        // `linear_bias` (Qwen2 4bit ships this). `is_affine` flag on
        // the Instruction picks which arm — the macro knows the
        // weight's StorageFormat at FUF construction time.
        I::MetalBiasAdd(in_slot, out_slot, layer, n, is_affine) => LoweredCommand {
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
                Binding::Weight {
                    kind: WeightBundleKind::LinearLayer,
                    which: if *is_affine {
                        WeightTensor::AffineLinearBias
                    } else {
                        WeightTensor::Bias
                    },
                    layer: super::ids::LayerId(*layer + layer_offset),
                    locator: WeightLocator {
                        bucket: tape_index,
                        op_idx: index as u32,
                        slot: 0,
                    },
                    binding_index: 1,
                },
                Binding::ArenaSlot {
                    slot: *out_slot,
                    binding_index: 2,
                },
            ]),
            gemm_dims: None,
        },

        // ── Metal MoE: SwitchGLU decomposition ─────────────────────
        //
        // `MetalFusedMoeImpl` / `MetalSharedFusedMoeImpl` (in
        // `scratchy-forward-compiler-macro/src/metal/moe.rs`) emit these
        // variants with full macro-baked shape. The arm emits the
        // 10-command Switch-GLU decomposition. Worker-side wiring for
        // `Binding::{Inline, MoeScratch}` lives in §3a; until it
        // lands the worker errors `Inline/MoeScratch not yet wired`
        // when these commands are executed. The lowering shape is
        // structurally complete here so the macro-side build stays
        // green and the worker side has a fixed contract to wire
        // against.
        I::MetalFusedMoe(
            in_slot,
            out_slot,
            layer,
            num_experts,
            top_k,
            moe_inter,
            hidden,
            group_size,
            bits,
        ) => {
            let cmds = lower_metal_moe(
                p,
                MetalMoeLowering {
                    in_slot: *in_slot,
                    out_slot: *out_slot,
                    layer: *layer + layer_offset,
                    num_experts: *num_experts,
                    top_k: *top_k,
                    moe_inter: *moe_inter,
                    hidden: *hidden,
                    group_size: *group_size,
                    // Mixtral is uniform-bit: same width for all projections.
                    gate_bits: *bits,
                    up_bits: *bits,
                    down_bits: *bits,
                    softmax_first: false,
                    norm_topk_prob: false,
                    shared_intermediate: 0,
                    tape_index,
                    op_idx: index as u32,
                    bucket_m,
                    is_shared: false,
                },
                moe_scratch_bytes,
            );
            return Ok(cmds);
        }
        I::MetalSharedFusedMoe(
            in_slot,
            out_slot,
            layer,
            num_experts,
            top_k,
            moe_inter,
            hidden,
            shared_intermediate,
            group_size,
            bits,
            norm_topk_prob,
        ) => {
            let (gate_bits, up_bits, down_bits) = scratchy_ir::unpack_moe_expert_bits(*bits);
            let cmds = lower_metal_moe(
                p,
                MetalMoeLowering {
                    in_slot: *in_slot,
                    out_slot: *out_slot,
                    layer: *layer + layer_offset,
                    num_experts: *num_experts,
                    top_k: *top_k,
                    moe_inter: *moe_inter,
                    hidden: *hidden,
                    group_size: *group_size,
                    // OptiQ mixed 4/8-bit: `bits` carries the three routed-expert
                    // projection widths packed by `repack_moe_expert_bits`.
                    gate_bits,
                    up_bits,
                    down_bits,
                    softmax_first: true,
                    norm_topk_prob: *norm_topk_prob,
                    shared_intermediate: *shared_intermediate,
                    tape_index,
                    op_idx: index as u32,
                    bucket_m,
                    is_shared: true,
                },
                moe_scratch_bytes,
            );
            return Ok(cmds);
        }
        // ── Gemma-4 fused MoE block (the `gemma_moe` op) ────────────
        I::GemmaMoe(
            router_in,
            expert_in,
            out_slot,
            layer,
            num_experts,
            top_k,
            moe_inter,
            hidden,
            group_size,
            bits,
        ) => {
            let cmds = lower_gemma_moe(
                p,
                GemmaMoeLowering {
                    router_in_slot: *router_in,
                    expert_in_slot: *expert_in,
                    out_slot: *out_slot,
                    layer: *layer + layer_offset,
                    num_experts: *num_experts,
                    top_k: *top_k,
                    moe_inter: *moe_inter,
                    hidden: *hidden,
                    group_size: *group_size,
                    bits: *bits,
                    tape_index,
                    op_idx: index as u32,
                    bucket_m,
                    is_nax: profile.is_some_and(|p| crate::targets::is_nax_capable(p.generation)),
                },
                moe_scratch_bytes,
            );
            return Ok(cmds);
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
        I::GatedDeltaNet(qkv_slot, z_slot, a_slot, b_slot, out_slot, layer) => {
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

            // The locator is identical for every weight in the bundle;
            // `which` picks the sub-tensor.
            let locator = WeightLocator {
                bucket: tape_index,
                op_idx: index as u32,
                slot: 0,
            };
            let weight = |which: WeightTensor, binding_index: u8| Binding::Weight {
                kind: WeightBundleKind::GatedDeltaNet,
                which,
                layer: layer_id,
                locator,
                binding_index,
            };
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
        I::MeanSubRmsNorm(in_slot, out_slot, layer) => {
            let layer_id = super::ids::LayerId(*layer + layer_offset);
            let locator = WeightLocator {
                bucket: tape_index,
                op_idx: index as u32,
                slot: 0,
            };
            LoweredCommand {
                kernel: KernelId::VisionLayerNorm,
                library: "vision_layernorm",
                function: vision_layernorm_static_name(
                    p.metal_dtype,
                    norm_gain_dtype(p, WeightBundleKind::RmsNorm),
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
                    Binding::Weight {
                        kind: WeightBundleKind::RmsNorm,
                        which: WeightTensor::Weight,
                        layer: layer_id,
                        locator,
                        binding_index: 2,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::RmsNorm,
                        which: WeightTensor::Weight,
                        layer: layer_id,
                        locator,
                        binding_index: 3,
                    },
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
        // one `WeightBundleKind::LayerNorm` bundle at the same locator,
        // mirroring the multi-tensor `GatedDeltaNet` binding pattern.
        I::MeanSubRmsNormBiasAdd(in_slot, out_slot, layer) => {
            let layer_id = super::ids::LayerId(*layer + layer_offset);
            let locator = WeightLocator {
                bucket: tape_index,
                op_idx: index as u32,
                slot: 0,
            };
            LoweredCommand {
                kernel: KernelId::VisionLayerNorm,
                library: "vision_layernorm",
                function: vision_layernorm_static_name(
                    p.metal_dtype,
                    norm_gain_dtype(p, WeightBundleKind::LayerNorm),
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
                    Binding::Weight {
                        kind: WeightBundleKind::LayerNorm,
                        which: WeightTensor::Weight,
                        layer: layer_id,
                        locator,
                        binding_index: 2,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LayerNorm,
                        which: WeightTensor::Bias,
                        layer: layer_id,
                        locator,
                        binding_index: 3,
                    },
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
        I::Gelu(in_slot, out_slot) => {
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
        I::GeluErf(in_slot, out_slot) => {
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
        I::QuickGelu(in_slot, out_slot) => {
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
        I::LoadPixels(out_slot) => {
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
        I::LoadPosEmbeds(out_slot) => {
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
        I::VisionRope(q_slot, k_slot, q_out_slot, k_out_slot) => {
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
        // VISION_ATTN_SCALE). Qwen3.5-VL emits cu_seqlens_kind == 0, so
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
        I::EncoderAttention(q_slot, k_slot, v_slot, out_slot) => {
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

        I::VarlenAttention(q_slot, k_slot, v_slot, out_slot, cu_seqlens_kind) => {
            // kind 0: single per-image segmentation in `cu_seqlens_q`
            // (Qwen3.5-VL / LocateAnything / Qwen2-VL). kinds 1/2:
            // Qwen2.5-VL windowed attention — the full-attention layers
            // read per-image boundaries, the window layers per-window
            // boundaries; both arrive as dedicated runtime externs. The
            // kernel is identical across kinds (its segment-search walks
            // whatever cu_seqlens it's bound to; `VA_NUM_SEGS` stays the
            // bucket_m upper bound and zero-padded tail entries form
            // empty ranges).
            let cu_binding = match cu_seqlens_kind {
                0 => RuntimeBindingKind::CuSeqlensQ,
                1 => RuntimeBindingKind::VisionCuSeqlensFull,
                2 => RuntimeBindingKind::VisionCuSeqlensWindow,
                _ => {
                    return Err(LoweringError::UnsupportedVariant {
                        index,
                        variant_type: "VarlenAttention(cu_seqlens_kind > 2)",
                    });
                }
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
        // kind-0 gather so window permutation moves whole merge
        // groups, and runs kind 1 on the `[L/S², d_model]` merger
        // output. Mirrors the cuda `kernels::embedding_gather` row
        // semantics; indices come from the `vision_window_index` /
        // `vision_reverse_indices` runtime externs (kind 0 / 1).
        I::EmbeddingGather(in_slot, out_slot, indices_kind) => {
            let idx_binding = match indices_kind {
                0 => RuntimeBindingKind::VisionWindowIndex,
                1 => RuntimeBindingKind::VisionReverseIndices,
                _ => {
                    return Err(LoweringError::UnsupportedVariant {
                        index,
                        variant_type: "EmbeddingGather(indices_kind > 1)",
                    });
                }
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

        // ── 2-D average pool (Gemma3-MM SigLIP→text projector) ─────
        //
        // Collapses the [ph², e] post-encoder grid by a k×k cell to
        // [(ph/k)², e]. ph / k are compile-time (`p.vision_patch_grid_side`
        // / `p.vision_pool_kernel`); e is the vision residual width
        // (`p.hidden_size` on the vision Weights). Fixed per-image
        // reduction (cuda eval asserts input rows == ph²), so the grid
        // is the static output-element count — no m_scaling. Faithful to
        // the cuda `avg_pool_2d_kernel`: one thread per output element.
        I::AvgPool2d(in_slot, out_slot) => {
            let ph = p.vision_patch_grid_side;
            let k = p.vision_pool_kernel;
            let e = p.hidden_size as u32;
            assert!(
                ph > 0 && k > 0 && ph.is_multiple_of(k) && e > 0,
                "AvgPool2d: VISION_PATCH_GRID_SIDE={ph} VISION_POOL_KERNEL={k} \
                 HIDDEN_SIZE={e} — ph must be a positive multiple of k, e > 0"
            );
            let ph_out = ph / k;
            let total = ph_out * ph_out * e;
            LoweredCommand {
                kernel: KernelId::AvgPool2d,
                library: "avg_pool_2d",
                function: avg_pool_2d_static_name(p.metal_dtype),
                constants: &[],
                dispatch: DispatchShape::dispatch_1d(total, THREADS_PER_GROUP),
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
                        value: ph,
                    },
                    Binding::Inline {
                        binding_index: 3,
                        value: k,
                    },
                    Binding::Inline {
                        binding_index: 4,
                        value: e,
                    },
                ]),
                gemm_dims: None,
            }
        }

        // ── Metadata-only: no Metal dispatch ───────────────────────
        I::Reshape(_, _, _, _, _, _) | I::Alias(_, _) | I::Free(_) => {
            // These rebind / drop slots in the dispatcher's logical
            // view but don't touch device memory. Subsequent commands
            // in the lowered tape see the new logical shape via the
            // worker's slot tracker (resolved at worker init).
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
        I::SpliceMmEmbeds(slot) => {
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

        // ── Loop already handled by the caller ─────────────────────
        I::Loop(_, _, _) => unreachable!("Loop is handled in `lower()` directly"),

        // ── Opcodes with NO metal lowering ─────────────────────────
        //
        // Enumerated, never a `_` catch-all. With a
        // catch-all, adding an `Instruction` variant and forgetting
        // its metal arm still COMPILES, and the gap only surfaces
        // when some arch happens to emit the op — which is exactly
        // how both modernbert gaps were found, one
        // serve-read-the-log cycle each. Enumerated, a new variant is
        // E0004 right here, at the macro's own compile, and adding an
        // op is what §2 claims it is: a registry row plus one arm per
        // target.
        //
        // Each group below is a family metal cannot run, not a family
        // metal has not gotten to. The refusal is still a refusal —
        // it is the closed set of them that is the point.

        // CUTLASS: the cuda GEMM/epilogue library. Selected by the
        // cuda solver, which does not run for metal.
        I::CutlassFusedAddRmsNormGemm(..)
        | I::CutlassFusedGateUpGeluMul(..)
        | I::CutlassFusedGateUpSiluMul(..)
        | I::CutlassFusedGemmBias(..)
        | I::CutlassFusedMeanSubRmsNormGemm(..)
        | I::CutlassFusedQkvRopeCache(..)
        | I::CutlassFusedQkvRopePrefill(..)
        | I::CutlassFusedRmsNormGemm(..)
        | I::CutlassGemm(..)
        | I::CutlassGemmAdd(..)
        | I::CutlassGemmSplitK(..)
        | I::CutlassGemv(..) => return no_metal_kernel(index, "CUTLASS"),

        // Quantized cuda backends: Marlin, bitsandbytes-4bit, ggml
        // and FP8 each carry their own packing and their own kernels.
        // Metal's quantized path is the affine/NVFP4 family above.
        I::Bnb4FusedGateUpGeluMul(..)
        | I::Bnb4FusedGateUpSiluMul(..)
        | I::Bnb4FusedQkvRopeCache(..)
        | I::Bnb4FusedQkvRopePrefill(..)
        | I::Bnb4Gemm(..)
        | I::Fp8FusedGateUpGeluMul(..)
        | I::Fp8FusedGateUpSiluMul(..)
        | I::Fp8FusedGemmBias(..)
        | I::Fp8FusedQkvRopeCache(..)
        | I::Fp8FusedQkvRopePrefill(..)
        | I::Fp8Gemm(..)
        | I::GgmlFusedGateUpGeluMul(..)
        | I::GgmlFusedGateUpSiluMul(..)
        | I::GgmlFusedQkvRopeCache(..)
        | I::GgmlFusedQkvRopePrefill(..)
        | I::GgmlGemm(..)
        | I::MarlinFusedGateUpGeluMul(..)
        | I::MarlinFusedGateUpSiluMul(..)
        | I::MarlinFusedQkvRopeCache(..)
        | I::MarlinFusedQkvRopePrefill(..)
        | I::MarlinGemm(..) => return no_metal_kernel(index, "cuda quantized backend"),

        // cuda attention backends (FlashInfer, MLA's split/absorb
        // form, the contiguous sliding-window prefill). Metal serves
        // attention through its own paged/SDPA kernels above.
        I::FlashInferAttentionDecode(..)
        | I::FlashInferAttentionPrefill(..)
        | I::MlaAttention(..)
        | I::MlaSplit(..)
        | I::SlidingAttentionPrefillContiguous(..) => {
            return no_metal_kernel(index, "cuda attention backend");
        }

        // cuda MoE dispatch. Metal serves MoE through its OWN
        // opcodes (`I::MetalFusedMoe` / `I::MetalSharedFusedMoe`,
        // lowered above); these are the cuda-shaped peers, including
        // DeepSeek's three flavors.
        I::FusedMoe(..)
        | I::SharedFusedMoe(..)
        | I::DeepSeekMoe(..)
        | I::DeepSeekMoeFp8Block(..)
        | I::DeepSeekMoeGgml(..) => return no_metal_kernel(index, "cuda MoE"),

        // cuBLAS-shaped fusions: the epilogue lives in the cuda
        // library call, not in a kernel metal could compile.
        I::FusedCublasGemmAdd(..)
        | I::FusedGemmBias(..)
        | I::FusedQkvQkNormRopeCache(..)
        | I::FusedQkvRopePrefill(..) => return no_metal_kernel(index, "cuBLAS fusion"),

        // LLaVA-1.5's `vision_feature_select_strategy = "default"`
        // CLS strip. No metal arch serves LLaVA; this is a genuine
        // gap, not a cuda-only shape, and it is named as one.
        I::StripCls(..) => return no_metal_kernel(index, "LLaVA StripCls (no metal arch uses it)"),
        // NOT listed, deliberately: `AllGather`/`AllReduce` exist
        // only under `scratchy-ir/nccl` and `FlashAttention3Decode`
        // only under `cfg(fa3_built)` — both reachable exclusively
        // from a cuda build, and cuda and metal are mutually
        // exclusive. Arms for them would not compile here. If that
        // ever changes, this match fails with E0004 naming them,
        // which is the correct way to find out.
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
fn norm_gain_dtype(p: &MetalModelConsts, bundle: WeightBundleKind) -> ScaleDtype {
    match bundle {
        WeightBundleKind::RmsNorm => scale_dtype_for(p),
        WeightBundleKind::LayerNorm => match p.metal_dtype {
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
    tape_index: u32,
    index: usize,
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
            Binding::Weight {
                kind: WeightBundleKind::LinearLayer,
                which: WeightTensor::Weight,
                layer: super::ids::LayerId(layer + layer_offset),
                locator: WeightLocator {
                    bucket: tape_index,
                    op_idx: index as u32,
                    slot: 0,
                },
                binding_index: 2,
            },
        ]),
        gemm_dims: None,
    }
}

fn avg_pool_2d_static_name(dtype: MetalDtype) -> &'static str {
    match dtype {
        MetalDtype::F16 => "avg_pool_2d_f16",
        MetalDtype::Bf16 => "avg_pool_2d_bf16",
        MetalDtype::Int4 => {
            panic!("avg_pool_2d: Int4 unsupported (activations are bf16/f16)")
        }
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
            "metal lowering: Instruction::AffineQmm requires p.metal_dtype \
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

/// Bindings shared by every `Instruction::AffineQmm` lowering's qmv
/// and qmm_t Standard kernels — both bind buffers 0..4 in the same
/// order: (packed weight, scales, biases, x in, y out). Worker
/// resolves the `Affine*` `WeightTensor` arms via
/// `LinearLayer::AffineQuant` (`worker.rs:1414`).
/// MSL symbol for the generic NVFP4 decode-matvec kernel. Scales are
/// always folded to F16 at load (`Nvfp4Linear::load`) and group_size is
/// always 16, so only the activation dtype varies. Must match the
/// instantiations in `quantized_qmv.metal`.
fn nvfp4_qmv_name(act: DequantDtype) -> &'static str {
    match act {
        DequantDtype::Bf16 => "nvfp4_qmv_bf16_s_f16_gs_16_b_4_batch_0",
        DequantDtype::F16 => "nvfp4_qmv_f16_s_f16_gs_16_b_4_batch_0",
    }
}

/// MSL symbol for the standard NVFP4 prefill matmul kernel. Must match
/// the instantiations in `quantized_qmm.metal`.
fn nvfp4_qmm_t_name(act: DequantDtype, aligned_n: bool) -> &'static str {
    match (act, aligned_n) {
        (DequantDtype::Bf16, true) => "nvfp4_qmm_t_bf16_s_f16_gs_16_b_4_alN_true_batch_0",
        (DequantDtype::Bf16, false) => "nvfp4_qmm_t_bf16_s_f16_gs_16_b_4_alN_false_batch_0",
        (DequantDtype::F16, true) => "nvfp4_qmm_t_f16_s_f16_gs_16_b_4_alN_true_batch_0",
        (DequantDtype::F16, false) => "nvfp4_qmm_t_f16_s_f16_gs_16_b_4_alN_false_batch_0",
    }
}

/// MSL symbol for the NAX (M4+) NVFP4 prefill matmul kernel. Must match
/// the instantiations in `quantized_qmm_nax.metal`. `aligned_n` keys off
/// `N % 64 == 0` (NAX tile BN=64), vs `N % 32` for the standard kernel.
fn nvfp4_qmm_t_nax_name(act: DequantDtype, aligned_n: bool) -> &'static str {
    match (act, aligned_n) {
        (DequantDtype::Bf16, true) => "nvfp4_qmm_t_nax_bf16_s_f16_gs_16_b_4_alN_true_batch_0",
        (DequantDtype::Bf16, false) => "nvfp4_qmm_t_nax_bf16_s_f16_gs_16_b_4_alN_false_batch_0",
        (DequantDtype::F16, true) => "nvfp4_qmm_t_nax_f16_s_f16_gs_16_b_4_alN_true_batch_0",
        (DequantDtype::F16, false) => "nvfp4_qmm_t_nax_f16_s_f16_gs_16_b_4_alN_false_batch_0",
    }
}

/// Buffer bindings for an NVFP4 qmv / qmm_t dispatch. Four buffers —
/// `weight[0]`, folded `scales[1]`, activation `x[2]`, output `y[3]`.
/// NVFP4 has no per-group bias, so (unlike `affine_qmm_bindings`) there
/// is no biases buffer and the activation/output indices shift down by
/// one. The nvfp4 shaders in `quantized_qmv.metal` / `quantized_qmm.metal`
/// declare exactly this layout.
fn nvfp4_qmm_bindings(
    in_slot: u32,
    out_slot: u32,
    layer: super::ids::LayerId,
    locator: crate::tape::lowered::WeightLocator,
) -> Vec<Binding> {
    vec![
        Binding::Weight {
            kind: WeightBundleKind::LinearLayer,
            which: WeightTensor::Weight,
            layer,
            locator,
            binding_index: 0,
        },
        Binding::Weight {
            kind: WeightBundleKind::LinearLayer,
            which: WeightTensor::Nvfp4Scales,
            layer,
            locator,
            binding_index: 1,
        },
        // Dummy `biases` at index 2 = the same scales buffer. NVFP4 has
        // no per-group bias, but the 5-buffer layout MUST match affine's
        // (w,scales,biases,x,y): MTL4 mis-dispatches the 4-buffer
        // (x@2,y@3) layout — the kernel reads garbage for x/y under the
        // MTL4 argument table even though the identical math is correct
        // under MTL3 (unit test) and affine (5-buffer) is coherent under
        // MTL4. Binding x@3/y@4 (matching affine) fixes it.
        Binding::Weight {
            kind: WeightBundleKind::LinearLayer,
            which: WeightTensor::Nvfp4Scales,
            layer,
            locator,
            binding_index: 2,
        },
        Binding::ArenaSlot {
            slot: in_slot,
            binding_index: 3,
        },
        Binding::ArenaSlot {
            slot: out_slot,
            binding_index: 4,
        },
    ]
}

fn affine_qmm_bindings(
    in_slot: u32,
    out_slot: u32,
    layer: super::ids::LayerId,
    locator: crate::tape::lowered::WeightLocator,
) -> Vec<Binding> {
    vec![
        Binding::Weight {
            kind: WeightBundleKind::LinearLayer,
            which: WeightTensor::Weight,
            layer,
            locator,
            binding_index: 0,
        },
        Binding::Weight {
            kind: WeightBundleKind::LinearLayer,
            which: WeightTensor::AffineScales,
            layer,
            locator,
            binding_index: 1,
        },
        Binding::Weight {
            kind: WeightBundleKind::LinearLayer,
            which: WeightTensor::AffineBiases,
            layer,
            locator,
            binding_index: 2,
        },
        Binding::ArenaSlot {
            slot: in_slot,
            binding_index: 3,
        },
        Binding::ArenaSlot {
            slot: out_slot,
            binding_index: 4,
        },
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
    locator: crate::tape::lowered::WeightLocator,
) -> Vec<Binding> {
    vec![
        Binding::Weight {
            kind: WeightBundleKind::LinearLayer,
            which: WeightTensor::Weight,
            layer,
            locator,
            binding_index: 0,
        },
        Binding::Weight {
            kind: WeightBundleKind::LinearLayer,
            which: WeightTensor::AffineScales,
            layer,
            locator,
            binding_index: 1,
        },
        Binding::Weight {
            kind: WeightBundleKind::LinearLayer,
            which: WeightTensor::AffineBiases,
            layer,
            locator,
            binding_index: 2,
        },
        Binding::ArenaSlot {
            slot: in_slot,
            binding_index: 3,
        },
        Binding::Scratch { binding_index: 4 },
    ]
}

/// Byte width of one element in the activation dtype the worker
/// allocates the SplitK scratch buffer against. F16 / Bf16 = 2 bytes.
fn elem_size_bytes(dtype: DequantDtype) -> u32 {
    match dtype {
        DequantDtype::F16 | DequantDtype::Bf16 => 2,
    }
}

/// Format the kernel symbol name for an `Instruction::RmsNorm`
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

/// As [`rmsnorm_kernel_static_name`] for `Instruction::FusedAddRmsNorm`.
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

/// `Instruction::NormAddScalarMul` symbol — the norm-THEN-add mirror
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

/// `Instruction::RopeAppendNormed` symbol — same dtype enumeration as
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
// Metal MoE lowering (`I::MetalFusedMoe` / `I::MetalSharedFusedMoe`)
// ────────────────────────────────────────────────────────────────────

/// Tuple of inputs to [`lower_metal_moe`]. Keeps the two arms in
/// `lower_one` from having to pass 14 positional args each.
struct MetalMoeLowering {
    in_slot: u32,
    out_slot: u32,
    layer: u32,
    num_experts: u32,
    top_k: u32,
    moe_inter: u32,
    hidden: u32,
    group_size: u32,
    /// On-disk routed-expert bit-widths (4 or 8) for gate / up / down. MLX
    /// mixed/dynamic quant (OptiQ) ships each projection at its OWN width and
    /// they differ per layer (e.g. layer 1: gate=4, up=8, down=8), so each
    /// gather-qmv symbol must pick the matching `_b_{bits}` kernel. Uniform
    /// checkpoints (and Mixtral) set all three equal.
    gate_bits: u32,
    up_bits: u32,
    down_bits: u32,
    /// `false` = Mixtral order (`topk → softmax(scores)`).
    /// `true`  = Qwen-MoE order (`softmax(probs) → topk → take_along_axis(probs)`).
    softmax_first: bool,
    /// Qwen3-MoE `norm_topk_prob=True` — renorm gathered top-k scores
    /// by their sum so they sum to 1 again.
    /// `NotYetWired`: the renorm kernel itself doesn't exist; the lowering
    /// arm currently asserts this is false. Path will land alongside §3a.
    norm_topk_prob: bool,
    /// Shared-expert intermediate size; `0` means no shared expert.
    /// Non-zero requires emitting the shared-expert tail (3× AffineQmm
    /// plus sigmoid-gate fusion), which depends on a fused-add-sigmoid-
    /// gate-mul kernel that isn't ported yet. The lowering arm
    /// asserts this is `0` for now and panics otherwise. Both Mixtral
    /// and Qwen3-MoE-30B-A3B-Instruct ship `shared_intermediate=0` —
    /// the shared-expert path is only needed for Qwen1.5-MoE-A2.7B
    /// (and falls under §3 follow-up work).
    shared_intermediate: u32,
    /// Carries through to `WeightLocator` on the emitted commands.
    tape_index: u32,
    op_idx: u32,
    bucket_m: u32,
    /// `false` → caller is `I::MetalFusedMoe` (Mixtral); emitted
    /// `Binding::Weight`s carry `WeightBundleKind::FusedMoe` so the
    /// worker resolves through `WeightAccessors::fused_moe_at`.
    /// `true` → caller is `I::MetalSharedFusedMoe` (Qwen2/3-MoE);
    /// flip to `SharedFusedMoe` so the worker resolves through
    /// `shared_fused_moe_at`. The macro emits exactly one of the two
    /// accessor methods per arch — picking the wrong bundle here
    /// panics with `fused_moe_at not implemented` (or vice versa).
    is_shared: bool,
}

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

/// Long-form dtype infix used by softmax / take_along_axis /
/// moe_weighted_sum shader symbols (`float16` / `bfloat16`). Distinct
/// from `dequant_dtype_for(p).symbol_infix()` which yields the
/// compact form (`f16` / `bf16`) used by the qmv/qmm_t/affine_gather
/// kernels.
fn long_dtype_infix(p: &MetalModelConsts) -> &'static str {
    match p.metal_dtype {
        MetalDtype::F16 => "float16",
        MetalDtype::Bf16 => "bfloat16",
        MetalDtype::Int4 => panic!("MoE lowering: p.metal_dtype must be F16 or Bf16, got Int4"),
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

/// Emit the multi-`LoweredCommand` decomposition for a single
/// `I::MetalFusedMoe` / `I::MetalSharedFusedMoe` instance.
fn lower_metal_moe(
    mc: &MetalModelConsts,
    p: MetalMoeLowering,
    moe_scratch_bytes: &mut u32,
) -> Vec<LoweredCommand> {
    let dtype = dequant_dtype_for(mc);
    let scale_dtype = scale_dtype_for(mc);
    let elem = elem_size_bytes(dtype);
    // Shared-expert tail + norm_topk_prob require kernels that aren't
    // ported yet; assert this session's scope. SharedFusedMoe
    // variants with shared_intermediate==0 and norm_topk_prob=false
    // (modern Qwen3-MoE-30B-A3B-Instruct) lower through fine; both
    // additions land in §3-followup.
    assert!(
        p.shared_intermediate == 0,
        "lower_metal_moe: shared_intermediate={} > 0 requires the shared-expert tail \
         kernels (sigmoid-gate fusion) which aren't ported yet. Modern \
         Qwen3-MoE-30B-A3B-Instruct ships shared_intermediate=0 and works through \
         this arm; Qwen1.5-MoE-A2.7B-Chat (shared_intermediate=5632) needs the \
         §3 follow-up tail.",
        p.shared_intermediate
    );
    // norm_topk_prob handled inline via the topk_renorm kernel (see
    // softmax.metal). Lowering inserts a renorm step between
    // TakeAlongAxis (gathered top-k scores) and AffineGatherQmv
    // (expert dispatch) so the per-row weights sum to 1 — matches
    // `weights / weights.sum(-1, keepdims=True)` in MLX/PyTorch
    // Qwen3MoE source.

    let layout = MoeScratchLayout::compute(
        p.bucket_m,
        p.num_experts,
        p.top_k,
        p.moe_inter,
        p.hidden,
        elem,
        false,
    );
    *moe_scratch_bytes = (*moe_scratch_bytes).max(layout.total);

    let layer_id = super::ids::LayerId(p.layer);
    let locator0 = crate::tape::lowered::WeightLocator {
        bucket: p.tape_index,
        op_idx: p.op_idx,
        slot: 0,
    };
    // Bundle kind picks which `WeightAccessors` method the worker
    // resolves through. The macro emits exactly one of
    // `fused_moe_at` (MetalFusedMoeImpl, FusedMoELayer accessor
    // type) or `shared_fused_moe_at` (MetalSharedFusedMoeImpl,
    // SharedFusedMoELayer accessor type) per arch. Using the wrong
    // bundle panics with `..._at not implemented for this arch`.
    let bundle_kind = if p.is_shared {
        WeightBundleKind::SharedFusedMoe
    } else {
        WeightBundleKind::FusedMoe
    };

    let mut cmds: Vec<LoweredCommand> = Vec::with_capacity(10);

    // ── Step 1: router_logits = Gemm(x, W_router_gate) ─────────────
    //
    // Dense BF16/F16 GEMM. Output goes to moe_scratch[router_logits]
    // — the worker resolves `Binding::MoeScratch` against
    // `moe_scratch + byte_offset` (§3a).
    //
    // Reuses `KernelId::Gemm` (opaque to the pipeline cache; routed
    // through the worker's MPS / hand-rolled bf16 path). Worker
    // changes needed for §3a: `resolve_gemm_buffers` accepts
    // `Binding::MoeScratch` outputs.
    {
        let tg_x = p.bucket_m.div_ceil(GEMM_TILE_M);
        let tg_y = p.num_experts.div_ceil(GEMM_TILE_N);
        cmds.push(LoweredCommand {
            kernel: KernelId::Gemm,
            library: "",
            function: "",
            constants: &[],
            dispatch: DispatchShape {
                threadgroups: (tg_x, tg_y, 1),
                threads_per_threadgroup: (GEMM_TILE_M, GEMM_TILE_N, 1),
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::X,
                    bucket_m: super::ids::BucketM(p.bucket_m),
                }),
            },
            bindings: baked(vec![
                Binding::MoeScratch {
                    binding_index: 0,
                    byte_offset: layout.router_logits,
                },
                Binding::ArenaSlot {
                    slot: p.in_slot,
                    binding_index: 1,
                },
                Binding::Weight {
                    kind: bundle_kind,
                    which: WeightTensor::MoeRouterGate,
                    layer: layer_id,
                    locator: locator0,
                    binding_index: 2,
                },
            ]),
            gemm_dims: Some(GemmDims {
                m: p.bucket_m,
                n: p.num_experts,
                k: p.hidden,
            }),
        });
    }

    // ── Step 2a (Qwen, softmax_first): router_probs = Softmax(logits) ─
    //
    // Writes back into the same router_logits region (in-place is OK
    // per MLX). Mixtral does softmax AFTER topk on the gathered
    // scores; see Step 5b below.
    if p.softmax_first {
        cmds.push(make_moe_command(
            KernelId::Softmax,
            "softmax",
            softmax_precise_symbol(mc),
            Vec::new(),
            DispatchShape {
                threadgroups: (p.bucket_m, 1, 1),
                threads_per_threadgroup: (256, 1, 1),
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::X,
                    bucket_m: super::ids::BucketM(p.bucket_m),
                }),
            },
            vec![
                Binding::MoeScratch {
                    binding_index: 0,
                    byte_offset: layout.router_logits,
                },
                Binding::MoeScratch {
                    binding_index: 1,
                    byte_offset: layout.router_logits,
                },
                Binding::Inline {
                    binding_index: 2,
                    value: p.num_experts,
                },
            ],
        ));
    }

    // ── Step 3: sorted_full_inds = ArgPartitionTopK(probs|logits) ──
    //
    // Full ascending sort over the per-row num_experts axis. Output
    // is `[M, num_experts]` u32 with NaN entries (none in practice
    // for the small E we target) trailing. SliceTrailingColsU32
    // pulls the top-k window next.
    {
        // bn*tn=128 covers Mixtral E=8 / Qwen3-MoE E=128; Qwen3.5-MoE
        // E=256 takes the bn=64 (N_PER_BLOCK=256) instantiation.
        let bn: u32 = if p.num_experts > 128 { 64 } else { 32 };
        cmds.push(make_moe_command(
            KernelId::ArgPartitionTopK,
            "argpartition",
            argpartition_symbol(mc, p.num_experts),
            Vec::new(),
            DispatchShape {
                threadgroups: (1, p.bucket_m, 1),
                threads_per_threadgroup: (bn, 1, 1),
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::Y,
                    bucket_m: super::ids::BucketM(p.bucket_m),
                }),
            },
            vec![
                Binding::MoeScratch {
                    binding_index: 0,
                    byte_offset: layout.router_logits,
                },
                Binding::MoeScratch {
                    binding_index: 1,
                    byte_offset: layout.sorted_full,
                },
                Binding::Inline {
                    binding_index: 2,
                    value: p.num_experts,
                },
                Binding::Inline {
                    binding_index: 3,
                    value: 1,
                },
                Binding::Inline {
                    binding_index: 4,
                    value: 1,
                },
                Binding::Inline {
                    binding_index: 5,
                    value: p.num_experts,
                },
                Binding::Inline {
                    binding_index: 6,
                    value: p.num_experts,
                },
            ],
        ));
    }

    // ── Step 4: topk_inds = SliceTrailingColsU32(sorted_full_inds) ─
    //
    // `slice_trailing_cols.rs` uses `dispatchThreads`; convert to
    // threadgroup form: tg_x = ceil(top_k / min(32, top_k)).
    {
        let tg_x_threads = p.top_k.min(32);
        let tg_x_count = p.top_k.div_ceil(tg_x_threads);
        cmds.push(make_moe_command(
            KernelId::SliceTrailingColsU32,
            "slice_trailing_cols",
            "slice_trailing_cols_u32",
            Vec::new(),
            DispatchShape {
                threadgroups: (tg_x_count, p.bucket_m, 1),
                threads_per_threadgroup: (tg_x_threads, 1, 1),
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::Y,
                    bucket_m: super::ids::BucketM(p.bucket_m),
                }),
            },
            vec![
                Binding::MoeScratch {
                    binding_index: 0,
                    byte_offset: layout.sorted_full,
                },
                Binding::MoeScratch {
                    binding_index: 1,
                    byte_offset: layout.topk_inds,
                },
                Binding::Inline {
                    binding_index: 2,
                    value: p.num_experts,
                },
                Binding::Inline {
                    binding_index: 3,
                    value: p.top_k,
                },
            ],
        ));
    }

    // ── Step 5a: topk_scores = TakeAlongAxis(scores_src, topk_inds) ─
    //
    // Qwen: `scores_src = router_probs` (after Step 2a softmax).
    // Mixtral: `scores_src = router_logits` (raw, softmax happens
    //          AFTER this step on the gathered scores).
    //
    // Both cases read from `layout.router_logits` since the Qwen
    // softmax wrote back in-place there.
    {
        let tg_x_threads = p.top_k.min(32);
        let tg_x_count = p.top_k.div_ceil(tg_x_threads);
        cmds.push(make_moe_command(
            KernelId::TakeAlongAxis,
            "take_along_axis",
            take_along_axis_symbol(mc),
            Vec::new(),
            DispatchShape {
                threadgroups: (tg_x_count, p.bucket_m, 1),
                threads_per_threadgroup: (tg_x_threads, 1, 1),
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::Y,
                    bucket_m: super::ids::BucketM(p.bucket_m),
                }),
            },
            vec![
                Binding::MoeScratch {
                    binding_index: 0,
                    byte_offset: layout.router_logits,
                },
                Binding::MoeScratch {
                    binding_index: 1,
                    byte_offset: layout.topk_inds,
                },
                Binding::MoeScratch {
                    binding_index: 2,
                    byte_offset: layout.topk_scores,
                },
                Binding::Inline {
                    binding_index: 3,
                    value: p.num_experts,
                },
                Binding::Inline {
                    binding_index: 4,
                    value: p.top_k,
                },
            ],
        ));
    }

    // ── Step 5b (Mixtral, !softmax_first): softmax(topk_scores) ───
    //
    // In-place row-softmax over the gathered scores (axis_size = top_k).
    // 256-thread blocks handle top_k ≤ 1024 (we're at 2..8).
    if !p.softmax_first {
        cmds.push(make_moe_command(
            KernelId::Softmax,
            "softmax",
            softmax_precise_symbol(mc),
            Vec::new(),
            DispatchShape {
                threadgroups: (p.bucket_m, 1, 1),
                threads_per_threadgroup: (256, 1, 1),
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::X,
                    bucket_m: super::ids::BucketM(p.bucket_m),
                }),
            },
            vec![
                Binding::MoeScratch {
                    binding_index: 0,
                    byte_offset: layout.topk_scores,
                },
                Binding::MoeScratch {
                    binding_index: 1,
                    byte_offset: layout.topk_scores,
                },
                Binding::Inline {
                    binding_index: 2,
                    value: p.top_k,
                },
            ],
        ));
    }

    // ── Step 5c (Qwen3-MoE, norm_topk_prob): topk_scores /= sum ───
    //
    // In-place row-renorm of the gathered top-k scores so they sum
    // to 1. Matches `weights / weights.sum(-1, keepdims=True)` from
    // `qwen3_moe.py` when `norm_topk_prob=True`. Same dispatch shape
    // as the !softmax_first softmax above (256 threads × bucket_m
    // threadgroups). Symbol `topk_renorm_{float16,bfloat16}` —
    // instantiated in `shaders/softmax.metal`.
    if p.norm_topk_prob {
        let renorm_symbol: &'static str = match mc.metal_dtype {
            MetalDtype::F16 => "topk_renorm_float16",
            MetalDtype::Bf16 => "topk_renorm_bfloat16",
            MetalDtype::Int4 => panic!("topk_renorm: int4 unreachable"),
        };
        cmds.push(make_moe_command(
            KernelId::Softmax, // pipeline cache routing only — library/function disambiguate
            "softmax",
            renorm_symbol,
            Vec::new(),
            DispatchShape {
                threadgroups: (p.bucket_m, 1, 1),
                threads_per_threadgroup: (256, 1, 1),
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::X,
                    bucket_m: super::ids::BucketM(p.bucket_m),
                }),
            },
            vec![
                Binding::MoeScratch {
                    binding_index: 0,
                    byte_offset: layout.topk_scores,
                },
                Binding::MoeScratch {
                    binding_index: 1,
                    byte_offset: layout.topk_scores,
                },
                Binding::Inline {
                    binding_index: 2,
                    value: p.top_k,
                },
            ],
        ));
    }

    // ── Steps 6-8: gate_out / up_out / down_out via affine_gather_qmv
    //
    // Common dispatch shape: `(1, n_out/8, M*top_k)` threadgroups ×
    // `(32, 2, 1)` threads. K and N specialize via function_constant
    // 0 / 1. top_k Inline at buffer slot 6 for the kernel's pointer
    // arithmetic.
    let gather_qmv = |which_w,
                      which_s,
                      which_b,
                      x_off,
                      y_off,
                      n_out: u32,
                      k_in: u32,
                      bits: u32|
     -> LoweredCommand {
        let symbol = affine_gather_qmv_symbol(n_out, k_in, dtype, scale_dtype, p.group_size, bits);
        let bn: u32 = 8;
        let kernel = if symbol.contains("_fast_") {
            KernelId::AffineGatherQmvFast
        } else {
            KernelId::AffineGatherQmv
        };
        make_moe_command(
            kernel,
            "quantized_qmv",
            symbol,
            vec![
                ConstantValue::int(0, k_in as i32),
                ConstantValue::int(1, n_out as i32),
            ],
            DispatchShape {
                threadgroups: (1, n_out.div_ceil(bn), p.bucket_m * p.top_k),
                threads_per_threadgroup: (32, 2, 1),
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::Z,
                    bucket_m: super::ids::BucketM(p.bucket_m),
                }),
            },
            vec![
                Binding::Weight {
                    kind: bundle_kind,
                    which: which_w,
                    layer: layer_id,
                    locator: locator0,
                    binding_index: 0,
                },
                Binding::Weight {
                    kind: bundle_kind,
                    which: which_s,
                    layer: layer_id,
                    locator: locator0,
                    binding_index: 1,
                },
                Binding::Weight {
                    kind: bundle_kind,
                    which: which_b,
                    layer: layer_id,
                    locator: locator0,
                    binding_index: 2,
                },
                x_off, // buffer 3 = x
                Binding::MoeScratch {
                    binding_index: 4,
                    byte_offset: layout.topk_inds,
                }, // rhs_indices
                y_off, // buffer 5 = y
                Binding::Inline {
                    binding_index: 6,
                    value: p.top_k,
                },
            ],
        )
    };

    // Step 6: gate_out = affine_gather_qmv(x, W_expert_gate)
    cmds.push(gather_qmv(
        WeightTensor::MoeExpertGateW,
        WeightTensor::MoeExpertGateS,
        WeightTensor::MoeExpertGateB,
        Binding::ArenaSlot {
            slot: p.in_slot,
            binding_index: 3,
        },
        Binding::MoeScratch {
            binding_index: 5,
            byte_offset: layout.gate_out,
        },
        p.moe_inter,
        p.hidden,
        p.gate_bits,
    ));
    // Step 7: up_out = affine_gather_qmv(x, W_expert_up)
    cmds.push(gather_qmv(
        WeightTensor::MoeExpertUpW,
        WeightTensor::MoeExpertUpS,
        WeightTensor::MoeExpertUpB,
        Binding::ArenaSlot {
            slot: p.in_slot,
            binding_index: 3,
        },
        Binding::MoeScratch {
            binding_index: 5,
            byte_offset: layout.up_out,
        },
        p.moe_inter,
        p.hidden,
        p.up_bits,
    ));
    // Step 8: act_out = SiluMul(gate_out, up_out) — write into gate_out
    //
    // Reuses the existing `silu_mul_<dtype>` kernel from `silu_mul.metallib`.
    // Bindings match the standard SiluMul lowering (gate, up, out, n).
    {
        // silu_mul.metal is a 1-element-per-thread kernel
        // (`uint gid [[thread_position_in_grid]]`, `if gid >= SILU_MUL_N
        // return`). Total flat elements across all (token, slot)
        // pairs at bake time = `bucket_m * top_k * moe_inter`.
        // The MoE block writes gate_out / up_out as `[bucket_m,
        // top_k, moe_inter]` row-major, so a single 1D dispatch
        // covering that flat extent computes silu(gate)*up over the
        // whole region. m_scaling::X with `BucketM(bucket_m)` scales
        // the dispatch by `num_tokens` at runtime: the live thread
        // count becomes `num_tokens * top_k * moe_inter`, exactly
        // what the down_proj reads back. Mirrors the standalone
        // `I::SiluMul` lowering above.
        let n_total = p.bucket_m * p.top_k * p.moe_inter;
        let groups = n_total.div_ceil(256);
        cmds.push(make_moe_command(
            KernelId::SiluMul,
            "silu_mul",
            silu_mul_static_name(dtype),
            // n_features is bound as `[[function_constant(0)]]` of
            // type `uint` in silu_mul.metal — must use
            // `ConstantValue::uint` to match the kernel's expected
            // MTLDataType (the standalone SiluMul lowering uses
            // `SiluMulConstants` which wraps `::uint(...)` for the
            // same reason). Baked for max bucket; under-dispatched
            // tiles (num_tokens < bucket_m) just don't run, since
            // m_scaling shrinks the thread count proportionally.
            vec![ConstantValue::uint(0, n_total)],
            DispatchShape {
                threadgroups: (groups, 1, 1),
                threads_per_threadgroup: (256, 1, 1),
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::X,
                    bucket_m: super::ids::BucketM(p.bucket_m),
                }),
            },
            // `silu_mul` kernel signature: `(out @ 0, gate @ 1, up @ 2)`.
            // Computes `out = silu(gate) * up`. In Qwen3-MoE we want
            // `silu(gate_proj_out) * up_proj_out` — so `gate` (binding 1)
            // must point at `gate_out` and `up` (binding 2) at `up_out`.
            // Out aliases `gate_out` so the next kernel (down_proj)
            // reads the silumul result from there.
            //
            // Pre-2026-05-17 these were swapped: bind 1 = up_out, bind 2
            // = gate_out, producing `silu(up_proj) * gate_proj` which
            // matches neither Qwen3-MoE nor any other arch's MLP. Caught
            // by MLX-vs-metal layer-0 parity bisect at dispatch 24
            // (down_proj reads silumul values that diverge from MLX
            // expert-53 reference).
            vec![
                Binding::MoeScratch {
                    binding_index: 0,
                    byte_offset: layout.gate_out,
                },
                Binding::MoeScratch {
                    binding_index: 1,
                    byte_offset: layout.gate_out,
                },
                Binding::MoeScratch {
                    binding_index: 2,
                    byte_offset: layout.up_out,
                },
            ],
        ));
    }

    // Step 9: down_out = affine_gather_qmv(act_out, W_expert_down)
    //
    // act_out lives at `layout.gate_out` (the SiluMul wrote it there).
    // BUT this kernel's "x" input expects `[num_tokens, K]` shape; for
    // the expert-aware gather we actually need `x[token_n]` indexed by
    // `token_n = nk / top_k` inside the kernel. The MLX
    // affine_gather_qmv kernel does this lookup itself based on
    // rhs_indices stride. The `x` parameter here is the activation
    // **before** the per-token-broadcast — for the down projection
    // that's the `act_out` of dimension `[M*top_k, moe_inter]`. The
    // gather kernel handles the per-token replication via its
    // `token_n = nk / top_k` arithmetic; for the down step the
    // top_k-axis is already materialized in act_out, so we need a
    // different code path. **This is the unresolved binding shape
    // for the down projection.** Plan: use a 1-to-1 (top_k=1) view
    // since act_out is pre-replicated, OR port the
    // `affine_gather_qmm_rhs_*` kernel, which handles the
    // `[M, top_k, I] × [E, H, I]` shape natively.
    //
    // For this initial structural lowering, emit a single
    // affine_gather_qmv call with `top_k = 1` against an `x` of
    // shape `[M*top_k, moe_inter]` and same `rhs_indices`. The kernel
    // sees N = M*top_k rows and reads one expert per row. This is
    // **functionally correct** for the down step when `act_out` is
    // already replicated per-expert and `rhs_indices[token_n]` picks
    // the right expert. Validate empirically in §5.
    {
        let n_out = p.hidden;
        let k_in = p.moe_inter;
        let bn: u32 = 8;
        // down_proj's own on-disk width (OptiQ mixes it independently of
        // gate/up within a layer).
        let symbol =
            affine_gather_qmv_symbol(n_out, k_in, dtype, scale_dtype, p.group_size, p.down_bits);
        let kernel = if symbol.contains("_fast_") {
            KernelId::AffineGatherQmvFast
        } else {
            KernelId::AffineGatherQmv
        };
        cmds.push(make_moe_command(
            kernel,
            "quantized_qmv",
            symbol,
            vec![
                ConstantValue::int(0, k_in as i32),
                ConstantValue::int(1, n_out as i32),
            ],
            DispatchShape {
                threadgroups: (1, n_out.div_ceil(bn), p.bucket_m * p.top_k),
                threads_per_threadgroup: (32, 2, 1),
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::Z,
                    bucket_m: super::ids::BucketM(p.bucket_m),
                }),
            },
            vec![
                Binding::Weight {
                    kind: bundle_kind,
                    which: WeightTensor::MoeExpertDownW,
                    layer: layer_id,
                    locator: locator0,
                    binding_index: 0,
                },
                Binding::Weight {
                    kind: bundle_kind,
                    which: WeightTensor::MoeExpertDownS,
                    layer: layer_id,
                    locator: locator0,
                    binding_index: 1,
                },
                Binding::Weight {
                    kind: bundle_kind,
                    which: WeightTensor::MoeExpertDownB,
                    layer: layer_id,
                    locator: locator0,
                    binding_index: 2,
                },
                // x = act_out (M*top_k rows of moe_inter cols).
                Binding::MoeScratch {
                    binding_index: 3,
                    byte_offset: layout.gate_out,
                },
                Binding::MoeScratch {
                    binding_index: 4,
                    byte_offset: layout.topk_inds,
                },
                Binding::MoeScratch {
                    binding_index: 5,
                    byte_offset: layout.down_out,
                },
                // top_k = 1 for the down step since act_out is already
                // per-expert-replicated; kernel's `token_n = nk / 1`
                // walks the rows 1-for-1.
                Binding::Inline {
                    binding_index: 6,
                    value: 1,
                },
            ],
        ));
    }

    // ── Step 10: moe_out = MoeWeightedSum(down_out, topk_scores) ───
    //
    // Reduction: `out[n, d] = Σ_k expert[n, k, d] * scores[n, k]`.
    // Top-k + hidden ride on function_constants (0, 1). Dispatch is
    // 2D over (hidden, M); convert dispatchThreads → threadgroups.
    {
        let tg_x_threads = p.hidden.min(64);
        let tg_x_count = p.hidden.div_ceil(tg_x_threads);
        cmds.push(LoweredCommand {
            kernel: KernelId::MoeWeightedSum,
            library: "moe_weighted_sum",
            function: moe_weighted_sum_symbol(mc),
            constants: baked(vec![
                ConstantValue::int(0, p.top_k as i32),
                ConstantValue::int(1, p.hidden as i32),
            ]),
            dispatch: DispatchShape {
                threadgroups: (tg_x_count, p.bucket_m, 1),
                threads_per_threadgroup: (tg_x_threads, 1, 1),
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::Y,
                    bucket_m: super::ids::BucketM(p.bucket_m),
                }),
            },
            bindings: baked(vec![
                Binding::MoeScratch {
                    binding_index: 0,
                    byte_offset: layout.down_out,
                },
                Binding::MoeScratch {
                    binding_index: 1,
                    byte_offset: layout.topk_scores,
                },
                Binding::ArenaSlot {
                    slot: p.out_slot,
                    binding_index: 2,
                },
            ]),
            gemm_dims: None,
        });
        // suppress unused-helper warning when the long-form dtype
        // helper isn't otherwise referenced (it's used inside the
        // panic message of softmax_precise_symbol's Int4 arm but
        // optimizers strip those).
        let _ = long_dtype_infix;
    }

    cmds
}

/// `moe_per_expert_scale_{float16,bfloat16}` symbol picker.
fn moe_per_expert_scale_symbol(p: &MetalModelConsts) -> &'static str {
    match p.metal_dtype {
        MetalDtype::F16 => "moe_per_expert_scale_float16",
        MetalDtype::Bf16 => "moe_per_expert_scale_bfloat16",
        MetalDtype::Int4 => panic!("moe_per_expert_scale: int4 unreachable"),
    }
}

/// Inputs to [`lower_gemma_moe`] — the `I::GemmaMoe` decomposition.
struct GemmaMoeLowering {
    router_in_slot: u32,
    expert_in_slot: u32,
    out_slot: u32,
    layer: u32,
    num_experts: u32,
    top_k: u32,
    moe_inter: u32,
    hidden: u32,
    group_size: u32,
    /// On-disk expert bit-width (4 or 8). MLX-native mixed/dynamic quant
    /// (OptiQ) ships 8-bit switch_glu experts on the sensitive edge layers
    /// and 4-bit in the middle, so this is resolved per THIS layer.
    bits: u32,
    tape_index: u32,
    op_idx: u32,
    bucket_m: u32,
    /// M5 matrix accelerator available — routes the grouped expert GEMM
    /// to the NAX kernel (the dominant prefill lever; the steel grouped
    /// GEMM is ~5-10x slower per call).
    is_nax: bool,
}

/// Emit the multi-`LoweredCommand` decomposition for one `I::GemmaMoe`
/// instance (the Gemma-4 `gemma_moe` op). Adapts [`lower_metal_moe`] with the
/// Gemma-specific differences:
///   * a leading RMSNorm of `router_in` by the router gain (`router.scale`),
///   * the router GEMM reads the normed `xr` against the dequant'd dense gate,
///   * the gathered top-k scores are multiplied by the temperature
///     `hidden^-0.5` BEFORE the softmax,
///   * a per-expert-scale gather-multiply after the softmax,
///   * GeGLU (gelu_mul) instead of SiLU,
///   * two input slots (`router_in` for routing, `expert_in` for the experts),
///   * two weight bundles (`GemmaRouter` + `GemmaSwitchGlu`).
fn lower_gemma_moe(
    mc: &MetalModelConsts,
    p: GemmaMoeLowering,
    moe_scratch_bytes: &mut u32,
) -> Vec<LoweredCommand> {
    let dtype = dequant_dtype_for(mc);
    let scale_dtype = scale_dtype_for(mc);
    let elem = elem_size_bytes(dtype);

    let grouped = moe_grouped_enabled(p.bucket_m, p.top_k);
    let layout = MoeScratchLayout::compute(
        p.bucket_m,
        p.num_experts,
        p.top_k,
        p.moe_inter,
        p.hidden,
        elem,
        grouped,
    );
    // The router pre-norm output `xr` ([bucket_m, hidden]) is appended past
    // the shared MoE layout regions.
    let xr_off = align_256(layout.total);
    let total = align_256(xr_off + p.bucket_m * p.hidden * elem);
    *moe_scratch_bytes = (*moe_scratch_bytes).max(total);

    let layer_id = super::ids::LayerId(p.layer);
    let locator0 = crate::tape::lowered::WeightLocator {
        bucket: p.tape_index,
        op_idx: p.op_idx,
        slot: 0,
    };
    let router_kind = WeightBundleKind::GemmaRouter;
    let experts_kind = WeightBundleKind::GemmaSwitchGlu;

    let mut cmds: Vec<LoweredCommand> = Vec::with_capacity(13);

    // ── Step 0: xr = RmsNorm(router_in, router.scale) ──────────────
    //
    // Plain RMSNorm with the router gain; the hidden^-0.5 temperature is
    // folded into the softmax (Step 6), NOT this gain. Gemma's
    // `(1 + weight)` offset does NOT apply to `router.scale` (mlx uses the
    // gain verbatim) → weight_offset 0.0.
    cmds.push(LoweredCommand {
        kernel: KernelId::RmsNorm,
        library: "rmsnorm",
        function: rmsnorm_kernel_static_name(mc, scale_dtype_for(mc)),
        constants: super::kernel_constants::RmsNormConstants {
            bucket_m: super::ids::BucketM(p.bucket_m),
            q_size: super::ids::QSize(p.hidden),
            rms_norm_eps: super::ids::RmsNormEps(mc.rms_norm_eps),
            weight_offset: 0.0,
        }
        .into_baked(),
        dispatch: DispatchShape {
            threadgroups: (p.bucket_m, 1, 1),
            threads_per_threadgroup: (THREADS_PER_GROUP, 1, 1),
            m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                seq_axis: None,
                axis: crate::tape::lowered::MScaleAxis::X,
                bucket_m: super::ids::BucketM(p.bucket_m),
            }),
        },
        bindings: baked(vec![
            Binding::MoeScratch {
                binding_index: 0,
                byte_offset: xr_off,
            },
            Binding::ArenaSlot {
                slot: p.router_in_slot,
                binding_index: 1,
            },
            Binding::Weight {
                kind: router_kind,
                which: WeightTensor::GemmaRouterScale,
                layer: layer_id,
                locator: locator0,
                binding_index: 2,
            },
        ]),
        gemm_dims: None,
    });

    // ── Step 1: router_logits = Gemm(xr, router_gate) ─────────────
    {
        let tg_x = p.bucket_m.div_ceil(GEMM_TILE_M);
        let tg_y = p.num_experts.div_ceil(GEMM_TILE_N);
        cmds.push(LoweredCommand {
            kernel: KernelId::Gemm,
            library: "",
            function: "",
            constants: &[],
            dispatch: DispatchShape {
                threadgroups: (tg_x, tg_y, 1),
                threads_per_threadgroup: (GEMM_TILE_M, GEMM_TILE_N, 1),
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::X,
                    bucket_m: super::ids::BucketM(p.bucket_m),
                }),
            },
            bindings: baked(vec![
                Binding::MoeScratch {
                    binding_index: 0,
                    byte_offset: layout.router_logits,
                },
                Binding::MoeScratch {
                    binding_index: 1,
                    byte_offset: xr_off,
                },
                Binding::Weight {
                    kind: router_kind,
                    which: WeightTensor::GemmaRouterGate,
                    layer: layer_id,
                    locator: locator0,
                    binding_index: 2,
                },
            ]),
            gemm_dims: Some(GemmDims {
                m: p.bucket_m,
                n: p.num_experts,
                k: p.hidden,
            }),
        });
    }

    // ── Step 2: sorted_full_inds = ArgPartitionTopK(router_logits) ─
    {
        let bn: u32 = if p.num_experts > 128 { 64 } else { 32 };
        cmds.push(make_moe_command(
            KernelId::ArgPartitionTopK,
            "argpartition",
            argpartition_symbol(mc, p.num_experts),
            Vec::new(),
            DispatchShape {
                threadgroups: (1, p.bucket_m, 1),
                threads_per_threadgroup: (bn, 1, 1),
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::Y,
                    bucket_m: super::ids::BucketM(p.bucket_m),
                }),
            },
            vec![
                Binding::MoeScratch {
                    binding_index: 0,
                    byte_offset: layout.router_logits,
                },
                Binding::MoeScratch {
                    binding_index: 1,
                    byte_offset: layout.sorted_full,
                },
                Binding::Inline {
                    binding_index: 2,
                    value: p.num_experts,
                },
                Binding::Inline {
                    binding_index: 3,
                    value: 1,
                },
                Binding::Inline {
                    binding_index: 4,
                    value: 1,
                },
                Binding::Inline {
                    binding_index: 5,
                    value: p.num_experts,
                },
                Binding::Inline {
                    binding_index: 6,
                    value: p.num_experts,
                },
            ],
        ));
    }

    // ── Step 3: topk_inds = SliceTrailingColsU32(sorted_full_inds) ─
    {
        let tg_x_threads = p.top_k.min(32);
        let tg_x_count = p.top_k.div_ceil(tg_x_threads);
        cmds.push(make_moe_command(
            KernelId::SliceTrailingColsU32,
            "slice_trailing_cols",
            "slice_trailing_cols_u32",
            Vec::new(),
            DispatchShape {
                threadgroups: (tg_x_count, p.bucket_m, 1),
                threads_per_threadgroup: (tg_x_threads, 1, 1),
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::Y,
                    bucket_m: super::ids::BucketM(p.bucket_m),
                }),
            },
            vec![
                Binding::MoeScratch {
                    binding_index: 0,
                    byte_offset: layout.sorted_full,
                },
                Binding::MoeScratch {
                    binding_index: 1,
                    byte_offset: layout.topk_inds,
                },
                Binding::Inline {
                    binding_index: 2,
                    value: p.num_experts,
                },
                Binding::Inline {
                    binding_index: 3,
                    value: p.top_k,
                },
            ],
        ));
    }

    // ── Step 4: topk_scores = TakeAlongAxis(router_logits, topk_inds) ─
    {
        let tg_x_threads = p.top_k.min(32);
        let tg_x_count = p.top_k.div_ceil(tg_x_threads);
        cmds.push(make_moe_command(
            KernelId::TakeAlongAxis,
            "take_along_axis",
            take_along_axis_symbol(mc),
            Vec::new(),
            DispatchShape {
                threadgroups: (tg_x_count, p.bucket_m, 1),
                threads_per_threadgroup: (tg_x_threads, 1, 1),
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::Y,
                    bucket_m: super::ids::BucketM(p.bucket_m),
                }),
            },
            vec![
                Binding::MoeScratch {
                    binding_index: 0,
                    byte_offset: layout.router_logits,
                },
                Binding::MoeScratch {
                    binding_index: 1,
                    byte_offset: layout.topk_inds,
                },
                Binding::MoeScratch {
                    binding_index: 2,
                    byte_offset: layout.topk_scores,
                },
                Binding::Inline {
                    binding_index: 3,
                    value: p.num_experts,
                },
                Binding::Inline {
                    binding_index: 4,
                    value: p.top_k,
                },
            ],
        ));
    }

    // ── Step 5: TEMPERATURE — topk_scores *= hidden^-0.5 (before softmax) ─
    //
    // mlx folds `hidden^-0.5` as the softmax temperature on the gathered
    // top-k scores. ArgPartition is scale-invariant for positive scale, so
    // applying it here (rather than to the router gain) is mathematically
    // identical and avoids a load-time weight transform.
    {
        let temperature = (p.hidden as f32).powf(-0.5);
        let n_total = p.bucket_m * p.top_k;
        cmds.push(LoweredCommand {
            kernel: KernelId::ScalarMul,
            library: "elementwise",
            function: pick_specialized_symbol(
                "scalar_mul_f16_specialized",
                "scalar_mul_bf16_specialized",
                mc.metal_dtype,
            ),
            constants: baked(vec![ConstantValue::float(2, temperature)]),
            dispatch: {
                let mut d = DispatchShape::dispatch_1d(n_total, THREADS_PER_GROUP);
                d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::X,
                    bucket_m: super::ids::BucketM(p.bucket_m),
                });
                d
            },
            bindings: baked(vec![
                Binding::MoeScratch {
                    binding_index: 0,
                    byte_offset: layout.topk_scores,
                },
                Binding::MoeScratch {
                    binding_index: 1,
                    byte_offset: layout.topk_scores,
                },
            ]),
            gemm_dims: None,
        });
    }

    // ── Step 6: softmax over the top-k gathered scores (in place) ──
    cmds.push(make_moe_command(
        KernelId::Softmax,
        "softmax",
        softmax_precise_symbol(mc),
        Vec::new(),
        DispatchShape {
            threadgroups: (p.bucket_m, 1, 1),
            threads_per_threadgroup: (256, 1, 1),
            m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                seq_axis: None,
                axis: crate::tape::lowered::MScaleAxis::X,
                bucket_m: super::ids::BucketM(p.bucket_m),
            }),
        },
        vec![
            Binding::MoeScratch {
                binding_index: 0,
                byte_offset: layout.topk_scores,
            },
            Binding::MoeScratch {
                binding_index: 1,
                byte_offset: layout.topk_scores,
            },
            Binding::Inline {
                binding_index: 2,
                value: p.top_k,
            },
        ],
    ));

    // ── Step 7: topk_scores[m,j] *= per_expert_scale[topk_inds[m,j]] ─
    {
        let n_total = p.bucket_m * p.top_k;
        cmds.push(make_moe_command(
            KernelId::MoePerExpertScale,
            "moe_per_expert_scale",
            moe_per_expert_scale_symbol(mc),
            vec![ConstantValue::uint(0, n_total)],
            {
                let mut d = DispatchShape::dispatch_1d(n_total, THREADS_PER_GROUP);
                d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::X,
                    bucket_m: super::ids::BucketM(p.bucket_m),
                });
                d
            },
            vec![
                Binding::MoeScratch {
                    binding_index: 0,
                    byte_offset: layout.topk_scores,
                },
                Binding::MoeScratch {
                    binding_index: 1,
                    byte_offset: layout.topk_inds,
                },
                Binding::Weight {
                    kind: router_kind,
                    which: WeightTensor::GemmaPerExpertScale,
                    layer: layer_id,
                    locator: locator0,
                    binding_index: 2,
                },
            ],
        ));
    }

    // ── Steps 8-11: expert gate/up/gelu/down via affine_gather_qmv ──
    //
    // Reads `expert_in` (input 1) against the GemmaSwitchGlu expert stack.
    let gather_qmv =
        |which_w, which_s, which_b, x_off, y_off, n_out: u32, k_in: u32| -> LoweredCommand {
            let symbol =
                affine_gather_qmv_symbol(n_out, k_in, dtype, scale_dtype, p.group_size, p.bits);
            let bn: u32 = 8;
            let kernel = if symbol.contains("_fast_") {
                KernelId::AffineGatherQmvFast
            } else {
                KernelId::AffineGatherQmv
            };
            make_moe_command(
                kernel,
                "quantized_qmv",
                symbol,
                vec![
                    ConstantValue::int(0, k_in as i32),
                    ConstantValue::int(1, n_out as i32),
                ],
                DispatchShape {
                    threadgroups: (1, n_out.div_ceil(bn), p.bucket_m * p.top_k),
                    threads_per_threadgroup: (32, 2, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::Z,
                        bucket_m: super::ids::BucketM(p.bucket_m),
                    }),
                },
                vec![
                    Binding::Weight {
                        kind: experts_kind,
                        which: which_w,
                        layer: layer_id,
                        locator: locator0,
                        binding_index: 0,
                    },
                    Binding::Weight {
                        kind: experts_kind,
                        which: which_s,
                        layer: layer_id,
                        locator: locator0,
                        binding_index: 1,
                    },
                    Binding::Weight {
                        kind: experts_kind,
                        which: which_b,
                        layer: layer_id,
                        locator: locator0,
                        binding_index: 2,
                    },
                    x_off,
                    Binding::MoeScratch {
                        binding_index: 4,
                        byte_offset: layout.topk_inds,
                    },
                    y_off,
                    Binding::Inline {
                        binding_index: 6,
                        value: p.top_k,
                    },
                ],
            )
        };

    if grouped {
        // ── Grouped expert GEMM (mlx gather_qmm): sort tokens by expert
        //    into a padded layout, run a batched per-expert GEMM
        //    (gate/up/gelu/down), then un-scatter to token order. Writes
        //    layout.down_out so the shared Step 12 reduces it unchanged.
        use crate::tape::lowered::{MScaleAxis, MScaling};
        let elem_dtype = mc.metal_dtype;
        let m_pairs = p.bucket_m * p.top_k;
        let mpad_max = layout.mpad_max;
        let bm = super::ids::BucketM(p.bucket_m);
        // G1: histogram + padded-offset scan (single threadgroup).
        cmds.push(make_moe_command(
            KernelId::MoeGroupOffsets,
            "moe_group",
            "moe_group_offsets",
            vec![
                ConstantValue::int(0, m_pairs as i32),
                ConstantValue::int(1, p.num_experts as i32),
            ],
            DispatchShape {
                threadgroups: (1, 1, 1),
                threads_per_threadgroup: (256, 1, 1),
                m_scaling: None,
            },
            vec![
                Binding::MoeScratch {
                    binding_index: 0,
                    byte_offset: layout.topk_inds,
                },
                Binding::MoeScratch {
                    binding_index: 1,
                    byte_offset: layout.grp_count,
                },
                Binding::MoeScratch {
                    binding_index: 2,
                    byte_offset: layout.grp_offset,
                },
                Binding::MoeScratch {
                    binding_index: 3,
                    byte_offset: layout.grp_total,
                },
            ],
        ));
        // G2: sentinel-fill indices_pad + zero fill counters.
        cmds.push(make_moe_command(
            KernelId::MoeGroupInit,
            "moe_group",
            "moe_group_init",
            vec![
                ConstantValue::int(1, p.num_experts as i32),
                ConstantValue::int(2, mpad_max as i32),
            ],
            DispatchShape {
                threadgroups: (mpad_max.div_ceil(256), 1, 1),
                threads_per_threadgroup: (256, 1, 1),
                m_scaling: None,
            },
            vec![
                Binding::MoeScratch {
                    binding_index: 0,
                    byte_offset: layout.grp_indices_pad,
                },
                Binding::MoeScratch {
                    binding_index: 1,
                    byte_offset: layout.grp_fill,
                },
            ],
        ));
        // G3: scatter each real (token,expert) row to its padded slot
        //     (m-scaled over actual pairs).
        cmds.push(make_moe_command(
            KernelId::MoeGroupScatter,
            "moe_group",
            moe_group_scatter_symbol(elem_dtype),
            vec![
                ConstantValue::int(0, m_pairs as i32),
                ConstantValue::int(1, p.num_experts as i32),
                ConstantValue::int(3, p.top_k as i32),
                ConstantValue::int(4, p.hidden as i32),
            ],
            DispatchShape {
                threadgroups: (1, m_pairs, 1),
                threads_per_threadgroup: (p.hidden.min(256), 1, 1),
                m_scaling: Some(MScaling {
                    seq_axis: None,
                    axis: MScaleAxis::Y,
                    bucket_m: bm,
                }),
            },
            vec![
                Binding::MoeScratch {
                    binding_index: 0,
                    byte_offset: layout.topk_inds,
                },
                Binding::MoeScratch {
                    binding_index: 1,
                    byte_offset: layout.grp_offset,
                },
                Binding::ArenaSlot {
                    slot: p.expert_in_slot,
                    binding_index: 2,
                },
                Binding::MoeScratch {
                    binding_index: 3,
                    byte_offset: layout.grp_fill,
                },
                Binding::MoeScratch {
                    binding_index: 4,
                    byte_offset: layout.grp_pos,
                },
                Binding::MoeScratch {
                    binding_index: 5,
                    byte_offset: layout.grp_indices_pad,
                },
                Binding::MoeScratch {
                    binding_index: 6,
                    byte_offset: layout.grp_x_pad,
                },
            ],
        ));
        // Grouped GEMM helper: y[Mpad, n_out] = gather_qmm(x_pad, W, indices_pad).
        let gemm = |which_w, which_s, which_b, x_byte: u32, y_byte: u32, n_out: u32, k_in: u32| {
            // NAX (M5 matrix accelerator) is ~5-10x faster per call than
            // the steel grouped GEMM; use it when available + N%64==0 +
            // gs in {64,128} (BM=64 tile). The host padded to BM=64, so a
            // 64-row tile maps to one expert. Else fall back to steel.
            let use_nax =
                p.is_nax && n_out.is_multiple_of(64) && (p.group_size == 64 || p.group_size == 128);
            let (kernel, library, symbol, tile): (KernelId, &'static str, &'static str, u32) =
                if use_nax {
                    (
                        KernelId::AffineGatherQmmTNax,
                        "quantized_qmm_nax",
                        affine_gather_qmm_t_nax_symbol(dtype, scale_dtype, p.group_size, p.bits),
                        64,
                    )
                } else {
                    (
                        KernelId::AffineGatherQmmT,
                        "quantized_qmm",
                        affine_gather_qmm_t_symbol(
                            dtype,
                            scale_dtype,
                            p.group_size,
                            n_out.is_multiple_of(32),
                            p.bits,
                        ),
                        32,
                    )
                };
            make_moe_command(
                kernel,
                library,
                symbol,
                vec![
                    ConstantValue::int(0, k_in as i32),
                    ConstantValue::int(1, n_out as i32),
                    ConstantValue::int(2, mpad_max as i32),
                    ConstantValue::int(4, p.num_experts as i32),
                ],
                DispatchShape {
                    threadgroups: (n_out.div_ceil(tile), mpad_max.div_ceil(tile), 1),
                    threads_per_threadgroup: (32, 2, 2),
                    m_scaling: None,
                },
                vec![
                    Binding::Weight {
                        kind: experts_kind,
                        which: which_w,
                        layer: layer_id,
                        locator: locator0,
                        binding_index: 0,
                    },
                    Binding::Weight {
                        kind: experts_kind,
                        which: which_s,
                        layer: layer_id,
                        locator: locator0,
                        binding_index: 1,
                    },
                    Binding::Weight {
                        kind: experts_kind,
                        which: which_b,
                        layer: layer_id,
                        locator: locator0,
                        binding_index: 2,
                    },
                    Binding::MoeScratch {
                        binding_index: 3,
                        byte_offset: x_byte,
                    },
                    Binding::MoeScratch {
                        binding_index: 4,
                        byte_offset: y_byte,
                    },
                    Binding::MoeScratch {
                        binding_index: 5,
                        byte_offset: layout.grp_indices_pad,
                    },
                ],
            )
        };
        // G4/G5: gate + up.
        cmds.push(gemm(
            WeightTensor::MoeExpertGateW,
            WeightTensor::MoeExpertGateS,
            WeightTensor::MoeExpertGateB,
            layout.grp_x_pad,
            layout.grp_gate_pad,
            p.moe_inter,
            p.hidden,
        ));
        cmds.push(gemm(
            WeightTensor::MoeExpertUpW,
            WeightTensor::MoeExpertUpS,
            WeightTensor::MoeExpertUpB,
            layout.grp_x_pad,
            layout.grp_up_pad,
            p.moe_inter,
            p.hidden,
        ));
        // G6: GeGLU activation (gemma uses gelu_pytorch_tanh) over Mpad rows.
        {
            let n_total = mpad_max * p.moe_inter;
            cmds.push(make_moe_command(
                KernelId::GeluMul,
                "silu_mul",
                gelu_mul_static_name(dtype),
                vec![ConstantValue::uint(0, n_total)],
                DispatchShape {
                    threadgroups: (n_total.div_ceil(256), 1, 1),
                    threads_per_threadgroup: (256, 1, 1),
                    m_scaling: None,
                },
                vec![
                    Binding::MoeScratch {
                        binding_index: 0,
                        byte_offset: layout.grp_gate_pad,
                    },
                    Binding::MoeScratch {
                        binding_index: 1,
                        byte_offset: layout.grp_gate_pad,
                    },
                    Binding::MoeScratch {
                        binding_index: 2,
                        byte_offset: layout.grp_up_pad,
                    },
                ],
            ));
        }
        // G7: down (act = gate_pad → down_pad).
        cmds.push(gemm(
            WeightTensor::MoeExpertDownW,
            WeightTensor::MoeExpertDownS,
            WeightTensor::MoeExpertDownB,
            layout.grp_gate_pad,
            layout.grp_down_pad,
            p.hidden,
            p.moe_inter,
        ));
        // G8: un-scatter down_pad → down_out (token order, m-scaled).
        cmds.push(make_moe_command(
            KernelId::MoeGroupGather,
            "moe_group",
            moe_group_gather_symbol(elem_dtype),
            vec![ConstantValue::int(5, p.hidden as i32)],
            DispatchShape {
                threadgroups: (1, m_pairs, 1),
                threads_per_threadgroup: (p.hidden.min(256), 1, 1),
                m_scaling: Some(MScaling {
                    seq_axis: None,
                    axis: MScaleAxis::Y,
                    bucket_m: bm,
                }),
            },
            vec![
                Binding::MoeScratch {
                    binding_index: 0,
                    byte_offset: layout.grp_down_pad,
                },
                Binding::MoeScratch {
                    binding_index: 1,
                    byte_offset: layout.grp_pos,
                },
                Binding::MoeScratch {
                    binding_index: 2,
                    byte_offset: layout.down_out,
                },
            ],
        ));
    } else {
        // Step 8: gate_out = affine_gather_qmv(expert_in, W_gate)
        cmds.push(gather_qmv(
            WeightTensor::MoeExpertGateW,
            WeightTensor::MoeExpertGateS,
            WeightTensor::MoeExpertGateB,
            Binding::ArenaSlot {
                slot: p.expert_in_slot,
                binding_index: 3,
            },
            Binding::MoeScratch {
                binding_index: 5,
                byte_offset: layout.gate_out,
            },
            p.moe_inter,
            p.hidden,
        ));
        // Step 9: up_out = affine_gather_qmv(expert_in, W_up)
        cmds.push(gather_qmv(
            WeightTensor::MoeExpertUpW,
            WeightTensor::MoeExpertUpS,
            WeightTensor::MoeExpertUpB,
            Binding::ArenaSlot {
                slot: p.expert_in_slot,
                binding_index: 3,
            },
            Binding::MoeScratch {
                binding_index: 5,
                byte_offset: layout.up_out,
            },
            p.moe_inter,
            p.hidden,
        ));

        // Step 10: act_out = GeluMul(gate_out, up_out) → write into gate_out.
        //
        // Gemma experts use GeGLU (gelu_pytorch_tanh), NOT SiLU. Reuses the
        // existing `gelu_mul_<dtype>` kernel (silu_mul.metal:66).
        {
            let n_total = p.bucket_m * p.top_k * p.moe_inter;
            let groups = n_total.div_ceil(256);
            cmds.push(make_moe_command(
                KernelId::GeluMul,
                "silu_mul",
                gelu_mul_static_name(dtype),
                vec![ConstantValue::uint(0, n_total)],
                DispatchShape {
                    threadgroups: (groups, 1, 1),
                    threads_per_threadgroup: (256, 1, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::X,
                        bucket_m: super::ids::BucketM(p.bucket_m),
                    }),
                },
                vec![
                    Binding::MoeScratch {
                        binding_index: 0,
                        byte_offset: layout.gate_out,
                    },
                    Binding::MoeScratch {
                        binding_index: 1,
                        byte_offset: layout.gate_out,
                    },
                    Binding::MoeScratch {
                        binding_index: 2,
                        byte_offset: layout.up_out,
                    },
                ],
            ));
        }

        // Step 11: down_out = affine_gather_qmv(act_out, W_down). act_out lives
        // at gate_out (the GeluMul wrote it there); the activation is already
        // per-expert-replicated so top_k=1 walks the rows 1-for-1 (same as
        // lower_metal_moe's down step).
        {
            let n_out = p.hidden;
            let k_in = p.moe_inter;
            let bn: u32 = 8;
            let symbol =
                affine_gather_qmv_symbol(n_out, k_in, dtype, scale_dtype, p.group_size, p.bits);
            let kernel = if symbol.contains("_fast_") {
                KernelId::AffineGatherQmvFast
            } else {
                KernelId::AffineGatherQmv
            };
            cmds.push(make_moe_command(
                kernel,
                "quantized_qmv",
                symbol,
                vec![
                    ConstantValue::int(0, k_in as i32),
                    ConstantValue::int(1, n_out as i32),
                ],
                DispatchShape {
                    threadgroups: (1, n_out.div_ceil(bn), p.bucket_m * p.top_k),
                    threads_per_threadgroup: (32, 2, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                        seq_axis: None,
                        axis: crate::tape::lowered::MScaleAxis::Z,
                        bucket_m: super::ids::BucketM(p.bucket_m),
                    }),
                },
                vec![
                    Binding::Weight {
                        kind: experts_kind,
                        which: WeightTensor::MoeExpertDownW,
                        layer: layer_id,
                        locator: locator0,
                        binding_index: 0,
                    },
                    Binding::Weight {
                        kind: experts_kind,
                        which: WeightTensor::MoeExpertDownS,
                        layer: layer_id,
                        locator: locator0,
                        binding_index: 1,
                    },
                    Binding::Weight {
                        kind: experts_kind,
                        which: WeightTensor::MoeExpertDownB,
                        layer: layer_id,
                        locator: locator0,
                        binding_index: 2,
                    },
                    Binding::MoeScratch {
                        binding_index: 3,
                        byte_offset: layout.gate_out,
                    },
                    Binding::MoeScratch {
                        binding_index: 4,
                        byte_offset: layout.topk_inds,
                    },
                    Binding::MoeScratch {
                        binding_index: 5,
                        byte_offset: layout.down_out,
                    },
                    Binding::Inline {
                        binding_index: 6,
                        value: 1,
                    },
                ],
            ));
        }
    } // end else (matvec expert path)

    // ── Step 12: moe_out = MoeWeightedSum(down_out, topk_scores) ───
    {
        let tg_x_threads = p.hidden.min(64);
        let tg_x_count = p.hidden.div_ceil(tg_x_threads);
        cmds.push(LoweredCommand {
            kernel: KernelId::MoeWeightedSum,
            library: "moe_weighted_sum",
            function: moe_weighted_sum_symbol(mc),
            constants: baked(vec![
                ConstantValue::int(0, p.top_k as i32),
                ConstantValue::int(1, p.hidden as i32),
            ]),
            dispatch: DispatchShape {
                threadgroups: (tg_x_count, p.bucket_m, 1),
                threads_per_threadgroup: (tg_x_threads, 1, 1),
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    seq_axis: None,
                    axis: crate::tape::lowered::MScaleAxis::Y,
                    bucket_m: super::ids::BucketM(p.bucket_m),
                }),
            },
            bindings: baked(vec![
                Binding::MoeScratch {
                    binding_index: 0,
                    byte_offset: layout.down_out,
                },
                Binding::MoeScratch {
                    binding_index: 1,
                    byte_offset: layout.topk_scores,
                },
                Binding::ArenaSlot {
                    slot: p.out_slot,
                    binding_index: 2,
                },
            ]),
            gemm_dims: None,
        });
        let _ = long_dtype_infix;
    }

    cmds
}

#[cfg(test)]
mod tests {
    use super::*;
    use scratchy_ir::CanonicalParams;

    fn tp() -> MetalModelConsts {
        MetalModelConsts::from_canonical::<TestParams>()
    }
    use scratchy_ir::Instruction;

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
    // resolve a weight bundle (resolution is a worker bake-time concern), so
    // the empty impl suffices.
    impl scratchy_ir::WeightAccessors for TestParams {}

    /// Lower one layer — the KV writer, then its attention — at `bucket_m`.
    fn lower_tq_layer(attention: Instruction, bucket_m: u32) -> LoweredMetalTape {
        let writer = Instruction::RopeAppend(0, 1, 2, 3, 4, 5, /*layer=*/ 0, false, true);
        lower_tq(&tp(), &[writer, attention], bucket_m)
    }

    fn lower_tq(p: &MetalModelConsts, backbone: &[Instruction], bucket_m: u32) -> LoweredMetalTape {
        lower_pair(
            p,
            /*chunked=*/ false,
            backbone,
            &[],
            /*backbone_barriers=*/ &vec![true; backbone.len()],
            /*lm_head_barriers=*/ &[],
            bucket_m,
            /*num_arena_slots=*/ 8,
            /*backbone_tape_index=*/ 0,
            /*lm_head_tape_index=*/ 1,
            /*block_cap=*/ 128,
            /*profile=*/ None,
        )
        .expect("lower_pair")
    }

    /// Each injected TurboQuant command must carry the `OnlyIfTurboquant` gate
    /// so it is skipped on a non-turboquant (fp16) KV cache — otherwise the TQ
    /// kernels dispatch on fp16, read OOB from the unbound packed/norms fallback
    /// bindings, and silently WEDGE the GPU. The gate is fused onto each
    /// `GatedCommand`, so it can no longer be dropped by a parallel-vec mishap
    /// (that is now a compile error); these tests verify `inject_tq` assigns
    /// the gate VALUE correctly and `lower_pair` carries it. Pure-CPU lowering
    /// checks — never submit a Metal command buffer.
    ///
    /// Decode: no dequant pass; the plain attention runs only off TurboQuant
    /// and its `AttentionViaCacheTq` twin — same geometry, plus the codebook
    /// width and the packed-store bindings — only on TurboQuant KV.
    #[test]
    fn decode_turboquant_reads_packed_store_without_dequant() {
        use crate::tape::lowered::RuntimeGate::{
            self, OnlyIfTurboquant, OnlyIfTurboquantDecode, UnlessTurboquantDecode,
        };
        let tape = lower_tq_layer(Instruction::AttentionViaCache(3, 6, 0, true), 1);
        let steps: Vec<(KernelId, Option<RuntimeGate>)> = tape
            .commands
            .iter()
            .map(|c| (c.command.kernel, c.gate))
            .collect();
        assert_eq!(
            steps,
            [
                (KernelId::RopeAppend, None),
                (KernelId::TqQuantizeToPacked, Some(OnlyIfTurboquant)),
                (KernelId::TqQuantizeToPacked, Some(OnlyIfTurboquant)),
                (KernelId::AttentionViaCache, Some(UnlessTurboquantDecode)),
                (KernelId::AttentionViaCacheTq, Some(OnlyIfTurboquantDecode)),
            ]
        );
        let (fp16, tq) = (&tape.commands[3].command, &tape.commands[4].command);
        assert_eq!(
            (fp16.library, fp16.function, fp16.dispatch),
            (tq.library, tq.function, tq.dispatch)
        );
        let bits = crate::turboquant::tq_bits(tp().tq_kv_bits);
        assert_eq!(tq.constants[..fp16.constants.len()], *fp16.constants);
        assert_eq!(
            tq.constants[fp16.constants.len()..],
            [ConstantValue::uint(13, bits)]
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

    use crate::tape::lowered::RuntimeGate;

    /// The TurboQuant commands a multi-token tape wraps around one layer's
    /// paged attention: stage K and V and rotate q before it, rotate its output
    /// back after it (off decode steps), its decode twin (on decode steps).
    fn tq_attention_steps(
        prefill_attention: &[(KernelId, Option<RuntimeGate>)],
    ) -> Vec<(KernelId, Option<RuntimeGate>)> {
        use crate::tape::lowered::RuntimeGate::{
            OnlyIfTurboquantDecode, OnlyIfTurboquantNotDecode, UnlessTurboquantDecode,
        };
        let mut steps = vec![
            (KernelId::TqStageRotated, Some(OnlyIfTurboquantNotDecode)),
            (KernelId::TqStageRotated, Some(OnlyIfTurboquantNotDecode)),
            (KernelId::TqRotateRows, Some(OnlyIfTurboquantNotDecode)),
        ];
        steps.extend(
            prefill_attention
                .iter()
                .map(|&(k, _)| (k, Some(UnlessTurboquantDecode))),
        );
        steps.push((KernelId::TqRotateRows, Some(OnlyIfTurboquantNotDecode)));
        steps.push((KernelId::AttentionViaCacheTq, Some(OnlyIfTurboquantDecode)));
        steps
    }

    fn gated_steps(tape: &LoweredMetalTape) -> Vec<(KernelId, Option<RuntimeGate>)> {
        tape.commands
            .iter()
            .map(|c| (c.command.kernel, c.gate))
            .collect()
    }

    /// Hybrid arches (gemma-4) compress only the GLOBAL layers: the global
    /// writer is followed by its quantize and the global attention wrapped in
    /// the TurboQuant commands, while the sliding layer stays plain fp16.
    #[test]
    fn hybrid_turboquant_compresses_global_layers_only() {
        use crate::tape::lowered::RuntimeGate::{
            OnlyIfTurboquant, OnlyIfTurboquantDecode, UnlessTurboquantDecode,
        };
        let p = MetalModelConsts {
            global_head_dim: 512,
            num_global_kv_heads: 1,
            global_block_size: 32,
            global_rot_dim: 128,
            sliding_window: 1024,
            ..tp()
        };
        let global_writer = Instruction::RopeAppend(0, 1, 2, 3, 4, 5, 0, false, true);
        let sliding_writer = Instruction::RopeAppend(0, 1, 2, 3, 4, 5, 1, false, false);
        let quantize = [
            (KernelId::TqQuantizeToPacked, Some(OnlyIfTurboquant)),
            (KernelId::TqQuantizeToPacked, Some(OnlyIfTurboquant)),
        ];
        let decode = lower_tq(
            &p,
            &[
                global_writer,
                Instruction::AttentionViaCache(3, 6, 0, true),
                sliding_writer,
                Instruction::SlidingAttentionViaCache(3, 6, 1, true),
            ],
            1,
        );
        let mut want = vec![(KernelId::RopeAppend, None)];
        want.extend(quantize);
        want.extend([
            (KernelId::AttentionViaCache, Some(UnlessTurboquantDecode)),
            (KernelId::AttentionViaCacheTq, Some(OnlyIfTurboquantDecode)),
            (KernelId::RopeAppend, None),
            (KernelId::AttentionViaCache, None),
        ]);
        assert_eq!(gated_steps(&decode), want);

        let global_attention = Instruction::AttentionPrefillPaged(3, 6, 0, false);
        let sliding_attention = Instruction::SlidingAttentionPrefillPaged(3, 6, 1, false);
        let plain = |i: Instruction| gated_steps(&lower_tq(&p, &[i], 64));
        let own = |i: Instruction| {
            plain(i)
                .into_iter()
                .filter(|&(_, g)| g == Some(UnlessTurboquantDecode))
                .collect::<Vec<_>>()
        };
        let prefill = lower_tq(
            &p,
            &[
                global_writer,
                global_attention,
                sliding_writer,
                sliding_attention,
            ],
            64,
        );
        let mut want = vec![(KernelId::RopeAppend, None)];
        want.extend(quantize);
        want.extend(tq_attention_steps(&own(global_attention)));
        want.push((KernelId::RopeAppend, None));
        want.extend(plain(sliding_attention));
        assert_eq!(gated_steps(&prefill), want);
    }

    /// On NAX, an MLX-affine 4-bit GEMM in a bucket that can see a 5–16-token
    /// step gets its small-M twin — the 8-row tile in the 8-token bucket, the
    /// 16-row one in the 64-token bucket — gated against the GEMM it replaces,
    /// with the GEMM's own weights and slots. Batch 1, the prefill buckets,
    /// 8-bit weights and non-NAX devices are left as they were.
    #[test]
    fn small_m_twin_serves_decode_batches_on_nax() {
        use crate::tape::lowered::RuntimeGate::{OnlyIfSmallMTokens, UnlessSmallMTokens};
        let gemm = |bits| Instruction::AffineQmm(0, 1, 0, 3072, 8192, 64, bits, 10);
        let lower_at = |instruction, bucket_m, profile| {
            lower_pair(
                &tp(),
                false,
                &[instruction],
                &[],
                &[true],
                &[],
                bucket_m,
                8,
                0,
                1,
                128,
                profile,
            )
            .expect("lower_pair")
        };
        let m5 = Some(&crate::targets::M5_10CORE);
        for (bucket_m, own, tile) in [
            (8, KernelId::AffineQmvFast, SmallMTile::Rows8),
            (64, KernelId::AffineQmmTNax, SmallMTile::Rows16),
        ] {
            let tape = lower_at(gemm(4), bucket_m, m5);
            let steps: Vec<_> = gated_steps(&tape);
            assert_eq!(
                steps,
                [
                    (own, Some(UnlessSmallMTokens)),
                    (KernelId::AffineQmmSmallM, Some(OnlyIfSmallMTokens)),
                ]
            );
            let (gemm_cmd, small) = (&tape.commands[0].command, &tape.commands[1].command);
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
                small.bindings == gemm_cmd.bindings,
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

    /// A multi-token tape runs its own attention off decode steps, in the
    /// codebook's rotated domain: K/V staged and q rotated before it, the
    /// output rotated back after it. On a decode step — one token per
    /// sequence, e.g. a batch of decoding sequences — it yields to the
    /// `AttentionViaCacheTq` twin of its decode-kernel form. There is no
    /// dequant pass on any step.
    #[test]
    fn prefill_turboquant_attends_in_the_rotated_domain() {
        use crate::tape::lowered::RuntimeGate::{OnlyIfTurboquant, UnlessTurboquantDecode};
        let attention = Instruction::AttentionPrefillPaged(3, 6, 0, false);
        let tape = lower_tq_layer(attention, 64);
        let mut want = vec![
            (KernelId::RopeAppend, None),
            (KernelId::TqQuantizeToPacked, Some(OnlyIfTurboquant)),
            (KernelId::TqQuantizeToPacked, Some(OnlyIfTurboquant)),
        ];
        let own: Vec<_> = gated_steps(&tape)
            .into_iter()
            .filter(|&(_, g)| g == Some(UnlessTurboquantDecode))
            .collect();
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
        // K then V staged, only K re-roping span blocks (cos_sin at slot 9).
        let staged: Vec<bool> = tape
            .commands
            .iter()
            .filter(|c| c.command.kernel == KernelId::TqStageRotated)
            .map(|c| {
                c.command
                    .bindings
                    .iter()
                    .any(|b| matches!(b, Binding::Weight { .. }))
            })
            .collect();
        assert_eq!(staged, [tp().rope_on_read, false]);

        let twin = |tape: &LoweredMetalTape| {
            tape.commands
                .iter()
                .find(|c| c.command.kernel == KernelId::AttentionViaCacheTq)
                .map(|c| c.command)
                .expect("Tq twin")
        };
        let decode_form = lower_tq_layer(Instruction::AttentionViaCache(3, 6, 0, false), 64);
        assert!(
            twin(&tape) == twin(&decode_form),
            "the twin is the decode kernel's form of the same attention"
        );
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
        let gemm = Instruction::Gemm(0, 1, 2, VOCAB, 0);
        update_shape_state(&tp(), &gemm, &mut width, &mut m_divisor);
        assert_eq!(
            width.get(),
            VOCAB,
            "Gemm output-N becomes the activation width"
        );

        let scalar_mul = Instruction::ScalarMul(2, 3, 0.0625);
        let mut scratch = 0u32;
        let mut moe_scratch = 0u32;
        let mut roped_k = 0u32;
        let mut attn_unfused = 0u32;
        let cmds = lower_one(
            &tp(),
            /*chunked=*/ false,
            &scalar_mul,
            0,
            BUCKET_M,
            0,
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
        let inst: Instruction = Instruction::AffineQmm(
            /*in_slot=*/ 7, /*out_slot=*/ 11, /*layer=*/ 3, /*n=*/ 2048,
            /*k=*/ 2048, /*group_size=*/ 64, /*bits=*/ 4, /*vector_limit=*/ 18,
        );
        let mut scratch = 0u32;
        let mut moe_scratch = 0u32;
        let mut roped_k = 0u32;
        let mut attn_unfused = 0u32;
        let cmds = lower_one(
            &tp(),
            /*chunked=*/ false,
            &inst,
            0,
            /*bucket_m=*/ 1,
            /*layer_offset=*/ 5,
            /*tape_index=*/ 0,
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
        match &cmd.bindings[0] {
            Binding::Weight {
                which,
                layer,
                binding_index,
                ..
            } => {
                assert_eq!(*which, WeightTensor::Weight);
                assert_eq!(layer.0, 8);
                assert_eq!(*binding_index, 0);
            }
            _ => panic!("bindings[0]: expected Weight"),
        }
        match &cmd.bindings[1] {
            Binding::Weight { which, .. } => assert_eq!(*which, WeightTensor::AffineScales),
            _ => panic!("bindings[1]: expected AffineScales Weight"),
        }
        match &cmd.bindings[2] {
            Binding::Weight { which, .. } => assert_eq!(*which, WeightTensor::AffineBiases),
            _ => panic!("bindings[2]: expected AffineBiases Weight"),
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
        let inst: Instruction = Instruction::AffineQmm(
            /*in_slot=*/ 7, /*out_slot=*/ 11, /*layer=*/ 3, /*n=*/ 2048,
            /*k=*/ 2048, /*group_size=*/ 64, /*bits=*/ 4, /*vector_limit=*/ 18,
        );
        let mut scratch = 0u32;
        let mut moe_scratch = 0u32;
        let mut roped_k = 0u32;
        let mut attn_unfused = 0u32;
        let cmds = lower_one(
            &tp(),
            /*chunked=*/ false,
            &inst,
            0,
            /*bucket_m=*/ 512,
            /*layer_offset=*/ 0,
            /*tape_index=*/ 0,
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
        let inst: Instruction = Instruction::AffineQmm(
            /*in_slot=*/ 7, /*out_slot=*/ 11, /*layer=*/ 3, /*n=*/ 2048,
            /*k=*/ 2048, /*group_size=*/ 64, /*bits=*/ 4, /*vector_limit=*/ 18,
        );
        let mut scratch = 0u32;
        let mut moe_scratch = 0u32;
        let mut roped_k = 0u32;
        let mut attn_unfused = 0u32;
        let cmds = lower_one(
            &tp(),
            /*chunked=*/ false,
            &inst,
            0,
            /*bucket_m=*/ 64,
            /*layer_offset=*/ 0,
            /*tape_index=*/ 0,
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
        let inst: Instruction = Instruction::SiluMul(
            /*gate=*/ 5, /*up=*/ 6, /*out=*/ 7, /*width=*/ 11008,
        );
        let mut scratch = 0u32;
        let mut moe_scratch = 0u32;
        let mut roped_k = 0u32;
        let mut attn_unfused = 0u32;
        let cmds = lower_one(
            &tp(),
            /*chunked=*/ false,
            &inst,
            0,
            /*bucket_m=*/ 64,
            /*layer_offset=*/ 0,
            /*tape_index=*/ 0,
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
        let inst: Instruction = Instruction::AffineQmm(
            7, 11, 0, /*n=*/ 2050, /*k=*/ 2048, /*group_size=*/ 64,
            /*bits=*/ 4, /*vector_limit=*/ 18,
        );
        let mut scratch = 0u32;
        let mut moe_scratch = 0u32;
        let mut roped_k = 0u32;
        let mut attn_unfused = 0u32;
        let cmds = lower_one(
            &tp(),
            /*chunked=*/ false,
            &inst,
            0,
            /*bucket_m=*/ 512,
            /*layer_offset=*/ 0,
            /*tape_index=*/ 0,
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
        let inst: Instruction = Instruction::AffineEmbed(
            /*out_slot=*/ 0, /*group_size=*/ 64, /*bits=*/ 4,
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
            0,
            bucket_m,
            /*layer_offset=*/ 5,
            /*tape_index=*/ 0,
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
        match &cmd.bindings[0] {
            Binding::Weight {
                kind: WeightBundleKind::AffineQuantEmbedding,
                which,
                layer,
                locator: _,
                binding_index,
            } => {
                assert_eq!(*which, WeightTensor::Weight);
                // Embed is unlayered — `layer = 0` regardless of layer_offset.
                assert_eq!(layer.0, 0);
                assert_eq!(*binding_index, 0);
            }
            _ => panic!("bindings[0]: expected AffineQuantEmbedding Weight"),
        }
        match &cmd.bindings[1] {
            Binding::Weight { which, .. } => assert_eq!(*which, WeightTensor::AffineScales),
            _ => panic!("bindings[1]: expected AffineScales"),
        }
        match &cmd.bindings[2] {
            Binding::Weight { which, .. } => assert_eq!(*which, WeightTensor::AffineBiases),
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

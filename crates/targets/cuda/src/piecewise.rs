// SPDX-License-Identifier: Apache-2.0
//! Piecewise CUDA-graph capture/replay for scratchy-forward-compiler at TP>1.
//!
//! Splits a forward tape (`backbone` + `lm_head`) at NCCL collective
//! boundaries. Each segment is a contiguous span of kernels with no
//! collectives in the middle, captured into its own CUgraph and replayed
//! back-to-back; collectives run eagerly between graph launches.
//!
//! Why: NCCL inside a monolithic CUDA graph fails with
//! `CUDA_ERROR_ILLEGAL_ADDRESS` on L40S sm_89 (verified empirically).
//! At tp=1 there are zero AllReduce/AllGather rows so monolithic
//! capture still works; this module is the tp>1 path.
//!
//! Design:
//! - `Loop(count, body_len)` is expanded once at capture so each
//!   layer iteration's pointer values bake into its own segment(s).
//! - `Instruction::AllReduce` is in-place. Capture ends BEFORE it;
//!   replay-side eager NCCL mutates the same captured address that
//!   the next segment reads.
//! - `Instruction::AllGather` is always the last op in the lm_head
//!   tape. The terminal segment ends just before it; the runner runs
//!   eager NCCL all_gather + rearrange post-replay and returns the
//!   resulting `OwnedTensor`. No new instruction split is needed.
//!
//! Tile-table state during capture is shared across segments — kernels
//! are dispatched normally, only the graph capture stream is
//! end/begin-cycled at NCCL boundaries.

#![cfg(feature = "cuda")]

use anyhow::Result;

use crate::CUgraphExec;
use crate::OwnedTensor;
use crate::alloc::CachingAllocator;
use crate::device::GpuDevice;
use crate::driver;
use crate::eval::{InstructionEval, InterpreterCtx};
use crate::forward_ctx::ForwardCtx;
use crate::tensor::GpuTensor;
use crate::tile_table::{TileEntry, tile_ref};
use scratchy_ir::{CanonicalParams, Instruction};

#[cfg(feature = "nccl")]
use crate::nccl::SuppressNcclGuard;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// One concrete instruction to dispatch — `Loop` already expanded so
/// each kernel runs with a single, baked-in `(layer_offset, op_idx,
/// bucket)`.
#[derive(Clone, Copy)]
pub struct ExpandedInstr {
    pub instr: Instruction,
    pub op_idx: u32,
    pub layer_offset: u32,
    pub bucket: u32,
}

/// What follows a captured graph segment.
#[derive(Clone, Copy)]
pub enum SegmentTerminator {
    /// In-place all-reduce on this captured-address tensor. Replay
    /// runs `NcclGroup::all_reduce_inplace_promote` between segments;
    /// the in-place mutation lands at the same address the next
    /// captured segment expects.
    AllReduce { tensor: GpuTensor },
    /// Last segment terminator at TP>1: this is the lm_head per-rank
    /// output (`[N, vocab/tp]`). Replay runs eager `all_gather_last_dim`
    /// and returns the resulting OwnedTensor.
    AllGatherFinal { in_tensor: GpuTensor },
    /// Last segment terminator at TP=1: the captured tensor at this
    /// address is the forward's logits. Replay memcpy_dtod's it into a
    /// fresh OwnedTensor for the caller.
    Terminal { out_tensor: GpuTensor },
}

pub struct CapturedSegment {
    pub exec: CUgraphExec,
    pub terminator: SegmentTerminator,
}

pub struct PiecewiseRunner {
    pub captured: Vec<CapturedSegment>,
}

// `CUgraphExec` is a raw pointer; `GpuTensor` is `Copy` over raw ptrs.
// Both stay valid for the lifetime of the private allocator pool the
// runner pinned, which lives as long as the runner does.
unsafe impl Send for PiecewiseRunner {}
unsafe impl Sync for PiecewiseRunner {}

impl Drop for PiecewiseRunner {
    fn drop(&mut self) {
        for seg in &self.captured {
            unsafe {
                let _ = driver::graph_exec_destroy(seg.exec);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tape segmentation (Loop expansion)
// ---------------------------------------------------------------------------

/// Expand `Loop` instructions and produce a flat sequence of
/// `ExpandedInstr` for the given slice.
pub fn expand_slice(slice: &[Instruction], bucket: u32, out: &mut Vec<ExpandedInstr>) {
    let mut i = 0usize;
    while i < slice.len() {
        match slice[i] {
            Instruction::Loop(count, body_len, layer_stride) => {
                let body_start = i + 1;
                let body_end = body_start + body_len as usize;
                let body = &slice[body_start..body_end];
                for l in 0..count {
                    for (j, instr) in body.iter().enumerate() {
                        out.push(ExpandedInstr {
                            instr: *instr,
                            op_idx: (body_start + j) as u32,
                            // Scaled by the stride — see `eval.rs`'s Loop arm.
                            layer_offset: l * layer_stride,
                            bucket,
                        });
                    }
                }
                i = body_end;
            }
            instr => {
                out.push(ExpandedInstr {
                    instr,
                    op_idx: i as u32,
                    layer_offset: 0,
                    bucket,
                });
                i += 1;
            }
        }
    }
}

/// Expand backbone + lm_head into a single flat instruction list.
pub fn expand_tape(
    backbone: &[Instruction],
    backbone_bucket: u32,
    lm_head: &[Instruction],
    lm_head_bucket: u32,
) -> Vec<ExpandedInstr> {
    let mut out = Vec::with_capacity(backbone.len() + lm_head.len());
    expand_slice(backbone, backbone_bucket, &mut out);
    expand_slice(lm_head, lm_head_bucket, &mut out);
    out
}

/// Count how many graph segments a tape will produce. Used by the
/// caller to log expected segment counts during capture. Safe to call
/// without a CUDA context.
pub fn segment_count(
    backbone: &[Instruction],
    backbone_bucket: u32,
    lm_head: &[Instruction],
    lm_head_bucket: u32,
) -> usize {
    let expanded = expand_tape(backbone, backbone_bucket, lm_head, lm_head_bucket);
    let mut segments: usize = 0;
    let mut had_instr_in_current = false;
    for ex in &expanded {
        match ex.instr {
            #[cfg(feature = "nccl")]
            Instruction::AllReduce(_) => {
                segments += 1;
                had_instr_in_current = false;
            }
            #[cfg(feature = "nccl")]
            Instruction::AllGather(_, _) => {
                segments += 1;
                had_instr_in_current = false;
            }
            _ => {
                had_instr_in_current = true;
            }
        }
    }
    if had_instr_in_current || segments == 0 {
        segments += 1;
    }
    segments
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

/// Capture the tape into a sequence of CUDA graphs split at NCCL
/// boundaries.
///
/// # Caller contract
/// The caller MUST have called `device.caching.begin_allocate_to_pool()`
/// before this fn so private-pool addresses are stable across capture
/// and replay. The pool stays live for the runner's lifetime; call
/// `end_allocate_to_pool()` after the LAST runner is captured.
///
/// # Safety
/// All `ForwardCtx` tensors valid; `device.compute_stream` live; the
/// FORWARD_TABLE bucket entry's `terminal_slot` matches the tape.
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_piecewise_capture<W: CanonicalParams>(
    backbone: &[Instruction],
    backbone_bucket: u32,
    lm_head: &[Instruction],
    lm_head_bucket: u32,
    wm: &W,
    fwd: &ForwardCtx,
    device: &mut GpuDevice,
    num_slots: u32,
    terminal_slot: u32,
) -> Result<PiecewiseRunner> {
    let expanded = expand_tape(backbone, backbone_bucket, lm_head, lm_head_bucket);
    let stream = device.compute_stream;
    let mut tiles: Vec<Option<TileEntry>> = (0..num_slots).map(|_| None).collect();
    let mut pinned_owned: Vec<OwnedTensor> = Vec::new();
    let mut captured: Vec<CapturedSegment> = Vec::new();

    // Suppress NCCL collectives so the in-place AllReduce arm becomes a
    // no-op when (rare) someone calls eval on a collective during
    // capture. We don't call eval on AllReduce/AllGather ourselves —
    // we end_capture / begin_capture around them — but the guard is
    // cheap insurance.
    #[cfg(feature = "nccl")]
    let _suppress = SuppressNcclGuard::new();

    // Open the first segment.
    unsafe { driver::stream_begin_capture(stream)? };
    // `mut` is required at `--features nccl` (the AllReduce/AllGather
    // arms below reassign it); without nccl those arms cfg-out and
    // `capture_open` stays `true` the whole loop, so clippy flags the
    // unused-mut. Allow it locally so both build configurations stay
    // clean under `-D warnings`.
    #[allow(unused_mut)]
    let mut capture_open = true;

    for ex in &expanded {
        match ex.instr {
            #[cfg(feature = "nccl")]
            Instruction::AllReduce(slot) => {
                let snap = tile_ref(&tiles, slot).as_gpu_tensor(&tiles);
                let exec = unsafe {
                    let g = driver::stream_end_capture(stream)?;
                    let exec = driver::graph_instantiate(g)?;
                    driver::graph_destroy(g)?;
                    exec
                };
                captured.push(CapturedSegment {
                    exec,
                    terminator: SegmentTerminator::AllReduce { tensor: snap },
                });
                // Open the next segment.
                unsafe { driver::stream_begin_capture(stream)? };
                capture_open = true;
            }
            #[cfg(feature = "nccl")]
            Instruction::AllGather(in_slot, _out_slot) => {
                let snap = tile_ref(&tiles, in_slot).as_gpu_tensor(&tiles);
                let exec = unsafe {
                    let g = driver::stream_end_capture(stream)?;
                    let exec = driver::graph_instantiate(g)?;
                    driver::graph_destroy(g)?;
                    exec
                };
                captured.push(CapturedSegment {
                    exec,
                    terminator: SegmentTerminator::AllGatherFinal { in_tensor: snap },
                });
                capture_open = false;
                // AllGather is always the last op — break out.
                break;
            }
            _ => {
                let mut ctx = InterpreterCtx {
                    wm,
                    tiles: &mut tiles,
                    fwd,
                    device,
                    layer_offset: ex.layer_offset,
                    kv_layer: ex.layer_offset,
                    pinned_owned: std::mem::take(&mut pinned_owned),
                };
                unsafe { ex.instr.eval(&mut ctx, ex.bucket, ex.op_idx) };
                pinned_owned = std::mem::take(&mut ctx.pinned_owned);
            }
        }
    }

    if capture_open {
        let snap = tile_ref(&tiles, terminal_slot).as_gpu_tensor(&tiles);
        let exec = unsafe {
            let g = driver::stream_end_capture(stream)?;
            let exec = driver::graph_instantiate(g)?;
            driver::graph_destroy(g)?;
            exec
        };
        captured.push(CapturedSegment {
            exec,
            terminator: SegmentTerminator::Terminal { out_tensor: snap },
        });
    }

    // The captured tile_table state (Owned tensors backed by the
    // private allocator pool) drops here. Their addresses live on as
    // long as the pool does — the caller is responsible for keeping
    // the pool alive (i.e. NOT calling `release_pool`) for the
    // lifetime of the runner.
    drop(tiles);
    drop(pinned_owned);

    Ok(PiecewiseRunner { captured })
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

/// Replay the captured segments. Between segments runs eager NCCL
/// collectives. Returns the forward's output OwnedTensor.
///
/// # Safety
/// All ForwardCtx tensors at the same shapes/dtypes as capture-time;
/// `device` matches the runner's capture device; tp_group is set when
/// any segment terminator is AllReduce or AllGatherFinal.
// `fwd` is read only inside the `#[cfg(feature = "nccl")]` arms below
// (AllReduce/AllGather terminators that pull `tp_group` from ForwardCtx);
// without nccl those arms cfg-out and `fwd` becomes unused. Allow at the
// fn level so both configurations stay clean under `-D warnings`.
#[allow(unused_variables)]
pub unsafe fn run_piecewise_replay(
    runner: &PiecewiseRunner,
    fwd: &ForwardCtx,
    device: &mut GpuDevice,
) -> OwnedTensor {
    let stream = device.compute_stream;
    let mut output: Option<OwnedTensor> = None;

    for seg in &runner.captured {
        unsafe {
            driver::graph_launch(seg.exec, stream).expect("piecewise: graph_launch failed");
        }
        match seg.terminator {
            #[cfg(feature = "nccl")]
            SegmentTerminator::AllReduce { tensor } => {
                let group = fwd.tp_group.expect(
                    "piecewise replay: AllReduce terminator but ForwardCtx::tp_group is None",
                );
                // Use the non-promote (bf16-native) variant to avoid the
                // transient fp32 buffer alloc — that alloc would land in
                // the same private pool the captured graphs draw from,
                // and a freshly-recycled block can alias a captured slot
                // whose contents the NEXT segment then reads as garbage.
                // ~1 ULP per-reduce drift on small models is acceptable
                // for the smoke run; precision-promoted variant requires
                // a pre-allocated dedicated fp32 buffer (TODO).
                unsafe {
                    group
                        .all_reduce_inplace(tensor)
                        .expect("piecewise replay: NCCL all_reduce_inplace failed");
                }
            }
            #[cfg(feature = "nccl")]
            SegmentTerminator::AllGatherFinal { in_tensor } => {
                let group = fwd.tp_group.expect(
                    "piecewise replay: AllGather terminator but ForwardCtx::tp_group is None",
                );
                let out = unsafe { group.all_gather_last_dim(in_tensor, &mut device.caching) };
                output = Some(out);
            }
            SegmentTerminator::Terminal { out_tensor } => {
                output =
                    Some(unsafe { copy_to_fresh_owned(out_tensor, &mut device.caching, stream) });
            }
            // When nccl is disabled, the AllReduce / AllGatherFinal
            // arms above are removed by cfg; the segmenter doesn't
            // emit those terminators in that build. The Terminal arm
            // covers every captured run.
            #[cfg(not(feature = "nccl"))]
            _ => unreachable!("non-Terminal terminator at nccl-disabled build"),
        }
    }

    output.expect("piecewise replay produced no output (zero segments captured?)")
}

/// Memcpy the contents of a captured-address tensor into a fresh
/// OwnedTensor allocated from the (regular, NOT private-pool) caching
/// allocator. Used at replay time to hand the caller an OwnedTensor
/// they can free independently of the runner's private pool.
unsafe fn copy_to_fresh_owned(
    src: GpuTensor,
    alloc: &mut CachingAllocator,
    stream: crate::CUstream,
) -> OwnedTensor {
    let shape: Vec<usize> = (0..src.ndim()).map(|d| src.dim(d)).collect();
    let out = alloc.alloc_tensor(&shape, src.dtype());
    let bytes = src.size_bytes();
    unsafe {
        driver::memcpy_dtod_async(out.raw_ptr(), src.raw_ptr() as *const u8, bytes, stream)
            .expect("piecewise replay: memcpy_dtod_async of terminal tensor failed");
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_loop(body: Vec<Instruction>, count: u32) -> Vec<Instruction> {
        let mut v = vec![Instruction::Loop(count, body.len() as u32, 1)];
        v.extend(body);
        v
    }

    #[test]
    fn expand_loop_unrolls_body_count_times() {
        let body = vec![
            Instruction::Embed(0),
            Instruction::Add(1, 2),
            Instruction::Free(0),
        ];
        let tape = dummy_loop(body, 4);
        let mut out = Vec::new();
        expand_slice(&tape, 7, &mut out);
        assert_eq!(out.len(), 12, "4 iters × 3 body ops = 12");
        // Layer offset cycles 0,0,0,1,1,1,2,2,2,3,3,3.
        for i in 0..12usize {
            assert_eq!(out[i].layer_offset, (i / 3) as u32);
            assert_eq!(out[i].bucket, 7);
        }
        // op_idx is the absolute position in the bucket, body_start = 1.
        assert_eq!(out[0].op_idx, 1);
        assert_eq!(out[1].op_idx, 2);
        assert_eq!(out[2].op_idx, 3);
        assert_eq!(out[3].op_idx, 1, "second iter reuses op_idx 1");
    }

    #[test]
    fn expand_handles_no_loop() {
        let tape = vec![Instruction::Embed(0), Instruction::Add(1, 2)];
        let mut out = Vec::new();
        expand_slice(&tape, 0, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].layer_offset, 0);
        assert_eq!(out[0].op_idx, 0);
        assert_eq!(out[1].op_idx, 1);
    }

    #[cfg(feature = "nccl")]
    #[test]
    fn segment_count_splits_at_allreduce() {
        // Tape: [Embed, AllReduce, Add, AllGather]
        let backbone = vec![Instruction::Embed(0), Instruction::AllReduce(0)];
        let lm_head = vec![Instruction::Add(1, 2), Instruction::AllGather(2, 3)];
        // Expected: segment 1 (Embed), segment 2 (Add), final = 2 segments.
        // (AllReduce/AllGather are terminators, not segments themselves.)
        assert_eq!(segment_count(&backbone, 0, &lm_head, 1), 2);
    }

    #[cfg(feature = "nccl")]
    #[test]
    fn segment_count_with_loop_expansion() {
        // Loop with 3 layers, each containing one AllReduce. Expanded:
        // 3 segments inside the loop + 1 final = 4 segments.
        let body = vec![Instruction::Embed(0), Instruction::AllReduce(0)];
        let backbone = dummy_loop(body, 3);
        let lm_head = vec![Instruction::Add(1, 2)];
        assert_eq!(segment_count(&backbone, 0, &lm_head, 1), 4);
    }

    #[test]
    fn segment_count_no_collectives_one_segment() {
        // No AllReduce/AllGather → single Terminal segment.
        let backbone = vec![Instruction::Embed(0), Instruction::Add(1, 2)];
        let lm_head = vec![];
        assert_eq!(segment_count(&backbone, 0, &lm_head, 1), 1);
    }
}

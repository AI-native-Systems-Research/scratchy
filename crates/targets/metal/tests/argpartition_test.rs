// SPDX-License-Identifier: Apache-2.0
//! Argsort kernel parity test — compare against a CPU stable sort
//! (ties broken by index) for the MoE router shapes Mixtral (E=8),
//! Qwen2-MoE (E=60), Qwen3-MoE (E=128), Qwen3.5-MoE (E=256, the
//! bn=64 / N_PER_BLOCK=256 instantiation). Dispatches on the
//! production MTL4 path (see `common::dispatch_threadgroups`).
//!
//! Top-k semantics: the trailing `top_k` indices of the sorted-
//! ascending output are the indices of the top-k entries by value.

mod common;

use half::{bf16, f16};
use objc2_metal::MTLSize;
use scratchy_target_metal::argpartition::{ArgsortDType, ArgsortKernels, pick_pipeline_shape};
use scratchy_target_metal::device::detect_device;

fn fill_random_f32(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut out = vec![0.0_f32; rows * cols];
    for slot in &mut out {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let bits = (state as u32) & 0x00FF_FFFF;
        *slot = (bits as f32 / (1u32 << 23) as f32) * 4.0 - 2.0;
    }
    out
}

fn cpu_argsort_ascending_f32(input: &[f32], rows: usize, cols: usize) -> Vec<u32> {
    let mut out = vec![0u32; rows * cols];
    for r in 0..rows {
        let mut idx: Vec<usize> = (0..cols).collect();
        let base = r * cols;
        // Stable sort ascending; ties broken by index (matches the
        // MLX block_sort stable-equality behavior).
        idx.sort_by(|&a, &b| {
            input[base + a]
                .partial_cmp(&input[base + b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, &j) in idx.iter().enumerate() {
            out[base + i] = j as u32;
        }
    }
    out
}

/// Argsort kernel binding contract (matches the production
/// `ArgPartitionTopK` lowering): `buffer(0)=input`, `buffer(1)=output`
/// (u32), `buffer(2)=axis` (=axis_size), `buffer(3)=one` (=1),
/// `buffer(4)=one` (=1), `buffer(5)=stride_in` (=axis_size),
/// `buffer(6)=stride_out` (=axis_size). The five trailing scalars are
/// `device const int&` in the kernel; their values are non-negative so
/// the `shared_u32` bit pattern is identical to the classic path's
/// `setBytes` of an `i32`. Grid is one threadgroup per row
/// (`(1, rows, 1)`); the block is `bn` threads wide, where `(bn, tn)`
/// is the `pick_pipeline_shape` choice for `axis_size`.
fn run_argsort_mtl4(
    dtype: ArgsortDType,
    in_buf: &common::Buffer,
    out_buf: &common::Buffer,
    rows: u32,
    axis_size: u32,
) -> bool {
    let device = detect_device()
        .expect("Metal 4 GPU present (caller pre-guards)")
        .device;
    let (bn, tn) = pick_pipeline_shape(axis_size as usize).expect("pick pipeline shape");
    let kernels = ArgsortKernels::new(&device).expect("argsort kernels");
    let pipeline = kernels
        .pipeline_for(dtype, bn, tn)
        .expect("argsort pipeline");

    let axis_buf = common::shared_u32(&device, axis_size);
    let one_a = common::shared_u32(&device, 1);
    let one_b = common::shared_u32(&device, 1);
    let stride_in = common::shared_u32(&device, axis_size);
    let stride_out = common::shared_u32(&device, axis_size);

    common::dispatch_threadgroups(
        &device,
        pipeline,
        &[
            in_buf,
            out_buf,
            &axis_buf,
            &one_a,
            &one_b,
            &stride_in,
            &stride_out,
        ],
        MTLSize {
            width: 1,
            height: rows as usize,
            depth: 1,
        },
        MTLSize {
            width: bn,
            height: 1,
            depth: 1,
        },
    )
}

/// One parity case at any router dtype. Host values are generated
/// as f32; for the half dtypes the GPU input is the cast values and
/// the CPU reference sorts the SAME rounded values (so reference and
/// kernel see identical keys).
fn run_case(dtype: ArgsortDType, rows: usize, cols: usize, top_k: usize, seed: u64) {
    let Some(__dev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = __dev.device;

    let host = fill_random_f32(rows, cols, seed);
    // (gpu input bytes, the values the CPU reference must sort)
    let (in_host_bytes, cmp_host): (Vec<u8>, Vec<f32>) = match dtype {
        ArgsortDType::F32 => {
            let mut bytes = vec![0u8; host.len() * 4];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    host.as_ptr() as *const u8,
                    bytes.as_mut_ptr(),
                    bytes.len(),
                );
            }
            (bytes, host.clone())
        }
        ArgsortDType::F16 => {
            let cast: Vec<f16> = host.iter().map(|&v| f16::from_f32(v)).collect();
            let mut bytes = vec![0u8; cast.len() * 2];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    cast.as_ptr() as *const u8,
                    bytes.as_mut_ptr(),
                    bytes.len(),
                );
            }
            (bytes, cast.iter().map(|v| v.to_f32()).collect())
        }
        ArgsortDType::Bf16 => {
            let cast: Vec<bf16> = host.iter().map(|&v| bf16::from_f32(v)).collect();
            let mut bytes = vec![0u8; cast.len() * 2];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    cast.as_ptr() as *const u8,
                    bytes.as_mut_ptr(),
                    bytes.len(),
                );
            }
            (bytes, cast.iter().map(|v| v.to_f32()).collect())
        }
        other => unreachable!("router-prob parity cases are float-typed, got {other:?}"),
    };

    let in_buf = common::shared_bytes(&device, &in_host_bytes);
    let out_bytes = rows * cols * 4;
    let out_buf = common::shared_zeroed(&device, out_bytes);

    if !run_argsort_mtl4(dtype, &in_buf, &out_buf, rows as u32, cols as u32) {
        return;
    }

    let got: Vec<u32> = common::read_slice(&out_buf, rows * cols);

    let want = cpu_argsort_ascending_f32(&cmp_host, rows, cols);

    // Compare trailing top-k slot by slot: those should be exactly
    // the top-k indices (stable order). The interior of the sort
    // can disagree on equal-value runs, but we don't care about
    // those for MoE.
    for r in 0..rows {
        let base = r * cols;
        let mut got_topk: Vec<u32> = got[base + cols - top_k..base + cols].to_vec();
        let mut want_topk: Vec<u32> = want[base + cols - top_k..base + cols].to_vec();
        // Stable sort to make comparison set-wise; the relative
        // order within the top-k can legitimately differ from the
        // CPU stable sort for ties.
        got_topk.sort();
        want_topk.sort();
        assert_eq!(
            got_topk, want_topk,
            "row {r}: {dtype:?} top-{top_k} indices differ \
             (got={got_topk:?}, want={want_topk:?})"
        );
    }
}

#[test]
fn argsort_f32_mixtral_topk() {
    // E=8, top_k=2 (Mixtral default).
    run_case(
        ArgsortDType::F32,
        /*rows=*/ 11,
        /*cols=*/ 8,
        /*top_k=*/ 2,
        0xC0FFEE,
    );
}

#[test]
fn argsort_f32_qwen2_moe_topk() {
    run_case(
        ArgsortDType::F32,
        /*rows=*/ 7,
        /*cols=*/ 60,
        /*top_k=*/ 4,
        0xBEEF_F00D,
    );
}

#[test]
fn argsort_f32_qwen3_moe_topk() {
    run_case(
        ArgsortDType::F32,
        /*rows=*/ 13,
        /*cols=*/ 128,
        /*top_k=*/ 8,
        0xDEAD_BEEF,
    );
}

#[test]
fn argsort_f16_qwen3_moe_topk() {
    // f16 router path at E=128.
    run_case(
        ArgsortDType::F16,
        /*rows=*/ 13,
        /*cols=*/ 128,
        /*top_k=*/ 8,
        0x5EED,
    );
}

#[test]
fn argsort_f32_qwen3_5_moe_topk_bn64() {
    // E=256, top_k=8 (Qwen3.5-MoE-35B-A3B) — exercises the bn=64
    // (N_PER_BLOCK=256) instantiation via pick_pipeline_shape.
    run_case(
        ArgsortDType::F32,
        /*rows=*/ 9,
        /*cols=*/ 256,
        /*top_k=*/ 8,
        0xFEED_FACE,
    );
}

#[test]
fn argsort_f16_qwen3_5_moe_topk_bn64() {
    // E=256 f16 — the W::METAL_DTYPE=F16 router path at bn=64.
    run_case(
        ArgsortDType::F16,
        /*rows=*/ 9,
        /*cols=*/ 256,
        /*top_k=*/ 8,
        0x0ACE_0F16,
    );
}

#[test]
fn argsort_bf16_qwen3_5_moe_topk_bn64() {
    // E=256 bf16 — the W::METAL_DTYPE=Bf16 router path at bn=64
    // (Qwen3.x MLX checkpoints carry bf16 scales → bf16 router probs).
    run_case(
        ArgsortDType::Bf16,
        /*rows=*/ 9,
        /*cols=*/ 256,
        /*top_k=*/ 8,
        0xBF16_CAFE,
    );
}

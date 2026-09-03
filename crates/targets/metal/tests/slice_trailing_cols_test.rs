// SPDX-License-Identifier: Apache-2.0
//! Parity test for `slice_trailing_cols_u32`, dispatched on the
//! production MTL4 path (see `common::dispatch_threadgroups`).

mod common;

use objc2_metal::MTLSize;
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::slice_trailing_cols::SliceTrailingColsKernels;

/// `slice_trailing_cols_u32` binding contract: buffer(0)=src,
/// buffer(1)=dst, buffer(2)=axis_size (i32), buffer(3)=top_k (i32).
/// One thread per (k, n): grid (top_k, rows). `top_k <= 32` for every
/// MoE-router shape here, so a single threadgroup of `top_k` threads
/// covers the k axis exactly (no over-dispatch — the kernel has no
/// bounds guard) and there is one threadgroup per row.
fn run_slice_trailing_cols_mtl4(
    src: &common::Buffer,
    dst: &common::Buffer,
    rows: u32,
    axis: u32,
    top_k: u32,
) -> bool {
    let device = detect_device()
        .expect("Metal 4 GPU present (caller pre-guards)")
        .device;
    let axis_buf = common::shared_u32(&device, axis);
    let top_k_buf = common::shared_u32(&device, top_k);
    let kernels = SliceTrailingColsKernels::new(&device).expect("slice_trailing_cols kernels");
    let threads_w = top_k.min(32) as usize;
    common::dispatch_threadgroups(
        &device,
        &kernels.u32_pipeline,
        &[src, dst, &axis_buf, &top_k_buf],
        MTLSize {
            width: (top_k as usize).div_ceil(threads_w),
            height: rows as usize,
            depth: 1,
        },
        MTLSize {
            width: threads_w,
            height: 1,
            depth: 1,
        },
    )
}

fn run_case(rows: usize, axis: usize, top_k: usize) {
    let Some(__dev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = __dev.device;

    let mut host = vec![0u32; rows * axis];
    for r in 0..rows {
        for c in 0..axis {
            host[r * axis + c] = (r * 1000 + c) as u32;
        }
    }

    let src = common::shared_slice(&device, &host);
    let dst = common::shared_zeroed(&device, rows * top_k * std::mem::size_of::<u32>());

    if !run_slice_trailing_cols_mtl4(&src, &dst, rows as u32, axis as u32, top_k as u32) {
        return;
    }

    let got: Vec<u32> = common::read_slice(&dst, rows * top_k);

    for r in 0..rows {
        for k in 0..top_k {
            let want = (r * 1000 + (axis - top_k) + k) as u32;
            assert_eq!(got[r * top_k + k], want, "row {r} k {k}");
        }
    }
}

#[test]
fn slice_trailing_cols_u32_qwen3_moe_top8() {
    run_case(5, 128, 8);
}

#[test]
fn slice_trailing_cols_u32_qwen3_5_moe_top8_e256() {
    // axis=256 (Qwen3.5-MoE router width).
    run_case(9, 256, 8);
}

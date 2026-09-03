// SPDX-License-Identifier: Apache-2.0
//! take_along_axis_2d_contig parity test for the MoE router pattern.
//! Dispatches on the production MTL4 path (see
//! `common::dispatch_threadgroups`).

mod common;

use half::bf16;
use objc2_metal::MTLSize;
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::take_along_axis::{TakeAlongAxisKernels, TakeAlongDType};

/// take_along_axis binding contract (`buffer(0)=src, buffer(1)=indices,
/// buffer(2)=out, buffer(3)=src_axis_size, buffer(4)=idx_axis_size`):
/// one thread per (k, n). The kernel bounds-checks against
/// `threads_per_grid`, so the threadgroup count is sized to reproduce the
/// classic `dispatchThreads` extent `(idx_axis_size, rows, 1)` exactly.
fn run_take_along_axis_mtl4(
    dtype: TakeAlongDType,
    src_buf: &common::Buffer,
    idx_buf: &common::Buffer,
    out_buf: &common::Buffer,
    rows: u32,
    src_axis_size: u32,
    idx_axis_size: u32,
) -> bool {
    let device = detect_device()
        .expect("Metal 4 GPU present (caller pre-guards)")
        .device;
    // The classic path passed these as `setBytes` i32 scalars at
    // buffer(3)/buffer(4); for the positive axis sizes here a u32 has the
    // identical byte layout the kernel reads as `int`.
    let src_axis_buf = common::shared_u32(&device, src_axis_size);
    let idx_axis_buf = common::shared_u32(&device, idx_axis_size);
    let kernels = TakeAlongAxisKernels::new(&device).expect("take_along_axis kernels");

    let tg_width = idx_axis_size.min(32) as usize;
    let threadgroups = MTLSize {
        width: (idx_axis_size as usize).div_ceil(tg_width),
        height: rows as usize,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    common::dispatch_threadgroups(
        &device,
        kernels.pipeline_for(dtype),
        &[src_buf, idx_buf, out_buf, &src_axis_buf, &idx_axis_buf],
        threadgroups,
        threads_per_tg,
    )
}

#[test]
fn take_along_axis_bf16_qwen3_moe_topk() {
    // gates [N=5, E=128] bf16; inds [N=5, top_k=8] u32.
    let Some(__dev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = __dev.device;

    let rows = 5usize;
    let e = 128usize;
    let top_k = 8usize;

    let mut state = 0xDEAD_BEEF_DEAD_BEEFu64;
    let mut gates_f32 = vec![0.0_f32; rows * e];
    for slot in &mut gates_f32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *slot = ((state as u32) & 0xFFFF) as f32 / 65535.0;
    }
    let gates_bf16: Vec<bf16> = gates_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    // Choose indices: top-8 by gate value per row.
    let mut indices = vec![0u32; rows * top_k];
    for r in 0..rows {
        let mut idx: Vec<usize> = (0..e).collect();
        idx.sort_by(|&a, &b| {
            gates_f32[r * e + b]
                .partial_cmp(&gates_f32[r * e + a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for k in 0..top_k {
            indices[r * top_k + k] = idx[k] as u32;
        }
    }

    let src_buf = common::shared_slice(&device, &gates_bf16);
    let idx_buf = common::shared_slice(&device, &indices);
    let out_buf = common::shared_zeroed(&device, rows * top_k * std::mem::size_of::<bf16>());

    if !run_take_along_axis_mtl4(
        TakeAlongDType::BF16,
        &src_buf,
        &idx_buf,
        &out_buf,
        rows as u32,
        e as u32,
        top_k as u32,
    ) {
        return;
    }

    let got: Vec<bf16> = common::read_slice(&out_buf, rows * top_k);

    for r in 0..rows {
        for k in 0..top_k {
            let want = gates_bf16[r * e + indices[r * top_k + k] as usize];
            assert_eq!(
                got[r * top_k + k].to_bits(),
                want.to_bits(),
                "row {r} k {k}"
            );
        }
    }
}

#[test]
fn take_along_axis_bf16_qwen3_5_moe_topk_e256() {
    // gates [N=9, E=256] bf16; inds [N=9, top_k=8] u32 (Qwen3.5-MoE).
    let Some(__dev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = __dev.device;

    let rows = 9usize;
    let e = 256usize;
    let top_k = 8usize;

    let mut state = 0xFEED_FACE_DEAD_BEEFu64;
    let mut gates_f32 = vec![0.0_f32; rows * e];
    for slot in &mut gates_f32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *slot = ((state as u32) & 0xFFFF) as f32 / 65535.0;
    }
    let gates_bf16: Vec<bf16> = gates_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let mut indices = vec![0u32; rows * top_k];
    for r in 0..rows {
        let mut idx: Vec<usize> = (0..e).collect();
        idx.sort_by(|&a, &b| {
            gates_f32[r * e + b]
                .partial_cmp(&gates_f32[r * e + a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for k in 0..top_k {
            indices[r * top_k + k] = idx[k] as u32;
        }
    }

    let src_buf = common::shared_slice(&device, &gates_bf16);
    let idx_buf = common::shared_slice(&device, &indices);
    let out_buf = common::shared_zeroed(&device, rows * top_k * std::mem::size_of::<bf16>());

    if !run_take_along_axis_mtl4(
        TakeAlongDType::BF16,
        &src_buf,
        &idx_buf,
        &out_buf,
        rows as u32,
        e as u32,
        top_k as u32,
    ) {
        return;
    }

    let got: Vec<bf16> = common::read_slice(&out_buf, rows * top_k);

    for r in 0..rows {
        for k in 0..top_k {
            let want = gates_bf16[r * e + indices[r * top_k + k] as usize];
            assert_eq!(
                got[r * top_k + k].to_bits(),
                want.to_bits(),
                "row {r} k {k}"
            );
        }
    }
}

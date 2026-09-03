// SPDX-License-Identifier: Apache-2.0
//! Parity test for the precise softmax kernel against a CPU
//! reference, exercising the MoE-router shapes from Mixtral (E=8),
//! Qwen2-MoE (E=60), Qwen3-MoE (E=128). Dispatches on the production
//! MTL4 path (see `common::dispatch_threadgroups`).

mod common;

use half::{bf16, f16};
use objc2_metal::MTLSize;
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::softmax::{
    SOFTMAX_BLOCK_THREADS, SoftmaxDType, SoftmaxKernels, softmax_cpu_f32,
};

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

/// Softmax kernel binding contract (`buffer(0)=input, buffer(1)=output,
/// buffer(2)=axis_size`); grid is one threadgroup per row.
fn run_softmax_mtl4(
    dtype: SoftmaxDType,
    in_buf: &common::Buffer,
    out_buf: &common::Buffer,
    rows: u32,
    cols: u32,
) -> bool {
    let device = detect_device()
        .expect("Metal 4 GPU present (caller pre-guards)")
        .device;
    let axis_buf = common::shared_u32(&device, cols);
    let kernels = SoftmaxKernels::new(&device).expect("softmax kernels");
    common::dispatch_threadgroups(
        &device,
        kernels.pipeline_for(dtype),
        &[in_buf, out_buf, &axis_buf],
        MTLSize {
            width: rows as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: SOFTMAX_BLOCK_THREADS,
            height: 1,
            depth: 1,
        },
    )
}

fn run_case_bf16(rows: usize, cols: usize, seed: u64) {
    let Some(__dev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = __dev.device;
    let host_f32 = fill_random_f32(rows, cols, seed);
    let host_bf16: Vec<bf16> = host_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let in_buf = common::shared_slice(&device, &host_bf16);
    let out_buf = common::shared_zeroed(&device, host_bf16.len() * std::mem::size_of::<bf16>());

    if !run_softmax_mtl4(
        SoftmaxDType::BF16,
        &in_buf,
        &out_buf,
        rows as u32,
        cols as u32,
    ) {
        return;
    }

    let got_bf16: Vec<bf16> = common::read_slice(&out_buf, rows * cols);
    let mut want = vec![0.0_f32; rows * cols];
    softmax_cpu_f32(&host_f32, rows, cols, &mut want);

    let mut max_err = 0.0_f32;
    for (i, (&got, &expected)) in got_bf16.iter().zip(want.iter()).enumerate() {
        let g = got.to_f32();
        let err = (g - expected).abs();
        if err > max_err {
            max_err = err;
        }
        // bf16 carries ~3 decimal digits; +/- ulp at the typical
        // probability magnitudes lands around 8e-3 in the worst row.
        assert!(
            err < 1e-2,
            "row{}_col{}: got {} want {} err {}",
            i / cols,
            i % cols,
            g,
            expected,
            err
        );
    }
    eprintln!(
        "softmax_bf16 rows={rows} cols={cols} max_abs_err={:.2e}",
        max_err
    );
}

fn run_case_f16(rows: usize, cols: usize, seed: u64) {
    let Some(__dev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = __dev.device;
    let host_f32 = fill_random_f32(rows, cols, seed);
    let host_f16: Vec<f16> = host_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let in_buf = common::shared_slice(&device, &host_f16);
    let out_buf = common::shared_zeroed(&device, host_f16.len() * std::mem::size_of::<f16>());

    if !run_softmax_mtl4(
        SoftmaxDType::F16,
        &in_buf,
        &out_buf,
        rows as u32,
        cols as u32,
    ) {
        return;
    }

    let got: Vec<f16> = common::read_slice(&out_buf, rows * cols);
    let mut want = vec![0.0_f32; rows * cols];
    softmax_cpu_f32(&host_f32, rows, cols, &mut want);

    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let err = (g.to_f32() - w).abs();
        assert!(
            err < 1e-3,
            "f16 row{}_col{}: got {} want {} err {}",
            i / cols,
            i % cols,
            g.to_f32(),
            w,
            err
        );
    }
}

#[test]
fn softmax_bf16_mixtral_shape() {
    run_case_bf16(17, 8, 0xC0FFEE);
}

#[test]
fn softmax_bf16_qwen2_moe_shape() {
    run_case_bf16(5, 60, 0xBEEF_F00D);
}

#[test]
fn softmax_bf16_qwen3_moe_shape() {
    run_case_bf16(33, 128, 0xDEAD_BEEF);
}

#[test]
fn softmax_f16_mixtral_shape() {
    run_case_f16(7, 8, 0xCAFE);
}

#[test]
fn softmax_f16_qwen3_moe_shape() {
    run_case_f16(11, 128, 0x1234_5678);
}

#[test]
fn softmax_bf16_qwen3_5_moe_shape() {
    // E=256 (Qwen3.5-MoE-35B-A3B router width).
    run_case_bf16(9, 256, 0xFEED_FACE);
}

#[test]
fn softmax_f16_qwen3_5_moe_shape() {
    run_case_f16(9, 256, 0xACE0_CAFE);
}

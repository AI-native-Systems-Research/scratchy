// SPDX-License-Identifier: Apache-2.0
//! Parity test for the precise softmax kernel and the top-k renormalize
//! against a CPU reference, exercising the MoE-router shapes from Mixtral
//! (E=8), Qwen2-MoE (E=60), Qwen3-MoE (E=128). Dispatches on the production
//! MTL4 path (see `common::dispatch_threadgroups`), the row width baked.

mod common;

use half::{bf16, f16};
use objc2_metal::MTLSize;
use scratchy_target_metal::aot::baked_pipeline;
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::tape::ids::{NumExperts, TopK};
use scratchy_target_metal::tape::kernel_constants::{ScoresRow, SoftmaxConstants};

/// The block the MoE lowering dispatches `softmax.metal`'s kernels with.
const SOFTMAX_BLOCK_THREADS: usize = 256;

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

/// CPU reference of precise softmax: MLX's `mx.softmax(x, axis=-1,
/// precise=True)` — float accumulation, then cast back to the input dtype.
fn softmax_cpu_f32(input: &[f32], rows: usize, axis_size: usize, out: &mut [f32]) {
    assert_eq!(input.len(), rows * axis_size);
    assert_eq!(out.len(), rows * axis_size);
    for r in 0..rows {
        let row = &input[r * axis_size..(r + 1) * axis_size];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        let mut exps = vec![0.0_f32; axis_size];
        for (i, &v) in row.iter().enumerate() {
            let e = (v - m).exp();
            exps[i] = e;
            sum += e;
        }
        let inv = 1.0_f32 / sum;
        for (i, e) in exps.iter().enumerate() {
            out[r * axis_size + i] = *e * inv;
        }
    }
}

/// `softmax.metal` binding contract (`buffer(0)=input, buffer(1)=output`, the
/// row width baked); grid is one threadgroup per row.
fn run_softmax_mtl4(
    symbol: &str,
    row: ScoresRow,
    in_buf: &common::Buffer,
    out_buf: &common::Buffer,
    rows: u32,
) -> bool {
    let device = detect_device()
        .expect("Metal 4 GPU present (caller pre-guards)")
        .device;
    let constants = SoftmaxConstants { row }.into();
    let pipeline = baked_pipeline(&device, "softmax", symbol, constants).expect("softmax pipeline");
    common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[in_buf, out_buf],
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

fn experts(cols: usize) -> ScoresRow {
    ScoresRow::Experts(NumExperts(cols as u32))
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
        "block_softmax_precise_bfloat16",
        experts(cols),
        &in_buf,
        &out_buf,
        rows as u32,
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
        "block_softmax_precise_float16",
        experts(cols),
        &in_buf,
        &out_buf,
        rows as u32,
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

/// `topk_renorm_<dtype>` (Qwen3-MoE `norm_topk_prob`): each row of the
/// gathered top-k scores divided by its sum, in place.
fn run_renorm(f16_scores: bool, rows: usize, top_k: usize, seed: u64) {
    let Some(__dev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = __dev.device;
    let scores: Vec<f32> = fill_random_f32(rows, top_k, seed)
        .iter()
        .map(|&v| (v + 2.0) * 0.25)
        .collect();
    let (symbol, bits): (_, Vec<u16>) = match f16_scores {
        true => (
            "topk_renorm_float16",
            scores.iter().map(|&v| f16::from_f32(v).to_bits()).collect(),
        ),
        false => (
            "topk_renorm_bfloat16",
            scores
                .iter()
                .map(|&v| bf16::from_f32(v).to_bits())
                .collect(),
        ),
    };
    let value = |b: u16| match f16_scores {
        true => f16::from_bits(b).to_f32(),
        false => bf16::from_bits(b).to_f32(),
    };
    let buf = common::shared_slice(&device, &bits);

    let row = ScoresRow::TopK(TopK(top_k as u32));
    if !run_softmax_mtl4(symbol, row, &buf, &buf, rows as u32) {
        return;
    }

    let got: Vec<u16> = common::read_slice(&buf, rows * top_k);
    for r in 0..rows {
        let host = &bits[r * top_k..][..top_k];
        let sum: f32 = host.iter().map(|&b| value(b)).sum();
        for (k, &s) in host.iter().enumerate() {
            let want = value(s) / sum;
            let g = value(got[r * top_k + k]);
            assert!(
                (g - want).abs() < 1e-2,
                "{symbol} row{r}_k{k}: got {g} want {want}"
            );
        }
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

#[test]
fn topk_renorm_bf16_qwen3_moe_top8() {
    run_renorm(false, 13, 8, 0x0DD5_EED5);
}

#[test]
fn topk_renorm_f16_qwen3_moe_top8() {
    run_renorm(true, 13, 8, 0x0F16_5EED);
}

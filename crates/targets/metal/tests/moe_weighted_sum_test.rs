// SPDX-License-Identifier: Apache-2.0
//! Parity test: moe_weighted_sum kernel vs CPU reference for the
//! MoE-decode shape (N=1, top_k=8, hidden=2048 ≈ Qwen3-MoE).
//! Dispatches on the production MTL4 path (see
//! `common::dispatch_threadgroups`).

mod common;

use half::bf16;
use objc2_metal::MTLSize;
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::moe_weighted_sum::{
    MOE_WEIGHTED_SUM_TG_WIDTH, MoeSumDType, MoeWeightedSumKernels, moe_weighted_sum_cpu_f32,
};

fn rand_f32(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    let mut out = vec![0.0_f32; n];
    for v in &mut out {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let b = ((s as u32) & 0xFFFF) as f32 / 65535.0;
        *v = (b - 0.5) * scale;
    }
    out
}

fn run_bf16(rows: usize, top_k: usize, hidden: usize, seed: u64) {
    let Some(__dev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = __dev.device;
    let kernels = MoeWeightedSumKernels::new(&device).expect("kernels");

    let expert_f32 = rand_f32(rows * top_k * hidden, seed, 2.0);
    // Probability-like scores in (0, 1) summing to ~1 across top_k.
    let mut scores_f32 = rand_f32(rows * top_k, seed.wrapping_mul(31), 1.0)
        .into_iter()
        .map(|v| v.abs() + 1e-3)
        .collect::<Vec<_>>();
    for r in 0..rows {
        let s = scores_f32[r * top_k..(r + 1) * top_k].iter().sum::<f32>();
        for k in 0..top_k {
            scores_f32[r * top_k + k] /= s;
        }
    }

    let expert_bf16: Vec<bf16> = expert_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let scores_bf16: Vec<bf16> = scores_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let expert_buf = common::shared_slice(&device, &expert_bf16);
    let scores_buf = common::shared_slice(&device, &scores_bf16);
    let out_buf = common::shared_zeroed(&device, rows * hidden * std::mem::size_of::<bf16>());

    // Binding contract (matches shaders/moe_weighted_sum.metal):
    //   buffer(0)=expert_out, buffer(1)=scores, buffer(2)=out.
    // `top_k`/`hidden` are baked as function constants at pipeline
    // build time, not bound buffers. The classic path used
    // `dispatchThreads (hidden, rows, 1)`; the MTL4 helper dispatches
    // by threadgroup count, so we tile `hidden` into lanes of
    // `MOE_WEIGHTED_SUM_TG_WIDTH` with one threadgroup row per token.
    // This keeps `[[threads_per_grid]].y == rows` (the kernel's `n`
    // bound) and over-dispatched `d >= hidden` lanes self-guard.
    let pipeline = kernels
        .build_pipeline(MoeSumDType::BF16, top_k as u32, hidden as u32)
        .expect("pipeline");
    let tg_width = hidden.min(MOE_WEIGHTED_SUM_TG_WIDTH);
    let threadgroups = MTLSize {
        width: hidden.div_ceil(tg_width),
        height: rows,
        depth: 1,
    };
    let threads = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[&expert_buf, &scores_buf, &out_buf],
        threadgroups,
        threads,
    ) {
        return;
    }

    let got: Vec<bf16> = common::read_slice(&out_buf, rows * hidden);

    let expert_f32_round: Vec<f32> = expert_bf16.iter().map(|v| v.to_f32()).collect();
    let scores_f32_round: Vec<f32> = scores_bf16.iter().map(|v| v.to_f32()).collect();
    let mut want = vec![0.0_f32; rows * hidden];
    moe_weighted_sum_cpu_f32(
        &expert_f32_round,
        &scores_f32_round,
        &mut want,
        rows,
        top_k,
        hidden,
    );

    let mut max_err = 0.0_f32;
    for (g, w) in got.iter().zip(want.iter()) {
        let e = (g.to_f32() - w).abs();
        if e > max_err {
            max_err = e;
        }
    }
    // bf16 accumulation through float fma is ~ulp at the typical
    // |Σ_k probability * expert_out| ≈ ~1 magnitudes; tolerance
    // tracks bf16's mantissa precision (~3 decimal digits) plus
    // top_k accumulation.
    assert!(
        max_err < 5e-2,
        "moe_weighted_sum_bf16 rows={rows} top_k={top_k} hidden={hidden} max_err={max_err}"
    );
}

#[test]
fn moe_weighted_sum_bf16_mixtral_decode() {
    run_bf16(
        /*rows=*/ 1, /*top_k=*/ 2, /*hidden=*/ 4096, 0xC0FFEE,
    );
}

#[test]
fn moe_weighted_sum_bf16_qwen3_moe_decode() {
    run_bf16(
        /*rows=*/ 1,
        /*top_k=*/ 8,
        /*hidden=*/ 2048,
        0xDEAD_BEEF,
    );
}

#[test]
fn moe_weighted_sum_bf16_prefill_batch() {
    run_bf16(
        /*rows=*/ 32, /*top_k=*/ 8, /*hidden=*/ 2048, 0x12345,
    );
}

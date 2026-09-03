// SPDX-License-Identifier: Apache-2.0
//
// `nvfp4_qmv_<dtype>_s_f16_gs_16_b_4_batch_0` parity test: kernel ≡ CPU.
//
// NVFP4 weight-only dequant matvec. The CPU reference reconstructs
// `w = E2M1_signed[code] * scale[n, k/16]` (the per-group scale is the
// already-folded `e4m3(weight_scale) * weight_scale_2`, supplied here as
// a plain F16 per-group value) and does a row-major matmul with an f32
// accumulator. K=512 exercises both the main block loop and the
// remainder path; gs=16 is the NVFP4 group size (the affine generic-qmv
// test only covers gs∈{32,64,128}).

#![cfg(target_os = "macos")]

mod common;

use objc2_metal::MTLSize;
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::quantized::{
    QmmTKernel, QmvKernel, qmm_t_dispatch_shape, qmv_dispatch_shape,
};
use scratchy_target_metal::shader_cache::ShaderCache;
use scratchy_target_metal::specialized_pipeline_cache::ConstantValue;

const KE2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn e2m1(code: u8) -> f32 {
    let mag = KE2M1[(code & 0x7) as usize];
    if code & 0x8 != 0 { -mag } else { mag }
}

// SplitMix64 — deterministic.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f32_unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn byte(&mut self) -> u8 {
        (self.next_u64() >> 56) as u8
    }
}

/// Build random NVFP4 inputs. weight `[N, K/2]` u8 (2 codes/byte),
/// folded scales `[N, K/gs]` f16, activations `x` `[M, K]` bf16.
fn make_inputs(
    seed: u64,
    n: usize,
    k: usize,
    m: usize,
    gs: usize,
) -> (Vec<u8>, Vec<half::f16>, Vec<half::bf16>) {
    let mut rng = Rng(seed);
    let weight: Vec<u8> = (0..n * k / 2).map(|_| rng.byte()).collect();
    // small positive-ish folded scales (e4m3*weight_scale_2 ≈ 1e-3..1e-1)
    let scales: Vec<half::f16> = (0..n * k / gs)
        .map(|_| half::f16::from_f32(0.001 + 0.05 * rng.f32_unit()))
        .collect();
    let x: Vec<half::bf16> = (0..m * k)
        .map(|_| half::bf16::from_f32(rng.f32_unit() * 2.0 - 1.0))
        .collect();
    (weight, scales, x)
}

/// CPU reference: y[m,n] = Σ_k x[m,k] · E2M1[code(n,k)] · scale[n, k/gs].
fn cpu_nvfp4_qmv(
    weight: &[u8],
    scales: &[half::f16],
    x: &[half::bf16],
    m: usize,
    n: usize,
    k: usize,
    gs: usize,
) -> Vec<half::bf16> {
    let kg = k / gs;
    let kp = k / 2;
    let mut y = vec![half::bf16::from_f32(0.0); m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                let byte = weight[ni * kp + ki / 2];
                let code = if ki % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                let w = e2m1(code) * scales[ni * kg + ki / gs].to_f32();
                acc += x[mi * k + ki].to_f32() * w;
            }
            y[mi * n + ni] = half::bf16::from_f32(acc);
        }
    }
    y
}

fn mtl_size(t: (u32, u32, u32)) -> MTLSize {
    MTLSize {
        width: t.0 as usize,
        height: t.1 as usize,
        depth: t.2 as usize,
    }
}

/// Dispatch the nvfp4 generic-qmv kernel on the production MTL4 path.
/// Returns `None` if the host has no MTL4 queue (caller skips).
fn run_nvfp4_qmv(
    weight: &[u8],
    scales: &[half::f16],
    x: &[half::bf16],
    m: usize,
    n: usize,
    k: usize,
) -> Option<Vec<half::bf16>> {
    let device = detect_device()?.device;
    let cache = ShaderCache::new(device.clone()).expect("ShaderCache");

    // nvfp4 5-buffer layout (matches affine): w[0], scales[1],
    // biases[2]=scales (dummy, unused for nvfp4), x[3], y[4].
    let w_buf = common::shared_slice(&device, weight);
    let s_buf = common::shared_slice(&device, scales);
    let x_buf = common::shared_slice(&device, x);
    let y_buf = common::shared_zeroed(&device, m * n * std::mem::size_of::<half::bf16>());

    let constants = [
        ConstantValue::int(0, k as i32),
        ConstantValue::int(1, n as i32),
    ];
    let pipeline = cache
        .get_pipeline_specialized("nvfp4_qmv_bf16_s_f16_gs_16_b_4_batch_0", &constants)
        .expect("nvfp4_qmv pipeline");

    let (tg, tpg) = qmv_dispatch_shape(QmvKernel::Generic, m as u32, n as u32, 1);
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[&w_buf, &s_buf, &s_buf, &x_buf, &y_buf],
        mtl_size(tg),
        mtl_size(tpg),
    ) {
        return None;
    }
    Some(common::read_slice(&y_buf, m * n))
}

/// Dispatch the nvfp4 qmm_t kernel on the production MTL4 path.
/// Returns `None` if the host has no MTL4 queue (caller skips).
fn run_nvfp4_qmm_t(
    weight: &[u8],
    scales: &[half::f16],
    x: &[half::bf16],
    m: usize,
    n: usize,
    k: usize,
    aligned_n: bool,
) -> Option<Vec<half::bf16>> {
    let device = detect_device()?.device;
    let cache = ShaderCache::new(device.clone()).expect("ShaderCache");

    // 5-buffer layout (matches affine): w,scales,biases=scales,x,y.
    let w_buf = common::shared_slice(&device, weight);
    let s_buf = common::shared_slice(&device, scales);
    let x_buf = common::shared_slice(&device, x);
    let y_buf = common::shared_zeroed(&device, m * n * std::mem::size_of::<half::bf16>());

    // qmm_t function constants: QMM_K(0), QMM_N(1), QMM_M(2).
    let constants = [
        ConstantValue::int(0, k as i32),
        ConstantValue::int(1, n as i32),
        ConstantValue::int(2, m as i32),
    ];
    let fname = if aligned_n {
        "nvfp4_qmm_t_bf16_s_f16_gs_16_b_4_alN_true_batch_0"
    } else {
        "nvfp4_qmm_t_bf16_s_f16_gs_16_b_4_alN_false_batch_0"
    };
    let pipeline = cache
        .get_pipeline_specialized(fname, &constants)
        .expect("nvfp4_qmm_t pipeline");

    let (tg, tpg) = qmm_t_dispatch_shape(QmmTKernel::Standard, m as u32, n as u32, 1);
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[&w_buf, &s_buf, &s_buf, &x_buf, &y_buf],
        mtl_size(tg),
        mtl_size(tpg),
    ) {
        return None;
    }
    Some(common::read_slice(&y_buf, m * n))
}

#[test]
fn nvfp4_qmm_t_gs16_matches_cpu() {
    let (m, n, k, gs) = (64usize, 64usize, 512usize, 16usize);
    let (weight, scales, x) = make_inputs(0xBEEF_1234, n, k, m, gs);
    let expected = cpu_nvfp4_qmv(&weight, &scales, &x, m, n, k, gs);
    let Some(metal) = run_nvfp4_qmm_t(&weight, &scales, &x, m, n, k, /*aligned_n=*/ true) else {
        return;
    };
    let mut worst = 0.0f32;
    let mut wi = 0usize;
    for i in 0..m * n {
        let d = (expected[i].to_f32() - metal[i].to_f32()).abs();
        if d > worst {
            worst = d;
            wi = i;
        }
    }
    eprintln!(
        "nvfp4_qmm_t: worst_abs={worst:.5} at {wi} expected={:.4} metal={:.4}",
        expected[wi].to_f32(),
        metal[wi].to_f32()
    );
    eprintln!(
        "expected[0..8]: {:?}",
        expected[..8].iter().map(|v| v.to_f32()).collect::<Vec<_>>()
    );
    eprintln!(
        "metal[0..8]:    {:?}",
        metal[..8].iter().map(|v| v.to_f32()).collect::<Vec<_>>()
    );
    assert!(
        worst < 0.6,
        "nvfp4_qmm_t gs=16 mismatch: worst={worst:.5} at {wi}"
    );
}

/// Same kernel as `nvfp4_qmm_t_gs16_matches_cpu` but at **production
/// dims** (K=4096, N=4096 — Llama-3.1-8B q/o/down proj). gs=16 < BK=32
/// means a thread's contiguous K-tile straddles 2 scale groups; this
/// exercises that indexing across 128 K-tiles and 256 scale groups,
/// where an off-by-one in the per-byte scale index that's invisible at
/// K=512 (16 tiles) would blow up. Uses a relative-error gate: a correct
/// kernel only carries bf16 store noise (~0.4%); the straddle bug makes
/// whole terms wrong → order-of-magnitude error.
#[test]
fn nvfp4_qmm_t_gs16_production_dims_matches_cpu() {
    let (m, n, k, gs) = (64usize, 4096usize, 4096usize, 16usize);
    let (weight, scales, x) = make_inputs(0x5EED_4096, n, k, m, gs);
    let expected = cpu_nvfp4_qmv(&weight, &scales, &x, m, n, k, gs);
    let Some(metal) = run_nvfp4_qmm_t(&weight, &scales, &x, m, n, k, /*aligned_n=*/ true) else {
        return;
    };
    let mut worst = 0.0f32;
    let mut wi = 0usize;
    let mut max_abs = 0.0f32;
    for i in 0..m * n {
        let e = expected[i].to_f32();
        max_abs = max_abs.max(e.abs());
        let d = (e - metal[i].to_f32()).abs();
        if d > worst {
            worst = d;
            wi = i;
        }
    }
    let rel = worst / max_abs.max(1e-6);
    eprintln!(
        "nvfp4_qmm_t K=4096 N=4096: worst_abs={worst:.4} rel={rel:.4} max_abs={max_abs:.4} \
         at {wi} expected={:.4} metal={:.4}",
        expected[wi].to_f32(),
        metal[wi].to_f32()
    );
    eprintln!(
        "expected[0..8]: {:?}",
        expected[..8].iter().map(|v| v.to_f32()).collect::<Vec<_>>()
    );
    eprintln!(
        "metal[0..8]:    {:?}",
        metal[..8].iter().map(|v| v.to_f32()).collect::<Vec<_>>()
    );
    assert!(
        rel < 0.05,
        "nvfp4_qmm_t gs=16 prod-dims mismatch: worst={worst:.4} rel={rel:.4} at {wi}"
    );
}

#[test]
fn nvfp4_qmv_gs16_m4_matches_cpu() {
    let (m, n, k, gs) = (4usize, 8usize, 512usize, 16usize);
    let (weight, scales, x) = make_inputs(0xD15EA5E0, n, k, m, gs);
    let expected = cpu_nvfp4_qmv(&weight, &scales, &x, m, n, k, gs);
    let Some(metal) = run_nvfp4_qmv(&weight, &scales, &x, m, n, k) else {
        return;
    };
    let mut worst = 0.0f32;
    for i in 0..m * n {
        worst = worst.max((expected[i].to_f32() - metal[i].to_f32()).abs());
    }
    eprintln!("nvfp4_qmv M=4: worst_abs={worst:.5}");
    eprintln!(
        "expected: {:?}",
        expected.iter().map(|v| v.to_f32()).collect::<Vec<_>>()
    );
    eprintln!(
        "metal:    {:?}",
        metal.iter().map(|v| v.to_f32()).collect::<Vec<_>>()
    );
    assert!(worst < 0.5, "nvfp4_qmv M=4 mismatch: worst={worst:.5}");
}

/// qmv (decode matvec) at **production dims** — Llama-3.1-8B q_proj
/// (N=4096, K=4096), M=1. The shipped test only covered N=8/K=512; this
/// checks the generic-qmv grid actually covers all 4096 output rows and
/// that the gs=16 straddle decode holds at K=4096 (256 scale groups).
#[test]
fn nvfp4_qmv_gs16_production_dims_matches_cpu() {
    let (m, n, k, gs) = (1usize, 4096usize, 4096usize, 16usize);
    let (weight, scales, x) = make_inputs(0x90D1_4096, n, k, m, gs);
    let expected = cpu_nvfp4_qmv(&weight, &scales, &x, m, n, k, gs);
    let Some(metal) = run_nvfp4_qmv(&weight, &scales, &x, m, n, k) else {
        return;
    };
    let mut worst = 0.0f32;
    let mut wi = 0usize;
    let mut max_abs = 0.0f32;
    for i in 0..m * n {
        let e = expected[i].to_f32();
        max_abs = max_abs.max(e.abs());
        let d = (e - metal[i].to_f32()).abs();
        if d > worst {
            worst = d;
            wi = i;
        }
    }
    let rel = worst / max_abs.max(1e-6);
    eprintln!(
        "nvfp4_qmv N=4096 K=4096: worst_abs={worst:.4} rel={rel:.4} max_abs={max_abs:.4} at {wi} \
         expected={:.4} metal={:.4}",
        expected[wi].to_f32(),
        metal[wi].to_f32()
    );
    // Count how many rows are badly wrong — a grid-coverage bug shows as
    // a contiguous tail of zeros / wildly-off rows, not isolated noise.
    let bad = (0..n)
        .filter(|&i| (expected[i].to_f32() - metal[i].to_f32()).abs() > 0.1 * max_abs)
        .count();
    eprintln!("nvfp4_qmv N=4096: {bad}/{n} rows off by >10% of max");
    assert!(
        rel < 0.05,
        "nvfp4_qmv gs=16 prod-dims mismatch: worst={worst:.4} rel={rel:.4} at {wi} ({bad} bad rows)"
    );
}

#[test]
fn nvfp4_qmv_gs16_matches_cpu() {
    let (m, n, k, gs) = (1usize, 8usize, 512usize, 16usize);
    let (weight, scales, x) = make_inputs(0xC0FF_EE00, n, k, m, gs);
    let expected = cpu_nvfp4_qmv(&weight, &scales, &x, m, n, k, gs);
    let Some(metal) = run_nvfp4_qmv(&weight, &scales, &x, m, n, k) else {
        return;
    };

    let mut worst = 0.0f32;
    let mut worst_i = 0usize;
    for i in 0..m * n {
        let e = expected[i].to_f32();
        let g = metal[i].to_f32();
        let d = (e - g).abs();
        if d > worst {
            worst = d;
            worst_i = i;
        }
    }
    let e = expected[worst_i].to_f32();
    let g = metal[worst_i].to_f32();
    // bf16 accumulation noise floor: ~sqrt(K)*eps*|terms| — generous 0.5 abs.
    eprintln!("nvfp4_qmv: expected[{worst_i}]={e:.4} metal={g:.4} worst_abs={worst:.5}");
    eprintln!(
        "expected: {:?}",
        expected.iter().map(|v| v.to_f32()).collect::<Vec<_>>()
    );
    eprintln!(
        "metal:    {:?}",
        metal.iter().map(|v| v.to_f32()).collect::<Vec<_>>()
    );
    assert!(
        worst < 0.5,
        "nvfp4_qmv gs=16 mismatch: worst abs_err={worst:.5} at {worst_i} (expected {e:.4}, got {g:.4})"
    );
}

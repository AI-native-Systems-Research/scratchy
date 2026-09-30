// SPDX-License-Identifier: Apache-2.0
//! GEMM kernel goldens on the production MTL4 dispatch path:
//! `gemm_{f16,bf16}_specialized` vs the CPU reference across Llama
//! Q/K/V/O/down/lm_head shapes (M=1 decode and M=64 prefill, K up to
//! 8192). These custom MMA kernels replace MPS' `MPSMatrixMultiplication`;
//! M/N/K are baked into the pipeline as function constants, so the only
//! bindings are output(0), input(1), weight(2).

mod common;

use half::{bf16, f16};
use objc2_metal::MTLSize;
use scratchy_target_metal::cpu_golden;
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::specialized_pipeline_cache::{PipelineKey, SpecializedPipelineCache};
use scratchy_target_metal::tape::lowered::{GemmDims, MetalDtype};
use scratchy_target_metal::tape::lowering::gemm_command;

const SHAPES: &[(usize, usize, usize)] = &[
    (64, 2048, 2048),  // TinyLlama Q/K/V/O K-side
    (64, 3072, 3072),  // Llama-3.2-3B Q/O
    (64, 1024, 3072),  // Llama-3.2-3B K/V
    (64, 3072, 8192),  // Llama-3.2-3B down-proj
    (1, 128256, 3072), // Llama-3.2-3B lm_head/embed (tied)
    (1, 3072, 3072),   // Llama-3.2-3B Q/O decode
    (1, 1024, 3072),   // Llama-3.2-3B K/V decode
    (1, 3072, 8192),   // Llama-3.2-3B down decode
    (1, 512, 2048),    // Llama-3.2-1B K/V decode
    (1, 2048, 8192),   // Llama-3.2-1B down decode
    (1, 128256, 2048), // Llama-3.2-1B lm_head decode
    (1, 2048, 2048),   // Llama-3.2-1B Q/O decode
    (1, 8192, 2048),   // Llama-3.2-1B gate/up decode
];

/// The pipeline and grid of the GEMM command the lowering emits for `(dtype, m, n, k)` — the
/// production identity (library, symbol, M/N/K constants, grid), nothing re-derived here.
fn gemm_dispatch(
    cache: &SpecializedPipelineCache,
    dtype: MetalDtype,
    (m, n, k): (usize, usize, usize),
) -> (objc2::rc::Retained<common::Pipeline>, MTLSize, MTLSize) {
    let dims = GemmDims {
        m: m as u32,
        n: n as u32,
        k: k as u32,
    };
    let cmd = gemm_command(dtype, dims, Vec::new());
    let key = PipelineKey::new(cmd.library, cmd.function, cmd.constants.to_vec());
    let pso = cache.get_or_build(&key).expect("gemm pipeline");
    let size = |(width, height, depth): (u32, u32, u32)| MTLSize {
        width: width as usize,
        height: height as usize,
        depth: depth as usize,
    };
    let d = cmd.dispatch;
    (pso, size(d.threadgroups), size(d.threads_per_threadgroup))
}

fn make_pipelines() -> Option<(common::Device, SpecializedPipelineCache)> {
    let device = detect_device()?.device;
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone())
        .expect("compile standard shaders");
    Some((device, cache))
}

#[test]
fn gemm_bf16_matches_cpu_golden() {
    let Some((device, pl)) = make_pipelines() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    for &(m, n, k) in SHAPES {
        let input_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32) * 0.013).sin() * 0.3)
            .collect();
        let weight_f32: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.019).cos() * 0.3)
            .collect();
        let input: Vec<bf16> = input_f32.iter().map(|&v| bf16::from_f32(v)).collect();
        let weight: Vec<bf16> = weight_f32.iter().map(|&v| bf16::from_f32(v)).collect();

        let in_buf = common::shared_slice(&device, &input);
        let w_buf = common::shared_slice(&device, &weight);
        let out_buf = common::shared_zeroed(&device, m * n * std::mem::size_of::<bf16>());

        let (pso, grid, threads) = gemm_dispatch(&pl, MetalDtype::Bf16, (m, n, k));
        if !common::dispatch_threadgroups(
            &device,
            &pso,
            &[&out_buf, &in_buf, &w_buf],
            grid,
            threads,
        ) {
            return;
        }

        let got: Vec<bf16> = common::read_slice(&out_buf, m * n);
        let inb: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
        let wb: Vec<f32> = weight.iter().map(|v| v.to_f32()).collect();
        let mut want = vec![0.0_f32; m * n];
        cpu_golden::gemm(&inb, &wb, &mut want, m, k, n);

        // bf16 reduction over K up to 8192: looser tol than the
        // matmul-into-f32 path strictly needs, sized so a real numerical
        // break still trips.
        for i in 0..want.len() {
            let diff = (got[i].to_f32() - want[i]).abs();
            assert!(
                diff < 5e-2,
                "gemm_bf16 m={m} n={n} k={k} [{i}] (row {} col {}) metal={} cpu={} diff={}",
                i / n,
                i % n,
                got[i].to_f32(),
                want[i],
                diff
            );
        }
    }
}

#[test]
fn gemm_f16_matches_cpu_golden() {
    let Some((device, pl)) = make_pipelines() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    for &(m, n, k) in SHAPES {
        let input_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32) * 0.013).sin() * 0.3)
            .collect();
        let weight_f32: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.019).cos() * 0.3)
            .collect();
        let input: Vec<f16> = input_f32.iter().map(|&v| f16::from_f32(v)).collect();
        let weight: Vec<f16> = weight_f32.iter().map(|&v| f16::from_f32(v)).collect();

        let in_buf = common::shared_slice(&device, &input);
        let w_buf = common::shared_slice(&device, &weight);
        let out_buf = common::shared_zeroed(&device, m * n * std::mem::size_of::<f16>());

        let (pso, grid, threads) = gemm_dispatch(&pl, MetalDtype::F16, (m, n, k));
        if !common::dispatch_threadgroups(
            &device,
            &pso,
            &[&out_buf, &in_buf, &w_buf],
            grid,
            threads,
        ) {
            return;
        }

        let got: Vec<f16> = common::read_slice(&out_buf, m * n);
        let inb: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
        let wb: Vec<f32> = weight.iter().map(|v| v.to_f32()).collect();
        let mut want = vec![0.0_f32; m * n];
        cpu_golden::gemm(&inb, &wb, &mut want, m, k, n);

        for i in 0..want.len() {
            let diff = (got[i].to_f32() - want[i]).abs();
            assert!(
                diff < 5e-2,
                "gemm_f16 m={m} n={n} k={k} [{i}] (row {} col {}) metal={} cpu={} diff={}",
                i / n,
                i % n,
                got[i].to_f32(),
                want[i],
                diff
            );
        }
    }
}

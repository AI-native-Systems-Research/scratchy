// SPDX-License-Identifier: Apache-2.0
//! Dense GEMM goldens on the production MTL4 dispatch path, the kernel and dispatch the worker
//! picks (`pipeline_for_gemm`: up to 8 rows run MLX's GEMV per row, `gemv_{f16,bf16}_specialized`;
//! more rows the MMA GEMM, `gemm_{f16,bf16}_specialized`, `gemm_bf16_blocked` from 256 rows) vs
//! the CPU reference across Llama Q/K/V/O/down/lm_head shapes (M=1 decode and M=64 prefill, K up
//! to 8192) and the Gemma-4 and Qwen3.6 MoE routers at decode, speculative-verify and prefill row
//! counts. M/N/K are compiled into the kernel, so the only bindings are
//! output(0), input(1), weight(2).

mod common;

use half::{bf16, f16};
use objc2_metal::MTLSize;
use scratchy_target_metal::aot::{baked_build, baked_kernels};
use scratchy_target_metal::cpu_golden;
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::interpreter::metal::__re::ComputePipelineState;
use scratchy_target_metal::interpreter::metal::pipelines::{GemmBody, gemm_pipeline};
use scratchy_target_metal::interpreter::metal::{GemmDims, MetalDtype, SpecializedPipelines};
use scratchy_target_metal::specialized_pipeline_cache::{PipelineKey, SpecializedPipelineCache};
use scratchy_target_metal::targets::is_nax_capable;

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
    (1, 128, 2816),    // Gemma-4-26B-A4B MoE router decode
    (3, 128, 2816),    // Gemma-4-26B-A4B MoE router, 3 rows
    (2, 256, 2048),    // Qwen3.6-35B-A3B MoE router, verify buckets (2 / 4 / 8 rows)
    (4, 256, 2048),
    (8, 256, 2048),
    (2048, 128, 2816), // Gemma-4-26B-A4B MoE router prefill (bf16: blocked)
    (256, 256, 2048),  // Qwen3.6-35B-A3B MoE router, the first blocked bucket on main (NAX here)
];

/// The blocked bf16 GEMM runs the 8×8-tile GEMM's MMAs in its order: the same bits, at the MoE
/// routers' prefill shapes and at M/N/K tails (K = 16q + 8).
#[test]
fn gemm_bf16_blocked_is_the_tile8_gemm_bit_for_bit() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device;
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("pipeline cache");
    for (m, n, k) in [(2048, 128, 2816), (256, 256, 2048), (300, 100, 1032)] {
        let dims = GemmDims { m, n, k };
        let (blocked, shape) =
            gemm_pipeline(MetalDtype::Bf16, dims, GemmBody::Blocked).expect("gemm key");
        assert_eq!(blocked.kernel_name, "gemm_bf16_blocked");
        let tile8 = PipelineKey::new("gemm", "gemm_bf16_specialized", blocked.constants.clone());
        let (m, n, k) = (m as usize, n as usize, k as usize);
        let bf = |len: usize, f: fn(f32) -> f32| -> Vec<bf16> {
            (0..len)
                .map(|i| bf16::from_f32(f(i as f32) * 0.3))
                .collect()
        };
        let input = common::shared_slice(&device, &bf(m * k, |i| (i * 0.013).sin()));
        let weight = common::shared_slice(&device, &bf(n * k, |i| (i * 0.019).cos()));
        let run = |key: &PipelineKey, grid: (u32, u32), threads: u32| -> Vec<u16> {
            let pso = baked_build(&cache, key).expect("pipeline");
            let out = common::shared_zeroed(&device, m * n * 2);
            let size = |w: u32, h: u32| MTLSize {
                width: w as usize,
                height: h as usize,
                depth: 1,
            };
            let bufs = [&out, &input, &weight];
            assert!(common::dispatch_threadgroups(
                &device,
                &pso,
                &bufs,
                size(grid.0, grid.1),
                size(threads, 1)
            ));
            common::read_slice(&out, m * n)
        };
        let (g, t) = (shape.threadgroups, shape.threads_per_threadgroup.0);
        let got = run(&blocked, (g.0, g.1), t);
        let want = run(&tile8, ((n as u32).div_ceil(8), (m as u32).div_ceil(8)), 32);
        assert!(
            got == want,
            "m={m} n={n} k={k}: the blocked GEMM's bits differ"
        );
    }
}

/// The pipeline the worker plays a dense GEMM of `(m, n, k)` with, and its dispatch as MTL sizes.
fn gemm(
    pl: &SpecializedPipelines,
    dtype: MetalDtype,
    (m, n, k): (usize, usize, usize),
) -> (ComputePipelineState, MTLSize, MTLSize) {
    let dims = GemmDims {
        m: m as u32,
        n: n as u32,
        k: k as u32,
    };
    let (pso, shape) = pl.pipeline_for_gemm(dtype, dims).expect("gemm pipeline");
    let size = |(width, height, depth): (u32, u32, u32)| MTLSize {
        width: width as usize,
        height: height as usize,
        depth: depth as usize,
    };
    (
        pso,
        size(shape.threadgroups),
        size(shape.threads_per_threadgroup),
    )
}

/// The pipelines, every `SHAPES` GEMM baked at both dtypes as a model's tape bakes its own.
fn make_pipelines() -> Option<(common::Device, SpecializedPipelines)> {
    let device = detect_device()?.device;
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("pipeline cache");
    let keys: Vec<_> = [MetalDtype::Bf16, MetalDtype::F16]
        .into_iter()
        .flat_map(|dtype| SHAPES.iter().map(move |&(m, n, k)| (dtype, m, n, k)))
        .map(|(dtype, m, n, k)| {
            let (m, n, k) = (m as u32, n as u32, k as u32);
            GemmBody::ALL.map(|body| {
                gemm_pipeline(dtype, GemmDims { m, n, k }, body)
                    .expect("gemm key")
                    .0
            })
        })
        .collect();
    let keys: Vec<_> = keys.into_iter().flatten().collect();
    cache.register_baked(&baked_kernels(&keys));
    // A GEMM binds no variant-bound constant: any variant serves.
    let variant = scratchy_target_metal::tape::constants::TapeVariant {
        cap: scratchy_target_metal::tape::ids::MaxBlocksPerSeq(128),
        tq_heads: None,
    };
    Some((
        device,
        SpecializedPipelines::new(std::sync::Arc::new(cache), variant),
    ))
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

        let (pso, grid, threads) = gemm(&pl, MetalDtype::Bf16, (m, n, k));
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

/// The NAX body's GEMM (`gemm_nax_bf16_dense`) matches the CPU reference at the MoE-router
/// shapes it is picked for, and at M/N tails — its contraction steps 64, so a K tail cannot
/// reach it (`gemm_body` falls those to the simdgroup bodies). Skips on a GPU without the matrix
/// unit: the kernel's fragment layout is the unit's own, emulated differently on an M4.
#[test]
fn gemm_bf16_nax_matches_cpu_golden() {
    let Some(di) = detect_device().filter(|di| is_nax_capable(di.profile.generation)) else {
        eprintln!("skipping: no NAX matrix unit");
        return;
    };
    let device = di.device;
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("pipeline cache");
    for (m, n, k) in [(2048, 128, 2816), (256, 256, 2048), (300, 100, 1024)] {
        let dims = GemmDims {
            m: m as u32,
            n: n as u32,
            k: k as u32,
        };
        let (key, shape) = gemm_pipeline(MetalDtype::Bf16, dims, GemmBody::Nax).expect("gemm key");
        assert_eq!(key.kernel_name, "gemm_nax_bf16_dense");
        let pso = baked_build(&cache, &key).expect("pipeline");

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

        let size = |(w, h, d): (u32, u32, u32)| MTLSize {
            width: w as usize,
            height: h as usize,
            depth: d as usize,
        };
        assert!(common::dispatch_threadgroups(
            &device,
            &pso,
            &[&out_buf, &in_buf, &w_buf],
            size(shape.threadgroups),
            size(shape.threads_per_threadgroup),
        ));

        let got: Vec<bf16> = common::read_slice(&out_buf, m * n);
        let inb: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
        let wb: Vec<f32> = weight.iter().map(|v| v.to_f32()).collect();
        let mut want = vec![0.0_f32; m * n];
        cpu_golden::gemm(&inb, &wb, &mut want, m, k, n);
        for i in 0..want.len() {
            let diff = (got[i].to_f32() - want[i]).abs();
            assert!(
                diff < 5e-2,
                "gemm_bf16_nax m={m} n={n} k={k} [{i}] (row {} col {}) metal={} cpu={} diff={}",
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

        let (pso, grid, threads) = gemm(&pl, MetalDtype::F16, (m, n, k));
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

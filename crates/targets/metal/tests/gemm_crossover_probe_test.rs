// SPDX-License-Identifier: Apache-2.0
//! The dense bf16 GEMM's three bodies — the 8×8-tile simdgroup body, the 32×32 blocked body, and
//! the 64×64 NAX matrix-unit body (`gemm_nax_bf16_dense`) — timed against each other at the
//! MoE-router and dense-projection geometries across M. This is the sweep `NAX_GEMM_MIN_M` is
//! measured on: rerun it on a new chip before moving the constant. It runs 3 ms bursts of one
//! kernel, so its cells sit at un-ramped clocks — it pins the tile8-vs-nax range and the
//! bodies' orderings, but NOT the wide-N floor (`NAX_GEMM_WIDE_N`): at high M a narrow-N NAX
//! grid (n/64 × m/64 threadgroups, 2 wide at the MoE routers' n = 128) loses to the blocked
//! body with the clocks a real prefill runs at, a regime only the interleaved e2e A/B sees.
//!
//! Run: cargo test -p scratchy-target-metal --release --test gemm_crossover_probe_test -- \
//!     --ignored --nocapture

mod common;

use half::bf16;
use objc2_metal::MTLSize;
use scratchy_target_metal::aot::{baked_build, baked_kernels};
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::interpreter::metal::pipelines::{GemmBody, gemm_pipeline};
use scratchy_target_metal::interpreter::metal::{GemmDims, MetalDtype};
use scratchy_target_metal::mtl4_dispatch::{
    Mtl4DispatchBatch, Pipeline, shared_slice, shared_zeroed,
};
use scratchy_target_metal::specialized_pipeline_cache::{PipelineKey, SpecializedPipelineCache};
use scratchy_target_metal::targets::is_nax_capable;

/// Dispatches per timed command buffer.
const ITERS: usize = 64;

/// Time `pso` at `grid`×`threads` over one command buffer of `ITERS` dispatches, in µs/dispatch.
fn time_body(
    device: &common::Device,
    pso: &Pipeline,
    bufs: &[&common::Buffer; 3],
    grid: MTLSize,
    threads: MTLSize,
) -> f64 {
    let mut batch = Mtl4DispatchBatch::begin(device).expect("mtl4 batch");
    for _ in 0..ITERS {
        batch.encode(
            pso,
            &[(bufs[0], 0), (bufs[1], 1), (bufs[2], 2)],
            &[],
            &[],
            &[],
            grid,
            threads,
        );
    }
    let t0 = std::time::Instant::now();
    batch.commit(true);
    t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64
}

#[test]
#[ignore = "measurement probe, not an assertion: run with --ignored --nocapture"]
fn gemm_tile_blocked_nax_crossover_on_this_gpu() {
    let Some(di) = detect_device().filter(|di| is_nax_capable(di.profile.generation)) else {
        eprintln!("skipping: no NAX matrix unit");
        return;
    };
    let device = di.device;
    {
        use objc2_metal::MTLDevice as _;
        println!(
            "device: {} — Metal architecture name: {} — has_nax: {}",
            device.name(),
            device.architecture().name(),
            is_nax_capable(di.profile.generation),
        );
    }
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("pipeline cache");

    // The two MoE-router geometries the ladder actually runs (gemm_golden_test's SHAPES):
    // Qwen3.6-35B-A3B (256 experts, hidden 2048) and Gemma-4-26B-A4B (128, 2816), plus a wide-N
    // projection geometry. The M sweep includes the decode-small rows (the ladder's 2/4/8 buckets)
    // to pin the law's lower boundary.
    let geoms: &[(u32, u32, &str)] = &[
        (256, 2048, "qwen3.6-35b"),
        (128, 2816, "gemma-4-26b"),
        (2048, 2048, "wide-n"),
    ];
    let ms: &[u32] = &[
        2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 2048,
    ];

    let mut keys: Vec<PipelineKey> = Vec::new();
    for &(n, k, _) in geoms {
        for &m in ms {
            for body in GemmBody::ALL {
                keys.push(
                    gemm_pipeline(MetalDtype::Bf16, GemmDims { m, n, k }, body)
                        .expect("gemm key")
                        .0,
                );
            }
        }
    }
    keys.dedup();
    cache.register_baked(&baked_kernels(&keys));

    let size = |(w, h, _): (u32, u32, u32)| MTLSize {
        width: w as usize,
        height: h as usize,
        depth: 1,
    };
    let threads3 = |(w, h, d): (u32, u32, u32)| MTLSize {
        width: w as usize,
        height: h as usize,
        depth: d as usize,
    };
    for &(n, k, name) in geoms {
        println!("== {name} router: n={n} k={k} (µs/dispatch, best of 3 batches of {ITERS})");
        println!(
            "    {:>5} {:>10} {:>10} {:>10}  winner",
            "M", "tile8", "blocked", "nax"
        );
        for &m in ms {
            let bf = |len: usize, f: fn(f32) -> f32| -> Vec<bf16> {
                (0..len)
                    .map(|i| bf16::from_f32(f(i as f32) * 0.3))
                    .collect()
            };
            let out = shared_zeroed(&device, m as usize * n as usize * 2);
            let input = shared_slice(&device, &bf(m as usize * k as usize, f32::sin));
            let weight = shared_slice(&device, &bf(n as usize * k as usize, f32::cos));
            let bufs = [&out, &input, &weight];

            let mut best = [f64::MAX, f64::MAX, f64::MAX];
            for body in GemmBody::ALL {
                let (key, shape) =
                    gemm_pipeline(MetalDtype::Bf16, GemmDims { m, n, k }, body).expect("gemm key");
                let pso = baked_build(&cache, &key).expect("pipeline");
                let grid = size(shape.threadgroups);
                let threads = threads3(shape.threads_per_threadgroup);
                // Warm-up batch (first-run pipeline + residency), then best of 3.
                time_body(&device, &pso, &bufs, grid, threads);
                for _ in 0..3 {
                    best[body as usize] =
                        best[body as usize].min(time_body(&device, &pso, &bufs, grid, threads));
                }
            }
            let (tile, blocked, nax) = (best[0], best[1], best[2]);
            let winner = match (tile <= blocked, tile <= nax) {
                (true, true) => "tile8",
                (false, _) if blocked <= nax => "blocked",
                _ => "nax",
            };
            println!(
                "    {:>5} {:>10.2} {:>10.2} {:>10.2}  {}",
                m, tile, blocked, nax, winner
            );
        }
    }
}

// SPDX-License-Identifier: Apache-2.0
//! The W4A8 GEMM (`affine_qmm_w4a8_*`) against its split-K twin (`affine_qmm_w4a8_splitk_*`) at
//! Qwen3.8-27B's dense projections (and a 2048-hidden model's), in the 32-row verify bucket and
//! the 64-row one: the tiled GEMM at both tiles, each split, and which of them the wide class's
//! lowering picks (`W4a8Gemm::for_shape`, marked `*`). This is the sweep
//! `W4A8_SPLIT_K_MIN_SIMDGROUPS` was set on (40-core M5 Max); rerun it on a new chip before moving
//! it. Each cell is µs a dispatch over one command buffer streaming `BYTES_PER_BATCH` of weights,
//! the dispatches a barrier apart and the weights rotated over copies past the system cache, the
//! cells in alternating rounds, best of `ROUNDS`; GB/s is the codes + scales + biases a dispatch
//! streams.
//!
//! Run: cargo test -p scratchy-target-metal --release --test w4a8_splitk_probe_test -- \
//!     --ignored --nocapture

mod common;

use half::bf16;
use objc2_metal::MTLSize;
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::{Mtl4DispatchBatch, shared_slice, shared_zeroed};
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use scratchy_target_metal::tape::kernel_constants::AffineCodes;
use scratchy_target_metal::tape::lowered::GenClass;
use scratchy_target_metal::tape::quantized::{
    DequantDtype, ScaleDtype, W4A8_TILE_ROWS, W4a8Gemm, W4a8SplitK, W4a8Tile, w4a8_scratch_bytes,
};

/// Weight bytes one timed command buffer streams, so its submit costs a few percent at most.
const BYTES_PER_BATCH: usize = 4 << 30;

/// Weight bytes the copies span, past the system cache.
const ROTATED_BYTES: usize = 256 << 20;

const GROUP: u32 = 64;

/// Rounds of a shape's cells, after a warm-up round.
const ROUNDS: usize = 5;

/// Buckets: the 32-row verify bucket, and the next.
const ROWS: [u32; 2] = [32, 64];

/// Qwen3.8-27B's dense W4A8 projections, then a 2048-hidden model's: (name, n, k).
const SHAPES: &[(&str, u32, u32)] = &[
    ("mlp gate/up", 17408, 5120),
    ("mlp down", 5120, 17408),
    ("o_proj/gdn out", 5120, 6144),
    ("gdn qkv", 10240, 5120),
    ("gdn z", 6144, 5120),
    ("attn q+gate", 12288, 5120),
    ("attn k/v", 1024, 5120),
    ("h2048 n8192", 8192, 2048),
    ("h2048 n4096", 4096, 2048),
    ("h2048 n512", 512, 2048),
    ("h2048 n2048 k4096", 2048, 4096),
];

fn bf16_row(len: usize, v: f32) -> Vec<u16> {
    (0..len)
        .map(|i| bf16::from_f32(v * (1.0 + 0.001 * (i % 97) as f32)).to_bits())
        .collect()
}

fn size(width: u32, height: u32) -> MTLSize {
    MTLSize {
        width: width as usize,
        height: height as usize,
        depth: 1,
    }
}

#[test]
#[ignore = "measurement probe, not an assertion: run with --ignored --nocapture"]
fn w4a8_against_split_k_at_qwen38_27b_shapes() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    if !scratchy_target_metal::targets::is_nax_capable(di.profile.generation) {
        eprintln!("skipping: no NAX matrix unit");
        return;
    }
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("pipeline cache");
    let (act, scale) = (DequantDtype::Bf16, ScaleDtype::Bf16);
    for m in ROWS {
        println!("µs (GB/s) a dispatch at {m} rows");
        for &(name, n, k) in SHAPES {
            let (nu, ku) = (n as usize, k as usize);
            let code_bytes = nu * ku / 2;
            let groups = nu * ku / GROUP as usize;
            let streamed = code_bytes + 2 * groups * 2;
            let copies = ROTATED_BYTES / streamed + 1;
            let iters = (BYTES_PER_BATCH / streamed).max(32);
            let weights: Vec<_> = (0..copies)
                .map(|c| {
                    let codes: Vec<u32> = (0..code_bytes as u32 / 4)
                        .map(|i| i.wrapping_mul(2654435761).wrapping_add(c as u32))
                        .collect();
                    (
                        shared_slice(&device, &codes),
                        shared_slice(&device, &bf16_row(groups, 0.002)),
                        shared_slice(&device, &bf16_row(groups, -0.015)),
                    )
                })
                .collect();
            let scratch = shared_zeroed(&device, w4a8_scratch_bytes(m, k) as usize);
            let y = shared_zeroed(&device, m as usize * nu * 2);
            let mut constants = vec![
                ConstantValue::int(0, k as i32),
                ConstantValue::int(1, n as i32),
                ConstantValue::int(2, m as i32),
            ];
            constants.extend(AffineCodes::Offset8.constant());
            let tiles = [W4a8Tile::Cols128, W4a8Tile::Cols64];
            let kernels: Vec<W4a8Gemm> = (tiles.into_iter())
                .filter(|t| n.is_multiple_of(t.cols()))
                .map(W4a8Gemm::Tiled)
                .chain(W4a8SplitK::ALL.map(W4a8Gemm::SplitK))
                .collect();
            let picked = W4a8Tile::for_n(n)
                .map(|t| W4a8Gemm::for_shape(t, n, k, m, GenClass::M5Wide))
                .expect("a W4A8 width");
            let cells: Vec<(String, &'static str, u32, u32)> = (kernels.iter())
                .map(|&g| {
                    let tag = match g {
                        W4a8Gemm::Tiled(t) => format!("tn{}", t.cols()),
                        W4a8Gemm::SplitK(s) => format!("k{}", s.ways()),
                        W4a8Gemm::SmallM(s) => format!("m16k{}", s.ways()),
                    };
                    let tag = if g == picked { tag + "*" } else { tag };
                    (
                        tag,
                        g.static_name(act, scale, GROUP),
                        g.cols(),
                        g.simdgroups(),
                    )
                })
                .collect();
            let psos: Vec<_> = (cells.iter())
                .map(|(_, sym, _, _)| {
                    baked_build(
                        &cache,
                        &PipelineKey::new("quantized_qmm_nax", sym, constants.clone()),
                    )
                    .expect(sym)
                })
                .collect();
            let time = |cell: usize| {
                let (_, _, tn, sgs) = cells[cell];
                let mut batch = Mtl4DispatchBatch::begin(&device).expect("mtl4 batch");
                for i in 0..iters {
                    let (w, s, b) = &weights[i % copies];
                    batch.encode(
                        &psos[cell],
                        &[(w, 0), (s, 1), (b, 2), (&scratch, 3), (&y, 4)],
                        &[],
                        &[],
                        &[],
                        size(n / tn, m.div_ceil(W4A8_TILE_ROWS)),
                        size(32 * sgs, 1),
                    );
                    batch.barrier();
                }
                let t0 = std::time::Instant::now();
                batch.commit(true);
                t0.elapsed().as_secs_f64() * 1e6 / iters as f64
            };
            // A warm-up round, then the cells in alternating rounds, best of `ROUNDS`.
            (0..cells.len()).for_each(|c| {
                time(c);
            });
            let mut best = vec![f64::MAX; cells.len()];
            for _ in 0..ROUNDS {
                for (c, b) in best.iter_mut().enumerate() {
                    *b = b.min(time(c));
                }
            }
            let mut line = format!("{name:<16}");
            for ((tag, ..), us) in cells.iter().zip(&best) {
                line += &format!("  {tag:>6} {us:6.1} ({:3.0})", streamed as f64 / us / 1e3);
            }
            println!("{line}");
        }
    }
}

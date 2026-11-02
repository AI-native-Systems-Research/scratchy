// SPDX-License-Identifier: Apache-2.0
//! A small-M step in the 8-row bucket, whose own GEMM is the wide matvec (no W4A8 pre-pass of its
//! own): the bucket's wide matvec (`wide`), the bf16 small-M twin (`twin`, `affine_qmm_small_m_*`
//! at the 8-row tile, what the 8-row bucket ran before), and the W4A8 pre-pass + 16-row split-K
//! GEMM (`W4a8Gemm::SmallM`, `W4a8Gemm::for_small_m`'s split), both dispatches timed, baked at a
//! 16-row bucket (`pp16`) and at the 8-row one (`pp8`, the tape's: the pre-pass at the tile's 16
//! rows, the GEMM storing 8), at Qwen3.8-27B's and Qwen3.6-35B-A3B's dense projections with a 4- or 8-token
//! step's rows live. This is the sweep the wide class's 8-row route (`route_small_m`) was set on (40-core M5
//! Max); rerun it on a new chip before moving it. Each cell is µs a dispatch (the pair's two for
//! `pp`) over one command buffer streaming `BYTES_PER_BATCH` of weights, the dispatches a barrier
//! apart and the weights rotated over copies past the system cache, the cells in alternating
//! rounds, median of `ROUNDS`; GB/s is the codes + scales + biases a dispatch streams.
//!
//! Run: cargo test -p scratchy-target-metal --release --test row8_verify_probe_test -- \
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
use scratchy_target_metal::tape::ids::{KDimI32, MDimI32, NDimI32};
use scratchy_target_metal::tape::kernel_constants::{AffineCodes, AffineQmvWideConstants};
use scratchy_target_metal::tape::quantized::{
    DequantDtype, SMALL_M_SIMDGROUPS, SMALL_M_TILE_COLS, ScaleDtype, SmallMTile, W4a8Gemm,
    W4a8Rows, pick_qmv_kernel_wide, qmv_dispatch_shape, qmv_kernel_static_name,
    small_m_kernel_static_name, w4a8_quant_static_name, w4a8_scratch_bytes,
};

/// Weight bytes one timed command buffer streams, so its submit costs a few percent at most.
const BYTES_PER_BATCH: usize = 4 << 30;

/// Weight bytes the copies span, past the system cache.
const ROTATED_BYTES: usize = 256 << 20;

const GROUP: u32 = 64;

/// Rounds of a shape's cells, after a warm-up round.
const ROUNDS: usize = 7;

/// The bucket, and the step's live rows in it: a 1-lane verify's 8, and a 4-token step.
const BUCKET: u32 = 8;
const LIVE: [u32; 2] = [8, 4];

/// The 16-row tile's bucket: the rows of the scratch the pre-pass writes in either bucket.
const PAIR_BUCKET: u32 = 16;

/// Qwen3.8-27B's dense projections: (name, n, k).
const SHAPES: &[(&str, u32, u32)] = &[
    ("mlp gate/up", 17408, 5120),
    ("mlp down", 5120, 17408),
    ("o_proj/gdn out", 5120, 6144),
    ("gdn qkv", 10240, 5120),
    ("gdn z", 6144, 5120),
    ("attn q+gate", 12288, 5120),
    ("attn k/v", 1024, 5120),
    ("lm_head", 248320, 5120),
    // Qwen3.6-35B-A3B's dense projections, and the narrow-projection test's 1024 x 2048.
    ("3.6 lm_head", 248320, 2048),
    ("3.6 qkv / q", 8192, 2048),
    ("3.6 gdn z", 4096, 2048),
    ("3.6 out_proj", 2048, 4096),
    ("1024 x 2048", 1024, 2048),
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

type Pipeline =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>;

/// What a cell dispatches: a kernel over the bf16 rows, or the pre-pass then the GEMM over its
/// int8 rows.
enum Run {
    Rows {
        pso: Pipeline,
        grid: (u32, u32),
        threads: (u32, u32),
    },
    Pair {
        quant: Pipeline,
        gemm: Pipeline,
        grid: (u32, u32),
        threads: u32,
    },
}

#[test]
#[ignore = "measurement probe, not an assertion: run with --ignored --nocapture"]
fn small_m_steps_in_the_eight_row_bucket_at_qwen38_27b_shapes() {
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
    let codes = AffineCodes::of(Some(&di.profile), 4);
    let twin = SmallMTile::for_bucket(BUCKET).expect("a small-M bucket");
    for live in LIVE {
        println!("µs (GB/s) a dispatch, {live} live rows in the {BUCKET}-row bucket");
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
            let x = shared_slice(&device, &bf16_row(PAIR_BUCKET as usize * ku, 0.3));
            let scratch = shared_zeroed(&device, w4a8_scratch_bytes(PAIR_BUCKET, k) as usize);
            let y = shared_zeroed(&device, PAIR_BUCKET as usize * nu * 2);
            let consts = |m: u32| {
                let mut c = vec![
                    ConstantValue::int(0, k as i32),
                    ConstantValue::int(1, n as i32),
                    ConstantValue::int(2, m as i32),
                ];
                c.extend(codes.constant());
                c
            };
            let pso = |library, symbol, constants: Vec<ConstantValue>| {
                baked_build(&cache, &PipelineKey::new(library, symbol, constants)).expect(symbol)
            };
            let mut cells: Vec<(String, Run)> = Vec::new();
            let wide = pick_qmv_kernel_wide(n, k, 4, BUCKET, true);
            let wide_constants: Vec<ConstantValue> = AffineQmvWideConstants {
                k: KDimI32(k as i32),
                n: NDimI32(n as i32),
                m: MDimI32(BUCKET as i32),
                codes,
            }
            .into();
            // The wide grid is the bucket's at every live count (its vector tile is baked).
            let ((gx, gy, _), (tx, ty, _)) = qmv_dispatch_shape(wide, BUCKET, n, 1);
            cells.push((
                "wide".into(),
                Run::Rows {
                    pso: pso(
                        "quantized_qmv",
                        qmv_kernel_static_name(wide, act, scale, 4, GROUP),
                        wide_constants,
                    ),
                    grid: (gx, gy),
                    threads: (tx, ty),
                },
            ));
            cells.push((
                "twin".into(),
                Run::Rows {
                    pso: pso(
                        "quantized_qmm_nax",
                        small_m_kernel_static_name(act, scale, GROUP, twin),
                        consts(BUCKET),
                    ),
                    grid: (n / SMALL_M_TILE_COLS, live.div_ceil(twin.rows())),
                    threads: (32 * SMALL_M_SIMDGROUPS, 1),
                },
            ));
            if let Some(small) = W4a8Gemm::for_small_m(n, k) {
                for (tag, rows) in [("pp16", PAIR_BUCKET), ("pp8", BUCKET)] {
                    cells.push((
                        format!("{tag}+m16k{}", small.simdgroups()),
                        Run::Pair {
                            // The scratch is laid out for the 16-row tile in either bucket.
                            quant: pso(
                                "quantized_qmm_nax",
                                w4a8_quant_static_name(W4a8Rows::Dense, act),
                                consts(PAIR_BUCKET),
                            ),
                            gemm: pso(
                                "quantized_qmm_nax",
                                small.static_name(act, scale, GROUP),
                                consts(rows),
                            ),
                            grid: (n / small.cols(), 1),
                            threads: 32 * small.simdgroups(),
                        },
                    ));
                }
            }
            let time = |run: &Run| {
                let mut batch = Mtl4DispatchBatch::begin(&device).expect("mtl4 batch");
                for i in 0..iters {
                    let (w, s, b) = &weights[i % copies];
                    match run {
                        Run::Rows { pso, grid, threads } => batch.encode(
                            pso,
                            &[(w, 0), (s, 1), (b, 2), (&x, 3), (&y, 4)],
                            &[],
                            &[],
                            &[],
                            size(grid.0, grid.1),
                            size(threads.0, threads.1),
                        ),
                        Run::Pair {
                            quant,
                            gemm,
                            grid,
                            threads,
                        } => {
                            batch.encode(
                                quant,
                                &[(&x, 0), (&scratch, 1)],
                                &[],
                                &[],
                                &[],
                                size((k / 64).div_ceil(16), live),
                                size(128, 1),
                            );
                            batch.barrier();
                            batch.encode(
                                gemm,
                                &[(w, 0), (s, 1), (b, 2), (&scratch, 3), (&y, 4)],
                                &[],
                                &[],
                                &[],
                                size(grid.0, grid.1),
                                size(*threads, 1),
                            );
                        }
                    }
                    batch.barrier();
                }
                let t0 = std::time::Instant::now();
                batch.commit(true);
                t0.elapsed().as_secs_f64() * 1e6 / iters as f64
            };
            // A warm-up round, then the cells in alternating rounds, median of `ROUNDS`.
            cells.iter().for_each(|(_, r)| {
                time(r);
            });
            let mut samples = vec![Vec::new(); cells.len()];
            for _ in 0..ROUNDS {
                for (s, (_, r)) in samples.iter_mut().zip(&cells) {
                    s.push(time(r));
                }
            }
            let mut line = format!("{name:<15}");
            for ((tag, _), s) in cells.iter().zip(&mut samples) {
                s.sort_by(f64::total_cmp);
                let us = s[s.len() / 2];
                line += &format!("  {tag:>8} {us:7.1} ({:3.0})", streamed as f64 / us / 1e3);
            }
            println!("{line}");
        }
    }
}

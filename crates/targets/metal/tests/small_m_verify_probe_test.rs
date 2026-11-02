// SPDX-License-Identifier: Apache-2.0
//! A small-M step's GEMM in a W4A8 bucket: the bf16 small-M twin (`affine_qmm_small_m_*`, the
//! 16-row tile) against the W4A8 GEMMs over the pre-pass's int8 rows (the bucket's 32-row tiled
//! and split-K GEMMs, and the 16-row split-K GEMM `W4a8Gemm::SmallM` at each split), at
//! Qwen3.8-27B's dense projections with a 4-16-token step's rows live in the 64-row bucket, and
//! which the wide class's lowering picks (`W4a8Gemm::for_small_m`, marked `*`). This is the sweep
//! `W4A8_SMALL_M_MIN_WAYS` and the wide class's route (`route_small_m`) were set on (40-core M5
//! Max); rerun it on a new chip before moving them. Each cell is µs a dispatch over one command
//! buffer streaming `BYTES_PER_BATCH` of weights, the dispatches a barrier apart and the weights
//! rotated over copies past the system cache, the cells in alternating rounds, median of
//! `ROUNDS`; GB/s is the codes + scales + biases a dispatch streams. The pre-pass (`quant`) is
//! timed alone: the W4A8 cells exclude it.
//!
//! Run: cargo test -p scratchy-target-metal --release --test small_m_verify_probe_test -- \
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
use scratchy_target_metal::tape::quantized::{
    DequantDtype, SMALL_M_SIMDGROUPS, SMALL_M_TILE_COLS, ScaleDtype, SmallMTile, W4a8Gemm,
    W4a8Rows, W4a8SplitK, W4a8Tile, small_m_kernel_static_name, w4a8_quant_static_name,
    w4a8_scratch_bytes,
};

/// Weight bytes one timed command buffer streams, so its submit costs a few percent at most.
const BYTES_PER_BATCH: usize = 4 << 30;

/// Weight bytes the copies span, past the system cache.
const ROTATED_BYTES: usize = 256 << 20;

const GROUP: u32 = 64;

/// Rounds of a shape's cells, after a warm-up round.
const ROUNDS: usize = 7;

/// The bucket, and the step's live rows in it: the small-M range's ends and its middle.
const BUCKET: u32 = 64;
const LIVE: [u32; 3] = [16, 8, 4];

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

/// A probe cell: its tag, symbol, columns and rows a threadgroup, threads, and whether it reads
/// the pre-pass's int8 rows (else the bf16 activation).
struct Cell {
    tag: String,
    symbol: &'static str,
    cols: u32,
    rows: u32,
    threads: u32,
    int8: bool,
}

#[test]
#[ignore = "measurement probe, not an assertion: run with --ignored --nocapture"]
fn small_m_steps_in_a_w4a8_bucket_at_qwen38_27b_shapes() {
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
            let x = shared_slice(&device, &bf16_row(BUCKET as usize * ku, 0.3));
            let scratch = shared_zeroed(&device, w4a8_scratch_bytes(BUCKET, k) as usize);
            let y = shared_zeroed(&device, BUCKET as usize * nu * 2);
            let mut constants = vec![
                ConstantValue::int(0, k as i32),
                ConstantValue::int(1, n as i32),
                ConstantValue::int(2, BUCKET as i32),
            ];
            constants.extend(AffineCodes::Offset8.constant());
            let picked = W4a8Gemm::for_small_m(n, k);
            let mut cells = vec![Cell {
                tag: "twin".into(),
                symbol: small_m_kernel_static_name(act, scale, GROUP, twin),
                cols: SMALL_M_TILE_COLS,
                rows: twin.rows(),
                threads: 32 * SMALL_M_SIMDGROUPS,
                int8: false,
            }];
            let gemms = (W4a8Tile::for_n(n).map(W4a8Gemm::Tiled).into_iter())
                .chain(W4a8SplitK::ALL.map(W4a8Gemm::SplitK))
                .chain(W4a8SplitK::ALL.map(W4a8Gemm::SmallM));
            cells.extend(gemms.map(|g| {
                let tag = match g {
                    W4a8Gemm::Tiled(t) => format!("tn{}", t.cols()),
                    W4a8Gemm::SplitK(s) => format!("k{}", s.ways()),
                    W4a8Gemm::SmallM(s) => format!("m16k{}", s.ways()),
                };
                Cell {
                    tag: if Some(g) == picked { tag + "*" } else { tag },
                    symbol: g.static_name(act, scale, GROUP),
                    cols: g.cols(),
                    rows: g.rows(),
                    threads: 32 * g.simdgroups(),
                    int8: true,
                }
            }));
            let pso = |symbol| {
                baked_build(
                    &cache,
                    &PipelineKey::new("quantized_qmm_nax", symbol, constants.clone()),
                )
                .expect(symbol)
            };
            let psos: Vec<_> = cells.iter().map(|c| pso(c.symbol)).collect();
            let quant = pso(w4a8_quant_static_name(W4a8Rows::Dense, act));
            let time = |cell: Option<usize>| {
                let mut batch = Mtl4DispatchBatch::begin(&device).expect("mtl4 batch");
                let dispatches = if cell.is_some() { iters } else { 512 };
                for i in 0..dispatches {
                    match cell {
                        Some(c) => {
                            let cell = &cells[c];
                            let (w, s, b) = &weights[i % copies];
                            let input = if cell.int8 { &scratch } else { &x };
                            batch.encode(
                                &psos[c],
                                &[(w, 0), (s, 1), (b, 2), (input, 3), (&y, 4)],
                                &[],
                                &[],
                                &[],
                                size(n / cell.cols, live.div_ceil(cell.rows)),
                                size(cell.threads, 1),
                            );
                        }
                        None => batch.encode(
                            &quant,
                            &[(&x, 0), (&scratch, 1)],
                            &[],
                            &[],
                            &[],
                            size((k / 64).div_ceil(16), live),
                            size(128, 1),
                        ),
                    }
                    batch.barrier();
                }
                let t0 = std::time::Instant::now();
                batch.commit(true);
                t0.elapsed().as_secs_f64() * 1e6 / dispatches as f64
            };
            // A warm-up round, then the cells in alternating rounds, median of `ROUNDS`.
            (0..cells.len()).for_each(|c| {
                time(Some(c));
            });
            let mut samples = vec![Vec::new(); cells.len()];
            let mut quant_us = Vec::new();
            for _ in 0..ROUNDS {
                for (c, s) in samples.iter_mut().enumerate() {
                    s.push(time(Some(c)));
                }
                quant_us.push(time(None));
            }
            let median = |s: &mut Vec<f64>| {
                s.sort_by(f64::total_cmp);
                s[s.len() / 2]
            };
            let mut line = format!("{name:<15} quant {:5.1}", median(&mut quant_us));
            for (cell, s) in cells.iter().zip(&mut samples) {
                let us = median(s);
                line += &format!(
                    "  {:>6} {us:7.1} ({:3.0})",
                    cell.tag,
                    streamed as f64 / us / 1e3
                );
            }
            println!("{line}");
        }
    }
}

// SPDX-License-Identifier: Apache-2.0
//! The affine matvec at 1-4 rows, plain and normalizing its input on load (`QmvEnds::norm`), at
//! decode shapes of Qwen3.6-35B-A3B (4-bit) and GLM-4.5-Air (3-bit). This is the sweep
//! `ModelFoldFacts::matvec_norms` (the norm folds on the one-row bucket alone) was measured on:
//! base M5, the plain wide matvec at 3-4 rows costs what one row costs, while normalizing on load
//! makes it ALU-bound (Qwen3.6's 248k-row lm_head at 4 rows 2137 vs 3302 µs; one row 2131 vs
//! 2134). Rerun it on a new chip before moving the fold. Each cell is µs a dispatch over one
//! command buffer of `ITERS` dispatches a barrier apart, the weights rotated over copies past the
//! system cache, best of 3.
//!
//! Run: cargo test -p scratchy-target-metal --release --test qmv_rows_probe_test -- \
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
use scratchy_target_metal::tape::ids::{KDimI32, LayerId, MDimI32, NDimI32};
use scratchy_target_metal::tape::kernel_constants::{
    AffineCodes, AffineQmmTConstants, AffineQmvConstants, AffineQmvWideConstants,
};
use scratchy_target_metal::tape::quantized::{
    DequantDtype, QmvKernel, ScaleDtype, pick_qmv_kernel, pick_qmv_kernel_wide, qmv_dispatch_shape,
    qmv_kernel_static_name,
};
use scratchy_target_metal::tape::step::{Eps, GainOffset, QmvEnds, RowNorm};

/// Dispatches per timed command buffer.
const ITERS: usize = 48;

/// Weight bytes the copies span, past the system cache.
const ROTATED_BYTES: usize = 96 << 20;

/// The group size of every shape's codes.
const GROUP: u32 = 64;

/// One matvec: `n` rows out of `k`, `bits`-bit codes.
struct Shape {
    name: &'static str,
    n: u32,
    k: u32,
    bits: u32,
}

const fn shape(name: &'static str, n: u32, k: u32, bits: u32) -> Shape {
    Shape { name, n, k, bits }
}

const SHAPES: &[Shape] = &[
    shape("qwen3.6 lm_head", 248320, 2048, 4),
    shape("qwen3.6 gdn in_proj", 12288, 2048, 4),
    shape("qwen3.6 q+gate", 8192, 2048, 4),
    shape("glm lm_head", 151552, 4096, 3),
    shape("glm q_proj", 12288, 4096, 3),
];

fn bf16_row(len: usize, v: f32) -> Vec<u16> {
    (0..len)
        .map(|i| bf16::from_f32(v * (1.0 + 0.001 * (i % 97) as f32)).to_bits())
        .collect()
}

fn size((width, height, depth): (u32, u32, u32)) -> MTLSize {
    MTLSize {
        width: width as usize,
        height: height as usize,
        depth: depth as usize,
    }
}

#[test]
#[ignore = "measurement probe, not an assertion: run with --ignored --nocapture"]
fn qmv_rows_plain_and_normalizing_on_load() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("pipeline cache");
    let normed = QmvEnds {
        norm: Some(RowNorm {
            layer: LayerId(0),
            eps: Eps(1e-6),
            offset: GainOffset(1.0),
        }),
        ..QmvEnds::default()
    };
    println!("µs a dispatch: rows 1-4, plain (p) and normalizing on load (n)");
    for s in SHAPES {
        let (n, k) = (s.n as usize, s.k as usize);
        let bytes = n * k * s.bits as usize / 8;
        let copies = ROTATED_BYTES / bytes + 1;
        let groups = n * k / GROUP as usize;
        let weights: Vec<_> = (0..copies)
            .map(|c| {
                let codes: Vec<u32> = (0..bytes as u32 / 4)
                    .map(|i| i.wrapping_mul(2654435761).wrapping_add(c as u32))
                    .collect();
                (
                    shared_slice(&device, &codes),
                    shared_slice(&device, &bf16_row(groups, 0.002)),
                    shared_slice(&device, &bf16_row(groups, -0.015)),
                )
            })
            .collect();
        let x = shared_slice(&device, &bf16_row(4 * k, 0.5));
        let gain = shared_slice(&device, &bf16_row(k, 0.1));
        let y = shared_zeroed(&device, 4 * n * 2);
        let mut line = format!("{:<20}", s.name);
        for m in 1..=4u32 {
            for ends in [QmvEnds::default(), normed] {
                let kernel = match m {
                    1 => pick_qmv_kernel(s.n, s.k, s.bits),
                    _ => pick_qmv_kernel_wide(s.n, s.k, s.bits, m, true),
                };
                let name = qmv_kernel_static_name(
                    kernel,
                    DequantDtype::Bf16,
                    ScaleDtype::Bf16,
                    s.bits,
                    GROUP,
                );
                let (k_dim, n_dim) = (KDimI32(s.k as i32), NDimI32(s.n as i32));
                let codes = AffineCodes::AsWritten;
                let mut constants: Vec<ConstantValue> = match kernel {
                    QmvKernel::Wide { .. } => AffineQmvWideConstants {
                        k: k_dim,
                        n: n_dim,
                        m: MDimI32(m as i32),
                        codes,
                    }
                    .into(),
                    _ => AffineQmvConstants {
                        k: k_dim,
                        n: n_dim,
                        codes,
                    }
                    .into(),
                };
                constants.extend(Vec::<ConstantValue>::from(ends));
                let pso = baked_build(&cache, &PipelineKey::new("quantized_qmv", name, constants))
                    .expect(name);
                let (grid, threads) = qmv_dispatch_shape(kernel, m, s.n, 1);
                let time = || {
                    let mut batch = Mtl4DispatchBatch::begin(&device).expect("mtl4 batch");
                    for i in 0..ITERS {
                        let (w, scales, biases) = &weights[i % copies];
                        let binds = [
                            (w, 0),
                            (scales, 1),
                            (biases, 2),
                            (&x, 3),
                            (&y, 4),
                            (&gain, 15),
                        ];
                        batch.encode(&pso, &binds, &[], &[], &[], size(grid), size(threads));
                        batch.barrier();
                    }
                    let t0 = std::time::Instant::now();
                    batch.commit(true);
                    t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64
                };
                // Warm-up batch (first-run pipeline + residency), then best of 3.
                time();
                let best = (0..3).map(|_| time()).fold(f64::MAX, f64::min);
                let tag = if ends.norm.is_some() { 'n' } else { 'p' };
                line += &format!("  {m}{tag} {best:7.1}");
            }
        }
        println!("{line}   ({:.1} MB of codes)", bytes as f64 / 1e6);
    }
}

/// Rounds of a shape's by-bucket cells, after a warm-up round.
const ROUNDS: usize = 5;

/// Qwen3.6-35B-A3B's 4-bit dense projections a decode bucket's matvecs run.
const BUCKET_SHAPES: &[Shape] = &[
    shape("qwen3.6 lm_head", 248320, 2048, 4),
    shape("qwen3.6 gdn in_proj", 12288, 2048, 4),
    shape("qwen3.6 q+gate", 8192, 2048, 4),
    shape("qwen3.6 gdn out_proj", 2048, 4096, 4),
];

/// A decode bucket's dense matvec, as the tape bakes it at the bucket's rows and codes: the wide
/// matvec (`AffineQmvWide`, its rows in tiles of at most 5 — one weight stream at 4 rows, two at
/// 8) against the small-M matrix-unit twin (`AffineQmmSmallM`, one 8-row tile), which the tape
/// runs instead on a `SMALL_M_TOKENS` step where the bucket's wide matvec streams the weights more
/// than once (`route_small_m`). A shape's four cells run in alternating rounds, best of `ROUNDS`.
/// NAX devices only.
#[test]
#[ignore = "measurement probe, not an assertion: run with --ignored --nocapture"]
fn qmv_wide_against_small_m_twin_by_bucket() {
    use scratchy_target_metal::tape::quantized::{
        SMALL_M_SIMDGROUPS, SMALL_M_TILE_COLS, SmallMTile, small_m_kernel_static_name,
    };
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
    println!("µs a dispatch at a bucket's rows: wide matvec (w) and small-M twin (t)");
    for s in BUCKET_SHAPES {
        let (n, k) = (s.n as usize, s.k as usize);
        let bytes = n * k * s.bits as usize / 8;
        let copies = ROTATED_BYTES / bytes + 1;
        let groups = n * k / GROUP as usize;
        let weights: Vec<_> = (0..copies)
            .map(|c| {
                let codes: Vec<u32> = (0..bytes as u32 / 4)
                    .map(|i| i.wrapping_mul(2654435761).wrapping_add(c as u32))
                    .collect();
                (
                    shared_slice(&device, &codes),
                    shared_slice(&device, &bf16_row(groups, 0.002)),
                    shared_slice(&device, &bf16_row(groups, -0.015)),
                )
            })
            .collect();
        let x = shared_slice(&device, &bf16_row(8 * k, 0.5));
        let y = shared_zeroed(&device, 8 * n * 2);
        let mut line = format!("{:<22}", s.name);
        // Both kernels read the codes as the tape's do on this device (offset-8 wherever the twin
        // exists).
        let codes = AffineCodes::of(Some(&di.profile), s.bits);
        let (k_dim, n_dim) = (KDimI32(s.k as i32), NDimI32(s.n as i32));
        let mut cells = Vec::new();
        for bucket in [4u32, 8] {
            let m = MDimI32(bucket as i32);
            let kernel = pick_qmv_kernel_wide(s.n, s.k, s.bits, bucket, true);
            let wide_name =
                qmv_kernel_static_name(kernel, DequantDtype::Bf16, ScaleDtype::Bf16, s.bits, GROUP);
            let wide_constants: Vec<ConstantValue> = AffineQmvWideConstants {
                k: k_dim,
                n: n_dim,
                m,
                codes,
            }
            .into();
            let wide = baked_build(
                &cache,
                &PipelineKey::new("quantized_qmv", wide_name, wide_constants),
            )
            .expect(wide_name);
            cells.push((
                format!("{bucket}w"),
                wide,
                qmv_dispatch_shape(kernel, bucket, s.n, 1),
            ));
            let tile = SmallMTile::for_bucket(bucket).expect("a small-M bucket");
            let twin_name =
                small_m_kernel_static_name(DequantDtype::Bf16, ScaleDtype::Bf16, GROUP, tile);
            let twin_constants: Vec<ConstantValue> = AffineQmmTConstants {
                k: k_dim,
                n: n_dim,
                m,
                codes,
            }
            .into();
            let twin = baked_build(
                &cache,
                &PipelineKey::new("quantized_qmm_nax", twin_name, twin_constants),
            )
            .expect(twin_name);
            cells.push((
                format!("{bucket}t"),
                twin,
                (
                    (s.n / SMALL_M_TILE_COLS, bucket.div_ceil(tile.rows()), 1),
                    (32 * SMALL_M_SIMDGROUPS, 1, 1),
                ),
            ));
        }
        let time = |pso, (grid, threads)| {
            let mut batch = Mtl4DispatchBatch::begin(&device).expect("mtl4 batch");
            for i in 0..ITERS {
                let (w, scales, biases) = &weights[i % copies];
                let binds = [(w, 0), (scales, 1), (biases, 2), (&x, 3), (&y, 4)];
                batch.encode(pso, &binds, &[], &[], &[], size(grid), size(threads));
                batch.barrier();
            }
            let t0 = std::time::Instant::now();
            batch.commit(true);
            t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64
        };
        // The shape's cells in alternating rounds, a warm-up round first: a GPU clock band's
        // shift lands on every cell of a round, not on whichever cell it falls in.
        let mut best = vec![f64::MAX; cells.len()];
        for round in 0..=ROUNDS {
            for (b, (_, pso, shape)) in best.iter_mut().zip(&cells) {
                let us = time(pso, *shape);
                if round > 0 {
                    *b = b.min(us);
                }
            }
        }
        for ((tag, _, _), us) in cells.iter().zip(&best) {
            line += &format!("  {tag} {us:8.1}");
        }
        println!("{line}   ({:.1} MB of codes)", bytes as f64 / 1e6);
    }
}

/// What folding a layer's RMSNorm into the projections reading it buys on a 40-core chip emulated by
/// width (columns / 4 on a 10-core base M5): µs a group, min / median / max of `ROUNDS`, at the bucket's
/// rows, DRAM-cold. Each consumer set runs as the tape runs it, every member between the same barriers:
/// - `norm + plain`: the RMSNorm, a barrier, then the plain wide matvecs;
/// - `normed`: the wide matvecs normalizing their input as they load it (`QmvEnds::norm`), no norm.
///
/// The sets are Qwen3.6-35B-A3B's: the GDN input norm's readers (qkv, z, a, b) and the post-attention
/// norm's weight-carrying readers (shared gate, up). NAX devices only.
#[test]
#[ignore = "measurement probe, not an assertion: run with --ignored --nocapture"]
fn norm_folded_into_its_readers_on_a_chip_emulated_by_width() {
    use scratchy_target_metal::tape::ids::{BucketM, QSize, RmsNormEps};
    use scratchy_target_metal::tape::kernel_constants::RmsNormConstants;
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    if !scratchy_target_metal::targets::is_nax_capable(di.profile.generation) {
        eprintln!("skipping: no NAX matrix unit");
        return;
    }
    let device = di.device.clone();
    let cores = scratchy_target_metal::device::gpu_cores(&device)
        .expect("the device's GPU cores")
        .get() as usize;
    let width = 40 / cores;
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("pipeline cache");
    let codes = AffineCodes::of(Some(&di.profile), 4);
    let hidden = 2048u32;
    let normed = QmvEnds {
        norm: Some(RowNorm {
            layer: LayerId(0),
            eps: Eps(1e-6),
            offset: GainOffset(0.0),
        }),
        ..QmvEnds::default()
    };
    println!(
        "µs a group, min/median/max of {ROUNDS} rounds, a 40-core chip emulated by width on {cores} cores \
         (columns / {width})"
    );
    let sets: &[(&str, &[u32])] = &[
        ("gdn input norm: qkv | z | a | b", &[8192, 4096, 32, 32]),
        ("post-attention norm: shared gate | up", &[512, 512]),
    ];
    for &(name, columns) in sets {
        for rows in [3u32, 4] {
            // Each member at the emulated width (a narrow one at its own).
            let ns: Vec<u32> = columns
                .iter()
                .map(|&n| {
                    if n / width as u32 >= 32 {
                        n / width as u32
                    } else {
                        n
                    }
                })
                .collect();
            let bytes: usize = ns.iter().map(|&n| (n * hidden / 2) as usize).sum();
            let copies = ROTATED_BYTES / bytes + 1;
            let iters = ((1usize << 30) / bytes).clamp(48, 4096);
            let weights: Vec<Vec<_>> = (0..copies)
                .map(|c| {
                    ns.iter()
                        .map(|&n| {
                            let words = (n * hidden / 8) as usize;
                            let codes: Vec<u32> = (0..words as u32)
                                .map(|i| i.wrapping_mul(2654435761).wrapping_add(c as u32))
                                .collect();
                            let groups = (n * hidden / GROUP) as usize;
                            (
                                shared_slice(&device, &codes),
                                shared_slice(&device, &bf16_row(groups, 0.002)),
                                shared_slice(&device, &bf16_row(groups, -0.015)),
                            )
                        })
                        .collect()
                })
                .collect();
            let x = shared_slice(&device, &bf16_row(4 * hidden as usize, 0.5));
            let xn = shared_zeroed(&device, 4 * hidden as usize * 2);
            let gain = shared_slice(&device, &bf16_row(hidden as usize, 0.1));
            let ys: Vec<_> = ns
                .iter()
                .map(|&n| shared_zeroed(&device, 4 * n as usize * 2))
                .collect();
            let matvec = |n: u32, ends: QmvEnds| {
                let kernel = pick_qmv_kernel_wide(n, hidden, 4, rows, true);
                let name =
                    qmv_kernel_static_name(kernel, DequantDtype::Bf16, ScaleDtype::Bf16, 4, GROUP);
                let mut constants: Vec<ConstantValue> = AffineQmvWideConstants {
                    k: KDimI32(hidden as i32),
                    n: NDimI32(n as i32),
                    m: MDimI32(rows as i32),
                    codes,
                }
                .into();
                constants.extend(Vec::<ConstantValue>::from(ends));
                let pso = baked_build(&cache, &PipelineKey::new("quantized_qmv", name, constants))
                    .expect(name);
                (pso, qmv_dispatch_shape(kernel, rows, n, 1))
            };
            let plain: Vec<_> = ns.iter().map(|&n| matvec(n, QmvEnds::default())).collect();
            let folded: Vec<_> = ns.iter().map(|&n| matvec(n, normed)).collect();
            let norm_constants: Vec<ConstantValue> = RmsNormConstants {
                bucket_m: BucketM(rows),
                q_size: QSize(hidden),
                rms_norm_eps: RmsNormEps(1e-6),
                weight_offset: 0.0,
            }
            .into();
            let norm_name = "rmsnorm_bf16_s_bf16_specialized";
            let norm = baked_build(
                &cache,
                &PipelineKey::new("rmsnorm", norm_name, norm_constants),
            )
            .expect(norm_name);
            // 0: the norm, a barrier, the plain readers; 1: the readers normalizing on load.
            let time = |cell: usize| {
                let mut batch = Mtl4DispatchBatch::begin(&device).expect("mtl4 batch");
                for i in 0..iters {
                    let copy = &weights[i % copies];
                    if cell == 0 {
                        batch.encode(
                            &norm,
                            &[(&xn, 0), (&x, 1), (&gain, 2)],
                            &[],
                            &[],
                            &[],
                            size((rows, 1, 1)),
                            size((1024, 1, 1)),
                        );
                        batch.barrier();
                    }
                    let readers = if cell == 0 { &plain } else { &folded };
                    for (((pso, (grid, threads)), (w, scales, biases)), y) in
                        readers.iter().zip(copy).zip(&ys)
                    {
                        let input = if cell == 0 { &xn } else { &x };
                        let binds = [
                            (w, 0),
                            (scales, 1),
                            (biases, 2),
                            (input, 3),
                            (y, 4),
                            (&gain, 15),
                        ];
                        batch.encode(pso, &binds, &[], &[], &[], size(*grid), size(*threads));
                    }
                    batch.barrier();
                }
                let t0 = std::time::Instant::now();
                batch.commit(true);
                t0.elapsed().as_secs_f64() * 1e6 / iters as f64
            };
            let mut samples = [Vec::new(), Vec::new()];
            for round in 0..=ROUNDS {
                for (cell, s) in samples.iter_mut().enumerate() {
                    let us = time(cell);
                    if round > 0 {
                        s.push(us);
                    }
                }
            }
            println!("{name}, {rows} rows (columns {ns:?}):");
            for (tag, s) in ["norm + plain", "normed"].iter().zip(&mut samples) {
                s.sort_by(f64::total_cmp);
                println!(
                    "    {tag:<14} {:8.1} {:8.1} {:8.1}",
                    s[0],
                    s[s.len() / 2],
                    s[s.len() - 1]
                );
            }
        }
    }
}

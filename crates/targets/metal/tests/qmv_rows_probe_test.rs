// SPDX-License-Identifier: Apache-2.0
//! The affine matvec at 1-4 rows, plain and normalizing its input on load (`QmvEnds::norm`), at
//! decode shapes of Qwen3.6-35B-A3B (4-bit), GLM-4.5-Air (3-bit), gpt-oss (4-bit) and its P-EAGLE
//! drafter (3-bit MLP), and past one row the fast kernel's fold twin beside the wide pick. On a
//! base M5 gpt-oss's 4-bit shapes cost at 4 rows what one row costs (q_proj 72.9 vs 72.8 µs), the
//! drafter's 3-bit MLP 35% more (gate/up 151 → 206 µs), the fold twin more again. This is the sweep
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
    AffineCodes, AffineQmvConstants, AffineQmvWideConstants,
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
    shape("gpt-oss lm_head", 201088, 2880, 4),
    shape("gpt-oss q_proj", 4096, 2880, 4),
    shape("gpt-oss k/v_proj", 512, 2880, 4),
    shape("gpt-oss o_proj", 2880, 4096, 4),
    shape("p-eagle mid q_proj", 4096, 5760, 4),
    shape("p-eagle mlp gate/up", 14336, 2880, 3),
    shape("p-eagle mlp down", 2880, 14336, 3),
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
    println!(
        "µs a dispatch: rows 1-4, plain (p), normalizing on load (n), and past one row the fast \
         kernel's fold twin (f)"
    );
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
            // Past one row, the fast kernel's fold twin (`wide_ok` false, every row streaming
            // the weights) beside the wide matvec the gen-15 pick makes.
            let twin = (m > 1).then_some((QmvEnds::default(), false));
            for (ends, wide_ok) in [(QmvEnds::default(), true), (normed, true)]
                .into_iter()
                .chain(twin)
            {
                let kernel = match m {
                    1 => pick_qmv_kernel(s.n, s.k, s.bits),
                    _ => pick_qmv_kernel_wide(s.n, s.k, s.bits, m, wide_ok),
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
                let tag = match (ends.norm.is_some(), wide_ok) {
                    (true, _) => 'n',
                    (false, true) => 'p',
                    (false, false) => 'f',
                };
                line += &format!("  {m}{tag} {best:7.1}");
            }
        }
        println!("{line}   ({:.1} MB of codes)", bytes as f64 / 1e6);
    }
}

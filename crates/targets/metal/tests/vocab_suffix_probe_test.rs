// SPDX-License-Identifier: Apache-2.0
//! What a wide head pass costs two ways at Qwen3.6's lm_head (248320 x 2048, 4-bit g64, the head's
//! prefix 65536): the whole vocabulary in one matvec and `argmax` (a head drafting over every id),
//! against the prefix's matvec and [`SuffixDraft`] (the suffix's matvec and `argmax_split`).
//! µs a pass over one command buffer of `ITERS` passes, the weights rotated past the system cache,
//! best of 3.
//!
//! Run: cargo test -p scratchy-target-metal --release --test vocab_suffix_probe_test -- \
//!     --ignored --nocapture

#![cfg(target_os = "macos")]

use objc2_metal::{MTLBuffer as _, MTLSize};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::{Mtl4DispatchBatch, shared_slice, shared_zeroed};
use scratchy_target_metal::off_tape::VocabSuffix;
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use scratchy_target_metal::tape::ids::{KDimI32, MDimI32, NDimI32};
use scratchy_target_metal::tape::kernel_constants::{
    AffineCodes, AffineQmvConstants, AffineQmvWideConstants,
};
use scratchy_target_metal::tape::lowered::MetalDtype;
use scratchy_target_metal::tape::quantized::{
    DequantDtype, QmvKernel, ScaleDtype, pick_qmv_kernel, pick_qmv_kernel_wide, qmv_dispatch_shape,
    qmv_kernel_static_name,
};
use scratchy_target_metal::vocab_suffix::{LmHead, LmHeadPart, PassPick, SuffixDraft};

const V: u32 = 248_320;
const H: u32 = 2048;
const P: u32 = 65_536;
const ITERS: usize = 24;
/// Copies of the lm_head the passes rotate over: past the system cache.
const COPIES: usize = 2;

fn size((width, height, depth): (u32, u32, u32)) -> MTLSize {
    MTLSize {
        width: width as usize,
        height: height as usize,
        depth: depth as usize,
    }
}

fn qmv_key(n: u32, m: u32) -> (PipelineKey, QmvKernel) {
    let kernel = match m {
        1 => pick_qmv_kernel(n, H, 4),
        _ => pick_qmv_kernel_wide(n, H, 4, m, true),
    };
    let name = qmv_kernel_static_name(kernel, DequantDtype::Bf16, ScaleDtype::Bf16, 4, 64);
    let (k, n_dim) = (KDimI32(H as i32), NDimI32(n as i32));
    let codes = AffineCodes::AsWritten;
    let constants: Vec<ConstantValue> = match kernel {
        QmvKernel::Wide { .. } => AffineQmvWideConstants {
            k,
            n: n_dim,
            m: MDimI32(m as i32),
            codes,
        }
        .into(),
        _ => AffineQmvConstants { k, n: n_dim, codes }.into(),
    };
    (PipelineKey::new("quantized_qmv", name, constants), kernel)
}

#[test]
#[ignore = "measurement probe, not an assertion: run with --ignored --nocapture"]
fn a_wide_pass_whole_vocabulary_vs_prefix_and_suffix() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("pipeline cache");
    let build = |key: &PipelineKey| baked_build(&cache, key).expect("pipeline");
    let suffix = VocabSuffix {
        prefix: P,
        rows: V - P,
        hidden: H,
        bits: 4,
        group: 64,
        scale: ScaleDtype::Bf16,
        verify_rows: 8,
        window: 4,
    };
    let keys = suffix.kernel_keys(MetalDtype::Bf16);
    let heads: Vec<_> = (0..COPIES)
        .map(|c| {
            let codes: Vec<u32> = (0..V * H / 8)
                .map(|i| i.wrapping_mul(2654435761).wrapping_add(c as u32))
                .collect();
            let groups = (V * H / 64) as usize;
            let scales: Vec<u16> = (0..groups).map(|i| 0x3c00 + (i % 7) as u16).collect();
            (
                shared_slice(&device, &codes),
                shared_slice(&device, &scales),
                shared_slice(&device, &scales),
            )
        })
        .collect();
    let x = shared_slice(&device, &vec![0x3c00u16; 8 * H as usize]);
    let full = shared_zeroed(&device, 8 * V as usize * 2);
    let prefix_logits = shared_zeroed(&device, 8 * P as usize * 2);
    let picked = shared_zeroed(&device, 8 * 4);
    println!(
        "µs a wide head pass, m rows: whole vocabulary (one matvec + argmax) | prefix + suffix"
    );
    for m in [1u32, 2, 3] {
        let batch_const = shared_slice(&device, &[m]);
        let argmax = build(&PipelineKey::new(
            "argmax",
            "argmax_bf16",
            vec![ConstantValue::uint(0, V)],
        ));
        let one = |w: usize| MTLSize {
            width: w,
            height: 1,
            depth: 1,
        };
        let whole = || {
            let (key, kernel) = qmv_key(V, m);
            let pso = build(&key);
            let (grid, threads) = qmv_dispatch_shape(kernel, m, V, 1);
            let mut batch = Mtl4DispatchBatch::begin(&device).expect("batch");
            for i in 0..ITERS {
                let (w, s, b) = &heads[i % COPIES];
                batch.encode(
                    &pso,
                    &[(w, 0), (s, 1), (b, 2), (&x, 3), (&full, 4)],
                    &[],
                    &[],
                    &[],
                    size(grid),
                    size(threads),
                );
                batch.barrier();
                batch.encode(
                    &argmax,
                    &[(&full, 0), (&picked, 1)],
                    &[(m, 2)],
                    &[],
                    &[],
                    one(m as usize),
                    one(1024),
                );
                batch.barrier();
            }
            let t0 = std::time::Instant::now();
            batch.commit(true);
            t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64
        };
        let split = || {
            let (key, kernel) = qmv_key(P, m);
            let pso = build(&key);
            let (grid, threads) = qmv_dispatch_shape(kernel, m, P, 1);
            let mut batch = Mtl4DispatchBatch::begin(&device).expect("batch");
            let res = batch.residency();
            let pins: Vec<_> = (heads.iter())
                .flat_map(|(w, s, b)| [w, s, b])
                .chain([&x, &prefix_logits, &picked, &batch_const])
                .map(|b| res.pin(b.clone()))
                .collect();
            let drafts: Vec<_> = (heads.iter())
                .map(|(w, s, b)| {
                    let part = |buffer: &scratchy_target_metal::mtl4_dispatch::Buffer| LmHeadPart {
                        buffer: buffer.clone(),
                        offset: 0,
                    };
                    let logits = res.pin(shared_zeroed(&device, 8 * (V - P) as usize * 2));
                    SuffixDraft::new(
                        suffix,
                        LmHead {
                            codes: part(w),
                            scales: part(s),
                            biases: part(b),
                        },
                        (keys.qmv.as_written.iter())
                            .map(|k| Box::new(build(k)))
                            .collect(),
                        Box::new(build(&keys.argmax)),
                        logits,
                    )
                    .expect("suffix draft")
                })
                .collect();
            for i in 0..ITERS {
                let (w, s, b) = &heads[i % COPIES];
                batch.encode(
                    &pso,
                    &[(w, 0), (s, 1), (b, 2), (&x, 3), (&prefix_logits, 4)],
                    &[],
                    &[],
                    &[],
                    size(grid),
                    size(threads),
                );
                let pick = PassPick {
                    hidden: x.gpuAddress(),
                    prefix_logits: prefix_logits.gpuAddress(),
                    picked: picked.gpuAddress(),
                    batch_const: batch_const.gpuAddress(),
                    batch: m as usize,
                };
                drafts[i % COPIES]
                    .encode(batch.encoder(), &device, pick)
                    .expect("encode");
            }
            let t0 = std::time::Instant::now();
            batch.commit(true);
            drop(pins);
            t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64
        };
        whole();
        split();
        let best = |f: &dyn Fn() -> f64| (0..3).map(|_| f()).fold(f64::MAX, f64::min);
        println!("  m={m}: {:8.1} | {:8.1}", best(&whole), best(&split));
    }
}

// Standalone timing of the sampler's stages on one gemma-4-26b-shaped logits
// row (vocab 262144, bf16, one job) — the exact sizes `scr chat`'s decode pays.
// Times the full prepared pipeline (cast → penalties? → softmax → descent →
// compaction → finalize) as the worker encodes it, plus the cast alone and a
// 4-job batch, at the default sampling settings (temp=1, top_k=64, top_p=0.95).
//
// Run:
//   cargo test --release -p scratchy-target-metal --test sampling_bench -- --nocapture
use objc2_metal::{MTLBuffer as _, MTLCreateSystemDefaultDevice};
use scratchy_core_common::GpuSampleParams;
use scratchy_target_metal::mtl4_dispatch::{
    Device, Mtl4DispatchBatch, shared_slice, shared_zeroed,
};
use scratchy_target_metal::sampling::{CastDtype, PendingSampler, SamplerKernels};
use std::time::Instant;

fn one_row_logits(vocab: u32) -> Vec<u16> {
    // Realistic spread: mostly-negative noise plus a few strong tokens, so the
    // descent's histogram rounds count a real distribution.
    let mut rng: u32 = 0x1234_5678;
    (0..vocab)
        .map(|i| {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let noise = ((rng % 2000) as i32 - 1500) as f32 / 100.0;
            let v = if i == 1000 {
                20.0
            } else if i % 4096 == 0 {
                10.0
            } else {
                noise
            };
            half::bf16::from_f32(v).to_bits()
        })
        .collect()
}

fn params(nrows: usize, top_k: i32, top_p: f32) -> GpuSampleParams {
    GpuSampleParams {
        row_indices: (0..nrows as u32).collect(),
        temperatures: vec![1.0; nrows],
        top_ks: vec![top_k; nrows],
        top_ps: vec![top_p; nrows],
        min_ps: vec![0.0; nrows],
        uniforms: vec![0.5; nrows],
        ..Default::default()
    }
}

/// Median host wall time (µs) of prepare+encode+commit(wait) over `reps` runs
/// after `warm` warmups — an upper bound on the GPU-only time (includes the
/// host-side encode + event wait).
#[allow(clippy::too_many_arguments)]
fn time_run(
    device: &Device,
    kernels: &SamplerKernels,
    logits: &[u16],
    nrows: u32,
    vocab: u32,
    top_k: i32,
    top_p: f32,
    warm: usize,
    reps: usize,
) -> f64 {
    let mut samples = Vec::with_capacity(reps);
    for r in 0..(warm + reps) {
        let p = params(nrows as usize, top_k, top_p);
        let t0 = Instant::now();
        let logits_buf = shared_slice(device, logits);
        let batch = Mtl4DispatchBatch::begin(device).expect("batch");
        let pending = {
            let res = batch.residency();
            let pin = res.pin(logits_buf.clone());
            let pending = PendingSampler::prepare(device, res, &p, nrows, vocab, CastDtype::Bf16);
            std::mem::forget(pin);
            pending
        };
        let enc = batch.encoder();
        pending.encode_into(enc, logits_buf.gpuAddress(), kernels);
        let t_encode = t0.elapsed().as_secs_f64() * 1e6;
        let t1 = Instant::now();
        batch.commit(true);
        let us = t1.elapsed().as_secs_f64() * 1e6;
        if r >= warm {
            samples.push(us);
        }
        if r == warm + reps - 1 {
            println!("  (host prepare+encode: {t_encode:.1} µs, commit-wait: {us:.1} µs)");
        }
        if r >= warm {
            samples.push(us);
        }
    }
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

#[test]
fn time_sampler_stages() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        panic!("no Metal device");
    };
    let vocab: u32 = 262_144;
    let kernels = SamplerKernels::new(&device).expect("sampler kernels");
    let logits = one_row_logits(vocab);
    let _out = shared_zeroed(&device, 4);

    let us = time_run(&device, &kernels, &logits, 1, vocab, 64, 0.95, 3, 10);
    println!("sample (top_k=64, top_p=0.95, 1 row, bf16): {us:8.1} µs");

    let us_k0 = time_run(&device, &kernels, &logits, 1, vocab, 0, 1.0, 3, 10);
    println!("sample (top_k=0 → 1024 cap):                {us_k0:8.1} µs");

    let us4 = time_run(&device, &kernels, &logits, 4, vocab, 64, 0.95, 3, 10);
    println!("sample ×4 jobs:                             {us4:8.1} µs");
}

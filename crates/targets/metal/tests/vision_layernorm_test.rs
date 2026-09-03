// SPDX-License-Identifier: Apache-2.0
//! Golden test for `vision_layernorm` (Qwen3.5-VL / Qwen3-VL ViT LayerNorm).
//!
//! Dispatched standalone via `SpecializedPipelineCache` (production path), f32,
//! on the MTL4 command path (see `common::dispatch_threadgroups`).
//! Reference = the affine LayerNorm with biased variance + eps-inside-sqrt,
//! independently verified == `mlx.nn.LayerNorm` (max_abs_err 2.4e-7).

mod common;

use objc2_metal::MTLSize;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};

/// y = (x-mean)*rsqrt(var+eps)*w + b; var = E[x^2]-mean^2 (biased). [M, D].
fn ln_ref(x: &[f32], w: &[f32], b: &[f32], m: usize, d: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; m * d];
    for r in 0..m {
        let row = &x[r * d..][..d];
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = (var + eps).sqrt().recip();
        for i in 0..d {
            out[r * d + i] = (row[i] - mean) * inv * w[i] + b[i];
        }
    }
    out
}

#[test]
fn vision_layernorm_matches_reference() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone())
        .expect("compile standard shaders");

    // Production hidden (1152), a handful of rows.
    let (m, d) = (5usize, 1152usize);
    let eps = 1e-6f32;
    let n = m * d;
    let x: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.017).sin() * 1.3 + 0.2)
        .collect();
    let w: Vec<f32> = (0..d)
        .map(|i| 1.0 + ((i as f32) * 0.003).cos() * 0.1)
        .collect();
    let b: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.005).sin() * 0.05).collect();

    let want = ln_ref(&x, &w, &b, m, d, eps);

    let key = PipelineKey::new(
        "vision_layernorm",
        "vision_layernorm_f32_s_f32",
        vec![
            ConstantValue::uint(0, m as u32),
            ConstantValue::uint(1, d as u32),
            ConstantValue::float(2, eps),
        ],
    );
    let pipeline = cache.get_or_build(&key).expect("vision_layernorm pipeline");

    let x_buf = common::shared_slice(&device, &x);
    let w_buf = common::shared_slice(&device, &w);
    let b_buf = common::shared_slice(&device, &b);
    let out_buf = common::shared_zeroed(&device, n * std::mem::size_of::<f32>());

    // Binding contract: buffer(0)=out, buffer(1)=x, buffer(2)=w, buffer(3)=b.
    // M/D/eps are function constants on the pipeline, not buffer args.
    // One threadgroup per row; 256 threads stride over hidden.
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[&out_buf, &x_buf, &w_buf, &b_buf],
        MTLSize {
            width: m,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    let got: Vec<f32> = common::read_slice(&out_buf, n);
    let err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(err < 1e-4, "vision_layernorm max_abs_err={err}");
}

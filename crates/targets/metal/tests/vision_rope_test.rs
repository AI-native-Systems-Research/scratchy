// SPDX-License-Identifier: Apache-2.0
//! Golden test for `vision_rope_2d` (Qwen3.5-VL / Qwen3-VL ViT 2D RoPE).
//!
//! The kernel is dispatched standalone via `SpecializedPipelineCache` +
//! `get_or_build` (the production path), f32 instantiation, on the MTL4
//! tape (`common::dispatch_threadgroups`). The reference is the GPT-NeoX
//! `rotate_half` rope from mlx-vlm
//! `qwen3_vl/vision.py::apply_rotary_pos_emb_vision`, transcribed here and
//! independently verified == the mlx-vlm golden fixture (rope_block0_q:
//! max_abs_err 9.5e-7, cosine 1.0) — see vllm-rs/tools/vision_parity.

mod common;

use objc2_metal::MTLSize;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};

/// Interleaved (GPT-J / "traditional") rope reference — MoonViT / LocateAnything
/// `apply_rope` (complex multiply on adjacent pairs):
///   out[2i]   = x[2i]*cos(f_i) - x[2i+1]*sin(f_i)
///   out[2i+1] = x[2i]*sin(f_i) + x[2i+1]*cos(f_i),  f_i = freqs[t*half + i].
/// Independently verified == the mlx-vlm LocateAnything golden fixture
/// (rope_block0_q: max_abs_err 4.8e-7, cosine 1.0) — see tools/vision_parity.
fn rope_ref_interleaved(x: &[f32], freqs: &[f32], l: usize, h: usize, d: usize) -> Vec<f32> {
    let half = d / 2;
    let mut out = vec![0f32; l * h * d];
    for t in 0..l {
        for hh in 0..h {
            let base = (t * h + hh) * d;
            for i in 0..half {
                let f = freqs[t * half + i];
                let (c, s) = (f.cos(), f.sin());
                let (x0, x1) = (x[base + 2 * i], x[base + 2 * i + 1]);
                out[base + 2 * i] = x0 * c - x1 * s;
                out[base + 2 * i + 1] = x0 * s + x1 * c;
            }
        }
    }
    out
}

/// NeoX rotate_half rope reference: out[t,h,d] = x*cos(f) + rotate_half(x)*sin(f),
/// f = freqs[t*half + d%half]; rotate_half = concat(-x[half:], x[:half]).
fn rope_ref(x: &[f32], freqs: &[f32], l: usize, h: usize, d: usize) -> Vec<f32> {
    let half = d / 2;
    let mut out = vec![0f32; l * h * d];
    for t in 0..l {
        for hh in 0..h {
            let base = (t * h + hh) * d;
            for dd in 0..d {
                let f = freqs[t * half + (dd % half)];
                let (c, s) = (f.cos(), f.sin());
                let partner = if dd < half {
                    -x[base + dd + half]
                } else {
                    x[base + dd - half]
                };
                out[base + dd] = x[base + dd] * c + partner * s;
            }
        }
    }
    out
}

#[test]
fn vision_rope_2d_matches_neox_reference() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone())
        .expect("compile standard shaders");

    // Small config exercising the d%half tiling + the rotate_half split.
    let (l, h, d) = (3usize, 2usize, 8usize);
    let half = d / 2;
    let n = l * h * d;
    let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).sin() * 0.7).collect();
    let freqs: Vec<f32> = (0..l * half).map(|i| 0.05 + (i as f32) * 0.013).collect();

    let want = rope_ref(&x, &freqs, l, h, d);

    let key = PipelineKey::new(
        "vision_rope_2d",
        "vision_rope_2d_f32",
        vec![
            ConstantValue::uint(0, d as u32),
            ConstantValue::uint(1, h as u32),
            ConstantValue::uint(2, n as u32),
        ],
    );
    let pipeline = cache.get_or_build(&key).expect("vision_rope_2d pipeline");

    let x_buf = common::shared_slice(&device, &x);
    let fr_buf = common::shared_slice(&device, &freqs);
    let out_buf = common::shared_zeroed(&device, n * std::mem::size_of::<f32>());

    // Kernel bindings: buffer(0)=out, buffer(1)=x, buffer(2)=freqs.
    let tg = n.div_ceil(256);
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[&out_buf, &x_buf, &fr_buf],
        MTLSize {
            width: tg,
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
    assert!(
        err < 1e-4,
        "vision_rope_2d max_abs_err={err} (want≈{want:?} got≈{got:?})"
    );
}

#[test]
fn vision_rope_2d_interleaved_matches_reference() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone())
        .expect("compile standard shaders");

    // Small config exercising the adjacent-pair rotation + per-pair angle
    // indexing (freqs row stride = half, one angle per pair).
    let (l, h, d) = (3usize, 2usize, 8usize);
    let half = d / 2;
    let n = l * h * d;
    let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).sin() * 0.7).collect();
    let freqs: Vec<f32> = (0..l * half).map(|i| 0.05 + (i as f32) * 0.013).collect();

    let want = rope_ref_interleaved(&x, &freqs, l, h, d);

    let key = PipelineKey::new(
        "vision_rope_2d",
        "vision_rope_2d_interleaved_f32",
        vec![
            ConstantValue::uint(0, d as u32),
            ConstantValue::uint(1, h as u32),
            ConstantValue::uint(2, n as u32),
        ],
    );
    let pipeline = cache
        .get_or_build(&key)
        .expect("vision_rope_2d_interleaved pipeline");

    let x_buf = common::shared_slice(&device, &x);
    let fr_buf = common::shared_slice(&device, &freqs);
    let out_buf = common::shared_zeroed(&device, n * std::mem::size_of::<f32>());

    // Kernel bindings: buffer(0)=out, buffer(1)=x, buffer(2)=freqs.
    let tg = n.div_ceil(256);
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[&out_buf, &x_buf, &fr_buf],
        MTLSize {
            width: tg,
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
    assert!(
        err < 1e-4,
        "vision_rope_2d_interleaved max_abs_err={err} (want≈{want:?} got≈{got:?})"
    );
}

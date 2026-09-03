// SPDX-License-Identifier: Apache-2.0
//! Golden test for `vision_varlen_attn` (Qwen3.5-VL / Qwen3-VL ViT attention).
//!
//! Bidirectional, non-causal, per-segment SDPA over `cu_seqlens`. The kernel is
//! dispatched standalone via `SpecializedPipelineCache` (production path), f32.
//! The reference is the exact softmax(QK^T*scale)V SDPA from mlx-vlm
//! `qwen3_vl/vision.py::Attention` (`base.ensure_fused_sdpa`); that formula is
//! independently verified == mlx's `scaled_dot_product_attention` (cosine 1.0;
//! mlx's fused kernel adds ~3e-3 fp noise — see vllm-rs/tools/vision_parity).
//!
//! A 2-segment input (`cu_seqlens=[0,3,7]`) checks **cross-segment isolation**:
//! the per-segment reference encodes it, so kernel==reference proves a token in
//! segment 0 never attends segment 1.
//!
//! Dispatches on the production MTL4 path (see `common::dispatch_threadgroups`).

mod common;

use objc2_metal::MTLSize;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};

/// Exact per-segment bidirectional SDPA. q/k/v/out: [L, H, D] token-major.
fn sdpa_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cu: &[i32],
    h: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let l = q.len() / (h * d);
    let mut out = vec![0f32; l * h * d];
    let at = |t: usize, hh: usize| (t * h + hh) * d;
    for s in 0..cu.len() - 1 {
        let (bos, eos) = (cu[s] as usize, cu[s + 1] as usize);
        for hh in 0..h {
            for i in bos..eos {
                // scores over the segment
                let mut score = vec![0f32; eos - bos];
                for (jj, j) in (bos..eos).enumerate() {
                    let mut dot = 0f32;
                    for dd in 0..d {
                        dot += q[at(i, hh) + dd] * k[at(j, hh) + dd];
                    }
                    score[jj] = dot * scale;
                }
                let mx = score.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut den = 0f32;
                for sc in score.iter_mut() {
                    *sc = (*sc - mx).exp();
                    den += *sc;
                }
                for dd in 0..d {
                    let mut acc = 0f32;
                    for (jj, j) in (bos..eos).enumerate() {
                        acc += score[jj] * v[at(j, hh) + dd];
                    }
                    out[at(i, hh) + dd] = acc / den;
                }
            }
        }
    }
    out
}

#[test]
fn vision_varlen_attn_matches_sdpa_reference() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone())
        .expect("compile standard shaders");

    // 2 segments (3 + 4 tokens) → exercises cross-segment isolation.
    let cu = [0i32, 3, 7];
    let (h, d) = (2usize, 4usize);
    let l = *cu.last().unwrap() as usize; // 7
    let n = l * h * d;
    let scale = (d as f32).powf(-0.5);

    let mk = |seed: f32| -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + seed) * 0.137).sin() * 0.9)
            .collect()
    };
    let q = mk(1.0);
    let k = mk(2.0);
    let v = mk(3.0);

    let want = sdpa_ref(&q, &k, &v, &cu, h, d, scale);

    let key = PipelineKey::new(
        "vision_varlen_attn",
        "vision_varlen_attn_f32",
        vec![
            ConstantValue::uint(0, d as u32),
            ConstantValue::uint(1, h as u32),
            ConstantValue::uint(2, (cu.len() - 1) as u32),
            ConstantValue::uint(3, l as u32),
            ConstantValue::float(4, scale),
        ],
    );
    let pipeline = cache
        .get_or_build(&key)
        .expect("vision_varlen_attn pipeline");

    let q_buf = common::shared_slice(&device, &q);
    let k_buf = common::shared_slice(&device, &k);
    let v_buf = common::shared_slice(&device, &v);
    let cu_buf = common::shared_slice(&device, &cu);
    let out_buf = common::shared_zeroed(&device, n * 4);

    // buffer(0)=out, (1)=q, (2)=k, (3)=v, (4)=cu_seqlens.
    let threads = l * h;
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[&out_buf, &q_buf, &k_buf, &v_buf, &cu_buf],
        MTLSize {
            width: threads.div_ceil(64),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    let got = common::read_slice::<f32>(&out_buf, n);
    let err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        err < 1e-4,
        "vision_varlen_attn max_abs_err={err}\nwant={want:?}\n got={got:?}"
    );
}

/// Production config: head_dim 72, 16 heads, 2 segments. head_dim 72 is absent
/// from every wired metal attention instantiation (steel head-dims [64,96,128,
/// 256]; attention_via_cache_v2 needs head_dim%32==0) — this confirms the
/// function-constant kernel handles it.
#[test]
fn vision_varlen_attn_head_dim_72() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone())
        .expect("compile standard shaders");

    let cu = [0i32, 5, 12];
    let (h, d) = (16usize, 72usize);
    let l = *cu.last().unwrap() as usize;
    let n = l * h * d;
    let scale = (d as f32).powf(-0.5);
    let mk = |seed: f32| -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + seed) * 0.011).sin() * 0.8)
            .collect()
    };
    let (q, k, v) = (mk(1.0), mk(2.0), mk(3.0));
    let want = sdpa_ref(&q, &k, &v, &cu, h, d, scale);

    let key = PipelineKey::new(
        "vision_varlen_attn",
        "vision_varlen_attn_f32",
        vec![
            ConstantValue::uint(0, d as u32),
            ConstantValue::uint(1, h as u32),
            ConstantValue::uint(2, (cu.len() - 1) as u32),
            ConstantValue::uint(3, l as u32),
            ConstantValue::float(4, scale),
        ],
    );
    let pipeline = cache
        .get_or_build(&key)
        .expect("vision_varlen_attn pipeline");

    let q_buf = common::shared_slice(&device, &q);
    let k_buf = common::shared_slice(&device, &k);
    let v_buf = common::shared_slice(&device, &v);
    let cu_buf = common::shared_slice(&device, &cu);
    let out_buf = common::shared_zeroed(&device, n * 4);

    // buffer(0)=out, (1)=q, (2)=k, (3)=v, (4)=cu_seqlens.
    let threads = l * h;
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[&out_buf, &q_buf, &k_buf, &v_buf, &cu_buf],
        MTLSize {
            width: threads.div_ceil(64),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    let got = common::read_slice::<f32>(&out_buf, n);
    let err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(err < 2e-4, "vision_varlen_attn(d=72) max_abs_err={err}");
}

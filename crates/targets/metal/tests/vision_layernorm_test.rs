// SPDX-License-Identifier: Apache-2.0
//! Goldens for the vision tower's standalone kernels: `vision_layernorm`
//! (Qwen3.5-VL / Qwen3-VL ViT LayerNorm), the GELU flavours, the staged-pixel
//! blit (`copy_rows`), the window row gather (`embedding_gather_rows`), and
//! the VL decoder's embedding splice (`mm_embed_splice`).
//!
//! Dispatched standalone on the MTL4 command path (see
//! `common::dispatch_threadgroups`); the baked kernels carry the constants
//! the lowering bakes, compiled in (`aot::baked_pipeline`).
//! LayerNorm reference = the affine LayerNorm with biased variance +
//! eps-inside-sqrt, independently verified == `mlx.nn.LayerNorm`
//! (max_abs_err 2.4e-7).

mod common;

use half::{bf16, f16};
use objc2_metal::MTLSize;
use scratchy_target_metal::aot::{baked_build, baked_pipeline};
use scratchy_target_metal::detect_device;
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use scratchy_target_metal::tape::ids::{ElementCount, HiddenSize};
use scratchy_target_metal::tape::kernel_constants::{
    CopyRowsConstants, EmbeddingGatherConstants, GeluConstants, MmEmbedSpliceConstants,
};
use scratchy_target_metal::tape::lowered::ActivationWidth;

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
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");

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
            // LN_HAS_BIAS: the norm carries its `.bias`.
            ConstantValue::uint(3, 1),
        ],
    );
    let pipeline = baked_build(&cache, &key).expect("vision_layernorm pipeline");

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

/// The activation dtypes the kernels are instantiated for: each one's symbol suffix and its
/// 16-bit encoding.
const DTYPES: [&str; 2] = ["f16", "bf16"];

fn encode(dtype: &str, v: &[f32]) -> Vec<u16> {
    let bits = |x: f32| match dtype {
        "f16" => f16::from_f32(x).to_bits(),
        _ => bf16::from_f32(x).to_bits(),
    };
    v.iter().map(|&x| bits(x)).collect()
}

fn decode(dtype: &str, v: &[u16]) -> Vec<f32> {
    let value = |b: u16| match dtype {
        "f16" => f16::from_bits(b).to_f32(),
        _ => bf16::from_bits(b).to_f32(),
    };
    v.iter().map(|&b| value(b)).collect()
}

/// One 256-wide threadgroup per 256 elements: the lowering's
/// `dispatch_1d`, rounded up past `n` so the kernels' `gid >= n` guard runs.
fn dispatch_rounded(pipeline: &common::Pipeline, bufs: &[&common::Buffer], n: usize) -> bool {
    let device = detect_device().expect("caller pre-guards").device;
    common::dispatch_threadgroups(
        &device,
        pipeline,
        bufs,
        MTLSize {
            width: n.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    )
}

fn gelu_tanh_ref(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

/// erf by Abramowitz–Stegun 7.1.26 (max abs error 1.5e-7) — far below a 16-bit ulp.
fn gelu_erf_ref(x: f32) -> f32 {
    let z = x * std::f32::consts::FRAC_1_SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * z.abs());
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_7 + t * (-1.453_152 + t * 1.061_405_4))));
    let erf = (1.0 - poly * (-z * z).exp()).copysign(z);
    0.5 * x * (1.0 + erf)
}

fn quick_gelu_ref(x: f32) -> f32 {
    x / (1.0 + (-1.702 * x).exp())
}

/// The three GELU flavours at both dtypes over `n` elements, the dispatch rounded past `n`:
/// every element below `n` matches its reference, none past it is written.
#[test]
fn vision_gelu_flavours_match_reference() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let (n, padded) = (1000usize, 1024usize);
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.037).sin() * 6.0).collect();
    let flavours = [
        ("gelu_tanh", gelu_tanh_ref as fn(f32) -> f32),
        ("gelu_erf", gelu_erf_ref),
        ("quick_gelu", quick_gelu_ref),
    ];
    for (flavour, reference) in flavours {
        for dtype in DTYPES {
            let symbol = format!("{flavour}_{dtype}");
            let constants = GeluConstants {
                elements: ElementCount(n as u32),
            };
            let pipeline = baked_pipeline(&device, "activation", &symbol, constants.into())
                .expect("gelu pipeline");
            let input = common::shared_slice(&device, &encode(dtype, &x));
            let sentinel = encode(dtype, &[7.0])[0];
            let out = common::shared_slice(&device, &vec![sentinel; padded]);
            if !dispatch_rounded(&pipeline, &[&out, &input], n) {
                return;
            }
            let got: Vec<u16> = common::read_slice(&out, padded);
            let values = decode(dtype, &got[..n]);
            for (i, xi) in decode(dtype, &encode(dtype, &x)).iter().enumerate() {
                let want = reference(*xi);
                let g = values[i];
                assert!(
                    (g - want).abs() <= 1e-2 * want.abs().max(1.0),
                    "{symbol}[{i}]: got {g} want {want}"
                );
            }
            assert!(
                got[n..].iter().all(|&g| g == sentinel),
                "{symbol} wrote past n"
            );
        }
    }
}

/// `copy_rows` blits the staged pixels: the first `n` elements, nothing past them.
#[test]
fn vision_copy_rows_blits_n_elements() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let (n, padded) = (1000usize, 1024usize);
    for dtype in DTYPES {
        let symbol = format!("copy_rows_{dtype}");
        let constants = CopyRowsConstants {
            elements: ElementCount(n as u32),
        };
        let pipeline = baked_pipeline(&device, "elementwise", &symbol, constants.into())
            .expect("copy_rows pipeline");
        let x = encode(
            dtype,
            &(0..n).map(|i| i as f32 * 0.25 - 60.0).collect::<Vec<_>>(),
        );
        let input = common::shared_slice(&device, &x);
        let sentinel = encode(dtype, &[-3.0])[0];
        let out = common::shared_slice(&device, &vec![sentinel; padded]);
        if !dispatch_rounded(&pipeline, &[&out, &input], n) {
            return;
        }
        let got: Vec<u16> = common::read_slice(&out, padded);
        assert_eq!(&got[..n], &x[..], "{symbol} payload");
        assert!(
            got[n..].iter().all(|&g| g == sentinel),
            "{symbol} wrote past n"
        );
    }
}

/// `embedding_gather_rows`: `out[i, :] = src[indices[i], :]` over `[rows, width]`.
#[test]
fn vision_embedding_gather_permutes_rows() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let (rows, width) = (6usize, 40usize);
    let n = rows * width;
    let indices: Vec<u32> = vec![3, 0, 5, 1, 4, 2];
    for dtype in DTYPES {
        let symbol = format!("embedding_gather_rows_{dtype}");
        let constants = EmbeddingGatherConstants {
            elements: ElementCount(n as u32),
            width: ActivationWidth::of_cols(width as u32),
        };
        let pipeline = baked_pipeline(&device, "embedding_gather", &symbol, constants.into())
            .expect("embedding_gather pipeline");
        let src = encode(dtype, &(0..n).map(|i| i as f32).collect::<Vec<_>>());
        let src_buf = common::shared_slice(&device, &src);
        let idx_buf = common::shared_slice(&device, &indices);
        let out = common::shared_zeroed(&device, n * std::mem::size_of::<u16>());
        if !dispatch_rounded(&pipeline, &[&out, &src_buf, &idx_buf], n) {
            return;
        }
        let got: Vec<u16> = common::read_slice(&out, n);
        for (i, &row) in indices.iter().enumerate() {
            let want = &src[row as usize * width..][..width];
            assert_eq!(
                &got[i * width..][..width],
                want,
                "{symbol} row {i} <- {row}"
            );
        }
    }
}

/// `mm_embed_splice`: each projected row lands on its placeholder row; a
/// `u32::MAX` destination skips; every other text row is untouched.
#[test]
fn mm_embed_splice_scatters_to_placeholder_rows() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let (text_rows, hidden) = (8usize, 64usize);
    let dst_rows: Vec<u32> = vec![5, 1, u32::MAX, 6];
    for dtype in DTYPES {
        let symbol = format!("mm_embed_splice_{dtype}");
        let constants = MmEmbedSpliceConstants {
            hidden: HiddenSize(hidden as u32),
        };
        let pipeline = baked_pipeline(&device, "elementwise", &symbol, constants.into())
            .expect("mm_embed_splice pipeline");
        let text = encode(
            dtype,
            &(0..text_rows * hidden)
                .map(|i| -(i as f32))
                .collect::<Vec<_>>(),
        );
        let mm = encode(
            dtype,
            &(0..dst_rows.len() * hidden)
                .map(|i| i as f32 + 0.5)
                .collect::<Vec<_>>(),
        );
        let embed = common::shared_slice(&device, &text);
        let mm_buf = common::shared_slice(&device, &mm);
        let dst_buf = common::shared_slice(&device, &dst_rows);
        if !dispatch_rounded(
            &pipeline,
            &[&embed, &mm_buf, &dst_buf],
            dst_rows.len() * hidden,
        ) {
            return;
        }
        let got: Vec<u16> = common::read_slice(&embed, text_rows * hidden);
        for row in 0..text_rows {
            let want = match dst_rows.iter().position(|&d| d as usize == row) {
                Some(s) => &mm[s * hidden..][..hidden],
                None => &text[row * hidden..][..hidden],
            };
            assert_eq!(
                &got[row * hidden..][..hidden],
                want,
                "{symbol} text row {row}"
            );
        }
    }
}

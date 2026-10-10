// SPDX-License-Identifier: Apache-2.0
//! Parity test for `affine_gather_qmv_fast` / `affine_gather_qmv`:
//! each (token, slot) gather should yield the same result as a
//! plain `qmm_t` over the gathered expert's weight slab.

mod common;

use std::ffi::c_void;
use std::ptr::NonNull;

use half::{bf16, f16};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::aot::baked_pipeline;
use scratchy_target_metal::cpu_reference::{
    affine_qmm_t_b8_bf16_s_bf16, affine_qmv_b2_bf16_s_bf16, affine_qmv_b3_bf16_s_bf16,
    affine_qmv_b4_bf16, affine_qmv_b4_bf16_s_bf16,
};
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::quantized::{DequantDtype, ScaleDtype};
use scratchy_target_metal::tape::ids::{KDimI32, NDimI32, TopK};
use scratchy_target_metal::tape::kernel_constants::{
    AffineCodes, AffineGatherQmvConstants, AffineQmvConstants, GatherRows,
};

/// The gather matvec's baked constants: K / N, codes as written, `top_k` output rows per x row.
fn gather_constants(k: u32, n_out: u32, top_k: u32) -> AffineGatherQmvConstants {
    AffineGatherQmvConstants {
        qmv: AffineQmvConstants {
            k: KDimI32(k as i32),
            n: NDimI32(n_out as i32),
            codes: AffineCodes::AsWritten,
        },
        rows: GatherRows::Tokens(TopK(top_k)),
    }
}

fn buf_from_bytes(
    device: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>,
    bytes: &[u8],
) -> objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>> {
    unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
            .expect("newBufferWithBytes nil")
    }
}

fn zeros_buf(
    device: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>,
    n: usize,
) -> objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>> {
    device
        .newBufferWithLength_options(n, MTLResourceOptions::StorageModeShared)
        .expect("newBufferWithLength nil")
}

fn read_bf16(
    buf: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>,
    n: usize,
) -> Vec<bf16> {
    let ptr = buf.contents().as_ptr() as *const bf16;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

/// Build the gather-qmv pipeline for the given shape and dispatch it on
/// the production MTL4 path. Replicates the bindings of
/// `MetalAffineGatherQmv::execute` (buffer 0=packed_w, 1=scales,
/// 2=biases, 3=x, 4=rhs_indices, 5=y; `top_k` baked with K / N,
/// `AffineGatherQmvConstants`) and the same Fast-vs-Generic
/// heuristic + `(1, n_out/8, num_tokens*top_k)` grid. `bits` is fixed at
/// 4 (only width the kernel wires). Returns `false` if the host has no
/// MTL4 queue (caller should skip).
#[allow(clippy::too_many_arguments)]
fn dispatch_gather_qmv(
    device: &common::Device,
    x_buf: &common::Buffer,
    w_buf: &common::Buffer,
    s_buf: &common::Buffer,
    b_buf: &common::Buffer,
    idx_buf: &common::Buffer,
    y_buf: &common::Buffer,
    num_tokens: u32,
    top_k: u32,
    n_out: u32,
    k: u32,
    group_size: u32,
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
) -> bool {
    let bits: u32 = 4;
    // The lowering's Fast-vs-Generic rule.
    let kernel = if scratchy_target_metal::tape::quantized::qmv_fast_covers(n_out, k) {
        "affine_gather_qmv_fast"
    } else {
        "affine_gather_qmv"
    };
    let dt = dtype.symbol_infix();
    let sdt = scale_dtype.symbol_infix();
    let kernel_name = format!("{kernel}_{dt}_s_{sdt}_gs_{group_size}_b_{bits}");
    let constants = gather_constants(k, n_out, top_k).into();
    let pipeline = baked_pipeline(device, "quantized_qmv", &kernel_name, constants)
        .expect("gather qmv pipeline");

    // grid = (1, n_out/8, num_tokens*top_k); threads_per_tg = (32, 2, 1).
    let bn: u32 = 8;
    let threadgroups = MTLSize {
        width: 1,
        height: n_out.div_ceil(bn) as usize,
        depth: (num_tokens as usize) * (top_k as usize),
    };
    let threads_per_threadgroup = MTLSize {
        width: 32,
        height: 2,
        depth: 1,
    };
    common::dispatch_threadgroups(
        device,
        &pipeline,
        &[w_buf, s_buf, b_buf, x_buf, idx_buf, y_buf],
        threadgroups,
        threads_per_threadgroup,
    )
}

fn splitmix(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[test]
fn affine_gather_qmv_bf16_mixtral_decode_fast() {
    // Shape: num_experts=8, n_out=512, k=512 (mini), top_k=2, M=3.
    // n_out=512 multiple of 8, k=512 multiple of 512 → Fast variant.
    let num_experts = 8usize;
    let n_out = 512usize;
    let k = 512usize;
    let group_size = 64usize;
    let top_k = 2usize;
    let num_tokens = 3usize;

    let mut seed = 0xABCD_EF01_2345_6789u64;

    // Pack the per-expert weight slab.
    let mut packed = vec![0u8; num_experts * n_out * k / 2];
    for b in &mut packed {
        *b = splitmix(&mut seed) as u8;
    }
    let mut scales_f16 = vec![f16::ZERO; num_experts * n_out * k / group_size];
    for s in &mut scales_f16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0 + 0.001;
        *s = f16::from_f32(r * 0.05);
    }
    let mut biases_f16 = vec![f16::ZERO; num_experts * n_out * k / group_size];
    for b in &mut biases_f16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *b = f16::from_f32((r - 0.5) * 0.1);
    }

    let mut x_bf16 = vec![bf16::ZERO; num_tokens * k];
    for v in &mut x_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *v = bf16::from_f32((r - 0.5) * 2.0);
    }

    // Indices: each token picks `top_k` distinct experts; choices
    // span the expert range.
    let indices: Vec<u32> = vec![
        0, 5, // token 0
        2, 7, // token 1
        1, 4, // token 2
    ];

    let Some(mdev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };

    let bytes_of =
        |s: &[bf16]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 2) };
    let bytes_of_f16 =
        |s: &[f16]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 2) };
    let bytes_of_u32 =
        |s: &[u32]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4) };

    let w_buf = buf_from_bytes(&mdev.device, &packed);
    let s_buf = buf_from_bytes(&mdev.device, bytes_of_f16(&scales_f16));
    let b_buf = buf_from_bytes(&mdev.device, bytes_of_f16(&biases_f16));
    let x_buf = buf_from_bytes(&mdev.device, bytes_of(&x_bf16));
    let idx_buf = buf_from_bytes(&mdev.device, bytes_of_u32(&indices));
    let y_buf = zeros_buf(&mdev.device, num_tokens * top_k * n_out * 2);

    if !dispatch_gather_qmv(
        &mdev.device,
        &x_buf,
        &w_buf,
        &s_buf,
        &b_buf,
        &idx_buf,
        &y_buf,
        num_tokens as u32,
        top_k as u32,
        n_out as u32,
        k as u32,
        group_size as u32,
        DequantDtype::Bf16,
        ScaleDtype::F16,
    ) {
        return;
    }

    let got = read_bf16(&y_buf, num_tokens * top_k * n_out);

    let packed_per_expert = n_out * k / 2;
    let sb_per_expert = n_out * k / group_size;
    let mut max_err = 0.0_f32;
    for n in 0..num_tokens {
        for slot in 0..top_k {
            let expert = indices[n * top_k + slot] as usize;
            let want = affine_qmv_b4_bf16(
                &packed[expert * packed_per_expert..(expert + 1) * packed_per_expert],
                &scales_f16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
                &biases_f16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
                &x_bf16[n * k..(n + 1) * k],
                1,
                n_out,
                k,
                group_size,
            );
            let base = (n * top_k + slot) * n_out;
            for c in 0..n_out {
                let g = got[base + c].to_f32();
                let w = want[c].to_f32();
                let err = (g - w).abs();
                if err > max_err {
                    max_err = err;
                }
                assert!(
                    err < 3e-2 || (g - w).abs() / w.abs().max(1e-3) < 3e-2,
                    "n={n} slot={slot} expert={expert} c={c} got={g} want={w} err={err}",
                );
            }
        }
    }
    eprintln!(
        "affine_gather_qmv_bf16 num_tokens={num_tokens} top_k={top_k} n={n_out} k={k} max_err={:.3e}",
        max_err
    );
}

#[test]
fn affine_gather_qmv_bf16_generic() {
    // Trigger Generic kernel: k=384 (not %512), n=384.
    let num_experts = 4usize;
    let n_out = 384usize;
    let k = 384usize;
    let group_size = 64usize;
    let top_k = 2usize;
    let num_tokens = 2usize;

    let mut seed = 0x1234_5678_9ABC_DEF0u64;
    let mut packed = vec![0u8; num_experts * n_out * k / 2];
    for b in &mut packed {
        *b = splitmix(&mut seed) as u8;
    }
    let mut scales_f16 = vec![f16::ZERO; num_experts * n_out * k / group_size];
    for s in &mut scales_f16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0 + 0.001;
        *s = f16::from_f32(r * 0.05);
    }
    let mut biases_f16 = vec![f16::ZERO; num_experts * n_out * k / group_size];
    for b in &mut biases_f16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *b = f16::from_f32((r - 0.5) * 0.1);
    }
    let mut x_bf16 = vec![bf16::ZERO; num_tokens * k];
    for v in &mut x_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *v = bf16::from_f32((r - 0.5) * 2.0);
    }
    let indices: Vec<u32> = vec![0, 3, 1, 2];

    let Some(mdev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };

    let bytes_of =
        |s: &[bf16]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 2) };
    let bytes_of_f16 =
        |s: &[f16]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 2) };
    let bytes_of_u32 =
        |s: &[u32]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4) };

    let w_buf = buf_from_bytes(&mdev.device, &packed);
    let s_buf = buf_from_bytes(&mdev.device, bytes_of_f16(&scales_f16));
    let b_buf = buf_from_bytes(&mdev.device, bytes_of_f16(&biases_f16));
    let x_buf = buf_from_bytes(&mdev.device, bytes_of(&x_bf16));
    let idx_buf = buf_from_bytes(&mdev.device, bytes_of_u32(&indices));
    let y_buf = zeros_buf(&mdev.device, num_tokens * top_k * n_out * 2);

    if !dispatch_gather_qmv(
        &mdev.device,
        &x_buf,
        &w_buf,
        &s_buf,
        &b_buf,
        &idx_buf,
        &y_buf,
        num_tokens as u32,
        top_k as u32,
        n_out as u32,
        k as u32,
        group_size as u32,
        DequantDtype::Bf16,
        ScaleDtype::F16,
    ) {
        return;
    }

    let got = read_bf16(&y_buf, num_tokens * top_k * n_out);

    let packed_per_expert = n_out * k / 2;
    let sb_per_expert = n_out * k / group_size;
    for n in 0..num_tokens {
        for slot in 0..top_k {
            let expert = indices[n * top_k + slot] as usize;
            let want = affine_qmv_b4_bf16(
                &packed[expert * packed_per_expert..(expert + 1) * packed_per_expert],
                &scales_f16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
                &biases_f16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
                &x_bf16[n * k..(n + 1) * k],
                1,
                n_out,
                k,
                group_size,
            );
            let base = (n * top_k + slot) * n_out;
            for c in 0..n_out {
                let g = got[base + c].to_f32();
                let w = want[c].to_f32();
                let err = (g - w).abs();
                assert!(
                    err < 4e-2 || (g - w).abs() / w.abs().max(1e-3) < 4e-2,
                    "generic n={n} slot={slot} expert={expert} c={c} got={g} want={w} err={err}",
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// BF16 SCALES — Qwen3-MoE-30B-A3B-Instruct ships BF16 scales/biases
// for `switch_mlp.{gate,up,down}_proj.{scales,biases}` (probed across
// mlx-community/Qwen3-30B-A3B-4bit). Validates
// `ScaleDtype::Bf16` end-to-end against a Qwen3-MoE-shaped gather.
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_gather_qmv_bf16_s_bf16_qwen3_moe_decode_fast() {
    // Shape mirrors Qwen3-MoE-30B-A3B inner block at decode: M=1
    // (one decoded token), num_experts=128, top_k=8, hidden=2048,
    // moe_intermediate=768. We shrink num_experts to 8 and use the
    // gate_proj shape (n_out=moe_intermediate, k=hidden) at a
    // group_size of 64 (Qwen3 mlx-affine-b4-g64).
    let num_experts = 8usize;
    let n_out = 768usize; // moe_intermediate (multiple of 8)
    let k = 2048usize; // hidden (multiple of 512)
    let group_size = 64usize;
    let top_k = 4usize; // top-4 like Mixtral, fine for parity
    let num_tokens = 1usize; // M=1 decode

    let mut seed = 0xC0DE_BA5E_DEAD_BEEFu64;

    let mut packed = vec![0u8; num_experts * n_out * k / 2];
    for b in &mut packed {
        *b = splitmix(&mut seed) as u8;
    }
    let mut scales_bf16 = vec![bf16::ZERO; num_experts * n_out * k / group_size];
    for s in &mut scales_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0 + 0.001;
        *s = bf16::from_f32(r * 0.05);
    }
    let mut biases_bf16 = vec![bf16::ZERO; num_experts * n_out * k / group_size];
    for b in &mut biases_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *b = bf16::from_f32((r - 0.5) * 0.1);
    }
    let mut x_bf16 = vec![bf16::ZERO; num_tokens * k];
    for v in &mut x_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *v = bf16::from_f32((r - 0.5) * 2.0);
    }
    let indices: Vec<u32> = vec![0, 1, 5, 7];

    let Some(mdev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };

    let bytes_of =
        |s: &[bf16]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 2) };
    let bytes_of_u32 =
        |s: &[u32]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4) };
    let w_buf = buf_from_bytes(&mdev.device, &packed);
    let s_buf = buf_from_bytes(&mdev.device, bytes_of(&scales_bf16));
    let b_buf = buf_from_bytes(&mdev.device, bytes_of(&biases_bf16));
    let x_buf = buf_from_bytes(&mdev.device, bytes_of(&x_bf16));
    let idx_buf = buf_from_bytes(&mdev.device, bytes_of_u32(&indices));
    let y_buf = zeros_buf(&mdev.device, num_tokens * top_k * n_out * 2);

    if !dispatch_gather_qmv(
        &mdev.device,
        &x_buf,
        &w_buf,
        &s_buf,
        &b_buf,
        &idx_buf,
        &y_buf,
        num_tokens as u32,
        top_k as u32,
        n_out as u32,
        k as u32,
        group_size as u32,
        DequantDtype::Bf16,
        ScaleDtype::Bf16,
    ) {
        return;
    }

    let got = read_bf16(&y_buf, num_tokens * top_k * n_out);
    let packed_per_expert = n_out * k / 2;
    let sb_per_expert = n_out * k / group_size;
    let mut max_err = 0.0_f32;
    for n in 0..num_tokens {
        for slot in 0..top_k {
            let expert = indices[n * top_k + slot] as usize;
            let want = affine_qmv_b4_bf16_s_bf16(
                &packed[expert * packed_per_expert..(expert + 1) * packed_per_expert],
                &scales_bf16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
                &biases_bf16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
                &x_bf16[n * k..(n + 1) * k],
                1,
                n_out,
                k,
                group_size,
            );
            let base = (n * top_k + slot) * n_out;
            // K=2048 inflates bf16 accumulation noise: stdev grows as
            // sqrt(K) * bf16_eps * per-elem-magnitude. With bf16_eps ≈
            // 6e-3, per-elem ≈ 0.25, K=2048 → stdev ≈ 0.07. Allowed
            // tolerance ≈ 4× stdev (matches the
            // `worst_abs_error_vs_noise_floor` shape in qmv tests).
            let allowed_abs = 0.3_f32;
            let allowed_rel = 0.20_f32;
            for c in 0..n_out {
                let g = got[base + c].to_f32();
                let w = want[c].to_f32();
                let err = (g - w).abs();
                if err > max_err {
                    max_err = err;
                }
                assert!(
                    err < allowed_abs || err / w.abs().max(1e-3) < allowed_rel,
                    "qwen3-shape n={n} slot={slot} expert={expert} c={c} got={g} want={w} err={err}",
                );
            }
        }
    }
    eprintln!(
        "affine_gather_qmv_bf16_s_bf16 qwen3 num_tokens={num_tokens} top_k={top_k} n={n_out} k={k} max_err={:.3e}",
        max_err
    );
}

#[test]
fn affine_gather_qmv_bf16_s_bf16_qwen3_moe_down_proj_shape() {
    // Reproduces the exact Qwen3-MoE-30B-A3B down_proj shape that fails
    // end-to-end at dispatch 24: N=hidden=2048, K=moe_inter=768, gs=64.
    // K=768 is NOT a multiple of 512 → generic (non-fast) variant.
    // K=2048 (gate/up_proj) IS mult-of-512 → fast variant; that one
    // is already covered by `qwen3_moe_decode_fast`.
    let num_experts = 8usize;
    let n_out = 2048usize;
    let k = 768usize;
    let group_size = 64usize;
    let top_k = 4usize;
    let num_tokens = 1usize;

    let mut seed = 0xDEAD_C0DE_BABE_F00Du64;
    let mut packed = vec![0u8; num_experts * n_out * k / 2];
    for b in &mut packed {
        *b = splitmix(&mut seed) as u8;
    }
    let mut scales_bf16 = vec![bf16::ZERO; num_experts * n_out * k / group_size];
    for s in &mut scales_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0 + 0.001;
        *s = bf16::from_f32(r * 0.05);
    }
    let mut biases_bf16 = vec![bf16::ZERO; num_experts * n_out * k / group_size];
    for b in &mut biases_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *b = bf16::from_f32((r - 0.5) * 0.1);
    }
    let mut x_bf16 = vec![bf16::ZERO; num_tokens * k];
    for v in &mut x_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *v = bf16::from_f32((r - 0.5) * 2.0);
    }
    let indices: Vec<u32> = vec![0, 3, 5, 7];

    let Some(mdev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };

    let bytes_of =
        |s: &[bf16]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 2) };
    let bytes_of_u32 =
        |s: &[u32]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4) };
    let w_buf = buf_from_bytes(&mdev.device, &packed);
    let s_buf = buf_from_bytes(&mdev.device, bytes_of(&scales_bf16));
    let b_buf = buf_from_bytes(&mdev.device, bytes_of(&biases_bf16));
    let x_buf = buf_from_bytes(&mdev.device, bytes_of(&x_bf16));
    let idx_buf = buf_from_bytes(&mdev.device, bytes_of_u32(&indices));
    let y_buf = zeros_buf(&mdev.device, num_tokens * top_k * n_out * 2);

    if !dispatch_gather_qmv(
        &mdev.device,
        &x_buf,
        &w_buf,
        &s_buf,
        &b_buf,
        &idx_buf,
        &y_buf,
        num_tokens as u32,
        top_k as u32,
        n_out as u32,
        k as u32,
        group_size as u32,
        DequantDtype::Bf16,
        ScaleDtype::Bf16,
    ) {
        return;
    }

    let got = read_bf16(&y_buf, num_tokens * top_k * n_out);
    let packed_per_expert = n_out * k / 2;
    let sb_per_expert = n_out * k / group_size;
    let mut max_err = 0.0_f32;
    for n in 0..num_tokens {
        for slot in 0..top_k {
            let expert = indices[n * top_k + slot] as usize;
            let want = affine_qmv_b4_bf16_s_bf16(
                &packed[expert * packed_per_expert..(expert + 1) * packed_per_expert],
                &scales_bf16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
                &biases_bf16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
                &x_bf16[n * k..(n + 1) * k],
                1,
                n_out,
                k,
                group_size,
            );
            let base = (n * top_k + slot) * n_out;
            let allowed_abs = 0.3_f32; // K=768 with bf16 accumulation tolerance
            let allowed_rel = 0.25_f32;
            for c in 0..n_out {
                let g = got[base + c].to_f32();
                let w = want[c].to_f32();
                let err = (g - w).abs();
                if err > max_err {
                    max_err = err;
                }
                assert!(
                    err < allowed_abs || err / w.abs().max(1e-3) < allowed_rel,
                    "down_proj_shape n={n} slot={slot} expert={expert} c={c} got={g} want={w} err={err}",
                );
            }
        }
    }
    eprintln!("affine_gather_qmv down_proj shape: max_err={:.3e}", max_err);
}

#[test]
fn affine_gather_qmv_bf16_s_bf16_qwen3_moe_down_proj_top_k_1_workaround() {
    // Production lowering passes `top_k=1` to the kernel for the
    // down-projection step (handoff workaround in
    // `lower_moe_step`). x layout is `[T*top_k, moe_inter]` — one
    // row per (token, slot), not per-token broadcast. rhs_indices
    // has `T*top_k` entries.
    //
    // This test reproduces that invocation EXACTLY: top_k_in_kernel=1,
    // x has one row per (token, slot), and the CPU reference iterates
    // the same way.
    let num_experts = 8usize;
    let n_out = 2048usize;
    let k = 768usize;
    let group_size = 64usize;
    let real_top_k = 4usize;
    let num_tokens = 2usize;
    // Effective rows = T*top_k. The kernel sees this as M with top_k=1.
    let m_rows = num_tokens * real_top_k;

    let mut seed = 0x1357_9BDF_2468_ACE0u64;
    let mut packed = vec![0u8; num_experts * n_out * k / 2];
    for b in &mut packed {
        *b = splitmix(&mut seed) as u8;
    }
    let mut scales_bf16 = vec![bf16::ZERO; num_experts * n_out * k / group_size];
    for s in &mut scales_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0 + 0.001;
        *s = bf16::from_f32(r * 0.05);
    }
    let mut biases_bf16 = vec![bf16::ZERO; num_experts * n_out * k / group_size];
    for b in &mut biases_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *b = bf16::from_f32((r - 0.5) * 0.1);
    }
    // x is m_rows rows of k elements — one per (token, slot).
    let mut x_bf16 = vec![bf16::ZERO; m_rows * k];
    for v in &mut x_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *v = bf16::from_f32((r - 0.5) * 2.0);
    }
    // indices has m_rows entries — one expert id per row.
    let indices: Vec<u32> = vec![0, 3, 5, 7, 1, 2, 4, 6];

    let Some(mdev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };

    let bytes_of =
        |s: &[bf16]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 2) };
    let bytes_of_u32 =
        |s: &[u32]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4) };
    let w_buf = buf_from_bytes(&mdev.device, &packed);
    let s_buf = buf_from_bytes(&mdev.device, bytes_of(&scales_bf16));
    let b_buf = buf_from_bytes(&mdev.device, bytes_of(&biases_bf16));
    let x_buf = buf_from_bytes(&mdev.device, bytes_of(&x_bf16));
    let idx_buf = buf_from_bytes(&mdev.device, bytes_of_u32(&indices));
    let y_buf = zeros_buf(&mdev.device, m_rows * n_out * 2);

    // The KEY: invoke with `top_k = 1` (workaround) so the kernel
    // reads one x row per nk and one expert per nk. num_tokens is
    // set to m_rows = T*top_k so total dispatched rows = m_rows.
    if !dispatch_gather_qmv(
        &mdev.device,
        &x_buf,
        &w_buf,
        &s_buf,
        &b_buf,
        &idx_buf,
        &y_buf,
        m_rows as u32, // num_tokens, but in workaround we pass T*top_k
        1u32,          // top_k_in_kernel = 1 (workaround)
        n_out as u32,
        k as u32,
        group_size as u32,
        DequantDtype::Bf16,
        ScaleDtype::Bf16,
    ) {
        return;
    }

    let got = read_bf16(&y_buf, m_rows * n_out);
    let packed_per_expert = n_out * k / 2;
    let sb_per_expert = n_out * k / group_size;
    let mut max_err = 0.0_f32;
    for nk in 0..m_rows {
        let expert = indices[nk] as usize;
        let want = affine_qmv_b4_bf16_s_bf16(
            &packed[expert * packed_per_expert..(expert + 1) * packed_per_expert],
            &scales_bf16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
            &biases_bf16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
            &x_bf16[nk * k..(nk + 1) * k],
            1,
            n_out,
            k,
            group_size,
        );
        let base = nk * n_out;
        let allowed_abs = 0.3_f32;
        let allowed_rel = 0.25_f32;
        for c in 0..n_out {
            let g = got[base + c].to_f32();
            let w = want[c].to_f32();
            let err = (g - w).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < allowed_abs || err / w.abs().max(1e-3) < allowed_rel,
                "top_k_1 nk={nk} expert={expert} c={c} got={g} want={w} err={err}",
            );
        }
    }
    eprintln!(
        "affine_gather_qmv top_k=1 workaround down_proj: max_err={:.3e}",
        max_err
    );
}

#[test]
fn affine_gather_qmv_bf16_s_bf16_qwen3_moe_down_proj_production_dispatch() {
    // Reproduces the EXACT production dispatch shape that diverges at
    // dispatch 24 in `scr chat`:
    //
    //   tg = (1, n_out/8 = 256, bucket_m * top_k = 72)
    //   threads = (32, 2, 1)
    //
    // Qwen3-30B-A3B-Instruct has top_k=8; bucket_m=9 (prefill chunk
    // with 9 tokens) gives m_rows = 72. The production lowering uses
    // the top_k=1 workaround: the kernel sees one x row per (token,
    // slot) and reads one expert per nk.
    //
    // The smaller parity test (`top_k_1_workaround`) uses m_rows=8
    // (tg.z=8) which left the tg.y=256 × tg.z=72 path unexercised.
    // If this test passes, the integration bug is NOT in the kernel
    // — look at binding/weight resolution. If it fails, the kernel
    // has a scale-dependent bug (threadgroup memory / simd ordering /
    // dispatch grid bounds).
    let num_experts = 128usize;
    let n_out = 2048usize;
    let k = 768usize;
    let group_size = 64usize;
    let m_rows = 72usize; // bucket_m=9 * top_k=8

    let mut seed = 0xCAFE_F00D_8BAD_F00Du64;
    let mut packed = vec![0u8; num_experts * n_out * k / 2];
    for b in &mut packed {
        *b = splitmix(&mut seed) as u8;
    }
    let mut scales_bf16 = vec![bf16::ZERO; num_experts * n_out * k / group_size];
    for s in &mut scales_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0 + 0.001;
        *s = bf16::from_f32(r * 0.05);
    }
    let mut biases_bf16 = vec![bf16::ZERO; num_experts * n_out * k / group_size];
    for b in &mut biases_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *b = bf16::from_f32((r - 0.5) * 0.1);
    }
    let mut x_bf16 = vec![bf16::ZERO; m_rows * k];
    for v in &mut x_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *v = bf16::from_f32((r - 0.5) * 2.0);
    }
    // Indices: pick experts spanning the 128-expert range to match the
    // bisect-observed `[53, 116, 70, 29, 16, 19, 58, 49, ...]` pattern.
    let indices: Vec<u32> = (0..m_rows)
        .map(|i| ((i * 53 + 7) % num_experts) as u32)
        .collect();

    let Some(mdev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };

    let bytes_of =
        |s: &[bf16]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 2) };
    let bytes_of_u32 =
        |s: &[u32]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4) };
    let w_buf = buf_from_bytes(&mdev.device, &packed);
    let s_buf = buf_from_bytes(&mdev.device, bytes_of(&scales_bf16));
    let b_buf = buf_from_bytes(&mdev.device, bytes_of(&biases_bf16));
    let x_buf = buf_from_bytes(&mdev.device, bytes_of(&x_bf16));
    let idx_buf = buf_from_bytes(&mdev.device, bytes_of_u32(&indices));
    let y_buf = zeros_buf(&mdev.device, m_rows * n_out * 2);

    if !dispatch_gather_qmv(
        &mdev.device,
        &x_buf,
        &w_buf,
        &s_buf,
        &b_buf,
        &idx_buf,
        &y_buf,
        m_rows as u32, // num_tokens param == m_rows (workaround)
        1u32,          // top_k_in_kernel = 1 (workaround)
        n_out as u32,
        k as u32,
        group_size as u32,
        DequantDtype::Bf16,
        ScaleDtype::Bf16,
    ) {
        return;
    }

    let got = read_bf16(&y_buf, m_rows * n_out);
    let packed_per_expert = n_out * k / 2;
    let sb_per_expert = n_out * k / group_size;
    let mut max_err = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut fail_count = 0usize;
    for nk in 0..m_rows {
        let expert = indices[nk] as usize;
        let want = affine_qmv_b4_bf16_s_bf16(
            &packed[expert * packed_per_expert..(expert + 1) * packed_per_expert],
            &scales_bf16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
            &biases_bf16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
            &x_bf16[nk * k..(nk + 1) * k],
            1,
            n_out,
            k,
            group_size,
        );
        let base = nk * n_out;
        let allowed_abs = 0.3_f32;
        let allowed_rel = 0.25_f32;
        for c in 0..n_out {
            let g = got[base + c].to_f32();
            let w = want[c].to_f32();
            let err = (g - w).abs();
            let rel = err / w.abs().max(1e-3);
            if err > max_err {
                max_err = err;
            }
            if rel > max_rel {
                max_rel = rel;
            }
            if err > allowed_abs && rel > allowed_rel {
                fail_count += 1;
                if fail_count < 8 {
                    eprintln!(
                        "FAIL nk={nk} expert={expert} c={c} got={g} want={w} err={err} rel={rel}",
                    );
                }
            }
        }
    }
    eprintln!(
        "affine_gather_qmv production dispatch m_rows={m_rows}: max_err={max_err:.3e} \
         max_rel={max_rel:.3e} fail_count={fail_count}",
    );
    assert_eq!(
        fail_count, 0,
        "production dispatch produced {fail_count} out-of-tolerance elements"
    );
}

#[test]
fn affine_gather_qmv_bf16_s_bf16_qwen3_5_moe_down_proj_fast_k512() {
    // Qwen3.5-MoE-35B-A3B down_proj at decode: num_experts=256,
    // n_out=hidden=2048, k=moe_intermediate=512, gs=64, top_k=8, M=1.
    // k=512 is a multiple of 512 → the FAST gather variant (unlike
    // Qwen3-MoE's k=768 down which takes the generic variant); the
    // single-512-block K loop is the newly exercised edge.
    let num_experts = 256usize;
    let n_out = 2048usize;
    let k = 512usize;
    let group_size = 64usize;
    let top_k = 8usize;
    let num_tokens = 1usize;

    let mut seed = 0x35B0_A3B0_DEAD_BEEFu64;

    let mut packed = vec![0u8; num_experts * n_out * k / 2];
    for b in &mut packed {
        *b = splitmix(&mut seed) as u8;
    }
    let mut scales_bf16 = vec![bf16::ZERO; num_experts * n_out * k / group_size];
    for s in &mut scales_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0 + 0.001;
        *s = bf16::from_f32(r * 0.05);
    }
    let mut biases_bf16 = vec![bf16::ZERO; num_experts * n_out * k / group_size];
    for b in &mut biases_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *b = bf16::from_f32((r - 0.5) * 0.1);
    }
    let mut x_bf16 = vec![bf16::ZERO; num_tokens * k];
    for v in &mut x_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *v = bf16::from_f32((r - 0.5) * 2.0);
    }
    // Spread experts across the full 256 range incl. the last expert.
    let indices: Vec<u32> = vec![0, 31, 64, 127, 128, 200, 254, 255];

    let Some(mdev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };

    let bytes_of =
        |s: &[bf16]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 2) };
    let bytes_of_u32 =
        |s: &[u32]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4) };
    let w_buf = buf_from_bytes(&mdev.device, &packed);
    let s_buf = buf_from_bytes(&mdev.device, bytes_of(&scales_bf16));
    let b_buf = buf_from_bytes(&mdev.device, bytes_of(&biases_bf16));
    let x_buf = buf_from_bytes(&mdev.device, bytes_of(&x_bf16));
    let idx_buf = buf_from_bytes(&mdev.device, bytes_of_u32(&indices));
    let y_buf = zeros_buf(&mdev.device, num_tokens * top_k * n_out * 2);

    if !dispatch_gather_qmv(
        &mdev.device,
        &x_buf,
        &w_buf,
        &s_buf,
        &b_buf,
        &idx_buf,
        &y_buf,
        num_tokens as u32,
        top_k as u32,
        n_out as u32,
        k as u32,
        group_size as u32,
        DequantDtype::Bf16,
        ScaleDtype::Bf16,
    ) {
        return;
    }

    let got = read_bf16(&y_buf, num_tokens * top_k * n_out);
    let packed_per_expert = n_out * k / 2;
    let sb_per_expert = n_out * k / group_size;
    let mut max_err = 0.0_f32;
    for n in 0..num_tokens {
        for slot in 0..top_k {
            let expert = indices[n * top_k + slot] as usize;
            let want = affine_qmv_b4_bf16_s_bf16(
                &packed[expert * packed_per_expert..(expert + 1) * packed_per_expert],
                &scales_bf16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
                &biases_bf16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
                &x_bf16[n * k..(n + 1) * k],
                1,
                n_out,
                k,
                group_size,
            );
            let base = (n * top_k + slot) * n_out;
            let allowed_abs = 0.2_f32;
            let allowed_rel = 0.20_f32;
            for c in 0..n_out {
                let g = got[base + c].to_f32();
                let w = want[c].to_f32();
                let err = (g - w).abs();
                if err > max_err {
                    max_err = err;
                }
                assert!(
                    err < allowed_abs || err / w.abs().max(1e-3) < allowed_rel,
                    "qwen3.5-down n={n} slot={slot} expert={expert} c={c} got={g} want={w} err={err}",
                );
            }
        }
    }
    eprintln!(
        "affine_gather_qmv qwen3.5 down fast k512 E=256 max_err={:.3e}",
        max_err
    );
}

#[test]
fn affine_gather_qmv_bf16_s_bf16_qwen3_5_moe_gate_proj_e256() {
    // Qwen3.5-MoE-35B-A3B gate/up_proj at decode: num_experts=256,
    // n_out=moe_intermediate=512, k=hidden=2048, gs=64, top_k=8, M=1.
    let num_experts = 256usize;
    let n_out = 512usize;
    let k = 2048usize;
    let group_size = 64usize;
    let top_k = 8usize;
    let num_tokens = 1usize;

    let mut seed = 0xA5A5_5A5A_DEAD_BEEFu64;

    let mut packed = vec![0u8; num_experts * n_out * k / 2];
    for b in &mut packed {
        *b = splitmix(&mut seed) as u8;
    }
    let mut scales_bf16 = vec![bf16::ZERO; num_experts * n_out * k / group_size];
    for s in &mut scales_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0 + 0.001;
        *s = bf16::from_f32(r * 0.05);
    }
    let mut biases_bf16 = vec![bf16::ZERO; num_experts * n_out * k / group_size];
    for b in &mut biases_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *b = bf16::from_f32((r - 0.5) * 0.1);
    }
    let mut x_bf16 = vec![bf16::ZERO; num_tokens * k];
    for v in &mut x_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *v = bf16::from_f32((r - 0.5) * 2.0);
    }
    let indices: Vec<u32> = vec![3, 17, 99, 130, 177, 201, 233, 255];

    let Some(mdev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };

    let bytes_of =
        |s: &[bf16]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 2) };
    let bytes_of_u32 =
        |s: &[u32]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4) };
    let w_buf = buf_from_bytes(&mdev.device, &packed);
    let s_buf = buf_from_bytes(&mdev.device, bytes_of(&scales_bf16));
    let b_buf = buf_from_bytes(&mdev.device, bytes_of(&biases_bf16));
    let x_buf = buf_from_bytes(&mdev.device, bytes_of(&x_bf16));
    let idx_buf = buf_from_bytes(&mdev.device, bytes_of_u32(&indices));
    let y_buf = zeros_buf(&mdev.device, num_tokens * top_k * n_out * 2);

    if !dispatch_gather_qmv(
        &mdev.device,
        &x_buf,
        &w_buf,
        &s_buf,
        &b_buf,
        &idx_buf,
        &y_buf,
        num_tokens as u32,
        top_k as u32,
        n_out as u32,
        k as u32,
        group_size as u32,
        DequantDtype::Bf16,
        ScaleDtype::Bf16,
    ) {
        return;
    }

    let got = read_bf16(&y_buf, num_tokens * top_k * n_out);
    let packed_per_expert = n_out * k / 2;
    let sb_per_expert = n_out * k / group_size;
    let mut max_err = 0.0_f32;
    for n in 0..num_tokens {
        for slot in 0..top_k {
            let expert = indices[n * top_k + slot] as usize;
            let want = affine_qmv_b4_bf16_s_bf16(
                &packed[expert * packed_per_expert..(expert + 1) * packed_per_expert],
                &scales_bf16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
                &biases_bf16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
                &x_bf16[n * k..(n + 1) * k],
                1,
                n_out,
                k,
                group_size,
            );
            let base = (n * top_k + slot) * n_out;
            let allowed_abs = 0.3_f32;
            let allowed_rel = 0.20_f32;
            for c in 0..n_out {
                let g = got[base + c].to_f32();
                let w = want[c].to_f32();
                let err = (g - w).abs();
                if err > max_err {
                    max_err = err;
                }
                assert!(
                    err < allowed_abs || err / w.abs().max(1e-3) < allowed_rel,
                    "qwen3.5-gate n={n} slot={slot} expert={expert} c={c} got={g} want={w} err={err}",
                );
            }
        }
    }
    eprintln!(
        "affine_gather_qmv qwen3.5 gate E=256 max_err={:.3e}",
        max_err
    );
}

/// 3-bit gather parity — GLM-4.5-Air-3bit's MoE decode kernels
/// (`affine_gather_qmv_gated_fast` / `affine_gather_qmv_combine` at
/// `b_3`), which no prior test covered (b4/b8 only). The plain gather
/// (`affine_gather_qmv{,_fast}`) is the compute core the routed kernels
/// share (`gather_qmv_pair`), so parity here covers the b3 byte-offset
/// math (`IN_VEC_SIZE * bits / 32` expert strides, the 8-codes-per-3-bytes
/// qdot) the fused MoE decode kernels also run. GLM-4.5-Air shapes:
/// gate/up n_out=1408 k=4096, down n_out=4096 k=1408, top_k=8, gs=64.
#[test]
fn affine_gather_qmv_b3_bf16_glm_4_5_air_decode() {
    // (n_out, k) covers both routed-expert matvec shapes; 1408 and 4096
    // are both K-multiples of 512? 1408 = 2.75×512 — NOT. So gate/up
    // (k=4096) is Fast, down (k=1408) is Generic: both variants covered.
    for (n_out, k, seed) in [
        (1408usize, 4096usize, 0x6C17_4D5A_3B1D_F00Du64),
        (4096usize, 1408usize, 0xD1A6_0F1E_5EED_B00Bu64),
    ] {
        let num_experts = 16usize;
        let group_size = 64usize;
        let top_k = 8usize;
        let num_tokens = 1usize;
        let bits = 3u32;

        let mut s = seed;
        // 3-bit: 8 codes per 3 bytes → n*k*3/8 bytes per expert slab.
        let mut packed = vec![0u8; num_experts * n_out * k * 3 / 8];
        for b in &mut packed {
            *b = splitmix(&mut s) as u8;
        }
        let mut scales = vec![bf16::ZERO; num_experts * n_out * k / group_size];
        for v in &mut scales {
            *v = bf16::from_f32(0.0008 + 0.0004 * ((splitmix(&mut s) % 1000) as f32 / 1000.0));
        }
        let mut biases = vec![bf16::ZERO; num_experts * n_out * k / group_size];
        for v in &mut biases {
            *v = bf16::from_f32(-0.1 + 0.02 * ((splitmix(&mut s) % 1000) as f32 / 1000.0));
        }
        let mut x = vec![bf16::ZERO; num_tokens * k];
        for v in &mut x {
            *v = bf16::from_f32(((splitmix(&mut s) % 1000) as f32 / 1000.0 - 0.5) * 2.0);
        }
        // top_k distinct experts across the 16-expert range.
        let indices: Vec<u32> = vec![0, 2, 5, 7, 9, 11, 13, 15];

        let Some(mdev) = detect_device() else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };
        let bytes_of = |v: &[bf16]| unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2)
        };
        let bytes_u32 =
            |v: &[u32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        let w_buf = buf_from_bytes(&mdev.device, &packed);
        let s_buf = buf_from_bytes(&mdev.device, bytes_of(&scales));
        let b_buf = buf_from_bytes(&mdev.device, bytes_of(&biases));
        let x_buf = buf_from_bytes(&mdev.device, bytes_of(&x));
        let idx_buf = buf_from_bytes(&mdev.device, bytes_u32(&indices));
        let y_buf = zeros_buf(&mdev.device, num_tokens * top_k * n_out * 2);

        let kernel = if (n_out as u32).is_multiple_of(8) && (k as u32).is_multiple_of(512) {
            "affine_gather_qmv_fast"
        } else {
            "affine_gather_qmv"
        };
        let name = format!("{kernel}_bf16_s_bf16_gs_{group_size}_b_{bits}");
        let constants = gather_constants(k as u32, n_out as u32, top_k as u32).into();
        let pipeline = baked_pipeline(&mdev.device, "quantized_qmv", &name, constants)
            .expect("b3 gather pipeline");
        let bn: u32 = 8;
        if !common::dispatch_threadgroups(
            &mdev.device,
            &pipeline,
            &[&w_buf, &s_buf, &b_buf, &x_buf, &idx_buf, &y_buf],
            MTLSize {
                width: 1,
                height: n_out.div_ceil(bn as usize),
                depth: num_tokens * top_k,
            },
            MTLSize {
                width: 32,
                height: 2,
                depth: 1,
            },
        ) {
            return;
        }

        let got = read_bf16(&y_buf, num_tokens * top_k * n_out);
        let w_pe = n_out * k * 3 / 8; // 3-bit bytes per expert
        let sb_pe = n_out * k / group_size;
        let mut max_err = 0.0_f32;
        for n in 0..num_tokens {
            for slot in 0..top_k {
                let e = indices[n * top_k + slot] as usize;
                let want = affine_qmv_b3_bf16_s_bf16(
                    &packed[e * w_pe..(e + 1) * w_pe],
                    &scales[e * sb_pe..(e + 1) * sb_pe],
                    &biases[e * sb_pe..(e + 1) * sb_pe],
                    &x[n * k..(n + 1) * k],
                    1,
                    n_out,
                    k,
                    group_size,
                );
                let base = (n * top_k + slot) * n_out;
                for c in 0..n_out {
                    let g = got[base + c].to_f32();
                    let w = want[c].to_f32();
                    let err = (g - w).abs();
                    max_err = max_err.max(err);
                    assert!(
                        err < 0.3 || err / w.abs().max(1e-3) < 0.2,
                        "b3-gather n_out={n_out} k={k} slot={slot} e={e} c={c} got={g} want={w} err={err}",
                    );
                }
            }
        }
        eprintln!("affine_gather_qmv b3 n_out={n_out} k={k} max_err={max_err:.3e}");
    }
}

/// 2-bit gather parity — gpt-oss-120b-mlx-2Bit's MoE decode path
/// (`affine_gather_qmv{,_fast}` at `b_2`, the compute core the routed
/// gated/combine kernels share via `gather_qmv_pair`). The 2-bit law:
/// element i at bits `[2i, 2i+2)` — 4 codes per byte, expert stride
/// `IN_VEC_SIZE * bits / 32` u32. gpt-oss shapes: every routed-expert
/// matvec is (n_out=2880, k=2880) — Generic, since 2880 % 512 ≠ 0; the
/// second case uses a %512 K so the Fast variant is covered too
/// (top_k=4 and E=128 in production; 16 experts / top_k=4 here for
/// speed).
#[test]
fn affine_gather_qmv_b2_bf16_gpt_oss_decode() {
    for (n_out, k, seed) in [
        (2880usize, 2880usize, 0x2B17_4D5A_3B1D_F00Du64),
        (1408usize, 4096usize, 0xD1A6_0F1E_5EED_B02Bu64),
    ] {
        let num_experts = 16usize;
        let group_size = 64usize;
        let top_k = 4usize;
        let num_tokens = 1usize;
        let bits = 2u32;

        let mut s = seed;
        // 2-bit: 4 codes per byte → n*k/4 bytes per expert slab.
        let mut packed = vec![0u8; num_experts * n_out * k / 4];
        for b in &mut packed {
            *b = splitmix(&mut s) as u8;
        }
        // Realistic 2-bit scales/biases: codes span 0..3, so scale ≈
        // range/3 keeps dequant weights near ±0.1.
        let mut scales = vec![bf16::ZERO; num_experts * n_out * k / group_size];
        for v in &mut scales {
            *v = bf16::from_f32(0.01 + 0.006 * ((splitmix(&mut s) % 1000) as f32 / 1000.0));
        }
        let mut biases = vec![bf16::ZERO; num_experts * n_out * k / group_size];
        for v in &mut biases {
            *v = bf16::from_f32(-0.08 + 0.015 * ((splitmix(&mut s) % 1000) as f32 / 1000.0));
        }
        let mut x = vec![bf16::ZERO; num_tokens * k];
        for v in &mut x {
            *v = bf16::from_f32(((splitmix(&mut s) % 1000) as f32 / 1000.0 - 0.5) * 2.0);
        }
        // top_k distinct experts across the 16-expert range.
        let indices: Vec<u32> = vec![0, 3, 5, 7];

        let Some(mdev) = detect_device() else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };
        let bytes_of = |v: &[bf16]| unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2)
        };
        let bytes_u32 =
            |v: &[u32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        let w_buf = buf_from_bytes(&mdev.device, &packed);
        let s_buf = buf_from_bytes(&mdev.device, bytes_of(&scales));
        let b_buf = buf_from_bytes(&mdev.device, bytes_of(&biases));
        let x_buf = buf_from_bytes(&mdev.device, bytes_of(&x));
        let idx_buf = buf_from_bytes(&mdev.device, bytes_u32(&indices));
        let y_buf = zeros_buf(&mdev.device, num_tokens * top_k * n_out * 2);

        let kernel = if (n_out as u32).is_multiple_of(8) && (k as u32).is_multiple_of(512) {
            "affine_gather_qmv_fast"
        } else {
            "affine_gather_qmv"
        };
        let name = format!("{kernel}_bf16_s_bf16_gs_{group_size}_b_{bits}");
        let constants = gather_constants(k as u32, n_out as u32, top_k as u32).into();
        let pipeline = baked_pipeline(&mdev.device, "quantized_qmv", &name, constants)
            .expect("b2 gather pipeline");
        let bn: u32 = 8;
        if !common::dispatch_threadgroups(
            &mdev.device,
            &pipeline,
            &[&w_buf, &s_buf, &b_buf, &x_buf, &idx_buf, &y_buf],
            MTLSize {
                width: 1,
                height: n_out.div_ceil(bn as usize),
                depth: num_tokens * top_k,
            },
            MTLSize {
                width: 32,
                height: 2,
                depth: 1,
            },
        ) {
            return;
        }

        let got = read_bf16(&y_buf, num_tokens * top_k * n_out);
        let w_pe = n_out * k / 4; // 2-bit bytes per expert
        let sb_pe = n_out * k / group_size;
        let mut max_err = 0.0_f32;
        for n in 0..num_tokens {
            for slot in 0..top_k {
                let e = indices[n * top_k + slot] as usize;
                let want = affine_qmv_b2_bf16_s_bf16(
                    &packed[e * w_pe..(e + 1) * w_pe],
                    &scales[e * sb_pe..(e + 1) * sb_pe],
                    &biases[e * sb_pe..(e + 1) * sb_pe],
                    &x[n * k..(n + 1) * k],
                    1,
                    n_out,
                    k,
                    group_size,
                );
                let base = (n * top_k + slot) * n_out;
                for c in 0..n_out {
                    let g = got[base + c].to_f32();
                    let w = want[c].to_f32();
                    let err = (g - w).abs();
                    max_err = max_err.max(err);
                    assert!(
                        err < 0.3 || err / w.abs().max(1e-3) < 0.2,
                        "b2-gather n_out={n_out} k={k} slot={slot} e={e} c={c} got={g} want={w} err={err}",
                    );
                }
            }
        }
        eprintln!("affine_gather_qmv b2 n_out={n_out} k={k} max_err={max_err:.3e}");
    }
}

/// 8-bit gather parity. `dispatch_gather_qmv` hardcodes bits=4, so the 8-bit
/// `affine_gather_qmv{,_fast}_..._b_8` path — required by Qwen3.5 OptiQ (which
/// ships all-8-bit routed experts on layers 0/2 and within-layer-mixed on
/// 1/3/39) and NEVER exercised by uniform (all-4-bit experts) — is otherwise
/// untested. 8-bit packs 1 code/byte (pack_factor 4); reference is the b8 dense
/// qmm_t with M=1 per gathered expert. Covers the FAST variant (gate/up:
/// n_out=512,k=2048 ; down: n_out=2048,k=512 — both N%8==0 && K%512==0).
#[test]
fn affine_gather_qmv_b8_bf16_qwen3_5_optiq_fast() {
    for (n_out, k, seed) in [
        (512usize, 2048usize, 0x0B8Au64),
        (2048usize, 512usize, 0x0B8Bu64),
    ] {
        let num_experts = 8usize;
        let group_size = 64usize;
        let top_k = 4usize;
        let num_tokens = 1usize;
        let bits = 8u32;

        let mut s = seed;
        // 8-bit: one code per byte → num_experts*n_out*k bytes.
        let mut packed = vec![0u8; num_experts * n_out * k];
        for b in &mut packed {
            *b = splitmix(&mut s) as u8;
        }
        // Realistic 8-bit scales/biases (match make_inputs_b8_f32 in the dense
        // b8 test): scale ≈ range/255 so dequant weights land near ±0.1, not
        // the ±12 that oversized scales would produce (which only inflates
        // cancellation noise, not a real kernel error).
        let mut scales = vec![bf16::ZERO; num_experts * n_out * k / group_size];
        for v in &mut scales {
            *v = bf16::from_f32(0.0006 + 0.0004 * ((splitmix(&mut s) % 1000) as f32 / 1000.0));
        }
        let mut biases = vec![bf16::ZERO; num_experts * n_out * k / group_size];
        for v in &mut biases {
            *v = bf16::from_f32(-0.1 + 0.02 * ((splitmix(&mut s) % 1000) as f32 / 1000.0));
        }
        let mut x = vec![bf16::ZERO; num_tokens * k];
        for v in &mut x {
            *v = bf16::from_f32(((splitmix(&mut s) % 1000) as f32 / 1000.0 - 0.5) * 2.0);
        }
        let indices: Vec<u32> = vec![0, 3, 5, 7];

        let Some(mdev) = detect_device() else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };
        let bytes_of = |v: &[bf16]| unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2)
        };
        let bytes_u32 =
            |v: &[u32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        let w_buf = buf_from_bytes(&mdev.device, &packed);
        let s_buf = buf_from_bytes(&mdev.device, bytes_of(&scales));
        let b_buf = buf_from_bytes(&mdev.device, bytes_of(&biases));
        let x_buf = buf_from_bytes(&mdev.device, bytes_of(&x));
        let idx_buf = buf_from_bytes(&mdev.device, bytes_u32(&indices));
        let y_buf = zeros_buf(&mdev.device, num_tokens * top_k * n_out * 2);

        // Build + dispatch the b_8 fast gather (mirrors dispatch_gather_qmv, bits=8).
        let kernel =
            if scratchy_target_metal::tape::quantized::qmv_fast_covers(n_out as u32, k as u32) {
                "affine_gather_qmv_fast"
            } else {
                "affine_gather_qmv"
            };
        let name = format!("{kernel}_bf16_s_bf16_gs_{group_size}_b_{bits}");
        let constants = gather_constants(k as u32, n_out as u32, top_k as u32).into();
        let pipeline = baked_pipeline(&mdev.device, "quantized_qmv", &name, constants)
            .expect("b8 gather pipeline");
        let bn: u32 = 8;
        if !common::dispatch_threadgroups(
            &mdev.device,
            &pipeline,
            &[&w_buf, &s_buf, &b_buf, &x_buf, &idx_buf, &y_buf],
            MTLSize {
                width: 1,
                height: n_out.div_ceil(bn as usize),
                depth: num_tokens * top_k,
            },
            MTLSize {
                width: 32,
                height: 2,
                depth: 1,
            },
        ) {
            return;
        }

        let got = read_bf16(&y_buf, num_tokens * top_k * n_out);
        let w_pe = n_out * k; // 8-bit bytes per expert
        let sb_pe = n_out * k / group_size;
        let mut max_err = 0.0_f32;
        for n in 0..num_tokens {
            for slot in 0..top_k {
                let e = indices[n * top_k + slot] as usize;
                let want = affine_qmm_t_b8_bf16_s_bf16(
                    &packed[e * w_pe..(e + 1) * w_pe],
                    &scales[e * sb_pe..(e + 1) * sb_pe],
                    &biases[e * sb_pe..(e + 1) * sb_pe],
                    &x[n * k..(n + 1) * k],
                    1,
                    n_out,
                    k,
                    group_size,
                );
                let base = (n * top_k + slot) * n_out;
                for c in 0..n_out {
                    let g = got[base + c].to_f32();
                    let w = want[c].to_f32();
                    let err = (g - w).abs();
                    max_err = max_err.max(err);
                    assert!(
                        err < 0.3 || err / w.abs().max(1e-3) < 0.2,
                        "b8-gather n_out={n_out} k={k} slot={slot} e={e} c={c} got={g} want={w} err={err}",
                    );
                }
            }
        }
        eprintln!("affine_gather_qmv b8 n_out={n_out} k={k} max_err={max_err:.3e}");
    }
}

/// The FUSED routed MoE gate/up kernel — GLM-4.5-Air's M1-only path
/// (`MetalFusion::MoeRouted`): `affine_gather_qmv_gated_fast` computes its
/// routing IN-KERNEL from the router Gemm's logits (ROUTED_EXPERTS at slot
/// 13, PRE=2 sigmoid+bias at 14, POST=2 renorm at 16), picks its own
/// experts, stores rhs_indices + scores, then runs the gathered matvecs
/// and the gated activation. M2+ buckets never fuse (pairs >= 64 or a
/// separate MoeRoute command): this exact pairing of b3 gather + in-kernel
/// routing is otherwise untested. The reference is the same CPU b3 qmv per
/// chosen expert + the routing program of `moe_route_kernel`'s GLM test:
/// biased-sigmoid top-k, unbiased-sigmoid scores renormalized.
#[test]
fn affine_gather_qmv_gated_b3_routed_glm_decode() {
    // Two geometries: E=16 (PER_LANE=1) and the production E=128
    // (PER_LANE=4 — route_top_k's ballot is 4 u32 lanes wide, untested
    // at E=16), both fused (E <= the kernel's 128 threads).
    routed_gated_b3_case(16, 4096, 1408, 8, 0x6D6F_6520_726F_7574);
    routed_gated_b3_case(128, 512, 1408, 8, 0x6D6F_6520_726F_7575);
}

fn routed_gated_b3_case(num_experts: usize, k: usize, n_out: usize, top_k: usize, seed: u64) {
    use scratchy_target_metal::tape::constants::ConstantValue;
    use scratchy_target_metal::tape::ids::NumExperts;
    use scratchy_target_metal::tape::kernel_constants::RoutedConstants;
    use scratchy_target_metal::tape::step::{LayerId, RoutePost, RoutePre, RouteProgram};

    // GLM shapes, shrunk for the test: top_k=8, hidden k (Fast: k%512),
    // moe_intermediate n_out=1408 (Fast: n%8), gs=64, b3.
    let group_size = 64usize;
    let num_tokens = 1usize;
    let bits = 3u32;

    let mut s = seed;
    // Router logits [tokens, E] and the F32 correction bias [E].
    let mut logits = vec![bf16::ZERO; num_tokens * num_experts];
    for v in logits.iter_mut() {
        *v = bf16::from_f32((splitmix(&mut s) % 2000) as f32 / 1000.0 - 1.0);
    }
    let mut bias = vec![0f32; num_experts];
    for v in bias.iter_mut() {
        *v = (splitmix(&mut s) % 1000) as f32 / 1000.0 - 0.5;
    }

    // Gate/up packed slabs per expert.
    let pack = |s: &mut u64| {
        let packed = vec![0u8; num_experts * n_out * k * 3 / 8];
        let mut packed = packed;
        for b in &mut packed {
            *b = splitmix(s) as u8;
        }
        let mut scales = vec![bf16::ZERO; num_experts * n_out * k / group_size];
        for v in scales.iter_mut() {
            *v = bf16::from_f32(0.0008 + 0.0004 * (splitmix(s) % 1000) as f32 / 1000.0);
        }
        let mut biases = vec![bf16::ZERO; num_experts * n_out * k / group_size];
        for v in biases.iter_mut() {
            *v = bf16::from_f32(-0.1 + 0.02 * (splitmix(s) % 1000) as f32 / 1000.0);
        }
        (packed, scales, biases)
    };
    let (g_packed, g_scales, g_biases) = pack(&mut s);
    let (u_packed, u_scales, u_biases) = pack(&mut s);
    let mut x = vec![bf16::ZERO; num_tokens * k];
    for v in x.iter_mut() {
        *v = bf16::from_f32((splitmix(&mut s) % 1000) as f32 / 1000.0 - 0.5);
    }

    let Some(mdev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let bytes_of =
        |v: &[bf16]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) };
    let bytes_f32 =
        |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };

    // Device buffers. Scratch layout mirrors MoeScratchLayout::compute's
    // Gathered M1 plan (rhs_indices, scores, gate_y, up_y regions), but
    // only relative offsets between separately-bound regions matter here:
    // bind each region as its own buffer.
    let pairs = num_tokens * top_k;
    let rhs_buf = zeros_buf(&mdev.device, pairs * 4);
    let scores_buf = zeros_buf(&mdev.device, pairs * 2);
    let gate_y = zeros_buf(&mdev.device, pairs * n_out * 2);
    let up_y = zeros_buf(&mdev.device, pairs * n_out * 2);
    let logits_buf = buf_from_bytes(&mdev.device, bytes_of(&logits));
    let bias_buf = buf_from_bytes(&mdev.device, bytes_f32(&bias));

    // Constants: AffineGatherQmvConstants (K, N, rows) + gated act (3) +
    // RoutedConstants (13-17).
    let mut constants: Vec<ConstantValue> =
        gather_constants(k as u32, n_out as u32, top_k as u32).into();
    constants.push(ConstantValue::int(
        scratchy_target_metal::tape::constants::ConstSlot(3),
        0, // SiLU
    ));
    let routed: Vec<ConstantValue> = RoutedConstants {
        experts: NumExperts(num_experts as u32),
        program: RouteProgram {
            pre: RoutePre::SigmoidBias(LayerId(0)),
            scale: None,
            post: RoutePost::Renorm,
            expert_scale: None,
        },
    }
    .into();
    constants.extend(routed);
    let name = format!("affine_gather_qmv_gated_fast_bf16_s_bf16_gs_{group_size}_b_{bits}");
    let pipeline = baked_pipeline(&mdev.device, "quantized_qmv", &name, constants)
        .expect("routed gated b3 pipeline");

    // Bindings 0-13 per the kernel signature.
    let gw = buf_from_bytes(&mdev.device, &g_packed);
    let gs = buf_from_bytes(&mdev.device, bytes_of(&g_scales));
    let gb = buf_from_bytes(&mdev.device, bytes_of(&g_biases));
    let xb = buf_from_bytes(&mdev.device, bytes_of(&x));
    let uw = buf_from_bytes(&mdev.device, &u_packed);
    let us = buf_from_bytes(&mdev.device, bytes_of(&u_scales));
    let ub = buf_from_bytes(&mdev.device, bytes_of(&u_biases));
    let bufs: [&common::Buffer; 14] = [
        &gw,
        &gs,
        &gb,
        &xb,
        &rhs_buf,
        &gate_y,
        &uw,
        &us,
        &ub,
        &up_y,
        &logits_buf,
        &scores_buf,
        &scores_buf, // buffer(12) expert_scale — unused (EXPERT_SCALE off)
        &bias_buf,
    ];
    // Dispatch (pairs, n_out/8, 1) × (32, 4, 1): a pair a threadgroup along X.
    if !common::dispatch_threadgroups(
        &mdev.device,
        &pipeline,
        &bufs,
        MTLSize {
            width: pairs,
            height: n_out / 8,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 4,
            depth: 1,
        },
    ) {
        return;
    }

    // ── Reference ──
    // Routing: biased F32 sigmoids order the top-k (index-ascending
    // ties), unbiased sigmoids renormalized (bf16 rounding where the
    // kernel rounds: score = bf16(sigmoid) then renorm in f32/rounded).
    let sigmoid = |x: f32| 1.0f32 / (1.0 + (-x).exp());
    let mut picks_all = Vec::new();
    let mut scores_all = Vec::new();
    for n in 0..num_tokens {
        let row = &logits[n * num_experts..(n + 1) * num_experts];
        let mut order: Vec<usize> = (0..num_experts).collect();
        order.sort_by(|&a, &b| {
            (sigmoid(row[b].to_f32()) + bias[b])
                .total_cmp(&(sigmoid(row[a].to_f32()) + bias[a]))
                .then(b.cmp(&a))
        });
        let picks: Vec<usize> = order[..top_k].iter().rev().copied().collect();
        let mut unbiased: Vec<f32> = picks
            .iter()
            .map(|&i| bf16::from_f32(sigmoid(row[i].to_f32())).to_f32())
            .collect();
        let sum: f32 = unbiased.iter().sum();
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for u in &mut unbiased {
            *u *= inv;
        }
        picks_all.extend(picks);
        scores_all.extend(unbiased);
    }

    // rhs_indices written by the kernel (u32).
    let got_idx: Vec<u32> = {
        let ptr = rhs_buf.contents().as_ptr() as *const u32;
        unsafe { std::slice::from_raw_parts(ptr, pairs) }.to_vec()
    };
    let want_idx: Vec<u32> = picks_all.iter().map(|&i| i as u32).collect();
    assert_eq!(
        got_idx, want_idx,
        "routed picks mismatch: got {got_idx:?} want {want_idx:?}"
    );

    // gate_y holds act(gate)*up per pair row.
    let got = read_bf16(&gate_y, pairs * n_out);
    let w_pe = n_out * k * 3 / 8;
    let sb_pe = n_out * k / group_size;
    let mut max_err = 0.0_f32;
    let mut worst = (0usize, 0usize);
    for nk in 0..pairs {
        let e = picks_all[nk];
        let g = affine_qmv_b3_bf16_s_bf16(
            &g_packed[e * w_pe..(e + 1) * w_pe],
            &g_scales[e * sb_pe..(e + 1) * sb_pe],
            &g_biases[e * sb_pe..(e + 1) * sb_pe],
            &x[..k],
            1,
            n_out,
            k,
            group_size,
        );
        let u = affine_qmv_b3_bf16_s_bf16(
            &u_packed[e * w_pe..(e + 1) * w_pe],
            &u_scales[e * sb_pe..(e + 1) * sb_pe],
            &u_biases[e * sb_pe..(e + 1) * sb_pe],
            &x[..k],
            1,
            n_out,
            k,
            group_size,
        );
        for c in 0..n_out {
            let gv = g[c].to_f32();
            let uv = u[c].to_f32();
            let want = (gv / (1.0 + (-gv).exp())) * uv;
            let have = got[nk * n_out + c].to_f32();
            let err = (have - want).abs();
            if err > max_err {
                max_err = err;
                worst = (nk, c);
            }
        }
    }
    eprintln!(
        "routed gated b3 worst pair={} row={} got={:.4} max_err={max_err:.3e}",
        worst.0,
        worst.1,
        got[worst.0 * n_out + worst.1].to_f32(),
    );
    assert!(
        max_err < 0.3,
        "routed gated b3: pair {} row {}: max_err {max_err}",
        worst.0,
        worst.1
    );
    eprintln!("routed gated b3 E={num_experts} k={k}: picks exact, max_err {max_err:.3e}");
}

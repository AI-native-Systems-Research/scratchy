// SPDX-License-Identifier: Apache-2.0
//! 8-bit MLX-affine parity tests (Gemma4 MLP projections: 8-bit g64).
//!
//! The qmv/qmm_t template bodies are bits-generic MLX ports; the `_b_8_`
//! entry-point instantiations are new — these tests pin the kernels
//! against the CPU reference (`cpu_reference::affine_qmm_t_b8`) for the
//! production dtype combo (bf16 act × bf16 scales, Gemma4) and the
//! f16 × f16 combo.
//!
//! GPU tests — run with `--test-threads=1`.

mod common;

use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::cpu_reference::{affine_qmm_t_b8_bf16_s_bf16, affine_qmm_t_b8_f16};
use scratchy_target_metal::detect_device;
use scratchy_target_metal::quantized::{
    DequantDtype, QmmTKernel, ScaleDtype, pick_qmm_t_kernel, pick_qmv_kernel, qmm_t_dispatch_shape,
    qmm_t_kernel_name, qmv_dispatch_shape, qmv_kernel_name,
};
use scratchy_target_metal::shader_cache::ShaderCache;
use scratchy_target_metal::specialized_pipeline_cache::ConstantValue;

/// `(w, h, d)` triple from a dispatch-shape helper → `MTLSize`.
fn mtl_size(t: (u32, u32, u32)) -> MTLSize {
    MTLSize {
        width: t.0 as usize,
        height: t.1 as usize,
        depth: t.2 as usize,
    }
}

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;

struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_byte(&mut self) -> u8 {
        (self.next() >> 56) as u8
    }
    fn next_unit_f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

fn buffer_from_bytes(device: &Device, bytes: &[u8]) -> Buffer {
    let buf = device
        .newBufferWithLength_options(bytes.len().max(4), MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            buf.contents().as_ptr() as *mut u8,
            bytes.len(),
        );
    }
    buf
}

fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn zeroed_buffer(device: &Device, n_bytes: usize) -> Buffer {
    let buf = device
        .newBufferWithLength_options(n_bytes.max(4), MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe { std::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, n_bytes) };
    buf
}

/// Realistic 8-bit affine inputs: one byte per element; scale ≈
/// weight_range/255 with bias = range minimum, so dequantized weights
/// land in ±0.1 like real checkpoint tensors.
#[allow(clippy::type_complexity)]
fn make_inputs_b8_f32(
    seed: u64,
    n: usize,
    k: usize,
    m: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(k % group_size, 0);
    let n_groups = n * k / group_size;
    let mut rng = SplitMix64(seed);
    let packed: Vec<u8> = (0..n * k).map(|_| rng.next_byte()).collect();
    let scales: Vec<f32> = (0..n_groups)
        .map(|_| 0.0006 + 0.0004 * rng.next_unit_f32())
        .collect();
    let biases: Vec<f32> = (0..n_groups)
        .map(|_| -0.1 + 0.02 * rng.next_unit_f32())
        .collect();
    let x: Vec<f32> = (0..m * k)
        .map(|_| 2.0 * rng.next_unit_f32() - 1.0)
        .collect();
    (packed, scales, biases, x)
}

enum Op {
    Qmv,
    QmmT,
    /// Force the NAX (Apple9 MMA) qmm_t via `execute_with_kernel` —
    /// the b8 W-loader (byte-per-element dequant) parity gate.
    QmmTNax,
}

#[allow(clippy::too_many_arguments)]
fn run_case_bf16(op: Op, m: usize, n: usize, k: usize, group_size: usize, seed: u64, tol: f32) {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let (packed, scales_f, biases_f, x_f) = make_inputs_b8_f32(seed, n, k, m, group_size);
    let scales: Vec<half::bf16> = scales_f.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let biases: Vec<half::bf16> = biases_f.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let x: Vec<half::bf16> = x_f.iter().map(|&v| half::bf16::from_f32(v)).collect();

    let expected = affine_qmm_t_b8_bf16_s_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);

    let packed_buf = buffer_from_bytes(&device, &packed);
    let scales_buf = buffer_from_bytes(&device, as_bytes(&scales));
    let biases_buf = buffer_from_bytes(&device, as_bytes(&biases));
    let x_buf = buffer_from_bytes(&device, as_bytes(&x));
    let y_buf = zeroed_buffer(&device, m * n * 2);

    // Resolve the exact pipeline + dispatch grid the production
    // `execute*` helpers would pick (same kernel-name builder, same
    // function-constants, same dispatch-shape), then dispatch on the
    // MTL4 path. Buffer-index contract (both qmv and qmm_t): buffer(0)=
    // packed weight, (1)=scales, (2)=biases, (3)=x, (4)=y; K/N(/M) ride
    // as function constants 0/1(/2), not buffers.
    let cache = ShaderCache::new(device.clone()).expect("ShaderCache");
    let bits = 8u32;
    let (pipeline, tg, tpg) = match op {
        Op::Qmv => {
            let kernel = pick_qmv_kernel(n as u32, k as u32, bits);
            let name = qmv_kernel_name(
                kernel,
                DequantDtype::Bf16,
                ScaleDtype::Bf16,
                group_size as u32,
                bits,
                false,
            );
            let constants = [
                ConstantValue::int(0, k as i32),
                ConstantValue::int(1, n as i32),
            ];
            let p = cache
                .get_pipeline_specialized(&name, &constants)
                .expect("qmv b8 pipeline");
            let (tg, tpg) = qmv_dispatch_shape(kernel, m as u32, n as u32, 1);
            (p, tg, tpg)
        }
        Op::QmmT => {
            // `MetalAffineQmmT::execute` routes b8 SplitK → Standard.
            let kernel = match pick_qmm_t_kernel(
                m as u32,
                n as u32,
                k as u32,
                1,
                group_size as u32,
                false,
            ) {
                QmmTKernel::SplitK { .. } => QmmTKernel::Standard,
                other => other,
            };
            assert!(
                matches!(kernel, QmmTKernel::Standard),
                "b8 must route to Standard (got {kernel:?})"
            );
            let aligned_n = (n as u32).is_multiple_of(32);
            let name = qmm_t_kernel_name(
                kernel,
                DequantDtype::Bf16,
                ScaleDtype::Bf16,
                group_size as u32,
                bits,
                aligned_n,
            );
            let constants = [
                ConstantValue::int(0, k as i32),
                ConstantValue::int(1, n as i32),
                ConstantValue::int(2, m as i32),
            ];
            let p = cache
                .get_pipeline_specialized(&name, &constants)
                .expect("qmm_t b8 pipeline");
            let (tg, tpg) = qmm_t_dispatch_shape(kernel, m as u32, n as u32, 1);
            (p, tg, tpg)
        }
        Op::QmmTNax => {
            let kernel = QmmTKernel::Nax;
            let aligned_n = (n as u32).is_multiple_of(64);
            let name = qmm_t_kernel_name(
                kernel,
                DequantDtype::Bf16,
                ScaleDtype::Bf16,
                group_size as u32,
                bits,
                aligned_n,
            );
            let constants = [
                ConstantValue::int(0, k as i32),
                ConstantValue::int(1, n as i32),
                ConstantValue::int(2, m as i32),
            ];
            let p = cache
                .get_pipeline_specialized(&name, &constants)
                .expect("qmm_t nax b8 pipeline");
            let (tg, tpg) = qmm_t_dispatch_shape(kernel, m as u32, n as u32, 1);
            (p, tg, tpg)
        }
    };

    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[&packed_buf, &scales_buf, &biases_buf, &x_buf, &y_buf],
        mtl_size(tg),
        mtl_size(tpg),
    ) {
        return;
    }

    let got: Vec<f32> = unsafe {
        std::slice::from_raw_parts(y_buf.contents().as_ptr() as *const half::bf16, m * n)
            .iter()
            .map(|h| h.to_f32())
            .collect()
    };
    let mut max_err = 0f32;
    for i in 0..m * n {
        max_err = max_err.max((got[i] - expected[i].to_f32()).abs());
    }
    assert!(max_err < tol, "b8 bf16 max_err {max_err} (tol {tol})");
}

#[test]
fn qmv_b8_bf16_gemma4_gate_shape() {
    // Gemma4 MLP gate decode: M=1, N=intermediate 15360 (trimmed to
    // 1920 for CPU-ref speed; same K), K=hidden 3840.
    run_case_bf16(Op::Qmv, 1, 1920, 3840, 64, 11, 0.06);
}

#[test]
fn qmv_b8_bf16_gemma4_down_shape() {
    // Gemma4 MLP down decode: K=intermediate 15360, N=hidden (trimmed).
    run_case_bf16(Op::Qmv, 1, 480, 15360, 64, 13, 0.12);
}

#[test]
fn qmm_t_b8_bf16_prefill_shape() {
    // Prefill M=64 through the Standard qmm_t.
    run_case_bf16(Op::QmmT, 64, 256, 1536, 64, 17, 0.05);
}

#[test]
fn qmm_t_nax_b8_bf16_prefill_shape() {
    // Prefill M=64 through the NAX qmm_t (b8 byte-per-element
    // W-loader, the Gemma4 MLP prefill path on M5+). Skips on
    // non-NAX-capable hardware.
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    if !scratchy_target_metal::targets::is_nax_capable(di.profile.generation) {
        eprintln!(
            "skipping NAX b8 parity on non-NAX hardware ({:?})",
            di.profile.generation
        );
        return;
    }
    // N multiple of 64 (aligned) + an unaligned-N case for the
    // N-tail zeroing path.
    run_case_bf16(Op::QmmTNax, 64, 256, 1536, 64, 19, 0.05);
    run_case_bf16(Op::QmmTNax, 64, 224, 1536, 64, 29, 0.05);
    // M-tail: M not a multiple of 64.
    run_case_bf16(Op::QmmTNax, 40, 256, 1536, 64, 31, 0.05);
    // gs=128 instantiation.
    run_case_bf16(Op::QmmTNax, 64, 256, 1536, 128, 37, 0.05);
}

#[test]
fn qwen_gdn_b8_bf16_projection_shapes() {
    // Qwen3.5 OptiQ ships the Gated-DeltaNet in_proj_* at 8-bit. The
    // small-N in_proj_a / in_proj_b (N = num_v_heads = 32) is a shape no
    // other shipping 8-bit model exercises (gemma's 8-bit MLP is large-N),
    // so verify the b8 qmv (decode M=1) + qmm_t (prefill M=20) match the
    // CPU reference at every GDN projection shape.
    //
    // in_proj_a / in_proj_b: N=32, K=hidden=2048.
    run_case_bf16(Op::Qmv, 1, 32, 2048, 64, 41, 0.05);
    run_case_bf16(Op::QmmT, 20, 32, 2048, 64, 43, 0.05);
    // out_proj: N=hidden=2048, K=value_dim=4096.
    run_case_bf16(Op::Qmv, 1, 2048, 4096, 64, 45, 0.08);
    run_case_bf16(Op::QmmT, 20, 2048, 4096, 64, 47, 0.08);
    // in_proj_z (N=value_dim=4096) / in_proj_qkv (N=8192): N trimmed to 512
    // for CPU-ref speed, same K=hidden=2048 + kernel path.
    run_case_bf16(Op::Qmv, 1, 512, 2048, 64, 49, 0.06);
    run_case_bf16(Op::QmmT, 20, 512, 2048, 64, 51, 0.06);
}

#[test]
fn qwen_lm_head_b8_bf16_huge_n() {
    // Qwen3.5 OptiQ ships lm_head at 8-bit (uniform is 4-bit). N=vocab=248320,
    // K=hidden=2048 — a huge-N 8-bit qmv (decode / prefill-last-token) that no
    // other model exercises. Trim N to keep the CPU reference fast but stay on
    // the same qmv_fast _b_8 kernel + dispatch grid.
    run_case_bf16(Op::Qmv, 1, 65536, 2048, 64, 61, 0.06);
    run_case_bf16(Op::Qmv, 1, 8192, 2048, 64, 63, 0.06);
}

#[test]
fn qmv_b8_f16_matches_cpu() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let (m, n, k, gs) = (1usize, 512usize, 1024usize, 64usize);
    let (packed, scales_f, biases_f, x_f) = make_inputs_b8_f32(23, n, k, m, gs);
    let scales: Vec<half::f16> = scales_f.iter().map(|&v| half::f16::from_f32(v)).collect();
    let biases: Vec<half::f16> = biases_f.iter().map(|&v| half::f16::from_f32(v)).collect();
    let x: Vec<half::f16> = x_f.iter().map(|&v| half::f16::from_f32(v)).collect();
    let expected = affine_qmm_t_b8_f16(&packed, &scales, &biases, &x, m, n, k, gs);

    let packed_buf = buffer_from_bytes(&device, &packed);
    let scales_buf = buffer_from_bytes(&device, as_bytes(&scales));
    let biases_buf = buffer_from_bytes(&device, as_bytes(&biases));
    let x_buf = buffer_from_bytes(&device, as_bytes(&x));
    let y_buf = zeroed_buffer(&device, m * n * 2);

    let cache = ShaderCache::new(device.clone()).expect("ShaderCache");
    let bits = 8u32;
    let kernel = pick_qmv_kernel(n as u32, k as u32, bits);
    let name = qmv_kernel_name(
        kernel,
        DequantDtype::F16,
        ScaleDtype::F16,
        gs as u32,
        bits,
        false,
    );
    let constants = [
        ConstantValue::int(0, k as i32),
        ConstantValue::int(1, n as i32),
    ];
    let pipeline = cache
        .get_pipeline_specialized(&name, &constants)
        .expect("qmv b8 f16 pipeline");
    let (tg, tpg) = qmv_dispatch_shape(kernel, m as u32, n as u32, 1);
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[&packed_buf, &scales_buf, &biases_buf, &x_buf, &y_buf],
        mtl_size(tg),
        mtl_size(tpg),
    ) {
        return;
    }

    let got: Vec<f32> = unsafe {
        std::slice::from_raw_parts(y_buf.contents().as_ptr() as *const half::f16, m * n)
            .iter()
            .map(|h| h.to_f32())
            .collect()
    };
    let mut max_err = 0f32;
    for i in 0..m * n {
        max_err = max_err.max((got[i] - expected[i].to_f32()).abs());
    }
    assert!(max_err < 0.02, "b8 f16 max_err {max_err}");
}

#[test]
fn qwen_shared_expert_gate_b8_bf16_n1() {
    // OptiQ ships mlp.shared_expert_gate at 8-bit (uniform is 4-bit): a
    // scalar gate, N=1, K=hidden=2048. Verify the b8 qmv (decode) + qmm_t
    // (prefill M=20) at N=1 match the CPU reference.
    run_case_bf16(Op::Qmv, 1, 1, 2048, 64, 71, 0.03);
    run_case_bf16(Op::QmmT, 20, 1, 2048, 64, 73, 0.03);
}

#[test]
fn qwen_gdn_b8_bf16_nax_prefill_shapes() {
    // Production prefill on M5+ routes 8-bit dense projections to the NAX
    // qmm_t (the earlier run showed AffineQmmTNax). The GDN in_proj_a/b are
    // N=32 (< the 64-wide NAX tile) at 8-bit — an unaligned, sub-tile NAX b8
    // shape no other model exercises (gemma's 8-bit is large-N). M=20 prefill
    // is also a partial M-tile. Verify NAX b8 vs CPU ref at the GDN shapes.
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    if !scratchy_target_metal::targets::is_nax_capable(di.profile.generation) {
        eprintln!(
            "skipping NAX b8 GDN on non-NAX hardware ({:?})",
            di.profile.generation
        );
        return;
    }
    // in_proj_a / in_proj_b: N=32 (sub-tile, unaligned), K=2048.
    run_case_bf16(Op::QmmTNax, 20, 32, 2048, 64, 81, 0.08);
    // out_proj: N=2048 (aligned), K=4096.
    run_case_bf16(Op::QmmTNax, 20, 2048, 4096, 64, 83, 0.10);
    // in_proj_z/qkv trimmed to 512 (aligned) K=2048.
    run_case_bf16(Op::QmmTNax, 20, 512, 2048, 64, 85, 0.08);
    // A second sub-tile N to bracket (N=64 exactly one tile, N=40 partial).
    run_case_bf16(Op::QmmTNax, 20, 64, 2048, 64, 87, 0.08);
    run_case_bf16(Op::QmmTNax, 20, 40, 2048, 64, 89, 0.08);
}

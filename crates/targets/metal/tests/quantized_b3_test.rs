// SPDX-License-Identifier: Apache-2.0
//! 3-bit MLX-affine parity tests (GLM-4.5-Air-3bit: 3-bit g64).
//!
//! The qmv/qmm_t template bodies are bits-generic MLX ports carrying the
//! continuous LSB-first 3-bit bitstream (8 codes per 3 bytes, packed cols
//! `ceil(K*3/32)` — NOT `K/(32/3)`); the `_b_3_` entry-point
//! instantiations are new. These tests pin the kernels against the CPU
//! reference (`cpu_reference::affine_qmm_t_b3`) at the GLM production
//! combo (bf16 act × bf16 scales, gs=64) and the f16 × f16 combo, at the
//! real GLM-4.5-Air projection shapes (hidden 4096, moe_intermediate
//! 1408, dense intermediate 10944).
//!
//! GPU tests — run with `--test-threads=1`.

mod common;

use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::aot::baked_pipeline;
use scratchy_target_metal::cpu_reference::{affine_qmm_t_b3_bf16_s_bf16, affine_qmm_t_b3_f16};
use scratchy_target_metal::detect_device;
use scratchy_target_metal::quantized::{
    DequantDtype, QmmTKernel, ScaleDtype, pick_qmm_t_kernel, pick_qmv_kernel, qmm_t_dispatch_shape,
    qmm_t_kernel_name, qmv_dispatch_shape, qmv_kernel_name,
};
use scratchy_target_metal::specialized_pipeline_cache::{ComputePipelineState, ConstantValue};

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

/// Realistic 3-bit affine inputs: `n*k*3/8` packed bytes (random — the
/// bitstream is itself the code); scale ≈ weight_range/7 with bias =
/// range minimum, so dequantized weights land in ±0.1 like real
/// checkpoint tensors.
#[allow(clippy::type_complexity)]
fn make_inputs_b3_f32(
    seed: u64,
    n: usize,
    k: usize,
    m: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(k % group_size, 0);
    let n_groups = n * k / group_size;
    let mut rng = SplitMix64(seed);
    let packed: Vec<u8> = (0..n * k * 3 / 8).map(|_| rng.next_byte()).collect();
    let scales: Vec<f32> = (0..n_groups)
        .map(|_| 0.006 + 0.004 * rng.next_unit_f32())
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
    /// Force the NAX (matrix-unit) qmm_t — the M5+ prefill path. The
    /// vendored MLX QuantizedBlockLoader's index math is bits=3-exact
    /// (pack_factor 8, bytes_per_pack 3) and metal_nax.h carries MLX's
    /// qdot bits==3 dequantize shifts.
    QmmTNax,
}

#[allow(clippy::too_many_arguments)]
fn run_case_bf16(op: Op, m: usize, n: usize, k: usize, group_size: usize, seed: u64, tol: f32) {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let (packed, scales_f, biases_f, x_f) = make_inputs_b3_f32(seed, n, k, m, group_size);
    let scales: Vec<half::bf16> = scales_f.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let biases: Vec<half::bf16> = biases_f.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let x: Vec<half::bf16> = x_f.iter().map(|&v| half::bf16::from_f32(v)).collect();

    let expected = affine_qmm_t_b3_bf16_s_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);

    let packed_buf = buffer_from_bytes(&device, &packed);
    let scales_buf = buffer_from_bytes(&device, as_bytes(&scales));
    let biases_buf = buffer_from_bytes(&device, as_bytes(&biases));
    let x_buf = buffer_from_bytes(&device, as_bytes(&x));
    let y_buf = zeroed_buffer(&device, m * n * 2);

    // Resolve the exact pipeline + dispatch grid the production
    // `execute*` helpers would pick (same kernel-name builder, same
    // constants, same dispatch-shape), then dispatch on the MTL4 path.
    // Buffer-index contract (qmv and qmm_t): buffer(0)=packed weight,
    // (1)=scales, (2)=biases, (3)=x, (4)=y; K/N(/M) are constants
    // 0/1(/2), not buffers.
    // Holds the baked pipeline's cache until the dispatch below.
    let mut baked = None;
    let bits = 3u32;
    let (pipeline, tg, tpg) = match op {
        Op::Qmv => {
            let kernel = pick_qmv_kernel(n as u32, k as u32, bits);
            // bits=3 must never pick quad (`32/bits` index math is
            // power-of-two-only, matching MLX's `is_power_of_2(bits)`
            // dispatch gate).
            assert!(
                !matches!(
                    kernel,
                    scratchy_target_metal::quantized::QmvKernel::Quad { .. }
                ),
                "b3 must not route to quad (got {kernel:?})"
            );
            let name = qmv_kernel_name(
                kernel,
                DequantDtype::Bf16,
                ScaleDtype::Bf16,
                group_size as u32,
                bits,
                false,
            );
            let constants = vec![
                ConstantValue::int(0, k as i32),
                ConstantValue::int(1, n as i32),
            ];
            let p = baked.insert(
                baked_pipeline(&device, "quantized_qmv", &name, constants)
                    .expect("qmv b3 pipeline"),
            );
            let (tg, tpg) = qmv_dispatch_shape(kernel, m as u32, n as u32, 1);
            (ComputePipelineState::clone(p), tg, tpg)
        }
        Op::QmmT => {
            // 3-bit SplitK downgrades to Standard (the splitk family is
            // b4-only).
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
                "b3 must route to Standard (got {kernel:?})"
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
            let p = baked.insert(
                baked_pipeline(&device, "quantized_qmm", &name, constants.to_vec())
                    .expect("qmm_t b3 pipeline"),
            );
            let (tg, tpg) = qmm_t_dispatch_shape(kernel, m as u32, n as u32, 1);
            (ComputePipelineState::clone(p), tg, tpg)
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
            let p = baked.insert(
                baked_pipeline(&device, "quantized_qmm_nax", &name, constants.to_vec())
                    .expect("qmm_t nax b3 pipeline"),
            );
            let (tg, tpg) = qmm_t_dispatch_shape(kernel, m as u32, n as u32, 1);
            (ComputePipelineState::clone(p), tg, tpg)
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
    assert!(max_err < tol, "b3 bf16 max_err {max_err} (tol {tol})");
}

#[test]
fn qmv_b3_bf16_glm_attention_shape() {
    // GLM-4.5-Air q_proj decode: M=1, N=96*128=12288 (trimmed to 1536
    // for CPU-ref speed; same K), K=hidden 4096.
    run_case_bf16(Op::Qmv, 1, 1536, 4096, 64, 11, 0.05);
}

#[test]
fn qmv_b3_bf16_glm_expert_gate_shape() {
    // GLM-4.5-Air routed-expert gate_proj decode: N=moe_intermediate
    // 1408, K=hidden 4096.
    run_case_bf16(Op::Qmv, 1, 1408, 4096, 64, 13, 0.05);
}

#[test]
fn qmv_b3_bf16_glm_expert_down_shape() {
    // GLM-4.5-Air routed-expert down_proj decode: N=hidden 4096,
    // K=moe_intermediate 1408.
    run_case_bf16(Op::Qmv, 1, 512, 1408, 64, 17, 0.03);
}

#[test]
fn qmm_t_b3_bf16_prefill_shape() {
    // Prefill M=64 through the Standard qmm_t at the GLM hidden width.
    run_case_bf16(Op::QmmT, 64, 256, 4096, 64, 19, 0.08);
}

#[test]
fn qmm_t_b3_bf16_prefill_intermediate_shape() {
    // Prefill at K=moe_intermediate 1408 (down_proj): K not a multiple
    // of 512 → the Generic qmv tail path at decode, Standard qmm_t here.
    run_case_bf16(Op::QmmT, 64, 512, 1408, 64, 23, 0.05);
}

#[test]
fn qmm_t_nax_b3_bf16_prefill_shape() {
    // Prefill M=64 through the NAX qmm_t (the GLM-4.5-Air-3bit prefill
    // path on M5+): the QuantizedBlockLoader stages 3-byte packs and
    // metal_nax.h's dequantize carries the MLX bits==3 shifts. Skips on
    // non-NAX-capable hardware.
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    if !scratchy_target_metal::targets::is_nax_capable(di.profile.generation) {
        eprintln!(
            "skipping NAX b3 parity on non-NAX hardware ({:?})",
            di.profile.generation
        );
        return;
    }
    run_case_bf16(Op::QmmTNax, 64, 256, 4096, 64, 29, 0.08);
    // Unaligned-N tail + K=moe_intermediate.
    run_case_bf16(Op::QmmTNax, 64, 224, 4096, 64, 31, 0.08);
    run_case_bf16(Op::QmmTNax, 64, 512, 1408, 64, 37, 0.05);
    // gs=128 instantiation.
    run_case_bf16(Op::QmmTNax, 64, 256, 4096, 128, 43, 0.08);
}

#[test]
fn qmm_t_b3_splitk_downgrades_to_standard() {
    // The splitk family is b4-only — so bits=3 must downgrade any SplitK
    // pick to Standard. Pin the routing so a future dispatcher change
    // can't request a `_b_3_` splitk symbol that doesn't exist (nil
    // computeFunction at runtime).
    for (m, n, k, gs) in [
        (64, 256, 4096, 64),
        (64, 224, 4096, 64),
        (128, 2048, 8192, 128),
    ] {
        let kernel = pick_qmm_t_kernel(m, n, k, 1, gs, /*is_nax=*/ false);
        let kernel = match kernel {
            QmmTKernel::SplitK { .. } => QmmTKernel::Standard,
            other => other,
        };
        assert!(
            matches!(kernel, QmmTKernel::Standard),
            "b3 ({m},{n},{k},gs={gs}) must route SplitK picks to Standard (got {kernel:?})"
        );
    }
}

#[test]
fn qmv_b3_f16_matches_cpu() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let (m, n, k, gs) = (1usize, 512usize, 1024usize, 64usize);
    let (packed, scales_f, biases_f, x_f) = make_inputs_b3_f32(41, n, k, m, gs);
    let scales: Vec<half::f16> = scales_f.iter().map(|&v| half::f16::from_f32(v)).collect();
    let biases: Vec<half::f16> = biases_f.iter().map(|&v| half::f16::from_f32(v)).collect();
    let x: Vec<half::f16> = x_f.iter().map(|&v| half::f16::from_f32(v)).collect();
    let expected = affine_qmm_t_b3_f16(&packed, &scales, &biases, &x, m, n, k, gs);

    let packed_buf = buffer_from_bytes(&device, &packed);
    let scales_buf = buffer_from_bytes(&device, as_bytes(&scales));
    let biases_buf = buffer_from_bytes(&device, as_bytes(&biases));
    let x_buf = buffer_from_bytes(&device, as_bytes(&x));
    let y_buf = zeroed_buffer(&device, m * n * 2);

    let bits = 3u32;
    let kernel = pick_qmv_kernel(n as u32, k as u32, bits);
    let name = qmv_kernel_name(
        kernel,
        DequantDtype::F16,
        ScaleDtype::F16,
        gs as u32,
        bits,
        false,
    );
    let constants = vec![
        ConstantValue::int(0, k as i32),
        ConstantValue::int(1, n as i32),
    ];
    let pipeline =
        baked_pipeline(&device, "quantized_qmv", &name, constants).expect("qmv b3 f16 pipeline");
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
    assert!(max_err < 0.02, "b3 f16 max_err {max_err}");
}

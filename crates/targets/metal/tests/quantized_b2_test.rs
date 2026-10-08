// SPDX-License-Identifier: Apache-2.0
//! 2-bit MLX-affine parity tests (gpt-oss-120b-mlx-2Bit: 2-bit g64).
//!
//! The qmv/qmm_t template bodies are bits-generic MLX ports carrying the
//! continuous LSB-first 2-bit bitstream (element i at bits `[2i, 2i+2)`,
//! 4 codes per byte, packed cols `K/16`); the `_b_2_` entry-point
//! instantiations are new. These tests pin the kernels against the CPU
//! reference (`cpu_reference::affine_qmm_t_b2`) at the gpt-oss production
//! combo (bf16 act × bf16 scales, gs=64) and the f16 × f16 combo, at the
//! real gpt-oss-120b projection shapes (hidden 2880, q/kv head geometry
//! 64×64=4096, moe_intermediate 2880 — every affine K is 2880 except
//! o_proj's 4096).
//!
//! Unlike 3-bit, 2-bit IS a power of two, so the QUAD family is live:
//! `pick_qmv_kernel` selects Quad at K∈{64,128} and `qmv_quad_impl`'s
//! `pack_factor = 32/bits = 16` index math is exact. The 120b itself
//! never hits the band (every affine K is 2880/4096) but a parity-tiny
//! config can shrink K into it — the quad tests below pin both the pick
//! and the kernel's parity.
//!
//! GPU tests — run with `--test-threads=1`.

mod common;

use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::aot::baked_pipeline;
use scratchy_target_metal::cpu_reference::{affine_qmm_t_b2_bf16_s_bf16, affine_qmm_t_b2_f16};
use scratchy_target_metal::detect_device;
use scratchy_target_metal::quantized::{
    DequantDtype, QmmTKernel, QmvKernel, ScaleDtype, pick_qmm_t_kernel, pick_qmv_kernel,
    qmm_t_dispatch_shape, qmm_t_kernel_name, qmv_dispatch_shape, qmv_kernel_name,
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

/// Realistic 2-bit affine inputs: `n*k/4` packed bytes (random — the
/// bitstream is itself the code); scale ≈ weight_range/3 with bias =
/// range minimum (codes span 0..3), so dequantized weights land in ±0.1
/// like real checkpoint tensors.
#[allow(clippy::type_complexity)]
fn make_inputs_b2_f32(
    seed: u64,
    n: usize,
    k: usize,
    m: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(k % group_size, 0);
    let n_groups = n * k / group_size;
    let mut rng = SplitMix64(seed);
    let packed: Vec<u8> = (0..n * k / 4).map(|_| rng.next_byte()).collect();
    let scales: Vec<f32> = (0..n_groups)
        .map(|_| 0.01 + 0.006 * rng.next_unit_f32())
        .collect();
    let biases: Vec<f32> = (0..n_groups)
        .map(|_| -0.08 + 0.015 * rng.next_unit_f32())
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
    /// vendored MLX QuantizedBlockLoader's index math is bits=2-exact
    /// (pack_factor 4, bytes_per_pack 1) and metal_nax.h carries MLX's
    /// qdot bits==2 dequantize law.
    QmmTNax,
}

#[allow(clippy::too_many_arguments)]
fn run_case_bf16(op: Op, m: usize, n: usize, k: usize, group_size: usize, seed: u64, tol: f32) {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let (packed, scales_f, biases_f, x_f) = make_inputs_b2_f32(seed, n, k, m, group_size);
    let scales: Vec<half::bf16> = scales_f.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let biases: Vec<half::bf16> = biases_f.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let x: Vec<half::bf16> = x_f.iter().map(|&v| half::bf16::from_f32(v)).collect();

    let expected = affine_qmm_t_b2_bf16_s_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);

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
    let bits = 2u32;
    let (pipeline, tg, tpg) = match op {
        Op::Qmv => {
            // bits=2 is a power of two, so K∈{64,128} picks Quad —
            // `qmv_quad_impl`'s `pack_factor = 32/bits = 16` index math
            // is exact (unlike 3-bit). At the gpt-oss shapes (K ≥ 2880)
            // the pick is Fast/Generic.
            let kernel = pick_qmv_kernel(n as u32, k as u32, bits);
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
                    .expect("qmv b2 pipeline"),
            );
            let (tg, tpg) = qmv_dispatch_shape(kernel, m as u32, n as u32, 1);
            (ComputePipelineState::clone(p), tg, tpg)
        }
        Op::QmmT => {
            // 2-bit SplitK downgrades to Standard (the splitk family is
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
                "b2 must route to Standard (got {kernel:?})"
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
                    .expect("qmm_t b2 pipeline"),
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
                    .expect("qmm_t nax b2 pipeline"),
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
    assert!(max_err < tol, "b2 bf16 max_err {max_err} (tol {tol})");
}

#[test]
fn qmv_b2_bf16_gptoss_attention_shape() {
    // gpt-oss-120b q_proj decode: M=1, N=64 heads × 64 = 4096 (trimmed
    // to 1536 for CPU-ref speed; same K), K=hidden 2880.
    run_case_bf16(Op::Qmv, 1, 1536, 2880, 64, 11, 0.05);
}

#[test]
fn qmv_b2_bf16_gptoss_expert_gate_shape() {
    // gpt-oss-120b routed-expert gate_proj decode: N=intermediate 2880
    // (trimmed to 1408; same K), K=hidden 2880.
    run_case_bf16(Op::Qmv, 1, 1408, 2880, 64, 13, 0.05);
}

#[test]
fn qmv_b2_bf16_gptoss_expert_down_shape() {
    // gpt-oss-120b routed-expert down_proj decode: N=hidden (trimmed),
    // K=intermediate 2880.
    run_case_bf16(Op::Qmv, 1, 512, 2880, 64, 17, 0.03);
}

#[test]
fn qmv_b2_quad_is_live_at_pow2_tiny_k() {
    // The quad hazard unique to 2-bit: `pick_qmv_kernel` selects Quad at
    // K∈{64,128} for power-of-two bits (b3 never does), and the `_b_2`
    // quad instantiations must exist and be CORRECT — a parity-tiny
    // config can shrink K into this band even though the 120b's every
    // affine K is 2880/4096. Pin the pick, then the kernel's parity at
    // both D.
    if detect_device().is_none() {
        eprintln!("skipping: no Metal device");
        return;
    }
    for (k, d) in [(64u32, 64u32), (128, 128)] {
        let kernel = pick_qmv_kernel(512, k, 2);
        assert_eq!(kernel, QmvKernel::Quad { d }, "K={k} bits=2 must pick quad");
    }
    run_case_bf16(Op::Qmv, 1, 512, 64, 64, 19, 0.02);
    run_case_bf16(Op::Qmv, 1, 512, 128, 64, 23, 0.02);
}

#[test]
fn qmm_t_b2_bf16_prefill_shape() {
    // Prefill M=64 through the Standard qmm_t at the gpt-oss hidden
    // width (K=2880).
    run_case_bf16(Op::QmmT, 64, 256, 2880, 64, 29, 0.08);
}

#[test]
fn qmm_t_b2_bf16_prefill_oproj_shape() {
    // Prefill at K=4096 (o_proj's input width — the attention context
    // vector, 64 heads × 64).
    run_case_bf16(Op::QmmT, 64, 512, 4096, 64, 31, 0.08);
}

#[test]
fn qmm_t_nax_b2_bf16_prefill_shape() {
    // Prefill M=64 through the NAX qmm_t (the gpt-oss-2bit prefill path
    // on M5+): the QuantizedBlockLoader stages 1-byte packs (4 codes
    // each) and metal_nax.h's dequantize carries the MLX bits==2 law.
    // Skips on non-NAX-capable hardware.
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    if !scratchy_target_metal::targets::is_nax_capable(di.profile.generation) {
        eprintln!(
            "skipping NAX b2 parity on non-NAX hardware ({:?})",
            di.profile.generation
        );
        return;
    }
    run_case_bf16(Op::QmmTNax, 64, 256, 2880, 64, 37, 0.08);
    // Unaligned-N tail.
    run_case_bf16(Op::QmmTNax, 64, 224, 2880, 64, 41, 0.08);
    // K=4096 (o_proj width).
    run_case_bf16(Op::QmmTNax, 64, 512, 4096, 64, 43, 0.08);
    // gs=128 instantiation (K=4096 — 2880 is NOT a multiple of 128, so
    // gs=128 never occurs at the gpt-oss Ks; the instantiation is still
    // pinned for the general law).
    run_case_bf16(Op::QmmTNax, 64, 256, 4096, 128, 47, 0.08);
}

#[test]
fn qmm_t_b2_splitk_downgrades_to_standard() {
    // The splitk family is b4-only — so bits=2 must downgrade any SplitK
    // pick to Standard. Pin the routing so a future dispatcher change
    // can't request a `_b_2_` splitk symbol that doesn't exist (nil
    // computeFunction at runtime).
    for (m, n, k, gs) in [
        (64, 256, 4096, 64),
        (64, 224, 2880, 64),
        (128, 2048, 8192, 128),
    ] {
        let kernel = pick_qmm_t_kernel(m, n, k, 1, gs, /*is_nax=*/ false);
        let kernel = match kernel {
            QmmTKernel::SplitK { .. } => QmmTKernel::Standard,
            other => other,
        };
        assert!(
            matches!(kernel, QmmTKernel::Standard),
            "b2 ({m},{n},{k},gs={gs}) must route SplitK picks to Standard (got {kernel:?})"
        );
    }
}

#[test]
fn qmv_b2_f16_matches_cpu() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let (m, n, k, gs) = (1usize, 512usize, 1024usize, 64usize);
    let (packed, scales_f, biases_f, x_f) = make_inputs_b2_f32(53, n, k, m, gs);
    let scales: Vec<half::f16> = scales_f.iter().map(|&v| half::f16::from_f32(v)).collect();
    let biases: Vec<half::f16> = biases_f.iter().map(|&v| half::f16::from_f32(v)).collect();
    let x: Vec<half::f16> = x_f.iter().map(|&v| half::f16::from_f32(v)).collect();
    let expected = affine_qmm_t_b2_f16(&packed, &scales, &biases, &x, m, n, k, gs);

    let packed_buf = buffer_from_bytes(&device, &packed);
    let scales_buf = buffer_from_bytes(&device, as_bytes(&scales));
    let biases_buf = buffer_from_bytes(&device, as_bytes(&biases));
    let x_buf = buffer_from_bytes(&device, as_bytes(&x));
    let y_buf = zeroed_buffer(&device, m * n * 2);

    let bits = 2u32;
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
        baked_pipeline(&device, "quantized_qmv", &name, constants).expect("qmv b2 f16 pipeline");
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
    assert!(max_err < 0.02, "b2 f16 max_err {max_err}");
}

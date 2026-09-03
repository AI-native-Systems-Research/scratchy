// SPDX-License-Identifier: Apache-2.0
//
// `affine_qmm_n_*` parity test: kernel ≡ CPU reference within bf16
// accumulation noise. Mirrors `tests/quantized_qmm_test.rs`'s
// structure but exercises the transpose=false matmul path (W stored
// as `[K, N]` packed; scales/biases laid out as `[K, N/gs]`).
//
// CPU reference: dequantize the packed int4 weight per MLX
// `affine_dequantize` math (4-bit branch — same `(scale, bias)`
// per `group_size` columns of N), then do a row-major matmul with
// f32 accumulator and bf16 cast on the way out. The N-axis is the
// quantized axis (last axis in the W tensor stored as `[K, N]`).

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::cpu_reference::affine_qmm_n_b4_bf16 as cpu_qmm_n_bf16;
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::quantized::{
    DequantDtype, ScaleDtype, qmm_n_dispatch_shape, qmm_n_kernel_name,
};
use scratchy_target_metal::shader_cache::{ComputePipelineState, ShaderCache};
use scratchy_target_metal::specialized_pipeline_cache::ConstantValue;

mod common;

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Device = Retained<ProtocolObject<dyn MTLDevice>>;

/// Build the specialized `affine_qmm_n` pipeline exactly as
/// `MetalAffineQmmN::execute` does: kernel name from
/// `qmm_n_kernel_name`, K/N/M baked as function constants 0/1/2.
fn qmm_n_pipeline(
    device: &Device,
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    m: u32,
    n: u32,
    k: u32,
    group_size: u32,
) -> ComputePipelineState {
    let cache = ShaderCache::new(device.clone()).expect("ShaderCache");
    let name = qmm_n_kernel_name(dtype, scale_dtype, group_size, 4);
    let constants = [
        ConstantValue::int(0, k as i32),
        ConstantValue::int(1, n as i32),
        ConstantValue::int(2, m as i32),
    ];
    cache
        .get_pipeline_specialized(&name, &constants)
        .expect("qmm_n pipeline")
}

/// `(threadgroups, threads_per_threadgroup)` for qmm_n at B=1, from
/// the production `qmm_n_dispatch_shape`.
fn qmm_n_grid(m: u32, n: u32) -> (MTLSize, MTLSize) {
    let (tg, tpg) = qmm_n_dispatch_shape(m, n, 1);
    (
        MTLSize {
            width: tg.0 as usize,
            height: tg.1 as usize,
            depth: tg.2 as usize,
        },
        MTLSize {
            width: tpg.0 as usize,
            height: tpg.1 as usize,
            depth: tpg.2 as usize,
        },
    )
}

// ─────────────────────────────────────────────────────────────────
// Test helpers (mirror tests/quantized_qmm_test.rs)
// ─────────────────────────────────────────────────────────────────

fn buffer_from_bytes(device: &Device, bytes: &[u8]) -> Buffer {
    unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
            .expect("newBufferWithBytes returned nil")
    }
}

fn zeroed_buffer(device: &Device, n_bytes: usize) -> Buffer {
    device
        .newBufferWithLength_options(n_bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBufferWithLength returned nil")
}

fn read_buffer_bf16(buf: &Buffer, n_elements: usize) -> Vec<half::bf16> {
    let ptr = buf.contents().as_ptr() as *const half::bf16;
    unsafe { std::slice::from_raw_parts(ptr, n_elements) }.to_vec()
}

/// SplitMix64 — same deterministic PRNG as the qmm test.
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
        (self.next() & 0xff) as u8
    }
    fn next_unit_f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// Make `[K, N]` packed W + `[K, N/gs]` scales/biases + `[M, K]` x
/// for the transpose=false layout. The packing convention follows
/// MLX `affine_dequantize` at `quantized.h:2536`: each byte holds
/// two nibbles, low nibble = output element `o`, high nibble =
/// output element `o+1`. For W shape `[K, N]` stored as `[K,
/// N/pack_factor]` u8 (= `[K, N/2]` bytes for bits=4), the byte at
/// position `(k, j)` holds W[k, 2*j] (low) and W[k, 2*j+1] (high).
fn make_inputs_bf16(
    seed: u64,
    n: usize,
    k: usize,
    m: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<half::f16>, Vec<half::f16>, Vec<half::bf16>) {
    assert_eq!(n % group_size, 0, "qmm_n requires N % group_size == 0");
    let n_bytes = k * n / 2; // pack_factor = 2 nibbles/byte
    let n_groups = k * n / group_size;

    let mut rng = SplitMix64(seed);
    let packed: Vec<u8> = (0..n_bytes).map(|_| rng.next_byte()).collect();
    // P10b: scales/biases ship F16 on disk (`T_scale = half`).
    let scales: Vec<half::f16> = (0..n_groups)
        .map(|_| half::f16::from_f32(0.01 + 0.04 * rng.next_unit_f32()))
        .collect();
    let biases: Vec<half::f16> = (0..n_groups)
        .map(|_| half::f16::from_f32(rng.next_unit_f32() - 0.5))
        .collect();
    let x: Vec<half::bf16> = (0..(m * k))
        .map(|_| half::bf16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
        .collect();

    (packed, scales, biases, x)
}

// CPU reference now lives in `scratchy_target_metal::cpu_reference`
// (imported above as `cpu_qmm_n_bf16`). Shares the transpose=false
// matmul with qvm.

#[allow(clippy::too_many_arguments)]
fn run_qmm_n_bf16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    x: &[half::bf16],
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
) -> Option<Vec<half::bf16>> {
    let device = detect_device()?.device;

    let packed_buf = buffer_from_bytes(&device, packed);
    let scales_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(scales.as_ptr() as *const u8, std::mem::size_of_val(scales))
    };
    let biases_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(biases.as_ptr() as *const u8, std::mem::size_of_val(biases))
    };
    let x_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, std::mem::size_of_val(x)) };
    let scales_buf = buffer_from_bytes(&device, scales_bytes);
    let biases_buf = buffer_from_bytes(&device, biases_bytes);
    let x_buf = buffer_from_bytes(&device, x_bytes);
    let y_buf = zeroed_buffer(&device, m * n * std::mem::size_of::<half::bf16>());

    let pso = qmm_n_pipeline(
        &device,
        DequantDtype::Bf16,
        ScaleDtype::F16,
        m as u32,
        n as u32,
        k as u32,
        group_size,
    );
    // Buffer-index order matches `MetalAffineQmmN::execute`:
    // packed_w(0), scales(1), biases(2), x(3), y(4).
    let (threadgroups, threads) = qmm_n_grid(m as u32, n as u32);
    if !common::dispatch_threadgroups(
        &device,
        &pso,
        &[&packed_buf, &scales_buf, &biases_buf, &x_buf, &y_buf],
        threadgroups,
        threads,
    ) {
        return None;
    }

    Some(read_buffer_bf16(&y_buf, m * n))
}

/// Per-element bf16 accumulation noise bound, parameterized on K and
/// the per-element product magnitude. Same shape as the qmv/qmm test.
fn worst_abs_error_vs_noise_floor(
    metal: &[half::bf16],
    expected: &[half::bf16],
    k_dim: usize,
    per_elem_magnitude: f32,
) -> (usize, f32, f32, f32, f32) {
    let bf16_eps: f32 = 1.0 / 128.0;
    let sum_noise_std = (k_dim as f32).sqrt() * 0.5 * bf16_eps * per_elem_magnitude;
    let safety = 4.0;
    let mut worst = (0usize, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    for (i, (m, e)) in metal.iter().zip(expected.iter()).enumerate() {
        let mf = m.to_f32();
        let ef = e.to_f32();
        let abs_err = (mf - ef).abs();
        let allowed = safety * (sum_noise_std + ef.abs() * bf16_eps);
        if abs_err > worst.3 {
            worst = (i, mf, ef, abs_err, allowed);
        }
    }
    worst
}

// ─────────────────────────────────────────────────────────────────
// qmm_n parity — aligned tile (M_tile = BM, N_tile = BN)
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_qmm_n_b4_bf16_aligned_matches_cpu_reference() {
    // M = 64 (≥ vector_limit=4 for transpose=false matmul branch);
    // N = 128 (% 128 == 0, satisfies N % gs == 0 for every gs in
    // {32, 64, 128}); K = 256 (% 32 == 0, divisible by every gs).
    let m = 64;
    let n = 128;
    let k = 256;
    for &group_size in &[32usize, 64, 128] {
        assert_eq!(n % group_size, 0);
        let (packed, scales, biases, x) =
            make_inputs_bf16(0xC1A1_u64 ^ group_size as u64, n, k, m, group_size);
        let expected = cpu_qmm_n_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
        let Some(metal) = run_qmm_n_bf16(&packed, &scales, &biases, &x, m, n, k, group_size as u32)
        else {
            return;
        };
        let (idx, mv, ev, abs_err, allowed) =
            worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
        assert!(
            abs_err <= allowed,
            "qmm_n aligned gs={group_size}: worst abs_err={abs_err:.5} at idx {idx} \
             (allowed {allowed:.5}; metal={mv}, cpu={ev})"
        );
    }
}

// ─────────────────────────────────────────────────────────────────
// qmm_n parity — M-tail path (m_tile < BM forces load_safe)
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_qmm_n_b4_bf16_m_tail_matches_cpu_reference() {
    // M = 50 → ceil(50/32) = 2 M-tiles, second tile m_tile = 18
    // (< BM=32). N = 64. K = 128. Exercises the load_safe + tail
    // store path for the m_full=false branch in qmm_n_impl_inline.
    let m = 50;
    let n = 64;
    let k = 128;
    let group_size = 64;

    let (packed, scales, biases, x) =
        make_inputs_bf16(0xC2A2_u64 ^ group_size as u64, n, k, m, group_size);
    let expected = cpu_qmm_n_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
    let Some(metal) = run_qmm_n_bf16(&packed, &scales, &biases, &x, m, n, k, group_size as u32)
    else {
        return;
    };
    let (idx, mv, ev, abs_err, allowed) = worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
    assert!(
        abs_err <= allowed,
        "qmm_n m-tail gs={group_size}: worst abs_err={abs_err:.5} at idx {idx} \
         (allowed {allowed:.5}; metal={mv}, cpu={ev})"
    );
}

// ─────────────────────────────────────────────────────────────────
// qmm_n parity — K-tail path (K % BK != 0 forces tail load)
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_qmm_n_b4_bf16_k_tail_matches_cpu_reference() {
    // K = 96 → 3 BK iters (3 * 32 = 96, k_tail = 0)... not enough.
    // For k_tail>0 we need K % 32 != 0. But K must also be a
    // multiple of group_size. With gs=32, K=64 → no tail. With
    // gs=32 and a K like 96, K_blocks=3 + k_tail=0. We can't get
    // K_tail>0 while keeping K % gs == 0 (since gs ≥ BK=32 and gs
    // divides K). MLX's qmm_n only handles K_tail in this case
    // when K is unaligned to BK — which the dispatcher prevents
    // by requiring K % gs == 0 for affine quantization. So the
    // K-tail branch is dead code in production. Smoke-test the
    // aligned path on a K that exercises the BK-iter loop more
    // heavily (K = 8 * 32 = 256 with gs=32 → 8 K-iters per BK).
    let m = 32;
    let n = 64;
    let k = 256;
    let group_size = 32;

    let (packed, scales, biases, x) =
        make_inputs_bf16(0xC3A3_u64 ^ group_size as u64, n, k, m, group_size);
    let expected = cpu_qmm_n_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
    let Some(metal) = run_qmm_n_bf16(&packed, &scales, &biases, &x, m, n, k, group_size as u32)
    else {
        return;
    };
    let (idx, mv, ev, abs_err, allowed) = worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
    assert!(
        abs_err <= allowed,
        "qmm_n k-heavy gs={group_size}: worst abs_err={abs_err:.5} at idx {idx} \
         (allowed {allowed:.5}; metal={mv}, cpu={ev})"
    );
}

// ─────────────────────────────────────────────────────────────────
// qmm_n parity — f16 dtype + larger gs
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_qmm_n_b4_f16_gs_128_matches_cpu_reference() {
    // Larger gs=128: each scale/bias covers 128 N-cols → fewer
    // scale loads per K-row. Exercises the same kernel as the
    // bf16 tests but through the f16 instantiation.
    let m = 64;
    let n = 128;
    let k = 256;
    let group_size: u32 = 128;

    let Some(__dev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = __dev.device;

    let n_bytes = k * n / 2;
    let n_groups = k * n / group_size as usize;
    let mut rng = SplitMix64(0xC4A4_u64);
    let packed: Vec<u8> = (0..n_bytes).map(|_| rng.next_byte()).collect();
    let scales: Vec<half::f16> = (0..n_groups)
        .map(|_| half::f16::from_f32(0.01 + 0.04 * rng.next_unit_f32()))
        .collect();
    let biases: Vec<half::f16> = (0..n_groups)
        .map(|_| half::f16::from_f32(rng.next_unit_f32() - 0.5))
        .collect();
    let x: Vec<half::f16> = (0..(m * k))
        .map(|_| half::f16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
        .collect();

    // Dequantize CPU side
    let mut w = vec![half::f16::ZERO; k * n];
    for kk in 0..k {
        for byte_j in 0..(n / 2) {
            let byte = packed[kk * (n / 2) + byte_j];
            let n_col = 2 * byte_j;
            let group_idx = kk * (n / group_size as usize) + n_col / group_size as usize;
            let scale = scales[group_idx].to_f32();
            let bias = biases[group_idx].to_f32();
            let lo = (byte & 0x0f) as f32;
            let hi = ((byte >> 4) & 0x0f) as f32;
            w[kk * n + n_col] = half::f16::from_f32(scale * lo + bias);
            w[kk * n + n_col + 1] = half::f16::from_f32(scale * hi + bias);
        }
    }
    let mut expected = vec![half::f16::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc: f32 = 0.0;
            for kk in 0..k {
                acc += x[i * k + kk].to_f32() * w[kk * n + j].to_f32();
            }
            expected[i * n + j] = half::f16::from_f32(acc);
        }
    }

    let packed_buf = buffer_from_bytes(&device, &packed);
    let scales_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            scales.as_ptr() as *const u8,
            std::mem::size_of_val(&scales[..]),
        )
    };
    let biases_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            biases.as_ptr() as *const u8,
            std::mem::size_of_val(&biases[..]),
        )
    };
    let x_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(x.as_ptr() as *const u8, std::mem::size_of_val(&x[..]))
    };
    let scales_buf = buffer_from_bytes(&device, scales_bytes);
    let biases_buf = buffer_from_bytes(&device, biases_bytes);
    let x_buf = buffer_from_bytes(&device, x_bytes);
    let y_buf = zeroed_buffer(&device, m * n * std::mem::size_of::<half::f16>());

    let pso = qmm_n_pipeline(
        &device,
        DequantDtype::F16,
        ScaleDtype::F16,
        m as u32,
        n as u32,
        k as u32,
        group_size,
    );
    // packed_w(0), scales(1), biases(2), x(3), y(4) — same order as
    // `MetalAffineQmmN::execute`.
    let (threadgroups, threads) = qmm_n_grid(m as u32, n as u32);
    if !common::dispatch_threadgroups(
        &device,
        &pso,
        &[&packed_buf, &scales_buf, &biases_buf, &x_buf, &y_buf],
        threadgroups,
        threads,
    ) {
        return;
    }

    // f16 has ~1024 ULP step at 1.0; bf16 has ~128 ULP step.
    let ptr = y_buf.contents().as_ptr() as *const half::f16;
    let metal: Vec<half::f16> = unsafe { std::slice::from_raw_parts(ptr, m * n) }.to_vec();
    let f16_eps: f32 = 1.0 / 1024.0;
    let sum_noise_std = (k as f32).sqrt() * 0.5 * f16_eps * 0.5;
    let safety = 4.0;
    let mut worst_err = 0.0_f32;
    let mut worst_idx = 0usize;
    let mut worst_allowed = 0.0_f32;
    for (i, (mm, ee)) in metal.iter().zip(expected.iter()).enumerate() {
        let mf = mm.to_f32();
        let ef = ee.to_f32();
        let abs_err = (mf - ef).abs();
        let allowed = safety * (sum_noise_std + ef.abs() * f16_eps);
        if abs_err > worst_err {
            worst_err = abs_err;
            worst_idx = i;
            worst_allowed = allowed;
        }
    }
    assert!(
        worst_err <= worst_allowed,
        "qmm_n f16 gs=128: worst abs_err={worst_err:.5} at idx {worst_idx} \
         (allowed {worst_allowed:.5})"
    );
}

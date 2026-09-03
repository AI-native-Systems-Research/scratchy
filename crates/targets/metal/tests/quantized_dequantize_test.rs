// SPDX-License-Identifier: Apache-2.0
//
// `affine_dequantize_*_gs_*_b_4` parity test: kernel ≡ CPU reference.
//
// Mirrors the slow-reference math in `mlx/backend/metal/kernels/quantized.h:2536`.
// The CPU reference is inlined here (rather than pulled from
// `scratchy_forward_compiler::cpu_golden`) so the test file stays free of a cycle
// against scratchy-target-metal' own dev-deps.

mod common;

use objc2_metal::{MTLComputePipelineState, MTLSize};
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::quantized::{DequantDtype, ScaleDtype};
use scratchy_target_metal::shader_cache::ShaderCache;

/// CPU reference: `out[oindex] = scale * nibble + bias` with FMA-style
/// single-rounding semantics (f32 multiply + add, then one round to the
/// target dtype). The GPU emits an fp16/bf16 hardware FMA for this
/// expression — modelling each op individually in half precision
/// double-rounds and drifts ~4 ULPs vs the kernel.
fn cpu_affine_dequantize_b4_f16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    group_size: usize,
) -> Vec<half::f16> {
    let out_len = packed.len() * 2;
    let mut out = vec![half::f16::ZERO; out_len];
    for (offset, &byte) in packed.iter().enumerate() {
        let oindex = offset * 2;
        let gindex = oindex / group_size;
        let scale = scales[gindex].to_f32();
        let bias = biases[gindex].to_f32();
        let lo = (byte & 0x0f) as f32;
        let hi = ((byte >> 4) & 0x0f) as f32;
        out[oindex] = half::f16::from_f32(scale * lo + bias);
        out[oindex + 1] = half::f16::from_f32(scale * hi + bias);
    }
    out
}

fn cpu_affine_dequantize_b4_bf16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    group_size: usize,
) -> Vec<half::bf16> {
    let out_len = packed.len() * 2;
    let mut out = vec![half::bf16::ZERO; out_len];
    for (offset, &byte) in packed.iter().enumerate() {
        let oindex = offset * 2;
        let gindex = oindex / group_size;
        // P10b: kernel casts T_scale (f16) → T_act (bf16) in-register
        // before the dequant math. Match that exactly in the CPU ref so
        // we can compare bit-for-bit (within ~2 ULP for bf16 FMA).
        let scale = half::bf16::from_f32(scales[gindex].to_f32()).to_f32();
        let bias = half::bf16::from_f32(biases[gindex].to_f32()).to_f32();
        let lo = (byte & 0x0f) as f32;
        let hi = ((byte >> 4) & 0x0f) as f32;
        out[oindex] = half::bf16::from_f32(scale * lo + bias);
        out[oindex + 1] = half::bf16::from_f32(scale * hi + bias);
    }
    out
}

/// Deterministic PRNG seeded per-shape so failures reproduce.
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
        // Uniform [0, 1). 24 bits of mantissa precision.
        ((self.next() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

#[test]
fn affine_dequantize_b4_f16_kernel_matches_cpu_reference() {
    for &group_size in &[32usize, 64, 128] {
        let n: usize = 4;
        let k: usize = 256; // K must be multiple of group_size; 256 covers all three gs values.
        let n_bytes = n * k / 2; // 2 nibbles per byte
        let n_groups = n * k / group_size;

        let mut rng = SplitMix64(0xC0FFEE_u64 ^ group_size as u64);
        let packed: Vec<u8> = (0..n_bytes).map(|_| rng.next_byte()).collect();
        // Scales in [0.1, 1.0] to keep activations finite; biases in [-1, 1).
        let scales: Vec<half::f16> = (0..n_groups)
            .map(|_| half::f16::from_f32(0.1 + 0.9 * rng.next_unit_f32()))
            .collect();
        let biases: Vec<half::f16> = (0..n_groups)
            .map(|_| half::f16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
            .collect();

        let expected = cpu_affine_dequantize_b4_f16(&packed, &scales, &biases, group_size);
        let Some(metal) = run_kernel_f16(&packed, &scales, &biases, n, k, group_size as u32) else {
            return;
        };

        for (i, (m, e)) in metal.iter().zip(expected.iter()).enumerate() {
            // CPU and GPU both model FMA single-rounding (f32 intermediate
            // → one round to half). Expect bit-exact match.
            let mb = m.to_bits() as i32;
            let eb = e.to_bits() as i32;
            assert!(
                (mb - eb).abs() <= 1,
                "f16 gs={group_size} idx={i}: metal={m} (bits={mb:04x}) cpu={e} (bits={eb:04x})"
            );
        }
    }
}

#[test]
fn affine_dequantize_b4_bf16_kernel_matches_cpu_reference() {
    for &group_size in &[32usize, 64, 128] {
        let n: usize = 4;
        let k: usize = 256;
        let n_bytes = n * k / 2;
        let n_groups = n * k / group_size;

        let mut rng = SplitMix64(0xDEADBEEF_u64 ^ group_size as u64);
        let packed: Vec<u8> = (0..n_bytes).map(|_| rng.next_byte()).collect();
        // P10b: scales/biases ship F16 on disk; the kernel reads them
        // as `T_scale = half` and casts to `T_act` in-register.
        let scales: Vec<half::f16> = (0..n_groups)
            .map(|_| half::f16::from_f32(0.1 + 0.9 * rng.next_unit_f32()))
            .collect();
        let biases: Vec<half::f16> = (0..n_groups)
            .map(|_| half::f16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
            .collect();

        let expected = cpu_affine_dequantize_b4_bf16(&packed, &scales, &biases, group_size);
        let Some(metal) = run_kernel_bf16(&packed, &scales, &biases, n, k, group_size as u32)
        else {
            return;
        };

        for (i, (m, e)) in metal.iter().zip(expected.iter()).enumerate() {
            let mb = m.to_bits() as i32;
            let eb = e.to_bits() as i32;
            assert!(
                (mb - eb).abs() <= 2,
                "bf16 gs={group_size} idx={i}: metal={m} (bits={mb:04x}) cpu={e} (bits={eb:04x})"
            );
        }
    }
}

/// Affine-dequant kernel binding contract (mirrors
/// `MetalAffineDequantize::execute`): `buffer(0)=packed_weight,
/// buffer(1)=scales, buffer(2)=biases, buffer(3)=output`. group_size,
/// dtype, scale_dtype, and bits are baked into the kernel name (function
/// specialization) — there are no `setBytes` scalars. The grid is a 1D
/// dispatch of `nthreads = out_n_elements / 2` threads (2 nibbles/byte),
/// `threads_per_threadgroup = min(nthreads, maxTotalThreadsPerThreadgroup)`.
///
/// Returns `None` if the host has no MTL4 queue (caller early-returns).
fn run_affine_dequant(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    n: usize,
    k: usize,
    group_size: u32,
    dtype: DequantDtype,
) -> Option<common::Buffer> {
    let device = detect_device()?.device;
    let cache = ShaderCache::new(device.clone()).expect("ShaderCache");

    let packed_buf = common::shared_slice(&device, packed);
    let scales_buf = common::shared_slice(&device, scales);
    let biases_buf = common::shared_slice(&device, biases);
    let n_out = n * k;
    let out_buf = common::shared_zeroed(&device, n_out * std::mem::size_of::<half::f16>());

    let kernel_name = format!(
        "affine_dequantize_{}_s_{}_gs_{}_b_{}",
        dtype.symbol_infix(),
        ScaleDtype::F16.symbol_infix(),
        group_size,
        4,
    );
    let pipeline = cache.get_pipeline(&kernel_name).expect("dequant pipeline");

    // nthreads = out_n_elements / packs_per_int (packs_per_int = 2 for bits=4).
    let nthreads = n_out / 2;
    let tpt = nthreads.min(pipeline.maxTotalThreadsPerThreadgroup());
    let threads_per_threadgroup = MTLSize {
        width: tpt,
        height: 1,
        depth: 1,
    };
    let threadgroups = MTLSize {
        width: nthreads.div_ceil(tpt),
        height: 1,
        depth: 1,
    };

    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[&packed_buf, &scales_buf, &biases_buf, &out_buf],
        threadgroups,
        threads_per_threadgroup,
    ) {
        return None;
    }
    Some(out_buf)
}

fn run_kernel_f16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    n: usize,
    k: usize,
    group_size: u32,
) -> Option<Vec<half::f16>> {
    let out_buf = run_affine_dequant(packed, scales, biases, n, k, group_size, DequantDtype::F16)?;
    Some(common::read_slice(&out_buf, n * k))
}

fn run_kernel_bf16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    n: usize,
    k: usize,
    group_size: u32,
) -> Option<Vec<half::bf16>> {
    let out_buf = run_affine_dequant(packed, scales, biases, n, k, group_size, DequantDtype::Bf16)?;
    Some(common::read_slice(&out_buf, n * k))
}

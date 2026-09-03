// SPDX-License-Identifier: Apache-2.0
//
// `affine_embed_*_gs_*_b_4` parity test: kernel ≡ CPU reference.
//
// Mirrors the slow-reference math in MLX
// `nn.QuantizedEmbedding.__call__` (`python/mlx/nn/layers/quantized.py:144`):
//   y[token, col] = scale[vocab_idx, col/gs] * nibble(w[vocab_idx, col/2])
//                 + bias[vocab_idx, col/gs]
// where `vocab_idx = indices[token]` and `nibble` extracts the low / high
// 4 bits depending on parity of `col`. The CPU reference uses FMA-style
// single-rounding (f32 multiply + add, then one round to the target
// dtype), matching the GPU FMA — same convention as
// `quantized_dequantize_test.rs`.

mod common;

use objc2_metal::{MTLComputePipelineState, MTLSize};
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::quantized::{DequantDtype, ScaleDtype};
use scratchy_target_metal::shader_cache::ShaderCache;
use scratchy_target_metal::specialized_pipeline_cache::ConstantValue;

/// CPU reference: gather-then-dequant. Each output element is
/// `scale[vocab_idx, group] * nibble + bias[vocab_idx, group]` with
/// FMA single-rounding (the GPU emits one hardware FMA per element).
fn cpu_affine_embed_b4_f16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    indices: &[u32],
    hidden_size: usize,
    group_size: usize,
) -> Vec<half::f16> {
    let bytes_per_row = hidden_size / 2;
    let groups_per_row = hidden_size / group_size;
    let n_tokens = indices.len();
    let mut out = vec![half::f16::ZERO; n_tokens * hidden_size];
    for (token, &vocab_idx) in indices.iter().enumerate() {
        let vocab_idx = vocab_idx as usize;
        for byte_idx in 0..bytes_per_row {
            let byte = packed[vocab_idx * bytes_per_row + byte_idx];
            let col_base = byte_idx * 2;
            let g = vocab_idx * groups_per_row + col_base / group_size;
            let scale = scales[g].to_f32();
            let bias = biases[g].to_f32();
            let lo = (byte & 0x0f) as f32;
            let hi = ((byte >> 4) & 0x0f) as f32;
            out[token * hidden_size + col_base] = half::f16::from_f32(scale * lo + bias);
            out[token * hidden_size + col_base + 1] = half::f16::from_f32(scale * hi + bias);
        }
    }
    out
}

fn cpu_affine_embed_b4_bf16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    indices: &[u32],
    hidden_size: usize,
    group_size: usize,
) -> Vec<half::bf16> {
    let bytes_per_row = hidden_size / 2;
    let groups_per_row = hidden_size / group_size;
    let n_tokens = indices.len();
    let mut out = vec![half::bf16::ZERO; n_tokens * hidden_size];
    for (token, &vocab_idx) in indices.iter().enumerate() {
        let vocab_idx = vocab_idx as usize;
        for byte_idx in 0..bytes_per_row {
            let byte = packed[vocab_idx * bytes_per_row + byte_idx];
            let col_base = byte_idx * 2;
            let g = vocab_idx * groups_per_row + col_base / group_size;
            // Match the kernel's in-register T_scale (f16) → T_act (bf16)
            // cast so the CPU ref matches within ~2 ULP of the GPU FMA.
            let scale = half::bf16::from_f32(scales[g].to_f32()).to_f32();
            let bias = half::bf16::from_f32(biases[g].to_f32()).to_f32();
            let lo = (byte & 0x0f) as f32;
            let hi = ((byte >> 4) & 0x0f) as f32;
            out[token * hidden_size + col_base] = half::bf16::from_f32(scale * lo + bias);
            out[token * hidden_size + col_base + 1] = half::bf16::from_f32(scale * hi + bias);
        }
    }
    out
}

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
    fn next_u32_below(&mut self, n: u32) -> u32 {
        ((self.next() >> 32) as u32) % n
    }
}

#[test]
fn affine_embed_b4_f16_kernel_matches_cpu_reference() {
    // Vocab=32 keeps the test fast; hidden_size=256 covers all three gs
    // values; n_tokens=16 exercises multi-row dispatch + 2D grid wraps.
    let vocab_size: usize = 32;
    let hidden_size: usize = 256;
    let n_tokens: usize = 16;

    for &group_size in &[32usize, 64, 128] {
        let n_bytes = vocab_size * hidden_size / 2;
        let n_groups = vocab_size * hidden_size / group_size;

        let mut rng = SplitMix64(0xC0FFEE_u64 ^ group_size as u64);
        let packed: Vec<u8> = (0..n_bytes).map(|_| rng.next_byte()).collect();
        let scales: Vec<half::f16> = (0..n_groups)
            .map(|_| half::f16::from_f32(0.1 + 0.9 * rng.next_unit_f32()))
            .collect();
        let biases: Vec<half::f16> = (0..n_groups)
            .map(|_| half::f16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
            .collect();
        let indices: Vec<u32> = (0..n_tokens)
            .map(|_| rng.next_u32_below(vocab_size as u32))
            .collect();

        let expected =
            cpu_affine_embed_b4_f16(&packed, &scales, &biases, &indices, hidden_size, group_size);
        let metal = run_kernel_f16(
            &packed,
            &scales,
            &biases,
            &indices,
            hidden_size as u32,
            group_size as u32,
        );

        for (i, (m, e)) in metal.iter().zip(expected.iter()).enumerate() {
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
fn affine_embed_b4_bf16_kernel_matches_cpu_reference() {
    let vocab_size: usize = 32;
    let hidden_size: usize = 256;
    let n_tokens: usize = 16;

    for &group_size in &[32usize, 64, 128] {
        let n_bytes = vocab_size * hidden_size / 2;
        let n_groups = vocab_size * hidden_size / group_size;

        let mut rng = SplitMix64(0xDEADBEEF_u64 ^ group_size as u64);
        let packed: Vec<u8> = (0..n_bytes).map(|_| rng.next_byte()).collect();
        // P10b: scales/biases ship F16 on disk.
        let scales: Vec<half::f16> = (0..n_groups)
            .map(|_| half::f16::from_f32(0.1 + 0.9 * rng.next_unit_f32()))
            .collect();
        let biases: Vec<half::f16> = (0..n_groups)
            .map(|_| half::f16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
            .collect();
        let indices: Vec<u32> = (0..n_tokens)
            .map(|_| rng.next_u32_below(vocab_size as u32))
            .collect();

        let expected =
            cpu_affine_embed_b4_bf16(&packed, &scales, &biases, &indices, hidden_size, group_size);
        let metal = run_kernel_bf16(
            &packed,
            &scales,
            &biases,
            &indices,
            hidden_size as u32,
            group_size as u32,
        );

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

/// Exercises the partial-trailing-threadgroup bounds-check: hidden=2112
/// = 33 groups × 64 → bytes_per_row=1056 > 1024 (typical max
/// threads-per-threadgroup), forces 2 threadgroups in the X dim where
/// the second has only 32 active threads. Without the kernel's
/// `if (index.x * 2 >= hidden_size) return;` guard the trailing 992
/// threads chase past-end bytes of `w` and corrupt out[].
#[test]
fn affine_embed_b4_bf16_partial_trailing_threadgroup() {
    let vocab_size: usize = 8;
    let hidden_size: usize = 2112;
    let n_tokens: usize = 4;
    let group_size: usize = 64;

    let n_bytes = vocab_size * hidden_size / 2;
    let n_groups = vocab_size * hidden_size / group_size;

    let mut rng = SplitMix64(0xBADC0FFE);
    let packed: Vec<u8> = (0..n_bytes).map(|_| rng.next_byte()).collect();
    // P10b: scales/biases ship F16 on disk.
    let scales: Vec<half::f16> = (0..n_groups)
        .map(|_| half::f16::from_f32(0.1 + 0.9 * rng.next_unit_f32()))
        .collect();
    let biases: Vec<half::f16> = (0..n_groups)
        .map(|_| half::f16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
        .collect();
    let indices: Vec<u32> = (0..n_tokens)
        .map(|_| rng.next_u32_below(vocab_size as u32))
        .collect();

    let expected =
        cpu_affine_embed_b4_bf16(&packed, &scales, &biases, &indices, hidden_size, group_size);
    let metal = run_kernel_bf16(
        &packed,
        &scales,
        &biases,
        &indices,
        hidden_size as u32,
        group_size as u32,
    );

    for (i, (m, e)) in metal.iter().zip(expected.iter()).enumerate() {
        let mb = m.to_bits() as i32;
        let eb = e.to_bits() as i32;
        assert!(
            (mb - eb).abs() <= 2,
            "bf16 unaligned hidden={hidden_size} idx={i}: \
             metal={m} (bits={mb:04x}) cpu={e} (bits={eb:04x})"
        );
    }
}

/// Build the specialized `affine_embed_<dtype>_s_f16_gs_<gs>_b_4`
/// pipeline (`hidden_size` rides as `function_constant(0)`) and dispatch
/// it on the production MTL4 path with the exact same binding contract /
/// 2D grid as `MetalAffineEmbed::execute`: buffers
/// `0=packed, 1=scales, 2=biases, 3=indices, 4=output`; threadgroups
/// `(ceil(bytes_per_row / tpg_x), num_tokens, 1)` with `tpg_x =
/// min(bytes_per_row, pipeline.maxTotalThreadsPerThreadgroup())`. Returns
/// the result buffer, or `None` if the host has no MTL4 queue.
fn dispatch_affine_embed(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    indices: &[u32],
    hidden_size: u32,
    group_size: u32,
    dtype: DequantDtype,
) -> Option<common::Buffer> {
    let device = detect_device()?.device;

    let packed_buf = common::shared_slice(&device, packed);
    let scales_buf = common::shared_slice(&device, scales);
    let biases_buf = common::shared_slice(&device, biases);
    let indices_buf = common::shared_slice(&device, indices);
    let n_out = indices.len() * hidden_size as usize;
    let elem_size = dtype.elem_size();
    let out_buf = common::shared_zeroed(&device, n_out * elem_size);

    let cache = ShaderCache::new(device.clone()).expect("ShaderCache");
    let kernel_name = format!(
        "affine_embed_{}_s_{}_gs_{}_b_4",
        dtype.symbol_infix(),
        ScaleDtype::F16.symbol_infix(),
        group_size
    );
    // `hidden_size` rides as function_constant(0) — see
    // `AFFINE_EMBED_HIDDEN_SIZE` in `quantized_dequantize.metal`.
    let constants = [ConstantValue::uint(0u16, hidden_size)];
    let pipeline = cache
        .get_pipeline_specialized(&kernel_name, &constants)
        .expect("affine_embed pipeline");

    // Replicates `MetalAffineEmbed::execute`'s 2D grid exactly.
    let packs_per_int: u32 = 2;
    let bytes_per_row = hidden_size / packs_per_int;
    let max_tpg = pipeline
        .maxTotalThreadsPerThreadgroup()
        .min(u32::MAX as usize) as u32;
    let tpg_x = bytes_per_row.min(max_tpg).max(1);
    let groups_x = bytes_per_row.div_ceil(tpg_x);
    let threads_per_threadgroup = MTLSize {
        width: tpg_x as usize,
        height: 1,
        depth: 1,
    };
    let threadgroups = MTLSize {
        width: groups_x as usize,
        height: indices.len(),
        depth: 1,
    };

    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[
            &packed_buf,
            &scales_buf,
            &biases_buf,
            &indices_buf,
            &out_buf,
        ],
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
    indices: &[u32],
    hidden_size: u32,
    group_size: u32,
) -> Vec<half::f16> {
    let n_out = indices.len() * hidden_size as usize;
    match dispatch_affine_embed(
        packed,
        scales,
        biases,
        indices,
        hidden_size,
        group_size,
        DequantDtype::F16,
    ) {
        Some(out_buf) => common::read_slice(&out_buf, n_out),
        None => Vec::new(),
    }
}

fn run_kernel_bf16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    indices: &[u32],
    hidden_size: u32,
    group_size: u32,
) -> Vec<half::bf16> {
    let n_out = indices.len() * hidden_size as usize;
    match dispatch_affine_embed(
        packed,
        scales,
        biases,
        indices,
        hidden_size,
        group_size,
        DequantDtype::Bf16,
    ) {
        Some(out_buf) => common::read_slice(&out_buf, n_out),
        None => Vec::new(),
    }
}

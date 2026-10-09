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
use scratchy_target_metal::aot::baked_pipeline;
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::quantized::{DequantDtype, ScaleDtype};
use scratchy_target_metal::specialized_pipeline_cache::ConstantValue;
use scratchy_target_metal::tape::kernel_constants::AffineCodes;

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
    (hidden_size, group_size): (u32, u32),
    dtype: DequantDtype,
    codes: AffineCodes,
) -> Option<common::Buffer> {
    let device = detect_device()?.device;

    let packed_buf = common::shared_slice(&device, packed);
    let scales_buf = common::shared_slice(&device, scales);
    let biases_buf = common::shared_slice(&device, biases);
    let indices_buf = common::shared_slice(&device, indices);
    let n_out = indices.len() * hidden_size as usize;
    let elem_size = dtype.elem_size();
    let out_buf = common::shared_zeroed(&device, n_out * elem_size);

    let kernel_name = format!(
        "affine_embed_{}_s_{}_gs_{}_b_4",
        dtype.symbol_infix(),
        ScaleDtype::F16.symbol_infix(),
        group_size
    );
    // `hidden_size` is constant slot 0 — see `AFFINE_EMBED_HIDDEN_SIZE`
    // in `quantized_dequantize.metal`.
    let constants: Vec<ConstantValue> = std::iter::once(ConstantValue::uint(0u16, hidden_size))
        .chain(codes.constant())
        .collect();
    let pipeline = baked_pipeline(&device, "quantized_dequantize", &kernel_name, constants)
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
        (hidden_size, group_size),
        DequantDtype::F16,
        AffineCodes::AsWritten,
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
        (hidden_size, group_size),
        DequantDtype::Bf16,
        AffineCodes::AsWritten,
    ) {
        Some(out_buf) => common::read_slice(&out_buf, n_out),
        None => Vec::new(),
    }
}

/// Codes stored offset-8 (XOR 0x88, as an M5 target stores 4-bit codes)
/// read under `AFFINE_CODES_OFFSET8` gather bit-identical rows to the codes
/// as written.
#[test]
fn affine_embed_b4_offset8_codes_match_as_written() {
    let (vocab_size, hidden_size, group_size) = (32usize, 256u32, 64u32);
    let mut rng = SplitMix64(0x0FF5E7);
    let packed: Vec<u8> = (0..vocab_size * hidden_size as usize / 2)
        .map(|_| rng.next_byte())
        .collect();
    let groups = vocab_size * (hidden_size / group_size) as usize;
    let scales: Vec<half::f16> = (0..groups)
        .map(|_| half::f16::from_f32(0.1 + 0.9 * rng.next_unit_f32()))
        .collect();
    let biases: Vec<half::f16> = (0..groups)
        .map(|_| half::f16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
        .collect();
    let indices: Vec<u32> = (0..16)
        .map(|_| rng.next_u32_below(vocab_size as u32))
        .collect();
    let run = |packed: &[u8], codes| {
        dispatch_affine_embed(
            packed,
            &scales,
            &biases,
            &indices,
            (hidden_size, group_size),
            DequantDtype::Bf16,
            codes,
        )
        .map(|b| common::read_slice::<half::bf16>(&b, indices.len() * hidden_size as usize))
    };
    let Some(as_written) = run(&packed, AffineCodes::AsWritten) else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let offset8: Vec<u8> = packed.iter().map(|b| b ^ 0x88).collect();
    assert_ne!(
        run(&offset8, AffineCodes::AsWritten).as_ref(),
        Some(&as_written)
    );
    assert_eq!(run(&offset8, AffineCodes::Offset8), Some(as_written));
}

// ─────────────────────────────────────────────────────────────────
// 2-bit embed (`affine_embed_*_gs_64_b_2`) — gpt-oss-120b-mlx-2Bit's
// quantized embed. Same gather-then-dequant shape as b4, but the
// bitstream packs 4 codes per byte (element i at bits [2i, 2i+2)), so
// one thread reads one byte and writes 4 output elements. No XOR path
// (Offset8 is 4-bit-only) — codes read as written. The production combo
// is bf16 activations × bf16 scales (unlike b4's f16 scales); f16 × f16
// is the unit-test combo, both instantiated at gs=64 only.
// ─────────────────────────────────────────────────────────────────

fn cpu_affine_embed_b2_f16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    indices: &[u32],
    hidden_size: usize,
    group_size: usize,
) -> Vec<half::f16> {
    let bytes_per_row = hidden_size / 4;
    let groups_per_row = hidden_size / group_size;
    let n_tokens = indices.len();
    let mut out = vec![half::f16::ZERO; n_tokens * hidden_size];
    for (token, &vocab_idx) in indices.iter().enumerate() {
        let vocab_idx = vocab_idx as usize;
        for byte_idx in 0..bytes_per_row {
            let byte = packed[vocab_idx * bytes_per_row + byte_idx];
            let col_base = byte_idx * 4;
            let g = vocab_idx * groups_per_row + col_base / group_size;
            let scale = scales[g].to_f32();
            let bias = biases[g].to_f32();
            for lane in 0..4 {
                let code = ((byte >> (2 * lane)) & 0x03) as f32;
                out[token * hidden_size + col_base + lane] =
                    half::f16::from_f32(scale * code + bias);
            }
        }
    }
    out
}

fn cpu_affine_embed_b2_bf16(
    packed: &[u8],
    scales: &[half::bf16],
    biases: &[half::bf16],
    indices: &[u32],
    hidden_size: usize,
    group_size: usize,
) -> Vec<half::bf16> {
    let bytes_per_row = hidden_size / 4;
    let groups_per_row = hidden_size / group_size;
    let n_tokens = indices.len();
    let mut out = vec![half::bf16::ZERO; n_tokens * hidden_size];
    for (token, &vocab_idx) in indices.iter().enumerate() {
        let vocab_idx = vocab_idx as usize;
        for byte_idx in 0..bytes_per_row {
            let byte = packed[vocab_idx * bytes_per_row + byte_idx];
            let col_base = byte_idx * 4;
            let g = vocab_idx * groups_per_row + col_base / group_size;
            // Match the kernel's in-register T_scale → T_act cast.
            let scale = half::bf16::from_f32(scales[g].to_f32()).to_f32();
            let bias = half::bf16::from_f32(biases[g].to_f32()).to_f32();
            for lane in 0..4 {
                let code = ((byte >> (2 * lane)) & 0x03) as f32;
                out[token * hidden_size + col_base + lane] =
                    half::bf16::from_f32(scale * code + bias);
            }
        }
    }
    out
}

/// Dispatch `affine_embed_<dtype>_s_<scale_dtype>_gs_<gs>_b_2` on the
/// production MTL4 path — same binding contract as
/// [`dispatch_affine_embed`], but 4 codes per byte (one thread per
/// packed byte) and a parametric scale dtype (b2 ships bf16 scales in
/// production). Returns the output buffer.
fn dispatch_affine_embed_b2<TScale: Copy>(
    packed: &[u8],
    scales: &[TScale],
    biases: &[TScale],
    indices: &[u32],
    (hidden_size, group_size): (u32, u32),
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
) -> Option<common::Buffer> {
    let device = detect_device()?.device;

    let packed_buf = common::shared_slice(&device, packed);
    let scales_buf = common::shared_slice(&device, scales);
    let biases_buf = common::shared_slice(&device, biases);
    let indices_buf = common::shared_slice(&device, indices);
    let n_out = indices.len() * hidden_size as usize;
    let out_buf = common::shared_zeroed(&device, n_out * dtype.elem_size());

    let kernel_name = format!(
        "affine_embed_{}_s_{}_gs_{}_b_2",
        dtype.symbol_infix(),
        scale_dtype.symbol_infix(),
        group_size
    );
    // `hidden_size` is constant slot 0 — see `AFFINE_EMBED_HIDDEN_SIZE`
    // in `quantized_dequantize.metal`.
    let constants: Vec<ConstantValue> =
        std::iter::once(ConstantValue::uint(0u16, hidden_size)).collect();
    let pipeline = baked_pipeline(&device, "quantized_dequantize", &kernel_name, constants)
        .expect("affine_embed b2 pipeline");

    // Replicates `MetalAffineEmbed::execute`'s 2D grid at bits=2: one
    // thread per packed byte (4 codes each).
    let bytes_per_row = hidden_size / 4;
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

#[test]
fn affine_embed_b2_f16_kernel_matches_cpu_reference() {
    let vocab_size: usize = 32;
    let hidden_size: usize = 256;
    let n_tokens: usize = 16;
    let group_size: usize = 64;

    let n_bytes = vocab_size * hidden_size / 4;
    let n_groups = vocab_size * hidden_size / group_size;

    let mut rng = SplitMix64(0x2B1D_5EED_u64);
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
        cpu_affine_embed_b2_f16(&packed, &scales, &biases, &indices, hidden_size, group_size);
    let Some(out_buf) = dispatch_affine_embed_b2(
        &packed,
        &scales,
        &biases,
        &indices,
        (hidden_size as u32, group_size as u32),
        DequantDtype::F16,
        ScaleDtype::F16,
    ) else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let metal = common::read_slice::<half::f16>(&out_buf, n_tokens * hidden_size);

    for (i, (m, e)) in metal.iter().zip(expected.iter()).enumerate() {
        let mb = m.to_bits() as i32;
        let eb = e.to_bits() as i32;
        assert!(
            (mb - eb).abs() <= 1,
            "b2 f16 gs={group_size} idx={i}: metal={m} (bits={mb:04x}) cpu={e} (bits={eb:04x})"
        );
    }
}

#[test]
fn affine_embed_b2_bf16_kernel_matches_cpu_reference() {
    // The gpt-oss production combo: bf16 activations × bf16 scales.
    let vocab_size: usize = 32;
    let hidden_size: usize = 256;
    let n_tokens: usize = 16;
    let group_size: usize = 64;

    let n_bytes = vocab_size * hidden_size / 4;
    let n_groups = vocab_size * hidden_size / group_size;

    let mut rng = SplitMix64(0x5EED_2B1D_u64);
    let packed: Vec<u8> = (0..n_bytes).map(|_| rng.next_byte()).collect();
    let scales: Vec<half::bf16> = (0..n_groups)
        .map(|_| half::bf16::from_f32(0.1 + 0.9 * rng.next_unit_f32()))
        .collect();
    let biases: Vec<half::bf16> = (0..n_groups)
        .map(|_| half::bf16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
        .collect();
    let indices: Vec<u32> = (0..n_tokens)
        .map(|_| rng.next_u32_below(vocab_size as u32))
        .collect();

    let expected =
        cpu_affine_embed_b2_bf16(&packed, &scales, &biases, &indices, hidden_size, group_size);
    let Some(out_buf) = dispatch_affine_embed_b2(
        &packed,
        &scales,
        &biases,
        &indices,
        (hidden_size as u32, group_size as u32),
        DequantDtype::Bf16,
        ScaleDtype::Bf16,
    ) else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let metal = common::read_slice::<half::bf16>(&out_buf, n_tokens * hidden_size);

    for (i, (m, e)) in metal.iter().zip(expected.iter()).enumerate() {
        let mb = m.to_bits() as i32;
        let eb = e.to_bits() as i32;
        assert!(
            (mb - eb).abs() <= 2,
            "b2 bf16 gs={group_size} idx={i}: metal={m} (bits={mb:04x}) cpu={e} (bits={eb:04x})"
        );
    }
}

/// The b2 partial-trailing-threadgroup bounds-check: hidden=4224 →
/// bytes_per_row = 1056 > 1024 (typical max threads-per-threadgroup),
/// forcing 2 X-dim threadgroups where the second has only 32 active
/// threads. Without the kernel's `if (index.x * 4 >= hidden_size)`
/// guard the trailing threads chase past-end bytes of `w`.
#[test]
fn affine_embed_b2_bf16_partial_trailing_threadgroup() {
    let vocab_size: usize = 8;
    let hidden_size: usize = 4224;
    let n_tokens: usize = 4;
    let group_size: usize = 64;

    let n_bytes = vocab_size * hidden_size / 4;
    let n_groups = vocab_size * hidden_size / group_size;

    let mut rng = SplitMix64(0xB2B2_C0DE);
    let packed: Vec<u8> = (0..n_bytes).map(|_| rng.next_byte()).collect();
    let scales: Vec<half::bf16> = (0..n_groups)
        .map(|_| half::bf16::from_f32(0.1 + 0.9 * rng.next_unit_f32()))
        .collect();
    let biases: Vec<half::bf16> = (0..n_groups)
        .map(|_| half::bf16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
        .collect();
    let indices: Vec<u32> = (0..n_tokens)
        .map(|_| rng.next_u32_below(vocab_size as u32))
        .collect();

    let expected =
        cpu_affine_embed_b2_bf16(&packed, &scales, &biases, &indices, hidden_size, group_size);
    let Some(out_buf) = dispatch_affine_embed_b2(
        &packed,
        &scales,
        &biases,
        &indices,
        (hidden_size as u32, group_size as u32),
        DequantDtype::Bf16,
        ScaleDtype::Bf16,
    ) else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let metal = common::read_slice::<half::bf16>(&out_buf, n_tokens * hidden_size);

    for (i, (m, e)) in metal.iter().zip(expected.iter()).enumerate() {
        let mb = m.to_bits() as i32;
        let eb = e.to_bits() as i32;
        assert!(
            (mb - eb).abs() <= 2,
            "b2 bf16 unaligned hidden={hidden_size} idx={i}: \
             metal={m} (bits={mb:04x}) cpu={e} (bits={eb:04x})"
        );
    }
}

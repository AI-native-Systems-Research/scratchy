// SPDX-License-Identifier: Apache-2.0
//! Pure MLX-affine 4-bit dequantization math (backend-neutral).
//!
//! `affine_dequant_b4_to_dtype` turns the three on-disk MLX-affine int4
//! safetensors entries — packed weight nibbles (`U32`, two per byte), per-group
//! `scales` and `biases` (`F16`/`BF16`, 2 bytes each) — into a dense `[N, K]`
//! `F16`/`BF16` weight, as raw little-endian bytes. The dequant rule is
//! `w[i] = nibble[i] * scale[group(i)] + bias[group(i)]`, computed through an
//! `f32` intermediate so the single FMA rounding matches the metal kernel's
//! hardware fp16/bf16 FMA (see `cpu_golden::affine_dequantize_b4_*` in
//! scratchy-forward-compiler).
//!
//! No tensor store, no device, no backend — callers fetch the bytes (via the
//! neutral `WeightSource`) and pass them in. The numerics are byte-for-byte the
//! arithmetic that previously lived in `GpuWeights::take_affine_dequant_b4_bytes`.

use anyhow::Result;
use scratchy_tensors::DType;

/// CPU-dequantize one MLX-affine int4 weight to dense `[N, K]` bytes.
///
/// * `weight_bytes` — packed nibbles, `n * k / 2` bytes (low nibble first).
/// * `scales_bytes` / `biases_bytes` — `n_groups` elements of `scale_dtype`
///   (`F16` or `BF16`), 2 bytes each, where `n_groups = n * k / group_size`.
/// * `n` / `k` — dequantized output dims (`k` already in elements, i.e.
///   `packed_cols * 8`).
/// * `group_size` — affine group width along `k`.
/// * `bits` — must be 4.
/// * `scale_dtype` — dtype of the scales/biases (`F16` or `BF16`).
/// * `dtype_out` — output dtype (`F16` or `BF16`).
///
/// Returns the dequantized `[N, K]` weight as `n * k * 2` little-endian bytes.
#[allow(clippy::too_many_arguments)]
pub fn affine_dequant_b4_to_dtype(
    weight_bytes: &[u8],
    scales_bytes: &[u8],
    biases_bytes: &[u8],
    n: usize,
    k: usize,
    group_size: u32,
    bits: u32,
    scale_dtype: DType,
    dtype_out: DType,
) -> Result<Vec<u8>> {
    anyhow::ensure!(
        bits == 4 || bits == 8,
        "affine_dequant_b4_to_dtype: only bits=4 or bits=8 is supported, got bits={bits}"
    );
    anyhow::ensure!(
        matches!(dtype_out, DType::F16 | DType::BF16),
        "affine_dequant_b4_to_dtype: dtype_out must be F16 or BF16, got {dtype_out}"
    );
    anyhow::ensure!(
        matches!(scale_dtype, DType::F16 | DType::BF16),
        "affine_dequant_b4_to_dtype: scale_dtype must be F16 or BF16, got {scale_dtype}"
    );
    anyhow::ensure!(
        k.is_multiple_of(group_size as usize),
        "affine_dequant_b4_to_dtype: K={k} not divisible by group_size={group_size}"
    );
    let n_groups = (n * k) / group_size as usize;

    // bits=4 packs two nibbles per byte (n*k/2 bytes); bits=8 stores one
    // element per byte (n*k bytes). The unpack loop below branches on this.
    let elems_per_byte = if bits == 4 { 2 } else { 1 };
    let n_packed_bytes = n * k / elems_per_byte;
    anyhow::ensure!(
        weight_bytes.len() == n_packed_bytes,
        "affine_dequant_b4_to_dtype: packed weight bytes {} != expected {}",
        weight_bytes.len(),
        n_packed_bytes,
    );
    anyhow::ensure!(
        scales_bytes.len() == n_groups * 2 && biases_bytes.len() == n_groups * 2,
        "affine_dequant_b4_to_dtype: scales/biases byte length mismatch \
         (scales={} biases={} expected={})",
        scales_bytes.len(),
        biases_bytes.len(),
        n_groups * 2,
    );
    let s_halves =
        unsafe { std::slice::from_raw_parts(scales_bytes.as_ptr() as *const u16, n_groups) };
    let b_halves =
        unsafe { std::slice::from_raw_parts(biases_bytes.as_ptr() as *const u16, n_groups) };

    // f32-intermediate FMA single-rounding matches the kernel's
    // hardware fp16/bf16 FMA — see the matching note on
    // `cpu_golden::affine_dequantize_b4_*` in scratchy-forward-compiler.
    let mut out_bytes = vec![0u8; n * k * 2];
    let gs = group_size as usize;
    let out_halves =
        unsafe { std::slice::from_raw_parts_mut(out_bytes.as_mut_ptr() as *mut u16, n * k) };
    let decode_half = |bits: u16| -> f32 {
        match scale_dtype {
            DType::F16 => half::f16::from_bits(bits).to_f32(),
            DType::BF16 => half::bf16::from_bits(bits).to_f32(),
            _ => unreachable!("scale_dtype guarded F16|BF16 above"),
        }
    };
    let store = |out_halves: &mut [u16], idx: usize, v: f32| match dtype_out {
        DType::F16 => out_halves[idx] = half::f16::from_f32(v).to_bits(),
        // MLX casts F16 → BF16 inline at kernel time (`T scale =
        // scales[gindex]` with `T = bfloat`); we do the same here at dequant
        // time. The f32 intermediate covers the cross-dtype expansion exactly
        // (F16 fits in f32 mantissa).
        DType::BF16 => out_halves[idx] = half::bf16::from_f32(v).to_bits(),
        _ => unreachable!("dtype_out guarded above"),
    };
    if bits == 4 {
        for (offset, &byte) in weight_bytes.iter().enumerate() {
            let oindex = offset * 2;
            let gindex = oindex / gs;
            let scale = decode_half(s_halves[gindex]);
            let bias = decode_half(b_halves[gindex]);
            let lo = (byte & 0x0f) as f32;
            let hi = ((byte >> 4) & 0x0f) as f32;
            store(out_halves, oindex, scale * lo + bias);
            store(out_halves, oindex + 1, scale * hi + bias);
        }
    } else {
        // bits == 8: one element per byte, value in 0..=255.
        for (oindex, &byte) in weight_bytes.iter().enumerate() {
            let gindex = oindex / gs;
            let scale = decode_half(s_halves[gindex]);
            let bias = decode_half(b_halves[gindex]);
            store(out_halves, oindex, scale * (byte as f32) + bias);
        }
    }

    Ok(out_bytes)
}

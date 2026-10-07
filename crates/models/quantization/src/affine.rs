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
/// * `bits` — 3, 4 or 8.
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
        bits == 3 || bits == 4 || bits == 8,
        "affine_dequant_b4_to_dtype: only bits=3, bits=4 or bits=8 is supported, got bits={bits}"
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
    // element per byte (n*k bytes); bits=3 packs a continuous LSB-first
    // bitstream (n*k*3/8 bytes — element i spans bits [3i, 3i+3) of the
    // little-endian byte array, exactly MLX's `qdot` bits==3 layout).
    // K is a multiple of group_size ∈ {32, 64, 128}, so the row's bitstream
    // never straddles a byte boundary mid-row: 3*K is a multiple of 8.
    let n_packed_bytes = match bits {
        4 => n * k / 2,
        8 => n * k,
        _ => n * k * 3 / 8,
    };
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
    } else if bits == 3 {
        // Continuous LSB-first bitstream: element i's 3 code bits live at
        // bit offset 3*i of the (little-endian) byte array. 8 elements span
        // bits [0, 24) — exactly 3 bytes — and since group_size (32/64/128)
        // is a multiple of 8, every 8-element run starts byte-aligned at
        // byte 3*(i/8). Mirrors MLX's `qdot` bits==3 shifts so a code that
        // straddles a byte boundary decodes identically to the metal
        // kernels.
        let total = n * k;
        let (packs, _rem) = weight_bytes.as_chunks::<3>();
        for (g, chunk) in packs.iter().enumerate() {
            let triple =
                (chunk[0] as u32) | ((chunk[1] as u32) << 8) | ((chunk[2] as u32) << 16);
            for j in 0..8 {
                let oindex = g * 8 + j;
                if oindex >= total {
                    break;
                }
                let code = ((triple >> (3 * j)) & 0x07) as f32;
                let gindex = oindex / gs;
                let scale = decode_half(s_halves[gindex]);
                let bias = decode_half(b_halves[gindex]);
                store(out_halves, oindex, scale * code + bias);
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack `codes` (each in 0..8) as MLX's continuous LSB-first 3-bit
    /// bitstream: code i occupies bits [3i, 3i+3) of the little-endian
    /// byte array. Independent of the dequant loop's byte-pair reads —
    /// written straight from the bit-offset definition so a wrong shift
    /// in either direction fails the round-trip.
    fn pack_b3(codes: &[u8]) -> Vec<u8> {
        let n_bytes = codes.len() * 3 / 8;
        let mut out = vec![0u8; n_bytes];
        for (i, &c) in codes.iter().enumerate() {
            let bit = 3 * i;
            let byte = bit / 8;
            let off = bit % 8;
            out[byte] |= c << off;
            if off + 3 > 8 {
                let spill = off + 3 - 8;
                out[byte + 1] |= c >> (3 - spill);
            }
        }
        out
    }

    /// 3-bit codes whose boundaries straddle bytes: element 2 starts at
    /// bit 6 (2 bits in byte 0, 1 in byte 1), element 8 at bit 24 (byte
    /// aligned). Codes chosen to exercise every shift residue of one
    /// byte-pair.
    #[test]
    fn affine_dequant_b3_bitstream_straddles_bytes() {
        let codes: Vec<u8> = vec![
            0b000, 0b111, 0b010, 0b101, 0b011, 0b110, 0b001, 0b100, 0b111, 0b000, 0b101, 0b010,
            0b110, 0b001, 0b100, 0b011,
        ];
        let (n, k) = (1usize, codes.len());
        // gs=16 keeps K=16 a multiple of the group; the bit math is
        // group-agnostic.
        let gs = 16u32;
        let packed = pack_b3(&codes);
        assert_eq!(packed.len(), codes.len() * 3 / 8);
        // One group covering the whole row.
        let scale_f = 0.5f32;
        let bias_f = -1.25f32;
        let scale = half::f16::from_f32(scale_f).to_bits().to_le_bytes();
        let bias = half::f16::from_f32(bias_f).to_bits().to_le_bytes();
        let out = affine_dequant_b4_to_dtype(
            &packed,
            &scale,
            &bias,
            n,
            k,
            gs,
            3,
            DType::F16,
            DType::F16,
        )
        .unwrap();
        let out_halves = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const u16, n * k) };
        for (i, &c) in codes.iter().enumerate() {
            let want = half::f16::from_f32(scale_f * (c as f32) + bias_f).to_bits();
            assert_eq!(out_halves[i], want, "element {i} (code {c:03b})");
        }
    }

    /// Multi-group, multi-row: every element must pick its OWN group's
    /// scale/bias — a group-index off-by-one inside the straddling
    /// bitstream fails here.
    #[test]
    fn affine_dequant_b3_groups_and_rows() {
        let (n, k, gs) = (2usize, 64, 32);
        let codes: Vec<u8> = (0..n * k).map(|i| (i % 8) as u8).collect();
        let packed = pack_b3(&codes);
        let n_groups = n * k / gs;
        let scales: Vec<u8> = (0..n_groups)
            .flat_map(|g| half::f16::from_f32(0.25 * (g as f32) + 0.5).to_bits().to_le_bytes())
            .collect();
        let biases: Vec<u8> = (0..n_groups)
            .flat_map(|g| half::f16::from_f32(-0.125 * (g as f32)).to_bits().to_le_bytes())
            .collect();
        let out = affine_dequant_b4_to_dtype(
            &packed, &scales, &biases, n, k, gs as u32, 3, DType::F16, DType::F16,
        )
        .unwrap();
        let out_halves = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const u16, n * k) };
        for (i, &c) in codes.iter().enumerate() {
            let g = i / gs;
            let s = 0.25 * (g as f32) + 0.5;
            let b = -0.125 * (g as f32);
            let want = half::f16::from_f32(s * (c as f32) + b).to_bits();
            assert_eq!(out_halves[i], want, "element {i}");
        }
    }

    /// bits=4 and bits=8 must remain byte-identical to the pre-b3 shapes:
    /// the packed-byte count formula changed shape, so pin it.
    #[test]
    fn affine_dequant_b4_b8_shapes_unchanged() {
        let (n, k, gs) = (2usize, 64, 32);
        let w4 = vec![0x12u8; n * k / 2];
        let w8 = vec![0xabu8; n * k];
        let n_groups = n * k / gs;
        let scales: Vec<u8> = vec![0x00; n_groups * 2];
        let biases: Vec<u8> = vec![0x00; n_groups * 2];
        let out4 = affine_dequant_b4_to_dtype(
            &w4, &scales, &biases, n, k, gs as u32, 4, DType::F16, DType::F16,
        )
        .unwrap();
        assert_eq!(out4.len(), n * k * 2);
        let out8 = affine_dequant_b4_to_dtype(
            &w8, &scales, &biases, n, k, gs as u32, 8, DType::F16, DType::F16,
        )
        .unwrap();
        assert_eq!(out8.len(), n * k * 2);
    }
}

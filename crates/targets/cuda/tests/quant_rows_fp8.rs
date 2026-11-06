// SPDX-License-Identifier: Apache-2.0
//! FP8 S3 on the H100: the per-token quantization (`kernels/quant_rows_fp8.cu`) against the host
//! quantizer bit for bit, and the host e4m3 encoder itself.
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips. A binary of its
//! own, so a fault in an fp8 GEMM's test (`tests/gemm_fp8.rs`) cannot leave these without a
//! verdict: a CUDA fault poisons the context for every later test in its process.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod fp8;
    pub mod numerics;
}

use common::fp8::{E4M3_MAX, e4m3_satfinite, e4m3_value, quantize_rows};
use common::numerics::bf16_inputs;
use half::bf16;
use scratchy_target_cuda::kernels::{KernelError, quant_rows_fp8};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};

fn h100() -> CudaDevice {
    CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"))
}

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

fn down_bytes(b: &DeviceBuffer) -> Vec<u8> {
    b.download()
        .expect("download")
        .iter()
        .flat_map(|h| h.to_le_bytes())
        .collect()
}

fn down_f32(b: &DeviceBuffer) -> Vec<f32> {
    down_bytes(b)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect()
}

// ── QuantRowsFp8 ───────────────────────────────────────────────────

/// Quantize `x [rows, cols]` on the device; returns `(q, scale)`.
fn device_quant(device: &CudaDevice, x: &[u16], rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
    let xb = DeviceBuffer::upload(device, x).expect("upload x");
    let q = DeviceBuffer::upload_bytes(device, &vec![0xA5; rows * cols + 256]).expect("q");
    let s = DeviceBuffer::upload_bytes(device, &vec![0xA5; rows * 4 + 256]).expect("scale");
    // SAFETY: `x` holds rows·cols bf16, `q` rows·cols bytes and `scale` rows f32, all cudaMalloc'd
    // (256-byte aligned) and disjoint; the download waits for the default stream.
    unsafe {
        quant_rows_fp8(
            xb.ptr(),
            (q.ptr(), s.ptr()),
            (rows as u32, cols as u32),
            std::ptr::null_mut(),
        )
    }
    .expect("quant_rows_fp8 launch");
    let (qd, sd) = (down_bytes(&q), down_f32(&s));
    let q_tail = qd[rows * cols..].iter().all(|&b| b == 0xA5);
    let s_tail = down_bytes(&s)[rows * 4..].iter().all(|&b| b == 0xA5);
    assert!(
        q_tail && s_tail,
        "rows={rows} cols={cols}: wrote past q or scale"
    );
    (qd[..rows * cols].to_vec(), sd[..rows].to_vec())
}

/// Every `k` of the three FP8 checkpoints' fp8 GEMMs (FP8 plan §1a), and a partial last 256-column
/// tile (320): bytes and scales bit-identical to the host quantizer at 16 and 4096 rows, one row
/// at each scale extreme (all zeros: the floor; one at ±448 times a large value).
#[test]
fn quant_rows_fp8_is_the_host_quantizer_bit_for_bit() {
    let device = h100();
    let mut pass = true;
    for cols in [
        320usize, 512, 704, 2048, 2112, 2816, 4096, 5120, 6144, 8192, 17408,
    ] {
        for rows in [16usize, 4096] {
            let mut x = bf16_inputs(rows * cols, 0xF8_0000 ^ cols as u64 ^ ((rows as u64) << 32));
            // Row 0 all zeros (the scale floor), row 1 scaled up past e4m3's range.
            x[..cols].fill(0);
            for v in &mut x[cols..2 * cols] {
                *v = bf16::from_f32(bf16::from_bits(*v).to_f32() * 3.0e4).to_bits();
            }
            let (hq, hs) = quantize_rows(&x, rows, cols);
            let (dq, ds) = device_quant(&device, &x, rows, cols);
            let q_bad = dq.iter().zip(&hq).filter(|(a, b)| a != b).count();
            let s_bad = (ds.iter().zip(&hs))
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let ok = q_bad == 0 && s_bad == 0;
            eprintln!(
                "quant_rows_fp8 rows={rows} cols={cols}: mismatched_bytes={q_bad} \
                 mismatched_scales={s_bad} {}",
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(
        pass,
        "a quantization differs from the host's: see the lines above"
    );
}

/// All 65,536 bf16 inputs through the device conversion: row `v` is `[v, 448, 0, …]` (64 columns),
/// so its scale is exactly 1 for every `|v| ≤ 448` and the device's byte for `v` is
/// `e4m3_satfinite(clamp(v / 1))`; above 448 (and ±inf, NaN) the scale is the row's own, and the
/// bytes still the host quantizer's. Bytes and scales bit-identical.
#[test]
fn quant_rows_fp8_converts_every_bf16_as_the_host_does() {
    let device = h100();
    let (rows, cols) = (65_536usize, 64usize);
    let mut x = vec![0u16; rows * cols];
    for v in 0..rows {
        x[v * cols] = v as u16;
        x[v * cols + 1] = bf16::from_f32(E4M3_MAX).to_bits();
    }
    let (hq, hs) = quantize_rows(&x, rows, cols);
    let (dq, ds) = device_quant(&device, &x, rows, cols);
    let mut first = None;
    let mut q_bad = 0usize;
    for v in 0..rows {
        if dq[v * cols..][..cols] != hq[v * cols..][..cols] {
            q_bad += 1;
            first.get_or_insert(v);
        }
    }
    let s_bad = (ds.iter().zip(&hs))
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let ok = q_bad == 0 && s_bad == 0;
    eprintln!(
        "quant_rows_fp8 every bf16: mismatched_rows={q_bad} first={first:?} \
         mismatched_scales={s_bad} {}",
        verdict(ok)
    );
    assert!(ok, "the device conversion differs from the host's");
}

#[test]
fn quant_rows_fp8_refuses_shapes_it_does_not_take() {
    let device = h100();
    let x = DeviceBuffer::upload(&device, &[0u16; 4096]).expect("x");
    let q = DeviceBuffer::upload_bytes(&device, &[0xA5u8; 4096]).expect("q");
    let mut pass = true;
    for (rows, cols) in [
        (0u32, 64u32),
        (8, 64),
        (17, 64),
        (16, 0),
        (16, 32),
        (16, 100),
    ] {
        // SAFETY: the launcher refuses each shape before it touches a pointer.
        let got = unsafe {
            quant_rows_fp8(
                x.ptr(),
                (q.ptr(), q.ptr()),
                (rows, cols),
                std::ptr::null_mut(),
            )
        };
        let kept = down_bytes(&q).iter().all(|&b| b == 0xA5);
        let ok = got == Err(KernelError::RowShape { rows, cols }) && kept;
        eprintln!(
            "quant_rows_fp8 refusal rows={rows} cols={cols}: {got:?} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "a shape was not refused");
}

// The host e4m3 itself, with no GPU: spot values of CUDA's satfinite conversion.
#[test]
fn the_host_e4m3_is_cudas_satfinite_conversion() {
    for (x, b) in [
        (0.0f32, 0x00u8),
        (-0.0, 0x80),
        (1.0, 0x38),
        (448.0, 0x7E),
        (500.0, 0x7E),
        (-1e9, 0xFE),
        (f32::INFINITY, 0x7E),
        (f32::NEG_INFINITY, 0xFE),
        (2f32.powi(-9), 0x01),
        (2f32.powi(-10), 0x00),
        (2f32.powi(-10) * 1.0001, 0x01),
        (1.0625, 0x38),
        (1.1875, 0x3A),
        (464.0, 0x7E),
    ] {
        assert_eq!(e4m3_satfinite(x), b, "{x}");
    }
    assert_eq!(e4m3_satfinite(f32::NAN), 0x7F);
    for b in 0u8..=0xFF {
        let v = e4m3_value(b);
        if v.is_nan() {
            continue;
        }
        assert_eq!(e4m3_satfinite(v as f32), b, "round trip of {b:#04x}");
    }
}

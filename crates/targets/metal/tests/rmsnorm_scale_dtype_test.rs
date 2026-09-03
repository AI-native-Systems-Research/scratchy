// SPDX-License-Identifier: Apache-2.0
//! Regression test for the metal RMSNorm gain-dtype mismatch.
//!
//! The `_s_<scale>_` rmsnorm kernel binds the gain through a
//! `device const T_scale*` whose dtype is fixed at compile time from
//! `W::SCALE_DTYPE` (F16 for the Llama family), NOT the on-disk dtype.
//! A standard HF bf16 checkpoint (e.g. `unsloth/Llama-3.2-1B-Instruct`)
//! ships BF16 gains; reading those bytes as F16 mis-scales every gain →
//! garbage generation (`defdef…`). `mlx-community/*-4bit` shipped F16
//! gains so it accidentally matched the default. The metal `load` body
//! pins `SCALE_DTYPE` via `GpuWeights::set_rmsnorm_scale_dtype`, and
//! `RmsNorm::load` routes through `take_as_dtype`, which converts the
//! gain ONLY on a genuine mismatch (the mlx zero-copy path is preserved).
//!
//! This test loads a tiny bf16 gain through the real metal loader and
//! asserts the resident buffer's dtype matches the pinned scale dtype
//! (converting when needed, keeping it otherwise).

use safetensors::Dtype;
use safetensors::tensor::TensorView;
use scratchy_target_metal::MetalAllocator;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::layers::{RmsNorm, RmsNormOps};
use scratchy_target_metal::weights::GpuWeights;
use scratchy_tensors::{DType, ScaleDtype};

/// Eight values exactly representable in both bf16 and f16, so the
/// bf16→f16 conversion round-trips losslessly and the readback is exact.
const VALS: [f32; 8] = [1.0, 0.5, 2.0, -1.5, 0.25, -4.0, 8.0, -0.125];

fn write_bf16_gain_fixture() -> std::path::PathBuf {
    let raw: Vec<u8> = VALS
        .iter()
        .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
        .collect();
    let view = TensorView::new(Dtype::BF16, vec![VALS.len()], &raw).expect("tensor view");
    let bytes = safetensors::serialize([("model.norm.weight", view)], None).expect("serialize");
    let path = std::env::temp_dir().join(format!(
        "scratchy_rmsnorm_scale_{}.safetensors",
        std::process::id()
    ));
    std::fs::write(&path, &bytes).expect("write fixture");
    path
}

/// Returns `(gw, gain)`. The caller MUST keep `gw` alive while reading the
/// gain: the `GpuTensor` is a raw pointer into `gw`'s allocator arena, which
/// is freed when `gw` drops (in production the model owns the allocator).
fn load_gain(
    path: &std::path::Path,
    gpu: &scratchy_target_metal::MetalDevice,
    pin: ScaleDtype,
) -> (GpuWeights, RmsNorm) {
    let alloc = MetalAllocator::new(gpu.device.clone());
    let mut gw = GpuWeights::from_single_file(path, alloc).expect("load fixture");
    gw.set_rmsnorm_scale_dtype(pin);
    let gain = <RmsNorm as RmsNormOps>::load(&mut gw, "model.norm", 1e-5).expect("rmsnorm load");
    (gw, gain)
}

#[test]
fn rmsnorm_gain_converts_to_pinned_scale_dtype() {
    let Some(gpu) = detect_device() else {
        eprintln!("skipping: no metal device");
        return;
    };
    let path = write_bf16_gain_fixture();

    // F16-pinned (Llama family): the bf16 on-disk gain MUST be converted to
    // F16 so it matches the `_s_f16_` kernel's `T_scale` pointer. This is the
    // exact case that produced garbage before the fix.
    let (_gw_f16, f16_gain) = load_gain(&path, &gpu, ScaleDtype::F16);
    assert_eq!(
        f16_gain.weight.dtype(),
        DType::F16,
        "gain must be converted to F16"
    );
    assert_eq!(f16_gain.weight.numel(), VALS.len());
    let p = f16_gain.weight.as_ptr::<half::f16>();
    for (i, &want) in VALS.iter().enumerate() {
        let got = unsafe { *p.add(i) }.to_f32();
        assert_eq!(got, want, "f16 gain[{i}] = {got}, want {want}");
    }

    // BF16-pinned (Qwen3 family / dense bf16): on-disk already bf16, so the
    // gain stays bf16 — NO conversion (mlx/zero-copy-class fast path).
    let (_gw_bf16, bf16_gain) = load_gain(&path, &gpu, ScaleDtype::Bf16);
    assert_eq!(
        bf16_gain.weight.dtype(),
        DType::BF16,
        "matching dtype must be left untouched (no needless conversion)"
    );
    let p = bf16_gain.weight.as_ptr::<half::bf16>();
    for (i, &want) in VALS.iter().enumerate() {
        let got = unsafe { *p.add(i) }.to_f32();
        assert_eq!(got, want, "bf16 gain[{i}] = {got}, want {want}");
    }

    let _ = std::fs::remove_file(&path);
}

// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for fused kernel implementations, dispatched on the
//! production MTL4 path (see `common::dispatch_threadgroups`).
//!
//! The classic `MetalStream` + `setBuffer`/`setBytes` encoder path is
//! replaced by argument-table `gpuAddress` bindings: every kernel input
//! is a `StorageModeShared` buffer bound at its `[[buffer(i)]]` index,
//! and each `setBytes` scalar becomes a tiny address-bound buffer
//! (`common::shared_u32` / `shared_f32`). Buffer DATA setup, CPU
//! references, shapes, tolerances, and assertions are unchanged.

mod common;

use half::f16;
use objc2_metal::MTLSize;
use scratchy_target_metal::{detect_device, fused_kernels::*};
use std::sync::Arc;

const ATOL_F16: f32 = 1e-2; // Relaxed tolerance for FP16 with transcendental functions (1%)

/// Reference implementation: RMSNorm
#[allow(dead_code)]
fn rmsnorm_reference(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = weight.len();
    let m = input.len() / n;
    let mut output = vec![0.0f32; input.len()];

    for i in 0..m {
        let row = &input[i * n..(i + 1) * n];

        // Compute mean of squares
        let mean_sq: f32 = row.iter().map(|x| x * x).sum::<f32>() / n as f32;
        let rms = (mean_sq + eps).sqrt();

        // Normalize and scale
        for j in 0..n {
            output[i * n + j] = (row[j] / rms) * weight[j];
        }
    }

    output
}

/// Reference implementation: Add + RMSNorm
fn add_rmsnorm_reference(
    input: &[f32],
    residual: &[f32],
    weight: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = weight.len();
    let m = input.len() / n;
    let mut output = vec![0.0f32; input.len()];
    let mut residual_out = vec![0.0f32; input.len()];

    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            residual_out[idx] = input[idx] + residual[idx];
        }

        let row = &residual_out[i * n..(i + 1) * n];
        let mean_sq: f32 = row.iter().map(|x| x * x).sum::<f32>() / n as f32;
        let rms = (mean_sq + eps).sqrt();

        for j in 0..n {
            output[i * n + j] = (row[j] / rms) * weight[j];
        }
    }

    (output, residual_out)
}

/// Reference implementation: SiLU
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Reference implementation: Gate-Up-SiLU-Mul
fn gate_up_silu_mul_reference(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(g, u)| silu(*g) * u)
        .collect()
}

#[test]
fn test_fused_add_rmsnorm_f16() {
    let Some(metal_device) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let fused_norm =
        FusedAddRmsNorm::new(Arc::new(metal_device.clone())).expect("Failed to create kernel");
    let device = &metal_device.device;

    // Test parameters
    let m = 4u32;
    let n = 128u32;
    let eps = 1e-5f32;

    // Generate test data
    let input: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01).collect();
    let residual: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.005).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();

    // Convert to f16
    let input_f16: Vec<f16> = input.iter().map(|&x| f16::from_f32(x)).collect();
    let residual_f16: Vec<f16> = residual.iter().map(|&x| f16::from_f32(x)).collect();
    let weight_f16: Vec<f16> = weight.iter().map(|&x| f16::from_f32(x)).collect();

    // Create buffers
    let input_buf = common::shared_slice(device, &input_f16);
    let residual_buf = common::shared_slice(device, &residual_f16);
    let weight_buf = common::shared_slice(device, &weight_f16);
    let output_buf = common::shared_zeroed(device, (m * n) as usize * std::mem::size_of::<f16>());
    let residual_out_buf =
        common::shared_zeroed(device, (m * n) as usize * std::mem::size_of::<f16>());

    // f16 + N divisible by 4 -> vectorized (vec4) pipeline; N_div4 = N/4.
    let n_param = n / 4;
    let m_buf = common::shared_u32(device, m);
    let n_buf = common::shared_u32(device, n_param);
    let eps_buf = common::shared_f32(device, eps);

    // Bindings (fused_add_rmsnorm_f16_vec4):
    //   0=input 1=residual 2=weight 3=output 4=residual_out
    //   5=M 6=N_div4 7=eps
    if !common::dispatch_threadgroups(
        device,
        &fused_norm.pipeline_f16_vec4,
        &[
            &input_buf,
            &residual_buf,
            &weight_buf,
            &output_buf,
            &residual_out_buf,
            &m_buf,
            &n_buf,
            &eps_buf,
        ],
        MTLSize {
            width: m as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: (n_param as usize).min(1024),
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    // Read results
    let output_f16: Vec<f16> = common::read_slice(&output_buf, (m * n) as usize);
    let residual_out_f16: Vec<f16> = common::read_slice(&residual_out_buf, (m * n) as usize);

    let output: Vec<f32> = output_f16.iter().map(|&x| x.to_f32()).collect();
    let residual_out: Vec<f32> = residual_out_f16.iter().map(|&x| x.to_f32()).collect();

    // Compute reference
    let (expected_output, expected_residual) =
        add_rmsnorm_reference(&input, &residual, &weight, eps);

    // Verify results
    for i in 0..output.len() {
        let diff = (output[i] - expected_output[i]).abs();
        assert!(
            diff < ATOL_F16,
            "Output mismatch at index {}: got {}, expected {}, diff {}",
            i,
            output[i],
            expected_output[i],
            diff
        );
    }

    for i in 0..residual_out.len() {
        let diff = (residual_out[i] - expected_residual[i]).abs();
        assert!(
            diff < ATOL_F16,
            "Residual output mismatch at index {}: got {}, expected {}, diff {}",
            i,
            residual_out[i],
            expected_residual[i],
            diff
        );
    }

    println!("✓ Fused Add+RMSNorm F16: All values within tolerance");
}

#[test]
fn test_fused_add_rmsnorm_vectorized() {
    let Some(metal_device) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let fused_norm =
        FusedAddRmsNorm::new(Arc::new(metal_device.clone())).expect("Failed to create kernel");
    let device = &metal_device.device;

    // Test with N divisible by 4 for vectorization
    let m = 2u32;
    let n = 256u32; // Divisible by 4
    let eps = 1e-5f32;

    let input: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01).collect();
    let residual: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.005).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();

    let input_f16: Vec<f16> = input.iter().map(|&x| f16::from_f32(x)).collect();
    let residual_f16: Vec<f16> = residual.iter().map(|&x| f16::from_f32(x)).collect();
    let weight_f16: Vec<f16> = weight.iter().map(|&x| f16::from_f32(x)).collect();

    let input_buf = common::shared_slice(device, &input_f16);
    let residual_buf = common::shared_slice(device, &residual_f16);
    let weight_buf = common::shared_slice(device, &weight_f16);
    let output_buf = common::shared_zeroed(device, (m * n) as usize * std::mem::size_of::<f16>());
    // residual_out is `None` in the classic path. MTL4 needs index 4
    // bound to keep the slice 0..N consecutive; a throwaway scratch
    // buffer absorbs the kernel's (now non-null) residual_out writes.
    let residual_out_scratch =
        common::shared_zeroed(device, (m * n) as usize * std::mem::size_of::<f16>());

    let n_param = n / 4;
    let m_buf = common::shared_u32(device, m);
    let n_buf = common::shared_u32(device, n_param);
    let eps_buf = common::shared_f32(device, eps);

    if !common::dispatch_threadgroups(
        device,
        &fused_norm.pipeline_f16_vec4,
        &[
            &input_buf,
            &residual_buf,
            &weight_buf,
            &output_buf,
            &residual_out_scratch,
            &m_buf,
            &n_buf,
            &eps_buf,
        ],
        MTLSize {
            width: m as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: (n_param as usize).min(1024),
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    let output_f16: Vec<f16> = common::read_slice(&output_buf, (m * n) as usize);
    let output: Vec<f32> = output_f16.iter().map(|&x| x.to_f32()).collect();

    let (expected_output, _) = add_rmsnorm_reference(&input, &residual, &weight, eps);

    for i in 0..output.len() {
        let diff = (output[i] - expected_output[i]).abs();
        assert!(
            diff < ATOL_F16,
            "Vectorized output mismatch at index {}: got {}, expected {}, diff {}",
            i,
            output[i],
            expected_output[i],
            diff
        );
    }

    println!("✓ Fused Add+RMSNorm Vectorized: All values within tolerance");
}

#[test]
fn test_fused_gate_up_silu_mul_separate() {
    let Some(metal_device) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let fused_silu =
        FusedGateUpSiluMul::new(Arc::new(metal_device.clone())).expect("Failed to create kernel");
    let device = &metal_device.device;

    let m = 4u32;
    let n = 128u32;

    // Generate test data
    let gate: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01 - 2.0).collect();
    let up: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.005 + 1.0).collect();

    let gate_f16: Vec<f16> = gate.iter().map(|&x| f16::from_f32(x)).collect();
    let up_f16: Vec<f16> = up.iter().map(|&x| f16::from_f32(x)).collect();

    let gate_buf = common::shared_slice(device, &gate_f16);
    let up_buf = common::shared_slice(device, &up_f16);
    let output_buf = common::shared_zeroed(device, (m * n) as usize * std::mem::size_of::<f16>());

    // f16 + N%4==0 -> vec4 pipeline. N_div4 bound as the N scalar; the
    // dispatch threadgroup width stays N (matches `execute_separate`).
    let n_param = n / 4;
    let m_buf = common::shared_u32(device, m);
    let n_buf = common::shared_u32(device, n_param);

    // Bindings (fused_gate_up_silu_mul_f16_vec4):
    //   0=gate_out 1=up_out 2=output 3=M 4=N_div4
    if !common::dispatch_threadgroups(
        device,
        &fused_silu.pipeline_f16_vec4,
        &[&gate_buf, &up_buf, &output_buf, &m_buf, &n_buf],
        MTLSize {
            width: m as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: (n as usize).min(1024),
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    let output_f16: Vec<f16> = common::read_slice(&output_buf, (m * n) as usize);
    let output: Vec<f32> = output_f16.iter().map(|&x| x.to_f32()).collect();

    let expected = gate_up_silu_mul_reference(&gate, &up);

    for i in 0..output.len() {
        let diff = (output[i] - expected[i]).abs();
        assert!(
            diff < ATOL_F16,
            "SiLU output mismatch at index {}: got {}, expected {}, diff {}",
            i,
            output[i],
            expected[i],
            diff
        );
    }

    println!("✓ Fused Gate-Up-SiLU-Mul (separate): All values within tolerance");
}

#[test]
fn test_fused_gate_up_silu_mul_concat() {
    let Some(metal_device) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let fused_silu =
        FusedGateUpSiluMul::new(Arc::new(metal_device.clone())).expect("Failed to create kernel");
    let device = &metal_device.device;

    let m = 4u32;
    let n = 128u32;

    // Generate concatenated gate_up data [M, 2*N]
    let gate: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01 - 2.0).collect();
    let up: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.005 + 1.0).collect();

    // Concatenate: [gate | up] for each row
    let mut gate_up = Vec::with_capacity((m * n * 2) as usize);
    for i in 0..m as usize {
        gate_up.extend_from_slice(&gate[i * n as usize..(i + 1) * n as usize]);
        gate_up.extend_from_slice(&up[i * n as usize..(i + 1) * n as usize]);
    }

    let gate_up_f16: Vec<f16> = gate_up.iter().map(|&x| f16::from_f32(x)).collect();

    let gate_up_buf = common::shared_slice(device, &gate_up_f16);
    let output_buf = common::shared_zeroed(device, (m * n) as usize * std::mem::size_of::<f16>());

    // f16 + N%4==0 -> concat vec4 pipeline. N_div4 bound as the N scalar.
    let n_param = n / 4;
    let m_buf = common::shared_u32(device, m);
    let n_buf = common::shared_u32(device, n_param);

    // Bindings (fused_gate_up_silu_mul_concat_f16_vec4):
    //   0=gate_up 1=output 2=M 3=N_div4
    if !common::dispatch_threadgroups(
        device,
        &fused_silu.pipeline_f16_concat_vec4,
        &[&gate_up_buf, &output_buf, &m_buf, &n_buf],
        MTLSize {
            width: m as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: (n as usize).min(1024),
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    let output_f16: Vec<f16> = common::read_slice(&output_buf, (m * n) as usize);
    let output: Vec<f32> = output_f16.iter().map(|&x| x.to_f32()).collect();

    let expected = gate_up_silu_mul_reference(&gate, &up);

    for i in 0..output.len() {
        let diff = (output[i] - expected[i]).abs();
        assert!(
            diff < ATOL_F16,
            "SiLU concat output mismatch at index {}: got {}, expected {}, diff {}",
            i,
            output[i],
            expected[i],
            diff
        );
    }

    println!("✓ Fused Gate-Up-SiLU-Mul (concat): All values within tolerance");
}

#[test]
fn test_fused_gate_up_gelu_mul() {
    let Some(metal_device) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let fused_silu =
        FusedGateUpSiluMul::new(Arc::new(metal_device.clone())).expect("Failed to create kernel");
    let device = &metal_device.device;

    let m = 2u32;
    let n = 64u32;

    let gate: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.02 - 1.0).collect();
    let up: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01 + 0.5).collect();

    let gate_f16: Vec<f16> = gate.iter().map(|&x| f16::from_f32(x)).collect();
    let up_f16: Vec<f16> = up.iter().map(|&x| f16::from_f32(x)).collect();

    let gate_buf = common::shared_slice(device, &gate_f16);
    let up_buf = common::shared_slice(device, &up_f16);
    let output_buf = common::shared_zeroed(device, (m * n) as usize * std::mem::size_of::<f16>());

    // Approximate GELU (`exact == false` -> pipeline_gelu_f16). The GELU
    // path binds N (not N/4) as the N scalar.
    let m_buf = common::shared_u32(device, m);
    let n_buf = common::shared_u32(device, n);

    // Bindings (fused_gate_up_gelu_mul_f16):
    //   0=gate_out 1=up_out 2=output 3=M 4=N
    if !common::dispatch_threadgroups(
        device,
        &fused_silu.pipeline_gelu_f16,
        &[&gate_buf, &up_buf, &output_buf, &m_buf, &n_buf],
        MTLSize {
            width: m as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: (n as usize).min(1024),
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    let output_f16: Vec<f16> = common::read_slice(&output_buf, (m * n) as usize);
    let output: Vec<f32> = output_f16.iter().map(|&x| x.to_f32()).collect();

    // Verify output is reasonable (GELU should be in [-1, 1] range roughly)
    for &val in &output {
        assert!(
            val.is_finite(),
            "GELU output contains non-finite value: {}",
            val
        );
    }

    println!("✓ Fused Gate-Up-GELU-Mul: Output is finite and reasonable");
}

#[test]
fn test_fused_kernels_numerical_stability() {
    let Some(metal_device) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let fused_norm =
        FusedAddRmsNorm::new(Arc::new(metal_device.clone())).expect("Failed to create kernel");
    let device = &metal_device.device;

    // Test with large values to check numerical stability
    let m = 2u32;
    let n = 64u32;
    let eps = 1e-5f32;

    let input: Vec<f32> = vec![100.0; (m * n) as usize];
    let residual: Vec<f32> = vec![50.0; (m * n) as usize];
    let weight: Vec<f32> = vec![1.0; n as usize];

    let input_f16: Vec<f16> = input.iter().map(|&x| f16::from_f32(x)).collect();
    let residual_f16: Vec<f16> = residual.iter().map(|&x| f16::from_f32(x)).collect();
    let weight_f16: Vec<f16> = weight.iter().map(|&x| f16::from_f32(x)).collect();

    let input_buf = common::shared_slice(device, &input_f16);
    let residual_buf = common::shared_slice(device, &residual_f16);
    let weight_buf = common::shared_slice(device, &weight_f16);
    let output_buf = common::shared_zeroed(device, (m * n) as usize * std::mem::size_of::<f16>());
    // residual_out `None` -> throwaway scratch at index 4 (see above).
    let residual_out_scratch =
        common::shared_zeroed(device, (m * n) as usize * std::mem::size_of::<f16>());

    let n_param = n / 4;
    let m_buf = common::shared_u32(device, m);
    let n_buf = common::shared_u32(device, n_param);
    let eps_buf = common::shared_f32(device, eps);

    if !common::dispatch_threadgroups(
        device,
        &fused_norm.pipeline_f16_vec4,
        &[
            &input_buf,
            &residual_buf,
            &weight_buf,
            &output_buf,
            &residual_out_scratch,
            &m_buf,
            &n_buf,
            &eps_buf,
        ],
        MTLSize {
            width: m as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: (n_param as usize).min(1024),
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    let output_f16: Vec<f16> = common::read_slice(&output_buf, (m * n) as usize);
    let output: Vec<f32> = output_f16.iter().map(|&x| x.to_f32()).collect();

    // Verify no NaN or Inf
    for (i, &val) in output.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Output contains non-finite value at index {}: {}",
            i,
            val
        );
    }

    println!("✓ Fused kernels numerical stability: No NaN/Inf with large inputs");
}

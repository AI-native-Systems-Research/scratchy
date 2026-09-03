// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for `scratchy_target_metal::cpu_reference`. Lives in
// `tests/` (rather than as an inline `#[cfg(test)] mod` next to the
// implementation) because the crate's lib-test target pulls in other
// test modules that currently fail to compile after the objc2-metal
// migration. Integration tests bypass that scope.

use scratchy_target_metal::cpu_reference::{
    affine_dequantize_b4_bf16, affine_dequantize_b4_f16, affine_qmm_n_b4_bf16,
    affine_qmm_t_b4_bf16, affine_qmm_t_b4_f16, affine_qmv_b4_f16, affine_qvm_b4_bf16,
};

#[test]
fn dequant_b4_f16_single_group_round_trip() {
    // scale=1, bias=0 → nibble values come straight through as 0..15.
    // Two output halves per packed byte (low nibble first).
    let packed: Vec<u8> = (0..8).map(|i| (i << 4) | (15 - i)).collect();
    let scales = vec![half::f16::from_f32(1.0); 1];
    let biases = vec![half::f16::ZERO; 1];
    let mut out = vec![half::f16::ZERO; 16];
    affine_dequantize_b4_f16(&packed, &scales, &biases, &mut out, 16);
    for (i, v) in out.iter().enumerate() {
        let want = if i % 2 == 0 {
            (15 - (i as i32 / 2)) as f32
        } else {
            (i as i32 / 2) as f32
        };
        assert_eq!(v.to_f32(), want, "i={i}");
    }
}

#[test]
fn dequant_b4_bf16_scale_bias_apply_per_group() {
    // gs=4 → 2 groups in 8 output elems.
    // group0 scale=2, bias=1 → nibbles (3,4,5,6) → 7, 9, 11, 13
    // group1 scale=0.5, bias=-1 → nibbles (1,2,3,4) → -0.5, 0, 0.5, 1
    // Packed bytes pack (lo, hi) → byte = hi<<4 | lo.
    let packed: Vec<u8> = vec![0x43, 0x65, 0x21, 0x43];
    // P10b: bf16-act path takes F16 scales/biases (`T_scale = half`).
    let scales = vec![half::f16::from_f32(2.0), half::f16::from_f32(0.5)];
    let biases = vec![half::f16::from_f32(1.0), half::f16::from_f32(-1.0)];
    let mut out = vec![half::bf16::ZERO; 8];
    affine_dequantize_b4_bf16(&packed, &scales, &biases, &mut out, 4);
    let want = [7.0, 9.0, 11.0, 13.0, -0.5, 0.0, 0.5, 1.0];
    for (i, (got, w)) in out.iter().zip(want).enumerate() {
        assert!(
            (got.to_f32() - w).abs() < 1e-3,
            "i={i}: want {w}, got {}",
            got.to_f32()
        );
    }
}

/// All-ones weights (scale=1, bias=0, every nibble=1) make `w[j,k]=1`
/// everywhere; transpose=true matmul collapses to row sums of `x`.
#[test]
fn qmm_t_b4_bf16_all_ones_matches_row_sum() {
    let m = 2;
    let n = 4;
    let k = 8;
    let gs = 8;
    let packed = vec![0x11_u8; n * k / 2];
    // P10b: bf16-act path takes F16 scales/biases.
    let scales = vec![half::f16::from_f32(1.0); n * k / gs];
    let biases = vec![half::f16::ZERO; n * k / gs];
    let x: Vec<half::bf16> = (0..(m * k))
        .map(|i| half::bf16::from_f32(i as f32))
        .collect();
    let y = affine_qmm_t_b4_bf16(&packed, &scales, &biases, &x, m, n, k, gs);
    for i in 0..m {
        let row_sum: f32 = (0..k).map(|kk| (i * k + kk) as f32).sum();
        for j in 0..n {
            assert!(
                (y[i * n + j].to_f32() - row_sum).abs() < 1e-2,
                "i={i} j={j}: want {row_sum}, got {}",
                y[i * n + j].to_f32()
            );
        }
    }
}

/// Same all-ones trick, transpose=false storage.
#[test]
fn qmm_n_b4_bf16_all_ones_matches_row_sum() {
    let m = 2;
    let n = 4;
    let k = 8;
    let gs = 4; // qmm_n requires N % gs == 0
    let packed = vec![0x11_u8; k * n / 2];
    // P10b: bf16-act path takes F16 scales/biases.
    let scales = vec![half::f16::from_f32(1.0); k * n / gs];
    let biases = vec![half::f16::ZERO; k * n / gs];
    let x: Vec<half::bf16> = (0..(m * k))
        .map(|i| half::bf16::from_f32(i as f32))
        .collect();
    let y = affine_qmm_n_b4_bf16(&packed, &scales, &biases, &x, m, n, k, gs);
    for i in 0..m {
        let row_sum: f32 = (0..k).map(|kk| (i * k + kk) as f32).sum();
        for j in 0..n {
            assert!(
                (y[i * n + j].to_f32() - row_sum).abs() < 1e-2,
                "i={i} j={j}: want {row_sum}, got {}",
                y[i * n + j].to_f32()
            );
        }
    }
}

#[test]
fn qmv_alias_matches_qmm_t() {
    let m = 1;
    let n = 4;
    let k = 8;
    let gs = 4;
    let packed: Vec<u8> = (0..(n * k / 2) as u8).collect();
    let scales: Vec<half::f16> = (0..(n * k / gs))
        .map(|i| half::f16::from_f32(0.01 + i as f32 * 0.001))
        .collect();
    let biases: Vec<half::f16> = (0..(n * k / gs))
        .map(|i| half::f16::from_f32(-0.05 + i as f32 * 0.002))
        .collect();
    let x: Vec<half::f16> = (0..(m * k))
        .map(|i| half::f16::from_f32((i as f32) * 0.1))
        .collect();
    let a = affine_qmm_t_b4_f16(&packed, &scales, &biases, &x, m, n, k, gs);
    let b = affine_qmv_b4_f16(&packed, &scales, &biases, &x, m, n, k, gs);
    assert_eq!(a, b);
}

#[test]
fn qvm_alias_matches_qmm_n() {
    let m = 1;
    let n = 4;
    let k = 8;
    let gs = 4;
    let packed: Vec<u8> = (0..(k * n / 2) as u8).collect();
    // P10b: bf16-act path takes F16 scales/biases.
    let scales: Vec<half::f16> = (0..(k * n / gs))
        .map(|i| half::f16::from_f32(0.01 + i as f32 * 0.001))
        .collect();
    let biases: Vec<half::f16> = (0..(k * n / gs))
        .map(|i| half::f16::from_f32(-0.05 + i as f32 * 0.002))
        .collect();
    let x: Vec<half::bf16> = (0..(m * k))
        .map(|i| half::bf16::from_f32((i as f32) * 0.1))
        .collect();
    let a = affine_qmm_n_b4_bf16(&packed, &scales, &biases, &x, m, n, k, gs);
    let b = affine_qvm_b4_bf16(&packed, &scales, &biases, &x, m, n, k, gs);
    assert_eq!(a, b);
}

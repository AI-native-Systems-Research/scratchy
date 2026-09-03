// SPDX-License-Identifier: Apache-2.0
//
// Slow CPU reference implementations of the affine-quantized
// matmul/dequant kernels, used by every `quantized_*_test.rs` in this
// crate to validate the Metal-side kernels.
//
// Layered here (rather than in `scratchy-forward-compiler::cpu_golden`) because
// `scratchy-target-metal` is below `scratchy-forward-compiler` in the crate
// graph — kernel tests cannot reach up. `cpu_golden` is welcome to
// re-export these (or move down) once a non-test consumer needs them.
//
// Each helper mirrors the math in `mlx/backend/metal/kernels/quantized.h`:
//
//   - `affine_dequantize_b4_*`: `w_q = scale * nibble + bias` per group.
//   - `affine_qmm_t_b4_*`: dequant `[N, K]` then `y[i,j] = Σ_k x[i,k] *
//     w[j,k]` (transpose=true). Same math as `affine_qmv` — the kernel
//     split is shape-driven, not math-driven.
//   - `affine_qmm_n_b4_*`: dequant `[K, N]` then `y[i,j] = Σ_k x[i,k] *
//     w[k,j]` (transpose=false). Same math as `affine_qvm`.
//
// All accumulations are f32 and cast back to the in/out dtype per
// output element, matching MLX's accumulation rule.

/// Internal trait covering `to_f32` / `from_f32` for `half::f16` and
/// `half::bf16`. Lets a single body cover both dtypes without runtime
/// dispatch.
trait HalfF: Copy {
    const ZERO: Self;
    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
}

impl HalfF for half::f16 {
    const ZERO: Self = half::f16::ZERO;
    #[inline]
    fn to_f32(self) -> f32 {
        half::f16::to_f32(self)
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        half::f16::from_f32(v)
    }
}

impl HalfF for half::bf16 {
    const ZERO: Self = half::bf16::ZERO;
    #[inline]
    fn to_f32(self) -> f32 {
        half::bf16::to_f32(self)
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        half::bf16::from_f32(v)
    }
}

/// Dequantize a packed 4-bit affine tensor where every two nibbles in
/// `packed` correspond to two consecutive output elements (contiguous-K
/// row order). `output.len()` must equal `packed.len() * 2` and be a
/// multiple of `group_size`; scales/biases are one entry per group.
///
/// `TAct` is the output dtype (the int4 kernels' `TAct` template
/// parameter); `TScale` is the scales/biases storage dtype (`TScale`).
/// After P10b the two diverge: every mlx-community 4bit ships F16
/// scales (`TScale = f16`) but the activation path is bf16 on Llama,
/// Qwen, Gemma. The dequant arithmetic mirrors MLX `quantized.h:521-527`:
/// scale + bias load into the activation type, multiply-add in
/// activation type.
fn affine_dequantize_b4<TAct: HalfF, TScale: HalfF>(
    packed: &[u8],
    scales: &[TScale],
    biases: &[TScale],
    output: &mut [TAct],
    group_size: usize,
) {
    assert_eq!(output.len(), packed.len() * 2);
    assert_eq!(
        output.len() % group_size,
        0,
        "output length {} not divisible by group_size {group_size}",
        output.len(),
    );
    assert_eq!(scales.len(), output.len() / group_size);
    assert_eq!(biases.len(), output.len() / group_size);

    for (offset, &byte) in packed.iter().enumerate() {
        let oindex = offset * 2;
        let gindex = oindex / group_size;
        // TScale → TAct cast happens here (mirrors the kernel's
        // in-register cast).
        let scale = TAct::from_f32(scales[gindex].to_f32()).to_f32();
        let bias = TAct::from_f32(biases[gindex].to_f32()).to_f32();
        let lo = (byte & 0x0f) as f32;
        let hi = ((byte >> 4) & 0x0f) as f32;
        output[oindex] = TAct::from_f32(scale * lo + bias);
        output[oindex + 1] = TAct::from_f32(scale * hi + bias);
    }
}

/// `affine_qmm_t` reference: dequantize `W` stored as `[N, K/2]` packed
/// bytes (row-major K-fast, one (scale, bias) per group of `gs`
/// consecutive K-elements in a row), then compute
/// `y[i,j] = Σ_k x[i,k] * w[j,k]` (i.e. `y = x @ wᵀ`).
///
/// Also serves as the reference for `affine_qmv` — the small-M and
/// large-M kernels differ in dispatch, not math.
fn affine_qmm_t_b4<TAct: HalfF, TScale: HalfF>(
    packed: &[u8],
    scales: &[TScale],
    biases: &[TScale],
    x: &[TAct],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<TAct> {
    assert_eq!(k % group_size, 0, "K must be a multiple of group_size");
    assert_eq!(packed.len(), n * k / 2);
    assert_eq!(scales.len(), n * k / group_size);
    assert_eq!(biases.len(), n * k / group_size);
    assert_eq!(x.len(), m * k);

    let mut w = vec![TAct::ZERO; n * k];
    affine_dequantize_b4::<TAct, TScale>(packed, scales, biases, &mut w, group_size);

    let mut y = vec![TAct::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc: f32 = 0.0;
            for kk in 0..k {
                acc += x[i * k + kk].to_f32() * w[j * k + kk].to_f32();
            }
            y[i * n + j] = TAct::from_f32(acc);
        }
    }
    y
}

/// `affine_qmm_n` reference: dequantize `W` stored as `[K, N/2]` packed
/// bytes (row-major N-fast), with one (scale, bias) per group of `gs`
/// consecutive N-columns per K-row, then compute
/// `y[i,j] = Σ_k x[i,k] * w[k,j]` (i.e. `y = x @ w`).
///
/// Also serves as the reference for `affine_qvm`.
fn affine_qmm_n_b4<TAct: HalfF, TScale: HalfF>(
    packed: &[u8],
    scales: &[TScale],
    biases: &[TScale],
    x: &[TAct],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<TAct> {
    assert_eq!(n % group_size, 0, "N must be a multiple of group_size");
    assert_eq!(packed.len(), k * n / 2);
    assert_eq!(scales.len(), k * n / group_size);
    assert_eq!(biases.len(), k * n / group_size);
    assert_eq!(x.len(), m * k);

    let groups_per_row = n / group_size;
    let bytes_per_row = n / 2;
    let mut w = vec![TAct::ZERO; k * n];
    for kk in 0..k {
        for byte_j in 0..bytes_per_row {
            let byte = packed[kk * bytes_per_row + byte_j];
            let n_col = 2 * byte_j;
            let group_idx = kk * groups_per_row + n_col / group_size;
            // Match the kernel's in-register TScale → TAct cast.
            let scale = TAct::from_f32(scales[group_idx].to_f32()).to_f32();
            let bias = TAct::from_f32(biases[group_idx].to_f32()).to_f32();
            let lo = (byte & 0x0f) as f32;
            let hi = ((byte >> 4) & 0x0f) as f32;
            w[kk * n + n_col] = TAct::from_f32(scale * lo + bias);
            w[kk * n + n_col + 1] = TAct::from_f32(scale * hi + bias);
        }
    }
    let mut y = vec![TAct::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc: f32 = 0.0;
            for kk in 0..k {
                acc += x[i * k + kk].to_f32() * w[kk * n + j].to_f32();
            }
            y[i * n + j] = TAct::from_f32(acc);
        }
    }
    y
}

/// 8-bit sibling of [`affine_qmm_t_b4`]: `W` is `[N, K]` with ONE byte
/// per element (`pack_factor = 1`), one (scale, bias) per group of
/// `group_size` consecutive K-elements per row:
/// `w = scale * byte + bias`. Mirrors MLX `quantized.h` bits=8 dequant.
/// Also serves as the reference for the 8-bit qmv kernels (the small-M
/// and large-M kernels differ in dispatch, not math). Gemma4-12B ships
/// its MLP projections in this layout (8-bit g64).
fn affine_qmm_t_b8<TAct: HalfF, TScale: HalfF>(
    packed: &[u8],
    scales: &[TScale],
    biases: &[TScale],
    x: &[TAct],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<TAct> {
    assert_eq!(k % group_size, 0, "K must be a multiple of group_size");
    assert_eq!(packed.len(), n * k);
    assert_eq!(scales.len(), n * k / group_size);
    assert_eq!(biases.len(), n * k / group_size);
    assert_eq!(x.len(), m * k);

    let groups_per_row = k / group_size;
    let mut w = vec![TAct::ZERO; n * k];
    for j in 0..n {
        for kk in 0..k {
            let group_idx = j * groups_per_row + kk / group_size;
            // Match the kernel's in-register TScale → TAct cast.
            let scale = TAct::from_f32(scales[group_idx].to_f32()).to_f32();
            let bias = TAct::from_f32(biases[group_idx].to_f32()).to_f32();
            let q = packed[j * k + kk] as f32;
            w[j * k + kk] = TAct::from_f32(scale * q + bias);
        }
    }
    let mut y = vec![TAct::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc: f32 = 0.0;
            for kk in 0..k {
                acc += x[i * k + kk].to_f32() * w[j * k + kk].to_f32();
            }
            y[i * n + j] = TAct::from_f32(acc);
        }
    }
    y
}

// ---- Concrete-dtype entry points ----------------------------------
//
// Post-P10b: the kernels' `TAct` and `TScale` template parameters are
// independent. The mlx-community 4bit checkpoints all ship `TScale =
// f16`; `TAct` follows `torch_dtype` (f16 or bf16). The `_bf16` /
// `_f16` suffix below names `TAct`; both variants now consume F16
// scales/biases. The previous symmetric (`TAct = TScale = bf16`)
// path is gone — that was the P1-P6 regression site.

pub fn affine_dequantize_b4_f16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    output: &mut [half::f16],
    group_size: usize,
) {
    affine_dequantize_b4::<half::f16, half::f16>(packed, scales, biases, output, group_size);
}

pub fn affine_dequantize_b4_bf16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    output: &mut [half::bf16],
    group_size: usize,
) {
    affine_dequantize_b4::<half::bf16, half::f16>(packed, scales, biases, output, group_size);
}

pub fn affine_qmm_t_b4_f16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    x: &[half::f16],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<half::f16> {
    affine_qmm_t_b4::<half::f16, half::f16>(packed, scales, biases, x, m, n, k, group_size)
}

pub fn affine_qmm_t_b4_bf16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    x: &[half::bf16],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<half::bf16> {
    affine_qmm_t_b4::<half::bf16, half::f16>(packed, scales, biases, x, m, n, k, group_size)
}

/// `TAct = bfloat, TScale = bfloat` reference — Qwen3 mlx-community
/// 4bit convention (see `INST_QMV_ALL(bf16, bfloat, bf16, bfloat, …)`
/// instantiations in `shaders/quantized_qmv.metal`).
pub fn affine_qmm_t_b4_bf16_s_bf16(
    packed: &[u8],
    scales: &[half::bf16],
    biases: &[half::bf16],
    x: &[half::bf16],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<half::bf16> {
    affine_qmm_t_b4::<half::bf16, half::bf16>(packed, scales, biases, x, m, n, k, group_size)
}

pub fn affine_qmm_n_b4_f16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    x: &[half::f16],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<half::f16> {
    affine_qmm_n_b4::<half::f16, half::f16>(packed, scales, biases, x, m, n, k, group_size)
}

pub fn affine_qmm_n_b4_bf16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    x: &[half::bf16],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<half::bf16> {
    affine_qmm_n_b4::<half::bf16, half::f16>(packed, scales, biases, x, m, n, k, group_size)
}

// Kernel-name aliases. `qmv` and `qmm_t` share math (transpose=true);
// `qvm` and `qmm_n` share math (transpose=false). The MLX dispatcher
// picks `qmv` vs `qmm_t` (and `qvm` vs `qmm_n`) on M and shape, not
// arithmetic — so the slow reference is identical.
pub use affine_qmm_n_b4_bf16 as affine_qvm_b4_bf16;
pub use affine_qmm_n_b4_f16 as affine_qvm_b4_f16;
pub use affine_qmm_t_b4_bf16 as affine_qmv_b4_bf16;
pub use affine_qmm_t_b4_bf16_s_bf16 as affine_qmv_b4_bf16_s_bf16;

pub fn affine_qmm_t_b8_f16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    x: &[half::f16],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<half::f16> {
    affine_qmm_t_b8::<half::f16, half::f16>(packed, scales, biases, x, m, n, k, group_size)
}

pub fn affine_qmm_t_b8_bf16_s_bf16(
    packed: &[u8],
    scales: &[half::bf16],
    biases: &[half::bf16],
    x: &[half::bf16],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<half::bf16> {
    affine_qmm_t_b8::<half::bf16, half::bf16>(packed, scales, biases, x, m, n, k, group_size)
}

pub use affine_qmm_t_b4_f16 as affine_qmv_b4_f16;
pub use affine_qmm_t_b8_bf16_s_bf16 as affine_qmv_b8_bf16_s_bf16;
pub use affine_qmm_t_b8_f16 as affine_qmv_b8_f16;

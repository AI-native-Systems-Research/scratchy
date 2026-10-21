// SPDX-License-Identifier: Apache-2.0
//! The GPU tests' inputs and reference: splitmix64 bf16 in [-1, 1], an f64 dot product of the same
//! bf16 inputs the GPU read, and the per-element bound
//! `|gpu − ref| ≤ 2⁻⁷·|ref| + 2⁻¹⁴·S`, with `S = Σₖ |A[i,k]·B[j,k]|`.
//!
//! The relative term is 2× the half-ulp the single bf16 rounding on store costs, and the absolute
//! term covers f32 accumulation-order error (about √K·2⁻²⁴·S) with a wide margin. A dropped or
//! repeated K step, or a wrong-layout B operand, moves an element far past the bound. NaN or Inf
//! fails.

use half::bf16;

/// splitmix64, mapped to bf16 in [-1, 1].
pub fn bf16_inputs(len: usize, seed: u64) -> Vec<u16> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let unit = (z >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
            bf16::from_f64(unit * 2.0 - 1.0).to_bits()
        })
        .collect()
}

pub fn widen(bits: &[u16]) -> Vec<f64> {
    bits.iter()
        .map(|&b| f64::from(bf16::from_bits(b).to_f32()))
        .collect()
}

/// `(ref, S)` for every element of `C = A · Bᵀ`, row-major, split across threads by row.
pub fn reference(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<(f64, f64)> {
    let mut out = vec![(0.0, 0.0); m * n];
    let threads = std::thread::available_parallelism().map_or(1, |p| p.get());
    let rows_per = m.div_ceil(threads);
    std::thread::scope(|s| {
        for (chunk, rows) in out.chunks_mut(rows_per * n).enumerate() {
            s.spawn(move || {
                for (r, row) in rows.chunks_mut(n).enumerate() {
                    let ar = &a[(chunk * rows_per + r) * k..][..k];
                    for (j, cell) in row.iter_mut().enumerate() {
                        let br = &b[j * k..][..k];
                        let (mut dot, mut abs) = (0.0, 0.0);
                        for (x, y) in ar.iter().zip(br) {
                            dot += x * y;
                            abs += (x * y).abs();
                        }
                        *cell = (dot, abs);
                    }
                }
            });
        }
    });
    out
}

/// How a GPU output compares with its reference.
pub struct Comparison {
    pub max_abs_err: f64,
    pub mean_abs_err: f64,
    /// The element with the largest error relative to its bound: `(row, col, gpu, ref, bound)`.
    pub worst: (usize, usize, f64, f64, f64),
    pub out_of_bound: usize,
}

impl Comparison {
    pub fn pass(&self) -> bool {
        self.out_of_bound == 0
    }

    /// `max_abs_err=.. mean_abs_err=.. worst=(i,j) gpu=.. ref=.. bound=.. out_of_bound=..`.
    pub fn line(&self) -> String {
        let (wi, wj, wgpu, wref, wbound) = self.worst;
        format!(
            "max_abs_err={:.6} mean_abs_err={:.6} worst=({wi},{wj}) gpu={wgpu} ref={wref:.6} \
             bound={wbound:.6} out_of_bound={}",
            self.max_abs_err, self.mean_abs_err, self.out_of_bound
        )
    }
}

/// Compare `gpu` (`[m, n]` bf16 bits) with `A · Bᵀ` of the bf16 inputs `a` (`[m, k]`) and `b`
/// (`[n, k]`) under the bound.
pub fn compare(gpu: &[u16], a: &[u16], b: &[u16], m: usize, n: usize, k: usize) -> Comparison {
    let refs = reference(&widen(a), &widen(b), m, n, k);
    let (mut max_err, mut sum_err, mut bad) = (0.0f64, 0.0f64, 0usize);
    let (mut worst, mut worst_ratio) = ((0, 0, 0.0f64, 0.0f64, 0.0f64), -1.0f64);
    for (idx, (&bits, &(r, s))) in gpu.iter().zip(&refs).enumerate() {
        let g = f64::from(bf16::from_bits(bits).to_f32());
        let bound = r.abs() / 128.0 + s / 16384.0;
        let err = (g - r).abs();
        if !(g.is_finite() && err <= bound) {
            bad += 1;
        }
        let ratio = if g.is_finite() {
            err / bound.max(f64::MIN_POSITIVE)
        } else {
            f64::INFINITY
        };
        if ratio > worst_ratio {
            worst_ratio = ratio;
            worst = (idx / n, idx % n, g, r, bound);
        }
        if g.is_finite() {
            max_err = max_err.max(err);
            sum_err += err;
        }
    }
    Comparison {
        max_abs_err: max_err,
        mean_abs_err: sum_err / (m * n) as f64,
        worst,
        out_of_bound: bad,
    }
}

// ── The row kernels (RMSNorm, add, SiluMul) ─────────────────────────
//
// Each reference is computed in f64 from the same bf16 values the GPU read. RMSNorm and SiluMul
// pass under `|gpu − ref| ≤ 2⁻⁸·|ref| + 2⁻²⁴`: 2⁻⁸ is twice the half-ulp of their one bf16
// rounding, and covers the f32 arithmetic and the fast-math `rsqrtf`, `__expf` and division with
// a wide margin; 2⁻²⁴ is the floor flush-to-zero needs. The add is bit-exact: its reference is the
// kernel's own sequence (an exact widening, one f32 add, one rounding).

/// The row kernels' relative bound, `2⁻⁸`.
pub const ROW_REL: f64 = 1.0 / 256.0;
/// The row kernels' absolute floor, `2⁻²⁴`.
pub const ROW_ABS: f64 = 1.0 / 16_777_216.0;

/// The row kernels' bound on an element whose reference is `r`.
pub fn row_bound(r: f64) -> f64 {
    r.abs() * ROW_REL + ROW_ABS
}

/// `x / sqrt(mean(x²) + eps) · (offset + g)` per element of `[rows, cols]`, in f64.
pub fn rms_norm_reference(
    x: &[u16],
    g: &[u16],
    (rows, cols): (usize, usize),
    eps: f64,
    offset: f64,
) -> Vec<f64> {
    let (x, g) = (widen(x), widen(g));
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let row = &x[r * cols..][..cols];
        let mean = row.iter().map(|v| v * v).sum::<f64>() / cols as f64;
        let inv = 1.0 / (mean + eps).sqrt();
        out.extend(row.iter().zip(&g).map(|(v, w)| v * inv * (offset + w)));
    }
    out
}

/// `g / (1 + e^(−g)) · u` per element, in f64.
pub fn silu_mul_reference(gate: &[u16], up: &[u16]) -> Vec<f64> {
    widen(gate)
        .iter()
        .zip(widen(up))
        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
        .collect()
}

/// `bf16::from_f32(f32(a) + f32(b))` per element: the add kernel's own sequence.
pub fn add_reference(a: &[u16], b: &[u16]) -> Vec<u16> {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| {
            let sum = bf16::from_bits(a).to_f32() + bf16::from_bits(b).to_f32();
            bf16::from_f32(sum).to_bits()
        })
        .collect()
}

/// Compare `gpu` (`[rows, cols]` bf16 bits) with `refs` under [`row_bound`]. A NaN or Inf in the
/// GPU output fails.
pub fn compare_rows(gpu: &[u16], refs: &[f64], cols: usize) -> Comparison {
    assert_eq!(
        gpu.len(),
        refs.len(),
        "the GPU output and the reference differ in length"
    );
    let (mut max_err, mut sum_err, mut bad) = (0.0f64, 0.0f64, 0usize);
    let (mut worst, mut worst_ratio) = ((0, 0, 0.0f64, 0.0f64, 0.0f64), -1.0f64);
    for (idx, (&bits, &r)) in gpu.iter().zip(refs).enumerate() {
        let g = f64::from(bf16::from_bits(bits).to_f32());
        let bound = row_bound(r);
        let err = (g - r).abs();
        if !(g.is_finite() && err <= bound) {
            bad += 1;
        }
        let ratio = if g.is_finite() {
            err / bound
        } else {
            f64::INFINITY
        };
        if ratio > worst_ratio {
            worst_ratio = ratio;
            worst = (idx / cols, idx % cols, g, r, bound);
        }
        if g.is_finite() {
            max_err = max_err.max(err);
            sum_err += err;
        }
    }
    Comparison {
        max_abs_err: max_err,
        mean_abs_err: sum_err / gpu.len().max(1) as f64,
        worst,
        out_of_bound: bad,
    }
}

/// How many of `gpu`'s elements differ in any bit from `want`'s.
pub fn mismatched_bits(gpu: &[u16], want: &[u16]) -> usize {
    assert_eq!(
        gpu.len(),
        want.len(),
        "the GPU output and the reference differ in length"
    );
    gpu.iter().zip(want).filter(|(g, w)| g != w).count()
}

/// `f` as bf16 bits.
pub fn bits(f: f32) -> u16 {
    bf16::from_f32(f).to_bits()
}

/// `len` bf16 gains in [-1/4, 1/4], as zero-centred checkpoints store them.
pub fn centred_gains(len: usize, seed: u64) -> Vec<u16> {
    bf16_inputs(len, seed)
        .into_iter()
        .map(|b| bits(bf16::from_bits(b).to_f32() / 4.0))
        .collect()
}

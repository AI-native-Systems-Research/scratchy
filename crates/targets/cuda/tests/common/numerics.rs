// SPDX-License-Identifier: Apache-2.0
//! The GPU tests' inputs and reference: splitmix64 bf16 in [-1, 1], an f64 dot product of the same
//! bf16 inputs the GPU read, and the per-element bound
//! `|gpu − ref| ≤ 2⁻⁷·|ref| + 2⁻¹⁴·S`, with `S = Σₖ |A[i,k]·B[j,k]|`.
//!
//! The relative term is twice the worst case of the single bf16 rounding on store (half an ulp,
//! just under 2⁻⁸ of the value), and the absolute term covers f32 accumulation-order error (about
//! √K·2⁻²⁴·S) with a wide margin. A dropped or
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

    /// `max_abs_err=.. mean_abs_err=.. worst=(i,j) gpu=.. ref=.. err=.. bound=.. ratio=..
    /// out_of_bound=..`: the global errors, then the element with the largest error relative to
    /// its bound, with its OWN error, its bound and their ratio (a pass has every ratio ≤ 1).
    pub fn line(&self) -> String {
        let (wi, wj, wgpu, wref, wbound) = self.worst;
        let werr = (wgpu - wref).abs();
        let ratio = werr / wbound.max(f64::MIN_POSITIVE);
        format!(
            "max_abs_err={:.6e} mean_abs_err={:.6e} worst=({wi},{wj}) gpu={wgpu} ref={wref:.6e} \
             err={werr:.6e} bound={wbound:.6e} ratio={ratio:.4} out_of_bound={}",
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
// pass under `|gpu − ref| ≤ 2⁻⁸·|ref| + 2⁻²⁴`. One bf16 rounding costs at most half an ulp, which
// is just under 2⁻⁸ of the value (the worst case is a midpoint just above a power of two), so
// 2⁻⁸ is the worst case of that one rounding, not twice it, and a worst-element ratio near 0.99
// is expected. The margin left there, about 2⁻¹⁶·|ref|, covers the f32 arithmetic (about 2⁻²⁰ or
// less, the fast-math `rsqrtf`, `__expf` and division included); 2⁻²⁴ is the floor
// flush-to-zero needs. The add is bit-exact: its reference is the
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

// ── The attention prelude and gate (rope, gate split, gate apply) ──
//
// Each reference is f64, from the same bf16 values the GPU read, the rotary table's included.
// A rotated pair passes under `|gpu − ref| ≤ 2⁻⁸·|ref| + 2⁻¹⁶·S + 2⁻²⁴`, `S = |x0·c| + |x1·s|`:
// the `S` term covers cancellation, where `ref` is near 0 but the f32 products are not. The
// pass-through lanes and the gate split are bit-exact. The gate apply passes under the row
// kernels' bound.

/// The rope's bound on a rotated element whose reference is `r`, its products' magnitudes `s`.
pub fn rope_bound(r: f64, s: f64) -> f64 {
    r.abs() * ROW_REL + s / 65_536.0 + ROW_ABS
}

/// The rope's reference for `x [rows, cols]` under `table [rows, rot_dim]`, as `(ref, S)` per
/// element: each head's lane `d < rot_dim/2` paired with lane `d + pair`, both rotated by the
/// table's pair `d`. `pair` is `rot_dim / 2` for the real rope; `head_dim / 2` is the full-head
/// pairing of the self-check, the values a kernel pairing across the whole head would write at
/// lanes `d` and `d + 128`. `None` for a lane the pairing does not rotate, which the real rope
/// passes through bit for bit.
pub fn rope_reference(
    x: &[u16],
    table: &[u16],
    (rows, cols): (usize, usize),
    (head_dim, rot_dim): (usize, usize),
    pair: usize,
) -> Vec<Option<(f64, f64)>> {
    let (x, t) = (widen(x), widen(table));
    let half = rot_dim / 2;
    let mut out = vec![None; rows * cols];
    for r in 0..rows {
        let (c, s) = t[r * rot_dim..][..rot_dim].split_at(half);
        for h in 0..cols / head_dim {
            let base = r * cols + h * head_dim;
            for d in 0..half {
                let (x0, x1) = (x[base + d], x[base + d + pair]);
                let mag = (x0 * c[d]).abs() + (x1 * s[d]).abs();
                out[base + d] = Some((x0 * c[d] - x1 * s[d], mag));
                out[base + d + pair] = Some((x1 * c[d] + x0 * s[d], mag));
            }
        }
    }
    out
}

/// How a rope's output compares with its reference: the rotated lanes under [`rope_bound`], and
/// the count of pass-through lanes that differ from the input in any bit.
pub struct RopeComparison {
    pub rotated: Comparison,
    pub pass_through_mismatched: usize,
}

/// Compare `gpu` (`[rows, cols]` bf16 bits) with `refs` ([`rope_reference`]) and the input `x`.
pub fn compare_rope(
    gpu: &[u16],
    x: &[u16],
    refs: &[Option<(f64, f64)>],
    cols: usize,
) -> RopeComparison {
    assert_eq!(
        gpu.len(),
        refs.len(),
        "the GPU output and the reference differ in length"
    );
    let (mut max_err, mut sum_err, mut bad, mut n) = (0.0f64, 0.0f64, 0usize, 0usize);
    let (mut worst, mut worst_ratio) = ((0, 0, 0.0f64, 0.0f64, 0.0f64), -1.0f64);
    let mut pass_through_mismatched = 0;
    for (idx, (&bits, r)) in gpu.iter().zip(refs).enumerate() {
        let Some((r, s)) = *r else {
            pass_through_mismatched += usize::from(bits != x[idx]);
            continue;
        };
        n += 1;
        let g = f64::from(bf16::from_bits(bits).to_f32());
        let bound = rope_bound(r, s);
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
    RopeComparison {
        rotated: Comparison {
            max_abs_err: max_err,
            mean_abs_err: sum_err / n.max(1) as f64,
            worst,
            out_of_bound: bad,
        },
        pass_through_mismatched,
    }
}

/// How many of the rotated elements of `gpu` a reference with another pairing (`wrong`) puts
/// outside its bound, and how many rotated elements there are: the full-head-pairing self-check.
pub fn rope_misses(gpu: &[u16], wrong: &[Option<(f64, f64)>]) -> (usize, usize) {
    let (mut missed, mut n) = (0, 0);
    for (&bits, r) in gpu.iter().zip(wrong) {
        if let Some((r, s)) = *r {
            n += 1;
            let g = f64::from(bf16::from_bits(bits).to_f32());
            missed += usize::from((g - r).abs() > rope_bound(r, s));
        }
    }
    (missed, n)
}

/// The per-head split of `qg [rows, 2·half_cols]`: `(q, gate)`, each `[rows, half_cols]`.
pub fn gate_split_reference(
    qg: &[u16],
    (rows, half_cols): (usize, usize),
    head_dim: usize,
) -> (Vec<u16>, Vec<u16>) {
    let (mut q, mut gate) = (Vec::with_capacity(rows * half_cols), Vec::new());
    gate.reserve(rows * half_cols);
    for row in qg.chunks_exact(2 * half_cols).take(rows) {
        for head in row.chunks_exact(2 * head_dim) {
            q.extend_from_slice(&head[..head_dim]);
            gate.extend_from_slice(&head[head_dim..]);
        }
    }
    (q, gate)
}

/// The split by halves of the whole row (`qg[:, :half]`, `qg[:, half:]`), the self-check's wrong
/// split.
pub fn whole_row_split(qg: &[u16], half_cols: usize) -> (Vec<u16>, Vec<u16>) {
    let (mut q, mut gate) = (Vec::new(), Vec::new());
    for row in qg.chunks_exact(2 * half_cols) {
        q.extend_from_slice(&row[..half_cols]);
        gate.extend_from_slice(&row[half_cols..]);
    }
    (q, gate)
}

/// `a / (1 + e^(−g))` per element, in f64.
pub fn gate_apply_reference(attn: &[u16], gate: &[u16]) -> Vec<f64> {
    widen(attn)
        .iter()
        .zip(widen(gate))
        .map(|(a, g)| a / (1.0 + (-g).exp()))
        .collect()
}

// ── Attention over the paged KV cache ──────────────────────────────
//
// The reference is f64, from the same bf16 Q, K and V the GPU read (the cache's contents read
// back). With `s_ij = scale · Σ_d q_id k_jd`, the allowed keys `A_i = { j : j ≤ P + i, j <
// kv_len }`, `p_ij = softmax_{j∈A_i}(s_ij)` and `ref_id = Σ_j p_ij v_jd`, every element passes
// under
//
//     |gpu_id − ref_id| ≤ 2⁻⁸·|ref_id| + 2⁻⁷·Σ_j p_ij·|v_jd| + 2⁻²⁴.
//
// `2⁻⁸·|ref|` is the one rounding of `o` to bf16 at the store. The kernel rounds each `p` to bf16
// before the P·V WGMMA, at most `2⁻⁸·Σ p |v|`; the other `2⁻⁸·Σ p |v|` is headroom for the score
// error (256 bf16 products accumulated in f32), `exp2.approx` and the f32 accumulation and
// rescales, each far smaller while `max scale · Σ_d |q k| ≤ 64` (printed by every case).

/// The attention bound's relative term, `2⁻⁸`.
pub const ATTN_REL: f64 = 1.0 / 256.0;
/// The attention bound's `Σ p |v|` term, `2⁻⁷`.
pub const ATTN_PV: f64 = 1.0 / 128.0;
/// The largest `scale · Σ_d |q_id k_jd|` the bound covers.
pub const ATTN_SCORE_MAX: f64 = 64.0;

/// One attention's inputs as the reference reads them: `q [rows, q_heads · hd]` and the keys and
/// values `[kv_len, kv_heads · hd]`, all bf16 bits; `P` and `L`.
pub struct AttnCase<'a> {
    pub q: &'a [u16],
    pub k: &'a [u16],
    pub v: &'a [u16],
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub prefix_len: usize,
    pub q_len: usize,
    pub scale: f64,
}

/// `(ref, Σ p |v|)` for every element of the first `q_len` rows of the output `[q_len, q_heads ·
/// hd]`, and the largest `scale · Σ_d |q k|` over the allowed scores.
pub fn attention_reference(c: &AttnCase<'_>) -> (Vec<(f64, f64)>, f64) {
    let (hd, qh, kvh) = (c.head_dim, c.q_heads, c.kv_heads);
    let (q, k, v) = (widen(c.q), widen(c.k), widen(c.v));
    let kv_len = c.prefix_len + c.q_len;
    let hr = qh / kvh;
    let mut out = vec![(0.0, 0.0); c.q_len * qh * hd];
    let mut score_max = 0.0f64;
    for i in 0..c.q_len {
        let allowed = (c.prefix_len + i + 1).min(kv_len);
        for g in 0..qh {
            let kh = g / hr;
            let qi = &q[(i * qh + g) * hd..][..hd];
            let mut s = Vec::with_capacity(allowed);
            for j in 0..allowed {
                let kj = &k[(j * kvh + kh) * hd..][..hd];
                let (mut dot, mut abs) = (0.0, 0.0);
                for (a, b) in qi.iter().zip(kj) {
                    dot += a * b;
                    abs += (a * b).abs();
                }
                score_max = score_max.max(c.scale * abs);
                s.push(c.scale * dot);
            }
            let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
            let sum: f64 = e.iter().sum();
            for d in 0..hd {
                let (mut r, mut pv) = (0.0, 0.0);
                for (j, ej) in e.iter().enumerate() {
                    let p = ej / sum;
                    let vj = v[(j * kvh + kh) * hd + d];
                    r += p * vj;
                    pv += p * vj.abs();
                }
                out[(i * qh + g) * hd + d] = (r, pv);
            }
        }
    }
    (out, score_max)
}

/// The attention's bound on an element whose reference is `r`, its `Σ p |v|` `pv`.
pub fn attention_bound(r: f64, pv: f64) -> f64 {
    r.abs() * ATTN_REL + pv * ATTN_PV + ROW_ABS
}

/// Compare the first `refs.len()` elements of `gpu` (rows of `cols`) with `refs`.
pub fn compare_attention(gpu: &[u16], refs: &[(f64, f64)], cols: usize) -> Comparison {
    let (mut max_err, mut sum_err, mut bad) = (0.0f64, 0.0f64, 0usize);
    let (mut worst, mut worst_ratio) = ((0, 0, 0.0f64, 0.0f64, 0.0f64), -1.0f64);
    for (idx, (&bits, &(r, pv))) in gpu.iter().zip(refs).enumerate() {
        let g = f64::from(bf16::from_bits(bits).to_f32());
        let bound = attention_bound(r, pv);
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
        mean_abs_err: sum_err / refs.len().max(1) as f64,
        worst,
        out_of_bound: bad,
    }
}

/// The global slot of token `pos` of a sequence over `table`, with `block_size` slots per block.
pub fn slot_of(table: &[u32], pos: usize, block_size: usize) -> usize {
    table[pos / block_size] as usize * block_size + pos % block_size
}

/// Gather a sequence's first `tokens` rows of one layer's paged cache (`[num_blocks, kv_heads,
/// block_size, hd]` bf16) into `[tokens, kv_heads · hd]`, through `table`.
pub fn gather_paged(
    cache: &[u16],
    table: &[u32],
    tokens: usize,
    (kv_heads, block_size, hd): (usize, usize, usize),
) -> Vec<u16> {
    let mut out = Vec::with_capacity(tokens * kv_heads * hd);
    for t in 0..tokens {
        let (block, off) = (table[t / block_size] as usize, t % block_size);
        for h in 0..kv_heads {
            let at = ((block * kv_heads + h) * block_size + off) * hd;
            out.extend_from_slice(&cache[at..][..hd]);
        }
    }
    out
}

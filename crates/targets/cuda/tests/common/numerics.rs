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
/// And the sliding window, when the attention has one.
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
    /// A row at position `q` also drops the keys `j ≤ q − window` (`q − j < window` is allowed:
    /// HF's mask, metal's `attention.metal`); `None` for full causal attention.
    pub window: Option<usize>,
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
        // The window's first key for the row at `P + i`; 0 without a window.
        let first = c
            .window
            .map_or(0, |w| (c.prefix_len + i + 1).saturating_sub(w));
        for g in 0..qh {
            let kh = g / hr;
            let qi = &q[(i * qh + g) * hd..][..hd];
            let mut s = Vec::with_capacity(allowed);
            for j in 0..allowed {
                // Below the window: a score of −inf, so its p is exactly 0 and it adds exactly
                // +0 to `ref` and `Σ p |v|`, as a key never visited would.
                if j < first {
                    s.push(f64::NEG_INFINITY);
                    continue;
                }
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

// ── The Gated-DeltaNet kernels ─────────────────────────────────────
//
// Every reference is f64, from the same values the GPU read: bf16 activations widened exactly,
// the f32 bundle, and the GPU's own f32 scratch for a stage's inputs, so each check measures one
// kernel's error. The bounds:
//
// - conv1d (f32): `|gpu − ref| ≤ 2⁻¹⁶·(|ref| + Σ_j |w_j·x_j|) + 2⁻²⁴`; the ring bit-exact.
// - gating (f32): `|gpu − ref| ≤ 2⁻¹⁶·|ref| + 2⁻²²` for `g` and for `β`.
// - scan (f32), the long-recurrence bound: for token `t` (counting earlier chunks), value head
//   `h`, row `i`, `|o_gpu − o_ref| ≤ 2⁻¹⁴·|o_ref| + 2⁻¹⁷·(t+1)·scale·Smax[t,h,i]`, `Smax` the
//   largest `‖S_ref,s[h,i,:]‖₂` over `s ≤ t`; the state after the last token
//   `|S_gpu − S_ref| ≤ 2⁻¹⁷·(t_last+1)·Smax[t_last,h,i]` elementwise. Each token's transition
//   `α·(I − β·k̂·k̂ᵀ)` has 2-norm at most 1, so an error made at token `s` is never amplified and
//   the errors at most add; `|δo| ≤ scale·‖δS row‖₂`. `2⁻¹⁷` is 4× over one token's error.
// - gated norm (bf16): the row kernels' `2⁻⁸·|ref| + 2⁻²⁴`; padding rows exactly `+0.0`.

/// `2⁻¹⁶`.
pub const GDN_CONV_REL: f64 = 1.0 / 65_536.0;
/// The gating's absolute floor, `2⁻²²`.
pub const GDN_GATING_ABS: f64 = 1.0 / 4_194_304.0;
/// The scan's relative term, `2⁻¹⁴`.
pub const GDN_SCAN_REL: f64 = 1.0 / 16_384.0;
/// The scan's per-token term, `2⁻¹⁷`.
pub const GDN_SCAN_STEP: f64 = 1.0 / 131_072.0;

/// One GDN chunk's geometry.
#[derive(Clone, Copy, Debug)]
pub struct GdnDims {
    pub nk: usize,
    pub nv: usize,
    pub head_k: usize,
    pub head_v: usize,
    pub conv_kernel: usize,
}

impl GdnDims {
    pub fn key_dim(&self) -> usize {
        self.nk * self.head_k
    }
    pub fn value_dim(&self) -> usize {
        self.nv * self.head_v
    }
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
    pub fn scale(&self) -> f64 {
        1.0 / (self.head_k as f64).sqrt()
    }
}

/// `x [q_len, conv_dim]`'s conv1d after `ring` (`[conv_dim, K−1]`, oldest first; zeros on a fresh
/// start) under `w [conv_dim, K]`: `(ref, Σ_j |w_j·x_j|)` per element of `[q_len, conv_dim]`, and
/// the new ring. `reversed` applies the taps newest first: the self-check's wrong kernel.
pub fn gdn_conv_reference(
    x: &[f64],
    w: &[f64],
    ring: &[f64],
    (q_len, conv_dim, k): (usize, usize, usize),
    reversed: bool,
) -> (Vec<(f64, f64)>, Vec<f64>) {
    let state = k - 1;
    let mut out = vec![(0.0, 0.0); q_len * conv_dim];
    let mut new_ring = vec![0.0; conv_dim * state];
    for c in 0..conv_dim {
        // `[ring ++ x[.., c]]`.
        let mut stream: Vec<f64> = ring[c * state..][..state].to_vec();
        stream.extend((0..q_len).map(|t| x[t * conv_dim + c]));
        for t in 0..q_len {
            let (mut sum, mut mag) = (0.0, 0.0);
            for j in 0..k {
                let tap = if reversed { k - 1 - j } else { j };
                let p = w[c * k + tap] * stream[t + j];
                sum += p;
                mag += p.abs();
            }
            out[t * conv_dim + c] = (sum / (1.0 + (-sum).exp()), mag);
        }
        new_ring[c * state..][..state].copy_from_slice(&stream[q_len..]);
    }
    (out, new_ring)
}

/// The conv's bound on an element whose reference is `r`, its taps' magnitudes `s`.
pub fn gdn_conv_bound(r: f64, s: f64) -> f64 {
    GDN_CONV_REL * (r.abs() + s) + ROW_ABS
}

/// `softplus(x)`: `ln(1 + eˣ)` for `x ≤ 20`, else `x`.
pub fn softplus(x: f64) -> f64 {
    if x <= 20.0 { x.exp().ln_1p() } else { x }
}

/// The gating of `a`, `b` (`[q_len, nv]`): `(g, β)` per element. `sigmoid` false is the
/// self-check's wrong gate, `β = b`.
pub fn gdn_gating_reference(
    a: &[f64],
    b: &[f64],
    a_log: &[f64],
    dt_bias: &[f64],
    nv: usize,
    sigmoid: bool,
) -> (Vec<f64>, Vec<f64>) {
    let g = (a.iter().enumerate())
        .map(|(i, &x)| -a_log[i % nv].exp() * softplus(x + dt_bias[i % nv]))
        .collect();
    let beta = (b.iter())
        .map(|&x| if sigmoid { 1.0 / (1.0 + (-x).exp()) } else { x })
        .collect();
    (g, beta)
}

/// The gating's bound on an element whose reference is `r`.
pub fn gdn_gating_bound(r: f64) -> f64 {
    GDN_CONV_REL * r.abs() + GDN_GATING_ABS
}

/// Which mutation of the scan a reference computes: the real one, or a self-check's wrong kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanVariant {
    Real,
    /// `u = β·v`: no `− S·k̂`.
    NoCorrection,
    /// The decay after the update instead of before.
    DecayLast,
    /// Key head `h % nk` instead of `h / (nv/nk)`.
    ModuloHeads,
}

/// The scan's f64 reference over `q_len` tokens of `conv_out [q_len, conv_dim]` (the GPU's f32
/// values), `g` and `beta` (`[q_len, nv]`), from `state` (`[nv, head_v, head_k]`), starting at
/// token `t0` of the sequence: per element of `o [q_len, value_dim]`, `(ref, (t0 + t + 1) · scale ·
/// Smax)`, the bound's second factor; and the final state with each row's `(t_last + 1) · Smax`.
pub struct ScanReference {
    pub o: Vec<(f64, f64)>,
    pub state: Vec<f64>,
    /// Per state row `(h, i)`: `(t_last + 1) · Smax`.
    pub state_scale: Vec<f64>,
    /// Per state row: the running `Smax`, for a continuation's reference.
    pub smax: Vec<f64>,
}

/// One scan reference's inputs: the GPU's f32 `conv_out`, `g` and `beta` of the chunk, the start
/// `state` and, for a continuation, the running `Smax` it left (`None` starts it from `state`),
/// and the chunk's first token `t0` within the sequence and its `q_len`.
pub struct ScanInputs<'a> {
    pub conv_out: &'a [f64],
    pub g: &'a [f64],
    pub beta: &'a [f64],
    pub state: &'a [f64],
    pub smax: Option<&'a [f64]>,
    pub t0: usize,
    pub q_len: usize,
}

pub fn gdn_scan_reference(x: &ScanInputs<'_>, d: GdnDims, variant: ScanVariant) -> ScanReference {
    let ScanInputs {
        conv_out,
        g,
        beta,
        state,
        smax: smax0,
        t0,
        q_len,
    } = *x;
    let (kd, vd, cd, hk, hv) = (d.key_dim(), d.value_dim(), d.conv_dim(), d.head_k, d.head_v);
    let scale = d.scale();
    let l2 = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
    // Each value head's recurrence is independent of the others': one thread per head.
    let per_head = |h: usize| {
        let kh = match variant {
            ScanVariant::ModuloHeads => h % d.nk,
            _ => h / (d.nv / d.nk),
        };
        let mut s = state[h * hv * hk..][..hv * hk].to_vec();
        let mut smax: Vec<f64> = match smax0 {
            Some(m) => m[h * hv..][..hv].to_vec(),
            None => (0..hv).map(|i| l2(&s[i * hk..][..hk]).sqrt()).collect(),
        };
        let mut o = vec![(0.0, 0.0); q_len * hv];
        for t in 0..q_len {
            let row = &conv_out[t * cd..][..cd];
            let q = &row[kh * hk..][..hk];
            let k = &row[kd + kh * hk..][..hk];
            let qn = 1.0 / (l2(q) + 1e-6).sqrt() * scale;
            let kn = 1.0 / (l2(k) + 1e-6).sqrt();
            let alpha = g[t * d.nv + h].exp();
            let b = beta[t * d.nv + h];
            for i in 0..hv {
                let sr = &mut s[i * hk..][..hk];
                let v = row[2 * kd + h * hv + i];
                if variant != ScanVariant::DecayLast {
                    sr.iter_mut().for_each(|x| *x *= alpha);
                }
                let sk: f64 = sr.iter().zip(k).map(|(a, b)| a * b * kn).sum();
                let u = match variant {
                    ScanVariant::NoCorrection => b * v,
                    _ => b * (v - sk),
                };
                sr.iter_mut().zip(k).for_each(|(x, kj)| *x += u * kj * kn);
                if variant == ScanVariant::DecayLast {
                    sr.iter_mut().for_each(|x| *x *= alpha);
                }
                let out: f64 = sr.iter().zip(q).map(|(a, b)| a * b * qn).sum();
                smax[i] = smax[i].max(l2(sr).sqrt());
                o[t * hv + i] = (out, (t0 + t + 1) as f64 * scale * smax[i]);
            }
        }
        (o, s, smax)
    };
    let heads: Vec<_> = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..d.nv).map(|h| sc.spawn(move || per_head(h))).collect();
        handles
            .into_iter()
            .map(|j| j.join().expect("a scan reference thread panicked"))
            .collect()
    });
    let mut o = vec![(0.0, 0.0); q_len * vd];
    let (mut s, mut smax) = (Vec::with_capacity(vd * hk), Vec::with_capacity(vd));
    for (h, (ho, hs, hm)) in heads.into_iter().enumerate() {
        for t in 0..q_len {
            o[t * vd + h * hv..][..hv].copy_from_slice(&ho[t * hv..][..hv]);
        }
        s.extend(hs);
        smax.extend(hm);
    }
    let t_last = t0 + q_len;
    let state_scale = smax.iter().map(|m| t_last as f64 * m).collect();
    ScanReference {
        o,
        state: s,
        state_scale,
        smax,
    }
}

/// The scan's bound on an output element `(ref, linear)`.
pub fn gdn_scan_bound(r: f64, linear: f64) -> f64 {
    GDN_SCAN_REL * r.abs() + GDN_SCAN_STEP * linear
}

/// The gated norm's reference of `o [q_len, value_dim]` (the GPU's f32) under `z` (bf16 widened)
/// and `w [head_v]`: one value per element. `silu` false is the self-check's sigmoid gate.
pub fn gdn_norm_reference(
    o: &[f64],
    z: &[f64],
    w: &[f64],
    (q_len, nv, head_v): (usize, usize, usize),
    eps: f64,
    silu: bool,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(q_len * nv * head_v);
    for row in 0..q_len * nv {
        let ov = &o[row * head_v..][..head_v];
        let zv = &z[row * head_v..][..head_v];
        let mean = ov.iter().map(|v| v * v).sum::<f64>() / head_v as f64;
        let inv = 1.0 / (mean + eps).sqrt();
        for i in 0..head_v {
            let sig = 1.0 / (1.0 + (-zv[i]).exp());
            let gate = if silu { zv[i] * sig } else { sig };
            out.push(ov[i] * inv * w[i] * gate);
        }
    }
    out
}

/// How an f32 GPU output compares with `(ref, bound)` pairs: out-of-bound count, worst ratio,
/// non-finite count.
pub struct F32Comparison {
    pub out_of_bound: usize,
    pub worst_ratio: f64,
    pub worst: usize,
    pub non_finite: usize,
    pub n: usize,
}

impl F32Comparison {
    pub fn pass(&self) -> bool {
        self.out_of_bound == 0
    }

    pub fn fraction(&self) -> f64 {
        self.out_of_bound as f64 / self.n.max(1) as f64
    }

    pub fn line(&self) -> String {
        format!(
            "n={} out_of_bound={} ({:.4}) worst_ratio={:.4} at {} non_finite={}",
            self.n,
            self.out_of_bound,
            self.fraction(),
            self.worst_ratio,
            self.worst,
            self.non_finite
        )
    }
}

/// Compare `gpu` with `refs` under `bound`, over the indices `keep` admits.
pub fn compare_f32(
    gpu: &[f32],
    refs: &[f64],
    bound: impl Fn(usize) -> f64,
    keep: impl Fn(usize) -> bool,
) -> F32Comparison {
    let mut c = F32Comparison {
        out_of_bound: 0,
        worst_ratio: -1.0,
        worst: 0,
        non_finite: 0,
        n: 0,
    };
    for (i, (&g, &r)) in gpu.iter().zip(refs).enumerate() {
        if !keep(i) {
            continue;
        }
        c.n += 1;
        let g = f64::from(g);
        let b = bound(i);
        let ratio = if g.is_finite() {
            (g - r).abs() / b.max(f64::MIN_POSITIVE)
        } else {
            c.non_finite += 1;
            f64::INFINITY
        };
        // NaN's ratio is infinite, so it is out of bound too.
        if ratio > 1.0 {
            c.out_of_bound += 1;
        }
        if ratio > c.worst_ratio {
            c.worst_ratio = ratio;
            c.worst = i;
        }
    }
    c
}

/// `bits` (f32 bit patterns as u16 pairs, little-endian) as f32.
pub fn f32_from_halves(halves: &[u16]) -> Vec<f32> {
    let (pairs, _) = halves.as_chunks::<2>();
    pairs
        .iter()
        .map(|&[lo, hi]| f32::from_bits(u32::from(lo) | (u32::from(hi) << 16)))
        .collect()
}

/// `f` as the u16 pairs `DeviceBuffer::upload` takes (little-endian).
pub fn f32_to_halves(f: &[f32]) -> Vec<u16> {
    f.iter()
        .flat_map(|v| {
            let b = v.to_bits();
            [b as u16, (b >> 16) as u16]
        })
        .collect()
}

/// splitmix64 values uniform in `[lo, hi]`, as f32.
pub fn f32_inputs(len: usize, seed: u64, (lo, hi): (f64, f64)) -> Vec<f32> {
    bf16_inputs(len, seed)
        .into_iter()
        .map(|b| {
            let unit = (f64::from(bf16::from_bits(b).to_f32()) + 1.0) / 2.0;
            (lo + (hi - lo) * unit) as f32
        })
        .collect()
}

/// splitmix64 bf16 uniform in `[lo, hi]`.
pub fn bf16_in(len: usize, seed: u64, (lo, hi): (f64, f64)) -> Vec<u16> {
    f32_inputs(len, seed, (lo, hi))
        .into_iter()
        .map(|v| bf16::from_f32(v).to_bits())
        .collect()
}

// ── Gemma 4's row maps (soft-cap, GeGLU) ───────────────────────────
//
// Each reference is f64 from the same bf16 inputs. Under `--use_fast_math` (build.rs) the CUDA
// programming guide's table of functions `--use_fast_math` affects maps `tanhf` to `__tanhf`
// (maximum relative error 2⁻¹¹), so the bounds are #921's `tanh.approx` ones: the soft-cap
// `(2⁻⁸ + 2⁻¹⁰)·|r| + 2⁻²⁴`, the GeGLU `2⁻⁸·|r| + 2⁻¹⁰·|g·u| + 2⁻²⁴` (the middle term is the tanh
// error, carried by `0.5·g·u`). The GKE SASS (`MUFU.TANH`) confirms which tanh ran.

/// The soft-cap's relative bound, `2⁻⁸ + 2⁻¹⁰`.
pub const SOFT_CAP_REL: f64 = 1.0 / 256.0 + 1.0 / 1024.0;
/// The GeGLU's tanh term, `2⁻¹⁰` of `|g·u|`.
pub const GELU_TANH: f64 = 1.0 / 1024.0;

/// `cap · tanh(x / cap)` per element, in f64.
pub fn soft_cap_reference(x: &[u16], cap: f64) -> Vec<f64> {
    widen(x).iter().map(|v| cap * (v / cap).tanh()).collect()
}

/// The soft-cap's bound on an element whose reference is `r`.
pub fn soft_cap_bound(r: f64) -> f64 {
    r.abs() * SOFT_CAP_REL + ROW_ABS
}

/// `0.5·g·(1 + tanh(√(2/π)·(g + 0.044715·g³)))·u` per element, in f64.
pub fn gelu_mul_reference(gate: &[u16], up: &[u16]) -> Vec<f64> {
    let c = (2.0 / std::f64::consts::PI).sqrt();
    widen(gate)
        .iter()
        .zip(widen(up))
        .map(|(g, u)| 0.5 * g * (1.0 + (c * (g + 0.044715 * g * g * g)).tanh()) * u)
        .collect()
}

/// The GeGLU's bound on an element whose reference is `r`, from inputs `g` and `u`.
pub fn gelu_mul_bound(r: f64, g: f64, u: f64) -> f64 {
    r.abs() * ROW_REL + (g * u).abs() * GELU_TANH + ROW_ABS
}

/// Compare `gpu` (`[rows, cols]` bf16 bits) with `refs` under `bound(index, ref)`. A NaN or Inf in
/// the GPU output fails.
pub fn compare_rows_with(
    gpu: &[u16],
    refs: &[f64],
    cols: usize,
    bound: impl Fn(usize, f64) -> f64,
) -> Comparison {
    assert_eq!(
        gpu.len(),
        refs.len(),
        "the GPU output and the reference differ in length"
    );
    let (mut max_err, mut sum_err, mut bad) = (0.0f64, 0.0f64, 0usize);
    let (mut worst, mut worst_ratio) = ((0, 0, 0.0f64, 0.0f64, 0.0f64), -1.0f64);
    for (idx, (&bits, &r)) in gpu.iter().zip(refs).enumerate() {
        let g = f64::from(bf16::from_bits(bits).to_f32());
        let b = bound(idx, r);
        let err = (g - r).abs();
        if !(g.is_finite() && err <= b) {
            bad += 1;
        }
        let ratio = if g.is_finite() {
            err / b
        } else {
            f64::INFINITY
        };
        if ratio > worst_ratio {
            worst_ratio = ratio;
            worst = (idx / cols, idx % cols, g, r, b);
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

/// A self-check's fraction: of the elements whose bf16 rounding `wrong` changes from `right`'s,
/// the share `gpu` is out of `bound` against `wrong`. `(fraction, changed)`.
pub fn out_of_bound_fraction(
    gpu: &[u16],
    (right, wrong): (&[f64], &[f64]),
    bound: impl Fn(usize, f64) -> f64,
) -> (f64, usize) {
    let (mut changed, mut out) = (0usize, 0usize);
    for (i, ((&g, &r), &w)) in gpu.iter().zip(right).zip(wrong).enumerate() {
        if bf16::from_f64(r).to_bits() == bf16::from_f64(w).to_bits() {
            continue;
        }
        changed += 1;
        let g = f64::from(bf16::from_bits(g).to_f32());
        let err = (g - w).abs();
        out += usize::from(err.is_nan() || err > bound(i, w));
    }
    (out as f64 / changed.max(1) as f64, changed)
}

// SPDX-License-Identifier: Apache-2.0
//! TurboQuant / PolarQuant — a faithful port of arozanov's `turboquant-mlx`
//! (`turboquant_mlx/{rotation,packing,quantizer}.py`) to Rust. Same algorithm,
//! same hardcoded Lloyd-Max codebook, same scaling structure, same bit-packing
//! layout. The host reference; the Metal kernels (`metal.py`) and the
//! dequant-to-buffer + incremental decode cache (`cache.py`) port on top.

// ── rotation.py ────────────────────────────────────────────────────────────

/// Fast Walsh-Hadamard transform, normalized by 1/sqrt(d) (self-inverse).
/// Port of `walsh_hadamard_transform`. `x.len()` must be a power of two.
pub fn walsh_hadamard_transform(x: &mut [f32]) {
    let d = x.len();
    debug_assert!(d > 0 && d & (d - 1) == 0, "dim must be power of 2, got {d}");
    let mut h = 1;
    while h < d {
        let mut i = 0;
        while i < d {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += 2 * h;
        }
        h <<= 1;
    }
    let inv = 1.0 / (d as f32).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Random ±1 diagonal. Port of `random_diagonal_sign` (p=0.5 Bernoulli);
/// splitmix64 stands in for MLX's RNG — any fixed random sign vector is valid
/// since the scheme is data-oblivious.
pub fn random_diagonal_sign(d: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..d)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            if z & 1 == 0 { 1.0 } else { -1.0 }
        })
        .collect()
}

// ── packing.py ───────────────────────────────────────────────────────────────

/// Values packed per uint32 word (port of `VALS_PER_WORD`). 3-bit → 10 (30/32
/// bits used), no straddling.
pub const fn vals_per_word(bits: u32) -> usize {
    match bits {
        1 => 32,
        2 => 16,
        3 => 10,
        4 => 8,
        _ => panic!("unsupported bit width"),
    }
}

/// uint32 words to pack `dim` indices at `bits` each. Port of `packed_dim`.
pub fn packed_dim(dim: usize, bits: u32) -> usize {
    let vpw = vals_per_word(bits);
    dim.div_ceil(vpw)
}

/// Pack u8 indices into uint32 words. Port of `pack_indices` (LSB-first, value
/// `i` shifted by `i*bits`; tail padded with zeros).
pub fn pack_indices(indices: &[u8], bits: u32) -> Vec<u32> {
    let vpw = vals_per_word(bits);
    let dim = indices.len();
    let pdim = packed_dim(dim, bits);
    let mut out = vec![0u32; pdim];
    for (w, word) in out.iter_mut().enumerate() {
        for i in 0..vpw {
            let idx = w * vpw + i;
            if idx < dim {
                *word |= (indices[idx] as u32) << (i as u32 * bits);
            }
        }
    }
    out
}

/// Unpack uint32 words to u8 indices. Port of `unpack_indices`.
pub fn unpack_indices(packed: &[u32], bits: u32, dim: usize) -> Vec<u8> {
    let vpw = vals_per_word(bits);
    let mask = (1u32 << bits) - 1;
    let mut out = vec![0u8; dim];
    for (w, &word) in packed.iter().enumerate() {
        for i in 0..vpw {
            let idx = w * vpw + i;
            if idx < dim {
                out[idx] = ((word >> (i as u32 * bits)) & mask) as u8;
            }
        }
    }
    out
}

// ── quantizer.py ─────────────────────────────────────────────────────────────

/// Hardcoded optimal Lloyd-Max centroids for N(0,1) (port of
/// `_compute_gaussian_codebook` — well-known values).
fn gaussian_codebook(bits: u32) -> Vec<f32> {
    match bits {
        1 => vec![-0.7979, 0.7979],
        2 => vec![-1.5104, -0.4528, 0.4528, 1.5104],
        3 => vec![
            -2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520,
        ],
        4 => vec![
            -2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3881, -0.1284, 0.1284, 0.3881,
            0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326,
        ],
        _ => panic!("unsupported bit width: {bits} (use 1-4)"),
    }
}

/// PolarQuant quantizer for a fixed dim + bit width. Port of `PolarQuantizer`.
#[derive(Clone, Debug)]
pub struct PolarQuantizer {
    pub dim: usize,
    pub bits: u32,
    signs: Vec<f32>,
    centroids: Vec<f32>,
    /// midpoints between adjacent centroids (`_compute_gaussian_boundaries`).
    boundaries: Vec<f32>,
    /// `1/sqrt(dim)` — the post-rotation coordinate std.
    scale: f32,
}

impl PolarQuantizer {
    pub fn new(dim: usize, bits: u32, seed: u64) -> Self {
        assert!(dim.is_power_of_two(), "dim must be power of 2 (got {dim})");
        let centroids = gaussian_codebook(bits);
        let boundaries = centroids.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();
        Self {
            dim,
            bits,
            signs: random_diagonal_sign(dim, seed),
            centroids,
            boundaries,
            scale: 1.0 / (dim as f32).sqrt(),
        }
    }

    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }
    pub fn signs(&self) -> &[f32] {
        &self.signs
    }
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Quantize one vector. Port of `PolarQuantizer.quantize`: f32 norm, unit,
    /// randomized-Hadamard rotation, divide by `scale` to lift coords to
    /// N(0,1), then digitize against the (unscaled) N(0,1) boundaries.
    pub fn quantize(&self, x: &[f32]) -> (Vec<u8>, f32) {
        debug_assert_eq!(x.len(), self.dim);
        let norm = x
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>()
            .sqrt() as f32;
        let safe = norm.max(1e-8);
        let mut r: Vec<f32> = x
            .iter()
            .zip(&self.signs)
            .map(|(&v, &s)| (v / safe) * s)
            .collect();
        walsh_hadamard_transform(&mut r);
        let inv_scale = 1.0 / self.scale;
        let idx = r
            .iter()
            .map(|&c| {
                let xs = c * inv_scale;
                let mut k = 0u8;
                for &b in &self.boundaries {
                    if xs > b {
                        k += 1;
                    }
                }
                k
            })
            .collect();
        (idx, norm)
    }

    /// Dequantize. Port of `PolarQuantizer.dequantize`: centroid lookup,
    /// `* scale`, inverse randomized-Hadamard, `* norm`.
    pub fn dequantize(&self, indices: &[u8], norm: f32) -> Vec<f32> {
        debug_assert_eq!(indices.len(), self.dim);
        let mut y: Vec<f32> = indices
            .iter()
            .map(|&i| self.centroids[i as usize] * self.scale)
            .collect();
        walsh_hadamard_transform(&mut y); // self-inverse
        y.iter()
            .zip(&self.signs)
            .map(|(&v, &s)| v * s * norm)
            .collect()
    }

    /// Quantize + pack (matches the cache's stored form).
    pub fn quantize_packed(&self, x: &[f32]) -> (Vec<u32>, f32) {
        let (idx, norm) = self.quantize(x);
        (pack_indices(&idx, self.bits), norm)
    }
}

/// Bytes stored per vector at `(dim, bits)`: packed codes + one f32 norm.
pub fn bytes_per_vec(dim: usize, bits: u32) -> usize {
    packed_dim(dim, bits) * 4 + 4
}

/// Single-stream TurboQuant KV store — the mechanism of arozanov's
/// `cache.py::TurboQuantKVCache.update_and_fetch` (standard K+V path): store
/// bit-packed codes + f32 norms; on read, fill an fp32 dequant buffer (full on
/// prefill, ONLY the new tokens on decode — the incremental decode buffer), and
/// hand that buffer to ordinary attention. Per-token dequant is independent, so
/// the incremental buffer is bit-identical to a full re-dequant; this struct is
/// the host correctness reference for that mechanism before the paged-cache +
/// worker wiring. (Production metal attention reads the packed store itself,
/// `attention.metal`; this uses the host quantizer to validate the logic.)
pub struct TurboQuantKvStore {
    q: PolarQuantizer,
    pdim: usize,
    /// packed codes, `offset * pdim` u32.
    packed: Vec<u32>,
    /// per-token norms, `offset`.
    norms: Vec<f32>,
    /// fp32 dequant buffer, `deq_offset * dim` — filled incrementally.
    deq_buf: Vec<f32>,
    deq_offset: usize,
    offset: usize,
}

impl TurboQuantKvStore {
    pub fn new(dim: usize, bits: u32, seed: u64) -> Self {
        Self {
            pdim: packed_dim(dim, bits),
            q: PolarQuantizer::new(dim, bits, seed),
            packed: Vec::new(),
            norms: Vec::new(),
            deq_buf: Vec::new(),
            deq_offset: 0,
            offset: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.offset
    }
    pub fn is_empty(&self) -> bool {
        self.offset == 0
    }

    /// Append `n_new` token vectors (`new_vecs` = `n_new * dim` row-major):
    /// quantize+pack+store, then dequant ONLY the new tokens into the buffer
    /// (the incremental decode buffer). Returns the dequant buffer slice for the
    /// full `offset` tokens — what attention reads.
    pub fn update_and_fetch(&mut self, new_vecs: &[f32]) -> &[f32] {
        let dim = self.q.dim;
        debug_assert_eq!(new_vecs.len() % dim, 0);
        let n_new = new_vecs.len() / dim;
        let prev = self.offset;
        let total = prev + n_new;

        // Quantize + store packed codes + norms.
        self.packed.resize(total * self.pdim, 0);
        self.norms.resize(total, 0.0);
        for t in 0..n_new {
            let (packed, norm) = self.q.quantize_packed(&new_vecs[t * dim..(t + 1) * dim]);
            let dst = (prev + t) * self.pdim;
            self.packed[dst..dst + self.pdim].copy_from_slice(&packed);
            self.norms[prev + t] = norm;
        }

        // Dequant: incremental (only the new tokens) when the buffer is current;
        // otherwise full (prefill / first fill).
        let incremental = self.deq_offset == prev && !self.deq_buf.is_empty();
        let (fill_from, fill_to) = if incremental {
            (prev, total)
        } else {
            (0, total)
        };
        if self.deq_buf.len() < total * dim {
            self.deq_buf.resize(total * dim, 0.0);
        }
        for t in fill_from..fill_to {
            let idx = unpack_indices(
                &self.packed[t * self.pdim..(t + 1) * self.pdim],
                self.q.bits,
                dim,
            );
            let recon = self.q.dequantize(&idx, self.norms[t]);
            self.deq_buf[t * dim..(t + 1) * dim].copy_from_slice(&recon);
        }
        self.offset = total;
        self.deq_offset = total;
        &self.deq_buf[..total * dim]
    }

    /// Bytes of compressed storage (codes + norms) vs the fp16 it replaces.
    pub fn compression_ratio(&self) -> f32 {
        if self.offset == 0 {
            return 1.0;
        }
        let stored = self.offset * (self.pdim * 4 + 4);
        let fp16 = self.offset * self.q.dim * 2;
        fp16 as f32 / stored as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::SQRT_2;

    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            // crude standard-normal via Box-Muller-ish CLT (3 uniforms).
            let mut a = 0.0f32;
            for _ in 0..3 {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                a += ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
            }
            a * SQRT_2
        }
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let d: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
        let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        (d / (na * nb).max(1e-12)) as f32
    }

    #[test]
    fn pack_round_trip_exact() {
        for bits in 1..=4u32 {
            let dim = 128;
            let mut rng = Lcg(7);
            let idx: Vec<u8> = (0..dim)
                .map(|_| (rng.0.wrapping_add(1) % (1 << bits)) as u8)
                .collect();
            let mut rng2 = Lcg(7);
            let idx: Vec<u8> = idx
                .iter()
                .map(|_| {
                    rng2.0 = rng2.0.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng2.0 >> 40) % (1 << bits)) as u8
                })
                .collect();
            let packed = pack_indices(&idx, bits);
            assert_eq!(packed.len(), packed_dim(dim, bits));
            let back = unpack_indices(&packed, bits, dim);
            assert_eq!(idx, back, "pack round-trip must be exact (bits={bits})");
        }
    }

    #[test]
    fn cosine_fidelity_matches_paper() {
        let dim = 128;
        let mut rng = Lcg(0xC0FFEE);
        let vecs: Vec<Vec<f32>> = (0..256)
            .map(|_| (0..dim).map(|_| rng.next_f32()).collect())
            .collect();
        for bits in [2u32, 3, 4] {
            let q = PolarQuantizer::new(dim, bits, 42);
            let mean: f32 = vecs
                .iter()
                .map(|v| {
                    let (idx, n) = q.quantize(v);
                    cosine(v, &q.dequantize(&idx, n))
                })
                .sum::<f32>()
                / vecs.len() as f32;
            println!(
                "PolarQuant {bits}-bit dim {dim}: mean cosine {mean:.4}  ({} B/vec)",
                bytes_per_vec(dim, bits)
            );
            if bits == 3 {
                assert!(mean > 0.97, "3-bit cosine {mean}");
            }
            if bits == 4 {
                assert!(mean > 0.99, "4-bit cosine {mean}");
            }
        }
    }

    /// Fidelity at every head_dim the metal path actually provisions,
    /// not just 128. `qwen2.5-0.5b` (head_dim 64) decoded garbage under
    /// TurboQuant while `llama-3.2-1b` (also 64) was fine, so the
    /// question "is the CODEBOOK weak at 64?" needed an answer that was
    /// a number rather than an argument.
    #[test]
    fn cosine_fidelity_across_head_dims() {
        for dim in [64usize, 128, 256] {
            let mut rng = Lcg(0xC0FFEE);
            let vecs: Vec<Vec<f32>> = (0..256)
                .map(|_| (0..dim).map(|_| rng.next_f32()).collect())
                .collect();
            for bits in [3u32, 4] {
                let q = PolarQuantizer::new(dim, bits, 42);
                let mean: f32 = vecs
                    .iter()
                    .map(|v| {
                        let (idx, n) = q.quantize(v);
                        cosine(v, &q.dequantize(&idx, n))
                    })
                    .sum::<f32>()
                    / vecs.len() as f32;
                println!("dim {dim} bits {bits}: mean cosine {mean:.4}");
                let floor = if bits == 3 { 0.97 } else { 0.99 };
                assert!(
                    mean > floor,
                    "dim {dim} {bits}-bit cosine {mean} <= {floor}"
                );
            }
        }
    }

    /// Fidelity on OUTLIER-HEAVY vectors — the regime the bit-width
    /// policy comment in `codegen.rs` warns about ("Qwen-class massive
    /// activations"). `qwen2.5-0.5b` decodes garbage under TurboQuant at
    /// the maximum 4 bits while fp16 KV is clean, and every kernel-level
    /// test passes, so the open question is whether the CODEBOOK simply
    /// cannot represent this distribution.
    #[test]
    fn cosine_fidelity_outlier_heavy() {
        let dim = 64usize;
        for (label, spike) in [
            ("gaussian", 0.0f32),
            ("one 10x outlier", 10.0),
            ("one 50x outlier", 50.0),
        ] {
            let mut rng = Lcg(0xA11CE);
            let vecs: Vec<Vec<f32>> = (0..256)
                .map(|i| {
                    let mut v: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
                    if spike > 0.0 {
                        v[i % dim] = spike;
                    }
                    v
                })
                .collect();
            for bits in [3u32, 4] {
                let q = PolarQuantizer::new(dim, bits, 42);
                let mean: f32 = vecs
                    .iter()
                    .map(|v| {
                        let (idx, n) = q.quantize(v);
                        cosine(v, &q.dequantize(&idx, n))
                    })
                    .sum::<f32>()
                    / vecs.len() as f32;
                println!("  {label:16} {bits}-bit dim {dim}: mean cosine {mean:.4}");
            }
        }
    }

    #[test]
    fn packed_quantize_matches_unpacked() {
        let dim = 128;
        let q = PolarQuantizer::new(dim, 3, 42);
        let mut rng = Lcg(1);
        let v: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
        let (idx, n) = q.quantize(&v);
        let (packed, np) = q.quantize_packed(&v);
        assert_eq!(n, np);
        assert_eq!(unpack_indices(&packed, 3, dim), idx);
    }

    #[test]
    fn incremental_decode_matches_full() {
        // The cache.py mechanism: prefill (full dequant) then decode tokens one
        // at a time (incremental dequant) must leave the buffer bit-identical to
        // a full re-dequant of every stored token.
        let (dim, bits, seed) = (128usize, 3u32, 42u64);
        let mut store = TurboQuantKvStore::new(dim, bits, seed);
        let refq = PolarQuantizer::new(dim, bits, seed);
        let mut rng = Lcg(0xDECA);

        // Prefill 40 tokens, then 24 single-token decode steps.
        let prefill: Vec<f32> = (0..40 * dim).map(|_| rng.next_f32()).collect();
        store.update_and_fetch(&prefill);
        let mut all: Vec<f32> = prefill.clone();
        for _ in 0..24 {
            let tok: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
            store.update_and_fetch(&tok);
            all.extend_from_slice(&tok);
        }
        let total = store.len();
        assert_eq!(total, 64);

        // The incrementally-built buffer must equal a full re-dequant.
        let buf = store.update_and_fetch(&[]); // no-op append, returns full buffer
        for t in 0..total {
            let (idx, n) = refq.quantize(&all[t * dim..(t + 1) * dim]);
            let want = refq.dequantize(&idx, n);
            let got = &buf[t * dim..(t + 1) * dim];
            for (a, b) in want.iter().zip(got) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "incremental buffer != full dequant at tok {t}"
                );
            }
        }
        println!(
            "incremental==full over {total} tokens; compression {:.2}x",
            store.compression_ratio()
        );
        assert!(
            store.compression_ratio() > 4.0,
            "3-bit compression should be ~4.6x"
        );
    }
}

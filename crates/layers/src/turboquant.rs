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
///
/// Held in ten-thousandths, the precision the tables are published at. Both
/// operands of `n / 1e4` are exact in f32, so each centroid is the correctly
/// rounded f32 of its 4-decimal value: bit-identical to the decimal literal.
/// Written as decimals, the 4-bit table's ±1.6180 trips clippy's
/// `approx_constant` as the golden ratio, which it only resembles.
fn gaussian_codebook(bits: u32) -> Vec<f32> {
    let ten_thousandths: &[i16] = match bits {
        1 => &[-7979, 7979],
        2 => &[-15104, -4528, 4528, 15104],
        3 => &[-21520, -13440, -7560, -2451, 2451, 7560, 13440, 21520],
        4 => &[
            -27326, -20690, -16180, -12562, -9423, -6568, -3881, -1284, 1284, 3881, 6568, 9423,
            12562, 16180, 20690, 27326,
        ],
        _ => panic!("unsupported bit width: {bits} (use 1-4)"),
    };
    ten_thousandths
        .iter()
        .map(|&n| f32::from(n) / 1e4)
        .collect()
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

/// A TurboQuant code width, in bits per rotated element. Only the widths
/// [`vals_per_word`] has a packing for exist: `TqBits::new(5)` in a const is a
/// compile error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TqBits(u32);

impl TqBits {
    pub const fn new(bits: u32) -> Self {
        let _ = vals_per_word(bits);
        Self(bits)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// How a model's KV cache is stored. Fixed per model when it is built: the
/// `turboquant` feature gives every model whose geometry the codec supports
/// [`KvCodec::TurboQuant`], and nothing chooses between them at runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvCodec {
    /// Uncompressed, in the model's own dtype.
    Dense,
    /// One packed code per rotated element and one f32 norm per head vector
    /// ([`bytes_per_vec`]).
    TurboQuant(TqBits),
}

impl KvCodec {
    pub const fn is_turboquant(self) -> bool {
        matches!(self, Self::TurboQuant(_))
    }
}

impl std::fmt::Display for KvCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense => f.write_str("dense"),
            Self::TurboQuant(bits) => write!(f, "TurboQuant {}-bit", bits.get()),
        }
    }
}

/// Minimum fp16 KV footprint (bytes per token, all layers, K+V) for a model
/// to be built with TurboQuant.
///
/// TurboQuant trades fidelity for KV CAPACITY. Below this, the capacity
/// is not the constraint and the trade is a bad one. Sized to sit
/// between the models measured on metal:
///
///     qwen2.5-0.5b   24 x 2 kv x 64  =  12 KiB/token   -> dense
///     llama-3.2-1b   16 x 8 kv x 64  =  32 KiB/token   -> TurboQuant
///     granite-4.1-3b 40 x 8 kv x 64  =  80 KiB/token   -> TurboQuant
///     gemma-3-4b     34 x 4 kv x 256 = 544 KiB/token   -> TurboQuant
///
/// A model between 12 and 32 KiB/token is untested either way; the
/// threshold is set at 24 KiB so the two measured points stay on the
/// sides they were measured on, and is a POLICY knob, not a law.
pub const MIN_KV_BYTES_PER_TOKEN: usize = 24 * 1024;

/// The widest head the codec's metal kernels take: their threadgroup arrays
/// hold one element per thread of a head.
pub const MAX_HEAD_DIM: u32 = 512;

/// The attention geometry a model's KV codec is decided from.
#[derive(Clone, Copy, Debug)]
pub struct KvGeometry {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: u32,
    /// The full-context layers' head_dim; `head_dim` on a uniform model.
    pub global_head_dim: u32,
    /// The KV row is a compressed latent (MLA), not per-head K and V.
    pub latent: bool,
    /// The model has a KV cache at all (encoders and vision towers don't).
    pub has_kv_cache: bool,
}

/// Why a model keeps a dense KV cache in a `turboquant` build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseReason {
    NoKvCache,
    /// The codec rotates head vectors; an MLA latent is not one.
    LatentKv,
    /// The rotation is a Walsh-Hadamard transform, over a power-of-two
    /// length no wider than [`MAX_HEAD_DIM`].
    HeadDim(u32),
    /// Below [`MIN_KV_BYTES_PER_TOKEN`].
    SmallKv {
        bytes_per_token: usize,
    },
}

impl std::fmt::Display for DenseReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoKvCache => f.write_str("it has no KV cache"),
            Self::LatentKv => f.write_str("its KV cache is an MLA latent, not per-head K and V"),
            Self::HeadDim(hd) => write!(
                f,
                "head_dim {hd} is not a power of two no wider than {MAX_HEAD_DIM}"
            ),
            Self::SmallKv { bytes_per_token } => write!(
                f,
                "its KV row is {} KiB/token, below the {} KiB TurboQuant threshold",
                bytes_per_token / 1024,
                MIN_KV_BYTES_PER_TOKEN / 1024
            ),
        }
    }
}

/// `bits`-bit TurboQuant for a model of `geometry`, or why it stays dense.
/// The one rule: the lowering injects the codec, the factory provisions it and
/// the worker sizes its pool from the `KV_CODEC` this decides.
pub fn codec_for(geometry: KvGeometry, bits: TqBits) -> Result<TqBits, DenseReason> {
    let KvGeometry {
        num_layers,
        num_kv_heads,
        head_dim,
        global_head_dim,
        latent,
        has_kv_cache,
    } = geometry;
    let supported = |hd: u32| hd.is_power_of_two() && hd <= MAX_HEAD_DIM;
    if !has_kv_cache {
        return Err(DenseReason::NoKvCache);
    }
    if latent {
        return Err(DenseReason::LatentKv);
    }
    if let Some(hd) = [head_dim, global_head_dim]
        .into_iter()
        .find(|&hd| !supported(hd))
    {
        return Err(DenseReason::HeadDim(hd));
    }
    let bytes_per_token = kv_bytes_per_token(
        KvCodec::Dense,
        num_layers,
        num_kv_heads,
        head_dim as usize,
        2,
    );
    if bytes_per_token < MIN_KV_BYTES_PER_TOKEN {
        return Err(DenseReason::SmallKv { bytes_per_token });
    }
    Ok(bits)
}

/// The width of the fp16 scratch attention stages TurboQuant K/V into.
pub const SCRATCH_ELEM_BYTES: usize = 2;

/// Bytes one token's K and V cost in a KV pool of `num_layers` layers of
/// `num_kv_heads × head_dim` stored as `codec`: dense, every layer's
/// `dense_elem_bytes`-wide row; TurboQuant, every layer's packed codes and
/// norms ([`bytes_per_vec`]) plus the one-layer fp16 scratch — what the metal
/// target's `build_tq_provision` allocates per token.
pub fn kv_bytes_per_token(
    codec: KvCodec,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dense_elem_bytes: usize,
) -> usize {
    match codec {
        KvCodec::Dense => num_layers * 2 * num_kv_heads * head_dim * dense_elem_bytes,
        KvCodec::TurboQuant(bits) => {
            num_layers * 2 * num_kv_heads * bytes_per_vec(head_dim, bits.get())
                + scratch_bytes_per_token(num_kv_heads, head_dim)
        }
    }
}

/// Bytes one token costs in a TurboQuant pool's one-layer fp16 scratch, K and V: what a model
/// staging through another pool's scratch does not pay.
pub fn scratch_bytes_per_token(num_kv_heads: usize, head_dim: usize) -> usize {
    2 * num_kv_heads * head_dim * SCRATCH_ELEM_BYTES
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

// SPDX-License-Identifier: Apache-2.0
//! Host-side glue shared by scratchy-models vision towers.
//!
//! Each vision arch module composes its own per-block GPU forward,
//! but the helpers here — 2D RoPE cos/sin tables, varlen `cu_seqlens`,
//! pixel patch flatten, K-pad-to-multiple-of-8, env-driven trace dump —
//! are arch-independent. Centralizing them avoids the per-arch
//! duplication that grew during Phase D/E.
//!
//! A mechanical lift, no behavior change. Later work moves the actual
//! per-block math from imperative `vision.rs` into a `#[vision_forward]`
//! DSL body; this crate stays as the host-side scaffold around it.
//!
//! [`VisionConfig`] carries the geometric fields every VL/MM tower
//! shares (embed_dim, num_heads, spatial_merge_size, patch_size,
//! temporal_patch_size, in_chans, depth, d_model, norm_eps). Host
//! helpers (`build_rope_cos_sin_bf16`, `patches_from_normalized_chw`)
//! hang off `VisionConfig` as methods. Each VL crate carries its own
//! arch-specific extras (`intermediate_size`, `window_size`, …) in a
//! sibling struct.

pub mod mm_meta;
pub mod preprocess;

pub use mm_meta::{MmMetadata, PlaceholderPolicy, PreprocessFn, SizePolicy, TokensPerImage};

/// RoPE base used by every Qwen2/2.5-VL tower AND MoonViT (LocateAnything).
/// Lift to a `VisionConfig` field if a future tower picks a different theta.
pub const ROPE_THETA: f32 = 10000.0;

/// Per-token 2D-RoPE angle layout + pairing convention of the vision tower.
/// Selected by the optional `vision_rope_style` config key; absent = `NeoxHw`
/// (every Qwen-VL tower).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VisionRopeStyle {
    /// Qwen2/2.5/3.5-VL: per-token angles `[h·f0..h·f17, w·f0..w·f17]`
    /// (h-block then w-block), applied GPT-NeoX `rotate_half`.
    NeoxHw,
    /// MoonViT (LocateAnything): angles `[x·f0, y·f0, x·f1, y·f1, …]`
    /// (interleaved, x = column first), applied to adjacent pairs
    /// (GPT-J / "traditional" complex multiply). Same `inv_freq` series
    /// as `NeoxHw` (`theta^(-2i/half_rot)` ≡ MoonViT's `theta^(-4i/dim)`).
    InterleavedXy,
}

/// Learned positional-embedding interpolation flavor (`vision_pos_emb_interp`
/// config key; absent = `Bilinear`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PosEmbInterp {
    /// Qwen3.5-VL `fast_pos_embed_interpolate`: 4-corner bilinear with
    /// `linspace(0, ng-1, n)` source mapping.
    Bilinear,
    /// MoonViT (LocateAnything) `Learnable2DInterpPosEmb`: torch-style
    /// bicubic (a = -0.75, align_corners = false, border taps dropped and
    /// weights renormalized), `src = (dst + 0.5)·in/out − 0.5` mapping.
    Bicubic,
}

/// Common vision-tower config — the geometric fields every VL/MM arch
/// shares. Arch-specific extras (`mlp_ratio`, `intermediate_size`,
/// `window_size`, …) live alongside this struct in the per-arch crate.
///
/// All host-side helpers (`build_rope_cos_sin_bf16`,
/// `build_cu_seqlens_i32`, `patches_from_normalized_chw`) hang off this
/// struct as methods, mirroring the way the text path's [`#[forward]`]
/// DSL hangs off a single `Weights`/`Config` surface.
#[derive(Clone, Copy, Debug)]
pub struct VisionConfig {
    pub embed_dim: u32,
    pub depth: u32,
    pub num_heads: u32,
    pub patch_size: u32,
    pub temporal_patch_size: u32,
    pub spatial_merge_size: u32,
    /// k for a `k×k` non-overlapping AvgPool2d the projector applies to
    /// reduce the patch grid (gemma3-mm SigLIP: k=4, 4096→256 tokens).
    /// `0` (or `1`) means no pooling — the merge-only token reduction
    /// (`spatial_merge_size²`) governs the output count. Token output
    /// rows = `total_patches / (spatial_merge_size² · pool_kernel²)`.
    pub pool_kernel: u32,
    pub in_chans: u32,
    /// Text-decoder hidden the patch-merger projects into. Equals
    /// `text_config.hidden_size`.
    pub d_model: u32,
    pub norm_eps: f32,
    pub rope_style: VisionRopeStyle,
    pub pos_emb_interp: PosEmbInterp,
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        (self.embed_dim / self.num_heads) as usize
    }
    pub fn half_rot(&self) -> usize {
        self.head_dim() / 2
    }
}

// ── 2D RoPE cos/sin construction ───────────────────────────────────
//
// Mirrors Python `Qwen2VisionTransformer.rot_pos_emb`:
//   for each (t, h, w) in grid_thw:
//     hpos = arange(h)[:,None].expand(h,w)        # [h, w]
//     wpos = arange(w)[None,:].expand(h,w)        # [h, w]
//     hpos = hpos.reshape(h/S, S, w/S, S).permute(0,2,1,3).flatten()
//     wpos = wpos.reshape(h/S, S, w/S, S).permute(0,2,1,3).flatten()
//     pos = stack([hpos, wpos], -1).repeat(t, 1)  # [t*h*w, 2]
//   pos = cat(per-image)                          # [total_l, 2]
//   inv_freq = 1 / theta**(arange(0, half_rot/2*2, 2) / (half_rot))
//                                                 # length half_rot/2
//   freqs = arange(max_grid).outer(inv_freq)     # [max_grid, half_rot/2]
//   cos_table = freqs.cos(); sin_table = freqs.sin()
//   cos_per_token = cos_table[pos].flatten(1)    # [total_l, half_rot]
// where S = spatial_merge_size, half_rot = head_dim / 2.
//
// Per-token cos has 2 halves: [cos(h_pos · inv_freq[..]), cos(w_pos · inv_freq[..])].
// `vision_rope_apply` reads cos[token, hi] for hi ∈ [0, half_rot), so the
// h/w split lives entirely in the cos/sin layout — kernel is generic.

impl VisionConfig {
    /// Build per-token bf16 cos/sin tables `[total_l, head_dim/2]` for the 2D
    /// vision RoPE. Result lives on the host; each VL crate uploads to the
    /// device with `alloc_gpu_tensor_from_host`.
    pub fn build_rope_cos_sin_bf16(
        &self,
        grid_thw: &[(u32, u32, u32)],
        total_l: usize,
    ) -> (Vec<u16>, Vec<u16>) {
        let head_dim = self.head_dim();
        let half_rot = head_dim / 2;
        let freq_axis_dim = half_rot / 2;
        let s = self.spatial_merge_size as usize;

        let inv_freq: Vec<f32> = (0..freq_axis_dim)
            .map(|i| 1.0 / ROPE_THETA.powf((2 * i) as f32 / (half_rot as f32)))
            .collect();

        let mut cos = Vec::<u16>::with_capacity(total_l * half_rot);
        let mut sin = Vec::<u16>::with_capacity(total_l * half_rot);
        for &(t, h, w) in grid_thw {
            let (h, w, t) = (h as usize, w as usize, t as usize);
            debug_assert_eq!(h % s, 0, "h must be divisible by spatial_merge_size");
            debug_assert_eq!(w % s, 0, "w must be divisible by spatial_merge_size");
            let h_blocks = h / s;
            let w_blocks = w / s;
            let frame_len = h * w;
            let mut hpos = vec![0u32; frame_len];
            let mut wpos = vec![0u32; frame_len];
            let mut idx = 0usize;
            for hb in 0..h_blocks {
                for wb in 0..w_blocks {
                    for sh in 0..s {
                        for sw in 0..s {
                            hpos[idx] = (hb * s + sh) as u32;
                            wpos[idx] = (wb * s + sw) as u32;
                            idx += 1;
                        }
                    }
                }
            }
            for _ in 0..t {
                for token in 0..frame_len {
                    let hp = hpos[token] as f32;
                    let wp = wpos[token] as f32;
                    match self.rope_style {
                        VisionRopeStyle::NeoxHw => {
                            for &f in inv_freq.iter() {
                                let theta_h = hp * f;
                                cos.push(f32_to_bf16(theta_h.cos()));
                                sin.push(f32_to_bf16(theta_h.sin()));
                            }
                            for &f in inv_freq.iter() {
                                let theta_w = wp * f;
                                cos.push(f32_to_bf16(theta_w.cos()));
                                sin.push(f32_to_bf16(theta_w.sin()));
                            }
                        }
                        VisionRopeStyle::InterleavedXy => {
                            // x (= column) angle first, then y, per freq.
                            for &f in inv_freq.iter() {
                                let theta_x = wp * f;
                                cos.push(f32_to_bf16(theta_x.cos()));
                                sin.push(f32_to_bf16(theta_x.sin()));
                                let theta_y = hp * f;
                                cos.push(f32_to_bf16(theta_y.cos()));
                                sin.push(f32_to_bf16(theta_y.sin()));
                            }
                        }
                    }
                }
            }
        }
        debug_assert_eq!(cos.len(), total_l * half_rot);
        (cos, sin)
    }

    /// Build the raw per-token 2D-RoPE angle table (`freqs`, f32),
    /// shape `[total_l, half_rot]` row-major (theta_h's then theta_w's
    /// per token). Identical position logic to
    /// [`Self::build_rope_cos_sin_bf16`] but emits the raw angles
    /// `theta = pos * inv_freq` instead of their cos/sin — the metal
    /// `vision_rope_2d` kernel reads `freqs` and computes cos/sin
    /// internally, whereas the cuda `vision_rope_apply` kernel consumes
    /// the precomputed cos/sin from `build_rope_cos_sin_bf16`.
    pub fn build_rope_freqs_f32(&self, grid_thw: &[(u32, u32, u32)], total_l: usize) -> Vec<f32> {
        let head_dim = self.head_dim();
        let half_rot = head_dim / 2;
        let freq_axis_dim = half_rot / 2;
        let s = self.spatial_merge_size as usize;

        let inv_freq: Vec<f32> = (0..freq_axis_dim)
            .map(|i| 1.0 / ROPE_THETA.powf((2 * i) as f32 / (half_rot as f32)))
            .collect();

        let mut freqs = Vec::<f32>::with_capacity(total_l * half_rot);
        for &(t, h, w) in grid_thw {
            let (h, w, t) = (h as usize, w as usize, t as usize);
            debug_assert_eq!(h % s, 0, "h must be divisible by spatial_merge_size");
            debug_assert_eq!(w % s, 0, "w must be divisible by spatial_merge_size");
            let h_blocks = h / s;
            let w_blocks = w / s;
            let frame_len = h * w;
            let mut hpos = vec![0u32; frame_len];
            let mut wpos = vec![0u32; frame_len];
            let mut idx = 0usize;
            for hb in 0..h_blocks {
                for wb in 0..w_blocks {
                    for sh in 0..s {
                        for sw in 0..s {
                            hpos[idx] = (hb * s + sh) as u32;
                            wpos[idx] = (wb * s + sw) as u32;
                            idx += 1;
                        }
                    }
                }
            }
            for _ in 0..t {
                for token in 0..frame_len {
                    let hp = hpos[token] as f32;
                    let wp = wpos[token] as f32;
                    match self.rope_style {
                        VisionRopeStyle::NeoxHw => {
                            for &f in inv_freq.iter() {
                                freqs.push(hp * f);
                            }
                            for &f in inv_freq.iter() {
                                freqs.push(wp * f);
                            }
                        }
                        VisionRopeStyle::InterleavedXy => {
                            // x (= column) angle first, then y, per freq —
                            // pairs with the `vision_rope_2d_interleaved`
                            // kernel's per-pair indexing.
                            for &f in inv_freq.iter() {
                                freqs.push(wp * f);
                                freqs.push(hp * f);
                            }
                        }
                    }
                }
            }
        }
        debug_assert_eq!(freqs.len(), total_l * half_rot);
        freqs
    }

    /// Host-side `fast_pos_embed_interpolate` (Qwen3.5-VL). For every
    /// output token — emitted directly in spatial-merge order so it
    /// pairs elementwise with the patch packing — bilinearly
    /// interpolates the learned positional table at the token's
    /// `(row, col)` mapped onto a `num_grid_per_side × num_grid_per_side`
    /// grid via `linspace(0, num_grid_per_side - 1, h|w)`. Mirror of
    /// mlx-vlm `qwen3_vl/vision.py::fast_pos_embed_interpolate` (the
    /// trailing spatial-merge `reshape/transpose` is fused into the
    /// `(hb, wb, sh, sw)` loop, exactly as `build_rope_freqs_f32` does).
    ///
    /// `table` is the learned `pos_embed.weight`, f32
    /// `[num_grid_per_side², embed_dim]`. Returns `[total_l, embed_dim]`
    /// f32 in merge order; the caller converts to the model dtype and
    /// uploads it as the `pos_embeds` runtime extern.
    pub fn fast_pos_embed_interpolate(
        &self,
        grid_thw: &[(u32, u32, u32)],
        num_grid_per_side: usize,
        table: &[f32],
        total_l: usize,
    ) -> Vec<f32> {
        let e = self.embed_dim as usize;
        let s = self.spatial_merge_size as usize;
        let ng = num_grid_per_side;
        debug_assert_eq!(table.len(), ng * ng * e, "pos_embed table shape");
        // linspace(0, ng-1, n)[i] = i*(ng-1)/(n-1); n==1 → 0 (np.linspace).
        let lin = |i: usize, n: usize| -> f32 {
            if n <= 1 {
                0.0
            } else {
                (i as f32) * ((ng - 1) as f32) / ((n - 1) as f32)
            }
        };
        let mut out = Vec::<f32>::with_capacity(total_l * e);
        for &(t, h, w) in grid_thw {
            let (h, w, t) = (h as usize, w as usize, t as usize);
            let (h_blocks, w_blocks) = (h / s, w / s);
            // One spatial frame in merge order; tiled `t` times below.
            let mut frame = Vec::<f32>::with_capacity(h * w * e);
            for hb in 0..h_blocks {
                for wb in 0..w_blocks {
                    for sh in 0..s {
                        for sw in 0..s {
                            let (row, col) = (hb * s + sh, wb * s + sw);
                            let (hf, wf) = (lin(row, h), lin(col, w));
                            let (h_floor, w_floor) = (hf as usize, wf as usize);
                            let h_ceil = (h_floor + 1).min(ng - 1);
                            let w_ceil = (w_floor + 1).min(ng - 1);
                            let (dh, dw) = (hf - h_floor as f32, wf - w_floor as f32);
                            let w00 = (1.0 - dh) * (1.0 - dw);
                            let w01 = (1.0 - dh) * dw;
                            let w10 = dh * (1.0 - dw);
                            let w11 = dh * dw;
                            let i00 = (h_floor * ng + w_floor) * e;
                            let i01 = (h_floor * ng + w_ceil) * e;
                            let i10 = (h_ceil * ng + w_floor) * e;
                            let i11 = (h_ceil * ng + w_ceil) * e;
                            for c in 0..e {
                                frame.push(
                                    w00 * table[i00 + c]
                                        + w01 * table[i01 + c]
                                        + w10 * table[i10 + c]
                                        + w11 * table[i11 + c],
                                );
                            }
                        }
                    }
                }
            }
            for _ in 0..t {
                out.extend_from_slice(&frame);
            }
        }
        debug_assert_eq!(out.len(), total_l * e);
        out
    }

    /// Host-side bicubic pos-embed interpolation (MoonViT /
    /// LocateAnything `Learnable2DInterpPosEmb`). Resamples the learned
    /// `[ng, ng, embed_dim]` table to the image's `(h, w)` patch grid with
    /// torch-style bicubic: kernel a = -0.75, `align_corners = false`
    /// (`src = (dst + 0.5)·in/out − 0.5`), out-of-range taps dropped and
    /// the remaining weights renormalized (mirrors mlx-vlm
    /// `kernels.bicubic_interpolate`'s accumulate-and-normalize Metal
    /// path; numpy transcription == mlx golden, cosine 0.9999991 — see
    /// tools/vision_parity).
    ///
    /// Tokens are emitted in spatial-merge order — the same `(hb, wb,
    /// sh, sw)` walk as [`Self::fast_pos_embed_interpolate`] /
    /// [`Self::patches_from_normalized_chw`] — so the result pairs
    /// elementwise with the packed patches. Returns `[total_l,
    /// embed_dim]` f32.
    pub fn bicubic_pos_embed_interpolate(
        &self,
        grid_thw: &[(u32, u32, u32)],
        num_grid_per_side: usize,
        table: &[f32],
        total_l: usize,
    ) -> Vec<f32> {
        let e = self.embed_dim as usize;
        let s = self.spatial_merge_size as usize;
        let ng = num_grid_per_side;
        debug_assert_eq!(table.len(), ng * ng * e, "pos_embed table shape");
        // Torch ATen bicubic kernel, a = -0.75, support 2.
        let cubic = |t: f32| -> f32 {
            const A: f32 = -0.75;
            let t = t.abs();
            if t <= 1.0 {
                (A + 2.0) * t * t * t - (A + 3.0) * t * t + 1.0
            } else if t < 2.0 {
                A * (t * t * t - 5.0 * t * t + 8.0 * t - 4.0)
            } else {
                0.0
            }
        };
        // align_corners=false source mapping + the 4-tap window
        // `floor(src - 2) + 1 .. floor(src + 2) + 1` clamped to [0, ng).
        let src_window = |dst: usize, out_n: usize| -> (f32, usize, usize) {
            let src = (dst as f32 + 0.5) * (ng as f32) / (out_n as f32) - 0.5;
            let start = ((src - 2.0).floor() as i64 + 1).max(0) as usize;
            let end = (((src + 2.0).floor() as i64 + 1).max(0) as usize).min(ng);
            (src, start, end)
        };
        let mut out = Vec::<f32>::with_capacity(total_l * e);
        let mut acc = vec![0f32; e];
        for &(t, h, w) in grid_thw {
            let (h, w, t) = (h as usize, w as usize, t as usize);
            let (h_blocks, w_blocks) = (h / s, w / s);
            // One spatial frame in merge order; tiled `t` times below.
            let mut frame = Vec::<f32>::with_capacity(h * w * e);
            for hb in 0..h_blocks {
                for wb in 0..w_blocks {
                    for sh in 0..s {
                        for sw in 0..s {
                            let (row, col) = (hb * s + sh, wb * s + sw);
                            let (y, ys, ye) = src_window(row, h);
                            let (x, xs, xe) = src_window(col, w);
                            acc.fill(0.0);
                            let mut wsum = 0f32;
                            for yy in ys..ye {
                                let wy = cubic(yy as f32 - y);
                                for xx in xs..xe {
                                    let wgt = wy * cubic(xx as f32 - x);
                                    wsum += wgt;
                                    let base = (yy * ng + xx) * e;
                                    for (a, &tv) in acc.iter_mut().zip(&table[base..base + e]) {
                                        *a += wgt * tv;
                                    }
                                }
                            }
                            let inv = 1.0 / wsum;
                            frame.extend(acc.iter().map(|&a| a * inv));
                        }
                    }
                }
            }
            for _ in 0..t {
                out.extend_from_slice(&frame);
            }
        }
        debug_assert_eq!(out.len(), total_l * e);
        out
    }
} // impl VisionConfig (rope)

/// Build per-image varlen `cu_seqlens` (one segment per (T, H, W) frame —
/// a (t, h, w) tuple contributes `t` segments of length `h * w`).
/// Returns the prefix-sum buffer and the max segment length.
///
/// Free function (not a `VisionConfig` method) because cu_seqlens depends
/// only on `grid_thw`; no config fields enter.
pub fn build_cu_seqlens_i32(grid_thw: &[(u32, u32, u32)]) -> (Vec<i32>, usize) {
    let mut cu = Vec::<i32>::with_capacity(grid_thw.len() + 1);
    cu.push(0);
    let mut max_seqlen = 0usize;
    let mut acc: i32 = 0;
    for &(t, h, w) in grid_thw {
        let seg = (h as usize) * (w as usize);
        for _ in 0..t {
            acc += seg as i32;
            cu.push(acc);
            if seg > max_seqlen {
                max_seqlen = seg;
            }
        }
    }
    (cu, max_seqlen)
}

// ── Qwen2VLImageProcessor patch-flatten ────────────────────────────
//
// Mirrors Python `Qwen2VLImageProcessor._preprocess`'s reshape +
// 9D transpose:
//
//   patches: [T, C, H, W]                              (image: T = temporal_patch_size, frames repeated)
//      -> reshape [grid_t, T, C, gH, mH, P, gW, mW, P]
//      -> transpose (0, 3, 6, 4, 7, 2, 1, 5, 8)
//         dims: [grid_t, gH, gW, mH, mW, C, T, P, P]
//      -> reshape [grid_t * grid_h * grid_w, C * T * P * P]
//
// where grid_h = H / P, grid_w = W / P, gH = grid_h / mH, gW = grid_w / mW.
//
// We don't materialize the 9D intermediate — we walk output index space
// directly, which costs ~4 MB f32 read + ~2 MB bf16 write per 392²
// image, negligible vs the encoder's per-block GEMM traffic.

impl VisionConfig {
    /// Patch-flatten one normalized CHW image to `[L, C·T·P²]` bf16
    /// patches + `(grid_t, grid_h, grid_w)`. `H` and `W` must be multiples
    /// of `patch_size · spatial_merge_size` (Python's smart_resize step
    /// guarantees this; the engine layer mirrors it).
    pub fn patches_from_normalized_chw(
        &self,
        pixels: &[f32],
        height: u32,
        width: u32,
    ) -> (Vec<u16>, (u32, u32, u32)) {
        let p = self.patch_size as usize;
        let s = self.spatial_merge_size as usize;
        let t = self.temporal_patch_size as usize;
        let c = self.in_chans as usize;
        let h = height as usize;
        let w = width as usize;
        assert_eq!(
            pixels.len(),
            c * h * w,
            "patches_from_normalized_chw: pixels.len() = {} but c*h*w = {}",
            pixels.len(),
            c * h * w
        );
        assert_eq!(
            h % (p * s),
            0,
            "image height {h} must be multiple of patch_size*merge_size = {}",
            p * s
        );
        assert_eq!(
            w % (p * s),
            0,
            "image width {w} must be multiple of patch_size*merge_size = {}",
            p * s
        );
        let grid_h = h / p;
        let grid_w = w / p;
        let g_h = grid_h / s;
        let g_w = grid_w / s;
        let grid_t: u32 = 1;
        let l = (grid_t as usize) * grid_h * grid_w;
        let feat = c * t * p * p;
        let mut out = vec![0u16; l * feat];
        let mut out_idx = 0usize;
        let stride_c = h * w;
        let stride_h = w;
        for gh in 0..g_h {
            for gw in 0..g_w {
                for mh in 0..s {
                    for mw in 0..s {
                        for ci in 0..c {
                            for _ti in 0..t {
                                let img_h_base = gh * (s * p) + mh * p;
                                let img_w_base = gw * (s * p) + mw * p;
                                for ph in 0..p {
                                    let img_h = img_h_base + ph;
                                    let row = ci * stride_c + img_h * stride_h + img_w_base;
                                    for pw in 0..p {
                                        let v = pixels[row + pw];
                                        out[out_idx] = f32_to_bf16(v);
                                        out_idx += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        debug_assert_eq!(out_idx, l * feat);
        (out, (grid_t, grid_h as u32, grid_w as u32))
    }
} // impl VisionConfig (patch flatten)

// ── K-pad to multiple of 8 ─────────────────────────────────────────
//
// cuBLAS BF16 GEMM rejects K=3420 (Qwen2.5-VL-3B intermediate_size)
// with `CUBLAS_STATUS_INTERNAL_ERROR`; padding K to the next multiple
// of 8 lands the GEMM on an algo cuBLAS supports. Tail columns are
// zero so `out[i, j] = sum_k input[i, k] * weight[j, k]` is unchanged
// when the activation feeds zeros into those padded columns (paired
// with the activation-side packer at the call site).

// CUDA device-side vision helpers (`pad_linear_k_to_mult8`) live in
// `scratchy-target-cuda::vision_device` — they name CUDA runtime
// types, kept out of this backend-neutral host-glue crate.

// ── Slice helpers ──────────────────────────────────────────────────

pub fn f32_to_bf16(v: f32) -> u16 {
    half::bf16::from_f32(v).to_bits()
}

pub fn bf16_slice_as_bytes(s: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

pub fn i32_slice_as_bytes(s: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

pub fn u32_slice_as_bytes(s: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

pub fn f32_slice_as_bytes(s: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

// ── Window-attention dispatch (Qwen2.5-VL family) ──────────────────
//
// Mirrors Python `Qwen2_5_VisionTransformer.get_window_index_thw`.
// Operates on the post-spatial-merge grid (`llm_h = H/S`, `llm_w = W/S`)
// in row-major. Pads each `(t, llm_h, llm_w)` grid up to a multiple of
// `vit_merger_window_size = window_size / S / patch_size` cells, then
// reshuffles by (window_h, window_w, intra_h, intra_w) so window-sized
// chunks land contiguously.

/// Host-built window-attention dispatch tables for one batch of images.
/// `window_index` permutes natural→window-grouped order (gather indices:
/// `permuted[i] = natural[window_index[i]]`); `reverse_indices` is its
/// inverse. `cu_window_seqlens` is the i32 prefix-sum of per-window
/// segment lengths in TOKEN units (post-multiply by S²); `max_seqlen`
/// is the max segment length in TOKEN units. The full-image
/// (per-frame) cu_seqlens is built separately via
/// [`build_cu_seqlens_i32`].
#[derive(Debug, Clone)]
pub struct WindowDispatch {
    /// Per-merged-cell natural→window-grouped permutation, length `total_l / S²`.
    pub window_index: Vec<u32>,
    /// Inverse of `window_index`, same length.
    pub reverse_indices: Vec<u32>,
    /// Prefix-sum of windowed segment lengths in TOKEN units. Length
    /// `(num_images * num_windows + 1)` (deduplicated by consecutive
    /// equality, matching Python `torch.unique_consecutive`).
    pub cu_window_seqlens: Vec<i32>,
    /// Max segment length in TOKEN units.
    pub max_seqlen_window: usize,
}

/// Build the window-attention dispatch for a Qwen2.5-VL-family vision
/// encoder. `window_size` is the spatial window edge in pixels (112 on
/// the family's standard config). `cfg` carries `spatial_merge_size` +
/// `patch_size` which together set the per-window merged-cell count
/// (`win_cells = window_size / S / patch_size`).
pub fn build_qwen2_5_window_dispatch(
    grid_thw: &[(u32, u32, u32)],
    cfg: &VisionConfig,
    window_size: u32,
) -> WindowDispatch {
    let s = cfg.spatial_merge_size as usize;
    let s2 = s * s;
    let p = cfg.patch_size as usize;
    let win_cells = (window_size as usize) / s / p;
    debug_assert!(win_cells > 0, "window_size/S/patch_size must be > 0");

    let total_merged: usize = grid_thw
        .iter()
        .map(|&(t, h, w)| (t as usize) * ((h as usize) / s) * ((w as usize) / s))
        .sum();
    let mut window_index = Vec::<u32>::with_capacity(total_merged);
    let mut cu = Vec::<i32>::with_capacity(grid_thw.len() * 4 + 1);
    cu.push(0);
    let mut window_index_id: u32 = 0;
    let mut cu_last: i32 = 0;
    let mut max_seqlen_cells = 0usize;

    for &(t, h, w) in grid_thw {
        let (t, h, w) = (t as usize, h as usize, w as usize);
        let llm_h = h / s;
        let llm_w = w / s;
        let pad_h = (win_cells - llm_h % win_cells) % win_cells;
        let pad_w = (win_cells - llm_w % win_cells) % win_cells;
        let nh = (llm_h + pad_h) / win_cells;
        let nw = (llm_w + pad_w) / win_cells;
        for ti in 0..t {
            let frame_base = (ti * llm_h * llm_w) as u32 + window_index_id;
            for wh in 0..nh {
                for ww in 0..nw {
                    let mut segment_cells: i32 = 0;
                    for ih in 0..win_cells {
                        for iw in 0..win_cells {
                            let row = wh * win_cells + ih;
                            let col = ww * win_cells + iw;
                            if row < llm_h && col < llm_w {
                                window_index.push(frame_base + (row * llm_w + col) as u32);
                                segment_cells += 1;
                            }
                        }
                    }
                    cu_last += segment_cells * (s2 as i32);
                    cu.push(cu_last);
                    if segment_cells as usize > max_seqlen_cells {
                        max_seqlen_cells = segment_cells as usize;
                    }
                }
            }
        }
        window_index_id += (t * llm_h * llm_w) as u32;
    }

    let cu_window_seqlens = dedup_consecutive(cu);
    let reverse_indices = invert_permutation(&window_index);
    WindowDispatch {
        window_index,
        reverse_indices,
        cu_window_seqlens,
        max_seqlen_window: max_seqlen_cells * s2,
    }
}

fn dedup_consecutive(v: Vec<i32>) -> Vec<i32> {
    let mut out = Vec::with_capacity(v.len());
    for x in v {
        if out.last().is_none_or(|&y| y != x) {
            out.push(x);
        }
    }
    out
}

fn invert_permutation(perm: &[u32]) -> Vec<u32> {
    let mut inv = vec![0u32; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p as usize] = i as u32;
    }
    inv
}

/// Permute a row-major `[total_l, inner]` bf16 tensor by S²-block
/// granularity using a `[total_l / S²]` permutation. Each merged cell
/// (S²-row block) is moved as a unit; intra-block row order is
/// preserved. Mirrors the Qwen2.5-VL host-side cos/sin permutation
/// pattern: reshape to `[L/S², S²·inner]`, gather by `window_index`,
/// reshape back to `[L, inner]`.
/// f32 sibling of [`permute_rows_block_grouped_bf16`] — same S²-block
/// row permutation over an f32 table. Used for the metal `freqs`
/// angle extern on windowed towers (Qwen2.5-VL): the `vision_rope_2d`
/// kernel reads per-token angle rows, so window-permuted tokens need
/// window-permuted rows exactly like the cuda cos/sin tables.
pub fn permute_rows_block_grouped_f32(
    src: &[f32],
    total_l: usize,
    inner: usize,
    spatial_merge_size: usize,
    permutation: &[u32],
) -> Vec<f32> {
    let s2 = spatial_merge_size * spatial_merge_size;
    debug_assert_eq!(total_l % s2, 0, "total_l ({total_l}) must be ÷ S² ({s2})");
    debug_assert_eq!(src.len(), total_l * inner, "src length mismatch");
    debug_assert_eq!(
        permutation.len(),
        total_l / s2,
        "permutation length mismatch"
    );
    let block_elems = s2 * inner;
    let mut out = vec![0f32; src.len()];
    for (dst_block, &src_block) in permutation.iter().enumerate() {
        let dst_start = dst_block * block_elems;
        let src_start = (src_block as usize) * block_elems;
        out[dst_start..dst_start + block_elems]
            .copy_from_slice(&src[src_start..src_start + block_elems]);
    }
    out
}

pub fn permute_rows_block_grouped_bf16(
    src: &[u16],
    total_l: usize,
    inner: usize,
    spatial_merge_size: usize,
    permutation: &[u32],
) -> Vec<u16> {
    let s2 = spatial_merge_size * spatial_merge_size;
    debug_assert_eq!(total_l % s2, 0, "total_l ({total_l}) must be ÷ S² ({s2})");
    debug_assert_eq!(src.len(), total_l * inner, "src length mismatch");
    debug_assert_eq!(
        permutation.len(),
        total_l / s2,
        "permutation length mismatch"
    );
    let block_elems = s2 * inner;
    let mut out = vec![0u16; src.len()];
    for (dst_block, &src_block) in permutation.iter().enumerate() {
        let dst_start = dst_block * block_elems;
        let src_start = (src_block as usize) * block_elems;
        out[dst_start..dst_start + block_elems]
            .copy_from_slice(&src[src_start..src_start + block_elems]);
    }
    out
}

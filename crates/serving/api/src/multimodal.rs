// SPDX-License-Identifier: Apache-2.0
//! Generic multimodal-processor consumer surface.
//!
//! scratchy-vision declares [`scratchy_vision::MmMetadata`] per arch
//! (placeholder token id key, size policy, tokens-per-image policy,
//! preprocess fn pointer); scratchy-forward-compiler collects those declarations
//! into an inventory; this module pairs the static metadata with
//! runtime knobs resolved from HF config files. No arch names appear
//! here — every per-arch knob is data on the metadata struct. Adding
//! a new MM arch requires zero edits to scratchy-serving-api.
//!
//! Resolution flow at startup ([`resolve`]):
//!
//! 1. `scratchy_forward_compiler::resolve_mm_metadata(&hf_config.architectures)`
//!    walks the inventory and returns the matching registration (or
//!    `None` for text-only).
//! 2. The metadata's [`scratchy_vision::SizePolicy`] +
//!    [`scratchy_vision::TokensPerImage`] are resolved against the
//!    `hf_config.extra` JSON object (image_size, patch_size,
//!    spatial_merge_size, mm_tokens_per_image, …) plus the optional
//!    `preprocessor_config.json` for smart-resize bounds.
//! 3. Returns a [`ResolvedMmProcessor`] the engine threads into
//!    request handling.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use scratchy_core_common::multimodal::ImageData;
use scratchy_vision::{MmMetadata, SizePolicy, TokensPerImage};

/// Lightweight `serde_json::Value` lookup used by [`ResolvedMmProcessor::resolve`]
/// — abstracts over the HF-config map shape (HashMap from `scratchy-core-config`,
/// `serde_json::Map` from arbitrary JSON readers).
pub trait HfExtra {
    fn get_value(&self, key: &str) -> Option<&serde_json::Value>;
}

impl HfExtra for HashMap<String, serde_json::Value> {
    fn get_value(&self, key: &str) -> Option<&serde_json::Value> {
        self.get(key)
    }
}

impl HfExtra for serde_json::Map<String, serde_json::Value> {
    fn get_value(&self, key: &str) -> Option<&serde_json::Value> {
        self.get(key)
    }
}

/// Static metadata + runtime-resolved knobs.
///
/// Created at engine init from
/// [`scratchy_forward_compiler::resolve_mm_metadata`] + `hf_config` +
/// `preprocessor_config.json`. Threaded to the engine via
/// `AsyncEngine::set_mm_processor`. Cheap to clone (`Arc`).
#[derive(Clone)]
pub struct ResolvedMmProcessor {
    /// Static per-arch declaration (frozen at compile time).
    pub metadata: &'static MmMetadata,
    /// Placeholder token id resolved from
    /// `hf_config.extra.<metadata.hf_token_id_key>` (fallback
    /// `metadata.hf_token_id_default`). The chat template emits this
    /// token; we expand each occurrence into N copies and the vision
    /// splice overwrites those positions.
    pub image_token_id: u32,
    /// Square fixed-size resolution from
    /// `vision_config.image_size` (fallback `metadata.default_image_size`).
    /// Always populated even for `SmartResize` — used by the encoder
    /// path for default-image-size sizing of single-image cases.
    pub image_size: usize,
    /// `vision_config.patch_size`. Used by `PerImageGrid` to compute
    /// per-image token counts.
    pub patch_size: usize,
    /// `vision_config.spatial_merge_size` (fallback from policy).
    /// Used by `PerImageGrid`.
    pub spatial_merge_size: u32,
    /// Static tokens-per-image when [`TokensPerImage::FromConfig`].
    /// Read from `hf_config.extra.mm_tokens_per_image`; fallback to
    /// `(image_size / patch_size)²`.
    pub mm_tokens_per_image: usize,
    /// Smart-resize lower bound. Honored only when `metadata.size_policy`
    /// is `SmartResize`. Read from `preprocessor_config.json::min_pixels`
    /// when present; falls back to the policy's default.
    pub min_pixels: usize,
    /// Smart-resize upper bound. Same source as `min_pixels`.
    pub max_pixels: usize,
}

impl ResolvedMmProcessor {
    /// Resolve runtime knobs from HF config inputs. `hf_extra` is the
    /// flat `hf_config.extra` JSON object. `model_dir` (when present)
    /// lets us read a colocated `preprocessor_config.json` for
    /// `SmartResize` bounds.
    pub fn resolve<E: HfExtra + ?Sized>(
        metadata: &'static MmMetadata,
        hf_extra: &E,
        model_dir: Option<&Path>,
    ) -> Self {
        let image_token_id = hf_extra
            .get_value(metadata.hf_token_id_key)
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(metadata.hf_token_id_default);

        let vision_config = hf_extra
            .get_value("vision_config")
            .and_then(|v| v.as_object());

        let image_size = vision_config
            .and_then(|vc| vc.get("image_size"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(metadata.default_image_size);

        let patch_size = vision_config
            .and_then(|vc| vc.get("patch_size"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(14);

        let spatial_merge_size = match metadata.tokens_per_image {
            TokensPerImage::PerImageGrid {
                spatial_merge_default,
            } => vision_config
                .and_then(|vc| vc.get("spatial_merge_size"))
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(spatial_merge_default),
            TokensPerImage::FromConfig => 1,
        };

        let mm_tokens_per_image = match metadata.tokens_per_image {
            TokensPerImage::FromConfig => hf_extra
                .get_value("mm_tokens_per_image")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or_else(|| (image_size / patch_size).pow(2)),
            TokensPerImage::PerImageGrid { .. } => {
                // Variable per-image — placeholder used for fallback only.
                let grid = image_size / patch_size;
                let merged = grid / spatial_merge_size as usize;
                merged * merged
            }
        };

        let (min_pixels, max_pixels) = match metadata.size_policy {
            SizePolicy::SmartResize {
                default_min_pixels,
                default_max_pixels,
                ..
            } => {
                let from_file = model_dir
                    .map(|dir| dir.join("preprocessor_config.json"))
                    .and_then(|p| std::fs::read_to_string(&p).ok())
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                    .and_then(|v| {
                        // Two schemas: Qwen2-VL / Qwen2.5-VL write top-level
                        // `min_pixels` / `max_pixels`; Qwen3-VL / Qwen3.5-VL
                        // moved them under `size.shortest_edge` /
                        // `size.longest_edge` (matching transformers'
                        // Qwen3VLImageProcessor + mlx-vlm). Prefer the
                        // top-level keys, then fall back to the `size`
                        // dict so the newer models smart-resize correctly
                        // (e.g. a 224×224 image bumps up to the 256×256 /
                        // 16×16-patch grid instead of staying 14×14).
                        let mn = v.get("min_pixels").and_then(|x| x.as_u64()).or_else(|| {
                            v.get("size")
                                .and_then(|s| s.get("shortest_edge"))
                                .and_then(|x| x.as_u64())
                        })?;
                        let mx = v.get("max_pixels").and_then(|x| x.as_u64()).or_else(|| {
                            v.get("size")
                                .and_then(|s| s.get("longest_edge"))
                                .and_then(|x| x.as_u64())
                        })?;
                        Some((mn as usize, mx as usize))
                    });
                from_file.unwrap_or((default_min_pixels, default_max_pixels))
            }
            // `max_pixels` carries the PATCH cap for this policy
            // (LocateAnything `in_token_limit`, counted in raw
            // patch_size² patches, not pixels). Read from
            // `preprocessor_config.json::in_token_limit`; fall back to
            // the declared default.
            SizePolicy::PatchCapCeilPad {
                default_max_patches,
                ..
            } => {
                let from_file = model_dir
                    .map(|dir| dir.join("preprocessor_config.json"))
                    .and_then(|p| std::fs::read_to_string(&p).ok())
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                    .and_then(|v| v.get("in_token_limit").and_then(|x| x.as_u64()))
                    .map(|v| v as usize);
                (0, from_file.unwrap_or(default_max_patches))
            }
            SizePolicy::FixedSquare => (0, 0),
        };

        Self {
            metadata,
            image_token_id,
            image_size,
            patch_size,
            spatial_merge_size,
            mm_tokens_per_image,
            min_pixels,
            max_pixels,
        }
    }

    /// Resolve `(target_h, target_w)` for a live image given its
    /// decoded dimensions. Picks fixed-square or smart-resize based
    /// on the metadata's policy.
    pub fn target_dims(&self, src_h: usize, src_w: usize) -> (usize, usize) {
        match self.metadata.size_policy {
            SizePolicy::FixedSquare => (self.image_size, self.image_size),
            SizePolicy::SmartResize { factor, .. } => scratchy_vision::preprocess::smart_resize(
                src_h,
                src_w,
                factor as usize,
                self.min_pixels,
                self.max_pixels,
            ),
            // mlx-vlm `image_processing_locateanything.py::rescale`:
            // downscale-only patch cap (`scale = sqrt(limit/patches)`,
            // dims truncated), then CEIL each dim to a multiple of
            // `pad_multiple` (= patch · merge = 28). `self.max_pixels`
            // carries the patch limit for this policy.
            SizePolicy::PatchCapCeilPad {
                patch,
                pad_multiple,
                ..
            } => {
                let p = (patch as usize).max(1);
                let patches = (src_h / p) * (src_w / p);
                let limit = self.max_pixels;
                let (mut nh, mut nw) = (src_h, src_w);
                if limit > 0 && patches > limit {
                    let scale = ((limit as f64) / (patches as f64)).sqrt();
                    nh = ((src_h as f64) * scale) as usize;
                    nw = ((src_w as f64) * scale) as usize;
                }
                let m = (pad_multiple as usize).max(1);
                (nh.max(1).div_ceil(m) * m, nw.max(1).div_ceil(m) * m)
            }
        }
    }

    /// Tokens this image contributes after the vision encoder. For
    /// fixed-token policies this is `mm_tokens_per_image`; for
    /// `PerImageGrid` it depends on the post-resize dims.
    pub fn tokens_for_image(&self, image_h: usize, image_w: usize) -> usize {
        match self.metadata.tokens_per_image {
            TokensPerImage::FromConfig => self.mm_tokens_per_image,
            TokensPerImage::PerImageGrid { .. } => {
                let f = self.patch_size * self.spatial_merge_size as usize;
                if f == 0 {
                    return self.mm_tokens_per_image;
                }
                (image_h / f) * (image_w / f)
            }
        }
    }

    /// Run the per-arch CPU preprocess fn for one image, given the
    /// decoded `DynamicImage`. Resolves target dims internally.
    pub fn preprocess(&self, img: &image::DynamicImage) -> ImageData {
        let (h, w) = self.target_dims(img.height() as usize, img.width() as usize);
        (self.metadata.preprocess)(img, h, w)
    }
}

/// Top-level resolver. Returns `None` for text-only models, or when
/// no MM arch in the inventory claims any of the HF arch strings.
pub fn resolve<E: HfExtra + ?Sized>(
    hf_arches: &[String],
    hf_extra: &E,
    model_dir: Option<&Path>,
) -> Option<Arc<ResolvedMmProcessor>> {
    let reg = scratchy_forward_compiler::resolve_mm_metadata(hf_arches)?;
    Some(Arc::new(ResolvedMmProcessor::resolve(
        &reg.mm_metadata,
        hf_extra,
        model_dir,
    )))
}

// SPDX-License-Identifier: Apache-2.0
//! CPU-side image preprocessing primitives shared by every multimodal arch.
//!
//! scratchy is a compiler for declarative arch specs — these functions are
//! the primitive ops a per-arch [`crate::MmMetadata`] points at. scratchy
//! itself names no arch; an arch crate composes a `MmMetadata` from these
//! primitives + size/token policies, and the inventory submission carries
//! it through to scratchy-serving-api.
//!
//! Two normalization conventions cover every VL/MM arch shipping today:
//!
//! - **`(pixel/255 − 0.5) / 0.5`** — symmetric ±1. Used by SigLIP, the
//!   Gemma3-MM vision tower. Implemented as [`preprocess_symmetric_unit`].
//! - **CLIP mean/std** — `mean=[0.48145466, 0.4578275, 0.40821073]`,
//!   `std=[0.26862954, 0.26130258, 0.27577711]`. Used by Qwen2-VL,
//!   Qwen2.5-VL, and the LLaVA family. Implemented as
//!   [`preprocess_clip_normalized`].
//!
//! Each takes `(img, target_h, target_w)` and returns a CHW-flat
//! `ImageData`. The `target_h, target_w` are resolved by the caller from
//! the `SizePolicy` declared in the arch's `MmMetadata` — fixed-square
//! arches pass `(image_size, image_size)`; smart-resize arches pass
//! whatever [`smart_resize`] returned for the live image dimensions.
//!
//! Image decoding helpers ([`decode_image`], [`decode_data_uri`],
//! [`base64_decode`]) are arch-agnostic and live here for the same
//! reason — scratchy-serving-api consumes them through scratchy-vision so it
//! never re-implements MM host-side mechanics.

use image::DynamicImage;
use image::imageops::FilterType;
use scratchy_core_common::multimodal::ImageData;

/// Decode image bytes (JPEG, PNG, WebP) into a `DynamicImage`.
pub fn decode_image(bytes: &[u8]) -> Result<DynamicImage, String> {
    image::load_from_memory(bytes).map_err(|e| format!("failed to decode image: {e}"))
}

/// Extract raw bytes from a `data:image/...;base64,...` URI.
pub fn decode_data_uri(uri: &str) -> Result<Vec<u8>, String> {
    let comma_pos = uri
        .find(',')
        .ok_or_else(|| "invalid data URI: no comma separator".to_string())?;
    let encoded = &uri[comma_pos + 1..];
    base64_decode(encoded).map_err(|e| format!("failed to decode base64 data URI: {e}"))
}

/// Simple base64 decoder (standard alphabet with optional padding).
pub fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    let clean: String = input.chars().filter(|c| !c.is_whitespace()).collect();
    let bytes = clean.as_bytes();

    let mut output = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;

    for &b in bytes {
        let val = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => continue,
            _ => return Err(format!("invalid base64 character: {}", b as char)),
        };
        buf = (buf << 6) | val as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }

    Ok(output)
}

/// Resize to `target_h x target_w` (bilinear) and normalize to symmetric
/// ±1 via `(pixel/255 − 0.5) / 0.5`. SigLIP / Gemma3-MM convention.
///
/// Returns `ImageData` with pixels in `[3, H, W]` CHW layout.
pub fn preprocess_symmetric_unit(
    img: &DynamicImage,
    target_h: usize,
    target_w: usize,
) -> ImageData {
    let resized = img.resize_exact(target_w as u32, target_h as u32, FilterType::Triangle);
    let rgb = resized.to_rgb8();

    let mut pixels = vec![0.0f32; 3 * target_h * target_w];
    for y in 0..target_h {
        for x in 0..target_w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                let normalized = (val - 0.5) / 0.5;
                pixels[c * target_h * target_w + y * target_w + x] = normalized;
            }
        }
    }

    ImageData {
        pixels,
        height: target_h,
        width: target_w,
    }
}

/// CLIP mean/std normalization constants. Frozen across every CLIP-class
/// vision tower (Qwen2-VL, Qwen2.5-VL, LLaVA, …).
pub const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
pub const CLIP_STD: [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];

/// Resize to `target_h x target_w` (bicubic — matches HF's
/// `Qwen2VLImageProcessor.resample = 3`) and normalize with
/// [`CLIP_MEAN`] / [`CLIP_STD`].
///
/// Returns `ImageData` with pixels in `[3, H, W]` CHW layout.
pub fn preprocess_clip_normalized(
    img: &DynamicImage,
    target_h: usize,
    target_w: usize,
) -> ImageData {
    let resized = img.resize_exact(target_w as u32, target_h as u32, FilterType::CatmullRom);
    let rgb = resized.to_rgb8();

    let mut pixels = vec![0.0f32; 3 * target_h * target_w];
    for y in 0..target_h {
        for x in 0..target_w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                let normalized = (val - CLIP_MEAN[c]) / CLIP_STD[c];
                pixels[c * target_h * target_w + y * target_w + x] = normalized;
            }
        }
    }

    ImageData {
        pixels,
        height: target_h,
        width: target_w,
    }
}

/// Round image dimensions to multiples of `factor`, then clamp total
/// pixel count to `[min_pixels, max_pixels]`. Returns `(new_h, new_w)`.
///
/// The `factor` is typically `patch_size · spatial_merge_size` (28 for
/// Qwen2-VL: patch=14, merge=2). Matches HF's `smart_resize` helper.
pub fn smart_resize(
    h: usize,
    w: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    let mut new_h = ((h.max(1) + factor / 2) / factor).max(1) * factor;
    let mut new_w = ((w.max(1) + factor / 2) / factor).max(1) * factor;

    if new_h * new_w < min_pixels {
        let scale = (min_pixels as f64 / (new_h * new_w) as f64).sqrt();
        new_h = ((new_h as f64 * scale / factor as f64).ceil() as usize) * factor;
        new_w = ((new_w as f64 * scale / factor as f64).ceil() as usize) * factor;
    }

    if new_h * new_w > max_pixels {
        let scale = (max_pixels as f64 / (new_h * new_w) as f64).sqrt();
        new_h = ((new_h as f64 * scale / factor as f64).floor() as usize) * factor;
        new_w = ((new_w as f64 * scale / factor as f64).floor() as usize) * factor;
        if new_h == 0 {
            new_h = factor;
        }
        if new_w == 0 {
            new_w = factor;
        }
    }

    (new_h, new_w)
}

/// Per-image variant of placeholder-token expansion. Each occurrence of
/// `image_token_id` in `token_ids` is replaced with `counts[k]` copies
/// (one per image in scan order). Returns the expanded ranges.
///
/// Required for arches whose vision encoder produces a variable
/// post-merger token count per image (Qwen2-VL: count depends on each
/// image's `smart_resize` output dims). Fixed-count arches (SigLIP-class)
/// pass a uniform `counts` slice.
///
/// If `counts` is shorter than the number of placeholder occurrences,
/// trailing extras stay unexpanded (length 1) — the caller is responsible
/// for `images.len() == counts.len()`.
pub fn expand_image_placeholders_per_image(
    token_ids: &mut Vec<u32>,
    image_token_id: u32,
    counts: &[usize],
) -> Vec<scratchy_core_common::multimodal::PlaceholderRange> {
    let mut ranges = Vec::new();
    let mut i = 0;
    let mut k = 0;
    while i < token_ids.len() {
        if token_ids[i] == image_token_id {
            if k >= counts.len() {
                i += 1;
                continue;
            }
            let n = counts[k];
            let offset = i;
            let extra = n.saturating_sub(1);
            for _ in 0..extra {
                token_ids.insert(i + 1, image_token_id);
            }
            ranges.push(scratchy_core_common::multimodal::PlaceholderRange { offset, length: n });
            i += n;
            k += 1;
        } else {
            i += 1;
        }
    }
    ranges
}

/// Generic placeholder expansion driven by [`crate::PlaceholderPolicy`].
/// Each `marker_token_id` occurrence in `token_ids` is replaced
/// according to the policy; returned ranges cover only the splice
/// positions (where the vision encoder's projected embeddings are
/// written).
///
/// - [`crate::PlaceholderPolicy::RepeatMarker`]: marker → N copies of
///   marker; splice covers all N positions.
/// - [`crate::PlaceholderPolicy::BoiSoftEoiWrap`]: marker → `[wrap,
///   marker, soft × N, eoi, wrap]`; splice covers only the N soft
///   positions in the middle.
///
/// `counts[k]` is the per-image splice count (variable for `Repeat`-
/// style arches with smart-resize per-image grid). For `BoiSoftEoiWrap`
/// the count is typically uniform (= `MmMetadata::mm_tokens_per_image`).
pub fn expand_placeholders_by_policy(
    token_ids: &mut Vec<u32>,
    marker_token_id: u32,
    counts: &[usize],
    policy: &crate::PlaceholderPolicy,
) -> Vec<scratchy_core_common::multimodal::PlaceholderRange> {
    match policy {
        crate::PlaceholderPolicy::RepeatMarker => {
            expand_image_placeholders_per_image(token_ids, marker_token_id, counts)
        }
        crate::PlaceholderPolicy::BoiSoftEoiWrap {
            soft_token_id,
            eoi_token_id,
            wrap_token_id,
        } => {
            let mut ranges = Vec::new();
            let mut i = 0;
            let mut k = 0;
            while i < token_ids.len() {
                if token_ids[i] == marker_token_id {
                    if k >= counts.len() {
                        i += 1;
                        continue;
                    }
                    let n = counts[k];
                    // Replace the single marker at index `i` with the
                    // wrapped sequence: [wrap, marker, soft × N, eoi, wrap].
                    // Total length = N + 4. Splice positions are the
                    // soft tokens (offset i+2..i+2+N).
                    let mut expansion: Vec<u32> = Vec::with_capacity(n + 4);
                    expansion.push(*wrap_token_id);
                    expansion.push(marker_token_id);
                    for _ in 0..n {
                        expansion.push(*soft_token_id);
                    }
                    expansion.push(*eoi_token_id);
                    expansion.push(*wrap_token_id);

                    token_ids.splice(i..i + 1, expansion);
                    let splice_offset = i + 2;
                    ranges.push(scratchy_core_common::multimodal::PlaceholderRange {
                        offset: splice_offset,
                        length: n,
                    });
                    i += n + 4;
                    k += 1;
                } else {
                    i += 1;
                }
            }
            ranges
        }
        crate::PlaceholderPolicy::BracketRepeat {
            start_token_id,
            end_token_id,
        } => {
            let mut ranges = Vec::new();
            let mut i = 0;
            let mut k = 0;
            while i < token_ids.len() {
                if token_ids[i] == marker_token_id {
                    if k >= counts.len() {
                        i += 1;
                        continue;
                    }
                    let n = counts[k];
                    // Replace the single marker at index `i` with
                    // `[start, marker × N, end]` (LocateAnything:
                    // `<img>` + `<IMG_CONTEXT>` × N + `</img>`). Total
                    // length = N + 2. Splice positions are the N
                    // marker tokens (offset i+1..i+1+N) — the same
                    // positions mlx-vlm's masked scatter writes.
                    let mut expansion: Vec<u32> = Vec::with_capacity(n + 2);
                    expansion.push(*start_token_id);
                    for _ in 0..n {
                        expansion.push(marker_token_id);
                    }
                    expansion.push(*end_token_id);

                    token_ids.splice(i..i + 1, expansion);
                    ranges.push(scratchy_core_common::multimodal::PlaceholderRange {
                        offset: i + 1,
                        length: n,
                    });
                    i += n + 2;
                    k += 1;
                } else {
                    i += 1;
                }
            }
            ranges
        }
    }
}

/// Replace every literal numbered image tag (`<image-1>`, `<image-23>`,
/// …) in a chat-template-rendered prompt with `marker` (the placeholder
/// token's text). Mirrors mlx-vlm `processing_locateanything.py`'s
/// `re.sub(r"<image-\d+>", …)` pre-tokenization step — required for
/// arches whose template renders numbered text instead of a marker
/// token (see [`crate::MmMetadata::numbered_image_tag_marker`]).
pub fn replace_numbered_image_tags(text: &str, marker: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<image-") {
        let tail = &rest[start + "<image-".len()..];
        let digits = tail.bytes().take_while(|b| b.is_ascii_digit()).count();
        if digits > 0 && tail.as_bytes().get(digits) == Some(&b'>') {
            out.push_str(&rest[..start]);
            out.push_str(marker);
            rest = &tail[digits + 1..];
        } else {
            // `<image-` without `digits>` — not a tag; keep scanning.
            out.push_str(&rest[..start + "<image-".len()]);
            rest = tail;
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_unit_white_is_one() {
        let mut img = image::RgbImage::new(4, 4);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([255, 255, 255]);
        }
        let dyn_img = DynamicImage::ImageRgb8(img);
        let result = preprocess_symmetric_unit(&dyn_img, 4, 4);
        for &val in &result.pixels {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn symmetric_unit_black_is_minus_one() {
        let img = image::RgbImage::new(4, 4);
        let dyn_img = DynamicImage::ImageRgb8(img);
        let result = preprocess_symmetric_unit(&dyn_img, 4, 4);
        for &val in &result.pixels {
            assert!((val + 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn clip_normalized_white_matches_hf_constants() {
        let mut img = image::RgbImage::new(4, 4);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([255, 255, 255]);
        }
        let dyn_img = DynamicImage::ImageRgb8(img);
        let result = preprocess_clip_normalized(&dyn_img, 4, 4);
        let expected_r = (1.0 - CLIP_MEAN[0]) / CLIP_STD[0];
        assert!((result.pixels[0] - expected_r).abs() < 1e-3);
    }

    #[test]
    fn smart_resize_rounds_to_factor() {
        let (h, w) = smart_resize(100, 200, 28, 256 * 28 * 28, 1280 * 28 * 28);
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn smart_resize_clamps_min() {
        let (h, w) = smart_resize(10, 10, 28, 256 * 28 * 28, 1280 * 28 * 28);
        assert!(h * w >= 256 * 28 * 28);
    }

    #[test]
    fn smart_resize_clamps_max() {
        let (h, w) = smart_resize(5000, 5000, 28, 256 * 28 * 28, 1280 * 28 * 28);
        assert!(h * w <= 1280 * 28 * 28);
    }

    #[test]
    fn decode_data_uri_strips_prefix() {
        let bytes = decode_data_uri("data:text/plain;base64,SGVsbG8=").unwrap();
        assert_eq!(bytes, b"Hello");
    }

    #[test]
    fn placeholder_expansion_per_image() {
        let mut tokens = vec![1, 99, 2, 99, 3];
        let ranges = expand_image_placeholders_per_image(&mut tokens, 99, &[3, 2]);
        assert_eq!(tokens, vec![1, 99, 99, 99, 2, 99, 99, 3]);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].offset, 1);
        assert_eq!(ranges[0].length, 3);
        assert_eq!(ranges[1].offset, 5);
        assert_eq!(ranges[1].length, 2);
    }

    #[test]
    fn bracket_repeat_expansion() {
        // LocateAnything: marker → [start, marker × N, end], splice at
        // the N marker positions (mirrors mlx-vlm `<img>` +
        // `<IMG_CONTEXT>` × N + `</img>`).
        let policy = crate::PlaceholderPolicy::BracketRepeat {
            start_token_id: 7,
            end_token_id: 8,
        };
        let mut tokens = vec![1, 99, 2, 99, 3];
        let ranges = expand_placeholders_by_policy(&mut tokens, 99, &[3, 2], &policy);
        assert_eq!(tokens, vec![1, 7, 99, 99, 99, 8, 2, 7, 99, 99, 8, 3]);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].offset, 2);
        assert_eq!(ranges[0].length, 3);
        assert_eq!(ranges[1].offset, 8);
        assert_eq!(ranges[1].length, 2);
    }

    #[test]
    fn numbered_image_tags_replaced() {
        assert_eq!(
            replace_numbered_image_tags("a<image-1>b<image-23>c", "<M>"),
            "a<M>b<M>c"
        );
        // Non-tags pass through untouched.
        assert_eq!(
            replace_numbered_image_tags("<image->x<image-1y><imagez>", "<M>"),
            "<image->x<image-1y><imagez>"
        );
        assert_eq!(replace_numbered_image_tags("no tags", "<M>"), "no tags");
        // Tag at string boundaries.
        assert_eq!(replace_numbered_image_tags("<image-9>", "<M>"), "<M>");
    }
}

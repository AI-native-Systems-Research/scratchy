include!(concat!(env!("OUT_DIR"), "/qwen3_5_vl.rs"));

/// Qwen3.5-VL CPU preprocessing. patch 16 × spatial_merge 2 → smart-resize
/// factor 32, **symmetric ±1 normalization** via `preprocess_symmetric_unit`
/// (`image_mean = image_std = 0.5` → `(px/255 − 0.5)/0.5`, matches the HF
/// `Qwen3VLImageProcessor` / mlx-vlm; CLIP mean/std is WRONG for this arch
/// and corrupts the ViT input — verified pixel_values cosine 0.99997 vs
/// mlx-vlm with this fn, vs 0.9946 with CLIP), `<|image_pad|>` placeholder,
/// per-image grid. `default_min_pixels` / `default_max_pixels` mirror the
/// processor's `size.shortest_edge` / `size.longest_edge` (65536 /
/// 16777216) so a small image (e.g. 224×224) smart-resizes UP to the
/// 256×256 / 16×16-patch grid the model expects.
pub const PROCESSOR: scratchy_vision::MmMetadata = scratchy_vision::MmMetadata {
    hf_token_id_key: "image_token_id",
    hf_token_id_default: 151655,
    size_policy: scratchy_vision::SizePolicy::SmartResize {
        factor: 32,
        default_min_pixels: 65536,
        default_max_pixels: 16_777_216,
    },
    tokens_per_image: scratchy_vision::TokensPerImage::PerImageGrid {
        spatial_merge_default: 2,
    },
    preprocess: scratchy_vision::preprocess::preprocess_symmetric_unit,
    default_image_size: 384,
    chat_template_image_part_type: "image",
    placeholder_policy: scratchy_vision::PlaceholderPolicy::RepeatMarker,
    mrope_positions: true,
    numbered_image_tag_marker: None,
};

include!(concat!(env!("OUT_DIR"), "/qwen2_5_vl.rs"));

/// Qwen2.5-VL CPU preprocessing: same family conventions as Qwen2-VL —
/// `<|image_pad|>` placeholder, smart-resize at factor 28, CLIP
/// normalization. Per-image grid token count.
pub const PROCESSOR: scratchy_vision::MmMetadata = scratchy_vision::MmMetadata {
    hf_token_id_key: "image_token_id",
    hf_token_id_default: 151655,
    size_policy: scratchy_vision::SizePolicy::SmartResize {
        factor: 28,
        default_min_pixels: 3136,
        default_max_pixels: 12_845_056,
    },
    tokens_per_image: scratchy_vision::TokensPerImage::PerImageGrid {
        spatial_merge_default: 2,
    },
    preprocess: scratchy_vision::preprocess::preprocess_clip_normalized,
    default_image_size: 392,
    chat_template_image_part_type: "image",
    placeholder_policy: scratchy_vision::PlaceholderPolicy::RepeatMarker,
    mrope_positions: true,
    numbered_image_tag_marker: None,
};

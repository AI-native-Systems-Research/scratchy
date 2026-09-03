include!(concat!(env!("OUT_DIR"), "/qwen2_vl.rs"));

/// Per-arch CPU preprocessing declaration baked into every emitted
/// `ScratchyMmRegistration` row by the `#[vision_forward(processor = ...)]`
/// arg. Qwen2-VL uses smart-resize (post-resize dims vary per image)
/// with CLIP-mean/std normalization; placeholder token id is the
/// `<|image_pad|>` token id from `hf_config.image_token_id` (defaults
/// to 151655). Tokens-per-image is per-image grid divided by spatial
/// merge size (= 2 for every Qwen2-VL variant).
pub const PROCESSOR: scratchy_vision::MmMetadata = scratchy_vision::MmMetadata {
    hf_token_id_key: "image_token_id",
    hf_token_id_default: 151655,
    size_policy: scratchy_vision::SizePolicy::SmartResize {
        // patch_size(14) · spatial_merge_size(2)
        factor: 28,
        // HF Qwen2VLImageProcessor defaults; per-checkpoint
        // `preprocessor_config.json` overrides at init time.
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

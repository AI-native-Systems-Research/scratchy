include!(concat!(env!("OUT_DIR"), "/gemma3_mm.rs"));

/// Gemma3-MM CPU preprocessing: SigLIP encoder + 4×4 avg-pool projector.
/// HF chat template emits `<start_of_image>` (= `boi_token_index`,
/// 255999) per image — NOT the soft-token id (262144) carried under
/// `image_token_index`. Tokens-per-image is fixed at 256 (post-pool);
/// `hf_config.mm_tokens_per_image` ships this value, so the
/// `FromConfig` policy reads it directly. Pixels are resized to a
/// fixed 896² square and normalized to symmetric ±1.
pub const PROCESSOR: scratchy_vision::MmMetadata = scratchy_vision::MmMetadata {
    hf_token_id_key: "boi_token_index",
    hf_token_id_default: 255999,
    size_policy: scratchy_vision::SizePolicy::FixedSquare,
    tokens_per_image: scratchy_vision::TokensPerImage::FromConfig,
    preprocess: scratchy_vision::preprocess::preprocess_symmetric_unit,
    default_image_size: 896,
    chat_template_image_part_type: "image",
    // HF Gemma3 processor expands `<start_of_image>` to
    // `\n\n<start_of_image><image_soft_token>×256<end_of_image>\n\n`.
    // The model is trained on this exact bracketed structure — a flat
    // RepeatMarker of boi gives garbled output.
    // Token IDs: boi=255999 (set as hf_token_id_default above),
    // soft=262144, eoi=256000, wrap=108 (Gemma tokenizer's `\n\n`).
    placeholder_policy: scratchy_vision::PlaceholderPolicy::BoiSoftEoiWrap {
        soft_token_id: 262144,
        eoi_token_id: 256000,
        wrap_token_id: 108,
    },
    // Gemma3 text decoder uses standard 1D RoPE — no MRoPE override.
    mrope_positions: false,
    numbered_image_tag_marker: None,
};

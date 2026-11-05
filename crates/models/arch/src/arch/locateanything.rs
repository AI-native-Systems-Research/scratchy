include!(concat!(env!("OUT_DIR"), "/locateanything.rs"));

/// LocateAnything CPU preprocessing — mirrors mlx-vlm
/// `image_processing_locateanything.py` + `processing_locateanything.py`:
/// patch-cap resize (`in_token_limit` 25600 raw 14×14 patches, downscale
/// only) then ceil-pad both dims to a multiple of 28 (= patch · merge);
/// symmetric ±1 normalization (`image_mean = image_std = 0.5`);
/// `<IMG_CONTEXT>` (151665) placeholder bracketed by `<img>`/`</img>`
/// (151666/151667) with `(h/28)·(w/28)` context tokens per image; the
/// chat template renders literal `<image-N>` text that serve replaces
/// with the marker pre-tokenization (`numbered_image_tag_marker`).
/// Plain 1D positions — NO mrope.
pub const PROCESSOR: scratchy_vision::MmMetadata = scratchy_vision::MmMetadata {
    hf_token_id_key: "image_token_index",
    hf_token_id_default: 151665,
    size_policy: scratchy_vision::SizePolicy::PatchCapCeilPad {
        patch: 14,
        pad_multiple: 28,
        default_max_patches: 25600,
    },
    tokens_per_image: scratchy_vision::TokensPerImage::PerImageGrid {
        spatial_merge_default: 2,
    },
    preprocess: scratchy_vision::preprocess::preprocess_symmetric_unit,
    default_image_size: 224,
    chat_template_image_part_type: "image",
    placeholder_policy: scratchy_vision::PlaceholderPolicy::BracketRepeat {
        start_token_id: 151666,
        end_token_id: 151667,
    },
    mrope_positions: false,
    numbered_image_tag_marker: Some("<IMG_CONTEXT>"),
};

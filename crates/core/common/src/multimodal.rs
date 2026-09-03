// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Multimodal data types for vision-language model support.
//!
//! These types carry image data and placeholder information from the API layer
//! through the engine core to the model workers. Images are preprocessed on CPU
//! before being converted to backend-specific tensors in the worker.

/// A range within the token sequence where image embeddings should be inserted.
///
/// During tokenization, image placeholder tokens are expanded to `length` copies.
/// The vision encoder's output embeddings replace tokens at `offset..offset+length`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlaceholderRange {
    /// Start position in the token ID sequence.
    pub offset: usize,
    /// Number of tokens to replace with image embeddings.
    pub length: usize,
}

/// Preprocessed image data ready for tensor conversion.
///
/// Pixels are stored as a flat `Vec<f32>` in `[C, H, W]` (channels-first) layout,
/// normalized to the range expected by the vision encoder (e.g., `(pixel/255 - 0.5) / 0.5`
/// for SigLIP).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImageData {
    /// Pixel values in `[3, height, width]` CHW layout, normalized.
    pub pixels: Vec<f32>,
    /// Image height after preprocessing.
    pub height: usize,
    /// Image width after preprocessing.
    pub width: usize,
}

/// Multimodal data attached to an engine core request.
///
/// Contains preprocessed images and their corresponding placeholder ranges
/// within the token sequence. Workers convert `ImageData` to backend tensors
/// and pass them through the vision encoder during the prefill step.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct MultimodalData {
    /// Preprocessed images (one per image in the request).
    pub images: Vec<ImageData>,
    /// Placeholder ranges indicating where each image's embeddings go in the
    /// token sequence. Length must equal `images.len()`.
    pub image_placeholders: Vec<PlaceholderRange>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_placeholder_range() {
        let range = PlaceholderRange {
            offset: 10,
            length: 256,
        };
        assert_eq!(range.offset, 10);
        assert_eq!(range.length, 256);
    }

    #[test]
    fn test_image_data() {
        let img = ImageData {
            pixels: vec![0.5; 3 * 224 * 224],
            height: 224,
            width: 224,
        };
        assert_eq!(img.pixels.len(), 3 * 224 * 224);
        assert_eq!(img.height, 224);
        assert_eq!(img.width, 224);
    }

    #[test]
    fn test_multimodal_data_default() {
        let mm = MultimodalData::default();
        assert!(mm.images.is_empty());
        assert!(mm.image_placeholders.is_empty());
    }

    #[test]
    fn test_multimodal_data_with_images() {
        let mm = MultimodalData {
            images: vec![
                ImageData {
                    pixels: vec![0.0; 3 * 224 * 224],
                    height: 224,
                    width: 224,
                },
                ImageData {
                    pixels: vec![1.0; 3 * 384 * 384],
                    height: 384,
                    width: 384,
                },
            ],
            image_placeholders: vec![
                PlaceholderRange {
                    offset: 5,
                    length: 256,
                },
                PlaceholderRange {
                    offset: 300,
                    length: 256,
                },
            ],
        };
        assert_eq!(mm.images.len(), 2);
        assert_eq!(mm.image_placeholders.len(), 2);
        assert_eq!(mm.image_placeholders[0].offset, 5);
        assert_eq!(mm.image_placeholders[1].offset, 300);
    }
}

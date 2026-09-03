// SPDX-License-Identifier: Apache-2.0
//! Device-runtime multimodal forward surface: the [`PixelInput`]
//! interior shape, the [`EmbedPatch`] embed-splice descriptor, and the
//! [`MultimodalForward`] trait the executor calls at request time.
//!
//! These names are referenced by the cross-arch MM registry
//! (`ScratchyMmRegistration` / `try_load_mm` in [`crate::arch_registry`]),
//! so they live in the cfg-free compiler alongside it. The trait stays
//! backend-neutral by taking the live device as a
//! [`scratchy_tensors::ForwardDeviceHandle`] (the per-arch
//! `VisionWrapper` impl recovers the concrete `&mut GpuDevice` from it),
//! the same neutralization the text-side `ScratchyWeights` trait applies.

/// One image's preprocessed pixel input, post-CPU-normalization.
/// Pixels are uploaded to GPU by `vision_forward`; CPU-side
/// `MultimodalData` (`scratchy-core-common::ImageData`) is the boundary
/// type the engine plumbs in, this is the interior shape the
/// vision encoder consumes.
#[derive(Debug)]
pub struct PixelInput<'a> {
    /// Flat normalized pixels in CHW layout, length =
    /// `3 * height * width`. Borrowed; `vision_forward` uploads
    /// to GPU and the borrow ends when it returns.
    pub pixels: &'a [f32],
    pub height: u32,
    pub width: u32,
}

/// One projected image's embedding location in the token
/// sequence, plus a slice of the encoder output that occupies
/// it. The slice is `[token_offset .. token_offset + length]`
/// of `vision_forward`'s returned `OwnedTensor`. Mirrors
/// `scratchy-core-common::PlaceholderRange` but in token-space (post-
/// expansion); the worker's splice consumes these directly.
///
/// `grid_t` / `grid_h_merged` / `grid_w_merged` carry the
/// per-image grid dimensions (post spatial-merge) so the
/// executor can build `[3, n_tokens]` MRoPE positions for image-
/// bearing batches on Qwen2-VL-class arches: image tokens
/// scan in `(t, h, w)` row-major order with each row of the
/// positions tensor holding the corresponding axis's coordinate.
/// Caller (the worker) initializes these to zero when
/// constructing the input `placeholders`; `vision_forward` fills
/// them on the returned `Vec<EmbedPatch>` by zipping its
/// per-image `grid_thw` with `placeholders`. Text-only arches
/// that never call `vision_forward` leave them at zero —
/// `length == grid_t * grid_h_merged * grid_w_merged` is the
/// post-merger invariant for filled patches.
#[derive(Debug, Clone, Default)]
pub struct EmbedPatch {
    /// Position in the input-id sequence where this image's
    /// projected embeddings start.
    pub token_offset: u32,
    /// Number of token slots this image occupies (= number of
    /// rows in the corresponding slice of the projected
    /// `OwnedTensor`).
    pub length: u32,
    /// Temporal-axis grid size (1 for still images on Qwen2-VL,
    /// >1 for video frames).
    pub grid_t: u32,
    /// Height-axis grid size after spatial merge (= raw `grid_h
    /// / spatial_merge_size`).
    pub grid_h_merged: u32,
    /// Width-axis grid size after spatial merge (= raw `grid_w
    /// / spatial_merge_size`).
    pub grid_w_merged: u32,
}

/// Sibling trait to `ScratchyWeights`. Implemented ONLY by
/// arches with a vision component — the qwen2 carrier-fn's
/// emitted `Weights` does NOT implement it; only the Qwen2-VL
/// variant's wrapper does. No `unimplemented!` defaults: arches
/// that don't carry a vision encoder simply don't implement
/// the trait, and their inventory rows don't surface here.
#[cfg(feature = "vision")]
pub trait MultimodalForward: Send + Sync {
    /// Encode one or more pixel batches and project into the
    /// language model's hidden space.
    ///
    /// Returns:
    /// - An `OwnedTensor` `[total_mm_tokens, hidden]` — the
    ///   stacked projected embeddings for every image, in the
    ///   order they appear in the input batch.
    /// - A `Vec<EmbedPatch>` of the same length as `pixel_batches`
    ///   recording where each image's slice lands in the token
    ///   sequence. The worker's splice consumes these to D2D-
    ///   copy each slice into the corresponding `Embed` tile rows.
    ///
    /// # Safety
    /// `device` must be the live CUDA device the encoder
    /// kernels launch on; the caller must keep `pixel_batches`
    /// alive for the duration of the call (the host-side
    /// borrow ends before any returned GPU memory is read).
    unsafe fn vision_forward(
        &self,
        pixel_batches: &[PixelInput<'_>],
        placeholders: &[EmbedPatch],
        device: ::scratchy_tensors::ForwardDeviceHandle<'_>,
    ) -> (::scratchy_tensors::OwnedTensor, Vec<EmbedPatch>);

    /// Pure-CPU companion to [`Self::vision_forward`]. Returns one
    /// `(grid_t, grid_h_merged, grid_w_merged)` tuple per input image
    /// — the same metadata that `vision_forward` would fill on each
    /// returned `EmbedPatch`, without launching any GPU work.
    fn embed_patch_grids(&self, pixel_batches: &[PixelInput<'_>]) -> Vec<(u32, u32, u32)>;

    /// Per-arch CPU-side preprocessing metadata. Lets the executor
    /// read declarative flags (e.g. `mrope_positions`) the
    /// per-arch `pub const PROCESSOR: scratchy_vision::MmMetadata`
    /// declared. The macro-emitted `VisionWrapper<W>` impl returns
    /// the per-variant baked const.
    fn mm_metadata(&self) -> &'static scratchy_vision::MmMetadata;
}

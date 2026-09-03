// SPDX-License-Identifier: Apache-2.0
//! Device-runtime multimodal forward surface re-export.
//!
//! [`PixelInput`] / [`MultimodalForward`] are now backend-neutral types in the
//! compiler (the trait takes `scratchy_tensors::ForwardDeviceHandle` instead of
//! `&mut GpuDevice`), so the cross-arch MM registry can name them without
//! depending on a target crate. They're re-exported here so
//! `scratchy_target_cuda::{PixelInput, MultimodalForward}` and the lib.rs
//! re-export keep resolving.

pub use scratchy_forward_compiler::{MultimodalForward, PixelInput};

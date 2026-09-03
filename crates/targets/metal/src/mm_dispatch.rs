// SPDX-License-Identifier: Apache-2.0
//! Device-runtime multimodal forward surface re-export.
//!
//! [`PixelInput`] / [`MultimodalForward`] are backend-neutral types in the
//! compiler (the trait takes `scratchy_tensors::ForwardDeviceHandle` instead of
//! `&mut GpuDevice`), so the cross-arch MM registry names them without a target
//! dep. Re-exported here so `scratchy_target_metal::{PixelInput,
//! MultimodalForward}` resolve for the relocated `vision_arch` glue and the
//! macro emissions under the flipped `__gpu` alias.
pub use scratchy_forward_compiler::{MultimodalForward, PixelInput};

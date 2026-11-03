// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Loading precompiled metallibs.

use dispatch2::DispatchData;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary};

use crate::tape::lowered::BakedMetallib;

pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Wrap a static byte slice as a `dispatch_data_t` and load it as a Metal
/// library. The bytes typically come from `include_bytes!` so we keep the
/// destructor as no-op (default behavior of `DispatchData::from`'s
/// implementation copies into a managed buffer).
pub fn load_library_from_bytes(device: &Device, bytes: &'static [u8]) -> Result<Library, String> {
    let data = DispatchData::from_static_bytes(bytes);
    device
        .newLibraryWithData_error(&data)
        .map_err(|e| format!("newLibraryWithData failed: {:?}", e))
}

/// Load a baked metallib as a Metal library, decompressed into a buffer the library's
/// `dispatch_data_t` owns and frees.
pub fn load_baked_library(device: &Device, metallib: &BakedMetallib) -> Result<Library, String> {
    let data = DispatchData::from_boxed(metallib.decompress()?);
    device
        .newLibraryWithData_error(&data)
        .map_err(|e| format!("newLibraryWithData failed: {:?}", e))
}

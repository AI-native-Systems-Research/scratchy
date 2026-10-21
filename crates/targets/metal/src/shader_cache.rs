// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Loading precompiled metallibs, and building their pipelines one at a time.

use std::sync::{Mutex, MutexGuard, PoisonError};

use dispatch2::DispatchData;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary};

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

/// This thread's turn to build a pipeline, which every pipeline build in the crate takes. Apple's
/// GPU compiler corrupts its own heap when two threads of a process build pipelines at once, each
/// through its own compiler and descriptors, so sharing nothing of ours: it traps in malloc under
/// `ReflectionKey::initGPUCompilerSPIOptionsHash` (#288 on an M4 Max, #507 on an M5 Max, both on
/// macOS 26). A pipeline is built once and then reused, so a build waits only while another
/// thread builds.
pub(crate) fn pipeline_build_turn() -> MutexGuard<'static, ()> {
    static BUILDS: Mutex<()> = Mutex::new(());
    BUILDS.lock().unwrap_or_else(PoisonError::into_inner)
}

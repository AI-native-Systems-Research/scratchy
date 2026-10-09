// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal device detection and management.

use crate::tape::ids::GpuCores;
use crate::targets::MetalTargetProfile;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::{CFNumber, CFRetained, CFString, CFType};
use objc2_metal::{
    MTL4CompilerDescriptor, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice,
};
use std::ffi::c_void;
use std::ptr::NonNull;

pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

/// Wrapper around Metal device with target profile
#[derive(Clone)]
pub struct MetalDevice {
    pub device: Device,
    pub profile: MetalTargetProfile,
    pub queue: CommandQueue,
}

impl MetalDevice {
    pub fn new(device: Device, profile: MetalTargetProfile) -> Self {
        let queue = device
            .newCommandQueue()
            .expect("newCommandQueue returned nil");
        Self {
            device,
            profile,
            queue,
        }
    }
}

/// The target of `device`'s architecture (`MTLDevice.architecture.name`); `None` for a GPU that is
/// not Apple silicon.
pub(crate) fn known_profile(device: &ProtocolObject<dyn MTLDevice>) -> Option<MetalTargetProfile> {
    MetalTargetProfile::of_architecture(&device.architecture().name().to_string())
}

/// Whether the host has a Metal device: the `#[forward]` proc-macro's probe. The bake lowers a tape
/// for every target, so it reads nothing off the device, and does NOT gate on Metal 4 — the models
/// must build on ANY Metal host, including non-Metal-4 CI build hosts (GitHub's "Apple Paravirtual
/// device"). Runtime device acquisition uses [`detect_device`], which DOES gate on Metal 4.
pub fn metal_device_present() -> bool {
    MTLCreateSystemDefaultDevice().is_some()
}

/// Detect the current Metal device for RUNTIME use.
///
/// Returns `None` when there is no Metal device, the device does not support
/// Metal 4, or its architecture is not Apple silicon. This backend builds every compute pipeline through an `MTL4Compiler`
/// (`SpecializedPipelineCache` → `newCompilerWithDescriptor`), so a non-Metal-4
/// device — e.g. GitHub's hosted "Apple Paravirtual device", which is Metal 3
/// only — is unusable here. Reporting it as absent lets the worker and the GPU
/// test suite (which guard on `detect_device()`) skip cleanly instead of
/// panicking at the first pipeline build.
pub fn detect_device() -> Option<MetalDevice> {
    let device = MTLCreateSystemDefaultDevice()?;
    if device
        .newCompilerWithDescriptor_error(&MTL4CompilerDescriptor::new())
        .is_err()
    {
        eprintln!(
            "Metal device '{}' does not support Metal 4 (no MTL4 compiler); \
             treating as unavailable",
            device.name()
        );
        return None;
    }
    let Some(profile) = known_profile(&device) else {
        eprintln!(
            "Metal device '{}' (architecture '{}') is not Apple silicon; treating as unavailable",
            device.name(),
            device.architecture().name()
        );
        return None;
    };
    Some(MetalDevice::new(device, profile))
}

/// The GPU cores of `device`: the `gpu-core-count` property of its IO-registry
/// entry. Metal has no API for it and the chip's name does not determine it
/// (an M1 Max has 24 or 32 cores, an M3 Max 30 or 40), so no
/// [`MetalTargetProfile`] carries it. `None` if the entry lacks it.
pub fn gpu_cores(device: &ProtocolObject<dyn MTLDevice>) -> Option<GpuCores> {
    #[link(name = "IOKit", kind = "framework")]
    unsafe extern "C" {
        fn IORegistryEntryIDMatching(entry_id: u64) -> *mut c_void;
        fn IOServiceGetMatchingService(main_port: u32, matching: *mut c_void) -> u32;
        fn IORegistryEntryCreateCFProperty(
            entry: u32,
            key: &CFString,
            allocator: *const c_void,
            options: u32,
        ) -> Option<NonNull<CFType>>;
        fn IOObjectRelease(object: u32) -> i32;
    }
    let key = CFString::from_static_str("gpu-core-count");
    // SAFETY: `IOServiceGetMatchingService` consumes the matching dictionary
    // and returns a service we own (0 if none), released once read; the
    // property comes back retained, and `CFRetained` releases it.
    let property = unsafe {
        let service =
            IOServiceGetMatchingService(0, IORegistryEntryIDMatching(device.registryID()));
        if service == 0 {
            return None;
        }
        let property = IORegistryEntryCreateCFProperty(service, &key, std::ptr::null(), 0);
        IOObjectRelease(service);
        CFRetained::from_raw(property?)
    };
    let cores = property.downcast_ref::<CFNumber>()?.as_i32()?;
    u32::try_from(cores).ok().filter(|&n| n > 0).map(GpuCores)
}

/// Whether the default Metal device can create an `MTL4Compiler` (a real
/// Metal-4 GPU is present). Equivalent to `detect_device().is_some()` without
/// building the profile; kept for call sites that only need the boolean.
pub fn metal4_available() -> bool {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        return false;
    };
    device
        .newCompilerWithDescriptor_error(&MTL4CompilerDescriptor::new())
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_detection() {
        if let Some(device) = detect_device() {
            println!("Detected device: {}", device.device.name());
            println!("Architecture: {}", device.device.architecture().name());
            println!("Profile: {:?}", device.profile);
            let cores = gpu_cores(&device.device);
            println!("GPU cores: {cores:?}");
            assert!(cores.is_some_and(|n| n.get() > 0));
        }
    }
}

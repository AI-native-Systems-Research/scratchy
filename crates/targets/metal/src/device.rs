// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal device detection and management.

use crate::targets::MetalTargetProfile;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4CompilerDescriptor, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice,
};

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

/// Cost/target profile for a Metal device, chosen by chip name.
///
/// Each chip routes to the profile whose embedded cost CSV was swept on that
/// chip. Costs are analytical (roofline from the profile's bandwidth /
/// TFLOPS figures) — there is no empirical cost table.
fn profile_for_device(device: &Device) -> MetalTargetProfile {
    let name = device.name().to_string();
    if name.contains("M1") {
        if name.contains("Max") {
            crate::targets::M1_MAX
        } else {
            crate::targets::M1_8CORE
        }
    } else if name.contains("M2") {
        crate::targets::M2_10CORE
    } else if name.contains("M3") {
        crate::targets::M3_10CORE
    } else if name.contains("M4") {
        crate::targets::M4_10CORE
    } else if name.contains("M5") {
        crate::targets::M5_10CORE
    } else {
        // GitHub's hosted "Apple Paravirtual device" legitimately has no cost
        // profile (it can't run the kernels anyway — see `detect_device`), so
        // don't spam the build/test log for it; only warn for genuinely
        // unexpected hardware.
        if !name.contains("Paravirtual") {
            eprintln!("Warning: Unknown Metal device '{name}', defaulting to M1 profile");
        }
        crate::targets::M1_8CORE
    }
}

/// COMPILE-TIME cost profile for the default Metal device, chosen by chip name.
///
/// Used by the `#[forward]` proc-macro to pick a cost model at build time.
/// Deliberately does NOT gate on Metal 4: profile selection only needs the
/// device name, and the models must build on ANY Metal host — including
/// non-Metal-4 CI build hosts (GitHub's "Apple Paravirtual device"). Runtime
/// device acquisition uses [`detect_device`], which DOES gate on Metal 4.
pub fn detect_metal_profile() -> Option<MetalTargetProfile> {
    let device = MTLCreateSystemDefaultDevice()?;
    Some(profile_for_device(&device))
}

/// Detect the current Metal device for RUNTIME use.
///
/// Returns `None` when there is no Metal device OR the device does not support
/// Metal 4. This backend builds every compute pipeline through an `MTL4Compiler`
/// (`SpecializedPipelineCache` → `newCompilerWithDescriptor`), so a non-Metal-4
/// device — e.g. GitHub's hosted "Apple Paravirtual device", which is Metal 3
/// only — is unusable here. Reporting it as absent lets the worker and the GPU
/// test suite (which guard on `detect_device()`) skip cleanly instead of
/// panicking at the first pipeline build. Compile-time profile selection uses
/// [`detect_metal_profile`], which does NOT gate on Metal 4.
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
    let profile = profile_for_device(&device);
    Some(MetalDevice::new(device, profile))
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
            println!("Profile: {:?}", device.profile.generation);
            assert!(device.profile.gpu_cores > 0);
        }
    }
}

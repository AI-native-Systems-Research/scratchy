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

/// Cost/target profile for a Metal device, chosen by chip name.
///
/// Each chip routes to the profile whose embedded cost CSV was swept on that
/// chip. Costs are analytical (roofline from the profile's bandwidth /
/// TFLOPS figures) — there is no empirical cost table.
pub(crate) fn profile_for_device(device: &Device) -> MetalTargetProfile {
    known_profile(device).unwrap_or_else(|| {
        // GitHub's hosted "Apple Paravirtual device" legitimately has no cost
        // profile (it can't run the kernels anyway — see `detect_device`), so
        // don't spam the build/test log for it; only warn for genuinely
        // unexpected hardware.
        let name = device.name().to_string();
        if !name.contains("Paravirtual") {
            eprintln!("Warning: Unknown Metal device '{name}', defaulting to M1 profile");
        }
        crate::targets::M1_8CORE
    })
}

/// The profile of the chip `device`'s name names; `None` for a device no profile was swept on.
pub(crate) fn known_profile(device: &ProtocolObject<dyn MTLDevice>) -> Option<MetalTargetProfile> {
    let name = device.name().to_string();
    Some(if name.contains("M1") {
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
        return None;
    })
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

/// The most memory the GPU driver wires at once, the whole system's GPU buffers together: the
/// `iogpu.wired_limit_mb` sysctl where set, else macOS's default of it — 2/3 of physical memory
/// up to 36 GiB, 3/4 above.
///
/// `recommendedMaxWorkingSetSize` reports the limit where the sysctl sets it, but overstates the
/// default: 28.08 GiB on a 36 GiB M3 Pro, whose command buffers fault out of memory once the
/// system's wired memory nears 24 GiB (and run at a 24 GiB sysctl, where it reports 24). `None`
/// if the kernel answers neither sysctl.
pub fn gpu_wired_limit_bytes() -> Option<u64> {
    fn sysctl_u64(name: &std::ffi::CStr) -> Option<u64> {
        let mut value = 0u64;
        let mut len = std::mem::size_of::<u64>();
        // SAFETY: `value` is a `u64` of `len` bytes; a sysctl narrower than that writes fewer and
        // says so in `len`, which zero-initialised `value` covers on little-endian Apple silicon.
        let status = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                (&raw mut value).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (status == 0).then_some(value)
    }
    match sysctl_u64(c"iogpu.wired_limit_mb") {
        Some(mb) if mb > 0 => Some(mb << 20),
        _ => sysctl_u64(c"hw.memsize").map(default_gpu_wired_limit),
    }
}

/// macOS's GPU wired limit when `iogpu.wired_limit_mb` is unset: 2/3 of `memsize` up to 36 GiB,
/// 3/4 above.
fn default_gpu_wired_limit(memsize: u64) -> u64 {
    match memsize <= 36 << 30 {
        true => memsize / 3 * 2,
        false => memsize / 4 * 3,
    }
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
    fn the_default_gpu_wired_limit_is_two_thirds_up_to_36_gib_and_three_quarters_above() {
        assert_eq!(default_gpu_wired_limit(36 << 30), 24 << 30);
        assert_eq!(default_gpu_wired_limit(24 << 30), 16 << 30);
        assert_eq!(default_gpu_wired_limit(64 << 30), 48 << 30);
        assert_eq!(default_gpu_wired_limit(128 << 30), 96 << 30);
    }

    #[test]
    fn the_gpu_wired_limit_is_read_from_the_kernel() {
        let limit = gpu_wired_limit_bytes().expect("iogpu.wired_limit_mb or hw.memsize");
        assert!(limit > 0);
    }

    #[test]
    fn test_device_detection() {
        if let Some(device) = detect_device() {
            println!("Detected device: {}", device.device.name());
            println!("Profile: {:?}", device.profile.generation);
            let cores = gpu_cores(&device.device);
            println!("GPU cores: {cores:?}");
            assert!(cores.is_some_and(|n| n.get() > 0));
        }
    }
}

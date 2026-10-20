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

/// The most memory the GPU driver wires at once, the whole system's GPU buffers together, in
/// bytes ([`gpu_wired_limit`]).
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct GpuWiredLimit(pub u64);

impl GpuWiredLimit {
    pub fn get(self) -> u64 {
        self.0
    }
}

/// The GPU wired limit on `device`: the `iogpu.wired_limit_mb` sysctl where set, else the default
/// measured for its chip and memory ([`MEASURED_DEFAULT_LIMITS`]). `None` where neither says: the
/// default is not a function of memory alone (a 36 GiB M3 Pro faults near 24 GiB, but a 32 GiB M5
/// serves with 26.9 GiB wired), so a Mac nobody measured keeps the figure Metal reports.
///
/// `recommendedMaxWorkingSetSize` reports the limit where the sysctl sets it, but can overstate
/// the default: 28.08 GiB on the 36 GiB M3 Pro, whose command buffers fault out of memory once the
/// system's wired memory nears 24 GiB (and run at a 24 GiB sysctl, where it reports 24).
pub fn gpu_wired_limit(device: &ProtocolObject<dyn MTLDevice>) -> Option<GpuWiredLimit> {
    match sysctl_uint(c"iogpu.wired_limit_mb") {
        Some(mb) if mb > 0 => Some(GpuWiredLimit(mb << 20)),
        _ => measured_default_limit(&device.name().to_string(), sysctl_uint(c"hw.memsize")?),
    }
}

/// The default GPU wired limits measured where `iogpu.wired_limit_mb` is unset: `(chip family,
/// memory, limit)`, the family as the device name carries it. An entry is a fault measured at its
/// limit on that chip, not a rule for others.
const MEASURED_DEFAULT_LIMITS: &[(&str, u64, u64)] = &[
    // macOS 26.6.2: gemma-4-26b's 8x4096-token batch faults once the system's wired memory nears
    // 24 GiB, and runs at an iogpu.wired_limit_mb of 24576 (#412).
    ("M3", 36 << 30, 24 << 30),
];

/// The measured default limit for a device named `name` with `memsize` bytes of memory.
fn measured_default_limit(name: &str, memsize: u64) -> Option<GpuWiredLimit> {
    (MEASURED_DEFAULT_LIMITS.iter())
        .find(|&&(chip, mem, _)| memsize == mem && name.split(' ').any(|w| w == chip))
        .map(|&(_, _, limit)| GpuWiredLimit(limit))
}

/// The working set a budget on `device` sizes from: the smaller of what Metal recommends and the
/// GPU wired limit ([`gpu_wired_limit`]). Every Metal budget reads this one figure, so a default
/// sized against it never trips a guard sized against another.
pub fn usable_working_set(device: &ProtocolObject<dyn MTLDevice>) -> u64 {
    let recommended = device.recommendedMaxWorkingSetSize();
    gpu_wired_limit(device).map_or(recommended, |limit| recommended.min(limit.get()))
}

/// An unsigned integer sysctl, 4 or 8 bytes wide (`iogpu.wired_limit_mb` is an `int`,
/// `hw.memsize` a `uint64_t`). `None` if the kernel does not answer, or answers another width.
fn sysctl_uint(name: &std::ffi::CStr) -> Option<u64> {
    let mut buf = [0u8; 8];
    let mut len = buf.len();
    // SAFETY: `buf` holds `len` bytes; the kernel writes at most that many and sets `len` to the
    // count it wrote, which the match below decodes by width.
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            buf.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    match (status, len) {
        (0, 4) => Some(u64::from(u32::from_ne_bytes(buf[..4].try_into().ok()?))),
        (0, 8) => Some(u64::from_ne_bytes(buf)),
        _ => None,
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

    /// Only a measured Mac gets a default limit: the 36 GiB M3 Pro, whatever its M3 variant is
    /// named. A 32 GiB M5 serves past what any memory-only rule would allow, and an M3 of another
    /// size, or an M13, is not this one.
    #[test]
    fn only_a_measured_mac_gets_a_default_limit() {
        let limit = |name, gib: u64| measured_default_limit(name, gib << 30).map(|l| l.get() >> 30);
        assert_eq!(limit("Apple M3 Pro", 36), Some(24));
        assert_eq!(limit("Apple M3 Max", 36), Some(24));
        assert_eq!(limit("Apple M3 Pro", 18), None);
        assert_eq!(limit("Apple M5", 32), None);
        assert_eq!(limit("Apple M1 Max", 64), None);
        assert_eq!(limit("Apple M13", 36), None);
    }

    /// Both sysctls the limit reads answer on a Mac, each at its own width:
    /// `iogpu.wired_limit_mb` (an `int`, 0 where unset) and `hw.memsize` (a `uint64_t`). The
    /// limit is the sysctl where it is set, else the default measured for this Mac, if any.
    #[test]
    fn the_gpu_wired_limit_is_read_from_the_kernel() {
        let mb = sysctl_uint(c"iogpu.wired_limit_mb").expect("iogpu.wired_limit_mb answers");
        let memsize = sysctl_uint(c"hw.memsize").expect("hw.memsize answers");
        assert!(memsize > 0, "hw.memsize {memsize}");
        let Some(device) = MTLCreateSystemDefaultDevice() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let limit = gpu_wired_limit(&device).map(GpuWiredLimit::get);
        match mb {
            0 => assert_eq!(
                limit,
                measured_default_limit(&device.name().to_string(), memsize).map(GpuWiredLimit::get)
            ),
            mb => assert_eq!(limit, Some(mb << 20)),
        }
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

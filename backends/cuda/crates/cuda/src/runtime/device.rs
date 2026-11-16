// SPDX-License-Identifier: Apache-2.0
//! The GPU: device 0, checked to be an H100, with every kernel's one-time setup done.

use std::cell::Cell;
use std::ffi::c_void;
use std::fmt;

use super::ffi;
use super::{CudaError, check};
use crate::kernels::{self, KernelError};

/// Why the GPU could not be opened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceError {
    /// A runtime call failed: no driver, the toolkit's stub `libcuda` (`cudaErrorStubLibrary`,
    /// 34), or no device it can use.
    Cuda(CudaError),
    /// The runtime reports zero devices.
    NoDevice,
    /// Device 0 is not an H100.
    NotH100 { major: i32, minor: i32 },
    /// A kernel's one-time setup failed.
    Prepare {
        kernel: &'static str,
        error: KernelError,
    },
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(e) => write!(f, "cannot open the CUDA device: {e}"),
            Self::NoDevice => write!(f, "cannot open the CUDA device: the runtime sees no GPU"),
            Self::NotH100 { major, minor } => write!(
                f,
                "scratchy's CUDA kernels are built for sm_90a only (H100); device 0 is compute \
                 capability {major}.{minor}"
            ),
            Self::Prepare { kernel, error } => {
                write!(f, "the {kernel} kernel's one-time setup failed: {error}")
            }
        }
    }
}

impl std::error::Error for DeviceError {}

/// Device 0, opened. Only device 0 is used, which is CUDA's default on every host thread.
///
/// It owns the only function that allocates device memory, and counts the allocations that
/// succeed: the observable fact a "no allocation per forward" check reads.
pub struct CudaDevice {
    allocations: Cell<u64>,
}

impl CudaDevice {
    /// Open device 0: refuse no GPU and anything but compute capability 9.0, then run every
    /// kernel's one-time setup ([`kernels::prepare_all`]).
    pub fn open() -> Result<Self, DeviceError> {
        let mut count = 0;
        // SAFETY: `count` is a valid out-pointer.
        check(
            unsafe { ffi::cudaGetDeviceCount(&mut count) },
            "cudaGetDeviceCount",
        )
        .map_err(DeviceError::Cuda)?;
        if count == 0 {
            return Err(DeviceError::NoDevice);
        }
        // SAFETY: no pointer arguments; device 0 exists.
        check(unsafe { ffi::cudaSetDevice(0) }, "cudaSetDevice").map_err(DeviceError::Cuda)?;
        let attribute = |attr, call| {
            let mut value = 0;
            // SAFETY: `value` is a valid out-pointer.
            check(
                unsafe { ffi::cudaDeviceGetAttribute(&mut value, attr, 0) },
                call,
            )
            .map(|()| value)
            .map_err(DeviceError::Cuda)
        };
        let major = attribute(
            ffi::ATTR_COMPUTE_CAPABILITY_MAJOR,
            "cudaDeviceGetAttribute(major)",
        )?;
        let minor = attribute(
            ffi::ATTR_COMPUTE_CAPABILITY_MINOR,
            "cudaDeviceGetAttribute(minor)",
        )?;
        if (major, minor) != (9, 0) {
            return Err(DeviceError::NotH100 { major, minor });
        }
        kernels::prepare_all().map_err(|(kernel, error)| DeviceError::Prepare { kernel, error })?;
        Ok(Self {
            allocations: Cell::new(0),
        })
    }

    /// How many device allocations this device has made.
    pub fn allocations(&self) -> u64 {
        self.allocations.get()
    }

    /// `cudaMemGetInfo`: the device's `(free, total)` bytes.
    pub fn mem_info(&self) -> Result<(usize, usize), CudaError> {
        let (mut free, mut total) = (0, 0);
        // SAFETY: both out-pointers are valid for one usize.
        check(
            unsafe { ffi::cudaMemGetInfo(&mut free, &mut total) },
            "cudaMemGetInfo",
        )?;
        Ok((free, total))
    }

    /// `bytes` of uninitialized device memory, counted. The one function that calls `cudaMalloc`.
    pub(crate) fn malloc(&self, bytes: usize) -> Result<*mut c_void, CudaError> {
        let mut ptr = std::ptr::null_mut();
        // SAFETY: `ptr` is a valid out-pointer.
        check(unsafe { ffi::cudaMalloc(&mut ptr, bytes) }, "cudaMalloc")?;
        self.allocations.set(self.allocations.get() + 1);
        Ok(ptr)
    }
}

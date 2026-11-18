// SPDX-License-Identifier: Apache-2.0
//! The GPU: device 0, with every kernel's one-time setup done.

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
    /// A kernel's one-time setup failed. On a device other than an H100 this is the first
    /// kernel's: its `sm_90a` image does not load there (`cudaErrorNoKernelImageForDevice`).
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
    /// Open device 0: refuse no GPU, then run every kernel's one-time setup
    /// ([`kernels::prepare_all`]), which is where a device the kernels were not built for fails.
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

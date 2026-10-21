// SPDX-License-Identifier: Apache-2.0
//! THE CUDA RUNTIME — the counterpart of metal's `interpreter/metal/` worker and pool.
//!
//! It opens the GPU ([`CudaDevice`]), holds device memory ([`DeviceBuffer`]), and plays a
//! compile-time tape ([`CudaWorker`]): one arena allocated once per worker, one stream, and per
//! forward the input copy, every step's launch in tape order, the result copy and one wait. It
//! makes no decision: the steps, the arena's size and every offset are constants of the tape.
//!
//! Only under `nvcc`: without the kernels nothing links `cudart`, so nothing here could run.

mod buffer;
mod device;
mod ffi;
mod worker;

use std::ffi::CStr;
use std::fmt;

pub use buffer::DeviceBuffer;
pub use device::{CudaDevice, DeviceError};
pub use worker::{CudaWorker, ForwardError, WorkerError};

/// A failed CUDA runtime call: which one, its `cudaError_t`, and `cudaGetErrorString`'s text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaError {
    pub call: &'static str,
    pub code: i32,
    pub message: String,
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} failed: CUDA error {} ({})",
            self.call, self.code, self.message
        )
    }
}

impl std::error::Error for CudaError {}

/// `Ok` for `cudaSuccess`, else the call's error with its text.
pub(crate) fn check(code: ffi::CudaErrorT, call: &'static str) -> Result<(), CudaError> {
    if code == 0 {
        return Ok(());
    }
    // SAFETY: cudaGetErrorString returns a static NUL-terminated string for any code.
    let message = unsafe { CStr::from_ptr(ffi::cudaGetErrorString(code)) };
    Err(CudaError {
        call,
        code,
        message: message.to_string_lossy().into_owned(),
    })
}

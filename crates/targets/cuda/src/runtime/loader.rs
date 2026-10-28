// SPDX-License-Identifier: Apache-2.0
//! THE WEIGHT LOADER'S DEVICE HALF: a checkpoint into one device buffer per weight index.
//!
//! The host half is `crate::weights`: [`load_weights`] opens the checkpoint through the shared
//! `GpuWeights` reader, checks every entry of the emitted table before the first device
//! allocation, then stages and uploads one entry at a time. Peak host memory is one weight's
//! staging copy above the mmap.

use std::path::Path;

use super::buffer::DeviceBuffer;
use super::device::CudaDevice;
use crate::tape::CudaTape;
use crate::weights::{CudaWeight, LoadError, check, open_checkpoint, stage};

/// The device buffers for `table` from the checkpoint at `checkpoint_dir`, indexed by
/// `WeightIx`: what `CudaWorker::new` takes for `tapes`.
///
/// Every refusal but [`LoadError::Upload`] comes before any device allocation: the checkpoint is
/// opened ([`open_checkpoint`]) and every entry checked against its header ([`check`]) first. Then each entry is staged
/// on the host and uploaded into one new buffer of exactly its bytes (one `cudaMalloc` and one
/// copy); an index with no entry gets [`DeviceBuffer::empty`], which allocates nothing. A failed
/// upload drops the buffers already made.
pub fn load_weights(
    device: &CudaDevice,
    checkpoint_dir: &Path,
    tapes: &'static [CudaTape],
    table: &'static [Option<CudaWeight>],
) -> Result<Vec<DeviceBuffer>, LoadError> {
    let mut weights = open_checkpoint(checkpoint_dir)?;
    check(table, tapes, &weights)?;
    (table.iter().enumerate())
        .map(|(ix, entry)| match entry {
            None => Ok(DeviceBuffer::empty()),
            Some(entry) => {
                let host = stage(ix, entry, &mut weights)?;
                DeviceBuffer::upload_bytes(device, &host)
                    .map_err(|error| LoadError::Upload { ix, error })
            }
        })
        .collect()
}

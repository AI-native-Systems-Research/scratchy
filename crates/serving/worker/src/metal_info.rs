// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Pre-load Metal-device queries.
//!
//! Lets `scratchy-serving-api::init` consult the system Metal device for things like
//! its usable working set *before* any worker is constructed, so
//! sizing/memory guards (e.g. the target + draft model pair check) can
//! fail fast without touching the
//! heavy worker init path.

use objc2_metal::MTLCreateSystemDefaultDevice;

/// The working set every Metal budget sizes from on the default system device
/// ([`scratchy_target_metal::device::usable_working_set`]): the smaller of
/// `recommendedMaxWorkingSetSize` and the GPU wired limit.
///
/// Returns `None` when no Metal device is available (e.g. running the
/// metal-feature build inside a non-macOS container).
pub fn usable_working_set() -> Option<u64> {
    let device = MTLCreateSystemDefaultDevice()?;
    Some(scratchy_target_metal::device::usable_working_set(&device))
}

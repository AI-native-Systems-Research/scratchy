// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Pre-load Metal-device queries.
//!
//! Lets `scratchy-serving-api::init` consult the system Metal device for things like
//! `recommendedMaxWorkingSetSize` *before* any worker is constructed, so
//! sizing/memory guards (e.g. the target + draft model pair check) can
//! fail fast without touching the
//! heavy worker init path.

use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

/// Apple's recommended budget for resident MTLBuffers on the default
/// system device (`recommendedMaxWorkingSetSize`).
///
/// Returns `None` when no Metal device is available (e.g. running the
/// metal-feature build inside a non-macOS container).
pub fn recommended_max_working_set_size() -> Option<u64> {
    let device = MTLCreateSystemDefaultDevice()?;
    Some(device.recommendedMaxWorkingSetSize())
}

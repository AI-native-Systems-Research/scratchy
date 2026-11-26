// SPDX-License-Identifier: Apache-2.0
//! Re-exports `crate::tape::quantized`, the pure qmv/qmm selection, naming and dispatch-shape
//! layer the tape construction reads. The objc type aliases live here: they are not part of the
//! selection logic.
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice};

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;

pub use crate::tape::quantized::*;

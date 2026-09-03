// SPDX-License-Identifier: Apache-2.0
//! MOVED to `crate::tape::quantized` — the pure qmv/qmm
//! selection, naming and dispatch-shape layer the tape construction
//! reads. The objc type aliases stay here (they never belonged to the
//! selection logic).
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice};

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;

pub use crate::tape::quantized::*;

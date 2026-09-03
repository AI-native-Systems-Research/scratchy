// SPDX-License-Identifier: Apache-2.0
//! Qwen3.5-MoE shared-expert combine dispatcher. Mirrors the trailing
//! Python expression `routed + shared_y * sigmoid(shared_expert_gate(x))`
//! from mlx-lm `Qwen3NextSparseMoeBlock` / transformers
//! `Qwen3_5MoeSparseMoeBlock`. The gate is `[rows, 1]` — one scalar per
//! token, row-broadcast across the hidden axis.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDataType, MTLDevice,
    MTLFunctionConstantValues, MTLLibrary,
};
use std::ffi::c_void;
use std::ptr::NonNull;

use crate::shader_cache::load_library_from_bytes;
use crate::stream::MetalStreamError;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateScaleDType {
    F16,
    BF16,
}

impl GateScaleDType {
    fn symbol(self) -> &'static str {
        match self {
            Self::F16 => "gate_scale_f16",
            Self::BF16 => "gate_scale_bf16",
        }
    }
}

pub struct GateScaleKernels {
    library: Library,
    device: Device,
}

impl GateScaleKernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library = load_library_from_bytes(device, crate::embedded_metallib!("gate_scale"))
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "load `gate_scale.metallib`: {e}"
                ))
            })?;
        Ok(Self {
            library,
            device: device.clone(),
        })
    }

    pub fn build_pipeline(
        &self,
        dtype: GateScaleDType,
        n: u32,
        cols: u32,
    ) -> Result<ComputePipelineState, MetalStreamError> {
        let constants = MTLFunctionConstantValues::new();
        unsafe {
            constants.setConstantValue_type_atIndex(
                NonNull::new(&n as *const u32 as *mut c_void).unwrap(),
                MTLDataType::UInt,
                0,
            );
            constants.setConstantValue_type_atIndex(
                NonNull::new(&cols as *const u32 as *mut c_void).unwrap(),
                MTLDataType::UInt,
                1,
            );
        }
        let name = NSString::from_str(dtype.symbol());
        let func = self
            .library
            .newFunctionWithName_constantValues_error(&name, &constants)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("{}: {e:?}", dtype.symbol()))
            })?;
        self.device
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "{} pipeline: {e:?}",
                    dtype.symbol()
                ))
            })
    }
}

/// CPU reference: `out[r, c] = routed[r, c] + shared_y[r, c] * sigmoid(g[r])`.
pub fn gate_scale_cpu_f32(
    routed: &[f32],
    shared_y: &[f32],
    gate: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    assert_eq!(routed.len(), rows * cols);
    assert_eq!(shared_y.len(), rows * cols);
    assert_eq!(gate.len(), rows);
    assert_eq!(out.len(), rows * cols);
    for r in 0..rows {
        let sig = 1.0_f32 / (1.0 + (-gate[r]).exp());
        for c in 0..cols {
            out[r * cols + c] = routed[r * cols + c] + shared_y[r * cols + c] * sig;
        }
    }
}

// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Fused kernel implementations for memory-bandwidth optimization.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLBuffer, MTLComputePipelineState, MTLDevice, MTLLibrary};
use std::sync::Arc;

use crate::{MetalDevice, MetalStreamError};

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

pub struct FusedAddRmsNorm {
    pub pipeline_f16_vec4: ComputePipelineState,
}

fn compile_library(device: &Device, source: &str) -> Result<Library, MetalStreamError> {
    let opts = objc2_metal::MTLCompileOptions::new();
    let ns_source = NSString::from_str(source);
    device
        .newLibraryWithSource_options_error(&ns_source, Some(&opts))
        .map_err(|e| MetalStreamError::ShaderCompilationFailed(format!("{e:?}")))
}

fn compile_pipeline(
    device: &Device,
    library: &Library,
    name: &str,
) -> Result<ComputePipelineState, MetalStreamError> {
    let ns_name = NSString::from_str(name);
    let function = library
        .newFunctionWithName(&ns_name)
        .ok_or_else(|| MetalStreamError::ShaderCompilationFailed(format!("missing fn {name}")))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| MetalStreamError::ShaderCompilationFailed(format!("{e:?}")))
}

impl FusedAddRmsNorm {
    pub fn new(device: Arc<MetalDevice>) -> Result<Self, MetalStreamError> {
        let library = compile_library(
            &device.device,
            include_str!("../shaders/fused_add_rmsnorm.metal"),
        )?;
        let pipeline_f16_vec4 =
            compile_pipeline(&device.device, &library, "fused_add_rmsnorm_f16_vec4")?;

        Ok(Self { pipeline_f16_vec4 })
    }
}

pub struct FusedGateUpSiluMul {
    pub pipeline_f16_vec4: ComputePipelineState,
    pub pipeline_f16_concat_vec4: ComputePipelineState,
    pub pipeline_gelu_f16: ComputePipelineState,
}

impl FusedGateUpSiluMul {
    pub fn new(device: Arc<MetalDevice>) -> Result<Self, MetalStreamError> {
        let library = compile_library(
            &device.device,
            include_str!("../shaders/fused_gate_up_silu_mul.metal"),
        )?;

        Ok(Self {
            pipeline_f16_vec4: compile_pipeline(
                &device.device,
                &library,
                "fused_gate_up_silu_mul_f16_vec4",
            )?,
            pipeline_f16_concat_vec4: compile_pipeline(
                &device.device,
                &library,
                "fused_gate_up_silu_mul_concat_f16_vec4",
            )?,
            pipeline_gelu_f16: compile_pipeline(
                &device.device,
                &library,
                "fused_gate_up_gelu_mul_f16",
            )?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect_device;

    #[test]
    fn test_fused_add_rmsnorm_creation() {
        let Some(device) = detect_device() else {
            eprintln!("skipping: no Metal device (or no Metal 4 support)");
            return;
        };
        let fused_norm = FusedAddRmsNorm::new(std::sync::Arc::new(device));
        assert!(fused_norm.is_ok());
    }

    #[test]
    fn test_fused_gate_up_silu_mul_creation() {
        let Some(device) = detect_device() else {
            eprintln!("skipping: no Metal device (or no Metal 4 support)");
            return;
        };
        let fused_silu = FusedGateUpSiluMul::new(std::sync::Arc::new(device));
        assert!(fused_silu.is_ok());
    }
}

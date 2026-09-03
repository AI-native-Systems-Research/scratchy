// SPDX-License-Identifier: Apache-2.0
//
// Rust dispatcher for the precise softmax kernel (port of MLX
// `softmax_single_row` from `mlx/backend/metal/kernels/softmax.h`).
//
// Used by the MoE router lowering arm: `router_logits` →
// `router_probs` over axis=-1 with shape `[N, num_experts]`.
// `precise=True` is the only mode the Qwen / Mixtral MoE blocks
// use (qwen3_moe.py:128, qwen2_moe.py:131, mixtral.py:116) — they
// route through float accumulation regardless of input dtype.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLLibrary};

use crate::shader_cache::load_library_from_bytes;
use crate::stream::MetalStreamError;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

/// Block size used to dispatch `softmax_single_row`. MLX picks 256
/// for small axes; we hold to that until we have benchmark numbers
/// from a real MoE forward pass.
pub const SOFTMAX_BLOCK_THREADS: usize = 256;

/// `N_READS` baked into the shader template arg — must match
/// `MLX_N_READS` in shaders/softmax.metal.
pub const SOFTMAX_N_READS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxDType {
    F32,
    F16,
    BF16,
}

impl SoftmaxDType {
    fn precise_symbol(self) -> &'static str {
        match self {
            // F32 has no precise variant in MLX — the precise mode
            // exists to lift half/bfloat → float accum, which is
            // a no-op when T is already float. Callers that ask
            // for F32 precise route to the non-precise symbol.
            SoftmaxDType::F32 => "block_softmax_float32",
            SoftmaxDType::F16 => "block_softmax_precise_float16",
            SoftmaxDType::BF16 => "block_softmax_precise_bfloat16",
        }
    }
}

pub struct SoftmaxKernels {
    pub precise_f16: ComputePipelineState,
    pub precise_bf16: ComputePipelineState,
    pub f32: ComputePipelineState,
    _library: Library,
}

impl SoftmaxKernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library = load_library_from_bytes(device, crate::embedded_metallib!("softmax"))
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("load `softmax.metallib`: {e}"))
            })?;
        let precise_f16 = build_pipeline(device, &library, "block_softmax_precise_float16")?;
        let precise_bf16 = build_pipeline(device, &library, "block_softmax_precise_bfloat16")?;
        let f32 = build_pipeline(device, &library, "block_softmax_float32")?;
        Ok(Self {
            precise_f16,
            precise_bf16,
            f32,
            _library: library,
        })
    }

    pub fn pipeline_for(&self, dtype: SoftmaxDType) -> &ComputePipelineState {
        match dtype {
            SoftmaxDType::F16 => &self.precise_f16,
            SoftmaxDType::BF16 => &self.precise_bf16,
            SoftmaxDType::F32 => &self.f32,
        }
    }
}

fn build_pipeline(
    device: &Device,
    library: &Library,
    name: &str,
) -> Result<ComputePipelineState, MetalStreamError> {
    let ns_name = NSString::from_str(name);
    let function = library
        .newFunctionWithName(&ns_name)
        .ok_or_else(|| MetalStreamError::ShaderCompilationFailed(format!("{name} fn missing")))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| MetalStreamError::ShaderCompilationFailed(format!("{name} pipeline: {e:?}")))
}

/// Dispatch precise row-wise softmax over `[rows, axis_size]`. The
/// in/out buffers may alias (MLX in-place softmax pattern). Used by
/// the MoE router lowering arm for `mx.softmax(gates, axis=-1,
/// precise=True)`.
/// CPU reference implementation of precise softmax for parity tests.
/// Matches MLX's `mx.softmax(x, axis=-1, precise=True)` — float accum
/// then cast back to input dtype.
pub fn softmax_cpu_f32(input: &[f32], rows: usize, axis_size: usize, out: &mut [f32]) {
    assert_eq!(input.len(), rows * axis_size);
    assert_eq!(out.len(), rows * axis_size);
    for r in 0..rows {
        let row = &input[r * axis_size..(r + 1) * axis_size];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        let mut exps = vec![0.0_f32; axis_size];
        for (i, &v) in row.iter().enumerate() {
            let e = (v - m).exp();
            exps[i] = e;
            sum += e;
        }
        let inv = 1.0_f32 / sum;
        for (i, e) in exps.iter().enumerate() {
            out[r * axis_size + i] = *e * inv;
        }
    }
}

pub fn symbol_for(dtype: SoftmaxDType) -> &'static str {
    dtype.precise_symbol()
}

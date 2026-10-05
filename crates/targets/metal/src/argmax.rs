// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `argmax` — greedy-sample MSL kernel + Rust dispatcher, baked per model
//! ([`crate::off_tape`]).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};

use crate::off_tape::{OffTapeKernels, OffTapePipeline};
use crate::shader_cache::ComputePipelineState;
use crate::stream::MetalStreamError;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;

pub const ARGMAX_DEFAULT_TG_SIZE: usize = 1024;

pub struct ArgmaxKernels {
    pub argmax: OffTapePipeline,
    /// Phase 6 dual-write variant: writes the argmax to BOTH the
    /// per-iter output buffer (host-visible draft target) AND a
    /// second buffer (the next K-step iter's `runtime.input_ids`).
    /// Encoded into the same MTL4 compute encoder as the forward; the
    /// next iter's embed kernel reads from `next_in` and Metal's
    /// intra-encoder write→read hazard tracking serializes them.
    pub dual_write: OffTapePipeline,
}

impl ArgmaxKernels {
    pub fn new(device: &Device, kernels: &OffTapeKernels) -> Result<Self, MetalStreamError> {
        Ok(Self {
            argmax: OffTapePipeline::new(device, &kernels.argmax)?,
            dual_write: OffTapePipeline::new(device, &kernels.argmax_dual_write)?,
        })
    }
}

/// MTL4 encoder-tail argmax dispatcher.
///
/// Encodes `setComputePipelineState` + `setArgumentTable` +
/// `dispatchThreadgroups` onto a caller-supplied
/// [`MTL4ComputeCommandEncoder`]. The caller is responsible for
/// encoder/CB/queue lifecycle (typically the same encoder being used
/// to encode the model forward — append argmax as the tail of the
/// forward CB so they share one commit and one host wait).
///
/// The argument table must be pre-built with:
/// - index 0: logits GPU address
/// - index 1: output GPU address
/// - index 2: 4-byte address holding `batch` (u32)
pub fn encode_argmax_into_mtl4(
    kernels: &ArgmaxKernels,
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    arg_table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
    batch: u32,
) -> Result<(), MetalStreamError> {
    encode_argmax_into_mtl4_inner(&kernels.argmax, encoder, arg_table, batch, "argmax")
}

/// Phase 6 dual-write argmax — writes argmax to TWO buffers in one
/// dispatch. Bindings (must match `argmax_dual_write` in
/// `shaders/argmax.metal`):
///   index 0: logits      (read)
///   index 1: output      (write — host-visible draft buffer)
///   index 2: batch       (read const u32)
///   index 3: next_in     (write — next iter's input_ids buffer)
pub fn encode_argmax_dual_write_into_mtl4(
    kernels: &ArgmaxKernels,
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    arg_table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
    batch: u32,
) -> Result<(), MetalStreamError> {
    encode_argmax_into_mtl4_inner(
        &kernels.dual_write,
        encoder,
        arg_table,
        batch,
        "argmax_dual_write",
    )
}

fn encode_argmax_into_mtl4_inner(
    pipeline: &ComputePipelineState,
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    arg_table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
    batch: u32,
    name: &'static str,
) -> Result<(), MetalStreamError> {
    use objc2_metal::{
        MTL4CommandEncoder as _, MTL4ComputeCommandEncoder as _, MTL4VisibilityOptions, MTLStages,
    };
    // `barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions`
    // is on the `MTL4CommandEncoder` super-trait; the `as _`
    // imports above bring its methods into scope on the
    // `&ProtocolObject<dyn MTL4ComputeCommandEncoder>` we received.
    if batch == 0 {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "{name} encode: batch=0"
        )));
    }
    // **Barrier before argmax** — this kernel reads the lm_head output
    // (the forward encoder's last write). MTL4 compute encoders do
    // NOT auto-serialize same-encoder dispatches; without this
    // barrier argmax can fire concurrently with the forward's tail
    // dispatches and read stale logits. Pre-fusion (when argmax ran
    // on a separate command buffer with host wait) the host
    // wait_until_completed acted as the barrier; once argmax was
    // fused onto the forward encoder
    // (`scratchy-serving-worker::gpu_worker::execute_model` followup hook),
    // the implicit cross-CB sync was lost and the race surfaced as
    // wrong first tokens for any forward whose tail writes target
    // the lm_head output slot — most visibly the lm_head slice
    // (`scatter_first_to_last_row`) at single-seq prefill.
    //
    // `Device` visibility (cache-coherent): the forward's last
    // store may live in L2 only; argmax loads from `device` address
    // space and needs the store visible. Unlike the worker's
    // intra-tape barriers (execution-only `None` wins there), argmax
    // always needs Device because it crosses the implicit
    // producer/consumer boundary the worker's intra-tape
    // `barrier_before` flags don't model.
    encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
        MTLStages::Dispatch,
        MTLStages::Dispatch,
        MTL4VisibilityOptions::Device,
    );
    encoder.setComputePipelineState(pipeline);
    encoder.setArgumentTable(Some(arg_table));
    let threadgroups = MTLSize {
        width: batch as usize,
        height: 1,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: ARGMAX_DEFAULT_TG_SIZE,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
    Ok(())
}

/// Tiny helper: stage `data` into a fresh `StorageModeShared` buffer.
pub fn upload_shared_buffer<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let nbytes = std::mem::size_of_val(data);
    let buffer = device
        .newBufferWithLength_options(nbytes.max(1), MTLResourceOptions::StorageModeShared)
        .expect("upload_shared_buffer");
    if nbytes > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                buffer.contents().as_ptr() as *mut u8,
                nbytes,
            );
        }
    }
    buffer
}

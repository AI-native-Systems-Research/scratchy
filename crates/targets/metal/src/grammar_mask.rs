// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `grammar_mask_{f16,bf16}` — constrained-decoding logit mask + Rust
//! dispatcher.
//!
//! Forces every token NOT permitted by a request's grammar FSM to `-inf`
//! before the greedy `argmax`, so constrained / guided decoding (JSON
//! schema, regex, EBNF, structural tags) produces only grammar-valid
//! tokens. The allowed set is a dense bitset (1 bit/token, set =
//! allowed), one row per constrained request; `rows[gid]` maps the
//! grammar row to the logits batch row to mask, so a mixed batch (some
//! requests constrained, some free) is handled by dispatching exactly
//! `num_rows` threadgroups and leaving the other logits rows untouched.
//!
//! Mirrors [`crate::argmax`]: a fused [`encode_grammar_mask_f16_into_mtl4`]
//! that appends the mask onto the forward's MTL4 compute encoder right
//! before argmax. The kernel unit tests exercise the same kernel through
//! the shared MTL4 single-op dispatch helper.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};

use crate::shader_cache::load_library_from_bytes;
use crate::stream::MetalStreamError;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

/// Threads per threadgroup; one threadgroup masks one constrained row,
/// striding the threads over the vocab axis (no reduction).
pub const GRAMMAR_MASK_TG_SIZE: usize = 256;

pub struct GrammarMaskKernels {
    pub f16: ComputePipelineState,
    pub bf16: ComputePipelineState,
    _library: Library,
}

impl GrammarMaskKernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library = load_library_from_bytes(device, crate::embedded_metallib!("grammar_mask"))
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "load `grammar_mask.metallib`: {e}"
                ))
            })?;
        let f16 = build_pipeline(device, &library, "grammar_mask_f16")?;
        let bf16 = build_pipeline(device, &library, "grammar_mask_bf16")?;
        Ok(Self {
            f16,
            bf16,
            _library: library,
        })
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

/// Number of u32 words in a dense vocab bitset row.
#[inline]
pub fn words_per_row(vocab: u32) -> u32 {
    vocab.div_ceil(32)
}

/// MTL4 encoder-tail grammar-mask dispatcher (f16 logits).
///
/// Appends `setComputePipelineState` + `setArgumentTable` +
/// `dispatchThreadgroups(num_rows)` onto a caller-supplied
/// [`objc2_metal::MTL4ComputeCommandEncoder`] — typically the same
/// encoder used to encode the forward, so the mask runs inside the
/// forward CB right before the fused argmax (one commit, one host wait).
///
/// The argument table must be pre-built with:
/// - index 0: logits GPU address        (`[total_n, vocab]` half, in/out)
/// - index 1: allow_bits GPU address    (`[num_rows * words_per_row]` u32)
/// - index 2: rows GPU address          (`[num_rows]` u32; grammar row -> logits row)
/// - index 3: 4-byte address holding `vocab` (u32)
/// - index 4: 4-byte address holding `words_per_row` (u32)
pub fn encode_grammar_mask_f16_into_mtl4(
    kernels: &GrammarMaskKernels,
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    arg_table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
    num_rows: u32,
) -> Result<(), MetalStreamError> {
    encode_grammar_mask_into_mtl4_inner(
        &kernels.f16,
        encoder,
        arg_table,
        num_rows,
        "grammar_mask_f16",
    )
}

/// BF16 variant of [`encode_grammar_mask_f16_into_mtl4`].
pub fn encode_grammar_mask_bf16_into_mtl4(
    kernels: &GrammarMaskKernels,
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    arg_table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
    num_rows: u32,
) -> Result<(), MetalStreamError> {
    encode_grammar_mask_into_mtl4_inner(
        &kernels.bf16,
        encoder,
        arg_table,
        num_rows,
        "grammar_mask_bf16",
    )
}

fn encode_grammar_mask_into_mtl4_inner(
    pipeline: &ComputePipelineState,
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    arg_table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
    num_rows: u32,
    name: &'static str,
) -> Result<(), MetalStreamError> {
    use objc2_metal::{
        MTL4CommandEncoder as _, MTL4ComputeCommandEncoder as _, MTL4VisibilityOptions, MTLStages,
    };
    if num_rows == 0 {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "{name} encode: num_rows=0"
        )));
    }
    // **Barrier before the mask** — this kernel reads+writes the lm_head
    // output (the forward encoder's last write). MTL4 compute encoders do
    // NOT auto-serialize same-encoder dispatches, so without this barrier
    // the mask could fire concurrently with the forward's tail dispatches
    // and read stale logits. The argmax encode that follows emits its own
    // pre-barrier, which then orders mask-write -> argmax-read. `Device`
    // visibility (cache-coherent) mirrors the argmax path: the forward's
    // last store may live in L2 only and the mask loads from `device`
    // address space. See `argmax::encode_argmax_into_mtl4_inner`.
    encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
        MTLStages::Dispatch,
        MTLStages::Dispatch,
        MTL4VisibilityOptions::Device,
    );
    encoder.setComputePipelineState(pipeline);
    encoder.setArgumentTable(Some(arg_table));
    let threadgroups = MTLSize {
        width: num_rows as usize,
        height: 1,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: GRAMMAR_MASK_TG_SIZE,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
    Ok(())
}

/// Build a dense allow-bitset row (`words_per_row(vocab)` u32 words) from
/// a list of allowed token ids: bit `t` set iff token `t` is allowed.
/// Token ids `>= vocab` are skipped. Appends into `out` (row-major), so a
/// batch of rows can be packed by calling this once per row.
pub fn push_allow_bitset_row(out: &mut Vec<u32>, allowed: &[u32], vocab: u32) {
    let wpr = words_per_row(vocab) as usize;
    let base = out.len();
    out.resize(base + wpr, 0);
    let row = &mut out[base..base + wpr];
    for &tok in allowed {
        if tok < vocab {
            row[(tok >> 5) as usize] |= 1u32 << (tok & 31);
        }
    }
}

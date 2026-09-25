// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal kernel infrastructure for scratchy.
//!
//! Provides Metal device management, shader compilation, and kernel dispatch
//! primitives for Apple Silicon GPUs.

// Kernel-launch / dispatch / record entry points pass the full set of
// buffers, dims, scales, and offsets positionally — bundling them into
// param structs would only obscure the 1:1 mapping to the MSL kernel
// signatures, so `too_many_arguments` is expected here. Likewise the
// CPU-reference helpers mirror those wide signatures. `type_complexity`
// is allowed for the same shader-table lookup return tuples.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

/// Compile-time-generated kernel instantiation tables. The
/// `STEEL_PAGED_HEAD_DIMS` slice and `steel_paged_symbol()` lookup
/// in here are emitted by `build.rs` from the same head-dim list
/// that produces `attention_steel_paged_instantiations.h`. The
/// dispatcher in `scratchy-forward-compiler/.../lowering.rs` must consume
/// these — never hand-list head dims at the call site.
pub mod steel_paged {
    include!(concat!(
        env!("OUT_DIR"),
        "/steel_paged_kernels_generated.rs"
    ));

    /// MSL symbol for the NAX (matrix-accelerator) paged-attention kernel.
    ///
    /// Unlike `steel_paged_symbol` (a generated table over
    /// `STEEL_PAGED_HEAD_DIMS`), the NAX kernel is hand-instantiated in
    /// `attention_steel_nax_paged.metal` at a single tile shape — BQ=64,
    /// BK=32, WM=4, WN=1, BLOCK_SIZE=16 — and ONLY for head_dim 128
    /// (`INST_STEEL_NAX_PAGED(.., 128)`). Returns `Some` only for that
    /// (dtype, head_dim) combo; the dispatcher falls back to the simdgroup
    /// `steel_paged_symbol` / SDPA path otherwise.
    pub fn nax_paged_symbol(dtype_tag: &str, head_dim: u32) -> Option<&'static str> {
        match (dtype_tag, head_dim) {
            ("f16", 64) => Some("attention_steel_nax_paged_f16_bq64_bk32_bd64_wm4_wn1_bs16"),
            ("bf16", 64) => Some("attention_steel_nax_paged_bf16_bq64_bk32_bd64_wm4_wn1_bs16"),
            ("f16", 128) => Some("attention_steel_nax_paged_f16_bq64_bk32_bd128_wm4_wn1_bs16"),
            ("bf16", 128) => Some("attention_steel_nax_paged_bf16_bq64_bk32_bd128_wm4_wn1_bs16"),
            _ => None,
        }
    }

    /// MSL symbol for the spans rope-once kernel (`rope_once_nax`), which
    /// ropes a request's K ONCE into the dense scratch so the NAX attention
    /// reads pre-roped K with no per-tile rotation. Lives in the same runtime-
    /// compiled `attention_steel_nax_paged` library and is instantiated at the
    /// same (dtype, head_dim 128) combos as `nax_paged_symbol`.
    pub fn rope_once_nax_symbol(dtype_tag: &str, head_dim: u32) -> Option<&'static str> {
        match (dtype_tag, head_dim) {
            ("f16", 64) => Some("rope_once_nax_f16_bd64_bs16"),
            ("bf16", 64) => Some("rope_once_nax_bf16_bd64_bs16"),
            ("f16", 128) => Some("rope_once_nax_f16_bd128_bs16"),
            ("bf16", 128) => Some("rope_once_nax_bf16_bd128_bs16"),
            _ => None,
        }
    }

    /// MSL symbol for the spans rope-once kernel on the SIMDGROUP steel path
    /// (`rope_once_steel`). The simdgroup twin of `rope_once_nax_symbol`: ropes
    /// a request's K ONCE into the dense scratch so `attention_steel_paged`
    /// reads pre-roped K with no per-tile rotation. Instantiated (alongside the
    /// `attention_steel_paged` body) for every `STEEL_PAGED_HEAD_DIMS` entry by
    /// the `INST_STEEL_PAGED` macro in `attention_steel_paged.metal`, so it
    /// covers the simdgroup head dims (64/96/128/256) — including SmolLM's hd64
    /// which the NAX (hd128-only) rope-once does not. Returns `None` for a
    /// head_dim with no steel instantiation; caller falls back to the in-kernel
    /// path / SDPA.
    pub fn rope_once_steel_symbol(dtype_tag: &str, head_dim: u32) -> Option<&'static str> {
        if !STEEL_PAGED_HEAD_DIMS.contains(&head_dim) {
            return None;
        }
        match dtype_tag {
            "f16" => Some(match head_dim {
                64 => "rope_once_steel_f16_bd64_bs16",
                96 => "rope_once_steel_f16_bd96_bs16",
                128 => "rope_once_steel_f16_bd128_bs16",
                256 => "rope_once_steel_f16_bd256_bs16",
                _ => return None,
            }),
            "bf16" => Some(match head_dim {
                64 => "rope_once_steel_bf16_bd64_bs16",
                96 => "rope_once_steel_bf16_bd96_bs16",
                128 => "rope_once_steel_bf16_bd128_bs16",
                256 => "rope_once_steel_bf16_bd256_bs16",
                _ => return None,
            }),
            _ => None,
        }
    }

    /// MSL symbol for the spans rope-once kernel on the GQA-COOPERATIVE shared
    /// prefill path (`rope_once_gqa_shared`). The gqa_shared twin of
    /// `rope_once_{steel,nax}_symbol`: ropes a request's K ONCE into the dense
    /// scratch so `attention_prefill_sdpa_gqa_shared_*` reads pre-roped K with
    /// no per-tile smem rotation. Lives in the AOT `attention` library and reads
    /// head_dim/block_size/rot_dim/pair_off from FUNCTION CONSTANTS (not a
    /// template), so ONE pair covers every (head_dim, block_size) the
    /// gqa_shared kernel takes — including gemma4 global (head_dim 512, bs 32),
    /// which has no steel/NAX instantiation. Only f16/bf16 (the gqa_shared
    /// kernel's two dtypes).
    pub fn rope_once_gqa_shared_symbol(dtype_tag: &str) -> Option<&'static str> {
        match dtype_tag {
            "f16" => Some("rope_once_gqa_shared_f16_specialized"),
            "bf16" => Some("rope_once_gqa_shared_bf16_specialized"),
            _ => None,
        }
    }
}

pub mod allocator;
pub mod argmax;
pub mod argpartition;
pub mod chain_advance;
pub mod cpu_reference;
pub mod device;
/// Metal lowered from the SHARED `SubtileTape` — the same artifact spyre lowers. Lives here,
/// in the target crate, beside spyre's `lower_subtile_tape_to_superdsc`.
pub mod from_tape;
pub mod grammar_mask;
/// ⛔ THE METAL OP ABI — which operand a kernel writes over, which ops are in-place,
/// the source/unary constructor tables. It lived in `compiler/macros/src/metal_op_abi.rs`:
/// target ABI facts inside the shared compiler, which is the arrangement CLAUDE.md says
/// these tables must NOT be in — they are INPUTS to the shared passes, and they belong to
/// the target. It depends only on `scratchy_ir` + `scratchy_subtile`, so nothing had to
/// move with it.
pub mod op_abi;
// Core-facing metal runtime: the `DeviceAllocator`-trait weight-loader
// (`MetalAllocator`) and the worker-facing `GpuDevice`. The canonical metal
// allocator — distinct from `allocator::PooledBufferAllocator` (legacy
// kernel-buffer pool).
pub mod device_metal;
pub mod fused_kernels;
pub mod gate_scale;
pub mod layers;
pub mod layers_moe;
pub mod layers_quant;
// Per-accessor layered weight-load helpers the forward-compiler macro emits as
// `crate::__gpu::load_layered_*`: the neutral ones re-shared from
// scratchy-layers, the metal-divergent (RmsNorm keep-dtype, direct-write pack)
// + metal-only quant loaders defined locally.
pub mod loaders;
pub mod weights_metal;
pub use weights_metal::MetalWeightsExt;
pub mod metal_allocator;
pub mod metal_mem;
pub mod moe_weighted_sum;
pub mod mtl4_dispatch;
pub mod owned_metal;
pub mod quantized;
pub mod residency;
pub mod sampling;
pub mod shader_cache;
pub mod single_buffer_kv;
pub mod slice_trailing_cols;
pub mod softmax;
pub mod specialized_pipeline_cache;
pub mod stream;
pub mod take_along_axis;
/// The pure tape-construction layer (no objc) — shared with the
/// `#[forward]` macro, which constructs these values at expansion.
pub mod tape;
/// Apple Silicon target profiles and cost models.
pub mod targets;
pub mod turboquant;

// Atom-driven metal megakernel synthesis (atom IR + fuse passes + AoT
// metallib compile). Consumed by the compiler's metal codegen and the
// metal cost-sweep.
pub mod aot;
pub mod atom;
pub mod atom_lib;
pub mod fuse_pass;

// Metal interpreter (lowering pass + worker pool) relocated out of the
// compiler — a `cfg(feature = "metal")` there was a cfg-elimination
// violation. cpu_golden + paged_kv_layout are its pure-CPU test oracles
// (used by the interpreter's `#[cfg(test)]` golden checks).
pub mod cpu_golden;
pub mod interpreter;
pub mod paged_kv_layout;

/// Number of paged-KV-cache blocks backed by one physical chunk buffer
/// in the metal reactive (chunked) KV pool. SINGLE SOURCE OF TRUTH for
/// the chunk granularity, baked into synth MSL and set as a
/// `[[function_constant]]` on the hand-written kernels.
pub const BLOCKS_PER_CHUNK: u32 = 128;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

/// Embed a precompiled `.metallib` produced by `build.rs` from
/// `shaders/<name>.metal`. Returns a `&'static [u8]` suitable for
/// `device.newLibraryWithData_error`. Replaces the runtime-MSL-compile
/// path that called `newLibraryWithSource:options:error:` —
/// the AoT version skips the MSL→AIR frontend on every process start.
#[macro_export]
macro_rules! embedded_metallib {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".metallib"))
    };
}

/// The build.rs-embedded `quantized_qmm_nax.metallib` bytes (compiled
/// with `-mmacosx-version-min=26.2` so the MPP `matmul2d` codegen is
/// correct). Exposed for the AOT-vs-JIT startup timing probe.
pub fn embedded_nax_metallib() -> &'static [u8] {
    embedded_metallib!("quantized_qmm_nax")
}

pub use allocator::{AllocatorError, PooledBuffer, PooledBufferAllocator};
pub use device::{MetalDevice, detect_device, metal4_available};
pub use device_metal::GpuDevice;
pub use metal_allocator::MetalAllocator;
pub use metal_mem::MetalMem;
pub use owned_metal::owned_from_metal_buffer;
pub use stream::MetalStreamError;

// ── `__gpu` surface scaffolding ──────────────────────────────────────────
// Backend-neutral re-exports + aliases under the historic
// `scratchy_target_cuda::…` paths so the forward-compiler macro's emitted
// `crate::__gpu::…` references resolve against this crate once the metal
// `__gpu` alias flips here. The neutral tensor core + layer types + RoPE
// cache live in scratchy-tensors / scratchy-layers; only the metal-specific
// upload/runtime surface is owned here.
pub use scratchy_tensors::{DType, DeviceAllocator, GpuTensor, OwnedTensor, RawGpuMem, TensorView};
pub use scratchy_tensors::{ForwardCtxHandle, ForwardDeviceHandle};
pub use scratchy_tensors::{device_allocator, dtype, tensor};
// RoPE cache + rope-scaling configs are cfg-free in scratchy-layers; metal
// uses only the stream-free `*_from_gpuweights` constructors (no metal-side
// upload helper needed), so a plain re-export suffices.
pub use scratchy_layers::rotary;
pub use scratchy_layers::rotary::{
    Llama3RopeScaling, LlamaConfig, LongRopeScaling, RotaryCache, YarnRopeScaling,
};

/// Backend-neutral model weights, allocator defaulted to [`MetalAllocator`].
/// Mirrors `scratchy_target_cuda::weights` so the macro's
/// `crate::__gpu::weights::GpuWeights` resolves under metal.
pub mod weights {
    /// `GpuWeights<A = MetalAllocator>` — the historic
    /// `scratchy_target_cuda::weights::GpuWeights` path under metal.
    pub type GpuWeights<A = crate::MetalAllocator> = scratchy_layers::weights::GpuWeights<A>;
    pub use crate::weights_metal::MetalWeightsExt;
    pub use scratchy_layers::weights::UploadSrc;
}
pub use weights::GpuWeights;

// Backend-neutral pools (storage generic over the mem type) the `ForwardCtx`
// names, defaulted to the metal `PoolMem` so the macro's `crate::__gpu::{kv_cache,
// gdn_state}` paths + the worker's `scratchy_target_metal::{kv_cache, gdn_state}`
// resolve under the flipped alias.
pub mod gdn_state;
pub mod kv_cache;

// Ambient forward args (`ForwardCtx`) + the multimodal-forward surface
// (`MultimodalForward` / `PixelInput` / `EmbedPatch`) + the generic VL `Weights`
// ↔ `MultimodalForward` glue (`VisionWrapper` / `VisionArchWeights`). These name
// the per-backend `GpuDevice` / `ForwardCtx`, so they can't live in the neutral
// crates — the metal copy lives here next to the cuda copy in targets/cuda.
pub mod forward_ctx;
#[cfg(feature = "vision")]
pub mod mm_dispatch;
#[cfg(feature = "vision")]
pub mod vision_arch;
pub use forward_ctx::{EmbedPatch, ForwardCtx};
#[cfg(feature = "vision")]
pub use mm_dispatch::{MultimodalForward, PixelInput};
#[cfg(feature = "vision")]
pub use vision_arch::{VisionArchWeights, VisionWrapper};

// Layered-load helpers re-exported at the crate root so the macro's emitted
// `crate::__gpu::load_layered_*` resolves under metal. Neutral ones come from
// scratchy-layers (via the loaders module's re-export); the RmsNorm keep-dtype
// + direct-write pack + metal-only MLX-affine / NVFP4 loaders are local.
pub use loaders::{
    load_layered_embedding, load_layered_layer_norm, load_layered_layer_norm_vision,
    load_layered_linear_affine_dequant_as_dense,
    load_layered_linear_affine_dequant_concat_as_dense,
    load_layered_linear_affine_dequant_concat_as_dense_mixed, load_layered_linear_affine_quant,
    load_layered_linear_affine_quant_mixed, load_layered_linear_dense,
    load_layered_linear_dense_concat_packed, load_layered_linear_dense_vision,
    load_layered_linear_nvfp4_quant, load_layered_linear_nvfp4_quant_concat, load_layered_rms_norm,
    load_layered_rms_norm_vision,
};

/// Per-layer device-memory RAII type for the pooled caches under metal:
/// `MetalMem` carries the `MTLBuffer` + `gpuAddress` the dispatch binding needs.
pub type PoolMem = MetalMem;
/// The concrete allocator type for the metal backend (held by `GpuWeights`).
pub type BackendAllocator = MetalAllocator;
/// Metal weight loading is synchronous — no CUDA-stream concept — so the
/// macro-emitted `stream` parameter resolves to `()` under metal.
pub type CUstream = ();
/// Backend-neutral weight-load stream handle; `()` under metal (no async load).
pub type LoadStream = ();

/// Metal buffer wrapper with automatic memory management
pub struct MetalBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    size: usize,
}

impl MetalBuffer {
    pub fn new(device: &Retained<ProtocolObject<dyn MTLDevice>>, size: usize) -> Self {
        let buffer = device
            .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
            .expect("newBufferWithLength returned nil");
        Self { buffer, size }
    }

    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.buffer.contents().as_ptr()
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn metal_buffer(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_detection() {
        // `detect_device()` gates on Metal 4, so it is `Some` exactly when a
        // real Metal-4 GPU is present. On GitHub's hosted "Apple Paravirtual
        // device" (Metal 3 only) both are false; on Apple Silicon both are
        // true. Assert the invariant rather than "always Some".
        assert_eq!(detect_device().is_some(), metal4_available());
    }

    #[test]
    fn test_buffer_creation() {
        let Some(device) = detect_device() else {
            eprintln!("skipping: no Metal device (or no Metal 4 support)");
            return;
        };
        let buffer = MetalBuffer::new(&device.device, 1024);
        assert_eq!(buffer.size(), 1024);
    }
}

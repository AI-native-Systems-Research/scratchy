// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal kernel infrastructure for scratchy.
//!
//! Provides Metal device management, shader compilation, and kernel dispatch
//! primitives for Apple Silicon GPUs.

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

    /// The head dims one warp of the NAX paged attention holds of O: eight 16-dim NAX fragments.
    /// A hand-picked constant (measured on a base M5: head_dim 256 at two warps a Q-row block,
    /// 512 at four).
    pub const NAX_WARP_HEAD_DIMS: u32 = 128;

    /// The NAX (matrix-accelerator) paged attention's tile at a head dim and page
    /// (`attention_steel_nax_paged.metal`): four warps along a 64-row Q block, each a 16-row NAX
    /// fragment, two 16-key fragments of keys a step, and the head dims in slices of at most
    /// [`NAX_WARP_HEAD_DIMS`] a warp. One slice runs `attention_nax_paged`, which takes one
    /// fragment a page (16-token pages); several run `attention_nax_paged_wide` over whole-page
    /// `matmul2d`s, a step whole K-tiles of whole pages.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct NaxTile {
        bq: u32,
        bk: u32,
        wm: u32,
        wn: u32,
    }

    impl NaxTile {
        /// The NAX fragment edge: rows, columns, keys of one fragment.
        const FRAGMENT: u32 = 16;

        /// The tile at `head_dim` over `block_size`-token pages; `None` where neither kernel takes
        /// the shape (a head dim not of whole fragments a slice, or a page the kernel's steps do
        /// not tile).
        pub fn of(head_dim: u32, block_size: u32) -> Option<Self> {
            let f = Self::FRAGMENT;
            let wn = head_dim.div_ceil(NAX_WARP_HEAD_DIMS).max(1);
            let tile = Self {
                bq: 4 * f,
                bk: 2 * f,
                wm: 4,
                wn,
            };
            let slices = head_dim.is_multiple_of(f * wn);
            let pages = match wn {
                1 => block_size == f,
                _ => block_size.is_multiple_of(f) && (wn * block_size).is_multiple_of(tile.bk),
            };
            (slices && pages).then_some(tile)
        }

        /// The kernel's symbol at `dtype_tag`.
        pub fn symbol(self, dtype_tag: &str) -> &'static str {
            match (self.wn, dtype_tag) {
                (1, "bf16") => "attention_nax_paged_bf16",
                (1, _) => "attention_nax_paged_f16",
                (_, "bf16") => "attention_nax_paged_wide_bf16",
                (_, _) => "attention_nax_paged_wide_f16",
            }
        }

        /// The verify lanes' kernels at `dtype_tag` — `attention_verify_pack_q`, then the wide
        /// kernel's packed tile split its keys `NAXP_SPLITS` ways — where this tile runs the wide
        /// kernel, whose packed form alone holds a KV head's group of lanes.
        pub fn verify_symbols(self, dtype_tag: &str) -> Option<(&'static str, &'static str)> {
            (self.wn > 1).then_some(match dtype_tag {
                "bf16" => (
                    "attention_verify_pack_q_bf16",
                    "attention_nax_paged_verify_split_bf16",
                ),
                _ => (
                    "attention_verify_pack_q_f16",
                    "attention_nax_paged_verify_split_f16",
                ),
            })
        }

        /// The rope-once kernel's symbol at `dtype_tag` (its head dim and page the bake's).
        pub fn rope_once_symbol(dtype_tag: &str) -> &'static str {
            match dtype_tag {
                "bf16" => "rope_once_nax_bf16",
                _ => "rope_once_nax_f16",
            }
        }

        /// The threads its warps make.
        pub fn threads(self) -> u32 {
            self.wm * self.wn * 32
        }

        /// The Q rows a threadgroup takes: its grid's x tile.
        pub fn bq(self) -> u32 {
            self.bq
        }

        /// The tile's constants (`NAXP_TILE_*`, slots 30..=33).
        pub fn constants(self) -> [crate::specialized_pipeline_cache::ConstantValue; 4] {
            use crate::specialized_pipeline_cache::ConstantValue as C;
            [
                C::uint(30, self.bq),
                C::uint(31, self.bk),
                C::uint(32, self.wm),
                C::uint(33, self.wn),
            ]
        }
    }

    /// The head dims one warp of the simdgroup wide paged attention holds of O. A hand-picked
    /// constant, measured on an M1 Max at head_dim 512 (m 4096, 8 query heads over 2): eight
    /// slices of 64 ran 1.64 TFLOP/s against four's 1.30 and sixteen's 1.54 — every doubling of
    /// the slices halves a warp's V/Q fragment loads a key (loads, not the MMA, bound the kernel)
    /// and doubles the slice-duplicated softmax.
    pub const STEEL_WIDE_WARP_HEAD_DIMS: u32 = 64;

    /// The most head dims one warp of the classic simdgroup paged attention holds of O whole
    /// (`attention_steel_paged`): its widest instantiation. A wider head takes the wide kernel.
    pub const STEEL_WARP_O_HEAD_DIMS: u32 = 256;

    /// The simdgroup (pre-NAX) wide paged attention's tile at a head dim and page
    /// (`attention_steel_wide_paged.metal`): two 16-row Q blocks a threadgroup, the head dims in
    /// slices of [`STEEL_WIDE_WARP_HEAD_DIMS`] a warp, a step one page group (at most 32 keys) a
    /// warp. It takes the head dims one warp's whole-head O cannot hold (past
    /// [`STEEL_WARP_O_HEAD_DIMS`]).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct SteelWideTile {
        bq: u32,
        bk: u32,
        wm: u32,
        wn: u32,
    }

    impl SteelWideTile {
        /// The simdgroup MMA fragment edge.
        const FRAGMENT: u32 = 8;

        /// The tile at `head_dim` over `block_size`-token pages; `None` where the kernel does not
        /// take the shape (a head the classic kernel holds whole, a head dim not of whole
        /// fragments a slice, or a page its page groups do not tile).
        pub fn of(head_dim: u32, block_size: u32) -> Option<Self> {
            let wn = head_dim / STEEL_WIDE_WARP_HEAD_DIMS;
            let group = block_size.min(32);
            let tile = Self {
                bq: 2 * 16,
                bk: group,
                wm: 2,
                wn,
            };
            let slices =
                head_dim > STEEL_WARP_O_HEAD_DIMS && head_dim.is_multiple_of(Self::FRAGMENT * wn);
            let pages = group.is_multiple_of(Self::FRAGMENT) && block_size.is_multiple_of(group);
            (slices && pages).then_some(tile)
        }

        /// The kernel's symbol at `dtype_tag`.
        pub fn symbol(dtype_tag: &str) -> &'static str {
            match dtype_tag {
                "bf16" => "attention_steel_wide_paged_bf16",
                _ => "attention_steel_wide_paged_f16",
            }
        }

        /// The rope-once kernel's symbol at `dtype_tag` (its head dim and page the bake's).
        pub fn rope_once_symbol(dtype_tag: &str) -> &'static str {
            match dtype_tag {
                "bf16" => "rope_once_steel_wide_bf16",
                _ => "rope_once_steel_wide_f16",
            }
        }

        /// The threads its warps make.
        pub fn threads(self) -> u32 {
            self.wm * self.wn * 32
        }

        /// The Q rows a threadgroup takes: its grid's x tile.
        pub fn bq(self) -> u32 {
            self.bq
        }

        /// The tile's constants (`STEEL_TILE_*`, slots 30..=33).
        pub fn constants(self) -> [crate::specialized_pipeline_cache::ConstantValue; 4] {
            use crate::specialized_pipeline_cache::ConstantValue as C;
            [
                C::uint(30, self.bq),
                C::uint(31, self.bk),
                C::uint(32, self.wm),
                C::uint(33, self.wn),
            ]
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
pub mod chain_advance;
pub mod cpu_reference;
pub mod device;
pub mod dflash2;
/// Metal lowered from the SHARED `SubtileTape` — the same artifact every target lowers. Lives
/// here, in the target crate: target-specific lowering never lives in the shared compiler.
pub mod from_tape;
pub mod grammar_mask;
pub mod logprobs;
pub mod mtp_chain;
/// ⛔ THE METAL OP ABI — which operand a kernel writes over, which ops are in-place,
/// the source/unary constructor tables. It lived in `compiler/macros/src/metal_op_abi.rs`:
/// target ABI facts inside the shared compiler, which is the arrangement CLAUDE.md says
/// these tables must NOT be in — they are INPUTS to the shared passes, and they belong to
/// the target. It depends only on `scratchy_ir` + `scratchy_subtile`, so nothing had to
/// move with it.
pub mod op_abi;
pub mod select_rows;
// Core-facing metal runtime: the `DeviceAllocator`-trait weight-loader
// (`MetalAllocator`) and the worker-facing `GpuDevice`. The canonical metal
// allocator — distinct from `allocator::PooledBufferAllocator` (legacy
// kernel-buffer pool).
pub mod device_metal;
pub mod layers;
pub mod layers_moe;
pub mod layers_quant;
/// Every weight-loading extension trait, for an emitted `load` to glob-import: the path-syntax
/// `LinearLayer::load_dense_or_ggml(..)`, `RmsNorm::load(..)`, `FusedMoE::load(..)`, … calls it
/// emits resolve through them.
pub mod load_ops {
    pub use crate::layers::{
        EmbeddingOps as _, GatedDeltaNetOps as _, LayerNormOps as _, LinearLayerOps as _,
        MetalAffineEmbedOps as _, MetalLinearLayerOps as _, RmsNormOps as _,
    };
    pub use crate::layers_moe::{
        DeepSeekV2MoEOps as _, FusedMoEOps as _, Gemma4RouterOps as _, GptOssMoEOps as _,
        SharedFusedMoEOps as _, SwitchGluExpertsOps as _,
    };
}
// Per-accessor layered weight-load helpers the forward-compiler macro emits as
// `crate::__gpu::load_layered_*`: the neutral ones re-shared from
// scratchy-layers, the metal-divergent (RmsNorm keep-dtype, direct-write pack)
// + metal-only quant loaders defined locally.
pub mod loaders;
pub mod weights_metal;
pub use weights_metal::MetalWeightsExt;
pub mod metal_allocator;
pub mod metal_mem;
// A loaded model as metal runs it (`MetalWeights`), and the registry its loaders live in.
pub mod model;
pub use model::{MetalArchRegistration, MetalKvCodec, MetalWeights, try_load};
pub mod kv_int8;
pub mod kv_int8_heads;
pub mod mtl4_dispatch;
pub mod off_tape;
pub mod owned_metal;
pub mod quantized;
pub mod residency;
pub mod sampling;
pub mod shader_cache;
pub mod sparse_arena;
pub mod sparse_kv;
pub mod specialized_pipeline_cache;
pub mod stream;
/// The pure tape-construction layer (no objc) — shared with the
/// `#[forward]` macro, which constructs these values at expansion.
pub mod tape;
/// Apple Silicon target profiles and cost models.
pub mod targets;
pub mod vocab_suffix;

// The bake: AoT metallib compiles.
pub mod aot;

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

/// Bit 31 of a `slot_mapping` or `block_table` entry: the block stores K
/// unrotated (a span block), and attention re-ropes it on read. Kernels strip
/// it (`& 0x7FFFFFFFu`, `ATTN_BT_MASK`) before indexing.
pub const UNROTATED_BLOCK_BIT: u32 = 0x8000_0000;

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

pub use allocator::{AllocatorError, PooledBuffer, PooledBufferAllocator};
pub use device::{MetalDevice, detect_device, metal4_available};
pub use device_metal::GpuDevice;
pub use metal_allocator::MetalAllocator;
pub use metal_mem::MetalMem;
pub use owned_metal::owned_from_metal_buffer;
pub use stream::MetalStreamError;

// ── `__gpu` surface scaffolding ──────────────────────────────────────────
// Backend-neutral re-exports + aliases so the forward-compiler macro's emitted
// `crate::__gpu::…` references resolve against this crate. The neutral tensor core + layer types + RoPE
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

/// Backend-neutral model weights, allocator defaulted to [`MetalAllocator`],
/// so the macro's `crate::__gpu::weights::GpuWeights` resolves under metal.
pub mod weights {
    /// `GpuWeights<A = MetalAllocator>`.
    pub type GpuWeights<A = crate::MetalAllocator> = scratchy_layers::weights::GpuWeights<A>;
    pub use crate::weights_metal::MetalWeightsExt;
    pub use scratchy_layers::weights::{TensorRefs, UploadSrc};
}
pub use weights::GpuWeights;

// The metal kernels' recurrent-state dtype, and the backend-neutral KV pool (storage generic over
// the mem type) the `ForwardCtx` names, defaulted to the metal `PoolMem` so the macro's
// `crate::__gpu::kv_cache` path + the worker's `scratchy_target_metal::kv_cache` resolve under
// the flipped alias.
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
pub use forward_ctx::{EmbedPatch, ForwardCtx, HiddenRowsOut};
pub use interpreter::metal::ForwardReads;
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
    load_layered_linear_affine_quant_mixed, load_layered_linear_affine_quant_rows_concat,
    load_layered_linear_dense, load_layered_linear_dense_concat_packed,
    load_layered_linear_dense_vision, load_layered_linear_nvfp4_quant,
    load_layered_linear_nvfp4_quant_concat, load_layered_rms_norm, load_layered_rms_norm_vision,
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

// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Core GPU types for the scratchy inference framework.
//!
//! This crate provides the foundational types for GPU tensor management:
//! `GpuTensor` (32-byte descriptor), `TensorView` (lifetime-checked borrow),
//! `OwnedTensor` (RAII allocation), `GpuDevice` (streams + cublas + allocator),
//! and the caching allocator.
//!
//! Backend coupling: the metal / backend-less code paths route the
//! forward-compiler macro's `__gpu` alias through `scratchy-target-metal` /
//! `scratchy-layers`, so a metal build does NOT pull this crate at all (verify:
//! `cargo tree -p scratchy-cli --features metal -i scratchy-target-cuda` is
//! empty). The crate still compiles in a feature-less "core types + GPU target
//! profiles" mode WITHOUT `cuda` — the forward-compiler proc-macro reads
//! `targets::*` (cost-model profiles) from it at expansion time in cuda builds
//! without dragging in the `cudarc` runtime — so this is NOT gated to `cuda`.

// Backend-neutral tensor core. Lives in `scratchy-tensors`; re-exported
// here (modules + types) so `crate::tensor::…` and
// `scratchy_target_cuda::GpuTensor` keep resolving for in-crate and
// downstream callers.
pub use scratchy_tensors::{DType, DeviceAllocator, GpuTensor, TensorView};
pub use scratchy_tensors::{device_allocator, dtype, tensor};
// Neutral opaque handles the `ScratchyWeights::forward*` trait methods take.
// Re-exported so the worker can build them from its concrete `&ForwardCtx` /
// `&mut GpuDevice` without a direct `scratchy-tensors` dep.
pub use scratchy_tensors::{ForwardCtxHandle, ForwardDeviceHandle};

// Re-export `inventory` so the macro-emitted
// `::scratchy_target_cuda::inventory::submit!` blocks for the relocated
// `ScratchyArchRegistration` / `ScratchyMmRegistration` rows resolve without the
// consuming arch crate adding its own `inventory` dep. (The neutral
// `BackboneDumpRegistration` submission still goes through the compiler's
// `::scratchy_forward_compiler::inventory`.)
#[cfg(feature = "cuda")]
pub use inventory;

// GGML quant metadata stays here — it depends on `scratchy-quantizations`,
// which sits above the neutral tensor core.
pub mod ggml_quant;
pub use ggml_quant::{GgmlDType, GgmlStorage};

// CUDA runtime (requires CUDA toolkit).
#[cfg(feature = "cuda")]
pub mod alloc;
#[cfg(feature = "cuda")]
pub mod arena;
#[cfg(feature = "cuda")]
pub mod cpu_gpu_buf;
#[cfg(feature = "cuda")]
pub mod cublas;
#[cfg(feature = "cuda")]
pub mod cuda_allocator;
#[cfg(feature = "cuda")]
pub mod device;
#[cfg(feature = "cuda")]
pub mod driver;
#[cfg(feature = "cuda")]
pub mod gguf_loader;
// `OwnedTensor` and `RawGpuMem` are backend-neutral (lifecycle behind thunks),
// from `scratchy-tensors`. `weights` stays here — its storage/drop path is
// still cfg-mutexed.
pub use scratchy_tensors::OwnedTensor;
#[cfg(feature = "cuda")]
pub mod weights;
#[cfg(feature = "cuda")]
pub mod weights_cuda;
// Runtime tile-output table consumed by the generated interpreter
// (relocated from the compiler alongside the cuda eval body).
#[cfg(feature = "cuda")]
pub mod tile_table;
// Tile-table helpers re-exported at the crate root: the macro emits
// `::scratchy_target_cuda::view` (and `tile_ref` / `take_owned`).
#[cfg(feature = "cuda")]
pub use tile_table::{TileEntry, take_owned, tile_ref, view};
// The CUDA interpreter runtime: `InstructionEval` (the eval match) +
// `InterpreterCtx` + the `run` / `run_backbone` entry points the
// generated forward fn calls. Relocated out of the compiler so its
// cuda cfg lives in the target crate.
#[cfg(feature = "cuda")]
pub mod eval;
// `run` / `run_backbone` re-exported at the crate root: the macro emits
// `::scratchy_target_cuda::run` / `run_backbone` from the generated forward fn.
#[cfg(feature = "cuda")]
pub use eval::{InstructionEval, InterpreterCtx, run, run_backbone};
// `ForwardCtx` (the ambient-args bundle the emitted forward fn takes) +
// `EmbedPatch` (multimodal embed-splice descriptor). Cross-backend
// (cuda + metal); relocated from the compiler so the cuda eval body and
// the metal interpreter both name them locally.
#[cfg(feature = "cuda")]
pub mod forward_ctx;
#[cfg(feature = "cuda")]
pub use forward_ctx::{EmbedPatch, ForwardCtx};
// Device-runtime multimodal forward surface (`PixelInput` +
// `MultimodalForward`). Names backend-runtime types, so it lives here;
// the compiler's cfg-free dispatcher imports `MultimodalForward` for
// its MM registry.
#[cfg(feature = "cuda")]
pub mod mm_dispatch;
#[cfg(feature = "cuda")]
pub use mm_dispatch::{MultimodalForward, PixelInput};
// Generic VL/MM `Weights` ↔ `MultimodalForward` glue (relocated from
// the compiler alongside the cuda forward-execution code).
#[cfg(feature = "cuda")]
pub mod vision_arch;
#[cfg(feature = "cuda")]
pub use vision_arch::{VisionArchWeights, VisionWrapper};
// CUDA device-side vision helpers (K-pad + trace dump). Names CUDA
// runtime types, so cuda-only. Relocated from `scratchy-vision` to keep
// that crate backend-dep-free (it now depends only on CPU types).
#[cfg(feature = "cuda")]
pub mod vision_device;
#[cfg(feature = "cuda")]
pub use vision_device::pad_linear_k_to_mult8;
// Per-accessor layered weight-load helpers. Each names the concrete
// `GpuWeights` container + the cuda/metal `*Ops` upload traits, so they're
// backend code; relocated from the compiler (which can't name `GpuWeights`).
// The macro emits `::scratchy_target_cuda::load_layered_*` against them.
#[cfg(feature = "cuda")]
pub mod loaders;
// Cross-arch load/forward registry: the `ScratchyArchRegistration` /
// `try_load` text-side registry + the `ScratchyMmRegistration` /
// `MultimodalForward` MM sibling. Names the concrete `GpuWeights`, so it
// lives here (below the neutral `ScratchyWeights` trait the compiler owns).
#[cfg(feature = "cuda")]
pub mod arch_registry;
#[cfg(feature = "cuda")]
pub use arch_registry::{
    ArchTryLoadFn, MmTryLoadFn, ScratchyArchRegistration, ScratchyMmRegistration,
    resolve_mm_metadata, try_load, try_load_mm,
};
// Layered-load helpers re-exported at the crate root (the macro emits
// `::scratchy_target_cuda::load_layered_*`). Same cfg split the compiler used:
// stream-free helpers under either backend; stream/quant variants cuda-only;
// MLX-affine variants metal-only.
#[cfg(feature = "cuda")]
pub use loaders::{
    load_layered_bnb4, load_layered_bnb4_concat, load_layered_embedding_sharded,
    load_layered_fp8_block_linear, load_layered_fp8_block_linear_concat, load_layered_fp8_linear,
    load_layered_fp8_linear_concat, load_layered_linear_dense_concat,
    load_layered_linear_dense_concat_sharded, load_layered_linear_dense_concat_vision,
    load_layered_linear_dense_sharded, load_layered_marlin_linear,
    load_layered_marlin_linear_concat,
};
// Fully-neutral layered loaders (every per-item load is a neutral free fn) are
// shared from scratchy-layers so both backends resolve them; the keep-dtype
// `RmsNorm` loads + the direct-write `load_dense_concat_packed` stay per-target.
#[cfg(feature = "cuda")]
pub use alloc::{CachingAllocator, RawGpuAlloc};
#[cfg(feature = "cuda")]
pub use cuda_allocator::CudaAllocator;
#[cfg(feature = "cuda")]
pub use loaders::{
    load_layered_linear_dense_concat_packed, load_layered_rms_norm, load_layered_rms_norm_vision,
};
#[cfg(feature = "cuda")]
pub use scratchy_layers::loaders::{
    load_layered_embedding, load_layered_layer_norm, load_layered_layer_norm_vision,
    load_layered_linear_dense, load_layered_linear_dense_vision,
};
// Neutral raw device allocation (CUDA: ptr+size+free-thunk).
#[cfg(feature = "cuda")]
pub use scratchy_tensors::RawGpuMem;

/// Per-layer device-memory RAII type for the pooled caches (`KvCachePool`,
/// `GdnStatePool`): the neutral [`RawGpuMem`] (pointer + RAII free) on CUDA.
#[cfg(feature = "cuda")]
pub type PoolMem = RawGpuMem;

/// Construct a neutral [`RawGpuMem`] owning a CUDA driver allocation: its `Drop`
/// thunk releases `ptr` via [`driver::mem_free`]. The single CUDA constructor
/// for the neutral descriptor — replaces the old cfg-mutexed `RawGpuMem::new`.
///
/// # Safety
/// `ptr` must be a CUDA device allocation of at least `size` bytes, valid until
/// this value (or its `leak`ed pointer) is freed.
#[cfg(feature = "cuda")]
pub unsafe fn raw_cuda(ptr: *mut u8, size: usize) -> RawGpuMem {
    // SAFETY: caller guarantees `ptr` is a valid CUDA allocation of `size`
    // bytes; the thunk frees it exactly once on drop.
    RawGpuMem::from_parts(
        ptr,
        size,
        Some(Box::new(|p| unsafe {
            let _ = driver::mem_free(p);
        })),
    )
}

/// The concrete allocator type for the active backend. CUDA xor
/// Metal — features are mutually exclusive — so this is statically
/// determined at build time. `GpuWeights` holds one as a field, and
/// backend-specific accessors (e.g. `take_gpu_allocs` returning
/// `Vec<RawGpuMem>`) live on `impl GpuWeights` blocks gated to the
/// matching feature.
#[cfg(feature = "cuda")]
pub type BackendAllocator = CudaAllocator;
#[cfg(feature = "cuda")]
pub use cpu_gpu_buf::{CpuGpuBuf, PinnedBuf};
#[cfg(feature = "cuda")]
pub use cublas::CublasHandle;
#[cfg(feature = "cuda")]
pub use device::GpuDevice;
#[cfg(feature = "cuda")]
pub use weights::GpuWeights;

/// Re-export of `CUgraphExec` for downstream crates that need to hold
/// instantiated CUDA graph handles (scratchy-forward-compiler's piecewise runner)
/// without pulling cudarc into their own Cargo.toml.
#[cfg(feature = "cuda")]
pub use cudarc::driver::sys::CUgraphExec;
/// Re-export `cudarc::driver::sys::CUstream` at a stable path so
/// generated code (scratchy-forward-compiler, scratchy-models) doesn't have to
/// pull cudarc into its own Cargo.toml.
#[cfg(feature = "cuda")]
pub use cudarc::driver::sys::CUstream;

/// Backend-neutral weight-load stream handle threaded through the
/// cross-arch dispatcher's `ArchTryLoadFn` / `try_load` seam.
///
/// CUDA loads enqueue H2D DMAs on a real `CUstream`; Metal weight
/// loading is synchronous (no stream concept) and resolves to `()`;
/// any backend without an async load path (e.g. the Spyre host
/// loader, or a backend-less build) likewise gets the `()` no-op.
/// Defining it for every config — not just `cuda`/`metal` — is what
/// lets the dispatcher's registration metadata + fingerprint logic
/// name one neutral type and drop its backend gate, so a third
/// target registers through the SAME inventory registry instead of
/// a parallel copy.
#[cfg(feature = "cuda")]
pub type LoadStream = CUstream;
#[cfg(not(feature = "cuda"))]
pub type LoadStream = ();

#[cfg(feature = "nccl")]
pub mod nccl;
#[cfg(feature = "nccl")]
pub use nccl::{NcclGroup, NcclId};

// ---------------------------------------------------------------------------
// Kernel wrappers + layer types
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub mod attention_helpers;
#[cfg(feature = "cuda")]
pub mod cutlass;
#[cfg(all(feature = "cuda", fa3_built))]
pub mod flash_attn_3;
#[cfg(feature = "cuda")]
pub mod flashinfer;
#[cfg(feature = "cuda")]
pub mod forward_output;
// Piecewise CUDA-graph capture/replay for tp>1 decode (relocated from the
// compiler so its cuda forward-execution code lives in the target crate).
#[cfg(feature = "cuda")]
pub mod piecewise;
#[cfg(feature = "cuda")]
pub use piecewise::{
    CapturedSegment, ExpandedInstr, PiecewiseRunner, SegmentTerminator, expand_slice, expand_tape,
    run_piecewise_capture, run_piecewise_replay, segment_count,
};
#[cfg(feature = "cuda")]
pub mod ggml;
#[cfg(feature = "cuda")]
pub mod kernels;
// `kv_cache` / `gdn_state` are dual-mode: layout/sizing/bookkeeping are
// backend-neutral; the cuda-only machinery is gated inside each file.
#[cfg(feature = "cuda")]
pub mod gdn_state;
#[cfg(feature = "cuda")]
pub mod kv_cache;
// `layers` / `layers_moe` / `rotary` struct *definitions* compile without
// `cuda` (they reference only `GpuTensor`); the cudarc-using impls are
// individually `#[cfg(feature = "cuda")]`-gated inside each file.
pub mod layers;
pub mod layers_moe;
#[cfg(feature = "cuda")]
pub mod layers_quant;
pub mod rotary;

#[cfg(feature = "cuda")]
pub use forward_output::ForwardOutput;
#[cfg(feature = "cuda")]
pub use gdn_state::GdnStatePool;
#[cfg(feature = "cuda")]
pub use kv_cache::KvCachePool;
pub use layers::{
    Bnb4bitLinear, ColumnParallelLinear, Embedding, GatedDeltaNetLayer, GgmlLinear, Linear,
    LinearLayer, MarlinLinear, RmsNorm, RowParallelLinear, VocabParallelEmbedding,
};
pub use layers_moe::{DeepSeekV2MoELayer, MarlinFusedMoELayer, MarlinSharedFusedMoELayer};
pub use rotary::{Llama3RopeScaling, LlamaConfig, LongRopeScaling, RotaryCache, YarnRopeScaling};

// ---------------------------------------------------------------------------
// GPU target profiles (hardware specs + cost tables) read by the
// scratchy-forward-compiler proc-macro at expansion time.
// ---------------------------------------------------------------------------
pub mod targets;

// ---------------------------------------------------------------------------
// Serving-side CUDA worker runtime (absorbed from the dissolved
// `scratchy-serving-cuda` crate): the `Worker` impl, CUDA-graph capture,
// logits processing, quant config/weight loaders, and the TCP/NCCL
// bootstrap. `pp` (pipeline-parallel index math) and `tcp_store`
// (control-plane TCP store + NCCL-id exchange) are backend-neutral and stay
// ungated so the metal build and the non-NCCL control plane reach them.
// ---------------------------------------------------------------------------
#[cfg(feature = "cuda")]
pub mod cuda_worker;
#[cfg(feature = "cuda")]
pub mod graph;
#[cfg(feature = "cuda")]
pub mod graph_piece;
#[cfg(feature = "cuda")]
pub mod logits_processor;
#[cfg(feature = "cuda")]
pub mod model;
pub mod pp;
#[cfg(feature = "cuda")]
pub mod quant;
pub mod tcp_store;
#[cfg(feature = "cuda")]
pub mod weights_quant;

// Layers test code (depends on the cuda worker's `quant` / `weights_quant`
// modules). Built only under `cfg(test)` with the cuda backend.
#[cfg(all(feature = "cuda", test))]
#[path = "layers_tests.rs"]
mod layers_tests;

#[cfg(feature = "cuda")]
pub use cuda_worker::{CudaWorker, CudaWorkerFactory};
pub use pp::PpConfig;
pub use tcp_store::TcpControlChannel;

/// Total memory (bytes) and name of the current CUDA device.
///
/// Used to pick device-aware scheduler batch defaults (mirroring Python vLLM's
/// `EngineArgs.get_batch_defaults`). Requires a current CUDA context; returns
/// `None` if the device cannot be queried (caller falls back to base defaults).
#[cfg(feature = "cuda")]
pub fn current_device_total_bytes_and_name() -> Option<(u64, String)> {
    let (_free, total) = cudarc::driver::result::mem_get_info().ok()?;
    let dev = cudarc::driver::result::device::get(0).ok()?;
    let name = cudarc::driver::result::device::get_name(dev).ok()?;
    Some((total as u64, name))
}

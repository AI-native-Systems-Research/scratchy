// SPDX-License-Identifier: Apache-2.0
//! Runtime support types the `#[forward]`-emitted code depends on:
//! most importantly [`ForwardCtx`], the ambient-args bundle the
//! emitted forward fn takes, and the [`Instruction`] enum the
//! generated tape rows construct.
//!
//! The `#[forward]` / `#[vision_forward]` attribute macros live in
//! [`scratchy_forward_compiler_macro`] — consumer crates import them
//! directly:
//!
//! ```ignore
//! use scratchy_forward_compiler_macro::{forward, vision_forward};
//! ```
//!
//! scratchy-forward-compiler intentionally does NOT re-export the proc-macro
//! crate so it can serve as a build-time dependency of
//! scratchy-forward-compiler-macro itself, so that
//! `Implementation::fan_out` can return `Vec<Instruction>` typed at
//! proc-macro time. Re-exporting the macro would re-introduce the
//! macro→forward→macro cycle.

pub mod attack_surface;
pub mod backend_compat;
pub mod gdn_slot_allocator;
pub mod gdn_state_layout;
// `info` is the non-generic, hashable backbone view used by
// `scr model info`. The whole pipeline (BackboneDumpRegistration
// inventory + Instruction::normalize) is backend-agnostic.
pub mod info;
// The Metal interpreter (lowering pass + worker pool) lives in
// `scratchy-target-metal`; this crate no longer re-exports it. Consumers
// (the metal worker, per-arch crates, macro emissions) name
// `scratchy_target_metal::interpreter::metal::*` directly — they all depend on
// the metal target under their own `metal` feature. Severing the re-export
// drops this crate's `scratchy-target-metal` dependency, which lets the metal
// target depend UP on this compiler (for `MultimodalForward` / `EmbedPatch`).

pub use info::{
    BackboneDumpRegistration, BucketDump, NormalizedField, NormalizedStep, VariantDump,
    normalize_slice,
};

// `Instruction` is unconditional so the proc-macro can use it at
// codegen time without any backend feature. Moved to the cfg-free
// `scratchy-ir` crate; re-exported here so the macro path
// `::scratchy_forward_compiler::Instruction` keeps resolving.
pub use scratchy_ir::Instruction;
// Routed-expert per-projection bit packing for `MetalSharedFusedMoe`
// (OptiQ mixed 4/8-bit). Re-exported so the macro codegen can pack and
// the metal lowering can unpack via the same convention.
pub use scratchy_ir::{pack_moe_expert_bits, unpack_moe_expert_bits};
// `CanonicalParams` + `WeightAccessors` are cross-backend — the
// metal interpreter (`interpreter::metal::worker::resolve_weight`)
// invokes both, mirroring the cuda eval body's pattern. Both now live
// in the cfg-free `scratchy-ir` crate (`CanonicalParams`'s
// `METAL_DTYPE` / `SCALE_DTYPE` consts name the relocated
// `scratchy_tensors::{MetalDtype, ScaleDtype}` enums), so the
// re-exports are unconditional and the macro path
// `::scratchy_forward_compiler::CanonicalParams` keeps resolving.
pub use backend_compat::{BackendCompat, Cuda, Metal, Wgpu};
pub use scratchy_ir::{CanonicalParams, WeightAccessors};
// The cuda runtime entry points (`run` / `run_backbone` / `InstructionEval` /
// `InterpreterCtx`), tile-table helpers (`view` / `tile_ref` / `take_owned`),
// piecewise capture, vision-arch wrappers (`VisionWrapper` /
// `VisionArchWeights`), and the layered-load helpers (`load_layered_*`) are
// all weight-upload / device-runtime code that names the concrete
// `GpuWeights` / `GpuDevice` / `ForwardCtx`. They live in
// `scratchy-target-cuda`; the macro reaches them via `::scratchy_target_cuda::*`
// (every arch crate deps that crate directly), so the compiler no longer
// re-exports them — that re-export is what made the compiler depend UP on the
// target, the edge this crate's contract inverts.

/// One row in a per-canonical forward dispatch table. Replaces the
/// O(N×M) nested-match `pub fn forward()` + per-bucket
/// `forward_m_<N>` / `forward_backbone_m_<N>` shim fns the
/// generated code used to emit. Each canonical's
/// `static FORWARD_TABLE: &[BucketEntry<Instruction<Weights>>]`
/// describes the workload-bucket boundaries
/// (`m_min` / `m_max_excl` / `sk_min` / `sk_max_excl`), the static
/// `Instruction` slices (backbone + lm_head), and the slot-map
/// metadata that varies per bucket because the solver picks
/// different Impls per workload point (e.g. `CutlassGemmAdd` fuses
/// the residual add into the GEMM at prefill, saving a slot vs the
/// decode-bucket's separate-Add path; that shifts the terminal
/// output's slot index).
///
/// Tuple-struct so each entry renders on one line of expanded
/// source. Field order:
///
///   0 = m_min          (inclusive)
///   1 = m_max_excl     (exclusive; `u64::MAX` for the final bucket)
///   2 = sk_min         (inclusive; `0` when the model has no sk axis)
///   3 = sk_max_excl    (exclusive; `u64::MAX` when no sk axis)
///   4 = backbone       — static `Op` slice for this bucket's body
///   5 = lm_head        — static `Op` slice for this bucket's tail
///   6 = num_slots      — tile-table size for `run`/`run_backbone`
///   7 = backbone_slot  — slot `forward_backbone` returns
///   8 = terminal_slot  — slot `forward` returns (after lm_head)
// `BucketEntry` is a plain tuple struct — no backend deps. Lifted to
// `any(cuda, metal)` so the metal-side macro emits `FORWARD_TABLE`
// too and `scr model info`'s `dump()` walks it. The cuda runtime
// uses fields 6/7/8 (num_slots / backbone_slot / terminal_slot) to
// size the tile table and pick return slots; the metal pool uses
// `METAL_BUCKETS` instead, so those fields stay populated but
// untouched on metal.
pub struct BucketEntry<Op: 'static>(
    pub u64,
    pub u64,
    pub u64,
    pub u64,
    pub &'static [Op],
    pub &'static [Op],
    pub u32,
    pub u32,
    pub u32,
    /// Field 9: bucket id passed to `run_slice` for this row's
    /// backbone slice. The proc-macro emits a unique id per
    /// canonical lowered entry so the per-arch
    /// [`crate::WeightAccessors`] match can disambiguate
    /// "same op_idx, different canonical".
    pub u32,
    /// Field 10: bucket id for this row's lm_head slice. See field 9.
    pub u32,
);

/// Linear-scan bucket lookup. Falls back to `table[0]` when no row
/// matches — the smallest bucket comes first by convention, so out-
/// of-range inputs route there (matches the old
/// `_ => unsafe { forward_m_<smallest>(...) }` arm).
pub fn find_bucket<Op: 'static>(
    table: &'static [BucketEntry<Op>],
    num_tokens: u64,
    sk: u64,
) -> &'static BucketEntry<Op> {
    for e in table {
        if e.0 <= num_tokens && num_tokens < e.1 && e.2 <= sk && sk < e.3 {
            return e;
        }
    }
    &table[0]
}

/// Runtime gate for the per-op trace `Instruction::eval` opens
/// each match with. Reads `SCRATCHY_TRACE` from the environment on
/// the first call and caches the result. Set `SCRATCHY_TRACE=1`
/// (or any non-empty, non-"0" value) before launch to enable;
/// pair with `CUDA_LAUNCH_BLOCKING=1` so the trace lines align
/// with kernel completion order.
///
/// Unconditionally compiled in — the cost when disabled is one
/// atomic-load + branch per dispatched op. `Instruction<W>` carries
/// `#[derive(Debug)]` so the trace can pretty-print variants.
#[inline]
pub fn trace_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SCRATCHY_TRACE")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    })
}

/// Build a `model.layers.<layer>.<suffix>` weight-name string. The
/// emitted `Weights::load_with` body fires one of these per layered
/// accessor, per layer, per canonical (~9 accessors × 32 layers ×
/// 57 canonicals on llama). Routing through a helper fn instead
/// of inlining `format!("model.layers.{}.{suffix}", layer)` keeps
/// the expanded source one line per call site (a fn call) instead
/// of five (the post-expansion `format!` →
/// `::alloc::__export::must_use({ ::alloc::fmt::format(...) })`
/// pipeline). Returns a `String` because
/// `LinearLayer::load`/`RmsNorm::load`/etc. take `&str` and the
/// binding outlives the `&str` borrow.
#[inline]
pub fn layer_weight_path(layer: u32, suffix: &str) -> String {
    format!("model.layers.{layer}.{suffix}")
}

// The `<root>.<layer>.<suffix>` decoder + vision-tower path builders are pure
// `String` helpers in the universal leaf crate (scratchy-tensors), so the
// per-target layered loaders can name them without depending on this compiler
// crate. Re-exported here so every `scratchy_forward_compiler::…` reference
// (the codegen macro's emitted `::scratchy_forward_compiler::layer_weight_path_with_root`
// / `vision_block_weight_path`) keeps resolving unchanged.
pub use scratchy_tensors::{layer_weight_path_with_root, vision_block_weight_path};
// Allocator-erased weight-store handle the cross-arch registry fn-pointers
// take. Re-exported so the `#[forward]` macro's emitted `try_load` /
// `try_load_mm` closures name it as `::scratchy_forward_compiler::GpuWeightsHandle`
// — the arch crates dep this compiler crate but not `scratchy-tensors` directly.
pub use scratchy_tensors::GpuWeightsHandle;
// Allocator-erased handle for the Metal worker's per-forward `RuntimeBindings`,
// threaded through `MetalChainBody` so this crate's chain-body type names no
// metal-target type. Re-exported so the `#[forward]` macro's emitted
// `forward_chain_with_encoder` wraps the concrete `&RuntimeBindings` as
// `::scratchy_forward_compiler::MetalRuntimeHandle` without a direct
// `scratchy-tensors` dep on the arch crate.
pub use scratchy_tensors::MetalRuntimeHandle;

/// Deterministic hash of a `serde_json::Value` for `HfFingerprint`
/// content discrimination. Canonicalizes object key order and
/// hashes numbers as f64 bits so manifest-time and runtime produce
/// the same output regardless of how the JSON was parsed.
///
/// Used to distinguish checkpoints whose `rope_scaling` has the same
/// `type` + `max_position_embeddings` but different `short_factor` /
/// `long_factor` vectors (Phi-3.5-mini vs Phi-3-mini-128k,
/// Phi-4-mini-instruct vs Phi-4-mini-reasoning). The macro's
/// `emit_fingerprint_check` bakes the manifest's hash as a `u64`
/// literal; the executor passes the live HF config's hash via
/// `HfFingerprint::rope_scaling_hash`.
pub fn hash_json_value(v: &serde_json::Value) -> u64 {
    use std::hash::{Hash, Hasher};
    fn recurse<H: Hasher>(v: &serde_json::Value, h: &mut H) {
        match v {
            serde_json::Value::Null => 0u8.hash(h),
            serde_json::Value::Bool(b) => {
                1u8.hash(h);
                b.hash(h);
            }
            serde_json::Value::Number(n) => {
                2u8.hash(h);
                // Canonicalize as f64 bits so `10000` vs `10000.0`
                // produce the same hash across JSON parsers.
                let f = n.as_f64().unwrap_or(0.0);
                f.to_bits().hash(h);
            }
            serde_json::Value::String(s) => {
                3u8.hash(h);
                s.hash(h);
            }
            serde_json::Value::Array(arr) => {
                4u8.hash(h);
                arr.len().hash(h);
                for v in arr {
                    recurse(v, h);
                }
            }
            serde_json::Value::Object(obj) => {
                5u8.hash(h);
                let mut keys: Vec<&String> = obj.keys().collect();
                keys.sort();
                keys.len().hash(h);
                for k in keys {
                    k.hash(h);
                    recurse(&obj[k], h);
                }
            }
        }
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    recurse(v, &mut h);
    h.finish()
}

// `ForwardCtx` (ambient-args bundle the emitted forward fn takes) +
// `EmbedPatch` (multimodal embed-splice descriptor) name the backend-coupled
// `KvCachePool` / `GdnStatePool`, so they stay in `scratchy-target-cuda`. The
// `ScratchyWeights` trait below takes the neutral `ForwardCtxHandle` instead of
// `&ForwardCtx`; the macro reaches the concrete type via
// `::scratchy_target_cuda::ForwardCtx` (every arch crate deps it directly).

// ── Cross-arch dispatcher ────────────────────────────────────────
//
// Every `#[forward] fn <arch>()` macro invocation auto-emits an
// `impl ScratchyWeights for Weights` + an `inventory::submit!`
// registration, so `try_load` discovers every compiled arch
// without a hand-written central list. Adding a new arch touches
// only its source file plus a single `pub mod <arch>;` in the
// calling crate's lib.rs (Rust module system requirement).
//
// Same trait + registration + walk under both backends. The trait
// method signatures are identical character-for-character — the
// types they reference (`GpuDevice`, `OwnedTensor`, `ForwardCtx`)
// are cfg-mutex'd to their cuda or metal incarnation, so a single
// signature serves both. Per-arch `impl ScratchyWeights` bodies
// (macro-emitted) cfg-mutex'd internally where the call shape
// genuinely differs (cuda calls cuda's `forward(weights, ctx,
// device, num_tokens)`; metal goes through `MetalWorkerPool::forward`).

mod dispatcher;

#[cfg(feature = "metal")]
pub use dispatcher::{ChainStepHandle, MetalChainBody, MetalForwardFollowup};
// Backend-neutral contracts — the `ScratchyWeights` trait (its device-runtime
// forward methods take the neutral `ForwardCtxHandle` / `ForwardDeviceHandle`)
// and the pure-config `HfFingerprint` carry no cuda/metal coupling, so they're
// re-exported unconditionally; a third target (Spyre) names them without a
// backend feature.
pub use dispatcher::{BoundWeight, HfFingerprint, ScratchyWeights};

// Device-runtime multimodal forward surface (`PixelInput` / `EmbedPatch` /
// `MultimodalForward`). The trait takes the neutral `ForwardDeviceHandle`
// (the per-arch `VisionWrapper` impl recovers the concrete `&mut GpuDevice`),
// so it carries no backend coupling and lives here next to the MM registry.
pub mod mm;
#[cfg(feature = "vision")]
pub use mm::MultimodalForward;
pub use mm::{EmbedPatch, PixelInput};

// Cross-arch load registry: `ScratchyArchRegistration` / `try_load` (text-side)
// + the MM sibling `ScratchyMmRegistration` / `try_load_mm` /
// `resolve_mm_metadata`, plus the `inventory::collect!` for both. The
// fn-pointers name the allocator-erased `scratchy_tensors::GpuWeightsHandle`
// (the `try_load` wrapper is generic over the concrete `GpuWeights<A>` and
// builds the handle), so the registry holds no backend allocator and lives
// here — the `#[forward]` macro emits `::scratchy_forward_compiler::Scratchy*`
// directly, which keeps resolving after the `__gpu` alias flips to metal (a
// crate that cannot depend on this one).
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "spyre",
    feature = "vision"
))]
pub mod arch_registry;
/// SuperDSC device code as `&'static` bytes in the binary, submitted by the `#[forward]` emit after
/// `dxp_standalone` compiles it. Replaces the fingerprint→directory table and the bundle cache it
/// pointed at — nothing is cached; see the module docs.
// Text-side registry (`ScratchyArchRegistration` / `try_load`) is allocator-
// generic and names no device-runtime or vision types, so it compiles + is
// reachable under spyre too — the spyre worker fingerprint-loads through this
// SAME registry (with `GpuWeights<SpyreAllocator>`).
#[cfg(any(feature = "cuda", feature = "metal", feature = "spyre"))]
pub use arch_registry::{ArchTryLoadFn, ScratchyArchRegistration, try_load};
// Read-only KTIR bundle resolver — spyre fetches its forward bundle without
// consuming `GpuWeights` (so the worker keeps the live tensors for the runner).
#[cfg(feature = "spyre")]
pub use arch_registry::{resolve_ktir_bundle, resolve_sengraph_bundle};
// MM sibling names `scratchy-vision` (`MmMetadata`) + device-runtime types, so
// it's gated on the `vision` CAPABILITY (not a target): cuda/metal enable it
// today; spyre opts in the same way when it grows a vision path.
#[cfg(feature = "vision")]
pub use arch_registry::{MmTryLoadFn, ScratchyMmRegistration, resolve_mm_metadata, try_load_mm};
/// Re-exports of the objc2/objc2_metal types referenced by macro-emitted
/// `forward_with_metal_followup` / trait `MetalForwardFollowup` so consuming
/// crates don't need direct `objc2`/`objc2_metal` deps.
#[cfg(feature = "metal")]
pub mod metal_followup_reexports {
    pub use objc2::runtime::ProtocolObject;
    pub use objc2_metal::{MTL4ComputeCommandEncoder, MTLBuffer};
}

/// Re-export `inventory` so the macro-emitted
/// `::scratchy_forward_compiler::inventory::submit!` block for
/// [`BackboneDumpRegistration`] (the neutral `scr model info` dump
/// registry that stays in this crate) resolves without the consuming crate
/// adding its own `inventory` dep. The `GpuWeights`-coupled
/// `ScratchyArchRegistration` / `ScratchyMmRegistration` submissions now go
/// through `::scratchy_target_cuda::inventory` instead.
pub use inventory;

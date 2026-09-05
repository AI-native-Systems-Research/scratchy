// SPDX-License-Identifier: Apache-2.0
//! IBM Spyre (KTIR) backend target for scratchy — the `-Fspyre` peer of
//! `crates/targets/{cuda,metal}`.
//!
//! The Spyre AIU executes per-bundle KTIR; there is no GPU device memory in the
//! CUDA/Metal sense, and the hardware-free path runs the same KTIR on the
//! `ktir-emulator` emulator. So the platform-specific seam the rest of scratchy needs
//! from a target — "allocate device memory and put these host bytes on it" — is,
//! for Spyre, a host malloc + memcpy. [`SpyreAllocator`] provides that as the
//! host impl of [`scratchy_tensors::DeviceAllocator`], so `GpuWeights` loads
//! weights through the exact same disk→host pipeline cuda/metal use, with no GPU
//! toolchain pulled in.

pub mod allocator;
/// ⭐ THE BAKED SuperDSC BUNDLE — `scratchy-spyre-bundle`, re-exported here so ONE path names it.
///
/// The `#[forward]` macro emits against this path
/// (`::scratchy_target_spyre::bundle_code::BundleCode { … }`), so its tokens depend on the target crate
/// rather than on where the types happen to live.
pub use scratchy_spyre_bundle as bundle_code;
/// Where a launched op reads and writes KV: the shim launch loop's per-op, per-page offset
/// arithmetic, lifted out of C++. No SDK dependency and no feature gate — it is arithmetic, so it
/// builds and is tested on any host, including one with no Spyre toolchain.
pub mod fold_plan;
/// The real `fxa_*` C-ABI entrypoints, natively over `flex-rs`. `csrc/spyre_sdk_abi.cpp`
/// only still provides the two deeptools functions out of flex-rs's scope; this module
/// provides every other `fxa_*` symbol `sdk_abi.rs` declares.
#[cfg(feature = "sendnn")]
pub mod fxa_rust_abi;
pub mod ir;
/// KTIR emission — MOVED here from `scratchy-subtile` (M3): target
/// emission belongs in the target crate; the shared crate keeps only
/// the SubtileIR substrate both backends consume.
pub mod lower_subtile_tape_to_ktir;
/// Bridge 1's scratchy side: the tape in a vocabulary `deeptools` can name.
///
/// Not feature-gated: this crate already depends on `scratchy-subtile` with `superdsc` on, which is
/// what the `with_config_model` door needs, and `lower_subtile_tape_to_superdsc` imports the sibling
/// geometry door unconditionally for the same reason.
pub mod lower_subtile_tape_to_dataflow_ir;
pub mod lower_subtile_tape_to_superdsc;
mod op_abi;
pub mod superdsc_bake;
/// Path shims kept from `scratchy-subtile`'s lib so the moved lowering
/// keeps its existing `crate::…` references: the modules themselves
/// live under `ir::island` / `ir::bridge`.
pub mod tile_op {
    pub use crate::ir::island::tile_op::*;
    pub use crate::ir::island::tiled_op::TiledOp;
}
pub mod subtile_tape_to_tile_ir {
    pub use crate::ir::bridge::subtile_tape_tile_op::*;
}
/// THE FORWARD TAPE — one decode step as ordered steps rather than a function
/// body: which host kernels run, what each fills, and where the single device
/// launch sits. The LAUNCHER is the worker's (it owns the session).
pub mod forward_tape;
pub mod manifest;
/// Bundle execution on the `ktir_emulator` emulator — behind the `runner` feature
/// (it links the emulator). Emit/weight-load paths use [`manifest`] alone.
#[cfg(feature = "runner")]
pub mod runner;
/// The ONE C-ABI seam to the IBM Spyre SDK (flex device runtime + the deeptools host functions) —
/// everything above it is Rust. Behind `sendnn` (it links the SDK via `build.rs`).
#[cfg(feature = "sendnn")]
pub mod sdk_abi;
/// The worker-facing session wrapper over [`superdsc_exec`] (f32 in, f32 logits out).
#[cfg(feature = "sendnn")]
pub mod sdsc_runner;
/// IEEE-fp16 ⟷ SEN169 (DLF16) device-format conversion + the device-tile weight staging walk —
/// ported verbatim from the shim's C++ (which now calls these via `extern "C"`, same link unit).
/// Pure bit arithmetic: no SDK dependency, no feature gate, tested on any host.
pub mod sen_convert;
pub mod stage_weight;
/// The resident SuperDSC executor: allocate the bundle's program + weights + KV pool once, then
/// launch each per-op kernel against the shared resident segments. The `--target sendnn` run path.
#[cfg(feature = "sendnn")]
pub mod superdsc_exec;
/// The per-model launch wiring as spyre's OWN types, emitted by `#[forward]`
/// as a `static` — the typed replacement for parsing `manifest.json` at load.
pub mod wiring;
pub use allocator::SpyreAllocator;

// ── `__gpu` facade for the `#[forward]` macro's spyre path ──
//
// The macro aliases `crate::__gpu` to this crate under `-Fspyre` (the way it
// aliases to scratchy-target-cuda under cuda) and emits `crate::__gpu::…` for
// the neutral surface its per-arch `Weights::load` / `fingerprint_matches` name.
// We re-export the neutral pieces (NEVER another `targets/` crate — only the
// cfg-free `scratchy-layers` / `scratchy-tensors` leaves):
//   - `weights::GpuWeights<A = SpyreAllocator>` mirrors target-cuda's defaulted
//     alias, so `crate::__gpu::weights::GpuWeights` resolves with no `<A>`;
//   - `LoadStream` (= `()` host) for the load signatures;
//   - `BackendAllocator` = the host `SpyreAllocator`.
pub use scratchy_tensors::LoadStream;
/// The active backend allocator under `-Fspyre` — the host `SpyreAllocator`.
pub type BackendAllocator = SpyreAllocator;
pub mod weights {
    /// `GpuWeights` defaulted to the host `SpyreAllocator`, mirroring
    /// `scratchy_target_cuda::weights::GpuWeights<A = CudaAllocator>`.
    pub type GpuWeights<A = crate::SpyreAllocator> = scratchy_layers::weights::GpuWeights<A>;
}

// ── `crate::__gpu::…` loader surface ──
//
// The `#[forward]` macro emits one `Weights` struct + one `Weights::load` body,
// the same for every target, written entirely against `crate::__gpu::…`. spyre
// supplies that surface the same way `crates/targets/{cuda,metal}` do: the layer
// TYPES come from the neutral `scratchy-layers` / `scratchy-quantizations` /
// `scratchy-tensors` leaves; the LOADER `*Ops` traits + `load_layered_*` free fns
// are spyre's own (host, allocator-generic — no GPU kernels), living in the
// modules below. No backend is special-cased in the macro.
pub use scratchy_tensors::{DType, GpuTensor, OwnedTensor, dtype, tensor};
pub mod layers;
pub mod layers_moe;
pub mod layers_quant;
pub mod loaders;
pub mod rotary;
// Re-export the `load_layered_*` free fns at the crate root so the macro's
// `crate::__gpu::load_layered_*` calls resolve (mirrors metal's lib.rs).
pub use loaders::{
    load_layered_embedding, load_layered_fp8_linear, load_layered_layer_norm,
    load_layered_layer_norm_vision, load_layered_linear_affine_dequant_as_dense,
    load_layered_linear_affine_dequant_concat_as_dense, load_layered_linear_affine_quant,
    load_layered_linear_dense, load_layered_linear_dense_concat_packed,
    load_layered_linear_dense_vision, load_layered_linear_nvfp4_quant,
    load_layered_linear_nvfp4_quant_concat, load_layered_rms_norm, load_layered_rms_norm_vision,
};

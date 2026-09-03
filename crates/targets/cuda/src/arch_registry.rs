// SPDX-License-Identifier: Apache-2.0
//! Cross-arch load registry re-export.
//!
//! The registry (`ScratchyArchRegistration` / `try_load` + the MM sibling
//! `ScratchyMmRegistration` / `try_load_mm` / `resolve_mm_metadata`) is now in
//! the cfg-free compiler: its fn-pointers name the allocator-erased
//! `scratchy_tensors::GpuWeightsHandle`, and the generic `try_load` is
//! monomorphized over the concrete `GpuWeights<A>` at the worker call site. The
//! `#[forward]` macro submits to `::scratchy_forward_compiler::Scratchy*`
//! directly (so the `inventory::submit!` target keeps resolving after the
//! `__gpu` alias flips to metal, a crate that cannot dep the compiler).
//!
//! Re-exported here so `scratchy_target_cuda::{try_load, try_load_mm, …}` —
//! called by the serving workers — keep resolving unchanged.

pub use scratchy_forward_compiler::{
    ArchTryLoadFn, MmTryLoadFn, ScratchyArchRegistration, ScratchyMmRegistration,
    resolve_mm_metadata, try_load, try_load_mm,
};

// `MultimodalForward` / `PixelInput` are re-exported via `crate::mm_dispatch`
// (from the same compiler module); kept available here for any in-crate code
// that historically reached them through `arch_registry`.
pub use crate::mm_dispatch::{MultimodalForward, PixelInput};

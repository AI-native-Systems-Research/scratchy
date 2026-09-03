// SPDX-License-Identifier: Apache-2.0
//! Runtime tile-output table consumed by the generated interpreter.
//!
//! Today's scratchy-forward-compiler emits a fully-unrolled forward fn with one
//! `let tile_42 = ...;` per FUF tile. Output aliasing rides on Rust's
//! scope rules — `let normed = unsafe { (*delta).as_view() };` borrows
//! from the upstream's `OwnedTensor`, the borrow lives until the
//! upstream goes out of scope.
//!
//! The instruction-list interpreter doesn't have per-tile let-bindings;
//! instead each tile's output occupies a slot in this table, indexed
//! by tile id. The instruction's fields encode "read from these slot
//! ids, write to that slot id." The table is what stands in for Rust
//! scope.
//!
//! # Aliasing
//!
//! [`TileEntry::Owned`] is the storage-owning variant — equivalent to
//! a `let tile_X = OwnedTensor` binding today.
//!
//! [`TileEntry::View`] is the non-owning variant — equivalent to a
//! `let tile_Y = (*tile_X).as_view()` borrow today. The `ref_slot`
//! field points at the slot whose `OwnedTensor` actually owns the
//! storage; reads through the view resolve to that owner.
//!
//! No GPU memory is copied to materialize an alias — `View` is a
//! pointer-sized indirection that the interpreter dereferences when
//! passing arguments to a kernel call.
//!
//! # Drop discipline
//!
//! Slots are dropped via the `FREE` opcode (`opcode::FREE`), emitted
//! by the macro's drop-pass after each slot's last reader. The
//! interpreter handles `FREE` by calling `tiles[slot] = None`,
//! which:
//! - For `Owned`, drops the `OwnedTensor` and returns its GPU memory
//!   to the caching allocator.
//! - For `View`, drops the indirection (no GPU memory belongs to a
//!   view, so this is a Rust-level no-op except for clearing the
//!   slot).
//!
//! Reading a `View` whose `ref_slot` has already been dropped is a
//! programming error — the macro's drop-pass guarantees the owner
//! outlives every view of it. Debug builds assert; release builds
//! UB.
//!
//! # Why not store `TensorView<'_>` directly
//!
//! `TensorView` carries a borrow lifetime; the table can't hold
//! borrows that outlive their owners under Rust's borrow rules.
//! `View { ref_slot }` is the workaround — the indirection is
//! unchecked at compile time but checked structurally by the
//! drop-pass.

#![allow(dead_code)]

use crate::alloc::OwnedTensor;
use crate::tensor::{GpuTensor, TensorView};

/// One slot in the runtime tile table.
#[derive(Debug)]
pub enum TileEntry {
    /// Storage-owning entry. Drops the `OwnedTensor` when the slot
    /// is freed.
    Owned(OwnedTensor),

    /// Non-owning view of another slot's storage. The macro's
    /// drop-pass guarantees `ref_slot` is still live (i.e.
    /// `Some(_)`) for as long as this entry is.
    View { ref_slot: u32 },

    /// Metadata-only reshape result — equivalent to today's
    /// `let tile_X = (input).reshape(&[..])` binding. `tensor`
    /// holds the new shape/ndim with the upstream's ptr; `ref_slot`
    /// pins the slot whose `OwnedTensor` actually owns the storage,
    /// so the drop-pass keeps the underlying memory alive across
    /// every consumer of the reshape. No GPU memory is copied.
    Reshaped { ref_slot: u32, tensor: GpuTensor },
}

impl TileEntry {
    /// Produce the underlying `GpuTensor` for this entry, resolving
    /// `View` indirection through the table. Used by interpreter
    /// arms when passing arguments to kernel calls.
    ///
    /// # Safety
    ///
    /// Caller must guarantee that `tiles` is indexed by tile id and
    /// that any `View::ref_slot` chain terminates at an `Owned`
    /// entry (no view-of-view chains in practice; the macro's
    /// alias-resolution emits direct ref_slot to the owning slot).
    /// Cycles and dangling refs are programming errors.
    #[inline]
    pub fn as_gpu_tensor(&self, tiles: &[Option<TileEntry>]) -> GpuTensor {
        match self {
            Self::Owned(t) => t.as_gpu_tensor(),
            Self::View { ref_slot } => {
                let owner = tiles[*ref_slot as usize]
                    .as_ref()
                    .expect("FREE'd slot referenced via View — drop-pass invariant violated");
                // The drop-pass currently emits direct refs (no view
                // chains). If we ever introduce view-of-view, this
                // single-step resolution becomes recursive and the
                // termination guarantee shifts.
                match owner {
                    Self::Owned(t) => t.as_gpu_tensor(),
                    Self::Reshaped { tensor, .. } => *tensor,
                    Self::View { .. } => panic!(
                        "view chain not supported: alias-resolution should always point at the \
                         owning slot directly"
                    ),
                }
            }
            // Reshape carries its own `GpuTensor` metadata (new
            // shape/ndim, same ptr as `ref_slot`'s owner). The
            // ref_slot lives only to keep the storage owner from
            // being freed; reads return the reshaped metadata
            // directly.
            Self::Reshaped { tensor, .. } => *tensor,
        }
    }

    /// Borrow as a `TensorView` for kernels that take views.
    ///
    /// Returns `TensorView<'static>` so the view doesn't carry a
    /// borrow on `tiles`. `TensorView` is a Copy wrapper around a
    /// raw GPU pointer (`GpuTensor`); the lifetime parameter on
    /// `TensorView` is a marker, not a real borrow. Tying the view
    /// to `tiles`'s lifetime would block any arm that wants to
    /// write back into `__tiles` after a kernel call (E0502 in
    /// every interpreter arm that produces a `Reshaped` slot
    /// from view-derived metadata, e.g. `RopeAppendRefImpl`).
    ///
    /// # Safety
    ///
    /// Same as [`as_gpu_tensor`] — caller guarantees the entry is
    /// still live (the drop-pass invariant: every alias's upstream
    /// `OwnedTensor` outlives every consumer) and any indirection
    /// resolves cleanly. The `'static` return is unsafe; a caller
    /// holding the view past the upstream's drop is a bug.
    #[inline]
    pub unsafe fn as_view(&self, tiles: &[Option<TileEntry>]) -> TensorView<'static> {
        unsafe { TensorView::from_raw(self.as_gpu_tensor(tiles)) }
    }
}

/// One-shot constructor for the alias-prelude rows the codegen emits
/// at the top of every per-bucket forward fn. Lets the macro write
/// `__tiles[dst] = Some(view(src));` on a single line, instead of
/// `Some(TileEntry::View { ref_slot: src })`.
#[inline]
pub fn view(ref_slot: u32) -> TileEntry {
    TileEntry::View { ref_slot }
}

/// Convenience accessor used by generated interpreter arms.
///
/// Reads slot `idx` from `tiles`, panics with a clear message if
/// the slot was never written or has been freed. Generated arms
/// use this rather than open-coding the unwrap so the panic site
/// has a stable name in profiles + backtraces.
#[inline]
pub fn tile_ref(tiles: &[Option<TileEntry>], idx: u32) -> &TileEntry {
    tiles[idx as usize]
        .as_ref()
        .unwrap_or_else(|| panic!("tile slot {idx} read before write or after free"))
}

/// Move the `OwnedTensor` out of slot `idx`, leaving the slot
/// empty. Used by interpreter arms whose kernel consumes its
/// upstream — in-place mutators like `scale_inplace`,
/// `tanh_softcap_inplace`, `fused_add_rms_norm_inplace` mutate
/// the upstream buffer and rebind it as the kernel's output.
/// The codegen drop-pass already excludes consumed upstreams
/// from `Free` instructions (see
/// `Implementation::consumes_input_tiles`), so this is the
/// runtime mirror: the consume pattern is "take, mutate,
/// reinsert".
///
/// Panics if the slot is empty (programming error: the consume
/// declaration should have ensured the upstream is alive at
/// this point) or holds a `View` (an aliased entry doesn't own
/// its storage; in-place consume requires actual ownership).
#[inline]
pub fn take_owned(tiles: &mut [Option<TileEntry>], idx: u32) -> OwnedTensor {
    match tiles[idx as usize].take() {
        Some(TileEntry::Owned(t)) => t,
        Some(entry @ TileEntry::View { .. }) | Some(entry @ TileEntry::Reshaped { .. }) => {
            // Put it back so the panic message can reflect the
            // pre-take state — the slot must be Owned for
            // consume to be sound.
            tiles[idx as usize] = Some(entry);
            panic!(
                "tile slot {idx} is a non-owning entry but consume requires Owned — \
                 Impl::consumes_input_tiles must only point at slots whose producer \
                 wrote `TileEntry::Owned`"
            );
        }
        None => panic!(
            "tile slot {idx} consumed but empty — drop-pass invariant \
             violated (consumed slot must outlive its consumer)"
        ),
    }
}

#[cfg(test)]
mod tests {
    // Tests for the table semantics belong with a real OwnedTensor
    // available — i.e. as integration tests under a feature-gated
    // GPU harness, not unit tests here. Keeping the unit-test stub
    // so future feature-gated tests have a home.
}

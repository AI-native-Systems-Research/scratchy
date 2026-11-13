// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ THE OPERAND WINDOW — which tensor, where in it, how much of it, minted as ONE value.

use super::{DevOff, View};
use crate::sdsc_abstract::{KindTag, Stk};

/// A checked 2-D operand window: the [`Stk`] handle (identity + kind + the ONE stick layout), the
/// minted element offset of its corner, and the row/column counts the op sweeps from there.
///
/// ⛔ WHY THIS TYPE EXISTS. A [`DevOff`] proves an offset came from a coordinate law; it does NOT
/// prove that the op sweeping `rows × cols` elements from it stays inside the tensor the offset
/// names. Until now the name and the offset travelled SEPARATELY — the builder took
/// `(&Stk, DevOff)`, split them into `name` + raw `u32` at the opspec boundary, re-paired them by
/// convention inside the descriptor layer, and the only extent check came at bake
/// (`resolve_seg_base`), reconstructing the window from the built descriptor's own dims. That is
/// the seam the card walked through twice in one week: an offset from one law applied to an
/// operand named by another (the windowed matmul reading 1/8 past its own tensor), and a pad law
/// sweeping 512 columns out of a 64-column window. Both were EXTENT-side — nothing between the
/// mint of the offset and the bake could refuse them.
///
/// A `Win` closes the class by construction: identity, offset and extents are minted together and
/// checked against the tensor's extent AT THE MINT, by the authority that owns that extent. There
/// is no raw constructor, no `From<(u32, u32, u32)>`, no field access — the rules [`DevOff`] holds
/// for offsets, held for the (identity, offset, extent) triple. The kind tag rides the type, so an
/// op builder taking `Win<KernelTag>` cannot be handed a `RowBlocked` window.
///
/// ⭐ ONE UNIT: ELEMENTS. The ×`word_length` conversion happens exactly once, at the descriptor
/// edge (`TensorArg::with_offset`), never at a mint. The mints convert against the handle's own
/// `df`, so an fp8 operand cannot be checked at fp16 width by forgetting a `*2`.
///
/// ⭐ TWO-D, LIKE EVERYTHING THAT CONSUMES IT. The swept extents are (`rows`, `cols`) — the
/// vocabulary every family already speaks (`RowCount`/`BlockCols`, `MatM`/`MatN`) — because a
/// matmul operand window is a row window AND a column window, and the pad law that decides a
/// device width must be able to consult `cols()` so it can never pad past what the tensor holds.
///
/// ⭐ THE MINTS (both checked, both authority-backed — there is no third):
/// - [`View::win`] — the SHAPE authority (the nest that declared the tensor): the layout-free door
///   every lowering that has no [`BundleLayout`](crate::placement::BundleLayout) mints through.
///   Checks the window against the nest's own element count.
/// - [`View::win_at`] — the STRICT shape mint: the corner comes from THIS view (per-axis starts),
///   so each axis's `start + len` is checked against the nest's own extent for that axis, not just
///   the flat total.
/// - [`BundleLayout::win`](crate::placement::BundleLayout::win) — the PLACEMENT authority: checks
///   the window against the tensor's declared shape AND its placement footprint, resolving the
///   name exactly as the bake-time guard does, so the mint and the guard can never disagree about
///   where a tensor ends.
///
/// A mint refusal is an expansion-time `Err` — a `cargo build` failure. "This stuff cannot
/// compile."
///
/// KIND guard — a builder that requires a `RowBlocked` A refuses a `Kernel` window by TYPE:
/// ```compile_fail
/// use ktir_superdsc::addr::{DevOff, Nest};
/// use ktir_superdsc::sdsc_abstract::{KernelTag, RowBlockedTag, Stk, StickLayout};
/// use ktir_superdsc::superdsc_opspec::Df;
///
/// fn matmul_a(_: Win<'_, RowBlockedTag>) {}
/// let nest = Nest::new(&["row", "feat"], &[8, 2048], Df::Fp16);
/// let w = Stk::<KernelTag>::new("t1", StickLayout::kernel(2048, 16)).unwrap();
/// matmul_a(nest.view().win(&w, DevOff::ZERO, 8, 16, "doctest").unwrap());
/// // E0308: expected `Win<'_, RowBlockedTag>`, found `Win<'_, KernelTag>`
/// ```
#[derive(Clone, Copy)]
pub struct Win<'a, K: KindTag> {
    stk: &'a Stk<K>,
    off: DevOff,
    rows: u32,
    cols: u32,
}

/// The mint facts, not the phantom — a `Win` prints as the checked window it is. (Deliberately
/// NOT derived: a derive would demand `K: Debug` from every kind tag for a parameter the type
/// never stores a value of.)
impl<K: KindTag> std::fmt::Debug for Win<'_, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Win")
            .field("name", &self.stk.name())
            .field("off", &self.off.into_raw_elems())
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .finish()
    }
}

impl<'a, K: KindTag> Win<'a, K> {
    /// ⛔ THE CONSTRUCTION SEAM — `pub(crate)`, for the authority mints only
    /// ([`View::win`], [`View::win_at`], and
    /// [`BundleLayout::win`](crate::placement::BundleLayout::win)). Every check lives in a mint;
    /// this assembles the already-checked triple. A new mint site is an architecture change, not
    /// a refactor: it must state its authority and its check where this crate's mints do.
    pub(crate) fn minted(stk: &'a Stk<K>, off: DevOff, rows: u32, cols: u32) -> Self {
        Win {
            stk,
            off,
            rows,
            cols,
        }
    }

    /// The minted element offset of the window's corner — read-only; a `Win` is a checked fact,
    /// not a mutable view.
    pub fn off(&self) -> DevOff {
        self.off
    }

    /// The row count the op sweeps from `off`.
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// The column count the op sweeps from `off` — the number a device-width pad law must consult
    /// so it can never pad past what the tensor holds.
    pub fn cols(&self) -> u32 {
        self.cols
    }

    /// The swept element count — `rows × cols`, the quantity every mint checked (as an element
    /// bound; the byte bound is the mint's own conversion at the handle's `df`).
    pub fn elems(&self) -> u64 {
        self.rows as u64 * self.cols as u64
    }

    /// The operand's emitter name — the handle's own, the one the descriptor layer needs.
    pub fn name(&self) -> &str {
        self.stk.name()
    }

    /// The handle the window was minted over — kind tag and the ONE stick layout travel with it.
    pub fn stk(&self) -> &'a Stk<K> {
        self.stk
    }
}

impl<'a> View<'a> {
    /// ⭐ THE SHAPE-AUTHORITY MINT — [`Win`]'s layout-free door.
    ///
    /// Checks the window against THIS view's nest — the same authority that minted the offset —
    /// so `off + rows·cols` past the tensor's declared element count is an `Err` here, at
    /// expansion time, not a read past the tensor's end on the card. Wherever a
    /// [`BundleLayout`](crate::placement::BundleLayout) IS at hand, prefer its
    /// [`win`](crate::placement::BundleLayout::win): it checks the placement footprint too.
    ///
    /// The offset arrives as a parameter (a corner minted by ANY law — `col_of`, `rc_of`, an
    /// `OperandPlacement`'s row law), so this door cannot force the corner through this view; the
    /// check is the flat element bound. Use [`Self::win_at`] when the corner IS this view's — it
    /// checks each axis's `start + len` separately.
    ///
    /// `what` names the computation the window is for (op + operand role); the refusal carries
    /// it, so a failed build names the defect, not just the number.
    pub fn win<K: KindTag>(
        &self,
        stk: &'a Stk<K>,
        off: DevOff,
        rows: u32,
        cols: u32,
        what: &str,
    ) -> Result<Win<'a, K>, String> {
        let off_elems = off.into_raw_elems() as u64;
        let end = off_elems + rows as u64 * cols as u64;
        let shape = self.nest.elems() as u64;
        if end > shape {
            return Err(format!(
                "{what}: the operand window [{off_elems}, {end}) (elements) of '{}' runs past the \
                 end its nest declares — {shape} elements ({:?}). An op that sweeps past its own \
                 tensor is a defect in the computation that produced this offset; it cannot \
                 compile.",
                stk.name(),
                self.nest.extents,
            ));
        }
        Ok(Win::minted(stk, off, rows, cols))
    }

    /// ⭐ THE STRICT SHAPE MINT — the corner comes from THIS view, so each axis is checked alone.
    ///
    /// The window is the LAST TWO axes of the nest (rows on the second-to-last, columns on the
    /// stick axis): a `[row, feat]` nest windows rows × feats, a `[head, row, feat]` nest windows
    /// one head's rows × feats. Each axis's `corner + len` is refused past that axis's own
    /// extent — which catches a window that fits the FLAT element count but straddles a row
    /// boundary: legal bytes, wrong elements, the "clean bake, no fault" shape. The flat bound is
    /// checked too.
    ///
    /// The corner's BLOCK displacement (`Nest::block`) is the caller's block law and is not
    /// counted against the nest — the same rule [`View::off`]'s own bound follows.
    pub fn win_at<K: KindTag>(
        &self,
        stk: &'a Stk<K>,
        rows: u32,
        cols: u32,
        what: &str,
    ) -> Result<Win<'a, K>, String> {
        let rank = self.nest.names.len();
        if rank < 2 {
            return Err(format!(
                "{what}: win_at over a rank-{rank} nest — the 2-D window law needs a rows axis \
                 under a columns axis. State the window with `win` and an offset instead."
            ));
        }
        let (r_pos, c_pos) = (rank - 2, rank - 1);
        let (r_start, c_start) = (self.corner[r_pos], self.corner[c_pos]);
        let (r_ext, c_ext) = (
            self.nest.extents[r_pos] as u64,
            self.nest.extents[c_pos] as u64,
        );
        if r_start as u64 + rows as u64 > r_ext || c_start as u64 + cols as u64 > c_ext {
            return Err(format!(
                "{what}: the operand window rows [{r_start}, {}) × columns [{c_start}, {}) of \
                 '{}' runs past that axis's own extent — the nest declares extents {:?}. A \
                 window that straddles a row boundary reads another row's bytes as this one's \
                 elements; it cannot compile.",
                r_start as u64 + rows as u64,
                c_start as u64 + cols as u64,
                stk.name(),
                self.nest.extents,
            ));
        }
        // The flat bound on the CORNER (the block displacement is the caller's block law, not
        // counted — the same rule [`View::off`]'s own bound follows). Implied by the per-axis
        // bound for a contiguous window; checked anyway so the two laws can never silently
        // disagree. The minted offset is the full device offset, block term included.
        let corner_elems = self.off() as u64;
        let end = corner_elems + rows as u64 * cols as u64;
        let shape = self.nest.elems() as u64;
        if end > shape {
            return Err(format!(
                "{what}: the operand window [{corner_elems}, {end}) (elements) of '{}' runs past \
                 the end its nest declares — {shape} elements (extents {:?}).",
                stk.name(),
                self.nest.extents,
            ));
        }
        Ok(Win::minted(stk, self.dev(), rows, cols))
    }
}

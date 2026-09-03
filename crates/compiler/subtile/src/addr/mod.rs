// SPDX-License-Identifier: Apache-2.0
//! THE address law, once — and composable views that defer every `+` and `*` to one place.
//!
//! Every addressing bug this backend has shipped is the same shape: a hand-written multiplier at a
//! call site (`h*mq*hd`, `b*stick*hd`, `slot*hd + d`, `inn*hd + o`, `(d/64)*(64*64)`) that is correct
//! only because some axis happened to be 1. An fp16 stick is 64 elements, so at `head_dim == 64` the
//! `(c/eps)` factor is always 0 and at decode `m == 1` the row factor is always 0 — two of the three
//! non-lane axes have been pinned for this backend's whole life, so a missing multiplier by either
//! is invisible on every model anyone has run.
//!
//! The law itself is NOT new and is NOT reimplemented here. `SpyreTensorLayout::with_dim_order`
//! (`sdsc_abstract.rs`) is the ported dxp rule, and every existing formula is that rule at a
//! different rank with different extents — proven element-by-element in `tests/addr_one_law.rs`:
//!
//! | declared order (storage extents) | equals |
//! |---|---|
//! | `[row, col]` `[rows, cols]`        | `dev_off_stk` rank-2 |
//! | `[row, outer, col]` `[mq, nqh, hd]`| `dev_off_head` |
//! | `[slot, qhead, dim]` `[cap, nqh, hd]` | `vcache_write_offset` (+ head base) |
//! | `[dim, kvhead, slot]` `[hd, nkvh, cap]` | `kcache_kt_write_offset` (+ head base) |
//!
//! So `StickKind`'s variants are one nest at different ranks. This module makes that the *only* way
//! to get an address: declare a tensor's axes and its OWN storage extents once, then narrow by
//! semantic coordinate. A caller never writes a stride, so it cannot omit one.

pub mod shape;
pub use shape::{Gqa, PadRows, Rows, Shape, Slabs};

use crate::sdsc_abstract::{ElementArrangement, SpyreTensorLayout};
use crate::superdsc_opspec::Df;

/// A semantic axis. Identity is compile-time (these are distinct types), extent is runtime — `cap`,
/// `mq` and `active_cap` genuinely vary per bucket, so const-generic extents would monomorphise the
/// emitter per ladder rung and fork decode from prefill. What must NOT vary silently is which axis
/// you are indexing, and that is what the type pins.
pub trait Axis: Copy + 'static {
    const NAME: &'static str;
}
macro_rules! axes {
    ($($t:ident => $n:literal),* $(,)?) => { $(
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub struct $t;
        impl Axis for $t { const NAME: &'static str = $n; }
    )* };
}
axes! {
    Row   => "row",    // mq / mq_pad — query rows
    Head  => "head",   // nqh / nkvh
    Slot  => "slot",   // cap — KV cache depth
    Feat  => "feat",   // hd / hidden / intermediate — the stick axis
}

/// An INDEX along one axis. `Idx<Head>` and `Idx<Slot>` are different types, so they cannot be
/// swapped or multiplied together into an "offset".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Idx<A: Axis>(u32, std::marker::PhantomData<fn() -> A>);
impl<A: Axis> Idx<A> {
    pub const fn n(v: u32) -> Self {
        Idx(v, std::marker::PhantomData)
    }
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// A tensor's DECLARED storage: its axis order (outermost first, stick axis LAST) and its own
/// extents. Not the op's iteration space — that distinction is exactly what B6 got wrong, reading a
/// 64-slot window of a `cap`-deep cache with the window's depth as the stride.
#[derive(Clone, Debug)]
pub struct Nest {
    /// Row-major (no stick blocking) — see [`Nest::flat`]. The device slot model below does not apply.
    row_major: bool,
    names: Vec<&'static str>,
    extents: Vec<i64>,
    df: Df,
    strides: Vec<i64>,
    stick_pos: usize,
}

impl Nest {
    /// Declare a tensor. `axes` are host axis names outermost-first with the STICK axis last;
    /// `extents` are that tensor's OWN storage extents. The device nest is derived by calling the
    /// ported dxp law — never re-transcribed.
    /// A ROW-MAJOR nest — the device's flat arrangement, where element `(r, c)` is at `r*cols + c`
    /// with no stick blocking. Most device tensors are stick-blocked and must use [`new`](Self::new);
    /// this is for the ones the emitter genuinely places flat, and it exists so those sites can still
    /// name a coordinate instead of writing `r*cols + c` and leaving a reader to guess which
    /// arrangement was meant. Getting that guess wrong in either direction is the bug class this
    /// module is here to remove, so the two forms should be equally easy to say.
    pub fn flat(axes: &[&'static str], extents: &[u32], df: Df) -> Self {
        let host: Vec<i64> = extents.iter().map(|&e| e as i64).collect();
        let order: Vec<i32> = (0..axes.len() as i32).collect();
        let _ = order;
        // Row-major: each axis strides by the product of the axes to its right. No stick plane.
        let mut strides = vec![1i64; host.len()];
        let mut acc = 1i64;
        for i in (0..host.len()).rev() {
            strides[i] = acc;
            acc *= host[i];
        }
        Nest {
            row_major: true,
            names: axes.to_vec(),
            extents: host,
            df,
            strides,
            stick_pos: axes.len() - 1,
        }
    }

    pub fn new(axes: &[&'static str], extents: &[u32], df: Df) -> Self {
        assert_eq!(
            axes.len(),
            extents.len(),
            "addr::Nest: axes/extents rank mismatch"
        );
        // POSITION decides the device slot, not the NAME: axis 0 takes the `rows` slot, the last axis
        // is the stick, and anything between is outer. The names were decorative, which let a nest be
        // written in an order that reads correctly and addresses wrongly — `["head","row","feat"]` puts
        // HEAD in the row slot and gives it a stride of one lane. I wrote exactly that twice this week,
        // for the online-softmax buffers and for `new_kt`, and only the bake caught it. So if `row`
        // appears at all it must be where the law will treat it as the row.
        if let Some(pos) = axes.iter().position(|&a| a == "row") {
            assert_eq!(
                pos, 0,
                "addr::Nest {axes:?}: `row` is at index {pos}, but the law gives the ROW slot to axis \
                 0 — as written, `{}` would be strided as the row and `row` as an outer axis. Reorder \
                 to put `row` first, or use `block()` if the leading axis is really a repeat.",
                axes[0]
            );
        }
        // (No constraint on WHICH axis is last: the stick axis is whatever the tensor sticks on —
        // `feat` for an activation, `slot` for the transposed K caches.)
        let host: Vec<i64> = extents.iter().map(|&e| e as i64).collect();
        let order: Vec<i32> = (0..axes.len() as i32).collect();
        let l = SpyreTensorLayout::with_dim_order(&host, df, &order, ElementArrangement::Standard);
        Nest {
            row_major: false,
            names: axes.to_vec(),
            extents: host,
            df,
            strides: l.dense_strides(),
            stick_pos: axes.len() - 1,
        }
    }

    pub fn df(&self) -> Df {
        self.df
    }
    pub fn lanes(&self) -> i64 {
        self.df.elems_per_stick() as i64
    }
    /// Total elements — the footprint any address must stay inside.
    pub fn elems(&self) -> i64 {
        self.extents.iter().product()
    }

    /// THE evaluation point. This is the only place in the module where `+` and `*` build an
    /// address; every caller composes coordinates and lands here exactly once.
    ///
    /// The device nest is `[.. , stick_groups, rows, lanes]`: the leading host axis takes the "rows"
    /// slot, the remaining outer axes stay outermost, and the stick axis splits into
    /// (group, lane). `dense_strides` supplies each slot's stride, so no multiplier is written here
    /// either — the strides come from the law.
    pub fn off(&self, corner: &[u32]) -> i64 {
        assert_eq!(
            corner.len(),
            self.names.len(),
            "addr::Nest::off: corner rank mismatch"
        );
        if self.row_major {
            // No stick plane: each axis simply strides by the product of those to its right.
            return corner
                .iter()
                .zip(&self.strides)
                .map(|(&v, &s)| v as i64 * s)
                .sum();
        }
        let eps = self.lanes();
        let n = self.strides.len();
        let c = corner[self.stick_pos] as i64;
        // device slots: [outer axes.., stick_group, row, lane]
        let mut off =
            (c / eps) * self.strides[n - 3] + (corner[0] as i64) * self.strides[n - 2] + (c % eps);
        for (i, &v) in corner.iter().enumerate().take(self.stick_pos).skip(1) {
            // outer axes occupy the leading device slots, in declared order
            off += (v as i64) * self.strides[i - 1];
        }
        off
    }

    /// A view of the whole tensor.
    /// The `i`-th REPEAT of this nest, laid end to end. Several device tensors are `n` independent
    /// blocks of one shape — `vc` is nqh separate `[cap, hd]`, `new_kt` is nkvh separate
    /// `[hd, mq_pad]` — and describing them as one rank-3 nest is WRONG: the law would give the
    /// outer axis a stride of one stick rather than the block's footprint. This says what is meant,
    /// and takes the block size from the nest instead of a hand-written product.
    pub fn block(&self, i: u32) -> View<'_> {
        let mut v = self.view();
        v.base = i as i64 * self.elems();
        v
    }

    pub fn view(&self) -> View<'_> {
        View {
            nest: self,
            corner: vec![0; self.names.len()],
            base: 0,
        }
    }

    /// ⭐⭐⭐ THE ELEMENTS BETWEEN TWO POSITIONS ON ONE NAMED AXIS — evaluated by [`Nest::off`], like
    /// every other address in this module.
    ///
    /// ⛔ THIS EXISTS SO NO CALLER WRITES A STRIDE AS A PRODUCT. "How far apart are two heads" is the
    /// question every batched op has to answer, and written by hand it is `mq*hd` for the token stream
    /// and `mq*lanes` for a one-stick buffer — two expressions of DIFFERENT units that are EQUAL at
    /// `hd == lanes` and nowhere else. That is the whole head_dim-64 assumption, and it is unwritable
    /// here: the stride comes from `dense_strides` (the ported dxp law) through the same evaluation
    /// point the offsets do, so a nest cannot report a stride that disagrees with the addresses it
    /// hands out.
    ///
    /// The axis is a TYPE, so one nest's head stride and its row stride cannot be confused.
    pub fn span<A: Axis>(&self, lo: Idx<A>, hi: Idx<A>) -> AxisStride {
        let at = |i: Idx<A>| self.view().at(i).off();
        AxisStride {
            elems: at(hi) - at(lo),
            lanes: self.lanes(),
        }
    }

    /// This nest's own leading-axis extent, in rows — the PITCH half of a plane-walked placement
    /// ([`crate::sdsc_abstract::OperandPlacement::of_plane_walk_by_head`]). Asked of the nest so the
    /// rows a plane is packed with cannot be supplied separately from the nest whose planes they are.
    pub fn rows(&self) -> u32 {
        self.extents[0] as u32
    }

    /// The footprint of ONE BLOCK of this nest — the distance [`Nest::block`] steps, as a stride
    /// rather than a count, so a caller can ask "how far apart are two blocks" through the same law
    /// that placed them instead of re-deriving `elems()` as a product.
    pub fn block_span(&self) -> AxisStride {
        AxisStride {
            elems: self.elems(),
            lanes: self.lanes(),
        }
    }
}

/// ⭐ A DISTANCE ALONG ONE AXIS, in device elements — minted only by [`Nest::span`], so it is always
/// the law's answer and never a caller's product.
///
/// Not an offset (it is not a position) and not a count (it sizes nothing): it is what a declared walk
/// must step to advance one position on that axis. The distinction is load-bearing — a placement's
/// head PITCH (in rows) and its head STRIDE (in elements) differ by the stick, and on an `hd`-wide
/// buffer by the slab count too, so those were exactly the pair that got substituted at `hd == 64`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AxisStride {
    elems: i64,
    lanes: i64,
}

impl AxisStride {
    /// The distance itself, in device elements.
    pub fn elems(self) -> u32 {
        self.elems as u32
    }

    /// This stride expressed as ONE-STICK ROWS — the `mb` DEVICE extent a walk that iterates one
    /// stick of its stick axis must declare to advance one position on this axis. The division lives
    /// here, next to `lanes`, rather than at an emitter site writing `stride / 64`. `None` when the
    /// stride is not a whole number of sticks: such a buffer cannot be addressed by a one-stick walk
    /// at all, and the caller must refuse rather than round.
    pub fn one_stick_rows(self) -> Option<u32> {
        (self.lanes > 0 && self.elems % self.lanes == 0).then(|| (self.elems / self.lanes) as u32)
    }
}

/// A narrowed region of a tensor. Each `at` fixes one axis; nothing is multiplied until `off()`.
#[derive(Clone, Debug)]
pub struct View<'a> {
    /// Whole-block displacement from [`Nest::block`]; 0 for a plain view.
    base: i64,
    nest: &'a Nest,
    corner: Vec<u32>,
}

impl<'a> View<'a> {
    /// Fix one axis by NAME + typed index. Composable: the result is another `View`.
    pub fn at<A: Axis>(mut self, i: Idx<A>) -> Self {
        let pos = self
            .nest
            .names
            .iter()
            .position(|&n| n == A::NAME)
            .unwrap_or_else(|| panic!("addr: axis {} not in {:?}", A::NAME, self.nest.names));
        self.corner[pos] = i.get();
        self
    }

    /// Fix the STICK axis to the start of slab `s` — the `s * lanes` multiplier lives HERE, in the
    /// primitive that owns `lanes`, instead of at each call site. `n_slabs` loops are the single most
    /// common place a head_dim > 64 multiplier goes missing (`s * mq * stick`, `s * cap * stick`,
    /// `b * stick * hd`), because at one stick every one of them is 0.
    pub fn slab(mut self, s: u32) -> Self {
        let pos = self.nest.stick_pos;
        self.corner[pos] = s * self.nest.lanes() as u32;
        self
    }

    /// Fix a NAMED axis to the start of its slab `s` — [`Self::slab`] for an axis that is NOT the
    /// stick axis, with the `s * lanes` multiplier in the primitive that owns `lanes` for the same
    /// reason.
    ///
    /// ⛔ THIS IS THE SCORE LEG'S CONTRACTION SPLIT. Its Kᵀ kernel is `["feat", "slot"]` sticked on
    /// `slot`, so the head-dim slab steps `feat` — a non-stick axis, which `slab` cannot reach. The
    /// alternative was `at(Idx::<Feat>::n(s * stick))` at the call site, i.e. exactly the
    /// `s * <lanes>` multiplier `slab`'s own doc exists to keep out of call sites, and exactly the
    /// shape of every hd>64 bug in this file (inert at one stick, wrong above it).
    pub fn slab_of<A: Axis>(mut self, s: u32) -> Self {
        let pos = self
            .nest
            .names
            .iter()
            .position(|&n| n == A::NAME)
            .unwrap_or_else(|| panic!("addr: axis {} not in {:?}", A::NAME, self.nest.names));
        self.corner[pos] = s * self.nest.lanes() as u32;
        self
    }

    /// The device element offset of this view's corner — the deferred math, finally evaluated.
    pub fn off(&self) -> i64 {
        let o = self.nest.off(&self.corner);
        debug_assert!(
            o >= 0 && o < self.nest.elems(),
            "addr: {o} outside footprint {} for nest {:?} extents {:?} corner {:?}",
            self.nest.elems(),
            self.nest.names,
            self.nest.extents,
            self.corner
        );
        o
    }

    /// The offset in BYTES, at this tensor's own format — so an fp8 tensor cannot be addressed at
    /// fp16 width by forgetting a `*2`.
    pub fn bytes(&self) -> i64 {
        self.off() * self.nest.df.word_length() as i64
    }

    /// The op-builder-facing offset. THE ONLY WAY to obtain a [`DevOff`].
    pub fn dev(&self) -> DevOff {
        DevOff((self.base + self.off()) as u32)
    }
}

/// Column offset into a rank-2 `[row, feat]` tensor — the commonest slice in the emitter (an MLP
/// column chunk, a mask block, a RoPE half). Nest-derived, so the `(c/lanes)*(rows*lanes)` term is
/// present by construction rather than by the caller remembering it.
pub fn col_of(rows: u32, cols: u32, c: u32, df: Df) -> DevOff {
    Nest::new(&["row", "feat"], &[rows, cols], df)
        .view()
        .at(Idx::<Feat>::n(c))
        .dev()
}

/// Row+column corner of a rank-2 `[row, feat]` tensor.
pub fn rc_of(rows: u32, cols: u32, r: u32, c: u32, df: Df) -> DevOff {
    Nest::new(&["row", "feat"], &[rows, cols], df)
        .view()
        .at(Idx::<Row>::n(r))
        .at(Idx::<Feat>::n(c))
        .dev()
}

/// A device element offset that an op builder will accept.
///
/// INVIOLABLE BY CONSTRUCTION: the field is private, there is no `From<u32>`, no `new`, and no
/// arithmetic impl. The only constructor in the crate is [`View::dev`], so a `DevOff` can only ever
/// be the result of narrowing a declared [`Nest`] by semantic coordinate. `b * stick * hd`,
/// `qh * cap * hd`, `h * mq * hd`, `slot * hd + d` are all `u32` expressions and none of them will
/// coerce — they stop being *wrong*, and start being *unwritable*.
///
/// That distinction is the whole point. Every fix in this area so far replaced one hand-written
/// multiplier with a different hand-written multiplier, which leaves the next axis change to
/// rediscover the same class the hard way. A missing `mq` cost weeks; a missing `hd` cost this one.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
/// A DEVICE ELEMENT OFFSET. The only ways to make one are [`DevOff::ZERO`] and [`View::dev`], so an
/// address cannot be built from arithmetic a caller wrote — it comes from a nest that knows the
/// tensor's extents, or it is the whole-tensor corner.
///
/// The escape hatch this used to carry (`from_arg_raw`) is gone. It existed to re-wrap values that
/// were already device offsets threaded through a `u32`, and it became a marker for "not yet shown to
/// come from a nest" — 72 of them at the start of this work. At zero call sites it is worth deleting
/// rather than keeping, because while it exists a new site can reach for it instead of naming a
/// coordinate, and that is exactly how every hd>64 addressing bug in this backend was written.
pub struct DevOff(u32);

impl std::ops::Add for DevOff {
    type Output = DevOff;
    /// Compose a corner with a within-view step. Both sides are device offsets by construction.
    fn add(self, rhs: DevOff) -> DevOff {
        DevOff(self.0 + rhs.0)
    }
}

impl DevOff {
    /// Zero — the whole-tensor corner. The one value that needs no nest because it is nest-independent.
    pub const ZERO: DevOff = DevOff(0);

    /// A step WITHIN a view that is already stick-aligned and single-stick wide: at one stick the
    /// plane term is identically 0, so a whole number of sticks along the last axis is just that many
    /// lanes, whatever the tensor's other extents are. This is the one displacement that needs no
    /// nest because it cannot depend on one.
    pub const fn from_view_step(elems: u32) -> DevOff {
        DevOff(elems)
    }

    /// The corner of a prefix-mask slab — an address derived by [`PrefixMaskShape`]'s own element
    /// law, the one law the emitter's READ and the worker's STAGING share. Its own door, because a
    /// mask corner is a whole plane term (`slab * rows * 64`), not the single-stick view step
    /// [`Self::from_view_step`] is contracted for.
    ///
    /// [`PrefixMaskShape`]: crate::sdsc_abstract::PrefixMaskShape
    pub const fn of_mask_corner(c: crate::sdsc_abstract::MaskCorner) -> DevOff {
        DevOff(c.elems())
    }
    /// Consume into the raw element offset. Deliberately named to be greppable: every call is a
    /// place where the typed offset re-enters untyped code, i.e. a remaining migration seam.
    pub fn into_raw_elems(self) -> u32 {
        self.0
    }
}

//! The DECLARED WALK of a matmul operand — the axis ORDER the descriptor carries
//! (`layoutDimOrder_`), which IS a stride assignment:
//!
//!   * a row-major (`Flat`-classified) tensor is walked with axis `i` striding by the
//!     product of the extents of every axis AFTER it ([`StickLayout::off_view`]'s
//!     `RowMajor` fold, and dxp's own reconstruction of a pinned `maxDimSizes_`);
//!   * a rank-2 stick-on-last tensor is stick-blocked (`RowBlocked`/`Kernel`), where the
//!     row axis strides by the stick width WITHIN each stick group and dxp reconstructs
//!     the walk from `-1` extents.
//!
//! So "which stride does `mb` get" is decided by nothing but the ORDER of two string
//! literals — a quantity with no name and no type, and the recorded mq>1 defect class
//! (the deleted hand-written attention emitter, "the two are SWAPPED"): `[mb, y, in]` assigns mb the batch BLOCK `y·in`
//! while the physical head-major operand's row pitch is `in`, numerically equal exactly
//! at `mb == 1`. This module gives every matmul walk axis a NAME ([`WalkAxis`]) and every
//! walk a CONSTRUCTOR named for the stride assignment it encodes — and the batch-inner
//! pair (`[mb, y, ·]`, the assignment that swaps at `mb > 1`) has PRIVATE constructors
//! whose only door is the proven regime's [`RungRegime`] impl, so a batch-decode rung
//! cannot be handed the walk that is right only at one row: its own regime's associated
//! order IS `[y, mb, ·]`, and the batch-inner walk is a different type.
//!
//! The stick axis of every matmul walk is its LAST axis — [`Walk2`]/[`Walk3`] have no
//! other constructor shape, so a stick-not-innermost matmul operand is unrepresentable.
//!
//! [`StickLayout::off_view`]: crate::sdsc_abstract::StickLayout::off_view

use std::marker::PhantomData;

/// ⭐ WHICH FORM a SHARED-KERNEL batched (`y > 1`, 2-D-kernel) matmul takes: the RUNG REGIME it is
/// emitted under, carried as a VALUE of that regime's sealed type. Threaded from the caller that
/// knows the bundle's row kind down through the opspec builders — no env var, no thread-local, no
/// site that infers it from a row count. The declared rank-3 walk ORDER and the work-division
/// policy are not stored here at all: both are ASSOCIATED to the regime type ([`RungRegime`]), so
/// this form can only relay them, never recombine them.
///
/// The field is private and every door names the row kind or walk it yields:
///   * [`SharedKernelBmmForm::batch_inner_proven`] — every batched matmul whose rows are not
///     head planes (the dense wrappers, the Kani-tower seam, the fp8 chain): the walk those
///     callers have always emitted. It demands a [`NonAttnBmmSite`] witness, mintable only inside
///     the matmul wrapper module and the tape lowering.
///   * [`SharedKernelBmmForm::of_head_batched`] — a matmul NEWLY `y`-batched over heads (the tape
///     lowering's RoPE rotate and its kin): head-outermost at every head dim.
///   * [`SharedKernelBmmForm::of_token_rows`] and [`SharedKernelBmmForm::of_head_outermost`] —
///     the same two regimes stated as a row kind / walk fact, for a caller outside the crate
///     driving the pub matmul wrappers directly.
///
/// The two regimes' walks are DIFFERENT TYPES ([`RungRegime`]'s associated orders), so a rung
/// handed the wrong form is an `E0308` at the emission site, not a silently crossed stride pair.
#[derive(Clone, Copy)]
pub struct SharedKernelBmmForm(SharedKernelBmmRegime);

impl SharedKernelBmmForm {
    /// A matmul whose rows are TOKEN ROWS — one request's positions, or a dense projection's
    /// plain rows: never an axis that steps whole head planes. The on-hardware-proven
    /// batch-inner walk, the same regime [`Self::batch_inner_proven`] mints in-crate for the
    /// dense wrappers, the Kani-tower seam, and the fp8 chain.
    pub fn of_token_rows() -> Self {
        SharedKernelBmmForm(SharedKernelBmmRegime::Proven(BatchInnerMq1Proven(())))
    }

    /// A matmul whose `y` axis steps WHOLE HEAD PLANES on its operands — head-major residency
    /// at every head dim, the head-OUTERMOST order `[y, mb, ·]`. The same regime
    /// [`Self::of_head_batched`] mints in-crate for the tape lowering's newly head-batched
    /// sites (RoPE's rotate and its kin).
    pub fn of_head_outermost() -> Self {
        SharedKernelBmmForm(SharedKernelBmmRegime::Requests(HeadOutermostRequests(())))
    }

    /// ⭐⭐⭐ A MATMUL THAT IS **NEWLY** `y`-BATCHED OVER HEADS — always the head-OUTERMOST order, at
    /// EVERY head dim.
    ///
    /// ⛔ NOT for a leg that was ALREADY batched at one slab — such a site's emission must not
    /// move, and it keeps the proven batch-inner form ([`Self::batch_inner_proven`]). For an op
    /// whose `y` axis is NEW, the batch-inner order cannot express a head stride at
    /// all: `[mb, y, in]` puts `mb` outside `y`, so `y` strides exactly `in`. At hd=64/mq=7 that is 64
    /// against a real head stride of `mq*hd` = 448 — the build-time refusal RoPE's rotate matmul hit on
    /// granite-3.1-2b fp8 the moment it stopped looping heads.
    ///
    /// Head-outermost derives `mb_dev * in`, which is `mq*hd` at every head dim once the operand
    /// declares its own pitch (`mq` with a full-`hd` `in`, `mq*nslab` with a one-stick `in`).
    pub(crate) fn of_head_batched(_: impl NonAttnBmmSite) -> Self {
        SharedKernelBmmForm(SharedKernelBmmRegime::Requests(HeadOutermostRequests(())))
    }

    /// Every batched matmul whose rows are not head planes: the dense/bmm wrappers, the
    /// Kani-tower seam, the fp8 chain. None of them has request rows, so none of them can be
    /// the rung the head-outermost walk exists for — they keep the walk they have always
    /// emitted. It takes a [`NonAttnBmmSite`] witness, and both witness mints live in modules
    /// the tape lowering hands a form to as a parameter — the value that parameter is fed is
    /// one minted here or by [`Self::of_head_batched`].
    pub(crate) fn batch_inner_proven(_: impl NonAttnBmmSite) -> Self {
        SharedKernelBmmForm(SharedKernelBmmRegime::Proven(BatchInnerMq1Proven(())))
    }

    /// The regime VALUE this form carries — the only exit, so the emission path can become generic
    /// over a [`RungRegime`] only by being handed one that came through a boundary door.
    pub(crate) fn regime(self) -> SharedKernelBmmRegime {
        self.0
    }
}

/// The two rung regimes a shared-kernel batched matmul can be emitted under, as VALUES of their
/// sealed types. Matching this enum is the ONE runtime dispatch; everything downstream of an arm is
/// monomorphic in that arm's regime, and the walk pair exists only as that regime's associated
/// order — an input order from one regime cannot ride with the output order of the other, because
/// no value carries them separately.
#[derive(Clone, Copy)]
pub(crate) enum SharedKernelBmmRegime {
    Proven(BatchInnerMq1Proven),
    Requests(HeadOutermostRequests),
}

/// The batch-inner `[mb, y, ·]` regime + joint cost-model split — what every shipped bundle
/// carries, on-hardware-proven for the mq==1 decode g-form. The private field seals construction
/// into this module's form doors.
#[derive(Clone, Copy)]
pub(crate) struct BatchInnerMq1Proven(());

/// The head-outermost `[y, mb, ·]` regime + batch-divides-first split — for matmuls whose `y`
/// axis steps whole head planes (the tape lowering's newly head-batched sites, such as RoPE's
/// rotate). NOT yet proven on hardware. The private field seals construction into this module's
/// form doors.
#[derive(Clone, Copy)]
pub(crate) struct HeadOutermostRequests(());

mod sealed {
    /// [`super::RungRegime`]'s implementor set is CLOSED over the two rung regimes: a third
    /// "regime" (or a re-implementation that pairs the proven split with the request walk) cannot
    /// exist outside this module, because it cannot name this supertrait.
    pub(crate) trait Sealed {}
    impl Sealed for super::BatchInnerMq1Proven {}
    impl Sealed for super::HeadOutermostRequests {}
}

mod site_sealed {
    /// [`super::NonAttnBmmSite`]'s implementor set is CLOSED over the two site witnesses below:
    /// a module cannot admit itself to the proven-form door by implementing the witness trait for
    /// a type of its own, because it cannot name this supertrait.
    pub trait Sealed {}
    impl Sealed for super::MatmulWrapperSite {}
    impl Sealed for crate::emit::bmm_site::TapeLoweringSite {}
}

/// ⭐ A CALL SITE whose batched matmuls can never carry request rows, as a WITNESS VALUE. This is
/// what [`SharedKernelBmmForm::batch_inner_proven`] charges at its door: the proven batch-inner
/// walk is only mintable by a module that holds one of these, and the two mints —
/// [`MatmulWrapperSite::witness`] (the matmul wrapper family: the untyped dense wrapper and the
/// Kani-tower seam) and `TapeLoweringSite::witness` (the tape lowering's dense/fp8/RoPE/krep/vrep
/// wrappers) — are each private to their owning module, so no other module in the crate can
/// state the proven form locally.
pub(crate) trait NonAttnBmmSite: site_sealed::Sealed {}

/// The matmul wrapper family's [`NonAttnBmmSite`]: [`super::opspec::matmul_opspec`] (the untyped
/// dense-projection wrapper) and [`super::assemble::assemble_matmul_split`] (the Kani-tower seam).
/// The private field seals construction into [`Self::witness`], whose visibility is the matmul
/// module.
pub(crate) struct MatmulWrapperSite(());

impl MatmulWrapperSite {
    /// Mintable only inside the matmul module — the wrapper sites named on the type.
    pub(in crate::ir::bridge::tiled_op_sdsc_op::matmul) fn witness() -> Self {
        MatmulWrapperSite(())
    }
}

impl NonAttnBmmSite for MatmulWrapperSite {}
impl NonAttnBmmSite for crate::emit::bmm_site::TapeLoweringSite {}

/// ⭐ THE RUNG REGIME AS A TYPE. Its associated types `A0`/`A1` ARE the axis order of the batched
/// rank-3 walk (`Walk3<A0, A1, stick>`), so `[mb, y, ·]` under the proven regime and `[y, mb, ·]`
/// under the request regime are DIFFERENT TYPES, and the stride assignment each order encodes is a
/// consequence of which regime a bundle parsed into — not of a constructor picked at an emission
/// site. Handing the solo-decode walk to a batch rung is an `E0308`, not a wrong bundle.
///
/// One impl per regime also fixes, in the same breath:
///   * input/output orders move TOGETHER — both walks project the SAME `A0`/`A1`, so a crossed
///     pair has no type;
///   * the stick axis is the LAST axis of both walks, by the trait's own return types;
///   * the work-division policy ([`Self::split_map`]) is paired with the walk in the impl, so the
///     split policy and the declared order cannot be paired across regimes.
pub(crate) trait RungRegime: sealed::Sealed + Copy {
    /// The OUTERMOST axis of this regime's declared batched walk.
    type A0: WalkAxis;
    /// The middle axis — with `A0`, the whole stride assignment.
    type A1: WalkAxis;
    /// This regime's batched INPUT walk, `[A0, A1, in]`. Takes the regime VALUE: a walk exists
    /// only downstream of a boundary door, never from naming a type.
    fn input_walk(self) -> Walk3<Self::A0, Self::A1, InAxis>;
    /// This regime's batched OUTPUT walk, `[A0, A1, out]` — the same order on the N stick.
    fn output_walk(self) -> Walk3<Self::A0, Self::A1, OutAxis>;
    /// The work-division policy this regime pairs with its walk: the joint cost-model split for
    /// the proven regime, batch-divides-first for the request regime (each documented on its
    /// splitter in `matmul/dims.rs`).
    fn split_map(self) -> super::dims::SplitMapFn;

    /// ⭐⭐⭐ THE ELEMENT DISTANCE THIS REGIME'S WALK ASSIGNS TO `y` — the stride the op will actually
    /// use to reach the next head.
    ///
    /// A row-major walk gives an axis the product of the extents AFTER it, so the answer is decided
    /// entirely by where `y` sits in this regime's order — which is why it belongs HERE, paired with
    /// the walk in one impl, exactly as `split_map` is. Derived from the same `A0`/`A1` that define
    /// `input_walk`/`output_walk`, so a regime cannot report a stride its own order does not produce.
    ///
    /// `mb_dev` is the operand's DEVICE `mb` extent (its `phys_mb` when it declares one, else the
    /// op's own `m`); `stick_extent` is the walk's last axis — `in` for the input, `out` for the
    /// output. The caller compares the answer against the stride the OPERAND really has
    /// ([`crate::sdsc_abstract::BatchStrides`]) and refuses on mismatch.
    fn y_stride_elems(self, mb_dev: u32, stick_extent: u32) -> u32;
}

impl RungRegime for BatchInnerMq1Proven {
    type A0 = MbAxis;
    type A1 = YAxis;
    fn input_walk(self) -> Walk3<MbAxis, YAxis, InAxis> {
        Walk3::input_batch_inner_mq1_proven()
    }
    fn output_walk(self) -> Walk3<MbAxis, YAxis, OutAxis> {
        Walk3::output_batch_inner_mq1_proven()
    }
    fn split_map(self) -> super::dims::SplitMapFn {
        super::dims::matmul_split_map
    }
    /// `[mb, y, stick]` — `y` is followed only by the stick, so it strides by ONE stick extent. `mb`
    /// is OUTSIDE `y` here and therefore contributes nothing: the batch-inner order's whole
    /// character is that `mb` gets the batch block and `y` gets a single stick.
    fn y_stride_elems(self, _mb_dev: u32, stick_extent: u32) -> u32 {
        stick_extent
    }
}

impl RungRegime for HeadOutermostRequests {
    type A0 = YAxis;
    type A1 = MbAxis;
    fn input_walk(self) -> Walk3<YAxis, MbAxis, InAxis> {
        Walk3::input_head_outermost()
    }
    fn output_walk(self) -> Walk3<YAxis, MbAxis, OutAxis> {
        Walk3::output_head_outermost()
    }
    fn split_map(self) -> super::dims::SplitMapFn {
        super::dims::matmul_split_map_batch_requests
    }
    /// `[y, mb, stick]` — `y` is OUTERMOST, so it strides by everything after it: one whole `mb`
    /// PLANE, `mb_dev * stick`. `mb_dev` is the DEVICE extent, so an operand that declares a
    /// `phys_mb` larger than the rows this op sweeps gets the plane it really has.
    fn y_stride_elems(self, mb_dev: u32, stick_extent: u32) -> u32 {
        mb_dev * stick_extent
    }
}

/// A named matmul walk axis. `NAME` is the ONE spelling of the axis's `ItDim`/layout/split
/// key — `matmul_dims`, the split maps and every declared walk read it from here, so the
/// axis vocabulary cannot fork into per-file string literals.
pub(crate) trait WalkAxis {
    const NAME: &'static str;
}

/// M — the row / token / request axis (`mb`). Carried across the PT array rows.
pub(crate) struct MbAxis;
impl WalkAxis for MbAxis {
    const NAME: &'static str = "mb";
}

/// The batch axis (`y`) — a GQA group's query head, or the request in the request-axis form.
pub(crate) struct YAxis;
impl WalkAxis for YAxis {
    const NAME: &'static str = "y";
}

/// K — the reduction stick axis (`in`).
pub(crate) struct InAxis;
impl WalkAxis for InAxis {
    const NAME: &'static str = "in";
}

/// N — the output stick axis (`out`).
pub(crate) struct OutAxis;
impl WalkAxis for OutAxis {
    const NAME: &'static str = "out";
}

/// A rank-2 declared walk `[A, S]`, sticked on `S` (the last axis, by construction).
/// The axis order lives in the TYPE, so two walks with different stride assignments are
/// different types; the field is private, so one can only be minted by the named
/// constructors below — each of which states the stride assignment it encodes.
pub(crate) struct Walk2<A: WalkAxis, S: WalkAxis> {
    _order: PhantomData<(A, S)>,
}

/// A rank-3 declared walk `[A, B, S]`, sticked on `S`. Same sealing as [`Walk2`].
pub(crate) struct Walk3<A: WalkAxis, B: WalkAxis, S: WalkAxis> {
    _order: PhantomData<(A, B, S)>,
}

impl<A: WalkAxis, S: WalkAxis> Walk2<A, S> {
    /// The `layoutDimOrder_` this walk declares, in walk order.
    pub(crate) fn order(&self) -> [&'static str; 2] {
        [A::NAME, S::NAME]
    }
    /// The stick axis — always the walk's LAST axis.
    pub(crate) fn stick(&self) -> &'static str {
        S::NAME
    }
}

impl<A: WalkAxis, B: WalkAxis, S: WalkAxis> Walk3<A, B, S> {
    /// The `layoutDimOrder_` this walk declares, in walk order.
    pub(crate) fn order(&self) -> [&'static str; 3] {
        [A::NAME, B::NAME, S::NAME]
    }
    /// The stick axis — always the walk's LAST axis.
    pub(crate) fn stick(&self) -> &'static str {
        S::NAME
    }
}

impl Walk2<InAxis, OutAxis> {
    /// The SHARED 2-D KERNEL walk `[in, out]`, stick `out` — the weight broadcast across
    /// `mb` and the batch. Rank-2 stick-on-last classifies stick-blocked (`Kernel`): the
    /// device tile is `[out/stick, in, stick]`, dxp reconstructs it from `-1` extents, and
    /// the per-core `out` corner strides by `in·stick` (a whole stick group).
    pub(crate) fn kernel_shared() -> Self {
        Walk2 {
            _order: PhantomData,
        }
    }
}

impl Walk2<MbAxis, InAxis> {
    /// The plain 2-D matmul ACTIVATION walk `[mb, in]`, stick `in`. Rank-2 stick-on-last
    /// classifies stick-blocked (`RowBlocked`): rows interleave inside each stick group,
    /// the `mb` corner strides by one stick width, and dxp reconstructs from `-1`.
    pub(crate) fn input_rows_by_k() -> Self {
        Walk2 {
            _order: PhantomData,
        }
    }
}

impl Walk2<MbAxis, OutAxis> {
    /// The plain 2-D matmul OUTPUT walk `[mb, out]`, stick `out`. Same stick-blocked
    /// residency as [`Walk2::input_rows_by_k`], on the N stick.
    pub(crate) fn output_rows_by_n() -> Self {
        Walk2 {
            _order: PhantomData,
        }
    }
}

impl Walk3<MbAxis, YAxis, InAxis> {
    /// ⛔ The ON-HARDWARE-PROVEN mq==1 shared-kernel batched INPUT walk `[mb, y, in]` — the walk
    /// whose row-major stride assignment gives `mb` the batch BLOCK `y·in` and `y` ONE stick
    /// (`in`). A head-major operand (`qs` head `h`, row `r` at `h·mb·in + r·in`) has row pitch
    /// `in` and head stride `mb·in` — the OTHER assignment. The two coincide exactly at
    /// `mb == 1`, which is the ONLY regime this walk has ever been proven in: at `mb > 1` a
    /// PINNED (`maxDimSizes_` = actual) walk of this order declares the strides SWAPPED
    /// (measured at mq=23: `{mb:23, y:1}`, `maxDimSizes_ [23,4,64]` — mb-stride 256 / y-stride
    /// 64 against the tensor's real 64 / mq·hd).
    ///
    /// PRIVATE — the only door is the proven regime's [`RungRegime::input_walk`], and the
    /// request regime's associated order is `[y, mb, ·]`, so a batch-decode rung cannot take the
    /// walk that is right only at one row: this type does not fit its regime's walk.
    fn input_batch_inner_mq1_proven() -> Self {
        Walk3 {
            _order: PhantomData,
        }
    }
}

impl Walk3<MbAxis, YAxis, OutAxis> {
    /// ⛔ The ON-HARDWARE-PROVEN mq==1 batched OUTPUT walk `[mb, y, out]` — same stride
    /// assignment as [`Walk3::input_batch_inner_mq1_proven`] on the N stick: `mb` gets the
    /// batch block `y·out`, `y` one stick, while the head-major score buffer's row pitch is
    /// `out` and head stride `mb·out`. Coincides with the physical layout only at `mb == 1`,
    /// and is private for the same reason as its input twin.
    fn output_batch_inner_mq1_proven() -> Self {
        Walk3 {
            _order: PhantomData,
        }
    }
}

impl Walk3<YAxis, MbAxis, InAxis> {
    /// The batched bmm INPUT walk `[y, mb, in]` — batch OUTERMOST. Row-major provenance:
    /// `y` strides by `mb·in` (one head PLANE — the head-major head stride at any `mb`,
    /// `mb` being the DEVICE extent so a declared physical row count sets the plane) and
    /// `mb` by `in` (the row pitch). This is the assignment that matches a head-major
    /// activation at EVERY `mb`; the batch-inner order `[mb, y, in]` swaps those two
    /// strides when pinned (see [`Walk3::input_batch_inner_mq1_proven`]) and is reachable
    /// only through the proven regime.
    pub(crate) fn input_head_outermost() -> Self {
        Walk3 {
            _order: PhantomData,
        }
    }
}

impl Walk3<YAxis, InAxis, OutAxis> {
    /// The per-batch 3-D KERNEL walk `[y, in, out]`, stick `out` — `y` carries the head,
    /// striding one `in·out` block per batch (each head its OWN K/V; the whole difference
    /// from [`Walk2::kernel_shared`]).
    pub(crate) fn kernel_per_head() -> Self {
        Walk3 {
            _order: PhantomData,
        }
    }
}

impl Walk3<YAxis, MbAxis, OutAxis> {
    /// The batched bmm OUTPUT walk `[y, mb, out]` — batch outermost, same provenance as
    /// [`Walk3::input_head_outermost`] on the N stick: `y` one head plane (`mb·out`),
    /// `mb` one row (`out`). The batch-inner `[mb, y, out]` is reachable only through
    /// the proven regime.
    pub(crate) fn output_head_outermost() -> Self {
        Walk3 {
            _order: PhantomData,
        }
    }
}

/// One work-slice corner component ALONG A NAMED AXIS, in device elements. The axis is the
/// TYPE, so a start-address term cannot be composed without saying which axis it steps —
/// the per-core fp8 kernel offset takes `Corner<InAxis>`/`Corner<OutAxis>`, and handing it
/// the corner components in the wrong order is an `E0308`, not a silent transposition.
pub(crate) struct Corner<A: WalkAxis> {
    elems: u64,
    _axis: PhantomData<A>,
}

impl<A: WalkAxis> Corner<A> {
    /// The corner component, for the offset fold.
    pub(crate) fn elems(&self) -> u64 {
        self.elems
    }
}

impl Corner<InAxis> {
    /// The fp8 W8A8 KERNEL's `in` (K-row) corner: layout position 0 of the rank-2
    /// [`Walk2::kernel_shared`] walk `[in, out]`. POSITIONAL — the corner slice itself
    /// does not carry axis names, so this constructor is where "index 0 IS `in`" is
    /// asserted (valid for every constructible fp8 matmul kernel, which is 2-D W8A8).
    pub(crate) fn fp8_kernel_in(corner: &[usize]) -> Corner<InAxis> {
        Corner {
            elems: corner[0] as u64,
            _axis: PhantomData,
        }
    }
}

impl Corner<OutAxis> {
    /// The fp8 W8A8 KERNEL's `out` (N-column) corner: layout position 1 of the rank-2
    /// [`Walk2::kernel_shared`] walk `[in, out]`. POSITIONAL — see [`Corner::fp8_kernel_in`].
    pub(crate) fn fp8_kernel_out(corner: &[usize]) -> Corner<OutAxis> {
        Corner {
            elems: corner[1] as u64,
            _axis: PhantomData,
        }
    }
}

/// The `in` (K) EXTENT of the fp8 W8A8 kernel's device shape — the row count of its
/// staged `[N/64, K/2, 2, 64]` pack. Positional like the corners: device extent index 0
/// of the rank-2 `[in, out]` kernel walk.
pub(crate) struct Fp8KernelKRows(u64);

impl Fp8KernelKRows {
    /// Extent index 0 of the fp8 kernel's device-extent vector (`[in, out]` order).
    pub(crate) fn of_device_extents(host_size: &[u64]) -> Fp8KernelKRows {
        Fp8KernelKRows(host_size[0])
    }
}

/// Device element offset of an fp8 W8A8 KERNEL work-slice corner. The weight is staged
/// OUT-STICK-MAJOR packed `[N/64, K/2, 2, 64]` (its RetileDescriptor), so the corner's
/// device offset must use THAT layout, not the flat 128-stick `for_view_df` model:
/// `(n/64)·(K·64) + (n%64) + (i/2)·128 + (i%2)·64`. For the decode out-split corner
/// (`i = 0`, `n = c·perN`, `perN % 64 == 0`) this is `c·perN·K` — per-core-contiguous.
/// Every stride here is NAMED by its parameter's axis type; the bare-`corner[_]`
/// arithmetic this replaces carried the same numbers with no axis attached.
pub(crate) fn fp8_kernel_stage_off(
    i: Corner<InAxis>,
    n: Corner<OutAxis>,
    k: Fp8KernelKRows,
) -> u64 {
    let (i, n, k) = (i.elems(), n.elems(), k.0);
    (n / 64) * (k * 64) + (n % 64) + (i / 2) * 128 + (i % 2) * 64
}

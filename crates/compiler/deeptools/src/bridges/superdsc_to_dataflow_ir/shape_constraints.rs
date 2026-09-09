//! THE DDL COMPILER'S DERIVATIONS — and the reason this campaign exists.
//! ⛔ scratchy currently INVENTS these in islands/.../shape.rs. That invention is the defect being removed.
//! ⛔ ABSOLUTE STAGE EXTENTS ARE A CONSTRAINT SYSTEM, NOT A FORMULA: ratios against a tensor, multiples,
//! stage-relative bounds, SETs, bare bounds, and a different rule per `ddl.if` arm. TRIP COUNTS need none
//! of it — a loop's own two stages are dimensionless.
//! ⭐ THE PORT IDENTITY IS `data_connect=`, NOT THE SSA NAME (createDataConnectMetadata).
//!
//! 4 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e001_checkConstraints` | 0 | 132 | `ddc/ddcv1.cpp:792` |
//! | `e002_createDataConnectMetadata` | 0 | 45 | `ddc/ddcv1.cpp:3283` |
//! | `e041_getStickSizes` | 0 | 41 | `dsc/dsc2.cpp:4066` |
//! | `e071_getCumulativeStickSizes` | 1 | 17 | `dsc/dsc2.cpp:4108` |

use crate::arch::{Arch, Elements, Target};
use crate::generated::DataConnect;
use crate::schedule::ddc::metadata::{Ends, LoopMultiple, MetaDimKind, NodeIndex};
use crate::units::{Corelet, Row};

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

/// A LAYOUT DIMENSION — `PrimaryDimTypes` (`dsc/dims.h:34`), less its `PrimaryDimTypesCount`
/// terminator, which is a count and not a dim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrimaryDim {
    In,
    Out,
    Ij,
    Mb,
    X,
    Y,
    Kij,
    I,
    J,
    Ki,
    Kj,
    X1,
}

/// ONE STICK'S DIMS, OUTERMOST FIRST — `PrimaryDsInfo::stickDimOrder_` zipped with `stickSize_`
/// (`dsc/dscdefn.h:474`), so the two orders cannot disagree in length. (`stickSize_` is a `double`
/// vector there and every entry of it is read as an `int`.)
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StickDims(pub Vec<(PrimaryDim, Elements)>);

/// `elemInSlice` — what one slice of the stick holds, ALREADY DIVIDED (`dsc/dsc2.cpp:4084`).
///
/// ⛔ THE `DT_CHECK` IS THIS CONSTRUCTOR, NOT A REFUSAL LATER: a stick whose extents do not
/// multiply to a positive multiple of the slice count has no `SliceElems`, so neither slice arm of
/// [`StickPart`] can be asked for it and the division is done once, here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceElems(Elements);

impl SliceElems {
    /// `stickSliceOnly` / `stickWithoutSlice` — `numSlices = 8` is [`Arch::SLICES_PER_STICK`]
    /// (`sysdef.cpp:229`), 8 on both arches this crate compiles for.
    #[must_use]
    pub fn per_stick<A: Arch>(dims: &StickDims) -> Option<Self> {
        Self::of(dims, u64::from(A::SLICES_PER_STICK))
    }

    /// `l0SliceOnly` — `numSlices = numL0Slices`, which is `sysDef.numPTRows` at the one call that
    /// sets the flag (`ddc/ddcv1.cpp:491`), so it is [`Arch::PT_ROWS`] and its `> 0` check is a
    /// constant of the arch rather than an argument that can be forgotten.
    #[must_use]
    pub fn per_l0_row<A: Arch>(dims: &StickDims) -> Option<Self> {
        Self::of(dims, u64::from(A::PT_ROWS))
    }

    /// `DT_CHECK(elemInSlice > 0 && elemInSlice % numSlices == 0)` (`dsc/dsc2.cpp:4083`), and the
    /// zero divisor a `numSlices` of nought would be.
    fn of(dims: &StickDims, slices: u64) -> Option<Self> {
        let stick = dims
            .0
            .iter()
            .try_fold(1u64, |acc, &(_, extent)| acc.checked_mul(extent.0))?;
        if stick == 0 || slices == 0 || stick % slices != 0 {
            return None;
        }
        Some(SliceElems(Elements(stick / slices)))
    }
}

/// WHICH PART OF THE STICK [`stick_sizes`] REPORTS — the reference's three mutually exclusive
/// `bool`s, whose "no two of them at once" `DT_CHECK` (`dsc/dsc2.cpp:4070`) is this enum. The
/// vendor's own names for the two slice arms are `withinSlice` and `crossSlice`
/// (`ddc/ddcv1.cpp:3578-3579`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StickPart {
    /// All three flags false — every stick dim at its full extent.
    Whole,
    /// `stickSliceOnly`, or `l0SliceOnly` when the slice is [`SliceElems::per_l0_row`].
    WithinSlice(SliceElems),
    /// `stickWithoutSlice` — what is left of the stick once one slice has been filled.
    CrossSlice(SliceElems),
}

/// Replaces: e041_getStickSizes
///
/// The stick's dims with their extents clipped to one slice ([`StickPart::WithinSlice`], stopping at
/// the dim that fills it) or to what crosses beyond it ([`StickPart::CrossSlice`], skipping the dims
/// the slice already covers).
///
/// ⛔ THE CLIP IS BY THE RATIO, NOT BY THE ROOM LEFT — `size /= elemSoFar / elemInSlice` with the
/// reference's truncating `int` division, which is a different number as soon as an extent does not
/// divide the slice, and the two arms then no longer partition the stick.
#[must_use]
pub fn stick_sizes(dims: &StickDims, part: StickPart) -> Vec<(PrimaryDim, Elements)> {
    let mut result = Vec::new();
    let mut elem_so_far = 1u64;
    for &(dim, extent) in &dims.0 {
        let mut size = extent.0;
        match part {
            StickPart::Whole => {}
            StickPart::WithinSlice(slice) => {
                // Can not fit more elements into the slice.
                if elem_so_far >= slice.0.0 {
                    break;
                }
                elem_so_far *= size;
                if elem_so_far > slice.0.0 {
                    size /= elem_so_far / slice.0.0;
                }
            }
            StickPart::CrossSlice(slice) => {
                let new_elem_so_far = elem_so_far * size;
                if elem_so_far < slice.0.0 && new_elem_so_far > slice.0.0 {
                    size /= slice.0.0 / elem_so_far;
                }
                elem_so_far = new_elem_so_far;
                if elem_so_far <= slice.0.0 {
                    // dim included in slice
                    continue;
                }
            }
        }
        result.push((dim, Elements(size)));
    }
    result
}

/// Replaces: e071_getCumulativeStickSizes
///
/// [`stick_sizes`]' list folded to ONE extent per dim: a dim the stick names twice contributes the
/// PRODUCT of its two extents, not the last one.
///
/// ⭐ THE `unordered_map` RETURN CARRIES NO ORDER, so the caller cannot have depended on one; this
/// keeps first-appearance order, which is the stick's own dim order.
///
/// ⛔ AND THE PRODUCT IS THE ONE PLACE THIS CAN FAIL: the reference multiplies into an `int` and a
/// split dim whose extents overflow it wraps silently. A width that cannot be counted is no answer
/// at all.
#[must_use]
pub fn cumulative_stick_sizes(
    dims: &StickDims,
    part: StickPart,
) -> Option<Vec<(PrimaryDim, Elements)>> {
    let mut result: Vec<(PrimaryDim, Elements)> = Vec::new();
    for (dim, extent) in stick_sizes(dims, part) {
        if let Some(seen) = result.iter_mut().find(|(walked, _)| *walked == dim) {
            seen.1 = Elements(seen.1.0.checked_mul(extent.0)?);
        } else {
            result.push((dim, extent));
        }
    }
    Some(result)
}

// ═══ e001 — THE DATASTAGE CONSTRAINT CHECK ══════════════════════════════════════════════════════

/// WHICH VECTOR UNIT A PE/SFP-SPLIT DIM IS SAMPLED ON — the reference's `comps` vector, which is
/// either `{NO_COMPONENT}` or exactly `{PE, SFP}` (`ddc/ddcv1.cpp:817-823`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorComp {
    /// `SenComponents::PE`.
    Pe,
    /// `SenComponents::SFP`.
    Sfp,
}

/// WHICH SPLIT A CONSTRAINT CHECK IS SAMPLING AT — the `(cl, row, comp)` triple
/// `checkConstraintsImpl` is called with (`ddc/ddcv1.cpp:838`).
///
/// ⛔ THE `-1`s AND `NO_COMPONENT` ARE ABSENCES, NOT INDEX ZERO. `row = -1` under
/// `rowSplit_.empty()` (`:849-852`) and the `row = -1` of the PE/SFP loop (`:929`) both mean *the
/// whole dimension*, which a row 0 does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sample {
    /// `cl` — always a corelet, and corelet 0 where no dim of the set is corelet-split.
    pub corelet: Corelet,
    /// `row`, absent where the reference passes `-1`.
    pub row: Option<Row>,
    /// `comp`, absent where the reference passes `NO_COMPONENT`.
    pub comp: Option<VectorComp>,
}

/// ONE STAGE'S EXTENT FOR ONE DIM AT ONE SAMPLE — `primaryDimToVal_st`'s `int` (`dsc/dims.h:269`).
///
/// ⛔ SIGNED, AND THAT IS NOT AN ERROR CHANNEL. Every dimension of `DataStructDims` defaults to `-1`
/// (`dsc/dims.h:161-193`) and the reference reads that as *this stage has no such dim*: `if (dimSize
/// > 0)` on the reference side (`:864`) is the entire handling of it, while the candidate side
/// multiplies it in unchecked (`:857`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent(pub i64);

/// A PADDED EXTENT, POSITIVE BY TYPE — `primaryDimToVal_st(dim, comp, row, cl, padding)` with
/// `PADDED_WZEROPAD` set on the dim (`:881-884`).
///
/// ⛔⛔ `DT_ERROR("Cannot check datastage no-epilogue constraint if dim not relevant for reference
/// data stage")` (`:886-891`) IS THIS TYPE, AND ON THE REFERENCE SIDE IT WAS ALREADY UNREACHABLE.
/// That line is reached only past `foundValidDim`, which for a single-dim key means the reference
/// stage's UNPADDED extent is `> 0`; `calculate_padded`'s `PADDED_WZEROPAD` arm is then `wSize + (val
/// - 1) * stride` with `wSize >= 1` enforced one line above (`dsc/dims.cpp:596-602`), which cannot
/// come back non-positive.
///
/// ⭐ ON THE CANDIDATE SIDE IT REMOVES A DIVISION BY ZERO — `min(sizeLoop, refSizeLoop)` is the
/// divisor at `:902-904` and the reference tests nothing about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaddedExtent(core::num::NonZeroU64);

impl PaddedExtent {
    /// A padded extent, or [`None`] for the non-positive value the reference aborts on.
    #[must_use]
    pub fn of(extent: i64) -> Option<Self> {
        if extent <= 0 {
            return None;
        }
        core::num::NonZeroU64::new(extent.unsigned_abs()).map(PaddedExtent)
    }

    /// The extent.
    #[must_use]
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

/// WHAT A CONSTRAINT CHECK ASKS OF A DATA STAGE — `DataStructDims` (`dsc/dims.h:158`) reduced to the
/// seven questions entry 001 puts to it.
///
/// ⭐ A TRAIT BECAUSE THE ANSWERS ARE THE DSC'S, NOT THIS BRIDGE'S. `primaryDimToVal_st`
/// (`dsc/dims.h:269`) is a hundred lines of corelet-, row- and component-view derivation over the
/// stage's own split tables; reaching an operand is the mechanism, and the constraint check is what
/// this file owns.
pub trait Stage {
    /// `ds.symbolicDimInfo_.count(dim)` (`:803`).
    fn is_symbolic(&self, dim: PrimaryDim) -> bool;
    /// `ds.coreletSplit_.count(dim)` (`:812`).
    fn is_corelet_split(&self, dim: PrimaryDim) -> bool;
    /// `ds.rowSplit_.count(dim)` (`:815`).
    fn is_row_split(&self, dim: PrimaryDim) -> bool;
    /// `ds.peSfpSplit_.count(dim)` (`:818`).
    fn is_pe_sfp_split(&self, dim: PrimaryDim) -> bool;
    /// `ds.rowSplit_.empty()` (`:849`) — whether the stage splits ANY dim across rows, which is a
    /// different question from [`Self::is_row_split`] on one dim.
    fn splits_any_row(&self) -> bool;
    /// `primaryDimToVal_st(dim, comp, row, cl)` (`:857`, `:862`).
    fn extent(&self, dim: PrimaryDim, at: Sample) -> Extent;
    /// The same with `PADDED_WZEROPAD` on `dim`, or [`None`] where the dim is not relevant to this
    /// stage — see [`PaddedExtent`].
    fn padded_extent(&self, dim: PrimaryDim, at: Sample) -> Option<PaddedExtent>;
}

/// WHICH LOOP EXTENT A NO-EPILOGUE CONSTRAINT COMPARES — `loopDimKind_`, less the six kinds it
/// cannot be.
///
/// ⛔⛔ TWO `DT_ERROR`s COLLAPSE INTO THIS. `MetaDimKind::Count` is the field's UNSET sentinel
/// (`ddc/ddc_metadata.h:35`) and gives *"Datastage no-epilogue constraint without dim kind"*
/// (`:871-874`); anything outside `{Unpadded, Padded, WindowDim}` gives *"Unhandled dim kind in
/// no-epiloge datastage constraint"* (`:893-901`). Three of `MetaDimKind`'s nine are the whole
/// domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoEpilogueDimKind {
    /// `Unpadded` — the extents already computed.
    Unpadded,
    /// `Padded` — BOTH extents re-read with `PADDED_WZEROPAD` on the single dim (`:877-892`).
    Padded,
    /// `WindowDim` — the extents already computed, exactly like `Unpadded`: the only thing that
    /// distinguishes the two here is which of them the `DT_ERROR` at `:893` lets through.
    WindowDim,
}

/// AN ABSOLUTE CONSTRAINT'S `min_`, AND WHETHER THE SIZE MUST BE A MULTIPLE OF IT.
///
/// ⛔⛔ ONE NUMBER, NOT TWO. On the absolute arm `min_` is read TWICE — as the multiple in
/// `fmodf(size, *min_ * refSize)` (`:868`) and as the lower bound in `size < *min_ * refSize`
/// (`:906-908`) — so a `must_be_multiple` field beside a `min` field would be two copies of one
/// number that could disagree.
///
/// ⛔ AND `DT_ERROR("Must-be-multiple constraint but no min set")` (`:864-866`) IS THE MISSING FOURTH
/// STATE: `mustBeMultiple_` with no `min_` is not one of these.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AbsoluteMin {
    /// `min_` unset, and so `mustBeMultiple_` false — no lower bound at all.
    Unset,
    /// `min_` set, `mustBeMultiple_` false — a lower bound only.
    Bound(f32),
    /// `min_` set and `mustBeMultiple_` — the size must be a multiple of it AND at least it.
    Multiple(f32),
}

/// WHICH SIDE A CONSTRAINT'S SIZES ARE COMPARED AGAINST — the outer `constraints_` key
/// (`ddc/ddc_metadata.h:73-75`) turned into the presence of a reference stage.
///
/// ⛔⛔ THE KEY IS THE SPLIT, WHICH IS WHAT MAKES THE `DT_CHECK` UNNECESSARY. `refDsId < 0 ? nullptr
/// : &currDsc->dataStageParam_.at(refDsId).ss_` (`:794-796`) already decides it, and
/// `DT_CHECK(refDs == nullptr)` under `cannotBeSymbolic_` (`:801`) then asserts a pairing the key
/// fixed. Two variants say it instead — and each carries only the fields its own arm reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstraintKind<'a, S> {
    /// `refDsId < 0` — an ABSOLUTE constraint, whose `refSize` is 1.
    Absolute {
        /// `cannotBeSymbolic_` — ⛔ ONLY HERE, by the `DT_CHECK` above.
        cannot_be_symbolic: bool,
        /// `loopDimKind_`, which this arm's check never reads and `dump` always prints — set on
        /// absolute constraints at `ddc/ddc_transformation.cpp:968-1035`.
        dim_kind: Option<MetaDimKind>,
        /// `min_`, and whether the size must be a multiple of it.
        min: AbsoluteMin,
    },
    /// `refDsId >= 0` — a RELATIVE constraint, against `dataStageParam_.at(refDsId).ss_`.
    Relative {
        /// That reference stage.
        reference: &'a S,
        /// `min_` — a lower bound on `size / refSize`, with no multiple relationship.
        min: Option<f32>,
        /// `mustBeMultiple_` and `loopDimKind_`. ⛔ THE NO-EPILOGUE ARM IS STILL GATED BY
        /// `allowEpilogue` at the call (`:870`).
        multiple: LoopMultiple,
    },
}

/// ONE `Metadata::Datastage::Constraints` (`ddc/ddc_metadata.h:33`), with `min_` and
/// `mustBeMultiple_` folded into [`ConstraintKind`].
#[derive(Debug, Clone, PartialEq)]
pub struct Constraint<'a, S> {
    /// Which side the sizes are compared against.
    pub kind: ConstraintKind<'a, S>,
    /// `max_` — `size > *max_ * refSize` fails (`:903-905`).
    pub max: Option<f32>,
    /// `values_` — the permitted `float(size) / refSize` ratios (`:909-912`).
    ///
    /// ⛔ EXACT FLOAT EQUALITY, as the reference's `std::set<float>::count` is. A `std::set`'s
    /// ordering is not observable through `count`, so a list is the same question.
    pub values: Option<Vec<f32>>,
}

/// A NON-EMPTY, SORTED, DEDUPLICATED SET OF DIMS — the `std::set<PrimaryDimTypes>` inner key.
///
/// ⛔ NON-EMPTY BECAUSE `*dims.begin()` (`:880`) DEREFERENCES IT, and sorted because that is what
/// `std::set` makes "begin" mean.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimSet {
    head: PrimaryDim,
    rest: Vec<PrimaryDim>,
}

impl DimSet {
    /// The set, or [`None`] for the empty key `*dims.begin()` could not be taken of.
    #[must_use]
    pub fn of(dims: &[PrimaryDim]) -> Option<DimSet> {
        let mut sorted = dims.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        let (head, rest) = sorted.split_first()?;
        Some(DimSet {
            head: *head,
            rest: rest.to_vec(),
        })
    }

    /// `*dims.begin()` — the smallest dim, which is the one a single-dim key holds.
    #[must_use]
    pub fn first(&self) -> PrimaryDim {
        self.head
    }

    /// `dims.size()`.
    #[must_use]
    pub fn len(&self) -> usize {
        1 + self.rest.len()
    }

    /// `dims.count(dim)`.
    #[must_use]
    pub fn contains(&self, dim: PrimaryDim) -> bool {
        self.head == dim || self.rest.contains(&dim)
    }

    /// Every dim, ascending.
    pub fn iter(&self) -> impl Iterator<Item = PrimaryDim> + '_ {
        core::iter::once(self.head).chain(self.rest.iter().copied())
    }
}

/// ONE `(dims, constraint)` PAIR OF THE INNER MAP, with the pairing the no-epilogue arm demands
/// already established.
///
/// ⛔⛔ `DT_ERROR("Cannot check datastage no-epilogue constraint on multiple dimensions")`
/// (`:875-880`) IS THIS CONSTRUCTOR. The reference discovers the clash mid-check, once per
/// (corelet, row, component) sample and after both sizes have been computed; here a multi-dim key and
/// a no-epilogue constraint simply do not form a pair.
#[derive(Debug, Clone, PartialEq)]
pub struct DimConstraint<'a, S> {
    dims: DimSet,
    constraint: Constraint<'a, S>,
}

impl<'a, S> DimConstraint<'a, S> {
    /// The pair, or [`None`] where a no-epilogue constraint names more than one dim.
    #[must_use]
    pub fn of(dims: DimSet, constraint: Constraint<'a, S>) -> Option<Self> {
        let multi_dim_no_epilogue = matches!(
            constraint.kind,
            ConstraintKind::Relative {
                multiple: LoopMultiple::NoEpilogue(_),
                ..
            }
        ) && dims.len() > 1;
        if multi_dim_no_epilogue {
            return None;
        }
        Some(DimConstraint { dims, constraint })
    }

    /// The dims it applies to.
    #[must_use]
    pub fn dims(&self) -> &DimSet {
        &self.dims
    }

    /// The constraint itself.
    #[must_use]
    pub fn constraint(&self) -> &Constraint<'a, S> {
        &self.constraint
    }
}

/// Replaces: e001_checkConstraints
///
/// WHETHER EVERY DATASTAGE CONSTRAINT MENTIONING ONE DIMENSION HOLDS — `ddc/ddcv1.cpp:792`.
///
/// ⛔⛔ THE SAMPLES ARE TWO DIFFERENT NESTS, NOT ONE. Without a PE/SFP split it is corelets × rows at
/// `NO_COMPONENT` (`:920-926`); with one it is corelets × `{PE, SFP}` at `row = -1` (`:927-933`) — so
/// a dim split BOTH ways is never sampled per row, and the counts come from
/// `dscGlobal.sysDef.numCoreletsPerCore` / `numPTRows`, which are this build's arch.
///
/// ⛔ AND THE OUTER `constraints_` GROUPING IS NOT OBSERVABLE: the two nested loops are a flat
/// conjunction over `(dims, constraint)` pairs, and the reference stage each pair is measured against
/// is the outer key ([`ConstraintKind`]).
#[must_use]
pub fn check_constraints<S: Stage>(
    ds: &S,
    constraints: &[DimConstraint<'_, S>],
    dim_to_check: PrimaryDim,
    allow_epilogue: bool,
) -> bool {
    for entry in constraints {
        let dims = entry.dims();
        let constraint = entry.constraint();
        if !dims.contains(dim_to_check) {
            continue;
        }
        if let ConstraintKind::Absolute {
            cannot_be_symbolic: true,
            ..
        } = constraint.kind
        {
            if dims.iter().any(|dim| ds.is_symbolic(dim)) {
                return false;
            }
        }

        let mut dim_num_cl = 1;
        let mut dim_num_row = 1;
        let mut consider_pe_sfp_split = false;
        for dim in dims.iter() {
            if ds.is_corelet_split(dim) {
                dim_num_cl = Target::CORELETS_PER_CORE;
            }
            if ds.is_row_split(dim) {
                dim_num_row = Target::PT_ROWS;
            }
            if ds.is_pe_sfp_split(dim) {
                consider_pe_sfp_split = true;
            }
        }

        for corelet in (0..dim_num_cl).filter_map(Corelet::checked) {
            if consider_pe_sfp_split {
                for comp in [VectorComp::Pe, VectorComp::Sfp] {
                    let at = Sample {
                        corelet,
                        row: None,
                        comp: Some(comp),
                    };
                    if !constraint_holds(ds, dims, constraint, at, allow_epilogue) {
                        return false;
                    }
                }
            } else {
                for row in (0..dim_num_row).filter_map(Row::checked) {
                    let at = Sample {
                        corelet,
                        row: Some(row),
                        comp: None,
                    };
                    if !constraint_holds(ds, dims, constraint, at, allow_epilogue) {
                        return false;
                    }
                }
            }
        }
    }
    true
}

/// `checkConstraintsImpl` — one constraint at one sample (`ddc/ddcv1.cpp:838`).
///
/// ⛔ `row` IS CLEARED PER SAMPLE, NOT PER DIM SET: `ds.rowSplit_.empty() || (refDs &&
/// refDs->rowSplit_.empty())` (`:849-852`) drops to the full-dimension extent whenever EITHER stage
/// has no row interaction, so a row-split candidate measured against a row-flat reference is compared
/// whole.
fn constraint_holds<S: Stage>(
    ds: &S,
    dims: &DimSet,
    constraint: &Constraint<'_, S>,
    at: Sample,
    allow_epilogue: bool,
) -> bool {
    let reference = match constraint.kind {
        ConstraintKind::Relative { reference, .. } => Some(reference),
        ConstraintKind::Absolute { .. } => None,
    };
    let at = if !ds.splits_any_row() || reference.is_some_and(|refer| !refer.splits_any_row()) {
        Sample { row: None, ..at }
    } else {
        at
    };

    let mut size: i64 = 1;
    for dim in dims.iter() {
        size = size.saturating_mul(ds.extent(dim, at).0);
    }

    let mut ref_size: i64 = 1;
    if let Some(refer) = reference {
        let mut found_valid_dim = false;
        for dim in dims.iter() {
            let dim_size = refer.extent(dim, at).0;
            if dim_size > 0 {
                found_valid_dim = true;
                ref_size = ref_size.saturating_mul(dim_size);
            }
        }
        // ⭐ NO DIM OF THE SET IS THE REFERENCE STAGE'S — the constraint does not apply (`:869`).
        if !found_valid_dim {
            return true;
        }
    }

    let size_f = size as f32;
    let ref_size_f = ref_size as f32;

    let min = match constraint.kind {
        ConstraintKind::Absolute { min, .. } => match min {
            AbsoluteMin::Unset => None,
            AbsoluteMin::Bound(min) | AbsoluteMin::Multiple(min) => Some(min),
        },
        ConstraintKind::Relative { min, .. } => min,
    };

    match constraint.kind {
        // ⭐ `refSize` IS 1 ON THIS ARM: there is no reference stage to multiply by, so the
        // reference's `*min_ * refSize` is `*min_` — and `fmodf(x, 0)` is NaN, which fails, exactly as
        // it does there.
        ConstraintKind::Absolute {
            min: AbsoluteMin::Multiple(multiple),
            ..
        } => {
            if size_f % (multiple * ref_size_f) != 0.0 {
                return false;
            }
        }
        ConstraintKind::Relative {
            reference: refer,
            multiple: LoopMultiple::NoEpilogue(kind),
            ..
        } if !allow_epilogue => {
            let loops = match kind {
                NoEpilogueDimKind::Unpadded | NoEpilogueDimKind::WindowDim => {
                    Some((size, ref_size))
                }
                NoEpilogueDimKind::Padded => {
                    let dim = dims.first();
                    // ⛔ ONE DIM, BY [`DimConstraint::of`] — `*dims.begin()` is the only dim there is.
                    match (ds.padded_extent(dim, at), refer.padded_extent(dim, at)) {
                        (Some(size_loop), Some(ref_size_loop)) => Some((
                            i64::try_from(size_loop.get()).unwrap_or(i64::MAX),
                            i64::try_from(ref_size_loop.get()).unwrap_or(i64::MAX),
                        )),
                        // ⭐ A DIM NEITHER STAGE HAS A PADDED EXTENT FOR IS THE `!foundValidDim` CASE
                        // ONE BRANCH UP, AND THE ANSWER THERE IS THE SAME.
                        _ => None,
                    }
                }
            };
            if let Some((size_loop, ref_size_loop)) = loops {
                let hi = size_loop.max(ref_size_loop);
                let lo = size_loop.min(ref_size_loop);
                // ⛔ `checked_rem` BECAUSE THE REFERENCE DIVIDES BY `min(sizeLoop, refSizeLoop)`
                // UNTESTED (`:902-904`): a candidate stage without the dim makes that zero.
                if hi.checked_rem(lo).is_some_and(|rem| rem != 0) {
                    return false;
                }
            }
        }
        _ => {}
    }

    if let Some(max) = constraint.max {
        if size_f > max * ref_size_f {
            return false;
        }
    }
    if let Some(min) = min {
        if size_f < min * ref_size_f {
            return false;
        }
    }
    if let Some(values) = &constraint.values {
        if !values.contains(&(size_f / ref_size_f)) {
            return false;
        }
    }
    true
}

// ═══ e002 — THE DATA-CONNECT CENSUS ═════════════════════════════════════════════════════════════

/// ONE END A NODE READS FROM — the `data_connect=` and whether the unit feeding it is the CONSTANT
/// source.
///
/// ⛔⛔ THE TWO LISTS ARE ONE FACT. The reference indexes `inputs_[ind]` by the length of
/// `inputsLdsAndLoopOffsets_` (`ddc/ddcv1.cpp:3303-3310`), so a shorter `inputs_` reads past its end;
/// pairing them removes the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reads {
    /// `di.dataConnect_`.
    pub data_connect: DataConnect,
    /// `transfer->src_.unit_ == CONSTANT` (`:3292`) / `compute->inputs_[ind] == CONSTANT` (`:3306`) —
    /// ⭐ A CONSTANT SOURCE CONSUMES NOTHING, so this end is not recorded at all.
    pub from_constant: bool,
}

/// ONE SCHEDULE NODE THE CENSUS VISITS — `traverseTreeDFSMutable(nullptr, {COMPUTE, TRANSFER})`
/// (`:3287-3289`) keeps these two kinds and no others.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleNode {
    /// `dsc2::TransferNode`.
    Transfer {
        /// `srcLdsAndLoopOffsets_` — ONE source end, which CONSUMES.
        src: Reads,
        /// `dstLdsAndLoopOffsets_` — the destination ends, which PRODUCE. ⛔ NO CONSTANT TEST ON THIS
        /// SIDE (`:3299-3300`).
        dsts: Vec<DataConnect>,
    },
    /// `dsc2::ComputeNode`.
    Compute {
        /// `inputsLdsAndLoopOffsets_` zipped with `inputs_`.
        inputs: Vec<Reads>,
        /// `outputsLdsAndLoopOffsets_` (`:3313-3314`).
        outputs: Vec<DataConnect>,
        /// `instrAttribute_.input_data_connects_` (`:3316-3318`) — an opaque body's read ports. ⛔ NO
        /// CONSTANT TEST HERE EITHER: the reference does not have one to make.
        opaque_reads: Vec<DataConnect>,
        /// `instrAttribute_.output_data_connects_` (`:3319-3321`).
        opaque_writes: Vec<DataConnect>,
    },
}

/// THE CENSUS' ANSWER — `metadata.dataConnects_`, or the label that has no producer.
///
/// ⛔⛔ `DT_ERROR("Illegal DDL: data_connect " + label + " does not have any producer.")`
/// (`:3324-3329`) IS THIS ENUM, AND IT NAMES THE OFFENDER. An [`Option`] would have thrown the label
/// away, which is the whole content of the diagnostic.
///
/// ⭐ WHICH offender, where several qualify, is not the reference's to fix: `dataConnects_` is an
/// `unordered_map` (`ddc/ddc_metadata.h:194`) and its walk order is unspecified. This reports the
/// first in walk order, which is at least the same one every time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataConnects {
    /// Every connect the schedule touches has at least one producer.
    Census(Vec<(DataConnect, Ends)>),
    /// ⛔ ILLEGAL DDL — this connect is read and never written.
    NoProducer(DataConnect),
}

/// Replaces: e002_createDataConnectMetadata
///
/// WHICH NODES WRITE AND WHICH READ EACH `data_connect=` — `ddc/ddcv1.cpp:3283`.
///
/// ⛔⛔ THE PORT IDENTITY IS THE `data_connect=` AND NOT THE SSA NAME, which is why the whole census
/// keys on [`DataConnect`] and the node is only ever a value in it.
///
/// ⛔ THE CONSTANT TEST IS ON THE CONSUMER SIDE ONLY, AND NOT ON AN OPAQUE'S PORTS: a transfer's
/// destinations, a compute's outputs and both opaque lists are recorded unconditionally (`:3299`,
/// `:3313`, `:3316`, `:3319`).
#[must_use]
pub fn create_data_connect_metadata(nodes: &[ScheduleNode]) -> DataConnects {
    let mut census: Vec<(DataConnect, Ends)> = Vec::new();

    for (index, node) in nodes.iter().enumerate() {
        let node = match node {
            ScheduleNode::Transfer { src, dsts } => {
                if !src.from_constant {
                    let at = ends_of(&mut census, src.data_connect);
                    census[at].1.insert_consumer(NodeIndex(index));
                }
                for dst in dsts {
                    let at = ends_of(&mut census, *dst);
                    census[at].1.insert_producer(NodeIndex(index));
                }
                continue;
            }
            compute @ ScheduleNode::Compute { .. } => compute,
        };
        if let ScheduleNode::Compute {
            inputs,
            outputs,
            opaque_reads,
            opaque_writes,
        } = node
        {
            for input in inputs {
                if !input.from_constant {
                    let at = ends_of(&mut census, input.data_connect);
                    census[at].1.insert_consumer(NodeIndex(index));
                }
            }
            for output in outputs {
                let at = ends_of(&mut census, *output);
                census[at].1.insert_producer(NodeIndex(index));
            }
            for read in opaque_reads {
                let at = ends_of(&mut census, *read);
                census[at].1.insert_consumer(NodeIndex(index));
            }
            for write in opaque_writes {
                let at = ends_of(&mut census, *write);
                census[at].1.insert_producer(NodeIndex(index));
            }
        }
    }

    for (label, ends) in &census {
        if ends.producers().is_empty() {
            return DataConnects::NoProducer(*label);
        }
    }
    DataConnects::Census(census)
}

/// The entry for one connect, appending it in first-touch order where the census has not seen it —
/// `dcMap[label]`'s own insert-on-lookup.
fn ends_of(census: &mut Vec<(DataConnect, Ends)>, connect: DataConnect) -> usize {
    match census.iter().position(|(label, _)| *label == connect) {
        Some(at) => at,
        None => {
            census.push((connect, Ends::default()));
            census.len() - 1
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        AbsoluteMin, Constraint, ConstraintKind, DataConnects, DimConstraint, DimSet, Ends, Extent,
        LoopMultiple, NoEpilogueDimKind, NodeIndex, PaddedExtent, PrimaryDim, Reads, Sample,
        ScheduleNode, SliceElems, Stage, StickDims, StickPart, VectorComp, check_constraints,
        create_data_connect_metadata, cumulative_stick_sizes, stick_sizes,
    };
    use crate::arch::{Arch, Dd2, Elements, Target};
    use crate::generated::DataConnect;

    /// A [`Stage`] THAT ANSWERS FROM TABLES — a constraint check asks nothing of `DataStructDims`
    /// but these seven questions, and `primaryDimToVal_st`'s split-view derivation is the mechanism
    /// for reaching them.
    #[derive(Default)]
    struct Table {
        symbolic: Vec<PrimaryDim>,
        corelet_split: Vec<PrimaryDim>,
        row_split: Vec<PrimaryDim>,
        pe_sfp_split: Vec<PrimaryDim>,
        /// WHOLE-dimension extents; a sample divides them by the splits it names.
        extents: Vec<(PrimaryDim, i64)>,
        padded: Vec<(PrimaryDim, i64)>,
    }

    impl Table {
        fn whole(table: &[(PrimaryDim, i64)], dim: PrimaryDim) -> i64 {
            // ⛔ `-1` IS THE DEFAULT OF EVERY DIM OF `DataStructDims` (`dsc/dims.h:161-193`).
            table
                .iter()
                .find(|(named, _)| *named == dim)
                .map_or(-1, |(_, extent)| *extent)
        }
    }

    impl Stage for Table {
        fn is_symbolic(&self, dim: PrimaryDim) -> bool {
            self.symbolic.contains(&dim)
        }
        fn is_corelet_split(&self, dim: PrimaryDim) -> bool {
            self.corelet_split.contains(&dim)
        }
        fn is_row_split(&self, dim: PrimaryDim) -> bool {
            self.row_split.contains(&dim)
        }
        fn is_pe_sfp_split(&self, dim: PrimaryDim) -> bool {
            self.pe_sfp_split.contains(&dim)
        }
        fn splits_any_row(&self) -> bool {
            !self.row_split.is_empty()
        }
        fn extent(&self, dim: PrimaryDim, at: Sample) -> Extent {
            let mut extent = Table::whole(&self.extents, dim);
            if extent > 0 {
                if self.is_corelet_split(dim) {
                    extent /= i64::from(Target::CORELETS_PER_CORE);
                }
                if at.row.is_some() && self.is_row_split(dim) {
                    extent /= i64::from(Target::PT_ROWS);
                }
                if at.comp == Some(VectorComp::Sfp) && self.is_pe_sfp_split(dim) {
                    extent /= 4;
                }
            }
            Extent(extent)
        }
        fn padded_extent(&self, dim: PrimaryDim, _at: Sample) -> Option<PaddedExtent> {
            PaddedExtent::of(Table::whole(&self.padded, dim))
        }
    }

    /// THE PACKED fp8 KERNEL STICK — `[in:2, out:64]`, the one two-dim stick this bridge's own
    /// input side emits, and the vendor's two invariants on it: `crossSlice.size() == 1` and
    /// `withinSlice.size() <= 2` (`ddc/ddcv1.cpp:2032,3580`).
    #[test]
    fn the_slice_and_the_crossing_partition_a_packed_fp8_stick() {
        let dims = StickDims(vec![
            (PrimaryDim::In, Elements(2)),
            (PrimaryDim::Out, Elements(64)),
        ]);
        // 2 * 64 = 128 elements to a stick, over 8 slices: 16 to a slice.
        let slice = SliceElems::per_stick::<Dd2>(&dims).expect("128 elements divide into 8 slices");

        assert_eq!(stick_sizes(&dims, StickPart::Whole), dims.0);
        // The slice is the whole `in` and 8 of the 64 `out`: 2 * 8 = 16.
        assert_eq!(
            stick_sizes(&dims, StickPart::WithinSlice(slice)),
            vec![
                (PrimaryDim::In, Elements(2)),
                (PrimaryDim::Out, Elements(8))
            ]
        );
        // And the stick crosses 8 slices along `out` alone: 16 * 8 = 128.
        assert_eq!(
            stick_sizes(&dims, StickPart::CrossSlice(slice)),
            vec![(PrimaryDim::Out, Elements(8))]
        );
        // ⛔ AND THE CHECK IS THE CONSTRUCTOR: 60 elements do not divide into 8 slices, so there is
        // no slice to ask either arm about.
        assert!(
            SliceElems::per_stick::<Dd2>(&StickDims(vec![(PrimaryDim::Out, Elements(60))]))
                .is_none()
        );
        // `numL0Slices` is `numPTRows`, 8 on this arch, so the L0 slice of this stick is the same 16.
        assert_eq!(SliceElems::per_l0_row::<Dd2>(&dims), Some(slice));
    }

    /// ⛔ A SPLIT DIM IS A PRODUCT, NOT A REPLACEMENT — the only behaviour that distinguishes this
    /// from `stick_sizes` at all.
    #[test]
    fn a_dim_the_stick_names_twice_folds_to_the_product_of_its_extents() {
        // A stick blocked along `out`: the DDL is free to name one primary dim at two positions.
        let split = StickDims(vec![
            (PrimaryDim::Out, Elements(4)),
            (PrimaryDim::In, Elements(3)),
            (PrimaryDim::Out, Elements(8)),
        ]);
        assert_eq!(
            cumulative_stick_sizes(&split, StickPart::Whole),
            Some(vec![
                (PrimaryDim::Out, Elements(32)),
                (PrimaryDim::In, Elements(3)),
            ])
        );
        // An unsplit stick is its own list.
        let plain = StickDims(vec![
            (PrimaryDim::In, Elements(2)),
            (PrimaryDim::Out, Elements(64)),
        ]);
        assert_eq!(
            cumulative_stick_sizes(&plain, StickPart::Whole),
            Some(plain.0.clone())
        );
        assert_eq!(
            cumulative_stick_sizes(&StickDims(Vec::new()), StickPart::Whole),
            Some(Vec::new())
        );
        // ⛔ AND A PRODUCT THAT DOES NOT FIT IS NOT REPORTED AS A SMALL ONE.
        let overflow = StickDims(vec![
            (PrimaryDim::Out, Elements(u64::MAX)),
            (PrimaryDim::Out, Elements(2)),
        ]);
        assert_eq!(cumulative_stick_sizes(&overflow, StickPart::Whole), None);
    }

    /// 🎯 001/110 A PE/SFP SPLIT REPLACES THE ROW NEST, AND BOTH COMPONENTS ARE SAMPLED —
    /// `ddc/ddcv1.cpp:920-933`.
    #[test]
    fn a_pe_sfp_split_is_sampled_per_component_and_never_per_row() {
        // Split BOTH ways: the row nest is what the component nest displaces (`row = -1` at `:929`),
        // so `out` is measured whole on PE and quartered on SFP, never divided by 8 rows.
        let ds = Table {
            row_split: vec![PrimaryDim::Out],
            pe_sfp_split: vec![PrimaryDim::Out],
            extents: vec![(PrimaryDim::Out, 64)],
            ..Table::default()
        };
        let of = |min, max| {
            DimConstraint::of(
                DimSet::of(&[PrimaryDim::Out]).expect("one dim is a set"),
                Constraint {
                    kind: ConstraintKind::Absolute {
                        cannot_be_symbolic: false,
                        dim_kind: None,
                        min,
                    },
                    max,
                    values: None,
                },
            )
            .expect("an absolute constraint has no no-epilogue arm to clash")
        };
        // PE sees 64 and SFP 16; both are multiples of 16, and neither is 8 — the row extent.
        let held = of(AbsoluteMin::Multiple(16.0), None);
        assert!(check_constraints(&ds, &[held], PrimaryDim::Out, true));
        // ⛔ SFP IS SAMPLED: 16 is not a multiple of 32, so the whole check fails on the second comp.
        let sfp_fails = of(AbsoluteMin::Multiple(32.0), None);
        assert!(!check_constraints(&ds, &[sfp_fails], PrimaryDim::Out, true));
        // ⛔ AND PE IS SAMPLED: only PE's 64 exceeds a max of 32.
        let pe_fails = of(AbsoluteMin::Unset, Some(32.0));
        assert!(!check_constraints(&ds, &[pe_fails], PrimaryDim::Out, true));
        // A dim the constraint does not name is not this constraint's business (`:800`).
        let unnamed = of(AbsoluteMin::Multiple(32.0), None);
        assert!(check_constraints(&ds, &[unnamed], PrimaryDim::Mb, true));
    }

    /// 🎯 001/110 ⛔ A ROW-SPLIT STAGE MEASURED AGAINST A ROW-FLAT REFERENCE IS COMPARED WHOLE —
    /// `ddc/ddcv1.cpp:849-852`, and it is what decides the no-epilogue divisibility.
    #[test]
    fn a_row_flat_reference_clears_the_row_from_both_sides_of_the_no_epilogue_test() {
        let ds = Table {
            row_split: vec![PrimaryDim::Ij],
            extents: vec![(PrimaryDim::Ij, 64)],
            ..Table::default()
        };
        let refer = |extent| Table {
            extents: vec![(PrimaryDim::Ij, extent)],
            ..Table::default()
        };
        let of = |refer| {
            DimConstraint::of(
                DimSet::of(&[PrimaryDim::Ij]).expect("one dim is a set"),
                Constraint {
                    kind: ConstraintKind::Relative {
                        reference: refer,
                        min: None,
                        multiple: LoopMultiple::NoEpilogue(NoEpilogueDimKind::Unpadded),
                    },
                    max: None,
                    values: None,
                },
            )
            .expect("one dim and a no-epilogue constraint do pair")
        };
        // Cleared to the full dimension, 64 over 16 divides.
        let divides = refer(16);
        assert!(check_constraints(
            &ds,
            &[of(&divides)],
            PrimaryDim::Ij,
            false
        ));
        // ⛔ AND 64 OVER 24 DOES NOT — the per-row extent 8 would have divided 24 evenly, so this is
        // the assertion that the row was cleared and not merely ignored.
        let epilogue = refer(24);
        assert!(!check_constraints(
            &ds,
            &[of(&epilogue)],
            PrimaryDim::Ij,
            false
        ));
        // `allowEpilogue` skips the whole arm (`:870`).
        assert!(check_constraints(
            &ds,
            &[of(&epilogue)],
            PrimaryDim::Ij,
            true
        ));
        // A dim the reference stage does not have is not a constraint on it at all (`:869`).
        let elsewhere = refer(-1);
        assert!(check_constraints(
            &ds,
            &[of(&elsewhere)],
            PrimaryDim::Ij,
            false
        ));
    }

    /// The two lists of one connect, through the inserts that own them
    /// (`schedule::ddc::metadata`).
    fn ends(producers: &[usize], consumers: &[usize]) -> Ends {
        let mut ends = Ends::default();
        for producer in producers {
            ends.insert_producer(NodeIndex(*producer));
        }
        for consumer in consumers {
            ends.insert_consumer(NodeIndex(*consumer));
        }
        ends
    }

    /// 🎯 002/110 WHICH SIDE EACH END LANDS ON, AND THAT A CONSTANT SOURCE CONSUMES NOTHING —
    /// `ddc/ddcv1.cpp:3290-3321`.
    #[test]
    fn a_transfer_produces_what_a_compute_then_consumes() {
        let census = create_data_connect_metadata(&[
            ScheduleNode::Transfer {
                src: Reads {
                    data_connect: DataConnect::AconstConnect,
                    from_constant: true,
                },
                dsts: vec![DataConnect::PeHtOut],
            },
            ScheduleNode::Compute {
                inputs: vec![Reads {
                    data_connect: DataConnect::PeHtOut,
                    from_constant: false,
                }],
                outputs: vec![DataConnect::OuttensorToSfp],
                opaque_reads: vec![],
                // ⛔ THE SAME NODE ON THE SAME SIDE TWICE IS ONE ENTRY — `insertProducer`
                // deduplicates (`ddc/ddc_metadata.h:147-152`).
                opaque_writes: vec![DataConnect::OuttensorToSfp],
            },
        ]);
        // ⛔ `aconst_connect` IS ABSENT: a CONSTANT source is not a consumer, and nothing else
        // touched it.
        assert_eq!(
            census,
            DataConnects::Census(vec![
                (DataConnect::PeHtOut, ends(&[0], &[1])),
                (DataConnect::OuttensorToSfp, ends(&[1], &[])),
            ])
        );
    }

    /// 🎯 002/110 ⛔ A CONNECT NOTHING WRITES IS ILLEGAL DDL, AND THE ANSWER NAMES IT —
    /// `ddc/ddcv1.cpp:3324-3329`.
    #[test]
    fn an_opaque_body_reading_an_unwritten_connect_names_that_connect() {
        let census = create_data_connect_metadata(&[ScheduleNode::Compute {
            inputs: vec![],
            outputs: vec![DataConnect::PeHtOut],
            opaque_reads: vec![DataConnect::OuttensorToSfp],
            opaque_writes: vec![],
        }]);
        assert_eq!(
            census,
            DataConnects::NoProducer(DataConnect::OuttensorToSfp)
        );
    }
}

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

use crate::arch::{Arch, Elements};

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e001_checkConstraints
// crustify:todo: e002_createDataConnectMetadata

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

#[cfg(test)]
mod unit_tests {
    use super::{PrimaryDim, SliceElems, StickDims, StickPart, stick_sizes};
    use crate::arch::{Dd2, Elements};

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
}
// crustify:todo: e071_getCumulativeStickSizes

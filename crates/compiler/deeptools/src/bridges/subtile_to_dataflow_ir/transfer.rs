//! HOW A TRANSFER IS WALKED — the AGEN time dimensions, ported from `DataTransferLowering.cpp`.
//!
//! ⛔⛔ AT MOST ONE HARDWARE VECTOR PER TIME STEP. "An AGEN composite transfer moves at most one
//! hardware vector per time step, so a transfer wider than that has to walk the remaining elements
//! over AGEN time dimensions instead of widening `load_iv`" (`:306-310`). Everything in this file
//! follows from that one sentence: a transfer either fits in a vector — one pinned time step, zero
//! offsets — or it is split, and then the innermost axis steps by a whole vector while every other
//! non-unit axis steps by one.
//!
//! ⭐ THIS IS SCHEDULING, WHICH IS WHY IT IS NOT IN THE ISLAND. The island holds the op and its
//! types; deciding how many time steps a transfer takes is a lowering decision, and putting it in
//! the data type is what would make it untestable on its own.

use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap, IntegerSet};

/// A HARDWARE VECTOR WIDTH, IN ELEMENTS.
///
/// ⛔⛔ NOT A BYTE COUNT AND NOT DERIVED FROM ONE. `getVectorLanes` reads the arch's SIMD feature —
/// `max(SIMD.getLanes(elem_type), 1)` (`Utils.cpp:58-68`) — and `getLanes` returns **0** for an
/// element type the device does not list, which that `max` turns into **one lane**, not a wide one.
/// So an unlisted type does not fall back to "stick width / element size"; it degrades to a
/// one-element-per-step walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Lanes(u64);

impl Lanes {
    /// The only width the device description declares: `lanes = #ktdf_arch.map<f16 = 64>`
    /// (`sys-arch-spec/KTDFArchGraphDevice/spyre_dd2_basic.mlir:35`, the one device file there is).
    ///
    /// ⭐ AND IT AGREES WITH THE DATAPATH GRANULARITY FROM THE OTHER SIDE:
    /// `#LXLU_CORELET_FIFO` carries `transfer_granularity = array<i64: 64, 2>` (`:53`) — 64 elements
    /// of 2 bytes — and the L3-to-LX datapath carries `array<i64: 128>` (`:20`), the same 128 bytes
    /// counted in bytes rather than elements.
    pub const F16: Lanes = Lanes(64);

    /// A width read from somewhere, for an element type this crate has established a value for.
    ///
    /// ⛔ THERE IS NO `for_elem_type` HERE ON PURPOSE. Writing one would mean answering for fp8, and
    /// the device file does not declare fp8 — by the stock arithmetic above that is ONE lane, not
    /// the 128 that "128-byte stick / 1-byte element" would suggest. Which of those is right has to
    /// be read out of the fp8 templates' own `.ddl`/`.smc` before the fp8 acceptance target, and
    /// guessing it here would put the guess everywhere at once.
    #[must_use]
    pub const fn new(lanes: u64) -> Lanes {
        Lanes(lanes)
    }

    /// The width, for the arithmetic below.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// WHY A TRANSFER COULD NOT BE WALKED.
///
/// ⛔ EACH CARRIES THE VALUES THAT DISAGREED, because "the transfer is too wide" names no defect.
/// The C++ emits the same numbers into its diagnostics (`:341-374`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferError {
    /// Wider than one vector, but one side has no dimensions to split along.
    NothingToSplit {
        /// How many elements the transfer moves.
        total: u64,
        /// The hardware vector width.
        lanes: u64,
    },
    /// Wider than one vector, but an innermost extent is not a whole number of vectors.
    ///
    /// ⛔ BOTH SIDES ARE CARRIED. The C++ reports them together — "requires the innermost source and
    /// destination sizes to be a multiple of the vector width, but they are X and Y" — because
    /// which one is at fault is the whole question.
    InnermostNotWhole {
        /// The source's innermost extent.
        src_innermost: u64,
        /// The destination's innermost extent.
        dst_innermost: u64,
        /// The hardware vector width.
        lanes: u64,
    },
    /// The two sides walk different numbers of steps.
    ///
    /// ⭐ ONLY THE EXTENTS ARE COMPARED, NOT POSITIONS OR COEFFICIENTS: "the two sides may reach the
    /// same walk through different shapes, e.g. src [2, 64] and dst [1, 128] with 64 lanes both walk
    /// 2 steps" (`:365-367`).
    WalkMismatch {
        /// The source's time extents.
        src: Vec<u64>,
        /// The destination's time extents.
        dst: Vec<u64>,
    },
}

/// HOW ONE SIDE OF A TRANSFER TRAVERSES THE TIME AXIS.
///
/// `extents` becomes that side's contribution to `time_set`; `offsets` becomes the results of its
/// `*_time_addr_map`, one per memref dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeDims {
    /// The extent of each time dimension, SLOWEST-VARYING FIRST.
    pub extents: Vec<u64>,
    /// The offset added to each memref index at time step `(d0, .., dn-1)` — one per dimension.
    pub offsets: Vec<AffineExpr>,
}

/// HOW `sizes` IS TRAVERSED OVER TIME — `describeTransferTimeDims` (`:111-143`).
///
/// Every non-unit dimension except the innermost contributes a time dimension stepping by ONE; the
/// innermost contributes one stepping by a WHOLE VECTOR, and only when it holds more than one.
/// IBM's own worked table, at 64 lanes:
///
/// ```text
///   sizes           extents      offsets           time_set
///   [1, 256, 64]    [256]        (0, d0, 0)        (d0) : 0 <= d0 <= 255
///   [1, 1, 128]     [2]          (0, 0, 64 * d0)   (d0) : 0 <= d0 <= 1
///   [2, 4, 8, 64]   [2, 4, 8]    (d0, d1, d2, 0)   3 dims of those extents
///   [1, 64]         []           (0, 0)            nothing walked
/// ```
///
/// ⭐ AN IDENTITY `time_order` IS CORRECT BECAUSE OF THE ORDER THINGS ARE APPENDED. A dimension's
/// number is `extents.len()` at the moment it is added, and they are added outermost first — so d0
/// is the slowest-varying and the last is the fastest.
#[must_use]
pub fn describe_transfer_time_dims(sizes: &[u64], lanes: Lanes) -> TimeDims {
    let mut dims = TimeDims {
        extents: Vec::new(),
        offsets: vec![AffineExpr::Const(0); sizes.len()],
    };

    let add = |dims: &mut TimeDims, at: usize, extent: u64, coeff: i64| {
        let numbered = u32::try_from(dims.extents.len()).expect("a time rank fits a u32");
        dims.offsets[at] = if coeff == 1 {
            AffineExpr::dim(numbered)
        } else {
            AffineExpr::dim(numbered).times(coeff)
        };
        dims.extents.push(extent);
    };

    // Every non-unit dimension except the innermost, stepping by one.
    for (i, &extent) in sizes.iter().enumerate().take(sizes.len().saturating_sub(1)) {
        if extent != 1 {
            add(&mut dims, i, extent, 1);
        }
    }
    // Then the innermost, stepping by a whole vector — but only if there is more than one vector.
    if let Some(&innermost) = sizes.last() {
        let vectors = innermost / lanes.get();
        if vectors > 1 {
            let step = i64::try_from(lanes.get()).expect("a vector width fits an i64");
            add(&mut dims, sizes.len() - 1, vectors, step);
        }
    }
    dims
}

/// EVERYTHING A `composite_load_and_store` NEEDS THAT IS NOT AN ADDRESS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferPlan {
    /// How many elements each loaded vector holds — one hardware vector at most.
    pub vector_lanes: u64,
    /// Which elements form each loaded vector.
    pub load_set: IntegerSet,
    /// How those elements are packed.
    pub load_order: AffineMap,
    /// Which elements form each stored vector.
    pub store_set: IntegerSet,
    /// How those are packed.
    pub store_order: AffineMap,
    /// The time steps the transfer takes.
    pub time_set: IntegerSet,
    /// The order among them.
    pub time_order: AffineMap,
    /// The source offset at each time step.
    pub load_time_addr_map: AffineMap,
    /// The destination offset at each time step.
    pub store_time_addr_map: AffineMap,
}

/// PLAN A MEMORY-TO-MEMORY TRANSFER — `lowerAsLoadAndStore` (`:311-421`).
///
/// `total` is the element count of the whole transfer, which the caller knows from the vector type
/// the two views are being moved through.
///
/// ⭐ THE UNSPLIT CASE IS NOT A DEGENERATE ONE. A transfer that fits in a vector keeps the full
/// sizes as its load/store sets, walks nothing, and takes a single PINNED time step —
/// `affine_set<(d0) : (d0 == 0)>`, which is `#set2` of the reference. The C++ writes that as
/// "`time_extents` empty, so push a 1" (`:397-399`) rather than as a separate path.
///
/// # Errors
///
/// Returns [`TransferError`] where the walk cannot be built: nothing to split along, an innermost
/// extent that is not a whole number of vectors, or two sides that walk different numbers of steps.
pub fn plan(
    src_sizes: &[u64],
    dst_sizes: &[u64],
    total: u64,
    lanes: Lanes,
) -> Result<TransferPlan, TransferError> {
    let mut load_sizes = src_sizes.to_vec();
    let mut store_sizes = dst_sizes.to_vec();
    let mut vector_lanes = total;
    let mut src_time = TimeDims {
        extents: Vec::new(),
        offsets: vec![AffineExpr::Const(0); src_sizes.len()],
    };
    let mut dst_time = TimeDims {
        extents: Vec::new(),
        offsets: vec![AffineExpr::Const(0); dst_sizes.len()],
    };

    if total > lanes.get() {
        let (Some(&src_innermost), Some(&dst_innermost)) = (src_sizes.last(), dst_sizes.last())
        else {
            return Err(TransferError::NothingToSplit {
                total,
                lanes: lanes.get(),
            });
        };
        if src_innermost % lanes.get() != 0 || dst_innermost % lanes.get() != 0 {
            return Err(TransferError::InnermostNotWhole {
                src_innermost,
                dst_innermost,
                lanes: lanes.get(),
            });
        }

        src_time = describe_transfer_time_dims(src_sizes, lanes);
        dst_time = describe_transfer_time_dims(dst_sizes, lanes);
        if src_time.extents != dst_time.extents {
            return Err(TransferError::WalkMismatch {
                src: src_time.extents,
                dst: dst_time.extents,
            });
        }

        // One vector per step: every axis pinned but the innermost, which holds exactly the lanes.
        vector_lanes = lanes.get();
        load_sizes = vec![1; src_sizes.len()];
        store_sizes = vec![1; dst_sizes.len()];
        if let Some(last) = load_sizes.last_mut() {
            *last = lanes.get();
        }
        if let Some(last) = store_sizes.last_mut() {
            *last = lanes.get();
        }
    }

    // A transfer of at most one vector walks nothing, and the single pinned step is written as an
    // extent of one rather than as an empty time set.
    let mut time_extents = src_time.extents.clone();
    if time_extents.is_empty() {
        time_extents.push(1);
    }
    let time_rank = u32::try_from(time_extents.len()).expect("a time rank fits a u32");

    Ok(TransferPlan {
        vector_lanes,
        load_set: IntegerSet::from_sizes(&load_sizes),
        load_order: AffineMap::identity(u32::try_from(src_sizes.len()).expect("a rank fits a u32")),
        store_set: IntegerSet::from_sizes(&store_sizes),
        store_order: AffineMap::identity(
            u32::try_from(dst_sizes.len()).expect("a rank fits a u32"),
        ),
        time_set: IntegerSet::from_sizes(&time_extents),
        time_order: AffineMap::identity(time_rank),
        load_time_addr_map: AffineMap {
            dims: time_rank,
            results: src_time.offsets,
        },
        store_time_addr_map: AffineMap {
            dims: time_rank,
            results: dst_time.offsets,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::{Lanes, TransferError, describe_transfer_time_dims, plan};
    use crate::islands::dataflow_ir::ty::AffineExpr;

    /// ⭐⭐ IBM'S FOUR WORKED ROWS, CARRIED AS VALUES.
    ///
    /// The table is `DataTransferLowering.cpp:111-123`, written by the author of the function to
    /// say what it does. Reproducing it is what distinguishes a port from a paraphrase: each row
    /// pins the extents AND the per-dimension offsets, so stepping the wrong axis or by the wrong
    /// amount is a diff rather than a plausible-looking walk.
    #[test]
    fn walks_ibms_worked_examples() {
        let lanes = Lanes::F16;
        let d = AffineExpr::dim;
        let zero = AffineExpr::Const(0);

        // [1, 256, 64] -> extents [256], offsets (0, d0, 0): the middle axis steps by ONE, and the
        // innermost is exactly one vector so it contributes nothing.
        let walk = describe_transfer_time_dims(&[1, 256, 64], lanes);
        assert_eq!(walk.extents, vec![256]);
        assert_eq!(walk.offsets, vec![zero.clone(), d(0), zero.clone()]);

        // [1, 1, 128] -> extents [2], offsets (0, 0, 64 * d0): TWO vectors on the innermost axis,
        // stepping by a whole vector rather than by one.
        let walk = describe_transfer_time_dims(&[1, 1, 128], lanes);
        assert_eq!(walk.extents, vec![2]);
        assert_eq!(
            walk.offsets,
            vec![zero.clone(), zero.clone(), d(0).times(64)]
        );

        // [2, 4, 8, 64] -> extents [2, 4, 8], offsets (d0, d1, d2, 0): three outer axes walked,
        // slowest-varying first, which is what makes an identity `time_order` correct.
        let walk = describe_transfer_time_dims(&[2, 4, 8, 64], lanes);
        assert_eq!(walk.extents, vec![2, 4, 8]);
        assert_eq!(walk.offsets, vec![d(0), d(1), d(2), zero.clone()]);

        // [1, 64] -> nothing walked at all.
        let walk = describe_transfer_time_dims(&[1, 64], lanes);
        assert!(walk.extents.is_empty());
        assert_eq!(walk.offsets, vec![zero.clone(), zero]);
    }

    /// ⭐⭐ THE REFERENCE'S OWN TRANSFER, ATTRIBUTE FOR ATTRIBUTE.
    ///
    /// `/tmp/ktir_ref/export/debug/dfir.mlir:78-84` moves `memref<12x64x64xf16>` into
    /// `memref<2x2x1x1x64xf16>` one 64-lane vector at a time. Its access covers `[1, 1, 64]` on the
    /// source and `[1, 1, 1, 1, 64]` on the destination — 64 elements, exactly one vector — so the
    /// planner must take the UNSPLIT path and emit the single pinned time step.
    ///
    /// Every attribute below is the reference's, by name: `#set`, `#set1`, `#set2`, `#map2`,
    /// `#map4`, `#map3`, `#map5`, `#map6`.
    #[test]
    fn reproduces_the_references_transfer_attributes() {
        let got = plan(&[1, 1, 64], &[1, 1, 1, 1, 64], 64, Lanes::F16)
            .expect("64 elements at 64 lanes is a single unsplit vector");

        assert_eq!(
            got.vector_lanes, 64,
            "one hardware vector, not the transfer"
        );

        use crate::islands::dataflow_ir::print::{affine_map as mapped, integer_set as printed};

        // #set
        assert_eq!(
            printed(&got.load_set),
            "affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0)>"
        );
        // #set1
        assert_eq!(
            printed(&got.store_set),
            "affine_set<(d0, d1, d2, d3, d4) : (d0 == 0, d1 == 0, d2 == 0, d3 == 0, d4 >= 0, \
             -d4 + 63 >= 0)>"
        );
        // #set2 — the single pinned time step of a transfer that walks nothing.
        assert_eq!(printed(&got.time_set), "affine_set<(d0) : (d0 == 0)>");
        // #map2, #map4 — the two identities.
        assert_eq!(
            mapped(&got.load_order),
            "affine_map<(d0, d1, d2) -> (d0, d1, d2)>"
        );
        assert_eq!(
            mapped(&got.store_order),
            "affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>"
        );
        // #map3, #map5 — one time dimension in, one zero offset per memref dimension out.
        assert_eq!(
            mapped(&got.load_time_addr_map),
            "affine_map<(d0) -> (0, 0, 0)>"
        );
        assert_eq!(
            mapped(&got.store_time_addr_map),
            "affine_map<(d0) -> (0, 0, 0, 0, 0)>"
        );
        // #map6
        assert_eq!(mapped(&got.time_order), "affine_map<(d0) -> (d0)>");
    }

    /// ⛔ AND THE REFUSALS CARRY THE NUMBERS THAT DISAGREED.
    ///
    /// A transfer wider than a vector whose innermost extent is not a whole number of vectors
    /// cannot be split, and a walk the two sides disagree on is not a walk. Both are refusals in
    /// the C++ (`:341-374`); neither may become a silently truncated transfer.
    #[test]
    fn refuses_what_the_cpp_refuses() {
        // 100 elements at 64 lanes: wider than a vector, and 100 is not a multiple of 64.
        assert_eq!(
            plan(&[1, 100], &[1, 100], 100, Lanes::F16),
            Err(TransferError::InnermostNotWhole {
                src_innermost: 100,
                dst_innermost: 100,
                lanes: 64,
            })
        );

        // Both sides a whole number of vectors, but they walk different numbers of steps:
        // src [4, 64] walks 4, dst [2, 64] walks 2.
        assert_eq!(
            plan(&[4, 64], &[2, 64], 256, Lanes::F16),
            Err(TransferError::WalkMismatch {
                src: vec![4],
                dst: vec![2],
            })
        );

        // ⭐ AND THE SHAPES NEED NOT MATCH, ONLY THE EXTENTS — the C++ says so in as many words:
        // "src [2, 64] and dst [1, 128] with 64 lanes both walk 2 steps".
        let got = plan(&[2, 64], &[1, 128], 128, Lanes::F16)
            .expect("two shapes reaching the same two-step walk");
        assert_eq!(got.vector_lanes, 64);
    }
}

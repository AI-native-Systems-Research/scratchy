// SPDX-License-Identifier: Apache-2.0
//! THE SHAPES THE LOWERING ACCEPTS — legality checks, as types where possible.
//!
//! Ported from `AgenToSentientLoweringPass`: `checkCompositeRegion` (`Helper.cpp:206`),
//! `checkStoreOpFromExtractPattern` (`:433`), `isLoadAndExtractScalarPattern` (`:463`),
//! `isReceiveAndExtractScalarPattern` (`:493`), `checkIndirectMemViewForExtractOp` (`:388`), and
//! `getStoreProducer`'s producer requirement (`:1285`).
//!
//! ⭐ EVERY ONE OF THESE IS A PREDICATE THE REFERENCE ANSWERS BY WALKING AN IR IT RECEIVED. We build
//! the IR, so the useful half is *what shape is legal*, and the checks become constructors that admit
//! only that.

use crate::islands::dataflow_ir::dialects::Val;
use crate::units::DfirUnit;

/// THE VIRTUAL IBR — a unit these patterns require and our island cannot yet name.
///
/// ⛔⛔ `SenComponents::LXVIRTUALIBR` IS LOAD-BEARING IN THREE OF THESE CHECKS, and
/// [`crate::units::DfirUnit`] has no variant for it. `isLoadAndExtractScalarPattern` requires the
/// STORE side to be on it (`:487`), `isReceiveAndExtractScalarPattern` requires it (`:506`), and
/// `checkIndirectMemViewForExtractOp` refuses anything else with *"indirect memory view is not
/// operating on a virtual IBR"* (`:404`).
///
/// ⚠️ **SO THE EXTRACT-SCALAR PATTERNS CANNOT BE EMITTED UNTIL `DfirUnit` GAINS THIS UNIT.** That is a
/// gap in the emitter, recorded here rather than worked around: the alternative — treating the LX as
/// the virtual IBR because it is the nearest thing we can name — would emit memory views on the wrong
/// unit and the reference would refuse them with the message above.
///
/// ⛔ AND THE REFERENCE WARNS WHY THE OBVIOUS SHORTCUT FAILS, in a comment on
/// `isReceiveAndExtractScalarPattern` (`:502-504`): *"Can't use dcc::getUnitType() here because the
/// memory views use components in the form L0/LX/PE etc. and these don't exist in the
/// senCompToGenericComp enum."* A memory view names a MEMORY, not an execution unit, and the two
/// vocabularies are different — which is the same distinction that keeps `Lx` and `Hbm` in `DfirUnit`
/// beside the load and store halves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VirtualIbrOwed;

/// A COMPOSITE LOAD'S REGION — `checkCompositeRegion` (`Helper.cpp:206-294`).
///
/// ⛔⛔ TWO OR THREE OPS, AND THE REFERENCE'S ERROR NAMES THE WHOLE SET: *"composite_load's region size
/// must be 2 or 3 and only dataflow.send, vectorchain.select + dataflow.send, or vectorchain.shuffle +
/// dataflow.send are allowed!"* (`:216-220`). So the region is a send, optionally preceded by one
/// select or shuffle, then the yield.
///
/// ⭐ NOTE IT IS **SELECT OR SHUFFLE ONLY** HERE — not the select/shuffle/**rotate** trio that
/// [`super::load_chain::Rearrangement`] admits outside a composite region (`:236-238` against
/// `:1268-1269`). The two sets are genuinely different and reading one for the other admits a rotate
/// where the reference refuses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeRegion {
    /// `dataflow.send` then `agen.yield` — region size 2.
    JustSend {
        /// The value sent, which must be the composite's own.
        data: Val,
    },
    /// One rearrangement, then the send, then the yield — region size 3.
    ///
    /// ⛔ THE SEND'S DATA MUST BE THE REARRANGEMENT'S RESULT. The reference checks that the send's
    /// defining op *is* the op before it (`:234-241`); a send carrying something else with a stray
    /// select in the region is the error.
    RearrangedThenSend {
        /// Which of the two — ⛔ NOT `rotate`; see the type's note.
        via: CompositeRearrangement,
        /// The rearrangement's result, which the send must carry.
        data: Val,
    },
}

/// THE TWO REARRANGEMENTS LEGAL INSIDE A COMPOSITE REGION.
///
/// ⛔ NO `Rotate`. See [`CompositeRegion`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeRearrangement {
    /// `vectorchain.select`.
    Select,
    /// `vectorchain.shuffle`.
    Shuffle,
}

impl CompositeRegion {
    /// How many ops the region holds, including the yield — the reference's `region_size`.
    #[must_use]
    pub const fn region_size(&self) -> usize {
        match self {
            CompositeRegion::JustSend { .. } => 2,
            CompositeRegion::RearrangedThenSend { .. } => 3,
        }
    }

    /// What the send carries.
    #[must_use]
    pub const fn sent(&self) -> Val {
        match self {
            CompositeRegion::JustSend { data } | CompositeRegion::RearrangedThenSend { data, .. } => {
                *data
            }
        }
    }
}

/// A ONE-DIMENSIONAL, ZERO-BASED, IDENTITY-MAPPED STORE — `checkStoreOpFromExtractPattern`
/// (`Helper.cpp:433-459`).
///
/// ⛔⛔ FOUR CONDITIONS, ALL OF THEM SEPARATE ERRORS IN THE REFERENCE, and every one of them is about
/// the store being degenerate rather than about what it stores:
///
/// * exactly one index (*"expecting indices size 1"*)
/// * that index a constant **zero** (*"expecting start offset of 0 from vector_store"*)
/// * a 1-D `store_set` (*"expecting 1D store_set"*)
/// * a 1-D **identity** `store_order` (*"expecting 1D identity map for store_map"*)
///
/// ⭐ TOGETHER THEY SAY: THE STORE WRITES ONE ELEMENT AT OFFSET ZERO AND REARRANGES NOTHING. That is
/// what makes it an extract rather than a transfer, so the four conditions are one fact and this is one
/// witness rather than four booleans a caller must remember to check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DegenerateStore(());

impl DegenerateStore {
    /// A store that meets all four conditions, or `None`.
    ///
    /// ⛔ `None` MEANS "NOT AN EXTRACT PATTERN", which is a classification. The caller has other
    /// patterns to try; the reference emits four different errors here because it has already decided
    /// this must be one.
    #[must_use]
    pub fn of(indices: usize, start_offset: i64, set_dims: usize, order_is_1d_identity: bool) -> Option<DegenerateStore> {
        (indices == 1 && start_offset == 0 && set_dims == 1 && order_is_1d_identity)
            .then_some(DegenerateStore(()))
    }
}

/// WHAT MAY PRODUCE A STORED VALUE — `getStoreProducer`'s requirement (`Helper.cpp:1300-1303`).
///
/// ⛔⛔ THE REFERENCE ASSERTS IT: *"expecting a ReceiveOp or a ShuffleOp as input to VectorStoreOp"*.
/// And for a composite store it admits a third route — a `vectorchain.coalesce` whose own data is a
/// receive (`:1315-1325`), refusing anything else with *"must be preceded by dataflow.receiveOp"*.
///
/// ⭐ SO EVERY LEGAL PRODUCER BOTTOMS OUT AT A RECEIVE. A shuffle or a coalesce may sit between, but
/// the value being stored always came off a wire — which is why
/// [`crate::islands::dataflow_ir::dialects::dataflow::Received`] is the witness a store should be
/// spending.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreProducer {
    /// The receive itself.
    Receive,
    /// A shuffle over a receive.
    ShuffleOverReceive,
    /// A coalesce over a receive — composite stores only.
    CoalesceOverReceive,
}

/// WHETHER A UNIT MAY HOLD A MEMORY VIEW THESE PATTERNS ADDRESS.
///
/// ⛔ THE LOAD SIDE OF `isLoadAndExtractScalarPattern` REQUIRES THE **`LX`** MEMORY (`:474`), not the
/// LX load unit — a memory view names a memory. The store side requires the virtual IBR, which we
/// cannot name; see [`VirtualIbrOwed`].
#[must_use]
pub const fn is_lx_memory(unit: DfirUnit) -> bool {
    matches!(unit, DfirUnit::Lx)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 🎯 A COMPOSITE REGION IS TWO OR THREE OPS, AND ONLY SELECT OR SHUFFLE MAY INTERVENE.
    #[test]
    fn the_composite_region_shapes() {
        assert_eq!(CompositeRegion::JustSend { data: Val(1) }.region_size(), 2);
        let r = CompositeRegion::RearrangedThenSend {
            via: CompositeRearrangement::Shuffle,
            data: Val(2),
        };
        assert_eq!(r.region_size(), 3);
        assert_eq!(r.sent(), Val(2));
    }

    /// 🎯 ALL FOUR DEGENERACY CONDITIONS ARE REQUIRED, and each alone is enough to fail.
    #[test]
    fn a_degenerate_store_needs_all_four() {
        assert!(DegenerateStore::of(1, 0, 1, true).is_some());
        assert!(DegenerateStore::of(2, 0, 1, true).is_none(), "two indices");
        assert!(DegenerateStore::of(1, 4, 1, true).is_none(), "non-zero offset");
        assert!(DegenerateStore::of(1, 0, 2, true).is_none(), "2D store_set");
        assert!(DegenerateStore::of(1, 0, 1, false).is_none(), "non-identity order");
    }

    /// 🎯 THE LOAD SIDE ADDRESSES THE LX **MEMORY**, not a load unit.
    #[test]
    fn the_load_side_is_the_lx_memory() {
        assert!(is_lx_memory(DfirUnit::Lx));
        assert!(!is_lx_memory(DfirUnit::Lxlu), "the load unit is not the memory");
        assert!(!is_lx_memory(DfirUnit::Hbm));
    }
}

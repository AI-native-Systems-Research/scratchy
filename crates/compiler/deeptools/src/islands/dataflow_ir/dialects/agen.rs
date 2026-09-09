//! `Agen.td` — THE ADDRESS GENERATOR'S ACCESSES AND COMPOSITE TRANSFERS.
//!
//! The dialect declares fifteen operations. Four of the eight here are the ones an emitted program
//! contains; the two symbolic accesses are the input of the `AgenToSentient` lowerings that this
//! crate ports (entries 374 and 375), which is why they are spelled even though our own producer
//! emits the affine pair — see [`Op::SymbolicVectorLoad`]. The interleave and the mask state are the
//! input of the two sweeps `e384_runOnOperation` runs after the fusion (entries 271 and 218).

use std::fmt::Write as _;

use crate::arch::Elements;
use crate::islands::dataflow_ir::dialects::{Index, Val};
use crate::islands::dataflow_ir::print;
use crate::islands::dataflow_ir::ty::{
    AffineExpr, AffineMap, Constraint, IntegerSet, MemRef, Vector,
};

/// A `composite_load_and_store`'s operands and attributes.
///
/// See [`Op::CompositeLoadAndStore`] for what the op means and why it is the thing that gets a
/// weight out of the HBM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeTransfer {
    /// The view read from.
    pub src: Val,
    /// Its subscript.
    pub src_indices: Vec<Index>,
    /// Its type.
    pub src_ty: MemRef,
    /// The view written to.
    pub dst: Val,
    /// Its subscript.
    pub dst_indices: Vec<Index>,
    /// Its type.
    pub dst_ty: MemRef,
    /// The block argument carrying the vector loaded at each time step.
    pub load_iv: Val,
    /// That vector's type — ONE hardware vector, never the whole transfer.
    pub load_iv_ty: Vector,
    /// Which elements form each loaded vector.
    pub load_set: IntegerSet,
    /// How those elements are packed.
    pub load_order: AffineMap,
    /// Which elements form each stored vector.
    pub store_set: IntegerSet,
    /// How those are packed.
    pub store_order: AffineMap,
    /// `$time_symbols` — the values this transfer's `time_set` writes its symbolic bounds against
    /// (`Agen.td:436`, `Variadic<Index>`). ⛔ THE VENDOR'S OWN KEY CARRIES ONE: `time_symbols(%c3)`
    /// beside `-d0 + s0 - 1 >= 0` (`mutable_addr_splitting_time_dims.mlir:130`, `:136`), and without
    /// it `calculateTimeBounds` cannot resolve that dimension at all.
    pub time_symbols: Vec<Val>,
    /// The time steps the transfer takes — a single pinned step when it fits in one vector.
    pub time_set: IntegerSet,
    /// The order among them.
    pub time_order: AffineMap,
    /// The source offset at each time step, one result per source dimension.
    pub load_time_addr_map: AffineMap,
    /// The destination offset at each time step, one result per destination dimension.
    pub store_time_addr_map: AffineMap,
    /// `$dir` — *"`$dir` and `$multicast_info`, when present, carry routing information for the
    /// transfer"* (`Agen.td:306-307`, `:340`).
    pub dir: Option<RoutingDirection>,
    /// `$multicast_info` — the group this transfer's destinations form (`Agen.td:341`).
    ///
    /// ⛔ IT IS WHAT MAKES A MULTICAST A MULTICAST. `constructLoadAndStoreStmt` copies it straight
    /// onto the `sentient.load_and_store` (`Helper.cpp:2293-2295`), so a transfer that loses it here
    /// is emitted as a transfer to ONE destination.
    pub multicast_info: Option<Val>,
    /// `dbgName` — the scheduler's name for this transfer.
    ///
    /// ⛔ TWO PORTS READ IT: `e268_constructLoadAndStoreStmt` copies it onto the
    /// `sentient.load_and_store`, and `e267_constructTimeLoopsAndVectorOperations` builds each time
    /// loop's `Time-Loop(<name>, t-dim N)` out of it (`l3-burst-calc.mlir:759`, `:364`, `:370`).
    pub dbg_name: Option<String>,
    /// The region, entered once per time step.
    pub body: Vec<super::Op>,
}

/// A `composite_load`'s operands and attributes.
///
/// See [`Op::CompositeLoad`] for what the op means. It is one side of a [`CompositeTransfer`]: one
/// view, one element set and order, and the same time triple, with the vector loaded at each step
/// reaching its consumers inside the region instead of a second view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeAccess {
    /// `$mem_ref` — the view read from.
    pub view: Val,
    /// Its subscript — `$affine_map` and `$map_operands` written inline, as everything in this
    /// island is (see [`Index::Strided`]).
    pub indices: Vec<Index>,
    /// Its type.
    pub view_ty: MemRef,
    /// The block argument carrying the vector loaded at each time step —
    /// `getLoadInductionVar()` (`Agen.td:454-456`).
    pub load_iv: Val,
    /// That vector's type — ONE hardware vector, as [`CompositeTransfer::load_iv_ty`].
    pub load_iv_ty: Vector,
    /// `$load_set` — which elements form each loaded vector.
    pub load_set: IntegerSet,
    /// `$load_order` — how those elements are packed.
    pub load_order: AffineMap,
    /// `$time_symbols` — the values the time set's bounds are written against. ⛔ NON-EMPTY HERE
    /// WHERE THE ISLAND'S TRANSFERS CARRY NONE: the vendor's own key prints `time_symbols(%c3)`
    /// against a `time_set` with an `s0` (`paged_mem_view_loads.mlir:332-337`).
    pub time_symbols: Vec<Val>,
    /// `$time_set` — the time steps the load takes.
    pub time_set: IntegerSet,
    /// `$time_order` — the order among them.
    pub time_order: AffineMap,
    /// `$time_addr_map` — the offset at each time step, one result per view dimension.
    pub time_addr_map: AffineMap,
    /// `dbgName` — the scheduler's name for this load.
    pub dbg_name: Option<String>,
    /// The region, entered once per time step.
    pub body: Vec<super::Op>,
}

/// WHERE A `composite_store` GETS THE VECTOR IT WRITES.
///
/// ⛔⛔ THE TWO FORMS ARE MUTUALLY EXCLUSIVE AND THE VERIFIER SAYS SO: *"either store_set/order or
/// input vector has to be present in composite store"* (`Agen.cpp:786-790`). The operand form carries
/// no region; the region form carries no operand, so a struct with both would spell a state the op
/// cannot hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositeStoreSource {
    /// `input_vector=%v` — the COALESCE store, whose vector is already packed
    /// (`int8-kg3-l0su-compositestore_coalesce.mlir:87`).
    ///
    /// ⛔ `store_order` IS NOT THE USER'S HERE: the parser overwrites it with the identity map over
    /// `affine_map`'s results *"for coalesce store, store_order attr is not allowed from users"*
    /// (`Agen.cpp:771-776`), and the printer elides it and `store_set` both — which is why neither is
    /// a field of this arm.
    InputVector {
        /// `$input_vector`.
        value: Val,
        /// Its type.
        ty: Vector,
    },
    /// The region form: the vector written is the one the region hands back
    /// (`l3-core2core-multicast-burst.mlir:116-124`).
    Region {
        /// `$store_set` — which elements form each stored vector.
        store_set: IntegerSet,
        /// `$store_order` — how those are packed.
        store_order: AffineMap,
        /// The value the region's terminator yields — `agen.yield %data : vector<..>`.
        ///
        /// ⛔ IT IS A FIELD BECAUSE THIS ISLAND'S [`Op::Yield`] IS OPERAND-LESS, and the region's
        /// block binds NO argument (`Agen.cpp:884-886`): a store receives its data inside the region
        /// instead of on one. [`body`](Self::Region::body) therefore holds the ops BEFORE the
        /// terminator and the terminator is printed from this pair.
        stored: Val,
        /// Its type.
        stored_ty: Vector,
        /// The region, entered once per time step, terminator excluded.
        body: Vec<super::Op>,
    },
}

/// A `composite_store`'s operands and attributes.
///
/// See [`Op::CompositeStore`] for what the op means. It is [`CompositeAccess`] written the other way
/// round: one view, the same time quadruple, and a [`CompositeStoreSource`] where the load carries a
/// `load_iv` (`Agen.td:486-554`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeStoreAccess {
    /// `$mem_ref` — the view written to.
    pub view: Val,
    /// Its subscript — `$affine_map` and `$map_operands` written inline.
    pub indices: Vec<Index>,
    /// Its type.
    pub view_ty: MemRef,
    /// Which of the two forms this store is.
    pub source: CompositeStoreSource,
    /// `$time_symbols` — the values the time set's bounds are written against.
    pub time_symbols: Vec<Val>,
    /// `$time_set` — the time steps the store takes.
    pub time_set: IntegerSet,
    /// `$time_order` — the order among them.
    pub time_order: AffineMap,
    /// `$time_addr_map` — the offset at each time step, one result per view dimension.
    pub time_addr_map: AffineMap,
    /// `dbgName` — the scheduler's name for this store.
    pub dbg_name: Option<String>,
}

/// A `composite_indirect_load`'s operands and attributes.
///
/// See [`Op::CompositeIndirectLoad`] for what the op means. One [`IndirectAccess`] holding the
/// ADDRESSES and one direct view holding the data, then [`CompositeAccess`]'s load pair and time
/// quadruple (`Agen.td:677-757`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeIndirectLoadAccess {
    /// `$indirect_memref` and its access — ⛔ NOT OPTIONAL HERE, unlike the two-sided op's
    /// (`Agen.td:677-690`): a one-sided indirect load without its address view is not this op.
    pub indirect: IndirectAccess,
    /// `$direct_memref` — the view the DATA is read from, and the one every access record reads.
    pub direct: Val,
    /// `$direct_map_indices`.
    pub direct_indices: Vec<Index>,
    /// The direct view's type.
    pub direct_ty: MemRef,
    /// `$multicast_info`, printed as an operand ahead of `time_symbols` (`Agen.cpp:1581`).
    pub multicast_info: Option<Val>,
    /// The block argument carrying the vector loaded at each time step — `getLoadInductionVar()`.
    pub load_iv: Val,
    /// That vector's type — ONE hardware vector.
    pub load_iv_ty: Vector,
    /// `$load_set` — which elements form each loaded vector.
    pub load_set: IntegerSet,
    /// `$load_order` — how those elements are packed.
    pub load_order: AffineMap,
    /// `$time_symbols`.
    pub time_symbols: Vec<Val>,
    /// `$time_set`.
    pub time_set: IntegerSet,
    /// `$time_order`.
    pub time_order: AffineMap,
    /// `$time_addr_map` — the offset at each time step, over the DIRECT view's dimensions.
    pub time_addr_map: AffineMap,
    /// `dbgName`.
    pub dbg_name: Option<String>,
    /// The region, entered once per time step.
    pub body: Vec<super::Op>,
}

/// A `composite_indirect_store`'s operands and attributes.
///
/// See [`Op::CompositeIndirectStore`] for what the op means. [`CompositeIndirectLoadAccess`] with the
/// store pair in place of the load pair and NO `load_iv` — the reference declares
/// `getLoadInductionVar()` on the indirect load only (`Agen.td:758-838`), so the stored vector comes
/// out of the region as [`CompositeStoreSource::Region::stored`] does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeIndirectStoreAccess {
    /// `$indirect_memref` and its access, on the same terms as the load's.
    pub indirect: IndirectAccess,
    /// `$direct_memref` — the view the DATA is written to.
    pub direct: Val,
    /// `$direct_map_indices`.
    pub direct_indices: Vec<Index>,
    /// The direct view's type.
    pub direct_ty: MemRef,
    /// `$multicast_info` — ⛔ PRINTED AFTER THE FIRST NEWLINE HERE, where the load prints it before
    /// (`Agen.cpp:1803-1805` against `:1581`).
    pub multicast_info: Option<Val>,
    /// `$store_set` — which elements form each stored vector.
    pub store_set: IntegerSet,
    /// `$store_order` — how those are packed.
    pub store_order: AffineMap,
    /// The value the region's terminator yields — see [`CompositeStoreSource::Region::stored`].
    pub stored: Val,
    /// Its type.
    pub stored_ty: Vector,
    /// `$time_symbols`.
    pub time_symbols: Vec<Val>,
    /// `$time_set`.
    pub time_set: IntegerSet,
    /// `$time_order`.
    pub time_order: AffineMap,
    /// `$time_addr_map`.
    pub time_addr_map: AffineMap,
    /// `dbgName`.
    pub dbg_name: Option<String>,
    /// The region, entered once per time step, terminator excluded.
    pub body: Vec<super::Op>,
}

/// WHICH WAY ROUND THE RING A TRANSFER IS ROUTED — `AgenRoutingDirection`
/// (`AgenEnums.td:29-42`), whose four cases and their order are the four
/// [`sentient::RoutingDirection`](crate::islands::sentient::dialects::sentient::RoutingDirection)
/// cases and theirs. `e268_constructLoadAndStoreStmt` is the map between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingDirection {
    /// `PseudoRandom` — case 0.
    PseudoRandom,
    /// `CounterClockwise` — case 1.
    CounterClockwise,
    /// `Clockwise` — case 2.
    Clockwise,
    /// `BothWays` — case 3.
    BothWays,
}

impl RoutingDirection {
    /// The spelling inside `#agen<direction ..>`
    /// (`dcc/test/Transform/TransformPagedMemView/paged_mem_view_load_and_store.mlir:101`).
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::PseudoRandom => "PseudoRandom",
            Self::CounterClockwise => "CounterClockwise",
            Self::Clockwise => "Clockwise",
            Self::BothWays => "BothWays",
        }
    }
}

/// ONE OF A COMPOSITE INDIRECT TRANSFER'S TWO OPTIONAL INDIRECT ACCESSES — the view of ADDRESSES a
/// gather reads its source offsets from, or a scatter its destination offsets.
///
/// ⛔ THE THREE MEMBERS ARE ONE FACT. `hasIndirectSrc()`/`hasIndirectDst()` are operand PRESENCE
/// (`Agen.td:656-661`), so a view without its subscript or its type is a state the op cannot hold —
/// which is why this is one [`Option`] and not three.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndirectAccess {
    /// `$indirect_src_memref` / `$indirect_dst_memref`.
    pub view: Val,
    /// Its subscript — `$indirect_*_map` and `$indirect_*_map_indices` written inline, as everything
    /// in this island is (see [`Index::Strided`]).
    pub indices: Vec<Index>,
    /// Its type. Its elements are addresses, so this is the `memref<32xi32>` of the vendor's key and
    /// not the transfer's data type.
    pub ty: MemRef,
}

/// A `composite_indirect_load_and_store`'s operands and attributes.
///
/// See [`Op::CompositeIndirectLoadAndStore`] for what the op means. It is
/// [`CompositeTransfer`] with an optional [`IndirectAccess`] on each side, a second time-address map
/// per side to go with it, and NO `$dir` — the indirect op declares no routing direction
/// (`Agen.td:576-605`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeIndirectTransfer {
    /// `$indirect_src_memref` and its access — [`Some`] makes this transfer a GATHER.
    pub indirect_src: Option<IndirectAccess>,
    /// `$direct_src_memref` — the data read, always present.
    pub direct_src: Val,
    /// Its subscript.
    pub direct_src_indices: Vec<Index>,
    /// Its type.
    pub direct_src_ty: MemRef,
    /// `$indirect_dst_memref` and its access — [`Some`] makes this transfer a SCATTER.
    pub indirect_dst: Option<IndirectAccess>,
    /// `$direct_dst_memref` — the data written, always present.
    pub direct_dst: Val,
    /// Its subscript.
    pub direct_dst_indices: Vec<Index>,
    /// Its type.
    pub direct_dst_ty: MemRef,
    /// The block argument carrying the vector loaded at each time step.
    pub load_iv: Val,
    /// That vector's type — ONE hardware vector, as [`CompositeTransfer::load_iv_ty`].
    pub load_iv_ty: Vector,
    /// Which elements form each loaded vector.
    pub load_set: IntegerSet,
    /// How those elements are packed.
    pub load_order: AffineMap,
    /// Which elements form each stored vector.
    pub store_set: IntegerSet,
    /// How those are packed.
    pub store_order: AffineMap,
    /// `$time_symbols` — the values this transfer's `time_set` writes its symbolic bounds against
    /// (`Agen.td:436`, `Variadic<Index>`). ⛔ THE VENDOR'S OWN KEY CARRIES ONE: `time_symbols(%c3)`
    /// beside `-d0 + s0 - 1 >= 0` (`mutable_addr_splitting_time_dims.mlir:130`, `:136`), and without
    /// it `calculateTimeBounds` cannot resolve that dimension at all.
    pub time_symbols: Vec<Val>,
    /// The time steps the transfer takes.
    pub time_set: IntegerSet,
    /// The order among them.
    pub time_order: AffineMap,
    /// `$load_indirect_time_addr_map` — the ADDRESS view's offset at each time step. ⛔ [`None`] is
    /// `getEmptyAffineMap()`, which is what `cloneWithNewAccessInfo` substitutes for an absent one
    /// (`Agen.cpp:1404-1406`) and what the vendor's key prints as `affine_map<() -> ()>`.
    pub load_indirect_time_addr_map: Option<AffineMap>,
    /// `$load_direct_time_addr_map` — the data source's offset at each time step.
    pub load_direct_time_addr_map: AffineMap,
    /// `$store_indirect_time_addr_map`, on the same terms as its load twin.
    pub store_indirect_time_addr_map: Option<AffineMap>,
    /// `$store_direct_time_addr_map` — the data destination's offset at each time step.
    pub store_direct_time_addr_map: AffineMap,
    /// `$multicast_info` — the group this transfer's destinations form, as
    /// [`CompositeTransfer::multicast_info`].
    pub multicast_info: Option<Val>,
    /// `dbgName` — the scheduler's name for this transfer.
    pub dbg_name: Option<String>,
    /// The region, entered once per time step.
    pub body: Vec<super::Op>,
}

/// WHICH ELEMENTS ONE ACCESS TOUCHES — the `load_set`/`store_set` an `agen` access carries.
///
/// ⛔⛔ TWO PRODUCERS ANSWER THIS QUESTION AND ONLY ONE OF THEM CAN DERIVE IT. A whole-stick access
/// is [`access_set`] over the view's rank, which is what every access the subtile bridge emits is
/// and why the printer computed it rather than storing it. A TRANSFER's is
/// `constructLoadOrStoreSet`'s (`SNTransferLowering.cpp:135-222`): chunked by
/// `unitTimeTransferChunkSize_`, strided by `unitTimeTransferChunkStride_` and repeated
/// `unitTimeTransferNumChunks_` times — a set no view type implies. See
/// [`crate::bridges::superdsc_to_dataflow_ir::transfer::load_or_store_set`].
///
/// ⭐ THE ORDER STAYS DERIVED. `load_order` is `getMultiDimIdentityMap(getLayoutMap().getNumDims())`
/// on BOTH producers (`SNTransferLowering.cpp:1250`, and [`access_order`]), so the identity over the
/// view's rank is the whole answer and a field for it would be a second one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Access {
    /// The set the view's own shape implies — [`access_set`] over its rank and the vector's lanes.
    OfView,
    /// The set its producer computed.
    Stated(IntegerSet),
}

impl Access {
    /// The set this access prints, resolving [`Access::OfView`] against the view and the lane count.
    #[must_use]
    pub fn set(&self, view_ty: &MemRef, lanes: u64) -> IntegerSet {
        match self {
            Access::OfView => access_set(view_ty, lanes),
            Access::Stated(set) => set.clone(),
        }
    }
}

/// ONE MASK PATTERN — `(unmasked = N, masked = M)`, repeated to fill the slice it is applied to.
///
/// ⛔ THE SUM MUST DIVIDE THE SLICE. *"The sum of num_unmasked_elements and num_masked elements at a
/// given index must evenly divide into the number of elements in a slice"* (`Agen.td:1085-1086`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskPattern {
    /// `num_unmasked_elements[i]` — consecutive elements the mask leaves alone.
    pub unmasked: Elements,
    /// `num_masked_elements[i]` — consecutive elements it masks.
    pub masked: Elements,
}

/// WHICH MASK — `A` IS THE FIRST ONE. *"Lettering starts with A and continues sequentially for each
/// mask thereafter"* (`Agen.td:1064-1065`), so this is an index into
/// [`Op::SetTransferMaskState::masks`] and the letter is derived from it, never stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MaskId(pub u8);

impl MaskId {
    /// `A`, `B`, … — the spelling `slice_mask_map` and the `maskX` attribute names share.
    #[must_use]
    pub const fn letter(self) -> char {
        b'A'.saturating_add(self.0) as char
    }
}

/// WHAT ONE SLICE OF `slice_mask_map` CARRIES — `(0)`, `(1)`, `(A)` or `(A|B)`.
///
/// ⛔⛔ A STRING IS WHAT THE REFERENCE CARRIES AND IT IS NOT WHAT THIS ISLAND MAY CARRY.
/// `slice_mask_map = "(A)(A)(A)(A)(A)(A|B)(1)(1)"` is a closed grammar with four productions
/// (`Agen.td:1058-1072`); as an `enum` a slice cannot spell a fifth, and the slice COUNT is
/// `slices.len()` rather than a `num_slices` attribute that could disagree with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceMask {
    /// `(0)` — no masking.
    Unmasked,
    /// `(1)` — fully masked.
    Full,
    /// `(A)` — masked by one pattern.
    By(MaskId),
    /// `(A|B)` — masked by the OR of two. The only combination the reference supports.
    Or(MaskId, MaskId),
}

impl SliceMask {
    /// This slice as `slice_mask_map` spells it, parentheses included.
    #[must_use]
    pub fn spelling(self) -> String {
        match self {
            Self::Unmasked => "(0)".to_owned(),
            Self::Full => "(1)".to_owned(),
            Self::By(mask) => format!("({})", mask.letter()),
            Self::Or(lhs, rhs) => format!("({}|{})", lhs.letter(), rhs.letter()),
        }
    }
}

/// ONE `agen` OPERATION.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// `agen.vector_load %view[..] {load_order, load_set} : memref<..>, vector<..>`.
    ///
    /// ⭐⭐ THIS IS THE FORM THE BACKEND'S OWN PRODUCER EMITS. The dataflow scheduler's DataflowIR
    /// contains ONLY `agen.vector_load`/`agen.vector_store` — zero affine ones — and `dbo-opt` is
    /// what consumes that. The `affine` pair below appears in `dcc/test/PT/*.mlir`, which is run
    /// through `dcc-opt --kEmitProgIR`: a different tool entering at a different stage.
    ///
    /// ⛔ WHICH IS WHY EMITTING THE AFFINE FORM CRASHED THE PIPELINE. `VectorChainToSentientPESFP`
    /// has a working `VectorStoreOpLowering` for `agen::VectorStoreOp` and a `StoreOpLowering` for
    /// `vector::StoreOp` whose `reuse_info_.getId(..).value()` is unguarded
    /// (`VectorChainToSentientPESFP.cpp:123`) — undertested, because the scheduler never produces a
    /// `vector.store` for it to see.
    VectorLoad {
        /// The vector it binds.
        result: Val,
        /// The view read.
        view: Val,
        /// The indices.
        indices: Vec<Index>,
        /// `$multicast_info` — *"when present, carries routing information for transfers"*
        /// (`Agen.td:111`, `:129`), printed `multicast_info = %g`
        /// (`dcc/test/Conversion/AgenToSentient/mem2core-multicast.mlir:86`).
        ///
        /// ⛔ `constructLoadAndStoreStmt` READS IT (`Helper.cpp:2290-2292`) and hands it to the
        /// `sentient.load_and_store`, so with no field here a multicast load lowered to a transfer
        /// with a single destination.
        multicast_info: Option<Val>,
        /// `dbgName` — 101 of the corpus's loads carry one (`PT/fp8-bmm.mlir:1080`).
        ///
        /// ⛔ `constructLoadAndExtractScalarOp` COPIES IT onto the `sentient.load_and_extract_scalar`
        /// (`Helper.cpp:2444`, `lx_indirect_loads_stores_composite.mlir:75`).
        dbg_name: Option<String>,
        /// `$load_set` — which elements this load touches. See [`Access`] for why it is not always
        /// derivable from the view.
        access: Access,
        /// The view's type. Its INNERMOST extent is the lane count.
        view_ty: MemRef,
        /// The vector's type.
        ty: Vector,
    },

    /// `agen.composite_load_and_store src:%s[..] dst:%d[..] time_symbols(..), load_iv(%v:vector<..>)
    /// {..} { .. } : memref<..>, memref<..>`.
    ///
    /// ⭐⭐ THIS IS HOW A WEIGHT LEAVES THE HBM. A view is only an address; nothing crosses a
    /// datapath until a transfer says so. The device declares the route explicitly —
    /// `datapath %dram to %l3lu` then `datapath #L3LU_LX %l3lu to %lx`
    /// (`spyre_dd2_basic.mlir:82-83`) — and this op is what runs it. Emitting the compute against
    /// an LX view with no transfer into it is a program that reads memory nothing ever filled.
    ///
    /// ⛔⛔ AT MOST ONE HARDWARE VECTOR PER TIME STEP. "An AGEN composite transfer moves at most one
    /// hardware vector per time step, so a transfer wider than that has to walk the remaining
    /// elements over AGEN time dimensions instead of widening `load_iv`"
    /// (`DataTransferLowering.cpp:306-310`). The walk is what [`time_set`](Self::CompositeLoadAndStore::time_set)
    /// and the two `*_time_addr_map`s describe; see
    /// [`crate::bridges::subtile_to_dataflow_ir::transfer`], which computes them.
    ///
    /// ⛔ THE REGION IS ENTERED ONCE PER TIME STEP and its block argument carries the vector loaded
    /// at that step. A plain memory-to-memory move yields immediately; a transfer that also sends
    /// the value onward puts that in the body.
    /// ⛔ BOXED, because it carries four affine maps, four integer sets and two subscripts, and an
    /// enum is as large as its largest variant. Every other op in this IR is a handful of words.
    CompositeLoadAndStore(Box<CompositeTransfer>),

    /// `agen.composite_load %view[..] time_symbols(..)(%v:vector<..>) {..} { .. } : memref<..>` —
    /// a sequence of vector loads, one per time step, whose loaded vector reaches its consumers
    /// inside the op's own region rather than a second view (`Agen.td:423-486`).
    ///
    /// ⛔ IT IS HERE BECAUSE ENTRY 326 CANNOT BE SPELLED WITHOUT IT. `TPMVCompositeLoad::initialize_time`
    /// `dyn_cast`s exactly this op (`TransformPagedMemViewImpl.cpp:1096-1098`), so declaring the input
    /// is what AGENT-BRIEF.md:87 asks for rather than calling the unit unnecessary.
    /// ⚠️ NOTHING THIS CRATE EMITS PRODUCES ONE — the vendor's keys do
    /// (`paged_mem_view_loads.mlir:331`), and
    /// [`TpmvCompositeLoad`](crate::bridges::dataflow_ir_to_sentient::tf_transform_paged_mem_view_impl::TpmvCompositeLoad)
    /// is its only reader. ⛔ BOXED, as its two-sided twin is.
    CompositeLoad(Box<CompositeAccess>),

    /// `agen.composite_store %view[..] [input_vector=%v] time_symbols(..) {..} [{ .. }] :
    /// memref<..>[ , vector<..>]` — a sequence of vector stores, one per time step
    /// (`Agen.td:486-554`).
    ///
    /// ⛔ IT IS HERE BECAUSE ENTRY 330 CANNOT BE SPELLED WITHOUT IT. `lowerCompositeStoreOp`
    /// `dyn_cast`s exactly this op (`Helper.cpp:3133`) and `constructTimeLoopsAndVectorOperations`
    /// reads its memref's element type to pick the store statement (`:1870-1877`), so declaring the
    /// input is what AGENT-BRIEF.md:87 asks for rather than calling the unit unnecessary.
    /// ⛔ BOXED, as the other composites are.
    CompositeStore(Box<CompositeStoreAccess>),

    /// `agen.composite_indirect_load indirect:%i[..] direct:%d[..] time_symbols(..),
    /// load_iv(%v:vector<..>) {..} { .. } : memref<..>, memref<..>` (`Agen.td:677`).
    ///
    /// ⭐⭐ THE COMPOSITE GATHER: the address comes out of the INDIRECT view — the virtual IBR an
    /// `agen.vector_store` wrote the index into — and the data out of the DIRECT one, once per time
    /// step.
    ///
    /// ⛔ IT IS HERE BECAUSE ENTRY 332 CANNOT BE SPELLED WITHOUT IT, and entry 332 is the LXLU-only
    /// lowering that pairs this op with the `sentient.load_and_extract_scalar` its `extract_idx`
    /// names (`Helper.cpp:3267-3311`). ⛔ BOXED, as the other composites are.
    CompositeIndirectLoad(Box<CompositeIndirectLoadAccess>),

    /// `agen.composite_indirect_store indirect:%i[..] direct:%d[..] time_symbols(..) {..} { .. } :
    /// memref<..>, memref<..>` (`Agen.td:758`).
    ///
    /// ⭐⭐ THE COMPOSITE SCATTER — the store twin of [`Op::CompositeIndirectLoad`], whose stored
    /// vector arrives on the region's `agen.yield` rather than on a `load_iv`
    /// (`lx_indirect_loads_stores_composite.mlir:241-247`).
    ///
    /// ⛔ IT IS HERE BECAUSE ENTRY 333 CANNOT BE SPELLED WITHOUT IT — the LXSU-only lowering, which
    /// pairs it with a `sentient.receive_and_extract_scalar` (`Helper.cpp:3313-3356`).
    /// ⛔ BOXED, as the other composites are.
    CompositeIndirectStore(Box<CompositeIndirectStoreAccess>),

    /// `agen.composite_indirect_load_and_store [indirect_src:%is[..]] direct_src:%ds[..]
    /// [indirect_dst:%id[..]] direct_dst:%dd[..] time_symbols(..), load_iv(%v:vector<..>) {..} { .. } :
    /// memref<..>, ..` (`Agen.td:559`).
    ///
    /// ⭐⭐ THE GATHER AND THE SCATTER AS ONE TRANSFER. *"composite_indirect_load_and_store
    /// operations are used for [indirectly] loading from one memory address and [indirectly] storing
    /// into another as part of each time step. All of the indirect access functions are optional."*
    /// The indirect view holds ADDRESSES; the direct one holds the data.
    ///
    /// ⛔ AT MOST ONE INDIRECT SIDE IS SPLITTABLE. `e322_transformCompIndLoadAndStore`'s
    /// `DT_CHECK` refuses a candidate whose own side carries the indirect
    /// (`MutableAddrSplitting.cpp:581-585`): *"The immutable address must be zero for the side of the
    /// transfer involving the indirect."*
    /// ⛔ BOXED for the same reason as its direct twin, and it carries two more maps than that one.
    CompositeIndirectLoadAndStore(Box<CompositeIndirectTransfer>),

    /// `agen.yield` — the terminator of a composite transfer's region.
    Yield,

    /// `agen.indirect_vector_load indirect:%iv[..] direct:%dv[..] {load_order, load_set} :
    /// memref<..>, memref<..>, vector<..>` (`Agen.td:838`).
    ///
    /// ⭐⭐ THE GATHER ITSELF — the address comes out of the INDIRECT view (the virtual IBR, written
    /// by the extract pattern) and the data out of the DIRECT one. `constructLoadAndExtractScalarOp`
    /// (entry 269) matches the indirect view's TWO users, this op and the `agen.vector_store` that
    /// fed it, and without this op in the island that pattern has no second user to find.
    ///
    /// ⛔ THE `load_set`/`load_order` ARE OVER THE **DIRECT** VIEW, as the store twin's are
    /// (`Agen.cpp:2198-2201` elides `indirect_map`/`direct_map` from the printed dict).
    IndirectVectorLoad {
        /// The vector it binds.
        result: Val,
        /// `$indirect_memref` — the view the address is READ from.
        indirect_view: Val,
        /// `$indirect_map_indices`.
        indirect_indices: Vec<Index>,
        /// The indirect view's type.
        indirect_view_ty: MemRef,
        /// `$direct_memref` — the view the DATA is read from, and the one every access record reads.
        direct_view: Val,
        /// `$direct_map_indices`.
        direct_indices: Vec<Index>,
        /// The direct view's type.
        direct_view_ty: MemRef,
        /// `$multicast_info` — optional (`Agen.td:862`), and nothing in this island fills it yet.
        multicast_info: Option<Val>,
        /// The vector's type.
        ty: Vector,
    },

    /// `agen.vector_store %value, %view[..] {store_order, store_set} : memref<..>, vector<..>`.
    ///
    /// See [`Op::VectorLoad`] for why this is the form the bridge emits.
    VectorStore {
        /// The vector stored.
        value: Val,
        /// The view written.
        view: Val,
        /// The indices.
        indices: Vec<Index>,
        /// `$dbgName` — optional, as the load's is, and `cloneWithNewAccessInfo` substitutes an EMPTY
        /// one for an absent one (`Agen.cpp:240-248`), which is why a rebuilt store prints
        /// `dbgName = ""` where the original printed nothing.
        dbg_name: Option<String>,
        /// `$store_set` — which elements this store touches. See [`Access`].
        access: Access,
        /// The view's type. Its INNERMOST extent is the lane count.
        view_ty: MemRef,
        /// The vector's type.
        ty: Vector,
    },
    /// `agen.indirect_vector_store %value, %indirect_view[..], %direct_view[..] {store_order,
    /// store_set} : memref<..>, memref<..>, vector<..>` — A SCATTER: the address written is the one
    /// the INDIRECT view holds, applied to the DIRECT view being written.
    ///
    /// ⛔⛔ TWO DEREFERENCED MEMREFS, AND THE REFERENCE SAYS SO ITSELF (`Agen.td:911-922`): that is
    /// why it cannot carry `AffineMapAccessInterface`. Every consumer that reads "the" view of an
    /// access reads the DIRECT one — `AccessDetailsAffine::initialize` takes `getDirectMemref()` and
    /// `getDirectMapIndices()` (`AccessDetails.cpp:332-342`).
    /// ⭐ THE INDIRECT VIEW IS A VIRTUAL IBR SHARED WITH AN `agen.vector_store` THAT WRITES THE INDEX
    /// INTO IT. That two-user pair is the extract pattern `constructReceiveAndExtractScalarOp`
    /// matches (`Helper.cpp:2471-2512`), and it is why this op is in the island: without it the
    /// pattern has no second user to find.
    IndirectVectorStore {
        /// `$value` — the vector stored.
        value: Val,
        /// `$indirect_memref` — the view the address is READ from.
        indirect_view: Val,
        /// `$indirect_map_indices`.
        indirect_indices: Vec<Index>,
        /// The indirect view's type.
        indirect_view_ty: MemRef,
        /// `$direct_memref` — the view WRITTEN, and the one every access record reads.
        direct_view: Val,
        /// `$direct_map_indices`.
        direct_indices: Vec<Index>,
        /// The direct view's type.
        direct_view_ty: MemRef,
        /// `$multicast_info` — optional (`Agen.td:934`), and nothing in this island fills it yet.
        multicast_info: Option<Val>,
        /// The vector's type.
        ty: Vector,
    },

    /// `agen.symbolic_vector_load %view[indices:(..), strides:(..)] {load_order, load_set} :
    /// memref<..>, vector<..>`.
    ///
    /// ⭐⭐ THE ADDRESS IS A RUNTIME VALUE, WHICH IS THE WHOLE DIFFERENCE. A plain
    /// [`Op::VectorLoad`] carries an affine subscript the lowering folds into a constant offset; this
    /// one carries a subscript AND A STRIDE PER DIMENSION, both `Value`s, so the offset is
    /// accumulated at run time — `sentient.load_and_send mutable_addr(%acc)` with an `arith.addi` of
    /// the stride per iteration (`dcc/test/Conversion/AgenToSentient/symbolic_vector_load_store.mlir:47-49`).
    ///
    /// ⛔ `num_indices` AND `num_strides` ARE NOT FIELDS. The reference packs subscript, strides and
    /// the optional multicast handle into one `Variadic<Index>:$operands1` and stores the two counts
    /// as `I32Attr`s so `getIndices()`/`getStrides()` can slice it
    /// (`Agen.td:1128-1130`, `:1155-1172`). Three typed fields say the same thing and cannot
    /// disagree with the counts.
    SymbolicVectorLoad {
        /// The vector it binds.
        result: Val,
        /// The view read.
        view: Val,
        /// `getIndices()` — `operands1.slice(0, num_indices)`.
        indices: Vec<Index>,
        /// `getStrides()` — `operands1.slice(num_indices, num_strides)`, one per subscript, and
        /// values rather than constants: that is what makes this access symbolic.
        strides: Vec<Val>,
        /// `getMulticastInfo()` — the one trailing operand, when there is one
        /// (`Agen.td:1163-1172`: one or none, and anything else is `llvm_unreachable`).
        multicast: Option<Val>,
        /// The view's type.
        view_ty: MemRef,
        /// The vector's type.
        ty: Vector,
    },

    /// `agen.symbolic_vector_store %value, %view[indices:(..), strides:(..)] {store_order,
    /// store_set} : memref<..>, vector<..>`
    /// (`dcc/test/Conversion/AgenToSentient/symbolic_vector_load_store.mlir:302`).
    ///
    /// See [`Op::SymbolicVectorLoad`] for why the strides are operands. ⛔ AND IT HAS NO MULTICAST
    /// HANDLE — `Agen.td:1194-1225` declares `getIndices`/`getStrides` and no `getMulticastInfo`,
    /// because a store has one destination.
    SymbolicVectorStore {
        /// The vector stored — `$value_to_store`, operand 0.
        value: Val,
        /// The view written — `$memref`, operand 1.
        view: Val,
        /// `getIndices()`.
        indices: Vec<Index>,
        /// `getStrides()`.
        strides: Vec<Val>,
        /// The view's type.
        view_ty: MemRef,
        /// The vector's type.
        ty: Vector,
    },

    /// `agen.composite_memory_interleave {granularity = 64 : i32} { .. }`
    /// (`Agen.td:982-1039`, `dcc/test/Conversion/AgenToSentient/comp_mem_interleave.mlir:309`).
    ///
    /// ⭐ IT SPLITS THE BURST OF EVERY TRANSFER IN ITS REGION so the region's transfers alternate:
    /// A@64, B@64, A@16, B@16 out of two A@80/B@80. `e271_lowerCompositeMemoryInterleaveOp` is the
    /// lowering and `e384_runOnOperation`'s fourth step is the sweep that finds these.
    ///
    /// ⛔ `granularity` ABSENT IS THE HARDWARE MAXIMUM, NOT ZERO, and a granularity above the
    /// region's own burst is read DOWN to that burst — *"because some of the upstream components may
    /// insert CompositeMemoryInterleaveOps without knowing the burst of the contained operations"*.
    CompositeMemoryInterleave {
        /// `granularity` — how much of each transfer's burst goes before the next transfer's.
        granularity: Option<Elements>,
        /// The composite transfers it interleaves. Its terminator is implicit
        /// (`ImplicitAgenTerminator`), so no [`Op::Yield`] is written here.
        body: Vec<super::Op>,
    },

    /// `%m = agen.set_transfer_mask_state mask_value(%c0) {num_slices = 8 : i32, slice_mask_map =
    /// "(A)(A)(A)(A)(A)(A|B)(1)(1)", maskA = "(unmasked = 8 : i32, masked = 8 : i32)", ..} : index,
    /// vector<128xi8>` (`Agen.td:1041-1116`,
    /// `dcc/test/Conversion/AgenToSentient/set_transfer_mask_state.mlir:24`).
    ///
    /// ⭐ IT IS THE UNIT'S MASK STATE, NOT A VALUE A COMPUTE READS: *"collects information from
    /// various mask patterns and forms one final mask that will be applied to transfers for the given
    /// unit"*. `e218_lowerSetTransferMaskStateOp` turns it into `sentient.samv`.
    SetTransferMaskState {
        /// The vector it binds.
        result: Val,
        /// `mask_value` — the value written over the masked elements. An `index` operand.
        mask_value: Val,
        /// `slice_mask_map`, one entry per slice; `num_slices` IS this length.
        slices: Vec<SliceMask>,
        /// The `maskA`, `maskB`, … patterns in order — the two optional `i32` arrays zipped, since
        /// *"the number of consecutive unmasked elements for each mask is provided in an ordered
        /// list"* and a `MaskId` indexes this.
        masks: Vec<MaskPattern>,
        /// The result's type.
        ty: Vector,
    },
}

/// ONE `agen` OP AS TEXT. The caller has already indented the opening line.
pub(crate) fn emit(out: &mut String, op: &Op, depth: usize) {
    match op {
        Op::VectorLoad {
            result,
            view,
            indices,
            multicast_info,
            dbg_name,
            access,
            view_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = agen.vector_load {}[{}]{} {{{}load_order = {}, load_set = {}}} : {}, {}",
                print::val(*result),
                print::val(*view),
                print::index_list(indices),
                multicast_info.map_or(String::new(), |group| format!(
                    " multicast_info = {}",
                    print::val(group)
                )),
                // `dbgName` sorts ahead of `load_order` (`PT/fp8-bmm.mlir:1080`).
                dbg_name
                    .as_ref()
                    .map_or(String::new(), |name| format!("dbgName = \"{name}\", ")),
                print::affine_map(&access_order(view_ty.shape.len())),
                print::integer_set(&access.set(view_ty, ty.len)),
                print::memref(view_ty),
                print::vector(*ty)
            );
        }
        Op::CompositeLoadAndStore(transfer) => {
            let CompositeTransfer {
                src,
                src_indices,
                src_ty,
                dst,
                dst_indices,
                dst_ty,
                load_iv,
                load_iv_ty,
                load_set,
                load_order,
                store_set,
                store_order,
                time_symbols,
                time_set,
                time_order,
                load_time_addr_map,
                store_time_addr_map,
                dir,
                multicast_info,
                dbg_name,
                body,
            } = transfer.as_ref();
            // Three lines, as the scheduler writes it: the two accesses, then the induction
            // variable, then the attributes — alphabetical, which puts the load trio first.
            let _ = writeln!(
                out,
                "agen.composite_load_and_store src:{}[{}] dst:{}[{}]",
                print::val(*src),
                print::index_list(src_indices),
                print::val(*dst),
                print::index_list(dst_indices),
            );
            print::indent(out, depth);
            let _ = writeln!(
                out,
                " time_symbols({}), load_iv({}:{}){}",
                time_symbols
                    .iter()
                    .map(|sym| print::val(*sym))
                    .collect::<Vec<String>>()
                    .join(", "),
                print::val(*load_iv),
                print::vector(*load_iv_ty),
                // `p << ", multicast_info = " << multicast_info` (`Agen.cpp:358-359`), which is an
                // OPERAND and so prints outside the attribute dictionary
                // (`mem2core-composite-multicast.mlir:83`).
                multicast_info.map_or(String::new(), |group| format!(
                    ", multicast_info = {}",
                    print::val(group)
                ))
            );
            print::indent(out, depth);
            let _ = writeln!(
                out,
                " {{{}{}load_order = {}, load_set = {}, load_time_addr_map = {}, store_order = {}, \
                 store_set = {}, store_time_addr_map = {}, time_order = {}, time_set = {}}}",
                // `dbgName` sorts ahead of `dir` (`l3-burst-calc.mlir:759`).
                dbg_name
                    .as_ref()
                    .map_or(String::new(), |name| format!("dbgName = \"{name}\", ")),
                // ⭐ `dir` IS AN ATTRIBUTE AND SORTS ALPHABETICALLY, which puts it ahead of every
                // other one this op carries
                // (`paged_mem_view_load_and_store.mlir:101`).
                dir.map_or(String::new(), |dir| format!(
                    "dir = #agen<direction {}>, ",
                    dir.spelling()
                )),
                print::affine_map(load_order),
                print::integer_set(load_set),
                print::affine_map(load_time_addr_map),
                print::affine_map(store_order),
                print::integer_set(store_set),
                print::affine_map(store_time_addr_map),
                print::affine_map(time_order),
                print::integer_set(time_set),
            );
            print::indent(out, depth);
            out.push_str("{\n");
            for inner in body {
                print::emit(out, inner, depth + 1);
            }
            print::indent(out, depth);
            let _ = writeln!(
                out,
                "}} : {}, {}",
                print::memref(src_ty),
                print::memref(dst_ty)
            );
        }
        Op::CompositeLoad(access) => {
            let CompositeAccess {
                view,
                indices,
                view_ty,
                load_iv,
                load_iv_ty,
                load_set,
                load_order,
                time_symbols,
                time_set,
                time_order,
                time_addr_map,
                dbg_name,
                body,
            } = access.as_ref();
            // Three lines, as the vendor's key writes it (`paged_mem_view_loads.mlir:331-341`): the
            // access, then the time symbols with the induction variable, then the attributes.
            let _ = writeln!(
                out,
                "agen.composite_load {}[{}]",
                print::val(*view),
                print::index_list(indices),
            );
            print::indent(out, depth);
            let _ = writeln!(
                out,
                " time_symbols({})({}:{})",
                time_symbols
                    .iter()
                    .map(|sym| print::val(*sym))
                    .collect::<Vec<String>>()
                    .join(", "),
                print::val(*load_iv),
                print::vector(*load_iv_ty),
            );
            print::indent(out, depth);
            let _ = writeln!(
                out,
                " {{{}load_order = {}, load_set = {}, time_addr_map = {}, time_order = {}, \
                 time_set = {}}}",
                dbg_name
                    .as_ref()
                    .map_or(String::new(), |name| format!("dbgName = \"{name}\", ")),
                print::affine_map(load_order),
                print::integer_set(load_set),
                print::affine_map(time_addr_map),
                print::affine_map(time_order),
                print::integer_set(time_set),
            );
            print::indent(out, depth);
            out.push_str("{\n");
            for inner in body {
                print::emit(out, inner, depth + 1);
            }
            print::indent(out, depth);
            let _ = writeln!(out, "}} : {}", print::memref(view_ty));
        }
        Op::CompositeStore(access) => {
            let CompositeStoreAccess {
                view,
                indices,
                view_ty,
                source,
                time_symbols,
                time_set,
                time_order,
                time_addr_map,
                dbg_name,
            } = access.as_ref();
            let symbols = time_symbols
                .iter()
                .map(|sym| print::val(*sym))
                .collect::<Vec<String>>()
                .join(", ");
            let _ = writeln!(
                out,
                "agen.composite_store {}[{}]",
                print::val(*view),
                print::index_list(indices),
            );
            print::indent(out, depth);
            // `if (input_vector) p << "input_vector=" << v` then ` time_symbols(..)`
            // (`Agen.cpp:812-817`) — no leading space on the operand, one before the symbols.
            let _ = writeln!(
                out,
                "{} time_symbols({symbols})",
                match source {
                    CompositeStoreSource::InputVector { value, .. } => {
                        format!("input_vector={}", print::val(*value))
                    }
                    CompositeStoreSource::Region { .. } => String::new(),
                }
            );
            print::indent(out, depth);
            // ⛔ THE COALESCE FORM ELIDES `store_order` AND `store_set` (`Agen.cpp:823-827`); the
            // region form prints them, alphabetically ahead of the time trio.
            let _ = writeln!(
                out,
                " {{{}{}time_addr_map = {}, time_order = {}, time_set = {}}}",
                dbg_name
                    .as_ref()
                    .map_or(String::new(), |name| format!("dbgName = \"{name}\", ")),
                match source {
                    CompositeStoreSource::InputVector { .. } => String::new(),
                    CompositeStoreSource::Region {
                        store_set,
                        store_order,
                        ..
                    } => format!(
                        "store_order = {}, store_set = {}, ",
                        print::affine_map(store_order),
                        print::integer_set(store_set)
                    ),
                },
                print::affine_map(time_addr_map),
                print::affine_map(time_order),
                print::integer_set(time_set),
            );
            match source {
                CompositeStoreSource::InputVector { ty, .. } => {
                    print::indent(out, depth);
                    // `p << " : " << memref` then `p << " , " << vector` — ⛔ SPACES BOTH SIDES OF
                    // THE COMMA on this one (`Agen.cpp:832-834`).
                    let _ = writeln!(
                        out,
                        " : {} , {}",
                        print::memref(view_ty),
                        print::vector(*ty)
                    );
                }
                CompositeStoreSource::Region {
                    stored,
                    stored_ty,
                    body,
                    ..
                } => {
                    print::indent(out, depth);
                    out.push_str("{\n");
                    for inner in body {
                        print::emit(out, inner, depth + 1);
                    }
                    // The terminator this island holds as a field — see
                    // [`CompositeStoreSource::Region::stored`].
                    print::indent(out, depth + 1);
                    let _ = writeln!(
                        out,
                        "agen.yield {} : {}",
                        print::val(*stored),
                        print::vector(*stored_ty)
                    );
                    print::indent(out, depth);
                    let _ = writeln!(out, "}} : {}", print::memref(view_ty));
                }
            }
        }
        Op::CompositeIndirectLoad(access) => {
            let CompositeIndirectLoadAccess {
                indirect,
                direct,
                direct_indices,
                direct_ty,
                multicast_info,
                load_iv,
                load_iv_ty,
                load_set,
                load_order,
                time_symbols,
                time_set,
                time_order,
                time_addr_map,
                dbg_name,
                body,
            } = access.as_ref();
            // One line through `load_iv`, then the attributes, then the region
            // (`Agen.cpp:1569-1594`, `lx_indirect_loads_stores_composite.mlir:205-210`).
            let _ = writeln!(
                out,
                "agen.composite_indirect_load indirect:{}[{}] direct:{}[{}]{} time_symbols({}), \
                 load_iv({}:{})",
                print::val(indirect.view),
                print::index_list(&indirect.indices),
                print::val(*direct),
                print::index_list(direct_indices),
                multicast_info.map_or(String::new(), |group| format!(
                    " multicast_info = {}",
                    print::val(group)
                )),
                time_symbols
                    .iter()
                    .map(|sym| print::val(*sym))
                    .collect::<Vec<String>>()
                    .join(", "),
                print::val(*load_iv),
                print::vector(*load_iv_ty),
            );
            print::indent(out, depth);
            let _ = writeln!(
                out,
                " {{{}load_order = {}, load_set = {}, time_addr_map = {}, time_order = {}, \
                 time_set = {}}}",
                dbg_name
                    .as_ref()
                    .map_or(String::new(), |name| format!("dbgName = \"{name}\", ")),
                print::affine_map(load_order),
                print::integer_set(load_set),
                print::affine_map(time_addr_map),
                print::affine_map(time_order),
                print::integer_set(time_set),
            );
            print::indent(out, depth);
            out.push_str("{\n");
            for inner in body {
                print::emit(out, inner, depth + 1);
            }
            print::indent(out, depth);
            let _ = writeln!(
                out,
                "}} : {}, {}",
                print::memref(&indirect.ty),
                print::memref(direct_ty)
            );
        }
        Op::CompositeIndirectStore(access) => {
            let CompositeIndirectStoreAccess {
                indirect,
                direct,
                direct_indices,
                direct_ty,
                multicast_info,
                store_set,
                store_order,
                stored,
                stored_ty,
                time_symbols,
                time_set,
                time_order,
                time_addr_map,
                dbg_name,
                body,
            } = access.as_ref();
            // ⛔ THE NEWLINE COMES BEFORE THE MULTICAST HANDLE HERE, where the load prints it on the
            // access line (`Agen.cpp:1803-1808` against `:1581`).
            let _ = writeln!(
                out,
                "agen.composite_indirect_store indirect:{}[{}] direct:{}[{}]",
                print::val(indirect.view),
                print::index_list(&indirect.indices),
                print::val(*direct),
                print::index_list(direct_indices),
            );
            print::indent(out, depth);
            let _ = writeln!(
                out,
                "{} time_symbols({})",
                multicast_info.map_or(String::new(), |group| format!(
                    " multicast_info = {}",
                    print::val(group)
                )),
                time_symbols
                    .iter()
                    .map(|sym| print::val(*sym))
                    .collect::<Vec<String>>()
                    .join(", "),
            );
            print::indent(out, depth);
            let _ = writeln!(
                out,
                " {{{}store_order = {}, store_set = {}, time_addr_map = {}, time_order = {}, \
                 time_set = {}}}",
                dbg_name
                    .as_ref()
                    .map_or(String::new(), |name| format!("dbgName = \"{name}\", ")),
                print::affine_map(store_order),
                print::integer_set(store_set),
                print::affine_map(time_addr_map),
                print::affine_map(time_order),
                print::integer_set(time_set),
            );
            print::indent(out, depth);
            out.push_str("{\n");
            for inner in body {
                print::emit(out, inner, depth + 1);
            }
            print::indent(out, depth + 1);
            let _ = writeln!(
                out,
                "agen.yield {} : {}",
                print::val(*stored),
                print::vector(*stored_ty)
            );
            print::indent(out, depth);
            let _ = writeln!(
                out,
                "}} : {}, {}",
                print::memref(&indirect.ty),
                print::memref(direct_ty)
            );
        }
        Op::CompositeIndirectLoadAndStore(transfer) => {
            let CompositeIndirectTransfer {
                indirect_src,
                direct_src,
                direct_src_indices,
                direct_src_ty,
                indirect_dst,
                direct_dst,
                direct_dst_indices,
                direct_dst_ty,
                load_iv,
                load_iv_ty,
                load_set,
                load_order,
                store_set,
                store_order,
                time_symbols,
                time_set,
                time_order,
                load_indirect_time_addr_map,
                load_direct_time_addr_map,
                store_indirect_time_addr_map,
                store_direct_time_addr_map,
                multicast_info,
                dbg_name,
                body,
            } = transfer.as_ref();
            // `if (hasIndirectSrc()) p << " indirect_src:" ..` then the direct source, then the same
            // pair for the destination (`Agen.cpp:1185-1211`) — an absent indirect prints nothing.
            let access = |label: &str, view: Val, indices: &[Index]| {
                format!(
                    "{label}:{}[{}]",
                    print::val(view),
                    print::index_list(indices)
                )
            };
            let indirect = |label: &str, side: &Option<IndirectAccess>| {
                side.as_ref().map_or(String::new(), |side| {
                    format!("{} ", access(label, side.view, &side.indices))
                })
            };
            let _ = writeln!(
                out,
                "agen.composite_indirect_load_and_store {}{} {}{}",
                indirect("indirect_src", indirect_src),
                access("direct_src", *direct_src, direct_src_indices),
                indirect("indirect_dst", indirect_dst),
                access("direct_dst", *direct_dst, direct_dst_indices),
            );
            print::indent(out, depth);
            let _ = writeln!(
                out,
                " time_symbols({}), load_iv({}:{}){}",
                time_symbols
                    .iter()
                    .map(|sym| print::val(*sym))
                    .collect::<Vec<String>>()
                    .join(", "),
                print::val(*load_iv),
                print::vector(*load_iv_ty),
                multicast_info.map_or(String::new(), |group| format!(
                    ", multicast_info = {}",
                    print::val(group)
                ))
            );
            print::indent(out, depth);
            // ⛔ ALPHABETICAL, AND `load_direct_time_addr_map` SORTS AHEAD OF THE INDIRECT ONE
            // (`mutable_addr_splitting_one_dim.mlir:160`). An absent optional map is elided.
            let optional = |name: &str, map: &Option<AffineMap>| {
                map.as_ref().map_or(String::new(), |map| {
                    format!("{name} = {}, ", print::affine_map(map))
                })
            };
            let _ = writeln!(
                out,
                " {{{}{}{}load_order = {}, load_set = {}, {}{}store_order = {}, store_set = {}, \
                 time_order = {}, time_set = {}}}",
                dbg_name
                    .as_ref()
                    .map_or(String::new(), |name| format!("dbgName = \"{name}\", ")),
                format_args!(
                    "load_direct_time_addr_map = {}, ",
                    print::affine_map(load_direct_time_addr_map)
                ),
                optional("load_indirect_time_addr_map", load_indirect_time_addr_map),
                print::affine_map(load_order),
                print::integer_set(load_set),
                format_args!(
                    "store_direct_time_addr_map = {}, ",
                    print::affine_map(store_direct_time_addr_map)
                ),
                optional("store_indirect_time_addr_map", store_indirect_time_addr_map),
                print::affine_map(store_order),
                print::integer_set(store_set),
                print::affine_map(time_order),
                print::integer_set(time_set),
            );
            print::indent(out, depth);
            out.push_str("{\n");
            for inner in body {
                print::emit(out, inner, depth + 1);
            }
            print::indent(out, depth);
            // `p << " : "`, indirect type first on each side, and only where that side has one.
            let _ = writeln!(
                out,
                "}} : {}{}{}{}",
                indirect_src.as_ref().map_or(String::new(), |side| format!(
                    "{}, ",
                    print::memref(&side.ty)
                )),
                print::memref(direct_src_ty),
                indirect_dst.as_ref().map_or(String::new(), |side| format!(
                    ", {}",
                    print::memref(&side.ty)
                )),
                format_args!(", {}", print::memref(direct_dst_ty)),
            );
        }
        Op::Yield => {
            out.push_str("agen.yield\n");
        }
        Op::VectorStore {
            value,
            view,
            indices,
            dbg_name,
            access,
            view_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "agen.vector_store {}, {}[{}] {{{}store_order = {}, store_set = {}}} : {}, {}",
                print::val(*value),
                print::val(*view),
                print::index_list(indices),
                // `dbgName` sorts ahead of `store_order`, as it does ahead of `load_order`.
                dbg_name
                    .as_ref()
                    .map_or(String::new(), |name| format!("dbgName = \"{name}\", ")),
                print::affine_map(&access_order(view_ty.shape.len())),
                print::integer_set(&access.set(view_ty, ty.len)),
                print::memref(view_ty),
                print::vector(*ty)
            );
        }
        // ⭐ SAME SHAPE AS THE STORE BELOW, and the result's vector type prints LAST of the three
        // (`dcc/test/Conversion/AgenToSentient/lx_indirect_loads_stores.mlir:344-346`).
        Op::IndirectVectorLoad {
            result,
            indirect_view,
            indirect_indices,
            indirect_view_ty,
            direct_view,
            direct_indices,
            direct_view_ty,
            multicast_info,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = agen.indirect_vector_load indirect:{}[{}] direct:{}[{}]{} {{load_order = {}, \
                 load_set = {}}} : {}, {}, {}",
                print::val(*result),
                print::val(*indirect_view),
                print::index_list(indirect_indices),
                print::val(*direct_view),
                print::index_list(direct_indices),
                multicast_info.map_or(String::new(), |group| format!(
                    " multicast_info = {}",
                    print::val(group)
                )),
                print::affine_map(&access_order(direct_view_ty.shape.len())),
                print::integer_set(&access_set(direct_view_ty, ty.len)),
                print::memref(indirect_view_ty),
                print::memref(direct_view_ty),
                print::vector(*ty)
            );
        }
        // ⭐ THE ORDER AND SET ARE OVER THE **DIRECT** VIEW, and `indirect_map`/`direct_map` are
        // elided from the dict (`Agen.cpp:2198-2201`), which is why only two attributes print.
        Op::IndirectVectorStore {
            value,
            indirect_view,
            indirect_indices,
            indirect_view_ty,
            direct_view,
            direct_indices,
            direct_view_ty,
            multicast_info,
            ty,
        } => {
            let _ = writeln!(
                out,
                "agen.indirect_vector_store {} indirect:{}[{}] direct:{}[{}]{} {{store_order = {}, \
                 store_set = {}}} : {}, {}, {}",
                print::val(*value),
                print::val(*indirect_view),
                print::index_list(indirect_indices),
                print::val(*direct_view),
                print::index_list(direct_indices),
                multicast_info.map_or(String::new(), |group| format!(
                    " multicast_info = {}",
                    print::val(group)
                )),
                print::affine_map(&access_order(direct_view_ty.shape.len())),
                print::integer_set(&access_set(direct_view_ty, ty.len)),
                print::vector(*ty),
                print::memref(indirect_view_ty),
                print::memref(direct_view_ty)
            );
        }
        Op::SymbolicVectorLoad {
            result,
            view,
            indices,
            strides,
            multicast,
            view_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = agen.symbolic_vector_load {}[indices:({}), strides:({})]{} {{load_order = {}, \
                 load_set = {}}} : {}, {}",
                print::val(*result),
                print::val(*view),
                print::index_list(indices),
                val_list(strides),
                multicast.map_or(String::new(), |group| format!(
                    ", multicast_info({})",
                    print::val(group)
                )),
                print::affine_map(&access_order(view_ty.shape.len())),
                print::integer_set(&access_set(view_ty, ty.len)),
                print::memref(view_ty),
                print::vector(*ty)
            );
        }
        Op::SymbolicVectorStore {
            value,
            view,
            indices,
            strides,
            view_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "agen.symbolic_vector_store {}, {}[indices:({}), strides:({})] {{store_order = {}, \
                 store_set = {}}} : {}, {}",
                print::val(*value),
                print::val(*view),
                print::index_list(indices),
                val_list(strides),
                print::affine_map(&access_order(view_ty.shape.len())),
                print::integer_set(&access_set(view_ty, ty.len)),
                print::memref(view_ty),
                print::vector(*ty)
            );
        }
        // ⛔ THE ATTRIBUTE DICTIONARY IS PRINTED EVEN WHEN EMPTY — `agen.composite_memory_interleave
        // { } {` is how the vendor's own IR writes an absent granularity
        // (`comp_mem_interleave.mlir:309`), and the region follows it as a second brace pair.
        Op::CompositeMemoryInterleave { granularity, body } => {
            let _ = writeln!(
                out,
                "agen.composite_memory_interleave {} {{",
                granularity.map_or_else(
                    || "{}".to_owned(),
                    |grain| format!("{{granularity = {} : i32}}", grain.0)
                )
            );
            for inner in body {
                print::emit(out, inner, depth + 1);
            }
            print::indent(out, depth);
            out.push_str("}\n");
        }
        Op::SetTransferMaskState {
            result,
            mask_value,
            slices,
            masks,
            ty,
        } => {
            let map: String = slices.iter().map(|slice| slice.spelling()).collect();
            // ⭐ ONE `maskX` ATTRIBUTE PER PATTERN, NAMED BY ITS LETTER — the prefix the op's own
            // `getMaskPrefixAttrStrName()` returns (`Agen.td:1110`).
            let patterns: String = masks
                .iter()
                .enumerate()
                .map(|(i, pattern)| {
                    format!(
                        ", mask{} = \"(unmasked = {} : i32, masked = {} : i32)\"",
                        MaskId(u8::try_from(i).unwrap_or(u8::MAX)).letter(),
                        pattern.unmasked.0,
                        pattern.masked.0
                    )
                })
                .collect();
            let _ = writeln!(
                out,
                "{} = agen.set_transfer_mask_state mask_value({}) {{num_slices = {} : i32, \
                 slice_mask_map = \"{}\"{}}} : index, {}",
                print::val(*result),
                print::val(*mask_value),
                slices.len(),
                map,
                patterns,
                print::vector(*ty)
            );
        }
    }
}

/// `%a, %b` — a symbolic access's strides, which are plain values rather than affine subscripts.
fn val_list(vals: &[Val]) -> String {
    vals.iter()
        .map(|val| print::val(*val))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `affine_map<(d0, .., dn) -> (d0, .., dn)>` — the `load_order`/`store_order` of a vector access.
///
/// ⭐ `load_order`/`store_order` SAY WHICH AXIS MOVES FASTEST, and the scheduler writes the identity
/// for every access in its own output (`#map4`, `#map8`). Row-major order is what the view's own
/// `layout_map` already states, so ordering it again differently would be two answers to one
/// question.
///
/// ⛔ TYPED RATHER THAN PRINTED, BECAUSE THE LOWERING READS IT. `AccessDetailsAffine::initialize`
/// takes `load_op.getLoadOrder()` into `transfer_order_` (`AccessDetails.cpp:299`) and
/// `checkBasicConditions` asks it for an inverse permutation (`Helper.cpp:143`); a `String` answers
/// neither. [`emit`] prints what this returns, so the two cannot drift.
#[must_use]
pub fn access_order(rank: usize) -> AffineMap {
    AffineMap::identity(u32::try_from(rank).expect("a rank fits a u32"))
}

/// `affine_set<(d0, .., dn) : (d0 == 0, .., dn >= 0, -dn + LANES-1 >= 0)>` — which lanes are live.
///
/// ⭐⭐ THE INNERMOST AXIS IS THE LANE AXIS, and it is the only one that spans: every outer dim is
/// pinned to 0 and the last runs `0 .. lanes-1`. That is verbatim the shape the scheduler emits
/// (`#set1`, `#set3`), and it is the same "continuous prefix of live lanes" a
/// `create_affine_mask` carries — stated over the memref's dims instead of the vector's.
///
/// ⛔⛔ THE SPAN IS THE VECTOR'S, THE RANK IS THE MEMREF'S. This took `lanes` from the memref's
/// innermost dim, which is only the same number while every access reads a whole stick. A narrower
/// load — the single activation the PT's west port takes — is `vector<1xf16>` out of a
/// `memref<8x2x64xf16>`, and the backend refuses the mismatch outright: "Number of elements in
/// return type not matching with load_set/store_set elements".
///
/// ⛔ THE LANE AXIS SPANS EVEN FOR ONE LANE — `d{last} >= 0, -d{last} + 0 >= 0`, the pair, and not
/// the `d{last} == 0` [`IntegerSet::from_sizes`] would write. The two are the same set; this is the
/// spelling the printed form has always carried, so it is the spelling the typed form states.
#[must_use]
pub fn access_set(view_ty: &MemRef, lanes: u64) -> IntegerSet {
    let rank = view_ty.shape.len();
    let last = u32::try_from(rank.saturating_sub(1)).expect("a rank fits a u32");
    let mut constraints: Vec<Constraint> = (0..last)
        .map(|d| Constraint {
            expr: AffineExpr::dim(d),
            is_equality: true,
        })
        .collect();
    constraints.push(Constraint {
        expr: AffineExpr::dim(last),
        is_equality: false,
    });
    constraints.push(Constraint {
        expr: AffineExpr::dim(last).times(-1).plus(AffineExpr::Const(
            i64::try_from(lanes.saturating_sub(1)).unwrap_or(i64::MAX),
        )),
        is_equality: false,
    });
    IntegerSet {
        dims: u32::try_from(rank).expect("a rank fits a u32"),
        symbols: 0,
        constraints,
    }
}

#[cfg(test)]
mod tests {
    use crate::islands::dataflow_ir::dialects::agen::{CompositeTransfer, Op};
    use crate::islands::dataflow_ir::dialects::{self, Index, Val};
    use crate::islands::dataflow_ir::print::emit;

    /// ⭐⭐ THE HBM-TO-LX TRANSFER, AS THE REFERENCE WRITES IT.
    ///
    /// `/tmp/ktir_ref/export/debug/dfir.mlir:78-84` moves a `memref<12x64x64xf16>` view of the HBM
    /// into a `memref<2x2x1x1x64xf16>` view of the LX, one 64-lane vector per time step. This is
    /// the op whose ABSENCE was the defect: without it a program holds an LX address and nothing
    /// ever puts a weight behind it.
    ///
    /// ⛔ THE ATTRIBUTES ARE INLINED, NOT ALIASED. The reference writes `load_order = #map2` and
    /// declares `#map2` in a preamble; MLIR accepts either, and this printer has no alias table. So
    /// the comparison below is against the reference's attributes SPELLED OUT — same maps, same
    /// sets, same order — rather than against its `#map` names.
    #[test]
    fn prints_the_hbm_to_lx_transfer() {
        use crate::bridges::subtile_to_dataflow_ir::transfer::{Lanes, plan};
        use crate::islands::dataflow_ir::ty::{ElemType, MemRef, Vector};

        let planned = plan(&[1, 1, 64], &[1, 1, 1, 1, 64], 64, Lanes::F16)
            .expect("one 64-lane vector is the unsplit case");

        let op = dialects::Op::Agen(Op::CompositeLoadAndStore(Box::new(CompositeTransfer {
            src: Val(21),
            src_indices: vec![Index::Val(Val(1)), Index::Val(Val(4)), Index::Const(0)],
            src_ty: MemRef {
                shape: vec![12, 64, 64],
                elem: ElemType::F16,
            },
            dst: Val(27),
            dst_indices: vec![
                Index::Const(0),
                Index::Const(0),
                Index::Const(0),
                Index::Const(0),
                Index::Const(0),
            ],
            dst_ty: MemRef {
                shape: vec![2, 2, 1, 1, 64],
                elem: ElemType::F16,
            },
            load_iv: Val(9),
            load_iv_ty: Vector {
                len: planned.vector_lanes,
                elem: ElemType::F16,
            },
            load_set: planned.load_set,
            load_order: planned.load_order,
            store_set: planned.store_set,
            store_order: planned.store_order,
            time_symbols: Vec::new(),
            time_set: planned.time_set,
            time_order: planned.time_order,
            load_time_addr_map: planned.load_time_addr_map,
            store_time_addr_map: planned.store_time_addr_map,
            body: vec![dialects::Op::Agen(Op::Yield)],
            dir: None,
            multicast_info: None,
            dbg_name: None,
        })));

        let mut got = String::new();
        emit(&mut got, &op, 0);

        let want = "\
agen.composite_load_and_store src:%21[%1, %4, 0] dst:%27[0, 0, 0, 0, 0]
 time_symbols(), load_iv(%9:vector<64xf16>)
 {load_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, \
load_set = affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0)>, \
load_time_addr_map = affine_map<(d0) -> (0, 0, 0)>, \
store_order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, \
store_set = affine_set<(d0, d1, d2, d3, d4) : (d0 == 0, d1 == 0, d2 == 0, d3 == 0, d4 >= 0, \
-d4 + 63 >= 0)>, \
store_time_addr_map = affine_map<(d0) -> (0, 0, 0, 0, 0)>, \
time_order = affine_map<(d0) -> (d0)>, \
time_set = affine_set<(d0) : (d0 == 0)>}
{
  agen.yield
} : memref<12x64x64xf16>, memref<2x2x1x1x64xf16>
";

        for (at, (want_line, got_line)) in want.lines().zip(got.lines()).enumerate() {
            assert_eq!(want_line, got_line, "transfer line {at} diverges");
        }
        assert_eq!(want.lines().count(), got.lines().count());
    }

    /// ⭐⭐ THE VENDOR'S OWN SYMBOLIC PAIR, from the input module of
    /// `dcc/test/Conversion/AgenToSentient/symbolic_vector_load_store.mlir:224` and `:302`.
    ///
    /// ⛔ TWO INDICES OVER A ONE-DIMENSIONAL MEMREF, and the order/set are still over `d0` alone
    /// (`:184-185`): the subscript count is the loop nest's, the map's arity is the view's.
    /// ⛔ AND THE ATTRIBUTES ARE INLINED rather than aliased, for the reason the transfer test above
    /// records.
    #[test]
    fn prints_the_vendors_symbolic_access_pair() {
        use crate::islands::dataflow_ir::ty::{ElemType, MemRef, Vector};

        let view_ty = MemRef {
            shape: vec![128],
            elem: ElemType::Int(8),
        };
        let ty = Vector {
            len: 128,
            elem: ElemType::Int(8),
        };
        let indices = vec![Index::Val(Val(6)), Index::Val(Val(5))];
        let strides = vec![Val(64), Val(1)];

        let mut got = String::new();
        emit(
            &mut got,
            &dialects::Op::Agen(Op::SymbolicVectorLoad {
                result: Val(30),
                view: Val(20),
                indices: indices.clone(),
                strides: strides.clone(),
                multicast: None,
                view_ty: view_ty.clone(),
                ty,
            }),
            0,
        );
        emit(
            &mut got,
            &dialects::Op::Agen(Op::SymbolicVectorStore {
                value: Val(30),
                view: Val(22),
                indices,
                strides,
                view_ty,
                ty,
            }),
            0,
        );

        let want = "\
%30 = agen.symbolic_vector_load %20[indices:(%6, %5), strides:(%64, %1)] \
{load_order = affine_map<(d0) -> (d0)>, \
load_set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>} : memref<128xi8>, vector<128xi8>
agen.symbolic_vector_store %30, %22[indices:(%6, %5), strides:(%64, %1)] \
{store_order = affine_map<(d0) -> (d0)>, \
store_set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>} : memref<128xi8>, vector<128xi8>
";

        for (at, (want_line, got_line)) in want.lines().zip(got.lines()).enumerate() {
            assert_eq!(want_line, got_line, "symbolic access line {at} diverges");
        }
        assert_eq!(want.lines().count(), got.lines().count());
    }
}

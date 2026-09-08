//! `Agen.td` — THE ADDRESS GENERATOR'S ACCESSES AND COMPOSITE TRANSFERS.
//!
//! The dialect declares fifteen operations; the five here are the ones an emitted program contains.

use std::fmt::Write as _;

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
    /// The time steps the transfer takes — a single pinned step when it fits in one vector.
    pub time_set: IntegerSet,
    /// The order among them.
    pub time_order: AffineMap,
    /// The source offset at each time step, one result per source dimension.
    pub load_time_addr_map: AffineMap,
    /// The destination offset at each time step, one result per destination dimension.
    pub store_time_addr_map: AffineMap,
    /// The region, entered once per time step.
    pub body: Vec<super::Op>,
}

/// ONE VALUE AN `agen.yield` HANDS BACK, and the type its assembly format prints beside it.
///
/// ⛔ THE OPERAND LIST AND THE TYPE LIST ARE ONE GROUP: `attr-dict ($operands^ `:`
/// type($operands))?` (`Agen.td:91`) prints the values with their types or prints neither, so a
/// value here cannot arrive without the type it is stated at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Yielded {
    /// The value.
    pub val: Val,
    /// Its type.
    pub ty: Vector,
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
/// on BOTH producers (`:1250` of the extract, and [`access_order`]), so the identity over the view's
/// rank is the whole answer and a field for it would be a second one.
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

/// ONE SLICE'S ENTRY IN A `slice_mask_map` — which of a stick's masks covers that slice.
///
/// ⭐ FOUR SPELLINGS AND NOTHING ELSE, and the op's own predicates are what census them:
/// `isUnmask()` matches `(0)` eight times over and `isFullMask()` `(1)` eight times over
/// (`Agen.cpp:2726-2739`), while the stick-mask lowering writes `(A)` before its transition slice,
/// `(A|B)` at it and `(1)` after (`SNStickMaskLowering.cpp:52-59`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceMask {
    /// `(0)` — nothing is masked in this slice.
    Unmasked,
    /// `(A)` — mask A covers it.
    A,
    /// `(A|B)` — the TRANSITION slice, where both masks apply.
    AOrB,
    /// `(1)` — the whole slice is masked.
    Full,
}

impl SliceMask {
    /// How one entry prints inside the `slice_mask_map` string.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            SliceMask::Unmasked => "(0)",
            SliceMask::A => "(A)",
            SliceMask::AOrB => "(A|B)",
            SliceMask::Full => "(1)",
        }
    }
}

/// HOW MANY ELEMENTS ONE MASK LEAVES AND HOW MANY IT TAKES — `maskA`/`maskB`'s
/// `(unmasked = N : i32, masked = M : i32)`.
///
/// ⛔ THE PAIR IS ONE VALUE BECAUSE THE VERIFIER TREATS IT AS ONE: `num_unmasked_elements` without
/// `num_masked_elements` and two arrays of different lengths are both refused
/// (`Agen.cpp:2708-2723`), and neither is spellable here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskCounts {
    /// `num_unmasked_elements[i]` — the `first` of the view's `maskX_` pair.
    pub unmasked: i32,
    /// `num_masked_elements[i]` — its `second`.
    pub masked: i32,
}

/// A `composite_load`'s operands, attributes and region.
///
/// See [`Op::CompositeLoad`] for what the op means and how it differs from the pair
/// [`CompositeTransfer`] describes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeLoad {
    /// `$mem_ref` — the view read from.
    pub view: Val,
    /// `dbgName`.
    pub dbg_name: Option<String>,
    /// `$affine_map` applied to `$map_operands` — the base address, printed as the subscript.
    pub indices: Vec<Index>,
    /// The view's type.
    pub view_ty: MemRef,
    /// The block argument carrying the vector loaded at each time step — `getLoadInductionVar()`.
    pub load_iv: Val,
    /// That vector's type — ONE hardware vector, never the whole transfer.
    pub load_iv_ty: Vector,
    /// `load_set` — which elements form each loaded vector.
    pub load_set: IntegerSet,
    /// `load_order` — how those elements are packed.
    pub load_order: AffineMap,
    /// `$time_symbols` — the symbols `time_set` is parameterised by.
    pub time_symbols: Vec<Val>,
    /// `time_set` — the time steps the load takes.
    pub time_set: IntegerSet,
    /// `time_order` — the order among them.
    pub time_order: AffineMap,
    /// `time_addr_map` — the source offset at each time step, one result per view dimension.
    pub time_addr_map: AffineMap,
    /// The region, entered once per time step. Its terminator is [`Op::Yield`].
    pub body: Vec<super::Op>,
}

/// A `composite_store`'s operands and attributes — [`CompositeLoad`] with the direction reversed.
///
/// ⛔⛔ NO BLOCK ARGUMENT, AND THAT IS WHY ITS REGION YIELDS. `CompositeStoreOp`'s region is built
/// from a bare `Block` (`Agen.cpp:735`), so the vector it writes cannot arrive as an argument the
/// way a load's does: the producing op is CLONED into the region and its result becomes the
/// terminator's operand (`SNTransferLowering.cpp:1313-1319`). See [`Op::Yield`].
///
/// ⛔ THE `input_vector` FORM IS THE OTHER HALF OF ONE `verify()`, and it is not this one. The op
/// takes EXACTLY one of `input_vector` or (`store_set` + `store_order`) (`Agen.cpp:782-800`); the
/// region form is what entry 089 builds, so the operand has no field here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeStore {
    /// `$mem_ref` — the view written to.
    pub view: Val,
    /// `dbgName`.
    pub dbg_name: Option<String>,
    /// `$affine_map` applied to `$map_operands` — the base address, printed as the subscript.
    pub indices: Vec<Index>,
    /// The view's type.
    pub view_ty: MemRef,
    /// `store_set` — which elements each stored vector covers.
    pub store_set: IntegerSet,
    /// `store_order` — how those elements are packed.
    pub store_order: AffineMap,
    /// `$time_symbols` — the symbols `time_set` is parameterised by.
    pub time_symbols: Vec<Val>,
    /// `time_set` — the time steps the store takes.
    pub time_set: IntegerSet,
    /// `time_order` — the order among them.
    pub time_order: AffineMap,
    /// `time_addr_map` — the destination offset at each time step, one result per view dimension.
    pub time_addr_map: AffineMap,
    /// The region, entered once per time step. Its terminator is [`Op::Yield`], and it is what
    /// carries the stored vector.
    pub body: Vec<super::Op>,
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
        /// `dbgName` — the transfer's own name, and [`None`] for the accesses that carry none.
        dbg_name: Option<String>,
        /// `load_set` — see [`Access`] for why one of its two spellings is stored and not derived.
        access: Access,
        /// The view's type. Its INNERMOST extent is the lane count.
        view_ty: MemRef,
        /// The vector's type.
        ty: Vector,
    },

    /// `agen.composite_load_and_store src:%s[..] dst:%d[..] time_symbols(), load_iv(%v:vector<..>)
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

    /// `agen.composite_load %view[..] time_symbols()(%v:vector<..>) {..} { .. } : memref<..>`.
    ///
    /// ⭐⭐ THE LOAD HALF ALONE, FOR THE TRANSFER WHOSE DESTINATION IS NOT A VIEW. A streaming or
    /// double-buffered load out of the HBM hands its vector to a `dataflow.send` inside the region
    /// instead of storing it (`SNTransferLowering.cpp:1053-1102`) — there is no `dst` to name, so the
    /// pair op above cannot state it.
    ///
    /// ⛔ IT BINDS NOTHING. `CompositeLoadOp` declares no results (`Agen.td:423-442`); the loaded
    /// vector reaches its consumers as the region's argument, and a caller that wants it reads
    /// [`CompositeLoad::load_iv`] — the reference's own `composite_load.getLoadInductionVar()`.
    ///
    /// ⛔ BOXED for the reason [`Op::CompositeLoadAndStore`] is: three affine maps, two integer sets
    /// and a subscript, in an enum as large as its largest variant.
    CompositeLoad(Box<CompositeLoad>),

    /// `agen.composite_store %view[..] time_symbols() {..} { .. } : memref<..>` — the store half of
    /// a composite transfer, walking its own time axis with the chain that produces each vector in
    /// its region.
    ///
    /// ⛔ IT BINDS NOTHING, like [`Op::CompositeLoad`], and unlike the load it takes nothing from
    /// its region either — see [`CompositeStore`].
    ///
    /// ⛔ BOXED for the reason [`Op::CompositeLoad`] is.
    CompositeStore(Box<CompositeStore>),

    /// `agen.yield` — the terminator of a composite transfer's region, carrying the values that
    /// region hands back to its parent.
    ///
    /// ⛔ EMPTY FOR EVERY LOAD-SIDE REGION AND NOT FOR THE STORE'S. A composite load's region
    /// reads its vector from the op's own block argument and yields nothing; a
    /// [`composite_store`](Op::CompositeStore)'s region has no block argument, so the vector it
    /// writes reaches the op through this terminator (`SNTransferLowering.cpp:1316-1319`).
    Yield {
        /// `$operands`.
        values: Vec<Yielded>,
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
        /// `dbgName` — the transfer's own name, and [`None`] for the accesses that carry none.
        dbg_name: Option<String>,
        /// `store_set` — see [`Access`], and [`Op::VectorLoad`] for the load half of the same pair.
        access: Access,
        /// The view's type. Its INNERMOST extent is the lane count.
        view_ty: MemRef,
        /// The vector's type.
        ty: Vector,
    },

    /// `%m = agen.set_transfer_mask_state mask_value(%c) { num_slices, slice_mask_map, maskA, maskB }
    /// :  index , vector<..>` — the SAMV that arms a stick's transfer mask.
    ///
    /// ⭐⭐ WITHOUT IT A PARTIAL STICK IS TRANSFERRED WHOLE. A stick whose live elements stop
    /// part-way needs the masked tail suppressed, and this op is what states where that tail begins;
    /// `constructStickMaskOperation` (`SNStickMaskLowering.cpp:21`) has no other emission.
    ///
    /// ⛔ `num_slices` IS NOT A FIELD — it is [`slice_mask_map`](Self::SetTransferMaskState::slice_mask_map)'s
    /// length. The reference writes `8` beside an eight-entry map and every reader indexes one by the
    /// other (`Agen.td:1041-1116`), so two fields would be two answers to one question.
    SetTransferMaskState {
        /// The vector this binds — the armed mask.
        result: Val,
        /// `$mask_value` — the constant the mask is programmed from.
        mask_value: Val,
        /// `dbgName`.
        ///
        /// ⚠️ CARRIED AND NEVER PRINTED. `constructStickMaskOperation` sets it from the stick mask's
        /// own name (`SNStickMaskLowering.cpp:72`), and the op's custom printer emits the operand and
        /// four attributes without it (`Agen.cpp:2677-2705`). [`emit`] does the same.
        dbg_name: Option<String>,
        /// `slice_mask_map`, one entry per slice, in slice order.
        slice_mask_map: Vec<SliceMask>,
        /// `maskA`, `maskB`, … — one per mask, and EMPTY for a reset or a full mask, which name no
        /// element counts at all (`set_transfer_mask_state_rt.mlir:11-12`).
        masks: Vec<MaskCounts>,
        /// The result's type — 128 BYTES' WORTH of the element, which is what the mask covers.
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
            dbg_name,
            access,
            view_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = agen.vector_load {}[{}] {{{}load_order = {}, load_set = {}}} : {}, {}",
                print::val(*result),
                print::val(*view),
                print::index_list(indices),
                // ⛔ `dbgName` COMES FIRST BECAUSE THE DICTIONARY IS ALPHABETICAL — the op has no
                // custom printer (`Agen.td:390-421`), so `printOptionalAttrDict` writes its
                // attributes in name order and `d` precedes `l`.
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
                time_set,
                time_order,
                load_time_addr_map,
                store_time_addr_map,
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
                " time_symbols(), load_iv({}:{})",
                print::val(*load_iv),
                print::vector(*load_iv_ty)
            );
            print::indent(out, depth);
            let _ = writeln!(
                out,
                " {{load_order = {}, load_set = {}, load_time_addr_map = {}, store_order = {}, \
                 store_set = {}, store_time_addr_map = {}, time_order = {}, time_set = {}}}",
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
        Op::CompositeLoad(load) => {
            let CompositeLoad {
                view,
                dbg_name,
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
                body,
            } = load.as_ref();
            let _ = writeln!(
                out,
                "agen.composite_load {}[{}]",
                print::val(*view),
                print::index_list(indices),
            );
            print::indent(out, depth);
            // ⛔ NO COMMA BETWEEN THE TWO PARENTHESISED GROUPS, unlike the pair op above:
            // `" time_symbols(" << syms << ')' << '(' << iv << ':' << type << ')'`
            // (`Agen.cpp:591-596`), where the sibling writes `"), load_iv("` (`:353`).
            let _ = writeln!(
                out,
                " time_symbols({})({}:{})",
                time_symbols
                    .iter()
                    .map(|sym| print::val(*sym))
                    .collect::<Vec<_>>()
                    .join(", "),
                print::val(*load_iv),
                print::vector(*load_iv_ty)
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
        Op::CompositeStore(store) => {
            let CompositeStore {
                view,
                dbg_name,
                indices,
                view_ty,
                store_set,
                store_order,
                time_symbols,
                time_set,
                time_order,
                time_addr_map,
                body,
            } = store.as_ref();
            let _ = writeln!(
                out,
                "agen.composite_store {}[{}]",
                print::val(*view),
                print::index_list(indices),
            );
            print::indent(out, depth);
            // ⛔ NO `(iv:type)` GROUP: the store has no block argument to name, so its printer stops
            // at the closing paren (`Agen.cpp:815-817`).
            let _ = writeln!(
                out,
                " time_symbols({})",
                time_symbols
                    .iter()
                    .map(|sym| print::val(*sym))
                    .collect::<Vec<_>>()
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
            print::indent(out, depth);
            let _ = writeln!(out, "}} : {}", print::memref(view_ty));
        }
        Op::Yield { values } => {
            if values.is_empty() {
                out.push_str("agen.yield\n");
            } else {
                let _ = writeln!(
                    out,
                    "agen.yield {} : {}",
                    values
                        .iter()
                        .map(|yielded| print::val(yielded.val))
                        .collect::<Vec<_>>()
                        .join(", "),
                    values
                        .iter()
                        .map(|yielded| print::vector(yielded.ty))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
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
                // ⛔ `dbgName` FIRST, for the reason [`Op::VectorLoad`]'s printer records: the op has
                // no custom printer (`Agen.td:188-231`) and `d` precedes `s`.
                dbg_name
                    .as_ref()
                    .map_or(String::new(), |name| format!("dbgName = \"{name}\", ")),
                print::affine_map(&access_order(view_ty.shape.len())),
                print::integer_set(&access.set(view_ty, ty.len)),
                print::memref(view_ty),
                print::vector(*ty)
            );
        }
        Op::SetTransferMaskState {
            result,
            mask_value,
            dbg_name: _,
            slice_mask_map,
            masks,
            ty,
        } => {
            // ⛔ THE MASKS ARE LETTERED FROM 'A' IN LIST ORDER — `char mask_id = 'A'; .. ++mask_id`
            // over `llvm::zip(num_unmasked_elems, num_masked_elems)` (`Agen.cpp:2699-2703`), so the
            // FIRST entry is `maskA` and position is the whole of its identity.
            let mut elements = String::new();
            for (at, counts) in masks.iter().enumerate() {
                let id = char::from(b'A'.saturating_add(u8::try_from(at).unwrap_or(u8::MAX)));
                let _ = write!(
                    elements,
                    ", mask{id} = \"(unmasked = {} : i32, masked = {} : i32)\"",
                    counts.unmasked, counts.masked
                );
            }
            // ⛔ TWO SPACES AFTER THE COLON AND ONE BEFORE THE COMMA — `" } :  " << getMaskValueType()
            // << " , " << op.getType()` (`Agen.cpp:2705`), which is what the round-trip CHECK line
            // compares against (`set_transfer_mask_state_rt.mlir:10`).
            let _ = writeln!(
                out,
                "{} = agen.set_transfer_mask_state mask_value({}) {{ num_slices = {} : i32, \
                 slice_mask_map = \"{}\"{elements} }} :  index , {}",
                print::val(*result),
                print::val(*mask_value),
                slice_mask_map.len(),
                slice_mask_map
                    .iter()
                    .map(|slice| slice.spelling())
                    .collect::<String>(),
                print::vector(*ty)
            );
        }
    }
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
    use crate::islands::dataflow_ir::dialects::agen::{CompositeLoad, CompositeTransfer, Op};
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
            time_set: planned.time_set,
            time_order: planned.time_order,
            load_time_addr_map: planned.load_time_addr_map,
            store_time_addr_map: planned.store_time_addr_map,
            body: vec![dialects::Op::Agen(Op::Yield { values: Vec::new() })],
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

    /// ⭐⭐ THE L0LU'S COMPOSITE LOAD, AS THE AUTHORITY WRITES IT.
    ///
    /// `dcc/test/L0LU/sync_send_recv_L0LUrow0_src_unit.mlir:71-78` loads four `i8` elements per time
    /// step out of a `memref<8x2x1x1x1x1xi8>` view of the L0, splats them to sixteen lanes and sends
    /// them to the PT. Every attribute below is that file's `#map`/`#set` alias SPELLED OUT — the
    /// island's printer has no alias table, for the reason [`prints_the_hbm_to_lx_transfer`] records.
    ///
    /// ⛔ THIS IS THE SHAPE ENTRY 080's COMPOSITE ARM PRODUCES, which is why the fixture is the one
    /// chosen: a chunked `load_set` (four elements on `d0`, every other dim pinned), a five-dim
    /// `time_set`, a reversed-identity `time_order` and a `time_addr_map` with one result per view
    /// dimension. See [`crate::bridges::superdsc_to_dataflow_ir::transfer`].
    ///
    /// ⚠️ THE FIXTURE'S REGION BRACE CARRIES ONE EXTRA SPACE and this printer's does not. The two
    /// composite printers reach it identically — `printNewline()` then
    /// `printRegion(region, false, true)` (`Agen.cpp:598-599` and `:369-370`) — and the
    /// `composite_load_and_store` fixture puts the brace at the op's own column
    /// (`dcc/test/Conversion/SentientToProgIR/L3/gather.mlir:68-73`), so the extra space in this
    /// hand-written input file is the file's and not the printer's. Indentation is not part of the
    /// IR: MLIR's parser accepts either.
    #[test]
    fn prints_the_l0lu_composite_load() {
        use crate::islands::dataflow_ir::dialects::{dataflow, vectorchain};
        use crate::islands::dataflow_ir::link::{L0lu, Link, PtRowUnit};
        use crate::islands::dataflow_ir::ty::{
            AffineExpr, AffineMap, Constraint, ElemType, IntegerSet, MemRef, Vector,
        };

        /// `d{n} >= 0` and `-d{n} + span >= 0` — the pair a spanning dimension is written as.
        fn spans(n: u32, span: i64) -> [Constraint; 2] {
            [
                Constraint {
                    expr: AffineExpr::dim(n),
                    is_equality: false,
                },
                Constraint {
                    // ⛔ `-d4 >= 0` AND NOT `-d4 + 0 >= 0` — a zero span adds no term at all
                    // (`#set3`, `:41`), which is `simplifyAdd`'s `x + 0` fold.
                    expr: match span {
                        0 => AffineExpr::dim(n).times(-1),
                        span => AffineExpr::dim(n).times(-1).plus(AffineExpr::Const(span)),
                    },
                    is_equality: false,
                },
            ]
        }

        /// `d{n} == 0` — a pinned dimension.
        fn pinned(n: u32) -> Constraint {
            Constraint {
                expr: AffineExpr::dim(n),
                is_equality: true,
            }
        }

        let elem = ElemType::Int(8);
        let loaded = Vector { len: 4, elem };
        let sent = Vector { len: 16, elem };
        let load_iv = Val(31);
        let selected = Val(22);
        // `dataflow.send %3, %22` — the PT the l0lu hands each splatted vector to.
        let (to_pt, _) = Link::<L0lu, PtRowUnit<0>>::between(Val(4), Val(3)).ends();

        let op = dialects::Op::Agen(Op::CompositeLoad(Box::new(CompositeLoad {
            view: Val(21),
            dbg_name: None,
            indices: vec![Index::Const(0); 6],
            view_ty: MemRef {
                shape: vec![8, 2, 1, 1, 1, 1],
                elem,
            },
            load_iv,
            load_iv_ty: loaded,
            // `#set2` — four elements on `d0`, the other five dims pinned.
            load_set: IntegerSet {
                dims: 6,
                symbols: 0,
                constraints: spans(0, 3).into_iter().chain((1..6).map(pinned)).collect(),
            },
            // `#map1`.
            load_order: super::access_order(6),
            // ⭐ EMPTY, AND THE PRINTED `time_symbols()` IS THAT — the fixture's is empty too.
            time_symbols: Vec::new(),
            // `#set3`.
            time_set: IntegerSet {
                dims: 5,
                symbols: 0,
                constraints: [(0, 1), (1, 1), (2, 2), (3, 2), (4, 0)]
                    .into_iter()
                    .flat_map(|(dim, span)| spans(dim, span))
                    .collect(),
            },
            // `#map3` — the reversed identity, which is what `time_order` always is.
            time_order: AffineMap {
                dims: 5,
                syms: 0,
                results: (0..5).rev().map(AffineExpr::dim).collect(),
            },
            // `#map2` — one result per VIEW dimension, from the five time dimensions.
            time_addr_map: AffineMap {
                dims: 5,
                syms: 0,
                results: vec![
                    AffineExpr::dim(1).times(4),
                    AffineExpr::dim(0),
                    AffineExpr::Const(0),
                    AffineExpr::Const(0),
                    AffineExpr::Const(0),
                    AffineExpr::dim(4),
                ],
            },
            body: vec![
                dialects::Op::VectorChain(vectorchain::Op::Select {
                    result: selected,
                    input: load_iv,
                    // `#map4` — the splat the replication factor asks for.
                    selection_map: AffineMap {
                        dims: 1,
                        syms: 0,
                        results: vec![AffineExpr::dim(0).modulo(4)],
                    },
                    input_ty: loaded,
                    ty: sent,
                }),
                dialects::Op::Dataflow(dataflow::Op::Send {
                    to: to_pt,
                    data: selected,
                    ty: sent,
                }),
                dialects::Op::Agen(Op::Yield { values: Vec::new() }),
            ],
        })));

        let mut got = String::new();
        emit(&mut got, &op, 0);

        let want = "\
agen.composite_load %21[0, 0, 0, 0, 0, 0]
 time_symbols()(%31:vector<4xi8>)
 {load_order = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>, \
load_set = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 + 3 >= 0, d1 == 0, d2 == 0, \
d3 == 0, d4 == 0, d5 == 0)>, \
time_addr_map = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 4, d0, 0, 0, 0, d4)>, \
time_order = affine_map<(d0, d1, d2, d3, d4) -> (d4, d3, d2, d1, d0)>, \
time_set = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 1 >= 0, \
d2 >= 0, -d2 + 2 >= 0, d3 >= 0, -d3 + 2 >= 0, d4 >= 0, -d4 >= 0)>}
{
  %22 = vectorchain.select %31 {selection_map = affine_map<(d0) -> (d0 mod 4)>} : \
vector<4xi8>, vector<16xi8>
  dataflow.send %3, %22 : vector<16xi8>
  agen.yield
} : memref<8x2x1x1x1x1xi8>
";

        for (at, (want_line, got_line)) in want.lines().zip(got.lines()).enumerate() {
            assert_eq!(want_line, got_line, "composite load line {at} diverges");
        }
        assert_eq!(want.lines().count(), got.lines().count());
    }

    /// ⭐ AND THE NAME AND THE STATED SET, WHICH ARE THE TWO FIELDS A TRANSFER'S LOAD ADDS.
    ///
    /// ⛔ `dbgName` PRINTS FIRST. The op has no custom printer, so `printOptionalAttrDict` orders its
    /// attributes by name (`Agen.td:390-421`) and `dbgName` precedes `load_order` and `load_set`.
    ///
    /// ⛔ AND THE SET IS THE ONE STATED, NOT THE ONE THE VIEW IMPLIES: a two-element chunk out of a
    /// four-lane view is what [`Access::Stated`] exists for, and [`access_set`] would have written
    /// `-d1 + 3 >= 0` from the same memref.
    #[test]
    fn prints_a_named_vector_load_with_a_stated_set() {
        use crate::islands::dataflow_ir::dialects::agen::Access;
        use crate::islands::dataflow_ir::ty::{
            AffineExpr, Constraint, ElemType, IntegerSet, MemRef, Vector,
        };

        let stated = IntegerSet {
            dims: 2,
            symbols: 0,
            constraints: vec![
                Constraint {
                    expr: AffineExpr::dim(0),
                    is_equality: true,
                },
                Constraint {
                    expr: AffineExpr::dim(1),
                    is_equality: false,
                },
                Constraint {
                    expr: AffineExpr::dim(1).times(-1).plus(AffineExpr::Const(1)),
                    is_equality: false,
                },
            ],
        };
        let op = dialects::Op::Agen(Op::VectorLoad {
            result: Val(7),
            view: Val(5),
            indices: vec![Index::Const(0), Index::Val(Val(6))],
            dbg_name: Some("hbm-to-lx".to_owned()),
            access: Access::Stated(stated.clone()),
            view_ty: MemRef {
                shape: vec![2, 4],
                elem: ElemType::F16,
            },
            ty: Vector {
                len: 2,
                elem: ElemType::F16,
            },
        });

        let mut got = String::new();
        emit(&mut got, &op, 0);
        assert_eq!(
            got,
            "%7 = agen.vector_load %5[0, %6] {dbgName = \"hbm-to-lx\", \
             load_order = affine_map<(d0, d1) -> (d0, d1)>, \
             load_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 1 >= 0)>} : \
             memref<2x4xf16>, vector<2xf16>\n"
        );

        // And [`Access::OfView`] is the derivation the same view and lane count give.
        assert_eq!(
            Access::Stated(stated).set(
                &MemRef {
                    shape: vec![2, 4],
                    elem: ElemType::F16,
                },
                2
            ),
            Access::OfView.set(
                &MemRef {
                    shape: vec![2, 2],
                    elem: ElemType::F16,
                },
                2
            ),
            "a stated set and a derived one over the same extents are the same set"
        );
    }
}

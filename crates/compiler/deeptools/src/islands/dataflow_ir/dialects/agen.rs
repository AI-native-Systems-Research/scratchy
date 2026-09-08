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

    /// `agen.yield` — the terminator of a composite transfer's region.
    Yield,

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
        /// The view's type. Its INNERMOST extent is the lane count.
        view_ty: MemRef,
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
            view_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = agen.vector_load {}[{}] {{load_order = {}, load_set = {}}} : {}, {}",
                print::val(*result),
                print::val(*view),
                print::index_list(indices),
                print::affine_map(&access_order(view_ty.shape.len())),
                print::integer_set(&access_set(view_ty, ty.len)),
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
        Op::Yield => {
            out.push_str("agen.yield\n");
        }
        Op::VectorStore {
            value,
            view,
            indices,
            view_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "agen.vector_store {}, {}[{}] {{store_order = {}, store_set = {}}} : {}, {}",
                print::val(*value),
                print::val(*view),
                print::index_list(indices),
                print::affine_map(&access_order(view_ty.shape.len())),
                print::integer_set(&access_set(view_ty, ty.len)),
                print::memref(view_ty),
                print::vector(*ty)
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
            time_set: planned.time_set,
            time_order: planned.time_order,
            load_time_addr_map: planned.load_time_addr_map,
            store_time_addr_map: planned.store_time_addr_map,
            body: vec![dialects::Op::Agen(Op::Yield)],
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

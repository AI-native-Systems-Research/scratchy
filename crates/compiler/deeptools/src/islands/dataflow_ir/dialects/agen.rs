//! `Agen.td` — THE ADDRESS GENERATOR'S ACCESSES AND COMPOSITE TRANSFERS.
//!
//! The dialect declares fifteen operations; the four here are the ones an emitted program contains.

use std::fmt::Write as _;

use crate::islands::dataflow_ir::dialects::{Index, Val};
use crate::islands::dataflow_ir::print;
use crate::islands::dataflow_ir::ty::{AffineMap, IntegerSet, MemRef, Vector};

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
                identity_map(view_ty.shape.len()),
                lane_set(view_ty, ty.len),
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
                identity_map(view_ty.shape.len()),
                lane_set(view_ty, ty.len),
                print::memref(view_ty),
                print::vector(*ty)
            );
        }
    }
}

/// `affine_map<(d0, .., dn) -> (d0, .., dn)>` — the identity over `rank` dims.
///
/// ⭐ `load_order`/`store_order` SAY WHICH AXIS MOVES FASTEST, and the scheduler writes the identity
/// for every access in its own output (`#map4`, `#map8`). Row-major order is what the view's own
/// `layout_map` already states, so ordering it again differently would be two answers to one
/// question.
fn identity_map(rank: usize) -> String {
    let dims: Vec<String> = (0..rank).map(|d| format!("d{d}")).collect();
    format!("affine_map<({}) -> ({})>", dims.join(", "), dims.join(", "))
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
fn lane_set(ty: &MemRef, lanes: u64) -> String {
    let rank = ty.shape.len();
    let dims: Vec<String> = (0..rank).map(|d| format!("d{d}")).collect();
    let last = rank.saturating_sub(1);
    let mut constraints: Vec<String> = (0..last).map(|d| format!("d{d} == 0")).collect();
    constraints.push(format!("d{last} >= 0"));
    constraints.push(format!("-d{last} + {} >= 0", lanes.saturating_sub(1)));
    format!(
        "affine_set<({}) : ({})>",
        dims.join(", "),
        constraints.join(", ")
    )
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
}

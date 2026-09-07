//! UPSTREAM `vector` — the two plain memory accesses the vectorchain lowerings still accept.
//!
//! Not one of the scheduler's own dialects, and not a form this bridge emits: `agen.vector_load` and
//! `agen.vector_store` are what a producer here writes, for the reason recorded on
//! [`super::agen::Op::VectorLoad`]. These two exist because the PT and PE/SFP lowerings **read**
//! them, and a lowering's branch on an input it can be given is not a branch that may go missing.
//!
//! ⭐⭐ THE VENDOR'S OWN TESTS FEED THEM TO EXACTLY THOSE PASSES. Nine occurrences in the authority
//! tree's `dcc/test`, all in the two conversion suites this bridge is a port of: eight
//! `vector.store`s in `Conversion/VectorChainToSentientPT/dynamic_pt_masking.mlir` (`:254`, `:262`,
//! `:298`, `:308`, `:360`, `:368`, `:404`, `:414`) and one `vector.load` in
//! `Conversion/VectorChainToSentientPESFP/sfp-to-sfp-ring.mlir:171`.
//!
//! ⛔ AND THEY ARE THE UNDERTESTED ARM, WHICH IS WHY THE SHAPE MATTERS. `VectorChainToSentientPESFP`
//! carries a `StoreOpLowering` for `vector::StoreOp` beside its `VectorStoreOpLowering` for the
//! `agen` one, and the former's `reuse_info_.getId(..).value()` is unguarded
//! (`VectorChainToSentientPESFP.cpp:123`) — see [`super::agen::Op::VectorLoad`], which records that
//! the scheduler never produces a `vector.store` for it to see.

use std::fmt::Write as _;

use crate::islands::dataflow_ir::dialects::{Index, Val};
use crate::islands::dataflow_ir::print;
use crate::islands::dataflow_ir::ty::{MemRef, Vector};

/// ONE `vector` OPERATION.
///
/// # ⛔ NO ORDER MAP AND NO LANE SET, AND THAT ABSENCE IS THE WHOLE DIFFERENCE
///
/// ⛔⛔ `Vector_LoadOp`'s arguments are `$base` plus `Variadic<Index>:$indices` and nothing else that
/// an emitted program carries; its format is
/// `"$base `[` $indices `]` attr-dict `:` type($base) `,` type($result)"`, and `Vector_StoreOp` is
/// the same with `$valueToStore` in front. The `agen` pair instead carry `load_order`/`store_order`
/// and `load_set`/`store_set`, which is why [`super::agen`] derives them at print time and this
/// dialect has nothing to derive.
///
/// ⭐ WHICH IS EXACTLY WHAT `getLayoutMapAndIndices` READS OFF THEM. Its `agen` arms compose the
/// view's `layout_map` with the access's own map and then its order map; its two `vector` arms take
/// `getIndices()` and the view's `layout_map` **alone**, with no compose at all
/// (`Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:879-921`). No order map exists
/// to compose with.
///
/// ⛔ THE OPTIONAL ATTRIBUTES ARE NOT HERE. Upstream declares `nontemporal` (a
/// `DefaultValuedOptionalAttr<BoolAttr, "false">`) and an optional `alignment`; neither appears on
/// any of the nine occurrences in the authority tree, and `false` prints nothing. A producer that
/// needs one adds the field; representing it as an always-default is a field with one value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// `%v = vector.load %base[..] : memref<..>, vector<..>`.
    ///
    /// `%data1 = vector.load %lrf_memory_fp16[%c4, %c0] : memref<8x64xf16>, vector<64xf16>`
    /// (`dcc/test/Conversion/VectorChainToSentientPESFP/sfp-to-sfp-ring.mlir:171`).
    Load {
        /// The vector it binds.
        result: Val,
        /// `$base` — the view read. ⭐ SPELLED `base`, NOT `mem_ref`: the `agen` pair's accessor is
        /// `getMemRef()` and this one's is `getBase()`, and
        /// [`getLayoutMapAndIndices`](crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::layout_map_and_indices)
        /// calls each by its own name.
        base: Val,
        /// `$indices` — one per view dimension.
        indices: Vec<Index>,
        /// `type($base)`.
        base_ty: MemRef,
        /// `type($result)`.
        ty: Vector,
    },

    /// `vector.store %value, %base[..] : memref<..>, vector<..>`.
    ///
    /// `vector.store %270, %19[%c0, %c0] : memref<4x64xf16>, vector<64xf16>`
    /// (`dcc/test/Conversion/VectorChainToSentientPT/dynamic_pt_masking.mlir:254`).
    Store {
        /// `$valueToStore`.
        value: Val,
        /// `$base` — the view written; see [`Op::Load::base`].
        base: Val,
        /// `$indices` — one per view dimension.
        indices: Vec<Index>,
        /// `type($base)`.
        base_ty: MemRef,
        /// `type($valueToStore)`.
        ty: Vector,
    },
}

/// ONE `vector` OP AS TEXT. The caller has already indented the opening line.
pub(crate) fn emit(out: &mut String, op: &Op) {
    match op {
        Op::Load {
            result,
            base,
            indices,
            base_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = vector.load {}[{}] : {}, {}",
                print::val(*result),
                print::val(*base),
                print::index_list(indices),
                print::memref(base_ty),
                print::vector(*ty)
            );
        }
        Op::Store {
            value,
            base,
            indices,
            base_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "vector.store {}, {}[{}] : {}, {}",
                print::val(*value),
                print::val(*base),
                print::index_list(indices),
                print::memref(base_ty),
                print::vector(*ty)
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::islands::dataflow_ir::dialects::vector::Op;
    use crate::islands::dataflow_ir::dialects::{self, Index, Val};
    use crate::islands::dataflow_ir::print::emit;
    use crate::islands::dataflow_ir::ty::{ElemType, MemRef, Vector};

    /// ⭐ THE PT MASKING TEST'S STORE, VERBATIM.
    ///
    /// `vector.store %270, %19[%c0, %c0] : memref<4x64xf16>, vector<64xf16>`
    /// (`dcc/test/Conversion/VectorChainToSentientPT/dynamic_pt_masking.mlir:254`). ⛔ ITS INDICES
    /// ARE `%c0` TWICE, not the literal `0` an `agen` access prints — the vendor writes them as
    /// operands, which is what [`Index::Val`] spells.
    #[test]
    fn prints_the_pt_masking_stores_access() {
        let op = dialects::Op::Vector(Op::Store {
            value: Val(270),
            base: Val(19),
            indices: vec![Index::Val(Val(2)), Index::Val(Val(2))],
            base_ty: MemRef {
                shape: vec![4, 64],
                elem: ElemType::F16,
            },
            ty: Vector {
                len: 64,
                elem: ElemType::F16,
            },
        });

        let mut got = String::new();
        emit(&mut got, &op, 0);
        assert_eq!(
            got,
            "vector.store %270, %19[%2, %2] : memref<4x64xf16>, vector<64xf16>\n"
        );
    }

    /// ⭐ THE SFP RING TEST'S LOAD, VERBATIM —
    /// `%data1 = vector.load %lrf_memory_fp16[%c4, %c0] : memref<8x64xf16>, vector<64xf16>`
    /// (`dcc/test/Conversion/VectorChainToSentientPESFP/sfp-to-sfp-ring.mlir:171`).
    #[test]
    fn prints_the_sfp_rings_load() {
        let op = dialects::Op::Vector(Op::Load {
            result: Val(7),
            base: Val(3),
            indices: vec![Index::Val(Val(4)), Index::Val(Val(0))],
            base_ty: MemRef {
                shape: vec![8, 64],
                elem: ElemType::F16,
            },
            ty: Vector {
                len: 64,
                elem: ElemType::F16,
            },
        });

        let mut got = String::new();
        emit(&mut got, &op, 0);
        assert_eq!(
            got,
            "%7 = vector.load %3[%4, %0] : memref<8x64xf16>, vector<64xf16>\n"
        );
    }
}

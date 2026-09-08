//! THE SYNC STATEMENTS.
//! ⛔ THE ORDER IS dxp's AND CITED: reordering syncs times the card out (CB state=TimedOut, sync rc=-1).
//!
//! 4 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e033_constructUnitsForUniformization` | 0 | 90 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNSyncLowering.cpp:47` |
//! | `e061_constructUnits` | 1 | 25 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNSyncLowering.cpp:20` |
//! | `e062_constructImplicitSyncOperation` | 1 | 66 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNSyncLowering.cpp:138` |
//! | `e076_constructSyncOperation` | 2 | 69 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNSyncLowering.cpp:205` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e033_constructUnitsForUniformization
// crustify:todo: e076_constructSyncOperation

use std::num::NonZeroI64;

use super::dsc_lowering::{
    Component, Handlers, Retrieved, constant_index, retrieve_get_unit_op_in_same_core,
};
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, dataflow};
use crate::islands::dataflow_ir::ty::{AffineMap, ElemType, MemRef};
use crate::units::{Core, Corelet, DfirUnit, Row};

/// WHICH END A SYNC SIGNAL COMES FROM — `getComponentsFromOtherEnds`' first field.
///
/// ⛔ THE PT ROW IS NOT A [`DfirUnit`]: `L0LUROW0..7` (`sys-arch-spec/arch_enums.h:80-87`) are named
/// nowhere in this conversion except the two filters that SKIP rows 1-7 (`SNSyncLowering.cpp:26`,
/// `:57`), and only `L0LUROW0` is ever bound to a handler
/// (`DSC2ToDataflowIRUtils.hpp:285-296`) — so widening the island's unit vocabulary by seven
/// spellings would add seven units no `get_unit` can name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncEnd {
    /// A unit named outright.
    Unit(DfirUnit),
    /// One row's spelling of the L0 load unit.
    L0luRow(Row),
}

/// Replaces: e061_constructUnits
///
/// EVERY OTHER END OF A SYNC, AS A HANDLE IN **THIS** CORE — one per corelet, except that the L3
/// halves are asked once with no corelet at all (`SNSyncLowering.cpp:20-44`).
///
/// ⛔ ROWS 1-7 OF THE L0 LOAD UNIT ARE DROPPED — the reference's own note, *"skip the signals from
/// the l0lorow1-7"* (`:25`); row 0's signal is kept and asks for `L0LU`.
///
/// ⛔ THE L3 ARM PASSES `-1` (`:36`), which is [`None`] here: an L3 half is core-wide and a
/// `corelet = -1` attribute is not a corelet.
#[must_use]
pub fn construct_units(
    vals: &mut Values,
    handlers: &Handlers,
    ends: &[(SyncEnd, Vec<(Core, Vec<Corelet>)>)],
) -> Vec<Retrieved> {
    let mut units = Vec::new();
    for (end, cores) in ends {
        let comp = match end {
            SyncEnd::Unit(unit) => Component::Unit(*unit),
            SyncEnd::L0luRow(row) if row.get() == 0 => Component::Unit(DfirUnit::L0lu),
            SyncEnd::L0luRow(_) => continue,
        };
        let core_wide = matches!(end, SyncEnd::Unit(DfirUnit::L3lu | DfirUnit::L3su));
        for (core, corelets) in cores {
            if core_wide {
                units.push(retrieve_get_unit_op_in_same_core(
                    vals, handlers, comp, *core, None,
                ));
            } else {
                for corelet in corelets {
                    units.push(retrieve_get_unit_op_in_same_core(
                        vals,
                        handlers,
                        comp,
                        *core,
                        Some(*corelet),
                    ));
                }
            }
        }
    }
    units
}

/// WHICH SIDE OF THE L0 AN IMPLICIT SYNC IS BUILT ON.
///
/// ⛔ THE TWO VARIANTS ARE `DT_CHECK_MSG(is_any_of(this->comp_, L0SU, L0LUROW0), "implicit sync is
/// supported only in L0SU/L0LUROW0")` (`SNSyncLowering.cpp:139-140`) as a type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImplicitSyncSide {
    /// `L0SU` — the store side.
    L0su,
    /// `L0LUROW0` — row 0 of the load side.
    L0luRow0,
}

impl ImplicitSyncSide {
    /// THE OTHER SIDE — `dst_comp = comp_ == L0SU ? L0LUROW0 : L0SU` (`:143`).
    #[must_use]
    pub const fn other(self) -> ImplicitSyncSide {
        match self {
            ImplicitSyncSide::L0su => ImplicitSyncSide::L0luRow0,
            ImplicitSyncSide::L0luRow0 => ImplicitSyncSide::L0su,
        }
    }
}

/// AN IMPLICIT SYNC'S TILE SIZE — the product of `getImplicitSyncTileSizePerDim`'s per-dim sizes.
///
/// ⛔ TWO CHECKS BECOME THIS TYPE: `tilesize_ss >= 1` and `tilesize_el == tilesize_ss`, the latter
/// because *"translator currently doesn't epilogues in implicit sync"* (`SNSyncLowering.cpp:183-186`).
/// Both stages' products go in, so there is nothing left for the sync itself to compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileSize(NonZeroI64);

impl TileSize {
    /// THE TWO PRODUCTS, AGREEING — [`None`] where they differ or the size is below one.
    #[must_use]
    pub const fn of(steady_state: i64, epilogue: i64) -> Option<TileSize> {
        if steady_state != epilogue || steady_state < 1 {
            return None;
        }
        match NonZeroI64::new(steady_state) {
            Some(size) => Some(TileSize(size)),
            None => None,
        }
    }
}

/// Replaces: e062_constructImplicitSyncOperation
///
/// THE `dataflow.implicit_sync_on_streaming_buffer` BETWEEN THE TWO SIDES OF ONE L0 — a tile-size
/// constant, a zero start address, a one-element view of the L0 and the sync itself
/// (`SNSyncLowering.cpp:138-203`).
///
/// ⛔ THE VIEW IS ONE ELEMENT AT A CONSTANT-ZERO LAYOUT — `MemRefType::get(1, element_type)` and
/// `AffineMap::get(1, 0, getAffineConstantExpr(0, ..))` (`:194-195`): the sync reads a doorbell, not
/// the buffer, so a layout derived from the transfer's own shape would name the wrong bytes.
///
/// ⭐ THE SYNC IS NAMED — `builder.getStringAttr(sync->name_)` (`:203`), which prints as
/// `{dbgName = ".."}` (`hcc/samples/Matmul_L0/matmul_l0.mlir:27`).
pub fn construct_implicit_sync_operation(
    vals: &mut Values,
    ops: &mut Vec<DfirOp>,
    side: ImplicitSyncSide,
    handler: impl Fn(ImplicitSyncSide) -> Val,
    l0_memory: Val,
    tile_size: TileSize,
    elem: ElemType,
    name: &str,
) {
    // step-1: `mlir::Value dst_unit = this->getComponentHandler(dst_comp);`
    let dst_unit = handler(side.other());

    // step-3: `tile_size = ConstantIndexOp::create(builder, loc, tilesize_ss)`
    let size = constant_index(vals, ops, tile_size.0.get());

    // step-4: the start address and the view over the L0.
    let start = constant_index(vals, ops, 0);
    let view_ty = MemRef {
        shape: vec![1],
        elem,
    };
    let view = vals.mint();
    ops.push(DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
        result: view,
        from: l0_memory,
        start,
        layout: AffineMap::constants(1, &[0]),
        ty: view_ty.clone(),
    }));

    // step-5: `ImplicitSyncOnStreamingBufferOp::create(builder, loc, view, dst_unit, tile_size, name)`
    ops.push(DfirOp::Dataflow(dataflow::Op::ImplicitSync {
        view,
        dst: dst_unit,
        size,
        view_ty,
        dbg_name: Some(name.to_owned()),
    }));
}

#[cfg(test)]
mod unit_tests {
    use super::{
        ImplicitSyncSide, SyncEnd, TileSize, construct_implicit_sync_operation, construct_units,
    };
    use crate::bridges::superdsc_to_dataflow_ir::dsc_lowering::{Bound, Handlers, Retrieved};
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, arith, dataflow};
    use crate::islands::dataflow_ir::ty::{AffineMap, ElemType, MemRef};
    use crate::units::{Core, Corelet, DfirUnit, Residency, Row};

    /// ⛔ ROWS 1-7 CONTRIBUTE NOTHING AND THE L3 HALVES ASK ONCE WITH NO CORELET.
    ///
    /// A port that let row 3's signal through would sync against a handle no `get_unit` binds, and one
    /// that asked the L3 per corelet would emit `CORELETS_PER_CORE` handles for a core-wide unit.
    #[test]
    fn the_l0_rows_above_zero_are_skipped_and_an_l3_half_is_asked_once() {
        let core = Core::checked(0).expect("core 0");
        let cl0 = Corelet::checked(0).expect("corelet 0");
        let cl1 = Corelet::checked(1).expect("corelet 1");
        let handlers = Handlers {
            units: vec![(
                DfirUnit::L0lu,
                Bound::Unit {
                    handle: Val(30),
                    corelet: Some(cl0),
                },
            )],
            own_lrf: Val(1),
            pt_xrf: Val(2),
        };
        let mut vals = Values::default();

        let got = construct_units(
            &mut vals,
            &handlers,
            &[
                (
                    SyncEnd::L0luRow(Row::checked(0).expect("row 0")),
                    vec![(core, vec![cl0])],
                ),
                (
                    SyncEnd::L0luRow(Row::checked(3).expect("row 3")),
                    vec![(core, vec![cl0, cl1])],
                ),
                (SyncEnd::Unit(DfirUnit::L3su), vec![(core, vec![cl0, cl1])]),
            ],
        );

        assert_eq!(
            got,
            vec![
                // Row 0 asks for `L0LU`, which corelet 0 has already bound.
                Retrieved::Reused(Val(30)),
                // The L3 store half: ONE handle, core-wide, for a core naming two corelets.
                Retrieved::Created(dataflow::Op::GetUnit {
                    result: Val(0),
                    residency: Residency::CoreWide { core },
                    unit: DfirUnit::L3su,
                    num_folds: None,
                }),
            ]
        );
    }

    /// ⛔ THE DESTINATION IS THE OTHER SIDE, AND THE VIEW IS ONE ELEMENT AT LAYOUT `(d0) -> (0)`.
    ///
    /// Syncing a side against itself is a doorbell nobody rings, and unequal stage tile sizes are an
    /// epilogue this op cannot express — which is why [`TileSize::of`] answers [`None`] for them.
    #[test]
    fn the_store_side_syncs_the_load_side_over_a_one_element_doorbell() {
        let mut vals = Values::default();
        let mut ops: Vec<DfirOp> = Vec::new();
        let tile = TileSize::of(4, 4).expect("equal stage products");
        assert_eq!(TileSize::of(4, 5), None);
        assert_eq!(TileSize::of(0, 0), None);

        construct_implicit_sync_operation(
            &mut vals,
            &mut ops,
            ImplicitSyncSide::L0su,
            |side| match side {
                ImplicitSyncSide::L0su => Val(60),
                ImplicitSyncSide::L0luRow0 => Val(61),
            },
            Val(62),
            tile,
            ElemType::F16,
            "sync_implicit_L0",
        );

        let view_ty = MemRef {
            shape: vec![1],
            elem: ElemType::F16,
        };
        assert_eq!(
            ops,
            vec![
                DfirOp::Arith(arith::Op::Constant {
                    result: Val(0),
                    value: 4,
                }),
                DfirOp::Arith(arith::Op::Constant {
                    result: Val(1),
                    value: 0,
                }),
                DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                    result: Val(2),
                    from: Val(62),
                    start: Val(1),
                    layout: AffineMap::constants(1, &[0]),
                    ty: view_ty.clone(),
                }),
                DfirOp::Dataflow(dataflow::Op::ImplicitSync {
                    view: Val(2),
                    dst: Val(61),
                    size: Val(0),
                    view_ty,
                    dbg_name: Some("sync_implicit_L0".to_owned()),
                }),
            ]
        );
    }
}

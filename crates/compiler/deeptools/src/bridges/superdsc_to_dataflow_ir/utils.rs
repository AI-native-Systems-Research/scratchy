//! THE SHARED SHAPE AND FOLD HELPERS every lowering below stands on.
//!
//! 11 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e006_getTranslatorVersion` | 0 | 15 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:22` |
//! | `e007_createGetLocalUnitOp` | 0 | 13 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:44` |
//! | `e008_setPrecisionInUnitOp` | 0 | 12 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:143` |
//! | `e009_constructAVectorOfIndexType` | 0 | 5 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:711` |
//! | `e010_emitError` | 0 | 4 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:718` |
//! | `e044_createGetUnitOp` | 1 | 24 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:64` |
//! | `e072_createUniformizedGetUnitOp` | 2 | 42 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:95` |
//! | `e073_buildNeighborUnits` | 2 | 213 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:161` |
//! | `e083_buildUniformizedNeighborUnits` | 3 | 239 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:380` |
//! | `e084_initializeUnit` | 3 | 33 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:624` |
//! | `e093_initializeUniformizedUnit` | 4 | 47 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:663` |

use super::dsc_lowering::{Bound, Handles, Retrieved, query_over_handles};
use crate::arch::{Arch, IsaGen, Target};
use crate::islands::dataflow_ir::dialects::uniform::MappedTy;
use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, dataflow};
use crate::islands::dataflow_ir::ty::{GenericComp, ScalarTy};
use crate::islands::dataflow_ir::{ProgramUnit, Values};
use crate::units::{Core, Corelet, DfirUnit, NumFolds, Residency, Row, adjacent, residency_of};

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

/// WHAT ONE DSC OF A SUPERDSC IS, as far as the version choice is concerned — `computeOp_.empty()`
/// and `isDSC2()` (`DSC2ToDataflowIRUtils.hpp:26,30`) as the three states they enumerate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DscKind {
    /// `dsc.computeOp_.empty()` — a DSC with no compute op at all.
    NoComputeOp,
    /// `!dsc.isDSC2()` — a DSC1.0 schedule.
    Dsc1,
    /// A DSC2.0 schedule with a compute op.
    Dsc2,
}

/// WHICH TRANSLATOR A SUPERDSC NEEDS — the `int &version` out-parameter and the `LogicalResult`
/// read as ONE answer.
///
/// ⛔⛔ THE TWO V1 ARMS ARE INDISTINGUISHABLE AT THE ONLY CALLER, AND THEY ARE STILL BOTH HERE.
/// `runTranslator` returns `failure()` both on `failed(getTranslatorVersion(..))`
/// (`DSC2ToDataflowIR.cpp:546-548`) and from its own trailing `else` for any version but 3
/// (`:558-560`) — and both arms set `version = 1`. So the `success()`/`failure()` distinction reaches
/// nothing; collapsing it would still be dropping a value the reference computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranslatorVersion {
    /// `version = 3`, `success()` — every DSC is a DSC2.0 with a compute op.
    V3,
    /// `version = 1`, `success()` — a DSC1.0 was found.
    V1,
    /// `version = 1`, `failure()` — a DSC with no compute op was found first.
    V1NoComputeOp,
}

/// Replaces: e006_getTranslatorVersion
///
/// WHICH TRANSLATOR THE SUPERDSC NEEDS — `DSC2ToDataflowIRUtils.hpp:22`.
///
/// ⛔ FIRST TRIP WINS AND THE LOOP STOPS: a compute-less DSC standing AFTER a DSC1.0 answers
/// [`TranslatorVersion::V1`], not [`TranslatorVersion::V1NoComputeOp`] (`:25-33`).
///
/// ⭐ AND AN EMPTY `dscs_` IS [`TranslatorVersion::V3`] — the initial `version = 3` with nothing to
/// lower it.
#[must_use]
pub fn translator_version(dscs: &[DscKind]) -> TranslatorVersion {
    for dsc in dscs {
        match dsc {
            DscKind::NoComputeOp => return TranslatorVersion::V1NoComputeOp,
            DscKind::Dsc1 => return TranslatorVersion::V1,
            DscKind::Dsc2 => {}
        }
    }
    TranslatorVersion::V3
}

/// Replaces: e007_createGetLocalUnitOp
///
/// THE HANDLE FOR A REGISTER FILE OF A UNIT ALREADY HELD — `DSC2ToDataflowIRUtils.hpp:44`.
///
/// ⛔⛔ THE REUSE ARM IS UNREACHABLE FROM EVERY CALLER IN THE TREE, AND IT IS STILL EMITTED.
/// `component_to_handler_` never keys the three LRFREGs — `buildNeighborUnits` files their results
/// under the generic `LRFREG` (`:177-179`) — and `PTXRF`, `PTARF` and `L0_SCALE` are each asked for
/// exactly once per unit, after `initializeUnit`'s `component_to_handler_.clear()` (`:628`). So
/// `count(comp)` is 0 at every call site; deleting the else arm would be a port that could not state
/// which arm its callers take.
///
/// ⭐ THE RESULT TYPE IS `index`, NOT THE FILE'S (`:48`): it is a HANDLE to the file, the same shape a
/// `get_unit` binds, and [`dataflow::LocalUnit::spelling`] is the `name=` the reference reads out of
/// `senComponentsToString`.
pub fn create_get_local_unit_op(
    vals: &mut Values,
    held: Option<Val>,
    which: dataflow::LocalUnit,
    of: Val,
) -> Retrieved {
    match held {
        Some(handle) => Retrieved::Reused(handle),
        None => Retrieved::Created(dataflow::Op::GetLocalUnit {
            result: vals.mint(),
            of,
            which,
        }),
    }
}

/// Replaces: e008_setPrecisionInUnitOp
///
/// PUTS `precision =` ON A COMPUTE UNIT'S PROGRAM AND ON NO OTHER — `DSC2ToDataflowIRUtils.hpp:143`.
///
/// ⛔⛔ THE PARTIAL-MAP `find` AND THE `is_any_of(PT, PE, SFP)` ARE ONE TEST. The `find != end()`
/// guard (`:146-147`) covers a map that lacks `L0`, `CONSTANT` and `SFPRING`
/// (`arch_enums.cpp:124-211`), none of which is PT, PE or SFP — so [`DfirUnit::generic`], which is
/// total where that map throws, answers both halves.
///
/// ⛔ THE COMPONENT IS THE UNIT'S OWN. `unit_op.on.kind()` is what `comp` names, so taking them
/// separately would be a pair that could disagree about which unit is being written.
///
/// ⛔ AND `DT_CHECK_MSG(precision != "", "Invalid compute precision")` (`:150`) IS
/// [`dataflow::Precision`] — an empty spelling is not one of its arms.
pub fn set_precision_in_unit_op<A: Arch>(
    unit_op: &mut ProgramUnit<A>,
    precision: dataflow::Precision,
) {
    if matches!(
        unit_op.on.kind().generic(),
        GenericComp::Pt | GenericComp::Pe | GenericComp::Sfp
    ) {
        unit_op.precision = Some(precision);
    }
}

/// Replaces: e009_constructAVectorOfIndexType
///
/// APPENDS ONE `index` PER FOLD to a type list — `DSC2ToDataflowIRUtils.hpp:711`.
///
/// ⛔ CORRECTION — NEITHER CALL SITE RELIES ON THE APPEND. An earlier note here said `:72` and
/// `:679` each hand it a list that already holds the unit's own result type; they do not. Both
/// declare a FRESH `SmallVector<Type> get_unit_type;` on the line above the call (`:71-72` and
/// `:678-679`), and those are the only two callers in the tree. The `emplace_back` near the second
/// one is onto `units`, a different vector (`:687-688`). So the shape is the reference's and the
/// list is always empty on entry — a version that cleared it would be observationally identical.
///
/// ⛔ THE COUNT IS ALWAYS `num_folds_`, never a free integer — a folded `get_unit` returns one
/// address per fold, so the arity IS the fold count.
pub fn append_index_types(folds: NumFolds, types: &mut Vec<ScalarTy>) {
    for _ in 0..folds.0 {
        types.push(ScalarTy::Index);
    }
}

/// Replaces: e010_emitError
///
/// THE TRANSLATOR'S DIAGNOSTIC TEXT — `DSC2ToDataflowIRUtils.hpp:718`.
///
/// ⛔ THE PREFIX IS THE WHOLE FUNCTION. `module_op_->emitError` is MLIR's sink and this crate has
/// none: a lowering that reported instead of emitting would be the runtime refusal
/// `dfir_never_runtime_refuses.rs` freezes at zero. The text is handed back so the caller that
/// cannot proceed names it in its own `todo!`.
#[must_use]
pub fn error_diagnostic(message: &str) -> String {
    format!("[DSC2.0 to Dataflow IR]: {message}")
}

/// Replaces: e044_createGetUnitOp
///
/// THE HANDLE FOR A UNIT OF A NAMED CORE AND CORELET, created once and reused after —
/// `DSC2ToDataflowIRUtils.hpp:64`.
///
/// ⛔⛔ THE MAP IT TESTS AND THE MAP IT WRITES ARE TWO DIFFERENT MAPS. It asks
/// `component_to_handler_.count(comp)` (`:69`) and stores into
/// `unit_to_value_map_[core_id][corelet_id][comp]` (`:82`), so a second call for the SAME triple does
/// not hit the entry the first one just made. That is not an oversight to repair: it is what makes
/// [`e072`](super::utils)'s cores×corelets loop emit one op per handle (`:101-119`) instead of
/// collapsing the whole set onto one, and a cache keyed by the triple would break that function's own
/// `DT_CHECK(get_unit_ops.size() == units.size())` at `:122`. So the reuse this takes is the
/// COMPONENT'S handler and nothing finer — one [`Option`], not a lookup.
///
/// ⛔ AND THAT IS WHY THERE ARE TWO ARMS AND NOT THREE. The reference's else branch is
/// `component_to_handler_[comp].getDefiningOp<dataflow::GetUnitOp>()` (`:86`), which yields NULL for a
/// handler whose defining op is something else — a `uniform.query_map` result, say. Only
/// `initializeUnit` (`:648-649`) and `buildNeighborUnits` (`:167-175`) ever write that map for a
/// component this can be called with, and both write `get_unit` results; the uniformized path stores
/// its query result under a guard that stops this function being reached at all (`:100`). A third
/// variant would be a state no caller can produce.
///
/// ⭐⭐ THE `index` LIST **IS** THE `num_folds` FIELD. `constructAVectorOfIndexType(num_folds_, ..)`
/// (`:72`, entry 009) sizes the op's RESULT GROUP, and [`dataflow::Op::GetUnit`] prints that group off
/// `num_folds` — `%28:11 = dataflow.get_unit {.. num_folds = 11 : i32 ..} : index, index, ..`. So the
/// type vector and the attribute are one fact here, and `getResult(i)` is `%28#i`.
///
/// ⛔ THE `-1` IS AN ABSENT ATTRIBUTE, NOT A CORELET ZERO. `if (corelet_id != -1)` gates the
/// `corelet` attribute (`:80-81`) and the default argument is `-1` (`DSC2ToDataflowIR.hpp:122`); the
/// island's [`Residency`] is where that gate lives, since `Residency::Scratchpad` prints `core` with
/// no `corelet` where `Residency::CoreWide` prints `corelet = 0`. Its one `-1` caller is the L3 pair
/// (`:104`), which the island's own golden already writes as a `Scratchpad` `l3lu`.
///
/// ⛔ AND THE NAME IS NOT `initializeUnit`'S. This one writes `name = type =
/// senComponentsToString.at(comp)` (`:75-77`) where `initializeUnit` writes `type + "-CL" +
/// corelet_id` (`:632-633`) for the same op kind; the island derives both from the residency and the
/// unit, and [`dataflow::Op::GetUnit`]'s own note records why.
pub fn create_get_unit_op(
    vals: &mut Values,
    held: Option<Val>,
    unit: DfirUnit,
    residency: Residency,
    folds: NumFolds,
) -> Retrieved {
    match held {
        Some(handle) => Retrieved::Reused(handle),
        None => Retrieved::Created(dataflow::Op::GetUnit {
            result: vals.mint(),
            residency,
            unit,
            num_folds: Some(folds),
        }),
    }
}

/// Replaces: e072_createUniformizedGetUnitOp
///
/// **072/110** `DSC2ToDataflowIRUtils.hpp:95` — ONE `get_unit` per `(core, corelet)`, or per CORE for
/// the L3 halves so both LXLUs point at the same L3, every fold of it paired with the key that asks
/// for it, mapped and queried by `program_unit_iterator_`.
///
/// ⛔ A FOLD RESULT CANNOT BE NAMED: the reference pairs `get_unit.getResult(i)` and [`Val`] spells
/// `%28`, not `%28#3`, so each of a group's folds is paired with the group's own value — exact at
/// `NumFolds::ONE` and an island gap above it. The size `DT_CHECK` is structural: `keys` builds both.
pub fn create_uniformized_get_unit_op(
    vals: &mut Values,
    ops: &mut Vec<DfirOp>,
    held: Option<Val>,
    keys: &Handles,
    comp: DfirUnit,
    folds: NumFolds,
) -> Val {
    // `if (component_to_handler_.count(comp) == 0)` — a held component answers with no op at all.
    if let Some(handle) = held {
        return handle;
    }

    // `is_any_of(comp, L3LU, L3SU)`, whose arm creates the op OUTSIDE the corelet loop.
    let core_wide = matches!(comp, DfirUnit::L3lu | DfirUnit::L3su);
    let mut pairs = Vec::new();
    let mut created: Option<(Core, Val)> = None;
    for group in keys.corelet_groups() {
        let Some(first) = group.first() else { continue };
        let value = match created {
            Some((core, value)) if core_wide && core == first.core => value,
            _ => {
                let residency = residency_of(comp, first.core, first.corelet);
                let value = create_get_unit_op(vals, None, comp, residency, folds).bind(ops);
                created = Some((first.core, value));
                value
            }
        };
        // `for (corelet_ids) for (i < num_folds_) units.emplace_back(get_unit.getResult(i))`.
        for handle in group {
            pairs.push((handle.unit, value));
        }
    }

    query_over_handles(vals, ops, keys.iterator(), pairs, MappedTy::Index)
}

/// `PTSOUTH`, `PTNORTH` and `PTWEST` — the RELATIVE keys a PT row's arm binds beside the absolute
/// ones (`DSC2ToDataflowIRUtils.hpp:164-172`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtDirection {
    /// The next row, or the PE at the last one.
    South,
    /// The row above, or the SFP at row 0.
    North,
    /// The L0 load unit — bound under THIS key only.
    West,
}

/// THE REGISTER FILES AND SCALE REGION ONE UNIT'S OWN PREAMBLE BINDS with `get_local_unit`, which is
/// a different set per unit kind and empty for most of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnFiles {
    /// A PT row: `PT_LRFREG` as `LRFREG`, `PTXRF`, and `L0_SCALE` from SEN1P5.
    PtRow {
        /// `LRFREG`.
        lrf: Val,
        /// `PTXRF`.
        xrf: Val,
        /// `L0_SCALE`, from SEN1P5 only.
        l0_scale: Option<Val>,
    },
    /// The PE: `PE_LRFREG` as `LRFREG`.
    Pe {
        /// `LRFREG`.
        lrf: Val,
    },
    /// The SFP: `SFP_LRFREG` as `LRFREG`.
    Sfp {
        /// `LRFREG`.
        lrf: Val,
    },
    /// The L0 store unit: no register file, and `L0_SCALE` from SEN1P5.
    L0su {
        /// `L0_SCALE`, from SEN1P5 only.
        l0_scale: Option<Val>,
    },
    /// Every other arm, the reference's empty trailing `else` among them.
    None,
}

/// WHAT ONE UNIT'S PREAMBLE BINDS — the ops it emits and the `component_to_handler_` entries they
/// fill.
#[derive(Debug, Clone, PartialEq)]
pub struct Neighbourhood {
    /// The `get_unit` and `get_local_unit` ops, in the reference's binding order.
    pub ops: Vec<DfirOp>,
    /// The absolute keys — `component_to_handler_[<unit>]`.
    pub units: Vec<(DfirUnit, Bound)>,
    /// The relative keys, whose handles are all in `units` too EXCEPT [`PtDirection::West`]'s.
    pub directions: Vec<(PtDirection, Val)>,
    /// This unit's own register files.
    pub own: OwnFiles,
}

/// A PT row by index, or `fallback` where this arch has no such row.
fn pt_row(index: u32, fallback: DfirUnit) -> DfirUnit {
    Row::checked(index).map_or(fallback, DfirUnit::PtRow)
}

/// `createGetUnitOp(builder, which, core_id, corelet_id).getResult(0)`, bound under `which`.
fn bind_unit(
    vals: &mut Values,
    n: &mut Neighbourhood,
    which: DfirUnit,
    core: Core,
    corelet: Corelet,
) -> Val {
    let residency = residency_of(which, core, corelet);
    let handle = create_get_unit_op(vals, None, which, residency, NumFolds::ONE).bind(&mut n.ops);
    let corelet = match residency {
        Residency::Corelet { corelet, .. } => Some(corelet),
        Residency::Global | Residency::Scratchpad { .. } | Residency::CoreWide { .. } => None,
    };
    n.units.push((which, Bound::Unit { handle, corelet }));
    handle
}

/// `createGetLocalUnitOp(builder, which, unit_op)`, which fills no absolute key.
fn bind_local(
    vals: &mut Values,
    n: &mut Neighbourhood,
    which: dataflow::LocalUnit,
    of: Val,
) -> Val {
    create_get_local_unit_op(vals, None, which, of).bind(&mut n.ops)
}

/// `if (dsc_global_.sysDef.coreArch >= SEN1P5_ISA) component_to_handler_[L0_SCALE] = ..`.
fn bind_l0_scale(vals: &mut Values, n: &mut Neighbourhood, of: Val) -> Option<Val> {
    matches!(Target::GEN, IsaGen::Sen1p5)
        .then(|| bind_local(vals, n, dataflow::LocalUnit::L0Scale, of))
}

/// Replaces: e073_buildNeighborUnits
///
/// **073/110** `DSC2ToDataflowIRUtils.hpp:161` — one unit's preamble: a `get_unit` per neighbour in
/// the reference's own order, the PT's relative keys, and its own register files. `own` is the handle
/// `initializeUnit` bound for `of` before the call (`:628-630`), which the first retrieval reuses.
///
/// ⛔ `PTWEST` IS THE L0 LOAD UNIT'S ONLY KEY on a PT row, so `units` does not carry it — a body
/// naming `L0LU` there creates a second `get_unit`, exactly as in the reference.
/// ⛔ THE L3 HALVES ARE ASKED FOR WITH A CORELET (`:377-386`) and bound core-wide by [`residency_of`].
/// ⛔ [`crate::units::local_units`] OMITS THE L0 STORE UNIT'S `L0_SCALE` (`:301-303`); this binds it.
pub fn build_neighbor_units(
    vals: &mut Values,
    of: DfirUnit,
    own: Val,
    core: Core,
    corelet: Corelet,
) -> Neighbourhood {
    let mut n = Neighbourhood {
        ops: Vec::new(),
        units: Vec::new(),
        directions: Vec::new(),
        own: OwnFiles::None,
    };
    match of {
        // `:164-231` — eight arms that are one rule: south, north, the LX load unit at row 0, west,
        // then the row's own files.
        DfirUnit::PtRow(row) => {
            let south = bind_unit(vals, &mut n, adjacent(row.south()), core, corelet);
            n.directions.push((PtDirection::South, south));
            let north = bind_unit(vals, &mut n, adjacent(row.north()), core, corelet);
            n.directions.push((PtDirection::North, north));
            if row.get() == 0 {
                bind_unit(vals, &mut n, DfirUnit::Lxlu, core, corelet);
            }
            let residency = residency_of(DfirUnit::L0lu, core, corelet);
            let west = create_get_unit_op(vals, None, DfirUnit::L0lu, residency, NumFolds::ONE)
                .bind(&mut n.ops);
            n.directions.push((PtDirection::West, west));
            let lrf = bind_local(vals, &mut n, dataflow::LocalUnit::PtLrf, own);
            let xrf = bind_local(vals, &mut n, dataflow::LocalUnit::PtXrf, own);
            let l0_scale = bind_l0_scale(vals, &mut n, own);
            n.own = OwnFiles::PtRow { lrf, xrf, l0_scale };
        }
        // `:285-290` — `L0LUROW0`.
        DfirUnit::L0lu => {
            for which in [DfirUnit::L0su, pt_row(0, DfirUnit::Sfp), DfirUnit::L0] {
                bind_unit(vals, &mut n, which, core, corelet);
            }
        }
        // `:291-304`.
        DfirUnit::L0su => {
            for which in [DfirUnit::Sfp, DfirUnit::L0lu, DfirUnit::L0, DfirUnit::Lxlu] {
                bind_unit(vals, &mut n, which, core, corelet);
            }
            n.own = OwnFiles::L0su {
                l0_scale: bind_l0_scale(vals, &mut n, own),
            };
        }
        // `:369-386` — the hub: EIGHT, both L3 halves and PT row 0 among them.
        DfirUnit::Lxlu => {
            for which in [
                DfirUnit::Lxsu,
                DfirUnit::Sfp,
                DfirUnit::Lx,
                DfirUnit::Pe,
                DfirUnit::L3lu,
                DfirUnit::L3su,
                pt_row(0, DfirUnit::Sfp),
                DfirUnit::L0su,
            ] {
                bind_unit(vals, &mut n, which, core, corelet);
            }
        }
        // `:387-398`.
        DfirUnit::Lxsu => {
            for which in [
                DfirUnit::Lxlu,
                DfirUnit::Sfp,
                DfirUnit::Lx,
                DfirUnit::Pe,
                DfirUnit::L3lu,
                DfirUnit::L3su,
            ] {
                bind_unit(vals, &mut n, which, core, corelet);
            }
        }
        // `:399-410` — the PE, whose north is the LAST row, and whose `LRFREG` is bound in the
        // MIDDLE of the sequence.
        DfirUnit::Pe => {
            for which in [
                DfirUnit::Lxlu,
                DfirUnit::Lxsu,
                pt_row(Target::PT_ROWS - 1, DfirUnit::Sfp),
            ] {
                bind_unit(vals, &mut n, which, core, corelet);
            }
            n.own = OwnFiles::Pe {
                lrf: bind_local(vals, &mut n, dataflow::LocalUnit::PeLrf, own),
            };
            for which in [DfirUnit::Sfp, DfirUnit::Constant, DfirUnit::PeState] {
                bind_unit(vals, &mut n, which, core, corelet);
            }
        }
        // `:354-368` — the SFP, `LRFREG` in the middle again.
        DfirUnit::Sfp => {
            for which in [
                DfirUnit::Lxlu,
                DfirUnit::Lxsu,
                DfirUnit::L0su,
                pt_row(0, DfirUnit::Pe),
            ] {
                bind_unit(vals, &mut n, which, core, corelet);
            }
            n.own = OwnFiles::Sfp {
                lrf: bind_local(vals, &mut n, dataflow::LocalUnit::SfpLrf, own),
            };
            for which in [DfirUnit::Pe, DfirUnit::Constant, DfirUnit::SfpState] {
                bind_unit(vals, &mut n, which, core, corelet);
            }
        }
        // `} else {}` — the memories and sources bind nothing.
        DfirUnit::Lx
        | DfirUnit::L0
        | DfirUnit::Hbm
        | DfirUnit::L3lu
        | DfirUnit::L3su
        | DfirUnit::Constant
        | DfirUnit::SfpState
        | DfirUnit::PeState
        | DfirUnit::SfpRing
        | DfirUnit::LxVirtualIbr
        | DfirUnit::CrossPtnLink => {}
    }
    n
}

/// `createUniformizedGetUnitOp(builder, which, iterator, ..)`, bound under `which`.
fn bind_uniformized(
    vals: &mut Values,
    n: &mut Neighbourhood,
    which: DfirUnit,
    keys: &Handles,
    folds: NumFolds,
) -> Val {
    let handle = create_uniformized_get_unit_op(vals, &mut n.ops, None, keys, which, folds);
    n.units.push((which, Bound::Local(handle)));
    handle
}

/// Replaces: e083_buildUniformizedNeighborUnits
///
/// **083/110** `DSC2ToDataflowIRUtils.hpp:380` — [`build_neighbor_units`]'s neighbour list, unit for
/// unit and in the same order, with every absolute key answered by a `uniform.query_map` over the
/// whole `(core, corelet, fold)` set instead of one core's `dataflow.get_unit`.
///
/// ⛔ THE OWN FILES STAY LOCAL AND UNIFORMIZED-FREE — `createGetLocalUnitOp(builder, PT_LRFREG,
/// unit_op)` (`:387-390`) takes the region ITERATOR as its unit, because inside a uniformized region
/// there is no single handle to ask.
/// ⛔ A QUERY RESULT IS [`Bound::Local`], NOT [`Bound::Unit`]: its defining op is a
/// `uniform.query_map`, which is exactly the handler [`create_get_unit_op`] cannot reuse (`:86`).
pub fn build_uniformized_neighbor_units(
    vals: &mut Values,
    of: DfirUnit,
    keys: &Handles,
    folds: NumFolds,
) -> Neighbourhood {
    // `auto unit_op = program_unit_op_iterator;` (`:385`) — the locals hang off the iterator.
    let own = keys.iterator();
    let mut n = Neighbourhood {
        ops: Vec::new(),
        units: Vec::new(),
        directions: Vec::new(),
        own: OwnFiles::None,
    };
    match of {
        // `:386-471` — the eight PT rows, south then north then the row-0 LX load unit then west.
        DfirUnit::PtRow(row) => {
            let south = bind_uniformized(vals, &mut n, adjacent(row.south()), keys, folds);
            n.directions.push((PtDirection::South, south));
            let north = bind_uniformized(vals, &mut n, adjacent(row.north()), keys, folds);
            n.directions.push((PtDirection::North, north));
            if row.get() == 0 {
                bind_uniformized(vals, &mut n, DfirUnit::Lxlu, keys, folds);
            }
            let west =
                create_uniformized_get_unit_op(vals, &mut n.ops, None, keys, DfirUnit::L0lu, folds);
            n.directions.push((PtDirection::West, west));
            let lrf = bind_local(vals, &mut n, dataflow::LocalUnit::PtLrf, own);
            let xrf = bind_local(vals, &mut n, dataflow::LocalUnit::PtXrf, own);
            let l0_scale = bind_l0_scale(vals, &mut n, own);
            n.own = OwnFiles::PtRow { lrf, xrf, l0_scale };
        }
        // `:472-480` — `L0LUROW0`.
        DfirUnit::L0lu => {
            for which in [DfirUnit::L0su, pt_row(0, DfirUnit::Sfp), DfirUnit::L0] {
                bind_uniformized(vals, &mut n, which, keys, folds);
            }
        }
        // `:481-495`.
        DfirUnit::L0su => {
            for which in [DfirUnit::Sfp, DfirUnit::L0lu, DfirUnit::L0, DfirUnit::Lxlu] {
                bind_uniformized(vals, &mut n, which, keys, folds);
            }
            n.own = OwnFiles::L0su {
                l0_scale: bind_l0_scale(vals, &mut n, own),
            };
        }
        // `:496-518` — the hub, both L3 halves among them.
        DfirUnit::Lxlu => {
            for which in [
                DfirUnit::Lxsu,
                DfirUnit::Sfp,
                DfirUnit::Lx,
                DfirUnit::Pe,
                DfirUnit::L3lu,
                DfirUnit::L3su,
                pt_row(0, DfirUnit::Sfp),
                DfirUnit::L0su,
            ] {
                bind_uniformized(vals, &mut n, which, keys, folds);
            }
        }
        // `:519-534`.
        DfirUnit::Lxsu => {
            for which in [
                DfirUnit::Lxlu,
                DfirUnit::Sfp,
                DfirUnit::Lx,
                DfirUnit::Pe,
                DfirUnit::L3lu,
                DfirUnit::L3su,
            ] {
                bind_uniformized(vals, &mut n, which, keys, folds);
            }
        }
        // `:535-556` — the PE, whose `LRFREG` lands in the MIDDLE of the sequence.
        DfirUnit::Pe => {
            for which in [
                DfirUnit::Lxlu,
                DfirUnit::Lxsu,
                pt_row(Target::PT_ROWS - 1, DfirUnit::Sfp),
            ] {
                bind_uniformized(vals, &mut n, which, keys, folds);
            }
            n.own = OwnFiles::Pe {
                lrf: bind_local(vals, &mut n, dataflow::LocalUnit::PeLrf, own),
            };
            for which in [DfirUnit::Sfp, DfirUnit::Constant, DfirUnit::PeState] {
                bind_uniformized(vals, &mut n, which, keys, folds);
            }
        }
        // `:557-577` — the SFP, `LRFREG` in the middle again.
        DfirUnit::Sfp => {
            for which in [
                DfirUnit::Lxlu,
                DfirUnit::Lxsu,
                DfirUnit::L0su,
                pt_row(0, DfirUnit::Pe),
            ] {
                bind_uniformized(vals, &mut n, which, keys, folds);
            }
            n.own = OwnFiles::Sfp {
                lrf: bind_local(vals, &mut n, dataflow::LocalUnit::SfpLrf, own),
            };
            for which in [DfirUnit::Pe, DfirUnit::Constant, DfirUnit::SfpState] {
                bind_uniformized(vals, &mut n, which, keys, folds);
            }
        }
        // `} else {}` (`:578`) — the memories and sources bind nothing.
        DfirUnit::Lx
        | DfirUnit::L0
        | DfirUnit::Hbm
        | DfirUnit::L3lu
        | DfirUnit::L3su
        | DfirUnit::Constant
        | DfirUnit::SfpState
        | DfirUnit::PeState
        | DfirUnit::SfpRing
        | DfirUnit::LxVirtualIbr
        | DfirUnit::CrossPtnLink => {}
    }
    n
}

/// ONE PROGRAM UNIT AND EVERYTHING ITS PREAMBLE BOUND — `initializeUnit`'s return, which is the
/// `program_unit` op plus the `component_to_handler_` state the caller lowers a body against.
#[derive(Debug, Clone, PartialEq)]
pub struct InitializedUnit {
    /// The `dataflow.get_unit` for this unit, then its `dataflow.program_unit`.
    pub ops: Vec<DfirOp>,
    /// The unit's own handle — `component_to_handler_[comp]`.
    pub own: Val,
    /// What that preamble bound.
    pub neighbours: Neighbourhood,
}

/// Replaces: e084_initializeUnit
///
/// **084/110** `DSC2ToDataflowIRUtils.hpp:624` — the `get_unit`/`program_unit` pair for one
/// `(core, corelet, unit)`, `component_to_handler_` CLEARED and re-seeded with this unit's own
/// handle, then [`build_neighbor_units`] at the top of the region.
///
/// ⛔ [`NumFolds::ONE`] IS THE TYPE, NOT A CHECK: `DT_CHECK(num_folds_ == 1 && "this initalization is
/// for only non-uniform and non-fold mode")` (`:645-646`) is the signature here, and the `num_folds`
/// attribute it then writes is that same 1.
/// ⛔ THE NAME IS `type + "-CL" + corelet_id` (`:632-633`), which the island derives from the
/// residency — see [`dataflow::Op::GetUnit`].
#[must_use]
pub fn initialize_unit(
    vals: &mut Values,
    core: Core,
    corelet: Corelet,
    comp: DfirUnit,
) -> InitializedUnit {
    let mut ops = Vec::new();
    let own = create_get_unit_op(
        vals,
        None,
        comp,
        Residency::Corelet { core, corelet },
        NumFolds::ONE,
    )
    .bind(&mut ops);
    let neighbours = build_neighbor_units(vals, comp, own, core, corelet);
    ops.push(DfirOp::Dataflow(dataflow::Op::ProgramUnit {
        units: vec![own],
        iter_arg: None,
        precision: None,
        body: neighbours.ops.clone(),
    }));
    InitializedUnit {
        ops,
        own,
        neighbours,
    }
}

/// ONE UNIFORMIZED PROGRAM UNIT AND EVERYTHING ITS PREAMBLE BOUND — [`InitializedUnit`]'s uniformized
/// twin, whose own handle is the region's `iter_arg` and whose `(core, corelet)` table IS
/// `unit_to_value_map_[core][corelet][comp]`.
#[derive(Debug, Clone, PartialEq)]
pub struct InitializedUniformizedUnit {
    /// Every `(core, corelet)`'s `dataflow.get_unit`, then the `dataflow.program_unit` over them.
    pub ops: Vec<DfirOp>,
    /// `program_unit_iterator` — `component_to_handler_[comp]` after the `clear()`.
    pub iterator: Val,
    /// `unit_to_value_map_` and `*units_involved_` as one table, keyed on that iterator.
    pub handles: Handles,
    /// What the preamble inside the region bound.
    pub neighbours: Neighbourhood,
}

/// Replaces: e093_initializeUniformizedUnit
///
/// **093/110** `DSC2ToDataflowIRUtils.hpp:663` — a `get_unit` per `(core, corelet)`, the
/// `program_unit` over EVERY FOLD of all of them, then [`build_uniformized_neighbor_units`] inside.
///
/// ⛔ THE UNIT'S OWN HANDLE IS THE REGION ARGUMENT: `component_to_handler_.clear()` then
/// `[comp] = program_unit_iterator = unit_op.getRegion().getArguments().front()`, so every neighbour
/// query in the body is keyed on `iter_arg` and no `get_unit` names this unit.
/// ⛔ AND EVERY COMPONENT IS NAMED `type + "-CL" + corelet_id` HERE, L3 included — unlike
/// [`create_uniformized_get_unit_op`], whose L3 halves are ONE core-wide op.
/// ⚠️ ONE `Val` PER GROUP, REPEATED PER FOLD — `getResult(i)` is unnameable, as in entry 072.
pub fn initialize_uniformized_unit(
    vals: &mut Values,
    comp: DfirUnit,
    cores: &[Core],
    corelets: &[Corelet],
    folds: NumFolds,
    units: &mut Vec<Val>,
) -> InitializedUniformizedUnit {
    let mut ops = Vec::new();
    let appended_at = units.len();
    for &core in cores {
        for &corelet in corelets {
            let unit = create_get_unit_op(
                vals,
                None,
                comp,
                Residency::Corelet { core, corelet },
                folds,
            )
            .bind(&mut ops);
            for _ in 0..folds.0 {
                units.push(unit);
            }
        }
    }
    let iterator = vals.mint();
    // ⭐ THE TWO WALKS ARE ONE WALK: this tail was filled by the same `for (core) for (corelet) for
    // (fold)` nesting [`Handles::new`] walks, so it holds exactly one entry per handle asked for.
    let mut walk = units[appended_at..].iter().copied();
    let handles = Handles::new(cores, corelets, folds, iterator, |_, _, _| {
        walk.next().unwrap_or(iterator)
    });
    let neighbours = build_uniformized_neighbor_units(vals, comp, &handles, folds);
    ops.push(DfirOp::Dataflow(dataflow::Op::ProgramUnit {
        units: units.clone(),
        iter_arg: Some(iterator),
        precision: None,
        body: neighbours.ops.clone(),
    }));
    InitializedUniformizedUnit {
        ops,
        iterator,
        handles,
        neighbours,
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        DscKind, InitializedUniformizedUnit, InitializedUnit, Neighbourhood, OwnFiles, PtDirection,
        TranslatorVersion, append_index_types, build_neighbor_units,
        build_uniformized_neighbor_units, create_get_local_unit_op, create_get_unit_op,
        create_uniformized_get_unit_op, error_diagnostic, initialize_uniformized_unit,
        initialize_unit, set_precision_in_unit_op, translator_version,
    };
    use crate::arch::{Arch, Dd2, IsaGen, Target};
    use crate::bridges::superdsc_to_dataflow_ir::dsc_lowering::{Bound, Handles, Retrieved};
    use crate::islands::dataflow_ir::dialects::uniform::{self, MappedTy};
    use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, dataflow};
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{ProgramUnit, Units, Values};
    use crate::units::{Core, Corelet, DfirUnit, NumFolds, Residency, Row};

    /// 🎯 072/110 — ⭐ ONE `get_unit` PER CORELET, then the mapping and the query the whole set is
    /// read through; and the L3 arm's ONE op named twice, which is the reference's own comment.
    #[test]
    fn each_corelet_gets_its_own_unit_and_the_l3_pair_share_one() {
        let mut vals = Values::default();
        let (first, second, iterator) = (vals.mint(), vals.mint(), vals.mint());
        let core = Core::checked(0).expect("core 0");
        let corelets = [
            Corelet::checked(0).expect("corelet 0"),
            Corelet::checked(1).expect("corelet 1"),
        ];
        let keys = Handles::new(
            &[core],
            &corelets,
            NumFolds::ONE,
            iterator,
            |_, corelet, _| {
                if corelet == corelets[0] {
                    first
                } else {
                    second
                }
            },
        );

        let mut ops = Vec::new();
        let answer = create_uniformized_get_unit_op(
            &mut vals,
            &mut ops,
            None,
            &keys,
            DfirUnit::Lxlu,
            NumFolds::ONE,
        );
        assert_eq!(answer, Val(6));
        assert_eq!(
            ops,
            vec![
                DfirOp::Dataflow(dataflow::Op::GetUnit {
                    result: Val(3),
                    residency: Residency::Corelet {
                        core,
                        corelet: corelets[0]
                    },
                    unit: DfirUnit::Lxlu,
                    num_folds: Some(NumFolds::ONE),
                }),
                DfirOp::Dataflow(dataflow::Op::GetUnit {
                    result: Val(4),
                    residency: Residency::Corelet {
                        core,
                        corelet: corelets[1]
                    },
                    unit: DfirUnit::Lxlu,
                    num_folds: Some(NumFolds::ONE),
                }),
                DfirOp::Uniform(uniform::Op::DefImmutableMapping {
                    result: Val(5),
                    pairs: vec![(first, Val(3)), (second, Val(4))],
                    values_ty: MappedTy::Index,
                }),
                DfirOp::Uniform(uniform::Op::QueryMap {
                    result: Val(6),
                    map: Val(5),
                    key: iterator,
                    ty: MappedTy::Index,
                }),
            ]
        );

        // ⛔ THE L3 HALF IS CREATED ONCE FOR THE CORE and paired with BOTH corelets' keys.
        let mut ops = Vec::new();
        create_uniformized_get_unit_op(
            &mut vals,
            &mut ops,
            None,
            &keys,
            DfirUnit::L3lu,
            NumFolds::ONE,
        );
        assert_eq!(
            ops[..1],
            [DfirOp::Dataflow(dataflow::Op::GetUnit {
                result: Val(7),
                residency: Residency::CoreWide { core },
                unit: DfirUnit::L3lu,
                num_folds: Some(NumFolds::ONE),
            })]
        );
        assert!(matches!(
            &ops[1],
            DfirOp::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })
                if pairs == &vec![(first, Val(7)), (second, Val(7))]
        ));
        // ⛔ AND A HELD COMPONENT EMITS NOTHING AT ALL.
        let mut ops = Vec::new();
        assert_eq!(
            create_uniformized_get_unit_op(
                &mut vals,
                &mut ops,
                Some(first),
                &keys,
                DfirUnit::L3lu,
                NumFolds::ONE
            ),
            first
        );
        assert!(ops.is_empty());
    }

    /// 🎯 073/110 — ⛔ THE L0 LOAD UNIT IS BOUND UNDER `PTWEST` AND NOWHERE ELSE, so `units` holds
    /// three entries for a row that emits four `get_unit`s.
    #[test]
    fn a_pt_row_binds_south_north_and_west_plus_its_own_files() {
        let mut vals = Values::default();
        let own = vals.mint();
        let core = Core::checked(0).expect("core 0");
        let corelet = Corelet::checked(1).expect("corelet 1");
        let row0 = Row::checked(0).expect("row 0");
        let row1 = Row::checked(1).expect("row 1");
        let Neighbourhood {
            ops,
            units,
            directions,
            own: files,
        } = build_neighbor_units(&mut vals, DfirUnit::PtRow(row0), own, core, corelet);
        let bound = |handle| Bound::Unit {
            handle,
            corelet: Some(corelet),
        };
        assert_eq!(
            units,
            vec![
                (DfirUnit::PtRow(row1), bound(Val(1))),
                (DfirUnit::Sfp, bound(Val(2))),
                (DfirUnit::Lxlu, bound(Val(3))),
            ]
        );
        assert_eq!(
            directions,
            vec![
                (PtDirection::South, Val(1)),
                (PtDirection::North, Val(2)),
                (PtDirection::West, Val(4)),
            ]
        );
        let scaled = matches!(Target::GEN, IsaGen::Sen1p5);
        assert_eq!(
            files,
            OwnFiles::PtRow {
                lrf: Val(5),
                xrf: Val(6),
                l0_scale: scaled.then_some(Val(7)),
            }
        );
        assert_eq!(ops.len(), if scaled { 7 } else { 6 });
        assert_eq!(
            ops[5],
            DfirOp::Dataflow(dataflow::Op::GetLocalUnit {
                result: Val(6),
                of: own,
                which: dataflow::LocalUnit::PtXrf,
            })
        );
    }

    /// 🎯 083/110 — ⛔ EVERY NEIGHBOUR IS A `uniform.query_map` RESULT AND SO [`Bound::Local`], while
    /// the unit's own file stays a `get_local_unit` off the region ITERATOR.
    #[test]
    fn a_uniformized_preamble_queries_its_neighbours_and_keeps_its_own_file_local() {
        let mut vals = Values::default();
        let (key, iterator) = (vals.mint(), vals.mint());
        let core = Core::checked(0).expect("core 0");
        let corelet = Corelet::checked(0).expect("corelet 0");
        let keys = Handles::new(&[core], &[corelet], NumFolds::ONE, iterator, |_, _, _| key);

        let Neighbourhood {
            ops,
            units,
            directions,
            own,
        } = build_uniformized_neighbor_units(&mut vals, DfirUnit::L0su, &keys, NumFolds::ONE);

        // ⭐ THE SAME FOUR NEIGHBOURS ENTRY 073 BINDS FOR `L0SU`, in the same order.
        assert_eq!(
            units.iter().map(|(which, _)| *which).collect::<Vec<_>>(),
            vec![DfirUnit::Sfp, DfirUnit::L0lu, DfirUnit::L0, DfirUnit::Lxlu]
        );
        let queried: Vec<Bound> = ops
            .iter()
            .filter_map(|op| match op {
                DfirOp::Uniform(uniform::Op::QueryMap { result, key, .. }) if *key == iterator => {
                    Some(Bound::Local(*result))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            units.iter().map(|(_, bound)| *bound).collect::<Vec<_>>(),
            queried
        );
        // ⭐ NO RELATIVE KEYS OUTSIDE A PT ROW.
        assert!(directions.is_empty());

        match (matches!(Target::GEN, IsaGen::Sen1p5), ops.last()) {
            (true, Some(DfirOp::Dataflow(dataflow::Op::GetLocalUnit { result, of, which }))) => {
                assert_eq!((*of, *which), (iterator, dataflow::LocalUnit::L0Scale));
                assert_eq!(
                    own,
                    OwnFiles::L0su {
                        l0_scale: Some(*result)
                    }
                );
            }
            (false, _) => assert_eq!(own, OwnFiles::L0su { l0_scale: None }),
            (true, other) => panic!("expected a get_local_unit last, found {other:?}"),
        }
    }

    /// 🎯 084/110 — ⭐ THE PAIR AND NOTHING ELSE: the unit's own `get_unit` at `num_folds = 1`, then a
    /// `program_unit` whose BODY is the whole preamble, so no neighbour handle escapes the region.
    #[test]
    fn one_unit_is_a_get_unit_then_a_program_unit_over_its_preamble() {
        let mut vals = Values::default();
        let core = Core::checked(0).expect("core 0");
        let corelet = Corelet::checked(1).expect("corelet 1");
        let InitializedUnit {
            ops,
            own,
            neighbours,
        } = initialize_unit(&mut vals, core, corelet, DfirUnit::L0lu);
        assert_eq!(
            ops,
            vec![
                DfirOp::Dataflow(dataflow::Op::GetUnit {
                    result: own,
                    residency: Residency::Corelet { core, corelet },
                    unit: DfirUnit::L0lu,
                    num_folds: Some(NumFolds::ONE),
                }),
                DfirOp::Dataflow(dataflow::Op::ProgramUnit {
                    units: vec![own],
                    iter_arg: None,
                    precision: None,
                    body: neighbours.ops.clone(),
                }),
            ]
        );
        // `L0LUROW0` binds three neighbours, and all three `get_unit`s are inside the region.
        assert_eq!(neighbours.units.len(), 3);
        assert_eq!(neighbours.ops.len(), 3);
    }

    /// 🎯 093/110 — ⛔ EVERY FOLD OF EVERY PAIR IS AN OPERAND, and the handle the body queries on is
    /// the region argument, which no `get_unit` produced.
    #[test]
    fn the_uniformized_unit_names_each_corelet_once_and_every_fold_as_an_operand() {
        let mut vals = Values::default();
        let core = Core::checked(0).expect("core 0");
        let corelets = [
            Corelet::checked(0).expect("corelet 0"),
            Corelet::checked(1).expect("corelet 1"),
        ];
        let mut units = Vec::new();
        let InitializedUniformizedUnit {
            ops,
            iterator,
            handles,
            neighbours,
        } = initialize_uniformized_unit(
            &mut vals,
            DfirUnit::Lxlu,
            &[core],
            &corelets,
            NumFolds(2),
            &mut units,
        );

        assert_eq!(units, vec![Val(0), Val(0), Val(1), Val(1)]);
        assert_eq!(
            ops[..2],
            [
                DfirOp::Dataflow(dataflow::Op::GetUnit {
                    result: Val(0),
                    residency: Residency::Corelet {
                        core,
                        corelet: corelets[0]
                    },
                    unit: DfirUnit::Lxlu,
                    num_folds: Some(NumFolds(2)),
                }),
                DfirOp::Dataflow(dataflow::Op::GetUnit {
                    result: Val(1),
                    residency: Residency::Corelet {
                        core,
                        corelet: corelets[1]
                    },
                    unit: DfirUnit::Lxlu,
                    num_folds: Some(NumFolds(2)),
                }),
            ]
        );
        assert_eq!(ops.len(), 3);
        assert_eq!(
            ops[2],
            DfirOp::Dataflow(dataflow::Op::ProgramUnit {
                units: units.clone(),
                iter_arg: Some(iterator),
                precision: None,
                body: neighbours.ops.clone(),
            })
        );
        // ⭐ THE ITERATOR IS MINTED AFTER THE UNITS and is what every neighbour query reads.
        assert_eq!(iterator, Val(2));
        assert_eq!(
            handles.iter().map(|handle| handle.unit).collect::<Vec<_>>(),
            units
        );
        assert!(neighbours.ops.iter().any(|op| matches!(
            op,
            DfirOp::Uniform(uniform::Op::QueryMap { key, .. }) if *key == iterator
        )));
    }

    /// ⛔ IT APPENDS, THOUGH NOTHING IN THE TREE DEPENDS ON THAT: both call sites hand it a fresh
    /// empty vector, so this pins the reference's shape rather than a caller's requirement.
    #[test]
    fn one_index_per_fold_appended_after_what_is_there() {
        let mut types = vec![ScalarTy::Int(1)];
        append_index_types(NumFolds(3), &mut types);
        assert_eq!(
            types,
            vec![
                ScalarTy::Int(1),
                ScalarTy::Index,
                ScalarTy::Index,
                ScalarTy::Index
            ]
        );
    }

    #[test]
    fn the_diagnostic_carries_the_translator_prefix() {
        assert_eq!(
            error_diagnostic("Invalid compute precision"),
            "[DSC2.0 to Dataflow IR]: Invalid compute precision"
        );
    }
    /// 🎯 044/110 — ⭐ THE FOLD COUNT IS THE OP'S RESULT ARITY, and the corelet the residency names.
    #[test]
    fn an_unheld_component_creates_a_folded_get_unit() {
        let mut vals = Values::default();
        let row0 = DfirUnit::PtRow(Row::checked(0).expect("row 0"));
        let residency = Residency::Corelet {
            core: Core::checked(0).expect("core 0"),
            corelet: Corelet::checked(1).expect("corelet 1"),
        };
        assert_eq!(
            create_get_unit_op(&mut vals, None, row0, residency, NumFolds(11)),
            Retrieved::Created(dataflow::Op::GetUnit {
                result: Val(0),
                residency,
                unit: row0,
                num_folds: Some(NumFolds(11)),
            })
        );
        // ⛔ THE `-1` CALLER: the L3 halves carry `core` and no `corelet` attribute.
        let core_only = Residency::Scratchpad {
            core: Core::checked(0).expect("core 0"),
        };
        assert_eq!(
            create_get_unit_op(&mut vals, None, DfirUnit::L3lu, core_only, NumFolds::ONE),
            Retrieved::Created(dataflow::Op::GetUnit {
                result: Val(1),
                residency: core_only,
                unit: DfirUnit::L3lu,
                num_folds: Some(NumFolds::ONE),
            })
        );
    }

    /// 🎯 044/110 — ⛔ A HELD COMPONENT MINTS NOTHING: the whole point of the cache is that the
    /// second request for a component emits no second `get_unit`.
    #[test]
    fn a_held_component_reuses_its_handler_and_mints_nothing() {
        let mut vals = Values::default();
        let handle = vals.mint();
        assert_eq!(
            create_get_unit_op(
                &mut vals,
                Some(handle),
                DfirUnit::L0lu,
                Residency::Corelet {
                    core: Core::checked(0).expect("core 0"),
                    corelet: Corelet::checked(0).expect("corelet 0"),
                },
                NumFolds::ONE,
            ),
            Retrieved::Reused(handle)
        );
        assert_eq!(vals.issued(), 1, "the reuse arm creates no value");
    }
    /// 🎯 006/110 — ⛔ FIRST TRIP WINS: the compute-less DSC behind a DSC1.0 is never reached, and an
    /// empty SuperDSC is the initial `version = 3`.
    #[test]
    fn the_version_is_decided_by_the_first_dsc_that_is_not_a_dsc2() {
        assert_eq!(
            translator_version(&[DscKind::Dsc2, DscKind::Dsc2]),
            TranslatorVersion::V3
        );
        assert_eq!(translator_version(&[]), TranslatorVersion::V3);
        assert_eq!(
            translator_version(&[DscKind::Dsc2, DscKind::NoComputeOp, DscKind::Dsc1]),
            TranslatorVersion::V1NoComputeOp
        );
        // ⛔ ORDER, NOT PRESENCE: the same two DSCs the other way round answer differently.
        assert_eq!(
            translator_version(&[DscKind::Dsc1, DscKind::NoComputeOp]),
            TranslatorVersion::V1
        );
    }

    /// 🎯 007/110 — ⛔⛔ THE REUSE ARM IS UNREACHABLE FROM EVERY CALLER, so this is the only place it
    /// is ever taken; the created arm is what the translator actually emits.
    #[test]
    fn an_unheld_register_file_creates_a_get_local_unit_on_the_unit_it_belongs_to() {
        let mut vals = Values::default();
        let unit = vals.mint();
        assert_eq!(
            create_get_local_unit_op(&mut vals, None, dataflow::LocalUnit::PtXrf, unit),
            Retrieved::Created(dataflow::Op::GetLocalUnit {
                result: Val(1),
                of: unit,
                which: dataflow::LocalUnit::PtXrf,
            })
        );
        let held = vals.mint();
        assert_eq!(
            create_get_local_unit_op(&mut vals, Some(held), dataflow::LocalUnit::PtXrf, unit),
            Retrieved::Reused(held)
        );
        assert_eq!(vals.issued(), 3, "the reuse arm mints nothing");
    }

    /// 🎯 008/110 — ⛔ THE PT, PE AND SFP TAKE IT AND NOTHING ELSE DOES: a mover's program unit is
    /// left without the attribute, which is what `find != end()` plus `is_any_of(PT, PE, SFP)` says.
    #[test]
    fn only_a_compute_unit_takes_the_precision_attribute() {
        let unit_of = |kind| ProgramUnit::<Dd2> {
            on: Units::one(kind, Val(0)),
            precision: None,
            body: Vec::new(),
            arch: core::marker::PhantomData,
        };
        for kind in [
            DfirUnit::PtRow(Row::checked(3).expect("row 3")),
            DfirUnit::Pe,
            DfirUnit::Sfp,
        ] {
            let mut unit = unit_of(kind);
            set_precision_in_unit_op(&mut unit, dataflow::Precision::Fp8);
            assert_eq!(unit.precision, Some(dataflow::Precision::Fp8), "{kind:?}");
        }
        // ⛔ AND THE THREE THE REFERENCE'S MAP THROWS ON ARE AMONG THE ONES IT SKIPS.
        for kind in [
            DfirUnit::Lxlu,
            DfirUnit::L0,
            DfirUnit::Constant,
            DfirUnit::SfpRing,
        ] {
            let mut unit = unit_of(kind);
            set_precision_in_unit_op(&mut unit, dataflow::Precision::Fp8);
            assert_eq!(unit.precision, None, "{kind:?}");
        }
    }
}

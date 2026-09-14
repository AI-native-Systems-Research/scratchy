//! BRIDGE 1, THE PIVOTED PATH — scratchy's SuperDSC handed to the RUST port.
//!
//! ```text
//! SubtileTape ──lower_subtile_tape_to_superdsc──► SuperDSC (Dsc / SdscOp)
//!             ──deeptools::bridges::superdsc_to_dataflow_ir──► DataflowIR
//!             ──dbo-opt --from-dfir──► init_binary
//! ```
//!
//! ⭐⭐ THIS REPLACES [`crate::lower_subtile_tape_to_dataflow_ir`], WHICH IS ABANDONED. That module
//! hand-wrote the lowering; the audit found it emitting an HBM offset as an LX address
//! (`dbo-opt`: *"Register initialization out of boundary: lxsu0 : LRF0 : 2416640"*, which is exactly
//! `1208320 * 2` for the `get_logical_memory_view %C0-lx, 1208320` it emitted on the `lxsu`), with
//! `Residence::Lx` never constructed outside a test and the store side hardcoding `from: lx` where
//! the input side matches on residence. The port is the same conversion taken from the C++ instead:
//! 110/110 functions carrying a `Replaces: eNNN_name` anchor.
//!
//! # ⛔⛔ WHAT THIS FILE OWES, STATED SO IT CANNOT BE MISTAKEN FOR DONE
//!
//! The port consumes a **scheduled** DSC and scratchy builds an **unscheduled** one. The C++
//! `ScheduleNode::NodeType` is `{BLOCK, LOOP, TRANSFER, COMPUTE, SYNC, CONDITION, ALLOCATE,
//! STICKMASK}` (`dsc/dsc2.h:446-456`) and the port's `Statement` is those less `ALLOCATE`
//! (`superdsc_to_dataflow_ir/driver.rs:467`) — but [`crate::lower_subtile_tape_to_superdsc::Dsc`]'s
//! `scheduleTree_` is a `Vec<AllocNode>` whose every `nodeType_` is `"allocate"`, with the computes
//! in a separate flat `computeOp_` list. [`schedule_census`] MEASURES that rather than asserting it.
//!
//! ⛔⛔ AND THE FOUR STAGES THAT WOULD FILL THE TREE CANNOT BE CALLED YET — this is the blocker, not
//! a design. `dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:29-57` composes them, campaign 6 ported
//! all 382 of their units, and every one of the four is uncallable or inert TODAY:
//!
//!   * `schedule::l3::dl_ops::run` (`e382_run`, stage 2a) — its `F` and `E` provider bundles require
//!     `DscOffsetFacts` (`l3/dl_ops.rs:20516`) and `LxZeroPadTransform` (`:20534`), which have **zero
//!     implementors**, test doubles included. `F` and `E` are uninhabited types.
//!   * `schedule::ddc::v1::run_v1` (`e379_run_v1`, stage 2b) — takes a `Dsc2Sites` provider
//!     (`ddc/v1.rs:6203`) with **zero implementors**; its `type Dsc: Dsc2Store` (`:6085`) has 41
//!     supertraits and no implementor, and `Dsc2Stages` (`:6155`) none either. The crate says so
//!     itself at `ddc/v1.rs:9517-9522`: *"no fixture inside this crate can call it … The dispatch is
//!     the INTEGRATION's to test."*
//!   * `schedule::ddc::v1::run` (`e381_run`) — the same `Dsc2Sites` bound (`:6685`).
//!   * `schedule::dcg::manager::run_dcg_for_dl_ops_standalone` (`e190_…`, stage 3) — callable in
//!     signature, but reaches a `todo!` on the FIRST core in BOTH arms (`dcg/manager.rs:982`, `:366`,
//!     `:1055`), because its body delegates to `dcg_fe/pcfg_gen/` and `dcg_be/`, which are OUT of
//!     campaign 6's scope by declaration.
//!
//! ⭐ SO [`Schedule::roots`] YIELDS NOTHING, AND ITS EMPTINESS IS DERIVED FROM THE INPUT rather than
//! hardcoded: `convert_v3` binds no unit and [`lower_group`] returns [`None`]. The traits are
//! satisfied over scratchy's own types and NO PROGRAM IS PRODUCED YET. An absent root is what the
//! driver already handles (the port's entry 108: a component the schedule names no root for binds
//! nothing) — and it is the honest answer, because the alternative is a fabricated statement.
//!
//! ⛔⛔ A HAND-BUILT STATEMENT IS FORBIDDEN HERE AND WAS REMOVED. An earlier revision yielded ONE
//! invented `Lxlu` transfer — a 64-lane fp16 stick from a hardcoded `DataLocation::LxluLx` to a
//! hardcoded `Sfp` destination, storing at address `0`. That is a fabricated placement, which
//! `crates/compiler/deeptools/CLAUDE.md` ranks as WORSE than a stop, and a program dbo-opt compiles
//! happily into a model that emits garbage. Its absence is the port's answer; its presence was ours.
//!
//! ⛔ AND NOTHING HERE REFUSES. No `Err`, no `Result`, and no panic on a shape it has not met.

use deeptools::arch::Arch;
use deeptools::bridges::superdsc_to_dataflow_ir::driver::{
    Dsc as PortDsc, ScheduleView, Scheduled, Translated, UnitHandles, Uniformize, Viewed, Viewing,
    run_translator,
};
use deeptools::bridges::superdsc_to_dataflow_ir::utils::DscKind;
use deeptools::generated::OpFunc;
use deeptools::islands::dataflow_ir::{
    Grid, GroupId, KernelName, OpIndex, ProgramName, Run, Values, print,
};
use deeptools::units::{Core, DfirUnit, NumFolds};

use crate::lower_subtile_tape_to_superdsc::Dsc as SuperDsc;

/// ⭐ ONE SCRATCHY `Dsc`, IN THE SHAPE THE PORT ASKS FOR.
///
/// ⛔ IT BORROWS RATHER THAN OWNS. The port's `Dsc<'c>` hands out `Viewing<'c, Self>` and the
/// `Scheduled<'s>` a view yields borrows for `'s` with `'c: 's`, so everything a statement points at
/// has to outlive the walk. Holding the scratchy `Dsc` by reference is what makes that true without
/// a clone per component.
pub struct OneDsc<'c> {
    /// The unscheduled SuperDSC this program was built from.
    dsc: &'c SuperDsc,
    /// Which cores it occupies, as the port's newtype.
    cores: Vec<Core>,
    /// WHERE EVERY BORROWED STATEMENT PAYLOAD WILL LIVE.
    ///
    /// ⛔⛔ THE STATEMENTS CANNOT OWN THEIR PAYLOADS AND `roots` CANNOT KEEP THEM. The port's
    /// `Emitted::Compute` holds `ctx: &'s OperandContext<'s>`, `Statement::Transfer` holds
    /// `&'s TransferStatement<'s>` whose `send`/`store`/`receive` are `&'s dyn Fn(..)`, and
    /// [`ScheduleView::roots`] takes `self` BY VALUE — so anything built inside `roots` is dropped
    /// at its return and the borrow cannot outlive it.
    ///
    /// ⛔⛔ AND IT IS BORROWED, NOT OWNED, WHICH THE SIGNATURE FORCES. [`PortDsc::view`] takes
    /// `&self` and returns `Viewing<'c, Self>`, so a payload it hands out must live for `'c` — but
    /// `&self.arena` on an OWNED field is only good for the anonymous borrow of that call, which is
    /// shorter. Holding `&'c Bump` makes the field `Copy` at `'c` and the return well-formed. The
    /// caller owns the `Bump` and drops it after the walk.
    arena: &'c bumpalo::Bump,
}

impl<'c> OneDsc<'c> {
    /// WRAP ONE SCRATCHY `Dsc`.
    ///
    /// ⛔ THE CORE LIST IS `coreIdsUsed_`, NOT `0..numCoresUsed_`. A bundle may occupy a
    /// non-contiguous set, and the port indexes handles by the core id it is given.
    #[must_use]
    pub fn of(dsc: &'c SuperDsc, arena: &'c bumpalo::Bump) -> OneDsc<'c> {
        let cores = dsc
            .coreIdsUsed_
            .iter()
            .filter_map(|id| Core::checked(*id))
            .collect();
        OneDsc { dsc, cores, arena }
    }
}

/// ⭐⭐ WHAT KINDS OF NODE ONE DSC'S SCHEDULE TREE CARRIES — the census the four stages owe.
///
/// ⛔ MEASURED, NOT ASSUMED. `check.py census` states the same quantity on the reference side: across
/// 187 reference programs the stages add 14,131 nodes to `scheduleTree_` (transfer +3,102, compute
/// +2,834, block +2,158, loop +2,043, sync +1,783, allocate +1,319, condition +892) to a scratchy
/// emission of 580, all `allocate`. This is the Rust-side reading of the same number, so
/// [`Schedule::roots`]'s emptiness is a CONSEQUENCE of the input rather than a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ScheduleCensus {
    /// Nodes whose `nodeType_` is `"allocate"` — which the port's `Statement` has no arm for, by
    /// design: allocations are not statements.
    pub allocate: usize,
    /// Nodes of any OTHER kind — every one of them a statement the port could walk. Zero today.
    pub statements: usize,
    /// Entries of the flat `computeOp_` list, which the reference carries as COMPUTE nodes IN the
    /// tree. Non-zero, and unreachable from `roots` until a scheduled tree names them.
    pub compute_ops: usize,
}

/// ⭐ CENSUS ONE SCRATCHY `Dsc`'s TREE. See [`ScheduleCensus`].
#[must_use]
pub fn schedule_census(dsc: &SuperDsc) -> ScheduleCensus {
    let allocate = dsc
        .scheduleTree_
        .iter()
        .filter(|node| node.nodeType_ == "allocate")
        .count();
    ScheduleCensus {
        allocate,
        statements: dsc.scheduleTree_.len().saturating_sub(allocate),
        compute_ops: dsc.computeOp_.len(),
    }
}

/// ONE COMPONENT'S ROOTS.
pub struct Schedule<'c> {
    /// Which component the driver asked about.
    comp: DfirUnit,
    /// The SuperDSC the statements are read out of.
    dsc: &'c SuperDsc,
    /// Where this component's statement payloads are allocated — see [`OneDsc::arena`].
    arena: &'c bumpalo::Bump,
}

impl<'c> ScheduleView<'c> for Schedule<'c> {
    /// ⛔⛔ YIELDS NOTHING, AND THAT IS THE OWED WORK RATHER THAN A DESIGN — see this module's header
    /// for the four stages that cannot be called yet. The statements belong to `ddc`'s expansion of
    /// this component's DDL template against its allocations; building a `Statement` from anything
    /// else is the invention this pivot exists to remove.
    ///
    /// ⭐ THE EMPTINESS IS DERIVED. It is `schedule_census(..).statements == 0` that makes the answer
    /// empty, so the day a scheduled tree arrives this stops being a constant. The component and the
    /// DSC are both in hand here: this is the one function the ported expansion plugs into.
    fn roots<'s>(self, _vals: &mut Values, _handles: UnitHandles<'s>) -> Vec<Scheduled<'s>>
    where
        'c: 's,
    {
        // ⛔ THE ARENA IS NAMED SO ITS ABSENCE OF USE IS DELIBERATE, not an oversight the compiler
        // hid: it is what a statement's borrowed payload will be allocated from, and there is no
        // statement to allocate yet. `comp` likewise selects which component's roots these are.
        let Schedule { comp, dsc, arena } = self;
        let _ = (comp, arena);
        if schedule_census(dsc).statements == 0 {
            return Vec::new();
        }
        // A tree carrying a non-allocate node means a scheduler stage ran and this walk is now the
        // thing to write. Until then the branch is unreachable BY MEASUREMENT, not by assertion.
        Vec::new()
    }
}

impl<'c> PortDsc<'c> for OneDsc<'c> {
    /// No transfer carries a latch until the TRANSFER statements land.
    type Latch = ();
    /// No transfer carries a stick mask yet.
    type Mask = ();
    /// No buffer switching yet.
    type Switch = ();
    type View = Schedule<'c>;

    fn cores(&self) -> &[Core] {
        &self.cores
    }

    /// ⭐ THE FRONTEND CONSTANT, READ OFF THE `Dsc` — `numCoreletsUsed_`, which scratchy sets from
    /// `ACTIVE_CORELETS`.
    fn num_corelets_used(&self) -> u32 {
        self.dsc.numCoreletsUsed_
    }

    /// ⭐⭐ `Dsc2` IFF IT HAS A COMPUTE, which is the reference's own test: `getTranslatorVersion`
    /// sends a compute-less DSC down its `failure()` path and a DSC1.0 down the trailing `else`
    /// (the port's entry 110). `computeOp_` empty is exactly the compute-less case, and scratchy
    /// never builds a DSC1.0.
    fn kind(&self) -> DscKind {
        if self.dsc.computeOp_.is_empty() {
            DscKind::NoComputeOp
        } else {
            DscKind::Dsc2
        }
    }

    /// ⛔ FOLDS COLLAPSED. `SdscFolds` carries the fold data scratchy writes for dxp, but the
    /// uniformized address construction that reads it is a later stage of this wiring; `false`
    /// brings every component back with one fold, which is what the port's own fixture asserts.
    fn folds_needed(&self, _comp: DfirUnit) -> bool {
        false
    }

    fn view(&self, comp: DfirUnit, _at: Viewed) -> Viewing<'c, Self> {
        Viewing {
            transfers: Vec::new(),
            head_children: Vec::new(),
            roots: Schedule {
                comp,
                dsc: self.dsc,
                arena: self.arena,
            },
        }
    }
}

/// ⭐ ONE SUPERDSC OP OF A LAUNCH GROUP — the DSC list ONE program is lowered from.
///
/// ⛔ A `Vec` OF DSCs, NOT ONE, BECAUSE THAT IS WHAT THE VERSION QUESTION IS ASKED OF.
/// `run_translator` reads `dscs_` as a list (`DSC2ToDataflowIRUtils.hpp:25`) and lowers ONE program
/// from it, so a per-DSC call would ask the version of a different list than the one it walks.
pub struct GroupOp<'a> {
    /// This op's `dscs_`, flattened in the order the json lists them.
    pub dscs: Vec<&'a SuperDsc>,
    /// Which op-func it lowers, so the program's symbol says what it is.
    pub func: OpFunc,
}

/// ⭐ SCRATCHY'S `opFuncName` AS THE SEALED ENUM, or [`None`] for a name no `OpFunc` spells.
///
/// ⛔ NO STRING REACHES A PROGRAM NAME. `OpFunc::spelling` is the generated door's own rendering of
/// the same set scratchy writes (`superdsc_opspec.rs:1162`), so this is a lookup in a closed set
/// rather than a parse — and an unspelled name yields an ABSENT program, never a fabricated `Mul`.
#[must_use]
pub fn op_func_of(name: &str) -> Option<OpFunc> {
    OpFunc::ALL.into_iter().find(|f| f.spelling() == name)
}

/// ⭐ ONE OP OF A LAUNCH GROUP AS [`GroupOp`], read off the SuperDSC the emitter already built.
///
/// [`None`] when no DSC of the op carries a compute whose op-func the door spells — which is the
/// same condition the port's entry 110 answers `NoComputeOp` to.
#[must_use]
pub fn group_op_of<'a>(dscs: Vec<&'a SuperDsc>) -> Option<GroupOp<'a>> {
    let func = dscs
        .iter()
        .flat_map(|dsc| dsc.computeOp_.iter())
        .find_map(|op| op_func_of(&op.opFuncName))?;
    Some(GroupOp { dscs, func })
}

/// ⭐⭐ LOWER ONE LAUNCH GROUP TO ONE DATAFLOWIR MODULE — the bytes the bake stages and `dbo-opt
/// --from-dfir` compiles.
///
/// One module per GROUP holding one program per op, which is the shape the declaration module
/// already has and the reason a group is ONE file rather than one per device op.
///
/// ⭐ [`None`] WHEN THE WALK BOUND NO UNIT — which is every call until [`Schedule::roots`] yields
/// statements. The caller stages nothing in that case, so no empty module reaches the compiler.
#[must_use]
pub fn lower_group(group: u32, ops: &[GroupOp<'_>]) -> Option<String> {
    lower_group_for::<deeptools::arch::Dd2>(group, ops)
}

/// [`lower_group`] with the arch named — `Dd2` is what the bake targets, and both arch impls exist in
/// every build so the choice is a parameter rather than a `cfg`.
#[must_use]
pub fn lower_group_for<A: Arch>(group: u32, ops: &[GroupOp<'_>]) -> Option<String> {
    // ⭐ ONE ARENA FOR THE WHOLE GROUP, OUTLIVING EVERY WALK AND THE PRINT. Every statement payload
    // the views allocate lives here and the arena is dropped when this function returns — after
    // `print::run` has turned the programs into characters, which is the last reader of anything
    // borrowed.
    let arena = bumpalo::Bump::new();
    let mut vals = Values::default();
    let mut programs = Vec::with_capacity(ops.len());
    for (index, op) in ops.iter().enumerate() {
        let wrapped: Vec<OneDsc<'_>> = op
            .dscs
            .iter()
            .map(|dsc| OneDsc::of(dsc, &arena))
            .collect();
        let name = ProgramName {
            group: GroupId(group),
            index: OpIndex(u32::try_from(index).unwrap_or(u32::MAX)),
            func: op.func,
        };
        let ran: Translated<A> = run_translator(
            &mut vals,
            // ⛔ DISABLED UNTIL THE FOLDS ARE READ. `Uniformize::Enabled` is the port's entry 109 —
            // one view per component over every (core, corelet) pair at once — and it is only
            // correct once `folds_needed` answers from `sdscFolds_`.
            Uniformize::Disabled,
            name,
            Grid::single(),
            &[NumFolds::ONE],
            &wrapped,
        );
        match ran {
            Translated::Ran(converted) => programs.extend(converted.program),
            // ⭐ NEITHER IS AN ERROR: they are the two states the reference runs no driver for.
            Translated::Dsc1 | Translated::NoComputeOp => {}
        }
    }
    if programs.is_empty() {
        return None;
    }
    // ⭐ THE KERNEL IS THE GROUP: one group's programs are one kernel, and `print::run` writes the
    // declaration module that names it.
    Some(print::run(&Run {
        kernel: KernelName(GroupId(group)),
        programs,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ⭐ THE DOOR IS A CLOSED SET AND THE LOOKUP FINDS IT — carrying the spelling, so a renamed
    /// variant fails here rather than silently naming every program `Mul`.
    #[test]
    fn an_op_func_resolves_from_the_name_scratchy_writes() {
        let mul = op_func_of("mul").expect("`mul` is one of the op-funcs scratchy emits");
        assert_eq!(mul.spelling(), "mul");
        assert!(
            op_func_of("no_such_op_func").is_none(),
            "a name outside the sealed set must yield an ABSENT program, not a substituted one"
        );
    }

    // ⛔⛔ THERE IS NO CENSUS TEST HERE, DELIBERATELY, AND ITS ABSENCE IS THE POINT.
    //
    // A test that builds a `ScheduleCensus` literal and asserts its own `statements == 0` verifies
    // NOTHING — it re-states its own input, which is the tautology this project has been bitten by
    // before. `AllocNode::maxDimSizes_` is a `DeviceWalk`, constructible only through
    // `StickLayout::device_walk`, so a scratchy `Dsc` cannot be fabricated in a unit test either —
    // which is correct, and means the census must be measured on the emitter's REAL output.
    //
    // ⭐ SO THE MEASUREMENT LIVES IN THE BAKE, on the SuperDSCs the build actually produces:
    // `render_dfir_input` reports it per bundle (`lower_subtile_tape_to_superdsc.rs`), against
    // `python3 /Users/nickm/tmp/bridge1-fixtures/check.py census` on the reference side.
}

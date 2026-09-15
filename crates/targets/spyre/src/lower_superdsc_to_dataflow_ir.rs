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
// ⭐ THE SCHEDULER'S TYPED TREE, THROUGH THE SEAM AND NOT THROUGH `deeptools::schedule`. `Kind` is
// `ScheduleNode::NodeType` as the closed set it is, which is what lets this file walk STATEMENTS
// instead of comparing the wire's `nodeType_` string — see
// `crates/targets/spyre/tests/scratchy_knows_nothing_about_l3.rs` for why the path matters.
use deeptools::sdsc::{DscIdx, DscState, DscTree, Kind, NodeKind};
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
    ///
    /// ⛔ STILL NEEDED, AND NOT REDUNDANT WITH [`Self::tree`]. `computeOp_`, `numCoreletsUsed_`,
    /// `coreIdsUsed_` and `labeledDs_` are the wire's and are not schedule nodes; only the TREE moves
    /// to the typed side.
    dsc: &'c SuperDsc,
    /// ⭐⭐ THE SCHEDULE STAGES 2A AND 2B ACTUALLY GREW — the object `Schedule::roots` walks.
    ///
    /// ⛔⛔ THIS IS THE OTHER TREE, AND READING THE WRONG ONE WAS THE DEFECT. `dsc.scheduleTree_` is a
    /// `Vec<AllocNode>` whose `nodeType_` is the string `"allocate"` at its ONLY construction site, so
    /// it cannot represent a loop, a transfer or a sync at all — and nothing writes the stages' nodes
    /// back into it. The census that read it was therefore measuring an unscheduled tree and reporting
    /// `statements == 0` for a reason that had nothing to do with the scheduler.
    ///
    /// ⛔ [`None`] IS A PROGRAM THAT DID NOT CONVERT OR THAT STOPPED, which is *no schedule* rather
    /// than an empty one — `Bundle::state_of`'s own distinction, and the difference between a stop and
    /// a program that scheduled to nothing.
    tree: Option<&'c DscTree>,
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
    ///
    /// ⛔ `tree` IS THIS DSC'S OWN, NOT THE PROGRAM'S. A [`DscState`] holds one [`DscTree`] per DSC of
    /// one program, so the caller picks by [`DscIdx`] — the same index the super-DSC keys `dscs_` by.
    #[must_use]
    pub fn of(
        dsc: &'c SuperDsc,
        tree: Option<&'c DscTree>,
        arena: &'c bumpalo::Bump,
    ) -> OneDsc<'c> {
        let cores = dsc
            .coreIdsUsed_
            .iter()
            .filter_map(|id| Core::checked(*id))
            .collect();
        OneDsc {
            dsc,
            tree,
            cores,
            arena,
        }
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

/// ⭐ CENSUS ONE PROGRAM'S SCHEDULED TREE. See [`ScheduleCensus`].
///
/// ⛔⛔ THE `nodeType_ == "allocate"` STRING COMPARISON IS GONE, AND THAT WAS TWO DEFECTS IN ONE LINE.
/// It read the WIRE `scheduleTree_`, which the stages never write — so the answer described scratchy's
/// emission and not the schedule — and it compared a STRING for membership of a closed set, which is
/// the *"NO STRINGS from the ddl/smc parsers — every closed set is a generated enum"* rule this crate
/// states. [`Kind::node_kind`] is that enum, and `TreeData::node_kinds` counts by it.
///
/// ⛔ AN ABSENT `tree` IS ALL-ZERO **STATEMENTS**, NOT A ZERO CENSUS: `compute_ops` still comes off the
/// wire, because `computeOp_` is a frontend fact that exists whether or not the program scheduled.
/// ⭐⭐ HOW MANY NODES SCRATCHY'S OWN `scheduleTree_` HOLDS — every one of them an ALLOCATE.
///
/// ⛔⛔ `len()`, AND THAT IS THE DELETION OF THE STRING COMPARISON RATHER THAN A LOOSENING OF IT. The
/// old census filtered `node.nodeType_ == "allocate"` over this same vec, which was redundant with the
/// TYPE: `scheduleTree_` is a `Vec<AllocNode>` and `AllocNode::nodeType_` is a `&'static str` written
/// at exactly one construction site, as `"allocate"`. So the filter could never exclude anything, and
/// comparing a string for membership of a closed set is what this crate's *"NO STRINGS … every closed
/// set is a generated enum"* rule forbids.
///
/// ⛔ AND IT IS A DIFFERENT QUANTITY FROM [`ScheduleCensus::allocate`], which is why it is its own
/// function: this is what scratchy EMITTED, that is what the scheduler MADE. Reporting one as the other
/// is how *"134 bundles, 0 launch groups"* read as a lowering defect for as long as it did.
#[must_use]
pub fn wire_allocate_nodes(dsc: &SuperDsc) -> usize {
    dsc.scheduleTree_.len()
}

/// ⭐ CENSUS ONE PROGRAM'S SCHEDULED TREE. See [`ScheduleCensus`].
#[must_use]
pub fn schedule_census(dsc: &SuperDsc, tree: Option<&DscTree>) -> ScheduleCensus {
    let kinds = tree.map(DscTree::kinds).unwrap_or_default();
    let allocate = kinds.get(&NodeKind::Allocate).copied().unwrap_or(0);
    ScheduleCensus {
        allocate,
        // ⛔ EVERY OTHER KIND, SUMMED — not `len() - allocate`. The old subtraction could not tell a
        // tree with no nodes from a tree of nothing but allocations, and it could not name WHICH kinds
        // it was counting.
        statements: kinds
            .iter()
            .filter(|(kind, _)| **kind != NodeKind::Allocate)
            .map(|(_, count)| *count)
            .sum(),
        compute_ops: dsc.computeOp_.len(),
    }
}

/// ONE COMPONENT'S ROOTS.
pub struct Schedule<'c> {
    /// Which component the driver asked about.
    comp: DfirUnit,
    /// The wire SuperDSC — `computeOp_` and `labeledDs_`, never the tree.
    dsc: &'c SuperDsc,
    /// ⭐ THE SCHEDULED TREE THIS WALK READS. See [`OneDsc::tree`].
    tree: Option<&'c DscTree>,
    /// Where this component's statement payloads are allocated — see [`OneDsc::arena`].
    arena: &'c bumpalo::Bump,
}

/// ⭐⭐ WHICH TREE KINDS ARE STATEMENTS — the set [`Schedule::roots`] would yield a [`Scheduled`] for.
///
/// ⛔ AN ALLOCATION IS NOT A STATEMENT, by [`ScheduleCensus::allocate`]'s own statement, and neither is
/// a `BLOCK`, a `LOOP` or a `CONDITION`: those are the STRUCTURE the walk descends, which is why
/// `isBlockNode()` (`dsc/dsc2.h:479`) separates them. What remains is the four kinds that carry work.
const fn is_statement(kind: &Kind) -> bool {
    matches!(
        kind,
        Kind::Transfer(_) | Kind::Compute(_) | Kind::Sync(_) | Kind::StickMask(_)
    )
}

impl<'c> ScheduleView<'c> for Schedule<'c> {
    /// ⛔⛔ YIELDS NOTHING, AND THAT IS THE OWED WORK RATHER THAN A DESIGN — see this module's header
    /// for the four stages that cannot be called yet. The statements belong to `ddc`'s expansion of
    /// this component's DDL template against its allocations; building a `Statement` from anything
    /// else is the invention this pivot exists to remove.
    ///
    /// ⭐⭐ IT NOW READS THE TREE THE STAGES GREW, WHICH CHANGES WHAT ITS EMPTINESS MEANS. It used to
    /// gate on `schedule_census(dsc).statements == 0` over the WIRE `scheduleTree_` — a vec that is all
    /// `"allocate"` by construction — so the gate was true for every program and the walk below was
    /// described as *"unreachable BY MEASUREMENT"*. Over the typed tree that is FALSE: stage 2a leaves
    /// blocks, loops and transfers (measured 4 -> 15 nodes on `rmsq_o728`), so the walk is REACHED and
    /// the honest statement is that it finds statement nodes and can build none of them yet.
    ///
    /// ⛔⛔ AND IT STILL BUILDS NOTHING, WHICH IS THE OWED WORK AND NOT A DESIGN. A `Scheduled` carries
    /// `Statement::Transfer`'s `send`/`store`/`receive` — `&'s dyn Fn(..)` payloads — and a compute's
    /// `OperandContext`, and those come from the DDL template's expansion against this component's
    /// allocations. That expansion currently writes into a DETACHED COPY of the tree
    /// (`ddc/v1.rs:6489`'s `DdlConversion::new(store.schedule_head_block())` hands back a `BlockNode`
    /// BY VALUE, where the reference holds a `DesignSpaceConfig&`, `ddl_conversion.h:511`), so the
    /// payloads do not exist to be pointed at.
    ///
    /// ⛔ SO NOTHING IS FABRICATED HERE. An earlier revision yielded ONE invented `Lxlu` transfer —
    /// 64-lane fp16, a hardcoded `Sfp` destination, storing at address `0` — and it was deleted for
    /// exactly that. Returning empty is the honest answer while the payloads are unreachable; building
    /// a `Statement` from anything but the expansion is the invention this pivot exists to remove.
    fn roots<'s>(self, _vals: &mut Values, _handles: UnitHandles<'s>) -> Vec<Scheduled<'s>>
    where
        'c: 's,
    {
        // ⛔ THE ARENA AND THE COMPONENT ARE NAMED SO THEIR ABSENCE OF USE IS DELIBERATE, not an
        // oversight the compiler hid: the arena is what a statement's borrowed payload will be
        // allocated from, and `comp` selects which component's roots these are.
        let Schedule {
            comp,
            dsc,
            tree,
            arena,
        } = self;
        let _ = (comp, dsc, arena);
        // ⛔ NO SCHEDULE AT ALL — the program did not convert or it stopped. Distinct from a program
        // that scheduled to nothing, which is the next branch.
        let Some(tree) = tree else {
            return Vec::new();
        };
        tree.with(|held| {
            // ⭐ THE WALK IS REAL AND ITS ANSWER IS DERIVED FROM THE TREE'S OWN NODES: DFS order, and
            // every node classified by the `Kind` it holds rather than by a re-looked-up type tag.
            let statements = held
                .dfs()
                .into_iter()
                .filter(|node| held.kind_of(*node).is_some_and(is_statement))
                .count();
            // ⛔ A COUNT AND NOT A `Vec` OF STATEMENTS, BECAUSE THE PAYLOADS ARE NOT REACHABLE YET —
            // see this function's own note. When they are, this loop is where each of those nodes
            // becomes a `Scheduled`, and the count is what says how many that must be.
            let _ = statements;
            Vec::new()
        })
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
                tree: self.tree,
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
    /// ⭐⭐ THE SCHEDULE STAGES 2A AND 2B LEFT FOR **THIS** PROGRAM — one [`DscTree`] per entry of
    /// [`Self::dscs`], in the same order.
    ///
    /// ⛔⛔ IT IS THE CALLER'S JOB TO PAIR IT WITH THE RIGHT PROGRAM, AND THE INDEX IS ABSOLUTE.
    /// `Bundle::state_of` is indexed into the `&[SdscOp]` the whole bundle was scheduled from, while
    /// `render_dfir_input` walks GROUPS — so a group's `i`-th program is input `r.start + i`. Passing a
    /// group-local `i` pairs every program past group 0 with ANOTHER program's addresses, and it
    /// compiles, lowers and is silently wrong.
    ///
    /// ⛔ [`None`] IS *NO SCHEDULE*, not an empty one — a program that did not convert or that stopped.
    pub state: Option<&'a DscState>,
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
pub fn group_op_of<'a>(
    dscs: Vec<&'a SuperDsc>,
    state: Option<&'a DscState>,
) -> Option<GroupOp<'a>> {
    let func = dscs
        .iter()
        .flat_map(|dsc| dsc.computeOp_.iter())
        .find_map(|op| op_func_of(&op.opFuncName))?;
    Some(GroupOp { dscs, func, state })
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
        // ⛔⛔ THE TREE IS PICKED BY THIS DSC'S OWN POSITION IN THE PROGRAM, `DscIdx(at)`, and that
        // enumerate index is the ONLY correct key: `op.dscs` is `dscs_` flattened in the order the json
        // lists them, which is the order `DscState` holds one `DscTree` per DSC in. Using the program's
        // index, or the group's, would hand a DSC another DSC's schedule — the same class of silent
        // mispairing as reading `state_of(i)` instead of `state_of(r.start + i)` at the call site.
        let wrapped: Vec<OneDsc<'_>> = op
            .dscs
            .iter()
            .enumerate()
            .map(|(at, dsc)| {
                let tree = op
                    .state
                    .and_then(|state| state.dsc(DscIdx(u32::try_from(at).unwrap_or(u32::MAX))));
                OneDsc::of(dsc, tree, &arena)
            })
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

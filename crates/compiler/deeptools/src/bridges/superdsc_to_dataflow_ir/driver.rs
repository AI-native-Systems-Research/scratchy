//! THE CONVERSION DRIVER — runTranslator, convertV3, convertV4, and the module scaffolding they build.
//!
//! 11 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e003_startDataflowIRGeneration` | 0 | 18 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:20` |
//! | `e004_stopDataflowIRGeneration` | 0 | 15 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:39` |
//! | `e005_areFoldedAddressesSameAcrossFoldsAndCoresForGivenCorelet` | 0 | 21 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:226` |
//! | `e042_terminate` | 1 | 3 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:221` |
//! | `e043_areFoldsNeeded` | 1 | 40 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:249` |
//! | `e105_constructOperationsRecursively` | 7 | 165 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:55` |
//! | `e106_ConstructAProgramUnit` | 8 | 81 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:293` |
//! | `e107_ConstructAUniformizedProgramUnit` | 8 | 84 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:378` |
//! | `e108_convertV3` | 9 | 29 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:466` |
//! | `e109_convertV4` | 9 | 33 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:499` |
//! | `e110_runTranslator` | 10 | 32 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:534` |

use super::compute::{ComputeFamily, ComputeOperation, OperandContext, compute_operation};
use super::control_flow::{
    Band, BandDim, Cond, CondComposite, CondPlace, ParametricIters, PrimaryDim, RootArgs,
    SchedNode, SwitchNode, SwitchingTransfer, TransferId, construct_loops,
};
use super::dsc_lowering::{Component, Handlers, Handles};
use super::stick_mask::{StickMaskView, construct_stick_mask_operation};
use super::sync::{SyncKind, construct_sync_operation};
use super::transfer::{
    ConstructedTransfer, ContiguousSticks, DataTransfer, LoadAndSend, LoadAndStore,
    ReceiveAndStore, StickCounts, construct_data_transfer,
};
use super::utils::{
    Neighbourhood, error_diagnostic, initialize_uniformized_unit, initialize_unit,
    set_precision_in_unit_op,
};
use crate::arch::Arch;
use crate::generated::{DataType, SyncSignal};
use crate::islands::dataflow_ir::dialects::{Op, Val, dataflow};
use crate::islands::dataflow_ir::link::RecvEnd;
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::islands::dataflow_ir::{
    Grid, Program, ProgramName, ProgramUnit, ProgramUnits, Units, Values,
};
use crate::units::{Core, Corelet, DfirUnit, NumFolds};
use core::marker::PhantomData;
use std::collections::VecDeque;

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

/// THE MODULE AND ITS ONE FUNCTION, BEFORE ANYTHING IS PUSHED INTO THEM — `module_op_` and
/// `dataflow_func_op_` between entry 003 and entry 004.
///
/// ⛔ NOTHING TO TEST FOR EMPTINESS, WHICH IS THE POINT. `if (module_op_) { emitRemark("Module op is
/// already created"); return; }` (`DSC2ToDataflowIR.cpp:21-24`) guards a FIELD that may or may not
/// have been filled; a value handed back by the call that makes it cannot be made twice, so
/// "already created" is not a state that exists here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scaffold {
    /// The module's symbol, which is also its function's name.
    pub name: ProgramName,
    /// `attributes {grid = [N]}` on the function.
    pub grid: Grid,
}

/// Replaces: e003_startDataflowIRGeneration
///
/// OPENS THE MODULE, ITS `func.func` AND ITS ONE EMPTY ENTRY BLOCK — `DSC2ToDataflowIR.cpp:20`.
///
/// ⛔ THE SYMBOL IS NOT THE LITERAL `"dataflowProgram"`, AND THAT IS DELIBERATE. The reference builds
/// ONE module per translator instance and names its function by that literal (`:29-31`); this island
/// emits one named module per program and its printer writes the same symbol on the module and the
/// function, so a fixed literal would make every program in a run collide.
///
/// ⭐ AND `auto &entry_block = *dataflow_func_op_.addEntryBlock()` (`:214`) IS UNUSED ON THE NEXT
/// LINE: the block is the function's, and what goes into it is entry 004's argument.
#[must_use]
pub fn start_dataflow_ir_generation(name: ProgramName, grid: Grid) -> Scaffold {
    Scaffold { name, grid }
}

/// Replaces: e004_stopDataflowIRGeneration
///
/// CLOSES THE FUNCTION AND HANDS BACK THE MODULE — `DSC2ToDataflowIR.cpp:39`.
///
/// ⛔ THE TWO INSERTION POINTS ARE ONE BEHAVIOUR. `setInsertionPointToStart(&front())` on an empty
/// block and `setInsertionPointAfter(&front().back())` otherwise (`:41-45`) both mean APPEND, so the
/// `func.return` lands last either way — which is where `print.rs` writes it, unconditionally.
///
/// ⛔ AND `mlir::verify(module_op_)` (`:50-52`) HAS NOTHING LEFT TO REFUSE. It checks a mutable op
/// graph; here the structure is in the types — [`ProgramUnits`] is non-empty, `Units` is non-empty
/// and every operand is a minted `Val` — so the ill-formed module it exists to catch is not
/// constructible, and the check is not a runtime one to reproduce.
#[must_use]
pub fn stop_dataflow_ir_generation<A: Arch>(
    scaffold: Scaffold,
    preamble: Vec<Op>,
    units: ProgramUnits<A>,
) -> Program<A> {
    Program {
        name: scaffold.name,
        grid: scaffold.grid,
        preamble,
        units,
        arch: core::marker::PhantomData,
    }
}

/// A NON-EMPTY LIST OF THE IDS A DSC SAYS IT USES — `core_ids_used` and `corelet_ids_used`
/// (`DSC2ToDataflowIR.cpp:227-228`).
///
/// ⛔⛔ THE TWO `DT_CHECK(!…empty())` (`:230-231`) ARE THIS TYPE, AND THEY MATTER BECAUSE THE
/// FUNCTION IS A CONJUNCTION. An empty list never enters the nested loop and falls straight to
/// `return true` — "the address is the same at every fold" asserted over no address at all, which is
/// the answer that makes folding look unnecessary.
///
/// ⭐ [`corelets_used`] ALWAYS YIELDS AT LEAST CORELET 0, so its head is always there to hand over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Used<T> {
    head: T,
    rest: Vec<T>,
}

impl<T: Copy> Used<T> {
    /// The list, its first entry being what makes it a list.
    #[must_use]
    pub fn of(head: T, rest: Vec<T>) -> Self {
        Self { head, rest }
    }

    /// Every id, head first.
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        core::iter::once(self.head).chain(self.rest.iter().copied())
    }
}

/// HOW MANY ADDRESSES ONE (core, corelet) PAIR'S UNROLLED FOLD MAP YIELDS —
/// `getAllDataWithMapUnrolled({{0, core_id}, {1, corelet_id}})`'s size, which is all entry 005 reads
/// of it (`DSC2ToDataflowIR.cpp:232-235`).
///
/// ⛔ `DT_CHECK_MSG(is_any_of(foldedAddresses.size(), 1, num_folds_), "Fold addresses can either be
/// constant or should be available for each fold")` (`:236-238`) IS THIS ENUM: two sizes and nothing
/// between them, so a third is not a case to reject after the fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldedAddresses {
    /// One address for the whole map — `size == 1`.
    Constant,
    /// One address per fold — `size == num_folds_`.
    PerFold(NumFolds),
}

impl FoldedAddresses {
    /// The size the reference tests.
    ///
    /// ⭐ SO `PerFold(NumFolds(1))` READS AS CONSTANT, exactly as `is_any_of(1, 1, 1)` does: with one
    /// fold the two states are the same state.
    #[must_use]
    pub fn count(self) -> u32 {
        match self {
            Self::Constant => 1,
            Self::PerFold(folds) => folds.0,
        }
    }
}

/// Replaces: e005_areFoldedAddressesSameAcrossFoldsAndCoresForGivenCorelet
///
/// WHETHER ONE START ADDRESS IS THE SAME AT EVERY FOLD OF EVERY (core, corelet) —
/// `DSC2ToDataflowIR.cpp:226`.
///
/// ⛔ THE TEST IS `size != 1`, NOT `size != num_folds_` (`:240-242`) — so the answer is *"there is
/// only one address"* and not *"the addresses agree"*, and [`FoldedAddresses::count`] is what keeps
/// the one-fold case answering the way the reference's `is_any_of` lets it.
///
/// ⭐ THE UNROLLED MAP IS THE CALLER'S. This crate has no `FoldManager` to unroll, so the lookup
/// arrives as a closure over the pair the loops name — the same seam [`folds_are_needed`] takes this
/// whole answer through.
pub fn folded_addresses_are_same<F>(
    cores: &Used<Core>,
    corelets: &Used<Corelet>,
    addresses: F,
) -> bool
where
    F: Fn(Core, Corelet) -> FoldedAddresses,
{
    for core in cores.iter() {
        for corelet in corelets.iter() {
            if addresses(core, corelet).count() != 1 {
                return false;
            }
        }
    }
    true
}

/// Replaces: e042_terminate
///
/// THE TRANSLATOR'S GIVE-UP MESSAGE — `DSC2ToDataflowIR.cpp:221`.
///
/// ⛔⛔ IT DOES NOT TERMINATE. The body is one `module_op_->emitError` (`:222`), which returns an
/// `InFlightDiagnostic` and neither throws nor aborts, and NONE of the four call sites returns after
/// it. `:349` and `:437` fall one line into `if (dsc_loops_to_mlir_loops_map.empty()) return;`
/// (`:352-353`) and are saved by that emptiness incidentally rather than by design; `:363` and `:451`
/// continue straight into `setPrecisionInUnitOp` on a unit whose operations failed to build
/// (`:366-371`). Handing the text back under `#[must_use]` is the guard: a caller cannot invoke this
/// and drop the result, so continuing anyway becomes a written decision instead of the default.
///
/// ⛔ AND IT IS NOT ENTRY 010. [`super::utils::error_diagnostic`] prefixes every message with
/// `[DSC2.0 to Dataflow IR]: ` (`DSC2ToDataflowIRUtils.hpp:719`); this calls `module_op_->emitError`
/// DIRECTLY, so its sentence carries no prefix. Those two are the only diagnostic shapes in the
/// translator, and routing this one through the other would print a prefix the reference does not.
#[must_use]
pub fn terminate() -> &'static str {
    "Unable to translate DSC2.0 to the Dataflow IR"
}

/// HOW ONE DIMENSION OF A CONSTANT'S FOLD DATA IS DESCRIBED — `BaseFuncType`
/// (`util/foldManager/foldInfrastructure.h:39-54`).
///
/// ⭐ ONLY THE FIRST ARM MEANS "NO FOLDING NEEDED"; the other four are the reasons folds exist, which
/// is why [`folds_are_needed`] tests inequality against one variant rather than matching four.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldDimFunc {
    /// `Constant` — one value for the whole dimension.
    Constant,
    /// `Map` — an explicit per-index table.
    Map,
    /// `Affine` — an affine function of the index.
    Affine,
    /// `WkSplit` — a work-split function.
    WkSplit,
    /// `Unknown`.
    Unknown,
}

/// ONE TRANSFER NODE, AS THE FOLD DECISION READS IT.
///
/// ⭐ TWO FIELDS AND NO MORE: which component the transfer starts at, and which component each of its
/// destination vias lands on. The fold maps behind them are the closure's business
/// ([`StartAddrOf`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Transfer<'a> {
    /// `transfer->src_.unit_`.
    pub src: Component,
    /// `transfer->dstVias_[i].loc_.unit_`, in the reference's own index order — the index is what
    /// selects `dstLdsAndLoopOffsets_[dst_idx]`.
    pub dst_vias: &'a [Component],
}

/// WHICH START ADDRESS THE FOLD-SAMENESS QUESTION IS ABOUT.
///
/// ⛔ THE INDICES ARE LOAD-BEARING. `transfer->dstLdsAndLoopOffsets_[dst_idx].startAddr_` is selected
/// by the via's POSITION in `dstVias_` (`DSC2ToDataflowIR.cpp:274-278`), so a via identified by its
/// component alone could not name its own address when two vias land on the same component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartAddrOf {
    /// `transfer->srcLdsAndLoopOffsets_.startAddr_`.
    Source {
        /// Which transfer, indexing the slice handed to [`folds_are_needed`].
        transfer: usize,
    },
    /// `transfer->dstLdsAndLoopOffsets_[via].startAddr_`.
    Destination {
        /// Which transfer.
        transfer: usize,
        /// Which via of that transfer.
        via: usize,
    },
}

/// THE CORELETS A DSC SAYS IT USES — `DSC2ToDataflowIR.cpp:251-252`.
///
/// ⛔⛔ THE COUNT IS COMPARED AGAINST 2, NOT COUNTED FROM. `std::vector<int> corelet_ids_used(1, 0)`
/// then `if (dsc.numCoreletsUsed_ == 2) emplace_back(1)` — so a DSC claiming 3 yields `{0}`, not
/// `{0, 1, 2}`, and a DSC claiming 0 still yields `{0}`. A `0..n` loop would answer differently for
/// both.
///
/// ⭐ AND THE ARCH'S OWN BOUND MAKES THE SECOND ENTRY UNCONDITIONAL ONCE THE TEST PASSES:
/// [`Corelet`] admits `0..CORELETS_PER_CORE`, which is 2 on both arches, so this list can never name
/// a corelet the hardware has not got.
#[must_use]
pub fn corelets_used(num_corelets_used: u32) -> Vec<Corelet> {
    let mut corelets = Vec::new();
    corelets.extend(Corelet::checked(0));
    if num_corelets_used == 2 {
        corelets.extend(Corelet::checked(1));
    }
    corelets
}

/// Replaces: e043_areFoldsNeeded
///
/// WHETHER THIS COMPONENT'S PROGRAM UNIT HAS TO BE FOLDED — `DSC2ToDataflowIR.cpp:249`.
///
/// ⭐ ANY NON-CONSTANT DIMENSION IN ANY CONSTANT ENDS IT. The two nested loops over `constantInfo_`
/// and its dims (`:255-261`) return `true` on the first `getFuncType(dim_idx) != Constant`, before a
/// single transfer is looked at — so the constants are a cheaper and STRICTLY EARLIER test, not one
/// more condition of the same kind.
///
/// ⛔⛔ THE TRANSFER WALK IS `if`/`else`, NOT TWO INDEPENDENT TESTS. A transfer whose SOURCE is this
/// component never has its destinations examined (`:263-283`) — not even a via that lands on the same
/// component. Reading it as *"check every end that mentions comp"* would consult a fold map the
/// reference never touches, and this answer decides the shape of the emitted program unit.
///
/// ⭐ `transfers` IS ALREADY THE COMPONENT'S OWN LIST, AND `comp` IS STILL NEEDED.
/// `traverseTreeDFS(nullptr, {TRANSFER}, comp, -1, -1)` keeps the nodes for which
/// `isNodeRelevant(comp, ..)` holds (`dsc/dsc2.cpp:2245`) — which is either END of the transfer — and
/// the `src_.unit_ == comp` test is what then picks WHICH end. Filtering alone cannot answer it.
///
/// ⛔ THE SAMENESS ANSWER IS ENTRY 005'S AND STAYS A PARAMETER.
/// `areFoldedAddressesSameAcrossFoldsAndCoresForGivenCorelet` (`:226`) unrolls a
/// `FoldManager<int64_t>` with `getAllDataWithMapUnrolled({{0, core_id}, {1, corelet_id}})` and this
/// crate has no fold manager to unroll — it is still an open item in this file. ⭐ THE CLOSURE
/// RECEIVES THE DERIVED CORELET LIST, so [`corelets_used`] stays live and observable here instead of
/// being restated inside the callee.
///
/// ⛔ AND THE ABORT BELONGS TO THAT CALLEE, NOT HERE. `DT_CHECK_MSG(is_any_of(size, 1, num_folds_),
/// "Fold addresses can either be constant or should be available for each fold")` (`:236-238`) fires
/// inside entry 005; `false` from the closure means only *"more than one address"*, which is exactly
/// the `foldedAddresses.size() != 1` the reference returns on (`:241`).
///
/// ⭐ `dsc.coreIdsUsed_` IS NOT DERIVED HERE. It is passed through untouched (`:266`, `:277`), so it
/// belongs to the closure's own capture rather than to this signature.
pub fn folds_are_needed(
    num_corelets_used: u32,
    constant_dim_funcs: &[FoldDimFunc],
    transfers: &[Transfer<'_>],
    comp: Component,
    same_across_folds: impl Fn(StartAddrOf, &[Corelet]) -> bool,
) -> bool {
    let corelets = corelets_used(num_corelets_used);

    if constant_dim_funcs
        .iter()
        .any(|func| *func != FoldDimFunc::Constant)
    {
        return true;
    }

    for (index, transfer) in transfers.iter().enumerate() {
        if transfer.src == comp {
            if !same_across_folds(StartAddrOf::Source { transfer: index }, &corelets) {
                return true;
            }
        } else {
            for (via, lands_on) in transfer.dst_vias.iter().enumerate() {
                if *lands_on == comp
                    && !same_across_folds(
                        StartAddrOf::Destination {
                            transfer: index,
                            via,
                        },
                        &corelets,
                    )
                {
                    return true;
                }
            }
        }
    }

    false
}

/// THE `emitError` SENTENCES THIS WALK RAISES — a closed set, because a diagnostic here is a REPORT:
/// entry 010 hands the text back and nothing is unwound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Raised {
    /// `:126` and `:159` — the SAME sentence from both coreCl arms.
    ConditionalsOnConditionals,
    /// `:173`. ⚠️ UNREACHABLE THROUGH ENTRY 103, whose [`None`] is a bystander unit, not a refusal.
    Compute,
    /// `:183` — entry 104's [`None`], a `BOOL`-formatted end.
    Transfer,
    /// `:190` — entry 076 signalling no one, which is its empty unit set.
    Sync,
    /// `:197` — entry 060's [`None`], a `BOOL` mask element.
    StickMask,
    /// `:208` — a child frame of a `BLOCK` that stopped.
    Block,
}

impl Raised {
    /// THE TEXT, PREFIXED AS ENTRY 010 PREFIXES IT.
    #[must_use]
    pub fn diagnostic(self) -> String {
        error_diagnostic(match self {
            Self::ConditionalsOnConditionals => "Unable to construct conditionals on conditionals.",
            Self::Compute => "Unable to construct a compute operation.",
            Self::Transfer => "Unable to construct a data transfer operation.",
            Self::Sync => "Unable to construct a sync operation.",
            Self::StickMask => "Unable to construct a stick mask operation.",
            Self::Block => "Unable to construct a block of operations.",
        })
    }
}

/// ENTRY 104'S ARGUMENTS, WHICH ARE TOO MANY TO INLINE INTO A STATEMENT — the transfer, the handles
/// it is lowered against, and the four seams it calls back into.
pub struct TransferStatement<'s> {
    /// `component_to_handler_`.
    pub handlers: &'s Handlers,
    /// The transfer as entry 104 reads it.
    pub transfer: DataTransfer<'s>,
    /// `getLocalUnitOp(comp_)`.
    pub own: Val,
    /// `getContiguousStickCounts(..)` per corelet.
    pub blocks: &'s dyn Fn(Option<Corelet>) -> Vec<(PrimaryDim, StickCounts)>,
    /// The source end, from entry 098.
    pub send:
        &'s dyn Fn(&mut Values, &mut Vec<Op>, &mut ContiguousSticks, Val) -> Option<LoadAndSend>,
    /// The destination store, from entry 091.
    pub store: &'s dyn Fn(&mut Values, &mut Vec<Op>, Component) -> LoadAndStore,
    /// One receiving end, from entry 102.
    pub receive: &'s dyn Fn(
        &mut Values,
        &mut Vec<Op>,
        &mut ContiguousSticks,
        usize,
        RecvEnd,
    ) -> Option<ReceiveAndStore>,
}

/// WHICH OF THE THREE CONDITION ARMS A `CONDITION` NODE TAKES — `hasCoreClCond()` and
/// `uniformization_` read as one answer (`:91`, `:114`).
///
/// ⛔ THE THREE `DT_CHECK`s ON THE CHILD COUNT ARE THIS SHAPE: `!children.empty()` (`:90`),
/// `children.size() == 1` (`:116`) and `1 <= num_regions <= 2` (`:134`) are a `then` that is always
/// there and an `else` only where a second child can exist at all.
pub enum CondStatement<'s> {
    /// `!hasCoreClCond()` — the `scf.if` entry 054 built for this node.
    Loops {
        /// `children[0]`, into `getThenRegion()`.
        then_: Box<Statement<'s>>,
        /// `children[1]`, into `getElseRegion()`, and [`None`] for `children.size() != 2`.
        otherwise: Option<Box<Statement<'s>>>,
    },
    /// `hasCoreClCond() && !uniformization_` — the one child, built where a dummy `arith.constant`
    /// stood and left in its place when it is erased (`:120-131`).
    CoreCl(Box<Statement<'s>>),
    /// `hasCoreClCond() && uniformization_` — the `uniform.uniformize_regions` entry 054 built.
    Uniform {
        /// `getRegionArg(i)` per region, so its LENGTH is `getNumRegions()` and its order is the
        /// region order.
        ///
        /// ⛔ `setUniformRegionArg(..)` (`:154`) REACHES THE LEAVES THROUGH THE STATEMENT, NOT
        /// THROUGH THIS WALK: entries 059 and 076 take that argument as an operand, so a leaf built
        /// for a region already holds it.
        args: &'s [Val],
        /// `getThenCoreCl(comp_)`, read for emptiness alone.
        then_units: &'s [Val],
        /// `getElseCoreCl(comp_)`, likewise.
        else_units: &'s [Val],
        /// `children[0]`.
        then_: Box<Statement<'s>>,
        /// `children[1]`, and [`None`] where the node has one branch.
        otherwise: Option<Box<Statement<'s>>>,
    },
}

/// ONE SCHEDULE NODE AS THE OPERATION WALK SEES IT — the seven `nodeType_` arms of `:63-214` and
/// nothing else, because that chain has no `else`: a node of any other kind contributes no statement.
///
/// ⛔⛔ THE FOUR LEAVES CARRY THEIR OWN OPERANDS, WHICH IS WHERE THIS PORT DIVERGES. The reference
/// hands each leaf `dsc_lowering` and the leaf reads its node back out of it; entries 060, 076, 103
/// and 104 take arguments, so whoever builds a statement has already resolved them.
pub enum Statement<'s> {
    /// `LOOP` — the children go INSIDE the loops entry 088 already built for this node.
    Loop(Vec<Statement<'s>>),
    /// `CONDITION`.
    Condition(CondStatement<'s>),
    /// `BLOCK` — the children of a node whose dummy loop is erased once they are placed.
    ///
    /// ⚠️ AND IS NOT ERASED ON A STOP, because `:212-214` sits after the `return` — a fact this walk
    /// cannot state, since the loop itself belongs to entry 088's tree.
    Block(Vec<Statement<'s>>),
    /// `COMPUTE` — entry 103's arguments.
    Compute {
        /// `SNComputeLowering(dsc_lowering, node, precision)`'s operand context.
        ctx: &'s OperandContext<'s>,
        /// Which of the five families `type_` routes to.
        family: ComputeFamily<'s>,
    },
    /// `TRANSFER`.
    Transfer(&'s TransferStatement<'s>),
    /// `SYNC` — entry 076's arguments.
    Sync {
        /// `sync->signal_`.
        signal: SyncSignal,
        /// Which of the three sync statements this is.
        kind: SyncKind<'s>,
    },
    /// `STICKMASK` — entry 060's arguments.
    StickMask {
        /// `stick_mask_->getView()`.
        view: StickMaskView,
        /// `dsc_->name_`.
        name: &'s str,
        /// `cst_info.dataFormat_`.
        format: DataType,
        /// `all_data.front()[0]`.
        mask_value: i64,
    },
}

/// WHAT ONE LEAF HANDED BACK, kept beside the ops it emitted because the ends of a transfer travel
/// inside it.
#[derive(Debug)]
pub enum Made<'s> {
    /// Entry 103's answer, and [`None`] for a unit that is not the execution unit.
    Compute(Option<ComputeOperation>),
    /// Entry 104's answer, and [`None`] for its one refusal.
    Transfer(Option<ConstructedTransfer<'s>>),
    /// Entry 076 hands nothing back.
    Sync,
    /// Entry 060's `agen.set_transfer_mask_state`, and [`None`] for its refusal.
    StickMask(Option<Val>),
}

/// WHERE ONE STATEMENT'S OPS WENT — the insertion point each arm moves the builder to, as a position
/// rather than as a mutation of one shared builder.
///
/// ⚠️ A POSITION IS NOT THE LOOP ITSELF: splicing these ops into the tree entry 088 built is entry
/// 106's join, and it is still open — that walk mints its own region argument and keeps the `BLOCK`
/// node this one erases.
#[derive(Debug)]
pub enum Placed<'s> {
    /// A leaf, at the insertion point it was reached at.
    Leaf {
        /// What it emitted, which stays placed even where it refused.
        ops: Vec<Op>,
        /// What it handed back.
        made: Made<'s>,
    },
    /// `tmp_builder.setInsertionPointToStart(<back()>.getBody())` — inside the INNERMOST loop entry
    /// 088 built for the node, while the next sibling resumes after the OUTERMOST (`:73-83`).
    InLoop(Vec<Placed<'s>>),
    /// The `scf.if`'s two regions; the next sibling resumes at `endif_builder`, which is the node
    /// AFTER the if (`:94`, `:111`).
    InIf {
        /// `getThenRegion().front()`.
        then_: Vec<Placed<'s>>,
        /// `getElseRegion().front()`, empty where there is no else branch.
        otherwise: Vec<Placed<'s>>,
    },
    /// One `uniform.uniformize_regions` region each, paired with the `getRegionArg(i)` it was built
    /// under and in region order.
    InRegions(Vec<(Val, Vec<Placed<'s>>)>),
}

/// WHAT ONE FRAME OF THE WALK LEFT BEHIND.
#[derive(Debug)]
pub struct Constructed<'s> {
    /// Every statement of this level, in order.
    pub placed: Vec<Placed<'s>>,
    /// `precision`, as the frame hands it back out through the reference's `std::string &`.
    pub precision: Option<dataflow::Precision>,
    /// Every sentence raised anywhere below, INCLUDING inside a frame whose answer was discarded.
    pub raised: Vec<Raised>,
    /// `return LogicalResult::failure()` — this frame ended early, so the siblings after the arm
    /// that stopped it were never built.
    pub stopped: bool,
}

/// A CHILD FRAME'S `precision` AND DIAGNOSTICS REACH THIS ONE WHETHER OR NOT ITS ANSWER IS READ —
/// the parameter is a `std::string &` and the diagnostics went to the module.
fn absorb<'s>(built: &mut Constructed<'s>, inner: Constructed<'s>) -> (Vec<Placed<'s>>, bool) {
    built.precision = inner.precision;
    built.raised.extend(inner.raised);
    (inner.placed, inner.stopped)
}

/// Replaces: e105_constructOperationsRecursively
///
/// EVERY STATEMENT OF ONE SCHEDULE LEVEL, at the insertion point its node kind owns, plus the
/// `precision` a MAC leaves for the caller's unit op (`DSC2ToDataflowIR.cpp:55-217`).
///
/// ⛔⛔ `LOOP` AND THE NON-CORECL `CONDITION` DISCARD THE ANSWER (`:80-81`, `:101-108`) — a refusal
/// under either does not end this frame, while the two coreCl arms and `BLOCK` test it and stop with
/// their own sentence (`:126`, `:159`, `:208`); and a stop is no rollback, because `emitError` RETURNS.
/// ⛔ `precision = getPrecision()` (`:177`) KEEPS, IT DOES NOT CLEAR: the member is a by-value copy
/// of this argument (`SNComputeLowering.hpp:59`) and only the MAC chain writes it.
#[must_use]
pub fn construct_operations_recursively<'s, A: Arch>(
    vals: &mut Values,
    statements: Vec<Statement<'s>>,
    precision: Option<dataflow::Precision>,
) -> Constructed<'s> {
    let mut built = Constructed {
        placed: Vec::new(),
        precision,
        raised: Vec::new(),
        stopped: false,
    };

    for statement in statements {
        match statement {
            Statement::Loop(children) => {
                let inner = construct_operations_recursively::<A>(vals, children, built.precision);
                // ⛔ `auto result = ..` (`:80-81`) IS NEVER READ.
                let (placed, _swallowed) = absorb(&mut built, inner);
                built.placed.push(Placed::InLoop(placed));
            }
            Statement::Condition(CondStatement::Loops { then_, otherwise }) => {
                // ⚠️ `LoopCondComposite condition = node->loopCond_` (`:98`) IS DEAD ON THE NEXT LINE.
                let inner =
                    construct_operations_recursively::<A>(vals, vec![*then_], built.precision);
                let (then_placed, _swallowed) = absorb(&mut built, inner);
                let mut else_placed = Vec::new();
                if let Some(other) = otherwise {
                    let inner =
                        construct_operations_recursively::<A>(vals, vec![*other], built.precision);
                    (else_placed, _) = absorb(&mut built, inner);
                }
                built.placed.push(Placed::InIf {
                    then_: then_placed,
                    otherwise: else_placed,
                });
            }
            Statement::Condition(CondStatement::CoreCl(child)) => {
                let inner =
                    construct_operations_recursively::<A>(vals, vec![*child], built.precision);
                let (placed, stopped) = absorb(&mut built, inner);
                // The dummy op is erased (`:131`), so the child's ops stand where it stood.
                built.placed.extend(placed);
                if stopped {
                    built.raised.push(Raised::ConditionalsOnConditionals);
                    built.stopped = true;
                    return built;
                }
            }
            Statement::Condition(CondStatement::Uniform {
                args,
                then_units,
                else_units,
                then_,
                otherwise,
            }) => {
                let mut children = vec![Some(*then_)];
                if let Some(other) = otherwise {
                    children.push(Some(*other));
                }
                // `if (uniform_region.getNumRegions() < children.size())` — a two-branch node with
                // ONE region gives that region to the `else` only where the `then` has no unit at
                // all (`:141-148`).
                let index_offset = usize::from(
                    args.len() < children.len() && then_units.is_empty() && !else_units.is_empty(),
                );
                let mut regions = Vec::new();
                let mut stopped = false;
                for (region, arg) in args.iter().enumerate() {
                    // ⚠️ `children[i + index_offset]` PAST THE PAIR BUILDS NOTHING, where the
                    // reference indexes a vector it has not checked.
                    let Some(child) = children
                        .get_mut(region + index_offset)
                        .and_then(Option::take)
                    else {
                        continue;
                    };
                    let inner =
                        construct_operations_recursively::<A>(vals, vec![child], built.precision);
                    let (placed, region_stopped) = absorb(&mut built, inner);
                    regions.push((*arg, placed));
                    if region_stopped {
                        stopped = true;
                        break;
                    }
                }
                built.placed.push(Placed::InRegions(regions));
                if stopped {
                    built.raised.push(Raised::ConditionalsOnConditionals);
                    built.stopped = true;
                    return built;
                }
            }
            Statement::Block(children) => {
                let inner = construct_operations_recursively::<A>(vals, children, built.precision);
                let (placed, stopped) = absorb(&mut built, inner);
                // `mlir_dummy_loop.erase()` (`:214`) — nothing wraps the children it held.
                built.placed.extend(placed);
                if stopped {
                    built.raised.push(Raised::Block);
                    built.stopped = true;
                    return built;
                }
            }
            Statement::Compute { ctx, family } => {
                let mut ops = Vec::new();
                let computed = compute_operation::<A>(vals, &mut ops, ctx, family);
                if let Some(written) = computed.as_ref().and_then(|made| made.precision) {
                    built.precision = Some(written);
                }
                built.placed.push(Placed::Leaf {
                    ops,
                    made: Made::Compute(computed),
                });
            }
            Statement::Transfer(transfer) => {
                let mut ops = Vec::new();
                let made = construct_data_transfer(
                    vals,
                    &mut ops,
                    transfer.handlers,
                    &transfer.transfer,
                    transfer.own,
                    |corelet| (transfer.blocks)(corelet),
                    transfer.send,
                    transfer.store,
                    transfer.receive,
                );
                let refused = made.is_none();
                built.placed.push(Placed::Leaf {
                    ops,
                    made: Made::Transfer(made),
                });
                if refused {
                    built.raised.push(Raised::Transfer);
                    built.stopped = true;
                    return built;
                }
            }
            Statement::Sync { signal, kind } => {
                let mut ops = Vec::new();
                construct_sync_operation(vals, &mut ops, signal, kind);
                // ⛔ ENTRY 076 HANDS BACK NOTHING, AND ITS ONE `failure()` IS AN EMPTY UNIT SET —
                // which is exactly the case where it signals no one and emits no op.
                let refused = ops.is_empty();
                built.placed.push(Placed::Leaf {
                    ops,
                    made: Made::Sync,
                });
                if refused {
                    built.raised.push(Raised::Sync);
                    built.stopped = true;
                    return built;
                }
            }
            Statement::StickMask {
                view,
                name,
                format,
                mask_value,
            } => {
                let mut ops = Vec::new();
                let made =
                    construct_stick_mask_operation(vals, &mut ops, view, name, format, mask_value);
                let refused = made.is_none();
                built.placed.push(Placed::Leaf {
                    ops,
                    made: Made::StickMask(made),
                });
                if refused {
                    built.raised.push(Raised::StickMask);
                    built.stopped = true;
                    return built;
                }
            }
        }
    }

    built
}

/// Replaces: e106_ConstructAProgramUnit
///
/// ONE COMPONENT'S PROGRAM UNIT ON ONE `(core, corelet)` — `DSC2ToDataflowIR.cpp:293`.
///
/// ⛔⛔ THE FOLD PRODUCT PICKS THE INITIALIZER (`:311-324`), NOT THE FUNCTION NAME: a product of
/// one takes entry 084's single-unit pair and a product above one takes entry 093's uniformized
/// set over the SAME one-core, one-corelet lists — so both unit builders here can uniformize.
/// ⛔ AND THE ROOTS ARRIVE AS A CLOSURE, because `getNextView(comp, corelet_id, core_id)` (`:298`)
/// is read against handles this function has not bound yet, which moves `roots.empty()` (`:300`)
/// to AFTER the `get_unit`s are minted — see [`UnitHandles`].
pub fn construct_a_program_unit<'c, A: Arch, L, M, T: Clone, R: ScheduleView<'c>>(
    vals: &mut Values,
    comp: DfirUnit,
    core: Core,
    corelet: Corelet,
    fold_dims: &[NumFolds],
    switching: SwitchInputs<'_, '_, L, M, T>,
    roots: R,
) -> ConstructedProgramUnit<A> {
    let folds = num_folds(fold_dims);
    // `if (num_folds_ == 1)` (`:316`) — the EQUALITY, so a fold dimension of size zero takes the
    // uniformized arm exactly as the reference's `else` does.
    let opened = if folds.0 == 1 {
        let init = initialize_unit(vals, core, corelet, comp);
        Opened {
            bound: get_units(init.ops),
            // `units_involved.emplace_back(unit_op.getUnits()[0])` (`:318`) — the one handle entry
            // 084 bound, which is why this is [`Units::one`] and not a filtered list.
            on: Units::one(comp, init.own),
            iterator: None,
            neighbours: init.neighbours,
            handles: None,
        }
    } else {
        let mut units = Vec::new();
        let init = initialize_uniformized_unit(vals, comp, &[core], &[corelet], folds, &mut units);
        let bound = get_units(init.ops);
        let Some(on) = units_of(comp, &units) else {
            return ConstructedProgramUnit::empty(bound);
        };
        Opened {
            bound,
            on,
            iterator: Some(init.iterator),
            neighbours: init.neighbours,
            handles: Some(init.handles),
        }
    };
    construct_a_units_program(vals, opened, switching, roots)
}

/// Replaces: e107_ConstructAUniformizedProgramUnit
///
/// ONE COMPONENT'S PROGRAM UNIT OVER EVERY CORE AND CORELET THE DSC USES — `:378`.
///
/// ⛔⛔ THE FOLD COUNT IS ASKED TWICE AND THE SECOND ASK IS CONDITIONAL: `areFoldsNeeded` runs
/// ONLY where the product already exceeds one (`:391-400`), so a one-fold unit never consults
/// entry 043 at all — which is why the answer arrives as a closure rather than a `bool`.
/// ⛔ AND THE CORELET LIST IS DERIVED WHILE THE CORE LIST IS GIVEN: `corelet_ids_used` is
/// [`corelets_used`]'s comparison against 2 (`:407-408`) while `dsc.coreIdsUsed_` is passed
/// through untouched (`:410`).
pub fn construct_a_uniformized_program_unit<'c, A: Arch, L, M, T: Clone, R: ScheduleView<'c>>(
    vals: &mut Values,
    comp: DfirUnit,
    cores: &[Core],
    num_corelets_used: u32,
    fold_dims: &[NumFolds],
    folds_needed: impl FnOnce() -> bool,
    switching: SwitchInputs<'_, '_, L, M, T>,
    roots: R,
) -> ConstructedProgramUnit<A> {
    let mut folds = num_folds(fold_dims);
    // `if (this->num_folds_ > 1) { if (!areFoldsNeeded(dsc, comp)) this->num_folds_ = 1; }`.
    if folds.0 > 1 && !folds_needed() {
        folds = NumFolds::ONE;
    }
    let corelets = corelets_used(num_corelets_used);
    let mut units = Vec::new();
    let init = initialize_uniformized_unit(vals, comp, cores, &corelets, folds, &mut units);
    let bound = get_units(init.ops);
    let Some(on) = units_of(comp, &units) else {
        return ConstructedProgramUnit::empty(bound);
    };
    let opened = Opened {
        bound,
        on,
        iterator: Some(init.iterator),
        neighbours: init.neighbours,
        handles: Some(init.handles),
    };
    construct_a_units_program(vals, opened, switching, roots)
}

/// Replaces: e108_convertV3
///
/// EVERY (DSC, CORE, CORELET, COMPONENT)'S PROGRAM UNIT, UNIFORMIZATION OFF — `:466`.
///
/// ⛔⛔ THE CORELET LOOP IS `0..num_corelets` UNCONDITIONALLY (`:483`) AND NEVER READS
/// `numCoreletsUsed_DSC2_`, which is entry 109's business alone — so a DSC that says it uses one
/// corelet still gets both corelets' units here.
/// ⛔ AND `core_idx` IS DEAD IN THE CALLEE. Entry 106 reads `core_id` and `corelet_id` and never the
/// index (`:293-372`), which is why this port hands over the id alone — and why `:523` can pass
/// `core_id` for it without changing anything.
#[must_use]
pub fn convert_v3<'c, A: Arch, D: Dsc<'c>>(
    vals: &mut Values,
    name: ProgramName,
    grid: Grid,
    components: &Used<DfirUnit>,
    fold_dims: &[NumFolds],
    dscs: &[D],
) -> Converted<A> {
    let scaffold = start_dataflow_ir_generation(name, grid);
    let mut assembled = Assembled::new();
    for dsc in dscs {
        for &core in dsc.cores() {
            for corelet in BOTH_CORELETS {
                for comp in components.iter() {
                    let Viewing {
                        transfers,
                        mut head_children,
                        roots,
                    } = dsc.view(comp, Viewed::OnPair(core, corelet));
                    assembled.push(construct_a_program_unit(
                        vals,
                        comp,
                        core,
                        corelet,
                        fold_dims,
                        SwitchInputs {
                            transfers: &transfers,
                            head_children: &mut head_children,
                        },
                        roots,
                    ));
                }
            }
        }
    }
    assembled.close(scaffold)
}

/// Replaces: e109_convertV4
///
/// EVERY (DSC, COMPONENT)'S UNIFORMIZED UNIT, PLUS THE CORELET-1 PASS THE L3 FIX NEEDS — `:499`.
///
/// ⛔⛔ THE SECOND PASS IS EXTRA, NOT A FALLBACK. `if (dsc.numCoreletsUsed_DSC2_ != num_corelets)`
/// (`:518`) adds a per-core, corelet-1 unit ON TOP of the uniformized one every component already
/// got, because DCG-generated L3 code syncs with both corelets while the DSC names one.
/// ⛔ AND IT NEVER CONSULTS ENTRY 043: entry 106 has no `areFoldsNeeded` (`:311-324`), so a fold
/// product the uniformized pass collapsed back to one stays folded in these units.
#[must_use]
pub fn convert_v4<'c, A: Arch, D: Dsc<'c>>(
    vals: &mut Values,
    name: ProgramName,
    grid: Grid,
    components: &Used<DfirUnit>,
    fold_dims: &[NumFolds],
    dscs: &[D],
) -> Converted<A> {
    let scaffold = start_dataflow_ir_generation(name, grid);
    let mut assembled = Assembled::new();
    for dsc in dscs {
        for comp in components.iter() {
            let Viewing {
                transfers,
                mut head_children,
                roots,
            } = dsc.view(comp, Viewed::OverEveryPair);
            assembled.push(construct_a_uniformized_program_unit(
                vals,
                comp,
                dsc.cores(),
                dsc.num_corelets_used(),
                fold_dims,
                // ⛔ LAZY, so a one-fold component never asks — see entry 107.
                || dsc.folds_needed(comp),
                SwitchInputs {
                    transfers: &transfers,
                    head_children: &mut head_children,
                },
                roots,
            ));
        }

        // `// TODO: Temporary fix in presence of DCG generated L3 code syncing with both corelets,
        // but dsc2.0 mentioning use of a single corelet.` (`:516-517`).
        if dsc.num_corelets_used() != NUM_CORELETS {
            for &core in dsc.cores() {
                for comp in components.iter() {
                    let Viewing {
                        transfers,
                        mut head_children,
                        roots,
                    } = dsc.view(comp, Viewed::OnPair(core, CORELET_ONE));
                    assembled.push(construct_a_program_unit(
                        vals,
                        comp,
                        core,
                        CORELET_ONE,
                        fold_dims,
                        SwitchInputs {
                            transfers: &transfers,
                            head_children: &mut head_children,
                        },
                        roots,
                    ));
                }
            }
        }
    }
    assembled.close(scaffold)
}

// crustify:todo: e110_runTranslator

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 106/110 · 107/110 — THE TWO WALKS OVER ONE SCHEDULE, PAIRED
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE LEAF STATEMENT — [`Statement`]'s FOUR LEAF ARMS and none of its three structural ones.
///
/// ⛔⛔ THE STRUCTURAL ARMS ARE MISSING ON PURPOSE, AND THAT IS THE COMPILE-TIME PAIRING. A
/// [`Scheduled::Leaf`] is one node of the schedule seen by BOTH walks at once: entry 096 asks it for
/// the ops that go inside its loops and entry 105 asks it for the statement that emits them. A
/// structural statement standing in a leaf slot would pair a loop-view leaf against a SUBTREE of ops,
/// and the only way to notice would be at run time — so the type does not admit one.
pub enum Emitted<'s> {
    /// `COMPUTE` — entry 103's arguments.
    Compute {
        /// `SNComputeLowering(dsc_lowering, node, precision)`'s operand context.
        ctx: &'s OperandContext<'s>,
        /// Which of the five families `type_` routes to.
        family: ComputeFamily<'s>,
    },
    /// `TRANSFER`.
    Transfer(&'s TransferStatement<'s>),
    /// `SYNC` — entry 076's arguments.
    Sync {
        /// `sync->signal_`.
        signal: SyncSignal,
        /// Which of the three sync statements this is.
        kind: SyncKind<'s>,
    },
    /// `STICKMASK` — entry 060's arguments.
    StickMask {
        /// `stick_mask_->getView()`.
        view: StickMaskView,
        /// `dsc_->name_`.
        name: &'s str,
        /// `cst_info.dataFormat_`.
        format: DataType,
        /// `all_data.front()[0]`.
        mask_value: i64,
    },
}

impl<'s> Emitted<'s> {
    /// The same leaf, as the operation walk takes it.
    fn statement(self) -> Statement<'s> {
        match self {
            Emitted::Compute { ctx, family } => Statement::Compute { ctx, family },
            Emitted::Transfer(transfer) => Statement::Transfer(transfer),
            Emitted::Sync { signal, kind } => Statement::Sync { signal, kind },
            Emitted::StickMask {
                view,
                name,
                format,
                mask_value,
            } => Statement::StickMask {
                view,
                name,
                format,
                mask_value,
            },
        }
    }
}

/// ONE SCHEDULE SUBTREE AS BOTH WALKS SEE IT — `dsc.scheduleTree_.getHead()->getNextView(..)`, once,
/// for the two functions that walk it twice.
///
/// # 🛑 THE REFERENCE HANDS THE SAME `roots` TO BOTH WALKS AND THIS IS THAT FACT
///
/// ⛔⛔ TWO INDEPENDENT TREES COULD DISAGREE ABOUT THE SCHEDULE. `constructLoops()` (`:347`) and
/// `constructOperationsRecursively(.., {roots}, ..)` (`:361`) read ONE `dsc.scheduleTree_`, and the
/// second finds the first's loops through `dsc_loops_to_mlir_loops_map_`. This port has no map to
/// find them through — it splices instead ([`fill`]) — so a mismatched pair of trees would silently
/// nest a statement's ops under the wrong loop. One tree, viewed twice, cannot mismatch.
///
/// ⚠️ AND THE WALKS RUN IN THE OPPOSITE ORDER HERE. The reference builds the loops and then aims a
/// builder back inside them; a value tree has no insertion point to return to, so the STATEMENT walk
/// runs first and [`SchedNode::Leaf`] carries its ops into the nest. The nesting is identical; what
/// changes is the SSA numbering and the order two diagnostics are raised in, and the latter is put
/// back by hand in [`ConstructedProgramUnit::raised`].
pub enum Scheduled<'s> {
    /// `loopNode->isParametricLoop()`.
    Parametric {
        /// `parametricIterCount(..)`.
        iters: ParametricIters,
        /// `getNextView(..)`.
        children: Vec<Scheduled<'s>>,
    },
    /// A `LoopNode` that is not parametric — [`Band`]'s fields, minus the children it shares.
    Band {
        /// `node->dims_`, in the order the node declares them.
        dims: Vec<BandDim>,
        /// This band's switching transfers, in iter-arg order.
        transfers: &'s [TransferId],
        /// What its outermost loop's iter args are built from where nothing encloses it.
        root: RootArgs<'s>,
        /// The two data-stage names every loop of this band is named from.
        stages: (&'s str, &'s str),
        /// `getNextView(..)`.
        children: Vec<Scheduled<'s>>,
    },
    /// A `ConditionNode`, whose children hang off the arm its condition takes.
    Condition {
        /// `sched_node->name_`.
        name: &'s str,
        /// Which arm, and its branches.
        place: ScheduledCond<'s>,
    },
    /// A `BlockNode`.
    Block {
        /// `getNextView(..)`.
        children: Vec<Scheduled<'s>>,
    },
    /// Any other node kind — a transfer, compute, sync or stick mask.
    Leaf(Emitted<'s>),
}

/// A CONDITION NODE'S ARM WITH ITS BRANCHES — [`CondPlace`] and [`CondStatement`] as one, because the
/// two walks agree on which arm a node takes and on how many branches it has.
///
/// ⛔ THE `then`/`else` PAIR IS A FIELD PER BRANCH AND NOT A `Vec`, which is the three child-count
/// `DT_CHECK`s (`:90`, `:116`, `:134` and `:5410`) written into the type: a `then` that is always
/// there and an `else` only where a second child can exist.
pub enum ScheduledCond<'s> {
    /// `!hasCoreClCond()` — an `scf.if` over loop iterators.
    Loops {
        /// `node->loopCond_`, which entry 054 builds the comparison chain from.
        composite: CondComposite<'s>,
        /// `children[0]`.
        then_: Box<Scheduled<'s>>,
        /// `children[1]`, and [`None`] for a one-branch node.
        otherwise: Option<Box<Scheduled<'s>>>,
    },
    /// `hasCoreClCond() && !uniformization_` — the one child, unguarded.
    CoreCl(Box<Scheduled<'s>>),
    /// `hasCoreClCond() && uniformization_` — `uniform.uniformize_regions`.
    Uniform {
        /// `getRegionArg(i)` per region, in region order.
        ///
        /// ⛔ ONE LIST FEEDS BOTH WALKS, and that is the join. Entry 105's leaves take these as
        /// OPERANDS (`setUniformRegionArg`, `:154`) while entry 088's regions are DECLARED under them
        /// (`:5474`) — [`view_one`] hands the same value to each, so no leaf can name an argument its
        /// region does not have.
        args: &'s [Val],
        /// `getThenCoreCl(comp_)`'s units.
        then_units: &'s [Val],
        /// `getElseCoreCl(comp_)`'s units.
        else_units: &'s [Val],
        /// `children[0]`.
        then_: Box<Scheduled<'s>>,
        /// `children[1]`, and [`None`] for a one-branch node.
        otherwise: Option<Box<Scheduled<'s>>>,
    },
}

/// EVERY ROOT SUBTREE, AS BOTH WALKS TAKE IT.
fn views<'s>(scheduled: Vec<Scheduled<'s>>) -> (Vec<SchedNode<'s>>, Vec<Statement<'s>>) {
    let mut nodes = Vec::with_capacity(scheduled.len());
    let mut statements = Vec::with_capacity(scheduled.len());
    for one in scheduled {
        let (node, statement) = view_one(one);
        nodes.push(node);
        statements.push(statement);
    }
    (nodes, statements)
}

/// ONE SUBTREE, AS BOTH WALKS TAKE IT — the loop view with its leaves' ops still empty, and the
/// statement view that fills them.
///
/// ⛔ THE LEAF'S `results` IS EMPTY HERE AND STAYS EMPTY. `sched_nodes_results` is filled by entry
/// 088's own loop arms (`:5265`, `:5343`), never by a node this walk turns into a leaf.
fn view_one<'s>(one: Scheduled<'s>) -> (SchedNode<'s>, Statement<'s>) {
    match one {
        Scheduled::Parametric { iters, children } => {
            let (nodes, statements) = views(children);
            (
                SchedNode::Parametric {
                    iters,
                    children: nodes,
                },
                Statement::Loop(statements),
            )
        }
        Scheduled::Band {
            dims,
            transfers,
            root,
            stages,
            children,
        } => {
            let (nodes, statements) = views(children);
            (
                SchedNode::Band(Band {
                    dims,
                    transfers,
                    root,
                    stages,
                    children: nodes,
                }),
                Statement::Loop(statements),
            )
        }
        Scheduled::Block { children } => {
            let (nodes, statements) = views(children);
            (
                SchedNode::Block { children: nodes },
                Statement::Block(statements),
            )
        }
        Scheduled::Leaf(emitted) => (
            SchedNode::Leaf {
                ops: Vec::new(),
                results: Vec::new(),
            },
            emitted.statement(),
        ),
        Scheduled::Condition { name, place } => match place {
            ScheduledCond::Loops {
                composite,
                then_,
                otherwise,
            } => {
                let (then_node, then_statement) = view_one(*then_);
                let mut children = vec![then_node];
                let mut else_statement = None;
                if let Some(other) = otherwise {
                    let (node, statement) = view_one(*other);
                    children.push(node);
                    else_statement = Some(Box::new(statement));
                }
                (
                    SchedNode::Condition(Cond {
                        name,
                        place: CondPlace::Loops(composite),
                        children,
                    }),
                    Statement::Condition(CondStatement::Loops {
                        then_: Box::new(then_statement),
                        otherwise: else_statement,
                    }),
                )
            }
            ScheduledCond::CoreCl(child) => {
                let (node, statement) = view_one(*child);
                (
                    SchedNode::Condition(Cond {
                        name,
                        place: CondPlace::CoreCl,
                        children: vec![node],
                    }),
                    Statement::Condition(CondStatement::CoreCl(Box::new(statement))),
                )
            }
            ScheduledCond::Uniform {
                args,
                then_units,
                else_units,
                then_,
                otherwise,
            } => {
                let (then_node, then_statement) = view_one(*then_);
                let mut children = vec![then_node];
                let mut else_statement = None;
                if let Some(other) = otherwise {
                    let (node, statement) = view_one(*other);
                    children.push(node);
                    else_statement = Some(Box::new(statement));
                }
                // ⛔ THE ARGUMENTS ARE HANDED OUT IN REGION ORDER TO THE BRANCHES THAT HAVE ONE, which
                // is `if (!then_cl_units.empty())` then `if (max_num_regions == 2 &&
                // !else_cl_units.empty())` (`:5477-5479`) — a branch with no unit consumes nothing,
                // so the `else` takes `args[0]` where the `then` is empty.
                let mut region_args = args.iter().copied();
                let then_region = (!then_units.is_empty())
                    .then(|| region_args.next())
                    .flatten()
                    .map(|arg| (arg, then_units));
                let else_region = (else_statement.is_some() && !else_units.is_empty())
                    .then(|| region_args.next())
                    .flatten()
                    .map(|arg| (arg, else_units));
                (
                    SchedNode::Condition(Cond {
                        name,
                        place: CondPlace::UniformCoreCl {
                            then_: then_region,
                            else_: else_region,
                        },
                        children,
                    }),
                    Statement::Condition(CondStatement::Uniform {
                        args,
                        then_units,
                        else_units,
                        then_: Box::new(then_statement),
                        otherwise: else_statement,
                    }),
                )
            }
        },
    }
}

/// WHAT A CONDITION NODE'S ARM DOES TO THE QUEUE, read off [`CondPlace`] before its children are
/// touched — the immutable borrow of `place` has to end before `children` is written.
enum CondShape {
    /// [`CondPlace::CoreCl`] — entry 105 spliced the child's ops into THIS level.
    Flat,
    /// [`CondPlace::Loops`] — one [`Placed::InIf`].
    If,
    /// [`CondPlace::UniformCoreCl`] — one [`Placed::InRegions`], and the two arguments that say which
    /// branch each region belongs to.
    Regions(Option<Val>, Option<Val>),
}

/// EVERY STATEMENT'S OPS INTO THE LOOP NEST THEY BELONG TO — `dsc_loops_to_mlir_loops_map_` and
/// `builder.setInsertionPoint(..)`, as a splice.
///
/// ⛔⛔ THIS IS THE JOIN THE TWO WALKS' INVERSION MAKES NECESSARY, and the queue is what keeps it
/// honest: entry 105 flattens a `BLOCK` and a coreCl `CONDITION` into its OWN level (`:131`, `:214`)
/// while entry 088 keeps a node for each, so those two arms recurse on the SAME queue and every other
/// arm pops one entry. Matching the trees positionally instead would mis-align at the first block.
///
/// ⭐ AN EARLY STOP LEAVES A PREFIX AND THAT IS TOTAL, NOT A REFUSAL. `constructOperationsRecursively`
/// returning `failure()` means the siblings after the refusing statement were never built, so the
/// queue runs dry and the leaves it did not reach keep the empty ops they were viewed with — which is
/// exactly the IR the reference is left holding.
fn fill(nodes: &mut [SchedNode<'_>], placed: &mut VecDeque<Placed<'_>>) {
    for node in nodes {
        match node {
            SchedNode::Parametric { children, .. } => {
                if let Some(Placed::InLoop(inner)) = placed.pop_front() {
                    fill(children, &mut VecDeque::from(inner));
                }
            }
            SchedNode::Band(band) => {
                if let Some(Placed::InLoop(inner)) = placed.pop_front() {
                    fill(&mut band.children, &mut VecDeque::from(inner));
                }
            }
            // `mlir_dummy_loop.erase()` (`:214`) — entry 105 left these at this level.
            SchedNode::Block { children } => fill(children, placed),
            SchedNode::Leaf { ops, .. } => {
                if let Some(Placed::Leaf { ops: emitted, .. }) = placed.pop_front() {
                    *ops = emitted;
                }
            }
            SchedNode::Condition(cond) => {
                let shape = match &cond.place {
                    CondPlace::Loops(_) => CondShape::If,
                    CondPlace::CoreCl => CondShape::Flat,
                    CondPlace::UniformCoreCl { then_, else_ } => {
                        CondShape::Regions(then_.map(|(arg, _)| arg), else_.map(|(arg, _)| arg))
                    }
                };
                match shape {
                    // The dummy `arith.constant` is erased (`:131`).
                    CondShape::Flat => fill(&mut cond.children, placed),
                    CondShape::If => {
                        if let Some(Placed::InIf { then_, otherwise }) = placed.pop_front() {
                            let (head, tail) = cond.children.split_at_mut(1);
                            fill(head, &mut VecDeque::from(then_));
                            fill(tail, &mut VecDeque::from(otherwise));
                        }
                    }
                    CondShape::Regions(then_arg, else_arg) => {
                        if let Some(Placed::InRegions(regions)) = placed.pop_front() {
                            for (arg, inner) in regions {
                                // ⛔ THE REGION ARGUMENT NAMES THE BRANCH, NOT THE REGION'S POSITION.
                                // Entry 105 counts regions and entry 088 shifts the child index past a
                                // `then` with no unit (`:5477-5479`), so region 0 is `children[1]`
                                // there and `children[0]` here — the argument is the one thing both
                                // walks agree on.
                                let which = if then_arg == Some(arg) {
                                    0
                                } else if else_arg == Some(arg) {
                                    1
                                } else {
                                    continue;
                                };
                                if let Some(child) = cond.children.get_mut(which) {
                                    fill(core::slice::from_mut(child), &mut VecDeque::from(inner));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `num_folds_ = 1; for (i) num_folds_ *= getFoldDimSize(i);` (`:311-314`, `:731-734`).
///
/// ⭐ THE PRODUCT OF NO DIMENSIONS IS ONE, which is the reference's initializer rather than a special
/// case: `sdscFolds_.getNumDims() == 0` leaves `num_folds_` at the 1 it was set to.
fn num_folds(dims: &[NumFolds]) -> NumFolds {
    NumFolds(dims.iter().map(|dim| dim.0).product())
}

/// THE `dataflow.get_unit` OPS ALONE — everything entries 084 and 093 emitted EXCEPT the trailing
/// `dataflow.program_unit`.
///
/// ⛔⛔ THE HANDLES ARE BOUND OUTSIDE THE REGION THEY ARE USED IN. `OpBuilder builder(unit_op)` in
/// entry 084 is aimed at `dataflow_func_op_.back()` (`:670-671`), so the `get_unit`s are siblings of
/// the program unit and not its contents — dropping the op they were emitted beside is how the same
/// unit is rebuilt as a typed [`ProgramUnit`], body and all, without duplicating them into it.
fn get_units(mut ops: Vec<Op>) -> Vec<Op> {
    ops.pop();
    ops
}

/// `units_involved_` AS ONE KIND'S SET — every handle entry 093 pushed, all of them `comp`.
///
/// ⭐ [`None`] IS AN EMPTY `units_involved`, WHICH IS AN ABSENT UNIT AND NOT A REFUSAL: a fold
/// dimension of size zero, or an empty core list, leaves the reference building a `ProgramUnitOp` over
/// no units at all — see [`Units::of`], where that shape is a crash in the backend.
fn units_of(comp: DfirUnit, units: &[Val]) -> Option<Units> {
    let bound: Vec<(DfirUnit, Val)> = units.iter().map(|unit| (comp, *unit)).collect();
    Units::of(comp, &bound)
}

/// THE BUFFER-SWITCH INPUTS ENTRY 096 READS — the two arguments that are neither the schedule nor the
/// unit, kept together so the shared tail takes one parameter for them.
pub struct SwitchInputs<'t, 'h, L, M, T> {
    /// The component's transfers, in the order that IS each one's [`TransferId`].
    pub transfers: &'t [SwitchingTransfer<'t, L, M>],
    /// The schedule head's children, which entry 020 writes `all` onto.
    pub head_children: &'h mut [SwitchNode<T>],
}

/// WHAT A UNIT'S PREAMBLE BOUND, HANDED TO THE SCHEDULE VIEW — `component_to_handler_`,
/// `unit_to_value_map_`, `program_unit_iterator` and `units_involved_` at the moment
/// `SNDSCLowering` is constructed (`:337-344`, `:425-432`).
///
/// ⛔⛔ THIS IS WHY THE ROOTS ARE A CLOSURE. `getNextView(..)` is read at `:298` BEFORE the unit is
/// built, but every leaf of the view it returns is lowered against these handles — the reference gets
/// away with it because the leaves read `dsc_lowering` back out much later, and a value tree has to
/// have them in hand when the leaf is made. So the emptiness test moves one step later than `:300`
/// and the `get_unit`s are minted for a component that may have no roots — and are DROPPED again
/// where it has none, so the function keeps exactly the handles the reference put in it.
pub struct UnitHandles<'u> {
    /// `component_to_handler_` and the register files behind it.
    pub neighbours: &'u Neighbourhood,
    /// `unit_to_value_map_`, and [`None`] for the non-uniformized arm that has none.
    pub handles: Option<&'u Handles>,
    /// `program_unit_iterator`.
    pub iterator: Option<Val>,
    /// `units_involved_`.
    pub units: &'u Units,
}

/// THE SCHEDULE, READ AGAINST A UNIT THAT DOES NOT EXIST YET —
/// `dsc.scheduleTree_.getHead()->getNextView(..)` (`:298`, `:380`) with the leaves already lowered.
///
/// # 🛑 WHY THIS IS A TRAIT AND NOT A CLOSURE
///
/// ⛔⛔ `for<'s> FnOnce(&mut Values, UnitHandles<'s>) -> Vec<Scheduled<'s>>` IS A BOUND NO REAL VIEW
/// CAN MEET. A universally quantified `'s` admits `'static`, so such a closure could return nothing
/// borrowed from the caller's own DSC — and every leaf is: entry 103's [`OperandContext`], entry 104's
/// [`TransferStatement`], the `&[Val]` unit lists. `'c: 's` is the bound that is actually true: the
/// caller's data outlives the call, and `'s` is the region the unit's handles live in INSIDE it.
///
/// ⛔ AND THE TWO SOURCES REALLY ARE TWO. `getNextView(..)` is read before the unit is built while its
/// leaves are lowered against `component_to_handler_` and `unit_to_value_map_`, which entries 084 and
/// 093 mint here — so a view has to reach both, at once, without either escaping.
pub trait ScheduleView<'c> {
    /// `{roots}`, with every leaf already resolved against [`UnitHandles`].
    fn roots<'s>(self, vals: &mut Values, handles: UnitHandles<'s>) -> Vec<Scheduled<'s>>
    where
        'c: 's;
}

/// ONE OPENED PROGRAM UNIT — entries 084 and 093's two returns as the one shape their shared tail
/// reads (`:346-372` and `:434-460` are the same eleven lines twice).
struct Opened {
    /// The `dataflow.get_unit`s, which stand outside the region.
    bound: Vec<Op>,
    /// `unit_op.getUnits()`.
    on: Units,
    /// `program_unit_iterator`.
    iterator: Option<Val>,
    /// What the preamble inside the region bound.
    neighbours: Neighbourhood,
    /// `unit_to_value_map_`.
    handles: Option<Handles>,
}

/// ONE COMPONENT'S WHOLE PROGRAM UNIT — what entries 106 and 107 left in the module.
pub struct ConstructedProgramUnit<A: Arch> {
    /// The `dataflow.get_unit`s, which belong to the FUNCTION and not to the unit — `Program`'s
    /// preamble, because entry 084's builder is aimed at `dataflow_func_op_.back()`.
    ///
    /// ⭐ EMPTY WHERE THE SCHEDULE NAMED NO ROOT, because `:300` and `:382` return ahead of the
    /// initializer: the handles this port mints first are dropped rather than preambled.
    pub bound: Vec<Op>,
    /// The unit and its body, or [`None`] for the reference's three empty returns: no roots
    /// (`:300`, `:382`), no units bound, and a loop walk that failed.
    pub unit: Option<ProgramUnit<A>>,
    /// `program_unit_iterator` — the region argument every uniformized query in the body is keyed on.
    ///
    /// ⚠️ THE ISLAND HAS NOWHERE TO PUT IT, AND THAT IS A GAP THIS ENTRY SURFACES.
    /// `dataflow::Op::ProgramUnit` carries `iter_arg` and its printer spells
    /// `iter_arg : %arg -> (%units)` (`dialects/dataflow.rs:747-770`), but the typed [`ProgramUnit`]
    /// has no such field and `print.rs:68-100` never prints one — so a uniformized unit rebuilt as a
    /// typed unit loses the name its body still uses. Carrying it here keeps the value reachable
    /// instead of minting a second one; closing the gap means a field on [`ProgramUnit`], which
    /// ripples through every construction site including one this campaign may not edit.
    ///
    /// ⚠️ AND IT IS [`None`] ON THE ONE-FOLD ARM, where the reference reads
    /// `unit_op.getRegion().getArgument(0)` all the same (`:319`) — entry 084 emits `iter_arg: None`,
    /// so there is no argument to name. That arm sets `uniformization_enabled_ = false` (`:339`) and
    /// nothing under it queries the map, which is why the absence is observationally quiet.
    pub iterator: Option<Val>,
    /// Every diagnostic, IN THE ORDER THE REFERENCE PRINTS THEM.
    ///
    /// ⛔⛔ THE ORDER IS REASSEMBLED BY HAND, because the walks are inverted here. The reference
    /// prints the loop failure and its `terminate()` (`:347-350`) BEFORE the operation walk raises
    /// anything (`:361-364`); this port has already run that walk by the time the loops are asked
    /// for, so its sentences are appended AFTER the loop pair rather than in the order they arose.
    /// ⛔ AND THE TWO SHAPES ARE NOT INTERCHANGEABLE: the loop sentence and [`terminate`] go to
    /// `module_op_->emitError` unprefixed, while every [`Raised`] carries entry 010's prefix.
    pub raised: Vec<String>,
}

impl<A: Arch> ConstructedProgramUnit<A> {
    /// A component with no unit to build, keeping the handles that were bound before it was known.
    fn empty(bound: Vec<Op>) -> Self {
        Self {
            bound,
            unit: None,
            iterator: None,
            raised: Vec::new(),
        }
    }
}

/// THE ELEVEN LINES ENTRIES 106 AND 107 SHARE — `:346-372` and `:434-460`, which are identical.
///
/// ⛔ THE COMPONENT IS THE UNIT'S OWN (`on.kind()`) rather than a parameter beside it, so the unit
/// being built and the component being lowered for cannot disagree.
/// ⚠️ AND THE ROOT-LEVEL LEAF SITS ONE PLACE FURTHER IN THAN THE REFERENCE PUTS IT.
/// `builder.setInsertionPoint(dsc_loops_to_mlir_loops_map[nullptr].front())` (`:357`) aims JUST BEFORE
/// entry 096's synthetic `affine.for 0..1`, so a root-level leaf ahead of the first structural root is
/// textually OUTSIDE it there and inside it here. The loop runs exactly once, so the two are
/// semantically the same program; special-casing it would be inventing a placement rule the reference
/// does not state.
fn construct_a_units_program<'c, A: Arch, L, M, T: Clone, R: ScheduleView<'c>>(
    vals: &mut Values,
    opened: Opened,
    switching: SwitchInputs<'_, '_, L, M, T>,
    roots: R,
) -> ConstructedProgramUnit<A> {
    let Opened {
        bound,
        on,
        iterator,
        neighbours,
        handles,
    } = opened;
    let comp = on.kind();
    // The preamble is CLONED for the body exactly as entries 084 and 093 clone theirs, so the
    // `Neighbourhood` stays whole for the view that is about to read it.
    let preamble = neighbours.ops.clone();

    let scheduled = roots.roots(
        vals,
        UnitHandles {
            neighbours: &neighbours,
            handles: handles.as_ref(),
            iterator,
            units: &on,
        },
    );
    // `if (roots.empty()) return;` (`:300`, `:382`).
    //
    // ⛔⛔ AND THE HANDLES MINTED ABOVE ARE DROPPED, WHICH IS WHAT ENTRIES 108 AND 109 NEED. The
    // reference returns AHEAD of its initializer, so a component the schedule names no root for
    // leaves NOTHING in the function; this port has to mint them first to lower the view's leaves
    // against (see [`UnitHandles`]), and preambling them would put a `dataflow.get_unit` in every
    // program for each rootless (core, corelet, component) — ten to fourteen components, times both
    // corelets of every core, for the few that have a unit. The mint still consumed its SSA numbers.
    if scheduled.is_empty() {
        return ConstructedProgramUnit {
            bound: Vec::new(),
            unit: None,
            iterator,
            raised: Vec::new(),
        };
    }

    let (mut nodes, statements) = views(scheduled);
    // `std::string precision = ""` (`:360`, `:448`) — the empty spelling, which has no arm.
    let built = construct_operations_recursively::<A>(vals, statements, None);
    let precision = built.precision;
    let stopped = built.stopped;
    let mut sentences: Vec<String> = built.raised.iter().map(|one| one.diagnostic()).collect();
    fill(&mut nodes, &mut VecDeque::from(built.placed));

    let nest = construct_loops(
        vals,
        comp,
        switching.transfers,
        switching.head_children,
        &nodes,
    );
    drop(nodes);

    let mut raised = Vec::new();
    if nest.is_none() {
        // ⛔ UNPREFIXED, LIKE [`terminate`] AND UNLIKE EVERY [`Raised`]: `module_op_->emitError`
        // directly (`:348`, `:436`).
        raised.push("Unable to construct loops.".to_owned());
        raised.push(terminate().to_owned());
    }
    raised.append(&mut sentences);
    if stopped {
        raised.push(terminate().to_owned());
    }

    // ⛔⛔ AND `if (dsc_loops_to_mlir_loops_map.empty()) return;` (`:352-353`) IS DEAD CODE — a
    // CORRECTION to what entry 042's own note claims. `constructLoops` records the synthetic root
    // under `[nullptr]` before it can return `failure()`, so the map is never empty at that line and
    // the reference walks ON after a loops failure into `setPrecisionInUnitOp` on a unit whose loops
    // do not exist. [`None`] is still the faithful answer, because an `emitError` fails the pass and
    // there is no nest here to put a body into.
    let Some(nest) = nest else {
        return ConstructedProgramUnit {
            bound,
            unit: None,
            iterator,
            raised,
        };
    };

    let mut body = preamble;
    body.extend(nest.ops);
    let mut unit = ProgramUnit {
        on,
        precision: None,
        body,
        arch: PhantomData,
    };
    // ⛔⛔ PT ONLY, AND THIS GATE IS NARROWER THAN THE CALLEE'S. `is_any_of(record->second, PT)`
    // (`:369`, `:457`) admits PT alone, while entry 008 admits PT, PE and SFP — so a MAC's precision
    // reaching an SFP unit through here is DROPPED, and reading either gate as the whole rule would
    // write an attribute onto a unit the reference leaves bare.
    if let (Some(written), GenericComp::Pt) = (precision, comp.generic()) {
        set_precision_in_unit_op(&mut unit, written);
    }
    ConstructedProgramUnit {
        bound,
        unit: Some(unit),
        iterator,
        raised,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 108/110 · 109/110 — THE TWO MAIN LOOPS, AND WHAT ONE DSC HANDS THEM
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// `int num_corelets = 2` (`:470`, `:503`) — A LITERAL IN BOTH DRIVERS AND NOT AN ARCH READ: entry
/// 108 visits this many corelets of every core and entry 109 compares `numCoreletsUsed_DSC2_`
/// against it. It is [`crate::arch::Arch::CORELETS_PER_CORE`] on both arches this crate has.
const NUM_CORELETS: u32 = 2;

/// THE CORELETS ENTRY 108 ITERATES, the second of which is the one entry 109's temporary fix rebuilds.
///
/// ⭐ A BUILD-TIME GUARD, NOT A RUNTIME ONE: an arch with one corelet cannot spell this array, and
/// const evaluation is where such a build stops.
const BOTH_CORELETS: [Corelet; NUM_CORELETS as usize] =
    match (Corelet::checked(0), Corelet::checked(1)) {
        (Some(zero), Some(one)) => [zero, one],
        _ => panic!("the DCG's L3 code syncs with two corelets; this arch has fewer"),
    };

/// `1 /*corelet_id*/` (`:523`).
const CORELET_ONE: Corelet = BOTH_CORELETS[1];

/// WHICH `getNextView` OVERLOAD A COMPONENT'S ROOTS COME FROM — the one thing that tells the two
/// drivers' reads of the same schedule tree apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Viewed {
    /// `getNextView(comp, corelet_id, core_id)` (`:298`) — one pair, for entry 106.
    OnPair(Core, Corelet),
    /// `getNextView(comp)` (`:380`) — over every core and corelet the DSC uses, for entry 107.
    OverEveryPair,
}

/// ONE COMPONENT'S VIEW OF ONE DSC — the roots, and the buffer-switch state entry 096 reads and writes.
///
/// ⛔ THE SWITCH STATE IS OWNED PER CALL BECAUSE IT IS PER CALL IN THE REFERENCE:
/// `dsc_all_parent_loops_to_buffers_switch_map` is a LOCAL of `ConstructAProgramUnit` (`:331-332`),
/// declared fresh and dropped at the end, so a view that lent it out of the DSC would carry one
/// component's propagation into the next.
pub struct Viewing<'c, D: Dsc<'c> + ?Sized> {
    /// The component's own switching transfers, in the order that IS each one's [`TransferId`].
    pub transfers: Vec<SwitchingTransfer<'c, D::Latch, D::Mask>>,
    /// The schedule head's children, as this component sees them.
    pub head_children: Vec<SwitchNode<D::Switch>>,
    /// The roots, whose leaves are lowered against the unit's handles.
    pub roots: D::View,
}

/// ONE `DesignSpaceConfig` AS THE TWO DRIVERS READ IT — `sdsc_->dscs_`'s element, minus everything
/// only its own schedule view touches.
///
/// ⭐ EVERY METHOD IS `&self`. The reference holds the DSC by reference for the whole loop
/// (`for (auto &dsc : sdsc_->dscs_)`) and mutates translator state alone, which is what lets
/// [`Dsc::folds_needed`] stay callable while a view of the same DSC is in hand.
///
/// ⚠️ `: 'c` IS THE VIEWED DATA OUTLIVING ITS VIEWER, which is [`ScheduleView`]'s `'c: 's` one step
/// out: a DSC hands its own borrows to the leaves it builds, so it cannot hold anything shorter-lived
/// than the schedule they are read against.
pub trait Dsc<'c>: 'c {
    /// The loop a `bufferSwitchPosition_` names, as entry 074 identifies one.
    type Latch;
    /// A switching side's mask payload.
    type Mask;
    /// What a [`SwitchNode`] carries — one switching transfer's identity.
    type Switch: Clone;
    /// The schedule head's view of one component.
    type View: ScheduleView<'c>;

    /// `dsc.coreIdsUsed_`.
    ///
    /// ⛔ POSSIBLY EMPTY, AND THE TWO DRIVERS DIFFER ON THAT: entry 108 builds nothing at all for a
    /// DSC that names no core, while entry 109's uniformized pass still runs and ends in an empty
    /// `units_involved` — so this is not [`Used`], whose emptiness entry 005 rules out.
    fn cores(&self) -> &[Core];

    /// `dsc.numCoreletsUsed_DSC2_` — read by entry 109 only.
    fn num_corelets_used(&self) -> u32;

    /// `areFoldsNeeded(dsc, comp)` — entry 043's answer, asked only where the fold product exceeds one.
    fn folds_needed(&self, comp: DfirUnit) -> bool;

    /// `dsc.scheduleTree_.getHead()->getNextView(..)` (`:298`, `:380`), with what entry 096 needs.
    fn view(&self, comp: DfirUnit, at: Viewed) -> Viewing<'c, Self>;
}

/// WHAT ONE DRIVER LEFT IN THE MODULE — entry 003's scaffold, closed by entry 004 over every unit the
/// loops built.
pub struct Converted<A: Arch> {
    /// The program, or [`None`] where no component of any DSC had a unit.
    ///
    /// ⛔⛔ THAT ABSENCE IS dbo-opt's *"found no program to compile"* AND NOT A REFUSAL OF OURS. The
    /// reference closes the empty `func.func` all the same (`:493`, `:530`) and
    /// `AdaptSchedulerDfir.cpp:63-78` is what then rejects it; [`ProgramUnits`] cannot spell an empty
    /// list, so the state lives here instead of inside the program.
    pub program: Option<Program<A>>,
    /// `program_unit_iterator` PER BUILT UNIT, in [`ProgramUnits::iter`] order — see
    /// [`ConstructedProgramUnit::iterator`] for the island gap this keeps reachable.
    pub iterators: Vec<Option<Val>>,
    /// Every diagnostic, in the order the reference prints it.
    pub raised: Vec<String>,
}

/// THE MODULE AS THE LOOPS FILL IT — every `dataflow.get_unit` ahead of every unit, which is where
/// entries 084 and 093 aim their builder.
struct Assembled<A: Arch> {
    preamble: Vec<Op>,
    units: Vec<ProgramUnit<A>>,
    iterators: Vec<Option<Val>>,
    raised: Vec<String>,
}

impl<A: Arch> Assembled<A> {
    fn new() -> Self {
        Self {
            preamble: Vec::new(),
            units: Vec::new(),
            iterators: Vec::new(),
            raised: Vec::new(),
        }
    }

    /// One `ConstructA*ProgramUnit` call's result, where that call left it.
    fn push(&mut self, built: ConstructedProgramUnit<A>) {
        self.preamble.extend(built.bound);
        self.raised.extend(built.raised);
        if let Some(unit) = built.unit {
            self.units.push(unit);
            self.iterators.push(built.iterator);
        }
    }

    /// `this->stopDataflowIRGeneration()` over what the loops left.
    fn close(self, scaffold: Scaffold) -> Converted<A> {
        let mut units = self.units.into_iter();
        let program = units.next().map(|head| {
            stop_dataflow_ir_generation(
                scaffold,
                self.preamble,
                ProgramUnits::of(head, units.collect()),
            )
        });
        Converted {
            program,
            iterators: self.iterators,
            raised: self.raised,
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::super::compute::{
        ComputeFamily, ComputeInput, ComputeMask, ComputeOutput, MacInputFormat, MacOp,
        OperandContext, OutputFormat,
    };
    use super::super::construction::MaskValue;
    use super::super::dsc_lowering::{Handlers, Retrieved};
    use super::super::stick_mask::StickMaskView;
    use super::super::sync::{SyncKind, SyncUnits};
    use super::super::transfer::Latch;
    use super::{
        Component, CondStatement, ConstructedProgramUnit, Converted, Dsc, Emitted, FoldDimFunc,
        FoldedAddresses, Made, ParametricIters, Placed, Raised, ScheduleView, Scheduled,
        ScheduledCond, StartAddrOf, Statement, SwitchInputs, Transfer, UnitHandles, Used, Viewed,
        Viewing, construct_a_program_unit, construct_a_uniformized_program_unit,
        construct_operations_recursively, convert_v3, convert_v4, corelets_used,
        folded_addresses_are_same, folds_are_needed, start_dataflow_ir_generation,
        stop_dataflow_ir_generation, terminate,
    };
    use crate::arch::{Dd2, Elements};
    use crate::generated::{DataType, OpFunc, OpaqueFunc, SyncSignal};
    use crate::islands::dataflow_ir::dialects::agen::MaskCounts;
    use crate::islands::dataflow_ir::dialects::{Op, Val, affine, dataflow, uniform};
    use crate::islands::dataflow_ir::ty::{ElemType, GenericComp, TensorCategory, Vector};
    use crate::islands::dataflow_ir::{
        Grid, GroupId, OpIndex, ProgramName, ProgramUnit, ProgramUnits, Units, Values,
    };
    use crate::units::{Core, Corelet, DfirUnit, NumFolds, Row};
    use std::cell::RefCell;

    /// The component under test and one that is not it.
    fn units() -> (Component, Component) {
        (
            Component::Unit(DfirUnit::PtRow(Row::checked(0).expect("row 0"))),
            Component::Unit(DfirUnit::Lxlu),
        )
    }

    /// 🎯 042/110 — ⛔ THE MESSAGE CARRIES NO PREFIX, unlike every other diagnostic in the file.
    #[test]
    fn the_give_up_message_is_unprefixed() {
        assert_eq!(terminate(), "Unable to translate DSC2.0 to the Dataflow IR");
        assert!(
            !terminate().starts_with("[DSC2.0 to Dataflow IR]"),
            "`module_op_->emitError` is called directly, not through entry 010"
        );
        assert_ne!(
            terminate(),
            super::super::utils::error_diagnostic(terminate())
        );
    }

    /// 🎯 043/110 — ⛔⛔ THE CORELET LIST IS A TEST AGAINST 2, so 3 and 0 both give one corelet.
    #[test]
    fn the_corelet_list_is_a_comparison_not_a_range() {
        let zero = Corelet::checked(0).expect("corelet 0");
        let one = Corelet::checked(1).expect("corelet 1");
        assert_eq!(corelets_used(2), vec![zero, one]);
        assert_eq!(corelets_used(1), vec![zero]);
        // ⛔ NOT `0..n`: three corelets is still the list `{0}`.
        assert_eq!(corelets_used(3), vec![zero]);
        assert_eq!(corelets_used(0), vec![zero]);
    }

    /// 🎯 043/110 — ⭐ A NON-CONSTANT DIMENSION ENDS IT BEFORE ANY TRANSFER IS READ.
    #[test]
    fn a_non_constant_dim_func_short_circuits_the_transfer_walk() {
        let (comp, other) = units();
        let asked = RefCell::new(0_u32);
        let transfers = [Transfer {
            src: comp,
            dst_vias: &[other],
        }];
        assert!(folds_are_needed(
            2,
            &[FoldDimFunc::Constant, FoldDimFunc::Affine],
            &transfers,
            comp,
            |_, _| {
                *asked.borrow_mut() += 1;
                true
            },
        ));
        assert_eq!(*asked.borrow(), 0, "the constants are read first");

        // ⭐ AND ALL-CONSTANT DIMS FALL THROUGH TO THE WALK.
        assert!(!folds_are_needed(
            2,
            &[FoldDimFunc::Constant, FoldDimFunc::Constant],
            &transfers,
            comp,
            |_, _| {
                *asked.borrow_mut() += 1;
                true
            },
        ));
        assert_eq!(*asked.borrow(), 1, "the source end was asked once");
    }

    /// 🎯 043/110 — ⛔⛔ A TRANSFER SOURCED AT THIS COMPONENT NEVER HAS ITS VIAS EXAMINED, even when a
    /// via lands on the same component. This is the `if`/`else` written as a test.
    #[test]
    fn a_source_match_hides_the_vias_of_the_same_transfer() {
        let (comp, _) = units();
        let asked = RefCell::new(Vec::new());
        // Both ends are `comp`: the source answers, the via is never asked about.
        let transfers = [Transfer {
            src: comp,
            dst_vias: &[comp, comp],
        }];
        assert!(!folds_are_needed(0, &[], &transfers, comp, |which, _| {
            asked.borrow_mut().push(which);
            true
        }));
        assert_eq!(
            *asked.borrow(),
            vec![StartAddrOf::Source { transfer: 0 }],
            "the destination arm is an `else`"
        );
    }

    /// 🎯 043/110 — ⛔ EVERY VIA THAT LANDS ON THE COMPONENT IS ASKED, AND BY POSITION.
    #[test]
    fn each_matching_via_is_asked_about_its_own_index() {
        let (comp, other) = units();
        let asked = RefCell::new(Vec::new());
        let transfers = [
            Transfer {
                src: other,
                dst_vias: &[other, comp, comp],
            },
            Transfer {
                src: other,
                dst_vias: &[other],
            },
        ];
        assert!(!folds_are_needed(
            2,
            &[],
            &transfers,
            comp,
            |which, corelets| {
                asked.borrow_mut().push(which);
                // ⭐ THE DERIVED LIST REACHES THE CALLEE, which is what keeps the derivation live.
                assert_eq!(corelets.len(), 2);
                true
            }
        ));
        assert_eq!(
            *asked.borrow(),
            vec![
                StartAddrOf::Destination {
                    transfer: 0,
                    via: 1
                },
                StartAddrOf::Destination {
                    transfer: 0,
                    via: 2
                },
            ],
            "via 0 lands elsewhere and transfer 1 has no matching via"
        );
    }

    /// 🎯 043/110 — ⭐ ONE DISAGREEING ADDRESS IS ENOUGH, AND IT STOPS THE WALK.
    #[test]
    fn the_first_differing_address_ends_it() {
        let (comp, other) = units();
        let asked = RefCell::new(0_u32);
        let transfers = [
            Transfer {
                src: other,
                dst_vias: &[comp],
            },
            Transfer {
                src: comp,
                dst_vias: &[other],
            },
        ];
        assert!(folds_are_needed(2, &[], &transfers, comp, |_, _| {
            *asked.borrow_mut() += 1;
            false
        }));
        assert_eq!(*asked.borrow(), 1, "the walk returns on the first `false`");
    }

    /// 🎯 043/110 — ⭐ AND A COMPONENT NO TRANSFER TOUCHES NEEDS NO FOLDS: the loop body never runs
    /// and the answer is the reference's final `return false`.
    #[test]
    fn a_component_at_neither_end_needs_no_folds() {
        let (comp, other) = units();
        let transfers = [Transfer {
            src: other,
            dst_vias: &[other],
        }];
        assert!(!folds_are_needed(2, &[], &transfers, comp, |_, _| {
            unreachable!("no end names this component")
        }));
    }
    /// 🎯 003/110 · 🎯 004/110 — ⭐ THE PAIR IS ONE MODULE: the scaffold carries the symbol and the
    /// grid, and closing it over a non-empty unit list is what makes a program.
    #[test]
    fn the_scaffold_and_its_close_make_one_named_module() {
        let name = ProgramName {
            group: GroupId(4),
            index: OpIndex(2),
            func: OpFunc::Add,
        };
        let scaffold = start_dataflow_ir_generation(name, Grid::single());
        assert_eq!(scaffold.name, name);
        assert_eq!(scaffold.grid, Grid::single());

        let unit = ProgramUnit::<Dd2> {
            on: Units::one(DfirUnit::Sfp, Val(0)),
            precision: None,
            body: Vec::new(),
            arch: core::marker::PhantomData,
        };
        let program = stop_dataflow_ir_generation(
            scaffold,
            Vec::new(),
            ProgramUnits::of(unit.clone(), vec![]),
        );
        assert_eq!(program.name, name);
        assert_eq!(program.grid, Grid::single());
        assert!(program.preamble.is_empty());
        assert_eq!(program.units.iter().collect::<Vec<_>>(), vec![&unit]);
        // ⛔ THE SYMBOL IS THE PROGRAM'S, NOT THE LITERAL THE REFERENCE HARDCODES.
        assert_eq!(program.name.to_string(), "g4_2_add");
        assert_ne!(program.name.to_string(), "dataflowProgram");
    }

    /// 🎯 005/110 — ⛔ THE TEST IS `size != 1`, so one fold reads as constant from either variant and
    /// the first pair with more than one address ends the walk.
    #[test]
    fn one_address_per_pair_is_the_whole_question() {
        let cores = Used::of(
            Core::checked(0).expect("core 0"),
            vec![Core::checked(1).expect("core 1")],
        );
        let corelets = Used::of(
            Corelet::checked(0).expect("corelet 0"),
            vec![Corelet::checked(1).expect("corelet 1")],
        );

        let asked = RefCell::new(Vec::new());
        assert!(folded_addresses_are_same(
            &cores,
            &corelets,
            |core, corelet| {
                asked.borrow_mut().push((core.get(), corelet.get()));
                FoldedAddresses::Constant
            }
        ));
        assert_eq!(*asked.borrow(), vec![(0, 0), (0, 1), (1, 0), (1, 1)]);

        // ⭐ ONE FOLD IS THE SAME STATE TWICE — `is_any_of(1, 1, num_folds_)` with `num_folds_ == 1`.
        assert!(folded_addresses_are_same(&cores, &corelets, |_, _| {
            FoldedAddresses::PerFold(NumFolds::ONE)
        }));

        // ⛔ AND ELEVEN ADDRESSES IS NOT ONE ADDRESS: the walk stops on the first such pair.
        let count = RefCell::new(0_u32);
        assert!(!folded_addresses_are_same(&cores, &corelets, |_, _| {
            *count.borrow_mut() += 1;
            FoldedAddresses::PerFold(NumFolds(11))
        }));
        assert_eq!(*count.borrow(), 1);
    }

    /// THE VIEW EVERY STICK MASK STATEMENT BELOW IS BUILT FROM.
    fn mask_view() -> StickMaskView {
        StickMaskView {
            mask_a: MaskCounts {
                unmasked: 8,
                masked: 8,
            },
            mask_b: MaskCounts {
                unmasked: 1,
                masked: 1,
            },
            transition_slice: 5,
        }
    }

    /// 🎯 105/110 — ⛔ EACH ARM'S OWN INSERTION POINT, and the `mxfp8` a MAC leaves outliving the
    /// compute that follows it.
    ///
    /// A walk that assigned `getPrecision()` unconditionally would clear the unit's precision on the
    /// next non-MAC compute; one that wrapped a `BLOCK`'s children would place them inside a loop the
    /// reference erases.
    #[test]
    fn each_node_kind_lands_where_its_arm_puts_it_and_a_mac_precision_outlives_it() {
        let handlers = Handlers {
            units: Vec::new(),
            own_lrf: Val(1),
            pt_xrf: Val(2),
        };
        let result_ty = Vector {
            len: 64,
            elem: ElemType::F16,
        };
        let ctx = |name| OperandContext {
            name,
            comp: GenericComp::Sfp,
            ex_unit: GenericComp::Sfp,
            handlers: &handlers,
            result_ty,
        };
        let mac_ctx = ctx("fma8_0");
        let opaque_ctx = ctx("recip_0");
        let format = |elements, lds| MacInputFormat {
            lds,
            operand: DataType::Sen169Fp16,
            elements: Elements(elements),
        };
        let inputs = [
            (
                ComputeInput::One,
                format(64, Some((DataType::Sen143Fp8, TensorCategory::Scaled))),
            ),
            (ComputeInput::One, format(128, None)),
            (ComputeInput::Zero, format(32, None)),
        ];
        let outputs = [(
            ComputeOutput::Latch(Latch::new(3).expect("latch 3")),
            OutputFormat {
                lds: None,
                operand: DataType::Sen169Fp16,
            },
        )];

        let mut vals = Values::default();
        let built = construct_operations_recursively::<Dd2>(
            &mut vals,
            vec![
                Statement::Loop(vec![Statement::Compute {
                    ctx: &mac_ctx,
                    family: ComputeFamily::Mac {
                        mac: MacOp::Fma8,
                        inputs: &inputs,
                        mask: ComputeMask::Static(MaskValue::Live8),
                        outputs: &outputs,
                    },
                }]),
                Statement::Compute {
                    ctx: &opaque_ctx,
                    family: ComputeFamily::Opaque {
                        func: OpaqueFunc::Reciprocal,
                        read_write: &[],
                        read_only: &[],
                        params: &[],
                    },
                },
                Statement::Block(vec![Statement::StickMask {
                    view: mask_view(),
                    name: "samv_0",
                    format: DataType::Senint8,
                    mask_value: 3,
                }]),
                Statement::Condition(CondStatement::Loops {
                    then_: Box::new(Statement::Sync {
                        signal: SyncSignal::InputToLxsuToLxluToSync,
                        kind: SyncKind::Receive {
                            units: SyncUnits::Plain(vec![Retrieved::Reused(Val(9))]),
                        },
                    }),
                    otherwise: None,
                }),
            ],
            None,
        );

        assert!(!built.stopped);
        assert!(built.raised.is_empty());
        // ⛔ THE MAC'S PRECISION SURVIVES THE OPAQUE COMPUTE AFTER IT.
        assert_eq!(built.precision, Some(dataflow::Precision::Mxfp8));

        // The `BLOCK`'s child is spliced where its erased dummy loop stood; the loop and the `scf.if`
        // hold theirs.
        assert!(matches!(
            built.placed.as_slice(),
            [
                Placed::InLoop(inner),
                Placed::Leaf {
                    made: Made::Compute(Some(_)),
                    ..
                },
                Placed::Leaf {
                    made: Made::StickMask(Some(_)),
                    ..
                },
                Placed::InIf { then_, otherwise },
            ] if inner.len() == 1 && then_.len() == 1 && otherwise.is_empty()
        ));
    }

    /// 🎯 105/110 — ⛔⛔ A REFUSAL UNDER A LOOP IS SWALLOWED AND THE SAME ONE UNDER A `BLOCK` IS NOT,
    /// which is the whole propagation rule of this walk.
    ///
    /// Stopping on the loop's child would drop every sibling after it; not stopping on the block's
    /// would carry on emitting into a region whose contents are already wrong.
    #[test]
    fn a_refused_leaf_stops_a_block_frame_and_not_a_loop_frame() {
        let good = || Statement::StickMask {
            view: mask_view(),
            name: "samv_0",
            format: DataType::Senint8,
            mask_value: 3,
        };
        // ⚠️ `BOOL` IS ENTRY 060'S ONE REFUSAL.
        let bad = || Statement::StickMask {
            view: mask_view(),
            name: "samv_1",
            format: DataType::Bool,
            mask_value: 3,
        };

        let mut vals = Values::default();
        let swallowed = construct_operations_recursively::<Dd2>(
            &mut vals,
            vec![Statement::Loop(vec![bad()]), good()],
            None,
        );
        assert!(!swallowed.stopped);
        // The child's diagnostic reached the module even though its answer was discarded.
        assert_eq!(swallowed.raised, vec![Raised::StickMask]);
        assert_eq!(swallowed.placed.len(), 2);

        let stopped = construct_operations_recursively::<Dd2>(
            &mut vals,
            vec![Statement::Block(vec![good(), bad()]), good()],
            None,
        );
        assert!(stopped.stopped);
        assert_eq!(stopped.raised, vec![Raised::StickMask, Raised::Block]);
        // ⛔ THE OPS BUILT BEFORE THE REFUSAL STAY PLACED, and the sibling after the block never is.
        assert!(matches!(
            stopped.placed.as_slice(),
            [
                Placed::Leaf {
                    ops,
                    made: Made::StickMask(Some(_)),
                },
                Placed::Leaf {
                    made: Made::StickMask(None),
                    ..
                },
            ] if ops.len() == 2
        ));

        let conditionals = construct_operations_recursively::<Dd2>(
            &mut vals,
            vec![Statement::Condition(CondStatement::CoreCl(Box::new(bad())))],
            None,
        );
        assert!(conditionals.stopped);
        assert_eq!(
            conditionals.raised,
            vec![Raised::StickMask, Raised::ConditionalsOnConditionals]
        );
        assert_eq!(
            Raised::ConditionalsOnConditionals.diagnostic(),
            "[DSC2.0 to Dataflow IR]: Unable to construct conditionals on conditionals."
        );
    }
    /// THE ONE REGION ARGUMENT THE UNIFORM CONDITION BELOW IS ENTERED UNDER.
    const REGION_ARG: &[Val] = &[Val(77)];
    /// `getElseCoreCl(comp_)`'s units — the `then` branch has none, which is the child-index shift.
    const ELSE_UNITS: &[Val] = &[Val(88)];

    /// ONE STICK MASK UNDER A PARAMETRIC LOOP, THEN A UNIFORM CONDITION WHOSE `then` HAS NO UNIT — a
    /// view that borrows nothing, which is why it is a unit struct rather than a closure over
    /// fixtures.
    struct MaskInAParametricLoopThenAUniformElse;

    impl<'c> ScheduleView<'c> for MaskInAParametricLoopThenAUniformElse {
        fn roots<'s>(self, _vals: &mut Values, _handles: UnitHandles<'s>) -> Vec<Scheduled<'s>>
        where
            'c: 's,
        {
            let mask = |name| {
                Scheduled::Leaf(Emitted::StickMask {
                    view: mask_view(),
                    name,
                    format: DataType::Senint8,
                    mask_value: 3,
                })
            };
            vec![
                Scheduled::Parametric {
                    iters: ParametricIters::balanced(&[4, 4]).expect("balanced corelets"),
                    children: vec![mask("samv_0")],
                },
                Scheduled::Condition {
                    name: "cond_0",
                    place: ScheduledCond::Uniform {
                        args: REGION_ARG,
                        // ⛔ NO `then` UNIT, SO NO `then` REGION — one argument for two branches.
                        then_units: &[],
                        else_units: ELSE_UNITS,
                        then_: Box::new(mask("samv_1")),
                        otherwise: Some(Box::new(mask("samv_2"))),
                    },
                },
            ]
        }
    }

    /// ONE ROOT-LEVEL MAC — ⛔ THE FIXTURES ARE THE CALLER'S AND THE HANDLES ARE THE CALLEE'S, which is
    /// the whole reason [`ScheduleView`]'s `'c: 's` cannot be a `for<'s>` closure bound.
    #[derive(Clone, Copy)]
    struct MacAtTheRoot<'c> {
        ctx: &'c OperandContext<'c>,
        inputs: &'c [(ComputeInput<'c>, MacInputFormat); 3],
        outputs: &'c [(ComputeOutput<'c>, OutputFormat)],
    }

    impl<'c> ScheduleView<'c> for MacAtTheRoot<'c> {
        fn roots<'s>(self, _vals: &mut Values, _handles: UnitHandles<'s>) -> Vec<Scheduled<'s>>
        where
            'c: 's,
        {
            vec![Scheduled::Leaf(Emitted::Compute {
                ctx: self.ctx,
                family: ComputeFamily::Mac {
                    mac: MacOp::Fma8,
                    inputs: self.inputs,
                    mask: ComputeMask::Static(MaskValue::Live8),
                    outputs: self.outputs,
                },
            })]
        }
    }

    /// A COMPONENT THE SCHEDULE NAMES NO ROOT FOR — `roots.empty()` (`:300`, `:382`).
    struct NoRoots;

    impl<'c> ScheduleView<'c> for NoRoots {
        fn roots<'s>(self, _vals: &mut Values, _handles: UnitHandles<'s>) -> Vec<Scheduled<'s>>
        where
            'c: 's,
        {
            Vec::new()
        }
    }

    /// 🎯 106/110 — ⛔ THE HANDLES STAND OUTSIDE THE UNIT AND THE STATEMENT'S OPS STAND INSIDE THE LOOP
    /// entry 096 built for its node, which is the whole join of this entry's two walks.
    ///
    /// A `get_unit` left inside the region would name a unit from within itself; a leaf whose ops were
    /// appended after the nest instead of spliced into it would run OUTSIDE the loop that bounds it —
    /// and neither is visible to either walk alone.
    #[test]
    fn the_get_unit_stays_outside_and_the_leaf_lands_inside_its_loop() {
        let mut vals = Values::default();
        let built: ConstructedProgramUnit<Dd2> = construct_a_program_unit(
            &mut vals,
            DfirUnit::Lxlu,
            Core::checked(0).expect("core 0"),
            Corelet::checked(0).expect("corelet 0"),
            // ⭐ A PRODUCT OF ONE TAKES ENTRY 084'S ARM.
            &[NumFolds::ONE, NumFolds::ONE],
            SwitchInputs::<(), (), ()> {
                transfers: &[],
                head_children: &mut [],
            },
            MaskInAParametricLoopThenAUniformElse,
        );

        // ⛔ THE `dataflow.get_unit` IS A SIBLING OF THE UNIT, NOT ITS CONTENT.
        assert!(matches!(
            built.bound.as_slice(),
            [Op::Dataflow(dataflow::Op::GetUnit { .. })]
        ));
        // ⚠️ ENTRY 084 EMITS NO REGION ARGUMENT, so `:319`'s read has nothing to name.
        assert_eq!(built.iterator, None);
        assert!(built.raised.is_empty());

        let unit = built.unit.expect("one unit of one kind");
        assert_eq!(unit.on.kind(), DfirUnit::Lxlu);
        // ⭐ NO MAC ANYWHERE, SO NO PRECISION.
        assert_eq!(unit.precision, None);
        // The neighbour preamble stands first and the whole nest hangs inside the synthetic root.
        let Some(Op::Affine(affine::Op::For {
            hi: affine::Bound::Const(1),
            dbg_name: Some(root_name),
            body: root,
            ..
        })) = unit.body.last()
        else {
            unreachable!("entry 096 closes the nest in one `affine.for 0..1`")
        };
        assert_eq!(root_name, "synthetic_root");
        assert!(
            unit.body.len() > 1,
            "the neighbour preamble stands ahead of it"
        );

        // ⛔ THE STICK MASK'S OPS ARE INSIDE THE PARAMETRIC LOOP, which is the splice.
        assert!(matches!(
            root.first(),
            Some(Op::Affine(affine::Op::For {
                hi: affine::Bound::Const(4),
                dbg_name: None,
                body: inner,
                ..
            })) if !inner.is_empty()
        ));

        // ⛔⛔ THE ONE REGION IS THE `else` BRANCH'S, UNDER THE CALLER'S OWN ARGUMENT. A fresh mint
        // here would leave the leaves inside it naming a value the region does not declare, and the
        // child-index shift past a unitless `then` is what picks `children[1]` (`:5477-5479`).
        let Some(Op::Uniform(uniform::Op::UniformizeRegions { regions, .. })) = root.get(1) else {
            unreachable!("the condition node lowers to one `uniform.uniformize_regions`")
        };
        assert!(matches!(
            regions.as_slice(),
            [region] if region.arg == Val(77)
                && region.units == ELSE_UNITS
                && region.body.len() > 1
        ));
    }

    /// 🎯 107/110 — ⛔ THE FOLD COUNT COLLAPSES BACK TO ONE WHEN THE FOLDS ARE IDENTICAL, and the
    /// precision a MAC left reaches the unit ONLY on a PT component.
    ///
    /// Skipping the collapse would build a uniformized unit over folds the DSC does not need; reading
    /// entry 008's own PT/PE/SFP gate as the whole rule would write `precision =` onto an SFP unit the
    /// reference leaves bare.
    #[test]
    fn identical_folds_collapse_to_one_and_only_a_pt_unit_takes_the_precision() {
        let handlers = Handlers {
            units: Vec::new(),
            own_lrf: Val(1),
            pt_xrf: Val(2),
        };
        let ctx = OperandContext {
            name: "fma8_0",
            comp: GenericComp::Sfp,
            ex_unit: GenericComp::Sfp,
            handlers: &handlers,
            result_ty: Vector {
                len: 64,
                elem: ElemType::F16,
            },
        };
        let format = |elements, lds| MacInputFormat {
            lds,
            operand: DataType::Sen169Fp16,
            elements: Elements(elements),
        };
        let inputs = [
            (
                ComputeInput::One,
                format(64, Some((DataType::Sen143Fp8, TensorCategory::Scaled))),
            ),
            (ComputeInput::One, format(128, None)),
            (ComputeInput::Zero, format(32, None)),
        ];
        let outputs = [(
            ComputeOutput::Latch(Latch::new(3).expect("latch 3")),
            OutputFormat {
                lds: None,
                operand: DataType::Sen169Fp16,
            },
        )];
        let mac = MacAtTheRoot {
            ctx: &ctx,
            inputs: &inputs,
            outputs: &outputs,
        };
        let cores = [Core::checked(0).expect("core 0")];

        let mut vals = Values::default();
        let on_pt: ConstructedProgramUnit<Dd2> = construct_a_uniformized_program_unit(
            &mut vals,
            DfirUnit::PtRow(Row::checked(0).expect("row 0")),
            &cores,
            2,
            &[NumFolds(2)],
            // ⛔ ASKED, AND ITS `false` IS THE COLLAPSE.
            || false,
            SwitchInputs::<(), (), ()> {
                transfers: &[],
                head_children: &mut [],
            },
            mac,
        );

        let unit = on_pt.unit.expect("a unit over two corelets");
        // ⛔ TWO CORELETS AND ONE FOLD: `corelets_used(2)` gave two and the collapse gave one fold, so
        // a folded set would have been four handles here.
        assert_eq!(unit.on.vals().len(), 2);
        // ⛔ THE REGION ARGUMENT IS BOUND, and the island has no field for it — see the doc.
        assert!(on_pt.iterator.is_some());
        assert!(on_pt.raised.is_empty());
        assert_eq!(unit.precision, Some(dataflow::Precision::Mxfp8));

        // ⛔ THE SAME MAC ON AN SFP UNIT LEAVES IT BARE, because `:457` admits PT alone.
        let on_sfp: ConstructedProgramUnit<Dd2> = construct_a_uniformized_program_unit(
            &mut vals,
            DfirUnit::Sfp,
            &cores,
            2,
            &[NumFolds(2)],
            || false,
            SwitchInputs::<(), (), ()> {
                transfers: &[],
                head_children: &mut [],
            },
            mac,
        );
        assert_eq!(
            on_sfp.unit.expect("an sfp unit").precision,
            None,
            "entry 008 would have taken it; this gate does not hand it over"
        );

        // ⭐ AND A COMPONENT WITH NO ROOTS BUILDS NO UNIT, which is `:382`'s early return.
        let minted_before = vals.issued();
        let empty: ConstructedProgramUnit<Dd2> = construct_a_uniformized_program_unit(
            &mut vals,
            DfirUnit::Sfp,
            &cores,
            2,
            &[NumFolds::ONE],
            || unreachable!("one fold is never asked"),
            SwitchInputs::<(), (), ()> {
                transfers: &[],
                head_children: &mut [],
            },
            NoRoots,
        );
        assert!(empty.unit.is_none());
        // ⛔ AND NOTHING OF IT REACHES THE FUNCTION: the handles were minted to lower the view's
        // leaves against, but `:382` returns ahead of `initializeUniformizedUnit`, so entries 108 and
        // 109 must not preamble them. The mint itself still happened.
        assert!(empty.bound.is_empty());
        assert!(
            vals.issued() > minted_before,
            "the handles were minted all the same"
        );
    }

    /// A DSC WHOSE SCHEDULE NAMES ONE ROOT — a stick mask — FOR THE LXLU AND FOR NOTHING ELSE.
    struct MaskOnLxlu {
        /// `dsc.coreIdsUsed_`.
        cores: Vec<Core>,
        /// `dsc.numCoreletsUsed_DSC2_`.
        num_corelets_used: u32,
        /// Every `getNextView` this DSC was asked for, in order.
        asked: RefCell<Vec<(DfirUnit, Viewed)>>,
        /// Every `areFoldsNeeded` it was asked for.
        folds_asked: RefCell<Vec<DfirUnit>>,
    }

    impl MaskOnLxlu {
        fn on(cores: Vec<Core>, num_corelets_used: u32) -> MaskOnLxlu {
            MaskOnLxlu {
                cores,
                num_corelets_used,
                asked: RefCell::default(),
                folds_asked: RefCell::default(),
            }
        }
    }

    impl<'c> Dsc<'c> for MaskOnLxlu {
        type Latch = ();
        type Mask = ();
        type Switch = ();
        type View = MaskOn;

        fn cores(&self) -> &[Core] {
            &self.cores
        }

        fn num_corelets_used(&self) -> u32 {
            self.num_corelets_used
        }

        fn folds_needed(&self, comp: DfirUnit) -> bool {
            self.folds_asked.borrow_mut().push(comp);
            // ⛔ THE COLLAPSE, so the uniformized units come back with one fold.
            false
        }

        fn view(&self, comp: DfirUnit, at: Viewed) -> Viewing<'c, Self> {
            self.asked.borrow_mut().push((comp, at));
            Viewing {
                transfers: Vec::new(),
                head_children: Vec::new(),
                roots: MaskOn(comp == DfirUnit::Lxlu),
            }
        }
    }

    /// One component's roots: the stick mask, or none at all.
    struct MaskOn(bool);

    impl<'c> ScheduleView<'c> for MaskOn {
        fn roots<'s>(self, _vals: &mut Values, _handles: UnitHandles<'s>) -> Vec<Scheduled<'s>>
        where
            'c: 's,
        {
            if self.0 {
                vec![Scheduled::Leaf(Emitted::StickMask {
                    view: mask_view(),
                    name: "samv_0",
                    format: DataType::Senint8,
                    mask_value: 3,
                })]
            } else {
                Vec::new()
            }
        }
    }

    /// The program both drivers below open.
    fn program_name() -> ProgramName {
        ProgramName {
            group: GroupId(1),
            index: OpIndex(0),
            func: OpFunc::Add,
        }
    }

    /// The two components every driver below walks: one the schedule has a root for, one it does not.
    fn both_components() -> Used<DfirUnit> {
        Used::of(DfirUnit::Lxlu, vec![DfirUnit::Sfp])
    }

    /// 🎯 108/110 — ⛔⛔ THE PRODUCT IS DSC × CORE × BOTH CORELETS × COMPONENT, and a component the
    /// schedule names no root for leaves NO handle in the preamble.
    ///
    /// Iterating `numCoreletsUsed_DSC2_` corelets instead of two would build half the units; keeping
    /// a rootless component's minted `get_unit` would put four dead handles in this one program.
    #[test]
    fn both_corelets_of_every_core_and_a_rootless_component_binds_nothing() {
        let dsc = MaskOnLxlu::on(
            vec![
                Core::checked(0).expect("core 0"),
                Core::checked(1).expect("core 1"),
            ],
            // ⛔ ONE CORELET, AND ENTRY 108 IGNORES IT.
            1,
        );
        let mut vals = Values::default();
        let converted: Converted<Dd2> = convert_v3(
            &mut vals,
            program_name(),
            Grid::single(),
            &both_components(),
            &[NumFolds::ONE],
            core::slice::from_ref(&dsc),
        );

        // TWO CORES × BOTH CORELETS × TWO COMPONENTS, component-innermost.
        let asked = dsc.asked.borrow();
        assert_eq!(asked.len(), 8);
        let pair = |core, corelet| {
            Viewed::OnPair(
                Core::checked(core).expect("a core"),
                Corelet::checked(corelet).expect("a corelet"),
            )
        };
        assert_eq!(
            asked[..4],
            [
                (DfirUnit::Lxlu, pair(0, 0)),
                (DfirUnit::Sfp, pair(0, 0)),
                (DfirUnit::Lxlu, pair(0, 1)),
                (DfirUnit::Sfp, pair(0, 1)),
            ]
        );
        assert_eq!(asked[4], (DfirUnit::Lxlu, pair(1, 0)));
        // ⛔ ENTRY 108 NEVER CONSULTS ENTRY 043 — that is entry 109's alone.
        assert!(dsc.folds_asked.borrow().is_empty());

        let program = converted.program.expect("four units of one component");
        let built: Vec<&ProgramUnit<Dd2>> = program.units.iter().collect();
        assert_eq!(built.len(), 4);
        assert!(built.iter().all(|unit| unit.on.kind() == DfirUnit::Lxlu));
        // ⛔ FOUR HANDLES, NOT EIGHT: the SFP had no root, so nothing was bound for it.
        assert_eq!(program.preamble.len(), 4);
        assert!(
            program
                .preamble
                .iter()
                .all(|op| matches!(op, Op::Dataflow(dataflow::Op::GetUnit { .. })))
        );
        // ⭐ A PRODUCT OF ONE FOLD TAKES ENTRY 084'S ARM, which binds no region argument.
        assert_eq!(converted.iterators, vec![None; 4]);
        assert!(converted.raised.is_empty());
    }

    /// 🎯 109/110 — ⛔⛔ THE CORELET-1 PASS IS EXTRA AND UNCOLLAPSED: the uniformized unit asked entry
    /// 043 and folded back to one, while the fix pass never asks and stays folded.
    ///
    /// Reading the pass as a fallback would replace the uniformized unit instead of adding to it;
    /// running it for a DSC that already names both corelets would double every unit.
    #[test]
    fn the_corelet_one_fix_pass_is_extra_and_never_collapses_its_folds() {
        let core = Core::checked(0).expect("core 0");
        let dsc = MaskOnLxlu::on(vec![core], 1);
        let mut vals = Values::default();
        let converted: Converted<Dd2> = convert_v4(
            &mut vals,
            program_name(),
            Grid::single(),
            &both_components(),
            &[NumFolds(2)],
            core::slice::from_ref(&dsc),
        );

        assert_eq!(
            *dsc.asked.borrow(),
            vec![
                (DfirUnit::Lxlu, Viewed::OverEveryPair),
                (DfirUnit::Sfp, Viewed::OverEveryPair),
                (DfirUnit::Lxlu, Viewed::OnPair(core, super::CORELET_ONE)),
                (DfirUnit::Sfp, Viewed::OnPair(core, super::CORELET_ONE)),
            ]
        );
        // ⛔ ASKED ONCE PER COMPONENT BY THE UNIFORMIZED PASS AND NEVER BY THE FIX PASS, even though
        // that pass builds a unit of its own at the same fold product.
        assert_eq!(
            *dsc.folds_asked.borrow(),
            vec![DfirUnit::Lxlu, DfirUnit::Sfp]
        );

        let program = converted.program.expect("two units on the one component");
        let built: Vec<&ProgramUnit<Dd2>> = program.units.iter().collect();
        assert_eq!(built.len(), 2);
        // ⛔ ONE HANDLE COLLAPSED, TWO STILL FOLDED — `corelets_used(1)` × one core × one fold, then
        // one core × corelet 1 × the uncollapsed two folds.
        assert_eq!(built[0].on.vals().len(), 1);
        assert_eq!(built[1].on.vals().len(), 2);
        assert_eq!(program.preamble.len(), 2);
        // ⭐ BOTH TOOK ENTRY 093'S ARM, so both name a region argument.
        assert!(converted.iterators.iter().all(Option::is_some));

        // ⛔ AND A DSC THAT ALREADY NAMES BOTH CORELETS GETS NO FIX PASS AT ALL.
        let paired = MaskOnLxlu::on(vec![core], super::NUM_CORELETS);
        let converted: Converted<Dd2> = convert_v4(
            &mut vals,
            program_name(),
            Grid::single(),
            &both_components(),
            &[NumFolds(2)],
            core::slice::from_ref(&paired),
        );
        assert_eq!(
            *paired.asked.borrow(),
            vec![
                (DfirUnit::Lxlu, Viewed::OverEveryPair),
                (DfirUnit::Sfp, Viewed::OverEveryPair),
            ]
        );
        let units = converted.program.expect("the uniformized unit").units;
        assert_eq!(units.iter().count(), 1);
    }
}

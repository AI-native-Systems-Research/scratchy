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
use super::control_flow::PrimaryDim;
use super::dsc_lowering::{Component, Handlers};
use super::stick_mask::{StickMaskView, construct_stick_mask_operation};
use super::sync::{SyncKind, construct_sync_operation};
use super::transfer::{
    ConstructedTransfer, ContiguousSticks, DataTransfer, LoadAndSend, LoadAndStore,
    ReceiveAndStore, StickCounts, construct_data_transfer,
};
use super::utils::error_diagnostic;
use crate::arch::Arch;
use crate::generated::{DataType, SyncSignal};
use crate::islands::dataflow_ir::dialects::{Op, Val, dataflow};
use crate::islands::dataflow_ir::link::RecvEnd;
use crate::islands::dataflow_ir::{Grid, Program, ProgramName, ProgramUnits, Values};
use crate::units::{Core, Corelet, NumFolds};

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

// crustify:todo: e106_ConstructAProgramUnit
// crustify:todo: e107_ConstructAUniformizedProgramUnit
// crustify:todo: e108_convertV3
// crustify:todo: e109_convertV4
// crustify:todo: e110_runTranslator

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
        Component, CondStatement, FoldDimFunc, FoldedAddresses, Made, Placed, Raised, StartAddrOf,
        Statement, Transfer, Used, construct_operations_recursively, corelets_used,
        folded_addresses_are_same, folds_are_needed, start_dataflow_ir_generation,
        stop_dataflow_ir_generation, terminate,
    };
    use crate::arch::{Dd2, Elements};
    use crate::generated::{DataType, OpFunc, OpaqueFunc, SyncSignal};
    use crate::islands::dataflow_ir::dialects::agen::MaskCounts;
    use crate::islands::dataflow_ir::dialects::{Val, dataflow};
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
}

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

use super::dsc_lowering::Component;
use crate::arch::Arch;
use crate::islands::dataflow_ir::dialects::Op;
use crate::islands::dataflow_ir::{Grid, Program, ProgramName, ProgramUnits};
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

// crustify:todo: e105_constructOperationsRecursively
// crustify:todo: e106_ConstructAProgramUnit
// crustify:todo: e107_ConstructAUniformizedProgramUnit
// crustify:todo: e108_convertV3
// crustify:todo: e109_convertV4
// crustify:todo: e110_runTranslator

#[cfg(test)]
mod unit_tests {
    use super::{
        Component, FoldDimFunc, FoldedAddresses, StartAddrOf, Transfer, Used, corelets_used,
        folded_addresses_are_same, folds_are_needed, start_dataflow_ir_generation,
        stop_dataflow_ir_generation, terminate,
    };
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::dialects::Val;
    use crate::islands::dataflow_ir::{
        Grid, GroupId, OpIndex, ProgramName, ProgramUnit, ProgramUnits, Units,
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
}

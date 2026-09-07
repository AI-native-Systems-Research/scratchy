// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ CRUSTIFY BRIDGE-2 CAMPAIGN — READ THIS BEFORE YOU FILL AN ANCHOR IN THIS FILE.               ║
// ║ Full brief: crustify-bridge2/AGENT-BRIEF.md   ·   campaign statement: crustify-bridge2/TASK.md║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
//
// 1. THE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT.
//       /Users/nickm/git/deeptools-src/<file>:<line>        (deeptools @ a0d29abbed — repo_info.txt)
//    That is the revision every citation below resolves against. `crustify-bridge2/source/bridge2.cpp`
//    says WHICH functions are in scope and IN WHAT ORDER; ⛔ its bodies are TRUNCATED AT THE TAIL —
//    366 of the 384 end in a blank line and bare closing braces, and a 48-entry sample against the
//    authority found 21 that had lost real trailing statements (a `return success();`, a
//    `return rhs;`, an entire `} else { … }` branch, an `initMASData(...)` call). Port from the
//    authority file at the cited line. ⛔ /Users/nickm/git/deeptools is a DIFFERENT revision.
//    ⛔ The pod (/project_src/deeptools) is NOT reachable from this host — use the mirror above.
//
// 2. PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EMISSION. The op a function emits IS the
//    function — its exact attribute names and values, branch order and early returns. A documented
//    predicate that emits nothing is NOT a port (that is how the previous attempt failed). What you
//    MAY drop is only the mechanism for REACHING operands: use-walks, memoising by
//    (core, corelet, component), positioning an OpBuilder. If the target IR cannot express a
//    function's input, ADD THE OP to `src/islands/{sentient,dataflow_ir}/` — never decide the
//    function is unnecessary.
//
// 3. THIS IS A PURE-LOGIC PORT WITH NO C ANYWHERE. Whatever the generic C-to-Rust conventions say:
//    ❌ no bindgen/allowlist/-sys, ❌ no `ffi::`/`mod ffi_export`/`#[unsafe(no_mangle)] extern "C"`,
//    ❌ no `CRUSTIFY_<FILE>` switch, ❌ no `Foo`/`FooRef`/`FooMut` layout triple, ❌ no `unsafe`,
//    ❌ no sanitizers and no C-vs-Rust equivalence harness (there is no C to call).
//
// 4. CRATE RULES BIND YOU — `crates/compiler/deeptools/CLAUDE.md`, read it in full.
//    🛑 NEVER RUNTIME REFUSE: no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`
//    (frozen at zero by crates/targets/spyre/tests/dfir_never_runtime_refuses.rs). A closed set is
//    an `enum`; an invariant is a TYPE. `todo!("<op> …")` is tolerated, capped and ratcheted down —
//    and ⛔ never substitute a stand-in op to dodge one. Newtypes, never raw scalars. No strings for
//    closed sets. `Arch`/`Model`/`Workload` flow through as const generics.
//
// 5. ANCHORS: each `// crustify:todo: e<NNN>_<name>` below is one scheduled unit. Replace it with
//    the ported function carrying the doc anchor `/// Replaces: e<NNN>_<name>` on the item itself.
//    A surviving TODO is open work; the TODO must not survive beside the filled anchor.
//
// 6. TESTS: `#[cfg(test)] mod unit_tests` beside the code. 668 of the authority tree's 825
//    `dcc/test/**/*.mlir` cases carry `CHECK-SENT-IR` expectations — port the EXPECTATION, build the
//    typed input in Rust (this crate has no MLIR parser and must not get one).
//    `crates/compiler/deeptools/tests/sentient_corpus/` is the answer key (our DataflowIR beside the
//    reference's SentientIR for the same program).
//
// 7. GATE: `cargo check -p deeptools` and `cargo test -p deeptools`. ⛔ NEVER run the workspace or
//    acceptance build in an agent worktree — ~6 GB of target/ each and <50 GB free on this host.
//
// 8. `dataflow_ir_to_sentient/agen_to_sentient.rs` is the EARLIER PARTIAL ATTEMPT (predicates, no
//    emission, called by nothing). Nothing in it counts as ported; reuse what is right, but every
//    unit gets its own anchored item here.

//! `ProgramUnitsReduction.cpp` — 2 of bridge 2's 384 functions (dependency level(s) [1, 2]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e193_matchUnits` | 193/384 | 77 | `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:69` |
//! | `e256_runOnOperation` | 256/384 | 76 | `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:152` |

use super::tf_cfgs_dataflow_conditional_tree::{EquivalenceTag, OperationEquivalence};
use super::vc_vector_chain_helper::ops_are_equivalent;
use crate::arch::Arch;
use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, dataflow};
use crate::islands::dataflow_ir::{Program, ProgramUnit};
use crate::units::{Core, Corelet, DfirUnit, Residency};

/// WHAT `sentient::getCoreOrCoreletID` READS OFF A `dataflow.get_unit` — its `core` and its
/// `corelet`, each either a number or absent.
///
/// ```cpp
/// static int getCoreOrCoreletID(dataflow::GetUnitOp op, std::string attr) {
///   return op->hasAttr(attr)
///              ? mlir::dyn_cast<IntegerAttr>(op->getAttr(attr)).getInt()
///              : -1;
/// }
/// ```
///
/// (`dcc/src/Dialect/Sentient/SentientOps.hpp:76-80`. ⛔ NOT ONE OF THE 384 — it is a static inline in
/// a dialect header, and it is here because entry 193 reads all four of its answers.)
///
/// ⭐ `-1` IS `None`, NOT A NUMBER. The reference's sentinel exists because `getInt()` needs one; two
/// units that both lack a `core` attribute compare equal on `-1 == -1`, which is exactly
/// `None == None`, and there is no `-1` that can leak into an emitted attribute.
///
/// # ⛔⛔ THE PAIR IS COARSER THAN A [`Residency`], AND ENTRY 193 CAN SEE THE DIFFERENCE
///
/// [`Residency::CoreWide`] writes `core = c, corelet = 0` and [`Residency::Corelet`] with corelet 0
/// writes the same two attributes with the same two values (`dataflow.rs`'s printer,
/// `UnitMaterializer.cpp:62-80` against `:82-115`) — so those two residencies are ONE placement here.
/// That is not a lossy shortcut, it is what the reference compares: the four ids in
/// `unit_matching_info` are recorded from the program unit's own first `get_unit` and then matched
/// against `get_unit`s of *other* kinds inside the region, where the residency VARIANTS differ by
/// construction ([`crate::units::residency_of`] is a function of the kind alone). Comparing
/// [`Residency`] values instead would answer `false` for pairs the reference accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnitPlacement {
    /// `sentient::getCoreOrCoreletID(op, "core")`.
    pub core: Option<Core>,
    /// `sentient::getCoreOrCoreletID(op, "corelet")`.
    pub corelet: Option<Corelet>,
}

/// Corelet 0, which is the `corelet = 0` a [`Residency::CoreWide`] unit carries.
///
/// ⭐ A BUILD-TIME GUARD, NOT A RUNTIME ONE. An arch whose `CORELETS_PER_CORE` were zero could not
/// have a core-wide unit at all, and this stops that build in const evaluation rather than leaving a
/// hole for a lowering to fall through.
const CORELET_ZERO: Corelet = match Corelet::checked(0) {
    Some(corelet) => corelet,
    None => panic!("an arch with no corelet 0 cannot bind a core-wide unit"),
};

impl UnitPlacement {
    /// The `core` and `corelet` attributes a unit of this residency carries.
    #[must_use]
    pub const fn of(residency: Residency) -> UnitPlacement {
        match residency {
            // Neither attribute is written, so both reads are the reference's `-1`.
            Residency::Global => UnitPlacement {
                core: None,
                corelet: None,
            },
            // `core` and no `corelet`.
            Residency::Scratchpad { core } => UnitPlacement {
                core: Some(core),
                corelet: None,
            },
            // `core` AND `corelet = 0`.
            Residency::CoreWide { core } => UnitPlacement {
                core: Some(core),
                corelet: Some(CORELET_ZERO),
            },
            Residency::Corelet { core, corelet } => UnitPlacement {
                core: Some(core),
                corelet: Some(corelet),
            },
        }
    }
}

/// `struct unit_matching_info` — the four ids the functor reads through its `void *context`
/// (`dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:85-97`).
///
/// ⭐ TWO PLACEMENTS, NOT FOUR INTS, and no heap. The reference `new`s this so a captureless lambda
/// can reach it and `delete`s it on both exits (`:99-101`, `:141`, `:145`); the pointer is the
/// mechanism, and a [`Copy`] value carried in the [`HighPreference`] is the same fact without the two
/// deletes to keep in step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnitMatchingInfo {
    /// `base_unit_core_id_` and `base_unit_corelet_id_`.
    pub base_unit: UnitPlacement,
    /// `curr_unit_core_id_` and `curr_unit_corelet_id_`.
    pub curr_unit: UnitPlacement,
}

/// `dcc::OperationEquivalence`'s `functor_` AND THE `context_` IT READS — the positive-only override.
///
/// # ⭐ POSITIVE-ONLY, AND THAT IS WHY IT IS "HIGH PREFERENCE"
///
/// `OperationEquivalence.cpp:115-118` is `if (functor_ && functor_(op_a, op_b, context_)) return
/// true;` under the comment *"Functor carries higher precedence as it is user-controlled."* — a
/// functor can only make two operations MORE equivalent, never less. It sits after the
/// pointer-identity and equivalence-class checks and before the dialect comparison, so everything
/// structural below it is skipped when it fires.
///
/// ⛔ A CLOSED SET, NOT A CLOSURE. Exactly one functor exists in the whole reference — the lambda at
/// `ProgramUnitsReduction.cpp:112-135`. Six other sites reach the same constructor and pass
/// `nullptr, nullptr` (`CFGSDataflowConditionalTree.hpp:112`,
/// `CFGSSentientLevelConditionalTree.hpp:316`, `CFGDeepMergingConditionalTree.hpp:29`,
/// `LoopMerging.cpp:58`, `LoopAbsorption.cpp:43`, `LoopRolling.cpp:987`), and nine more take a
/// constructor with no functor parameter at all, leaving the `function_ref` default-constructed and
/// so null (`UtilsHelper.cpp:127`, `:621`, `:696`, `Utils.cpp:34`, `:707`, `:1267`,
/// `UniformInstrAndBlock.cpp:92`, `VectorChainHelper.cpp:426` — entry 065's own — and
/// `ConditionalTree.hpp:184`'s member). The crate's rule that a closed set is an `enum` applies, and
/// it also keeps [`OperationEquivalence`] `Copy` and comparable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HighPreference {
    /// `functor_ == nullptr` — the comparison is structural all the way down.
    None,
    /// `ProgramUnitsReduction.cpp:112-135` — the lambda that makes core and corelet ids parametric.
    UnitsAreParametric(UnitMatchingInfo),
}

impl HighPreference {
    /// `functor_(op_a, op_b, context_)`.
    ///
    /// ```cpp
    /// [](Operation &op_a, Operation &op_b, void *context) -> bool {
    ///   unit_matching_info *info = (unit_matching_info *)context;
    ///   if (isa<GetUnitOp>(op_a) && isa<GetUnitOp>(op_b)) {
    ///     auto operation_a = llvm::dyn_cast<GetUnitOp>(op_a);
    ///     auto operation_b = llvm::dyn_cast<GetUnitOp>(op_b);
    ///     int a_core_id = sentient::getCoreOrCoreletID(operation_a, "core");
    ///     int a_corelet_id = sentient::getCoreOrCoreletID(operation_a, "corelet");
    ///     if (a_core_id == info->base_unit_core_id_ &&
    ///         a_corelet_id == info->base_unit_corelet_id_) {
    ///       int b_core_id = sentient::getCoreOrCoreletID(operation_b, "core");
    ///       int b_corelet_id = sentient::getCoreOrCoreletID(operation_b, "corelet");
    ///       if (b_core_id == info->curr_unit_core_id_ &&
    ///           b_corelet_id == info->curr_unit_corelet_id_) {
    ///         if (operation_a.getType() == operation_b.getType()) {
    ///           return true;
    ///         }
    ///       }
    ///     }
    ///   }
    ///   return false;
    /// }
    /// ```
    ///
    /// ⭐⭐ IT IS DIRECTIONAL. `a` must be where the group's BASE lives and `b` where the CANDIDATE
    /// does — the reference's own comment: *"If base_unit uses its default core_id in statement S,
    /// then the curr_unit should use its core_id in the statement S."* Swapping the two tests would
    /// accept a base that reads the candidate's core, which is a different program.
    ///
    /// ⛔ AND `getType()` IS THE `type` ATTRIBUTE, NOT A RESULT TYPE. `get_unit`'s ODS declares
    /// `(ins StrAttr:$name, StrAttr:$type)` with `Variadic<Index>` results
    /// (`dataflow-scheduler/.../Dataflow.td:54-58`), so the generated `getType()` returns the `type`
    /// string — which is a [`DfirUnit`] here. `name` is NOT compared, and it must not be: the vendor's
    /// own `mixed.mlir` merges `"l0lurow0-CL0"` into a group based on `"l0lurow0-CL1"`.
    #[must_use]
    pub fn prefers(self, op_a: &DfirOp, op_b: &DfirOp) -> bool {
        let HighPreference::UnitsAreParametric(info) = self else {
            return false;
        };

        // `isa<GetUnitOp>(op_a) && isa<GetUnitOp>(op_b)`, and the `dyn_cast`s that follow it.
        let (
            DfirOp::Dataflow(dataflow::Op::GetUnit {
                residency: residency_a,
                unit: unit_a,
                ..
            }),
            DfirOp::Dataflow(dataflow::Op::GetUnit {
                residency: residency_b,
                unit: unit_b,
                ..
            }),
        ) = (op_a, op_b)
        else {
            return false;
        };

        UnitPlacement::of(*residency_a) == info.base_unit
            && UnitPlacement::of(*residency_b) == info.curr_unit
            && unit_a == unit_b
    }
}

/// A GROUP OF `dataflow.program_unit`s THAT SHARE ONE CODE STRUCTURE — `ReducibleProgramUnits`
/// (`dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:39-48`).
///
/// ```cpp
/// struct ReducibleProgramUnits {
///   ProgramUnitOp base_unit_program_;
///   llvm::SmallVector<GetUnitOp, 8> units_list_;
///
///   ReducibleProgramUnits(ProgramUnitOp base) : base_unit_program_(base) {
///     for (auto unit : base.getUnits()) {
///       units_list_.push_back(unit.getDefiningOp<GetUnitOp>());
///     }
///   }
/// };
/// ```
///
/// ⛔ NOT ONE OF THE 384, and it is here because entry 193's first parameter is one of these. The
/// group starts out holding exactly the base's own units and entry 256 appends each matched
/// candidate's (`:190-193`), which is what the pass finally writes back as the surviving unit's
/// operands (`:213-221`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReducibleProgramUnits<'p, A: Arch> {
    /// `base_unit_program_` — the unit every candidate is compared against, borrowed because the
    /// reference holds an `OpBuilder` handle to an operation it does not own.
    pub base_unit_program: &'p ProgramUnit<A>,
    /// `units_list_` — the `dataflow.get_unit`s whose work this group's base now performs.
    pub units_list: Vec<Val>,
}

impl<'p, A: Arch> ReducibleProgramUnits<'p, A> {
    /// A new group over one base unit — the reference's constructor, whose whole body is copying the
    /// base's own operands into `units_list_`.
    #[must_use]
    pub fn of(base: &'p ProgramUnit<A>) -> ReducibleProgramUnits<'p, A> {
        ReducibleProgramUnits {
            base_unit_program: base,
            units_list: base.on.vals(),
        }
    }
}

/// THE TWO ANSWERS ENTRY 193 GIVES — `LogicalResult`.
///
/// ⭐ NOT A `Result`, AND NOT A REFUSAL EITHER WAY. `failure()` here means *"start a new group"*
/// (`ProgramUnitsReduction.cpp:203-209`); the pass only ever merges units it has proved equivalent,
/// so the negative answer costs a reduction and cannot change a program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitMatch {
    /// `LogicalResult::success()` — the candidate's work is the base's work, up to core and corelet.
    Matched,
    /// `LogicalResult::failure()`.
    Differs,
}

/// EVERY OP A [`Val`] MAY BE DEFINED BY, in one list — what the reference's `getDefiningOp()` reaches.
///
/// ⭐ THE MECHANISM FOR REACHING OPERANDS, which is the one thing the campaign brief lets a port
/// build its own way. MLIR's `Value` knows its own defining operation; this island's [`Val`] is an
/// index, so the ops have to be handed over. A program's preamble binds the units and views, and each
/// program unit's body binds the rest, so the two together are the module.
#[must_use]
pub fn module_scope<A: Arch>(program: &Program<A>) -> Vec<DfirOp> {
    let mut scope = program.preamble.clone();
    for unit in program.units.iter() {
        scope.extend(unit.body.iter().cloned());
    }
    scope
}

/// WHAT A `dataflow.get_unit` SAYS ABOUT ITS UNIT — the two things entry 193 reads off one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GetUnitBinding {
    /// What `getCoreOrCoreletID` answers for `"core"` and `"corelet"`.
    placement: UnitPlacement,
    /// `getType()`, the `type` attribute.
    unit: DfirUnit,
}

/// `unit.getUnits()[0].getDefiningOp<GetUnitOp>()`.
///
/// ⚠️ `None` IS A PROGRAM THIS ISLAND'S OWN EMITTER CANNOT BUILD — [`crate::islands::dataflow_ir::Units`]
/// is constructed from `dataflow.get_unit` bindings — and it is where the reference is at its least
/// defensible: `getDefiningOp<GetUnitOp>()` returning null is then dereferenced by
/// `getCoreOrCoreletID`'s `op->hasAttr(attr)`, which is a segfault with no diagnostic. The caller
/// answers [`UnitMatch::Differs`], the conservative direction.
fn head_get_unit<A: Arch>(unit: &ProgramUnit<A>, scope: &[DfirOp]) -> Option<GetUnitBinding> {
    let DfirOp::Dataflow(dataflow::Op::GetUnit {
        residency,
        unit: kind,
        ..
    }) = crate::islands::dataflow_ir::dialects::defining_op(unit.on.first(), scope)?
    else {
        return None;
    };
    Some(GetUnitBinding {
        placement: UnitPlacement::of(*residency),
        unit: *kind,
    })
}

/// `dcc::OperationEquivalence::regionsAreEquivalent` over the ONE BLOCK a `dataflow.program_unit`
/// region has (`dcc/src/Analysis/OperationEquivalence.cpp:65-83` calling `:25-63`).
///
/// ⛔ NOT AN ANCHORED UNIT — `dcc/src/Analysis/` contributes nothing to this campaign's 384 — and it
/// is here because entry 193 is the whole of its own answer only once this is called.
///
/// ⭐ THE BLOCK-ARGUMENT CHECKS ARE VACUOUS HERE AND THAT IS A FACT, NOT AN OMISSION. `:32-46`
/// compares argument counts and then argument types; a `dataflow.program_unit` region in this island
/// carries no arguments at all, so both regions bring zero and the counts and the type walk agree
/// trivially. (Entry 256 is what *adds* an `index` argument to the surviving base, at `:222-225` —
/// after every comparison this function performs.)
///
/// ⚠️ AND `range_size(region)` — THE BLOCK COUNT — IS 1 ON BOTH SIDES. A `program_unit` body is one
/// block; `body: Vec<Op>` is that block, and the loop below is the reference's single
/// `blocksAreEquivalent` call.
fn regions_are_equivalent(
    block_a: &[DfirOp],
    block_b: &[DfirOp],
    scope: &[DfirOp],
    oe: &OperationEquivalence,
) -> bool {
    // `if (utils::range_size(block_a) != utils::range_size(block_b))` — *"block sizes don't match"*.
    if block_a.len() != block_b.len() {
        return false;
    }

    // `for (auto pair : llvm::zip(block_a, block_b))`.
    block_a
        .iter()
        .zip(block_b)
        .all(|(op_a, op_b)| ops_are_equivalent(op_a, op_b, scope, oe.preference))
}

/// Replaces: e193_matchUnits
///
/// **193/384** `ProgramUnitsReductionPass::matchUnits` —
/// `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:69` (77L).
///
/// ```cpp
/// LogicalResult ProgramUnitsReductionPass::matchUnits(
///     ReducibleProgramUnits &group, ProgramUnitOp &curr_unit) {
///   auto base_get_unit =
///       group.base_unit_program_.getUnits()[0].getDefiningOp<GetUnitOp>();
///   auto curr_get_unit = curr_unit.getUnits()[0].getDefiningOp<GetUnitOp>();
///   int base_unit_core_id = sentient::getCoreOrCoreletID(base_get_unit, "core");
///   int base_unit_corelet_id =
///       sentient::getCoreOrCoreletID(base_get_unit, "corelet");
///   int curr_unit_core_id = sentient::getCoreOrCoreletID(curr_get_unit, "core");
///   int curr_unit_corelet_id =
///       sentient::getCoreOrCoreletID(curr_get_unit, "corelet");
///
///   // Check if units have same type (e.g., ptrow0)
///   if (base_get_unit.getType() != curr_get_unit.getType())
///     return LogicalResult::failure();
///
///   struct unit_matching_info { /* four ints, `:85-97` */ };
///
///   unit_matching_info *info =
///       new unit_matching_info(base_unit_core_id, base_unit_corelet_id,
///                              curr_unit_core_id, curr_unit_corelet_id);
///
///   dcc::OperationEquivalence equivalence_analysis(
///       /* the high-preference lambda, `:112-135` */,
///       (void *)info, "program-units-reduction");
///
///   // Check if regions of base unit and curr unit are same
///   if (!equivalence_analysis.regionsAreEquivalent(
///           group.base_unit_program_.getRegion(), curr_unit.getRegion())) {
///     delete info;
///     return LogicalResult::failure();
///   }
///
///   delete info;
///   return LogicalResult::success();
/// }
/// ```
///
/// # ⭐⭐ WHAT THIS DECIDES: FOUR PROGRAM UNITS BECOME TWO, AND WHICH TWO IS NOT OBVIOUS
///
/// `dcc/test/Transform/ProgramUnitsReduction/mixed.mlir` is the answer key. Its input is four
/// `l0lurow0` units — `(core 0, corelet 0)`, `(0, 1)`, `(1, 0)`, `(1, 1)` at `:201`, `:286`, `:371`,
/// `:456` — whose bodies bind an `l0su`, a `ptrow0` and an `l0` each. Three of them bind all three at
/// their own `(core, corelet)`; the odd one out is `(core 0, corelet 1)`, whose `l0` is at
/// `{core = 0, corelet = 0}` (`:290` against `:206`, `:375`, `:460`).
///
/// Entry 256 walks the units in REVERSE (`:181-184`), so `(1, 1)` becomes the first group's base:
///
/// | candidate | vs base `(1, 1)` | answer |
/// |---|---|---|
/// | `(1, 0)` | every inner unit sits at the candidate's own corelet | [`UnitMatch::Matched`] |
/// | `(0, 1)` | its `l0` is at `(0, 0)`, which is not `(0, 1)` | [`UnitMatch::Differs`] |
/// | `(0, 0)` | every inner unit sits at `(0, 0)` | [`UnitMatch::Matched`] |
///
/// and `(0, 1)` opens a second group of its own. The reference's own `CHECK-SENT-IR` at `:11` and
/// `:97` is exactly that: one surviving unit over `(%1)` and one over `(%3, %2, %0)`. ⛔ SO
/// `(core 0, corelet 0)` LANDS IN `(core 1, corelet 1)`'s GROUP, NOT IN ITS OWN CORE'S — the placement
/// is parametric, and a port that compared cores would have put it with `(0, 1)` and emitted a
/// program where one corelet does another's work.
///
/// # ⛔ THE THREE BOOLS ARE THE CONSTRUCTOR'S DEFAULTS, AND ONE OF THEM MATTERS
///
/// `:103-136` names `functor`, `extent` and `debug` and stops, so `do_recursive_compare`,
/// `all_block_args_are_equiv` and `use_equiv_classes` are all `true`
/// (`OperationEquivalence.hpp:36-45`). ⭐ `all_block_args_are_equiv = true` is what lets two units
/// compare at all: their loop induction variables and iteration arguments are region arguments of
/// DIFFERENT blocks, which `operator==` always calls unequal (`OperationEquivalence.cpp:30-32`'s own
/// comment). The conditional trees ask for `false` here; this pass must not.
///
/// ⚠️ AND `use_equiv_classes = true` IS THE ONE PART DROPPED, as everywhere else in this crate: the
/// memo only short-circuits a repeated question (`OperationEquivalence.cpp:107-113`), it cannot answer
/// one differently. See [`ops_are_equivalent`].
///
/// ⛔ THE TWO `delete info` CALLS HAVE NO COUNTERPART, and that is the point of carrying
/// [`UnitMatchingInfo`] by value: the reference has one `new` and two frees on two exit paths, and a
/// third exit added later leaks.
#[must_use]
pub fn match_units<A: Arch>(
    group: &ReducibleProgramUnits<'_, A>,
    curr_unit: &ProgramUnit<A>,
    scope: &[DfirOp],
) -> UnitMatch {
    let base_unit_program = group.base_unit_program;

    // `getUnits()[0].getDefiningOp<GetUnitOp>()`, twice, and the four `getCoreOrCoreletID` reads that
    // follow — which are one placement each.
    let (Some(base_get_unit), Some(curr_get_unit)) = (
        head_get_unit(base_unit_program, scope),
        head_get_unit(curr_unit, scope),
    ) else {
        return UnitMatch::Differs;
    };

    // `if (base_get_unit.getType() != curr_get_unit.getType()) return failure();` — *"Check if units
    // have same type (e.g., ptrow0)"*.
    if base_get_unit.unit != curr_get_unit.unit {
        return UnitMatch::Differs;
    }

    let info = UnitMatchingInfo {
        base_unit: base_get_unit.placement,
        curr_unit: curr_get_unit.placement,
    };
    let equivalence_analysis = OperationEquivalence::preferring(
        HighPreference::UnitsAreParametric(info),
        EquivalenceTag::ProgramUnitsReduction,
    );

    // `if (!equivalence_analysis.regionsAreEquivalent(...)) return failure(); return success();`
    if regions_are_equivalent(
        &base_unit_program.body,
        &curr_unit.body,
        scope,
        &equivalence_analysis,
    ) {
        UnitMatch::Matched
    } else {
        UnitMatch::Differs
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::generated::{OpFunc, SyncSignal};
    use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, dataflow};
    use crate::islands::dataflow_ir::{
        Grid, GroupId, OpIndex, ProgramName, ProgramUnit, ProgramUnits, Units,
    };
    use crate::units::{Core, Corelet, DfirUnit, Residency, Row};

    /// `{core = <core> : i32, corelet = <corelet> : i32}`.
    fn at(core: u32, corelet: u32) -> Residency {
        Residency::Corelet {
            core: Core::checked(core).expect("this arch has the core the fixture names"),
            corelet: Corelet::checked(corelet)
                .expect("this arch has the corelet the fixture names"),
        }
    }

    /// Row 0 of the PT — the vendor's `type = "ptrow0"`.
    fn row0() -> Row {
        Row::checked(0).expect("every arch's PT has a row 0")
    }

    fn get_unit(result: u32, residency: Residency, unit: DfirUnit) -> DfirOp {
        DfirOp::Dataflow(dataflow::Op::GetUnit {
            result: Val(result),
            residency,
            unit,
            num_folds: None,
        })
    }

    /// ONE UNIT'S BODY, in the shape `mixed.mlir:203-205` gives it: an `l0su`, a `ptrow0` and an `l0`
    /// binding followed by work that reads one of them.
    ///
    /// `own` is where the unit itself lives and `l0` is where its scratchpad binding says the `l0` is
    /// — separate parameters because the vendor's `(core 0, corelet 1)` unit is exactly the case where
    /// they disagree (`:290`).
    fn body(first: u32, own: Residency, l0: Residency) -> Vec<DfirOp> {
        vec![
            get_unit(first, own, DfirUnit::L0su),
            get_unit(first + 1, own, DfirUnit::PtRow(row0())),
            get_unit(first + 2, l0, DfirUnit::L0),
            // `dataflow.sync_send %l0su` and `dataflow.sync_recv %l0su` (`mixed.mlir:16`, `:30`) —
            // two ops whose operand is a `get_unit` the walk has to resolve rather than compare in
            // place.
            DfirOp::Dataflow(dataflow::Op::SyncSend {
                to: Val(first),
                signal: SyncSignal::InputToLxsuToLxluToSync,
            }),
            DfirOp::Dataflow(dataflow::Op::SyncRecv {
                from: Val(first),
                signal: SyncSignal::InputToLxsuToLxluToSync,
            }),
        ]
    }

    fn unit_on(bound: u32, kind: DfirUnit, body: Vec<DfirOp>) -> ProgramUnit<Dd2> {
        ProgramUnit {
            on: Units::one(kind, Val(bound)),
            precision: None,
            body,
            arch: core::marker::PhantomData,
        }
    }

    /// THE VENDOR'S `mixed.mlir` INPUT: four `l0lurow0` units at `(0,0)`, `(0,1)`, `(1,0)` and
    /// `(1,1)`, of which `(0,1)` binds its `l0` at `(0,0)` rather than at its own corelet.
    ///
    /// ⚠️ `l0lurow0` HAS NO SPELLING IN THIS ISLAND'S [`DfirUnit`], so the outer kind here is `l0lu`.
    /// Entry 193 never reads the kind except to compare the two units' kinds for equality, so the
    /// substitution cannot move an answer — and every placement, every inner kind and every expected
    /// answer below is the vendor's.
    fn mixed_program() -> Program<Dd2> {
        let preamble = vec![
            get_unit(0, at(0, 0), DfirUnit::L0lu),
            get_unit(1, at(0, 1), DfirUnit::L0lu),
            get_unit(2, at(1, 0), DfirUnit::L0lu),
            get_unit(3, at(1, 1), DfirUnit::L0lu),
        ];
        let units = ProgramUnits::of(
            unit_on(0, DfirUnit::L0lu, body(10, at(0, 0), at(0, 0))),
            vec![
                unit_on(1, DfirUnit::L0lu, body(20, at(0, 1), at(0, 0))),
                unit_on(2, DfirUnit::L0lu, body(30, at(1, 0), at(1, 0))),
                unit_on(3, DfirUnit::L0lu, body(40, at(1, 1), at(1, 1))),
            ],
        );
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            grid: Grid::single(),
            preamble,
            units,
            arch: core::marker::PhantomData,
        }
    }

    /// The four units of [`mixed_program`], in the order `mixed.mlir` declares them.
    fn mixed_units(program: &Program<Dd2>) -> Vec<&ProgramUnit<Dd2>> {
        program.units.iter().collect()
    }

    /// 🎯 193/384 — THE OTHER CORELET OF ONE CORE JOINS THE GROUP.
    ///
    /// `mixed.mlir` reduces four units to two and `(core 1, corelet 0)` is the first candidate the
    /// reverse walk offers to `(core 1, corelet 1)`'s group; the vendor's surviving unit runs on
    /// `(%3, %2, %0)` (`:97`), so this pair matched.
    #[test]
    fn the_other_corelet_of_the_same_core_matches() {
        let program = mixed_program();
        let units = mixed_units(&program);
        let scope = module_scope(&program);
        let group = ReducibleProgramUnits::of(units[3]);
        assert_eq!(
            match_units(&group, units[2], &scope),
            UnitMatch::Matched,
            "(1,0) binds every inner unit at its own corelet, which is what makes it parametric"
        );
    }

    /// 🎯 193/384 — THE UNIT WHOSE `l0` IS NOT WHERE IT LIVES IS REFUSED, AND IT IS THE ONLY REFUSAL
    /// IN THE VENDOR'S FILE.
    ///
    /// `(core 0, corelet 1)` binds `l0` at `{core = 0, corelet = 0}` (`mixed.mlir:290`) while the
    /// group's base binds its own at `(1, 1)` (`:460`). The functor wants the candidate's binding to
    /// sit at the candidate's placement, `(0, 1)`, and `(0, 0)` is not that — so the comparison falls
    /// through to the structural test, where two different residencies are two different ops. This is
    /// why the vendor's output has a SECOND `program_unit` (`:11`) instead of one.
    #[test]
    fn the_unit_whose_scratchpad_sits_on_another_corelet_differs() {
        let program = mixed_program();
        let units = mixed_units(&program);
        let scope = module_scope(&program);
        let group = ReducibleProgramUnits::of(units[3]);
        assert_eq!(
            match_units(&group, units[1], &scope),
            UnitMatch::Differs,
            "its l0 is at (0,0) and the parametric answer would need it at (0,1)"
        );
    }

    /// 🎯 193/384 — `(core 0, corelet 0)` MATCHES BOTH BASES, AND THE GROUP ORDER IS WHAT PUTS IT IN
    /// THE FAR CORE'S.
    ///
    /// ⛔⛔ THE ANSWER IS NOT "ITS OWN CORE". `mixed.mlir:97` gives the surviving unit over
    /// `(%3, %2, %0)` — the base `(1, 1)`, the candidate `(1, 0)` and this one — while `(0, 1)`'s group
    /// keeps `(%1)` alone (`:11`). Both bases answer [`UnitMatch::Matched`] here; entry 256 tries the
    /// groups in the order they were created (`:187-198`) and the reverse walk created `(1, 1)`'s
    /// first, so this unit's work is performed by a unit on the OTHER CORE. A port that compared cores
    /// would have emitted a program where one corelet does another's work.
    #[test]
    fn the_first_unit_matches_both_bases_and_the_group_order_decides() {
        let program = mixed_program();
        let units = mixed_units(&program);
        let scope = module_scope(&program);
        assert_eq!(
            match_units(&ReducibleProgramUnits::of(units[3]), units[0], &scope),
            UnitMatch::Matched,
            "every one of (0,0)'s bindings sits at (0,0), which is its own placement"
        );
        assert_eq!(
            match_units(&ReducibleProgramUnits::of(units[1]), units[0], &scope),
            UnitMatch::Matched,
            "and (0,1)'s group would take it too — the base's l0 at (0,0) matches structurally"
        );
    }

    /// 🎯 193/384 — WITHOUT THE FUNCTOR NOTHING IN THE VENDOR'S FILE WOULD MERGE AT ALL.
    ///
    /// The two bodies' first ops are `get_unit`s of one type at two placements. At
    /// [`HighPreference::None`] they are two different operations, because their `core`/`corelet`
    /// attributes differ; the preference is the whole of what makes them one.
    #[test]
    fn the_preference_is_what_makes_two_placements_one_operation() {
        let program = mixed_program();
        let units = mixed_units(&program);
        let scope = module_scope(&program);
        let base = &units[3].body[0];
        let curr = &units[2].body[0];

        assert!(
            !crate::bridges::dataflow_ir_to_sentient::vc_vector_chain_helper::ops_are_equivalent(
                base,
                curr,
                &scope,
                HighPreference::None,
            ),
            "an l0su at (1,1) and one at (1,0) are two different bindings structurally"
        );
        assert!(
            crate::bridges::dataflow_ir_to_sentient::vc_vector_chain_helper::ops_are_equivalent(
                base,
                curr,
                &scope,
                HighPreference::UnitsAreParametric(UnitMatchingInfo {
                    base_unit: UnitPlacement::of(at(1, 1)),
                    curr_unit: UnitPlacement::of(at(1, 0)),
                }),
            ),
            "and the functor is why the pass can merge them"
        );
    }

    /// 🎯 193/384 — THE FUNCTOR IS DIRECTIONAL.
    ///
    /// The reference tests `a` against the BASE's ids and `b` against the CANDIDATE's
    /// (`ProgramUnitsReduction.cpp:120-126`), under its own comment about respecting *"the use of
    /// core_id, corelet_id across the program"*. Reversed, the same pair is not preferred.
    #[test]
    fn the_functor_asks_the_base_side_first() {
        let a = get_unit(0, at(1, 1), DfirUnit::L0su);
        let b = get_unit(1, at(1, 0), DfirUnit::L0su);
        let forwards = HighPreference::UnitsAreParametric(UnitMatchingInfo {
            base_unit: UnitPlacement::of(at(1, 1)),
            curr_unit: UnitPlacement::of(at(1, 0)),
        });
        assert!(forwards.prefers(&a, &b));
        assert!(
            !forwards.prefers(&b, &a),
            "a base that reads the candidate's core is a different program"
        );
    }

    /// 🎯 193/384 — THE FUNCTOR PREFERS NOTHING BUT A PAIR OF `get_unit`s OF ONE TYPE.
    #[test]
    fn only_two_get_units_of_one_type_are_preferred() {
        let placed = HighPreference::UnitsAreParametric(UnitMatchingInfo {
            base_unit: UnitPlacement::of(at(1, 1)),
            curr_unit: UnitPlacement::of(at(1, 0)),
        });
        let base_side = get_unit(0, at(1, 1), DfirUnit::L0su);
        let curr_side = get_unit(1, at(1, 0), DfirUnit::L0su);
        let other_kind = get_unit(1, at(1, 0), DfirUnit::L0);
        let not_a_unit = DfirOp::Dataflow(dataflow::Op::SyncSend {
            to: Val(0),
            signal: SyncSignal::InputToLxsuToLxluToSync,
        });

        assert!(
            !placed.prefers(&base_side, &other_kind),
            "`operation_a.getType() == operation_b.getType()` is the innermost test"
        );
        assert!(
            !placed.prefers(&base_side, &not_a_unit),
            "isa<GetUnitOp>(op_b)"
        );
        assert!(
            !placed.prefers(&not_a_unit, &curr_side),
            "isa<GetUnitOp>(op_a)"
        );
        assert!(
            !HighPreference::None.prefers(&base_side, &curr_side),
            "a null functor is never consulted"
        );
    }

    /// 🎯 193/384 — TWO UNITS OF DIFFERENT TYPES NEVER MATCH, WHATEVER THEIR BODIES SAY.
    ///
    /// `:82-83`, *"Check if units have same type (e.g., ptrow0)"*. The two units here have BYTE
    /// IDENTICAL bodies and one placement, so only that check can be answering.
    #[test]
    fn two_units_of_different_kinds_differ_on_the_type_check_alone() {
        let preamble = vec![
            get_unit(0, at(0, 0), DfirUnit::L0lu),
            get_unit(1, at(0, 0), DfirUnit::L0su),
        ];
        let shared = body(10, at(0, 0), at(0, 0));
        let units = ProgramUnits::of(
            unit_on(0, DfirUnit::L0lu, shared.clone()),
            vec![unit_on(1, DfirUnit::L0su, shared)],
        );
        let program = Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            grid: Grid::single(),
            preamble,
            units,
            arch: core::marker::PhantomData,
        };
        let listed = mixed_units(&program);
        let scope = module_scope(&program);
        assert_eq!(
            match_units(&ReducibleProgramUnits::of(listed[0]), listed[1], &scope),
            UnitMatch::Differs
        );
    }

    /// 🎯 193/384 — A SHORTER BODY IS A DIFFERENT BLOCK.
    ///
    /// `blocksAreEquivalent`'s *"block sizes don't match"* (`OperationEquivalence.cpp:48-53`), which
    /// is checked before any operation is compared — so the prefix agreeing is not enough.
    #[test]
    fn a_candidate_missing_the_last_op_differs() {
        let preamble = vec![
            get_unit(0, at(1, 1), DfirUnit::L0lu),
            get_unit(1, at(1, 0), DfirUnit::L0lu),
        ];
        let mut short = body(20, at(1, 0), at(1, 0));
        short.pop();
        let units = ProgramUnits::of(
            unit_on(0, DfirUnit::L0lu, body(10, at(1, 1), at(1, 1))),
            vec![unit_on(1, DfirUnit::L0lu, short)],
        );
        let program = Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            grid: Grid::single(),
            preamble,
            units,
            arch: core::marker::PhantomData,
        };
        let listed = mixed_units(&program);
        let scope = module_scope(&program);
        assert_eq!(
            match_units(&ReducibleProgramUnits::of(listed[0]), listed[1], &scope),
            UnitMatch::Differs
        );
    }

    /// 🎯 193/384 — A FIRST OPERAND NO `get_unit` BINDS ANSWERS `Differs` INSTEAD OF FOLLOWING A NULL.
    ///
    /// `getUnits()[0].getDefiningOp<GetUnitOp>()` is unchecked in the reference and
    /// `getCoreOrCoreletID` dereferences it straight away, so this input segfaults dcc. The
    /// conservative answer costs one reduction.
    #[test]
    fn a_first_operand_nothing_binds_differs_rather_than_following_a_null() {
        let program = mixed_program();
        let units = mixed_units(&program);
        // Every body, and none of the preamble — so the `l0lu` bindings the units run on are absent.
        let scope: Vec<DfirOp> = units
            .iter()
            .flat_map(|unit| unit.body.iter().cloned())
            .collect();
        assert_eq!(
            match_units(&ReducibleProgramUnits::of(units[3]), units[2], &scope),
            UnitMatch::Differs
        );
    }

    /// 🎯 193/384 — THE PLACEMENT IS THE ATTRIBUTE PAIR, WHICH IS COARSER THAN THE RESIDENCY.
    ///
    /// A core-wide unit writes `core = c, corelet = 0` and so reads back as the same placement as a
    /// per-corelet unit on corelet 0; a scratchpad has a `core` and no `corelet`; a global has
    /// neither, which is the reference's `-1, -1`.
    #[test]
    fn a_core_wide_unit_and_corelet_zero_are_one_placement() {
        let core = Core::checked(3).expect("this arch has core 3");
        assert_eq!(
            UnitPlacement::of(Residency::CoreWide { core }),
            UnitPlacement::of(Residency::Corelet {
                core,
                corelet: CORELET_ZERO
            }),
            "both print `core = 3 : i32, corelet = 0 : i32`"
        );
        assert_eq!(
            UnitPlacement::of(Residency::Scratchpad { core }),
            UnitPlacement {
                core: Some(core),
                corelet: None
            },
            "`C3-lx` carries no `corelet`, so the read is -1"
        );
        assert_eq!(
            UnitPlacement::of(Residency::Global),
            UnitPlacement {
                core: None,
                corelet: None
            },
            "the HBM carries neither attribute"
        );
        assert_ne!(
            UnitPlacement::of(Residency::Scratchpad { core }),
            UnitPlacement::of(Residency::CoreWide { core }),
            "an absent `corelet` is not `corelet = 0`"
        );
    }

    /// 🎯 193/384 — A NEW GROUP HOLDS EXACTLY THE BASE'S OWN UNITS.
    ///
    /// The reference's constructor (`:42-47`) copies `base.getUnits()` and nothing else; entry 256 is
    /// what appends the matched candidates'.
    #[test]
    fn a_group_starts_out_holding_the_bases_own_units() {
        let program = mixed_program();
        let units = mixed_units(&program);
        let group = ReducibleProgramUnits::of(units[3]);
        assert_eq!(group.units_list, vec![Val(3)]);
        assert!(core::ptr::eq(group.base_unit_program, units[3]));
    }

    /// 🎯 193/384 — THE COMPARISON IS CONFIGURED WITH THE PASS'S TAG AND THE CONSTRUCTOR'S THREE
    /// DEFAULTS.
    ///
    /// ⭐ `all_block_args_are_equiv = true` IS THE ONE THAT MATTERS: two units' induction variables
    /// are arguments of different blocks and `operator==` always calls those unequal
    /// (`OperationEquivalence.cpp:30-32`). The conditional trees ask for `false`; this pass takes the
    /// default and must.
    #[test]
    fn the_comparison_is_tagged_for_this_pass_and_takes_every_default() {
        let oe = OperationEquivalence::preferring(
            HighPreference::None,
            EquivalenceTag::ProgramUnitsReduction,
        );
        assert_eq!(oe.debug.spelling(), "program-units-reduction");
        assert_eq!(
            oe,
            OperationEquivalence {
                debug: EquivalenceTag::ProgramUnitsReduction,
                preference: HighPreference::None,
                subregions: super::super::tf_cfgs_dataflow_conditional_tree::SubregionCompare::Recursive,
                block_args:
                    super::super::tf_cfgs_dataflow_conditional_tree::BlockArgEquivalence::AllEquivalent,
                cache: super::super::tf_cfgs_dataflow_conditional_tree::EquivalenceCache::Reuse,
            }
        );
    }

    /// 🎯 193/384 — THE MODULE SCOPE IS THE PREAMBLE FOLLOWED BY EVERY UNIT'S BODY.
    #[test]
    fn the_module_scope_reaches_a_binding_in_any_unit() {
        let program = mixed_program();
        let scope = module_scope(&program);
        assert_eq!(scope.len(), 4 + 4 * 5);
        assert_eq!(
            crate::islands::dataflow_ir::dialects::defining_op(Val(42), &scope),
            Some(&get_unit(42, at(1, 1), DfirUnit::L0)),
            "the fourth unit's own `l0` binding"
        );
    }
}

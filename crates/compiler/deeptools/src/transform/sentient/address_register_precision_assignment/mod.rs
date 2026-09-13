// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ CRUSTIFY SENTIENT-PASSES CAMPAIGN — READ THIS BEFORE YOU FILL AN ANCHOR IN THIS FILE.        ║
// ║ Campaign statement: crustify-senpass/TASK.md   ·   worklist: crustify-senpass/UNITS.tsv      ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
//
// 1. THE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT.
//       /Users/nickm/git/deeptools-src/<file>:<line>     (deeptools|master|a0d29abbed — repo_info.txt)
//    Every citation below resolves against that revision. `crustify-senpass/cpp/sentient.cpp` says
//    WHICH functions are in scope and IN WHAT ORDER; its bodies were verified byte-identical to the
//    authority (656/656, 962,619 bytes, two negative controls), so either may be read — but the
//    authority file is the one that carries the surrounding declarations you will need.
//    ⛔ /Users/nickm/git/deeptools is a DIFFERENT revision. ⛔ The pod is not reachable from here.
//
// 2. THESE PASSES REWRITE SentientIR IN PLACE. They are NOT a conversion between rungs like bridges
//    1–4: input and output are both `src/islands/sentient/`. Expect to EXTEND that island — ops
//    gaining an assigned register, a pinned address, a rolled loop — not to emit into a new one.
//    WHY IT MATTERS: bridge 2's emission assigns NO registers (`index: None` in 39 of 41 sites),
//    faithfully, because the reference's SentientIR carries `regIndex = -1 : i32` in all 54
//    occurrences of the committed golden corpus. THESE passes turn -1 into a real register file and
//    index. Without them ProgIR gets -1 where an instruction needs a register and the backend
//    refuses with `Register initialization out of boundary` (observed on lxsu0:LRF0, l3lu:LBR2).
//
// 3. PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EFFECT. For an in-place pass the effect IS the
//    port: WHICH ops are rewritten, WHICH attributes are set to WHAT, and IN WHAT ORDER. A hand
//    attempt on bridge 2 extracted each function's decision rule into a documented predicate,
//    omitted the part that changed the IR, and reported it done — nothing called any of it.
//    Droppable: only the mechanism for REACHING operands (walking uses, memoising, positioning a
//    builder). ⛔ If the island cannot express a result, EXTEND THE ISLAND. Deciding a function is
//    unnecessary is NOT the porter's call, and a predicate is not a port.
//
// 4. HARD CAPS — MEASURED, AND THEY BIND. Bridge 2's first 144 functions cost 43 h wall clock and
//    586 M cache-read tokens; of 30,991 lines produced only 5,306 were implementation (12,333 doc
//    comments, 12,575 tests). Per ported function:
//      • 8 DOC LINES: the `/// Replaces: eNNN_name` anchor, one line of what it does, any TRAP.
//        No tutorials, no restating the C++, no design essays.
//      • ONE TEST. Two only where the vendor's own case AND a negative both apply.
//      • `cargo check -p deeptools` + `cargo test -p deeptools` ONCE PER BATCH, not per function.
//      • Do NOT re-verify citations — the review pass owns that.
//      • Do NOT grep the crate to discover types; the anchor names what you need.
//    NOT capped: correctness, and the emission.
//
// 5. THIS IS A PURE-LOGIC PORT WITH NO C ANYWHERE. Whatever generic C-to-Rust conventions say:
//    ❌ no bindgen/allowlist/-sys, ❌ no `ffi::`/`extern "C"`, ❌ no `CRUSTIFY_<FILE>` switch,
//    ❌ no `Foo`/`FooRef`/`FooMut` triple, ❌ no `unsafe`, ❌ no C-vs-Rust equivalence harness.
//
// 6. CRATE RULES BIND YOU — `crates/compiler/deeptools/CLAUDE.md`, read it in full.
//    🛑 NEVER RUNTIME REFUSE: no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`.
//    A closed set is an `enum`; an invariant is a TYPE; newtypes, never raw scalars. No strings for
//    closed sets. `Arch`/`Model`/`Workload` flow through. `todo!` NAMING UNPORTED WORK IS ALLOWED
//    here — and ⛔ never substitute a stand-in op to dodge one.
//
// 7. ANCHORS: each `// crustify:todo: eNNN_name` below is one scheduled unit. Replace it with the
//    ported item carrying the doc anchor `/// Replaces: eNNN_name`. ⛔ NEVER DELETE AN ANCHOR YOU
//    DID NOT PORT — on bridge 2 a deleted anchor was indistinguishable from a finished unit and the
//    campaign reported DONE having silently lost 149 of 384 functions, the biggest in its span.
//
// 8. GATE: `cargo check -p deeptools` and `cargo test -p deeptools`. ⛔ NEVER run the workspace or
//    the acceptance build in an agent worktree — ~6 GB of target/ each, and this host is tight.
//
// 9. 43 OF THE 52 PASSES CONSUME AN ANALYSIS THAT IS OUT OF SCOPE (122 of the 656 units name one):
//    `Analyses/` (Liveness, PropagationAnalysis, GraphColoring, ExpressionEvaluatorUtils,
//    InstructionEstimation, TimeStamps, RegisterPressureAnalysis, AddressPinningScheme,
//    XRFRegisterAnalyzer, CorrelationAnalysis, RedundantDefinitionEliminationTree, …) and
//    `RegisterInitialization/` (Collector, Evaluator, Selector, Transformer, UniformGrouper) —
//    ~11,700 lines NOT in this campaign. Per pass: `crustify-senpass/OUTSIDE-DEPS.tsv`; per unit:
//    `OUTSIDE-UNITS.tsv`. ⭐ When a unit needs one, port the part that is present and
//    `todo!("<Analysis>::<method> — out of campaign scope")` for the part that is not, leaving the
//    anchor FILLED so the unit is not lost. ⛔ DO NOT INVENT THE ANALYSIS and do not substitute a
//    constant for its result. Extending `src/islands/sentient/` is a different case and IS expected.

//! `AddressRegisterPrecisionAssignment.cpp` — 6 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e019_initializePrecision` | 019 | 0 | 11 | `dcc/src/Transform/Sentient/AddressRegisterPrecisionAssignment.cpp:82` |
//! | `e020_assignPrecisionHelper` | 020 | 0 | 59 | `dcc/src/Transform/Sentient/AddressRegisterPrecisionAssignment.cpp:99` |
//! | `e283_assignPrecision` | 283 | 1 | 31 | `dcc/src/Transform/Sentient/AddressRegisterPrecisionAssignment.cpp:164` |
//! | `e428_addToWorkListAndAssignPrecision` | 428 | 2 | 10 | `dcc/src/Transform/Sentient/AddressRegisterPrecisionAssignment.cpp:204` |
//! | `e494_processAddressSSAValue` | 494 | 3 | 223 | `dcc/src/Transform/Sentient/AddressRegisterPrecisionAssignment.cpp:221` |
//! | `e554_runOnOperation` | 554 | 4 | 100 | `dcc/src/Transform/Sentient/AddressRegisterPrecisionAssignment.cpp:449` |

use std::collections::{BTreeMap, VecDeque};

use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;
use crate::formats::Bits;
use crate::islands::dataflow_ir::Values;
use crate::islands::sentient::dialects::{
    self, Op, UniformRegions, Val, sentient, symbol, uniform,
};

/// `-dcc-address-register-precision-assignment-const-commoning`, `cl::init(false)`.
///
/// ⛔ OFF BY DEFAULT ON PURPOSE, with the reference's own reason: caching and commoning
/// `scalar_copy`s *"can increase register pressure and result in unnecessary REGCOPYs (except for
/// LBRs)"* (`AddressRegisterPrecisionAssignment.cpp:45-50`). An LBR copy is commoned regardless,
/// because it is promoted to the program header anyway — see [`PrecisionAssignments`].
const DO_CONST_COMMONING: bool = false;

/// WHAT ONE PRECISION ASSIGNMENT DID — the reference's `bool` and its `Value &new_val` as one value.
///
/// ⭐ THE `bool` ALONE CANNOT SAY IT. `assignPrecisionHelper` returns `true` only for the fresh
/// assignment, writes `new_val` only on the commoning path, and reaches `emitError` +
/// `signalPassFailure` on the genuine conflict — three outcomes behind one boolean and one
/// out-parameter. The caller has to tell them apart: `true` pushes the value onto the worklist and a
/// non-null `new_val` is assigned OVER the operand that was asked about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionAssigned {
    /// `return true` — the slot was `-1` and now holds `element_size`. Push the value.
    Assigned,
    /// `return false` with `new_val` left null: the slot already agreed, or the op is one of the six
    /// this pass assigns no precision to.
    Unchanged,
    /// `new_val` — a `scalar_copy` of a constant, cloned for this element size and inserted before
    /// the user. Assign it over the operand.
    ///
    /// ⛔ EVERY [`OpId`] AT OR AFTER THE USER'S POSITION IN ITS BLOCK IS NOW STALE. MLIR's
    /// `Operation *` survives an insertion into its own block and a position does not, so the caller
    /// must re-derive the positions it holds; the ones this type keeps are re-keyed for it.
    Substituted(Val),
    /// `def_op->emitError("Unknown parent op for precision calculation"); signalPassFailure();`
    /// (`:170-172`) — `val` is a region argument of something that is not a `sentient.for`.
    ///
    /// ⭐ CARRIED AS A VALUE for the same reason [`PrecisionAssigned::DifferentPrecisions`] is: the
    /// reference's failure IS the pass stopping, and no slot was written.
    UnknownParentOp,
    /// `op->emitError("Different precisions assigned to same SSA"); signalPassFailure();`
    ///
    /// ⭐ CARRIED AS A VALUE BECAUSE THAT IS WHAT THE REFERENCE'S FAILURE IS — the pass stops, and
    /// the two disagreeing sizes are what its diagnostic prints.
    DifferentPrecisions {
        /// What the slot already held.
        held: Bits,
        /// What this caller asked for.
        wanted: Bits,
    },
}

/// ONE `signalPassFailure()` THE UPWARD WALK PASSED OVER — `assignPrecisionHelper`'s two `emitError`
/// arms (`:154-157`, `:170-172`) signal the failure and RETURN, so the walk carries on and the pass's
/// verdict is the accumulated list.
///
/// ⭐ DATA, BECAUSE THE REFERENCE'S OWN CONTROL FLOW MAKES IT DATA — unlike
/// [`AddressWalk::NotALoopArgument`], which stops the drain loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrecisionFailure {
    /// The operand the assignment was asked for.
    pub val: Val,
    /// Which of the two `emitError`s it was.
    pub outcome: PrecisionAssigned,
}

/// WHAT `processAddressSSAValue` ANSWERED — its `LogicalResult` (`:221`).
///
/// ⛔ ITS FAILURE STOPS THE PASS, unlike a [`PrecisionFailure`]: `if (failed(..)) { signalPassFailure();
/// break; }` (`:527-530`) abandons the rest of the worklist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressWalk {
    /// `LogicalResult::success()` — `val` was walked, to whatever effect its defining op has.
    Walked,
    /// `LogicalResult::failure()` (`:437`) — `val` is a region argument of something that is not a
    /// `sentient.for`, which is the only way the reference's `else` is reached.
    NotALoopArgument,
}

/// THE PASS'S TWO MAPS — `assignments_` and `cached_copies_`.
///
/// ⛔⛔ THE KEY IS A POSITION, WHICH IS NOT WHAT `Operation *` IS. [`OpId`] is this crate's stand-in
/// for an operation pointer (see its own note), and it answers identity and dominance — but unlike a
/// pointer it MOVES when something is inserted ahead of it in the same block, which
/// [`PrecisionAssignments::assign_precision_helper`] does. Every key it holds is re-keyed at that
/// moment; a caller's own copies are its own problem, and [`PrecisionAssigned::Substituted`] says so.
#[derive(Debug, Default)]
pub struct PrecisionAssignments {
    /// `assignments_` — per op, one slot per value it binds, `None` for the reference's `-1`.
    assignments: BTreeMap<OpId, Vec<Option<Bits>>>,
    /// `cached_copies_` — *"given a copy_op and an element_size map it to a cloned copy for that
    /// element_size"* (`AddressRegisterPrecisionAssignment.cpp:65-66`), keyed by the ORIGINAL copy's
    /// result.
    cached_copies: BTreeMap<Val, BTreeMap<Bits, Val>>,
    /// Every `signalPassFailure()` the upward walk passed over — see [`PrecisionFailure`].
    failures: Vec<PrecisionFailure>,
}

impl PrecisionAssignments {
    /// Replaces: e019_initializePrecision
    ///
    /// One unassigned slot per value `op` binds.
    /// ⛔ A `sentient.for`'S VECTOR IS `1 + carried.len()` LONGER THAN ITS RESULT COUNT, laid out
    /// `[iv, iter_args…, results…]`: entry 283 reads an iter arg at `i + 1` and a result at
    /// `i + offset` with `offset = 1 + getNumRegionIterArgs()` (`:173-190`), so slot 0 is the
    /// induction variable's. ⛔ AND IT APPENDS, exactly as `assignments_[op]` followed by
    /// `push_back` does — the reference initialises each op once, off one preorder walk.
    pub fn initialize_precision(&mut self, at: OpId, op: &Op) {
        let mut size = dialects::results(op).len();
        if let Op::Sentient(sentient::Op::For { carried, .. }) = op {
            size += 1 + carried.len();
        }
        self.assignments
            .entry(at)
            .or_default()
            .extend(core::iter::repeat_n(None, size));
    }

    /// Replaces: e020_assignPrecisionHelper
    ///
    /// Writes `element_size` into `at`'s `index`th slot, and where the slot already holds a
    /// different size clones the `scalar_copy` of a constant behind it so both sizes can coexist.
    /// ⛔ `uniform.equalize_pattern`, THE SIXTH MEMBER OF THE SKIP LIST (`:102-104`), HAS NO FORM AT
    /// THIS RUNG — `uniform::Op` declares four ops and nothing in this compiler emits it, so an op
    /// this list would skip cannot be built. The other five are matched below.
    /// ⛔ THE CLONE GOES BEFORE THE **USER**, not before the copy: `OpBuilder builder(user)` is
    /// anchored on the op that asked, which is what keeps the clone in scope for it (`:117`).
    pub fn assign_precision_helper(
        &mut self,
        body: &mut Vec<Op>,
        values: &mut Values,
        at: &OpId,
        index: usize,
        element_size: Bits,
        user: &OpId,
    ) -> PrecisionAssigned {
        let Some(op) = op_at(at, body) else {
            todo!(
                "assignPrecisionHelper: `assignments_.at(op)` on an op absent from this unit body \
                 ({at:?}) (AddressRegisterPrecisionAssignment.cpp:107)"
            )
        };
        // `isa<ConstantOp, ReceiveAndExtractScalarOp, DefImmutableMappingOp, QueryMapOp,
        // EqualizePatternOp, CreateSymbolOp>(op) -> return false`.
        if matches!(
            op,
            Op::Sentient(
                sentient::Op::ScalarConstant { .. } | sentient::Op::ReceiveAndExtractScalar { .. }
            ) | Op::Uniform(uniform::Op::DefImmutableMapping { .. } | uniform::Op::QueryMap { .. })
                | Op::Symbol(symbol::Op::CreateSymbol { .. })
        ) {
            return PrecisionAssigned::Unchanged;
        }
        // What the disagreement branch reads off the op, taken while `body` is only borrowed shared.
        let copy = if let Op::Sentient(sentient::Op::ScalarCopy {
            input,
            result,
            reg,
            element_size,
            program_header,
        }) = op
        {
            Some((*input, *result, *reg, *element_size, *program_header))
        } else {
            None
        };
        // `!isa<BlockArgument>(copy_op.getInp())` AND `dyn_cast<ConstantOp>(..getDefiningOp())`, which
        // are one lookup here: a block argument has no defining op.
        let copies_a_constant = copy.is_some_and(|(input, ..)| {
            matches!(
                dialects::defining_op(input, body),
                Some(Op::Sentient(sentient::Op::ScalarConstant { .. }))
            )
        });

        let Some(slots) = self.assignments.get_mut(at) else {
            todo!(
                "assignPrecisionHelper: `assignments_.at(op)` on an op initializePrecision never saw \
                 ({at:?}) (AddressRegisterPrecisionAssignment.cpp:479-480)"
            )
        };
        let Some(slot) = slots.get_mut(index) else {
            todo!(
                "assignPrecisionHelper: slot {index} past the {} initializePrecision gave {at:?} \
                 (AddressRegisterPrecisionAssignment.cpp:107)",
                slots.len()
            )
        };
        // `if (assignments_.at(op)[index] == -1)`
        let Some(held) = *slot else {
            *slot = Some(element_size);
            return PrecisionAssigned::Assigned;
        };
        // `if (assignments_.at(op)[index] != element_size)`
        if held == element_size {
            return PrecisionAssigned::Unchanged;
        }

        if let Some((input, copy_result, reg, copy_element_size, program_header)) = copy
            && copies_a_constant
        {
            // ⭐ AN LBR COPY IS COMMONED WHATEVER THE FLAG SAYS: it has to be promoted to the program
            // header anyway, so hoisting it does not lengthen its live range (`:120-122`).
            if (DO_CONST_COMMONING || reg.locale == sentient::RegType::Lbr)
                && let Some(cached) = self
                    .cached_copies
                    .get(&copy_result)
                    .and_then(|per_size| per_size.get(&element_size))
            {
                todo!(
                    "moveToCommonDominator (senpass e241, transform/sentient/utils) is not ported \
                     yet, and reusing the cached {cached:?} for {copy_result:?} at {element_size:?} \
                     is what needs it (AddressRegisterPrecisionAssignment.cpp:129-138)"
                )
            }
            // `auto *new_copy_op = builder.clone(*copy_op);` — a clone binds a FRESH result.
            let clone_result = values.mint();
            let clone = Op::Sentient(sentient::Op::ScalarCopy {
                input,
                result: clone_result,
                reg,
                // `builder.clone` COPIES THE DICTIONARY; the map entry below is what the flush
                // (`:535-547`) later overwrites it with.
                element_size: copy_element_size,
                program_header,
            });
            if insert_at(body, user.path(), clone).is_some() {
                todo!(
                    "assignPrecisionHelper: `OpBuilder builder(user)` on a user absent from this \
                     unit body ({user:?}) (AddressRegisterPrecisionAssignment.cpp:117)"
                )
            }
            self.shift_keys_at_or_after(user.path());
            // `assignments_[new_copy_op] = {element_size};` — ONE slot, not initializePrecision's
            // vector: a `scalar_copy` binds one value.
            self.assignments
                .insert(OpId::at(user.path()), vec![Some(element_size)]);
            // `cached_copies_[copy_op->getResult(0)][element_size] = new_val;`
            self.cached_copies
                .entry(copy_result)
                .or_default()
                .insert(element_size, clone_result);
            return PrecisionAssigned::Substituted(clone_result);
        }

        PrecisionAssigned::DifferentPrecisions {
            held,
            wanted: element_size,
        }
    }

    /// What `at` holds now — the slots [`PrecisionAssignments::initialize_precision`] gave it.
    #[must_use]
    pub fn slots(&self, at: &OpId) -> Option<&[Option<Bits>]> {
        self.assignments.get(at).map(Vec::as_slice)
    }

    /// The `signalPassFailure()`s so far — see [`PrecisionFailure`].
    #[must_use]
    pub fn failures(&self) -> &[PrecisionFailure] {
        &self.failures
    }

    /// Replaces: e283_assignPrecision
    ///
    /// Assigns `element_size` to the slot `val` ITSELF occupies on the op that binds it: an iter
    /// arg's `i + 1` (`:167`) or a result's `i + offset`, `offset = 1 + getNumRegionIterArgs()`
    /// (`:180-190`) — which is the layout [`PrecisionAssignments::initialize_precision`] gives an op.
    ///
    /// ⛔ A `val` THIS BODY DOES NOT BIND IS THE REFERENCE'S `else` ARM, NOT ITS `return false`: the
    /// only values left are region arguments of something other than a `sentient.for` — the program
    /// unit's own, or a `uniform` region's — which is `emitError("Unknown parent op for precision
    /// calculation")` (`:170-173`). Its `return false` is [`PrecisionAssigned::Unchanged`], and
    /// [`PrecisionAssignments::assign_precision_helper`] is the only thing that reaches it.
    pub fn assign_precision(
        &mut self,
        body: &mut Vec<Op>,
        values: &mut Values,
        val: Val,
        element_size: Bits,
        user: &OpId,
    ) -> PrecisionAssigned {
        match bound_slot(val, body, &[], 0) {
            Some((at, index)) => {
                self.assign_precision_helper(body, values, &at, index, element_size, user)
            }
            None => PrecisionAssigned::UnknownParentOp,
        }
    }

    /// Replaces: e428_addToWorkListAndAssignPrecision
    ///
    /// Assigns `element_size` to `val`'s own slot and queues `val` for the upward walk — but ONLY on a
    /// fresh assignment, *"otherwise, it leads to infinite loop"* (`:206-207`).
    ///
    /// ⭐ RETURNS THE WHOLE OUTCOME, WHICH IS MORE THAN THE REFERENCE'S `new_val`: the two failure
    /// arms of [`PrecisionAssigned`] are how this pass stops (`signalPassFailure`), and with no
    /// `Result` in this crate the caller has to receive them as a value. `Substituted(v)` is the
    /// `new_val` its callers assign over the operand (`:236`).
    pub fn add_to_worklist_and_assign_precision(
        &mut self,
        body: &mut Vec<Op>,
        values: &mut Values,
        worklist: &mut AddressWorklist,
        val: Val,
        element_size: Bits,
        user: &OpId,
    ) -> PrecisionAssigned {
        let outcome = self.assign_precision(body, values, val, element_size, user);
        if outcome == PrecisionAssigned::Assigned {
            worklist.push(val);
        }
        outcome
    }

    /// EVERY KEY AT OR AFTER `at` IN ITS OWN BLOCK MOVES DOWN ONE — the price of a positional
    /// identity, paid where the insertion happens rather than left for a reader to discover.
    fn shift_keys_at_or_after(&mut self, at: &[u32]) {
        let Some((&ordinal, block)) = at.split_last() else {
            return;
        };
        let depth = block.len();
        self.assignments = core::mem::take(&mut self.assignments)
            .into_iter()
            .map(|(key, slots)| {
                let mut path = key.path().to_vec();
                if path.len() > depth && path[..depth] == *block && path[depth] >= ordinal {
                    path[depth] += 1;
                }
                (OpId::at(&path), slots)
            })
            .collect();
    }
}

/// `std::queue<Value> worklist_` (`:63`) — the addresses still to walk upward, FIFO, drained by
/// `processAddressSSAValue` (`:525-527`).
///
/// ⛔ ITS OWN TYPE, NOT A FIELD OF [`PrecisionAssignments`]: the reference pushes onto it from
/// `addToWorkListAndAssignPrecision` while `assignments_` is being written through the same `this`,
/// and keeping the queue apart is what lets a Rust caller hold both at once.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct AddressWorklist(VecDeque<Val>);

impl AddressWorklist {
    /// `worklist_.push(val)`.
    pub fn push(&mut self, val: Val) {
        self.0.push_back(val);
    }

    /// `worklist_.front()` then `worklist_.pop()` — `None` is `worklist_.empty()`, which is the
    /// reference's own loop condition (`:525`) and not a refusal.
    pub fn pop(&mut self) -> Option<Val> {
        self.0.pop_front()
    }
}

/// WHERE A VALUE IS BOUND AND WHICH SLOT IS ITS OWN — `isa<BlockArgument>(val)` and
/// `val.getDefiningOp()` as ONE search, because a position has to be looked for where a pointer is
/// merely followed. `base` numbers a nested body the way [`op_at`] reads it: region after region.
///
/// ⭐ THE ISLAND'S OWN BLOCK-ARGUMENT INDEX **IS** THE REFERENCE'S `i + 1`
/// ([`crate::islands::sentient::dialects::sentient::block_args`] puts the induction variable at 0),
/// and `sentient.for` is the only op of this dialect that binds any.
fn bound_slot(val: Val, scope: &[Op], prefix: &[u32], base: usize) -> Option<(OpId, usize)> {
    for (position, op) in scope.iter().enumerate() {
        let mut path = prefix.to_vec();
        path.push(u32::try_from(base + position).unwrap_or_default());
        if let Op::Sentient(inner) = op
            && let Some(index) = sentient::block_args(inner)
                .iter()
                .position(|arg| *arg == val)
        {
            return Some((OpId::at(&path), index));
        }
        if let Some(index) = dialects::results(op)
            .iter()
            .position(|result| *result == val)
        {
            let offset = match op {
                Op::Sentient(sentient::Op::For { carried, .. }) => 1 + carried.len(),
                _ => 0,
            };
            return Some((OpId::at(&path), index + offset));
        }
        let mut region_base = 0usize;
        for region in dialects::regions_ref(op) {
            if let Some(found) = bound_slot(val, region, &path, region_base) {
                return Some(found);
            }
            region_base += region.len();
        }
    }
    None
}

/// The op at a position — this rung's copy of `vc_lowering_pt_masks::op_at`, regions concatenated,
/// which is the numbering [`OpId`] documents.
fn op_at<'a>(id: &OpId, scope: &'a [Op]) -> Option<&'a Op> {
    let (&ordinal, rest) = id.path().split_first()?;
    let op = scope.get(ordinal as usize)?;
    let Some(&next) = rest.first() else {
        return Some(op);
    };
    // ⛔ ONLY A `sentient.*` OP HOLDS A REGION OF THIS RUNG'S OPS at this point in the pipeline —
    // *"all loops have been lowered to sentient.for"* (`VectorChainToSentientPT/Helper.cpp:141`).
    let Op::Sentient(inner) = op else {
        return None;
    };
    let mut base = 0usize;
    for region in sentient::regions(inner) {
        if (next as usize) < base + region.len() {
            let mut sub: Vec<u32> = rest.to_vec();
            sub[0] = (next as usize - base) as u32;
            return op_at(&OpId::at(&sub), region);
        }
        base += region.len();
    }
    None
}

/// `OpBuilder builder(user); builder.clone(..)` — `op` takes the slot `path` names and everything at
/// or after it in that block shifts down. Hands `op` back when the path names nothing.
fn insert_at(block: &mut Vec<Op>, path: &[u32], op: Op) -> Option<Op> {
    let Some((&ordinal, rest)) = path.split_first() else {
        return Some(op);
    };
    if rest.is_empty() {
        if (ordinal as usize) > block.len() {
            return Some(op);
        }
        block.insert(ordinal as usize, op);
        return None;
    }
    let Some(Op::Sentient(inner)) = block.get_mut(ordinal as usize) else {
        return Some(op);
    };
    let next = rest[0] as usize;
    let mut base = 0usize;
    for region in sentient::regions_mut(inner) {
        if next < base + region.len() {
            let mut sub: Vec<u32> = rest.to_vec();
            sub[0] = (next - base) as u32;
            return insert_at(region, &sub, op);
        }
        base += region.len();
    }
    Some(op)
}

/// WHICH OPERANDS ONE ADDRESS PROPAGATES INTO — the reference's eleven `dyn_cast` arms (`:225-411`)
/// as the shape each writes, so that the op can be read once and then mutated.
enum Propagation {
    /// The address operands of a transfer, or the inputs of an `add`/`sub`/`copy` in an addressing
    /// register, in the reference's own order and all at `assignments_[op].front()`.
    Operands(Vec<Val>),
    /// `if_op` — operand `i` of the then-region terminator, and of the else region when it has one.
    IfYields(usize),
    /// `for_op`'s result `i` — its yield operand, its iter operand and its region iter arg.
    ForResult(usize),
    /// `for_op`'s region iter arg `i` — its iter operand, its yield operand and its result.
    ForIterArg(usize),
    /// `uniformize_regions` result `i` — operand `i` of EVERY region's terminator.
    UniformYields(usize),
    /// Nothing to propagate: a non-addressing `add`/`sub`/`copy`, or the induction variable, which
    /// `getRegionIterArgs()` excludes so the reference's loop matches it nowhere.
    Nothing,
}

impl PrecisionAssignments {
    /// Replaces: e494_processAddressSSAValue
    ///
    /// Propagates `val`'s element size into the operands its defining op reaches it from: a
    /// transfer's addresses, an addressing `add`/`sub`/`copy`'s inputs, or the terminator, iter
    /// operand and iter arg of an `if`/`for`/`uniformize_regions` (`:221-443`).
    ///
    /// TRAP: A LOOP'S YIELD OPERAND TAKES THE **YIELD** AS ITS USER (`:377`), so a clone lands INSIDE
    /// the body and every path into it is re-derived from the loop's CURRENT one.
    /// TRAP: `assignments_[op].front()` IS SLOT 0 whichever result reached the worklist (`:228`).
    pub fn process_address_ssa_value(
        &mut self,
        body: &mut Vec<Op>,
        values: &mut Values,
        worklist: &mut AddressWorklist,
        val: Val,
    ) -> AddressWalk {
        // `isa<BlockArgument>(val)` and `val.getDefiningOp()` as ONE search — and its `None` is the
        // reference's `else { return failure(); }` (`:436-437`): the only values this body binds
        // nowhere are region arguments of a program unit or a `uniform` region.
        let Some((at, slot)) = bound_slot(val, body, &[], 0) else {
            return AddressWalk::NotALoopArgument;
        };
        // `auto element_sizes = assignments_[op];` — a COPY, taken before anything is inserted.
        let Some(element_sizes) = self.slots(&at).map(<[Option<Bits>]>::to_vec) else {
            todo!(
                "processAddressSSAValue: `assignments_[op]` on an op initializePrecision never saw \
                 ({at:?}) (AddressRegisterPrecisionAssignment.cpp:226)"
            )
        };
        let Some(op) = op_at(&at, body) else {
            todo!(
                "processAddressSSAValue: {val:?} is bound at {at:?}, a position only a region this \
                 rung's `op_at` does not descend can hold \
                 (AddressRegisterPrecisionAssignment.cpp:224)"
            )
        };
        let plan = match op {
            Op::Sentient(
                sentient::Op::LoadAndSend {
                    mutable_addr,
                    immutable_addr,
                    increment,
                    ..
                }
                | sentient::Op::ReceiveAndStore {
                    mutable_addr,
                    immutable_addr,
                    increment,
                    ..
                }
                | sentient::Op::LoadAndExtractScalar {
                    mutable_addr,
                    immutable_addr,
                    increment,
                    ..
                }
                | sentient::Op::LoadComputeAndSend {
                    mutable_addr,
                    immutable_addr,
                    increment,
                    ..
                },
            ) => Propagation::Operands(vec![*mutable_addr, *immutable_addr, *increment]),
            Op::Sentient(sentient::Op::LoadAndStore {
                src_mutable_addr,
                src_immutable_addr,
                src_inc,
                dst_mutable_addr,
                dst_immutable_addr,
                dst_inc,
                ..
            }) => Propagation::Operands(vec![
                *src_mutable_addr,
                *src_immutable_addr,
                *src_inc,
                *dst_mutable_addr,
                *dst_immutable_addr,
                *dst_inc,
            ]),
            Op::Sentient(
                sentient::Op::ScalarAdd { lhs, rhs, reg, .. }
                | sentient::Op::ScalarSub { lhs, rhs, reg, .. },
            ) if addresses_a_register(*reg, false) => Propagation::Operands(vec![*lhs, *rhs]),
            Op::Sentient(sentient::Op::ScalarCopy { input, reg, .. })
                if addresses_a_register(Some(*reg), true) =>
            {
                Propagation::Operands(vec![*input])
            }
            // The three arms above with a locale the reference's `is_any_of` refuses (`:311`, `:327`,
            // `:339`): the whole body is inside that `if`, so the op is left alone.
            Op::Sentient(
                sentient::Op::ScalarAdd { .. }
                | sentient::Op::ScalarSub { .. }
                | sentient::Op::ScalarCopy { .. },
            ) => Propagation::Nothing,
            // `if_op->getResults()[i] == val` — the slot IS that `i`, since an `if` binds no block
            // argument and [`initialize_precision`](PrecisionAssignments::initialize_precision)
            // gives it one slot per result.
            Op::Sentient(sentient::Op::If { .. }) => Propagation::IfYields(slot),
            Op::Sentient(sentient::Op::For { carried, .. }) => {
                // `int offset = 1 + for_op.getNumRegionIterArgs();`
                let offset = 1 + carried.len();
                if slot >= offset {
                    Propagation::ForResult(slot - offset)
                } else if slot >= 1 {
                    Propagation::ForIterArg(slot - 1)
                } else {
                    Propagation::Nothing
                }
            }
            Op::UniformRegions(UniformRegions::UniformizeRegions { .. }) => {
                Propagation::UniformYields(slot)
            }
            other => panic!(
                "llvm_unreachable(\"unsupported op\") \
                 (`AddressRegisterPrecisionAssignment.cpp:410-411`): {other:?}"
            ),
        };

        let mut user = at.clone();
        match plan {
            Propagation::Nothing => {}
            Propagation::Operands(operands) => {
                let element_size = size_at(&element_sizes, 0, &at);
                for (which, operand) in operands.into_iter().enumerate() {
                    self.propagate(
                        body,
                        values,
                        worklist,
                        operand,
                        element_size,
                        &mut user,
                        move |op, copy_val| set_address_operand(op, which, copy_val),
                    );
                }
            }
            Propagation::IfYields(i) => {
                let element_size = size_at(&element_sizes, slot, &at);
                // The then region, then the else region — `!getElseRegion().empty()` (`:358`) is the
                // emptiness [`region_body`] already answers.
                for region in [0, 1] {
                    self.propagate_region_yield(
                        body,
                        values,
                        worklist,
                        &mut user,
                        region,
                        i,
                        element_size,
                    );
                }
            }
            Propagation::ForResult(i) => {
                let element_size = size_at(&element_sizes, slot, &at);
                self.propagate_for_yield(body, values, worklist, &user, i, element_size);
                self.propagate_for_carried(
                    body,
                    values,
                    worklist,
                    &mut user,
                    i,
                    element_size,
                    Carry::Init,
                );
                // "Adding region operand (No need for setting operand here)." (`:387-389`).
                self.propagate_for_carried(
                    body,
                    values,
                    worklist,
                    &mut user,
                    i,
                    element_size,
                    Carry::Arg,
                );
            }
            Propagation::ForIterArg(i) => {
                let element_size = size_at(&element_sizes, slot, &at);
                self.propagate_for_carried(
                    body,
                    values,
                    worklist,
                    &mut user,
                    i,
                    element_size,
                    Carry::Init,
                );
                self.propagate_for_yield(body, values, worklist, &user, i, element_size);
                // "Adding result operand (No need for copy)" (`:431-433`).
                self.propagate_for_carried(
                    body,
                    values,
                    worklist,
                    &mut user,
                    i,
                    element_size,
                    Carry::Result,
                );
            }
            Propagation::UniformYields(i) => {
                let element_size = size_at(&element_sizes, slot, &at);
                let regions = op_at(&user, body).map_or(0, region_count);
                for region in 0..regions {
                    self.propagate_region_yield(
                        body,
                        values,
                        worklist,
                        &mut user,
                        region,
                        i,
                        element_size,
                    );
                }
            }
        }
        AddressWalk::Walked
    }

    /// `if (auto copy_val = addToWorkListAndAssignPrecision(operand, element_size, user)) <write>` —
    /// one operand, with the reference's own two-line shape.
    ///
    /// ⛔ THE CLONE TAKES THE USER'S OWN SLOT (`:117`), so `user` moves down one before `write` can
    /// find it again; the two `emitError` arms are recorded rather than returned, because the
    /// reference's walk does not stop at one.
    fn propagate(
        &mut self,
        body: &mut Vec<Op>,
        values: &mut Values,
        worklist: &mut AddressWorklist,
        operand: Val,
        element_size: Bits,
        user: &mut OpId,
        write: impl FnOnce(&mut Op, Val),
    ) {
        let outcome = self.add_to_worklist_and_assign_precision(
            body,
            values,
            worklist,
            operand,
            element_size,
            user,
        );
        match outcome {
            PrecisionAssigned::Substituted(copy_val) => {
                *user = shifted(user);
                if let Some(op) = op_at_mut(user, body) {
                    write(op, copy_val);
                }
            }
            PrecisionAssigned::UnknownParentOp | PrecisionAssigned::DifferentPrecisions { .. } => {
                self.failures.push(PrecisionFailure {
                    val: operand,
                    outcome,
                })
            }
            PrecisionAssigned::Assigned | PrecisionAssigned::Unchanged => {}
        }
    }

    /// One `getTerminator()->getOperand(i)` of a region of the op at `user` (`:352-365`, `:399-406`).
    fn propagate_region_yield(
        &mut self,
        body: &mut Vec<Op>,
        values: &mut Values,
        worklist: &mut AddressWorklist,
        user: &mut OpId,
        region: usize,
        i: usize,
        element_size: Bits,
    ) {
        let Some(operand) = op_at(user, body)
            .and_then(|op| region_body(op, region))
            .and_then(|region_body| region_body.last().and_then(|last| yield_operand(last, i)))
        else {
            return;
        };
        self.propagate(
            body,
            values,
            worklist,
            operand,
            element_size,
            user,
            move |op, copy_val| set_region_yield_operand(op, region, i, copy_val),
        );
    }

    /// `yield_op->getOperand(i)` with the YIELD as the user (`:375-379`, `:425-429`).
    ///
    /// ⛔ THE YIELD'S PATH IS RE-DERIVED FROM THE LOOP'S CURRENT ONE, never remembered: an earlier
    /// operand of this same walk may have put a clone in front of the loop.
    fn propagate_for_yield(
        &mut self,
        body: &mut Vec<Op>,
        values: &mut Values,
        worklist: &mut AddressWorklist,
        for_op: &OpId,
        i: usize,
        element_size: Bits,
    ) {
        let Some(terminator) = op_at(for_op, body)
            .and_then(for_terminator_path)
            .map(|last| {
                let mut path = for_op.path().to_vec();
                path.push(last);
                OpId::at(&path)
            })
        else {
            return;
        };
        let mut user = terminator;
        let Some(operand) = op_at(&user, body).and_then(|op| yield_operand(op, i)) else {
            return;
        };
        self.propagate(
            body,
            values,
            worklist,
            operand,
            element_size,
            &mut user,
            move |op, copy_val| set_yield_operand(op, i, copy_val),
        );
    }

    /// `for_op.getIterOperands()[i]` / `getRegionIterArgs()[i]` / `getResult(i)` (`:381-390`,
    /// `:420-434`), with the loop as the user.
    fn propagate_for_carried(
        &mut self,
        body: &mut Vec<Op>,
        values: &mut Values,
        worklist: &mut AddressWorklist,
        user: &mut OpId,
        i: usize,
        element_size: Bits,
        which: Carry,
    ) {
        let Some(operand) = op_at(user, body).and_then(|op| carried_of(op, i, which)) else {
            return;
        };
        self.propagate(
            body,
            values,
            worklist,
            operand,
            element_size,
            user,
            move |op, copy_val| {
                // `setIterOperand(i, copy_val)` — the OTHER two are read-only (`:387`, `:431`).
                if which == Carry::Init
                    && let Op::Sentient(sentient::Op::For { carried, .. }) = op
                    && let Some(entry) = carried.get_mut(i)
                {
                    entry.init = copy_val;
                }
            },
        );
    }
}

/// WHICH OF A CARRIED ENTRY'S THREE VALUES — `getIterOperands()`, `getRegionIterArgs()` and
/// `getResults()` are one record at this rung.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Carry {
    /// `getIterOperands()[i]` — the only one the pass ever assigns over.
    Init,
    /// `getRegionIterArgs()[i]`.
    Arg,
    /// `for_op.getResult(i)`.
    Result,
}

/// `is_any_of(getRegLocale(), lar, lbr, ear, ebr, lrf, jcr)` (`:311-314`), and `mvr` as well for a
/// `scalar_copy` (`:339-342`).
///
/// ⭐ AN UNSAID REGISTER IS NONE OF THEM, which is `RegType::Unknown` read through the [`Option`].
fn addresses_a_register(reg: Option<sentient::Reg>, copy: bool) -> bool {
    reg.is_some_and(|reg| {
        matches!(
            reg.locale,
            sentient::RegType::Lar
                | sentient::RegType::Lbr
                | sentient::RegType::Ear
                | sentient::RegType::Ebr
                | sentient::RegType::Lrf
                | sentient::RegType::Jcr
        ) || (copy && reg.locale == sentient::RegType::Mvr)
    })
}

/// `element_sizes[k]`, whose `-1` the seeding walk (`:466-497`) makes unreachable: it assigns every
/// transfer's results before any value of one can reach the worklist.
fn size_at(element_sizes: &[Option<Bits>], k: usize, at: &OpId) -> Bits {
    match element_sizes.get(k) {
        Some(Some(size)) => *size,
        _ => todo!(
            "processAddressSSAValue: slot {k} of the {} that {at:?} holds is still -1, which the \
             seeding walk (AddressRegisterPrecisionAssignment.cpp:466-497) assigns first",
            element_sizes.len()
        ),
    }
}

/// `get<X>Mutable().assign(copy_val)` — the `which`th of the operands the arm listed, written back on
/// the op that listed them.
fn set_address_operand(op: &mut Op, which: usize, val: Val) {
    match op {
        Op::Sentient(
            sentient::Op::LoadAndSend {
                mutable_addr,
                immutable_addr,
                increment,
                ..
            }
            | sentient::Op::ReceiveAndStore {
                mutable_addr,
                immutable_addr,
                increment,
                ..
            }
            | sentient::Op::LoadAndExtractScalar {
                mutable_addr,
                immutable_addr,
                increment,
                ..
            }
            | sentient::Op::LoadComputeAndSend {
                mutable_addr,
                immutable_addr,
                increment,
                ..
            },
        ) => match which {
            0 => *mutable_addr = val,
            1 => *immutable_addr = val,
            _ => *increment = val,
        },
        Op::Sentient(sentient::Op::LoadAndStore {
            src_mutable_addr,
            src_immutable_addr,
            src_inc,
            dst_mutable_addr,
            dst_immutable_addr,
            dst_inc,
            ..
        }) => match which {
            0 => *src_mutable_addr = val,
            1 => *src_immutable_addr = val,
            2 => *src_inc = val,
            3 => *dst_mutable_addr = val,
            4 => *dst_immutable_addr = val,
            _ => *dst_inc = val,
        },
        Op::Sentient(
            sentient::Op::ScalarAdd { lhs, rhs, .. } | sentient::Op::ScalarSub { lhs, rhs, .. },
        ) => match which {
            0 => *lhs = val,
            _ => *rhs = val,
        },
        Op::Sentient(sentient::Op::ScalarCopy { input, .. }) => *input = val,
        _ => {}
    }
}

/// `if_op.getThenRegion()`/`getElseRegion()` (`:352`, `:358`) and `unif_region.getRegion(j)`
/// (`:399`) as a body — `None` where the reference asks `empty()`, which is the very distinction
/// [`dialects::regions_ref`] keeps for an `if` without an else.
fn region_body(op: &Op, region: usize) -> Option<Vec<Op>> {
    let body = *dialects::regions_ref(op).get(region)?;
    if body.is_empty() {
        None
    } else {
        Some(body.to_vec())
    }
}

/// `getTerminator()->setOperand(i, copy_val)` on region `region` of `op`.
fn set_region_yield_operand(op: &mut Op, region: usize, i: usize, val: Val) {
    if let Some(body) = dialects::regions_mut(op).get_mut(region)
        && let Some(terminator) = body.last_mut()
    {
        set_yield_operand(terminator, i, val);
    }
}

/// `unif_region.getNumRegions()` (`:398`).
fn region_count(op: &Op) -> usize {
    dialects::regions_ref(op).len()
}

/// `yield_op->getOperand(i)` — either dialect's terminator.
fn yield_operand(op: &Op, i: usize) -> Option<Val> {
    match op {
        Op::Sentient(sentient::Op::Yield { results }) => results.get(i).copied(),
        Op::Uniform(uniform::Op::Yield { operands }) => operands.get(i).copied(),
        _ => None,
    }
}

/// [`yield_operand`] FOR THE ASSIGNMENT.
fn set_yield_operand(op: &mut Op, i: usize, val: Val) {
    match op {
        Op::Sentient(sentient::Op::Yield { results }) => {
            if let Some(slot) = results.get_mut(i) {
                *slot = val;
            }
        }
        Op::Uniform(uniform::Op::Yield { operands }) => {
            if let Some(slot) = operands.get_mut(i) {
                *slot = val;
            }
        }
        _ => {}
    }
}

/// Where `for_op.getLoopBody().front().getTerminator()` sits WITHIN the loop — its last position.
fn for_terminator_path(op: &Op) -> Option<u32> {
    let Op::Sentient(sentient::Op::For { body, .. }) = op else {
        return None;
    };
    u32::try_from(body.len().checked_sub(1)?).ok()
}

/// One of a loop's carried values — see [`Carry`].
fn carried_of(op: &Op, i: usize, which: Carry) -> Option<Val> {
    let Op::Sentient(sentient::Op::For { carried, .. }) = op else {
        return None;
    };
    let entry = carried.get(i)?;
    Some(match which {
        Carry::Init => entry.init,
        Carry::Arg => entry.arg,
        Carry::Result => entry.result,
    })
}

/// THE USER, ONE SLOT FURTHER DOWN ITS BLOCK — `OpBuilder builder(user)` inserted the clone AT the
/// user's own position (`:117`), which is [`PrecisionAssigned::Substituted`]'s note made concrete.
fn shifted(user: &OpId) -> OpId {
    let mut path = user.path().to_vec();
    if let Some(last) = path.last_mut() {
        *last += 1;
    }
    OpId::at(&path)
}

/// [`op_at`]'s twin FOR THE ASSIGNMENT — what the reference gets for free from an `Operation *`.
fn op_at_mut<'a>(id: &OpId, scope: &'a mut [Op]) -> Option<&'a mut Op> {
    let (&ordinal, rest) = id.path().split_first()?;
    let op = scope.get_mut(ordinal as usize)?;
    let Some(&next) = rest.first() else {
        return Some(op);
    };
    let Op::Sentient(inner) = op else {
        return None;
    };
    let mut base = 0usize;
    for region in sentient::regions_mut(inner) {
        if (next as usize) < base + region.len() {
            let mut sub: Vec<u32> = rest.to_vec();
            sub[0] = (next as usize - base) as u32;
            return op_at_mut(&OpId::at(&sub), region);
        }
        base += region.len();
    }
    None
}

// crustify:todo: e554_runOnOperation
//   authority : dcc/src/Transform/Sentient/AddressRegisterPrecisionAssignment.cpp:449  (100 body lines, level 4)
//   original  : void AddressRegisterPrecisionAssignmentPass::runOnOperation()
//   calls     : e019_initializePrecision, e252_size, e428_addToWorkListAndAssignPrecision, e494_processAddressSSAValue

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::sentient::{Carried, Reg, RegType};

    fn constant(value: i64, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result,
            reg_locale: RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    fn copy(input: Val, result: Val, locale: RegType) -> Op {
        Op::Sentient(sentient::Op::ScalarCopy {
            input,
            result,
            reg: Reg {
                locale,
                index: None,
            },
            program_header: false,
            element_size: None,
        })
    }

    /// A loop's vector is `[iv, iter_args…, results…]`, so a `for` carrying two values gets five
    /// slots where an add gets one.
    #[test]
    fn e019_gives_a_loop_a_slot_for_its_iv_and_each_carried_value() {
        let add = Op::Sentient(sentient::Op::ScalarAdd {
            lhs: Val(1),
            rhs: Val(2),
            result: Val(3),
            reg: None,
            ty: ScalarTy::Index,
            element_size: None,
        });
        let carrier = |init, arg, result| Carried {
            init,
            arg,
            result,
            reg: Reg {
                locale: RegType::Lbr,
                index: None,
            },
            program_header: false,
            element_size: None,
        };
        let loop_op = Op::Sentient(sentient::Op::For {
            iv: Val(4),
            bound: Val(5),
            bound_reg: None,
            carried: vec![
                carrier(Val(6), Val(7), Val(8)),
                carrier(Val(9), Val(10), Val(11)),
            ],
            dbg_name: None,
            body: Vec::new(),
        });

        let mut assignments = PrecisionAssignments::default();
        assignments.initialize_precision(OpId::at(&[0]), &add);
        assignments.initialize_precision(OpId::at(&[1]), &loop_op);

        assert_eq!(assignments.slots(&OpId::at(&[0])), Some(&[None][..]));
        assert_eq!(assignments.slots(&OpId::at(&[1])), Some(&[None; 5][..]));
    }

    /// The second element size to reach one `scalar_copy` of a constant clones it before the user
    /// rather than failing the pass.
    #[test]
    fn e020_clones_a_constant_copy_for_a_second_element_size() {
        let mut body = vec![
            constant(0, Val(1)),
            copy(Val(1), Val(2), RegType::Lbr),
            copy(Val(2), Val(3), RegType::Lccr),
        ];
        let mut values = Values::default();
        let at = OpId::at(&[1]);
        let user = OpId::at(&[2]);

        let mut assignments = PrecisionAssignments::default();
        assignments.initialize_precision(at.clone(), &body[1]);
        assert_eq!(
            assignments.assign_precision_helper(&mut body, &mut values, &at, 0, Bits(8), &user),
            PrecisionAssigned::Assigned
        );
        assert_eq!(
            assignments.assign_precision_helper(&mut body, &mut values, &at, 0, Bits(8), &user),
            PrecisionAssigned::Unchanged
        );
        let outcome =
            assignments.assign_precision_helper(&mut body, &mut values, &at, 0, Bits(16), &user);

        let PrecisionAssigned::Substituted(clone_result) = outcome else {
            panic!("expected a clone, got {outcome:?}")
        };
        assert_eq!(
            body,
            vec![
                constant(0, Val(1)),
                copy(Val(1), Val(2), RegType::Lbr),
                copy(Val(1), clone_result, RegType::Lbr),
                copy(Val(2), Val(3), RegType::Lccr),
            ]
        );
        assert_eq!(assignments.slots(&at), Some(&[Some(Bits(8))][..]));
        assert_eq!(
            assignments.slots(&OpId::at(&[2])),
            Some(&[Some(Bits(16))][..])
        );
    }

    /// 283/656 — a loop-carried address takes the slot AFTER the induction variable's, and the loop's
    /// own result takes one past every carried value; a `uniform` region argument the loop does not
    /// bind is the pass's `Unknown parent op` failure and writes no slot.
    #[test]
    fn e283_assigns_the_iter_arg_slot_after_the_iv_and_the_result_slot_after_the_carried_values() {
        let carried = Carried {
            init: Val(1),
            arg: Val(2),
            result: Val(3),
            reg: Reg {
                locale: RegType::Lbr,
                index: None,
            },
            program_header: false,
            element_size: None,
        };
        let mut body = vec![Op::Sentient(sentient::Op::For {
            iv: Val(4),
            bound: Val(5),
            bound_reg: None,
            carried: vec![carried],
            dbg_name: None,
            body: vec![copy(Val(2), Val(6), RegType::Lbr)],
        })];
        let mut values = Values::default();
        let user = OpId::at(&[0, 0]);

        let mut assignments = PrecisionAssignments::default();
        assignments.initialize_precision(OpId::at(&[0]), &body[0]);

        assert_eq!(
            assignments.assign_precision(&mut body, &mut values, Val(2), Bits(32), &user),
            PrecisionAssigned::Assigned
        );
        assert_eq!(
            assignments.assign_precision(&mut body, &mut values, Val(3), Bits(16), &user),
            PrecisionAssigned::Assigned
        );
        // `[iv, arg, result]` — the iv's slot is untouched.
        assert_eq!(
            assignments.slots(&OpId::at(&[0])),
            Some(&[None, Some(Bits(32)), Some(Bits(16))][..])
        );
        assert_eq!(
            assignments.assign_precision(&mut body, &mut values, Val(99), Bits(32), &user),
            PrecisionAssigned::UnknownParentOp
        );
    }

    /// 428/656 — a fresh assignment queues the value; the same size again assigns nothing and must
    /// NOT queue it, which is the reference's stated infinite-loop guard.
    #[test]
    fn e428_queues_the_value_only_on_a_fresh_assignment() {
        let mut body = vec![
            constant(0, Val(1)),
            copy(Val(1), Val(2), RegType::Lbr),
            copy(Val(2), Val(3), RegType::Lccr),
        ];
        let mut values = Values::default();
        let user = OpId::at(&[2]);
        let mut worklist = AddressWorklist::default();

        let mut assignments = PrecisionAssignments::default();
        assignments.initialize_precision(OpId::at(&[1]), &body[1]);

        assert_eq!(
            assignments.add_to_worklist_and_assign_precision(
                &mut body,
                &mut values,
                &mut worklist,
                Val(2),
                Bits(8),
                &user,
            ),
            PrecisionAssigned::Assigned
        );
        assert_eq!(worklist.pop(), Some(Val(2)));
        assert_eq!(worklist.pop(), None);

        assert_eq!(
            assignments.add_to_worklist_and_assign_precision(
                &mut body,
                &mut values,
                &mut worklist,
                Val(2),
                Bits(8),
                &user,
            ),
            PrecisionAssigned::Unchanged
        );
        assert_eq!(worklist, AddressWorklist::default());
    }

    /// 494/656 — a loop RESULT propagates its size to the yield operand (with the YIELD as the user),
    /// to the iter operand and to the region iter arg, all off the ONE slot the result owns; a value
    /// no op of this body binds is the walk's own failure.
    #[test]
    fn e494_walks_a_loop_result_up_to_its_yield_its_init_and_its_iter_arg() {
        let add = Op::Sentient(sentient::Op::ScalarAdd {
            lhs: Val(4),
            rhs: Val(1),
            result: Val(6),
            reg: Some(Reg {
                locale: RegType::Lar,
                index: None,
            }),
            ty: ScalarTy::Index,
            element_size: None,
        });
        let mut body = vec![
            constant(0, Val(1)),
            Op::Sentient(sentient::Op::For {
                iv: Val(3),
                bound: Val(2),
                bound_reg: None,
                carried: vec![Carried {
                    init: Val(1),
                    arg: Val(4),
                    result: Val(5),
                    reg: Reg {
                        locale: RegType::Lar,
                        index: None,
                    },
                    program_header: false,
                    element_size: None,
                }],
                dbg_name: None,
                body: vec![
                    add.clone(),
                    Op::Sentient(sentient::Op::Yield {
                        results: vec![Val(6)],
                    }),
                ],
            }),
        ];
        let mut values = Values::default();
        let mut worklist = AddressWorklist::default();

        let mut assignments = PrecisionAssignments::default();
        assignments.initialize_precision(OpId::at(&[0]), &body[0]);
        assignments.initialize_precision(OpId::at(&[1]), &body[1]);
        assignments.initialize_precision(OpId::at(&[1, 0]), &add);
        // The seeding walk assigns the loop's result before its value can reach the worklist.
        assert_eq!(
            assignments.assign_precision(&mut body, &mut values, Val(5), Bits(32), &OpId::at(&[1])),
            PrecisionAssigned::Assigned
        );

        assert_eq!(
            assignments.process_address_ssa_value(&mut body, &mut values, &mut worklist, Val(5)),
            AddressWalk::Walked
        );

        // The yield's operand first, then the iter arg — the constant init is on the skip list.
        assert_eq!(worklist.pop(), Some(Val(6)));
        assert_eq!(worklist.pop(), Some(Val(4)));
        assert_eq!(worklist.pop(), None);
        assert_eq!(
            assignments.slots(&OpId::at(&[1, 0])),
            Some(&[Some(Bits(32))][..])
        );
        // `[iv, arg, result]`, the iv's slot still untouched.
        assert_eq!(
            assignments.slots(&OpId::at(&[1])),
            Some(&[None, Some(Bits(32)), Some(Bits(32))][..])
        );
        assert!(assignments.failures().is_empty());

        assert_eq!(
            assignments.process_address_ssa_value(&mut body, &mut values, &mut worklist, Val(99)),
            AddressWalk::NotALoopArgument
        );
    }
}

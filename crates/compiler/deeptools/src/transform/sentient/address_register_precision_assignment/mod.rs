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

use std::collections::BTreeMap;

use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;
use crate::formats::Bits;
use crate::islands::dataflow_ir::Values;
use crate::islands::sentient::dialects::{self, Op, Val, sentient, symbol, uniform};

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
    /// element_size"* (`AddressRegisterPrecisionAssignment.cpp:64-65`), keyed by the ORIGINAL copy's
    /// result.
    cached_copies: BTreeMap<Val, BTreeMap<Bits, Val>>,
}

impl PrecisionAssignments {
    /// Replaces: e019_initializePrecision
    ///
    /// One unassigned slot per value `op` binds.
    /// ⛔ A `sentient.for`'S VECTOR IS `1 + carried.len()` LONGER THAN ITS RESULT COUNT, laid out
    /// `[iv, iter_args…, results…]`: entry 283 reads an iter arg at `i + 1` and a result at
    /// `i + offset` with `offset = 1 + getNumRegionIterArgs()` (`:169-195`), so slot 0 is the
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
    /// ⛔ `uniform.equalize_pattern`, THE SIXTH MEMBER OF THE SKIP LIST (`:107-111`), HAS NO FORM AT
    /// THIS RUNG — `uniform::Op` declares four ops and nothing in this compiler emits it, so an op
    /// this list would skip cannot be built. The other five are matched below.
    /// ⛔ THE CLONE GOES BEFORE THE **USER**, not before the copy: `OpBuilder builder(user)` is
    /// anchored on the op that asked, which is what keeps the clone in scope for it (`:118`).
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
                 ({at:?}) (AddressRegisterPrecisionAssignment.cpp:113)"
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
            program_header,
        }) = op
        {
            Some((*input, *result, *reg, *program_header))
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
                 ({at:?}) (AddressRegisterPrecisionAssignment.cpp:474)"
            )
        };
        let Some(slot) = slots.get_mut(index) else {
            todo!(
                "assignPrecisionHelper: slot {index} past the {} initializePrecision gave {at:?} \
                 (AddressRegisterPrecisionAssignment.cpp:113)",
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

        if let Some((input, copy_result, reg, program_header)) = copy
            && copies_a_constant
        {
            // ⭐ AN LBR COPY IS COMMONED WHATEVER THE FLAG SAYS: it has to be promoted to the program
            // header anyway, so hoisting it does not lengthen its live range (`:121-124`).
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
                program_header,
            });
            if insert_at(body, user.path(), clone).is_some() {
                todo!(
                    "assignPrecisionHelper: `OpBuilder builder(user)` on a user absent from this \
                     unit body ({user:?}) (AddressRegisterPrecisionAssignment.cpp:118)"
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

// crustify:todo: e283_assignPrecision
//   authority : dcc/src/Transform/Sentient/AddressRegisterPrecisionAssignment.cpp:164  (31 body lines, level 1)
//   original  : bool AddressRegisterPrecisionAssignmentPass::assignPrecision(Value val, int element_size, Value &new_val, Operation *user)
//   calls     : e020_assignPrecisionHelper

// crustify:todo: e428_addToWorkListAndAssignPrecision
//   authority : dcc/src/Transform/Sentient/AddressRegisterPrecisionAssignment.cpp:204  (10 body lines, level 2)
//   original  : Value AddressRegisterPrecisionAssignmentPass::addToWorkListAndAssignPrecision( Value val, int element_size, Operation *user)
//   calls     : e283_assignPrecision

// crustify:todo: e494_processAddressSSAValue
//   authority : dcc/src/Transform/Sentient/AddressRegisterPrecisionAssignment.cpp:221  (223 body lines, level 3)
//   original  : LogicalResult AddressRegisterPrecisionAssignmentPass::processAddressSSAValue( Value val)
//   calls     : e428_addToWorkListAndAssignPrecision

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
        };
        let loop_op = Op::Sentient(sentient::Op::For {
            iv: Val(4),
            bound: Val(5),
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
}

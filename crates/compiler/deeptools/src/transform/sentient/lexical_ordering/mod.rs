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

//! `LexicalOrdering.cpp` — 4 of the campaign's 656 units (dependency level(s) [0, 1, 2]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e051_runOn` | 051 | 0 | 89 | `dcc/src/Transform/Sentient/LexicalOrdering.cpp:113` |
//! | `e302_matchAndRewrite` | 302 | 1 | 40 | `dcc/src/Transform/Sentient/LexicalOrdering.cpp:50` |
//! | `e303_runOn` | 303 | 1 | 8 | `dcc/src/Transform/Sentient/LexicalOrdering.cpp:203` |
//! | `e439_runOnOperation` | 439 | 2 | 12 | `dcc/src/Transform/Sentient/LexicalOrdering.cpp:99` |

use crate::arch::Arch;
use crate::islands::dataflow_ir::dialects as lower;
use crate::islands::dataflow_ir::dialects::{dataflow, uniform};
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{self, Definitions, Op, Val, arith, sentient};
use crate::islands::sentient::{Program, Run};
use crate::model::Model;
use crate::units::Residency;
use crate::workload::Workload;

/// THE FUNCTION'S ENTRY BLOCK — `func.getBody().front()`, the one block a constant is hoisted INTO.
///
/// ⭐ WHICH OPS IT HOLDS IS SETTLED BY THE PRINTER: the preamble, then one `dataflow.program_unit`
/// per unit ([`crate::islands::sentient::print`]). So index 0 of it is `preamble[0]` when the
/// preamble is non-empty, and otherwise the first (never-constant) `dataflow.program_unit`.
const ENTRY: u32 = 0;

/// ONE `arith.constant`'s LITERAL AS `APInt::slt` SEES IT — sign-extended by its own width.
///
/// ⛔ `arith.constant true` IS MINUS ONE. `APInt::slt` sign-extends each side to the wider width, so
/// the 1-bit `1` of `true` sorts BELOW every non-negative constant.
fn sign_extended(value: arith::IntConst) -> i64 {
    match value {
        arith::IntConst::Bool(bit) => {
            if bit {
                -1
            } else {
                0
            }
        }
        arith::IntConst::Int { value, bits } => {
            if (1..64).contains(&bits) {
                let shift = 64 - bits;
                (value << shift) >> shift
            } else {
                value
            }
        }
    }
}

/// WHAT `less_than` COMPARES — the sort key of one hoistable constant.
///
/// ⛔ THE DIALECT IS THE ORDERING'S FIRST TERM: every `arith.constant` sorts before every
/// `sentient.scalar_constant` (`LexicalOrdering.cpp:118-119`, `:141-142`), so which op it is belongs
/// in the key rather than in the comparison's caller.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ConstKey {
    /// `arith.constant` carrying an `IntegerAttr` — the value, then the type.
    Arith {
        /// The literal, already [`sign_extended`].
        value: i64,
        /// `getType()`, whose spelling breaks a value tie.
        ty: ScalarTy,
    },
    /// `arith.constant dense<..>` — ⛔ NO KEY THE REFERENCE CAN READ. See [`ConstKey::lt`].
    ArithDense,
    /// `sentient.scalar_constant` — the value, then the type.
    Sentient {
        /// `getValue()`, an `si64` attribute.
        value: i64,
        /// `getType()`, whose spelling breaks a value tie.
        ty: ScalarTy,
    },
}

impl ConstKey {
    /// `less_than` (`LexicalOrdering.cpp:114-158`) — the strict weak ordering the sort and the
    /// contiguity test share.
    ///
    /// ⛔ EQUAL VALUES OF DIFFERENT TYPES FALL BACK TO THE TYPE **SPELLING**, which the reference
    /// says out loud: *"A deterministic ordering is required so use the type strings."*
    ///
    /// ⛔ THIS DIVERGES FROM THE REFERENCE FOR EQUAL-VALUED NEGATIVES OF DIFFERENT WIDTHS, and it is the
    /// reference that is wrong. `v_a != v_b` (`:125`) compares APInt RAW WORDS — a width mismatch its
    /// assert catches only in a debug build — so `-1 : i32` and `-1 : index` DIFFER there and DO reach
    /// `slt`, which sign-extends each side by its OWN width and answers false both ways: a TIE, and the
    /// tie path RAUWs one onto the other. [`sign_extended`] has already made them equal here, so the
    /// type-spelling path orders them and they never tie — the side that does not rewire an `i32`
    /// reader onto an `index` value.
    fn lt(&self, other: &ConstKey) -> bool {
        use ConstKey::{Arith, ArithDense, Sentient};
        match (self, other) {
            // ⛔ THE REFERENCE ABORTS HERE, IT DOES NOT ORDER THEM: `mlir::cast<mlir::IntegerAttr>`
            // on a `DenseIntElementsAttr` is a failed cast (`:121-124`), not a null. Reached only
            // when a dense constant is actually compared — a lone one is still hoisted.
            (ArithDense, Arith { .. } | ArithDense) | (Arith { .. }, ArithDense) => todo!(
                "LexicalOrdering::less_than — mlir::cast<IntegerAttr> aborts on an arith.constant \
                 with a dense attribute (LexicalOrdering.cpp:122)"
            ),
            (Arith { .. } | ArithDense, Sentient { .. }) => true,
            (Sentient { .. }, Arith { .. } | ArithDense) => false,
            (
                Arith { value: a, ty: ty_a } | Sentient { value: a, ty: ty_a },
                Arith { value: b, ty: ty_b } | Sentient { value: b, ty: ty_b },
            ) => {
                if a != b {
                    a < b
                } else if ty_a == ty_b {
                    false
                } else {
                    ty_a.spelling() < ty_b.spelling()
                }
            }
        }
    }

    /// Whether two keys tie — `!less_than(a, b) && !less_than(b, a)`, the duplicate test (`:190`).
    fn ties(&self, other: &ConstKey) -> bool {
        !self.lt(other) && !other.lt(self)
    }
}

/// THE SORT KEY AND RESULT OF ONE HOISTABLE OP — `isa<arith::ConstantOp>` or
/// `isa<sentient::ConstantOp>` (`:163-165`), and nothing else.
///
/// ⛔ `sentient.vector_constant` IS NOT A `sentient::ConstantOp` — it is `Sentient_VectorConstantOp`,
/// a different op class — so it is not collected and not hoisted.
///
/// ⭐ ONE RESULT, NOT A LIST: the reference's `DT_CHECK(op->getNumResults() == prev_op->...)` at
/// `:192` is discharged by the shape of every constant op, so it is a field and not a check.
fn constant_of(op: &Op) -> Option<(ConstKey, Val)> {
    match op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value, result, ty, ..
        }) => Some((
            ConstKey::Sentient {
                value: *value,
                ty: *ty,
            },
            *result,
        )),
        Op::Arith(arith::Op::Constant { result, value }) => Some((
            ConstKey::Arith {
                value: *value,
                ty: ScalarTy::Index,
            },
            *result,
        )),
        Op::Arith(arith::Op::ConstantInt { result, value }) => Some((
            ConstKey::Arith {
                value: sign_extended(*value),
                ty: value.ty(),
            },
            *result,
        )),
        Op::Arith(arith::Op::DenseConstant { result, .. }) => Some((ConstKey::ArithDense, *result)),
        // ⛔ NO `_` ARM — a dialect reaching this rung must be a build error, per the island's rule.
        Op::Sentient(_)
        | Op::Arith(_)
        | Op::UniformRegions(_)
        | Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_)
        | Op::Affine(_)
        | Op::AffineFor(_)
        | Op::Vector(_)
        | Op::Scf(_)
        | Op::Symbol(_)
        | Op::Uniform(_) => None,
    }
}

/// [`constant_of`] for an op inside a shared-dialect REGION — an `scf.if` body holds the rung below's
/// ops, and the reference's walk descends into it.
fn constant_of_lower(op: &lower::Op) -> Option<(ConstKey, Val)> {
    match op {
        lower::Op::Arith(inner) => constant_of(&Op::Arith(inner.clone())),
        lower::Op::Dataflow(_)
        | lower::Op::Agen(_)
        | lower::Op::VectorChain(_)
        | lower::Op::Affine(_)
        | lower::Op::Vector(_)
        | lower::Op::Scf(_)
        | lower::Op::Symbol(_)
        | lower::Op::Uniform(_) => None,
    }
}

/// ONE CONSTANT THE WALK FOUND, and where — `func_entry_ops` plus what `++it` needs.
struct Found {
    /// Its `less_than` key.
    key: ConstKey,
    /// The value it binds, which a duplicate's readers are moved onto.
    result: Val,
    /// Which block held it, numbered in walk order. ⛔ CONTIGUITY IS WITHIN ONE BLOCK: the
    /// reference's `++it` walks the previous constant's OWN block, so the last op of a nested region
    /// is never adjacent to the op after that region.
    block: u32,
    /// Its index in that block.
    index: usize,
}

/// `func->walk<WalkOrder::PreOrder>` — every op, then every op's regions, both rungs.
struct Collect {
    /// The next block number to hand out; the entry block is [`ENTRY`].
    next_block: u32,
    /// `func_entry_ops`, in walk order.
    found: Vec<Found>,
}

impl Collect {
    fn fresh(&mut self) -> u32 {
        let block = self.next_block;
        self.next_block += 1;
        block
    }

    /// One block of this rung, at a block number already handed out.
    fn ops(&mut self, ops: &[Op], block: u32) {
        for (index, op) in ops.iter().enumerate() {
            if let Some((key, result)) = constant_of(op) {
                self.found.push(Found {
                    key,
                    result,
                    block,
                    index,
                });
            }
            match op {
                Op::Sentient(inner) => {
                    for region in sentient::regions(inner) {
                        let nested = self.fresh();
                        self.ops(region, nested);
                    }
                }
                Op::AffineFor(loop_op) => {
                    let nested = self.fresh();
                    self.ops(&loop_op.body, nested);
                }
                other => {
                    if let Some(low) = dialects::lowered(other) {
                        for region in lower::regions(&low) {
                            let nested = self.fresh();
                            self.ops_lower(region, nested);
                        }
                    }
                }
            }
        }
    }

    /// One block of the rung below — reached through a shared dialect's region.
    fn ops_lower(&mut self, ops: &[lower::Op], block: u32) {
        for (index, op) in ops.iter().enumerate() {
            if let Some((key, result)) = constant_of_lower(op) {
                self.found.push(Found {
                    key,
                    result,
                    block,
                    index,
                });
            }
            for region in lower::regions(op) {
                let nested = self.fresh();
                self.ops_lower(region, nested);
            }
        }
    }
}

/// TAKE ONE COLLECTED CONSTANT OUT OF WHEREVER IT SITS — the first half of both `moveBefore` and
/// `erase()`.
fn take_from(ops: &mut Vec<Op>, wanted: &[Val], taken: &mut Vec<(Val, Op)>) {
    let mut at = 0;
    while at < ops.len() {
        if let Some((_, result)) = constant_of(&ops[at])
            && wanted.contains(&result)
        {
            taken.push((result, ops.remove(at)));
            continue;
        }
        match &mut ops[at] {
            Op::Sentient(inner) => {
                for region in sentient::regions_mut(inner) {
                    take_from(region, wanted, taken);
                }
            }
            Op::AffineFor(loop_op) => take_from(&mut loop_op.body, wanted, taken),
            other => {
                if let Some(mut low) = dialects::lowered(other) {
                    for region in lower::regions_mut(&mut low) {
                        take_from_lower(region, wanted, taken);
                    }
                    *other = dialects::raised(low);
                }
            }
        }
        at += 1;
    }
}

/// [`take_from`] inside a shared dialect's region — the op comes back UP a rung, since the entry
/// block it is hoisted into is this rung's.
fn take_from_lower(ops: &mut Vec<lower::Op>, wanted: &[Val], taken: &mut Vec<(Val, Op)>) {
    let mut at = 0;
    while at < ops.len() {
        if let Some((_, result)) = constant_of_lower(&ops[at])
            && wanted.contains(&result)
        {
            taken.push((result, dialects::raised(ops.remove(at))));
            continue;
        }
        for region in lower::regions_mut(&mut ops[at]) {
            take_from_lower(region, wanted, taken);
        }
        at += 1;
    }
}

/// Replaces: e051_runOn
///
/// Hoists every constant in the program to the head of its entry block, sorted and de-duplicated.
///
/// ⛔ THE WHOLE EFFECT, NOT THE PREDICATE: constants leave the unit bodies they were written in, a
/// duplicate's readers are rewired onto the survivor and the duplicate is dropped.
///
/// ⛔ A NO-OP WHEN THE CONSTANTS ARE ALREADY A CONTIGUOUS, STRICTLY INCREASING PREFIX of the entry
/// block (`:167-178`) — otherwise EVERY collected constant moves, including ones already in place.
///
/// ⛔ NAMED FOR ITS ARGUMENT because `runOn(func::FuncOp)` and `runOn(ModuleOp)` (e303) are one
/// overload set in C++ and cannot both be `run_on` here.
///
/// ⛔ THE SORT IS STABLE WHERE `std::sort` IS NOT, which decides WHICH of a tie survives; the
/// reference leaves that arbitrary.
pub(crate) fn run_on_program<A: Arch, M: Model, W: Workload>(program: &mut Program<A, M, W>) {
    let mut collect = Collect {
        next_block: ENTRY + 1,
        found: Vec::new(),
    };
    collect.ops(&program.preamble, ENTRY);
    // The entry block's remaining ops are one `dataflow.program_unit` per unit — none is a constant,
    // and each one's region is a block of its own.
    for unit in program.units.iter() {
        let nested = collect.fresh();
        collect.ops(&unit.body, nested);
    }
    let mut found = collect.found;

    // `do_transform` (`:167-178`). ⛔ THE `||` SHORT-CIRCUITS: a constant that is not adjacent to the
    // previous one is never compared to it, which is what keeps a lone dense constant out of `lt`.
    let mut transform = false;
    for (at, current) in found.iter().enumerate() {
        if transform {
            break;
        }
        transform = match at.checked_sub(1).map(|prev| &found[prev]) {
            None => current.block != ENTRY || current.index != 0,
            Some(prev) => {
                prev.block != current.block
                    || prev.index + 1 != current.index
                    || !prev.key.lt(&current.key)
            }
        };
    }
    if !transform {
        return;
    }

    found.sort_by(|a, b| {
        if a.key.lt(&b.key) {
            core::cmp::Ordering::Less
        } else if b.key.lt(&a.key) {
            core::cmp::Ordering::Greater
        } else {
            core::cmp::Ordering::Equal
        }
    });

    let wanted: Vec<Val> = found.iter().map(|f| f.result).collect();
    let mut taken: Vec<(Val, Op)> = Vec::new();
    take_from(&mut program.preamble, &wanted, &mut taken);
    for unit in program.units.iter_mut() {
        take_from(&mut unit.body, &wanted, &mut taken);
    }

    // `:184-200`, walked back to front: the largest is placed at the entry block's head first, so
    // each survivor inserted before it leaves the block ascending.
    let mut rewires: Vec<(Val, Val)> = Vec::new();
    let mut kept: Option<&Found> = None;
    for current in found.iter().rev() {
        if let Some(survivor) = kept
            && current.key.ties(&survivor.key)
        {
            // "Replace and discard duplicates." ⛔ RAUW BEFORE THE ERASE — an op destroyed with uses
            // left is an MLIR abort; here the op is already out and only the rewire is owed.
            rewires.push((current.result, survivor.result));
            continue;
        }
        kept = Some(current);
        if let Some(position) = taken.iter().position(|(val, _)| *val == current.result) {
            let (_, op) = taken.remove(position);
            program.preamble.insert(0, op);
        }
    }

    for (of, with) in rewires {
        dialects::replace_all_uses_with(&mut program.preamble, of, with);
        for unit in program.units.iter_mut() {
            dialects::replace_all_uses_with(&mut unit.body, of, with);
        }
    }
}

/// WHAT `less_than` COMPARES — one `dataflow.get_unit` key's place in the lexical order.
///
/// ⛔ THE `type` ATTRIBUTE IS THE FIRST TERM AND IT IS A STRING. `getType()` on a `get_unit` is the
/// generated `$type` StrAttr accessor, not a result type — `get_unit`'s results are `Variadic<Index>`
/// so ODS generates no result-type getter — and `StringRef::operator<` is lexicographic over exactly
/// [`crate::units::DfirUnit::spelling`].
/// ⛔ `getResultNumber()` IS NOT A TERM HERE: this island's `dataflow.get_unit` binds ONE `result`, so
/// the reference's last tie-break is always `0 < 0` and the second `DT_CHECK_MSG` is unreachable once
/// the first cast succeeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct UnitKey {
    /// `unit_op.getType()`.
    ty: &'static str,
    /// `dcc::getCoreId` — the `core` attribute, or `-1` where the unit carries none
    /// (`Utils/DccExtContext.cpp:78-87`).
    core: i32,
    /// `dcc::getCoreletId` — the `corelet` attribute, or `-1` (`:115-124`).
    corelet: i32,
}

/// ONE MAPPING KEY'S [`UnitKey`] — `None` where it is not a `dataflow.get_unit`, which is the
/// `DT_CHECK_MSG(unit_op_a && unit_op_b, ..)` the reference makes of that case.
#[must_use]
fn unit_key(key: Val, defs: Definitions<'_>) -> Option<UnitKey> {
    let Some(Op::Dataflow(dataflow::Op::GetUnit {
        residency, unit, ..
    })) = defs.of(key)
    else {
        return None;
    };
    // The two attributes `createGetUnitOp` writes, read back off the residency that decides them.
    let (core, corelet) = match residency {
        Residency::Global => (-1, -1),
        Residency::Scratchpad { core } => (i32::try_from(core.get()).unwrap_or(-1), -1),
        Residency::CoreWide { core } => (i32::try_from(core.get()).unwrap_or(-1), 0),
        Residency::Corelet { core, corelet } => (
            i32::try_from(core.get()).unwrap_or(-1),
            i32::try_from(corelet.get()).unwrap_or(-1),
        ),
    };
    Some(UnitKey {
        ty: unit.spelling(),
        core,
        corelet,
    })
}

/// WHETHER ONE RUN OF THE PATTERN APPLIED — `LogicalResult` as data, as `ToggleDuplication` is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reordered {
    /// `success()` — the keys were out of order and the op now carries them sorted.
    Sorted,
    /// `failure()` — already sorted, so the pattern does not apply.
    AlreadySorted,
}

/// Replaces: e302_matchAndRewrite
///
/// Sorts a `uniform.def_immutable_mapping`'s key/value pairs by unit type, then core, then corelet.
///
/// ⛔ AN IN-PLACE REORDER WHERE THE REFERENCE CALLS `replaceOpWithNewOp`, and that is the whole
/// difference: the rebuilt op has the same result types and the same pairs in a new order, so the
/// rewriter's implied `replaceAllUsesWith` is an identity and the surviving `result` is this one.
/// ⛔ THE PERMUTATION SORT IS STABLE WHERE `std::sort` IS NOT, which decides the order of two keys
/// with the same unit, core and corelet — a case the reference's second `DT_CHECK_MSG` calls
/// "redundant keys".
/// ⛔ A KEY THAT IS NOT A `dataflow.get_unit` SORTS FIRST rather than asserting.
pub(crate) fn order_mapping_keys_lexically(op: &mut Op, defs: Definitions<'_>) -> Reordered {
    let Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. }) = op else {
        return Reordered::AlreadySorted;
    };
    let keys: Vec<Option<UnitKey>> = pairs.iter().map(|(key, _)| unit_key(*key, defs)).collect();
    // `std::is_sorted(keys.begin(), keys.end(), less_than)` — no adjacent pair out of order.
    if keys.windows(2).all(|adjacent| adjacent[0] <= adjacent[1]) {
        return Reordered::AlreadySorted;
    }
    let mut indices: Vec<usize> = (0..pairs.len()).collect();
    indices.sort_by_key(|at| keys[*at]);
    let ordered: Vec<(Val, Val)> = indices.iter().map(|at| pairs[*at]).collect();
    *pairs = ordered;
    Reordered::Sorted
}

/// Replaces: e303_runOn
///
/// Runs the constant hoist on every function of the module.
///
/// ⛔ `WalkResult::skip()` — *"no nested functions"* — IS A FACT ABOUT THE ISLAND HERE: a [`Run`] is a
/// FLAT list of programs, one `func.func` each, so there is no nested function for a pre-order walk to
/// reach and nothing to skip past.
pub(crate) fn run_on_module<A: Arch, M: Model, W: Workload>(run: &mut Run<A, M, W>) {
    for program in &mut run.programs {
        run_on_program(program);
    }
}

/// `-dcc-lexical-ordering-disable`, `cl::init(false)` (`LexicalOrdering.cpp:40-42`).
const DISABLE_THIS_PASS: bool = false;

/// Replaces: e439_runOnOperation
///
/// The pass: sort every `uniform.def_immutable_mapping`'s pairs, then hoist the constants.
///
/// ⭐ ONE SWEEP IS THE GREEDY FIXPOINT — [`order_mapping_keys_lexically`] leaves the op sorted, so a
/// second application answers [`Reordered::AlreadySorted`]; and a pattern that never signals failure
/// cannot fail the driver, so `signalPassFailure()` (`:107`) is unreachable.
pub fn run_on_operation<A: Arch, M: Model, W: Workload>(run: &mut Run<A, M, W>) {
    if DISABLE_THIS_PASS {
        return;
    }
    for program in &mut run.programs {
        order_every_mapping(program);
    }
    run_on_module(run);
}

/// `applyPatternsGreedily` OVER ONE PROGRAM, READ THEN WRITE: a mapping's keys are `dataflow.get_unit`
/// results in the preamble, which is one of the scopes the sort then writes into.
fn order_every_mapping<A: Arch, M: Model, W: Workload>(program: &mut Program<A, M, W>) {
    let mut sorted: Vec<(Val, Vec<(Val, Val)>)> = Vec::new();
    {
        let preamble: &[Op] = &program.preamble;
        sortings_of(preamble, &[preamble], &mut sorted);
        for unit in program.units.iter() {
            let enclosing: [&[Op]; 2] = [&unit.body, preamble];
            sortings_of(&unit.body, &enclosing, &mut sorted);
        }
    }
    if sorted.is_empty() {
        return;
    }
    reorder_mappings(&mut program.preamble, &sorted);
    for unit in program.units.iter_mut() {
        reorder_mappings(&mut unit.body, &sorted);
    }
}

/// The read half: the pairs each mapping in `scope` wants, by the [`Val`] it binds.
fn sortings_of(scope: &[Op], enclosing: &[&[Op]], sorted: &mut Vec<(Val, Vec<(Val, Val)>)>) {
    let defs = Definitions::from_innermost(enclosing);
    for op in scope {
        if let Op::Uniform(uniform::Op::DefImmutableMapping { result, .. }) = op {
            let mut candidate = op.clone();
            if order_mapping_keys_lexically(&mut candidate, defs) == Reordered::Sorted
                && let Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. }) = candidate
            {
                sorted.push((*result, pairs));
            }
        }
        for region in dialects::regions_ref(op) {
            sortings_of(region, enclosing, sorted);
        }
    }
}

/// The write half of [`order_every_mapping`].
fn reorder_mappings(scope: &mut [Op], sorted: &[(Val, Vec<(Val, Val)>)]) {
    for op in scope {
        if let Op::Uniform(uniform::Op::DefImmutableMapping { result, pairs }) = op
            && let Some((_, ordered)) = sorted.iter().find(|(val, _)| val == result)
        {
            *pairs = ordered.clone();
        }
        for region in dialects::regions_mut(op) {
            reorder_mappings(region, sorted);
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        Definitions, Reordered, order_mapping_keys_lexically, run_on_module, run_on_operation,
        run_on_program,
    };
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::KernelName;
    use crate::islands::dataflow_ir::dialects::arith::IntBinary;
    use crate::islands::dataflow_ir::dialects::{dataflow, uniform};
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::dialects::{Op, Val, arith, sentient};
    use crate::islands::sentient::{Program, ProgramUnit, ProgramUnits, Run};
    use crate::model::Model;
    use crate::units::{Core, Corelet, DfirUnit, Residency};
    use crate::workload::Workload;

    /// A model, so the program is typed; nothing here reads it.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyModel;
    impl Model for AnyModel {
        const QUERY_HEADS: u32 = 32;
        const KV_HEADS: u32 = 8;
        const HEAD_DIM: u32 = 64;
        const HIDDEN: u32 = 2048;
        const LAYERS: u32 = 40;
        const FFN: u32 = 8192;
        const VOCAB: u32 = 49152;
    }

    /// A decode rung, for the same reason.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyRung;
    impl Workload for AnyRung {
        const ROWS: u32 = 1;
        const ACTIVE_CAP: u32 = 64;
    }

    /// `%r = arith.constant N : index`.
    fn index_const(result: u32, value: i64) -> Op {
        Op::Arith(arith::Op::Constant {
            result: Val(result),
            value,
        })
    }

    /// `%r = sentient.scalar_constant {value = N : si64} : index`.
    fn scalar_const(result: u32, value: i64) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result: Val(result),
            reg_locale: sentient::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// `%r = arith.addi %a, %a : index` — a reader, so a rewire is observable.
    fn adds(result: u32, operand: u32) -> Op {
        Op::Arith(arith::Op::AddI(IntBinary {
            result: Val(result),
            lhs: Val(operand),
            rhs: Val(operand),
            ty: ScalarTy::Index,
        }))
    }

    /// A one-unit program with `preamble` before it and `body` inside it.
    fn program_of(preamble: Vec<Op>, body: Vec<Op>) -> Program<Dd2, AnyModel, AnyRung> {
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble,
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(DfirUnit::Sfp, Val(0)),
                    precision: None,
                    body,
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            bound: core::marker::PhantomData,
        }
    }

    /// e051 — constants leave the unit body for the entry block's head, sorted, and the duplicate's
    /// reader is rewired onto the survivor.
    #[test]
    fn constants_are_hoisted_sorted_and_deduplicated() {
        let mut program = program_of(
            Vec::new(),
            vec![
                index_const(1, 5),
                scalar_const(2, 3),
                index_const(3, 5),
                adds(4, 1),
            ],
        );

        run_on_program(&mut program);

        // Every `arith.constant` sorts before every `sentient.scalar_constant`, and the LAST of a tie
        // in sorted order is the one the back-to-front walk keeps.
        assert_eq!(
            program.preamble,
            vec![index_const(3, 5), scalar_const(2, 3)]
        );
        assert_eq!(
            program.units.iter().next().map(|unit| unit.body.clone()),
            Some(vec![adds(4, 3)]),
            "the duplicate %1 is gone and its reader now reads %3"
        );
    }

    /// e051 — the negative: a contiguous, strictly increasing prefix of the entry block is left alone.
    #[test]
    fn an_already_ordered_prefix_is_not_touched() {
        let before = program_of(vec![index_const(0, 1), index_const(1, 2)], vec![adds(2, 0)]);
        let mut program = before.clone();

        run_on_program(&mut program);

        assert_eq!(program, before);
    }

    /// `%r = dataflow.get_unit {type, core, corelet} : index`.
    fn get_unit(result: u32, unit: DfirUnit, residency: Residency) -> Op {
        Op::Dataflow(dataflow::Op::GetUnit {
            result: Val(result),
            residency,
            unit,
            num_folds: None,
        })
    }

    /// One core, and one of its corelets.
    fn corelet(core: u32, corelet: u32) -> Residency {
        Residency::Corelet {
            core: Core::checked(core).unwrap_or_else(|| unreachable!("core 0 exists")),
            corelet: Corelet::checked(corelet).unwrap_or_else(|| unreachable!("corelet 0 exists")),
        }
    }

    /// e302 — the pairs come out ordered by unit type, then core, then corelet, values carried along;
    /// and the same mapping a second time is already sorted.
    #[test]
    fn mapping_keys_are_sorted_by_unit_then_core_then_corelet() {
        let scope = vec![
            get_unit(1, DfirUnit::Sfp, corelet(0, 1)),
            get_unit(2, DfirUnit::Lxlu, corelet(0, 0)),
            get_unit(3, DfirUnit::Sfp, corelet(0, 0)),
        ];
        let regions: Vec<&[Op]> = vec![&scope];
        let defs = Definitions::from_innermost(&regions);
        let mut op = Op::Uniform(uniform::Op::DefImmutableMapping {
            result: Val(10),
            pairs: vec![(Val(1), Val(20)), (Val(2), Val(21)), (Val(3), Val(22))],
        });

        assert_eq!(
            order_mapping_keys_lexically(&mut op, defs),
            Reordered::Sorted
        );
        assert_eq!(
            op,
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(10),
                // `lxlu` before `sfp`, then corelet 0 before corelet 1.
                pairs: vec![(Val(2), Val(21)), (Val(3), Val(22)), (Val(1), Val(20))],
            })
        );
        assert_eq!(
            order_mapping_keys_lexically(&mut op, defs),
            Reordered::AlreadySorted
        );
    }

    /// e303 — every program of the run is hoisted, not just the first.
    #[test]
    fn the_module_walk_hoists_each_program() {
        let mut run = Run {
            kernel: KernelName(GroupId(0)),
            programs: vec![
                program_of(Vec::new(), vec![index_const(1, 5), adds(2, 1)]),
                program_of(Vec::new(), vec![scalar_const(3, 7), adds(4, 3)]),
            ],
        };

        run_on_module(&mut run);

        assert_eq!(run.programs[0].preamble, vec![index_const(1, 5)]);
        assert_eq!(run.programs[1].preamble, vec![scalar_const(3, 7)]);
    }

    /// e439 — both halves run: the unit body's mapping comes out sorted against the
    /// `dataflow.get_unit`s of the preamble, and then the body's constant is hoisted out of it.
    #[test]
    fn the_pass_orders_every_mapping_and_then_hoists() {
        let mut run = Run {
            kernel: KernelName(GroupId(0)),
            programs: vec![program_of(
                vec![
                    get_unit(1, DfirUnit::Sfp, corelet(0, 1)),
                    get_unit(2, DfirUnit::Lxlu, corelet(0, 0)),
                ],
                vec![
                    Op::Uniform(uniform::Op::DefImmutableMapping {
                        result: Val(10),
                        pairs: vec![(Val(1), Val(20)), (Val(2), Val(21))],
                    }),
                    index_const(3, 5),
                ],
            )],
        };

        run_on_operation(&mut run);

        assert_eq!(
            run.programs[0]
                .units
                .iter()
                .next()
                .map(|unit| unit.body.clone()),
            Some(vec![Op::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(10),
                pairs: vec![(Val(2), Val(21)), (Val(1), Val(20))],
            })])
        );
        assert_eq!(run.programs[0].preamble[0], index_const(3, 5));
    }
}

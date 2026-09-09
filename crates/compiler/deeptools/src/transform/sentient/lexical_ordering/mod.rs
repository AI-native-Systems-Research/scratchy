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

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so every item below is reachable only from
// this file's own tests until `e439_runOnOperation` lands and something calls it. CI runs clippy with
// `-D warnings`, so without this the first ported leaf of the module fails the gate.
// ⭐ REMOVE THIS WITH `e439_runOnOperation`: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use crate::arch::Arch;
use crate::islands::dataflow_ir::dialects as lower;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::Program;
use crate::islands::sentient::dialects::{self, Op, Val, arith, sentient};
use crate::model::Model;
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
/// `sentient.scalar_constant` (`LexicalOrdering.cpp:117-119`, `:139-141`), so which op it is belongs
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
    /// ⛔ THIS DIVERGES FROM THE REFERENCE FOR EQUAL-VALUED NEGATIVES OF DIFFERENT WIDTHS. It reaches
    /// `slt` only after `v_a != v_b`, an APInt comparison that is not sign-extending, so a `-1 : i32`
    /// and a `-1 : index` take its type-string path and take the equal path here — where the equal
    /// path RAUWs one onto the other. Sign extension is the ordering `slt` intends, and this is the
    /// side of the disagreement that does not rewire an `i32` reader onto an `index` value.
    fn lt(&self, other: &ConstKey) -> bool {
        use ConstKey::{Arith, ArithDense, Sentient};
        match (self, other) {
            // ⛔ THE REFERENCE ABORTS HERE, IT DOES NOT ORDER THEM: `mlir::cast<mlir::IntegerAttr>`
            // on a `DenseIntElementsAttr` is a failed cast (`:122-125`), not a null. Reached only
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

    /// Whether two keys tie — `!less_than(a, b) && !less_than(b, a)`, the duplicate test (`:187`).
    fn ties(&self, other: &ConstKey) -> bool {
        !self.lt(other) && !other.lt(self)
    }
}

/// THE SORT KEY AND RESULT OF ONE HOISTABLE OP — `isa<arith::ConstantOp>` or
/// `isa<sentient::ConstantOp>` (`:164-166`), and nothing else.
///
/// ⛔ `sentient.vector_constant` IS NOT A `sentient::ConstantOp` — it is `Sentient_VectorConstantOp`,
/// a different op class — so it is not collected and not hoisted.
///
/// ⭐ ONE RESULT, NOT A LIST: the reference's `DT_CHECK(op->getNumResults() == prev_op->...)` at
/// `:189` is discharged by the shape of every constant op, so it is a field and not a check.
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
/// block (`:166-179`) — otherwise EVERY collected constant moves, including ones already in place.
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

    // `do_transform` (`:166-179`). ⛔ THE `||` SHORT-CIRCUITS: a constant that is not adjacent to the
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

    // `:182-199`, walked back to front: the largest is placed at the entry block's head first, so
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

// crustify:todo: e302_matchAndRewrite
//   authority : dcc/src/Transform/Sentient/LexicalOrdering.cpp:50  (40 body lines, level 1)
//   original  : LogicalResult matchAndRewrite(uniform::DefImmutableMappingOp op, PatternRewriter& rewriter) const override
//   calls     : e252_size

// crustify:todo: e303_runOn
//   authority : dcc/src/Transform/Sentient/LexicalOrdering.cpp:203  (8 body lines, level 1)
//   original  : void runOn(ModuleOp module_op)
//   calls     : e051_runOn

// crustify:todo: e439_runOnOperation
//   authority : dcc/src/Transform/Sentient/LexicalOrdering.cpp:99  (12 body lines, level 2)
//   original  : void runOnOperation()
//   calls     : e051_runOn, e303_runOn

#[cfg(test)]
mod unit_tests {
    use super::run_on_program;
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::dialects::arith::IntBinary;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::dialects::{Op, Val, arith, sentient};
    use crate::islands::sentient::{Program, ProgramUnit, ProgramUnits};
    use crate::model::Model;
    use crate::units::DfirUnit;
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
}

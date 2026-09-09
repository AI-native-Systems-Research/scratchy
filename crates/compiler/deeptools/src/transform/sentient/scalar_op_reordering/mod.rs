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

//! `ScalarOpReordering.cpp` — 12 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e174_ScalarOpReorderingPass` | 174 | 0 | 4 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:66` |
//! | `e175_runOn` | 175 | 0 | 9 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:79` |
//! | `e176_computeOpIndexing` | 176 | 0 | 8 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:98` |
//! | `e177_getLiverangeEndPt` | 177 | 0 | 9 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:109` |
//! | `e178_findAncestorInBlock` | 178 | 0 | 7 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:374` |
//! | `e367_runOnOperation` | 367 | 1 | 5 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:74` |
//! | `e368_getFirstUseWithinBlock` | 368 | 1 | 23 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:383` |
//! | `e369_getLastUseWithinBlock` | 369 | 1 | 23 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:407` |
//! | `e370_localeHasFreeRegs` | 370 | 1 | 30 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:431` |
//! | `e470_isCandidateForReordering` | 470 | 2 | 67 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:462` |
//! | `e531_findAndProcessCandidates` | 531 | 3 | 165 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:207` |
//! | `e579_runOn` | 579 | 4 | 15 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:191` |

#![allow(dead_code)]
// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET — `e367_runOnOperation` (level 1) is what calls
// [`ScalarOpReordering::run_on_program`], and every item below is reachable only from this file's own
// tests until it lands. CI runs clippy with `-D warnings`. ⭐ REMOVE THIS WITH e367.

use std::collections::BTreeMap;

use crate::arch::Arch;
use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;
use crate::islands::sentient::dialects::{self as dialects, Op, Val, sentient};
use crate::islands::sentient::{Program, ProgramUnit};
use crate::model::Model;
use crate::workload::Workload;

/// THE NUMBER OF REGISTER LOCALES — fourteen, which is `getMaxEnumValForSentientRegType() + 1`.
const LOCALES: usize = 14;

/// Which slot of a [`PerLocale`] table a locale owns — the `.td`'s own numbering
/// (`SentientTypes.td:254-267`), exhaustive so a fifteenth locale is a build error here rather than a
/// silent alias of the fourteenth.
const fn locale_slot(locale: sentient::RegType) -> usize {
    match locale {
        sentient::RegType::Unknown => 0,
        sentient::RegType::Imm => 1,
        sentient::RegType::Jcr => 2,
        sentient::RegType::Lccr => 3,
        sentient::RegType::Lrf => 4,
        sentient::RegType::XrfRdPtr => 5,
        sentient::RegType::XrfWrPtr => 6,
        sentient::RegType::Lar => 7,
        sentient::RegType::Lbr => 8,
        sentient::RegType::Ear => 9,
        sentient::RegType::Ebr => 10,
        sentient::RegType::Gtr => 11,
        sentient::RegType::Mvr => 12,
        sentient::RegType::Unrelated => 13,
    }
}

/// ONE ENTRY PER REGISTER LOCALE — `llvm::BitVector locale_has_free_regs_` and
/// `std::array<unsigned, kMaxNumLocales> locale_to_num_regs_exceeded_` (`:189-196`) under one type.
///
/// ⛔⛔ TRAP, AND IT IS THE REFERENCE'S OWN: `kMaxNumLocales` IS THE LAST LOCALE'S VALUE, NOT THE
/// COUNT. `getMaxEnumValForSentientRegType()` is 13 (`SentientTypes.td:254-267` numbers
/// `unknown`..`unrelated` 0..13), so BOTH tables are one entry short and `localeHasFreeRegs(unrelated)`
/// (e370) reads and writes slot 13 of 13 in each — a `BitVector::test(13)` past the end and an
/// out-of-range `std::array` write. `findAndProcessCandidates` sizes its own bucket array
/// `getMaxEnumValForSentientRegType() + 1` (`:228`), which is what these two wanted. Fourteen here,
/// indexed by [`locale_slot`], so neither the short size nor the out-of-range index is expressible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerLocale<T>([T; LOCALES]);

impl<T: Copy> PerLocale<T> {
    /// Every locale holding `value` — `BitVector(n)`'s all-clear and the ctor's `= 0` loop, which are
    /// the same statement about a different element type.
    #[must_use]
    pub const fn filled(value: T) -> PerLocale<T> {
        PerLocale([value; LOCALES])
    }

    /// What `locale` holds.
    #[must_use]
    pub fn get(&self, locale: sentient::RegType) -> T {
        self.0[locale_slot(locale)]
    }

    /// Write `value` for `locale`.
    pub fn set(&mut self, locale: sentient::RegType, value: T) {
        self.0[locale_slot(locale)] = value;
    }
}

/// ONE OP'S LIVENESS INDEX — where it sits in its unit's preorder walk (`op_to_idx_`'s value).
///
/// ⛔ A SNAPSHOT, NOT A LIVE POSITION. The reference computes the numbering once per unit and never
/// updates it because *"Op reordering will render indices stale"* (`:92-96`); what stays valid is the
/// liverange ENDPOINT of a scalar op, which is the only thing it may be read for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LiverangeIndex(usize);

impl LiverangeIndex {
    /// `unsigned cur_idx = 0` (`:100`) and `unsigned endpt_idx = 0` (`:112`) — the first walk position,
    /// and the endpoint of an op with no indexed user.
    pub const FIRST: LiverangeIndex = LiverangeIndex(0);

    /// The index at `index`.
    #[must_use]
    pub const fn at(index: usize) -> LiverangeIndex {
        LiverangeIndex(index)
    }

    /// The index itself.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }

    /// `++cur_idx`.
    #[must_use]
    const fn next(self) -> LiverangeIndex {
        LiverangeIndex(self.0 + 1)
    }
}

/// A SCALAR OP'S ONE RESULT — `DT_CHECK_MSG(op->getNumResults() == 1, "Scalar ops expected to have one
/// result")` (`:110-111`, `:469-470`) AS A TYPE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarResult(Val);

impl ScalarResult {
    /// The one value `op` binds — `None` when it binds none or several, which is the `DT_CHECK`.
    #[must_use]
    pub fn of(op: &Op) -> Option<ScalarResult> {
        match dialects::results(op).as_slice() {
            [result] => Some(ScalarResult(*result)),
            _ => None,
        }
    }

    /// `op->getResult(0)`.
    #[must_use]
    pub const fn val(self) -> Val {
        self.0
    }
}

/// `ScalarOpReorderingPass`'s state (`:147-186`).
///
/// ⚠️ `dcc_ext_ctx_` AND `opts_` ARE NOT CARRIED YET: `dccExtContext()` is read by e370 alone (for
/// `getMaxRegNum`) and `opts_` by e579 (the include-list gate), so both enter with those units. The two
/// use caches — `val_to_first_use_in_block_cache_` and `val_to_last_use_in_block_cache_` — enter with
/// e368 and e369, which are what fill and read them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalarOpReordering {
    /// `op_to_idx_` — ⛔ KEYED BY POSITION, WHICH IS NOT WHAT `Operation *` IS: an [`OpId`] moves when
    /// something is moved or inserted ahead of it in its block, and e531 moves ops. See
    /// [`LiverangeIndex`] for why the VALUE is allowed to go stale, and
    /// [`super::address_register_precision_assignment::PrecisionAssignments`] for the re-keying a
    /// positional identity costs.
    op_to_idx: BTreeMap<OpId, LiverangeIndex>,
    /// `locale_has_free_regs_` — set once a locale has been measured to have room. ⛔ FALSE NEGATIVES
    /// ONLY, by the reference's own note (`:432-434`).
    locale_has_free_regs: PerLocale<bool>,
    /// `locale_to_num_regs_exceeded_` — the overestimate of how far past its register file a locale is.
    locale_to_num_regs_exceeded: PerLocale<u32>,
}

impl ScalarOpReordering {
    /// Replaces: e174_ScalarOpReorderingPass
    ///
    /// A pass with nothing indexed, no locale known to have free registers, and no locale over budget.
    ///
    /// ⛔ TRAP: THE REFERENCE'S TWO LOCALE TABLES ARE ONE ENTRY SHORT — this constructor is where the
    /// size is stated, so see [`PerLocale`]. Its `for (i < kMaxNumLocales) … = 0` loop is one `filled`.
    #[must_use]
    pub fn new() -> ScalarOpReordering {
        ScalarOpReordering {
            op_to_idx: BTreeMap::new(),
            locale_has_free_regs: PerLocale::filled(false),
            locale_to_num_regs_exceeded: PerLocale::filled(0),
        }
    }
}

impl Default for ScalarOpReordering {
    fn default() -> ScalarOpReordering {
        ScalarOpReordering::new()
    }
}

impl ScalarOpReordering {
    /// Replaces: e175_runOn
    ///
    /// Runs the pass over every `dataflow.program_unit` of one module.
    ///
    /// ⛔ NAMED FOR ITS ARGUMENT: `runOn(ModuleOp)` and `runOn(dataflow::ProgramUnitOp)` (e579) are one
    /// C++ overload set and cannot both be `run_on` here.
    /// ⭐ `WalkResult::skip()` (`:84`) IS WHY A UNIT NESTED IN A UNIT IS NEVER VISITED; a program's
    /// units are a flat list here, so the skip is that list.
    /// ⭐ `new DominanceInfo(unit_op)` / `delete dom_info_` IS DROPPABLE MECHANISM — this campaign has
    /// measured what the pass asks a dominance tree and it is one block position; see
    /// [`super::rematerialization_pass::InBlock`]. e368 and e369 are what do the asking.
    pub(crate) fn run_on_program<A: Arch, M: Model, W: Workload>(
        &mut self,
        program: &mut Program<A, M, W>,
    ) {
        for unit in program.units.iter_mut() {
            self.run_on_unit(unit);
        }
    }

    /// `runOn(dataflow::ProgramUnitOp)` — entry 579, level 4, not yet ported.
    fn run_on_unit<A: Arch>(&mut self, unit: &mut ProgramUnit<A>) -> ! {
        let _ = unit;
        todo!(
            "e579_runOn(dataflow::ProgramUnitOp) — the GetUnitOp read and the include-list gate \
             (ScalarOpReordering.cpp:191), which runs e176_computeOpIndexing and then \
             e531_findAndProcessCandidates over the unit"
        )
    }

    /// Replaces: e176_computeOpIndexing
    ///
    /// Numbers every op of one program unit in preorder, dropping the previous unit's numbering.
    ///
    /// ⭐ EVERY OP, NOT JUST THE SCALAR ONES — `sentient.yield`s included, which is the reference's own
    /// note (`:88-91`) and why this is simpler than constructing a `Liveness`.
    /// ⛔ THE UNIT OP ITSELF TAKES INDEX 0 IN THE REFERENCE (`Operation::walk` visits the op it is
    /// called on) and has no island `Op` to take one here, so every index below is one lower. Nothing
    /// can observe that: the map is only ever compared against itself.
    pub fn compute_op_indexing(&mut self, unit: &[Op]) {
        self.op_to_idx.clear();
        let op_to_idx = &mut self.op_to_idx;
        let mut cur_idx = LiverangeIndex::FIRST;
        walk_pre_order(unit, 0, &mut Vec::new(), &mut |at, _op| {
            op_to_idx.insert(at, cur_idx);
            cur_idx = cur_idx.next();
        });
    }

    /// Replaces: e177_getLiverangeEndPt
    ///
    /// The furthest liveness index at which `result` is still read — a scalar op's liverange endpoint.
    ///
    /// ⭐ THE `DT_CHECK` IS THE ARGUMENT: [`ScalarResult`] is `getNumResults() == 1`.
    /// ⛔ TRAP: `op_to_idx_[user]` IS `DenseMap::operator[]`, WHICH INSERTS 0 rather than refusing, so
    /// a user the indexing never saw contributes nothing to the maximum — which is what
    /// [`LiverangeIndex::FIRST`] answers for it here.
    /// ⛔ AN OP WITH NO USERS ANSWERS [`LiverangeIndex::FIRST`] TOO, not "no endpoint": the reference's
    /// `endpt_idx = 0` survives its empty loop. e470 is what rules a use-free op out, before this.
    #[must_use]
    pub fn liverange_end_pt(&self, result: ScalarResult, unit: &[Op]) -> LiverangeIndex {
        let mut endpt = LiverangeIndex::FIRST;
        walk_pre_order(unit, 0, &mut Vec::new(), &mut |at, op| {
            if dialects::operands(op).contains(&result.val()) {
                let user_idx = self
                    .op_to_idx
                    .get(&at)
                    .copied()
                    .unwrap_or(LiverangeIndex::FIRST);
                endpt = endpt.max(user_idx);
            }
        });
        endpt
    }
}

/// Replaces: e178_findAncestorInBlock
///
/// The ancestor of `op` that sits directly in `block`, or `None` when `op` lies outside it.
///
/// ⭐⭐ A BLOCK IS A PATH PREFIX, so the whole `getParentOp()` climb is one truncation — see
/// [`OpId::block`]. `op` is its own answer when it already sits in `block`, which is the `while` loop
/// not running at all.
/// ⛔ TRAP: A MULTI-REGION OP'S REGIONS ARE ONE PREFIX HERE (`OpId`'s own note), so a `sentient.if`'s
/// `then` and `else` bodies are one "block" and a cross-arm use answers a position where the reference
/// answers `nullptr`. Both callers (e368, e369) feed the answer to a same-block dominance comparison,
/// which is a question MLIR has no answer to across two arms either.
#[must_use]
pub fn ancestor_in_block(op: &OpId, block: &[u32]) -> Option<OpId> {
    let path = op.path();
    (path.len() > block.len() && path[..block.len()] == *block)
        .then(|| OpId::at(&path[..=block.len()]))
}

/// `unit_op->walk<WalkOrder::PreOrder>` — every op of one unit with its position, outermost first.
///
/// ⛔ REGIONS CONCATENATED, which is the numbering [`OpId`] documents and
/// [`super::address_register_precision_assignment`]'s `op_at` reads back.
/// ⛔ A SHARED DIALECT'S REGION HOLDS RUNG-BELOW OPS AND SO HAS NO POSITION AT THIS RUNG — see
/// [`dialects::regions_ref`]. The reference's walk reaches those ops; nothing here can name them.
fn walk_pre_order<'a>(
    scope: &'a [Op],
    base: u32,
    prefix: &mut Vec<u32>,
    visit: &mut impl FnMut(OpId, &'a Op),
) {
    for (ordinal, op) in scope.iter().enumerate() {
        prefix.push(base + ordinal as u32);
        visit(OpId::at(prefix), op);
        let mut offset = 0u32;
        for region in dialects::regions_ref(op) {
            walk_pre_order(region, offset, prefix, visit);
            offset += region.len() as u32;
        }
        prefix.pop();
    }
}

// crustify:todo: e367_runOnOperation
//   authority : dcc/src/Transform/Sentient/ScalarOpReordering.cpp:74  (5 body lines, level 1)
//   original  : void runOnOperation()
//   calls     : e175_runOn

// crustify:todo: e368_getFirstUseWithinBlock
//   authority : dcc/src/Transform/Sentient/ScalarOpReordering.cpp:383  (23 body lines, level 1)
//   original  : Operation *ScalarOpReorderingPass::getFirstUseWithinBlock(Value v)
//   calls     : e178_findAncestorInBlock

// crustify:todo: e369_getLastUseWithinBlock
//   authority : dcc/src/Transform/Sentient/ScalarOpReordering.cpp:407  (23 body lines, level 1)
//   original  : Operation *ScalarOpReorderingPass::getLastUseWithinBlock(Value v, Block *bb)
//   calls     : e178_findAncestorInBlock

// crustify:todo: e370_localeHasFreeRegs
//   authority : dcc/src/Transform/Sentient/ScalarOpReordering.cpp:431  (30 body lines, level 1)
//   original  : bool ScalarOpReorderingPass::localeHasFreeRegs(SentientRegType locale)
//   calls     : e056_set, e063_getMaxRegNum

// crustify:todo: e470_isCandidateForReordering
//   authority : dcc/src/Transform/Sentient/ScalarOpReordering.cpp:462  (67 body lines, level 2)
//   original  : bool ScalarOpReorderingPass::isCandidateForReordering(Operation &op)
//   calls     : e368_getFirstUseWithinBlock, e369_getLastUseWithinBlock

// crustify:todo: e531_findAndProcessCandidates
//   authority : dcc/src/Transform/Sentient/ScalarOpReordering.cpp:207  (165 body lines, level 3)
//   original  : void ScalarOpReorderingPass::findAndProcessCandidates( dataflow::ProgramUnitOp unit_op)
//   calls     : e177_getLiverangeEndPt, e368_getFirstUseWithinBlock, e370_localeHasFreeRegs, e422_insert, e470_isCandidateForReordering

// crustify:todo: e579_runOn
//   authority : dcc/src/Transform/Sentient/ScalarOpReordering.cpp:191  (15 body lines, level 4)
//   original  : void ScalarOpReorderingPass::runOn(dataflow::ProgramUnitOp unit_op)
//   calls     : e176_computeOpIndexing, e531_findAndProcessCandidates

#[cfg(test)]
mod unit_tests {
    use super::{LiverangeIndex, PerLocale, ScalarOpReordering, ScalarResult, ancestor_in_block};
    use crate::arch::Dd2;
    use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::dialects::{Op, Val, sentient};
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

    /// `%r = sentient.scalar_constant {value = 1 : si64} : index`.
    fn constant(result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value: 1,
            result,
            reg_locale: sentient::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// `%out = sentient.scalar_add %lhs, %rhs : index`.
    fn add(lhs: Val, rhs: Val, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarAdd {
            lhs,
            rhs,
            result,
            reg: None,
            element_size: None,
            ty: ScalarTy::Index,
        })
    }

    /// A `sentient.for` carrying one value, running `body`.
    fn for_op(iv: Val, bound: Val, body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv,
            bound,
            carried: vec![sentient::Carried {
                init: Val(90),
                arg: Val(91),
                result: Val(92),
                reg: sentient::Reg {
                    locale: sentient::RegType::Unknown,
                    index: None,
                },
                program_header: false,
                element_size: None,
            }],
            dbg_name: None,
            body,
        })
    }

    /// A program of one unit running `body`.
    fn program_of(body: Vec<Op>) -> Program<Dd2, AnyModel, AnyRung> {
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
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

    /// e174 — nothing indexed, no locale with free registers, no locale over budget, and the
    /// FOURTEENTH locale is addressable, which is the entry the reference's two tables lack.
    #[test]
    fn the_new_pass_has_fourteen_zeroed_locale_slots() {
        let pass = ScalarOpReordering::new();
        assert_eq!(pass, ScalarOpReordering::default());
        assert!(pass.op_to_idx.is_empty());
        for locale in [
            sentient::RegType::Unknown,
            sentient::RegType::Lrf,
            sentient::RegType::Mvr,
            sentient::RegType::Unrelated,
        ] {
            assert!(!pass.locale_has_free_regs.get(locale));
            assert_eq!(pass.locale_to_num_regs_exceeded.get(locale), 0);
        }
        let mut table = PerLocale::filled(0u32);
        table.set(sentient::RegType::Unrelated, 7);
        assert_eq!(table.get(sentient::RegType::Unrelated), 7);
        assert_eq!(table.get(sentient::RegType::Mvr), 0);
    }

    /// e175 — every unit of the module is visited, and what a unit is handed to is e579.
    #[test]
    #[should_panic(expected = "e579_runOn")]
    fn run_on_program_visits_each_unit_through_the_unported_unit_pass() {
        let mut program = program_of(vec![constant(Val(0))]);
        ScalarOpReordering::new().run_on_program(&mut program);
    }

    /// e176 — preorder, every op including the `sentient.yield`-less loop's body, regions inline.
    #[test]
    fn op_indexing_numbers_every_op_in_preorder() {
        let unit = vec![
            constant(Val(0)),
            for_op(Val(1), Val(0), vec![add(Val(0), Val(91), Val(2))]),
            add(Val(0), Val(0), Val(3)),
        ];
        let mut pass = ScalarOpReordering::new();
        pass.compute_op_indexing(&unit);
        let mut got: Vec<(Vec<u32>, usize)> = pass
            .op_to_idx
            .iter()
            .map(|(at, idx)| (at.path().to_vec(), idx.index()))
            .collect();
        got.sort();
        // The loop body's op follows the loop itself, not the block's next op.
        assert_eq!(
            got,
            vec![(vec![0], 0), (vec![1], 1), (vec![1, 0], 2), (vec![2], 3),]
        );
        // A second unit drops the first's numbering (`op_to_idx_.clear()`).
        pass.compute_op_indexing(&[constant(Val(0))]);
        assert_eq!(pass.op_to_idx.len(), 1);
    }

    /// e177 — the maximum over the users, a NESTED user included, and no user answers `FIRST`.
    #[test]
    fn liverange_end_pt_is_the_last_users_index() {
        let unit = vec![
            constant(Val(0)),
            add(Val(0), Val(0), Val(3)),
            for_op(Val(1), Val(0), vec![add(Val(0), Val(91), Val(2))]),
        ];
        let mut pass = ScalarOpReordering::new();
        pass.compute_op_indexing(&unit);
        let c = ScalarResult::of(&unit[0]).expect("a constant binds one value");
        // The nested `scalar_add` at index 3 reads `%0` and is later than the loop that encloses it.
        assert_eq!(pass.liverange_end_pt(c, &unit), LiverangeIndex::at(3));
        // `endpt_idx = 0` survives an empty user loop.
        let unread = ScalarResult::of(&unit[1]).expect("an add binds one value");
        assert_eq!(pass.liverange_end_pt(unread, &unit), LiverangeIndex::FIRST);
        // The `DT_CHECK_MSG` this type replaces: a `sentient.for` with one carried value binds one
        // result, and its body op is not a result of the unit's block.
        assert_eq!(ScalarResult::of(&unit[2]).map(|r| r.val()), Some(Val(92)));
    }

    /// e178 — an op already in the block is its own ancestor, a nested op answers the enclosing op,
    /// and an op outside the block answers `nullptr`.
    #[test]
    fn ancestor_in_block_truncates_the_path_to_the_blocks_prefix() {
        let block: &[u32] = &[4];
        assert_eq!(
            ancestor_in_block(&OpId::at(&[4, 2]), block),
            Some(OpId::at(&[4, 2]))
        );
        assert_eq!(
            ancestor_in_block(&OpId::at(&[4, 2, 0, 1]), block),
            Some(OpId::at(&[4, 2]))
        );
        assert_eq!(ancestor_in_block(&OpId::at(&[5, 2]), block), None);
        // The block's own op is not IN the block.
        assert_eq!(ancestor_in_block(&OpId::at(&[4]), block), None);
        // The unit's entry block is the empty prefix, so a top-level op is its own ancestor there.
        assert_eq!(
            ancestor_in_block(&OpId::at(&[7]), &[]),
            Some(OpId::at(&[7]))
        );
    }
}

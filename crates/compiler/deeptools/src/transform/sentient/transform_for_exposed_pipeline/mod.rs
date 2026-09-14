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

//! `TransformForExposedPipeline.cpp` — 9 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e234_printDependencies` | 234 | 0 | 9 | `dcc/src/Transform/Sentient/TransformForExposedPipeline.cpp:75` |
//! | `e235_cleanUp` | 235 | 0 | 12 | `dcc/src/Transform/Sentient/TransformForExposedPipeline.cpp:85` |
//! | `e236_macOpsFlowDependence` | 236 | 0 | 28 | `dcc/src/Transform/Sentient/TransformForExposedPipeline.cpp:100` |
//! | `e237_insertNOPOperations` | 237 | 0 | 36 | `dcc/src/Transform/Sentient/TransformForExposedPipeline.cpp:304` |
//! | `e391_getIndexOfEntry` | 391 | 1 | 6 | `dcc/src/Transform/Sentient/TransformForExposedPipeline.cpp:131` |
//! | `e482_computeDependenciesSameBlock` | 482 | 2 | 82 | `dcc/src/Transform/Sentient/TransformForExposedPipeline.cpp:139` |
//! | `e483_computeDependenciesAcrossBlocks` | 483 | 2 | 59 | `dcc/src/Transform/Sentient/TransformForExposedPipeline.cpp:224` |
//! | `e543_computeDependencies` | 543 | 3 | 13 | `dcc/src/Transform/Sentient/TransformForExposedPipeline.cpp:287` |
//! | `e585_runOnOperation` | 585 | 4 | 35 | `dcc/src/Transform/Sentient/TransformForExposedPipeline.cpp:341` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET — [`run_on_operation`] (e585) is the entry, and
// there is no ported D29–D75 pass driver to call it, so nothing but this file's own tests reaches
// anything here. CI runs clippy with `-D warnings`.
// ⭐ REMOVE THIS WITH THAT DRIVER, not with an anchor: e585 is filled and the pass is still unwired.
#![allow(dead_code)]

use crate::arch::{Arch, IsaGen};
use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;
use crate::islands::dataflow_ir::dialects::dataflow::Precision;
use crate::islands::sentient::dialects::{self as dialects, Op, sentient};
use crate::islands::sentient::print;
use crate::islands::sentient::{Program, ProgramUnit};
use crate::model::Model;
use crate::transform::sentient::analyses::{
    Cycles, Dependency, GapDirection, TimeStampColumnVal, TimeStamps,
};
use crate::units::DfirUnit;
use crate::workload::Workload;
use std::collections::BTreeMap;

/// WHAT THIS PASS READS OFF A `sentient.vector_mac` — `dyn_cast<sentient::MacOp>`'s result, and the
/// reason a `dyn_cast` that fails is a `None` rather than a null pointer to dereference later.
struct Mac<'a> {
    /// `getOpA()` with its `unrollIncrOpA`.
    op_a: &'a sentient::Operand,
    /// `getOpB()` with its `unrollIncrOpB`.
    op_b: &'a sentient::Operand,
    /// `getOpC()` with its `unrollIncrOpC`.
    op_c: &'a sentient::Operand,
    /// `getResultForwarding()` with its `getUnrollIncrResult()`.
    result: &'a sentient::ResultPorts,
    /// `getUnrollFactorVal()` (`SentientOps.cpp:1678`).
    unroll_factor: sentient::UnrollFactor,
    /// What `getNewDbgNameFromOp` reads.
    dbg_name: Option<&'a str>,
}

/// `dyn_cast<sentient::MacOp>(op)`.
fn mac_of(op: &Op) -> Option<Mac<'_>> {
    match op {
        Op::Sentient(sentient::Op::VectorMac {
            op_a,
            op_b,
            op_c,
            result,
            unroll_factor,
            dbg_name,
            ..
        }) => Some(Mac {
            op_a,
            op_b,
            op_c,
            result,
            unroll_factor: *unroll_factor,
            dbg_name: dbg_name.as_deref(),
        }),
        _ => None,
    }
}

/// `MacOp::getOutputRegister()` (`SentientOps.cpp:1643-1654`) — the first forwarded port that names a
/// register file, kept as a SPELLING because every use of it is a `contains`.
fn output_register(result: &sentient::ResultPorts) -> Option<String> {
    result
        .forwarding
        .iter()
        .map(|port| port.spelling())
        .find(|dest| dest.contains("irf") || dest.contains("lrf") || dest.contains("xrf"))
}

/// `atoi(&spelling.str().back())` (`:118`, `Dialect/Sentient/Utils.cpp:467-469`).
///
/// ⛔⛔ IT READS **ONE CHARACTER**: `lrf12` starts at 2, not 12. That bounds every synthesized name to
/// `lrf0`..`lrf17` (last digit 9 plus the largest unroll factor), and a spelling whose last character
/// is not a digit reads 0 exactly as `atoi` does — reachable only for a name already holding `lrf`.
fn last_digit(spelling: &str) -> u32 {
    spelling
        .chars()
        .next_back()
        .and_then(|last| last.to_digit(10))
        .unwrap_or(0)
}

/// `dcc::sentient::utils::doesMacOpUseRegister` (`Dialect/Sentient/Utils.cpp:460-499`) — does `mac`
/// read `reg` in any of its unrolled copies.
///
/// ⛔ TRAP: `contains` IS A SUBSTRING TEST AND IT IS PRESERVED — `lrf1` also matches `lrf10`..`lrf19`,
/// which over-reports a dependence (an extra NOP, never a missed hazard).
/// ⛔ THE `lrf1`→`lrf0` / `lrf3`→`lrf2` INT-PRECISION ALIASING IS DEAD ON BOTH MODELLED GENERATIONS:
/// it needs `arch < RCUDD1A_ISA` and [`IsaGen`] starts AT `Rcudd1a`, so the guard const-folds to false
/// rather than being dropped. ⚠️ Its `int2` arm is inexpressible — `dataflow::Precision` has no `int2`.
fn does_mac_op_use_register<A: Arch>(mac: &Mac<'_>, reg: &str, precision: Precision) -> bool {
    for copy in 0..mac.unroll_factor.count() {
        let unrolled = |operand: &sentient::Operand| -> String {
            let spelling = operand.port.spelling();
            if spelling.contains("lrf") && operand.unroll_incr {
                format!("lrf{}", last_digit(&spelling) + copy)
            } else {
                spelling
            }
        };
        if unrolled(mac.op_a).contains(reg)
            || unrolled(mac.op_b).contains(reg)
            || unrolled(mac.op_c).contains(reg)
        {
            return true;
        }
        if A::GEN < IsaGen::Rcudd1a && matches!(precision, Precision::Int8 | Precision::Int4) {
            if reg == "lrf1" {
                return does_mac_op_use_register::<A>(mac, "lrf0", precision);
            }
            if reg == "lrf3" {
                return does_mac_op_use_register::<A>(mac, "lrf2", precision);
            }
        }
    }
    false
}

/// Replaces: e236_macOpsFlowDependence
///
/// True when `b` reads the register `a`'s result is forwarded to — run over `a`'s whole unrolled run
/// of `lrf`s when its result increments per copy.
///
/// ⛔ A NON-MAC ON EITHER SIDE IS **TRUE** (`:103-105`), and it stays true: both callers walk only
/// MACs (`Analyses/TimeStamps.cpp:39-44`), and answering `false` there would drop real hazards.
/// ⛔ NO OUTPUT REGISTER AT ALL IS **FALSE** — a MAC whose result goes to a link cannot be written by.
/// ⭐ THE REFERENCE'S `precision = "fp16"` DEFAULT ARGUMENT IS THE CALLER'S JOB HERE.
pub(crate) fn mac_ops_flow_dependence<A: Arch>(a: &Op, b: &Op, precision: Precision) -> bool {
    let (Some(src), Some(dst)) = (mac_of(a), mac_of(b)) else {
        return true;
    };
    let Some(src_register) = output_register(src.result) else {
        return false;
    };
    if !src.result.unroll_incr || !src_register.contains("lrf") {
        return does_mac_op_use_register::<A>(&dst, &src_register, precision);
    }
    let start = last_digit(&src_register);
    (0..src.unroll_factor.count())
        .any(|copy| does_mac_op_use_register::<A>(&dst, &format!("lrf{}", start + copy), precision))
}

/// `AncestorWithCounts` (`src/Utils/Utils.hpp:72-78`) — what
/// [`outermost_ancestor_in_common_scope`] answers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct AncestorWithCounts {
    /// `ancestor`, as a position. ⛔ EMPTY IS THE PROGRAM UNIT and is the only reading `nullptr` has
    /// here: both ends are positions in one unit, so their chains always meet.
    path: Vec<u32>,
    /// `num_for_ops_from_src`.
    for_ops_from_src: i32,
    /// `num_if_ops_from_src`.
    if_ops_from_src: i32,
    /// `num_for_ops_from_dst`.
    for_ops_from_dst: i32,
    /// `num_if_ops_from_dst`.
    if_ops_from_dst: i32,
}

/// `dcc::utils::getOutermostAncestorInCommonScope` (`src/Utils/Utils.cpp:80-136`) — the outermost op
/// on `src`'s chain still strictly inside the innermost scope `src` and `dst` share, plus how many
/// `sentient.for`s and `sentient.if`s each chain crossed reaching it.
///
/// ⭐ THE PARENT CHAIN *IS* THE PATH: `getParentOp()` drops one ordinal, so the innermost common
/// ancestor is the longest common prefix and the `SmallPtrSet` of src's ancestors is not needed.
/// ⛔ ONLY `sentient.for` AND `sentient.if` COUNT (`:112-121`) — an `affine.for`, an `scf.if` or a
/// uniform region on either chain is crossed silently.
/// ⛔ `src` AT OR ABOVE `dst` IS ITS OWN ANSWER WITH NO CROSSINGS COUNTED (`:85-88`, `:103-106`).
fn outermost_ancestor_in_common_scope(src: &OpId, dst: &OpId, body: &[Op]) -> AncestorWithCounts {
    let (src_path, dst_path) = (src.path(), dst.path());
    let common = src_path
        .iter()
        .zip(dst_path)
        .take_while(|(from_src, from_dst)| from_src == from_dst)
        .count();
    if common == src_path.len() {
        return AncestorWithCounts {
            path: src_path.to_vec(),
            ..AncestorWithCounts::default()
        };
    }
    let mut result = AncestorWithCounts {
        path: src_path[..=common].to_vec(),
        ..AncestorWithCounts::default()
    };
    for depth in (common + 1)..=src_path.len() {
        match op_at(&OpId::at(&src_path[..depth]), body) {
            Some(Op::Sentient(sentient::Op::For { .. })) => result.for_ops_from_src += 1,
            Some(Op::Sentient(sentient::Op::If { .. })) => result.if_ops_from_src += 1,
            _ => {}
        }
    }
    for depth in (common + 1)..=dst_path.len() {
        match op_at(&OpId::at(&dst_path[..depth]), body) {
            Some(Op::Sentient(sentient::Op::For { .. })) => result.for_ops_from_dst += 1,
            Some(Op::Sentient(sentient::Op::If { .. })) => result.if_ops_from_dst += 1,
            _ => {}
        }
    }
    result
}

/// The op at a position — this module's copy of
/// [`address_register_precision_assignment`](super::address_register_precision_assignment)'s helper,
/// regions concatenated, which is the numbering [`OpId`] documents.
fn op_at<'a>(id: &OpId, scope: &'a [Op]) -> Option<&'a Op> {
    let (&ordinal, rest) = id.path().split_first()?;
    let op = scope.get(ordinal as usize)?;
    let Some(&next) = rest.first() else {
        return Some(op);
    };
    // ⛔ ONLY A `sentient.*` OP HOLDS A REGION OF THIS RUNG'S OPS by the time this pass runs — *"all
    // loops have been lowered to sentient.for"* (`VectorChainToSentientPT/Helper.cpp:141`).
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

/// `OpBuilder(op)`'s BLOCK and `op`'s ordinal in it — resolved once so a run of NOPs goes in without
/// re-walking, which is what `setInsertionPointAfter` plus repeated `create` amounts to.
fn block_of<'a>(path: &[u32], scope: &'a mut Vec<Op>) -> Option<(&'a mut Vec<Op>, usize)> {
    let (&ordinal, rest) = path.split_first()?;
    if rest.is_empty() {
        return Some((scope, ordinal as usize));
    }
    let Some(Op::Sentient(inner)) = scope.get_mut(ordinal as usize) else {
        return None;
    };
    let next = rest[0] as usize;
    let mut base = 0usize;
    for region in sentient::regions_mut(inner) {
        if next < base + region.len() {
            let mut sub: Vec<u32> = rest.to_vec();
            sub[0] = (next - base) as u32;
            return block_of(&sub, region);
        }
        base += region.len();
    }
    None
}

/// THE PASS'S TWO DEPENDENCE LISTS (`:53-57`) — what `computeDependencies` fills and what
/// `insertNOPOperations` spends.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Dependencies {
    /// `dependencies_debug_`, keyed by position rather than by `Operation *`.
    ///
    /// ⛔⛔ NOTHING IN THIS REVISION EVER WRITES IT. `computeDependenciesSameBlock` builds a **local**
    /// `dst_operations_debug_` and drops it on the floor (`:149`, `:185`) — the assignment into this
    /// map is missing — so [`Dependencies::print_dependencies`] has nothing to print. The field is
    /// still the pass's, and typed as the pass declares it, so `e482` can fill it.
    debug: BTreeMap<OpId, Vec<(OpId, Vec<TimeStampColumnVal>)>>,
    /// `dependencies_with_min_gap_` — one entry per source op, in discovery order.
    with_min_gap: Vec<Dependency>,
}

impl Dependencies {
    /// Replaces: e234_printDependencies
    ///
    /// Dumps each recorded source op, then every op recorded as depending on it.
    ///
    /// ⛔ ALWAYS EMPTY AT THIS REVISION, AND IT HAS NO CALLER — see [`Dependencies::debug`].
    /// ⭐ `record.first->getName().dump()` NEEDS NO OP-NAME ACCESSOR: the mnemonic is the head of the
    /// op's own text, which the `dump()` on the next line prints anyway.
    /// ⭐ DETERMINISTIC WHERE THE REFERENCE IS NOT — `std::map<Operation *, …>` iterates by POINTER.
    pub(crate) fn print_dependencies(&self, body: &[Op]) -> String {
        let mut out = String::new();
        for (src, dependents) in &self.debug {
            if let Some(op) = op_at(src, body) {
                print::emit(&mut out, op, 0);
            }
            for (dst, _time_stamps) in dependents {
                if let Some(op) = op_at(dst, body) {
                    print::emit(&mut out, op, 0);
                }
            }
        }
        out
    }

    /// Replaces: e235_cleanUp
    ///
    /// Empties both lists between program units.
    ///
    /// ⭐ THE NESTED `clear()`s ARE OWNERSHIP HERE: `op.second.clear()` and `record.second.clear()`
    /// (`:87-91`) free storage that `dependencies_debug_.clear()` frees on the line below anyway.
    pub(crate) fn clean_up(&mut self) {
        self.debug.clear();
        self.with_min_gap.clear();
    }

    /// Replaces: e237_insertNOPOperations
    ///
    /// Inserts, immediately after the outermost ancestor each hazard shares with its destination, the
    /// NOPs that open the exposed-pipeline gap the source MAC needs.
    ///
    /// ⛔ TRAP: THE COUNT GOES NEGATIVE AND IS STILL BANKED (`:333-337`), so a later hazard sharing
    /// that ancestor inserts MORE than its own budget — preserved.
    /// ⛔ TRAP: THE BUILDER IS REBUILT PER HAZARD, so of two hazards under one ancestor the LATER
    /// one's NOPs land FIRST.
    /// ⛔ AN UNNAMED MAC GIVES UNNAMED NOPs (`src/Utils/Utils.cpp:495-496`); a non-MAC `src` is a
    /// null dereference at `:308` and is skipped here.
    pub(crate) fn insert_nop_operations(&self, body: &mut Vec<Op>, cycles: Cycles) {
        // ⭐ EVERY COUNT IS TAKEN BEFORE ANYTHING MOVES, and the insertions then run deepest-and-last
        // first: inserting at a path shifts only paths lexicographically after it, where an
        // `Operation *` would not have moved at all.
        let mut banked: BTreeMap<Vec<u32>, i32> = BTreeMap::new();
        let mut groups: Vec<(Vec<u32>, Vec<Op>)> = Vec::new();
        for dependency in &self.with_min_gap {
            let Some(mac) = op_at(&dependency.src, body).and_then(mac_of) else {
                continue;
            };
            let mut insertions = cycles.0 - mac.unroll_factor.count() as i32 + 1;
            if dependency.gap.0 > 1 {
                insertions -= dependency.gap.0 - 2;
            }
            let ancestor =
                outermost_ancestor_in_common_scope(&dependency.src, &dependency.dst, body);
            insertions -= ancestor.for_ops_from_dst + ancestor.if_ops_from_dst;
            insertions -= banked.get(&ancestor.path).copied().unwrap_or(0);
            let nops: Vec<Op> = (0..insertions.max(0))
                .map(|which| {
                    Op::Sentient(sentient::Op::Nop {
                        dbg_name: mac
                            .dbg_name
                            .map(|name| format!("TFEP({name}, NOP #{which})")),
                    })
                })
                .collect();
            *banked.entry(ancestor.path.clone()).or_insert(0) += insertions;
            match groups.iter_mut().find(|(path, _)| *path == ancestor.path) {
                Some((_, standing)) => {
                    standing.splice(0..0, nops);
                }
                None => groups.push((ancestor.path, nops)),
            }
        }
        groups.sort_by(|(one, _), (other, _)| other.cmp(one));
        for (path, nops) in groups {
            let Some((block, ancestor)) = block_of(&path, body) else {
                continue;
            };
            for (offset, nop) in nops.into_iter().enumerate() {
                block.insert(ancestor + 1 + offset, nop);
            }
        }
    }

    /// Replaces: e482_computeDependenciesSameBlock
    ///
    /// Banks one RAW hazard per source MAC for every forward pair within `cycles` whose two ends sit
    /// in the same block, after collapsing the hazards under each enclosing loop or conditional onto
    /// their intersections.
    ///
    /// ⛔ THE INTERSECTION IS PER ENCLOSING REGION OP, NOT PER BLOCK: the key is the INNERMOST
    /// `sentient.for`/`sentient.if` on `src`'s own timestamp (`:170-183`), and `None` — the
    /// reference's `nullptr` — is the one bucket for every hazard under no loop and no conditional.
    /// ⛔ THE COLUMN SCAN STARTS AT `size() - 2`, so a source's OWN innermost region column is
    /// skipped and a timestamp of one column or none contributes nothing.
    /// ⛔ TRAP: `dst_operations_debug_` (`:149`, `:185`) IS BUILT AND DROPPED — the assignment into
    /// [`Dependencies::debug`] is missing in this revision, so it is not written here either.
    /// ⭐ DETERMINISTIC WHERE THE REFERENCE IS NOT: `std::map<Operation *, …>` reduces the buckets in
    /// POINTER order, which decides which source wins the one-entry-per-`src` filter below.
    /// ⚠️ THE `LLVM_DEBUG` INTERVAL DUMPS CHANGE NO IR AND ARE DROPPED.
    pub(crate) fn compute_dependencies_same_block<A: Arch>(
        &mut self,
        ts: &mut impl TimeStamps,
        body: &[Op],
        cycles: Cycles,
        precision: Precision,
    ) {
        // ⭐ THE ORDER IS COPIED, NOT BORROWED: `getCyclesGap` subscripts `time_stamps_` and so takes
        // the analyzer mutably, which a live borrow of its own op order would forbid.
        let order: Vec<OpId> = ts.op_order().to_vec();
        let mut dependencies_in_bb: BTreeMap<Option<OpId>, Vec<Dependency>> = BTreeMap::new();
        for src in &order {
            for dst in &order {
                if src == dst {
                    continue;
                }
                if !ts.is_in_same_block(ts.time_stamp(src), ts.time_stamp(dst)) {
                    continue;
                }
                // ⭐ AN UNRESOLVABLE POSITION CANNOT ARISE: the analyzer timestamped these very ops
                // of this very body, so both lookups answer `Some`.
                let (Some(src_op), Some(dst_op)) = (op_at(src, body), op_at(dst, body)) else {
                    continue;
                };
                if !mac_ops_flow_dependence::<A>(src_op, dst_op, precision) {
                    continue;
                }
                let (gap, direction) = ts.cycles_gap(src, dst);
                if gap.0 < 0 || gap.0 > cycles.0 {
                    continue;
                }
                if direction == GapDirection::Forward {
                    let columns = ts.time_stamp(src);
                    let parent = columns[..columns.len().saturating_sub(1)]
                        .iter()
                        .rev()
                        .find_map(TimeStampColumnVal::region_op)
                        .cloned();
                    dependencies_in_bb
                        .entry(parent)
                        .or_default()
                        .push(Dependency {
                            src: src.clone(),
                            dst: dst.clone(),
                            gap,
                        });
                }
            }
        }
        for (_parent, mut intervals) in dependencies_in_bb {
            ts.reduce_intervals(&mut intervals);
            for interval in intervals {
                // `getIndexOfEntry(dependencies_with_min_gap_, op_interval.src) == -1` — the FIRST
                // interval reduction to reach a source keeps it and every later one is dropped.
                if index_of_entry(&self.with_min_gap, &interval.src).is_none() {
                    self.with_min_gap.push(interval);
                }
            }
        }
    }

    /// Replaces: e483_computeDependenciesAcrossBlocks
    ///
    /// Banks the hazards e482 could not: the loop-carried ones, the self-dependent MACs, and the
    /// forward pairs whose two ends sit in DIFFERENT blocks, reduced per destination.
    ///
    /// ⛔ THE GROUPING KEY IS THE DESTINATION HERE, NOT AN ANCESTOR — and `reduceIntervals` runs only
    /// on a destination with more than one source (`:253`), so a lone hazard keeps its own gap.
    /// ⛔ TRAP: A SELF-DEPENDENT MAC IS BANKED WITH GAP **0**, NOT ITS OWN GAP (`:249`), and the
    /// update arm only ever LOWERS an existing entry — so once this pass has created one, no later
    /// pair can raise it, and the only entry the `gap > gap` test can usefully lower is one e482
    /// already made for that source.
    /// ⛔ A `nextIter` PAIR WITH `src != dst` REACHES NEITHER ARM and is dropped after being paid for.
    /// ⭐ DETERMINISTIC WHERE THE REFERENCE IS NOT, as e482 is; ⚠️ its `LLVM_DEBUG` dumps are dropped.
    pub(crate) fn compute_dependencies_across_blocks<A: Arch>(
        &mut self,
        ts: &mut impl TimeStamps,
        body: &[Op],
        cycles: Cycles,
        precision: Precision,
    ) {
        let order: Vec<OpId> = ts.op_order().to_vec();
        let mut deps_by_dst: BTreeMap<OpId, Vec<Dependency>> = BTreeMap::new();
        for src in &order {
            for dst in &order {
                let (Some(src_op), Some(dst_op)) = (op_at(src, body), op_at(dst, body)) else {
                    continue;
                };
                if !mac_ops_flow_dependence::<A>(src_op, dst_op, precision) {
                    continue;
                }
                let (gap, direction) = ts.cycles_gap(src, dst);
                if gap.0 < 0 || gap.0 > cycles.0 {
                    continue;
                }
                if direction == GapDirection::Forward
                    && src != dst
                    && ts.is_in_same_block(ts.time_stamp(src), ts.time_stamp(dst))
                {
                    continue;
                }
                if src == dst {
                    match index_of_entry(&self.with_min_gap, src) {
                        None => self.with_min_gap.push(Dependency {
                            src: src.clone(),
                            dst: dst.clone(),
                            gap: Cycles(0),
                        }),
                        Some(idx) if self.with_min_gap[idx].gap > gap => {
                            self.with_min_gap[idx].gap = gap;
                        }
                        Some(_) => {}
                    }
                } else if direction == GapDirection::Forward {
                    deps_by_dst
                        .entry(dst.clone())
                        .or_default()
                        .push(Dependency {
                            src: src.clone(),
                            dst: dst.clone(),
                            gap,
                        });
                }
            }
        }
        for (_dst, mut deps) in deps_by_dst {
            if deps.len() > 1 {
                ts.reduce_intervals(&mut deps);
            }
            for dep in deps {
                if index_of_entry(&self.with_min_gap, &dep.src).is_none() {
                    self.with_min_gap.push(dep);
                }
            }
        }
    }
}

/// Replaces: e391_getIndexOfEntry
///
/// Where `op` is banked as a SOURCE in `op_and_gap_list`, if it is (`:131-137`).
///
/// ⛔ THE MATCH IS ON `src` ONLY: a source already banked against ANY destination is found, which is
/// what makes the first hazard to reach a source the one that keeps it.
/// ⛔ `-1` IS `None` — every caller tests it against `-1` before indexing, never arithmetically.
fn index_of_entry(op_and_gap_list: &[Dependency], op: &OpId) -> Option<usize> {
    op_and_gap_list.iter().position(|entry| entry.src == *op)
}

impl Dependencies {
    /// Replaces: e543_computeDependencies
    ///
    /// Timestamps the whole unit, then banks every RAW hazard within `cycles` of it — the same-block
    /// ones first (e482), the rest after (e483).
    ///
    /// ⛔ THE ROOT IS THE ANALYZER'S OWN: `ts_analyzer.computeTimeStamps(ts_analyzer.root_, …)` hands
    /// back the `dataflow.program_unit` it was constructed with (`Analyses/TimeStamps.h:100`, `:112`),
    /// so `body` here IS that root and there is no second unit to pass.
    /// ⛔ AND `tmp_time_stamps` IS WRITE-ONLY: nothing reads it after the call (`:290-291`), which is
    /// why it is this function's own local and not a field.
    /// ⚠️ `LLVM_DEBUG(ts_analyzer.printTimeStamps())` IS DROPPED (`:293`): it changes no IR.
    pub(crate) fn compute_dependencies<A: Arch>(
        &mut self,
        ts: &mut impl TimeStamps,
        body: &[Op],
        cycles: Cycles,
        precision: Precision,
    ) {
        let mut tmp_time_stamps = Vec::new();
        ts.compute_time_stamps(body, &mut tmp_time_stamps);
        self.compute_dependencies_same_block::<A>(ts, body, cycles, precision);
        self.compute_dependencies_across_blocks::<A>(ts, body, cycles, precision);
    }
}

/// `-dcc-transform-for-exposed-pipeline-disable`, `cl::init(false)` (`:35-38`) — a `dcc-opt`
/// command-line flag, not a program property, and this crate has no flags.
const DISABLE_THIS_PASS: bool = false;

/// `sentient::getCyclesForExposedPipeline(precision, arch)` — the RAW gap the exposed pipeline needs
/// (`Dialect/Sentient/SentientOps.cpp:40-59`).
///
/// ⛔ NOT AN ANCHORED UNIT: it lives in `Dialect/`, outside this campaign's `Transform/Sentient/`
/// scope, and its `DT_ERROR("Unknown precision for exposed pipeline")` tail (`:57`) is an ABORT — a
/// precision this generation has no answer for returns no cycle count in the reference either.
/// ⭐ `arch <= RCUDD1A_ISA` IS THE SPLIT and [`IsaGen`] starts at `Rcudd1a`, so the two arms are the
/// two generations. ⛔ `"int2"` HAS NO ISLAND SPELLING — [`Precision`] is closed and does not carry
/// it — so its two arms are unreachable here rather than dropped.
fn cycles_for_exposed_pipeline<A: Arch>(precision: Precision) -> Cycles {
    match (A::GEN, precision) {
        (IsaGen::Rcudd1a, Precision::Fp16 | Precision::Fp8 | Precision::Fp80) => Cycles(3),
        (IsaGen::Rcudd1a, Precision::Int4 | Precision::Int8) => Cycles(2),
        (
            IsaGen::Sen1p5,
            Precision::Fp16
            | Precision::Bf16
            | Precision::Fp8
            | Precision::Fp80
            | Precision::Int4
            | Precision::Mxfp4
            | Precision::Mxfp8,
        ) => Cycles(4),
        (IsaGen::Sen1p5, Precision::Int8) => Cycles(2),
        (isa_gen, unknown) => panic!(
            "DT_ERROR(\"Unknown precision for exposed pipeline\") \
             (Dialect/Sentient/SentientOps.cpp:57) — {} on {isa_gen:?}",
            unknown.spelling()
        ),
    }
}

/// `unit.walk<WalkOrder::PreOrder>([&](sentient::NOPOp nop_op) { … }); nop_op->erase();` (`:353-359`)
/// — every `sentient.nop` of this unit, at every depth, gone.
fn erase_nops(scope: &mut Vec<Op>) {
    scope.retain(|op| !matches!(op, Op::Sentient(sentient::Op::Nop { .. })));
    for op in scope.iter_mut() {
        for region in dialects::regions_mut(op) {
            erase_nops(region);
        }
    }
}

/// Replaces: e585_runOnOperation
///
/// The pass entry: on each PT unit, throw away the NOPs a previous run left, then re-open every RAW
/// hazard the exposed pipeline needs by inserting the NOPs its precision's cycle budget asks for.
///
/// ⛔ THE NOP SWEEP IS UNCONDITIONAL AND COMES FIRST (`:353-359`), so a unit whose hazards have all
/// been closed since loses its NOPs and gains none back — this pass is idempotent by rebuilding.
/// ⛔ `precision->str()` ON A PT UNIT CARRYING NONE IS A `bad_optional_access` (`:362-364`), the same
/// unchecked deref [`super::canonicalize_xrf_pointers::run_on_operation`] preserves.
/// ⛔ `cycles -= 1` "covering the exiting FMA operation" (`:365-367`) happens BEFORE the analysis, so
/// both the hazard filter and the NOP count spend the reduced budget.
/// ⚠️ ONE `TimeStamps` HANDLE SERVES EVERY UNIT where the reference constructs `TimeStamp
/// ts_analyzer(unit)` per unit (`:368`): the analysis is out of campaign scope, and e543 re-timestamps
/// the body it is given as its own first act.
pub(crate) fn run_on_operation<A: Arch, M: Model, W: Workload>(
    program: &mut Program<A, M, W>,
    ts: &mut impl TimeStamps,
) {
    if DISABLE_THIS_PASS {
        return;
    }
    // `dependencies_with_min_gap_` IS THE PASS'S, and `cleanUp()` at the foot of each iteration is
    // what keeps one unit's hazards out of the next one's insertions.
    let mut dependencies = Dependencies::default();
    for unit in program.units.iter_mut() {
        // `dcc::getUnitType(unit_op.getUnits()[0].getDefiningOp<GetUnitOp>()) == PT` — every PT row is
        // the one generic component.
        if !matches!(unit.on.kind(), DfirUnit::PtRow(_)) {
            continue;
        }
        erase_nops(&mut unit.body);
        let precision = precision_of(unit);
        let budget = Cycles(cycles_for_exposed_pipeline::<A>(precision).0 - 1);
        dependencies.compute_dependencies::<A>(ts, &unit.body, budget, precision);
        if !dependencies.with_min_gap.is_empty() {
            dependencies.insert_nop_operations(&mut unit.body, budget);
        }
        dependencies.clean_up();
    }
}

/// `unit.getPrecision()` DEREFERENCED — `precision->str()` (`:362`, `:369`).
fn precision_of<A: Arch>(unit: &ProgramUnit<A>) -> Precision {
    match unit.precision {
        Some(precision) => precision,
        None => panic!(
            "std::bad_optional_access: `precision->str()` on {:?}, a PT unit with no precision \
             (`TransformForExposedPipeline.cpp:362`)",
            unit.on.kind()
        ),
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{Dependencies, index_of_entry, mac_ops_flow_dependence, run_on_operation};
    use crate::arch::Dd2;
    use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::dialects::dataflow::Precision;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::dialects::{Op, Val, sentient};
    use crate::islands::sentient::{Program, ProgramUnit, ProgramUnits};
    use crate::model::Model;
    use crate::transform::sentient::analyses::{
        Cycles, Dependency, GapDirection, TimeStampColumnVal, TimeStamps,
    };
    use crate::units::{DfirUnit, Row};
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

    /// A `TimeStamp` STATING ITS ANSWERS — the analysis is out of campaign scope
    /// ([`OutOfScopeTimeStamps`](crate::transform::sentient::analyses::OutOfScopeTimeStamps) is a
    /// `todo!`), so a test supplies the op order, each op's columns and each pair's gap, and counts
    /// the interval reductions it was asked for.
    #[derive(Default)]
    struct StatedTimeStamps {
        order: Vec<OpId>,
        stamps: Vec<(OpId, Vec<TimeStampColumnVal>)>,
        gaps: Vec<((OpId, OpId), (Cycles, GapDirection))>,
        reductions: usize,
        /// How many times the whole unit was timestamped — e543's own first act.
        computed: usize,
    }

    impl TimeStamps for StatedTimeStamps {
        fn op_order(&self) -> &[OpId] {
            &self.order
        }

        fn time_stamp(&self, op: &OpId) -> &[TimeStampColumnVal] {
            self.stamps
                .iter()
                .find(|(at, _)| at == op)
                .map_or(&[], |(_, columns)| columns)
        }

        fn is_in_same_block(&self, src: &[TimeStampColumnVal], dst: &[TimeStampColumnVal]) -> bool {
            // `isInSameBlock` (`Analyses/TimeStamps.cpp:287-296`): equal lengths, and every column
            // but the last `compareTypes`-equal.
            src.len() == dst.len()
                && src[..src.len().saturating_sub(1)] == dst[..dst.len().saturating_sub(1)]
        }

        fn cycles_gap(&mut self, src: &OpId, dst: &OpId) -> (Cycles, GapDirection) {
            // ⭐ A PAIR WITH NO STATED GAP IS NOT A HAZARD: `-1` is what the `gap < 0` filter drops.
            self.gaps
                .iter()
                .find(|((from, to), _)| from == src && to == dst)
                .map_or((Cycles(-1), GapDirection::NextIter), |(_, answer)| *answer)
        }

        fn reduce_intervals(&mut self, _intervals: &mut Vec<Dependency>) {
            self.reductions += 1;
        }

        fn compute_time_stamps(
            &mut self,
            _op: &[Op],
            _parent_time_stamps: &mut Vec<TimeStampColumnVal>,
        ) {
            self.computed += 1;
        }
    }

    /// A column naming the `sentient.for` at `[0]`.
    fn in_loop_at_zero() -> Vec<TimeStampColumnVal> {
        vec![
            TimeStampColumnVal::Loop(OpId::at(&[0])),
            TimeStampColumnVal::Constant,
        ]
    }

    /// A `sentient.vector_mac` reading `reads` on operand A and forwarding its result to `writes`,
    /// unrolled `factor` times, with `incr` on both that operand and the result.
    fn mac(
        dbg_name: &str,
        reads: sentient::Port,
        writes: &[sentient::Port],
        incr: bool,
        factor: sentient::UnrollFactor,
    ) -> Op {
        let mut op_a = sentient::Operand::from(reads);
        op_a.unroll_incr = incr;
        Op::Sentient(sentient::Op::VectorMac {
            mask: None,
            xrf_write_ptr: None,
            xrf_read_ptr: None,
            results: Vec::new(),
            op_a,
            op_b: sentient::Operand::from(sentient::Port::Zero),
            op_c: sentient::Operand::from(sentient::Port::Zero),
            result: sentient::ResultPorts {
                forwarding: writes.to_vec(),
                precision: sentient::Precision::Fp16,
                unroll_incr: incr,
            },
            mode: sentient::FmaMode::FusedMulAdd,
            compute_precision: sentient::Precision::Fp16,
            fold_mode: None,
            unroll_factor: factor,
            xrf_read_incr: 0,
            xrf_write_incr: 0,
            data_transfer_only: false,
            is_data_weight: None,
            dbg_name: Some(dbg_name.to_owned()),
        })
    }

    /// A `sentient.for` around `body`.
    fn loop_over(body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv: Val(1),
            bound: Val(2),
            bound_reg: None,
            carried: Vec::new(),
            dbg_name: None,
            body,
        })
    }

    /// A one-armed `sentient.if` around `then_body`.
    fn guard(then_body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::If {
            predicate: sentient::CmpPredicate::Eq,
            lhs: Val(3),
            rhs: Val(4),
            yielded: Vec::new(),
            dbg_name: None,
            then_body,
            else_body: Vec::new(),
        })
    }

    /// A NOP the pass would have written for `of`'s hazard.
    fn nop(of: &str, which: u32) -> Op {
        Op::Sentient(sentient::Op::Nop {
            dbg_name: Some(format!("TFEP({of}, NOP #{which})")),
        })
    }

    /// e236 — the vendor's own case: a result forwarded to `lrf2` that increments per unrolled copy
    /// covers `lrf2` AND `lrf3`, so a reader of either depends on it; a reader of `lrf9` does not, a
    /// result that reaches no register file is never depended on, and a non-MAC is always `true`.
    #[test]
    fn e236_flow_dependence_spans_the_unrolled_lrf_run() {
        let src = mac(
            "src",
            sentient::Port::Zero,
            &[sentient::Port::Lrf(sentient::LrfIndex::L2)],
            true,
            sentient::UnrollFactor::X2,
        );
        let reads_lrf3 = mac(
            "dst",
            sentient::Port::Lrf(sentient::LrfIndex::L3),
            &[],
            false,
            sentient::UnrollFactor::X1,
        );
        let reads_lrf9 = mac(
            "dst",
            sentient::Port::Lrf(sentient::LrfIndex::L9),
            &[],
            false,
            sentient::UnrollFactor::X1,
        );
        let to_link = mac(
            "src",
            sentient::Port::Zero,
            &[sentient::Port::North],
            false,
            sentient::UnrollFactor::X1,
        );
        assert!(mac_ops_flow_dependence::<Dd2>(
            &src,
            &reads_lrf3,
            Precision::Fp16
        ));
        assert!(!mac_ops_flow_dependence::<Dd2>(
            &src,
            &reads_lrf9,
            Precision::Fp16
        ));
        assert!(!mac_ops_flow_dependence::<Dd2>(
            &to_link,
            &reads_lrf3,
            Precision::Fp16
        ));
        // A `sentient.nop` on either side is not a MAC, and that answers `true`.
        let not_a_mac = Op::Sentient(sentient::Op::Nop { dbg_name: None });
        assert!(mac_ops_flow_dependence::<Dd2>(
            &not_a_mac,
            &reads_lrf3,
            Precision::Fp16
        ));
    }

    /// e234 — every recorded source op and its dependents are dumped, in position order; the pass's
    /// own empty map (which is all `computeDependencies` ever leaves it) prints nothing.
    #[test]
    fn e234_prints_each_source_then_its_dependents() {
        let body = vec![
            mac(
                "a",
                sentient::Port::Zero,
                &[],
                false,
                sentient::UnrollFactor::X1,
            ),
            mac(
                "b",
                sentient::Port::Zero,
                &[],
                false,
                sentient::UnrollFactor::X1,
            ),
        ];
        let mut dependencies = Dependencies::default();
        assert_eq!(dependencies.print_dependencies(&body), String::new());

        dependencies.debug.insert(
            OpId::at(&[0]),
            vec![(OpId::at(&[1]), vec![Default::default()])],
        );
        let printed = dependencies.print_dependencies(&body);
        assert_eq!(printed.matches("sentient.vector_mac").count(), 2);
        assert!(printed.find("dbgName = \"a\"") < printed.find("dbgName = \"b\""));
    }

    /// e235 — both lists are emptied, which is what the next program unit needs.
    #[test]
    fn e235_clears_both_lists() {
        let mut dependencies = Dependencies {
            debug: [(OpId::at(&[0]), vec![(OpId::at(&[1]), Vec::new())])]
                .into_iter()
                .collect(),
            with_min_gap: vec![Dependency {
                src: OpId::at(&[0]),
                dst: OpId::at(&[1]),
                gap: Cycles(1),
            }],
        };
        dependencies.clean_up();
        assert_eq!(dependencies, Dependencies::default());
    }

    /// e237 — the vendor's arithmetic on two hazards under ONE ancestor: `a`'s destination sits under
    /// two `sentient.for`s and a `sentient.if`, which costs it three of its four cycles, and `b`'s
    /// costs it nothing but pays the one `a` banked. ⛔ AND `b`'s THREE NOPs LAND BEFORE `a`'s ONE.
    #[test]
    fn e237_inserts_after_the_shared_ancestor_and_banks_what_it_spent() {
        let unrolled = sentient::UnrollFactor::X1;
        let mut body = vec![
            loop_over(vec![
                mac("a", sentient::Port::Zero, &[], false, unrolled),
                mac("b", sentient::Port::Zero, &[], false, unrolled),
            ]),
            loop_over(vec![loop_over(vec![guard(vec![mac(
                "dst1",
                sentient::Port::Zero,
                &[],
                false,
                unrolled,
            )])])]),
            mac("dst2", sentient::Port::Zero, &[], false, unrolled),
        ];
        let dependencies = Dependencies {
            debug: Default::default(),
            with_min_gap: vec![
                Dependency {
                    src: OpId::at(&[0, 0]),
                    dst: OpId::at(&[1, 0, 0, 0]),
                    gap: Cycles(1),
                },
                Dependency {
                    src: OpId::at(&[0, 1]),
                    dst: OpId::at(&[2]),
                    gap: Cycles(1),
                },
            ],
        };
        let before = body.clone();
        dependencies.insert_nop_operations(&mut body, Cycles(4));
        assert_eq!(
            body,
            vec![
                before[0].clone(),
                nop("b", 0),
                nop("b", 1),
                nop("b", 2),
                nop("a", 0),
                before[1].clone(),
                before[2].clone(),
            ]
        );
    }

    /// e482 — the two same-block forward hazards land in two different buckets, `d`→`e`'s under no
    /// enclosing region and `a`→`b`'s under the `sentient.for` on `a`'s timestamp, so `reduceIntervals`
    /// runs twice; `b`→`c` is a `nextIter` gap and is dropped, and the loop's MACs are never paired
    /// with the top-level ones because their timestamps are not the same length.
    #[test]
    fn e482_buckets_same_block_forward_hazards_under_their_innermost_region() {
        let unrolled = sentient::UnrollFactor::X1;
        let body = vec![
            loop_over(vec![
                mac(
                    "a",
                    sentient::Port::Zero,
                    &[sentient::Port::Lrf(sentient::LrfIndex::L2)],
                    false,
                    unrolled,
                ),
                mac(
                    "b",
                    sentient::Port::Lrf(sentient::LrfIndex::L2),
                    &[sentient::Port::Lrf(sentient::LrfIndex::L4)],
                    false,
                    unrolled,
                ),
                mac(
                    "c",
                    sentient::Port::Lrf(sentient::LrfIndex::L4),
                    &[],
                    false,
                    unrolled,
                ),
            ]),
            mac(
                "d",
                sentient::Port::Zero,
                &[sentient::Port::Lrf(sentient::LrfIndex::L6)],
                false,
                unrolled,
            ),
            mac(
                "e",
                sentient::Port::Lrf(sentient::LrfIndex::L6),
                &[],
                false,
                unrolled,
            ),
        ];
        let mut ts = StatedTimeStamps {
            order: vec![
                OpId::at(&[0, 0]),
                OpId::at(&[0, 1]),
                OpId::at(&[0, 2]),
                OpId::at(&[1]),
                OpId::at(&[2]),
            ],
            stamps: vec![
                (OpId::at(&[0, 0]), in_loop_at_zero()),
                (OpId::at(&[0, 1]), in_loop_at_zero()),
                (OpId::at(&[0, 2]), in_loop_at_zero()),
                (OpId::at(&[1]), vec![TimeStampColumnVal::Constant]),
                (OpId::at(&[2]), vec![TimeStampColumnVal::Constant]),
            ],
            gaps: vec![
                (
                    (OpId::at(&[0, 0]), OpId::at(&[0, 1])),
                    (Cycles(2), GapDirection::Forward),
                ),
                (
                    (OpId::at(&[0, 1]), OpId::at(&[0, 2])),
                    (Cycles(1), GapDirection::NextIter),
                ),
                (
                    (OpId::at(&[1]), OpId::at(&[2])),
                    (Cycles(3), GapDirection::Forward),
                ),
            ],
            reductions: 0,
            computed: 0,
        };

        let mut dependencies = Dependencies::default();
        dependencies.compute_dependencies_same_block::<Dd2>(
            &mut ts,
            &body,
            Cycles(4),
            Precision::Fp16,
        );
        assert_eq!(
            dependencies.with_min_gap,
            vec![
                Dependency {
                    src: OpId::at(&[1]),
                    dst: OpId::at(&[2]),
                    gap: Cycles(3),
                },
                Dependency {
                    src: OpId::at(&[0, 0]),
                    dst: OpId::at(&[0, 1]),
                    gap: Cycles(2),
                },
            ]
        );
        assert_eq!(ts.reductions, 2);
        // The bucketed `debug` map stays empty: the reference builds `dst_operations_debug_` and
        // never assigns it anywhere.
        assert!(dependencies.debug.is_empty());
    }

    /// e483 — `s` reads what it writes, so it is banked as a self-dependence with gap **0** rather
    /// than its stated 5; `s`→`t` is a same-block forward pair e482 already owns and is skipped;
    /// `t`→`u` crosses out of the loop and is grouped under `u` alone, so no reduction runs. Re-run
    /// over the list e482 would have left, the self-dependence LOWERS that entry to 5 instead.
    #[test]
    fn e483_banks_a_self_dependence_at_zero_and_only_ever_lowers_it() {
        let unrolled = sentient::UnrollFactor::X1;
        let body = vec![
            loop_over(vec![
                mac(
                    "s",
                    sentient::Port::Lrf(sentient::LrfIndex::L2),
                    &[sentient::Port::Lrf(sentient::LrfIndex::L2)],
                    false,
                    unrolled,
                ),
                mac(
                    "t",
                    sentient::Port::Lrf(sentient::LrfIndex::L2),
                    &[sentient::Port::Lrf(sentient::LrfIndex::L5)],
                    false,
                    unrolled,
                ),
            ]),
            mac(
                "u",
                sentient::Port::Lrf(sentient::LrfIndex::L5),
                &[],
                false,
                unrolled,
            ),
        ];
        let stated = || StatedTimeStamps {
            order: vec![OpId::at(&[0, 0]), OpId::at(&[0, 1]), OpId::at(&[1])],
            stamps: vec![
                (OpId::at(&[0, 0]), in_loop_at_zero()),
                (OpId::at(&[0, 1]), in_loop_at_zero()),
                (OpId::at(&[1]), vec![TimeStampColumnVal::Constant]),
            ],
            gaps: vec![
                (
                    (OpId::at(&[0, 0]), OpId::at(&[0, 0])),
                    (Cycles(5), GapDirection::NextIter),
                ),
                (
                    (OpId::at(&[0, 0]), OpId::at(&[0, 1])),
                    (Cycles(2), GapDirection::Forward),
                ),
                (
                    (OpId::at(&[0, 1]), OpId::at(&[1])),
                    (Cycles(4), GapDirection::Forward),
                ),
            ],
            reductions: 0,
            computed: 0,
        };
        let crossing = Dependency {
            src: OpId::at(&[0, 1]),
            dst: OpId::at(&[1]),
            gap: Cycles(4),
        };

        let mut ts = stated();
        let mut fresh = Dependencies::default();
        fresh.compute_dependencies_across_blocks::<Dd2>(&mut ts, &body, Cycles(6), Precision::Fp16);
        assert_eq!(
            fresh.with_min_gap,
            vec![
                Dependency {
                    src: OpId::at(&[0, 0]),
                    dst: OpId::at(&[0, 0]),
                    gap: Cycles(0),
                },
                crossing.clone(),
            ]
        );
        assert_eq!(ts.reductions, 0);

        let mut ts = stated();
        let mut banked = Dependencies {
            debug: Default::default(),
            with_min_gap: vec![Dependency {
                src: OpId::at(&[0, 0]),
                dst: OpId::at(&[0, 0]),
                gap: Cycles(9),
            }],
        };
        banked.compute_dependencies_across_blocks::<Dd2>(
            &mut ts,
            &body,
            Cycles(6),
            Precision::Fp16,
        );
        assert_eq!(
            banked.with_min_gap,
            vec![
                Dependency {
                    src: OpId::at(&[0, 0]),
                    dst: OpId::at(&[0, 0]),
                    gap: Cycles(5),
                },
                crossing,
            ]
        );
    }

    /// e543 — the unit is timestamped ONCE, then both halves run IN THAT ORDER: e482's same-block
    /// hazard takes source `s` first, which is what stops e483 banking `s`'s self-dependence at gap 0.
    #[test]
    fn e543_timestamps_the_unit_then_runs_both_halves_in_order() {
        let unrolled = sentient::UnrollFactor::X1;
        let body = vec![
            loop_over(vec![
                mac(
                    "s",
                    sentient::Port::Lrf(sentient::LrfIndex::L2),
                    &[sentient::Port::Lrf(sentient::LrfIndex::L2)],
                    false,
                    unrolled,
                ),
                mac(
                    "t",
                    sentient::Port::Lrf(sentient::LrfIndex::L2),
                    &[sentient::Port::Lrf(sentient::LrfIndex::L5)],
                    false,
                    unrolled,
                ),
            ]),
            mac(
                "u",
                sentient::Port::Lrf(sentient::LrfIndex::L5),
                &[],
                false,
                unrolled,
            ),
        ];
        let mut ts = StatedTimeStamps {
            order: vec![OpId::at(&[0, 0]), OpId::at(&[0, 1]), OpId::at(&[1])],
            stamps: vec![
                (OpId::at(&[0, 0]), in_loop_at_zero()),
                (OpId::at(&[0, 1]), in_loop_at_zero()),
                (OpId::at(&[1]), vec![TimeStampColumnVal::Constant]),
            ],
            gaps: vec![
                (
                    (OpId::at(&[0, 0]), OpId::at(&[0, 0])),
                    (Cycles(5), GapDirection::NextIter),
                ),
                (
                    (OpId::at(&[0, 0]), OpId::at(&[0, 1])),
                    (Cycles(2), GapDirection::Forward),
                ),
                (
                    (OpId::at(&[0, 1]), OpId::at(&[1])),
                    (Cycles(4), GapDirection::Forward),
                ),
            ],
            reductions: 0,
            computed: 0,
        };
        let mut deps = Dependencies::default();

        deps.compute_dependencies::<Dd2>(&mut ts, &body, Cycles(6), Precision::Fp16);

        assert_eq!(ts.computed, 1);
        // e482's one bucket was reduced, and e483 reduces nothing here.
        assert_eq!(ts.reductions, 1);
        assert_eq!(
            deps.with_min_gap,
            vec![
                Dependency {
                    src: OpId::at(&[0, 0]),
                    dst: OpId::at(&[0, 1]),
                    gap: Cycles(2),
                },
                Dependency {
                    src: OpId::at(&[0, 1]),
                    dst: OpId::at(&[1]),
                    gap: Cycles(4),
                },
            ]
        );
    }

    /// One program unit on `kind` holding `body`, at `fp16`.
    fn unit_on(kind: DfirUnit, body: Vec<Op>) -> ProgramUnit<Dd2> {
        ProgramUnit {
            on: Units::one(kind, Val(0)),
            precision: Some(Precision::Fp16),
            body,
            arch: core::marker::PhantomData,
        }
    }

    /// e585 — a PT unit loses the NOPs of a previous run at every depth and gains the two its `fp16`
    /// budget of `3 - 1` cycles asks for, while the unit that is not a PT row keeps its own NOP: the
    /// component gate is the whole pass.
    #[test]
    fn e585_rebuilds_the_nops_of_every_pt_unit_and_leaves_the_rest_alone() {
        let unrolled = sentient::UnrollFactor::X1;
        let s = mac(
            "s",
            sentient::Port::Lrf(sentient::LrfIndex::L2),
            &[sentient::Port::Lrf(sentient::LrfIndex::L2)],
            false,
            unrolled,
        );
        let t = mac(
            "t",
            sentient::Port::Lrf(sentient::LrfIndex::L2),
            &[sentient::Port::Lrf(sentient::LrfIndex::L5)],
            false,
            unrolled,
        );
        let u = mac(
            "u",
            sentient::Port::Lrf(sentient::LrfIndex::L5),
            &[],
            false,
            unrolled,
        );
        let pt = unit_on(
            DfirUnit::PtRow(Row::checked(0).expect("PT row 0 exists")),
            vec![
                loop_over(vec![s.clone(), t.clone(), nop("stale", 0)]),
                u.clone(),
                nop("stale", 1),
            ],
        );
        let sfp = unit_on(DfirUnit::Sfp, vec![nop("kept", 0)]);
        let mut program: Program<Dd2, AnyModel, AnyRung> = Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(pt, vec![sfp]),
            bound: core::marker::PhantomData,
        };
        // The e543 fixture's answers, over the body the NOP sweep leaves behind.
        let mut ts = StatedTimeStamps {
            order: vec![OpId::at(&[0, 0]), OpId::at(&[0, 1]), OpId::at(&[1])],
            stamps: vec![
                (OpId::at(&[0, 0]), in_loop_at_zero()),
                (OpId::at(&[0, 1]), in_loop_at_zero()),
                (OpId::at(&[1]), vec![TimeStampColumnVal::Constant]),
            ],
            gaps: vec![
                (
                    (OpId::at(&[0, 0]), OpId::at(&[0, 1])),
                    (Cycles(2), GapDirection::Forward),
                ),
                (
                    (OpId::at(&[0, 1]), OpId::at(&[1])),
                    (Cycles(4), GapDirection::Forward),
                ),
            ],
            reductions: 0,
            computed: 0,
        };

        run_on_operation(&mut program, &mut ts);

        // ⭐ ONLY THE PT UNIT WAS TIMESTAMPED, so the SFP one was never analysed at all.
        assert_eq!(ts.computed, 1);
        let mut units = program.units.iter();
        assert_eq!(
            units.next().expect("the PT unit").body,
            vec![
                loop_over(vec![s, nop("s", 0), nop("s", 1), t]),
                u,
            ]
        );
        // `t`→`u`'s gap of 4 is outside the reduced budget of 2, so `u` is left standing.
        assert_eq!(units.next().expect("the SFP unit").body, vec![nop("kept", 0)]);
    }

    /// 🎯 e391 — a banked SOURCE is found whatever destination it was banked against, and one that is
    /// banked only as a DESTINATION is not found at all.
    #[test]
    fn e391_finds_a_banked_source_and_not_a_banked_destination() {
        let banked = [
            Dependency {
                src: OpId::at(&[0]),
                dst: OpId::at(&[1]),
                gap: Cycles(3),
            },
            Dependency {
                src: OpId::at(&[2]),
                dst: OpId::at(&[2]),
                gap: Cycles(0),
            },
        ];
        assert_eq!(index_of_entry(&banked, &OpId::at(&[0])), Some(0));
        assert_eq!(index_of_entry(&banked, &OpId::at(&[2])), Some(1));
        assert_eq!(index_of_entry(&banked, &OpId::at(&[1])), None);
        assert_eq!(index_of_entry(&[], &OpId::at(&[0])), None);
    }
}

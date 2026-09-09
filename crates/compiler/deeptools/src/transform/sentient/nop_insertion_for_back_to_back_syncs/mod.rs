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

//! `NOPInsertionForBackToBackSyncs.cpp` — 3 of the campaign's 656 units (dependency level(s) [0, 1]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e100_runOn` | 100 | 0 | 14 | `dcc/src/Transform/Sentient/NOPInsertionForBackToBackSyncs.cpp:66` |
//! | `e101_runOn` | 101 | 0 | 39 | `dcc/src/Transform/Sentient/NOPInsertionForBackToBackSyncs.cpp:106` |
//! | `e325_runOnOperation` | 325 | 1 | 10 | `dcc/src/Transform/Sentient/NOPInsertionForBackToBackSyncs.cpp:81` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so everything below is reachable only from this
// file's own tests. CI runs clippy with `-D warnings`, so without this the first ported leaf of the
// module fails the gate.
// ⭐ REMOVE THIS WHEN THE PASS IS WIRED: `e325_runOnOperation` is the pass ENTRY and is now ported, but
// nothing calls the entry either, so this is still what keeps the module warning-free.
#![allow(dead_code)]

use crate::arch::{Arch, IsaGen};
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::islands::sentient::Program;
use crate::islands::sentient::dialects::{Op, sentient};
use crate::model::Model;
use crate::workload::Workload;

/// `marker_sync_op` — ⛔ A `SyncOp` HANDLE THAT IS ONLY EVER TESTED FOR NULLNESS (`:120`) and never
/// dereferenced, so the one fact it carries is whether a hard send sync stands immediately before.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Marker {
    /// `marker_sync_op = nullptr` — the last op that lowers to code was not a hard send sync.
    Cleared,
    /// A hard send sync is the last op that lowered to code, so the next one is back-to-back with it.
    AfterHardSend,
}

/// Replaces: e100_runOn
///
/// Runs [`run_on_unit`] over the program's `l0su` units and skips every other kind — *"The hardware
/// bug is only present in L0SU, hence the nop insertion should be done over L0SU programs only"*
/// (`:73-75`).
///
/// ⭐ `dcc::getUnitType((*unit.getUnits().begin()).getDefiningOp<GetUnitOp>())` NEEDS NO WALK HERE:
/// [`crate::islands::dataflow_ir::Units`] binds the kind to the unit list, so the `get_unit` this
/// island would have to find again is the one it was built from.
pub fn run_on_program<A: Arch, M: Model, W: Workload>(program: &mut Program<A, M, W>) {
    for unit in program.units.iter_mut() {
        if unit.on.kind().generic() == GenericComp::L0su {
            run_on_unit(&mut unit.body);
        }
    }
}

/// Replaces: e101_runOn
///
/// Inserts a `sentient.nop` before every hard send sync that follows another one with no
/// code-generating op in between, so no two syncs are back to back.
///
/// ⛔ TRAP: A SYNC THAT IS SOFT, IMPLICIT OR RECEIVE-ONLY IS NEITHER A TRIGGER NOR A CLEAR — it is
/// not in the `isa<>` list at `:126-128`, so it leaves the marker exactly as it was rather than
/// separating two hard sends.
///
/// ⭐ THE PRE-ORDER WALK IS WHY ONE MARKER SPANS THE NESTING: a `sentient.for` is seen (and clears
/// the marker) BEFORE its body, and whatever its body leaves set carries on to the loop's next
/// sibling.
pub fn run_on_unit(body: &mut Vec<Op>) {
    let mut marker = Marker::Cleared;
    insert_nops(body, &mut marker);
}

/// One block of the `walk<WalkOrder::PreOrder>`, with the walk's own `marker_sync_op` threaded
/// through it.
///
/// ⭐ STEP 1 AND STEP 2 IN ONE PASS: `sync_ops_to_insert_nops` exists because the reference cannot
/// mutate while it walks. `OpBuilder builder(sync_op)` puts each NOP immediately before its sync, so
/// inserting there and stepping past it is the same result.
fn insert_nops(block: &mut Vec<Op>, marker: &mut Marker) {
    let mut at = 0;
    while at < block.len() {
        let nop = match &block[at] {
            Op::Sentient(sentient::Op::Sync {
                mode,
                soft,
                implicit_sync_memory_boundary,
                dbg_name,
                ..
            }) => {
                let is_send = matches!(
                    mode,
                    sentient::SyncMode::Send | sentient::SyncMode::SendRecv
                );
                // `getImplicitSyncMemoryBoundary().has_value() && ..value() > 0` — ⛔ AND `None` IS
                // THE `.td`'S -1, which is not > 0 either way.
                let is_implicit =
                    implicit_sync_memory_boundary.is_some_and(|boundary| boundary > 0);
                if is_implicit || *soft || !is_send {
                    None
                } else {
                    // `getNewDbgNameFromOp("NOPInsB2BS(", sync_op, ")")` — ⛔ AND AN UNNAMED SYNC
                    // GIVES AN UNNAMED NOP: the helper returns `nullptr` rather than `NOPInsB2BS()`
                    // when the source carries no `dbgName` (`dcc/src/Utils/Utils.cpp:495-496`).
                    let nop = matches!(*marker, Marker::AfterHardSend).then(|| {
                        Op::Sentient(sentient::Op::Nop {
                            dbg_name: dbg_name.as_ref().map(|name| format!("NOPInsB2BS({name})")),
                        })
                    });
                    *marker = Marker::AfterHardSend;
                    nop
                }
            }
            // `isa<ForOp, IfOp, ReceiveAndStoreOp, AddOp, SubOp, CopyOp>` (`:126-128`) — the ops that
            // lower to code between two syncs. ⭐ `sentient.yield` IS DELIBERATELY NOT ONE: *"we are
            // conservatively ignoring yield .. because we may not know whether it gets manifested"*.
            Op::Sentient(
                sentient::Op::For { .. }
                | sentient::Op::If { .. }
                | sentient::Op::ReceiveAndStore { .. }
                | sentient::Op::ScalarAdd { .. }
                | sentient::Op::ScalarSub { .. }
                | sentient::Op::ScalarCopy { .. },
            ) => {
                *marker = Marker::Cleared;
                None
            }
            _ => None,
        };
        if let Some(nop) = nop {
            block.insert(at, nop);
            at += 1;
        }
        match &mut block[at] {
            Op::Sentient(inner) => {
                for region in sentient::regions_mut(inner) {
                    insert_nops(region, marker);
                }
            }
            Op::AffineFor(loop_op) => insert_nops(&mut loop_op.body, marker),
            Op::UniformRegions(regions) => {
                for region in regions.regions_mut() {
                    insert_nops(&mut region.body, marker);
                }
            }
            // ⛔ PROVED ABSENT BY THE TYPE: a lower-rung region holds
            // `crate::islands::dataflow_ir::dialects::Op`, which has no `Sentient` arm — so neither a
            // `sentient.sync` nor any op in the clear list can be inside one.
            Op::Dataflow(_)
            | Op::Agen(_)
            | Op::VectorChain(_)
            | Op::Affine(_)
            | Op::Vector(_)
            | Op::Arith(_)
            | Op::Scf(_)
            | Op::Symbol(_)
            | Op::Uniform(_) => {}
        }
        at += 1;
    }
}

/// Replaces: e325_runOnOperation
///
/// The pass entry: on DD2 and older the whole module gets [`run_on_program`]; on SEN1P5 and newer
/// nothing does, because the hardware bug the nops work around is fixed there.
///
/// ⛔ THE `DisableThisPass` GATE IS COMMENTED OUT IN THE REFERENCE (`:82`), so the flag is declared
/// (`:52-56`) and read by nothing — the pass runs whatever it is set to, and so does this.
/// ⭐ THE ARCH TEST IS A `const` COMPARISON ON [`IsaGen`], which is ordered: `getArch() <=
/// RCUDD1A_ISA`.
pub(crate) fn run_on_operation<A: Arch, M: Model, W: Workload>(program: &mut Program<A, M, W>) {
    if A::GEN <= IsaGen::Rcudd1a {
        run_on_program(program);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{run_on_operation, run_on_program, run_on_unit};
    use crate::arch::{Arch, Dd2, Sen1p5};
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

    /// A `sentient.sync`, named so an inserted NOP's name is observable.
    fn sync(mode: sentient::SyncMode, soft: bool, implicit: Option<i32>, dbg_name: &str) -> Op {
        Op::Sentient(sentient::Op::Sync {
            mode,
            peers: Vec::new(),
            soft,
            implicit_sync_memory_boundary: implicit,
            dbg_name: Some(dbg_name.to_owned()),
        })
    }

    /// A hard send sync — the kind that both triggers and arms the marker.
    fn hard_send(dbg_name: &str) -> Op {
        sync(sentient::SyncMode::Send, false, None, dbg_name)
    }

    /// `%r = sentient.scalar_add %a, %a : index` — a marker-clearing op.
    fn scalar_add(result: u32) -> Op {
        Op::Sentient(sentient::Op::ScalarAdd {
            lhs: Val(1),
            rhs: Val(1),
            result: Val(result),
            reg: None,
            element_size: None,
            ty: ScalarTy::Index,
        })
    }

    /// A one-unit program of `kind` running `body`, on whichever arch the caller asks for.
    fn program_of<A: Arch>(kind: DfirUnit, body: Vec<Op>) -> Program<A, AnyModel, AnyRung> {
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(
                ProgramUnit {
                    iter_arg: None,
                    on: Units::one(kind, Val(0)),
                    precision: None,
                    body,
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            bound: core::marker::PhantomData,
        }
    }

    /// The same on DD2, which is what e100's and e101's cases run on.
    fn program_on(kind: DfirUnit, body: Vec<Op>) -> Program<Dd2, AnyModel, AnyRung> {
        program_of::<Dd2>(kind, body)
    }

    /// e325 — DD2 gets the nop; SEN1P5, where the hardware bug is fixed, gets nothing.
    #[test]
    fn e325_runs_on_dd2_and_below_only() {
        let body = vec![hard_send("s0"), hard_send("s1")];
        let mut on_dd2 = program_of::<Dd2>(DfirUnit::L0su, body.clone());
        run_on_operation(&mut on_dd2);
        assert_eq!(
            on_dd2
                .units
                .iter()
                .next()
                .expect("the head unit")
                .body
                .len(),
            3
        );

        let mut on_sen1p5 = program_of::<Sen1p5>(DfirUnit::L0su, body.clone());
        run_on_operation(&mut on_sen1p5);
        assert_eq!(
            on_sen1p5.units.iter().next().expect("the head unit").body,
            body
        );
    }

    /// e100 — the `l0su` unit's back-to-back pair is separated; the same body on an `lxsu` is not
    /// touched at all.
    #[test]
    fn e100_inserts_only_into_l0su_units() {
        let body = vec![hard_send("s0"), hard_send("s1")];
        let mut on_l0su = program_on(DfirUnit::L0su, body.clone());
        run_on_program(&mut on_l0su);
        assert_eq!(
            on_l0su.units.iter().next().expect("the head unit").body,
            vec![
                hard_send("s0"),
                Op::Sentient(sentient::Op::Nop {
                    dbg_name: Some("NOPInsB2BS(s1)".to_owned()),
                }),
                hard_send("s1"),
            ]
        );

        let mut on_lxsu = program_on(DfirUnit::Lxsu, body.clone());
        run_on_program(&mut on_lxsu);
        assert_eq!(
            on_lxsu.units.iter().next().expect("the head unit").body,
            body
        );
    }

    /// e101 — a `scalar_add` between two sends clears the marker while a soft, an implicit and a
    /// receive-only sync do not, and one marker spans a loop body.
    #[test]
    fn e101_separates_back_to_back_hard_sends_across_the_nesting() {
        let mut body = vec![
            hard_send("s0"),
            scalar_add(2),
            hard_send("s1"),
            sync(sentient::SyncMode::Send, true, None, "soft"),
            sync(sentient::SyncMode::Send, false, Some(1), "implicit"),
            sync(sentient::SyncMode::Recv, false, None, "recv"),
            Op::Sentient(sentient::Op::For {
                iv_reg: sentient::Reg::UNALLOCATED,
                iv: Val(3),
                bound: Val(4),
                carried: Vec::new(),
                dbg_name: None,
                body: vec![hard_send("s2")],
            }),
            hard_send("s3"),
        ];
        run_on_unit(&mut body);

        let nop = |of: &str| {
            Op::Sentient(sentient::Op::Nop {
                dbg_name: Some(format!("NOPInsB2BS({of})")),
            })
        };
        assert_eq!(
            body,
            vec![
                hard_send("s0"),
                scalar_add(2),
                hard_send("s1"),
                sync(sentient::SyncMode::Send, true, None, "soft"),
                sync(sentient::SyncMode::Send, false, Some(1), "implicit"),
                sync(sentient::SyncMode::Recv, false, None, "recv"),
                Op::Sentient(sentient::Op::For {
                    iv_reg: sentient::Reg::UNALLOCATED,
                    iv: Val(3),
                    bound: Val(4),
                    carried: Vec::new(),
                    dbg_name: None,
                    // The loop clears the marker before its body, so `s2` stands alone inside it.
                    body: vec![hard_send("s2")],
                }),
                // ⭐ AND `s2` ARMED THE MARKER THE LOOP'S SIBLING THEN TRIPS.
                nop("s3"),
                hard_send("s3"),
            ]
        );
    }
}

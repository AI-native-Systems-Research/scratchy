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

//! `AnnotateMacXRFWtRange.cpp` — 2 of the campaign's 656 units (dependency level(s) [1, 2]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e284_annotateMacOp` | 284 | 1 | 48 | `dcc/src/Transform/Sentient/AnnotateMacXRFWtRange.cpp:66` |
//! | `e429_runOnOperation` | 429 | 2 | 40 | `dcc/src/Transform/Sentient/AnnotateMacXRFWtRange.cpp:116` |


use crate::arch::{Arch, IsaGen};
use crate::islands::sentient::Program;
use crate::islands::sentient::dialects::{self, Op, sentient};
use crate::model::Model;
use crate::transform::sentient::analyses::{MinMax, XrfRegisterAnalyzer};
use crate::units::DfirUnit;
use crate::workload::Workload;

/// `-dcc-annotate-mac-xrf-wt-range-disable`, `cl::init(false)` (`:42-45`).
const DISABLE_THIS_PASS: bool = false;

/// `dtGetEnv<bool>("SET_IFIFO_CONVERT").value_or(true)` (`:123`) — whether the IFIFO conversion this
/// annotation feeds is on, defaulting to on.
///
/// ⛔ STATED BY THE BUILD, NOT READ FROM THE ENVIRONMENT, exactly as `SENCORES` becomes
/// [`Arch::CORES`]: an emission that changes under an env var is an emission nothing can reproduce.
const SET_IFIFO_CONVERT: bool = true;

/// Replaces: e284_annotateMacOp
///
/// Sets `isDataWeight` on an mx-precision PT MAC that reads N-link into `opC` and zero into A or B:
/// its XRF WRITE pointer's constant range decides data weights (`< 64`) from scale weights (`:104`).
///
/// ⛔ NO POINTERS IS A `return`, NOT `false` (`:88-95`): the pass runs before
/// `LoopSplittingAndUnrolling`, so an un-annotated MAC is the range being left to ProgIR lowering —
/// writing `Some(false)` there would state the scale range for a MAC nobody measured.
pub(crate) fn annotate_mac_op(fma_op: &mut Op, xrf_reg_analyzer: &mut impl XrfRegisterAnalyzer) {
    let Op::Sentient(sentient::Op::VectorMac {
        xrf_write_ptr,
        op_a,
        op_b,
        op_c,
        compute_precision,
        is_data_weight,
        ..
    }) = fma_op
    else {
        return;
    };
    if !matches!(
        compute_precision,
        sentient::Precision::Mxfp8 | sentient::Precision::Mxfp4 | sentient::Precision::Mxint4
    ) || !(op_a.port == sentient::Port::Zero || op_b.port == sentient::Port::Zero)
        || op_c.port != sentient::Port::North
    {
        return;
    }
    if !matches!(
        op_c.precision,
        sentient::Precision::Fp4 | sentient::Precision::Fp8
    ) {
        panic!(
            "DT_CHECK(N-link has to be in fp4/fp8 precision) (`AnnotateMacXRFWtRange.cpp:79-81`): {:?}",
            op_c.precision
        );
    }
    // `fma_op.getPointers()[0]` — the WRITE pointer, which is `$pointers` front (`:89`).
    let Some(wt_ptr_result) = *xrf_write_ptr else {
        return;
    };
    let min_val = xrf_reg_analyzer.min_max_val_if_constant(wt_ptr_result, MinMax::Min);
    let max_val = xrf_reg_analyzer.min_max_val_if_constant(wt_ptr_result, MinMax::Max);
    let (Some(min_val), Some(max_val)) = (min_val, max_val) else {
        panic!(
            "DT_CHECK(min_val.has_value() && max_val.has_value()) (`AnnotateMacXRFWtRange.cpp:103`): {min_val:?}, {max_val:?}"
        );
    };
    let data_weight = min_val < 64;
    if data_weight != (max_val < 64) {
        panic!(
            "DT_CHECK(XRF-related SSA variable has values in both data and scale ranges) \
             (`AnnotateMacXRFWtRange.cpp:105-108`): {min_val}..={max_val}"
        );
    }
    *is_data_weight = Some(data_weight);
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::{Dd2, Sen1p5};
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::dialects::dataflow;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::dialects::Val;
    use crate::islands::sentient::dialects::sentient::{
        FmaMode, Operand, Port, Precision, ResultPorts, UnrollFactor,
    };
    use crate::islands::sentient::{ProgramUnit, ProgramUnits};
    use crate::transform::sentient::analyses::MinMax;
    use crate::units::Row;

    /// A `mxfp8` MAC reading zero into B and N-link `fp8` into C, writing XRF through `%7`.
    fn mac(xrf_write_ptr: Option<Val>, op_c_port: Port) -> Op {
        Op::Sentient(sentient::Op::VectorMac {
            mask: None,
            xrf_write_ptr,
            xrf_read_ptr: None,
            results: vec![Val(1)],
            op_a: Operand::from(Port::Xrf),
            op_b: Operand::from(Port::Zero),
            op_c: Operand {
                precision: Precision::Fp8,
                ..Operand::from(op_c_port)
            },
            result: ResultPorts::default(),
            mode: FmaMode::FusedMulAdd,
            compute_precision: Precision::Mxfp8,
            fold_mode: None,
            unroll_factor: UnrollFactor::X1,
            xrf_read_incr: 0,
            xrf_write_incr: 0,
            data_transfer_only: false,
            is_data_weight: None,
            dbg_name: None,
        })
    }

    fn annotation(op: &Op) -> Option<bool> {
        match op {
            Op::Sentient(sentient::Op::VectorMac { is_data_weight, .. }) => *is_data_weight,
            _ => None,
        }
    }

    /// A `[8, 40]` write pointer is the data range; the same MAC without pointers, and one whose
    /// `opC` is not N-link, are both left un-annotated for ProgIR lowering to decide.
    #[test]
    fn e284_annotates_a_constant_data_range_and_leaves_the_pointerless_mac_alone() {
        struct StatedAnalyzer;

        impl XrfRegisterAnalyzer for StatedAnalyzer {
            fn min_max_val_if_constant(&mut self, value: Val, end: MinMax) -> Option<i64> {
                assert_eq!(value, Val(7));
                Some(match end {
                    MinMax::Min => 8,
                    MinMax::Max => 40,
                })
            }
        }

        let mut annotated = mac(Some(Val(7)), Port::North);
        annotate_mac_op(&mut annotated, &mut StatedAnalyzer);
        assert_eq!(annotation(&annotated), Some(true));

        let mut pointerless = mac(None, Port::North);
        annotate_mac_op(&mut pointerless, &mut StatedAnalyzer);
        assert_eq!(annotation(&pointerless), None);

        let mut not_north = mac(Some(Val(7)), Port::West);
        annotate_mac_op(&mut not_north, &mut StatedAnalyzer);
        assert_eq!(annotation(&not_north), None);
    }

    /// A model and rung, for the same reason [`ProgramUnit`] needs an arch: the pass reads neither.
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

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyRung;
    impl Workload for AnyRung {
        const ROWS: u32 = 1;
        const ACTIVE_CAP: u32 = 64;
    }

    /// A `[8, 40]` write pointer — the data range, for any MAC that reaches the analyzer.
    struct DataRange;

    impl XrfRegisterAnalyzer for DataRange {
        fn min_max_val_if_constant(&mut self, _value: Val, end: MinMax) -> Option<i64> {
            Some(match end {
                MinMax::Min => 8,
                MinMax::Max => 40,
            })
        }
    }

    /// A PT-row unit whose MAC sits inside a `sentient.for`, beside an LXLU unit holding one at top
    /// level — so the walk's depth and the PT filter are both visible.
    fn program<A: Arch>() -> Program<A, AnyModel, AnyRung> {
        let loop_op = Op::Sentient(sentient::Op::For {
            iv: Val(50),
            bound: Val(51),
            bound_reg: None,
            carried: Vec::new(),
            dbg_name: None,
            body: vec![mac(Some(Val(7)), Port::North)],
        });
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(DfirUnit::Lxlu, Val(0)),
                    precision: None,
                    body: vec![mac(Some(Val(7)), Port::North)],
                    arch: core::marker::PhantomData,
                },
                vec![ProgramUnit {
                    on: Units::one(DfirUnit::PtRow(Row::checked(0).expect("row 0")), Val(1)),
                    precision: Some(dataflow::Precision::Int8),
                    body: vec![loop_op],
                    arch: core::marker::PhantomData,
                }],
            ),
            bound: core::marker::PhantomData,
        }
    }

    /// The MAC nested in the PT unit's loop, and the one the LXLU unit holds.
    fn annotations<A: Arch>(
        program: &Program<A, AnyModel, AnyRung>,
    ) -> (Option<bool>, Option<bool>) {
        let mut units = program.units.iter();
        let lxlu = annotation(&units.next().expect("the lxlu").body[0]);
        let Op::Sentient(sentient::Op::For { body, .. }) =
            &units.next().expect("the pt row").body[0]
        else {
            panic!("the loop survives")
        };
        (annotation(&body[0]), lxlu)
    }

    /// 429/656 — on SEN1P5 the PT unit's nested MAC is annotated and the LXLU's is not touched; on
    /// DD2 the arch gate stops the pass before either.
    #[test]
    fn e429_annotates_nested_pt_macs_on_sen1p5_only() {
        let mut sen1p5 = program::<Sen1p5>();
        run_on_program(&mut sen1p5, &mut DataRange);
        assert_eq!(annotations(&sen1p5), (Some(true), None));

        let mut dd2 = program::<Dd2>();
        run_on_program(&mut dd2, &mut DataRange);
        assert_eq!(annotations(&dd2), (None, None));
    }
}

/// Replaces: e429_runOnOperation
///
/// Annotates every MAC of every PT unit, on SEN1P5 and later only (`:130-132`).
///
/// ⛔ THE ARCH GATE IS `<`, ON AN ORDER THAT IS LOAD-BEARING: [`IsaGen`] is ordered so this reads as
/// the reference's `getArch() < SEN1P5_ISA`, and an RCUDD1A build must annotate NOTHING.
/// ⭐ THE PER-UNIT `make_unique<XRFRegisterAnalyzer>(unit)` (`:148`) IS THE SEAM'S OWN BUSINESS: the
/// analyzer is out of campaign scope, and every unit's MACs still reach it in unit order.
/// ⭐ `if (!get_unit_op) return;` needs no expression — a [`Units`](crate::islands::dataflow_ir::Units)
/// carries its [`DfirUnit`] as a closed enum rather than a nullable defining op.
pub(crate) fn run_on_program<A: Arch, M: Model, W: Workload>(
    program: &mut Program<A, M, W>,
    xrf_reg_analyzer: &mut impl XrfRegisterAnalyzer,
) {
    if DISABLE_THIS_PASS || !SET_IFIFO_CONVERT || A::GEN < IsaGen::Sen1p5 {
        return;
    }
    for unit in program.units.iter_mut() {
        // `senCompToGenericComp.at(...) != PT -> return` — every PT row is the one generic component.
        if !matches!(unit.on.kind(), DfirUnit::PtRow(_)) {
            continue;
        }
        annotate_macs(&mut unit.body, xrf_reg_analyzer);
    }
}

/// `unit.walk([&](sentient::MacOp fma_op) { annotateMacOp(..); })` — the nested walk, which is the
/// mechanism the campaign names droppable. The `MacOp` filter is
/// [`annotate_mac_op`]'s own first match, so every op is offered to it.
fn annotate_macs(body: &mut [Op], xrf_reg_analyzer: &mut impl XrfRegisterAnalyzer) {
    for op in body.iter_mut() {
        annotate_mac_op(op, xrf_reg_analyzer);
        for region in dialects::regions_mut(op) {
            annotate_macs(region, xrf_reg_analyzer);
        }
    }
}

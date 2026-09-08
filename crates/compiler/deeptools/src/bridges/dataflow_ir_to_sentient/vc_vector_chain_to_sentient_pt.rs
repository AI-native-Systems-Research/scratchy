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

//! `VectorChainToSentientPT.cpp` — 5 of bridge 2's 384 functions (dependency level(s) [0, 6, 7, 8]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e094_computeUnitPrecision` | 094/384 | 9 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:31` |
//! | `e346_lowerDanglingNonComputeOps` | 346/384 | 88 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:882` |
//! | `e368_fuseNonComputeOps` | 368/384 | 191 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:46` |
//! | `e369_fuseComputeOps` | 369/384 | 628 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:245` |
//! | `e379_runOnOperation` | 379/384 | 54 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:975` |

use crate::arch::Arch;
use crate::bridges::dataflow_ir_to_sentient::vc_vector_chain_helper::redefine_constant_vectors;
use crate::islands::dataflow_ir::dialects::dataflow;
use crate::islands::dataflow_ir::{self as dfir, Values};
use crate::islands::sentient::dialects::sentient;
use crate::units::DfirUnit;

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 094/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e094_computeUnitPrecision
///
/// # THE PRECISION A PT UNIT'S COMPUTES ARE EMITTED AT
///
/// ```cpp
/// std::string VectorChainToSentientPTLoweringPass::computeUnitPrecision(
///     dataflow::ProgramUnitOp &unit, const SenComponents &comp) {
///   DT_CHECK(comp == PT);
///   DT_CHECK_MSG(unit.getPrecision().has_value(),
///                "Precision attribute for PT is expected");
///   std::string precision = unit.getPrecision().value().str();
///   // Currently, we use fp80 type in MLIR to represent fp8.
///   if (precision == "fp80") return "fp8";
///
///   return precision;
/// }
/// ```
/// (`dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:30-41`)
///
/// # ⭐⭐ WHAT IT PRODUCES IS THE MAC'S `ComputePrecision`
///
/// Its one caller passes the answer straight into the compute lowering, and the vendor's own golden
/// shows where it lands: a unit declared `dataflow.program_unit … {precision = "mxfp4"}` yields
/// `sentient.vector_mac {ComputePrecision = #sentient<precision mxfp4>, …}` at all 24 of its MACs
/// (`dcc/test/Conversion/VectorChainToSentientPT/xrf_increments.mlir:374`). So this is a
/// `dataflow`-rung spelling crossing to the `sentient`-rung enum — which is why the port's signature
/// is [`dataflow::Precision`] in, [`sentient::Precision`] out, rather than `String` to `String`.
///
/// # ⛔⛔ ONE NON-IDENTITY ENTRY, AND IT IS THE WHOLE FUNCTION
///
/// `fp80 -> fp8`. Everything else passes through. The reference states the reason in a comment —
/// *"Currently, we use fp80 type in MLIR to represent fp8"* — and
/// [`dataflow::Precision::Fp80`] records that the spelling appears NOWHERE in the authority tree's
/// `dcc/test`, so this branch is defensive. It is still the only content this function has: a port
/// that dropped it would be `identity` with a citation attached, which is exactly the failure mode
/// the campaign brief names.
///
/// # THE TWO `DT_CHECK`s
///
/// * `DT_CHECK(comp == PT)` — ⛔ **UNREPRESENTABLE HERE.** The component parameter's only use in the
///   body is this comparison. Dropping it removes the way to call this with anything else: the
///   function names the PT lowering in its module and takes no component, so there is no value to
///   compare and no comparison to fail.
/// * `DT_CHECK_MSG(unit.getPrecision().has_value(), "Precision attribute for PT is expected")` —
///   ⛔ **DISCHARGED BY CONSTRUCTION.** The parameter is a [`dataflow::Precision`], not an
///   `Optional`. The island's `ProgramUnit::precision` IS an `Option` (a `dataflow.program_unit` may
///   legitimately carry no attribute, and 
///   [`crate::islands::dataflow_ir::dialects::dataflow::Precision`] documents why), so the absent
///   case is a fact about the UNIT that its reader states — and this function is only reachable once
///   that reader has one, which is what taking the value rather than the option means.
#[must_use]
pub const fn compute_unit_precision(precision: dataflow::Precision) -> sentient::Precision {
    match precision {
        // ⭐ THE ONE REMAP.
        dataflow::Precision::Fp80 | dataflow::Precision::Fp8 => sentient::Precision::Fp8,

        // ── `return precision` — the same spelling, at the sentient rung ─────────────────────────
        dataflow::Precision::Int8 => sentient::Precision::Int8,
        dataflow::Precision::Int4 => sentient::Precision::Int4,
        dataflow::Precision::Fp4 => sentient::Precision::Fp4,
        dataflow::Precision::Fp16 => sentient::Precision::Fp16,
        dataflow::Precision::Fp32 => sentient::Precision::Fp32,
        dataflow::Precision::Bf16 => sentient::Precision::Bf16,
        dataflow::Precision::Mxfp4 => sentient::Precision::Mxfp4,
        dataflow::Precision::Mxfp8 => sentient::Precision::Mxfp8,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 379/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e379_runOnOperation
///
/// `VectorChainToSentientPTLoweringPass::runOnOperation` (`VectorChainToSentientPT.cpp:975`) —
/// redefine the module's constant vectors once, then for each PT row unit: create the XRF index
/// modification ops, build the loop mask tree, build the reuse analysis, reset any existing sentient
/// FMAs, fuse non-compute then compute then dangling non-compute ops, validate, and finally insert
/// the PT mask ops.
///
/// ⛔ `redefineConstantVectors` IS OUTSIDE THE WALK HERE (`:983`) — the PESFP pass calls the same
/// helper from INSIDE its walk (`VectorChainToSentientPESFP.cpp:1385`), so this one runs once per
/// module and that one runs once per PE/SFP unit.
/// ⛔ `WalkResult::interrupt()` AT `:1019` IS A DISCARDED TEMPORARY — it is not returned, so a
/// failed validation signals pass failure and then STILL falls through to `insertPTMaskOps(:1024)`
/// and returns `skip()`. Treating it as a `break` would drop the mask ops of the failing unit.
/// ⛔ THE XRF MAP AND THE MASK TREE ARE BUILT BEFORE THE REUSE ANALYSIS (`:995-1003` before `:1006`)
/// and all three fusion steps take all three, so none of them can be defaulted away.
/// ⭐ THE MASK TREE IS POPULATED HERE, NOT IN THE FUSIONS — *"PT masking relies on loops being
/// stable"* (`:1000-1001`): it is read during fusion and only spent by `insertPTMaskOps`.
pub fn run_on_operation<A: Arch>(program: &mut dfir::Program<A>, values: &mut Values) {
    // `:983` — entry 067, once for the whole module.
    redefine_constant_vectors(program, values);

    // `:985` — the walk over the module's program units.
    for unit in program.units.iter() {
        // `:990` — `if (unit_comp != PT) return WalkResult::skip();`. A PT unit is one PT ROW here:
        // the island names the row the unit runs on, which is what `getUnitType` returns for each of
        // the rows the reference walks.
        let comp = unit.on.kind();
        if !matches!(comp, DfirUnit::PtRow(_)) {
            continue;
        }

        // `:995-997` — `LoweringXRF::createXrfIndexModifOps(unit, unit_comp)`, whose map every step
        // below takes.
        todo!(
            "e367_createXrfIndexModifOps is unported, so e368_fuseNonComputeOps, \
             e369_fuseComputeOps, e346_lowerDanglingNonComputeOps, \
             e228_validateLoweringAndSetMissingParameters and e239_insertPTMaskOps cannot run on the \
             {:?} unit (e227_OperandReuse and the LoopMaskTree of `:1003` are unported too)",
            comp
        );
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{compute_unit_precision, run_on_operation};
    use crate::arch::Target;
    use crate::islands::dataflow_ir::dialects::Val;
    use crate::islands::dataflow_ir::dialects::dataflow;
    use crate::islands::dataflow_ir::{self as dfir, Values};
    use crate::islands::sentient::dialects::sentient;
    use crate::units::{DfirUnit, Row};

    /// One program holding one unit on `on` — what this pass walks.
    fn program_of(on: DfirUnit) -> dfir::Program<Target> {
        use crate::generated::OpFunc;
        use crate::islands::dataflow_ir::{
            Grid, GroupId, OpIndex, ProgramName, ProgramUnit, ProgramUnits, Units,
        };
        dfir::Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            grid: Grid::single(),
            preamble: Vec::new(),
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(on, Val(0)),
                    precision: None,
                    body: Vec::new(),
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            arch: core::marker::PhantomData,
        }
    }

    /// 🎯 094/384 — THE ONE NON-IDENTITY ENTRY.
    ///
    /// `if (precision == "fp80") return "fp8";`
    /// (`VectorChainToSentientPT.cpp:37-38`) — the whole reason this function is not the identity.
    #[test]
    fn fp80_is_the_mlir_spelling_of_fp8() {
        assert_eq!(
            compute_unit_precision(dataflow::Precision::Fp80),
            sentient::Precision::Fp8
        );
    }

    /// 🎯 094/384 — AND EVERY OTHER SPELLING SURVIVES ITSELF.
    ///
    /// ⭐ THE FOUR MEASURED SPELLINGS ARE ALL HERE. A census of `precision = "…"` across the
    /// authority tree's `dcc/test` gives `int8` 500, `fp16` 492, `mxfp8` 4, `mxfp4` 4, `bf16` 3,
    /// `fp8` 2, `fp32` 2, `int4` 1 — and `xrf_increments.mlir:374` pins the `mxfp4` answer against
    /// `ComputePrecision = #sentient<precision mxfp4>` in its own `CHECK-SENT-IR`.
    #[test]
    fn every_other_precision_passes_through_unchanged() {
        for (from, to) in [
            (dataflow::Precision::Int8, sentient::Precision::Int8),
            (dataflow::Precision::Int4, sentient::Precision::Int4),
            (dataflow::Precision::Fp4, sentient::Precision::Fp4),
            (dataflow::Precision::Fp8, sentient::Precision::Fp8),
            (dataflow::Precision::Fp16, sentient::Precision::Fp16),
            (dataflow::Precision::Fp32, sentient::Precision::Fp32),
            (dataflow::Precision::Bf16, sentient::Precision::Bf16),
            (dataflow::Precision::Mxfp4, sentient::Precision::Mxfp4),
            (dataflow::Precision::Mxfp8, sentient::Precision::Mxfp8),
        ] {
            assert_eq!(compute_unit_precision(from), to, "{}", from.spelling());
            // ⭐⭐ AND THE ANSWER SPELLS ITSELF THE SAME. `return precision` returns the STRING, which
            // the caller then symbolizes — so a mapping that changed the spelling would change the
            // attribute, and this is the test that would catch it.
            assert_eq!(from.spelling(), to.spelling(), "{}", from.spelling());
        }
    }

    /// 🎯 379/384 — A PT ROW UNIT IS CLAIMED AND REACHES THE XRF PREPROCESSING. `redefine_constant_vectors`
    /// (entry 067) has already run for the whole module by then (`:983`), before the walk starts.
    #[test]
    #[should_panic(expected = "e367_createXrfIndexModifOps")]
    fn a_pt_row_unit_is_claimed_by_this_pass() {
        let Some(row) = Row::checked(0) else {
            unreachable!()
        };
        let mut program = program_of(DfirUnit::PtRow(row));
        run_on_operation(&mut program, &mut Values::default());
    }

    /// 🎯⛔ AND A NON-PT UNIT IS SKIPPED (`:990`) — the SFP is the PESFP pass's (entry 378), and a
    /// gate that let it through would lower it twice.
    #[test]
    fn a_non_pt_unit_is_left_to_the_pesfp_pass() {
        let mut program = program_of(DfirUnit::Sfp);
        run_on_operation(&mut program, &mut Values::default());
    }
}

// ⛔ RE-CREATED ANCHORS. These units' `crustify:todo:` markers were deleted without a
// `/// Replaces:` ever appearing, which removed them from every later schedule and let the
// driver report the campaign DONE. Outstanding work is now computed from UNITS.tsv.
// crustify:todo: e346_lowerDanglingNonComputeOps
// crustify:todo: e368_fuseNonComputeOps
// crustify:todo: e369_fuseComputeOps

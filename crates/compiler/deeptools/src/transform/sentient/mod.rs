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

//! The D29–D75 in-place SentientIR passes — 656 units across 52 passes.
//!
//! Input and output are both `crate::islands::sentient`. Every pass in the shipped driver
//! (`dcc/tools/dcc-standalone/dcc-standalone-main.cpp`, between `createAgenToSentientPass`
//! at :271 and `createSentientToProgIRPass` at :755) is added UNCONDITIONALLY — none is
//! behind a flag or an option, so there is no "just the required ones" subset.

pub mod address_pinning_and_toggle;
pub(crate) mod address_register_precision_assignment;
pub mod analyses;
pub(crate) mod annotate_mac_xrf_wt_range;
pub(crate) mod burst_splitting;
pub(crate) mod canonicalize_xrf_pointers;
pub(crate) mod cfg_deep_merging;
pub(crate) mod cfg_simplification_sentient_level;
pub(crate) mod deuniform;
pub(crate) mod enhanced_dead_variable_elimination;
pub(crate) mod implicit_sync_re;
pub(crate) mod lexical_ordering;
pub(crate) mod lightweight_simplification;
pub(crate) mod live_range_reduction;
pub(crate) mod local_region_splitting_for_value_commoning;
pub(crate) mod loop_absorption;
pub(crate) mod loop_coalescing;
pub(crate) mod loop_elimination_via_rerolling;
pub(crate) mod loop_merging;
pub(crate) mod loop_rolling;
pub(crate) mod loop_splitting_and_unrolling;
pub(crate) mod loop_tree;
pub(crate) mod multi_dim_loop_peeling;
pub(crate) mod multicast_canonicalization;
pub(crate) mod nop_insertion_for_back_to_back_syncs;
pub(crate) mod old_register_initialization;
pub(crate) mod op_rerolling;
pub(crate) mod port_assignment;
pub(crate) mod read_only_register_renumbering;
pub(crate) mod register_allocation;
pub(crate) mod register_initialization;
pub(crate) mod register_packing;
pub(crate) mod register_type_assignment;
pub(crate) mod rematerialization_pass;
pub(crate) mod remove_redundant_conditionals;
pub(crate) mod reuse_loop_iterator_arguments;
pub(crate) mod scalar_copy_insertion_for_symbols;
pub(crate) mod scalar_op_merging_and_hoisting;
pub(crate) mod scalar_op_reordering;
pub(crate) mod scalar_simplifications;
pub(crate) mod set_active_mask_value_re;
pub(crate) mod set_mask_re;
pub(crate) mod set_send_destination_re;
pub(crate) mod simplify_uniform_regions;
pub(crate) mod sink_scalar_copy;
pub(crate) mod smart_register_allocation;
pub(crate) mod specialized_canonicalization;
pub(crate) mod store_and_forward_fusion;
pub(crate) mod sync_send_recv_fusion;
pub(crate) mod toggle_reordering;
pub(crate) mod transform_for_exposed_pipeline;
pub(crate) mod uniform_map_canonicalization;
pub(crate) mod utils;
pub(crate) mod vector_register_initialization;

use crate::islands::sentient::dialects::Val;

/// THE `sentient.for` A DESCRIPTOR OR AN ANALYSIS POINTS AT — `sentient::ForOp`, named by its
/// induction variable.
///
/// ⛔⛔ AN IDENTITY, NOT A BORROW. These passes rewrite SentientIR IN PLACE, so a field holding
/// `&Op` into the tree being rewritten is unusable; `Op::For::iv` is minted once and names exactly
/// one loop ([`crate::islands::sentient::dialects::sentient::Op::For`] — the region's first
/// argument, and the first thing the op prints), so the handle outlives the rewrite.
///
/// ⛔ `None` AT A FIELD IS THE REFERENCE'S `nullptr`, which is what every `invalidate()` writes and
/// every `isValid()` tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ForRef(pub Val);

/// WHICH VALUE A LOOP CARRIES — an index into `Op::For::carried`.
///
/// ⛔ `Option<IterArgIndex>` IS THE REFERENCE'S `int iter_arg_index_ = -1`: absence, never a
/// negative index. `iter_arg_index_ >= 0` is half of four descriptors' `isValid()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IterArgIndex(pub u32);

/// WHETHER THE PROGRAMS ARE BEING STITCHED TOGETHER — `dccExtContext().getProgStitch()`, which is
/// `dsc_global_->doProgStitch` (`Utils/DccExtContext.cpp:329`).
///
/// ⛔ A PASS INPUT, NOT A CONSTANT HERE. It is a property of the whole compilation that four of these
/// passes branch on (`ReadOnlyRegisterRenumbering.cpp:111`, `RegisterPacking.cpp:186`,
/// `SetSendDestinationRE.cpp:147`, `AddressPinningAndToggle.cpp:1509`), so the pass that reads it takes
/// it and the pipeline that knows the answer states it — see
/// [`crate::bridges::dataflow_ir_to_sentient::tf_program_units_reduction`] for the sibling
/// `getFolding()` and why a context flag is not a runtime question in this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgStitch {
    /// `getProgStitch() == true` — this program is one piece of a stitched program.
    Stitched,
    /// `getProgStitch() == false` — the standalone compilation, and the crate's own path today.
    Standalone,
}

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

//! `AddressPinningAndToggle.cpp` — 78 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e001_invalidate` | 001 | 0 | 5 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:271` |
//! | `e002_getAllConstants` | 002 | 0 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:330` |
//! | `e003_invalidate` | 003 | 0 | 6 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:416` |
//! | `e004_invalidate` | 004 | 0 | 5 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:507` |
//! | `e005_invalidate` | 005 | 0 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:600` |
//! | `e006_dtor_DataTransferDescriptor` | 006 | 0 | 3 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:645` |
//! | `e007_isPartOfSomeChain` | 007 | 0 | 5 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:901` |
//! | `e008_setChainingInfo` | 008 | 0 | 10 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:945` |
//! | `e009_createOffsetValue` | 009 | 0 | 14 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1017` |
//! | `e010_updateVariableOffsetCalculation` | 010 | 0 | 3 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1065` |
//! | `e011_getOffset` | 011 | 0 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1104` |
//! | `e012_getOffset` | 012 | 0 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1133` |
//! | `e013_getOffset` | 013 | 0 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1163` |
//! | `e014_dump` | 014 | 0 | 8 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1666` |
//! | `e257_getBaseAddr` | 257 | 1 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:657` |
//! | `e258_isToggle` | 258 | 1 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:677` |
//! | `e259_isConditionalConstant` | 259 | 1 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:681` |
//! | `e260_isIntegerSequence` | 260 | 1 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:685` |
//! | `e261_isDiscreteIntegerSet` | 261 | 1 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:689` |
//! | `e262_isLoopingChainMutableAddr` | 262 | 1 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:693` |
//! | `e263_getToggleDescriptor` | 263 | 1 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:706` |
//! | `e264_getToggleDescriptor` | 264 | 1 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:710` |
//! | `e265_getConditionalConstantDescriptor` | 265 | 1 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:714` |
//! | `e266_getConditionalConstantDescriptor` | 266 | 1 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:718` |
//! | `e267_getIntegerSequenceDescriptor` | 267 | 1 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:723` |
//! | `e268_getIntegerSequenceDescriptor` | 268 | 1 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:727` |
//! | `e269_getDiscreteIntegerSetDescriptor` | 269 | 1 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:731` |
//! | `e270_getDiscreteIntegerSetDescriptor` | 270 | 1 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:735` |
//! | `e271_getLoopingChainMutableAddrDescriptor` | 271 | 1 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:739` |
//! | `e272_getLoopingChainMutableAddrDescriptor` | 272 | 1 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:743` |
//! | `e273_isHeadOfChain` | 273 | 1 | 6 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:907` |
//! | `e274_validate` | 274 | 1 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:929` |
//! | `e275_turnHBMConstantOpAddrsToQueryMapsHelper` | 275 | 1 | 59 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1595` |
//! | `e399_getMin` | 399 | 2 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:182` |
//! | `e400_getMax` | 400 | 2 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:187` |
//! | `e401_getC1` | 401 | 2 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:246` |
//! | `e402_getIterArgIndex` | 402 | 2 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:251` |
//! | `e403_getMin` | 403 | 2 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:320` |
//! | `e404_getMax` | 404 | 2 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:325` |
//! | `e405_getMin` | 405 | 2 | 13 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:360` |
//! | `e406_getMax` | 406 | 2 | 13 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:374` |
//! | `e407_getInit` | 407 | 2 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:388` |
//! | `e408_getAllConstants` | 408 | 2 | 16 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:393` |
//! | `e409_getMin` | 409 | 2 | 5 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:470` |
//! | `e410_getMax` | 410 | 2 | 5 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:476` |
//! | `e411_getInit` | 411 | 2 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:482` |
//! | `e412_getMin` | 412 | 2 | 7 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:566` |
//! | `e413_getMax` | 413 | 2 | 7 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:574` |
//! | `e414_getInit` | 414 | 2 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:582` |
//! | `e415_isSimpleConstant` | 415 | 2 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:673` |
//! | `e416_clear_all` | 416 | 2 | 6 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:886` |
//! | `e417_isHeadOfLoopingChain` | 417 | 2 | 7 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:920` |
//! | `e418_getOffset` | 418 | 2 | 7 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1073` |
//! | `e419_turnHBMConstantOpAddrsToQueryMapsInUniformRegions` | 419 | 2 | 13 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1579` |
//! | `e485_getX` | 485 | 3 | 5 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:227` |
//! | `e486_getSimpleConstantDescriptor` | 486 | 3 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:698` |
//! | `e487_getSimpleConstantDescriptor` | 487 | 3 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:702` |
//! | `e488_computeOrGetNumberOfStreams` | 488 | 3 | 10 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1439` |
//! | `e489_cleanup` | 489 | 3 | 9 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1656` |
//! | `e547_getMin` | 547 | 4 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:235` |
//! | `e548_getMax` | 548 | 4 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:240` |
//! | `e549_runOn` | 549 | 4 | 12 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1187` |
//! | `e587_getMin` | 587 | 5 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:749` |
//! | `e588_getMax` | 588 | 5 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:754` |
//! | `e589_getMin` | 589 | 5 | 9 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:829` |
//! | `e590_getMax` | 590 | 5 | 14 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:839` |
//! | `e591_runOnOperation` | 591 | 5 | 9 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1200` |
//! | `e614_computeAddressInfoList` | 614 | 6 | 59 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1675` |
//! | `e637_collectDataTransfers` | 637 | 7 | 41 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1395` |
//! | `e647_CollectDataTransfersAndComputeMaxStreams` | 647 | 8 | 39 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1293` |
//! | `e648_processToggle` | 648 | 8 | 13 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1522` |
//! | `e649_processConditionalConstant` | 649 | 8 | 10 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1539` |
//! | `e650_processIntegerSequence` | 650 | 8 | 8 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1553` |
//! | `e651_processSimpleConstant` | 651 | 8 | 6 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1565` |
//! | `e653_processDataTransfer` | 653 | 9 | 26 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1492` |
//! | `e654_processDataTransfer` | 654 | 10 | 34 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1456` |
//! | `e655_processDataTransfers` | 655 | 11 | 4 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1450` |
//! | `e656_runOn` | 656 | 12 | 57 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1337` |

pub(crate) mod abstract_data_transfer_updater;
pub(crate) mod conditional_const_data_transfer_updater;
pub(crate) mod conditional_constant_descriptor;
pub(crate) mod data_transfer_descriptor;
pub(crate) mod data_transfer_descriptor_container;
pub(crate) mod discrete_integer_set_descriptor;
pub(crate) mod hbm_data_transfer_descriptor;
pub(crate) mod integer_sequence_data_transfer_updater;
pub(crate) mod integer_sequence_descriptor;
pub(crate) mod looping_chain_mutable_addr_descriptor;
pub(crate) mod lx_data_transfer_descriptor;
pub(crate) mod simple_constant_data_transfer_updater;
pub(crate) mod simple_constant_descriptor;
pub(crate) mod toggle_data_transfer_updater;
pub(crate) mod toggle_descriptor;

use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::arch::Arch;
use crate::islands::sentient::dialects::{
    self, Definitions, Op, UniformRegions, Val, dataflow, sentient, uniform,
};
use crate::islands::sentient::{Program, ProgramUnit};
use crate::model::Model;
use crate::workload::Workload;
use crate::transform::sentient::analyses::{
    EvAddressInfo, EvaluatedValue, ExpressionEvaluator, MinMax, OffsetSites,
};
use crate::transform::sentient::{ForRef, IterArgIndex};
use crate::units::DfirUnit;
use looping_chain_mutable_addr_descriptor::TransferEnd;
use std::collections::BTreeSet;

pub use conditional_constant_descriptor::ConditionalConstantDescriptor;
pub use data_transfer_descriptor::{DataTransferDescriptor, DescriptorMemoryUnit};
pub use data_transfer_descriptor_container::{
    ChainFlag, ChainFlags, DataTransferDescriptorContainer, DescriptorId,
};
pub use discrete_integer_set_descriptor::DiscreteIntegerSetDescriptor;
pub use integer_sequence_descriptor::{IntegerSequenceDescriptor, SequenceSize};
pub use looping_chain_mutable_addr_descriptor::{ChainSize, LoopingChainMutableAddrDescriptor};
pub use simple_constant_descriptor::SimpleConstantDescriptor;
pub use toggle_descriptor::ToggleDescriptor;

/// THE CONSTANT BASE ADDRESSES ONE TRANSFER CAN USE — `using BaseAddrListTy =
/// SmallVector<const EvaluatedValue *, 1>` (`AddressPinningAndToggle.cpp:100`).
pub type BaseAddrList = Vec<EvaluatedValue>;

/// WHICH PATTERN A BASE ADDRESS FOLLOWS — the `DynamicPatternDescriptorBase *pattern_desc_`
/// hierarchy (`AddressPinningAndToggle.cpp:104-629`), whose six subclasses the reference
/// discriminates with `isa<>` and its `PatternKind` tag.
///
/// ⭐ AN ENUM BECAUSE THE SET IS CLOSED AND MUTUALLY EXCLUSIVE — the reference says so at `:858`
/// ("the descriptor types should be mutually exclusive"), which is why a non-looping chain gets no
/// arm here and lives on `HBMDataTransferDescriptor::total_chain_increment_` instead.
///
/// ⛔ NO `kUnknown` ARM: `PatternKind::kUnknown` (`:108`) is the base class's default tag and no
/// subclass constructs it — an unmatched transfer leaves `pattern_desc_` NULL, which is
/// [`DataTransferDescriptor::pattern_desc`]`== None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternDescriptor {
    /// `kSimpleConstant` — one constant.
    SimpleConstant(SimpleConstantDescriptor),
    /// `kToggle` — two constants alternating around a loop's iter arg.
    Toggle(ToggleDescriptor),
    /// `kConditionalConstant` — one of several constants, chosen by nested `sentient.if`s.
    ConditionalConstant(ConditionalConstantDescriptor),
    /// `kIntegerSequence` — `init + stride * i`.
    IntegerSequence(IntegerSequenceDescriptor),
    /// `kDiscreteIntegerSet` — independent increments from a chain of iter args.
    DiscreteIntegerSet(DiscreteIntegerSetDescriptor),
    /// `kLoopingChainMutableAddr` — the head of a looping chain of transfers.
    LoopingChainMutableAddr(LoopingChainMutableAddrDescriptor),
}

impl ToggleDescriptor {
    /// Replaces: e001_invalidate
    ///
    /// Clears exactly the three fields `ToggleDescriptor::isValid()` reads
    /// (`AddressPinningAndToggle.cpp:263`), so the toggle reads as unmatched.
    pub fn invalidate(&mut self) {
        self.outer_loop = None;
        self.iter_arg_index = None;
        self.c1 = None;
    }
}

impl ConditionalConstantDescriptor {
    /// Replaces: e002_getAllConstants
    ///
    /// APPENDS every constant this conditional can yield to `output`, in match order.
    ///
    /// ⛔ IT DOES NOT CLEAR `output` — the reference is `for_each(..., push_back)`, and its caller
    /// `initializeDescriptor` (`:2316`) owns whatever `base_addrs_` already holds.
    pub fn get_all_constants(&self, output: &mut BaseAddrList) {
        output.extend_from_slice(&self.yielded_constants);
    }
}

impl IntegerSequenceDescriptor {
    /// Replaces: e003_invalidate
    ///
    /// Clears everything `IntegerSequenceDescriptor::isValid()` reads (`:356`).
    ///
    /// ⛔ THE SIZE GOES TO [`SequenceSize::Cleared`], the reference's `size_ = 0` — NOT to
    /// [`SequenceSize::Symbolic`], which is the different `-1` meaning "length not known statically".
    pub fn invalidate(&mut self) {
        self.outer_loop = None;
        self.iter_arg_index = None;
        self.init = None;
        self.stride = None;
        self.size = SequenceSize::Cleared;
    }
}

impl DiscreteIntegerSetDescriptor {
    /// Replaces: e004_invalidate
    ///
    /// Clears all five fields, including BOTH delta totals — the reference's chained
    /// `init_ = total_positive_delta_ = total_negative_delta_ = nullptr`.
    pub fn invalidate(&mut self) {
        self.outer_loop = None;
        self.iter_arg_index = None;
        self.init = None;
        self.total_positive_delta = None;
        self.total_negative_delta = None;
    }
}

impl LoopingChainMutableAddrDescriptor {
    /// Replaces: e005_invalidate
    ///
    /// Clears the head flag and the outer loop, which is all `isValid()` (`:562`,
    /// `is_head_of_chain_ && size_ >= 1 && outer_loop_`) needs to go false.
    ///
    /// ⛔ DELIBERATELY LEAVES `size`, `iter_arg_index`, `init` AND `increment` AS IT FOUND THEM —
    /// unlike its four sibling `invalidate()`s. Restoring them would not change any observable
    /// result and is not what the reference does.
    pub fn invalidate(&mut self) {
        self.is_head_of_chain = false;
        self.outer_loop = None;
    }
}

// e006_dtor_DataTransferDescriptor IS PORTED AS THE OWNERSHIP OF THE FIELD IT FREES, and the filled
// anchor sits at that field: see `/// Replaces: e006_dtor_DataTransferDescriptor` on
// [`data_transfer_descriptor::DataTransferDescriptor::pattern_desc`]. Its whole body is
// `if (pattern_desc_) delete pattern_desc_;`, which an owned `Option<PatternDescriptor>` performs
// exactly and at the same point, so there is ⛔ no `impl Drop` here to write.

impl DataTransferDescriptorContainer {
    /// Replaces: e007_isPartOfSomeChain
    ///
    /// Whether `desc` belongs to some SSA chain — `kPartOfChain`, or false when it has no
    /// `chaining_info` entry at all.
    ///
    /// ⛔ THE KEY IS A [`DescriptorId`], not a borrow: the reference keys `chaining_info_` on the
    /// descriptor's ADDRESS (`:957`), which is not expressible while this container owns them.
    #[must_use]
    pub fn is_part_of_some_chain(&self, desc: DescriptorId) -> bool {
        self.chaining_info
            .get(&desc)
            .is_some_and(|flags| flags.get(ChainFlag::PartOfChain))
    }
}

impl DataTransferDescriptorContainer {
    /// Replaces: e008_setChainingInfo
    ///
    /// Sets or clears ONE chaining bit for `desc`, creating its entry only when setting.
    ///
    /// ⛔ CLEARING A BIT ON A DESCRIPTOR WITH NO ENTRY INSERTS NOTHING — the reference's `else if
    /// (it != chaining_info_.end())` guard. An unconditional `entry().or_default()` would make an
    /// absent descriptor indistinguishable from an all-clear one to `computeChainingInfo` (`:2192`).
    pub fn set_chaining_info(&mut self, desc: DescriptorId, flag: ChainFlag, val: bool) {
        match self.chaining_info.get_mut(&desc) {
            Some(flags) => flags.set(flag, val),
            None => {
                if val {
                    let mut flags = ChainFlags::default();
                    flags.set(flag, true);
                    self.chaining_info.insert(desc, flags);
                }
            }
        }
    }
}

/// `outer_loop_.getIterOperands()[iter_arg_index_]` — the initial value of ONE of a loop's carried
/// addresses, the three pattern constructors' (e280, e281, e282) shared first read of the loop their
/// walk resolved.
///
/// ⭐ REACHED THROUGH THE DEFINITIONS AND NOT A SCOPE SLICE: a [`ForRef`] names its loop by the
/// induction variable, which is block-argument position 0 of that loop's own body
/// ([`crate::islands::sentient::dialects::parent_for_arg`]), so the name alone finds the loop.
/// ⛔ `None` IS THE REFERENCE'S OUT-OF-RANGE READ, which is undefined there and which an index that
/// walk produced cannot reach; each caller treats it as the pattern not matching.
fn iter_operand_of(
    outer_loop: ForRef,
    iter_arg_index: IterArgIndex,
    defs: Definitions<'_>,
) -> Option<Val> {
    let (for_op, _) = defs.for_arg_of(outer_loop.0)?;
    let Op::Sentient(sentient::Op::For { carried, .. }) = for_op else {
        return None;
    };
    carried
        .get(iter_arg_index.0 as usize)
        .map(|entry| entry.init)
}

/// `outer_loop_.setIterOperand(iter_arg_index_, v)` — the WRITE half of [`iter_operand_of`], which
/// e490 and the toggle updaters (`:2006`, `:2036`) perform once the pinned base address is chosen.
///
/// ⭐ THE LOOP IS FOUND BY ITS INDUCTION VARIABLE at whatever depth, because that is what a
/// [`ForRef`] names; an index the walk produced is in range, and out of range writes nothing.
pub(super) fn set_iter_operand(
    body: &mut [Op],
    outer_loop: ForRef,
    iter_arg_index: IterArgIndex,
    val: Val,
) {
    for op in body {
        if let Op::Sentient(sentient::Op::For { iv, carried, .. }) = op
            && *iv == outer_loop.0
        {
            if let Some(entry) = carried.get_mut(iter_arg_index.0 as usize) {
                entry.init = val;
            }
            return;
        }
        for region in dialects::regions_mut(op) {
            set_iter_operand(region, outer_loop, iter_arg_index, val);
        }
    }
}

/// THE OP AT A POSITION — this module's copy of
/// [`address_register_precision_assignment`](crate::transform::sentient::address_register_precision_assignment)'s
/// walk, regions concatenated, which is the numbering [`OpId`] documents. `DataTransferDescriptor`
/// names its transfer by position, so every unit that has to READ that op comes through here.
pub(super) fn op_at<'a>(id: &OpId, scope: &'a [Op]) -> Option<&'a Op> {
    let (&ordinal, rest) = id.path().split_first()?;
    let op = scope.get(ordinal as usize)?;
    let Some(&next) = rest.first() else {
        return Some(op);
    };
    let mut base = 0usize;
    for region in dialects::regions_ref(op) {
        if (next as usize) < base + region.len() {
            let mut sub: Vec<u32> = rest.to_vec();
            sub[0] = (next as usize - base) as u32;
            return op_at(&OpId::at(&sub), region);
        }
        base += region.len();
    }
    None
}

/// Replaces: e009_createOffsetValue
///
/// The `sentient.scalar_constant` or `uniform.query_map` holding `ev`, typed like `mutable_addr_[0]`.
///
/// ⛔ WHICH OF THE TWO, AND WITH WHAT VALUE, IS `EvaluatedValue::buildOffsetValue`'s DECISION ALONE
/// (`Analyses/ExpressionEvaluatorUtils.cpp:148`, out of scope); all this adds is where the op goes,
/// which the campaign names droppable. ⭐ `DT_CHECK(parent_func)` needs no guard: a
/// `dataflow.program_unit` is a member of [`crate::islands::sentient::Program`].
#[must_use]
pub fn create_offset_value(_ev: EvaluatedValue, _ty: ScalarTy) -> Val {
    todo!(
        "EvaluatedValue::buildOffsetValue — out of campaign scope \
         (dcc/src/Transform/Sentient/Analyses/ExpressionEvaluatorUtils.cpp:148)"
    )
}

/// THE IMMUTABLE-ADDR OPERAND AS A PLACE — what `immutable_addr_.assign(v)` (`:1903`, `:1943`) writes
/// through, `mlir::MutableOperandRange` being a reference into the op the descriptor describes.
///
/// ⛔ THE THREE OPS `collectDataTransfers` ADMITS (`:1400-1402`) AND NO OTHER, which is why the stop
/// names that filter rather than `getMutableAndImmutableAddr`'s wider `DT_CHECK`.
pub(super) fn immutable_addr_mut(op: &mut Op, end: TransferEnd) -> &mut Val {
    match op {
        Op::Sentient(
            sentient::Op::LoadAndSend { immutable_addr, .. }
            | sentient::Op::ReceiveAndStore { immutable_addr, .. },
        ) => immutable_addr,
        Op::Sentient(sentient::Op::LoadAndStore {
            src_immutable_addr,
            dst_immutable_addr,
            ..
        }) => match end {
            TransferEnd::Src => src_immutable_addr,
            TransferEnd::Dst => dst_immutable_addr,
        },
        _ => todo!(
            "updateImmutableAddr: `immutable_addr_` on {op:?}, which \
             `isa<LoadAndSendOp, ReceiveAndStoreOp, LoadAndStoreOp>` rejects \
             (AddressPinningAndToggle.cpp:1400-1402)"
        ),
    }
}

/// THE MUTABLE-ADDR OPERAND AS A PLACE — the sibling of [`immutable_addr_mut`], which
/// `mutable_addr_.assign(v)` (`:1931`, `:2044`) writes through.
pub(super) fn mutable_addr_mut(op: &mut Op, end: TransferEnd) -> &mut Val {
    match op {
        Op::Sentient(
            sentient::Op::LoadAndSend { mutable_addr, .. }
            | sentient::Op::ReceiveAndStore { mutable_addr, .. },
        ) => mutable_addr,
        Op::Sentient(sentient::Op::LoadAndStore {
            src_mutable_addr,
            dst_mutable_addr,
            ..
        }) => match end {
            TransferEnd::Src => src_mutable_addr,
            TransferEnd::Dst => dst_mutable_addr,
        },
        _ => todo!(
            "updateConstantMutableAddr: `mutable_addr_` on {op:?}, which \
             `isa<LoadAndSendOp, ReceiveAndStoreOp, LoadAndStoreOp>` rejects \
             (AddressPinningAndToggle.cpp:1400-1402)"
        ),
    }
}

/// `SimpleConstantDataTransferUpdater` (`:1050`) — the updater for a transfer whose base address is
/// one constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimpleConstantDataTransferUpdater;

impl SimpleConstantDataTransferUpdater {
    /// Replaces: e010_updateVariableOffsetCalculation
    ///
    /// `return;  // nothing to do` (`:1065-1068`) — a simple constant has no toggle `scalar_sub` and no
    /// `sentient.if` whose yielded constants would need re-basing against the pinned address, so this
    /// override of the three that do is empty.
    pub const fn update_variable_offset_calculation(self, _new_immut_addr_ev: EvaluatedValue) {}

    /// Replaces: e418_getOffset
    ///
    /// `dtd_.getBaseAddr() - new_immut_addr_ev` BUILT AS A VALUE (`:1073-1079`), which is the
    /// `old_immutable - pinned_addr` term of `new_mutable = old_mutable + (old_immutable - pinned)`.
    ///
    /// ⛔ THE ONLY ONE OF THE FOUR `getOffset`s THAT MATERIALISES ANYTHING — e011/e012/e013 hand back
    /// a value the IR already computes, so this one alone needs the evaluator and the build sites.
    /// ⛔ `None` FROM [`DataTransferDescriptor::base_addr`] IS `DT_CHECK(base_addrs_.size() == 1)`.
    /// ⭐ `createOffsetValue` (e009) IS `buildOffsetValue` PLUS TWO BUILDER POSITIONS, which the
    /// campaign names droppable and which `sites`/`walked` carry instead — the same seam as e277.
    #[must_use]
    pub fn get_offset<E: ExpressionEvaluator>(
        self,
        dtd: &DataTransferDescriptor,
        new_immut_addr_ev: EvaluatedValue,
        ty: ScalarTy,
        evaluator: &mut E,
        sites: &mut OffsetSites<'_>,
        walked: &mut Vec<Op>,
    ) -> Val {
        let Some(base_addr) = dtd.base_addr() else {
            todo!(
                "SimpleConstantDataTransferUpdater::getOffset: DT_CHECK(base_addrs_.size() == 1) \
                 (`:657-660`) — {} base addrs",
                dtd.base_addrs.len()
            )
        };
        let offset = evaluator.evaluate_sub_handle(base_addr, new_immut_addr_ev);
        evaluator.build_offset_value_of(offset, sites, walked, ty)
    }
}

/// `SubOp toggle_sub_` — the `sentient.scalar_sub` computing a toggling transfer's immutable address,
/// named by its `$out`.
///
/// ⛔ A TYPE AND NOT A BARE [`Val`]: `DT_CHECK_MSG(toggle_sub, …)` (`:1529-1532`) is that
/// `immutable_addr[0]`'s DEFINING OP is a `scalar_sub`, and only a caller that matched
/// [`crate::islands::sentient::dialects::sentient::Op::ScalarSub`] can say so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToggleSub(Val);

impl ToggleSub {
    /// The sub named by the `$out` of a matched `sentient.scalar_sub`.
    #[must_use]
    pub const fn of(scalar_sub_result: Val) -> ToggleSub {
        ToggleSub(scalar_sub_result)
    }

    /// `toggle_sub_.getResult()`.
    #[must_use]
    pub const fn result(self) -> Val {
        self.0
    }

    /// `toggle_sub_.getInp1Mutable().assign(v)` (`:2003`, `:2041`) — the CONSTANT TERM operand of the
    /// toggle's own `scalar_sub`, which both toggle updaters re-point at the new `c1`.
    ///
    /// ⛔ `$inp1` AND NOT `$inp2`: the toggle is `X = c1 - Y` with `c1` first and the loop's iter arg
    /// second (`:2660-2662`), so writing `$inp2` would replace the carried half.
    /// ⛔ FALSE WHERE THE SUB IS NOT IN `scope` — `DT_CHECK_MSG(toggle_sub_, "expected valid
    /// toggle_sub")` (`:2002`) is the type's existence, but an op the caller cannot reach is a
    /// different failure and the caller names it.
    #[must_use]
    pub fn set_inp1(self, scope: &mut [Op], val: Val) -> bool {
        for op in scope.iter_mut() {
            if let Op::Sentient(sentient::Op::ScalarSub { lhs, result, .. }) = op
                && *result == self.0
            {
                *lhs = val;
                return true;
            }
            for region in dialects::regions_mut(op) {
                if self.set_inp1(region, val) {
                    return true;
                }
            }
        }
        false
    }
}

/// `ToggleDataTransferUpdater` (`:1082`) — the updater for a base address that toggles between two
/// constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToggleDataTransferUpdater {
    /// `toggle_sub_`, from `immutable_addr[0].get().getDefiningOp<SubOp>()` (`:1528`).
    pub toggle_sub: ToggleSub,
}

impl ToggleDataTransferUpdater {
    /// Replaces: e011_getOffset
    ///
    /// The toggle's own `scalar_sub` result IS the fall-back offset (`:1104-1107`) — nothing new is
    /// built, because the sub already computes the difference the pinning wants added.
    ///
    /// ⭐ `DT_CHECK(toggle_sub_)` is [`ToggleSub`]'s existence; `new_immut_addr_ev` is UNREAD.
    #[must_use]
    pub const fn get_offset(self, _new_immut_addr_ev: EvaluatedValue) -> Val {
        self.toggle_sub.result()
    }
}

/// WHICH OF A `sentient.if`'s RESULTS ORIGINALLY BOUND THE IMMUTABLE ADDRESS — `res_index_` (`:1140`).
///
/// ⛔ NOT AN `i32` DEFAULTING TO `-1`: `DT_CHECK(res_index_ >= 0)` guards both readers (`:1134`,
/// `:2070`), and `updateImmutableAddr` sets it from
/// `getIndexOfOperationResults(immutable_addr_[0].get())` (`:2053`) before `update()` reaches either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YieldedIndex(pub usize);

/// `if_op_.getResult(res_index_)` as a pair: `DT_CHECK(if_op_ && res_index_ >= 0)` is two facts about
/// one lookup, and carrying the value makes an out-of-range index inexpressible rather than caught. The
/// index stays because `updateVariableOffsetCalculation` (e277) needs it to reach every yield of that
/// result (`applyToAllYields(if_op_, …, res_index_)`, `:2071-2085`).
///
/// ⚠️ AND IT IS **NOT** `immutable_addr_[0]`: `updateImmutableAddr` REPLACES that operand (`:2060`)
/// after setting `res_index_` (`:2053`), so this is the pre-pinning value and the two differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConditionalConstResult {
    /// `res_index_`.
    pub index: YieldedIndex,
    /// `if_op_.getResult(res_index_)`.
    pub val: Val,
}

/// `ConditionalConstDataTransferUpdater` (`:1113`) — the updater for a base address an `scf`-style
/// conditional picks from a set of constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConditionalConstDataTransferUpdater {
    /// `if_op_` and `res_index_` together — see [`ConditionalConstResult`].
    pub if_result: ConditionalConstResult,
}

impl ConditionalConstDataTransferUpdater {
    /// Replaces: e012_getOffset
    ///
    /// The conditional's own selected result IS the fall-back offset (`:1133-1136`): once
    /// `updateVariableOffsetCalculation` has re-based every yielded constant against the pinned
    /// address, that result already carries the difference.
    ///
    /// ⭐ The two `DT_CHECK`s are [`ConditionalConstResult`]'s existence; `new_immut_addr_ev` is UNREAD.
    #[must_use]
    pub const fn get_offset(self, _new_immut_addr_ev: EvaluatedValue) -> Val {
        self.if_result.val
    }
}

/// `IntegerSequenceDataTransferUpdater` (`:1143`) — the updater for a base address that walks an
/// integer sequence through a loop's carried argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegerSequenceDataTransferUpdater {
    /// `iter_arg_` — the loop-carried BLOCK ARGUMENT the sequence arrives as, i.e.
    /// [`crate::islands::sentient::dialects::sentient::Carried::arg`] and never its `init` or
    /// `result`: substituting either hands back a value defined outside the loop.
    pub iter_arg: Val,
}

impl IntegerSequenceDataTransferUpdater {
    /// Replaces: e013_getOffset
    ///
    /// The carried iter arg IS the fall-back offset (`:1163-1166`) — the sequence's own induction
    /// already steps by what the pinning wants added.
    ///
    /// ⭐ `DT_CHECK(iter_arg_)` is the field's existence; `new_immut_addr_ev` is UNREAD.
    #[must_use]
    pub const fn get_offset(self, _new_immut_addr_ev: EvaluatedValue) -> Val {
        self.iter_arg
    }
}

/// `raw_ostream &operator<<(raw_ostream &, const EvaluatedValue &)`
/// (`Analyses/ExpressionEvaluatorUtils.h:405`, defined `.cpp:205-232`) — `kind(..), baseValue(..),
/// offsetValue(..)`, every field of it read off the arena entry an [`EvaluatedValue`] only NAMES.
///
/// ⛔ OUT OF CAMPAIGN SCOPE, AND EVERY `dump()` IN THIS PASS ENDS HERE. Each of them writes its own
/// literal text and stops at its first evaluated value, exactly as
/// [`Sequence::print`](crate::transform::sentient::cfg_simplification_sentient_level) does for the
/// same `operator<<`. ⛔ Do not render a handle from its number: that number is which arena entry,
/// not what the entry says.
///
/// ⛔ `None` IS THE REFERENCE DEREFERENCING A NULL `const EvaluatedValue *` — `DiscreteIntegerSet`'s
/// `isValid()` (`:468`) does not cover its own value fields, so `dump` can reach one. It is a stop
/// either way, and the message says which it was.
pub(crate) fn write_evaluated_value(ev: Option<EvaluatedValue>, _out: &mut String) {
    todo!(
        "EvaluatedValue::operator<< (Analyses/ExpressionEvaluatorUtils.h:405) — out of campaign \
         scope, asked for {ev:?}"
    )
}

/// WHAT e014 NEEDS OF A DESCRIPTOR — `DataTransferDescriptor::dump` is **e492** (`:2521`, level 3,
/// `data_transfer_descriptor.rs`), a later unit.
///
/// ⚠️ `UNITS.tsv` RECORDS e014's CALLS AS `-`: the call resolves through a virtual `dump()` the
/// extractor's detector did not follow.
///
/// ⛔ `e492_dump` DOES NOT IMPLEMENT THIS AND CANNOT: `os << op_` needs the body its [`OpId`] names
/// and its integer-sequence arm needs the evaluator `getAllConstants` asks, neither of which a bare
/// `&self` carries — see [`DataTransferDescriptor::dump`]. Whichever unit ports e014's callers
/// (`:1350-1354`, `:1371-1375`) has both in hand and binds them here.
pub trait DumpDescriptor {
    /// `dtd->dump()` (`:1669`) — one descriptor's block, `----------` delimited.
    fn dump(&self) -> String;
}

/// Replaces: e014_dump
///
/// The immutable descriptors under one header then the mutable ones under another, each container in
/// INSERTION order (`DataTransferDescriptorContainer` exposes the base `std::vector`'s `begin`/`end`,
/// not `sorted_list_`).
///
/// ⭐ THE INNER `LLVM_DEBUG` ON THE TWO HEADERS CHANGES NOTHING: both callsites already wrap the whole
/// call (`:1350-1354`, `:1371-1375`), so a returned `String` loses no gating the reference had.
#[must_use]
pub fn dump(immut: &[&dyn DumpDescriptor], mutable: &[&dyn DumpDescriptor]) -> String {
    let mut out = String::from("Immutable transfer descriptors:\n");
    for dtd in immut {
        out.push_str(&dtd.dump());
    }
    out.push_str("Mutable transfer descriptors:\n");
    for dtd in mutable {
        out.push_str(&dtd.dump());
    }
    out
}

impl DataTransferDescriptorContainer {
    /// Replaces: e273_isHeadOfChain
    ///
    /// Whether `desc` heads its chain — `kHeadOfChain`, and only for a descriptor that is in a chain
    /// at all.
    ///
    /// ⭐ THE `isPartOfSomeChain` GUARD CANNOT FIRE AND IS KEPT ANYWAY: `computeChainingInfo` sets
    /// `kHeadOfChain` only alongside `kPartOfChain` (`:2231-2232`, `:2245-2246`), so nothing in the
    /// reference makes the two bits observably independent — but the reference asks, and a later
    /// writer could part them.
    #[must_use]
    pub fn is_head_of_chain(&self, desc: DescriptorId) -> bool {
        self.is_part_of_some_chain(desc)
            && self
                .chaining_info
                .get(&desc)
                .is_some_and(|flags| flags.get(ChainFlag::HeadOfChain))
    }
}

impl DataTransferDescriptorContainer {
    /// Replaces: e274_validate
    ///
    /// ⭐ THE `DT_CHECK_MSG` IS GONE, NOT SKIPPED — *"parallel list is out of sync with the main
    /// list"* (`:930-931`) compares `sorted_list_.size()` against `size()` (the `e252_size` the
    /// anchor names), and this container has no `sorted_list_` at all
    /// ([`DataTransferDescriptorContainer`] says why): one `Vec`, one length, nothing to fall out of
    /// sync with. Same shape as e252_size itself (`utils/units_and_their_values.rs`).
    pub const fn validate(&self) {}
}

impl DataTransferDescriptorContainer {
    /// Replaces: e416_clear_all
    ///
    /// Empties the container — every descriptor and every chaining bit (`:886-891`).
    ///
    /// ⛔ `sorted_list_.clear()` HAS NOTHING TO CLEAR: there is no such field here, see the type.
    /// ⭐ AND IT FREES THE DESCRIPTORS, WHICH THE REFERENCE'S `clear_all` DOES NOT — its only two
    /// callers are `~DataTransferDescriptorContainer()` (`:881`) and `cleanup()` (e489, `:1656-1664`),
    /// and the latter deletes every element first and then clears. An owning `Vec`'s clear is that
    /// pair, minus the delete loop; the destructor's own leak is not a behaviour to port.
    pub fn clear_all(&mut self) {
        self.validate();
        self.descriptors.clear();
        self.chaining_info.clear();
    }
}

impl DataTransferDescriptorContainer {
    /// Replaces: e417_isHeadOfLoopingChain
    ///
    /// Whether `desc` heads a chain whose last link reaches back to it (`:920-926`).
    ///
    /// ⭐ THE `isHeadOfChain` GUARD DOES FIRE, UNLIKE e273'S: `computeChainingInfo`'s second loop runs
    /// over EVERY valid descriptor and sets `kHeadOfLoopingChain` on each one whose mutable-address
    /// result is yielded by its own enclosing `sentient.for` (`:2251-2278`) — which the tail link of a
    /// two-link chain is, holding `kPartOfChain` with `kHeadOfChain` explicitly cleared (`:2219`).
    #[must_use]
    pub fn is_head_of_looping_chain(&self, desc: DescriptorId) -> bool {
        self.is_head_of_chain(desc)
            && self
                .chaining_info
                .get(&desc)
                .is_some_and(|flags| flags.get(ChainFlag::HeadOfLoopingChain))
    }
}

/// WHICH END OF A `sentient.load_and_store` SITS ON THE HBM — the reference's `if
/// (getUnitType(src_unit) == HBM) .. else if (getUnitType(dst_unit) == HBM)`, whose `else if` gives
/// the source the win when both ends are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HbmEnd {
    /// `getSrcImmutableAddr()`, assigned through `getSrcImmutableAddrMutable()`.
    Src,
    /// `getDstImmutableAddr()`, assigned through `getDstImmutableAddrMutable()`.
    Dst,
}

/// ONE CONSTANT IMMUTABLE ADDRESS ON AN HBM END, RECORDED BEFORE ANYTHING IS WRITTEN.
///
/// ⭐ TWO PHASES BECAUSE THE LOOKUP RUNS OUTWARDS AND THE WRITE RUNS INWARDS: `getDefiningOp()` needs
/// the enclosing scopes borrowed while the walk sits inside the very region it is about to rewrite.
/// The op is named by its first result, which is its SSA identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConstantHbmAddr {
    /// `results.0` of the `sentient.load_and_store`.
    transfer: Val,
    /// Which end the HBM is, and so which address is rewritten.
    end: HbmEnd,
    /// `const_immut_addr.getValue()`.
    value: i64,
}

/// `dcc::getUnitType(val.getDefiningOp()) == HBM` (`dcc/src/Utils/DccExtContext.cpp:191-206`), for the
/// two arms a transfer end can take.
///
/// ⚠️ DIVERGENCE ON A BLOCK ARGUMENT: `DT_CHECK_MSG(op, "expected valid op")` (`:192`) aborts where
/// this answers `false`. And the reference's fourth arm — an op that is none of the three — answers
/// with the ENCLOSING `dataflow.program_unit`'s own type, which is never the HBM.
fn is_hbm(val: Val, defs: Definitions<'_>) -> bool {
    unit_type_of(val, defs) == Some(DfirUnit::Hbm)
}

/// `dcc::getUnitType(val.getDefiningOp())` ITSELF (`dcc/src/Utils/DccExtContext.cpp:191-206`) — which
/// unit the value lives on, `None` where the reference aborts or answers with the enclosing
/// `dataflow.program_unit`. [`is_hbm`] is the `== HBM` of it, and
/// `getMutableAddrResultIndex`'s `src_unit == mem_unit` (`Analyses/Utils.cpp:544-549`) is the general
/// question.
pub(super) fn unit_type_of(val: Val, defs: Definitions<'_>) -> Option<DfirUnit> {
    match defs.of(val) {
        Some(Op::Dataflow(dataflow::Op::GetUnit { unit, .. })) => Some(*unit),
        // `getUnitType(QueryMapOp)`: every QUERIED VALUE must be a `get_unit`, and
        // `DT_CHECK_MSG(type == unit_type, ..)` says they all agree — so the mapping's one agreed
        // unit is the answer, and the empty mapping its `DT_CHECK_MSG(!units.empty(), ..)` rejects
        // has none.
        Some(Op::Uniform(uniform::Op::QueryMap { map, .. })) => {
            let Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) = defs.of(*map)
            else {
                return None;
            };
            let mut units = pairs.iter().map(|(_, value)| match defs.of(*value) {
                Some(Op::Dataflow(dataflow::Op::GetUnit { unit, .. })) => Some(*unit),
                _ => None,
            });
            let first = units.next().flatten()?;
            units.all(|unit| unit == Some(first)).then_some(first)
        }
        _ => None,
    }
}

/// `curr_region.walk<WalkOrder::PreOrder>([&](sentient::LoadAndStoreOp) { .. })`, recording what it
/// would rewrite — the op before its own regions, and `scopes` grown one level per descent so
/// `getDefiningOp()` still searches outwards.
fn plan_constant_hbm_addrs(body: &[Op], enclosing: &[&[Op]], out: &mut Vec<ConstantHbmAddr>) {
    let mut scopes: Vec<&[Op]> = Vec::with_capacity(enclosing.len() + 1);
    scopes.push(body);
    scopes.extend_from_slice(enclosing);
    let defs = Definitions::from_innermost(&scopes);
    for op in body {
        if let Op::Sentient(sentient::Op::LoadAndStore {
            src,
            dst,
            src_immutable_addr,
            dst_immutable_addr,
            results,
            ..
        }) = op
        {
            let end = if is_hbm(*src, defs) {
                Some((HbmEnd::Src, *src_immutable_addr))
            } else if is_hbm(*dst, defs) {
                Some((HbmEnd::Dst, *dst_immutable_addr))
            } else {
                None
            };
            // `dyn_cast<sentient::ConstantOp>(..getDefiningOp())` — anything else is left alone.
            if let Some((end, addr)) = end
                && let Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) =
                    defs.of(addr)
            {
                out.push(ConstantHbmAddr {
                    transfer: results.0,
                    end,
                    value: *value,
                });
            }
        }
        for region in dialects::regions_ref(op) {
            plan_constant_hbm_addrs(region, &scopes, out);
        }
    }
}

/// `load_and_store.getSrcImmutableAddrMutable().assign(new_immut_addr)` — the one operand the walk
/// came back to change, found again by the result that named it.
fn assign_immutable_addr(body: &mut [Op], transfer: Val, end: HbmEnd, addr: Val) {
    for op in body {
        if let Op::Sentient(sentient::Op::LoadAndStore {
            src_immutable_addr,
            dst_immutable_addr,
            results,
            ..
        }) = op
            && results.0 == transfer
        {
            match end {
                HbmEnd::Src => *src_immutable_addr = addr,
                HbmEnd::Dst => *dst_immutable_addr = addr,
            }
            return;
        }
        for region in dialects::regions_mut(op) {
            assign_immutable_addr(region, transfer, end, addr);
        }
    }
}

/// Replaces: e275_turnHBMConstantOpAddrsToQueryMapsHelper
///
/// Turns every constant HBM-end immutable address inside `region_op`'s regions into a
/// `uniform.query_map` mapping that region's whole unit list to one copy each of the constant, and
/// hands the new `sentient.scalar_constant`s back for the caller's `const_builder` to place — e419
/// puts them at the front of [`crate::islands::sentient::Program::preamble`].
///
/// ⚠️ DIVERGENCE ON AN EMPTY UNIT LIST: `createQueryMapFromOperationsWithOneResults` returns
/// `nullptr` and the reference `assign`s it (`dcc/src/Dialect/Uniform/Utils.cpp:359`), leaving the
/// transfer with no immutable address at all; here the address is left as it was.
#[must_use]
pub fn turn_hbm_constant_op_addrs_to_query_maps_helper(
    region_op: &mut UniformRegions,
    enclosing: &[&[Op]],
    values: &mut Values,
) -> Vec<Op> {
    let mut constants: Vec<Op> = Vec::new();
    for region_idx in 0..region_op.regions().len() {
        // `getRegionArg(region_idx)` and `collectUnitVals(getRegionUnitList(region_idx))`, which BOTH
        // arms of the reference's `dyn_cast` chain read identically — and the enum has no third arm,
        // so the unwritten `else` that would leave `key` NULL is unreachable here.
        let (key, units, planned) = {
            let region = &region_op.regions()[region_idx];
            let mut planned = Vec::new();
            plan_constant_hbm_addrs(&region.body, enclosing, &mut planned);
            let units =
                dialects::collect_unit_ops(&region.units, Definitions::from_innermost(enclosing));
            (region.arg, units, planned)
        };
        if units.is_empty() {
            continue;
        }
        let body = &mut region_op.regions_mut()[region_idx].body;
        // `OpBuilder query_map_builder(&curr_region)` inserts at the region's start and then advances
        // past what it built, so successive pairs land in creation order ahead of their readers.
        let mut at = 0;
        for planned in planned {
            // `for (auto unit_op : units) const_list.push_back(const_immut_addr.getValue());` then
            // one `sentient::ConstantOp` per entry — ⭐ `DT_CHECK(keys.size() == const_vals.size())`
            // (`Utils.cpp:371`) IS GONE, NOT SKIPPED: the two lists are built from one iteration.
            let mut pairs = Vec::with_capacity(units.len());
            for &unit in &units {
                let result = values.mint();
                constants.push(Op::Sentient(sentient::Op::ScalarConstant {
                    value: planned.value,
                    result,
                    reg_locale: sentient::RegType::Imm,
                    ty: ScalarTy::Index,
                    is_symbol: false,
                }));
                pairs.push((unit, result));
            }
            let map = values.mint();
            let queried = values.mint();
            body.insert(
                at,
                Op::Uniform(uniform::Op::DefImmutableMapping { result: map, pairs }),
            );
            body.insert(
                at + 1,
                Op::Uniform(uniform::Op::QueryMap {
                    result: queried,
                    map,
                    key,
                }),
            );
            at += 2;
            assign_immutable_addr(body, planned.transfer, planned.end, queried);
        }
    }
    constants
}

/// `unit->walk<WalkOrder::PreOrder>` — each op before its own regions, `scopes` grown one level per
/// descent so `getDefiningOp()` still searches outwards, and `WalkResult::skip()` on a region op e275
/// has just rewritten. ⭐ THE BLOCK IS SNAPSHOT BEFORE THE DESCENT because the walk holds it
/// exclusively while the lookups inside want it shared.
fn query_map_uniform_regions(
    body: &mut [Op],
    enclosing: &[&[Op]],
    values: &mut Values,
    constants: &mut Vec<Op>,
) {
    let here = body.to_vec();
    let mut scopes: Vec<&[Op]> = Vec::with_capacity(enclosing.len() + 1);
    scopes.push(&here);
    scopes.extend_from_slice(enclosing);
    for op in body.iter_mut() {
        if let Op::UniformRegions(region_op) = op {
            constants.extend(turn_hbm_constant_op_addrs_to_query_maps_helper(
                region_op, &scopes, values,
            ));
            continue;
        }
        for region in dialects::regions_mut(op) {
            query_map_uniform_regions(region, &scopes, values, constants);
        }
    }
}

/// Replaces: e419_turnHBMConstantOpAddrsToQueryMapsInUniformRegions
///
/// Hands every `uniform.uniformize_regions`/`uniform.equalize_pattern` in one program unit to e275,
/// skipping its subtree, and puts the constants it built at the front of the enclosing function's
/// entry block — `OpBuilder const_builder(parent_func.getBody())`, ONE builder for the whole walk, so
/// they land in creation order ahead of every reader (`:1579-1591`).
///
/// ⭐ `DT_CHECK(parent_func)` IS THE ISLAND'S SHAPE: a program unit is a member of a
/// [`crate::islands::sentient::Program`], whose `preamble` is that entry block.
/// ⛔ [`Op::Uniform`] NEEDS NO ARM OF ITS OWN: it carries no regions at this rung, so it can hold no
/// `sentient.load_and_store` and descending into it rather than skipping is unobservable.
pub fn turn_hbm_constant_op_addrs_to_query_maps_in_uniform_regions(
    unit_body: &mut [Op],
    preamble: &mut Vec<Op>,
    values: &mut Values,
) {
    let scope = preamble.clone();
    let mut constants = Vec::new();
    let enclosing: [&[Op]; 1] = [&scope];
    query_map_uniform_regions(unit_body, &enclosing, values, &mut constants);
    preamble.splice(0..0, constants);
}

// THE FIVE DESCRIPTORS' OWN `isValid()` — `:263`, `:318`, `:356`, `:468`, `:562`, each one to three
// lines reading its own fields, and each on `EXCLUSIONS.tsv` as "a struct field in Rust, not a
// function". ⛔ NONE OF THEM IS `DataTransferDescriptor::isValid()`, which is e278
// (`data_transfer_descriptor.rs`): that one first requires a stored base address and then defers to
// exactly the arm below that matches its `pattern_desc_`.

impl ToggleDescriptor {
    /// `isValid()` (`:263`) — `c1_ && iter_arg_index_ >= 0 && outer_loop_`, i.e. the three fields
    /// [`ToggleDescriptor::invalidate`] clears being present.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.c1.is_some() && self.iter_arg_index.is_some() && self.outer_loop.is_some()
    }
}

impl ConditionalConstantDescriptor {
    /// `isValid()` (`:318`) — the conditional yielded at least one constant.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.yielded_constants.is_empty()
    }
}

impl IntegerSequenceDescriptor {
    /// `isValid()` (`:356`) — `iter_arg_index_ >= 0 && outer_loop_ && size_ > 0`.
    ///
    /// ⛔ [`SequenceSize::Symbolic`] IS NOT A LENGTH: it is the reference's `size_ == -1`, which fails
    /// `size_ > 0` exactly as [`SequenceSize::Cleared`]'s `0` does.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.iter_arg_index.is_some()
            && self.outer_loop.is_some()
            && matches!(self.size, SequenceSize::Terms(terms) if terms > 0)
    }
}

impl DiscreteIntegerSetDescriptor {
    /// `isValid()` (`:468`) — `iter_arg_index_ >= 0 && outer_loop_`; ⛔ the two delta totals are NOT
    /// part of it, unlike its four siblings' value fields.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.iter_arg_index.is_some() && self.outer_loop.is_some()
    }
}

impl LoopingChainMutableAddrDescriptor {
    /// `isValid()` (`:562`) — `is_head_of_chain_ && size_ >= 1 && outer_loop_`.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.is_head_of_chain && self.size.0 >= 1 && self.outer_loop.is_some()
    }
}

impl DataTransferDescriptor {
    /// Replaces: e257_getBaseAddr
    ///
    /// THE ONE constant base address this transfer resolved to — `*base_addrs_.back()` under
    /// `DT_CHECK(base_addrs_.size() == 1)`.
    ///
    /// ⛔ `None` IS THAT `DT_CHECK`, AND IT IS NOT "no base address": a matched toggle stores TWO and
    /// this accessor is the wrong one for it — a pair is read through [`Self::base_addrs`], which is
    /// `getBaseAddrList()` (`:661`, an `EXCLUSIONS.tsv` field accessor).
    #[must_use]
    pub fn base_addr(&self) -> Option<EvaluatedValue> {
        match self.base_addrs.as_slice() {
            [only] => Some(*only),
            _ => None,
        }
    }

    /// Replaces: e415_isSimpleConstant
    ///
    /// Whether this transfer's base address is ONE constant (`:673-676`).
    ///
    /// ⛔ THE `isValid()` CONJUNCT IS SPECIALISED, NOT DROPPED, exactly as in [`Self::is_toggle`]:
    /// with the arm destructured, `isValid() && isa<SimpleConstantDescriptor>` is e278's TAIL arm
    /// (`:2499`) — the reference's `dyn_cast` chain has no `SimpleConstant` case and this
    /// descriptor is never invalid (`:193`) — so exactly one stored base address is required.
    #[must_use]
    pub fn is_simple_constant(&self) -> bool {
        matches!(
            self.pattern_desc,
            Some(PatternDescriptor::SimpleConstant(_))
        ) && self.base_addrs.len() == 1
    }

    /// Replaces: e258_isToggle
    ///
    /// Whether this transfer's base address is a MATCHED toggle.
    ///
    /// ⛔ THE `isValid()` CONJUNCT IS SPECIALISED, NOT DROPPED: with the arm already destructured,
    /// `isValid() && isa<Toggle>` is e278's toggle arm alone (`:2482-2485`), so an `invalidate()`d
    /// toggle answers false, and so does a live one left holding one base address it never simplified
    /// to. ⭐ `base_addrs.empty()` (`:2480`) needs no line here: neither size this arm accepts is 0.
    #[must_use]
    pub fn is_toggle(&self) -> bool {
        let Some(PatternDescriptor::Toggle(toggle)) = &self.pattern_desc else {
            return false;
        };
        toggle.is_valid()
            && ((toggle.can_be_simplified && self.base_addrs.len() == 1)
                || self.base_addrs.len() == 2)
    }

    /// Replaces: e259_isConditionalConstant
    ///
    /// Whether this transfer's base address is a MATCHED set of conditionally yielded constants.
    ///
    /// ⛔ THE EMPTY-LIST GUARD IS THE TRANSFER'S, NOT THE DESCRIPTOR'S: `isValid()` refuses a
    /// transfer with no stored base address before it ever reaches this arm (`:2480`), and the arm
    /// itself imposes no count (`:2486-2488`).
    #[must_use]
    pub fn is_conditional_constant(&self) -> bool {
        let Some(PatternDescriptor::ConditionalConstant(cond_const)) = &self.pattern_desc else {
            return false;
        };
        !self.base_addrs.is_empty() && cond_const.is_valid()
    }

    /// Replaces: e260_isIntegerSequence
    ///
    /// Whether this transfer's base address is a MATCHED `init + stride * i` sequence (`:2489-2491`).
    #[must_use]
    pub fn is_integer_sequence(&self) -> bool {
        let Some(PatternDescriptor::IntegerSequence(int_seq)) = &self.pattern_desc else {
            return false;
        };
        !self.base_addrs.is_empty() && int_seq.is_valid()
    }

    /// Replaces: e261_isDiscreteIntegerSet
    ///
    /// Whether this transfer's base address is a MATCHED set of independent per-loop increments
    /// (`:2492-2494`).
    #[must_use]
    pub fn is_discrete_integer_set(&self) -> bool {
        let Some(PatternDescriptor::DiscreteIntegerSet(int_set)) = &self.pattern_desc else {
            return false;
        };
        !self.base_addrs.is_empty() && int_set.is_valid()
    }

    /// Replaces: e262_isLoopingChainMutableAddr
    ///
    /// Whether this transfer is the MATCHED head of a looping chain (`:2495-2497`).
    #[must_use]
    pub fn is_looping_chain_mutable_addr(&self) -> bool {
        let Some(PatternDescriptor::LoopingChainMutableAddr(chain)) = &self.pattern_desc else {
            return false;
        };
        !self.base_addrs.is_empty() && chain.is_valid()
    }

    /// Replaces: e263_getToggleDescriptor
    ///
    /// The toggle to MUTATE — the reference's non-const overload, which is what
    /// `ToggleDescriptor::invalidate` and the pinning updates are reached through.
    ///
    /// ⛔ `None` IS `DT_CHECK(isToggle())` (`:707`), so a descriptor that stopped being a valid toggle
    /// hands back nothing rather than a live reference to an invalidated pattern.
    #[must_use]
    pub fn toggle_descriptor_mut(&mut self) -> Option<&mut ToggleDescriptor> {
        if !self.is_toggle() {
            return None;
        }
        match &mut self.pattern_desc {
            Some(PatternDescriptor::Toggle(toggle)) => Some(toggle),
            _ => None,
        }
    }

    /// Replaces: e264_getToggleDescriptor
    ///
    /// The toggle to READ — the reference's `const` overload, a separate unit because the two differ
    /// in exactly the mutability Rust makes the caller ask for.
    ///
    /// ⛔ `None` IS `DT_CHECK(isToggle())` (`:711`), NOT "not a toggle": a `Toggle` arm that fails
    /// [`Self::is_toggle`] is present and refused, which is the whole point of the check.
    #[must_use]
    pub fn toggle_descriptor(&self) -> Option<&ToggleDescriptor> {
        match &self.pattern_desc {
            Some(PatternDescriptor::Toggle(toggle)) if self.is_toggle() => Some(toggle),
            _ => None,
        }
    }
}

// ⭐ THE EIGHT PATTERN GETTERS ARE ONE `match` ON [`DataTransferDescriptor::pattern_desc`]: the
// reference's `cast<T>(pattern_desc_)` IS the variant, and the `pattern_desc_ != nullptr` conjunct
// of each `DT_CHECK(isX())` IS the `Option`. The `panic!` IS the `DT_CHECK` — the reference aborts
// here, and `ConditionalConstDataTransferUpdater::updateImmutableAddr` (`:2050`) calls the getter
// with no `isConditionalConstant()` of its own, so this is a live abort and not a precondition some
// caller has already discharged.
// ⛔ THE `isValid()` CONJUNCT OF EACH `DT_CHECK` IS NOT HERE: `DataTransferDescriptor::isValid()`
// (e278, `:2478`) is the SCC edge the scheduler cut out of this batch, so a variant-matching but
// INVALID descriptor is RETURNED where the reference aborts. Each guard gains `self.is_valid() &&`
// when e278 lands; until then nothing silently accepts one, because every in-tree caller re-tests
// validity itself (`DT_CHECK_MSG(cc.isValid(), "descriptor may be corrupt")`, `:2051`).
impl DataTransferDescriptor {
    /// Replaces: e265_getConditionalConstantDescriptor
    ///
    /// The pattern as a mutable [`ConditionalConstantDescriptor`] (`:714-717`).
    #[must_use]
    pub fn conditional_constant_descriptor_mut(&mut self) -> &mut ConditionalConstantDescriptor {
        match &mut self.pattern_desc {
            Some(PatternDescriptor::ConditionalConstant(cc)) => cc,
            other => panic!(
                "DT_CHECK(isConditionalConstant()) (`AddressPinningAndToggle.cpp:715`): {other:?}"
            ),
        }
    }

    /// Replaces: e266_getConditionalConstantDescriptor
    ///
    /// The pattern as a shared [`ConditionalConstantDescriptor`] (`:718-722`).
    #[must_use]
    pub fn conditional_constant_descriptor(&self) -> &ConditionalConstantDescriptor {
        match &self.pattern_desc {
            Some(PatternDescriptor::ConditionalConstant(cc)) => cc,
            other => panic!(
                "DT_CHECK(isConditionalConstant()) (`AddressPinningAndToggle.cpp:720`): {other:?}"
            ),
        }
    }

    /// Replaces: e267_getIntegerSequenceDescriptor
    ///
    /// The pattern as a mutable [`IntegerSequenceDescriptor`] (`:723-726`).
    #[must_use]
    pub fn integer_sequence_descriptor_mut(&mut self) -> &mut IntegerSequenceDescriptor {
        match &mut self.pattern_desc {
            Some(PatternDescriptor::IntegerSequence(isq)) => isq,
            other => panic!(
                "DT_CHECK(isIntegerSequence()) (`AddressPinningAndToggle.cpp:724`): {other:?}"
            ),
        }
    }

    /// Replaces: e268_getIntegerSequenceDescriptor
    ///
    /// The pattern as a shared [`IntegerSequenceDescriptor`] (`:727-730`).
    #[must_use]
    pub fn integer_sequence_descriptor(&self) -> &IntegerSequenceDescriptor {
        match &self.pattern_desc {
            Some(PatternDescriptor::IntegerSequence(isq)) => isq,
            other => panic!(
                "DT_CHECK(isIntegerSequence()) (`AddressPinningAndToggle.cpp:728`): {other:?}"
            ),
        }
    }

    /// Replaces: e269_getDiscreteIntegerSetDescriptor
    ///
    /// The pattern as a mutable [`DiscreteIntegerSetDescriptor`] (`:731-734`).
    #[must_use]
    pub fn discrete_integer_set_descriptor_mut(&mut self) -> &mut DiscreteIntegerSetDescriptor {
        match &mut self.pattern_desc {
            Some(PatternDescriptor::DiscreteIntegerSet(dis)) => dis,
            other => panic!(
                "DT_CHECK(isDiscreteIntegerSet()) (`AddressPinningAndToggle.cpp:732`): {other:?}"
            ),
        }
    }

    /// Replaces: e270_getDiscreteIntegerSetDescriptor
    ///
    /// The pattern as a shared [`DiscreteIntegerSetDescriptor`] (`:735-738`).
    #[must_use]
    pub fn discrete_integer_set_descriptor(&self) -> &DiscreteIntegerSetDescriptor {
        match &self.pattern_desc {
            Some(PatternDescriptor::DiscreteIntegerSet(dis)) => dis,
            other => panic!(
                "DT_CHECK(isDiscreteIntegerSet()) (`AddressPinningAndToggle.cpp:736`): {other:?}"
            ),
        }
    }

    /// Replaces: e271_getLoopingChainMutableAddrDescriptor
    ///
    /// The pattern as a mutable [`LoopingChainMutableAddrDescriptor`] (`:739-742`).
    #[must_use]
    pub fn looping_chain_mutable_addr_descriptor_mut(
        &mut self,
    ) -> &mut LoopingChainMutableAddrDescriptor {
        match &mut self.pattern_desc {
            Some(PatternDescriptor::LoopingChainMutableAddr(lcma)) => lcma,
            other => panic!(
                "DT_CHECK(isLoopingChainMutableAddr()) (`AddressPinningAndToggle.cpp:740`): \
                 {other:?}"
            ),
        }
    }

    /// Replaces: e272_getLoopingChainMutableAddrDescriptor
    ///
    /// The pattern as a shared [`LoopingChainMutableAddrDescriptor`] (`:743-747`).
    #[must_use]
    pub fn looping_chain_mutable_addr_descriptor(&self) -> &LoopingChainMutableAddrDescriptor {
        match &self.pattern_desc {
            Some(PatternDescriptor::LoopingChainMutableAddr(lcma)) => lcma,
            other => panic!(
                "DT_CHECK(isLoopingChainMutableAddr()) (`AddressPinningAndToggle.cpp:745`): \
                 {other:?}"
            ),
        }
    }

    /// Replaces: e486_getSimpleConstantDescriptor
    ///
    /// The pattern as a mutable [`SimpleConstantDescriptor`] (`:698-701`).
    #[must_use]
    pub fn simple_constant_descriptor_mut(&mut self) -> &mut SimpleConstantDescriptor {
        match &mut self.pattern_desc {
            Some(PatternDescriptor::SimpleConstant(sc)) => sc,
            other => {
                panic!("DT_CHECK(isSimpleConstant()) (`AddressPinningAndToggle.cpp:699`): {other:?}")
            }
        }
    }

    /// Replaces: e487_getSimpleConstantDescriptor
    ///
    /// The pattern as a shared [`SimpleConstantDescriptor`] (`:702-705`).
    #[must_use]
    pub fn simple_constant_descriptor(&self) -> &SimpleConstantDescriptor {
        match &self.pattern_desc {
            Some(PatternDescriptor::SimpleConstant(sc)) => sc,
            other => {
                panic!("DT_CHECK(isSimpleConstant()) (`AddressPinningAndToggle.cpp:703`): {other:?}")
            }
        }
    }
}

impl SimpleConstantDescriptor {
    /// Replaces: e399_getMin
    ///
    /// The one constant a simple-constant base address is (`:182-185`).
    ///
    /// ⭐ NO `DT_CHECK(isValid())` GUARD TO WRITE: "Descriptor is always valid once constructed"
    /// (`:192-193`), so unlike its four siblings below this getter is total.
    #[must_use]
    pub const fn min(&self) -> EvaluatedValue {
        self.ev
    }

    /// Replaces: e400_getMax
    ///
    /// The SAME one constant (`:187-190`) — a single-valued base address's range is a point.
    ///
    /// ⛔ NOT A SLIP TO BE TIDIED INTO ONE ACCESSOR: `findClosestPinnedAddr` (`:2057`, `:2123`) is
    /// handed both ends of every pattern, and for this pattern both ends are `ev_`.
    #[must_use]
    pub const fn max(&self) -> EvaluatedValue {
        self.ev
    }
}

impl ToggleDescriptor {
    /// Replaces: e401_getC1
    ///
    /// The constant term `c1` in `X = c1 - Y` (`:246-249`).
    ///
    /// ⛔ `None` IS `DT_CHECK(isValid())` AND NOT MERELY AN ABSENT `c1_`: the check reads all three
    /// fields (`:263`), so a toggle that kept its `c1` but lost its loop answers nothing here — the
    /// same encoding as [`DataTransferDescriptor::base_addr`].
    #[must_use]
    pub fn c1(&self) -> Option<EvaluatedValue> {
        if self.is_valid() { self.c1 } else { None }
    }

    /// Replaces: e402_getIterArgIndex
    ///
    /// Which iter arg of the outer loop carries the toggle (`:251-254`) — the index
    /// `ToggleDataTransferUpdater` hands `setIterOperand` when it re-bases the toggle (`:2006`,
    /// `:2036`).
    ///
    /// ⛔ `None` IS `DT_CHECK(isValid())`, which subsumes the reference's own `iter_arg_index_ >= 0`:
    /// `invalidate()` stores `-1` there (`:271`), and an [`Option`] is that sentinel with no
    /// arithmetic reachable on it.
    #[must_use]
    pub fn iter_arg_index(&self) -> Option<IterArgIndex> {
        if self.is_valid() {
            self.iter_arg_index
        } else {
            None
        }
    }

    /// Replaces: e485_getX
    ///
    /// The toggle's OTHER constant — `evaluateSub(*c1_, getInit())`, the `X` of `X = c1 - Y`
    /// (`:227-231`), `Y` being [`Self::init`].
    ///
    /// ⛔ `None` IS `DT_CHECK(isValid())` (`:228`), exactly as in [`Self::c1`]; the `?` on `c1` after
    /// it is unreachable and is there only because the field is an [`Option`].
    #[must_use]
    pub fn x(
        &self,
        evaluator: &mut impl ExpressionEvaluator,
        body: &[Op],
        defs: Definitions<'_>,
    ) -> Option<EvaluatedValue> {
        if !self.is_valid() {
            return None;
        }
        let c1 = self.c1?;
        Some(evaluator.evaluate_sub_handle(c1, self.init(body, defs)))
    }
}

impl ConditionalConstantDescriptor {
    /// Replaces: e403_getMin
    ///
    /// The per-unit minimum over every constant the conditional can yield (`:320-323`).
    ///
    /// ⛔ `None` IS `DT_CHECK(isValid())`, which here is the EMPTY list (`:318`): `evaluateMinMax`
    /// would have nothing to fold. Its caller asserts the same thing before asking
    /// (`DT_CHECK_MSG(cc.isValid(), "descriptor may be corrupt")`, `:2051`).
    #[must_use]
    pub fn min(&self, evaluator: &mut impl ExpressionEvaluator) -> Option<EvaluatedValue> {
        if !self.is_valid() {
            return None;
        }
        Some(evaluator.evaluate_min_max(&self.yielded_constants, MinMax::Min))
    }

    /// Replaces: e404_getMax
    ///
    /// The per-unit maximum over the same list (`:325-328`), which is `compute_min = false` and
    /// nothing else — see [`ConditionalConstantDescriptor::min`] for the `None`.
    #[must_use]
    pub fn max(&self, evaluator: &mut impl ExpressionEvaluator) -> Option<EvaluatedValue> {
        if !self.is_valid() {
            return None;
        }
        Some(evaluator.evaluate_min_max(&self.yielded_constants, MinMax::Max))
    }
}

impl IntegerSequenceDescriptor {
    /// Replaces: e405_getMin
    ///
    /// The per-unit minimum of the sequence, folded over its first and last terms ONLY —
    /// `min(init_, init_ + stride_ * size_)` (`:360-372`), the two ends a monotone sequence has.
    ///
    /// ⛔ THAT LAST TERM IS ONE STRIDE PAST THE SEQUENCE: `getAllConstants` enumerates
    /// `init_ + stride_ * i` for `i` in `[0, size_)` (`:401-406`), so the sequence's own last term is
    /// `init_ + stride_ * (size_ - 1)`. PORTED VERBATIM, because the result only ever WIDENS the range
    /// `findClosestPinnedAddr` (`:2123`) must cover, and narrowing it would move this pass's chosen
    /// pinned addresses off the reference's.
    /// ⛔ THE SECOND `None` IS UNREACHABLE FROM A CONSTRUCTED DESCRIPTOR — e280 either sets BOTH
    /// `init_` and `stride_` or invalidates everything — but `isValid()` (`:356`) does not read them.
    #[must_use]
    pub fn min(&self, evaluator: &mut impl ExpressionEvaluator) -> Option<EvaluatedValue> {
        if !self.is_valid() {
            return None;
        }
        let (Some(init), Some(stride), SequenceSize::Terms(size)) =
            (self.init, self.stride, self.size)
        else {
            return None;
        };
        let total_stride = evaluator.evaluate_multiply_by_const(stride, i64::from(size));
        let last_term = evaluator.evaluate_sum_handle(init, total_stride);
        Some(evaluator.evaluate_min_max(&[init, last_term], MinMax::Min))
    }

    /// Replaces: e406_getMax
    ///
    /// The per-unit maximum of the same two terms (`:374-386`) — `compute_min = false` over the same
    /// `init_` and the same one-stride-past-the-end last term; see
    /// [`IntegerSequenceDescriptor::min`] for both traps.
    #[must_use]
    pub fn max(&self, evaluator: &mut impl ExpressionEvaluator) -> Option<EvaluatedValue> {
        if !self.is_valid() {
            return None;
        }
        let (Some(init), Some(stride), SequenceSize::Terms(size)) =
            (self.init, self.stride, self.size)
        else {
            return None;
        };
        let total_stride = evaluator.evaluate_multiply_by_const(stride, i64::from(size));
        let last_term = evaluator.evaluate_sum_handle(init, total_stride);
        Some(evaluator.evaluate_min_max(&[init, last_term], MinMax::Max))
    }
}

impl IntegerSequenceDescriptor {
    /// Replaces: e407_getInit
    ///
    /// The sequence's FIRST TERM — `*init_` under `DT_CHECK(isValid())` (`:388-391`).
    ///
    /// ⛔ `None` IS THAT `DT_CHECK`, not "no first term": every reference caller asks only inside
    /// `if (isq_desc->isValid())` (`:2381`), and an unmatched or [`Self::invalidate`]d sequence has
    /// nothing to give.
    #[must_use]
    pub fn init(&self) -> Option<EvaluatedValue> {
        if self.is_valid() { self.init } else { None }
    }

    /// Replaces: e408_getAllConstants
    ///
    /// APPENDS every address the sequence visits — `init_` alone for a zero stride, else
    /// `init_ + stride_ * i` for `i` in `0..size_` (`:393-408`) — keeping what `output` already holds.
    ///
    /// ⛔ ITS `isValid()` IS A DISCARDED STATEMENT (`:394`), NOT A `DT_CHECK`: nothing stops an
    /// invalid descriptor, so the reference reaches a null `stride_`. Absent fields append NOTHING.
    /// ⛔ `*stride_ == zero_ev` IS THE ANALYSIS'S CONTENT COMPARISON
    /// (`ExpressionEvaluatorUtils.h:63`), not handle identity.
    pub fn get_all_constants(
        &self,
        output: &mut BaseAddrList,
        evaluator: &mut impl ExpressionEvaluator,
    ) {
        let (Some(init), Some(stride), SequenceSize::Terms(size)) =
            (self.init, self.stride, self.size)
        else {
            return;
        };

        // `if (*stride_ == zero_ev) { output.push_back(init_); return; }` (`:396-400`).
        let zero = evaluator.constant(0);
        if evaluator.values_equal(stride, zero) {
            output.push(init);
            return;
        }

        for i in 0..size {
            let step = evaluator.evaluate_multiply_by_const(stride, i64::from(i));
            let term = evaluator.evaluate_sum_handle(init, step);
            output.push(term);
        }
    }
}

impl DiscreteIntegerSetDescriptor {
    /// Replaces: e409_getMin
    ///
    /// The set's lowest address — `init_ + total_negative_delta_` (`:470-474`).
    ///
    /// ⛔ `None` IS THE `DT_CHECK(isValid())` AND AN ABSENT FIELD BOTH: `isValid()` (`:468`) reads
    /// NEITHER `init_` nor either delta, so a valid descriptor can still be missing them.
    #[must_use]
    pub fn min(&self, evaluator: &mut impl ExpressionEvaluator) -> Option<EvaluatedValue> {
        if !self.is_valid() {
            return None;
        }
        Some(evaluator.evaluate_sum_handle(self.init?, self.total_negative_delta?))
    }

    /// Replaces: e410_getMax
    ///
    /// The set's highest address — `init_ + total_positive_delta_` (`:476-480`).
    ///
    /// ⛔ THE TWO DELTAS ARE NOT INTERCHANGEABLE: this one is the positive total, and `isValid()`
    /// guarantees neither, so `None` covers both the `DT_CHECK` and an absent field.
    #[must_use]
    pub fn max(&self, evaluator: &mut impl ExpressionEvaluator) -> Option<EvaluatedValue> {
        if !self.is_valid() {
            return None;
        }
        Some(evaluator.evaluate_sum_handle(self.init?, self.total_positive_delta?))
    }

    /// Replaces: e411_getInit
    ///
    /// The set's INITIAL address — `*init_` under `DT_CHECK(isValid())` (`:482-485`), which
    /// `initializeDescriptor` pushes as the transfer's one base address (`:2398`).
    ///
    /// ⛔ `None` IS THAT `DT_CHECK` PLUS THE ABSENT FIELD, which `isValid()` (`:468`) does not read.
    #[must_use]
    pub fn init(&self) -> Option<EvaluatedValue> {
        if self.is_valid() { self.init } else { None }
    }
}

impl LoopingChainMutableAddrDescriptor {
    /// Replaces: e412_getMin
    ///
    /// The lower of the chain's first and last address — `min(init_, init_ + increment_)`
    /// (`:566-572`), the pair in that order.
    ///
    /// ⛔ IT IS NOT `init_`: the chain's total increment may be negative, which is exactly why the
    /// reference asks `evaluateMinMax` rather than returning the first address.
    #[must_use]
    pub fn min(&self, evaluator: &mut impl ExpressionEvaluator) -> Option<EvaluatedValue> {
        if !self.is_valid() {
            return None;
        }
        let init = self.init?;
        let last_val = evaluator.evaluate_sum_handle(init, self.increment?);
        Some(evaluator.evaluate_min_max(&[init, last_val], MinMax::Min))
    }

    /// Replaces: e413_getMax
    ///
    /// The higher of the chain's first and last address — `max(init_, init_ + increment_)`
    /// (`:574-580`), the pair in that order.
    ///
    /// ⛔ `None` IS THE `DT_CHECK(isValid())` (`:562`) TOGETHER WITH AN ABSENT `init_` or
    /// `increment_`, neither of which `isValid()` reads.
    #[must_use]
    pub fn max(&self, evaluator: &mut impl ExpressionEvaluator) -> Option<EvaluatedValue> {
        if !self.is_valid() {
            return None;
        }
        let init = self.init?;
        let last_val = evaluator.evaluate_sum_handle(init, self.increment?);
        Some(evaluator.evaluate_min_max(&[init, last_val], MinMax::Max))
    }

    /// Replaces: e414_getInit
    ///
    /// The chain's INITIAL mutable address — `*init_` under `DT_CHECK(isValid())` (`:582-585`), which
    /// `initializeDescriptor` pushes as the transfer's one base address (`:2414`).
    ///
    /// ⛔ `None` IS THAT `DT_CHECK`, AND [`Self::invalidate`] REACHES IT WITHOUT CLEARING `init_` —
    /// the head flag and the outer loop are what it drops (`:600-603`).
    #[must_use]
    pub fn init(&self) -> Option<EvaluatedValue> {
        if self.is_valid() { self.init } else { None }
    }
}

/// HOW MANY DISTINCT BASE ADDRESSES A UNIT'S IMMUTABLE TRANSFERS STREAM FROM — the reference's
/// `int num_streams_` (`AddressPinningAndToggle.cpp:1268`), a COUNT and never an index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StreamCount(pub usize);

/// THE PASS'S PER-UNIT STATE — the four fields of `class AddressPinningAndTogglePass`
/// (`AddressPinningAndToggle.cpp:1266-1287`) that [`AddressPinningAndTogglePass::cleanup`] resets.
///
/// ⛔ THE REFERENCE'S OTHER FOUR FIELDS ARE AMBIENT SERVICES, NOT STATE: `dcc_ext_ctx_`, `opts_`,
/// `dom_info_` and `evaluator_` (`:1288-1291`) — and this crate hands the evaluator in as
/// `&mut impl ExpressionEvaluator` at each seam that needs one rather than storing it.
#[derive(Debug, Default)]
pub struct AddressPinningAndTogglePass {
    /// `num_streams_` — the memoised stream count; the reference's `-1` "not computed" is `None`.
    pub num_streams: Option<StreamCount>,
    /// `immut_data_transfer_descriptors_` — the descriptors that DO update the IR.
    pub immut_data_transfer_descriptors: DataTransferDescriptorContainer,
    /// `mut_data_transfer_descriptors_` — collected for analysis only, never written back
    /// (`:1274-1277`).
    pub mut_data_transfer_descriptors: DataTransferDescriptorContainer,
    /// `ev_addr_info_list_` — one entry per base address of each sorted transfer pair.
    pub ev_addr_info_list: Vec<EvAddressInfo>,
}

impl AddressPinningAndTogglePass {
    /// Replaces: e488_computeOrGetNumberOfStreams
    ///
    /// How many DISTINCT evaluated base addresses the immutable descriptors name, memoised on first
    /// ask (`:1439-1448`).
    ///
    /// ⛔ THE SET KEYS ON THE HANDLE, NOT THE VALUE: `SmallSet<const EvaluatedValue *, 4>` (`:1442`)
    /// dedups ARENA POINTERS, so two separately evaluated but equal addresses count twice — this is
    /// [`EvaluatedValue`] identity and never [`ExpressionEvaluator::values_equal`].
    pub fn compute_or_get_number_of_streams(&mut self) -> StreamCount {
        if let Some(computed) = self.num_streams {
            return computed;
        }
        let set_of_streams: BTreeSet<EvaluatedValue> = self
            .immut_data_transfer_descriptors
            .descriptors
            .iter()
            .flat_map(|dtd| dtd.base_addrs.iter().copied())
            .collect();
        let num_streams = StreamCount(set_of_streams.len());
        self.num_streams = Some(num_streams);
        num_streams
    }

    /// Replaces: e489_cleanup
    ///
    /// Drops everything the pass accumulated over one program unit (`:1656-1664`), so the next unit
    /// recomputes its stream count instead of reading this one's.
    ///
    /// ⭐ THE TWO `delete dtd` LOOPS (`:1658`, `:1661`) ARE THE OWNING [`Vec`]'S CLEAR, for the reason
    /// already argued at [`DataTransferDescriptorContainer::clear_all`].
    pub fn cleanup(&mut self) {
        self.num_streams = None;
        self.immut_data_transfer_descriptors.clear_all();
        self.mut_data_transfer_descriptors.clear_all();
        self.ev_addr_info_list.clear();
    }
}

impl ToggleDescriptor {
    /// Replaces: e547_getMin
    ///
    /// The lower of the two constants the base address alternates between —
    /// `evaluateMinMax({&getX(), &getY()}, compute_min = true)` (`:235-238`), `Y` being
    /// [`ToggleDescriptor::init`].
    ///
    /// ⛔ `None` IS `DT_CHECK(isValid())` (`:236`), the same encoding as [`Self::c1`]; the `?` after it
    /// is unreachable and is there only because [`Self::x`] answers an [`Option`].
    /// ⛔ `getX()` IS EVALUATED FIRST AND IT EVALUATES `getInit()` ITSELF — a braced init list is
    /// sequenced left to right, so an evaluator that interns in call order sees `X`'s operands before
    /// `Y`, and swapping the two lines would renumber its arena.
    #[must_use]
    pub fn min(
        &self,
        evaluator: &mut impl ExpressionEvaluator,
        body: &[Op],
        defs: Definitions<'_>,
    ) -> Option<EvaluatedValue> {
        if !self.is_valid() {
            return None;
        }
        let x = self.x(evaluator, body, defs)?;
        let y = self.init(body, defs);
        Some(evaluator.evaluate_min_max(&[x, y], MinMax::Min))
    }

    /// Replaces: e548_getMax
    ///
    /// The higher of the same pair (`:240-243`) — `compute_min = false` and nothing else; see
    /// [`ToggleDescriptor::min`] for the `None` and for why `X` is evaluated first.
    #[must_use]
    pub fn max(
        &self,
        evaluator: &mut impl ExpressionEvaluator,
        body: &[Op],
        defs: Definitions<'_>,
    ) -> Option<EvaluatedValue> {
        if !self.is_valid() {
            return None;
        }
        let x = self.x(evaluator, body, defs)?;
        let y = self.init(body, defs);
        Some(evaluator.evaluate_min_max(&[x, y], MinMax::Max))
    }
}

impl AddressPinningAndTogglePass {
    /// Replaces: e549_runOn
    ///
    /// Runs the pass over every program unit of one module, then canonicalises every query map the
    /// rewrites left behind (`:1187-1198`).
    ///
    /// ⛔ NAMED FOR ITS ARGUMENT: `runOn(ModuleOp)` and `runOn(dataflow::ProgramUnitOp)` (e656) are one
    /// C++ overload set and cannot both be `run_on` here.
    /// ⭐ `new DominanceInfo(unit)` / `delete dom_info_` IS THE MECHANISM FOR REACHING OPERANDS, which
    /// the campaign names droppable — the per-unit `runOn` is handed the unit and asks for dominance
    /// where it needs it.
    pub fn run_on_program<A: Arch, M: Model, W: Workload>(&mut self, program: &mut Program<A, M, W>) {
        for unit in program.units.iter_mut() {
            self.cleanup();
            self.run_on_unit(unit);
        }

        // THE CLEANUP WALK (`:1194-1197`) — every `uniform.query_map` in the module, the preamble
        // included, since a walk from the module op reaches both.
        let mut simplify = |op: &Op| {
            if matches!(op, Op::Uniform(uniform::Op::QueryMap { .. })) {
                todo!(
                    "dcc::uniform::utils::simplifyQueryMapWithSameTarget \
                     (Dialect/Uniform/Utils.cpp:1263) — out of campaign scope; bridge 2 ports it at \
                     the DataflowIR rung as the private `simplify_query_map_with_same_target`"
                )
            }
        };
        walk_pre_order(&program.preamble, &mut simplify);
        for unit in program.units.iter() {
            walk_pre_order(&unit.body, &mut simplify);
        }
    }

    /// `runOn(dataflow::ProgramUnitOp)` — entry 656, level 12, not yet ported.
    fn run_on_unit<A: Arch>(&mut self, unit: &mut ProgramUnit<A>) -> ! {
        let _ = unit;
        todo!(
            "e656_runOn(dataflow::ProgramUnitOp) — the L3LU/L3SU gate, the per-unit descriptor \
             collection and the toggle rewrite (AddressPinningAndToggle.cpp:1337)"
        )
    }
}

/// `Operation::walk<WalkOrder::PreOrder>` OVER A REGION — the op itself before its own regions, which
/// is the order [`AddressPinningAndTogglePass::run_on_program`]'s second walk asks for.
fn walk_pre_order(scope: &[Op], visit: &mut impl FnMut(&Op)) {
    for op in scope {
        visit(op);
        for region in dialects::regions_ref(op) {
            walk_pre_order(region, visit);
        }
    }
}

// crustify:todo: e587_getMin
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:749  (4 body lines, level 5)
//   original  : virtual const EvaluatedValue &getMin() const
//   calls     : e399_getMin, e403_getMin, e405_getMin, e409_getMin, e412_getMin, e547_getMin, e589_getMin

// crustify:todo: e588_getMax
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:754  (4 body lines, level 5)
//   original  : virtual const EvaluatedValue &getMax() const
//   calls     : e400_getMax, e404_getMax, e406_getMax, e410_getMax, e413_getMax, e548_getMax, e590_getMax

// crustify:todo: e589_getMin
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:829  (9 body lines, level 5)
//   original  : const EvaluatedValue &getMin() const override
//   calls     : e399_getMin, e403_getMin, e405_getMin, e409_getMin, e412_getMin, e547_getMin, e587_getMin

// crustify:todo: e590_getMax
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:839  (14 body lines, level 5)
//   original  : const EvaluatedValue &getMax() const override
//   calls     : e400_getMax, e404_getMax, e406_getMax, e410_getMax, e413_getMax, e548_getMax, e588_getMax

// crustify:todo: e591_runOnOperation
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1200  (9 body lines, level 5)
//   original  : void runOnOperation()
//   calls     : e549_runOn

// crustify:todo: e614_computeAddressInfoList
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1675  (59 body lines, level 6)
//   original  : void AddressPinningAndTogglePass::computeAddressInfoList(SenComponents comp)
//   calls     : e007_isPartOfSomeChain, e258_isToggle, e273_isHeadOfChain, e278_isValid, e399_getMin, e400_getMax, e403_getMin, e404_getMax, e405_getMin, e406_getMax, e409_getMin, e410_getMax, e412_getMin, e413_getMax …

// crustify:todo: e637_collectDataTransfers
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1395  (41 body lines, level 7)
//   original  : bool AddressPinningAndTogglePass::collectDataTransfers( SenComponents comp, Operation *op, SenComponents memory_unit, Operation *region_op, int region_num)
//   calls     : e422_insert, e619_LXDataTransferDescriptor, e620_HBMDataTransferDescriptor

// crustify:todo: e647_CollectDataTransfersAndComputeMaxStreams
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1293  (39 body lines, level 8)
//   original  : void AddressPinningAndTogglePass::CollectDataTransfersAndComputeMaxStreams( dataflow::ProgramUnitOp unit, SenComponents comp, SenComponents memory_unit)
//   calls     : e637_collectDataTransfers

// crustify:todo: e648_processToggle
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1522  (13 body lines, level 8)
//   original  : void AddressPinningAndTogglePass::processToggle( DataTransferDescriptor &dtd, const PinningSchemeManager &ps_manager, mlir::MutableOperandRange mutable_addr, mlir::MutableOperandRange immutable_addr, uint32_t element_size)
//   calls     : e263_getToggleDescriptor, e264_getToggleDescriptor, e278_isValid, e638_update

// crustify:todo: e649_processConditionalConstant
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1539  (10 body lines, level 8)
//   original  : void AddressPinningAndTogglePass::processConditionalConstant( DataTransferDescriptor &dtd, const PinningSchemeManager &ps_manager, mlir::MutableOperandRange mutable_addr, mlir::MutableOperandRange immutable_addr, uint32_t element_size)
//   calls     : e638_update

// crustify:todo: e650_processIntegerSequence
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1553  (8 body lines, level 8)
//   original  : void AddressPinningAndTogglePass::processIntegerSequence( DataTransferDescriptor &dtd, const PinningSchemeManager &ps_manager, mlir::MutableOperandRange mutable_addr, mlir::MutableOperandRange immutable_addr, uint32_t element_size)
//   calls     : e638_update

// crustify:todo: e651_processSimpleConstant
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1565  (6 body lines, level 8)
//   original  : void AddressPinningAndTogglePass::processSimpleConstant( DataTransferDescriptor &dtd, const PinningSchemeManager &ps_manager, mlir::MutableOperandRange mutable_addr, mlir::MutableOperandRange immutable_addr, uint32_t element_size)
//   calls     : e638_update

// crustify:todo: e653_processDataTransfer
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1492  (26 body lines, level 9)
//   original  : void AddressPinningAndTogglePass::processDataTransfer( DataTransferDescriptor &dtd, const PinningSchemeManager &ps_manager, mlir::MutableOperandRange mutable_addr, mlir::MutableOperandRange immutable_addr, uint32_t element_size)
//   calls     : e252_size, e258_isToggle, e259_isConditionalConstant, e260_isIntegerSequence, e278_isValid, e279_canBeSimplified, e488_computeOrGetNumberOfStreams, e648_processToggle, e649_processConditionalConstant, e650_processIntegerSequence, e651_processSimpleConstant

// crustify:todo: e654_processDataTransfer
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1456  (34 body lines, level 10)
//   original  : void AddressPinningAndTogglePass::processDataTransfer( DataTransferDescriptor &dtd, const PinningSchemeManager &ps_manager)
//   calls     : e653_processDataTransfer

// crustify:todo: e655_processDataTransfers
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1450  (4 body lines, level 11)
//   original  : void AddressPinningAndTogglePass::processDataTransfers( const PinningSchemeManager &ps_manager)
//   calls     : e653_processDataTransfer, e654_processDataTransfer

// crustify:todo: e656_runOn
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1337  (57 body lines, level 12)
//   original  : void AddressPinningAndTogglePass::runOn(dataflow::ProgramUnitOp unit)
//   calls     : e014_dump, e221_initialize, e252_size, e419_turnHBMConstantOpAddrsToQueryMapsInUniformRegions, e488_computeOrGetNumberOfStreams, e489_cleanup, e491_computeChainingInfo, e614_computeAddressInfoList, e647_CollectDataTransfersAndComputeMaxStreams, e655_processDataTransfers

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Elements;
    use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;
    use crate::formats::Bits;
    use crate::islands::sentient::dialects::sentient::{Extent, Reg, RegType, ShuffleMode};
    use crate::islands::sentient::dialects::{LocalRegion, Val, sentient};
    use crate::transform::sentient::analyses::{
        Evaluation, OffsetSites, OutOfScopeEvaluator, RegionSite,
    };
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::ProgramUnits;
    use crate::transform::sentient::{ForRef, IterArgIndex};
    use crate::units::Residency;

    /// `%u = dataflow.get_unit {type = <unit>}` at func scope.
    fn get_unit(result: u32, unit: DfirUnit) -> Op {
        Op::Dataflow(dataflow::Op::GetUnit {
            result: Val(result),
            residency: Residency::Global,
            unit,
            num_folds: None,
        })
    }

    /// `%c = sentient.scalar_constant {value = N : si64} : index`.
    fn scalar_const(result: u32, value: i64) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result: Val(result),
            reg_locale: RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// A `sentient.load_and_store` between two unit handles, with `src_immutable_addr` the only
    /// address that matters here.
    fn load_and_store(src: u32, dst: u32, src_immutable_addr: u32, result: u32) -> Op {
        Op::Sentient(sentient::Op::LoadAndStore {
            src: Val(src),
            dst: Val(dst),
            src_mutable_addr: Val(0),
            src_immutable_addr: Val(src_immutable_addr),
            src_inc: Val(0),
            dst_mutable_addr: Val(0),
            dst_immutable_addr: Val(0),
            dst_inc: Val(0),
            multicast_info: None,
            results: (Val(result), Val(result + 1)),
            extent: Extent::of(Elements(8), Bits(16)),
            stride: 1,
            rotate_val: None,
            shuffle_mode: ShuffleMode::NoShuffle,
            src_reg: Reg {
                locale: RegType::Lar,
                index: None,
            },
            dst_reg: Reg {
                locale: RegType::Lbr,
                index: None,
            },
            dir: None,
            is_ibr_write: false,
            dbg_name: None,
        })
    }

    /// A matched-looking descriptor field set, so an `invalidate()` has something to clear.
    fn loop_and_arg() -> (Option<ForRef>, Option<IterArgIndex>) {
        (Some(ForRef(Val(7))), Some(IterArgIndex(1)))
    }

    /// One transfer carrying `pattern_desc_` and `base_addrs_`, the only two fields any `is*()` reads.
    fn transfer(
        pattern_desc: Option<PatternDescriptor>,
        base_addrs: usize,
    ) -> DataTransferDescriptor {
        DataTransferDescriptor {
            op: OpId::at(&[0]),
            pattern_desc,
            base_addrs: (0..base_addrs).map(|i| EvaluatedValue(i as u32)).collect(),
            region: RegionSite::default(),
            memory_unit: DescriptorMemoryUnit::Lx,
        }
    }

    /// A toggle whose three validity fields are all present and that did NOT collapse to one value.
    fn matched_toggle() -> ToggleDescriptor {
        let (outer_loop, iter_arg_index) = loop_and_arg();
        ToggleDescriptor {
            outer_loop,
            iter_arg_index,
            c1: Some(EvaluatedValue(3)),
            can_be_simplified: false,
        }
    }

    #[test]
    fn toggle_invalidate_clears_all_three_validity_fields() {
        let (outer_loop, iter_arg_index) = loop_and_arg();
        let mut desc = ToggleDescriptor {
            outer_loop,
            iter_arg_index,
            c1: Some(EvaluatedValue(3)),
            // ⛔ `invalidate()` does not clear this one either, so the equality below is only a
            // statement about the three fields `isValid()` reads.
            can_be_simplified: false,
        };
        desc.invalidate();
        assert_eq!(desc, ToggleDescriptor::default());
    }

    #[test]
    fn get_all_constants_appends_in_order_without_clearing() {
        let desc = ConditionalConstantDescriptor {
            yielded_constants: vec![EvaluatedValue(11), EvaluatedValue(22)],
            can_be_simplified: false,
        };
        let mut output: BaseAddrList = vec![EvaluatedValue(99)];
        desc.get_all_constants(&mut output);
        assert_eq!(
            output,
            vec![EvaluatedValue(99), EvaluatedValue(11), EvaluatedValue(22)]
        );
    }

    #[test]
    fn integer_sequence_invalidate_clears_size_to_zero_not_symbolic() {
        let (outer_loop, iter_arg_index) = loop_and_arg();
        let mut desc = IntegerSequenceDescriptor {
            outer_loop,
            iter_arg_index,
            init: Some(EvaluatedValue(4)),
            stride: Some(EvaluatedValue(8)),
            size: SequenceSize::Terms(6),
            can_be_simplified: false,
        };
        desc.invalidate();
        assert_eq!(desc, IntegerSequenceDescriptor::default());
        assert_eq!(desc.size, SequenceSize::Cleared);
        assert_ne!(desc.size, SequenceSize::Symbolic);
    }

    #[test]
    fn discrete_integer_set_invalidate_clears_both_delta_totals() {
        let (outer_loop, iter_arg_index) = loop_and_arg();
        let mut desc = DiscreteIntegerSetDescriptor {
            outer_loop,
            iter_arg_index,
            init: Some(EvaluatedValue(4)),
            total_positive_delta: Some(EvaluatedValue(64)),
            total_negative_delta: Some(EvaluatedValue(32)),
            can_be_simplified: false,
        };
        desc.invalidate();
        assert_eq!(desc, DiscreteIntegerSetDescriptor::default());
    }

    #[test]
    fn looping_chain_invalidate_leaves_the_chain_measurements_alone() {
        let (outer_loop, iter_arg_index) = loop_and_arg();
        let mut desc = LoopingChainMutableAddrDescriptor {
            is_head_of_chain: true,
            outer_loop,
            iter_arg_index,
            size: ChainSize(3),
            init: Some(EvaluatedValue(4)),
            increment: Some(EvaluatedValue(8)),
            can_be_simplified: false,
        };
        desc.invalidate();
        assert!(!desc.is_head_of_chain);
        assert_eq!(desc.outer_loop, None);
        // The reference clears only those two: everything else survives.
        assert_eq!(desc.iter_arg_index, iter_arg_index);
        assert_eq!(desc.size, ChainSize(3));
        assert_eq!(desc.init, Some(EvaluatedValue(4)));
        assert_eq!(desc.increment, Some(EvaluatedValue(8)));
    }

    #[test]
    fn part_of_some_chain_is_false_without_an_entry_and_true_once_set() {
        let mut container = DataTransferDescriptorContainer::default();
        container.descriptors.push(transfer(None, 0));
        let desc = DescriptorId(0);
        assert!(!container.is_part_of_some_chain(desc));
        container.set_chaining_info(desc, ChainFlag::PartOfChain, true);
        assert!(container.is_part_of_some_chain(desc));
    }

    #[test]
    fn set_chaining_info_ors_into_an_existing_entry() {
        let mut container = DataTransferDescriptorContainer::default();
        let desc = DescriptorId(0);
        container.set_chaining_info(desc, ChainFlag::PartOfChain, true);
        container.set_chaining_info(desc, ChainFlag::HeadOfChain, true);
        container.set_chaining_info(desc, ChainFlag::PartOfChain, false);
        assert_eq!(
            container.chaining_info.get(&desc),
            Some(&ChainFlags {
                part_of_chain: false,
                head_of_chain: true,
                head_of_looping_chain: false,
            })
        );
    }

    #[test]
    fn clearing_a_flag_on_an_absent_descriptor_inserts_nothing() {
        let mut container = DataTransferDescriptorContainer::default();
        container.set_chaining_info(DescriptorId(0), ChainFlag::PartOfChain, false);
        assert!(container.chaining_info.is_empty());
    }

    /// 011/656 — the toggle's fall-back offset is the `scalar_sub`'s own `$out`, with no new op.
    #[test]
    fn toggle_get_offset_is_the_toggle_sub_result() {
        let updater = ToggleDataTransferUpdater {
            toggle_sub: ToggleSub::of(Val(17)),
        };
        assert_eq!(updater.get_offset(EvaluatedValue(7)), Val(17));
    }

    /// 012/656 — the conditional's fall-back offset is `if_op_.getResult(res_index_)`, and ⛔ NOT the
    /// immutable-addr operand: `updateImmutableAddr` reassigns that (`:2060`) after fixing the index
    /// (`:2053`), so the two are different values by the time `getOffset` runs.
    #[test]
    fn conditional_const_get_offset_is_the_selected_if_result_not_the_new_immutable() {
        let updater = ConditionalConstDataTransferUpdater {
            if_result: ConditionalConstResult {
                index: YieldedIndex(2),
                val: Val(9),
            },
        };
        let new_immutable_addr = Val(31);
        assert_eq!(updater.get_offset(EvaluatedValue(7)), Val(9));
        assert_ne!(updater.get_offset(EvaluatedValue(7)), new_immutable_addr);
    }

    /// 013/656 — the integer sequence's fall-back offset is the loop's carried ARGUMENT, ⛔ not the
    /// `init` it entered with nor the `result` it leaves behind.
    #[test]
    fn integer_sequence_get_offset_is_the_carried_arg() {
        let carried = sentient::Carried {
            init: Val(4),
            arg: Val(5),
            result: Val(6),
            reg: sentient::Reg {
                locale: sentient::RegType::Unknown,
                index: None,
            },
            program_header: false,
            element_size: None,
        };
        let updater = IntegerSequenceDataTransferUpdater {
            iter_arg: carried.arg,
        };
        assert_eq!(updater.get_offset(EvaluatedValue(7)), Val(5));
        assert_ne!(updater.get_offset(EvaluatedValue(7)), carried.init);
        assert_ne!(updater.get_offset(EvaluatedValue(7)), carried.result);
    }

    /// A descriptor that reports only which one it is — enough to check e014's ORDER and headers
    /// without standing in for `DataTransferDescriptor::dump` (e492), whose text is that unit's.
    struct NamedDescriptor(&'static str);

    impl DumpDescriptor for NamedDescriptor {
        fn dump(&self) -> String {
            format!("{}\n", self.0)
        }
    }

    /// 014/656 — both headers, immutable container first, each container in insertion order.
    #[test]
    fn dump_writes_immutable_descriptors_before_mutable_ones() {
        let a = NamedDescriptor("immut-a");
        let b = NamedDescriptor("immut-b");
        let c = NamedDescriptor("mut-c");
        let immut: [&dyn DumpDescriptor; 2] = [&a, &b];
        let mutable: [&dyn DumpDescriptor; 1] = [&c];
        assert_eq!(
            dump(&immut, &mutable),
            "Immutable transfer descriptors:\nimmut-a\nimmut-b\nMutable transfer descriptors:\nmut-c\n"
        );
    }

    /// A descriptor whose pattern is `kind` and whose one base address makes the fixture realistic —
    /// the reference's `base_addrs_` is what the absent `isValid()` conjunct would read.
    fn with_pattern(kind: PatternDescriptor) -> DataTransferDescriptor {
        DataTransferDescriptor {
            op: OpId::at(&[0]),
            pattern_desc: Some(kind),
            base_addrs: vec![EvaluatedValue(64)],
            region: RegionSite::default(),
            memory_unit: DescriptorMemoryUnit::Lx,
        }
    }

    /// 265+266/656 — both overloads reach the same `ConditionalConstantDescriptor`, and a write
    /// through the mutable one is visible to the shared one.
    ///
    /// ⛔ NO IN-TREE CALLER ACTUALLY WRITES THROUGH THE NON-CONST OVERLOAD: it exists because
    /// `updateImmutableAddr` binds `ConditionalConstantDescriptor &cc` (`:2050`) on a non-const
    /// `dtd_`, and what its `updateVariableOffsetCalculation` then rewrites is the IR, not the
    /// descriptor.
    #[test]
    fn conditional_constant_descriptor_is_read_and_written_through_both_overloads() {
        let cc = ConditionalConstantDescriptor {
            yielded_constants: vec![EvaluatedValue(16), EvaluatedValue(48)],
            can_be_simplified: false,
        };
        let mut dtd = with_pattern(PatternDescriptor::ConditionalConstant(cc.clone()));
        assert_eq!(dtd.conditional_constant_descriptor(), &cc);
        dtd.conditional_constant_descriptor_mut().can_be_simplified = true;
        assert!(dtd.conditional_constant_descriptor().can_be_simplified);
    }

    /// 267+268/656 — `updateImmutableAddr` (`:2117`) binds the mutable overload; the sequence it
    /// hands back is the same one the shared overload sees.
    #[test]
    fn integer_sequence_descriptor_is_read_and_written_through_both_overloads() {
        let isq = IntegerSequenceDescriptor {
            outer_loop: Some(ForRef(Val(7))),
            iter_arg_index: Some(IterArgIndex(1)),
            init: Some(EvaluatedValue(4)),
            stride: Some(EvaluatedValue(8)),
            size: SequenceSize::Terms(3),
            can_be_simplified: false,
        };
        let mut dtd = with_pattern(PatternDescriptor::IntegerSequence(isq));
        assert_eq!(dtd.integer_sequence_descriptor(), &isq);
        dtd.integer_sequence_descriptor_mut().size = SequenceSize::Cleared;
        assert_eq!(
            dtd.integer_sequence_descriptor().size,
            SequenceSize::Cleared
        );
    }

    /// 269+270/656 — the discrete set, whose only in-tree reader is `canBeSimplified()` (`:2511`).
    #[test]
    fn discrete_integer_set_descriptor_is_read_and_written_through_both_overloads() {
        let dis = DiscreteIntegerSetDescriptor {
            outer_loop: Some(ForRef(Val(7))),
            iter_arg_index: Some(IterArgIndex(0)),
            init: Some(EvaluatedValue(4)),
            total_positive_delta: Some(EvaluatedValue(24)),
            total_negative_delta: Some(EvaluatedValue(0)),
            can_be_simplified: false,
        };
        let mut dtd = with_pattern(PatternDescriptor::DiscreteIntegerSet(dis));
        assert_eq!(dtd.discrete_integer_set_descriptor(), &dis);
        dtd.discrete_integer_set_descriptor_mut().init = None;
        assert_eq!(dtd.discrete_integer_set_descriptor().init, None);
    }

    /// 271+272/656 — the looping chain head, whose only in-tree reader is `canBeSimplified()` (`:2513`).
    #[test]
    fn looping_chain_mutable_addr_descriptor_is_read_and_written_through_both_overloads() {
        let lcma = LoopingChainMutableAddrDescriptor {
            is_head_of_chain: true,
            outer_loop: Some(ForRef(Val(7))),
            iter_arg_index: Some(IterArgIndex(2)),
            size: ChainSize(3),
            init: Some(EvaluatedValue(4)),
            increment: Some(EvaluatedValue(8)),
            can_be_simplified: false,
        };
        let mut dtd = with_pattern(PatternDescriptor::LoopingChainMutableAddr(lcma));
        assert_eq!(dtd.looping_chain_mutable_addr_descriptor(), &lcma);
        dtd.looping_chain_mutable_addr_descriptor_mut()
            .is_head_of_chain = false;
        assert!(!dtd.looping_chain_mutable_addr_descriptor().is_head_of_chain);
    }

    /// The `isa<>` conjunct of the `DT_CHECK`: a pattern of ANOTHER kind aborts, it does not coerce.
    #[test]
    #[should_panic(expected = "DT_CHECK(isIntegerSequence())")]
    fn asking_a_toggle_for_the_integer_sequence_descriptor_is_the_dt_check() {
        let toggle = ToggleDescriptor {
            outer_loop: Some(ForRef(Val(7))),
            iter_arg_index: Some(IterArgIndex(1)),
            c1: Some(EvaluatedValue(3)),
            can_be_simplified: false,
        };
        let _ = with_pattern(PatternDescriptor::Toggle(toggle)).integer_sequence_descriptor();
    }

    /// The `pattern_desc_ != nullptr` conjunct: an unmatched transfer aborts too.
    #[test]
    #[should_panic(expected = "DT_CHECK(isConditionalConstant())")]
    fn asking_an_unmatched_transfer_for_a_pattern_descriptor_is_the_dt_check() {
        let _ = transfer(None, 0).conditional_constant_descriptor();
    }

    /// 257/656 — the `DT_CHECK(size() == 1)`: one stored address is the answer, a toggle's pair and an
    /// unmatched transfer's empty list are both refusals.
    #[test]
    fn e257_base_addr_is_the_single_stored_address_or_nothing() {
        assert_eq!(transfer(None, 1).base_addr(), Some(EvaluatedValue(0)));
        assert_eq!(transfer(None, 2).base_addr(), None);
        assert_eq!(transfer(None, 0).base_addr(), None);
    }

    /// 258/656 — a toggle needs TWO stored addresses, or one when it simplified; and an invalidated
    /// toggle is not one however many it holds.
    #[test]
    fn e258_a_toggle_needs_two_base_addrs_unless_it_simplified_to_one() {
        let toggle = matched_toggle();
        assert!(transfer(Some(PatternDescriptor::Toggle(toggle)), 2).is_toggle());
        assert!(!transfer(Some(PatternDescriptor::Toggle(toggle)), 1).is_toggle());

        let simplified = ToggleDescriptor {
            can_be_simplified: true,
            ..toggle
        };
        assert!(transfer(Some(PatternDescriptor::Toggle(simplified)), 1).is_toggle());

        let mut invalidated = toggle;
        invalidated.invalidate();
        assert!(!transfer(Some(PatternDescriptor::Toggle(invalidated)), 2).is_toggle());
        assert!(!transfer(None, 2).is_toggle());
    }

    /// 259/656 — one yielded constant and one stored address, and ⛔ the two conditions are separate:
    /// dropping either flips the answer.
    #[test]
    fn e259_conditional_constant_needs_a_yielded_constant_and_a_stored_address() {
        let matched = ConditionalConstantDescriptor {
            yielded_constants: vec![EvaluatedValue(11)],
            can_be_simplified: false,
        };
        let empty = ConditionalConstantDescriptor {
            yielded_constants: Vec::new(),
            can_be_simplified: false,
        };
        assert!(
            transfer(
                Some(PatternDescriptor::ConditionalConstant(matched.clone())),
                1
            )
            .is_conditional_constant()
        );
        assert!(
            !transfer(Some(PatternDescriptor::ConditionalConstant(matched)), 0)
                .is_conditional_constant()
        );
        assert!(
            !transfer(Some(PatternDescriptor::ConditionalConstant(empty)), 1)
                .is_conditional_constant()
        );
    }

    /// 260/656 — `size_ > 0`: a symbolic length is as invalid as a cleared one.
    #[test]
    fn e260_integer_sequence_refuses_a_symbolic_or_cleared_length() {
        let (outer_loop, iter_arg_index) = loop_and_arg();
        let seq = IntegerSequenceDescriptor {
            outer_loop,
            iter_arg_index,
            init: Some(EvaluatedValue(4)),
            stride: Some(EvaluatedValue(8)),
            size: SequenceSize::Terms(6),
            can_be_simplified: false,
        };
        assert!(transfer(Some(PatternDescriptor::IntegerSequence(seq)), 1).is_integer_sequence());
        for size in [SequenceSize::Symbolic, SequenceSize::Cleared] {
            let unsized_seq = IntegerSequenceDescriptor { size, ..seq };
            assert!(
                !transfer(Some(PatternDescriptor::IntegerSequence(unsized_seq)), 1)
                    .is_integer_sequence()
            );
        }
    }

    /// 261/656 — the loop and its iter arg are the whole validity; ⛔ the delta totals are not part of
    /// it, so a set that has them but no loop is still refused.
    #[test]
    fn e261_discrete_integer_set_is_the_loop_and_its_iter_arg_not_the_deltas() {
        let (outer_loop, iter_arg_index) = loop_and_arg();
        let deltas = DiscreteIntegerSetDescriptor {
            outer_loop,
            iter_arg_index,
            init: Some(EvaluatedValue(4)),
            total_positive_delta: Some(EvaluatedValue(64)),
            total_negative_delta: Some(EvaluatedValue(32)),
            can_be_simplified: false,
        };
        assert!(
            transfer(Some(PatternDescriptor::DiscreteIntegerSet(deltas)), 1)
                .is_discrete_integer_set()
        );
        let mut invalidated = deltas;
        invalidated.invalidate();
        assert!(
            !transfer(Some(PatternDescriptor::DiscreteIntegerSet(invalidated)), 1)
                .is_discrete_integer_set()
        );
        let no_loop = DiscreteIntegerSetDescriptor {
            outer_loop: None,
            ..deltas
        };
        assert!(
            !transfer(Some(PatternDescriptor::DiscreteIntegerSet(no_loop)), 1)
                .is_discrete_integer_set()
        );
    }

    /// 262/656 — head flag, `size_ >= 1` and a loop, all three.
    #[test]
    fn e262_looping_chain_needs_the_head_flag_a_size_and_a_loop() {
        let (outer_loop, iter_arg_index) = loop_and_arg();
        let head = LoopingChainMutableAddrDescriptor {
            is_head_of_chain: true,
            outer_loop,
            iter_arg_index,
            size: ChainSize(3),
            init: Some(EvaluatedValue(4)),
            increment: Some(EvaluatedValue(8)),
            can_be_simplified: false,
        };
        let chain = |desc| transfer(Some(PatternDescriptor::LoopingChainMutableAddr(desc)), 1);
        assert!(chain(head).is_looping_chain_mutable_addr());
        assert!(
            !chain(LoopingChainMutableAddrDescriptor {
                is_head_of_chain: false,
                ..head
            })
            .is_looping_chain_mutable_addr()
        );
        assert!(
            !chain(LoopingChainMutableAddrDescriptor {
                size: ChainSize(0),
                ..head
            })
            .is_looping_chain_mutable_addr()
        );
    }

    /// 263/656 — the mutable overload is what an `invalidate()` travels through, and the same call
    /// afterwards refuses: the descriptor stopped being a valid toggle.
    #[test]
    fn e263_toggle_descriptor_mut_is_how_a_toggle_gets_invalidated() {
        let mut dtd = transfer(Some(PatternDescriptor::Toggle(matched_toggle())), 2);
        dtd.toggle_descriptor_mut()
            .expect("a matched toggle")
            .invalidate();
        assert!(dtd.toggle_descriptor_mut().is_none());
        assert_eq!(
            dtd.pattern_desc,
            Some(PatternDescriptor::Toggle(ToggleDescriptor::default()))
        );
    }

    /// 264/656 — the shared overload answers `Some` exactly when e258 does, so a present-but-invalid
    /// `Toggle` arm is refused rather than handed out.
    #[test]
    fn e264_toggle_descriptor_is_none_for_a_present_but_invalid_toggle() {
        let toggle = matched_toggle();
        let matched = transfer(Some(PatternDescriptor::Toggle(toggle)), 2);
        assert_eq!(matched.toggle_descriptor(), Some(&toggle));
        // Present, and still refused: one stored address without `can_be_simplified`.
        let one_addr = transfer(Some(PatternDescriptor::Toggle(toggle)), 1);
        assert!(one_addr.toggle_descriptor().is_none());
        assert!(matches!(
            one_addr.pattern_desc,
            Some(PatternDescriptor::Toggle(_))
        ));
    }

    #[test]
    fn head_of_chain_needs_both_bits_and_an_entry() {
        let mut container = DataTransferDescriptorContainer::default();
        let desc = DescriptorId(0);
        // No entry at all — the `chaining_info_.find` miss.
        assert!(!container.is_head_of_chain(desc));
        // ⭐ THE GUARD: `kHeadOfChain` alone is not an answer of true.
        container.set_chaining_info(desc, ChainFlag::HeadOfChain, true);
        assert!(!container.is_head_of_chain(desc));
        container.set_chaining_info(desc, ChainFlag::PartOfChain, true);
        assert!(container.is_head_of_chain(desc));
    }

    #[test]
    fn an_hbm_source_constant_addr_becomes_a_query_map_over_the_regions_two_units() {
        // Func scope: the two HBM units the region runs on, a PE destination and the constant.
        let scope = vec![
            get_unit(1, DfirUnit::Hbm),
            get_unit(2, DfirUnit::Hbm),
            get_unit(3, DfirUnit::Pe),
            scalar_const(4, 64),
        ];
        let mut region_op = UniformRegions::EqualizePattern {
            regions: vec![LocalRegion {
                arg: Val(5),
                units: vec![Val(1), Val(2)],
                body: vec![load_and_store(1, 3, 4, 6)],
            }],
        };
        let mut values = Values::default();
        for _ in 0..7 {
            let _ = values.mint();
        }
        let enclosing: [&[Op]; 1] = [&scope];
        let constants = turn_hbm_constant_op_addrs_to_query_maps_helper(
            &mut region_op,
            &enclosing,
            &mut values,
        );

        // One `sentient.scalar_constant` per unit, all carrying the original address.
        assert_eq!(constants, vec![scalar_const(7, 64), scalar_const(8, 64)]);
        let body = &region_op.regions()[0].body;
        assert_eq!(
            body[0],
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(9),
                pairs: vec![(Val(1), Val(7)), (Val(2), Val(8))],
            })
        );
        // ⭐ THE `query_map`'S KEY IS THE REGION ARGUMENT, and the pair sits AHEAD of its reader.
        assert_eq!(
            body[1],
            Op::Uniform(uniform::Op::QueryMap {
                result: Val(10),
                map: Val(9),
                key: Val(5),
            })
        );
        assert_eq!(body[2], load_and_store(1, 3, 10, 6));
    }

    #[test]
    fn an_empty_region_unit_list_leaves_the_address_alone() {
        let scope = vec![get_unit(1, DfirUnit::Hbm), scalar_const(4, 64)];
        let mut region_op = UniformRegions::EqualizePattern {
            regions: vec![LocalRegion {
                arg: Val(5),
                units: Vec::new(),
                body: vec![load_and_store(1, 3, 4, 6)],
            }],
        };
        let mut values = Values::default();
        let enclosing: [&[Op]; 1] = [&scope];
        let constants = turn_hbm_constant_op_addrs_to_query_maps_helper(
            &mut region_op,
            &enclosing,
            &mut values,
        );
        // ⚠️ THE DIVERGENCE THE DOC NAMES: the reference assigns the NULL `createQueryMap..` returns.
        assert!(constants.is_empty());
        assert_eq!(
            region_op.regions()[0].body,
            vec![load_and_store(1, 3, 4, 6)]
        );
    }
    /// THE HANDLE FLAVOUR OF THE OUT-OF-SCOPE EVALUATOR WITH ITS ANSWERS STATED AS INTEGERS — a
    /// returned handle names a value these tests can read back.
    ///
    /// ⛔ IT DOES NOT DEDUPE BY VALUE, deliberately: the reference keys its arena on the SOURCE, so
    /// `evaluateValue(%c0)` and `getConstant(0)` are two entries holding the same 0 — and a double
    /// that folded them together would let handle identity pass e408's zero-stride test.
    #[derive(Default)]
    struct StatedEvaluator {
        held: Vec<i64>,
    }

    impl StatedEvaluator {
        fn hold(&mut self, value: i64) -> EvaluatedValue {
            self.held.push(value);
            EvaluatedValue(u32::try_from(self.held.len() - 1).unwrap_or_default())
        }

        fn value(&self, ev: EvaluatedValue) -> i64 {
            self.held
                .get(usize::try_from(ev.0).unwrap_or_default())
                .copied()
                .unwrap_or_default()
        }

        fn values(&self, evs: &[EvaluatedValue]) -> Vec<i64> {
            evs.iter().map(|ev| self.value(*ev)).collect()
        }
    }

    impl ExpressionEvaluator for StatedEvaluator {
        fn evaluate_value(&mut self, _value: Val) -> Evaluation {
            todo!("e407-e414 ask for handles, never for a decoded evaluation")
        }

        fn evaluate_sum(&mut self, _lhs: &Evaluation, _rhs: &Evaluation) -> Evaluation {
            todo!("e407-e414 ask for handles, never for a decoded evaluation")
        }

        fn build_offset_value(
            &mut self,
            _evaluation: &Evaluation,
            _sites: &mut OffsetSites<'_>,
            _walked: &mut Vec<Op>,
            _ty: ScalarTy,
        ) -> Val {
            todo!("e407-e414 build no value")
        }

        fn build_offset_value_of(
            &mut self,
            immutable: EvaluatedValue,
            sites: &mut OffsetSites<'_>,
            _walked: &mut Vec<Op>,
            ty: ScalarTy,
        ) -> Val {
            let value = self.value(immutable);
            let result = sites.values.mint();
            sites
                .consts
                .push(Op::Sentient(sentient::Op::ScalarConstant {
                    value,
                    result,
                    reg_locale: RegType::Imm,
                    ty,
                    is_symbol: false,
                }));
            result
        }

        fn constant(&mut self, value: i64) -> EvaluatedValue {
            self.hold(value)
        }

        /// ⛔ CONTENTS, NOT HANDLES — `EvaluatedValue::operator==`, which is what e408 tests its
        /// stride with, so a separately interned zero must compare EQUAL to a zero stride.
        fn values_equal(&mut self, lhs: EvaluatedValue, rhs: EvaluatedValue) -> bool {
            self.value(lhs) == self.value(rhs)
        }

        fn evaluate_sub_handle(
            &mut self,
            lhs: EvaluatedValue,
            rhs: EvaluatedValue,
        ) -> EvaluatedValue {
            let difference = self.value(lhs) - self.value(rhs);
            self.hold(difference)
        }

        fn evaluate_sum_handle(
            &mut self,
            lhs: EvaluatedValue,
            rhs: EvaluatedValue,
        ) -> EvaluatedValue {
            let sum = self.value(lhs) + self.value(rhs);
            self.hold(sum)
        }

        fn evaluate_multiply_by_const(&mut self, ev: EvaluatedValue, by: i64) -> EvaluatedValue {
            let product = self.value(ev) * by;
            self.hold(product)
        }

        fn evaluate_min_max(&mut self, values: &[EvaluatedValue], which: MinMax) -> EvaluatedValue {
            let held: Vec<i64> = values.iter().map(|ev| self.value(*ev)).collect();
            let picked = match which {
                MinMax::Min => held.iter().min().copied(),
                MinMax::Max => held.iter().max().copied(),
            };
            self.hold(picked.unwrap_or_default())
        }
    }

    /// A MATCHED integer sequence: an outer loop, an index, and both value fields present.
    fn integer_sequence(
        size: SequenceSize,
        init: EvaluatedValue,
        stride: EvaluatedValue,
    ) -> IntegerSequenceDescriptor {
        let (outer_loop, iter_arg_index) = loop_and_arg();
        IntegerSequenceDescriptor {
            outer_loop,
            iter_arg_index,
            init: Some(init),
            stride: Some(stride),
            size,
            can_be_simplified: false,
        }
    }

    /// A MATCHED discrete integer set, whose `isValid()` reads only the loop and the index.
    fn discrete_set(
        init: EvaluatedValue,
        total_positive_delta: EvaluatedValue,
        total_negative_delta: EvaluatedValue,
    ) -> DiscreteIntegerSetDescriptor {
        let (outer_loop, iter_arg_index) = loop_and_arg();
        DiscreteIntegerSetDescriptor {
            outer_loop,
            iter_arg_index,
            init: Some(init),
            total_positive_delta: Some(total_positive_delta),
            total_negative_delta: Some(total_negative_delta),
            can_be_simplified: false,
        }
    }

    /// A MATCHED looping chain of two transfers.
    fn looping_chain(
        init: EvaluatedValue,
        increment: EvaluatedValue,
    ) -> LoopingChainMutableAddrDescriptor {
        let (outer_loop, iter_arg_index) = loop_and_arg();
        LoopingChainMutableAddrDescriptor {
            is_head_of_chain: true,
            outer_loop,
            iter_arg_index,
            size: ChainSize(2),
            init: Some(init),
            increment: Some(increment),
            can_be_simplified: false,
        }
    }

    /// ⛔ [`SequenceSize::Symbolic`] IS THE REFERENCE'S `size_ == -1`, which fails `size_ > 0` and so
    /// takes the first term away again even though `init_` is still there.
    #[test]
    fn e407_gives_the_first_term_only_while_the_sequence_is_valid() {
        let mut evaluator = StatedEvaluator::default();
        let init = evaluator.constant(100);
        let stride = evaluator.constant(3);
        let desc = integer_sequence(SequenceSize::Terms(5), init, stride);
        assert_eq!(desc.init(), Some(init));
        assert_eq!(
            IntegerSequenceDescriptor {
                size: SequenceSize::Symbolic,
                ..desc
            }
            .init(),
            None
        );
    }

    /// The reference's own sequence (`:345-352`), and its zero-stride special case: `100 + 3 * i` for
    /// four terms APPENDED after what the caller already held, then one lone `init_` when the stride
    /// evaluates to zero — ⛔ a SEPARATELY interned zero, which only a content comparison matches.
    #[test]
    fn e408_appends_every_term_and_collapses_a_zero_stride_to_one() {
        let mut evaluator = StatedEvaluator::default();
        let init = evaluator.constant(100);
        let stride = evaluator.constant(3);
        let seed = evaluator.constant(-7);
        let desc = integer_sequence(SequenceSize::Terms(4), init, stride);

        let mut output: BaseAddrList = vec![seed];
        desc.get_all_constants(&mut output, &mut evaluator);
        assert_eq!(evaluator.values(&output), vec![-7, 100, 103, 106, 109]);

        let flat = IntegerSequenceDescriptor {
            stride: Some(evaluator.constant(0)),
            ..desc
        };
        let mut output = BaseAddrList::new();
        flat.get_all_constants(&mut output, &mut evaluator);
        assert_eq!(output, vec![init]);
    }

    /// `100 + (-9)`, and ⛔ the negative total is NOT the positive one.
    #[test]
    fn e409_adds_the_negative_delta_to_the_initial_value() {
        let mut evaluator = StatedEvaluator::default();
        let init = evaluator.constant(100);
        let up = evaluator.constant(24);
        let down = evaluator.constant(-9);
        let desc = discrete_set(init, up, down);
        let min = desc.min(&mut evaluator);
        assert_eq!(min.map(|ev| evaluator.value(ev)), Some(91));
        let cleared = DiscreteIntegerSetDescriptor {
            init: None,
            ..desc
        };
        assert_eq!(cleared.min(&mut evaluator), None);
    }

    /// `100 + 24`, the POSITIVE total — and `None` once the loop `isValid()` reads is gone.
    #[test]
    fn e410_adds_the_positive_delta_to_the_initial_value() {
        let mut evaluator = StatedEvaluator::default();
        let init = evaluator.constant(100);
        let up = evaluator.constant(24);
        let down = evaluator.constant(-9);
        let desc = discrete_set(init, up, down);
        let max = desc.max(&mut evaluator);
        assert_eq!(max.map(|ev| evaluator.value(ev)), Some(124));
        let unmatched = DiscreteIntegerSetDescriptor {
            outer_loop: None,
            ..desc
        };
        assert_eq!(unmatched.max(&mut evaluator), None);
    }

    /// ⛔ `isValid()` DOES NOT READ `init_` HERE (`:468`), so an absent one is its own `None`.
    #[test]
    fn e411_gives_the_initial_value_of_a_matched_set_only() {
        let mut evaluator = StatedEvaluator::default();
        let init = evaluator.constant(100);
        let zero = evaluator.constant(0);
        let desc = discrete_set(init, zero, zero);
        assert_eq!(desc.init(), Some(init));
        assert_eq!(
            DiscreteIntegerSetDescriptor {
                iter_arg_index: None,
                ..desc
            }
            .init(),
            None
        );
    }

    /// A chain that WALKS BACKWARDS: `increment_ = -12`, so the minimum is the LAST address, not
    /// `init_`.
    #[test]
    fn e412_takes_the_lower_of_the_first_and_last_address() {
        let mut evaluator = StatedEvaluator::default();
        let init = evaluator.constant(100);
        let increment = evaluator.constant(-12);
        let desc = looping_chain(init, increment);
        let min = desc.min(&mut evaluator);
        assert_eq!(min.map(|ev| evaluator.value(ev)), Some(88));
        let short = LoopingChainMutableAddrDescriptor {
            size: ChainSize(0),
            ..desc
        };
        assert_eq!(short.min(&mut evaluator), None);
    }

    /// A forward chain: `increment_ = 12`, so the maximum is the last address and the minimum would
    /// have been `init_`.
    #[test]
    fn e413_takes_the_higher_of_the_first_and_last_address() {
        let mut evaluator = StatedEvaluator::default();
        let init = evaluator.constant(100);
        let increment = evaluator.constant(12);
        let desc = looping_chain(init, increment);
        let max = desc.max(&mut evaluator);
        assert_eq!(max.map(|ev| evaluator.value(ev)), Some(112));
        let min = desc.min(&mut evaluator);
        assert_eq!(min.map(|ev| evaluator.value(ev)), Some(100));
    }

    /// ⛔ [`LoopingChainMutableAddrDescriptor::invalidate`] LEAVES `init_` ALONE, so the `None` here
    /// comes from the head flag and the loop it DOES clear.
    #[test]
    fn e414_gives_the_chain_initial_address_until_it_is_invalidated() {
        let mut evaluator = StatedEvaluator::default();
        let init = evaluator.constant(100);
        let increment = evaluator.constant(12);
        let mut desc = looping_chain(init, increment);
        assert_eq!(desc.init(), Some(init));
        desc.invalidate();
        assert_eq!(desc.init, Some(init));
        assert_eq!(desc.init(), None);
    }

    /// A simple constant's range is a point: both ends read `ev_`, and neither asks the evaluator.
    #[test]
    fn e399_e400_a_simple_constant_is_its_own_min_and_max() {
        let desc = SimpleConstantDescriptor {
            ev: EvaluatedValue(5),
        };
        assert_eq!(desc.min(), EvaluatedValue(5));
        assert_eq!(desc.max(), EvaluatedValue(5));
    }

    /// Both toggle reads sit behind `DT_CHECK(isValid())`, which is all THREE fields and not the one
    /// being read.
    #[test]
    fn e401_e402_read_the_toggle_only_while_all_three_validity_fields_stand() {
        let toggle = matched_toggle();
        assert_eq!(toggle.c1(), Some(EvaluatedValue(3)));
        assert_eq!(toggle.iter_arg_index(), Some(IterArgIndex(1)));
        // ⛔ A `c1` THAT SURVIVED A LOST LOOP IS STILL NOTHING HERE.
        let lost_loop = ToggleDescriptor {
            outer_loop: None,
            ..matched_toggle()
        };
        assert_eq!(lost_loop.c1(), None);
        assert_eq!(lost_loop.iter_arg_index(), None);
    }

    /// Both ends fold the WHOLE list, and an empty one is the invalid state that asks nothing.
    #[test]
    fn e403_e404_fold_every_constant_the_conditional_can_yield() {
        let mut evaluator = StatedEvaluator::default();
        let yielded_constants = vec![
            evaluator.constant(4096),
            evaluator.constant(64),
            evaluator.constant(8192),
        ];
        let desc = ConditionalConstantDescriptor {
            yielded_constants,
            can_be_simplified: false,
        };
        let min = desc.min(&mut evaluator);
        let max = desc.max(&mut evaluator);
        assert_eq!(min.map(|ev| evaluator.value(ev)), Some(64));
        assert_eq!(max.map(|ev| evaluator.value(ev)), Some(8192));
        let invalid = ConditionalConstantDescriptor::default();
        assert_eq!(invalid.min(&mut evaluator), None);
        assert_eq!(invalid.max(&mut evaluator), None);
    }

    /// The vendor's own arithmetic INCLUDING its one-stride-too-far last term: three terms of stride
    /// 32 from 1024 fold `1024` against `1024 + 32 * 3`, never against `1024 + 32 * 2`. A negative
    /// stride puts that same term at the MIN end, which is why both ends fold a list.
    #[test]
    fn e405_e406_fold_the_first_term_against_one_stride_past_the_last() {
        let mut evaluator = StatedEvaluator::default();
        let ascending = IntegerSequenceDescriptor {
            outer_loop: Some(ForRef(Val(7))),
            iter_arg_index: Some(IterArgIndex(1)),
            init: Some(evaluator.constant(1024)),
            stride: Some(evaluator.constant(32)),
            size: SequenceSize::Terms(3),
            can_be_simplified: false,
        };
        let min = ascending.min(&mut evaluator);
        let max = ascending.max(&mut evaluator);
        assert_eq!(min.map(|ev| evaluator.value(ev)), Some(1024));
        assert_eq!(max.map(|ev| evaluator.value(ev)), Some(1024 + 32 * 3));

        let descending = IntegerSequenceDescriptor {
            stride: Some(evaluator.constant(-32)),
            ..ascending
        };
        let min = descending.min(&mut evaluator);
        let max = descending.max(&mut evaluator);
        assert_eq!(min.map(|ev| evaluator.value(ev)), Some(1024 - 32 * 3));
        assert_eq!(max.map(|ev| evaluator.value(ev)), Some(1024));

        // ⛔ [`SequenceSize::Symbolic`] IS NOT A LENGTH: `isValid()` reads `size_ > 0`, so a symbolic
        // loop's sequence has no range at all and asks the evaluator nothing.
        let symbolic = IntegerSequenceDescriptor {
            size: SequenceSize::Symbolic,
            ..ascending
        };
        assert_eq!(symbolic.min(&mut evaluator), None);
        assert_eq!(symbolic.max(&mut evaluator), None);
    }
    /// 415/656 — the specialised `isValid()`: ⛔ a simple constant holding anything other than exactly
    /// one base address is refused, and so is every other arm.
    #[test]
    fn e415_a_simple_constant_needs_exactly_one_stored_base_address() {
        let simple = PatternDescriptor::SimpleConstant(SimpleConstantDescriptor {
            ev: EvaluatedValue(7),
        });
        assert!(transfer(Some(simple.clone()), 1).is_simple_constant());
        assert!(!transfer(Some(simple.clone()), 2).is_simple_constant());
        assert!(!transfer(Some(simple), 0).is_simple_constant());
        assert!(!transfer(None, 1).is_simple_constant());
        assert!(
            !transfer(Some(PatternDescriptor::Toggle(matched_toggle())), 2).is_simple_constant()
        );
    }

    /// 416/656 — ⭐ the descriptors go with the bits: the owning `Vec`'s clear IS e489's delete loop.
    #[test]
    fn clear_all_empties_the_descriptors_and_their_chaining_bits() {
        let mut container = DataTransferDescriptorContainer::default();
        container.descriptors.push(transfer(None, 1));
        container.set_chaining_info(DescriptorId(0), ChainFlag::PartOfChain, true);
        container.clear_all();
        assert_eq!(container, DataTransferDescriptorContainer::default());
    }

    /// 417/656 — ⭐ THE GUARD THAT DOES FIRE: the tail link of a chain carries `kHeadOfLoopingChain`
    /// (its result is what the loop yields) with `kHeadOfChain` cleared, and it is not a looping head.
    #[test]
    fn head_of_looping_chain_refuses_a_looping_tail_link() {
        let mut container = DataTransferDescriptorContainer::default();
        let tail = DescriptorId(0);
        container.set_chaining_info(tail, ChainFlag::PartOfChain, true);
        container.set_chaining_info(tail, ChainFlag::HeadOfLoopingChain, true);
        assert!(!container.is_head_of_looping_chain(tail));
        // The same bits on the chain's head, which is what `computeChainingInfo` gives a one-link one.
        container.set_chaining_info(tail, ChainFlag::HeadOfChain, true);
        assert!(container.is_head_of_looping_chain(tail));
    }

    /// 418/656 — `old_immutable - pinned_addr`, built into the `const_builder`'s block.
    #[test]
    fn e418_the_offset_is_the_stored_base_addr_minus_the_pinned_one() {
        let mut evaluator = StatedEvaluator::default();
        let base_addr = evaluator.constant(4096);
        let pinned = evaluator.constant(1024);
        let mut dtd = transfer(
            Some(PatternDescriptor::SimpleConstant(
                SimpleConstantDescriptor { ev: base_addr },
            )),
            1,
        );
        dtd.base_addrs = vec![base_addr];
        let mut consts = Vec::new();
        let mut values = Values::default();
        let mut walked = Vec::new();
        let offset = {
            let mut sites = OffsetSites {
                consts: &mut consts,
                query_maps: None,
                values: &mut values,
            };
            SimpleConstantDataTransferUpdater.get_offset(
                &dtd,
                pinned,
                ScalarTy::Index,
                &mut evaluator,
                &mut sites,
                &mut walked,
            )
        };
        assert_eq!(offset, Val(0));
        assert_eq!(consts, vec![scalar_const(0, 4096 - 1024)]);
        assert!(walked.is_empty());
    }

    /// 419/656 — the pre-order walk reaches a region op nested in a loop, and ⭐ the constants e275
    /// built land at the FRONT of the preamble in creation order: one `const_builder`, positioned at
    /// the enclosing function's entry block before the walk starts.
    #[test]
    fn a_constant_hbm_addr_inside_a_loop_is_query_mapped_and_its_constants_head_the_preamble() {
        let mut preamble = vec![
            get_unit(1, DfirUnit::Hbm),
            get_unit(2, DfirUnit::Hbm),
            get_unit(3, DfirUnit::Pe),
            scalar_const(4, 64),
        ];
        let mut unit_body = vec![Op::Sentient(sentient::Op::For {
            iv: Val(11),
            bound: Val(12),
            bound_reg: None,
            carried: Vec::new(),
            dbg_name: None,
            body: vec![Op::UniformRegions(UniformRegions::UniformizeRegions {
                regions: vec![LocalRegion {
                    arg: Val(5),
                    units: vec![Val(1), Val(2)],
                    body: vec![load_and_store(1, 3, 4, 6)],
                }],
                results: Vec::new(),
                yielded: Vec::new(),
            })],
        })];
        let mut values = Values::default();
        for _ in 0..13 {
            let _ = values.mint();
        }

        turn_hbm_constant_op_addrs_to_query_maps_in_uniform_regions(
            &mut unit_body,
            &mut preamble,
            &mut values,
        );

        assert_eq!(preamble[0], scalar_const(13, 64));
        assert_eq!(preamble[1], scalar_const(14, 64));
        assert_eq!(preamble[2], get_unit(1, DfirUnit::Hbm));
        let Op::Sentient(sentient::Op::For { body, .. }) = &unit_body[0] else {
            panic!("the loop the walk descended into")
        };
        let Op::UniformRegions(region_op) = &body[0] else {
            panic!("the region op the walk handed to e275")
        };
        // The mapping, its query and the transfer now reading the queried address.
        assert_eq!(region_op.regions()[0].body[2], load_and_store(1, 3, 16, 6));
    }

    /// 485/656 — an invalid toggle answers nothing and never reaches the evaluator, which
    /// [`OutOfScopeEvaluator`] proves by panicking if it is asked.
    #[test]
    fn e485_an_invalid_toggle_has_no_x() {
        let empty: [&[Op]; 1] = [&[]];
        assert_eq!(
            ToggleDescriptor::default().x(
                &mut OutOfScopeEvaluator,
                &[],
                Definitions::from_innermost(&empty)
            ),
            None
        );
    }

    /// 485/656 — a valid toggle subtracts its `init` from `c1`, so the seam it stops at is the
    /// evaluation of the iter arg's constant init.
    #[test]
    #[should_panic(expected = "evaluateValue")]
    fn e485_a_valid_toggle_reaches_the_evaluation_of_its_init() {
        let body = vec![
            scalar_const(1, 4096),
            Op::Sentient(sentient::Op::For {
                iv: Val(2),
                bound: Val(0),
                bound_reg: None,
                carried: vec![sentient::Carried {
                    init: Val(1),
                    arg: Val(3),
                    result: Val(4),
                    reg: Reg {
                        locale: RegType::Lbr,
                        index: None,
                    },
                    program_header: false,
                    element_size: None,
                }],
                dbg_name: None,
                body: Vec::new(),
            }),
        ];
        let toggle = ToggleDescriptor {
            outer_loop: Some(ForRef(Val(2))),
            iter_arg_index: Some(IterArgIndex(0)),
            c1: Some(EvaluatedValue(5)),
            can_be_simplified: false,
        };
        let regions: [&[Op]; 1] = [&body];
        let _x = toggle.x(
            &mut OutOfScopeEvaluator,
            &body,
            Definitions::from_innermost(&regions),
        );
    }

    /// 486+487/656 — both getters answer the pattern a simple-constant transfer holds, and neither
    /// tolerates another pattern.
    #[test]
    fn e486_e487_the_simple_constant_getters_answer_the_pattern_and_refuse_any_other() {
        let mut desc = transfer(
            Some(PatternDescriptor::SimpleConstant(SimpleConstantDescriptor {
                ev: EvaluatedValue(9),
            })),
            1,
        );
        assert_eq!(desc.simple_constant_descriptor().ev, EvaluatedValue(9));
        desc.simple_constant_descriptor_mut().ev = EvaluatedValue(10);
        assert_eq!(desc.simple_constant_descriptor().ev, EvaluatedValue(10));

        let toggled = transfer(Some(PatternDescriptor::Toggle(matched_toggle())), 1);
        let refused = std::panic::catch_unwind(move || toggled.simple_constant_descriptor().ev);
        assert!(refused.is_err());
    }

    /// 488/656 — the count is the number of DISTINCT base-address handles the immutable descriptors
    /// name, and it is memoised: a descriptor added afterwards does not change it.
    #[test]
    fn e488_counts_distinct_base_addresses_once_and_remembers_the_answer() {
        let mut pass = AddressPinningAndTogglePass::default();
        // Two transfers whose base addresses are the SAME two handles: two streams, not four.
        pass.immut_data_transfer_descriptors.insert(transfer(None, 2));
        pass.immut_data_transfer_descriptors.insert(transfer(None, 2));
        assert_eq!(pass.compute_or_get_number_of_streams(), StreamCount(2));

        pass.immut_data_transfer_descriptors
            .insert(transfer(None, 5));
        assert_eq!(pass.compute_or_get_number_of_streams(), StreamCount(2));
    }

    /// 489/656 — every accumulated field is dropped, so the next program unit recomputes its stream
    /// count instead of reading this one's.
    #[test]
    fn e489_cleanup_drops_the_count_both_containers_and_the_address_info() {
        let mut pass = AddressPinningAndTogglePass::default();
        pass.immut_data_transfer_descriptors.insert(transfer(None, 1));
        pass.mut_data_transfer_descriptors.insert(transfer(None, 1));
        pass.ev_addr_info_list.push(EvAddressInfo {
            ba_immut_addr_ev: EvaluatedValue(1),
            ba_max_mut_addr_ev: EvaluatedValue(2),
            is_toggle: true,
            region: RegionSite::default(),
        });
        let _ = pass.compute_or_get_number_of_streams();

        pass.cleanup();

        assert_eq!(pass.num_streams, None);
        assert!(pass.immut_data_transfer_descriptors.descriptors.is_empty());
        assert!(pass.mut_data_transfer_descriptors.descriptors.is_empty());
        assert!(pass.ev_addr_info_list.is_empty());
    }

    /// A model and a rung, so a program is typed; nothing this pass reads is on either.
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

    /// The loop [`matched_toggle`] names: `%7` is its induction variable, and the iter arg at index 1
    /// starts at the constant `%1` — which is the `Y` both ends of the pair fold `X` against.
    fn toggled_loop() -> Vec<Op> {
        let carried = |init, arg, result| sentient::Carried {
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
        vec![
            scalar_const(1, 4096),
            Op::Sentient(sentient::Op::For {
                iv: Val(7),
                bound: Val(0),
                bound_reg: None,
                carried: vec![
                    carried(Val(0), Val(8), Val(9)),
                    carried(Val(1), Val(10), Val(11)),
                ],
                dbg_name: None,
                body: Vec::new(),
            }),
        ]
    }

    /// 547/656 — the lower end is folded from `X` and `Y`, and `getX()` resolves `getInit()` on the way:
    /// the loop's own constant initializer, whose EVALUATION is the out-of-scope seam this stops at.
    #[test]
    #[should_panic(expected = "constant iter arg init Val(1)")]
    fn e547_min_folds_x_against_the_loops_constant_init() {
        let body = toggled_loop();
        let regions: [&[Op]; 1] = [&body];
        let _min = matched_toggle().min(
            &mut OutOfScopeEvaluator,
            &body,
            Definitions::from_innermost(&regions),
        );
    }

    /// 548/656 — `DT_CHECK(isValid())` (`:236`, `:241`) IS the `None`, and it is one encoding for both
    /// ends: an invalidated toggle answers neither a max nor a min, and asks the evaluator nothing.
    #[test]
    fn e548_max_and_min_answer_nothing_for_an_invalidated_toggle() {
        let mut toggle = matched_toggle();
        toggle.invalidate();
        let regions: [&[Op]; 1] = [&[]];
        let defs = Definitions::from_innermost(&regions);
        assert_eq!(toggle.max(&mut OutOfScopeEvaluator, &[], defs), None);
        assert_eq!(toggle.min(&mut OutOfScopeEvaluator, &[], defs), None);
    }

    /// 549/656 — the module walk clears the pass's accumulated state and hands each program unit to the
    /// per-unit `runOn`, which is entry 656 and not this worklist's.
    #[test]
    #[should_panic(expected = "e656_runOn(dataflow::ProgramUnitOp)")]
    fn e549_hands_each_program_unit_to_the_per_unit_pass() {
        let mut program: Program<Dd2, AnyModel, AnyRung> = Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(DfirUnit::L0lu, Val(100)),
                    precision: None,
                    body: Vec::new(),
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            bound: core::marker::PhantomData,
        };

        AddressPinningAndTogglePass::default().run_on_program(&mut program);
    }
}

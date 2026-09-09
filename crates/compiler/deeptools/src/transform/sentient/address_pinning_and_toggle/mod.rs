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

use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::Val;
use crate::transform::sentient::analyses::EvaluatedValue;

pub use conditional_constant_descriptor::ConditionalConstantDescriptor;
pub use data_transfer_descriptor::DataTransferDescriptor;
pub use data_transfer_descriptor_container::{
    ChainFlag, ChainFlags, DataTransferDescriptorContainer, DescriptorId,
};
pub use discrete_integer_set_descriptor::DiscreteIntegerSetDescriptor;
pub use integer_sequence_descriptor::{IntegerSequenceDescriptor, SequenceSize};
pub use looping_chain_mutable_addr_descriptor::{ChainSize, LoopingChainMutableAddrDescriptor};
pub use simple_constant_descriptor::SimpleConstantDescriptor;
pub use toggle_descriptor::ToggleDescriptor;

/// THE CONSTANT BASE ADDRESSES ONE TRANSFER CAN USE — `using BaseAddrListTy =
/// llvm::SmallVector<const EvaluatedValue *, 2>` (`AddressPinningAndToggle.cpp:157`).
pub type BaseAddrList = Vec<EvaluatedValue>;

/// WHICH PATTERN A BASE ADDRESS FOLLOWS — the `DynamicPatternDescriptorBase *pattern_desc_`
/// hierarchy (`AddressPinningAndToggle.cpp:100-628`), whose six subclasses the reference
/// discriminates with `isa<>` and its `PatternKind` tag.
///
/// ⭐ AN ENUM BECAUSE THE SET IS CLOSED AND MUTUALLY EXCLUSIVE — the reference says so at `:855`
/// ("the descriptor types should be mutually exclusive"), which is why a non-looping chain gets no
/// arm here and lives on `HBMDataTransferDescriptor::total_chain_increment_` instead.
///
/// ⛔ NO `kUnknown` ARM: `PatternKind::kUnknown` (`:104`) is the base class's default tag and no
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
    /// (`AddressPinningAndToggle.cpp:257`), so the toggle reads as unmatched.
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
    /// Clears everything `IntegerSequenceDescriptor::isValid()` reads (`:392`).
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
    /// descriptor's ADDRESS (`:955`), which is not expressible while this container owns them.
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

/// `SimpleConstantDataTransferUpdater` (`:1046`) — the updater for a transfer whose base address is
/// one constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimpleConstantDataTransferUpdater;

impl SimpleConstantDataTransferUpdater {
    /// Replaces: e010_updateVariableOffsetCalculation
    ///
    /// `return;  // nothing to do` (`:1063-1066`) — a simple constant has no toggle `scalar_sub` and no
    /// `sentient.if` whose yielded constants would need re-basing against the pinned address, so this
    /// override of the three that do is empty.
    pub const fn update_variable_offset_calculation(self, _new_immut_addr_ev: EvaluatedValue) {}
}

/// `SubOp toggle_sub_` — the `sentient.scalar_sub` computing a toggling transfer's immutable address,
/// named by its `$out`.
///
/// ⛔ A TYPE AND NOT A BARE [`Val`]: `DT_CHECK_MSG(toggle_sub, …)` (`:1528-1532`) is that
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
}

/// `ToggleDataTransferUpdater` (`:1080`) — the updater for a base address that toggles between two
/// constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToggleDataTransferUpdater {
    /// `toggle_sub_`, from `immutable_addr[0].get().getDefiningOp<SubOp>()` (`:1528`).
    pub toggle_sub: ToggleSub,
}

impl ToggleDataTransferUpdater {
    /// Replaces: e011_getOffset
    ///
    /// The toggle's own `scalar_sub` result IS the fall-back offset (`:1132-1135`) — nothing new is
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
/// result (`applyToAllYields(if_op_, …, res_index_)`, `:2071-2086`).
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

/// `ConditionalConstDataTransferUpdater` (`:1112`) — the updater for a base address an `scf`-style
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
    /// The carried iter arg IS the fall-back offset (`:1162-1165`) — the sequence's own induction
    /// already steps by what the pinning wants added.
    ///
    /// ⭐ `DT_CHECK(iter_arg_)` is the field's existence; `new_immut_addr_ev` is UNREAD.
    #[must_use]
    pub const fn get_offset(self, _new_immut_addr_ev: EvaluatedValue) -> Val {
        self.iter_arg
    }
}

/// WHAT e014 NEEDS OF A DESCRIPTOR — `DataTransferDescriptor::dump` is **e492** (`:2521`, level 3,
/// `data_transfer_descriptor.rs`), a later unit.
///
/// ⚠️ `UNITS.tsv` RECORDS e014's CALLS AS `-`: the call resolves through a virtual `dump()` the
/// extractor's detector did not follow. This trait is the seam e492 lands into later.
pub trait DumpDescriptor {
    /// `dtd->dump()` (`:2521`) — one descriptor's block, `----------` delimited.
    fn dump(&self) -> String;
}

/// Replaces: e014_dump
///
/// The immutable descriptors under one header then the mutable ones under another, each container in
/// INSERTION order (`DataTransferDescriptorContainer` exposes the base `std::vector`'s `begin`/`end`,
/// not `sorted_list_`).
///
/// ⭐ THE INNER `LLVM_DEBUG` ON THE TWO HEADERS CHANGES NOTHING: both callsites already wrap the whole
/// call (`:1349-1353`, `:1370-1375`), so a returned `String` loses no gating the reference had.
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

// crustify:todo: e257_getBaseAddr
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:657  (4 body lines, level 1)
//   original  : const EvaluatedValue &getBaseAddr() const
//   calls     : e252_size

// crustify:todo: e258_isToggle
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:677  (4 body lines, level 1)
//   original  : bool isToggle() const
//   calls     : e278_isValid

// crustify:todo: e259_isConditionalConstant
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:681  (4 body lines, level 1)
//   original  : bool isConditionalConstant() const
//   calls     : e278_isValid

// crustify:todo: e260_isIntegerSequence
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:685  (4 body lines, level 1)
//   original  : bool isIntegerSequence() const
//   calls     : e278_isValid

// crustify:todo: e261_isDiscreteIntegerSet
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:689  (4 body lines, level 1)
//   original  : bool isDiscreteIntegerSet() const
//   calls     : e278_isValid

// crustify:todo: e262_isLoopingChainMutableAddr
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:693  (4 body lines, level 1)
//   original  : bool isLoopingChainMutableAddr() const
//   calls     : e278_isValid

// crustify:todo: e263_getToggleDescriptor
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:706  (4 body lines, level 1)
//   original  : ToggleDescriptor &getToggleDescriptor()
//   calls     : e258_isToggle

// crustify:todo: e264_getToggleDescriptor
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:710  (4 body lines, level 1)
//   original  : const ToggleDescriptor &getToggleDescriptor() const
//   calls     : e258_isToggle

// crustify:todo: e265_getConditionalConstantDescriptor
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:714  (4 body lines, level 1)
//   original  : ConditionalConstantDescriptor &getConditionalConstantDescriptor()
//   calls     : e259_isConditionalConstant

// crustify:todo: e266_getConditionalConstantDescriptor
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:718  (4 body lines, level 1)
//   original  : const ConditionalConstantDescriptor &getConditionalConstantDescriptor() const
//   calls     : e259_isConditionalConstant

// crustify:todo: e267_getIntegerSequenceDescriptor
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:723  (4 body lines, level 1)
//   original  : IntegerSequenceDescriptor &getIntegerSequenceDescriptor()
//   calls     : e260_isIntegerSequence

// crustify:todo: e268_getIntegerSequenceDescriptor
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:727  (4 body lines, level 1)
//   original  : const IntegerSequenceDescriptor &getIntegerSequenceDescriptor() const
//   calls     : e260_isIntegerSequence

// crustify:todo: e269_getDiscreteIntegerSetDescriptor
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:731  (4 body lines, level 1)
//   original  : DiscreteIntegerSetDescriptor &getDiscreteIntegerSetDescriptor()
//   calls     : e261_isDiscreteIntegerSet

// crustify:todo: e270_getDiscreteIntegerSetDescriptor
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:735  (4 body lines, level 1)
//   original  : const DiscreteIntegerSetDescriptor &getDiscreteIntegerSetDescriptor() const
//   calls     : e261_isDiscreteIntegerSet

// crustify:todo: e271_getLoopingChainMutableAddrDescriptor
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:739  (4 body lines, level 1)
//   original  : LoopingChainMutableAddrDescriptor &getLoopingChainMutableAddrDescriptor()
//   calls     : e262_isLoopingChainMutableAddr

// crustify:todo: e272_getLoopingChainMutableAddrDescriptor
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:743  (4 body lines, level 1)
//   original  : const LoopingChainMutableAddrDescriptor & getLoopingChainMutableAddrDescriptor() const
//   calls     : e262_isLoopingChainMutableAddr

// crustify:todo: e273_isHeadOfChain
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:907  (6 body lines, level 1)
//   original  : bool isHeadOfChain(const DataTransferDescriptor &desc) const
//   calls     : e007_isPartOfSomeChain

// crustify:todo: e274_validate
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:929  (4 body lines, level 1)
//   original  : void validate() const
//   calls     : e252_size

// crustify:todo: e275_turnHBMConstantOpAddrsToQueryMapsHelper
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1595  (59 body lines, level 1)
//   original  : void AddressPinningAndTogglePass::turnHBMConstantOpAddrsToQueryMapsHelper( OpBuilder &const_builder, Operation &region_op)
//   calls     : e252_size

// crustify:todo: e399_getMin
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:182  (4 body lines, level 2)
//   original  : const EvaluatedValue &getMin() override
//   calls     : e278_isValid

// crustify:todo: e400_getMax
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:187  (4 body lines, level 2)
//   original  : const EvaluatedValue &getMax() override
//   calls     : e278_isValid

// crustify:todo: e401_getC1
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:246  (4 body lines, level 2)
//   original  : const EvaluatedValue &getC1() const
//   calls     : e278_isValid

// crustify:todo: e402_getIterArgIndex
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:251  (4 body lines, level 2)
//   original  : int getIterArgIndex() const
//   calls     : e278_isValid

// crustify:todo: e403_getMin
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:320  (4 body lines, level 2)
//   original  : const EvaluatedValue &getMin() override
//   calls     : e278_isValid

// crustify:todo: e404_getMax
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:325  (4 body lines, level 2)
//   original  : const EvaluatedValue &getMax() override
//   calls     : e278_isValid

// crustify:todo: e405_getMin
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:360  (13 body lines, level 2)
//   original  : const EvaluatedValue &getMin() override
//   calls     : e278_isValid

// crustify:todo: e406_getMax
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:374  (13 body lines, level 2)
//   original  : const EvaluatedValue &getMax() override
//   calls     : e278_isValid

// crustify:todo: e407_getInit
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:388  (4 body lines, level 2)
//   original  : const EvaluatedValue &getInit()
//   calls     : e278_isValid

// crustify:todo: e408_getAllConstants
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:393  (16 body lines, level 2)
//   original  : void getAllConstants(BaseAddrListTy &output) const
//   calls     : e278_isValid

// crustify:todo: e409_getMin
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:470  (5 body lines, level 2)
//   original  : const EvaluatedValue &getMin() override
//   calls     : e278_isValid

// crustify:todo: e410_getMax
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:476  (5 body lines, level 2)
//   original  : const EvaluatedValue &getMax() override
//   calls     : e278_isValid

// crustify:todo: e411_getInit
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:482  (4 body lines, level 2)
//   original  : const EvaluatedValue &getInit()
//   calls     : e278_isValid

// crustify:todo: e412_getMin
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:566  (7 body lines, level 2)
//   original  : const EvaluatedValue &getMin() override
//   calls     : e278_isValid

// crustify:todo: e413_getMax
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:574  (7 body lines, level 2)
//   original  : const EvaluatedValue &getMax() override
//   calls     : e278_isValid

// crustify:todo: e414_getInit
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:582  (4 body lines, level 2)
//   original  : const EvaluatedValue &getInit() const
//   calls     : e278_isValid

// crustify:todo: e415_isSimpleConstant
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:673  (4 body lines, level 2)
//   original  : bool isSimpleConstant() const
//   calls     : e278_isValid

// crustify:todo: e416_clear_all
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:886  (6 body lines, level 2)
//   original  : void clear_all()
//   calls     : e274_validate

// crustify:todo: e417_isHeadOfLoopingChain
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:920  (7 body lines, level 2)
//   original  : bool isHeadOfLoopingChain(const DataTransferDescriptor &desc) const
//   calls     : e273_isHeadOfChain

// crustify:todo: e418_getOffset
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1073  (7 body lines, level 2)
//   original  : Value getOffset(const EvaluatedValue &new_immut_addr_ev) override
//   calls     : e009_createOffsetValue, e257_getBaseAddr

// crustify:todo: e419_turnHBMConstantOpAddrsToQueryMapsInUniformRegions
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1579  (13 body lines, level 2)
//   original  : void AddressPinningAndTogglePass:: turnHBMConstantOpAddrsToQueryMapsInUniformRegions( dataflow::ProgramUnitOp unit)
//   calls     : e275_turnHBMConstantOpAddrsToQueryMapsHelper

// crustify:todo: e485_getX
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:227  (5 body lines, level 3)
//   original  : const EvaluatedValue &getX() const
//   calls     : e015_getInit, e278_isValid, e407_getInit, e411_getInit, e414_getInit

// crustify:todo: e486_getSimpleConstantDescriptor
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:698  (4 body lines, level 3)
//   original  : SimpleConstantDescriptor &getSimpleConstantDescriptor()
//   calls     : e415_isSimpleConstant

// crustify:todo: e487_getSimpleConstantDescriptor
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:702  (4 body lines, level 3)
//   original  : const SimpleConstantDescriptor &getSimpleConstantDescriptor() const
//   calls     : e415_isSimpleConstant

// crustify:todo: e488_computeOrGetNumberOfStreams
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1439  (10 body lines, level 3)
//   original  : int AddressPinningAndTogglePass::computeOrGetNumberOfStreams()
//   calls     : e252_size, e422_insert

// crustify:todo: e489_cleanup
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1656  (9 body lines, level 3)
//   original  : void AddressPinningAndTogglePass::cleanup()
//   calls     : e416_clear_all

// crustify:todo: e547_getMin
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:235  (4 body lines, level 4)
//   original  : const EvaluatedValue &getMin() override
//   calls     : e278_isValid, e485_getX

// crustify:todo: e548_getMax
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:240  (4 body lines, level 4)
//   original  : const EvaluatedValue &getMax() override
//   calls     : e278_isValid, e485_getX

// crustify:todo: e549_runOn
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1187  (12 body lines, level 4)
//   original  : void runOn(ModuleOp module_op)
//   calls     : e489_cleanup

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
    use crate::islands::sentient::dialects::{Val, sentient};
    use crate::transform::sentient::{ForRef, IterArgIndex};

    /// A matched-looking descriptor field set, so an `invalidate()` has something to clear.
    fn loop_and_arg() -> (Option<ForRef>, Option<IterArgIndex>) {
        (Some(ForRef(Val(7))), Some(IterArgIndex(1)))
    }

    #[test]
    fn toggle_invalidate_clears_all_three_validity_fields() {
        let (outer_loop, iter_arg_index) = loop_and_arg();
        let mut desc = ToggleDescriptor {
            outer_loop,
            iter_arg_index,
            c1: Some(EvaluatedValue(3)),
        };
        desc.invalidate();
        assert_eq!(desc, ToggleDescriptor::default());
    }

    #[test]
    fn get_all_constants_appends_in_order_without_clearing() {
        let desc = ConditionalConstantDescriptor {
            yielded_constants: vec![EvaluatedValue(11), EvaluatedValue(22)],
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
        container
            .descriptors
            .push(DataTransferDescriptor::default());
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
}

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

//! `AddressPinningAndToggle.cpp` — 4 of the campaign's 656 units (dependency level(s) [1, 3, 5]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e278_isValid` | 278 | 1 | 23 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2478` |
//! | `e279_canBeSimplified` | 279 | 1 | 18 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2502` |
//! | `e492_dump` | 492 | 3 | 28 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2521` |
//! | `e593_initializeDescriptor` | 593 | 5 | 161 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2316` |


use super::hbm_data_transfer_descriptor::{BurstIncrement, ChainIncrement};
use super::data_transfer_descriptor_container::mutable_addr_end;
use super::{
    ASSERT_ON_UNEXPECTED_PATTERNS, BaseAddrList, ConditionalConstantDescriptor,
    DiscreteIntegerSetDescriptor, HANDLE_CONDITIONAL_CONSTANTS, IntegerSequenceDescriptor,
    LoopingChainMutableAddrDescriptor, PatternDescriptor, SimpleConstantDescriptor, ToggleDescriptor,
    op_at, write_evaluated_value,
};
use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;
use crate::islands::sentient::dialects::{Definitions, Op, Val, sentient, symbol, uniform};
use crate::islands::sentient::print;
use crate::transform::sentient::analyses::{ExpressionEvaluator, RegionSite};
use crate::transform::sentient::utils::{ConstKind, is_constant};
use crate::units::DfirUnit;

/// WHICH MEMORY THIS DESCRIPTOR'S ADDRESS LIVES IN — the pure virtual `getMemoryUnit()` (`:671`),
/// answered `LX` by `LXDataTransferDescriptor` (`:806`) and `HBM` by `HBMDataTransferDescriptor`
/// (`:827`).
///
/// ⭐ A FIELD, NOT TWO TYPES: `collectDataTransfers` (`:1400-1435`) pushes BOTH subclasses into the
/// one `immut_data_transfer_descriptors_`, so descriptors in a single container disagree about it.
/// ⛔ COMPARE [`Self::dfir_unit`], NEVER THIS ENUM: the `Hbm` arm carries the HBM subclass's own two
/// private fields (`:861-863`), so two HBM descriptors are `!=` here while `getMemoryUnit()` says
/// `HBM` for both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DescriptorMemoryUnit {
    /// `LXDataTransferDescriptor` (`:806`), which adds no state of its own.
    #[default]
    Lx,
    /// `HBMDataTransferDescriptor` (`:827`) and the two increments only its overrides read.
    Hbm {
        /// `total_chain_increment_` (`:861`), the reference's `= 0` default.
        total_chain_increment: ChainIncrement,
        /// `increment_via_burst_` (`:863`), the reference's `= 0` default.
        increment_via_burst: BurstIncrement,
    },
}

impl DescriptorMemoryUnit {
    /// The island's spelling of the same unit, which `getMutableAddrResultIndex` compares against
    /// `dcc::getUnitType(..)` of each address operand (`Analyses/Utils.cpp:544-549`).
    #[must_use]
    pub const fn dfir_unit(self) -> DfirUnit {
        match self {
            Self::Lx => DfirUnit::Lx,
            Self::Hbm { .. } => DfirUnit::Hbm,
        }
    }
}

/// ONE DATA TRANSFER'S BASE-ADDRESS STORY — `class DataTransferDescriptor`
/// (`AddressPinningAndToggle.cpp:632-790`), one per `load_and_send`/`receive_and_store`/
/// `load_and_store` per address role (the HBM `load_and_store` case makes TWO, `:1410-1415`).
///
/// ⛔ NOT `evaluator_`: the reference stores `ExpressionEvaluator &`, one global the pass threads
/// through. A shared borrow in a field would make the descriptor unstorable while the pass rewrites
/// the IR, so the ported methods take the evaluator as an argument instead.
///
/// ⛔ NO `Default`, BECAUSE `DataTransferDescriptor() = delete` (`:634`): there is no transfer without
/// the op it describes, and `op` below has no meaningful default — the empty [`OpId`] path names a
/// BLOCK, not an op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataTransferDescriptor {
    /// `op_` — WHICH transfer this describes, the reference's `Operation &op_` (`:783`) behind the
    /// `getOperation()` pair (`:666-667`, both `EXCLUSIONS.tsv` field accessors).
    ///
    /// ⭐ A POSITION, NOT A POINTER: [`OpId`] is this crate's stand-in for `Operation *`, and it is
    /// the key [`super::DataTransferDescriptorContainer::lookup`] answers about — the reference's
    /// `sorted_list_` orders the very same identity by raw address.
    pub op: OpId,
    /// Replaces: e006_dtor_DataTransferDescriptor
    ///
    /// `pattern_desc_` — which recognised pattern this transfer follows, `None` for the reference's
    /// `nullptr` (every `is*()` tests it, `:673-696`).
    ///
    /// ⭐ THIS FIELD *IS* THE PORT OF `~DataTransferDescriptor()` (`:645-647`, whose whole body is
    /// `if (pattern_desc_) delete pattern_desc_;`): the destructor exists only because the reference
    /// holds `DynamicPatternDescriptorBase *` from a `new` in `initializeDescriptor`. An owned
    /// `Option` frees exactly that, at exactly that point, so ⛔ there is no `impl Drop` to write —
    /// a hand-written one restating the field's own drop would free nothing extra.
    pub pattern_desc: Option<PatternDescriptor>,
    /// `base_addrs_` — the possible constant values `base_addr_` can take (one, or two for a
    /// toggle); the list [`super::ConditionalConstantDescriptor::get_all_constants`] appends into.
    pub base_addrs: BaseAddrList,
    /// `region_op_` and `region_num_` as one value — `getRegionOp()` (`:760`) and `getRegionNum()`,
    /// both excluded field accessors, and never read apart: every caller pairs them for
    /// `findClosestPinnedAddr` (`:1897`, `:1937`, `:2054`, `:2120`). See [`RegionSite`].
    pub region: RegionSite,
    /// `getMemoryUnit()` — which of the two subclasses this descriptor is (`:806`, `:827`), read by
    /// every `getMutableAndImmutableAddr(op, getMemoryUnit())` call. See [`DescriptorMemoryUnit`].
    pub memory_unit: DescriptorMemoryUnit,
    /// `base_addr_` — the ORIGINAL SSA value carrying this transfer's address (`:779-781`), which the
    /// subclass constructors take from `getMutableAndImmutableAddr(op, getMemoryUnit())` (`:2573`,
    /// `:2591-2596`) and `getOriginalBaseAddrSSA()` (`:663-664`) hands back.
    ///
    /// ⛔ [`Self::initialize_descriptor`] REWRITES IT, through the `Value &` it takes (`:2317`): the
    /// value stored afterwards is the one at the far end of the `sentient.scalar_copy` chain, and
    /// `AbstractDataTransferUpdater::update` compares two descriptors' stored values (`:1775-1776`).
    pub base_addr: Val,
    /// `is_base_addr_mutable_` (`:787`) — whether [`Self::base_addr`] is the transfer's MUTABLE
    /// address rather than its immutable one, the base constructor's own defaulted parameter
    /// (`:637`, `= false`) that only `HBMDataTransferDescriptor` ever passes `true` (`:1414`,
    /// `:1426`).
    pub is_base_addr_mutable: bool,
}

impl DataTransferDescriptor {
    /// Replaces: e278_isValid
    ///
    /// Whether this transfer follows a recognised pattern: no base addresses is invalid, a matched
    /// pattern answers for itself, and anything else needs exactly one base address (`:2478-2500`).
    ///
    /// ⛔ THE TOGGLE ARM IS THE ONLY ONE THAT COUNTS BASE ADDRESSES (`:2483-2486`): two of them, or
    /// one when the toggle simplified away.
    /// ⛔ THERE IS NO `SimpleConstant` ARM IN THE REFERENCE'S `dyn_cast` CHAIN — a simple constant
    /// falls through to the `base_addrs.size() == 1` tail, exactly as `pattern_desc_ == nullptr` does.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        if self.base_addrs.is_empty() {
            return false;
        }
        match &self.pattern_desc {
            Some(PatternDescriptor::Toggle(toggle)) => {
                toggle.is_valid()
                    && ((toggle.can_be_simplified && self.base_addrs.len() == 1)
                        || self.base_addrs.len() == 2)
            }
            Some(PatternDescriptor::ConditionalConstant(cc)) => cc.is_valid(),
            Some(PatternDescriptor::IntegerSequence(isq)) => isq.is_valid(),
            Some(PatternDescriptor::DiscreteIntegerSet(dis)) => dis.is_valid(),
            Some(PatternDescriptor::LoopingChainMutableAddr(lcma)) => lcma.is_valid(),
            Some(PatternDescriptor::SimpleConstant(_)) | None => self.base_addrs.len() == 1,
        }
    }

    /// Replaces: e279_canBeSimplified
    ///
    /// Whether the matched pattern collapsed to a single constant base address — the pattern's own
    /// `can_be_simplified_`, and `false` for no pattern at all (`:2502-2519`).
    ///
    /// ⛔ EVERY `is*()` OPENS WITH `isValid()` (`:673-696`), so an invalid descriptor answers `false`
    /// however its pattern's own flag stands.
    /// ⛔ A SIMPLE CONSTANT ANSWERS `false` DESPITE ITS OWN `setCanBeSimplified(true)` (`:169`):
    /// `isSimpleConstant()` is absent from this `else if` chain, and nothing reads that flag here.
    #[must_use]
    pub fn can_be_simplified(&self) -> bool {
        let res = self.is_valid()
            && match &self.pattern_desc {
                Some(PatternDescriptor::Toggle(toggle)) => toggle.can_be_simplified,
                Some(PatternDescriptor::ConditionalConstant(cc)) => cc.can_be_simplified,
                Some(PatternDescriptor::IntegerSequence(isq)) => isq.can_be_simplified,
                Some(PatternDescriptor::DiscreteIntegerSet(dis)) => dis.can_be_simplified,
                Some(PatternDescriptor::LoopingChainMutableAddr(lcma)) => lcma.can_be_simplified,
                Some(PatternDescriptor::SimpleConstant(_)) | None => false,
            };
        // `DT_CHECK_MSG((!res || getBaseAddrList().size() == 1), ..)` (`:2514-2518`) — an ABORT in the
        // reference, so it stays a named stop rather than becoming a refusal.
        if res && self.base_addrs.len() != 1 {
            todo!(
                "DataTransferDescriptor::canBeSimplified: DT_CHECK_MSG(!res || \
                 getBaseAddrList().size() == 1, \"simplified pattern should have a single \
                 base_addr stored in the descriptor\") — {} base addrs (:2514-2518)",
                self.base_addrs.len()
            )
        }
        res
    }

    /// Replaces: e492_dump
    ///
    /// One transfer's record between two `----------` rules: the op, its base addresses, then its
    /// matched pattern's own dump (`:2521-2548`).
    ///
    /// ⛔ NOT [`super::DumpDescriptor`], WHOSE BARE `&self` CANNOT REACH EITHER ARGUMENT: `os << op_`
    /// needs the body [`Self::op`] indexes, and the integer-sequence arm's `getAllConstants` needs
    /// the evaluator. e014's caller has both.
    /// ⛔ TRAP: `getLocation(..).getLine()` HAS NOTHING TO READ — no op on any rung of this crate's
    /// islands carries a source location, so the line prints `?`, which is also what the reference's
    /// own `FileLineColLoc::get(ctx, "unknown", -1, -1)` fallback means (`Utils/Utils.cpp:43-55`).
    #[must_use]
    pub fn dump(&self, unit_body: &[Op], evaluator: &mut impl ExpressionEvaluator) -> String {
        const LINE: &str = "----------";
        let mut out = String::from(LINE);
        out.push_str("\n* operation:line:?:");
        match op_at(&self.op, unit_body) {
            // `print::emit` ends that line, which is the reference's `<< op_ << "\n"`.
            Some(op) => print::emit(&mut out, op, 0),
            None => out.push('\n'),
        }
        out.push_str("* base addrs:[\n");
        for (index, ev) in self.base_addrs.iter().enumerate() {
            if index > 0 {
                out.push_str(";\n");
            }
            out.push_str("  ");
            write_evaluated_value(Some(*ev), &mut out);
        }
        out.push_str("\n]\n");
        // The `is*()` chain (`:2536-2547`): each already tests its own variant, so the guards are
        // that chain and the fallthrough is its `else`.
        match &self.pattern_desc {
            Some(PatternDescriptor::SimpleConstant(sc)) if self.is_simple_constant() => {
                out.push_str(&sc.dump());
            }
            Some(PatternDescriptor::Toggle(_)) if self.is_toggle() => {
                todo!(
                    "ToggleDescriptor::dump (e553, AddressPinningAndToggle.cpp:2724) — a later \
                     unit of this campaign"
                )
            }
            Some(PatternDescriptor::ConditionalConstant(cc)) if self.is_conditional_constant() => {
                out.push_str(&cc.dump());
            }
            Some(PatternDescriptor::IntegerSequence(isq)) if self.is_integer_sequence() => {
                out.push_str(&isq.dump(evaluator));
            }
            Some(PatternDescriptor::DiscreteIntegerSet(dis)) if self.is_discrete_integer_set() => {
                out.push_str(&dis.dump());
            }
            Some(PatternDescriptor::LoopingChainMutableAddr(lcma))
                if self.is_looping_chain_mutable_addr() =>
            {
                out.push_str(&lcma.dump());
            }
            _ => out.push_str("* invalid pattern descriptor\n"),
        }
        out.push_str(LINE);
        out.push('\n');
        out
    }
}

impl DataTransferDescriptor {
    /// Replaces: e593_initializeDescriptor
    ///
    /// Recognises this transfer's base address, filling [`Self::pattern_desc`] and
    /// [`Self::base_addrs`] with the first pattern that fits and a toggle as the fall-back
    /// (`:2316-2467`).
    ///
    /// ⛔ THE COPY WALK IS A WRITE-BACK: `Value &base_addr = getOriginalBaseAddrSSA()` (`:2317`) is a
    /// REFERENCE to the field, so stripping the `scalar_copy` chain changes what every later reader of
    /// [`Self::base_addr`] sees.
    /// ⛔ TWO ARMS FALL THROUGH TO THE TOGGLE rather than returning — an unrecognised block argument
    /// (`:2437`) and an unrecognised `sentient.if` (`:2465`) both set the pattern to `None` and then
    /// try a toggle; only the memory-op arm (`:2443`) leaves without one.
    /// ⭐ `new`/`delete` PAIRS ARE THE `Option` ITSELF: the reference builds each candidate, keeps it
    /// in `pattern_desc_` and frees it on failure, so a candidate that fails simply is not stored.
    pub fn initialize_descriptor(
        &mut self,
        body: &[Op],
        defs: Definitions<'_>,
        evaluator: &mut impl ExpressionEvaluator,
    ) {
        while let Some(Op::Sentient(sentient::Op::ScalarCopy { input, .. })) = defs.of(self.base_addr)
        {
            self.base_addr = *input;
        }
        let base_addr = self.base_addr;

        // `if (dcc::utils::isSymbol(base_addr))` (`:2325-2328`).
        if is_symbol(base_addr, defs) {
            self.pattern_desc = None;
            return;
        }

        // Case 1: the address is itself constant (`:2336-2343`).
        if is_constant(base_addr, ConstKind::ScalarConstant, defs) {
            let ev = evaluator.evaluate_value_handle(base_addr);
            self.base_addrs.push(ev);
            self.pattern_desc = Some(PatternDescriptor::SimpleConstant(SimpleConstantDescriptor {
                ev,
            }));
            return;
        }

        // Case 2: `%base = scalar_add %c1, %c2`, `%c1` reached through its own copy chain
        // (`:2345-2367`). ⛔ `%c2` IS EVALUATED FIRST, before that walk, and the sum is `%c1 + %c2`.
        if let Some(Op::Sentient(sentient::Op::ScalarAdd { lhs, rhs, .. })) = defs.of(base_addr)
            && is_constant(*rhs, ConstKind::ScalarConstant, defs)
        {
            let right_summand_ev = evaluator.evaluate_value_handle(*rhs);
            let mut left_summand = *lhs;
            while let Some(Op::Sentient(sentient::Op::ScalarCopy { input, .. })) =
                defs.of(left_summand)
            {
                left_summand = *input;
            }
            if is_constant(left_summand, ConstKind::ScalarConstant, defs) {
                let left_summand_ev = evaluator.evaluate_value_handle(left_summand);
                let sum_ev = evaluator.evaluate_sum_handle(left_summand_ev, right_summand_ev);
                self.base_addrs.push(sum_ev);
                self.pattern_desc =
                    Some(PatternDescriptor::SimpleConstant(SimpleConstantDescriptor {
                        ev: sum_ev,
                    }));
                return;
            }
        }

        if defs.of(base_addr).is_none() {
            // `isa<BlockArgument>(base_addr)` (`:2377`) — an integer sequence, then a discrete
            // integer set, then a looping chain if this is the mutable address (`:2378-2437`).
            let isq_desc = IntegerSequenceDescriptor::new(base_addr, defs, evaluator);
            if isq_desc.is_valid() {
                isq_desc.get_all_constants(&mut self.base_addrs, evaluator);
                self.pattern_desc = Some(PatternDescriptor::IntegerSequence(isq_desc));
                return;
            }
            let dis_desc = DiscreteIntegerSetDescriptor::new(base_addr, defs, evaluator);
            if dis_desc.is_valid() {
                if let Some(init) = dis_desc.init() {
                    self.base_addrs.push(init);
                }
                self.pattern_desc = Some(PatternDescriptor::DiscreteIntegerSet(dis_desc));
                return;
            }
            if self.is_base_addr_mutable {
                let Some(op) = op_at(&self.op, body) else {
                    todo!(
                        "initializeDescriptor: `op_` is at {:?}, which this unit body does not reach",
                        self.op
                    )
                };
                let end = mutable_addr_end(op, self.memory_unit.dfir_unit(), defs);
                let lcma_desc =
                    LoopingChainMutableAddrDescriptor::new(base_addr, end, body, defs, evaluator);
                if lcma_desc.is_valid() {
                    if let Some(init) = lcma_desc.init() {
                        self.base_addrs.push(init);
                    }
                    self.pattern_desc = Some(PatternDescriptor::LoopingChainMutableAddr(lcma_desc));
                    return;
                }
            }
            self.pattern_desc = None;
        } else if matches!(
            defs.of(base_addr),
            Some(Op::Sentient(
                sentient::Op::ReceiveAndStore { .. }
                    | sentient::Op::LoadAndSend { .. }
                    | sentient::Op::LoadAndStore { .. }
                    | sentient::Op::For { .. }
            ))
        ) {
            // Part of a chain: the previous transfer owns the pattern (`:2438-2443`).
            if !self.is_base_addr_mutable {
                todo!(
                    "initializeDescriptor: DT_CHECK_MSG(is_base_addr_mutable_, \"Do not expect \
                     memory op result as immutable addr\") on {base_addr:?} (:2431-2432)"
                )
            }
            return;
        } else if HANDLE_CONDITIONAL_CONSTANTS
            && matches!(defs.of(base_addr), Some(Op::Sentient(sentient::Op::If { .. })))
        {
            let cc_desc = ConditionalConstantDescriptor::new(base_addr, defs, evaluator);
            if cc_desc.is_valid() {
                cc_desc.get_all_constants(&mut self.base_addrs);
                self.pattern_desc = Some(PatternDescriptor::ConditionalConstant(cc_desc));
                return;
            }
            self.pattern_desc = None;
        }

        // The fall-back: a toggle, whose two constants are `X` and — unless it simplified away — `Y`
        // (`:2452-2467`).
        let toggle_desc = ToggleDescriptor::new(base_addr, defs, evaluator);
        if ASSERT_ON_UNEXPECTED_PATTERNS && !toggle_desc.is_valid() {
            todo!(
                "initializeDescriptor: DT_CHECK(toggle_desc->isValid()) under \
                 -dcc-address-pinning-and-toggle-assert on {base_addr:?} (:2454)"
            )
        }
        if !toggle_desc.is_valid() {
            return;
        }
        if let Some(x) = toggle_desc.x(evaluator, body, defs) {
            self.base_addrs.push(x);
        }
        if !toggle_desc.can_be_simplified {
            self.base_addrs.push(toggle_desc.init(body, defs));
        }
        self.pattern_desc = Some(PatternDescriptor::Toggle(toggle_desc));
    }
}

/// `dcc::utils::isSymbol(val)` (`Analyses/Utils.cpp:141-154`) — the value, or ANY value of the
/// `uniform.query_map` it reads, is a `symbol.create_symbol`.
///
/// ⛔ `any`, WHERE [`is_constant`]'s QUERY-MAP ARM IS `all`: one symbolic entry makes the whole
/// address symbolic. ⭐ A block argument is `false`, which here is simply having no defining op.
fn is_symbol(val: Val, defs: Definitions<'_>) -> bool {
    let Some(def) = defs.of(val) else {
        return false;
    };
    if let Op::Uniform(uniform::Op::QueryMap { map, .. }) = def
        && let Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) = defs.of(*map)
        && pairs.iter().any(|(_, value)| {
            matches!(
                defs.of(*value),
                Some(Op::Symbol(symbol::Op::CreateSymbol { .. }))
            )
        })
    {
        return true;
    }
    matches!(def, Op::Symbol(symbol::Op::CreateSymbol { .. }))
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::sentient::{Reg, RegType};
    use crate::islands::sentient::dialects::{Val, symbol};
    use crate::transform::sentient::address_pinning_and_toggle::{
        SimpleConstantDescriptor, ToggleDescriptor,
    };
    use crate::transform::sentient::analyses::{
        Evaluation, EvaluatedValue, OffsetSites, OutOfScopeEvaluator,
    };
    use crate::transform::sentient::{ForRef, IterArgIndex};

    /// An evaluator that answers for a CONSTANT and a SUM, which is everything e593 asks before it
    /// reaches a pattern: the handle IS the value it names, so a sum's handle is the two added.
    struct StatedEvaluator;

    impl ExpressionEvaluator for StatedEvaluator {
        fn evaluate_value(&mut self, value: Val) -> Evaluation {
            OutOfScopeEvaluator.evaluate_value(value)
        }

        fn evaluate_sum(&mut self, lhs: &Evaluation, rhs: &Evaluation) -> Evaluation {
            OutOfScopeEvaluator.evaluate_sum(lhs, rhs)
        }

        fn build_offset_value(
            &mut self,
            evaluation: &Evaluation,
            sites: &mut OffsetSites<'_>,
            walked: &mut Vec<Op>,
            ty: ScalarTy,
        ) -> Val {
            OutOfScopeEvaluator.build_offset_value(evaluation, sites, walked, ty)
        }

        fn evaluate_value_handle(&mut self, value: Val) -> EvaluatedValue {
            EvaluatedValue(value.0)
        }

        fn evaluate_sum_handle(
            &mut self,
            lhs: EvaluatedValue,
            rhs: EvaluatedValue,
        ) -> EvaluatedValue {
            EvaluatedValue(lhs.0 + rhs.0)
        }
    }

    /// `%1 = 4096`, `%2 = copy %1`, `%3 = 64`, `%4 = %2 + %3`, and a symbol `%5`.
    fn constants() -> Vec<Op> {
        vec![
            Op::Sentient(sentient::Op::ScalarConstant {
                value: 4096,
                result: Val(1),
                reg_locale: RegType::Imm,
                ty: ScalarTy::Index,
                is_symbol: false,
            }),
            Op::Sentient(sentient::Op::ScalarCopy {
                input: Val(1),
                result: Val(2),
                reg: Reg {
                    locale: RegType::Lbr,
                    index: None,
                },
                element_size: None,
                program_header: false,
            }),
            Op::Sentient(sentient::Op::ScalarConstant {
                value: 64,
                result: Val(3),
                reg_locale: RegType::Imm,
                ty: ScalarTy::Index,
                is_symbol: false,
            }),
            Op::Sentient(sentient::Op::ScalarAdd {
                lhs: Val(2),
                rhs: Val(3),
                result: Val(4),
                reg: None,
                element_size: None,
                ty: ScalarTy::Index,
            }),
            Op::Symbol(symbol::Op::CreateSymbol {
                result: Val(5),
                symbol_id: 7,
                max_value: None,
            }),
        ]
    }

    /// 593/656 — a copy chain ending in a constant is a simple constant AND the walk is written back
    /// into `base_addr_`; `%c1 + %c2` is the one summed constant, its own left-hand copy chain walked
    /// without touching the field; a symbol is left with no pattern at all.
    #[test]
    fn e593_strips_the_copy_chain_and_recognises_both_simple_constant_shapes() {
        let body = constants();
        let scopes: [&[Op]; 1] = [&body];
        let defs = Definitions::from_innermost(&scopes);
        let initialized = |base_addr| {
            let mut dtd = DataTransferDescriptor {
                base_addr,
                ..descriptor(None, 0)
            };
            dtd.initialize_descriptor(&body, defs, &mut StatedEvaluator);
            dtd
        };

        let copied = initialized(Val(2));
        assert_eq!(copied.base_addr, Val(1));
        assert_eq!(copied.base_addrs, vec![EvaluatedValue(1)]);
        assert_eq!(
            copied.pattern_desc,
            Some(PatternDescriptor::SimpleConstant(SimpleConstantDescriptor {
                ev: EvaluatedValue(1),
            }))
        );

        let summed = initialized(Val(4));
        assert_eq!(summed.base_addr, Val(4));
        assert_eq!(summed.base_addrs, vec![EvaluatedValue(4)]);
        assert_eq!(
            summed.pattern_desc,
            Some(PatternDescriptor::SimpleConstant(SimpleConstantDescriptor {
                ev: EvaluatedValue(4),
            }))
        );

        let symbolic = initialized(Val(5));
        assert_eq!(symbolic.pattern_desc, None);
        assert!(symbolic.base_addrs.is_empty());
    }

    /// A toggle that matched, so only the base-address count decides.
    fn matched_toggle(can_be_simplified: bool) -> ToggleDescriptor {
        ToggleDescriptor {
            outer_loop: Some(ForRef(Val(1))),
            iter_arg_index: Some(IterArgIndex(0)),
            c1: Some(EvaluatedValue(2)),
            can_be_simplified,
        }
    }

    fn descriptor(pattern: Option<PatternDescriptor>, base_addrs: u32) -> DataTransferDescriptor {
        DataTransferDescriptor {
            op: OpId::at(&[0]),
            pattern_desc: pattern,
            base_addrs: (0..base_addrs).map(EvaluatedValue).collect(),
            region: RegionSite::default(),
            memory_unit: DescriptorMemoryUnit::Lx,
            base_addr: Val(0),
            is_base_addr_mutable: false,
        }
    }

    fn simple_constant() -> PatternDescriptor {
        PatternDescriptor::SimpleConstant(SimpleConstantDescriptor {
            ev: EvaluatedValue(2),
        })
    }

    /// The toggle arm's base-address count, and the `SimpleConstant` the `dyn_cast` chain skips.
    #[test]
    fn e278_counts_base_addrs_for_a_toggle_and_falls_through_for_a_simple_constant() {
        let toggle = |simplified, count| {
            descriptor(
                Some(PatternDescriptor::Toggle(matched_toggle(simplified))),
                count,
            )
            .is_valid()
        };
        assert!(toggle(false, 2));
        assert!(!toggle(false, 1));
        assert!(toggle(true, 1));
        // ⭐ `|| base_addrs.size() == 2` IS NOT GUARDED BY THE FLAG, so a simplified toggle holding
        // two is valid too.
        assert!(toggle(true, 2));
        // No `SimpleConstant` arm: it reaches the `base_addrs.size() == 1` tail, as `None` does.
        assert!(descriptor(Some(simple_constant()), 1).is_valid());
        assert!(!descriptor(Some(simple_constant()), 2).is_valid());
        assert!(descriptor(None, 1).is_valid());
        assert!(!descriptor(None, 0).is_valid());
    }

    /// The pattern's own flag, gated on validity — and the simple constant whose `true` is unread.
    #[test]
    fn e279_reads_the_patterns_flag_and_never_the_simple_constants() {
        assert!(
            descriptor(Some(PatternDescriptor::Toggle(matched_toggle(true))), 1)
                .can_be_simplified()
        );
        // Every `is*()` opens with `isValid()`, which an unmatched toggle fails.
        assert!(
            !descriptor(
                Some(PatternDescriptor::Toggle(ToggleDescriptor::default())),
                1
            )
            .can_be_simplified()
        );
        assert!(!descriptor(Some(simple_constant()), 1).can_be_simplified());
    }

    /// `e492` — THE ONLY PATH THAT COMPLETES: with no base addresses nothing reaches the
    /// out-of-scope `EvaluatedValue::operator<<`, and an unmatched pattern takes the `else`.
    #[test]
    fn e492_fences_the_operation_and_reports_an_invalid_pattern_descriptor() {
        let body = vec![Op::Symbol(symbol::Op::CreateSymbol {
            result: Val(0),
            symbol_id: 7,
            max_value: None,
        })];

        assert_eq!(
            descriptor(None, 0).dump(&body, &mut OutOfScopeEvaluator),
            "----------\n\
             * operation:line:?:%0 = symbol.create_symbol {SymbolId = 7 : i32} : index\n\
             * base addrs:[\n\
             \n]\n\
             * invalid pattern descriptor\n\
             ----------\n"
        );
    }

    /// `DT_CHECK_MSG((!res || getBaseAddrList().size() == 1), ..)` is an abort, and a simplified
    /// toggle holding two base addresses is a descriptor that reaches it.
    #[test]
    #[should_panic(expected = "single base_addr stored in the descriptor")]
    fn e279_aborts_on_a_simplified_pattern_with_more_than_one_base_addr() {
        let desc = descriptor(Some(PatternDescriptor::Toggle(matched_toggle(true))), 2);
        let _simplified = desc.can_be_simplified();
    }
}

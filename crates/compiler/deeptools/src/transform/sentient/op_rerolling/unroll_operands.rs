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

//! `OpRerolling.cpp` — 12 of the campaign's 656 units (dependency level(s) [0, 1]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e113_isXrfRdOp` | 113 | 0 | 11 | `dcc/src/Transform/Sentient/OpRerolling.cpp:58` |
//! | `e114_isXrfWtOp` | 114 | 0 | 9 | `dcc/src/Transform/Sentient/OpRerolling.cpp:69` |
//! | `e115_reset` | 115 | 0 | 31 | `dcc/src/Transform/Sentient/OpRerolling.cpp:439` |
//! | `e116_getUnrollOperandUsingPort` | 116 | 0 | 19 | `dcc/src/Transform/Sentient/OpRerolling.cpp:471` |
//! | `e117_fillMemOpAttrs` | 117 | 0 | 9 | `dcc/src/Transform/Sentient/OpRerolling.cpp:492` |
//! | `e118_areOperandsMatching` | 118 | 0 | 59 | `dcc/src/Transform/Sentient/OpRerolling.cpp:668` |
//! | `e119_createOperand` | 119 | 0 | 11 | `dcc/src/Transform/Sentient/OpRerolling.cpp:926` |
//! | `e120_createForwardingArray` | 120 | 0 | 17 | `dcc/src/Transform/Sentient/OpRerolling.cpp:939` |
//! | `e121_dump` | 121 | 0 | 27 | `dcc/src/Transform/Sentient/OpRerolling.cpp:966` |
//! | `e335_fill` | 335 | 1 | 165 | `dcc/src/Transform/Sentient/OpRerolling.cpp:502` |
//! | `e336_match` | 336 | 1 | 149 | `dcc/src/Transform/Sentient/OpRerolling.cpp:731` |
//! | `e337_updateUnrollInfo` | 337 | 1 | 44 | `dcc/src/Transform/Sentient/OpRerolling.cpp:881` |

// ⛔ NOTHING IN THE CRATE CALLS THIS FILE UNTIL `e607_runOnOperation` (level 5) LANDS, and CI runs
// clippy with `-D warnings`. ⭐ REMOVE THIS WITH e607.
#![allow(dead_code)]

use super::UnrollSize;
use crate::arch::Elements;
use crate::formats::Bits;
use crate::islands::sentient::dialects::{Op, Val};
use crate::islands::sentient::dialects::sentient as sen;
use crate::units::DfirUnit;

/// WHICH OPERAND SLOT OF A COMPUTE OP — `enum OperandName` (`OpRerolling.hpp:30-36`), the key of
/// every one of [`UnrollOperands`]'s three maps.
///
/// ⛔ NOT THE IR'S `opA`/`opB`/`opC` ([`sen::Port::OpA`]), which name a forwarding TARGET. These are
/// containers: `e335_fill` puts an op's operand under the slot its PORT ID selects (`:523-531`), so
/// `op_a` here is "whatever arrives on port0" and not the op's own `opA` attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum OperandName {
    /// `op_a = 0`.
    OpA,
    /// `op_b = 1`.
    OpB,
    /// `op_c = 2`.
    OpC,
    /// `result = 3`.
    Result,
    /// `logical_result = 4`.
    LogicalResult,
}

impl OperandName {
    /// ⭐ THE ITERATION ORDER OF EVERY `std::map` KEYED BY THIS ENUM, and so the order e121 prints:
    /// `std::map` walks sorted by key, and the key's order is its `= 0..4` (`OpRerolling.hpp:30-36`).
    pub(crate) const ALL: [Self; 5] = [
        Self::OpA,
        Self::OpB,
        Self::OpC,
        Self::Result,
        Self::LogicalResult,
    ];

    /// Which entry of [`UnrollOperands`]'s per-slot arrays this name is.
    #[must_use]
    pub(crate) const fn slot(self) -> usize {
        self as usize
    }

    /// What e121's `toString` lambda gives (`:967-982`) — the C++ ENUMERATOR spellings, `op_a` and
    /// not the IR's `opA`.
    #[must_use]
    pub(crate) const fn spelling(self) -> &'static str {
        match self {
            Self::OpA => "op_a",
            Self::OpB => "op_b",
            Self::OpC => "op_c",
            Self::Result => "result",
            Self::LogicalResult => "logical_result",
        }
    }
}

/// WHICH COMPUTE PORT AN OPERAND WAS ASSIGNED — the pass's OWN
/// `enum SentientComputePort {port0, port1, port2}` (`OpRerolling.hpp:37`).
///
/// ⛔ NOT THE DIALECT'S `SentientComputePort`, WHICH IS [`sen::Port`] — the reference shadows that
/// name with a three-value port INDEX and casts a raw `getOpAPortID()` to it (`:523-524`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ComputePortId {
    /// `port0`.
    Port0,
    /// `port1`.
    Port1,
    /// `port2`.
    Port2,
}

impl ComputePortId {
    /// An [`sen::Operand::port_id`](sen::Operand)'s value as the reference reads it: a C-style cast
    /// of the raw i32, whose only tested cases are 0 and 1.
    ///
    /// ⭐ ANYTHING ELSE — INCLUDING THE UNASSIGNED `-1` — IS `port2`, because the reference's third
    /// arm is a bare `else` and not a `port2` comparison (`:477-479`).
    #[must_use]
    pub(crate) const fn from_port_id(port_id: Option<i32>) -> Self {
        match port_id {
            Some(0) => Self::Port0,
            Some(1) => Self::Port1,
            _ => Self::Port2,
        }
    }
}

/// AN XRF POINTER INCREMENT — `int xrf_read_incr_` / `int xrf_write_incr_` (`OpRerolling.hpp:54-55`),
/// signed because the reference reads it through `getXrfReadIncrSigned()` (`:562`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct XrfIncr(pub(crate) i32);

impl XrfIncr {
    /// No increment.
    pub(crate) const ZERO: Self = Self(0);
}

/// WHICH OP A SNAPSHOT CAME FROM — `StringRef op_name_ = "NA"` (`OpRerolling.hpp:46`), the six kinds
/// `e335_fill`'s `dyn_cast` chain accepts (`:519`, `:564`, `:598`, `:621`, `:640`, `:652`).
///
/// ⛔ `Option`'s `None` IS THE `"NA"` SENTINEL, which the reference tests eight times over (`:189`,
/// `:324-325`, `:338`, `:376`, `:391`, `:415`, `:732`) — an absent snapshot, and not a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RolledOp {
    /// `"mac"` (`:520`).
    Mac,
    /// `stringifySentientBinaryOperator(...)` (`:565`).
    Binary(sen::BinaryOp),
    /// `stringifySentientUnary(...)` (`:599`).
    Unary(sen::UnaryOp),
    /// `sentient.splat` (`:624`).
    Splat,
    /// `sentient.load_and_send` (`:641`).
    LoadAndSend,
    /// `sentient.receive_and_store` (`:654`).
    ReceiveAndStore,
}

/// A MEMORY OP'S EXTENT WITHOUT ITS BURST SIZE — see [`UnrollOperands::fill_mem_op_attrs`] for why
/// `burst_size` is the one attribute left out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UnburstedExtent {
    /// [`sen::Extent::total_elements`].
    pub(crate) total_elements: Elements,
    /// [`sen::Extent::element_size`].
    pub(crate) element_size: Bits,
    /// [`sen::Extent::chunk_size`].
    pub(crate) chunk_size: Elements,
    /// [`sen::Extent::chunk_stride`].
    pub(crate) chunk_stride: Elements,
}

impl UnburstedExtent {
    /// Every extent attribute but the burst size.
    #[must_use]
    pub(crate) const fn of(extent: sen::Extent) -> Self {
        Self {
            total_elements: extent.total_elements,
            element_size: extent.element_size,
            chunk_size: extent.chunk_size,
            chunk_stride: extent.chunk_stride,
        }
    }
}

/// THE ATTRIBUTES OF ONE MEMORY OP — `std::map<std::string, Attribute> memory_op_attrs`
/// (`OpRerolling.hpp:67`), compared entry-for-entry by `e336_match` (`:795-799`).
///
/// ⛔ ATTRIBUTES ONLY, NOT OPERANDS: `op->getAttrs()` does not reach a `sentient.receive_and_store`'s
/// `$dst`, `$drop_first` or `$multicast_info`, which are SSA operands — `e336_match` compares those
/// separately, through `src_dst` and `addr_offsets` (`:764-793`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MemOpAttrs {
    /// `sentient.load_and_send`'s.
    LoadAndSend {
        /// `total_elements`/`element_size`/`chunk_size`/`chunk_stride`.
        extent: UnburstedExtent,
        /// `interleaved_group`.
        interleaved_group: Elements,
        /// `rotate_val`.
        rotate_val: Option<u32>,
        /// `dir`.
        dir: Option<sen::RoutingDirection>,
        /// `shuffle_mode`.
        shuffle_mode: sen::ShuffleMode,
        /// `regLocale`/`regIndex`.
        reg: sen::Reg,
        /// `dbgName`.
        dbg_name: Option<String>,
    },
    /// `sentient.receive_and_store`'s.
    ReceiveAndStore {
        /// `total_elements`/`element_size`/`chunk_size`/`chunk_stride`.
        extent: UnburstedExtent,
        /// `interleaved_group`.
        interleaved_group: Elements,
        /// `coalesce`.
        coalesce: bool,
        /// `subword_length`.
        subword_length: u32,
        /// `stride`.
        stride: u32,
        /// `permute`.
        permute: bool,
        /// `shuffle_mode`.
        shuffle_mode: Option<sen::ShuffleMode>,
        /// `regLocale`/`regIndex`.
        reg: sen::Reg,
        /// `dbgName`.
        dbg_name: Option<String>,
    },
}

/// ONE COMPUTE OR MEMORY OP'S UNROLL-RELEVANT FIELDS — `class UnrollOperands`
/// (`OpRerolling.hpp:43-164`), filled from one op by `e335_fill` and compared against another by
/// `e336_match`.
///
/// ⭐ AN ARRAY PER SLOT, NOT A MAP: the reference's `std::map<OperandName, …>` always holds exactly
/// the five keys — the constructor inserts all five (`OpRerolling.hpp:85-100`) and `e115_reset` only
/// overwrites them (`:440-451`) — so "every slot has an entry" is the array's length here.
///
/// ⛔ NO `derive(Clone)`, DELIBERATELY — see [`UnrollOperands::assign_from`].
///
/// ⭐ `MLIRContext *context` (`OpRerolling.hpp:56`) IS DROPPED: it exists only to be handed to
/// `StringAttr::get`/`SentientComputePortAttr::get`, and this island's attributes are values.
///
/// ⭐ AND THE ONE-TO-THREE-LINE ACCESSORS (`getUnrollSize`, `getOpName`, `isThisFieldUnrolled`, …)
/// ARE THESE FIELDS, which is why `crustify-senpass/EXCLUSIONS.tsv` excludes every one of them.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct UnrollOperands {
    /// `operand_list_` (`OpRerolling.hpp:57`) — where each slot's operand comes from.
    ///
    /// ⛔ `None` IS THE REFERENCE'S `"NA"` (`:442`, `OpRerolling.hpp:86-90`) AND NOT A PORT: every
    /// value it ever holds otherwise is a `stringifySentientComputePort` result (`:529`, `:571`), so
    /// the sentinel cannot be a [`sen::Port`] case. `Default` is therefore the constructor's state.
    pub(crate) operand_list: [Option<sen::Port>; OperandName::ALL.len()],
    /// `is_field_unroll_` (`OpRerolling.hpp:59`) — whether that slot's register index advances with
    /// the unroll factor.
    pub(crate) is_field_unroll: [bool; OperandName::ALL.len()],
    /// `forwarding_list_` (`OpRerolling.hpp:58`), in `e335_fill`'s lexicographic order (`:541-544`).
    pub(crate) forwarding_list: [Vec<sen::Port>; OperandName::ALL.len()],
    /// `unroll_size_`.
    pub(crate) unroll_size: UnrollSize,
    /// `op_name_` — `None` is the `"NA"` sentinel.
    pub(crate) op_name: Option<RolledOp>,
    /// `are_unroll_fields_updated_`.
    pub(crate) are_unroll_fields_updated: bool,
    /// `precisions_`.
    pub(crate) precisions: Vec<sen::Precision>,
    /// `xrf_read_incr_`.
    pub(crate) xrf_read_incr: XrfIncr,
    /// `xrf_write_incr_`.
    pub(crate) xrf_write_incr: XrfIncr,
    /// `mask_value_`.
    pub(crate) mask_value: Option<Val>,
    /// `is_splat_promoted_`.
    pub(crate) is_splat_promoted: bool,
    /// `splat_pad_`.
    pub(crate) splat_pad: sen::SplatPad,
    /// `splat_input_`.
    pub(crate) splat_input: Option<Val>,
    /// `fold_mode`.
    pub(crate) fold_mode: sen::FoldMode,
    /// `memory_op_attrs` — `None` until [`UnrollOperands::fill_mem_op_attrs`] runs.
    pub(crate) memory_op_attrs: Option<MemOpAttrs>,
    /// `src_dst` — a memory op's consumer or producer.
    pub(crate) src_dst: Vec<Val>,
    /// `addr_offsets` — a memory op's immutable address, then its increment.
    pub(crate) addr_offsets: Vec<Val>,
    /// `mutable_address`.
    pub(crate) mutable_address: Option<Val>,
    /// `result_address`.
    pub(crate) result_address: Option<Val>,
    /// `is_memory_unit`.
    pub(crate) is_memory_unit: bool,
}

impl Default for UnrollOperands {
    /// The reference's constructor (`OpRerolling.hpp:85-101`) and its member initialisers (`:45-47`,
    /// `:53-55`, `:62-65`, `:75`): every slot `"NA"`, no forwarding, nothing field-unrolled,
    /// `unroll_size_` 1.
    fn default() -> Self {
        Self {
            operand_list: [None; OperandName::ALL.len()],
            is_field_unroll: [false; OperandName::ALL.len()],
            forwarding_list: core::array::from_fn(|_| Vec::new()),
            unroll_size: UnrollSize::ONE,
            op_name: None,
            are_unroll_fields_updated: false,
            precisions: Vec::new(),
            xrf_read_incr: XrfIncr::ZERO,
            xrf_write_incr: XrfIncr::ZERO,
            mask_value: None,
            is_splat_promoted: false,
            splat_pad: sen::SplatPad::None,
            splat_input: None,
            fold_mode: sen::FoldMode::FoldA,
            memory_op_attrs: None,
            src_dst: Vec::new(),
            addr_offsets: Vec::new(),
            mutable_address: None,
            result_address: None,
            is_memory_unit: false,
        }
    }
}

impl UnrollOperands {
    /// Replaces: e121_dump
    ///
    /// The debugger's view of one `UnrollOperands`: a header, then all five operands, then all five
    /// unroll flags, each block in [`OperandName::ALL`] order.
    ///
    /// ⭐ A RETURNED `String` FOR `llvm::outs()`, as e014_dump and e065_dump did. Nothing in the
    /// reference tree calls `dump()` at all — it is reached from a debugger — so no output the
    /// reference produced is lost by not writing it here.
    ///
    /// ⭐ `1`/`0` FOR THE FLAG, NOT `true`/`false`: `raw_ostream` has no `bool` overload, so `bool`
    /// promotes to `operator<<(int)` (`raw_ostream.h:284`) and the reference prints the digit.
    ///
    /// ⛔ THE LAMBDA'S `default: "unknown operand name"` (`:979-980`) IS UNREACHABLE AND IS NOT PORTED:
    /// the switch already covers all five cases the enum declares, and the arms come from the keys
    /// of maps that only ever hold those five.
    #[must_use]
    pub(crate) fn dump(&self) -> String {
        let mut out = String::from("dumping:\n");
        for name in OperandName::ALL {
            let operand =
                self.operand_list[name.slot()].map_or_else(|| "NA".to_owned(), sen::Port::spelling);
            out.push_str(&format!("{}: {operand}\n", name.spelling()));
        }
        for name in OperandName::ALL {
            let flag = u8::from(self.is_field_unroll[name.slot()]);
            out.push_str(&format!("{}: {flag}\n", name.spelling()));
        }
        out
    }

    /// Replaces: e113_isXrfRdOp
    ///
    /// Whether any operand slot reads the PT's transposed register file (`:58`).
    #[must_use]
    pub(crate) fn is_xrf_rd_op(&self) -> bool {
        if self.is_memory_unit {
            return false;
        }
        self.operand_list.contains(&Some(sen::Port::Xrf))
    }

    /// Replaces: e114_isXrfWtOp
    ///
    /// Whether the RESULT slot forwards to the XRF, which is what writing it looks like (`:69`).
    #[must_use]
    pub(crate) fn is_xrf_wt_op(&self) -> bool {
        if self.is_memory_unit {
            return false;
        }
        self.forwarding_list[OperandName::Result.slot()].contains(&sen::Port::Xrf)
    }

    /// `isXrfOp()` (`OpRerolling.hpp:158`) — excluded from the worklist as a one-liner, and written
    /// here because `e338_setUnrollFieldsInStmt` calls it.
    #[must_use]
    pub(crate) fn is_xrf_op(&self) -> bool {
        self.is_xrf_rd_op() || self.is_xrf_wt_op()
    }

    /// Replaces: e115_reset
    ///
    /// Empties the snapshot and puts `unroll_size_` back to 1, ready for the next candidate (`:439`).
    ///
    /// ⛔⛔ TRAP — NOT `*self = Self::default()`. THREE FIELDS SURVIVE A RESET because the reference
    /// never writes them here (`:439-469`): `mask_value_`, `is_splat_promoted_` and `is_memory_unit`.
    /// A reset snapshot therefore still remembers it came from a memory unit, which is exactly what
    /// [`Self::is_xrf_rd_op`] and `e336_match` (`:742`) branch on.
    pub(crate) fn reset(&mut self) {
        self.operand_list.fill(None);
        for forwarding in &mut self.forwarding_list {
            forwarding.clear();
        }
        self.is_field_unroll.fill(false);
        self.op_name = None;
        self.precisions.clear();
        self.splat_pad = sen::SplatPad::None;
        self.splat_input = None;
        self.unroll_size = UnrollSize::ONE;
        self.are_unroll_fields_updated = false;
        self.xrf_read_incr = XrfIncr::ZERO;
        self.xrf_write_incr = XrfIncr::ZERO;
        self.memory_op_attrs = None;
        self.src_dst.clear();
        self.addr_offsets.clear();
        self.mutable_address = None;
        self.result_address = None;
        self.fold_mode = sen::FoldMode::FoldA;
    }

    /// Replaces: e116_getUnrollOperandUsingPort
    ///
    /// Which operand slot an assigned compute port fills: `port0`→`op_a`, and on a PT row `port1` and
    /// `port2` land the other way round from everywhere else (`:471`).
    #[must_use]
    pub(crate) const fn unroll_operand_using_port(
        component: DfirUnit,
        port_id: ComputePortId,
    ) -> OperandName {
        if component.is_pt_row() {
            match port_id {
                ComputePortId::Port0 => OperandName::OpA,
                ComputePortId::Port1 => OperandName::OpC,
                ComputePortId::Port2 => OperandName::OpB,
            }
        } else {
            match port_id {
                ComputePortId::Port0 => OperandName::OpA,
                ComputePortId::Port1 => OperandName::OpB,
                ComputePortId::Port2 => OperandName::OpC,
            }
        }
    }

    /// Replaces: e117_fillMemOpAttrs
    ///
    /// Snapshots a memory op's attributes for `e336_match` to compare, minus `burst_size` (`:492`).
    ///
    /// ⛔ DROPPING `burst_size` IS LOAD-BEARING, NOT TIDYING — *"burst_size is recorded by a
    /// standalone object"* (`:496`): `e335_fill` has already turned it into `unroll_size_` (`:643`,
    /// `:656`), which a reroll then INCREMENTS, so leaving it in the snapshot would make every
    /// already-rerolled memory op compare unequal and no memory reroll could ever grow.
    pub(crate) fn fill_mem_op_attrs(&mut self, op: &sen::Op) {
        if let sen::Op::LoadAndSend {
            extent,
            interleaved_group,
            rotate_val,
            dir,
            shuffle_mode,
            reg,
            dbg_name,
            ..
        } = op
        {
            self.memory_op_attrs = Some(MemOpAttrs::LoadAndSend {
                extent: UnburstedExtent::of(*extent),
                interleaved_group: *interleaved_group,
                rotate_val: *rotate_val,
                dir: *dir,
                shuffle_mode: *shuffle_mode,
                reg: *reg,
                dbg_name: dbg_name.clone(),
            });
        } else if let sen::Op::ReceiveAndStore {
            extent,
            interleaved_group,
            coalesce,
            subword_length,
            stride,
            permute,
            shuffle_mode,
            reg,
            dbg_name,
            ..
        } = op
        {
            self.memory_op_attrs = Some(MemOpAttrs::ReceiveAndStore {
                extent: UnburstedExtent::of(*extent),
                interleaved_group: *interleaved_group,
                coalesce: *coalesce,
                subword_length: *subword_length,
                stride: *stride,
                permute: *permute,
                shuffle_mode: *shuffle_mode,
                reg: *reg,
                dbg_name: dbg_name.clone(),
            });
        }
    }

    /// Replaces: e118_areOperandsMatching
    ///
    /// Whether the candidate's operand can reroll onto the reference's: identical unless indexed, and
    /// an LRF/ISTATE index gap the two unroll factors agree with (`:668`).
    ///
    /// ⛔⛔ TRAP — A MIXED `lrf`/`istate` PAIR IS THE REFERENCE'S OWN ABORT: `is_lrf` is false there,
    /// so it runs `getIStateIndex` over an `lrf` spelling, `substr(6)` of a four-character StringRef
    /// is empty and `std::stoi("")` throws in an `-fno-exceptions` build (`:704-709`, `:962`). `false`
    /// is the conservative direction — it can only prevent a reroll, never enable a wrong one.
    fn are_operands_matching(
        &self,
        ref_operand: Option<sen::Port>,
        is_ref_unrolled: bool,
        curr_operand: Option<sen::Port>,
        is_curr_unrolled: bool,
        curr_operand_list: &Self,
        is_xrf_read: bool,
    ) -> bool {
        let is_indexed = |port| matches!(port, Some(sen::Port::Lrf(_) | sen::Port::IState(_)));
        if !is_indexed(ref_operand) || !is_indexed(curr_operand) {
            // Non-lrf non-istate — xrf, reg, constants, forwarding targets — must be the SAME.
            if ref_operand != curr_operand {
                return false;
            }
            if ref_operand == Some(sen::Port::Xrf) {
                if is_xrf_read {
                    // A reroll may absorb a read-pointer increment of zero and nothing else.
                    if self.xrf_read_incr != XrfIncr::ZERO {
                        return false;
                    }
                } else if !self.xrf_write_incr_matches_unroll_size()
                    || !curr_operand_list.xrf_write_incr_matches_unroll_size()
                {
                    // The write pointer steps once per mac instance, so both ops' increments must
                    // already equal their own unroll factor.
                    return false;
                }
            }
        } else {
            let unroll = i64::from(self.unroll_size.0);
            let curr_unroll = i64::from(curr_operand_list.unroll_size.0);
            let index_gap = match (ref_operand, curr_operand) {
                (Some(sen::Port::Lrf(reference)), Some(sen::Port::Lrf(candidate))) => {
                    i64::from(candidate.get()) - i64::from(reference.get())
                }
                (Some(sen::Port::IState(reference)), Some(sen::Port::IState(candidate))) => {
                    i64::from(candidate.get()) - i64::from(reference.get())
                }
                _ => return false,
            };
            if index_gap < 0 {
                return false;
            }
            if is_curr_unrolled && (index_gap != unroll || (!is_ref_unrolled && unroll > 1)) {
                return false;
            }
            if !is_curr_unrolled && is_ref_unrolled && (index_gap != unroll || curr_unroll > 1) {
                return false;
            }
            if !is_curr_unrolled
                && !is_ref_unrolled
                && ((unroll * curr_unroll > 1 && index_gap != 0)
                    || (unroll * curr_unroll == 1 && index_gap > 1))
            {
                return false;
            }
        }

        true
    }

    /// `xrf_write_incr_ == unroll_size_` (`:689-691`), an `int` against an `unsigned` in the
    /// reference and so a plain numeric comparison.
    fn xrf_write_incr_matches_unroll_size(&self) -> bool {
        i64::from(self.xrf_write_incr.0) == i64::from(self.unroll_size.0)
    }

    /// Replaces: e119_createOperand
    ///
    /// The operand a rerolled op should carry in this slot: the slot's port, moved up the LRF by how
    /// far the new unroll factor undershoots the current one, when the slot is field-unrolled (`:926`).
    ///
    /// ⛔ `None` IS THE REFERENCE'S OWN DEATH AND NOT A CHECK ADDED HERE: its caller feeds this
    /// straight into `symbolizeSentientComputePort(...).value()` (`:1091`, `:1306-1310`), which
    /// is `std::nullopt` past `lrf31` and for the `"NA"` an unfilled slot still holds.
    #[must_use]
    pub(crate) fn create_operand(
        &self,
        op_name: OperandName,
        new_unroll_size: UnrollSize,
    ) -> Option<sen::Port> {
        let operand = self.operand_list[op_name.slot()];
        match (self.is_field_unroll[op_name.slot()], operand) {
            (true, Some(sen::Port::Lrf(index))) => {
                lrf_at(self.shifted_lrf(index, new_unroll_size)).map(sen::Port::Lrf)
            }
            _ => operand,
        }
    }

    /// Replaces: e120_createForwardingArray
    ///
    /// The forwarding array a rerolled op should carry in this slot — the RESULT slot's `lrf` targets
    /// moved by the same undershoot, every other slot's left exactly as they are (`:939`).
    ///
    /// ⛔ `None` for the reason [`Self::create_operand`] gives: the reference calls
    /// `symbolizeSentientComputePort(...).value()` on each entry it built (`:952-953`).
    #[must_use]
    pub(crate) fn create_forwarding_array(
        &self,
        op_name: OperandName,
        new_unroll_size: UnrollSize,
    ) -> Option<Vec<sen::Port>> {
        let shift = op_name == OperandName::Result && self.is_field_unroll[op_name.slot()];
        self.forwarding_list[op_name.slot()]
            .iter()
            .map(|forwarded| match (shift, forwarded) {
                (true, sen::Port::Lrf(index)) => {
                    lrf_at(self.shifted_lrf(*index, new_unroll_size)).map(sen::Port::Lrf)
                }
                _ => Some(*forwarded),
            })
            .collect()
    }

    /// `getLrfIndex(x) + (unroll_size_ - new_unroll_size)` (`:931`, `:946-947`).
    ///
    /// ⭐ WRAPPING, LIKE THE REFERENCE'S `unsigned`: `e338_setUnrollFieldsInStmt` counts the new
    /// factor DOWN from `unroll_size_` (`:1045`, `:1064`), so the undershoot is never negative
    /// there, and a wrap lands far past `lrf31` where [`lrf_at`] already says `None`.
    const fn shifted_lrf(&self, index: sen::LrfIndex, new_unroll_size: UnrollSize) -> u32 {
        (index.get() as u32).wrapping_add(self.unroll_size.0.wrapping_sub(new_unroll_size.0))
    }

    /// `UnrollOperands &operator=(const UnrollOperands &other)` (`OpRerolling.hpp:103-125`), which
    /// carries the candidate's snapshot over the reference's (`:195`).
    ///
    /// ⛔⛔ AND THIS IS WHY THE TYPE CARRIES NO `derive(Clone)`: the reference's assignment copies
    /// nineteen members and SKIPS `is_splat_promoted_` and `is_memory_unit`, so the destination keeps
    /// its own two. A derived clone would copy them as well, silently.
    pub(crate) fn assign_from(&mut self, other: &Self) {
        self.op_name = other.op_name;
        self.mask_value = other.mask_value;
        self.splat_pad = other.splat_pad;
        self.splat_input = other.splat_input;
        self.operand_list = other.operand_list;
        self.forwarding_list = other.forwarding_list.clone();
        self.is_field_unroll = other.is_field_unroll;
        self.precisions = other.precisions.clone();
        self.are_unroll_fields_updated = other.are_unroll_fields_updated;
        self.unroll_size = other.unroll_size;
        self.xrf_read_incr = other.xrf_read_incr;
        self.xrf_write_incr = other.xrf_write_incr;
        self.memory_op_attrs = other.memory_op_attrs.clone();
        self.src_dst = other.src_dst.clone();
        self.addr_offsets = other.addr_offsets.clone();
        self.mutable_address = other.mutable_address;
        self.result_address = other.result_address;
        self.fold_mode = other.fold_mode;
    }

    /// `UnrollOperands::fill(Operation *op, SenComponents type)` — SENPASS UNIT e335, whose anchor is
    /// still open below.
    ///
    /// ⛔ e335 IS A LEVEL-1 DEPENDENCY THAT NO REMAINING SCHEDULE OWNS, for the reason
    /// [`Self::update_unroll_info`] gives; its TODO is left untouched. ⭐ EVERY PATH THROUGH
    /// `e519_processOneBlock` REACHES THIS at the block's first candidate, so it is the seam a rerolled
    /// program stops at.
    pub(super) fn fill(&mut self, op: &Op, ty: DfirUnit) {
        let _ = op;
        todo!(
            "UnrollOperands::fill — senpass e335 (OpRerolling.cpp:502) is not ported yet, and \
             snapshotting this {ty:?} op needs it"
        )
    }

    /// `bool UnrollOperands::match(UnrollOperands &new_operand_list)` — SENPASS UNIT e336, whose anchor
    /// is still open below. ⭐ `match` IS A KEYWORD; the question it asks is the name.
    ///
    /// ⛔ e336 IS A LEVEL-1 DEPENDENCY THAT NO REMAINING SCHEDULE OWNS, for the reason
    /// [`Self::update_unroll_info`] gives; its TODO is left untouched.
    pub(super) fn matches(&self, new_operand_list: &UnrollOperands) -> bool {
        todo!(
            "UnrollOperands::match — senpass e336 (OpRerolling.cpp:731) is not ported yet, and \
             deciding whether {:?} rerolls into {:?} needs it",
            new_operand_list.op_name,
            self.op_name
        )
    }

    /// `UnrollOperands::updateUnrollInfo(UnrollOperands &new_operand_list)` — SENPASS UNIT e337, whose
    /// anchor is still open below.
    ///
    /// ⛔ e337 IS A LEVEL-1 DEPENDENCY THAT NO REMAINING SCHEDULE OWNS: sc2's port driver died on an
    /// authentication error with 12 batches unrun, `sentient.cpp: e334_mergeScalarOpIntoMac +7` among
    /// them. Isolating the call in a seam is the `2a8195231` precedent, and e337's TODO is left
    /// untouched — filling it is not this batch's work.
    pub(super) fn update_unroll_info(&mut self, new_operand_list: &UnrollOperands) {
        todo!(
            "UnrollOperands::updateUnrollInfo — senpass e337 (OpRerolling.cpp:881) is not ported \
             yet, and merging {:?} into {:?} needs it",
            new_operand_list.op_name,
            self.op_name
        )
    }
}

/// `symbolizeSentientComputePort("lrf" + std::to_string(index))` for a COMPUTED index (`:929-933`,
/// `:946-949`).
///
/// ⛔ `None` IS THE REFERENCE'S OWN DEATH, NOT A CHECK ADDED HERE — the shape
/// `bridges::dataflow_ir_to_sentient::vc_vector_operands::register_slice` already documents: every
/// reader calls `.value()` on the result, which is `std::nullopt` from `lrf32` up.
pub(crate) const fn lrf_at(index: u32) -> Option<sen::LrfIndex> {
    match index {
        0 => Some(sen::LrfIndex::L0),
        1 => Some(sen::LrfIndex::L1),
        2 => Some(sen::LrfIndex::L2),
        3 => Some(sen::LrfIndex::L3),
        4 => Some(sen::LrfIndex::L4),
        5 => Some(sen::LrfIndex::L5),
        6 => Some(sen::LrfIndex::L6),
        7 => Some(sen::LrfIndex::L7),
        8 => Some(sen::LrfIndex::L8),
        9 => Some(sen::LrfIndex::L9),
        10 => Some(sen::LrfIndex::L10),
        11 => Some(sen::LrfIndex::L11),
        12 => Some(sen::LrfIndex::L12),
        13 => Some(sen::LrfIndex::L13),
        14 => Some(sen::LrfIndex::L14),
        15 => Some(sen::LrfIndex::L15),
        16 => Some(sen::LrfIndex::L16),
        17 => Some(sen::LrfIndex::L17),
        18 => Some(sen::LrfIndex::L18),
        19 => Some(sen::LrfIndex::L19),
        20 => Some(sen::LrfIndex::L20),
        21 => Some(sen::LrfIndex::L21),
        22 => Some(sen::LrfIndex::L22),
        23 => Some(sen::LrfIndex::L23),
        24 => Some(sen::LrfIndex::L24),
        25 => Some(sen::LrfIndex::L25),
        26 => Some(sen::LrfIndex::L26),
        27 => Some(sen::LrfIndex::L27),
        28 => Some(sen::LrfIndex::L28),
        29 => Some(sen::LrfIndex::L29),
        30 => Some(sen::LrfIndex::L30),
        31 => Some(sen::LrfIndex::L31),
        _ => None,
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::units::Row;

    /// A snapshot with one operand slot filled, which is all most of these need.
    fn with_operand(name: OperandName, port: Option<sen::Port>) -> UnrollOperands {
        let mut operands = UnrollOperands::default();
        operands.operand_list[name.slot()] = port;
        operands
    }

    /// `sentient.load_and_send` with everything fixed but the burst size.
    fn load_and_send(burst_size: Elements) -> sen::Op {
        sen::Op::LoadAndSend {
            mutable_addr: Val(1),
            immutable_addr: Val(2),
            increment: Val(3),
            consumer: SendEnd::to_self(Val(4)),
            result: Val(5),
            extent: sen::Extent {
                total_elements: Elements(64),
                element_size: Bits(32),
                chunk_size: Elements(1),
                chunk_stride: Elements(1),
                burst_size,
            },
            interleaved_group: Elements(0),
            rotate_val: None,
            dir: None,
            shuffle_mode: sen::ShuffleMode::NoShuffle,
            reg: sen::Reg {
                locale: sen::RegType::Lar,
                index: None,
            },
            dbg_name: None,
        }
    }

    /// e113_isXrfRdOp — an `xrf` in any slot is an XRF read, and a memory unit is never one.
    #[test]
    fn is_xrf_rd_op_sees_an_xrf_operand_unless_the_snapshot_is_a_memory_unit() {
        let mut operands = with_operand(OperandName::OpB, Some(sen::Port::Xrf));
        assert!(operands.is_xrf_rd_op());

        operands.is_memory_unit = true;
        assert!(!operands.is_xrf_rd_op());
    }

    /// e114_isXrfWtOp — only the RESULT slot's forwarding counts as a write.
    #[test]
    fn is_xrf_wt_op_reads_only_the_result_slots_forwarding() {
        let mut operands = UnrollOperands::default();
        operands.forwarding_list[OperandName::OpA.slot()] = vec![sen::Port::Xrf];
        assert!(!operands.is_xrf_wt_op());

        operands.forwarding_list[OperandName::Result.slot()] =
            vec![sen::Port::Lrf(sen::LrfIndex::L0), sen::Port::Xrf];
        assert!(operands.is_xrf_wt_op());
    }

    /// e115_reset — every slot empties and `unroll_size_` returns to 1, but the three fields the
    /// reference never writes stay exactly as they were.
    #[test]
    fn reset_empties_the_slots_and_leaves_the_memory_unit_flag_standing() {
        let mut operands = with_operand(OperandName::OpA, Some(sen::Port::Lrf(sen::LrfIndex::L4)));
        operands.forwarding_list[OperandName::Result.slot()] = vec![sen::Port::Xrf];
        operands.is_field_unroll[OperandName::Result.slot()] = true;
        operands.unroll_size = UnrollSize(4);
        operands.xrf_write_incr = XrfIncr(4);
        operands.op_name = Some(RolledOp::Mac);
        operands.precisions = vec![sen::Precision::Fp16];
        operands.mask_value = Some(Val(9));
        operands.is_splat_promoted = true;
        operands.is_memory_unit = true;

        operands.reset();

        assert_eq!(operands.operand_list[OperandName::OpA.slot()], None);
        assert!(operands.forwarding_list[OperandName::Result.slot()].is_empty());
        assert!(!operands.is_field_unroll[OperandName::Result.slot()]);
        assert_eq!(operands.unroll_size, UnrollSize::ONE);
        assert_eq!(operands.xrf_write_incr, XrfIncr::ZERO);
        assert_eq!(operands.op_name, None);
        assert!(operands.precisions.is_empty());
        assert_eq!(operands.fold_mode, sen::FoldMode::FoldA);
        // The three survivors.
        assert_eq!(operands.mask_value, Some(Val(9)));
        assert!(operands.is_splat_promoted);
        assert!(operands.is_memory_unit);
    }

    /// e116_getUnrollOperandUsingPort — the PT row swaps `op_b` and `op_c`; everything else does not.
    #[test]
    fn unroll_operand_using_port_swaps_op_b_and_op_c_on_a_pt_row() {
        let pt = DfirUnit::PtRow(Row::checked(0).expect("PT row 0 exists"));

        assert_eq!(
            UnrollOperands::unroll_operand_using_port(pt, ComputePortId::Port1),
            OperandName::OpC
        );
        assert_eq!(
            UnrollOperands::unroll_operand_using_port(pt, ComputePortId::Port2),
            OperandName::OpB
        );
        assert_eq!(
            UnrollOperands::unroll_operand_using_port(DfirUnit::Lxlu, ComputePortId::Port1),
            OperandName::OpB
        );
        assert_eq!(
            UnrollOperands::unroll_operand_using_port(DfirUnit::Lxlu, ComputePortId::Port2),
            OperandName::OpC
        );
    }

    /// e117_fillMemOpAttrs — two loads that differ ONLY in burst size snapshot identically, which is
    /// what lets an already-rerolled memory op still match its unrerolled neighbour.
    #[test]
    fn fill_mem_op_attrs_snapshots_everything_except_the_burst_size() {
        let mut unbursted = UnrollOperands::default();
        unbursted.fill_mem_op_attrs(&load_and_send(Elements(0)));
        let mut bursted = UnrollOperands::default();
        bursted.fill_mem_op_attrs(&load_and_send(Elements(4)));

        assert_eq!(unbursted.memory_op_attrs, bursted.memory_op_attrs);
        assert_eq!(
            unbursted.memory_op_attrs,
            Some(MemOpAttrs::LoadAndSend {
                extent: UnburstedExtent {
                    total_elements: Elements(64),
                    element_size: Bits(32),
                    chunk_size: Elements(1),
                    chunk_stride: Elements(1),
                },
                interleaved_group: Elements(0),
                rotate_val: None,
                dir: None,
                shuffle_mode: sen::ShuffleMode::NoShuffle,
                reg: sen::Reg {
                    locale: sen::RegType::Lar,
                    index: None,
                },
                dbg_name: None,
            })
        );
    }

    /// e118_areOperandsMatching — an unrolled candidate matches when its LRF index sits exactly
    /// `unroll_size_` above the reference's, and not when the gap is anything else.
    #[test]
    fn are_operands_matching_wants_an_lrf_gap_of_exactly_the_unroll_size() {
        let reference = UnrollOperands {
            unroll_size: UnrollSize(2),
            ..Default::default()
        };
        let candidate = UnrollOperands::default();

        assert!(reference.are_operands_matching(
            Some(sen::Port::Lrf(sen::LrfIndex::L4)),
            true,
            Some(sen::Port::Lrf(sen::LrfIndex::L6)),
            true,
            &candidate,
            true,
        ));
        assert!(!reference.are_operands_matching(
            Some(sen::Port::Lrf(sen::LrfIndex::L4)),
            true,
            Some(sen::Port::Lrf(sen::LrfIndex::L7)),
            true,
            &candidate,
            true,
        ));
    }

    /// e118_areOperandsMatching — the negative the reference reaches by aborting: an `lrf` against an
    /// `istate` runs `getIStateIndex` over a four-character spelling.
    #[test]
    fn are_operands_matching_refuses_a_mixed_lrf_istate_pair() {
        let reference = UnrollOperands::default();
        let candidate = UnrollOperands::default();

        assert!(!reference.are_operands_matching(
            Some(sen::Port::Lrf(sen::LrfIndex::L0)),
            false,
            Some(sen::Port::IState(sen::IStateIndex::S1)),
            false,
            &candidate,
            true,
        ));
    }

    /// e119_createOperand — a field-unrolled `lrf` moves up by the undershoot; anything else, and any
    /// shift that leaves the register file, comes back untouched or as `None`.
    #[test]
    fn create_operand_shifts_a_field_unrolled_lrf_by_the_undershoot() {
        let mut operands = with_operand(OperandName::OpA, Some(sen::Port::Lrf(sen::LrfIndex::L4)));
        operands.unroll_size = UnrollSize(4);

        // Not field-unrolled: the operand exactly as it stands.
        assert_eq!(
            operands.create_operand(OperandName::OpA, UnrollSize(2)),
            Some(sen::Port::Lrf(sen::LrfIndex::L4))
        );

        operands.is_field_unroll[OperandName::OpA.slot()] = true;
        assert_eq!(
            operands.create_operand(OperandName::OpA, UnrollSize(2)),
            Some(sen::Port::Lrf(sen::LrfIndex::L6))
        );

        // `lrf31 + 3` is the reference's `.value()` on a `std::nullopt`.
        operands.operand_list[OperandName::OpA.slot()] = Some(sen::Port::Lrf(sen::LrfIndex::L31));
        assert_eq!(
            operands.create_operand(OperandName::OpA, UnrollSize(1)),
            None
        );
    }

    /// e120_createForwardingArray — only the RESULT slot's `lrf` targets move, and only its `lrf`
    /// ones.
    #[test]
    fn create_forwarding_array_shifts_only_the_result_slots_lrf_targets() {
        let mut operands = UnrollOperands {
            unroll_size: UnrollSize(4),
            ..Default::default()
        };
        operands.forwarding_list[OperandName::Result.slot()] =
            vec![sen::Port::Lrf(sen::LrfIndex::L2), sen::Port::Xrf];
        operands.forwarding_list[OperandName::OpA.slot()] = vec![sen::Port::Lrf(sen::LrfIndex::L2)];
        operands.is_field_unroll[OperandName::Result.slot()] = true;
        operands.is_field_unroll[OperandName::OpA.slot()] = true;

        assert_eq!(
            operands.create_forwarding_array(OperandName::Result, UnrollSize(1)),
            Some(vec![sen::Port::Lrf(sen::LrfIndex::L5), sen::Port::Xrf])
        );
        assert_eq!(
            operands.create_forwarding_array(OperandName::OpA, UnrollSize(1)),
            Some(vec![sen::Port::Lrf(sen::LrfIndex::L2)])
        );
    }

    /// Both blocks in key order over a mixed state: a port-backed slot, the `"NA"` sentinel for the
    /// untouched ones, and the flag as a digit.
    #[test]
    fn dump_prints_five_operands_then_five_flags_in_key_order() {
        let mut ops = UnrollOperands::default();
        ops.operand_list[OperandName::OpA.slot()] = Some(sen::Port::Lrf(sen::LrfIndex::L3));
        ops.operand_list[OperandName::Result.slot()] =
            Some(sen::Port::IState(sen::IStateIndex::S2));
        ops.is_field_unroll[OperandName::Result.slot()] = true;

        assert_eq!(
            ops.dump(),
            concat!(
                "dumping:\n",
                "op_a: lrf3\n",
                "op_b: NA\n",
                "op_c: NA\n",
                "result: istate2\n",
                "logical_result: NA\n",
                "op_a: 0\n",
                "op_b: 0\n",
                "op_c: 0\n",
                "result: 1\n",
                "logical_result: 0\n",
            )
        );
    }
}

// crustify:todo: e335_fill
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:502  (165 body lines, level 1)
//   original  : void UnrollOperands::fill(Operation *op, SenComponents type)
//   calls     : e115_reset, e116_getUnrollOperandUsingPort, e117_fillMemOpAttrs

// crustify:todo: e336_match
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:731  (149 body lines, level 1)
//   original  : bool UnrollOperands::match(UnrollOperands &new_operand_list)
//   calls     : e118_areOperandsMatching, e252_size

// crustify:todo: e337_updateUnrollInfo
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:881  (44 body lines, level 1)
//   original  : void UnrollOperands::updateUnrollInfo(UnrollOperands &new_operand_list)
//   calls     : e252_size

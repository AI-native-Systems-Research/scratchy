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

//! `ScalarOpMergingAndHoisting.cpp` — 12 of the campaign's 656 units (dependency level(s) [0, 1, 6, 7]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e157_doesImmutableImmExceedRange` | 157 | 0 | 18 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:145` |
//! | `e158_doesValueExceedLRFRange` | 158 | 0 | 21 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:167` |
//! | `e159_computeAddressScale` | 159 | 0 | 7 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:266` |
//! | `e160_MemoryOpInfo` | 160 | 0 | 38 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:288` |
//! | `e161_addSpeculativeImmutable` | 161 | 0 | 3 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:389` |
//! | `e162_addOpToBlock` | 162 | 0 | 4 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:429` |
//! | `e163_setEnablesMerging` | 163 | 0 | 4 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:434` |
//! | `e164_addUnrollCandidate` | 164 | 0 | 4 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:441` |
//! | `e361_isImmutableValueInRange` | 361 | 1 | 28 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:232` |
//! | `e366_runOn` | 366 | 1 | 130 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2258` |
//! | `e636_runOn` | 636 | 6 | 14 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2389` |
//! | `e646_runOnOperation` | 646 | 7 | 5 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2404` |

#![allow(dead_code)]
// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET — `e646_runOnOperation` (level 7) is the unit that
// calls everything below, and it is not in this batch. CI runs clippy with `-D warnings`, so without
// this the first ported leaf of the module fails the gate.
// ⭐ REMOVE THIS WITH e646: at that point an unused item here is a real defect again.

use std::num::NonZeroU32;

use super::analyses::{EvaluatedValue, Evaluation, InstructionCount, ScalarOffset};
use crate::arch::{Arch, Elements};
use crate::formats::Bits;
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::islands::sentient::dialects::sentient as ops;
use crate::islands::sentient::dialects::{Op, Val};

pub(crate) mod scalar_op_hoisting;
pub(crate) mod scalar_op_merging;

// ───────────────────────────────────────────────────────────────────────────────────────────────
// The types the four range/scale units are stated in.
// ───────────────────────────────────────────────────────────────────────────────────────────────

/// THE FOUR COMPONENTS THIS PASS RUNS ON — `is_any_of(gen_comp, LXLU, LXSU, L0LU, L0SU)` as a TYPE
/// (`ScalarOpMergingAndHoisting.cpp:2270-2273`).
///
/// ⭐ THE `DT_CHECK` IS THE CONSTRUCTOR. `doesValueExceedLRFRange`'s check (`:173`) and
/// `computeAddressScale`'s `return -1` (`:271`) are the same fact stated twice; with the set closed
/// here neither is reachable, so nothing below has to refuse at run time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScalarOpComp {
    /// `LXLU` — the LX load unit.
    Lxlu,
    /// `LXSU` — the LX store unit.
    Lxsu,
    /// `L0LU` — the L0 load unit.
    L0lu,
    /// `L0SU` — the L0 store unit.
    L0su,
}

impl ScalarOpComp {
    /// `is_LX_L0` (`:2270-2273`) — `None` for every other unit, which is the flag the caller's own
    /// branch turns on and the reason Scalar Op Merging is skipped entirely elsewhere.
    #[must_use]
    pub(crate) const fn of(comp: GenericComp) -> Option<ScalarOpComp> {
        match comp {
            GenericComp::Lxlu => Some(ScalarOpComp::Lxlu),
            GenericComp::Lxsu => Some(ScalarOpComp::Lxsu),
            GenericComp::L0lu => Some(ScalarOpComp::L0lu),
            GenericComp::L0su => Some(ScalarOpComp::L0su),
            GenericComp::Pt
            | GenericComp::Pe
            | GenericComp::Sfp
            | GenericComp::Lx
            | GenericComp::L3lu
            | GenericComp::L3su
            | GenericComp::Hbm
            | GenericComp::LxVirtualIbr
            | GenericComp::L3Ibr
            | GenericComp::CrossPtnLink
            | GenericComp::SfpState
            | GenericComp::PeState
            | GenericComp::L0
            | GenericComp::Constant
            | GenericComp::SfpRing
            | GenericComp::LxluScaleReg => None,
        }
    }
}

/// `address_granularity_scale` — the DIVISOR every scaled immediate goes through.
///
/// ⛔ NON-ZERO BECAUSE IT IS A FLOATING-POINT DIVISOR: the reference divides by
/// `(float)address_granularity_scale` (`:151`, `:179`), so a zero would make every scaled immediate
/// an infinity, and an infinity compares INSIDE no range and outside every one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AddressScale(NonZeroU32);

impl AddressScale {
    /// `= 1` — the table's entry for both LX units and for the L0 LOAD (`sysdef.cpp:538-542`).
    pub(crate) const ONE: AddressScale = AddressScale(NonZeroU32::new(1).expect("1 is not zero"));

    /// `= numPTRows` — the L0 STORE's entry, because a store fans across the rows
    /// (`sysdef.cpp:543`).
    #[must_use]
    pub(crate) fn rows<A: Arch>() -> AddressScale {
        // ⛔ `const { }`, so an arch that claimed no PT rows fails the BUILD where it is instantiated
        // rather than dividing an address by zero. Same mechanism as `Bounded::at`.
        const {
            assert!(
                A::PT_ROWS > 0,
                "an arch with no PT rows has no L0 store granularity"
            )
        }
        AddressScale(NonZeroU32::new(A::PT_ROWS).expect("the const above admits only a non-zero"))
    }

    /// The divisor itself, for the one place it becomes arithmetic.
    #[must_use]
    pub(crate) const fn get(self) -> u32 {
        self.0.get()
    }
}

/// `std::pair<int, int> ldsti_imm_range` — the INCLUSIVE window an immediate may occupy, which
/// `getImmRange` derives from the ISA's own LDSTI field width (`:2274-2290`, e366's).
///
/// ⛔ NAMED ENDS, NOT A PAIR: `.first` is the low bound and `.second` the high one at four call
/// sites, and a swapped pair rejects everything while type-checking perfectly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ImmRange {
    /// `.first` — the most negative value that fits.
    pub(crate) min: i32,
    /// `.second` — the largest value that fits.
    pub(crate) max: i32,
}

impl ImmRange {
    /// `imm < range.first || imm > range.second` (`:152-153`) — inclusive at both ends.
    #[must_use]
    pub(crate) const fn excludes(self, imm: i32) -> bool {
        imm < self.min || imm > self.max
    }
}

/// `offset * (float)element_size / 8.0 / (float)address_granularity_scale`, truncated to `int` — the
/// one arithmetic both range scans apply to every offset they can see (`:150`, `:157`, `:178`).
///
/// ⛔⛔ THE FIRST PRODUCT IS SINGLE PRECISION AND THAT IS LOAD-BEARING. `(float)element_size` makes
/// `offset * element_size` an `f32`; only the `/ 8.0` promotes to `double`. Computing the whole chain
/// in `f64` moves the rounding boundary past 2^24 and changes the truncated result.
fn scaled_imm(offset: ScalarOffset, element_size: Bits, scale: AddressScale) -> i32 {
    let product = offset.0 as f32 * element_size.0 as f32;
    (f64::from(product) / 8.0 / f64::from(scale.get())) as i32
}

/// Replaces: e157_doesImmutableImmExceedRange
///
/// TRUE IF ANY OFFSET OF `immutable_addr_ev`, SCALED, FALLS OUTSIDE THE LDSTI IMM WINDOW.
///
/// ⭐ ONE SCAN FOR BOTH `dyn_cast` ARMS (`:149-163`): they differ only in how they REACH the offsets,
/// and [`Evaluation::offset_values`] is that difference — the trailing `return false` covers the
/// neither-kind case, which [`super::analyses::Offsets`] makes unreachable.
pub(crate) fn does_immutable_imm_exceed_range(
    immutable_addr_ev: &Evaluation,
    element_size: Bits,
    ldsti_imm_range: ImmRange,
    address_granularity_scale: AddressScale,
) -> bool {
    immutable_addr_ev.offset_values().any(|offset| {
        ldsti_imm_range.excludes(scaled_imm(offset, element_size, address_granularity_scale))
    })
}

/// Replaces: e158_doesValueExceedLRFRange
///
/// TRUE IF ANY OFFSET OF `value_ev`, SCALED, FALLS OUTSIDE THE UNIT'S LRF WINDOW.
///
/// ⛔ THE WINDOW IS ASYMMETRIC AND ONE BIT WIDER THAN A SIGNED FIELD: `-(0x1 << bitSize)` to
/// `(0x1 << bitSize) - 1` (`:174-176`) — ⛔ NOT `-(1 << (bitSize - 1))`.
///
/// ⭐ `sys_def.regInfoPerUnit.at(comp).at(RegType::LRF).bitSize` IS [`Arch::LX_LRF_BITS`] /
/// [`Arch::L0_LRF_BITS`], and the `DT_CHECK` at `:173` is [`ScalarOpComp`] instead of a check.
pub(crate) fn does_value_exceed_lrf_range<A: Arch>(
    value_ev: &Evaluation,
    element_size: Bits,
    comp: ScalarOpComp,
    address_granularity_scale: AddressScale,
) -> bool {
    let reg_bitwidth = match comp {
        ScalarOpComp::Lxlu | ScalarOpComp::Lxsu => A::LX_LRF_BITS.get(),
        ScalarOpComp::L0lu | ScalarOpComp::L0su => A::L0_LRF_BITS.get(),
    };
    let lrf_range = ImmRange {
        min: -(1_i32 << reg_bitwidth),
        max: (1_i32 << reg_bitwidth) - 1,
    };
    value_ev.offset_values().any(|offset| {
        lrf_range.excludes(scaled_imm(offset, element_size, address_granularity_scale))
    })
}

/// Replaces: e159_computeAddressScale
///
/// `getAddressGranularityScale({comp, LX})` for the two LX units and `({comp, L0})` for the two L0
/// ones — 1, 1, 1 and `numPTRows` respectively (`sysdef.cpp:538-543`).
///
/// ⛔⛔ DO NOT REUSE `reginit::address_scale`: that one also applies `dsc2.cpp:2953-2957`'s
/// `if (genericUnit == L0LU) addrScale *= numPTRows`, which `getAddressGranularityScale` does NOT —
/// this is the raw table lookup. Reusing it scales every L0 LOAD immediate by the row count.
///
/// ⭐ THE `return -1` (`:271`) IS UNREACHABLE because [`ScalarOpComp`] closes the set at four.
#[must_use]
pub(crate) fn compute_address_scale<A: Arch>(comp: ScalarOpComp) -> AddressScale {
    match comp {
        ScalarOpComp::Lxlu | ScalarOpComp::Lxsu | ScalarOpComp::L0lu => AddressScale::ONE,
        ScalarOpComp::L0su => AddressScale::rows::<A>(),
    }
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
// The pass's own data structures.
// ───────────────────────────────────────────────────────────────────────────────────────────────

/// `if (b > 0) burst_ = b;` — the guard both `getBurstSize()` and `getInterleavedGroup()` go through
/// (`:293-297`), which keeps the field's DEFAULT when the op says zero.
const fn positive_or(value: Elements, default: Elements) -> Elements {
    if value.0 > 0 { value } else { default }
}

/// THE ATTRIBUTES A MEMORY OP CARRIES — `struct MemoryOpInfo` (`:283-336`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MemoryOpInfo {
    /// `mutable_addr_` — the double buffer's toggling half.
    pub(crate) mutable_addr: Val,
    /// `immutable_addr_` — the fixed base, and the operand every range test above is about.
    pub(crate) immutable_addr: Val,
    /// `increment_`.
    pub(crate) increment: Val,
    /// `burst_` — ⛔ DEFAULT **1**, not 0: an unbursted transfer's zero fails the `> 0` guard.
    pub(crate) burst: Elements,
    /// `il_` — the interleaved group, default 0 under the same guard.
    pub(crate) il: Elements,
    /// `element_size_` — a width in BITS, and what the address is computed in.
    pub(crate) element_size: Bits,
    /// `total_elements_`, default 0.
    pub(crate) total_elements: Elements,
    /// `chunk_size_`, default 0.
    pub(crate) chunk_size: Elements,
    /// `chunk_stride_`, default 0.
    pub(crate) chunk_stride: Elements,
}

impl MemoryOpInfo {
    /// Replaces: e160_MemoryOpInfo
    ///
    /// `explicit MemoryOpInfo(Operation *op)` — the LAS / RAS / LCAS `dyn_cast` chain (`:287-317`).
    ///
    /// ⛔ THE THREE ARMS FILL DIFFERENT FIELDS: RAS leaves `chunk_size`/`chunk_stride` at **0** and
    /// LCAS leaves `burst` at 1, `il` and `total_elements` at 0, even though the ops carry those —
    /// and LCAS takes `getDstElementSize()`, not the source's, because that is what the address is
    /// computed in (`:314-316`).
    ///
    /// ⭐ `None` RATHER THAN THE REFERENCE'S UNINITIALISED `element_size_`: its fourth (implicit)
    /// arm leaves that field garbage, and all five construction sites are already guarded by
    /// `isa<LAS, RAS, LCAS>` — so the defect is unreachable and is not copied here.
    #[must_use]
    pub(crate) fn of(op: &Op) -> Option<MemoryOpInfo> {
        match op {
            Op::Sentient(ops::Op::LoadAndSend {
                mutable_addr,
                immutable_addr,
                increment,
                extent,
                interleaved_group,
                ..
            }) => Some(MemoryOpInfo {
                mutable_addr: *mutable_addr,
                immutable_addr: *immutable_addr,
                increment: *increment,
                burst: positive_or(extent.burst_size, Elements(1)),
                il: positive_or(*interleaved_group, Elements(0)),
                element_size: extent.element_size,
                total_elements: extent.total_elements,
                chunk_size: extent.chunk_size,
                chunk_stride: extent.chunk_stride,
            }),
            Op::Sentient(ops::Op::ReceiveAndStore {
                mutable_addr,
                immutable_addr,
                increment,
                extent,
                interleaved_group,
                ..
            }) => Some(MemoryOpInfo {
                mutable_addr: *mutable_addr,
                immutable_addr: *immutable_addr,
                increment: *increment,
                burst: positive_or(extent.burst_size, Elements(1)),
                il: positive_or(*interleaved_group, Elements(0)),
                element_size: extent.element_size,
                total_elements: extent.total_elements,
                chunk_size: Elements(0),
                chunk_stride: Elements(0),
            }),
            Op::Sentient(ops::Op::LoadComputeAndSend {
                mutable_addr,
                immutable_addr,
                increment,
                dst_element_size,
                ..
            }) => Some(MemoryOpInfo {
                mutable_addr: *mutable_addr,
                immutable_addr: *immutable_addr,
                increment: *increment,
                burst: Elements(1),
                il: Elements(0),
                element_size: *dst_element_size,
                total_elements: Elements(0),
                chunk_size: Elements(0),
                chunk_stride: Elements(0),
            }),
            _ => None,
        }
    }
}

/// HOW ONE OPERATION WILL BE MODIFIED — `class OperationData` (`:338-368`).
///
/// ⭐ THE OP IS ITS BOUND RESULT. `applyOperationData` handles only add, sub, LAS, RAS and LCAS and
/// is `llvm_unreachable` otherwise (`:1747-1793`), and every one of those binds `getResult(0)` — so
/// the result value is the identity, and `defining_op` is the walk back to the op itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OperationData {
    /// `op_`.
    pub(crate) op: Val,
    /// `mod_` — the amount to modify by. ⛔ NOT OPTIONAL: both constructors take it and
    /// `applyOperationData` dereferences it unconditionally (`:1744`, `:1762`).
    pub(crate) mod_by: EvaluatedValue,
    /// `merging_increment_` — the merging increment AT THIS operation, not the block's running one.
    pub(crate) merging_increment: EvaluatedValue,
    /// `replace_with_mod_` — replace by `mod_` instead of modifying by it.
    pub(crate) replace_with_mod: bool,
}

/// A FIELD UNROLL CANDIDATE — `class FieldUnrollData` (`:370-402`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct FieldUnrollData {
    /// `op_ = nullptr` — `None` for the default-constructed candidate, which the reference relies on.
    pub(crate) op: Option<Val>,
    /// `cost_` — the additional IBuff entries unrolling this op would need
    /// (`calculateIBuffRequired`, e165).
    pub(crate) cost: InstructionCount,
    /// `speculative_immutables_` — one per memory op the unrolling would create.
    pub(crate) speculative_immutables: Vec<EvaluatedValue>,
    /// `enables_merging_`.
    pub(crate) enables_merging: bool,
    /// `marked_for_unrolling_`.
    pub(crate) marked_for_unrolling: bool,
}

impl FieldUnrollData {
    /// Replaces: e161_addSpeculativeImmutable
    ///
    /// `speculative_immutables_.push_back(&speculative_immutable)` (`:389-391`) — appends the
    /// immutable one of the memory ops this unrolling would create is going to need.
    pub(crate) fn add_speculative_immutable(&mut self, speculative_immutable: EvaluatedValue) {
        self.speculative_immutables.push(speculative_immutable);
    }
}

/// HOW MANY ADD/SUBS A MERGING BLOCK HOLDS — `unsigned num_scalar_ops_in_block_` (`:452`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ScalarOpCount(pub(crate) u32);

/// A LINEAR CHAIN ELIGIBLE FOR SCALAR OP MERGING — `class ScalarOpMergingBlock` (`:404-454`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ScalarOpMergingBlock {
    /// `input_value_to_block_` — the value from OUTSIDE the block feeding the chain. `None` is the
    /// default-constructed null `Value` the reference starts with, before `setInputValueToBlock`.
    pub(crate) input_value_to_block: Option<Val>,
    /// `block_ops_`.
    pub(crate) block_ops: Vec<OperationData>,
    /// `unroll_candidates_`.
    pub(crate) unroll_candidates: Vec<FieldUnrollData>,
    /// `num_scalar_ops_in_block_`.
    pub(crate) num_scalar_ops_in_block: ScalarOpCount,
    /// `required_ibuff_`.
    pub(crate) required_ibuff: InstructionCount,
}

impl ScalarOpMergingBlock {
    /// Replaces: e162_addOpToBlock
    ///
    /// `block_ops_.push_back(op_data); if (is_add_or_sub) ++num_scalar_ops_in_block_;` (`:429-432`).
    ///
    /// ⛔ THE COUNT IS NOT THE LENGTH. Only an add/sub bumps it, so a block of three memory ops and
    /// one add reports ONE — and that is the number `isProfitableForHoisting` (e166) weighs.
    pub(crate) fn add_op_to_block(&mut self, op_data: OperationData, is_add_or_sub: bool) {
        self.block_ops.push(op_data);
        if is_add_or_sub {
            self.num_scalar_ops_in_block.0 += 1;
        }
    }

    /// Replaces: e163_setEnablesMerging
    ///
    /// `if (!unroll_candidates_.empty()) unroll_candidates_.back().setEnablesMerging();`
    /// (`:434-437`).
    ///
    /// ⛔ IT MARKS THE **LAST CANDIDATE**, NOT THE BLOCK, and does nothing at all when there is no
    /// candidate yet — the same-named one-line setter on [`FieldUnrollData`] is the other half.
    pub(crate) fn set_enables_merging(&mut self) {
        if let Some(candidate) = self.unroll_candidates.last_mut() {
            candidate.enables_merging = true;
        }
    }

    /// Replaces: e164_addUnrollCandidate
    ///
    /// `unroll_candidates_.push_back(unroll_candidate); required_ibuff_ += unroll_candidate.getCost();`
    /// (`:441-444`) — the block's IBuff requirement is charged as the candidate is admitted.
    pub(crate) fn add_unroll_candidate(&mut self, unroll_candidate: FieldUnrollData) {
        self.required_ibuff.0 += unroll_candidate.cost.0;
        self.unroll_candidates.push(unroll_candidate);
    }
}

// crustify:todo: e361_isImmutableValueInRange
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:232  (28 body lines, level 1)
//   original  : bool isImmutableValueInRange(const EvaluatedValue &new_immutable_ev, const EvaluatedValue &original_immutable_ev, int element_size, mlir::Operation *op) const
//   calls     : e157_doesImmutableImmExceedRange, e158_doesValueExceedLRFRange

// crustify:todo: e366_runOn
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2258  (130 body lines, level 1)
//   original  : void runOn(dataflow::ProgramUnitOp unit, ModuleOp module_op)
//   calls     : e252_size

// crustify:todo: e636_runOn
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2389  (14 body lines, level 6)
//   original  : void runOn(ModuleOp module_op)
//   calls     : e366_runOn, e597_runLightWeightSimplifications

// crustify:todo: e646_runOnOperation
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2404  (5 body lines, level 7)
//   original  : void runOnOperation()
//   calls     : e366_runOn, e636_runOn

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::{Dd2, Sen1p5};
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::transform::sentient::analyses::Offsets;

    /// An evaluation of one all-unit offset.
    fn all_unit(offset: i64) -> Evaluation {
        Evaluation {
            known_absolute: true,
            base: None,
            offsets: Offsets::AllUnit(ScalarOffset(offset)),
        }
    }

    /// An evaluation of one offset PER UNIT, keyed as the header keys them.
    fn per_unit(offsets: &[i64]) -> Evaluation {
        Evaluation {
            known_absolute: true,
            base: None,
            offsets: Offsets::PerUnit(
                offsets
                    .iter()
                    .enumerate()
                    .map(|(i, &o)| (Val(i as u32), ScalarOffset(o)))
                    .collect(),
            ),
        }
    }

    /// A register-less `sentient.*` scalar placement, which none of these units reads.
    fn no_reg() -> ops::Reg {
        ops::Reg {
            locale: ops::RegType::Unknown,
            index: None,
        }
    }

    /// ⭐ BOTH `dyn_cast` ARMS AND THE WINDOW'S EDGES. `element_size = 32` and `scale = 1` make the
    /// scaled immediate `offset * 4`, so 31 lands on 124 and 32 on 128 — one inside a ±127 window and
    /// one out. The per-unit arm must fail on ANY entry, not just the first.
    #[test]
    fn an_immutable_imm_exceeds_the_range_on_any_offset_of_either_kind() {
        let range = ImmRange {
            min: -128,
            max: 127,
        };
        let scale = AddressScale::ONE;
        let size = Bits(32);
        assert!(!does_immutable_imm_exceed_range(
            &all_unit(31),
            size,
            range,
            scale
        ));
        assert!(does_immutable_imm_exceed_range(
            &all_unit(32),
            size,
            range,
            scale
        ));
        assert!(!does_immutable_imm_exceed_range(
            &per_unit(&[0, 31, -32]),
            size,
            range,
            scale
        ));
        assert!(does_immutable_imm_exceed_range(
            &per_unit(&[0, 1, 32]),
            size,
            range,
            scale
        ));
    }

    /// ⛔ THE ASYMMETRY IS THE TEST. Dd2's L0 LRF is 10 bits, so the window is `-1024 ..= 1023` — a
    /// full bit wider on the negative side than a signed 10-bit field, which is what `-(0x1 << 10)`
    /// says. With `element_size = 8` and `scale = 1` the offset IS the immediate.
    #[test]
    fn the_lrf_window_is_one_bit_wider_than_a_signed_field_and_asymmetric() {
        let scale = AddressScale::ONE;
        let size = Bits(8);
        let comp = ScalarOpComp::L0lu;
        assert!(!does_value_exceed_lrf_range::<Dd2>(
            &all_unit(1023),
            size,
            comp,
            scale
        ));
        assert!(does_value_exceed_lrf_range::<Dd2>(
            &all_unit(1024),
            size,
            comp,
            scale
        ));
        assert!(!does_value_exceed_lrf_range::<Dd2>(
            &all_unit(-1024),
            size,
            comp,
            scale
        ));
        assert!(does_value_exceed_lrf_range::<Dd2>(
            &all_unit(-1025),
            size,
            comp,
            scale
        ));
        // ⭐ THE LX LRF IS 21 BITS ON BOTH ARCHS, so the same offset is comfortably inside it.
        assert!(!does_value_exceed_lrf_range::<Dd2>(
            &all_unit(1024),
            size,
            ScalarOpComp::Lxlu,
            scale
        ));
    }

    /// ⛔ THE L0 STORE'S SCALE IS THE ROW COUNT AND THE L0 LOAD'S IS 1 — the same memory, two units,
    /// two scales. And the store's differs per arch, which is why the unit is generic.
    #[test]
    fn only_the_l0_store_scales_by_the_pt_rows() {
        assert_eq!(compute_address_scale::<Dd2>(ScalarOpComp::Lxlu).get(), 1);
        assert_eq!(compute_address_scale::<Dd2>(ScalarOpComp::Lxsu).get(), 1);
        assert_eq!(compute_address_scale::<Dd2>(ScalarOpComp::L0lu).get(), 1);
        assert_eq!(compute_address_scale::<Dd2>(ScalarOpComp::L0su).get(), 8);
        assert_eq!(compute_address_scale::<Sen1p5>(ScalarOpComp::L0su).get(), 4);
        assert_eq!(
            ScalarOpComp::of(GenericComp::L0su),
            Some(ScalarOpComp::L0su)
        );
        assert_eq!(ScalarOpComp::of(GenericComp::L3lu), None);
    }

    /// ⛔ THE THREE ARMS FILL DIFFERENT FIELDS. An unbursted LAS reports `burst = 1`, a RAS that
    /// carries a chunking reports `chunk_size = 0` anyway, and an LCAS reports the DESTINATION
    /// element size with every extent field left at its default.
    #[test]
    fn each_memory_op_kind_fills_its_own_subset_of_the_info() {
        let mut extent = ops::Extent::of(Elements(64), Bits(16));
        extent.chunk_size = Elements(4);
        extent.chunk_stride = Elements(8);

        let las = Op::Sentient(ops::Op::LoadAndSend {
            mutable_addr: Val(1),
            immutable_addr: Val(2),
            increment: Val(3),
            consumer: SendEnd::to_self(Val(0)),
            result: Val(4),
            extent,
            interleaved_group: Elements(0),
            rotate_val: None,
            dir: None,
            shuffle_mode: ops::ShuffleMode::NoShuffle,
            reg: no_reg(),
            dbg_name: None,
        });
        let info = MemoryOpInfo::of(&las).expect("a load_and_send is a memory op");
        assert_eq!(info.immutable_addr, Val(2));
        assert_eq!(info.burst, Elements(1));
        assert_eq!(info.il, Elements(0));
        assert_eq!(info.element_size, Bits(16));
        assert_eq!(info.total_elements, Elements(64));
        assert_eq!(info.chunk_size, Elements(4));
        assert_eq!(info.chunk_stride, Elements(8));

        let ras = Op::Sentient(ops::Op::ReceiveAndStore {
            mutable_addr: Val(1),
            immutable_addr: Val(2),
            increment: Val(3),
            producer: ops::StoreSource::Constant(Val(0)),
            result: Val(4),
            dst: None,
            drop_first: None,
            multicast_info: None,
            extent,
            interleaved_group: Elements(2),
            coalesce: false,
            subword_length: 0,
            stride: 0,
            permute: false,
            shuffle_mode: None,
            reg: no_reg(),
            dbg_name: None,
        });
        let info = MemoryOpInfo::of(&ras).expect("a receive_and_store is a memory op");
        assert_eq!(info.il, Elements(2));
        assert_eq!(info.total_elements, Elements(64));
        assert_eq!(info.chunk_size, Elements(0));
        assert_eq!(info.chunk_stride, Elements(0));

        let lcas = Op::Sentient(ops::Op::LoadComputeAndSend {
            mutable_addr: Val(1),
            immutable_addr: Val(2),
            increment: Val(3),
            element_index: Val(4),
            scale_index: Val(5),
            consumer: SendEnd::to_self(Val(0)),
            result: Val(6),
            src_total_elements: Elements(64),
            dst_total_elements: Elements(32),
            src_element_size: Bits(16),
            dst_element_size: Bits(8),
            dir: None,
            shuffle_mode: ops::ShuffleMode::NoShuffle,
            reg: no_reg(),
            dbg_name: None,
        });
        let info = MemoryOpInfo::of(&lcas).expect("a load_compute_and_send is a memory op");
        assert_eq!(info.element_size, Bits(8));
        assert_eq!(info.burst, Elements(1));
        assert_eq!(info.total_elements, Elements(0));

        assert_eq!(
            MemoryOpInfo::of(&Op::Sentient(ops::Op::Nop { dbg_name: None })),
            None
        );
    }

    /// One `OperationData` naming `op`.
    fn op_data(op: Val) -> OperationData {
        OperationData {
            op,
            mod_by: EvaluatedValue(0),
            merging_increment: EvaluatedValue(0),
            replace_with_mod: false,
        }
    }

    /// One candidate of `cost`.
    fn candidate(op: Val, cost: InstructionCount) -> FieldUnrollData {
        FieldUnrollData {
            op: Some(op),
            cost,
            ..FieldUnrollData::default()
        }
    }

    /// ⭐ THE LIST GROWS IN ORDER, which is what `getSpeculativeImmutables` hands the unroller.
    #[test]
    fn a_speculative_immutable_is_appended_to_the_candidate() {
        let mut unroll_candidate = FieldUnrollData::default();
        unroll_candidate.add_speculative_immutable(EvaluatedValue(7));
        unroll_candidate.add_speculative_immutable(EvaluatedValue(9));
        assert_eq!(
            unroll_candidate.speculative_immutables,
            vec![EvaluatedValue(7), EvaluatedValue(9)]
        );
        assert_eq!(unroll_candidate.op, None);
    }

    /// ⛔ THE COUNT IS NOT THE LENGTH: two ops in the block, one of them an add, counts ONE.
    #[test]
    fn only_an_add_or_sub_bumps_the_scalar_op_count() {
        let mut block = ScalarOpMergingBlock::default();
        block.add_op_to_block(op_data(Val(1)), true);
        block.add_op_to_block(op_data(Val(2)), false);
        assert_eq!(block.block_ops.len(), 2);
        assert_eq!(block.num_scalar_ops_in_block, ScalarOpCount(1));
    }

    /// ⛔ IT MARKS THE LAST CANDIDATE ONLY, and an empty list is a silent no-op rather than a refusal.
    #[test]
    fn enabling_merging_marks_the_last_candidate_and_an_empty_list_is_a_no_op() {
        let mut block = ScalarOpMergingBlock::default();
        block.set_enables_merging();
        assert!(block.unroll_candidates.is_empty());

        block.add_unroll_candidate(candidate(Val(1), InstructionCount(2)));
        block.add_unroll_candidate(candidate(Val(2), InstructionCount(3)));
        block.set_enables_merging();
        assert!(!block.unroll_candidates[0].enables_merging);
        assert!(block.unroll_candidates[1].enables_merging);
    }

    /// ⭐ THE COST IS CHARGED AS THE CANDIDATE IS ADMITTED, so the block's requirement is the sum.
    #[test]
    fn adding_a_candidate_charges_its_cost_to_the_required_ibuff() {
        let mut block = ScalarOpMergingBlock::default();
        block.add_unroll_candidate(candidate(Val(1), InstructionCount(4)));
        block.add_unroll_candidate(candidate(Val(2), InstructionCount(6)));
        assert_eq!(block.required_ibuff, InstructionCount(10));
        assert_eq!(block.unroll_candidates.len(), 2);
    }
}

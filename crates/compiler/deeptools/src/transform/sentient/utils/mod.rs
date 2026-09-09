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

//! `Utils.cpp` — 19 of the campaign's 656 units (dependency level(s) [0, 1, 3]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e239_getOperationOfBlock` | 239 | 0 | 8 | `dcc/src/Transform/Sentient/Utils.cpp:31` |
//! | `e240_selectIndicesForUnits` | 240 | 0 | 15 | `dcc/src/Transform/Sentient/Utils.cpp:66` |
//! | `e241_moveToCommonDominator` | 241 | 0 | 71 | `dcc/src/Transform/Sentient/Utils.cpp:84` |
//! | `e242_getForLoopInfoIfIV` | 242 | 0 | 37 | `dcc/src/Transform/Sentient/Utils.cpp:156` |
//! | `e243_roundDownUnrollFactor` | 243 | 0 | 32 | `dcc/src/Transform/Sentient/Utils.cpp:397` |
//! | `e244_negatePredicate` | 244 | 0 | 18 | `dcc/src/Transform/Sentient/Utils.cpp:431` |
//! | `e245_reversePredicate` | 245 | 0 | 18 | `dcc/src/Transform/Sentient/Utils.cpp:450` |
//! | `e246_getOutermostConstInitialization` | 246 | 0 | 55 | `dcc/src/Transform/Sentient/Utils.cpp:469` |
//! | `e247_memoryOpRequiresImmutAddrScalarCopy` | 247 | 0 | 64 | `dcc/src/Transform/Sentient/Utils.cpp:526` |
//! | `e248_getFoldModeAttributeIfExists` | 248 | 0 | 6 | `dcc/src/Transform/Sentient/Utils.cpp:593` |
//! | `e250_hasUniformizeRegion` | 250 | 0 | 12 | `dcc/src/Transform/Sentient/Utils.cpp:622` |
//! | `e251_add` | 251 | 0 | 4 | `dcc/src/Transform/Sentient/Utils.hpp:165` |
//! | `e252_size` | 252 | 0 | 4 | `dcc/src/Transform/Sentient/Utils.hpp:171` |
//! | `e253_areAllValuesEqual` | 253 | 0 | 6 | `dcc/src/Transform/Sentient/Utils.hpp:180` |
//! | `e392_getQueryKeyAndUnitsFromParentRegion` | 392 | 1 | 24 | `dcc/src/Transform/Sentient/Utils.cpp:40` |
//! | `e393_createSentientForOpWithAdditionalIterArgs` | 393 | 1 | 59 | `dcc/src/Transform/Sentient/Utils.cpp:195` |
//! | `e395_pruneOutOfScopeEntries` | 395 | 1 | 55 | `dcc/src/Transform/Sentient/Utils.cpp:635` |
//! | `e396_replaceValue` | 396 | 1 | 4 | `dcc/src/Transform/Sentient/Utils.hpp:175` |
//! | `e546_findAndReplaceRedundantIterArgsUsedInConditions` | 546 | 3 | 138 | `dcc/src/Transform/Sentient/Utils.cpp:257` |

#![allow(dead_code)]
// ⛔ NOTHING CALLS THIS FILE YET — `Utils.cpp` is the campaign's shared leaf library, and its
// consumers (`e392`, `e393`, `e395`, `e546` here, plus the passes) are not in this batch. CI runs
// clippy with `-D warnings`, so without this the first ported leaf fails the gate.
// ⭐ REMOVE THIS WITH THE FIRST CONSUMER: at that point an unused item here is a real defect again.

use super::register_packing::constant_target_values;
use super::scalar_op_merging_and_hoisting::{ScalarOpComp, compute_address_scale};
use crate::arch::{Arch, Elements};
use crate::formats::Bits;
use sys_arch_spec::fields::{self, ImmSpec, ImmWidth, Sign};
use crate::islands::sentient::dialects::sentient as ops;
use crate::islands::sentient::dialects::{Definitions, Op, Val, uniform};

pub(crate) mod units_and_their_values;


// crustify:todo: e239_getOperationOfBlock
//   authority : dcc/src/Transform/Sentient/Utils.cpp:31  (8 body lines, level 0)
//   original  : Operation &getOperationOfBlock(Block &block, int op_num)

// crustify:todo: e240_selectIndicesForUnits
//   authority : dcc/src/Transform/Sentient/Utils.cpp:66  (15 body lines, level 0)
//   original  : void selectIndicesForUnits( SmallVector<Value> &units, SmallVector<unsigned> &indices, std::unordered_map<std::string, unsigned> unit_name_to_index_map)

// crustify:todo: e241_moveToCommonDominator
//   authority : dcc/src/Transform/Sentient/Utils.cpp:84  (71 body lines, level 0)
//   original  : mlir::LogicalResult moveToCommonDominator(Operation *op, Operation *new_use)

// crustify:todo: e242_getForLoopInfoIfIV
//   authority : dcc/src/Transform/Sentient/Utils.cpp:156  (37 body lines, level 0)
//   original  : std::tuple<Operation *, int64_t, int64_t, int64_t, int64_t> getForLoopInfoIfIV( Value val, bool allow_normalized_iv)

// crustify:todo: e243_roundDownUnrollFactor
//   authority : dcc/src/Transform/Sentient/Utils.cpp:397  (32 body lines, level 0)
//   original  : unsigned roundDownUnrollFactor(unsigned n, Operation *compute_op, SenTargets sen_target)

// crustify:todo: e244_negatePredicate
//   authority : dcc/src/Transform/Sentient/Utils.cpp:431  (18 body lines, level 0)
//   original  : CmpIPredicate negatePredicate(CmpIPredicate pred)

// crustify:todo: e245_reversePredicate
//   authority : dcc/src/Transform/Sentient/Utils.cpp:450  (18 body lines, level 0)
//   original  : CmpIPredicate reversePredicate(CmpIPredicate pred)

// crustify:todo: e246_getOutermostConstInitialization
//   authority : dcc/src/Transform/Sentient/Utils.cpp:469  (55 body lines, level 0)
//   original  : std::tuple<mlir::sentient::ForOp, int, int> getOutermostConstInitialization( BlockArgument iter_arg)

// ───────────────────────────────────────────────────────────────────────────────────────────────
// The types e247 is stated in — its two `llvm_unreachable`s and its one `DT_CHECK`.
// ───────────────────────────────────────────────────────────────────────────────────────────────

/// THE SIX MEMORY UNITS `memoryOpRequiresImmutAddrScalarCopy` DECIDES FOR — the units its own arms
/// name, so the trailing `llvm_unreachable("unexpected operation or unit type")` (`Utils.cpp:589`) is
/// only ever about a MISMATCHED op/unit pair and never about an unknown unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryUnit {
    /// `L0LU` — the L0 load unit.
    L0lu,
    /// `L0SU` — the L0 store unit.
    L0su,
    /// `LXLU` — the LX load unit.
    Lxlu,
    /// `LXSU` — the LX store unit.
    Lxsu,
    /// `L3LU` — the L3 load half.
    L3lu,
    /// `L3SU` — the L3 store half.
    L3su,
}

impl MemoryUnit {
    /// `DT_CHECK(!is_any_of(unit, L3LU, L3SU))` (`Utils/DccExtContext.cpp:36`) AS A CONVERSION —
    /// `None` for the two L3 units, which have no IMM for these operations (`Utils.cpp:557-558`), and
    /// the four that remain are exactly [`ScalarOpComp`]'s.
    #[must_use]
    pub(crate) const fn with_imm(self) -> Option<ScalarOpComp> {
        match self {
            MemoryUnit::L0lu => Some(ScalarOpComp::L0lu),
            MemoryUnit::L0su => Some(ScalarOpComp::L0su),
            MemoryUnit::Lxlu => Some(ScalarOpComp::Lxlu),
            MemoryUnit::Lxsu => Some(ScalarOpComp::Lxsu),
            MemoryUnit::L3lu | MemoryUnit::L3su => None,
        }
    }
}

/// `do_range_check` (`Utils.hpp:154`, defaulted `true`) — whether the constant is also checked against
/// the unit's immediate width, or only tested for being constant at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeCheck {
    /// The default: check the immediate fits.
    Check,
    /// Skip the width check — the caller is asking only whether the address is constant.
    Skip,
}

/// WHICH OF THE FOUR OPS THIS IS, AND THE ATTRIBUTES THAT OP CONTRIBUTES — the `dyn_cast` chain
/// (`Utils.cpp:531-553`) with its `llvm_unreachable("unexpected op type")` as a type.
///
/// ⛔ EACH ARM CARRIES ONLY WHAT ITS ARM READS. The reference leaves `burst_size`/`il` at their
/// initialised 0 for the two scalar/compute ops and reads `getChunkStride()` on the load alone, so a
/// chunk stride is not a fact the other three have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImmutAddrMemoryOp {
    /// `sentient.receive_and_store`.
    ReceiveAndStore {
        /// `getBurstSize()`.
        burst_size: Elements,
        /// `getInterleavedGroup()`.
        interleaved_group: Elements,
    },
    /// `sentient.load_and_send`.
    LoadAndSend {
        /// `getBurstSize()`.
        burst_size: Elements,
        /// `getInterleavedGroup()`.
        interleaved_group: Elements,
        /// `getChunkStride()` — read by no other arm.
        chunk_stride: Elements,
    },
    /// `sentient.load_and_extract_scalar`.
    LoadAndExtractScalar,
    /// `sentient.load_compute_and_send`.
    LoadComputeAndSend,
}

/// ONE MEMORY OP AS e247 READS IT — the kind, the immutable address and the element size the address
/// is computed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImmutAddrMemoryOpInfo {
    /// Which op, and its own attributes.
    pub kind: ImmutAddrMemoryOp,
    /// `getImmutableAddr()` — the operand every test below is about.
    pub immutable_addr: Val,
    /// `getElementSize()`, or `getDstElementSize()` for an LCAS (`:550`) — a width in BITS.
    pub element_size: Bits,
}

impl ImmutAddrMemoryOpInfo {
    /// The `dyn_cast` chain (`Utils.cpp:531-553`) — `None` where the reference reaches
    /// `llvm_unreachable("unexpected op type")`.
    #[must_use]
    pub fn of(op: &Op) -> Option<ImmutAddrMemoryOpInfo> {
        match op {
            Op::Sentient(ops::Op::ReceiveAndStore {
                immutable_addr,
                extent,
                interleaved_group,
                ..
            }) => Some(ImmutAddrMemoryOpInfo {
                kind: ImmutAddrMemoryOp::ReceiveAndStore {
                    burst_size: extent.burst_size,
                    interleaved_group: *interleaved_group,
                },
                immutable_addr: *immutable_addr,
                element_size: extent.element_size,
            }),
            Op::Sentient(ops::Op::LoadAndSend {
                immutable_addr,
                extent,
                interleaved_group,
                ..
            }) => Some(ImmutAddrMemoryOpInfo {
                kind: ImmutAddrMemoryOp::LoadAndSend {
                    burst_size: extent.burst_size,
                    interleaved_group: *interleaved_group,
                    chunk_stride: extent.chunk_stride,
                },
                immutable_addr: *immutable_addr,
                element_size: extent.element_size,
            }),
            Op::Sentient(ops::Op::LoadAndExtractScalar {
                immutable_addr,
                element_size,
                ..
            }) => Some(ImmutAddrMemoryOpInfo {
                kind: ImmutAddrMemoryOp::LoadAndExtractScalar,
                immutable_addr: *immutable_addr,
                element_size: *element_size,
            }),
            Op::Sentient(ops::Op::LoadComputeAndSend {
                immutable_addr,
                dst_element_size,
                ..
            }) => Some(ImmutAddrMemoryOpInfo {
                kind: ImmutAddrMemoryOp::LoadComputeAndSend,
                immutable_addr: *immutable_addr,
                element_size: *dst_element_size,
            }),
            _ => None,
        }
    }
}

/// `isConstant<mlir::sentient::ConstantOp>` (`Utils/Utils.cpp:424-447`) AND THE CONSTANTS IT FOUND —
/// so `is_imm_size_valid`'s opening `DT_CHECK_MSG(isConstant(imm), …)` (`DccExtContext.cpp:34`) is
/// this value's existence rather than a check the range test repeats.
///
/// ⛔ NOT THE NARROWER PRIVATE COPIES: `rematerialization_pass/mod.rs:173` and
/// `scalar_copy_insertion_for_symbols/mod.rs:356` each hold one that omits the query-map arm, and
/// they should collapse into this when their units are reviewed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstantImm {
    /// A `sentient.scalar_constant` — one value for every unit.
    Scalar(i64),
    /// A `uniform.query_map` whose mapping targets are all constants — one value per unit.
    PerUnit(Vec<i64>),
}

/// `isConstant<mlir::sentient::ConstantOp>(val)` (`Utils/Utils.cpp:424-447`), keeping the values.
///
/// ⭐ `None` COVERS ALL THREE OF THE REFERENCE'S FALSE PATHS: a block argument (`defs.of` answers
/// `None`), an op that is neither a constant nor a query map, and a query map with a non-constant
/// target — which is also the empty list `getConstantTargetValues` returns for the last of those.
#[must_use]
pub fn constant_imm(val: Val, defs: Definitions<'_>) -> Option<ConstantImm> {
    match defs.of(val) {
        Some(Op::Sentient(ops::Op::ScalarConstant { value, .. })) => {
            Some(ConstantImm::Scalar(*value))
        }
        Some(Op::Uniform(uniform::Op::QueryMap { map, .. })) => {
            let values = constant_target_values(*map, defs);
            if values.is_empty() {
                None
            } else {
                Some(ConstantImm::PerUnit(values))
            }
        }
        _ => None,
    }
}

/// `Isa::typeToImmInfo.at(isa.getOpcodeType(OpCodeT::LDSTIU))` for one unit
/// (`Utils/DccExtContext.cpp:40-46`).
///
/// ⭐ `DT_CHECK(sizeSignMap.size() == 1)` (`:43`) FAILS THE BUILD HERE, not the run: this is a
/// `const fn` and the four call sites below are `const` items, so a table with two immediate fields on
/// the LDSTIU type is a compile error.
const fn ldstiu_imm_info(comp: fields::Comp) -> (ImmWidth, Sign) {
    let opcodes = comp.opcodes();
    let mut i = 0;
    let mut ldstiu = None;
    while i < opcodes.len() {
        if str_eq(opcodes[i].op, "LDSTIU") {
            ldstiu = Some(opcodes[i].ty);
        }
        i += 1;
    }
    let ty = match ldstiu {
        Some(ty) => ty.get(),
        None => panic!("every memory unit defines LDSTIU (`isa.cpp:1132`, `:1198`, `:1267`, `:1359`)"),
    };

    let all = comp.fields();
    let mut j = 0;
    let mut info = None;
    let mut found = 0;
    while j < all.len() {
        if all[j].ty.get() == ty
            && let ImmSpec::Imm { bits, sign } = all[j].imm
        {
            info = Some((bits, sign));
            found += 1;
        }
        j += 1;
    }
    match (info, found) {
        (Some(info), 1) => info,
        _ => panic!(
            "the LDSTIU type must have exactly one immediate field — \
             DT_CHECK(sizeSignMap.size() == 1) (`Utils/DccExtContext.cpp:43`)"
        ),
    }
}

/// `str::eq` is not `const`, and [`ldstiu_imm_info`] needs the opcode spelling compared at build time.
const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

const LDSTIU_IMM_L0LU: (ImmWidth, Sign) = ldstiu_imm_info(fields::Comp::L0lu);
const LDSTIU_IMM_L0SU: (ImmWidth, Sign) = ldstiu_imm_info(fields::Comp::L0su);
const LDSTIU_IMM_LXLU: (ImmWidth, Sign) = ldstiu_imm_info(fields::Comp::Lxlu);
const LDSTIU_IMM_LXSU: (ImmWidth, Sign) = ldstiu_imm_info(fields::Comp::Lxsu);

/// `DccExtContext::is_imm_size_valid` (`Utils/DccExtContext.cpp:32-75`) — whether the constant
/// immutable address still fits the unit's LDSTIU immediate field once scaled.
///
/// ⛔ NOT AN ANCHORED UNIT: `dcc/src/Utils/` is outside this campaign's file list, and this is the
/// same in-scope-because-it-is-needed case as
/// [`machine_max_reg_num`](super::local_region_splitting_for_value_commoning).
///
/// TRAP: the reference computes `(imm * element_size) / 8 / scale` in a 32-bit `int` and compares
/// against `pow(2, immSize)` in `double`; the width is at most 14 bits, so `i64` throughout is the
/// same answer without the overflow.
#[must_use]
pub(crate) fn imm_size_valid<A: Arch>(
    unit: ScalarOpComp,
    imm: &ConstantImm,
    element_size: Bits,
) -> bool {
    let (bits, sign) = match unit {
        ScalarOpComp::L0lu => LDSTIU_IMM_L0LU,
        ScalarOpComp::L0su => LDSTIU_IMM_L0SU,
        ScalarOpComp::Lxlu => LDSTIU_IMM_LXLU,
        ScalarOpComp::Lxsu => LDSTIU_IMM_LXSU,
    };
    let scale = i64::from(compute_address_scale::<A>(unit).get());
    let width = i64::from(bits.get());
    let valid = |value: i64| -> bool {
        let val = (value * i64::from(element_size.0)) / 8 / scale;
        match sign {
            Sign::Unsigned => val < (1 << width) && val >= 0,
            Sign::Signed => val < (1 << (width - 1)) && val >= -(1 << (width - 1)),
            Sign::ModuloUnsigned => val <= (1 << width) && val >= -(1 << width),
        }
    };
    match imm {
        ConstantImm::Scalar(value) => valid(*value),
        // ⭐ EVERY UNIT'S TARGET MUST FIT, and `llvm::all_of` over the empty list is true — which is
        // the symbol case the reference notes performs no range check (`:65-67`).
        ConstantImm::PerUnit(values) => values.iter().all(|value| valid(*value)),
    }
}

/// Replaces: e247_memoryOpRequiresImmutAddrScalarCopy
///
/// Whether a memory op's constant immutable address must be copied into a scalar register rather than
/// ridden as an LDSTIU immediate (`Utils.cpp:526-590`).
///
/// TRAP: the LCAS arm can only ever return `false` — `DT_CHECK_MSG(is_constant && is_in_range, …)`
/// (`:582`) throws whenever `res` would be true, and `DT_CHECK_MSG` is not debug-gated
/// (`util/dt_exception.hpp:107-118`). TRAP: `load_and_send`'s `chunk_stride` DEFAULTS TO **1**
/// (`SentientOps.td:517`), so an L0LU load's `chunk_stride_present` is normally TRUE.
#[must_use]
pub fn memory_op_requires_immut_addr_scalar_copy<A: Arch>(
    unit_type: MemoryUnit,
    op: &ImmutAddrMemoryOpInfo,
    do_range_check: RangeCheck,
    defs: Definitions<'_>,
) -> bool {
    // Every arm below is `is_constant && …`, and the L3 arm returns `is_constant` itself.
    let Some(imm) = constant_imm(op.immutable_addr, defs) else {
        return false;
    };
    let is_in_range = match (do_range_check, unit_type.with_imm()) {
        (RangeCheck::Check, Some(unit)) => imm_size_valid::<A>(unit, &imm, op.element_size),
        // `do_range_check` off, or an L3 unit with no IMM for these operations (`:557-558`).
        (RangeCheck::Check | RangeCheck::Skip, _) => true,
    };

    // ⛔ THE ORDER IS THE `else if` CHAIN'S: the two scalar/compute ops answer before the L3 arm, so an
    // LAE or LCAS on an L3 unit reaches its own arm with `is_in_range` forced true (`:576-586`).
    match (op.kind, unit_type) {
        (
            ImmutAddrMemoryOp::ReceiveAndStore {
                burst_size,
                interleaved_group,
            },
            MemoryUnit::L0su | MemoryUnit::Lxsu,
        ) => burst_size > Elements(1) || interleaved_group > Elements(0) || !is_in_range,
        (
            ImmutAddrMemoryOp::LoadAndSend {
                burst_size,
                interleaved_group,
                chunk_stride,
            },
            MemoryUnit::L0lu | MemoryUnit::Lxlu,
        ) => {
            // `chunk_stride_present` is an L0LU-only term — LX uses no LRF for the offset (`:543`).
            let chunk_stride_present =
                unit_type == MemoryUnit::L0lu && chunk_stride > Elements(0);
            burst_size > Elements(1)
                || interleaved_group > Elements(0)
                || chunk_stride_present
                || !is_in_range
        }
        (ImmutAddrMemoryOp::LoadAndExtractScalar, _) => !is_in_range,
        (ImmutAddrMemoryOp::LoadComputeAndSend, MemoryUnit::Lxlu) => {
            if is_in_range {
                false
            } else {
                panic!(
                    "sentient::LoadComputeAndSendOp requires immediate immutable address in range \
                     (`Utils.cpp:582`)"
                )
            }
        }
        (ImmutAddrMemoryOp::LoadComputeAndSend, _) => {
            panic!("DT_CHECK(unit_type == LXLU) (`Utils.cpp:580`): {unit_type:?}")
        }
        // No imm in L3LU/L3SU, so being constant at all is the whole answer (`:586-588`).
        (
            ImmutAddrMemoryOp::ReceiveAndStore { .. } | ImmutAddrMemoryOp::LoadAndSend { .. },
            MemoryUnit::L3lu | MemoryUnit::L3su,
        ) => true,
        (ImmutAddrMemoryOp::ReceiveAndStore { .. } | ImmutAddrMemoryOp::LoadAndSend { .. }, _) => {
            panic!(
                "llvm_unreachable(\"unexpected operation or unit type\") (`Utils.cpp:589`): \
                 {:?} on {unit_type:?}",
                op.kind
            )
        }
    }
}

/// Replaces: e248_getFoldModeAttributeIfExists
///
/// An op's `fold_mode` attribute, `None` when it carries none (`Utils.cpp:593-598`).
///
/// ⭐ THE `if (!op)` NULL CHECK IS THE `&Op` PARAMETER, and `hasAttr("fold_mode")` is WHICH VARIANT:
/// the four compute ops are the only ones the `.td` gives the attribute to, and on those it is already
/// an `Option` because it is a `DefaultValuedAttr`.
#[must_use]
pub fn fold_mode_attribute_if_exists(op: &Op) -> Option<ops::FoldMode> {
    match op {
        Op::Sentient(
            ops::Op::VectorMac { fold_mode, .. }
            | ops::Op::VectorBinary { fold_mode, .. }
            | ops::Op::VectorUnary { fold_mode, .. }
            | ops::Op::VectorTernary { fold_mode, .. },
        ) => *fold_mode,
        _ => None,
    }
}

/// Replaces: e250_hasUniformizeRegion
///
/// Whether a program unit holds a `uniform.uniformize_regions` or a `uniform.equalize_pattern`
/// anywhere inside it (`Utils.cpp:622-632`).
///
/// ⭐ ONE FUNCTION, TWO DECLARATIONS: this body and `OldRegisterInitialization.cpp:536-547` are
/// byte-identical, and that one is already ported as `e107_hasUniformizeRegion` — so this FORWARDS to
/// it rather than walking the unit a second time.
#[must_use]
pub fn has_uniformize_region(unit_body: &[Op]) -> bool {
    super::old_register_initialization::register_init_info::has_uniformize_region(unit_body)
}

// e251_add, e252_size and e253_areAllValuesEqual are ported ONE MODULE DOWN, on the
// `UnitsAndTheirValues` that already carries e249_normalizeNullValues:
// `units_and_their_values.rs`. The scheduler split one C++ class (`Utils.hpp:161-190`) across two
// homes; a second Rust type with the same fields would be the split made real.

// crustify:todo: e392_getQueryKeyAndUnitsFromParentRegion
//   authority : dcc/src/Transform/Sentient/Utils.cpp:40  (24 body lines, level 1)
//   original  : mlir::LogicalResult getQueryKeyAndUnitsFromParentRegion( Operation *op, Value &key, SmallVector<Value> &units)
//   calls     : e252_size

// crustify:todo: e393_createSentientForOpWithAdditionalIterArgs
//   authority : dcc/src/Transform/Sentient/Utils.cpp:195  (59 body lines, level 1)
//   original  : Operation *createSentientForOpWithAdditionalIterArgs( Operation *loop_op, std::vector<std::pair<Value, Value>> &start_vals_and_steps)
//   calls     : e252_size

// crustify:todo: e395_pruneOutOfScopeEntries
//   authority : dcc/src/Transform/Sentient/Utils.cpp:635  (55 body lines, level 1)
//   original  : void pruneOutOfScopeEntries(mlir::uniform::DefImmutableMappingOp map_op, SmallVectorImpl<Operation *> &to_be_deleted)
//   calls     : e252_size

// crustify:todo: e396_replaceValue
//   authority : dcc/src/Transform/Sentient/Utils.hpp:175  (4 body lines, level 1)
//   original  : void replaceValue(size_t index, mlir::Value new_val)
//   calls     : e252_size

// crustify:todo: e546_findAndReplaceRedundantIterArgsUsedInConditions
//   authority : dcc/src/Transform/Sentient/Utils.cpp:257  (138 body lines, level 3)
//   original  : LogicalResult findAndReplaceRedundantIterArgsUsedInConditions( mlir::sentient::ForOp loop, std::set<int> &iter_arg_indices_to_delete)
//   calls     : e422_insert

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::islands::dataflow_ir::ty::ScalarTy;

    /// `sentient.scalar_constant` — the constant an immutable address resolves to.
    fn scalar_constant(result: Val, value: i64) -> Op {
        Op::Sentient(ops::Op::ScalarConstant {
            value,
            result,
            reg_locale: ops::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// `sentient.load_and_send` reading `immutable_addr`, with the `.td`'s own extent defaults.
    fn load_and_send(immutable_addr: Val, chunk_stride: Elements) -> Op {
        Op::Sentient(ops::Op::LoadAndSend {
            mutable_addr: Val(1),
            immutable_addr,
            increment: Val(3),
            consumer: SendEnd::to_self(Val(4)),
            result: Val(5),
            extent: ops::Extent {
                chunk_stride,
                ..ops::Extent::of(Elements(64), Bits(32))
            },
            interleaved_group: Elements(0),
            rotate_val: None,
            dir: None,
            shuffle_mode: ops::ShuffleMode::NoShuffle,
            reg: ops::Reg {
                locale: ops::RegType::Lar,
                index: None,
            },
            dbg_name: None,
        })
    }

    /// `sentient.vector_unary`, carrying `fold_mode` or not.
    fn vector_unary(fold_mode: Option<ops::FoldMode>) -> Op {
        Op::Sentient(ops::Op::VectorUnary {
            mask: Val(2),
            op_a: ops::Operand::from(ops::Port::Zero),
            unary_op: ops::UnaryOp::Floor,
            result: ops::ResultPorts::default(),
            compute_precision: ops::Precision::Fp16,
            fold_mode,
            unroll_factor: ops::UnrollFactor::X1,
            dbg_name: None,
        })
    }

    /// AN L0LU LOAD NEEDS THE COPY FOR ITS CHUNK STRIDE ALONE, the same load on LXLU does not, and a
    /// non-constant address needs none — the three ways the `is_constant && (…)` arm resolves.
    #[test]
    fn e247_memory_op_requires_immut_addr_scalar_copy() {
        let scope = vec![
            scalar_constant(Val(2), 0),
            load_and_send(Val(2), Elements(1)),
        ];
        let regions: [&[Op]; 1] = [&scope];
        let defs = Definitions::from_innermost(&regions);
        let op = ImmutAddrMemoryOpInfo::of(&scope[1]).expect("a load_and_send is one of the four");

        assert!(memory_op_requires_immut_addr_scalar_copy::<Dd2>(
            MemoryUnit::L0lu,
            &op,
            RangeCheck::Check,
            defs
        ));
        assert!(!memory_op_requires_immut_addr_scalar_copy::<Dd2>(
            MemoryUnit::Lxlu,
            &op,
            RangeCheck::Check,
            defs
        ));

        // A block argument is not a constant, so no arm can ask for the copy.
        let unbound = ImmutAddrMemoryOpInfo {
            immutable_addr: Val(99),
            ..op
        };
        assert!(!memory_op_requires_immut_addr_scalar_copy::<Dd2>(
            MemoryUnit::L0lu,
            &unbound,
            RangeCheck::Check,
            defs
        ));
    }

    /// AN ADDRESS THAT DOES NOT FIT THE UNIT'S IMMEDIATE FORCES THE COPY on the unit whose field is
    /// narrower, and the wider one takes the same address as an immediate.
    #[test]
    fn e247_an_out_of_range_constant_needs_the_copy() {
        let scope = vec![
            scalar_constant(Val(2), 8_000),
            load_and_send(Val(2), Elements(0)),
        ];
        let regions: [&[Op]; 1] = [&scope];
        let defs = Definitions::from_innermost(&regions);
        let op = ImmutAddrMemoryOpInfo::of(&scope[1]).expect("a load_and_send is one of the four");

        // 8000 * 32 bits / 8 = 32,000, past LXLU's 14-bit signed field but not past `Skip`.
        assert!(memory_op_requires_immut_addr_scalar_copy::<Dd2>(
            MemoryUnit::Lxlu,
            &op,
            RangeCheck::Check,
            defs
        ));
        assert!(!memory_op_requires_immut_addr_scalar_copy::<Dd2>(
            MemoryUnit::Lxlu,
            &op,
            RangeCheck::Skip,
            defs
        ));
    }

    /// THE ATTRIBUTE IS RETURNED WHERE IT EXISTS, and an op the `.td` gives no `fold_mode` answers
    /// `None` rather than `FoldMode::None`.
    #[test]
    fn e248_get_fold_mode_attribute_if_exists() {
        assert_eq!(
            fold_mode_attribute_if_exists(&vector_unary(Some(ops::FoldMode::FoldAbBoth))),
            Some(ops::FoldMode::FoldAbBoth)
        );
        assert_eq!(fold_mode_attribute_if_exists(&vector_unary(None)), None);
        assert_eq!(
            fold_mode_attribute_if_exists(&scalar_constant(Val(2), 0)),
            None
        );
    }

    /// THE RE-EXPORT ANSWERS FOR BOTH MEMBERS OF THE `isa<>`, and `false` for a unit holding neither.
    #[test]
    fn e250_has_uniformize_region() {
        assert!(has_uniformize_region(&[Op::Uniform(
            uniform::Op::UniformizeRegions {
                regions: Vec::new(),
                results: Vec::new(),
            }
        )]));
        assert!(!has_uniformize_region(&[scalar_constant(Val(2), 0)]));
    }
}

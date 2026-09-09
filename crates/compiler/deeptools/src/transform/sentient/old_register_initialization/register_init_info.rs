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

//! `OldRegisterInitialization.cpp` — 12 of the campaign's 656 units (dependency level(s) [0, 1, 2]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e103_totalTripCount` | 103 | 0 | 15 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:198` |
//! | `e104_getRegisterInitPriority` | 104 | 0 | 36 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:214` |
//! | `e105_weightOfSubset` | 105 | 0 | 9 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:322` |
//! | `e106_collectAllRegCoalescingCandidatesImpl` | 106 | 0 | 18 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:340` |
//! | `e107_hasUniformizeRegion` | 107 | 0 | 11 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:536` |
//! | `e327_calcSSAWeight` | 327 | 1 | 59 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:262` |
//! | `e328_sortRegCoalescingCandidates` | 328 | 1 | 7 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:332` |
//! | `e329_collectAllRegCoalescingCandidates` | 329 | 1 | 133 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:371` |
//! | `e330_getNextMaxWeight` | 330 | 1 | 29 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:505` |
//! | `e449_calcSSAWeight` | 449 | 2 | 9 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:252` |
//! | `e450_collectAllRegCoalescingCandidates` | 450 | 2 | 9 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:361` |
//! | `e451_collectRegInitAndRegCoalescingCandidateFast` | 451 | 2 | 114 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:548` |

use std::collections::BTreeMap;

use crate::islands::sentient::dialects::{
    Definitions, Op, Val, arith, dataflow, lowered, sentient, symbol, uniform,
    uniform_mapping_values, use_count, value_reg_locale,
};
use crate::transform::sentient::analyses::Liveness;

/// WHETHER A VALUE'S REGISTER FILE MAY, MUST OR MAY NOT BE INITIALISED IN THE PROGRAM HEADER —
/// `RegisterInitInfo::RegInitLocalePriority` (`OldRegisterInitialization.cpp:100-105`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum RegInitLocalePriority {
    /// `kUnknown` — the enum's zero. ⛔ NEVER RETURNED by [`get_register_init_priority`], which ends
    /// in `kNotAllowed`; it is the value a default-constructed field holds.
    #[default]
    Unknown,
    /// `kNotAllowed`.
    NotAllowed,
    /// `kAllowed`.
    Allowed,
    /// `kRequired`.
    Required,
}

/// `allowed_reg_init_locales_` (`OldRegisterInitialization.cpp:110-116`) — `MaxAllowedLocales = 4` (`:108`).
const ALLOWED_REG_INIT_LOCALES: [sentient::RegType; 4] = [
    sentient::RegType::Lrf,
    sentient::RegType::Lar,
    sentient::RegType::Ear,
    sentient::RegType::Gtr,
];

/// `required_reg_init_locales_` (`OldRegisterInitialization.cpp:117-119`) — `MaxRequiredLocales = 3` (`:109`).
const REQUIRED_REG_INIT_LOCALES: [sentient::RegType; 3] = [
    sentient::RegType::Ebr,
    sentient::RegType::Lbr,
    sentient::RegType::Mvr,
];

/// `-dcc-old-register-initialization-gtr`, `cl::init(true)` (`OldRegisterInitialization.cpp:56-59`).
///
/// ⭐ AN ENUM AND NOT A `bool`, and [`Default`] IS THE REFERENCE'S `cl::init`: promoting a `gtr`
/// needs inter-unit analysis, so the translator sets `init_packet_opt_en` to say when it is safe and
/// this switch says whether to look.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GtrRegInit {
    /// The flag's default.
    #[default]
    Enabled,
    /// `-dcc-old-register-initialization-gtr=false`.
    Disabled,
}

/// ONE VALUE'S WEIGHT — `ssa_weight_`'s value, *"based on number of times it is accessed"*
/// (`OldRegisterInitialization.cpp:123-124`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct SsaWeight(pub i32);

/// `ssa_weight_` — the weight of every value the pass has scored, `DenseMap<mlir::Value, int>`
/// (`OldRegisterInitialization.cpp:125`).
///
/// ⭐ ORDERED, WHERE `DenseMap` IS NOT, for the reason `e102_dumpWeights` already gives: the dump is
/// the map's only observable and a `DenseMap`'s iteration order is unspecified, so a `BTreeMap` is the
/// only version of `e331_dumpWeight`'s output that is the same twice.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SsaWeights(BTreeMap<Val, SsaWeight>);

impl SsaWeights {
    /// `ssa_weight_[val]` READ — ⛔ AND WITHOUT THE INSERTION. See [`weight_of_subset`]'s TRAP.
    #[must_use]
    pub fn weight_of(&self, val: Val) -> SsaWeight {
        self.0.get(&val).copied().unwrap_or_default()
    }

    /// `ssa_weight_[val] = weight`.
    pub fn set(&mut self, val: Val, weight: SsaWeight) {
        self.0.insert(val, weight);
    }

    /// Every scored value and its weight, for `dumpWeights` (`e102`).
    pub fn iter(&self) -> impl Iterator<Item = (Val, SsaWeight)> {
        self.0.iter().map(|(val, weight)| (*val, *weight))
    }
}

/// HOW MANY TIMES AN OP RUNS — the product of the enclosing loops' trip counts, `int`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TripCount(pub i32);

/// THE OPS ENCLOSING ONE OP, INNERMOST FIRST — the chain `Operation::getParentOp()` walks.
///
/// ⛔⛔ THE ISLAND IS A TREE WITH NO PARENT POINTERS, which the campaign brief names as exactly the
/// *mechanism* a port may drop and supply instead — the same reason
/// [`Definitions::from_innermost`][crate::islands::sentient::dialects::Definitions::from_innermost]
/// takes the enclosing regions as a list. The walker that has descended to an op has this chain in
/// hand; nothing can recover it from the op alone.
#[derive(Debug, Clone, Copy)]
pub struct Enclosing<'a>(&'a [&'a Op]);

impl<'a> Enclosing<'a> {
    /// THE ENCLOSING OPS, INNERMOST FIRST.
    #[must_use]
    pub const fn from_innermost(ops: &'a [&'a Op]) -> Enclosing<'a> {
        Enclosing(ops)
    }

    /// The chain, innermost first.
    #[must_use]
    pub const fn ops(&self) -> &'a [&'a Op] {
        self.0
    }
}

/// Replaces: e103_totalTripCount
///
/// How many times an op runs: the product of the constant bounds of every `sentient.for` enclosing
/// it, walking outwards (`OldRegisterInitialization.cpp:198-212`).
///
/// TRAP: a loop whose bound is not a `sentient.scalar_constant` contributes NOTHING — not even a
/// factor of the loop's real trip count — which is the reference's failed `dyn_cast`, and on a bound
/// with no defining op at all the reference `dyn_cast`s a null pointer.
#[must_use]
pub fn total_trip_count(of: Enclosing<'_>, defs: Definitions<'_>) -> TripCount {
    let mut total_trip_count = 1_i32;
    for parent in of.ops() {
        if let Op::Sentient(sentient::Op::For { bound, .. }) = parent
            && let Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) = defs.of(*bound)
        {
            // `total_trip_count *= static_cast<int>(constant_bound.getValue());` — the attribute is
            // `SI64Attr` and the narrowing to `int` is the reference's own cast.
            total_trip_count =
                total_trip_count.wrapping_mul(i32::try_from(*value).unwrap_or(*value as i32));
        }
    }
    TripCount(total_trip_count)
}

/// Replaces: e104_getRegisterInitPriority
///
/// Whether a value's register file may be initialised in the program header: `ebr`/`lbr`/`mvr` must
/// be, `lrf`/`lar`/`ear` may be, and a `gtr` may only when the multicast group behind its
/// `sentient.scalar_copy` is one the translator marked optimisable
/// (`OldRegisterInitialization.cpp:214-249`).
///
/// TRAP: `isOptimizable` is a bare `hasAttr("init_packet_opt_en")`
/// (`dcc/src/Dialect/Dataflow/Utils.cpp:18-19`) — presence IS the value, and `dcc` never sets it.
#[must_use]
pub fn get_register_init_priority(
    val: Val,
    defs: Definitions<'_>,
    gtr_reg_init: GtrRegInit,
) -> RegInitLocalePriority {
    let locale = value_reg_locale(val, defs);
    // Promoting GTRs requires inter-unit analysis, so the translator's flag is what says when it is
    // safe (`:217-218`).
    if locale == sentient::RegType::Gtr {
        if gtr_reg_init == GtrRegInit::Enabled
            && let Some(Op::Sentient(sentient::Op::ScalarCopy { input, .. })) = defs.of(val)
        {
            match defs.of(*input) {
                Some(Op::Dataflow(dataflow::Op::CreateMulticastGroup {
                    init_packet_opt_en,
                    ..
                })) => {
                    return if *init_packet_opt_en {
                        RegInitLocalePriority::Allowed
                    } else {
                        RegInitLocalePriority::NotAllowed
                    };
                }
                // ⭐ EVERY UNIT'S GROUP MUST BE OPTIMISABLE, not just one: a uniformized copy reads
                // one constant per unit through the map, and the first that is not a multicast group
                // or is not optimisable decides the whole answer (`:227-238`).
                Some(Op::Uniform(uniform::Op::QueryMap { map, key, .. })) => {
                    for val_op in uniform_mapping_values(*map, *key, defs) {
                        match defs.of(val_op) {
                            Some(Op::Dataflow(dataflow::Op::CreateMulticastGroup {
                                init_packet_opt_en: true,
                                ..
                            })) => {}
                            _ => return RegInitLocalePriority::NotAllowed,
                        }
                    }
                    return RegInitLocalePriority::Allowed;
                }
                _ => {}
            }
        }
        return RegInitLocalePriority::NotAllowed;
    }
    if REQUIRED_REG_INIT_LOCALES.contains(&locale) {
        return RegInitLocalePriority::Required;
    }
    if ALLOWED_REG_INIT_LOCALES.contains(&locale) {
        return RegInitLocalePriority::Allowed;
    }
    RegInitLocalePriority::NotAllowed
}

/// Replaces: e105_weightOfSubset
///
/// A coalescing subset's weight: the sum of its members' weights HALVED, because a subset holds both
/// ends of every carried edge and so counts each weight twice (`OldRegisterInitialization.cpp:322-330`).
///
/// TRAP: `ssa_weight_[val]` is `DenseMap::operator[]`, which INSERTS a zero for a value the map does
/// not hold. This read does not, so a subset naming an unscored value leaves `ssa_weight_` one entry
/// shorter than the reference's — observable only through `dumpWeights` (`e102`).
#[must_use]
pub fn weight_of_subset(subset: &[Val], ssa_weight: &SsaWeights) -> SsaWeight {
    let mut subset_weight = 0_i32;
    for val in subset {
        subset_weight = subset_weight.wrapping_add(ssa_weight.weight_of(*val).0);
    }
    // `subset_weight = subset_weight / 2;` — C++ integer division truncates toward zero, as Rust's
    // does, so an odd sum loses its half exactly as the reference's does.
    SsaWeight(subset_weight / 2)
}

/// Replaces: e106_collectAllRegCoalescingCandidatesImpl
///
/// Grows `subset` with the chain of singly-used `sentient.for` iter args that carry one value inwards:
/// the arg matching `def`, then the arg the enclosing loop passes IN as its initial value, and so
/// outwards (`OldRegisterInitialization.cpp:340-357`).
///
/// TRAP: the reference tests `iter_arg.hasOneUse() && iter_arg == def`, so a value read twice inside
/// the body ends the chain even though the loop still carries it — `use_count(.., scope) == 1` is
/// exactly `hasOneUse()`, counting one per USE and not per user.
pub fn collect_all_reg_coalescing_candidates_impl(
    subset: &mut Vec<Val>,
    for_carried: &[sentient::Carried],
    def: Val,
    scope: &[Op],
    defs: Definitions<'_>,
) {
    for entry in for_carried {
        let iter_arg = entry.arg;
        if use_count(iter_arg, scope) == 1 && iter_arg == def {
            subset.push(iter_arg);
            // `auto def1 = for_op.getIterOperands()[i];` — the value passed in at this position.
            let def1 = entry.init;
            // `dyn_cast<BlockArgument>(def1)` then `arg0.getOwner()->getParentOp()` cast to a
            // `sentient::ForOp`: `for_arg_of` answers both at once and only ever for a `sentient.for`.
            if let Some((Op::Sentient(sentient::Op::For { carried, .. }), _)) =
                defs.for_arg_of(def1)
            {
                collect_all_reg_coalescing_candidates_impl(subset, carried, def1, scope, defs);
            }
        }
    }
}

/// Replaces: e107_hasUniformizeRegion
///
/// Whether a program unit holds a `uniform.uniformize_regions` or a `uniform.equalize_pattern`
/// anywhere inside it (`OldRegisterInitialization.cpp:536-546`).
///
/// TRAP: the reference's walk is `WalkOrder::PreOrder` with an `interrupt()`, but it only ever reports
/// whether it found ONE — so "any op in the subtree" is the whole observable result and the order is
/// mechanism.
#[must_use]
pub fn has_uniformize_region(unit_body: &[Op]) -> bool {
    unit_body.iter().any(|op| match op {
        Op::Uniform(
            uniform::Op::UniformizeRegions { .. } | uniform::Op::EqualizePattern { .. },
        )
        // ⭐ THE SAME TWO OPS, ONE RUNG UP — `isa<UniformizeRegionsOp, EqualizePatternOp>` cannot
        // tell them apart, so neither may this.
        | Op::UniformRegions(_) => true,
        Op::Sentient(inner) => sentient::regions(inner)
            .iter()
            .any(|region| has_uniformize_region(region)),
        Op::AffineFor(loop_op) => has_uniformize_region(&loop_op.body),
        // ⭐ A LOWER-RUNG REGION IS WALKED, NOT SKIPPED: a `uniform.uniformize_regions` nests inside
        // another one's region (`flatten_local_region.mlir:86-88`, three deep), and those inner ops
        // are values of the DataflowIR island's type.
        Op::Uniform(_)
        | Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_)
        | Op::Affine(_)
        | Op::Vector(_)
        | Op::Arith(_)
        | Op::Scf(_)
        | Op::Symbol(_) => lowered(op).is_some_and(|lower| {
            crate::islands::dataflow_ir::dialects::regions(&lower)
                .iter()
                .any(|region| has_uniform_group_below(region))
        }),
    })
}

/// [`has_uniformize_region`]'s walk once it has crossed into a lower-rung region.
fn has_uniform_group_below(body: &[crate::islands::dataflow_ir::dialects::Op]) -> bool {
    use crate::islands::dataflow_ir::dialects as lower;
    body.iter().any(|op| {
        matches!(
            op,
            lower::Op::Uniform(
                uniform::Op::UniformizeRegions { .. } | uniform::Op::EqualizePattern { .. }
            )
        ) || lower::regions(op)
            .iter()
            .any(|region| has_uniform_group_below(region))
    })
}

/// `dcc::utils::isTargetConstant(val, target)` (`Analyses/Utils.cpp:156-183`) — is this value the
/// literal `target`, on every unit.
///
/// ⛔ REPRODUCED FROM THE AUTHORITY RATHER THAN LEFT A `todo!`, WHICH IS A DELIBERATE DIVERGENCE FROM
/// `lightweight_simplification`'s stub of the same helper. It is a total predicate over the IR and
/// not an analysis result — all three of its arms are expressible here — and a `todo!` would panic
/// six of `e329`'s seven arms and take the whole `enforced_virtual_assign_` effect with it.
/// ⚠️ TWO REFERENCE ABORTS BECOME ANSWERS: `DT_CHECK_MSG(val, ..)` and `DT_CHECK(values.size() > 0)`,
/// the second of which an empty mapping reaches — an empty `all()` is `true`, the answer the loop it
/// guards would have given.
fn is_target_constant(val: Val, target: i64, defs: Definitions<'_>) -> bool {
    /// The two constant ops the reference `dyn_cast`s, and the literal each carries.
    fn literal(op: &Op) -> Option<i64> {
        match op {
            Op::Sentient(sentient::Op::ScalarConstant { value, .. })
            | Op::Arith(arith::Op::Constant { value, .. }) => Some(*value),
            Op::Arith(arith::Op::ConstantInt { value, .. }) => Some(match value {
                arith::IntConst::Bool(bit) => i64::from(*bit),
                arith::IntConst::Int { value, .. } => *value,
            }),
            _ => None,
        }
    }
    match defs.of(val) {
        // `if (mlir::isa<BlockArgument>(val)) return false;` — no defining op IS a block argument.
        None => false,
        // ⭐ EVERY UNIT'S CONSTANT MUST MATCH, and a mapped value that is neither constant op is
        // SKIPPED rather than refused, which is the reference's own fall-through (`:174-181`).
        Some(Op::Uniform(uniform::Op::QueryMap { map, key, .. })) => {
            uniform_mapping_values(*map, *key, defs)
                .into_iter()
                .all(|value| {
                    defs.of(value)
                        .is_none_or(|op| literal(op).is_none_or(|value| value == target))
                })
        }
        Some(op) => literal(op) == Some(target),
    }
}

/// `RegisterInitInfo`'s OWN STATE — the per-program-unit scoreboard the pass builds and then spends
/// (`OldRegisterInitialization.cpp:121-137`).
///
/// ⛔ `liveness_` IS NOT A FIELD. It is a reference to an analysis that is out of campaign scope, so
/// the one unit that consults it takes it as an argument — see
/// [`RegisterInitInfo::collect_all_reg_coalescing_candidates`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RegisterInitInfo {
    /// `ssa_weight_`.
    pub ssa_weight: SsaWeights,
    /// `valid_reginit_candidates_` — a `gtr`/`lrf`/`lar`/`ear` value that MAY move to the header.
    pub valid_reginit_candidates: Vec<Val>,
    /// `required_reginit_candidates_` — an `ebr`/`lbr`/`mvr` value that MUST.
    pub required_reginit_candidates: Vec<Val>,
    /// `valid_reg_coalescing_candidates_` — each entry is one subset that may share a register.
    pub valid_reg_coalescing_candidates: Vec<Vec<Val>>,
    /// `final_reginit_candidates_` — ⛔ A `std::stack`, so the LAST entry is `top()`. `e451` fills it.
    pub final_reginit_candidates: Vec<Val>,
    /// `final_reg_coalescing_candidates_` — the subsets `e451` kept.
    pub final_reg_coalescing_candidates: Vec<Vec<Val>>,
    /// `enforced_virtual_assign_` — ⛔ ORDERED PAIRS: `.0` must take `.1`'s register, and only `.1` is
    /// ever retargeted (see [`RegisterInitInfo::replace_virtual_assign_target`]).
    pub enforced_virtual_assign: Vec<(Val, Val)>,
}

impl RegisterInitInfo {
    /// `calcSSAWeight`'s `addCandidate` lambda (`:263-269`) — files a scored value under its priority.
    fn add_candidate(&mut self, val: Val, defs: Definitions<'_>, gtr_reg_init: GtrRegInit) {
        match get_register_init_priority(val, defs, gtr_reg_init) {
            RegInitLocalePriority::Required => self.required_reginit_candidates.push(val),
            RegInitLocalePriority::Allowed => self.valid_reginit_candidates.push(val),
            RegInitLocalePriority::Unknown | RegInitLocalePriority::NotAllowed => {}
        }
    }

    /// Replaces: e326_replaceVirtualAssignTarget
    ///
    /// Retargets every mention of `old_v` onto `new_v` after the promoter has replaced a value with a
    /// shared copy (`OldRegisterInitialization.cpp:159-166`).
    ///
    /// ⛔ TRAP: ASYMMETRIC. In `enforced_virtual_assign_` only the pair's SECOND member — the register
    /// being taken — is rewritten, so a promoted value that was somebody's `.0` keeps pointing at the
    /// old target; in a coalescing subset EVERY member is.
    pub fn replace_virtual_assign_target(&mut self, old_v: Val, new_v: Val) {
        for virtual_assign in &mut self.enforced_virtual_assign {
            if virtual_assign.1 == old_v {
                virtual_assign.1 = new_v;
            }
        }
        for values in &mut self.final_reg_coalescing_candidates {
            for value in values.iter_mut() {
                if *value == old_v {
                    *value = new_v;
                }
            }
        }
    }

    /// Replaces: e327_calcSSAWeight
    ///
    /// Scores one op's results with the number of times it runs, and files the values that may be
    /// initialised in the program header (`OldRegisterInitialization.cpp:262-320`).
    ///
    /// ⛔ TRAP: ONLY A LOOP-INVARIANT SOURCE IS PROMOTED. A carried value is scored only when its
    /// INITIAL value comes from a `sentient.scalar_constant` or a `uniform.query_map`, and a
    /// `scalar_copy` only when its input is one of four ops — or a multicast group whose producer is
    /// not itself a query map. Everything else is weighed and never offered.
    /// ⛔ TRAP: `isOuterMostLoop` GATES THE LOOP ARM ALONE, so an inner loop's carried constants are
    /// weighed but never become candidates (`Analyses/Utils.cpp:523-534`).
    /// ⚠️ `isa<uniform::QueryMapOp>(mc_op.getProducer().getDefiningOp())` is a BARE `isa<>` on a
    /// possibly-null op; a producer with no defining op is answered "not a query map" here rather than
    /// asserting, the same divergence `e112_isSameOpType` records.
    pub fn calc_ssa_weight(
        &mut self,
        op: &Op,
        enclosing: Enclosing<'_>,
        defs: Definitions<'_>,
        gtr_reg_init: GtrRegInit,
    ) {
        let Op::Sentient(inner) = op else { return };
        // `totalTripCount(op)` — the product of the ENCLOSING loops' bounds, this op's own excluded.
        let weight = SsaWeight(total_trip_count(enclosing, defs).0);
        match inner {
            sentient::Op::For { carried, .. } => {
                let outer_most = !enclosing
                    .ops()
                    .iter()
                    .any(|parent| matches!(parent, Op::Sentient(sentient::Op::For { .. })));
                for entry in carried {
                    if matches!(
                        defs.of(entry.init),
                        Some(
                            Op::Sentient(sentient::Op::ScalarConstant { .. })
                                | Op::Uniform(uniform::Op::QueryMap { .. })
                        )
                    ) {
                        self.ssa_weight.set(entry.arg, weight);
                        if outer_most {
                            self.add_candidate(entry.arg, defs, gtr_reg_init);
                        }
                    }
                }
            }
            sentient::Op::ScalarCopy { input, result, .. } => {
                let promotable = match defs.of(*input) {
                    Some(
                        Op::Sentient(sentient::Op::ScalarConstant { .. })
                        | Op::Uniform(uniform::Op::QueryMap { .. })
                        | Op::Dataflow(dataflow::Op::GetUnit { .. })
                        | Op::Symbol(symbol::Op::CreateSymbol { .. }),
                    ) => true,
                    Some(Op::Dataflow(dataflow::Op::CreateMulticastGroup { producer, .. })) => {
                        !matches!(
                            defs.of(*producer),
                            Some(Op::Uniform(uniform::Op::QueryMap { .. }))
                        )
                    }
                    _ => false,
                };
                if promotable {
                    self.ssa_weight.set(*result, weight);
                    self.add_candidate(*result, defs, gtr_reg_init);
                }
            }
            // WEIGHED AND NOT OFFERED — every value these bind is scored and none is a candidate.
            sentient::Op::LoadAndSend { result, .. }
            | sentient::Op::ReceiveAndStore { result, .. }
            | sentient::Op::ScalarAdd { result, .. }
            | sentient::Op::ScalarSub { result, .. }
            | sentient::Op::ReceiveAndExtractScalar { result, .. }
            | sentient::Op::LoadComputeAndSend { result, .. } => {
                self.ssa_weight.set(*result, weight);
            }
            sentient::Op::LoadAndStore { results, .. } => {
                self.ssa_weight.set(results.0, weight);
                self.ssa_weight.set(results.1, weight);
            }
            sentient::Op::LoadAndExtractScalar {
                addr_result,
                data_result,
                ..
            } => {
                self.ssa_weight.set(*addr_result, weight);
                self.ssa_weight.set(*data_result, weight);
            }
            _ => {}
        }
    }

    /// Replaces: e328_sortRegCoalescingCandidates
    ///
    /// Orders the coalescing subsets by decreasing weight, which is the order `e451` spends them in
    /// (`OldRegisterInitialization.cpp:332-338`).
    ///
    /// ⭐ STABLE, WHERE `std::sort` IS NOT: equal-weight subsets keep their collection order here and
    /// have no defined order there, so this is the version whose pick is the same twice.
    pub fn sort_reg_coalescing_candidates(&mut self) {
        let ssa_weight = &self.ssa_weight;
        self.valid_reg_coalescing_candidates
            .sort_by(|subset1, subset2| {
                weight_of_subset(subset2, ssa_weight).cmp(&weight_of_subset(subset1, ssa_weight))
            });
    }

    /// `{result, mutable_addr}` — the six transfer arms of `e329`, which differ only in which pair of
    /// values they name (`:401-467`).
    fn coalesce_with_mutable_addr<L: Liveness>(
        &mut self,
        result: Val,
        mutable_addr: Val,
        increment: Val,
        defs: Definitions<'_>,
        liveness: &L,
    ) {
        if !liveness.is_live_range_overlaps(result, mutable_addr) {
            self.valid_reg_coalescing_candidates
                .push(vec![result, mutable_addr]);
        }
        // A no-update transfer copies the address straight to the result, so the two MUST share the
        // register rather than merely being allowed to.
        if is_target_constant(increment, 0, defs) {
            self.enforced_virtual_assign.push((result, mutable_addr));
        }
    }

    /// `{result, operand}` — the `scalar_add`/`scalar_sub` arms (`:469-501`).
    ///
    /// ⛔ THE SECOND OPERAND IS ONLY REACHED WHEN THE FIRST IS A CONSTANT OR HAS NO DEFINING OP. The
    /// reference's `else if` tests the OUTER condition, so an accumulator whose first operand is a
    /// register but whose live range overlaps yields NO subset — it does not fall through to the
    /// other operand.
    fn coalesce_with_operand<L: Liveness>(
        &mut self,
        result: Val,
        operands: [Val; 2],
        defs: Definitions<'_>,
        liveness: &L,
    ) {
        for operand in operands {
            if defs.of(operand).is_some_and(|def| {
                !matches!(def, Op::Sentient(sentient::Op::ScalarConstant { .. }))
            }) {
                if !liveness.is_live_range_overlaps(result, operand) {
                    self.valid_reg_coalescing_candidates
                        .push(vec![result, operand]);
                }
                return;
            }
        }
    }

    /// Replaces: e329_collectAllRegCoalescingCandidates
    ///
    /// Collects the subsets of values that may share one register: the chain of singly-used carried
    /// loop args, each transfer's result with the mutable address it advances, and each scalar
    /// add/sub's result with its register operand (`OldRegisterInitialization.cpp:371-503`).
    ///
    /// ⛔ TRAP: A LOOP SUBSET IS PUSHED ONLY WHEN THE CARRIED INITIAL VALUE IS A BLOCK ARGUMENT. The
    /// `push_back` sits INSIDE the `dyn_cast<BlockArgument>(def)` (`:380-386`), so a loop carrying a
    /// constant in from outside collects a subset and then DROPS it.
    /// ⛔ TRAP: THE TRANSFER ARMS PRODUCE TWO DIFFERENT EFFECTS from two different tests — a
    /// non-overlapping live range makes a subset that `e451` may reject, while a zero increment
    /// ENFORCES the pairing outright. Neither implies the other, and a transfer can produce both.
    pub fn collect_all_reg_coalescing_candidates<L: Liveness>(
        &mut self,
        op: &Op,
        scope: &[Op],
        defs: Definitions<'_>,
        liveness: &L,
    ) {
        let Op::Sentient(inner) = op else { return };
        match inner {
            sentient::Op::For { carried, .. } => {
                for entry in carried {
                    if use_count(entry.arg, scope) != 1 {
                        continue;
                    }
                    let mut subset = vec![entry.arg];
                    let def = entry.init;
                    // `dyn_cast<BlockArgument>(def)`: no defining op IS a block argument.
                    if defs.of(def).is_none() {
                        if let Some((Op::Sentient(sentient::Op::For { carried: outer, .. }), _)) =
                            defs.for_arg_of(def)
                        {
                            collect_all_reg_coalescing_candidates_impl(
                                &mut subset,
                                outer,
                                def,
                                scope,
                                defs,
                            );
                        }
                        self.valid_reg_coalescing_candidates.push(subset);
                    }
                }
            }
            sentient::Op::LoadAndSend {
                mutable_addr,
                increment,
                result,
                ..
            }
            | sentient::Op::ReceiveAndStore {
                mutable_addr,
                increment,
                result,
                ..
            }
            | sentient::Op::LoadComputeAndSend {
                mutable_addr,
                increment,
                result,
                ..
            } => {
                self.coalesce_with_mutable_addr(*result, *mutable_addr, *increment, defs, liveness)
            }
            // ⭐ THE ADDRESS RESULT, NOT THE DATUM — `getAddrResult()` is what shares the register.
            sentient::Op::LoadAndExtractScalar {
                mutable_addr,
                increment,
                addr_result,
                ..
            } => self.coalesce_with_mutable_addr(
                *addr_result,
                *mutable_addr,
                *increment,
                defs,
                liveness,
            ),
            // BOTH ENDS, each with its own address and its own increment.
            sentient::Op::LoadAndStore {
                src_mutable_addr,
                src_inc,
                dst_mutable_addr,
                dst_inc,
                results,
                ..
            } => {
                self.coalesce_with_mutable_addr(
                    results.0,
                    *src_mutable_addr,
                    *src_inc,
                    defs,
                    liveness,
                );
                self.coalesce_with_mutable_addr(
                    results.1,
                    *dst_mutable_addr,
                    *dst_inc,
                    defs,
                    liveness,
                );
            }
            sentient::Op::ScalarAdd {
                lhs, rhs, result, ..
            }
            | sentient::Op::ScalarSub {
                lhs, rhs, result, ..
            } => self.coalesce_with_operand(*result, [*lhs, *rhs], defs, liveness),
            _ => {}
        }
    }

    /// Replaces: e330_getNextMaxWeight
    ///
    /// The heavier of the next register-init candidate and the next coalescing subset, with the index
    /// it was taken from advanced (`OldRegisterInitialization.cpp:505-533`).
    ///
    /// ⛔ TRAP: A TIE GOES TO THE COALESCING SUBSET — the test is `>`, so an equal-weight reg-init
    /// candidate waits.
    /// ⚠️ THE REFERENCE'S TWO INNER `if`s ARE DEAD: each re-tests the condition the enclosing branch
    /// has just excluded (`:509-511`, `:517-519`), so `pick: None` is reachable only when BOTH lists
    /// are spent — which is the `values.empty()` break signal `e451` reads.
    #[must_use]
    pub fn get_next_max_weight(
        &self,
        reg_init_index: usize,
        coalescing_index: usize,
    ) -> NextMaxWeight {
        match (
            self.valid_reginit_candidates.get(reg_init_index),
            self.valid_reg_coalescing_candidates.get(coalescing_index),
        ) {
            (None, None) => NextMaxWeight {
                reg_init_index,
                coalescing_index,
                pick: None,
            },
            (None, Some(subset)) => NextMaxWeight {
                reg_init_index,
                coalescing_index: coalescing_index + 1,
                pick: Some(Pick::RegCoalescing(subset.clone())),
            },
            (Some(&val), None) => NextMaxWeight {
                reg_init_index: reg_init_index + 1,
                coalescing_index,
                pick: Some(Pick::RegInit(val)),
            },
            (Some(&val), Some(subset)) => {
                if self.ssa_weight.weight_of(val) > weight_of_subset(subset, &self.ssa_weight) {
                    NextMaxWeight {
                        reg_init_index: reg_init_index + 1,
                        coalescing_index,
                        pick: Some(Pick::RegInit(val)),
                    }
                } else {
                    NextMaxWeight {
                        reg_init_index,
                        coalescing_index: coalescing_index + 1,
                        pick: Some(Pick::RegCoalescing(subset.clone())),
                    }
                }
            }
        }
    }
}

/// WHICH LIST THE GREEDY PICK CAME FROM AND WHAT IT WAS — `getNextMaxWeight`'s `bool` and its
/// `std::vector<mlir::Value>`, which are one fact and not two (`:504-506`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pick {
    /// The `bool` is `true`: one register-init candidate, the reference's one-element vector.
    RegInit(Val),
    /// The `bool` is `false`: a whole coalescing subset.
    RegCoalescing(Vec<Val>),
}

/// `getNextMaxWeight`'s four-element tuple — the two indices to resume from and what was taken.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NextMaxWeight {
    /// `i` — where to resume in `valid_reginit_candidates_`.
    pub reg_init_index: usize,
    /// `j` — where to resume in `valid_reg_coalescing_candidates_`.
    pub coalescing_index: usize,
    /// ⛔ `None` IS THE CALLER'S BREAK, the reference's empty vector: both lists are spent. It cannot
    /// be confused with an empty subset because `e329` never pushes one.
    pub pick: Option<Pick>,
}

// crustify:todo: e449_calcSSAWeight
//   authority : dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:252  (9 body lines, level 2)
//   original  : int RegisterInitInfo::calcSSAWeight(Region &reg)
//   calls     : e327_calcSSAWeight

// crustify:todo: e450_collectAllRegCoalescingCandidates
//   authority : dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:361  (9 body lines, level 2)
//   original  : void RegisterInitInfo::collectAllRegCoalescingCandidates(Region &region)
//   calls     : e329_collectAllRegCoalescingCandidates

// crustify:todo: e451_collectRegInitAndRegCoalescingCandidateFast
//   authority : dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:548  (114 body lines, level 2)
//   original  : void RegisterInitInfo::collectRegInitAndRegCoalescingCandidateFast( dataflow::ProgramUnitOp &unit, mlir::Value u)
//   calls     : e063_getMaxRegNum, e107_hasUniformizeRegion, e252_size, e330_getNextMaxWeight

#[cfg(test)]
mod unit_tests {
    use super::{
        Enclosing, GtrRegInit, NextMaxWeight, Pick, RegInitLocalePriority, RegisterInitInfo,
        SsaWeight, SsaWeights, TripCount, collect_all_reg_coalescing_candidates_impl,
        get_register_init_priority, has_uniformize_region, total_trip_count, weight_of_subset,
    };
    use crate::arch::Elements;
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::dialects::dataflow::{
        ConsumerCount, MulticastGroupId, OutstandingRequests,
    };
    use crate::islands::dataflow_ir::dialects::uniform::LocalRegion;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::sentient::{Carried, Reg, RegType, ShuffleMode};
    use crate::islands::sentient::dialects::{Definitions, Op, Val, dataflow, sentient, uniform};
    use crate::transform::sentient::analyses::Liveness;

    /// The out-of-scope liveness, answering from the pairs that DO overlap — the only way to observe
    /// which subsets `e329` offers.
    struct Overlapping(Vec<(Val, Val)>);

    impl Liveness for Overlapping {
        fn update_live_ranges_for_program_header_promotion(&mut self, _candidate: Val) {
            todo!("this fake answers overlaps only; no unit here promotes through it")
        }

        fn is_live_range_overlaps(&self, val1: Val, val2: Val) -> bool {
            self.0.contains(&(val1, val2))
        }
    }

    /// `%r = dataflow.get_unit`.
    fn get_unit(result: u32) -> Op {
        Op::Dataflow(dataflow::Op::GetUnit {
            result: Val(result),
            residency: crate::units::Residency::Global,
            unit: crate::units::DfirUnit::L3lu,
            num_folds: None,
        })
    }

    /// One carried position in the named register file.
    fn carried(init: u32, arg: u32, result: u32, locale: RegType) -> Carried {
        Carried {
            init: Val(init),
            arg: Val(arg),
            result: Val(result),
            reg: reg(locale),
            program_header: false,
            element_size: None,
        }
    }

    /// `%r = sentient.load_and_send %mutable_addr, .., %increment`.
    fn load_and_send(result: u32, mutable_addr: u32, increment: u32) -> Op {
        Op::Sentient(sentient::Op::LoadAndSend {
            mutable_addr: Val(mutable_addr),
            immutable_addr: Val(99),
            increment: Val(increment),
            consumer: SendEnd::to_self(Val(98)),
            result: Val(result),
            extent: sentient::Extent {
                total_elements: Elements(64),
                element_size: Bits(32),
                chunk_size: Elements(1),
                chunk_stride: Elements(1),
                burst_size: Elements(1),
            },
            interleaved_group: Elements(0),
            rotate_val: None,
            dir: None,
            shuffle_mode: ShuffleMode::NoShuffle,
            reg: reg(RegType::Lrf),
            dbg_name: None,
        })
    }

    /// `%out = sentient.scalar_add %lhs, %rhs`.
    fn scalar_add(result: u32, lhs: u32, rhs: u32) -> Op {
        Op::Sentient(sentient::Op::ScalarAdd {
            lhs: Val(lhs),
            rhs: Val(rhs),
            result: Val(result),
            reg: None,
            element_size: None,
            ty: ScalarTy::Index,
        })
    }

    /// An unassigned register in the named file.
    fn reg(locale: RegType) -> Reg {
        Reg {
            locale,
            index: None,
        }
    }

    /// `%r = sentient.scalar_constant {value = N : si64} : index`.
    fn constant(result: u32, value: i64) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result: Val(result),
            reg_locale: RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// `%out = sentient.scalar_copy %inp`, in the named register file.
    fn copy(result: u32, input: u32, locale: RegType) -> Op {
        Op::Sentient(sentient::Op::ScalarCopy {
            input: Val(input),
            result: Val(result),
            reg: reg(locale),
            element_size: None,
            program_header: false,
        })
    }

    /// `%g = dataflow.create_multicast_group(%p -> ())`, optimisable or not.
    fn multicast(result: u32, optimisable: bool) -> Op {
        Op::Dataflow(dataflow::Op::CreateMulticastGroup {
            result: Val(result),
            producer: Val(90),
            consumers: Vec::new(),
            num_consumers: ConsumerCount(1),
            group_id: MulticastGroupId(0),
            count: OutstandingRequests(0),
            init_packet_opt_en: optimisable,
        })
    }

    /// `sentient.for %iv = %bound` carrying whatever is given.
    fn for_op(iv: u32, bound: u32, carried: Vec<Carried>, body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv: Val(iv),
            bound: Val(bound),
            bound_reg: None,
            carried,
            dbg_name: None,
            body,
        })
    }

    /// The trip counts MULTIPLY outwards, and a loop whose bound is not a constant contributes
    /// NOTHING — not even its real trip count. See [`total_trip_count`]'s TRAP.
    #[test]
    fn the_trip_count_is_the_product_of_the_constant_bounds() {
        let inner = for_op(4, 2, Vec::new(), Vec::new());
        let outer = for_op(3, 1, Vec::new(), vec![inner.clone()]);
        // A bound with no defining op at all: the reference's `dyn_cast` of a null pointer.
        let dynamic = for_op(5, 99, Vec::new(), Vec::new());
        let scope = vec![constant(1, 4), constant(2, 3), outer.clone()];
        let module = [scope.as_slice()];
        let defs = Definitions::from_innermost(&module);

        let chain = [&inner, &outer];
        assert_eq!(
            total_trip_count(Enclosing::from_innermost(&chain), defs),
            TripCount(12)
        );

        let chain = [&dynamic, &inner, &outer];
        assert_eq!(
            total_trip_count(Enclosing::from_innermost(&chain), defs),
            TripCount(12)
        );
    }

    /// A `gtr` IS ALLOWED ONLY THROUGH AN OPTIMISABLE MULTICAST GROUP, and the required files are
    /// required whatever the flag says.
    #[test]
    fn a_gtr_is_allowed_only_when_its_multicast_group_is_optimisable() {
        let scope = vec![
            copy(1, 0, RegType::Gtr),
            multicast(0, true),
            copy(3, 2, RegType::Gtr),
            multicast(2, false),
            copy(4, 90, RegType::Ebr),
            copy(5, 90, RegType::Lrf),
            copy(6, 90, RegType::Jcr),
        ];
        let module = [scope.as_slice()];
        let defs = Definitions::from_innermost(&module);
        let priority = |val: u32| get_register_init_priority(Val(val), defs, GtrRegInit::Enabled);

        assert_eq!(priority(1), RegInitLocalePriority::Allowed);
        assert_eq!(priority(3), RegInitLocalePriority::NotAllowed);
        assert_eq!(priority(4), RegInitLocalePriority::Required);
        assert_eq!(priority(5), RegInitLocalePriority::Allowed);
        // `unrelated`/`jcr` is in neither array: the tail.
        assert_eq!(priority(6), RegInitLocalePriority::NotAllowed);
        // ⛔ AND THE FLAG CLOSES THE OPTIMISABLE CASE TOO.
        assert_eq!(
            get_register_init_priority(Val(1), defs, GtrRegInit::Disabled),
            RegInitLocalePriority::NotAllowed
        );
    }

    /// ⭐ EVERY UNIT'S GROUP MUST BE OPTIMISABLE — a uniformized copy reads one group per unit
    /// through the map, and one unoptimisable group decides the whole answer.
    #[test]
    fn a_uniformized_gtr_needs_every_units_group_optimisable() {
        let unit = |result: u32| {
            Op::Dataflow(dataflow::Op::GetUnit {
                result: Val(result),
                residency: crate::units::Residency::Global,
                unit: crate::units::DfirUnit::L3lu,
                num_folds: None,
            })
        };
        // `%20 = uniform.query_map (map:%10, key:%arg0)` feeding a `gtr` copy, inside a region
        // whose argument stands for both units.
        // ⚠️ THE COPY AND ITS QUERY SIT AT UNIT SCOPE, NOT INSIDE THE REGION, and that is an ISLAND
        // GAP rather than a choice: [`uniform::LocalRegion::body`] is the DataflowIR rung's op type,
        // so no `sentient.scalar_copy` can nest in a uniform region here, though the reference's own
        // `uniformization.mlir:816` writes one that way. What the region has to supply for this unit
        // is the BINDING of its argument, which is what the key is looked up through.
        let region = LocalRegion {
            arg: Val(30),
            units: vec![Val(10), Val(11)],
            body: Vec::new(),
        };
        let scope = |second_optimisable: bool| {
            vec![
                unit(10),
                unit(11),
                multicast(12, true),
                multicast(13, second_optimisable),
                Op::Uniform(uniform::Op::DefImmutableMapping {
                    result: Val(20),
                    pairs: vec![(Val(10), Val(12)), (Val(11), Val(13))],
                }),
                Op::Uniform(uniform::Op::UniformizeRegions {
                    regions: vec![region.clone()],
                    results: Vec::new(),
                }),
                Op::Uniform(uniform::Op::QueryMap {
                    result: Val(31),
                    map: Val(20),
                    key: Val(30),
                }),
                copy(32, 31, RegType::Gtr),
            ]
        };

        let both = scope(true);
        let module = [both.as_slice()];
        let defs = Definitions::from_innermost(&module);
        assert_eq!(
            get_register_init_priority(Val(32), defs, GtrRegInit::Enabled),
            RegInitLocalePriority::Allowed
        );

        let one = scope(false);
        let module = [one.as_slice()];
        let defs = Definitions::from_innermost(&module);
        assert_eq!(
            get_register_init_priority(Val(32), defs, GtrRegInit::Enabled),
            RegInitLocalePriority::NotAllowed
        );
    }

    /// A SUBSET'S WEIGHT IS HALVED, and an unscored member contributes zero.
    #[test]
    fn a_subsets_weight_is_half_the_sum_of_its_members() {
        let mut weights = SsaWeights::default();
        weights.set(Val(1), SsaWeight(3));
        weights.set(Val(2), SsaWeight(5));

        assert_eq!(weight_of_subset(&[Val(1), Val(2)], &weights), SsaWeight(4));
        // An odd sum truncates, and `%9` is not in the map at all.
        assert_eq!(weight_of_subset(&[Val(1), Val(9)], &weights), SsaWeight(1));
    }

    /// THE CHAIN OF CARRIED ITER ARGS IS FOLLOWED OUTWARDS, one loop at a time.
    #[test]
    fn the_candidate_subset_follows_the_carried_chain_outwards() {
        let inner_carried = vec![Carried {
            init: Val(2),
            arg: Val(5),
            result: Val(6),
            reg: reg(RegType::Lrf),
            program_header: false,
            element_size: None,
        }];
        let inner = for_op(
            4,
            11,
            inner_carried.clone(),
            // The single use of `%5` that `hasOneUse()` accepts.
            vec![copy(7, 5, RegType::Lrf)],
        );
        let outer = for_op(
            0,
            10,
            vec![Carried {
                init: Val(1),
                arg: Val(2),
                result: Val(3),
                reg: reg(RegType::Lrf),
                program_header: false,
                element_size: None,
            }],
            vec![inner],
        );
        let scope = vec![outer];
        let module = [scope.as_slice()];
        let defs = Definitions::from_innermost(&module);

        let mut subset = Vec::new();
        collect_all_reg_coalescing_candidates_impl(
            &mut subset,
            &inner_carried,
            Val(5),
            &scope,
            defs,
        );
        assert_eq!(subset, vec![Val(5), Val(2)]);
    }

    /// BOTH MEMBERS OF THE `isa<>` COUNT, at any depth — and a unit holding neither answers `false`.
    #[test]
    fn a_uniform_group_is_found_at_any_depth() {
        let nest = |op: Op| vec![for_op(0, 1, Vec::new(), vec![op])];

        assert!(has_uniformize_region(&nest(Op::Uniform(
            uniform::Op::UniformizeRegions {
                regions: Vec::new(),
                results: Vec::new(),
            }
        ))));
        assert!(has_uniformize_region(&nest(Op::Uniform(
            uniform::Op::EqualizePattern {
                regions: Vec::new(),
            }
        ))));
        assert!(!has_uniformize_region(&nest(copy(1, 0, RegType::Lrf))));
    }

    /// e326 — the pair's TARGET is retargeted and its source is not, and every member of a coalescing
    /// subset is.
    #[test]
    fn e326_retargets_the_second_of_a_pair_and_every_subset_member() {
        let mut info = RegisterInitInfo {
            enforced_virtual_assign: vec![(Val(1), Val(2)), (Val(3), Val(2)), (Val(2), Val(8))],
            final_reg_coalescing_candidates: vec![vec![Val(2), Val(4)], vec![Val(5)]],
            ..RegisterInitInfo::default()
        };

        info.replace_virtual_assign_target(Val(2), Val(7));

        assert_eq!(
            info.enforced_virtual_assign,
            // ⛔ `(2, 8)` KEEPS ITS SOURCE: only `.second` is ever rewritten.
            vec![(Val(1), Val(7)), (Val(3), Val(7)), (Val(2), Val(8))]
        );
        assert_eq!(
            info.final_reg_coalescing_candidates,
            vec![vec![Val(7), Val(4)], vec![Val(5)]]
        );
    }

    /// e327 — the weight is the enclosing loops' product, and only the OUTERMOST loop's carried
    /// constants and a promotable `scalar_copy` become candidates.
    #[test]
    fn e327_weighs_every_result_and_offers_only_the_outermost_loops_candidates() {
        let inner = for_op(44, 11, vec![carried(12, 46, 47, RegType::Lrf)], Vec::new());
        let promoted = copy(23, 13, RegType::Lbr);
        let weighed = load_and_send(60, 61, 12);
        let outer = for_op(
            40,
            10,
            vec![carried(12, 42, 43, RegType::Lrf)],
            vec![inner.clone(), promoted.clone(), weighed.clone()],
        );
        let scope = vec![
            constant(10, 4),
            constant(11, 3),
            constant(12, 0),
            get_unit(13),
            outer.clone(),
        ];
        let module = [scope.as_slice()];
        let defs = Definitions::from_innermost(&module);
        let mut info = RegisterInitInfo::default();

        let none: [&Op; 0] = [];
        info.calc_ssa_weight(
            &outer,
            Enclosing::from_innermost(&none),
            defs,
            GtrRegInit::Enabled,
        );
        let in_outer = [&outer];
        info.calc_ssa_weight(
            &inner,
            Enclosing::from_innermost(&in_outer),
            defs,
            GtrRegInit::Enabled,
        );
        info.calc_ssa_weight(
            &promoted,
            Enclosing::from_innermost(&in_outer),
            defs,
            GtrRegInit::Enabled,
        );
        info.calc_ssa_weight(
            &weighed,
            Enclosing::from_innermost(&in_outer),
            defs,
            GtrRegInit::Enabled,
        );

        // The outer loop runs once; everything inside it runs 4 times.
        assert_eq!(info.ssa_weight.weight_of(Val(42)), SsaWeight(1));
        assert_eq!(info.ssa_weight.weight_of(Val(46)), SsaWeight(4));
        assert_eq!(info.ssa_weight.weight_of(Val(23)), SsaWeight(4));
        assert_eq!(info.ssa_weight.weight_of(Val(60)), SsaWeight(4));
        // ⛔ `%46` IS WEIGHED AND NEVER OFFERED: its loop is not the outermost one. `%60` is weighed
        // and never offered because a transfer result is not a candidate at all.
        assert_eq!(info.valid_reginit_candidates, vec![Val(42)]);
        assert_eq!(info.required_reginit_candidates, vec![Val(23)]);
    }

    /// e328 — heaviest subset first, and equal weights keep their collection order.
    #[test]
    fn e328_orders_the_subsets_by_decreasing_weight() {
        let mut info = RegisterInitInfo {
            valid_reg_coalescing_candidates: vec![
                vec![Val(1), Val(2)],
                vec![Val(3), Val(4)],
                vec![Val(5), Val(6)],
            ],
            ..RegisterInitInfo::default()
        };
        for (val, weight) in [(1, 2), (2, 2), (3, 20), (4, 20), (5, 2), (6, 2)] {
            info.ssa_weight.set(Val(val), SsaWeight(weight));
        }

        info.sort_reg_coalescing_candidates();

        assert_eq!(
            info.valid_reg_coalescing_candidates,
            vec![
                vec![Val(3), Val(4)],
                // ⭐ THE TIE KEEPS ITS ORDER, which `std::sort` does not promise.
                vec![Val(1), Val(2)],
                vec![Val(5), Val(6)],
            ]
        );
    }

    /// e329 — a carried chain rooted in a block argument is offered and one rooted in a constant is
    /// DROPPED; a transfer's two effects are independent; and an add stops at its first register
    /// operand.
    #[test]
    fn e329_collects_the_subsets_and_enforces_only_a_zero_increment() {
        let inner = for_op(
            44,
            45,
            vec![carried(42, 46, 47, RegType::Lrf)],
            vec![copy(48, 46, RegType::Lrf)],
        );
        let outer = for_op(
            40,
            41,
            vec![carried(50, 42, 43, RegType::Lrf)],
            vec![inner.clone()],
        );
        let zero_increment = load_and_send(60, 61, 50);
        let overlapping = load_and_send(70, 71, 72);
        let add = scalar_add(80, 50, 81);
        let scope = vec![
            constant(41, 2),
            constant(45, 3),
            constant(50, 0),
            constant(72, 5),
            copy(81, 90, RegType::Lrf),
            outer.clone(),
            zero_increment.clone(),
            overlapping.clone(),
            add.clone(),
        ];
        let module = [scope.as_slice()];
        let defs = Definitions::from_innermost(&module);
        let liveness = Overlapping(vec![(Val(70), Val(71))]);
        let mut info = RegisterInitInfo::default();

        for op in [&outer, &inner, &zero_increment, &overlapping, &add] {
            info.collect_all_reg_coalescing_candidates(op, &scope, defs, &liveness);
        }

        assert_eq!(
            info.valid_reg_coalescing_candidates,
            vec![
                // ⛔ THE OUTER LOOP CONTRIBUTES NOTHING: its carried value arrives from a constant,
                // so its subset is built and then dropped.
                vec![Val(46), Val(42)],
                vec![Val(60), Val(61)],
                // `%70`'s live range overlaps `%71`'s, so no subset — and its increment is 5, so no
                // enforced pairing either.
                vec![Val(80), Val(81)],
            ]
        );
        assert_eq!(info.enforced_virtual_assign, vec![(Val(60), Val(61))]);
    }

    /// e330 — the heavier candidate wins, a TIE goes to the coalescing subset, each list drains on its
    /// own, and both spent is the caller's break.
    #[test]
    fn e330_takes_the_heavier_candidate_and_gives_a_tie_to_the_subset() {
        let mut info = RegisterInitInfo {
            valid_reginit_candidates: vec![Val(1), Val(2)],
            valid_reg_coalescing_candidates: vec![vec![Val(3), Val(4)]],
            ..RegisterInitInfo::default()
        };
        info.ssa_weight.set(Val(1), SsaWeight(10));
        // The subset's weight is HALF the sum: 4.
        info.ssa_weight.set(Val(3), SsaWeight(4));
        info.ssa_weight.set(Val(4), SsaWeight(4));
        // `%2` ties the subset at 4.
        info.ssa_weight.set(Val(2), SsaWeight(4));

        assert_eq!(
            info.get_next_max_weight(0, 0),
            NextMaxWeight {
                reg_init_index: 1,
                coalescing_index: 0,
                pick: Some(Pick::RegInit(Val(1))),
            }
        );
        // ⛔ THE TIE GOES TO THE SUBSET — the test is `>`.
        assert_eq!(
            info.get_next_max_weight(1, 0),
            NextMaxWeight {
                reg_init_index: 1,
                coalescing_index: 1,
                pick: Some(Pick::RegCoalescing(vec![Val(3), Val(4)])),
            }
        );
        // The coalescing list is spent, so the reg-init one drains alone.
        assert_eq!(
            info.get_next_max_weight(1, 1),
            NextMaxWeight {
                reg_init_index: 2,
                coalescing_index: 1,
                pick: Some(Pick::RegInit(Val(2))),
            }
        );
        // ⭐ BOTH SPENT: the `values.empty()` break signal.
        assert_eq!(
            info.get_next_max_weight(2, 1),
            NextMaxWeight {
                reg_init_index: 2,
                coalescing_index: 1,
                pick: None,
            }
        );
    }
}

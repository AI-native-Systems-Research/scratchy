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

//! `SetMaskRE.cpp` — 4 of the campaign's 656 units (dependency level(s) [0, 1]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e193_deadIncrMaskOptimization` | 193 | 0 | 31 | `dcc/src/Transform/Sentient/SetMaskRE.cpp:346` |
//! | `e194_absorbIncrMask` | 194 | 0 | 58 | `dcc/src/Transform/Sentient/SetMaskRE.cpp:378` |
//! | `e377_setSetMaskWrap` | 377 | 1 | 9 | `dcc/src/Transform/Sentient/SetMaskRE.cpp:299` |
//! | `e378_optimize` | 378 | 1 | 34 | `dcc/src/Transform/Sentient/SetMaskRE.cpp:311` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so every item below is reachable only from this
// module's own tests until `e474_runOn`/`e536_runOnOperation` (levels 2/3) land and something calls
// it. CI runs clippy with `-D warnings`, so without this the first ported leaf fails the gate.
// ⭐ REMOVE THIS WITH e474: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use core::num::NonZeroU32;

use sys_arch_spec::fields::{self, ImmSpec};

use super::incr_mask_gen_value::{IncrMaskGenValue, Increment};
use super::set_mask_gen_value::SetMaskGenValue;
use crate::islands::dataflow_ir::Values;
use crate::islands::sentient::dialects::{self as ir, Op, Val, sentient};
use crate::transform::sentient::cfg_simplification_sentient_level::pattern_simplification_manager::OpPath;
use crate::transform::sentient::utils::str_eq;

/// `RDENode::getDataflowGen()` AS e193/e194 READ IT — the two `dynamic_cast`s, as cases.
///
/// ⭐ THE GENS NAME THEIR OPS BY [`OpPath`] AND CARRY NO BORROW: both units rewrite the unit while
/// holding the sibling chain, which a `&Op` could not survive and an `Operation *` needs nothing for.
///
/// ⭐ e375 CONSTRUCTS THESE (`setDataflowGen`, `SetMaskRE.cpp:203-219`) — a batch filling that anchor
/// should UNION with this enum rather than declare a second one.
#[derive(Debug, Clone)]
pub(crate) enum DataflowGen {
    /// `dynamic_cast<SetMaskGenValue *>` succeeded.
    SetMask(SetMaskGenValue),
    /// `dynamic_cast<IncrMaskGenValue *>` succeeded.
    IncrMask(IncrMaskGenValue),
}

impl DataflowGen {
    /// `DataFlowDefinitionBase::isDead()`.
    #[must_use]
    pub(crate) const fn is_dead(&self) -> bool {
        match self {
            DataflowGen::SetMask(value) => value.is_dead(),
            DataflowGen::IncrMask(value) => value.is_dead(),
        }
    }

    /// `DataFlowDefinitionBase::isOptimized()`.
    #[must_use]
    pub(crate) const fn is_optimized(&self) -> bool {
        match self {
            DataflowGen::SetMask(value) => value.is_optimized(),
            DataflowGen::IncrMask(value) => value.is_optimized(),
        }
    }
}

/// `setmask_wrap_` (`SetMaskRE.hpp:123`) — the mask value's wrap point.
///
/// ⛔ `DT_CHECK(setmask_wrap_ != 0)` (`:414`) IS THIS TYPE. e377 derives it as `1 << imm_bits` from
/// the ISA's immediate info, which cannot be zero and cannot be negative — so the `int64_t` field's
/// two unrepresentable states are both gone rather than checked at the one site that divides by it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SetMaskWrap(NonZeroU32);

impl SetMaskWrap {
    /// `1 << imm_info.first`.
    #[must_use]
    pub(crate) const fn of(wrap: NonZeroU32) -> SetMaskWrap {
        SetMaskWrap(wrap)
    }

    /// The wrap point itself.
    #[must_use]
    pub(crate) const fn get(self) -> u32 {
        self.0.get()
    }
}

/// THE OPS e194 READS AND REWRITES — one program's preamble and the body of the unit being optimized.
///
/// ⭐ BOTH, BECAUSE A MASK CONSTANT MAY SIT IN EITHER: `LexicalOrdering::runOn`
/// (`dcc/src/Transform/Sentient/LexicalOrdering.cpp:113`) hoists constants OUT of unit bodies and
/// `replaceXRFImplicitIncrWithAdd` builds them straight into the preamble, while the RDE tree is
/// built over the unit's operation alone (`SetMaskRE.cpp:156-157`) — so the [`OpPath`]s are the
/// body's, and a `getDefiningOp()` searches outwards from it.
#[derive(Debug)]
pub(crate) struct UnitScope<'a> {
    /// `Program::preamble`.
    pub(crate) preamble: &'a mut Vec<Op>,
    /// `ProgramUnit::body` — what the tree's [`OpPath`]s are relative to.
    pub(crate) body: &'a mut Vec<Op>,
}

impl UnitScope<'_> {
    /// `Value::getDefiningOp()`, innermost region first.
    fn defining_op(&self, val: Val) -> Option<&Op> {
        ir::defining_op(val, self.body.as_slice())
            .or_else(|| ir::defining_op(val, self.preamble.as_slice()))
    }

    /// `setmask_op.getMaskValue()` — `None` when the path does not name a `sentient.set_mask`.
    fn mask_value_of(&self, at: &OpPath) -> Option<Val> {
        match op_at(self.body.as_slice(), at.path())? {
            Op::Sentient(sentient::Op::SetMask { mask_value, .. }) => Some(*mask_value),
            _ => None,
        }
    }

    /// `setmask_op.getMaskValueMutable().assign(mask_value)`, with the same `None` case.
    fn assign_mask_value(&mut self, at: &OpPath, mask_value: Val) -> bool {
        let Some(Op::Sentient(sentient::Op::SetMask {
            mask_value: operand,
            ..
        })) = op_at_mut(self.body.as_mut_slice(), at.path())
        else {
            return false;
        };
        *operand = mask_value;
        true
    }
}

/// The op an [`OpPath`] names, if it still names one.
fn op_at<'a>(root: &'a [Op], path: &[(u32, u32)]) -> Option<&'a Op> {
    let (&(_, index), rest) = path.split_first()?;
    let op = root.get(index as usize)?;
    if rest.is_empty() {
        return Some(op);
    }
    let (region, _) = rest[0];
    match op {
        Op::Sentient(inner) => op_at(
            sentient::regions(inner).into_iter().nth(region as usize)?,
            rest,
        ),
        Op::AffineFor(loop_op) => op_at(&loop_op.body, rest),
        // The [`OpPath`] numbers a `uniform.regions` local region like any other sub-region, so this
        // resolves one — same descent as [`pattern_simplification_manager::op_at`], which builds them.
        Op::UniformRegions(regions) => op_at(&regions.regions().get(region as usize)?.body, rest),
        Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_)
        | Op::Affine(_)
        | Op::Vector(_)
        | Op::Arith(_)
        | Op::Scf(_)
        | Op::Symbol(_)
        | Op::Uniform(_) => None,
    }
}

/// [`op_at`], mutably.
fn op_at_mut<'a>(root: &'a mut [Op], path: &[(u32, u32)]) -> Option<&'a mut Op> {
    let (&(_, index), rest) = path.split_first()?;
    let op = root.get_mut(index as usize)?;
    if rest.is_empty() {
        return Some(op);
    }
    let (region, _) = rest[0];
    match op {
        Op::Sentient(inner) => op_at_mut(
            sentient::regions_mut(inner)
                .into_iter()
                .nth(region as usize)?,
            rest,
        ),
        Op::AffineFor(loop_op) => op_at_mut(&mut loop_op.body, rest),
        Op::UniformRegions(regions) => op_at_mut(
            &mut regions.regions_mut().get_mut(region as usize)?.body,
            rest,
        ),
        Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_)
        | Op::Affine(_)
        | Op::Vector(_)
        | Op::Arith(_)
        | Op::Scf(_)
        | Op::Symbol(_)
        | Op::Uniform(_) => None,
    }
}

/// `SetMaskRDETreeOptimizer` (`SetMaskRE.hpp:83`) — the two `incrmask` optimizations and their state.
///
/// ⭐ `count_` AND `to_be_deleted_` ARE `RDETreeOptimizer`'s, whose class is OUT OF CAMPAIGN SCOPE:
/// both units write them, so the subclass holds them exactly as `ImplicitSyncGenValue` holds the
/// base-class flags e049 prints. `isa_`, `enable_dyn_loop_hoisting_` and `debug_name_` are read only
/// by units this batch does not own and arrive with them.
#[derive(Debug)]
pub(crate) struct SetMaskRdeTreeOptimizer {
    /// `enable_dead_incrmask_removal_` — `!DisableDeadIncrMaskRemoval` at the pass's own callsite.
    enable_dead_incrmask_removal: bool,
    /// `enable_incrmask_absorption_` — `!DisableIncrMaskAbsorption`.
    enable_incrmask_absorption: bool,
    /// `setmask_wrap_`, which the constructor derives before anything runs (e377).
    setmask_wrap: SetMaskWrap,
    /// `RDETreeOptimizer::count_` — how many optimizations fired, which `optimize()` returns.
    count: u32,
    /// `RDETreeOptimizer::to_be_deleted_` — the ops e378's `for (auto dead : to_be_deleted_)
    /// dead->erase()` removes once the walk is over.
    to_be_deleted: Vec<OpPath>,
}

impl SetMaskRdeTreeOptimizer {
    /// `SetMaskRDETreeOptimizer(tree, isa, .., enable_dead_incrmask_removal, ..)`.
    #[must_use]
    pub(crate) fn of(
        setmask_wrap: SetMaskWrap,
        enable_dead_incrmask_removal: bool,
        enable_incrmask_absorption: bool,
    ) -> SetMaskRdeTreeOptimizer {
        SetMaskRdeTreeOptimizer {
            enable_dead_incrmask_removal,
            enable_incrmask_absorption,
            setmask_wrap,
            count: 0,
            to_be_deleted: Vec::new(),
        }
    }

    /// `count_`.
    #[must_use]
    pub(crate) const fn count(&self) -> u32 {
        self.count
    }

    /// `to_be_deleted_`.
    #[must_use]
    pub(crate) fn to_be_deleted(&self) -> &[OpPath] {
        &self.to_be_deleted
    }

    /// Replaces: e193_deadIncrMaskOptimization
    ///
    /// An `incrmask` whose next LIVE sibling hard-sets the mask is dead: its op is recorded for
    /// erasure, its GenValue is marked dead and the count goes up — once per such sibling.
    ///
    /// ⛔ `DT_CHECK_MSG(node, "expected valid node")` IS THE SLICE: `siblings` is `node` followed by
    /// its `getNextSibling()` chain, so there is no null node left to check.
    ///
    /// ⛔ A NODE WITH NO GEN ENDS THE INNER SEARCH BUT NOT THE OUTER ONE — `if (!sib_gen) break;`
    /// against `if (!curr_gen) continue;`, which is what stops a use (a mac has no gen) from being
    /// walked past.
    pub(crate) fn dead_incr_mask_optimization(&mut self, siblings: &mut [Option<DataflowGen>]) {
        if !self.enable_dead_incrmask_removal {
            return;
        }
        for curr in 0..siblings.len() {
            let (head, tail) = siblings.split_at_mut(curr + 1);
            let Some(DataflowGen::IncrMask(incrmask_gen)) = &mut head[curr] else {
                continue;
            };
            if incrmask_gen.is_dead()
                || incrmask_gen.is_optimized()
                || incrmask_gen.is_unknown_value()
            {
                continue;
            }
            if !hard_sets_the_mask(tail) {
                continue;
            }
            if let Some(op) = incrmask_gen.op() {
                self.to_be_deleted.push(op.clone());
            }
            incrmask_gen.set_is_dead();
            self.count += 1;
        }
    }

    /// Replaces: e194_absorbIncrMask
    ///
    /// Every live `incrmask` reached from a live `set_mask` with no use between them folds into it: a
    /// NEW `sentient.scalar_constant` of `(mask + increment) % wrap` is built, the `set_mask` and its
    /// GenValue are pointed at it, and the `incrmask` is recorded for erasure and marked dead.
    ///
    /// ⛔ NO `break` AFTER AN ABSORPTION — consecutive `incrmask`es fold CUMULATIVELY, each one
    /// reading the constant its predecessor just created.
    ///
    /// ⛔ THE NEW CONSTANT GOES TO THE PROGRAM PREAMBLE rather than in front of the old one:
    /// positioning a builder is the mechanism a port may replace, the preamble dominates every unit
    /// body (`replaceXRFImplicitIncrWithAdd` builds its constants there for the same reason), and an
    /// insertion into `body` would shift the positional [`OpPath`]s every gen names its op by.
    pub(crate) fn absorb_incr_mask(
        &mut self,
        siblings: &mut [Option<DataflowGen>],
        scope: &mut UnitScope<'_>,
        values: &mut Values,
    ) {
        if !self.enable_incrmask_absorption {
            return;
        }
        for curr in 0..siblings.len() {
            let (head, tail) = siblings.split_at_mut(curr + 1);
            let Some(DataflowGen::SetMask(setmask_gen)) = &mut head[curr] else {
                continue;
            };
            if setmask_gen.is_dead() || setmask_gen.is_optimized() || setmask_gen.is_unknown_value()
            {
                continue;
            }
            let Some(setmask) = setmask_gen.op().cloned() else {
                continue;
            };
            for sib in tail.iter_mut() {
                let Some(sib_gen) = sib else { break };
                if sib_gen.is_dead() || sib_gen.is_optimized() {
                    continue;
                }
                let DataflowGen::IncrMask(incrmask_gen) = sib_gen else {
                    break;
                };
                let Some(increment) = incrmask_gen.increment() else {
                    break;
                };
                let Some(new_mask_value) = self.absorb_one(&setmask, increment, scope, values)
                else {
                    break;
                };
                setmask_gen.set_mask_value(new_mask_value);
                if let Some(op) = incrmask_gen.op() {
                    self.to_be_deleted.push(op.clone());
                }
                incrmask_gen.set_is_dead();
                self.count += 1;
            }
        }
    }

    /// The absorption itself (`SetMaskRE.cpp:401-425`) — the new mask value, or `None` for the
    /// reference's `if (!mask_value_op) break`.
    fn absorb_one(
        &self,
        setmask: &OpPath,
        increment: Increment,
        scope: &mut UnitScope<'_>,
        values: &mut Values,
    ) -> Option<Val> {
        let Some(mask_value) = scope.mask_value_of(setmask) else {
            todo!(
                "absorbIncrMask: DT_CHECK(setmask_op) (SetMaskRE.cpp:404) — {setmask:?} does not \
                 name a `sentient.set_mask`"
            )
        };
        let Some(Op::Sentient(sentient::Op::ScalarConstant { value, ty, .. })) =
            scope.defining_op(mask_value)
        else {
            return None;
        };
        let (curr_mask, ty) = (*value, *ty);
        let new_mask =
            (curr_mask + i64::from(increment.get())) % i64::from(self.setmask_wrap.get());
        if new_mask < 0 {
            todo!(
                "absorbIncrMask: DT_CHECK_MSG(new_mask >= 0, ..) (SetMaskRE.cpp:416) — the mask \
                 constant {curr_mask} is negative"
            )
        }
        // `ConstantOp::create(builder, mask_value_op.getLoc(), mask_value_op.getType(), new_mask)`.
        let result = values.mint();
        scope
            .preamble
            .push(Op::Sentient(sentient::Op::ScalarConstant {
                value: new_mask,
                result,
                reg_locale: sentient::RegType::Imm,
                ty,
                is_symbol: false,
            }));
        if !scope.assign_mask_value(setmask, result) {
            // Unreachable: `mask_value_of` above read this very operand.
            todo!(
                "absorbIncrMask: DT_CHECK(setmask_op) (SetMaskRE.cpp:404) — {setmask:?} stopped \
                 naming a `sentient.set_mask`"
            )
        }
        Some(result)
    }
}

/// e193's inner loop — whether the first LIVE sibling gen to the right hard-sets the mask.
fn hard_sets_the_mask(siblings: &[Option<DataflowGen>]) -> bool {
    for sib in siblings {
        let Some(sib_gen) = sib else { return false };
        if sib_gen.is_dead() || sib_gen.is_optimized() {
            continue;
        }
        return matches!(sib_gen, DataflowGen::SetMask(value) if !value.is_unknown_value());
    }
    false
}

/// Replaces: e377_setSetMaskWrap
///
/// `setmask_wrap_ = 1 << imm_info.first` — the mask wraps at the range of SETMASK's own immediate
/// field, which on the PT is 3 unsigned bits and so 8.
///
/// ⭐ BOTH OF THE REFERENCE'S CHECKS FAIL THE BUILD HERE, NOT THE RUN, exactly as
/// [`utils::ldstiu_imm_info`](crate::transform::sentient::utils) does for LDSTIU: this is a `const fn`
/// behind [`SETMASK_WRAP`], so a table with no SETMASK or with a second immediate on its type is a
/// compile error rather than an abort in a compilation the pass has already half-rewritten.
///
/// ⛔ THE COMPONENT IS ALWAYS THE PT: `runOn` returns unless the unit is one
/// (`SetMaskRE.cpp:150-153`) and `isa_` is `isa_per_unit_->at(comp)` for that same component (`:172`),
/// so there is no per-unit ISA left to carry.
const fn set_mask_wrap() -> SetMaskWrap {
    let opcodes = fields::Comp::Pt.opcodes();
    let mut i = 0;
    let mut setmask = None;
    while i < opcodes.len() {
        if str_eq(opcodes[i].op, "SETMASK") {
            setmask = Some(opcodes[i].ty);
        }
        i += 1;
    }
    let ty = match setmask {
        Some(ty) => ty.get(),
        None => panic!(
            "the PT defines SETMASK — DT_CHECK_MSG(.., \"Could not find opcode imm info!\") \
             (`SetMaskRE.cpp:301`)"
        ),
    };

    let all = fields::Comp::Pt.fields();
    let mut j = 0;
    let mut width = None;
    let mut found = 0;
    while j < all.len() {
        if all[j].ty.get() == ty
            && let ImmSpec::Imm { bits, .. } = all[j].imm
        {
            width = Some(bits);
            found += 1;
        }
        j += 1;
    }
    let bits = match (width, found) {
        (Some(bits), 1) => bits.get(),
        _ => panic!(
            "SETMASK's instruction type must have exactly one immediate field — \
             DT_CHECK(imm_info_map.size() == 1) (`SetMaskRE.cpp:305`)"
        ),
    };
    match NonZeroU32::new(1 << bits) {
        Some(wrap) => SetMaskWrap::of(wrap),
        None => panic!("`1 << imm_info.first` is never zero — see `ImmWidth::of`"),
    }
}

/// `setmask_wrap_` AS THE CONSTRUCTOR LEAVES IT (`SetMaskRE.hpp:96`) — [`set_mask_wrap`], evaluated at
/// build time so the two `DT_CHECK`s behind it cannot reach a run.
pub(crate) const SETMASK_WRAP: SetMaskWrap = set_mask_wrap();

/// `RDETreeOptimizer::count_` AND `::to_be_deleted_` AS THE OUT-OF-SCOPE BASE METHODS SEE THEM.
///
/// ⭐ BORROWED RATHER THAN OWNED because in the reference they are `this`: `deadDefOptimization` and
/// `redundancyOptimizations` are members of the same object e378 is a member of, so a seam that could
/// not touch the count would silently drop every optimization those two perform.
#[derive(Debug)]
pub(crate) struct Counters<'a> {
    /// `count_`.
    pub(crate) count: &'a mut u32,
    /// `to_be_deleted_`.
    pub(crate) to_be_deleted: &'a mut Vec<OpPath>,
}

/// `RDETreeOptimizer`'S OWN OPTIMIZATIONS (`Analyses/RedundantDefinitionEliminationTree.hpp:364-367`)
/// — a trait, because the base class is OUT OF CAMPAIGN SCOPE and a test must still be able to observe
/// that e378 called them, in this order, on these nodes.
///
/// ⛔ `SetMaskRDETreeOptimizer` OVERRIDES NEITHER, so both are the base's and neither is a unit of this
/// campaign; the crate's one implementation is [`OutOfScopeRdeTreeOptimizer`].
pub(crate) trait RdeTreeOptimizer {
    /// `deadDefOptimization(n)` — removes every dead definition in the node's block.
    fn dead_def_optimization(
        &mut self,
        siblings: &mut [Option<DataflowGen>],
        counters: Counters<'_>,
    );

    /// `redundancyOptimizations(n)` — horizontal commoning and vertical hoisting.
    fn redundancy_optimizations(
        &mut self,
        siblings: &mut [Option<DataflowGen>],
        counters: Counters<'_>,
    );
}

/// THE ONE CRATE IMPLEMENTATION: neither base-class optimization is ported, so both are a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct OutOfScopeRdeTreeOptimizer;

impl RdeTreeOptimizer for OutOfScopeRdeTreeOptimizer {
    fn dead_def_optimization(
        &mut self,
        _siblings: &mut [Option<DataflowGen>],
        _counters: Counters<'_>,
    ) {
        todo!(
            "RDETreeOptimizer::deadDefOptimization \
             (Analyses/RedundantDefinitionEliminationTree.hpp:364) — out of campaign scope"
        )
    }

    fn redundancy_optimizations(
        &mut self,
        _siblings: &mut [Option<DataflowGen>],
        _counters: Counters<'_>,
    ) {
        todo!(
            "RDETreeOptimizer::redundancyOptimizations \
             (Analyses/RedundantDefinitionEliminationTree.hpp:365) — out of campaign scope"
        )
    }
}

/// `tree_` AS e378 WALKS IT — the definitions the tree generated, grouped by parent, plus the order
/// `walk` visits them in.
///
/// ⛔ THE TREE ITSELF IS OUT OF CAMPAIGN SCOPE, so the ORDER IS AN INPUT: `walk` is
/// `kKeepOrderRBFS` over `Analyses/RedundantDefinitionEliminationTree` (`hpp:240`), which this campaign
/// does not port. e375 builds the gens; a caller that has the tree states which sibling group each
/// visit lands in.
///
/// ⭐ GROUPED BY PARENT, BECAUSE THAT IS WHAT THE TWO PORTED OPTIMIZATIONS READ: e193 and e194 both
/// take a node followed by its `getNextSibling()` chain, and `getParentNode()->getFirstChild() == node`
/// is position 0 of the group.
#[derive(Debug, Default)]
pub(crate) struct RdeTree {
    /// One entry per parent node: its children's gens, in sibling order.
    pub(crate) groups: Vec<Vec<Option<DataflowGen>>>,
    /// `walk`'s visit order over the non-root nodes, as `(group, position in group)`.
    pub(crate) order: Vec<(usize, usize)>,
}

impl SetMaskRdeTreeOptimizer {
    /// Replaces: e378_optimize
    ///
    /// The pass's whole effect: absorb every `incrmask` into the `set_mask` before it, drop the dead
    /// ones, then ERASE every op the two recorded and answer how many optimizations fired.
    ///
    /// ⛔ THE ORDER IS THE PORT. Absorption runs before the dead-definition removals because it
    /// creates work for them, the two per-block removals run only on a parent's FIRST child, and dead
    /// `incrmask` removal runs before `deadDefOptimization` for the same reason.
    ///
    /// ⛔ THE ERASURES HAPPEN AFTER THE WALK, IN DESCENDING PATH ORDER: an [`OpPath`] is positional, so
    /// removing a lower index first would shift every path still to come. The reference erases through
    /// `Operation *` and needs no order.
    ///
    /// ⛔ `if (node == tree_.getRoot()) return nullptr` IS THE GROUPING: [`RdeTree::order`] names only
    /// nodes that have a parent. And the `DEBUG_WITH_TYPE` trace `old_count` exists for is dropped —
    /// it is the one thing `count_ != old_count` gates.
    pub(crate) fn optimize(
        &mut self,
        tree: &mut RdeTree,
        scope: &mut UnitScope<'_>,
        values: &mut Values,
        base: &mut impl RdeTreeOptimizer,
    ) -> u32 {
        for (group, position) in tree.order.clone() {
            let Some(siblings) = tree
                .groups
                .get_mut(group)
                .and_then(|siblings| siblings.get_mut(position..))
            else {
                continue;
            };
            self.absorb_incr_mask(siblings, scope, values);
            if position == 0 {
                self.dead_incr_mask_optimization(siblings);
                base.dead_def_optimization(
                    siblings,
                    Counters {
                        count: &mut self.count,
                        to_be_deleted: &mut self.to_be_deleted,
                    },
                );
            }
            base.redundancy_optimizations(
                siblings,
                Counters {
                    count: &mut self.count,
                    to_be_deleted: &mut self.to_be_deleted,
                },
            );
        }

        let mut dead: Vec<&OpPath> = self.to_be_deleted.iter().collect();
        dead.sort_unstable_by(|a, b| b.path().cmp(a.path()));
        for op in dead {
            erase_at(scope.body, op.path());
        }
        self.count
    }
}

/// `Operation::erase()` for the op an [`OpPath`] names, and nothing when it names none.
fn erase_at(root: &mut Vec<Op>, path: &[(u32, u32)]) -> bool {
    let Some((&(_, index), rest)) = path.split_first() else {
        return false;
    };
    if rest.is_empty() {
        if index as usize >= root.len() {
            return false;
        }
        root.remove(index as usize);
        return true;
    }
    let (region, _) = rest[0];
    let Some(op) = root.get_mut(index as usize) else {
        return false;
    };
    let Some(inner) = ir::regions_mut(op).into_iter().nth(region as usize) else {
        return false;
    };
    erase_at(inner, rest)
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::ty::ScalarTy;

    fn wrap8() -> SetMaskWrap {
        SetMaskWrap::of(NonZeroU32::new(8).expect("8 is not zero"))
    }

    fn constant(result: Val, value: i64) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result,
            reg_locale: sentient::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    fn set_mask(mask_value: Val) -> Op {
        Op::Sentient(sentient::Op::SetMask {
            mask_value,
            dbg_name: None,
        })
    }

    fn incr_mask() -> Op {
        Op::Sentient(sentient::Op::IncrMask { dbg_name: None })
    }

    fn at(index: u32) -> OpPath {
        OpPath::at(&[(0, index)])
    }

    fn incr_mask_gen(index: u32) -> IncrMaskGenValue {
        IncrMaskGenValue::of_incr_mask(&incr_mask(), at(index)).expect("an incrmask generates one")
    }

    /// A base-class stand-in that RECORDS. Both of its methods are a `todo!` in the crate
    /// ([`OutOfScopeRdeTreeOptimizer`]) because the base class is out of campaign scope, so a test
    /// states their answers and observes the order e378 calls them in.
    #[derive(Debug, Default)]
    struct Recording(Vec<&'static str>);

    impl RdeTreeOptimizer for Recording {
        fn dead_def_optimization(
            &mut self,
            _siblings: &mut [Option<DataflowGen>],
            _counters: Counters<'_>,
        ) {
            self.0.push("dead_def");
        }

        fn redundancy_optimizations(
            &mut self,
            _siblings: &mut [Option<DataflowGen>],
            _counters: Counters<'_>,
        ) {
            self.0.push("redundancy");
        }
    }

    /// The header's own example: `incrmask; set_mask 4; vector_mac` loses the `incrmask`.
    #[test]
    fn an_incrmask_before_a_hard_set_is_dead() {
        let mut optimizer = SetMaskRdeTreeOptimizer::of(wrap8(), true, true);
        let mut siblings = vec![
            Some(DataflowGen::IncrMask(incr_mask_gen(0))),
            Some(DataflowGen::SetMask(SetMaskGenValue::of(Val(0), at(2)))),
        ];

        optimizer.dead_incr_mask_optimization(&mut siblings);

        assert_eq!(optimizer.count(), 1);
        assert_eq!(optimizer.to_be_deleted(), [at(0)]);
        let Some(DataflowGen::IncrMask(value)) = &siblings[0] else {
            unreachable!("the incrmask gen stays an incrmask gen")
        };
        assert!(value.is_dead());
    }

    /// The header's own example: `set_mask 4; incrmask` becomes `set_mask 5`.
    #[test]
    fn an_incrmask_after_a_hard_set_is_absorbed() {
        let mut values = Values::default();
        let mask = values.mint();
        let mut preamble = vec![constant(mask, 4)];
        let mut body = vec![set_mask(mask), incr_mask()];
        let mut optimizer = SetMaskRdeTreeOptimizer::of(wrap8(), true, true);
        let mut siblings = vec![
            Some(DataflowGen::SetMask(SetMaskGenValue::of(mask, at(0)))),
            Some(DataflowGen::IncrMask(incr_mask_gen(1))),
        ];

        optimizer.absorb_incr_mask(
            &mut siblings,
            &mut UnitScope {
                preamble: &mut preamble,
                body: &mut body,
            },
            &mut values,
        );

        assert_eq!(optimizer.count(), 1);
        assert_eq!(optimizer.to_be_deleted(), [at(1)]);
        assert_eq!(preamble, [constant(mask, 4), constant(Val(1), 5)]);
        assert_eq!(body, [set_mask(Val(1)), incr_mask()]);
    }

    /// ⛔ CUMULATIVE, AND WRAPPING: three `incrmask`es on a `set_mask 6` with a wrap of 8 leave 1,
    /// which only holds because the absorption does not `break` and each one reads its
    /// predecessor's new constant.
    #[test]
    fn consecutive_incrmasks_fold_cumulatively_and_wrap() {
        let mut values = Values::default();
        let mask = values.mint();
        let mut preamble = vec![constant(mask, 6)];
        let mut body = vec![set_mask(mask), incr_mask(), incr_mask(), incr_mask()];
        let mut optimizer = SetMaskRdeTreeOptimizer::of(wrap8(), true, true);
        let mut siblings = vec![
            Some(DataflowGen::SetMask(SetMaskGenValue::of(mask, at(0)))),
            Some(DataflowGen::IncrMask(incr_mask_gen(1))),
            Some(DataflowGen::IncrMask(incr_mask_gen(2))),
            Some(DataflowGen::IncrMask(incr_mask_gen(3))),
        ];

        optimizer.absorb_incr_mask(
            &mut siblings,
            &mut UnitScope {
                preamble: &mut preamble,
                body: &mut body,
            },
            &mut values,
        );

        assert_eq!(optimizer.count(), 3);
        assert_eq!(optimizer.to_be_deleted(), [at(1), at(2), at(3)]);
        assert_eq!(
            preamble,
            [
                constant(mask, 6),
                constant(Val(1), 7),
                constant(Val(2), 0),
                constant(Val(3), 1)
            ]
        );
        assert_eq!(body[0], set_mask(Val(3)));
    }
    /// e377 — the PT's SETMASK carries one 3-bit unsigned immediate, so the mask wraps at 8. Deriving
    /// it is a `const fn`, so this only states the number the two `DT_CHECK`s already guaranteed.
    #[test]
    fn the_set_mask_wrap_is_the_range_of_setmasks_immediate() {
        assert_eq!(SETMASK_WRAP.get(), 8);
        assert_eq!(SETMASK_WRAP, wrap8());
    }

    /// e378 — the header's own absorption, end to end: `set_mask 4; incrmask` becomes `set_mask 5`,
    /// the `incrmask` is ERASED from the body, one optimization is counted, and the two out-of-scope
    /// hooks run only where the reference runs them.
    #[test]
    fn optimize_walks_absorbs_erases_and_answers_the_count() {
        let mut values = Values::default();
        let mask = values.mint();
        let mut preamble = vec![constant(mask, 4)];
        let mut body = vec![set_mask(mask), incr_mask()];
        let mut optimizer = SetMaskRdeTreeOptimizer::of(SETMASK_WRAP, true, true);
        let mut tree = RdeTree {
            groups: vec![vec![
                Some(DataflowGen::SetMask(SetMaskGenValue::of(mask, at(0)))),
                Some(DataflowGen::IncrMask(incr_mask_gen(1))),
            ]],
            // `walk` is bottom-up, so the last sibling is visited before the first.
            order: vec![(0, 1), (0, 0)],
        };
        let mut base = Recording::default();

        let count = optimizer.optimize(
            &mut tree,
            &mut UnitScope {
                preamble: &mut preamble,
                body: &mut body,
            },
            &mut values,
            &mut base,
        );

        assert_eq!(count, 1);
        assert_eq!(body, [set_mask(Val(1))], "the incrmask is erased, not kept");
        assert_eq!(preamble, [constant(mask, 4), constant(Val(1), 5)]);
        assert_eq!(
            base.0,
            ["redundancy", "dead_def", "redundancy"],
            "`deadDefOptimization` runs once per parent, on its first child"
        );
    }
}

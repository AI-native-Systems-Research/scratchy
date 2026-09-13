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

//! `LocalRegionSplittingForValueCommoning.cpp` — 9 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4, 5, 6]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e061_addLocalRegion` | 061 | 0 | 3 | `dcc/src/Transform/Sentient/LocalRegionSplittingForValueCommoning.cpp:166` |
//! | `e062_collectUniformMaps` | 062 | 0 | 19 | `dcc/src/Transform/Sentient/LocalRegionSplittingForValueCommoning.cpp:308` |
//! | `e063_getMaxRegNum` | 063 | 0 | 15 | `dcc/src/Transform/Sentient/LocalRegionSplittingForValueCommoning.cpp:440` |
//! | `e308_cluster` | 308 | 1 | 14 | `dcc/src/Transform/Sentient/LocalRegionSplittingForValueCommoning.cpp:330` |
//! | `e444_analyze` | 444 | 2 | 32 | `dcc/src/Transform/Sentient/LocalRegionSplittingForValueCommoning.cpp:273` |
//! | `e505_transform` | 505 | 3 | 91 | `dcc/src/Transform/Sentient/LocalRegionSplittingForValueCommoning.cpp:347` |
//! | `e561_run` | 561 | 4 | 19 | `dcc/src/Transform/Sentient/LocalRegionSplittingForValueCommoning.cpp:252` |
//! | `e599_runOn` | 599 | 5 | 11 | `dcc/src/Transform/Sentient/LocalRegionSplittingForValueCommoning.cpp:240` |
//! | `e624_runOnOperation` | 624 | 6 | 6 | `dcc/src/Transform/Sentient/LocalRegionSplittingForValueCommoning.cpp:233` |

pub(crate) mod local_region;
pub(crate) mod uniform_region;

use crate::arch::Arch;
use crate::formats::Bits;
use crate::islands::dataflow_ir::{ValueMapping, Values};
use crate::islands::sentient::dialects::{
    self as dialects, Definitions, Op, UniformRegions, Val, sentient, uniform,
};
use crate::transform::sentient::utils::{Hoisted, NewUse, move_to_common_dominator, path_of};
use local_region::{LocalRegion, OriginalRegion};
use sentient::RegType;
use sys_arch_spec::regfile::{Component, Presence, RegType as RegFile, depth_of};
use uniform_region::UniformRegion;

impl UniformRegion {
    /// Replaces: e061_addLocalRegion
    ///
    /// APPENDS one transformed local region: the units it now represents, and the region of the
    /// original op they came from.
    ///
    /// ⛔ APPEND, NEVER REPLACE. `analyze` (e444) calls this once per cluster it decided on, and that
    /// order is the region order `transform` (e505) then emits (`:273-303`, `:347`).
    pub fn add_local_region(&mut self, units: Vec<Val>, orig_region: OriginalRegion) {
        self.local_regions.push(LocalRegion {
            units,
            original_region: orig_region,
        });
    }
}

/// ONE `uniform.def_immutable_mapping` THIS PASS FOUND — `uniform::DefImmutableMappingOp`, named by
/// the `index` handle it binds.
///
/// ⛔ A TYPE AND NOT A BARE [`Val`]: `DT_CHECK(dimo)` (`:322`) is that the map a `uniform.query_map`
/// reads really is defined by a `uniform.def_immutable_mapping`, and only a caller that matched
/// [`uniform::Op::DefImmutableMapping`] can say so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UniformMap(Val);

impl UniformMap {
    /// The mapping named by the `$result` of a matched `uniform.def_immutable_mapping`.
    #[must_use]
    pub const fn of(def_immutable_mapping_result: Val) -> UniformMap {
        UniformMap(def_immutable_mapping_result)
    }

    /// `dimo.getResult()` — the `index` handle every `uniform.query_map` of this mapping reads.
    #[must_use]
    pub const fn handle(self) -> Val {
        self.0
    }
}

/// Replaces: e062_collectUniformMaps
///
/// APPENDS every `uniform.def_immutable_mapping` that a `sentient.scalar_copy` OF THIS LOCALE reads
/// through a `uniform.query_map`, in pre-order over `local_region`'s ops.
///
/// ⛔ DUPLICATES ARE KEPT — `push_back` with no membership test: two copies reading one mapping put it
/// in twice, which is what `cluster` (e308) counts commoning opportunities from.
/// ⭐ `reg.locale` IS `sentient::getValueRegLocale(copy_op)`: on a `CopyOp` that function is exactly
/// `copy_op.getRegLocale()` (`Dialect/Sentient/SentientOps.cpp:1758`), and a `CopyOp` is all this walk
/// visits. ⭐ `defs` SUPPLIES `getDefiningOp` — a `def_immutable_mapping` ordinarily sits OUTSIDE the
/// local region, so the region alone cannot answer for it.
/// ⚠️ THE REGION ARRIVES AS OPS, NOT AS AN [`OriginalRegion`] — see that type's island-gap note.
pub fn collect_uniform_maps(
    uniform_maps: &mut Vec<UniformMap>,
    local_region: &[Op],
    locale: RegType,
    defs: Definitions<'_>,
) {
    walk_pre_order(local_region, &mut |op| {
        let sentient::Op::ScalarCopy { input, reg, .. } = op else {
            return;
        };
        if reg.locale != locale {
            return;
        }
        let Some(Op::Uniform(uniform::Op::QueryMap { map, .. })) = defs.of(*input) else {
            return;
        };
        match defs.of(*map) {
            Some(Op::Uniform(uniform::Op::DefImmutableMapping { result, .. })) => {
                uniform_maps.push(UniformMap::of(*result));
            }
            // ⛔ `DT_CHECK(dimo)` (`:322`): a `uniform.query_map` whose `$map` is defined by anything
            // else is not a program this dialect builds.
            _ => panic!(
                "a uniform.query_map's $map is not defined by a uniform.def_immutable_mapping \
                 (LocalRegionSplittingForValueCommoning.cpp:322)"
            ),
        }
    });
}

/// `op.walk<WalkOrder::PreOrder>(..)` over one region, nested regions included.
///
/// ⭐ THE REFERENCE'S `for (block) for (op) op.walk(..)` NEST (`:311-313`) FLATTENED: walking every op
/// of the region visits the region's own ops and everything under them, which is what the nest amounts
/// to. Blocks are mechanism this island drops — one region is one op list.
fn walk_pre_order(ops: &[Op], visit: &mut impl FnMut(&sentient::Op)) {
    for op in ops {
        match op {
            Op::Sentient(inner) => {
                visit(inner);
                for region in sentient::regions(inner) {
                    walk_pre_order(region, visit);
                }
            }
            Op::AffineFor(loop_op) => walk_pre_order(&loop_op.body, visit),
            // ⭐ THE GAP [`OriginalRegion`] NAMES IS CLOSED: this variant's regions hold THIS rung's
            // ops, so a `sentient.scalar_copy` inside a local region is visited.
            Op::UniformRegions(regions) => {
                for region in regions.regions() {
                    walk_pre_order(&region.body, visit);
                }
            }
            // ⛔ NO `_` ARM: a new dialect at this rung must be a build error here rather than a
            // subtree this walk quietly skips. None of the rest can hold a `sentient.scalar_copy` —
            // `Op::Uniform`'s own regions are the rung below's.
            Op::Dataflow(_)
            | Op::Agen(_)
            | Op::VectorChain(_)
            | Op::Affine(_)
            | Op::Vector(_)
            | Op::Arith(_)
            | Op::Scf(_)
            | Op::Symbol(_)
            | Op::Uniform(_) => {}
        }
    }
}

/// HOW MANY REGISTERS OF ONE FILE THIS PASS MAY USE — what `getMaxRegNum` returns.
///
/// ⛔ NOT [`sys_arch_spec::regfile::FileDepth`]: two of the three answers are not a file's depth at all —
/// `numLCCRRegisters` (`sysdef.cpp:228`), and the flat 0 an L3 unit gets for a non-address file — and
/// `FileDepth::of` is `pub(crate)` to `sys-arch-spec` precisely so a depth cannot be minted elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MaxRegNum(pub u32);

/// THE PASS'S TWO `cl::opt`s — *"Pretend there are only this number of LBRs/EBRs"* (`:122-129`).
///
/// ⛔ `None` IS `getNumOccurrences() == 0`, NOT ZERO. Both flags are `cl::init(0)` and the switch tests
/// OCCURRENCE (`:442-451`), so `-max-lbr=0` really does cap the file at no registers while an unset
/// flag falls through to the machine's own bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PretendRegLimits {
    /// `-dcc-local-region-splitting-for-value-commoning-max-lbr`.
    pub max_lbr: Option<MaxRegNum>,
    /// `-dcc-local-region-splitting-for-value-commoning-max-ebr`.
    pub max_ebr: Option<MaxRegNum>,
}

/// Replaces: e063_getMaxRegNum
///
/// How many registers of `locale` this pass may use on `comp`: the pretend-limit flag when one was
/// given, else the machine's own bound.
///
/// ⛔ THE FLAG WINS FOR `lbr`/`ebr` ONLY, AND ONLY WHEN GIVEN — every other locale, and both of those
/// unset, fall through the `default: break` to [`machine_max_reg_num`].
#[must_use]
pub fn get_max_reg_num<A: Arch>(
    locale: RegType,
    comp: Component,
    limits: PretendRegLimits,
) -> MaxRegNum {
    match locale {
        RegType::Lbr => {
            if let Some(max_lbr) = limits.max_lbr {
                return max_lbr;
            }
        }
        RegType::Ebr => {
            if let Some(max_ebr) = limits.max_ebr {
                return max_ebr;
            }
        }
        _ => {}
    }
    machine_max_reg_num::<A>(locale, comp)
}

/// `dccExtContext().getMaxRegNum(locale, comp)` — `DccExtContext::getMaxRegNum`
/// (`dcc/src/Utils/DccExtContext.cpp:282-292`).
///
/// ⭐ NOT AN OUT-OF-SCOPE ANALYSIS AND SO NOT A `todo!`: every number it reads is `sysDef`'s, and this
/// crate vendors `sysDef` as [`sys_arch_spec::regfile`] plus [`Arch::LCCR_REGISTERS`].
fn machine_max_reg_num<A: Arch>(locale: RegType, comp: Component) -> MaxRegNum {
    if locale == RegType::Lccr {
        return MaxRegNum(A::LCCR_REGISTERS);
    }
    let file = register_class(locale);
    // ⛔ AN L3 HALF MAY USE THE ADDRESSING FILES AND NOTHING ELSE (`DccExtContext.cpp:289-291`) — it
    // has no compute registers, and 0 here is what makes this pass leave such a locale alone.
    if matches!(comp, Component::L3lu | Component::L3su)
        && !matches!(
            file,
            RegFile::Lar | RegFile::Lbr | RegFile::Ear | RegFile::Ebr | RegFile::Gtr | RegFile::Jcr
        )
    {
        return MaxRegNum(0);
    }
    match depth_of(comp, file) {
        Presence::Present(info) => MaxRegNum(u32::from(info.depth.get())),
        // ⛔ THE REFERENCE'S `regInfoPerUnit.at(comp).at(reg_type)` THROWS HERE, and the caller is what
        // keeps it unreachable: `runOn` asks for `lbr` only below SEN1P5 (`:249-250`), which is the one
        // arch that drops the L3's LBR file outright.
        Presence::Absent => panic!(
            "this arch's {comp:?} has no {file:?} file, so `regInfoPerUnit` states no maxNum for it \
             (DccExtContext.cpp:292)"
        ),
    }
}

/// `localeToRegisterClass` (`dcc/src/Dialect/Sentient/Utils.cpp:203`) then
/// `ProgramAndStateInfo::stringToRegType` (`sys-arch-spec/progir/progir.cpp:674`) — the two string
/// hops between a locale and the register file it names, with the strings dropped.
const fn register_class(locale: RegType) -> RegFile {
    match locale {
        RegType::Lrf => RegFile::Lrf,
        RegType::Lar => RegFile::Lar,
        RegType::Lbr => RegFile::Lbr,
        RegType::Ear => RegFile::Ear,
        RegType::Ebr => RegFile::Ebr,
        RegType::Gtr => RegFile::Gtr,
        RegType::Jcr => RegFile::Jcr,
        RegType::Mvr => RegFile::Mvr,
        // ⭐ BOTH POINTERS NAME THE ONE FILE — the function's only special case (`Utils.cpp:204-205`).
        RegType::XrfRdPtr | RegType::XrfWrPtr => RegFile::Xrf,
        // ⛔ `stringToRegType.at()` THROWS for these four: `unknown`, `imm` and `unrelated` name no
        // register file, and `lccr` is answered before this is reached (`DccExtContext.cpp:284-285`).
        RegType::Lccr | RegType::Unknown | RegType::Imm | RegType::Unrelated => {
            panic!("this locale names no register file (stringToRegType has no entry for it)")
        }
    }
}

/// Replaces: e308_cluster
///
/// APPENDS one group per unit — the fall-back that splits a local region into one region per unit.
///
/// ⛔ ONE GROUP PER UNIT *IS* THE PORT. The reference's own `// todo: implement an actual
/// sophisticated clustering algorithm` (`:340`) sits above this very loop, so there is no algorithm
/// here to omit; `uniform_maps` is read only by the check.
/// ⭐ THE `DT_CHECK_MSG` IS WHAT MAKES `uniform_maps` A PARAMETER: every map captured must be keyed by
/// exactly the units of the local region being split (`:334-337`).
pub fn cluster(
    output: &mut Vec<Vec<Val>>,
    units: &[Val],
    uniform_maps: &[UniformMap],
    defs: Definitions<'_>,
) {
    for map in uniform_maps {
        // `uniform_map.getKeys()` — the `$keys` operands of the `uniform.def_immutable_mapping`.
        let keys = match defs.of(map.handle()) {
            Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) => pairs.len(),
            _ => panic!(
                "a captured uniform map's handle is not defined by a \
                 uniform.def_immutable_mapping (LocalRegionSplittingForValueCommoning.cpp:334)"
            ),
        };
        if keys != units.len() {
            panic!(
                "expected all the maps captured to be from the local region that represents the \
                 list of input units (LocalRegionSplittingForValueCommoning.cpp:334-337)"
            );
        }
    }

    // fall-back: naively splitting every unit into its own local region.
    for unit in units {
        output.push(vec![*unit]);
    }
}

/// WHETHER A SPLIT WAS FOUND — the `bool` `analyze` returns (`:271`), and the only thing deciding
/// whether `run` (e561) goes on to call `transform` (`:266-269`).
///
/// ⛔ NOT A BARE `bool`: `false` is neither failure nor "nothing found" — `ur` is filled EITHER WAY
/// (`:298-302`), so both answers are kinds of success and each has to name itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transformation {
    /// Some region held at least as many uniform maps as the file has registers and was clustered.
    Needed,
    /// Every region fit, and `ur` holds each of them unsplit.
    NotNeeded,
}

/// Replaces: e444_analyze
///
/// Per region of the original op: collect the uniform maps its `locale` copies read, append the region
/// unsplit when they fit `comp`'s register file, else append one local region per cluster.
///
/// ⛔ `<` AND NOT `<=` IS THE REFERENCE'S OWN HEURISTIC (`:295-297`) — being purely local it leaves one
/// register free for a globally shared map.
/// ⛔ EVERY REGION IS APPENDED EITHER WAY (`:298-302`), so a [`Transformation::NotNeeded`] `ur` is a
/// faithful copy of the original op and the answer says only whether ANY region was split.
/// ⭐ THE OP ARRIVES BESIDE `ur`: `ur.original_uro` is an identity ([`uniform_region::UniformizeRegions`])
/// and a region's ops are not reachable from a [`Val`], so e561 holds both.
#[must_use]
pub fn analyze<A: Arch>(
    ur: &mut UniformRegion,
    orig_ur: &UniformRegions,
    locale: RegType,
    comp: Component,
    limits: PretendRegLimits,
    defs: Definitions<'_>,
) -> Transformation {
    let max_reg_num = get_max_reg_num::<A>(locale, comp, limits);
    let UniformRegions::UniformizeRegions { regions, .. } = orig_ur else {
        panic!(
            "analyze takes a uniform.uniformize_regions and nothing else — that is its typed \
             parameter (LocalRegionSplittingForValueCommoning.cpp:271-273)"
        )
    };
    let mut answer = Transformation::NotNeeded;
    for local_region in regions {
        let mut uniform_maps = Vec::new();
        collect_uniform_maps(&mut uniform_maps, &local_region.body, locale, defs);
        // ⭐ `getUnitsOfRegion(orig_ur, region_index)` (`Dialect/Uniform/Utils.cpp:526-546`) IS THIS
        // FIELD: it slices the flat `$units` operand list by `list_sizes`, and the island stores the
        // slice per region, so the slicing is mechanism this port drops.
        if uniform_maps.len() < max_reg_num.0 as usize {
            ur.add_local_region(local_region.units.clone(), OriginalRegion(local_region.arg));
            continue;
        }
        answer = Transformation::Needed;
        let mut clustered_units = Vec::new();
        cluster(&mut clustered_units, &local_region.units, &uniform_maps, defs);
        for unit_group in clustered_units {
            ur.add_local_region(unit_group, OriginalRegion(local_region.arg));
        }
    }
    answer
}

/// THE COMMONING KEY — `std::make_tuple(const value, locale, element size)` (`:404-406`, `:418-420`).
///
/// ⛔ NOT A MAP KEY: [`RegType`] orders nothing and the reference's `DenseMap` is unordered too —
/// *first insertion wins* (`:407`) is the whole of that map's contract, and a `Vec` states it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CopyKey {
    /// `const_op.getValue()`.
    value: i64,
    /// `sentient::getValueRegLocale(copy_op.getResult())`, which on a `CopyOp` is its own
    /// `getRegLocale()` (`Dialect/Sentient/SentientOps.cpp:1758`).
    locale: RegType,
    /// `getAttr(copy_op.getResult(), "element_size")` — `None` is the absent attribute the reference's
    /// `cast<IntegerAttr>` would fail on.
    element_size: Option<Bits>,
}

/// Replaces: e505_transform
///
/// Rebuilds the `uniform.uniformize_regions` with one region per local region [`analyze`] decided on,
/// then commons the constant-fed `sentient.scalar_copy`s of one (value, locale, element size).
///
/// ⛔ `replaceAllUsesWith` (`:376`) IS A NO-OP: the new op reuses the original's `$results` verbatim.
/// ⛔ THE PRUNE (`:379-386`) IS THE UNPORTED `e395_pruneOutOfScopeEntries`, GATED on a map being in the
/// new op — an unconditional `todo!` would put the commoning below out of reach.
pub fn transform(unit_body: &mut Vec<Op>, ur: &UniformRegion, values: &mut Values) {
    let Some(at) = original_op_index(unit_body, ur) else {
        return;
    };
    // ── the new op, region for region with what `analyze` appended (`:349-374`) ──
    let Op::UniformRegions(UniformRegions::UniformizeRegions {
        regions,
        results,
        yielded,
    }) = &unit_body[at]
    else {
        panic!(
            "transform takes a uniform.uniformize_regions and nothing else \
             (LocalRegionSplittingForValueCommoning.cpp:347-360)"
        )
    };
    let (results, yielded) = (results.clone(), yielded.clone());
    let mut new_regions = Vec::with_capacity(ur.local_regions.len());
    for local_region in &ur.local_regions {
        let Some(original) = regions
            .iter()
            .find(|region| region.arg == local_region.original_region.0)
        else {
            panic!(
                "a transformed local region names a region the original op does not have \
                 (LocalRegionSplittingForValueCommoning.cpp:366-368)"
            )
        };
        // `bv_map.map(original.getArgument(0), new_uniform_op.getRegion(i).getArgument(0))` — the one
        // entry the clone is seeded with, and the reason each new region needs its own argument: one
        // original region may be split into several.
        let arg = values.mint();
        let mut mapping = ValueMapping::new();
        mapping.map(original.arg, arg);
        new_regions.push(dialects::LocalRegion {
            arg,
            units: local_region.units.clone(),
            body: dialects::clone_ops(&original.body, values, &mut mapping),
        });
    }
    unit_body[at] = Op::UniformRegions(UniformRegions::UniformizeRegions {
        regions: new_regions,
        results,
        yielded,
    });

    // ⛔ THE GATE: with no `uniform.def_immutable_mapping` in the new op the reference's walk visits
    // nothing and `to_be_deleted` stays empty, so this is the whole of `:379-386` for such an op.
    if holds_uniform_map(core::slice::from_ref(&unit_body[at])) {
        todo!(
            "e395_pruneOutOfScopeEntries (dcc/src/Transform/Sentient/Utils.cpp:635) — homed in \
             transform::sentient::utils, anchor not yet filled"
        )
    }

    // ── commoning for copies that have constant operands, no maps (`:388-412`) ──
    let scope: [&[Op]; 1] = [unit_body.as_slice()];
    let defs = Definitions::from_innermost(&scope);
    let mut copies: Vec<(CopyKey, Val)> = Vec::new();
    for region in dialects::regions_ref(&unit_body[at]) {
        walk_pre_order(region, &mut |op| {
            let sentient::Op::ScalarCopy {
                input,
                result,
                reg,
                element_size,
                ..
            } = op
            else {
                return;
            };
            let Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) = defs.of(*input)
            else {
                return;
            };
            copies.push((
                CopyKey {
                    value: *value,
                    locale: reg.locale,
                    element_size: *element_size,
                },
                *result,
            ));
        });
    }
    // `copy_op_aliases.insert(..)` — `DenseMap::insert` leaves an existing entry alone (`:407`).
    let mut aliases: Vec<(CopyKey, Val)> = Vec::new();
    for (key, result) in &copies {
        if !aliases.iter().any(|(seen, _)| seen == key) {
            aliases.push((*key, *result));
        }
    }

    for (key, result) in copies {
        let Some(alias) = aliases
            .iter()
            .find(|(seen, _)| *seen == key)
            .map(|(_, alias)| *alias)
        else {
            continue;
        };
        // `if (it->getSecond() == copy_op) continue;` (`:425`).
        if alias == result {
            continue;
        }
        // ⭐ THE POSITIONS ARE RE-FOUND EVERY ITERATION: the reference holds `Operation *`s, which
        // survive the erases and moves below, and a path does not.
        let (Some(alias_at), Some(copy_at)) =
            (path_of(unit_body, alias), path_of(unit_body, result))
        else {
            continue;
        };
        // ⭐ THE CLIMB LEAVES THE LOCAL REGION RATHER THAN STOPPING AT IT, and that is `e241`'s
        // `promote_above_uniform_region` (`Transform/Sentient/Utils.cpp:104-112`) answering true: every
        // copy collected above reads a `sentient.scalar_constant`, which is one of its three cases.
        match move_to_common_dominator(unit_body, &alias_at, &NewUse::SameUnit(copy_at)) {
            Hoisted::Done => {}
            // ⛔ UNREACHABLE BY [`NewUse`]'S OWN PROOF: both copies are ops of this unit body.
            Hoisted::OutsideProgramUnit => panic!(
                "failed to move copy op to common dominator \
                 (LocalRegionSplittingForValueCommoning.cpp:429-433)"
            ),
        }
        dialects::replace_all_uses_with(unit_body, result, alias);
        dialects::erase_defining_op(unit_body, result);
    }

    // TODO(reference `:436-437`): commoning of copies that have maps as operands is only needed once
    // `cluster` (e308) does actual clustering.
}

/// `ur.getOriginalOp()` — the `uniform.uniformize_regions` named by the argument of its first region.
///
/// ⭐ THE UNIT BODY'S OWN BLOCK AND NOT THE WHOLE TREE: `UniformizeRegions` builds these ops at the top
/// of a `dataflow.program_unit` and nothing nests one, which is also why `analyze` is handed one op.
fn original_op_index(unit_body: &[Op], ur: &UniformRegion) -> Option<usize> {
    unit_body.iter().position(|op| match op {
        Op::UniformRegions(regions) => regions
            .regions()
            .first()
            .is_some_and(|first| first.arg == ur.original_uro.0),
        _ => false,
    })
}

/// Whether `ops` hold a `uniform.def_immutable_mapping` anywhere under them — the walk at `:381-385`,
/// asked only whether it would visit anything.
fn holds_uniform_map(ops: &[Op]) -> bool {
    ops.iter().any(|op| {
        matches!(op, Op::Uniform(uniform::Op::DefImmutableMapping { .. }))
            || dialects::regions_ref(op).into_iter().any(holds_uniform_map)
    })
}

// crustify:todo: e561_run
//   authority : dcc/src/Transform/Sentient/LocalRegionSplittingForValueCommoning.cpp:252  (19 body lines, level 4)
//   original  : void LocalRegionSplittingForValueCommoningPass::run( const SentientRegType locale, const SenComponents comp)
//   calls     : e064_dump, e065_dump, e444_analyze, e505_transform

// crustify:todo: e599_runOn
//   authority : dcc/src/Transform/Sentient/LocalRegionSplittingForValueCommoning.cpp:240  (11 body lines, level 5)
//   original  : void LocalRegionSplittingForValueCommoningPass::runOn( dataflow::ProgramUnitOp unit)
//   calls     : e561_run

// crustify:todo: e624_runOnOperation
//   authority : dcc/src/Transform/Sentient/LocalRegionSplittingForValueCommoning.cpp:233  (6 body lines, level 6)
//   original  : void LocalRegionSplittingForValueCommoningPass::runOnOperation()
//   calls     : e599_runOn

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::islands::sentient::dialects::LocalRegion as IrLocalRegion;
    use crate::islands::sentient::dialects::sentient::Reg;

    /// `%out = sentient.scalar_copy %input {regLocale = locale}`.
    fn scalar_copy(input: Val, result: Val, locale: RegType) -> Op {
        Op::Sentient(sentient::Op::ScalarCopy {
            input,
            result,
            reg: Reg {
                locale,
                index: None,
            },
            program_header: false,
            element_size: None,
        })
    }

    #[test]
    fn add_local_region_appends_in_call_order() {
        let mut ur = UniformRegion {
            local_regions: Vec::new(),
            original_uro: uniform_region::UniformizeRegions(Val(0)),
        };
        ur.add_local_region(vec![Val(1), Val(2)], OriginalRegion(Val(10)));
        ur.add_local_region(vec![Val(3)], OriginalRegion(Val(20)));
        assert_eq!(ur.local_regions.len(), 2);
        assert_eq!(ur.local_regions[0].units, vec![Val(1), Val(2)]);
        assert_eq!(ur.local_regions[0].original_region, OriginalRegion(Val(10)));
        assert_eq!(ur.local_regions[1].units, vec![Val(3)]);
        assert_eq!(ur.local_regions[1].original_region, OriginalRegion(Val(20)));
    }

    #[test]
    fn collect_uniform_maps_takes_this_locale_through_a_query_map_only() {
        // The mapping and the three queries of it live outside the local region, as they do in the IR.
        let outer = vec![
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(100),
                pairs: vec![(Val(1), Val(2))],
            }),
            Op::Uniform(uniform::Op::QueryMap {
                result: Val(101),
                map: Val(100),
                key: Val(1),
            }),
            Op::Uniform(uniform::Op::QueryMap {
                result: Val(102),
                map: Val(100),
                key: Val(1),
            }),
            Op::Arith(crate::islands::sentient::dialects::arith::Op::Constant {
                result: Val(103),
                value: 7,
            }),
        ];
        let region = vec![
            // Taken: this locale, and its input is a query of the mapping.
            scalar_copy(Val(101), Val(200), RegType::Lbr),
            // Taken again — ⭐ the duplicate is KEPT.
            scalar_copy(Val(102), Val(201), RegType::Lbr),
            // Skipped: the other locale.
            scalar_copy(Val(101), Val(202), RegType::Ebr),
            // Skipped: its input is not a `uniform.query_map`.
            scalar_copy(Val(103), Val(203), RegType::Lbr),
            // Skipped: its input is a block argument, so it has no defining op at all.
            scalar_copy(Val(999), Val(204), RegType::Lbr),
        ];
        let mut maps = vec![UniformMap::of(Val(77))];
        collect_uniform_maps(
            &mut maps,
            &region,
            RegType::Lbr,
            Definitions::from_innermost(&[&region, &outer]),
        );
        // ⛔ APPENDED, not assigned: the pre-seeded entry survives.
        assert_eq!(
            maps,
            vec![
                UniformMap::of(Val(77)),
                UniformMap::of(Val(100)),
                UniformMap::of(Val(100)),
            ]
        );
    }

    #[test]
    fn get_max_reg_num_prefers_a_given_flag_and_otherwise_asks_the_machine() {
        let capped = PretendRegLimits {
            max_lbr: Some(MaxRegNum(3)),
            max_ebr: Some(MaxRegNum(0)),
        };
        // The flag wins for both locales it exists for — including `-max-ebr=0`, which really is 0.
        assert_eq!(
            get_max_reg_num::<Dd2>(RegType::Lbr, Component::L3lu, capped),
            MaxRegNum(3)
        );
        assert_eq!(
            get_max_reg_num::<Dd2>(RegType::Ebr, Component::L3lu, capped),
            MaxRegNum(0)
        );
        // Unset falls through to the machine: `numLCCRRegisters`, and 0 for a file an L3 half lacks.
        let unset = PretendRegLimits::default();
        assert_eq!(
            get_max_reg_num::<Dd2>(RegType::Lccr, Component::L3su, unset),
            MaxRegNum(Dd2::LCCR_REGISTERS)
        );
        assert_eq!(
            get_max_reg_num::<Dd2>(RegType::Lrf, Component::L3su, unset),
            MaxRegNum(0)
        );
        // ⭐ AND THE DEPTH ITSELF FOR AN ADDRESSING FILE THE UNIT HAS — RCUDD1A's L3LU LBR is 8 deep.
        #[cfg(feature = "arch-rcudd1a")]
        assert_eq!(
            get_max_reg_num::<Dd2>(RegType::Lbr, Component::L3lu, unset),
            MaxRegNum(8)
        );
    }

    /// e308 — every unit becomes its own group, APPENDED to whatever the caller already had.
    #[test]
    fn cluster_gives_every_unit_its_own_group() {
        let outer = vec![Op::Uniform(uniform::Op::DefImmutableMapping {
            result: Val(100),
            pairs: vec![(Val(1), Val(11)), (Val(2), Val(12)), (Val(3), Val(13))],
        })];
        let units = [Val(1), Val(2), Val(3)];
        let mut output = vec![vec![Val(9)]];
        cluster(
            &mut output,
            &units,
            &[UniformMap::of(Val(100))],
            Definitions::from_innermost(&[&outer]),
        );
        assert_eq!(
            output,
            vec![vec![Val(9)], vec![Val(1)], vec![Val(2)], vec![Val(3)]]
        );
    }

    /// e444 — one region under the pretend limit is kept whole, the next one at it is split per unit,
    /// and both answer with the SAME original region.
    #[test]
    fn analyze_keeps_a_fitting_region_whole_and_clusters_the_one_that_does_not_fit() {
        let outer = vec![
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(100),
                pairs: vec![(Val(1), Val(11)), (Val(2), Val(12))],
            }),
            Op::Uniform(uniform::Op::QueryMap {
                result: Val(101),
                map: Val(100),
                key: Val(1),
            }),
        ];
        let orig_ur = UniformRegions::UniformizeRegions {
            regions: vec![
                IrLocalRegion {
                    arg: Val(50),
                    units: vec![Val(3)],
                    body: vec![Op::Sentient(sentient::Op::Nop { dbg_name: None })],
                },
                IrLocalRegion {
                    arg: Val(51),
                    units: vec![Val(1), Val(2)],
                    body: vec![scalar_copy(Val(101), Val(200), RegType::Ebr)],
                },
            ],
            results: Vec::new(),
            yielded: Vec::new(),
        };
        let mut ur = UniformRegion {
            local_regions: Vec::new(),
            original_uro: uniform_region::UniformizeRegions(Val(50)),
        };
        let scopes: [&[Op]; 1] = [&outer];
        assert_eq!(
            analyze::<Dd2>(
                &mut ur,
                &orig_ur,
                RegType::Ebr,
                Component::L3lu,
                PretendRegLimits {
                    max_lbr: None,
                    max_ebr: Some(MaxRegNum(1)),
                },
                Definitions::from_innermost(&scopes),
            ),
            Transformation::Needed
        );
        // Region 0 found no map, so `0 < 1` kept it whole; region 1 found one, so `1 < 1` split it.
        assert_eq!(ur.local_regions.len(), 3);
        assert_eq!(ur.local_regions[0].units, vec![Val(3)]);
        assert_eq!(ur.local_regions[0].original_region, OriginalRegion(Val(50)));
        assert_eq!(ur.local_regions[1].units, vec![Val(1)]);
        assert_eq!(ur.local_regions[2].units, vec![Val(2)]);
        assert_eq!(ur.local_regions[1].original_region, OriginalRegion(Val(51)));
        assert_eq!(ur.local_regions[2].original_region, OriginalRegion(Val(51)));
    }
    /// e505 — two local regions each copying the SAME constant into the same locale: the second copy
    /// goes, its reader is repointed at the first, and the first is hoisted above the uniform op,
    /// which is the only block that dominates both regions.
    #[test]
    fn transform_commons_a_constant_copy_across_two_local_regions() {
        let constant = Op::Sentient(sentient::Op::ScalarConstant {
            value: 7,
            result: Val(1),
            reg_locale: RegType::Imm,
            ty: crate::islands::dataflow_ir::ty::ScalarTy::Index,
            is_symbol: false,
        });
        let mut unit_body = vec![
            constant,
            Op::UniformRegions(UniformRegions::UniformizeRegions {
                regions: vec![
                    IrLocalRegion {
                        arg: Val(10),
                        units: vec![],
                        body: vec![scalar_copy(Val(1), Val(11), RegType::Ebr)],
                    },
                    IrLocalRegion {
                        arg: Val(20),
                        units: vec![],
                        body: vec![
                            scalar_copy(Val(1), Val(21), RegType::Ebr),
                            scalar_copy(Val(21), Val(22), RegType::Ebr),
                        ],
                    },
                ],
                results: Vec::new(),
                yielded: Vec::new(),
            }),
        ];
        let ur = UniformRegion {
            local_regions: vec![
                LocalRegion {
                    units: Vec::new(),
                    original_region: OriginalRegion(Val(10)),
                },
                LocalRegion {
                    units: Vec::new(),
                    original_region: OriginalRegion(Val(20)),
                },
            ],
            original_uro: uniform_region::UniformizeRegions(Val(10)),
        };
        let mut values = Values::default();
        for _ in 0..30 {
            values.mint();
        }
        transform(&mut unit_body, &ur, &mut values);

        // The survivor sits between the constant and the uniform op.
        assert_eq!(unit_body.len(), 3);
        let Op::Sentient(sentient::Op::ScalarCopy {
            input,
            result: survivor,
            ..
        }) = unit_body[1]
        else {
            panic!("the commoned copy was hoisted out of its local region")
        };
        assert_eq!(input, Val(1));
        let Op::UniformRegions(new_op) = &unit_body[2] else {
            panic!("the rebuilt uniform.uniformize_regions is the last op")
        };
        // Region 0 gave up its copy to the hoist; region 1 kept only the reader, now on the survivor.
        assert!(new_op.regions()[0].body.is_empty());
        assert_eq!(new_op.regions()[1].body.len(), 1);
        let Op::Sentient(sentient::Op::ScalarCopy { input: read, .. }) =
            new_op.regions()[1].body[0]
        else {
            panic!("the reader of the erased copy survives")
        };
        assert_eq!(read, survivor);
    }
}

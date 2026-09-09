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
use crate::islands::sentient::dialects::{Definitions, Op, Val, sentient, uniform};
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
/// ⭐ THE REFERENCE'S `for (block) for (op) op.walk(..)` NEST (`:310-312`) FLATTENED: walking every op
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
            // ⛔ NO `_` ARM: a new dialect at this rung must be a build error here rather than a
            // subtree this walk quietly skips. None of these can hold a `sentient.scalar_copy` —
            // `Op::Uniform`'s own regions are the rung below's, the gap [`OriginalRegion`] names.
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

// crustify:todo: e308_cluster
//   authority : dcc/src/Transform/Sentient/LocalRegionSplittingForValueCommoning.cpp:330  (14 body lines, level 1)
//   original  : void LocalRegionSplittingForValueCommoningPass::cluster( GroupedValuesTy &output, const ArrayRef<Value> units, ListOfUniformMapsTy &uniform_maps) const
//   calls     : e252_size

// crustify:todo: e444_analyze
//   authority : dcc/src/Transform/Sentient/LocalRegionSplittingForValueCommoning.cpp:273  (32 body lines, level 2)
//   original  : bool LocalRegionSplittingForValueCommoningPass::analyze( lrs::UniformRegion &ur, const SentientRegType locale, const SenComponents comp)
//   calls     : e061_addLocalRegion, e062_collectUniformMaps, e063_getMaxRegNum, e252_size, e308_cluster

// crustify:todo: e505_transform
//   authority : dcc/src/Transform/Sentient/LocalRegionSplittingForValueCommoning.cpp:347  (91 body lines, level 3)
//   original  : void LocalRegionSplittingForValueCommoningPass::transform( const lrs::UniformRegion &ur)
//   calls     : e252_size, e395_pruneOutOfScopeEntries, e422_insert

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
}

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

//! `SinkScalarCopy.cpp` — 12 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e209_region_` | 209 | 0 | 3 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:89` |
//! | `e210_isUniformLocalRegion` | 210 | 0 | 4 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:94` |
//! | `e211_dump` | 211 | 0 | 5 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:100` |
//! | `e212_op_` | 212 | 0 | 3 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:116` |
//! | `e213_addRegion` | 213 | 0 | 12 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:125` |
//! | `e214_sinkCopyOps` | 214 | 0 | 16 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:215` |
//! | `e215_clear` | 215 | 0 | 5 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:232` |
//! | `e381_dump` | 381 | 1 | 7 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:144` |
//! | `e477_runOn` | 477 | 2 | 25 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:189` |
//! | `e538_runOn` | 538 | 3 | 7 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:176` |
//! | `e539_runOn` | 539 | 3 | 4 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:184` |
//! | `e584_runOnOperation` | 584 | 4 | 7 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:167` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so every item below is reachable only from this
// file's own tests until `e584_runOnOperation` (level 4) lands and something calls it. CI runs clippy
// with `-D warnings`, so without this the first ported leaf of a 12-unit module fails the gate.
// ⭐ REMOVE THIS WITH e584: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use std::fmt::Write as _;

use crate::islands::dataflow_ir::Values;
use crate::islands::sentient::dialects::{self as dialects, Op, Val, sentient};
use crate::islands::sentient::print;
use crate::transform::sentient::local_region_splitting_for_value_commoning::local_region::Indent;

/// WHICH REGION OF ITS OWNER — `Region::getRegionNumber()` (`:102`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RegionNumber(pub u32);

/// ONE LOCAL REGION'S IDENTITY — its `$arg`, which no other region of the unit binds.
///
/// ⛔ AN IDENTITY, NOT A BORROW: the `Region *` this stands for outlives every mutation
/// [`SinkScalarCopy::sink_copy_ops`] makes, and a `&mut` could not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LocalRegionArg(pub Val);

/// THE `Region *` A USE WAS FOUND IN — the two kinds `e477_runOn`'s `DT_CHECK` admits (`:204-206`),
/// which is why "is it a local one" is a match and not a lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Region {
    /// `dataflow.program_unit`'s own region — *"a global region is the program_unit region"* (`:84`).
    Global,
    /// One region of a [`Op::UniformRegions`], owned by a `uniform.uniformize_regions` or a
    /// `uniform.equalize_pattern`.
    UniformLocal {
        /// Which region it is, in its owner's region list.
        number: RegionNumber,
        /// The `$arg` it binds — see [`LocalRegionArg`].
        arg: LocalRegionArg,
    },
}

/// `RegionInfo` (`:87-110`) — a local or global region holding a use of the copy's result. The use
/// itself may be nested deeper inside it (`:85-86`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RegionInfo {
    region: Region,
}

impl RegionInfo {
    /// Replaces: e209_region_
    ///
    /// `RegionInfo(Region *region)`.
    ///
    /// ⭐ `DT_CHECK_MSG(region_, "expected valid region pointer")` IS DISCHARGED BY THE TYPE: there is
    /// no null [`Region`] to hold, and `RegionInfo() = delete` (`:92`) is `#[derive]`-free —
    /// no `Default`.
    #[must_use]
    pub fn new(region: Region) -> RegionInfo {
        RegionInfo { region }
    }

    /// `getPointer()` (`:106`) — the identity this was built from.
    #[must_use]
    pub fn region(&self) -> Region {
        self.region
    }

    /// Replaces: e210_isUniformLocalRegion
    ///
    /// Whether a uniformization construct owns the region — `isa<uniform::UniformizeRegionsOp,
    /// uniform::EqualizePatternOp>(region_->getParentOp())`.
    ///
    /// ⭐ THE TWO OWNERS ARE ONE ISLAND VARIANT: [`Op::UniformRegions`] is both of them, so the
    /// `isa<>` pair is this match arm and cannot go stale against a third owner.
    #[must_use]
    pub fn is_uniform_local_region(&self) -> bool {
        matches!(self.region, Region::UniformLocal { .. })
    }

    /// `isGlobalRegion()` (`:98`) — *"a global region is the program_unit region"*.
    #[must_use]
    pub fn is_global_region(&self) -> bool {
        !self.is_uniform_local_region()
    }

    /// Replaces: e211_dump
    ///
    /// The one line a region contributes to [`UsesInfo`]'s dump: its region number and its parent.
    ///
    /// ⚠️ THE PARENT'S SOURCE LINE IS NOT AVAILABLE AND IS NOT INVENTED: `getLocation(..).getLine()`
    /// reads an `mlir::FileLineColLoc` and this island is emitted rather than parsed, so the parent is
    /// named by what does identify it here — the region's `$arg`, or `dataflow.program_unit`.
    /// ⭐ A RETURNED `String`, as e014_dump and e064_dump are: the caller wraps it in `LLVM_DEBUG`.
    #[must_use]
    pub fn dump(&self, indent: Indent) -> String {
        let mut out = " ".repeat(indent.0);
        match self.region {
            Region::Global => {
                let _ = writeln!(out, "region 0 of parent dataflow.program_unit");
            }
            Region::UniformLocal { number, arg } => {
                let _ = writeln!(
                    out,
                    "region {} of parent binding {}",
                    number.0,
                    print::val(arg.0)
                );
            }
        }
        out
    }
}

/// THE `sentient.scalar_copy` A `UsesInfo` IS ABOUT, NAMED BY ITS `$out`.
///
/// ⭐ THE VALUE NAMES THE OP because `DT_CHECK(orig_op.getNumResults() == 1)` (`:224`) is the shape of
/// a `scalar_copy` and not a runtime question — and an identity survives the erase at `:228`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopyResult(pub Val);

/// `UsesInfo` (`:114-158`) — one copy op and the unique regions its result is used in.
#[derive(Debug)]
pub struct UsesInfo {
    /// `regions_` — kept sorted and unique by [`UsesInfo::add_region`].
    regions: Vec<RegionInfo>,
    /// `op_`, the op defining the SSA value.
    op: CopyResult,
}

impl UsesInfo {
    /// Replaces: e212_op_
    ///
    /// `UsesInfo(Operation *op)`.
    ///
    /// ⭐ `DT_CHECK_MSG(op, "expected valid operation")` IS DISCHARGED BY THE TYPE, as e209's is; the
    /// deleted default and copy constructors (`:119-120`) are the absence of `Default` and `Clone`.
    #[must_use]
    pub fn new(op: CopyResult) -> UsesInfo {
        UsesInfo {
            regions: Vec::new(),
            op,
        }
    }

    /// `getRegions()` (`:139`) — the unique regions the uses appear in.
    #[must_use]
    pub fn regions(&self) -> &[RegionInfo] {
        &self.regions
    }

    /// `getOperation()` (`:142`) — the copy op defining the SSA value.
    #[must_use]
    pub fn operation(&self) -> CopyResult {
        self.op
    }

    /// Replaces: e213_addRegion
    ///
    /// Adds `region` unless it is already there — the `lower_bound`-then-`sort` is one sorted set.
    ///
    /// ⚠️ ORDERED BY `(is-local, region number, $arg)` WHERE THE REFERENCE ORDERS BY `Region *`, and
    /// the order is observable: [`SinkScalarCopy::sink_copy_ops`] mints its clones in it. The derived
    /// `Ord` compares [`Region`]'s VARIANT first, so every [`Region::Global`] sorts ahead of every
    /// [`Region::UniformLocal`] — and sink skips the global ones anyway. Within one owner the two
    /// agree, region number BEING that order; across owners the reference's is allocation order.
    pub fn add_region(&mut self, region: RegionInfo) {
        let at = self.regions.partition_point(|each| *each < region);
        if self.regions.get(at) == Some(&region) {
            return;
        }
        self.regions.insert(at, region);
    }
}

/// `SinkScalarCopyPass`'s own state (`:160-246`).
#[derive(Debug, Default)]
pub struct SinkScalarCopy {
    /// `std::vector<UsesInfo *> uses_info_list_` (`:245`) — owned outright, so `delete` in
    /// [`SinkScalarCopy::clear`] is `Vec::clear`.
    uses_info_list: Vec<UsesInfo>,
}

impl SinkScalarCopy {
    /// Replaces: e214_sinkCopyOps
    ///
    /// THE PASS'S WHOLE EFFECT: a clone of each collected `sentient.scalar_copy` at the START of every
    /// LOCAL region that used it, that region's uses moved onto the clone, the original erased if unread.
    ///
    /// ⭐ `OpBuilder builder(region.getPointer())` INSERTS AT THE START of the region's entry block —
    /// why the sunk copy precedes its user in the reference's own example (`:33-43`).
    /// ⭐ A GLOBAL REGION IS SKIPPED (`:221`); a use left there keeps the original (`getUses()`).
    pub fn sink_copy_ops(&self, body: &mut Vec<Op>, values: &mut Values) {
        for uses in &self.uses_info_list {
            let of = uses.operation();
            for region in uses.regions() {
                let Region::UniformLocal { arg, .. } = region.region() else {
                    continue;
                };
                // ⭐ ONLY A `sentient.scalar_copy` CAN MATCH, because only `sentient::CopyOp` is
                // collected (`:178-179`); a miss is a copy already gone, so there is nothing to sink.
                let Some(Op::Sentient(sentient::Op::ScalarCopy {
                    input,
                    reg,
                    element_size,
                    program_header,
                    ..
                })) = dialects::defining_op(of.0, body)
                else {
                    continue;
                };
                let (input, reg, element_size, program_header) =
                    (*input, *reg, *element_size, *program_header);
                let result = values.mint();
                let clone = Op::Sentient(sentient::Op::ScalarCopy {
                    input,
                    result,
                    reg,
                    element_size,
                    program_header,
                });
                let Some(local) = local_region_body_mut(body, arg) else {
                    continue;
                };
                local.insert(0, clone);
                dialects::replace_all_uses_with(local, of.0, result);
            }
            if dialects::use_count(of.0, body) == 0 {
                dialects::erase_defining_op(body, of.0);
            }
        }
    }

    /// Replaces: e215_clear
    ///
    /// Drops every collected [`UsesInfo`] — the reference's `delete` loop, which Rust ownership does,
    /// and then `uses_info_list_.clear()`.
    pub fn clear(&mut self) {
        self.uses_info_list.clear();
    }

    /// The list `e477_runOn` fills and [`SinkScalarCopy::sink_copy_ops`] drains.
    pub fn push(&mut self, uses: UsesInfo) {
        self.uses_info_list.push(uses);
    }

    /// What [`SinkScalarCopy::clear`] empties.
    #[must_use]
    pub fn uses_info_list(&self) -> &[UsesInfo] {
        &self.uses_info_list
    }
}

/// `region.getPointer()` AS A PLACE TO WRITE — the body of the local region binding `arg`.
///
/// ⭐ NESTED LOCAL REGIONS ARE REACHED: a `uniform.uniformize_regions` sits inside another one's
/// region in `dcc/test/Transform/FlatteningLocalRegions/flatten_local_region.mlir:90-93`, and
/// [`dialects::regions_mut`] descends through both.
fn local_region_body_mut(scope: &mut [Op], arg: LocalRegionArg) -> Option<&mut Vec<Op>> {
    for op in scope.iter_mut() {
        if let Op::UniformRegions(uniform) = op {
            if let Some(at) = uniform
                .regions()
                .iter()
                .position(|region| region.arg == arg.0)
            {
                return Some(&mut uniform.regions_mut()[at].body);
            }
            for region in uniform.regions_mut() {
                if let Some(found) = local_region_body_mut(&mut region.body, arg) {
                    return Some(found);
                }
            }
            continue;
        }
        for region in dialects::regions_mut(op) {
            if let Some(found) = local_region_body_mut(region, arg) {
                return Some(found);
            }
        }
    }
    None
}

// crustify:todo: e381_dump
//   authority : dcc/src/Transform/Sentient/SinkScalarCopy.cpp:144  (7 body lines, level 1)
//   original  : void dump(raw_ostream &os, int indent = 0) const
//   calls     : e211_dump

// crustify:todo: e477_runOn
//   authority : dcc/src/Transform/Sentient/SinkScalarCopy.cpp:189  (25 body lines, level 2)
//   original  : void runOn(sentient::CopyOp copy_op)
//   calls     : e211_dump, e213_addRegion, e381_dump

// crustify:todo: e538_runOn
//   authority : dcc/src/Transform/Sentient/SinkScalarCopy.cpp:176  (7 body lines, level 3)
//   original  : void runOn(dataflow::ProgramUnitOp unit)
//   calls     : e214_sinkCopyOps, e215_clear, e477_runOn, e539_runOn

// crustify:todo: e539_runOn
//   authority : dcc/src/Transform/Sentient/SinkScalarCopy.cpp:184  (4 body lines, level 3)
//   original  : void runOn(ModuleOp module_op)
//   calls     : e477_runOn, e538_runOn

// crustify:todo: e584_runOnOperation
//   authority : dcc/src/Transform/Sentient/SinkScalarCopy.cpp:167  (7 body lines, level 4)
//   original  : void runOnOperation()
//   calls     : e477_runOn, e538_runOn, e539_runOn

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::{LocalRegion, UniformRegions};

    /// `%out = scalar_copy %in {locale=ebr}`.
    fn copy(input: Val, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarCopy {
            input,
            result,
            reg: sentient::Reg {
                locale: sentient::RegType::Ebr,
                index: None,
            },
            element_size: None,
            program_header: false,
        })
    }

    /// An op reading `lhs` inside a region — the `.. %10` of the reference's example (`:16`).
    fn add(lhs: Val, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarAdd {
            lhs,
            rhs: lhs,
            result,
            reg: None,
            element_size: None,
            ty: ScalarTy::Index,
        })
    }

    #[test]
    fn a_region_is_local_only_when_a_uniformization_construct_owns_it() {
        let local = RegionInfo::new(Region::UniformLocal {
            number: RegionNumber(1),
            arg: LocalRegionArg(Val(7)),
        });
        assert!(local.is_uniform_local_region());
        assert!(!local.is_global_region());
        assert_eq!(local.dump(Indent(1)), " region 1 of parent binding %7\n");

        let global = RegionInfo::new(Region::Global);
        assert!(!global.is_uniform_local_region());
        assert!(global.is_global_region());
        assert_eq!(
            global.dump(Indent(0)),
            "region 0 of parent dataflow.program_unit\n"
        );
    }

    #[test]
    fn add_region_is_a_sorted_set() {
        let region = |arg: u32| {
            RegionInfo::new(Region::UniformLocal {
                number: RegionNumber(0),
                arg: LocalRegionArg(Val(arg)),
            })
        };
        let mut uses = UsesInfo::new(CopyResult(Val(10)));
        assert_eq!(uses.operation(), CopyResult(Val(10)));
        uses.add_region(region(5));
        uses.add_region(region(3));
        // The second add of a region already there is ignored (`:130`).
        uses.add_region(region(5));
        uses.add_region(RegionInfo::new(Region::Global));
        assert_eq!(
            uses.regions(),
            [RegionInfo::new(Region::Global), region(3), region(5)]
        );
    }

    /// The reference's own example (`SinkScalarCopy.cpp:12-43`): two copies in the global region, each
    /// used in one local region, both sunk and both originals erased.
    #[test]
    fn sink_copy_ops_sinks_each_copy_into_the_local_region_that_uses_it() {
        let (in0, in1) = (Val(0), Val(1));
        let (copy0, copy1) = (Val(10), Val(11));
        let (arg0, arg1) = (Val(20), Val(21));
        let mut values = Values::default();
        for _ in 0..30 {
            let _ = values.mint();
        }
        let mut body = vec![
            copy(in0, copy0),
            copy(in1, copy1),
            Op::UniformRegions(UniformRegions::UniformizeRegions {
                regions: vec![
                    LocalRegion {
                        arg: arg0,
                        units: vec![Val(2)],
                        body: vec![add(copy0, Val(12))],
                    },
                    LocalRegion {
                        arg: arg1,
                        units: vec![Val(3)],
                        body: vec![add(copy1, Val(13))],
                    },
                ],
                results: Vec::new(),
                yielded: Vec::new(),
            }),
        ];

        let mut pass = SinkScalarCopy::default();
        for (result, arg) in [(copy0, arg0), (copy1, arg1)] {
            let mut uses = UsesInfo::new(CopyResult(result));
            uses.add_region(RegionInfo::new(Region::UniformLocal {
                number: RegionNumber(0),
                arg: LocalRegionArg(arg),
            }));
            pass.push(uses);
        }
        pass.sink_copy_ops(&mut body, &mut values);

        // Both originals are gone from the global region: the uniformize op is all that is left.
        assert_eq!(body.len(), 1);
        let Op::UniformRegions(regions) = &body[0] else {
            panic!("the uniformize op survives");
        };
        // Each region opens with its own sunk copy, reading the same `$inp` as the original, and its
        // user now reads the clone.
        for (region, input) in regions.regions().iter().zip([in0, in1]) {
            let [
                Op::Sentient(sentient::Op::ScalarCopy {
                    input: sunk,
                    result,
                    ..
                }),
                Op::Sentient(sentient::Op::ScalarAdd { lhs, .. }),
            ] = region.body.as_slice()
            else {
                panic!("a sunk copy then its user");
            };
            assert_eq!(*sunk, input);
            assert_eq!(lhs, result);
        }
    }

    /// A copy still read from the global region is NOT erased — `getUses().empty()` (`:228`).
    #[test]
    fn a_copy_still_used_globally_survives() {
        let mut values = Values::default();
        let mut body = vec![
            copy(Val(0), Val(10)),
            add(Val(10), Val(11)),
            Op::UniformRegions(UniformRegions::UniformizeRegions {
                regions: vec![LocalRegion {
                    arg: Val(20),
                    units: vec![Val(2)],
                    body: vec![add(Val(10), Val(12))],
                }],
                results: Vec::new(),
                yielded: Vec::new(),
            }),
        ];
        let mut pass = SinkScalarCopy::default();
        let mut uses = UsesInfo::new(CopyResult(Val(10)));
        uses.add_region(RegionInfo::new(Region::Global));
        uses.add_region(RegionInfo::new(Region::UniformLocal {
            number: RegionNumber(0),
            arg: LocalRegionArg(Val(20)),
        }));
        pass.push(uses);
        pass.sink_copy_ops(&mut body, &mut values);
        assert!(matches!(
            body[0],
            Op::Sentient(sentient::Op::ScalarCopy {
                result: Val(10),
                ..
            })
        ));
    }

    #[test]
    fn clear_drops_every_uses_info() {
        let mut pass = SinkScalarCopy::default();
        pass.push(UsesInfo::new(CopyResult(Val(10))));
        pass.push(UsesInfo::new(CopyResult(Val(11))));
        assert_eq!(pass.uses_info_list().len(), 2);
        pass.clear();
        assert!(pass.uses_info_list().is_empty());
    }
}

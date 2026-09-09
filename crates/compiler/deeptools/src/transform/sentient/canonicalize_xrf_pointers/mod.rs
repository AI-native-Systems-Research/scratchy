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

//! `CanonicalizeXRFPointers.cpp` — 3 of the campaign's 656 units (dependency level(s) [0, 1]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e037_replaceXRFImplicitIncrWithAdd` | 037 | 0 | 57 | `dcc/src/Transform/Sentient/CanonicalizeXRFPointers.cpp:78` |
//! | `e038_replaceConstXRFExpressions` | 038 | 0 | 33 | `dcc/src/Transform/Sentient/CanonicalizeXRFPointers.cpp:137` |
//! | `e296_runOnOperation` | 296 | 1 | 33 | `dcc/src/Transform/Sentient/CanonicalizeXRFPointers.cpp:172` |

use crate::arch::Arch;
use crate::bridges::dataflow_ir_to_sentient::vc_lowering_xrf::{
    MacXrfIncrements, xrf_rd_ptr_incr_val_after_mac,
};
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::ProgramUnit;
use crate::islands::sentient::dialects::{self, Op, Val, sentient};

/// WHICH SIDE OF A MARKER'S DEFINITION AN OP GOES ON — the two `OpBuilder` insertion points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Beside {
    /// `OpBuilder builder(owner)` — immediately before it.
    Before,
    /// `builder.setInsertionPointAfter(op)` — immediately after it.
    After,
}

/// PUT `op` BESIDE WHATEVER DEFINES `marker`, ANYWHERE IN THE NEST.
///
/// ⭐ POSITIONING A BUILDER IS MECHANISM, WHICH IS THE ONE THING A PORT MAY REPLACE. MLIR's builder
/// holds a block iterator; here the marker's own definition is that iterator.
///
/// ⛔ IT HANDS `op` BACK RATHER THAN DROPPING IT when nothing in the nest binds `marker`, so the
/// caller says what an unplaceable op means. A `Vec::push` fallback would silently move an add out
/// of its mac's block, which for this pass is a wrong program rather than a missing one.
fn insert_beside(body: &mut Vec<Op>, marker: Val, side: Beside, op: Op) -> Option<Op> {
    if let Some(at) = body
        .iter()
        .position(|each| dialects::results(each).contains(&marker))
    {
        body.insert(
            match side {
                Beside::Before => at,
                Beside::After => at + 1,
            },
            op,
        );
        return None;
    }
    let mut unplaced = Some(op);
    for each in body.iter_mut() {
        let Some(op) = unplaced.take() else { break };
        unplaced = match each {
            Op::Sentient(inner) => {
                let mut carried = Some(op);
                for region in sentient::regions_mut(inner) {
                    let Some(op) = carried.take() else { break };
                    carried = insert_beside(region, marker, side, op);
                }
                carried
            }
            Op::AffineFor(loop_op) => insert_beside(&mut loop_op.body, marker, side, op),
            Op::UniformRegions(regions) => {
                let mut carried = Some(op);
                for region in regions.regions_mut() {
                    let Some(op) = carried.take() else { break };
                    carried = insert_beside(&mut region.body, marker, side, op);
                }
                carried
            }
            // ⛔ NO `_` ARM. None of these holds a region of THIS rung's ops — every shared
            // dialect's region is a `Vec<dataflow_ir::dialects::Op>` — so a `sentient.vector_mac`
            // or a `sentient.scalar_add` cannot be defined inside one.
            Op::Dataflow(_)
            | Op::Agen(_)
            | Op::VectorChain(_)
            | Op::Affine(_)
            | Op::Vector(_)
            | Op::Arith(_)
            | Op::Scf(_)
            | Op::Symbol(_)
            | Op::Uniform(_) => Some(op),
        };
    }
    unplaced
}

/// ONE MAC'S IMPLICIT XRF INCREMENT MADE EXPLICIT — the constant and the add that carry it.
///
/// ⭐ PLANNED BEFORE ANYTHING IS INSERTED, because `replaceUsesWithIf(.., owner != update)` is what
/// keeps the add's own operand pointing at the mac: rewiring while the add is still outside the
/// block exempts it by construction, with no owner test to get wrong.
#[derive(Debug)]
struct XrfIncrAdd {
    /// The mac result every OTHER reader is rewired to the add.
    mac_result: Val,
    /// `sentient.scalar_constant` — the difference, built by the unit-anchored `const_builder`.
    constant: Op,
    /// `sentient.scalar_add`, built immediately after the mac.
    add: Op,
    /// What the add binds.
    add_result: Val,
}

/// The constant/add pair one increment becomes, minted in the reference's creation order.
fn xrf_incr_add(
    values: &mut Values,
    mac_result: Val,
    difference: i64,
    locale: sentient::RegType,
) -> XrfIncrAdd {
    let constant_result = values.mint();
    let add_result = values.mint();
    XrfIncrAdd {
        mac_result,
        constant: Op::Sentient(sentient::Op::ScalarConstant {
            value: difference,
            result: constant_result,
            // `ConstantOp`'s own default, which the reference never overrides.
            reg_locale: sentient::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        }),
        add: Op::Sentient(sentient::Op::ScalarAdd {
            lhs: mac_result,
            rhs: constant_result,
            result: add_result,
            // `setAttr("regLocale", ..)`, and `regIndex` stays the `.td`'s -1.
            reg: Some(sentient::Reg {
                locale,
                index: None,
            }),
            element_size: None,
            ty: ScalarTy::Index,
        }),
        add_result,
    }
}

/// One mac's turn of the walk body — the two `if` branches, in the reference's order.
fn plan_one_mac<A: Arch>(op: &mut sentient::Op, values: &mut Values, plan: &mut Vec<XrfIncrAdd>) {
    // `walk<MacOp>` never calls the body for anything else, and the two predicates are answered in
    // the one place the dialect answers them.
    let Some(incr) = MacXrfIncrements::of(op) else {
        return;
    };
    let (wt_related, rd_related) = (incr.wt_related(), incr.rd_related());
    if !wt_related && !rd_related {
        return;
    }
    if let sentient::Op::VectorMac {
        results,
        unroll_factor,
        compute_precision,
        xrf_read_incr,
        xrf_write_incr,
        ..
    } = op
    {
        // `DT_CHECK(mac_op->getNumResults() == 2)`, which both branches make.
        let &[wt_result, rd_result] = &results[..] else {
            todo!(
                "replaceXRFImplicitIncrWithAdd: DT_CHECK(getNumResults() == 2) on an xrf-related \
                 mac binding {} (CanonicalizeXRFPointers.cpp:83)",
                results.len()
            )
        };
        // `int unroll_factor = mac_op.getUnrollFactorVal();`
        let unroll = i64::from(unroll_factor.count().cast_signed());
        if wt_related && unroll != i64::from(xrf_write_incr.cast_signed()) {
            plan.push(xrf_incr_add(
                values,
                wt_result,
                i64::from(xrf_write_incr.cast_signed()) - unroll,
                sentient::RegType::XrfWrPtr,
            ));
            *xrf_write_incr = unroll_factor.count();
        }
        // `dccExtContext().getXrfRdPtrIncrValAfterMAC(precision)` — the arch is the type parameter.
        let required = xrf_rd_ptr_incr_val_after_mac::<A>(*compute_precision);
        if rd_related && required.cast_signed() != xrf_read_incr.cast_signed() {
            plan.push(xrf_incr_add(
                values,
                rd_result,
                i64::from(xrf_read_incr.cast_signed()) - i64::from(required.cast_signed()),
                sentient::RegType::XrfRdPtr,
            ));
            *xrf_read_incr = required;
        }
    }
}

/// The preorder `unit_op.walk<MacOp>` itself — reaching the macs, which is not part of the port.
fn plan_xrf_incr_adds<A: Arch>(body: &mut [Op], values: &mut Values, plan: &mut Vec<XrfIncrAdd>) {
    for each in body {
        match each {
            Op::Sentient(inner) => {
                plan_one_mac::<A>(inner, values, plan);
                for region in sentient::regions_mut(inner) {
                    plan_xrf_incr_adds::<A>(region, values, plan);
                }
            }
            Op::AffineFor(loop_op) => plan_xrf_incr_adds::<A>(&mut loop_op.body, values, plan),
            Op::UniformRegions(regions) => {
                for region in regions.regions_mut() {
                    plan_xrf_incr_adds::<A>(&mut region.body, values, plan);
                }
            }
            // ⛔ NO `_` ARM — no `sentient.vector_mac` can sit in a lower-rung region; see
            // [`insert_beside`].
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

/// Replaces: e037_replaceXRFImplicitIncrWithAdd
///
/// Every xrf-related mac whose carried increment differs from the implied one (write: the unroll
/// factor; read: `getXrfRdPtrIncrValAfterMAC`) gains an explicit `sentient.scalar_add` of the
/// difference, and the mac's own attribute is reset to the implied value.
/// ⛔ THE CONSTANTS GO TO THE PROGRAM PREAMBLE: `const_builder` is anchored on `unit_op` itself and
/// only the adds' builder follows the mac (`CanonicalizeXRFPointers.cpp:80-87`). The body ends
/// `[mac, add_rd, add_wt]` — both adds insert immediately after the mac, and write is planned first.
pub fn replace_xrf_implicit_incr_with_add<A: Arch>(
    preamble: &mut Vec<Op>,
    unit: &mut ProgramUnit<A>,
    values: &mut Values,
) {
    let mut plan = Vec::new();
    plan_xrf_incr_adds::<A>(&mut unit.body, values, &mut plan);
    for XrfIncrAdd {
        mac_result,
        constant,
        add,
        add_result,
    } in plan
    {
        dialects::replace_all_uses_with(&mut unit.body, mac_result, add_result);
        preamble.push(constant);
        if insert_beside(&mut unit.body, mac_result, Beside::After, add).is_some() {
            todo!(
                "replaceXRFImplicitIncrWithAdd: the mac binding {mac_result:?} left the unit body \
                 between the walk and the insertion"
            )
        }
    }
}

/// ONE ENTRY OF `XRFRegisterAnalyzer::getValToMinExprMap()`, WITH ITS CONSTANT ANSWER.
///
/// ⛔ `Analyses/XRFRegisterAnalyzer.*` IS OUT OF CAMPAIGN SCOPE (`crustify-senpass/TASK.md`), so
/// both of the analyzer's answers arrive as data: the key whose minimal expression it computed, and
/// what `getValIfConstant` said about that key. Inventing either is what the brief forbids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XrfMinExpr {
    /// The map key — the value whose minimal expression was computed.
    pub val: Val,
    /// `getValIfConstant(val)` — `None` where the expression is not a unit-independent constant.
    pub constant: Option<i64>,
}

/// Replaces: e038_replaceConstXRFExpressions
///
/// Each xrf-related `sentient.scalar_add` with a unit-independent constant expression is replaced by
/// a `sentient.scalar_constant` in the add's OWN block, and the adds are erased only after every
/// rewire — an op destroyed while still used is MLIR's own abort, not a diagnostic.
/// ⛔ ITER ARGS ARE SKIPPED EVEN WHEN CONSTANT (`CanonicalizeXRFPointers.cpp:145-149`), or
/// RegisterTypeAssignment answers them with a copy op inside the loop. A block argument has no
/// defining op, so the reference's two skip tests collapse into this island's one lookup.
pub fn replace_const_xrf_expressions<A: Arch>(
    unit: &mut ProgramUnit<A>,
    values: &mut Values,
    val_to_min_expr: &[XrfMinExpr],
) {
    let mut to_delete = Vec::new();
    for entry in val_to_min_expr {
        let Some(Op::Sentient(sentient::Op::ScalarAdd { reg, ty, .. })) =
            dialects::defining_op(entry.val, &unit.body)
        else {
            continue;
        };
        let (locale, ty) = (reg.map(|reg| reg.locale), *ty);
        // `DT_CHECK(is_any_of(getValueRegLocale(val), xrfrdptr, xrfwrptr))` — for an add that is the
        // add's own `regLocale` (`SentientOps.cpp:1805-1806`), `unknown` when it has none.
        match locale {
            Some(sentient::RegType::XrfRdPtr | sentient::RegType::XrfWrPtr) => {}
            other => todo!(
                "replaceConstXRFExpressions: DT_CHECK(is_any_of(locale, xrfrdptr, xrfwrptr)) on a \
                 scalar_add whose regLocale is {other:?} (CanonicalizeXRFPointers.cpp:152-153)"
            ),
        }
        let Some(value) = entry.constant else {
            continue;
        };
        let result = values.mint();
        // `ConstantOp::create(builder, owner->getLoc(), val.getType(), const_value.value())`.
        let constant = Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result,
            reg_locale: sentient::RegType::Imm,
            ty,
            is_symbol: false,
        });
        if insert_beside(&mut unit.body, entry.val, Beside::Before, constant).is_some() {
            todo!(
                "replaceConstXRFExpressions: nothing in the unit body binds {:?}, which its own \
                 defining op was just found in",
                entry.val
            )
        }
        dialects::replace_all_uses_with(&mut unit.body, entry.val, result);
        to_delete.push(entry.val);
    }
    // `for (auto op : to_delete) op->erase();`
    for val in to_delete {
        dialects::erase_defining_op(&mut unit.body, val);
    }
}

// crustify:todo: e296_runOnOperation
//   authority : dcc/src/Transform/Sentient/CanonicalizeXRFPointers.cpp:172  (33 body lines, level 1)
//   original  : void CanonicalizeXRFPointersPass::runOnOperation()
//   calls     : e037_replaceXRFImplicitIncrWithAdd, e038_replaceConstXRFExpressions, e252_size

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::islands::dataflow_ir::Units;
    use crate::units::DfirUnit;

    /// A unit running `body` on one PE — the pass reads nothing else off the unit.
    fn unit(body: Vec<Op>) -> ProgramUnit<Dd2> {
        ProgramUnit {
            on: Units::one(DfirUnit::Pe, Val(100)),
            precision: None,
            body,
            arch: core::marker::PhantomData,
        }
    }

    /// A `sentient.scalar_copy` reading `input`, standing for any other user of a mac's pointer.
    fn reader(input: Val, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarCopy {
            input,
            result,
            reg: sentient::Reg {
                locale: sentient::RegType::Lccr,
                index: None,
            },
            program_header: false,
            element_size: None,
        })
    }

    /// A mac that is xrf-related BOTH ways, so one walk exercises both branches: `x2` against a
    /// carried write increment of 5, and `int8` on DD2 (⇒ 2) against a carried read increment of 3.
    #[test]
    fn e037_splits_both_implicit_increments_into_adds() {
        let mac = Op::Sentient(sentient::Op::VectorMac {
            mask: None,
            xrf_write_ptr: None,
            xrf_read_ptr: None,
            results: vec![Val(10), Val(11)],
            op_a: sentient::Operand::from(sentient::Port::Xrf),
            op_b: sentient::Operand::from(sentient::Port::Lx),
            op_c: sentient::Operand::from(sentient::Port::Lx),
            result: sentient::ResultPorts {
                forwarding: vec![sentient::Port::Xrf],
                ..sentient::ResultPorts::default()
            },
            mode: sentient::FmaMode::FusedMulAdd,
            compute_precision: sentient::Precision::Int8,
            fold_mode: None,
            unroll_factor: sentient::UnrollFactor::X2,
            xrf_read_incr: 3,
            xrf_write_incr: 5,
            data_transfer_only: false,
            dbg_name: None,
        });
        let mut unit = unit(vec![
            mac,
            reader(Val(10), Val(12)),
            reader(Val(11), Val(13)),
        ]);
        let mut preamble = Vec::new();
        let mut values = Values::default();

        replace_xrf_implicit_incr_with_add(&mut preamble, &mut unit, &mut values);

        // The two differences, in the reference's creation order: write (5 - 2) then read (3 - 2).
        assert_eq!(
            preamble,
            vec![
                Op::Sentient(sentient::Op::ScalarConstant {
                    value: 3,
                    result: Val(0),
                    reg_locale: sentient::RegType::Imm,
                    ty: ScalarTy::Index,
                    is_symbol: false,
                }),
                Op::Sentient(sentient::Op::ScalarConstant {
                    value: 1,
                    result: Val(2),
                    reg_locale: sentient::RegType::Imm,
                    ty: ScalarTy::Index,
                    is_symbol: false,
                }),
            ]
        );
        // ⭐ THE READ ADD ENDS UP FIRST: both are inserted right after the mac.
        let add = |lhs, rhs, result, locale| {
            Op::Sentient(sentient::Op::ScalarAdd {
                lhs,
                rhs,
                result,
                reg: Some(sentient::Reg {
                    locale,
                    index: None,
                }),
                ty: ScalarTy::Index,
                element_size: None,
            })
        };
        assert_eq!(
            &unit.body[1..],
            &[
                add(Val(11), Val(2), Val(3), sentient::RegType::XrfRdPtr),
                add(Val(10), Val(0), Val(1), sentient::RegType::XrfWrPtr),
                // Both readers now read the adds, and neither add reads the other.
                reader(Val(1), Val(12)),
                reader(Val(3), Val(13)),
            ]
        );
        let Op::Sentient(sentient::Op::VectorMac {
            xrf_read_incr,
            xrf_write_incr,
            ..
        }) = &unit.body[0]
        else {
            panic!("the mac is still the first op")
        };
        assert_eq!((*xrf_write_incr, *xrf_read_incr), (2, 2));
    }

    /// A constant xrf read-pointer add is replaced in its own block and then erased; a second entry
    /// with no constant answer is left alone, which is the `has_value()` negative.
    #[test]
    fn e038_replaces_a_constant_add_and_erases_it() {
        let add = Op::Sentient(sentient::Op::ScalarAdd {
            lhs: Val(30),
            rhs: Val(31),
            result: Val(20),
            reg: Some(sentient::Reg {
                locale: sentient::RegType::XrfRdPtr,
                index: None,
            }),
            ty: ScalarTy::Index,
            element_size: None,
        });
        let kept = Op::Sentient(sentient::Op::ScalarAdd {
            lhs: Val(32),
            rhs: Val(33),
            result: Val(21),
            reg: Some(sentient::Reg {
                locale: sentient::RegType::XrfWrPtr,
                index: None,
            }),
            ty: ScalarTy::Index,
            element_size: None,
        });
        let mut unit = unit(vec![add, kept.clone(), reader(Val(20), Val(22))]);
        let mut values = Values::default();

        replace_const_xrf_expressions(
            &mut unit,
            &mut values,
            &[
                XrfMinExpr {
                    val: Val(20),
                    constant: Some(7),
                },
                XrfMinExpr {
                    val: Val(21),
                    constant: None,
                },
            ],
        );

        assert_eq!(
            unit.body,
            vec![
                Op::Sentient(sentient::Op::ScalarConstant {
                    value: 7,
                    result: Val(0),
                    reg_locale: sentient::RegType::Imm,
                    ty: ScalarTy::Index,
                    is_symbol: false,
                }),
                kept,
                reader(Val(0), Val(22)),
            ]
        );
    }
}

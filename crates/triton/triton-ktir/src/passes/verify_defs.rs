// SPDX-License-Identifier: Apache-2.0

//! THE SSA VERIFIER: a KTIR program that USES a value nothing defines, or
//! uses one where it is not in scope, is malformed IR, and the malformedness
//! is a REFUSAL, not a downstream accident.
//!
//! MLIR runs a verifier after every pass; a pass that leaves a use without a
//! definition is caught the moment it runs, by name. Nothing here played that
//! role: an undefined operand survived the whole pipeline and surfaced only
//! when some consumer happened to chase it -- a resolver's accidental refusal,
//! far from the pass that created the defect. This module is that verifier.
//!
//! ONE LAW, TWO REPRESENTATIONS. [`verify_module_defs`] states it over our own
//! generic [`Module`] (one `Op` per spelling, regions with block arguments);
//! [`verify_irfunction_defs`] states it over their typed
//! [`ktir_core::ir::IRFunction`] (a flat op list, `scf.for` carries its region
//! arguments as attributes). The law does not differ between them: every
//! operand of every op, regions included, must resolve to a definition that is
//! IN SCOPE at the use.
//!
//! DEFINEDNESS AND DOMINANCE. The def set is not flat: it is scoped, one per
//! region, walked in program order. A region's defs -- its block arguments
//! (an `scf.for`'s iter args and induction variable, a `tensor.generate`'s
//! index, a function's parameters) and the results of the ops inside it --
//! are visible inside that region only; an op's results are visible after the
//! op, to the rest of its own region and to the regions of later ops, never
//! to the op's own regions (a region cannot use the results of the op that
//! encloses it) and never past the region that defines them. A use that
//! passes a flat def set but violates either clause -- a value defined inside
//! an `scf.for` body read after the for, a use that textually precedes its
//! definition -- is refused here, at the pass that produced it, instead of
//! miscompiling in whatever consumer chased it later.
//!
//! CALLED BY THE DRIVERS, NOT BY THE PASSES. One call after each pass in
//! `make_ktir`, one after each step of `to_ktir::run`, one at the end of
//! `to_ktir_emit::lower` (the producer of the `IRFunction`), and one at the end
//! of `from_ttir::convert` (the producer of the `Module`). No pass invokes the
//! verifier on its own output, so none can forget it and none can bypass it.
//!
//! ATTRIBUTION IS "THE PASS THAT RAN LAST". The driver passes the name of the
//! step that just completed; a refusal names it. For a defect born inside the
//! pipeline that is the pass that produced it. For a defect already present on
//! entry it is the FIRST step to run -- the honest limit of a verifier placed
//! between passes: it can say when the program became unverifiable, not which
//! earlier transform outside this pipeline was responsible.
//!
//! THE LATER RESULTS OF A MULTI-RESULT OP. Their `Operation` records only the
//! FIRST result in `result`; the rest are named by the `ResultNames` attribute
//! `to_ktir_emit` stamps for exactly this purpose, and an `scf.for`'s induction
//! variable and carries ride `IterVar` / `IterArgs`, a `region.bb0_args`
//! marker's block arguments ride `Names`. The verifier resolves all of these
//! through a minimal attribute read that is LOCAL TO THIS FILE: when the
//! broadcast-predicate branch lands its `source_of` -- the one whole resolver
//! over `IRFunction` -- that resolver REPLACES this read, and the two are never
//! alive at once. Asking only `result == Some(v)` would flag every later result
//! of a multi-result op as undefined, which is a verifier defect, not a program
//! defect.
//!
//! NOT COVERED, STATED. `lower_subtile_tape_to_ktir` mints a class of
//! `KtirNode`s by hand, bypassing `to_ktir_emit::lower`; those functions do not
//! pass through this verifier. And [`Module::hints`] is never read: it exists
//! for the text instruments alone ("no pass may"), and a diagnostic that
//! changed shape with a value's spelling would be a pass in disguise.

use std::collections::HashSet;

use ktir_core::attrkey::AttrKey as KAttrKey;
use ktir_core::ir::{Attr as KAttr, IRFunction, Operation as KOperation, Ssa as KSsa};

use crate::ir::{Module, Op, Ssa};
use crate::Refusal;

/// THE TYPED VIOLATION. A use that breaks the law breaks it in exactly one of
/// two ways, and the way is a VARIANT, not a sentence: `Undefined` is a value
/// nothing in the program defines; `OutOfScope` is a value something defines
/// but not where this use sits. Both carry the facts of the use -- the pass
/// that ran last, the op's kind, the result the op defines (if any), the
/// operand's index, the value -- as FIELDS, so a test matches on the variant
/// and asserts them, and the [`Refusal`] a driver hands the user is BUILT from
/// the value rather than parsed back out of its message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Violation {
    /// The operand resolves to nothing: no op result, argument, carry or
    /// block argument anywhere in the program defines it.
    Undefined {
        pass: &'static str,
        op: String,
        defines: Option<u32>,
        operand: usize,
        value: u32,
    },
    /// The operand is defined somewhere in the program but not IN SCOPE at
    /// this use: a region's defs are visible inside it only, and an op's
    /// results are visible after the op.
    OutOfScope {
        pass: &'static str,
        op: String,
        defines: Option<u32>,
        operand: usize,
        value: u32,
    },
}

impl Violation {
    /// The refusal this violation becomes at a driver's hands. The message is
    /// the violation's own fields rendered; nothing reads it back.
    pub fn refusal(&self) -> Refusal {
        let (pass, op, defines, operand, value, law) = match self {
            Violation::Undefined {
                pass,
                op,
                defines,
                operand,
                value,
            } => (
                *pass,
                op,
                *defines,
                *operand,
                *value,
                "resolves to nothing -- no op result, argument, carry or block \
                 argument defines it",
            ),
            Violation::OutOfScope {
                pass,
                op,
                defines,
                operand,
                value,
            } => (
                *pass,
                op,
                *defines,
                *operand,
                *value,
                "is defined, but not in scope at this use -- a region's defs are \
                 visible inside it only, and an op's results are visible after the op",
            ),
        };
        let who = match defines {
            Some(r) => format!("op `{op}` (defines %{r})"),
            None => format!("op `{op}`"),
        };
        Refusal::new(
            pass,
            format!("use of %{value}: operand {operand} of {who} {law}"),
        )
    }
}

/// `pass` names the step that just ran; see the module doc for what that
/// attribution does and does not claim.
pub fn verify_module_defs(
    m: &Module,
    pass: &'static str,
) -> std::result::Result<(), Violation> {
    // The whole program's defs, unscoped: only so the violation can say WHICH
    // law the use broke -- a value defined somewhere but out of scope here
    // (dominance) reads differently from one nothing defines (definedness).
    let mut everywhere = HashSet::new();
    collect_module_defs(&m.ops, &mut everywhere);
    verify_module_region(&m.ops, &mut HashSet::new(), &everywhere, pass)
}

/// Same law, over the emitted function the consumers actually see.
pub fn verify_irfunction_defs(
    f: &IRFunction,
    pass: &'static str,
) -> std::result::Result<(), Violation> {
    let mut everywhere: HashSet<KSsa> = f.arguments.iter().map(|(v, _)| *v).collect();
    for o in f.ops_deep() {
        if let Some(r) = o.result {
            everywhere.insert(r);
        }
        for key in region_and_result_attr_keys() {
            if let Some(KAttr::Ssas(vs)) = o.attr(key) {
                everywhere.extend(vs.iter().copied());
            }
        }
    }
    let top: HashSet<KSsa> = f.arguments.iter().map(|(v, _)| *v).collect();
    verify_irfunction_region(f.operations, &top, &everywhere, pass)
}

/// One region of ours: `inherited` is what encloses this region (the defs
/// visible before the op that carries it), `everywhere` is the unscoped set
/// the refusal wording reads. Each op's operands are checked against what is
/// visible AT THAT POINT; its results join the visible set only after its own
/// regions are walked, so nothing inside an op can use the op's results.
fn verify_module_region(
    ops: &[Op],
    inherited: &HashSet<Ssa>,
    everywhere: &HashSet<Ssa>,
    pass: &'static str,
) -> std::result::Result<(), Violation> {
    let mut visible = inherited.clone();
    for op in ops {
        for (i, v) in op.operands.iter().enumerate() {
            if !visible.contains(v) {
                return Err(violation_of(
                    pass,
                    op.kind.spelling().to_string(),
                    op.result().map(|r| r.0),
                    i,
                    v.0,
                    everywhere.contains(v),
                ));
            }
        }
        for r in &op.regions {
            let mut inner = visible.clone();
            inner.extend(r.args.iter().map(|(v, _)| *v));
            verify_module_region(&r.ops, &inner, everywhere, pass)?;
        }
        visible.extend(op.results.iter().copied());
    }
    Ok(())
}

/// One region of theirs, same law: `IterVar` / `IterArgs` name an `scf.for`'s
/// region arguments and are visible inside its regions only; `result`,
/// `ResultNames` and a `region.bb0_args` marker's `Names` are the op's own
/// defs and are visible after it.
fn verify_irfunction_region(
    ops: &[KOperation],
    inherited: &HashSet<KSsa>,
    everywhere: &HashSet<KSsa>,
    pass: &'static str,
) -> std::result::Result<(), Violation> {
    let mut visible = inherited.clone();
    for o in ops {
        for (i, v) in o.operands.iter().enumerate() {
            if !visible.contains(v) {
                return Err(violation_of(
                    pass,
                    kind_of(o),
                    o.result.map(|r| r.0),
                    i,
                    v.0,
                    everywhere.contains(v),
                ));
            }
        }
        let mut inner = visible.clone();
        for key in [KAttrKey::IterVar, KAttrKey::IterArgs] {
            if let Some(KAttr::Ssas(vs)) = o.attr(key) {
                inner.extend(vs.iter().copied());
            }
        }
        for r in o.regions {
            verify_irfunction_region(r, &inner, everywhere, pass)?;
        }
        if let Some(r) = o.result {
            visible.insert(r);
        }
        for key in [KAttrKey::ResultNames, KAttrKey::Names] {
            if let Some(KAttr::Ssas(vs)) = o.attr(key) {
                visible.extend(vs.iter().copied());
            }
        }
    }
    Ok(())
}

/// Every value a module defines, unscoped: each region's block arguments (a
/// `scf.for`'s iter args and induction variable, a `tensor.generate`'s index,
/// a function's parameters) and every result of every op, regions included.
fn collect_module_defs(ops: &[Op], defs: &mut HashSet<Ssa>) {
    for op in ops {
        defs.extend(op.results.iter().copied());
        for r in &op.regions {
            defs.extend(r.args.iter().map(|(v, _)| *v));
            collect_module_defs(&r.ops, defs);
        }
    }
}

fn region_and_result_attr_keys() -> [KAttrKey; 4] {
    [
        KAttrKey::ResultNames,
        KAttrKey::IterVar,
        KAttrKey::IterArgs,
        KAttrKey::Names,
    ]
}

/// The violation for a use that does not resolve in scope: `defined` says
/// which law broke -- a value defined somewhere in the program but not
/// visible at this use is `OutOfScope` (dominance); one nothing anywhere
/// defines is `Undefined` (definedness). Both carry the op, the operand's
/// index, the value, and the pass that ran last.
fn violation_of(
    pass: &'static str,
    op: String,
    defines: Option<u32>,
    operand: usize,
    value: u32,
    defined: bool,
) -> Violation {
    if defined {
        Violation::OutOfScope {
            pass,
            op,
            defines,
            operand,
            value,
        }
    } else {
        Violation::Undefined {
            pass,
            op,
            defines,
            operand,
            value,
        }
    }
}

/// Their `OpKind` has no shared spelling, but it derives `Debug` and every
/// variant prints as its own name -- `ArithAddf`, `ScfFor` -- which is the
/// spelling a diagnostic needs.
fn kind_of(o: &KOperation) -> String {
    format!("{:?}", o.op_type)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::parse;
    use ktir_core::arena::Arena;
    use ktir_core::irtype::IrType as KIrType;
    use ktir_core::opkind::OpKind as KOpKind;

    /// A module from the parser, through the whole `to_ktir` pipeline, holds
    /// its definitions in scope: the well-formed case passes at every hook.
    #[test]
    fn a_well_formed_module_passes() {
        let mut m = parse::parse(WELL_FORMED).unwrap();
        crate::passes::to_ktir::run(&mut m, &[8]).unwrap();
        verify_module_defs(&m, "test").unwrap();
    }

    /// One undefined operand, and only that: the violation is `Undefined` and
    /// its fields name the op, the operand's index, the value, and the pass
    /// that ran last.
    #[test]
    fn an_undefined_operand_is_refused_naming_the_op_and_value() {
        let mut m = parse::parse(WELL_FORMED).unwrap();
        let (op, v) = break_one_operand(&mut m);
        match verify_module_defs(&m, "test-pass") {
            Err(Violation::Undefined {
                pass,
                op: got_op,
                operand,
                value,
                ..
            }) => {
                assert_eq!(pass, "test-pass");
                assert_eq!(value, v.0);
                assert_eq!(operand, 0);
                assert_eq!(got_op, op);
            }
            other => panic!("expected Undefined, got {other:?}"),
        }
    }

    /// DOMINANCE, the region clause: a value defined inside the `tt.reduce`
    /// body is a def of that region -- a use of it AFTER the reduce, outside
    /// the region, is `OutOfScope` even though a flat def set would pass it.
    #[test]
    fn a_use_of_a_region_def_outside_the_region_is_refused() {
        let mut m = parse::parse(WELL_FORMED).unwrap();
        let (op, v) = break_into_region_escape(&mut m);
        match verify_module_defs(&m, "test-pass") {
            Err(Violation::OutOfScope {
                pass,
                op: got_op,
                operand,
                value,
                ..
            }) => {
                assert_eq!(pass, "test-pass");
                assert_eq!(value, v.0);
                assert_eq!(operand, 0);
                assert_eq!(got_op, op);
            }
            other => panic!("expected OutOfScope, got {other:?}"),
        }
    }

    /// DOMINANCE, the ordering clause: an op's results are visible AFTER the
    /// op -- a use that textually precedes its definition is `OutOfScope`,
    /// where the flat def set passed any order.
    #[test]
    fn a_use_before_its_definition_is_refused() {
        let mut m = parse::parse(WELL_FORMED).unwrap();
        let (op, v) = break_into_forward_use(&mut m);
        match verify_module_defs(&m, "test-pass") {
            Err(Violation::OutOfScope {
                op: got_op,
                value,
                ..
            }) => {
                assert_eq!(value, v.0);
                assert_eq!(got_op, op);
            }
            other => panic!("expected OutOfScope, got {other:?}"),
        }
    }

    /// ATTRIBUTION: a defect present before the pipeline is attributed to the
    /// first step that runs, because that is the pass that ran last when the
    /// verifier first saw it.
    #[test]
    fn a_pre_broken_module_names_the_first_step_that_ran() {
        let mut m = parse::parse(WELL_FORMED).unwrap();
        break_one_operand(&mut m);
        let e = crate::passes::to_ktir::run(&mut m, &[8]).unwrap_err();
        assert_eq!(e.pass, "spyre-to-ktir:convert_func", "got {e}");
    }

    /// THEIR `Operation` RECORDS ONLY THE FIRST RESULT. A use of a multi-result
    /// op's SECOND result, named by `ResultNames`, must pass -- the guard
    /// against a verifier that asks only `result == Some(v)` and flags every
    /// later result of a multi-result op as undefined.
    #[test]
    fn an_irfunction_multi_result_second_result_use_passes() {
        let a = Arena::new();
        // Defines %1 (in `result`) and %2 (in `result_names`); the use of %2
        // must resolve.
        let multi = KOperation::new(&a, Some(KSsa(1)), KOpKind::ScfFor, &[KSsa(0)])
            .with_attr(
                &a,
                KAttrKey::ResultNames,
                KAttr::Ssas(a.ssa(vec![KSsa(1), KSsa(2)])),
            )
            .with_attr(&a, KAttrKey::NumResults, KAttr::Int(2));
        let use_second =
            KOperation::new(&a, Some(KSsa(3)), KOpKind::ArithAddf, &[KSsa(2), KSsa(2)]);
        let f = IRFunction {
            name: a.str("k".into()),
            arguments: a.args(vec![(KSsa(0), KIrType::Index)]),
            operations: a.ops(vec![multi, use_second]),
            grid: (1, 1, 1),
            return_type: None,
        };
        verify_irfunction_defs(&f, "test").unwrap();
    }

    /// A `scf.for`'s induction variable and carries are ATTRIBUTES on their
    /// side (`IterVar` / `IterArgs`), and a `region.bb0_args` marker's block
    /// arguments ride `Names`: uses of all three must resolve -- the carries
    /// inside the body, the for's result and the marker's name after it.
    #[test]
    fn an_irfunction_for_carries_and_bb0_args_resolve() {
        let a = Arena::new();
        let body = a.ops(vec![
            // Uses the induction variable %4 and the carry %5.
            KOperation::new(&a, Some(KSsa(6)), KOpKind::ArithAddf, &[KSsa(4), KSsa(5)]),
            KOperation::new(&a, None, KOpKind::ScfYield, &[KSsa(6)]),
        ]);
        let for_op = KOperation::new(&a, Some(KSsa(7)), KOpKind::ScfFor, &[KSsa(1), KSsa(1)])
            .with_attr(&a, KAttrKey::IterVar, KAttr::Ssas(a.ssa(vec![KSsa(4)])))
            .with_attr(&a, KAttrKey::IterArgs, KAttr::Ssas(a.ssa(vec![KSsa(5)])));
        let for_op = KOperation {
            regions: a.regions(vec![body]),
            ..for_op
        };
        // A `region.bb0_args` marker naming %8, then a use of %8.
        let marker = KOperation::new(&a, None, KOpKind::RegionBb0Args, &[]).with_attr(
            &a,
            KAttrKey::Names,
            KAttr::Ssas(a.ssa(vec![KSsa(8)])),
        );
        let use_bb0 = KOperation::new(&a, Some(KSsa(9)), KOpKind::ArithAddf, &[KSsa(8), KSsa(7)]);
        let f = IRFunction {
            name: a.str("k".into()),
            arguments: a.args(vec![(KSsa(1), KIrType::Index)]),
            operations: a.ops(vec![marker, for_op, use_bb0]),
            grid: (1, 1, 1),
            return_type: None,
        };
        verify_irfunction_defs(&f, "test").unwrap();
    }

    /// DOMINANCE over theirs, the region clause: a value defined inside an
    /// `scf.for` body (the body's own op result) is a def of that region -- a
    /// use of it after the for, outside the region, is refused naming the
    /// pass, the op, the operand and the value, and says IN SCOPE, not
    /// "resolves to nothing".
    #[test]
    fn an_irfunction_use_of_a_region_def_outside_the_region_is_refused() {
        let a = Arena::new();
        let body = a.ops(vec![
            KOperation::new(&a, Some(KSsa(5)), KOpKind::ArithAddf, &[KSsa(4), KSsa(4)]),
            KOperation::new(&a, None, KOpKind::ScfYield, &[KSsa(5)]),
        ]);
        let for_op = KOperation::new(&a, Some(KSsa(6)), KOpKind::ScfFor, &[KSsa(1), KSsa(1)])
            .with_attr(&a, KAttrKey::IterVar, KAttr::Ssas(a.ssa(vec![KSsa(4)])));
        let for_op = KOperation {
            regions: a.regions(vec![body]),
            ..for_op
        };
        // %5 is defined inside the for's region; this use sits after the for.
        let escape = KOperation::new(&a, Some(KSsa(7)), KOpKind::ArithAddf, &[KSsa(5), KSsa(6)]);
        let f = IRFunction {
            name: a.str("k".into()),
            arguments: a.args(vec![(KSsa(1), KIrType::Index)]),
            operations: a.ops(vec![for_op, escape]),
            grid: (1, 1, 1),
            return_type: None,
        };
        match verify_irfunction_defs(&f, "test-pass") {
            Err(Violation::OutOfScope {
                pass,
                op,
                operand,
                value,
                ..
            }) => {
                assert_eq!(pass, "test-pass");
                assert_eq!(value, 5);
                assert_eq!(operand, 0);
                assert_eq!(op, "ArithAddf");
            }
            other => panic!("expected OutOfScope, got {other:?}"),
        }
    }

    /// DOMINANCE over theirs, the ordering clause: a use that precedes its
    /// definition in the same flat list is refused.
    #[test]
    fn an_irfunction_use_before_its_definition_is_refused() {
        let a = Arena::new();
        let use_first = KOperation::new(&a, Some(KSsa(3)), KOpKind::ArithAddf, &[KSsa(1), KSsa(2)]);
        let def = KOperation::new(&a, Some(KSsa(2)), KOpKind::ArithAddf, &[KSsa(1), KSsa(1)]);
        let f = IRFunction {
            name: a.str("k".into()),
            arguments: a.args(vec![(KSsa(1), KIrType::Index)]),
            operations: a.ops(vec![use_first, def]),
            grid: (1, 1, 1),
            return_type: None,
        };
        match verify_irfunction_defs(&f, "test-pass") {
            Err(Violation::OutOfScope {
                operand,
                value,
                ..
            }) => {
                assert_eq!(value, 2);
                assert_eq!(operand, 1);
            }
            other => panic!("expected OutOfScope, got {other:?}"),
        }
    }

    /// The emitted function refuses an undefined operand, naming the op, the
    /// value and the pass it was handed.
    #[test]
    fn an_irfunction_undefined_operand_is_refused() {
        let a = Arena::new();
        let stray = KOperation::new(&a, Some(KSsa(3)), KOpKind::ArithAddf, &[KSsa(0), KSsa(9)]);
        let f = IRFunction {
            name: a.str("k".into()),
            arguments: a.args(vec![(KSsa(0), KIrType::Index)]),
            operations: a.ops(vec![stray]),
            grid: (1, 1, 1),
            return_type: None,
        };
        match verify_irfunction_defs(&f, "test-pass") {
            Err(Violation::Undefined {
                pass,
                op,
                operand,
                value,
                ..
            }) => {
                assert_eq!(pass, "test-pass");
                assert_eq!(value, 9);
                assert_eq!(operand, 1);
                assert_eq!(op, "ArithAddf");
            }
            other => panic!("expected Undefined, got {other:?}"),
        }
    }

    /// END TO END at the boundary: a real module through the whole pipeline,
    /// emitted through `to_ktir_emit::lower`, verifies green -- the producer
    /// emits definitions in scope for everything it emits uses of.
    #[test]
    fn a_lowered_irfunction_from_a_real_module_passes() {
        let mut m = parse::parse(WELL_FORMED).unwrap();
        crate::passes::to_ktir::run(&mut m, &[8]).unwrap();
        let a = Arena::new();
        let f = crate::passes::to_ktir_emit::lower(&m, &a).unwrap();
        verify_irfunction_defs(&f, "test").unwrap();
    }

    /// Point the first operand-bearing op's first operand at a value nothing
    /// defines, and report which op was broken so the refusal can be checked
    /// against it.
    fn break_one_operand(m: &mut Module) -> (String, Ssa) {
        let stray = Ssa(4000);
        let f = m.kernel_mut().unwrap();
        let mut broken = None;
        for op in f.regions[0].ops.iter_mut() {
            if !op.operands.is_empty() {
                op.operands[0] = stray;
                broken = Some(op.kind.spelling().to_string());
                break;
            }
        }
        (broken.expect("an operand-bearing op"), stray)
    }

    /// Point a LATER top-level op's first operand at a value defined inside
    /// an EARLIER op's region (the `tt.reduce` body's own result): defined,
    /// but a def of a region this use is outside.
    fn break_into_region_escape(m: &mut Module) -> (String, Ssa) {
        let f = m.kernel_mut().unwrap();
        let top = &f.regions[0].ops;
        let mut region_def = None;
        let mut region_op = 0;
        for (i, op) in top.iter().enumerate() {
            if let Some(v) = op
                .regions
                .first()
                .and_then(|first| first.ops.first())
                .and_then(|inner| inner.results.first())
            {
                region_def = Some(*v);
                region_op = i;
                break;
            }
        }
        let v = region_def.expect("an op with a region whose body defines a value");
        let mut broken = None;
        for (i, op) in f.regions[0].ops.iter_mut().enumerate() {
            if i > region_op && !op.operands.is_empty() {
                op.operands[0] = v;
                broken = Some(op.kind.spelling().to_string());
                break;
            }
        }
        (broken.expect("an operand-bearing op after the region"), v)
    }

    /// Point an EARLIER top-level op's first operand at a value a LATER op
    /// defines: defined, but not yet -- an op's results are visible after it.
    fn break_into_forward_use(m: &mut Module) -> (String, Ssa) {
        let f = m.kernel_mut().unwrap();
        let later = f.regions[0]
            .ops
            .iter()
            .rev()
            .find(|o| !o.results.is_empty())
            .map(|o| o.results[0])
            .expect("an op with a result");
        let mut broken = None;
        for op in f.regions[0].ops.iter_mut() {
            if !op.operands.is_empty() && !op.results.contains(&later) {
                op.operands[0] = later;
                broken = Some(op.kind.spelling().to_string());
                break;
            }
        }
        (broken.expect("an operand-bearing op before the def"), later)
    }

    const WELL_FORMED: &str = "\
module {
  tt.func public @k(%scale: f16) attributes {noinline = false} {
    %x = arith.constant dense<0.000000e+00> : tensor<64x64xf16>
    %m = \"tt.reduce\"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a: f16, %b: f16):
      %c = arith.maxnumf %a, %b : f16
      tt.reduce.return %c : f16
    }) : (tensor<64x64xf16>) -> tensor<64xf16>
    %one = arith.constant 1.0 : f32
    %seed = tensor.splat %one : tensor<64xf16>
    %s = tt.splat %scale : tensor<64xf16>
    %ss = arith.mulf %seed, %s : tensor<64xf16>
    %out = arith.addf %m, %ss : tensor<64xf16>
    tt.return
  }
}
";
}

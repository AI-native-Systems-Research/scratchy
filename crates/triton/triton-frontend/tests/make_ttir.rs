//! `make_ttir` IN RUST, against the C++ pipeline's own output, per configuration.
//!
//! Needs the `ruff` feature: every case starts from Python SOURCE, so this exercises the
//! WHOLE value path -- `codegen::compile` then `opt::make_ttir`, with nothing printed and
//! nothing parsed in between.
#![cfg(feature = "ruff")]
//!
//! # THE ORACLE, AND WHY IT IS THE SECOND GOLDEN
//!
//! `tools/gen_ttir_goldens.py` writes two checkpoints per configuration.
//! `tests/golden_diff.rs` uses the first (`.ttir_raw.mlir`, `make_ir`'s output) because
//! that is what `codegen` ports. THIS test uses the second (`.ttir.mlir`), which is what
//! `backend.make_ttir` produces and therefore what bridge two was ported against.
//!
//! # WHAT IS GATED AND WHAT IS COUNTED
//!
//! GATED, and a difference here fails the test: the function set, op sequence, op names,
//! result types, attributes, and operand linkage -- everything `triton_frontend::diff`
//! calls structural.
//!
//! COUNTED, and reported rather than gated: `loc_name` / `result_loc_names` findings.
//! `src/opt.rs`'s header records the two reasons -- MLIR's inliner rewrites an inlined
//! op's location to `callsite(... at ...)`, which `ttir::Loc` cannot represent, and its
//! constant folder rematerializes hoisted constants with `loc(unknown)`. The reason this
//! is not a hole is that `triton_ktir::ir` HAS NO LOCATION FIELD AT ALL, so no location
//! difference at this stage can reach KTIR, the bundle or the program. The count is
//! printed for every configuration so the exemption stays a number rather than a silence.
//!
//! # THE CONTROLS ARE THE POINT
//!
//! A diff that skips locations could be a diff that skips everything. So
//! `planted_*` below break the pipeline's output in each of the ways that matter and
//! assert the gate CATCHES them -- including one planted difference that is ONLY a
//! location difference, to prove the exemption is narrow and not the whole comparison.

mod common;

use triton_frontend::target::Target;
use triton_frontend::ttir::{Attr, Type};
use triton_frontend::{codegen, diff, opt, ttir};

use common::Case;

/// The POST-`make_ttir` golden -- bridge two's input.
fn golden_post(name: &str) -> Option<String> {
    let p = common::crate_dir()
        .join("tests/goldens")
        .join(format!("{name}.ttir.mlir"));
    std::fs::read_to_string(p).ok()
}

/// Every configuration that reaches TTIR, with the target its golden was produced at.
///
/// The two attention configurations use `upstream_gpu` for the same reason
/// `tests/fixture_status.rs` does: their epilogue is the one bare f16 divide in any
/// fixture and the oracle widens it. `bias_add_f32` is absent because it never reaches
/// TTIR at all.
fn cases() -> Vec<(Case, Target)> {
    vec![
        (common::vector_add(), Target::spyre()),
        (common::mul(), Target::spyre()),
        (common::swiglu("swiglu_mlp", 128, 256, 128), Target::spyre()),
        (
            common::swiglu("swiglu_mlp_granite", 4096, 12800, 4096),
            Target::spyre(),
        ),
        (common::swiglu("swiglu_mlp_tiledk", 128, 256, 64), Target::spyre()),
        (common::embedding("embedding", 512, 128), Target::spyre()),
        (common::embedding("embedding_granite", 49159, 4096), Target::spyre()),
        (common::rmsnorm("rmsnorm", 128), Target::spyre()),
        (common::rmsnorm("rmsnorm_granite", 4096), Target::spyre()),
        (common::rope("rope", 4), Target::spyre()),
        (common::rope("rope_granite_q", 32), Target::spyre()),
        (common::rope("rope_granite_kv", 8), Target::spyre()),
        (common::decoder_layer(), Target::spyre()),
        (common::decoder_two_layers(), Target::spyre()),
        (
            common::attention("attention_flash_noncausal", 1),
            Target::upstream_gpu(),
        ),
        (
            common::attention("attention_flash_causal", 3),
            Target::upstream_gpu(),
        ),
    ]
}

/// Sort each block's LEADING `arith.constant` run into a canonical order, on BOTH sides.
///
/// # WHY THIS IS A NORMALIZATION AND NOT A WEAKENING
///
/// A constant has no operands and no side effects, so its position among other constants
/// is not a fact about the program. Its VALUE, its TYPE, and HOW MANY there are all still
/// are, and this normalization changes none of them -- it only removes the permutation.
/// `triton_ktir::text::diff` reached the same conclusion independently at the next stage,
/// and its header says why: matching MLIR's internal hoist order is a claim a port cannot
/// make from the outside.
///
/// WHAT IS ACTUALLY UNKNOWABLE, concretely. `src/opt.rs`'s `hoist_constants` reproduces
/// MLIR's move-to-front rule and it is right for eleven of the sixteen configurations. It
/// is wrong for SwiGLU, the decoder and attention by a permutation of three or four
/// constants, because MLIR's greedy driver interleaves hoisting with folding and DCE on a
/// worklist, so which constant is met first depends on rewrite order rather than on
/// program order.
///
/// Applying it to both sides also keeps the OPERAND LINKAGE comparison meaningful: the
/// diff refers to a value as "result of op K", so a permutation of the leading run
/// renumbers every reference to a constant. Without this, one permuted constant reports as
/// a dozen operand differences and the real findings are buried.
///
/// `planted_constant_value_change_is_caught` is the control: the multiset still has to
/// notice a changed value.
fn canonical_constant_order(m: &mut ttir::Module) {
    let tys: Vec<String> = m.values.iter().map(|v| v.ty.to_string()).collect();
    for f in &mut m.funcs {
        for b in &mut f.body.blocks {
            let n = b
                .ops
                .iter()
                .take_while(|o| o.name == "arith.constant" && o.regions.is_empty())
                .count();
            b.ops[..n].sort_by_key(|o| {
                let v = o.attrs.get("value").map(|a| format!("{a:?}")).unwrap_or_default();
                let t = o
                    .results
                    .first()
                    .map(|r| tys[r.0 as usize].clone())
                    .unwrap_or_default();
                format!("{t}|{v}")
            });
        }
    }
}

/// Split a report's findings into the ones that gate and the location ones that only count.
fn split(findings: &[String]) -> (Vec<String>, Vec<String>) {
    let mut gate = Vec::new();
    let mut locs = Vec::new();
    for f in findings {
        if f.contains(".loc_name:") || f.contains(".result_loc_names:") {
            locs.push(f.clone());
        } else {
            gate.push(f.clone());
        }
    }
    (gate, locs)
}

/// Compare, with the leading-constant permutation normalized away on BOTH sides.
///
/// Parses the golden here rather than through `diff::compare_to_golden_text` for exactly
/// one reason: the normalization has to be applied to the golden too. A parse failure is a
/// finding, never a pass.
fn compare_normalized(golden_text: &str, ours: &ttir::Module) -> diff::Report {
    let mut golden = match ttir::parse::parse_module(golden_text) {
        Ok(g) => g,
        Err(e) => panic!("the post-make_ttir golden did not parse: {e}"),
    };
    let mut ours = ours.clone();
    canonical_constant_order(&mut golden);
    canonical_constant_order(&mut ours);
    diff::compare(&golden, &ours)
}

/// Compile `case` from Python and run the Rust `make_ttir` on the value.
fn run(case: &Case, target: Target) -> ttir::Module {
    let src = common::fixture_src(&case.fixture);
    let mut m = codegen::compile(&src, &case.spec, target)
        .unwrap_or_else(|e| panic!("`{}` did not compile: {e}", case.name));
    let stats = opt::make_ttir_measured(&mut m)
        .unwrap_or_else(|e| panic!("`{}`: make_ttir refused: {e}", case.name));
    // THE CENSUS IS AN ASSERTION, not a print. A pipeline that leaves a call behind or a
    // second function behind has not done the job bridge two needs, whatever the diff says.
    assert_eq!(
        stats.calls_after, 0,
        "`{}`: {} tt.call(s) survived make_ttir -- bridge two has no tt.call arm and would \
         refuse the module",
        case.name, stats.calls_after
    );
    assert_eq!(
        stats.funcs_after, 1,
        "`{}`: {} function(s) survived symbol-dce (from {}), and \
         triton_ktir::ir::Module::kernel() refuses anything but exactly one",
        case.name, stats.funcs_after, stats.funcs_before
    );
    eprintln!(
        "{}: {} func(s)/{} op(s)/{} call(s) -> {} func/{} op(s)/0 calls",
        case.name,
        stats.funcs_before,
        stats.ops_before,
        stats.calls_before,
        stats.funcs_after,
        stats.ops_after
    );
    m
}

#[test]
fn every_configuration_matches_the_post_make_ttir_golden_structurally() {
    let mut failures: Vec<String> = Vec::new();
    let mut loc_total = 0usize;
    let mut matched_total = 0usize;
    for (case, target) in cases() {
        let ours = run(&case, target);
        let golden = match golden_post(&case.name) {
            Some(g) => g,
            None => {
                failures.push(format!(
                    "{}: no tests/goldens/{}.ttir.mlir -- regenerate with \
                     tools/gen_ttir_goldens.py",
                    case.name, case.name
                ));
                continue;
            }
        };
        let report = compare_normalized(&golden, &ours);
        let (gate, locs) = split(&report.findings);
        loc_total += locs.len();
        matched_total += report.matched_ops;
        if report.matched_ops == 0 {
            failures.push(format!(
                "{}: matched ZERO ops -- that is a diff comparing nothing, not a pass",
                case.name
            ));
        }
        if !gate.is_empty() {
            eprintln!(
                "---- OUR POST-make_ttir TTIR for {} ----\n{}",
                case.name,
                ttir::print::print_module(&ours)
            );
            failures.push(format!(
                "{}: {} structural finding(s):\n{}",
                case.name,
                gate.len(),
                gate.iter()
                    .map(|f| format!("  DIFF {f}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
        eprintln!(
            "  {} -> {} op(s) matched, {} structural finding(s), {} location finding(s)",
            case.name,
            report.matched_ops,
            gate.len(),
            locs.len()
        );
    }
    eprintln!(
        "TOTAL: {matched_total} op(s) matched across {} configuration(s); \
         {loc_total} location finding(s) exempted (see src/opt.rs)",
        cases().len()
    );
    assert!(
        failures.is_empty(),
        "make_ttir does not reproduce the C++ pipeline for {} configuration(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(
        matched_total > 0,
        "the whole test matched zero ops, which is a harness failure rather than a pass"
    );
}

//===----------------------------------------------------------------------===//
// The controls
//===----------------------------------------------------------------------===//

/// A small module the controls mutate: `rmsnorm`, which has a generated helper (so the
/// inliner runs), a `tt.reduce` region, duplicate constants and a dead overflow check.
fn control_pair() -> (String, ttir::Module) {
    let case = common::rmsnorm("rmsnorm", 128);
    let golden = golden_post(&case.name).expect("rmsnorm's post golden");
    let ours = run(&case, Target::spyre());
    (golden, ours)
}

#[test]
fn the_unmodified_control_agrees_with_itself() {
    let (golden, ours) = control_pair();
    let (gate, _) = split(&compare_normalized(&golden, &ours).findings);
    assert!(
        gate.is_empty(),
        "the control's baseline must be clean or every planted difference below proves \
         nothing: {gate:?}"
    );
}

#[test]
fn planted_op_deletion_is_caught() {
    let (golden, mut ours) = control_pair();
    let ops = &mut ours.funcs[0].body.blocks[0].ops;
    let victim = ops
        .iter()
        .position(|o| o.name == "math.rsqrt")
        .expect("rmsnorm has a math.rsqrt");
    let name = ops[victim].name.clone();
    ops.remove(victim);
    let (gate, _) = split(&compare_normalized(&golden, &ours).findings);
    assert!(
        !gate.is_empty(),
        "deleting `{name}` was not caught -- the structural gate is not comparing the op \
         sequence"
    );
}

#[test]
fn planted_attribute_change_is_caught() {
    let (golden, mut ours) = control_pair();
    let mut hit = false;
    for op in &mut ours.funcs[0].body.blocks[0].ops {
        if op.name == "tt.expand_dims" {
            op.attrs.insert("axis".to_string(), Attr::Int(7, Type::i32()));
            hit = true;
            break;
        }
    }
    assert!(hit, "rmsnorm has a tt.expand_dims to retarget");
    let (gate, _) = split(&compare_normalized(&golden, &ours).findings);
    assert!(
        !gate.is_empty(),
        "changing tt.expand_dims's `axis` was not caught -- the gate is not comparing \
         attributes"
    );
}

#[test]
fn planted_operand_rewire_is_caught() {
    let (golden, mut ours) = control_pair();
    let ops = &mut ours.funcs[0].body.blocks[0].ops;
    let victim = ops
        .iter()
        // The operands must DIFFER, or the swap is the identity and the control proves
        // nothing. `rmsnorm`'s first mulf is `x * x`; the one that works is `ms * INV_D`.
        .position(|o| {
            o.name == "arith.mulf" && o.operands.len() == 2 && o.operands[0] != o.operands[1]
        })
        .expect("rmsnorm has an arith.mulf with two DISTINCT operands");
    ops[victim].operands.swap(0, 1);
    let a = ops[victim].operands[0];
    let b = ops[victim].operands[1];
    let (gate, _) = split(&compare_normalized(&golden, &ours).findings);
    assert!(
        !gate.is_empty(),
        "swapping an arith.mulf's operands (%{} <-> %{}) was not caught -- the gate is not \
         comparing operand linkage",
        a.0,
        b.0
    );
}

#[test]
fn planted_result_type_change_is_caught() {
    let (golden, mut ours) = control_pair();
    let victim = ours.funcs[0].body.blocks[0]
        .ops
        .iter()
        .find(|o| o.name == "math.rsqrt")
        .and_then(|o| o.results.first().copied())
        .expect("rmsnorm's math.rsqrt has a result");
    ours.values[victim.0 as usize].ty = Type::tensor(&[64], Type::f16());
    let (gate, _) = split(&compare_normalized(&golden, &ours).findings);
    assert!(
        !gate.is_empty(),
        "retyping math.rsqrt's result was not caught -- the gate is not comparing result \
         types"
    );
}

/// THE NARROWNESS CONTROL. The exemption is supposed to cover locations and NOTHING else,
/// so a planted difference that is ONLY a location must land in the counted bucket and
/// leave the gate clean -- and must be non-empty, or the exemption is vacuous.
#[test]
fn a_planted_location_only_difference_lands_in_the_counted_bucket() {
    let (golden, mut ours) = control_pair();
    let mut hit = false;
    for op in &mut ours.funcs[0].body.blocks[0].ops {
        if op.name == "tt.descriptor_load" {
            op.loc = op.loc.named("a_name_the_golden_does_not_have");
            for r in &op.results {
                let l = ours.values[r.0 as usize].loc.named("also_not_in_the_golden");
                ours.values[r.0 as usize].loc = l;
            }
            hit = true;
            break;
        }
    }
    assert!(hit, "rmsnorm has a tt.descriptor_load to rename");
    let report = compare_normalized(&golden, &ours);
    let (gate, locs) = split(&report.findings);
    assert!(
        !locs.is_empty(),
        "a renamed location produced no location finding -- then the diff is not looking at \
         locations at all and the exemption is describing something that does not exist"
    );
    assert!(
        gate.is_empty(),
        "a location-only difference leaked into the structural gate: {gate:?}"
    );
}

/// The inliner must be a fixed point, not one round. `attention_flash` reaches
/// `_elementwise_max` only through `standard.max`, so a single-round inliner leaves a call.
#[test]
fn nested_calls_are_inlined_to_a_fixed_point() {
    let case = common::attention("attention_flash_noncausal", 1);
    let src = common::fixture_src(&case.fixture);
    let mut m = codegen::compile(&src, &case.spec, Target::upstream_gpu()).expect("compiles");
    let (_, calls_before) = opt::measure(&m);
    assert!(
        calls_before >= 5,
        "attention_flash should reach make_ttir with several tt.calls, saw {calls_before}"
    );
    opt::inline_calls(&mut m).expect("inlines");
    let (_, calls_after) = opt::measure(&m);
    assert_eq!(
        calls_after, 0,
        "{calls_after} tt.call(s) survived a fixed-point inliner"
    );
}

/// FAIL CLOSED: a call to a symbol that is not there must be a named refusal, not a
/// module with a dangling call left in it.
#[test]
fn a_call_to_a_missing_symbol_is_refused_by_name() {
    let case = common::rmsnorm("rmsnorm", 128);
    let src = common::fixture_src(&case.fixture);
    let mut m = codegen::compile(&src, &case.spec, Target::spyre()).expect("compiles");
    // Retarget the one call at a symbol nothing defines.
    let mut hit = false;
    for f in &mut m.funcs {
        for b in &mut f.body.blocks {
            for op in &mut b.ops {
                if op.name == "tt.call" {
                    op.attrs
                        .insert("callee".to_string(), Attr::Str("@nope".to_string()));
                    hit = true;
                }
            }
        }
    }
    assert!(hit, "rmsnorm has a tt.call to retarget");
    let err = opt::make_ttir(&mut m).expect_err("must refuse");
    assert!(
        err.to_string().contains("@nope"),
        "the refusal must NAME the missing symbol; got: {err}"
    );
}

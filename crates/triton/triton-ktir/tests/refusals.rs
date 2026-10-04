// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! A FAITHFUL PORT REPRODUCES THE REFUSALS, BY NAME.
//!
//! `vector_add` and `mul` are REFUSED by the C++ chain, and a port that ACCEPTS them
//! is wrong -- not "more permissive", wrong. Their KTIR would carry a corelet plan
//! whose two halves are `[0, 0]` and `[0, 1]`: an empty range plus a whole one, which
//! is not a partition. The emitter downstream would honour it.
//!
//! Why it happens is worth stating, because it is a hardware fact and not an
//! implementation limit: `BLOCK=64` f16 is exactly ONE 64-element stick, and 1p0 has
//! exactly TWO corelets. An even split of one stick gives the first corelet nothing.
//!
//! `bias_add_f32` fails EARLIER STILL and on every backend -- a plain module global
//! (`BIAS = 1.0`) referenced inside a `@triton.jit` kernel. That is bridge one's
//! territory, so this crate never sees its ttir; the golden records `stage.txt =
//! make_ir` and there is no `0_ttir.mlir` to feed us. The test below asserts exactly
//! that, so the absence is a recorded measurement rather than a gap.

mod common;

use common::*;
use triton_ktir::text::parse;

/// The C++ verifier's message, verbatim, as the golden records it.
/// The refusal the ONE-STICK split USED to produce, kept because the two-corelet coherence
/// rule that emits it is byte-unchanged and still has to reject this shape.
const CORELET_PLAN_REFUSAL: &str = "the two data_bounds ranges must be disjoint and \
                                    contiguous and non-empty (each lo < hi, one's hi == \
                                    the other's lo), but got [0, 0] and [0, 1]";

/// A ONE-STICK TILE REACHES KTIR AS `single_corelet`, AND OUR KTIR MATCHES THE C++'s.
///
/// # THIS TEST HAS BEEN INVERTED TWICE, AND BOTH INVERSIONS ARE THE MECHANISM WORKING
///
/// It first asserted that this port REFUSES `vector_add` and `mul`, because the C++ did:
/// `PlanCorelets` planned them as `split` over one stick, the floor split gave corelet 0 the
/// empty range `[0, 0]`, and the `ktdf.corelet_plan` verifier rejected it by name. A port
/// that accepted what the C++ refused would have been the defect, and
/// `refusal.txt` / `stage.txt` recorded the C++ saying exactly that.
///
/// The C++ then FIXED it (`PlanCorelets.cpp:456-479` and `:744`): a tile with fewer sticks
/// than corelets is INDIVISIBLE and is re-patterned to `single_corelet`. So "reproduce the
/// refusal" became "reproduce a bug" -- the one direction a faithfulness doctrine can point
/// wrong. The test then asserted the PLAN SHAPE while `refusal.txt` was still checked in, and
/// asserted it AS STALE so a regeneration could not slip past.
///
/// The goldens are now REGENERATED: `refusal.txt` and `stage.txt` are gone and `1_ktir.mlir`
/// exists (60 lines each). So this asserts the strongest available thing -- our KTIR against
/// the C++'s, field by field, the same check `tests/pure_rust_ktir.rs` applies to the eleven
/// Granite configurations -- plus the plan shape, because a diff that agreed on everything
/// EXCEPT the plan would be the interesting failure and it should be named.
#[test]
fn a_one_stick_tile_is_planned_as_single_corelet_and_matches_the_cpp() {
    for config in ["vector_add", "mul"] {
        // THE STALE RECORDS MUST BE GONE. Keeping this assertion means a half-regeneration --
        // a new `1_ktir.mlir` beside an old `refusal.txt` -- is reported rather than silently
        // preferring one of the two.
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../test/goldens/ktir")
            .join(config);
        for stale in ["refusal.txt", "stage.txt"] {
            assert!(
                !dir.join(stale).exists(),
                "{config}: `{stale}` still records the PRE-FIX refusal beside a regenerated \
                 1_ktir.mlir. The C++ now lowers this kernel, so that record is false and one \
                 of the two is being believed over the other."
            );
        }

        let mut m = parse::parse(&golden(config, "0_ttir.mlir"))
            .unwrap_or_else(|e| panic!("{config}: the ttir golden must parse: {e}"));
        // vector_add / mul launch a 1-D grid of N/BLOCK = 1024/64 = 16 blocks.
        triton_ktir::make_ktir(&mut m, &[16]).unwrap_or_else(|e| {
            panic!(
                "{config}: a one-stick tile must now be planned as single_corelet, not \
                 refused -- the C++ fix at PlanCorelets.cpp:744 is what this port follows. \
                 Got: {e}"
            )
        });

        // OUR KTIR AGAINST THE C++'s, field by field. It MATCHES -- see
        // `the_runtime_extent_path_matches_the_cpp` for what had to be fixed to get here.
        let golden_m = parse::parse(&golden(config, "1_ktir.mlir"))
            .unwrap_or_else(|e| panic!("{config}: the C++ KTIR golden must parse: {e}"));
        let mut findings = triton_ktir::text::diff::census_diff(&golden_m, &m);
        findings.extend(triton_ktir::text::diff::diff(&golden_m, &m));
        assert!(
            findings.is_empty(),
            "{config}: our KTIR differs from the C++'s in {} place(s):{}",
            findings.len(),
            findings.iter().map(|f| format!("\n    {f}")).collect::<Vec<_>>().join("")
        );

        // AND THE PLAN MUST SAY `single_corelet`, named separately: a diff that agreed on
        // everything except the plan is the interesting failure.
        let plans: Vec<&triton_ktir::ir::Op> = m
            .ops_deep()
            .into_iter()
            .filter(|o| o.kind == triton_ktir::ir::OpKind::KtdfCoreletPlan)
            .collect();
        assert_eq!(plans.len(), 1, "{config}: expected exactly one corelet plan");
        assert_eq!(
            plans[0]
                .attr(&triton_ktir::ir::AttrKey::Pattern)
                .and_then(|a| a.as_str()),
            Some("single_corelet"),
            "{config}: the plan must NAME single_corelet -- a `split` plan carrying one \
             corelet is still rejected by the unchanged coherence rule"
        );
        let corelets = &plans[0].regions[0].ops;
        assert_eq!(
            corelets.len(),
            1,
            "{config}: single_corelet means ONE ktdf.corelet, got {}",
            corelets.len()
        );
        assert_eq!(
            corelets[0]
                .attr(&triton_ktir::ir::AttrKey::DataBounds)
                .and_then(|a| a.as_int_list()),
            Some(&[0i64, 1][..]),
            "{config}: the whole one-stick tile goes to corelet 0, so data_bounds is [0, 1]"
        );
    }
}

/// THE COHERENCE RULE WAS NOT RELAXED, and this is the control that says so. The fix is a
/// new plan SHAPE; the two-corelet rule still rejects the old malformed bounds, and still
/// rejects a `split` plan that claims one corelet.
#[test]
fn the_two_corelet_rule_still_rejects_the_old_shape() {
    use triton_ktir::passes::plan_corelets::{verify_plan, Pattern};
    let split_one_stick = triton_ktir::passes::plan_corelets::corelets_for_test(1);
    let e = verify_plan(Pattern::Split, &split_one_stick)
        .expect_err("a `split` plan over one stick is still malformed");
    assert!(
        e.message.contains(CORELET_PLAN_REFUSAL),
        "the unchanged rule must still refuse [0,0] + [0,1] by name; got: {e}"
    );
}

/// THE CONTROL FOR THE REFUSAL: two sticks split cleanly, so the refusal is about the
/// ONE-STICK case and not about elementwise kernels in general. Without this the test
/// above would also pass for a port that refuses everything.
#[test]
fn the_same_kernel_at_two_sticks_is_accepted() {
    // `vector_add`'s ttir with BLOCK widened to 128 -- two sticks, so the split is
    // [0, 1] and [1, 2], which the verifier accepts.
    let ttir = golden("vector_add", "0_ttir.mlir")
        .replace("64xf16", "128xf16")
        .replace("%c64_i32 = arith.constant 64 : i32", "%c64_i32 = arith.constant 128 : i32");
    let mut m = parse::parse(&ttir).expect("the widened ttir must parse");
    triton_ktir::make_ktir(&mut m, &[8]).unwrap_or_else(|e| {
        panic!(
            "TWO sticks must be accepted, or the one-stick refusal is not what it \
             claims to be about: {e}"
        )
    });
    // And the plan it produced is the clean split.
    use triton_ktir::ir::*;
    let plan = m
        .ops_deep()
        .into_iter()
        .find(|o| o.kind == OpKind::KtdfCoreletPlan)
        .expect("a plan is inserted");
    assert_eq!(plan.attr(&AttrKey::Pattern).and_then(|a| a.as_str()), Some("split"));
    let cs = &plan.regions[0].ops;
    assert_eq!(cs[0].attr(&AttrKey::DataBounds), Some(&Attr::IntList(vec![0, 1])));
    assert_eq!(cs[1].attr(&AttrKey::DataBounds), Some(&Attr::IntList(vec![1, 2])));
}

/// `bias_add_f32` never reaches this crate: it dies in bridge one.
#[test]
fn bias_add_f32_fails_before_bridge_two_and_that_is_recorded() {
    assert_eq!(
        golden("bias_add_f32", "stage.txt").trim(),
        "make_ir",
        "bias_add_f32 must fail at make_ir -- a module global inside a jitted kernel"
    );
    assert!(
        !has("bias_add_f32", "0_ttir.mlir"),
        "if bias_add_f32 ever produces ttir, bridge two must be given a target for it \
         rather than this test continuing to assert its absence"
    );
    let recorded = golden("bias_add_f32", "refusal.txt");
    assert!(
        recorded.contains("CompilationError"),
        "the recorded failure should be Triton's own front-end error:\n{recorded}"
    );
}

/// FAIL CLOSED, NEVER PARTIALLY. Every refusal this crate can produce names the pass
/// that made it, so a diagnostic is actionable rather than "compilation failed".
#[test]
fn every_refusal_names_its_pass_and_says_something() {
    use triton_ktir::Refusal;
    // A representative set, one per pass that can refuse.
    let cases: Vec<Refusal> = vec![
        // A raw-pointer tile access.
        {
            let mut m = parse::parse(
                "module {\n  tt.func public @k(%p: !tt.ptr<f16>) attributes {noinline = false} {\n    %x = tt.load %p : tensor<64xf16>\n    tt.return\n  }\n}\n",
            )
            .unwrap();
            triton_ktir::make_ktir(&mut m, &[1]).unwrap_err()
        },
        // A multi-axis grid with no extents.
        {
            let mut m = parse::parse(
                "module {\n  tt.func public @k(%p: !tt.ptr<f16>) attributes {noinline = false} {\n    %a = tt.get_program_id x : i32\n    %b = tt.get_program_id y : i32\n    tt.return\n  }\n}\n",
            )
            .unwrap();
            triton_ktir::make_ktir(&mut m, &[]).unwrap_err()
        },
        // Genuine f32 compute.
        {
            let mut m = parse::parse(
                "module {\n  tt.func public @k(%p: !tt.ptr<f16>) attributes {noinline = false} {\n    %a = arith.constant dense<1.000000e+00> : tensor<64xf32>\n    %b = arith.addf %a, %a : tensor<64xf32>\n    tt.return\n  }\n}\n",
            )
            .unwrap();
            triton_ktir::make_ktir(&mut m, &[1]).unwrap_err()
        },
    ];
    for r in cases {
        assert!(!r.pass.is_empty(), "a refusal with no pass name: {r:?}");
        assert!(
            r.message.len() > 40,
            "a refusal that does not explain itself: {r}"
        );
        // And it renders as `pass: message`.
        assert!(r.to_string().starts_with(r.pass), "{r}");
    }
}

/// THE RUNTIME-EXTENT PATH, and it matches the C++ now. This test is the third inversion in
/// this file, and each one was the guard firing rather than a change of mind.
///
/// `vector_add` and `mul` are the ONLY configurations in the tree whose descriptor has a
/// RUNTIME extent -- `tl.make_tensor_descriptor(a_ptr, [n], [1], [BLOCK])` with `n` a kernel
/// argument. Every Granite extent is constexpr, so this path is unverified by all eleven of
/// them, which is why it is worth more than its size suggests.
///
/// THREE THINGS WERE WRONG, all on that path, and all found the moment these two kernels
/// became comparable at all (they used to be refused, so nothing had ever diffed them):
///
/// 1. A DYNAMIC EXTENT IS AN OPERAND, NOT A SENTINEL. The C++ emits an `arith.index_cast` of
///    the runtime shape value and passes it to the view; this port recorded `kDynamic` in the
///    static `sizes` attribute and emitted no cast -- 7 casts against 4. The static list then
///    carries only the extents that ARE static, which for `vector_add` is none.
/// 2. THE READER WAS DROPPING THOSE OPERANDS. `parse_construct_memory_view` stopped scanning
///    at `sizes:`, so a `%token` INSIDE `sizes: [%a_desc_4]` was lost. Same class as
///    `construct_indirect_access_tile` being typed with its index vector's type: the reader
///    silently loses a real operand and nothing errors.
/// 3. MLIR NORMALISES THE AFFINE EXPRESSION. `s0 - 1 - d0` prints as `-d0 + s0 - 1`, negated
///    dim first, and the diff compares the printed body.
///
/// The assertion that the difference was ONLY OMISSION -- no op we emitted was absent from the
/// C++'s -- is what made closing it safe rather than a rewrite, so it is kept below.
#[test]
fn the_runtime_extent_path_matches_the_cpp() {
    let mut m = parse::parse(&golden("vector_add", "0_ttir.mlir")).expect("parses");
    triton_ktir::make_ktir(&mut m, &[16]).expect("lowers");
    let golden_m = parse::parse(&golden("vector_add", "1_ktir.mlir")).expect("parses");

    let casts = |x: &triton_ktir::ir::Module| {
        x.ops_deep()
            .iter()
            .filter(|o| o.kind == triton_ktir::ir::OpKind::ArithIndexCast)
            .count()
    };
    assert_eq!(
        casts(&m),
        casts(&golden_m),
        "one `arith.index_cast` per runtime shape value -- this count is the signature the \
         gap was pinned by, kept so a regression reads as the same number moving back"
    );
    // Still asserted: no op we emit is absent from the C++'s. A real miscompile here would not
    // look like the omission this used to be.
    let theirs: Vec<&str> = golden_m.ops_deep().iter().map(|o| o.kind.spelling()).collect();
    for op in m.ops_deep() {
        assert!(
            theirs.contains(&op.kind.spelling()),
            "we emit `{}`, which the C++ does not",
            op.kind.spelling()
        );
    }
    // And the dynamic extent must be an OPERAND of the view, with the static list empty.
    let view = m
        .ops_deep()
        .into_iter()
        .find(|o| o.kind == triton_ktir::ir::OpKind::KtdpConstructMemoryView)
        .expect("a memory view");
    assert_eq!(
        view.operands.len(),
        2,
        "the base plus one runtime extent; a sentinel in the static list is not an extent"
    );
    assert_eq!(
        view.attr(&triton_ktir::ir::AttrKey::Shape).and_then(|a| a.as_int_list()),
        Some(&[][..]),
        "a fully dynamic rank-1 view has an EMPTY static `sizes` list"
    );
}

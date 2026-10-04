//! THE FUSION EXPERIMENT, machine-checked: does the bridge accept a fused multi-layer kernel?
//!
//! Needs the `ruff` feature: every case starts from Python SOURCE.
#![cfg(feature = "ruff")]
//!
//! `tests/golden_diff.rs` already proves both `decoder_block.py` kernels match their raw
//! goldens field by field. This file records the NUMBERS the experiment was run for, as
//! assertions rather than prose, plus the one structural fact that is easy to state wrongly.
//!
//! MEASURED, at `M = 64`, `D_MODEL = 128`, `D_FF = 256`:
//!
//! | | one layer | two layers, fused |
//! |---|---|---|
//! | raw ops (`make_ir`, bridge one's oracle) | 287 | 329 |
//! | kernel arguments after descriptor flattening | 14 | 23 |
//! | widest function boundary | 57 | 57 |
//! | ops after `make_ttir` (inliner, canonicalizer, CSE) | 143 | 260 |
//! | this bridge | MATCHES its golden | MATCHES its golden |
//!
//! ⚠️ THE RAW COUNTS WERE 278 / 320 AND THE POST COUNTS 134 / 242, AND WHAT MOVED THEM IS THE
//! WEIGHT ORIENTATION, NOT A FUSION CHANGE. All three deltas are +9 ops in the LAYER BODY: the
//! fixture now presents every 2-D weight `[out, in]` (`torch.nn.Linear.weight`'s own order) and
//! each `tl.dot` operand carries `.T`, so the raw module gained nine `tt.trans` --
//! `decoder_layer.ttir_raw.mlir` goes from 2 of them to 11, the two survivors being the score
//! `k.T`s that were always there. +9 on one layer, +9 on two (the body is emitted ONCE), and the
//! post-pass +9 / +18 because the inliner then duplicates the body. Every number here was
//! re-measured from source, and the `.ttir_raw.mlir` goldens were regenerated in the same change
//! for the same reason.
//!
//! THE FACT THAT IS EASY TO GET WRONG: two layers is +42 raw ops, not +287, because the layer
//! body is emitted ONCE and CALLED TWICE -- both calls mangle to the same symbol, since the
//! constexprs are identical and only the descriptor VALUES differ. The 42 are the second
//! layer's nine descriptors with their shape/stride constants, plus one more `tt.call`. So a
//! raw-op count is NOT a measure of the work in a fused kernel; the post-pass count is, and
//! there the inliner duplicates the body (143 -> 260).

mod common;

use triton_frontend::codegen;
use triton_frontend::target::Target;
use triton_frontend::ttir::{Attr, Module};

fn compile(case: &common::Case) -> Module {
    let src = common::fixture_src(&case.fixture);
    codegen::compile(&src, &case.spec, Target::spyre())
        .unwrap_or_else(|e| panic!("`{}` must compile: {e}", case.name))
}

fn total_ops(m: &Module) -> usize {
    fn count_region(r: &triton_frontend::ttir::Region) -> usize {
        r.blocks
            .iter()
            .map(|b| {
                b.ops
                    .iter()
                    .map(|o| 1 + o.regions.iter().map(count_region).sum::<usize>())
                    .sum::<usize>()
            })
            .sum()
    }
    m.funcs.iter().map(|f| count_region(&f.body)).sum()
}

fn calls_to(m: &Module, needle: &str) -> usize {
    fn walk(r: &triton_frontend::ttir::Region, needle: &str) -> usize {
        r.blocks
            .iter()
            .flat_map(|b| b.ops.iter())
            .map(|o| {
                let here = if o.name == "tt.call" {
                    match o.attrs.get("callee") {
                        Some(Attr::Str(s)) if s.contains(needle) => 1,
                        _ => 0,
                    }
                } else {
                    0
                };
                here + o.regions.iter().map(|rg| walk(rg, needle)).sum::<usize>()
            })
            .sum()
    }
    m.funcs.iter().map(|f| walk(&f.body, needle)).sum()
}

/// The headline: the bridge accepts the fused kernel, and the counts are what they are.
#[test]
fn the_fused_two_layer_kernel_compiles_with_the_measured_counts() {
    let one = compile(&common::decoder_layer());
    let two = compile(&common::decoder_two_layers());
    assert_eq!(total_ops(&one), 287, "one layer's raw op count");
    assert_eq!(total_ops(&two), 329, "two fused layers' raw op count");
    assert_eq!(one.funcs[0].arg_types.len(), 14, "one layer's kernel arguments");
    assert_eq!(two.funcs[0].arg_types.len(), 23, "two layers' kernel arguments");
    // ⭐ THE +42 IS NOW ASSERTED, not only stated. It is the whole point of the table above, and
    // it is the one number that stays put when the layer BODY changes -- as it just did, by nine
    // `tt.trans` -- because both sides gain the same ops. A future body change that moved this
    // difference would mean the second layer was no longer a bare call.
    assert_eq!(
        total_ops(&two) - total_ops(&one),
        42,
        "fusing a second layer costs the second layer's nine descriptors with their \
         shape/stride constants plus one more tt.call -- NOT another copy of the body"
    );
    // ⭐ AND THE POST-PASS COUNTS, which the header used to state in prose only. They are what
    // the inliner does to the shared body, so leaving them unasserted is how the table above
    // came to carry two stale numbers (134 / 242) that nothing could fail on.
    for (name, m, want) in [("one layer", one, 143usize), ("two layers", two, 260)] {
        let mut m = m;
        triton_frontend::opt::make_ttir(&mut m).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(total_ops(&m), want, "{name}'s op count after make_ttir");
    }
}

/// The layer body is ONE function called TWICE -- not two copies. Both calls mangle to the
/// same symbol because the constexprs are identical and only the descriptor values differ.
///
/// This is the assertion that makes the +42 number readable. Without it, "320 vs 278" invites
/// the wrong conclusion that fusing a layer costs 42 ops.
#[test]
fn the_two_layers_share_one_emitted_layer_function() {
    let two = compile(&common::decoder_two_layers());
    let bodies: Vec<&String> = two
        .funcs
        .iter()
        .map(|f| &f.name)
        .filter(|n| n.contains("_decoder_layer__"))
        .collect();
    assert_eq!(
        bodies.len(),
        1,
        "the layer body should be emitted ONCE; got {bodies:?}"
    );
    assert_eq!(
        calls_to(&two, "_decoder_layer__"),
        2,
        "and called TWICE"
    );
    // The flattened boundary: x plus twelve descriptors -- ten of rank 2 (1 + 2*2 each) and
    // two of rank 1 (1 + 2*1 each) -- is 1 + 50 + 6 = 57.
    let layer = two
        .funcs
        .iter()
        .find(|f| f.name.contains("_decoder_layer__"))
        .unwrap();
    assert_eq!(
        layer.arg_types.len(),
        57,
        "the layer's flattened argument count: 1 tensor + 10 rank-2 descriptors (5 each) \
         + 2 rank-1 descriptors (3 each)"
    );
}

/// The one-layer kernel is a PREFIX of the fused one in the ops that matter: same layer body,
/// same generated `standard.*` helpers, one fewer call.
///
/// Stated as a test because the per-layer kernel and the fused kernel come from ONE source, and
/// that only stays true if nothing in the fused path needs a different body.
#[test]
fn both_kernels_use_the_same_layer_body_and_helpers() {
    let one = compile(&common::decoder_layer());
    let two = compile(&common::decoder_two_layers());
    let names = |m: &Module| -> Vec<String> {
        let mut v: Vec<String> = m.funcs.iter().skip(1).map(|f| f.name.clone()).collect();
        v.sort();
        v
    };
    assert_eq!(
        names(&one),
        names(&two),
        "the two kernels must share every generated function, layer body included"
    );
    assert_eq!(calls_to(&one, "_decoder_layer__"), 1);
    assert_eq!(calls_to(&two, "_decoder_layer__"), 2);
}

//! `argmax_first_max` through the WHOLE frontend value path, gated on structure.
//!
//! The real-Triton goldens (`.ttir_raw.mlir` / `.ttir.mlir`) for this fixture do not exist
//! yet -- producing them needs the triton toolchain, and until that run happens this test
//! is the gate: `codegen::compile` from Python source, `opt::make_ttir` on the value, and
//! the OP CENSUS asserted -- the algorithm's every link must be present as the generic op
//! that carries it, and nothing argmax-specific may exist below the kernel. When the
//! goldens land, this file grows the `diff::compare` halves and `make_ttir.rs`/`
//! `fixture_status.rs` take the case into their tables; until then a refusal here fails
//! loudly instead of a golden diff quietly never running.
//!
//! THE ALGORITHM'S OP FOOTPRINT, pinned (see the fixture's docstring for the arithmetic):
//!
//!   * `tt.reduce` × 4 — the two passes' per-stick maxima (pass 1 fp16 over logits,
//!     pass 2 i32 over `w`), each within ONE stick (the reduce-MAX single-stick law);
//!   * `arith.cmpf` — the equality mask;
//!   * `arith.extui` — the mask's i1 → i32 widening (the index math is INTEGER: fp16
//!     exact integers stop at 2048 and the device's f16 legalization refuses genuine
//!     f32 compute, so the ramp product runs in i32 lanes — exact at any vocab width);
//!   * `arith.muli` — mask × ramp (the ramp itself is LOADED from the staged table, the
//!     attention-mask precedent: no fixture adds a runtime scalar to an arange);
//!   * `arith.maxsi` — the integer running combine (and the reduce region's combiner);
//!   * `arith.subi` — the final `V - j`;
//!   * `tt.expand_dims`/`tt.broadcast` — the explicit [1] <-> [1, BLOCK] seams (the golden
//!     shapes: rank-2 tiles, rank-1 reduce results, no scalar anywhere, so no
//!     reduce-to-scalar and no splat of an interior value exists in the kernel);
//!   * `tt.descriptor_load`/`tt.descriptor_store` — the row and ramp in, the i32 stick out.
//!
//! WHAT MUST NOT BE PRESENT: any op whose existence is argmax-specific. The whole point of
//! the ladder form is that every op above is the same op `rmsnorm`/`attention`/`moe` use.

mod common;

use triton_frontend::target::Target;
use triton_frontend::{codegen, opt};

/// Compiles the fixture and returns (module, op-name census).
fn compile_argmax() -> (triton_frontend::ttir::Module, Vec<String>) {
    let case = common::argmax_first_max();
    let src = common::fixture_src(&case.fixture);
    let mut m = codegen::compile(&src, &case.spec, Target::spyre())
        .unwrap_or_else(|e| panic!("argmax_first_max did not compile: {e}"));
    let stats = opt::make_ttir_measured(&mut m)
        .unwrap_or_else(|e| panic!("argmax_first_max: make_ttir refused: {e}"));
    // The bridge-two contract, same as every `make_ttir.rs` case: no calls survive, and
    // exactly one function -- `triton_ktir::ir::Module::kernel()` refuses anything else.
    assert_eq!(stats.calls_after, 0, "tt.call survived make_ttir");
    assert_eq!(
        stats.funcs_after, 1,
        "symbol-dce left {} functions",
        stats.funcs_after
    );

    // THE WALK DESCENDS INTO REGIONS: the two passes live inside `scf.for` bodies, so a
    // top-level-only walk would count the loop shells and miss every op that matters.
    // (This exact mistake is why this walk exists as a named helper rather than an inline
    // loop -- an earlier version asserted on a region-blind census and "found" empty
    // bodies that were full.)
    let mut ops = Vec::new();
    fn walk_ops(ops: &mut Vec<String>, blocks: &[triton_frontend::ttir::Block]) {
        for b in blocks {
            for op in &b.ops {
                ops.push(op.name.clone());
                for r in &op.regions {
                    walk_ops(ops, &r.blocks);
                }
            }
        }
    }
    for f in &m.funcs {
        walk_ops(&mut ops, &f.body.blocks);
    }
    (m, ops)
}

fn count(ops: &[String], name: &str) -> usize {
    ops.iter().filter(|o| o.as_str() == name).count()
}

#[test]
fn argmax_first_max_compiles_and_states_the_algorithm_in_generic_ops() {
    let (_m, ops) = compile_argmax();
    // The census, printed for `--nocapture` runs: one line, sorted, so a changed lowering
    // shows exactly which op moved rather than failing an opaque count.
    let mut census = ops.clone();
    census.sort();
    census.dedup();
    eprintln!("argmax_first_max op census: {census:?}");
    eprintln!("argmax_first_max op sequence: {ops:?}");

    // Every link of the algorithm, as the generic op that carries it. One `tt.reduce`
    // PER LOOP BODY: pass 1's fp16 max over the logits, pass 2's i32 max over `w`.
    assert!(
        count(&ops, "tt.reduce") >= 2,
        "the two passes' per-stick maxima"
    );
    assert!(count(&ops, "arith.cmpf") >= 1, "the equality mask");
    assert!(
        count(&ops, "arith.extui") >= 1,
        "the mask's i1 -> i32 widening"
    );
    assert!(count(&ops, "arith.muli") >= 1, "mask * ramp");
    assert!(
        count(&ops, "arith.maxsi") >= 1,
        "the integer running combine"
    );
    assert!(count(&ops, "arith.subi") >= 1, "V - j");
    assert!(
        count(&ops, "tt.descriptor_load") >= 3,
        "the row loads (x twice, ramp once)"
    );
    assert!(
        count(&ops, "tt.descriptor_store") >= 1,
        "the i32 stick store"
    );
    assert!(count(&ops, "tt.expand_dims") >= 2, "the [1] -> [1,1] seams");
    assert!(
        count(&ops, "tt.broadcast") >= 2,
        "the [1,1] -> [1,BLOCK] seams"
    );
    // AND NO SCALAR REDUCE / INTERIOR SPLAT: the shapes are the golden forms, so a scalar
    // sneaking back in (a reduce-to-scalar, a splat of an interior value) is a lowering
    // regression the numeric legs would refuse -- name it here, one rung earlier.
    assert!(
        !ops.iter().any(|o| o == "tt.splat"),
        "a tt.splat survived: the kernel states no runtime-argument splat, so this is a \
         splat of an interior value -- the form to_ktir refuses by design"
    );

    // And the loops that carry the two passes (static trip counts).
    assert!(
        count(&ops, "scf.for") >= 2,
        "the two passes' loops -- got {} scf.for",
        count(&ops, "scf.for")
    );
}

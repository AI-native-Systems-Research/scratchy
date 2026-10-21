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
//!   * `tt.reduce` × 5 — the whole-row maxima: the global max `m`, the first-block
//!     combine `jA`, the first-lane combine `jB`, and the two read-out combines `b*`
//!     and `l*`. Each is the golden rank-2 → rank-1 reduce over the [1, PAD_V] tile's
//!     second axis, and the WHOLE ROW IS ONE TILE (the rmsnorm whole-row block), so
//!     there is NO per-stick loop to unroll — the program the door receives is
//!     STRAIGHT-LINE, ~20 ops, which is the door-lowerable form (a per-stick-loop
//!     kernel would splice to ~2300 trips);
//!   * `arith.maxnumf`/`arith.truncf` — each f16 reduce runs in f32 (upstream's own
//!     promotion) and narrows straight back, moe.py's spelling; the max of f16 values
//!     is exactly representable, so the round trip is exact;
//!   * `arith.cmpf` × 3 — the equality mask `(x == m)`, the block one-hot
//!     `(bramp == jA)`, and the lane one-hot `(lramp == jB)`;
//!   * `arith.uitofp` × 3 — the i1 masks' → f16 widening (the 0/1 selectands);
//!   * `arith.mulf` — `mask · bramp`, `mask · sel · lramp`, the one-hot `sel · selL`,
//!     the two read-outs `onehot · binc`/`onehot · linc`, and the two `ones`-table
//!     widens of the [1] answers to the [1, BLOCK] output stick;
//!   * ⛔ NO `arith.subf` ANYWHERE — the v5 form inverted the ramps by subtraction
//!     (`NB - jA`, `BLOCK - jB`), but those `[1]`-shaped `splat - tensor` subs have no
//!     whole-stick door emission, so the inversion rides the EQUALITY selection
//!     instead (the increasing 0-based tables read the numbers back out through the
//!     one-hot). A subf sneaking back in is the unlowerable form returning;
//!   * `arith.extf`/`arith.addf`/`arith.fptosi` — the pad law's `x + padoff`
//!     (the f16 offset-row add before the first max) and the ONE fp32 island:
//!     the final `b*64 + l*` reaches 49154 (past fp16's exact range), so it
//!     widens, multiply-adds in f32 (exact below 2^24), and casts to i32 —
//!     moe.py's gelu island and the KSPLIT pair's own device recipe;
//!   * `tt.descriptor_load` × 8 / `tt.descriptor_store` — the row, the four
//!     staged index tables, the two widen tables and the pad law's offset
//!     row in, the i32 stick out;
//!   * `tt.expand_dims`/`tt.broadcast` — the [1] → [1,1] → [1, BLOCK] seams (the golden
//!     shapes: rank-2 tiles, rank-1 reduce results, no scalar anywhere, so no
//!     reduce-to-scalar and no splat of an interior value exists in the kernel).
//!
//! WHAT MUST NOT BE PRESENT: any op whose existence is argmax-specific, any `scf.for`
//! (the straight-line law above), and any `tt.splat`. The whole point of the ladder
//! form is that every op above is the same op `rmsnorm`/`attention`/`moe` use.

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

    // Every link of the algorithm, as the generic op that carries it. FIVE whole-row
    // reduces: the global max, the first-block combine, the first-lane combine, and
    // the two read-out combines.
    assert!(count(&ops, "tt.reduce") >= 5, "the five whole-row maxima");
    assert!(
        count(&ops, "arith.maxnumf") >= 5,
        "the reduces' f32 combine (the promotion)"
    );
    assert!(
        count(&ops, "arith.truncf") >= 5,
        "the reduces' narrow back to f16"
    );
    assert!(
        count(&ops, "arith.extf") >= 2,
        "the fp32 island's two widens"
    );
    assert!(count(&ops, "arith.cmpf") >= 3, "the three equality masks");
    assert!(
        count(&ops, "arith.uitofp") >= 3,
        "the masks' i1 -> f16 widening"
    );
    assert!(
        count(&ops, "arith.mulf") >= 8,
        "mask * ramp, the one-hot, the read-outs, and the ones-table widens"
    );
    // ⛔ AND NO SUBTRACTION: the inversion rides the equality selection, so a subf
    // reappearing is the v5 form (an unlowerable [1]-shaped splat-minus-tensor) back.
    assert_eq!(
        count(&ops, "arith.subf"),
        0,
        "an arith.subf appeared: the kernel inverts through the equality selection, so a \
         subtraction is the v5 `NB - jA` form returning -- a [1]-shaped splat-minus-tensor \
         with no whole-stick door emission"
    );
    assert!(
        count(&ops, "arith.addf") >= 2,
        "the pad law's x + padoff and the island's b*64 + l*"
    );
    assert!(
        count(&ops, "arith.fptosi") >= 1,
        "the island's f32 -> i32 cast"
    );
    assert!(
        count(&ops, "tt.descriptor_load") >= 8,
        "the row loads (x, bramp, lramp, binc, linc, ones, sixtyfour, padoff)"
    );
    assert!(
        count(&ops, "tt.descriptor_store") >= 1,
        "the i32 stick store"
    );
    assert!(count(&ops, "tt.expand_dims") >= 3, "the [1] -> [1,1] seams");
    assert!(
        count(&ops, "tt.broadcast") >= 3,
        "the [1,1] -> [1,BLOCK]/[1,PAD_V] seams"
    );
    // AND NO SCALAR REDUCE / INTERIOR SPLAT: the shapes are the golden forms, so a scalar
    // sneaking back in (a reduce-to-scalar, a splat of an interior value) is a lowering
    // regression the numeric legs would refuse -- name it here, one rung earlier.
    assert!(
        !ops.iter().any(|o| o == "tt.splat"),
        "a tt.splat survived: the kernel states no runtime-argument splat, so this is a \
         splat of an interior value -- the form to_ktir refuses by design"
    );

    // AND STRAIGHT-LINE, NO LOOPS: the whole row is one tile, so no scf.for exists to
    // unroll -- the door receives ~20 generic ops, not ~2300 spliced trips. A loop
    // sneaking back in is the unroll bomb the door cannot afford.
    assert!(
        !ops.iter().any(|o| o == "scf.for"),
        "an scf.for survived: the kernel states whole-row tiles, so a loop is a lowering \
         regression -- and the trip-unroll would splice it into thousands of descriptors"
    );
}

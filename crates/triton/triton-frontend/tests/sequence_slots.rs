//! THE SEQUENCE SLOT — `(x,) * N` and `accs[s] = ...`, the two compile-time tuple forms
//! the split-slab attention kernels accumulate through.
//!
//! # WHY THEY EXIST
//!
//! A head dim past one pool stick splits its V windows into per-slab blocks, and the split
//! accumulator is the only spelling that needs no transpose, no `tl.where` landing and no
//! `tl.arange` — the last of which is not a preference: `tl.arange` carries a hard
//! power-of-two gate, so a slab id built from it REFUSES at `SLABS == 3` (head dim 192, two
//! census arms). The tuple forms are pure Python-level value plumbing, exactly the class of
//! thing upstream Triton's interpreter-shaped frontend gives kernels for free
//! (`accs = [tl.zeros(...)] * SLABS; accs[s] = ...` is a known upstream idiom); these tests
//! pin this front end's value-level twins of it.
//!
//! # THE TWO LAWS
//!
//! * `(x,) * N` is a COMPILE-TIME SEQUENCE REPEAT — Python's own tuple-spelling of "N slots
//!   of this". It never reaches the builder: the repeated slots hold the SAME SSA value, so
//!   a repeat costs one `tl.zeros` however large `N` is. Python's edge semantics hold: a
//!   negative count is the empty sequence.
//! * `accs[s] = v` is a REBIND, never a scatter: the index must lower to a compile-time
//!   integer (the `tl.static_range` unroller binds its variable as one), the base must
//!   resolve to a `Val::Seq`, and the assignment rebuilds the whole sequence with one slot
//!   replaced. A TENSOR base is refused by name — Triton has no tensor item assignment, and
//!   a silent reinterpretation of one would be exactly the wrong-answer-no-diagnostic class
//!   the slice census guards against.

mod common;

use triton_frontend::codegen;
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;
use triton_frontend::ttir::Op;

/// The split-accumulator body at `SLABS` slots: one load, one FMA and one store per slab,
/// each through its own column window — the decode kernel's per-slab pattern reduced to
/// the two forms under test.
fn kernel(body: &str) -> String {
    format!(
        r#"
import triton
import triton.language as tl


@triton.jit
def seq_kernel(desc_x, desc_o, M: tl.constexpr, D: tl.constexpr,
               SLABS: tl.constexpr, STK: tl.constexpr):
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, D], strides=[D, 1],
                                       block_shape=[M, STK])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, D], strides=[D, 1],
                                       block_shape=[M, STK])
    {body}
"#
    )
}

const BODY: &str = r#"
    accs = (tl.zeros([M, STK], tl.float16),) * SLABS
    for s in tl.static_range(SLABS):
        x = x_desc.load([0, s * STK])
        accs[s] = accs[s] + x
    for s in tl.static_range(SLABS):
        o_desc.store([0, s * STK], accs[s])
"#;

fn spec(slabs: i128) -> codegen::KernelSpec {
    let mut s = common::simple_spec(
        "seq_kernel",
        &[
            ("desc_x", "*fp16"),
            ("desc_o", "*fp16"),
            ("M", "constexpr"),
            ("D", "constexpr"),
            ("SLABS", "constexpr"),
            ("STK", "constexpr"),
        ],
    );
    for (k, v) in [("M", 64), ("D", 192), ("SLABS", slabs), ("STK", 64)] {
        s.constexprs.insert(k.to_string(), Val::Int(v));
    }
    s
}

fn ops(src: &str, slabs: i128) -> Vec<Op> {
    let m = codegen::compile(src, &spec(slabs), Target::spyre())
        .unwrap_or_else(|e| panic!("the split accumulator must compile: {e}"));
    m.funcs[0].body.blocks[0].ops.clone()
}

/// ⭐ THE NON-POWER-OF-TWO SLAB COUNT COMPILES — `SLABS == 3` is head dim 192's count, the
/// one an `tl.arange`-based landing refuses at the power-of-two gate. The tuple forms have
/// no range primitive anywhere in them, so the census's two hd-192 arms are servable.
#[test]
fn the_split_accumulator_compiles_at_a_non_power_of_two_slab_count() {
    let ops = ops(&kernel(BODY), 3);
    let named = |n: &str| -> Vec<&Op> { ops.iter().filter(|o| o.name == n).collect() };
    // One load, one add and one store per slab — the static unroll spelled each slot's
    // trip exactly once, and the repeat cost ONE zeros however many slots there are.
    assert_eq!(named("tt.descriptor_load").len(), 3, "one load per slab");
    assert_eq!(named("arith.addf").len(), 3, "one FMA per slab");
    assert_eq!(named("tt.descriptor_store").len(), 3, "one store per slab");
    // ⭐ THE REPEAT SHARES ONE SSA VALUE ACROSS SLOTS: `tl.zeros` is `@jit` upstream
    // (`standard.py:120`), so it lowers to ONE `tt.call` of a generated private func —
    // here exactly one, however many slots the repeat built. A repeat that re-traced its
    // element would emit one call per slot (three here), and this count is what catches it.
    assert_eq!(
        named("tt.call").len(),
        1,
        "the repeat holds ONE zeros SSA value across every slot"
    );
    // ⭐ THE REBIND IS PER-SLOT: every store consumes a DIFFERENT add's result, and every
    // add consumes a DIFFERENT load's result — a slot whose assignment had silently
    // written the wrong slot (or the same slot thrice) would collapse these sets.
    let adds = named("arith.addf");
    let loads = named("tt.descriptor_load");
    let stores = named("tt.descriptor_store");
    let add_results: std::collections::HashSet<_> = adds.iter().map(|o| o.results[0]).collect();
    let load_results: std::collections::HashSet<_> = loads.iter().map(|o| o.results[0]).collect();
    // `tt.descriptor_store`'s operands are [desc, offset..., value] — the value is LAST.
    let stored: std::collections::HashSet<_> = stores
        .iter()
        .map(|o| *o.operands.last().expect("a store's value operand"))
        .collect();
    assert_eq!(stored, add_results, "each slab stores its own add's result");
    // Each add's non-zeros operand is a distinct load result.
    for a in &adds {
        let from_load = a
            .operands
            .iter()
            .filter(|&&v| load_results.contains(&v))
            .count();
        assert_eq!(
            from_load, 1,
            "each slot's FMA reads exactly one slab's load — slot mixing would read two"
        );
    }
    assert_eq!(add_results.len(), 3, "three distinct add results");
    assert_eq!(load_results.len(), 3, "three distinct load results");
}

/// The degenerate control: at ONE slab the pattern is the un-split kernel — one load, one
/// add, one store, the whole row one column window.
#[test]
fn one_slab_is_the_unsplit_kernel() {
    let ops = ops(&kernel(BODY), 1);
    assert_eq!(
        ops.iter()
            .filter(|o| o.name == "tt.descriptor_store")
            .count(),
        1
    );
    assert_eq!(ops.iter().filter(|o| o.name == "arith.addf").count(), 1);
}

/// Python's own edge semantics: a negative count is the EMPTY sequence, and a slot use on
/// one is then out of range — the same error the read side gives, never a wrap.
#[test]
fn a_negative_repeat_is_the_empty_sequence() {
    let src = kernel(
        r#"
    accs = (tl.zeros([M, STK], tl.float16),) * -1
    accs[0] = accs[0] + x_desc.load([0, 0])
    o_desc.store([0, 0], accs[0])
"#,
    );
    let err = codegen::compile(&src, &spec(3), Target::spyre())
        .expect_err("an empty sequence has no slot 0");
    assert!(
        err.to_string().contains("out of range"),
        "the refusal must name the range; got: {err}"
    );
}

/// A RUNTIME index is refused by name — the slot is a compile-time fact of the unroll, and
/// a runtime-indexed rebind would have to lower to a scatter this front end does not state.
/// The census's own spelling of a runtime index is a plain `range` induction, which the
/// loop lowering binds as a TENSOR (`Val::Ir` of the loop's placeholder).
#[test]
fn a_runtime_index_is_refused_by_name() {
    let src = kernel(
        r#"
    accs = (tl.zeros([M, STK], tl.float16),) * SLABS
    x = x_desc.load([0, 0])
    for i in range(0, M):
        accs[i] = x
    o_desc.store([0, 0], accs[0])
"#,
    );
    let err = codegen::compile(&src, &spec(3), Target::spyre())
        .expect_err("a runtime (tensor) index must refuse");
    assert!(
        err.to_string().contains("compile-time integer"),
        "the refusal must name the law; got: {err}"
    );
}

/// A TENSOR base is refused by name — Triton has no tensor item assignment.
#[test]
fn a_tensor_subscript_assignment_is_refused_by_name() {
    let src = kernel(
        r#"
    x = x_desc.load([0, 0])
    x[0] = x + x
    o_desc.store([0, 0], x)
"#,
    );
    let err = codegen::compile(&src, &spec(3), Target::spyre())
        .expect_err("a tensor is not assignable through a subscript");
    assert!(
        err.to_string()
            .contains("not assignable through a subscript"),
        "the refusal must name the law; got: {err}"
    );
}

/// An out-of-range slot is refused — never wrapped, never silently dropped.
#[test]
fn an_out_of_range_slot_is_refused() {
    let src = kernel(
        r#"
    accs = (tl.zeros([M, STK], tl.float16),) * SLABS
    accs[3] = accs[3] + x_desc.load([0, 0])
    o_desc.store([0, 0], accs[0])
"#,
    );
    let err = codegen::compile(&src, &spec(3), Target::spyre())
        .expect_err("slot 3 of a 3-slot sequence is out of range");
    assert!(
        err.to_string().contains("out of range"),
        "the refusal must name the range; got: {err}"
    );
}

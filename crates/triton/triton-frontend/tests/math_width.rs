//! THE TRANSCENDENTALS' WIDTH GATE, which is the ORACLE'S and was missing here.
//!
//! Needs the `ruff` feature: every case starts from Python SOURCE.
#![cfg(feature = "ruff")]
//!
//! `tl.exp`, `tl.math.exp2` and `tl.rsqrt` are each decorated
//! `@_check_dtype(dtypes=["fp32", "fp64"])` in `python/triton/language/math.py`. On an f16
//! tensor Triton refuses, MEASURED with `make_ir` on a kernel written for it:
//!
//! ```text
//! ValueError: Expected dtype ['fp32', 'fp64'] but got fp16
//! CompilationError: at 9:8: ...
//! ```
//!
//! That refusal is WHY two fixtures are written the way they are -- `swiglu_mlp.py`'s
//! sigmoid (its delta 5) and `rmsnorm.py`'s reciprocal-sqrt (its delta 1) both spell out a
//! widen / call / truncate island. The island is FORCED by the oracle, not chosen for taste,
//! and `LegalizeTypes` is what collapses it afterwards.
//!
//! `semantic::math_unary` used to accept any float, so this front end would have compiled
//! `tl.rsqrt(x_f16)` that Triton rejects. No fixture reaches it from the wrong side, which is
//! exactly why the gap survived until `rmsnorm.py` made the f32 island a deliberate line in a
//! kernel rather than an accident. These are the tests for the gap.

mod common;

use triton_frontend::codegen;
use triton_frontend::target::Target;

/// A kernel that calls `fn_call` on a tile of `dtype`, and nothing else.
fn kernel(call: &str) -> String {
    format!(
        r#"
import triton
import triton.language as tl


@triton.jit
def math_kernel(desc_x, desc_o, M: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr):
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, D], strides=[D, 1],
                                       block_shape=[BLOCK_M, D])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, D], strides=[D, 1],
                                       block_shape=[BLOCK_M, D])
    x = x_desc.load([0, 0])
    y = {call}
    o_desc.store([0, 0], y)
"#
    )
}

fn spec() -> codegen::KernelSpec {
    let mut s = common::simple_spec(
        "math_kernel",
        &[
            ("desc_x", "*fp16"),
            ("desc_o", "*fp16"),
            ("M", "constexpr"),
            ("D", "constexpr"),
            ("BLOCK_M", "constexpr"),
        ],
    );
    for (k, v) in [("M", 64), ("D", 128), ("BLOCK_M", 64)] {
        s.constexprs
            .insert(k.to_string(), triton_frontend::semantic::Val::Int(v));
    }
    s
}

fn refuse(call: &str) -> String {
    match codegen::compile(&kernel(call), &spec(), Target::spyre()) {
        Ok(_) => panic!(
            "`{call}` COMPILED on f16, but Triton's @_check_dtype refuses it. A front end \
             more permissive than its own oracle is a silent-wrong-answer risk."
        ),
        Err(e) => e.to_string(),
    }
}

fn accept(call: &str, want_op: &str) {
    let m = codegen::compile(&kernel(call), &spec(), Target::spyre())
        .unwrap_or_else(|e| panic!("`{call}` must compile: {e}"));
    let n = m.funcs[0].body.blocks[0]
        .ops
        .iter()
        .filter(|o| o.name == want_op)
        .count();
    assert_eq!(n, 1, "expected exactly one `{want_op}` from `{call}`");
}

#[test]
fn rsqrt_on_f16_is_refused_with_tritons_own_wording() {
    for call in ["tl.rsqrt(x)", "tl.math.rsqrt(x)"] {
        let msg = refuse(call);
        assert!(
            msg.contains("Expected dtype ['fp32', 'fp64'] but got fp16"),
            "the refusal must echo Triton's wording so the two are recognisable as the \
             same rule; got: {msg}"
        );
    }
}

#[test]
fn exp_and_exp2_on_f16_are_refused_too() {
    for call in ["tl.exp(x)", "tl.math.exp2(x)"] {
        let msg = refuse(call);
        assert!(
            msg.contains("but got fp16"),
            "`{call}` must be refused on f16; got: {msg}"
        );
    }
}

/// The control WITHOUT WHICH the three refusals above prove nothing: the same call inside the
/// widen / truncate island compiles, and emits the op it should.
#[test]
fn the_f32_island_form_compiles_for_all_three() {
    for (call, op) in [
        ("tl.rsqrt(x.to(tl.float32)).to(tl.float16)", "math.rsqrt"),
        ("tl.exp(x.to(tl.float32)).to(tl.float16)", "math.exp"),
        ("tl.math.exp2(x.to(tl.float32)).to(tl.float16)", "math.exp2"),
    ] {
        accept(call, op);
    }
}

// SPDX-License-Identifier: Apache-2.0
//! A dtype comparison is a trace-time constant (`dtype.__eq__`): the `if` on it is resolved by the
//! front end, so the taken branch is emitted and the other is not. Both outcomes, and the refused
//! ordering, are pinned.

/// `x.dtype != y.dtype` over an fp8 and an f16 pointer's loads: the widen is emitted.
/// `x.dtype != x.dtype`: nothing is emitted for the branch.
/// `x.dtype < y.dtype`: refused by name.
const SRC: &str = r#"
import triton
import triton.language as tl


@triton.jit
def k(x, y, o, N: tl.constexpr):
    xd = tl.make_tensor_descriptor(x, shape=[N], strides=[1], block_shape=[N])
    yd = tl.make_tensor_descriptor(y, shape=[N], strides=[1], block_shape=[N])
    od = tl.make_tensor_descriptor(o, shape=[N], strides=[1], block_shape=[N])
    a = xd.load([0])
    b = yd.load([0])
    if a.dtype != b.dtype:
        a = a.to(b.dtype)
    if od.dtype != b.dtype:
        b = b * 2.0
    od.store([0], a + b)
"#;

use crate::common;

use triton_frontend::codegen::{self, KernelSpec};
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

fn spec(x: &str) -> KernelSpec {
    let mut s = common::simple_spec(
        "k",
        &[("x", x), ("y", "*fp16"), ("o", "*fp16"), ("N", "constexpr")],
    );
    s.constexprs.insert("N".to_string(), Val::Int(128));
    s
}

fn count(m: &triton_frontend::ttir::Module, name: &str) -> usize {
    m.funcs[0]
        .body
        .blocks
        .iter()
        .flat_map(|b| b.ops.iter())
        .filter(|o| o.name == name)
        .count()
}

/// THE WIDEN TAKEN — `x` an fp8 pointer against `y`'s f16: `a.dtype != b.dtype` is trace-time
/// true, so the widen is emitted (one `tt.fp_to_fp`); `od.dtype` is the descriptor's ELEMENT type
/// (f16), equal to `b`'s, so the second branch emits nothing (no `arith.mulf`).
#[test]
fn an_fp8_load_widened_by_a_taken_dtype_branch() {
    let m = codegen::compile(SRC, &spec("*fp8e4nv"), Target::spyre())
        .expect("the dtype-comparison kernel must compile");
    assert_eq!(
        count(&m, "tt.fp_to_fp"),
        1,
        "the taken branch's widen is emitted"
    );
    assert_eq!(
        count(&m, "arith.mulf"),
        0,
        "the untaken branch (a descriptor's .dtype equal to the load's) emits nothing"
    );
}

/// THE WIDEN NOT TAKEN — the same kernel at an f16 `x`: both comparisons are trace-time false,
/// so neither branch emits anything.
#[test]
fn an_f16_load_leaves_both_branches_untaken() {
    let m = codegen::compile(SRC, &spec("*fp16"), Target::spyre())
        .expect("the f16 dtype-comparison kernel must compile");
    assert_eq!(count(&m, "tt.fp_to_fp"), 0, "no widen is emitted");
    assert_eq!(count(&m, "arith.mulf"), 0, "no branch is taken");
}

/// THE ORDERING REFUSED — `x.dtype < y.dtype`: a dtype has no order, and the refusal names it.
#[test]
fn an_ordered_dtype_comparison_is_refused_by_name() {
    let src = SRC.replacen("a.dtype != b.dtype", "a.dtype < b.dtype", 1);
    let e = codegen::compile(&src, &spec("*fp8e4nv"), Target::spyre())
        .expect_err("an ordered dtype comparison must be refused");
    let msg = e.to_string();
    assert!(
        msg.contains("a dtype has no order"),
        "the refusal must name the law; got: {msg}"
    );
}

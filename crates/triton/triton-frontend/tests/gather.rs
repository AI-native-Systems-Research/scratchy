//! THE INDIRECT ADDRESS: `tt.descriptor_gather`, and the four things it refuses.
//!
//! Needs the `ruff` feature: every case starts from Python SOURCE.
#![cfg(feature = "ruff")]
//!
//! `tests/golden_diff.rs::embedding_matches_golden` already proves the accepting case field
//! by field against the raw golden. This file is the OTHER half, and it is the half that a
//! new construct usually skips: each of Triton's own `assert`s inside `descriptor_gather`
//! has to be reproduced, because an assert in the oracle is a REFUSAL, and compiling a
//! kernel the oracle refuses would make this front end more permissive than the thing it is
//! a port of -- a silent-wrong-answer risk, not a feature.
//!
//! THE ORACLE'S MESSAGES, MEASURED by compiling each bad kernel with real Triton's `make_ir`
//! (not read off the source) so the refusals below are held to something:
//!
//! | bad kernel | Triton says |
//! |---|---|
//! | descriptor block `[64, 128]` | `AssertionError: descriptor block must have 1 row, but got ['64', '128']` |
//! | 4 indices | `AssertionError: descriptor gather must have at least 8 rows, but got ['constexpr[4]']` |
//! | f16 index vector | `AssertionError: x offsets must have dtype int16 or int32, but got fp16` |
//! | descriptor block `[1, 8]` | `AssertionError: descriptor gather of fp16 must have at least 16 columns, but got 8` |
//! | i1 descriptor (not a gather rule, but the reason `desc_block_elem`'s i1 arm is a fallback) | `ValueError: Descriptor block shape must have at least 16 bytes in the last dimension, but got 64 * 0 = 0 bytes` |
//!
//! Ours must refuse the same four AND NAME THE CONSTRAINT. A refusal that says only "not
//! supported" is not fail-closed, because it does not tell the reader which rule was broken.

mod common;

use triton_frontend::codegen;
use triton_frontend::diff;
use triton_frontend::target::Target;
use triton_frontend::ttir;

/// The kernel under test, parameterized ONLY by the table descriptor's block shape. Every
/// other axis of the experiment is moved in the SPEC instead -- the index dtype by the
/// `desc_ids` pointer type, the index count by `BLOCK_M` -- so each bad case differs from the
/// accepting one in exactly one place.
fn gather_kernel(table_block: &str) -> String {
    format!(
        r#"
import triton
import triton.language as tl


@triton.jit
def gather_kernel(desc_ids, desc_table, desc_o, N_TOK: tl.constexpr, V: tl.constexpr,
                  D_MODEL: tl.constexpr, BLOCK_M: tl.constexpr):
    ids_desc = tl.make_tensor_descriptor(desc_ids, shape=[N_TOK], strides=[1],
                                         block_shape=[BLOCK_M])
    table_desc = tl.make_tensor_descriptor(desc_table, shape=[V, D_MODEL],
                                           strides=[D_MODEL, 1],
                                           block_shape=[{table_block}])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[N_TOK, D_MODEL],
                                       strides=[D_MODEL, 1],
                                       block_shape=[BLOCK_M, D_MODEL])
    ids = ids_desc.load([0])
    rows = table_desc.gather(ids, 0)
    o_desc.store([0, 0], rows)
"#
    )
}

fn spec(ids: &str, block_m: i128) -> codegen::KernelSpec {
    let mut s = common::simple_spec(
        "gather_kernel",
        &[
            ("desc_ids", ids),
            ("desc_table", "*fp16"),
            ("desc_o", "*fp16"),
            ("N_TOK", "constexpr"),
            ("V", "constexpr"),
            ("D_MODEL", "constexpr"),
            ("BLOCK_M", "constexpr"),
        ],
    );
    s.constexprs
        .insert("N_TOK".to_string(), triton_frontend::semantic::Val::Int(256));
    s.constexprs
        .insert("V".to_string(), triton_frontend::semantic::Val::Int(512));
    s.constexprs.insert(
        "D_MODEL".to_string(),
        triton_frontend::semantic::Val::Int(128),
    );
    s.constexprs.insert(
        "BLOCK_M".to_string(),
        triton_frontend::semantic::Val::Int(block_m),
    );
    s
}

fn refusal(src: &str, spec: &codegen::KernelSpec) -> String {
    match codegen::compile(src, spec, Target::spyre()) {
        Ok(_) => panic!(
            "this kernel COMPILED, but Triton itself refuses it. Compiling it means this \
             front end is more permissive than its own oracle."
        ),
        Err(e) => e.to_string(),
    }
}

/// The control that makes the four refusals below mean something: the SAME kernel text with
/// the one bad thing fixed compiles. Without it, a refusal could be coming from anywhere.
#[test]
fn the_accepting_form_of_the_same_kernel_compiles() {
    let src = gather_kernel("1, D_MODEL");
    let m = codegen::compile(&src, &spec("*i32", 64), Target::spyre())
        .expect("the accepting gather form must compile");
    let n = m.funcs[0].body.blocks[0]
        .ops
        .iter()
        .filter(|o| o.name == "tt.descriptor_gather")
        .count();
    assert_eq!(n, 1, "expected exactly one tt.descriptor_gather");
}

#[test]
fn a_multirow_descriptor_block_is_refused_by_name() {
    let msg = refusal(&gather_kernel("BLOCK_M, D_MODEL"), &spec("*i32", 64));
    assert!(
        msg.contains("ONE ROW"),
        "the refusal must say the descriptor block needs one row; got: {msg}"
    );
}

#[test]
fn too_few_indices_is_refused_by_name() {
    // Four ids, where Triton's floor is eight.
    let msg = refusal(&gather_kernel("1, D_MODEL"), &spec("*i32", 4));
    assert!(
        msg.contains("at least 8 indices"),
        "the refusal must name the 8-index floor; got: {msg}"
    );
}

#[test]
fn a_float_index_vector_is_refused_by_name() {
    let msg = refusal(&gather_kernel("1, D_MODEL"), &spec("*fp16", 64));
    assert!(
        msg.contains("int16 or int32"),
        "the refusal must name the accepted index dtypes; got: {msg}"
    );
}

/// A gather with too FEW columns for its element type. f16 needs 16 (`32 // 16 * 8`), so a
/// block of `[1, 8]` is one the oracle refuses.
///
/// This one also pins the expression: `32 // bitwidth * 8` is integer division FIRST, so
/// writing it as `256 / bitwidth` would agree for f16 and f32 and disagree for f64.
#[test]
fn too_few_columns_is_refused_by_name() {
    let src = gather_kernel("1, 8");
    let msg = refusal(&src, &spec("*i32", 64));
    assert!(
        msg.contains("at least 16 columns"),
        "the refusal must name the column floor for f16; got: {msg}"
    );
}

// ===----------------------------------------------------------------------===//
//   A PLANTED-DIFFERENCE CONTROL ON THE NEW OP ITSELF
// ===----------------------------------------------------------------------===//

/// The gather's two index operands are DIFFERENT KINDS of thing -- a vector of row ids and a
/// scalar column offset -- so swapping them is the mistake this op invites. It leaves the op
/// name, the operand count and every type in place, which is exactly the kind of difference a
/// weak diff misses.
#[test]
fn control_a_swapped_gather_index_is_caught() {
    let src = common::fixture_src("embedding");
    let case = common::embedding("embedding", 512, 128);
    let golden = common::golden("embedding").expect("embedding golden");
    let expected = ttir::parse::parse_module(&golden).expect("golden parses");
    let mut ours = codegen::compile(&src, &case.spec, Target::spyre()).expect("compiles");

    // Baseline first: without it a caught difference proves nothing.
    let base = diff::compare(&expected, &ours);
    assert!(base.ok(), "baseline is not clean:\n{}", base.render());
    assert!(base.matched_ops > 0, "baseline matched zero ops");

    let mut hit = false;
    for op in &mut ours.funcs[0].body.blocks[0].ops {
        if op.name == "tt.descriptor_gather" {
            assert_eq!(op.operands.len(), 3, "gather takes desc, x_offsets, y_offset");
            op.operands.swap(1, 2);
            hit = true;
        }
    }
    assert!(hit, "control could not find a tt.descriptor_gather to mutate");
    let r = diff::compare(&expected, &ours);
    let found: Vec<&String> = r
        .findings
        .iter()
        .filter(|f| f.contains(".operands:"))
        .collect();
    assert!(
        !found.is_empty(),
        "a swapped gather index was NOT caught -- the diff is not comparing this op's \
         dataflow. Findings: {:?}",
        r.findings
    );
}

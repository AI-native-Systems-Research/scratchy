//! THE INDIRECT WRITE: `tt.descriptor_scatter`, the gather's twin, and what it refuses.
//!
//! The descriptor and index-vector rules are the gather's own five (one helper states them for
//! both, so `tests/gather.rs` covers the shared messages); this file pins what is the scatter's:
//! the accepting form and its printed TTIR, the 8-row floor spelled `scatter`, and the op
//! verifier's value rule -- the stored value is exactly `[len(x_offsets), block[1]]` of the
//! descriptor's element type, with NO implicit cast (unlike `descriptor_store`).

mod common;

use triton_frontend::codegen;
use triton_frontend::target::Target;
use triton_frontend::ttir;

/// A scatter of `ROWS` rows into a `[V, D]` table, the value read as `[ROWS, VALUE_COLS]` of
/// `VALUE_DT`. Each bad case below moves exactly one of those away from the accepting form.
fn scatter_kernel(value_cols: &str, value_dt: &str) -> String {
    format!(
        r#"
import triton
import triton.language as tl


@triton.jit
def scatter_kernel(desc_ids, desc_src, desc_table, ROWS: tl.constexpr, V: tl.constexpr,
                   D: tl.constexpr):
    ids_desc = tl.make_tensor_descriptor(desc_ids, shape=[ROWS], strides=[1], block_shape=[ROWS])
    src_desc = tl.make_tensor_descriptor(desc_src, shape=[ROWS, D], strides=[D, 1],
                                         block_shape=[ROWS, {value_cols}])
    table_desc = tl.make_tensor_descriptor(desc_table, shape=[V, D], strides=[D, 1],
                                           block_shape=[1, D])
    ids = ids_desc.load([0])
    table_desc.scatter(src_desc.load([0, 0]){value_dt}, ids, 0)
"#
    )
}

fn spec(rows: i128) -> codegen::KernelSpec {
    let mut s = common::simple_spec(
        "scatter_kernel",
        &[
            ("desc_ids", "*i32"),
            ("desc_src", "*fp16"),
            ("desc_table", "*fp16"),
            ("ROWS", "constexpr"),
            ("V", "constexpr"),
            ("D", "constexpr"),
        ],
    );
    for (k, v) in [("ROWS", rows), ("V", 512), ("D", 64)] {
        s.constexprs
            .insert(k.to_string(), triton_frontend::semantic::Val::Int(v));
    }
    s
}

fn refusal(src: &str, spec: &codegen::KernelSpec) -> String {
    match codegen::compile(src, spec, Target::spyre()) {
        Ok(_) => panic!("this scatter COMPILED, but Triton refuses it"),
        Err(e) => e.to_string(),
    }
}

/// The control the refusals lean on: the same kernel with nothing wrong compiles, to exactly one
/// scatter, printed in MLIR's `$desc[$x_offsets, $y_offset], $src : type(operands)` form.
#[test]
fn the_accepting_form_compiles_and_prints_the_op_assembly() {
    let m = codegen::compile(&scatter_kernel("D", ""), &spec(8), Target::spyre())
        .expect("the accepting scatter must compile");
    let ops = &m.funcs[0].body.blocks[0].ops;
    let n = ops
        .iter()
        .filter(|o| o.name == "tt.descriptor_scatter")
        .count();
    assert_eq!(n, 1, "exactly one tt.descriptor_scatter");
    let text = ttir::print::print_module(&m);
    let line = text
        .lines()
        .find(|l| l.contains("tt.descriptor_scatter"))
        .expect("the scatter prints");
    assert!(
        line.contains("], %")
            && line.contains(": !tt.tensordesc<1x64xf16>, tensor<8xi32>, i32, tensor<8x64xf16>"),
        "the scatter prints as `%d[%ids, %y], %v : desc, ids, i32, value`; got: {line}"
    );
}

#[test]
fn too_few_rows_is_refused_with_the_scatter_spelling() {
    let msg = refusal(&scatter_kernel("D", ""), &spec(4));
    assert!(
        msg.contains("`.scatter` needs at least 8 indices")
            && msg.contains("descriptor scatter must have at least 8 rows"),
        "the refusal must name the scatter's 8-row floor; got: {msg}"
    );
}

#[test]
fn a_value_narrower_than_the_block_is_refused() {
    let msg = refusal(&scatter_kernel("32", ""), &spec(8));
    assert!(
        msg.contains("`.scatter`'s value must be [8, 64]"),
        "the refusal must state the value's required shape; got: {msg}"
    );
}

/// No implicit cast: `semantic.py` hands `value.handle` straight to the op, so an f32 value
/// against an f16 table is the verifier's element-type mismatch, not a silent `truncf`.
#[test]
fn a_value_of_another_element_type_is_refused_not_cast() {
    let msg = refusal(&scatter_kernel("D", ".to(tl.float32)"), &spec(8));
    assert!(
        msg.contains("x fp16") || msg.contains("x f16"),
        "the refusal must name the descriptor's element type; got: {msg}"
    );
}

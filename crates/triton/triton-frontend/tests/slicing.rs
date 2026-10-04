//! THE PARTIAL SLICE: refused by the LANGUAGE, and until `rope.py` was written this front end
//! compiled it into a WRONG ANSWER.
//!
//! Needs the `ruff` feature: every case starts from Python SOURCE.
#![cfg(feature = "ruff")]
//!
//! Hugging Face's `rotate_half` is `cat(-x[..., d//2:], x[..., :d//2])`, so the obvious
//! transcription of Granite's RoPE is a bounded slice. It does not exist in Triton.
//! `tensor.__getitem__` (`python/triton/language/core.py`) accepts exactly two subscript
//! items -- `None`, which is an `expand_dims`, and a FULL `:`, which is a no-op -- and raises
//! on anything else. MEASURED with `make_ir` on `x1 = x[:, :HALF]`:
//!
//! ```text
//! ValueError: unsupported tensor index: <triton.language.core.slice object at ...>
//! CompilationError: at 8:9
//! ```
//!
//! # THE BUG THIS FILE EXISTS FOR
//!
//! `codegen`'s slice arm returned `Val::Slice` for ANY slice and `subscript` treats
//! `Val::Slice` as a no-op. So `x[:, :64]` COMPILED here and produced the WHOLE 128-wide
//! tile -- a silently wrong answer, for a kernel Triton refuses outright. It was invisible
//! because no fixture had ever written a bounded slice: the four earlier subscripts in the
//! tree are all `[:, None]` or `[None, :]`.
//!
//! That is the shape of gap this crate is supposed to fail closed on, so the fix is a refusal
//! that NAMES the construct, and these are its tests -- including the two accepting forms,
//! without which a blanket refusal of every subscript would also pass.

mod common;

use triton_frontend::codegen;
use triton_frontend::target::Target;

fn kernel(body: &str) -> String {
    format!(
        r#"
import triton
import triton.language as tl


@triton.jit
def slice_kernel(desc_x, desc_o, M: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr,
                 HALF: tl.constexpr):
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, D], strides=[D, 1],
                                       block_shape=[BLOCK_M, D])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, D], strides=[D, 1],
                                       block_shape=[BLOCK_M, D])
    x = x_desc.load([0, 0])
    {body}
    y = x
    o_desc.store([0, 0], y)
"#
    )
}

fn spec() -> codegen::KernelSpec {
    let mut s = common::simple_spec(
        "slice_kernel",
        &[
            ("desc_x", "*fp16"),
            ("desc_o", "*fp16"),
            ("M", "constexpr"),
            ("D", "constexpr"),
            ("BLOCK_M", "constexpr"),
            ("HALF", "constexpr"),
        ],
    );
    for (k, v) in [("M", 64), ("D", 128), ("BLOCK_M", 64), ("HALF", 64)] {
        s.constexprs
            .insert(k.to_string(), triton_frontend::semantic::Val::Int(v));
    }
    s
}

/// Every bounded spelling, including the one with a step and the negative-index one.
#[test]
fn a_bounded_slice_is_refused_by_name() {
    for body in [
        "y = x[:, :HALF]",
        "y = x[:, HALF:]",
        "y = x[:HALF, :]",
        "y = x[:, ::2]",
        "y = x[:, 0:HALF]",
    ] {
        let src = kernel(body);
        let err = codegen::compile(&src, &spec(), Target::spyre()).err().unwrap_or_else(|| {
            panic!(
                "`{body}` COMPILED. Triton refuses it (`unsupported tensor index`), and here it \
                 silently yields the FULL tile -- a wrong answer with no diagnostic."
            )
        });
        let msg = err.to_string();
        assert!(
            msg.contains("BOUNDED tensor slice"),
            "the refusal must name the construct; got: {msg}"
        );
    }
}

/// The control. `[:, None]` and `[None, :]` must still compile and must still produce the
/// `tt.expand_dims` they are -- otherwise the test above would pass on a front end that had
/// simply stopped supporting subscripts.
///
/// The subscripted value is not what gets stored (a rank-3 tile does not fit the descriptor),
/// so the assertion is on the emitted op rather than on the output.
#[test]
fn the_two_accepted_subscript_forms_still_compile() {
    for (body, want) in [
        ("z = x[:, None]", Some(vec![64i64, 1, 128])),
        ("z = x[None, :]", Some(vec![1, 64, 128])),
        // A FULL slice is a no-op on both axes, so it emits nothing at all.
        ("z = x[:, :]", None),
    ] {
        let src = kernel(body);
        let m = codegen::compile(&src, &spec(), Target::spyre())
            .unwrap_or_else(|e| panic!("`{body}` must compile: {e}"));
        let expands: Vec<&triton_frontend::ttir::Op> = m.funcs[0].body.blocks[0]
            .ops
            .iter()
            .filter(|o| o.name == "tt.expand_dims")
            .collect();
        match want {
            Some(shape) => {
                assert_eq!(expands.len(), 1, "`{body}` should emit one tt.expand_dims");
                assert_eq!(
                    m.ty(expands[0].results[0]).shape(),
                    shape.as_slice(),
                    "`{body}` expanded to the wrong shape"
                );
            }
            None => assert!(
                expands.is_empty(),
                "`{body}` is a no-op and must emit no tt.expand_dims"
            ),
        }
    }
}

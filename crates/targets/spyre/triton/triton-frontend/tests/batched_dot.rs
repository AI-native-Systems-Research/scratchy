// SPDX-License-Identifier: Apache-2.0

//! THE RANK-3 BATCHED `tl.dot` -- OUR DIALECT EXTENSION, PINNED WITH ITS CONTROLS.
//!
//! Upstream Triton's `tl.dot` is rank-2 only (`semantic.py:1429`). The batched
//! paged-decode kernel (`crates/targets/spyre/kernels/batched_decode.py`) needs
//! `[batch, M, K] x [batch, K, N] -> [batch, M, N]` -- the KV heads and the GQA
//! group riding the dot's batch and M axes -- so this front end admits it as the
//! SAME `tt.dot` op with rank-3 types (the rank rides the types; the rank-2
//! emission is byte-identical to upstream's), and `dot_to_linalg` discriminates
//! on the rank downstream.
//!
//! The accepting case here is the WHOLE front end over that form: the reshapes
//! that produce rank-3 operands (themselves the extension `tests/` has no other
//! witness for), the dot, and the rank-reducing reshape back to a storable tile.
//! The controls each break ONE fact and assert the refusal that names it -- the
//! same discipline `gather.rs` holds Triton's own asserts to, because an
//! extension that accepted a shape its lowering cannot state would be a
//! silent-wrong-answer risk, not a feature.

mod common;

use triton_frontend::codegen::{self, KernelSpec};
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

/// The kernel under test, parameterized by the two reshapes' target shapes so
/// each control differs from the accepting form in exactly one place. The flat
/// geometries: q `[8, 64]` and kt `[128, 32]`, both re-chunked to `[2, 4, 64]`
/// and `[2, 64, 32]`; the dot contracts each batch's `[4, 64] x [64, 32]` and
/// the `[2, 4, 32]` result is re-chunked back to `[8, 32]` for the store.
fn batched_dot_kernel(q_shape: &str, kt_shape: &str) -> String {
    format!(
        r#"
import triton
import triton.language as tl


@triton.jit
def batched_dot(desc_q, desc_kt, desc_o, K: tl.constexpr, NQ: tl.constexpr):
    q_desc = tl.make_tensor_descriptor(desc_q, shape=[8, K], strides=[K, 1],
                                       block_shape=[8, K])
    kt_desc = tl.make_tensor_descriptor(desc_kt, shape=[128, NQ], strides=[NQ, 1],
                                         block_shape=[128, NQ])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[8, NQ], strides=[NQ, 1],
                                        block_shape=[8, NQ])
    q = tl.reshape(q_desc.load([0, 0]), [{q_shape}])
    kt = tl.reshape(kt_desc.load([0, 0]), [{kt_shape}])
    qk = tl.dot(q, kt, out_dtype=tl.float16)
    o_desc.store([0, 0], tl.reshape(qk, [8, NQ]))
"#
    )
}

fn spec() -> KernelSpec {
    let mut s = common::simple_spec(
        "batched_dot",
        &[
            ("desc_q", "*fp16"),
            ("desc_kt", "*fp16"),
            ("desc_o", "*fp16"),
            ("K", "constexpr"),
            ("NQ", "constexpr"),
        ],
    );
    s.constexprs.insert("K".to_string(), Val::Int(64));
    s.constexprs.insert("NQ".to_string(), Val::Int(32));
    s
}

/// THE ACCEPTING FORM: the whole front end over the rank-3 dot -- both reshapes,
/// the dot, the rank-reducing reshape, the store -- with exactly one `tt.dot`
/// whose result type is the rank-3 `[b, M, N]`.
#[test]
fn the_batched_form_compiles_as_one_rank_three_dot() {
    let src = batched_dot_kernel("2, 4, 64", "2, 64, 32");
    let m = codegen::compile(&src, &spec(), Target::spyre())
        .expect("the batched dot form must compile");
    let body = &m.funcs[0].body.blocks[0].ops;
    let dots: Vec<&triton_frontend::ttir::Op> =
        body.iter().filter(|o| o.name == "tt.dot").collect();
    assert_eq!(dots.len(), 1, "expected exactly one tt.dot");
    assert_eq!(
        m.ty(dots[0].results[0]).to_string(),
        "tensor<2x4x32xf16>",
        "the dot's result type is the batched [b, M, N] -- the rank rides the type, \
         which is what keeps the rank-2 emission byte-identical"
    );
}

/// CONTROL 1 -- mixed ranks: the kt operand left rank-2. Neither dialect states
/// what a 3x2 contraction means, so it is refused by name.
#[test]
fn a_mixed_rank_pair_is_refused_by_name() {
    let src = batched_dot_kernel("2, 4, 64", "4, 1024");
    let e = codegen::compile(&src, &spec(), Target::spyre())
        .expect_err("a mixed-rank dot must be refused");
    let msg = e.to_string();
    assert!(
        msg.contains("got ranks 3 and 2"),
        "the refusal must name both ranks; got: {msg}"
    );
}

/// CONTROL 2 -- batch dims that disagree: kt re-chunked to batch 4 against q's
/// batch 2. There is no pairing of the batches, so it is refused by name.
#[test]
fn disagreeing_batch_dims_are_refused_by_name() {
    let src = batched_dot_kernel("2, 4, 64", "4, 32, 32");
    let e = codegen::compile(&src, &spec(), Target::spyre())
        .expect_err("a batch-disagreeing dot must be refused");
    let msg = e.to_string();
    assert!(
        msg.contains("batch dims disagree"),
        "the refusal must name the batch disagreement; got: {msg}"
    );
}

/// CONTROL 3 -- the inner pair disagrees: kt's K is 32 against q's 64 (the
/// re-chunk keeps the batch but contracts the wrong extent).
#[test]
fn disagreeing_inner_dimensions_are_refused_by_name() {
    let src = batched_dot_kernel("2, 4, 64", "2, 32, 64");
    let e = codegen::compile(&src, &spec(), Target::spyre())
        .expect_err("an inner-disagreeing dot must be refused");
    let msg = e.to_string();
    assert!(
        msg.contains("inner dimensions disagree"),
        "the refusal must name the inner disagreement; got: {msg}"
    );
}

/// CONTROL 4 -- rank 4 exists in neither dialect's dot, so it is refused with
/// the same named message rather than half-modelled.
#[test]
fn a_rank_four_pair_is_refused_by_name() {
    let src = batched_dot_kernel("2, 2, 2, 64", "2, 2, 64, 16");
    let e =
        codegen::compile(&src, &spec(), Target::spyre()).expect_err("a rank-4 dot must be refused");
    let msg = e.to_string();
    assert!(
        msg.contains("got ranks 4 and 4"),
        "the refusal must name the rank; got: {msg}"
    );
}

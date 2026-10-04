//! THE ONE DELIBERATE DIVERGENCE FROM TRITON, exercised directly.
//!
//! Needs the `ruff` feature: every case starts from Python SOURCE. `tests/no_parser.rs`
//! covers what the dependency-free build can still be held to.
#![cfg(feature = "ruff")]
//!
//! `computation_type_impl` promotes `f16 / f16 -> f32` because "/ and % do not exist
//! natively in PTX for fp16" (`python/triton/language/semantic.py:88-92`). That is a PTX
//! fact and it is FALSE on this device, where divide is the templated `REALDIV`
//! (`broadcast_ops.ddl`, and `arith.divf -> "realdiv"` in
//! `../triton-superdsc/triton-superdsc-lower/src/opmap.rs`).
//!
//! **NO FIXTURE COVERS THIS**, and that is not an oversight: the fixtures were written to
//! work around the promotion, so `swiglu_mlp.py` calls `tl.fdiv` and never a bare `/` on
//! f16. So the goldens contain no evidence either way and the switch has to be tested on
//! its own, against a kernel written for the purpose. Both directions are asserted, since a
//! test that only checks the Spyre setting would pass with the switch hard-wired.



use std::collections::HashMap;

use triton_frontend::codegen::{self, ArgSpec, KernelSpec};
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;
use triton_frontend::ttir::Module;

/// A kernel whose only compute is a bare `/` on two f16 tensors.
const DIV_KERNEL: &str = r#"
import triton
import triton.language as tl


@triton.jit
def div_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[n], strides=[1],
                                       block_shape=[BLOCK])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[n], strides=[1],
                                       block_shape=[BLOCK])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[n], strides=[1],
                                       block_shape=[BLOCK])
    a = a_desc.load([0])
    b = b_desc.load([0])
    c_desc.store([0], a / b)
"#;

/// The same kernel written with `tl.fdiv`, which is the same `create_fdiv` with
/// `arithmetic_check` off -- the evidence that the promotion is front-end POLICY and not a
/// semantic requirement.
const FDIV_KERNEL: &str = r#"
import triton
import triton.language as tl


@triton.jit
def div_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[n], strides=[1],
                                       block_shape=[BLOCK])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[n], strides=[1],
                                       block_shape=[BLOCK])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[n], strides=[1],
                                       block_shape=[BLOCK])
    a = a_desc.load([0])
    b = b_desc.load([0])
    c_desc.store([0], tl.fdiv(a, b))
"#;

fn spec() -> KernelSpec {
    let mut signature = HashMap::new();
    for (k, v) in [
        ("a_ptr", "*fp16"),
        ("b_ptr", "*fp16"),
        ("c_ptr", "*fp16"),
        ("n", "i32"),
        ("BLOCK", "constexpr"),
    ] {
        signature.insert(k.to_string(), ArgSpec::parse(v).unwrap());
    }
    let mut constexprs = HashMap::new();
    constexprs.insert("BLOCK".to_string(), Val::Int(64));
    KernelSpec {
        kernel: "div_kernel".to_string(),
        signature,
        constexprs,
        file: "divergence_test.py".to_string(),
    }
}

/// Every op name in the kernel, in order.
fn op_names(m: &Module) -> Vec<String> {
    m.funcs[0].body.blocks[0]
        .ops
        .iter()
        .map(|o| o.name.clone())
        .collect()
}

/// The result type of the single `arith.divf`.
fn divf_result_type(m: &Module) -> String {
    let op = m.funcs[0].body.blocks[0]
        .ops
        .iter()
        .find(|o| o.name == "arith.divf")
        .expect("no arith.divf in the module");
    m.ty(op.results[0]).to_string()
}

#[test]
fn spyre_keeps_f16_for_a_bare_divide() {
    let m = codegen::compile(DIV_KERNEL, &spec(), Target::spyre())
        .unwrap_or_else(|e| panic!("div kernel did not compile: {e}"));
    assert_eq!(
        divf_result_type(&m),
        "tensor<64xf16>",
        "on Spyre `f16 / f16` must stay f16 -- an f32 quotient reaching a tl.dot beside an \
         f16 weight is rejected by the frontend with `Both operands must be same dtype`"
    );
    let names = op_names(&m);
    assert!(
        !names.iter().any(|n| n == "arith.extf"),
        "no widening should be emitted for f16 / f16 on Spyre, got {names:?}"
    );
    assert!(
        !names.iter().any(|n| n == "arith.truncf"),
        "no narrowing should be emitted for f16 / f16 on Spyre, got {names:?}"
    );
}

#[test]
fn upstream_promotes_a_bare_divide_to_f32() {
    // The CONTROL for the test above: with Triton's own rule the promotion DOES happen, so
    // the Spyre result is a real difference rather than the only thing this code can do.
    let m = codegen::compile(DIV_KERNEL, &spec(), Target::upstream_gpu())
        .unwrap_or_else(|e| panic!("div kernel did not compile: {e}"));
    assert_eq!(
        divf_result_type(&m),
        "tensor<64xf32>",
        "upstream Triton promotes f16 / f16 to f32 (computation_type_impl rule 3)"
    );
    let names = op_names(&m);
    assert_eq!(
        names.iter().filter(|n| *n == "arith.extf").count(),
        2,
        "upstream should widen BOTH operands to f32, got {names:?}"
    );
}

#[test]
fn fdiv_is_unaffected_by_the_switch_in_either_direction() {
    // `tl.fdiv` turns `arithmetic_check` off, so it never promotes -- which is precisely why
    // it is the evidence that the promotion is policy. If this ever differed by target, the
    // divergence would have been implemented in the wrong place.
    for target in [Target::spyre(), Target::upstream_gpu()] {
        let m = codegen::compile(FDIV_KERNEL, &spec(), target)
            .unwrap_or_else(|e| panic!("fdiv kernel did not compile: {e}"));
        assert_eq!(
            divf_result_type(&m),
            "tensor<64xf16>",
            "tl.fdiv must yield f16 from f16 operands regardless of target ({target:?})"
        );
    }
}

/// THE DIVERGENCE, MEASURED ON A REAL FIXTURE, and asserted to be LOCALIZED.
///
/// `attention_flash.py`'s epilogue is `acc = acc / l_i[:, None]` -- a bare f16 divide, and the
/// only one in any fixture. Its golden therefore carries the promotion: two `arith.extf` in,
/// an f32 `arith.divf`, and an `arith.truncf` out.
///
/// Two things are asserted, and they are the pair that makes the divergence a fact rather than
/// a claim:
///
///   1. Under `upstream_gpu` the module matches the golden EXACTLY. So the port is faithful --
///      the difference below is the switch, not a bug.
///   2. Under `spyre` it differs by exactly THREE ops, and the first divergence is at an
///      `arith.extf` in the golden. Three is the widen-widen-truncate island; the divide
///      itself stays. If the switch were leaking anywhere else, the count would be larger.
///
/// This is what `swiglu_mlp.py`' own delta 6 predicted in prose:
/// "NOTE FOR THE ATTENTION FIXTURE: its epilogue `acc = acc / l_i[:, None]` has the SAME
/// latent promotion, so its divide is an f32 island where a `realdiv` would do."
///
/// ⛔ `#[ignore]` AT RE-HOST (2026-10-02), because this test is RED AT THE SOURCE: the
/// identical failure (11 findings — `_attn_fwd_inner`'s fourth descriptor mangled `TDfp16`
/// in the golden vs `Pfp16` here, op_count 168 vs 163, a constant 64 vs 1024) reproduces
/// at triton-spyre's clean tip `5c51a1d7a`, before any re-host change. The golden was
/// evidently generated from a Triton whose descriptor flattening differed. The re-host is
/// FAITHFUL — it reproduces the source's failure exactly. Un-ignore when the golden is
/// regenerated or the front end's descriptor mangling is reconciled, on the triton-spyre
/// side; do not "fix" it here, because a green here would mean the re-host DIVERGED from
/// its source of truth.
#[test]
#[ignore = "red at triton-spyre tip 5c51a1d7a; see the comment above — golden regeneration, not a re-host defect"]
fn the_attention_epilogue_divide_is_the_whole_divergence() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = std::fs::read_to_string(dir.join("../test-fixtures/attention_flash.py"))
        .expect("attention_flash.py");
    let golden = std::fs::read_to_string(
        dir.join("tests/goldens/attention_flash_noncausal.ttir_raw.mlir"),
    )
    .expect("golden");

    let mut signature = HashMap::new();
    for (k, v) in [
        ("desc_q", "*fp16"),
        ("desc_k", "*fp16"),
        ("desc_v", "*fp16"),
        ("desc_o", "*fp16"),
        ("desc_mask", "*fp16"),
        ("Z", "constexpr"),
        ("H", "constexpr"),
        ("N_CTX", "constexpr"),
        ("HEAD_DIM", "constexpr"),
        ("BLOCK_M", "constexpr"),
        ("BLOCK_N", "constexpr"),
        ("GQA", "constexpr"),
        ("STAGE", "constexpr"),
        ("sm_scale", "constexpr"),
    ] {
        signature.insert(k.to_string(), ArgSpec::parse(v).unwrap());
    }
    let mut constexprs = HashMap::new();
    for (k, v) in [
        ("Z", 1i128),
        ("H", 4),
        ("N_CTX", 256),
        ("HEAD_DIM", 128),
        ("BLOCK_M", 64),
        ("BLOCK_N", 64),
        ("GQA", 2),
        ("STAGE", 1),
    ] {
        constexprs.insert(k.to_string(), Val::Int(v));
    }
    constexprs.insert("sm_scale".to_string(), Val::Float(1.0));
    let spec = KernelSpec {
        kernel: "attn_fwd".to_string(),
        signature,
        constexprs,
        file: dir
            .join("../test-fixtures/attention_flash.py")
            .to_string_lossy()
            .to_string(),
    };

    // 1. Upstream reproduces the oracle exactly.
    let up = codegen::compile(&src, &spec, Target::upstream_gpu()).expect("compiles");
    let r_up = triton_frontend::diff::compare_to_golden_text(&golden, &up);
    assert!(
        r_up.ok(),
        "with Triton's own promotion rule the port must match the golden exactly, else the \
         Spyre difference below cannot be attributed to the switch:\n{}",
        r_up.render()
    );

    // 2. Spyre differs by exactly the widen/truncate island.
    let sp = codegen::compile(&src, &spec, Target::spyre()).expect("compiles");
    let r_sp = triton_frontend::diff::compare_to_golden_text(&golden, &sp);
    assert!(
        !r_sp.ok(),
        "the Spyre target MUST differ from the golden at the epilogue divide; if it stopped \
         differing, the divergence has silently been lost"
    );
    let golden_extf = r_sp.expected_hist.get("arith.extf").copied().unwrap_or(0);
    let ours_extf = r_sp.actual_hist.get("arith.extf").copied().unwrap_or(0);
    let golden_truncf = r_sp.expected_hist.get("arith.truncf").copied().unwrap_or(0);
    let ours_truncf = r_sp.actual_hist.get("arith.truncf").copied().unwrap_or(0);
    assert_eq!(
        (golden_extf - ours_extf, golden_truncf - ours_truncf),
        (2, 1),
        "the divergence should be exactly two fewer `arith.extf` and one fewer `arith.truncf` \
         (the widen-widen-truncate island around one divide), got extf {golden_extf}->{ours_extf} \
         truncf {golden_truncf}->{ours_truncf}"
    );
    assert_eq!(
        r_sp.expected_hist.get("arith.divf"),
        r_sp.actual_hist.get("arith.divf"),
        "the DIVIDE itself must still be there on Spyre -- only its widening goes away"
    );
    // Every other op kind must be unchanged: the switch is not allowed to move anything else.
    let mut keys: Vec<&String> = r_sp
        .expected_hist
        .keys()
        .chain(r_sp.actual_hist.keys())
        .collect();
    keys.sort();
    keys.dedup();
    for k in keys {
        if k == "arith.extf" || k == "arith.truncf" {
            continue;
        }
        assert_eq!(
            r_sp.expected_hist.get(k).copied().unwrap_or(0),
            r_sp.actual_hist.get(k).copied().unwrap_or(0),
            "`{k}` count changed under the Spyre target; the f16-divide switch must only \
             remove the widen/truncate island"
        );
    }
}

/// The divergence must not leak into multiplication, which Triton does NOT promote. A
/// switch implemented too broadly would silently change every f16 op.
#[test]
fn the_switch_does_not_affect_non_division_ops() {
    let src = DIV_KERNEL.replace("a / b", "a * b");
    for target in [Target::spyre(), Target::upstream_gpu()] {
        let m = codegen::compile(&src, &spec(), target)
            .unwrap_or_else(|e| panic!("mul kernel did not compile: {e}"));
        let op = m.funcs[0].body.blocks[0]
            .ops
            .iter()
            .find(|o| o.name == "arith.mulf")
            .expect("no arith.mulf");
        assert_eq!(
            m.ty(op.results[0]).to_string(),
            "tensor<64xf16>",
            "f16 * f16 stays f16 on every target ({target:?}); the div switch must not \
             touch it"
        );
    }
}

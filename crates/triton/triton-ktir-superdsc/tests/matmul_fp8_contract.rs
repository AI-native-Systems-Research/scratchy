// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! ⛔⛔⛔ THE fp8 W8A8 DOOR IS OP-BLIND, SO ITS CONTRACT IS OURS TO REFUSE — NOT THEIRS.
//!
//! # WHAT THIS MEASURES
//!
//! `matmul_oriented`'s arity-3 fp8 arm (vendored `lower_ktir_to_superdsc.rs:3690`) reads
//! ONLY the region list: A, an `is_fp8` W, and a w_scale — and then lowers the W8A8
//! QUANTIZE-DEQUANTIZE CHAIN whatever the kernel's compute says. The device quantizes the
//! activation (abs → amax → clamp ±448 → qfp8ch), multiplies by `a_scale[m]` AND
//! `w_scale[n]`, and the emitted `batchmatmulfp8` runs in that frame. If the KERNEL spells
//! a different epilogue — a scale from the wrong parameter, a missing multiply, a swapped
//! ws/output pointer — nothing downstream compares the two. The kernel computes one
//! function; the door lowers another; the card answers the door's.
//!
//! `triton-ktir`'s `dot_to_linalg` is where they are pinned together: the fp8 verifier arm
//! (`verify_canonical_fp8_matmul_kernel`) requires exactly
//! `{1 tt.dot, 3 tt.descriptor_load, 1 arith.mulf, 1 tt.descriptor_store}` with the mulf
//! traced to the THIRD parameter's `[1, N]` load and the store bound to the FOURTH, and the
//! vendored arity/is_fp8 agreement guard (`:3673`) makes `is_fp8 == (ins.len() == 3)` a
//! build error rather than a reinterpretation.
//!
//! # WHY EACH REFUSAL IS A CORRECTNESS FACT AND NOT A STYLE RULE
//!
//! Every control here is a kernel the door WOULD otherwise lower, and each names the
//! wrong answer it would give:
//!   * an f16 W with an arity-3 list (w_scale present) — the vendored `:3673` guard: the
//!     door cannot bind a 2-input f16 shape to a 3-input dequant or vice versa;
//!   * an fp8 W with an arity-2 list (ws dropped) — same guard, mirror side: the W's
//!     `Dtype` says fp8 but the dequant has no scale operand to read;
//!   * a multi-block N (`BLOCK_N < N`) — the emitter emits ONE M×N program and discards the
//!     block size, so N/4 of the output would silently never be computed;
//!   * a K-tiled form (`BLOCK_K < K`) — same single-program assumption along the
//!     reduction axis;
//!   * a scale row with the wrong extent (`[1, K]` instead of `[1, N]`) — the door reads
//!     `w_scale[n]` per output channel; a K-wide row hands it input-channel scales;
//!   * an mulf whose other operand is NOT the w_scale load — the kernel computes
//!     `p * <something else>` and the door dequantizes by `w_scale` anyway: two different
//!     functions, one of them silently.
//!
//! The POSITIVE control is `matmul_fp8_small` itself — the same fixture, unmodified,
//! through the same chain — which must BAKE. A refusal suite with no positive control
//! cannot tell "the contract holds" from "the fixture is broken".
//!
//! # THE DRIVER
//!
//! Every case starts from the checked-in `.py` fixture and compiles it with a varied
//! `KernelSpec` — the same `triton_frontend::codegen::compile` + `opt::make_ttir` legs
//! `bake_py` runs — because a test that starts from a hand-written KTIR cannot refuse at
//! bridge one, and every gap this file is about is reachable from the kernel's own
//! spelling.

use ktir_superdsc::ktir_node::Program;
use std::collections::HashMap;
use std::path::PathBuf;
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;
use triton_frontend::{codegen, opt};

/// The fixture's path — `crates/triton/test-fixtures/matmul_fp8.py`.
fn fixture() -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|p| p.join("test-fixtures/matmul_fp8.py"))
        .expect("the crate sits one level under crates/triton");
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

/// The canonical single-tile constexprs: M=64, K=128, N=128, BLOCK_* = the full extent.
fn ces() -> HashMap<String, Val> {
    HashMap::from([
        ("M".into(), Val::Int(64)),
        ("K".into(), Val::Int(128)),
        ("N".into(), Val::Int(128)),
        ("BLOCK_M".into(), Val::Int(64)),
        ("BLOCK_K".into(), Val::Int(128)),
        ("BLOCK_N".into(), Val::Int(128)),
    ])
}

/// The canonical four-pointer signature: A f16, W fp8e4nv, w_scale f16, out f16.
fn sig_fp8() -> HashMap<String, codegen::ArgSpec> {
    sig_with_w("*fp8e4nv")
}

/// [`sig_fp8`] with the weight pointer's element type swapped — the arity/is_fp8 mismatch
/// controls are both ONE word away from the real signature.
fn sig_with_w(w: &str) -> HashMap<String, codegen::ArgSpec> {
    HashMap::from([
        ("desc_x".into(), codegen::ArgSpec::parse("*fp16").unwrap()),
        ("desc_w".into(), codegen::ArgSpec::parse(w).unwrap()),
        ("desc_ws".into(), codegen::ArgSpec::parse("*fp16").unwrap()),
        ("desc_o".into(), codegen::ArgSpec::parse("*fp16").unwrap()),
    ])
}

/// Compile the fixture to emitted ops — the whole `bake_py` chain, value-level, minus the
/// directory write. `Err` carries the first stage's refusal text.
fn drive(
    sig: &HashMap<String, codegen::ArgSpec>,
    ces: &HashMap<String, Val>,
    grid: &[i64],
) -> Result<Vec<String>, String> {
    drive_src(&fixture(), sig, ces, grid)
}

/// [`drive`] over a MUTATED source — the controls whose gap is a kernel spelling the
/// fixture does not carry (a dropped scale, a wrong-operand multiply) are one edit away
/// from the real fixture, and driving the edit through the same chain is what makes the
/// control reachable at the same stage the real kernel would be.
fn drive_src(
    src: &str,
    sig: &HashMap<String, codegen::ArgSpec>,
    ces: &HashMap<String, Val>,
    grid: &[i64],
) -> Result<Vec<String>, String> {
    // The constexpr parameters need their `constexpr` signature entries too — `cases::spec`
    // does the same, or codegen refuses the kernel for parameters with no signature row.
    let mut signature = sig.clone();
    for k in ces.keys() {
        signature.insert(
            k.clone(),
            codegen::ArgSpec::parse("constexpr").expect("constexpr"),
        );
    }
    let kspec = codegen::KernelSpec {
        kernel: "matmul_fp8_fwd".into(),
        signature,
        constexprs: ces.clone(),
        file: "matmul_fp8.py".into(),
    };
    let mut tt = codegen::compile(src, &kspec, Target::spyre()).map_err(|e| format!("codegen: {e}"))?;
    opt::make_ttir(&mut tt).map_err(|e| format!("make_ttir: {e}"))?;
    let mut m = triton_ktir::from_ttir::convert(&tt).map_err(|e| format!("from_ttir: {e}"))?;
    triton_ktir::make_ktir(&mut m, grid).map_err(|e| format!("make_ktir: {e}"))?;
    triton_ktir::passes::to_ktir::run(&mut m, grid).map_err(|e| format!("to_ktir: {e}"))?;
    let node = triton_ktir_superdsc::node_for(&m, Program::Matmul).map_err(|e| e.message)?;
    let ops = triton_ktir_superdsc::emit_node(&node).map_err(|e| e.message)?;
    Ok(ops.into_iter().map(|o| o.op_name.clone()).collect())
}

/// The refusal names the gap, not a stage — every control asserts on a SUBSTRING so the
/// check survives message wording but not meaning.
fn refuses_with(what: &str, needle: &str) {
    match drive(&sig_fp8(), &ces_override(what), &[1]) {
        Ok(names) => panic!(
            "{what}: BAKED ({} ops) — the door lowered a kernel that violates the \
             W8A8 contract it assumes",
            names.len()
        ),
        Err(msg) => assert!(
            msg.contains(needle),
            "{what}: refused, but not for the reason this test pins. Expected the \
             message to contain\n    {needle:?}\n  got\n    {msg:?}"
        ),
    }
}

/// `ces()` with one binding overridden.
fn ces_override(what: &str) -> HashMap<String, Val> {
    let mut c = ces();
    match what {
        "block_n_half" => *c.get_mut("BLOCK_N").unwrap() = Val::Int(64),
        "block_k_half" => *c.get_mut("BLOCK_K").unwrap() = Val::Int(64),
        other => panic!("unknown override `{other}`"),
    }
    c
}

/// THE POSITIVE CONTROL: the canonical fixture bakes, and the op list IS the W8A8 chain —
/// the quantize chain, one `batchmatmulfp8`, and the two dequant multiplies. A bake that
/// emits a plain `batchmatmul` would be the f16 door reached with an fp8 weight, which is
/// exactly the silent reinterpretation this suite exists to prevent.
#[test]
fn the_canonical_fp8_kernel_bakes_and_emits_the_w8a8_chain() {
    let names = drive(&sig_fp8(), &ces(), &[1]).expect("the canonical fp8 kernel bakes");
    assert!(
        names.len() == 12,
        "the W8A8 chain is 12 ops (abs→amax→amaxfl→ascale→invs→sc→chi→cl→qfp8ch→\
         batchmatmulfp8→dqa→dqw); got {}: {names:?}",
        names.len()
    );
    assert!(
        names.iter().any(|n| n.contains("fq_afp8_op")),
        "the quantize chain's qfp8ch convert is missing: {names:?}"
    );
    assert!(
        names.iter().any(|n| n.contains("fq_dqw_op")),
        "the w_scale dequant multiply is missing: {names:?}"
    );
}

/// AN f16 WEIGHT WITH A w_scale PARAMETER. The intended backstop is the vendored
/// arity/is_fp8 agreement guard (`lower_ktir_to_superdsc.rs:3673`), but OUR verifier
/// refuses first and for the right reason, MEASURED: with an f16 descriptor the peel
/// does not fire, so the spelled `.to(tl.float16)` is a stray `arith.extf` on B and the
/// direct-load guard red-stops it. An f16 kernel that casts its weight is not the
/// canonical f16 form (and one that dropped the cast would be refused by the f16 arm's
/// own counts — three loads, not two). Either way the kernel never reaches a door that
/// could misbind it; this pins the FIRST refusal so a future reorder of the guards
/// fails this test rather than silently routing to the next one.
#[test]
fn an_f16_weight_with_a_scale_parameter_is_refused_before_the_door() {
    let err = match drive(&sig_with_w("*fp16"), &ces(), &[1]) {
        Ok(names) => panic!(
            "BAKED ({} ops) — an f16 W with a w_scale parameter must not lower through \
             the W8A8 door or the f16 door unverified",
            names.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.contains("A and B must be direct tt.descriptor_load results")
            || err.contains("expects 2 inputs (A, W) or 3"),
        "expected the f16-W refusal (our direct-load guard, or the vendored arity \
         guard), got: {err:?}"
    );
}

/// A MULTI-BLOCK N — `BLOCK_N < N` with the same four pointers. The emitter emits ONE
/// M×N program and DISCARDS the block size, so half the output would never be computed.
/// MEASURED: the first guard to catch it is the W-extents check — the [64, BLOCK_K=128]
/// block against the [128, 128] weight descriptor — which is the same fact stated as
/// "the block does not cover the weight the contraction needs". Pinned by its own
/// message so a reorder of the checks fails here rather than routing on.
#[test]
fn a_multi_block_n_is_refused_naming_the_block_weight_mismatch() {
    refuses_with("block_n_half", "the fp8 weight descriptor is");
}

/// A K-TILED FORM — `BLOCK_K < K`. Same single-program assumption along the reduction
/// axis: the contraction would read K/2 columns and dequantize as if it were the whole
/// K. Refused by the fp8 verifier's K check, by name.
#[test]
fn a_k_tiled_kernel_is_refused_naming_the_k_contract() {
    refuses_with("block_k_half", "fp8 matmul does not tile K here");
}

/// AN fp8 WEIGHT WITH THE SCALE DROPPED — the arity/is_fp8 mismatch, fp8 side: the
/// kernel dequantizes nothing and multiplies by no row, but the W's view still says
/// `Fp8E4m3`, so the door would read three inputs into a two-input list (or vice
/// versa) and the vendored `:3673` guard refuses. OUR side refuses EARLIER and for the
/// right reason: with no ws load and no mulf, the kernel is not
/// `{1 dot, 3 loads, 1 mulf, 1 store}`, and the counts check red-stops it before any
/// buffer is bound. Either refusal is correct; the first one is pinned.
#[test]
fn an_fp8_weight_with_the_scale_dropped_is_refused_by_the_counts() {
    // The fixture with the scale load and the multiply removed: `p * ws` becomes `p`.
    // Two `replace`s, each of which must land exactly once or the fixture changed.
    let src = fixture();
    let src = src
        .replace(
            "    ws = ws_desc.load([0, 0])\n    o_desc.store([offs_m, 0], p * ws)",
            "    o_desc.store([offs_m, 0], p)",
        )
        .replace(
            "    ws_desc = tl.make_tensor_descriptor(desc_ws, shape=[1, N],\n                                        strides=[N, 1],\n                                        block_shape=[1, BLOCK_N])\n",
            "",
        );
    let mut sig = sig_fp8();
    // `desc_ws` is still a parameter — the kernel signature keeps four pointers — but
    // nothing addresses it, so `to_ktir`'s `every_parameter_states_its_width` is the
    // other guard this control may meet. Both are correct; assert on the counts one
    // with the width one as the accepted alternative.
    let _ = &mut sig;
    let err = match drive_src(&src, &sig, &ces(), &[1]) {
        Ok(names) => panic!(
            "BAKED ({} ops) — an fp8 W with no dequant multiply lowered through a door \
             that applies w_scale whatever the kernel says",
            names.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.contains("fp8 kernel is not exactly")
            || err.contains("states no width")
            || err.contains("every_parameter_states_its_width")
            || err.contains("expects 2 inputs (A, W) or 3"),
        "expected the dropped-scale refusal (counts, width, or arity), got: {err:?}"
    );
}

/// A mulf WHOSE OTHER OPERAND IS NOT THE SCALE LOAD — the kernel multiplies the product
/// by the ACTIVATION instead (`p * x`), spelling an epilogue the door does not lower:
/// the W8A8 body dequantizes by `w_scale[n]` whatever this kernel says, so the two
/// would disagree silently. This is the ONE binding check that exists anywhere — the
/// door itself never inspects the compute — so its refusal is the contract.
#[test]
fn a_mulf_not_on_the_scale_load_is_refused_naming_the_epilogue() {
    let src = fixture().replace(
        "    o_desc.store([offs_m, 0], p * ws)",
        "    o_desc.store([offs_m, 0], p * x)",
    );
    // `x` is [BLOCK_M, BLOCK_K]; the store wants [BLOCK_M, BLOCK_N] — at K == N the
    // shapes happen to agree, which is exactly the setting where a broadcast-based
    // confusion would pass silently. The mulf chain check must still catch it: the
    // multiplied value is not the ws load.
    let err = match drive_src(&src, &sig_fp8(), &ces(), &[1]) {
        Ok(names) => panic!(
            "BAKED ({} ops) — a kernel multiplying by the activation instead of the \
             w_scale row was lowered as if it dequantized correctly",
            names.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.contains("does not multiply the dot result by the w_scale load"),
        "expected the mulf-chain refusal, got: {err:?}"
    );
}

/// THE SCALE READ FROM THE WRONG POSITION — ws and the OUTPUT swapped in the signature
/// list. Both are f16 pointers, so no type check separates them; only the mulf chain's
/// positional binding does. A kernel that stores through `desc_ws` and loads its scale
/// from `desc_o` would write its product into the one-row scale buffer and dequantize by
/// garbage read from the output's first row.
#[test]
fn a_swapped_scale_and_output_is_refused_by_positional_binding() {
    // Swap the two descriptors' BASES: the scale row loads from desc_o, the product
    // stores through desc_ws.
    let src = fixture()
        .replace(
            "    ws_desc = tl.make_tensor_descriptor(desc_ws, shape=[1, N],",
            "    ws_desc = tl.make_tensor_descriptor(desc_o, shape=[1, N],",
        )
        .replace(
            "    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N],",
            "    o_desc = tl.make_tensor_descriptor(desc_ws, shape=[M, N],",
        );
    let err = match drive_src(&src, &sig_fp8(), &ces(), &[1]) {
        Ok(names) => panic!(
            "BAKED ({} ops) — a kernel storing through the scale buffer and reading its \
             scale from the output lowered without the positional binding firing",
            names.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.contains("scale load does not read arg2")
            || err.contains("output store does not read arg3")
            || err.contains("the scale load does not read arg2"),
        "expected the positional-binding refusal, got: {err:?}"
    );
}

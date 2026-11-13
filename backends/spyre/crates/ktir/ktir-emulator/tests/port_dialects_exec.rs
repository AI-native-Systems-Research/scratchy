#![allow(
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::approx_constant
)]
// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Port of `tests/test_dialects_exec.py` — dialect EXECUTION handlers.
//!
//! The Python test calls `dispatch(op_type)(op, context, env)` directly with
//! hand-built `Operation` objects and checks the numeric result. The Rust
//! equivalent uses the locked execution seam: a `Dispatch`, a single-core
//! `CoreContext` (via `single_core_context`), and an `ExecutionEnv`. Each case
//! seeds operands with `ctx.set_value`, then either dispatches the handler
//! directly ([`run_op`]) or runs the op through `execute_op` (so region-bodied
//! ops — linalg.reduce/generic, tensor.generate, scf.if — dispatch their nested
//! ops through the real registry, exactly as Python's `_exec_region` does).
//!
//! Translation notes (Python -> Rust)
//! ----------------------------------
//! * Python scalars are `np.float16`/`int`/`bool`; Rust binds them as
//!   `Value::Scalar(Scalar::F32 | I64 | Bool)` or `Value::Index`. Float ops
//!   return `Scalar::F32` (Python widens f16 -> we keep f32), integer scalar ops
//!   return `Scalar::I64`, comparisons return `Scalar::Bool`, index casts return
//!   `Value::Index`. Tiles store a flat `Vec<f32>` + `DType`.
//! * `arith.constant` integer scalar -> `Scalar::I64` (Python returns the int).
//! * `func.return` in the Rust slice is a value-less no-op (it does not surface
//!   the operand), so `test_return_with_value` has no faithful value check; the
//!   no-value form (`-> None`) is checked, and the value form is noted skipped.
//! * `scf.yield` returns a `Value::Tuple` (the `_YieldResult` analogue); its
//!   `.values` list maps to the tuple's elements.
//! * Python's symbolic-coordinate-set specialisation tests on
//!   `construct_memory_view` exercise the eager BoxSet specialise step, which the
//!   Rust `construct_memory_view` slice does not perform (it stores the raw
//!   `coordinate_set` AffineSet and does not bind dynamic dims). Those two cases
//!   are noted skipped. The symbolic-`access_tile_set` rejection is also not in
//!   the Rust slice and is noted skipped.
//! * The xfail Python cases (multi-axis reduce, outs-init folding) map to known
//!   Rust limitations and are left as `#[ignore]` stubs.
//! * `arith.bitcast` is not registered in the Rust crate; its three cases are
//!   noted skipped.

mod common;

use common::Ops;
use ktir_emulator::attrkey::AttrKey;
use ktir_emulator::context::CoreContext;
use ktir_emulator::dialects::Dispatch;
use ktir_emulator::dtypes::DType;
use ktir_emulator::env::{ExecutionEnv, GridExecutor};
use ktir_emulator::interpreter::{execute_op, single_core_context};
use ktir_emulator::ir::{Attr, Operation, Scalar, Value};
use ktir_emulator::memory::{STICK_BYTES, SpyreMemoryHierarchy};
use ktir_emulator::opkind::OpKind;
use ktir_emulator::tile::Tile;
use std::rc::Rc;

// ===========================================================================
// Harness
// ===========================================================================

/// Build `%r = kind operands...` and dispatch its handler directly (the Python
/// `_call` path), seeding operands first. Returns the produced `Value`.
fn run_op(kind: OpKind, operands: &[&'static str], seed: &[(&'static str, Value)]) -> Value {
    run_op_try(kind, operands, seed).unwrap_or_else(|e| panic!("op {kind:?} failed: {e}"))
}

/// Like [`run_op`] but surfaces the handler's `Result` (for error-path cases).
fn run_op_try(
    kind: OpKind,
    operands: &[&'static str],
    seed: &[(&'static str, Value)],
) -> Result<Value, String> {
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), kind, operands);
    run_built_try(&mut ir_ops, &o, seed)
}

/// Like [`run_op_try`], but against an already-built op (for attr/region-bearing
/// cases the caller assembled with its own [`Ops`]).
fn run_built_try(
    ir_ops: &mut Ops,
    o: &Operation<'static>,
    seed: &[(&'static str, Value)],
) -> Result<Value, String> {
    let dispatch = Dispatch::new();
    let grid = GridExecutor::new((1, 1, 1));
    let env = ExecutionEnv::new(&dispatch, &grid);
    let mut ctx = single_core_context();
    for (n, v) in seed {
        ctx.set_value(ir_ops.ssa(n), v.clone());
    }
    let handler = dispatch
        .handler(o.op_type)
        .unwrap_or_else(|| panic!("no handler for {:?}", o.op_type));
    handler(o, &mut ctx, &env)
        .map(|out| out.unwrap_or_else(|| panic!("op {:?} produced no value", o.op_type)))
}

fn run_built(ir_ops: &mut Ops, o: &Operation<'static>, seed: &[(&'static str, Value)]) -> Value {
    run_built_try(ir_ops, o, seed).unwrap_or_else(|e| panic!("op {:?} failed: {e}", o.op_type))
}

/// Run a region-bodied op through `execute_op`, which threads nested ops through
/// the real registry (the Python `_exec_region` override). Returns the produced
/// value.
fn run_op_execute(
    ir_ops: &mut Ops,
    o: &Operation<'static>,
    seed: &[(&'static str, Value)],
) -> Value {
    let dispatch = Dispatch::new();
    let grid = GridExecutor::new((1, 1, 1));
    let env = ExecutionEnv::new(&dispatch, &grid);
    let mut ctx = single_core_context();
    for (n, v) in seed {
        ctx.set_value(ir_ops.ssa(n), v.clone());
    }
    execute_op(o, &mut ctx, &env)
        .unwrap_or_else(|e| panic!("execute_op {:?} failed: {e}", o.op_type))
        .unwrap_or_else(|| panic!("op {:?} produced no value", o.op_type))
}

fn sf(x: f32) -> Value {
    Value::Scalar(Scalar::F32(x))
}
fn si(x: i64) -> Value {
    Value::Scalar(Scalar::I64(x))
}
fn idx(x: i64) -> Value {
    Value::Index(x)
}
fn f16_tile(data: &[f32]) -> Value {
    Value::Tile(Tile::compute(data.to_vec(), DType::F16, vec![data.len()]))
}
fn tile_with(data: &[f32], dt: DType, shape: &[usize]) -> Value {
    Value::Tile(Tile::compute(data.to_vec(), dt, shape.to_vec()))
}

fn as_tile(v: &Value) -> &Tile {
    match v {
        Value::Tile(t) => t,
        other => panic!("expected Tile, got {other:?}"),
    }
}
fn as_f32(v: &Value) -> f32 {
    match v {
        Value::Scalar(s) => s.as_f32().expect("float scalar"),
        other => panic!("expected float scalar, got {other:?}"),
    }
}
fn as_i64(v: &Value) -> i64 {
    match v {
        Value::Scalar(s) => s.as_i64().expect("int scalar"),
        Value::Index(i) => *i,
        other => panic!("expected int scalar, got {other:?}"),
    }
}
fn as_bool(v: &Value) -> bool {
    match v {
        Value::Scalar(Scalar::Bool(b)) => *b,
        other => panic!("expected bool, got {other:?}"),
    }
}

fn close(a: f32, b: f32, tol: f32) {
    assert!((a - b).abs() <= tol, "{a} != {b} (tol {tol})");
}
fn data_close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= tol, "{a:?} != {b:?} (tol {tol})");
    }
}

// ===========================================================================
// arith float (TestArithFloat)
// ===========================================================================

#[test]
fn addf_tiles() {
    let r = run_op(
        OpKind::ArithAddf,
        &["%a", "%b"],
        &[("%a", f16_tile(&[1.0, 2.0])), ("%b", f16_tile(&[3.0, 4.0]))],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![4.0, 6.0]);
}

#[test]
fn addf_scalars() {
    let r = run_op(
        OpKind::ArithAddf,
        &["%a", "%b"],
        &[("%a", sf(2.0)), ("%b", sf(3.0))],
    );
    close(as_f32(&r), 5.0, 1e-2);
}

#[test]
fn addf_scalar_tile() {
    let r = run_op(
        OpKind::ArithAddf,
        &["%a", "%b"],
        &[("%a", sf(1.0)), ("%b", f16_tile(&[1.0, 2.0, 3.0]))],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![2.0, 3.0, 4.0]);
}

#[test]
fn addf_tile_scalar() {
    let r = run_op(
        OpKind::ArithAddf,
        &["%a", "%b"],
        &[("%a", f16_tile(&[1.0, 2.0, 3.0])), ("%b", sf(1.0))],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![2.0, 3.0, 4.0]);
}

#[test]
fn subf_scalar_tile() {
    let r = run_op(
        OpKind::ArithSubf,
        &["%a", "%b"],
        &[("%a", sf(10.0)), ("%b", f16_tile(&[1.0, 2.0, 3.0]))],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![9.0, 8.0, 7.0]);
}

#[test]
fn mulf_tile_scalar() {
    let r = run_op(
        OpKind::ArithMulf,
        &["%a", "%b"],
        &[("%a", f16_tile(&[1.0, 2.0, 3.0])), ("%b", sf(2.0))],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn mulf_scalar_tile() {
    let r = run_op(
        OpKind::ArithMulf,
        &["%a", "%b"],
        &[("%a", sf(3.0)), ("%b", f16_tile(&[1.0, 2.0, 3.0]))],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![3.0, 6.0, 9.0]);
}

#[test]
fn divf_tile_scalar() {
    let r = run_op(
        OpKind::ArithDivf,
        &["%a", "%b"],
        &[("%a", f16_tile(&[4.0, 6.0, 8.0])), ("%b", sf(2.0))],
    );
    data_close(&as_tile(&r).as_f32(), &[2.0, 3.0, 4.0], 1e-2);
}

#[test]
fn divf_scalar_tile() {
    let r = run_op(
        OpKind::ArithDivf,
        &["%a", "%b"],
        &[("%a", sf(12.0)), ("%b", f16_tile(&[2.0, 3.0, 4.0]))],
    );
    data_close(&as_tile(&r).as_f32(), &[6.0, 4.0, 3.0], 1e-2);
}

#[test]
fn maxf() {
    let r = run_op(
        OpKind::ArithMaxf,
        &["%a", "%b"],
        &[
            ("%a", f16_tile(&[1.0, 5.0, 3.0])),
            ("%b", f16_tile(&[4.0, 2.0, 6.0])),
        ],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![4.0, 5.0, 6.0]);
}

#[test]
fn maxnumf() {
    let r = run_op(
        OpKind::ArithMaxnumf,
        &["%a", "%b"],
        &[("%a", f16_tile(&[1.0, 5.0])), ("%b", f16_tile(&[4.0, 2.0]))],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![4.0, 5.0]);
}

#[test]
fn maximumf_tiles() {
    let r = run_op(
        OpKind::ArithMaximumf,
        &["%a", "%b"],
        &[
            ("%a", f16_tile(&[1.0, 5.0, 3.0])),
            ("%b", f16_tile(&[4.0, 2.0, 6.0])),
        ],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![4.0, 5.0, 6.0]);
}

#[test]
fn minimumf() {
    let r = run_op(
        OpKind::ArithMinimumf,
        &["%a", "%b"],
        &[
            ("%a", f16_tile(&[1.0, 5.0, 3.0])),
            ("%b", f16_tile(&[4.0, 2.0, 6.0])),
        ],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn minnumf() {
    let r = run_op(
        OpKind::ArithMinnumf,
        &["%a", "%b"],
        &[("%a", f16_tile(&[1.0, 5.0])), ("%b", f16_tile(&[4.0, 2.0]))],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![1.0, 2.0]);
}

#[test]
fn minnumf_nan() {
    // fmin(NaN, 2) -> 2 ; fmin(3, NaN) -> 3 (NaN non-propagating)
    let r = run_op(
        OpKind::ArithMinnumf,
        &["%a", "%b"],
        &[
            ("%a", f16_tile(&[f32::NAN, 3.0])),
            ("%b", f16_tile(&[2.0, f32::NAN])),
        ],
    );
    let t = as_tile(&r);
    assert_eq!(t.as_f32()[0], 2.0);
    assert_eq!(t.as_f32()[1], 3.0);
}

#[test]
fn extf_promotes_to_f32() {
    let r = run_op(OpKind::ArithExtf, &["%a"], &[("%a", f16_tile(&[1.0, 2.0]))]);
    let t = as_tile(&r);
    assert_eq!(t.dtype, DType::F32);
    assert_eq!(t.as_f32().to_vec(), vec![1.0, 2.0]);
}

#[test]
fn truncf_passthrough_values() {
    // Python returns the same Tile object; in Rust we check the values round-trip
    // through f16 unchanged (1.0, 2.0 are exactly representable).
    let r = run_op(
        OpKind::ArithTruncf,
        &["%a"],
        &[("%a", f16_tile(&[1.0, 2.0]))],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![1.0, 2.0]);
}

// ===========================================================================
// arith int (TestArithInt)
// ===========================================================================

#[test]
fn addi_tile_broadcast() {
    let r = run_op(
        OpKind::ArithAddi,
        &["%a", "%b"],
        &[("%a", f16_tile(&[1.0, 2.0, 3.0])), ("%b", idx(5))],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![6.0, 7.0, 8.0]);
}

#[test]
fn addi_broadcast_tile() {
    let r = run_op(
        OpKind::ArithAddi,
        &["%a", "%b"],
        &[("%a", idx(10)), ("%b", f16_tile(&[1.0, 2.0, 3.0]))],
    );
    assert!(matches!(r, Value::Tile(_)));
}

#[test]
fn muli_tile_broadcast() {
    let r = run_op(
        OpKind::ArithMuli,
        &["%a", "%b"],
        &[("%a", f16_tile(&[1.0, 2.0, 3.0])), ("%b", idx(3))],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![3.0, 6.0, 9.0]);
}

#[test]
fn muli_broadcast_tile() {
    let r = run_op(
        OpKind::ArithMuli,
        &["%a", "%b"],
        &[("%a", idx(2)), ("%b", f16_tile(&[1.0, 2.0, 3.0]))],
    );
    assert!(matches!(r, Value::Tile(_)));
}

#[test]
fn subi() {
    let r = run_op(
        OpKind::ArithSubi,
        &["%a", "%b"],
        &[("%a", idx(10)), ("%b", idx(3))],
    );
    assert_eq!(as_i64(&r), 7);
}

#[test]
fn remui() {
    let r = run_op(
        OpKind::ArithRemui,
        &["%a", "%b"],
        &[("%a", idx(10)), ("%b", idx(3))],
    );
    assert_eq!(as_i64(&r), 1);
}

#[test]
fn divsi_scalar() {
    let r = run_op(
        OpKind::ArithDivsi,
        &["%a", "%b"],
        &[("%a", idx(7)), ("%b", idx(2))],
    );
    assert_eq!(as_i64(&r), 3);
}

#[test]
fn divsi_truncates_toward_zero() {
    let r = run_op(
        OpKind::ArithDivsi,
        &["%a", "%b"],
        &[("%a", si(-7)), ("%b", si(2))],
    );
    assert_eq!(as_i64(&r), -3);
}

#[test]
fn remsi_scalar() {
    let r = run_op(
        OpKind::ArithRemsi,
        &["%a", "%b"],
        &[("%a", idx(7)), ("%b", idx(3))],
    );
    assert_eq!(as_i64(&r), 1);
}

#[test]
fn remsi_negative() {
    // -7 % 3 = -1 (truncating), matching MLIR remsi sign-of-dividend.
    let r = run_op(
        OpKind::ArithRemsi,
        &["%a", "%b"],
        &[("%a", si(-7)), ("%b", si(3))],
    );
    assert_eq!(as_i64(&r), -1);
}

#[test]
fn ceildivsi_scalar() {
    let r = run_op(
        OpKind::ArithCeildivsi,
        &["%a", "%b"],
        &[("%a", idx(7)), ("%b", idx(2))],
    );
    assert_eq!(as_i64(&r), 4);
}

#[test]
fn ceildivui_scalar() {
    let r = run_op(
        OpKind::ArithCeildivui,
        &["%a", "%b"],
        &[("%a", idx(7)), ("%b", idx(2))],
    );
    assert_eq!(as_i64(&r), 4);
}

#[test]
fn minsi_scalar() {
    let r = run_op(
        OpKind::ArithMinsi,
        &["%a", "%b"],
        &[("%a", idx(3)), ("%b", idx(7))],
    );
    assert_eq!(as_i64(&r), 3);
}

#[test]
fn minsi_negative() {
    let r = run_op(
        OpKind::ArithMinsi,
        &["%a", "%b"],
        &[("%a", si(-5)), ("%b", si(2))],
    );
    assert_eq!(as_i64(&r), -5);
}

#[test]
fn maxsi_scalar() {
    let r = run_op(
        OpKind::ArithMaxsi,
        &["%a", "%b"],
        &[("%a", idx(3)), ("%b", idx(7))],
    );
    assert_eq!(as_i64(&r), 7);
}

#[test]
fn minsi_tiles() {
    let r = run_op(
        OpKind::ArithMinsi,
        &["%a", "%b"],
        &[
            ("%a", tile_with(&[1.0, 5.0, 3.0], DType::I32, &[3])),
            ("%b", tile_with(&[4.0, 2.0, 6.0], DType::I32, &[3])),
        ],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn maxsi_tiles() {
    let r = run_op(
        OpKind::ArithMaxsi,
        &["%a", "%b"],
        &[
            ("%a", tile_with(&[1.0, 5.0, 3.0], DType::I32, &[3])),
            ("%b", tile_with(&[4.0, 2.0, 6.0], DType::I32, &[3])),
        ],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![4.0, 5.0, 6.0]);
}

#[test]
fn minui_scalar() {
    let r = run_op(
        OpKind::ArithMinui,
        &["%a", "%b"],
        &[("%a", idx(3)), ("%b", idx(7))],
    );
    assert_eq!(as_i64(&r), 3);
}

#[test]
fn maxui_scalar() {
    let r = run_op(
        OpKind::ArithMaxui,
        &["%a", "%b"],
        &[("%a", idx(3)), ("%b", idx(7))],
    );
    assert_eq!(as_i64(&r), 7);
}

#[test]
fn floordivsi_scalar() {
    let r = run_op(
        OpKind::ArithFloordivsi,
        &["%a", "%b"],
        &[("%a", idx(7)), ("%b", idx(2))],
    );
    assert_eq!(as_i64(&r), 3);
}

#[test]
fn andi_scalar() {
    let r = run_op(
        OpKind::ArithAndi,
        &["%a", "%b"],
        &[("%a", idx(0b1010)), ("%b", idx(0b1100))],
    );
    assert_eq!(as_i64(&r), 0b1000);
}

#[test]
fn ori_scalar() {
    let r = run_op(
        OpKind::ArithOri,
        &["%a", "%b"],
        &[("%a", idx(0b1010)), ("%b", idx(0b1100))],
    );
    assert_eq!(as_i64(&r), 0b1110);
}

#[test]
fn xori_scalar() {
    let r = run_op(
        OpKind::ArithXori,
        &["%a", "%b"],
        &[("%a", idx(0b1010)), ("%b", idx(0b1100))],
    );
    assert_eq!(as_i64(&r), 0b0110);
}

#[test]
fn shli_scalar() {
    let r = run_op(
        OpKind::ArithShli,
        &["%a", "%b"],
        &[("%a", idx(1)), ("%b", idx(3))],
    );
    assert_eq!(as_i64(&r), 8);
}

#[test]
fn shrsi_scalar() {
    let r = run_op(
        OpKind::ArithShrsi,
        &["%a", "%b"],
        &[("%a", idx(8)), ("%b", idx(2))],
    );
    assert_eq!(as_i64(&r), 2);
}

#[test]
fn shrui_scalar() {
    let r = run_op(
        OpKind::ArithShrui,
        &["%a", "%b"],
        &[("%a", idx(8)), ("%b", idx(2))],
    );
    assert_eq!(as_i64(&r), 2);
}

#[test]
fn andi_tile() {
    let r = run_op(
        OpKind::ArithAndi,
        &["%a", "%b"],
        &[
            (
                "%a",
                tile_with(
                    &[0b1010 as f32, 0b1100 as f32, 0b1111 as f32],
                    DType::I32,
                    &[3],
                ),
            ),
            ("%b", idx(0b1010)),
        ],
    );
    assert_eq!(
        as_tile(&r).as_f32().to_vec(),
        vec![0b1010 as f32, 0b1000 as f32, 0b1010 as f32]
    );
}

// ===========================================================================
// arith float unary + cmpf (TestArithFloatUnary)
// ===========================================================================

#[test]
fn negf_scalar() {
    let r = run_op(OpKind::ArithNegf, &["%a"], &[("%a", sf(3.0))]);
    close(as_f32(&r), -3.0, 1e-2);
}

#[test]
fn negf_tile() {
    let r = run_op(
        OpKind::ArithNegf,
        &["%a"],
        &[("%a", f16_tile(&[1.0, -2.0, 3.0]))],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![-1.0, 2.0, -3.0]);
}

#[test]
fn absf_scalar() {
    let r = run_op(OpKind::ArithAbsf, &["%a"], &[("%a", sf(-5.0))]);
    close(as_f32(&r), 5.0, 1e-2);
}

#[test]
fn absf_tile() {
    let r = run_op(
        OpKind::ArithAbsf,
        &["%a"],
        &[("%a", f16_tile(&[-1.0, 2.0, -3.0]))],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn remf_scalars() {
    let r = run_op(
        OpKind::ArithRemf,
        &["%a", "%b"],
        &[("%a", sf(5.0)), ("%b", sf(3.0))],
    );
    close(as_f32(&r), 2.0, 1e-2);
}

#[test]
fn minf_tiles() {
    let r = run_op(
        OpKind::ArithMinf,
        &["%a", "%b"],
        &[
            ("%a", f16_tile(&[1.0, 5.0, 3.0])),
            ("%b", f16_tile(&[2.0, 4.0, 3.0])),
        ],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![1.0, 4.0, 3.0]);
}

#[test]
fn minimumf_tiles() {
    let r = run_op(
        OpKind::ArithMinimumf,
        &["%a", "%b"],
        &[
            ("%a", f16_tile(&[1.0, 5.0, 3.0])),
            ("%b", f16_tile(&[2.0, 4.0, 3.0])),
        ],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![1.0, 4.0, 3.0]);
}

fn run_predicated(
    kind: OpKind,
    pred: &'static str,
    operands: &[&'static str],
    seed: &[(&'static str, Value)],
) -> Value {
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), kind, operands);
    let o = ir_ops.attr(o, AttrKey::Predicate, Attr::Str(pred));
    run_built(&mut ir_ops, &o, seed)
}

#[test]
fn cmpf_olt_scalar() {
    let r = run_predicated(
        OpKind::ArithCmpf,
        "olt",
        &["%a", "%b"],
        &[("%a", sf(1.0)), ("%b", sf(2.0))],
    );
    assert!(as_bool(&r));
}

#[test]
fn cmpf_ogt_scalar() {
    let r = run_predicated(
        OpKind::ArithCmpf,
        "ogt",
        &["%a", "%b"],
        &[("%a", sf(3.0)), ("%b", sf(2.0))],
    );
    assert!(as_bool(&r));
}

#[test]
fn cmpf_oeq_tile() {
    let r = run_predicated(
        OpKind::ArithCmpf,
        "oeq",
        &["%a", "%b"],
        &[
            ("%a", f16_tile(&[1.0, 2.0, 3.0])),
            ("%b", f16_tile(&[1.0, 0.0, 3.0])),
        ],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![1.0, 0.0, 1.0]); // bool stored as 0/1
}

// ===========================================================================
// arith new casts (TestArithNewCasts)
// ===========================================================================

#[test]
fn extui_scalar() {
    let r = run_op(OpKind::ArithExtui, &["%a"], &[("%a", idx(5))]);
    assert_eq!(as_i64(&r), 5);
}

#[test]
fn trunci_scalar() {
    let r = run_op(OpKind::ArithTrunci, &["%a"], &[("%a", idx(300))]);
    assert_eq!(as_i64(&r), 300);
}

#[test]
fn uitofp_scalar() {
    let r = run_op(OpKind::ArithUitofp, &["%a"], &[("%a", idx(4))]);
    close(as_f32(&r), 4.0, 1e-2);
}

#[test]
fn fptosi_scalar() {
    let r = run_op(OpKind::ArithFptosi, &["%a"], &[("%a", sf(3.7))]);
    assert_eq!(as_i64(&r), 3);
}

#[test]
fn fptoui_scalar() {
    let r = run_op(OpKind::ArithFptoui, &["%a"], &[("%a", sf(2.9))]);
    assert_eq!(as_i64(&r), 2);
}

#[test]
fn extui_tile() {
    let r = run_op(
        OpKind::ArithExtui,
        &["%a"],
        &[("%a", tile_with(&[1.0, 2.0, 3.0], DType::I32, &[3]))],
    );
    let t = as_tile(&r);
    assert!(matches!(t.dtype, DType::I64));
}

#[test]
fn fptosi_tile() {
    let r = run_op(
        OpKind::ArithFptosi,
        &["%a"],
        &[("%a", f16_tile(&[1.7, 2.3, -3.9]))],
    );
    let t = as_tile(&r);
    assert_eq!(t.dtype, DType::I32);
    assert_eq!(t.as_f32().to_vec(), vec![1.0, 2.0, -3.0]);
}

// ===========================================================================
// arith casts / constants (TestArithCastsConstants)
// ===========================================================================

#[test]
fn constant_scalar() {
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::ArithConstant, &[]);
    let o = ir_ops.attr(o, AttrKey::Value, Attr::Int(42));
    let r = run_built(&mut ir_ops, &o, &[]);
    assert_eq!(as_i64(&r), 42);
}

#[test]
fn constant_tensor() {
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::ArithConstant, &[]);
    let o = ir_ops.attr(o, AttrKey::Value, Attr::Float(0.0));
    let o = ir_ops.attr(o, AttrKey::IsTensor, Attr::Bool(true));
    let o = ir_ops.attr(o, AttrKey::Shape, ir_ops.int_list(vec![4]));
    let o = ir_ops.attr(o, AttrKey::Dtype, Attr::Dtype(DType::F16));
    let r = run_built(&mut ir_ops, &o, &[]);
    let t = as_tile(&r);
    assert_eq!(t.shape, vec![4]);
    assert!(t.as_f32().iter().all(|&x| x == 0.0));
}

#[test]
fn constant_dense_list() {
    // dense<[16, 32]> materializes the list element-by-element.
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::ArithConstant, &[]);
    let o = ir_ops.attr(o, AttrKey::Value, ir_ops.int_list(vec![16, 32]));
    let o = ir_ops.attr(o, AttrKey::Shape, ir_ops.int_list(vec![2]));
    let o = ir_ops.attr(o, AttrKey::Dtype, Attr::Dtype(DType::I32));
    let o = ir_ops.attr(o, AttrKey::IsTensor, Attr::Bool(true));
    let o = ir_ops.attr(o, AttrKey::DenseList, Attr::Bool(true));
    let r = run_built(&mut ir_ops, &o, &[]);
    let t = as_tile(&r);
    assert_eq!(t.shape, vec![2]);
    assert_eq!(t.as_f32().to_vec(), vec![16.0, 32.0]);
}

#[test]
fn extsi() {
    let r = run_op(OpKind::ArithExtsi, &["%a"], &[("%a", idx(5))]);
    assert_eq!(as_i64(&r), 5);
}

#[test]
fn index_cast() {
    let r = run_op(OpKind::ArithIndexCast, &["%a"], &[("%a", idx(7))]);
    assert_eq!(as_i64(&r), 7);
}

#[test]
fn index_castui() {
    let r = run_op(OpKind::ArithIndexCastui, &["%a"], &[("%a", idx(7))]);
    assert_eq!(as_i64(&r), 7);
}

#[test]
fn convertf_f16_to_f32() {
    let r = run_op(
        OpKind::ArithConvertf,
        &["%a"],
        &[("%a", f16_tile(&[1.0, 2.0]))],
    );
    assert_eq!(as_tile(&r).dtype, DType::F32);
}

#[test]
fn convertf_f32_to_f16() {
    let r = run_op(
        OpKind::ArithConvertf,
        &["%a"],
        &[("%a", tile_with(&[1.0, 2.0], DType::F32, &[2]))],
    );
    assert_eq!(as_tile(&r).dtype, DType::F16);
}

#[test]
fn sitofp_scalar() {
    let r = run_op(OpKind::ArithSitofp, &["%a"], &[("%a", idx(3))]);
    close(as_f32(&r), 3.0, 1e-2);
}

#[test]
fn sitofp_respects_result_type_f16() {
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::ArithSitofp, &["%a"]);
    let o = ir_ops.ty(o, ktir_emulator::irtype::IrType::Scalar(DType::F16));
    let r = run_built(&mut ir_ops, &o, &[("%a", idx(3))]);
    // Scalar path returns an F32 scalar; the dtype is carried on tiles only.
    close(as_f32(&r), 3.0, 1e-2);
}

#[test]
fn sitofp_respects_result_type_f32() {
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::ArithSitofp, &["%a"]);
    let o = ir_ops.ty(o, ktir_emulator::irtype::IrType::Scalar(DType::F32));
    let r = run_built(
        &mut ir_ops,
        &o,
        &[("%a", tile_with(&[1.0, -2.0], DType::I32, &[2]))],
    );
    let t = as_tile(&r);
    assert_eq!(t.dtype, DType::F32);
    assert_eq!(t.as_f32().to_vec(), vec![1.0, -2.0]);
}

// ===========================================================================
// arith cmpi / select (TestArithCmpiSelect)
// ===========================================================================

#[test]
fn cmpi_scalar() {
    let r = run_predicated(
        OpKind::ArithCmpi,
        "slt",
        &["%a", "%b"],
        &[("%a", idx(1)), ("%b", idx(2))],
    );
    assert!(as_bool(&r));
}

#[test]
fn cmpi_tile() {
    let r = run_predicated(
        OpKind::ArithCmpi,
        "slt",
        &["%a", "%b"],
        &[
            ("%a", f16_tile(&[1.0, 5.0, 3.0])),
            ("%b", f16_tile(&[2.0, 4.0, 3.0])),
        ],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![1.0, 0.0, 0.0]);
}

#[test]
fn cmpi_ult() {
    let r = run_predicated(
        OpKind::ArithCmpi,
        "ult",
        &["%a", "%b"],
        &[("%a", idx(1)), ("%b", idx(2))],
    );
    assert!(as_bool(&r));
}

#[test]
fn cmpi_uge_tile() {
    let r = run_predicated(
        OpKind::ArithCmpi,
        "uge",
        &["%a", "%b"],
        &[
            ("%a", f16_tile(&[1.0, 5.0, 3.0])),
            ("%b", f16_tile(&[2.0, 4.0, 3.0])),
        ],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![0.0, 1.0, 1.0]);
}

#[test]
fn select_scalar() {
    let r = run_op(
        OpKind::ArithSelect,
        &["%cond", "%t", "%f"],
        &[
            ("%cond", Value::Scalar(Scalar::Bool(true))),
            ("%t", idx(10)),
            ("%f", idx(20)),
        ],
    );
    assert_eq!(as_i64(&r), 10);
}

#[test]
fn select_tile() {
    let r = run_op(
        OpKind::ArithSelect,
        &["%cond", "%t", "%f"],
        &[
            ("%cond", tile_with(&[1.0, 0.0, 1.0], DType::Bool, &[3])),
            ("%t", f16_tile(&[1.0, 2.0, 3.0])),
            ("%f", f16_tile(&[4.0, 5.0, 6.0])),
        ],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![1.0, 5.0, 3.0]);
}

// ===========================================================================
// arith.cmpf (TestArithCmpf)
// ===========================================================================

#[test]
fn cmpf_olt_tile() {
    let r = run_predicated(
        OpKind::ArithCmpf,
        "olt",
        &["%a", "%b"],
        &[
            ("%a", f16_tile(&[1.0, 5.0, 3.0])),
            ("%b", f16_tile(&[2.0, 4.0, 3.0])),
        ],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![1.0, 0.0, 0.0]);
}

#[test]
fn cmpf_oge_tile() {
    let r = run_predicated(
        OpKind::ArithCmpf,
        "oge",
        &["%a", "%b"],
        &[
            ("%a", f16_tile(&[1.0, 5.0, 3.0])),
            ("%b", f16_tile(&[2.0, 4.0, 3.0])),
        ],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![0.0, 1.0, 1.0]);
}

#[test]
fn cmpf_olt_nan() {
    // Ordered predicates are false when either operand is NaN.
    let r = run_predicated(
        OpKind::ArithCmpf,
        "olt",
        &["%a", "%b"],
        &[
            ("%a", f16_tile(&[f32::NAN, 1.0])),
            ("%b", f16_tile(&[2.0, f32::NAN])),
        ],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![0.0, 0.0]);
}

#[test]
fn cmpf_ueq_nan() {
    // Unordered predicates return true when either operand is NaN.
    let r = run_predicated(
        OpKind::ArithCmpf,
        "ueq",
        &["%a", "%b"],
        &[
            ("%a", f16_tile(&[f32::NAN, 3.0])),
            ("%b", f16_tile(&[2.0, 3.0])),
        ],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![1.0, 1.0]);
}

#[test]
fn cmpf_ord_uno() {
    let a = f16_tile(&[f32::NAN, 3.0]);
    let b = f16_tile(&[2.0, 4.0]);
    let r_ord = run_predicated(
        OpKind::ArithCmpf,
        "ord",
        &["%a", "%b"],
        &[("%a", a.clone()), ("%b", b.clone())],
    );
    assert_eq!(as_tile(&r_ord).as_f32().to_vec(), vec![0.0, 1.0]);
    let r_uno = run_predicated(
        OpKind::ArithCmpf,
        "uno",
        &["%a", "%b"],
        &[("%a", a), ("%b", b)],
    );
    assert_eq!(as_tile(&r_uno).as_f32().to_vec(), vec![1.0, 0.0]);
}

// ===========================================================================
// math (TestMath)
// ===========================================================================

#[test]
fn math_exp_tile() {
    let r = run_op(OpKind::MathExp, &["%x"], &[("%x", f16_tile(&[0.0, 1.0]))]);
    data_close(&as_tile(&r).as_f32(), &[1.0, std::f32::consts::E], 1e-2);
}

#[test]
fn math_exp_scalar() {
    let r = run_op(OpKind::MathExp, &["%x"], &[("%x", sf(0.0))]);
    close(as_f32(&r), 1.0, 1e-2);
}

#[test]
fn math_sqrt_tile() {
    let r = run_op(
        OpKind::MathSqrt,
        &["%x"],
        &[("%x", f16_tile(&[4.0, 9.0, 16.0]))],
    );
    data_close(&as_tile(&r).as_f32(), &[2.0, 3.0, 4.0], 1e-2);
}

#[test]
fn math_sqrt_scalar() {
    let r = run_op(OpKind::MathSqrt, &["%x"], &[("%x", sf(4.0))]);
    close(as_f32(&r), 2.0, 1e-2);
}

#[test]
fn math_log_tile() {
    let r = run_op(
        OpKind::MathLog,
        &["%x"],
        &[("%x", f16_tile(&[1.0, 2.0, 4.0]))],
    );
    data_close(
        &as_tile(&r).as_f32(),
        &[0.0, 2.0f32.ln(), 4.0f32.ln()],
        1e-2,
    );
}

#[test]
fn math_log_scalar() {
    let r = run_op(OpKind::MathLog, &["%x"], &[("%x", sf(1.0))]);
    close(as_f32(&r), 0.0, 1e-2);
}

#[test]
fn math_rsqrt_tile() {
    let r = run_op(
        OpKind::MathRsqrt,
        &["%x"],
        &[("%x", f16_tile(&[1.0, 4.0, 16.0]))],
    );
    data_close(&as_tile(&r).as_f32(), &[1.0, 0.5, 0.25], 1e-2);
}

#[test]
fn math_rsqrt_scalar() {
    let r = run_op(OpKind::MathRsqrt, &["%x"], &[("%x", sf(4.0))]);
    close(as_f32(&r), 0.5, 1e-2);
}

#[test]
fn math_log2_tile() {
    let r = run_op(
        OpKind::MathLog2,
        &["%x"],
        &[("%x", f16_tile(&[1.0, 2.0, 8.0]))],
    );
    data_close(&as_tile(&r).as_f32(), &[0.0, 1.0, 3.0], 1e-2);
}

#[test]
fn math_log2_scalar() {
    let r = run_op(OpKind::MathLog2, &["%x"], &[("%x", sf(8.0))]);
    close(as_f32(&r), 3.0, 1e-2);
}

#[test]
fn math_log1p_tile() {
    let r = run_op(
        OpKind::MathLog1p,
        &["%x"],
        &[("%x", f16_tile(&[0.0, 1.0, 2.0]))],
    );
    data_close(
        &as_tile(&r).as_f32(),
        &[0.0, 1.0f32.ln_1p(), 2.0f32.ln_1p()],
        1e-2,
    );
}

#[test]
fn math_log1p_scalar() {
    let r = run_op(OpKind::MathLog1p, &["%x"], &[("%x", sf(0.0))]);
    close(as_f32(&r), 0.0, 1e-2);
}

#[test]
fn math_tanh_tile() {
    let r = run_op(
        OpKind::MathTanh,
        &["%x"],
        &[("%x", f16_tile(&[0.0, 1.0, -1.0]))],
    );
    data_close(
        &as_tile(&r).as_f32(),
        &[0.0, 1.0f32.tanh(), (-1.0f32).tanh()],
        1e-2,
    );
}

#[test]
fn math_tanh_scalar() {
    let r = run_op(OpKind::MathTanh, &["%x"], &[("%x", sf(0.0))]);
    close(as_f32(&r), 0.0, 1e-2);
}

#[test]
fn math_sin_tile() {
    let r = run_op(
        OpKind::MathSin,
        &["%x"],
        &[("%x", f16_tile(&[0.0, 1.5708, 3.1416]))],
    );
    data_close(
        &as_tile(&r).as_f32(),
        &[0.0, 1.5708f32.sin(), 3.1416f32.sin()],
        2e-2,
    );
}

#[test]
fn math_sin_scalar() {
    let r = run_op(OpKind::MathSin, &["%x"], &[("%x", sf(0.0))]);
    close(as_f32(&r), 0.0, 1e-2);
}

#[test]
fn math_cos_tile() {
    let r = run_op(
        OpKind::MathCos,
        &["%x"],
        &[("%x", f16_tile(&[0.0, 1.5708, 3.1416]))],
    );
    data_close(
        &as_tile(&r).as_f32(),
        &[1.0, 1.5708f32.cos(), 3.1416f32.cos()],
        2e-2,
    );
}

#[test]
fn math_cos_scalar() {
    let r = run_op(OpKind::MathCos, &["%x"], &[("%x", sf(0.0))]);
    close(as_f32(&r), 1.0, 1e-2);
}

#[test]
fn math_absf_tile() {
    let r = run_op(
        OpKind::MathAbsf,
        &["%x"],
        &[("%x", f16_tile(&[-2.0, 0.0, 3.0]))],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![2.0, 0.0, 3.0]);
}

#[test]
fn math_absf_scalar() {
    let r = run_op(OpKind::MathAbsf, &["%x"], &[("%x", sf(-5.0))]);
    assert_eq!(as_f32(&r), 5.0);
}

#[test]
fn math_ceil_tile() {
    let r = run_op(
        OpKind::MathCeil,
        &["%x"],
        &[("%x", f16_tile(&[1.2, 2.7, -0.5]))],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![2.0, 3.0, 0.0]);
}

#[test]
fn math_ceil_scalar() {
    let r = run_op(OpKind::MathCeil, &["%x"], &[("%x", sf(1.3))]);
    assert_eq!(as_f32(&r), 2.0);
}

#[test]
fn math_floor_tile() {
    let r = run_op(
        OpKind::MathFloor,
        &["%x"],
        &[("%x", f16_tile(&[1.2, 2.7, -0.5]))],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![1.0, 2.0, -1.0]);
}

#[test]
fn math_floor_scalar() {
    let r = run_op(OpKind::MathFloor, &["%x"], &[("%x", sf(1.7))]);
    assert_eq!(as_f32(&r), 1.0);
}

#[test]
fn math_powf_tile() {
    let r = run_op(
        OpKind::MathPowf,
        &["%a", "%b"],
        &[
            ("%a", f16_tile(&[2.0, 3.0, 4.0])),
            ("%b", f16_tile(&[2.0, 2.0, 0.5])),
        ],
    );
    data_close(&as_tile(&r).as_f32(), &[4.0, 9.0, 2.0], 1e-2);
}

#[test]
fn math_fma_tile() {
    let r = run_op(
        OpKind::MathFma,
        &["%a", "%b", "%c"],
        &[
            ("%a", f16_tile(&[2.0, 3.0])),
            ("%b", f16_tile(&[4.0, 5.0])),
            ("%c", f16_tile(&[1.0, 1.0])),
        ],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![9.0, 16.0]);
}

#[test]
fn math_erf_tile() {
    let r = run_op(
        OpKind::MathErf,
        &["%x"],
        &[("%x", f16_tile(&[0.0, 1.0, -1.0]))],
    );
    data_close(&as_tile(&r).as_f32(), &[0.0, 0.8427, -0.8427], 1e-2);
}

#[test]
fn math_erf_scalar() {
    let r = run_op(OpKind::MathErf, &["%x"], &[("%x", sf(0.0))]);
    close(as_f32(&r), 0.0, 1e-2);
}

#[test]
fn math_absi_tile() {
    let r = run_op(
        OpKind::MathAbsi,
        &["%x"],
        &[("%x", tile_with(&[-3.0, 0.0, 5.0], DType::I32, &[3]))],
    );
    assert_eq!(as_tile(&r).as_f32().to_vec(), vec![3.0, 0.0, 5.0]);
}

#[test]
fn math_absi_scalar() {
    let r = run_op(
        OpKind::MathAbsi,
        &["%x"],
        &[("%x", Value::Scalar(Scalar::I32(-7)))],
    );
    assert_eq!(as_i64(&r), 7);
}

#[test]
fn math_powf_scalar() {
    let r = run_op(
        OpKind::MathPowf,
        &["%a", "%b"],
        &[("%a", sf(2.0)), ("%b", sf(3.0))],
    );
    assert_eq!(as_f32(&r), 8.0);
}

#[test]
fn math_fma_scalar() {
    let r = run_op(
        OpKind::MathFma,
        &["%a", "%b", "%c"],
        &[("%a", sf(3.0)), ("%b", sf(4.0)), ("%c", sf(1.0))],
    );
    assert_eq!(as_f32(&r), 13.0);
}

// ===========================================================================
// linalg (TestLinalg)
// ===========================================================================

#[test]
fn reduce_along_dim() {
    // reduce a 1x4 tile along dim 1 -> sum 10.
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::LinalgReduce, &["%x"]);
    let o = ir_ops.attr(o, AttrKey::ReduceFn, Attr::Op(OpKind::ArithAddf));
    let o = ir_ops.attr(o, AttrKey::Dim, Attr::Int(1));
    let outs_var = ir_ops.ssas_attr(&["%init"]);
    let o = ir_ops.attr(o, AttrKey::OutsVar, outs_var);
    let r = run_op_execute(
        &mut ir_ops,
        &o,
        &[
            ("%x", tile_with(&[1.0, 2.0, 3.0, 4.0], DType::F16, &[1, 4])),
            ("%init", tile_with(&[0.0], DType::F16, &[1])),
        ],
    );
    let val = match &r {
        Value::Tile(t) => t.as_f32()[0],
        Value::Scalar(s) => s.as_f32().unwrap(),
        other => panic!("got {other:?}"),
    };
    close(val, 10.0, 0.1);
}

#[test]
fn reduce_full_collapse() {
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::LinalgReduce, &["%x"]);
    let o = ir_ops.attr(o, AttrKey::ReduceFn, Attr::Op(OpKind::ArithAddf));
    let r = run_op_execute(&mut ir_ops, &o, &[("%x", f16_tile(&[1.0, 2.0, 3.0, 4.0]))]);
    close(as_f32(&r), 10.0, 0.1);
}

#[test]
fn reduce_scalar_input() {
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::LinalgReduce, &["%x"]);
    let o = ir_ops.attr(o, AttrKey::ReduceFn, Attr::Op(OpKind::ArithAddf));
    let r = run_op_execute(&mut ir_ops, &o, &[("%x", sf(5.0))]);
    close(as_f32(&r), 5.0, 1e-2);
}

#[test]
fn reduce_explicit_region_single_op() {
    // (%in, %out) { %s = addf %in,%out ; yield %s } over a 1x4 tile, dim 1.
    let mut ir_ops = Ops::new();
    let s = ir_ops.op(Some("%s"), OpKind::ArithAddf, &["%in", "%out"]);
    let y = ir_ops.op(None, OpKind::LinalgYield, &["%s"]);
    let region = vec![s, y];
    let o = ir_ops.op(Some("%r"), OpKind::LinalgReduce, &["%x"]);
    let o = ir_ops.attr(o, AttrKey::Dim, Attr::Int(1));
    let outs_var = ir_ops.ssas_attr(&["%init"]);
    let o = ir_ops.attr(o, AttrKey::OutsVar, outs_var);
    let o = ir_ops.with_region(o, region);
    let r = run_op_execute(
        &mut ir_ops,
        &o,
        &[
            ("%x", tile_with(&[1.0, 2.0, 3.0, 4.0], DType::F16, &[1, 4])),
            ("%init", tile_with(&[0.0], DType::F16, &[1])),
        ],
    );
    let val = match &r {
        Value::Tile(t) => t.as_f32()[0],
        Value::Scalar(s) => s.as_f32().unwrap(),
        other => panic!("got {other:?}"),
    };
    close(val, 10.0, 0.1);
}

#[test]
fn reduce_multiop_combiner() {
    // MULTI-OP combiner: max via cmpf(ogt) + select. The tree fold runs BOTH ops.
    let data = [0.1f32, 0.9, 0.3, 0.2, 0.5, 0.05, 0.7, 0.05];
    let mut ir_ops = Ops::new();
    let cmp = ir_ops.op(Some("%cmp"), OpKind::ArithCmpf, &["%in", "%out"]);
    let cmp = ir_ops.attr(cmp, AttrKey::Predicate, Attr::Str("ogt"));
    let m = ir_ops.op(Some("%m"), OpKind::ArithSelect, &["%cmp", "%in", "%out"]);
    let y = ir_ops.op(None, OpKind::LinalgYield, &["%m"]);
    let region = vec![cmp, m, y];
    let o = ir_ops.op(Some("%r"), OpKind::LinalgReduce, &["%x"]);
    let o = ir_ops.attr(o, AttrKey::Dim, Attr::Int(1));
    let outs_var = ir_ops.ssas_attr(&["%init"]);
    let o = ir_ops.attr(o, AttrKey::OutsVar, outs_var);
    let o = ir_ops.with_region(o, region);
    let r = run_op_execute(
        &mut ir_ops,
        &o,
        &[
            ("%x", tile_with(&data, DType::F16, &[1, 8])),
            ("%init", tile_with(&[f32::NEG_INFINITY], DType::F16, &[1])),
        ],
    );
    let val = match &r {
        Value::Tile(t) => t.as_f32()[0],
        Value::Scalar(s) => s.as_f32().unwrap(),
        other => panic!("got {other:?}"),
    };
    let expected = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    close(val, expected, 1e-2);
}

#[test]
fn reduce_multi_axis() {
    // #106: dimensions=[0,1] reduces BOTH axes of a 2x3 tile -> scalar 15.
    // Mirrors the now-passing Python `test_reduce_multi_axis`.
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::LinalgReduce, &["%x"]);
    let o = ir_ops.attr(o, AttrKey::ReduceFn, Attr::Op(OpKind::ArithAddf));
    let o = ir_ops.attr(o, AttrKey::Dimensions, ir_ops.int_list(vec![0, 1]));
    let outs_var = ir_ops.ssas_attr(&["%init"]);
    let o = ir_ops.attr(o, AttrKey::OutsVar, outs_var);
    let r = run_op_execute(
        &mut ir_ops,
        &o,
        &[
            (
                "%x",
                tile_with(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], DType::F16, &[2, 3]),
            ),
            // 0-D zero accumulator (identity for addf).
            ("%init", tile_with(&[0.0], DType::F16, &[])),
        ],
    );
    close(as_f32(&r), 15.0, 0.1);
}

#[test]
fn reduce_multi_axis_3d_disjoint() {
    // #106: dimensions=[0,2] on a (3,4,2) tile reduces the two DISJOINT axes,
    // leaving (4,). Mirrors Python `test_reduce_multi_axis_3d_disjoint_2d`.
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    // expected[j] = sum over i in 0..3, k in 0..2 of data[i,j,k].
    let mut expected = [0.0f32; 4];
    for i in 0..3 {
        for (j, e) in expected.iter_mut().enumerate() {
            for k in 0..2 {
                *e += data[i * 8 + j * 2 + k];
            }
        }
    }
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::LinalgReduce, &["%x"]);
    let o = ir_ops.attr(o, AttrKey::ReduceFn, Attr::Op(OpKind::ArithAddf));
    let o = ir_ops.attr(o, AttrKey::Dimensions, ir_ops.int_list(vec![0, 2]));
    let outs_var = ir_ops.ssas_attr(&["%init"]);
    let o = ir_ops.attr(o, AttrKey::OutsVar, outs_var);
    let r = run_op_execute(
        &mut ir_ops,
        &o,
        &[
            ("%x", tile_with(&data, DType::F16, &[3, 4, 2])),
            ("%init", tile_with(&[0.0; 4], DType::F16, &[4])),
        ],
    );
    let t = as_tile(&r);
    assert_eq!(t.shape, vec![4]);
    data_close(&t.as_f32(), &expected, 1e-1);
}

#[test]
fn reduce_multi_axis_zero_dims_identity() {
    // #106: dimensions=[] reduces ZERO axes — identity (shape & values unchanged).
    // Mirrors Python `test_reduce_multi_axis_3d_0d`.
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::LinalgReduce, &["%x"]);
    let o = ir_ops.attr(o, AttrKey::ReduceFn, Attr::Op(OpKind::ArithAddf));
    let o = ir_ops.attr(o, AttrKey::Dimensions, ir_ops.int_list(vec![]));
    let outs_var = ir_ops.ssas_attr(&["%init"]);
    let o = ir_ops.attr(o, AttrKey::OutsVar, outs_var);
    let r = run_op_execute(
        &mut ir_ops,
        &o,
        &[
            ("%x", tile_with(&data, DType::F16, &[3, 4, 2])),
            ("%init", tile_with(&[0.0; 24], DType::F16, &[3, 4, 2])),
        ],
    );
    let t = as_tile(&r);
    assert_eq!(t.shape, vec![3, 4, 2]);
    data_close(&t.as_f32(), &data, 1e-1);
}

#[test]
fn reduce_identity_outs_is_a_noop() {
    // The outs accumulator is folded as the INITIAL value (MLIR semantics):
    // combiner(reduced, outs). For the combiner's IDENTITY element (0 for addf) the
    // fold is a no-op — sum([1,2,3,4]) with outs 0 -> 10. The non-identity case (outs
    // 100 -> 110) is covered by `reduce_folds_outs_init` in the linalg.rs unit tests;
    // the fold is now unconditional on every path (fresh-context, harness, resident),
    // matching the Python oracle. The resident executor stays bit-exact because the
    // fusion prefixes each reduce's `outs_var` to its own per-node identity splat
    // (`rename_attrs` in ktir-optimizer), so no stale shared accumulator is folded.
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::LinalgReduce, &["%x"]);
    let o = ir_ops.attr(o, AttrKey::ReduceFn, Attr::Op(OpKind::ArithAddf));
    let outs_var = ir_ops.ssas_attr(&["%init"]);
    let o = ir_ops.attr(o, AttrKey::OutsVar, outs_var);
    let r = run_op_execute(
        &mut ir_ops,
        &o,
        &[
            ("%x", f16_tile(&[1.0, 2.0, 3.0, 4.0])),
            ("%init", tile_with(&[0.0], DType::F16, &[])),
        ],
    );
    close(as_f32(&r), 10.0, 0.1);
}

#[test]
fn fill() {
    let r = run_op(
        OpKind::LinalgFill,
        &["%val", "%out"],
        &[
            ("%val", sf(3.0)),
            ("%out", tile_with(&[0.0; 4], DType::F16, &[4])),
        ],
    );
    let t = as_tile(&r);
    assert_eq!(t.shape, vec![4]);
    assert!(t.as_f32().iter().all(|&x| x == 3.0));
}

#[test]
fn broadcast() {
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::LinalgBroadcast, &["%inp", "%out"]);
    let o = ir_ops.attr(o, AttrKey::Dimensions, ir_ops.int_list(vec![0]));
    let r = run_built(
        &mut ir_ops,
        &o,
        &[
            ("%inp", tile_with(&[1.0, 2.0, 3.0, 4.0], DType::F16, &[4])),
            ("%out", tile_with(&[0.0; 8], DType::F16, &[2, 4])),
        ],
    );
    let t = as_tile(&r);
    assert_eq!(t.shape, vec![2, 4]);
    assert_eq!(&t.as_f32()[0..4], &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(&t.as_f32()[4..8], &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn matmul() {
    // identity @ B == B.
    let r = run_op(
        OpKind::LinalgMatmul,
        &["%a", "%b"],
        &[
            ("%a", tile_with(&[1.0, 0.0, 0.0, 1.0], DType::F16, &[2, 2])),
            ("%b", tile_with(&[1.0, 2.0, 3.0, 4.0], DType::F16, &[2, 2])),
        ],
    );
    data_close(&as_tile(&r).as_f32(), &[1.0, 2.0, 3.0, 4.0], 1e-2);
}

#[test]
fn batch_matmul() {
    // 3 batches of identity @ B == B.
    let eye: Vec<f32> = (0..3).flat_map(|_| vec![1.0, 0.0, 0.0, 1.0]).collect();
    let bdata: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let r = run_op(
        OpKind::LinalgBatchMatmul,
        &["%a", "%b"],
        &[
            ("%a", tile_with(&eye, DType::F16, &[3, 2, 2])),
            ("%b", tile_with(&bdata, DType::F16, &[3, 2, 2])),
        ],
    );
    let t = as_tile(&r);
    assert_eq!(t.shape, vec![3, 2, 2]);
    data_close(&t.as_f32(), &bdata, 1e-2);
}

#[test]
fn generic_reads_outs_arg() {
    // linalg.generic body reads outs bb0 arg: outs (1,2) + ins (10,20) = (11,22).
    let mut ir_ops = Ops::new();
    let add = ir_ops.op(Some("%sum"), OpKind::ArithAddf, &["%in_arg", "%out_arg"]);
    let yld = ir_ops.op(None, OpKind::LinalgYield, &["%sum"]);
    let o = ir_ops.op(Some("%r"), OpKind::LinalgGeneric, &["%ins", "%outs"]);
    let o = ir_ops.attr(o, AttrKey::NIns, Attr::Int(1));
    let bb0 = ir_ops.ssas_attr(&["%in_arg", "%out_arg"]);
    let o = ir_ops.attr(o, AttrKey::Bb0Names, bb0);
    let o = ir_ops.with_region(o, vec![add, yld]);
    let r = run_op_execute(
        &mut ir_ops,
        &o,
        &[
            ("%ins", f16_tile(&[10.0, 20.0])),
            ("%outs", f16_tile(&[1.0, 2.0])),
        ],
    );
    data_close(&as_tile(&r).as_f32(), &[11.0, 22.0], 1e-2);
}

#[test]
fn linalg_index() {
    // linalg.index returns a broadcasting index array for a dimension.
    let dispatch = Dispatch::new();
    let grid = GridExecutor::new((1, 1, 1));
    let env = ExecutionEnv::new(&dispatch, &grid);
    let mut ctx = single_core_context();
    ctx.push_index_shape(vec![4, 3]);
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::LinalgIndex, &[]);
    let o = ir_ops.attr(o, AttrKey::Dim, Attr::Int(0));
    let r = execute_op(&o, &mut ctx, &env).unwrap().unwrap();
    let t = as_tile(&r);
    assert_eq!(t.shape, vec![4, 1]);
    assert_eq!(t.as_f32().to_vec(), vec![0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn linalg_yield() {
    // linalg.yield returns its operand's value directly (no SSA result binding).
    let dispatch = Dispatch::new();
    let grid = GridExecutor::new((1, 1, 1));
    let env = ExecutionEnv::new(&dispatch, &grid);
    let mut ctx = single_core_context();
    let mut ir_ops = Ops::new();
    ctx.set_value(ir_ops.ssa("%v"), idx(42));
    let o = ir_ops.op(None, OpKind::LinalgYield, &["%v"]);
    let produced = dispatch.handler(OpKind::LinalgYield).unwrap()(&o, &mut ctx, &env).unwrap();
    assert_eq!(as_i64(&produced.unwrap()), 42);
}

// ===========================================================================
// tensor (TestTensor)
// ===========================================================================

#[test]
fn tensor_empty() {
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::TensorEmpty, &[]);
    let o = ir_ops.attr(o, AttrKey::Shape, ir_ops.int_list(vec![2, 4]));
    let o = ir_ops.attr(o, AttrKey::Dtype, Attr::Str("f16"));
    let r = run_built(&mut ir_ops, &o, &[]);
    let t = as_tile(&r);
    assert_eq!(t.shape, vec![2, 4]);
}

#[test]
fn tensor_splat() {
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::TensorSplat, &["%val"]);
    let o = ir_ops.attr(o, AttrKey::Shape, ir_ops.int_list(vec![4]));
    let o = ir_ops.attr(o, AttrKey::Dtype, Attr::Str("f16"));
    let r = run_built(&mut ir_ops, &o, &[("%val", sf(7.0))]);
    let t = as_tile(&r);
    assert!(t.as_f32().iter().all(|&x| x == 7.0));
}

#[test]
fn tensor_extract() {
    // 2x2 tile [[1,2],[3,4]] at [1,0] -> 3.
    let r = run_op(
        OpKind::TensorExtract,
        &["%t", "%i", "%j"],
        &[
            ("%t", tile_with(&[1.0, 2.0, 3.0, 4.0], DType::F16, &[2, 2])),
            ("%i", idx(1)),
            ("%j", idx(0)),
        ],
    );
    close(as_f32(&r), 3.0, 1e-2);
}

#[test]
fn tensor_expand_shape() {
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::TensorExpandShape, &["%t"]);
    let o = ir_ops.attr(o, AttrKey::TargetShape, ir_ops.int_list(vec![1, 4]));
    let r = run_built(
        &mut ir_ops,
        &o,
        &[("%t", tile_with(&[1.0, 2.0, 3.0, 4.0], DType::F16, &[4]))],
    );
    assert_eq!(as_tile(&r).shape, vec![1, 4]);
}

#[test]
fn tensor_collapse_shape() {
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::TensorCollapseShape, &["%t"]);
    let o = ir_ops.attr(o, AttrKey::TargetShape, ir_ops.int_list(vec![4]));
    let r = run_built(
        &mut ir_ops,
        &o,
        &[("%t", tile_with(&[1.0, 2.0, 3.0, 4.0], DType::F16, &[2, 2]))],
    );
    let t = as_tile(&r);
    assert_eq!(t.shape, vec![4]);
    assert_eq!(t.as_f32().to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn tensor_reshape() {
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::TensorReshape, &["%t", "%s"]);
    let o = ir_ops.attr(o, AttrKey::TargetShape, ir_ops.int_list(vec![2, 4]));
    let o = ir_ops.attr(o, AttrKey::Dtype, Attr::Str("f16"));
    let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
    let r = run_built(
        &mut ir_ops,
        &o,
        &[
            ("%t", tile_with(&data, DType::F16, &[8])),
            ("%s", tile_with(&[2.0, 4.0], DType::I32, &[2])),
        ],
    );
    let t = as_tile(&r);
    assert_eq!(t.shape, vec![2, 4]);
    assert_eq!(t.as_f32().to_vec(), data);
}

#[test]
fn tensor_reshape_non_square_target() {
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::TensorReshape, &["%t", "%s"]);
    let o = ir_ops.attr(o, AttrKey::TargetShape, ir_ops.int_list(vec![3, 4]));
    let o = ir_ops.attr(o, AttrKey::Dtype, Attr::Str("f16"));
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let r = run_built(
        &mut ir_ops,
        &o,
        &[
            ("%t", tile_with(&data, DType::F16, &[12])),
            ("%s", tile_with(&[3.0, 4.0], DType::I32, &[2])),
        ],
    );
    let t = as_tile(&r);
    assert_eq!(t.shape, vec![3, 4]);
    assert_eq!(t.as_f32().to_vec(), data); // row-major preserved
}

#[test]
fn tensor_reshape_size_mismatch_raises() {
    // 7 elements cannot fill (3,3)=9. Must error, not silently truncate.
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::TensorReshape, &["%t", "%s"]);
    let o = ir_ops.attr(o, AttrKey::TargetShape, ir_ops.int_list(vec![3, 3]));
    let o = ir_ops.attr(o, AttrKey::Dtype, Attr::Str("f16"));
    let data: Vec<f32> = (0..7).map(|x| x as f32).collect();
    let err = run_built_try(
        &mut ir_ops,
        &o,
        &[
            ("%t", tile_with(&data, DType::F16, &[7])),
            ("%s", tile_with(&[3.0, 3.0], DType::I32, &[2])),
        ],
    )
    .unwrap_err();
    assert!(err.contains("cannot reshape"), "unexpected error: {err}");
}

#[test]
fn tensor_reshape_to_3d() {
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::TensorReshape, &["%t", "%s"]);
    let o = ir_ops.attr(o, AttrKey::TargetShape, ir_ops.int_list(vec![2, 3, 4]));
    let o = ir_ops.attr(o, AttrKey::Dtype, Attr::Str("f16"));
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let r = run_built(
        &mut ir_ops,
        &o,
        &[
            ("%t", tile_with(&data, DType::F16, &[24])),
            ("%s", tile_with(&[2.0, 3.0, 4.0], DType::I32, &[3])),
        ],
    );
    let t = as_tile(&r);
    assert_eq!(t.shape, vec![2, 3, 4]);
    assert_eq!(t.as_f32().to_vec(), data);
}

#[test]
fn tensor_from_elements() {
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::TensorFromElements, &["%a", "%b"]);
    let o = ir_ops.attr(o, AttrKey::Shape, ir_ops.int_list(vec![2]));
    let o = ir_ops.attr(o, AttrKey::Dtype, Attr::Str("index"));
    let r = run_built(&mut ir_ops, &o, &[("%a", idx(16)), ("%b", idx(32))]);
    let t = as_tile(&r);
    assert_eq!(t.shape, vec![2]);
    assert_eq!(t.as_f32().to_vec(), vec![16.0, 32.0]);
}

#[test]
fn tensor_from_elements_n1() {
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%r"), OpKind::TensorFromElements, &["%a"]);
    let o = ir_ops.attr(o, AttrKey::Shape, ir_ops.int_list(vec![1]));
    let o = ir_ops.attr(o, AttrKey::Dtype, Attr::Str("index"));
    let r = run_built(&mut ir_ops, &o, &[("%a", idx(128))]);
    let t = as_tile(&r);
    assert_eq!(t.shape, vec![1]);
    assert_eq!(t.as_f32().to_vec(), vec![128.0]);
}

// ===========================================================================
// tensor.generate (TestTensorGenerate)
// ===========================================================================

#[test]
fn generate_1d() {
    // ^bb0(%i): %val = muli %i, %c2 ; yield %val  over shape [4] -> [0,2,4,6].
    let mut ir_ops = Ops::new();
    let names = ir_ops.ssas_attr(&["%i"]);
    let bb0 = ir_ops.op(None, OpKind::RegionBb0Args, &[]);
    let bb0 = ir_ops.attr(bb0, AttrKey::Names, names);
    let mul = ir_ops.op(Some("%val"), OpKind::ArithMuli, &["%i", "%c2"]);
    let yld = ir_ops.op(None, OpKind::TensorYield, &["%val"]);
    let o = ir_ops.op(Some("%r"), OpKind::TensorGenerate, &[]);
    let o = ir_ops.attr(o, AttrKey::Shape, ir_ops.int_list(vec![4]));
    let o = ir_ops.attr(o, AttrKey::Dtype, Attr::Str("f16"));
    let o = ir_ops.with_region(o, vec![bb0, mul, yld]);
    let r = run_op_execute(&mut ir_ops, &o, &[("%c2", idx(2))]);
    let t = as_tile(&r);
    assert_eq!(t.shape, vec![4]);
    assert_eq!(t.as_f32().to_vec(), vec![0.0, 2.0, 4.0, 6.0]);
}

#[test]
fn generate_2d() {
    // ^bb0(%i, %j): %cmp = cmpi sge %i,%j ; yield %cmp  over 3x3.
    let mut ir_ops = Ops::new();
    let names = ir_ops.ssas_attr(&["%i", "%j"]);
    let bb0 = ir_ops.op(None, OpKind::RegionBb0Args, &[]);
    let bb0 = ir_ops.attr(bb0, AttrKey::Names, names);
    let cmp = ir_ops.op(Some("%cmp"), OpKind::ArithCmpi, &["%i", "%j"]);
    let cmp = ir_ops.attr(cmp, AttrKey::Predicate, Attr::Str("sge"));
    let yld = ir_ops.op(None, OpKind::TensorYield, &["%cmp"]);
    let o = ir_ops.op(Some("%r"), OpKind::TensorGenerate, &[]);
    let o = ir_ops.attr(o, AttrKey::Shape, ir_ops.int_list(vec![3, 3]));
    let o = ir_ops.attr(o, AttrKey::Dtype, Attr::Str("f16"));
    let o = ir_ops.with_region(o, vec![bb0, cmp, yld]);
    let r = run_op_execute(&mut ir_ops, &o, &[]);
    let t = as_tile(&r);
    assert_eq!(t.shape, vec![3, 3]);
    // i >= j lower-triangular (incl diagonal): [[1,0,0],[1,1,0],[1,1,1]]
    assert_eq!(
        t.as_f32().to_vec(),
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    );
}

// ===========================================================================
// scf / func (TestScfFunc)
// ===========================================================================

#[test]
fn scf_yield() {
    // scf.yield wraps operands in a Value::Tuple (the _YieldResult analogue).
    let r = run_op(
        OpKind::ScfYield,
        &["%a", "%b"],
        &[("%a", idx(5)), ("%b", idx(6))],
    );
    match r {
        Value::Tuple(vals) => {
            assert_eq!(vals.len(), 2);
            assert_eq!(as_i64(&vals[0]), 5);
            assert_eq!(as_i64(&vals[1]), 6);
        }
        other => panic!("expected Tuple, got {other:?}"),
    }
}

#[test]
fn return_no_value() {
    // func.return with no operands returns None.
    let dispatch = Dispatch::new();
    let grid = GridExecutor::new((1, 1, 1));
    let env = ExecutionEnv::new(&dispatch, &grid);
    let mut ctx = single_core_context();
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(None, OpKind::FuncReturn, &[]);
    let out = dispatch.handler(OpKind::FuncReturn).unwrap()(&o, &mut ctx, &env).unwrap();
    assert!(out.is_none());
}

#[test]
fn if_then_branch() {
    // condition=true runs the then-region; its yielded value surfaces as %r.
    let mut ir_ops = Ops::new();
    let t = ir_ops.op(Some("%t"), OpKind::ArithConstant, &[]);
    let t = ir_ops.attr(t, AttrKey::Value, Attr::Int(1));
    let yt = ir_ops.op(None, OpKind::ScfYield, &["%t"]);
    let then = vec![t, yt];
    let f = ir_ops.op(Some("%f"), OpKind::ArithConstant, &[]);
    let f = ir_ops.attr(f, AttrKey::Value, Attr::Int(2));
    let yf = ir_ops.op(None, OpKind::ScfYield, &["%f"]);
    let els = vec![f, yf];
    let o = ir_ops.op(Some("%r"), OpKind::ScfIf, &["%cond"]);
    let o = ir_ops.regions(o, vec![then, els]);
    let r = run_op_execute(
        &mut ir_ops,
        &o,
        &[("%cond", Value::Scalar(Scalar::Bool(true)))],
    );
    assert_eq!(as_i64(&r), 1);
}

#[test]
fn if_else_branch() {
    let mut ir_ops = Ops::new();
    let t = ir_ops.op(Some("%t"), OpKind::ArithConstant, &[]);
    let t = ir_ops.attr(t, AttrKey::Value, Attr::Int(1));
    let yt = ir_ops.op(None, OpKind::ScfYield, &["%t"]);
    let then = vec![t, yt];
    let f = ir_ops.op(Some("%f"), OpKind::ArithConstant, &[]);
    let f = ir_ops.attr(f, AttrKey::Value, Attr::Int(2));
    let yf = ir_ops.op(None, OpKind::ScfYield, &["%f"]);
    let els = vec![f, yf];
    let o = ir_ops.op(Some("%r"), OpKind::ScfIf, &["%cond"]);
    let o = ir_ops.regions(o, vec![then, els]);
    let r = run_op_execute(
        &mut ir_ops,
        &o,
        &[("%cond", Value::Scalar(Scalar::Bool(false)))],
    );
    assert_eq!(as_i64(&r), 2);
}

#[test]
fn if_then_else_yield_result() {
    // A yielding then-branch returns the unwrapped value (not a tuple wrapper).
    let mut ir_ops = Ops::new();
    let yv = ir_ops.op(None, OpKind::ScfYield, &["%val"]);
    let then = vec![yv];
    let o = ir_ops.op(Some("%res"), OpKind::ScfIf, &["%cond"]);
    let o = ir_ops.regions(o, vec![then, vec![]]);
    let r = run_op_execute(
        &mut ir_ops,
        &o,
        &[
            ("%cond", Value::Scalar(Scalar::Bool(true))),
            ("%val", idx(42)),
        ],
    );
    assert_eq!(as_i64(&r), 42);
}

// ===========================================================================
// ktdp (TestKtdp)
// ===========================================================================

#[test]
fn get_compute_tile_id_single() {
    // Single-dim returns the x grid coordinate as an Index. Core at grid x = 3.
    let dispatch = Dispatch::new();
    let grid = GridExecutor::new((4, 1, 1));
    let env = ExecutionEnv::new(&dispatch, &grid);
    let mem = SpyreMemoryHierarchy::new(4);
    let mut ctx = CoreContext::new(
        3,
        (3, 0, 0),
        Rc::clone(&mem.hbm),
        mem.get_lx(3),
        mem.lx_scratchpads.clone(),
    );
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%id"), OpKind::KtdpGetComputeTileId, &[]);
    let r = dispatch.handler(OpKind::KtdpGetComputeTileId).unwrap()(&o, &mut ctx, &env)
        .unwrap()
        .unwrap();
    assert_eq!(as_i64(&r), 3);
}

#[test]
fn get_compute_tile_id_multi() {
    // Multi-dim returns a tuple of grid coordinates. Core at grid (2,1,0).
    let dispatch = Dispatch::new();
    let grid = GridExecutor::new((4, 2, 1));
    let env = ExecutionEnv::new(&dispatch, &grid);
    let mem = SpyreMemoryHierarchy::new(8);
    let core = grid.grid_to_linear(2, 1, 0);
    let mut ctx = CoreContext::new(
        core,
        (2, 1, 0),
        Rc::clone(&mem.hbm),
        mem.get_lx(core),
        mem.lx_scratchpads.clone(),
    );
    let mut ir_ops = Ops::new();
    let o = ir_ops.op(Some("%x"), OpKind::KtdpGetComputeTileId, &[]);
    let o = ir_ops.attr(o, AttrKey::NumResults, Attr::Int(2));
    let r = dispatch.handler(OpKind::KtdpGetComputeTileId).unwrap()(&o, &mut ctx, &env)
        .unwrap()
        .unwrap();
    match r {
        Value::Tuple(vals) => {
            assert_eq!(vals.len(), 2);
            assert_eq!(as_i64(&vals[0]), 2);
            assert_eq!(as_i64(&vals[1]), 1);
        }
        other => panic!("expected Tuple, got {other:?}"),
    }
}

#[test]
fn construct_memory_view() {
    // Builds a MemRef at the given pointer with the given shape.
    let dispatch = Dispatch::new();
    let grid = GridExecutor::new((1, 1, 1));
    let env = ExecutionEnv::new(&dispatch, &grid);
    let mut ctx = single_core_context();
    let stick = ctx.hbm.borrow_mut().allocate(256 * 2);
    // The pointer SSA value is an ELEMENT index (RFC #110): elem = stick*128/2 (f16)
    // so the view's byte_address lands on stick*STICK_BYTES.
    let elem = stick * STICK_BYTES / DType::F16.bytes_per_elem() as i64;
    let mut ir_ops = Ops::new();
    ctx.set_value(ir_ops.ssa("%ptr"), Value::Index(elem));
    let o = ir_ops.op(Some("%view"), OpKind::KtdpConstructMemoryView, &["%ptr"]);
    let o = ir_ops.attr(o, AttrKey::Shape, ir_ops.int_list(vec![256]));
    let o = ir_ops.attr(o, AttrKey::Strides, ir_ops.int_list(vec![1]));
    let o = ir_ops.attr(o, AttrKey::MemorySpace, Attr::Str("HBM"));
    let o = ir_ops.attr(o, AttrKey::Dtype, Attr::Dtype(DType::F16));
    let r = execute_op(&o, &mut ctx, &env).unwrap().unwrap();
    match r {
        Value::MemRef(m) => {
            assert_eq!(m.shape, vec![256]);
            // byte_address == elem * bytes_per_elem == stick * STICK_BYTES.
            assert_eq!(m.byte_address(), stick * STICK_BYTES);
        }
        other => panic!("expected MemRef, got {other:?}"),
    }
}

#[test]
fn load_store_roundtrip() {
    // load reads from HBM; store writes a modified tile back. Drive the full
    // construct_memory_view + construct_access_tile + load/store chain so we use
    // the real MemRef/AccessTile types rather than hand-building them.
    use ktir_emulator::interpreter::execute_ops;

    let dispatch = Dispatch::new();
    let grid = GridExecutor::new((1, 1, 1));
    let env = ExecutionEnv::new(&dispatch, &grid);
    let mut ctx = single_core_context();

    let n = 8usize;
    let data: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let stick = ctx.hbm.borrow_mut().allocate((n * 2) as i64); // f16 = 2 bytes
    let bytes = ktir_emulator::codec::encode(&data, DType::F16);
    ctx.hbm
        .borrow_mut()
        .write_bytes(stick * STICK_BYTES, &bytes);

    // The pointer SSA value is an ELEMENT index (RFC #110): elem = stick*128/2 (f16)
    // so the view's byte_address lands on the seeded stick*STICK_BYTES.
    let elem = stick * STICK_BYTES / DType::F16.bytes_per_elem() as i64;
    let mut ir_ops = Ops::new();
    ctx.set_value(ir_ops.ssa("%p"), Value::Index(elem));
    ctx.set_value(ir_ops.ssa("%i"), Value::Index(0));

    let view = ir_ops.op(Some("%v"), OpKind::KtdpConstructMemoryView, &["%p"]);
    let view = ir_ops.attr(view, AttrKey::Shape, ir_ops.int_list(vec![n as i64]));
    let view = ir_ops.attr(view, AttrKey::Strides, ir_ops.int_list(vec![1]));
    let view = ir_ops.attr(view, AttrKey::MemorySpace, Attr::Str("HBM"));
    let view = ir_ops.attr(view, AttrKey::Dtype, Attr::Dtype(DType::F16));

    let acc = ir_ops.op(Some("%acc"), OpKind::KtdpConstructAccessTile, &["%v", "%i"]);
    let acc = ir_ops.attr(acc, AttrKey::Shape, ir_ops.int_list(vec![n as i64]));
    let acc = ir_ops.attr(
        acc,
        AttrKey::BaseMap,
        Attr::AffineMap(ir_ops.identity_map(1)),
    );
    let load = ir_ops.op(Some("%t"), OpKind::KtdpLoad, &["%acc"]);

    // Load and verify.
    execute_ops(&[view, acc, load], &mut ctx, &env).unwrap();
    match ctx.get_value(ir_ops.ssa("%t")).unwrap() {
        Value::Tile(t) => assert_eq!(t.as_f32().to_vec(), data),
        other => panic!("expected Tile, got {other:?}"),
    }

    // Store data*2 back through the same access tile and verify HBM.
    let doubled: Vec<f32> = data.iter().map(|&x| x * 2.0).collect();
    ctx.set_value(
        ir_ops.ssa("%tile"),
        Value::Tile(Tile::compute(doubled.clone(), DType::F16, vec![n])),
    );
    let acc2 = ir_ops.op(
        Some("%acc2"),
        OpKind::KtdpConstructAccessTile,
        &["%v", "%i"],
    );
    let acc2 = ir_ops.attr(acc2, AttrKey::Shape, ir_ops.int_list(vec![n as i64]));
    let acc2 = ir_ops.attr(
        acc2,
        AttrKey::BaseMap,
        Attr::AffineMap(ir_ops.identity_map(1)),
    );
    let store = ir_ops.op(None, OpKind::KtdpStore, &["%tile", "%acc2"]);
    execute_ops(&[acc2, store], &mut ctx, &env).unwrap();
    let raw = ctx.hbm.borrow().read_bytes(stick * STICK_BYTES, n * 2);
    let got = ktir_emulator::codec::decode(&raw, n, DType::F16);
    assert_eq!(got, doubled);
}

// SPDX-License-Identifier: Apache-2.0

//! The SLAB GATHER, executed: `test-fixtures/expert_slab.py` through the whole splice
//! chain and run by `ktir_emulator`, against a host model of the routed projection.
//!
//! ```text
//!   triton_frontend      .py -> TTIR       (tl.load scalar id, rank-3 W.load, tl.reshape)
//!   triton_ktir          TTIR -> KTIR      (the data offset -> an indirect access tile)
//!   to_ktir_emit         KTIR -> IRFunction
//!   ktir_emulator        the executed rows
//! ```
//!
//! `out[p, :] = x[p, :] @ W[ids[p]].T` for every routed pair `p`. The ids are DATA, so
//! a lowering that dropped the indirection (every pair reading slab 0, or the id read as an
//! index constant) still yields well-formed numbers -- which is why the comparison carries a
//! CONTROL: the same model under every id bumped `+1 mod E` must differ from what the kernel
//! produced, or the exact comparison could not see an expert-selection defect at all.

use std::collections::HashMap;

use ktir_core::arena::Arena;
use ktir_core::dtypes::DType;
use ktir_core::ir::IRFunction;
use ktir_core::opkind::OpKind;
use ktir_emulator::interpreter::Arg;
use ktir_emulator::latency::HardwareConfig;
use triton_frontend::codegen::{self, ArgSpec, KernelSpec};
use triton_frontend::opt;
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

const SRC: &str = include_str!("../../test-fixtures/expert_slab.py");

/// `expert_slab.constexprs()`: 8 routed pairs over 8 experts, 64x128 slabs.
const P: usize = 8;
const E: usize = 8;
const N: usize = 64;
const K: usize = 128;

fn spec() -> KernelSpec {
    let mut signature = HashMap::new();
    for (k, v) in [
        ("desc_ids", "*i32"),
        ("desc_w", "*fp16"),
        ("desc_x", "*fp16"),
        ("desc_o", "*fp16"),
    ] {
        signature.insert(k.to_string(), ArgSpec::parse(v).expect("pointer spec"));
    }
    for k in ["P", "E", "N", "K"] {
        signature.insert(
            k.to_string(),
            ArgSpec::parse("constexpr").expect("constexpr"),
        );
    }
    let mut constexprs = HashMap::new();
    for (k, v) in [("P", P), ("E", E), ("N", N), ("K", K)] {
        constexprs.insert(k.to_string(), Val::Int(v as i128));
    }
    KernelSpec {
        kernel: "expert_slab_fwd".to_string(),
        signature,
        constexprs,
        file: "expert_slab.py".to_string(),
    }
}

fn lowered() -> IRFunction<'static> {
    let mut tt = codegen::compile(SRC, &spec(), Target::spyre())
        .unwrap_or_else(|e| panic!("codegen refused the kernel: {e}"));
    opt::make_ttir(&mut tt).unwrap_or_else(|e| panic!("make_ttir refused the kernel: {e}"));
    let mut m = triton_ktir::from_ttir::convert(&tt)
        .unwrap_or_else(|e| panic!("from_ttir refused the kernel: {e}"));
    triton_ktir::make_ktir(&mut m, &[1])
        .unwrap_or_else(|e| panic!("make_ktir refused the kernel: {e}"));
    triton_ktir::passes::to_ktir::run(&mut m, &[1])
        .unwrap_or_else(|e| panic!("to_ktir refused the kernel: {e}"));
    triton_ktir::passes::to_ktir_emit::lower(&m, Arena::global())
        .unwrap_or_else(|e| panic!("to_ktir_emit refused the kernel: {e}"))
}

/// The handoff census: one indirect tile per pair survives every stage to THEIR IR, so the
/// numbers below are measuring the gather and not a folded-away direct read.
#[test]
fn every_pair_reaches_the_emulator_as_an_indirect_tile() {
    let f = lowered();
    let n = f
        .ops_deep()
        .into_iter()
        .filter(|op| op.op_type == OpKind::KtdpConstructIndirectAccessTile)
        .count();
    assert_eq!(n, P, "one indirect slab tile per routed pair");
}

/// Exactly representable f16 values whose PATTERN differs per expert (the period and phase
/// both move with `e`), so a row read from the wrong slab disagrees column by column rather
/// than by a constant the row sum could cancel.
fn bank() -> Vec<f64> {
    (0..E * N * K)
        .map(|i| {
            let e = i / (N * K);
            let (n, k) = ((i % (N * K)) / K, i % K);
            (((n * (e + 2) + k * (2 * e + 1)) % 9) as f64 - 4.0) * 0.125
        })
        .collect()
}

fn rows() -> Vec<f64> {
    (0..P * K)
        .map(|i| ((i % 5) as f64 - 2.0) * 0.0625)
        .collect()
}

/// Arbitrary routing, every expert reachable, two pairs sharing an expert.
fn ids() -> Vec<i32> {
    vec![3, 0, 7, 3, 5, 1, 6, 2]
}

/// `out[p, n] = Σ_k x[p, k] · W[ids[p], n, k]`, in f64.
fn routed(w: &[f64], x: &[f64], ids: &[i32]) -> Vec<f64> {
    let mut out = vec![0.0; P * N];
    for p in 0..P {
        let e = ids[p] as usize;
        for n in 0..N {
            let mut s = 0.0;
            for k in 0..K {
                s += x[p * K + k] * w[(e * N + n) * K + k];
            }
            out[p * N + n] = s;
        }
    }
    out
}

#[test]
fn the_executed_kernel_projects_each_pair_through_its_own_expert() {
    let f = lowered();
    let (w, x, ids) = (bank(), rows(), ids());
    let f16 = |v: &[f64]| {
        ktir_core::codec::encode(&v.iter().map(|&a| a as f32).collect::<Vec<_>>(), DType::F16)
    };
    let i32_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();

    let params: Vec<ktir_core::ir::Ssa> = f.arguments.iter().map(|(s, _)| *s).collect();
    assert_eq!(params.len(), 4, "desc_ids, desc_w, desc_x, desc_o");
    // POSITIONAL, in declaration order: the ids, the bank, the rows, the output.
    let bound: [(Vec<u8>, Vec<usize>, DType); 4] = [
        (i32_bytes, vec![P], DType::I32),
        (f16(&w), vec![E, N, K], DType::F16),
        (f16(&x), vec![P, K], DType::F16),
        (f16(&vec![0.0; P * N]), vec![P, N], DType::F16),
    ];
    let args: Vec<(ktir_core::ir::Ssa, Arg)> = params
        .iter()
        .zip(bound)
        .map(|(&ssa, (data, shape, dtype))| (ssa, Arg::TensorBytes { data, shape, dtype }))
        .collect();

    let (raw, _latency) = ktir_emulator::interpreter::execute_function_with_latency(
        &f,
        &args,
        HardwareConfig::default(),
    )
    .unwrap_or_else(|e| panic!("the emulator refused the kernel: {e}"));
    let got: Vec<f64> = raw
        .get(&params[3])
        .expect("desc_o is written by the stores and read back")
        .data
        .iter()
        .map(|&v| v as f64)
        .collect();

    let truth = routed(&w, &x, &ids);
    // The dot accumulates in f16 (the fixture's delta 3), so the floor is f16's, scaled by
    // the K-term sum's magnitude.
    let mut worst = (0.0f64, 0usize);
    for (i, (&g, &t)) in got.iter().zip(&truth).enumerate() {
        let rel = (g - t).abs() / t.abs().max(1.0);
        if rel > worst.0 {
            worst = (rel, i);
        }
    }
    println!(
        "expert_slab  n={}  worst rel err {:.3e} at [{}] (got {}, want {})",
        P * N,
        worst.0,
        worst.1,
        got[worst.1],
        truth[worst.1]
    );
    assert!(
        worst.0 < 1e-2,
        "pair {} col {}: got {:+.6e}, want {:+.6e} -- beyond the f16 floor",
        worst.1 / N,
        worst.1 % N,
        got[worst.1],
        truth[worst.1]
    );

    // THE CONTROL: every id bumped `+1 mod E` must be visibly different from what ran.
    let shifted: Vec<i32> = ids.iter().map(|&e| (e + 1) % E as i32).collect();
    let control = routed(&w, &x, &shifted);
    let control_err = got
        .iter()
        .zip(&control)
        .map(|(&g, &c)| (g - c).abs() / c.abs().max(1.0))
        .fold(0.0f64, f64::max);
    assert!(
        control_err > 1e-1,
        "the shifted-id model agrees with the kernel to {control_err:.3e}, so this \
         comparison cannot see which expert each pair read"
    );
}

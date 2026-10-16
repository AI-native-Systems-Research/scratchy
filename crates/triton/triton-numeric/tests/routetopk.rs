// SPDX-License-Identifier: Apache-2.0

//! THE FUSED TOP-K KERNEL, EXECUTED: the vendor pair's selection — the
//! descending insertion ripple over the RAW LOGITS — driven through the
//! whole ladder to the emulator's numbers.
//!
//! ⚠️ WHY THIS FILE EXISTS AT ALL — the spyre card never runs this body.
//! The door emits the vendor pair's DESCRIPTORS over the Kᵀ-restickified
//! logits (`lower_ktir_to_superdsc::route_topk`), and the door test
//! censuses those ops; the kernel itself is COMPILE-gated and VIEW-gated
//! there. This file is the only EXECUTION the body's numerics get, and it
//! drives the exact text that ships:
//!
//! ```text
//!   codegen::compile   .py -> raw ttir     (include_str!'d from the spyre
//!                                           target — the same file the
//!                                           splice's row splices, so a
//!                                           test-local copy cannot drift)
//!   opt::make_ttir     inline/fold/cse/dce
//!   from_ttir          the OP_SET gate
//!   make_ktir + to_ktir + to_ktir_emit     -> ktir_core::IRFunction
//!   ktir_emulator      the executed indices
//! ```
//!
//! # WHAT THE TWO TESTS OW
//!
//! 1. **The census**: every `arith.select` carries `Dtype = F16` — the
//!    same load-bearing fact as the argsort's (`LegalizeTypes` collapses
//!    every f32 island; `tl.where` types its pair WITHOUT promotion so the
//!    TENSOR must be the left operand). The numbers cannot see the dtype —
//!    every value is f16-exact and every id an integer below `E` — so the
//!    census is the only witness that the ripple's selects are the
//!    collapsible ones.
//! 2. **The numbers, against the DDL's own law, EXACTLY**: the ripple —
//!    K running `(value, index)` registers init `(−inf, −1)`, per candidate
//!    `h` per slot `j` a STRICT `run > cand` keep-else-take with the
//!    displaced pair rippling down — mirrored in Rust from `topk.ddl`'s
//!    k_loop prose. `max|err|` must be ZERO on both landings (the values
//!    AND the ids). Two controls prove the comparison can see the law at
//!    all: the ASCENDING bottom-k must differ (the orientation control),
//!    and on the duplicates row a ties-by-ascending-index model must differ
//!    (the NEWCOMER-TAKES-THE-SLOT control — among equals the later scan
//!    position ranks higher).
//!
//! ⚠️ WHY `E = 5` AND NOT A POWER OF TWO: the chain fixture pins `E = 5`,
//! and the column-window ripple is the only spelling that states the
//! function there — the front end refuses `tl.arange` at any
//! non-power-of-two length, the measured law that forced this kernel's
//! lane-free form in the first place.

use std::collections::HashMap;

use ktir_core::arena::Arena;
use ktir_core::attrkey::AttrKey;
use ktir_core::dtypes::DType;
use ktir_core::ir::{Attr, IRFunction};
use ktir_core::opkind::OpKind;
use ktir_emulator::interpreter::Arg;
use ktir_emulator::latency::HardwareConfig;
use triton_frontend::codegen::{self, ArgSpec, KernelSpec};
use triton_frontend::opt;
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

/// The kernel that ships — the splice's own row splices this same file, so the
/// executed proof and the spliced program are one text, not two that agree.
const SRC: &str = include_str!("../../../targets/spyre/kernels/routetopk.py");

/// The chain gate's own geometry: `M = 3` rows, `E = 5` experts, `k = 2`.
const M: usize = 3;
const E: usize = 5;
const K: usize = 2;

fn spec() -> KernelSpec {
    let mut signature = HashMap::new();
    for k in ["desc_l", "desc_v", "desc_o"] {
        signature.insert(
            k.to_string(),
            ArgSpec::parse("*fp16").expect("*fp16 parses"),
        );
    }
    for k in ["M", "E", "K"] {
        signature.insert(
            k.to_string(),
            ArgSpec::parse("constexpr").expect("constexpr parses"),
        );
    }
    let mut constexprs = HashMap::new();
    for (k, v) in [("M", M as i128), ("E", E as i128), ("K", K as i128)] {
        constexprs.insert(k.to_string(), Val::Int(v));
    }
    KernelSpec {
        kernel: "routetopk_fwd".to_string(),
        signature,
        constexprs,
        file: "routetopk.py".to_string(),
    }
}

/// The whole chain to THEIR `IRFunction`, in the process-wide arena. Panics rather
/// than returning an error: a kernel that cannot reach the handoff is measuring
/// nothing, and the stage name in the panic is the finding.
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

/// The census test: every select is f16-typed (see the module doc — the numbers
/// cannot see the dtype, so the census is the only witness that the ripple's
/// select form is the one the island collapse accepts).
#[test]
fn every_select_is_dtype_clean_f16() {
    let f = lowered();
    let selects: Vec<_> = f
        .ops_deep()
        .into_iter()
        .filter(|op| op.op_type == OpKind::ArithSelect)
        .collect();
    assert!(
        !selects.is_empty(),
        "the kernel's ripple selects are selects; an empty census means a pass \
         folded or DCE'd them, and the numbers test below is then measuring nothing"
    );
    for op in &selects {
        assert!(
            matches!(op.attr(AttrKey::Dtype), Some(Attr::Dtype(DType::F16))),
            "every select must carry Dtype = F16; got {:?}. `tl.where` types its pair \
             WITHOUT promotion — a float literal splats at its own f32 dtype while the \
             result takes the LEFT operand's type — so a select whose tensor operand is \
             not on the left goes f32-typed, and `LegalizeTypes` RED-stops on its \
             literal. A non-f16 select here is that form, not a benign dtype choice",
            op.attr(AttrKey::Dtype)
        );
    }
    println!("routetopk  {} select(s), all Dtype = F16", selects.len());
}

/// The chain gate's own fixture (the argsort's): distinct values, DUPLICATES,
/// and a NaN — the rows whose ripple exercise every branch of the DDL's
/// k_loop. All values are f16-exact.
fn scores() -> Vec<f64> {
    vec![
        3.0,
        1.0,
        4.0,
        1.5,
        2.0, // distinct:   top-2 = 4@2, 3@0
        5.0,
        2.0,
        5.0,
        2.0,
        7.0, // duplicates: top-2 = 7@4, 5@2 (the NEWCOMER 5@2
        //                          displaces 5@0 — strict >, ties by later index)
        1.0,
        f64::NAN,
        3.0,
        2.0,
        4.0, // NaN row: the NaN enters slot 0, sinks one
             //                          slot per later candidate, flushes off — 4@4, 3@2
    ]
}

/// The ripple, verbatim from `topk.ddl`'s k_loop (an independent Rust
/// restatement of the law the kernel's header states): K running
/// `(value, index)` registers init `(−inf, −1)`; per candidate `h` in scan
/// order, per slot `j`: strict `run > cand` keeps the register, else the
/// NEWCOMER takes the slot and the old pair becomes the candidate for the
/// next slot down. Returns `(values, ids)`, both `[M·K]` row-major.
fn ripple(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut vals = Vec::with_capacity(M * K);
    let mut ids = Vec::with_capacity(M * K);
    for r in 0..M {
        let mut run: Vec<(f64, i64)> = vec![(f64::NEG_INFINITY, -1); K];
        for h in 0..E {
            let mut cand = x[r * E + h];
            let mut cand_h = h as i64;
            for slot in run.iter_mut() {
                let old = *slot;
                if old.0 > cand {
                    continue; // strict `run > cand` keeps the register
                }
                // else the NEWCOMER takes the slot and the old pair ripples down
                *slot = (cand, cand_h);
                cand = old.0;
                cand_h = old.1;
            }
        }
        for &(v, i) in &run {
            vals.push(v);
            ids.push(i as f64);
        }
    }
    (vals, ids)
}

/// The orientation control's model: the k LOWEST-scoring experts, ascending
/// — a kernel whose ripple ran the other way would match THIS and not the
/// golden.
fn bottom_k(x: &[f64]) -> Vec<f64> {
    let san: Vec<f64> = x
        .iter()
        .map(|&v| if v.is_nan() { f64::INFINITY } else { v })
        .collect();
    (0..M)
        .flat_map(|r| {
            let base = r * E;
            let mut idx: Vec<usize> = (0..E).collect();
            idx.sort_by(|&a, &b| san[base + a].total_cmp(&san[base + b]));
            idx[..K].iter().map(|&i| i as f64).collect::<Vec<_>>()
        })
        .collect()
}

/// The tie-law control's model: descending, but ties broken by ASCENDING
/// scan position — the opposite of the DDL's newcomer-takes-the-slot. On
/// the duplicates row the two models disagree, which is what proves the
/// executed comparison can see the tie law at all.
fn descending_ties_ascending_index(x: &[f64]) -> Vec<f64> {
    let san: Vec<f64> = x
        .iter()
        .map(|&v| if v.is_nan() { f64::INFINITY } else { v })
        .collect();
    (0..M)
        .flat_map(|r| {
            let base = r * E;
            let mut idx: Vec<usize> = (0..E).collect();
            idx.sort_by(|&a, &b| san[base + b].total_cmp(&san[base + a]).then(a.cmp(&b)));
            idx[..K].iter().map(|&i| i as f64).collect::<Vec<_>>()
        })
        .collect()
}

/// The numbers, exactly: `max|err|` must be ZERO on BOTH landings (the
/// values and the ids — the ids are integers below E, exact in f16; the
/// values are the fixture's own f16-exact logits), and both controls must
/// differ from the golden.
#[test]
fn the_executed_kernel_runs_the_descending_ripple_exactly() {
    let f = lowered();
    let x = scores();
    let encode = |v: &[f64]| {
        ktir_core::codec::encode(&v.iter().map(|&a| a as f32).collect::<Vec<_>>(), DType::F16)
    };

    let params: Vec<ktir_core::ir::Ssa> = f.arguments.iter().map(|(s, _)| *s).collect();
    assert_eq!(
        params.len(),
        3,
        "desc_l, desc_v, desc_o — nothing else becomes an argument"
    );
    // THE BINDING IS POSITIONAL (the crate's own law): argument i is pointer
    // parameter i of the kernel, in declaration order — the logits, the
    // values landing, the ids landing. The landings are pre-zeroed; the
    // kernel's own init stores overwrite every slot.
    let bound: [Vec<u8>; 3] = [encode(&x), encode(&[0.0; M * K]), encode(&[0.0; M * K])];
    let shapes = [vec![M, E], vec![M, K], vec![M, K]];
    let args: Vec<(ktir_core::ir::Ssa, Arg)> = params
        .iter()
        .zip(bound)
        .zip(shapes)
        .map(|((&ssa, bytes), shape)| {
            (
                ssa,
                Arg::TensorBytes {
                    data: bytes,
                    shape,
                    dtype: DType::F16,
                },
            )
        })
        .collect();

    let (raw, _latency) = ktir_emulator::interpreter::execute_function_with_latency(
        &f,
        &args,
        HardwareConfig::default(),
    )
    .unwrap_or_else(|e| panic!("the emulator refused the kernel: {e}"));
    let vals = raw
        .get(&params[1])
        .expect("desc_v is written by the init stores and read back");
    let ids = raw
        .get(&params[2])
        .expect("desc_o is written by the init stores and read back");
    let got_vals: Vec<f64> = vals.data.iter().map(|&v| v as f64).collect();
    let got_ids: Vec<f64> = ids.data.iter().map(|&v| v as f64).collect();

    let (truth_vals, truth_ids) = ripple(&x);
    for (what, got, truth) in [
        ("values", &got_vals, &truth_vals),
        ("ids", &got_ids, &truth_ids),
    ] {
        let mut max_err = 0.0f64;
        let mut worst = 0usize;
        for (i, (&g, &w)) in got.iter().zip(truth).enumerate() {
            let e = (g - w).abs();
            if e > max_err {
                max_err = e;
                worst = i;
            }
        }
        println!(
            "routetopk {what}  n={}  max|err| = {max_err:.6e} at [{worst}] (got {}, want {})",
            M * K,
            got[worst],
            truth[worst]
        );
        assert_eq!(
            max_err,
            0.0,
            "the ripple lands the fixture's own f16-exact logits and integer scan \
             positions — any nonzero error is a wiring defect, not rounding. First \
             excess in {what} at [{worst}] (row {}, slot {}): got {:+.6e}, want {:+.6e}",
            worst / K,
            worst % K,
            got[worst],
            truth[worst]
        );
    }

    // CONTROL 1 — THE ORIENTATION: the k LOWEST-scoring experts must differ
    // from the golden, or the exact comparison above is vacuous about which
    // way the ripple runs.
    let bottom = bottom_k(&x);
    let flips: usize = truth_ids
        .iter()
        .zip(&bottom)
        .filter(|(t, w)| t != w)
        .count();
    assert!(
        flips > 0,
        "the bottom-k control differs NOWHERE: the fixture's top-k and bottom-k \
         coincide, so the exact comparison above cannot see a ripple-orientation defect"
    );
    // CONTROL 2 — THE TIE LAW: on the duplicates row, ties by ASCENDING
    // index must differ from the golden (the DDL's newcomer-takes-the-slot
    // puts the LATER equal candidate higher), or the comparison is vacuous
    // about the tie law.
    let tie_flip = descending_ties_ascending_index(&x);
    let tie_flips: usize = truth_ids
        .iter()
        .zip(&tie_flip)
        .filter(|(t, w)| t != w)
        .count();
    assert!(
        tie_flips > 0,
        "the ascending-tie control differs NOWHERE: the fixture's duplicates row \
         does not separate the two tie laws, so the exact comparison above cannot \
         see a tie-law defect"
    );
    println!(
        "controls  the bottom-k model differs at {flips}/{} ids; the ascending-tie \
         model at {tie_flips}/{} (the comparison can see both defects)",
        M * K,
        M * K
    );
}

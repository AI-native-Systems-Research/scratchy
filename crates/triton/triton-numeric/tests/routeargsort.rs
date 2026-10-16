// SPDX-License-Identifier: Apache-2.0

//! THE ROUTER ARGSORT KERNEL, EXECUTED: the stable ascending rank vector's
//! own body, driven through the whole ladder to the emulator's numbers.
//!
//! ⚠️ WHY THIS FILE EXISTS AT ALL — the spyre path never runs this body.
//! `moe_router_chain`'s argsort gate runs the DOOR's SuperDSC ops (the door
//! re-derives its `9W + 1` compare/reduce chain from the minted views), and
//! the door test censuses those ops; the kernel itself is COMPILE-gated and
//! VIEW-gated there. This file is the only EXECUTION the body's numerics
//! get, and it drives the exact text that ships:
//!
//! ```text
//!   codegen::compile   .py -> raw ttir     (include_str!'d from the spyre
//!                                           target — the same file the
//!                                           splice's row splices, so a
//!                                           test-local copy cannot drift)
//!   opt::make_ttir     inline/fold/cse/dce
//!   from_ttir          the OP_SET gate
//!   make_ktir + to_ktir + to_ktir_emit     -> ktir_core::IRFunction
//!   ktir_emulator      the executed ranks
//! ```
//!
//! # WHAT THE TWO TESTS OW
//!
//! 1. **The census**: every `arith.select` carries `Dtype = F16`. Two measured
//!    laws make that the load-bearing fact. `LegalizeTypes` (SP-E2-03) removes
//!    every `arith.extf` and collapses the whole f32 stability-widening island
//!    back to f16 — there are NO f32 compute islands on this ladder, and any
//!    f32 the collapse cannot reach is a named RED-stop. And `tl.where` types
//!    its pair WITHOUT promotion: a float literal splats at its OWN f32 dtype
//!    while the result takes the LEFT operand's type — so the TENSOR must be
//!    the left operand (`where(x == x, x, inf)`) or the select goes f32-typed
//!    and the collapse RED-stops on its literal splat. The numbers cannot see
//!    any of this — every rank is an integer below `E`, exact in f16 and f32
//!    alike — so the census is the only witness that the select form is the
//!    collapsible one.
//! 2. **The numbers, against a host model, EXACTLY**: the rank is a count of
//!    lanes, so the f16 count arithmetic and the store are both exact —
//!    `max|err|` must be ZERO, not within a derived bound. The fixture is the
//!    chain gate's own (`E = 5`, `M = 3`: distinct values, DUPLICATES, a NaN —
//!    every branch of the rank form, at the non-power-of-two `E` that refutes
//!    a `tl.arange`-keyed spelling). A TIE-DROPPED control (the host model
//!    without the stable tie-break) must DIFFER on the duplicates row, which
//!    is what proves the executed comparison can see a missing tie-break at
//!    all.
//!
//! ⚠️ WHY `E = 5` AND NOT A POWER OF TWO: the chain fixture pins `E = 5`, and
//! the count form is the only spelling that states the function there — the
//! front end refuses `tl.arange` at any non-power-of-two length, which is the
//! measured law that forced this kernel's lane-free form in the first place.
//! Running the executed proof at `E = 5` pins that the form, not a
//! power-of-two accident, is what computes.

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
const SRC: &str = include_str!("../../../targets/spyre/kernels/routeargsort.py");

/// The chain gate's own geometry: `M = 3` rows, `E = 5` experts.
const M: usize = 3;
const E: usize = 5;

fn spec() -> KernelSpec {
    let mut signature = HashMap::new();
    for k in ["desc_x", "desc_o"] {
        signature.insert(
            k.to_string(),
            ArgSpec::parse("*fp16").expect("*fp16 parses"),
        );
    }
    for k in ["M", "E"] {
        signature.insert(
            k.to_string(),
            ArgSpec::parse("constexpr").expect("constexpr parses"),
        );
    }
    let mut constexprs = HashMap::new();
    for (k, v) in [("M", M as i128), ("E", E as i128)] {
        constexprs.insert(k.to_string(), Val::Int(v));
    }
    KernelSpec {
        kernel: "routeargsort_fwd".to_string(),
        signature,
        constexprs,
        file: "routeargsort.py".to_string(),
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
/// cannot see the dtype, so the census is the only witness that the select form
/// is the one the island collapse accepts).
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
        "the kernel's sanitize and count forms are selects; an empty census means a pass \
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
    println!("routeargsort  {} select(s), all Dtype = F16", selects.len());
}

/// The chain gate's own fixture: distinct values, DUPLICATES (the tie-break),
/// and a NaN — every branch of the rank form. All values are f16-exact.
fn scores() -> Vec<f64> {
    vec![
        3.0,
        1.0,
        4.0,
        1.5,
        2.0, // distinct:   ranks 2,0,4,1,3
        5.0,
        2.0,
        5.0,
        2.0,
        7.0, // duplicates: ranks 2,0,3,1,4
        1.0,
        f64::NAN,
        3.0,
        2.0,
        4.0, // NaN row:    ranks 0,4,2,1,3
    ]
}

/// The host model of the kernel — the stable ascending rank with NaN-as-greater,
/// exactly the chain gate's golden. `drop_tie` removes the stable tie-break
/// (the `|{h < j : x[h] == x[j]}|` term), the control that proves the executed
/// numbers can see a missing tie-break.
fn model(x: &[f64], drop_tie: bool) -> Vec<f64> {
    let san: Vec<f64> = x
        .iter()
        .map(|&v| if v.is_nan() { f64::INFINITY } else { v })
        .collect();
    (0..x.len())
        .map(|j| {
            let row = j / E;
            let jj = j % E;
            let base = row * E;
            let mut r = 0.0f64;
            for h in 0..E {
                let idx = base + h;
                let prior = h < jj && san[idx] == san[base + jj];
                if san[idx] < san[base + jj] || (prior && !drop_tie) {
                    r += 1.0;
                }
            }
            r
        })
        .collect()
}

/// The numbers, exactly: `max|err|` must be ZERO, and the tie-dropped model
/// must differ on the duplicates row — the tie-break's own control.
#[test]
fn the_executed_kernel_computes_the_stable_ascending_rank_exactly() {
    let f = lowered();
    let n = M * E;
    let x = scores();
    let encode = |v: &[f64]| {
        ktir_core::codec::encode(&v.iter().map(|&a| a as f32).collect::<Vec<_>>(), DType::F16)
    };

    let params: Vec<ktir_core::ir::Ssa> = f.arguments.iter().map(|(s, _)| *s).collect();
    assert_eq!(
        params.len(),
        2,
        "desc_x, desc_o — nothing else becomes an argument"
    );
    // THE BINDING IS POSITIONAL (the crate's own law): argument i is pointer
    // parameter i of the kernel, in declaration order — the scores, then the
    // rank vector.
    let bound: [Vec<u8>; 2] = [encode(&x), encode(&vec![0.0; n])];
    let args: Vec<(ktir_core::ir::Ssa, Arg)> = params
        .iter()
        .zip(bound)
        .map(|(&ssa, bytes)| {
            (
                ssa,
                Arg::TensorBytes {
                    data: bytes,
                    shape: vec![M, E],
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
    let out = raw
        .get(&params[1])
        .expect("desc_o is written by the stores and read back");
    let got: Vec<f64> = out.data.iter().map(|&v| v as f64).collect();

    let truth = model(&x, false);
    let tie_dropped = model(&x, true);
    let mut max_err = 0.0f64;
    let mut worst = 0usize;
    for (i, (&g, &r)) in got.iter().zip(&truth).enumerate() {
        let e = (g - r).abs();
        if e > max_err {
            max_err = e;
            worst = i;
        }
    }
    println!(
        "routeargsort  n={n}  max|err| = {max_err:.6e} at [{worst}] (got {}, want {})",
        got[worst], truth[worst]
    );
    assert_eq!(
        max_err,
        0.0,
        "the rank is a count of lanes — an integer below E — so the f32 island and the \
         one f16 store rounding are EXACT; any nonzero error is a wiring defect, not \
         rounding. First excess at [{worst}] (row {}, col {}): got {:+.6e}, want {:+.6e} \
         (score {:+.6e})",
        worst / E,
        worst % E,
        got[worst],
        truth[worst],
        x[worst]
    );
    // THE CONTROL: the tie-dropped model must differ from the truth on the
    // duplicates row, or the exact comparison above is vacuous about the
    // stable tie-break — a kernel that dropped the `|{h < j : equal}|` term
    // would read as green.
    let flips: usize = truth
        .iter()
        .zip(&tie_dropped)
        .filter(|(t, w)| t != w)
        .count();
    assert!(
        flips > 0,
        "the tie-dropped control differs NOWHERE: the duplicates row's ranks do not \
         depend on the tie-break, so the exact comparison above cannot see a missing \
         tie term"
    );
    println!(
        "control  the tie-dropped model differs at {flips}/{n} elements (the comparison \
         can see a missing tie-break)"
    );
}

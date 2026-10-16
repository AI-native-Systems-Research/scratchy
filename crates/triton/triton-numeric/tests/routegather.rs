// SPDX-License-Identifier: Apache-2.0

//! THE ROUTER GATHER KERNEL, EXECUTED: the true `take_along_axis` body, driven
//! through the whole ladder to the emulator's numbers.
//!
//! ⚠️ WHY THIS FILE EXISTS AT ALL — the spyre path never runs this body.
//! `moe_router_chain`'s gather gate runs the DOOR's SuperDSC ops (the door
//! re-derives its one-hot select chain from the minted views), and the door
//! test censuses those ops; the kernel itself is COMPILE-gated and VIEW-gated
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
//!   ktir_emulator      the executed rows
//! ```
//!
//! # WHAT THE TWO TESTS OW
//!
//! 1. **The census**: every `arith.select` carries `Dtype = F16` — the same
//!    load-bearing fact as the argsort's (`LegalizeTypes` collapses every f32
//!    island; `tl.where` types its pair WITHOUT promotion so the TENSOR must
//!    be the left operand). The numbers cannot see the dtype — the fixture's
//!    values are exact in f16 and f32 alike — so the census is the only
//!    witness that the select form is the collapsible one.
//! 2. **The numbers, against a host model, EXACTLY**: the gather sums one
//!    matching source tile per row-slot (the index selects exactly one
//!    column), and the store is that exact value — `max|err|` must be ZERO.
//!    The fixture is the chain gate's OWN (`src = (0..15)·0.25 − 1` with the
//!    HAND-SET indices `[4,0,2,3,1,4]`), and the hand-set indices are the
//!    point: they are NOT top-k-derived, so a kernel that re-derived the
//!    top-k instead of reading its stated indices CANNOT satisfy this gate —
//!    the true-gather contract is pinned here, not assumed. A SHIFTED-index
//!    control (every index bumped `+1 mod E`) must DIFFER, which is what
//!    proves the executed comparison can see an index-selection defect at
//!    all.
//!
//! ⚠️ WHY `E = 5` AND NOT A POWER OF TWO: the chain fixture pins `E = 5`, and
//! the column-scan gather is the only spelling that states the function there
//! — the front end refuses `tl.arange` at any non-power-of-two length, the
//! measured law that forced this kernel's lane-free form in the first place.

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
const SRC: &str = include_str!("../../../targets/spyre/kernels/routegather.py");

/// The chain gate's own geometry: `M = 3` rows, `E = 5` source columns, `k = 2`
/// gathered slots.
const M: usize = 3;
const E: usize = 5;
const K: usize = 2;

fn spec() -> KernelSpec {
    let mut signature = HashMap::new();
    for k in ["desc_x", "desc_i", "desc_o"] {
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
        kernel: "routegather_fwd".to_string(),
        signature,
        constexprs,
        file: "routegather.py".to_string(),
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
        "the kernel's gather selects are selects; an empty census means a pass \
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
    println!("routegather  {} select(s), all Dtype = F16", selects.len());
}

/// The chain gate's own fixture: `src[i] = i·0.25 − 1` (`[3, 5]`) with the
/// HAND-SET indices `[4,0,2,3,1,4]` — arbitrary, not top-k-derived. All values
/// are exact in f16.
fn src() -> Vec<f64> {
    (0..M * E).map(|i| i as f64 * 0.25 - 1.0).collect()
}

/// The hand-set indices, `[3, 2]`, as the kernel reads them (f16 numerals).
fn idx() -> Vec<f64> {
    vec![4.0, 0.0, 2.0, 3.0, 1.0, 4.0]
}

/// The `take_along_axis` model: `out[n,j] = src[n, idx[n,j]]` — under the
/// given indices, or the shifted control's.
fn take_along(x: &[f64], i: &[f64]) -> Vec<f64> {
    (0..M * K)
        .map(|flat| {
            let n = flat / K;
            x[n * E + i[flat] as usize]
        })
        .collect()
}

/// The numbers, exactly: `max|err|` must be ZERO, and the shifted-index control
/// (every index bumped `+1 mod E`) must differ — the index-selection's own
/// control.
#[test]
fn the_executed_kernel_takes_along_the_stated_indices_exactly() {
    let f = lowered();
    let x = src();
    let i = idx();
    let encode = |v: &[f64]| {
        ktir_core::codec::encode(&v.iter().map(|&a| a as f32).collect::<Vec<_>>(), DType::F16)
    };

    let params: Vec<ktir_core::ir::Ssa> = f.arguments.iter().map(|(s, _)| *s).collect();
    assert_eq!(
        params.len(),
        3,
        "desc_x, desc_i, desc_o — nothing else becomes an argument"
    );
    // THE BINDING IS POSITIONAL (the crate's own law): argument i is pointer
    // parameter i of the kernel, in declaration order — the source, the
    // indices, then the output.
    let bound: [Vec<u8>; 3] = [encode(&x), encode(&i), encode(&[0.0; M * K])];
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
    let out = raw
        .get(&params[2])
        .expect("desc_o is written by the stores and read back");
    let got: Vec<f64> = out.data.iter().map(|&v| v as f64).collect();

    let truth = take_along(&x, &i);
    let shifted: Vec<f64> = i.iter().map(|&v| ((v as usize + 1) % E) as f64).collect();
    let control = take_along(&x, &shifted);
    let mut max_err = 0.0f64;
    let mut worst = 0usize;
    for (n, (&g, &w)) in got.iter().zip(&truth).enumerate() {
        let e = (g - w).abs();
        if e > max_err {
            max_err = e;
            worst = n;
        }
    }
    println!(
        "routegather  n={}  max|err| = {max_err:.6e} at [{worst}] (got {}, want {})",
        M * K,
        got[worst],
        truth[worst]
    );
    assert_eq!(
        max_err,
        0.0,
        "the gather sums ONE matching source tile per row-slot and the store is that \
         exact value — any nonzero error is a wiring defect, not rounding. First excess \
         at [{worst}] (row {}, slot {}): got {:+.6e}, want {:+.6e}",
        worst / K,
        worst % K,
        got[worst],
        truth[worst]
    );
    // THE CONTROL: the shifted-index model (every index bumped `+1 mod E`)
    // must differ from the truth, or the exact comparison above is vacuous
    // about which column was taken — a fixture whose per-row source values
    // agreed along the index choices would read any kernel green.
    let flips: usize = truth.iter().zip(&control).filter(|(t, w)| t != w).count();
    assert!(
        flips > 0,
        "the shifted-index control differs NOWHERE: the fixture's rows agree along \
         every index choice, so the exact comparison above cannot see an \
         index-selection defect"
    );
    println!(
        "control  the shifted-index model differs at {flips}/{} elements (the comparison \
         can see an index-selection defect)",
        M * K
    );
}

// SPDX-License-Identifier: Apache-2.0

//! THE ROUTER-INDEX LADDER EXTENSION, EXECUTED: compare / `tl.where` / fp<->int casts.
//!
//! `from_ttir`'s `OP_SET` deliberately carried no compare, no select and no fp<->int
//! cast until the MoE router rows needed them: no post-`make_ttir` golden contains a
//! LIVE compare (the sanitize chains mint `arith.cmpi` and DCE kills them; the one
//! fixture with a live compare, `attention_sdpa.py`, is not one of the twelve), and
//! the device-side story was the attention fixture's delta 10 -- "there is no
//! lane-wise integer compare on this device", so the causal mask is precomputed on
//! the host. The router rows cannot precompute their way out: `rank[i,j]` IS a
//! lane-wise compare, and the count-form argsort is card-proven, so the ladder has
//! to carry the five spellings the front end emits for them.
//!
//! This file drives the WHOLE value chain over one kernel that spells all five:
//!
//! ```text
//!   codegen::compile   .py -> raw ttir     (`tl.where` -> `arith.select`, compares
//!                                           -> `arith.cmpf`/`cmpi` with `predicate`)
//!   opt::make_ttir     inline/fold/cse/dce (the sanitize chains die here)
//!   from_ttir          the OP_SET/ATTR_KEYS gate -- the extension under test
//!   make_ktir + to_ktir + to_ktir_emit     -> ktir_core::IRFunction
//!   ktir_emulator      the executed numbers
//! ```
//!
//! # WHAT THE TWO TESTS OW
//!
//! 1. **The census**: the `IRFunction` contains exactly the ops the spellings map to
//!    -- one `ArithCmpf` carrying `predicate = "ogt"`, one `ArithCmpi` carrying
//!    `predicate = "slt"`, two `ArithSelect`, two `ArithSitofp`, one `ArithFptosi`.
//!    This is the WIRING claim: a spelling that fell to a wrong `KOpKind`, or a
//!    predicate attr that vanished between `Attr::Pred` and `KAttrKey::Predicate`,
//!    shows up here as a count or a name.
//! 2. **The numbers, against a host model, EXACTLY**: every value in the kernel is a
//!    small integer, so every step (the f32 compare, the f16 add, both casts) is
//!    exact and the comparison needs no envelope -- `max|err|` must be ZERO, not
//!    within a derived bound. A FLIPPED-PREDICATE control (the host model with
//!    `x > 0` inverted) must DIFFER, which is what proves the comparison can see a
//!    wrong predicate at all.
//!
//! ⚠️ WHY THE INT TILE COMES FROM AN f16 LOAD AND NOT AN `*i32` DESCRIPTOR: no
//! executed fixture plain-LOADS an i32 tile (embedding's `*i32` ids feed
//! `tt.descriptor_gather`, not a load), while f16 loads are exercised by every
//! configuration here. The kernel derives its i32 tile with `.to(tl.int32)` -- an
//! `arith.fptosi` the census counts anyway -- so the test adds no unexercised load
//! path while proving the casts.
//!
//! ⚠️ WHY NO `tl.arange` HERE: `tt.make_range` was not in `OP_SET` when this file
//! was written (in every golden it is dead code -- the attention mask is
//! host-precomputed), so the kernel derives its int tile from a load instead.
//! The MoE router rows could not precompute their way out, so the set was
//! extended after all -- `make_range.rs` is that extension's executed proof, and
//! this file's kernel stays load-derived to keep the five spellings it exists to
//! census in one place.

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

/// The kernel: three f16 tiles in, the five spellings, one f16 tile out.
///
/// ⛔ THE CAST SHAPES ARE THE ROUTER'S REAL ONES, and that is a measured law, not a
/// taste: a `.to(tl.float32)` whose f32 result feeds another cast is GENUINE f32 by
/// `LegalizeTypes`' island rule (the island is `extf ... truncf`; an f32 `sitofp`
/// result consumed by an `fptosi` is neither) and the pass RED-STOPS on it by name.
/// The router's index arithmetic goes DIRECTLY i32 -> f16 for the fma lanes and
/// f16 -> i32 for the index domain -- no f32 intermediate -- so that is what this
/// kernel spells, and an f32 round-trip here would be testing a form the ladder
/// refuses on purpose.
///
/// ⛔ AND THE INTEGER LANES ARE TENSOR-VS-TENSOR, ALSO A MEASURED LAW: broadcasting
/// an integer SCALAR constant against a tensor mints an INT dense splat
/// (`dense<4> : tensor<64x64xi32>`), which `from_ttir` carries as `Attr::Verbatim`
/// -- matching what the text reader produces, the two sides agree -- and
/// `to_ktir_emit` refuses by name: neither `Attr` vocabulary has an int-splat form,
/// because no fixture ever needed one (the only int splat in tree is
/// `tt.divisibility`, which is dropped). So `it < 4` is refused end-to-end while
/// `it < k` (a second derived tile) lowers clean. A router kernel that needs an int
/// splat will get the named refusal, which is the E0004 law working; the router's
/// own lanes are spellable tensor-vs-tensor (its one-hot and rank products are f16).
const SRC: &str = r#"
import triton
import triton.language as tl


@triton.jit
def cmpsel_fwd(desc_x, desc_x2, desc_x3, desc_o,
               M: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr):
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, D], strides=[D, 1],
                                       block_shape=[BLOCK_M, D])
    x2_desc = tl.make_tensor_descriptor(desc_x2, shape=[M, D], strides=[D, 1],
                                        block_shape=[BLOCK_M, D])
    x3_desc = tl.make_tensor_descriptor(desc_x3, shape=[M, D], strides=[D, 1],
                                        block_shape=[BLOCK_M, D])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, D], strides=[D, 1],
                                       block_shape=[BLOCK_M, D])
    x = x_desc.load([0, 0])
    x2 = x2_desc.load([0, 0])
    x3 = x3_desc.load([0, 0])
    it = x2.to(tl.int32)
    k = x3.to(tl.int32)
    g = it + k
    h = g.to(tl.float16)
    mask_f = x > 0.0
    mask_i = it < k
    picked = tl.where(mask_f, x, h)
    sel_i = tl.where(mask_i, g, it)
    o_desc.store([0, 0], picked + sel_i.to(tl.float16))
"#;

const M: usize = 64;
const D: usize = 64;

fn spec() -> KernelSpec {
    let mut signature = HashMap::new();
    for k in ["desc_x", "desc_x2", "desc_x3", "desc_o"] {
        signature.insert(
            k.to_string(),
            ArgSpec::parse("*fp16").expect("*fp16 parses"),
        );
    }
    for k in ["M", "D", "BLOCK_M"] {
        signature.insert(
            k.to_string(),
            ArgSpec::parse("constexpr").expect("constexpr parses"),
        );
    }
    let mut constexprs = HashMap::new();
    for k in ["M", "D", "BLOCK_M"] {
        constexprs.insert(k.to_string(), Val::Int(M as i128));
    }
    KernelSpec {
        kernel: "cmpsel_fwd".to_string(),
        signature,
        constexprs,
        file: "cmpsel_fwd_test.py".to_string(),
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

/// The census test: exactly the ops the five spellings map to, with their predicates.
#[test]
fn the_handoff_carries_the_five_spellings_and_their_predicates() {
    let f = lowered();
    let mut cmpf = 0;
    let mut cmpi = 0;
    let mut select = 0;
    let mut sitofp = 0;
    let mut fptosi = 0;
    for op in f.ops_deep() {
        match op.op_type {
            OpKind::ArithCmpf => {
                cmpf += 1;
                assert!(
                    matches!(op.attr(AttrKey::Predicate), Some(Attr::Str(s)) if *s == "ogt"),
                    "the cmpf's predicate must arrive as `ogt` under KAttrKey::Predicate; got \
                     {:?}. A predicate that vanished or changed spelling between \
                     `Attr::Pred` and the handoff is the wiring defect this census exists \
                     to catch",
                    op.attr(AttrKey::Predicate)
                );
            }
            OpKind::ArithCmpi => {
                cmpi += 1;
                assert!(
                    matches!(op.attr(AttrKey::Predicate), Some(Attr::Str(s)) if *s == "slt"),
                    "the cmpi's predicate must arrive as `slt` under KAttrKey::Predicate; got \
                     {:?}",
                    op.attr(AttrKey::Predicate)
                );
            }
            OpKind::ArithSelect => select += 1,
            OpKind::ArithSitofp => sitofp += 1,
            OpKind::ArithFptosi => fptosi += 1,
            _ => {}
        }
    }
    assert_eq!(
        (cmpf, cmpi, select, sitofp, fptosi),
        (1, 1, 2, 2, 2),
        "the census of the five spellings. The kernel states one float compare (`x > 0.0`), \
         one int compare (`it < k`), two `tl.where`s, two int->float casts (`g.to(f16)`, \
         `sel_i.to(f16)`) and two float->int casts (`x2.to(tl.int32)`, `x3.to(tl.int32)`). \
         A different count means a spelling fell to a wrong KOpKind or a pass folded \
         something it must not: (cmpf, cmpi, select, sitofp, fptosi) above"
    );
}

/// The input tiles: small integers only, so every step is exact (see the module doc).
/// `X` straddles zero for the float compare; `X2` goes negative so a mis-wired cast
/// (`fptosi` -> `fptoui`, say) moves `it` at those lanes; `X3` straddles `X2`'s range
/// so the int compare flips within the tile.
const X: &[f64] = &[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
const X2: &[f64] = &[-3.0, -2.0, 0.0, 1.0, 3.0, 4.0, 5.0, 7.0];
const X3: &[f64] = &[1.0, 6.0, 2.0, 5.0, 0.0, 8.0, 3.0, 4.0];

/// The host model of the kernel, `flip_predicate` inverting the float compare -- the
/// control that proves the executed comparison can see a wrong predicate.
fn model(x: &[f64], x2: &[f64], x3: &[f64], flip_predicate: bool) -> Vec<f64> {
    x.iter()
        .zip(x2)
        .zip(x3)
        .map(|((&xv, &x2v), &x3v)| {
            let it = x2v as i64;
            let k = x3v as i64;
            let g = it + k;
            let h = g as f64; // f16 of a small integer: exact
            let mask_f = if flip_predicate { xv < 0.0 } else { xv > 0.0 };
            let picked = if mask_f { xv } else { h };
            let sel_i = if it < k { g } else { it };
            // f16 addition of integers this small: exact.
            picked + sel_i as f64
        })
        .collect()
}

/// The numbers, exactly: `max|err|` must be ZERO, and the flipped-predicate model
/// must differ -- the comparison's own control.
#[test]
fn the_executed_kernel_computes_the_where_compare_cast_function_exactly() {
    let f = lowered();
    let n = M * D;
    let x: Vec<f64> = (0..n).map(|i| X[i % X.len()]).collect();
    let x2: Vec<f64> = (0..n).map(|i| X2[i % X2.len()]).collect();
    let x3: Vec<f64> = (0..n).map(|i| X3[i % X3.len()]).collect();
    let encode = |v: &[f64]| {
        ktir_core::codec::encode(&v.iter().map(|&a| a as f32).collect::<Vec<_>>(), DType::F16)
    };

    let params: Vec<ktir_core::ir::Ssa> = f.arguments.iter().map(|(s, _)| *s).collect();
    assert_eq!(
        params.len(),
        4,
        "desc_x, desc_x2, desc_x3, desc_o -- nothing else becomes an argument"
    );
    // THE BINDING IS POSITIONAL (the crate's own law): argument i is pointer parameter
    // i of the kernel, in declaration order -- x, x2, x3, then the output.
    let bound: [Vec<u8>; 4] = [encode(&x), encode(&x2), encode(&x3), encode(&vec![0.0; n])];
    let args: Vec<(ktir_core::ir::Ssa, Arg)> = params
        .iter()
        .zip(bound)
        .map(|(&ssa, bytes)| {
            (
                ssa,
                Arg::TensorBytes {
                    data: bytes,
                    shape: vec![M, D],
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
        .get(&params[3])
        .expect("desc_o is written by the store and read back");
    let got: Vec<f64> = out.data.iter().map(|&v| v as f64).collect();

    let truth = model(&x, &x2, &x3, false);
    let flipped = model(&x, &x2, &x3, true);
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
        "cmpsel  n={n}  max|err| = {max_err:.6e} at [{worst}] (got {}, want {})",
        got[worst], truth[worst]
    );
    assert_eq!(
        max_err,
        0.0,
        "every value is a small integer, so every step -- the f32 compare, the f16 add, both \
         casts -- is EXACT; any nonzero error is a wiring defect, not rounding. First excess \
         at [{worst}]: got {:+.6e}, want {:+.6e} (x = {:+.6e}, x2 = {:+.6e}, x3 = {:+.6e})",
        got[worst],
        truth[worst],
        X[worst % X.len()],
        X2[worst % X2.len()],
        X3[worst % X3.len()]
    );
    // THE CONTROL: the flipped-predicate model must differ from the truth somewhere,
    // or the comparison above cannot see a wrong predicate and its green result says
    // nothing about which way `ogt` points.
    let flips: usize = truth.iter().zip(&flipped).filter(|(t, w)| t != w).count();
    assert!(
        flips > 0,
        "the flipped-predicate control differs NOWHERE: the kernel's output does not depend \
         on the float compare's direction, so the exact comparison above is vacuous"
    );
    println!(
        "control  the flipped-predicate model differs at {flips}/{n} elements (the comparison \
         can see a wrong predicate)"
    );
}

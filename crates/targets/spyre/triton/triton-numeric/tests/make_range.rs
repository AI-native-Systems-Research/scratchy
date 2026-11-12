// SPDX-License-Identifier: Apache-2.0

//! THE LANE-VECTOR LADDER EXTENSION, EXECUTED: `tl.arange` -> `tt.make_range`
//! -> `tensor.generate`.
//!
//! `from_ttir`'s `OP_SET` deliberately carried no `tt.make_range` until the MoE
//! router rows needed it: in every golden the arange is dead code (the attention
//! mask is host-precomputed, so `make_ttir`'s DCE kills it), and the router
//! kernels CANNOT precompute their way out -- `rank[i, j]` compares every lane
//! against every lane, and the lane vector IS the computation.
//!
//! The extension is two rewrites, and this file drives the whole value chain
//! over one kernel that spells both forms:
//!
//! ```text
//!   codegen::compile   .py -> raw ttir     (`tl.arange` -> `tt.make_range` with
//!                                           `start`/`end` attrs)
//!   opt::make_ttir     inline/fold/cse/dce
//!   from_ttir          the OP_SET/ATTR_KEYS gate -- `start`/`end` ride
//!                       `AttrKey::Other`
//!   to_ktir            `convert_make_range`: the op becomes a `tensor.generate`
//!                       group (block arg + optional const/addi + `tensor.yield`)
//!   make_ktir + to_ktir + to_ktir_emit     -> ktir_core::IRFunction, the block
//!                       arg lifted into a `region.bb0_args` marker op
//!   ktir_emulator      the executed numbers
//! ```
//!
//! # WHAT THE TWO TESTS OW
//!
//! 1. **The census**: exactly two `TensorGenerate`, each carrying `Shape = [N]`
//!    and `Dtype = I32` -- the stamp `state_shape_and_dtype` derives from the
//!    result type, and it is LOAD-BEARING, not cosmetic: the emulator's generate
//!    handler DEFAULTS a missing `Dtype` to F16, and an f16-typed lane vector is
//!    a silent dtype change no numbers test at router scale would catch (every
//!    lane index under 2048 is exact in f16). Each region holds one
//!    `RegionBb0Args` whose `Names` entry is the grid argument its body reads --
//!    the wiring the executor's meshgrid binding keys on. The `start == 0` form
//!    yields that argument directly (the branch's own `index_grid` shape); the
//!    `start != 0` form holds `arith.constant 4 : i32` + `arith.addi` over it,
//!    the offset body this extension pins.
//! 2. **The numbers, against a host model, EXACTLY**: every value is a small
//!    integer, so the i32 grid, the offset addi, the i32->f16 cast and the f16
//!    adds are all exact -- `max|err|` must be ZERO. A WRONG-`start` control
//!    (the host model treating the offset arange as starting at 0) must DIFFER,
//!    which is what proves the executed comparison can see a wrong `start` at
//!    all -- an arange whose bounds vanished between `Attr::Other` and the
//!    rewrite would otherwise read as green.
//!
//! ⚠️ WHY THE LANES REACH THE OUTPUT THROUGH AN f16 ADD AND NOT A STORE OF THE
//! RAW i32 TILE: no executed fixture plain-STORES an i32 tile (embedding's
//! `*i32` ids feed `tt.descriptor_gather`, not a store), while f16 stores are
//! exercised by every configuration here -- so the kernel derives its check
//! value with `.to(tl.float16)`, the cast the router's own fma lanes use, and
//! adds no unexercised store path while proving the grid.

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

/// The kernel: one f16 tile in, both arange forms, one f16 tile out.
///
/// `lanes` is the `start == 0` form the router's one-hot selection spells;
/// `off` is the `start != 0` form, which is the one that exercises the
/// const+addi body -- and the one the wrong-`start` control perturbs.
const SRC: &str = r#"
import triton
import triton.language as tl


@triton.jit
def arange_fwd(desc_x, desc_o, M: tl.constexpr, N: tl.constexpr):
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, N], strides=[N, 1],
                                       block_shape=[M, N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N], strides=[N, 1],
                                       block_shape=[M, N])
    x = x_desc.load([0, 0])
    lanes = tl.arange(0, N).to(tl.float16)
    off = tl.arange(4, 4 + N).to(tl.float16)
    o_desc.store([0, 0], x + lanes[None, :] + off[None, :])
"#;

const M: usize = 64;
const N: usize = 64;

fn spec() -> KernelSpec {
    let mut signature = HashMap::new();
    for k in ["desc_x", "desc_o"] {
        signature.insert(
            k.to_string(),
            ArgSpec::parse("*fp16").expect("*fp16 parses"),
        );
    }
    for k in ["M", "N"] {
        signature.insert(
            k.to_string(),
            ArgSpec::parse("constexpr").expect("constexpr parses"),
        );
    }
    let mut constexprs = HashMap::new();
    for (k, v) in [("M", M as i128), ("N", N as i128)] {
        constexprs.insert(k.to_string(), Val::Int(v));
    }
    KernelSpec {
        kernel: "arange_fwd".to_string(),
        signature,
        constexprs,
        file: "arange_fwd_test.py".to_string(),
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

/// One generate group's region as (bb0 names, body ops), for the census.
fn region_of<'f>(op: &ktir_core::ir::Operation<'f>) -> &'f [ktir_core::ir::Operation<'f>] {
    op.regions.first().copied().unwrap_or(&[])
}

/// The census test: two generate groups, each shaped and typed as the executor
/// reads them, one bare-arg body and one const+addi body.
#[test]
fn the_handoff_carries_two_generate_groups_shaped_and_typed() {
    let f = lowered();
    let generates: Vec<_> = f
        .ops_deep()
        .into_iter()
        .filter(|op| op.op_type == OpKind::TensorGenerate)
        .collect();
    assert_eq!(
        generates.len(),
        2,
        "the kernel spells two aranges, so exactly two tensor.generate groups must \
         survive the whole chain. A different count means a pass folded or DCE'd one \
         it must not, or a make_range never became a generate"
    );

    let yields: usize = f
        .ops_deep()
        .iter()
        .filter(|op| op.op_type == OpKind::TensorYield)
        .count();
    assert_eq!(yields, 2, "one tensor.yield per generate body");

    let mut bare_arg = 0;
    let mut offset_body = 0;
    for op in &generates {
        // THE TYPE STAMP IS LOAD-BEARING: the emulator's generate handler defaults
        // a missing Dtype to F16, so the i32 lane vector must arrive with
        // `Dtype = I32` and `Shape = [N]` or it silently changes dtype.
        assert!(
            matches!(op.attr(AttrKey::Dtype), Some(Attr::Dtype(DType::I32))),
            "the generate's Dtype must arrive as I32; got {:?}. The emulator defaults \
             a missing Dtype to F16, which is a silent dtype change on the lane vector",
            op.attr(AttrKey::Dtype)
        );
        assert!(
            matches!(op.attr(AttrKey::Shape), Some(Attr::IntList(s)) if *s == [N as i64]),
            "the generate's Shape must arrive as [{N}]; got {:?}",
            op.attr(AttrKey::Shape)
        );

        let region = region_of(op);
        let bb0: Vec<_> = region
            .iter()
            .filter(|o| o.op_type == OpKind::RegionBb0Args)
            .collect();
        assert_eq!(
            bb0.len(),
            1,
            "exactly one region.bb0_args marker per generate"
        );
        let names = match bb0[0].attr(AttrKey::Names) {
            Some(Attr::Ssas(names)) => *names,
            other => panic!("the bb0 marker must carry Names = Ssas; got {other:?}"),
        };
        assert_eq!(
            names.len(),
            1,
            "tl.arange's grid is rank 1, so one block arg"
        );

        let body: Vec<_> = region
            .iter()
            .filter(|o| o.op_type != OpKind::RegionBb0Args)
            .collect();
        let last = body.last().expect("a generate body has a terminator");
        assert_eq!(
            last.op_type,
            OpKind::TensorYield,
            "the body's last op is the yield"
        );

        if body.len() == 1 {
            // THE start==0 FORM yields the block argument the marker names: the
            // executor binds only the names the marker states, so a yield of any
            // other SSA is a body whose index has no value.
            assert_eq!(
                last.operands,
                &[names[0]],
                "the start==0 body yields the block argument the bb0 marker names"
            );
            bare_arg += 1;
        } else {
            // THE start!=0 FORM: const(start) + addi(arg, const) + yield(addi).
            assert_eq!(
                body.len(),
                3,
                "the start!=0 body is const + addi + yield; got {} ops",
                body.len()
            );
            assert_eq!(body[0].op_type, OpKind::ArithConstant);
            assert!(
                matches!(body[0].attr(AttrKey::Value), Some(Attr::Int(4))),
                "the offset arange starts at 4; got {:?}",
                body[0].attr(AttrKey::Value)
            );
            assert_eq!(body[1].op_type, OpKind::ArithAddi);
            assert_eq!(
                body[1].operands,
                &[names[0], body[0].result.unwrap()],
                "the addi reads the grid arg plus the start constant"
            );
            assert_eq!(
                last.operands,
                &[body[1].result.unwrap()],
                "the offset body yields the addi's result"
            );
            offset_body += 1;
        }
    }
    assert_eq!(
        (bare_arg, offset_body),
        (1, 1),
        "one start==0 arange (`lanes`) and one start!=0 arange (`off`)"
    );
}

/// The input tile: small non-negative integers only, so the grid, the offset
/// addi, the i32->f16 cast and the f16 adds are all exact (see the module doc).
const X: &[f64] = &[0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0];

/// The host model of the kernel, `wrong_start` treating the offset arange as
/// starting at 0 -- the control that proves the executed numbers can see a
/// wrong `start`.
fn model(x: &[f64], wrong_start: bool) -> Vec<f64> {
    x.iter()
        .enumerate()
        .map(|(i, &xv)| {
            let j = (i % N) as i64;
            let off = if wrong_start { j } else { 4 + j };
            // f16 of integers this small: exact.
            xv + j as f64 + off as f64
        })
        .collect()
}

/// The numbers, exactly: `max|err|` must be ZERO, and the wrong-`start` model
/// must differ -- the extension's own control.
#[test]
fn the_executed_kernel_computes_the_lane_vectors_exactly() {
    let f = lowered();
    let n = M * N;
    let x: Vec<f64> = (0..n).map(|i| X[i % X.len()]).collect();
    let encode = |v: &[f64]| {
        ktir_core::codec::encode(&v.iter().map(|&a| a as f32).collect::<Vec<_>>(), DType::F16)
    };

    let params: Vec<ktir_core::ir::Ssa> = f.arguments.iter().map(|(s, _)| *s).collect();
    assert_eq!(
        params.len(),
        2,
        "desc_x, desc_o -- nothing else becomes an argument"
    );
    // THE BINDING IS POSITIONAL (the crate's own law): argument i is pointer
    // parameter i of the kernel, in declaration order -- x, then the output.
    let bound: [Vec<u8>; 2] = [encode(&x), encode(&vec![0.0; n])];
    let args: Vec<(ktir_core::ir::Ssa, Arg)> = params
        .iter()
        .zip(bound)
        .map(|(&ssa, bytes)| {
            (
                ssa,
                Arg::TensorBytes {
                    data: bytes,
                    shape: vec![M, N],
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
        .expect("desc_o is written by the store and read back");
    let got: Vec<f64> = out.data.iter().map(|&v| v as f64).collect();

    let truth = model(&x, false);
    let wrong = model(&x, true);
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
        "arange  n={n}  max|err| = {max_err:.6e} at [{worst}] (got {}, want {})",
        got[worst], truth[worst]
    );
    assert_eq!(
        max_err,
        0.0,
        "every value is a small integer, so every step -- the i32 grid, the offset \
         addi, the i32->f16 cast, the f16 adds -- is EXACT; any nonzero error is a \
         wiring defect, not rounding. First excess at [{worst}]: got {:+.6e}, want \
         {:+.6e} (x = {:+.6e})",
        got[worst],
        truth[worst],
        X[worst % X.len()]
    );
    // THE CONTROL: the wrong-`start` model must differ from the truth somewhere,
    // or the comparison above cannot see a vanished `start` and its green result
    // says nothing about which value the offset grid begins at.
    let flips: usize = truth.iter().zip(&wrong).filter(|(t, w)| t != w).count();
    assert!(
        flips > 0,
        "the wrong-start control must differ from the truth, or this test cannot see \
         a `start` that vanished between the ttir attrs and the generate body"
    );
}

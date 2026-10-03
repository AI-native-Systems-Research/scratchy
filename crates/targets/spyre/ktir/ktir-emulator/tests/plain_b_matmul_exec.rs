// SPDX-License-Identifier: Apache-2.0
//! THE PLAIN-B MATMUL THROUGH THE FULL EXECUTION PATH — the splice's runtime gate.
//!
//! The byte-identity golden (`scratchy-target-spyre/tests/triton_splice_golden.rs`)
//! pins the DESCRIPTOR emission of a spliced dense matmul, whose Triton ladder form
//! loads the weight `[k, n]` DIRECTLY (the canonical single-dot contract
//! `verify_canonical_matmul_kernel` pins: B must be a direct descriptor_load, and a
//! `.T` on it is refused by name). But the descriptor layer is not the only consumer:
//! the resident/segmented EXECUTION path re-tiles every `linalg.matmul` for the 2 MB
//! LX (`ktir_optimizer::matmul_tile`), and that tiler historically recognized only
//! the builder's `[n, k]` transpose-B W tile — a plain-`[k, n]` contraction was left
//! UNTILED (latent LX overflow) before the orientation was taught to it.
//!
//! This test drives ONE plain-B contraction through `program::execute` — the same
//! entry the resident serving path takes, with `apply_attention_rewrites` (and its
//! matmul tiling) applied — and checks the numbers against a plain host reference.

use ktir_emulator::ktir_optimizer::fusion::{Binding, NodeSpec, ProgramSpec};
use ktir_emulator::interpreter::Arg;
use ktir_emulator::program::execute;
use ktir_emulator::{
    arena::Arena, attrkey::AttrKey, codec, dtypes::DType, ir::Attr, ir::IRFunction,
    ir::Operation, ir::Ssa, irtype::IrType, opkind::OpKind,
};

/// Build the plain-B single-dot form the splice mints: A `[m, k]`, W `[k, n]`, out
/// `[m, n]`, all whole-region, matmul with NO `indexing_maps` (MLIR's plain default).
fn plain_b_func(m: usize, k: usize, n: usize) -> IRFunction<'static> {
    let a = Arena::global();
    let hbm: &'static str = Box::leak("HBM".to_string().into_boxed_str());
    let (pa, pw, po) = (Ssa(0), Ssa(1), Ssa(2));
    let mut ops: Vec<Operation<'static>> = Vec::new();
    let view = |ops: &mut Vec<Operation<'static>>, res: Ssa, ptr: Ssa, r: usize, c: usize| {
        let op = Operation::new(a, Some(res), OpKind::KtdpConstructMemoryView, &[ptr])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r as i64, c as i64])))
            .with_attr(a, AttrKey::Strides, Attr::IntList(a.ints(vec![c as i64, 1])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
            .with_attr(a, AttrKey::MemorySpace, Attr::Str(hbm));
        ops.push(op);
    };
    view(&mut ops, Ssa(10), pa, m, k);
    view(&mut ops, Ssa(11), pw, k, n);
    view(&mut ops, Ssa(12), po, m, n);
    // The splat seed the matmul's outs reads — an f16 `[m, n]` zero, exactly the
    // `KtirFunc::splat_zero` form (scalar constant -> `tensor.splat`).
    let zc = Ssa(13);
    ops.push(Operation {
        result_type: Some(IrType::Scalar(DType::F16)),
        ..Operation::new(a, Some(zc), OpKind::ArithConstant, &[])
            .with_attr(a, AttrKey::Value, Attr::Float(0.0))
    });
    let zero = Ssa(14);
    ops.push(Operation {
        result_type: Some(IrType::Tensor {
            dims: a.ints(vec![m as i64, n as i64]),
            elem: DType::F16,
        }),
        ..Operation::new(a, Some(zero), OpKind::TensorSplat, &[zc])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![m as i64, n as i64])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    });
    // The index-typed zero every access-tile corner reads.
    let idx0 = Ssa(15);
    ops.push(Operation {
        result_type: Some(IrType::Index),
        ..Operation::new(a, Some(idx0), OpKind::ArithConstant, &[])
            .with_attr(a, AttrKey::Value, Attr::Int(0))
    });
    let load = |ops: &mut Vec<Operation<'static>>, res: Ssa, view_: Ssa, r: usize, c: usize| {
        let acc = Operation::new(a, Some(res), OpKind::KtdpConstructAccessTile, &[view_, idx0])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r as i64, c as i64])));
        ops.push(acc);
        let ld = Operation::new(a, Some(res), OpKind::KtdpLoad, &[res])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r as i64, c as i64])));
        ops.push(ld);
    };
    // ⛔ SAME SSA FOR THE ACCESS TILE AND THE LOAD, exactly the bug class this test
    // exists to catch being papered over: the real splice mints distinct SSAs; the
    // recognizer reads the LOAD's result type, so the shapes must be stated on both.
    load(&mut ops, Ssa(20), Ssa(10), m, k);
    load(&mut ops, Ssa(21), Ssa(11), k, n);
    let res = Ssa(22);
    let mm = Operation::new(a, Some(res), OpKind::LinalgMatmul, &[Ssa(20), Ssa(21), zero])
        .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![m as i64, n as i64])));
    ops.push(mm);
    // The output store: the drain the recognizer requires.
    let oat = Operation::new(a, Some(Ssa(23)), OpKind::KtdpConstructAccessTile, &[Ssa(12), idx0])
        .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![m as i64, n as i64])));
    ops.push(oat);
    let st = Operation::new(a, None, OpKind::KtdpStore, &[res, Ssa(23)]);
    ops.push(st);
    IRFunction {
        name: a.str("plain_b_mm".to_string()),
        arguments: a.args(vec![
            (pa, IrType::Index),
            (pw, IrType::Index),
            (po, IrType::Index),
        ]),
        operations: a.ops(ops),
        grid: (1, 1, 1),
        return_type: None,
    }
}

#[test]
fn plain_b_matmul_executes_through_the_tiled_segmented_path() {
    let (m, k, n) = (1usize, 256, 128); // NON-SQUARE on purpose: k != n.
    let func = plain_b_func(m, k, n);
    let funcs = Arena::global().funcs(vec![func]);
    let spec = ProgramSpec {
        nodes: vec![NodeSpec {
            func: "plain_b_mm".to_string(),
            bindings: vec![
                Binding { arg: Ssa(0), tensor: 0, is_output: false },
                Binding { arg: Ssa(1), tensor: 1, is_output: false },
                Binding { arg: Ssa(2), tensor: 2, is_output: true },
            ],
        }],
        sources: [0u64, 1u64].into_iter().collect(),
        results: [2u64].into_iter().collect(),
    };
    let host: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01 - 1.0).collect();
    let weight: Vec<f32> = (0..k * n).map(|i| ((i % 17) as f32) * 0.02 - 0.16).collect();
    // The host reference: out[m, n] = Σ_k host[m, k] · weight[k, n] (PLAIN B).
    let mut want = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += host[mi * k + ki] * weight[ki * n + ni];
            }
            want[mi * n + ni] = acc;
        }
    }
    let args: Vec<(u64, Arg)> = vec![
        (
            0,
            Arg::TensorBytes {
                data: codec::encode(&host, DType::F16),
                shape: vec![m, k],
                dtype: DType::F16,
            },
        ),
        (
            1,
            Arg::TensorBytes {
                data: codec::encode(&weight, DType::F16),
                shape: vec![k, n],
                dtype: DType::F16,
            },
        ),
        (
            2,
            Arg::TensorBytes {
                data: codec::encode(&vec![0.0f32; m * n], DType::F16),
                shape: vec![m, n],
                dtype: DType::F16,
            },
        ),
    ];
    let out = execute(funcs, &spec, &args, &[2]).expect("execute plain-B matmul");
    let out = &out[&2];
    let got = &out.data;
    let mut max_abs = 0.0f32;
    for (g, w) in got.iter().zip(&want) {
        max_abs = max_abs.max((g - w).abs());
    }
    assert!(
        max_abs < 0.05,
        "plain-B matmul diverged from the host reference: max abs err {max_abs}"
    );
}

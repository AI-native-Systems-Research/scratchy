// SPDX-License-Identifier: Apache-2.0
//! A direct probe: the SAME rmsnorm shape lowered twice through the splice must hit
//! the compile memo — measured by wall time (first = compile, second = cache hit).
use std::time::Instant;

use scratchy_subtile::subtile_ir::{
    GainConvention, SubOp, SubtileIR, SubtileId, SubtileNode, TensorId, TensorRegion, TensorShape,
};

fn main() {
    let ir = probe_ir();
    let node = &ir.nodes[0];
    let t0 = Instant::now();
    let _a = scratchy_triton_splice::lower(node, &ir, false)
        .unwrap_or_else(|e| panic!("first compile: {e}"));
    let d0 = t0.elapsed();
    let t1 = Instant::now();
    let _b = scratchy_triton_splice::lower(node, &ir, false)
        .unwrap_or_else(|e| panic!("second compile: {e}"));
    let d1 = t1.elapsed();
    println!(
        "first:  {d0:?}\nsecond: {d1:?}\nratio:  {:.1}x",
        d0.as_secs_f64() / d1.as_secs_f64()
    );
}

fn probe_ir() -> SubtileIR<scratchy_subtile::subtile_ir::NeoX> {
    let tensors = vec![
        TensorShape {
            rows: 1,
            cols: 2048,
        },
        TensorShape {
            rows: 1,
            cols: 2048,
        },
        TensorShape {
            rows: 1,
            cols: 2048,
        },
    ];
    let whole = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let node = SubtileNode {
        id: SubtileId::from_index(0),
        op: SubOp::RmsNorm {
            eps: 1e-5,
            gain: GainConvention::Scale,
        },
        inputs: vec![whole(0), whole(1)],
        output: whole(2),
    };
    SubtileIR {
        tensors,
        num_sources: 2,
        nodes: vec![node],
        result: TensorId::from_index(2),
        op_output: Vec::new(),
    }
}

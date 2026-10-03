// SPDX-License-Identifier: Apache-2.0
//! THE SPLICE'S BYTE-IDENTITY GATE — the first Triton-sourced kernel vs the builder.
//!
//! One rmsnorm node, lowered twice through the SAME door (`ktir_superdsc_door::lower`,
//! under the SAME `BundleLayout` the tape walk computes):
//!
//! 1. the BUILDER path — `lower_graph_to_ktir`'s `KtirFunc::rmsnorm`, the card-proven
//!    status quo;
//! 2. the SPLICE — `scratchy_triton_splice::lower`, which compiles
//!    `crates/targets/spyre/kernels/rmsnorm.py` at expansion time through the re-hosted
//!    Triton ladder and hands back the same `EmittedOp`.
//!
//! The descriptors must be byte-identical. That is the landing gate for every registry
//! row: the builder arm for an op is deleted only when this passes for the shapes in
//! scope, and a row that passes here cannot emit a different descriptor than the one
//! the card has already run.
//!
//! ⛔ DESCRIPTOR LEVEL, NOT KTIR LEVEL. The two producers legitimately spell the program
//! differently (`math.sqrt(mean + eps)` with an f32 divisor vs `rsqrt` with a folded
//! reciprocal) — the consumer assembles descriptors from `regions()` + the program's
//! stated constants, so the op soup never reaches them, and the emitted `SdscOp` JSON is
//! the strongest gate that is not also a false one.
//!
//! Lives in `tests/` so it can name the builder as the control (a private-free public
//! API walk) and the splice (a dev-dependency), without the crate's own feature gates.

use std::collections::HashSet;

use ktir_superdsc::ktir_node::ActiveCap;
use scratchy_subtile::subtile_ir::{
    GainConvention, SubOp, SubtileIR, SubtileNode, TensorId, TensorRegion, TensorShape,
};
use scratchy_target_spyre::ktir_superdsc_door::lower as door_lower;
use scratchy_target_spyre::lower_subtile_tape_to_ktir::lower_graph_to_ktir;

/// The rmsnorm shapes that matter for the delivery scope: granite 3.2/3.3 at 2b and 8b
/// both normalize at hidden 2048 (2b) and 4096 (8b), decode rows 1 and a prefill rung's
/// width. (M, D_MODEL).
const SHAPES: &[(u32, u32)] = &[(1, 2048), (1, 4096), (31, 2048), (64, 4096)];

#[test]
fn spliced_rmsnorm_is_byte_identical_to_the_builder() {
    for &(m, c) in SHAPES {
        let ir = rmsnorm_ir(m, c);
        let weight_ids: HashSet<u32> = [0u32, 1u32].into_iter().collect();

        // 1. The builder path — the control.
        let (builder_ops, layout) =
            lower_graph_to_ktir(&ir, &weight_ids, ActiveCap::FULL, false)
                .unwrap_or_else(|e| panic!("builder lowered m={m} c={c}: {e}"));
        let [builder] = &builder_ops[..] else {
            panic!("one rmsnorm node lowers to one op, got {}", builder_ops.len())
        };
        let builder_ktir = builder.ktir.as_ref().expect("builder op carries its program");

        // 2. The splice — the row compiles the kernel for this node.
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower(node, &ir)
            .unwrap_or_else(|e| panic!("splice compiled m={m} c={c}: {e}"))
            .expect("registry has a row for Scale-gain RmsNorm");

        // ⛔ THE NAME LAW IS PART OF THE GATE. The builder names its program
        // `rmsnorm_s{id}`; the splice reuses the law so the op_name and the emulator's
        // function key are identical.
        assert_eq!(spliced.op_name, builder.op_name, "op_name (m={m} c={c})");

        // 3. Both programs go through the SAME door under the SAME layout — the consumer
        //    is the thing being pinned.
        let mut sym = 0i64;
        let mut quantized = HashSet::new();
        let builder_emitted = door_lower(
            builder_ktir,
            &mut sym,
            Some(&layout),
            &mut quantized,
            None,
        )
        .unwrap_or_else(|e| panic!("builder program lowered (m={m} c={c}): {}", e.message));
        let mut sym = 0i64;
        let spliced_emitted = door_lower(
            spliced.ktir.as_ref().expect("spliced op carries its program"),
            &mut sym,
            Some(&layout),
            &mut quantized,
            None,
        )
        .unwrap_or_else(|e| panic!("spliced program lowered (m={m} c={c}): {}", e.message));

        assert_eq!(
            builder_emitted.len(),
            spliced_emitted.len(),
            "op count (m={m} c={c})"
        );
        for (b, s) in builder_emitted.iter().zip(spliced_emitted.iter()) {
            let bj = serde_json::to_string(b.dsc()).unwrap();
            let sj = serde_json::to_string(s.dsc()).unwrap();
            assert_eq!(
                bj, sj,
                "descriptor bytes (m={m} c={c}): builder vs splice diverged"
            );
            assert_eq!(b.op_name, s.op_name, "emitted op_name (m={m} c={c})");
        }
    }
}

/// `rmsnorm(x, gamma) -> out` as a one-node [`SubtileIR`] — the same fixture shape
/// `superdsc_time_tile.rs` mints for its matmul control. t0 = x source, t1 = gamma
/// source, t2 = result. Eps is granite's `1e-5`.
fn rmsnorm_ir(m: u32, c: u32) -> SubtileIR {
    let tensors = vec![
        TensorShape { rows: m, cols: c },
        TensorShape { rows: 1, cols: c },
        TensorShape { rows: m, cols: c },
    ];
    let whole = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let node = SubtileNode {
        id: scratchy_subtile::subtile_ir::SubtileId::from_index(0),
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

/// The silu-mul shapes that matter for the delivery scope: granite's d_ff (2b: 0, 8b:
/// 12800) at decode rows and a prefill rung's width. (M, N).
const SILUMUL_SHAPES: &[(u32, u32)] = &[(1, 4096), (1, 12800), (31, 4096), (64, 12800)];

#[test]
fn spliced_silumul_is_byte_identical_to_the_builder() {
    for &(m, c) in SILUMUL_SHAPES {
        let ir = silumul_ir(m, c);
        let weight_ids: HashSet<u32> = [0u32, 1u32].into_iter().collect();

        // 1. The builder path — the control.
        let (builder_ops, layout) =
            lower_graph_to_ktir(&ir, &weight_ids, ActiveCap::FULL, false)
                .unwrap_or_else(|e| panic!("builder lowered m={m} c={c}: {e}"));
        let [builder] = &builder_ops[..] else {
            panic!("one silumul node lowers to one op, got {}", builder_ops.len())
        };
        let builder_ktir = builder.ktir.as_ref().expect("builder op carries its program");

        // 2. The splice — the row compiles the kernel for this node.
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower(node, &ir)
            .unwrap_or_else(|e| panic!("splice compiled m={m} c={c}: {e}"))
            .expect("registry has a row for SiluMul");

        // ⛔ THE NAME LAW IS PART OF THE GATE — `silumul_s{id}` on both paths.
        assert_eq!(spliced.op_name, builder.op_name, "op_name (m={m} c={c})");

        // 3. Both programs go through the SAME door under the SAME layout.
        let mut sym = 0i64;
        let mut quantized = HashSet::new();
        let builder_emitted = door_lower(
            builder_ktir,
            &mut sym,
            Some(&layout),
            &mut quantized,
            None,
        )
        .unwrap_or_else(|e| panic!("builder program lowered (m={m} c={c}): {}", e.message));
        let mut sym = 0i64;
        let spliced_emitted = door_lower(
            spliced.ktir.as_ref().expect("spliced op carries its program"),
            &mut sym,
            Some(&layout),
            &mut quantized,
            None,
        )
        .unwrap_or_else(|e| panic!("spliced program lowered (m={m} c={c}): {}", e.message));

        assert_eq!(
            builder_emitted.len(),
            spliced_emitted.len(),
            "op count (m={m} c={c})"
        );
        for (b, s) in builder_emitted.iter().zip(spliced_emitted.iter()) {
            let bj = serde_json::to_string(b.dsc()).unwrap();
            let sj = serde_json::to_string(s.dsc()).unwrap();
            assert_eq!(
                bj, sj,
                "descriptor bytes (m={m} c={c}): builder vs splice diverged"
            );
            assert_eq!(b.op_name, s.op_name, "emitted op_name (m={m} c={c})");
        }
    }
}

/// `silu(gate) * up -> out` as a one-node [`SubtileIR`]. t0 = gate source, t1 = up
/// source, t2 = result. Both operands are activations, but the builder path is
/// shape-driven and does not read the distinction, so the fixture pins both as sources.
fn silumul_ir(m: u32, c: u32) -> SubtileIR {
    let tensors = vec![
        TensorShape { rows: m, cols: c },
        TensorShape { rows: m, cols: c },
        TensorShape { rows: m, cols: c },
    ];
    let whole = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let node = SubtileNode {
        id: scratchy_subtile::subtile_ir::SubtileId::from_index(0),
        op: SubOp::SiluMul,
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

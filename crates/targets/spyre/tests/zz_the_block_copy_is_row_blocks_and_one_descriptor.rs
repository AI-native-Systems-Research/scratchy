// SPDX-License-Identifier: Apache-2.0
//! ⭐ THE BLOCK COPY IS WHOLE-WIDTH ROW BLOCKS IN ITS PROGRAM AND ONE IDENTITY ON THE WIRE.
//!
//! `blockcopy.py` states its copy as `N_BLOCKS` `[BLOCK_M, C_SRC]` tiles under the elementwise LX
//! law (`rows_per_block`), not one copy per stick. Pinned here:
//!
//! * THE WIRE IS UNCHANGED. `fixtures/sdsc_blockcopy_splice_per_stick.json` is the door's output for
//!   seven copies through the splice as the per-stick kernel stated them (generated on nickm5 through
//!   `lower_block_copy` and the door at the per-stick kernel, main's `blockcopy.py`: column blocks
//!   of a wider destination, whole-width copies up to a `[512, 2048]` prefill rung, and a `[515,
//!   2048]` column block whose rows leave a 3-row tail, the tail entry's one case) — the descriptor,
//!   its bindings, the symbol counter. Both row-blocked entries emit exactly those bytes.
//! * THE PROGRAM IS SMALL. A `[512, 2048]` copy is a few dozen KTIR ops, not ~200K (313 MB of
//!   build-script memory per such program, measured).

use scratchy_subtile::subtile_ir::{
    Range, Region, SubOp, SubtileId, SubtileNode, TensorId, TensorRegion,
};

type W = ktir_superdsc::collective::TpWorld<8>;

/// `(rows, c_src, c_dst, off)` — the golden's seven copies, in its order. The last one's rows are a
/// whole 512-row block and a 3-row tail at the 2048-wide law, so it takes `blockcopy_tail_fwd`.
const CASES: [(u32, u32, u32, u32); 7] = [
    (1, 1024, 8192, 2048),
    (4, 128, 512, 256),
    (2, 64, 192, 0),
    (1, 2048, 2048, 0),
    (7, 2048, 2048, 0),
    (512, 2048, 2048, 0),
    (515, 2048, 4096, 1024),
];

fn copy(
    i: usize,
    (rows, c_src, c_dst, off): (u32, u32, u32, u32),
) -> ktir_superdsc::emit::EmittedOp<W> {
    let tr = |t: usize, c0: u32, w: u32| TensorRegion {
        tensor: TensorId::from_index(t),
        region: Region {
            rows: Range::new(0, rows),
            cols: Range::new(c0, w),
        },
    };
    let node: SubtileNode = SubtileNode {
        id: SubtileId::from_index(12),
        op: SubOp::AllReduce,
        inputs: vec![tr(11, 0, c_src)],
        output: tr(12, off, c_src),
    };
    scratchy_triton_splice::lower_block_copy::<W, _>(
        &node,
        tr(11, 0, c_src),
        tr(12, off, c_src),
        c_dst,
        &format!("bc{i}"),
    )
    .unwrap_or_else(|e| panic!("case {i}: {e}"))
}

/// ⭐ The door's descriptors are the per-stick kernel's, byte for byte.
#[test]
fn the_row_blocked_copy_emits_the_per_stick_copys_descriptors() {
    let golden: Vec<serde_json::Value> = serde_json::from_str(include_str!(
        "fixtures/sdsc_blockcopy_splice_per_stick.json"
    ))
    .expect("the golden parses");
    let mut all = Vec::new();
    for (i, case) in CASES.into_iter().enumerate() {
        let e = copy(i, case);
        let k = e.ktir.as_ref().expect("a program");
        let mut sym = 0i64;
        let mut q = ktir_superdsc::emit::ktir_matmul_fp8::QuantizedActivations::default();
        let ops =
            scratchy_target_spyre::ktir_superdsc_door::lower::<W>(k, &mut sym, None, None, &mut q)
                .unwrap_or_else(|e| panic!("case {i}: {}", e.message));
        assert_eq!(ops.len(), 1, "case {i}: one identity");
        for op in ops {
            all.push(serde_json::json!({
                "op_name": op.op_name,
                "op": serde_json::to_value(&op.op).expect("serializes"),
                "arg_bindings": format!("{:?}", op.arg_bindings),
                "sym": sym,
            }));
        }
    }
    assert_eq!(all.len(), golden.len());
    for (i, (got, want)) in all.iter().zip(&golden).enumerate() {
        assert_eq!(got, want, "case {i}: the door's descriptor");
    }
}

/// ⭐ The widest rung's copy is a handful of row blocks, not one program op per stick.
#[test]
fn a_prefill_rung_copy_is_a_few_dozen_ops() {
    let e = copy(5, CASES[5]);
    let n = e.ktir.as_ref().expect("a program").func.ops_deep().len();
    assert!(
        n < 200,
        "a [512, 2048] copy is {n} KTIR ops — the per-stick walk was 196,616"
    );
}

/// ⛔ THE WHOLE-STICK LAW IS THE SPLICE'S TYPES. A sub-stick source width, destination width or
/// destination column is refused by its own constructor (`StickExtent`, `ColWindow`), naming which,
/// before any kernel is compiled — the kernel and the door restate no check of their own.
#[test]
fn a_sub_stick_width_or_column_is_refused_naming_which() {
    let refuse = |rows: u32, c_src: u32, c_dst: u32, off: u32| {
        let tr = |t: usize, c0: u32, w: u32| TensorRegion {
            tensor: TensorId::from_index(t),
            region: Region {
                rows: Range::new(0, rows),
                cols: Range::new(c0, w),
            },
        };
        let node: SubtileNode = SubtileNode {
            id: SubtileId::from_index(12),
            op: SubOp::AllReduce,
            inputs: vec![tr(11, 0, c_src)],
            output: tr(12, off, c_src),
        };
        match scratchy_triton_splice::lower_block_copy::<W, _>(
            &node,
            tr(11, 0, c_src),
            tr(12, off, c_src),
            c_dst,
            "bc",
        ) {
            Ok(_) => panic!("[{rows}, {c_src}] at {off} of {c_dst} is not whole sticks"),
            Err(e) => e,
        }
    };
    for (case, what) in [
        ((4, 96, 512, 0), "source width"),
        ((4, 128, 520, 0), "destination width"),
        ((4, 128, 512, 32), "destination column"),
    ] {
        let (r, s, d, o) = case;
        let e = refuse(r, s, d, o);
        assert!(
            e.contains(&format!("the block copy's {what}")),
            "{case:?} names its {what}: {e}"
        );
    }
}

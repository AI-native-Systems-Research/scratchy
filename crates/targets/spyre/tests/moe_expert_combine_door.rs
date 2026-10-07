// SPDX-License-Identifier: Apache-2.0
//! THE EXPERT COMBINE DOOR — the third arm of the gemma-4 MoE SuperDSC port.
//!
//! `Program::ExpertCombine` was refused by name in the door
//! (`ktir_superdsc_door::lower`); this file is the acceptance gate for its
//! lowering, driving the REAL path (`lower_graph_to_superdsc` → the door, with
//! the bundle layout) and pinning what the door now emits:
//!
//! * **The chain shape**: `2k−1` ops for `k` slots — slot 0's `multiply` seeds
//!   the accumulator, every later slot is a `multiply` then an `add`, and the
//!   FINAL add writes the output directly (no trailing copy). For `k=1` the
//!   single `multiply` IS the combine and writes the output.
//! * **The operand addressing**: the multiply legs read the pair rows at
//!   DISTINCT stick-block column offsets (`j·w`, the sort's own leg law), and
//!   the score column rides the `In::col` broadcast — the per-row-scalar
//!   operand mode, pinned here by the chain's survival through the
//!   descriptor assembler (a wrong broadcast mode is a build error in
//!   `assemble_pointwise_broadcast_off`, not a silent mis-read).
//!
//! ⛔ WHY THE DOOR PATH AND NOT THE EMULATOR: same law as the sort/unsort gate
//! (`moe_expert_sort_unsort_door.rs`) — the emulator runs the KTIR ops
//! directly, so an emu-side test cannot see the descriptors. The NUMERIC gate
//! (tiny26 EMU-vs-card tensordump parity, including the fp16-rounding
//! deviation class the door's doc states) is the port's own and runs on the
//! pod; what belongs HERE is the descriptor-level contract.

use scratchy_subtile::subtile_ir::{
    NeoX, SharedExpertBound, SubOp, SubtileId, SubtileIR, SubtileNode, TensorId, TensorRegion,
    TensorShape,
};

const W: u32 = 64;
const K: u32 = 3;
const M: u32 = 5;

/// One `SubOp::ExpertCombine` node: pair rows `[m, k·w]` + scores `[m, k]` →
/// `[m, w]`. The routing indices are NOT an input — the combine reads only
/// the pair rows and the scores (`inputs = [rows, scores]`, the SubOp's own
/// contract).
fn combine_ir() -> SubtileIR<NeoX> {
    let tensors = vec![
        TensorShape { rows: M, cols: K * W }, // t0 pair rows
        TensorShape { rows: M, cols: K },     // t1 scores
        TensorShape { rows: M, cols: W },     // t2 tokens (out)
    ];
    let tr = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let node = SubtileNode {
        id: SubtileId::from_index(0),
        op: SubOp::ExpertCombine {
            hidden: W,
            shared: SharedExpertBound(None),
        },
        inputs: vec![tr(0), tr(1)],
        output: tr(2),
    };
    SubtileIR {
        result: TensorId::from_index(2),
        tensors,
        num_sources: 2,
        nodes: vec![node],
        op_output: Vec::new(),
    }
}

/// Lower through the REAL door path and return the emitted ops.
fn through_the_door(ir: &SubtileIR<NeoX>) -> Vec<ktir_superdsc::emit::EmittedOp> {
    let weight_ids = std::collections::HashSet::new();
    let (ops, _layout) =
        scratchy_target_spyre::lower_subtile_tape_to_ktir::lower_graph_to_superdsc(
            ir,
            &weight_ids,
            ktir_superdsc::ktir_node::ActiveCap::FULL,
            false,
        )
        .unwrap_or_else(|e| panic!("lower through the door: {e:?}"));
    ops
}

/// The emitted `opFuncName`s of a door's ops, in emission order.
fn opfunc_names(ops: &[ktir_superdsc::emit::EmittedOp]) -> Vec<String> {
    ops.iter()
        .map(|o| {
            let dsc = o.dsc();
            let d = dsc
                .dscs_
                .first()
                .and_then(|m| m.values().next())
                .expect("an emitted op has at least one dsc");
            d.computeOp_
                .first()
                .map(|c| c.opFuncName.clone())
                .expect("a dsc has a computeOp")
        })
        .collect()
}

/// The first input operand's per-core start HBM BYTE address on core 0 —
/// where the leg's READ window begins (the sort gate's helper, pointed at
/// `inputLabeledDs[0]` instead of the output).
fn input0_start_bytes(o: &ktir_superdsc::emit::EmittedOp) -> u64 {
    let dsc = o.dsc();
    let d = dsc
        .dscs_
        .first()
        .and_then(|m| m.values().next())
        .expect("an emitted op has at least one dsc");
    let in_lds = d
        .computeOp_
        .first()
        .and_then(|c| c.inputLabeledDs.first())
        .cloned()
        .unwrap_or_default();
    let idx: u32 = in_lds
        .rsplit("idx")
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(u32::MAX);
    let alloc = d
        .scheduleTree_
        .iter()
        .find(|n| n.nodeType_ == "allocate" && n.component_ == "hbm" && n.ldsIdx_ == idx)
        .unwrap_or_else(|| panic!("the input's hbm allocate (lds {idx}, from {in_lds:?})"));
    alloc
        .startAddressCoreCorelet_
        .data_
        .get("[0, 0, 0]")
        .and_then(|s| s.parse().ok())
        .expect("core 0's start address")
}

/// THE CHAIN SHAPE: `2k−1` ops — mul, then (mul, add) per remaining slot, the
/// final add writing the output. Emission order IS the chain order (the bake
/// orders by the emit sequence), so the opFuncName sequence pins both the
/// count and the dependency shape.
#[test]
fn expert_combine_emits_the_fma_chain() {
    let ops = through_the_door(&combine_ir());
    // 2k−1: k multiplies and k−1 adds.
    assert_eq!(
        ops.len(),
        2 * K as usize - 1,
        "k multiplies + k−1 adds for k={K}"
    );
    let mut expected = vec!["mul"];
    for _ in 1..K {
        expected.push("mul");
        expected.push("add");
    }
    let names = opfunc_names(&ops);
    assert_eq!(
        names, expected,
        "the chain order: mul seeds, then (mul, add) per slot"
    );
}

/// ⭐ THE LOAD-BEARING ASSERT: the MULTIPLY legs read the pair rows at
/// DISTINCT stick-block column offsets. Leg `j`'s first input starts at slot
/// `j`'s block of the `[m, k·w]` pair rows — `j·M·W·2` bytes past leg 0 (the
/// same stick-block law the sort's own leg test derives: consecutive slots
/// are `rows·lanes = M·64` elements apart). k muls all reading block 0 is the
/// failure this exists to catch: it bakes clean and every slot scales slot
/// 0's expert output.
#[test]
fn expert_combine_muls_read_distinct_slot_blocks() {
    let ops = through_the_door(&combine_ir());
    let muls: Vec<_> = ops
        .iter()
        .filter(|o| {
            let dsc = o.dsc();
            let d = dsc
                .dscs_
                .first()
                .and_then(|m| m.values().next())
                .expect("an emitted op has at least one dsc");
            d.computeOp_
                .first()
                .map(|c| c.opFuncName == "mul")
                .unwrap_or(false)
        })
        .collect();
    assert_eq!(muls.len(), K as usize, "one multiply per slot");
    let starts: Vec<u64> = muls.iter().map(|o| input0_start_bytes(o)).collect();
    let base = starts[0];
    for j in 0..K as usize {
        assert_eq!(
            starts[j] - base,
            (j as u64) * M as u64 * W as u64 * 2,
            "mul leg {j}'s pair-row start − leg 0's — slot {j}'s column block in the \
             [{M}, {}] pair rows",
            K * W
        );
    }
}

/// ⛔ A SHARED EXPERT IS A BUNDLE-CONTRACT REFUSAL, NOT A SILENT DROP. The
/// emu's combine would add the shared expert's contribution as an extra fma
/// term; the door's chain has no term for it, so a bundle that declares one
/// must fail at the door, naming it — the same law the emu's own doc states
/// ("a bundle that declares one is refused at the loader"). The door re-states
/// the guard because the KTIR program is the last place the bound is visible.
#[test]
fn expert_combine_refuses_a_shared_expert_bound() {
    let mut ir = combine_ir();
    let SubOp::ExpertCombine { shared, .. } = &mut ir.nodes[0].op else {
        panic!("fixture is an ExpertCombine");
    };
    *shared = SharedExpertBound(std::num::NonZeroU32::new(1));
    let weight_ids = std::collections::HashSet::new();
    let err = match scratchy_target_spyre::lower_subtile_tape_to_ktir::lower_graph_to_superdsc(
        &ir,
        &weight_ids,
        ktir_superdsc::ktir_node::ActiveCap::FULL,
        false,
    ) {
        Err(e) => e,
        Ok(_) => panic!("a shared expert bound must be refused, not dropped"),
    };
    let msg = format!("{err:?}");
    assert!(
        msg.contains("shared"),
        "the refusal names the shared expert, got: {msg}"
    );
}

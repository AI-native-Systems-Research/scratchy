// SPDX-License-Identifier: Apache-2.0
//! THE EXPERT SORT/UNSORT DOORS — the first two arms of the gemma-4 MoE SuperDSC port.
//!
//! `Program::ExpertSort` and `Program::ExpertUnsort` were refused by name in the
//! door (`ktir_superdsc_door::lower`); this file is the acceptance gate for their
//! lowerings, driving the REAL path (`lower_graph_to_superdsc` → the door, per
//! program, with the bundle layout) and pinning what the door now emits:
//!
//! * **Sort = `k` `identity` legs.** The gathered regime's pair-row layout
//!   `[m, k·w]` is `k` copies of the `[m, w]` tile at stick-aligned column
//!   offsets — `j·w` for slot `j`, the same column-block law `silumul`'s chunked
//!   wide ops already lock. The test asserts the leg COUNT (`k`), the opFuncName
//!   (`identity`), and — the load-bearing part — that the legs' OUTPUT operands
//!   carry DISTINCT, monotonically increasing offsets, each exactly one slot's
//!   stick-block address apart. A door that emitted `k` legs all writing column
//!   block 0 would bake clean and compute one slot of `k`.
//! * **Unsort = ONE `identity`** over the whole `[m, k·w]` tile, same extent as
//!   its input.
//!
//! ⛔ WHY THE DOOR PATH AND NOT THE EMULATOR: the emulator runs the KTIR ops
//! directly (`ktir_groups` ignores the Program field off-card), so an
//! emu-side test cannot see the descriptors at all. The DOOR is what `-Fspyre-hw`
//! bakes, and this file drives exactly its code path — the same call
//! `ktir_groups_via_superdsc` makes per program. It needs no card and no
//! `SCRATCHY_PLAN_ONLY_BAKE` escape: `lower_graph_to_superdsc` stops at the
//! in-memory descriptors.
//!
//! The NUMERIC gate for these ops (tiny26 EMU-vs-card tensordump parity) is the
//! port's own and runs on the pod; what belongs HERE is the descriptor-level
//! contract, which is the part the emu cannot check.

use scratchy_subtile::subtile_ir::{
    ExpertBundle, NeoX, NumExperts, SubOp, SubtileId, SubtileIR, SubtileNode, TensorId,
    TensorRegion, TensorShape, TopK,
};

/// `w` (the expert width) at a stick-legal size: 64, one whole fp16 stick —
/// the smallest width the pointwise doors accept, so the leg-offset arithmetic
/// is exact and legible in a failure message.
const W: u32 = 64;
const K: u32 = 3;
const M: u32 = 5;

/// One `SubOp::ExpertSort` node: `x [m, w]` + indices `[m, k]` → `[m, k·w]`.
fn sort_ir() -> SubtileIR<NeoX> {
    let tensors = vec![
        TensorShape { rows: M, cols: W },          // t0 x
        TensorShape { rows: M, cols: K },          // t1 indices
        TensorShape { rows: M, cols: K * W },      // t2 pairs (out)
    ];
    let tr = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let node = SubtileNode {
        id: SubtileId::from_index(0),
        op: SubOp::ExpertSort {
            experts: NumExperts::new(std::num::NonZeroU32::new(8).unwrap()),
            k: TopK::new(std::num::NonZeroU32::new(K).unwrap()),
            bundle: ExpertBundle::SwitchGlu,
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

/// One `SubOp::ExpertUnsort` node: rows `[m, k·w]` + routing → `[m, k·w]`.
fn unsort_ir() -> SubtileIR<NeoX> {
    let tensors = vec![
        TensorShape { rows: M, cols: K * W },      // t0 pair rows
        TensorShape { rows: M, cols: K },          // t1 routing (rides along)
        TensorShape { rows: M, cols: K * W },      // t2 tokens (out)
    ];
    let tr = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let node = SubtileNode {
        id: SubtileId::from_index(0),
        op: SubOp::ExpertUnsort,
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

/// The OUTPUT tensor's per-core start HBM BYTE address on core 0 — where the
/// leg's window begins. This is the same field the bake renders
/// (`startAddressCoreCorelet_.data_`), so what this reads is what the card runs.
fn output_start_bytes(o: &ktir_superdsc::emit::EmittedOp) -> u64 {
    let dsc = o.dsc();
    let d = dsc
        .dscs_
        .first()
        .and_then(|m| m.values().next())
        .expect("an emitted op has at least one dsc");
    // The output's allocate node is the one whose `ldsIdx_` matches the
    // computeOp's LAST outputLabeledDs — "Tensor{i}-idx{i}" names it. Walk the
    // hbm allocates and take the one the computeOp names as output.
    let out_lds = d
        .computeOp_
        .first()
        .and_then(|c| c.outputLabeledDs.last())
        .cloned()
        .unwrap_or_default();
    let idx: u32 = out_lds
        .rsplit("idx")
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(u32::MAX);
    let alloc = d
        .scheduleTree_
        .iter()
        .find(|n| n.nodeType_ == "allocate" && n.component_ == "hbm" && n.ldsIdx_ == idx)
        .unwrap_or_else(|| panic!("the output's hbm allocate (lds {idx}, from {out_lds:?})"));
    alloc
        .startAddressCoreCorelet_
        .data_
        .get("[0, 0, 0]")
        .and_then(|s| s.parse().ok())
        .expect("core 0's start address")
}

/// THE SORT: `k` identity legs, distinct stick-block output offsets, one tile apart.
#[test]
fn expert_sort_emits_k_identity_legs_at_slot_offsets() {
    let ops = through_the_door(&sort_ir());
    // The node is ONE program; the door lowers it to K legs. (A fused or split
    // emit that changes the count changes this assert, which is its job.)
    assert_eq!(ops.len(), K as usize, "one identity leg per slot");
    let names = opfunc_names(&ops);
    assert!(
        names.iter().all(|n| n == "identity"),
        "every leg is an identity copy, got {names:?}"
    );
    // ⭐ THE LOAD-BEARING ASSERT: the legs write DISTINCT column blocks. Each
    // leg's OUTPUT allocate carries core 0's start address — the byte address of
    // slot `j`'s block in the `[m, k·w]` stick-blocked layout, plus the output
    // tensor's segment base (the same base for every leg, so it cancels in the
    // DIFFERENCE). The stick-block law (`Nest`'s `(c/lanes)·(rows·lanes)` term)
    // puts consecutive slots `rows·lanes = M·64` elements = `M·W·2` bytes apart,
    // so leg `j`'s start minus leg 0's must be exactly `j·M·W·2`. k legs all
    // writing block 0 is the failure this exists to catch: it bakes clean, and
    // the projections then read one slot of k.
    let starts: Vec<u64> = ops.iter().map(output_start_bytes).collect();
    let base = starts[0];
    for j in 0..K as usize {
        assert_eq!(
            starts[j] - base,
            (j as u64) * M as u64 * W as u64 * 2,
            "leg {j}'s output start − leg 0's — slot {j}'s column block in the [{M}, {}] pair \
             rows",
            K * W
        );
    }
}

/// THE UNSORT: one identity over the whole pair-row tile.
#[test]
fn expert_unsort_emits_one_identity() {
    let ops = through_the_door(&unsort_ir());
    assert_eq!(ops.len(), 1, "the unsort is ONE whole-tile identity");
    let names = opfunc_names(&ops);
    assert_eq!(names, vec!["identity"]);
}

/// ⛔ A SUB-STICK EXPERT WIDTH IS A BUILD ERROR, NOT A MIS-STRIDED COPY. The sort's
/// legs step whole sticks (`j·w` must be a 64-column block corner); a model whose
/// expert width is not a 64-multiple cannot lay its pair rows out at stick
/// granularity, and the door must say so at build time. This is the refusal the
/// narrow-tensor law predicts for any future sub-stick geometry.
#[test]
fn expert_sort_refuses_a_sub_stick_width() {
    let ir: SubtileIR<NeoX> = SubtileIR {
        result: TensorId::from_index(2),
        tensors: vec![
            TensorShape { rows: M, cols: 24 }, // sub-stick width
            TensorShape { rows: M, cols: K },
            TensorShape { rows: M, cols: K * 24 },
        ],
        num_sources: 2,
        nodes: vec![SubtileNode {
            id: SubtileId::from_index(0),
            op: SubOp::ExpertSort {
                experts: NumExperts::new(std::num::NonZeroU32::new(8).unwrap()),
                k: TopK::new(std::num::NonZeroU32::new(K).unwrap()),
                bundle: ExpertBundle::SwitchGlu,
            },
            inputs: vec![
                TensorRegion {
                    tensor: TensorId::from_index(0),
                    region: Region24::whole(0),
                },
                TensorRegion {
                    tensor: TensorId::from_index(1),
                    region: Region24::whole(1),
                },
            ],
            output: TensorRegion {
                tensor: TensorId::from_index(2),
                region: Region24::whole(2),
            },
        }],
        op_output: Vec::new(),
    };
    let weight_ids = std::collections::HashSet::new();
    let err = match scratchy_target_spyre::lower_subtile_tape_to_ktir::lower_graph_to_superdsc(
        &ir,
        &weight_ids,
        ktir_superdsc::ktir_node::ActiveCap::FULL,
        false,
    ) {
        Err(e) => e,
        Ok(_) => panic!("a sub-stick expert width must be refused"),
    };
    let msg = format!("{err:?}");
    assert!(
        msg.contains("not a multiple of") && msg.contains("stick"),
        "the refusal names the stick law, got: {msg}"
    );
}

/// A tiny helper so the refusal fixture above can build `whole()` regions over
/// its own tensor list without duplicating the closure three times.
struct Region24;
impl Region24 {
    fn whole(t: usize) -> scratchy_subtile::subtile_ir::Region {
        let cols = [24u32, K, K * 24][t];
        scratchy_subtile::subtile_ir::Region {
            rows: scratchy_subtile::subtile_ir::Range::new(0, M),
            cols: scratchy_subtile::subtile_ir::Range::new(0, cols),
        }
    }
}

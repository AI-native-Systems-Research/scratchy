// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ AN EMBEDDING-TABLE GATHER IS THE VENDOR'S OWN DECLARATION AT FULL ROW WIDTH.
//!
//! `table[ids[row], :]` is a row lookup: one index entry IS one table row. That is
//! `entry_dim: KernelAxis::Batch` with `PageExtent::single_position()` and no second paged axis —
//! the SAME pair `zz_the_gather_op_is_kernel_less.rs` names `VENDOR_DECL`, because
//! `dxp/test/test_gather_1core/sdsc_1.json`'s value tensor is `["mb","out","x"]` with
//! `maxDimSizes_ [1,-1,-1]` and its op is an `identity`, not an attention. So the marking a table
//! lookup needs is not an extension of the paged-KV gather; it is the narrower of the two cases the
//! one door already spans.
//!
//! ## What this file adds over that one
//! The GEOMETRY. `gather_copy_opspec` refuses `out != 64` because ITS destination must be contiguous
//! for a matmul kernel to read per-batch at a dxp-derived stride. An embedding's destination is an
//! ordinary activation — written stick-major, read stick-major — and its row is `d_model` wide:
//! granite's is 4096, i.e. 64 sticks. These assertions are at that width, against the descriptor the
//! pod bakes, so "the pin still means one row at 64 sticks" is measured rather than reasoned.
//!
//! ## ⛔ AND THE TWO FIELDS THAT ARE **NOT** THE SAME AS THE KV GATHER'S
//! * `primaryDsInfo_` is `{KERNEL_IDX, OUTPUT}` with NO `KERNEL` — `pointwise_broadcast_opspec_from_tile`
//!   types every operand `Role::Output`, which is what lets `emit_sdsc`'s KERNEL refusal pass. A
//!   pointwise gather is legal for exactly the reason a gathered matmul is not.
//! * The op has THREE data operands, not two, so the index's position is load-bearing in a way the
//!   2-operand fixtures cannot show. See `the_gathered_operand_must_be_the_last_input`.

use ktir_superdsc::emit as superdsc;
use ktir_superdsc::emit::{In, PointwiseGather, assemble_pointwise_broadcast_gather};
use ktir_superdsc::ir::island::tile_op::{TileOp, TileOpKind};
use ktir_superdsc::sdsc_abstract::{RowBlockedTag, StickLayout, Stk};
use ktir_superdsc::superdsc_opspec::{Df, ItDim};

/// Granite's embedding geometry, read off the descriptor the pod bakes
/// (`/work/bake_revendor/embedding_granite`, `N_ { out_: 4096, mb_: 256, y_: 1 }`).
const ROWS: u32 = 256;
const COLS: u32 = 4096;
/// `COLS` in fp16 sticks — the number `gather_copy_opspec` would refuse, stated so the refusal this
/// file is the counter-case to is a NUMBER here rather than a remark.
const COLS_STICKS: u32 = COLS / 64;

fn dim(name: &'static str, size: u32, is_stick: bool) -> ItDim {
    ItDim {
        name,
        size,
        is_reduction: false,
        is_stick,
        df: Df::Fp16,
    }
}

fn tile(rows: u32, cols: u32) -> TileOp {
    TileOp {
        // The scalar rides as an operand, so three live tensors: scalar, table, output.
        kind: TileOpKind::PointwiseOrReduce { n_operands: 3 },
        dims: vec![
            dim("mb", rows, false),
            dim("out", cols, true),
            dim("y", 1, false),
        ],
        df: Df::Fp16,
    }
}

fn h(name: &str, rows: u32, cols: u32) -> Stk<RowBlockedTag> {
    Stk::new(name, StickLayout::row_blocked(rows as usize, cols as usize)).expect("row-blocked")
}

/// The embedding op exactly as `scalarmul`'s gathered branch builds it: `[scalar, table]` with the
/// TABLE LAST, `multiply`, a row-blocked output.
fn emit(rows: u32, cols: u32) -> Result<serde_json::Value, String> {
    let table = h("EmbTable", rows, cols);
    let scale = h("EmbScale", rows, cols);
    let out = h("EmbOut", rows, cols);
    let t = tile(rows, cols);
    let inputs = [In::scalar(&scale).ew(), In::full(&table).ew()];
    let mut sym = 0i64;
    let e = assemble_pointwise_broadcast_gather(
        PointwiseGather {
            op_name: "scalarmul_o2",
            tile_op: &t,
            op_func: "multiply",
            rows,
            cols,
            inputs: &inputs,
            gathered_input: 1,
            index_name: "EmbIds",
            o: &out,
        },
        &mut sym,
        None,
    )
    .map_err(|e| e.0)?;
    // `EmittedOp` carries the descriptor in `op`; a KTIR-only op would have `None` there, and this one
    // is not that, so an absent descriptor is a failure rather than something to tolerate.
    let d =
        e.op.as_ref()
            .expect("a gathered pointwise has a descriptor");
    Ok(serde_json::to_value(d).expect("serializes"))
}

fn body(j: &serde_json::Value) -> &serde_json::Value {
    &j["dscs_"][0]["scalarmul_o2"]
}

fn alloc(j: &serde_json::Value, i: usize) -> &serde_json::Value {
    &body(j)["scheduleTree_"][i]
}

/// ⭐⭐⭐ THE PIN IS `[1, -1]` AT 64 STICKS — one entry is ONE ROW, whatever the row's width.
///
/// This is the whole claim. `getPageSize` erases the negative entries, so the pinned dim IS the paged
/// dim; `getBufferCapacityForNodePerDim` clamps it to the pin and multiplies the rest, so
/// `skip_addr = 1 x cols` = one table row. A `-1` there would make one entry the op's whole `mb` tile,
/// i.e. 256x too far for every index.
#[test]
fn the_value_tensor_pins_its_row_axis_to_one_position() {
    let j = emit(ROWS, COLS).expect("the embedding op emits");
    assert_eq!(
        COLS_STICKS, 64,
        "the width this file exists to cover is 64 sticks; `gather_copy_opspec` refuses anything but 1"
    );
    // The table is input 1, so lds 1 — `[scalar, table, index, output]`.
    let v = alloc(&j, 1);
    assert_eq!(
        v["indirectAllocType_"], "value_tensor",
        "the gathered table must be the value side of the pair"
    );
    assert_eq!(
        v["layoutDimOrder_"],
        serde_json::json!(["mb", "out"]),
        "a row-blocked activation is `[mb, out]`, and the pin below is resolved against THIS list"
    );
    assert_eq!(
        v["maxDimSizes_"],
        serde_json::json!([1, -1]),
        "`mb` pinned to ONE POSITION and `out` left whole — the vendor's `[1,-1,-1]` at this rank. A \
         pin on `out` instead would make an entry a single element; no pin at all would make it the \
         whole tile."
    );
}

/// ⭐ THE INDEX IS THE VENDOR'S OWN OPERAND: `KERNEL_IDX`, `SENUINT32`, 4 bytes, HBM-ONLY, rank 1
/// over the pinned dim. Every one of those is `attach_gather_index`'s, not this caller's.
#[test]
fn the_index_operand_is_the_vendors_own() {
    let j = emit(ROWS, COLS).expect("emits");
    let lds: Vec<&serde_json::Value> = body(&j)["labeledDs_"]
        .as_array()
        .expect("a list")
        .iter()
        .collect();
    let idx = lds
        .iter()
        .find(|l| l["dsType_"] == "KERNEL_IDX")
        .expect("exactly one KERNEL_IDX operand");
    assert_eq!(idx["dataFormat_"], "SENUINT32");
    assert_eq!(idx["wordLength"], 4, "SENUINT32 is 4 bytes");
    let mem = idx["memOrg_"].as_object().expect("a memOrg");
    assert!(
        mem.contains_key("hbm") && !mem.contains_key("lx"),
        "an index is read into the IBR, never staged — `allocAllMem` gives it no LX chunk, and \
         declaring one leaves `calculateFlopPerByte` a null node (L3DlOpsScheduler.cpp:2337). Got {mem:?}"
    );
    // The index sits immediately before the output: `[scalar, table, index, output]`.
    let n = alloc(&j, 2);
    assert_eq!(n["indirectAllocType_"], "index_tensor");
    assert_eq!(
        n["indexTensorType_"], "index",
        "only the INDEX side is typed"
    );
    assert_eq!(
        n["layoutDimOrder_"],
        serde_json::json!(["mb"]),
        "the index's dims are the value's PINNED dims — one pin, so rank 1"
    );
    // The pair CROSS-LINKS: neither node alone is a gather.
    assert_eq!(n["relatedIndirectAccessAlloc_"], alloc(&j, 1)["name_"]);
    assert_eq!(alloc(&j, 1)["relatedIndirectAccessAlloc_"], n["name_"]);
    assert_eq!(
        body(&j)["computeOp_"][0]["indirectAccessIndexLabeledDs"],
        serde_json::json!(["Tensor2-idx2"]),
        "the op names the INDEX side, at the index's own operand position"
    );
}

/// ⛔⛔⛔ NO `KERNEL` IN `primaryDsInfo_`, WHICH IS WHY A POINTWISE GATHER IS SCHEDULABLE AT ALL.
///
/// `hasDimensionReuse` is `primaryDsInfo_.size() > 1 && count(KERNEL)`, and a reuse op runs the
/// arithmetic-intensity explorer that demands an LX allocate node for every HBM-pinned labeledDs —
/// which an index never gets. `emit_sdsc` refuses a gather on a KERNEL-bearing op outright, so this
/// asserts the reason this op is NOT refused rather than merely that it emitted.
#[test]
fn a_pointwise_gather_carries_no_kernel() {
    let j = emit(ROWS, COLS).expect("emits");
    let p = body(&j)["primaryDsInfo_"].as_object().expect("a map");
    assert!(
        !p.contains_key("KERNEL"),
        "every operand of a pointwise is typed OUTPUT, so a gather is schedulable on it. Got {:?}",
        p.keys().collect::<Vec<_>>()
    );
    assert!(
        p.contains_key("KERNEL_IDX"),
        "the index gets its OWN role, or its layout is dropped"
    );
    assert_eq!(
        body(&j)["computeOp_"][0]["inputLabeledDs"],
        serde_json::json!(["Tensor0-idx0", "Tensor1-idx1"]),
        "the index is NOT an arithmetic input — it appears in exactly one list and this is not it"
    );
}

/// ⛔ THE GATHERED OPERAND MUST BE THE **LAST** INPUT, and a 3-operand op is what makes that visible.
///
/// `attach_gather_index` inserts the index immediately before the OUTPUT; dbo needs it NEXT TO the
/// tensor it indexes (`operand #1 does not dominate this use`, `DSC2ToDataflowIR.cpp:51`). Those are
/// the same slot only for the last input. With two inputs there are two candidate positions and this
/// refusal is the one that keeps us in the vendor's.
///
/// ⛔ ONE-SIDED ON PURPOSE: the passing arm is every other test in this file, so this asserts only the
/// refusal. A swapped-arm version would be covered by whichever arm happened to run.
#[test]
fn the_gathered_operand_must_be_the_last_input() {
    let table = h("EmbTable", ROWS, COLS);
    let scale = h("EmbScale", ROWS, COLS);
    let out = h("EmbOut", ROWS, COLS);
    let t = tile(ROWS, COLS);
    // The UNGATHERED order — table first. Gathering input 0 leaves the index at position 2, two
    // operands away from the table at 0.
    let inputs = [In::full(&table).ew(), In::scalar(&scale).ew()];
    let mut sym = 0i64;
    // `EmittedOp` is not `Debug`, so `expect_err` cannot be used — matched instead, which also names
    // what an unexpected success WOULD be.
    let msg = match assemble_pointwise_broadcast_gather(
        PointwiseGather {
            op_name: "scalarmul_o2",
            tile_op: &t,
            op_func: "multiply",
            rows: ROWS,
            cols: COLS,
            inputs: &inputs,
            gathered_input: 0,
            index_name: "EmbIds",
            o: &out,
        },
        &mut sym,
        None,
    ) {
        Err(e) => e.0,
        Ok(_) => panic!(
            "a gather on input 0 of 2 EMITTED — the index would sit two operands from the table it \
             indexes, which is the dominance failure this refusal exists for"
        ),
    };
    assert!(
        msg.contains("dominate"),
        "the refusal must name the dbo rule it protects, not merely decline. Got: {msg}"
    );
}

/// ⛔⛔⛔ ONLY THE ENTRY AXIS MAY CARRY A SPLIT — the half of `gather_copy_opspec`'s reasoning that
/// transfers to any gather.
///
/// `skip_addr` is `page x the PER-CORE extents of the unpinned dims`, so a split `out` divides the
/// entry. At one row there are no `mb` positions to spend cores on, so `distribute_cores` must take
/// them from `out` — and that is the case this refuses.
///
/// ⛔ ONE-SIDED, like the one above: `ROWS = 256` is the passing arm and it is every other test here.
#[test]
fn a_split_outside_the_entry_axis_is_refused() {
    let msg = emit(1, COLS).expect_err("a 1-row lookup must be refused, not emitted");
    assert!(
        msg.contains("skip_addr") && msg.contains("out"),
        "the refusal must name the derived quantity and the axis. Got: {msg}"
    );
}

/// ⭐ THE DESCRIPTOR, WRITTEN OUT for the pod's `dxp_standalone` acceptance run — the same
/// `{name: dsc}` envelope `/work/bake_revendor/embedding_granite/sdsc_0.json` carries, so the bundle
/// dir's `bundle.mlir` (a stub that names only the filename) needs no change.
///
/// A test rather than a binary because the emission it dumps is the emission the assertions above
/// check: one build, one artifact, no second path that could drift from it.
#[test]
fn dump_the_descriptor_for_the_pod_bake() {
    let j = emit(ROWS, COLS).expect("emits");
    let wrapped = serde_json::json!({ "0_scalarmul_o2": j });
    let p = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("embedding_gather_sdsc_0.json");
    std::fs::write(&p, serde_json::to_string_pretty(&wrapped).expect("json"))
        .unwrap_or_else(|e| panic!("writing {}: {e}", p.display()));
    eprintln!("WROTE {}", p.display());
}

/// ⭐ AND THE UNGATHERED OP AT THE SAME GEOMETRY DECLARES **NO INDIRECTION ANYWHERE** — the control
/// for "this PR moves no non-gathering op".
///
/// Emitted through the sibling assembler the ungathered `scalarmul` branch uses, so it is the shipped
/// path rather than a stand-in.
#[test]
fn the_same_op_without_a_gather_declares_none() {
    let table = h("EmbTable", ROWS, COLS);
    let scale = h("EmbScale", ROWS, COLS);
    let out = h("EmbOut", ROWS, COLS);
    let t = tile(ROWS, COLS);
    let inputs = [In::full(&table).ew(), In::scalar(&scale).ew()];
    let op = ktir_superdsc::ir::bridge::tiled_op_sdsc_op::pointwise_broadcast_opspec_from_tile(
        &t,
        ROWS,
        COLS,
        superdsc::op_func_from_str("multiply"),
        &inputs,
        out.name(),
        0,
        false,
    )
    .expect("the ungathered op builds");
    assert!(
        op.indirect.is_none(),
        "no declaration without a call to the door"
    );
    let folds = ktir_superdsc::superdsc_opspec::SdscFoldSet::new(op.iter.cores_used());
    let e = superdsc::emit_sdsc("scalarmul_o2", &op, &folds, None).expect("emits");
    let j = serde_json::to_value(&e).unwrap();
    for i in 0..op.args.len() {
        assert_eq!(
            alloc(&j, i)["indirectAllocType_"],
            "no_indirection",
            "operand {i} must stay directly addressed"
        );
    }
    assert!(
        body(&j)["computeOp_"][0]["indirectAccessIndexLabeledDs"]
            .as_array()
            .is_none_or(|a| a.is_empty()),
        "an op that declares no gather names no index"
    );
}

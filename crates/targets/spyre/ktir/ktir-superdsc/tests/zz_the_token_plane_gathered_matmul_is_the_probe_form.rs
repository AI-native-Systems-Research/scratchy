// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ THE TOKEN-PLANE GATHERED MATMUL IS THE PROBE'S FORM — the prefill-m>1 per-expert batch
//! matmul's descriptor, pinned field by field against the JSON the one-matmul card probe actually
//! baked (2026-10-09, `/tmp/prefill-m1-probe-report.md`, artifacts `/tmp/gprobe-emit/fused/`).
//!
//! ## Why this file exists
//! For a year the emitter refused EVERY gather on a KERNEL-bearing op, on two bracketing card
//! refusals (`calculateFlopPerByte`, L3DlOpsScheduler.cpp:2334/:2337). The probe root-caused those
//! refusals to the CHUNK-PARAM EXPLORATION path only — which demands transfer users on the index's
//! HBM alloc that only a pass AFTER `addChunkDataStage` creates (a vendor ordering gap) — and
//! escaped them: with `DISABLE_ABOVE_LX_CHUNK_EXPLORE=1` at bake and the index spelled
//! `indexTensorType_ "index"` (the wall-#8 law — the vendor's idx2addr conversion owns the
//! address arithmetic; see the spelling assertion below), the token-plane form schedules
//! END TO END (`RC=0` at the small AND 26b-gate geometries, `init_binary.bin` emitted, the
//! gather ENGAGED: kernel transfer
//! `srcIndirect=YES`, index staged to the L3LU IBR, the vendor's own paged loop nest).
//!
//! The probe validated the descriptor by transplanting gather fields onto our base emission with
//! JSON surgery. This file is the emission-side landing: `GatherIndex::of_expert_planes` through
//! `matmul_opspec_off_operands_phys_gathered` now produces that spelling NATIVELY, and these tests
//! pin it so it cannot drift. What is still NOT landed is the door's per-slot index MINT (a
//! lane-granular transpose of the router's `[m, w_pad]` table — see the ExpertMatmul door's m>1
//! refusal); this file pins the descriptor the mint will feed.
//!
//! ## The pinned spelling (gate26 geometry: m=96 tokens, in=2816, out=64, 32 cores)
//! ```text
//! labeledDs_ (array position == ldsIdx_):
//!   0 INPUT      Tensor0  SEN143_FP8   wl 1  scale [1,1]     memOrg hbm+lx
//!   1 KERNEL     Tensor1  SEN143_FP8   wl 1  scale [1,1,1]   memOrg hbm+lx   <- RANK-3 scale
//!   2 KERNEL_IDX Tensor2  SENUINT32    wl 4  scale [1]       memOrg hbm-ONLY
//!   3 OUTPUT     Tensor3  SEN169_FP16  wl 2  scale [1,1]     memOrg hbm+lx
//! primaryDsInfo_:
//!   KERNEL     layoutDimOrder_ ["mb","in","out"]  stickDimOrder_ ["in","out"]  stickSize_ [2,64]
//!   KERNEL_IDX layoutDimOrder_ ["mb"]             stickDimOrder_ ["mb"]        stickSize_ [32]
//! scheduleTree_:
//!   allocate-Tensor1_hbm  ldo ["mb","in","out"]  mds [1,-1,-1]  value_tensor  sym 0  -> Tensor2
//!   allocate-Tensor2_hbm  ldo ["mb"]             mds [96]       index_tensor  "index"     -> Tensor1
//! computeOp_: inputs [Tensor0, Tensor1] (the index is NOT an input), output [Tensor3-idx3],
//!   indirectAccessIndexLabeledDs [Tensor2-idx2], opFuncName batchmatmulfp8.
//! ```

use ktir_superdsc::emit as superdsc;
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::matmul::{
    SharedKernelBmmForm, matmul_opspec_off_operands_phys_gathered,
};
use ktir_superdsc::sdsc_abstract::{MatK, MatM, MatN, MatY, QueryRowCount};
use ktir_superdsc::superdsc_opspec::{
    EntryBase, Fp8, Fp16, GatherIndex, KernelAxis, PageExtent, SdscFoldSet,
};

/// The probe's gate26 matmul — the 26b GATE projection's per-(slot, chunk) shape — with the gather
/// attached in the token-plane form. `o_name` is `Tensor3` and the index is `Tensor2` so the four
/// `dsName_`s land exactly where the probe's post-surgery JSON has them. The W8A8 law is the door's
/// own two steps: `::<Fp16>` geometry, then paint every NON-OUTPUT operand `Df::Fp8` (the door's
/// section-2 loop verbatim — the paint is what selects `batchmatmulfp8`, whose opFuncName reads the
/// first non-output operand's df).
fn probe_matmul(gather: Option<GatherIndex>) -> ktir_superdsc::superdsc_opspec::OpSpec {
    let mut op = matmul_opspec_off_operands_phys_gathered::<Fp16>(
        MatM::of_token_rows(96),
        MatN::of_out_features(64),
        MatK::of_in_features(2816),
        MatY::unbatched(),
        SharedKernelBmmForm::of_attn_rows(false, QueryRowCount::of_mq(1)),
        "Tensor0",
        "Tensor1",
        "Tensor3",
        0,
        0,
        0,
        // The W8A8 law: fp16 stick geometry, 1-byte operands — the door's own spelling.
        <Fp8 as ktir_superdsc::superdsc_opspec::DataFormat>::DF,
        None,
        gather,
    )
    .expect("the probe's matmul builds");
    // ⛔ AND THE PAINT SKIPS THE INDEX — a law the m>1 door will need too: `attach_gather_index`
    // forces SENUINT32 on the index whatever the value operands are (the DT_CHECK pair
    // GatherIndexConversion.cpp:133 + L3DlOpsScheduler.cpp:5928), so a W8A8 paint that touches it
    // is a stick-extent refusal at best and a wrong-dtype index at worst.
    for arg in &mut op.args {
        if !matches!(
            arg.view().role,
            ktir_superdsc::superdsc_opspec::Role::Output
                | ktir_superdsc::superdsc_opspec::Role::Index
        ) {
            arg.set_df(<Fp8 as ktir_superdsc::superdsc_opspec::DataFormat>::DF);
        }
    }
    op
}

fn token_plane() -> GatherIndex {
    GatherIndex {
        name: "Tensor2".to_string(),
        entry_dim: KernelAxis::Batch,
        page: PageExtent::single_position(),
        per_position: None,
        first_entry: EntryBase::ZERO,
    }
}

fn emit(op: &ktir_superdsc::superdsc_opspec::OpSpec) -> serde_json::Value {
    let folds = SdscFoldSet::new(op.iter.cores_used());
    let (e, _) = superdsc::emit_sdsc("MatMul_0", op, &folds, None)
        .expect("the token-plane gathered matmul emits");
    serde_json::to_value(&e).unwrap()
}

fn dsc(j: &serde_json::Value) -> &serde_json::Value {
    &j["dscs_"][0]["MatMul_0"]
}

fn lds<'a>(j: &'a serde_json::Value, ds_type: &str) -> &'a serde_json::Value {
    j["labeledDs_"]
        .as_array()
        .expect("labeledDs_")
        .iter()
        .find(|l| l["dsType_"] == ds_type)
        .unwrap_or_else(|| panic!("a {ds_type} labeledDs"))
}

fn alloc<'a>(j: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    j["scheduleTree_"]
        .as_array()
        .expect("scheduleTree_")
        .iter()
        .find(|n| n["name_"] == name)
        .unwrap_or_else(|| panic!("alloc node {name}"))
}

/// ⭐⭐⭐⭐⭐ THE HEADLINE: the token-plane form EMITS — the refusal that stood for a year is
/// narrowed to the shapes that were never probed — and the emission is the probe's own spelling.
#[test]
fn the_token_plane_form_emits_the_probe_spelling() {
    let j = emit(&probe_matmul(Some(token_plane())));
    let d = dsc(&j);

    // The op is the probe's: 32 cores over 96 token rows, batchmatmulfp8 on the pt.
    assert_eq!(d["numCoresUsed_"], 32, "the probe's core count");
    assert_eq!(d["N_"]["mb_"], 96, "the probe's token count");
    assert_eq!(d["N_"]["in_"], 2816, "the 26b gate projection's K");
    assert_eq!(d["N_"]["out_"], 64, "the chunk's N");
    assert_eq!(d["computeOp_"][0]["opFuncName"], "batchmatmulfp8");
    assert_eq!(d["computeOp_"][0]["exUnit"], "pt");

    // ⛔ TENSOR ORDER [INPUT, KERNEL, INDEX, OUTPUT] WITH ARRAY POSITION == ldsIdx_ — dxp reads the
    // operand list POSITIONALLY (getMinParamBmm front/.at(1)/.back), so the index must sit at 2
    // with the output slid to 3.
    let order: Vec<(u64, &str, &str)> = d["labeledDs_"]
        .as_array()
        .expect("labeledDs_")
        .iter()
        .map(|l| {
            (
                l["ldsIdx_"].as_u64().expect("ldsIdx_"),
                l["dsName_"].as_str().expect("dsName_"),
                l["dsType_"].as_str().expect("dsType_"),
            )
        })
        .collect();
    assert_eq!(
        order,
        vec![
            (0, "Tensor0", "INPUT"),
            (1, "Tensor1", "KERNEL"),
            (2, "Tensor2", "KERNEL_IDX"),
            (3, "Tensor3", "OUTPUT"),
        ],
        "the probe's tensor order — the index before the output, positions == ldsIdx_"
    );

    // The KERNEL is RANK-3 with a 3-entry scale — the token-plane re-declaration.
    let kern = lds(d, "KERNEL");
    assert_eq!(kern["scale_"], serde_json::json!([1, 1, 1]));
    assert_eq!(kern["dataFormat_"], "SEN143_FP8");
    assert_eq!(kern["wordLength"], 1);
    // And the index is the vendor's dtype at 4 bytes, HBM-ONLY (no lx key: an index is read into
    // the IBR, never staged).
    let idx = lds(d, "KERNEL_IDX");
    assert_eq!(idx["dataFormat_"], "SENUINT32");
    assert_eq!(idx["wordLength"], 4);
    assert_eq!(idx["scale_"], serde_json::json!([1]));
    assert_eq!(
        idx["memOrg_"],
        serde_json::json!({"hbm": {"isPresent": 1}}),
        "the index is HBM-only — an LX entry here is the :2337 refusal"
    );

    // primaryDsInfo_: the kernel's rank-3 PINNED-DIM-FIRST layout with the fp8 2-D stick; the
    // index's own rank-1 layout and 32-entry u32 stick.
    assert_eq!(
        d["primaryDsInfo_"]["KERNEL"]["layoutDimOrder_"],
        serde_json::json!(["mb", "in", "out"]),
        "pinned dim FIRST — not the KV arm's [in,out,mb]; the mds pin lands on position 0"
    );
    assert_eq!(
        d["primaryDsInfo_"]["KERNEL"]["stickDimOrder_"],
        serde_json::json!(["in", "out"])
    );
    assert_eq!(
        d["primaryDsInfo_"]["KERNEL"]["stickSize_"],
        serde_json::json!([2, 64])
    );
    assert_eq!(
        d["primaryDsInfo_"]["KERNEL_IDX"]["layoutDimOrder_"],
        serde_json::json!(["mb"])
    );
    assert_eq!(
        d["primaryDsInfo_"]["KERNEL_IDX"]["stickDimOrder_"],
        serde_json::json!(["mb"])
    );
    assert_eq!(
        d["primaryDsInfo_"]["KERNEL_IDX"]["stickSize_"],
        serde_json::json!([32])
    );

    // The value side: rank-3, mb PINNED AT ONE POSITION (mds [1,-1,-1] — one expert plane per
    // entry), concrete start (NOT symbolic), cross-linked to the index.
    let kern_alloc = alloc(d, "allocate-Tensor1_hbm");
    assert_eq!(
        kern_alloc["layoutDimOrder_"],
        serde_json::json!(["mb", "in", "out"])
    );
    assert_eq!(
        kern_alloc["maxDimSizes_"],
        serde_json::json!([1, -1, -1]),
        "mb pinned to ONE position — the page; in/out derived per core"
    );
    assert_eq!(kern_alloc["indirectAllocType_"], "value_tensor");
    assert_eq!(
        kern_alloc["relatedIndirectAccessAlloc_"], "allocate-Tensor2_hbm",
        "the value node names the index node"
    );
    assert_eq!(
        kern_alloc["isStartAddrSymbolic_"], 0,
        "a concrete per-core start — symbolic is the VariableDefinition.cpp:303 symbol-id refusal"
    );

    // The index side: the WHOLE 96-entry table (positive mds marks mb paged for
    // getPagedDimensions), and the spelling the probe proved LOAD-BEARING.
    let idx_alloc = alloc(d, "allocate-Tensor2_hbm");
    assert_eq!(idx_alloc["layoutDimOrder_"], serde_json::json!(["mb"]));
    assert_eq!(
        idx_alloc["maxDimSizes_"],
        serde_json::json!([96]),
        "the whole table — of_index_entries divides by the pin (1) and removes the mb split"
    );
    assert_eq!(idx_alloc["indirectAllocType_"], "index_tensor");
    assert_eq!(
        idx_alloc["relatedIndirectAccessAlloc_"], "allocate-Tensor1_hbm",
        "and the index node names the value node — the link is mutual"
    );
    // ⛔ "index", on EVERY path — the 2026-10-09 wall-#8 law. An "address"-typed index means the
    // buffer already holds page addresses and the value start is only a within-page offset
    // (GatherBuffers.cpp:198-208 zeroes it after the vendor's own conversion); spelling it while
    // the mint writes RAW EXPERT IDS leaves the ABSOLUTE bank base in the value start, which dcc
    // routes into the EAR-domain mutable constant (mod-2^31 wrap) — at 26b, chunk c10's base
    // 17,460,359,424 refused with MutableAddrSplitting.cpp:814 EAR overflow (range 2^28 fp8
    // elements). "index" engages GatherIndexConversion: INT32IDXTOADDR computes base + e·skip in
    // the integer domain (the fp32 mint cannot — the base needs 26+ bits), zeroes the value start,
    // and flips the type to ADDRESS itself (GatherBuffers.cpp:199) before the scheduler's reuse
    // path reads it. The pre-wall-2/3 mint form DID die at :1517/:1535 with "index"; the current
    // form bakes RC=0 with the conversion engaged (init_binary +185,856 B over 128 gathered sdscs).
    assert_eq!(
        idx_alloc["indexTensorType_"], "index",
        "the wall-#8 spelling: the vendor's idx2addr conversion owns the address arithmetic"
    );

    // The compute op names the index in exactly one place, and it is not the input list.
    let c = &d["computeOp_"][0];
    assert_eq!(
        c["inputLabeledDs"],
        serde_json::json!(["Tensor0-idx0", "Tensor1-idx1"]),
        "the index is NOT an arithmetic input — one more operand than the DDL declares"
    );
    assert_eq!(c["outputLabeledDs"], serde_json::json!(["Tensor3-idx3"]));
    assert_eq!(
        c["indirectAccessIndexLabeledDs"],
        serde_json::json!(["Tensor2-idx2"]),
        "the annotation is what makes the op a gather rather than an inert extra operand"
    );
}

/// ⛔ THE ADMISSION IS ONE SHAPE, NOT A DOOR OPENED FOR EVERY GATHERED KERNEL. A token-plane
/// declaration whose pin is NOT one position (a multi-position page on `mb`) was never probed, and
/// an unprobed descriptor that reaches dxp re-runs the card loop the guard exists to close.
#[test]
fn a_token_plane_pin_above_one_position_stays_refused() {
    let gather = GatherIndex {
        name: "Tensor2".to_string(),
        entry_dim: KernelAxis::Batch,
        page: PageExtent::of_positions(2),
        per_position: None,
        first_entry: EntryBase::ZERO,
    };
    let op = probe_matmul(Some(gather));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    let err = match superdsc::emit_sdsc("MatMul_0", &op, &folds, None) {
        Err(e) => e.0,
        Ok(_) => panic!(
            "a pin of TWO positions on the entry axis EMITTED. The probe validated exactly one \
             position per entry (mds [1,-1,-1]); anything else is an unprobed shape and must be a \
             build error, not a descriptor that reaches dxp"
        ),
    };
    assert!(
        err.contains("KERNEL"),
        "the refusal must name the KERNEL path (got: {err})"
    );
    assert!(
        err.contains("TOKEN-PLANE") || err.contains("token-plane"),
        "the refusal must name the one shape that IS admitted (got: {err})"
    );
}

/// ⛔ skip_addr IS THE PIN × THE PER-CORE EXTENTS OF THE UNPINNED DIMS — so a plan that splits an
/// unpinned axis of the gathered kernel shrinks every index step below one expert plane: wrong
/// addresses from a clean bake. The probe's plan split `mb` only ({in:1, mb:32, out:1}); a K-split
/// under a gather is a build error.
#[test]
fn a_split_unpinned_axis_of_the_kernel_is_refused() {
    let mut op = probe_matmul(Some(token_plane()));
    // Re-divide the SAME dims with a plan that splits the K axis — the shape a K-split matmul
    // would naturally produce, and the one skip_addr cannot survive.
    let dims: Vec<ktir_superdsc::superdsc_opspec::ItDim> = op.iter.dims().to_vec();
    op.iter = ktir_superdsc::superdsc_opspec::WorkPlan::divide(
        &dims,
        ktir_superdsc::superdsc_opspec::MaxCores::<32>,
        |_, _| {
            let mut m = std::collections::BTreeMap::new();
            m.insert("in", 2u32);
            m
        },
    )
    .expect("a plan that splits the K axis builds");
    let folds = SdscFoldSet::new(op.iter.cores_used());
    let err = match superdsc::emit_sdsc("MatMul_0", &op, &folds, None) {
        Err(e) => e.0,
        Ok(_) => panic!(
            "a gathered kernel whose op splits the UNPINNED `in` axis EMITTED. skip_addr is the \
             pin × the per-core extents of the unpinned dims, so every index entry would step \
             half an expert plane — this must be a build error, not a silently-short gather"
        ),
    };
    assert!(
        err.contains("skip_addr") && err.contains("'in'"),
        "the refusal must name skip_addr and the split axis (got: {err})"
    );
}

/// ⭐ THE UNGATHERED MATMUL IS BYTE-IDENTICAL — the same builder with `None` emits the 2-D kernel
/// it always did, which is what every shipped bundle goes through.
#[test]
fn the_ungathered_form_is_unchanged() {
    let j = emit(&probe_matmul(None));
    let d = dsc(&j);
    assert_eq!(
        d["primaryDsInfo_"]["KERNEL"]["layoutDimOrder_"],
        serde_json::json!(["in", "out"]),
        "no gather, no rank-3 re-declaration — the bare 2-D shared weight"
    );
    let names: Vec<&str> = d["labeledDs_"]
        .as_array()
        .expect("labeledDs_")
        .iter()
        .map(|l| l["dsType_"].as_str().expect("dsType_"))
        .collect();
    assert_eq!(names, vec!["INPUT", "KERNEL", "OUTPUT"], "no index operand");
    assert!(
        d["scheduleTree_"]
            .as_array()
            .expect("scheduleTree_")
            .iter()
            .all(|n| n["indirectAllocType_"] == "no_indirection"),
        "every node unremarkable — byte-identical to before the gather existed"
    );
    // And the Df of the operands is the W8A8 law's, unchanged by the gather plumbing.
    assert_eq!(lds(d, "KERNEL")["dataFormat_"], "SEN143_FP8");
    assert_eq!(lds(d, "OUTPUT")["dataFormat_"], "SEN169_FP16");
}

/// ⭐ THE INDEX'S OWN PLAN: `of_index_entries` divides by the pin and REMOVES the mb split — the
/// whole table per core, which is what the probe's `[96]` mds states and what dxp's
/// `createIdx2AddrSdsc` conversion count is derived from.
#[test]
fn the_index_plan_is_the_whole_table_per_core() {
    let op = probe_matmul(Some(token_plane()));
    let plan = op
        .iter
        .of_index_entries(&[("mb", PageExtent::single_position())])
        .expect("96 tokens divide by a pin of 1");
    assert_eq!(plan.extent("mb"), 96, "the whole table");
    assert_eq!(
        plan.split_of("mb"),
        1,
        "the mb split removed — every core reads the same table"
    );
    assert_eq!(
        op.iter.split_of("mb"),
        32,
        "the OP still divides mb 32 ways (3 rows/core)"
    );
}

/// ⭐⭐⭐ THE PLAN UNDER THE GATHER SPLITS `mb` ONLY — AT A GEOMETRY THE COST MODEL K-SPLITS.
/// The 26b DOWN chunk's own shape [mb=2, out=704 (11 sticks), in=768] sends the generic
/// prefill splitter to `{in:2, out:11}` — legal and PSUM-accumulated on an ungathered
/// matmul, but under the token-plane gather `skip_addr` would step a FRACTION of the expert
/// plane (the guard test above pins that refusal). The builder now enforces the probe's own
/// division where the plan is made: `core_split(mb)` on the entry axis, `in`/`out` whole —
/// byte-identical at every single-stick-`out` geometry the probe ran (the reduce branch
/// takes the same expression), and the multi-stick geometries emit the probe's shape instead
/// of refusing.
#[test]
fn the_plan_under_a_token_plane_gather_splits_mb_only_at_a_multi_stick_out() {
    let mut op = matmul_opspec_off_operands_phys_gathered::<Fp16>(
        MatM::of_token_rows(2),
        MatN::of_out_features(704),
        MatK::of_in_features(768),
        MatY::unbatched(),
        SharedKernelBmmForm::of_attn_rows(false, QueryRowCount::of_mq(1)),
        "Tensor0",
        "Tensor1",
        "Tensor3",
        0,
        0,
        0,
        <Fp8 as ktir_superdsc::superdsc_opspec::DataFormat>::DF,
        None,
        Some(token_plane()),
    )
    .expect("the DOWN chunk's gathered matmul builds");
    // The W8A8 paint, the door's own loop verbatim (skipping OUTPUT and the INDEX).
    for arg in &mut op.args {
        if !matches!(
            arg.view().role,
            ktir_superdsc::superdsc_opspec::Role::Output
                | ktir_superdsc::superdsc_opspec::Role::Index
        ) {
            arg.set_df(<Fp8 as ktir_superdsc::superdsc_opspec::DataFormat>::DF);
        }
    }
    // THE PLAN IS THE PROBE'S LAW: the entry axis at its own core_split, nothing else.
    let mut expect: std::collections::BTreeMap<&'static str, u32> =
        std::collections::BTreeMap::new();
    expect.insert("mb", 2);
    assert_eq!(
        op.iter.splits(),
        &expect,
        "mb only — core_split(2, 32); the cost model's {{in:2, out:11}} is unwritable under \
         the gather (skip_addr would step a fraction of the expert plane)"
    );
    // AND IT EMITS — where the cost model's own plan would have been refused by the guard.
    let folds = SdscFoldSet::new(op.iter.cores_used());
    let (e, _) = superdsc::emit_sdsc("MatMul_0", &op, &folds, None)
        .expect("the mb-only plan emits at the multi-stick-out geometry");
    let j = serde_json::to_value(&e).unwrap();
    assert_eq!(
        j["dscs_"][0]["MatMul_0"]["numCoresUsed_"], 2,
        "core_split(2, 32) = 2 cores"
    );
    // The index plan follows the probe's law too: the whole table per core, unsplit.
    let plan = op
        .iter
        .of_index_entries(&[("mb", PageExtent::single_position())])
        .expect("2 tokens divide by a pin of 1");
    assert_eq!(
        plan.split_of("mb"),
        1,
        "the mb split removed from the index plan — every core reads the same table"
    );
}

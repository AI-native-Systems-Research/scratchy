// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ THE PREFILL (m>1) EXPERT-MATMUL BODY — the door's per-(slot, chunk) gathered
//! form, pinned op by op at the 26b gate/up geometry (the ALIGNED tail — `inn=2816` is a
//! whole fp8 stick, so the whole-tensor quantize convert is m-generic and the body lowers
//! END TO END).
//!
//! ## The form under test (design §4.4)
//! ```text
//!   P0  rtkstg_g{g}  identity staging, per 64-col group: the router's `[m, W]` stick-major
//!                   group corner → the padded `[m_pad, W]` contract (m_pad = 64-multiple)
//!                   — THE TOP-K DOOR'S OWN STAGE (the pair owns the whole index chain:
//!                   the sanitize `add` fires only at W > E)
//!   P1  rtkkt        the Kᵀ restickify transpose `[m_pad, W] → [W, m_pad]` (the
//!                   card-proven builder — the pair's first op)
//!       rtkix/rtkvl  the VENDOR top-k pair (topkindex/topkvalue) at
//!                   TopKGeom{candidates: W, slots: 64, tokens: m_pad}
//!       rtkfl        the FLAT restickify: the pair's token-sticked `[64, m_pad]` index
//!                   output → FLAT `[64, m_pad]` fp16 (stick `mb`, dim 0 — the V-door's
//!                   card-proven declaration on the write side; probe-validated binding)
//!       rtkxf        dl16tofp32 FLAT at `[64, m_pad]` — BOTH operands stick `mb` (the
//!                   shared-%stick_dim law quantization_double_pad.ddl binds on the leg;
//!                   the flat-tail spelling died at ddl_conversion.cpp:2539). ⛔ THE SLOT
//!                   AXIS IS ONE FP16 STICK (64), NOT W — the pair emits 64 slots at
//!                   every candidate count
//!       rtkxi        fp32toint32 INDEX-ROWS at `[64, m_pad]` — BOTH operands stick `mb`
//!                   (the DDL equality, unary_parallel.ddl:59-60); result s's HEAD SLOT's
//!                   run lands at elements `(s·rep)·m_pad` (rep = 64/k, the landing's
//!                   replica law), the contiguous run every per-slot gather reads
//!       rtkti/rtktv  the landing transposes → token-major `[m_pad, 64]` ids/vals (one
//!                   stick per token — the bytes every score-side consumer addresses)
//!   P2  t4_fq_*     the SHARED activation quantize over the whole `[m, k·in]` (one chain,
//!                   gate/up dedup — the m=1 body's own call, unchanged)
//!   P3  emmm_s{s}_c{c}  per-(slot, chunk) gathered matmuls: A = the `afp8` window at the
//!                   col corner `s·inn_pad`, B = the bank's chunk plane `χ` GATHERED by
//!                   slot s's index run (`GatherIndex::of_expert_planes` — the probe's
//!                   token-plane form, `zz_the_token_plane_gathered_matmul_is_the_probe_form`),
//!                   C = the `xraw` window at group corner `s·n + χ·chunk_n`
//!   P4  emsg_s{s}_g{g}_r{r}  per-(slot, group, run) scale gathers: ONE fp16 stick per
//!                   entry, the raw expert id composing with the group-major bank's plane
//!                   base `g·E·64` — dest entry `(s·G+g)·m + i` is the stick-major
//!                   `[m, k·n]` position of group `s·G+g`, row i
//!   P5  ems / emo   the dequant, TWO whole-tensor stick-major legs at `[m, k·n]`
//! ```
//!
//! NO `emwg_` weight gathers and NO `xwscratch` at m>1 — the matmuls gather the BANK
//! directly through the kernel index, which is the whole point of the form. The m=1 body
//! is UNTOUCHED (pinned by `moe_expert_chunk_cut.rs` and cross-checked below). The
//! half-stick tail (the 26b DOWN's `inn=704`) lowers m>1 too — the shared quantize
//! chain's per-slot §4.5 arms at the `col_of` group corners (pinned below at the DOWN
//! geometry, seam-matched to the matmul A windows).
//!
//! ⛔ THE HONEST BOUNDARY: these tests pin the EMITTED DESCRIPTORS — names, counts, dims,
//! dtypes, gather presence, and the baked offsets readable as DELTAS between ops. The
//! numerics are the pod bake's to prove (the card-risk register: the kernel-bearing
//! token-plane gather has no green card run at production shape yet).

use scratchy_subtile::subtile_ir::{
    ExpertBundle, ExpertProj, NeoX, SubOp, SubtileIR, SubtileId, SubtileNode, TensorId,
    TensorRegion, TensorShape, TopK,
};

/// The 26b gate/up door's own numbers (moe_expert_chunk_cut's), at a PREFILL row count:
/// two tokens ⇒ `m_pad = 64` (the pad is REAL, so the P0 staging fires).
const M: u32 = 2;
const K: u32 = 8;
const E: u32 = 128;
const N_OUT: u32 = 704;
const IN: u32 = 2816;

/// The expert-matmul door at the given geometry — `moe_expert_chunk_cut`'s own harness:
/// an honest router chain — `RouteArgsort` over the raw logits, `RouteTopK` inverting
/// the ranks into the `[m, k]` indices (P1's provenance law — the sort's ids must be
/// a top-k output, the only mint of in-range ids), a `RouteGatherScores` closing the
/// chain so the pair's binding walk binds (the argsort and the gather both FOLD into
/// the fused pair, so only the topk program mints) — then an `ExpertSort` produces
/// the pair rows (the routing tile the lowering binds THROUGH the sort to `t6`), and
/// the `ExpertMatmul` reads `[rows, routing, w, s]`. Tensors: `t0` x `[m, in]`, `t1`
/// the router logits `[m, e]`, `t2` the fp8 bank `[E·n, in]`, `t3` the scales
/// `[E·n, 1]`, `t4` the sort's pair rows `[m, k·in]`, `t5` the matmul out `[m, k·n]`,
/// `t6` the top-k indices `[m, k]`, `t7` the argsort's rank vector `[m, e]`, `t8`
/// the gather's scores `[m, k]` (t6–t8 appended after the result so every `_o{tid}`
/// pin and every `t{tid}_` synth pin keeps its id).
fn expert_matmul_ir(
    m: u32,
    k: u32,
    e: u32,
    n_out: u32,
    in_dim: u32,
    proj: ExpertProj,
) -> SubtileIR<NeoX> {
    let nz = |n: u32| std::num::NonZeroU32::new(n).unwrap();
    let tensors = vec![
        TensorShape {
            rows: m,
            cols: in_dim,
        }, // 0: x (source)
        TensorShape { rows: m, cols: e }, // 1: the router logits (source)
        TensorShape {
            rows: e * n_out,
            cols: in_dim,
        }, // 2: the fp8 bank (source)
        TensorShape {
            rows: e * n_out,
            cols: 1,
        }, // 3: the scale bank (source)
        TensorShape {
            rows: m,
            cols: in_dim * k,
        }, // 4: the sort's pair rows
        TensorShape {
            rows: m,
            cols: n_out * k,
        }, // 5: the matmul's out
        TensorShape { rows: m, cols: k }, // 6: the top-k indices (the sort's routing)
        TensorShape { rows: m, cols: e }, // 7: the argsort's rank vector (folds)
        TensorShape { rows: m, cols: k }, // 8: the gather's scores (folds)
    ];
    let tr = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    // The honest router chain's two folding legs: the argsort (its only
    // consumer is the topk) and the gather (its ids come from the topk over
    // the same logits). Both fold into the fused pair, so the pair's binding
    // walk binds and only the topk program mints.
    let argsort = SubtileNode {
        id: SubtileId::from_index(3),
        op: SubOp::RouteArgsort,
        inputs: vec![tr(1)],
        output: tr(7),
    };
    let topk = SubtileNode {
        id: SubtileId::from_index(0),
        op: SubOp::RouteTopK {
            k: TopK::new(nz(k)),
        },
        inputs: vec![tr(7)],
        output: tr(6),
    };
    let gather = SubtileNode {
        id: SubtileId::from_index(4),
        op: SubOp::RouteGatherScores,
        inputs: vec![tr(1), tr(6)],
        output: tr(8),
    };
    let sort = SubtileNode {
        id: SubtileId::from_index(1),
        op: SubOp::ExpertSort {
            experts: scratchy_subtile::subtile_ir::NumExperts::new(nz(e)),
            k: TopK::new(nz(k)),
            bundle: ExpertBundle::SwitchGlu,
        },
        inputs: vec![tr(0), tr(6)],
        output: tr(4),
    };
    let matmul = SubtileNode {
        id: SubtileId::from_index(2),
        op: SubOp::ExpertMatmul {
            proj,
            n: n_out,
            k: TopK::new(nz(k)),
            quant: scratchy_subtile::lower::ExpertQuant::declared(64, 8),
            bundle: ExpertBundle::SwitchGlu,
        },
        inputs: vec![tr(4), tr(4), tr(2), tr(3)],
        output: tr(5),
    };
    SubtileIR {
        result: TensorId::from_index(5),
        tensors,
        num_sources: 4,
        nodes: vec![argsort, topk, gather, sort, matmul],
        op_output: Vec::new(),
    }
}

/// Lower through the REAL door path and return the emitted ops AND the layout (the
/// reservations are half of what these tests pin).
fn through_the_door(
    ir: &SubtileIR<NeoX>,
) -> (
    Vec<ktir_superdsc::emit::EmittedOp>,
    ktir_superdsc::placement::BundleLayout,
) {
    let weight_ids = std::collections::HashSet::new();
    scratchy_target_spyre::lower_subtile_tape_to_ktir::lower_graph_to_superdsc(
        ir,
        &weight_ids,
        ktir_superdsc::ktir_node::ActiveCap::FULL,
        false,
    )
    .unwrap_or_else(|e| panic!("lower through the door: {e:?}"))
}

/// The emitted op names, in emission order.
fn op_names(ops: &[ktir_superdsc::emit::EmittedOp]) -> Vec<&str> {
    ops.iter().map(|o| o.op_name.as_str()).collect()
}

/// One op's single dsc, as JSON — the deep-field assertions (dims, dtypes, allocs) read
/// the same serialization the bake writes.
fn op_json(o: &ktir_superdsc::emit::EmittedOp) -> serde_json::Value {
    let dsc = o.dsc();
    let d = dsc
        .dscs_
        .first()
        .and_then(|m| m.values().next())
        .unwrap_or_else(|| panic!("{}: an emitted op has at least one dsc", o.op_name));
    serde_json::to_value(d).unwrap()
}

fn find_op<'a>(
    ops: &'a [ktir_superdsc::emit::EmittedOp],
    name: &str,
) -> &'a ktir_superdsc::emit::EmittedOp {
    ops.iter()
        .find(|o| o.op_name == name)
        .unwrap_or_else(|| panic!("the op {name} exists (have {:?})", op_names(ops)))
}

/// An operand's core-0 HBM start, in bytes — the `AllocNode` behind the labeledDs the
/// compute op names (the per-op offsets `resolve_seg_base` folds in make this a per-op
/// fact, which is what the DELTA laws below read).
fn operand_start_bytes(o: &ktir_superdsc::emit::EmittedOp, ds_type: &str) -> u64 {
    let j = op_json(o);
    let lds = j["labeledDs_"]
        .as_array()
        .expect("labeledDs_")
        .iter()
        .find(|l| l["dsType_"] == ds_type)
        .unwrap_or_else(|| panic!("{}: a {ds_type} labeledDs", o.op_name));
    let idx = lds["ldsIdx_"].as_u64().expect("ldsIdx_");
    let alloc = j["scheduleTree_"]
        .as_array()
        .expect("scheduleTree_")
        .iter()
        .find(|n| n["nodeType_"] == "allocate" && n["component_"] == "hbm" && n["ldsIdx_"] == idx)
        .unwrap_or_else(|| panic!("{}: the {ds_type} hbm allocate", o.op_name));
    alloc["startAddressCoreCorelet_"]["data_"]["[0, 0, 0]"]
        .as_str()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("{}: core 0's start address", o.op_name))
}

/// A GATHERED source's core-0 HBM start — the `value_tensor` indirect alloc (a
/// kernel-less copy labels its gathered source as an indirect value alloc, not an
/// `INPUT`; a gathered matmul has exactly one besides its `KERNEL`, so callers pick by
/// `ds_type` there and by this predicate here).
fn gathered_source_start_bytes(o: &ktir_superdsc::emit::EmittedOp) -> u64 {
    let j = op_json(o);
    let alloc = j["scheduleTree_"]
        .as_array()
        .expect("scheduleTree_")
        .iter()
        .find(|n| n["indirectAllocType_"] == "value_tensor")
        .unwrap_or_else(|| panic!("{}: the gathered value allocate", o.op_name));
    alloc["startAddressCoreCorelet_"]["data_"]["[0, 0, 0]"]
        .as_str()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("{}: core 0's start address", o.op_name))
}

/// The LAST operand's (the copy's DESTINATION's) core-0 HBM start.
fn dest_start_bytes(o: &ktir_superdsc::emit::EmittedOp) -> u64 {
    let j = op_json(o);
    let lds = j["labeledDs_"]
        .as_array()
        .expect("labeledDs_")
        .last()
        .unwrap();
    let idx = lds["ldsIdx_"].as_u64().expect("ldsIdx_");
    let alloc = j["scheduleTree_"]
        .as_array()
        .expect("scheduleTree_")
        .iter()
        .find(|n| n["nodeType_"] == "allocate" && n["component_"] == "hbm" && n["ldsIdx_"] == idx)
        .unwrap_or_else(|| panic!("{}: the destination's hbm allocate", o.op_name));
    alloc["startAddressCoreCorelet_"]["data_"]["[0, 0, 0]"]
        .as_str()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("{}: core 0's start address", o.op_name))
}

/// ⭐ THE PAIR'S CHAIN: the staging fires (m_pad=64 ≠ m=2), the Kᵀ restickify,
/// the vendor pair, the flat restickify + two converts, and the two landing
/// transposes — and the converts run at the SLOT-MAJOR `[64, m_pad]`, pinned by
/// the `dl16tofp32` leg's own dims (the pair emits ONE 64-slot stick per token
/// regardless of the candidate count W; a chain keyed on W would state
/// `[W, m_pad]` = mb 128 instead).
#[test]
fn the_mint_stages_transposes_and_converts_at_the_padded_shape() {
    let (ops, _layout) = through_the_door(&expert_matmul_ir(M, K, E, N_OUT, IN, ExpertProj::Gate));
    let names = op_names(&ops);
    // P0: W/64 = 2 staging copies, one per 64-column group, BEFORE the transpose
    // (the top-k door's own stage — the identity form at W == E, the sanitize
    // `add` only at W > E).
    let staging: Vec<&str> = names
        .iter()
        .copied()
        .filter(|n| n.starts_with("rtkstg_"))
        .collect();
    assert_eq!(
        staging,
        vec!["rtkstg_g0_o6", "rtkstg_g1_o6"],
        "two groups at W=128"
    );
    // P1: the whole pair chain, in order — the Kᵀ restickify, the vendor pair
    // (index then value), the flat restickify, the two converts, then the two
    // landing transposes.
    let pos = |n: &str| names.iter().position(|x| *x == n).unwrap();
    for (a, b) in [
        ("rtkstg_g1_o6", "rtkkt_o6"),
        ("rtkkt_o6", "rtkix_o6"),
        ("rtkix_o6", "rtkvl_o6"),
        ("rtkvl_o6", "rtkfl_o6"),
        ("rtkfl_o6", "rtkxf_o6"),
        ("rtkxf_o6", "rtkxi_o6"),
        ("rtkxi_o6", "rtkti_o6"),
        ("rtkti_o6", "rtktv_o6"),
    ] {
        assert!(pos(a) < pos(b), "{a} precedes {b} (names: {names:?})");
    }
    // THE DIRECTION PIN: the fp32 leg runs at `[rows=64, cols=m_pad]` = [64, 64]
    // — the SLOT axis is one fp16 stick, not W.
    let xf = op_json(find_op(&ops, "rtkxf_o6"));
    assert_eq!(
        xf["N_"]["mb_"], 64,
        "rows = the 64-slot stick (the pair's slot count), NOT the padded expert width"
    );
    assert_eq!(
        xf["N_"]["out_"], 64,
        "cols = m_pad (the padded token count)"
    );
    // And the index leg's OUTPUT is the u32 table the gathers read.
    let idx = op_json(find_op(&ops, "rtkxi_o6"));
    let out_lds = idx["labeledDs_"]
        .as_array()
        .expect("labeledDs_")
        .iter()
        .find(|l| l["dsType_"] == "OUTPUT")
        .expect("the convert's output");
    assert_eq!(
        out_lds["dataFormat_"], "SENUINT32",
        "the gather's index dtype"
    );
    assert_eq!(out_lds["wordLength"], 4);
    // ⭐ THE WORK DIVISION IS WHOLE STICKS — the vendor's chunk-param law: every param along
    // a dim must be a multiple of that dim's stick size in EVERY non-index lds
    // (`isDimensionParamMultipleOfStickSize`, L3DlOpsScheduler.cpp:1429; `getStickSize` :447
    // answers the stick size for a stick dim and 1 otherwise). The index-rows OUTPUT's u32
    // stick rides `mb` at 32 elems, so the core stage's per-core `mb` must be a WHOLE
    // 32-multiple — the 32-way element split this replaces left per-core `mb = 2` and every
    // candidate `param ≤ 2` failed `param % 32`, the EMPTY set the whole bundle was refused
    // with (`DtException: There must be at least one valid candidate.`, :1535 — measured ×2
    // deterministic at 26b and tiny26 on the pod). `mb` now rides unsplit and `out` (the
    // division's fp32-wide stick basis) takes one core per stick.
    let core_stage = &idx["dataStageParam_"]["0"]["ss_"];
    assert_eq!(
        core_stage["mb_"], 64,
        "the 64-slot axis rides unsplit — the per-core `mb` is whole u32 sticks"
    );
    assert_eq!(
        core_stage["out_"], 32,
        "one whole fp32 stick of m_pad per core"
    );
    // ⭐ THE DDL LAWS — TWO independent ones, one per convert leg. `unary_parallel.ddl:59-60`
    // binds the fp32toint32 operands' slice AND stick `(dim,size)` vectors EQUAL
    // (`ddl_conversion.cpp:2715-2729` compares them; `getStickSizes`'s dim component is
    // the PrimaryDimTypes enum). The round-1 spelling (INPUT stick `out`, OUTPUT stick
    // `mb`) died on the pod in the `UNREACHABLE` "slice size does not match" — the DIM
    // component differed. And `quantization_double_pad.ddl` SHARES `%stick_dim` between
    // the dl16tofp32 leg's two operands (input via `%slice_layout_in`, output via
    // `%padded_dim_out`'s `nonPaddedDim`) — the round-2 FLAT-TAIL spelling (input stick
    // `out`, output stick `mb`) emptied the binding intersection and died at
    // `ddl_conversion.cpp:2539`. So the flat RESTICKIFY (`rtkfl`) exists: it re-sticks
    // the pair's m_pad-sticked K-form into the FLAT `[64, m_pad]` fp16 table (the slot
    // axis is the fp16 stick — the pair emits 64 slots at EVERY candidate count, so
    // this is NOT `[W, m_pad]`), and BOTH legs declare the `mb` stick on BOTH operands.
    let sticks = |j: &serde_json::Value, ds: &str| {
        let pds = &j["primaryDsInfo_"][ds];
        (pds["stickDimOrder_"].clone(), pds["stickSize_"].clone())
    };
    let fl = op_json(find_op(&ops, "rtkfl_o6"));
    assert_eq!(
        fl["N_"]["mb_"], 64,
        "the flat restickify runs at [slots, m_pad] too"
    );
    assert_eq!(
        sticks(&fl, "INPUT"),
        (serde_json::json!(["out"]), serde_json::json!([64])),
        "rtkfl reads the K-form on its own `out` (m_pad) stick — the pair's output form"
    );
    assert_eq!(
        sticks(&fl, "OUTPUT"),
        (serde_json::json!(["mb"]), serde_json::json!([64])),
        "rtkfl writes the FLAT fp16 table — stick `mb`, dim 0"
    );
    assert_eq!(
        sticks(&xf, "INPUT"),
        (serde_json::json!(["mb"]), serde_json::json!([64])),
        "rtkxf reads the FLAT fp16 table on the `mb` stick — the shared-%stick_dim law"
    );
    assert_eq!(
        sticks(&xf, "OUTPUT"),
        (serde_json::json!(["mb"]), serde_json::json!([32])),
        "rtkxf writes the fp32 table FLAT — the SAME `mb` dim, the shared stick"
    );
    assert_eq!(
        sticks(&idx, "INPUT"),
        (serde_json::json!(["mb"]), serde_json::json!([32])),
        "rtkxi's input carries the SAME mb stick — the equality's dim component"
    );
    assert_eq!(
        sticks(&idx, "OUTPUT"),
        (serde_json::json!(["mb"]), serde_json::json!([32])),
        "rtkxi's output — the vectors are EQUAL, the DDL constraint's demand"
    );
    // The FLAT division: `mb` (a stick dim in the flat form) rides unsplit and `out`
    // takes one core per fp16 stick of the WIDER basis — at `[64, 64]` that is ONE core
    // holding the whole table (legal: param_mb 64 % 32 == 0, param_out 64 % 64 == 0).
    let xf_stage = &xf["dataStageParam_"]["0"]["ss_"];
    assert_eq!(
        xf_stage["mb_"], 64,
        "the widen leg's `mb` rides unsplit too"
    );
    assert_eq!(
        xf_stage["out_"], 64,
        "one whole fp16 stick of m_pad — the whole table on one core"
    );
}

/// ⭐ THE RESERVATIONS: the pair's padded/transposed scratches at `[m_pad, W]` /
/// `[W, m_pad]` / the 64-slot `[64, m_pad]` trio, the dequant tensors at the
/// STICK-MAJOR `[m, k·n]` — and NO `xwscratch`, NO `xsub` (the kernel gather needs
/// neither the copied slab nor the fine index).
#[test]
fn the_reservations_are_the_padded_mint_and_stick_major_dequant_shapes() {
    let (ops, layout) = through_the_door(&expert_matmul_ir(M, K, E, N_OUT, IN, ExpertProj::Gate));
    let s = layout.synth.borrow();
    let size = |k: &str| -> u64 { *s.sizes.get(k).unwrap_or_else(|| panic!("synth {k} placed")) };
    assert_eq!(
        size("t6_xp"),
        64 * 128 * 2,
        "the staging table [m_pad, W] fp16"
    );
    assert_eq!(
        size("t6_xt"),
        128 * 64 * 2,
        "the transposed table [W, m_pad] fp16"
    );
    assert_eq!(
        size("t6_topkidx"),
        64 * 64 * 2,
        "the pair's index output [64, m_pad] fp16 (the slot axis is one stick, not W)"
    );
    assert_eq!(
        size("t6_topkval"),
        64 * 64 * 2,
        "the pair's value output [64, m_pad] fp16"
    );
    assert_eq!(
        size("t6_xfl"),
        64 * 64 * 2,
        "the flat table [64, m_pad] fp16 — the flat restickify's output"
    );
    assert_eq!(size("t6_xf32"), 64 * 64 * 4, "the fp32 leg [64, m_pad]");
    assert_eq!(size("t6_xidx"), 64 * 64 * 4, "the u32 index [64, m_pad]");
    for k in ["t5_xsscratch", "t5_xraw", "t5_xdqa"] {
        assert_eq!(
            size(k),
            2 * 5632 * 2,
            "{k} is the stick-major [m, k·n] fp16"
        );
    }
    assert_eq!(
        size("t4_fq_afp8"),
        2 * 22528,
        "the quantized activation [m, k·inn_pad] fp8 (1 B/elem)"
    );
    assert!(
        !s.sizes.contains_key("t5_xwscratch"),
        "NO weight scratch at m>1"
    );
    assert!(!s.sizes.contains_key("t6_xsub"), "NO fine index at m>1");
    assert!(
        !ops.iter().any(|o| o.op_name.starts_with("emwg_")),
        "NO weight-gather copies at m>1 — the matmuls gather the bank directly"
    );
}

/// ⭐⭐ THE PER-(SLOT, CHUNK) MATMULS: 8 slots × 11 chunks, `batchmatmulfp8`, the probe's
/// token-plane gather attached (rank-3 KERNEL, SENUINT32 index, the whole-table index
/// plan), A windows at the stick-major col corners, and the per-slot index runs stepping
/// `m_pad` u32 elements.
#[test]
fn the_matmuls_are_per_slot_per_chunk_with_the_kernel_gather() {
    let (ops, _layout) = through_the_door(&expert_matmul_ir(M, K, E, N_OUT, IN, ExpertProj::Gate));
    let matmuls: Vec<&ktir_superdsc::emit::EmittedOp> = ops
        .iter()
        .filter(|o| o.op_name.starts_with("emmm_"))
        .collect();
    assert_eq!(matmuls.len(), 88, "8 slots × 11 chunks");
    assert_eq!(matmuls[0].op_name, "emmm_s0_c0_o5");
    assert_eq!(matmuls[1].op_name, "emmm_s0_c1_o5");
    assert_eq!(matmuls[11].op_name, "emmm_s1_c0_o5");
    for o in &matmuls {
        let j = op_json(o);
        assert_eq!(
            j["computeOp_"][0]["opFuncName"], "batchmatmulfp8",
            "{}",
            o.op_name
        );
        assert_eq!(j["N_"]["mb_"], 2, "{}: M = the token count", o.op_name);
        assert_eq!(j["N_"]["in_"], 2816, "{}: K = the padded inn", o.op_name);
        assert_eq!(
            j["N_"]["out_"], 64,
            "{}: N = the chunk's columns",
            o.op_name
        );
        // The probe's spelling: rank-3 KERNEL pinned-dim-first, the index BEFORE the
        // output at HBM-only.
        assert_eq!(
            j["primaryDsInfo_"]["KERNEL"]["layoutDimOrder_"],
            serde_json::json!(["mb", "in", "out"]),
            "{}: the token-plane kernel re-declaration",
            o.op_name
        );
        let idx_lds = j["labeledDs_"]
            .as_array()
            .expect("labeledDs_")
            .iter()
            .find(|l| l["dsType_"] == "KERNEL_IDX")
            .unwrap_or_else(|| panic!("{}: a KERNEL_IDX operand", o.op_name));
        assert_eq!(idx_lds["dataFormat_"], "SENUINT32", "{}", o.op_name);
        assert_eq!(idx_lds["wordLength"], 4, "{}", o.op_name);
        let idx_alloc = j["scheduleTree_"]
            .as_array()
            .expect("scheduleTree_")
            .iter()
            .find(|n| n["indirectAllocType_"] == "index_tensor")
            .unwrap_or_else(|| panic!("{}: the index allocate", o.op_name));
        assert_eq!(
            idx_alloc["maxDimSizes_"],
            serde_json::json!([2]),
            "{}: the WHOLE 2-entry table — the per-slot run, not a per-core slice",
            o.op_name
        );
    }
    // ⭐ THE A-WINDOW LAW: slot s's activation window starts `s·(m·inn_pad)` fp8 bytes in —
    // `col_of(m, k·inn_pad, s·inn_pad)` = the stick-major col corner (a whole fp8 stick
    // group per slot, m rows each).
    let a0 = operand_start_bytes(matmuls[0], "INPUT");
    let a1 = operand_start_bytes(matmuls[11], "INPUT");
    assert_eq!(
        a1 - a0,
        2 * 2816,
        "slot 1's A window is m·inn_pad fp8 bytes past slot 0's"
    );
    // ⭐ THE PER-SLOT INDEX RUNS: slot s gathers result s's HEAD SLOT's run —
    // u32 elements `(s·rep)·m_pad`, rep = 64/k (the landing's replica law: the
    // pair writes result r at slots [r·rep, (r+1)·rep), so the head slot is the
    // only run that holds result s at every token — the old `s·m_pad` read
    // result s/rep's replicas and staged the top-2 experts ×4 at the 26b) —
    // read here as the KERNEL_IDX start stepping rep·m_pad·4 B.
    let i0 = operand_start_bytes(matmuls[0], "KERNEL_IDX");
    let i1 = operand_start_bytes(matmuls[11], "KERNEL_IDX");
    assert_eq!(
        i1 - i0,
        8 * 64 * 4,
        "slot 1's index run is rep·m_pad u32 elements in (rep=8 at k=8 — the head slot)"
    );
}

/// ⭐⭐ THE PER-(SLOT, GROUP) SCALE GATHERS: ONE fp16 stick per entry, the raw expert id
/// composing with the GROUP-MAJOR bank's plane base — the source advances one group plane
/// `E·64` fp16 elements per group and does NOT move with the slot (the slot is in the
/// index), while the index runs step exactly like the matmuls'.
#[test]
fn the_scale_gathers_stage_one_stick_per_entry_at_the_group_planes() {
    let (ops, _layout) = through_the_door(&expert_matmul_ir(M, K, E, N_OUT, IN, ExpertProj::Gate));
    let gathers: Vec<&ktir_superdsc::emit::EmittedOp> = ops
        .iter()
        .filter(|o| o.op_name.starts_with("emsg_"))
        .collect();
    // m=2 ≤ the 32-entry cap ⇒ ONE run per (slot, group): 8 × 11 ops, `r0`.
    assert_eq!(gathers.len(), 88, "8 slots × 11 groups × 1 run");
    assert_eq!(gathers[0].op_name, "emsg_s0_g0_r0_o5");
    assert_eq!(gathers[1].op_name, "emsg_s0_g1_r0_o5");
    // The group-plane law: fixed slot, group g's source is `g·E·64` fp16 elements in.
    let src = |s: usize, g: usize| {
        let name = format!("emsg_s{s}_g{g}_r0_o5");
        gathered_source_start_bytes(find_op(&ops, &name))
    };
    for g in 0..11usize {
        assert_eq!(
            src(0, g) - src(0, 0),
            (g as u64) * E as u64 * 64 * 2,
            "group {g}'s source is g·E·64 fp16 elements past group 0's"
        );
    }
    // The slot is in the INDEX, not the source: fixed group, every slot reads the same base.
    for s in 0..8usize {
        assert_eq!(
            src(s, 0),
            src(0, 0),
            "slot {s}'s gather reads slot 0's source"
        );
    }
    // And the index runs step `rep·m_pad` u32 elements — the matmuls' own
    // head-slot law (rep=8 at k=8).
    let idx_start = |s: usize| {
        let name = format!("emsg_s{s}_g0_r0_o5");
        operand_start_bytes(find_op(&ops, &name), "KERNEL_IDX")
    };
    assert_eq!(
        idx_start(1) - idx_start(0),
        8 * 64 * 4,
        "the per-slot index runs step rep·m_pad u32 elements"
    );
    // ⭐ THE DESTINATION LAW — the staging arrangement itself: dest entry `(s·G+g)·m + i`
    // lands at `((s·G+g)·m + i)·64` fp16 elements, which IS the stick-major `[m, k·n]`
    // position of group `s·G+g`, row i. Read as deltas: group g of slot 0 steps
    // `m·64·2` B; slot 1's group 0 sits `G·m·64·2` B past slot 0's.
    let dst = |s: usize, g: usize| {
        let name = format!("emsg_s{s}_g{g}_r0_o5");
        dest_start_bytes(find_op(&ops, &name))
    };
    for g in 0..11usize {
        assert_eq!(
            dst(0, g) - dst(0, 0),
            (g as u64) * 2 * 64 * 2,
            "group {g}'s dest is m·64 fp16 elements past group 0's (one row of the group)"
        );
    }
    assert_eq!(
        dst(1, 0) - dst(0, 0),
        11 * 2 * 64 * 2,
        "slot 1's group 0 is G·m·64 elements past slot 0's group 0 — the stick-major \
         [m, k·n] group corner, not a flat row"
    );
}

/// ⭐ THE DEQUANT: two whole-tensor legs at the stick-major `[m, k·n]`, emitted AFTER
/// every matmul and scale gather (the legs read what they wrote).
#[test]
fn the_dequant_is_two_whole_tensor_legs_after_the_gathers() {
    let (ops, _layout) = through_the_door(&expert_matmul_ir(M, K, E, N_OUT, IN, ExpertProj::Gate));
    let names = op_names(&ops);
    let pos = |n: &str| names.iter().position(|x| *x == n).unwrap();
    let last_matmul = names
        .iter()
        .rposition(|n| n.starts_with("emmm_"))
        .expect("matmuls");
    let last_gather = names
        .iter()
        .rposition(|n| n.starts_with("emsg_"))
        .expect("scale gathers");
    assert!(last_matmul < pos("ems_o5"), "ems follows the matmuls");
    assert!(last_gather < pos("ems_o5"), "ems follows the scale gathers");
    assert!(pos("ems_o5") < pos("emo_o5"), "emo follows ems");
    for name in ["ems_o5", "emo_o5"] {
        let j = op_json(find_op(&ops, name));
        assert_eq!(j["computeOp_"][0]["opFuncName"], "mul", "{name}");
        assert_eq!(j["N_"]["mb_"], 2, "{name}: the token rows");
        assert_eq!(j["N_"]["out_"], 5632, "{name}: the whole k·n width");
    }
}

/// ⭐ THE SHARED QUANTIZE: the m=1 body's own chain call, unchanged — one `afp8` convert
/// over the whole `[m, k·in]`, NO dense `fq_mm` (the per-slot matmuls are this door's
/// body; nothing is emitted only to be spliced out).
#[test]
fn the_quantize_is_the_shared_whole_tensor_chain() {
    let (ops, _layout) = through_the_door(&expert_matmul_ir(M, K, E, N_OUT, IN, ExpertProj::Gate));
    let names = op_names(&ops);
    assert_eq!(
        names.iter().filter(|n| n.ends_with("fq_afp8_op")).count(),
        1,
        "one whole-tensor convert at the aligned inn"
    );
    assert!(
        !names.iter().any(|n| n.ends_with("fq_mm")),
        "NO dense body — the per-slot matmuls are the door's own body"
    );
}

/// ⭐ THE HALF-STICK TAIL AT PREFILL (design §4.5 LANDED): the 26b DOWN's own geometry
/// (in=704 → inn_pad=768) lowers m>1 END TO END — the shared quantize chain's per-slot
/// arms replace the flat per-pair slices: k rank-2 stick-major `qfp8ch` converts at the
/// [`crate::addr::col_of`] group corners `s·inn → s·inn_pad`. Pins the whole seam:
///   · EIGHT arms (slot 0 keeps the shipped `fq_afp8_op` name), ONE zero-fill for the
///     last slot's overhang;
///   · THE SEAM LAW: arm s's OUTPUT starts exactly where slot s's chunk-0 matmul READS
///     its A window (`col_of(m, k·inn_pad, s·inn_pad)` — one law at both ends), and the
///     arms advance by the PADDED group-corner stride `m·inn_pad` B (the `m` term is the
///     §4.5 law — the flat m=1 form would say `inn_pad`);
///   · the pair's chain is the aligned tail's own chain (the tail touches ONLY the
///     quantize
///     arms), the chunk cut at this geometry is 4 chunks of 704 columns (32 matmuls), and
///     the scale gathers run per (slot, group) at n=2816 (8·44);
///   · the reservations widen m-generically: `cl` `[m, cl_reach]`, `afp8` `[m, k_pad]`.
#[test]
fn the_half_stick_tail_at_prefill_lowers_through_the_per_slot_arms() {
    // The DOWN projection's own geometry: in=704 (5.5 fp8 sticks), n=2816.
    let (ops, layout) = through_the_door(&expert_matmul_ir(2, K, E, 2816, 704, ExpertProj::Down));
    let names = op_names(&ops);
    let (m, pairs, inn) = (2u32, K, 704u32);
    let inn_pad = 768; // fp8_k_pad(704) — the whole reason the padded layout exists
    let k_pad = pairs * inn_pad; // 6144
    let cl_reach = (pairs - 1) * inn + inn_pad; // 5696
    // The pair's chain is the aligned tail's own chain — the half-stick tail
    // touches ONLY the quantize arms (everything else in the body was already
    // tail-agnostic).
    assert_eq!(
        names.iter().filter(|n| n.starts_with("rtkstg_")).count(),
        2,
        "the staging still fires (W=128 → 2 groups)"
    );
    assert!(
        names.contains(&"rtkxi_o6"),
        "the index-rows convert still mints the per-slot runs"
    );
    // EIGHT per-slot arms, slot 0 keeping the shipped name, and ONE zero-fill.
    let arms: Vec<&ktir_superdsc::emit::EmittedOp> = ops
        .iter()
        .filter(|o| o.op_name.ends_with("fq_afp8_op") || o.op_name.contains("fq_afp8_p"))
        .collect();
    assert_eq!(arms.len(), pairs as usize, "one arm per slot (k={pairs})");
    assert_eq!(
        ops.iter()
            .filter(|o| o.op_name.ends_with("fq_zfpad_op"))
            .count(),
        1,
        "exactly one pad-window zero fill (the last slot's overhang)"
    );
    // THE SEAM LAW: arm s writes `afp8` exactly where slot s's chunk-0 matmul reads its
    // A window — `col_of(m, k·inn_pad, s·inn_pad)` at both ends of the seam — and the
    // arms advance by the PADDED group-corner stride m·inn_pad B.
    let arm_out = |s: usize| operand_start_bytes(arms[s], "OUTPUT");
    for s in 0..pairs as usize {
        let mm = find_op(&ops, &format!("emmm_s{s}_c0_o5"));
        assert_eq!(
            arm_out(s),
            operand_start_bytes(mm, "INPUT"),
            "arm {s}'s afp8 window IS slot {s}'s matmul A window (one col_of law, both ends)"
        );
    }
    for s in 0..(pairs as usize - 1) {
        assert_eq!(
            arm_out(s + 1) - arm_out(s),
            m as u64 * inn_pad as u64,
            "the arms advance by the PADDED GROUP-CORNER stride m·inn_pad = {} B — the m \
             term is the §4.5 law (the flat m=1 form would say {inn_pad})",
            m * inn_pad
        );
    }
    // The chunk cut at this geometry: 4 chunks of 704 columns (the LX-fit law at
    // group_sub_rows=384), the scale gathers per (slot, group) at n=2816.
    assert_eq!(
        names.iter().filter(|n| n.starts_with("emmm_")).count(),
        (pairs * 4) as usize,
        "8 slots × 4 chunks — the chunk cut at the DOWN geometry"
    );
    assert_eq!(
        names.iter().filter(|n| n.starts_with("emsg_")).count(),
        (pairs * 44) as usize,
        "8 slots × 44 scale groups × 1 run (rows=2 under the 32-entry cap)"
    );
    // The reservations, m-generic: `cl` [m, cl_reach] fp16, `afp8` [m, k_pad] fp8.
    let s = layout.synth.borrow();
    assert_eq!(
        s.sizes.get("t4_fq_cl").copied().unwrap_or(0),
        m as u64 * cl_reach as u64 * 2,
        "cl owns [m, cl_reach] = [{m}, {cl_reach}] fp16"
    );
    assert_eq!(
        s.sizes.get("t4_fq_afp8").copied().unwrap_or(0),
        m as u64 * k_pad as u64,
        "afp8 owns [m, k_pad] = [{m}, {k_pad}] fp8 (1 B/elem)"
    );
}

/// A POINTWISE-BROADCAST op's first operand's core-0 HBM start — by POSITION,
/// not dsType: the broadcast opspec builds every operand, inputs included, with
/// `Role::Output` (pointwise.rs's `mk` closure), so ALL its labeledDs read
/// "OUTPUT" and a dsType lookup cannot find the stage's source. The emission's
/// own order — inputs first, output last (`dest_start_bytes`' `.last()` law) —
/// is the positional contract.
fn first_operand_start_bytes(o: &ktir_superdsc::emit::EmittedOp) -> u64 {
    let j = op_json(o);
    let lds = j["labeledDs_"]
        .as_array()
        .expect("labeledDs_")
        .first()
        .unwrap();
    let idx = lds["ldsIdx_"].as_u64().expect("ldsIdx_");
    let alloc = j["scheduleTree_"]
        .as_array()
        .expect("scheduleTree_")
        .iter()
        .find(|n| n["nodeType_"] == "allocate" && n["component_"] == "hbm" && n["ldsIdx_"] == idx)
        .unwrap_or_else(|| panic!("{}: the first operand's hbm allocate", o.op_name));
    alloc["startAddressCoreCorelet_"]["data_"]["[0, 0, 0]"]
        .as_str()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("{}: core 0's start address", o.op_name))
}

/// ⭐ THE STICK-MULTIPLE ROW COUNT STILL STAGES — FOR THE WRITER'S WIDTH: at
/// m=64 (m_pad = m) the ROW pad is gone, but the stage fires anyway — the
/// RouterLogits writer holds 512-wide rows at m ≥ 3 (the macs threshold the
/// router-width carve-out states: 64·128·2816 ≈ 23M ≥ 2^20, against
/// 720,896 < 2^20 at m=2), and the Kᵀ contract is w=128, so the stage's
/// job at this geometry is the WIDTH SQUEEZE: each group reads its corner
/// INSIDE the writer's wide rows. The scale gathers split into
/// `ceil(64/32) = 2` runs per (slot, group) under the vendor's
/// one-index-stick-per-op law.
#[test]
fn a_stick_multiple_row_count_stages_for_the_writer_width() {
    let (ops, layout) = through_the_door(&expert_matmul_ir(64, K, E, N_OUT, IN, ExpertProj::Gate));
    let names = op_names(&ops);
    let staging: Vec<&str> = names
        .iter()
        .copied()
        .filter(|n| n.starts_with("rtkstg_"))
        .collect();
    assert_eq!(
        staging,
        vec!["rtkstg_g0_o6", "rtkstg_g1_o6"],
        "the stage fires at m_pad == m — for the writer's width, not the row pad"
    );
    assert!(names.contains(&"rtkkt_o6"), "the Kᵀ restickify runs");
    // THE WIDTH-SQUEEZE PIN: group 1's source corner is one [rows, 64-lane]
    // stick-group past group 0's — the `col_of` group-corner stride inside
    // the writer's wide rows (the stride itself is rows·64·2 B whatever the
    // writer's width; the LAW is that the stage still runs to squeeze it).
    // Read by POSITION: a broadcast op's labeledDs are all "OUTPUT" (see
    // `first_operand_start_bytes`).
    let g0 = find_op(&ops, "rtkstg_g0_o6");
    let g1 = find_op(&ops, "rtkstg_g1_o6");
    assert_eq!(
        first_operand_start_bytes(g1) - first_operand_start_bytes(g0),
        64 * 64 * 2,
        "group 1 reads its corner one [64, 64]-lane stick-group past group 0's"
    );
    // The XP reservation exists (staged) at the Kᵀ contract [m_pad, W].
    assert_eq!(
        layout
            .synth
            .borrow()
            .sizes
            .get("t6_xp")
            .copied()
            .unwrap_or(0),
        64 * 128 * 2,
        "XP owns [m_pad, W] = [64, 128] fp16"
    );
    let gathers = names.iter().filter(|n| n.starts_with("emsg_")).count();
    assert_eq!(
        gathers,
        8 * 11 * 2,
        "8 slots × 11 groups × ceil(64/32) runs"
    );
    assert_eq!(
        names.iter().filter(|n| n.starts_with("emmm_")).count(),
        88,
        "the matmul count is row-count-independent (per slot × chunk)"
    );
}

/// ⭐ THE m=1 CROSS-CHECK: the decode regime keeps its own matmul body — the
/// COPY-gather form (`emwg_` weight gathers into `xwscratch`, pair-named
/// matmuls) and the SAME index mint as m>1. The PAIR's chain still runs (the
/// stage fires at every `m_pad ≠ m`, and the Kᵀ restickify + the pair + the
/// landings are m-generic) — and the CONVERT TRIO (`rtkfl`/`rtkxf`/`rtkxi`)
/// runs at m=1 too: it replaced the decode arm's own staged `[1, W]`
/// converts (the 26b staging defect — the halved widening + the replica-run
/// window that staged the top-2 experts ×4), so the door's index mint is
/// ONE chain at every m and nothing else names XF32/XIdx.
#[test]
fn decode_m1_keeps_the_copy_gather_body() {
    let (ops, layout) = through_the_door(&expert_matmul_ir(1, K, E, N_OUT, IN, ExpertProj::Gate));
    let names = op_names(&ops);
    assert!(
        names.iter().any(|n| n.starts_with("emwg_")),
        "the decode weight gathers still run"
    );
    assert_eq!(
        names.iter().filter(|n| n.starts_with("emmm_")).count(),
        88,
        "the decode matmuls stay pair-named and per-chunk (moe_expert_chunk_cut pins the names)"
    );
    assert!(
        names.iter().all(|n| !n.starts_with("emmm_s")),
        "the decode matmuls keep the _p(pair) spelling, not the m>1 _s(slot)"
    );
    // The pair's stage/Kᵀ/pair/landings run at m=1 too — the landing carve
    // reserves the ids and the values at `[m_pad, 64]` for exactly this.
    for name in [
        "rtkstg_g0_o6",
        "rtkstg_g1_o6",
        "rtkkt_o6",
        "rtkix_o6",
        "rtkvl_o6",
        "rtkti_o6",
        "rtktv_o6",
    ] {
        assert!(names.contains(&name), "the pair's {name} runs at m=1");
    }
    // ⭐ THE TRIO RUNS AT m=1 — the ONLY index mint at every m.
    for name in ["rtkfl_o6", "rtkxf_o6", "rtkxi_o6"] {
        assert!(
            names.contains(&name),
            "the convert trio's {name} runs at m=1 (the decode arm's staged converts are dead)"
        );
    }
    assert!(
        !names
            .iter()
            .any(|n| n.starts_with("emxf_") || n.starts_with("emidx_")),
        "NO door-local index converts — the trio owns the scratch at every m"
    );
    assert!(
        layout.synth.borrow().sizes.contains_key("t5_xwscratch"),
        "the decode weight scratch is still reserved"
    );
}

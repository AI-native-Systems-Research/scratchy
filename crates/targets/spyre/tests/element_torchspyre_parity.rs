//! 100% Element* COMPLIANCE = scratchy's emitted SDSC descriptor is BYTE-IDENTICAL to torch-spyre's.
//! The reference values are the ground truth an agent extracted directly from ~/git/torch-spyre
//! (compute_ops.py gen_coord_info_value/generate_sdsc, spyre_tensor_impl.cpp, _create_sdsc_tensors) —
//! NOT hand-invented. torch-spyre is proven on Spyre hardware, so byte-parity == correctness.
//!
//! The load-bearing facts (agent diff):
//!  • on-device layout = DENSE `device_size` packing `[feat/stk, rows, stk]` ⇒ RowBlocked strides
//!    `[rows·stk, stk, 1]`: row stride = stk(64), feat-group stride = rows·stk. The host `stride_map`
//!    row-stride=cols is HOST bookkeeping, NEVER in the descriptor.
//!  • per-core start (row-split W) = c·(rows/W)·stk elements. For [31,2048] fp16, W=31 ⇒ c·64 elems = c·128 B.
//!  • coordInfo alpha_ = iteration extents + the (stk,1) stick fold — NOT physical strides.
//!  • maxDimSizes_ = -1 for EVERY non-indirect tensor (card reconstructs device_size from N_/layout/stick).

use emit::In;
use ktir_superdsc::emit;
use scratchy_subtile::sdsc_abstract::{
    BlockCols, KernelTag, RowBlockedTag, RowCount, SpyreTensorLayout, Staged, StickKind,
    StickLayout, Stk, stage_2d,
};
use scratchy_subtile::superdsc_opspec::Df;
use scratchy_target_spyre::lower_subtile_tape_to_superdsc as superdsc;

/// Typed matmul operands — the compile-time addressing guard: a matmul's A/O are RowBlocked, W is a
/// Kernel. Passing a `&str` (or a wrong-kind handle) here is now a `cargo build` error.
fn mm_a(name: &str, m: u32, k: u32) -> Stk<RowBlockedTag> {
    Stk::<RowBlockedTag>::new(name, StickLayout::row_blocked(m as usize, k as usize)).unwrap()
}
fn mm_w(name: &str, k: u32, n: u32) -> Stk<KernelTag> {
    Stk::<KernelTag>::kernel(k as usize, n as usize, name)
}

fn dsc(op: &emit::EmittedOp, name: &str) -> serde_json::Value {
    serde_json::to_value(&op.op).unwrap()["dscs_"][0][name].clone()
}

#[test]
fn pointwise_31x2048_matches_torchspyre() {
    let (rows, cols) = (31u32, 2048u32);
    let mut sid = 0i64;
    let op = emit::assemble_pointwise_broadcast(
        "pw",
        "multiply",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(cols),
        &[
            In::full(&mm_a("x", rows, cols)).ew(),
            In::scalar(&mm_a("scale", 1, 64)).ew(),
        ],
        &mm_a("out", rows, cols),
        &mut sid,
        None,
    );
    let d = dsc(&op, "pw");
    let out_node = d["scheduleTree_"].as_array().unwrap().last().unwrap();

    // (a) primaryDsInfo_ — layoutDimOrder/stickDimOrder/stickSize (agent spec table).
    let p = &d["primaryDsInfo_"]["OUTPUT"];
    assert_eq!(
        p["layoutDimOrder_"],
        serde_json::json!(["mb", "out"]),
        "layoutDimOrder_"
    );
    assert_eq!(
        p["stickDimOrder_"],
        serde_json::json!(["out"]),
        "stickDimOrder_"
    );
    assert_eq!(p["stickSize_"], serde_json::json!([64]), "stickSize_");

    // (b) maxDimSizes_ = [-1,-1] (torch-spyre emits -1 for every non-indirect tensor).
    assert_eq!(
        out_node["maxDimSizes_"],
        serde_json::json!([-1, -1]),
        "maxDimSizes_ must be all -1"
    );

    // (c) per-core START = RowBlocked dense stride c·64 elems = c·128 bytes (NOT flat c·cols·2=4096).
    let sa = &out_node["startAddressCoreCorelet_"]["data_"];
    let a0: i64 = sa["[0, 0, 0]"].as_str().unwrap().parse().unwrap();
    let a1: i64 = sa["[1, 0, 0]"].as_str().unwrap().parse().unwrap();
    assert_eq!(
        a1 - a0,
        128,
        "per-core start delta must be rows-split RowBlocked stk*2=128 (c*64 elems), \
                              got {} (4096 = the removed flat r*cols hack)",
        a1 - a0
    );

    // (d) coordInfo — iteration folds (agent spec): mb alpha [1,0,0,1] core_fold=31; out alpha [2048,0,0,64,1].
    let ci = &out_node["coordinates_"]["coordInfo"];
    let mb_alpha: Vec<i64> = ci["mb"]["folds"]["dim_prop_func"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["Affine"]["alpha_"].as_i64().unwrap())
        .collect();
    let out_alpha: Vec<i64> = ci["out"]["folds"]["dim_prop_func"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["Affine"]["alpha_"].as_i64().unwrap())
        .collect();
    assert_eq!(
        mb_alpha,
        vec![1, 0, 0, 1],
        "coordInfo[mb] alpha_ (rows//W=1 core_fold, elem_arr_0)"
    );
    assert_eq!(
        out_alpha,
        vec![2048, 0, 0, 64, 1],
        "coordInfo[out] alpha_ (2048, .., stk=64, lane=1)"
    );
    let mb_core_fold = ci["mb"]["folds"]["dim_prop_attr"][0]["factor_"]
        .as_i64()
        .unwrap();
    assert_eq!(
        mb_core_fold, 31,
        "coordInfo[mb] core_fold factor = split W = 31"
    );
    let out_elem1 = ci["out"]["folds"]["dim_prop_attr"][3]["factor_"]
        .as_i64()
        .unwrap();
    assert_eq!(out_elem1, 32, "coordInfo[out] elem_arr_1 = cols/stk = 32");
}

fn start_delta(node: &serde_json::Value) -> i64 {
    let sa = &node["startAddressCoreCorelet_"]["data_"];
    let a0: i64 = sa["[0, 0, 0]"].as_str().unwrap().parse().unwrap();
    let a1: i64 = sa["[1, 0, 0]"].as_str().unwrap_or("0").parse().unwrap();
    a1 - a0
}

#[test]
fn matmul_31x2048x2048_matches_torchspyre() {
    // A[31,2048]@W[2048,2048]->O[31,2048], mb-split. Agent spec (section c):
    //  INPUT  layoutDimOrder ["mb","in"], stick ["in"];  KERNEL ["in","out"], stick ["out"];
    //  OUTPUT ["mb","out"], stick ["out"]; ALL maxDimSizes_ = -1; INPUT/OUTPUT start = RowBlocked c*64=128 B.
    //
    // The split is pinned EXPLICITLY ({"mb":31}) rather than taken from the live cost model. This
    // test exists to lock the mb-split per-core START bytes against torch-spyre, so it must emit an
    // mb split regardless of what the cost model currently prefers. It previously called
    // `assemble_matmul_seeded`, which runs the real model -- and once the reduction-dim K-split was
    // re-enabled the model started choosing {out:8,in:4} for this shape, so the assertions below
    // were measuring an IN-split delta (31,744 B = (512/64)*(31*64) elems, core 1's K-slice corner)
    // instead of the mb delta they name. Pinning the splitter keeps this a test of mb addressing.
    let mut sid = 0i64;
    let mm = superdsc::assemble_matmul_split(
        "mm",
        31,
        2048,
        2048,
        1,
        "a",
        "w",
        "o",
        &mut sid,
        None,
        |_dims, _max_cores| {
            let mut s = std::collections::BTreeMap::new();
            s.insert("mb", 31u32);
            s
        },
    )
    .expect("explicit mb-split matmul must assemble");
    let d = dsc(&mm, "mm");
    // maxDimSizes_ = -1 on EVERY node (input, kernel, output).
    for node in d["scheduleTree_"].as_array().unwrap() {
        assert!(
            node["maxDimSizes_"]
                .as_array()
                .unwrap()
                .iter()
                .all(|v| v.as_i64() == Some(-1)),
            "matmul node {} maxDimSizes_ must be all -1, got {}",
            node["name_"],
            node["maxDimSizes_"]
        );
    }
    // primaryDsInfo_ layout per role (torch-spyre matmul dim labels [mb,in,out]).
    let p = &d["primaryDsInfo_"];
    assert_eq!(
        p["INPUT"]["layoutDimOrder_"],
        serde_json::json!(["mb", "in"]),
        "INPUT layoutDimOrder_"
    );
    assert_eq!(
        p["INPUT"]["stickDimOrder_"],
        serde_json::json!(["in"]),
        "INPUT stick = in (contracted axis)"
    );
    assert_eq!(
        p["KERNEL"]["layoutDimOrder_"],
        serde_json::json!(["in", "out"]),
        "KERNEL layoutDimOrder_"
    );
    assert_eq!(
        p["KERNEL"]["stickDimOrder_"],
        serde_json::json!(["out"]),
        "KERNEL stick = out (N)"
    );
    assert_eq!(
        p["OUTPUT"]["layoutDimOrder_"],
        serde_json::json!(["mb", "out"]),
        "OUTPUT layoutDimOrder_"
    );
    assert_eq!(
        p["OUTPUT"]["stickDimOrder_"],
        serde_json::json!(["out"]),
        "OUTPUT stick = out (N)"
    );
    // INPUT + OUTPUT per-core start = RowBlocked dense c*stk*2 = 128 B (mb-split); NOT flat c*cols*2.
    let tree = d["scheduleTree_"].as_array().unwrap();
    let input = &tree[0]; // Tensor0 = A (input)
    let output = tree.last().unwrap(); // last = O (output)
    assert_eq!(
        start_delta(input),
        128,
        "INPUT mb-split start must be RowBlocked c*64 elems = 128 B"
    );
    assert_eq!(
        start_delta(output),
        128,
        "OUTPUT mb-split start must be RowBlocked c*64 elems = 128 B"
    );
}

#[test]
fn staged_ctors_place_bytes_at_dev_off() {
    // The TYPE-LEVEL firewall's constructors (Staged::tiled/blocks) place bytes at the emit's dev_off, and
    // into_bytes() is the only exit — so a worker acts.push MUST go through a StickLayout (no raw Vec).
    let layout = StickLayout::for_view_df(&[19, 128], 1, Df::Fp16);
    let bytes = Staged::tiled(&layout, |r, c| (r * 128 + c + 1) as f32).into_bytes();
    for r in 0..19 {
        for c in 0..128 {
            assert_eq!(
                bytes[layout.dev_off(r, c)],
                (r * 128 + c + 1) as f32,
                "tiled at dev_off"
            );
        }
    }
    // blocks: 3 stacked RowBlocked [8,128] head-blocks (the selector shape).
    let blk = StickLayout::for_view_df(&[8, 128], 1, Df::Fp16);
    let bsz = 8 * 128;
    let sb = Staged::blocks(&blk, 3, |b, r, c| (b * 100000 + r * 128 + c + 1) as f32).into_bytes();
    for b in 0..3 {
        for r in 0..8 {
            for c in 0..128 {
                assert_eq!(
                    sb[b * bsz + blk.dev_off(r, c)],
                    (b * 100000 + r * 128 + c + 1) as f32,
                    "block {b} ({r},{c}) at b*block + dev_off"
                );
            }
        }
    }
    // filled is layout-invariant.
    assert_eq!(Staged::filled(0.5, 64).into_bytes(), vec![0.5f32; 64]);
}

#[test]
fn stage_2d_is_the_worker_emitter_layout_firewall() {
    // The worker↔emitter mismatch firewall (the bug: worker staged embedding FLAT, emit read RowBlocked).
    // Prove the emit classifies the residual as RowBlocked AND stage_2d places every logical (r,c) exactly
    // where the emit's dev_off(r,c) reads — so a worker routing through stage_2d(&for_view_df(...)) is
    // BYTE-CONSISTENT with the emit by construction.
    for &(rows, cols) in &[(31usize, 2048usize), (19, 128)] {
        let emit = StickLayout::for_view_df(&[rows, cols], 1, Df::Fp16);
        assert_eq!(
            emit,
            StickLayout::row_blocked(rows, cols),
            "residual [{rows},{cols}] is RowBlocked"
        );
        let staged = stage_2d(&emit, |r, c| (r * cols + c + 1) as f32);
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(
                    staged[emit.dev_off(r, c)],
                    (r * cols + c + 1) as f32,
                    "stage_2d({r},{c}) must land at emit dev_off({r},{c}) — else worker/emit disagree"
                );
            }
        }
    }
}

#[test]
fn per_core_start_derives_from_spyre_layout() {
    // THE SINGLE-SOURCE PROOF. The emitted per-core START must equal the SpyreTensorLayout's DENSE row
    // stride (torch-spyre `_calculate_device_stride`), so per_core_addr is provably derived from the ONE
    // SpyreTensorLayout — not an independent flat/RowBlocked choice that could silently diverge.
    let stl = SpyreTensorLayout::standard(&[31, 2048], Df::Fp16);
    assert_eq!(
        stl.device_size,
        vec![32, 31, 64],
        "SpyreTensorLayout device_size"
    );
    let row_stride_elems = stl.dense_strides()[stl.row_device_dim()]; // dense row stride = 64
    assert_eq!(
        row_stride_elems, 64,
        "dense row stride (RowBlocked) = 64, NOT cols=2048"
    );
    let want_delta_bytes = row_stride_elems * 2; // fp16 = 2 bytes/elem

    let (rows, cols) = (31u32, 2048u32);
    let mut sid = 0i64;
    let op = emit::assemble_pointwise_broadcast(
        "pw",
        "multiply",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(cols),
        &[
            In::full(&mm_a("x", rows, cols)).ew(),
            In::scalar(&mm_a("scale", 1, 64)).ew(),
        ],
        &mm_a("out", rows, cols),
        &mut sid,
        None,
    );
    let out_node = dsc(&op, "pw")["scheduleTree_"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()
        .clone();
    assert_eq!(
        start_delta(&out_node),
        want_delta_bytes,
        "per-core start delta {} must equal SpyreTensorLayout dense row stride {} elems * 2 = {} B — \
         proves the emitted start derives from the single SpyreTensorLayout (RowBlocked), not the removed \
         flat r*cols hack (which would give {} B)",
        start_delta(&out_node),
        row_stride_elems,
        want_delta_bytes,
        cols * 2
    );
}

#[test]
fn coordinfo_reconstructs_spyre_device_size() {
    // SINGLE-SOURCE PROOF #2: build_coordinates' fold factors must reconstruct the SAME device_size the
    // SpyreTensorLayout holds — so the coordInfo (path 2) derives from the one layout, not independently.
    // device_size [feat-groups, rows, lanes] = [32, 31, 64] for [31,2048] fp16.
    let stl = SpyreTensorLayout::standard(&[31, 2048], Df::Fp16);
    let ds = &stl.device_size;
    let (rows, cols) = (31u32, 2048u32);
    let mut sid = 0i64;
    let op = emit::assemble_pointwise_broadcast(
        "pw",
        "multiply",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(cols),
        &[
            In::full(&mm_a("x", rows, cols)).ew(),
            In::scalar(&mm_a("scale", 1, 64)).ew(),
        ],
        &mm_a("out", rows, cols),
        &mut sid,
        None,
    );
    let out_node = dsc(&op, "pw")["scheduleTree_"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()
        .clone();
    let ci = &out_node["coordinates_"]["coordInfo"];
    let attr = |dim: &str| -> Vec<i64> {
        ci[dim]["folds"]["dim_prop_attr"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a["factor_"].as_i64().unwrap())
            .collect()
    };
    // OUT (stick) dim: elem_arr_1 = feat-groups = device_size[0]; elem_arr_0 = lanes = device_size[2].
    let out = attr("out"); // [core_fold, corelet, row_fold, elem_arr_1, elem_arr_0]
    assert_eq!(
        out[3], ds[0],
        "coordInfo[out] elem_arr_1 must == device_size[feat-groups]={}",
        ds[0]
    );
    assert_eq!(
        out[4], ds[2],
        "coordInfo[out] elem_arr_0 must == device_size[lanes]={}",
        ds[2]
    );
    // MB (row) dim: core_fold * elem_arr_0 = total rows = device_size[1].
    let mb = attr("mb"); // [core_fold, corelet, row_fold, elem_arr_0]
    assert_eq!(
        mb[0] * mb[3],
        ds[1],
        "coordInfo[mb] core_fold*elem_arr_0 must == device_size[rows]={}",
        ds[1]
    );
}

#[test]
fn arrangement_is_rank_independent() {
    // THE fix (task #7): a tensor's arrangement must NOT depend on an op's phantom unit dims. A pointwise
    // view [mb,out,1] and a matmul view [m,k] of the SAME tensor must classify to the SAME device layout —
    // else per_core_addr's start (rank-3 → Flat) disagrees with the descriptor's stick (RowBlocked): the
    // M>1 scramble. for_view_df folds trailing unit dims, so both are RowBlocked and addr_eq.
    let rank2 = StickLayout::for_view_df(&[31, 2048], 1, Df::Fp16);
    let rank3 = StickLayout::for_view_df(&[31, 2048, 1], 1, Df::Fp16);
    assert_eq!(
        rank2.kind,
        StickKind::RowBlocked,
        "rank-2 [mb,out] is RowBlocked"
    );
    assert_eq!(
        rank3.kind,
        StickKind::RowBlocked,
        "rank-3 [mb,out,1] must ALSO be RowBlocked (fold the phantom y) — the whole bug was that it was Flat"
    );
    assert!(
        rank2.addr_eq(&rank3),
        "the two views of one tensor must address identically"
    );
    // multiple trailing units fold too; a GENUINE non-unit trailing dim (head-major) stays Flat.
    assert_eq!(
        StickLayout::for_view_df(&[31, 2048, 1, 1], 1, Df::Fp16).kind,
        StickKind::RowBlocked
    );
    assert_eq!(
        StickLayout::for_view_df(&[8, 31, 64], 2, Df::Fp16).kind,
        StickKind::Flat,
        "a genuine [heads,mb,hd] head-major tensor is untouched (stays Flat)"
    );
}

#[test]
fn addr_eq_catches_only_the_m_gt_1_dense_vs_stick_scramble() {
    // rows==1 (decode): RowBlocked and Flat COINCIDE ⇒ addr_eq true (no false conflict on the working path).
    assert!(
        StickLayout::row_blocked(1, 2048).addr_eq(&StickLayout::flat(1, 2048)),
        "rows==1: RowBlocked==Flat (decode is byte-identical either way)"
    );
    // single stick (cols<=lanes): coincide regardless of rows.
    assert!(
        StickLayout::row_blocked(31, 64).addr_eq(&StickLayout::flat(31, 64)),
        "single stick coincides"
    );
    // rows>1 AND cols>lanes: they DIVERGE ⇒ addr_eq false (the exact scramble the authority must catch).
    assert!(
        !StickLayout::row_blocked(31, 2048).addr_eq(&StickLayout::flat(31, 2048)),
        "rows>1 cols>lane: RowBlocked != Flat (the M>1 scramble)"
    );
    // different shape never matches.
    assert!(!StickLayout::row_blocked(31, 2048).addr_eq(&StickLayout::row_blocked(19, 2048)));
    // RESHAPE (the granite `t729` false-positive the authority hit): a [1,2048] activation and its
    // [32,64] head-major view (cols==64==lanes) are BYTE-IDENTICAL (both map element k to byte k).
    assert!(
        StickLayout::row_blocked(1, 2048).addr_eq(&StickLayout::row_blocked(32, 64)),
        "[1,2048] flat == [32,64] single-stick head-major — a legitimate reshape, not a conflict"
    );
    // The head-major expansion holds at m>1 TOO, and that is a law, not a leak: `addr_eq`'s own proof
    // (`HEAD-MAJOR ROW EXPANSION`, Kani `row_expansion_is_byte_identical`) shows `[m, H·L]` and
    // `[H·m, L]` are the same bytes for every m — which is exactly what lets a `[mq, heads·hd]` Q/K
    // tensor and its per-head-block view be ONE declared arrangement instead of a conflict.
    assert!(
        StickLayout::row_blocked(8, 2048).addr_eq(&StickLayout::row_blocked(8 * 32, 64)),
        "m>1: [8,2048] and [256,64] are both stick-blocked — the expansion is byte-identical"
    );
    // What the authority MUST still catch at m>1 is DENSE-vs-STICK: a `Flat` side really is row-major,
    // so it diverges from the stick-blocked one no matter which shape it is written in.
    assert!(
        !StickLayout::row_blocked(8, 2048).addr_eq(&StickLayout::flat(8, 2048)),
        "m>1: RowBlocked vs Flat is the scramble — a conflict"
    );
    // And the CONTIGUOUS spellings all agree with each other: a row-major `Flat [8,2048]` and a
    // single-stick `RowBlocked [256,64]` both place element k at byte k. So the stick-blocked
    // `[8,2048]` is the ONE layout in this family that differs — which is precisely why declaring an
    // activation's arrangement is load-bearing at m>1 and free at m==1.
    assert!(
        StickLayout::flat(8, 2048).addr_eq(&StickLayout::row_blocked(8 * 32, 64)),
        "m>1: Flat [8,2048] and single-stick [256,64] are both contiguous — element k at byte k"
    );
}

#[test]
fn arrangement_authority_rejects_producer_consumer_disagreement() {
    // THE LOCK (task #6): ONE device layout per tensor. First declaration wins; a later NON-equivalent
    // declaration for the SAME tensor is a build Err naming it — so the M>1 Dense-vs-stick scramble is a
    // COMPILE-TIME failure, not silent on-card garbage. This is the type-system replacement for the
    // scattered per-op row_blocked/stickmajor knobs.
    let bl = superdsc::BundleLayout::default();
    // producer writes 'resid' RowBlocked [31,2048]; a second RowBlocked access is the canonical (Ok).
    assert!(
        bl.declare_arrangement("resid", StickLayout::row_blocked(31, 2048))
            .is_ok()
    );
    assert!(
        bl.declare_arrangement("resid", StickLayout::row_blocked(31, 2048))
            .is_ok()
    );
    // a consumer reading it FLAT at rows>1 is a hard Err that names the tensor.
    let bad = bl.declare_arrangement("resid", StickLayout::flat(31, 2048));
    assert!(
        bad.is_err(),
        "reading 'resid' Flat after it was written RowBlocked (rows>1) MUST be a build Err"
    );
    assert!(
        bad.unwrap_err().0.contains("resid"),
        "the arrangement-conflict error must name the tensor"
    );
    // rows==1 decode: RowBlocked then Flat is NOT a conflict (they address identically) — working path safe.
    let bl1 = superdsc::BundleLayout::default();
    assert!(
        bl1.declare_arrangement("dec", StickLayout::row_blocked(1, 2048))
            .is_ok()
    );
    assert!(
        bl1.declare_arrangement("dec", StickLayout::flat(1, 2048))
            .is_ok(),
        "rows==1 RowBlocked/Flat is not a conflict — the authority must not false-fire on decode"
    );
}

#[test]
fn decode_matmul_mb1_also_minus_one() {
    // Decode (mb==1) matmul must ALSO emit -1 (full parity — no mb==1 actual-extents exception).
    let mut sid = 0i64;
    let mm = superdsc::assemble_matmul_seeded(
        "dm",
        1,
        2048,
        2048,
        1,
        &mm_a("a", 1, 2048),
        &mm_w("w", 2048, 2048),
        &mm_a("o", 1, 2048),
        &mut sid,
        None,
    );
    let d = dsc(&mm, "dm");
    for node in d["scheduleTree_"].as_array().unwrap() {
        assert!(
            node["maxDimSizes_"]
                .as_array()
                .unwrap()
                .iter()
                .all(|v| v.as_i64() == Some(-1)),
            "decode matmul (mb=1) node {} maxDimSizes_ must be all -1 (parity), got {}",
            node["name_"],
            node["maxDimSizes_"]
        );
    }
}

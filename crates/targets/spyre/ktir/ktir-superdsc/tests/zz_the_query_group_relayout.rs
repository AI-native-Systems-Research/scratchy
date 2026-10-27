// SPDX-License-Identifier: Apache-2.0
//! ⭐ THE ONE-REQUEST DECODE's QUERY, RE-LAID GROUP-MAJOR ABOVE ONE FEATURE SLAB — the copy the hd=128
//! chunk needs before it can attend a kv head's whole GQA group in one score (`QGroupBlock`).
//!
//! The token stream `[1, nqh·hd]` puts query head `h`'s slab `s` at plane `h·nslab + s`, so a group's
//! `gqa` rows are `nslab` planes apart. The re-layout is ONE identity-kernel matmul per (kv head, slab):
//! `y` walks the group's heads (`nslab` planes apart in the stream, one row apart in the block), `mb` is
//! the single row, and the 64×64 identity copies one stick. That is the KV cache write's own descriptor
//! shape (`cachewr_kcs{s}`: `y` = the kv heads of one slab, one stick in and out, head-outermost) with
//! the destination a standalone `[gqa, hd]` block instead of a page plane.
//!
//! What these tests pin, from the EMITTED descriptors rather than from the law that produced them:
//!   * the restickify door cannot express this copy at all (the reason the copy is a matmul);
//!   * every element of every query head lands exactly once, at the RowBlocked address the score reads
//!     it from — a permutation of the 4096 query elements, evaluated through each op's own per-core
//!     starts, `maxDimSizes_` walk and `N_`;
//!   * the op count, the walk order and the core split.
//!
//! Set `QGROUP_PROBE_DIR=<dir>` to also write one copy's dxp input dir (`sdsc_0.json` + `bundle.mlir`)
//! for `dxp_standalone -d <dir> -b sentient`.

use ktir_superdsc::addr::DevOff;
use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{SharedKernelBmmForm, assemble_matmul_placed};
use ktir_superdsc::sdsc_abstract::{
    KernelTag, Lanes, MatK, MatM, MatN, MatY, QGroupBlock, QueryRowCount, RowBlockedTag,
    StickLayout, Stk,
};

/// granite-3.1-8b's attention geometry.
const NQH: u32 = 32;
const NKVH: u32 = 8;
const GQA: u32 = NQH / NKVH;
const HD: u32 = 128;
const STICK: u32 = 64;
const NSLAB: u32 = HD / STICK;

fn rbh(name: &str, rows: u32, cols: u32) -> Stk<RowBlockedTag> {
    Stk::<RowBlockedTag>::new(name, StickLayout::row_blocked(rows as usize, cols as usize))
        .expect("RowBlocked by construction")
}

fn qg() -> QGroupBlock {
    QGroupBlock::of_decode(NKVH, GQA, HD, 1).expect("granite-8b's one-request decode re-lays")
}

/// The re-layout's ops, kv-head-major then slab — exactly what the chunk would emit.
fn relayout_ops() -> Vec<EmittedOp> {
    let q = qg();
    ops_with(|kvh, s| q.copy_dest(kvh, s).expect("in range"))
}

/// The re-layout with the DESTINATION placement chosen per (kv head, slab) — the real law, or a control.
fn ops_with(
    dest: impl Fn(u32, u32) -> ktir_superdsc::sdsc_abstract::OperandPlacement,
) -> Vec<EmittedOp> {
    let q = qg();
    let mut sym = 0i64;
    let mut ops = Vec::new();
    for kvh in 0..NKVH {
        for s in 0..NSLAB {
            let src = q.copy_src(kvh, s).expect("in range");
            let dst = dest(kvh, s);
            ops.push(assemble_matmul_placed(
                &format!("attn_cqg_g{kvh}s{s}_o0"),
                MatM::of_query_rows(QueryRowCount::of_mq(1)),
                MatN::one_stick(Lanes::FP16),
                MatK::one_stick(Lanes::FP16),
                MatY::of_gqa_group(GQA, src, dst),
                SharedKernelBmmForm::of_query_group_copy(),
                &rbh("t_qs", 1, NQH * HD),
                src,
                // I[64,64] at offset 0 of the class's `[hd, hd]` identity — the cache write's kernel.
                &Stk::<KernelTag>::one_stick_identity("t_ident"),
                DevOff::ZERO,
                &rbh("t_qg", NKVH * GQA, HD),
                dst,
                &mut sym,
                None,
            ));
        }
    }
    ops
}

fn body(e: &EmittedOp) -> (serde_json::Value, serde_json::Value) {
    let v = serde_json::to_value(&e.op).unwrap();
    let b = v["dscs_"][0]
        .as_object()
        .and_then(|m| m.values().next().cloned())
        .expect("one named dsc");
    (v, b)
}

fn sizes(v: &serde_json::Value) -> Vec<i64> {
    v.as_array()
        .map(|a| a.iter().map(|x| x.as_i64().unwrap_or(0)).collect())
        .unwrap_or_default()
}

fn strs(v: &serde_json::Value) -> Vec<String> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|x| x.as_str().unwrap_or("?").to_string())
                .collect()
        })
        .unwrap_or_default()
}

/// Element offset (fp16, within the operand's own segment) of each core's start.
fn starts(node: &serde_json::Value) -> Vec<(u32, u64)> {
    const SEG: u64 = 1u64 << 34;
    let m = node["startAddressCoreCorelet_"]["data_"]
        .as_object()
        .expect("a start-address fold");
    let mut out: Vec<(u32, u64)> = m
        .iter()
        .map(|(k, v)| {
            let core: u32 = k
                .trim_start_matches('[')
                .split(',')
                .next()
                .and_then(|c| c.trim().parse().ok())
                .expect("core key");
            let a: u64 = v.as_str().and_then(|s| s.parse().ok()).expect("address");
            (core, (a % SEG) / 2)
        })
        .collect();
    out.sort();
    out
}

/// The row-major walk a pinned operand declares, or `None` for a reconstructed (`-1`) one.
fn pinned(node: &serde_json::Value) -> Option<Vec<i64>> {
    let m = sizes(&node["maxDimSizes_"]);
    (!m.is_empty() && m.iter().all(|&d| d > 0)).then_some(m)
}

/// Element offset of position `pos` (in `layoutDimOrder_` order) inside one core's tile of an operand,
/// through the walk the descriptor declares: row-major over `maxDimSizes_` when pinned; stick-blocked
/// over the op's own extents when reconstructed. The stick axis is last in every matmul walk.
fn walk_off(node: &serde_json::Value, n_: &serde_json::Value, pos: &[u64]) -> u64 {
    let order = strs(&node["layoutDimOrder_"]);
    let ext: Vec<u64> = match pinned(node) {
        Some(p) => p.iter().map(|&d| d as u64).collect(),
        None => order
            .iter()
            .map(|d| n_[format!("{d}_")].as_u64().expect("an iterated dim"))
            .collect(),
    };
    if pinned(node).is_some() {
        let mut off = 0u64;
        for (i, &p) in pos.iter().enumerate() {
            off = off * ext[i] + p;
        }
        off
    } else {
        // Stick-blocked: `[sticks, rows, 64]` with every non-stick axis folded into rows, outermost first.
        let lead = &ext[..ext.len() - 1];
        let rows: u64 = lead.iter().product();
        let mut r = 0u64;
        for (i, &e) in lead.iter().enumerate() {
            r = r * e + pos[i];
        }
        let c = pos[lead.len()];
        (c / 64) * rows * 64 + r * 64 + c % 64
    }
}

/// ⛔ THE RESTICKIFY DOOR CANNOT DO THIS COPY, and that is why it is a matmul. A restickify moves the
/// STICK AXIS of one tensor (`[a, b]` sticked on `b` ⇒ sticked on `a`) and is otherwise the identity on
/// coordinates; the group re-layout keeps the stick axis (features) and PERMUTES the stick planes
/// (plane `h·nslab + s` of the stream ⇒ plane `s` of block `h/gqa` at row `h % gqa`). No restickify
/// descriptor's input and output stick on the same axis — the restickify DDL refuses that binding
/// (the two ports' stick dims must be DISTINCT actual dims; the flat-restickify probe's "silent"
/// control measured that refusal at `ddl_conversion.cpp:2539`).
#[test]
fn the_relayout_keeps_the_stick_axis_so_no_restickify_can_express_it() {
    // In the stream, element (h, f) sits on stick plane h·nslab + f/64, lane f%64; in the block it sits on
    // plane f/64 of block h/gqa, row h%gqa, lane f%64. The LANE never moves — the stick axis is the same
    // feature axis on both sides — so the copy is a plane permutation, not a restickify.
    for h in 0..NQH {
        for f in [0, 63, 64, 127] {
            let src = (h * NSLAB + f / STICK) * STICK + f % STICK;
            let dst =
                (h / GQA) * GQA * HD + (f / STICK) * GQA * STICK + (h % GQA) * STICK + f % STICK;
            assert_eq!(src % STICK, dst % STICK, "lane of (h {h}, f {f})");
        }
    }
}

/// ⭐⭐⭐ THE PERMUTATION PROOF: every query element lands exactly once, where the score reads it.
///
/// Each op is evaluated from its OWN emitted descriptor — the per-core starts, the declared walk
/// (`layoutDimOrder_`, `maxDimSizes_`) and `N_` — over every (core, y, mb, k) it covers: the input
/// address names the stream element `(h, f)` (decoded through the stream's own RowBlocked law) and the
/// output address names the block element. The test asserts the map is a BIJECTION over all
/// `nqh·hd = 4096` elements onto the `[nkvh][gqa, hd]` block, and that each lands at the RowBlocked
/// address `(row h%gqa, feat f)` of block `h/gqa` — the address the per-kv-head score reads.
#[test]
fn the_relayout_is_a_permutation_of_every_query_element_onto_its_group_block() {
    let ops = relayout_ops();
    assert_eq!(ops.len() as u32, NKVH * NSLAB, "one op per (kv head, slab)");
    check_permutation(&ops).unwrap_or_else(|e| panic!("{e}"));
}

/// ⛔ THE CHECK MUST BE ABLE TO FAIL, so two plausible wrong copies are run through it and must be refused:
///   * the slabs swapped at the destination — every element written once, to the wrong plane;
///   * the destination framed as the STREAM itself (no re-layout at all) — the copy that does nothing,
///     which writes each group's rows `nslab` planes apart, past the `[gqa, hd]` block they belong to.
#[test]
fn the_permutation_check_refuses_a_slab_swap_and_a_copy_that_does_not_relayout() {
    let q = qg();
    let swapped = ops_with(|kvh, s| q.copy_dest(kvh, NSLAB - 1 - s).expect("in range"));
    let e = check_permutation(&swapped).expect_err("a slab-swapped destination must be refused");
    assert!(
        e.contains("the score reads"),
        "slab swap refused for the wrong reason: {e}"
    );
    let stream = ops_with(|kvh, s| q.copy_src(kvh, s).expect("in range"));
    assert!(
        check_permutation(&stream).is_err(),
        "a destination framed as the token stream must be refused"
    );
}

/// The evaluator behind the permutation proof (see the test above): `Ok` iff every emitted op, walked
/// from its own descriptor, writes each query element `(h, f)` exactly once at the RowBlocked address
/// `(row h%gqa, feat f)` of block `h/gqa`, and every block element is written.
fn check_permutation(ops: &[EmittedOp]) -> Result<(), String> {
    let block_elems = (GQA * HD) as u64;
    let mut landed = vec![None::<(u32, u32)>; (NKVH * GQA * HD) as usize];
    for e in ops {
        let (v, b) = body(e);
        let n_ = &b["N_"];
        let (y, mb, out) = (
            n_["y_"].as_u64().unwrap_or(1),
            n_["mb_"].as_u64().unwrap_or(1),
            n_["out_"].as_u64().unwrap_or(1),
        );
        if (y, mb, out) != (GQA as u64, 1, STICK as u64) {
            return Err(format!("{}: N_ = {n_}", e.op_name));
        }
        let tree = b["scheduleTree_"].as_array().expect("schedule tree");
        let (inp, outp) = (&tree[0], &tree[2]);
        let split = |d: &str| v["numWkSlicesPerDim_"][d].as_u64().unwrap_or(1);
        if split("in") != 1 {
            return Err(format!("{}: the one-stick contraction is split", e.op_name));
        }
        // Each core's tile, walked from its own start on both operands (the folds key cores alike).
        let (pc_y, pc_m, pc_o) = (y / split("y"), mb / split("mb"), out / split("out"));
        let (in_starts, out_starts) = (starts(inp), starts(outp));
        if in_starts.len() != out_starts.len() {
            return Err(format!(
                "{}: the two operands' folds differ in length",
                e.op_name
            ));
        }
        for ((c_in, a0), (c_out, o0)) in in_starts.iter().zip(&out_starts) {
            if c_in != c_out {
                return Err(format!("{}: core {c_in} vs {c_out}", e.op_name));
            }
            for yy in 0..pc_y {
                for mm in 0..pc_m {
                    for oo in 0..pc_o {
                        // The INPUT walk is `[y, mb, in]`; through the identity kernel, out lane `oo`
                        // reads in lane `oo`.
                        let a = a0 + walk_off(inp, n_, &[yy, mm, oo]);
                        let o = o0 + walk_off(outp, n_, &[yy, mm, oo]);
                        // The stream `[1, nqh·hd]` is one RowBlocked row, so its element offset IS the
                        // feature column `h·hd + f`.
                        let (h, f) = ((a / HD as u64) as u32, (a % HD as u64) as u32);
                        if h >= NQH {
                            return Err(format!("{}: input {a} past the stream", e.op_name));
                        }
                        // ⭐ WHERE THE SCORE READS (h, f): block h/gqa, RowBlocked `[gqa, hd]`, row h%gqa.
                        let want = (h / GQA) as u64 * block_elems
                            + (f / STICK) as u64 * (GQA * STICK) as u64
                            + (h % GQA) as u64 * STICK as u64
                            + (f % STICK) as u64;
                        if o != want {
                            return Err(format!(
                                "{}: query (h {h}, f {f}) lands at {o}, the score reads {want}",
                                e.op_name
                            ));
                        }
                        if landed[o as usize].replace((h, f)).is_some() {
                            return Err(format!("{}: block element {o} written twice", e.op_name));
                        }
                    }
                }
            }
        }
    }
    let missed = landed.iter().filter(|x| x.is_none()).count();
    if missed > 0 {
        return Err(format!(
            "{missed} of {} block elements never written",
            landed.len()
        ));
    }
    Ok(())
}

/// The copy is the KV cache write's own descriptor shape: head-outermost `[y, mb, ·]` on both
/// operands, one stick in and out, `y` the GQA group.
#[test]
fn the_relayout_is_the_cache_writes_descriptor_shape() {
    for e in relayout_ops() {
        let (v, b) = body(&e);
        let tree = b["scheduleTree_"].as_array().unwrap();
        assert_eq!(strs(&tree[0]["layoutDimOrder_"]), ["y", "mb", "in"]);
        assert_eq!(strs(&tree[1]["layoutDimOrder_"]), ["in", "out"]);
        assert_eq!(strs(&tree[2]["layoutDimOrder_"]), ["y", "mb", "out"]);
        println!(
            "{:22} split={} cores={} in.max={:?} out.max={:?}",
            e.op_name,
            v["numWkSlicesPerDim_"],
            v["numCoresUsed_"],
            sizes(&tree[0]["maxDimSizes_"]),
            sizes(&tree[2]["maxDimSizes_"]),
        );
    }
    if let Ok(dir) = std::env::var("QGROUP_PROBE_DIR") {
        let ops = relayout_ops();
        let one = &ops[ops.len() - 1];
        let key = format!("0_{}", one.op_name);
        let json = serde_json::to_string(&std::collections::BTreeMap::from([(
            key.as_str(),
            one.op.as_ref().unwrap(),
        )]))
        .unwrap();
        let p = std::path::Path::new(&dir);
        std::fs::create_dir_all(p).unwrap();
        std::fs::write(p.join("sdsc_0.json"), json).unwrap();
        std::fs::write(
            p.join("bundle.mlir"),
            "module {\n  func.func @sdsc_bundle() {\n    sdscbundle.sdsc_execute () {sdsc_filename=\"sdsc_0.json\"}\n    return\n  }\n}\n",
        )
        .unwrap();
        println!("wrote {}", p.display());
    }
}

/// The witness admits exactly the one-request decode above one slab.
#[test]
fn the_group_block_is_minted_only_for_a_one_request_decode_above_one_slab() {
    assert!(QGroupBlock::of_decode(NKVH, GQA, 128, 1).is_some());
    assert!(
        QGroupBlock::of_decode(NKVH, GQA, 64, 1).is_none(),
        "one slab: the stream has it"
    );
    assert!(
        QGroupBlock::of_decode(NKVH, GQA, 128, 2).is_none(),
        "a batch or prompt chunk"
    );
    assert!(
        QGroupBlock::of_decode(NKVH, GQA, 96, 1).is_none(),
        "not whole sticks"
    );
}

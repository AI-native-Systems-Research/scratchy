// SPDX-License-Identifier: Apache-2.0
//! ⭐ A ROW BAND OF A TALLER PARENT STATES THE PARENT'S PITCH AS THE WALK'S BACK GAP.
//!
//! A pointwise op over rows `[j·r, (j+1)·r)` of a stick-major `[p, c]` parent must stride the
//! parent's stick-column planes at `p·64`, not at the band's own rows.
//! `maxDimSizes_` is a size cap dxp never reads as a stride (`dsc/dsc2.cpp:2806-2815`), so the
//! pitch is `backGapCore_[mb] = p − r` on every banded HBM operand — the gap past the band's WHOLE
//! sweep (`N_.mb_`), because an HBM view's capacity is `N_`, core fold included
//! (`dsc/dsc2.cpp:3626-3632`). MEASURED on the card (microbenchmark, 2026-11-08, branch
//! `tp-allreduce` `0e55fb8ab1` / `5d64673767`): the pinned walk placed stick `j` at `j·r·64`
//! ([1,4096] of [4,4096], [1,8192] of [4,8192], [16,384] of [64,384]), and this gap read and wrote
//! them bit-exact; a per-core gap (`p − rows per core`) read the gap fill wherever the cores split
//! the swept axis. The output (a whole tensor) carries none, and no operand's walk is pinned to
//! the parent.
//!
//! ONE LAW AT EVERY WIDTH: a band one stick wide has no second column plane for its gap to move,
//! so it states the same gap at the same addresses.
//!
//! Ported from `tp-allreduce` (`0e55fb8ab1`, `76d187281c`, `5d64673767`) onto this tree's door.
//! The cores a band spends are the byte law's here — a band's stride is its back gap, never its
//! budget — so the cases are the branch's own, which reach 1, 8 and 32 cores under that law too.

use ktir_superdsc::collective::TpWorld;
use ktir_superdsc::emit::lower_ktir_to_superdsc::{Region, elementwise};
use ktir_superdsc::ktir_node::Elementwise;
use ktir_superdsc::place::{PlaceId, act_name};
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};

type W = TpWorld<1>;

const PARENT: u32 = 10;
const OUT: u32 = 11;

fn layout(p: u32, r: u32, c: u32) -> BundleLayout {
    let mut l = BundleLayout::default();
    let mut off = 0u64;
    for (tid, rows) in [(PARENT, p), (OUT, r)] {
        let size = u64::from(rows) * u64::from(c) * 2;
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: 3,
                bank: 0,
                offset: off,
                size,
            },
        );
        off += size.next_multiple_of(128);
    }
    l.segment_bytes[3] = off;
    l
}

/// The `add` of bands 0 and 1 of the parent into a whole `[r, c]` output; returns each operand's
/// allocate node, `numCoresUsed_` and the band's whole row sweep (`N_.mb_`).
fn band_add(p: u32, r: u32, c: u32) -> (Vec<serde_json::Value>, u64, u64) {
    let l = layout(p, r, c);
    let band = |j: u32| Region {
        tid: PARENT,
        v_rows: p,
        v_cols: c,
        r_start: j * r,
        r_len: r,
        r_cover: (j * r, (j + 1) * r),
        c_start: 0,
        c_len: c,
        c_cover: (0, c),
        is_out: false,
        is_fp8: false,
        band: None,
    };
    let o = Region {
        tid: OUT,
        v_rows: r,
        v_cols: c,
        r_start: 0,
        r_len: r,
        r_cover: (0, r),
        c_start: 0,
        c_len: c,
        c_cover: (0, c),
        is_out: true,
        is_fp8: false,
        band: None,
    };
    let mut sym = 0i64;
    let ops = elementwise::<W>(
        "band",
        Elementwise::Add,
        &[band(0), band(1), o],
        &[],
        &mut sym,
        Some(&l),
    )
    .unwrap_or_else(|e| panic!("[{r}, {c}] of [{p}, {c}]: {}", e.message));
    assert_eq!(ops.len(), 1, "one descriptor");
    let v = serde_json::to_value(&ops[0].op).expect("the op serializes");
    let d = v["dscs_"][0]
        .as_object()
        .expect("a dsc map")
        .values()
        .next()
        .expect("one dsc")
        .clone();
    let cores = d["numCoresUsed_"].as_u64().expect("a core count");
    // The band's whole row sweep — what the HBM unit view's capacity spans (core fold included).
    let swept = d["N_"]["mb_"].as_u64().expect("a swept mb");
    (
        d["scheduleTree_"]
            .as_array()
            .expect("a schedule tree")
            .clone(),
        cores,
        swept,
    )
}

/// ⭐ THE LAW, at every core count the byte law reaches (1, 8 and 32 among them): both banded
/// operands carry `backGapCore_ {mb: {"-1": p − r}}` past the band's whole `r`-row sweep, their
/// walks reconstruct (`-1` per dim — never pinned to the parent), and the whole output carries no
/// gap.
#[test]
fn a_multi_stick_band_states_the_parent_pitch_as_its_back_gap() {
    let mut seen = std::collections::BTreeSet::new();
    let mut cases = Vec::new();
    for (p, r, c) in [
        (4u32, 1u32, 4096u32),
        (4, 1, 8192),
        (64, 16, 384),
        (1024, 512, 128),
        (16, 8, 128),
        (8, 2, 256),
        (2, 1, 128),
        (6, 3, 192),
        (3, 1, 320),
        // 37 sticks — prime past 32, so the stick axis divides over ONE core.
        (2, 1, 37 * 64),
    ] {
        let (nodes, cores, swept) = band_add(p, r, c);
        seen.insert(cores);
        cases.push(format!(
            "[{r},{c}]/[{p},{c}] -> {cores} core(s), {swept} row(s) swept"
        ));
        assert_eq!(
            swept,
            u64::from(r),
            "[{r}, {c}] of [{p}, {c}]: the op sweeps the band's own rows"
        );
        let gap = (p - r).to_string();
        for (i, n) in nodes.iter().enumerate() {
            let what = format!("[{r}, {c}] of [{p}, {c}] on {cores} core(s), operand {i}");
            if i < 2 {
                assert_eq!(
                    n["backGapCore_"],
                    serde_json::json!({"mb": {"-1": gap}}),
                    "{what}: the parent's pitch past the band's rows"
                );
                assert_eq!(
                    n["maxDimSizes_"],
                    serde_json::json!([-1, -1]),
                    "{what}: the band walks rank-2 stick-major, reconstructing"
                );
            } else {
                assert!(
                    n.get("backGapCore_").is_none(),
                    "{what}: a whole output has no gap"
                );
            }
            let walk = n["maxDimSizes_"].as_array().expect("a walk");
            assert!(
                walk.iter()
                    .all(|e| e.as_i64() != Some(i64::from(p)) || p == r),
                "{what}: no walk is pinned to the parent's rows: {walk:?}"
            );
        }
    }
    for want in [1u64, 8, 32] {
        assert!(
            seen.contains(&want),
            "the cases reach {want} core(s): {cases:#?}"
        );
    }
}

/// ⭐ A ONE-STICK BAND STATES THE SAME PITCH — one law at every width: both banded operands carry
/// `backGapCore_ {mb: {"-1": p − r}}` past the band's whole `r`-row sweep, no walk is pinned to the
/// parent, and the whole output carries no gap.
#[test]
fn a_one_stick_band_states_the_parent_pitch_too() {
    let (p, r, c) = (8u32, 2u32, 64u32);
    let (nodes, _, swept) = band_add(p, r, c);
    assert_eq!(swept, u64::from(r), "the op sweeps the band's own rows");
    let gap = (p - r).to_string();
    for (i, n) in nodes.iter().enumerate() {
        let what = format!("[{r}, {c}] of [{p}, {c}], operand {i}");
        if i < 2 {
            assert_eq!(
                n["backGapCore_"],
                serde_json::json!({"mb": {"-1": gap}}),
                "{what}: the parent's pitch past the band's rows"
            );
        } else {
            assert!(
                n.get("backGapCore_").is_none(),
                "{what}: a whole output has no gap"
            );
        }
        let walk = n["maxDimSizes_"].as_array().expect("a walk");
        assert!(
            walk.iter().all(|e| e.as_i64() != Some(i64::from(p))),
            "{what}: no walk is pinned to the parent's rows: {walk:?}"
        );
    }
}

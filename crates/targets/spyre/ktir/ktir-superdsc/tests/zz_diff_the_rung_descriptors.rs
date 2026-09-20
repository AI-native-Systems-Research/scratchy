// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ WHAT DIFFERS BETWEEN THE RUNG-8 AND RUNG-16 ATTENTION EMISSIONS.
//!
//! ## The card measurement this exists to explain
//! Sweeping the LIVE count across the rung boundary (granite-3.1-2b fp8, distinct subjects per row):
//! ```text
//!   live= 8  rung= 8   good= 8  bad=0   CLEAN
//!   live= 9  rung=16   good= 7  bad=2
//!   live=10  rung=16   good= 5  bad=5
//!   live=12  rung=16   good= 5  bad=7
//!   live=16  rung=16   good= 8  bad=8
//! ```
//! Nine live on the 16-row rung already corrupt, and sixteen live on it have ZERO padding rows and
//! still corrupt — so it is neither a live-count limit nor the padding-row aliasing the worker
//! documents. Something about running on rung 16 is wrong.
//!
//! ⛔⛔⛔ AND IT IS **NOT** THE ADDRESSING, WHICH IS WHAT THIS FILE ENDED UP PROVING. The suspicion it
//! was built to confirm — rung 16 declares 64 `y*mb` row-slots against only 32 per-core starts, so half
//! the batch is unaddressed — is REFUTED by the emitted walk: the per-core `mb` fold factor goes 1 → 2
//! with an affine step of 2, and every start scales with it. A core LOOPS; 32 starts covering 64 slots
//! is correct emission. Acting on the unrefuted version would have "fixed" working code.
//!
//! So the emitter's per-rung descriptors are COHERENT at rung 16 — all four projections here agree —
//! and the corruption lives somewhere these do not describe. That is a real elimination, and it is why
//! the surviving assertion is the PROPORTIONALITY law rather than a start count.
//!
//! ## Why this diff, and why THESE three fields
//! `attn.rs` prescribes it: *"the wide rungs are still the ones to distrust first … verify by DIFFING
//! `layoutDimOrder_` / `maxDimSizes_` / `numWkSlicesPerDim_` per rung, not addresses — the previous
//! attempt compared only start addresses, found them a strict subset, and shipped complete noise."*
//!
//! ⛔ AND IT MUST BE THE SHIPPED OP. An earlier attempt built the score leg by hand with
//! `MatY::of_requests`, which puts the BATCH on `y` (measured: `mq=8` gave `y=8, mb=1`) while shipped
//! attention uses `MatY::of_gqa_group` — `y` is the GQA group and `mb` carries the batch. That harness
//! measured a different op. This one calls `assemble_attn` itself, so whatever it reports is what the
//! bundle actually contains.

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::assemble_attn;
use ktir_superdsc::sdsc_abstract::{AttnGeometry, attn_bundle_rows};

/// granite-3.1-2b: 32 query heads, 8 kv heads, head dim 64 — the model every card number above is from.
const NQH: u32 = 32;
const NKVH: u32 = 8;
const HD: u32 = 64;
/// The resident context the decode bundles are baked for, and the swept extent per fold pass.
const CAP: u32 = 256;
const ACTIVE_CAP: u32 = 256;

/// One emitted op's identity plus the three fields `attn.rs` says to diff, per operand.
#[derive(Debug, PartialEq, Eq)]
struct OpFields {
    name: String,
    /// `(layoutDimOrder_, maxDimSizes_)` per operand, in operand order.
    operands: Vec<(Vec<String>, Vec<i64>)>,
    /// The op's work split, as sorted `(dim, slices)`.
    split: Vec<(String, i64)>,
    /// ⭐ HOW MANY DISTINCT PER-CORE START ADDRESSES each operand declares.
    ///
    /// This is the quantity the prescribed three-field diff cannot see. MEASURED: 32 at both rungs,
    /// while `mb`'s extent goes 8 → 16.
    ///
    /// ⛔ THAT IS NOT THE DEFECT, though it reads like one. A core walks its slice, so the count of
    /// starts need not track the row count — what must track it is the per-core fold
    /// ([`doubling_the_rung_doubles_the_per_core_row_fold`]), and that one does scale. Kept because the
    /// pair (starts, fold) is what makes the emission legible: 32 starts with a fold of 2 is coherent,
    /// 32 starts with a fold of 1 would not be.
    starts: Vec<usize>,
    /// Total ENTRIES in each operand's start map, distinct or not.
    ///
    /// ⭐ THIS IS WHAT SEPARATES THE TWO FAILURE SHAPES, and they need different fixes:
    /// * `entries == distinct` but fewer than the rows declared ⇒ rows with NO address at all.
    /// * `entries > distinct` ⇒ two rows COLLIDING on one address, which is a silent overwrite.
    entries: Vec<usize>,
    /// The `(label, factor)` fold attributes — core / corelet / TIME. A time factor > 1 is how 32
    /// cores legitimately cover more than 32 row-slots, so its value decides whether a wide rung was
    /// ever given a way to address its extra rows.
    fold: Vec<(String, i64)>,
}

/// Emit the whole attention body at one decode width and project the three fields out of every op.
fn emit_at(mq: u32) -> Vec<OpFields> {
    let geom = AttnGeometry::<NQH, NKVH, HD>::minted();
    let bundle_rows =
        attn_bundle_rows(geom, mq, true).unwrap_or_else(|| panic!("mq={mq} is not a baked rung"));
    let mut sym = 0i64;
    let ops = assemble_attn(
        0,
        geom,
        bundle_rows,
        CAP,
        ACTIVE_CAP,
        "t_qs",
        "t_new_k",
        "t_new_v",
        "t_kct",
        "t_vc",
        "t_pmask",
        "t_cmask",
        None,
        ktir_superdsc::place::PlaceId::Act(900),
        true,
        &mut sym,
        None,
    )
    .unwrap_or_else(|e| panic!("mq={mq}: assemble_attn refused: {}", e.0));

    ops.iter()
        .map(|e| {
            let v = serde_json::to_value(&e.op).unwrap();
            let dscs = &v["dscs_"][0];
            let (name, body) = dscs
                .as_object()
                .and_then(|m| m.iter().next())
                .map(|(k, b)| (k.clone(), b))
                .expect("one named dsc per emitted op");
            let operands = body["scheduleTree_"]
                .as_array()
                .map(|nodes| {
                    nodes
                        .iter()
                        .map(|n| {
                            let layout = n["layoutDimOrder_"]
                                .as_array()
                                .map(|a| {
                                    a.iter()
                                        .map(|d| d.as_str().unwrap_or("?").to_string())
                                        .collect()
                                })
                                .unwrap_or_default();
                            let maxd = n["maxDimSizes_"]
                                .as_array()
                                .map(|a| a.iter().map(|d| d.as_i64().unwrap_or(0)).collect())
                                .unwrap_or_default();
                            (layout, maxd)
                        })
                        .collect()
                })
                .unwrap_or_default();
            let mut split: Vec<(String, i64)> = body["numWkSlicesPerDim_"]
                .as_object()
                .map(|m| {
                    m.iter()
                        .map(|(k, n)| (k.clone(), n.as_i64().unwrap_or(0)))
                        .collect()
                })
                .unwrap_or_default();
            split.sort();
            // Distinct per-core start addresses per operand, counted off the emitted map.
            let starts = body["scheduleTree_"]
                .as_array()
                .map(|nodes| {
                    nodes
                        .iter()
                        .map(|n| {
                            // ⛔ THE VALUES ARE STRINGS (`{"[0, 0, 0]": "0"}`), not integers. A first
                            // version filtered with `as_i64()` and reported ZERO distinct starts for
                            // every operand at both rungs — a broken extractor reading as a finding.
                            n["startAddressCoreCorelet_"]["data_"]
                                .as_object()
                                .map(|m| {
                                    m.values()
                                        .filter_map(|v| {
                                            v.as_str()
                                                .and_then(|s| s.trim().parse::<i64>().ok())
                                                .or_else(|| v.as_i64())
                                        })
                                        .collect::<std::collections::BTreeSet<_>>()
                                        .len()
                                })
                                .unwrap_or(0)
                        })
                        .collect()
                })
                .unwrap_or_default();
            let entries = body["scheduleTree_"]
                .as_array()
                .map(|nodes| {
                    nodes
                        .iter()
                        .map(|n| {
                            n["startAddressCoreCorelet_"]["data_"]
                                .as_object()
                                .map(|m| m.len())
                                .unwrap_or(0)
                        })
                        .collect()
                })
                .unwrap_or_default();
            let fold = body["scheduleTree_"][0]["startAddressCoreCorelet_"]["dim_prop_attr"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|d| {
                            (
                                d["label_"].as_str().unwrap_or("?").to_string(),
                                d["factor_"].as_i64().unwrap_or(0),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default();
            OpFields {
                name,
                operands,
                split,
                starts,
                entries,
                fold,
            }
        })
        .collect()
}

/// ⛔ VALIDATE THE START-ADDRESS EXTRACTOR BEFORE BELIEVING A ZERO FROM IT.
///
/// The first version of the `starts` projection filtered `data_`'s values with `as_i64()` and reported
/// ZERO distinct starts for every operand at both rungs — which is indistinguishable from "the values
/// are not integers and my filter dropped all of them". A count of zero from an extractor that has
/// never returned nonzero is not a measurement. This dumps the real shape.
#[test]
fn the_start_address_extractor_sees_real_values() {
    let geom = AttnGeometry::<NQH, NKVH, HD>::minted();
    let bundle_rows = attn_bundle_rows(geom, 8, true).expect("rung 8");
    let mut sym = 0i64;
    let ops = assemble_attn(
        0,
        geom,
        bundle_rows,
        CAP,
        ACTIVE_CAP,
        "t_qs",
        "t_new_k",
        "t_new_v",
        "t_kct",
        "t_vc",
        "t_pmask",
        "t_cmask",
        None,
        ktir_superdsc::place::PlaceId::Act(900),
        true,
        &mut sym,
        None,
    )
    .expect("emits");
    let v = serde_json::to_value(&ops[0].op).unwrap();
    let body = v["dscs_"][0].as_object().unwrap().values().next().unwrap();
    let node0 = &body["scheduleTree_"][0];
    eprintln!(
        "[starts-probe] startAddressCoreCorelet_ = {}",
        serde_json::to_string(&node0["startAddressCoreCorelet_"])
            .unwrap_or_default()
            .chars()
            .take(400)
            .collect::<String>()
    );
    assert!(
        !node0["startAddressCoreCorelet_"].is_null(),
        "the emitted node carries no startAddressCoreCorelet_ at all — the projection is looking in \
         the wrong place, not observing an absence of starts"
    );
}

/// Dump the score op's first-operand node at both rungs, so the PER-CORE WALK (`coordinates_`) can be
/// diffed rather than argued about.
///
/// ⛔ WHY THIS IS NEEDED. "32 starts for 64 row-slots" is only a defect if a core cannot walk more than
/// one slot — and a core DOES loop, using the declared strides. At rung 8 each core owns exactly one
/// slot, so the per-core `mb` stride is never exercised; at rung 16 it is exercised for the first time.
/// Two readings of the same start count have already contradicted each other in one sitting, so the
/// strides get written to disk and compared.
#[test]
fn dump_the_score_node_for_stride_comparison() {
    for (mq, path) in [(8u32, "/tmp/node_r8.json"), (16, "/tmp/node_r16.json")] {
        let geom = AttnGeometry::<NQH, NKVH, HD>::minted();
        let bundle_rows = attn_bundle_rows(geom, mq, true).expect("a baked rung");
        let mut sym = 0i64;
        let ops = assemble_attn(
            0,
            geom,
            bundle_rows,
            CAP,
            ACTIVE_CAP,
            "t_qs",
            "t_new_k",
            "t_new_v",
            "t_kct",
            "t_vc",
            "t_pmask",
            "t_cmask",
            None,
            ktir_superdsc::place::PlaceId::Act(900),
            true,
            &mut sym,
            None,
        )
        .expect("emits");
        let op = ops
            .iter()
            .find(|e| {
                serde_json::to_value(&e.op).unwrap()["dscs_"][0]
                    .as_object()
                    .and_then(|m| m.keys().next().cloned())
                    .is_some_and(|k| k.starts_with("attn_nsc_g0"))
            })
            .expect("the group-0 new-block score op");
        let v = serde_json::to_value(&op.op).unwrap();
        let body = v["dscs_"][0].as_object().unwrap().values().next().unwrap();
        std::fs::write(
            path,
            serde_json::to_string_pretty(&body["scheduleTree_"][0]).unwrap(),
        )
        .expect("write the node");
        eprintln!("[node-dump] mq={mq} -> {path}");
    }
}

/// ⭐⭐⭐⭐⭐ WHAT `skip_addr` WOULD BE IF THE SCORE LEG'S Kᵗ WERE GATHERED — measured off the SHIPPED
/// attention emission, because this one number decides whether wiring the gather works or silently
/// reads another request's keys.
///
/// dxp derives `skip_addr` from the value tensor's own per-dim capacities
/// (`getBufferCapacityForNodePerDim`): an unbounded dim contributes its per-core DATASTAGE extent, the
/// pinned dim is clamped to the page. So it equals ONE ENTRY'S SIZE, and the index must step exactly
/// that far in HBM.
///
/// ⛔ AND MY OWN EARLIER FIGURE FOR THIS WAS WRONG. I read the kernel's `maxDimSizes_` (`[-1,-1]`,
/// i.e. "reconstruct") and concluded the entry was `hd = 64` against a required layer-page of 131,072 —
/// "2048x short". `maxDimSizes_` is the WALK, not the capacity: the capacity comes from the datastage
/// extents. This prints the real product so the wiring decision rests on a measurement.
#[test]
fn what_skip_addr_the_gathered_score_leg_would_get() {
    let geom = AttnGeometry::<NQH, NKVH, HD>::minted();
    let bundle_rows = attn_bundle_rows(geom, 8, true).expect("rung 8");
    let mut sym = 0i64;
    let ops = assemble_attn(
        0,
        geom,
        bundle_rows,
        CAP,
        ACTIVE_CAP,
        "t_qs",
        "t_new_k",
        "t_new_v",
        "t_kct",
        "t_vc",
        "t_pmask",
        "t_cmask",
        None,
        ktir_superdsc::place::PlaceId::Act(900),
        true,
        &mut sym,
        None,
    )
    .expect("emits");

    // The PREFIX score leg — the op whose kernel is the resident Kᵗ cache, i.e. the operand a gather
    // would read through. (`attn_*sc_*`, not the NEW-block `attn_nsc_*`.)
    let op = ops
        .iter()
        .find(|e| {
            serde_json::to_value(&e.op).unwrap()["dscs_"][0]
                .as_object()
                .and_then(|m| m.keys().next().cloned())
                .is_some_and(|k| k.contains("sc_g0") && !k.contains("nsc"))
        })
        .expect("the group-0 prefix score op");
    let v = serde_json::to_value(&op.op).unwrap();
    let (name, body) = v["dscs_"][0]
        .as_object()
        .and_then(|m| m.iter().next())
        .map(|(k, b)| (k.clone(), b))
        .unwrap();

    // N_ carries the op's iteration extents by dim name (`out` -> `out_`).
    let n = body["N_"].as_object().expect("N_");
    let kernel = &body["scheduleTree_"][1];
    let layout: Vec<String> = kernel["layoutDimOrder_"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|d| d.as_str().unwrap_or("?").to_string())
                .collect()
        })
        .unwrap_or_default();
    let slices = body["numWkSlicesPerDim_"].as_object();

    let mut product: i64 = 1;
    let mut terms = Vec::new();
    for dim in &layout {
        let extent = n
            .get(&format!("{dim}_"))
            .and_then(|v| v.as_i64())
            .unwrap_or(1);
        let split = slices
            .and_then(|s| s.get(dim))
            .and_then(|v| v.as_i64())
            .unwrap_or(1)
            .max(1);
        let per_core = extent / split;
        terms.push(format!("{dim}={per_core}"));
        product *= per_core;
    }
    eprintln!(
        "[skip-addr] {name}: kernel layout={layout:?} per-core {terms:?} => entry = {product} elems"
    );
    eprintln!(
        "[skip-addr] pool block sizes for comparison: one stick-group = {} elems, one kv-head plane = {} elems",
        HD * 64,
        HD * 256
    );
    assert!(product > 0, "an entry of zero elements is not addressable");
}

/// ⭐ FIRST: both widths must EMIT. If rung 16 refused, the card could not have run it — so a refusal
/// here would mean this harness is not reproducing the shipped emission.
#[test]
fn both_rungs_emit() {
    let r8 = emit_at(8);
    let r16 = emit_at(16);
    assert!(!r8.is_empty(), "rung 8 emitted no ops");
    assert!(!r16.is_empty(), "rung 16 emitted no ops");
    eprintln!(
        "[rung-diff] rung 8: {} ops, rung 16: {} ops",
        r8.len(),
        r16.len()
    );

    // ⭐ THE ACTUAL NUMBERS FOR THE SCORE LEG, printed rather than inferred. "32 starts at both rungs"
    // only indicts the emission if a core CANNOT legitimately own two rows — and it can, provided the
    // declared walk strides for them. So the split, the extents and the start count must be read
    // together before any of it is called a defect.
    for (label, set) in [("rung8", &r8), ("rung16", &r16)] {
        for op in set
            .iter()
            .filter(|o| o.name.starts_with("attn_nsc_g0") || o.name.starts_with("attn_sc_g0"))
        {
            eprintln!(
                "[rung-facts] {label} {}: split={:?} distinct_starts={:?} entries={:?} \
                 fold={:?} operand_max={:?}",
                op.name,
                op.split,
                op.starts,
                op.entries,
                op.fold,
                op.operands.iter().map(|(_, m)| m).collect::<Vec<_>>()
            );
        }
    }
}

/// ⭐⭐⭐ THE DIFF ITSELF. Prints every op whose three fields differ between the rungs, and every op
/// present at one width and not the other.
///
/// This is a REPORTER, not a pass/fail assertion on a specific defect: the point is to surface the
/// difference so it can be judged, and pinning a value before knowing which one is wrong is how a green
/// test gets written around a bug. It fails only if the two emissions are IDENTICAL in these fields —
/// because then the defect is somewhere these three do not describe, and the prescribed diff has been
/// answered in the negative.
#[test]
fn the_rung_16_descriptors_differ_from_rung_8() {
    let r8 = emit_at(8);
    let r16 = emit_at(16);

    // Ops are matched by NAME, since a width change may add or drop ops (per-head arms, slab loops).
    let names8: Vec<&str> = r8.iter().map(|o| o.name.as_str()).collect();
    let names16: Vec<&str> = r16.iter().map(|o| o.name.as_str()).collect();

    let only8: Vec<&&str> = names8.iter().filter(|n| !names16.contains(n)).collect();
    let only16: Vec<&&str> = names16.iter().filter(|n| !names8.contains(n)).collect();

    eprintln!(
        "[rung-diff] ops only at rung 8  ({}): {only8:?}",
        only8.len()
    );
    eprintln!(
        "[rung-diff] ops only at rung 16 ({}): {only16:?}",
        only16.len()
    );

    let mut differing = 0usize;
    for a in &r8 {
        if let Some(b) = r16.iter().find(|b| b.name == a.name) {
            if a.split != b.split {
                differing += 1;
                eprintln!(
                    "[rung-diff] {}: SPLIT  rung8={:?}  rung16={:?}",
                    a.name, a.split, b.split
                );
            }
            for (i, (x, y)) in a.operands.iter().zip(b.operands.iter()).enumerate() {
                if x != y {
                    differing += 1;
                    eprintln!(
                        "[rung-diff] {} operand {i}: rung8 layout={:?} max={:?} | rung16 layout={:?} max={:?}",
                        a.name, x.0, x.1, y.0, y.1
                    );
                }
            }
            // ⭐ THE ADDRESS COVERAGE, which the three prescribed fields cannot show.
            if a.starts != b.starts {
                eprintln!(
                    "[rung-starts] {}: rung8={:?} rung16={:?}  (DISTINCT per-core starts per operand)",
                    a.name, a.starts, b.starts
                );
            } else if a.starts.iter().any(|&n| n > 1) {
                eprintln!(
                    "[rung-starts] {}: UNCHANGED at {:?} while mb doubled — each core now owns two \
                     rows and nothing distinguishes them",
                    a.name, a.starts
                );
            }
        }
    }
    eprintln!("[rung-diff] differing field(s): {differing}");

    assert!(
        differing > 0 || !only16.is_empty() || !only8.is_empty(),
        "rung 8 and rung 16 emit IDENTICAL layoutDimOrder_/maxDimSizes_/numWkSlicesPerDim_ — so the \
         corruption measured on rung 16 is NOT described by the three fields attn.rs prescribes, and \
         the next probe has to look elsewhere (start addresses, the fold's pass count, or the mask \
         staging)."
    );
}

/// ⛔⛔⛔⭐⭐⭐⭐⭐ THE DEFECT: A WIDER RUNG DECLARES MORE ROWS AND NOT ONE MORE ADDRESS.
///
/// Every attention op emits **32 distinct per-core start addresses at BOTH rungs** while `mb`'s extent
/// goes 8 → 16, and the work split is byte-identical. At rung 8 that saturates the device exactly —
/// 32 cores, `mb=8` rows × the GQA group of 4 = 32 — and every row has an address of its own. At rung 16
/// the emission asks for sixteen rows from the SAME 32 starts: there is no 33rd address to give the
/// extra rows, so half the batch is declared in the extents with nothing pointing at it.
///
/// That is the card result exactly: rung 16 corrupts roughly half its rows, nondeterministically, with
/// `layoutDimOrder_` / `maxDimSizes_` / `numWkSlicesPerDim_` all clean — which is why the diff those
/// three prescribe came back benign (232 diffs, every one the legitimate `mb` doubling, zero anomalies).
///
/// ⛔⛔⛔ RETRACTED HYPOTHESIS, KEPT AS A TEST OF WHAT IS ACTUALLY TRUE.
///
/// This started as a "law" asserting that a rung declaring `N` rows must declare `N` distinct per-core
/// START addresses, on the reasoning that rung 16 declares 64 `y*mb` row-slots against only 32 starts
/// and therefore leaves half the batch unaddressed. **That was wrong**, and the emitted walk says so:
/// ```text
///   coordInfo.mb.folds.dim_prop_attr[3].factor_        1 -> 2
///   coordInfo.mb.folds.dim_prop_func[0].Affine.alpha_  1 -> 2
///   all 32 start addresses                            doubled (1024 -> 2048, ...)
/// ```
/// A core LOOPS. Rung 16 tells each of the 32 cores to walk TWO `mb` rows with an affine step of 2, and
/// scales every start accordingly — which is coherent, not broken. 32 starts for 64 slots is the
/// correct emission, and the would-be fix would have "corrected" working code.
///
/// ⭐ SO THE REAL LAW IS THE PROPORTIONALITY, and it HOLDS today: doubling the rung must double the
/// per-core `mb` fold factor, because that factor is the only thing that lets 32 cores cover more than
/// 32 row-slots. A regression that widened the extents WITHOUT scaling the fold would be exactly the
/// silent half-batch corruption first suspected here — so the assertion is worth keeping even though
/// the original suspicion was unfounded.
#[test]
fn doubling_the_rung_doubles_the_per_core_row_fold() {
    fn mb_fold(mq: u32) -> i64 {
        let geom = AttnGeometry::<NQH, NKVH, HD>::minted();
        let bundle_rows = attn_bundle_rows(geom, mq, true).expect("a baked rung");
        let mut sym = 0i64;
        let ops = assemble_attn(
            0,
            geom,
            bundle_rows,
            CAP,
            ACTIVE_CAP,
            "t_qs",
            "t_new_k",
            "t_new_v",
            "t_kct",
            "t_vc",
            "t_pmask",
            "t_cmask",
            None,
            ktir_superdsc::place::PlaceId::Act(900),
            true,
            &mut sym,
            None,
        )
        .expect("emits");
        let op = ops
            .iter()
            .find(|e| {
                serde_json::to_value(&e.op).unwrap()["dscs_"][0]
                    .as_object()
                    .and_then(|m| m.keys().next().cloned())
                    .is_some_and(|k| k.starts_with("attn_nsc_g0"))
            })
            .expect("the group-0 new-block score op");
        let v = serde_json::to_value(&op.op).unwrap();
        let body = v["dscs_"][0].as_object().unwrap().values().next().unwrap();
        let alpha = &body["scheduleTree_"][0]["coordinates_"]["coordInfo"]["mb"]["folds"]["dim_prop_func"]
            [0]["Affine"]["alpha_"];
        alpha
            .as_i64()
            .unwrap_or_else(|| panic!("mq={mq}: no affine mb fold alpha in the emitted walk"))
    }

    // 32 cores cover mq*GQA row-slots; past mq=8 that exceeds the core count and the per-core fold is
    // what makes up the difference.
    assert_eq!(mb_fold(8), 1, "at mq=8 each core owns exactly one row-slot");
    assert_eq!(
        mb_fold(16),
        2,
        "at mq=16 each core must walk TWO row-slots — the extents doubled, so a fold of 1 would leave \
         half the batch uncomputed"
    );
}

/// ⛔⛔⛔⛔⛔ THE SHIPPED GATHERED FOLD PUTS THE GATHER ON **EXACTLY THE TWO KERNEL-LESS COPIES**, AND ON
/// NO MATMUL — read off the real emission, at the real call path.
///
/// ## The constraint this is the shipped-path half of
/// deeptools cannot schedule a gather on an op that has a `KERNEL`. MEASURED as a bake refusal on the
/// card at BOTH possible index `memOrg_` values:
/// ```text
/// hbm + lx  ->  sbf-ddc: Expect a valid allocate node.      L3DlOpsScheduler.cpp:2337
/// hbm       ->  sbf-ddc: Expect LX in labeledDs memOrg_.    L3DlOpsScheduler.cpp:2334
/// ```
/// Both in `calculateFlopPerByte`, which demands an LX allocate node for every HBM-pinned labeledDs and
/// which `hasDimensionReuse` (`:303`, `primaryDsInfo_.size() > 1 && count(KERNEL)`) turns on at `:1550`.
/// An index never gets an LX chunk, so no declaration escapes it. `emit_sdsc` refuses that shape at BUILD
/// time and `zz_the_gather_op_is_kernel_less::a_gather_on_a_matmul_is_refused_at_build_time` pins the
/// refusal itself.
///
/// ## Why THIS test is the one that can regress
/// The refusal only fires if someone puts the gather back on a matmul. What it cannot catch is the gather
/// going MISSING — an `assemble_attn` that emits a bundle with no `indirectAccessIndexLabeledDs` anywhere
/// is a bundle whose scratch is never filled, and an unfilled seg0 scratch reads as ZERO, which the
/// softmax turns into a uniform distribution rather than a fault. So this test asserts the POSITIVE: the
/// declaration exists, it is on the copies, and the copies have no `KERNEL` in `primaryDsInfo_`.
///
/// ⛔ IT ASSERTED THE OPPOSITE (a build refusal) WHILE THE GATHER HAD NO HOME. That was correct for the
/// state it was written in — the score leg carrying a gather was never a working descriptor — and it is
/// exactly the assertion that must invert when the copy lands, because the shipped path is no longer the
/// refused shape.
#[test]
fn the_shipped_gathered_fold_carries_the_gather_only_on_the_kernel_less_copies() {
    let geom = AttnGeometry::<NQH, NKVH, HD>::minted();
    let bundle_rows = attn_bundle_rows(geom, 8, true).expect("rung 8");
    let mut sym = 0i64;
    let ops = assemble_attn(
        0,
        geom,
        bundle_rows,
        CAP,
        ACTIVE_CAP,
        "t_qs",
        "t_new_k",
        "t_new_v",
        "t_kct",
        "t_vc",
        "t_pmask",
        "t_cmask",
        // The name the caller passes — the reserved tid's own spelling.
        Some("t4294967175"),
        ktir_superdsc::place::PlaceId::Act(900),
        true,
        &mut sym,
        None,
    )
    .expect("a gathered attention emits");

    // WHICH OPS DECLARE A GATHER, off the emitted JSON rather than off the call site — the same reading
    // that caught the previous version of this test believing the score leg was a working home.
    let mut gathered: Vec<String> = Vec::new();
    for op in &ops {
        let j = serde_json::to_value(&op.op).expect("an op serializes");
        let Some(dscs) = j.get("dscs_").and_then(|d| d.as_array()) else {
            continue;
        };
        for d in dscs {
            let Some(obj) = d.as_object() else { continue };
            for (name, dsc) in obj {
                let carries = dsc
                    .get("scheduleTree_")
                    .and_then(|t| t.as_array())
                    .is_some_and(|nodes| {
                        nodes.iter().any(|n| {
                            n.get("indirectAllocType_")
                                .and_then(|v| v.as_str())
                                .is_some_and(|s| s == "index_tensor")
                        })
                    });
                if carries {
                    gathered.push(name.clone());
                    // ⭐⭐⭐⭐⭐ AND THIS OP'S INDEX FITS **ONE STICK**, read off the emitted walk rather
                    // than off the type that minted it. `maxDimSizes_` of the index node IS its entry
                    // count (`WorkPlan::of_index_entries`), and dxp transfers an index HBM→L3LUIBR "in
                    // the granularity of one stick", taking each core's read offset inside it modulo
                    // that stick (`L3DlOpsScheduler`). Above 32 entries there is no fault and no
                    // refusal: the cores past the wrap gather another core's pages. That is exactly
                    // what separated the two card outcomes — 32 entries `solo=EXACT`, 128 entries
                    // every row's first decode token wrong — so it is asserted on the DESCRIPTOR,
                    // which is the only place a regression would show.
                    let cap = ktir_superdsc::sdsc_abstract::CopyDims::ENTRIES_PER_OP as i64;
                    let idx_node = dsc
                        .get("scheduleTree_")
                        .and_then(|t| t.as_array())
                        .and_then(|nodes| {
                            nodes.iter().find(|n| {
                                n.get("indirectAllocType_").and_then(|v| v.as_str())
                                    == Some("index_tensor")
                            })
                        })
                        .expect("the node that made `carries` true");
                    let walk: Vec<i64> = idx_node
                        .get("maxDimSizes_")
                        .and_then(|w| w.as_array())
                        .map(|a| a.iter().map(|v| v.as_i64().unwrap_or(0)).collect())
                        .unwrap_or_default();
                    // ⭐ `-1` IS THE PINNED-DIM SENTINEL, NOT AN UNFILLED FIELD, and a page-granular index
                    // declares exactly that: its `mb` is the paged axis and the op names ONE entry
                    // (`mb == page`), so `getPageSize` (`dsc2.cpp:4493-4526`) erases the negative entry and
                    // reads the dim as paged. A positive count is the multi-entry form, which is the one
                    // that can exceed a stick — so the cap still applies to it and only to it.
                    //
                    // ⛔ THE >32 PROTECTION IS UNCHANGED FOR THE SHAPE IT WAS WRITTEN FOR. The IBR is
                    // loaded one stick at a time and read `% bytesPerStick`, so a positive count above
                    // `cap` WRAPS silently — the measured rung-8 corruption. What is new is that a
                    // page-granular op cannot reach that shape at all: one entry per op.
                    assert!(
                        !walk.is_empty() && walk.iter().all(|&e| e == -1 || (e > 0 && e <= cap)),
                        "op '{name}' declares an index walk of {walk:?}; a gather op's index must be one \
                         PINNED axis (-1) or a positive count within ONE {cap}-entry stick — cut the pass \
                         into one op per entry (`PageScratch::copies`)"
                    );
                    // The KERNEL-less requirement, on the very op that carries it: `primaryDsInfo_`
                    // must hold no `KERNEL` role, which is what `hasDimensionReuse` counts.
                    let roles: Vec<String> = dsc
                        .get("primaryDsInfo_")
                        .and_then(|p| p.as_object())
                        .map(|o| o.keys().cloned().collect())
                        .unwrap_or_default();
                    assert!(
                        !roles.iter().any(|r| r == "KERNEL"),
                        "op '{name}' carries a gather AND a KERNEL ({roles:?}) — deeptools refuses \
                         exactly this at bake (L3DlOpsScheduler.cpp:2334/2337), so it must not be \
                         emitted"
                    );
                }
            }
        }
    }
    assert!(
        !gathered.is_empty(),
        "a gathered `assemble_attn` emitted {} ops and NONE of them declares an index. An unfilled \
         gather scratch reads as ZERO, which the online softmax turns into a uniform distribution \
         rather than a fault — so a missing gather is silent, and this is the assertion that catches it.",
        ops.len()
    );
    // TWO LEGS — the Kᵗ plane and the V plane — and one op per INDEX STICK of each, and nothing else.
    //
    // ⛔ IT ASSERTED EXACTLY **2**, ON THE REASONING THAT "a third would mean a per-window or per-head
    // copy crept back in". The premise was right and the number was a consequence of a shape the card
    // refutes: one op for the whole pass declares `nkvh * nb * mq` index entries, and dxp loads a
    // gather's index one stick at a time. The cut that replaces it is neither per-window nor per-head —
    // both of those are non-contiguous under the window-major row law and cannot be named by an
    // `[mb, out]` operand — it is a contiguous ONE-STICK RUN of rows, and the expected count is
    // therefore the scratch's own `copy_count`, derived here from the same geometry the emission used
    // rather than written down.
    // ⭐ PAGE GRANULARITY MAKES THE CUT UNNECESSARY, so the expected count is ONE OP PER PLANE. A row is
    // now a REQUEST holding a whole page plane, so a pass names `mq` entries — and `mq` never exceeds one
    // index stick at any rung the ladder admits, which is why the `rows / 32` split that used to produce
    // 16 ops here is gone rather than merely satisfied. Still derived from the geometry, not written down.
    let scratch = ktir_superdsc::sdsc_abstract::PageScratch::of_pass(
        ktir_superdsc::sdsc_abstract::PagedKvPool::new(NKVH as usize, HD as usize),
        ktir_superdsc::sdsc_abstract::QueryRowCount::of_mq(8),
    )
    .expect("this geometry admits the page gather");
    let per_leg = scratch.copies().count();
    assert_eq!(
        per_leg as u32,
        scratch.mq() * scratch.ops_per_row(),
        "a page-granular pass is one copy op per (REQUEST, index-stick cut) — `ops_per_row` is 1 at \
         every geometry in the ladder, so this is one op per request there. An op per PLANE is refused \
         at bake, 'The initial chunk parameters must fit in LX for SuperDSC' \
         (L3DlOpsScheduler.cpp:1534): the pin is the op's unit of work division, so a plane-sized pin \
         leaves ONE entry, hence ONE CORE, and `getInitialChunkParams` starts from the CORE data stage \
         — the per-core share IS the whole op there. (It is NOT that the chunk is sized before work \
         division; see `PagePlaneExtent::lx_entries_per_op` for the vendor line that settles it.)"
    );
    assert_eq!(
        gathered.len(),
        2 * per_leg,
        "expected {per_leg} Kᵗ copies and {per_leg} V copies, each naming ONE entry — got {gathered:?}",
    );
    // ⛔ AND THE ENTRY COUNT IS THE THING THAT USED TO WRAP. `rows()` is the live entry count per pass;
    // above `CopyDims::ENTRIES_PER_OP` the IBR is read modulo one stick and the cores past the wrap use
    // another core's page address (the measured rung-8 corruption). Asserted here so the property is
    // pinned at the emission, not only at the type's own door.
    assert!(
        scratch.rows() <= ktir_superdsc::sdsc_abstract::CopyDims::ENTRIES_PER_OP,
        "a pass names {} entries against a {}-entry index stick — this WRAPS silently on card",
        scratch.rows(),
        ktir_superdsc::sdsc_abstract::CopyDims::ENTRIES_PER_OP,
    );
    let (kt, v) = (
        gathered.iter().filter(|n| n.contains("gkt")).count(),
        gathered.iter().filter(|n| n.contains("gv")).count(),
    );
    assert_eq!(
        (kt, v),
        (per_leg, per_leg),
        "each leg must be covered by exactly its own runs — a leg short of one run leaves those rows \
         holding whatever the allocator handed out, which is a valid block number. Got {gathered:?}"
    );
    // ⛔ AND THE RUNS MUST BE DISTINCT OPS, not the same name emitted twice: the cut is expressed as a
    // baked base, so two identically-named ops would be two gathers of run 0 and the rows of every other
    // run would never be written.
    let distinct: std::collections::BTreeSet<&String> = gathered.iter().collect();
    assert_eq!(
        distinct.len(),
        gathered.len(),
        "every run is its own op and must say so in its name, got {gathered:?}"
    );
}

/// ⭐⭐⭐⭐⭐ EVERY RUN OF THE CUT PASS ADDRESSES **ITS OWN** ENTRIES AND **ITS OWN** ROWS — the two bases
/// that make the cut mean anything, read off the emitted per-core start addresses.
///
/// ⛔ THE FAILURE THIS CATCHES IS A DROPPED BASE, AND IT IS SILENT BOTH WAYS. If the INDEX base does not
/// reach the descriptor, every run converts run 0's 32 entries and writes run 0's pages into its own
/// rows; if the DESTINATION base does not, every run writes run 0's rows and the rows above them are
/// never written at all — and an unwritten scratch row is whatever the allocator handed out, read as
/// keys. MEASURED on the card as exactly that progression: one run `solo=EXACT`, two runs the first
/// token right and then divergence, four runs wrong from the first token.
///
/// This is the emission-side half; `GatherCopy` is the type-side half. Both are needed because the type
/// can mint a correct pair and the emitter can still fail to carry one of them through.
#[test]
fn every_run_of_the_cut_pass_addresses_its_own_entries_and_rows() {
    let geom = AttnGeometry::<NQH, NKVH, HD>::minted();
    // ⭐ THE RUNG-8 BODY, WHICH IS THE ONE THE CARD RUNS: `active_cap = 128` (`nb = 2`), mq = 8 — so
    // `2 * 8 * 8 = 128` rows and FOUR runs. The ceiling body (`active_cap = 256`) is eight, and a
    // one-run geometry proves nothing here.
    let bundle_rows = attn_bundle_rows(geom, 8, true).expect("rung 8");
    let mut sym = 0i64;
    let ops = assemble_attn(
        0,
        geom,
        bundle_rows,
        CAP,
        128,
        "t_qs",
        "t_new_k",
        "t_new_v",
        "t_kct",
        "t_vc",
        "t_pmask",
        "t_cmask",
        Some("t4294967175"),
        ktir_superdsc::place::PlaceId::Act(900),
        true,
        &mut sym,
        None,
    )
    .expect("a gathered attention emits");

    // Per gathered op: the smallest per-core start address of its INDEX node and of its OUTPUT node.
    // The minimum is the right reading: dxp's own idx2addr takes the minimum across cores as the run's
    // base and preserves the per-core spread around it.
    let min_start = |node: &serde_json::Value| -> Option<i64> {
        node.get("startAddressCoreCorelet_")?
            .get("data_")?
            .as_object()?
            .values()
            .filter_map(|v| {
                v.as_str()
                    .and_then(|s| s.parse::<i64>().ok())
                    .or(v.as_i64())
            })
            .min()
    };
    let mut runs: Vec<(String, i64, i64)> = Vec::new();
    for op in &ops {
        let j = serde_json::to_value(&op.op).expect("serializes");
        for d in j["dscs_"].as_array().into_iter().flatten() {
            for (name, dsc) in d.as_object().into_iter().flatten() {
                let Some(nodes) = dsc.get("scheduleTree_").and_then(|t| t.as_array()) else {
                    continue;
                };
                let idx = nodes.iter().find(|n| {
                    n.get("indirectAllocType_").and_then(|v| v.as_str()) == Some("index_tensor")
                });
                let Some(idx) = idx else { continue };
                // The destination is the LAST node — `emit_sdsc` defines the output as the last arg,
                // which is exactly why the index is INSERTED before it rather than appended.
                let out = nodes.last().expect("a tree has nodes");
                runs.push((
                    name.clone(),
                    min_start(idx).expect("an index base"),
                    min_start(out).expect("an output base"),
                ));
            }
        }
    }
    // ⭐ ONE RUN PER REQUEST AT PAGE GRANULARITY — `mq` runs per leg, not the window-granular `rows/32`.
    //
    // ⛔ AND IT IS `mq`, NOT 1, BECAUSE THE CARD SAID SO. One op per PLANE (all `mq` requests in a single
    // 2 MB copy) is what I emitted first, and dxp refuses it: "The initial chunk parameters must fit in LX
    // for SuperDSC" (`L3DlOpsScheduler.cpp:1534`). LX is a per-core budget but the INITIAL CHUNK is sized
    // before work division, so the op's whole footprint is measured. One request per op is 256 KB.
    let kt: Vec<_> = runs.iter().filter(|(n, ..)| n.contains("gkt")).collect();
    assert_eq!(
        kt.len(),
        8,
        "a page-granular rung-8 body is ONE run per REQUEST (mq=8), got {runs:?}"
    );
    // ⛔ THE BASES MUST BE STRICTLY INCREASING AND DISTINCT. Equal bases across runs IS the dropped-base
    // defect, and it is what a `startAddressCoreCorelet_` diff of two runs would otherwise hide.
    //
    // ⚠️ VACUOUS AT ONE RUN, AND KEPT DELIBERATELY: it is the guard for the multi-run case, which a wider
    // rung or a smaller entry page would reintroduce. Deleting it would mean a future cut lands unguarded.
    // The count assertion above is what carries the check today.
    for w in kt.windows(2) {
        let (a, b) = (w[0], w[1]);
        assert!(
            b.1 > a.1,
            "run '{}' must read entries PAST run '{}' (index bases {} vs {}) — equal bases mean every \
             run converts run 0's entries and writes run 0's pages into its own rows",
            b.0,
            a.0,
            b.1,
            a.1
        );
        assert!(
            b.2 > a.2,
            "run '{}' must write rows PAST run '{}' (output bases {} vs {}) — equal bases mean the rows \
             above the first run are never written, and an unwritten scratch row is read as keys",
            b.0,
            a.0,
            b.2,
            a.2
        );
    }
    // And the SPACING is the run: one 128-byte index stick, and 32 rows of `cols` fp16 elements.
    let stick_bytes = (ktir_superdsc::sdsc_abstract::CopyDims::ENTRIES_PER_OP * 4) as i64;
    let row_bytes = (ktir_superdsc::sdsc_abstract::CopyDims::ENTRIES_PER_OP * HD * 64 * 2) as i64;
    for w in kt.windows(2) {
        assert_eq!(
            w[1].1 - w[0].1,
            stick_bytes,
            "consecutive runs are ONE index stick apart, or a run reads entries that are not its own"
        );
        assert_eq!(
            w[1].2 - w[0].2,
            row_bytes,
            "consecutive runs are one RUN of rows apart, or the runs overlap in the scratch"
        );
    }
    eprintln!("[cut-probe] {runs:?}");
}

/// ⭐ AND THE UNGATHERED SHIPPED FOLD STILL EMITS — the control, so the refusal above is attributable
/// to the GATHER and not to anything else this harness does.
#[test]
fn the_ungathered_shipped_prefix_fold_still_emits() {
    let geom = AttnGeometry::<NQH, NKVH, HD>::minted();
    let bundle_rows = attn_bundle_rows(geom, 8, true).expect("rung 8");
    let mut sym = 0i64;
    let ops = assemble_attn(
        0,
        geom,
        bundle_rows,
        CAP,
        ACTIVE_CAP,
        "t_qs",
        "t_new_k",
        "t_new_v",
        "t_kct",
        "t_vc",
        "t_pmask",
        "t_cmask",
        None,
        ktir_superdsc::place::PlaceId::Act(900),
        true,
        &mut sym,
        None,
    )
    .unwrap_or_else(|e| panic!("the shipped ungathered fold must emit: {}", e.0));
    assert!(!ops.is_empty(), "the attention body emitted no ops at all");
    // ⛔ AND NOT ONE OF THEM CARRIES A GATHER. This is what "the shipped bundle is unchanged" means as
    // an assertion rather than as a claim.
    for e in &ops {
        let v = serde_json::to_value(&e.op).unwrap();
        let Some((name, body)) = v["dscs_"][0]
            .as_object()
            .and_then(|m| m.iter().next())
            .map(|(k, b)| (k.clone(), b.clone()))
        else {
            continue;
        };
        let tree = body["scheduleTree_"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        assert!(
            tree.iter()
                .all(|n| n["indirectAllocType_"] == "no_indirection"
                    || n["indirectAllocType_"].is_null()),
            "{name}: an ungathered fold emitted an indirect access"
        );
    }
}

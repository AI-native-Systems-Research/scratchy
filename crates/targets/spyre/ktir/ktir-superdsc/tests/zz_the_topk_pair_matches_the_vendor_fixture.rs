// SPDX-License-Identifier: Apache-2.0
//! THE VENDOR TOP-K PAIR'S FIRST EMISSION — pinned against the vendor's own fixtures.
//!
//! # THE ORACLE, AND WHAT IT DOES AND DOES NOT LICENSE
//!
//! `ddc/ddl_templates/test/sdsc_topKIndex.json` and `sdsc_topKValue.json` (in the deeptools
//! checkout beside this repo — the same tree `topk.ddl` ships in) are the only descriptors
//! anywhere that name either opFunc. They are HAND-BUILT (`hbmSize_ 256/32` against element
//! volumes of 16384/2048, `lxSize_` at a wrap-around sentinel), so this is NOT byte parity and
//! no test below claims one. What they pin — read out of THEM, not out of our own emission:
//!
//! ```text
//!   N_                    {"in_": 64, "out_": 8, "y_": 64, "mb_": 2, "i_": 2}
//!   numWkSlicesPerDim_    {"in": 1, "out": 8, "mb": 2, "y": 1, "i": 2}   <- the RING is out:8
//!   primaryDsInfo_ INPUT  layoutDimOrder_ ["in","i","y","mb"]  stickDimOrder_ ["y"] stickSize_ [64]
//!   primaryDsInfo_ OUTPUT layoutDimOrder_ ["out","i","y","mb"] stickDimOrder_ ["y"] stickSize_ [64]
//!   computeOp_            exUnit "sfp", opFuncName "topkindex"/"topkvalue",
//!                         attributes_ {dataFormat_ SEN169_FP16, fidelity_ regular},
//!                         location "Inner", ONE input lds, ONE output lds
//!   constantInfo_         {} — the FALSY empty OBJECT, not the truthy "{}" string
//! ```
//!
//! The fixtures' computeOp_ ALSO carries a five-field scheduling row (`auxLoopName ""`,
//! `isAtMainLoop` 1, `isAtTop` 1, `level` 0, `opConsts` {}) that this emission OMITS — see
//! the documented divergences below.
//!
//! # WHY THE DIM NAMES DON'T MATTER TO DDC — THE BINDING IS A SOLVER
//!
//! `DdlConversion::matchDdl2Dsc` (`ddc/ddl/ddl_conversion.cpp:2360-2460`) does not read dim
//! names. Its constraint pass pins the DDL's ORDER-FIXED slice/stick layouts positionally — for
//! `topk.ddl` both are `layout(%outer_dim#0)`, so `outer#0` is pinned to the SDSC's stick dim —
//! then the FLEXIBLE global layouts restrict each DDL symbol's candidates to the dims of the
//! tensors that reference it (the input's layout covers `outer#0..#4, reduce#0..#2`; the
//! output's covers `outer#0..#4, %k`), and a backtracking search assigns the SDSC's dims to the
//! DDL's symbols. With input `[in, y]`, output `[out, y]`, stick `y`, the solver derives
//! reduce=`in`, k=`out`, `outer#0`=`y` — exactly the roles `topk.ddl`'s dataflow needs (the
//! element loop over the reduce dims, the `core_to_core_communication` ring over `%k`). The
//! fixtures' own 4-dim `["in","i","y","mb"]` form is the same solver's output at a geometry with
//! two extra instance dims; the 3-dim form this builder emits is the minimal one.
//!
//! # THE DOCUMENTED DIVERGENCES (asserted, not hidden)
//!
//! • no `mb`/`i` dims — this builder instances the token volume on the multi-stick `y` axis
//!   (the fixtures' `mb: 2, i: 2` splits played the instance-parallelism role `y`-splitting
//!   plays here; the ring is scoped to the k dim, so cores in different token slices never
//!   communicate and each runs its own ring);
//! • `stickRepl_` and `pdsRelation_` are never emitted by this crate (the transpose-door law);
//! • the fixtures' five-field scheduling row (`auxLoopName ""` / `isAtMainLoop` 1 / `isAtTop`
//!   1 / `level` 0 / `opConsts` {}) is OMITTED: the dev stack's new SDSC-JSON validator
//!   rejects all five fields outright ("Unrecognized field") — MEASURED on the tiny26 bake's
//!   dxp, which refused the pair's group with exactly that error on both members (manual
//!   dxp_standalone on the persisted stage dir; the same validator class as the
//!   `location: "invalid"` rejection). The fixtures predate the validator; the omitted row
//!   is the only form that compiles.
//!
//! # RE-RUNNING THE ddc LEG
//!
//! `cargo test` cannot run ddc. The bake is the gate that closes it: a bundle carrying these
//! descriptors runs `matchDdl2Dsc` against `topk.ddl` for real, and a form the solver cannot
//! bind dies there as `DtException: Scheduler failed to find a suitable op mapping` — which is
//! exactly how the arity law ([`ktir_superdsc::superdsc_opspec::OpFunc::ddl_input_arity`]) was
//! found. Everything asserted here is the descriptor's own side of that contract.

use ktir_superdsc::emit::{TopKGeom, try_assemble_topk, try_assemble_transpose};
use ktir_superdsc::superdsc_opspec::OpFunc;
use serde_json::json;

/// Emit one pair member and hand back the whole `SdscOp` as JSON. The wire format splits the
/// facts across its two levels — `numWkSlicesPerDim_`/`numCoresUsed_` ride the op, `N_`/
/// `primaryDsInfo_`/`computeOp_`/`constantInfo_` ride the `dscs_[0]` body — so both are read
/// from one value and a missing field is a failed index rather than a silently-passed `Option`.
fn emit(member: OpFunc, geom: TopKGeom) -> Result<serde_json::Value, String> {
    let mut sid = -1i64;
    let e = try_assemble_topk("topk", member, geom, "sxt", "six", &mut sid, None)?;
    Ok(serde_json::to_value(&e.op).unwrap())
}

/// The emitted dsc body (`dscs_[0]["topk"]`) — the level `N_`, the layouts, the computeOp row
/// and `constantInfo_` live at.
fn body(d: &serde_json::Value) -> &serde_json::Value {
    &d["dscs_"][0]["topk"]
}

/// ⭐ THE FIXTURE'S OWN GEOMETRY — every layout number, the ring, and the core computeOp row,
/// for BOTH members. This is the contract `topk.ddl`'s solver and `matchDdl2Dsc` see; the
/// divergences asserted inline are the three the module docs record (no `mb`/`i`; no
/// `stickRepl_`/`pdsRelation_`; the validator-rejected scheduling row omitted).
#[test]
fn the_fixture_geometry_emits_the_fixture_contract() {
    for (member, wire) in [
        (OpFunc::TopKIndex, "topkindex"),
        (OpFunc::TopKValue, "topkvalue"),
    ] {
        // The fixtures' N_ {in_: 64, out_: 8, y_: 64} — their mb_/i_ are the instance dims this
        // form folds into the multi-stick y, so ours carries NO mb_ slot at all.
        let d = emit(
            member,
            TopKGeom {
                candidates: 64,
                slots: 8,
                tokens: 64,
            },
        )
        .unwrap_or_else(|e| panic!("{wire} at the fixture geometry refused: {e}"));
        let b = body(&d);
        assert_eq!(b["N_"]["in_"], 64, "{wire}: N_.in_");
        assert_eq!(b["N_"]["out_"], 8, "{wire}: N_.out_");
        assert_eq!(b["N_"]["y_"], 64, "{wire}: N_.y_");
        assert!(
            b["N_"]["mb_"].is_null() && b["N_"]["i_"].is_null(),
            "{wire}: this form has no mb/i instance dims — the documented divergence from the \
             fixtures' 4-dim N_. A mb_/i_ slot appearing here means someone re-added phantom \
             unit dims, the exact divergence class that collapsed the SFP rsqrt refinement."
        );
        // The ring: the fixtures' `numWkSlicesPerDim_ {"out": 8}` verbatim, and their `in: 1` —
        // the scan axis NEVER splits. (numWkSlicesPerDim_ rides the op level, not the dsc body.)
        assert_eq!(d["numWkSlicesPerDim_"]["out"], 8, "{wire}: the ring");
        assert_eq!(
            d["numWkSlicesPerDim_"]["in"], 1,
            "{wire}: the reduce dim must stay whole per core — the fixtures carry in:1, and a \
             split scan would need the partial-reduce collision machinery the pair never binds"
        );
        assert_eq!(d["numCoresUsed_"], 8, "{wire}: ring 8 x y 1");
        // The layouts: the fixtures' INPUT/OUTPUT stick contracts, on the 3-dim form.
        assert_eq!(
            b["primaryDsInfo_"]["INPUT"]["layoutDimOrder_"],
            json!(["in", "y"]),
            "{wire}"
        );
        assert_eq!(
            b["primaryDsInfo_"]["INPUT"]["stickDimOrder_"],
            json!(["y"]),
            "{wire}"
        );
        assert_eq!(
            b["primaryDsInfo_"]["INPUT"]["stickSize_"],
            json!([64]),
            "{wire}"
        );
        assert_eq!(
            b["primaryDsInfo_"]["OUTPUT"]["layoutDimOrder_"],
            json!(["out", "y"]),
            "{wire}"
        );
        assert_eq!(
            b["primaryDsInfo_"]["OUTPUT"]["stickDimOrder_"],
            json!(["y"]),
            "{wire}"
        );
        assert_eq!(
            b["primaryDsInfo_"]["OUTPUT"]["stickSize_"],
            json!([64]),
            "{wire}"
        );
        // The computeOp row — the six core fields the fixtures share with every op. The five
        // scheduling fields the fixtures ALSO carry are the documented divergence: the dev
        // stack's SDSC-JSON validator rejects them outright, so the pair OMITS them exactly
        // like the generic path ([`OpFunc::is_topk_pair`] for the measurement).
        let c = &b["computeOp_"][0];
        assert_eq!(c["exUnit"], "sfp", "{wire}");
        assert_eq!(c["opFuncName"], wire, "{wire}");
        assert_eq!(c["attributes_"]["dataFormat_"], "SEN169_FP16", "{wire}");
        assert_eq!(c["attributes_"]["fidelity_"], "regular", "{wire}");
        assert_eq!(c["location"], "Inner", "{wire}");
        for field in [
            "auxLoopName",
            "isAtMainLoop",
            "isAtTop",
            "level",
            "opConsts",
        ] {
            assert!(
                c[field].is_null(),
                "{wire}: {field} must stay OMITTED — the fixtures carry it, but the dev \
                 stack's SDSC-JSON validator rejects the field outright (\"Unrecognized \
                 field\", measured on the tiny26 bake's dxp; value: {}). Re-adding it \
                 re-breaks the bake.",
                c[field]
            );
        }
        assert_eq!(
            c["inputLabeledDs"].as_array().map(Vec::len),
            Some(1),
            "{wire}: topk.ddl binds ONE input — a second operand makes the template unmatchable"
        );
        assert_eq!(
            c["outputLabeledDs"].as_array().map(Vec::len),
            Some(1),
            "{wire}"
        );
        assert!(
            c["indirectAccessIndexLabeledDs"]
                .as_array()
                .is_none_or(|a| a.is_empty()),
            "{wire}: the pair is not a gather — the index-side field is empty (omitted on the \
             wire when empty, the Vec::is_empty skip law)"
        );
        // constantInfo_ — the falsy empty OBJECT, the discriminator against the generic path.
        assert_eq!(
            b["constantInfo_"],
            json!({}),
            "{wire}: the fixtures ship the falsy empty OBJECT; the truthy \"{{}}\" string would \
             make dxp run the SFP constant-table setup for a template that reads no external \
             constant"
        );
    }
}

/// ⭐ THE DISCRIMINATING CONTROL: the SAME emission path, a GENERIC op, gets the truthy
/// string and the omitted scheduling row — so the pair's ONE divergence (`constantInfo_`)
/// is the pair's, not a global change every shipped descriptor silently inherited, and the
/// scheduling row stays omitted on the whole tree (the validator's law; the guard against
/// re-introducing the fixtures' row citing the fixtures again).
#[test]
fn the_generic_op_keeps_the_truthy_string_and_the_omitted_row() {
    let mut sid = -1i64;
    let e = try_assemble_transpose("ctrl", 64, 512, "t_in", "t_out", &mut sid, None)
        .expect("a splittable transpose emits");
    let j = serde_json::to_value(&e.op).unwrap();
    let generic = &j["dscs_"][0]["ctrl"];
    assert_eq!(
        generic["constantInfo_"],
        json!("{}"),
        "the generic path's empty constantInfo_ is the truthy STRING — the form that keeps the \
         SFP Newton-Raphson refine running. If this is the falsy object, the pair's exception \
         leaked into every op on the tree."
    );
    let c = &generic["computeOp_"][0];
    for field in [
        "auxLoopName",
        "isAtMainLoop",
        "isAtTop",
        "level",
        "coreExclude",
        "coreClExclude",
        "opConsts",
    ] {
        assert!(
            c[field].is_null(),
            "the generic row OMITS {field} — the torch-spyre-frontend law, and the \
             SDSC-JSON validator's (\"Unrecognized field\"). If it is present, the fixtures' \
             scheduling row leaked into every op on the tree (value: {}).",
            c[field]
        );
    }
}

/// ⭐ THE SPLIT LAW — the ring first (the largest divisor of the slot count that is ≤ 8, both
/// fixtures' own ring), then the token sticks to the budget, ODD stick counts included, and the
/// scan axis NEVER. Every shape in the sweep must EMIT: a refusal here fails the test loudly, so
/// the sweep cannot pass vacuously.
#[test]
fn the_split_is_the_ring_first_then_token_sticks() {
    // (candidates, slots, tokens, expect ring, expect y split, expect cores, label)
    const SWEEP: &[(u32, u32, u32, u64, u64, u64, &str)] = &[
        (
            64,
            8,
            64,
            8,
            1,
            8,
            "the fixtures' own geometry — the ring verbatim, one stick",
        ),
        (
            128,
            64,
            64,
            8,
            1,
            8,
            "the decode shape (one padded token stick)",
        ),
        (
            128,
            64,
            320,
            8,
            1,
            8,
            "the gemma-4 26b prefill: 5 sticks, but 8x5 > 32, so y stays whole",
        ),
        (128, 64, 128, 8, 2, 16, "two sticks split — 16 cores"),
        (
            128,
            64,
            192,
            8,
            3,
            24,
            "an ODD stick count splits too (3, not a power of two)",
        ),
        (128, 64, 256, 8, 4, 32, "the full budget — 32 cores"),
        (
            128,
            8,
            128,
            8,
            2,
            16,
            "the fixtures' own slot count at a splittable token count",
        ),
    ];
    for &(cand, slots, tokens, ring, y, cores, label) in SWEEP {
        let d = emit(
            OpFunc::TopKIndex,
            TopKGeom {
                candidates: cand,
                slots,
                tokens,
            },
        )
        .unwrap_or_else(|e| panic!("{label} refused: {e}"));
        assert_eq!(d["numWkSlicesPerDim_"]["out"], ring, "{label}");
        assert_eq!(d["numWkSlicesPerDim_"]["y"], y, "{label}");
        assert_eq!(
            d["numWkSlicesPerDim_"]["in"], 1,
            "{label}: the scan axis never splits"
        );
        assert_eq!(d["numCoresUsed_"], cores, "{label}");
        // The N_ extents ride along un-degenerate at every shape in the sweep.
        let b = body(&d);
        assert_eq!(b["N_"]["in_"], cand, "{label}");
        assert_eq!(b["N_"]["out_"], slots, "{label}");
        assert_eq!(b["N_"]["y_"], tokens, "{label}");
    }
}

/// ⛔ THE REFUSAL CONTROLS — a non-pair opFunc and a sub-stick token count are NAMED refusals,
/// never descriptors. A `sum` handed to the topk builder would emit a reduce-shaped descriptor
/// with the pair's ring law — semantically wrong, structurally accepted — so it must refuse; and
/// a token count that is not a whole stick is the whole-stick half of the stick law.
#[test]
fn a_non_pair_member_and_a_sub_stick_token_count_are_refused_by_name() {
    let err = emit(
        OpFunc::Sum,
        TopKGeom {
            candidates: 64,
            slots: 8,
            tokens: 64,
        },
    )
    .expect_err("a plain reduce must not ride the topk builder");
    assert!(
        err.contains("not a member of the vendor top-k pair"),
        "the refusal must name the pair contract (got: {err})"
    );
    let err = emit(
        OpFunc::TopKIndex,
        TopKGeom {
            candidates: 64,
            slots: 8,
            tokens: 63,
        },
    )
    .expect_err("a sub-stick token count must not emit");
    assert!(
        err.contains("not a whole multiple"),
        "the refusal must name the whole-stick law (got: {err})"
    );
}

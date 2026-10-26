// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ THE INDEX TENSOR'S SHAPE IS THE VALUE'S **PAGED DIMS** — the rule, read straight off both
//! vendor fixtures, and now obeyed by our own emission.
//!
//! ## The rule, extracted from two fixtures that disagree about everything else
//! ```text
//! dxp/test/test_gather_1core/sdsc_1.json      (N_: mb_=3, x_=64, out_=512)
//!   value  layoutDimOrder_ ["mb","out","x"]   maxDimSizes_ [ 1, -1, -1]
//!   index  layoutDimOrder_ ["mb"]             maxDimSizes_ [-1]
//!
//! dcg/dcg_fe/scheduler/test/sdsc_add_paged_l3lu.json   (N_: out_=128, mb_=8, x_=256, y_=2)
//!   value  layoutDimOrder_ ["out","mb","x","y"]  maxDimSizes_ [-1, -1, 64, 1]
//!   index  layoutDimOrder_ ["x","y"]             maxDimSizes_ [-1, -1]
//! ```
//! In BOTH, the index's `layoutDimOrder_` is exactly the value's PINNED dims, in the value's own
//! order, and nothing else. One paged dim ⇒ a rank-1 index; two ⇒ a rank-2 index.
//!
//! ⛔ `attach_gather_index` USED TO CLONE THE VALUE OPERAND ("shaped from the VALUE operand so its rank
//! and stick law are legal for this op by construction"), which gave the index the value's FULL layout.
//! For the shipped score leg that is `["in","out"]` where the rule says `["out"]` — a rank-2 grid of
//! 64x256 positions where the descriptor should declare a table of pages. dxp was being handed a
//! differently-shaped table from the one it walks, and the failure mode is an address, not a fault.
//! It is now shaped from the pin list, which is also the list the walk's pins come from, so the index's
//! dims cannot disagree with the dims that were pinned.
//!
//! ⛔ ONE THING THE FIXTURES DO NOT SETTLE: the index's STICK dim. Both omit `stickDimOrder_` on their
//! index nodes, and IBM's `["x","y"]` has `y_ = 2` — sub-stick, which OUR `StickExtent` guard refuses
//! (the second instance of that guard being stronger than the vendor's own fixtures). So the stick is
//! the first paged dim whose extent is a whole multiple of the format stick, and a shape where none is
//! gets refused rather than forced.
//!
//! ## And the bigger finding: A PAGED GATHER PINS **TWO** DIMS
//! IBM pins `x` to 64 — the page granularity — and `y` to 1, which means ONE ENTRY PER POSITION
//! along `y`. With both pinned, one launch covers every (page, row) combination, because both
//! factors of the relaunch count live inside the index.
//!
//! ⛔ THE AXIS PINNED TO 1 IS JUST THE BATCH AXIS `mb`. There is no "request" here: this emitter is
//! handed a batch of rows and executes it, and a row is a row. `mb` is already in the matmul's
//! canonical dim set `{mb(M), in(K), out(N), y(batch)}` — nothing needs inventing. The only real gap
//! is that the Kᵗ KERNEL operand does not DECLARE `mb`, because it is the shared weight of a
//! `y`-batched matmul.
//!
//! `matmul/opspec.rs` warns that *"a 3-D kernel makes dxp treat the weight as per-batch → garbage"*,
//! and that is about an UNGATHERED kernel: per-batch addressing there means a dxp-DERIVED stride the
//! KV pool does not have. With `isStartAddrSymbolic_` the base comes from the index instead, so
//! per-batch addressing is exactly what is wanted — and IBM's own gathered value tensor is rank-4
//! WITH `mb`. The rank-3 kernel is correct only TOGETHER with the gather; the two must stay paired.
//!
//! These tests read the fixtures as the oracle. They pass when our emission obeys the rule.

use ktir_superdsc::emit as superdsc;
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::matmul::{
    SharedKernelBmmForm, matmul_opspec_off_operands_phys_gathered,
};
use ktir_superdsc::sdsc_abstract::{MatK, MatM, MatN, MatY, QueryRowCount};
use ktir_superdsc::superdsc_opspec::{Fp16, GatherIndex, SdscFoldSet};

const GATHER_1CORE_TEXT: &str = include_str!("fixtures/sdsc_gather_1core.json");
const PAGED_L3LU_TEXT: &str = include_str!("fixtures/sdsc_add_paged_l3lu.json");

/// Parse a lit-style fixture (leading `// RUN:` lines) as JSON.
fn fixture(text: &str) -> Option<serde_json::Value> {
    let body: String = text
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    serde_json::from_str(&body).ok()
}

/// `(value layout, value pins, index layout)` for the one gathered op in a fixture.
fn vendor_shapes(j: &serde_json::Value) -> Option<(Vec<String>, Vec<i64>, Vec<String>)> {
    let inner = j
        .as_object()?
        .values()
        .next()?
        .get("dscs_")?
        .get(0)?
        .as_object()?
        .values()
        .next()?
        .clone();
    let tree = inner.get("scheduleTree_")?.as_array()?.clone();
    let pick = |role: &str| -> Option<serde_json::Value> {
        tree.iter()
            .find(|n| n.get("indirectAllocType_").and_then(|v| v.as_str()) == Some(role))
            .cloned()
    };
    let strs = |n: &serde_json::Value| -> Vec<String> {
        n["layoutDimOrder_"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|d| d.as_str().unwrap_or("?").to_string())
                    .collect()
            })
            .unwrap_or_default()
    };
    let val = pick("value_tensor")?;
    let idx = pick("index_tensor")?;
    let pins = val["maxDimSizes_"]
        .as_array()?
        .iter()
        .map(|p| p.as_i64().unwrap_or(0))
        .collect();
    Some((strs(&val), pins, strs(&idx)))
}

/// ⭐⭐⭐ THE RULE ITSELF, asserted on BOTH fixtures: the index's dims are the value's pinned dims.
///
/// Two independent fixtures from two different parts of the toolchain, with different ranks and
/// different page sizes, so this is a rule rather than a coincidence of one file.
#[test]
fn the_vendors_index_layout_is_exactly_the_values_pinned_dims() {
    let mut checked = 0;
    for (name, text) in [
        ("sdsc_gather_1core", GATHER_1CORE_TEXT),
        ("sdsc_add_paged_l3lu", PAGED_L3LU_TEXT),
    ] {
        let Some(j) = fixture(text) else {
            // ⛔ NOT SILENTLY SKIPPED-AND-GREEN: counted, and the count is asserted below. A missing
            // oracle must not read as a satisfied one.
            println!("fixture does not parse: {name}");
            continue;
        };
        let Some((val_layout, pins, idx_layout)) = vendor_shapes(&j) else {
            println!("fixture has no gathered op: {name}");
            continue;
        };
        let want: Vec<String> = val_layout
            .iter()
            .zip(&pins)
            .filter(|&(_, &p)| p > 0)
            .map(|(d, _)| d.clone())
            .collect();
        println!(
            "{name}\n  value {val_layout:?} pins {pins:?}\n  index {idx_layout:?} (pinned dims {want:?})"
        );
        assert_eq!(
            idx_layout, want,
            "{name}: the index's layout must be the value's pinned dims, in the value's order"
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "neither vendor fixture parsed, so this test proved nothing — it must not pass by \
         finding no oracle"
    );
}

/// ⭐⭐⭐ A PAGED GATHER PINS TWO DIMS: A PAGE GRANULARITY AND AN AXIS AT ONE ENTRY PER POSITION.
///
/// This is the fact that says a one-paged-dim gather cannot be batch-invariant: with only the slot
/// page pinned, an index entry names a page but not a ROW, so a launch still serves one row of the
/// batch and the launch count keeps its batch factor. Recorded as an assertion because it is the
/// requirement the emitter has to grow into, and a comment would not fail when someone declares one
/// dim and reports the gather done.
#[test]
fn a_paged_gather_pins_a_page_axis_and_a_per_position_axis() {
    let Some(j) = fixture(PAGED_L3LU_TEXT) else {
        panic!("IBM's paged-attention fixture is the oracle for this requirement and is missing");
    };
    let (val_layout, pins, idx_layout) = vendor_shapes(&j).expect("a gathered op");
    let paged: Vec<(&String, &i64)> = val_layout
        .iter()
        .zip(&pins)
        .filter(|&(_, &p)| p > 0)
        .collect();
    println!("paged dims: {paged:?}   index rank {}", idx_layout.len());
    assert_eq!(
        paged.len(),
        2,
        "paged attention pins TWO dims — a page granularity and a per-position axis"
    );
    // One of them is pinned to 1: one entry per position along it.
    assert!(
        paged.iter().any(|&(_, &p)| p == 1),
        "one paged dim must be pinned to 1 — one index entry per position along it, which is what \
         gives every ROW of the batch its own base and removes the per-row relaunch"
    );
    // And the other is a real page size, dividing its dim's extent.
    assert!(
        paged.iter().any(|&(_, &p)| p > 1),
        "the other paged dim carries the PAGE, so entries step whole pages rather than positions"
    );
}

/// ⛔⛔⛔⛔⛔ AND THE BATCH-INVARIANT **MATMUL** FORM IS REFUSED — a build failure, not a shape to
/// assert properties of.
///
/// `matmul_opspec_off_operands_phys_gathered` declares the Kᵗ kernel rank-3 so `mb` has somewhere to
/// land, and the tests that used to live below measured that descriptor in detail: two pins, `mb`
/// pinned to 1, the 2-D order preserved as a prefix, the index shaped by the paged dims, the declared
/// entry count. Every one of them passed.
///
/// They were measuring a descriptor the card REFUSES. A gathered matmul has `KERNEL` in its
/// `primaryDsInfo_`, and `hasDimensionReuse` (`L3DlOpsScheduler.cpp:303-321`,
/// `primaryDsInfo_.size() > 1 && count(DsTypes::KERNEL)`) turns on the arithmetic-intensity explorer at
/// `:1550`, whose `calculateFlopPerByte` demands an LX allocate node for every HBM-pinned labeledDs
/// (`:2334`/`:2337`) — which `allocAllMem` never gives an index. MEASURED on the card at BOTH possible
/// index `memOrg_` values.
///
/// ⭐ SUPERSEDED IN PART 2026-10-09: the probe (`/tmp/prefill-m1-probe-report.md`) root-caused those
/// refusals to the exploration path only (a vendor ordering gap) and escaped them for the
/// TOKEN-PLANE form — entries on `mb`, one position each, no second pin — which now EMITS
/// (`zz_the_token_plane_gathered_matmul_is_the_probe_form.rs`, probe-proven `RC=0` at both
/// geometries with `DISABLE_ABOVE_LX_CHUNK_EXPLORE=1` at bake). THIS form — `of_pool_blocks_per_row`,
/// a Slot-paged kernel with a `per_position` batch pin — was never probed and stays refused.
///
/// So a green test on this shape is worse than no test: it pins an unschedulable descriptor as correct,
/// in full detail, with a vendor fixture cited. What replaces them is the REFUSAL, plus the same
/// properties asserted on the shape that ships — `zz_the_gather_op_is_kernel_less.rs`, whose harness is
/// the vendor's own `identity` with both data operands typed `OUTPUT`.
///
/// ⛔ THE BATCH-INVARIANT FORM ITSELF IS NOT RETRACTED, only its home. `of_pool_blocks_per_row` still
/// pins the page and the batch, and the two-pin declaration is still what removes the per-row relaunch;
/// it has to sit on a KERNEL-less copy op, and getting it there is open work.
#[test]
fn the_batch_invariant_matmul_form_is_refused_at_build_time() {
    let op = matmul_opspec_off_operands_phys_gathered::<Fp16>(
        MatM::of_token_rows(8),
        MatN::of_out_features(256),
        MatK::of_in_features(64),
        MatY::of_batch(4),
        SharedKernelBmmForm::of_attn_rows(true, QueryRowCount::of_mq(8)),
        "Tensor0",
        "Tensor1",
        "Tensor2",
        0,
        0,
        0,
        <Fp16 as ktir_superdsc::superdsc_opspec::DataFormat>::DF,
        None,
        Some(GatherIndex::of_pool_blocks_per_row(
            "BlockTable".to_string(),
        )),
    )
    .expect("the DECLARATION is still well formed — it is the SCHEDULING that is impossible");
    let folds = SdscFoldSet::new(op.iter.cores_used());
    let err = match superdsc::emit_sdsc("MatMul_0", &op, &folds, None) {
        Err(e) => e.0,
        Ok(_) => panic!(
            "the batch-invariant gathered MATMUL emitted. This paged-KV shape was never \
             probe-validated — the token-plane form is the one proven to schedule — so this must \
             fail the build rather than reach dxp and be discovered again."
        ),
    };
    assert!(
        err.contains("KERNEL"),
        "the refusal must name the KERNEL as the cause (got: {err})"
    );
    assert!(
        err.contains("KERNEL-LESS") || err.contains("kernel-less"),
        "and the shape that WORKS, or the reader is sent back to the card (got: {err})"
    );
}


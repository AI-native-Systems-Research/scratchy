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

const GATHER_1CORE: &str = "/Users/nickm/git/deeptools/dxp/test/test_gather_1core/sdsc_1.json";
const PAGED_L3LU: &str =
    "/Users/nickm/git/deeptools/dcg/dcg_fe/scheduler/test/sdsc_add_paged_l3lu.json";

/// Parse a lit-style fixture (leading `// RUN:` lines) as JSON.
fn fixture(path: &str) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(path).ok()?;
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
    for path in [GATHER_1CORE, PAGED_L3LU] {
        let Some(j) = fixture(path) else {
            // ⛔ NOT SILENTLY SKIPPED-AND-GREEN: counted, and the count is asserted below. A missing
            // oracle must not read as a satisfied one.
            println!("fixture unavailable: {path}");
            continue;
        };
        let Some((val_layout, pins, idx_layout)) = vendor_shapes(&j) else {
            println!("fixture has no gathered op: {path}");
            continue;
        };
        let want: Vec<String> = val_layout
            .iter()
            .zip(&pins)
            .filter(|&(_, &p)| p > 0)
            .map(|(d, _)| d.clone())
            .collect();
        println!(
            "{path}\n  value {val_layout:?} pins {pins:?}\n  index {idx_layout:?} (pinned dims {want:?})"
        );
        assert_eq!(
            idx_layout, want,
            "{path}: the index's layout must be the value's pinned dims, in the value's order"
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "neither vendor fixture was readable, so this test proved nothing — it must not pass by \
         finding no oracle. Fixture paths are absolute into ~/git/deeptools."
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
    let Some(j) = fixture(PAGED_L3LU) else {
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

/// ⛔⛔⛔⛔⛔ AND THE BATCH-INVARIANT **MATMUL** FORM IS UNSCHEDULABLE — a build failure, not a shape to
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
/// index `memOrg_` values; there is no declaration that escapes it.
///
/// So a green test on that shape is worse than no test: it pins an unschedulable descriptor as correct,
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
            "a gathered MATMUL emitted. deeptools refuses to schedule it — measured as a bake failure \
             on the card at both index memOrg_ values — so this must fail the build rather than reach \
             dxp and be discovered again."
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

/// ⭐⭐⭐⭐⭐ THE INDEX DECLARES **ENTRIES**, AND EVERY CORE READS THE SAME TABLE — the law dxp's own
/// index→address conversion forces, and the one our emission broke for as long as the gather existed.
///
/// ## What dxp does, read off the C++ and MEASURED on the pod
/// `createIdx2AddrSdsc` (`dbo/src/Transforms/sdsc_bundle/GatherIndexConversion.cpp`) builds a
/// SINGLE-CORE SDSC whose `N_` is the gather op's `N_` divided by the value tensor's page size per dim
/// (`AllocateNode::getPageSize`, `dsc/dsc2.cpp:4493`) and rounded up to the index's stick, then
/// converts that many entries from ONE CONTIGUOUS RUN at the MINIMUM of the index's per-core start
/// addresses into one contiguous run. `allocateAndModifyGather`
/// (`dbo/src/Utils/sdsc_bundle/GatherBuffers.cpp`) sizes that output at
/// `getBufferCapacityForNode(indexAlloc)` and then shifts every per-core index address by ONE constant,
/// preserving their spread.
///
/// Baked on the pod (`dxp_standalone -b sentient`, `DBO_DEBUG=1`) against the shipped `mb = 2048`,
/// page `64` gather:
/// ```text
/// idx2addr_lds1_sdsc_0   N_ mb_ = 32          <- 2048 / 64, one core
///   input  51540193920   output 120259084288  <- 32 entries = 128 BYTES, contiguous
/// allocate-Tensor1_hbm (index, now "address")
///   BEFORE  120259084288 + 256*c   for c in 0..32   <- 8192 B of reads into a 128 B buffer
///   AFTER   120259084288           for every core
/// ```
/// and in the DataflowIR the word each core takes out of its IBR is `c` in BOTH cases
/// (`get_logical_memory_view %ibr, %c<c>`) — dxp derives that from the op's own work-slice, never from
/// the index allocation. So the per-core start decides only WHICH 32 words are loaded, and declaring it
/// from the op's `mb` made 31 of 32 cores load words nothing had converted and use them as ABSOLUTE
/// stick addresses. The table's pad entry `0` becomes address `0` that way: a PCIe bus-master abort,
/// not a wrong answer.
///
/// ## ⛔ AND THE 32 IS A CEILING, NOT A COINCIDENCE — WHICH IS WHY THE PASS IS CUT INTO RUNS
/// `L3DlOpsScheduler` says it twice for the IBR-driven schedule it generates once
/// `indexTensorType_ == ADDRESS`:
/// ```text
/// // 2.1. The index tensor is transferred from HBM to L3LUIBR in the granularity of one stick.
/// DT_CHECK_MSG(size <= indexLdsCumulativeStickSizes.at(dim),
///              "IBR stick dimension size without rounding should always be not greater than
///               the stick size.");
/// offsetInBytes += wkSliceId * dimIbrSize * indexLds.wordLength;
/// offsetInBytes  = offsetInBytes % numBytesInStick;      // ⬅ ONE 128-BYTE STICK
/// ```
/// One stick transfer, and each core's read offset inside it taken MODULO that stick — so
/// `wkSlices × per-core entries` above 32 words does not fault and does not refuse, it WRAPS, and the
/// cores past the wrap gather another core's pages. That is what separated the two card outcomes
/// exactly (32 entries `solo=EXACT`, 128 entries every row's first token wrong), and it is why one
/// gather op now serves ONE RUN of at most `CopyDims::ENTRIES_PER_OP` entries.
///
/// ⛔ NOT `dsm/progCorrection.cpp:524`, which an earlier version of this note cited for the same law.
/// That file is `runDsmProgCorrect` / `insertProgCorrectNodes` — the PROGRAM-CORRECTION scatter, a
/// different feature that happens to spell the same 32 — and its `DT_CHECK(32 % numCores == 0)` says
/// nothing about this op. A citation that merely contains the right number is not evidence.
#[test]
fn the_index_declares_entries_and_every_core_reads_the_same_table() {
    use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::gather_copy_opspec;
    use ktir_superdsc::sdsc_abstract::{GatherScratch, PagedKvPool, SlotCount, SlotWindow};
    use ktir_superdsc::superdsc_opspec::GatherIndex;

    // The SHIPPED geometry, from the pool the fold actually reads — so the numbers below are the
    // descriptor's, not a harness's.
    let pool = PagedKvPool::new(8, 64);
    let scratch = GatherScratch::of_fold_pass(
        pool,
        SlotWindow::count_in(SlotCount::new(256)),
        ktir_superdsc::sdsc_abstract::QueryRowCount::of_mq(8),
    )
    .expect("the shipped fold pass admits a flat gather");
    // ⭐⭐⭐ THE RUNS PARTITION THE PASS, AND EVERY ONE OF THEM FITS ONE INDEX STICK. The pass's rows are
    // still the host's table (`gather_index_table` writes `scratch.rows()` of them, window-major), so
    // the two halves of the gather agree on every row — the cut is in how many OPS read that table, not
    // in what a row index means.
    let runs: Vec<_> = scratch.copies().collect();
    assert_eq!(
        runs.len() as u32,
        scratch.copy_count(),
        "`copy_count` must be what `copies()` yields, or the diagnostics price a different cut than \
         the emitter makes"
    );
    let cap = ktir_superdsc::sdsc_abstract::CopyDims::ENTRIES_PER_OP;
    let mut covered = 0u32;
    for (i, r) in runs.iter().enumerate() {
        let e = r
            .dims()
            .entries()
            .expect("the page divides `mb` by construction");
        assert!(
            e <= cap && r.dims().fits_one_index_stick(),
            "run {i} declares {e} entries and one gather op's index is ONE {cap}-entry stick — above \
             it dxp's per-core IBR offset wraps and those cores gather another core's pages"
        );
        assert_eq!(
            r.index_base().entries(),
            covered,
            "run {i} must start exactly where run {} ended, or a row of the host's table is read \
             twice and another never",
            i.saturating_sub(1)
        );
        covered += e;
    }
    assert_eq!(
        covered,
        scratch.rows(),
        "the runs must cover the pass's rows exactly — a short cover leaves rows holding whatever the \
         allocator handed out, which is a valid block number"
    );

    // The FIRST run is the one emitted below: at the shipped geometry it is a full stick, and every
    // assertion past here is about the descriptor shape a run has.
    let d = runs[0].dims();
    let entries = d.entries().expect("the page divides `mb` by construction");
    let op = gather_copy_opspec(
        "KtPool",
        "KtScratch",
        d.mb(),
        d.out(),
        GatherIndex::of_scratch_rows("BlockTable".to_string(), d.page(), runs[0].index_base()),
    )
    .expect("the shipped gather-copy builds");
    let idx_pos = op.indirect.expect("a declaration").index;
    let folds = SdscFoldSet::new(op.iter.cores_used());
    let e = superdsc::emit_sdsc("attn_gkt", &op, &folds, None).expect("it emits");
    let j = serde_json::to_value(&e).unwrap();
    let tree = &j["dscs_"][0]["attn_gkt"]["scheduleTree_"];
    let idx = &tree[idx_pos];
    assert_eq!(
        idx["indirectAllocType_"], "index_tensor",
        "operand {idx_pos} must be the index node"
    );

    // ── the DECLARED EXTENT is the entry count, not the op's `mb` ──
    let walk: Vec<i64> = idx["maxDimSizes_"]
        .as_array()
        .expect("a walk")
        .iter()
        .map(|v| v.as_i64().unwrap_or(0))
        .collect();
    assert_eq!(
        walk,
        vec![i64::from(entries)],
        "the index's walk must be its ENTRY count ({entries}), not the op's {} `mb` positions — dxp \
         converts exactly `mb / page` entries and allocates exactly that many",
        d.mb()
    );

    // ── and EVERY CORE shares one start address ──
    let addrs: Vec<i64> = idx["startAddressCoreCorelet_"]["data_"]
        .as_object()
        .expect("per-core starts")
        .values()
        .map(|v| {
            v.as_str()
                .and_then(|s| s.parse::<i64>().ok())
                .or_else(|| v.as_i64())
                .expect("an address")
        })
        .collect();
    assert!(
        addrs.len() > 1,
        "this must be the MULTI-CORE case or it proves nothing (got {} core(s))",
        addrs.len()
    );
    let distinct: std::collections::BTreeSet<i64> = addrs.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        1,
        "every core must read the SAME index table — a per-core spread puts all but core 0 outside \
         the {} bytes dxp converts (got {} distinct starts: {:?})",
        entries * 4,
        distinct.len(),
        distinct
    );

    // ── sanity: the VALUE side still strides per core, so this is not a global flattening ──
    let val = tree
        .as_array()
        .expect("a tree")
        .iter()
        .find(|n| n["indirectAllocType_"] == "value_tensor")
        .expect("a value node");
    let val_distinct: std::collections::BTreeSet<String> = val["startAddressCoreCorelet_"]["data_"]
        .as_object()
        .expect("per-core starts")
        .values()
        .map(|v| v.to_string())
        .collect();
    assert!(
        val_distinct.len() > 1,
        "the gathered VALUE tensor still addresses per core — only the INDEX is the whole table on \
         every core"
    );
}

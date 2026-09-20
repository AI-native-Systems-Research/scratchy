// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ COMPUTE `skip_addr` THE WAY dxp DOES, FROM OUR OWN EMITTED JSON — so the one number the
//! gather's correctness hangs on is checkable WITHOUT a card.
//!
//! ## Why this instrument has to exist
//! `skip_addr` is the HBM distance between consecutive index entries. dxp does not take it from us; it
//! DERIVES it from the value tensor's per-dim capacities (`getBufferCapacityForNodePerDim`), where an
//! unbounded dim contributes its per-core datastage extent and the pinned dim is clamped to the page.
//! Then `ConvertData_gather_idx` computes `addr = idx * skip_addr + base_addr`.
//!
//! ⛔ SO A WRONG `skip_addr` IS NOT A BAKE FAILURE. It is a clean build whose every index step lands a
//! fraction of a page short — fluent text assembled from another request's keys, with every counter
//! green. That is the single most dangerous outcome available here, and it is invisible to "does it
//! compile", "does it bake", and "is the output non-garbage".
//!
//! The inputs to dxp's derivation are all in the JSON we already emit (`N_`, `numWkSlicesPerDim_`,
//! `maxDimSizes_`), so we can compute the same product locally and assert it against the pool's real
//! stride. That turns "hope the declaration was right" into arithmetic.

use ktir_superdsc::emit as superdsc;
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::matmul_opspec;
use ktir_superdsc::superdsc_opspec::{KernelAxis, PageExtent, SdscFoldSet};

/// Granite-3.1-2b's KV geometry, the bundle this work targets.
const NKVH: i64 = 8;
const HD: i64 = 64;
const PAGE_SLOTS: i64 = 256;

/// One layer's page in ELEMENTS — what one index step must advance by.
const LAYER_PAGE_ELEMS: i64 = NKVH * HD * PAGE_SLOTS;

/// dxp's `getBufferCapacityForNodePerDim`, reimplemented over the JSON we emit.
///
/// Per dim: the per-core datastage extent is `N_[d] / numWkSlicesPerDim_[d]` (the work division is
/// what makes a dim per-core). A dim PINNED in `maxDimSizes_` is clamped to its pin; an unbounded
/// (`-1`) dim contributes its full per-core extent. `skip_addr` is the product.
///
/// Returns `None` if the JSON lacks what the computation needs, rather than substituting a 1 — a
/// missing extent silently contributing 1 is how an instrument reports a plausible wrong number.
fn skip_addr_elems(dsc: &serde_json::Value, node: &serde_json::Value) -> Option<i64> {
    let n = dsc.get("N_")?.as_object()?;
    let slices = dsc.get("numWkSlicesPerDim_").and_then(|v| v.as_object());
    let layout = node.get("layoutDimOrder_")?.as_array()?;
    let pins = node.get("maxDimSizes_")?.as_array()?;
    if layout.len() != pins.len() {
        return None;
    }

    let mut product: i64 = 1;
    for (d, pin) in layout.iter().zip(pins) {
        let dim = d.as_str()?;
        // `N_` keys are the dim name with a trailing underscore (`out` -> `out_`).
        let extent = n.get(&format!("{dim}_"))?.as_i64()?;
        let split = slices
            .and_then(|s| s.get(dim))
            .and_then(|v| v.as_i64())
            .unwrap_or(1)
            .max(1);
        let per_core = extent / split;
        product *= match pin.as_i64()? {
            -1 => per_core,       // unbounded: the whole per-core extent
            p => per_core.min(p), // pinned: clamped to the page
        };
    }
    Some(product)
}

fn dsc_and_node<'a>(
    j: &'a serde_json::Value,
    op: &str,
    arg: usize,
) -> (&'a serde_json::Value, &'a serde_json::Value) {
    let dsc = &j["dscs_"][0][op];
    (dsc, &dsc["scheduleTree_"][arg])
}

/// ⭐ THE INSTRUMENT WORKS — validated against a case whose answer is known by construction.
///
/// A gather-free operand has every dim `-1`, so its "skip_addr" is simply the product of its per-core
/// extents. Asserting that first means a later surprising number is a real finding rather than a bug
/// in this file. ⛔ An instrument that has never been checked against a known answer is not an
/// instrument.
#[test]
fn the_instrument_reproduces_a_known_product() {
    let op = matmul_opspec(384, 384, 64, 16, "Tensor0", "Tensor1", "Tensor2").expect("a matmul");
    let folds = SdscFoldSet::new(op.iter.cores_used());
    let e = superdsc::emit_sdsc("MatMul_0", &op, &folds, None).expect("emits");
    let j = serde_json::to_value(&e).unwrap();
    let (dsc, node) = dsc_and_node(&j, "MatMul_0", 0);

    let computed = skip_addr_elems(dsc, node).expect("the JSON carries what the derivation needs");

    // Independently: the product of this operand's own per-core extents.
    let n = dsc["N_"].as_object().unwrap();
    let layout = node["layoutDimOrder_"].as_array().unwrap();
    let expect: i64 = layout
        .iter()
        .map(|d| {
            let dim = d.as_str().unwrap();
            let extent = n[&format!("{dim}_")].as_i64().unwrap();
            let split = dsc["numWkSlicesPerDim_"]
                .get(dim)
                .and_then(|v| v.as_i64())
                .unwrap_or(1)
                .max(1);
            extent / split
        })
        .product();
    assert_eq!(
        computed, expect,
        "with every dim unbounded, skip_addr is just the per-core product"
    );
    assert!(
        computed > 0,
        "a zero product would make every index step a no-op"
    );
}

/// ⭐⭐⭐⭐⭐ THE SHIPPED GATHER-COPY'S `skip_addr` **IS ONE POOL STICK BLOCK** — dxp's own derivation, run
/// over our own emitted JSON, against the pool's own `stick_block_elems`.
///
/// ⛔ THIS IS THE ONE NUMBER THE WHOLE GATHER HANGS ON AND THE CARD CANNOT CHECK IT. `skip_addr` is the
/// HBM distance between consecutive index entries; the host stages entries as global stick-block numbers
/// (`gather_entries_per_page` divides a page by `stick_block_bytes`), so if the derived distance is
/// anything else, every index step lands a fraction of a block off — a clean bake producing fluent text
/// from inside the previous block, with every counter green.
///
/// ⛔ AND IT IS A FUNCTION OF THE WORK DIVISION, WHICH IS WHY THE GEOMETRY HERE IS THE REAL ONE. dxp
/// clamps the pinned axis to `min(per_core_extent, page)`, so a split that hands a core fewer than `page`
/// positions SHRINKS the entry. Measured at the two widths that bracket the effect: bs=8 (64 entries, the
/// split lands on 32 cores at 2 entries each) and bs=2 (16 entries), where an `mb`-position split rather
/// than an ENTRY split would halve `skip_addr`.
#[test]
fn the_shipped_gather_copys_skip_addr_is_one_pool_stick_block() {
    use ktir_superdsc::sdsc_abstract::{
        GatherScratch, PagedKvPool, QueryRowCount, SlotCount, SlotWindow,
    };
    let pool = PagedKvPool::new(NKVH as usize, HD as usize);
    for mq in [2u32, 8] {
        for swept in [64u32, 128, 256] {
            let scratch = GatherScratch::of_fold_pass(
                pool,
                SlotWindow::count_in(SlotCount::new(swept)),
                QueryRowCount::of_mq(mq),
            )
            .expect("granite's geometry admits the flat copy");
            // ⭐ ONE COPY OF THE PASS, NOT THE PASS — `scratch.copies()` cuts it into one op per index
            // stick and every one of them has the same `skip_addr`, because `skip_addr` is
            // `page × the per-core extents of the UNPINNED dims` and the cut only shortens `mb`, the
            // pinned one. Measuring the first run is therefore measuring all of them, and it is the only
            // shape that exists: a whole-pass copy is refused at build (`CopyDims::ENTRIES_PER_OP`).
            let cp = scratch
                .copies()
                .next()
                .expect("a pass has at least one run");
            let op = ktir_superdsc::ir::bridge::tiled_op_sdsc_op::gather_copy_opspec(
                "Tensor0",
                "Tensor2",
                cp.dims().mb(),
                ktir_superdsc::sdsc_abstract::POOL_STICK,
                ktir_superdsc::superdsc_opspec::GatherIndex::of_scratch_rows(
                    "BlockTable".to_string(),
                    PageExtent::of_positions(scratch.entry_page()),
                    cp.index_base(),
                ),
                ktir_superdsc::superdsc_opspec::DestEntry::of_entries(cp.dest_entry()),
            )
            .expect("the gather-copy op builds");
            let value_idx = op.indirect.expect("a declared gather").value;
            let folds = SdscFoldSet::new(op.iter.cores_used());
            let e = superdsc::emit_sdsc("Gather_0", &op, &folds, None).expect("emits");
            let j = serde_json::to_value(&e).unwrap();
            let (dsc, node) = dsc_and_node(&j, "Gather_0", value_idx);
            let got = skip_addr_elems(dsc, node).expect("derivable");
            assert_eq!(
                got,
                pool.stick_block_elems() as i64,
                "mq={mq} swept={swept}: skip_addr must be ONE pool stick block \
                 ({} elems) — the unit the host's entries are counted in. Got {got}, so every index \
                 step would land {} of a block off.",
                pool.stick_block_elems(),
                got as f64 / pool.stick_block_elems() as f64,
            );
            // And the entry is exactly the scratch's row, which is what the matmul's `y` steps.
            assert_eq!(
                got as u32,
                scratch.cols(),
                "the entry and the scratch's block must be one number"
            );
        }
    }
}

/// ⭐⭐⭐⭐⭐ THE SAME DERIVATION OVER THE **SHIPPED** DOOR — [`PageScratch`], the one the decode batch
/// actually emits.
///
/// ⛔⛔⛔ AND THIS IS THE TEST THAT WAS MISSING, WHICH IS WHY A PLANE-SIZED PIN SHIPPED. The measurement
/// above is over [`GatherScratch`] — the window-granular door, which the batched path stopped emitting
/// when the gather went page-granular. It stayed GREEN while the live door's pin was a whole page PLANE,
/// i.e. `nkvh * PAGE_SLOTS / POOL_STICK` = 32× the unit the host's entries are counted in. A green test
/// over a dead door is not coverage; it is a divergence pinned as correct.
///
/// Two facts, from the one emission:
/// 1. `skip_addr` is ONE POOL STICK BLOCK — the unit `gather_entries_per_page` counts in, so the host's
///    entry and the card's address are one derivation rather than two.
/// 2. The op's work division admits MORE THAN ONE CORE. That is not a performance property: a pinned axis
///    with one entry gives dxp one core, whose double-buffered chunk is then measured against LX and
///    refused at bake (`L3DlOpsScheduler.cpp:1534`, `isDoubleBuffering`).
#[test]
fn the_shipped_page_gather_copys_skip_addr_is_one_pool_stick_block() {
    use ktir_superdsc::sdsc_abstract::{PageScratch, PagedKvPool, QueryRowCount};
    // ⭐⭐⭐⭐⭐ SWEPT OVER `nkvh` x `hd`, BECAUSE `skip_addr` IS `hd * POOL_STICK` AND THE PIN IS `hd`
    // SUB-ROWS — two quantities that are BOTH `POOL_STICK` at hd=64 and neither of them at hd=128. This
    // measurement was taken at the single point (nkvh=8, hd=64) where the pin, the stick and the IBR
    // width all read 64/32, which is precisely how "the pin is one stick" and "the pin is `hd`" stayed
    // indistinguishable. `nkvh` is swept too: at nkvh>8 a request's run exceeds one IBR stick, so the
    // pass is several ops per request and EVERY one of them must still derive this same `skip_addr`.
    for (nkvh, hd) in [
        (8i64, 64i64),
        (8, 128),
        (8, 256),
        (4, 128),
        (16, 128),
        (32, 64),
    ] {
        let pool = PagedKvPool::new(nkvh as usize, hd as usize);
        for mq in [2u32, 8] {
            let scratch = PageScratch::of_pass(pool, QueryRowCount::of_mq(mq))
                .expect("granite's geometry admits the page copy");
            for cp in scratch.copies() {
                let op = ktir_superdsc::ir::bridge::tiled_op_sdsc_op::gather_copy_opspec(
                    "Tensor0",
                    "Tensor2",
                    cp.dims().mb(),
                    ktir_superdsc::sdsc_abstract::POOL_STICK,
                    ktir_superdsc::superdsc_opspec::GatherIndex::of_scratch_rows(
                        "BlockTable".to_string(),
                        PageExtent::of_positions(
                            u32::try_from(scratch.entry_page()).expect("a stick-sized pin"),
                        ),
                        cp.index_base(),
                    ),
                    ktir_superdsc::superdsc_opspec::DestEntry::of_entries(cp.dest_entry()),
                )
                .expect("the page gather-copy op builds");
                let value_idx = op.indirect.expect("a declared gather").value;
                let cores = op.iter.cores_used().get();
                let folds = SdscFoldSet::new(op.iter.cores_used());
                let e = superdsc::emit_sdsc("PageGather_0", &op, &folds, None).expect("emits");
                let j = serde_json::to_value(&e).unwrap();
                let (dsc, node) = dsc_and_node(&j, "PageGather_0", value_idx);
                let got = skip_addr_elems(dsc, node).expect("derivable");
                assert_eq!(
                    got,
                    pool.stick_block_elems() as i64,
                    "nkvh={nkvh} hd={hd} mq={mq}: the SHIPPED copy's skip_addr must be ONE pool stick \
                 block ({} elems) — the unit the host's entries are counted in. Got {got}, i.e. {}× \
                 off, which lands every index step inside a different page with a clean bake.",
                    pool.stick_block_elems(),
                    got as f64 / pool.stick_block_elems() as f64,
                );
                // ⭐ AND IT IS `hd * POOL_STICK`, SPELLED AGAINST `hd` — the pin is `hd` sub-rows and the
                // derived stride is `pin * out`, so at hd=128 both DOUBLE and neither is the stick.
                assert_eq!(
                    got,
                    hd * ktir_superdsc::sdsc_abstract::POOL_STICK as i64,
                    "nkvh={nkvh} hd={hd} mq={mq}: skip_addr is `hd * POOL_STICK`, a quantity that MOVES \
                 with the head dim — it equals POOL_STICK² only at hd=64"
                );
                assert!(
                    cores > 1,
                    "nkvh={nkvh} hd={hd} mq={mq}: the copy is planned onto {cores} core(s). A pinned axis \
                 holding ONE entry admits no core split, and dxp then measures that one core's \
                 double-buffered chunk against LX — which is the bake refusal 'The initial chunk \
                 parameters must fit in LX for SuperDSC' (L3DlOpsScheduler.cpp:1534)."
                );
                // ⭐ AND THE PER-CORE DOUBLE-BUFFERED CHUNK FITS LX, measured from the op's OWN planned core
                // count rather than from the 2048-sub-row ceiling the LX refusal was once read as. Source +
                // destination, twice over for double buffering.
                let per_core_bytes = (cp.dims().mb() as u64 / cores as u64)
                    * ktir_superdsc::sdsc_abstract::POOL_STICK as u64
                    * 2
                    * 4;
                assert!(
                    per_core_bytes <= ktir_superdsc::superdsc_opspec::USABLE_LX_BYTES,
                    "nkvh={nkvh} hd={hd} mq={mq}: one core's double-buffered chunk is {per_core_bytes} B \
                 over {} cores, against LX's {} B. `getInitialChunkParams` starts from the CORE data \
                 stage, so this — not the op's whole footprint — is what dxp measures.",
                    cores,
                    ktir_superdsc::superdsc_opspec::USABLE_LX_BYTES,
                );
            }
        }
    }
}

/// ⭐⭐⭐ THE PIN COLLAPSES THE INDEXED DIM TO ITS PAGE, which is the mechanism the test above rests on.
///
/// With the entry dim pinned, `skip_addr` becomes `page × the product of the OTHER dims` — i.e. exactly
/// one entry's size. This is the mechanism by which "declare the pin" and "an index step advances one
/// entry" are the same statement, isolated from the pool's geometry.
#[test]
fn pinning_the_entry_dim_makes_skip_addr_one_entrys_size() {
    // ⛔ ON THE **GATHER-COPY** OP, NOT A MATMUL. This measured the pin on `matmul_opspec`, which
    // `emit_sdsc` now refuses at build time: deeptools cannot schedule a gather on an op that has a
    // KERNEL (`hasDimensionReuse` -> the arithmetic-intensity explorer -> an LX allocate node an index
    // never gets; measured as a bake refusal on the card at both index `memOrg_` values). The
    // arithmetic being tested — a pinned dim contributes its page and `skip_addr` is the product of
    // the rest — is a property of the DECLARATION, identical on either op, so the measurement survives
    // the move intact. See `zz_the_gather_op_is_kernel_less.rs`.
    let axis = KernelAxis::Batch;
    let entry_dim = axis.dim();
    // ONE POSITION per entry, so the pinned dim contributes exactly 1 and the assertion below — that
    // `skip_addr` is the product of the OTHER dims — is the statement being tested. A page larger than
    // 1 makes the pinned dim contribute the page, which is a different (also correct) arithmetic and
    // is measured by `the_shipped_gather_copys_skip_addr_is_one_pool_stick_block` above.
    //
    // ⛔ `out` IS ONE STICK, WHICH THE BUILDER NOW REQUIRES. It was 384 — a width at which a `[mb, out]`
    // operand classifies STICK-MAJOR and each gathered block is scattered `rows*64` apart instead of
    // contiguous. That is invisible in a `skip_addr` measurement (the product is the same either way) and
    // it is exactly the layout that hands the score kernel another request's slots, so the shape is refused
    // rather than measured.
    let op = ktir_superdsc::ir::bridge::tiled_op_sdsc_op::gather_copy_opspec(
        "Tensor0",
        "Tensor2",
        16,
        ktir_superdsc::sdsc_abstract::POOL_STICK,
        ktir_superdsc::superdsc_opspec::GatherIndex {
            name: "BlockTable".to_string(),
            entry_dim: axis,
            page: PageExtent::single_position(),
            per_position: None,
            first_entry: ktir_superdsc::superdsc_opspec::EntryBase::ZERO,
        },
        ktir_superdsc::superdsc_opspec::DestEntry::ZERO,
    )
    .expect("the gather-copy op builds");
    let value_idx = op.indirect.expect("a declared gather").value;
    let folds = SdscFoldSet::new(op.iter.cores_used());
    let e = superdsc::emit_sdsc("Gather_0", &op, &folds, None).expect("emits");
    let j = serde_json::to_value(&e).unwrap();
    let (dsc, node) = dsc_and_node(&j, "Gather_0", value_idx);

    let with_pin = skip_addr_elems(dsc, node).expect("derivable");

    // The same product WITHOUT the pinned dim's contribution, computed independently.
    let n = dsc["N_"].as_object().unwrap();
    let layout = node["layoutDimOrder_"].as_array().unwrap();
    let others: i64 = layout
        .iter()
        .filter(|d| d.as_str() != Some(entry_dim))
        .map(|d| {
            let dim = d.as_str().unwrap();
            let extent = n[&format!("{dim}_")].as_i64().unwrap();
            let split = dsc["numWkSlicesPerDim_"]
                .get(dim)
                .and_then(|v| v.as_i64())
                .unwrap_or(1)
                .max(1);
            extent / split
        })
        .product();

    assert_eq!(
        with_pin, others,
        "the pinned dim contributes 1, so skip_addr is exactly one entry: the product of the rest"
    );
}

/// ⛔⛔⛔ THE GAP, QUANTIFIED LOCALLY: what a per-kv-head KV operand yields versus what one layer-page
/// needs.
///
/// This is the number the next step has to close, and having it here means the fix is verified against
/// arithmetic rather than against a card run that would not have detected the error anyway (a short
/// `skip_addr` bakes clean and generates fluent text from the wrong keys).
///
/// Recorded as a RATIO, not a literal, so the pool's geometry can change without this test asserting a
/// stale constant — and so the message names the factor a reader has to account for.
#[test]
fn a_per_kv_head_operand_is_short_of_a_layer_page_by_exactly_nkvh() {
    // What one score op's KV operand covers today: one kv head's `[hd, page_slots]`.
    let per_head_elems = HD * PAGE_SLOTS;
    assert_eq!(
        LAYER_PAGE_ELEMS / per_head_elems,
        NKVH,
        "a per-kv-head operand is nkvh short of a layer-page"
    );

    // ⭐ THE REQUIREMENT, stated so the wiring can be checked against it: the value operand's
    // unpinned per-core extents must multiply to a whole layer-page. Anything less and the index
    // walks a fraction of a page per step.
    assert_eq!(
        NKVH * HD * PAGE_SLOTS,
        LAYER_PAGE_ELEMS,
        "the operand must span (nkvh, hd, page_slots) for skip_addr to equal one layer-page"
    );
    // And the failure is silent, so say so where someone changing the declaration will read it.
    assert!(
        LAYER_PAGE_ELEMS % per_head_elems == 0,
        "if the span is not a whole multiple the error is not even uniform across pages"
    );
}

/// ⭐⭐⭐⭐⭐ THE hd=128 DECLARATION, CHECKED AGAINST dxp'S OWN DERIVATION **BEFORE** ANYONE BUILDS IT —
/// `skip_addr` is the pinned sub-row count times one stick, so the two pins the two planes need give
/// exactly the two window strides the pool has.
///
/// ⛔ THIS IS THE ONE NUMBER A CARD RUN WOULD NOT HAVE CAUGHT. A `skip_addr` that is half a V window
/// bakes clean and gathers from inside the previous block; the design that fixes it is arithmetic, so it
/// is checkable here, and checking it here is what stops the hd=128 build from starting on prose.
/// `zz_the_two_planes_block_numbering_diverges_above_one_stick.rs` derives the two strides from
/// [`ktir_superdsc::sdsc_abstract::PagedKvPool::addr`]; this asserts the EMITTED DESCRIPTOR produces
/// them, through the same reimplementation of `getBufferCapacityForNodePerDim` the shipped case uses.
///
/// | plane | pin (`mb` sub-rows) | derived `skip_addr` | the pool's own stride |
/// |---|---|---|---|
/// | Kᵗ | `hd` = 128 | 8192 | one stick block (`hd * POOL_STICK`) |
/// | V  | `POOL_STICK` = 64 | 4096 | one slab window (`POOL_STICK²`) |
///
/// ⛔ AND THE PIN MUST BE **EVEN**, which both are. dxp does
/// `DT_CHECK(skip_addr_sticks % 2 == 0); skip_addr_sticks /= 2` for `coreArch >= SEN1P5_ISA`
/// (`GatherIndexConversion.cpp::computeGatherMetadata`), and `skip_addr_sticks` is the pin itself
/// (`mb` is unsticked so it contributes its clamped extent; `out` is one stick so it contributes 1). An
/// odd pin is a DT_CHECK abort, not a refusal — stated here because it is the one constraint on the pin
/// that is invisible from this side.
#[test]
fn the_two_pins_hd_128_needs_derive_the_two_window_strides_the_pool_has() {
    use ktir_superdsc::sdsc_abstract::{POOL_STICK, PagedKvPool};
    use ktir_superdsc::superdsc_opspec::{EntryBase, GatherIndex};

    const HD_8B: i64 = 128;
    let pool = PagedKvPool::new(NKVH as usize, HD_8B as usize);
    // The two strides, from the pool — not spelled as 8192/4096.
    let kt_window = pool.stick_block_elems() as i64;
    let v_window = (POOL_STICK * POOL_STICK) as i64;
    assert_eq!(kt_window, HD_8B * POOL_STICK as i64);
    assert_eq!(
        kt_window / v_window,
        HD_8B / POOL_STICK as i64,
        "at hd=128 a Kᵗ window is `nslab` V windows, which is the whole reason the pins differ"
    );

    for (plane, pin, want) in [("Kt", HD_8B as u32, kt_window), ("V", POOL_STICK, v_window)] {
        assert_eq!(
            pin % 2,
            0,
            "{plane}: dxp DT_CHECKs an even skip_addr in sticks"
        );
        // ONE INDEX STICK of entries, the only run a copy op may serve — so `mb` is `32 * pin`
        // sub-rows and the op is the real shape, not a reduced one.
        let mb = ktir_superdsc::sdsc_abstract::CopyDims::ENTRIES_PER_OP * pin;
        let op = ktir_superdsc::ir::bridge::tiled_op_sdsc_op::gather_copy_opspec(
            "Tensor0",
            "Tensor2",
            mb,
            POOL_STICK,
            GatherIndex::of_scratch_rows(
                "BlockTable".to_string(),
                PageExtent::of_positions(pin),
                EntryBase::ZERO,
            ),
            ktir_superdsc::superdsc_opspec::DestEntry::ZERO,
        )
        .expect("the gather-copy op builds at both pins");
        let value_idx = op.indirect.expect("a declared gather").value;
        let folds = SdscFoldSet::new(op.iter.cores_used());
        let e = superdsc::emit_sdsc("Gather_0", &op, &folds, None).expect("emits");
        let j = serde_json::to_value(&e).unwrap();
        let (dsc, node) = dsc_and_node(&j, "Gather_0", value_idx);
        let got = skip_addr_elems(dsc, node).expect("derivable");
        assert_eq!(
            got,
            want,
            "{plane} at hd=128 pinned {pin} sub-rows: dxp derives skip_addr = {got} elements where \
             the pool's own window stride is {want}. A step of {got} lands {:.2} of a window off — a \
             clean bake reading from inside another block.",
            got as f64 / want as f64,
        );
    }
}

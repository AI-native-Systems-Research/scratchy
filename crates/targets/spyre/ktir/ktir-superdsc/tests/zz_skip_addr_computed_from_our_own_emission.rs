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
    let (e, _) = superdsc::emit_sdsc("MatMul_0", &op, &folds, None).expect("emits");
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

/// ⭐⭐⭐⭐⭐ `skip_addr` OVER THE **SHIPPED** COPY — `batched_gathered::copy_opspecs`, the copy legs the
/// batched gathered dot emits for every decode attention's gathered K/V (and the MoE gather-dot), at
/// the pool's own entry unit.
///
/// Two facts, from the one emission, for EVERY leg:
/// 1. `skip_addr` is ONE SLAB BLOCK (`POOL_STICK²` = 4096 elements) at EVERY head dim — the unit
///    `GatherEntry::of_slab_block` counts the host's entries in and `gather_entries_per_page` divides a
///    page by, so the host's entry and the card's address are one derivation rather than two. A copy
///    whose derived stride is anything else lands every index step a fraction of a block off — a clean
///    bake producing fluent text from another block, with every counter green.
/// 2. The leg is planned onto MORE THAN ONE CORE, and one core's double-buffered chunk fits LX. A
///    pinned axis holding one entry admits no core split, and dxp then measures that one core's
///    double-buffered chunk against LX — the bake refusal 'The initial chunk parameters must fit in LX
///    for SuperDSC' (`L3DlOpsScheduler.cpp:1534`, gated on `isDoubleBuffering`).
///
/// ⭐ SWEPT OVER `nkvh` × `hd` AND THE BATCH, because the per-head-dim stick-block unit this file used
/// to pin (`hd · POOL_STICK`) coincides with the slab block at exactly one point — hd 64 — which is how
/// a hd-128 stride error stays invisible to a single-geometry measurement. Every leg of a multi-leg
/// copy (an entry count past one index stick) must derive the same stride.
#[test]
fn the_shipped_gather_copys_skip_addr_is_one_slab_block() {
    use ktir_superdsc::emit::batched_gathered::{BatchedGathered, copy_opspecs};
    use ktir_superdsc::sdsc_abstract::{POOL_STICK, PagedKvPool};
    for (nkvh, hd) in [
        (8u32, 64u32),
        (8, 128),
        (8, 256),
        (4, 128),
        (16, 128),
        (32, 64),
    ] {
        let pool = PagedKvPool::new(nkvh, hd);
        let block = pool.slab_block_elems() as u32;
        assert_eq!(
            block,
            POOL_STICK * POOL_STICK,
            "the slab block is POOL_STICK² at hd={hd}"
        );
        // A decode-attention gathered operand: one gathered row per (entry, batch row), each a
        // `[STK, STK]` slab block — the batched frame `[batch, STK, STK]` over `entries` entries.
        for entries in [32u32, 64, 96] {
            let bg = BatchedGathered::of(
                &[entries as i64, 4, POOL_STICK as i64],
                &[entries as i64, POOL_STICK as i64, POOL_STICK as i64],
                entries,
                block,
            )
            .expect("a slab-block gathered dot is a batched gathered geometry");
            let legs = copy_opspecs(
                &bg,
                "Tensor0",
                "BlockTable",
                "Tensor2",
                ktir_superdsc::superdsc_opspec::EntryBase::ZERO,
            )
            .expect("the shipped copy builds");
            assert_eq!(
                legs.len() as u32,
                entries.div_ceil(ktir_superdsc::sdsc_abstract::CopyDims::ENTRIES_PER_OP),
                "one leg per index stick"
            );
            for (leg, op) in legs.iter().enumerate() {
                let name = format!("Gather_{leg}");
                let value_idx = op.indirect.expect("a declared gather").value;
                let cores = op.iter.cores_used().get();
                let folds = SdscFoldSet::new(op.iter.cores_used());
                let (e, _) = superdsc::emit_sdsc(&name, op, &folds, None).expect("emits");
                let j = serde_json::to_value(&e).unwrap();
                let (dsc, node) = dsc_and_node(&j, &name, value_idx);
                let got = skip_addr_elems(dsc, node).expect("derivable");
                assert_eq!(
                    got,
                    i64::from(block),
                    "nkvh={nkvh} hd={hd} entries={entries} leg={leg}: the shipped copy's skip_addr must \
                     be ONE slab block ({block} elems) — the unit the host's entries are counted in. \
                     Got {got}, i.e. {}x off, which lands every index step inside another block.",
                    got as f64 / f64::from(block),
                );
                assert!(
                    cores > 1,
                    "nkvh={nkvh} hd={hd} entries={entries} leg={leg}: the copy is planned onto {cores} \
                     core(s); a pinned axis holding one entry admits no core split, and dxp then \
                     refuses the one core's double-buffered chunk at bake"
                );
                let per_core_bytes =
                    (op.iter.extent("mb") as u64 / cores as u64) * POOL_STICK as u64 * 2 * 4;
                assert!(
                    per_core_bytes <= ktir_superdsc::superdsc_opspec::USABLE_LX_BYTES,
                    "nkvh={nkvh} hd={hd} entries={entries} leg={leg}: one core's double-buffered chunk \
                     is {per_core_bytes} B over {cores} cores, against LX's {} B",
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
        ktir_superdsc::superdsc_opspec::GatherIndex {
            name: "BlockTable".to_string(),
            entry_dim: axis,
            page: PageExtent::single_position(),
            per_position: None,
            first_entry: ktir_superdsc::superdsc_opspec::EntryBase::ZERO,
        },
        ktir_superdsc::sdsc_abstract::GatherCopySpec {
            rows: 16,
            cols: ktir_superdsc::sdsc_abstract::POOL_STICK,
            df: ktir_superdsc::superdsc_opspec::Df::Fp16,
            run: ktir_superdsc::superdsc_opspec::GatherRun::CONTIGUOUS,
            // The destination's own law: the copy's `mb × out` = 1024-element window at entry 0, exact.
            dest: ktir_superdsc::sdsc_abstract::GatherDest::Scratch {
                rows: 16,
                cols: ktir_superdsc::sdsc_abstract::POOL_STICK,
            },
        },
    )
    .expect("the gather-copy op builds");
    let value_idx = op.indirect.expect("a declared gather").value;
    let folds = SdscFoldSet::new(op.iter.cores_used());
    let (e, _) = superdsc::emit_sdsc("Gather_0", &op, &folds, None).expect("emits");
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
    let pool = PagedKvPool::new(NKVH as u32, HD_8B as u32);
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
            GatherIndex::of_scratch_rows(
                "BlockTable".to_string(),
                PageExtent::of_positions(pin),
                EntryBase::ZERO,
            ),
            ktir_superdsc::sdsc_abstract::GatherCopySpec {
                rows: mb,
                cols: POOL_STICK,
                df: ktir_superdsc::superdsc_opspec::Df::Fp16,
                run: ktir_superdsc::superdsc_opspec::GatherRun::CONTIGUOUS,
                // The destination's own law in the entry frame: one index stick of entries × the pin's
                // stick — exactly the copy's `mb × out` window, at either plane's pin.
                dest: ktir_superdsc::sdsc_abstract::GatherDest::Scratch {
                    rows: ktir_superdsc::sdsc_abstract::CopyDims::ENTRIES_PER_OP,
                    cols: pin * POOL_STICK,
                },
            },
        )
        .expect("the gather-copy op builds at both pins");
        let value_idx = op.indirect.expect("a declared gather").value;
        let folds = SdscFoldSet::new(op.iter.cores_used());
        let (e, _) = superdsc::emit_sdsc("Gather_0", &op, &folds, None).expect("emits");
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

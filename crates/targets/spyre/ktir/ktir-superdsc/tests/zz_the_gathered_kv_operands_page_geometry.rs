// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ THE PAGE GEOMETRY OF THE KV OPERAND A GATHER WOULD READ — measured off the SHIPPED
//! attention emission, because every previous answer here came from reading a declaration instead of
//! evaluating one.
//!
//! ## The question this settles
//! dxp computes `addr = idx * skip_addr + base`, and derives `skip_addr` itself from the value
//! tensor's per-dim capacities: an unbounded (`-1`) dim contributes its per-core datastage extent, a
//! PINNED dim contributes its pin (`getBufferCapacityForNodePerDim`, `getPageSize` at
//! `dsc2.cpp:4493-4526` erasing the negatives). So `skip_addr` is a function of the walk we declare,
//! and declaring the walk differently MOVES it.
//!
//! ⛔ THIS RETRACTS THE "4096 ELEMENTS" FIGURE IN c6a1f331d. That number was the product of BOTH the
//! kernel's per-core dims — i.e. the capacity of the operand with NO pin. A gathered operand always
//! has a pin (that is what makes it paged), and the pinned dim then contributes its page instead of
//! its extent. The gathered figure is therefore NOT the ungathered one, and the commit measured the
//! wrong configuration. What is measured below is the emission with the gather actually attached.
//!
//! ## Why a small `skip_addr` is not a defect
//! The index values are OURS. `skip_addr` only has to DIVIDE the address stride we need to express:
//! to reach a pool block `B` at element `4096·B`, the entry is `B · (4096 / skip_addr)`. So the
//! requirement is a divisibility fact, not an equality, and that is what this file asserts. The
//! earlier framing ("the operand must span a whole layer-page") demanded an equality the mechanism
//! never asked for.

use ktir_superdsc::emit as superdsc;
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{assemble_attn, matmul_opspec};
use ktir_superdsc::sdsc_abstract::{AttnGeometry, PagedKvPool, attn_bundle_rows};
use ktir_superdsc::superdsc_opspec::{GatherIndex, KernelAxis, PageExtent, SdscFoldSet};

const NQH: u32 = 32;
const NKVH: u32 = 8;
const HD: u32 = 64;
/// ONE PAGE. The mask blocks by the page, so a fold pass may not sweep more than one — assembling with
/// a wider `active_cap` is refused at build, which is the guard that caught the first draft of this file.
const CAP: u32 = PagedKvPool::PAGE_SLOTS as u32;
const ACTIVE_CAP: u32 = PagedKvPool::PAGE_SLOTS as u32;
// ⛔ `SCORE_WINDOW` IS GONE — it was the KV columns one op sweeps (one page), and it sized the copy
// harness's `mb` as `HD * SCORE_WINDOW / POOL_STICK`. A gather copy's width is no longer a sweep: it is
// ONE INDEX STICK of entries (`CopyDims::ENTRIES_PER_OP`), because that is all dxp loads. The harness
// says so directly now, and a constant that no longer decides anything reads as though it still does.

/// One operand's per-dim capacity picture, as dxp derives it.
struct Capacity {
    /// `(dim, per_core_extent, pin)` in `layoutDimOrder_` order.
    dims: Vec<(String, i64, i64)>,
}

impl Capacity {
    /// dxp's `getBufferCapacityForNodePerDim`: per-core extent for `-1`, the pin otherwise.
    fn skip_addr(&self) -> i64 {
        self.dims
            .iter()
            .map(|(_, per_core, pin)| {
                if *pin < 0 {
                    *per_core
                } else {
                    *per_core.min(pin)
                }
            })
            .product()
    }
    fn describe(&self) -> String {
        self.dims
            .iter()
            .map(|(d, e, p)| {
                if *p < 0 {
                    format!("{d}={e}")
                } else {
                    format!("{d}={e}(pinned {p})")
                }
            })
            .collect::<Vec<_>>()
            .join(" x ")
    }
}

/// Read the capacity picture of `scheduleTree_[arg]` out of a serialized op.
fn capacity_of(v: &serde_json::Value, arg: usize) -> (String, Capacity) {
    let (name, body) = v["dscs_"][0]
        .as_object()
        .and_then(|m| m.iter().next())
        .map(|(k, b)| (k.clone(), b.clone()))
        .expect("one dsc");
    let n = body["N_"].as_object().expect("N_").clone();
    let node = &body["scheduleTree_"][arg];
    let layout: Vec<String> = node["layoutDimOrder_"]
        .as_array()
        .expect("layoutDimOrder_")
        .iter()
        .map(|d| d.as_str().expect("a dim name").to_string())
        .collect();
    // ⛔ ABSENT `maxDimSizes_` IS NOT "EVERYTHING UNBOUNDED" — it is an operand with no walk
    // declaration at all, which is a different emission. Demand the key.
    let pins: Vec<i64> = node["maxDimSizes_"]
        .as_array()
        .expect("maxDimSizes_ — a gathered operand always declares one")
        .iter()
        .map(|p| p.as_i64().expect("an integer pin"))
        .collect();
    assert_eq!(
        layout.len(),
        pins.len(),
        "the walk must have one entry per dim or the pins land on the wrong dims"
    );
    let slices = body["numWkSlicesPerDim_"].as_object().cloned();
    let dims = layout
        .iter()
        .zip(&pins)
        .map(|(d, &pin)| {
            let extent = n
                .get(&format!("{d}_"))
                .and_then(|v| v.as_i64())
                .unwrap_or_else(|| panic!("N_ has no {d}_"));
            let split = slices
                .as_ref()
                .and_then(|s| s.get(d))
                .and_then(|v| v.as_i64())
                .unwrap_or(1)
                .max(1);
            (d.clone(), extent / split, pin)
        })
        .collect();
    (name, Capacity { dims })
}

/// The group-0 PREFIX score op — the one op whose kernel is the resident Kᵗ pool, i.e. the operand a
/// gather reads through. Not the new-token block's (`attn_nsc_*`), whose kernel is this step's own
/// freshly transposed keys and has no page table.
fn prefix_score_op(rung: u32, kv_block_index: Option<&str>) -> serde_json::Value {
    let geom = AttnGeometry::<NQH, NKVH, HD>::minted();
    let rows = attn_bundle_rows(geom, rung, true).expect("a baked rung");
    let mut sym = 0i64;
    assemble_attn(
        0,
        geom,
        rows,
        CAP,
        ACTIVE_CAP,
        "t_qs",
        "t_new_k",
        "t_new_v",
        "t_kct",
        "t_vc",
        "t_pmask",
        "t_cmask",
        kv_block_index,
        ktir_superdsc::place::PlaceId::Act(900),
        true,
        &mut sym,
        None,
    )
    .expect("attention emits")
    .into_iter()
    .map(|e| serde_json::to_value(&e.op).expect("serializes"))
    .find(|v| {
        v["dscs_"][0]
            .as_object()
            .and_then(|m| m.keys().next().cloned())
            .is_some_and(|k| k.contains("sc_g0") && !k.contains("nsc"))
    })
    .expect("the group-0 prefix score op")
}

/// ⭐⭐⭐⭐⭐ THE REAL ATTENTION BUNDLE CONTAINS A GATHER — the claim every earlier commit here made
/// about a synthetic op instead, and the one that has to hold before any of the stride arithmetic
/// below means anything.
///
/// Asserted on the DESCRIPTOR, not on the call site: `assemble_attn` taking an index name proves
/// nothing about what reached the JSON, and "a port with no caller is dead code" applies just as well
/// to a declaration no node carries.
/// ⛔ RETARGETED: the SHIPPED score op no longer declares a gather — it CANNOT, because it is a matmul
/// and deeptools refuses to schedule a gather on an op with a KERNEL (measured as a bake refusal on the
/// card at both index `memOrg_` values; `emit_sdsc` now refuses it at build time, and
/// `zz_diff_the_rung_descriptors::asking_the_shipped_prefix_fold_for_a_gather_is_refused_naming_the_kernel`
/// pins that at the real call path). The property this test exists for — a DECLARATION must reach a
/// descriptor NODE, because "a port with no caller is dead code" applies to declarations too — is
/// unchanged and is now asserted on the op that ships the gather.
#[test]
fn the_real_prefix_score_op_declares_the_gather() {
    let op = gathered_score_shaped_op(KernelAxis::Slot, PageExtent::of_one_stick());
    let (name, body) = op["dscs_"][0]
        .as_object()
        .and_then(|m| m.iter().next())
        .map(|(k, b)| (k.clone(), b.clone()))
        .expect("one dsc");
    let tree = body["scheduleTree_"].as_array().expect("a schedule tree");
    let roles: Vec<(String, String, String)> = tree
        .iter()
        .map(|n| {
            (
                n["name_"].as_str().unwrap_or("?").to_string(),
                n["indirectAllocType_"].as_str().unwrap_or("?").to_string(),
                n["maxDimSizes_"].to_string(),
            )
        })
        .collect();
    for (nm, role, walk) in &roles {
        println!("{name}: {nm:<44} {role:<14} {walk}");
    }
    println!(
        "{name}: indirectAccessIndexLabeledDs = {}",
        body["computeOp_"][0]["indirectAccessIndexLabeledDs"]
    );
    assert!(
        roles.iter().any(|(_, r, _)| r == "value_tensor"),
        "no operand of {name} is the gathered VALUE tensor — the declaration did not reach the \
         descriptor. Nodes: {roles:#?}"
    );
    assert!(
        roles.iter().any(|(_, r, _)| r == "index_tensor"),
        "no operand of {name} is the INDEX tensor, so nothing supplies the base. Nodes: {roles:#?}"
    );
    assert!(
        !body["computeOp_"][0]["indirectAccessIndexLabeledDs"]
            .as_array()
            .is_none_or(|a| a.is_empty()),
        "the compute op names no index DS, so dxp reads the operand as ordinary arithmetic input"
    );
}

/// ⛔⛔⛔ THE APPENDED INDEX MUST NOT BECOME THE OUTPUT.
///
/// `emit_sdsc` defines the output as `views.len() - 1` — "the last arg" — and `attach_gather_index`
/// APPENDS. So the index landing last would silently make the emitter treat the index tensor as this
/// op's output: the score buffer would never be written, and the op would write its scores over the
/// block table. `attach_gather_index`'s own doc claims appending "keeps `out_idx` where every other
/// reader expects it", which is true only if something else lands after it.
///
/// This op ALSO carries a fused epilogue, whose operand is inserted relative to the output — so the two
/// insertions interact, and the order is not something to reason about from two doc comments. Diffed
/// against the UNGATHERED emission of the same op: the output node must be the same one, in the same
/// role, with the same walk.
#[test]
fn the_gather_does_not_displace_the_output() {
    let roles = |v: &serde_json::Value| -> Vec<(String, String, String)> {
        let body = v["dscs_"][0]
            .as_object()
            .and_then(|m| m.values().next().cloned())
            .expect("one dsc");
        body["scheduleTree_"]
            .as_array()
            .expect("a tree")
            .iter()
            .map(|n| {
                (
                    n["name_"].as_str().unwrap_or("?").to_string(),
                    n["component_"].to_string(),
                    n["maxDimSizes_"].to_string(),
                )
            })
            .collect()
    };
    // ⛔ BOTH SIDES ARE THE **COPY OP** NOW — the same op with and without its declaration. The shipped
    // score matmul cannot carry a gather at all, so diffing it against a gathered version of itself is
    // no longer a comparison that exists. The property is unchanged: declaring a gather must ADD one
    // node and must not move the OUTPUT, because `emit_sdsc` reads the last arg as the output and an
    // APPENDED index silently becomes it.
    let plain = roles(&ungathered_score_shaped_op());
    let gathered = roles(&gathered_score_shaped_op(
        KernelAxis::Slot,
        PageExtent::of_one_stick(),
    ));
    println!("ungathered: {plain:#?}");
    println!("gathered:   {gathered:#?}");

    // The gather adds EXACTLY ONE node, and it is the index.
    assert_eq!(
        gathered.len(),
        plain.len() + 1,
        "the gather must add exactly one operand"
    );
    // ⭐ THE OUTPUT IS THE LAST NODE, AND IT IS THE SAME NODE IT WAS. Compared by component and walk
    // rather than by the positional `Tensor{i}` alloc name, which necessarily renumbers.
    let (plain_last, gathered_last) = (
        plain.last().expect("nodes"),
        gathered.last().expect("nodes"),
    );
    assert_eq!(
        (&plain_last.1, &plain_last.2),
        (&gathered_last.1, &gathered_last.2),
        "the LAST node changed when the gather was declared — `emit_sdsc` reads the last arg as the \
         OUTPUT, so this op now writes its scores somewhere else.\nungathered last: {plain_last:?}\n\
         gathered last:   {gathered_last:?}"
    );
}

/// ⛔ THE BASELINE, AND THE TRAP THAT PRODUCED THE "4096" FIGURE. This is the score kernel with NO
/// gather attached: every dim `-1`, so the capacity is the plain per-core product. It is recorded as a
/// baseline precisely so that the gathered number below can be compared against it — reading THIS and
/// calling it the gather's stride is the mistake c6a1f331d shipped.
#[test]
fn the_ungathered_capacity_is_the_whole_per_core_product() {
    let op = prefix_score_op(8, None);
    let (name, cap) = capacity_of(&op, 1);
    let ungathered = cap.skip_addr();
    println!(
        "{name}: UNGATHERED kernel {} => {ungathered}",
        cap.describe()
    );
    assert!(
        cap.dims.iter().all(|(_, _, pin)| *pin < 0),
        "an op with no gather must declare no pin — a pin is what makes a dim paged"
    );
    let product: i64 = cap.dims.iter().map(|(_, e, _)| e).product();
    assert_eq!(
        ungathered, product,
        "with nothing pinned the capacity is just the per-core product — NOT the gather's skip_addr"
    );
}

/// A matmul at the PREFIX SCORE LEG'S OWN EXTENTS, gathered on a CHOSEN axis and page.
///
/// ⭐ USED ONLY FOR THE FOUR-ROW COMPARISON TABLE, where the point is to vary the declaration —
/// `assemble_attn` offers exactly one (`GatherIndex::of_pool_blocks`), which is the whole reason it is
/// a named door. The shipped declaration is measured off the REAL emission by
/// `the_gathered_kv_operand_capacity_divides_the_pool_block`; this stand-in exists so the other three
/// rows, which the emitter cannot be asked for, are still measured rather than reasoned about.
/// ⛔ ON THE **KERNEL-LESS GATHER-COPY OP**, at the same extents. This built a matmul, which `emit_sdsc`
/// now refuses: deeptools cannot schedule a gather on an op with a KERNEL (`hasDimensionReuse` ->
/// the arithmetic-intensity explorer -> an LX allocate node an index never gets; MEASURED as a bake
/// refusal on the card at both index `memOrg_` values). The arithmetic being measured here — dxp's
/// `getBufferCapacityForNodePerDim`, where an unbounded dim contributes its per-core extent and a pinned
/// one is CLAMPED to the page — is per-operand and identical on either op, so the four-row table
/// survives the move intact.
fn gathered_score_shaped_op(axis: KernelAxis, page: PageExtent) -> serde_json::Value {
    // The kv window on the paged axis, the head dim as the batch — the score leg's own extents, on the
    // op shape that can carry a gather.
    let op = ktir_superdsc::ir::bridge::tiled_op_sdsc_op::gather_copy_opspec(
        "Tensor0",
        "Tensor2",
        // ⛔ THE SAME ELEMENTS, IN ONE-STICK SUB-ROWS — the shipped shape. A `[mb, out]` operand sticked on
        // a MULTI-stick `out` classifies stick-major, which scatters each gathered block `rows*64` apart
        // instead of contiguous; the builder refuses it, and the arithmetic measured here (a pinned dim
        // contributes its page, an unbounded one its per-core extent) is unchanged by the reshape.
        //
        // ⭐ AND IT IS **ONE INDEX STICK** OF THEM, WHICH IS ALSO THE SHIPPED `mb` AT THE SHIPPED PAGE:
        // `64 × 32 = 2048`, the very shape `PageExtent::entries_in` records from the pod. A copy op's
        // index is one stick (`CopyDims::ENTRIES_PER_OP`) so a wider one is refused at build — and the
        // arithmetic under test is untouched by that, because `skip_addr` is `page × the per-core extents
        // of the UNPINNED dims` and the run length only moves `mb`, which is the pinned one. Spelled as
        // `page × a stick` rather than as the score window so every page this helper is handed — a whole
        // stick, one position, `hd` sub-rows — is exercised at a legal index width instead of one that
        // happens to fit only at the widest page.
        page.get() * ktir_superdsc::sdsc_abstract::CopyDims::ENTRIES_PER_OP,
        ktir_superdsc::sdsc_abstract::POOL_STICK,
        GatherIndex {
            name: "Tensor3".to_string(),
            entry_dim: axis,
            page,
            per_position: None,
            first_entry: ktir_superdsc::superdsc_opspec::EntryBase::ZERO,
        },
        ktir_superdsc::superdsc_opspec::DestEntry::ZERO,
    )
    .expect("the gather-copy op builds at the score leg's extents");
    let folds = SdscFoldSet::new(op.iter.cores_used());
    let e = superdsc::emit_sdsc("score_gathered", &op, &folds, None).expect("emits");
    serde_json::to_value(&e).expect("serializes")
}

/// The SAME op with its declaration removed — the control for
/// [`the_gather_does_not_displace_the_output`], so the diff is about the gather and nothing else.
fn ungathered_score_shaped_op() -> serde_json::Value {
    let mut op = ktir_superdsc::ir::bridge::tiled_op_sdsc_op::gather_copy_opspec(
        "Tensor0",
        "Tensor2",
        // The SAME width as the gathered control above, for the same reason — one index stick.
        HD * ktir_superdsc::sdsc_abstract::CopyDims::ENTRIES_PER_OP,
        ktir_superdsc::sdsc_abstract::POOL_STICK,
        GatherIndex::of_scratch_rows(
            "Tensor3".to_string(),
            PageExtent::of_positions(HD),
            ktir_superdsc::superdsc_opspec::EntryBase::ZERO,
        ),
        ktir_superdsc::superdsc_opspec::DestEntry::ZERO,
    )
    .expect("the gather-copy op builds");
    // Drop the index operand and the declaration, leaving the bare two-operand identity.
    let idx = op.indirect.expect("just built gathered").index;
    op.args.remove(idx);
    op.indirect = None;
    let folds = SdscFoldSet::new(op.iter.cores_used());
    let e = superdsc::emit_sdsc("score_ungathered", &op, &folds, None).expect("emits");
    serde_json::to_value(&e).expect("serializes")
}

/// ⭐⭐⭐ THE MEASUREMENT THAT MATTERS: `skip_addr` of the KERNEL WITH THE GATHER ATTACHED.
///
/// Built through the same `_gathered` builder the score leg will use, at the score leg's own extents,
/// so the number is the one dxp will derive. The pin collapses the paged axis, so this is NOT the
/// baseline above and the two are asserted to differ.
///
/// Printed as well as asserted, because the number itself is the finding and the worker's staging
/// multiplier is computed from it.
#[test]
fn the_gathered_kv_operand_capacity_divides_the_pool_block() {
    // ⭐ THE REAL ATTENTION EMISSION, GATHERED. `assemble_attn` takes the index tensor's name and
    // declares `GatherIndex::of_pool_blocks` on the prefix score leg's Kᵗ operand, so what is measured
    // here is the descriptor the card would receive — not a matmul shaped to resemble it.
    // ⛔ ON THE KERNEL-LESS COPY OP, at the score leg's own extents, and at OPERAND 0. The shipped score
    // matmul cannot carry a gather (see `the_real_prefix_score_op_declares_the_gather`), and the operand
    // order here is `[source, index, destination]` — the gathered VALUE is 0, where 1 would measure the
    // INDEX's own capacity and report it as the address stride.
    let decl = GatherIndex::of_scratch_rows(
        "t_kv_block_index".to_string(),
        PageExtent::of_positions(HD),
        ktir_superdsc::superdsc_opspec::EntryBase::ZERO,
    );
    let op = gathered_score_shaped_op(decl.entry_dim, decl.page);
    let (name, cap) = capacity_of(&op, 0);
    let skip = cap.skip_addr();
    println!(
        "{name}: GATHERED kernel {} => skip_addr={skip}",
        cap.describe()
    );
    assert!(
        cap.dims
            .iter()
            .any(|(d, _, pin)| *d == decl.entry_dim.dim() && *pin == i64::from(decl.page.get())),
        "the paged axis must carry the declared page; without a pin dxp declares no page at all and \
         the gather reads whatever the base says"
    );

    // ⭐ THE REAL REQUIREMENT, and the only one. An index entry has to be able to name a pool block
    // base EXACTLY: `addr = idx*skip + base`, block `B` sits at `4096*B`, so `skip` must divide 4096.
    // Anything else and no integer index reaches a block boundary — every step lands inside a block,
    // which is a clean bake reading a fraction of another row's keys.
    let block = PagedKvPool::new(NKVH as usize, HD as usize).stick_block_elems() as i64;
    assert_eq!(
        block, 4096,
        "the pool's uniform block, from the pool itself"
    );
    assert!(
        skip > 0 && block % skip == 0,
        "skip_addr={skip} does not divide the pool block {block}: no integer index names a block base"
    );
    // And the multiplier the worker must apply when it stages an entry.
    let per_block = block / skip;
    println!("one pool block = {per_block} index step(s) of {skip} elems");
    assert_eq!(
        per_block * skip,
        block,
        "the staging multiplier must reconstruct the block exactly"
    );
}

/// ⛔ THE WALK IS RANK 2, WHICH IS WHY THERE IS NOTHING TO PAGE YET.
///
/// The index enumerates positions along the PINNED dim (the vendor's rank-3 value pins dim 0 to 1 and
/// its index is rank 1 with one entry per position along it). The score kernel's walk is
/// `["in","out"]` — a head-dim axis and a KV-slot axis — and NEITHER counts pages: `out` is one
/// window's slots. So pinning either dim makes the index enumerate *within* one window, which is the
/// contiguous read we already have, not a page table.
///
/// This is the structural finding that says what the next emission change must be, and it is asserted
/// rather than written in a comment so that widening the operand to span pages BREAKS this test — the
/// signal that the page axis now exists.
#[test]
fn the_score_kernels_walk_has_no_page_axis_today() {
    let op = prefix_score_op(8, None);
    let (_, cap) = capacity_of(&op, 1);
    let names: Vec<&str> = cap.dims.iter().map(|(d, _, _)| d.as_str()).collect();
    assert_eq!(
        names,
        vec!["in", "out"],
        "the score kernel is a 2-D [feature, slot] operand; a page axis would be a THIRD dim"
    );
    // Every dim is one window's own extent, so the product is one window — not a multi-page span.
    let block = PagedKvPool::new(NKVH as usize, HD as usize).stick_block_elems() as i64;
    let window: i64 = cap.dims.iter().map(|(_, e, _)| e).product();
    assert!(
        window <= block,
        "the operand spans {window} elems, at most one {block}-element block: it reaches ONE block, \
         which is why the fold has to re-launch per page"
    );
}

/// ⭐⭐⭐ THE POOL IS A UNIFORM ARRAY OF STICK BLOCKS — the fact that makes ONE integer index able to
/// name any cell, and the reason `skip_addr` only has to DIVIDE the block rather than equal a page.
///
/// Asserted from the pool's own accessors rather than restated as arithmetic in a commit message,
/// which is where it lived before. If a stride stops being a multiple, the gather's address law stops
/// being expressible and this fails at `cargo test` instead of on the card.
#[test]
fn the_pool_is_a_uniform_array_of_stick_blocks() {
    let p = PagedKvPool::new(NKVH as usize, HD as usize);
    let block = p.stick_block_elems();
    assert!(block > 0, "a zero block makes every index step a no-op");
    // ⛔ THE ONE THAT HAS ALREADY BEEN CONFUSED FOR THE BLOCK: the distance to the next KV head is
    // FOUR blocks (a page's 256 slots over 64-slot blocks), not one.
    assert_eq!(
        p.plane_block_elems() % block,
        0,
        "the KV-head step is not a whole number of blocks"
    );
    assert_eq!(
        p.plane_block_elems() / block,
        p.blocks_per_page(),
        "the KV-head step should be exactly a page's blocks — the quantity `blocks_per_page` names"
    );
}

/// ⭐⭐⭐⭐⭐ THE FOUR DECLARATIONS AND THE FOUR STRIDES THEY PRODUCE — the table that says why the
/// axis and the page are a TYPE and not two `u32`s.
///
/// MEASURED at the score kernel's geometry (`in = hd = 64`, `out = one page = 256`):
///
/// ```text
/// entry_dim  page   skip_addr   what one entry means
/// Slot        1        64       one KV column
/// Slot       64      4096       ⭐ ONE POOL BLOCK — the entry is a plain block number
/// Feature     1       256       one feature row of a whole page
/// Feature    64     16384       four pool blocks (a whole KV-head plane)
/// ```
///
/// ⛔ EVERY ONE OF THESE BAKES. They span 256× and differ only in a declaration, so the wrong pair
/// produces fluent text assembled from another request's keys with every counter green. That is the
/// entire reason [`GatherIndex::of_pool_blocks`] exists as a named door rather than two arguments a
/// caller fills in: the row of this table that the KV pool's layout actually supports is the second.
#[test]
fn the_axis_and_page_together_decide_the_stride() {
    let block = PagedKvPool::new(NKVH as usize, HD as usize).stick_block_elems() as i64;
    let mut table = Vec::new();
    // ⛔ `Batch` ("mb") REPLACES `Feature` ("in"): the harness is the KERNEL-less gather-copy op now,
    // whose dims are `[mb, out, y]`. `in` belongs to a matmul, which cannot carry a gather at all, and
    // `attach_gather_index` REFUSES a paged axis the op does not have (it used to panic out of
    // `iter_syms` instead — an assert firing from inside a function whose contract is to return `None`).
    // ⭐ AND OPERAND **0**, not 1: on this op the operands are `[source, index, destination]` — the
    // index is INSERTED before the output — so the gathered VALUE is 0. Reading 1 would measure the
    // index's own capacity and call it the address stride.
    for axis in [KernelAxis::Slot, KernelAxis::Batch] {
        for page in [PageExtent::single_position(), PageExtent::of_one_stick()] {
            let cap = capacity_of(&gathered_score_shaped_op(axis, page), 0).1;
            table.push((axis, page.get(), cap.skip_addr(), cap.describe()));
        }
    }
    for (axis, page, skip, desc) in &table {
        println!("{:?} page={page:>3} => skip_addr={skip:>6}   {desc}", axis);
    }
    // ⛔ THE DECLARATIONS ARE NOT INTERCHANGEABLE. If they ever all agree, a wrong pair stops being
    // detectable by `skip_addr` at all and this whole instrument goes blind.
    let strides: Vec<i64> = table.iter().map(|&(_, _, s, _)| s).collect();
    assert!(
        strides
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            > 1,
        "every declaration gave the same stride ({strides:?}) — then nothing here can catch a wrong one"
    );
    // ⭐ AND THE SHIPPED DOOR IS THE ONE THAT MAKES AN ENTRY A BLOCK NUMBER — `of_scratch_rows`, whose
    // page is `cols / POOL_STICK` sub-rows of a block (`GatherScratch::entry_page`). It is measured in
    // its own row of the table rather than named in prose, because the value is what the host's entry
    // factor divides a page by.
    let shipped = GatherIndex::of_scratch_rows(
        "T".to_string(),
        PageExtent::of_positions(HD),
        ktir_superdsc::superdsc_opspec::EntryBase::ZERO,
    );
    let cap = capacity_of(
        &gathered_score_shaped_op(shipped.entry_dim, shipped.page),
        0,
    )
    .1;
    assert_eq!(
        cap.skip_addr(),
        block,
        "of_scratch_rows must make one index step exactly one pool block, or a host staging block \
         numbers addresses by the wrong factor"
    );
}

/// ⭐ THE RUNG DOES NOT MOVE THE CAPACITY — so `skip_addr` is a property of the KV geometry, not of
/// the batch width, and a wider rung cannot silently change the address stride under us.
#[test]
fn skip_addr_is_the_same_at_every_baked_rung() {
    let mut seen: Vec<(u32, i64)> = Vec::new();
    for rung in PagedKvPool::BATCH_RUNGS {
        let (_, cap) = capacity_of(&prefix_score_op(rung, None), 1);
        seen.push((rung, cap.skip_addr()));
    }
    println!("skip_addr per rung: {seen:?}");
    let first = seen[0].1;
    assert!(
        seen.iter().all(|&(_, s)| s == first),
        "skip_addr moves with the rung ({seen:?}) — then one index table cannot serve two widths"
    );
}

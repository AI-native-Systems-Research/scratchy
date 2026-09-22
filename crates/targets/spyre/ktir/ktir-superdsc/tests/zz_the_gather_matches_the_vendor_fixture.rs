// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ THE HARDWARE GATHER, DIFFED FIELD-BY-FIELD AGAINST THE VENDOR FIXTURE THAT BAKES.
//!
//! `dxp/test/test_gather_1core/sdsc_1.json` is IBM's own indirect-access SDSC and it compiles on
//! their toolchain. This file asserts our emission carries the SAME fields with the SAME shapes,
//! because the alternative — inventing a plausible field set and reading dxp's exception — is what
//! four weeks of attempts did.
//!
//! ## What the vendor fixture actually contains (extracted, not remembered)
//! ```text
//! allocate-Tensor1_hbm   indirectAllocType_ "index_tensor"   indexTensorType_ "index"
//!                        relatedIndirectAccessAlloc_ "allocate-Tensor2_hbm"
//!                        maxDimSizes_ [-1]
//! allocate-Tensor2_hbm   indirectAllocType_ "value_tensor"   isStartAddrSymbolic_ 1
//!                        relatedIndirectAccessAlloc_ "allocate-Tensor1_hbm"
//!                        maxDimSizes_ [1, -1, -1]
//! computeOp_[0]          indirectAccessIndexLabeledDs ["Tensor1-idx1"]
//! ```
//! Three facts worth stating because each was got wrong at least once:
//! * the two nodes CROSS-LINK — each names the other, so neither alone is a gather;
//! * only the INDEX side carries `indexTensorType_`, and only the VALUE side is symbolic;
//! * `maxDimSizes_` on the value side MIXES a pin with `-1`s. `getPageSize` erases the negatives,
//!   so the pinned dim IS the paged dim and every other dim is unpaged and unchecked.
//!
//! ## Why a pin of exactly 1
//! `L3DlOpsScheduler:6655` requires `ss_(dim) % pageSize(dim) == 0`. With the pin at 1 that holds
//! for every extent, so the geometry check can never be the thing that rejects us — which matters,
//! because that check sits UPSTREAM of op mapping (`SchedulerStages.cpp:29-42`) and both stages
//! print the same `sbf-ddc:` prefix, so a failure there looks exactly like a rejected op.

use ktir_superdsc::emit as superdsc;
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{gather_copy_opspec, matmul_opspec};
use ktir_superdsc::superdsc_opspec::{
    GatherIndex, IndirectAccess, KernelAxis, PageExtent, SdscFoldSet,
};

/// The vendor fixture's own declaration: one entry per position along its value tensor's PAGED dim,
/// which is `mb` — `layoutDimOrder_ ["mb","out","x"]` with `maxDimSizes_ [1,-1,-1]`. Named once so
/// every assertion below reproduces the same fixture rather than each choosing a page.
///
/// ⛔ IT USED TO SAY `KernelAxis::Feature`, i.e. `"in"`, WHICH IS NOT A DIM THE VENDOR'S OP HAS. That
/// worked only because the harness was a matmul, which does have an `in` — so the "vendor declaration"
/// this file reproduced was the vendor's PIN VALUE on OUR axis. It is now the vendor's axis too.
const VENDOR_DECL: (KernelAxis, PageExtent) = (KernelAxis::Batch, PageExtent::single_position());

/// The vendor's declaration as the ONE value `attach_gather_index` takes — see its own note on why it
/// takes a whole [`GatherIndex`] rather than four loose fields. `EntryBase::ZERO` because the fixture's
/// index is the whole tensor: only the KV gather's one-stick runs start anywhere else.
fn vendor_index(name: &str) -> GatherIndex {
    GatherIndex {
        name: name.to_string(),
        entry_dim: VENDOR_DECL.0,
        page: VENDOR_DECL.1,
        per_position: None,
        first_entry: ktir_superdsc::superdsc_opspec::EntryBase::ZERO,
    }
}

/// ⛔⛔⛔ THE HARNESS IS THE **GATHER-COPY** OP, NOT A MATMUL — and that is not cosmetic.
///
/// `emit_sdsc` now REFUSES a gather on an op that has a KERNEL, because deeptools cannot schedule one:
/// `hasDimensionReuse` (`L3DlOpsScheduler.cpp:303`) turns on the arithmetic-intensity explorer, whose
/// `calculateFlopPerByte` (`:2334`/`:2337`) demands an LX allocate node for every HBM-pinned labeledDs
/// — and `allocAllMem` never gives one to an index. MEASURED as a bake refusal on the card at BOTH
/// possible index `memOrg_` values, so no declaration escapes it.
///
/// Which means every test in this file was, until now, asserting the descriptor of a shape the card
/// REFUSES: green, detailed, fixture-checked, and about nothing that could ever run. The vendor's own
/// op is `identity` with both data operands typed `OUTPUT`, and `gather_copy_opspec` is that op — so
/// this harness now reproduces the fixture more closely than the matmul ever did, not less.
fn plain_matmul() -> ktir_superdsc::superdsc_opspec::OpSpec {
    gather_copy_base()
}

/// The gather-copy op WITH its real index operand and declaration — `[value, index, output]`, the only
/// arrangement `attach_gather_index` will build (the index is INSERTED before the output so the output
/// stays last). Tests that need a declared gather use THIS rather than bolting an `IndirectAccess` onto
/// the two-operand form, where position 1 is the output.
fn gather_copy_declared() -> ktir_superdsc::superdsc_opspec::OpSpec {
    gather_copy_opspec(
        "Tensor0",
        "Tensor2",
        16,
        ktir_superdsc::sdsc_abstract::POOL_STICK,
        vendor_index("Tensor1"),
        ktir_superdsc::superdsc_opspec::DestEntry::ZERO,
    )
    .expect("the gather-copy op builds")
}

/// The gather-copy op WITHOUT a declaration — the base every test below attaches its own to.
fn gather_copy_base() -> ktir_superdsc::superdsc_opspec::OpSpec {
    // A gather attached and immediately discarded is how a "bare" op of the right SHAPE is obtained:
    // `gather_copy_opspec` is the only door to the KERNEL-less two-operand identity, and the tests that
    // want the un-gathered form (`no_declaration_emits_no_indirection_anywhere`) clear it below.
    // ⛔ `out` IS ONE STICK, WHICH THE BUILDER REQUIRES — it was 384. A `[mb, out]` operand sticked on a
    // MULTI-stick `out` classifies stick-major, which scatters each gathered block `rows*64` apart instead
    // of leaving it contiguous, and the score kernel that reads it then gets another request's slots from a
    // clean bake. The vendor's own value tensor is one stick wide too, so this brings the harness closer to
    // the fixture rather than further from it.
    let mut op = gather_copy_opspec(
        "Tensor0",
        "Tensor2",
        16,
        ktir_superdsc::sdsc_abstract::POOL_STICK,
        vendor_index("Tensor1"),
        ktir_superdsc::superdsc_opspec::DestEntry::ZERO,
    )
    .expect("the gather-copy op builds");
    // Drop the index operand and the declaration, leaving the bare two-operand identity.
    let idx = op.indirect.expect("just attached").index;
    op.args.remove(idx);
    op.indirect = None;
    op
}

/// The vendor fixture, as ground truth for the field NAMES and SHAPES.
const VENDOR: &str = "/Users/nickm/git/deeptools/dxp/test/test_gather_1core/sdsc_1.json";

fn node<'a>(j: &'a serde_json::Value, op: &str, arg: usize) -> &'a serde_json::Value {
    &j["dscs_"][0][op]["scheduleTree_"][arg]
}

/// ⭐ THE PAIR IS ATOMIC. `IndirectAccess::of` refuses a tensor indexed by itself, because the two
/// nodes cross-link and a self-link is a cycle dxp reads as one.
#[test]
fn an_indirect_access_cannot_name_one_operand_twice() {
    assert!(IndirectAccess::of(1, 2, VENDOR_DECL.0, VENDOR_DECL.1, None).is_some());
    assert!(
        IndirectAccess::of(1, 1, VENDOR_DECL.0, VENDOR_DECL.1, None).is_none(),
        "index == value cross-links a node to itself"
    );
}

/// ⭐ EVERY OP THAT DECLARES NO GATHER EMITS EXACTLY WHAT IT EMITTED BEFORE THIS EXISTED.
///
/// Stated as a whole-`maxDimSizes_`-and-three-fields check rather than argued. The claim "adding a
/// field nobody sets cannot change the output" is the same shape as an earlier claim that a reduce
/// handle was emission-neutral — which was FALSE, because the handle also set the output's rank. An
/// argument about an emitter is worth exactly one diff.
#[test]
fn no_declaration_emits_no_indirection_anywhere() {
    let op = plain_matmul();
    assert!(op.indirect.is_none(), "the default is no gather");
    let folds = SdscFoldSet::new(op.iter.cores_used());
    let e = superdsc::emit_sdsc("MatMul_0", &op, &folds, None).expect("emits");
    let j = serde_json::to_value(&e).unwrap();

    // Over the op's ACTUAL operands — the KERNEL-less copy op has two, a matmul had three, and a
    // hardcoded `0..3` read a `Null` node as a missing field rather than as an absent operand.
    for arg in 0..op.args.len() {
        let n = node(&j, "MatMul_0", arg);
        assert_eq!(
            n["indirectAllocType_"], "no_indirection",
            "operand {arg} must stay directly addressed"
        );
        assert!(
            n.get("relatedIndirectAccessAlloc_")
                .is_none_or(|v| v.is_null()),
            "operand {arg} must carry no cross-link"
        );
        assert!(
            n.get("indexTensorType_").is_none_or(|v| v.is_null()),
            "operand {arg} must carry no index type"
        );
        assert!(
            n.get("isStartAddrSymbolic_").is_none_or(|v| v.is_null()),
            "operand {arg} must keep a concrete start address"
        );
        // And no dim is pinned to 1 by this machinery — the walk is whatever the layout says.
        let m = n["maxDimSizes_"].as_array().expect("a walk");
        assert!(
            m.iter().all(|d| d.as_i64() == Some(-1)) || m.iter().all(|d| d.as_i64() != Some(1)),
            "operand {arg} walk {m:?} must not have acquired a page pin"
        );
    }
    // ⭐ AND THE FIELD IS OMITTED ENTIRELY, not emitted empty — which is STRONGER than the
    // neutrality this test was written to check. `skip_serializing_if` drops an empty vec, so a
    // gather-free op's JSON is byte-for-byte what it was before `indirect` existed, rather than
    // merely semantically equivalent. Asserted as "absent or empty" so either serialization is
    // accepted, but recorded here because the distinction is the whole difference between "no
    // shipped bundle changes" and "no shipped bundle changes MEANINGFULLY".
    let named = &j["dscs_"][0]["MatMul_0"]["computeOp_"][0]["indirectAccessIndexLabeledDs"];
    assert!(
        named.is_null() || named == &serde_json::json!([]),
        "an op with no gather names no index (got {named})"
    );
}

/// ⭐⭐⭐ A DECLARED GATHER EMITS THE VENDOR'S FIELD SET — asserted against the fixture's own values.
#[test]
fn a_declared_gather_matches_the_vendor_field_set() {
    // ⛔ THE **ATTACHED** OP, NOT A DECLARATION BOLTED ONTO THE BARE ONE. This was
    // `plain_matmul()` (the gather-copy with its index operand REMOVED) plus
    // `op.indirect = IndirectAccess::of(1, 0, …)` — and in a two-operand op, position 1 is the
    // OUTPUT, so the harness declared the destination to be its own index. `attach_gather_index`
    // refuses exactly that ("gathering INTO a destination is a scatter"), so the shape was
    // unreachable from the real builder; it survived only while nothing about the index side touched
    // an address. Now that the index is measured in ENTRIES and shares one base across cores
    // ([`WorkPlan::of_index_entries`]), declaring the output as the index trips the output-aliasing
    // guard — correctly. Every assertion below is unchanged; only the operand that plays the index is
    // now a real index (`Tensor1`, inserted BEFORE the output), which is what the node indices this
    // test reads (`0` = value, `1` = index) already assumed.
    let op = gather_copy_declared();
    let folds = SdscFoldSet::new(op.iter.cores_used());
    let e = superdsc::emit_sdsc("MatMul_0", &op, &folds, None).expect("a gather emits");
    let j = serde_json::to_value(&e).unwrap();

    let idx = node(&j, "MatMul_0", 1);
    let val = node(&j, "MatMul_0", 0);

    // ── the INDEX side ──
    assert_eq!(idx["indirectAllocType_"], "index_tensor");
    assert_eq!(idx["indexTensorType_"], "index");
    assert_eq!(
        idx["relatedIndirectAccessAlloc_"], "allocate-Tensor0_hbm",
        "the index node must name the VALUE node"
    );
    assert!(
        idx.get("isStartAddrSymbolic_").is_none_or(|v| v.is_null()),
        "only the value side is symbolic in the vendor fixture"
    );

    // ── the VALUE side ──
    assert_eq!(val["indirectAllocType_"], "value_tensor");
    assert_eq!(
        val["relatedIndirectAccessAlloc_"], "allocate-Tensor1_hbm",
        "the value node must name the INDEX node — the link is mutual"
    );
    // ⛔⛔⛔ `0`, NOT `1`, AND THE CARD SETTLED IT. This asserted `1` — "the gathered base is passed
    // in, not baked: that is what selects dxp's symbolic index-to-address path" — transcribed from
    // `test_gather_1core`, which is a unit test OF that path and pairs the flag with `data_ "-1"` plus a
    // `%base_addr` operand in its `bundle.mlir`. We emit a CONCRETE per-core address, so on the pod dbo
    // refused with `Symbol operand does not exist: 34360262656` — the byte address read as a symbol id.
    //
    // IBM's paged-attention fixture, which is the case this emitter reproduces, has
    // `isStartAddrSymbolic_ 0` with `data_ "128000"`. Two fixtures, one of them about a different
    // feature; the paged one governs. `the_shipped_prefix_fold_actually_emits_the_gather_nodes` and
    // `our_shapes_agree_with_the_vendor_fixtures_own_values` (below) now read the paged fixture for this
    // one field and the dxp fixture for the rest.
    assert_eq!(
        val["isStartAddrSymbolic_"], 0,
        "a gathered operand carrying a CONCRETE start address must not claim a symbolic one — dbo \
         resolves the value as a symbol id and refuses the bake"
    );
    assert!(
        val.get("indexTensorType_").is_none_or(|v| v.is_null()),
        "the value node carries no index type"
    );

    // ── the PAGED PIN: exactly one 1, the rest -1, same shape as the vendor's [1, -1, -1] ──
    let walk: Vec<i64> = val["maxDimSizes_"]
        .as_array()
        .expect("a walk")
        .iter()
        .map(|d| d.as_i64().expect("an integer extent"))
        .collect();
    assert_eq!(
        walk.iter().filter(|&&d| d == 1).count(),
        1,
        "exactly ONE dim may be paged; walk was {walk:?}"
    );
    assert!(
        walk.iter().all(|&d| d == 1 || d == -1),
        "every unpaged dim must be -1 so getPageSize erases it; walk was {walk:?}"
    );
    // ⭐ THE PIN MUST LAND ON THE DECLARED AXIS'S OWN POSITION IN THIS OPERAND'S LAYOUT — found by
    // name, exactly as the emitter finds it. Asserting `walk[0]` instead would pass for any operand
    // whose declared axis happens to be first, and this operand's is not: its layout is
    // `["mb","in"]` and the declaration names "in", so the pin belongs at index 1. That off-by-a-dim
    // is precisely the bug a positional `KernelAxis::position()` introduced.
    let layout = op.args[0].view().layout.to_vec();
    let want = layout
        .iter()
        .position(|&d| d == VENDOR_DECL.0.dim())
        .unwrap_or_else(|| panic!("the declared axis must be in the layout {layout:?}"));
    assert_eq!(
        walk[want],
        i64::from(VENDOR_DECL.1.get()),
        "the pin must land on '{}' (layout {layout:?}, walk {walk:?})",
        VENDOR_DECL.0.dim()
    );
    assert_eq!(
        walk.len(),
        op.args[0].view().layout.len(),
        "the walk's rank must equal the tensor's rank"
    );

    // ── the COMPUTE OP names the index side, in the labeled-DS spelling ──
    assert_eq!(
        j["dscs_"][0]["MatMul_0"]["computeOp_"][0]["indirectAccessIndexLabeledDs"],
        serde_json::json!(["Tensor1-idx1"]),
        "the op must name the index operand, exactly as the vendor's [\"Tensor1-idx1\"]"
    );
}

/// ⭐⭐⭐ `attach_gather_index` ADDS THE OPERAND AND THE DECLARATION TOGETHER, and the result emits
/// the vendor's field set end-to-end — the path attention will actually take.
///
/// The two failure modes this method exists to prevent, both asserted below: an index operand pushed
/// with no declaration (dxp reads it as a real arithmetic input), and a declaration naming an operand
/// that is not there.
#[test]
fn attach_gather_index_wires_the_operand_and_the_declaration_as_one() {
    let mut op = plain_matmul();
    let before = op.args.len();
    let value_idx = 0usize;

    let declared = op
        .attach_gather_index(vendor_index("BlockTable"), value_idx)
        .expect("a valid pair");
    assert_eq!(op.args.len(), before + 1, "exactly one operand is added");
    assert_eq!(declared.value, value_idx);
    // ⛔⛔⛔ THE INDEX GOES BEFORE THE OUTPUT, AND THIS ASSERTION USED TO SAY THE OPPOSITE.
    //
    // It read `declared.index == op.args.len() - 1` with the message "the index is APPENDED, so it
    // cannot displace the output" — which had the mechanism exactly backwards. `emit_sdsc` defines the
    // output as `views.len() - 1`, so being last IS being the output: the op would have written its
    // scores over the block table. A green test asserting the defect is worse than no test, and this is
    // the second time that shape of error has been caught in this file's subject matter.
    assert_eq!(
        declared.index,
        op.args.len() - 2,
        "the index must sit immediately BEFORE the output, as the vendor's [input, index, output] does"
    );
    assert!(
        matches!(
            op.args[op.args.len() - 1].view().role,
            ktir_superdsc::superdsc_opspec::Role::Output
        ),
        "the LAST arg must still be the real output — that is what every reader's `out_idx` resolves to"
    );
    assert_eq!(
        op.indirect,
        Some(declared),
        "the declaration is stored, not just returned"
    );

    // And it emits the vendor's shape.
    let folds = SdscFoldSet::new(op.iter.cores_used());
    let e = superdsc::emit_sdsc("MatMul_0", &op, &folds, None).expect("a gather emits");
    let j = serde_json::to_value(&e).unwrap();
    assert_eq!(
        node(&j, "MatMul_0", value_idx)["indirectAllocType_"],
        "value_tensor"
    );
    assert_eq!(
        node(&j, "MatMul_0", declared.index)["indirectAllocType_"],
        "index_tensor"
    );
    assert_eq!(
        j["dscs_"][0]["MatMul_0"]["computeOp_"][0]["indirectAccessIndexLabeledDs"],
        serde_json::json!([format!("Tensor{}-idx{}", declared.index, declared.index)]),
    );
}

/// ⛔ GATHERING INTO THE OUTPUT IS A SCATTER, so the method refuses rather than emitting a plausible
/// wrong thing. An out-of-range value operand is refused for the same reason.
#[test]
fn attach_gather_index_refuses_the_output_and_a_missing_operand() {
    let mut op = plain_matmul();
    let out_idx = op.args.len() - 1;
    assert!(
        op.attach_gather_index(vendor_index("T"), out_idx).is_none(),
        "a write through an index is a SCATTER, not this"
    );
    assert!(
        op.attach_gather_index(vendor_index("T"), 99).is_none(),
        "a declaration cannot name an operand that is not there"
    );
    // Both refusals leave the op untouched — no half-wired state.
    assert!(op.indirect.is_none(), "a refused attach declares nothing");
}

/// ⭐ THE VENDOR FIXTURE IS THE ORACLE, so read it and assert our shapes against ITS values — not
/// against numbers transcribed into this file. If the fixture is not present the test says so
/// instead of passing quietly: a skipped oracle is not a satisfied one.
#[test]
fn our_shapes_agree_with_the_vendor_fixtures_own_values() {
    let Ok(text) = std::fs::read_to_string(VENDOR) else {
        panic!(
            "vendor gather fixture missing at {VENDOR} — this test's oracle is gone, so it can \
                neither pass nor be trusted. Restore the deeptools checkout."
        );
    };
    let v: serde_json::Value = serde_json::from_str(&text).expect("the fixture parses");
    let tree = &v["1_identity"]["dscs_"][0]["identity"]["scheduleTree_"];

    // Find the vendor's index and value nodes by their own annotations.
    let nodes = tree.as_array().expect("a schedule tree");
    let vidx = nodes
        .iter()
        .find(|n| n["indirectAllocType_"] == "index_tensor")
        .expect("the vendor has an index node");
    let vval = nodes
        .iter()
        .find(|n| n["indirectAllocType_"] == "value_tensor")
        .expect("the vendor has a value node");

    // The three facts we build against, read off the fixture rather than remembered.
    assert_eq!(vidx["indexTensorType_"], "index");
    // ⛔ THIS ONE FIELD IS THIS FIXTURE'S OWN FEATURE, NOT THE GATHER'S. `test_gather_1core` exists to
    // exercise the SYMBOLIC index-to-address path (`data_ "-1"`, `%base_addr` in its `bundle.mlir`), so
    // it reads `1` here — and copying that while emitting a concrete address is what dbo refused on the
    // pod. Asserted as the fixture's value, with the paged fixture named as the one our emission follows,
    // so this file records the DISAGREEMENT rather than silently preferring the wrong side.
    assert_eq!(
        vval["isStartAddrSymbolic_"], 1,
        "test_gather_1core is the symbolic-path test; IBM's sdsc_add_paged_l3lu.json has 0, and ours \
         follows the paged one because our start address is concrete"
    );
    assert_eq!(
        vidx["relatedIndirectAccessAlloc_"], vval["name_"],
        "the vendor's index node names its value node"
    );
    assert_eq!(
        vval["relatedIndirectAccessAlloc_"], vidx["name_"],
        "and the vendor's value node names its index node — the link is mutual"
    );

    // The value side's walk mixes a pin with -1s, which is the shape `DeviceWalk::paged_at` exists
    // to make representable without allowing an arbitrary mix on a directly-addressed tensor.
    let vwalk: Vec<i64> = vval["maxDimSizes_"]
        .as_array()
        .expect("the vendor walk")
        .iter()
        .map(|d| d.as_i64().expect("integer"))
        .collect();
    assert!(
        vwalk.contains(&1) && vwalk.contains(&-1),
        "the vendor's value walk is a MIX (theirs: {vwalk:?}) — a pin plus unbounded dims"
    );
    assert_eq!(
        vwalk.iter().filter(|&&d| d == 1).count(),
        1,
        "and exactly one dim is paged (theirs: {vwalk:?})"
    );

    // The compute op names the index side with the `Tensor{i}-idx{i}` spelling we reproduce.
    let vop =
        &v["1_identity"]["dscs_"][0]["identity"]["computeOp_"][0]["indirectAccessIndexLabeledDs"];
    let named = vop[0].as_str().expect("the vendor names one index DS");
    assert!(
        named.contains("-idx"),
        "the vendor's index reference is a labeled-DS name (theirs: {named})"
    );
}

/// ⛔⛔⛔⛔⛔ **DEEPTOOLS CANNOT SCHEDULE A GATHER ON AN OP THAT HAS A `KERNEL`** — the constraint that
/// stops the shipped prefix score leg from carrying one, computed from the vendor's own predicate.
///
/// ## Measured on the card, twice, bracketing the cause
/// granite-3.1-2b fp8, pod `nickm-7db9667cdd-z2jc6`:
/// ```text
/// index memOrg_ = hbm + lx  ->  sbf-ddc: DtException: Expect a valid allocate node.
///                               L3DlOpsScheduler.cpp:2337
/// index memOrg_ = hbm       ->  sbf-ddc: DtException: Expect LX in labeledDs memOrg_.
///                               L3DlOpsScheduler.cpp:2334
/// ```
/// Both are in `calculateFlopPerByte`: for every HBM-pinned labeledDs it requires an LX `memOrg_` entry
/// (2334) AND a non-null LX allocate node (2337). `allocAllMem` gives an LX chunk to each STAGED
/// operand and never to an index — an index is read into the IBR, not staged — so no `memOrg_` can
/// satisfy both. There is no third option to try.
///
/// ## And it only runs for a REUSE op
/// `L3DlOpsScheduler.cpp:1550` gates the whole arithmetic-intensity explorer on `isReuse`, and
/// `hasDimensionReuse` (`:303-321`) is exactly
/// `primaryDsInfo_.size() > 1 && primaryDsInfo_.count(DsTypes::KERNEL)`.
///
/// This test evaluates that predicate over BOTH vendor fixtures and over our own gathered matmul. The
/// fixtures are false (their ops are `identity` and `AddZero` — no KERNEL); ours is true. So the
/// requirement is not a bug we tripped, it is a shape the vendor never schedules: IBM's paged attention
/// gathers on a KERNEL-less elementwise op and feeds the RESULT to its matmul.
///
/// ⛔ WHY AS A TEST AND NOT A COMMENT. The next attempt's instinct is to try another `memOrg_`, another
/// dtype, another pin — the same four-week loop this file's header describes. The predicate is three
/// lines of C++ and it is decidable from the JSON, so it is decided here instead.
#[test]
fn deeptools_only_schedules_a_gather_on_a_kernel_less_op() {
    /// `L3DlOpsScheduler::hasDimensionReuse`, verbatim in the part that matters: more than one role,
    /// and one of them a KERNEL.
    fn has_dimension_reuse(primary: &serde_json::Value) -> bool {
        let Some(m) = primary.as_object() else {
            return false;
        };
        m.len() > 1 && m.contains_key("KERNEL")
    }

    // ── the two vendor fixtures: neither op has a KERNEL, so neither reaches the explorer ──
    for path in [
        VENDOR,
        "/Users/nickm/git/deeptools/dcg/dcg_fe/scheduler/test/sdsc_add_paged_l3lu.json",
    ] {
        let Ok(text) = std::fs::read_to_string(path) else {
            panic!("vendor fixture missing at {path} — this test's oracle is gone");
        };
        // A lit-test fixture carries `//` lines; strip them before parsing.
        let json: String = text
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        let v: serde_json::Value = serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("{path} does not parse after stripping comments: {e}"));
        // Find every `primaryDsInfo_` in the file and check the ones belonging to a gather.
        let mut found = 0usize;
        let mut stack = vec![v];
        while let Some(node) = stack.pop() {
            if let Some(m) = node.as_object() {
                if let Some(p) = m.get("primaryDsInfo_")
                    && p.get("KERNEL_IDX").is_some()
                {
                    found += 1;
                    assert!(
                        !has_dimension_reuse(p),
                        "{path}: a vendor gather op DOES have dimension reuse — the premise of this \
                         test (that the vendor never gathers on a KERNEL-bearing op) is wrong, and \
                         the card refusal needs another explanation"
                    );
                }
                stack.extend(m.values().cloned());
            } else if let Some(a) = node.as_array() {
                stack.extend(a.iter().cloned());
            }
        }
        assert!(
            found > 0,
            "{path}: no gather op with a KERNEL_IDX role found — the fixture moved and this oracle \
             checked nothing"
        );
    }

    // ── A MATMUL, which HAS a KERNEL, so the explorer runs and the bake is refused ──
    //
    // ⛔ `matmul_opspec` EXPLICITLY, not this file's harness. The harness is the KERNEL-less gather-copy
    // op now, so asking it would compare the shape that WORKS against itself and pass vacuously — the
    // shape of tautology this codebase has been burned by (`a-spot-check-against-its-own-parameters-
    // verifies-nothing`). The op that is refused is the one that must be measured.
    let mm = matmul_opspec(384, 384, 64, 16, "Tensor0", "Tensor1", "Tensor2").expect("a matmul");
    let folds = SdscFoldSet::new(mm.iter.cores_used());
    let e = superdsc::emit_sdsc("MatMul_0", &mm, &folds, None).expect("an UNGATHERED matmul emits");
    let j = serde_json::to_value(&e).unwrap();
    let ours = &j["dscs_"][0]["MatMul_0"]["primaryDsInfo_"];
    assert!(
        has_dimension_reuse(ours),
        "a matmul no longer reports dimension reuse ({ours:?}) — if `primaryDsInfo_` lost its KERNEL \
         role, the scheduler's arithmetic-intensity explorer would be SKIPPED, and the bake refusal \
         this test records should be re-measured on the card rather than assumed"
    );
    // ⭐ AND THE GATHER-COPY OP IS ON THE OTHER SIDE OF THE SAME PREDICATE — the comparison that makes
    // this test about a DIFFERENCE rather than about one op.
    let copy = plain_matmul();
    let cfolds = SdscFoldSet::new(copy.iter.cores_used());
    let ce = superdsc::emit_sdsc("Gather_0", &copy, &cfolds, None).expect("the copy op emits");
    let cj = serde_json::to_value(&ce).unwrap();
    let theirs = &cj["dscs_"][0]["Gather_0"]["primaryDsInfo_"];
    assert!(
        !has_dimension_reuse(theirs),
        "the gather-copy op reports dimension reuse ({theirs:?}) — it has acquired a KERNEL, and the \
         card refuses a gather on that"
    );
}

/// ⛔⛔⛔⛔⛔ THE INDEX'S **DTYPE AND LAYOUT**, WHICH EVERY OTHER TEST IN THIS FILE WAS BLIND TO.
///
/// This file's other assertions read the `scheduleTree_` **allocate** node — and an allocate node
/// carries NO dataFormat field at all. The dtype lives on the `labeledDs_` entry joined by `ldsIdx_`,
/// and the per-role stick geometry in `primaryDsInfo_`. So the emitter could give the index operand
/// the VALUE operand's fp16 format and this file stayed green: MEASURED, by changing the dtype and
/// re-running the crate — 102 passed, 0 failed, before AND after.
///
/// Two defects were hiding behind that blindness, and both are wrong ADDRESSES, not wrong numbers:
///
/// 1. `attach_gather_index` took the index's `df` from `self.args[value_idx]` — fp16, 2 bytes,
///    64/stick. The vendor's index is `SENUINT32`, 4 bytes, 32/stick, in EVERY fixture, and the
///    hardware reads one entry as a 4-byte uint32 out of the 128-byte IBR stick
///    (`senulator/memoryElement.cpp:810-813` asserts `SENUINT32`; `sendefs.h:63` gives it 32/stick).
/// 2. the index operand was `Role::Input`, and `primaryDsInfo_` is keyed by `ds_type()` with
///    first-writer-wins — so the ACTIVATION's `["mb","in"]`/`[64]` held the "INPUT" slot and the
///    index's own `["out"]`/`[32]` was never emitted at all. The vendor gives it a separate
///    `KERNEL_IDX` role precisely so the two layouts can differ.
///
/// Asserted against the fixture's own values, never against numbers transcribed into this file.
#[test]
fn the_index_operand_carries_the_vendors_own_dtype_and_its_own_layout() {
    // ── the ORACLE ──
    let Ok(text) = std::fs::read_to_string(VENDOR) else {
        panic!(
            "vendor gather fixture missing at {VENDOR} — this test's oracle is gone, so it can \
             neither pass nor be trusted. Restore the deeptools checkout."
        );
    };
    let v: serde_json::Value = serde_json::from_str(&text).expect("the fixture parses");
    let vdsc = &v["1_identity"]["dscs_"][0]["identity"];
    let vlabeled = vdsc["labeledDs_"]
        .as_array()
        .expect("the vendor's labeledDs_");
    let vindex = vlabeled
        .iter()
        .find(|l| l["dsType_"] == "KERNEL_IDX")
        .expect("the vendor's index LDS is a KERNEL_IDX");
    let vvalue = vlabeled
        .iter()
        .find(|l| l["dsType_"] == "OUTPUT")
        .expect("the vendor's value LDS");
    let vfmt = vindex["dataFormat_"]
        .as_str()
        .expect("the vendor's index format");
    let vword = vindex["wordLength"]
        .as_i64()
        .expect("the vendor's index word length");
    let vstick = vdsc["primaryDsInfo_"]["KERNEL_IDX"]["stickSize_"][0]
        .as_i64()
        .expect("the vendor's KERNEL_IDX stick size");
    let vvalue_stick = vdsc["primaryDsInfo_"]["OUTPUT"]["stickSize_"][0]
        .as_i64()
        .expect("the vendor's OUTPUT stick size");
    // ⭐ THE ORACLE IS CHECKED AGAINST ITSELF FIRST: the index and the value must genuinely DIFFER,
    // or "we match the vendor" would be satisfied by copying the value's format — which is the very
    // thing that was wrong.
    assert_ne!(
        vfmt,
        vvalue["dataFormat_"]
            .as_str()
            .expect("the vendor's value format"),
        "the vendor's index and value formats must differ, else this test proves nothing"
    );
    assert_ne!(
        vword,
        vvalue["wordLength"]
            .as_i64()
            .expect("the vendor's value word length"),
        "the vendor's index and value word lengths must differ"
    );
    assert_ne!(
        vstick, vvalue_stick,
        "the vendor's index and value stick sizes must differ"
    );

    // ── OUR emission ──
    let mut op = plain_matmul();
    let declared = op
        .attach_gather_index(vendor_index("BlockTable"), 0)
        .expect("a valid pair");
    let folds = SdscFoldSet::new(op.iter.cores_used());
    let e = superdsc::emit_sdsc("MatMul_0", &op, &folds, None).expect("a gather emits");
    let j = serde_json::to_value(&e).unwrap();
    let dsc = &j["dscs_"][0]["MatMul_0"];
    let labeled = dsc["labeledDs_"].as_array().expect("our labeledDs_");

    // Found by the `ldsIdx_` the DECLARATION names, not by scanning for the dsType this test is about
    // to assert — otherwise a wrong dsType would make the lookup miss and the test would pass on an
    // operand it never found.
    let ours = labeled
        .iter()
        .find(|l| l["ldsIdx_"].as_u64() == Some(declared.index as u64))
        .unwrap_or_else(|| {
            panic!(
                "no labeledDs_ for the index operand at ldsIdx_ {}",
                declared.index
            )
        });

    assert_eq!(
        ours["dataFormat_"].as_str(),
        Some(vfmt),
        "the index LDS must carry the vendor's INDEX format, not the value operand's — \
         GatherIndexConversion.cpp:133 is DT_CHECK(dataFormat_ == SENUINT32)"
    );
    assert_eq!(
        ours["wordLength"].as_i64(),
        Some(vword),
        "the index LDS must be {vword} bytes per entry — L3DlOpsScheduler.cpp:5928 is \
         DT_CHECK(indexLds.wordLength == 4), and it strides the index buffer by THAT width"
    );
    assert_eq!(
        ours["dsType_"].as_str(),
        Some("KERNEL_IDX"),
        "the index is its own dsType, which is what gives it its own primaryDsInfo_ entry"
    );

    // ── ITS OWN primaryDsInfo_ ENTRY, distinct from the activation's ──
    let ourinfo = &dsc["primaryDsInfo_"]["KERNEL_IDX"];
    assert!(
        !ourinfo.is_null(),
        "the index operand emitted no KERNEL_IDX layout — under Role::Input it shares (and loses) \
         the INPUT slot with the activation, so dxp reads the activation's rank and stick for it"
    );
    assert_eq!(
        ourinfo["stickSize_"][0].as_i64(),
        Some(vstick),
        "the index's stick is {vstick} entries per 128-byte stick (128 / {vword} bytes)"
    );
    // And it describes the INDEX's dims, which for a one-pin gather is rank 1 — not the activation's
    // rank 2. This is the assertion that fails if the two roles collapse again.
    let ourdims = ourinfo["layoutDimOrder_"]
        .as_array()
        .expect("the index's dims");
    assert_eq!(
        ourdims.len(),
        declared.pins().len(),
        "the index's layout must be exactly its PAGED dims (got {ourdims:?})"
    );
    // ⛔ AGAINST THE **OUTPUT** ROLE, NOT `INPUT`. On the KERNEL-less gather-copy op every data operand
    // is typed OUTPUT (that is what keeps KERNEL out of `primaryDsInfo_`, which is the only shape the
    // card will schedule a gather on) — so there is no INPUT entry to compare against, and asking for
    // one panicked. The property being checked is unchanged: the index must NOT have inherited a data
    // operand's layout.
    let value_dims = dsc["primaryDsInfo_"]["OUTPUT"]["layoutDimOrder_"]
        .as_array()
        .expect("the gathered source's dims");
    assert_ne!(
        ourdims, value_dims,
        "the index and the gathered source resolved to the SAME layout — the KERNEL_IDX entry is the \
         source's, so the index's own shape was dropped"
    );
    // ⛔ THE **OUTPUT** ROLE AGAIN — see the note above. On this op the data operands are OUTPUT, which
    // is also the role the vendor's own value tensor carries (`lds 0 Tensor0 OUTPUT`), so this compares
    // like with like where the `INPUT` spelling compared against a role that does not exist.
    assert_eq!(
        dsc["primaryDsInfo_"]["OUTPUT"]["stickSize_"][0].as_i64(),
        Some(vvalue_stick),
        "adding the index must not have moved the gathered SOURCE's own stick geometry"
    );
}

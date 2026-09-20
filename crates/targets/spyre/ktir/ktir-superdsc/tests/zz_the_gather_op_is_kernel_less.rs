// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ THE GATHER'S OP IS **KERNEL-LESS**, AND THE CARD IS WHY.
//!
//! ## What the card refused, twice, and what the C++ said
//! With the gather on the prefix score leg's Kᵗ operand (a matmul), granite-3.1-2b fp8 on pod
//! `nickm-7db9667cdd-z2jc6`:
//! ```text
//! index memOrg_ = hbm + lx  ->  sbf-ddc: DtException: Expect a valid allocate node.
//!                               L3DlOpsScheduler.cpp:2337
//! index memOrg_ = hbm       ->  sbf-ddc: DtException: Expect LX in labeledDs memOrg_.
//!                               L3DlOpsScheduler.cpp:2334
//! ```
//! The two refusals BRACKET the cause. Both lines sit in `calculateFlopPerByte`, which for every
//! HBM-pinned labeledDs requires an LX `memOrg_` entry (2334) AND a non-null LX allocate node (2337).
//! `allocAllMem` gives an LX chunk to each STAGED operand and never to an index — an index is read
//! into the IBR, not staged — so no `memOrg_` satisfies both. There is no third value.
//!
//! And it only runs for a REUSE op: `:1550` gates the whole arithmetic-intensity explorer on
//! `isReuse`, and `hasDimensionReuse` (`:303-321`) is verbatim
//! `primaryDsInfo_.size() > 1 && primaryDsInfo_.count(DsTypes::KERNEL)`.
//!
//! ## So the condition is exactly "has a KERNEL", and both vendor fixtures avoid it
//! ```text
//! dxp/test/test_gather_1core/sdsc_1.json   op `identity`   primaryDsInfo_ {OUTPUT, KERNEL_IDX}
//!   lds 0 Tensor0  OUTPUT      wl 2  SEN169_FP16   <- the gathered SOURCE, typed OUTPUT
//!   lds 1 Tensor1  KERNEL_IDX  wl 4  SENUINT32     <- the index
//!   lds 2 Tensor2  OUTPUT      wl 2  SEN169_FP16   <- the contiguous DESTINATION
//! dcg/.../sdsc_add_paged_l3lu.json          op `AddZero`   primaryDsInfo_ {OUTPUT, KERNEL_IDX}
//! ```
//! Neither gathers on a matmul. IBM's paged attention gathers on an elementwise copy and feeds the
//! RESULT to its matmul.
//!
//! ⛔ THIS FILE IS THE SHAPE WE SHIP. The other gather tests in this crate were written against a
//! matmul, which `emit_sdsc` now REFUSES at build time — so they were moved onto `gather_copy_opspec`,
//! the same door production uses. A descriptor test that exercises a shape the card cannot bake is
//! worse than no test: it is green, detailed, and about nothing.

use ktir_superdsc::emit as superdsc;
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{gather_copy_opspec, matmul_opspec};
use ktir_superdsc::superdsc_opspec::{
    GatherIndex, IndirectAccess, KernelAxis, PageExtent, SdscFoldSet,
};

/// The vendor's own declaration: its value tensor is `["mb","out","x"]` with `maxDimSizes_ [1,-1,-1]`
/// — so the PAGED axis is `mb`, pinned to ONE POSITION. Reproduced exactly, which a matmul harness
/// could not do (a matmul has no `mb` on its kernel; this op does).
const VENDOR_DECL: (KernelAxis, PageExtent) = (KernelAxis::Batch, PageExtent::single_position());

const VENDOR: &str = "/Users/nickm/git/deeptools/dxp/test/test_gather_1core/sdsc_1.json";

fn gather_copy(decl: (KernelAxis, PageExtent)) -> ktir_superdsc::superdsc_opspec::OpSpec {
    // ⛔ `out` IS ONE STICK, WHICH THE BUILDER REQUIRES — it was 256. The gathered destination has to be
    // ROW-MAJOR for the score kernel to read a block contiguously, and `[mb, out]` is only row-major at one
    // stick of `out` (above it `for_view_df` classifies RowBlocked/stick-major, scattering each block
    // `rows*64` apart). A block is therefore declared as `hd` one-stick sub-rows with the entry `page`
    // covering them — `GatherScratch::sub_rows`/`entry_page`.
    gather_copy_opspec(
        "KtPool",
        "KtScratch",
        8,
        ktir_superdsc::sdsc_abstract::POOL_STICK,
        GatherIndex {
            name: "BlockTable".to_string(),
            entry_dim: decl.0,
            page: decl.1,
            per_position: None,
            first_entry: ktir_superdsc::superdsc_opspec::EntryBase::ZERO,
        },
    )
    .expect("the gather-copy op builds")
}

fn dsc(j: &serde_json::Value) -> &serde_json::Value {
    &j["dscs_"][0]["Gather_0"]
}

fn emit(op: &ktir_superdsc::superdsc_opspec::OpSpec) -> serde_json::Value {
    let folds = SdscFoldSet::new(op.iter.cores_used());
    let e = superdsc::emit_sdsc("Gather_0", op, &folds, None).expect("the gather-copy op emits");
    serde_json::to_value(&e).unwrap()
}

/// ⭐⭐⭐⭐⭐ THE ROLE SET IS THE VENDOR'S: `{OUTPUT, KERNEL_IDX}` and nothing else.
///
/// This is the ONE fact the card refuses on, so it is asserted against the fixture's own key set
/// rather than against a list typed into this file.
#[test]
fn the_gather_ops_role_set_is_exactly_the_vendors() {
    let Ok(text) = std::fs::read_to_string(VENDOR) else {
        panic!("vendor gather fixture missing at {VENDOR} — this test's oracle is gone");
    };
    let v: serde_json::Value = serde_json::from_str(&text).expect("the fixture parses");
    let mut vendor_roles: Vec<String> = v["1_identity"]["dscs_"][0]["identity"]["primaryDsInfo_"]
        .as_object()
        .expect("the vendor's primaryDsInfo_")
        .keys()
        .cloned()
        .collect();
    vendor_roles.sort();
    // The oracle, checked against itself: the vendor's gather op must genuinely lack a KERNEL, or
    // "we match the vendor" would be satisfied by a matmul.
    assert!(
        !vendor_roles.iter().any(|r| r == "KERNEL"),
        "the vendor's gather op has a KERNEL ({vendor_roles:?}) — the premise of this whole file is \
         wrong and the card refusal needs another explanation"
    );

    let j = emit(&gather_copy(VENDOR_DECL));
    let mut ours: Vec<String> = dsc(&j)["primaryDsInfo_"]
        .as_object()
        .expect("our primaryDsInfo_")
        .keys()
        .cloned()
        .collect();
    ours.sort();
    assert_eq!(
        ours, vendor_roles,
        "our gather op's role set must be the vendor's exactly — anything with a KERNEL turns on \
         `hasDimensionReuse` and the bake is refused"
    );
}

/// ⛔⛔⛔ AND A GATHER ON A KERNEL-BEARING OP IS A **BUILD FAILURE**, not a card refusal.
///
/// The card told us twice; from here the emitter tells us at `cargo build`, with the shape that DOES
/// work named in the message. This is the assertion that stops the next session from re-running the
/// `memOrg_` / dtype / pin loop.
#[test]
fn a_gather_on_a_matmul_is_refused_at_build_time() {
    let mut mm =
        matmul_opspec(384, 384, 64, 16, "Tensor0", "Tensor1", "Tensor2").expect("a matmul");
    // The DECLARATION still attaches — it is a legal declaration, and refusing it here would hide
    // WHICH stage cannot serve it.
    mm.attach_gather_index(GatherIndex::of_pool_blocks("BlockTable".to_string()), 0)
        .expect("the declaration itself is well formed");
    let folds = SdscFoldSet::new(mm.iter.cores_used());
    let err = match superdsc::emit_sdsc("MatMul_0", &mm, &folds, None) {
        Err(e) => e.0,
        Ok(_) => panic!(
            "a gather on a matmul EMITTED. deeptools refuses to schedule it \
             (L3DlOpsScheduler.cpp:2334/2337 via hasDimensionReuse:303 + :1550), measured as a bake \
             failure on the card at BOTH index memOrg_ values — so this must be a build error, not a \
             descriptor that reaches dxp"
        ),
    };
    assert!(
        err.contains("KERNEL"),
        "the refusal must name the KERNEL as the cause (got: {err})"
    );
    assert!(
        err.contains("KERNEL-LESS") || err.contains("kernel-less"),
        "the refusal must name the shape that WORKS, or it sends the reader back to the card \
         (got: {err})"
    );
}

/// ⭐ THE INDEX OPERAND KEEPS EVERY PROPERTY THE MATMUL HARNESS ESTABLISHED — dtype, word length,
/// stick size and its own layout — now on the op that can actually bake.
#[test]
fn the_index_keeps_its_dtype_and_its_own_layout_on_the_gather_op() {
    let Ok(text) = std::fs::read_to_string(VENDOR) else {
        panic!("vendor gather fixture missing at {VENDOR}");
    };
    let v: serde_json::Value = serde_json::from_str(&text).expect("parses");
    let vd = &v["1_identity"]["dscs_"][0]["identity"];
    let vidx = vd["labeledDs_"]
        .as_array()
        .expect("labeledDs_")
        .iter()
        .find(|l| l["dsType_"] == "KERNEL_IDX")
        .expect("the vendor's index LDS");

    let op = gather_copy(VENDOR_DECL);
    let ia = op.indirect.expect("a declared gather");
    let j = emit(&op);
    let d = dsc(&j);
    let ours = d["labeledDs_"]
        .as_array()
        .expect("labeledDs_")
        .iter()
        .find(|l| l["ldsIdx_"].as_u64() == Some(ia.index as u64))
        .expect("our index LDS at the declared operand position");

    assert_eq!(ours["dsType_"], "KERNEL_IDX");
    assert_eq!(
        ours["dataFormat_"], vidx["dataFormat_"],
        "the index's format is the vendor's SENUINT32, never the source operand's fp16"
    );
    assert_eq!(
        ours["wordLength"], vidx["wordLength"],
        "4 bytes per entry — L3DlOpsScheduler.cpp:5928 is DT_CHECK(indexLds.wordLength == 4)"
    );
    assert_eq!(
        d["primaryDsInfo_"]["KERNEL_IDX"]["stickSize_"][0],
        vd["primaryDsInfo_"]["KERNEL_IDX"]["stickSize_"][0],
        "32 entries per 128-byte stick"
    );
    // ⛔ AND THE INDEX IS HBM-ONLY. With an LX residency the scheduler dereferences an allocate node
    // `allocAllMem` never created — measured, L3DlOpsScheduler.cpp:2337.
    assert_eq!(
        ours["memOrg_"], vidx["memOrg_"],
        "the index must be HBM-only, exactly as the vendor's is"
    );
}

/// ⭐ THE CROSS-LINK, THE ANNOTATION, AND THE CONCRETE BASE — the three fields dxp reads, on the op
/// that can bake.
#[test]
fn the_pair_cross_links_and_the_base_stays_concrete() {
    let op = gather_copy(VENDOR_DECL);
    let ia = op.indirect.expect("a declared gather");
    let j = emit(&op);
    let d = dsc(&j);
    let tree = d["scheduleTree_"].as_array().expect("a schedule tree");

    let idx = &tree[ia.index];
    let val = &tree[ia.value];
    assert_eq!(idx["indirectAllocType_"], "index_tensor");
    assert_eq!(idx["indexTensorType_"], "index");
    assert_eq!(val["indirectAllocType_"], "value_tensor");
    assert_eq!(
        idx["relatedIndirectAccessAlloc_"], val["name_"],
        "the index node must name the value node"
    );
    assert_eq!(
        val["relatedIndirectAccessAlloc_"], idx["name_"],
        "and the value node must name the index node — the link is mutual"
    );
    // ⛔ `0`, NOT `1`. At `1` — transcribed from `test_gather_1core`, which is a unit test OF the
    // symbolic path and pairs the flag with `data_ "-1"` plus a `%base_addr` operand in its
    // `bundle.mlir` — dbo refused on the card with
    // `Symbol operand does not exist: 34360262656, VariableDefinition.cpp:303`, that number being the
    // operand's real BYTE address read as a symbol id. IBM's paged fixture has
    // `isStartAddrSymbolic_ 0` with `data_ "128000"`, and the paged one governs.
    //
    // ⭐ WHICH SETTLES WHAT `base_addr` IS: `idx * skip_addr + base_addr` adds the index to the
    // operand's OWN declared start, so a source offset survives and the index supplies only the term
    // it enumerates. Do not "fix" this back to 1; there is a card refusal for that value.
    assert_eq!(
        val["isStartAddrSymbolic_"], 0,
        "a concrete per-core start address must not be marked symbolic — dbo resolves it as a symbol \
         id and refuses the bake"
    );
    assert_eq!(
        d["computeOp_"][0]["indirectAccessIndexLabeledDs"],
        serde_json::json!([format!("Tensor{}-idx{}", ia.index, ia.index)]),
        "the compute op must NAME the index — the annotation is what makes it a gather rather than an \
         inert extra operand"
    );
    // The index is NOT an arithmetic input: the vendor lists it in exactly one place, and it is not
    // `inputLabeledDs`.
    let inputs = d["computeOp_"][0]["inputLabeledDs"]
        .as_array()
        .expect("inputs");
    assert!(
        !inputs
            .iter()
            .any(|r| r.as_str() == Some(&format!("Tensor{}-idx{}", ia.index, ia.index))),
        "the index appeared as an arithmetic input, giving the op one more operand than its DDL \
         declares"
    );
    // And the op is the vendor's own `identity` on the SFP.
    assert_eq!(d["computeOp_"][0]["opFuncName"], "identity");
    assert_eq!(d["computeOp_"][0]["exUnit"], "sfp");
}

/// ⛔ A SELF-INDEXED PAIR AND A DOUBLE-PINNED AXIS ARE STILL REFUSED — the declaration's own
/// invariants, unchanged by the move to a KERNEL-less op.
#[test]
fn the_declarations_own_refusals_survive_the_move() {
    assert!(IndirectAccess::of(1, 1, VENDOR_DECL.0, VENDOR_DECL.1, None).is_none());
    assert!(
        IndirectAccess::of(
            1,
            0,
            KernelAxis::Slot,
            PageExtent::of_one_stick(),
            Some(KernelAxis::Slot)
        )
        .is_none(),
        "two pins on one dim is one pin, and the survivor silently decides skip_addr"
    );
    // A paged axis this op does not have is refused by the builder rather than dropped: `in` belongs
    // to a matmul, and this op's dims are [mb, out, y].
    assert!(
        gather_copy_opspec(
            "KtPool",
            "KtScratch",
            8,
            256,
            GatherIndex {
                name: "BlockTable".to_string(),
                entry_dim: KernelAxis::Feature,
                page: PageExtent::single_position(),
                per_position: None,
                first_entry: ktir_superdsc::superdsc_opspec::EntryBase::ZERO,
            },
        )
        .is_err(),
        "an `in` axis is not a dim of the gather-copy op, and a silently-dropped pin is a gather with \
         no page declared — wrong addresses from a clean build"
    );
}

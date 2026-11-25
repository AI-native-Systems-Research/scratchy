// SPDX-License-Identifier: Apache-2.0
//! The ported work division's design targets: torch-spyre's OWN gemma sliding prefill
//! attention descriptors, stated as splits — never a JSON golden comparer.
//!
//! The two fixtures are `.claude/scratch/ts-attn-spec`'s `13_batchmatmul`
//! (`numWkSlicesPerDim_ {y:8, x:2, mb:2, out:1, in:1}`, 32 cores) and `14_add`
//! (`{mb:8, x:2, y:2, i:1, out:1}`, 32 cores), at gemma-4 12B sliding shapes:
//! 8 kv heads, 2 query groups, 128 padded queries, 128-token blocks, head size 256.

use ktir_superdsc::work_division::{DimRole, IterDim, SplitDomain, WorkDivision};

fn dim(name: &'static str, extent: u32, role: DimRole, domain: SplitDomain) -> IterDim {
    IterDim {
        name,
        extent,
        role,
        domain,
    }
}

/// `13_batchmatmul`: the scores contraction `[8, 256, 256] @ [8, 256, 128]`, one op over
/// its stated batch. `y` is the kv-head batch, `x` the query group, `mb` the query rows
/// (the gathered LHS's row dim — its per-core block is whole 64-row stage slabs), `out`
/// the N dim of the gathered K page (indirectly accessed — whole), `in` the contraction
/// (whole: no combine op exists).
#[test]
fn the_batchmatmul_split_is_torch_spyres() {
    let dims = [
        dim("y", 8, DimRole::Output, SplitDomain::Any),
        dim("x", 2, DimRole::Output, SplitDomain::Any),
        dim(
            "mb",
            128,
            DimRole::Output,
            SplitDomain::StageSlabs { slab_rows: 64 },
        ),
        dim("out", 128, DimRole::Output, SplitDomain::Whole),
        dim("in", 256, DimRole::Reduction, SplitDomain::Whole),
    ];
    let d = WorkDivision::law(&dims, &[], 32).expect("the bmm divides");
    assert_eq!(d.split_of("y"), 8, "one core per kv head");
    assert_eq!(d.split_of("x"), 2, "the group dim fills the budget");
    assert_eq!(d.split_of("mb"), 2, "whole 64-row stage slabs per core");
    assert_eq!(d.split_of("out"), 1, "the gathered operand's N stays whole");
    assert_eq!(d.split_of("in"), 1, "no K split without a combine op");
    assert_eq!(d.cores(), 32);
}

/// `14_add`: the mask add fused in the same walk body as the contraction. The query dim
/// (`y` here, the bmm's `mb`) carries the anchor's committed split — the group's ops
/// cannot disagree about who owns which slice — and the add fills the remaining budget on
/// its own free dims.
#[test]
fn the_elementwise_split_takes_the_anchor_s_commitment() {
    let dims = [
        dim("mb", 8, DimRole::Output, SplitDomain::Any),
        dim("x", 2, DimRole::Output, SplitDomain::Any),
        dim("y", 128, DimRole::Output, SplitDomain::Any),
        dim("i", 128, DimRole::Output, SplitDomain::Whole),
        dim("out", 1, DimRole::Output, SplitDomain::Any),
    ];
    // The contraction committed query:2 and block:1; block:1 is the identity and drops.
    let d = WorkDivision::law(&dims, &[("y", 2)], 32).expect("the add divides");
    assert_eq!(
        d.split_of("y"),
        2,
        "the anchor's committed query split, ungrown"
    );
    assert_eq!(d.split_of("mb"), 8, "the kv heads fill the free budget");
    assert_eq!(d.split_of("x"), 2, "the group dim fills the rest");
    assert_eq!(d.split_of("i"), 1, "the block dim stays whole");
    assert_eq!(d.cores(), 32);
}

/// The budget is a law, not a preference: the product of the splits never exceeds it,
/// whatever the extents.
#[test]
fn the_core_budget_is_never_exceeded() {
    let dims = [
        dim("mb", 128, DimRole::Output, SplitDomain::Any),
        dim("x", 128, DimRole::Output, SplitDomain::Any),
    ];
    let d = WorkDivision::law(&dims, &[], 32).expect("the wide op divides");
    assert!(d.cores() <= 32);
    assert_eq!(d.split_of("mb") * d.split_of("x"), d.cores());
}

/// A committed split that does not divide its dim's extent is a refusal, not a rounding.
#[test]
fn an_illegal_commitment_refuses() {
    let dims = [dim("y", 128, DimRole::Output, SplitDomain::Any)];
    let err = WorkDivision::law(&dims, &[("y", 3)], 32).expect_err("3 does not divide 128");
    assert!(err.contains("does not divide"), "{err}");
    let err = WorkDivision::law(&dims, &[("y", 64)], 32).expect_err("over budget");
    assert!(err.contains("cores"), "{err}");
}

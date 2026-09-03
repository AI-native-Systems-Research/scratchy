// SPDX-License-Identifier: Apache-2.0

//! ⭐⭐⭐⭐⭐ WHO OWNS THE PAGE A BATCHED STEP WRITES PAST A REQUEST'S OWN TOKENS — AND WHAT IT COSTS.
//!
//! A batched step appends every row at ONE shared slot, so a request shorter than its batch-mates has a write
//! page beyond its own keys. **The host owns it**: the scheduler allocates each request its own pages plus ONE
//! (`KvSlotSpan::blocks_this_step`), and the pool withholds a single read-only scratch page that every
//! fully-masked logical page aliases.
//!
//! ⛔⛔⛔ THIS FILE WAS `zz_the_hole_reserve_is_cheaper_than_host_allocation.rs` AND ARGUED THE OPPOSITE, WITH A
//! COST MODEL THAT WAS WRONG. It charged host allocation `(rows - 1) * pages_for(deepest_prompt)` — every row
//! allocated out to the deepest row's depth — measured that at 24-45 pages against the reserve's 9, and
//! concluded that a per-request `KvRow` and its reserved hole run should STAY. The pages between a row's keys
//! and its write page hold NO KEY: `map_row` aliases them to one scratch page. Host allocation costs each row
//! ONE page, flat in depth, and there is no crossover.
//!
//! ⛔ THAT WRONG NUMBER WAS QUOTED AS A REASON NOT TO DO THE WORK — it became "the row stays" in memory and in
//! a commit message, i.e. a cost measurement used to justify keeping the very thing the mandate says to
//! remove. The rule the emitter implements was readable in `map_row` the whole time. **VERIFY A COST MODEL
//! AGAINST THE CODE THAT SPENDS THE PAGES.**
//!
//! What the reserve cost while it existed, for the record: `rows * 2 + 1` pages, so 9 at width 4 and 65 at the
//! widest baked rung — pages the host could never allocate from, on a pool sized by the declaration.

use scratchy_subtile::sdsc_abstract::{
    AdmittedRequests, PagedKvPool, PoolPages, PoolPartition, PoolRows,
};

/// Pages a request needs to reach `tokens`, at the pool's own page size.
fn pages_for(tokens: u32) -> u32 {
    tokens.div_ceil(PagedKvPool::PAGE_SLOTS as u32)
}

/// ⭐⭐⭐ THE POOL WITHHOLDS ONE PAGE, WHATEVER THE WIDTH — the property that says the per-request reserve is
/// gone rather than merely unused.
#[test]
fn the_pool_withholds_exactly_one_page_at_every_width() {
    assert_eq!(
        PoolPartition::reserve(),
        1,
        "one shared read-only scratch page, and nothing per request"
    );
    for rows in [1u32, 2, 4, 8, 32] {
        let admitted = AdmittedRequests::new(rows as usize).expect("a width");
        let part = PoolPartition::of_pool(
            PoolPages::of_pool(136).expect("136 pages"),
            PoolRows::for_admission(admitted),
        )
        .expect("splits");
        assert_eq!(
            part.host_blocks(),
            135,
            "at width {rows} the host must be offered every page but the scratch one — a reserve that grew \
             with the width is what `rows * HOLE_PAGES_PER_ROW + 1` did (9 pages at width 4, 65 at 32)"
        );
    }
}

/// ⭐ THE WRITE PAGE COSTS ONE PAGE PER ROW, FLAT IN CONTEXT DEPTH. The old model made it linear in the
/// deepest prompt, which is the only reason a "crossover" ever existed.
#[test]
fn the_write_page_costs_one_page_per_row_at_any_depth() {
    let host_extra = |rows: u32, _deepest: u32| rows;
    assert_eq!(
        host_extra(4, 512),
        host_extra(4, 8192),
        "the masked gap between a row's keys and the write page is an aliased scratch page, not storage"
    );
    // Against the deleted reserve, at the shapes this branch gates on.
    for (rows, deepest) in [(4u32, 1840u32), (4, 3667), (32, 1840)] {
        let old_reserve = rows * 2 + 1;
        assert!(
            host_extra(rows, deepest) + PoolPartition::reserve() <= old_reserve,
            "rows={rows} deepest={deepest}tok: the write page plus the scratch page must not cost more than \
             the reserve it replaced ({old_reserve} pages) — if it ever does, the reserve was the better \
             design and this file must say so"
        );
    }
    // And the depth a row's own keys need is unchanged by any of this.
    assert_eq!(pages_for(1840), 8);
    assert_eq!(pages_for(3667), 15);
}

// SPDX-License-Identifier: Apache-2.0
//! ⭐ THE POOL-PAGES LAW — how many pages of KV pool a card holds, as ONE typed formula.
//!
//! The serving step's own question — "how many KV blocks fit in the memory left after the
//! weights?" — answered at the types the answer is spent from:
//!
//!   * the CARD's budget ([`CardBudget`]): its generation's tensor memory at the serving
//!     step's own utilization fraction
//!     ([`scratchy_core_config::DEFAULT_GPU_MEMORY_UTILIZATION`]);
//!   * the SESSION's residency ([`Residency`]): the bytes that sit whatever the pool's page
//!     count ([`Residency::fixed`]) and the bytes one more page costs ([`Residency::page`]) —
//!     read off the bake's own layout by [`session_residency`], never re-derived;
//!   * the MODEL's reach ([`ContextPages`]): the context's own pages, the floor.
//!
//! [`PoolPages::of`] is the whole law: the pages the budget's remainder holds at the
//! per-page bytes, never fewer than the context's own pages. Everything else here reads a
//! pool already sized — [`Residency::at`] its bytes, [`PoolPages::extra_rows`] the rows past
//! the context.

use std::num::NonZeroU32;
use std::num::NonZeroU64;
use std::ops::Add;

use ktir_superdsc::placement::{BundleLayout, SegRole};
use ktir_superdsc::sdsc_abstract::SchedulerBlockTokens;

/// ⭐ THE CARD'S BUDGET — the tensor memory a session's pools are sized within: the card
/// generation's own tensor memory at the serving step's utilization fraction. A newtype, not
/// a bare count: the value leaves through [`Self::get`], and only [`Self::of`] puts one in,
/// from the generation the bake targets.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CardBudget(u64);

impl CardBudget {
    /// State a budget in bytes. Tests name a figure directly; production reads [`Self::of`].
    #[cfg(test)]
    pub const fn new(bytes: u64) -> Self {
        Self(bytes)
    }

    /// The bytes, for the arithmetic that spends them.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// The budget a generation's card holds: its tensor memory at the serving step's own
    /// utilization fraction — the same fraction the reference reserves its KV pool with.
    ///
    /// [`TensorMemoryBytes`]: ktir_core::ddl_binds::TensorMemoryBytes
    pub fn of<G: ktir_core::ddl_binds::DeviceGen>() -> Self {
        Self(
            (G::TENSOR_MEMORY_BYTES.get() as f64
                * scratchy_core_config::DEFAULT_GPU_MEMORY_UTILIZATION) as u64,
        )
    }
}

/// ⭐ THE MODEL'S REACH, IN PAGES — the context's own pages: the positions a request's
/// attention may read, counted in scheduler blocks. NONZERO BY TYPE: a model with no context
/// is not a model. The FLOOR of [`PoolPages::of`] — the pool is sized for the model's own
/// reach, never below it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ContextPages(NonZeroU32);

impl ContextPages {
    /// State the context's pages.
    pub const fn new(pages: NonZeroU32) -> Self {
        Self(pages)
    }

    /// The context's pages.
    pub const fn get(self) -> NonZeroU32 {
        self.0
    }
}

/// The bytes that sit whatever the pool's page count — the weights, the activations, every
/// segment but the pools themselves.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FixedBytes(u64);

impl FixedBytes {
    /// State the fixed bytes.
    pub const fn new(bytes: u64) -> Self {
        Self(bytes)
    }

    /// The bytes.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// ⭐ THE BYTES ONE PAGE COSTS, AS THE SESSION SUMS THEM — every pool's page and the row
/// pool's page summed. A plain count, not a nonzero one: a session whose layout holds no
/// paged pools and whose step reads no row pool costs no page, and that state is legal here.
/// [`PageBytes::of`] is the gate that turns a page cost into the division's own divisor.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PerPageBytes(u64);

impl PerPageBytes {
    /// State the per-page bytes.
    pub const fn new(bytes: u64) -> Self {
        Self(bytes)
    }

    /// The per-page bytes.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// ⭐ THE BYTES ONE PAGE COSTS, AS THE DIVISION SPENDS THEM — nonzero by type: the divisor
/// of [`PoolPages::of`]'s own division. Stated from the session's summed page cost by
/// [`Self::of`], which holds the zero state at its one home — a session that costs no page
/// has no page law to divide by, and the `None` says so.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PageBytes(NonZeroU64);

impl PageBytes {
    /// State the per-page bytes.
    pub const fn new(bytes: NonZeroU64) -> Self {
        Self(bytes)
    }

    /// The per-page bytes.
    pub const fn get(self) -> NonZeroU64 {
        self.0
    }

    /// The page cost as the division's own divisor — `None` when the session costs no page.
    pub fn of(per: PerPageBytes) -> Option<Self> {
        NonZeroU64::new(per.get()).map(PageBytes)
    }
}

/// ⭐ THE SESSION'S RESIDENCY — what a session costs the card per page count: the bytes that
/// sit whatever the pool ([`Self::fixed`]) and the bytes one more page costs
/// ([`Self::page`]). Read off the bake's own layout by [`session_residency`]; two sessions'
/// residencies add ([`impl Add`]) to the residency their pages cost together.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Residency {
    /// The bytes that sit whatever the pool's page count.
    fixed: FixedBytes,
    /// The bytes one more page costs.
    page: PerPageBytes,
}

impl Residency {
    /// The bytes that sit whatever the pool's page count.
    pub const fn fixed(self) -> FixedBytes {
        self.fixed
    }

    /// The bytes one more page costs.
    pub const fn page(self) -> PerPageBytes {
        self.page
    }

    /// The bytes `pages` of this residency cost the card: the fixed bytes plus a page's bytes
    /// per page.
    pub const fn at(self, pages: PoolPages) -> u64 {
        self.fixed().get() + pages.get().get() as u64 * self.page().get()
    }
}

impl Add for Residency {
    type Output = Residency;
    fn add(self, rhs: Residency) -> Residency {
        Residency {
            fixed: FixedBytes::new(self.fixed().get() + rhs.fixed().get()),
            page: PerPageBytes::new(self.page().get() + rhs.page().get()),
        }
    }
}

/// ⭐ THE POOL'S PAGES — how many pages of KV pool the card holds. NONZERO BY TYPE, and
/// floored at the context's own pages by [`Self::of`]: a pool smaller than the model's reach
/// is not a pool this law sizes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PoolPages(NonZeroU32);

impl PoolPages {
    /// State the pool's pages.
    pub const fn new(pages: NonZeroU32) -> Self {
        Self(pages)
    }

    /// The pool's pages.
    pub const fn get(self) -> NonZeroU32 {
        self.0
    }

    /// ⭐ THE LAW — the pages the budget's remainder holds at the page's bytes, never fewer
    /// than the context's own pages. [`blocks_within`] is the serving step's own division
    /// (its sixteen-page floor and its zero-page fallback included — the page's own zero law
    /// lives there, at its one home), so a card that cannot hold the context still yields
    /// the context's pages — the pool is sized for the model's reach first and the card's
    /// remainder second. A zero count falls to the context's own pages: total, with no arm
    /// to reach.
    ///
    /// [`blocks_within`]: scratchy_core_config::blocks_within
    pub fn of(
        budget: CardBudget,
        fixed: FixedBytes,
        page: PageBytes,
        context: ContextPages,
    ) -> Self {
        let pages = scratchy_core_config::blocks_within(
            budget.get().saturating_sub(fixed.get()) as usize,
            page.get().get() as usize,
        ) as u32;
        Self(NonZeroU32::new(pages).map_or(context.get(), |p| p.max(context.get())))
    }

    /// The pool's rows past the context's pages — the positions a request may still fill
    /// before the pool is the context, at the scheduler's own page. SATURATES below the
    /// context (a state [`Self::of`]'s floor excludes), the same spelling the saturating row
    /// laws of the prefix capacity take.
    pub const fn extra_rows(self, context: ContextPages) -> u32 {
        self.get().get().saturating_sub(context.get().get()) * SchedulerBlockTokens::REFERENCE.get()
    }
}

/// ⭐ THE SESSION'S RESIDENCY, READ OFF THE BAKE'S OWN LAYOUT AND STEP — the fixed bytes are
/// every segment's extent but the KV segment's ([`BundleLayout::segment_extents`]: the
/// intermediate segment covers every synthetic the door declared; the pools are the pages,
/// charged per page), minus
/// the row pool's bytes when the session's step reads it (those bytes ride inside the
/// intermediate segment's total, and they are per-page too — the step's own
/// [`card_kv_row_pool_bytes`](crate::wiring::PagedStep::card_kv_row_pool_bytes), never a
/// layout lookup); the per-page bytes are every pool's page
/// ([`crate::wiring::pool_page_bytes`] at each pool's columns) plus the row pool's page
/// ([`crate::wiring::KV_ROW_POOL_PAGE_BYTES`]) when the step reads it. It reads the layout's
/// own numbers and the step's own bytes, and computes nothing else.
pub fn session_residency(
    layout: &BundleLayout,
    page_cols: &[u32],
    step: Option<crate::wiring::PagedStep>,
) -> Residency {
    let kv_seg = SegRole::Kv.segment();
    let mut fixed: u64 = layout
        .segment_extents()
        .into_iter()
        .enumerate()
        .filter(|&(seg, _)| seg != kv_seg)
        .map(|(_, bytes)| bytes)
        .sum();
    let mut page: u64 = page_cols
        .iter()
        .map(|&cols| crate::wiring::pool_page_bytes(cols))
        .sum();
    if let Some(step) = step
        && step.reads_kv_row_pool()
    {
        // The row pool's bytes sit inside the intermediate segment's total — take them out of
        // the fixed bytes and charge the run per page instead, both figures the step itself
        // states (`card_kv_row_pool_bytes` is its pages at the one page law).
        fixed -= step.card_kv_row_pool_bytes();
        page += crate::wiring::KV_ROW_POOL_PAGE_BYTES;
    }
    Residency {
        fixed: FixedBytes::new(fixed),
        page: PerPageBytes::new(page),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nz32(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).expect("test literal is nonzero")
    }

    fn nz64(n: u64) -> NonZeroU64 {
        NonZeroU64::new(n).expect("test literal is nonzero")
    }

    /// The fixture's extent pair — the same mint the paged step's own tests use
    /// (`of_baked_rung` over the rung's own width and capacity).
    fn pair(per_request: u32, requests: u32) -> crate::wiring::QueryExtent {
        crate::wiring::QueryExtent::of_baked_rung(
            crate::wiring::RungWidth::of_baked_rows(per_request)
                .expect("the fixture's row count is nonzero"),
            crate::wiring::RungCap::of_baked_seqs(requests)
                .expect("the fixture's request count is nonzero"),
        )
        .expect("the fixture's rows fit a u32")
    }

    /// A step of `pages` pages at `rows` query rows — the reserved sentinels every real
    /// wiring's step carries, one sequence's rows.
    fn step(pages: u32, rows: u32) -> crate::wiring::PagedStep {
        let tids = scratchy_triton_splice::PagedStepTids::RESERVED;
        crate::wiring::PagedStep {
            mask: tids.mask,
            page_table: tids.page_table,
            write_rows: tids.write_rows,
            query_rows: tids.query_rows,
            kv_row_pool: tids.kv_row_pool,
            rep_rows: tids.rep_rows,
            chunk_pages: tids.chunk_pages,
            chunk_masks: tids.chunk_masks,
            int32_zero: tids.int32_zero,
            pages,
            page_slots: SchedulerBlockTokens::REFERENCE.get(),
            rows,
            nkvh: 8,
            q_width: 32 * 64,
            head_size: 64,
            walk: pages,
            paged_rows: crate::paged_step::PagedRows::Sequence,
            extent: pair(rows, 1),
        }
    }

    /// A budget that cannot hold the context still yields the context's pages — and a budget
    /// below the fixed bytes saturates at zero remainder rather than refusing, so the floor
    /// is the context's own pages either way.
    #[test]
    fn the_pool_floors_at_the_contexts_pages() {
        let context = ContextPages::new(nz32(100));
        // Sixteen pages fit — `blocks_within`'s own floor — and the context needs a hundred.
        let pool = PoolPages::of(
            CardBudget::new(16 * 1024),
            FixedBytes::new(0),
            PageBytes::new(nz64(1024)),
            context,
        );
        assert_eq!(pool.get().get(), 100);
        // The budget below the fixed bytes: a zero remainder, the same floor.
        let pool = PoolPages::of(
            CardBudget::new(100),
            FixedBytes::new(4096),
            PageBytes::new(nz64(1024)),
            context,
        );
        assert_eq!(pool.get().get(), 100);
    }

    /// Above the floor, the pool is the budget's remainder divided by the page: floor
    /// division, the remainder dropped, the residency's own [`Residency::at`] closing the
    /// sum. The page reaches the law through [`PageBytes::of`]'s gate, as production does.
    #[test]
    fn the_pool_divides_the_remainder_by_the_page() {
        let page =
            PageBytes::of(PerPageBytes::new(1024)).expect("the fixture's page cost is nonzero");
        assert_eq!(PageBytes::of(PerPageBytes::new(0)), None);
        let fixed = FixedBytes::new(4096);
        let context = ContextPages::new(nz32(10));
        let pool = PoolPages::of(
            CardBudget::new(4096 + 100 * 1024 + 512),
            fixed,
            page,
            context,
        );
        assert_eq!(pool.get().get(), 100, "the 512-byte remainder is dropped");
        let residency = Residency {
            fixed,
            page: PerPageBytes::new(1024),
        };
        assert_eq!(residency.at(pool), 4096 + 100 * 1024);
        let pool = PoolPages::of(CardBudget::new(4096 + 250 * 1024), fixed, page, context);
        assert_eq!(pool.get().get(), 250);
    }

    /// The extra rows are the pages past the context, at the scheduler's own page; below the
    /// context the extra rows saturate at zero — a state [`PoolPages::of`]'s floor excludes.
    #[test]
    fn extra_rows_are_the_pages_past_the_context() {
        let pool = PoolPages::new(nz32(100));
        let context = ContextPages::new(nz32(40));
        assert_eq!(
            pool.extra_rows(context),
            60 * SchedulerBlockTokens::REFERENCE.get()
        );
        assert_eq!(PoolPages::new(nz32(7)).extra_rows(context), 0);
    }

    /// ⭐ BYTE-IDENTITY OF THE PAGE LAWS —
    ///
    /// * the row pool's page is the index run's int32 words, the figure the existing
    ///   `the_walk_covers_the_context_not_the_pool` test pins end-to-end through
    ///   `PagedStep::card_placements` (sixteen pages at `INDEX_RUN_WORDS × 4` each); pinning
    ///   the const here closes the chain through `card_kv_row_pool_bytes`'s page spelling.
    /// * a whole-page pool's bytes are the same counted by page or by row — `pages ×
    ///   pool_page_bytes(cols)` is the row-count spelling (`rows × row_footprint_bytes`)
    ///   the layout's pool placement was sized by, so the re-route through the page law
    ///   moves no byte.
    #[test]
    fn the_page_laws_are_byte_identical_to_the_row_spelling() {
        use ktir_superdsc::superdsc_opspec::Df;
        assert_eq!(
            crate::wiring::KV_ROW_POOL_PAGE_BYTES,
            u64::from(ktir_superdsc::emit::lower_ktir_to_superdsc::INDEX_RUN_WORDS) * 4,
        );
        for cols in [4u32, 64, 512, 576, 1024, 4224, 49152] {
            let pages = 16u32;
            let rows = pages * SchedulerBlockTokens::REFERENCE.get();
            assert_eq!(
                u64::from(pages) * crate::wiring::pool_page_bytes(cols),
                u64::from(rows)
                    * crate::lower_subtile_tape_to_superdsc::row_footprint_bytes(cols, Df::Fp16),
                "cols {cols}: a whole-page pool's bytes are the same by page or by row"
            );
        }
    }

    /// The session's residency charges the KV segment per page, never as fixed bytes, and
    /// the row pool's bytes leave the fixed total to ride per page when the session's step
    /// reads it — a step whose kernel does not, and a layout with no paged step at all, keep
    /// the intermediate segment whole and charge no run.
    #[test]
    fn session_residency_charges_the_kv_segment_per_page() {
        let mut layout = BundleLayout::default();
        // seg0 intermediate, seg1 weights, seg2 KV, seg3 activations.
        layout.segment_bytes = [8192, 200, 3072, 400, 0, 0, 0];
        let reading = session_residency(&layout, &[512, 512], Some(step(2, 1)));
        assert_eq!(
            reading.fixed().get(),
            8192 + 200 + 400 - 2 * crate::wiring::KV_ROW_POOL_PAGE_BYTES
        );
        assert_eq!(
            reading.page().get(),
            2 * crate::wiring::pool_page_bytes(512) + crate::wiring::KV_ROW_POOL_PAGE_BYTES
        );
        let not_reading = session_residency(&layout, &[512, 512], Some(step(2, 2)));
        assert_eq!(not_reading.fixed().get(), 8192 + 200 + 400);
        assert_eq!(
            not_reading.page().get(),
            2 * crate::wiring::pool_page_bytes(512)
        );
        let no_step = session_residency(&layout, &[512, 512], None);
        assert_eq!(no_step.fixed(), not_reading.fixed());
        assert_eq!(no_step.page(), not_reading.page());
    }

    /// The residency's fixed bytes hold every synthetic the door declared: the intermediate
    /// segment's extent is the synthetic allocator's high-water mark once the synthetics run past
    /// the packed intermediates, and the packed extent while none is declared.
    #[test]
    fn session_residency_charges_the_declared_synthetics() {
        let mut layout = BundleLayout::default();
        layout.segment_bytes = [8192, 200, 3072, 400, 0, 0, 0];
        layout.synth = std::cell::RefCell::new(ktir_superdsc::placement::SynthAlloc::new(8192));
        assert_eq!(
            session_residency(&layout, &[512], None).fixed().get(),
            8192 + 200 + 400
        );
        layout.synth(ktir_superdsc::place::SynthId::Act(900), &[4, 64]);
        assert_eq!(layout.segment_extents(), [8704, 200, 3072, 400, 0, 0, 0]);
        let reading = session_residency(&layout, &[512], None);
        assert_eq!(reading.fixed().get(), 8704 + 200 + 400);
        assert_eq!(reading.page().get(), crate::wiring::pool_page_bytes(512));
    }
}

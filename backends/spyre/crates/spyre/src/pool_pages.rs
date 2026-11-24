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
//!     read off the bake's own layout by [`session_residency`], never re-derived.
//!
//! [`within`] is the whole law: the pages the budget's remainder holds at the
//! per-page bytes — the pool is what the card holds. Everything else here reads a
//! pool already sized — [`Residency::at`] its bytes, [`extra_rows`] the rows past
//! the context.

use std::num::NonZeroU32;
use std::num::NonZeroU64;
use std::ops::Add;

use ktir_superdsc::placement::{BundleLayout, SegRole};
use ktir_superdsc::sdsc_abstract::SchedulerBlockTokens;

/// The context's pages and the pool's pages — each one type, beside the scheduler's block.
pub use ktir_superdsc::sdsc_abstract::{ContextPages, PoolPages};
pub use scratchy_spyre_bundle::{SegmentHold, SessionRole};

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
        self.fixed().get() + pages.get() as u64 * self.page().get()
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

impl std::iter::Sum for Residency {
    /// What every session of a world costs the card together.
    fn sum<I: Iterator<Item = Residency>>(sessions: I) -> Residency {
        sessions.fold(
            Residency {
                fixed: FixedBytes::new(0),
                page: PerPageBytes::new(0),
            },
            Add::add,
        )
    }
}

/// ⭐ THE LAW — the pages the budget's remainder holds at the per-page bytes
/// ([`blocks_within`], the serving step's own division, its sixteen-page floor included) —
/// the pool is what the card holds, never forced to the model's whole context; a context
/// longer than the pool is refused at serve time by the engine's own context law, and the
/// bake's placement checks the batch rule against it (`place_kv_pools_at`).
///
/// [`blocks_within`]: scratchy_core_config::blocks_within
pub fn within(budget: CardBudget, fixed: FixedBytes, page: PageBytes) -> PoolPages {
    let pages = scratchy_core_config::blocks_within(
        budget.get().saturating_sub(fixed.get()) as usize,
        page.get().get() as usize,
    ) as u32;
    PoolPages::of_nonzero(NonZeroU32::MIN.saturating_add(pages.saturating_sub(1)))
}

/// The pool's rows past the context's pages — the positions a request may still fill
/// before the pool is the context, at the scheduler's own page. SATURATES at zero when
/// the pool holds no more than the context, the same spelling the saturating row laws of
/// the prefix capacity take.
pub const fn extra_rows(pool: PoolPages, context: ContextPages) -> u32 {
    pool.get().saturating_sub(context.get()) * SchedulerBlockTokens::REFERENCE.get()
}

/// ⭐ THE BAKE'S POOL — the pages a world's sessions run at, read once off their summed residency.
/// On a card build (`spyre-lower`) it is [`within`] at the generation the bake targets. Off the card
/// the KV cache is the emulator's KTIR buffers, host memory no card budget sizes, so the pool is the
/// context's own pages — as it is for a residency that costs no page ([`PageBytes::of`]'s `None`).
/// The ONE statement of which pool a world places.
pub fn of_bake(residency: Residency, context: ContextPages) -> PoolPages {
    match PageBytes::of(residency.page()) {
        Some(page) if cfg!(feature = "spyre-lower") => within(
            CardBudget::of::<ktir_core::ddl_binds::Baked>(),
            residency.fixed(),
            page,
        ),
        _ => PoolPages::of_nonzero(context.nonzero()),
    }
}

/// ⭐ THE POOL THE BAKE PLACED — its pages ([`of_bake`]) and one page of the world's pools
/// ([`pools_page_bytes`]): the serving step's KV budget and the block it is cut into, carried in the
/// wiring.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BakedPool {
    pages: PoolPages,
    page: PageBytes,
}

impl BakedPool {
    pub const fn new(pages: PoolPages, page: PageBytes) -> Self {
        Self { pages, page }
    }

    pub const fn pages(self) -> PoolPages {
        self.pages
    }

    pub const fn page(self) -> PageBytes {
        self.page
    }

    /// The pool's bytes: its pages at its page.
    pub const fn bytes(self) -> u64 {
        self.pages.get() as u64 * self.page.get().get()
    }
}

/// The bytes one page of a session's pools holds — every pool's page
/// ([`crate::wiring::KvPool::page_bytes`]), summed.
pub fn pools_page_bytes(pools: &[crate::wiring::KvPool]) -> PerPageBytes {
    PerPageBytes::new(pools.iter().map(|p| p.page_bytes()).sum())
}

/// ⭐ THE SESSION'S RESIDENCY, READ OFF THE BAKE'S OWN LAYOUTS AND STEP, AT THE SESSION'S ROLE.
///
/// The fixed bytes are what the session allocates on the card ([`SessionRole::holds`] and
/// [`SegmentHold::alloc_bytes`], the card's own rule, its host page cover included). That is every
/// segment at the largest extent any layout the session launches declares
/// ([`session_segment_bytes`](crate::lower_subtile_tape_to_superdsc::session_segment_bytes), over
/// the ceiling and every sweep rung; [`BundleLayout::segment_extents`] covers every declared
/// synthetic), plus every extra weight bank.
/// - The OWNER allocates the weights and their banks, and its pool is its pages, so its KV segment
///   is charged per page.
/// - A BORROWER pays a placeholder for each resident segment and each bank it aliases. Its spilled
///   weight tail sits in its own non-resident segment, so it is its own.
///
/// The per-page bytes are the pool's page ([`pools_page_bytes`]), paid by the owner alone because
/// every borrower aliases its pool, plus this session's own row pool page
/// ([`crate::wiring::KV_ROW_POOL_PAGE_BYTES`]) when its step reads one. That row pool's bytes sit
/// inside its segment's extent and leave the fixed total to ride per page, charged one host page of
/// cover past its bytes ([`crate::bundle_code::HOST_PAGE_BYTES`]). Two sessions' residencies add
/// ([`impl Add`]) to what they cost together.
pub fn session_residency<'a>(
    role: SessionRole,
    ceiling: &BundleLayout,
    rungs: impl IntoIterator<Item = &'a BundleLayout>,
    pools: &[crate::wiring::KvPool],
    step: Option<crate::wiring::PagedStep>,
) -> Residency {
    let kv_seg = SegRole::Kv.segment();
    let extents = crate::lower_subtile_tape_to_superdsc::session_segment_bytes(
        ceiling.segment_extents(),
        rungs.into_iter().map(BundleLayout::segment_extents),
    );
    let banks = role.holds(SegRole::Weight.segment());
    let mut fixed: u64 = extents
        .into_iter()
        .enumerate()
        .map(|(seg, extent)| match (seg == kv_seg, role.holds(seg)) {
            // The pool this session allocates is its pages, charged per page below.
            (true, SegmentHold::Allocated) => 0,
            (_, hold) => hold.alloc_bytes(extent),
        })
        .sum::<u64>()
        + ceiling
            .weight_bank_bytes
            .iter()
            .map(|&bank| banks.alloc_bytes(bank))
            .sum::<u64>();
    let mut page: u64 = match role.holds(kv_seg) {
        SegmentHold::Allocated => pools_page_bytes(pools).get(),
        SegmentHold::Borrowed => 0,
    };
    if let Some(step) = step
        && step.reads_kv_row_pool()
    {
        // The row pool sits inside its segment's extent; charge its run per page instead, and its
        // region one host page past the run — the most its page cover can add as the run grows.
        fixed = fixed - step.card_kv_row_pool_bytes() + crate::bundle_code::HOST_PAGE_BYTES;
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

    fn fp16(cols: u32) -> crate::wiring::KvPool {
        crate::wiring::KvPool {
            tid: 0,
            cols,
            format: scratchy_triton_splice::KvPoolFormat::Fp16,
        }
    }

    fn fp8(cols: u32) -> crate::wiring::KvPool {
        crate::wiring::KvPool {
            tid: 0,
            cols,
            format: scratchy_triton_splice::KvPoolFormat::Fp8(scratchy_ir::KvCacheScales::Unscaled),
        }
    }

    /// A budget smaller than the context gives a smaller pool — the pool is what the card's
    /// budget holds, never forced to the context; and a budget below the fixed bytes saturates
    /// at zero remainder, where `blocks_within`'s own sixteen-page floor is the only one left.
    #[test]
    fn a_budget_smaller_than_the_context_gives_a_smaller_pool() {
        let context = ContextPages::new(nz32(100));
        // Sixteen pages fit — `blocks_within`'s own floor — and the context needs a hundred.
        let pool = within(
            CardBudget::new(16 * 1024),
            FixedBytes::new(0),
            PageBytes::new(nz64(1024)),
        );
        assert_eq!(
            pool.get(),
            scratchy_core_config::blocks_within(16 * 1024, 1024) as u32
        );
        assert_eq!(pool.get(), 16);
        assert!(pool.get() < context.get());
        // The budget below the fixed bytes: a zero remainder, the same sixteen-page floor.
        let pool = within(
            CardBudget::new(100),
            FixedBytes::new(4096),
            PageBytes::new(nz64(1024)),
        );
        assert_eq!(
            pool.get(),
            scratchy_core_config::blocks_within(0, 1024) as u32
        );
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
        let pool = within(CardBudget::new(4096 + 100 * 1024 + 512), fixed, page);
        assert_eq!(pool.get(), 100, "the 512-byte remainder is dropped");
        let residency = Residency {
            fixed,
            page: PerPageBytes::new(1024),
        };
        assert_eq!(residency.at(pool), 4096 + 100 * 1024);
        let pool = within(CardBudget::new(4096 + 250 * 1024), fixed, page);
        assert_eq!(pool.get(), 250);
    }

    /// A context's pages are its slots in whole scheduler blocks, a partial block counted whole.
    #[test]
    fn a_context_is_whole_pages() {
        let b = SchedulerBlockTokens::REFERENCE.get();
        assert_eq!(ContextPages::of_slots(nz32(1)).get(), 1);
        assert_eq!(ContextPages::of_slots(nz32(b)).get(), 1);
        assert_eq!(ContextPages::of_slots(nz32(b + 1)).get(), 2);
        assert_eq!(ContextPages::of_slots(nz32(16 * b)).get(), 16);
    }

    /// The extra rows are the pages past the context, at the scheduler's own page; a pool that
    /// holds no more than the context saturates them at zero.
    #[test]
    fn extra_rows_are_the_pages_past_the_context() {
        let pool = PoolPages::of_nonzero(nz32(100));
        let context = ContextPages::new(nz32(40));
        assert_eq!(
            extra_rows(pool, context),
            60 * SchedulerBlockTokens::REFERENCE.get()
        );
        assert_eq!(extra_rows(PoolPages::of_nonzero(nz32(7)), context), 0);
    }

    /// ⭐ BYTE-IDENTITY OF THE PAGE LAWS —
    ///
    /// * the row pool's page is the index run's int32 words, the figure the existing
    ///   `the_walk_covers_the_context_not_the_pool` test pins end-to-end through
    ///   `PagedStep::card_placements` (sixteen pages at `INDEX_RUN_WORDS × 4` each); pinning
    ///   the const here closes the chain through `card_kv_row_pool_bytes`'s page spelling.
    /// * a whole-page pool's bytes are the same counted by page or by row — `pages ×
    ///   pool_page_bytes(cols)` is the row-count spelling (`rows × row_stick_bytes`)
    ///   the layout's pool placement was sized by, so the re-route through the page law
    ///   moves no byte.
    #[test]
    fn the_page_laws_are_byte_identical_to_the_row_spelling() {
        assert_eq!(
            crate::wiring::KV_ROW_POOL_PAGE_BYTES,
            u64::from(ktir_superdsc::emit::lower_ktir_to_superdsc::INDEX_RUN_WORDS) * 4,
        );
        for format in [
            scratchy_triton_splice::KvPoolFormat::Fp16,
            scratchy_triton_splice::KvPoolFormat::Fp8(scratchy_ir::KvCacheScales::Unscaled),
        ] {
            for cols in [4u32, 64, 512, 576, 1024, 4224, 49152] {
                let pages = 16u32;
                let rows = pages * SchedulerBlockTokens::REFERENCE.get();
                assert_eq!(
                    u64::from(pages)
                        * crate::wiring::KvPool {
                            tid: 0,
                            cols,
                            format
                        }
                        .page_bytes(),
                    u64::from(rows)
                        * crate::lower_subtile_tape_to_superdsc::row_stick_bytes(cols, format.df()),
                    "cols {cols}: a whole-page pool's bytes are the same by page or by row"
                );
            }
        }
    }

    /// The owner's residency charges the KV segment per page, never as fixed bytes; an empty
    /// segment costs its placeholder; and the row pool's bytes leave the fixed total to ride per
    /// page when the step reads it.
    #[test]
    fn session_residency_charges_the_kv_segment_per_page() {
        let a = |e: u64| SegmentHold::Allocated.alloc_bytes(e);
        let run = crate::wiring::KV_ROW_POOL_PAGE_BYTES;
        let grow = crate::bundle_code::HOST_PAGE_BYTES;
        // seg0 intermediate, seg1 weights, seg2 KV, seg3 activations.
        let layout = BundleLayout {
            segment_bytes: [8192, 200, 3072, 400, 0, 0, 0],
            ..BundleLayout::default()
        };
        let reading = session_residency(
            SessionRole::Owner,
            &layout,
            [],
            &[fp16(512), fp16(512)],
            Some(step(2, 1)),
        );
        let unread = a(8192) + a(200) + a(400) + 3 * a(0);
        assert_eq!(reading.fixed().get(), unread - 2 * run + grow);
        assert_eq!(reading.page().get(), 2 * fp16(512).page_bytes() + run);
        let not_reading = session_residency(
            SessionRole::Owner,
            &layout,
            [],
            &[fp16(512), fp16(512)],
            Some(step(2, 2)),
        );
        assert_eq!(not_reading.fixed().get(), unread);
        assert_eq!(not_reading.page().get(), 2 * fp16(512).page_bytes());
        let no_step = session_residency(
            SessionRole::Owner,
            &layout,
            [],
            &[fp16(512), fp16(512)],
            None,
        );
        assert_eq!(no_step.fixed(), not_reading.fixed());
        assert_eq!(no_step.page(), not_reading.page());
    }

    /// The residency's fixed bytes hold every synthetic the door declared.
    #[test]
    fn session_residency_charges_the_declared_synthetics() {
        let a = |e: u64| SegmentHold::Allocated.alloc_bytes(e);
        let layout = BundleLayout {
            segment_bytes: [8192, 200, 3072, 400, 0, 0, 0],
            synth: std::cell::RefCell::new(ktir_superdsc::placement::SynthAlloc::new(8192)),
            ..BundleLayout::default()
        };
        assert_eq!(
            session_residency(SessionRole::Owner, &layout, [], &[fp16(512)], None)
                .fixed()
                .get(),
            a(8192) + a(200) + a(400) + 3 * a(0)
        );
        layout.synth(ktir_superdsc::place::SynthId::Act(900), &[4, 64]);
        assert_eq!(layout.segment_extents(), [8704, 200, 3072, 400, 0, 0, 0]);
        let reading = session_residency(SessionRole::Owner, &layout, [], &[fp16(512)], None);
        assert_eq!(reading.fixed().get(), a(8704) + a(200) + a(400) + 3 * a(0));
        assert_eq!(reading.page().get(), fp16(512).page_bytes());
    }

    /// The session's fixed bytes are the widest extent any layout it launches declares.
    #[test]
    fn session_residency_charges_the_widest_layout_the_session_launches() {
        let a = |e: u64| SegmentHold::Allocated.alloc_bytes(e);
        let ceiling = BundleLayout {
            segment_bytes: [8192, 200, 3072, 400, 0, 0, 0],
            ..BundleLayout::default()
        };
        let rung = BundleLayout {
            segment_bytes: [12288, 200, 3072, 300, 0, 0, 4096],
            ..BundleLayout::default()
        };
        let r = session_residency(SessionRole::Owner, &ceiling, [&rung], &[fp16(512)], None);
        assert_eq!(
            r.fixed().get(),
            a(12288) + a(200) + a(400) + a(0) + a(0) + a(4096)
        );
        assert_eq!(r.page().get(), fp16(512).page_bytes());
    }

    /// ⭐ THE RESIDENT BYTES ARE CHARGED ONCE — the owner allocates the weights (every bank) and the
    /// KV pool. Each borrower charges its own segments and a placeholder for every resident segment
    /// and bank it aliases, its spilled tail included as its own. The pool's page is paid once, by
    /// the owner; each session that reads its own row pool pays that pool's page.
    #[test]
    fn the_owner_pays_the_resident_bytes_and_every_session_its_own() {
        let a = |e: u64| SegmentHold::Allocated.alloc_bytes(e);
        let ph = SegmentHold::Borrowed.alloc_bytes(0);
        let run = crate::wiring::KV_ROW_POOL_PAGE_BYTES;
        let grow = crate::bundle_code::HOST_PAGE_BYTES;
        let pool_page = 2 * fp16(512).page_bytes();
        let (weights, bank) = (1_000_000u64, 500_000u64);
        let layout = |intermediate: u64, spill: u64| BundleLayout {
            segment_bytes: [intermediate, weights, 3072, 400, 256, 256, spill],
            weight_bank_bytes: vec![bank],
            ..BundleLayout::default()
        };
        let (owner, reader, batch) = (layout(8192, 0), layout(16384, 4096), layout(12288, 0));
        let cols = [fp16(512), fp16(512)];
        let o = session_residency(SessionRole::Owner, &owner, [], &cols, Some(step(2, 1)));
        let r = session_residency(SessionRole::Borrower, &reader, [], &cols, Some(step(2, 1)));
        let b = session_residency(SessionRole::Borrower, &batch, [], &cols, Some(step(2, 2)));
        assert_eq!(
            o.fixed().get(),
            a(8192) + a(weights) + a(400) + a(256) + a(256) + a(0) + a(bank) - 2 * run + grow
        );
        assert_eq!(o.page().get(), pool_page + run);
        assert_eq!(
            r.fixed().get(),
            a(16384) + ph + ph + a(400) + a(256) + a(256) + a(4096) + ph - 2 * run + grow
        );
        assert_eq!(r.page().get(), run);
        assert_eq!(
            b.fixed().get(),
            a(12288) + ph + ph + a(400) + a(256) + a(256) + a(0) + ph
        );
        assert_eq!(b.page().get(), 0);
        let all = o + r + b;
        assert!(
            all.fixed().get() < 2 * weights,
            "the weights and their bank are charged once"
        );
        assert_eq!(
            all.fixed().get(),
            o.fixed().get() + r.fixed().get() + b.fixed().get()
        );
        assert_eq!(
            all.page().get(),
            pool_page + 2 * run,
            "one pool page; a row-pool page per reading session"
        );
    }

    /// The bake's pool: the law's on a card, the context's off it, and the context's for a
    /// residency that costs no page.
    #[test]
    fn the_bakes_pool_is_the_laws_on_the_card_and_the_context_off_it() {
        let slots = SchedulerBlockTokens::REFERENCE.get();
        assert_eq!(ContextPages::of_slots(nz32(slots + 1)).get(), 2);
        let context = ContextPages::of_slots(nz32(16 * slots));
        let layout = BundleLayout {
            segment_bytes: [8192, 200, 3072, 400, 0, 0, 0],
            ..BundleLayout::default()
        };
        let residency = session_residency(
            SessionRole::Owner,
            &layout,
            [],
            &[fp16(512), fp16(512)],
            None,
        );
        let pool = of_bake(residency, context);
        if cfg!(feature = "spyre-lower") {
            let page = PageBytes::of(residency.page()).expect("two pools cost a page");
            assert_eq!(
                pool,
                within(
                    CardBudget::of::<ktir_core::ddl_binds::Baked>(),
                    residency.fixed(),
                    page,
                )
            );
            assert!(pool.get() > context.get());
        } else {
            assert_eq!(pool.get(), context.get());
        }
        let no_page = session_residency(
            SessionRole::Borrower,
            &layout,
            [],
            &[fp16(512), fp16(512)],
            None,
        );
        assert_eq!(of_bake(no_page, context).get(), context.get());
        let page = PageBytes::new(nz64(1024));
        assert_eq!(
            BakedPool::new(pool, page).bytes(),
            u64::from(pool.get()) * 1024
        );
    }

    /// ⭐ A FLOAT8 POOL PAYS ITS OWN FORMAT'S PAGE — the tape's format, at its whole-stick row.
    #[test]
    fn a_float8_pool_pays_its_own_page() {
        use ktir_superdsc::superdsc_opspec::Df;
        for cols in [512u32, 1024] {
            assert_eq!(
                fp8(cols).page_bytes(),
                u64::from(SchedulerBlockTokens::REFERENCE.get())
                    * crate::lower_subtile_tape_to_superdsc::row_stick_bytes(cols, Df::Fp8)
            );
            assert!(fp8(cols).page_bytes() < fp16(cols).page_bytes());
        }
        let layout = BundleLayout {
            segment_bytes: [8192, 200, 3072, 400, 0, 0, 0],
            ..BundleLayout::default()
        };
        let r = session_residency(
            SessionRole::Owner,
            &layout,
            [],
            &[fp8(1024), fp8(1024)],
            None,
        );
        assert_eq!(r.page().get(), 2 * fp8(1024).page_bytes());
    }

    /// ⭐ A POOL ROW IS ITS WHOLE STICKS, NEVER THE OCCUPANCY PAD — a rank's 256-column share of a
    /// tensor-parallel world pays its own bytes a row, where a streamed width of the same columns
    /// reserves the pad's; a width the pad leaves alone is the same either way.
    #[test]
    fn a_pool_row_pays_its_whole_sticks_not_the_occupancy_pad() {
        use ktir_superdsc::superdsc_opspec::Df;
        let slots = u64::from(SchedulerBlockTokens::REFERENCE.get());
        assert_eq!(fp16(256).page_bytes(), slots * 512);
        assert_eq!(fp8(256).page_bytes(), slots * 256);
        assert!(
            fp16(256).page_bytes()
                < slots * crate::lower_subtile_tape_to_superdsc::row_footprint_bytes(256, Df::Fp16)
        );
        assert_eq!(
            fp16(1024).page_bytes(),
            slots * crate::lower_subtile_tape_to_superdsc::row_footprint_bytes(1024, Df::Fp16)
        );
    }
}

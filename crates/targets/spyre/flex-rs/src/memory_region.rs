//! Port of `flex/include/flex/allocator/memory_types.hpp`.
//!
//! `MemoryBlock`/`MemoryRegion`/`AllocationBackingStore` are the free-list
//! bookkeeping and coalescing algorithm that the previous port attempt wrongly
//! hid behind FFI. None of this touches senlib — it is pure Rust logic over
//! an in-process free list, identical in spirit to the C++ `std::set` +
//! `std::multimap` bookkeeping in `memory_types.hpp` / the corresponding `.cpp`.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use crate::address::{ByteOffset, ByteSize, RegionId};
use crate::device_memory_allocator::DeviceMemoryAllocation;

/// NUMA / physical memory-domain identifier (orthogonal to `MemoryType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DomainId(pub u32);

/// Sequentially-assigned segment identifier used for device dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SegmentId(pub u32);

/// Port of `flex::MemoryType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryType {
    /// Tensor data — activations, weights, KV caches. Default for allocations.
    Tensor,
    /// Program segment (segment 7) — execute-only program binaries.
    Program,
}

/// Port of `operator<<(std::ostream&, MemoryType)` (`memory_types.hpp:46-56`).
/// The C++ `default:`-less switch falls through to a `"MemoryType(N)"` branch
/// reachable only via `static_cast<MemoryType>(255)` on the underlying
/// `uint8_t`; this closed Rust enum has no such numeric escape hatch, so that
/// branch has no equivalent here (same precedent as `PlacementPolicy`'s
/// `Display` impl in `domain.rs`).
impl std::fmt::Display for MemoryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tensor => write!(f, "Tensor"),
            Self::Program => write!(f, "Program"),
        }
    }
}

/// Port of `flex::MemoryBlock`: a `[start, end)` byte range within a region,
/// free or occupied. Ordered by `start` for `BTreeSet` usage, matching the
/// C++ `std::set<MemoryBlock>` with `operator<` on `start_`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBlock {
    start: ByteOffset,
    end: ByteOffset,
    is_free: bool,
}

impl MemoryBlock {
    /// Port of `flex::MemoryBlock`'s constructor (`memory_types.hpp`). The
    /// real C++ constructor performs no validation at all: `end < start`
    /// silently wraps `size()` (`end_ - start_`, unsigned subtraction) to a
    /// huge value. The `assert!` below is a DELIBERATE DEVIATION, not a port
    /// of any real check — it converts that C++ footgun into an explicit
    /// panic for a case that should never arise from valid callers, rather
    /// than reproducing the wraparound. See
    /// `memory_block_end_before_start_construction_panics` (port of
    /// `EndBeforeStartConstructionWrapsAround`) for the corresponding test.
    pub fn new(start: ByteOffset, end: ByteOffset, is_free: bool) -> Self {
        assert!(start.0 <= end.0, "MemoryBlock start must not exceed end");
        Self {
            start,
            end,
            is_free,
        }
    }

    pub fn start(&self) -> ByteOffset {
        self.start
    }
    pub fn end(&self) -> ByteOffset {
        self.end
    }
    pub fn size(&self) -> ByteSize {
        ByteSize(self.end.0 - self.start.0)
    }
    pub fn is_free(&self) -> bool {
        self.is_free
    }
}

impl PartialOrd for MemoryBlock {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MemoryBlock {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.start.0.cmp(&other.start.0)
    }
}

/// One entry per chunk backing a `CompositeAddress`. Port of
/// `AllocationBackingStore::ChunkBacking` — kept as region/block *identifiers*
/// (not raw pointers, since `MemoryRegion` lives inside `FlexAllocator`'s
/// `Vec` and blocks live inside that region's `BTreeSet`) so the allocator can
/// re-look-up both by id in O(log n) without unsafe pointer aliasing.
#[derive(Debug, Clone, Copy)]
pub struct ChunkBacking {
    pub region_id: RegionId,
    pub block_start: ByteOffset,
}

/// Port of `flex::AllocationBackingStore`.
#[derive(Debug, Clone)]
pub struct AllocationBackingStore {
    pub chunk_backings: Vec<ChunkBacking>,
}

impl AllocationBackingStore {
    pub fn single(backing: ChunkBacking) -> Self {
        Self {
            chunk_backings: vec![backing],
        }
    }

    /// Port of the multi-chunk constructor; panics on empty input like the
    /// C++ `throw std::invalid_argument`.
    pub fn multi(chunk_backings: Vec<ChunkBacking>) -> Self {
        assert!(
            !chunk_backings.is_empty(),
            "AllocationBackingStore requires at least one chunk backing"
        );
        Self { chunk_backings }
    }
}

/// Errors from `MemoryRegion::allocate_block` / `free_block`, mirroring the
/// C++ `std::invalid_argument` throw sites.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryRegionError {
    RequestExceedsBlock,
    BlockNotFree,
    BlockNotFound,
    AlreadyFree,
}

/// Text mirrors the real C++ `std::invalid_argument` throw strings in
/// `memory_types.cpp`'s `allocateBlock`/`freeBlock` as closely as a single
/// shared enum variant can: `BlockNotFound` is thrown from both call sites
/// with different C++ text ("allocateBlock: free_block not found in
/// blocks_" vs. "freeBlock: block not found in blocks_"), so it uses a
/// generic message covering either origin rather than duplicating the
/// variant per call site.
impl std::fmt::Display for MemoryRegionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RequestExceedsBlock => write!(f, "allocateBlock: nbytes exceeds block size"),
            Self::BlockNotFree => write!(f, "allocateBlock: block is not free"),
            Self::BlockNotFound => write!(f, "block not found in blocks_"),
            Self::AlreadyFree => write!(f, "freeBlock: block is already free"),
        }
    }
}
impl std::error::Error for MemoryRegionError {}

/// Port of `flex::MemoryRegion`.
///
/// The C++ version indexes free blocks via a `std::multimap<size, const
/// MemoryBlock*>` (`free_sizes_`) for O(log R) best-fit lookup, where R is the
/// number of *free* blocks only — completely independent of how many blocks
/// in the region are occupied — and a `std::set<MemoryBlock>` ordered by
/// start offset for O(log N) coalescing (`acquisition_strategy.hpp:141-158`'s
/// `BestFitStrategy::operator()` does `free_sizes.lower_bound(nbytes)`, never
/// touching occupied blocks at all).
///
/// BUG FOUND AND FIXED HERE: this previously had no equivalent of
/// `free_sizes_` at all — `find_best_fit` did a linear scan over `blocks`
/// (ALL blocks, occupied *and* free), filtering for free ones inline. A prior
/// pass's doc comment excused this as "region counts are small — tens, not
/// millions", which conflates the number of *regions* (small, fixed at
/// startup) with the number of *blocks resident in one region* — which is
/// exactly what grows unbounded under paged KV-cache decode: every live page
/// of every concurrently-batched, long-context request is an occupied block
/// that stays in `blocks` for the lifetime of that request, and every new
/// decode step's page allocation re-scans the entire set. With long context
/// (many pages per request) and batching (many concurrently-live requests)
/// together, resident block counts reach into the hundreds or thousands, and
/// every single allocation call — made while holding the process-wide
/// allocator lock (`Arc<Mutex<FlexAllocator>>` in `runtime.rs`) — pays an
/// O(total-live-blocks) scan instead of the real C++'s O(log free-blocks).
/// This is the textbook shape of "solo long-context is fine, short-context
/// batching is fine, but long-context *and* batching together hang": neither
/// dimension alone drives resident-block count high enough to matter, but the
/// product of both does, and the resulting per-allocation latency — under a
/// global lock, on the hot per-decode-step path — is exactly what could blow
/// a hardware watchdog's microsecond-scale queue-drain deadline. `free_index`
/// below restores the real O(log F) shape: a `BTreeMap<size, VecDeque<start>>`
/// over free blocks only, mirroring `free_sizes_`'s (size, block) ordering.
///
/// TIE-BREAK BUG FOUND AND FIXED HERE: an earlier version of this index was a
/// `BTreeSet<(size, start)>`, which breaks ties between equal-size free
/// blocks by ASCENDING START OFFSET. The real C++ `free_sizes_` is a
/// `std::multimap<uint64_t, const MemoryBlock*>`
/// (`memory_types.hpp:305`); for equal keys, `std::multimap::emplace`
/// preserves insertion order among elements with equivalent keys (this is
/// guaranteed by the standard), and `BestFitStrategy::operator()`
/// (`acquisition_strategy.hpp:153-157`) always returns
/// `free_sizes.lower_bound(nbytes)`, i.e. the *first* (oldest-inserted) entry
/// at that size — FIFO order, not offset order. Every insertion point in the
/// real C++ (`memory_types.cpp:35` in the constructor, `:83` for a
/// post-`allocateBlock` remainder, `:200` for a post-`freeBlock` merged
/// block) is a plain `free_sizes_.emplace(size, &block)` with no reordering.
/// Concretely: free a 1024-byte block at offset 1024, then free a *different*
/// 1024-byte block at offset 0 — C++ returns the offset-1024 entry (inserted
/// first); a start-offset-tie-broken index would wrongly return the
/// offset-0 entry instead. A `VecDeque<start>` per size bucket, appended to
/// on insert and searched-and-removed-in-place on removal (removal is by
/// value since either block can be reallocated/coalesced out of order, not
/// necessarily FIFO), reproduces the multimap's per-size FIFO order exactly.
#[derive(Debug)]
pub struct MemoryRegion {
    region_id: RegionId,
    segment_id: Option<SegmentId>,
    domain_id: DomainId,
    memory_type: MemoryType,
    data: Option<Arc<DeviceMemoryAllocation>>,
    total_size: ByteSize,
    free_size: ByteSize,
    blocks: BTreeSet<MemoryBlock>,
    /// Free-block index keyed by size, with a FIFO-ordered `VecDeque` of
    /// `start` offsets per size bucket — mirroring the real C++
    /// `free_sizes_: std::multimap<uint64_t, const MemoryBlock*>`'s
    /// (size, insertion-order) semantics exactly (see the tie-break bug
    /// writeup on `MemoryRegion` above). Every free block in `blocks` has
    /// exactly one corresponding entry here, and vice versa; kept in
    /// lock-step by `allocate_block`/`free_block` via `free_index_insert`/
    /// `free_index_remove`.
    free_index: BTreeMap<u64, VecDeque<u64>>,
}

impl MemoryRegion {
    /// Port of `MemoryRegion::MemoryRegion` (`memory_types.cpp:20-36`).
    ///
    /// NEW BUG FOUND (fixed here): `data` was previously typed as a
    /// non-optional `Arc<DeviceMemoryAllocation>`, but the real C++ field is
    /// `DeviceMemoryAllocationPtr` (`std::shared_ptr<DeviceMemoryAllocation>`),
    /// which every one of the 30+ `MemoryRegionTest`/`MemoryRegionAllocateBlockTest`/
    /// `MemoryRegionFreeBlockTest` cases in `memory_types_test.cpp` constructs
    /// as `nullptr` (`MemoryRegion(nullptr, domain_id, total_size)`) — a
    /// legal, exercised C++ state, not just an untested corner. Forcing a
    /// non-optional `Arc` here silently made that entire state unreachable in
    /// this port and, with it, made every one of those C++ tests un-portable
    /// (this file had zero unit tests before this change). Widening `data` to
    /// `Option<Arc<..>>` matches the real nullable pointer semantics exactly
    /// and does not touch the senlib boundary: production call sites
    /// (`allocator.rs::allocate_new_region`) still always pass `Some(..)`,
    /// mirroring every *production* C++ call site, which always passes a
    /// real, non-null allocation too — only the test-only null state was
    /// missing.
    pub fn new(
        region_id: RegionId,
        data: Option<Arc<DeviceMemoryAllocation>>,
        domain_id: DomainId,
        total_size: ByteSize,
        memory_type: MemoryType,
    ) -> Self {
        let mut blocks = BTreeSet::new();
        blocks.insert(MemoryBlock::new(
            ByteOffset(0),
            ByteOffset(total_size.0),
            true,
        ));
        // Port of `memory_types.cpp:35`: `free_sizes_.emplace(total_size, &(*it))`.
        let mut free_index: BTreeMap<u64, VecDeque<u64>> = BTreeMap::new();
        free_index.entry(total_size.0).or_default().push_back(0);
        Self {
            region_id,
            segment_id: None,
            domain_id,
            memory_type,
            data,
            total_size,
            free_size: total_size,
            blocks,
            free_index,
        }
    }

    pub fn region_id(&self) -> RegionId {
        self.region_id
    }
    pub fn segment_id(&self) -> Option<SegmentId> {
        self.segment_id
    }
    pub fn set_segment_id(&mut self, segment_id: SegmentId) {
        self.segment_id = Some(segment_id);
    }
    pub fn domain_id(&self) -> DomainId {
        self.domain_id
    }
    pub fn memory_type(&self) -> MemoryType {
        self.memory_type
    }
    pub fn data(&self) -> Option<&Arc<DeviceMemoryAllocation>> {
        self.data.as_ref()
    }
    pub fn total_size(&self) -> ByteSize {
        self.total_size
    }
    pub fn free_size(&self) -> ByteSize {
        self.free_size
    }
    pub fn blocks(&self) -> &BTreeSet<MemoryBlock> {
        &self.blocks
    }

    /// Best-fit search over free blocks: smallest free block that is still
    /// `>= nbytes`. Port of `BestFitStrategy::operator()`
    /// (`acquisition_strategy.hpp:141-158`): `free_sizes.lower_bound(nbytes)`
    /// — O(log F) over the free-block index (F = number of free blocks),
    /// never touching occupied blocks. See the `free_index` field doc comment
    /// above for why this must not be a scan over all of `blocks`.
    pub fn find_best_fit(&self, nbytes: ByteSize) -> Option<MemoryBlock> {
        // Port of `free_sizes.lower_bound(nbytes)`: the first size bucket
        // `>= nbytes`, then the oldest-inserted (`front()`) start within it —
        // FIFO tie-break, matching `std::multimap`'s insertion-order-among-
        // equal-keys guarantee (see `MemoryRegion`'s doc comment above).
        let (&size, starts) = self.free_index.range(nbytes.0..).next()?;
        let &start = starts.front()?;
        let block = self.blocks.get(&MemoryBlock::new(
            ByteOffset(start),
            ByteOffset(start),
            true,
        ))?;
        debug_assert_eq!(block.size().0, size);
        debug_assert!(block.is_free());
        Some(*block)
    }

    /// Inserts a free block's `(size, start)` into `free_index`, appending to
    /// the FIFO order for that size bucket. Port of `free_sizes_.emplace(size,
    /// &block)`.
    fn free_index_insert(&mut self, size: u64, start: u64) {
        self.free_index.entry(size).or_default().push_back(start);
    }

    /// Removes a specific `(size, start)` entry from `free_index`. Port of
    /// `free_sizes_.erase(free_sizes_it)` — a value-based removal since this
    /// port holds no multimap iterator, but every free block still has
    /// exactly one entry keyed by its own `(size, start)`.
    fn free_index_remove(&mut self, size: u64, start: u64) {
        if let Some(starts) = self.free_index.get_mut(&size) {
            if let Some(pos) = starts.iter().position(|&s| s == start) {
                starts.remove(pos);
            }
            if starts.is_empty() {
                self.free_index.remove(&size);
            }
        }
    }

    /// Port of `MemoryRegion::allocateBlock`. Splits `block` (must be free and
    /// `>= nbytes`) into an occupied block of `nbytes` and, if there is a
    /// remainder, a new free block for the rest. Returns the occupied block.
    ///
    /// NEW BUG FOUND (fixed here): this previously did `self.blocks.remove(&block)`
    /// directly to both look up and erase in one step. `BTreeSet<MemoryBlock>`'s
    /// `Ord` — deliberately start-offset-only, matching the real C++
    /// `std::set<MemoryBlock>`'s `operator<` (`memory_types.hpp:103`) — means
    /// `remove`/`get` treat any two blocks with the same `start` as
    /// equivalent, regardless of `end`/`is_free`. So a caller holding a
    /// stale `MemoryBlock` value (e.g. a `find_best_fit` snapshot taken
    /// before some other allocation split/coalesced that same start offset)
    /// would silently match and erase *whatever block currently sits at that
    /// start* — not the block the caller thinks it has — corrupting the free
    /// list instead of failing. The real C++ guards against exactly this:
    /// `allocateBlock` re-finds the block via `blocks_.find(*free_block)`
    /// before erasing, and `freeBlock`'s own comment is explicit about why
    /// ("Scanning by address is the only safe check... If the block was
    /// already freed its node was erased... any read through it would be
    /// undefined behaviour"). This port has no pointer/iterator to scan by
    /// identity, but the equivalent safety net is a full-value check: the
    /// entry currently at `block.start` must still equal `block` in every
    /// field, or this is a stale reference.
    pub fn allocate_block(
        &mut self,
        block: MemoryBlock,
        nbytes: ByteSize,
    ) -> Result<MemoryBlock, MemoryRegionError> {
        if !block.is_free {
            return Err(MemoryRegionError::BlockNotFree);
        }
        if nbytes.0 > block.size().0 {
            return Err(MemoryRegionError::RequestExceedsBlock);
        }
        match self.blocks.get(&block) {
            Some(current) if *current == block => {}
            _ => return Err(MemoryRegionError::BlockNotFound),
        }
        self.blocks.remove(&block);
        // Keep `free_index` in lock-step with `blocks`: the block being
        // consumed here must have a matching free_index entry (it was free,
        // and every free block has exactly one entry keyed by its own
        // (size, start) — see the `free_index` field doc comment).
        self.free_index_remove(block.size().0, block.start.0);

        let occupied_end = ByteOffset(block.start.0 + nbytes.0);
        let occupied = MemoryBlock::new(block.start, occupied_end, false);
        self.blocks.insert(occupied);

        if occupied_end.0 < block.end.0 {
            let remainder = MemoryBlock::new(occupied_end, block.end, true);
            self.blocks.insert(remainder);
            self.free_index_insert(remainder.size().0, remainder.start.0);
        }

        self.free_size = ByteSize(self.free_size.0 - nbytes.0);
        Ok(occupied)
    }

    /// Port of `MemoryRegion::freeBlock`: marks `block` free and coalesces
    /// with an adjacent predecessor/successor free block, if any.
    ///
    /// NEW BUG FOUND (fixed here): same class of bug as `allocate_block`
    /// above (see its doc comment) — this previously trusted the caller's
    /// `block.is_free` field and did a plain `self.blocks.remove(&block)`,
    /// so a stale `MemoryBlock` value (e.g. one already freed and since
    /// coalesced/reallocated at the same `start`) would silently match and
    /// remove *whatever's currently there* instead of being rejected —
    /// exactly the unsound-double-free scenario the real C++ `freeBlock`
    /// explicitly guards against by identity-scanning `blocks_` before
    /// trusting the pointer at all. Ordering matters here and mirrors the
    /// C++ source (`memory_types.cpp:98-117`): existence/currency is checked
    /// *before* the already-free check, since C++ checks `block_it->is_free()`
    /// (the block's *current*, re-found state) rather than the caller's
    /// (possibly stale) `block->is_free()`.
    pub fn free_block(&mut self, block: MemoryBlock) -> Result<(), MemoryRegionError> {
        match self.blocks.get(&block) {
            Some(current) if *current == block => {}
            _ => return Err(MemoryRegionError::BlockNotFound),
        }
        if block.is_free {
            return Err(MemoryRegionError::AlreadyFree);
        }
        self.blocks.remove(&block);

        let mut new_start = block.start;
        let mut new_end = block.end;

        // Coalesce with predecessor (largest block whose start < new_start).
        if let Some(&pred) = self
            .blocks
            .range(..MemoryBlock::new(new_start, new_start, true))
            .next_back()
            && pred.is_free
            && pred.end.0 == new_start.0
        {
            self.blocks.remove(&pred);
            self.free_index_remove(pred.size().0, pred.start.0);
            new_start = pred.start;
        }

        // Coalesce with successor (smallest block whose start >= new_end).
        if let Some(&succ) = self
            .blocks
            .range(MemoryBlock::new(new_end, new_end, true)..)
            .next()
            && succ.is_free
            && succ.start.0 == new_end.0
        {
            self.blocks.remove(&succ);
            self.free_index_remove(succ.size().0, succ.start.0);
            new_end = succ.end;
        }

        let merged = MemoryBlock::new(new_start, new_end, true);
        self.blocks.insert(merged);
        self.free_index_insert(merged.size().0, merged.start.0);
        self.free_size = ByteSize(self.free_size.0 + block.size().0);
        Ok(())
    }
}

/// Port of `flex/tests/allocator/data_structures/memory_types_test.cpp`.
///
/// None of `MemoryBlockTest`/`MemoryRegionTest`/`MemoryRegionAllocateBlockTest`/
/// `MemoryRegionFreeBlockTest`/`AllocationBackingStoreTest`/`MemoryTypeTest`
/// needs a real `DeviceMemoryAllocator` — every C++ case constructs
/// `MemoryRegion(nullptr, domain_id, total_size)` directly, which is why
/// widening `MemoryRegion::data` to `Option<Arc<..>>` above (see that
/// constructor's doc comment) was the one production fix needed to port this
/// whole file: this module had zero tests before that change.
///
/// Two groups of C++ cases are NOT ported 1:1 (see notes at each spot):
/// - The `AllocationBackingStoreTest` cases that assert on raw
///   `MemoryRegion*`/`const MemoryBlock*` pointer identity
///   (`SingleChunkConstructionsNullRegionAndBlockPointers`,
///   `MultiChunkConstructionPreservesNullPointers`, parts of
///   `ChunkOrderingPreserved`) have no equivalent: this port's
///   `AllocationBackingStore::ChunkBacking` deliberately stores
///   `RegionId`/`ByteOffset` *identifiers*, not raw pointers (see that type's
///   own doc comment in this file) — there is no null-pointer state to
///   reproduce, and no pointer-identity to assert on. The ordering-preserved
///   invariant itself (not sorted by size/domain) is still ported using
///   identifiers instead of pointers.
/// - `MemoryRegionAllocateBlockTest.NotFreeBlockThrows` (C++: pass the
///   `free_sizes().end()` iterator) and `.WrappedNegativeLikeSizeThrows`
///   (C++: `size_t(-1)` wraparound) are C++-multimap-iterator/size_t-specific;
///   this port's `allocate_block` takes a `MemoryBlock` *value* rather than an
///   iterator, so the equivalent failure mode is "the block is no longer in
///   the free set" (`BlockNotFound`), and there is no `u64` wraparound
///   equivalent to `size_t(-1)` construction — both are ported below using
///   the closest matching value-based invariant instead.
#[cfg(test)]
mod ported_cxx_tests {
    use super::*;

    fn region_none(domain_id: u32, total_size: u64) -> MemoryRegion {
        MemoryRegion::new(
            RegionId(0),
            None,
            DomainId(domain_id),
            ByteSize(total_size),
            MemoryType::Tensor,
        )
    }

    // ---- MemoryBlock ----

    #[test]
    fn memory_block_construct_free_block() {
        let block = MemoryBlock::new(ByteOffset(0), ByteOffset(256), true);
        assert_eq!(block.start(), ByteOffset(0));
        assert_eq!(block.end(), ByteOffset(256));
        assert!(block.is_free());
    }

    #[test]
    fn memory_block_construct_used_block() {
        let block = MemoryBlock::new(ByteOffset(128), ByteOffset(384), false);
        assert_eq!(block.start(), ByteOffset(128));
        assert_eq!(block.end(), ByteOffset(384));
        assert!(!block.is_free());
    }

    #[test]
    fn memory_block_size_returns_end_minus_start() {
        let block = MemoryBlock::new(ByteOffset(128), ByteOffset(512), true);
        assert_eq!(block.size(), ByteSize(384));
    }

    #[test]
    fn memory_block_size_zero() {
        let block = MemoryBlock::new(ByteOffset(256), ByteOffset(256), true);
        assert_eq!(block.size(), ByteSize(0));
    }

    #[test]
    fn memory_block_size_one_alignment_unit() {
        let block = MemoryBlock::new(ByteOffset(0), ByteOffset(128), true);
        assert_eq!(block.size(), ByteSize(128));
    }

    #[test]
    fn memory_block_operator_less_compares_by_start() {
        let a = MemoryBlock::new(ByteOffset(0), ByteOffset(128), true);
        let b = MemoryBlock::new(ByteOffset(128), ByteOffset(256), false);
        assert!(a < b);
        assert!(b >= a);
    }

    #[test]
    fn memory_block_equality_all_fields_match() {
        let a = MemoryBlock::new(ByteOffset(0), ByteOffset(128), true);
        let b = MemoryBlock::new(ByteOffset(0), ByteOffset(128), true);
        assert_eq!(a, b);
    }

    #[test]
    fn memory_block_inequality_different_end() {
        let a = MemoryBlock::new(ByteOffset(0), ByteOffset(128), true);
        let b = MemoryBlock::new(ByteOffset(0), ByteOffset(256), true);
        assert_ne!(a, b);
    }

    #[test]
    fn memory_block_inequality_different_free_flag() {
        let a = MemoryBlock::new(ByteOffset(0), ByteOffset(128), true);
        let b = MemoryBlock::new(ByteOffset(0), ByteOffset(128), false);
        assert_ne!(a, b);
    }

    #[test]
    fn memory_block_inequality_different_start() {
        let a = MemoryBlock::new(ByteOffset(0), ByteOffset(128), true);
        let b = MemoryBlock::new(ByteOffset(1), ByteOffset(128), true);
        assert_ne!(a, b);
    }

    #[test]
    fn memory_block_set_orders_by_start() {
        let mut blocks = BTreeSet::new();
        blocks.insert(MemoryBlock::new(ByteOffset(256), ByteOffset(512), true));
        blocks.insert(MemoryBlock::new(ByteOffset(0), ByteOffset(128), false));
        blocks.insert(MemoryBlock::new(ByteOffset(128), ByteOffset(256), true));

        let starts: Vec<u64> = blocks.iter().map(|b| b.start().0).collect();
        assert_eq!(starts, vec![0, 128, 256]);
    }

    #[test]
    fn memory_block_end_before_start_construction_panics() {
        // Port of `EndBeforeStartConstructionWrapsAround`: the real C++
        // `MemoryBlock(uint64_t start, uint64_t end, bool)` has no
        // constructor-time validation at all, so `end < start` silently
        // wraps `size()` (`end_ - start_`) to a huge unsigned value. This
        // port's `MemoryBlock::new` (deliberately, per its own doc comment)
        // asserts `start <= end` instead of reproducing that wraparound —
        // the C++ behavior is an accepted footgun, not an invariant worth
        // porting bit-for-bit.
        let result =
            std::panic::catch_unwind(|| MemoryBlock::new(ByteOffset(512), ByteOffset(256), true));
        assert!(result.is_err());
    }

    #[test]
    fn memory_region_find_best_fit_ties_break_by_fifo_insertion_order_not_start_offset() {
        // Regression test for the free_index tie-break bug: free a block at
        // the HIGHER offset first, then one at the LOWER offset, both the
        // same size. The real C++ `free_sizes_` multimap returns the
        // first-inserted (offset-1024) entry via `lower_bound`
        // (`acquisition_strategy.hpp:153-157`); an ascending-start-offset
        // tie-break would wrongly return the offset-0 entry instead.
        let mut region = region_none(0, 2048);
        let c1 = region.find_best_fit(ByteSize(1024)).unwrap();
        let alloc_low = region.allocate_block(c1, ByteSize(1024)).unwrap();
        let c2 = region.find_best_fit(ByteSize(1024)).unwrap();
        let alloc_high = region.allocate_block(c2, ByteSize(1024)).unwrap();
        assert_eq!(alloc_low.start(), ByteOffset(0));
        assert_eq!(alloc_high.start(), ByteOffset(1024));

        // Free higher offset first, then lower offset — both size 1024.
        region.free_block(alloc_high).unwrap();
        region.free_block(alloc_low).unwrap();

        // FIFO order means the offset-1024 free block (freed first) must be
        // returned, not the offset-0 one (freed second), despite offset-0
        // sorting first by start.
        let best = region.find_best_fit(ByteSize(1024)).unwrap();
        assert_eq!(best.start(), ByteOffset(1024));
    }

    // ---- MemoryRegion: construction ----

    #[test]
    fn memory_region_constructor_sets_domain_id() {
        let region = region_none(42, 1024);
        assert_eq!(region.domain_id(), DomainId(42));
    }

    #[test]
    fn memory_region_constructor_sets_total_size() {
        let region = region_none(0, 4096);
        assert_eq!(region.total_size(), ByteSize(4096));
    }

    #[test]
    fn memory_region_initial_free_size_equals_total_size() {
        let region = region_none(0, 2048);
        assert_eq!(region.free_size(), region.total_size());
    }

    #[test]
    fn memory_region_single_free_block_spans_full_range() {
        let region = region_none(0, 512);
        let blocks = region.blocks();
        assert_eq!(blocks.len(), 1);
        let block = blocks.iter().next().unwrap();
        assert_eq!(block.start(), ByteOffset(0));
        assert_eq!(block.end(), ByteOffset(512));
        assert!(block.is_free());
    }

    #[test]
    fn memory_region_data_accessor_returns_stored_option() {
        let region = region_none(0, 1024);
        assert!(region.data().is_none());
    }

    #[test]
    fn memory_region_zero_sized_region_starts_with_single_zero_sized_free_block() {
        let region = region_none(9, 0);
        assert_eq!(region.total_size(), ByteSize(0));
        assert_eq!(region.free_size(), ByteSize(0));
        assert_eq!(region.blocks().len(), 1);

        let block = region.blocks().iter().next().unwrap();
        assert_eq!(block.start(), ByteOffset(0));
        assert_eq!(block.end(), ByteOffset(0));
        assert_eq!(block.size(), ByteSize(0));
        assert!(block.is_free());
    }

    #[test]
    fn memory_region_null_data_sets_region_id_to_zero() {
        // Port of `NullDataSetsRegionIdToZero`. In the real C++, `region_id_`
        // is derived from `data_` *inside* the constructor and defaults to
        // its `{0}` member initializer when `data_` is null. This port moved
        // that derivation to the caller (`allocator.rs::allocate_new_region`
        // computes `RegionId(allocation.device_address_bytes())` and passes
        // it in), so the equivalent invariant is simply that `region_id` is
        // whatever the caller passed — `RegionId(0)` for a null-data region,
        // matching the C++ default.
        let region = region_none(0, 1024);
        assert_eq!(region.region_id(), RegionId(0));
    }

    // ---- MemoryRegion::allocate_block ----

    #[test]
    fn memory_region_allocate_block_exact_fit_no_split() {
        let mut region = region_none(0, 1024);
        let candidate = region.find_best_fit(ByteSize(1024)).unwrap();
        let allocated = region.allocate_block(candidate, ByteSize(1024)).unwrap();

        assert_eq!(allocated.start(), ByteOffset(0));
        assert_eq!(allocated.end(), ByteOffset(1024));
        assert!(!allocated.is_free());

        assert_eq!(region.blocks().len(), 1);
        assert_eq!(region.free_size(), ByteSize(0));
        assert!(region.find_best_fit(ByteSize(1)).is_none());
    }

    #[test]
    fn memory_region_allocate_block_split_with_remainder() {
        let mut region = region_none(0, 2048);
        let candidate = region.find_best_fit(ByteSize(512)).unwrap();
        let allocated = region.allocate_block(candidate, ByteSize(512)).unwrap();

        assert_eq!(allocated.size(), ByteSize(512));
        assert!(!allocated.is_free());
        assert_eq!(region.blocks().len(), 2);

        let remainder = *region.blocks().iter().nth(1).unwrap();
        assert_eq!(remainder.start(), ByteOffset(512));
        assert_eq!(remainder.end(), ByteOffset(2048));
        assert_eq!(remainder.size(), ByteSize(1536));
        assert!(remainder.is_free());

        assert_eq!(region.free_size(), ByteSize(1536));
    }

    #[test]
    fn memory_region_allocate_block_stale_block_not_found() {
        // Port of `NotFreeBlockThrows` (see module doc: no `end()`-iterator
        // equivalent in this value-based API — the closest matching failure
        // is re-presenting a free-list snapshot that's no longer current).
        let mut region = region_none(0, 1024);
        let candidate = region.find_best_fit(ByteSize(512)).unwrap();
        region.allocate_block(candidate, ByteSize(512)).unwrap();

        // `candidate` (the original full-region free block) has been split
        // and removed from the free set; re-presenting it is a stale
        // reference, so allocate_block can't find it.
        assert_eq!(
            region.allocate_block(candidate, ByteSize(256)),
            Err(MemoryRegionError::BlockNotFound)
        );
    }

    #[test]
    fn memory_region_allocate_block_nbytes_exceeds_block_size_throws() {
        let mut region = region_none(0, 1024);
        let candidate = region.find_best_fit(ByteSize(1024)).unwrap();
        assert_eq!(
            region.allocate_block(candidate, ByteSize(2048)),
            Err(MemoryRegionError::RequestExceedsBlock)
        );
    }

    #[test]
    fn memory_region_allocate_block_extreme_size_throws() {
        // Port of `WrappedNegativeLikeSizeThrows`'s intent (see module doc:
        // no `size_t(-1)`-wraparound equivalent in `ByteSize(u64)`) — the
        // invariant that actually matters, an out-of-range request against a
        // small block, is exercised directly with `u64::MAX`.
        let mut region = region_none(0, 1024);
        let candidate = region.find_best_fit(ByteSize(1024)).unwrap();
        assert_eq!(
            region.allocate_block(candidate, ByteSize(u64::MAX)),
            Err(MemoryRegionError::RequestExceedsBlock)
        );
    }

    #[test]
    fn memory_region_allocate_block_not_free_throws() {
        let mut region = region_none(0, 1024);
        let candidate = region.find_best_fit(ByteSize(512)).unwrap();
        let allocated = region.allocate_block(candidate, ByteSize(512)).unwrap();
        // `allocated` is now occupied; trying to allocate from it again must fail.
        assert_eq!(
            region.allocate_block(allocated, ByteSize(256)),
            Err(MemoryRegionError::BlockNotFree)
        );
    }

    // ---- MemoryRegion::free_block ----

    #[test]
    fn memory_region_free_block_no_free_predecessor_or_successor() {
        let mut region = region_none(0, 3072);
        let c1 = region.find_best_fit(ByteSize(1024)).unwrap();
        region.allocate_block(c1, ByteSize(1024)).unwrap();
        let c2 = region.find_best_fit(ByteSize(1024)).unwrap();
        let alloc2 = region.allocate_block(c2, ByteSize(1024)).unwrap();
        let c3 = region.find_best_fit(ByteSize(1024)).unwrap();
        region.allocate_block(c3, ByteSize(1024)).unwrap();

        region.free_block(alloc2).unwrap();

        assert_eq!(region.blocks().len(), 3);
        let freed = *region.blocks().iter().nth(1).unwrap();
        assert_eq!(freed.start(), ByteOffset(1024));
        assert_eq!(freed.end(), ByteOffset(2048));
        assert!(freed.is_free());
        assert_eq!(region.free_size(), ByteSize(1024));
    }

    #[test]
    fn memory_region_free_block_free_predecessor_coalesces() {
        let mut region = region_none(0, 3072);
        let c1 = region.find_best_fit(ByteSize(1024)).unwrap();
        let alloc1 = region.allocate_block(c1, ByteSize(1024)).unwrap();
        let c2 = region.find_best_fit(ByteSize(1024)).unwrap();
        let alloc2 = region.allocate_block(c2, ByteSize(1024)).unwrap();
        let c3 = region.find_best_fit(ByteSize(1024)).unwrap();
        region.allocate_block(c3, ByteSize(1024)).unwrap();

        region.free_block(alloc1).unwrap();
        region.free_block(alloc2).unwrap();

        assert_eq!(region.blocks().len(), 2);
        let coalesced = *region.blocks().iter().next().unwrap();
        assert_eq!(coalesced.start(), ByteOffset(0));
        assert_eq!(coalesced.end(), ByteOffset(2048));
        assert!(coalesced.is_free());
        assert_eq!(region.free_size(), ByteSize(2048));
    }

    #[test]
    fn memory_region_free_block_free_successor_coalesces() {
        let mut region = region_none(0, 3072);
        let c1 = region.find_best_fit(ByteSize(1024)).unwrap();
        region.allocate_block(c1, ByteSize(1024)).unwrap();
        let c2 = region.find_best_fit(ByteSize(1024)).unwrap();
        let alloc2 = region.allocate_block(c2, ByteSize(1024)).unwrap();
        let c3 = region.find_best_fit(ByteSize(1024)).unwrap();
        let alloc3 = region.allocate_block(c3, ByteSize(1024)).unwrap();

        region.free_block(alloc3).unwrap();
        region.free_block(alloc2).unwrap();

        assert_eq!(region.blocks().len(), 2);
        let coalesced = *region.blocks().iter().nth(1).unwrap();
        assert_eq!(coalesced.start(), ByteOffset(1024));
        assert_eq!(coalesced.end(), ByteOffset(3072));
        assert!(coalesced.is_free());
        assert_eq!(region.free_size(), ByteSize(2048));
    }

    #[test]
    fn memory_region_free_block_predecessor_and_successor_coalesce() {
        let mut region = region_none(0, 4096);
        let c1 = region.find_best_fit(ByteSize(1024)).unwrap();
        let alloc1 = region.allocate_block(c1, ByteSize(1024)).unwrap();
        let c2 = region.find_best_fit(ByteSize(1024)).unwrap();
        let alloc2 = region.allocate_block(c2, ByteSize(1024)).unwrap();
        let c3 = region.find_best_fit(ByteSize(1024)).unwrap();
        let alloc3 = region.allocate_block(c3, ByteSize(1024)).unwrap();
        let c4 = region.find_best_fit(ByteSize(1024)).unwrap();
        region.allocate_block(c4, ByteSize(1024)).unwrap();

        region.free_block(alloc1).unwrap();
        region.free_block(alloc3).unwrap();
        region.free_block(alloc2).unwrap();

        assert_eq!(region.blocks().len(), 2);
        let coalesced = *region.blocks().iter().next().unwrap();
        assert_eq!(coalesced.start(), ByteOffset(0));
        assert_eq!(coalesced.end(), ByteOffset(3072));
        assert!(coalesced.is_free());
        assert_eq!(region.free_size(), ByteSize(3072));
    }

    #[test]
    fn memory_region_free_block_already_free_throws() {
        let mut region = region_none(0, 1024);
        let initial_free_block = *region.blocks().iter().next().unwrap();
        assert_eq!(
            region.free_block(initial_free_block),
            Err(MemoryRegionError::AlreadyFree)
        );
    }

    #[test]
    fn memory_region_free_block_double_free_throws() {
        let mut region = region_none(0, 1024);
        let candidate = region.find_best_fit(ByteSize(512)).unwrap();
        let allocated = region.allocate_block(candidate, ByteSize(512)).unwrap();
        region.free_block(allocated).unwrap();
        assert_eq!(
            region.free_block(allocated),
            Err(MemoryRegionError::BlockNotFound)
        );
    }

    // Port of `MemoryRegionFreeBlockTest.NullBlockThrows`: the real C++
    // `freeBlock` takes a raw `const MemoryBlock*` and explicitly rejects
    // `nullptr`. This port's `free_block` takes an owned `MemoryBlock` value
    // (there is no pointer, let alone a null one, to pass) — the invariant
    // this test protects ("can't free a block that doesn't exist") is
    // already covered by `memory_region_free_block_double_free_throws`
    // and `_already_free_throws` above, so it is not ported separately.

    // ---- AllocationBackingStore ----

    #[test]
    fn allocation_backing_store_single_chunk_construction() {
        let backing = ChunkBacking {
            region_id: RegionId(7),
            block_start: ByteOffset(0),
        };
        let store = AllocationBackingStore::single(backing);
        assert_eq!(store.chunk_backings.len(), 1);
        assert_eq!(store.chunk_backings[0].region_id, RegionId(7));
        assert_eq!(store.chunk_backings[0].block_start, ByteOffset(0));
    }

    #[test]
    fn allocation_backing_store_multi_chunk_construction() {
        let backings = vec![
            ChunkBacking {
                region_id: RegionId(0),
                block_start: ByteOffset(0),
            },
            ChunkBacking {
                region_id: RegionId(1),
                block_start: ByteOffset(1024),
            },
            ChunkBacking {
                region_id: RegionId(2),
                block_start: ByteOffset(2048),
            },
        ];
        let store = AllocationBackingStore::multi(backings);
        assert_eq!(store.chunk_backings.len(), 3);
        assert_eq!(store.chunk_backings[1].region_id, RegionId(1));
        assert_eq!(store.chunk_backings[2].block_start, ByteOffset(2048));
    }

    #[test]
    fn allocation_backing_store_chunk_ordering_preserved() {
        // Port of `ChunkOrderingPreserved`'s intent, adapted to this port's
        // identifier-based `ChunkBacking` (see module doc): input order must
        // survive construction untouched, not get re-sorted by region_id or
        // offset.
        let backings = vec![
            ChunkBacking {
                region_id: RegionId(3),
                block_start: ByteOffset(4096),
            },
            ChunkBacking {
                region_id: RegionId(0),
                block_start: ByteOffset(512),
            },
            ChunkBacking {
                region_id: RegionId(2),
                block_start: ByteOffset(2048),
            },
            ChunkBacking {
                region_id: RegionId(1),
                block_start: ByteOffset(1024),
            },
        ];
        let store = AllocationBackingStore::multi(backings);
        assert_eq!(store.chunk_backings.len(), 4);
        assert_eq!(store.chunk_backings[0].region_id, RegionId(3));
        assert_eq!(store.chunk_backings[1].region_id, RegionId(0));
        assert_eq!(store.chunk_backings[2].region_id, RegionId(2));
        assert_eq!(store.chunk_backings[3].region_id, RegionId(1));
    }

    #[test]
    fn allocation_backing_store_empty_backing_store_throws() {
        let result = std::panic::catch_unwind(|| AllocationBackingStore::multi(vec![]));
        assert!(result.is_err());
    }

    #[test]
    fn allocation_backing_store_chunk_backing_construction() {
        let backing = ChunkBacking {
            region_id: RegionId(0),
            block_start: ByteOffset(0),
        };
        assert_eq!(backing.region_id, RegionId(0));
        assert_eq!(backing.block_start, ByteOffset(0));
    }

    // ---- MemoryType stream/Display ----

    #[test]
    fn memory_type_display_tensor() {
        assert_eq!(MemoryType::Tensor.to_string(), "Tensor");
    }

    #[test]
    fn memory_type_display_program() {
        assert_eq!(MemoryType::Program.to_string(), "Program");
    }

    // `MemoryTypeTest.StreamOperatorUnknownValue` is NOT ported: it casts an
    // out-of-range `uint8_t` to `MemoryType`, which is legal for a C++ `enum
    // class` but has no equivalent for this closed Rust enum (same precedent
    // as `PlacementPolicy`'s analogous case in `domain.rs`).
}

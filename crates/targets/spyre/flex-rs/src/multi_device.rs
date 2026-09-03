//! Native port of `flex/multi_device`: RDMA/HDMA peer-to-peer data exchange,
//! barrier/broadcast rendezvous, and bitmask/bitset bookkeeping helpers.
//!
//! Source: flex-cxx/flex/include/flex/multi_device/**, flex-cxx/flex/src/multi_device/**.
//! See SENLIB_BOUNDARY_multi-device.md for the FFI boundary rationale.

use std::collections::{HashSet, VecDeque};
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::senlib_ffi_multi_device as ffi;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// A rank within a collective/multi-device job.
/// Source: multi_device/topology.hpp:23 `using rank_t = unsigned;`
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RankId(pub u32);

/// Byte offset into an HDMA-managed buffer.
/// Reuses the domain-meaning of `flex::address::ByteOffset`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct ChunkOffset(pub u64);

/// Byte size of an HDMA-managed range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct ChunkSize(pub u64);

/// Index of a queued send/receive message, matching `flex::midx_t`.
/// Source: multi_device/topology.hpp:38 `using midx_t = int;`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MessageIndex(pub i32);

/// Source/target segment id used by RDMA write-done signaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SegId(pub u8);

const MAX_RANK_BITS: u32 = 64;

/// Error raised when a rank index exceeds the 64-bit bitmask capacity.
/// Source: bitmask_helper.cpp:26-29 `RAS::UNCLASSIFIED::RankExceedsMax()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankExceedsMax {
    pub rank: u64,
    pub max_supported: u32,
}

// ---------------------------------------------------------------------------
// BitmaskHelper — rank bitmask for collectives
// Source: hdma/bitmask_helper.{hpp,cpp}
// ---------------------------------------------------------------------------

/// Manages a 64-bit bitmask of participating ranks in a collective operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BitmaskHelper {
    bitmask: u64,
}

impl BitmaskHelper {
    pub const fn new() -> Self {
        Self { bitmask: 0 }
    }

    pub const fn from_bitmask(bitmask: u64) -> Self {
        Self { bitmask }
    }

    pub fn from_ranks(ranks: impl IntoIterator<Item = RankId>) -> Result<Self, RankExceedsMax> {
        let mut bitmask = 0u64;
        for RankId(rank) in ranks {
            if u64::from(rank) >= u64::from(MAX_RANK_BITS) {
                return Err(RankExceedsMax {
                    rank: u64::from(rank),
                    max_supported: MAX_RANK_BITS,
                });
            }
            bitmask |= 1u64 << rank;
        }
        Ok(Self { bitmask })
    }

    pub const fn bitmask(&self) -> u64 {
        self.bitmask
    }

    /// Returns and clears the lowest set rank, or `RankId(64)` if none are set.
    /// Source: bitmask_helper.cpp:43-51 `BitmaskHelper::NextRank`.
    pub fn next_rank(&mut self) -> RankId {
        let rank = self.bitmask.trailing_zeros();
        if rank < MAX_RANK_BITS {
            // ClearRank without the redundant bounds check: rank is already valid.
            self.bitmask ^= 1u64 << rank;
        }
        RankId(rank)
    }

    fn check_rank(rank: u64) -> Result<u32, RankExceedsMax> {
        if rank >= u64::from(MAX_RANK_BITS) {
            Err(RankExceedsMax {
                rank,
                max_supported: MAX_RANK_BITS,
            })
        } else {
            Ok(rank as u32)
        }
    }

    pub fn is_rank_set(&self, rank: RankId) -> Result<bool, RankExceedsMax> {
        let r = Self::check_rank(u64::from(rank.0))?;
        Ok((self.bitmask & (1u64 << r)) != 0)
    }

    pub fn set_rank(&mut self, rank: RankId) -> Result<(), RankExceedsMax> {
        let r = Self::check_rank(u64::from(rank.0))?;
        self.bitmask |= 1u64 << r;
        Ok(())
    }

    pub fn clear_rank(&mut self, rank: RankId) -> Result<(), RankExceedsMax> {
        let r = Self::check_rank(u64::from(rank.0))?;
        self.bitmask ^= 1u64 << r;
        Ok(())
    }

    /// Comma-separated list of set ranks, matching `convert_to_string()`.
    pub fn convert_to_string(&self) -> String {
        let mut out = String::new();
        for rank in 0..MAX_RANK_BITS {
            if (self.bitmask & (1u64 << rank)) != 0 {
                if !out.is_empty() {
                    out.push(',');
                }
                out.push_str(&rank.to_string());
            }
        }
        out
    }
}

impl std::fmt::Display for BitmaskHelper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BitmaskHelper( mask: 0x{:x})", self.bitmask)
    }
}

// ---------------------------------------------------------------------------
// BitSetHelper — chunk-based memory allocation tracker
// Source: hdma/bitmask_helper.{hpp,cpp} (BitSetHelperRange, BitSetHelper)
// ---------------------------------------------------------------------------

/// A contiguous byte range, e.g. an allocated or free HDMA buffer region.
/// Source: bitmask_helper.hpp:87-91 `struct BitSetHelperRange`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChunkRange {
    pub base_offset: ChunkOffset,
    pub size: ChunkSize,
}

impl std::fmt::Display for ChunkRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ChunkRange(offset: {}, size: {})",
            self.base_offset.0, self.size.0
        )
    }
}

/// Chunk-based bitset allocator for a fixed-size memory region.
/// Source: bitmask_helper.hpp:110-190, bitmask_helper.cpp:102-277 `BitSetHelper`.
#[derive(Debug, Clone)]
pub struct BitSetHelper {
    bits: Vec<bool>,
    bytes_max: u64,
    bytes_per_chunk: u64,
}

impl BitSetHelper {
    pub fn new(size_bytes: u64, bytes_per_chunk: u64) -> Self {
        let num_chunks = Self::num_chunks(size_bytes, bytes_per_chunk);
        Self {
            bits: vec![false; num_chunks as usize],
            bytes_max: size_bytes,
            bytes_per_chunk,
        }
    }

    fn index(&self, bytes: u64) -> usize {
        (bytes / self.bytes_per_chunk) as usize
    }

    fn num_chunks(bytes: u64, bytes_per_chunk: u64) -> u64 {
        let mut n = bytes / bytes_per_chunk;
        if !bytes.is_multiple_of(bytes_per_chunk) {
            n += 1;
        }
        n
    }

    fn chunks_for(&self, bytes: u64) -> u64 {
        Self::num_chunks(bytes, self.bytes_per_chunk)
    }

    pub fn set(&mut self, bytes: u64) {
        let i = self.index(bytes);
        self.bits[i] = true;
    }

    pub fn set_range(&mut self, base_bytes: u64, len_bytes: u64) {
        let start = self.index(base_bytes);
        let n = self.chunks_for(len_bytes) as usize;
        for b in &mut self.bits[start..start + n] {
            *b = true;
        }
    }

    pub fn set_chunk_range(&mut self, range: ChunkRange) {
        self.set_range(range.base_offset.0, range.size.0);
    }

    pub fn is_set(&self, bytes: u64) -> bool {
        self.bits[self.index(bytes)]
    }

    pub fn is_set_range(&self, base_bytes: u64, len_bytes: u64) -> bool {
        let start = self.index(base_bytes);
        let n = self.chunks_for(len_bytes) as usize;
        self.bits[start..start + n].iter().all(|&b| !b)
    }

    pub fn is_set_chunk_range(&self, range: ChunkRange) -> bool {
        self.is_set_range(range.base_offset.0, range.size.0)
    }

    pub fn clear_all(&mut self) {
        self.bits.iter_mut().for_each(|b| *b = false);
    }

    pub fn clear(&mut self, byte_pos: u64) {
        let i = self.index(byte_pos);
        self.bits[i] = false;
    }

    pub fn clear_range(&mut self, base_bytes: u64, len_bytes: u64) {
        let start = self.index(base_bytes);
        let n = self.chunks_for(len_bytes) as usize;
        for b in &mut self.bits[start..start + n] {
            *b = false;
        }
    }

    pub fn clear_chunk_range(&mut self, range: ChunkRange) {
        self.clear_range(range.base_offset.0, range.size.0);
    }

    pub fn bytes_max(&self) -> u64 {
        self.bytes_max
    }

    pub fn bytes_used(&self) -> u64 {
        self.bits.iter().filter(|&&b| b).count() as u64 * self.bytes_per_chunk
    }

    pub fn bytes_free(&self) -> u64 {
        self.bytes_max - self.bytes_used()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes_used() == 0
    }

    pub fn is_full(&self) -> bool {
        self.bytes_free() == 0
    }

    /// Finds the first contiguous free range, or a zero-sized range if full.
    /// Source: bitmask_helper.cpp:162-197 `BitSetHelper::GetNextFree`.
    pub fn get_next_free(&self) -> ChunkRange {
        let Some(first_pos) = self.bits.iter().position(|&b| !b) else {
            return ChunkRange::default();
        };
        let end = self.bits[first_pos..]
            .iter()
            .position(|&b| b)
            .map_or(self.bits.len(), |i| first_pos + i);
        ChunkRange {
            base_offset: ChunkOffset(first_pos as u64 * self.bytes_per_chunk),
            size: ChunkSize((end - first_pos) as u64 * self.bytes_per_chunk),
        }
    }

    /// Allocates `sz` bytes, splitting across fragments capped at
    /// `max_fragment_size` (0 = unlimited). Returns an empty vec and rolls
    /// back partial allocations if there isn't enough free space.
    /// Source: bitmask_helper.cpp:199-251 `BitSetHelper::AllocateBuffer`.
    pub fn allocate_buffer(&mut self, sz: u64, max_fragment_size: u64) -> Vec<ChunkRange> {
        let mut all = Vec::new();
        let mut remaining = sz;
        while remaining > 0 {
            let mut this_range = self.get_next_free();

            if max_fragment_size != 0 && this_range.size.0 > max_fragment_size {
                this_range.size.0 = max_fragment_size;
            }

            if this_range.size.0 > remaining {
                this_range.size.0 = remaining;
                remaining = 0;
            } else if this_range.size.0 == 0 {
                for r in &all {
                    self.clear_chunk_range(*r);
                }
                return Vec::new();
            } else {
                remaining -= this_range.size.0;
            }

            self.set_chunk_range(this_range);
            all.push(this_range);
        }
        all
    }
}

impl std::fmt::Display for BitSetHelper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let perc_used = (self.bytes_used() as f64 / self.bytes_max as f64) * 100.0;
        write!(
            f,
            "BitSetHelper(Used: {perc_used:6.2}% , bytes_max: {} / bytes_per_chunk: {} = num_chunks: {}, bytes_free: {})",
            self.bytes_max,
            self.bytes_per_chunk,
            self.bits.len(),
            self.bytes_free()
        )
    }
}

// ---------------------------------------------------------------------------
// Hdma_Allocation_Manager — buffer allocations keyed to send-op completion
// Source: hdma/hdma_shm.hpp:327-446 `Hdma_Allocation_Manager`
// ---------------------------------------------------------------------------

struct AllocationRecord {
    ranges: Vec<ChunkRange>,
    creating_send_op: MessageIndex,
}

/// Error raised when there are no outstanding send-op allocations to free
/// (i.e. no ACKs available yet to reclaim HDMA buffer space).
/// Source: `RAS::CBRB::HDMANoAcksAvail` (hdma_shm.cpp:1691).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HdmaNoAcksAvail {
    /// The `sz` the caller was trying to send; 0 if none was supplied,
    /// matching hdma_shm.cpp:1676-1688's message-detail branch on `sz == 0`.
    pub requested_bytes: u64,
}

impl std::fmt::Display for HdmaNoAcksAvail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.requested_bytes == 0 {
            write!(
                f,
                "Host DMA: Requested to send with no ACKs available to free memory (likely a larger \
                 FLEX_HDMA_P2PSIZE is needed)."
            )
        } else {
            write!(
                f,
                "Host DMA: Requested to send bytes ({}) but there are no ACKs available to free memory \
                 (likely a larger FLEX_HDMA_P2PSIZE is needed).",
                self.requested_bytes
            )
        }
    }
}

impl std::error::Error for HdmaNoAcksAvail {}

/// Tracks HDMA send-buffer allocations FIFO-ordered by the send operation
/// that created them, so buffers can be freed as sends complete in order.
pub struct HdmaAllocationManager {
    bitmap: BitSetHelper,
    outstanding: VecDeque<AllocationRecord>,
    max_used: u64,
    // Mirrors the C++ member of the same name; kept for parity even though
    // no ported method reads it back yet (only `bitmap`, sized from it at
    // construction, is queried at runtime).
    #[allow(dead_code)]
    bytes_per_chunk: ChunkSize,
    max_fragment_size: ChunkSize,
}

impl HdmaAllocationManager {
    pub fn new(
        max_num_bytes_p2p: ChunkSize,
        bytes_per_chunk: ChunkSize,
        max_fragment_size: ChunkSize,
    ) -> Self {
        Self {
            bitmap: BitSetHelper::new(max_num_bytes_p2p.0, bytes_per_chunk.0),
            outstanding: VecDeque::new(),
            max_used: 0,
            bytes_per_chunk,
            max_fragment_size,
        }
    }

    pub fn reset(&mut self) {
        self.bitmap.clear_all();
        self.max_used = 0;
        self.outstanding.clear();
    }

    pub fn max_used(&self) -> u64 {
        self.max_used
    }

    pub fn bytes_max(&self) -> u64 {
        self.bitmap.bytes_max()
    }

    pub fn bytes_free(&self) -> u64 {
        self.bitmap.bytes_free()
    }

    pub fn bytes_used(&self) -> u64 {
        self.bitmap.bytes_used()
    }

    pub fn num_allocated_buffers(&self) -> usize {
        self.outstanding.len()
    }

    /// Allocates a buffer for `send_op`, tracking it for in-order release.
    pub fn allocate_buffer(&mut self, sz: ChunkSize, send_op: MessageIndex) -> Vec<ChunkRange> {
        let ranges = self.bitmap.allocate_buffer(sz.0, self.max_fragment_size.0);
        if !ranges.is_empty() {
            self.outstanding.push_back(AllocationRecord {
                ranges: ranges.clone(),
                creating_send_op: send_op,
            });
            self.max_used = self.max_used.max(self.bytes_used());
        }
        ranges
    }

    /// Peeks at the send op whose allocation would be freed next, if any.
    pub fn browse_next_send_op_that_could_be_completed(&self) -> Option<MessageIndex> {
        self.outstanding.front().map(|r| r.creating_send_op)
    }

    /// Frees the oldest outstanding allocation, returning the send op it belonged to.
    ///
    /// `sz` is the size (in bytes) of the send the caller is trying to make
    /// room for; it does not affect which allocation gets freed -- it is
    /// used only to make the error message more informative if there is
    /// nothing to free. Source: hdma_shm.cpp:1671-1691
    /// `Hdma_Allocation_Manager::free_buffer_for_next_send_op_to_complete`
    /// ("Note: the optional sz arg is only used to make the error message
    /// more informative in the case of illegal usage where there isn't
    /// enough mem in HDMA_P2P_SIZE to satisfy the request"); on an empty
    /// queue real C++ throws `RAS::CBRB::HDMANoAcksAvail` (hdma_shm.cpp:1691)
    /// rather than returning a sentinel.
    pub fn free_buffer_for_next_send_op_to_complete(
        &mut self,
        sz: u64,
    ) -> Result<MessageIndex, HdmaNoAcksAvail> {
        let Some(record) = self.outstanding.pop_front() else {
            return Err(HdmaNoAcksAvail {
                requested_bytes: sz,
            });
        };
        for range in &record.ranges {
            self.bitmap.clear_chunk_range(*range);
        }
        Ok(record.creating_send_op)
    }

    /// Frees an explicit set of ranges (used for out-of-order/error-path cleanup).
    pub fn free_buffer(&mut self, ranges: &[ChunkRange]) -> u64 {
        let mut freed = 0u64;
        for range in ranges {
            self.bitmap.clear_chunk_range(*range);
            freed += range.size.0;
        }
        freed
    }
}

// ---------------------------------------------------------------------------
// CollBufferAllocator_t — circular double-buffered collective allocator
// Source: hdma/hdma_shm.hpp:800-911, hdma_shm.cpp:110-260-ish `CollBufferAllocator_t`
// ---------------------------------------------------------------------------

/// Error raised when a requested collective allocation exceeds the buffer's
/// per-allocation cap. Source: `RAS::DEVICE_PF::HdmaCollAllocTooLarge`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollAllocTooLarge {
    pub requested: u64,
    pub max_size: u64,
}

/// Circular buffer allocator for collective (barrier/broadcast) messages,
/// split into left/right halves so a barrier can safely free everything
/// preceding it.
pub struct CollBufferAllocator {
    max_single_coll_allocation: u64,
    total_coll_buffer_size: u64,
    coll_buf_alignment: u64,

    allocated_nobar_start: u64,
    allocated_nobar_end: u64,
    prev_nobar_allocations_exist: bool,
    prev_start: u64,
    prev_end: u64,
    prev_allocations_exist: bool,
    prev_allocation_was_a_barrier: bool,

    n_allocations_signaled: u64,
    n_allocations_waited: u64,
}

impl CollBufferAllocator {
    pub fn new(max_single_coll_allocation: u64, coll_buf_alignment: u64) -> Self {
        let max_single = Self::round_up_to(max_single_coll_allocation, coll_buf_alignment);
        Self {
            max_single_coll_allocation: max_single,
            total_coll_buffer_size: 2 * max_single,
            coll_buf_alignment,
            allocated_nobar_start: 0,
            allocated_nobar_end: 0,
            prev_nobar_allocations_exist: false,
            prev_start: 0,
            prev_end: 0,
            prev_allocations_exist: false,
            prev_allocation_was_a_barrier: false,
            n_allocations_signaled: 0,
            n_allocations_waited: 0,
        }
    }

    pub fn reset(&mut self) {
        self.prev_allocations_exist = false;
        self.prev_nobar_allocations_exist = false;
        self.n_allocations_signaled = 0;
        self.n_allocations_waited = 0;
    }

    fn round_up_to(val: u64, alignment: u64) -> u64 {
        let mut rv = val / alignment * alignment;
        if rv < val {
            rv += alignment;
        }
        rv
    }

    fn round_up(&self, val: u64) -> u64 {
        Self::round_up_to(val, self.coll_buf_alignment)
    }

    /// Allocates `sz` bytes in the circular buffer, returning the starting
    /// offset. Sets `needs_previous_completions` when the caller must wait
    /// for earlier in-flight operations before reusing that space.
    /// Source: hdma_shm.cpp `CollBufferAllocator_t::allocate` (full
    /// left/right-half + wraparound state machine).
    pub fn allocate(
        &mut self,
        sz: u64,
        is_barrier: bool,
    ) -> Result<(u64, bool), CollAllocTooLarge> {
        let sz = self.round_up(sz);
        if sz > self.max_single_coll_allocation {
            return Err(CollAllocTooLarge {
                requested: sz,
                max_size: self.max_single_coll_allocation,
            });
        }

        let mut needs_previous_completions = false;

        // Case 1: buffer completely empty.
        if !self.prev_allocations_exist {
            let start = 0;
            self.prev_start = start;
            self.prev_end = sz - 1;
            self.prev_allocations_exist = true;
            self.prev_nobar_allocations_exist = false;
            self.prev_allocation_was_a_barrier = is_barrier;
            return Ok((start, needs_previous_completions));
        }

        // Case 2: previous allocation was a barrier (or no pending nobar
        // allocations) — everything before it is guaranteed complete.
        if self.prev_allocation_was_a_barrier || !self.prev_nobar_allocations_exist {
            self.allocated_nobar_start = self.prev_start;
            self.allocated_nobar_end = self.prev_end;
            self.prev_nobar_allocations_exist = true;
            self.prev_allocations_exist = true;

            let mut start = self.round_up(self.prev_end + 1);
            if start < self.max_single_coll_allocation
                && start + sz > self.max_single_coll_allocation
            {
                start = self.max_single_coll_allocation;
            }
            if start + sz > self.total_coll_buffer_size {
                start = 0;
            }
            self.prev_start = start;
            self.prev_end = start + sz - 1;
            // Note: real C++ does NOT update `prev_allocation_was_a_barrier`
            // here — it is only ever set in the empty-buffer case below, so
            // it keeps whatever value the very first allocation gave it.
            // Source: hdma_shm.cpp:203-231 `CollBufferAllocator_t::allocate` (case 2).
            return Ok((start, needs_previous_completions));
        }

        // Case 3: previous allocation was NOT a barrier — search for a free
        // spot without freeing anything, retrying three candidate locations
        // in order (matches `try_allocation_locations`).
        // Source: hdma_shm.cpp:251-315 `CollBufferAllocator_t::allocate` (case 3)
        // + hdma_shm.cpp:110-168 `try_allocation_locations`.
        let mut first_unavailable = self.total_coll_buffer_size;
        if self.allocated_nobar_start > self.prev_end {
            first_unavailable = self.allocated_nobar_start - 1;
        }

        // Location 1: directly after previous allocation.
        let mut start = self.round_up(self.prev_end + 1);
        let mut found = true;
        if start < self.max_single_coll_allocation && start + sz > self.max_single_coll_allocation {
            found = false;
        }
        if start <= first_unavailable && start + sz > first_unavailable {
            found = false;
        }

        // Location 2: at the left/right boundary.
        if !found && start < self.max_single_coll_allocation {
            start = self.max_single_coll_allocation;
            found = true;
            if start >= first_unavailable {
                found = false;
            }
            if start < first_unavailable && start + sz > first_unavailable {
                found = false;
            }
        }

        // Location 3: wrap to offset 0.
        if !found {
            start = 0;
            found = true;
            if self.allocated_nobar_start > self.allocated_nobar_end {
                found = false;
            }
            if self.allocated_nobar_start == 0 {
                found = false;
            }
            first_unavailable = self.allocated_nobar_start - 1;
            if start < first_unavailable && start + sz > first_unavailable {
                found = false;
            }
        }

        if found {
            self.allocated_nobar_end = self.prev_end;
            self.prev_nobar_allocations_exist = true;
            self.prev_allocations_exist = true;

            start = self.round_up(start);
            self.prev_start = start;
            self.prev_end = start + sz - 1;
            return Ok((start, false));
        }

        // No space found anywhere: caller must wait for the previous
        // (non-barrier) allocations to complete before we can safely reuse
        // that space; the next allocation proceeds as if previous was a
        // barrier (case 2 above will fire next time).
        needs_previous_completions = true;
        self.allocated_nobar_start = self.prev_start;
        self.allocated_nobar_end = self.prev_end;
        self.prev_nobar_allocations_exist = false;
        self.prev_allocations_exist = true;

        let mut start = self.round_up(self.prev_end + 1);
        if start < self.max_single_coll_allocation && start + sz >= self.max_single_coll_allocation
        {
            start = self.max_single_coll_allocation;
        }
        if start + sz > self.total_coll_buffer_size {
            start = 0;
        }
        self.prev_start = start;
        self.prev_end = start + sz - 1;
        Ok((start, needs_previous_completions))
    }

    pub fn range_overlaps(a: u64, b: u64, c: u64, d: u64) -> bool {
        a <= d && c < b
    }
}

// ---------------------------------------------------------------------------
// hdma_op_t / InstructionFromCard — write-done instruction framing
// Source: hdma/host_wdone.hpp:120-289
// ---------------------------------------------------------------------------

/// HDMA operation types for host-coordinated write-done signaling.
/// Source: host_wdone.hpp:123-128 `enum hdma_op_t`. Explicit discriminants
/// match the wire values exactly, since `opcode_tail` (below) packs these
/// into a `u16` alongside the flag bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdmaOp {
    /// `HDMA_OP_NULL = 0`: no-op. Source: host_wdone.hpp:124.
    Null = 0,
    SetWdone = 1,
    AincWdone = 2,
    Vsid = 3,
}

/// Typed view of the flag bits packed alongside `hdma_op_t` in the wire opcode.
/// Source: host_wdone.hpp:129-135 `HDMA_OPFLAG_*`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HdmaOpFlags {
    pub is_sid_format: bool,
    pub is_vsid_format: bool,
    pub is_vsid_ainc: bool,
    pub is_vsid_wait: bool,
    pub notify_on_completion: bool,
}

/// Packs `op`/`flags` into the wire-format `u16` opcode value real C++ builds
/// via `instr.opcode = opcode | HDMA_OPFLAG_IS_..._FORMAT | ...`
/// (host_wdone.cpp:212-215, 232-236, 264, 286-288) and then copies verbatim
/// into `opcode_tail` for wire-format verification (host_wdone.cpp:221, 256,
/// 277, 307: `instr.opcode_tail = instr.opcode;`).
/// Source: host_wdone.hpp:129-135 `HDMA_OPFLAG_*` bit positions.
fn pack_hdma_opcode(op: HdmaOp, flags: HdmaOpFlags) -> u16 {
    let mut bits = op as u16;
    if flags.is_sid_format {
        bits |= 1 << 5;
    }
    if flags.is_vsid_format {
        bits |= 1 << 6;
    }
    if flags.is_vsid_ainc {
        bits |= 1 << 7;
    }
    if flags.is_vsid_wait {
        bits |= 1 << 8;
    }
    if flags.notify_on_completion {
        bits |= 1 << 9;
    }
    bits
}

/// A write-done instruction addressed by plain segment id (SID format).
/// Source: host_wdone.hpp:142-153 `InstructionFromCard_SID_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SidInstruction {
    pub op: HdmaOp,
    pub flags: HdmaOpFlags,
    pub sid: SegId,
    pub local_completion_sid: Option<SegId>,
    pub peer_mask: BitmaskHelper,
    pub val: i32,
    /// Wire-format verification field: a verbatim copy of the packed
    /// `op`/`flags` opcode. Source: host_wdone.hpp:152
    /// `uint16_t opcode_tail; ///< Opcode verification tail`, set at
    /// host_wdone.cpp:221 `instr.opcode_tail = instr.opcode;`.
    pub opcode_tail: u16,
}

impl SidInstruction {
    /// Source: host_wdone.cpp:205-223 `HostWdone::make_instruction` (SID overload).
    pub fn new(
        op: HdmaOp,
        peer_mask: BitmaskHelper,
        sid: SegId,
        local_completion_sid: Option<SegId>,
        val: i32,
    ) -> Self {
        let flags = HdmaOpFlags {
            is_sid_format: true,
            notify_on_completion: local_completion_sid.is_some(),
            ..Default::default()
        };
        let opcode_tail = pack_hdma_opcode(op, flags);
        Self {
            op,
            flags,
            sid,
            local_completion_sid,
            peer_mask,
            val,
            opcode_tail,
        }
    }
}

/// A write-done instruction addressed by virtual SID (VSID format), able to
/// target up to 448 (`VSID_MASK_BITS`) peers/sids via a wide bitmask.
/// Source: host_wdone.hpp:159-169 `InstructionFromCard_VSID_t`.
const VSID_MASK_WORDS: usize = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VsidInstruction {
    pub op: HdmaOp,
    pub flags: HdmaOpFlags,
    pub vsid: u16,
    pub peer_mask: [u64; VSID_MASK_WORDS],
    pub sid_mask: [u64; VSID_MASK_WORDS],
    /// Wire-format verification field, see `SidInstruction::opcode_tail`.
    /// Source: host_wdone.hpp:168, set at host_wdone.cpp:256/277/307.
    pub opcode_tail: u16,
}

impl VsidInstruction {
    fn set_bit(mask: &mut [u64; VSID_MASK_WORDS], bit: usize) {
        mask[bit / 64] |= 1u64 << (bit % 64);
    }

    /// Source: host_wdone.cpp:228-258 `make_instruction` (VSID increment overload).
    ///
    /// `op` is the caller-supplied base opcode (real C++'s `hdma_op_t opcode`
    /// parameter, host_wdone.cpp:228) that `HDMA_OPFLAG_IS_VSID_FORMAT` is
    /// OR'd onto -- real callers pass e.g. `HDMA_OP_AINC_WDONE`, not always
    /// `HDMA_OP_VSID` (see control_block_stream.cpp:5927
    /// `inst_create_and_stage(flex::HDMA_OP_AINC_WDONE, mylist, ...)`), so it
    /// is not hardcoded here.
    pub fn new_increment(
        op: HdmaOp,
        vsid: u16,
        peers: impl IntoIterator<Item = usize>,
        notify_on_completion: bool,
    ) -> Self {
        let mut peer_mask = [0u64; VSID_MASK_WORDS];
        for p in peers {
            Self::set_bit(&mut peer_mask, p);
        }
        let flags = HdmaOpFlags {
            is_vsid_format: true,
            is_vsid_ainc: true,
            notify_on_completion,
            ..Default::default()
        };
        let opcode_tail = pack_hdma_opcode(op, flags);
        Self {
            op,
            flags,
            vsid,
            peer_mask,
            sid_mask: [0; VSID_MASK_WORDS],
            opcode_tail,
        }
    }

    /// Source: host_wdone.cpp:261-279 `make_instruction` (VSID wait overload).
    /// `op` is the caller-supplied base opcode; see `new_increment`'s doc.
    pub fn new_wait(op: HdmaOp, vsids: impl IntoIterator<Item = usize>) -> Self {
        let mut sid_mask = [0u64; VSID_MASK_WORDS];
        for v in vsids {
            Self::set_bit(&mut sid_mask, v);
        }
        let flags = HdmaOpFlags {
            is_vsid_format: true,
            is_vsid_wait: true,
            notify_on_completion: true,
            ..Default::default()
        };
        let opcode_tail = pack_hdma_opcode(op, flags);
        Self {
            op,
            flags,
            vsid: 0,
            peer_mask: [0; VSID_MASK_WORDS],
            sid_mask,
            opcode_tail,
        }
    }

    /// Source: host_wdone.cpp:282-309 `make_instruction` (combined increment+wait overload).
    /// `op` is the caller-supplied base opcode; see `new_increment`'s doc.
    pub fn new_increment_and_wait(
        op: HdmaOp,
        vsid: u16,
        peers: impl IntoIterator<Item = usize>,
        vsids_to_wait: impl IntoIterator<Item = usize>,
    ) -> Self {
        let mut peer_mask = [0u64; VSID_MASK_WORDS];
        for p in peers {
            Self::set_bit(&mut peer_mask, p);
        }
        let mut sid_mask = [0u64; VSID_MASK_WORDS];
        for v in vsids_to_wait {
            Self::set_bit(&mut sid_mask, v);
        }
        let flags = HdmaOpFlags {
            is_vsid_format: true,
            is_vsid_ainc: true,
            is_vsid_wait: true,
            notify_on_completion: true,
            ..Default::default()
        };
        let opcode_tail = pack_hdma_opcode(op, flags);
        Self {
            op,
            flags,
            vsid,
            peer_mask,
            sid_mask,
            opcode_tail,
        }
    }
}

/// Maximum number of instructions the card can have outstanding to the host
/// monitor before the caller must wait. Source: host_wdone.hpp:45
/// `#define MAX_OUTSTANDING_HDMA_INSTR 64`.
pub const MAX_OUTSTANDING_HDMA_INSTR: usize = 64;

/// A slot index into the outstanding-instruction ring, distinct from a raw
/// `usize` because it wraps modulo `MAX_OUTSTANDING_HDMA_INSTR`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstrSlot(pub u32);

/// Card-side instruction queue bookkeeping: tracks how many instructions are
/// outstanding and whether the caller must block for a completion barrier
/// before issuing more. Purely local counter arithmetic — no senlib calls.
/// Source: host_wdone.cpp:314-335 `iqueue_get_instr_slot` / `iqueue_waited` / `iqueue_reset`.
#[derive(Debug, Default)]
pub struct InstructionQueue {
    outstanding: u32,
    next: u32,
}

impl InstructionQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserves the next slot. Returns the slot and whether the caller must
    /// now block on a completion barrier (queue is full).
    pub fn get_instr_slot(&mut self) -> (InstrSlot, bool) {
        let slot = self.next;
        self.next = (self.next + 1) % MAX_OUTSTANDING_HDMA_INSTR as u32;

        self.outstanding += 1;
        let will_wait = self.outstanding == MAX_OUTSTANDING_HDMA_INSTR as u32;
        (InstrSlot(slot), will_wait)
    }

    pub fn waited(&mut self) {
        self.outstanding = 0;
    }

    pub fn reset(&mut self) {
        self.next = 0;
    }
}

// ---------------------------------------------------------------------------
// RdmaRendezvousManager — leader/follower barrier & broadcast
// Source: multi_device/rdma/rdma_rendezvous_manager.{hpp,cpp}
// ---------------------------------------------------------------------------

/// Coordinates a barrier (and optional value broadcast from rank 0) across
/// `world_size` ranks using a shared rendezvous file under a common
/// directory. Ported natively: the rendezvous file is a plain POSIX file
/// (via `std::fs`), not a senlib primitive — the C++ used `ShmFile` purely
/// as a convenient shared-memory-file wrapper around the byte-addressable
/// value-based protocol implemented here (each rank's state lives at its
/// own fixed byte offset, polled BY CONTENT — this is NOT a file-presence
/// protocol: the file's mere existence/absence is only ever used by
/// followers to detect that the leader has published the barrier at all,
/// never as the release signal itself).
/// Source: rdma_rendezvous_manager.cpp:20-186.
pub struct RdmaRendezvousManager {
    local_rank: RankId,
    world_size: u32,
    rendezvous_dir: PathBuf,
    barrier_seq: u32,
}

const BARRIER_WARNING_INTERVAL: Duration = Duration::from_secs(10);
const BARRIER_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// Per-rank barrier state values, matching the real C++ 3-state protocol
/// exactly (rdma_rendezvous_manager.cpp:26-28
/// `BARRIER_NOT_ARRIVED`/`BARRIER_ARRIVED`/`BARRIER_RELEASED`). Each rank's
/// state lives at a fixed byte offset (its rank index) in the rendezvous
/// file; only `LEADER_RANK`'s slot ever transitions to `RELEASED`.
const BARRIER_NOT_ARRIVED: u8 = 0;
const BARRIER_ARRIVED: u8 = 1;
const BARRIER_RELEASED: u8 = 2;
const LEADER_RANK: u32 = 0;

impl RdmaRendezvousManager {
    pub fn new(local_rank: RankId, world_size: u32, rendezvous_dir: PathBuf) -> Self {
        Self {
            local_rank,
            world_size,
            rendezvous_dir,
            barrier_seq: 0,
        }
    }

    fn barrier_path(&self, name: &str, seq: u32) -> PathBuf {
        self.rendezvous_dir.join(format!("barrier_{seq}_{name}"))
    }

    fn barrier_tmp_path(&self, name: &str, seq: u32) -> PathBuf {
        self.rendezvous_dir
            .join(format!("barrier_{seq}_{name}_tmp"))
    }

    /// Byte offset of the broadcast payload (`sync_data[world_size_]` in the
    /// real C++ `size_t` array; here, 8 bytes right after the per-rank state
    /// bytes). Source: rdma_rendezvous_manager.cpp:49 `sync_data[world_size_] = rdata;`.
    fn payload_offset(&self) -> u64 {
        u64::from(self.world_size)
    }

    /// Blocks until all ranks reach the barrier.
    /// Source: rdma_rendezvous_manager.cpp:154 `RdmaRendezvousManager::Barrier`.
    pub fn barrier(&mut self, name: &str) -> std::io::Result<()> {
        self.barrier_and_bcast(name, 0).map(|_| ())
    }

    /// Barrier with a value broadcast from rank 0 (`LEADER_RANK`) to all peers.
    /// Source: rdma_rendezvous_manager.cpp:156-186 `RdmaRendezvousManager::BarrierAndBcast`.
    pub fn barrier_and_bcast(&mut self, name: &str, data: u64) -> std::io::Result<u64> {
        self.barrier_seq += 1;
        if self.world_size <= 1 {
            return Ok(data);
        }
        let path = self.barrier_path(name, self.barrier_seq);

        if self.local_rank.0 == LEADER_RANK {
            let tmp_path = self.barrier_tmp_path(name, self.barrier_seq);
            self.lead_barrier(&tmp_path, &path, data)
        } else {
            self.follow_barrier(&path)
        }
    }

    /// Leader: publish the rendezvous file via write-to-temp-then-atomic-rename,
    /// wait (by polling each follower's state byte BY CONTENT) for every rank
    /// to report `BARRIER_ARRIVED`, then release by writing `BARRIER_RELEASED`
    /// into its own state byte, and finally unlink the file as cleanup (NOT as
    /// the release signal — followers have already observed `BARRIER_RELEASED`
    /// by content before this unlink happens).
    /// Source: rdma_rendezvous_manager.cpp:34-73 `setBarrier`.
    fn lead_barrier(
        &self,
        tmp_path: &std::path::Path,
        path: &std::path::Path,
        data: u64,
    ) -> std::io::Result<u64> {
        std::fs::create_dir_all(&self.rendezvous_dir)?;

        // Build the initial buffer and publish it via write-to-temp +
        // atomic rename (rdma_rendezvous_manager.cpp:41-53): every rank's
        // state starts NOT_ARRIVED except the leader's own, which is set to
        // ARRIVED before publishing; the broadcast payload occupies the
        // trailing 8 bytes.
        let mut buf = vec![BARRIER_NOT_ARRIVED; self.world_size as usize];
        buf[LEADER_RANK as usize] = BARRIER_ARRIVED;
        buf.extend_from_slice(&data.to_le_bytes());
        std::fs::write(tmp_path, &buf)?;
        std::fs::rename(tmp_path, path)?;

        // Hold one read handle open across all polls, mirroring the real
        // C++'s persistent shared-memory mapping: each follower writes only
        // to its own byte offset, so polling this handle's content never
        // races with those disjoint writes the way a whole-buffer
        // read-modify-write would.
        let mut file = std::fs::File::open(path)?;
        let mut last_warn = Instant::now();
        for peer in 1..self.world_size {
            loop {
                let state = read_state_byte(&mut file, u64::from(peer))?;
                if state == BARRIER_ARRIVED {
                    break;
                }
                if last_warn.elapsed() >= BARRIER_WARNING_INTERVAL {
                    last_warn = Instant::now();
                }
                std::thread::sleep(BARRIER_POLL_INTERVAL);
            }
        }

        // Release the barrier: write ONLY the leader's own state byte.
        // Source: rdma_rendezvous_manager.cpp:69 `sync_data[LEADER_RANK] = BARRIER_RELEASED;`.
        write_state_byte(path, u64::from(LEADER_RANK), BARRIER_RELEASED)?;

        // Cleanup the file -- this happens strictly after release, and is
        // not itself the release signal (followers already saw
        // BARRIER_RELEASED by content).
        // Source: rdma_rendezvous_manager.cpp:72 `sync_file.Unlink();`.
        let _ = std::fs::remove_file(path);
        Ok(data)
    }

    /// Follower: wait for the leader's rendezvous file to appear, announce
    /// arrival by writing ONLY this rank's own state byte, then poll the
    /// leader's state byte BY CONTENT for `BARRIER_RELEASED` before reading
    /// the broadcast payload.
    /// Source: rdma_rendezvous_manager.cpp:77-147 `waitForBarrier`.
    fn follow_barrier(&self, path: &std::path::Path) -> std::io::Result<u64> {
        while !path.exists() {
            std::thread::sleep(BARRIER_POLL_INTERVAL);
        }

        // I have arrived at the barrier: write only my own state byte, never
        // the whole buffer, so concurrent followers' arrivals can't clobber
        // each other via a stale read-modify-write.
        // Source: rdma_rendezvous_manager.cpp:117 `sync_data[local_rank_] = BARRIER_ARRIVED;`.
        write_state_byte(path, u64::from(self.local_rank.0), BARRIER_ARRIVED)?;

        // Hold one read handle open across all polls (see `lead_barrier`'s
        // comment on why this matters: it mirrors the real code's
        // persistent shared-memory mapping, which is still valid even after
        // the leader unlinks the directory entry as post-release cleanup).
        //
        // Deviation from real C++: real `waitForBarrier` re-opens the shm
        // file and re-announces arrival if the inner wait exceeds
        // `BARRIER_WARNING_INTERVAL_SECONDS` without seeing release (guarding
        // against the file having been recreated for a subsequent job); this
        // port polls a single open handle without that reattach-on-timeout
        // retry, which is a robustness feature, not part of the core
        // 3-state/atomic-rename correctness fix this port is pinning.
        let mut file = std::fs::File::open(path)?;
        loop {
            let leader_state = read_state_byte(&mut file, u64::from(LEADER_RANK))?;
            if leader_state == BARRIER_RELEASED {
                break;
            }
            std::thread::sleep(BARRIER_POLL_INTERVAL);
        }

        // Read the broadcast payload.
        // Source: rdma_rendezvous_manager.cpp:124 `rdata = sync_data[world_size_];`.
        let mut payload = [0u8; 8];
        {
            use std::io::{Read, Seek, SeekFrom};
            file.seek(SeekFrom::Start(self.payload_offset()))?;
            file.read_exact(&mut payload)?;
        }
        Ok(u64::from_le_bytes(payload))
    }
}

/// Reads a single per-rank state byte from `file` at `rank`'s offset,
/// without touching (or racing on) any other rank's byte.
fn read_state_byte(file: &mut std::fs::File, rank: u64) -> std::io::Result<u8> {
    use std::io::{Read, Seek, SeekFrom};
    let mut byte = [0u8; 1];
    file.seek(SeekFrom::Start(rank))?;
    file.read_exact(&mut byte)?;
    Ok(byte[0])
}

/// Writes a single per-rank state byte at `rank`'s offset via its own file
/// handle, without reading (or rewriting) the rest of the buffer -- this is
/// the fix for the lost-update race a whole-buffer read-modify-write would
/// otherwise have under concurrent followers.
fn write_state_byte(path: &std::path::Path, rank: u64, value: u8) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom, Write};
    let mut file = std::fs::OpenOptions::new().write(true).open(path)?;
    file.seek(SeekFrom::Start(rank))?;
    file.write_all(&[value])?;
    Ok(())
}

// ---------------------------------------------------------------------------
// CollectiveIPC — Unix-domain-socket collectives (basic_ipc)
// Source: multi_device/hdma/basic_ipc.{hpp,cpp}
// ---------------------------------------------------------------------------

/// Collective IPC primitives (send/recv/barrier/bcast/allgatherv) over Unix
/// domain sockets, ported natively with `std::os::unix::net` — no senlib
/// involvement anywhere in `basic_ipc.cpp`.
/// Source: basic_ipc.cpp:47-557.
pub struct CollectiveIpc {
    my_rank: RankId,
    n_ranks: u32,
    sockets: Vec<Option<UnixStream>>,
}

impl CollectiveIpc {
    /// Establishes connections to all peers via a rendezvous directory
    /// keyed by `common_key`. Source: basic_ipc.cpp:204-339 `setup_communication`.
    pub fn connect(
        my_rank: RankId,
        n_ranks: u32,
        common_key: u32,
        sock_dir: &std::path::Path,
    ) -> std::io::Result<Self> {
        let mut sockets: Vec<Option<UnixStream>> = (0..n_ranks).map(|_| None).collect();
        if n_ranks == 1 {
            return Ok(Self {
                my_rank,
                n_ranks,
                sockets,
            });
        }

        std::fs::create_dir_all(sock_dir)?;
        // Harden the listen directory to owner-only rwx, matching real C++'s
        // explicit permission tightening after directory creation.
        // Source: basic_ipc.cpp:223-237 `std::filesystem::create_directories(USOCK_DIR, ...);`
        // followed by `std::filesystem::permissions(USOCK_DIR, std::filesystem::perms::owner_all,
        // std::filesystem::perm_options::replace, ...);`.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(sock_dir, std::fs::Permissions::from_mode(0o700))?;
        }
        let my_path = sock_dir.join(format!("group.{common_key}.R.{}", my_rank.0));
        let _ = std::fs::remove_file(&my_path);
        let listener = UnixListener::bind(&my_path)?;

        // Accept from all higher-numbered ranks (they connect to us).
        for _ in (my_rank.0 + 1)..n_ranks {
            let (mut stream, _) = listener.accept()?;
            let mut peer_buf = [0u8; 4];
            stream.read_exact(&mut peer_buf)?;
            let peer = u32::from_le_bytes(peer_buf);
            sockets[peer as usize] = Some(stream);
        }

        // Connect to all lower-numbered ranks, retrying until they're listening.
        for peer in (0..my_rank.0).rev() {
            let peer_path = sock_dir.join(format!("group.{common_key}.R.{peer}"));
            let deadline = Instant::now() + Duration::from_secs(300);
            let stream = loop {
                match UnixStream::connect(&peer_path) {
                    Ok(s) => break s,
                    Err(_) if Instant::now() < deadline => {
                        std::thread::sleep(Duration::from_millis(10))
                    }
                    Err(e) => return Err(e),
                }
            };
            let mut stream = stream;
            stream.write_all(&my_rank.0.to_le_bytes())?;
            sockets[peer as usize] = Some(stream);
        }

        let _ = std::fs::remove_file(&my_path);
        Ok(Self {
            my_rank,
            n_ranks,
            sockets,
        })
    }

    fn socket_mut(&mut self, peer: RankId) -> &mut UnixStream {
        self.sockets[peer.0 as usize]
            .as_mut()
            .expect("peer socket not connected")
    }

    /// Source: basic_ipc.cpp:358-365 `CollectiveIPC::recv`.
    pub fn recv(&mut self, buf: &mut [u8], peer: RankId) -> std::io::Result<()> {
        self.socket_mut(peer).read_exact(buf)
    }

    /// Source: basic_ipc.cpp:367-374 `CollectiveIPC::send`.
    pub fn send(&mut self, buf: &[u8], peer: RankId) -> std::io::Result<()> {
        self.socket_mut(peer).write_all(buf)
    }

    /// Source: basic_ipc.cpp:376-395 `CollectiveIPC::barrier`.
    pub fn barrier(&mut self) -> std::io::Result<()> {
        let mut unused = [0u8; 1];
        if self.my_rank.0 == 0 {
            for i in 1..self.n_ranks {
                self.recv(&mut unused, RankId(i))?;
            }
            for i in 1..self.n_ranks {
                self.send(&unused, RankId(i))?;
            }
        } else {
            self.send(&unused, RankId(0))?;
            self.recv(&mut unused, RankId(0))?;
        }
        Ok(())
    }

    /// Source: basic_ipc.cpp:397-414 `CollectiveIPC::bcast`.
    pub fn bcast(&mut self, buf: &mut [u8], root: RankId) -> std::io::Result<()> {
        if self.my_rank == root {
            for i in 0..self.n_ranks {
                if i == root.0 {
                    continue;
                }
                self.send(buf, RankId(i))?;
            }
            Ok(())
        } else {
            self.recv(buf, root)
        }
    }

    /// Variable-size allgather: local data of arbitrary size from every rank
    /// is combined and delivered in full to every rank via a hypercube
    /// gather-then-broadcast tree.
    /// Source: basic_ipc.cpp:416-555 `perform_tree_gather` / `perform_tree_broadcast`
    /// / `CollectiveIPC::packed_allgatherv`.
    pub fn packed_allgatherv(
        &mut self,
        local: &[u8],
    ) -> std::io::Result<(Vec<u8>, Vec<usize>, Vec<usize>)> {
        let n = self.n_ranks as usize;
        let mut peer_sizes = vec![0usize; n];

        if self.my_rank.0 == 0 {
            peer_sizes[0] = local.len();
            // `r` drives both the RankId to receive from and the peer_sizes
            // index — not a plain by-value iteration, so `enumerate()` over
            // `peer_sizes` wouldn't remove the need for the index anyway.
            for (r, slot) in peer_sizes.iter_mut().enumerate().take(n).skip(1) {
                let mut sz = [0u8; 8];
                self.recv(&mut sz, RankId(r as u32))?;
                *slot = u64::from_le_bytes(sz) as usize;
            }
        } else {
            self.send(&(local.len() as u64).to_le_bytes(), RankId(0))?;
        }

        if self.my_rank.0 == 0 {
            let flat: Vec<u8> = peer_sizes
                .iter()
                .flat_map(|s| (*s as u64).to_le_bytes())
                .collect();
            for r in 1..n {
                self.send(&flat, RankId(r as u32))?;
            }
        } else {
            let mut flat = vec![0u8; n * 8];
            self.recv(&mut flat, RankId(0))?;
            for r in 0..n {
                peer_sizes[r] =
                    u64::from_le_bytes(flat[r * 8..r * 8 + 8].try_into().unwrap()) as usize;
            }
        }

        let mut peer_byte_indexes = vec![0usize; n];
        let mut offset = 0usize;
        for r in 0..n {
            peer_byte_indexes[r] = offset;
            offset += peer_sizes[r];
        }
        let total = offset;

        let mut rbuf = vec![0u8; total];
        rbuf[peer_byte_indexes[self.my_rank.0 as usize]
            ..peer_byte_indexes[self.my_rank.0 as usize] + local.len()]
            .copy_from_slice(local);

        let (stage, pow2) = self.tree_gather(&mut rbuf, &peer_sizes, &peer_byte_indexes)?;
        self.tree_broadcast(&mut rbuf, total, stage, pow2)?;

        Ok((rbuf, peer_sizes, peer_byte_indexes))
    }

    fn tree_gather(
        &mut self,
        rbuf: &mut [u8],
        peer_sizes: &[usize],
        peer_byte_indexes: &[usize],
    ) -> std::io::Result<(i32, u64)> {
        let my = self.my_rank.0;
        let n = self.n_ranks;
        let mut stage = 0i32;
        let mut pow2 = 1u64;
        while pow2 < u64::from(n) {
            let peer = my ^ pow2 as u32;
            if my & (pow2 as u32 - 1) != 0 {
                stage += 1;
                pow2 *= 2;
                continue;
            }
            if my < peer && peer < n {
                let recv_sz: usize = (peer..peer + pow2 as u32)
                    .map(|r| peer_sizes[r as usize])
                    .sum();
                let start = peer_byte_indexes[peer as usize];
                self.recv(&mut rbuf[start..start + recv_sz], RankId(peer))?;
            } else if my > peer {
                let send_sz: usize = (my..my + pow2 as u32).map(|r| peer_sizes[r as usize]).sum();
                let start = peer_byte_indexes[my as usize];
                self.send(&rbuf[start..start + send_sz], RankId(peer))?;
            }
            stage += 1;
            pow2 *= 2;
        }
        Ok((stage, pow2))
    }

    fn tree_broadcast(
        &mut self,
        rbuf: &mut [u8],
        tot_size: usize,
        stage: i32,
        pow2: u64,
    ) -> std::io::Result<()> {
        let my = self.my_rank.0;
        let n = self.n_ranks;
        let mut stage = stage - 1;
        let mut pow2 = pow2 / 2;
        while stage >= 0 {
            let peer = my ^ pow2 as u32;
            if my & (pow2 as u32).wrapping_sub(1) != 0 {
                stage -= 1;
                pow2 /= 2;
                continue;
            }
            if my < peer && peer < n {
                self.send(&rbuf[..tot_size], RankId(peer))?;
            } else if my > peer {
                self.recv(&mut rbuf[..tot_size], RankId(peer))?;
            }
            stage -= 1;
            pow2 /= 2;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AIUTopo — device-pair communication protocol selection
// Source: multi_device/topology.{hpp,cpp}
// ---------------------------------------------------------------------------

/// Communication protocol chosen between a pair of devices.
/// Source: topology.hpp:93-98 `enum p2p_protocol_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum P2pProtocol {
    Self_,
    P2pRdma,
    HostDma,
}

/// Error raised when the topology has no protocol recorded for a rank.
/// Source: `RAS::DEVICE_PF::TopoNoProtocolForRank`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopoNoProtocolForRank {
    pub rank: u32,
}

/// Per-device-pair protocol table plus force-override flags, deciding
/// whether HDMA or P2P RDMA is used for a given peer.
/// Source: topology.hpp:57-126 `class AIUTopo`.
pub struct AiuTopo {
    local_rank: u32,
    world_size: u32,
    force_hdma: bool,
    force_hdma_load: bool,
    force_p2p_rdma: bool,
    world_peers_protocols: Vec<HashSet<P2pProtocol>>,
}

impl AiuTopo {
    pub fn new(
        local_rank: u32,
        world_size: u32,
        force_hdma: bool,
        force_hdma_load: bool,
        force_p2p_rdma: bool,
    ) -> Self {
        let mut protocols = vec![HashSet::new(); world_size as usize];
        for (rank, set) in protocols.iter_mut().enumerate() {
            if rank as u32 == local_rank {
                set.insert(P2pProtocol::Self_);
            } else {
                set.insert(P2pProtocol::HostDma);
            }
        }
        Self {
            local_rank,
            world_size,
            force_hdma,
            force_hdma_load,
            force_p2p_rdma,
            world_peers_protocols: protocols,
        }
    }

    /// Records that `peer` supports P2P RDMA (in addition to whatever else is set).
    pub fn add_peer_protocol(&mut self, peer: u32, protocol: P2pProtocol) {
        self.world_peers_protocols[peer as usize].insert(protocol);
    }

    /// Source: topology.cpp:223-267 `AIUTopo::is_hdma_needed`.
    ///
    /// Takes `&mut self`: when no peer needs HDMA, real C++ caches
    /// `force_p2p_rdma_ = true` as a side effect before returning `false`
    /// (topology.cpp:263-265, "If everyone has P2P (R)DMA then we don't need
    /// the Host DMA setup / Cache this information for later"), which
    /// permanently short-circuits every future call via the `force_p2p_rdma_`
    /// check at the top (topology.cpp:227-230). This memoization is ported
    /// here exactly, so callers must hold `&mut AiuTopo`.
    pub fn is_hdma_needed(&mut self) -> Result<bool, TopoNoProtocolForRank> {
        if self.world_size <= 1 {
            return Ok(false);
        }
        if self.force_p2p_rdma {
            return Ok(false);
        }
        if self.force_hdma || self.force_hdma_load {
            return Ok(true);
        }
        for rank in 0..self.world_size {
            if rank == self.local_rank {
                continue;
            }
            let protocols = self
                .world_peers_protocols
                .get(rank as usize)
                .ok_or(TopoNoProtocolForRank { rank })?;
            let needs_hdma = !protocols.contains(&P2pProtocol::P2pRdma);
            if needs_hdma {
                return Ok(true);
            }
        }

        // If everyone has P2P (R)DMA then we don't need the Host DMA setup.
        // Cache this information for later (topology.cpp:263-265).
        self.force_p2p_rdma = true;
        Ok(false)
    }

    /// Source: topology.cpp:269-287 `AIUTopo::comm_by_p2p_rdma_to`.
    pub fn comm_by_p2p_rdma_to(&self, peer_rank: u32) -> bool {
        if self.force_p2p_rdma {
            return true;
        }
        if self.force_hdma {
            return false;
        }
        self.world_peers_protocols[peer_rank as usize].contains(&P2pProtocol::P2pRdma)
    }
}

// ---------------------------------------------------------------------------
// RdmaUnit — mock/backend-agnostic point-to-point RDMA transfer bookkeeping
// Source: multi_device/rdma/rdma_unit.{hpp,cpp}
// ---------------------------------------------------------------------------

/// Address-translation segment for a remote rank/segment pair. Wraps the
/// parity/valid-bit-checked raw register described by `Xseg`.
/// Source: rdma_unit.cpp:504-529 `RdmaUnit::WriteXseg`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Xseg {
    pub bytes: u64,
    pub valid: bool,
}

/// Error returned by RdmaUnit operations, mirroring the `sendnn::Status`
/// error paths in the C++ (as opposed to hardware failures, which surface
/// through the FFI call's own return code).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RdmaUnitError {
    RdmaNotEnabled,
    LocalRankTarget,
    RemoteXsegNotValid { target: RankId, sid: SegId },
    MessageTooLarge { size: u64, max: u64 },
    XsegParityMismatch { expected_parity: bool },
    WriteDoneWrapped,
    Timeout,
    HardwareError(i32),
}

/// One poll's classification of the write-done counter `WaitForBarrier`
/// reads, extracted as pure logic so it can be regression-tested without a
/// live RDMA shared-memory segment (see `classify_wdone_check`'s own doc and
/// the `wdone_check_classification_tests` module below).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WdoneCheckOutcome {
    /// Top byte set: hardware reported an error code (rdma_unit.cpp's own
    /// `val >> 24` check).
    HardwareError(i32),
    /// Strictly positive: the peer's write-done signal has arrived.
    Ready,
    /// Negative: the peer's counter wrapped -- in practice this is what a
    /// peer rank's `RdmaUnit` destructor sets every counter to on exit (see
    /// `rdma_negative_counter_test.cpp`'s `DetectPeerExit`), so this must be
    /// surfaced as an error immediately, not treated as "not yet ready".
    Wrapped,
    /// Zero: no signal yet, keep polling until `deadline`.
    NotYetReady,
}

/// Port of the branch structure in `RdmaUnit::WaitForBarrier`'s poll loop
/// (rdma_unit.cpp:443-484). Real flex-cxx shipped a bug here once: `if(val <
/// 0)` was nested inside `if(val > 0)`, making the negative-counter branch
/// unreachable (a value can't be both `> 0` and `< 0`) and forcing a peer's
/// exit-triggered `-1` counter to be misread as "not yet ready", looping
/// until `WaitForBarrier`'s full timeout instead of failing immediately (see
/// `rdma_negative_counter_test.cpp`'s docstring). This function's three
/// checks are siblings via a `match`, not C++ `if`-nesting, so that class of
/// bug cannot reappear silently; `wdone_check_classification_tests` below
/// pins the `val < 0` behavior directly.
fn classify_wdone_check(val: i64) -> WdoneCheckOutcome {
    if (val >> 24) > 0 {
        return WdoneCheckOutcome::HardwareError(val as i32);
    }
    if val > 0 {
        return WdoneCheckOutcome::Ready;
    }
    if val < 0 {
        return WdoneCheckOutcome::Wrapped;
    }
    WdoneCheckOutcome::NotYetReady
}

const MAX_HDMA_MESSAGE_BYTES: u64 = 4 * 1024 * 1024;
const XSEG_VALID_BIT: u32 = 1 << 30;
const XSEG_PARITY_BIT: u32 = 1 << 31;
const XSEG_ADDR_MASK: u32 = XSEG_VALID_BIT - 1;

/// Point-to-point RDMA transfer coordination: validation, multicast fan-out,
/// and write-done wait/signal bookkeeping natively; the literal memory copy
/// and hardware write-done counters cross FFI (`senlib_ffi_multi_device`).
/// Source: rdma_unit.hpp, rdma_unit.cpp:382-529.
pub struct RdmaUnit {
    local_rank: RankId,
    // Mirrors the C++ member of the same name; not yet read by any ported
    // method (the C++ class keeps it for bounds-checking peer ranks in
    // methods outside this port's current scope).
    #[allow(dead_code)]
    world_size: u32,
    rdma_enabled: bool,
    bc_list: BitmaskHelper,
    shm: *mut ffi::SenlibRdmaShm,
}

impl RdmaUnit {
    /// Source: rdma_unit.cpp (constructor establishing PID rendezvous over shm) +
    /// rdma_unit.cpp:340 `rdma_shm_ = std::make_shared<RdmaShm>(local_rank_)`.
    pub fn new(local_rank: RankId, world_size: u32, rdma_enabled: bool) -> Self {
        let shm = if rdma_enabled {
            // SAFETY: opens/creates the mock-RDMA shm region for this rank via
            // the senlib-adjacent device layer; ownership is released in Drop.
            unsafe { ffi::flex_senlib_rdma_shm_open(i64::from(local_rank.0)) }
        } else {
            std::ptr::null_mut()
        };
        Self {
            local_rank,
            world_size,
            rdma_enabled,
            bc_list: BitmaskHelper::new(),
            shm,
        }
    }

    /// Source: rdma_unit.cpp:421-426 `RdmaUnit::SetBcList`.
    pub fn set_bc_list(&mut self, bc_list: BitmaskHelper) {
        self.bc_list = bc_list;
    }

    /// Source: rdma_unit.cpp:504-531 `RdmaUnit::WriteXseg`.
    ///
    /// SEMANTIC DIFFERENCE FROM REAL C++ (flagged for future wiring, not
    /// fixed here): real `WriteXseg` takes a caller-supplied, already
    /// wire-encoded `uint32_t xseg` (parity bit included) and *validates*
    /// that its parity bit matches the recomputed parity of
    /// `xseg & (XsegValidMask | XsegAddrMask)`, returning
    /// `INVALID_ARGUMENT` on mismatch (rdma_unit.cpp:512-519) rather than
    /// ever computing/overwriting that bit itself. This Rust port instead
    /// always self-computes the parity bit from `xseg.bytes`/`xseg.valid`
    /// and writes a value that is correct by construction -- it can never
    /// observe or reject a caller-supplied bad-parity encoding the way the
    /// real function can. This function currently has zero callers in this
    /// crate (confirmed dead code), so the implementation is left as-is
    /// rather than risk changing behavior with no real caller to verify
    /// against; when this gets wired up to a real caller, decide then
    /// whether `Xseg` should instead carry a pre-encoded `u32` (with parity)
    /// for this function to validate, matching real C++ exactly.
    pub fn write_xseg(
        &mut self,
        target: RankId,
        sid: SegId,
        xseg: Xseg,
    ) -> Result<(), RdmaUnitError> {
        if !self.rdma_enabled {
            return Err(RdmaUnitError::RdmaNotEnabled);
        }
        if target == self.local_rank {
            return Err(RdmaUnitError::LocalRankTarget);
        }
        let raw =
            (xseg.bytes as u32 & XSEG_ADDR_MASK) | if xseg.valid { XSEG_VALID_BIT } else { 0 };
        let should_parity = (raw & (XSEG_VALID_BIT | XSEG_ADDR_MASK)).count_ones() % 2 == 1;
        let parity = should_parity; // by construction we always write correct parity natively
        let wire = raw | if parity { XSEG_PARITY_BIT } else { 0 };
        // SAFETY: shm handle is non-null whenever rdma_enabled (checked above),
        // and outlives this call for the lifetime of `self`.
        unsafe { ffi::flex_senlib_rdma_xseg_write(self.shm, target.0, u32::from(sid.0), wire) };
        Ok(())
    }

    /// Source: rdma_unit.cpp:382-419 `RdmaUnit::CopyToRemote`.
    pub fn copy_to_remote(
        &mut self,
        target: RankId,
        local_ptr: &[u8],
    ) -> Result<(), RdmaUnitError> {
        if !self.rdma_enabled {
            return Err(RdmaUnitError::RdmaNotEnabled);
        }
        if local_ptr.len() as u64 > MAX_HDMA_MESSAGE_BYTES {
            return Err(RdmaUnitError::MessageTooLarge {
                size: local_ptr.len() as u64,
                max: MAX_HDMA_MESSAGE_BYTES,
            });
        }
        if target == self.local_rank {
            return Err(RdmaUnitError::LocalRankTarget);
        }
        let sid = SegId(self.local_rank.0 as u8);
        // SAFETY: shm handle validated by rdma_enabled above.
        let valid =
            unsafe { ffi::flex_senlib_rdma_xseg_valid(self.shm, target.0, u32::from(sid.0)) };
        if !valid {
            return Err(RdmaUnitError::RemoteXsegNotValid { target, sid });
        }
        let xlat =
            unsafe { ffi::flex_senlib_rdma_xseg_bytes(self.shm, target.0, u32::from(sid.0)) };
        // SAFETY: local_ptr is a valid slice for its stated length; the FFI
        // callee performs the literal device-memory copy (rdma_unit.cpp:416).
        let rc = unsafe {
            ffi::flex_senlib_rdma_copy_to_remote(
                self.shm,
                target.0,
                xlat,
                local_ptr.as_ptr(),
                local_ptr.len(),
            )
        };
        if rc != 0 {
            return Err(RdmaUnitError::HardwareError(rc));
        }
        Ok(())
    }

    /// Source: rdma_unit.cpp:486-502 `RdmaUnit::SendWriteDone`.
    pub fn send_write_done(&mut self, target: RankId, sid: SegId) -> Result<i64, RdmaUnitError> {
        if !self.rdma_enabled {
            return Err(RdmaUnitError::RdmaNotEnabled);
        }
        if target == self.local_rank {
            return Err(RdmaUnitError::LocalRankTarget);
        }
        // SAFETY: shm handle validated by rdma_enabled above.
        Ok(unsafe { ffi::flex_senlib_rdma_wdone_inc(self.shm, target.0, u32::from(sid.0)) })
    }

    /// Source: rdma_unit.cpp:428-441 `RdmaUnit::MultiCast`.
    pub fn multicast(
        &mut self,
        sid: SegId,
        local_ptr: &[u8],
        is_last: bool,
    ) -> Result<(), RdmaUnitError> {
        let mut mask = self.bc_list;
        loop {
            let target = mask.next_rank();
            if target.0 == 64 {
                break;
            }
            self.copy_to_remote(target, local_ptr)?;
            if is_last {
                self.send_write_done(target, sid)?;
            }
        }
        Ok(())
    }

    /// Source: rdma_unit.cpp:443-484 `RdmaUnit::WaitForBarrier`.
    ///
    /// `timeout` is a CUMULATIVE deadline shared across the entire
    /// `wait_list`, not a per-`sid` budget: real C++ creates a single
    /// `EveryNSeconds every_n_seconds` timer before the `for(auto sid :
    /// wait_list)` loop (rdma_unit.cpp:446, outside the loop that starts at
    /// rdma_unit.cpp:448), so `timeout_msecs` is measured against elapsed
    /// time since entering `WaitForBarrier` as a whole, not reset per `sid`.
    /// The deadline is therefore computed once here, before the loop, to
    /// match that exactly -- computing it fresh inside the loop would give
    /// each `sid` its own full timeout budget (up to `wait_list.len()` times
    /// the intended total).
    pub fn wait_for_barrier(
        &mut self,
        wait_list: &[SegId],
        timeout: Duration,
    ) -> Result<(), RdmaUnitError> {
        if !self.rdma_enabled {
            return Err(RdmaUnitError::RdmaNotEnabled);
        }
        let tid = self.local_rank.0;
        let deadline = Instant::now() + timeout;
        for sid in wait_list {
            loop {
                // SAFETY: shm handle validated by rdma_enabled above.
                let val =
                    unsafe { ffi::flex_senlib_rdma_wdone_check(self.shm, tid, u32::from(sid.0)) };
                match classify_wdone_check(val) {
                    WdoneCheckOutcome::HardwareError(code) => {
                        return Err(RdmaUnitError::HardwareError(code));
                    }
                    WdoneCheckOutcome::Ready => break,
                    // Port of a real flex-cxx bug fix (see
                    // `rdma_negative_counter_test.cpp`): a peer rank's
                    // destructor sets its write-done counter to -1 on exit,
                    // and that must be detected as `WriteDoneWrapped`
                    // immediately -- NOT fall through to the timeout loop.
                    // `classify_wdone_check`'s own tests are this crate's
                    // regression coverage for that: unlike the described
                    // bug (`if(val < 0)` nested inside `if(val > 0)`,
                    // making it unreachable), this match's `Wrapped` arm is
                    // a sibling of `Ready`, not nested inside it.
                    WdoneCheckOutcome::Wrapped => return Err(RdmaUnitError::WriteDoneWrapped),
                    WdoneCheckOutcome::NotYetReady => {}
                }
                if Instant::now() >= deadline {
                    return Err(RdmaUnitError::Timeout);
                }
                // Intentional deviation from real C++: rdma_unit.cpp's poll
                // loop busy-spins with no sleep between `WriteDoneCounterCheck`
                // calls. This port sleeps 1ms per iteration to reduce CPU usage;
                // it does not change the cumulative-deadline semantics above.
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        Ok(())
    }
}

impl Drop for RdmaUnit {
    fn drop(&mut self) {
        if !self.shm.is_null() {
            // SAFETY: `self.shm` was obtained from `flex_senlib_rdma_shm_open`
            // in `new` and has not been freed elsewhere; releasing it here is
            // the only safe place to do so given single ownership by `self`.
            unsafe { ffi::flex_senlib_rdma_shm_close(self.shm) };
        }
    }
}

/// Ported (adapted) C++ test, source:
/// `flex/tests/multi_device/rdma_negative_counter_test.cpp`
/// (`RdmaUnitNegativeCounter.DetectPeerExit`).
///
/// The real test requires exactly 2 live RDMA ranks (`GTEST_SKIP()`s
/// otherwise, per its own body) and asserts that `WaitForBarrier` detects a
/// peer's exit-triggered `-1` write-done counter within milliseconds rather
/// than looping until its full timeout. That end-to-end shape is
/// fundamentally untestable in this crate's harness: `RdmaUnit` talks to a
/// real shared-memory segment through `senlib_ffi_multi_device`, and there
/// is no mock/fake for it here (unlike `control_blocks.rs`'s
/// software-simulated `MockDeviceMemory`) -- exercising it would need a
/// second real process and real senlib RDMA shared memory, same as the C++
/// test's own environment gate.
///
/// What IS portable, and is the actual bug the C++ test guards against, is
/// the branch structure that decides what a negative counter means:
/// `classify_wdone_check` above is that logic, extracted out of the
/// FFI-calling loop specifically so it can be pinned here. Verified while
/// porting: `wait_for_barrier`'s branch structure in this crate was already
/// correct (`if val > 0 {break} ... else if val < 0 {return Err}` as
/// siblings) -- the described flex-cxx bug (`val < 0` nested inside `val >
/// 0`, making it unreachable) was never present in this port. This module
/// closes the actual gap the audit found: that invariant had no test of its
/// own, so a future refactor could silently reintroduce the nesting bug
/// without any test noticing.
#[cfg(test)]
mod wdone_check_classification_tests {
    use super::*;

    /// The exact scenario `DetectPeerExit` exercises end-to-end: a peer
    /// rank's `RdmaUnit` destructor sets its counter to -1 on exit. Must be
    /// classified as `Wrapped` (surfaced as `RdmaUnitError::WriteDoneWrapped`
    /// by `wait_for_barrier`) immediately -- if this were misclassified as
    /// `NotYetReady`, `wait_for_barrier` would loop until its full timeout
    /// instead, exactly the regression `DetectPeerExit` was written to catch.
    #[test]
    fn negative_counter_is_wrapped_not_not_yet_ready() {
        assert_eq!(classify_wdone_check(-1), WdoneCheckOutcome::Wrapped);
        assert_eq!(classify_wdone_check(i64::MIN), WdoneCheckOutcome::Wrapped);
    }

    #[test]
    fn positive_counter_is_ready() {
        assert_eq!(classify_wdone_check(1), WdoneCheckOutcome::Ready);
        assert_eq!(classify_wdone_check(42), WdoneCheckOutcome::Ready);
    }

    #[test]
    fn zero_counter_is_not_yet_ready() {
        assert_eq!(classify_wdone_check(0), WdoneCheckOutcome::NotYetReady);
    }

    /// Top byte set (`val >> 24 > 0`) is a hardware error code, checked
    /// BEFORE the sign-based branches -- a value can be positive (by plain
    /// integer comparison) yet still carry an error code in its top byte, so
    /// this check must win regardless of whether `val` would otherwise read
    /// as `Ready`.
    #[test]
    fn top_byte_set_is_hardware_error_even_when_positive() {
        let val = 0x02_0000_0001i64; // top byte = 2, but val > 0
        assert_eq!(
            classify_wdone_check(val),
            WdoneCheckOutcome::HardwareError(val as i32)
        );
    }
}

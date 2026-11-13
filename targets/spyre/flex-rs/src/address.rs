//! Port of `flex/include/flex/allocator/allocation_address.hpp`.
//!
//! `LogicalAddress`, `Chunk`, `CompositeAddress` are pure value/bookkeeping types —
//! no senlib involvement anywhere in this file.

use std::fmt;

use crate::memory_region::DomainId;

/// Region-relative identifier for a `MemoryRegion` (VF firmware table index, or PF
/// physical region-start address, depending on device mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegionId(pub u64);

/// Byte offset within a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ByteOffset(pub u64);

/// A generic byte size / count, used pervasively for allocation sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ByteSize(pub u64);

impl ByteSize {
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

/// Device-side location: region + byte offset. Mirrors `flex::LogicalAddress`.
///
/// `Hash` is hand-implemented (not derived) below, delegating to the custom
/// `.hash()` method that ports the C++ `boost::hash_combine`-style algorithm
/// (`allocation_address.hpp:69-75`) — a `#[derive(Hash)]` here would use the
/// compiler's structural hash instead, silently bypassing that ported
/// algorithm for every `HashMap`/`HashSet` use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicalAddress {
    pub region_id: RegionId,
    pub offset: ByteOffset,
}

impl LogicalAddress {
    pub const fn new(region_id: RegionId, offset: ByteOffset) -> Self {
        Self { region_id, offset }
    }

    /// Port of `LogicalAddress::hash()` (`allocation_address.hpp:69`): a
    /// `boost::hash_combine`-style mix of `region_id` and `offset`, seeded
    /// with `std::hash<uint64_t>` — libstdc++ implements that as identity for
    /// 64-bit integral types, hence `region_id.0` used directly as the seed.
    pub fn hash(&self) -> u64 {
        let seed: u64 = self.region_id.0;
        seed ^ self
            .offset
            .0
            .wrapping_add(0x9e3779b9)
            .wrapping_add(seed << 6)
            .wrapping_add(seed >> 2)
    }
}

impl std::hash::Hash for LogicalAddress {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Delegate to the ported `boost::hash_combine` algorithm above,
        // rather than a `#[derive(Hash)]` structural hash — see the struct
        // doc comment. `std::hash::Hash::hash` has no return value, so feed
        // the resulting `u64` through as a single write.
        state.write_u64(self.hash());
    }
}

impl fmt::Display for LogicalAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LogicalAddress{{region_id=0x{:x}, offset=0x{:x}}}",
            self.region_id.0, self.offset.0
        )
    }
}

/// A contiguous memory chunk within a `CompositeAddress`. Mirrors `flex::Chunk`.
///
/// `Chunk` has no C++ `.hash()` of its own (only `LogicalAddress` and
/// `CompositeAddress` do), so a plain derived `Hash` — consistent with
/// derived `PartialEq`/`Eq` — is correct here; it delegates field-by-field to
/// `LogicalAddress`'s hand-written `Hash` impl (which itself matches the
/// ported algorithm), so there is no bypass concern for this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Chunk {
    pub addr: LogicalAddress,
    pub size: ByteSize,
    pub domain_id: DomainId,
}

impl Chunk {
    pub const fn new(addr: LogicalAddress, size: ByteSize, domain_id: DomainId) -> Self {
        Self {
            addr,
            size,
            domain_id,
        }
    }
}

impl fmt::Display for Chunk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Chunk{{addr={}, size={}, domain_id={}}}",
            self.addr, self.size.0, self.domain_id.0
        )
    }
}

/// Device-side location of an allocation: one chunk for simple allocations,
/// multiple chunks for interleaved (NUMA-striped) allocations.
///
/// The C++ `CompositeAddress` is a move-only RAII owner that auto-deallocates
/// via a raw `FlexAllocator*` back-pointer. This port does not reproduce that
/// self-deallocating ownership: `CompositeAddress` here is a plain,
/// `Clone`-able value (used as the allocation-map key and for read access),
/// and freeing is the caller's/allocator's own explicit responsibility —
/// there is no RAII-owning counterpart type anywhere in this crate.
///
/// `Hash` is hand-implemented (not derived) below, delegating to the custom
/// `.hash()` method that ports the C++ `boost::hash_combine`-style algorithm
/// (`allocation_address.hpp:281-295`) — see `LogicalAddress`'s doc comment
/// for why a derive here would be wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeAddress {
    chunks: ChunkList,
    /// Port of `cached_total_size_` (`allocation_address.hpp:311`): computed
    /// once at construction (`allocation_address.hpp:184,187`) and returned
    /// O(1) by `total_size()`, rather than resummed on every call.
    total_size: ByteSize,
}

/// Single-chunk vs multi-chunk (interleaved) is a real distinction the C++
/// callers branch on (`is_single_chunk()`), so it is encoded as an enum
/// invariant rather than "a `Vec` that happens to have length 1". An empty
/// `Interleaved(vec![])` is a valid (if degenerate) state, matching the real
/// `std::vector<Chunk>` constructor's total lack of an empty-input check
/// (`allocation_address.hpp:186-189`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ChunkList {
    Single(Chunk),
    Interleaved(Vec<Chunk>),
}

impl CompositeAddress {
    /// Construct a single-chunk `CompositeAddress`.
    pub fn from_chunk(chunk: Chunk) -> Self {
        Self {
            chunks: ChunkList::Single(chunk),
            total_size: chunk.size,
        }
    }

    /// Construct an interleaved `CompositeAddress`.
    ///
    /// Port of `explicit CompositeAddress(const std::vector<Chunk>&
    /// chunk_list)` (`allocation_address.hpp:186-189`): the real `vector`
    /// constructor performs no empty-input validation at all — an empty
    /// `chunk_list` simply yields a zero-chunk, zero-size address. A
    /// previous version of this port `assert!`ed (panicked) on empty input,
    /// which has no C++ counterpart and turns a reachable, data-controlled
    /// input into a crash; this now matches the C++ by accepting it.
    pub fn from_chunks(chunks: Vec<Chunk>) -> Self {
        // Port of `computeTotalSize` (`allocation_address.hpp:347-355`):
        // `size_t total = 0; for(...) total += chunk.size;` — an unchecked
        // `size_t` accumulation, i.e. modulo 2^64 on wrap. `wrapping_add`
        // reproduces that exactly. (A previous version used `saturating_add`,
        // which clamps at `u64::MAX` and so reports a different total_size
        // than the C++ for a wrapping chunk list; `Sum`/`+` would instead
        // panic in debug builds. Neither has a C++ counterpart.)
        let total_size = ByteSize(
            chunks
                .iter()
                .fold(0u64, |acc, c| acc.wrapping_add(c.size.as_u64())),
        );
        if chunks.len() == 1 {
            Self {
                chunks: ChunkList::Single(chunks.into_iter().next().unwrap()),
                total_size,
            }
        } else {
            Self {
                chunks: ChunkList::Interleaved(chunks),
                total_size,
            }
        }
    }

    pub fn chunks(&self) -> &[Chunk] {
        match &self.chunks {
            ChunkList::Single(c) => std::slice::from_ref(c),
            ChunkList::Interleaved(v) => v,
        }
    }

    /// Port of the public `total_size()` accessor (`allocation_address.hpp:238`):
    /// O(1), returning the value cached at construction — NOT the private
    /// construction-time helper `computeTotalSize()` it was previously
    /// mis-cited as porting. Resumming on every call (the prior behavior)
    /// diverges from the real O(1) accessor and can panic on overflow; both
    /// are avoided by caching the (wrapping, per the C++) sum in `from_chunk`/`from_chunks`.
    pub fn total_size(&self) -> ByteSize {
        self.total_size
    }

    pub fn num_chunks(&self) -> usize {
        self.chunks().len()
    }

    pub fn is_single_chunk(&self) -> bool {
        matches!(self.chunks, ChunkList::Single(_))
    }

    /// Port of the private `nonOwningCopy()` — in Rust this is just a `Clone`,
    /// since `CompositeAddress` never owns senlib resources itself (see module doc).
    pub(crate) fn non_owning_copy(&self) -> Self {
        self.clone()
    }

    /// Port of `CompositeAddress::hash()` (`allocation_address.hpp:281`):
    /// `boost::hash_combine`-style mix over each chunk's address hash, size,
    /// and domain_id, seeded at 0. `std::hash<size_t>`/`std::hash<uint32_t>`
    /// are libstdc++ identity hashes for integral types.
    pub fn hash(&self) -> u64 {
        let mut seed: u64 = 0;
        for chunk in self.chunks() {
            let chunk_hash = chunk.addr.hash();
            seed ^= chunk_hash
                .wrapping_add(0x9e3779b9)
                .wrapping_add(seed << 6)
                .wrapping_add(seed >> 2);
            seed ^= chunk
                .size
                .as_u64()
                .wrapping_add(0x9e3779b9)
                .wrapping_add(seed << 6)
                .wrapping_add(seed >> 2);
            seed ^= u64::from(chunk.domain_id.0)
                .wrapping_add(0x9e3779b9)
                .wrapping_add(seed << 6)
                .wrapping_add(seed >> 2);
        }
        seed
    }
}

impl std::hash::Hash for CompositeAddress {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Delegate to the ported algorithm above rather than deriving over
        // `chunks`/`total_size` — see the struct doc comment. Note this
        // intentionally hashes only `chunks_` (via `.hash()`), matching the
        // C++ comment "operator== and hash() consider only chunks_ —
        // ownership is not part of address identity" (`allocation_address.hpp:174`).
        state.write_u64(self.hash());
    }
}

impl fmt::Display for CompositeAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CompositeAddress{{total_size={}, num_chunks={}, chunks=[",
            self.total_size().0,
            self.num_chunks()
        )?;
        for (i, chunk) in self.chunks().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{chunk}")?;
        }
        write!(f, "]}}")
    }
}

/// Port of `flex/tests/allocator/data_structures/allocation_address_test.cpp`.
///
/// NOT ported: `CompositeAddressTest.StreamOutput*`/`MoveConstructorTransfers
/// OwnershipStateForNonOwningAddress`/`MoveAssignmentReplacesContentsForNon
/// OwningAddress`/`SwapExchangesContentsWithoutChangingOwnershipForNonOwning
/// Addresses` — the C++ `CompositeAddress` is a move-only RAII type carrying
/// its own `owns_allocation()`/`FlexAllocator*` back-pointer and prints
/// `owns=<bool>` in its `operator<<`; this port's `CompositeAddress` (see
/// module doc above) is a plain `Clone` value with no ownership flag or
/// self-deallocating RAII counterpart at all, so there is no
/// `owns_allocation()` to assert on and `Display` never prints an `owns=`
/// field. Porting these verbatim would require inventing that field, which
/// the C++ source does not describe.
#[cfg(test)]
mod ported_cxx_tests {
    use super::*;

    // ---- LogicalAddress ----

    #[test]
    fn logical_address_construction_with_valid_values() {
        let addr = LogicalAddress::new(RegionId(42), ByteOffset(1280));
        assert_eq!(addr.region_id, RegionId(42));
        assert_eq!(addr.offset, ByteOffset(1280));
    }

    #[test]
    fn logical_address_construction_with_zero_values() {
        let addr = LogicalAddress::new(RegionId(0), ByteOffset(0));
        assert_eq!(addr.region_id, RegionId(0));
        assert_eq!(addr.offset, ByteOffset(0));
    }

    #[test]
    fn logical_address_construction_with_max_values() {
        let addr = LogicalAddress::new(RegionId(u64::MAX), ByteOffset(u64::MAX));
        assert_eq!(addr.region_id, RegionId(u64::MAX));
        assert_eq!(addr.offset, ByteOffset(u64::MAX));
    }

    #[test]
    fn logical_address_equality_with_same_fields() {
        let addr1 = LogicalAddress::new(RegionId(100), ByteOffset(256));
        let addr2 = LogicalAddress::new(RegionId(100), ByteOffset(256));
        assert_eq!(addr1, addr2);
    }

    #[test]
    fn logical_address_inequality_with_different_region_id() {
        let addr1 = LogicalAddress::new(RegionId(100), ByteOffset(256));
        let addr2 = LogicalAddress::new(RegionId(200), ByteOffset(256));
        assert_ne!(addr1, addr2);
    }

    #[test]
    fn logical_address_inequality_with_different_offset() {
        let addr1 = LogicalAddress::new(RegionId(100), ByteOffset(256));
        let addr2 = LogicalAddress::new(RegionId(100), ByteOffset(512));
        assert_ne!(addr1, addr2);
    }

    #[test]
    fn logical_address_hash_consistency() {
        let addr = LogicalAddress::new(RegionId(42), ByteOffset(1280));
        assert_eq!(addr.hash(), addr.hash());
    }

    #[test]
    fn logical_address_equal_addresses_produce_equal_hashes() {
        let addr1 = LogicalAddress::new(RegionId(42), ByteOffset(1280));
        let addr2 = LogicalAddress::new(RegionId(42), ByteOffset(1280));
        assert_eq!(addr1.hash(), addr2.hash());
    }

    #[test]
    fn logical_address_different_addresses_produce_different_hashes() {
        let addr1 = LogicalAddress::new(RegionId(42), ByteOffset(1280));
        let addr2 = LogicalAddress::new(RegionId(43), ByteOffset(1280));
        let addr3 = LogicalAddress::new(RegionId(42), ByteOffset(1408));
        assert_ne!(addr1.hash(), addr2.hash());
        assert_ne!(addr1.hash(), addr3.hash());
        assert_ne!(addr2.hash(), addr3.hash());
    }

    #[test]
    fn logical_address_use_as_hashmap_key() {
        use std::collections::HashMap;
        let addr1 = LogicalAddress::new(RegionId(1), ByteOffset(128));
        let addr2 = LogicalAddress::new(RegionId(2), ByteOffset(256));
        let addr3 = LogicalAddress::new(RegionId(3), ByteOffset(384));
        let mut map = HashMap::new();
        map.insert(addr1, 100);
        map.insert(addr2, 200);
        map.insert(addr3, 300);
        assert_eq!(map.len(), 3);
        assert_eq!(map[&LogicalAddress::new(RegionId(1), ByteOffset(128))], 100);
    }

    #[test]
    fn logical_address_use_in_hashset() {
        use std::collections::HashSet;
        let addr1 = LogicalAddress::new(RegionId(1), ByteOffset(128));
        let addr2 = LogicalAddress::new(RegionId(2), ByteOffset(256));
        let addr3 = LogicalAddress::new(RegionId(1), ByteOffset(128)); // duplicate of addr1
        let mut set = HashSet::new();
        set.insert(addr1);
        set.insert(addr2);
        set.insert(addr3);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn logical_address_extreme_boundary_hashes_are_stable_and_usable_in_hashset() {
        use std::collections::HashSet;
        let zero_addr = LogicalAddress::new(RegionId(0), ByteOffset(0));
        let max_addr = LogicalAddress::new(RegionId(u64::MAX), ByteOffset(u64::MAX));
        assert_eq!(zero_addr.hash(), zero_addr.hash());
        assert_eq!(max_addr.hash(), max_addr.hash());
        let set: HashSet<LogicalAddress> = HashSet::from([zero_addr, max_addr]);
        assert_eq!(set.len(), 2);
        assert!(set.contains(&LogicalAddress::new(RegionId(0), ByteOffset(0))));
        assert!(set.contains(&LogicalAddress::new(
            RegionId(u64::MAX),
            ByteOffset(u64::MAX)
        )));
    }

    #[test]
    fn logical_address_stream_output_produces_expected_format() {
        let addr = LogicalAddress::new(RegionId(0xAB), ByteOffset(0xCD));
        assert_eq!(
            format!("{addr}"),
            "LogicalAddress{region_id=0xab, offset=0xcd}"
        );
    }

    // ---- Chunk ----

    #[test]
    fn chunk_construction_with_valid_values() {
        let addr = LogicalAddress::new(RegionId(7), ByteOffset(128));
        let chunk = Chunk::new(addr, ByteSize(1024), DomainId(0));
        assert_eq!(chunk.addr, addr);
        assert_eq!(chunk.size, ByteSize(1024));
        assert_eq!(chunk.domain_id, DomainId(0));
    }

    #[test]
    fn chunk_construction_with_zero_size() {
        let addr = LogicalAddress::new(RegionId(7), ByteOffset(256));
        let chunk = Chunk::new(addr, ByteSize(0), DomainId(1));
        assert_eq!(chunk.size, ByteSize(0));
        assert_eq!(chunk.domain_id, DomainId(1));
    }

    #[test]
    fn chunk_equality_with_same_fields() {
        let addr = LogicalAddress::new(RegionId(7), ByteOffset(256));
        let chunk1 = Chunk::new(addr, ByteSize(2048), DomainId(1));
        let chunk2 = Chunk::new(addr, ByteSize(2048), DomainId(1));
        assert_eq!(chunk1, chunk2);
    }

    #[test]
    fn chunk_inequality_with_different_address() {
        let addr1 = LogicalAddress::new(RegionId(7), ByteOffset(256));
        let addr2 = LogicalAddress::new(RegionId(8), ByteOffset(256));
        let chunk1 = Chunk::new(addr1, ByteSize(2048), DomainId(1));
        let chunk2 = Chunk::new(addr2, ByteSize(2048), DomainId(1));
        assert_ne!(chunk1, chunk2);
    }

    #[test]
    fn chunk_inequality_with_different_size() {
        let addr = LogicalAddress::new(RegionId(7), ByteOffset(256));
        let chunk1 = Chunk::new(addr, ByteSize(2048), DomainId(1));
        let chunk2 = Chunk::new(addr, ByteSize(4096), DomainId(1));
        assert_ne!(chunk1, chunk2);
    }

    #[test]
    fn chunk_inequality_with_different_domain_id() {
        let addr = LogicalAddress::new(RegionId(7), ByteOffset(256));
        let chunk1 = Chunk::new(addr, ByteSize(2048), DomainId(1));
        let chunk2 = Chunk::new(addr, ByteSize(2048), DomainId(2));
        assert_ne!(chunk1, chunk2);
    }

    #[test]
    fn chunk_stream_output_produces_expected_format() {
        let chunk = Chunk::new(
            LogicalAddress::new(RegionId(0x10), ByteOffset(0x20)),
            ByteSize(4096),
            DomainId(2),
        );
        let result = format!("{chunk}");
        assert!(result.contains("Chunk{addr="));
        assert!(result.contains("size=4096"));
        assert!(result.contains("domain_id=2"));
    }

    // ---- CompositeAddress: single-chunk ----

    #[test]
    fn composite_single_chunk_construction_is_contiguous() {
        let addr = LogicalAddress::new(RegionId(7), ByteOffset(128));
        let chunk = Chunk::new(addr, ByteSize(1024 * 1024), DomainId(0));
        let composite = CompositeAddress::from_chunk(chunk);
        assert!(composite.is_single_chunk());
        assert_eq!(composite.num_chunks(), 1);
        assert_eq!(composite.total_size(), ByteSize(1024 * 1024));
        assert_eq!(composite.chunks().len(), 1);
        assert_eq!(composite.chunks()[0], chunk);
    }

    #[test]
    fn composite_single_chunk_construction_with_zero_size_chunk() {
        let chunk = Chunk::new(
            LogicalAddress::new(RegionId(11), ByteOffset(0)),
            ByteSize(0),
            DomainId(2),
        );
        let composite = CompositeAddress::from_chunk(chunk);
        assert!(composite.is_single_chunk());
        assert_eq!(composite.num_chunks(), 1);
        assert_eq!(composite.total_size(), ByteSize(0));
    }

    #[test]
    fn composite_single_chunk_total_size() {
        let addr = LogicalAddress::new(RegionId(7), ByteOffset(256));
        let chunk_size = 2 * 1024 * 1024;
        let chunk = Chunk::new(addr, ByteSize(chunk_size), DomainId(0));
        let composite = CompositeAddress::from_chunk(chunk);
        assert_eq!(composite.total_size(), ByteSize(chunk_size));
    }

    #[test]
    fn composite_single_chunk_num_chunks() {
        let addr = LogicalAddress::new(RegionId(7), ByteOffset(384));
        let chunk = Chunk::new(addr, ByteSize(512 * 1024), DomainId(0));
        let composite = CompositeAddress::from_chunk(chunk);
        assert_eq!(composite.num_chunks(), 1);
    }

    #[test]
    fn composite_single_chunk_construction_equivalence() {
        let addr = LogicalAddress::new(RegionId(7), ByteOffset(512));
        let chunk_size = 1024 * 1024;
        let chunk = Chunk::new(addr, ByteSize(chunk_size), DomainId(0));

        let composite1 = CompositeAddress::from_chunk(chunk);
        let composite2 = CompositeAddress::from_chunks(vec![chunk]);

        assert_eq!(composite1, composite2);
        assert_eq!(composite1.num_chunks(), composite2.num_chunks());
        assert_eq!(composite1.total_size(), composite2.total_size());
        assert_eq!(composite1.is_single_chunk(), composite2.is_single_chunk());
        assert!(composite1.is_single_chunk());
        assert_eq!(composite1.chunks()[0], composite2.chunks()[0]);
    }

    // ---- CompositeAddress: multi-chunk ----

    #[test]
    fn composite_multi_chunk_construction_is_not_contiguous() {
        let chunk1 = Chunk::new(
            LogicalAddress::new(RegionId(7), ByteOffset(128)),
            ByteSize(1024 * 1024),
            DomainId(0),
        );
        let chunk2 = Chunk::new(
            LogicalAddress::new(RegionId(8), ByteOffset(256)),
            ByteSize(1024 * 1024),
            DomainId(1),
        );
        let composite = CompositeAddress::from_chunks(vec![chunk1, chunk2]);
        assert!(!composite.is_single_chunk());
        assert_eq!(composite.num_chunks(), 2);
    }

    #[test]
    fn composite_multi_chunk_total_size_sum() {
        let size1 = 1024 * 1024;
        let size2 = 2 * 1024 * 1024;
        let size3 = 512 * 1024;
        let chunk1 = Chunk::new(
            LogicalAddress::new(RegionId(7), ByteOffset(128)),
            ByteSize(size1),
            DomainId(0),
        );
        let chunk2 = Chunk::new(
            LogicalAddress::new(RegionId(8), ByteOffset(256)),
            ByteSize(size2),
            DomainId(1),
        );
        let chunk3 = Chunk::new(
            LogicalAddress::new(RegionId(9), ByteOffset(384)),
            ByteSize(size3),
            DomainId(2),
        );
        let composite = CompositeAddress::from_chunks(vec![chunk1, chunk2, chunk3]);
        assert_eq!(composite.total_size(), ByteSize(size1 + size2 + size3));
    }

    #[test]
    fn composite_multi_chunk_num_chunks() {
        let chunks = vec![
            Chunk::new(
                LogicalAddress::new(RegionId(7), ByteOffset(128)),
                ByteSize(1024 * 1024),
                DomainId(0),
            ),
            Chunk::new(
                LogicalAddress::new(RegionId(8), ByteOffset(256)),
                ByteSize(1024 * 1024),
                DomainId(1),
            ),
            Chunk::new(
                LogicalAddress::new(RegionId(9), ByteOffset(384)),
                ByteSize(1024 * 1024),
                DomainId(2),
            ),
            Chunk::new(
                LogicalAddress::new(RegionId(10), ByteOffset(512)),
                ByteSize(1024 * 1024),
                DomainId(3),
            ),
        ];
        let composite = CompositeAddress::from_chunks(chunks);
        assert_eq!(composite.num_chunks(), 4);
    }

    #[test]
    fn composite_multi_chunk_chunks_ordering() {
        let chunk1 = Chunk::new(
            LogicalAddress::new(RegionId(7), ByteOffset(128)),
            ByteSize(1024 * 1024),
            DomainId(0),
        );
        let chunk2 = Chunk::new(
            LogicalAddress::new(RegionId(8), ByteOffset(256)),
            ByteSize(2 * 1024 * 1024),
            DomainId(1),
        );
        let chunk3 = Chunk::new(
            LogicalAddress::new(RegionId(9), ByteOffset(384)),
            ByteSize(512 * 1024),
            DomainId(2),
        );
        let composite = CompositeAddress::from_chunks(vec![chunk1, chunk2, chunk3]);
        let result_chunks = composite.chunks();
        assert_eq!(result_chunks.len(), 3);
        assert_eq!(result_chunks[0], chunk1);
        assert_eq!(result_chunks[1], chunk2);
        assert_eq!(result_chunks[2], chunk3);
    }

    #[test]
    fn composite_multi_chunk_construction_with_all_zero_size_chunks() {
        let chunks = vec![
            Chunk::new(
                LogicalAddress::new(RegionId(1), ByteOffset(0)),
                ByteSize(0),
                DomainId(0),
            ),
            Chunk::new(
                LogicalAddress::new(RegionId(2), ByteOffset(128)),
                ByteSize(0),
                DomainId(1),
            ),
            Chunk::new(
                LogicalAddress::new(RegionId(3), ByteOffset(256)),
                ByteSize(0),
                DomainId(2),
            ),
        ];
        let composite = CompositeAddress::from_chunks(chunks.clone());
        assert!(!composite.is_single_chunk());
        assert_eq!(composite.num_chunks(), 3);
        assert_eq!(composite.total_size(), ByteSize(0));
        assert_eq!(composite.chunks(), chunks.as_slice());
    }

    // ---- CompositeAddress: edge cases ----

    // Regression test: the real `explicit CompositeAddress(const
    // std::vector<Chunk>&)` constructor performs no empty-input validation
    // (`allocation_address.hpp:186-189`) — a previous version of this port
    // panicked on empty input instead, which has no C++ counterpart.
    #[test]
    fn from_chunks_with_empty_vector_does_not_panic() {
        let composite = CompositeAddress::from_chunks(vec![]);
        assert!(!composite.is_single_chunk());
        assert_eq!(composite.num_chunks(), 0);
        assert_eq!(composite.total_size(), ByteSize(0));
        assert!(composite.chunks().is_empty());
    }

    #[test]
    fn composite_equality_with_same_chunks() {
        let chunks = vec![
            Chunk::new(
                LogicalAddress::new(RegionId(7), ByteOffset(128)),
                ByteSize(1024 * 1024),
                DomainId(0),
            ),
            Chunk::new(
                LogicalAddress::new(RegionId(8), ByteOffset(256)),
                ByteSize(1024 * 1024),
                DomainId(1),
            ),
        ];
        let composite1 = CompositeAddress::from_chunks(chunks.clone());
        let composite2 = CompositeAddress::from_chunks(chunks);
        assert_eq!(composite1, composite2);
    }

    #[test]
    fn composite_inequality_with_different_chunks() {
        let chunk1 = Chunk::new(
            LogicalAddress::new(RegionId(7), ByteOffset(128)),
            ByteSize(1024 * 1024),
            DomainId(0),
        );
        let chunk2 = Chunk::new(
            LogicalAddress::new(RegionId(8), ByteOffset(256)),
            ByteSize(1024 * 1024),
            DomainId(1),
        );
        let chunk3 = Chunk::new(
            LogicalAddress::new(RegionId(8), ByteOffset(256)),
            ByteSize(2 * 1024 * 1024),
            DomainId(1),
        );
        let composite1 = CompositeAddress::from_chunks(vec![chunk1, chunk2]);
        let composite2 = CompositeAddress::from_chunks(vec![chunk1, chunk3]);
        assert_ne!(composite1, composite2);
    }

    #[test]
    fn composite_inequality_with_different_chunk_count() {
        let chunk1 = Chunk::new(
            LogicalAddress::new(RegionId(7), ByteOffset(128)),
            ByteSize(1024 * 1024),
            DomainId(0),
        );
        let chunk2 = Chunk::new(
            LogicalAddress::new(RegionId(8), ByteOffset(256)),
            ByteSize(1024 * 1024),
            DomainId(1),
        );
        let composite1 = CompositeAddress::from_chunk(chunk1);
        let composite2 = CompositeAddress::from_chunks(vec![chunk1, chunk2]);
        assert_ne!(composite1, composite2);
    }

    // ---- Realistic scenarios ----

    #[test]
    fn composite_realistic_single_region_allocation() {
        let addr = LogicalAddress::new(RegionId(7), ByteOffset(256));
        let tensor_size = 4 * 1024 * 1024;
        let chunk = Chunk::new(addr, ByteSize(tensor_size), DomainId(0));
        let composite = CompositeAddress::from_chunk(chunk);
        assert!(composite.is_single_chunk());
        assert_eq!(composite.num_chunks(), 1);
        assert_eq!(composite.total_size(), ByteSize(tensor_size));
        assert_eq!(composite.chunks()[0].addr.region_id, RegionId(7));
        assert_eq!(composite.chunks()[0].addr.offset, ByteOffset(256));
        assert_eq!(composite.chunks()[0].domain_id, DomainId(0));
    }

    #[test]
    fn composite_realistic_interleaved_allocation_four_domains() {
        let chunk_size = 2 * 1024 * 1024;
        let chunks: Vec<Chunk> = (0..4u32)
            .map(|i| {
                Chunk::new(
                    LogicalAddress::new(RegionId(7 + i as u64), ByteOffset(128 + 128 * i as u64)),
                    ByteSize(chunk_size),
                    DomainId(i),
                )
            })
            .collect();
        let composite = CompositeAddress::from_chunks(chunks);
        assert!(!composite.is_single_chunk());
        assert_eq!(composite.num_chunks(), 4);
        assert_eq!(composite.total_size(), ByteSize(8 * 1024 * 1024));
        for (i, chunk) in composite.chunks().iter().enumerate() {
            assert_eq!(chunk.domain_id, DomainId(i as u32));
            assert_eq!(chunk.size, ByteSize(chunk_size));
        }
    }

    // ---- Hash ----

    #[test]
    fn composite_hash_method_consistency() {
        let chunk = Chunk::new(
            LogicalAddress::new(RegionId(7), ByteOffset(128)),
            ByteSize(1024 * 1024),
            DomainId(0),
        );
        let composite = CompositeAddress::from_chunk(chunk);
        assert_eq!(composite.hash(), composite.hash());
    }

    #[test]
    fn composite_equal_addresses_produce_equal_hashes() {
        let chunks = vec![
            Chunk::new(
                LogicalAddress::new(RegionId(7), ByteOffset(128)),
                ByteSize(1024 * 1024),
                DomainId(0),
            ),
            Chunk::new(
                LogicalAddress::new(RegionId(8), ByteOffset(256)),
                ByteSize(1024 * 1024),
                DomainId(1),
            ),
        ];
        let composite1 = CompositeAddress::from_chunks(chunks.clone());
        let composite2 = CompositeAddress::from_chunks(chunks);
        assert_eq!(composite1.hash(), composite2.hash());
    }

    #[test]
    fn composite_different_addresses_produce_different_hashes() {
        let chunk1 = Chunk::new(
            LogicalAddress::new(RegionId(7), ByteOffset(128)),
            ByteSize(1024 * 1024),
            DomainId(0),
        );
        let chunk2 = Chunk::new(
            LogicalAddress::new(RegionId(8), ByteOffset(256)),
            ByteSize(1024 * 1024),
            DomainId(1),
        );
        let chunk3 = Chunk::new(
            LogicalAddress::new(RegionId(8), ByteOffset(256)),
            ByteSize(2 * 1024 * 1024),
            DomainId(1),
        );

        let composite1 = CompositeAddress::from_chunk(chunk1);
        let composite2 = CompositeAddress::from_chunks(vec![chunk1, chunk2]);
        let composite3 = CompositeAddress::from_chunks(vec![chunk1, chunk3]);

        assert_ne!(composite1.hash(), composite2.hash());
        assert_ne!(composite1.hash(), composite3.hash());
        assert_ne!(composite2.hash(), composite3.hash());
    }

    #[test]
    fn composite_multi_chunk_hash_includes_all_chunks() {
        let chunk1 = Chunk::new(
            LogicalAddress::new(RegionId(7), ByteOffset(128)),
            ByteSize(1024 * 1024),
            DomainId(0),
        );
        let chunk2 = Chunk::new(
            LogicalAddress::new(RegionId(8), ByteOffset(256)),
            ByteSize(1024 * 1024),
            DomainId(1),
        );
        let chunk3 = Chunk::new(
            LogicalAddress::new(RegionId(9), ByteOffset(384)),
            ByteSize(1024 * 1024),
            DomainId(2),
        );

        let composite_ab = CompositeAddress::from_chunks(vec![chunk1, chunk2]);
        let composite_abc = CompositeAddress::from_chunks(vec![chunk1, chunk2, chunk3]);

        assert_ne!(composite_ab.hash(), composite_abc.hash());
    }

    #[test]
    fn composite_stream_output_multi_chunk_includes_separator() {
        let chunks = vec![
            Chunk::new(
                LogicalAddress::new(RegionId(1), ByteOffset(0)),
                ByteSize(512),
                DomainId(0),
            ),
            Chunk::new(
                LogicalAddress::new(RegionId(2), ByteOffset(0)),
                ByteSize(512),
                DomainId(1),
            ),
        ];
        let composite = CompositeAddress::from_chunks(chunks);
        let result = format!("{composite}");
        assert!(result.contains("num_chunks=2"));
        let first_chunk_pos = result.find("Chunk{").unwrap();
        let second_chunk_pos = result[first_chunk_pos + 1..]
            .find("Chunk{")
            .map(|p| p + first_chunk_pos + 1)
            .unwrap();
        assert_eq!(&result[second_chunk_pos - 2..second_chunk_pos], ", ");
    }
}

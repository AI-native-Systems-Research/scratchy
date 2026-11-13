//! Port of `flex/include/flex/allocator/memory_domain.hpp`. Pure value types.

use crate::memory_region::DomainId;

/// A physical CPU core index with fast (local) access to a `MemoryDomain`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CoreId(pub u32);

/// Port of `flex::MemoryDomain`: a group of regions sharing a physical
/// characteristic (e.g. all backed by the same HBM stack).
#[derive(Debug, Clone)]
pub struct MemoryDomain {
    pub domain_id: DomainId,
    pub total_bytes: u64,
    pub core_affinity: Vec<CoreId>,
}

impl MemoryDomain {
    pub fn new(domain_id: DomainId, total_bytes: u64, core_affinity: Vec<CoreId>) -> Self {
        Self {
            domain_id,
            total_bytes,
            core_affinity,
        }
    }
}

/// Port of `MemoryDomain::operator==`: equality depends only on `domain_id`,
/// not on `total_bytes`/`core_affinity` (`memory_domain.hpp:58`).
impl PartialEq for MemoryDomain {
    fn eq(&self, other: &Self) -> bool {
        self.domain_id == other.domain_id
    }
}
impl Eq for MemoryDomain {}

/// Port of `flex::PlacementPolicy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlacementPolicy {
    /// Hard constraint: must succeed on the specified domain(s) or fail entirely.
    #[default]
    Bind,
    /// Stripe the allocation evenly across the specified domains, round-robin.
    Interleave,
}

/// Port of `operator<<(std::ostream&, PlacementPolicy)` (`memory_domain.hpp:99-108`).
/// The C++ enum's `default:` "Unknown(N)" branch has no Rust equivalent: this
/// enum is closed (no numeric escape hatch like a raw `static_cast<PlacementPolicy>(55)`),
/// so only the two named variants are reachable here.
impl std::fmt::Display for PlacementPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bind => write!(f, "Bind"),
            Self::Interleave => write!(f, "Interleave"),
        }
    }
}

#[cfg(test)]
mod placement_policy_display_tests {
    use super::*;

    // Port of PlacementPolicyStreamTest.BindPrintsCorrectly
    // (memory_domain_test.cpp / placement_policy_test.cpp).
    #[test]
    fn bind_prints_correctly() {
        assert_eq!(PlacementPolicy::Bind.to_string(), "Bind");
    }

    // Port of PlacementPolicyStreamTest.InterleavePrintsCorrectly.
    #[test]
    fn interleave_prints_correctly() {
        assert_eq!(PlacementPolicy::Interleave.to_string(), "Interleave");
    }

    // PlacementPolicyStreamTest.UnknownValuePrintsWithNumericCode is NOT
    // ported: it constructs an out-of-range enum value via
    // `static_cast<PlacementPolicy>(42)`, which is legal for a C++ enum but
    // has no equivalent for this Rust `enum` (a closed, exhaustively-matched
    // type with no numeric escape hatch).
}

/// Port of `flex/tests/allocator/data_structures/memory_domain_test.cpp`.
#[cfg(test)]
mod memory_domain_tests {
    use super::*;

    #[test]
    fn constructor_initializes_correctly() {
        let domain = MemoryDomain::new(
            DomainId(42),
            1024,
            vec![CoreId(0), CoreId(1), CoreId(2), CoreId(3)],
        );
        assert_eq!(domain.domain_id, DomainId(42));
        assert_eq!(domain.total_bytes, 1024);
        assert_eq!(
            domain.core_affinity,
            vec![CoreId(0), CoreId(1), CoreId(2), CoreId(3)]
        );
    }

    #[test]
    fn constructor_accepts_zero_total_bytes() {
        let domain = MemoryDomain::new(DomainId(7), 0, vec![CoreId(1), CoreId(3)]);
        assert_eq!(domain.domain_id, DomainId(7));
        assert_eq!(domain.total_bytes, 0);
        assert_eq!(domain.core_affinity, vec![CoreId(1), CoreId(3)]);
    }

    #[test]
    fn constructor_accepts_empty_core_affinity() {
        let domain = MemoryDomain::new(DomainId(5), 4096, vec![]);
        assert_eq!(domain.domain_id, DomainId(5));
        assert_eq!(domain.total_bytes, 4096);
        assert!(domain.core_affinity.is_empty());
    }

    #[test]
    fn constructor_accepts_max_values() {
        let domain = MemoryDomain::new(
            DomainId(u32::MAX),
            u64::MAX,
            vec![CoreId(0), CoreId(u32::MAX)],
        );
        assert_eq!(domain.domain_id, DomainId(u32::MAX));
        assert_eq!(domain.total_bytes, u64::MAX);
        assert_eq!(domain.core_affinity, vec![CoreId(0), CoreId(u32::MAX)]);
    }

    // Port of `MemoryDomainTest.EqualityDependsOnlyOnDomainId`: real C++
    // `operator==` compares only `domain_id` (`memory_domain.hpp:58`).
    #[test]
    fn equality_depends_only_on_domain_id() {
        let a = MemoryDomain::new(DomainId(9), 1024, vec![CoreId(0), CoreId(1)]);
        let b = MemoryDomain::new(DomainId(9), 8192, vec![CoreId(7), CoreId(8), CoreId(9)]);
        assert_eq!(a, b);
    }

    #[test]
    fn inequality_when_domain_ids_differ_even_if_other_fields_match() {
        let a = MemoryDomain::new(DomainId(1), 2048, vec![CoreId(0), CoreId(2)]);
        let b = MemoryDomain::new(DomainId(2), 2048, vec![CoreId(0), CoreId(2)]);
        assert_ne!(a, b);
    }

    #[test]
    fn equality_ignores_core_affinity_ordering() {
        let a = MemoryDomain::new(DomainId(11), 4096, vec![CoreId(0), CoreId(1), CoreId(2)]);
        let b = MemoryDomain::new(DomainId(11), 4096, vec![CoreId(2), CoreId(1), CoreId(0)]);
        assert_eq!(a, b);
    }

    #[test]
    fn core_affinity_ordering_is_preserved() {
        let core_affinity = vec![CoreId(5), CoreId(1), CoreId(7), CoreId(3)];
        let domain = MemoryDomain::new(DomainId(13), 16384, core_affinity.clone());
        assert_eq!(domain.core_affinity, core_affinity);
    }

    #[test]
    fn duplicate_core_affinity_entries_are_preserved() {
        let domain = MemoryDomain::new(DomainId(21), 512, vec![CoreId(4), CoreId(4), CoreId(6)]);
        assert_eq!(domain.core_affinity, vec![CoreId(4), CoreId(4), CoreId(6)]);
    }
}

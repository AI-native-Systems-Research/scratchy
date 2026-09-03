//! Port of `flex/include/flex/allocator/device_topology.hpp`. Pure value type;
//! the actual topology *query* against hardware happens once, in
//! [`populate_domain_topology`] below, and crosses senlib via
//! `flex_senlib_rcu_num_cores` (see `SENLIB_BOUNDARY.md`).
//! `DeviceTopology` itself is just the resulting immutable description.

use std::collections::BTreeSet;

use crate::domain::{CoreId, MemoryDomain};
use crate::memory_region::DomainId;

#[derive(Debug, Clone)]
pub struct DeviceTopology {
    domains: Vec<MemoryDomain>,
    num_cores: usize,
}

impl DeviceTopology {
    pub fn new(domains: Vec<MemoryDomain>) -> Self {
        let mut cores = BTreeSet::new();
        for domain in &domains {
            for core in &domain.core_affinity {
                cores.insert(*core);
            }
        }
        Self {
            domains,
            num_cores: cores.len(),
        }
    }

    pub fn domains(&self) -> &[MemoryDomain] {
        &self.domains
    }

    pub fn num_domains(&self) -> usize {
        self.domains.len()
    }

    pub fn num_cores(&self) -> usize {
        self.num_cores
    }

    pub fn is_valid_domain(&self, domain_id: DomainId) -> bool {
        self.domains.iter().any(|d| d.domain_id == domain_id)
    }

    /// Relative access latency between two domains. Analogous to Linux
    /// `numa_distance()`; a domain's distance to itself is 0, otherwise 1
    /// (matches the current, simplistic C++ implementation).
    pub fn distance(&self, domain_a: DomainId, domain_b: DomainId) -> u32 {
        if domain_a == domain_b { 0 } else { 1 }
    }
}

/// Port of `flex::populateDomainTopology`
/// (`flex/src/runtime_stream/1p0/device_memory_topology.cpp`). For the 1p0
/// target there is a single NUMA domain owning all device memory, with every
/// RCU core on the card showing affinity to it. `device_memory_size` is the
/// real per-card HBM capacity (`flex_senlib_device_memory_size()`, i.e.
/// `DeviceHandle::GetDmpaSize()`), not a placeholder — callers must not
/// invent one (see `fxa_rust_abi.rs::build_runtime`'s prior 32 GiB
/// placeholder, which this replaces).
pub fn populate_domain_topology(device_memory_size: u64) -> DeviceTopology {
    // SAFETY: static senlib query, no bring-up prerequisite — see this
    // extern's own doc comment in senlib_ffi.rs.
    let num_cores = unsafe { crate::senlib_ffi::flex_senlib_rcu_num_cores() };
    let affinity: Vec<CoreId> = (0..num_cores).map(CoreId).collect();
    let domain0 = MemoryDomain::new(DomainId(0), device_memory_size, affinity);
    DeviceTopology::new(vec![domain0])
}

/// Port of `flex/tests/allocator/data_structures/device_topology_test.cpp`.
#[cfg(test)]
mod ported_cxx_tests {
    use super::*;

    #[test]
    fn construct_4_domain_topology() {
        let test_total_bytes = 1024;
        let domains = vec![
            MemoryDomain::new(
                DomainId(0),
                test_total_bytes,
                vec![CoreId(0), CoreId(1), CoreId(2), CoreId(3)],
            ),
            MemoryDomain::new(
                DomainId(1),
                test_total_bytes,
                vec![CoreId(4), CoreId(5), CoreId(6), CoreId(7)],
            ),
            MemoryDomain::new(
                DomainId(2),
                test_total_bytes,
                vec![CoreId(8), CoreId(9), CoreId(10), CoreId(11)],
            ),
            MemoryDomain::new(
                DomainId(3),
                test_total_bytes,
                vec![CoreId(12), CoreId(13), CoreId(14), CoreId(15)],
            ),
        ];
        let topology = DeviceTopology::new(domains);
        assert_eq!(topology.num_domains(), 4);
        assert_eq!(topology.num_cores(), 16);
    }

    #[test]
    fn distance_basic() {
        let test_total_bytes = 1024;
        let domains = vec![
            MemoryDomain::new(
                DomainId(0),
                test_total_bytes,
                vec![CoreId(0), CoreId(1), CoreId(2), CoreId(3)],
            ),
            MemoryDomain::new(
                DomainId(1),
                test_total_bytes,
                vec![CoreId(4), CoreId(5), CoreId(6), CoreId(7)],
            ),
        ];
        let topology = DeviceTopology::new(domains);
        assert_eq!(topology.distance(DomainId(0), DomainId(0)), 0);
        assert!(topology.distance(DomainId(0), DomainId(1)) > 0);
    }

    #[test]
    fn empty_topology_has_zero_domains_and_zero_cores() {
        let topology = DeviceTopology::new(vec![]);
        assert_eq!(topology.num_domains(), 0);
        assert_eq!(topology.num_cores(), 0);
    }

    #[test]
    fn duplicate_cores_across_domains_counted_once() {
        let domains = vec![
            MemoryDomain::new(DomainId(0), 1024, vec![CoreId(0), CoreId(1), CoreId(2)]),
            MemoryDomain::new(DomainId(1), 1024, vec![CoreId(2), CoreId(3), CoreId(4)]),
        ];
        let topology = DeviceTopology::new(domains);
        assert_eq!(topology.num_domains(), 2);
        assert_eq!(topology.num_cores(), 5);
    }

    #[test]
    fn empty_core_affinity_domains_do_not_increase_core_count() {
        let domains = vec![
            MemoryDomain::new(DomainId(0), 1024, vec![]),
            MemoryDomain::new(DomainId(1), 2048, vec![CoreId(5), CoreId(6)]),
        ];
        let topology = DeviceTopology::new(domains);
        assert_eq!(topology.num_domains(), 2);
        assert_eq!(topology.num_cores(), 2);
    }

    #[test]
    fn zero_sized_domains_are_retained() {
        let domains = vec![
            MemoryDomain::new(DomainId(3), 0, vec![CoreId(1), CoreId(2)]),
            MemoryDomain::new(DomainId(4), 0, vec![]),
        ];
        let topology = DeviceTopology::new(domains);
        assert_eq!(topology.domains().len(), 2);
        assert_eq!(topology.domains()[0].domain_id, DomainId(3));
        assert_eq!(topology.domains()[0].total_bytes, 0);
        assert_eq!(topology.domains()[1].domain_id, DomainId(4));
        assert_eq!(topology.domains()[1].total_bytes, 0);
    }

    #[test]
    fn domain_ordering_is_preserved() {
        let domains = vec![
            MemoryDomain::new(DomainId(7), 1024, vec![CoreId(7)]),
            MemoryDomain::new(DomainId(2), 2048, vec![CoreId(2)]),
            MemoryDomain::new(DomainId(9), 4096, vec![CoreId(9)]),
        ];
        let topology = DeviceTopology::new(domains);
        assert_eq!(topology.domains().len(), 3);
        assert_eq!(topology.domains()[0].domain_id, DomainId(7));
        assert_eq!(topology.domains()[1].domain_id, DomainId(2));
        assert_eq!(topology.domains()[2].domain_id, DomainId(9));
    }

    #[test]
    fn distance_is_symmetric_for_different_domains() {
        let domains = vec![
            MemoryDomain::new(DomainId(0), 1024, vec![CoreId(0)]),
            MemoryDomain::new(DomainId(5), 1024, vec![CoreId(1)]),
        ];
        let topology = DeviceTopology::new(domains);
        assert_eq!(
            topology.distance(DomainId(0), DomainId(5)),
            topology.distance(DomainId(5), DomainId(0))
        );
        assert_eq!(topology.distance(DomainId(0), DomainId(5)), 1);
    }

    #[test]
    fn accepts_max_domain_and_core_values() {
        let domains = vec![MemoryDomain::new(
            DomainId(u32::MAX),
            u64::MAX,
            vec![CoreId(u32::MAX)],
        )];
        let topology = DeviceTopology::new(domains);
        assert_eq!(topology.num_domains(), 1);
        assert_eq!(topology.num_cores(), 1);
        assert_eq!(topology.domains()[0].domain_id, DomainId(u32::MAX));
        assert_eq!(topology.domains()[0].total_bytes, u64::MAX);
        assert_eq!(topology.domains()[0].core_affinity[0], CoreId(u32::MAX));
    }

    #[test]
    fn is_valid_domain_returns_false_for_missing_id() {
        let domains = vec![
            MemoryDomain::new(DomainId(0), 1024, vec![CoreId(0)]),
            MemoryDomain::new(DomainId(1), 1024, vec![CoreId(1)]),
        ];
        let topology = DeviceTopology::new(domains);
        assert!(!topology.is_valid_domain(DomainId(99)));
    }

    #[test]
    fn is_valid_domain_returns_false_for_empty_topology() {
        let topology = DeviceTopology::new(vec![]);
        assert!(!topology.is_valid_domain(DomainId(0)));
    }
}

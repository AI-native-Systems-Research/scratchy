//! Port of `flex/include/flex/device_interface/iommu/{iommu_mapping,iommu_mapper}.hpp`
//! and `flex/src/device_interface/iommu/iommu_raii_buffer.hpp` +
//! `flex/src/device_types/pf_device/pf_iommu_mapper.{hpp,cpp}`.
//!
//! `IommuMapperInterface` is abstract in the C++ (PF/VF/mock impls), but
//! only `PfIommuMapper` is ported: the mock backend does real memcpy
//! simulation with no IOMMU involvement at all (see `scheduler.rs`), so
//! there's nothing for a second impl to do. `PfIommuMapper` is a concrete
//! type rather than a trait object for the same reason `PfBackend`/
//! `VfBackend`/`MockBackend` are plain enum variants, not `dyn` — see
//! `scheduler::Backend`.

use std::sync::{Arc, Mutex};

use crate::host_interface::RaiiBuffer;
use crate::util::Status;

/// Port of `flex::IommuMapping`: an HMVA→IOVA mapping, unmapped on drop.
pub struct IommuMapping {
    size_bytes: usize,
    hmva: usize,
    iova: usize,
    writable: bool,
    mapper: Arc<PfIommuMapper>,
}

impl IommuMapping {
    pub fn hmva(&self) -> usize {
        self.hmva
    }
    pub fn iova(&self) -> usize {
        self.iova
    }
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }
    pub fn writable(&self) -> bool {
        self.writable
    }
}

impl Drop for IommuMapping {
    fn drop(&mut self) {
        // Port of `IommuMapping::~IommuMapping` (`iommu_mapping.cpp:30-38`):
        // guarded by `mapped_`, then `LOG_WARNING_IF(!s) << "Failed to unmap: "
        // << *this;`. `mapped_` is not modeled as a field here because it can
        // only ever be observed as `true`: the C++ sets it `true` in the
        // constructor, and the only write to `false` is the last statement of
        // the destructor — Rust's move/drop rules make a second drop
        // impossible, so the flag has no reachable `false` state (which is
        // also why `operator<<`'s `"IommuMapping(not mapped)"` branch,
        // `iommu_mapping.cpp:54-57`, is unreachable in `Display` below).
        if self
            .mapper
            .unmap(self.hmva, self.iova, self.size_bytes)
            .is_err()
        {
            tracing::warn!("Failed to unmap: {self}");
        }
    }
}

impl std::fmt::Display for IommuMapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IommuMapping(hmva: {:#x} -> iova: {:#x}, size: {} [B], {})",
            self.hmva,
            self.iova,
            self.size_bytes,
            if self.writable { "RW" } else { "RO" }
        )
    }
}

/// Port of `flex::IommuRaiiBuffer`.
pub struct IommuRaiiBuffer {
    pub raii_buffer: RaiiBuffer,
    pub iommu_mapping: Arc<IommuMapping>,
}

impl std::fmt::Display for IommuRaiiBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FlexHostBuffer( {}, iomap: {})",
            self.raii_buffer, self.iommu_mapping
        )
    }
}

const PF_IOMMU_TOTAL_ALLOCATION_SIZE: usize = 4 * 1024 * 1024 * 1024;

/// Port of `flex::PfIommuMapper`.
/// The `/proc/self/maps` line whose address range contains `addr` — permissions, and the backing
/// file if any. Diagnostic only, and only on a failure path: it reads and scans a small proc file.
fn proc_mapping_of(addr: usize) -> Option<String> {
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    for line in maps.lines() {
        let range = line.split_whitespace().next()?;
        let (lo, hi) = range.split_once('-')?;
        let lo = usize::from_str_radix(lo, 16).ok()?;
        let hi = usize::from_str_radix(hi, 16).ok()?;
        if (lo..hi).contains(&addr) {
            return Some(line.trim().to_string());
        }
    }
    None
}

pub struct PfIommuMapper {
    page_size: usize,
    table: Mutex<std::collections::HashMap<u64, (u64, usize)>>,
}

/// Port of `common::page_size()` (`common/addr_helper.hpp:15`):
/// `static_cast<size_t>(sysconf(_SC_PAGESIZE))`.
fn common_page_size() -> usize {
    // SAFETY: `sysconf` with `_SC_PAGESIZE` has no preconditions and never
    // fails on a real system (a negative return would only occur for an
    // unsupported `name` argument, which `_SC_PAGESIZE` is not).
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    page_size.max(0) as usize
}

impl PfIommuMapper {
    /// Port of `PfIommuMapper::PfIommuMapper(std::shared_ptr<senlib::v2::PfWrapper>)`
    /// (`pf_iommu_mapper.cpp:41-42`): `page_size_(common::page_size())` — the
    /// real C++ constructor derives the page size itself and never accepts
    /// it from the caller.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            page_size: common_page_size(),
            table: Mutex::new(std::collections::HashMap::new()),
        })
    }

    /// Port of `PfIommuMapper::Map(IommuMappingPtr*, void*, size_t, bool)`
    /// (`pf_iommu_mapper.cpp:46-76`).
    pub fn map(
        self: &Arc<Self>,
        hmva: usize,
        size_bytes: usize,
        writable: bool,
    ) -> Result<IommuMapping, Status> {
        let mut iova = std::ptr::null_mut();
        let mut mapped_size = 0u64;
        // SAFETY: out-params are only read after the shim reports success.
        // `writable` is threaded through to the FFI call end-to-end (see
        // `senlib_ffi::flex_senlib_iommu_register_and_map`'s doc comment for
        // why the real `Map()` this mirrors doesn't forward it any further
        // than that).
        let rc = unsafe {
            crate::senlib_ffi::flex_senlib_iommu_register_and_map(
                hmva as *mut std::ffi::c_void,
                size_bytes as u64,
                writable,
                &mut iova,
                &mut mapped_size,
            )
        };
        if rc != 0 {
            // ⭐ SAY WHAT WE TRIED TO PIN. `EFAULT` from the VFIO pin has two very different causes
            // and the errno cannot tell them apart: the range runs past the end of the buffer, or
            // the range is in memory the kernel will not pin for DMA (a read-only file-backed
            // mapping — `.rodata`, where a baked program's `Cow::Borrowed` bytes live — as opposed
            // to anonymous heap). The mapping that owns `hmva` distinguishes them at a glance.
            tracing::error!(
                "IOMMU map FAILED for hmva={hmva:#x} size={size_bytes} writable={writable} — \
                 owning mapping: {}",
                proc_mapping_of(hmva).unwrap_or_else(|| "<not found in /proc/self/maps>".into())
            );
            return Err(Status::runtime_error("IOMMU mapping failed"));
        }
        let iomap = IommuMapping {
            size_bytes: mapped_size as usize,
            hmva,
            iova: iova as usize,
            writable,
            mapper: self.clone(),
        };
        // `LOG_TRACE << "Mapped: " << *iomap;` (pf_iommu_mapper.cpp:62) —
        // emitted before the table insert, as in the C++.
        tracing::trace!("Mapped: {iomap}");
        {
            let mut table = self.table.lock().unwrap_or_else(|e| e.into_inner());
            // Port of `LOG_ERROR_IF(pf_iommu_mapping_table_.count(iova_as_uint) > 0)
            //   << "iova pointer " << std::hex << iova << std::dec << " already present in table";`
            // (pf_iommu_mapper.cpp:66-67). The C++ only *logs* — it still
            // overwrites the entry — so this must not become an error return.
            let iova_as_uint = iova as u64;
            if table.contains_key(&iova_as_uint) {
                tracing::error!("iova pointer {iova_as_uint:x} already present in table");
            }
            // Note the table stores the *requested* `size_bytes`, not the
            // sglist entry's `length` that `IommuMapping` carries
            // (pf_iommu_mapper.cpp:68-69). Verified faithful.
            table.insert(iova_as_uint, (hmva as u64, size_bytes));
        }
        Ok(iomap)
    }

    /// Port of the non-virtual `IommuMapperInterface::Map(IommuMappingPtr*,
    /// const RaiiBuffer&, bool)` (`iommu_mapper.cpp:15-23`): rejects a null
    /// `Pointer()` up front, otherwise forwards `Pointer()`/`SizeBytes()` to
    /// the virtual `Map`.
    pub fn map_buffer(
        self: &Arc<Self>,
        buf: &RaiiBuffer,
        writable: bool,
    ) -> Result<IommuMapping, Status> {
        if buf.as_ptr().is_null() {
            return Err(Status::invalid_argument("Trying to map nullptr into IOMMU"));
        }
        self.map(buf.as_ptr() as usize, buf.size_bytes(), writable)
    }

    /// `pf_iommu_mapper.cpp:106`: `return -1;` from a `size_t`-returning
    /// function, i.e. `SIZE_MAX` — "unlimited", not an error code.
    pub fn max_allocation_count(&self) -> usize {
        usize::MAX
    }
    /// `pf_iommu_mapper.cpp:108`: `return GetTotalAllocationSize();`
    pub fn max_allocation_size(&self) -> usize {
        self.total_allocation_size()
    }
    /// `pf_iommu_mapper.cpp:110`: `return 4ULL * 1024 * 1024 * 1024;`
    pub fn total_allocation_size(&self) -> usize {
        PF_IOMMU_TOTAL_ALLOCATION_SIZE
    }
    /// `pf_iommu_mapper.cpp:112`: `return page_size_;`
    pub fn alignment(&self) -> usize {
        self.page_size
    }

    /// Port of `PfIommuMapper::GetHmvaFromTable` (`pf_iommu_mapper.cpp:115-126`).
    /// `None` here is the C++'s `return 1` (out-params untouched); `Some` is
    /// its `return 0`.
    pub fn get_hmva_from_table(&self, iova: u64) -> Option<(u64, usize)> {
        let entry = self
            .table
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&iova)
            .copied();
        if entry.is_none() {
            // `LOG_ERROR << "iova pointer " << std::hex << iova << " not present in table";`
            // (pf_iommu_mapper.cpp:120).
            tracing::error!("iova pointer {iova:x} not present in table");
        }
        entry
    }

    /// Port of `PfIommuMapper::Unmap` (`private` in the C++, reached only
    /// via `friend IommuMapping`; called here from `IommuMapping::drop`).
    fn unmap(&self, hmva: usize, iova: usize, size_bytes: usize) -> Result<(), Status> {
        // `LOG_TRACE << "Unmapping: " << mapping;` (pf_iommu_mapper.cpp:87).
        tracing::trace!(
            "Unmapping: IommuMapping(hmva: {hmva:#x} -> iova: {iova:#x}, size: {size_bytes} [B])"
        );
        // Port of `auto size_bytes = sendnn::ceil_to(mapping.SizeInBytes(), page_size_);`
        // (pf_iommu_mapper.cpp:89): both the unpin/unmap and the
        // UnregisterMemForIOMMU calls use this page-rounded size, not the raw
        // mapping size.
        let size_bytes = crate::util::ceil_to(size_bytes as u64, self.page_size as u64);
        // SAFETY: `hmva`/`iova` came from a prior `map()` on this mapper.
        let rc = unsafe {
            crate::senlib_ffi::flex_senlib_iommu_unmap_and_unregister(
                hmva as *mut std::ffi::c_void,
                iova as *mut std::ffi::c_void,
                size_bytes,
            )
        };
        self.table
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&(iova as u64));
        if rc != 0 {
            return Err(Status::runtime_error("IOMMU unmap failed"));
        }
        Ok(())
    }
}

// Ported from flex/tests/memory_interface/iommu_mapper_test.cpp. That suite's
// fixture (`IommuMapperTest::SetUp`) calls `RuntimeContext::create()` and gets
// a real device handle, so 3 of its 4 `TEST_F`s exercise an actual senlib
// mapping and cannot run as a unit test on a machine with no Spyre SDK/hardware:
//
// - MapValidBufferSucceeds: calls the real `iommu_mapper->Map(...)` on live
//   hardware and inspects the resulting mapping (`Mapped()`, `HmvaAsPointer()`,
//   `SizeInBytes()`, `operator<<` contents).
//   SKIPPED: requires a live senlib device via RuntimeContext::create().
// - MapReadOnlyBufferSucceeds: same as above, for a read-only mapping.
//   SKIPPED: requires a live senlib device via RuntimeContext::create().
// - MappingDestructorDoesNotCrash: drops a real mapping obtained the same way.
//   SKIPPED: requires a live senlib device via RuntimeContext::create().
//
// The 4th (`MapNullptrBufferReturnsError`) IS portable: it's pure logic in the
// C++ that never reaches the virtual `Map()`/senlib call at all, since
// `IommuMapperInterface::Map` checks `buf.Pointer() == nullptr` up front —
// exactly the same check `map_buffer` below does. `host_interface.rs` now
// has a real `Default for RaiiBuffer` (the faithful port of the C++'s
// `RaiiBuffer() = default`), so a null-pointer `RaiiBuffer` is constructible
// through public API without any test-only backdoor. See
// `map_buffer_rejects_nullptr_buffer` below.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::StatusCode;

    #[test]
    fn map_buffer_rejects_nullptr_buffer() {
        // Port of iommu_mapper_test.cpp's MapNullptrBufferReturnsError.
        let mapper = PfIommuMapper::new();
        let null_buffer = RaiiBuffer::default();

        let status = match mapper.map_buffer(&null_buffer, true) {
            Err(status) => status,
            Ok(_) => panic!("mapping a nullptr buffer must not succeed"),
        };

        assert_eq!(status.code(), StatusCode(400)); // INVALID_ARGUMENT, per util::Status::invalid_argument
        assert!(status.message().as_str().contains("nullptr"));
        assert!(status.message().as_str().contains("IOMMU"));
    }
}

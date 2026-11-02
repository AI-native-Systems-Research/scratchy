//! Port of flex's shared host pool: a fixed-geometry set of DMA-pinned host
//! slots behind ONE IOMMU mapping, addressed by index.
//!
//! Ported from `flex/docs/architecture/shared_host_pool.md` (the design
//! doc); `flex/include/flex/memory_interface/shared_pool.hpp` with
//! `flex/src/memory_interface/shared_pool.cpp` (the `SharedPool` interface:
//! geometry getters, `ResolvedSlot`, `EnqueueCopy`, `CopyRawPreMapped`,
//! `ResolveSlot`); `flex/include/flex/memory_interface/shared_host_pool.hpp`
//! with `flex/src/memory_interface/shared_host_pool.cpp` (the
//! implementation: `CreateOrAttach`, `MapAndPin`, slot arithmetic,
//! teardown); and the pre-mapped copy entry that consumes it on
//! `RuntimeStream` (`runtime_stream.hpp:148`/`:175`, `copyRaw`/`copyRawImpl`
//! at `runtime_stream.cpp:434-600`).
//!
//! # Why it exists
//!
//! Per-step host buffers in this crate go through freshly mapped DmaBufs, so
//! their IOVAs move per step and a DMA control block that embeds a host IOVA
//! cannot be pre-built. A pool slot's IOVA is stable for the life of the
//! pool — pinned once at construction, addressed by `slot_index` — which is
//! exactly the property that lets a per-step DMA become a patch-free member
//! of a pre-built CB image (issue #606, the host-side half of #603 PR3b's
//! double-buffered per-step slots).
//!
//! # The invariants this port keeps
//!
//! - **Pinned, index-addressed, fixed-size slots.** Slot `i` lives at
//!   `base + i * stride` for both the host virtual address and the IOVA;
//!   `stride` is the requested slot size rounded up to the mapper's IOMMU
//!   alignment (the vendor's `SlotBytes()` law). Slots are caller-managed by
//!   index — the vendor pool has NO per-slot acquire/release refcount
//!   either; ownership of which slot is whose is a caller convention, and
//!   the borrow discipline is pool-level (below).
//! - **ONE IOMMU mapping over the whole region.** A single
//!   `PfIommuMapper::map_buffer` pins the entire data segment at
//!   construction (`writable = true`: a slot is both an H2D source and a
//!   D2H target), so per-slot IOVAs are stable for the life of the pool and
//!   no per-step pin/unmap traffic exists.
//! - **A null IOVA means "not pre-mapped" to the DMA backends**, so a region
//!   that pinned to IOVA 0 is refused at construction rather than handing
//!   every slot an IOVA that silently disables the pre-mapped path.
//! - **The transfer-length law.** A pre-mapped copy always transfers the
//!   device object's `total_size`, and that must fit within the slot's
//!   capacity — a larger device object would DMA past the slot's end into
//!   the NEIGHBORING slot. The check throws before any DMA is issued.
//! - **The pool is borrowed by its in-flight transfers.** The caller must
//!   `synchronize()` the streams it submitted pre-mapped copies on before
//!   dropping the pool. Note `synchronize()` does NOT wait on streams in
//!   the error state — draining those is `drainForTeardown`'s job (issue
//!   #609, a separate port; until it lands, an errored stream's outstanding
//!   pre-mapped copies are the caller's hazard).
//! - **Drop order: mapping, then mapper, then data segment.** Rust field
//!   declaration order *is* drop order, so [`SharedHostPool`] declares
//!   `mapping` first: the region is unpinned before the backing buffer is
//!   freed. The mapper-retention law — the pool keeps its mapper alive so
//!   the pool outlives `RuntimeContext::reset()` — is enforced by the type
//!   system here rather than by discipline: [`IommuMapping`] itself holds an
//!   `Arc<PfIommuMapper>`, so the mapping's lifetime pins the mapper's.
//!
//! # What this port does NOT carry, on purpose (the named delta)
//!
//! The vendor pool is CROSS-PROCESS: a named POSIX shared-memory object with
//! a one-page control segment (magic/version/geometry/attach-count/
//! init-done), creator election via `shm_open(O_CREAT | O_EXCL)`, an
//! attach-refcount with `init_done` publish + bounded spin-wait, loud
//! refusal on geometry mismatch between attacher and creator, and
//! `UnlinkByName` teardown. None of that is ported: #606's consumer
//! (#603 PR3b) is single-process, and process-local backing is this crate's
//! existing idiom — a [`RaiiBuffer`] for the data segment and one whole-
//! region [`PfIommuMapper::map_buffer`] for the pin. That keeps senlib the
//! only allocator and adds ZERO new senlib edges: this module makes no
//! `extern "C"` call of its own, it only consumes `iommu`'s
//! [`PfIommuMapper::map_buffer`] and hands the resulting IOVA to `dma`'s
//! existing `DmaParams.iova` pre-mapped field (the consumer path the
//! scheduler already honors).
//!
//! Two smaller deliberate divergences:
//!
//! - The vendor's `SlotPtr` is a private class handed to `EnqueueCopy`; a
//!   cross-process user writes slot bytes through its OWN mapping of the
//!   shared segment. Our single-process owner has no other way to write
//!   slot bytes, so the port exposes checked byte views ([`SharedHostPool::slot`]
//!   / [`SharedHostPool::slot_mut`]) instead.
//! - `SharedPool::Name()`/`UnlinkByName` disappear with the rendezvous — a
//!   process-local pool has no name to rendezvous by.
//!
//! # Vendor test mapping
//!
//! `flex/tests/memory_interface/shared_host_pool_test.cpp` runs against a
//! mock `IommuMapperInterface` seam (`FailingIommuMapper` with
//! throw-std/throw-non-std/null-mapping/zero-iova modes, plus a
//! `RecordingIommuMapper`) and real POSIX SHM. This crate has no mock-mapper
//! seam on purpose — `PfIommuMapper` is concrete because there is nothing
//! for a second impl to do (see `iommu`'s module doc) — so the suite maps as:
//!
//! - The geometry/stride laws, slot-range refusal, and `copyRaw`'s
//!   validation set: ported below as pure-logic tests
//!   (`PoolGeometry`/`validate_copy_raw_device`), runnable cardless.
//! - Construction-time validation firing BEFORE any senlib call: ported
//!   (`pool_construction_validates_geometry_before_any_mapping`) — this is
//!   the cardless guarantee that a refused pool never pins.
//! - The SHM rendezvous family (`CreateThenAttachSameProcess`,
//!   `CrossProcessSlotVisibility`, `AttachRefcountLifecycle`,
//!   `GeometryMismatchFailsLoudly`, and the unlink/attach tails of the
//!   suite): NOT PORTED — no cross-process machinery exists to test (the
//!   named delta above).
//! - The map-failure family (`AttacherMapFailureKeepsLivePoolAttachable`,
//!   `CreatorMapFailureUnlinksBothSegments`, and the mock's `kZeroIova` /
//!   `kNullMapping` modes): in the process-local port a map failure or a
//!   null/zero IOVA is a construction error that leaves no residue (the
//!   half-built `IommuMapping` drops and unpins itself), but reaching it
//!   requires a live senlib map. SKIPPED cardless; exercised by the
//!   `shared_host_pool_probe` example on a card.
//! - `PoolRetainsMapperUntilUnmapped`: enforced by construction
//!   ([`IommuMapping`] holds the `Arc<PfIommuMapper>`); nothing to run.

use std::sync::Arc;

use crate::address::CompositeAddress;
use crate::dma::{
    DmaDirection, DmaParams, DmaParamsError, DmaPipelineShape, DmaShaping, HostVirtualAddress,
    IovaAddress, OpName,
};
use crate::host_interface::{Alignment, RaiiBuffer, RaiiBufferError};
use crate::iommu::{IommuMapping, PfIommuMapper};
use crate::stream::{RuntimeStream, RuntimeStreamError};
use crate::util::Status;

/// The pool's fixed geometry: slot count and the ALIGNED per-slot stride.
/// Pure arithmetic — carries no memory, makes no calls, and is the half of
/// the pool that is fully testable without a card.
///
/// `slot_bytes` is the *aligned* stride (requested size rounded up to the
/// IOMMU alignment), which is also the capacity a pre-mapped copy may
/// transfer into a slot — the vendor's `SlotBytes()`, not the caller's
/// requested byte count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolGeometry {
    num_slots: usize,
    slot_bytes: usize,
}

impl PoolGeometry {
    /// Port of the creator-side geometry computation: at least one slot, a
    /// non-zero requested slot size, and a stride of
    /// `ceil_to(requested, iova_alignment)` — the vendor rounds `SlotBytes()`
    /// up to `DeviceHandle::GetIovaAlignment()` so every slot start keeps
    /// the pin's alignment (the backing buffer is allocated at that same
    /// alignment, so `base + i * stride` is aligned for every `i`).
    ///
    /// Computed with checked arithmetic rather than [`crate::util::ceil_to`]:
    /// `ceil_to`'s add can wrap on overflow, and a geometry whose rounded
    /// stride or `num_slots * stride` total does not fit a `usize` must
    /// refuse, not wrap.
    ///
    /// # Errors
    /// - [`SharedHostPoolError::ZeroSlots`] — no slots requested.
    /// - [`SharedHostPoolError::ZeroSlotBytes`] — empty slot requested.
    /// - [`SharedHostPoolError::ZeroAlignment`] — a mapper reporting
    ///   alignment 0 (division by zero in the round-up; a broken contract).
    /// - [`SharedHostPoolError::StrideOverflow`] — the rounded stride does
    ///   not fit a `usize`.
    /// - [`SharedHostPoolError::TotalSizeOverflow`] — `num_slots * stride`
    ///   does not fit a `usize`.
    pub fn new(
        num_slots: usize,
        requested_slot_bytes: usize,
        iova_alignment: usize,
    ) -> Result<Self, SharedHostPoolError> {
        if num_slots == 0 {
            return Err(SharedHostPoolError::ZeroSlots);
        }
        if requested_slot_bytes == 0 {
            return Err(SharedHostPoolError::ZeroSlotBytes);
        }
        if iova_alignment == 0 {
            return Err(SharedHostPoolError::ZeroAlignment);
        }
        // `ceil_to(requested, alignment)` with the overflow checked: `rem` is
        // in `0..alignment` so `alignment - rem` cannot overflow, and the
        // `checked_add` is what refuses a stride past `usize::MAX`.
        let rem = requested_slot_bytes % iova_alignment;
        let slot_bytes = if rem == 0 {
            requested_slot_bytes
        } else {
            requested_slot_bytes
                .checked_add(iova_alignment - rem)
                .ok_or(SharedHostPoolError::StrideOverflow)?
        };
        let total = num_slots
            .checked_mul(slot_bytes)
            .ok_or(SharedHostPoolError::TotalSizeOverflow)?;
        debug_assert!(total >= slot_bytes && slot_bytes >= requested_slot_bytes);
        Ok(Self {
            num_slots,
            slot_bytes,
        })
    }

    /// Port of `SharedPool::SlotCount()`.
    pub fn slot_count(&self) -> usize {
        self.num_slots
    }

    /// Port of `SharedPool::SlotBytes()` — the ALIGNED stride, i.e. the
    /// per-slot capacity a pre-mapped copy may transfer.
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// Port of `SharedPool::TotalBytes()` — `SlotCount() * SlotBytes()`,
    /// the size of the one pinned region.
    pub fn total_bytes(&self) -> usize {
        // Infallible: `new` checked `num_slots * slot_bytes` fits.
        self.num_slots * self.slot_bytes
    }

    fn validate_slot(&self, slot: usize) -> Result<(), SharedHostPoolError> {
        if slot >= self.num_slots {
            return Err(SharedHostPoolError::SlotOutOfRange {
                slot,
                slot_count: self.num_slots,
            });
        }
        Ok(())
    }

    /// Byte offset of slot `slot`'s start within the pool's region:
    /// `slot * stride`, with the index range-checked first. This is the ONE
    /// piece of slot arithmetic everything else (host address, IOVA, byte
    /// views) is derived from.
    ///
    /// # Errors
    /// [`SharedHostPoolError::SlotOutOfRange`] if `slot >= slot_count()`.
    pub fn slot_offset(&self, slot: usize) -> Result<usize, SharedHostPoolError> {
        self.validate_slot(slot)?;
        // Infallible after validation: `slot < num_slots` and
        // `num_slots * slot_bytes` fits (checked in `new`), so this product
        // is < `total_bytes()`.
        Ok(slot * self.slot_bytes)
    }
}

/// Port of `SharedPool::ResolvedSlot`: the resolved addresses of one slot —
/// its host virtual address, its capacity (the aligned stride, the vendor's
/// `host_capacity`), and its IOVA. Plain data: resolving is pure arithmetic
/// over the pool's bases and enqueues nothing (the vendor's `ResolveSlot`
/// makes no submission either).
///
/// The IOVA is stable for the life of the pool: the whole region is pinned
/// once at construction and unmapped only at drop, so `pool_base_iova +
/// slot * stride` cannot move while the pool exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedSlot {
    /// Host virtual address of the slot's first byte.
    pub host_address: HostVirtualAddress,
    /// The aligned per-slot stride — the maximum bytes a pre-mapped copy may
    /// transfer into/out of this slot.
    pub host_capacity: usize,
    /// The slot's IO virtual address, stable for the life of the pool.
    pub iova: IovaAddress,
}

/// Errors from constructing a pool, addressing a slot, or submitting a
/// pre-mapped copy through one. Debug-only like [`RuntimeStreamError`]:
/// `Status` (carried by [`SharedHostPoolError::MapFailed`]) is deliberately
/// not equality-comparable, and the launch error is a whole error tree.
#[derive(Debug)]
pub enum SharedHostPoolError {
    // --- geometry (construction; fires before any allocation or senlib call) ---
    /// Zero slots requested — a pool with no slots cannot exist.
    ZeroSlots,
    /// Zero slot bytes requested.
    ZeroSlotBytes,
    /// The mapper reported IOMMU alignment 0 — a broken mapper contract
    /// (the round-up would divide by zero).
    ZeroAlignment,
    /// The alignment-rounded slot stride does not fit a `usize`.
    StrideOverflow,
    /// `slot_count * stride` does not fit a `usize`.
    TotalSizeOverflow,
    /// The mapper's IOMMU alignment is not expressible as a host allocation
    /// alignment (not a power of two, smaller than a pointer, or wider than
    /// `u32`). `PfIommuMapper::alignment()` is the page size, so this is a
    /// broken-mapper contract, not an expected caller error.
    UnrepresentableBufferAlignment,
    // --- construction (only reachable with a live senlib link) ---
    /// The backing host buffer could not be allocated.
    HostAllocation(RaiiBufferError),
    /// The whole-region IOMMU map failed (the senlib pin refused the
    /// buffer).
    MapFailed(Status),
    /// The region pinned to IOVA 0. A null IOVA is the DMA backends'
    /// "not pre-mapped" sentinel, so the pool refuses to exist rather than
    /// hand every slot an IOVA that silently disables the pre-mapped path.
    NullIova,
    // --- slot addressing ---
    /// Slot index out of range for the pool's geometry.
    SlotOutOfRange { slot: usize, slot_count: usize },
    // --- the copyRaw validation set (enqueue_copy) ---
    /// Port of `copyRawImpl`'s first check: the stream is in the shutdown
    /// state and refuses new work.
    NeedsShutdown,
    /// Port of `copyRawImpl`'s null-device check (fires before anything
    /// dereferences the device address).
    NullDeviceAddress,
    /// Port of `requireSingleChunk`: the DMA backends are inconsistent on
    /// multi-chunk device addresses, so the pre-mapped path only accepts a
    /// single-chunk one.
    MultiChunkDevice { chunk_count: usize },
    /// Port of `deviceSizeFitsHostCapacity`: the transfer length is always
    /// the device object's `total_size`, and it exceeds the slot's capacity
    /// — the DMA would run past the slot's end into the NEIGHBORING slot.
    /// Refused before any DMA is issued.
    DeviceSizeExceedsSlot {
        device_size: u64,
        host_capacity: usize,
    },
    // --- launch ---
    /// The stream rejected the submission itself.
    Launch(RuntimeStreamError),
}

impl std::fmt::Display for SharedHostPoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroSlots => write!(f, "shared host pool: zero slots requested"),
            Self::ZeroSlotBytes => write!(f, "shared host pool: zero slot bytes requested"),
            Self::ZeroAlignment => {
                write!(f, "shared host pool: mapper reported IOMMU alignment 0")
            }
            Self::StrideOverflow => {
                write!(
                    f,
                    "shared host pool: alignment-rounded slot stride overflows usize"
                )
            }
            Self::TotalSizeOverflow => {
                write!(f, "shared host pool: slot_count * stride overflows usize")
            }
            Self::UnrepresentableBufferAlignment => write!(
                f,
                "shared host pool: mapper IOMMU alignment is not an expressible host allocation alignment"
            ),
            Self::HostAllocation(e) => write!(
                f,
                "shared host pool: backing buffer: {}",
                // `RaiiBufferError` has no `Display` of its own; these are
                // the two C++ throw messages its variants port (see
                // `host_interface`'s docs on `RaiiAllocFail`).
                match e {
                    RaiiBufferError::ZeroSizeRequested => "attempted to allocate zero bytes",
                    RaiiBufferError::AllocationFailed => "aligned memory allocation failed",
                }
            ),
            Self::MapFailed(s) => write!(f, "shared host pool: whole-region IOMMU map failed: {s}"),
            Self::NullIova => write!(
                f,
                "shared host pool: region pinned to IOVA 0 (null IOVA = not pre-mapped)"
            ),
            Self::SlotOutOfRange { slot, slot_count } => {
                write!(
                    f,
                    "shared host pool: slot {slot} out of range (slot count {slot_count})"
                )
            }
            Self::NeedsShutdown => {
                write!(f, "RuntimeStreamCopyRaw: stream is in the shutdown state")
            }
            Self::NullDeviceAddress => write!(f, "RuntimeStreamCopyRaw: device address was null"),
            Self::MultiChunkDevice { chunk_count } => write!(
                f,
                "RuntimeStreamCopyRaw: device address has {chunk_count} chunks; the pre-mapped path requires a single chunk"
            ),
            Self::DeviceSizeExceedsSlot {
                device_size,
                host_capacity,
            } => write!(
                f,
                "RuntimeStreamCopyRaw: device total_size {device_size} exceeds slot capacity {host_capacity} — would DMA past the slot into its neighbor"
            ),
            Self::Launch(e) => write!(f, "RuntimeStreamCopyRaw: launch failed: {e}"),
        }
    }
}

impl std::error::Error for SharedHostPoolError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            // `RaiiBufferError` and `Status` carry their messages through
            // this enum's own `Display` (neither implements
            // `std::error::Error`); only the launch error is an error tree
            // of its own.
            Self::Launch(e) => Some(e),
            _ => None,
        }
    }
}

/// Port of `copyRawImpl`'s `requireSingleChunk` + `deviceSizeFitsHostCapacity`
/// (`runtime_stream.cpp:434-600`): the device-side half of the pre-mapped
/// copy's validation, in the vendor's order. Pure logic over the device
/// descriptor and the slot capacity — no pool, stream, or senlib involved —
/// so the full set is testable cardless (see the tests below).
fn validate_copy_raw_device(
    device: &CompositeAddress,
    host_capacity: usize,
) -> Result<(), SharedHostPoolError> {
    if !device.is_single_chunk() {
        return Err(SharedHostPoolError::MultiChunkDevice {
            chunk_count: device.num_chunks(),
        });
    }
    if device.total_size().0 > host_capacity as u64 {
        return Err(SharedHostPoolError::DeviceSizeExceedsSlot {
            device_size: device.total_size().0,
            host_capacity,
        });
    }
    Ok(())
}

/// The index-addressing law, factored pure so it is testable without a
/// pinned pool: slot `i` sits at `base + i * stride` in BOTH address spaces,
/// with the aligned stride as its capacity. [`SharedHostPool::resolve_slot`]
/// is this function over the pool's real bases.
fn resolved_slot_at(
    base_hmva: usize,
    base_iova: usize,
    geometry: &PoolGeometry,
    slot: usize,
) -> Result<ResolvedSlot, SharedHostPoolError> {
    let offset = geometry.slot_offset(slot)?;
    Ok(ResolvedSlot {
        host_address: HostVirtualAddress(base_hmva + offset),
        host_capacity: geometry.slot_bytes(),
        iova: IovaAddress(base_iova + offset),
    })
}

/// Port of `flex::SharedHostPool`, process-local: `num_slots` pinned,
/// fixed-size host slots behind ONE IOMMU mapping, addressed by index.
///
/// Construction allocates the whole data segment as one page-law-aligned
/// [`RaiiBuffer`], pins it with a single whole-region
/// [`PfIommuMapper::map_buffer`] (`writable = true`), and refuses a null
/// IOVA. Geometry validation fires BEFORE any allocation or senlib call, so
/// every refusal below [`SharedHostPoolError::HostAllocation`] is reachable
/// and testable on a machine with no hardware.
///
/// # Field order is drop order
///
/// `mapping` is declared FIRST because Rust drops fields in declaration
/// order: the region must be unpinned before the backing buffer is freed
/// (the vendor's teardown order — mapping, mapper, then data segment; the
/// mapper is released by the mapping's own `Arc` after the unmap). Do not
/// reorder the fields.
///
/// # The pool is borrowed by in-flight transfers
///
/// A submitted pre-mapped copy reads/writes slot bytes until it completes.
/// `synchronize()` every stream used with [`Self::enqueue_copy`] before
/// dropping the pool — and note `synchronize()` does not wait on streams in
/// the error state; draining those is `drainForTeardown`'s job (issue #609,
/// not yet ported).
pub struct SharedHostPool {
    // ⚔️ DECLARATION ORDER = DROP ORDER — see the struct doc. Unmap first,
    // free second; the mapper lives inside the mapping's Arc and is released
    // only after the unmap returns.
    mapping: IommuMapping,
    buffer: RaiiBuffer,
    geometry: PoolGeometry,
}

impl SharedHostPool {
    /// Port of the creator leg of `CreateOrAttach` + `MapAndPin`,
    /// process-local: validate the geometry, allocate the data segment at
    /// the mapper's IOMMU alignment (so every `base + i * stride` slot start
    /// keeps the pin's alignment), and pin the WHOLE region with one
    /// `map_buffer` call.
    ///
    /// The mapper is taken as an argument — the vendor's pool owns its
    /// mapper handle the same way, which is what lets a pool outlive
    /// `RuntimeContext::reset()`: the mapping (and the `Arc<PfIommuMapper>`
    /// inside it) is independent of any runtime-owned mapper instance.
    ///
    /// # Errors
    /// Geometry refusals ([`SharedHostPoolError::ZeroSlots`] …
    /// [`SharedHostPoolError::TotalSizeOverflow`]) fire before any
    /// allocation or senlib call. Beyond those: backing-buffer allocation
    /// ([`SharedHostPoolError::HostAllocation`]), the whole-region pin
    /// ([`SharedHostPoolError::MapFailed`]), and the null-IOVA refusal
    /// ([`SharedHostPoolError::NullIova`] — a half-built mapping drops and
    /// unpins itself, so a refused pool leaves no pinned residue).
    pub fn new(
        num_slots: usize,
        slot_bytes: usize,
        mapper: &Arc<PfIommuMapper>,
    ) -> Result<Self, SharedHostPoolError> {
        // Geometry first: every refusal in here returns before the buffer
        // allocation and long before the senlib pin.
        let geometry = PoolGeometry::new(num_slots, slot_bytes, mapper.alignment())?;
        // The data segment's base must keep the pin's alignment for every
        // slot start; the vendor gets that from mmap's page-aligned base,
        // this port asks the allocator for it explicitly.
        let alignment = u32::try_from(mapper.alignment())
            .ok()
            .and_then(Alignment::new)
            .ok_or(SharedHostPoolError::UnrepresentableBufferAlignment)?;
        let buffer = RaiiBuffer::new(geometry.total_bytes(), alignment)
            .map_err(SharedHostPoolError::HostAllocation)?;
        // ONE whole-region pin, writable: a slot is both an H2D source and a
        // D2H target. `map_buffer` rejects a null backing pointer up front
        // (unreachable here — `RaiiBuffer::new` just succeeded).
        let mapping = mapper
            .map_buffer(&buffer, true)
            .map_err(SharedHostPoolError::MapFailed)?;
        // The pin-time null-IOVA refusal: IOVA 0 is the backends'
        // "not pre-mapped" sentinel. The just-built mapping drops (and
        // unpins) on this return — no residue.
        if mapping.iova() == 0 {
            return Err(SharedHostPoolError::NullIova);
        }
        tracing::info!(
            "SharedHostPool pinned: {} slot(s) x {} [B] (requested {}), hmva {:#x}, iova {:#x}",
            geometry.slot_count(),
            geometry.slot_bytes(),
            slot_bytes,
            buffer.as_ptr() as usize,
            mapping.iova()
        );
        Ok(Self {
            mapping,
            buffer,
            geometry,
        })
    }

    /// Port of `SharedPool::SlotCount()`.
    pub fn slot_count(&self) -> usize {
        self.geometry.slot_count()
    }

    /// Port of `SharedPool::SlotBytes()` — the ALIGNED stride, i.e. the
    /// per-slot capacity a pre-mapped copy may transfer.
    pub fn slot_bytes(&self) -> usize {
        self.geometry.slot_bytes()
    }

    /// Port of `SharedPool::TotalBytes()` — the size of the one pinned
    /// region.
    pub fn total_bytes(&self) -> usize {
        self.geometry.total_bytes()
    }

    /// Port of `SharedPool::ResolveSlot`: resolve slot `slot`'s host address,
    /// capacity, and stable IOVA. Enqueues nothing; the resolved IOVA stays
    /// valid until the pool is dropped (the region is pinned once, at
    /// construction).
    ///
    /// # Errors
    /// [`SharedHostPoolError::SlotOutOfRange`] if `slot >= slot_count()`.
    pub fn resolve_slot(&self, slot: usize) -> Result<ResolvedSlot, SharedHostPoolError> {
        resolved_slot_at(
            self.buffer.as_ptr() as usize,
            self.mapping.iova(),
            &self.geometry,
            slot,
        )
    }

    /// Checked read view of slot `slot`'s bytes (the full aligned stride).
    ///
    /// The vendor's `SlotPtr` is private and a cross-process user writes
    /// slot bytes through its own mapping of the shared segment; this
    /// single-process port has no other writer path, so the views are public
    /// API here (see the module doc's divergences).
    ///
    /// # Errors
    /// [`SharedHostPoolError::SlotOutOfRange`] if `slot >= slot_count()`.
    pub fn slot(&self, slot: usize) -> Result<&[u8], SharedHostPoolError> {
        let offset = self.geometry.slot_offset(slot)?;
        Ok(&self.buffer.as_slice()[offset..offset + self.geometry.slot_bytes()])
    }

    /// Checked write view of slot `slot`'s bytes (the full aligned stride).
    ///
    /// The caller owns the discipline the vendor's design assumes: do not
    /// rewrite a slot's bytes while a transfer that has not completed is
    /// still reading/writing them (`synchronize()` first — see the struct
    /// doc's transfer-lifetime law).
    ///
    /// # Errors
    /// [`SharedHostPoolError::SlotOutOfRange`] if `slot >= slot_count()`.
    pub fn slot_mut(&mut self, slot: usize) -> Result<&mut [u8], SharedHostPoolError> {
        let offset = self.geometry.slot_offset(slot)?;
        let stride = self.geometry.slot_bytes();
        Ok(&mut self.buffer.as_mut_slice()[offset..offset + stride])
    }

    /// Port of `SharedPool::EnqueueCopy` / `SharedHostPool`'s
    /// `CopyRawPreMapped` entry (reached on `RuntimeStream` via
    /// `runtime_stream.hpp:148`/`:175`): enqueue a DMA between slot `slot`
    /// and a single-chunk device object on the DEFAULT (async DMA) pipeline,
    /// riding the pool's pre-mapped IOVA — no staging buffer, no per-step pin.
    ///
    /// Validation is the vendor's `copyRawImpl` set, in the vendor's order,
    /// all firing BEFORE any DMA is issued: the stream must not be in the
    /// shutdown state; the device address must be non-null; the device
    /// object must be single-chunk; and its `total_size` (always the
    /// transfer length) must fit the slot's capacity — a larger object would
    /// DMA past the slot into its NEIGHBOR. Then the copy is submitted as an
    /// ordinary `DmaParams` with `iova` set, which is the field the
    /// scheduler's pre-mapped path already honors (it skips the backend's
    /// RAII shadow-buffer copy and uses the caller's IOVA verbatim).
    ///
    /// # Errors
    /// See [`SharedHostPoolError`]: slot range first (the
    /// `CopyRawPreMapped`-side precondition), then the `copyRawImpl` set, or
    /// [`SharedHostPoolError::Launch`] if the stream rejects the submission.
    pub fn enqueue_copy(
        &self,
        stream: &Arc<RuntimeStream>,
        slot: usize,
        device_address: Option<Arc<CompositeAddress>>,
        to_device: bool,
    ) -> Result<(), SharedHostPoolError> {
        self.enqueue_copy_shaped(
            stream,
            slot,
            device_address,
            to_device,
            DmaPipelineShape::AsyncDma,
        )
    }

    /// ⭐ #603 PR3b — [`Self::enqueue_copy`] on the COMPUTE pipeline: the pool's
    /// twin of the runtime's `h2d_on_compute`/`d2h_on_compute` pair, for the
    /// copies that must ride FIFO-behind a step's compute CBs (the decode
    /// window's idx-stick D2H, the LAST op of its step). The same junction law
    /// those doors carry (#544): on the default pipeline the copy's switch
    /// barrier host-blocks on the compute deque; on COMPUTE the barrier drains
    /// the (empty) DMA deques instead and the queue's own FIFO carries the
    /// ordering the host-side wait used to.
    ///
    /// # Errors
    /// Identical to [`Self::enqueue_copy`] — the doors differ only in the
    /// pipeline the copy rides.
    pub fn enqueue_copy_on_compute(
        &self,
        stream: &Arc<RuntimeStream>,
        slot: usize,
        device_address: Option<Arc<CompositeAddress>>,
        to_device: bool,
    ) -> Result<(), SharedHostPoolError> {
        self.enqueue_copy_shaped(
            stream,
            slot,
            device_address,
            to_device,
            DmaPipelineShape::ComputePipeline,
        )
    }

    /// The one body behind [`Self::enqueue_copy`] and
    /// [`Self::enqueue_copy_on_compute`]: the `CopyRawPreMapped` precondition
    /// and null-IOVA belt, the `copyRawImpl` validation set in the vendor's
    /// order, and the pre-mapped submission — parameterized ONLY by the
    /// pipeline the copy rides, the same factor the runtime's own DMA door
    /// pairs take.
    fn enqueue_copy_shaped(
        &self,
        stream: &Arc<RuntimeStream>,
        slot: usize,
        device_address: Option<Arc<CompositeAddress>>,
        to_device: bool,
        pipeline_shape: DmaPipelineShape,
    ) -> Result<(), SharedHostPoolError> {
        // CopyRawPreMapped's side of the contract: resolve the slot and
        // refuse a null IOVA before descending into the copy validation.
        // (A constructed pool always has a non-zero base IOVA — the
        // constructor rejects 0 — so this is the belt to that braces.)
        let resolved = self.resolve_slot(slot)?;
        if resolved.iova.0 == 0 {
            return Err(SharedHostPoolError::NullIova);
        }
        // copyRawImpl's checks, in order.
        if stream.needs_shutdown() {
            return Err(SharedHostPoolError::NeedsShutdown);
        }
        let device = device_address.ok_or(SharedHostPoolError::NullDeviceAddress)?;
        validate_copy_raw_device(&device, resolved.host_capacity)?;
        let params = copy_raw_params(resolved, &device, to_device, pipeline_shape)?;
        let launch = if to_device {
            stream.launch_h2d(params, None)
        } else {
            stream.launch_d2h(params, None)
        };
        launch.map_err(SharedHostPoolError::Launch)
    }
}

/// `createDmaParams(host, total_size, to_device, dev, dci = null, iova)` — the
/// pre-mapped `DmaParams` both copy doors submit, extracted as a pure function
/// because the launch half needs a live stream while the shaping half is
/// exactly what a cardless test must pin: the transfer length is ALWAYS the
/// device object's `total_size` (checked to fit the slot by the caller), the
/// op name is the copyRaw pair, the pool's stable slot IOVA rides the
/// pre-mapped field, and the pipeline is the door's own shaping.
fn copy_raw_params(
    resolved: ResolvedSlot,
    device: &Arc<CompositeAddress>,
    to_device: bool,
    pipeline_shape: DmaPipelineShape,
) -> Result<DmaParams, SharedHostPoolError> {
    let op_name = if to_device {
        "RuntimeStreamCopyRawH2D"
    } else {
        "RuntimeStreamCopyRawD2H"
    };
    let direction = if to_device {
        DmaDirection::HostToDevice
    } else {
        DmaDirection::DeviceToHost
    };
    match DmaParams::with_explicit_size(
        resolved.host_address,
        device.total_size(),
        direction,
        Some(Arc::clone(device)),
        DmaShaping {
            dci: None,
            op_name: OpName(Some(op_name.to_string())),
            iova: Some(resolved.iova),
            pipeline_shape,
            ..DmaShaping::default()
        },
    ) {
        Ok(p) => Ok(p),
        // Unreachable — `device` is a shared reference to a live `Arc` — but
        // the null check is `with_explicit_size`'s own, so map it rather
        // than assert it away.
        Err(DmaParamsError::NullDeviceAddress) => Err(SharedHostPoolError::NullDeviceAddress),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::{ByteOffset, ByteSize, Chunk, LogicalAddress, RegionId};
    use crate::memory_region::DomainId;

    // dma.rs's dummy-address idiom: a CompositeAddress is a plain host-side
    // descriptor, so a synthetic single-chunk one exercises the copyRaw
    // validation set with no hardware in the loop.
    fn dummy_addr(size: u64) -> Arc<CompositeAddress> {
        let chunk = Chunk::new(
            LogicalAddress::new(RegionId(1), ByteOffset(0)),
            ByteSize(size),
            DomainId(0),
        );
        Arc::new(CompositeAddress::from_chunk(chunk))
    }

    fn dummy_two_chunk_addr(size: u64) -> Arc<CompositeAddress> {
        let chunk = Chunk::new(
            LogicalAddress::new(RegionId(1), ByteOffset(0)),
            ByteSize(size),
            DomainId(0),
        );
        Arc::new(CompositeAddress::from_chunks(vec![chunk, chunk]))
    }

    // --- geometry (all pure; no allocation, no senlib) ---

    #[test]
    fn geometry_rejects_zero_slots() {
        let err = PoolGeometry::new(0, 4096, 4096).unwrap_err();
        assert!(matches!(err, SharedHostPoolError::ZeroSlots));
    }

    #[test]
    fn geometry_rejects_zero_slot_bytes() {
        let err = PoolGeometry::new(4, 0, 4096).unwrap_err();
        assert!(matches!(err, SharedHostPoolError::ZeroSlotBytes));
    }

    #[test]
    fn geometry_rejects_zero_alignment() {
        // A mapper reporting alignment 0 would make the round-up divide by
        // zero; garbage refuses instead.
        let err = PoolGeometry::new(4, 4096, 0).unwrap_err();
        assert!(matches!(err, SharedHostPoolError::ZeroAlignment));
    }

    #[test]
    fn geometry_rounds_slot_bytes_up_to_iova_alignment() {
        // The SlotBytes() law: requested 100 at alignment 64 rounds to 128.
        let g = PoolGeometry::new(4, 100, 64).unwrap();
        assert_eq!(g.slot_count(), 4);
        assert_eq!(g.slot_bytes(), 128);
        assert_eq!(g.total_bytes(), 512);
        // An already-aligned request is untouched.
        let g = PoolGeometry::new(2, 4096, 4096).unwrap();
        assert_eq!(g.slot_bytes(), 4096);
        assert_eq!(g.total_bytes(), 8192);
    }

    #[test]
    fn geometry_stride_overflow_refuses() {
        // requested near usize::MAX with a non-zero remainder: the round-up
        // add overflows and must refuse, not wrap.
        let requested = usize::MAX - 10;
        let err = PoolGeometry::new(2, requested, 64).unwrap_err();
        assert!(matches!(err, SharedHostPoolError::StrideOverflow));
    }

    #[test]
    fn geometry_total_size_overflow_refuses() {
        // num_slots * stride past usize::MAX refuses at construction, so
        // total_bytes() stays infallible.
        let err = PoolGeometry::new(usize::MAX, 1 << 40, 1).unwrap_err();
        assert!(matches!(err, SharedHostPoolError::TotalSizeOverflow));
    }

    #[test]
    fn slot_offset_is_index_times_stride() {
        let g = PoolGeometry::new(4, 100, 64).unwrap();
        assert_eq!(g.slot_offset(0).unwrap(), 0);
        assert_eq!(g.slot_offset(1).unwrap(), 128);
        assert_eq!(g.slot_offset(3).unwrap(), 384);
    }

    #[test]
    fn slot_offset_rejects_out_of_range() {
        let g = PoolGeometry::new(4, 100, 64).unwrap();
        let err = g.slot_offset(4).unwrap_err();
        assert!(matches!(
            err,
            SharedHostPoolError::SlotOutOfRange {
                slot: 4,
                slot_count: 4
            }
        ));
    }

    // --- the index-addressing law (synthetic bases, the scheduler-test
    // precedent for feeding synthetic IOVAs into pure arithmetic) ---

    #[test]
    fn resolved_slots_advance_by_stride_in_both_address_spaces() {
        let g = PoolGeometry::new(4, 100, 64).unwrap();
        let base_hmva = 0x1_0000_0000usize;
        let base_iova = 0x2_0000_0000usize;
        let r0 = resolved_slot_at(base_hmva, base_iova, &g, 0).unwrap();
        assert_eq!(r0.host_address, HostVirtualAddress(base_hmva));
        assert_eq!(r0.iova, IovaAddress(base_iova));
        assert_eq!(r0.host_capacity, 128);
        let r2 = resolved_slot_at(base_hmva, base_iova, &g, 2).unwrap();
        assert_eq!(r2.host_address, HostVirtualAddress(base_hmva + 256));
        assert_eq!(r2.iova, IovaAddress(base_iova + 256));
        // The stable-IOVA shape: two resolves of the same slot are identical.
        assert_eq!(resolved_slot_at(base_hmva, base_iova, &g, 2).unwrap(), r2);
    }

    #[test]
    fn resolved_slot_rejects_out_of_range() {
        let g = PoolGeometry::new(2, 4096, 4096).unwrap();
        assert!(matches!(
            resolved_slot_at(0x1000, 0x2000, &g, 2).unwrap_err(),
            SharedHostPoolError::SlotOutOfRange {
                slot: 2,
                slot_count: 2
            }
        ));
    }

    // --- construction validation fires before any senlib call ---
    //
    // PfIommuMapper::new() itself is cardless (sysconf + a table), so these
    // prove the refusal order: an invalid geometry returns an error before
    // the buffer allocation and long before the senlib pin, which is what
    // makes them safe to run on a machine with no hardware.

    #[test]
    fn pool_construction_validates_geometry_before_any_mapping() {
        let mapper = PfIommuMapper::new();
        // `matches!` rather than `unwrap_err()`: the Ok type owns a live
        // mapping and has no `Debug`, and the point of the test is only
        // WHICH refusal came back.
        assert!(matches!(
            SharedHostPool::new(0, 4096, &mapper),
            Err(SharedHostPoolError::ZeroSlots)
        ));
        assert!(matches!(
            SharedHostPool::new(4, 0, &mapper),
            Err(SharedHostPoolError::ZeroSlotBytes)
        ));
        assert!(matches!(
            SharedHostPool::new(usize::MAX, 1 << 40, &mapper),
            Err(SharedHostPoolError::TotalSizeOverflow)
        ));
    }

    // --- the copyRaw validation set (pure, over synthetic device
    // descriptors — dma.rs's dummy_addr idiom) ---

    #[test]
    fn copy_raw_accepts_single_chunk_device_fitting_the_slot() {
        // Equal to capacity is legal: the transfer fills the slot exactly.
        assert!(validate_copy_raw_device(&dummy_addr(4096), 4096).is_ok());
        assert!(validate_copy_raw_device(&dummy_addr(2048), 4096).is_ok());
    }

    #[test]
    fn copy_raw_refuses_device_larger_than_the_slot() {
        // The neighboring-slot overrun refusal — fires before any DMA.
        let err = validate_copy_raw_device(&dummy_addr(8192), 4096).unwrap_err();
        assert!(matches!(
            err,
            SharedHostPoolError::DeviceSizeExceedsSlot {
                device_size: 8192,
                host_capacity: 4096
            }
        ));
    }

    #[test]
    fn copy_raw_refuses_multi_chunk_device() {
        // requireSingleChunk: backends are inconsistent on multi-chunk
        // device addresses, so the pre-mapped path refuses them.
        let err = validate_copy_raw_device(&dummy_two_chunk_addr(4096), 8192).unwrap_err();
        assert!(matches!(
            err,
            SharedHostPoolError::MultiChunkDevice { chunk_count: 2 }
        ));
    }

    // --- the copy doors' shaping (pure params; the launch itself needs a live
    // stream, but a copy riding the WRONG pipeline is a silent mis-submission,
    // so the shaping is pinned here) ---

    #[test]
    fn copy_raw_params_rides_the_pool_iova_and_the_doors_pipeline() {
        let resolved = ResolvedSlot {
            host_address: HostVirtualAddress(0x1000_0000),
            host_capacity: 4096,
            iova: IovaAddress(0x2000_0000),
        };
        let device = dummy_addr(256);

        let d2h_compute =
            copy_raw_params(resolved, &device, false, DmaPipelineShape::ComputePipeline).unwrap();
        // The compute door's whole point: #544's junction law, FIFO-behind the
        // step's compute CBs instead of host-blocking on the compute deque.
        assert_eq!(
            d2h_compute.pipeline_shape(),
            DmaPipelineShape::ComputePipeline
        );
        // The pre-mapped fields: the slot's OWN stable IOVA (never a fresh
        // mapping), the slot's host address, the device object's total size as
        // the transfer length, and the direction.
        assert_eq!(d2h_compute.iova(), Some(IovaAddress(0x2000_0000)));
        assert_eq!(d2h_compute.hmva(), HostVirtualAddress(0x1000_0000));
        assert_eq!(d2h_compute.dma_size(), ByteSize(256));
        assert_eq!(d2h_compute.direction(), DmaDirection::DeviceToHost);

        let h2d_default =
            copy_raw_params(resolved, &device, true, DmaPipelineShape::AsyncDma).unwrap();
        assert_eq!(h2d_default.pipeline_shape(), DmaPipelineShape::AsyncDma);
        assert_eq!(h2d_default.direction(), DmaDirection::HostToDevice);
        assert_eq!(h2d_default.iova(), Some(IovaAddress(0x2000_0000)));
    }

    #[test]
    fn copy_raw_params_op_name_names_the_direction() {
        let resolved = ResolvedSlot {
            host_address: HostVirtualAddress(0x1000_0000),
            host_capacity: 4096,
            iova: IovaAddress(0x2000_0000),
        };
        let device = dummy_addr(512);
        // `OpName` has no `PartialEq`; the vendor pair is the observable, and
        // it is what a CB trace names the op by.
        let h2d =
            copy_raw_params(resolved, &device, true, DmaPipelineShape::ComputePipeline).unwrap();
        assert_eq!(h2d.op_name().0.as_deref(), Some("RuntimeStreamCopyRawH2D"));
        let d2h =
            copy_raw_params(resolved, &device, false, DmaPipelineShape::ComputePipeline).unwrap();
        assert_eq!(d2h.op_name().0.as_deref(), Some("RuntimeStreamCopyRawD2H"));
    }

    // Everything else the vendor suite covers needs a live senlib link
    // (a real whole-region pin, a real stream, a real device object): the
    // null-IOVA-at-pin refusal, resolve/slot views over a real mapping, the
    // H2D/D2H round trip, and teardown ordering. See the module doc's vendor
    // test mapping — the `shared_host_pool_probe` example in
    // scratchy-target-spyre is the card leg for exactly that set.
}

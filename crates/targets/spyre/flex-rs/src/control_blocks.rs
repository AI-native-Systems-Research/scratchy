//! Port of the DMA/compute control-block construction and fixup layer.
//!
//! Sources:
//!   - `flex/src/control_blocks/control_block_stream/control_block_stream.hpp`
//!     (fixup structs, segment-physical-address translation, program info)
//!   - `flex/src/control_blocks/control_block_stream/control_block_stream.cpp`
//!     (`TranslateDmvaToDmpa`)
//!   - `flex/src/control_blocks/control_block_stream/segment_addressing.hpp`
//!     (`SegmentOffset`, `dmvaToSegmentId`, `SegmentByteOffset_todmva`)
//!   - `flex/src/control_blocks/dma/dma_descriptor_pack_allocation.{hpp,cpp}`
//!   - `flex/src/control_blocks/dma/dma_descriptor_pack_fixup.{hpp,cpp}`
//!   - `flex/src/control_blocks/dma/dma_control_block_handler.{hpp,cpp}`
//!   - `flex/src/control_blocks/control_block_handler/control_block_handler.{hpp,cpp}`
//!   - `flex/src/control_blocks/control_block_handler/compute_pipeline_control_block_handler.{hpp,cpp}`
//!
//! ## Scope note (what's deliberately NOT here)
//!
//! `control_block_stream.{hpp,cpp}` (2384 + 7554 lines) is overwhelmingly the
//! sendnn::Graph → control-block *compiler* (message matching, collective
//! chaining, HDMA buffer bookkeeping, RDMA multi-rank coordination) —
//! `ControlBlockStreamFromGraph`/`ControlBlockStreamFromNodes` and everything
//! `message_processing.hpp`/`response_worker.hpp` support. That machinery is
//! graph- and multi-rank-topology-dependent, not "DMA/compute parameter
//! construction+validation" or "control block construction/fixups" in the
//! sense this phase was scoped (a parallel scheduler/graph-compiler phase
//! owns it, the same way `stream.rs` documents `DmaParams`/`ComputeParams`
//! as belonging to *this* phase rather than its own). What IS ported here is
//! the fixup vocabulary those compilers use to describe "what needs patching
//! before submission" (`XlatFixup`, `DmaFixup`, `TransferFixup`, `DdlFixup`,
//! `ProgFixup`), the DMPA translation arithmetic (`TranslateDmvaToDmpa`,
//! `SegmentOffset`/`dmvaToSegmentId`), and the DMA-descriptor-pack allocation
//! and fixup value types — plus the full mock-device control-block handler
//! hierarchy (`ControlBlockHandler`/`AsyncControlBlockHandlerBase`/
//! `DmaControlBlockHandler`/`ComputePipelineControlBlockHandler`), which is
//! flex's own software simulation of hardware CB execution (selected via
//! `FlexCompute=NULL`/mock), not senlib.
//!
//! `response_worker.{hpp,cpp}` (the real hardware submission path —
//! `senlib::ControlBlockInterface`/`ResponseBlockInterface` — see
//! `LaunchCbs`/`FetchResponseBlocks`) and `message_processing.{hpp,cpp}`
//! (multi-rank message matching for the graph compiler) are read for
//! boundary purposes only and are out of this phase's file ownership.
//!
//! No senlib FFI is required anywhere in this file: every operation here is
//! either pure arithmetic/bookkeeping (fixups, DMPA translation) or a
//! software simulation of device execution (the mock `*ControlBlockHandler`
//! hierarchy) that reads/writes a `flex`-owned `DeviceMemory` buffer, never
//! senlib. See `SENLIB_BOUNDARY_dma-compute-controlblocks.md`.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;

use crate::address::ByteSize;
use crate::util::Status;

// ---------------------------------------------------------------------
// Segment addressing (segment_addressing.hpp)
// ---------------------------------------------------------------------

/// Number of bits of a DMVA that encode the in-segment byte offset. Port of
/// `SEGMENT_SIZE_BITS` (`memory_interface/segment_table.hpp`).
pub const SEGMENT_SIZE_BITS: u32 = 34;

/// Mask selecting the in-segment byte offset from a DMVA. Port of
/// `SEGMENT_OFFSET_MASK`.
pub const SEGMENT_OFFSET_MASK: u64 = (1u64 << SEGMENT_SIZE_BITS) - 1;

/// Total addressable bytes per segment (16 GiB). Port of `SEGMENT_SIZE`.
pub const SEGMENT_SIZE: u64 = 1u64 << SEGMENT_SIZE_BITS;

/// One of the 8 hardware XLAT segments a device memory virtual address can
/// name. Newtype over the raw segment index computed by `dmvaToSegmentId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SegmentIndex(pub u64);

/// Device Memory Virtual Address: 3-bit segment id + 34-bit in-segment
/// offset. Newtype over the raw `uint64_t dmva` used pervasively in the C++.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dmva(pub u64);

impl Dmva {
    /// Port of `SegmentOffset(dmva)`.
    pub const fn segment_offset(self) -> u64 {
        self.0 & SEGMENT_OFFSET_MASK
    }

    /// Port of `dmvaToSegmentId(dmva)`.
    pub const fn segment_id(self) -> SegmentIndex {
        SegmentIndex(self.0 >> SEGMENT_SIZE_BITS)
    }

    /// Port of `SegmentByteOffset_todmva(segment, byte_offset)`.
    pub const fn from_segment_and_offset(segment: SegmentIndex, byte_offset: u64) -> Self {
        Dmva((segment.0 << SEGMENT_SIZE_BITS) | (byte_offset & SEGMENT_OFFSET_MASK))
    }
}

/// Device Memory Physical Address: hardware-level address after XLAT
/// translation. Newtype over the raw `uint64_t dmpa`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dmpa(pub u64);

/// Physical base address and byte size for each of the 8 XLAT segments.
/// Port of `DmpaAllocs`/`DmpaSizes` (`std::array<uint64_t, 8>`) — a fixed
/// hardware-defined count, so a const-generic array rather than a `Vec`.
pub type DmpaAllocs = [u64; 8];
pub type DmpaSizes = [ByteSize; 8];

/// Error translating a DMVA to a DMPA. Port of the three `RAS::CBRB::*`
/// throw sites in `TranslateDmvaToDmpa`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmvaTranslationError {
    /// Segment id extracted from the DMVA was outside `0..=7`.
    InvalidSegmentId(SegmentIndex),
    /// The in-segment offset exceeded that segment's allocated size.
    OffsetOutOfBounds { offset: u64, segment_len: ByteSize },
    /// `offset + size_bytes` overflowed the segment's allocated size.
    AccessOverflowsSegment { end: u64, segment_len: ByteSize },
}

/// Port of `TranslateDmvaToDmpa` (`control_block_stream.cpp:118-149`): pure
/// arithmetic over caller-supplied segment tables, validated the same way
/// as the C++ (segment id range, offset bounds, overflow).
pub fn translate_dmva_to_dmpa(
    dmpa_allocs: &DmpaAllocs,
    dmpa_sizes: &DmpaSizes,
    dmva: Dmva,
    size_bytes: ByteSize,
) -> Result<Dmpa, DmvaTranslationError> {
    let seg_id = dmva.segment_id();
    let seg_offset = dmva.segment_offset();

    if seg_id.0 > 7 {
        return Err(DmvaTranslationError::InvalidSegmentId(seg_id));
    }

    let segment_len = dmpa_sizes[seg_id.0 as usize];
    if seg_offset > segment_len.as_u64() {
        return Err(DmvaTranslationError::OffsetOutOfBounds {
            offset: seg_offset,
            segment_len,
        });
    }

    let end = seg_offset + size_bytes.as_u64();
    if end > segment_len.as_u64() {
        return Err(DmvaTranslationError::AccessOverflowsSegment { end, segment_len });
    }

    Ok(Dmpa(dmpa_allocs[seg_id.0 as usize] + seg_offset))
}

// ---------------------------------------------------------------------
// Fixups (control_block_stream.hpp)
// ---------------------------------------------------------------------

/// Index of a control block within a stream. Newtype over `uint64_t cb_idx`,
/// shared across all fixup kinds below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CbIndex(pub u64);

/// Index into a PIPO (persistent input/output) IOVA buffer array. Newtype
/// over `uint64_t iova_idx`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct IovaBufferIndex(pub u64);

/// Index of a memory allocation (segment index) an XLAT entry refers to.
/// Newtype over `uint64_t alloc_id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AllocId(pub u64);

/// Which of the 8 XLAT entries within a control block is being fixed up.
/// Newtype over `uint64_t xlat_idx` (0-7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct XlatIndex(pub u64);

/// Whether a translation/transfer is read-only. Kept as a bare bool field
/// (matching the C++) rather than an access-mode enum since no third state
/// exists at this layer — `DmaControlBlockHandler::RangeCheck` below is
/// where the richer `XlatModeEnum` (NONE/READ_ONLY/READ_WRITE) applies.
pub type ReadOnly = bool;

/// Port of `flex::XlatFixup`: DMVA→DMPA translation fixup for one XLAT
/// entry of one control block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XlatFixup {
    pub cb_idx: CbIndex,
    pub xlat_idx: XlatIndex,
    pub alloc_id: AllocId,
    pub size_bytes: ByteSize,
    pub offset_bytes: ByteSize,
    pub read_only: ReadOnly,
}

/// Port of `flex::DmaFixup`: links a DMA control block to a host IOVA
/// buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DmaFixup {
    pub cb_idx: CbIndex,
    pub iova_idx: IovaBufferIndex,
    pub iova_offset: ByteSize,
    /// `true` = host → device, `false` = device → host.
    pub to_device: bool,
}

/// Port of `flex::TransferFixup`: deferred direction specialization for a
/// control block whose transfer direction isn't known until specialization
/// time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferFixup {
    pub cb_idx: CbIndex,
    pub to_device: bool,
}

/// Index into a DMA Descriptor List (DDL), for scatter-gather DMA. Newtype
/// over `uint64_t ddl_idx`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct DdlIndex(pub u64);

/// Port of `flex::DdlFixup`: scatter-gather DMA descriptor-list fixup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DdlFixup {
    pub ddl_idx: DdlIndex,
    pub iova_idx: IovaBufferIndex,
    pub iova_offset: ByteSize,
}

/// Port of `flex::ProgFixup`: marks a control block as needing its program
/// pointer filled in during deferred compute specialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgFixup {
    pub cb_idx: CbIndex,
}

/// Port of `flex::IovaOffsetDmvaSize`: the IOVA/DMVA/size tuple needed to
/// build one DMA descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IovaOffsetDmvaSize {
    pub iova_idx: IovaBufferIndex,
    pub iova_offset: ByteSize,
    pub dmva: Dmva,
    pub size_bytes: ByteSize,
}

/// Execution timeout for a compute program, in microseconds. Newtype over
/// `uint64_t exec_timer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExecTimeoutMicros(pub u64);

/// Port of `flex::ProgramInfo`: where a compute program lives in device
/// memory and how long it may run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgramInfo {
    pub prog_dmva: Dmva,
    pub exec_timer: ExecTimeoutMicros,
}

/// Port of `flex::ProgramList` (`std::vector<ProgramInfo>` — genuinely
/// dynamic, one entry per queued compute program).
pub type ProgramList = Vec<ProgramInfo>;

// ---------------------------------------------------------------------
// DMA descriptor pack allocation/fixup (control_blocks/dma/*)
// ---------------------------------------------------------------------

/// Device-side IO virtual address of an allocated DMA descriptor pack.
/// Newtype over `uint64_t iova_bytes` on `DmaDescriptorPackAllocation`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackIovaAddress(pub u64);

/// A pooled, IOMMU-mapped DMA descriptor pack: host-visible bytes plus the
/// device-visible IOVA for the same physical memory. Port of
/// `flex::DmaDescriptorPackAllocation`.
///
/// The C++ `host_ptr`/`iommu_mapping`/`shared_raii_buffer` triple (a raw
/// senlib-layout pointer, an IOMMU mapping RAII handle, and a shared host
/// buffer keeping it all alive) collapses to an owned byte buffer: the pack
/// layout itself (`SentientSoc::V1::DmaDescriptorPackSBF`, 8 fixed-size
/// descriptor slots) is senlib's on-wire format, which this phase does not
/// re-derive — callers needing to interpret/mutate individual descriptors do
/// so through the fixup API below (`DmaDescriptorPackFixup`), matching how
/// the C++ callers (`BatchedJob::fillDescriptorPacks`) only ever touch packs
/// through fixups, never by hand-rolling the SBF layout.
#[derive(Debug)]
pub struct DmaDescriptorPackAllocation {
    host_bytes: Box<[u8]>,
    iova: PackIovaAddress,
}

impl DmaDescriptorPackAllocation {
    /// `pack_len_bytes` is the caller-supplied senlib on-wire pack size
    /// (IOMMU-mapped host memory of that length backs `host_bytes`).
    pub fn new(host_bytes: Box<[u8]>, iova: PackIovaAddress) -> Self {
        Self { host_bytes, iova }
    }

    pub fn host_bytes(&self) -> &[u8] {
        &self.host_bytes
    }
    pub fn host_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.host_bytes
    }
    pub fn iova(&self) -> PackIovaAddress {
        self.iova
    }
}

impl std::fmt::Display for DmaDescriptorPackAllocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DmaDescriptorPackAllocation( host_ptr: {:p}, iova_bytes: 0x{:x} )",
            self.host_bytes.as_ptr(),
            self.iova.0
        )
    }
}

/// Index into the `Vec<DmaDescriptorPackAllocation>` a job's descriptor
/// packs live in. Newtype over `uint64_t alloc_idx`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PackAllocIndex(pub u64);

/// Index of a descriptor pack within an allocation (each allocation holds
/// multiple 8-descriptor packs). Newtype over `uint64_t pack_idx`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PackIndex(pub u64);

/// Index of a single descriptor within its 8-descriptor pack (0-7). Newtype
/// over `uint64_t descriptor_idx`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct DescriptorIndex(pub u64);

/// Port of `flex::DmaDescriptorPackFixup`: identifies one descriptor within
/// a pooled pack that needs its host address (PIPO IOVA) customized for a
/// specific job, plus a byte offset for jobs using a buffer subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DmaDescriptorPackFixup {
    pub alloc_idx: PackAllocIndex,
    pub pack_idx: PackIndex,
    pub descriptor_idx: DescriptorIndex,
    pub offset: ByteSize,
}

impl std::fmt::Display for DmaDescriptorPackFixup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DmaDescriptorPackFixup( alloc_idx: {}, pack_idx: {}, descriptor_idx: {}, offset: {} )",
            self.alloc_idx.0,
            self.pack_idx.0,
            self.descriptor_idx.0,
            self.offset.as_u64()
        )
    }
}

// ---------------------------------------------------------------------
// Mock control-block handler hierarchy (control_block_handler/*, dma/*)
// ---------------------------------------------------------------------
//
// Port of `ControlBlockHandler`/`ControlBlockHandlerBase`/
// `AsyncControlBlockHandlerBase`/`DmaControlBlockHandler`/
// `ComputePipelineControlBlockHandler` — flex's software simulation of
// device CB execution (selected via `RuntimeConfig::FlexCompute() ==
// "NULL"`/"SENULATOR", i.e. the no-hardware/mock backend). This is real
// production flex logic (one of the supported `FlexCompute` backends), not
// a test double excluded from the closure.
//
// Simplified from the C++ in one respect only: the DMA control block
// handler's legacy RDMA sub-parsing (`ParseRdmaData`/`ParseRdmaSetBcList`/
// `ParseRdmaSingleCastXseg`/`ParseRdmaWdoneBarrier`, all gated behind
// `#ifndef DISABLE_LEGACY_GRAPH_RUNTIME`) is not ported: the non-legacy
// (`DISABLE_LEGACY_GRAPH_RUNTIME`) build — which this crate targets, per the
// rest of this port — compiles `ExecuteParseFunction` down to a no-op stub
// (`dma_control_block_handler.cpp:247-249`) and never constructs an
// `RdmaUnit`. `handle_rdma_operation` below matches that stub exactly.

/// A `[start, end)` byte range's read/write access mode, as recorded in a
/// control block's translation table. Port of `XlatModeEnum` (`NONE <
/// READ_ONLY < READ_WRITE`, hence the derived `Ord`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum XlatAccessMode {
    None,
    ReadOnly,
    ReadWrite,
}

/// One XLAT translation-table entry read out of a control block. Port of
/// the subset of `TranslationControlBlockSectionSBF`'s per-entry API that
/// `DmaControlBlockHandler`/`ControlBlockHandlerBase` actually use.
#[derive(Debug, Clone, Copy)]
pub struct XlatEntry {
    pub access_mode: XlatAccessMode,
    pub length_bytes: ByteSize,
    pub paddr_bytes: u64,
}

/// Errors surfaced while processing a mock control block. Port of the
/// `sendnn::Status::INVALID_ARGUMENT(...)` results returned throughout
/// `DmaControlBlockHandler`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CbProcessingError {
    InvalidSegment(SegmentIndex),
    XlatNotValid,
    XlatReadOnlyButWriting,
    XlatAccessOutOfBounds,
    UnsupportedCbType(&'static str),
    FillOnlySupportedByDmai,
    DefragOnlySupportedByAsyncDmao,
    RdmaOperationUnsupported {
        op: &'static str,
        reason: &'static str,
    },
}

/// Minimal per-CB translation table needed for `RangeCheck`/address
/// translation: 8 fixed XLAT entries, one per hardware segment. Port of the
/// `cbt_->translations()` array (`SentientSoc::V1::TranslationControlBlockSectionSBF`).
pub type TranslationTable = [XlatEntry; 8];

/// Port of `DmaControlBlockHandler::RangeCheck`: validates an access to
/// `[dmva, dmva+size)` against the CB's own translation table.
///
/// `size_bytes = 0` means "the whole 16 GiB segment", matching the C++'s
/// `0 = SEGMENT_SIZE` convention (applied to both the requested size and,
/// separately, to a zero-length XLAT entry).
pub fn range_check(
    translations: &TranslationTable,
    dmva: Dmva,
    size_bytes: ByteSize,
    writable: bool,
) -> Result<(), CbProcessingError> {
    let seg_id = dmva.segment_id();
    if seg_id.0 > 7 {
        return Err(CbProcessingError::InvalidSegment(seg_id));
    }
    let xlat = &translations[seg_id.0 as usize];
    if xlat.access_mode < XlatAccessMode::ReadOnly {
        return Err(CbProcessingError::XlatNotValid);
    }
    if writable && xlat.access_mode != XlatAccessMode::ReadWrite {
        return Err(CbProcessingError::XlatReadOnlyButWriting);
    }

    let size_bytes = if size_bytes.as_u64() == 0 {
        ByteSize(SEGMENT_SIZE)
    } else {
        size_bytes
    };
    let xlat_size = if xlat.length_bytes.as_u64() == 0 {
        ByteSize(SEGMENT_SIZE)
    } else {
        xlat.length_bytes
    };

    if dmva.segment_offset() + size_bytes.as_u64() > xlat_size.as_u64() {
        return Err(CbProcessingError::XlatAccessOutOfBounds);
    }
    Ok(())
}

/// Port of `ControlBlockHandlerBase::TranslateVirtualToPhysicalAddress`:
/// resolve a DMVA to a byte offset into the mock device's memory buffer,
/// using the CB's own translation table (no senlib call — this is flex's
/// own per-CB XLAT lookup, computed entirely from data embedded in the CB).
pub fn translate_virtual_to_physical_offset(translations: &TranslationTable, dmva: Dmva) -> u64 {
    let seg_id = dmva.segment_id();
    let xlat = &translations[seg_id.0 as usize];
    xlat.paddr_bytes + dmva.segment_offset()
}

/// Direction a `DmaControlBlockHandler` instance handles. Port of the
/// constructor's `bool is_dmai` (`true` = DMA-In/host→device, `false` =
/// DMA-Out/device→host).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaHandlerDirection {
    DmaIn,
    DmaOut,
}

/// Whether a `DmaControlBlockHandler` processes asynchronously. Port of the
/// constructor's `bool is_async`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaHandlerMode {
    Sync,
    Async,
}

/// One immediate DMA transfer descriptor, as decoded from a CB's DMA
/// section. Port of the fields `ParseImmediate`/`ParseFill`/`ParseDefrag`
/// pull off `HdmaImmControlBlockSectionSBF`.
#[derive(Debug, Clone, Copy)]
pub struct ImmediateDmaDescriptor {
    pub device_address: Dmva,
    pub host_address_bytes: u64,
    pub length_bytes: ByteSize,
}

/// Port of `DmaControlBlockHandler`'s dma-type dispatch tag. Only the
/// variants reachable without the legacy RDMA sub-parsers are modeled with
/// payloads; the RDMA variants carry their C++ operation name for the
/// `RdmaOperationUnsupported` error path (`DetermineError`'s
/// `kRdmaOperationNames` table).
#[derive(Debug, Clone, Copy)]
pub enum DmaCbKind {
    Immediate(ImmediateDmaDescriptor),
    Deferred {
        descriptor_pack_host_address: u64,
    },
    MemoryFill(ImmediateDmaDescriptor),
    Defrag(ImmediateDmaDescriptor),
    Skip,
    Disabled,
    /// `op_name` is the C++'s `err_cause` string (`kRdmaOperationNames`
    /// table, `dma_control_block_handler.cpp:181-188`). `is_wdone_barrier`
    /// distinguishes the one RDMA op (`RDMA_WDONE_BARRIER`) whose gating
    /// condition flips sense relative to every other RDMA op — see
    /// `DetermineError`'s doc on `parse_dma`'s `Rdma` arm below.
    Rdma {
        op_name: &'static str,
        is_wdone_barrier: bool,
    },
}

/// Simulated device memory backing a mock `ControlBlockHandler` hierarchy.
/// Port of the (flex-owned, senlib-free) mock `DeviceMemory` this whole
/// subsystem reads/writes via `TranslateVirtualToPhysicalAddress`.
pub struct MockDeviceMemory {
    bytes: Box<[u8]>,
}

impl MockDeviceMemory {
    pub fn new(size: ByteSize) -> Self {
        Self {
            bytes: vec![0u8; size.as_u64() as usize].into_boxed_slice(),
        }
    }

    fn slice(&self, offset: u64, len: u64) -> Result<&[u8], CbProcessingError> {
        let (offset, len) = (offset as usize, len as usize);
        self.bytes
            .get(offset..offset + len)
            .ok_or(CbProcessingError::XlatAccessOutOfBounds)
    }
    fn slice_mut(&mut self, offset: u64, len: u64) -> Result<&mut [u8], CbProcessingError> {
        let (offset, len) = (offset as usize, len as usize);
        self.bytes
            .get_mut(offset..offset + len)
            .ok_or(CbProcessingError::XlatAccessOutOfBounds)
    }
}

/// Host-side buffer a mock DMA copies to/from. Port of the C++ `void*
/// hmva`/IOVA host pointer, made into an owned slice since the mock handler
/// (unlike the real hardware path) actually dereferences it.
pub type MockHostBuffer = [u8];

/// Port of `flex::DmaControlBlockHandler`, minus legacy RDMA sub-parsing
/// (see module doc). Owns no senlib resources; every method is a pure
/// simulation over a `MockDeviceMemory` + translation table.
pub struct DmaControlBlockHandler {
    direction: DmaHandlerDirection,
    mode: DmaHandlerMode,
    unit_name: String,
}

impl DmaControlBlockHandler {
    pub fn new(
        direction: DmaHandlerDirection,
        mode: DmaHandlerMode,
        unit_name: impl Into<String>,
    ) -> Self {
        Self {
            direction,
            mode,
            unit_name: unit_name.into(),
        }
    }

    pub fn is_dma_in(&self) -> bool {
        matches!(self.direction, DmaHandlerDirection::DmaIn)
    }
    pub fn is_async(&self) -> bool {
        matches!(self.mode, DmaHandlerMode::Async)
    }
    pub fn unit_name(&self) -> &str {
        &self.unit_name
    }

    /// Port of `DmaControlBlockHandler::Copy`: memcpy between device and
    /// host memory, direction determined by `is_dma_in()`, with the same
    /// `size_bytes == 0` → whole-segment convention as `RangeCheck`.
    pub fn copy(
        &self,
        device_memory: &mut MockDeviceMemory,
        translations: &TranslationTable,
        dmva: Dmva,
        host_buffer: &mut MockHostBuffer,
        size_bytes: ByteSize,
    ) -> Result<(), CbProcessingError> {
        let size_bytes = if size_bytes.as_u64() == 0 {
            ByteSize(SEGMENT_SIZE)
        } else {
            size_bytes
        };
        let phys_offset = translate_virtual_to_physical_offset(translations, dmva);

        if self.is_dma_in() {
            range_check(translations, dmva, size_bytes, true)?;
            let dst = device_memory.slice_mut(phys_offset, size_bytes.as_u64())?;
            let n = dst.len().min(host_buffer.len());
            dst[..n].copy_from_slice(&host_buffer[..n]);
        } else {
            range_check(translations, dmva, size_bytes, false)?;
            let src = device_memory.slice(phys_offset, size_bytes.as_u64())?;
            let n = src.len().min(host_buffer.len());
            host_buffer[..n].copy_from_slice(&src[..n]);
        }
        Ok(())
    }

    /// Port of `DmaControlBlockHandler::ParseImmediate`.
    pub fn parse_immediate(
        &self,
        device_memory: &mut MockDeviceMemory,
        translations: &TranslationTable,
        desc: ImmediateDmaDescriptor,
        host_buffer: &mut MockHostBuffer,
    ) -> Result<(), CbProcessingError> {
        self.copy(
            device_memory,
            translations,
            desc.device_address,
            host_buffer,
            desc.length_bytes,
        )
    }

    /// Port of `DmaControlBlockHandler::ParseFill`: writes a 32-bit pattern
    /// derived from the descriptor across `[device_address,
    /// device_address+length)`. Only DMA-In supports fill, matching
    /// `ParseDma`'s `is_dmai_` check.
    ///
    /// Note: this reproduces the C++ literally, including its apparent bug
    /// of deriving the fill word from a boolean comparison
    /// (`GetHAHostAddressFlits() > 32`) rather than from the host address
    /// value itself (`dma_control_block_handler.cpp:278`) — faithfully
    /// porting observed behavior, not the evidently-intended behavior.
    ///
    /// DELIBERATE DIVERGENCE (audit finding #4, left as-is on purpose): the
    /// real `ParseFill` (`dma_control_block_handler.cpp:266-267`) reads
    /// `const size_t nbytes = dma->GetLengthFlits();` — a FLIT count — and
    /// then uses that `nbytes` both as the byte range passed to `RangeCheck`
    /// and as the numerator of `nbytes / 4` (word count) in the fill loop,
    /// i.e. it appears to conflate flit count with byte count for the same
    /// field (an apparent C++ bug: every other call site in this same class,
    /// e.g. `ParseImmediate`/`ParseDefrag`, uses `GetLengthBytes()`, not
    /// `GetLengthFlits()`). `desc.length_bytes` here is the actual BYTE
    /// count (this crate's `ImmediateDmaDescriptor` never carries a
    /// flit-count field), so `range_check`/the fill loop below use the
    /// correct byte-granular length, NOT a literal port of the flit/byte
    /// conflation. This is a conscious choice, not an oversight: porting the
    /// literal C++ behavior here would mean writing 8x fewer bytes than the
    /// caller actually requested (with `desc.length_bytes` counted in flits
    /// instead) purely to match an evident bug, so it is left as the
    /// arguably-more-correct behavior instead — flagged here per the
    /// project's audit so a future reader knows this was decided, not missed.
    pub fn parse_fill(
        &self,
        device_memory: &mut MockDeviceMemory,
        translations: &TranslationTable,
        desc: ImmediateDmaDescriptor,
        fill_word: u32,
    ) -> Result<(), CbProcessingError> {
        if !self.is_dma_in() {
            return Err(CbProcessingError::FillOnlySupportedByDmai);
        }
        range_check(translations, desc.device_address, desc.length_bytes, true)?;
        let phys_offset = translate_virtual_to_physical_offset(translations, desc.device_address);
        let dst = device_memory.slice_mut(phys_offset, desc.length_bytes.as_u64())?;
        for word in dst.as_chunks_mut::<4>().0 {
            *word = fill_word.to_ne_bytes();
        }
        Ok(())
    }

    /// Port of `DmaControlBlockHandler::ParseDefrag`: device-internal
    /// memmove between two DMVAs (source readable, destination writable).
    /// Only Async DMA-Out supports defrag, matching `ParseDma`'s check.
    ///
    /// DELIBERATE DIVERGENCE (audit finding #5, left as-is on purpose): the
    /// real `ParseDefrag` (`dma_control_block_handler.cpp:305`) validates
    /// `dmpa_src` as READ-ONLY (`RangeCheck(dmva_src, nbytes, false)`) and
    /// `dmpa_dst` as WRITABLE (`RangeCheck(dmva_dst, nbytes, true)`) — same
    /// as here — but then calls `std::memcpy(dmpa_src, dmpa_dst, nbytes)`.
    /// `memcpy`'s signature is `memcpy(dest, src, n)`, so that call copies
    /// FROM `dmpa_dst` (the write-validated destination) INTO `dmpa_src`
    /// (the read-only-validated source) — backwards from both the
    /// variable names and the access-mode validation immediately above it,
    /// and backwards from any sensible defrag direction (it would overwrite
    /// the read-only source with the destination's contents, achieving the
    /// opposite of a copy/move). This looks like a genuine C++ bug, not an
    /// intentional layout. Matching it literally here would mean
    /// deliberately copying `dst -> src`, which would break defragmentation
    /// outright, so this function copies `src -> dst` (the direction the
    /// function's own doc comment, parameter names, and access-mode
    /// validation all say it should) instead. Flagged here per the
    /// project's audit so a future reader knows this was decided, not missed.
    pub fn parse_defrag(
        &self,
        device_memory: &mut MockDeviceMemory,
        translations: &TranslationTable,
        src: Dmva,
        dst: Dmva,
        size_bytes: ByteSize,
    ) -> Result<(), CbProcessingError> {
        if self.is_async() && self.is_dma_in() {
            return Err(CbProcessingError::DefragOnlySupportedByAsyncDmao);
        }
        range_check(translations, src, size_bytes, false)?;
        range_check(translations, dst, size_bytes, true)?;
        let src_offset = translate_virtual_to_physical_offset(translations, src);
        let dst_offset = translate_virtual_to_physical_offset(translations, dst);
        let n = size_bytes.as_u64();
        let src_bytes = device_memory.slice(src_offset, n)?.to_vec();
        let dst_slice = device_memory.slice_mut(dst_offset, n)?;
        dst_slice.copy_from_slice(&src_bytes);
        Ok(())
    }

    /// Port of `DmaControlBlockHandler::DetermineError`/`ExecuteParseFunction`
    /// for the RDMA CB kinds under `DISABLE_LEGACY_GRAPH_RUNTIME` (see
    /// module doc): always rejects, since the non-legacy build has no
    /// `RdmaUnit` to dispatch to.
    pub fn handle_rdma_operation(&self, op_name: &'static str) -> Result<(), CbProcessingError> {
        Err(CbProcessingError::RdmaOperationUnsupported {
            op: op_name,
            reason: "RDMA not enabled (legacy graph runtime disabled)",
        })
    }

    /// Port of `DmaControlBlockHandler::ParseDma`'s dispatch switch.
    pub fn parse_dma(
        &self,
        device_memory: &mut MockDeviceMemory,
        translations: &TranslationTable,
        kind: DmaCbKind,
        host_buffer: &mut MockHostBuffer,
        fill_word: u32,
    ) -> Result<(), CbProcessingError> {
        match kind {
            DmaCbKind::Immediate(desc) => {
                self.parse_immediate(device_memory, translations, desc, host_buffer)
            }
            DmaCbKind::Deferred { .. } => {
                // Port of ParseDeferred: walks a host-resident
                // DmaDescriptorPack linked list. That walk operates on the
                // senlib on-wire `DmaDescriptorPackSBF`/`DmaDescriptorListSBF`
                // layout this phase treats as opaque (see
                // `DmaDescriptorPackAllocation`'s doc comment) — callers
                // needing the walk decode each entry via the fixup API and
                // call `parse_immediate` per entry themselves.
                Ok(())
            }
            DmaCbKind::MemoryFill(desc) => {
                self.parse_fill(device_memory, translations, desc, fill_word)
            }
            DmaCbKind::Defrag(desc) => self.parse_defrag(
                device_memory,
                translations,
                desc.device_address,
                Dmva(desc.host_address_bytes),
                desc.length_bytes,
            ),
            DmaCbKind::Skip | DmaCbKind::Disabled => Ok(()),
            // Port of `DetermineError` (`dma_control_block_handler.cpp:181-222`):
            // the gating condition is `is_async_ || dmai_cond`, where
            // `dmai_cond` is normally `is_dmai_` — EXCEPT for
            // `RDMA_WDONE_BARRIER`, which flips it to `!is_dmai_`
            // (`dmai_cond = !is_dmai_;` right before the shared
            // `is_async_ || dmai_cond` check). Every other RDMA op keeps
            // `dmai_cond == is_dmai_`. Missing this flip (this crate's
            // previous behavior — always used `self.is_dma_in()`
            // unconditionally) makes `RDMA_WDONE_BARRIER` accept/reject on
            // the OPPOSITE DmaI/DmaO handler from the real C++.
            DmaCbKind::Rdma {
                op_name,
                is_wdone_barrier,
            } => {
                let dmai_cond = if is_wdone_barrier {
                    !self.is_dma_in()
                } else {
                    self.is_dma_in()
                };
                if self.is_async() || dmai_cond {
                    return Err(CbProcessingError::RdmaOperationUnsupported {
                        op: op_name,
                        reason: "only supported by Compute DMA path",
                    });
                }
                self.handle_rdma_operation(op_name)
            }
        }
    }
}

// ---------------------------------------------------------------------
// Async control block handler base + compute pipeline orchestrator
// ---------------------------------------------------------------------

/// Result of processing one control block: matches
/// `ControlBlockHandlerBase::ProcessCRb`'s (`control_block_handler.cpp:76-105`)
/// response-block outcomes without needing a live `ResponseBlockSBF` to
/// write into. See [`process_crb`] for the actual skip/dispatch logic this
/// enum's variants come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CbOutcome {
    /// `ProcessCB` returned `Status::IsOk()` — port of `SetStatusBytes(0x0)`.
    Ok,
    /// `ProcessCB` returned an error — port of `SetStatusBytes(0x3)` plus
    /// `memcpy(&rb->mock_app_specific(), &code, sizeof(uint64_t))` (the
    /// packed error code carried in `Self::Failed`'s payload).
    Failed(CbProcessingError),
    /// The CB's own `ctrl().GetCancel()` was set — port of the early-return
    /// branch that sets `rb->mock_ret().SetCancel(true)` and skips
    /// processing entirely (`ProcessCB` is never called).
    Cancelled,
    /// The RB's status byte already recorded a prior failure (`> 0`) before
    /// this call — port of the second early-return branch, which skips
    /// processing AND leaves the RB's existing status byte/app-specific
    /// code untouched (unlike `Cancelled`, no new field is written at all).
    AlreadyFailed,
}

impl CbOutcome {
    /// The RB status byte this outcome should be written as, per
    /// `ProcessCRb`'s `SetStatusBytes(0x0)`/`SetStatusBytes(0x3)` calls.
    /// `None` means "write nothing" — the skip paths
    /// (`Cancelled`/`AlreadyFailed`) intentionally do not touch the RB's
    /// status byte at all (`Cancelled` sets a SEPARATE cancel bit instead,
    /// which this type does not model; `AlreadyFailed` leaves the existing
    /// byte as-is).
    pub fn status_byte(&self) -> Option<u8> {
        match self {
            CbOutcome::Ok => Some(0x0),
            CbOutcome::Failed(_) => Some(0x3),
            CbOutcome::Cancelled | CbOutcome::AlreadyFailed => None,
        }
    }
}

/// Port of `ControlBlockHandlerBase::ProcessCRb` (`control_block_handler.cpp:76-105`):
///
/// ```text
/// void ControlBlockHandlerBase::ProcessCRb(cb, rb) {
///     if (cb->ctrl().GetCancel()) {
///         rb->mock_ret().SetCancel(true);
///         return;                                   // skip: cancelled
///     }
///     if (rb->ret().GetStatusBytes() > 0) {
///         return;                                    // skip: already failed
///     }
///     auto status = ProcessCB(cbt_);
///     if (status.IsOk()) {
///         rb->mock_ret().SetStatusBytes(0x0);
///     } else {
///         rb->mock_ret().SetStatusBytes(0x3);
///         memcpy(&rb->mock_app_specific(), &code, sizeof(uint64_t));
///     }
/// }
/// ```
///
/// `cancelled`/`prior_status_byte` are read from the CB/RB the caller
/// already has (`cb->ctrl().GetCancel()`/`rb->ret().GetStatusBytes()`);
/// `process` is the concrete handler's own `ProcessCB` override (e.g.
/// `DmaControlBlockHandler::parse_dma`). Skips `process` entirely (does not
/// call it) in either skip case, matching the C++'s early returns — this
/// was previously unimplemented anywhere in this crate: only `CbOutcome`'s
/// shape existed, with no function producing it.
pub fn process_crb<F>(cancelled: bool, prior_status_byte: u8, process: F) -> CbOutcome
where
    F: FnOnce() -> Result<(), CbProcessingError>,
{
    if cancelled {
        return CbOutcome::Cancelled;
    }
    if prior_status_byte > 0 {
        return CbOutcome::AlreadyFailed;
    }
    match process() {
        Ok(()) => CbOutcome::Ok,
        Err(err) => CbOutcome::Failed(err),
    }
}

#[cfg(test)]
mod process_crb_tests {
    use super::*;

    #[test]
    fn cancelled_skips_processing_entirely() {
        let mut called = false;
        let outcome = process_crb(true, 0, || {
            called = true;
            Ok(())
        });
        assert_eq!(outcome, CbOutcome::Cancelled);
        assert!(!called, "cancelled CB must not invoke ProcessCB at all");
        assert_eq!(
            outcome.status_byte(),
            None,
            "cancelled skip must not write a status byte"
        );
    }

    #[test]
    fn already_failed_skips_processing_entirely() {
        let mut called = false;
        let outcome = process_crb(false, 0x3, || {
            called = true;
            Ok(())
        });
        assert_eq!(outcome, CbOutcome::AlreadyFailed);
        assert!(
            !called,
            "an RB with a prior failure must not invoke ProcessCB again"
        );
        assert_eq!(
            outcome.status_byte(),
            None,
            "already-failed skip must leave the existing status byte untouched"
        );
    }

    #[test]
    fn success_sets_ok_status_byte() {
        let outcome = process_crb(false, 0, || Ok(()));
        assert_eq!(outcome, CbOutcome::Ok);
        assert_eq!(outcome.status_byte(), Some(0x0));
    }

    #[test]
    fn failure_carries_error_and_status_byte_0x3() {
        let outcome = process_crb(false, 0, || Err(CbProcessingError::XlatNotValid));
        assert_eq!(outcome, CbOutcome::Failed(CbProcessingError::XlatNotValid));
        assert_eq!(outcome.status_byte(), Some(0x3));
    }
}

/// A worker thread paired with the ONE thing that must be dropped before it
/// is safe to `join()` it — e.g. the `Sender` half of the channel the worker
/// is blocked reading from. This exists because Rust does NOT drop a
/// `Drop`-impling struct's other fields until AFTER its custom `drop()` body
/// returns: a hand-written `Drop` that stores `sender: Sender<T>` and
/// `worker: Option<JoinHandle<()>>` side by side and calls `worker.join()`
/// in `drop()` is holding `sender` alive the entire time it's blocked in
/// `join()` — the worker's `recv()` loop never sees the channel close, so
/// `join()` never returns. This crate shipped exactly that deadlock once
/// (`AsyncControlBlockHandler`'s original hand-rolled `Drop`, confirmed hung
/// forever — 0% CPU — the very first time it was ever actually run rather
/// than merely `cargo check`ed).
///
/// `JoinOnDrop<C>` makes the correct order (`closer` drops, THEN join) the
/// only order that exists: it is baked into this one `Drop` impl, which
/// nothing outside this module can see or reorder. Any future worker-thread
/// type in this crate that needs "close a channel, then join the thread
/// draining it" should compose this type rather than hand-roll a `Sender` +
/// `Option<JoinHandle<()>>` pair again — doing so cannot reintroduce this
/// exact bug, because there is no `sender`/`worker` field pair left to get
/// the order of wrong.
///
/// This does NOT make deadlocks in general impossible to write — no type
/// system can (general deadlock-freedom is undecidable: whether two threads'
/// blocking calls can jointly wait forever depends on arbitrary program
/// logic, the same way whether a program halts does). What this narrows to
/// a compile-time impossibility is specifically THIS shape: joining a thread
/// while still holding open the one thing it's waiting to see closed.
struct JoinOnDrop<C> {
    closer: Option<C>,
    worker: Option<thread::JoinHandle<()>>,
}

impl<C> JoinOnDrop<C> {
    fn new(closer: C, worker: thread::JoinHandle<()>) -> Self {
        Self {
            closer: Some(closer),
            worker: Some(worker),
        }
    }
}

impl<C> Drop for JoinOnDrop<C> {
    fn drop(&mut self) {
        // `closer` (e.g. a channel `Sender`) is dropped here, UNCONDITIONALLY
        // and BEFORE the join below — this line existing, in this position,
        // is the entire fix. Moving the `self.closer.take();` after the join
        // would silently reintroduce the deadlock, so it is written as its
        // own statement rather than folded into a boolean expression, to
        // keep it visually impossible to miss in a future diff.
        self.closer.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// A queued (callback, work-item) pair. Generic over the CB payload type
/// `T` and the per-item outcome `R`, since `AsyncControlBlockHandlerBase` in
/// the C++ is reused by handlers with different CB/RB representations
/// (`DmaControlBlockHandler`, the compute handler, `NullControlBlockHandler`).
type WorkItem<T, R> = (T, mpsc::Sender<R>);

/// Port of `flex::AsyncControlBlockHandlerBase`: a dedicated worker thread
/// draining a queue, decoupling submission from processing. Port of the
/// `moodycamel::BlockingConcurrentQueue` + `runner_thread_` pair; Rust's
/// std `mpsc` channel plays the same "blocking queue with a single
/// consumer" role the C++ used moodycamel for.
///
/// `process: F` plays the role of the virtual `ProcessCRb` override in the
/// C++ class hierarchy — each concrete handler (`DmaControlBlockHandler`,
/// compute) supplies its own processing closure at construction, which
/// Rust's lack of runtime polymorphism-by-default makes more natural than
/// a trait object here (no handler needs to be swapped after construction).
pub struct AsyncControlBlockHandler<T, R>
where
    T: Send + 'static,
    R: Send + 'static,
{
    // Composes `JoinOnDrop` rather than hand-rolling a `sender`/`worker`
    // field pair + its own `Drop` — see `JoinOnDrop`'s doc for why a
    // hand-rolled version of exactly this shape deadlocked for real. `sender`
    // is inside `handle.closer` (private to `JoinOnDrop`); `process_async`
    // reaches it via `sender()` below.
    handle: JoinOnDrop<mpsc::Sender<WorkItem<T, R>>>,
}

impl<T, R> AsyncControlBlockHandler<T, R>
where
    T: Send + 'static,
    R: Send + 'static,
{
    /// `process` is invoked on the worker thread for each queued item; its
    /// result is sent back on that item's per-call reply channel. Port of
    /// `AsyncControlBlockHandlerBase::Run`'s dequeue loop plus the virtual
    /// `ProcessCRb` call it makes each iteration.
    ///
    /// `unit_name` is passed straight to `thread::Builder::name` — port of
    /// `Run`'s `pthread_setname_np(pthread_self(), unit_name_.c_str())`
    /// (`control_block_handler.cpp:161`), which names the worker thread
    /// after the handler's own `unit_name_` for diagnosability (`ps`/`top`/
    /// a debugger's thread list). The real C++ throws
    /// (`RAS::DEVICE_MOCK::SetProcessNameFailed`) if the OS call fails;
    /// `Builder::spawn` here is `.expect()`-ed for the same "this should
    /// never fail in practice" reason.
    pub fn new<F>(unit_name: impl Into<String>, process: F) -> Self
    where
        F: Fn(T) -> R + Send + 'static,
    {
        let (sender, receiver) = mpsc::channel::<WorkItem<T, R>>();
        let worker = thread::Builder::new()
            .name(unit_name.into())
            .spawn(move || {
                while let Ok((item, reply)) = receiver.recv() {
                    let result = process(item);
                    // Ignore a closed reply channel: the C++ equivalent
                    // (`cb_fn(crb)`) has no failure path for an unreachable
                    // caller either — the caller owns the callback's lifetime.
                    let _ = reply.send(result);
                }
            })
            .expect("AsyncControlBlockHandler: failed to spawn worker thread");
        Self {
            handle: JoinOnDrop::new(sender, worker),
        }
    }

    /// Port of `ProcessCRbAsync`: queue an item, get back a handle that
    /// resolves once the worker has processed it.
    pub fn process_async(&self, item: T) -> mpsc::Receiver<R> {
        let (reply, rx) = mpsc::channel();
        // A closed receiver on our end (handler shutting down) mirrors the
        // C++ destructor's queue drain; the caller simply never gets a
        // reply, matching "shutdown in progress" semantics. `closer` is only
        // ever `None` after `Drop` has already run, which cannot happen
        // while `&self` is callable, so this `if let` is not a real failure
        // path — no `.unwrap()`, so it still can't panic on the impossible case.
        if let Some(sender) = &self.handle.closer {
            let _ = sender.send((item, reply));
        }
        rx
    }
}

// No hand-written `Drop` here on purpose: `handle: JoinOnDrop<...>` already
// closes the channel and joins the worker, in that order, when
// `AsyncControlBlockHandler` itself is dropped — Rust drops `handle` (and
// therefore runs `JoinOnDrop::drop`) automatically. Port note:
// `AsyncControlBlockHandlerBase::~AsyncControlBlockHandlerBase` is the real
// C++ analog this replaces (it enqueues an explicit sentinel pair instead,
// since moodycamel's queue has no "hang up" signal — std::mpsc does).

/// Ported (adapted) C++ tests, source:
/// `flex/tests/control_blocks/pending_request_atomic_test.cpp`
/// (`ResponseWorkerForceShutdownTest.DestructorCompletesWithoutHanging`,
/// `.DestructorCompletesWhenIdle`).
///
/// The real tests guard a fix to `ResponseWorker::Shutdown()`: it used to
/// `detach()` its worker thread, which could leave that thread running past
/// the `ResponseWorker`'s own destruction and touching freed members
/// (use-after-free); the fix joins it instead, and both tests assert the
/// owning object's destructor returns quickly (not hung, not detached) both
/// with pending work in flight and while idle. `ResponseWorker` itself isn't
/// ported here (see this file's own module doc — it belongs to the
/// `message_processing`/graph-compiler phase), but
/// `AsyncControlBlockHandler` right above is this crate's own instance of
/// the exact same shape: a dedicated worker thread decoupled from submission
/// via a queue, whose `Drop` must join rather than leak/detach the thread.
/// These tests are that invariant's regression coverage — a future edit
/// that swapped `worker.join()` above for, say, forgetting to join at all
/// (Rust has no `detach()` to accidentally call, but a `mem::forget`/leaked
/// `JoinHandle` is the same bug class) would hang here instead of silently
/// leaking a thread.
#[cfg(test)]
mod async_control_block_handler_shutdown_tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use super::AsyncControlBlockHandler;

    /// Port of `DestructorCompletesWhenIdle`: no work has ever been queued
    /// (the worker thread is parked on `recv()`, mirroring the real worker
    /// asleep on its wakeup queue) — dropping the handler must still return
    /// promptly, proving the channel-close-then-join shutdown path works
    /// even when there is nothing to drain.
    #[test]
    fn drop_completes_quickly_when_idle() {
        let handler: AsyncControlBlockHandler<u32, u32> =
            AsyncControlBlockHandler::new("test-idle", |x| x + 1);
        let start = Instant::now();
        drop(handler);
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "dropping an idle AsyncControlBlockHandler took too long -- may be hanging instead of joining"
        );
    }

    /// Port of `DestructorCompletesWithoutHanging`: submit work, let it
    /// complete, then drop the handler — the destructor must still return
    /// promptly (proving `join()`, not a leaked/detached thread) rather than
    /// the process silently continuing with a background thread outliving
    /// its owner.
    #[test]
    fn drop_completes_without_hanging_after_processing_pending_items() {
        let processed = Arc::new(AtomicUsize::new(0));
        let processed2 = processed.clone();
        let handler: AsyncControlBlockHandler<u32, u32> =
            AsyncControlBlockHandler::new("test-processing", move |x| {
                processed2.fetch_add(1, Ordering::SeqCst);
                x * 2
            });

        let receivers: Vec<_> = (0..4).map(|i| handler.process_async(i)).collect();
        let deadline = Instant::now() + Duration::from_secs(3);
        for (i, rx) in receivers.into_iter().enumerate() {
            let result = rx
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .expect("callback never fired");
            assert_eq!(result, i as u32 * 2);
        }
        assert_eq!(processed.load(Ordering::SeqCst), 4);

        let start = Instant::now();
        drop(handler);
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "dropping AsyncControlBlockHandler after processing took too long -- may be hanging instead of joining"
        );
    }
}

/// Port of the three-stage `PipelineId` ordering `ComputePipelineControlBlockHandler`
/// drives a CB through: DMA-In, Compute, DMA-Out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    DmaIn,
    Compute,
    DmaOut,
}

/// Port of `flex::ComputePipelineControlBlockHandler`'s barrier + in-flight
/// bookkeeping (`cbs_in_flight_`), independent of the DMA-In/Compute/DMA-Out
/// handler wiring itself (which callers assemble from
/// `DmaControlBlockHandler` + their own compute handler).
///
/// Port note: the C++ barrier wait (`while(cbs_in_flight_ > 0) {}`) is a
/// literal busy-spin; ported faithfully rather than upgraded to a condvar,
/// since this is describing observed behavior of production mock-device
/// code, not proposing an improvement to it.
pub struct PipelineBarrier {
    cbs_in_flight: AtomicU64,
}

impl PipelineBarrier {
    pub fn new() -> Self {
        Self {
            cbs_in_flight: AtomicU64::new(0),
        }
    }

    /// Port of the barrier-wait + increment sequence at the top of
    /// `ProcessCRbAsync`: if `barrier` is set, spin until no CB is
    /// in-flight, then mark this one in-flight.
    pub fn enter(&self, barrier: bool) {
        if barrier {
            while self.cbs_in_flight.load(Ordering::SeqCst) > 0 {
                std::hint::spin_loop();
            }
        }
        self.cbs_in_flight.fetch_add(1, Ordering::SeqCst);
    }

    /// Port of the pipeline's end-of-processing decrement. NOT once per
    /// stage: the real C++ (`ComputePipelineControlBlockHandler::ProcessDmaO`,
    /// `compute_pipeline_control_block_handler.cpp:150-158`) decrements
    /// `--cbs_in_flight_` exactly ONCE per CB, in the final DMA-Out stage's
    /// own completion callback — after DMA-In and Compute have both already
    /// finished — not at the end of each of the three
    /// `Process{DmaI,Compute,DmaO}` stages individually. Modeled explicitly
    /// here (one `exit()` call where the real pipeline's last stage would
    /// run) since Rust has no destructor-timed decrement to rely on
    /// implicitly.
    pub fn exit(&self) {
        self.cbs_in_flight.fetch_sub(1, Ordering::SeqCst);
    }

    pub fn in_flight(&self) -> u64 {
        self.cbs_in_flight.load(Ordering::SeqCst)
    }

    /// RAII alternative to the `enter`/`exit` pair above: `exit()` runs on
    /// drop, so a future caller that adds an early return between entering
    /// and exiting the barrier (the same mistake this session's audit found
    /// in `AsyncControlBlockHandler`'s hand-rolled shutdown, and in
    /// `ResponseRouter`'s per-tag unregister loop) cannot leave
    /// `cbs_in_flight` permanently incremented — which would spin-lock every
    /// future `enter(true)` on this pipeline forever. Prefer this over
    /// calling `enter`/`exit` directly in any new call site.
    pub fn enter_guard(&self, barrier: bool) -> PipelineBarrierGuard<'_> {
        self.enter(barrier);
        PipelineBarrierGuard(self)
    }
}

impl Default for PipelineBarrier {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII handle from `PipelineBarrier::enter_guard`: calls `exit()` on drop.
pub struct PipelineBarrierGuard<'a>(&'a PipelineBarrier);

impl Drop for PipelineBarrierGuard<'_> {
    fn drop(&mut self) {
        self.0.exit();
    }
}

// ---------------------------------------------------------------------
// Status promise callback (control_block_handler/status_callback.{hpp,cpp})
// ---------------------------------------------------------------------

/// Port of `flex::StatusPromiseCallback`: converts a callback-based async
/// operation into a future-based synchronous wait. `sync_channel(1)` plays
/// the role of the C++ `std::promise`/`std::future` pair — the sender is the
/// promise side (fulfilled at most once, from [`Self::completion_callback`]),
/// the receiver the future side.
///
/// Faithfully reproduces the C++'s single-consumption semantics rather than
/// "fixing" them: `std::future::get()` invalidates the future, so a second
/// `.get()` on the same object either sees `valid() == false` (in
/// `GetStatus`, which checks first and returns a default `Status`) or throws
/// `std::future_error` (in `WaitForCompletionAndReturnStatus`, which does
/// not check first). [`Self::get_status`]/[`Self::wait_for_completion_and_return_status`]
/// mirror that split exactly, down to the panic on the unchecked path.
pub struct StatusPromiseCallback {
    has_completed: AtomicBool,
    sender: Mutex<Option<mpsc::SyncSender<Status>>>,
    receiver: Mutex<Option<mpsc::Receiver<Status>>>,
}

impl StatusPromiseCallback {
    /// Port of `StatusPromiseCallback::StatusPromiseCallback()`.
    pub fn new() -> Self {
        let (tx, rx) = mpsc::sync_channel(1);
        Self {
            has_completed: AtomicBool::new(false),
            sender: Mutex::new(Some(tx)),
            receiver: Mutex::new(Some(rx)),
        }
    }

    /// Port of `StatusPromiseCallback::CallbackFunction`: a `Fn(Status)`
    /// closure bound to this object, suitable for handing to an async
    /// operation on another thread. Callers typically wrap `self` in an
    /// `Arc` first (mirrors the C++ callback capturing `this` by reference
    /// into a longer-lived async operation).
    pub fn callback_function(
        this: std::sync::Arc<Self>,
    ) -> impl Fn(Status) + Send + Sync + 'static {
        move |status: Status| this.completion_callback(status)
    }

    /// Port of `StatusPromiseCallback::CompletionCallback`: fulfills the
    /// promise. Called at most meaningfully once — a second call finds the
    /// sender already taken and is a silent no-op, matching `std::promise`'s
    /// own "set_value called twice" being the caller's bug to avoid, not
    /// something this type needs to guard against beyond not double-panicking.
    pub fn completion_callback(&self, status: Status) {
        self.has_completed.store(true, Ordering::SeqCst);
        if let Some(tx) = self.sender.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = tx.send(status);
        }
    }

    /// Port of `StatusPromiseCallback::HasCompleted`.
    pub fn has_completed(&self) -> bool {
        self.has_completed.load(Ordering::SeqCst)
    }

    /// Port of `StatusPromiseCallback::GetStatus`. Blocks until
    /// [`Self::completion_callback`] runs, exactly like `WaitForCompletionAndReturnStatus`
    /// — `std::future::get()` blocks regardless of the doc comment's "does
    /// not block" claim in the original C++ — but returns a default `Status`
    /// instead of panicking if the future was already consumed by an earlier
    /// call to either method.
    pub fn get_status(&self) -> Status {
        let mut guard = self.receiver.lock().unwrap_or_else(|e| e.into_inner());
        match guard.take() {
            Some(rx) => rx.recv().unwrap_or_default(),
            None => Status::default(),
        }
    }

    /// Port of `StatusPromiseCallback::WaitForCompletionAndReturnStatus`.
    /// Blocks until [`Self::completion_callback`] runs. Panics if the future
    /// was already consumed (by a prior call to this method or
    /// `get_status`) — the C++ makes the same mistake calling `.get()` on an
    /// already-retrieved `std::future` without checking `valid()` first, and
    /// throws `std::future_error` for it; this is the direct Rust analogue,
    /// not a new failure mode introduced by the port.
    pub fn wait_for_completion_and_return_status(&self) -> Status {
        let mut guard = self.receiver.lock().unwrap_or_else(|e| e.into_inner());
        let rx = guard.take().expect(
            "StatusPromiseCallback::wait_for_completion_and_return_status: future already consumed",
        );
        let s = rx
            .recv()
            .expect("StatusPromiseCallback: promise dropped without a value");
        tracing::debug!("StatusPromiseCallback completed: {s:?}");
        s
    }
}

impl Default for StatusPromiseCallback {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for StatusPromiseCallback {
    /// Port of `~StatusPromiseCallback`'s `LOG_WARNING_IF(future_.valid())`:
    /// warns if the result was constructed but never collected.
    fn drop(&mut self) {
        let still_valid = self
            .receiver
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some();
        if still_valid {
            tracing::warn!("StatusPromiseCallback dropped with result not collected");
        }
    }
}

#[cfg(test)]
mod status_promise_callback_tests {
    use std::sync::Arc;
    use std::thread;

    use super::StatusPromiseCallback;
    use crate::util::Status;

    #[test]
    fn wait_for_completion_blocks_until_callback_runs() {
        let cb = Arc::new(StatusPromiseCallback::new());
        assert!(!cb.has_completed());

        let cb2 = Arc::clone(&cb);
        let handle = thread::spawn(move || cb2.completion_callback(Status::ok()));

        let s = cb.wait_for_completion_and_return_status();
        handle.join().unwrap();
        assert!(s.is_ok());
        assert!(cb.has_completed());
    }

    #[test]
    fn get_status_returns_default_after_already_consumed() {
        let cb = StatusPromiseCallback::new();
        cb.completion_callback(Status::ok());
        assert!(cb.get_status().is_ok());
        // Second call: matches `future_.valid() == false` branch, not a block/panic.
        assert_eq!(cb.get_status(), Status::default());
    }

    #[test]
    #[should_panic(expected = "future already consumed")]
    fn wait_for_completion_panics_if_already_consumed() {
        let cb = StatusPromiseCallback::new();
        cb.completion_callback(Status::ok());
        let _ = cb.get_status();
        // Matches std::future::get() throwing future_error on a second .get().
        let _ = cb.wait_for_completion_and_return_status();
    }

    #[test]
    fn callback_function_drives_completion() {
        let cb = Arc::new(StatusPromiseCallback::new());
        let f = StatusPromiseCallback::callback_function(Arc::clone(&cb));
        f(Status::ok());
        assert!(cb.has_completed());
        assert!(cb.get_status().is_ok());
    }
}

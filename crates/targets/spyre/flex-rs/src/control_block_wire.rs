//! Real senlib control-block / response-block wire-format encoder.
//!
//! This is the piece `senlib_ffi.rs`'s "REAL LINK-TARGET AUDIT" doc comment
//! identified as entirely missing: something that turns `DmaParams`/
//! `ComputeParams`/`FillParams` into the actual CTRL/R5/DMI/DMO/CMPT/XLAT
//! byte layout hardware expects, per `SentientSoc::V1::ControlBlockSBF`
//! (`hal/1p0/control_block_sbf.hpp`) and `ResponseBlockSBF`
//! (`hal/1p0/response_block_sbf.hpp`).
//!
//! Byte layout verified 2026-08-15 against the real headers pulled from the
//! on-pod SDK (`nickm-78f5c6fb4b-2fk84`, `/opt/ibm/spyre/senlib/include/`):
//!   - `hal/1p0/control_block_sbf.hpp` — 128-byte CB, 6 fixed-offset sections:
//!     CTRL@0 (8B), R5@8 (8B), DMI@16 (16B), DMO@32 (16B), CMPT@48 (16B),
//!     XLAT@64 (64B = 8 x 8-byte `TranslationSBF`).
//!   - `hal/1p0/ctrl_section_sbf.hpp` — `CtrlSectionSBF` bitfields.
//!   - `hal/1p0/dma_control_block_section_base_sbf.hpp` +
//!     `dma_control_block_section_sbf.hpp` — DMA section bitfields
//!     (`HdmaImmControlBlockSectionSBF`/`HdmaFillControlBlockSectionSBF`
//!     layouts used here).
//!   - `hal/1p0/compute_control_block_section_sbf.hpp` — compute section.
//!   - `hal/1p0/translation_control_block_section_sbf.hpp` — XLAT entries.
//!   - `hal/1p0/r5_control_block_section_sbf.hpp` — R5 section (this crate
//!     never issues R5/HDMA work, so only `R5Section::none()`, the all-zero
//!     QGI encoding, is provided).
//!   - `hal/1p0/return_section_sbf.hpp` — RB `ReturnSectionSBF` (decode side).
//!   - `hal/1p0/virtual_address_sbf.hpp` — `VirtualAddressSBF`: 3-bit segment
//!     + 27-bit offset-in-flits, 1 flit = 128 bytes (`FLIT_SIZE_BITS = 7`).
//!
//! Every bitfield's `(hi, lo)` range below is copied 1:1 from those headers'
//! `ADD_SAFE_BITFIELD_MEMBER_PDF(name, hi, lo)` declarations. For 2-word
//! (16-byte) sections the C++ union spans 128 bits little-endian across two
//! `uint64_t` words (word0 = bits[63:0], word1 = bits[127:64]); this module
//! encodes each section as `[u64; N]` in that same word order and only
//! converts to raw bytes (native little-endian `to_le_bytes`) at the
//! `ControlBlockWire::to_bytes`/`ResponseBlockWire::from_bytes` boundary,
//! matching the hardware's documented little-endian section format.
//!
//! Scope: this crate is a graph-free / pass-through runtime (no compiler,
//! no RDMA/HDMA, no message matching), so only the CB shapes the real C++
//! graph-free path builds are ported: `ControlBlockStream::CreateGraphFreeCompute`,
//! `CreateComputeDataTransfer` (immediate DMA), and `CreateFillDma`
//! (`flex/src/control_blocks/control_block_stream/control_block_stream.hpp`).
//! RDMA/HDMA/R5 section encoding is out of scope (never emitted by this
//! runtime) and is not ported here.

/// A bit range `[HI, LO]` (inclusive, hardware convention — matches the real
/// headers' own `ADD_SAFE_BITFIELD_MEMBER_PDF(name, hi, lo)` argument order)
/// within a 64-bit word. `pack`/`unpack` derive both the shift and the mask
/// from the SAME two constants, so "shifted by the high bit instead of the
/// low bit" — the real bug this crate shipped once in `DmaSection::immediate`'s
/// `cbtype_` field — becomes impossible to write: there is no second place to
/// get `hi`/`lo` wrong independently, because the shift amount is always `LO`
/// and nothing else ever appears in this type.
///
/// `LO` is the ONLY shift this type will ever apply — `pack`/`unpack` do not
/// take a shift parameter, so a future caller cannot accidentally pass `HI`
/// where a shift belongs. Compare a bitfield's `(HI, LO)` here against the
/// real header's `ADD_SAFE_BITFIELD_MEMBER_PDF(name, HI, LO)` and it matches
/// literally, not "matches after mental arithmetic".
struct BitField<const HI: u32, const LO: u32>;

impl<const HI: u32, const LO: u32> BitField<HI, LO> {
    const _ASSERT_RANGE: () = assert!(HI >= LO && HI < 64, "BitField: HI must be >= LO and < 64");
    const WIDTH: u32 = HI - LO + 1;
    const MASK: u64 = if Self::WIDTH == 64 {
        u64::MAX
    } else {
        (1u64 << Self::WIDTH) - 1
    };

    /// Mask `value` to this field's width and shift it into position — the
    /// only shift used anywhere in this file is `LO`, derived here, never a
    /// hand-written literal at the call site.
    const fn pack(value: u64) -> u64 {
        let () = Self::_ASSERT_RANGE;
        (value & Self::MASK) << LO
    }

    /// Extract this field's value back out of a word, already shifted down
    /// and masked to its width.
    const fn unpack(word: u64) -> u64 {
        let () = Self::_ASSERT_RANGE;
        (word >> LO) & Self::MASK
    }
}

/// This specific DMA/fill transfer's own byte count. NEVER an XLAT bound —
/// see `RegionTotalSize`/`AllocationTotalSize`, two DIFFERENT fields that
/// this crate has twice (independently, in opposite directions) conflated
/// with a transfer's own size when they were all bare `u64`: DMA's XLAT
/// length was once wrongly set to this value instead of `RegionTotalSize`,
/// and compute's XLAT length was once wrongly set to the enclosing region's
/// `RegionTotalSize` instead of `AllocationTotalSize`. Three distinct types
/// with no `From`/arithmetic between them means a call site can no longer
/// pass whichever `u64` happens to be lying around.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferSize(pub u64);

/// This DMA/fill transfer's own extent within its XLAT segment — the
/// correct XLAT length for a DMA-immediate or fill control block
/// (`build_dma_cb`/a future `build_fill_cb`), per the REAL functions these
/// wire-format methods port: `ControlBlockStream::CreateComputeDataTransfer`
/// and `CreateFillDma` (`control_block_stream.cpp:3803`/`3870`), NOT the
/// legacy-graph-only `buildDmaControlBlocks`
/// (`pf_runtime_scheduler_utils.cpp:48`) an earlier version of this file
/// mistakenly cited as the ported function.
///
/// Both real functions compute this identically:
/// ```text
/// auto seg_id     = dmvaToSegmentId(dmva);
/// auto seg_offset = SegmentOffset(dmva);       // THIS transfer's own dmva
/// xlat.SetLengthBytes(EncodeLength(seg_offset + size_bytes));  // THIS transfer's own size
/// ```
/// i.e. this transfer's own starting offset within its segment, plus its
/// own byte count — never the owning region's total allocated size (that
/// quantity, `RegionTotalSize`-shaped, belongs to the different
/// `buildDmaControlBlocks` call path this crate does not implement, and
/// was wrongly plumbed into this exact field for one session before this
/// fix). A region loaded via more than one H2D call must NOT report the
/// whole region's size here — each transfer reports only its own
/// `seg_offset + size_bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentExtent(pub u64);

/// The three fields describing one immediate DMA's data movement: where on the
/// device (`dmva`), where on the host (`host_addr`, already an IOVA), and how
/// many bytes. Grouped because they travel together into
/// `DmaSection::immediate` and are meaningless apart — the alternative is an
/// eight-parameter `dma_immediate` whose adjacent same-shaped arguments are
/// easy to transpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DmaTransfer {
    pub dmva: VirtualAddressSbf,
    pub host_addr: HostPhysicalAddress,
    pub size_bytes: TransferSize,
}

/// The XLAT entry a DMA control block installs for its own segment: the
/// region's base physical address plus this transfer's extent within that
/// segment. See [`SegmentExtent`] for which real formula `length` is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XlatWindow {
    pub region_paddr: DevicePhysicalAddress,
    pub length: SegmentExtent,
}

/// One specific allocation's (a tensor operand or a program bundle) own
/// total size — the correct XLAT length for a compute control block's
/// per-tensor/program entries (`build_compute_cb`), per the real C++'s
/// `tensor_sizes.push_back(inp_out_allocs[i]->total_size())` —
/// `CompositeAddress::total_size()`, NOT the owning region's. Distinct from
/// `RegionTotalSize`: many independently-addressed allocations share one
/// region (`allocator.rs`'s sub-chunk carving), so the region can be orders
/// of magnitude larger than any allocation living in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocationTotalSize(pub u64);

/// A hardware/device *physical* byte address, already resolved from a
/// `CompositeAddress`/`DeviceMemoryAllocation` (see
/// `DeviceMemoryAllocation::device_address_bytes`). Distinct from the
/// logical/region-relative addresses in `crate::address` — this is what
/// actually gets written into `TranslationSBF::paddr_`/DMA `device_address_`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevicePhysicalAddress(pub u64);

/// A host-side physical/IOVA byte address (already IOMMU-mapped), written
/// into a DMA section's `host_address_` field.
///
/// ⛔ SEALED, and that is the point. The field is private and there is no
/// constructor from a bare integer, because the defect this type exists to
/// prevent was exactly such a cast: `HostPhysicalAddress(dma.hmva().0 as u64)`
/// in `VfBackend::submit_dma`, which built a perfectly well-formed CB around a
/// HOST VIRTUAL address. The device's IOMMU cannot resolve one, so the card's
/// attempt to master it is not a per-CB rejection but a PCIe-level fault
/// (`RAS::PCI::BusFence`, "Reset, reseat, or replace the card"). A host
/// virtual address and a device-visible one are two different quantities that
/// were both spelled `u64`.
///
/// The only ways to obtain one: [`Self::of_mapping`] (an IOMMU mapping this
/// crate holds — borrowed, because the IOVA is only meaningful while the
/// mapping lives) and [`Self::of_caller_supplied_iova`] (an IOVA the caller
/// already mapped, per `DmaParams`' pre-mapped contract).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostPhysicalAddress(u64);

impl HostPhysicalAddress {
    /// The device-visible address of a live IOMMU mapping.
    pub fn of_mapping(mapping: &crate::iommu::IommuMapping) -> Self {
        Self(mapping.iova() as u64)
    }

    /// An IOVA the caller mapped itself and passed in through
    /// `DmaParams::iova()` — the documented "skip the backend's RAII
    /// shadow-buffer copy" path, where the caller owns the mapping.
    pub fn of_caller_supplied_iova(iova: crate::dma::IovaAddress) -> Self {
        Self(iova.0 as u64)
    }

    /// The address of a buffer that has not been mapped yet — the value a
    /// freshly constructed `DmaBuf` holds before `setup_dma_buf` establishes a
    /// real one. Never written into a control block: every CB-building path
    /// takes the address from a mapping or a caller-supplied IOVA.
    pub const UNMAPPED: Self = Self(0);

    /// The raw field, for the CB encoder and for logging.
    pub fn as_u64(self) -> u64 {
        self.0
    }

    /// Test-only: a literal device address, so encoder tests can pin exact
    /// wire bytes without an IOMMU. Named so it cannot appear in a real path
    /// unnoticed, and `cfg`-gated so it cannot appear there at all.
    #[cfg(any(test, kani))]
    pub const fn from_raw_for_tests(raw: u64) -> Self {
        Self(raw)
    }
}

/// One flit = 128 bytes (`VirtualAddressSBF::FLIT_SIZE_BITS = 7`). Every
/// hardware address/length field in the CB is flit-granular; this newtype
/// keeps byte<->flit conversion in one place instead of scattering `>> 7`/
/// `<< 7` through the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Flits(pub u64);

impl Flits {
    const FLIT_SIZE_BITS: u32 = 7;
    /// A flit is 128 bytes. Confirmed from IBM source, not inferred: `qg.h`'s `QGHeader` is a
    /// `union { uint8_t u8[128]; uint64_t u64[16]; }` — one flit — and
    /// `QG::getCurrentOffsetFlits()` is `getCurrentOffset() / 128`.
    pub const BYTES: u64 = 1 << Self::FLIT_SIZE_BITS;

    /// Convert a byte count that MUST be flit-aligned.
    ///
    /// ⛔⛔⛔ THE SHIFT IS RIGHT AND THE SILENCE WAS THE BUG. `bytes >> 7` is the correct
    /// conversion; what it did not do is notice when `bytes` was not a multiple of 128. An
    /// unaligned input truncates DOWNWARD — so a DMA's device address silently became a different
    /// address, while the XLAT length beside it was computed from the untruncated value. The
    /// transfer then runs past its window, and the card answers with `0xa35e RAS::PCI::BusFence`:
    /// a fault with no host-side statement of which transfer caused it.
    ///
    /// IBM's own C++ has this validator and it is COMMENTED OUT:
    ///   `// DT_CHECK_MSG((reinterpret_cast<const uintptr_t>(this) & 127) == 0,`
    ///   `//              "QG header not aligned on a 128-byte boundary");`
    ///
    /// Every real caller is already aligned — placements come from `align128` — so this asserts a
    /// property that holds rather than imposing a new one. That is the point: the card enforces it
    /// by fencing, so the host proves it before the doorbell.
    pub fn from_bytes(bytes: u64) -> Self {
        // ⛔ `assert!`, NOT `debug_assert!`. A `debug_assert` compiles to NOTHING in the release
        // build that actually runs on the card — the only build where the fence happens. One
        // integer modulo against a DMA is not a cost worth being silent for.
        assert_eq!(
            bytes % Self::BYTES,
            0,
            "flit conversion of {bytes} B is not 128-byte aligned: `>> 7` would truncate it to \
             {} B and address a different location than the length field describes",
            (bytes >> Self::FLIT_SIZE_BITS) << Self::FLIT_SIZE_BITS,
        );
        Self(bytes >> Self::FLIT_SIZE_BITS)
    }

    /// The same conversion where truncation is INTENDED — a floor to the containing flit.
    ///
    /// Separate name so a caller that wants rounding says so, and the silent case cannot be
    /// reached by writing the ordinary one.
    pub const fn floor_from_bytes(bytes: u64) -> Self {
        Self(bytes >> Self::FLIT_SIZE_BITS)
    }

    pub const fn to_bytes(self) -> u64 {
        self.0 << Self::FLIT_SIZE_BITS
    }
}

/// Port of `SentientSoc::V1::VirtualAddressSBF`: 3-bit segment id (0-7) +
/// 27-bit offset in flits, packed into the low 30 bits of a field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VirtualAddressSbf {
    encoded: u64,
}

impl VirtualAddressSbf {
    /// `segment` must be in `0..=7`; out-of-range segments are truncated to
    /// their low 3 bits, mirroring the C++ bitfield's silent truncation
    /// (`BitField::pack` masks to width before shifting, same as the C++
    /// bitfield's own truncate-on-assign behavior).
    pub fn new(segment: u8, offset: Flits) -> Self {
        let seg = BitField::<29, 27>::pack(segment as u64);
        let off = BitField::<26, 0>::pack(offset.0);
        Self { encoded: seg | off }
    }

    pub fn from_physical_address(segment: u8, addr: DevicePhysicalAddress) -> Self {
        Self::new(segment, Flits::from_bytes(addr.0))
    }

    pub const fn value(self) -> u64 {
        self.encoded
    }

    pub const fn segment(self) -> u8 {
        BitField::<29, 27>::unpack(self.encoded) as u8
    }

    pub const fn offset(self) -> Flits {
        Flits(BitField::<26, 0>::unpack(self.encoded))
    }
}

/// Port of `SentientSoc::V1::CtrlSectionSBF` (8 bytes, `CTRL_SECTION` @ 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CtrlSection {
    word: u64,
}

/// Port of the response-tag field carried in both `CtrlSectionSBF::tag_`
/// (request side) and `ReturnSectionSBF::tag_` (response side) — used by
/// the caller to correlate a completion (`ResponseBlockWire`) back to the
/// control block that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResponseTag(pub u32);

impl CtrlSection {
    /// `barrier`: `SetBarrier` (bit 32). `tag`: `SetTag` (bits 31:8, 24-bit
    /// field — truncated silently like the C++ bitfield on overflow).
    pub fn new(tag: ResponseTag, barrier: bool) -> Self {
        let mut word = BitField::<31, 8>::pack(tag.0 as u64);
        if barrier {
            word |= BitField::<32, 32>::pack(1);
        }
        Self { word }
    }

    pub const fn to_word(self) -> u64 {
        self.word
    }
}

/// Port of `SentientSoc::V1::R5ControlBlockSectionSBF` (8 bytes,
/// `R5_SECTION` @ 8). This crate never issues R5/HDMA work, so only the
/// all-zero `R5TypeEnum::QGI` encoding ("skip the R5 section entirely",
/// per the header's own doc: "The QGI mode means that the R5 section will
/// be skipped") is provided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct R5Section {
    word: u64,
}

impl R5Section {
    pub const fn none() -> Self {
        Self { word: 0 }
    }

    pub const fn to_word(self) -> u64 {
        self.word
    }
}

/// Port of `SentientSoc::V1::TranslationSBF` (8 bytes). One of the 8 XLAT
/// entries making up `TranslationControlBlockSectionSBF` (64 bytes @ 64).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XlatEntry {
    word: u64,
}

/// `XlatModeEnum` (`translation_control_block_section_sbf.hpp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XlatAccessMode {
    Invalid,
    ReadOnly,
    ReadWrite,
}

impl XlatAccessMode {
    const fn bits(self) -> u64 {
        match self {
            XlatAccessMode::Invalid => 0x0,
            XlatAccessMode::ReadOnly => 0x2,
            XlatAccessMode::ReadWrite => 0x3,
        }
    }
}

/// Port of `RAS::CBRB::InvalidXlatSize`: the error the real `EncodeLength`
/// (`control_block_stream.cpp:92-103`) throws when a length exceeds the
/// hardware's maximum single-segment translation size. Previously this file
/// silently truncated an oversized length via bit-masking instead of
/// rejecting it (`BitField::pack` masks to width before shifting) — this
/// type is what turns that truncation into a caught error instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XlatLengthTooLarge {
    pub num_bytes: u64,
    pub max_xlat_size: u64,
}

impl std::fmt::Display for XlatLengthTooLarge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "XLAT length {} exceeds the max translation size {}",
            self.num_bytes, self.max_xlat_size
        )
    }
}

impl std::error::Error for XlatLengthTooLarge {}

/// Every way `ControlBlockWire::compute` can fail, matching the three real
/// throw sites in `ControlBlockStream::CreateGraphFreeCompute`:
/// `RAS::CBRB::InvalidXlatSize` (`control_block_stream.cpp:3704` for the
/// operand count, and via `EncodeLength` for an oversized allocation) and
/// `RAS::CBRB::InvalidBootstrapAddress` (`:3728-3736`).
///
/// These were `assert!`/`assert_eq!` — i.e. panics. That is not merely
/// unfaithful (the C++ throws, and its caller catches): the builders now run on
/// the submitting thread while it holds the `ResponseWorker`'s
/// `submission_mtx_`, so a panic there POISONS that mutex for the whole process
/// and strands every subsequent submission. A typed error propagates out
/// through `queue_cbs` the way the exception does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeCbError {
    /// `InvalidXlatSize`: an XLAT length exceeded the hardware maximum.
    XlatLength(XlatLengthTooLarge),
    /// `InvalidXlatSize`: more operands than the 7 tensor segments (0..=6),
    /// segment 7 being reserved for the program bundle.
    TooManyTensors { num_tensors: usize },
    /// `InvalidBootstrapAddress`: the bootstrap DMVA must name virtual
    /// segment 7, since hardware resolves the program through `xlat[7]`.
    InvalidBootstrapSegment { segment: u8 },
}

impl From<XlatLengthTooLarge> for ComputeCbError {
    fn from(e: XlatLengthTooLarge) -> Self {
        Self::XlatLength(e)
    }
}

impl std::fmt::Display for ComputeCbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::XlatLength(e) => write!(f, "{e}"),
            Self::TooManyTensors { num_tensors } => write!(
                f,
                "InvalidXlatSize: graph-free compute CB has 7 tensor segments (0..=6) plus the program segment (7), got {num_tensors} tensors"
            ),
            Self::InvalidBootstrapSegment { segment } => write!(
                f,
                "InvalidBootstrapAddress: bootstrap must reference virtual segment 7 (the program segment), got segment {segment}"
            ),
        }
    }
}

impl std::error::Error for ComputeCbError {}

/// Port of `control_block_stream.cpp:79`'s `TRANSLATION_SIZE` constant:
/// `16ULL * 1024 * 1024 * 1024` — "16 GB max XLAT translation size".
const TRANSLATION_SIZE: u64 = 16 * 1024 * 1024 * 1024;

/// 1:1 port of `EncodeLength` (`control_block_stream.cpp:92-103`):
///
/// ```text
/// static auto EncodeLength(size_t num_bytes) -> uint64_t
/// {
///     if(num_bytes > TRANSLATION_SIZE)
///     {
///         RAS::CBRB::InvalidXlatSize().NumBytes(num_bytes).MaxXlatSize(TRANSLATION_SIZE).Throw();
///     }
///     if(num_bytes == TRANSLATION_SIZE)
///     {
///         return 0;  // Hardware encoding: 0 = max size
///     }
///     return num_bytes;
/// }
/// ```
///
/// Returns a BYTE count (the caller then feeds it to a `SetLengthBytes`,
/// which is what does the `>> FLIT_SIZE_BITS`). Every real length in the
/// three ported CB shapes goes through this function — not just the XLAT
/// lengths but the DMA/fill section lengths too
/// (`control_block_stream.cpp:3794` in `CreateFillDma` and `:3846` in
/// `CreateComputeDataTransfer` both call `SetLengthBytes(EncodeLength(...))`).
fn encode_length(num_bytes: u64) -> Result<u64, XlatLengthTooLarge> {
    if num_bytes > TRANSLATION_SIZE {
        return Err(XlatLengthTooLarge {
            num_bytes,
            max_xlat_size: TRANSLATION_SIZE,
        });
    }
    if num_bytes == TRANSLATION_SIZE {
        return Ok(0); // Hardware encoding: 0 = max size
    }
    Ok(num_bytes)
}

impl XlatEntry {
    /// `paddr`/`len_bytes` in bytes; both are flit-granular in hardware,
    /// matching `TranslationSBF`'s byte-API constructor
    /// (`SetPaddrBytes`/`SetLengthBytes`). A `len_bytes` of exactly `1 << 34`
    /// bytes (2^27 flits, `TRANSLATION_SIZE`) encodes as the hardware's
    /// `0 == max length` sentinel, per the header's own note; anything
    /// STRICTLY GREATER than that is rejected — port of `EncodeLength`
    /// (`control_block_stream.cpp:92-103`): `if(num_bytes > TRANSLATION_SIZE)
    /// { RAS::CBRB::InvalidXlatSize().Throw(); }`.
    ///
    /// Private and untyped on purpose: this crate's two confirmed XLAT-length
    /// bugs were both "the wrong SOURCE of a plain `u64` got passed here" —
    /// so the public API is `for_segment_extent`/`for_allocation` below,
    /// which name which of `SegmentExtent`/`AllocationTotalSize` a caller has
    /// and unwrap it in exactly one place each, right next to the type it
    /// unwraps. No other function in this file calls `new_raw` directly.
    fn new_raw(
        paddr: DevicePhysicalAddress,
        len_bytes: u64,
        mode: XlatAccessMode,
    ) -> Result<Self, XlatLengthTooLarge> {
        // `xlat.SetPaddrBytes(paddr); xlat.SetLengthBytes(EncodeLength(len));
        //  xlat.SetAccessMode(mode);` — the exact 3-call sequence all three
        // ported shapes use (`control_block_stream.cpp:3747-3750`,
        // `:3801-3804`, `:3868-3871`). `SetPaddrBytes`/`SetLengthBytes` are
        // `bytes_to_flits` wrappers (`translation_control_block_section_sbf.hpp`
        // `SetPaddrBytes`/`SetLengthBytes`), hence `Flits::from_bytes` here.
        let encoded_len = encode_length(len_bytes)?;
        let mut word = BitField::<61, 32>::pack(Flits::from_bytes(paddr.0).0); // paddr_
        word |= BitField::<31, 30>::pack(mode.bits()); // mode_
        word |= BitField::<26, 0>::pack(Flits::from_bytes(encoded_len).0); // length_
        Ok(Self { word })
    }

    /// XLAT entry for a DMA-immediate/fill transfer: length is THIS
    /// transfer's own extent within its segment (`seg_offset + size_bytes`)
    /// — see `SegmentExtent`'s doc for exactly which real function and
    /// formula this ports (`CreateComputeDataTransfer`/`CreateFillDma`, NOT
    /// the region's total size).
    pub fn for_segment_extent(
        paddr: DevicePhysicalAddress,
        len: SegmentExtent,
        mode: XlatAccessMode,
    ) -> Result<Self, XlatLengthTooLarge> {
        Self::new_raw(paddr, len.0, mode)
    }

    /// XLAT entry for a compute tensor/program operand: length is that
    /// specific allocation's own size (`build_compute_cb`'s only source for
    /// this field), never the region it lives in.
    pub fn for_allocation(
        paddr: DevicePhysicalAddress,
        len: AllocationTotalSize,
        mode: XlatAccessMode,
    ) -> Result<Self, XlatLengthTooLarge> {
        Self::new_raw(paddr, len.0, mode)
    }

    pub const fn invalid() -> Self {
        Self { word: 0 }
    }

    pub const fn to_word(self) -> u64 {
        self.word
    }
}

/// Port of `SentientSoc::V1::ComputeControlBlockSectionSBF` (16 bytes,
/// `CMPT_SECTION` @ 48).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeSection {
    words: [u64; 2],
}

impl ComputeSection {
    /// Port of `CreateGraphFreeCompute`'s compute-CB construction: `SetValid()`
    /// + `SetBootstrapAddress(bootstrap)`, optionally `SetSync`.
    pub fn new(bootstrap: VirtualAddressSbf, sync: bool) -> Self {
        let mut word0 = BitField::<63, 63>::pack(1); // valid_
        if sync {
            word0 |= BitField::<62, 62>::pack(1); // sync_
        }
        word0 |= BitField::<36, 7>::pack(bootstrap.value()); // bootstrap_address_
        Self { words: [word0, 0] }
    }

    pub const fn skip() -> Self {
        Self { words: [0, 0] }
    }

    pub const fn to_words(self) -> [u64; 2] {
        self.words
    }
}

/// Port of `SentientSoc::V1::DmaTypeEnum` — only the subset this
/// graph-free-only runtime emits (immediate H2D/D2H DMA, memory fill, and
/// the required `SKIP` no-op default for the unused DMAI/DMAO slot).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DmaType {
    Skip,
    Immediate,
    MemoryFill,
}

impl DmaType {
    const fn bits(self) -> u64 {
        match self {
            DmaType::Skip => 0b0011,
            DmaType::Immediate => 0b0100,
            DmaType::MemoryFill => 0b0110,
        }
    }
}

/// Port of `SentientSoc::V1::DmaControlBlockSectionSBF` and its
/// `HdmaImmControlBlockSectionSBF`/`HdmaFillControlBlockSectionSBF`
/// specializations (16 bytes; used for both `DMI_SECTION` @ 16 and
/// `DMO_SECTION` @ 32 — direction is which slot the CB places it in, not a
/// field within the section itself).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DmaSection {
    words: [u64; 2],
}

impl DmaSection {
    /// The queue's other DMA slot when only one of DMAI/DMAO is used for
    /// this CB (e.g. a DMAI-only H2D transfer leaves DMAO as `skip()`).
    /// Port of `ControlBlockSBF`'s constructor default
    /// (`dmi().SetDmaType(SKIP); dmo().SetDmaType(SKIP);`).
    pub const fn skip() -> Self {
        Self {
            words: [0, DmaType::Skip.bits()],
        }
    }

    /// Port of `HdmaImmControlBlockSectionSBF` construction as used by
    /// `ControlBlockStream::CreateComputeDataTransfer`: immediate host<->device
    /// copy, no descriptor list. `device_addr`/`len_bytes` describe the
    /// device (DMVA) side; `host_addr` the host/IOVA side.
    ///
    /// The real construction sequence (`control_block_stream.cpp:3841-3845`) is
    /// ```text
    /// dma_section.SetDmaType(DmaTypeEnum::IMMEDIATE);
    /// dma_section.SetLengthBytes(EncodeLength(size_bytes));
    /// dma_section.SetDeviceAddress(dev_addr);
    /// dma_section.SetHAHostAddressBytes(host_iova);
    /// ```
    /// — note the DMA length goes through the SAME `EncodeLength` as an XLAT
    /// length, so a `len_bytes` above `TRANSLATION_SIZE` is a
    /// `RAS::CBRB::InvalidXlatSize` throw, not a silent 27-bit truncation.
    pub fn immediate(
        device_addr: VirtualAddressSbf,
        host_addr: HostPhysicalAddress,
        len_bytes: TransferSize,
    ) -> Result<Self, XlatLengthTooLarge> {
        let encoded_len = encode_length(len_bytes.0)?;
        let mut word0 = BitField::<61, 32>::pack(device_addr.value()); // device_address_
        // endianess_ (31:30) left NONE (0b00): no endian conversion — the real
        // `CreateComputeDataTransfer` never calls `SetEndianType`, so the
        // section keeps its zero-initialized `EndianTypeEnum::NONE`.
        word0 |= BitField::<26, 0>::pack(Flits::from_bytes(encoded_len).0); // length_

        // cbtype_ is global bits 67:64, i.e. word1-local bits 3:0 (subtract
        // 64 from both ends of the header's own (67, 64) range — NOT just
        // the low end). Once this crate shipped a real bug here: the shift
        // was computed from the field's HIGH bit (67-64=3) instead of the
        // LOW bit (64-64=0), so `word1 & 0xF` — what every reader of this
        // field actually checks — always read 0. `BitField` takes the
        // already-word1-local (3, 0) range directly and always shifts by
        // its own LO, so that specific mistake can no longer compile to a
        // different bug: the only way to get this wrong now is to write
        // down the wrong (HI, LO) pair, not to shift by the wrong one of them.
        let mut word1 = BitField::<3, 0>::pack(DmaType::Immediate.bits()); // cbtype_
        // host_address_ is global bits 127:71 -> word1-local bits 63:7.
        // (`SetHAHostAddressBytes` == `SetHAHostAddressFlits(bytes_to_flits(x))`,
        // `dma_control_block_section_base_sbf.hpp:268` + its byte setter.)
        word1 |= BitField::<63, 7>::pack(Flits::from_bytes(host_addr.as_u64()).0);

        Ok(Self {
            words: [word0, word1],
        })
    }

    /// Port of `HdmaFillControlBlockSectionSBF` construction as used by
    /// `ControlBlockStream::CreateFillDma`: writes a 32-bit pattern into
    /// `len_bytes` of device memory at `device_addr`. Placed on either slot —
    /// the real `CreateFillDma` picks the slot with
    /// `use_dmai ? cb.dmi<HdmaFillControlBlockSectionSBF&>() : cb.dmo<...>()`
    /// (`control_block_stream.cpp:3791`), and the header's own doc for
    /// `HdmaFillControlBlockSectionSBF` says the fill runs "on the DMAI or the
    /// DMAO queues".
    ///
    /// Real sequence (`control_block_stream.cpp:3792-3795`):
    /// ```text
    /// dma.SetDmaType(DmaTypeEnum::MEMORY_FILL);
    /// dma.SetDeviceAddress(VirtualAddressSBF::bytes_to_flits(dmva));
    /// dma.SetLengthBytes(EncodeLength(size_bytes));
    /// dma.SetHAFill(fill_pattern);
    /// ```
    // Argument count mirrors the real `ControlBlockStream::Create*` signature
    pub fn fill(
        device_addr: VirtualAddressSbf,
        len_bytes: TransferSize,
        pattern: u32,
    ) -> Result<Self, XlatLengthTooLarge> {
        let encoded_len = encode_length(len_bytes.0)?;
        let mut word0 = BitField::<61, 32>::pack(device_addr.value()); // device_address_
        word0 |= BitField::<26, 0>::pack(Flits::from_bytes(encoded_len).0); // length_

        // cbtype_ (word1-local 3:0, see immediate()'s comment above for why
        // this is (3, 0) and not (67, 64)).
        let mut word1 = BitField::<3, 0>::pack(DmaType::MemoryFill.bits());
        // fill_ is global bits 127:96 -> word1-local bits 63:32.
        word1 |= BitField::<63, 32>::pack(pattern as u64);

        Ok(Self {
            words: [word0, word1],
        })
    }

    pub const fn to_words(self) -> [u64; 2] {
        self.words
    }
}

/// Byte offsets of the 6 CB sections within the 128-byte control block, per
/// `hal/1p0/control_block_sbf.hpp`'s `CTRL_SECTION`..`XLAT_SECTION`
/// constants.
const CTRL_SECTION: usize = 0;
const R5_SECTION: usize = 8;
const DMI_SECTION: usize = 16;
const DMO_SECTION: usize = 32;
const CMPT_SECTION: usize = 48;
const XLAT_SECTION: usize = 64;
/// Port of `hal/1p0/control_block_sbf.hpp`'s `CB_NUM_BYTES`.
pub const CB_NUM_BYTES: usize = 128;
/// Port of `hal/1p0/response_block_sbf.hpp`'s `RB_NUM_BYTES`.
pub const RB_NUM_BYTES: usize = 64;

/// A fully-encoded, hardware-ready control block: the real bytes that get
/// queued via `senlib_ffi_scheduler::cbi_queue_control_blocks_sbf`. This is
/// the type `HardwareSubmit::submit` now carries (see `scheduler.rs`) —
/// previously that call site carried no payload at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlBlockWire {
    bytes: [u8; CB_NUM_BYTES],
}

impl ControlBlockWire {
    fn assemble(
        ctrl: CtrlSection,
        r5: R5Section,
        dmi: DmaSection,
        dmo: DmaSection,
        cmpt: ComputeSection,
        xlat: [XlatEntry; 8],
    ) -> Self {
        let mut bytes = [0u8; CB_NUM_BYTES];
        bytes[CTRL_SECTION..CTRL_SECTION + 8].copy_from_slice(&ctrl.to_word().to_le_bytes());
        bytes[R5_SECTION..R5_SECTION + 8].copy_from_slice(&r5.to_word().to_le_bytes());
        for (i, w) in dmi.to_words().iter().enumerate() {
            bytes[DMI_SECTION + i * 8..DMI_SECTION + i * 8 + 8].copy_from_slice(&w.to_le_bytes());
        }
        for (i, w) in dmo.to_words().iter().enumerate() {
            bytes[DMO_SECTION + i * 8..DMO_SECTION + i * 8 + 8].copy_from_slice(&w.to_le_bytes());
        }
        for (i, w) in cmpt.to_words().iter().enumerate() {
            bytes[CMPT_SECTION + i * 8..CMPT_SECTION + i * 8 + 8].copy_from_slice(&w.to_le_bytes());
        }
        for (i, entry) in xlat.iter().enumerate() {
            let off = XLAT_SECTION + i * 8;
            bytes[off..off + 8].copy_from_slice(&entry.to_word().to_le_bytes());
        }
        Self { bytes }
    }

    /// Port of `ControlBlockStream::CreateGraphFreeCompute`: one compute CB
    /// with `xlat[i] = tensor_paddrs[i]` (RW) for each input/output tensor,
    /// `xlat[7] = prog_paddr` (RW), and `cmpt` pointing at `bootstrap_addr`
    /// (which must resolve into segment 7, the program segment, per the
    /// header's doc: "the bootstrap must reference a valid xlat segment (7)").
    ///
    /// `tensors` is in inp_out order exactly like the C++ (`tensor_paddrs`/
    /// `tensor_sizes`), i.e. index == segment id for indices `0..=6`; the
    /// program occupies segment 7 and is passed separately as
    /// `(prog_paddr, prog_size)`.
    ///
    /// # Errors
    /// Returns `Err` if any tensor/program length exceeds the hardware's max
    /// XLAT translation size (`XlatEntry::new_raw`'s `EncodeLength` port),
    /// or if `bootstrap_addr` does not resolve into segment 7 — port of
    /// `RAS::CBRB::InvalidBootstrapAddress` (`control_block_stream.cpp:3728-3736`):
    /// "the bootstrap must reference virtual segment 7" (the program
    /// segment), since the hardware interprets the bootstrap address as a
    /// virtual DMVA and looks up `xlat[segment_id]` to find the program.
    // Argument count mirrors the real `ControlBlockStream::Create*` signature
    pub fn compute(
        tag: ResponseTag,
        tensors: &[(DevicePhysicalAddress, AllocationTotalSize)],
        prog: (DevicePhysicalAddress, AllocationTotalSize),
        bootstrap_addr: VirtualAddressSbf,
    ) -> Result<Self, ComputeCbError> {
        if tensors.len() > 7 {
            return Err(ComputeCbError::TooManyTensors {
                num_tensors: tensors.len(),
            });
        }
        if bootstrap_addr.segment() != 7 {
            return Err(ComputeCbError::InvalidBootstrapSegment {
                segment: bootstrap_addr.segment(),
            });
        }
        let mut xlat = [XlatEntry::invalid(); 8];
        for (i, (paddr, size)) in tensors.iter().enumerate() {
            xlat[i] = XlatEntry::for_allocation(*paddr, *size, XlatAccessMode::ReadWrite)?;
        }
        xlat[7] = XlatEntry::for_allocation(prog.0, prog.1, XlatAccessMode::ReadWrite)?;

        Ok(Self::assemble(
            CtrlSection::new(tag, false),
            R5Section::none(),
            DmaSection::skip(),
            DmaSection::skip(),
            // REAL BUG, CONFIRMED AND FIXED: the real `CreateGraphFreeCompute`
            // (`control_block_stream.cpp:3741`) unconditionally sets
            // `SetSync(true)` for every graph-free compute CB, with its own
            // comment explaining why: "generates a response block that acts
            // as a cross-pipeline fence, ensuring subsequent DMAO (D2H)
            // operations on a different pipeline wait for compute to
            // complete before reading output tensors." This crate previously
            // hardcoded `sync=false` unconditionally instead — the exact
            // opposite of the real, unconditional value — meaning no such
            // fence/response guarantee was ever requested for any compute CB
            // this crate ever submitted.
            ComputeSection::new(bootstrap_addr, true),
            xlat,
        ))
    }

    /// Port of `ControlBlockStream::CreateComputeDataTransfer`
    /// (`control_block_stream.cpp:3803`) — NOT the legacy-graph-only
    /// `buildDmaControlBlocks` (`pf_runtime_scheduler_utils.cpp`), which an
    /// earlier version of this doc comment wrongly cited: a graph-free
    /// immediate DMA CB with physical addresses set directly (no IOMMU/PIPO
    /// mapping, no XLAT needed for the DMA path itself — `region_paddr` is
    /// carried in `xlat[dmva.segment()]` purely so a subsequent compute CB in
    /// the same submission can address the same region, matching the header's
    /// note that this method "builds a graph-free CB with physical addresses
    /// set directly").
    ///
    /// # Errors
    /// Returns `Err` if `xlat_length` exceeds the hardware's max XLAT
    /// translation size (`XlatEntry::new_raw`'s `EncodeLength` port).
    // Argument count mirrors the real `ControlBlockStream::Create*` signature
    /// `xlat.length` is a port of `CreateComputeDataTransfer`'s own xlat
    /// computation (`control_block_stream.cpp:3907-3911`):
    /// ```text
    /// auto seg_offset = SegmentOffset(dmva);
    /// xlat.SetLengthBytes(EncodeLength(seg_offset + size_bytes));
    /// ```
    /// i.e. THIS transfer's own starting offset within its segment plus its own
    /// byte count — never the owning region's total allocated size. An earlier
    /// version of that field was typed `RegionTotalSize` and documented as
    /// porting `buildDmaControlBlocks` (`pf_runtime_scheduler_utils.cpp:48`) —
    /// a DIFFERENT, legacy-graph-only call path that does not call this
    /// function at all; it was cited here in error. [`SegmentExtent`] and
    /// [`TransferSize`] are deliberately distinct types with no conversion
    /// between them, so passing one where the other belongs is a compile error
    /// rather than a hardware XLAT-bounds rejection.
    pub fn dma_immediate(
        tag: ResponseTag,
        to_device: bool,
        transfer: DmaTransfer,
        xlat: XlatWindow,
        pipeline_barrier: bool,
    ) -> Result<Self, XlatLengthTooLarge> {
        let DmaTransfer {
            dmva,
            host_addr,
            size_bytes,
        } = transfer;
        let XlatWindow {
            region_paddr,
            length: xlat_length,
        } = xlat;
        let section = DmaSection::immediate(dmva, host_addr, size_bytes)?;
        let (dmi, dmo) = if to_device {
            (section, DmaSection::skip())
        } else {
            (DmaSection::skip(), section)
        };

        let mut xlat = [XlatEntry::invalid(); 8];
        xlat[dmva.segment() as usize] =
            XlatEntry::for_segment_extent(region_paddr, xlat_length, XlatAccessMode::ReadWrite)?;

        Ok(Self::assemble(
            CtrlSection::new(tag, pipeline_barrier),
            R5Section::none(),
            dmi,
            dmo,
            ComputeSection::skip(),
            xlat,
        ))
    }

    /// Port of `ControlBlockStream::CreateFillDma` (`control_block_stream.cpp:3870`):
    /// writes `fill_pattern` into `size_bytes` of device memory at `dmva`.
    /// `use_dmai` selects the DMAI vs DMAO queue, per the header's doc.
    /// `xlat_length` is THIS transfer's own extent within its segment
    /// (`seg_offset + size_bytes`) — see `dma_immediate`'s identical param
    /// and `SegmentExtent`'s doc for exactly which real formula this is
    /// (`CreateFillDma`'s own `xlat.SetLengthBytes(EncodeLength(seg_offset +
    /// size_bytes))`, not the owning region's total size — an earlier
    /// version of this parameter was wrongly typed `RegionTotalSize`).
    ///
    /// # Errors
    /// Returns `Err` if `xlat_length` exceeds the hardware's max XLAT
    /// translation size (`XlatEntry::new_raw`'s `EncodeLength` port).
    // Argument count mirrors the real `ControlBlockStream::Create*` signature
    pub fn fill(
        tag: ResponseTag,
        dmva: VirtualAddressSbf,
        size_bytes: TransferSize,
        fill_pattern: u32,
        use_dmai: bool,
        region_paddr: DevicePhysicalAddress,
        xlat_length: SegmentExtent,
    ) -> Result<Self, XlatLengthTooLarge> {
        let section = DmaSection::fill(dmva, size_bytes, fill_pattern)?;
        let (dmi, dmo) = if use_dmai {
            (section, DmaSection::skip())
        } else {
            (DmaSection::skip(), section)
        };

        let mut xlat = [XlatEntry::invalid(); 8];
        xlat[dmva.segment() as usize] =
            XlatEntry::for_segment_extent(region_paddr, xlat_length, XlatAccessMode::ReadWrite)?;

        Ok(Self::assemble(
            CtrlSection::new(tag, false),
            R5Section::none(),
            dmi,
            dmo,
            ComputeSection::skip(),
            xlat,
        ))
    }

    /// Raw bytes ready to hand to `senlib_ffi_scheduler::SenlibControlBlockSbf`
    /// (which is byte-for-byte this same 128-byte layout).
    pub const fn to_bytes(&self) -> [u8; CB_NUM_BYTES] {
        self.bytes
    }

    /// This CB's own `CtrlSectionSBF::tag_` (bits 31:8 of the CTRL word,
    /// same encoding `CtrlSection::new`/`ResponseBlockWire::tag` use) — the
    /// tag the matching `ResponseBlockWire` will carry back once senlib
    /// completes this exact CB. Lets a caller correlate a drained response
    /// to the request that produced it instead of assuming responses arrive
    /// in submission order.
    pub fn tag(&self) -> ResponseTag {
        let mut word = [0u8; 8];
        word.copy_from_slice(&self.bytes[CTRL_SECTION..CTRL_SECTION + 8]);
        let word = u64::from_le_bytes(word);
        ResponseTag(BitField::<31, 8>::unpack(word) as u32)
    }
}

/// A decoded response block: the real bytes senlib hands back via
/// `senlib_ffi_scheduler::rbi_receive_responses_sbf`. Port of the
/// `ReturnSectionSBF` decode used by `ResponseWorker`/`DmaControlBlockHandler`
/// to determine per-CB completion status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResponseBlockWire {
    ret_word: u64,
}

/// Port of `SentientSoc::V1::RBStatusTypeEnum`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseStatus {
    Good,
    Retry,
    Abend,
    Error,
    /// Any encoded value outside the 4 defined `RBStatusTypeEnum` variants —
    /// kept explicit rather than panicking, since this is decoding bytes
    /// written by hardware/firmware, not values this crate constructed.
    Unknown(u8),
}

impl ResponseBlockWire {
    /// The value a `PendingRequest` holds when NO response has arrived — the
    /// port of the C++'s zero-initialised `ResponseBlockSBF response_block;`
    /// member (`response_worker.hpp:369`), which is why `pr.response_block.ret()`
    /// is always legal there and all three arms of `onCbComplete`'s
    /// disjunction are always evaluated. Decodes to `status = Good`,
    /// `cancel = 0`: absence contributes NOTHING to a completion verdict, and
    /// in particular cannot erase the `state == TIMED_OUT` arm.
    pub const ZERO: Self = Self { ret_word: 0 };

    /// Decode from the raw 64-byte RB. Only the `ReturnSectionSBF` (bytes
    /// 0..8, `RET_SECTION`) is decoded — timestamps/power/app-specific
    /// sections are not consumed by this crate's completion path.
    pub fn from_bytes(bytes: &[u8; RB_NUM_BYTES]) -> Self {
        let mut word = [0u8; 8];
        word.copy_from_slice(&bytes[0..8]);
        Self {
            ret_word: u64::from_le_bytes(word),
        }
    }

    /// `ReturnSectionSBF::GetTag` (bits 31:8).
    pub const fn tag(self) -> ResponseTag {
        ResponseTag(BitField::<31, 8>::unpack(self.ret_word) as u32)
    }

    /// `ReturnSectionSBF::GetStatus` (bits 6:5).
    pub const fn status(self) -> ResponseStatus {
        match BitField::<6, 5>::unpack(self.ret_word) {
            0x0 => ResponseStatus::Good,
            0x1 => ResponseStatus::Retry,
            0x2 => ResponseStatus::Abend,
            0x3 => ResponseStatus::Error,
            other => ResponseStatus::Unknown(other as u8),
        }
    }

    /// `ReturnSectionSBF::GetLocator` (bits 39:32) — which RB queue this
    /// response came from (`APPLICATION`/`CB_QUEUE`/`QGI_NO_DIAG`/
    /// `QGI_DIAG`/`DMO`/`DMI`/`RESERVED`), raw byte value.
    pub const fn locator_bytes(self) -> u8 {
        BitField::<39, 32>::unpack(self.ret_word) as u8
    }

    /// `ReturnSectionSBF::GetMark` (bit 7).
    pub const fn mark(self) -> bool {
        BitField::<7, 7>::unpack(self.ret_word) != 0
    }

    /// `ReturnSectionSBF::GetEdep` (bit 4).
    pub const fn edep(self) -> bool {
        BitField::<4, 4>::unpack(self.ret_word) != 0
    }

    /// `ReturnSectionSBF::GetFlr` (bit 3).
    pub const fn flr(self) -> bool {
        BitField::<3, 3>::unpack(self.ret_word) != 0
    }

    /// `ReturnSectionSBF::GetCancel` (bit 2).
    pub const fn cancelled(self) -> bool {
        BitField::<2, 2>::unpack(self.ret_word) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cb_is_128_bytes_and_sections_do_not_overlap() {
        // Mirrors the C++ static_asserts in control_block_sbf.hpp.
        assert_eq!(CB_NUM_BYTES, 128);
        assert_eq!(CTRL_SECTION + 8, R5_SECTION);
        assert_eq!(R5_SECTION + 8, DMI_SECTION);
        assert_eq!(DMI_SECTION + 16, DMO_SECTION);
        assert_eq!(DMO_SECTION + 16, CMPT_SECTION);
        assert_eq!(CMPT_SECTION + 16, XLAT_SECTION);
        assert_eq!(XLAT_SECTION + 64, CB_NUM_BYTES);
    }

    #[test]
    fn rb_is_64_bytes() {
        assert_eq!(RB_NUM_BYTES, 64);
    }

    #[test]
    fn virtual_address_round_trips_segment_and_offset() {
        let va = VirtualAddressSbf::new(7, Flits::from_bytes(1 << 20));
        assert_eq!(va.segment(), 7);
        assert_eq!(va.offset().to_bytes(), 1 << 20);
    }

    #[test]
    fn compute_cb_sets_valid_and_bootstrap_and_xlats() {
        let bootstrap = VirtualAddressSbf::new(7, Flits::from_bytes(0));
        let prog = (
            DevicePhysicalAddress(0x1_0000_0000),
            AllocationTotalSize(4096),
        );
        let tensors = [(
            DevicePhysicalAddress(0x2_0000_0000),
            AllocationTotalSize(65536),
        )];
        let cb = ControlBlockWire::compute(ResponseTag(42), &tensors, prog, bootstrap).unwrap();
        let bytes = cb.to_bytes();

        // CMPT section: valid_ bit (word0 bit 63) must be set.
        let cmpt_word0 =
            u64::from_le_bytes(bytes[CMPT_SECTION..CMPT_SECTION + 8].try_into().unwrap());
        assert_ne!(cmpt_word0 & (1 << 63), 0, "valid_ bit must be set");

        // XLAT[0] should carry the tensor's physical address (as flits).
        let xlat0 = u64::from_le_bytes(bytes[XLAT_SECTION..XLAT_SECTION + 8].try_into().unwrap());
        let paddr_flits = (xlat0 >> 32) & 0x3FFF_FFFF;
        assert_eq!(paddr_flits, Flits::from_bytes(0x2_0000_0000).0);
        let mode = (xlat0 >> 30) & 0b11;
        assert_eq!(mode, XlatAccessMode::ReadWrite.bits());

        // XLAT[7] should carry the program's physical address.
        let xlat7_off = XLAT_SECTION + 7 * 8;
        let xlat7 = u64::from_le_bytes(bytes[xlat7_off..xlat7_off + 8].try_into().unwrap());
        assert_eq!(
            (xlat7 >> 32) & 0x3FFF_FFFF,
            Flits::from_bytes(0x1_0000_0000).0
        );
    }

    #[test]
    fn dma_immediate_cb_selects_dmi_slot_for_host_to_device() {
        let dmva = VirtualAddressSbf::new(2, Flits::from_bytes(0x1000));
        let cb = ControlBlockWire::dma_immediate(
            ResponseTag(1),
            true,
            DmaTransfer {
                dmva,
                host_addr: HostPhysicalAddress::from_raw_for_tests(0x7f00_0000),
                size_bytes: TransferSize(256),
            },
            XlatWindow {
                region_paddr: DevicePhysicalAddress(0x2_0000_0000),
                length: SegmentExtent(0x1000 + 256),
            }, // not asserted on by this test, using the transfer size is inert
            false,
        )
        .unwrap();
        let bytes = cb.to_bytes();
        let dmi_word1 =
            u64::from_le_bytes(bytes[DMI_SECTION + 8..DMI_SECTION + 16].try_into().unwrap());
        let dmo_word1 =
            u64::from_le_bytes(bytes[DMO_SECTION + 8..DMO_SECTION + 16].try_into().unwrap());
        assert_eq!(
            dmi_word1 & 0xF,
            DmaType::Immediate.bits(),
            "DMI slot must carry the IMMEDIATE cbtype"
        );
        assert_eq!(
            dmo_word1 & 0xF,
            DmaType::Skip.bits(),
            "unused DMO slot must be SKIP"
        );
    }

    #[test]
    fn fill_cb_carries_pattern_in_dmai_slot() {
        let dmva = VirtualAddressSbf::new(3, Flits::from_bytes(0));
        let cb = ControlBlockWire::fill(
            ResponseTag(9),
            dmva,
            TransferSize(4096),
            0xDEAD_BEEF,
            true,
            DevicePhysicalAddress(0x3_0000_0000),
            SegmentExtent(4096), // not asserted on by this test, using the transfer size is inert
        )
        .unwrap();
        let bytes = cb.to_bytes();
        let dmi_word1 =
            u64::from_le_bytes(bytes[DMI_SECTION + 8..DMI_SECTION + 16].try_into().unwrap());
        assert_eq!(dmi_word1 & 0xF, DmaType::MemoryFill.bits());
        assert_eq!((dmi_word1 >> 32) & 0xFFFF_FFFF, 0xDEAD_BEEFu64);
    }

    /// Regression test for CRITICAL FIX #1: the real functions these wire
    /// methods port — `CreateComputeDataTransfer`/`CreateFillDma`
    /// (`control_block_stream.cpp:3803`/`3870`) — set the XLAT length to
    /// `EncodeLength(seg_offset + size_bytes)`: THIS transfer's own starting
    /// offset within its segment, plus its own byte count. An earlier
    /// version of this crate instead passed the owning region's total size
    /// (confusing this with the DIFFERENT, legacy-graph-only
    /// `buildDmaControlBlocks` call path), which does not depend on this
    /// transfer's own offset/size at all. This test asserts the XLAT length
    /// actually written into the wire bytes is `seg_offset + size_bytes`,
    /// not some unrelated region-total value.
    #[test]
    fn xlat_length_is_seg_offset_plus_this_transfers_own_size() {
        let seg_offset_bytes: u64 = 0x2000;
        let size_bytes: u64 = 512;
        let dmva = VirtualAddressSbf::new(4, Flits::from_bytes(seg_offset_bytes));
        let cb = ControlBlockWire::dma_immediate(
            ResponseTag(3),
            true,
            DmaTransfer {
                dmva,
                host_addr: HostPhysicalAddress::from_raw_for_tests(0x7f00_1000),
                size_bytes: TransferSize(size_bytes),
            },
            XlatWindow {
                region_paddr: DevicePhysicalAddress(0x4_0000_0000),
                length: SegmentExtent(seg_offset_bytes + size_bytes),
            },
            false,
        )
        .unwrap();
        let bytes = cb.to_bytes();
        let xlat_off = XLAT_SECTION + 4 * 8;
        let xlat = u64::from_le_bytes(bytes[xlat_off..xlat_off + 8].try_into().unwrap());
        let length_flits = xlat & 0x7FF_FFFF;
        assert_eq!(
            length_flits,
            Flits::from_bytes(seg_offset_bytes + size_bytes).0,
            "XLAT length must be seg_offset + size_bytes (this transfer's own extent), not the region's total size"
        );
    }

    /// Port of `EncodeLength`'s (`control_block_stream.cpp:92-103`) bounds
    /// check: `RAS::CBRB::InvalidXlatSize` when the length exceeds
    /// `TRANSLATION_SIZE` (16 GiB). Previously `XlatEntry::new_raw` had no
    /// such check and silently truncated an oversized length via bitmasking.
    #[test]
    fn xlat_length_over_16gib_is_rejected() {
        let dmva = VirtualAddressSbf::new(5, Flits::from_bytes(0));
        let result = ControlBlockWire::fill(
            ResponseTag(1),
            dmva,
            TransferSize(4096),
            0,
            true,
            DevicePhysicalAddress(0x5_0000_0000),
            SegmentExtent((1u64 << 34) + 1),
        );
        assert!(
            result.is_err(),
            "a length exceeding TRANSLATION_SIZE (1<<34) must be rejected, not silently truncated"
        );
    }

    /// `EncodeLength`'s exact boundary: `== TRANSLATION_SIZE` is the
    /// hardware's "0 == max length" sentinel, not an error.
    #[test]
    fn xlat_length_of_exactly_16gib_encodes_as_max_sentinel() {
        let dmva = VirtualAddressSbf::new(6, Flits::from_bytes(0));
        let cb = ControlBlockWire::fill(
            ResponseTag(1),
            dmva,
            TransferSize(4096),
            0,
            true,
            DevicePhysicalAddress(0x6_0000_0000),
            SegmentExtent(1u64 << 34),
        )
        .unwrap();
        let bytes = cb.to_bytes();
        let xlat_off = XLAT_SECTION + 6 * 8;
        let xlat = u64::from_le_bytes(bytes[xlat_off..xlat_off + 8].try_into().unwrap());
        assert_eq!(
            xlat & 0x7FF_FFFF,
            0,
            "exactly TRANSLATION_SIZE must encode as the 0 == max-length sentinel"
        );
    }

    /// Port of `RAS::CBRB::InvalidBootstrapAddress`
    /// (`control_block_stream.cpp:3728-3736`): the bootstrap address must
    /// resolve into segment 7 (the program segment).
    /// Now a typed `Err`, not a panic: these builders run on the submitting
    /// thread under the `ResponseWorker`'s `submission_mtx_`, where a panic
    /// would poison that mutex process-wide (the C++ throws and its caller
    /// catches).
    #[test]
    fn compute_cb_rejects_bootstrap_outside_segment_7() {
        let bootstrap = VirtualAddressSbf::new(3, Flits::from_bytes(0));
        let prog = (
            DevicePhysicalAddress(0x1_0000_0000),
            AllocationTotalSize(4096),
        );
        let err = ControlBlockWire::compute(ResponseTag(1), &[], prog, bootstrap)
            .expect_err("a bootstrap outside segment 7 must be rejected");
        assert_eq!(err, ComputeCbError::InvalidBootstrapSegment { segment: 3 });
        assert!(err.to_string().contains("InvalidBootstrapAddress"));
    }

    /// The operand-count limit is likewise an `Err`, not a panic
    /// (`RAS::CBRB::InvalidXlatSize`, `control_block_stream.cpp:3704`): 7 tensor
    /// segments (0..=6), with segment 7 reserved for the program bundle.
    #[test]
    fn compute_cb_rejects_more_than_seven_tensors() {
        let bootstrap = VirtualAddressSbf::new(7, Flits::from_bytes(0));
        let prog = (
            DevicePhysicalAddress(0x1_0000_0000),
            AllocationTotalSize(4096),
        );
        let tensors = vec![
            (
                DevicePhysicalAddress(0x2_0000_0000),
                AllocationTotalSize(4096)
            );
            8
        ];
        let err = ControlBlockWire::compute(ResponseTag(1), &tensors, prog, bootstrap)
            .expect_err("8 tensors must be rejected");
        assert_eq!(err, ComputeCbError::TooManyTensors { num_tensors: 8 });
        assert!(err.to_string().contains("InvalidXlatSize"));
    }

    /// The DMA/fill section's OWN length also goes through `EncodeLength`
    /// (`control_block_stream.cpp:3794` / `:3846`), so a transfer size above
    /// `TRANSLATION_SIZE` is rejected exactly like an oversized XLAT length
    /// rather than silently truncated into the 27-bit `length_` field.
    #[test]
    fn dma_transfer_size_over_16gib_is_rejected() {
        let dmva = VirtualAddressSbf::new(1, Flits::from_bytes(0));
        let result = ControlBlockWire::dma_immediate(
            ResponseTag(1),
            true,
            DmaTransfer {
                dmva,
                host_addr: HostPhysicalAddress::from_raw_for_tests(0x7f00_0000),
                size_bytes: TransferSize((1u64 << 34) + 1),
            },
            XlatWindow {
                region_paddr: DevicePhysicalAddress(0x1_0000_0000),
                length: SegmentExtent(4096),
            },
            false,
        );
        assert!(
            result.is_err(),
            "a DMA length exceeding TRANSLATION_SIZE must be rejected, not truncated"
        );
    }

    /// `EncodeLength(TRANSLATION_SIZE) == 0`, and `SetLengthBytes(0)` writes 0
    /// flits — the DMA `length_` field's own `0 == maximum + 1` sentinel
    /// (`DmaDlength::GetLengthFlits`).
    #[test]
    fn dma_transfer_size_of_exactly_16gib_encodes_as_max_sentinel() {
        let dmva = VirtualAddressSbf::new(1, Flits::from_bytes(0));
        let cb = ControlBlockWire::dma_immediate(
            ResponseTag(1),
            true,
            DmaTransfer {
                dmva,
                host_addr: HostPhysicalAddress::from_raw_for_tests(0x7f00_0000),
                size_bytes: TransferSize(1u64 << 34),
            },
            XlatWindow {
                region_paddr: DevicePhysicalAddress(0x1_0000_0000),
                length: SegmentExtent(4096),
            },
            false,
        )
        .unwrap();
        let dmi_word0 = u64::from_le_bytes(
            cb.to_bytes()[DMI_SECTION..DMI_SECTION + 8]
                .try_into()
                .unwrap(),
        );
        assert_eq!(
            dmi_word0 & 0x7FF_FFFF,
            0,
            "exactly TRANSLATION_SIZE must encode as the 0 == max-length sentinel"
        );
    }

    #[test]
    fn response_block_decodes_status_tag_and_cancel() {
        // GOOD status (0b00), tag=0x1234, not cancelled.
        let mut ret_word = 0u64;
        ret_word |= 0x1234u64 << 8;
        let mut bytes = [0u8; RB_NUM_BYTES];
        bytes[0..8].copy_from_slice(&ret_word.to_le_bytes());
        let rb = ResponseBlockWire::from_bytes(&bytes);
        assert_eq!(rb.tag(), ResponseTag(0x1234));
        assert_eq!(rb.status(), ResponseStatus::Good);
        assert!(!rb.cancelled());

        // ERROR status (0b11), cancelled bit set.
        let ret_word = (0b11u64 << 5) | (1 << 2);
        let mut bytes = [0u8; RB_NUM_BYTES];
        bytes[0..8].copy_from_slice(&ret_word.to_le_bytes());
        let rb = ResponseBlockWire::from_bytes(&bytes);
        assert_eq!(rb.status(), ResponseStatus::Error);
        assert!(rb.cancelled());
    }
}

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
    /// The field's width mask, UNshifted (bits WIDTH-1:0 of a VALUE) — the
    /// operand for masking a field value before it is packed or after it is
    /// unpacked. NEVER the clear operand against a word: a word's copy of
    /// this field sits at bits HI:LO, so clearing it wants
    /// [`Self::IN_POSITION_MASK`] (#655: `set_tag` cleared with `!MASK` and
    /// wiped the wrong byte range).
    const MASK: u64 = if Self::WIDTH == 64 {
        u64::MAX
    } else {
        (1u64 << Self::WIDTH) - 1
    };

    /// This field's width mask shifted into its word position — the mask
    /// that selects exactly this field's bits WITHIN a 64-bit word. `MASK`
    /// is the UNshifted width mask, so a clear written as `word & !MASK`
    /// hits the wrong bit range whenever `LO != 0`: #655 shipped exactly
    /// that in `set_tag`, where `!BitField::<31, 8>::MASK` (bits 23:0)
    /// zeroed the CTRL cancel byte (bits 7:0) while KEEPING the tag field's
    /// own top byte (bits 31:24) — the reverse of the intent on both sides.
    /// Every mask-and-set field edit clears with `!IN_POSITION_MASK` and
    /// sets with [`Self::pack`], so the unshifted-`MASK` clear has no
    /// remaining spelling.
    const IN_POSITION_MASK: u64 = Self::pack(Self::MASK);

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

    /// `SetCancelled` (`cancel_`, bits 7:0 — the CTRL word's low byte, per
    /// the `hal/snt1p0/ctrl_section_sbf.hpp` bitfield table). Applied after
    /// construction, exactly where the vendor applies it: the real
    /// `CreateCancelledCb` builds the whole CB first and only then sets the
    /// flag. The field is 8 bits wide, but the vendor's setter only ever
    /// writes the boolean (`true` ⇒ `1`), so this packs `1` and takes no
    /// argument.
    pub const fn with_cancel(mut self) -> Self {
        self.word |= BitField::<7, 0>::pack(1);
        self
    }

    /// `GetCancel` — the same low-byte field, read back. The vendor's
    /// readers test it as a boolean, so any nonzero value reads cancelled.
    pub const fn is_cancelled(self) -> bool {
        BitField::<7, 0>::unpack(self.word) != 0
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
/// Port of `hal/1p0/response_block_sbf.hpp`'s `APP_SECTION` — where the application-specific
/// (QGI/HMI, DMA, R5) union starts in a response block. The section runs to [`RB_NUM_BYTES`], i.e.
/// four 64-bit beats; see [`AppQgiSection`].
pub const APP_SECTION: usize = 32;
/// Port of `hal/1p0/response_block_sbf.hpp`'s `TS_SECTION` — where the timestamp section starts:
/// `TimeStampSectionSBF` is `array<TimeStampSBF, NUM_SECTION_TIMESTAMPS>` with `NUM_SECTION_TIMESTAMPS
/// == 5` and `sizeof(TimeStampSBF) == 4` (`time_stamp_sbf.hpp`), so the section is bytes 8..28 as
/// five little-endian 32-bit words.
pub const TS_SECTION: usize = 8;
/// Port of `hal/1p0/response_block_sbf.hpp`'s `PW_SECTION` — the 4-byte power measurement word
/// between the timestamp section and [`APP_SECTION`] (`sizeof(PowerMeasurementSectionSBF) == 4`,
/// pinned by the response block's own `static_assert(PW_SECTION + sizeof(...) == APP_SECTION)`).
pub const PW_SECTION: usize = 28;

/// A fully-encoded, hardware-ready control block: the real bytes that get
/// queued via `senlib_ffi::cbi_queue_control_blocks_sbf`. This is
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

    /// The 128 wire bytes, read-only — for diagnostics that name the first
    /// differing byte of two rendered images (#603 PR2's parity gate).
    pub const fn bytes(&self) -> &[u8; CB_NUM_BYTES] {
        &self.bytes
    }

    /// The 128 wire bytes, mutable — for the replay player's patch stage
    /// (#603 PR2): an image copied into a ring slot is patched in place
    /// (per-step word edits at prepare-derived offsets), never re-encoded.
    /// The wire stays 128 bytes; the caller owns keeping the fields it does
    /// not touch intact.
    pub fn bytes_mut(&mut self) -> &mut [u8; CB_NUM_BYTES] {
        &mut self.bytes
    }

    /// Replace the CTRL section's response tag in place (#603 PR2's replay
    /// stage). An image is rendered once at caller-chosen tags; the submit
    /// path then stamps the tags the `ResponseWorker` actually leased onto
    /// its copy — the port of the vendor's `cbr->SpecializeTags(raw_tags)`
    /// (`PrepareCbs`, `response_worker.cpp`), which the immutable
    /// [`ControlBlockWire`] could not express, so this owns the same field
    /// through the same [`BitField`] the constructor packs it with: one
    /// derivation of "tag lives at bits 31:8", never a second.
    ///
    /// A mask-and-set, deliberately NOT an add: the canvas was rendered at
    /// arbitrary tags, so what is being replaced is a baked-in value, not a
    /// zero. The clear is IN POSITION (#655): clearing with the unshifted
    /// `BitField::MASK` (bits 23:0) did the reverse of the intent — it kept
    /// the tag field's own top byte (bits 31:24), so a canvas rendered at a
    /// high tag kept its stale top byte through every patch, and it zeroed
    /// the CTRL cancel byte (bits 7:0) that [`Self::cancelled`] stamps, so a
    /// tag-patched cancelled twin would read back un-cancelled. Everything
    /// else in word 0 — the barrier bit and the cancel byte included — is
    /// kept.
    pub fn set_tag(&mut self, tag: ResponseTag) {
        let in_position = BitField::<31, 8>::IN_POSITION_MASK;
        let mut w = [0u8; 8];
        w.copy_from_slice(&self.bytes[CTRL_SECTION..CTRL_SECTION + 8]);
        let word = u64::from_le_bytes(w);
        let word = (word & !in_position) | BitField::<31, 8>::pack(tag.0 as u64);
        self.bytes[CTRL_SECTION..CTRL_SECTION + 8].copy_from_slice(&word.to_le_bytes());
    }

    /// Set the CTRL section's error-dependency mode (`CtrlSectionSBF::edep_`,
    /// bits 34:33, four-valued EDEP0..EDEP3 — `ctrl_section_sbf.hpp:45-49`).
    ///
    /// The vendor's `ControlBlockStream::Finalize` (`control_block_stream.cpp:423`)
    /// stamps EDEP1 on the first CB of EVERY submitted stream, unconditionally —
    /// the only `SetEdep` in the vendor tree, and purely device-facing (no
    /// software path reads it back). The device consumes the field: EDEP is a
    /// named CB-kill cause in the response block's app-data "CB KILL" enum
    /// (`app_data_sbf.hpp:163-171`, alongside LATE_CANCEL/TIMEOUT/CONTEXT_FENCE),
    /// so a stream that never carries the vendor's value sits outside the
    /// error-dependency behavior every real submission exercises.
    /// [`CtrlSection::new`] packs only tag+barrier, so bits 34:33 default to
    /// EDEP0 — this method is how the first-CB stamp reaches the wire.
    ///
    /// Same mask-and-set discipline as [`Self::set_tag`], with one
    /// correction that discipline needs here: the field is cleared IN
    /// POSITION before the new value is OR'd in (`MASK` is the unshifted
    /// width mask, so `pack(MASK)` is what puts it at bits 34:33). The
    /// stamp must be able to CLEAR the field too — a re-stamp that could
    /// only set bits would leave a stale EDEP1 on every CB after the
    /// first. Everything else in word 0 — the tag, the barrier bit,
    /// `completion_interrupt_` (bit 35) — is kept.
    pub fn set_edep(&mut self, edep: u64) {
        let in_position = BitField::<34, 33>::pack(BitField::<34, 33>::MASK);
        let mut w = [0u8; 8];
        w.copy_from_slice(&self.bytes[CTRL_SECTION..CTRL_SECTION + 8]);
        let word = u64::from_le_bytes(w);
        let word = (word & !in_position) | BitField::<34, 33>::pack(edep);
        self.bytes[CTRL_SECTION..CTRL_SECTION + 8].copy_from_slice(&word.to_le_bytes());
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

    /// Port of `ControlBlockStream::CreateNullCb`
    /// (`flex/src/control_blocks/control_block_stream.cpp:375-388`, declared
    /// at `control_block_stream.hpp:252-257`, pod C++ @65ad807bb): a control
    /// block that carries no work, only ordering. The real creator allocates
    /// a fresh CB template — a default-constructed `ControlBlockSBF`, whose
    /// constructor leaves both DMA sections `SKIP` and the CMPT section
    /// invalid — and, when `pipeline_barrier` is set, sets the CTRL barrier
    /// bit and nothing else. No DMA in either slot, no compute, no XLAT
    /// entries: below the CTRL word the bytes are exactly the SBF defaults.
    ///
    /// The vendor's sole construction site is
    /// `RuntimeScheduler::buildNullControlBlocks`
    /// (`runtime_scheduler.cpp:1267-1276`): a single-CB stream over an EMPTY
    /// segment table (so `CreateCbTemplate`'s per-segment XLAT loop installs
    /// nothing), `Finalize()`d like any stream. That makes the null CB the
    /// stream's first CB, so the vendor's first-CB EDEP stamp lands on it at
    /// submission time — in this crate that stamp belongs to the submission
    /// choke (#608, PR #656), not to any constructor: `null` stays EDEP0 at
    /// build time like every other constructor here.
    ///
    /// VERIFIED CONTRACT (pod C++, issue #607):
    ///
    /// - **The barrier bit is the payload.** `pipeline_barrier` is
    ///   `CtrlSectionSBF::SetBarrier` (bit 32), which the vendor header's
    ///   own parameter doc describes as the device-wide pipeline barrier.
    ///   `CreateNullCb` is one of only three `SetBarrier` sites in the
    ///   vendor's `control_block_stream.cpp` — alongside
    ///   `CreateComputeDataTransfer`'s own `pipeline_barrier` parameter
    ///   (this crate's `dma_immediate`) and the `FLEX_FORCE_CB_BARRIER`
    ///   debug loop in `Optimize` — and the only one that can set the bit
    ///   on a CB carrying no other work: the vendor's cheapest device-side
    ///   fence.
    ///
    /// - **It consumes a ring slot and retires like any submitted CB.**
    ///   `PfRuntimeScheduler::submitNullOp` (`pf_runtime_scheduler.cpp:509-520`)
    ///   submits the single-CB stream through the same `submitToHardware`
    ///   COMPUTE-pipeline path as a compute CB, the caller's completion
    ///   callback attached; the scheduler's null-op `schedule`
    ///   (`runtime_scheduler.cpp:1025-1058`) wraps that callback in the
    ///   per-op compute fence, so the fence resolves exactly when the null
    ///   CB's own completion fires — the one-completion-signal property
    ///   #603's one-RB-per-step shape is after.
    ///
    /// - **The host-side half of the barrier already exists here.** Before
    ///   submitting a barrier-bearing null op, the vendor's `schedule` calls
    ///   `issueBarrier(Compute, stream)` (`runtime_scheduler.cpp:433-530`),
    ///   waiting out every OTHER pipeline's outstanding per-op fences for
    ///   the stream. The vendor routes null ops over `PipelineId::COMPUTE`,
    ///   so the equivalent submission in this crate is the existing
    ///   `launch_compute_wires` door (`RuntimeOperationKind::Compute`) with
    ///   the operation's `pipeline_barrier` flag set — the same
    ///   `maybe_set_barrier`/`issue_barrier` machinery every barrier-bearing
    ///   op already crosses.
    ///
    /// - **Not settled by any readable source:** the null CB's barrier
    ///   interaction with the ring (issue #607's own open question) is a
    ///   card measurement, not a source fact — it lands in #603 PR2's
    ///   cell, with a real consumer driving it.
    ///
    /// Nothing in tree calls this yet. Like `cancelled` (#611) this is a
    /// prerequisite port: #603 PR2's decode-rung player wants exactly one
    /// completion signal per step, and a trailing null CB with barrier=1 is
    /// the candidate end-of-step marker — cheaper than attaching the fence
    /// to the last real CB, and it keeps the step's real CBs bit-identical
    /// to today's.
    // The real creator takes a debug label and defaults `pipeline_barrier`
    // to false; `tag` is this crate's lease parameter, as with every other
    // constructor here.
    pub fn null(tag: ResponseTag, pipeline_barrier: bool) -> Self {
        Self::assemble(
            CtrlSection::new(tag, pipeline_barrier),
            R5Section::none(),
            DmaSection::skip(),
            DmaSection::skip(),
            ComputeSection::skip(),
            [XlatEntry::invalid(); 8],
        )
    }

    /// Port of `ControlBlockStream::CreateCancelledCb`
    /// (`flex/src/control_blocks/control_block_stream.cpp:390-398`, pod C++
    /// @ed96487): a zero-work compute CB whose CTRL word carries the cancel
    /// byte. The real creator delegates to `CreateComputeVirtual` with a
    /// null program address and an empty segment table — i.e. a CMPT section
    /// that is valid with `sync=false` and a zero bootstrap address, no DMA
    /// in either slot, and every XLAT entry left invalid — and only then
    /// sets the CTRL cancel flag. The vendor's sole construction site
    /// (`response_worker.cpp:326-328`) builds one per stream, at
    /// construction, as the shutdown wake-up marker.
    ///
    /// VERIFIED CONTRACT (pod C++, issue #611): a cancelled CB carries no
    /// work by construction, consumes its ring slot, and still produces an
    /// RB — the vendor's shutdown path queues it and waits on its
    /// completion (with a 5 s timeout for hardware stuck after a page
    /// fault). Two boundaries on what a design may lean on, both from the
    /// vendor's own sources:
    ///
    /// (a) the device skipping the WORK of a cancelled CB that carries ops
    ///     is NOT established — the vendor's mock models the skip at every
    ///     unit but carries its own FIXME at `mock_compute_pipeline.cpp:62`
    ///     (the mock still lets the CB pass through each pipeline stage),
    ///     and no readable source ever constructs a work-carrying cancelled
    ///     CB. Callers must swap in this zero-work form rather than
    ///     cancelling a CB that already has work in it.
    ///
    /// (b) the vendor runtime treats a cancel RB as an ERROR on the normal
    ///     queuing path (`EvalResponseBlockStatus`,
    ///     `response_worker.cpp:128-135`) — only `QueuingMode::STOP_THREADS`
    ///     accepts it as the graceful-shutdown signal. A caller using this
    ///     for overrun discard therefore defines its own completion policy;
    ///     the mock-port side of the skip is
    ///     [`crate::control_blocks::process_crb`]'s `Cancelled` outcome.
    // The real creator takes no arguments (the vendor leases tags at its
    // stream layer); `tag` is this crate's lease parameter, as with every
    // other constructor here.
    pub fn cancelled(tag: ResponseTag) -> Self {
        Self::assemble(
            CtrlSection::new(tag, false).with_cancel(),
            R5Section::none(),
            DmaSection::skip(),
            DmaSection::skip(),
            ComputeSection::new(VirtualAddressSbf::new(0, Flits(0)), false),
            [XlatEntry::invalid(); 8],
        )
    }

    /// Raw bytes ready to hand to `senlib_ffi::SenlibControlBlockSbf`
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

    /// This CB's own `CtrlSectionSBF::GetCancel` (bits 7:0 of the CTRL
    /// word) — true for a CB built by [`ControlBlockWire::cancelled`], false
    /// for every work-carrying constructor. This is the flag
    /// [`crate::control_blocks::process_crb`] consumes as `cancelled`: the
    /// mock-port's early-return branch, which skips processing entirely and
    /// reports the RB cancel bit instead of a status byte.
    pub fn ctrl_cancelled(&self) -> bool {
        let mut word = [0u8; 8];
        word.copy_from_slice(&self.bytes[CTRL_SECTION..CTRL_SECTION + 8]);
        BitField::<7, 0>::unpack(u64::from_le_bytes(word)) != 0
    }

    /// This CB's own `CtrlSectionSBF::edep_` (bits 34:33 of the CTRL word,
    /// same encoding [`Self::set_edep`] stamps) — the error-dependency mode.
    /// EDEP0 on every CB the constructors build; EDEP1 on the first CB of a
    /// submitted stream, where `ResponseWorker::prepare_cbs` (the `Finalize`
    /// port) and the replay image derivation both stamp it.
    pub fn edep(&self) -> u64 {
        let mut word = [0u8; 8];
        word.copy_from_slice(&self.bytes[CTRL_SECTION..CTRL_SECTION + 8]);
        BitField::<34, 33>::unpack(u64::from_le_bytes(word))
    }

    /// This CB's own `CtrlSectionSBF::GetBarrier` (bit 32 of the CTRL word)
    /// — the device-wide pipeline barrier flag, and the only work a
    /// [`ControlBlockWire::null`] CB carries. Set by that constructor (the
    /// `CreateNullCb` port, one of the vendor's three `SetBarrier` sites)
    /// and by [`ControlBlockWire::dma_immediate`] when its
    /// `pipeline_barrier` parameter is true (the `CreateComputeDataTransfer`
    /// port, the vendor's other `SetBarrier` site on a real CB); clear on
    /// every other constructor.
    pub fn ctrl_barrier(&self) -> bool {
        let mut word = [0u8; 8];
        word.copy_from_slice(&self.bytes[CTRL_SECTION..CTRL_SECTION + 8]);
        BitField::<32, 32>::unpack(u64::from_le_bytes(word)) != 0
    }
}

/// A decoded response block: the real bytes senlib hands back via
/// `senlib_ffi::rbi_receive_responses_sbf`. Port of the
/// `ReturnSectionSBF` decode used by `ResponseWorker`/`DmaControlBlockHandler`
/// to determine per-CB completion status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResponseBlockWire {
    ret_word: u64,
    /// The TIMESTAMP section, `TS_SECTION..PW_SECTION` (bytes 8..28) as five little-endian 32-bit
    /// words — event snapshots of one wrapping cycle counter, never durations. See the
    /// [`Self::dmai_cycles`] family for the card-measured semantics.
    ts_words: [u32; 5],
    /// The POWER-MEASUREMENT word at `PW_SECTION..APP_SECTION` (bytes 28..32) — the vendor's
    /// `ResponseBlockWrapper::power()`.
    power_word: u32,
    /// The APPLICATION-SPECIFIC section, `APP_SECTION..RB_NUM_BYTES` (bytes 32..64) as four
    /// little-endian 64-bit BEATS — kept verbatim rather than decoded here because the section is a
    /// UNION whose meaning depends on `locator()`. [`Self::app_qgi`] reads it as the QGI/HMI variant,
    /// which is the one that carries a faulting device address. See [`AppQgiSection`].
    app_beats: [u64; 4],
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
    pub const ZERO: Self = Self {
        ret_word: 0,
        ts_words: [0; 5],
        power_word: 0,
        app_beats: [0; 4],
    };

    /// Decode from the raw 64-byte RB: the `ReturnSectionSBF` (bytes 0..8, `RET_SECTION`) that decides
    /// completion; the timestamp section (bytes 8..28) as five u32 cycle counts; the power word
    /// (bytes 28..32); and the APP section (bytes 32..64) held as raw beats for [`Self::app_qgi`].
    pub fn from_bytes(bytes: &[u8; RB_NUM_BYTES]) -> Self {
        let mut word = [0u8; 8];
        word.copy_from_slice(&bytes[0..8]);
        // `TS_SECTION = 8` and five 4-byte `TimeStampSBF`s in wire order (`time_stamp_sbf.hpp`:
        // `TimeStampSectionSBF = std::array<TimeStampSBF, 5>`, each a single 31:0 bitfield over a
        // u32 — no sub-field packing to undo).
        let mut ts_words = [0u32; 5];
        for (i, ts) in ts_words.iter_mut().enumerate() {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[TS_SECTION + i * 4..TS_SECTION + i * 4 + 4]);
            *ts = u32::from_le_bytes(b);
        }
        // `PW_SECTION = 28`, one 31:0 bitfield over a u32 (`power_measurement_section_sbf.hpp`).
        let mut pw = [0u8; 4];
        pw.copy_from_slice(&bytes[PW_SECTION..PW_SECTION + 4]);
        // `APP_SECTION = 32`, `RB_NUM_BYTES = 64` (`response_block_sbf.hpp:36-39`), and the QGI
        // variant is a `BEGIN_SAFE_BITFIELD_UNION(QGIBeatUnion, 4, std::uint64_t)` — four u64 beats,
        // beat `i` at bits `64i..64i+63`, which the header's own comments confirm ("Beat 2: QGI" for
        // 191:158, "Beat 1: HMI" for 119:112..69:64, "Beat 0" for 63:56..15:0).
        let mut app_beats = [0u64; 4];
        for (i, beat) in app_beats.iter_mut().enumerate() {
            let mut b = [0u8; 8];
            b.copy_from_slice(&bytes[APP_SECTION + i * 8..APP_SECTION + i * 8 + 8]);
            *beat = u64::from_le_bytes(b);
        }
        Self {
            ret_word: u64::from_le_bytes(word),
            ts_words,
            power_word: u32::from_le_bytes(pw),
            app_beats,
        }
    }

    /// The TIMESTAMP section verbatim: five raw 32-bit device cycle counts, `TS1..TS5` in wire
    /// order (`TimeStampSectionSBF`; libaiupti's activity enum calls them `CYCLES_TS1..TS5`
    /// "cycle count events"). They are event snapshots of one free-running wrapping counter, not
    /// durations — see the [`Self::dmai_cycles`] family for the card-measured semantics, and
    /// never sum these raw.
    pub const fn timestamps(self) -> [u32; 5] {
        self.ts_words
    }

    /// ⭐ THE TS SLOTS' CARD-MEASURED SEMANTICS (dd2, granite-3.1-2b fp8, PR #523's on-card run):
    /// the five words are snapshots of ONE free-running u32 cycle counter (rate ≈
    /// `CYCLES_PER_USEC` — 1012 cyc/µs cross-checked on card against the phase-line wall clock),
    /// taken at `word[0]` = CB start / DMA-in start, `word[1]` = DMA-in end (**exactly equal to
    /// `word[0]` on every CB with no DMA-in — every compute CB**), `word[2]` = CMPT start,
    /// `word[3]` = CMPT end, `word[4]` = CB end. Slots a CB does not write read zero: an H2D CB
    /// zeroes words 3-4, a D2H CB zeroes words 0-2. A phase duration is therefore the WRAPPING
    /// difference of its pair — the counter wraps every ~4.2 s, and a per-CB duration (≤ ~1M
    /// cycles) is far under the wrap, so `wrapping_sub` is exact. The vendor's
    /// `ResponseBlockWrapper` names these same five phases (`dmai_cycles()` … `total_cmpt_cycles()`)
    /// but ships no `.cpp`, so these accessors carry the measurement, not a transcription.
    ///
    /// On a D2H CB, `dmai_cycles`/`prep_cmpt_cycles`/`exec_cmpt_cycles` difference zeroed slots
    /// and are meaningless; on a compute CB, `dmao_cycles` is the ~9-cycle writeback tail. Use
    /// the pair that matches the CB's pipeline.
    pub const fn dmai_cycles(self) -> u32 {
        self.ts_words[1].wrapping_sub(self.ts_words[0])
    }

    /// The CMPT queue+prep wait: `word[2] - word[1]`, from DMA-in end to the compute engine
    /// picking the CB up. On a pipelined stream this includes waiting behind the previous CB's
    /// execution — consecutive compute CBs overlap exactly this way (the next CB's prep runs
    /// during the previous CB's exec). See [`Self::dmai_cycles`] for the full measured layout.
    pub const fn prep_cmpt_cycles(self) -> u32 {
        self.ts_words[2].wrapping_sub(self.ts_words[1])
    }

    /// The compute engine's execution: `word[3] - word[2]`. The number that sums to the device
    /// span of a forward (cross-checked on card: Σ exec over a forward's CBs ≈ its
    /// `[sdsc-phase] compute` wall time). See [`Self::dmai_cycles`].
    pub const fn exec_cmpt_cycles(self) -> u32 {
        self.ts_words[3].wrapping_sub(self.ts_words[2])
    }

    /// The CB's last phase: `word[4] - word[3]` — the D2H duration on a D2H CB, the ~9-cycle
    /// writeback tail on a compute CB. See [`Self::dmai_cycles`].
    pub const fn dmao_cycles(self) -> u32 {
        self.ts_words[4].wrapping_sub(self.ts_words[3])
    }

    /// The CB's whole CMPT residency: `word[4] - word[1]` = prep + exec + tail (== the full CB
    /// span on compute CBs, where `word[1] == word[0]`). See [`Self::dmai_cycles`].
    pub const fn total_cmpt_cycles(self) -> u32 {
        self.ts_words[4].wrapping_sub(self.ts_words[1])
    }

    /// The POWER-MEASUREMENT word (`ResponseBlockWrapper::power()`).
    pub const fn power(self) -> u32 {
        self.power_word
    }

    /// ⭐ THE APP SECTION READ AS THE QGI/HMI VARIANT — the faulting device address of a compute-side
    /// fault, which is otherwise thrown away.
    ///
    /// `Some` exactly when the vendor's stated validity condition holds: `status == RB_ERROR` and
    /// `locator` is `QGI_DIAG` or `QGI_NO_DIAG` (`app_data_sbf.hpp:299-305` — "This section is valid
    /// when rb.ret().GetStatus() == RB_ERROR and rb.ret().GetLocator() == QGI || ... == QGI_NO_DIAG").
    ///
    /// ⛔ `QGI_NO_DIAG` (`0x2`) DOES NOT MEAN THE SYNDROME IS ABSENT. It means the CB was submitted
    /// with `extended_diagnostic_` clear, which is what `CreateComputePhysical` and our own
    /// `ComputeSection::new` both do. The syndrome beats are still written. A fault reported only as
    /// `locator=0x2` is therefore a fault whose address we HAD and discarded.
    pub const fn app_qgi(self) -> Option<AppQgiSection> {
        match (self.status(), self.locator_bytes()) {
            (ResponseStatus::Error, LOCATOR_QGI_DIAG | LOCATOR_QGI_NO_DIAG) => {
                Some(AppQgiSection {
                    beats: self.app_beats,
                })
            }
            _ => None,
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

/// `RBLocatorTypeEnum::QGI_DIAG` (`return_section_sbf.hpp:26-34`).
pub const LOCATOR_QGI_DIAG: u8 = 0x3;
/// `RBLocatorTypeEnum::QGI_NO_DIAG` (`return_section_sbf.hpp:26-34`) — the locator our compute faults
/// report, because compute CBs are submitted with extended diagnostics off.
pub const LOCATOR_QGI_NO_DIAG: u8 = 0x2;

/// A flit is 128 B (`sys-arch-spec/sysdef.cpp:206 bytesPerStick`), which is the unit BOTH QGI and HMI
/// report their addresses in. Corroborated by the field widths: `qgi_addr_` is 30 bits of flits =
/// exactly the 128 GiB `DMVA_SIZE` (`control_block_stream.cpp:58`).
pub const FLIT_BYTES: u64 = 128;

/// ⭐ THE QGI/HMI SECTION OF A RESPONSE BLOCK — what turns `locator=0x2` from an opaque fault into a
/// faulting DEVICE ADDRESS and a named cause.
///
/// Port of `SentientSoc::V1::AppQGISection` (`senlib/include/hal/1p0/app_data_sbf.hpp:309-431`). Four
/// `u64` beats; every bit range below is the header's, and the header labels the beats itself. Two
/// independent error classes share the section, each with its own validity flag:
///
/// * **HMI** — a memory-access error. Valid when [`Self::hmi_syndrome`] is non-zero.
/// * **QGI** — an RCU error. Valid when [`Self::qgi_syndrome`] is non-zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppQgiSection {
    beats: [u64; 4],
}

impl AppQgiSection {
    /// `job_count_` (bits 15:0, beat 0).
    pub const fn job_count(self) -> u64 {
        BitField::<15, 0>::unpack(self.beats[0])
    }

    /// `logout_resp_` (bits 23:16, beat 0).
    pub const fn logout_response(self) -> u64 {
        BitField::<23, 16>::unpack(self.beats[0])
    }

    /// `beat0_byte8_` (bits 63:56, beat 0) — documented UNUSED, and both `isQGI()` and `isHMI()`
    /// require it to be ZERO, so a non-zero value means this is not really a QGI/HMI section.
    pub const fn beat0_byte8(self) -> u64 {
        BitField::<63, 56>::unpack(self.beats[0])
    }

    /// `hmi_snid_` (bits 69:64 ⇒ beat 1, local 5:0).
    pub const fn hmi_snid(self) -> u64 {
        BitField::<5, 0>::unpack(self.beats[1])
    }

    /// `hmi_mode_` (bit 70 ⇒ beat 1, local 6). `false` = FETCH, `true` = STORE — the header's own
    /// "0-fetch 1-store". Which side faulted is the first thing you want: a store fault past a
    /// translation's `length` is a write running off the end of a region.
    pub const fn hmi_is_store(self) -> bool {
        BitField::<6, 6>::unpack(self.beats[1]) != 0
    }

    /// `hmi_addr_` (bits 101:71 ⇒ beat 1, local 37:7), in FLITS.
    pub const fn hmi_address_flits(self) -> u64 {
        BitField::<37, 7>::unpack(self.beats[1])
    }

    /// The HMI faulting address in BYTES — `hmi_address_flits × FLIT_BYTES`. This is the number to
    /// compare against a segment's base and declared extent.
    pub const fn hmi_address_bytes(self) -> u64 {
        self.hmi_address_flits() * FLIT_BYTES
    }

    /// `hmi_synd_` (bits 111:104 ⇒ beat 1, local 47:40). Non-zero ⇒ the HMI fields are valid.
    pub const fn hmi_syndrome(self) -> u64 {
        BitField::<47, 40>::unpack(self.beats[1])
    }

    /// `hmi_tag_` (bits 119:112 ⇒ beat 1, local 55:48).
    pub const fn hmi_tag(self) -> u64 {
        BitField::<55, 48>::unpack(self.beats[1])
    }

    /// `qgi_addr_` (bits 157:128 ⇒ beat 2, local 29:0), in FLITS.
    pub const fn qgi_address_flits(self) -> u64 {
        BitField::<29, 0>::unpack(self.beats[2])
    }

    /// The QGI faulting address in BYTES.
    pub const fn qgi_address_bytes(self) -> u64 {
        self.qgi_address_flits() * FLIT_BYTES
    }

    /// `qgi_synd_` (bits 191:158 ⇒ beat 2, local 63:30) — `NUM_QGI_SYND_BITS = 34`. Non-zero ⇒ the QGI
    /// fields are valid. A BITMASK, not an ordinal: bit `n` set means [`QgiErrorCase`] `n` fired.
    pub const fn qgi_syndrome(self) -> u64 {
        BitField::<63, 30>::unpack(self.beats[2])
    }

    /// `isHMI()` — a memory-access error is described here.
    pub const fn is_hmi(self) -> bool {
        self.hmi_syndrome() != 0 && self.beat0_byte8() == 0
    }

    /// `isQGI()` — an RCU error is described here.
    pub const fn is_qgi(self) -> bool {
        self.qgi_syndrome() != 0 && self.beat0_byte8() == 0
    }

    /// Every [`QgiErrorCase`] whose bit is set in [`Self::qgi_syndrome`], lowest bit first.
    pub fn qgi_error_cases(self) -> impl Iterator<Item = QgiErrorCase> {
        let synd = self.qgi_syndrome();
        (0u32..QGI_SYNDROME_BITS)
            .filter(move |b| synd & (1u64 << b) != 0)
            .map(|b| QgiErrorCase::from_bit(b as u8))
    }
}

/// `NUM_QGI_SYND_BITS` (`app_data_sbf.hpp:308`), and the width of `qgi_synd_` (191:158).
pub const QGI_SYNDROME_BITS: u32 = 34;

/// `AppQGISection::ErrorCaseTypes` (`app_data_sbf.hpp:336-372`), transcribed verbatim. These are BIT
/// POSITIONS within [`AppQgiSection::qgi_syndrome`], not values — 34 cases for 34 syndrome bits.
///
/// ⭐ The two that name an address the hardware could not translate are [`Self::RiuUnmpErr`] (19) and
/// [`Self::PrepUnmpErr`] (20) — i.e. an access outside the `paddr..paddr+length` a translation
/// declared. Those are the ones to look for when a segment's declared extent is suspected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QgiErrorCase {
    SocIoErr,
    AbortHang,
    LocalRing,
    CoreSwFault,
    CoreHwFault,
    RiuSyncErr,
    PrepTagCorr,
    PrepCntCorr,
    PrepHeadFlitCnt,
    PrepJobFlitCnt,
    PrepZeroFlitCnt,
    PrepSwVer,
    PrepDecision,
    ExcNoJob,
    HmiErrNoPrep,
    SlhKillMiss,
    HmiErrNoExec,
    PrepDaCap,
    RiuSlfrErr,
    RiuUnmpErr,
    PrepUnmpErr,
    PrepRqfcCorr,
    PrepAddrPerr,
    PrepBufWrap,
    PrepSttqErr,
    PrepSttqCorr,
    PrepLsmCorr,
    PrepUsmCorr,
    ExecEsmCorr,
    RiuDataPerr,
    HmiErrPerr,
    SlhReqPerr,
    PrepBuffPerr,
    LogoBuffPerr,
    /// A syndrome bit at or above [`QGI_SYNDROME_BITS`], or any position the header does not name —
    /// kept explicit because this decodes bytes written by hardware.
    Unknown(u8),
}

impl QgiErrorCase {
    /// The case at syndrome bit `bit`.
    pub const fn from_bit(bit: u8) -> Self {
        match bit {
            0 => Self::SocIoErr,
            1 => Self::AbortHang,
            2 => Self::LocalRing,
            3 => Self::CoreSwFault,
            4 => Self::CoreHwFault,
            5 => Self::RiuSyncErr,
            6 => Self::PrepTagCorr,
            7 => Self::PrepCntCorr,
            8 => Self::PrepHeadFlitCnt,
            9 => Self::PrepJobFlitCnt,
            10 => Self::PrepZeroFlitCnt,
            11 => Self::PrepSwVer,
            12 => Self::PrepDecision,
            13 => Self::ExcNoJob,
            14 => Self::HmiErrNoPrep,
            15 => Self::SlhKillMiss,
            16 => Self::HmiErrNoExec,
            17 => Self::PrepDaCap,
            18 => Self::RiuSlfrErr,
            19 => Self::RiuUnmpErr,
            20 => Self::PrepUnmpErr,
            21 => Self::PrepRqfcCorr,
            22 => Self::PrepAddrPerr,
            23 => Self::PrepBufWrap,
            24 => Self::PrepSttqErr,
            25 => Self::PrepSttqCorr,
            26 => Self::PrepLsmCorr,
            27 => Self::PrepUsmCorr,
            28 => Self::ExecEsmCorr,
            29 => Self::RiuDataPerr,
            30 => Self::HmiErrPerr,
            31 => Self::SlhReqPerr,
            32 => Self::PrepBuffPerr,
            33 => Self::LogoBuffPerr,
            other => Self::Unknown(other),
        }
    }

    /// True for the two cases that mean "the hardware could not translate this address".
    pub const fn is_unmapped_address(self) -> bool {
        matches!(self, Self::RiuUnmpErr | Self::PrepUnmpErr)
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
        // The response block's own section static asserts (`response_block_sbf.hpp`): RET is 8
        // bytes, TS is five 4-byte stamps, PW is one 4-byte word, APP runs to the end.
        assert_eq!(TS_SECTION, 8);
        assert_eq!(TS_SECTION + 5 * 4, PW_SECTION);
        assert_eq!(PW_SECTION + 4, APP_SECTION);
    }

    #[test]
    fn virtual_address_round_trips_segment_and_offset() {
        let va = VirtualAddressSbf::new(7, Flits::from_bytes(1 << 20));
        assert_eq!(va.segment(), 7);
        assert_eq!(va.offset().to_bytes(), 1 << 20);
    }

    /// #608: EDEP occupies CTRL bits 34:33 exactly, and stamping it moves
    /// nothing else — the encoding-level half of the first-CB rule. The
    /// constructors pack only tag+barrier, so a freshly built CB is EDEP0:
    /// before #608 every CB we submitted carried EDEP0 where the vendor's
    /// `Finalize` (`control_block_stream.cpp:423`) stamps EDEP1 on the first
    /// CB of each stream.
    #[test]
    fn edep_occupies_ctrl_bits_34_33_and_leaves_neighbors_intact() {
        let dmva = VirtualAddressSbf::new(3, Flits::from_bytes(0));
        let mut cb = ControlBlockWire::fill(
            ResponseTag(0xABC),
            dmva,
            TransferSize(4096),
            0xDEAD_BEEF,
            true,
            DevicePhysicalAddress(0x3_0000_0000),
            SegmentExtent(4096),
        )
        .unwrap();
        let word0 = |cb: &ControlBlockWire| {
            u64::from_le_bytes(
                cb.to_bytes()[CTRL_SECTION..CTRL_SECTION + 8]
                    .try_into()
                    .unwrap(),
            )
        };

        let before = word0(&cb);
        assert_eq!(cb.edep(), 0, "constructors leave EDEP0 (bits 34:33 zero)");
        assert_eq!(before >> 33 & 0b11, 0);

        cb.set_edep(1);
        let stamped = word0(&cb);
        assert_eq!(cb.edep(), 1);
        // EXACT bit pattern, asserted as literal bits (not through the same
        // BitField the setter uses): bit 33 set, bit 34 clear — EDEP1, not
        // some other neighbor.
        assert_eq!(stamped >> 33 & 0b11, 0b01);
        // The mask-and-set kept every other bit of word 0: tag (31:8),
        // barrier (32), completion_interrupt (35) included.
        assert_eq!(stamped & !(0b11u64 << 33), before);

        // The replay-image agreement case, both directions: re-stamping the
        // same value is a no-op (derivation and submit path both stamp
        // EDEP1), and the mask-and-set can CLEAR — the reverse of the
        // stale-image case — so a stamp can never leave a stale EDEP1
        // behind on a non-first CB.
        cb.set_edep(1);
        assert_eq!(word0(&cb), stamped);
        cb.set_edep(0);
        assert_eq!(cb.edep(), 0);
        assert_eq!(word0(&cb), before);
    }

    /// #611 + #608 in ONE CTRL word: the cancel byte (`cancel_`, bits 7:0,
    /// stamped by the `cancelled` constructor) and EDEP (`edep_`, bits
    /// 34:33, stamped by the first-CB rule) are disjoint fields of the same
    /// 64-bit word, written by independent setters. This is the exact CTRL
    /// word a #611 cancelled twin occupying a stream's first-CB slot would
    /// carry across #608's `prepare_cbs` stamp — the combined-field case
    /// #603 PR3b's overrun discard is shaped like — so both fields must
    /// read back and each setter's mask-and-set must leave the other's
    /// field untouched.
    #[test]
    fn cancel_byte_and_edep_coexist_in_one_ctrl_word() {
        // The cancel constructor alone leaves EDEP0 — neither side's
        // constructor sets the other's field.
        let mut cb = ControlBlockWire::cancelled(ResponseTag(0xABC));
        assert_eq!(cb.edep(), 0, "the cancelled constructor stamps no EDEP");
        assert!(cb.ctrl_cancelled());

        // BOTH stamps on one CB: the constructor's cancel byte, then the
        // first-CB rule's EDEP1.
        cb.set_edep(1);

        // BOTH read back through their own readers.
        assert!(
            cb.ctrl_cancelled(),
            "the cancel byte survives the EDEP stamp"
        );
        assert_eq!(cb.edep(), 1, "the EDEP stamp survives on a cancelled CB");

        // And on the wire as literal bits (not through either field's own
        // BitField): cancel byte exactly 1 at bits 7:0, EDEP exactly 0b01
        // at bits 34:33, tag preserved at bits 31:8, barrier clear at 32.
        let ctrl = u64::from_le_bytes(
            cb.to_bytes()[CTRL_SECTION..CTRL_SECTION + 8]
                .try_into()
                .unwrap(),
        );
        assert_eq!(ctrl & 0xFF, 1, "cancel_ (bits 7:0) is exactly 1");
        assert_eq!(
            ctrl >> 33 & 0b11,
            0b01,
            "edep_ (bits 34:33) is exactly EDEP1"
        );
        assert_eq!((ctrl >> 8) & 0xFF_FFFF, 0xABC, "tag survives both stamps");
        assert_eq!(ctrl >> 32 & 1, 0, "barrier stays clear");

        // Disjointness in the clear direction too: `set_edep`'s in-position
        // clear (back to EDEP0) must not disturb the cancel byte.
        cb.set_edep(0);
        assert_eq!(cb.edep(), 0);
        assert!(
            cb.ctrl_cancelled(),
            "clearing EDEP leaves the cancel byte set — the fields are disjoint"
        );
        let ctrl_after = u64::from_le_bytes(
            cb.to_bytes()[CTRL_SECTION..CTRL_SECTION + 8]
                .try_into()
                .unwrap(),
        );
        assert_eq!(ctrl_after & !(0b11u64 << 33), ctrl & !(0b11u64 << 33));
        assert_eq!(ctrl_after & 0xFF, 1);
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

    /// The wire facts of a `CreateCancelledCb` port: the CTRL cancel byte is
    /// set (and only that byte — tag preserved, barrier clear), the CMPT
    /// section is a valid-but-unsynced zero-bootstrap compute section, both
    /// DMA slots are SKIP, and every XLAT entry is invalid. These are the
    /// bytes the vendor's shutdown marker puts on the ring, pinned so the
    /// port cannot silently drift from the verified shape.
    #[test]
    fn cancelled_cb_ports_the_vendor_shutdown_marker_shape() {
        let cb = ControlBlockWire::cancelled(ResponseTag(7));
        let bytes = cb.to_bytes();

        // CTRL word: cancel_ (bits 7:0) == 1 exactly — the vendor's setter
        // writes the boolean, not an arbitrary byte; tag at bits 31:8
        // preserved for RB correlation; barrier (bit 32) clear.
        let ctrl = u64::from_le_bytes(bytes[CTRL_SECTION..CTRL_SECTION + 8].try_into().unwrap());
        assert_eq!(ctrl & 0xFF, 1, "cancel_ (bits 7:0) must be exactly 1");
        assert_eq!(
            (ctrl >> 8) & 0xFF_FFFF,
            7,
            "tag must survive the cancel set"
        );
        assert_eq!((ctrl >> 32) & 1, 0, "barrier must be clear");
        assert!(cb.ctrl_cancelled());

        // CMPT word0: valid_ set (CreateComputeVirtual sets it
        // unconditionally), sync_ clear, bootstrap_address_ (bits 36:7)
        // zero — the null program address.
        let cmpt_word0 =
            u64::from_le_bytes(bytes[CMPT_SECTION..CMPT_SECTION + 8].try_into().unwrap());
        assert_ne!(cmpt_word0 & (1 << 63), 0, "valid_ must be set");
        assert_eq!(cmpt_word0 & (1 << 62), 0, "sync_ must be clear");
        assert_eq!(
            cmpt_word0 & BitField::<36, 7>::IN_POSITION_MASK,
            0,
            "bootstrap must be null"
        );

        // Both DMA slots SKIP — the marker carries no DMA work.
        for section in [DMI_SECTION, DMO_SECTION] {
            let word1 = u64::from_le_bytes(bytes[section + 8..section + 16].try_into().unwrap());
            assert_eq!(
                word1 & 0xF,
                DmaType::Skip.bits(),
                "both DMA slots must be SKIP"
            );
        }

        // Every XLAT entry invalid — the vendor's sole construction site
        // builds it on a stream with an empty segment table.
        for i in 0..8 {
            let xlat = u64::from_le_bytes(
                bytes[XLAT_SECTION + i * 8..XLAT_SECTION + i * 8 + 8]
                    .try_into()
                    .unwrap(),
            );
            assert_eq!(xlat, XlatEntry::invalid().word, "xlat[{i}] must be invalid");
        }
    }

    /// Negative control: only the cancelled constructor sets the CTRL cancel
    /// byte — a work-carrying CB must never read back cancelled, or the
    /// discard mechanism would drop live work.
    #[test]
    fn only_a_cancelled_cb_reads_cancelled() {
        let bootstrap = VirtualAddressSbf::new(7, Flits::from_bytes(0));
        let prog = (
            DevicePhysicalAddress(0x1_0000_0000),
            AllocationTotalSize(4096),
        );
        let tensors = [(
            DevicePhysicalAddress(0x2_0000_0000),
            AllocationTotalSize(65536),
        )];
        let compute = ControlBlockWire::compute(ResponseTag(1), &tensors, prog, bootstrap).unwrap();
        assert!(!compute.ctrl_cancelled());

        let dmva = VirtualAddressSbf::new(2, Flits::from_bytes(0x1000));
        let dma = ControlBlockWire::dma_immediate(
            ResponseTag(2),
            true,
            DmaTransfer {
                dmva,
                host_addr: HostPhysicalAddress::from_raw_for_tests(0x7f00_0000),
                size_bytes: TransferSize(256),
            },
            XlatWindow {
                region_paddr: DevicePhysicalAddress(0x2_0000_0000),
                length: SegmentExtent(0x1000 + 256),
            },
            false,
        )
        .unwrap();
        assert!(!dma.ctrl_cancelled());

        let fill = ControlBlockWire::fill(
            ResponseTag(3),
            dmva,
            TransferSize(4096),
            0xDEAD_BEEF,
            true,
            DevicePhysicalAddress(0x3_0000_0000),
            SegmentExtent(4096),
        )
        .unwrap();
        assert!(!fill.ctrl_cancelled());
    }

    /// Regression test for #655: `set_tag`'s clear must be the tag field's
    /// IN-POSITION mask (CTRL bits 31:8), not the unshifted width mask
    /// (bits 23:0). The unshifted mask failed in BOTH directions at once —
    /// it KEPT the tag field's own top byte (bits 31:24) and CLEARED the
    /// CTRL cancel byte (bits 7:0, #611's `cancelled` stamp):
    ///
    /// - canvas-at-high-tag → small-tag patch: the stale top byte survived
    ///   every patch, so the patched CB correlated to a tag the
    ///   `ResponseWorker` never leased;
    /// - tag-patching a cancelled CB silently un-cancelled it — the forward
    ///   hazard #655 was filed over (PR3b's overrun-discard twin): a
    ///   tag-patched cancelled twin would read back un-cancelled, and
    ///   `process_crb` would route it as a normal compute CB instead of
    ///   `Cancelled`.
    ///
    /// Literal bit patterns, not the `BitField` the setter uses — a reader
    /// bug must not be able to mask a writer bug.
    #[test]
    fn set_tag_clears_in_position_keeping_ctrl_neighbors() {
        let word0 = |cb: &ControlBlockWire| {
            u64::from_le_bytes(
                cb.to_bytes()[CTRL_SECTION..CTRL_SECTION + 8]
                    .try_into()
                    .unwrap(),
            )
        };

        // Direction 1 — the tag field's own top byte must CLEAR: a CB
        // patched to a tag with bits 31:24 set, then patched to a small
        // tag, must read back as exactly the small tag.
        let dmva = VirtualAddressSbf::new(2, Flits::from_bytes(0x1000));
        let mut cb = ControlBlockWire::dma_immediate(
            ResponseTag(0x11),
            true,
            DmaTransfer {
                dmva,
                host_addr: HostPhysicalAddress::from_raw_for_tests(0x7f00_0000),
                size_bytes: TransferSize(256),
            },
            XlatWindow {
                region_paddr: DevicePhysicalAddress(0x2_0000_0000),
                length: SegmentExtent(0x1000 + 256),
            },
            true, // barrier — must survive the patch
        )
        .unwrap();
        cb.set_tag(ResponseTag(0xAB_CDEF)); // top byte 0xAB into bits 31:24
        assert_eq!(
            (word0(&cb) >> 8) & 0xFF_FFFF,
            0xAB_CDEF,
            "the high tag must land in the field first"
        );
        cb.set_tag(ResponseTag(0x42));
        assert_eq!(
            (word0(&cb) >> 8) & 0xFF_FFFF,
            0x42,
            "the whole 24-bit tag field must be replaced, top byte included"
        );
        assert_eq!(
            (word0(&cb) >> 24) & 0xFF,
            0,
            "no stale high tag byte may survive the patch"
        );
        assert_eq!(
            (word0(&cb) >> 32) & 1,
            1,
            "barrier bit must survive the tag patch"
        );
        assert_eq!(cb.tag(), ResponseTag(0x42));

        // Direction 2 — the CTRL cancel byte must SURVIVE a tag patch: a
        // tag-patched cancelled CB must still read cancelled.
        let mut cancelled = ControlBlockWire::cancelled(ResponseTag(7));
        cancelled.set_tag(ResponseTag(9));
        assert_eq!(
            (word0(&cancelled) >> 8) & 0xFF_FFFF,
            9,
            "the patch must land"
        );
        assert_eq!(
            word0(&cancelled) & 0xFF,
            1,
            "#655: the tag patch must not eat the cancel byte"
        );
        assert!(
            cancelled.ctrl_cancelled(),
            "#655: a tag-patched cancelled CB must still read cancelled"
        );
    }

    /// The wire facts of a `CreateNullCb` port (`control_block_stream.cpp:375-388`):
    /// the CTRL word carries the tag and NOTHING else — no cancel byte, no
    /// barrier, no EDEP, no timers — and every section below it is exactly the
    /// `ControlBlockSBF` constructor default: R5 zero, both DMA slots SKIP with
    /// zero payloads, CMPT invalid (all-zero — the creator never touches it),
    /// every XLAT entry invalid (the vendor's sole construction site builds the
    /// stream over an empty segment table). A CB that is "no work, only
    /// ordering" must be indistinguishable from a freshly default-constructed
    /// one below the CTRL word, or it would be carrying work after all.
    #[test]
    fn null_cb_carries_no_work_only_the_ctrl_word() {
        let cb = ControlBlockWire::null(ResponseTag(0x0BAD), false);
        let bytes = cb.to_bytes();

        // CTRL word: tag at bits 31:8 and nothing else.
        let ctrl = u64::from_le_bytes(bytes[CTRL_SECTION..CTRL_SECTION + 8].try_into().unwrap());
        assert_eq!(
            ctrl,
            0x0BADu64 << 8,
            "CTRL must be the tag and nothing else"
        );
        assert_eq!(cb.tag(), ResponseTag(0x0BAD));
        assert!(!cb.ctrl_cancelled());
        assert!(!cb.ctrl_barrier());

        // R5: zero.
        assert_eq!(
            u64::from_le_bytes(bytes[R5_SECTION..R5_SECTION + 8].try_into().unwrap()),
            R5Section::none().to_word()
        );

        // Both DMA slots: SKIP with zero payloads — the SBF constructor default.
        for section in [DMI_SECTION, DMO_SECTION] {
            assert_eq!(
                u64::from_le_bytes(bytes[section..section + 8].try_into().unwrap()),
                0,
                "DMA word0 must be zero"
            );
            assert_eq!(
                u64::from_le_bytes(bytes[section + 8..section + 16].try_into().unwrap()),
                DmaType::Skip.bits(),
                "DMA word1 must be exactly the SKIP cbtype"
            );
        }

        // CMPT: invalid — all-zero. This is what separates the null CB from
        // the cancelled marker (#611), whose CMPT is a VALID compute section.
        for i in 0..2 {
            assert_eq!(
                u64::from_le_bytes(
                    bytes[CMPT_SECTION + i * 8..CMPT_SECTION + i * 8 + 8]
                        .try_into()
                        .unwrap()
                ),
                0,
                "CMPT must be all-zero (invalid)"
            );
        }

        // Every XLAT entry invalid.
        for i in 0..8 {
            assert_eq!(
                u64::from_le_bytes(
                    bytes[XLAT_SECTION + i * 8..XLAT_SECTION + i * 8 + 8]
                        .try_into()
                        .unwrap()
                ),
                XlatEntry::invalid().word,
                "xlat[{i}] must be invalid"
            );
        }
    }

    /// `pipeline_barrier` is the null CB's entire payload: setting it must move
    /// EXACTLY CTRL bit 32 (`SetBarrier`) and nothing else in the 128 bytes —
    /// the cheapest device-side fence is cheap precisely because the barrier
    /// bit is the only thing it adds over an idle CB.
    #[test]
    fn null_cb_barrier_sets_exactly_ctrl_bit_32() {
        let plain = ControlBlockWire::null(ResponseTag(0x11), false);
        let fenced = ControlBlockWire::null(ResponseTag(0x11), true);
        assert!(fenced.ctrl_barrier());
        assert!(!plain.ctrl_barrier());

        // Bit 32 of the CTRL word is byte 4, bit 0 (little-endian).
        let mut diff = Vec::new();
        for (i, (a, b)) in plain
            .to_bytes()
            .into_iter()
            .zip(fenced.to_bytes())
            .enumerate()
        {
            if a != b {
                diff.push(i);
            }
        }
        assert_eq!(diff, vec![CTRL_SECTION + 4], "only CTRL byte 4 may differ");
        assert_eq!(
            fenced.to_bytes()[CTRL_SECTION + 4],
            0x01,
            "the differing byte must be exactly bit 32 set"
        );
    }

    /// The vendor's two zero-work markers are NOT interchangeable on the wire:
    /// `CreateNullCb` leaves the CMPT section INVALID (it never touches the
    /// template) and can carry the barrier bit, while `CreateCancelledCb`
    /// builds a VALID-but-unsynced compute section and carries the cancel
    /// byte. A future caller that swaps one for the other changes what the
    /// device sees below the CTRL word, so the distinction is pinned here.
    #[test]
    fn null_and_cancelled_are_distinct_zero_work_markers() {
        let null = ControlBlockWire::null(ResponseTag(5), true);
        let cancelled = ControlBlockWire::cancelled(ResponseTag(5));

        let cmpt_valid = |cb: &ControlBlockWire| {
            (u64::from_le_bytes(
                cb.to_bytes()[CMPT_SECTION..CMPT_SECTION + 8]
                    .try_into()
                    .unwrap(),
            ) >> 63)
                & 1
        };
        assert_eq!(cmpt_valid(&null), 0, "null CB's CMPT is invalid");
        assert_eq!(cmpt_valid(&cancelled), 1, "cancelled CB's CMPT is valid");

        assert!(null.ctrl_barrier(), "the fence is the null CB's payload");
        assert!(!cancelled.ctrl_barrier());
        assert!(!null.ctrl_cancelled());
        assert!(cancelled.ctrl_cancelled());
    }

    /// `ctrl_barrier` reads the same bit the constructors pack: true only for
    /// `null(tag, true)` and for `dma_immediate` with `pipeline_barrier = true`
    /// (the vendor's `CreateComputeDataTransfer` — the other `SetBarrier` site
    /// on a real CB); false for every other constructor, which never set it.
    #[test]
    fn ctrl_barrier_reads_the_bit_the_constructors_pack() {
        let dmva = VirtualAddressSbf::new(2, Flits::from_bytes(0x1000));
        let transfer = DmaTransfer {
            dmva,
            host_addr: HostPhysicalAddress::from_raw_for_tests(0x7f00_0000),
            size_bytes: TransferSize(256),
        };
        let window = XlatWindow {
            region_paddr: DevicePhysicalAddress(0x2_0000_0000),
            length: SegmentExtent(0x1000 + 256),
        };
        let dma_fenced =
            ControlBlockWire::dma_immediate(ResponseTag(1), true, transfer, window, true).unwrap();
        let dma_plain =
            ControlBlockWire::dma_immediate(ResponseTag(2), true, transfer, window, false).unwrap();
        assert!(dma_fenced.ctrl_barrier());
        assert!(!dma_plain.ctrl_barrier());

        let bootstrap = VirtualAddressSbf::new(7, Flits::from_bytes(0));
        let prog = (
            DevicePhysicalAddress(0x1_0000_0000),
            AllocationTotalSize(4096),
        );
        let compute = ControlBlockWire::compute(ResponseTag(3), &[], prog, bootstrap).unwrap();
        assert!(!compute.ctrl_barrier());

        let fill = ControlBlockWire::fill(
            ResponseTag(4),
            dmva,
            TransferSize(4096),
            0xDEAD_BEEF,
            true,
            DevicePhysicalAddress(0x3_0000_0000),
            SegmentExtent(4096),
        )
        .unwrap();
        assert!(!fill.ctrl_barrier());
        assert!(!ControlBlockWire::cancelled(ResponseTag(6)).ctrl_barrier());
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

    // ── the timestamp + power sections (bytes 8..32) ──────────────────────────────────────────────

    /// The TS section is five little-endian u32 cycle counts at bytes 8..28 and the power word a
    /// u32 at 28..32. Values are placed by ABSOLUTE byte offset — the header's own numbering, like
    /// [`rb_with_qgi_bits`] places by absolute bit number — so the test exercises the decode's
    /// offsets instead of restating them.
    #[test]
    fn response_block_decodes_timestamps_and_power() {
        let mut bytes = [0u8; RB_NUM_BYTES];
        let ts = [0xDEAD_BEEFu32, 0, 1, u32::MAX, 0x1234_5678];
        for (i, v) in ts.iter().enumerate() {
            bytes[TS_SECTION + i * 4..TS_SECTION + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let power = 0x0BAD_C0DEu32;
        bytes[PW_SECTION..PW_SECTION + 4].copy_from_slice(&power.to_le_bytes());
        let rb = ResponseBlockWire::from_bytes(&bytes);
        assert_eq!(rb.timestamps(), ts);
        assert_eq!(rb.power(), power);

        // The sections do not bleed into each other: a set APP beat must not reach the power
        // word, and a set power word must not reach TS.
        let mut bytes = [0u8; RB_NUM_BYTES];
        bytes[APP_SECTION..APP_SECTION + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        bytes[PW_SECTION..PW_SECTION + 4].copy_from_slice(&1u32.to_le_bytes());
        let rb = ResponseBlockWire::from_bytes(&bytes);
        assert_eq!(rb.timestamps(), [0; 5]);
        assert_eq!(rb.power(), 1);
    }

    /// The phase accessors take WRAPPING differences, because the counter wraps every ~4.2 s of
    /// card time: words straddling a wrap must difference to the small true gap, which plain
    /// subtraction would misread as ~4.29e9.
    #[test]
    fn response_block_ts_phase_differences_wrap() {
        let mut bytes = [0u8; RB_NUM_BYTES];
        // CB start and dmai end equal (every compute CB), then prep end, cmpt end and CB end
        // straddling the u32 wrap.
        let ts = [4_294_967_000u32, 4_294_967_000, 4_294_967_295, 90, 100];
        for (i, v) in ts.iter().enumerate() {
            bytes[TS_SECTION + i * 4..TS_SECTION + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let rb = ResponseBlockWire::from_bytes(&bytes);
        assert_eq!(rb.dmai_cycles(), 0);
        assert_eq!(rb.prep_cmpt_cycles(), 295);
        assert_eq!(rb.exec_cmpt_cycles(), 91);
        assert_eq!(rb.dmao_cycles(), 10);
        assert_eq!(rb.total_cmpt_cycles(), 396);
    }

    // ── the QGI/HMI app section (bytes 32..64) ────────────────────────────────────────────────────

    /// Build a 64-byte RB with `status`/`locator` in the return section and one QGI app-section field
    /// set, addressed by the header's ABSOLUTE bit number (`app_data_sbf.hpp:313-322`).
    ///
    /// ⛔ THIS IS THE POINT OF THE HELPER: it places bits by `bit / 64` and `bit % 64` over the whole
    /// 256-bit section, which is the vendor's own numbering, and NEVER reuses the per-beat local
    /// ranges the accessors are written in. So these tests exercise the beat-splitting arithmetic
    /// instead of restating it — a test that re-derived the local ranges would be a tautology and
    /// would pass even if every accessor read the wrong beat.
    fn rb_with_qgi_bits(status: u64, locator: u8, fields: &[(u32, u32, u64)]) -> ResponseBlockWire {
        let ret_word = (status << 5) | ((locator as u64) << 32);
        let mut bytes = [0u8; RB_NUM_BYTES];
        bytes[0..8].copy_from_slice(&ret_word.to_le_bytes());
        let mut beats = [0u64; 4];
        for &(hi, lo, value) in fields {
            let width = hi - lo + 1;
            assert!(hi / 64 == lo / 64, "test fields must not cross a beat");
            let mask = if width == 64 {
                u64::MAX
            } else {
                (1u64 << width) - 1
            };
            beats[(lo / 64) as usize] |= (value & mask) << (lo % 64);
        }
        for (i, beat) in beats.iter().enumerate() {
            bytes[APP_SECTION + i * 8..APP_SECTION + i * 8 + 8]
                .copy_from_slice(&beat.to_le_bytes());
        }
        ResponseBlockWire::from_bytes(&bytes)
    }

    /// The section is `Some` ONLY under the vendor's stated validity condition — `RB_ERROR` plus a QGI
    /// locator (`app_data_sbf.hpp:299-305`). A `Good` response, or an error from a non-QGI locator,
    /// carries no QGI syndrome and must not be read as one.
    #[test]
    fn qgi_section_is_only_valid_for_an_error_with_a_qgi_locator() {
        let synd = [(191, 158, 1u64 << 19)];
        assert!(
            rb_with_qgi_bits(0b11, LOCATOR_QGI_NO_DIAG, &synd)
                .app_qgi()
                .is_some(),
            "RB_ERROR + QGI_NO_DIAG (the locator our compute faults report) must decode"
        );
        assert!(
            rb_with_qgi_bits(0b11, LOCATOR_QGI_DIAG, &synd)
                .app_qgi()
                .is_some(),
            "RB_ERROR + QGI_DIAG must decode"
        );
        assert!(
            rb_with_qgi_bits(0b00, LOCATOR_QGI_NO_DIAG, &synd)
                .app_qgi()
                .is_none(),
            "a GOOD response has no QGI section however the app bytes read"
        );
        assert!(
            rb_with_qgi_bits(0b11, 0x1, &synd).app_qgi().is_none(),
            "an error from CB_QUEUE is not a compute-section fault"
        );
    }

    /// ⭐ THE FAULTING ADDRESS, which is the whole reason to decode this section: `hmi_addr_` is bits
    /// 101:71 in FLITS, so it lands in beat 1 at local 37:7 and 128 B/flit converts it to the byte
    /// address to compare against a translation's `paddr..paddr+length`. `hmi_mode_` (bit 70) says
    /// which side faulted.
    #[test]
    fn qgi_section_decodes_the_hmi_faulting_address_and_direction() {
        let addr_flits = 0x1234_5678u64;
        let qgi = rb_with_qgi_bits(
            0b11,
            LOCATOR_QGI_NO_DIAG,
            &[
                (101, 71, addr_flits),
                (70, 70, 1), // store
                (111, 104, 0x42),
                (119, 112, 0x7),
                (69, 64, 0x2A),
            ],
        )
        .app_qgi()
        .expect("RB_ERROR + QGI_NO_DIAG");

        assert_eq!(qgi.hmi_address_flits(), addr_flits);
        assert_eq!(qgi.hmi_address_bytes(), addr_flits * 128);
        assert!(
            qgi.hmi_is_store(),
            "hmi_mode_ = 1 is a STORE (header: 0-fetch 1-store)"
        );
        assert_eq!(qgi.hmi_syndrome(), 0x42);
        assert_eq!(qgi.hmi_tag(), 0x7);
        assert_eq!(qgi.hmi_snid(), 0x2A);
        assert!(
            qgi.is_hmi(),
            "a non-zero HMI syndrome makes the HMI fields valid"
        );
        assert!(!qgi.is_qgi(), "no QGI syndrome was set");
    }

    /// `qgi_synd_` is a 34-bit BITMASK of `ErrorCaseTypes` BIT POSITIONS, not an ordinal — the
    /// distinction that decides whether "19" means `riu_unmp_err` or bit 19. Both unmapped-address
    /// cases are named, since those are the ones that mean an access fell outside a translation's
    /// declared extent.
    #[test]
    fn qgi_syndrome_is_a_bitmask_of_named_error_cases() {
        let qgi = rb_with_qgi_bits(
            0b11,
            LOCATOR_QGI_NO_DIAG,
            &[
                (191, 158, (1u64 << 19) | (1u64 << 20) | (1u64 << 4)),
                (157, 128, 0x2_0000),
            ],
        )
        .app_qgi()
        .expect("RB_ERROR + QGI_NO_DIAG");

        let cases: Vec<_> = qgi.qgi_error_cases().collect();
        assert_eq!(
            cases,
            vec![
                QgiErrorCase::CoreHwFault, // 4
                QgiErrorCase::RiuUnmpErr,  // 19
                QgiErrorCase::PrepUnmpErr, // 20
            ],
            "lowest syndrome bit first, decoded by POSITION"
        );
        assert!(cases.iter().filter(|c| c.is_unmapped_address()).count() == 2);
        assert!(qgi.is_qgi());
        assert_eq!(qgi.qgi_address_bytes(), 0x2_0000 * 128);
    }

    /// Both validity predicates ALSO require `beat0_byte8_` (bits 63:56) to be zero — the header's
    /// `isQGI()`/`isHMI()` check the documented-unused byte, so a non-zero value means the section is
    /// not really a QGI/HMI record and its addresses must not be trusted.
    #[test]
    fn a_nonzero_unused_beat0_byte_invalidates_both_predicates() {
        let qgi = rb_with_qgi_bits(
            0b11,
            LOCATOR_QGI_NO_DIAG,
            &[
                (191, 158, 1u64 << 19),
                (111, 104, 0x42),
                (63, 56, 0x1), // documented unused -> must invalidate
            ],
        )
        .app_qgi()
        .expect("the section still decodes; it is the PREDICATES that must refuse");
        assert_eq!(qgi.beat0_byte8(), 0x1);
        assert!(!qgi.is_qgi(), "isQGI() requires beat0_byte8_ == 0");
        assert!(!qgi.is_hmi(), "isHMI() requires beat0_byte8_ == 0");
    }

    /// The beats are read from `APP_SECTION` (32), not from the start of the block — an off-by-one
    /// beat would still pass a test that only ever set fields in one beat.
    #[test]
    fn app_section_starts_at_byte_32() {
        assert_eq!(APP_SECTION, 32);
        assert_eq!(RB_NUM_BYTES - APP_SECTION, 4 * 8, "four 64-bit beats");
        // job_count_ is bits 15:0 => beat 0 => the FIRST two bytes of the app section.
        let mut bytes = [0u8; RB_NUM_BYTES];
        let ret_word = (0b11u64 << 5) | ((LOCATOR_QGI_NO_DIAG as u64) << 32);
        bytes[0..8].copy_from_slice(&ret_word.to_le_bytes());
        bytes[APP_SECTION] = 0xCD;
        bytes[APP_SECTION + 1] = 0xAB;
        let qgi = ResponseBlockWire::from_bytes(&bytes).app_qgi().unwrap();
        assert_eq!(qgi.job_count(), 0xABCD);
    }
}

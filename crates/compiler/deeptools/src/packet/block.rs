// SPDX-License-Identifier: Apache-2.0
//! THE UNIT BLOCK — one unit's regions as a run of flits, and several units' blocks laid into the SAME flits by
//! SLICE COLUMN.
//!
//! ⛔ A BLOCK IS A COLUMN, NOT A BYTE RANGE. A flit is eight 16-byte slices and each slice COLUMN belongs to one
//! unit — `senCompOfSliceId` (`dip.h:232-238`) states the assignment outright:
//!
//! ```text
//! {0, L0SU}, {1, L0LU}, {2, LXSU}, {3, LXLU}, {4, PE}, {5, SFP}, {6, L3SU}, {7, L3LU}
//! ```
//!
//! and the generator passes those same numbers positionally — `generatePESFPInitPacket(psInfo, SFP0, 5, …)`
//! (`dip.cpp:1716`), `PE0, 4` (`:1761`), `LXLU0, 3` (`:1807`), `LXSU0, 2` (`:1852`), `L0LU0, 1` (`:1897`),
//! `L0SU0, 0` (`:1942`), and `if (senComp == L3SU) slice_id = 6; if (senComp == L3LU) slice_id = 7;`
//! (`dip.cpp:2868-2871`). So a unit's block is a COLUMN of slices running DOWN a set of flits, and eight units'
//! blocks share those flits side by side.
//!
//! # EACH COLUMN ADVANCES ITS OWN CURSOR
//!
//! `Dip::fillGlobalBuffer` (`dip.cpp:4335-4351`):
//!
//! ```cpp
//! if ((rowPtrPerSlice[slice_id] + fltCnt) > globalInitPacket.size())
//!   globalInitPacket.resize(rowPtrPerSlice[slice_id] + fltCnt, nullInit);
//! for (const auto& entry : initFlits) {
//!   globalInitPacket.at(rowPtrPerSlice.at(slice_id)).at(slice_id) = entry;
//!   rowPtrPerSlice[slice_id]++;
//! }
//! ```
//!
//! There is ONE cursor per column, and the columns re-synchronise in exactly one place per core: the barrier
//! before the PT flits, "Before starting PT flits, row ptrs of all slices set to max row ptr"
//! (`dip.cpp:1988-1996`, `coreArch <= RCUDD1A_ISA` only). Because the cursors are independent, laying is
//! PERMUTATION-INDEPENDENT ACROSS COLUMNS: the order the generator walks its units in (SFP, PE, LXLU, LXSU,
//! L0LU, L0SU, L3LU, L3SU, then PT — `dip.cpp:1714-1985`, then `:2001-2143`) changes nothing about where a slice
//! lands. What DOES matter is the order of blocks WITHIN one column, and that is the caller's: two blocks in one
//! column stack, which is how corelet 0's and corelet 1's differing inits are both sent (`dip.cpp:2596-2604`).
//!
//! ⛔ THE REFERENCE PORT ONCE ALIGNED EVERY COLUMN'S k-TH BLOCK TO A SHARED ROW, on the theory that
//! `readInitPacket` segments from slice 0 alone. Set BOUNDARIES do come from slice 0 (`initpacket.cpp:574-576`,
//! `DT_CHECK(sl == 0)`), but the HEADER/SPR/LRF/IBUFF state machine does not: `awaitHeader`, `awaitSpr`,
//! `awaitLrf`, `awaitIbuff` are declared INSIDE `for (int sl = 0; sl < 8; sl++)` (`initpacket.cpp:372-382`), so
//! every column re-runs them over its own slices and a column's header is found wherever it lands.
//! Over-aligning was a real divergence in the emitted layout.
//!
//! # ⭐ WHAT OF dxp's GENERALITY IS GONE, AND WHY IT CANNOT ARISE
//!
//! * **`column: usize` + `is_pt_row: bool`** — the reference carried the two meanings of the 0..8 index in a
//!   number and a flag. They are [`Placement`]'s two variants here, so a PT row cannot be read as a unit.
//! * **The column as a PARAMETER.** For a unit block the column is DERIVED from the header's own target
//!   ([`SliceColumn::of_target`]), so a block whose header says L0SU and whose column says L3LU is not
//!   expressible. Only a PT row's index is supplied, because a PT header names PT and nothing narrower.
//! * **`fill: Slice`, a free parameter.** The fill is not free: `fillGlobalBuffer` builds its `nullInit` with
//!   `initslicepart[3] = core_mask` (`dip.cpp:4336-4342`), so it is DERIVED from the blocks' own headers
//!   ([`NullRow`]) and the reference's two entry points, `lay_blocks` and `lay_blocks_filled`, collapse to one.
//! * **`PacketBody::PerCore` / `assemble_flits_patched` / `do_patch_init`.** DROPPED, and this is a stated GAP
//!   rather than a simplification: a packet carrying one complete block set per core is not expressible here,
//!   exactly as the reference records — placing by column, 32 L3LU sets would overwrite each other and only the
//!   last core's addresses would survive. What makes it expressible is `Dip::doPatchInit`'s compression, which
//!   is not block geometry and is not ported in this crate. Whoever needs a multi-core packet must port it,
//!   not widen this file.
//! * **`Packet::flits`, `assemble_flits`, `InitBinary`, `serialize`, `split`, `packet_offsets`, `sentinel_cb`.**
//!   Already present in island 4 or out of scope — see the list at the foot of this doc.
//! * **`Result<Vec<Flit>, String>`.** The only fallible step was `do_patch_init`; with it dropped, nothing here
//!   can fail at run time — and there is no run time. Every refusal below is a `panic!`, a compile error.
//! * **The end-of-core barrier (`dip.cpp:2151-2157`).** Recorded and deliberately NOT executed: it assigns
//!   `rowPtrPerSlice = globalInitPacket.size()` with no resize, so its only effect is on where the NEXT core's
//!   blocks start — and one call to [`lay_blocks`] lays ONE core, because multi-core is the dropped `PerCore`
//!   above. The reference added a `resize` there that the C++ does not have, and it is unreachable in any case:
//!   `fillGlobalBuffer` keeps the buffer's size equal to the largest cursor.
//!
//! # WHAT ISLAND 4 ALREADY OWNS AND THIS FILE DOES NOT REDEFINE
//!
//! [`Slice`], [`Flit`] (with `Flit::SLICES` and `Flit::BYTES`), `QgiHeader`, [`FlitCount`] and `Packet`. In
//! particular `Packet::header_with_count` already derives `myflits` FROM the flits, so the reference's
//! `assemble_flits` — a QGI flit, then `lay_blocks`, with `qgi.myflits` patched in afterwards — has nothing left
//! in it: this file returns the packet BODY and island 4 states the count. `InitBinary` is
//! `islands::init_packet::Packets`, and the byte walk is `Flit::to_le_bytes`.

use crate::isa::regfile::Component as RegUnit;
use crate::packet::header::{
    Bits, CoreMask, CoreletUnit, Corelets, Header, InstrFlitCount, L3Unit, LrfFlitCount, Target,
};
use crate::packet::ibuff::Ibuff;
use crate::packet::lrf::LrfRegion;
use crate::packet::spr::SprSlice;

use crate::islands::init_packet::{Flit, FlitCount, Slice};

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// THE FOUR QUANTITIES THIS GEOMETRY INVITES CONFLATING.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/// WHICH OF A FLIT'S EIGHT SLICES — a position ACROSS a flit.
///
/// ⛔ NOT A FLIT INDEX, WHICH RUNS DOWN THE FLITS, and for a short packet both are small numbers in `0..8`. The
/// reference held both as `usize`, and `globalInitPacket.at(row).at(slice_id)` is the one place swapping them
/// still compiles. Minted only by [`SliceColumn::index`] and [`PtRow::index`] — the two things the C++ passes as
/// `slice_id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ColumnIndex(usize);

impl ColumnIndex {
    /// The position itself.
    pub const fn get(self) -> usize {
        self.0
    }
}

/// WHICH FLIT — a position DOWN the run of flits, the value of `rowPtrPerSlice`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FlitIndex(usize);

impl FlitIndex {
    /// The first flit of a packet's body. The QGI header's own flit is island 4's and is not counted here.
    pub const START: Self = Self(0);

    /// The index itself.
    pub const fn get(self) -> usize {
        self.0
    }

    /// The next flit down — `rowPtrPerSlice[slice_id]++`.
    pub const fn next(self) -> Self {
        Self(self.0 + 1)
    }

    /// A position counted from the start of a section — the frame `patchinit.cpp` walks its votes in.
    ///
    /// ⛔ STILL NOT A [`ColumnIndex`]. This exists so a patch can name the flit it sits at without holding a
    /// `usize` beside the column it also holds; the two are the axes `globalInitPacket.at(row).at(slice_id)`
    /// takes in that order.
    pub const fn of(flit: usize) -> Self {
        Self(flit)
    }
}

/// WHERE A REGION STARTS WITHIN ONE BLOCK — the subscript in `initFlits.at(2 + pktLRFFlits + i)`.
///
/// ⛔ NOT A [`FlitIndex`]. A block starts at whatever row its column's cursor had reached, so the two frames
/// differ by that cursor; they coincide only for the first block laid in a column, which is the case every small
/// fixture exercises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RowOffset(usize);

impl RowOffset {
    /// An offset within a block.
    pub const fn of(row: usize) -> Self {
        Self(row)
    }

    /// The offset itself.
    pub const fn get(self) -> usize {
        self.0
    }
}

/// A BYTE POSITION IN `init_binary.bin` — `f * 128 + s * 16` (island 4's geometry note).
///
/// Here so the column arithmetic can be pinned where it actually lands: two columns of one flit are two disjoint
/// 16-byte ranges, and that is the statement a column calculation gets wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ByteOffset(usize);

impl ByteOffset {
    /// The offset itself.
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Where the slice at (`flit`, `at`) begins in the file.
pub const fn byte_offset(flit: FlitIndex, at: ColumnIndex) -> ByteOffset {
    ByteOffset(flit.get() * Flit::BYTES + at.get() * Slice::BYTES)
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// THE COLUMN ASSIGNMENT — `senCompOfSliceId`, `dip.h:232-238`.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/// ONE UNIT'S SLICE COLUMN. `senCompOfSliceId` (`dip.h:232-238`), confirmed by the positional `slice_id`
/// arguments at `dip.cpp:1716`, `:1761`, `:1807`, `:1852`, `:1897`, `:1942` and `:2868-2871`.
///
/// ⭐ THE VARIANTS ARE DECLARED IN COLUMN ORDER AND [`Self::ALL`] REPEATS THAT ORDER, so the sweep below checks
/// the vendored numbers against the table's own positions — a variant moved in one place and not the other is a
/// build error rather than a rotated packet.
///
/// ⛔ THERE IS NO PT COLUMN. The PT's programs are placed by ROW (see [`PtRow`]), so a ninth variant here would
/// be a column the hardware does not have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceColumn {
    /// Column 0.
    L0Su,
    /// Column 1.
    L0Lu,
    /// Column 2.
    LxSu,
    /// Column 3.
    LxLu,
    /// Column 4.
    Pe,
    /// Column 5.
    Sfp,
    /// Column 6.
    L3Su,
    /// Column 7.
    L3Lu,
}

impl SliceColumn {
    /// Every column, IN COLUMN ORDER — the vendored map itself.
    pub const ALL: [SliceColumn; Flit::SLICES] = [
        SliceColumn::L0Su,
        SliceColumn::L0Lu,
        SliceColumn::LxSu,
        SliceColumn::LxLu,
        SliceColumn::Pe,
        SliceColumn::Sfp,
        SliceColumn::L3Su,
        SliceColumn::L3Lu,
    ];

    /// This column's position across a flit — the number `dip.h:232-238` gives it.
    pub const fn index(self) -> ColumnIndex {
        match self {
            Self::L0Su => ColumnIndex(0),
            Self::L0Lu => ColumnIndex(1),
            Self::LxSu => ColumnIndex(2),
            Self::LxLu => ColumnIndex(3),
            Self::Pe => ColumnIndex(4),
            Self::Sfp => ColumnIndex(5),
            Self::L3Su => ColumnIndex(6),
            Self::L3Lu => ColumnIndex(7),
        }
    }

    /// Which unit owns this column — the forward direction of `senCompOfSliceId` (`dip.h:232-238`).
    pub const fn unit(self) -> RegUnit {
        match self {
            Self::L0Su => RegUnit::L0su,
            Self::L0Lu => RegUnit::L0lu,
            Self::LxSu => RegUnit::Lxsu,
            Self::LxLu => RegUnit::Lxlu,
            Self::Pe => RegUnit::Pe,
            Self::Sfp => RegUnit::Sfp,
            Self::L3Su => RegUnit::L3su,
            Self::L3Lu => RegUnit::L3lu,
        }
    }

    /// Which column a unit's block goes in — the inverse of [`Self::unit`].
    ///
    /// ⛔ THE PT IS A REFUSAL, NOT A NINTH COLUMN. `generatePTInitPacket(psInfo, senComp, int slice_id, …)` is
    /// called with `p`, the PT ROW (`dip.cpp:2126-2142`), and so is `finalizeInitFlit` (`:2137`). A PT program
    /// therefore has no unit column at all, and answering this question with one would put the PT's slices in
    /// whichever unit's column that number named.
    pub const fn of_unit(unit: RegUnit) -> Self {
        match unit {
            RegUnit::L0su => Self::L0Su,
            RegUnit::L0lu => Self::L0Lu,
            RegUnit::Lxsu => Self::LxSu,
            RegUnit::Lxlu => Self::LxLu,
            RegUnit::Pe => Self::Pe,
            RegUnit::Sfp => Self::Sfp,
            RegUnit::L3su => Self::L3Su,
            RegUnit::L3lu => Self::L3Lu,
            RegUnit::Pt => panic!(
                "the PT has no slice column: its blocks are placed by PT ROW, because generatePTInitPacket and \
                 finalizeInitFlit are both called with the row in the slice_id parameter (dip.cpp:2126-2142). \
                 Build the block with UnitBlock::of_pt_row"
            ),
        }
    }

    /// Which column a corelet-resident unit's block goes in.
    ///
    /// ⭐ THE CORELETS DO NOT ENTER INTO IT, AND THAT IS STRUCTURAL HERE. Corelet 0 and corelet 1 SHARE a slice
    /// column (`dip.cpp:851-853`): when their inits are identical the block is sent once for both, and when they
    /// differ both are laid into the same column, one after the other. Taking the unit alone means no corelet
    /// bit can reach the column calculation.
    pub const fn of_corelet_unit(unit: CoreletUnit) -> Self {
        match unit {
            CoreletUnit::L0Su => Self::L0Su,
            CoreletUnit::L0Lu => Self::L0Lu,
            CoreletUnit::LxSu => Self::LxSu,
            CoreletUnit::LxLu => Self::LxLu,
            CoreletUnit::Pe => Self::Pe,
            CoreletUnit::Sfp => Self::Sfp,
            CoreletUnit::Pt => panic!(
                "the PT has no slice column: its blocks are placed by PT ROW (dip.cpp:2126-2142). Build the \
                 block with UnitBlock::of_pt_row"
            ),
        }
    }

    /// Which column an L3 unit's block goes in — `slice_id = 6` for L3SU, `7` for L3LU (`dip.cpp:2868-2871`).
    pub const fn of_l3_unit(unit: L3Unit) -> Self {
        match unit {
            L3Unit::L3Su => Self::L3Su,
            L3Unit::L3Lu => Self::L3Lu,
        }
    }

    /// Which column the block belonging to THIS HEADER goes in.
    ///
    /// ⭐ THE HEADER ALREADY NAMES THE UNIT, so the column is not a second statement of it. `target_unit_mask`
    /// is what the reader turns back into a component (`dip.cpp:844-870`), and it is checked AGAINST the column
    /// there: `slice_id >= 6` demands a bare L3 id, `slice_id < 6` demands a corelet bit. Deriving the column
    /// from the target is how a block that lands in one unit's column while claiming another's becomes
    /// unrepresentable rather than merely wrong.
    pub const fn of_target(target: Target) -> Self {
        match target {
            Target::OnCorelets { unit, corelets: _ } => Self::of_corelet_unit(unit),
            Target::L3(unit) => Self::of_l3_unit(unit),
        }
    }
}

/// A PT ROW — the OTHER meaning of the same 0..8 index.
///
/// ⛔ THE INDEX IS SHARED AND THE MEANING IS NOT. `finalizeInitFlit(p, …)` (`dip.cpp:2137-2142`) passes the PT
/// row `p` in the very parameter every other component passes its `slice_id` in, and `fillGlobalBuffer` writes it
/// to `.at(slice_id)`. So PT row 7's slices sit where L3LU's slices sit — a coincidence of numbering, which is
/// why the two are different types and only [`Placement`] joins them.
///
/// ⭐ AND ONLY PT SITS AFTER THE BARRIER. The pre-PT resynchronisation (`dip.cpp:1988-1996`) is what makes the
/// distinction load-bearing rather than cosmetic: [`lay_blocks`] must be able to tell a PT row's block from a
/// unit's to know which side of the barrier it belongs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PtRow(usize);

// ⛔ THE ADDRESSING BOUND AND THE MACHINE'S ROW COUNT AGREE, AND THEY ARE STILL TWO FACTS.
//
// [`PtRow::COUNT`] is the flit's slice count; `sys_arch_spec::PT_ROWS` is how many rows a corelet has. They are
// equal today, which is what makes writing a row into a slice work at all — so the equality is asserted rather
// than assumed, and a part with more rows than a flit has slices fails the build here instead of silently
// landing row 8 in no column.
//
// 🛑 AT MODULE LEVEL, BECAUSE A NAMED ASSOCIATED CONST NOTHING FORCES IS NEVER EVALUATED. `const _: ()` here is
// evaluated whatever else the crate does; the same assertion written inside the `impl` as a named const is inert.
const _: () = assert!(
    PtRow::COUNT == sys_arch_spec::PT_ROWS,
    "a PT row is written to one of a flit's slices, so a corelet cannot have more rows than a flit has slices"
);

impl PtRow {
    /// How many PT rows can be ADDRESSED. `dip.cpp:2036-2041` sizes every per-row array at EIGHT
    /// (`std::array<HeaderFlitSlice, 8> PTheaderCorelet0`, …) and the row is written to one of a flit's eight
    /// slices, so eight is the bound both sides agree on.
    ///
    /// ⛔ THIS IS NOT THE REAL ROW COUNT. `for (int p = 0; p < dscGlobal->sysDef.numPTRows; p++)`
    /// (`dip.cpp:2043`) reads a system-definition field this crate does not carry, so how many rows a part HAS
    /// is not stated here — only that a ninth cannot be addressed. A row at or above the bound is refused rather
    /// than wrapped into some unit's column.
    pub const COUNT: usize = Flit::SLICES;

    /// The row `generatePTInitPacket` is called with.
    #[cfg(feature = "arch-rcudd1a")]
    pub const fn of(row: usize) -> Self {
        assert!(
            row < Self::COUNT,
            "a PT row indexes one of a flit's eight slices (dip.cpp:2036-2041, fillGlobalBuffer at \
             dip.cpp:4335-4351); how many rows this part actually has is dscGlobal->sysDef.numPTRows \
             (dip.cpp:2043), which this crate does not carry"
        );
        Self(row)
    }

    /// ⛔ REFUSED ON EVERY ARCH BEYOND RCUDD1A. The row-as-column placement, and the barrier before it, are
    /// inside `if (dscGlobal->sysDef.coreArch <= IsaCoreGen::RCUDD1A_ISA)` (`dip.cpp:1988`); SEN1P5 takes
    /// `constructPTInitSnt1p5` (`dip.cpp:2147`, defined at `:4437`) instead, which is a different construction
    /// and is not ported. A PT row here would be this arm's geometry applied to an arch that does not use it.
    #[cfg(not(feature = "arch-rcudd1a"))]
    pub const fn of(row: usize) -> Self {
        let _ = row;
        panic!(
            "PT rows are placed as slice columns only for coreArch <= RCUDD1A_ISA (dip.cpp:1988-1996); this arch \
             takes constructPTInitSnt1p5 (dip.cpp:2147, :4437), which is not ported"
        )
    }

    /// Which of a flit's slices this row occupies — the shared `slice_id` parameter (`dip.cpp:2137-2142`).
    pub const fn index(self) -> ColumnIndex {
        ColumnIndex(self.0)
    }
}

/// WHERE A BLOCK GOES — a unit's column, or a PT row.
///
/// ⭐ TWO VARIANTS WHERE THE REFERENCE HAD A NUMBER AND A `bool`. `UnitBlock { column: usize, is_pt_row: bool }`
/// can say "column 4, is a PT row" and "column 4, is not" with the same number, and nothing but the flag
/// distinguishes them. Here the two facts are one value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// A unit's block, in that unit's column. Derived from the header's target, never supplied.
    Unit(SliceColumn),
    /// A PT row's block, in the slice its row number names.
    Pt(PtRow),
}

impl Placement {
    /// Which slice of each flit this block writes.
    pub const fn column(self) -> ColumnIndex {
        match self {
            Self::Unit(column) => column.index(),
            Self::Pt(row) => row.index(),
        }
    }

    /// Which side of the pre-PT barrier this block is laid on (`dip.cpp:1988-1996`).
    const fn pass(self) -> Pass {
        match self {
            Self::Unit(..) => Pass::Units,
            Self::Pt(..) => Pass::PtRows,
        }
    }
}

/// Which side of the pre-PT barrier a block is laid on.
///
/// `dip.cpp:1988-1996` resynchronises every column's cursor between the units and the PT rows, so the two groups
/// are laid in two passes and no unit block can land after the barrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pass {
    /// The eight units, each into its own column (`dip.cpp:1714-1985`).
    Units,
    /// The PT rows, after the barrier (`dip.cpp:2001-2143`).
    PtRows,
}

impl Pass {
    /// Both passes, in the order `generateInitPacket` runs them.
    const ALL: [Pass; 2] = [Pass::Units, Pass::PtRows];
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// THE REGIONS OF ONE BLOCK — `initFlits.at(…)`, `dip.cpp:3384-3452`.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/// THE FOUR REGIONS OF A BLOCK, IN HARDWARE ROW ORDER.
///
/// `generateL3InitPacket` composes them at fixed subscripts (`dip.cpp:3384-3452`): the HEADER at `at(0)`
/// (`:3386-3393`), the SPR flit at `at(1)` (`:3397-3405`), the LRF flits at `at(2 + i)` (`:3408-3417`) and the
/// IBUFF flits at `at(2 + pktLRFFlits + i)` (`:3445-3452`). The same shape appears again for the compute units
/// (`:3894-3901`) and for a PT row (`:4321-4324`), which is why it is stated once here.
///
/// ⛔ THE IBUFF START MOVES WITH THE LRF COUNT. A block with no LRF region starts its instructions at row 2, so a
/// hardcoded 2 is right for exactly the blocks that initialise no registers — and every one of those is a legal
/// block, which is why the assertions below measure a non-empty LRF region as well.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    /// One row: which unit, which regions follow, how many of each.
    Header,
    /// One row: the eight 16-bit special-purpose words.
    Spr,
    /// `pktLRF_flit` rows of register initialisations.
    Lrf,
    /// `pktIbuff_flit` rows of instruction words.
    Ibuff,
}

impl Region {
    /// Every region, in row order.
    pub const ALL: [Region; 4] = [Region::Header, Region::Spr, Region::Lrf, Region::Ibuff];

    /// How many rows the HEADER occupies — `totalFlits += 1` (`dip.cpp:3378`).
    pub const HEADER_ROWS: usize = 1;

    /// How many rows the SPR flit occupies — `totalFlits += 1` (`dip.cpp:2885`).
    pub const SPR_ROWS: usize = 1;

    /// THE HEADER IS ROW 0 WHATEVER THE BLOCK'S SHAPE IS — the one region whose position needs no count, which is
    /// what lets [`NullRow::of_block`] read a block's core mask without knowing the rest of it.
    pub const fn header_row() -> RowOffset {
        RowOffset(0)
    }

    /// Where this region begins within a block that declares `lrf` LRF flits.
    ///
    /// ⭐ THE HEADER'S OWN FIELD SUPPLIES THE ONE VARIABLE. `pktLRF_flit` is what the device walks the block by,
    /// so taking the region's length from anywhere else is how a header comes to describe a layout the block
    /// does not have.
    pub const fn first_row(self, lrf: LrfFlitCount) -> RowOffset {
        let spr = Self::header_row().get() + Self::HEADER_ROWS;
        let regs = spr + Self::SPR_ROWS;
        match self {
            Self::Header => Self::header_row(),
            Self::Spr => RowOffset(spr),
            Self::Lrf => RowOffset(regs),
            Self::Ibuff => RowOffset(regs + lrf.field().get() as usize),
        }
    }
}

/// How many flits a block spans: HEADER, SPR, its LRF region and its IBUFF region.
///
/// `totalFlits` accumulated at `dip.cpp:2885` (the SPR flit), `:3328` (`+= pktLRFFlits`), `:3360`
/// (`+= pktIbuffFlits`) and `:3378` (the header), checked against the composed count by
/// `DT_CHECK(composedFlits == totalFlits)` (`dip.cpp:3455`).
///
/// ⭐ THE COUNT IS ISLAND 4'S [`FlitCount`], NOT A TWIN OF IT. A number of flits is ONE quantity however it was
/// derived, and a second newtype whose body was `Self(other.0)` would only be a way to launder a block's height
/// into a packet's. Its 14-bit bound is the QGI field's, and a block cannot exceed what the packet containing it
/// is able to declare.
pub const fn block_flit_count(lrf: LrfFlitCount, instrs: InstrFlitCount) -> FlitCount {
    FlitCount::of(
        (Region::HEADER_ROWS + Region::SPR_ROWS) as u32 + lrf.field().get() + instrs.field().get(),
    )
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// THE BLOCK.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/// ONE UNIT'S INIT BLOCK — its rows in hardware order, and the column they run down.
///
/// ⛔ NO `Default`. The reference derived one, which is a block of no rows at column 0 — a claim about L0SU that
/// no unit made. Every block here comes from a HEADER, and the header states which unit it is for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitBlock {
    placement: Placement,
    rows: Vec<Slice>,
}

impl UnitBlock {
    /// A UNIT's block, whose column comes from its own header's target.
    pub fn of_unit(header: Header, spr: SprSlice, lrf: &LrfRegion, ibuff: &Ibuff) -> Self {
        let placement = Placement::Unit(SliceColumn::of_target(header.target));
        Self::assemble(placement, header, spr, lrf, ibuff)
    }

    /// A PT ROW's block. The row is supplied because a PT header names PT and nothing narrower —
    /// `unit_mask = CORELET0ID | CORELET1ID | PTID` (`dip.cpp:4611`) is the same mask for every row.
    ///
    /// ⛔ AND THE HEADER MUST AGREE THAT THIS IS THE PT. A non-PT block placed by row would land in whichever
    /// column that number named, which is exactly what [`Placement`]'s two variants exist to prevent — so the
    /// target is checked here rather than trusted.
    pub fn of_pt_row(
        row: PtRow,
        header: Header,
        spr: SprSlice,
        lrf: &LrfRegion,
        ibuff: &Ibuff,
    ) -> Self {
        match header.target {
            Target::OnCorelets {
                unit: CoreletUnit::Pt,
                corelets: _,
            } => {}
            Target::OnCorelets {
                unit: _,
                corelets: _,
            }
            | Target::L3(_) => panic!(
                "a block placed by PT ROW must be the PT's: the row is written to .at(slice_id) \
                 (fillGlobalBuffer, dip.cpp:4335-4351), so any other unit's block would land in whichever \
                 column the row number named. Use UnitBlock::of_unit"
            ),
        }
        Self::assemble(Placement::Pt(row), header, spr, lrf, ibuff)
    }

    /// The rows, in the order `initFlits` holds them.
    ///
    /// ⛔ AND THE HEADER'S DECLARED LENGTHS ARE CHECKED AGAINST THE REGIONS THEY DESCRIBE. `pktLRF_flit` and
    /// `pktIbuff_flit` are how the device finds a block's IBUFF start and its end; a header that declares one
    /// length while the block carries another does not corrupt one row, it makes every row after it read as part
    /// of a different region.
    fn assemble(
        placement: Placement,
        header: Header,
        spr: SprSlice,
        lrf: &LrfRegion,
        ibuff: &Ibuff,
    ) -> Self {
        let declared = block_flit_count(header.lrf_flits, header.instr_flits);
        let mut rows: Vec<Slice> = Vec::with_capacity(declared.get() as usize);

        assert!(
            RowOffset::of(rows.len()) == Region::Header.first_row(header.lrf_flits),
            "the HEADER is row 0 of a block (dip.cpp:3386-3393)"
        );
        rows.push(header.slice());

        assert!(
            RowOffset::of(rows.len()) == Region::Spr.first_row(header.lrf_flits),
            "the SPR flit is row 1 of a block (dip.cpp:3397-3405)"
        );
        rows.push(spr.slice());

        assert!(
            RowOffset::of(rows.len()) == Region::Lrf.first_row(header.lrf_flits),
            "the LRF region begins at row 2 of a block (dip.cpp:3408-3417)"
        );
        let registers = lrf.slices();
        assert!(
            registers.len() == header.lrf_flits.field().get() as usize,
            "the header's pktLRF_flit must be the LRF region's real length: it is what the device walks the \
             block by (dip.cpp:3328, :3408-3417)"
        );
        rows.extend_from_slice(&registers);

        assert!(
            RowOffset::of(rows.len()) == Region::Ibuff.first_row(header.lrf_flits),
            "the IBUFF region begins at row 2 + pktLRFFlits of a block (dip.cpp:3445-3452)"
        );
        assert!(
            ibuff.slices().len() == header.instr_flits.field().get() as usize,
            "the header's pktIbuff_flit must be the IBUFF region's real length (dip.cpp:3360, :3445-3452)"
        );
        rows.extend_from_slice(ibuff.slices());

        assert!(
            rows.len() == declared.get() as usize,
            "a block's height is its four regions and nothing else — DT_CHECK(composedFlits == totalFlits), \
             dip.cpp:3455"
        );
        Self { placement, rows }
    }

    /// Where this block goes.
    pub const fn placement(&self) -> Placement {
        self.placement
    }

    /// WHICH CORE THIS BLOCK BELONGS TO, read back out of its own HEADER row.
    ///
    /// ⛔ READ BACK, NOT STORED — the same reason [`NullRow::of_block`] reads it: a `cores` field here would be a
    /// second copy of the number the header already told the device, and the two could then disagree.
    pub fn core_mask(&self) -> u32 {
        self.rows()[Region::header_row().get()].0[NullRow::CORE_MASK_WORD]
    }

    /// The block's rows, in order.
    pub fn rows(&self) -> &[Slice] {
        &self.rows
    }

    /// How many flits this block spans — DERIVED from the rows, never stored beside them.
    ///
    /// ⛔ THE HEIGHT AND THE ROWS ARE ONE FACT, the same reason `Packet::header_with_count` derives `myflits` and
    /// [`Ibuff`] derives its flit count: a stored height is a second place a length is decided, and the device
    /// walks the block by it.
    pub fn flit_count(&self) -> FlitCount {
        FlitCount::of(u32::try_from(self.rows.len()).expect("a block holds fewer than 2^32 flits"))
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// LAYING BLOCKS INTO FLITS.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/// Write one slice into one column of a flit, leaving the other seven alone.
///
/// `globalInitPacket.at(row).at(slice_id) = entry` (`dip.cpp:4348-4349`). A `const fn`, so the assertions below
/// can run it: a column calculation that is right for column 0 and wrong for column 7 is what this file exists to
/// make impossible, and only a compile-time check of the LAST column can say so.
pub const fn write_column(flit: Flit, at: ColumnIndex, slice: Slice) -> Flit {
    let mut out = flit;
    out.0[at.get()] = slice;
    out
}

/// THE SLICE THAT FILLS A ROW A COLUMN DOES NOT COVER.
///
/// `fillGlobalBuffer`'s `nullInit` (`dip.cpp:4336-4342`): every word zero, then `initslicepart[3] = core_mask`.
/// The mask is what tells a reader whose row an empty one is, which is why the fill is not a free parameter and
/// certainly not zero — the reference's default of `Slice::default()` claimed no core at all.
///
/// ⭐ TWO DOORS, ONE FORMULA. [`Self::of_cores`] takes the mask as a value; [`Self::of_block`] reads it back out
/// of the word a block's HEADER already wrote it to, so the fill cannot disagree with what the block told the
/// device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NullRow(Slice);

impl NullRow {
    /// Which word of a HEADER slice carries `target_core_mask` — `initslicepart.at(3)` (`dip.cpp:3393`), the
    /// same word `nullInit` puts it in (`dip.cpp:4341`).
    pub const CORE_MASK_WORD: usize = 3;

    /// The null row for a given core mask.
    pub const fn of_cores(cores: CoreMask) -> Self {
        let mut words = [0u32; 4];
        words[Self::CORE_MASK_WORD] = cores.bits();
        Self(Slice(words))
    }

    /// The null row for a core mask already read out of a header word.
    const fn of_mask(mask: u32) -> Self {
        let mut words = [0u32; 4];
        words[Self::CORE_MASK_WORD] = mask;
        Self(Slice(words))
    }

    /// The null row for the core this block belongs to, read back out of its HEADER row.
    ///
    /// ⛔ READ BACK, NOT STORED. A `cores` field on [`UnitBlock`] would be a second copy of a number the header
    /// row already carries, and the fill would then be able to name a core the block does not.
    fn of_block(block: &UnitBlock) -> Self {
        let header = block.rows()[Region::header_row().get()];
        let mut words = [0u32; 4];
        words[Self::CORE_MASK_WORD] = header.0[Self::CORE_MASK_WORD];
        Self(Slice(words))
    }

    /// The sixteen bytes.
    pub const fn slice(self) -> Slice {
        self.0
    }
}

/// `rowPtrPerSlice` — ONE CURSOR PER COLUMN (`dip.cpp:4346-4350`).
///
/// ⛔ EIGHT CURSORS, NOT ONE. A single shared cursor would put a unit's second row on the same flit as every
/// other unit's second row only by accident, and would move a column's block whenever a different column grew.
struct Cursors([FlitIndex; Flit::SLICES]);

impl Cursors {
    /// Every column at the body's first flit.
    const START: Self = Self([FlitIndex::START; Flit::SLICES]);

    /// The row this column writes next, advancing it — `at(rowPtrPerSlice.at(slice_id)); rowPtr++`.
    fn take(&mut self, at: ColumnIndex) -> FlitIndex {
        let row = self.0[at.get()];
        self.0[at.get()] = row.next();
        row
    }

    /// The highest row any column has reached — `globalInitPacket.size()`, which `fillGlobalBuffer` keeps equal
    /// to the largest cursor.
    fn high_water(&self) -> FlitIndex {
        let mut high = FlitIndex::START;
        for cursor in self.0 {
            if cursor > high {
                high = cursor;
            }
        }
        high
    }

    /// THE BARRIER: "row ptrs of all slices set to max row ptr" (`dip.cpp:1988-1996`). Raise-only in the C++
    /// (`if (rowPtrPerSlice.at(slice_id) < globalInitPacket.size())`), which is the same assignment because the
    /// argument is the high-water mark.
    fn align_all(&mut self, to: FlitIndex) {
        for cursor in &mut self.0 {
            if *cursor < to {
                *cursor = to;
            }
        }
    }
}

/// LAY EVERY CORE'S BLOCKS INTO THE FLITS OF ONE PACKET'S BODY.
///
/// Row `r`, column `c` holds column `c`'s `r`-th slice; a row a column does not reach holds a [`NullRow`]. The QGI
/// header's flit is not here: island 4's `Packet` holds the header beside the body and derives `myflits` from it.
///
/// ⛔⛔ ONE BODY HOLDS EVERY CORE, AND THE NULL FILL IS PER GROWTH RATHER THAN PER BODY. `fillGlobalBuffer` is
/// called once per (core, unit) into the SAME `globalInitPacket`, and the rows a call CREATES take that call's
/// `core_mask` (`dip.cpp:4334-4351`: `globalInitPacket.resize(rowPtrPerSlice[slice_id] + fltCnt, nullInit)`, where
/// `nullInit` is zero but for `initslicepart[3] = core_mask`). So the mask in a null row is the mask of whichever
/// core was being written when that row came into existence, and a body spanning several cores has several.
pub fn lay_blocks(blocks: &[UnitBlock]) -> Vec<Flit> {
    let mut flits: Vec<Flit> = Vec::new();
    let mut cursors = Cursors::START;

    // ⛔⛔ ONE CORE AT A TIME, AND EVERY BARRIER IS THAT CORE'S OWN. `constructInitPacket` is a PER-CORE function
    // (`dip.cpp:1700`, `start_initFlits = globalInitPacket.size() - 1`), so its two levellings and its tail pad
    // bound one core's rows. Laying every core's units and then every core's PT rows instead put core 0's PT
    // program below core 31's L3 program — measured against `dip_standalone`'s own `init.txt`, which gave EIGHT
    // rows per core where this gave seven and disagreed from the fourth flit on.
    for core in cores_of(blocks) {
        let started = flits.len();
        for pass in Pass::ALL {
            match pass {
                Pass::Units => {}
                // The barrier before this core's PT blocks (`dip.cpp:1988-1996`).
                Pass::PtRows => {
                    let high = cursors.high_water();
                    cursors.align_all(high);
                }
            }
            for block in blocks
                .iter()
                .filter(|block| block.core_mask() == core && block.placement().pass() == pass)
            {
                let column = block.placement().column();
                // The `core_mask` this block's own `fillGlobalBuffer` call passes, read off its header row — so the
                // rows this block brings into existence are stamped with it.
                let fill = Flit([NullRow::of_block(block).slice(); Flit::SLICES]);
                for slice in block.rows() {
                    let row = cursors.take(column);
                    if row.get() >= flits.len() {
                        flits.resize(row.get() + 1, fill);
                    }
                    flits[row.get()] = write_column(flits[row.get()], column, *slice);
                }
            }
        }
        // "Before finishing, row ptrs of all slices set to max row ptr" (`dip.cpp:2150-2156`) — so the next core
        // starts at the same row in every column.
        let high = cursors.high_water();
        cursors.align_all(high);
        // ⛔⛔ AND THIS CORE'S ROW COUNT IS A MULTIPLE OF FOUR. `dip.cpp:2176-2193`:
        // `core_initFlits = globalInitPacket.size() - start_initFlits - 1` — and since `start_initFlits` was
        // `size - 1` when the core began, the two `-1`s cancel and `core_initFlits` is exactly the rows THIS core
        // added. The pad is null flits carrying `1 << coreNum`, written into all eight slices, and the reason is
        // the reader's: *"insert nullFlits so that QGI can inject the next header when datasync = 0"*.
        let added = flits.len() - started;
        let over = added % CORE_ROWS_MULTIPLE;
        if over != 0 {
            let pad = Flit([NullRow::of_mask(core).slice(); Flit::SLICES]);
            flits.resize(flits.len() + (CORE_ROWS_MULTIPLE - over), pad);
            cursors.align_all(FlitIndex(flits.len()));
        }
        // ⛔⛔ AND THE PAD IS CHECKED, NOT ASSUMED. `core_initFlits % 4` (`dip.cpp:2179`) is the reader's
        // requirement — *"insert nullFlits so that QGI can inject the next header when datasync = 0"* — and this
        // crate emitted NO pad at all until a byte diff against dip showed 7 rows per core where 8 belong. An
        // off-by-one in the levelling above would silently restore that.
        assert!(
            (flits.len() - started).is_multiple_of(CORE_ROWS_MULTIPLE),
            "core mask {core:#010x} added {} rows, which is not a multiple of {CORE_ROWS_MULTIPLE} \
             (`dip.cpp:2176-2193`)",
            flits.len() - started
        );
    }
    flits
}

/// HOW MANY ROWS ONE CORE'S SHARE IS ROUNDED UP TO — `core_initFlits % 4` (`dip.cpp:2179`).
const CORE_ROWS_MULTIPLE: usize = 4;

/// THE CORE MASKS IN THIS PACKET, ASCENDING — one per core that has a block.
///
/// ⛔ NOT A RUN OVER THE BLOCK LIST. The caller builds blocks stream-major and core-minor, so a core's blocks are
/// NOT adjacent; grouping by adjacency would make every run one block long and every barrier a no-op.
fn cores_of(blocks: &[UnitBlock]) -> Vec<u32> {
    let mut masks: Vec<u32> = blocks.iter().map(UnitBlock::core_mask).collect();
    masks.sort_unstable();
    masks.dedup();
    masks
}

// ───────────── THE INVARIANTS, `const` SO A WRONG GEOMETRY DOES NOT BUILD ─────────────

/// THE COLUMN MAP IS `dip.h:232-238`'s, NUMBER FOR NUMBER.
///
/// ⛔ THIS IS THE ONE TABLE THE WHOLE FILE RESTS ON. A rotated map still emits a file of exactly the right
/// length, in which every unit is initialised with its neighbour's program.
const _: () = assert!(SliceColumn::ALL.len() == Flit::SLICES);
const _: () = assert!(SliceColumn::L0Su.index().get() == 0);
const _: () = assert!(SliceColumn::L0Lu.index().get() == 1);
const _: () = assert!(SliceColumn::LxSu.index().get() == 2);
const _: () = assert!(SliceColumn::LxLu.index().get() == 3);
const _: () = assert!(SliceColumn::Pe.index().get() == 4);
const _: () = assert!(SliceColumn::Sfp.index().get() == 5);
const _: () = assert!(SliceColumn::L3Su.index().get() == 6);
const _: () = assert!(SliceColumn::L3Lu.index().get() == 7);

/// ⛔ AND THE WRONG ANSWERS ARE PINNED ABSENT, PAIR BY PAIR. Each of these swaps is a load unit fed a store
/// unit's program: the unit ids differ by one bit (`dip.h:38-48`) and the columns by one, so afterwards neither
/// the mask nor the length looks wrong.
const _: () = assert!(SliceColumn::L3Lu.index().get() != 6);
const _: () = assert!(SliceColumn::L3Su.index().get() != 7);
const _: () = assert!(SliceColumn::L0Lu.index().get() != SliceColumn::L0Su.index().get());
const _: () = assert!(SliceColumn::LxLu.index().get() != SliceColumn::LxSu.index().get());
const _: () = assert!(SliceColumn::Pe.index().get() != SliceColumn::Sfp.index().get());

/// EVERY COLUMN IS LISTED ONCE, ITS INDEX IS ITS POSITION IN THE TABLE, AND NO TWO SHARE A SLICE — so a variant
/// reordered in one place and not the other is a build error.
const _: () = {
    let mut at = 0;
    while at < SliceColumn::ALL.len() {
        assert!(
            SliceColumn::ALL[at].index().get() == at,
            "a column's index is its place in ALL"
        );
        let mut other = 0;
        while other < SliceColumn::ALL.len() {
            let shared =
                SliceColumn::ALL[at].index().get() == SliceColumn::ALL[other].index().get();
            assert!(
                shared == (at == other),
                "two units cannot share a slice column"
            );
            other += 1;
        }
        at += 1;
    }
};

/// THE UNIT MAP ROUND-TRIPS, so `senCompOfSliceId` and its inverse are one table and not two.
const _: () = {
    let mut at = 0;
    while at < SliceColumn::ALL.len() {
        let column = SliceColumn::ALL[at];
        assert!(SliceColumn::of_unit(column.unit()).index().get() == column.index().get());
        at += 1;
    }
};

/// THE HEADER'S TARGET AGREES WITH THE COLUMN, at both ends of the row.
///
/// ⛔ L3 IS THE PAIR THAT MATTERS. `slice_id >= 6` is the reader's own test for "no corelet bit"
/// (`dip.cpp:866-868`), so columns 6 and 7 are exactly the two an L3 target must land on.
const _: () = assert!(
    SliceColumn::of_target(Target::L3(L3Unit::L3Lu))
        .index()
        .get()
        == 7
);
const _: () = assert!(
    SliceColumn::of_target(Target::L3(L3Unit::L3Su))
        .index()
        .get()
        == 6
);
const _: () = assert!(
    SliceColumn::of_target(Target::L3(L3Unit::L3Lu))
        .index()
        .get()
        != SliceColumn::of_target(Target::L3(L3Unit::L3Su))
            .index()
            .get()
);
const _: () = assert!(
    SliceColumn::of_corelet_unit(CoreletUnit::L0Su)
        .index()
        .get()
        == 0
);
const _: () = assert!(SliceColumn::of_corelet_unit(CoreletUnit::Sfp).index().get() == 5);

/// ⭐ THE CORELETS SHARE A COLUMN (`dip.cpp:851-853`) — the same unit on corelet 0, on corelet 1 or on both lands
/// in one place, which is what makes stacking two corelets' differing blocks in one column the right layout.
const _: () = {
    let mut at = 0;
    while at < Corelets::ALL.len() {
        let target = Target::OnCorelets {
            unit: CoreletUnit::Pe,
            corelets: Corelets::ALL[at],
        };
        assert!(
            SliceColumn::of_target(target).index().get()
                == SliceColumn::of_corelet_unit(CoreletUnit::Pe).index().get(),
            "a corelet bit cannot move a block's column"
        );
        at += 1;
    }
};

/// A PT ROW IS THE SAME EIGHT SLICES, AND ROW 7 IS EXACTLY WHERE L3LU'S COLUMN IS.
///
/// ⛔ PINNING THE COINCIDENCE IS THE POINT. The two indices are equal and the two meanings are not, so a port
/// that "simplified" [`PtRow`] into a [`SliceColumn`] would pass every byte comparison of a packet with no PT
/// program — and put the PT's instructions into the L3 load unit in one that has it.
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(PtRow::of(0).index().get() == 0);
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(PtRow::of(7).index().get() == 7);
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(PtRow::of(7).index().get() == SliceColumn::L3Lu.index().get());
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(PtRow::of(4).index().get() != SliceColumn::L3Lu.index().get());
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(Placement::Pt(PtRow::of(4)).column().get() == 4);
const _: () = assert!(PtRow::COUNT == Flit::SLICES);
const _: () = assert!(Placement::Unit(SliceColumn::L3Lu).column().get() == 7);
const _: () = assert!(Placement::Unit(SliceColumn::L0Su).column().get() == 0);

/// A SLICE LANDS IN ITS OWN COLUMN AND LEAVES ITS NEIGHBOURS ALONE — checked at column 0, at column 7, and at
/// every column between.
///
/// ⛔ THIS IS THE ASSERTION THE FILE EXISTS FOR. An off-by-one column calculation is right at one end of the flit
/// and wrong at the other, so a check that only writes column 0 — which is what a first-unit-only fixture
/// exercises — cannot tell a correct one from a broken one.
const _: () = {
    let marker = Slice([0xDEAD_BEEF; 4]);
    let payload = Slice([1, 2, 3, 4]);
    let mut at = 0;
    while at < SliceColumn::ALL.len() {
        let column = SliceColumn::ALL[at].index();
        let written = write_column(Flit([marker; Flit::SLICES]), column, payload);
        let mut other = 0;
        while other < Flit::SLICES {
            if other == column.get() {
                assert!(
                    written.0[other].0[0] == 1,
                    "the slice must land in its own column"
                );
                assert!(written.0[other].0[3] == 4, "all four words, in order");
            } else {
                assert!(
                    written.0[other].0[0] == 0xDEAD_BEEF,
                    "writing one column must not touch a neighbour's"
                );
            }
            other += 1;
        }
        at += 1;
    }
};

/// THE COLUMNS TILE THE FLIT AND DO NOT OVERLAP, in bytes.
///
/// ⛔ COLUMN 7 IS THE ONE THAT CAN RUN OFF THE END. Its sixteen bytes must finish exactly where the next flit
/// begins — a column stride of anything but [`Slice::BYTES`] is invisible at column 0 and corrupts the following
/// flit at column 7.
const _: () = assert!(byte_offset(FlitIndex::START, SliceColumn::L0Su.index()).get() == 0);
const _: () =
    assert!(byte_offset(FlitIndex::START, SliceColumn::L3Lu.index()).get() == 7 * Slice::BYTES);
const _: () = assert!(
    byte_offset(FlitIndex::START, SliceColumn::L3Lu.index()).get() + Slice::BYTES
        == byte_offset(FlitIndex::START.next(), SliceColumn::L0Su.index()).get()
);
const _: () = {
    let mut at = 0;
    while at < SliceColumn::ALL.len() {
        let mine = byte_offset(FlitIndex::START, SliceColumn::ALL[at].index()).get();
        assert!(
            mine + Slice::BYTES <= Flit::BYTES,
            "a column's slice stays inside its flit"
        );
        let mut other = 0;
        while other < SliceColumn::ALL.len() {
            if other != at {
                let theirs = byte_offset(FlitIndex::START, SliceColumn::ALL[other].index()).get();
                let disjoint = mine + Slice::BYTES <= theirs || theirs + Slice::BYTES <= mine;
                assert!(disjoint, "two columns' byte ranges cannot overlap");
            }
            other += 1;
        }
        at += 1;
    }
};

/// A FLIT INDEX AND A COLUMN INDEX ARE NOT THE SAME NUMBER, and the arithmetic says which is which: one strides
/// by 128 bytes and the other by 16.
const _: () =
    assert!(byte_offset(FlitIndex::START.next(), SliceColumn::L0Su.index()).get() == Flit::BYTES);
const _: () = assert!(
    byte_offset(FlitIndex::START, SliceColumn::L0Lu.index()).get()
        != byte_offset(FlitIndex::START.next(), SliceColumn::L0Su.index()).get()
);

/// THE REGION ORDER: HEADER, SPR, then LRF, then IBUFF (`dip.cpp:3386-3452`).
const _: () = assert!(Region::ALL.len() == 4);
const _: () = assert!(Region::HEADER_ROWS == 1);
const _: () = assert!(Region::SPR_ROWS == 1);
const _: () = assert!(Region::header_row().get() == 0);

/// Offsets with NO registers to initialise: the IBUFF starts at row 2.
const _: () = assert!(Region::Header.first_row(LRF_NONE).get() == 0);
const _: () = assert!(Region::Spr.first_row(LRF_NONE).get() == 1);
const _: () = assert!(Region::Lrf.first_row(LRF_NONE).get() == 2);
const _: () = assert!(Region::Ibuff.first_row(LRF_NONE).get() == 2);

/// ⛔ AND WITH REGISTERS THE IBUFF MOVES — `at(2 + pktLRFFlits + i)` (`dip.cpp:3445-3452`). A hardcoded 2 is
/// right for exactly the blocks that initialise nothing, and every one of those is a legal block, so a fixture
/// full of them proves nothing. At the field's largest value the other three offsets must still be these.
const _: () = assert!(Region::Ibuff.first_row(LRF_SOME).get() == 2 + 5);
const _: () = assert!(Region::Ibuff.first_row(LRF_MAX).get() == 2 + LRF_MAX_FLITS as usize);
const _: () =
    assert!(Region::Ibuff.first_row(LRF_MAX).get() != Region::Ibuff.first_row(LRF_NONE).get());
const _: () = assert!(Region::Header.first_row(LRF_MAX).get() == 0);
const _: () = assert!(Region::Spr.first_row(LRF_MAX).get() == 1);
const _: () = assert!(Region::Lrf.first_row(LRF_MAX).get() == 2);

/// THE REGIONS ARE CONTIGUOUS AND EXHAUST THE BLOCK: the IBUFF begins where the LRF region ends, and the height
/// is the last region's start plus its length.
const _: () =
    assert!(Region::Ibuff.first_row(LRF_SOME).get() == Region::Lrf.first_row(LRF_SOME).get() + 5);
const _: () = assert!(block_flit_count(LRF_NONE, INSTR_NONE).get() == 2);
const _: () = assert!(block_flit_count(LRF_SOME, INSTR_NONE).get() == 2 + 5);
const _: () = assert!(block_flit_count(LRF_NONE, INSTR_SOME).get() == 2 + 3);
const _: () = assert!(block_flit_count(LRF_SOME, INSTR_SOME).get() == 2 + 5 + 3);
const _: () = assert!(
    block_flit_count(LRF_SOME, INSTR_SOME).get() as usize
        == Region::Ibuff.first_row(LRF_SOME).get() + 3
);

/// ⛔ A BLOCK IS NEVER SHORTER THAN ITS TWO FIXED ROWS. A height of 1 or 0 is a block whose SPR flit is somebody
/// else's header — neither region is optional, which is why neither count is an `Option`.
const _: () = assert!(block_flit_count(LRF_NONE, INSTR_NONE).get() >= 2);

/// The LRF and IBUFF counts the assertions above measure with. `Bits::exactly` refuses anything the field cannot
/// hold, so [`LRF_MAX`] is the widest an `pktLRF_flit` can declare.
const LRF_NONE: LrfFlitCount = LrfFlitCount::of_flits(Bits::exactly(0));
/// A non-empty LRF region — the value that discriminates a moving IBUFF start from a hardcoded 2.
const LRF_SOME: LrfFlitCount = LrfFlitCount::of_flits(Bits::exactly(5));
/// The largest `pktLRF_flit` the header's field holds.
const LRF_MAX_FLITS: u32 = Bits::<{ LrfFlitCount::FIELD_BITS }>::MAX;
/// The widest LRF region a header can declare.
const LRF_MAX: LrfFlitCount = LrfFlitCount::of_flits(Bits::exactly(LRF_MAX_FLITS));
/// A block with no instructions — a unit the group has no op for.
const INSTR_NONE: InstrFlitCount = InstrFlitCount::of_flits(Bits::exactly(0));
/// A block with instructions.
const INSTR_SOME: InstrFlitCount = InstrFlitCount::of_flits(Bits::exactly(3));

/// THE NULL ROW CARRIES THE CORE MASK IN WORD 3 AND ZERO ELSEWHERE (`dip.cpp:4336-4342`).
///
/// ⛔ NOT AN ALL-ZERO SLICE, which is what the reference's `Slice::default()` fill was: a padding row that claims
/// no core at all, where the mask is how a reader tells whose row an uncovered one is.
const _: () = assert!(NullRow::CORE_MASK_WORD == 3);
const _: () = assert!(NullRow::of_cores(FILL_CORES).slice().0[3] == FILL_CORES.bits());
const _: () = assert!(NullRow::of_cores(FILL_CORES).slice().0[0] == 0);
const _: () = assert!(NullRow::of_cores(FILL_CORES).slice().0[1] == 0);
const _: () = assert!(NullRow::of_cores(FILL_CORES).slice().0[2] == 0);

/// ⭐ AND THE MASK IS IN THE WORD THE HEADER WRITES IT TO — `target_core_mask` is `initslicepart.at(3)`
/// (`dip.cpp:3393`), the word [`NullRow::of_block`] reads. One number, one place, so the fill cannot name a core
/// the block does not.
const _: () = assert!(NullRow::of_cores(FILL_CORES).slice().0[NullRow::CORE_MASK_WORD] != 0);

/// A core mask whose bit is NOT bit 0, so the assertions above cannot pass by reading a zero.
const FILL_CORES: CoreMask =
    CoreMask::of_cores(&[crate::packet::header::CoreIdx::of_index(Bits::exactly(3))]);

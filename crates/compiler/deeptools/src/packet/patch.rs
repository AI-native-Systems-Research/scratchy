//! PATCH INIT — the port of `Dip::constructPatchInit` (`dip/patchinit.cpp:200-941`), driven by
//! `Dip::doPatchInit` (`:186-198`).
//!
//! ⛔⛔ WHY IT EXISTS: `target_core_mask` is a SET of cores (`dip.h:176`), so ONE block can initialise every core
//! that wants the same bytes. Writing a single-core mask and repeating the block per core makes a different image:
//! measured against dxp on one 512-op group, 23,933 flits against our 272,960, first packet 43 against 257.
//!
//! ⛔ AND IT IS NOT "COLLAPSE IDENTICAL BLOCKS". dxp's 43-flit packet is 1 QGI header + 11 flits carrying
//! `0xffffffff` + 31 content flits — about one per core. The headers, SPR and IBUFF are SHARED under one mask while
//! the LRF data stays per core in the same packet. Collapsing identical blocks would have produced one core's eight
//! rows and been wrong.
//!
//! ⛔ `constructCompoundPatchInit` (`:1018`) IS COMMENTED OUT of the driver — the compound form is not the live
//! path and must not be ported first.

use super::block::{NullRow, UnitBlock};
use crate::islands::init_packet::{Flit, Slice};
use crate::packet::header::CoreMask;

/// ONE SLICE COLUMN'S ROWS FOR ONE CORE, core field cleared — what `isSliceWiseIdentical` compares.
type ColumnBytes = Vec<[u32; Slice::WORDS]>;

/// EVERY BLOCK A CORE HAS IN ONE COLUMN, sorted so the comparison is about content and not emission order.
type ColumnOfCore = Vec<ColumnBytes>;

/// ONE CORE'S BYTES IN ONE SLICE COLUMN, with the target-core field removed.
///
/// ⭐ THE MASKING IS `isSliceWiseIdentical`'s OWN. With `maskTgtCore` set it copies each HEADER or ISNULL slice,
/// calls `targetCores.clear()`, re-serialises, and compares THAT (`patchinit.cpp:158-176`) — so two cores whose
/// only difference is which core they name compare equal. Word 3 is where that field lives
/// (`NullRow::CORE_MASK_WORD`), so clearing it is the same operation on our side.
fn column_bytes(block: &UnitBlock) -> ColumnBytes {
    block
        .rows()
        .iter()
        .map(|row| {
            let mut words = row.0;
            words[NullRow::CORE_MASK_WORD] = 0;
            words
        })
        .collect()
}

/// WHICH SLICE COLUMN `isSliceWiseIdentical` IS ASKED ABOUT WHEN THE GROUPS ARE FORMED.
///
/// ⛔⛔ SLICE 0 ALONE, AND THAT IS THE C++'s CHOICE RATHER THAN A SIMPLIFICATION OF IT.
/// `constructPatchInit` compares `isSliceWiseIdentical(pair.second.get(), ris.get(), 0, true)`
/// (`patchinit.cpp:385-386`) — column 0, the L0SU — and phase 2 then ASSERTS that within a group slice 0 is
/// unanimous, `DT_CHECK(sl > 0 || flitValToCoreId.size() == 1)` (`:404`). The other seven columns are allowed to
/// differ and become patches, so a stricter key here would split groups the reference keeps together.
const GROUPING_COLUMN: usize = 0;

/// THE CORES THAT SHARE ONE INIT — `patchCoreGroups` (`patchinit.cpp:379-399`).
///
/// A core joins the FIRST existing group whose slice 0 matches its own; otherwise it becomes a new group's base.
/// The order is the base cores' order, which is `initcores`' ascending order, because `baseRegInitSets` is keyed by
/// the base core.
///
/// ⛔⛔ A CORE WITH NO SLICE-0 BLOCK HAS AN *EMPTY* SLICE 0, NOT AN UNKNOWN ONE — and getting this backwards made
/// the port a no-op. A `RegInitSet` carries a slice for every one of the eight columns of every flit, with an
/// `ISNULL` slice wherever that unit has nothing (`patchinit.cpp:622-630` fills exactly that when a slice runs
/// short), and two null slices compare EQUAL once the core field is cleared. Treating absence as "no identity"
/// instead put every core in its own group: measured `groups=32 cores=32` on a group whose ops never touch the
/// L0SU, while `slice_zero_divergence` reported `None` — the content was identical and the grouping still split it.
pub fn core_groups(blocks: &[UnitBlock]) -> Vec<Vec<u32>> {
    // ⛔⛔ ONE CORE AT A TIME, NOT ONE MASK AT A TIME. A block's `target_core_mask` names a SET now that identical
    // content is shared, so a mask is not a core: keying on masks put every core whose column-0 content had been
    // merged into another block's mask in a group of its own, and the assertion below caught it (`groups=2` with no
    // slice-0 divergence at all). Expand the mask and ask about each core.
    let mut cores: Vec<u32> = blocks
        .iter()
        .flat_map(|block| {
            let mask = block.core_mask();
            (0..u32::BITS)
                .filter(move |bit| mask & (1 << bit) != 0)
                .map(|bit| 1u32 << bit)
        })
        .collect();
    cores.sort_unstable();
    cores.dedup();

    let slice_zero = |core: u32| -> ColumnOfCore {
        let mut found: ColumnOfCore = blocks
            .iter()
            .filter(|block| {
                block.core_mask() & core != 0 && block.placement().column().get() == GROUPING_COLUMN
            })
            .map(column_bytes)
            .collect();
        found.sort();
        found
    };

    let mut groups: Vec<(ColumnOfCore, Vec<u32>)> = Vec::new();
    for core in cores {
        let mine = slice_zero(core);
        match groups.iter_mut().find(|(base, _)| *base == mine) {
            Some((_, members)) => members.push(core),
            None => groups.push((mine, vec![core])),
        }
    }
    let formed: Vec<Vec<u32>> = groups.into_iter().map(|(_, members)| members).collect();
    // ⛔⛔⛔ IF NO TWO CORES' SLICE 0 DIFFER, THERE IS EXACTLY ONE GROUP. This is the bug that made the port a no-op:
    // a core with no block in column 0 was treated as having NO IDENTITY rather than an EMPTY slice 0, so every core
    // started its own group — `groups=32 cores=32` while `slice_zero_divergence` reported `None`. A `RegInitSet`
    // carries an `ISNULL` slice wherever a unit has nothing (`patchinit.cpp:622-630`) and two nulls compare EQUAL
    // once `targetCores` is cleared, so absence is a VALUE. Splitting on it costs every core its own init.
    assert!(
        !(formed.len() > 1 && slice_zero_divergence(blocks).is_none()),
        "no two cores' slice 0 differ, yet the grouping produced {} groups — absence of a column-0 block is an \
         EMPTY slice 0, not an unknown one (`patchinit.cpp:379-399`, `:622-630`)",
        formed.len()
    );
    formed
}

/// WHERE TWO CORES' SLICE 0 FIRST DIFFER — why [`core_groups`] split them.
///
/// ⛔ THE GROUPING IS ONLY AS GOOD AS SLICE 0's CORE-INVARIANCE. Every core addresses ITS OWN LX identically, so an
/// L0 register — which holds an LX-side address — has no reason to vary by core; only the L3's HBM side does
/// (`reginit.rs`'s per-core fold). So `groups == cores` is a claim about our own data worth reading back rather than
/// assuming, and this names the row and word that made it.
pub fn slice_zero_divergence(blocks: &[UnitBlock]) -> Option<(u32, u32, usize, usize)> {
    let mut per_core: std::collections::BTreeMap<u32, ColumnOfCore> =
        std::collections::BTreeMap::new();
    for block in blocks
        .iter()
        .filter(|b| b.placement().column().get() == GROUPING_COLUMN)
    {
        // One entry per CORE the block's mask names, for the same reason `core_groups` expands it.
        let mask = block.core_mask();
        for bit in (0..u32::BITS).filter(|bit| mask & (1 << bit) != 0) {
            per_core
                .entry(1u32 << bit)
                .or_default()
                .push(column_bytes(block));
        }
    }
    let mut it = per_core.iter();
    let (first_core, first) = it.next()?;
    for (core, other) in it {
        for (row, (a, b)) in first.concat().iter().zip(other.concat().iter()).enumerate() {
            for (word, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                if x != y {
                    return Some((*first_core, *core, row, word));
                }
            }
        }
    }
    None
}

/// ONE FLIT OF A PATCH — `PatchInitFlit` (built at `patchinit.cpp:566-579`).
///
/// A patch writes ONE slice column at one flit position for the cores that wanted a value the shared base does not
/// carry. `lxAddress` is the flit index within the section; the section offset and `reservedProgLxAddr` are added
/// afterwards (`:653-663`), which is a placement question and not this value's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Patch {
    /// ⛔⛔ THE TWO AXES ARE DIFFERENT TYPES, AND THIS IS THE STRUCT [`FlitIndex`]'s WARNING NAMES. Both were
    /// `usize` here — a position DOWN the flits and a position ACROSS one — and for a short packet both are small
    /// numbers in `0..8`, so swapping them compiled and addressed a real slice of the wrong flit.
    pub flit: crate::packet::block::FlitIndex,
    pub column: crate::packet::block::ColumnIndex,
    pub value: Slice,
    pub cores: Vec<u32>,
}

/// A PATCH GROUP'S SHARED INIT AND THE PATCHES THAT COVER THE CORES THAT DISAGREE WITH IT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupInit {
    pub cores: Vec<u32>,
    /// Per flit, per column: the value the MOST cores want. `None` where no core has a slice there at all.
    pub base: Vec<[Option<Slice>; Flit::SLICES]>,
    pub patches: Vec<Patch>,
}

/// ONE COLUMN'S ROWS FOR ONE CORE, with the header rows flagged — the C++ walks `regInitsEachSlice_` per slice and
/// needs the TYPE of each slice to know whether its core field participates in the comparison.
fn column_rows(
    blocks: &[UnitBlock],
    core: u32,
    column: crate::packet::block::ColumnIndex,
) -> Vec<(Slice, bool)> {
    let mut rows: Vec<(Slice, bool)> = Vec::new();
    for block in blocks
        .iter()
        .filter(|b| b.core_mask() == core && b.placement().column() == column)
    {
        for (at, row) in block.rows().iter().enumerate() {
            rows.push((*row, at == super::block::Region::header_row().get()));
        }
    }
    rows
}

/// A SLICE AS THE GROUP WOULD SEE IT — a header's core field replaced by the whole group's mask.
///
/// ⭐ `patchinit.cpp:368-384` and `:436-446`: before comparing or storing, a HEADER or ISNULL slice is copied and
/// every core of the group inserted into `targetCores`. So two cores whose headers differ only in which core they
/// name become ONE value, which is the entire point — that shared header is what addresses all of them.
fn as_group_sees_it(row: Slice, is_header: bool, mask: u32) -> Slice {
    let mut words = row.0;
    if is_header {
        words[NullRow::CORE_MASK_WORD] = mask;
    }
    Slice(words)
}

/// BUILD ONE PATCH GROUP'S SHARED INIT AND ITS PATCHES — `constructPatchInit`'s inner loops (`:302-581`).
///
/// ⭐ THE TARGET, MEASURED AGAINST dxp ON A 1-OP / 32-CORE GROUP: 8 shared flits carrying `0xffffffff` in every
/// header, then 31 patch flits — one per core that is not the base — each touching only columns 7, 6 and 0, the
/// L3LU / L3SU / L0SU. `31 == 32 - 1`, the base holding the most-common core's value.
///
/// ⛔ THE `fl == 0 || fl == 1` RULE IS THE C++'s AND IT IS NOT AN OPTIMISATION. If ANY core has a non-null slice at
/// flit 0 or 1, null candidates are struck out of the vote (`:536-549`) — the first two flits of a section are the
/// HEADER and the SPR, and a base whose header were null would describe no block at all.
pub fn group_init(blocks: &[UnitBlock], cores: &[u32]) -> GroupInit {
    let mask = cores.iter().fold(0u32, |acc, core| acc | core);
    let per_core: Vec<Vec<(Slice, bool)>> = crate::packet::block::SliceColumn::ALL
        .into_iter()
        .flat_map(|column| {
            cores
                .iter()
                .map(move |core| (column, *core))
                .collect::<Vec<_>>()
        })
        .map(|(column, core)| column_rows(blocks, core, column.index()))
        .collect();
    let rows_of = |column: crate::packet::block::ColumnIndex, at: usize| {
        &per_core[column.get() * cores.len() + at]
    };

    let mut base: Vec<[Option<Slice>; Flit::SLICES]> = Vec::new();
    let mut patches: Vec<Patch> = Vec::new();
    // ⭐ THE COLUMNS ARE THE EIGHT NAMED UNITS, WALKED AS THEMSELVES. `0..8` made the column a loop counter
    // that had to be re-bounded at every use; `SliceColumn::ALL` is the vendored map (`dip.h:232-238`), so a
    // ninth column is not expressible and each iteration already knows which unit it is.
    for column in crate::packet::block::SliceColumn::ALL {
        let longest = (0..cores.len())
            .map(|at| rows_of(column.index(), at).len())
            .max()
            .unwrap_or(0);
        if base.len() < longest {
            base.resize(longest, [None; 8]);
        }
        #[allow(clippy::needless_range_loop)]
        // `flit` indexes `base` AND every core's rows in lockstep
        for flit in 0..longest {
            // `flitValToCoreId` — (value seen as the group sees it) -> the cores that want it (`:355-402`).
            let mut votes: std::collections::BTreeMap<[u32; Slice::WORDS], (Vec<u32>, bool)> =
                std::collections::BTreeMap::new();
            for (at, core) in cores.iter().enumerate() {
                let Some((row, is_header)) = rows_of(column.index(), at).get(flit).copied() else {
                    continue;
                };
                let seen = as_group_sees_it(row, is_header, mask);
                let entry = votes.entry(seen.0).or_insert((Vec::new(), is_header));
                entry.0.push(*core);
            }
            if votes.is_empty() {
                continue;
            }
            // `skipNullCommonVal` (`:536-549`): the first two flits of a section are the header and the SPR, so a
            // null cannot win the vote there if any core has real content.
            let head_matters =
                flit < 2 && votes.iter().any(|(words, _)| words.iter().any(|w| *w != 0));
            let winner = votes
                .iter()
                .filter(|(words, _)| !(head_matters && words.iter().all(|w| *w == 0)))
                .max_by_key(|(_, (holders, _))| holders.len())
                .map(|(words, _)| *words)
                .unwrap_or_else(|| *votes.keys().next().expect("votes is not empty"));
            base[flit][column.index().get()] = Some(Slice(winner));
            for (words, (holders, _)) in &votes {
                if *words != winner {
                    patches.push(Patch {
                        flit: crate::packet::block::FlitIndex::of(flit),
                        column: column.index(),
                        value: Slice(*words),
                        cores: holders.clone(),
                    });
                }
            }
        }
    }
    GroupInit {
        cores: cores.to_vec(),
        base,
        patches,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  THE EMITTER — `constructPatchInit`'s output side, and `convertInitPacketToGip`'s ordering.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHERE A PROGRAM'S LX ADDRESSES START — `reservedProgLxAddr = 0x3f00` ("in stick", `sysdef.cpp:237`).
///
/// ⭐ THIS IS THE **WRITER'S** CONSTANT AND IT IS THE ONE TO USE. `constructPatchInit` sets
/// `pif.lxAddress += (cumSecLength + dscGlobal->sysDef.reservedProgLxAddr)` (`patchinit.cpp:658-660`) over a
/// `pif.lxAddress = fl` that is the flit's index WITHIN ITS SECTION (`:566`). The reader then subtracts a
/// DIFFERENT constant — `lxOffset = 0x3F00 - 1` (`:1084`, `:1109`) — so what it hands back is `fl + 1`, one
/// more than the index. The two constants differ by exactly one ON PURPOSE, and writing `fl + 1 + lxOffset`
/// instead of `fl + reservedProgLxAddr` is the same number arrived at from the reader's side; this states it
/// the way the writer does, so a future reader is not left deriving the identity.
const RESERVED_PROG_LX_ADDR: u32 = 0x3f00;

/// How many bits of the header's word 0 hold that biased address — `header[0] & 0x3FFF` (`:1109`).
const LX_ADDRESS_BITS: u32 = 14;

/// ⛔ SLICE 0 IS NOT PATCHABLE — `for (int slice_id = 1; slice_id < 8; slice_id++)` under the reader's own
/// comment *"slice0 cannot be patched"* (`patchinit.cpp:1128-1129`). It does not need refusing here: a patch
/// GROUP is cores whose slice 0 is identical (`isSliceWiseIdentical(.., 0, true)`, `:283-286`), so within a
/// group slice 0 cannot differ. The two facts are one mechanism, and this names it.
const FIRST_PATCHABLE_SLICE: usize = 1;

/// ONE PATCH FLIT'S HEADER — slice 0 of a patch flit, which carries no unit and no block.
///
/// ⛔⛔ THE PATCH BIT IS BIT 0 OF WORD 2 AND THE UNIT MASK MUST BE ZERO. The reader tests
/// `header_patch_bit = header[2] & 1` together with `header_unit_mask = header[2] >> 21 == 0`
/// (`patchinit.cpp:1096-1098`) — a flit is a patch only when it names NO unit, which is what distinguishes it
/// from a block header at the same position.
///
/// ⛔ `mismatchedFlits` SITS ABOVE THE PATCH BIT ON THIS ARCH: the reader takes `(header[2] >> 1) & 0xFF` for
/// `coreArch >= RCUDD1A_ISA` and `(header[2] << 1) & 0xFF` below it (`:1112-1117`), so the shift direction is a
/// GENERATION fact. This crate is RCUDD1A and up, hence `<< 1`.
///
/// ⭐ WORD 1'S TOP BIT IS THE COMPOUND FLAG and stays clear: `constructCompoundPatchInit` is commented out of
/// `doPatchInit` (`patchinit.cpp:194`), and the reader refuses what it cannot yet read —
/// `DT_ERROR("compound patch not supported yet in reversePatchInit")` (`:1102`).
/// THE UNION OF SOME BLOCKS' CORE MASKS, as a [`crate::packet::header::CoreMask`].
///
/// ⛔ THE INPUT IS MASKS. Every value in a patch group is `UnitBlock::core_mask`'s, so this ORs rather than
/// shifts — and it goes back through `CoreMask::of_cores`, so the union is a mask this crate built rather
/// than a bare `u32` assembled here.
fn cores_named(masks: &[u32]) -> crate::packet::header::CoreMask {
    let union = masks.iter().fold(0u32, |acc, mask| acc | mask);
    let named: Vec<crate::packet::header::CoreIdx> = (0
        ..crate::packet::header::CoreMask::FIELD_BITS)
        .filter(|bit| union & (1u32 << bit) != 0)
        .map(|bit| {
            crate::packet::header::CoreIdx::of_index(crate::packet::header::Bits::exactly(bit))
        })
        .collect();
    crate::packet::header::CoreMask::of_cores(&named)
}

/// WHERE IN ITS SECTION A PATCH APPLIES — `pif.lxAddress = fl` (`patchinit.cpp:566`), before the bias.
///
/// ⛔ A NEWTYPE BECAUSE THE HEADER'S OTHER NUMBER IS ALSO A `u32`. `patch_header(lx, mismatched, cores)` took
/// a bare `u32` here and a bare `u32` for the core mask, so the two were swappable and the compiler could not
/// see it — a patch at "core mask 5" addressed to "cores 0x3f04" is a legal-looking flit that patches the
/// wrong rows of the wrong cores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LxAddress(u32);

impl LxAddress {
    /// The flit's index within its section.
    #[must_use]
    pub const fn of_flit(at: u32) -> Self {
        Self(at)
    }

    /// The value the header carries — biased by [`RESERVED_PROG_LX_ADDR`], which is where the bias lives.
    #[must_use]
    const fn biased(self) -> u32 {
        self.0 + RESERVED_PROG_LX_ADDR
    }

    #[must_use]
    const fn get(self) -> u32 {
        self.0
    }
}

/// WHICH SLICES A PATCH FLIT REPLACES — `mismatchedFlits` (`patchinit.cpp:1111-1117`).
///
/// ⛔ A SET, NOT A COUNT. It is a bitmask over slice ids, and the one thing that must never be true of it is
/// that it names slice 0: *"slice0 cannot be patched"* (`:1128-1129`). [`Self::of_slice`] is the only way to
/// add one and it refuses that slice, so the illegal mask cannot be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MismatchedSlices(u8);

impl MismatchedSlices {
    /// Add one slice to the set. Total — [`PatchableColumn`] already excluded the slice that cannot be one.
    #[must_use]
    pub const fn with(self, column: PatchableColumn) -> Self {
        Self(self.0 | (1u8 << column.0))
    }

    /// The field value, which sits ABOVE the patch bit on this arch — see [`patch_header`].
    #[must_use]
    const fn field(self) -> u32 {
        (self.0 as u32) << 1
    }

    #[must_use]
    const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// EVERY SLICE ONE PATCH FLIT REPLACES, gathered before the flit is built.
///
/// ⭐ A STRUCT AND NOT A TUPLE, for the reason this crate already has `Walked`: clippy called the threaded
/// form "very complex", and three positional fields — an index, a core list and a column array — are three
/// things a reader has to keep straight by position.
struct MergedPatch {
    /// The flit's index within its section.
    at: usize,
    /// The cores this one flit is addressed to — masks, as [`GroupInit::cores`] holds them.
    cores: Vec<u32>,
    /// Per column, the replacement value and the proof that the column may be patched.
    columns: [Option<(PatchableColumn, Slice)>; Flit::SLICES],
}

/// A SLICE A PATCH MAY REPLACE — `1..8`.
///
/// ⛔⛔ SLICE 0 IS NOT ONE, AND THAT IS THE POINT OF THE TYPE. `reversePatchInit` applies a patch only over
/// `for (int slice_id = 1; slice_id < 8; slice_id++)`, under its own comment *"slice0 cannot be patched"*
/// (`patchinit.cpp:1128-1129`). It never needs to be: a patch GROUP is cores whose slice 0 is IDENTICAL
/// (`isSliceWiseIdentical(.., 0, true)`, `:283-286`), so within a group slice 0 cannot differ and a patch for
/// it cannot arise. A `usize` here made `with(0)` a legal call that only a runtime assert could catch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatchableColumn(usize);

impl PatchableColumn {
    /// `None` for slice 0 — which means the grouping and the vote disagree, and must not be dropped.
    #[must_use]
    pub const fn of(column: crate::packet::block::ColumnIndex) -> Option<Self> {
        let at = column.get();
        if at >= FIRST_PATCHABLE_SLICE && at < Flit::SLICES {
            Some(Self(at))
        } else {
            None
        }
    }
}

/// 🔒 THE THREE HEADER NUMBERS ARE THREE DIFFERENT TYPES, so no two can be transposed at the call site.
const _: () = {
    assert!(LxAddress::of_flit(5).biased() == 5 + RESERVED_PROG_LX_ADDR);
    // The patch bit is bit 0, so the slice set must start at bit 1 and never collide with it.
    assert!(MismatchedSlices(0x40).field() == 0x80);
};

fn patch_header(lx: LxAddress, mismatched: MismatchedSlices, cores: CoreMask) -> Slice {
    assert!(
        lx.biased() < (1 << LX_ADDRESS_BITS),
        "a patch's lx address {} does not fit the header's {LX_ADDRESS_BITS} bits once biased by \
         {RESERVED_PROG_LX_ADDR:#x} — `patchinit.cpp:1109` masks it to `0x3FFF`",
        lx.get()
    );
    assert!(
        !mismatched.is_empty(),
        "a patch flit that replaces NO slice: the reader would apply nothing and the flit is one the chain \
         still has to step over (`patchinit.cpp:1124-1136`)"
    );
    Slice([lx.biased(), 0, mismatched.field() | 1, cores.bits()])
}

/// ONE PATCH CORE GROUP AS FLITS: the shared base, then the patches that cover the cores disagreeing with it.
///
/// ⭐⭐ THE SHAPE IS dxp's OWN, MEASURED. `dip_standalone -p` on this crate's senprog for a 28-core group emits
/// 1 QGI header + 7 base flits carrying `0x0fffffff` in every header + 27 patch flits — one per core that is not
/// the base — each with `mismatchedFlits = 0xC0` (slices 6 and 7) at lx address 5. Unflagged slices of a patch
/// flit are ZERO, which is what the reader's slice loop implies and the artefact confirms.
pub fn emit_group(init: &GroupInit) -> Vec<Flit> {
    // ⛔ `GroupInit::cores` HOLDS MASKS, NOT INDICES — `column_rows` filters on `block.core_mask() == core`
    // and `group_init` ORs them, so a group's mask is their union and shifting one again names another core.
    let mask = cores_named(&init.cores);
    let mut flits: Vec<Flit> = Vec::with_capacity(init.base.len() + init.patches.len());

    // ⭐ THE BASE IS ALREADY THE SECTION'S LENGTH — `group_init_laid` sized it from the valid flits per
    // column, so there is nothing to trim here.
    let live = init.base.len();

    // ⛔ A COLUMN WITH NO SLICE HERE IS A NULL ROW ADDRESSED TO THE WHOLE GROUP, not a zero slice.
    // `constructPatchInit` fills a short column with `NullInitSlice { targetCores = grpCoreIds }`
    // (`patchinit.cpp:622-631`), and a null row is what carries the core mask that `updateLxAddress` counts.
    for row in &init.base[..live] {
        let mut slices = [NullRow::of_cores(mask).slice(); Flit::SLICES];
        for (column, held) in row.iter().enumerate() {
            if let Some(value) = held {
                slices[column] = *value;
            }
        }
        flits.push(Flit(slices));
    }

    // ⭐ ONE FLIT PER (LX ADDRESS, CORE SET), NOT PER PATCH. The golden puts slices 6 and 7 in ONE flit with
    // `mismatchedFlits = 0xC0`; emitting a flit per column would double the patch count and say the same thing
    // twice.
    let mut merged: Vec<MergedPatch> = Vec::new();
    for patch in &init.patches {
        let at = patch.flit.get();
        let column = patch.column.get();
        let patchable = PatchableColumn::of(patch.column).unwrap_or_else(|| {
            panic!(
                "a patch names slice {column}, which cannot be patched (`patchinit.cpp:1128`) — a patch group \
                 is cores whose slice 0 is IDENTICAL, so the grouping and the vote disagree"
            )
        });
        match merged
            .iter_mut()
            .find(|held| held.at == at && held.cores == patch.cores)
        {
            Some(held) => held.columns[column] = Some((patchable, patch.value)),
            None => {
                let mut columns = [None; Flit::SLICES];
                columns[column] = Some((patchable, patch.value));
                merged.push(MergedPatch {
                    at,
                    cores: patch.cores.clone(),
                    columns,
                });
            }
        }
    }

    for MergedPatch { at, cores, columns } in merged {
        let mut mismatched = MismatchedSlices::default();
        let mut slices = [Slice([0; Slice::WORDS]); Flit::SLICES];
        for (column, held) in columns.iter().enumerate() {
            if let Some((patchable, value)) = held {
                mismatched = mismatched.with(*patchable);
                slices[column] = *value;
            }
        }
        // ⛔ THE INDEX IS THE FLIT'S POSITION WITHIN ITS SECTION — `pif.lxAddress = fl` (`patchinit.cpp:566`),
        // biased by [`RESERVED_PROG_LX_ADDR`] on the way out. ⚠️ `cumSecLength` (`:658-663`) is NOT added here
        // because sections are not ported yet; for a single-section program it is zero, which is why this
        // matches `dip_standalone -p` today.
        slices[0] = patch_header(
            LxAddress::of_flit(u32::try_from(at).expect("a flit index that fits a patch header")),
            mismatched,
            cores_named(&cores),
        );
        flits.push(Flit(slices));
    }
    flits
}

/// WHAT ONE ROW OF A LAID COLUMN IS.
///
/// ⛔⛔ THE KIND COMES FROM THE WALK, NOT FROM THE BYTES. `readInitPacket` learns it by stepping the chain — a
/// header at the walk position, then the flits its own counts declare — which is why a DATA row that happens to
/// look like a header cannot be mistaken for one. The only kind a row states about itself is ISNULL, and that is
/// `isNullHeader`: words 0..2 zero with a non-zero core mask (`patchinit.cpp:14-18`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RowKind {
    /// A block's header row — its core field is replaced by the group's mask before comparing.
    Header,
    /// A row inside a block: the SPR, the LRF or the IBUFF. Compared as it stands.
    Data,
    /// A null row, which also carries a core mask (`:16-18`).
    Null,
}

impl RowKind {
    /// ⭐ HEADER **AND** NULL CARRY THE CORE FIELD — `constructPatchInit` inserts the group's cores into
    /// `targetCores` for both before comparing or storing (`patchinit.cpp:368-384`, `:436-446`), which is what
    /// makes two cores' otherwise-identical headers ONE value.
    const fn carries_the_core_field(self) -> bool {
        match self {
            Self::Header | Self::Null => true,
            Self::Data => false,
        }
    }
}

/// CLASSIFY ONE COLUMN OF A LAID RUN, by walking it the way the reader does.
///
/// ⛔ THE STEP IS `2 + lrf + ibuff + corr` (`patchinit.cpp:1273`) — the two being the header and the SPR, so a
/// block occupies its declared flits plus those two. A null is one row (`:1262-1265`).
fn classify(run: &[Flit], column: usize) -> Vec<RowKind> {
    let mut kinds = vec![RowKind::Data; run.len()];
    let mut row = 0;
    while row < run.len() {
        let v = run[row].0[column].0[2];
        if v & 0xFFFF_FF80 == 0 {
            kinds[row] = RowKind::Null;
            row += 1;
            continue;
        }
        kinds[row] = RowKind::Header;
        let lrf = ((v >> 9) & 0x1F) as usize;
        let ibuff = ((v >> 14) & 0x7F) as usize;
        let corr = (v & 0x3F) as usize;
        row += 2 + lrf + ibuff + corr;
    }
    kinds
}

/// ONE PATCH CORE GROUP'S SHARED INIT AND PATCHES, VOTED OVER THE **LAID** FLITS.
///
/// ⛔⛔⛔ THE C++ PATCHES THE LAID INIT, NOT THE BLOCKS. `doPatchInit` starts with
/// `convertGipToInitPacket(ipReg)` and `globalInitPacket` is what `lay_blocks` produces — cursor alignment and
/// null fill included. Voting over raw block rows instead produces a base whose flit count disagrees with the
/// counts its own headers declare, and the chain then fails to tile: MEASURED, *"slice 7's block chain stopped
/// at flit 433 of 409"*.
fn group_init_laid(runs: &[(u32, Vec<Flit>)]) -> GroupInit {
    let mask = cores_named(&runs.iter().map(|(core, _)| *core).collect::<Vec<_>>()).bits();
    let kinds: Vec<Vec<Vec<RowKind>>> = runs
        .iter()
        .map(|(_, run)| (0..Flit::SLICES).map(|c| classify(run, c)).collect())
        .collect();

    // ⛔⛔⛔ A COLUMN'S TRAILING NULLS ARE NOT PART OF THE SECTION, AND THAT IS PER (CORE, SLICE).
    // `validFlitPerCoreSlSec[core] = sectionSize - tailNullFlits` (`patchinit.cpp:328-337`), and only those
    // valid flits are voted on: a core whose column ends earlier contributes a NULL to the vote instead
    // (`:391-402`). Voting a column's tail nulls as if they were content is what made the shared base disagree
    // with dip's in both length and content.
    // Whether this patch group holds a single core — see the strip below.
    let lone = runs.len() == 1;
    let valid: Vec<Vec<usize>> = kinds
        .iter()
        .map(|per_column| {
            per_column
                .iter()
                .map(|rows| {
                    // ⛔⛔ A COLUMN'S TRAILING NULLS ARE NOT VOTED ON — `validFlitPerCoreSlSec = sectionSize -
                    // tailNullFlits` (`patchinit.cpp:328-337`), and a core whose column ends earlier votes a
                    // NULL instead (`:391-402`).
                    //
                    // ⭐ BUT THE STRIP IS ONLY FOR RECONCILING CORES, so a group of ONE has nothing to strip:
                    // its shared init IS its own init, QGI pad and all (`dip.cpp:2176-2193`). MEASURED against
                    // `dip_standalone -p` over 1198 ops: with the strip applied to single-core groups, ALL 148
                    // of them differ from dip and every one is SHORT; multi-core groups match.
                    if lone {
                        return rows.len();
                    }
                    rows.iter()
                        .rposition(|kind| *kind != RowKind::Null)
                        .map_or(0, |last| last + 1)
                })
                .collect()
        })
        .collect();

    // ⭐ TWO LEVELS OF MAXIMUM, BOTH THE C++'s: `maxSecSize` is the longest VALID column across the group's
    // cores (`:338-344`), and the section's emitted length is the longest of those across slices
    // (`lengthPerSec`, `:604-618`), with shorter columns padded by nulls (`:619-631`).
    let per_column: Vec<usize> = (0..Flit::SLICES)
        .map(|column| valid.iter().map(|core| core[column]).max().unwrap_or(0))
        .collect();
    let length = per_column.iter().copied().max().unwrap_or(0);

    let mut base: Vec<[Option<Slice>; Flit::SLICES]> = vec![[None; Flit::SLICES]; length];
    let mut patches: Vec<Patch> = Vec::new();

    for at in 0..length {
        // ⭐ NAMED COLUMNS, NOT `0..8` — `ColumnIndex` has no constructor from a bare index, so a position
        // across a flit is always some `SliceColumn`'s and never a loop counter that happens to be in range.
        for named in crate::packet::block::SliceColumn::ALL {
            let column = named.index().get();
            // Every core's value here, as the GROUP sees it.
            let mut votes: Vec<(u32, Slice)> = Vec::new();
            for (idx, (core, run)) in runs.iter().enumerate() {
                // ⛔ PAST THIS CORE'S VALID FLITS IT VOTES A NULL, not nothing and not its own tail —
                // `else { NullInitSlice with the group's cores }` (`:391-402`).
                if at >= valid[idx][column] {
                    votes.push((*core, NullRow::of_cores(cores_named(&[mask])).slice()));
                    continue;
                }
                let Some(flit) = run.get(at) else { continue };
                let kind = kinds[idx][column][at];
                votes.push((
                    *core,
                    as_group_sees_it(flit.0[column], kind.carries_the_core_field(), mask),
                ));
            }
            if votes.is_empty() {
                continue;
            }
            // The value the MOST cores want wins the base; the rest become patches.
            let mut tally: Vec<(Slice, Vec<u32>)> = Vec::new();
            for (core, value) in &votes {
                match tally.iter_mut().find(|(held, _)| held == value) {
                    Some((_, holders)) => holders.push(*core),
                    None => tally.push((*value, vec![*core])),
                }
            }
            // ⛔ THE FIRST MAXIMAL VALUE WINS, NOT THE LAST. `std::max_element` returns the FIRST element
            // among equals (`patchinit.cpp:566-570`) while Rust's `max_by_key` returns the last, and every
            // vote here is a 28-way tie when each core holds a different value — so the two disagree on the
            // base for exactly the case that matters. MEASURED against `dip_standalone -p`: it makes core 0
            // the base and patches cores 1.., where taking the last max patched core 0 instead.
            let winner = tally
                .iter()
                .fold(
                    None::<(Slice, usize)>,
                    |best, (value, holders)| match best {
                        Some((_, most)) if most >= holders.len() => best,
                        _ => Some((*value, holders.len())),
                    },
                )
                .map(|(value, _)| value)
                .expect("votes is not empty");
            base[at][column] = Some(winner);
            for (value, holders) in &tally {
                if *value != winner {
                    patches.push(Patch {
                        flit: crate::packet::block::FlitIndex::of(at),
                        column: named.index(),
                        value: *value,
                        cores: holders.clone(),
                    });
                }
            }
        }
    }
    GroupInit {
        cores: runs.iter().map(|(core, _)| *core).collect(),
        base,
        patches,
    }
}

/// `Dip::doPatchInit` (`patchinit.cpp:186-198`) — the whole transform: regular init in, patch init out.
///
/// ⛔⛔ WHAT `convertInitPacketToGip` DOES THAT THIS DOES NOT, AND WHY THAT IS SAFE. Its flit loop also fills
/// TWO side tables (`patchinit.cpp:84-140`): `getFoldInitMetaData()` and `mergeCorrectionTable`. Neither is
/// emitted here, and the reason is a number, not an assumption —
///
/// * FOLDS: the metadata is written only `if (entry.at(sl)->value.size() > 1)` — a slice holding MORE THAN ONE
///   fold value. On this arch there is only ever one: `numFoldsPerUnit[SFP] = 1`, `[PE] = 1`, `[PT] = 1` for
///   `coreArch <= RCUDD1A_ISA` (`sysdef.cpp:556-560`); SEN1P5 is where SFP and PE rise to 2. So `value.size()`
///   is 1 for every slice this crate can produce and the table stays empty.
///   ⛔ ON SEN1P5 IT WOULD NOT. That arch is a cargo feature here, and folded SPRs bring their own cross-value
///   invariant — one precision per unit, or dxp refuses with *"Precision SPR1 conflicts with previous
///   definition"* (`progir/progir.cpp:922`). See [`crate::packet::spr`]'s module doc.
/// * CORRECTIONS: `correctWithSymbol` carries LCCR flits, which this arch does not generate at all — see
///   [`crate::packet::header::Header::CORRECTION_FLITS`] and the four sites that say so.
///
/// ⭐ THE GROUPS ARE `patchCoreGroups`' (`:379-399`); the shared init and its patches are
/// `constructPatchInit`'s vote (`:302-581`) taken over the LAID flits; the flits are [`emit_group`]'s.
/// Concatenated in group order, which is what `convertInitPacketToGip` does with `initSets_` (`:84-140`).
pub fn patch_init(blocks: &[UnitBlock]) -> Vec<Flit> {
    core_groups(blocks)
        .into_iter()
        .flat_map(|group| {
            let runs: Vec<(u32, Vec<Flit>)> = group
                .iter()
                .map(|core| {
                    let mine: Vec<UnitBlock> = blocks
                        .iter()
                        .filter(|block| block.core_mask() == *core)
                        .cloned()
                        .collect();
                    (*core, crate::packet::block::lay_blocks(&mine))
                })
                .collect();
            emit_group(&group_init_laid(&runs))
        })
        .collect()
}

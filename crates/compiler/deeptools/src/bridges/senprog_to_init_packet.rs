//! BRIDGE 3 — island 3 → island 4. **LAYOUT**: ops become addressed packets.
//!
//! ⛔ OP ORDER STOPS BEING EXPRESSIBLE HERE. A packet is addressed, so "the op after this one" has no meaning on
//! the far side — which means everything that depends on order, and every sync is such a thing, must be resolved
//! BEFORE this bridge rather than after.
//!
//! # The layout rule, from `Dip::fillGlobalBuffer` (`dip.cpp:4334-4351`)
//!
//! `globalInitPacket` is a `vector<array<InitPerslice, 8>>` — a list of flits, each holding eight 16-byte slices —
//! and the writer keeps **one row cursor per slice column**, `rowPtrPerSlice[slice_id]`. Appending a unit's slices
//! advances only that unit's cursor, and grows the flit list when that column outruns it.
//!
//! ⭐ SO A UNIT'S PROGRAM IS A COLUMN, NOT A RANGE OF FLITS, and the columns are INDEPENDENT: two units emitting
//! different numbers of slices leave a ragged grid, which is why the C++ needs a cursor per column rather than one
//! write pointer. A layout that packed each unit's slices into consecutive flits would produce a file of the right
//! length with every unit but the first in the wrong place.
//!
//! ⛔ AND AN UNFILLED CELL IS A NULL SLICE CARRYING THE CORE MASK — `nullInit[i].initslicepart[3] = core_mask`
//! (`dip.cpp:4336-4343`). Not zero: the mask says which cores the flit is addressed to, so zeroed padding is
//! padding addressed to no core, and the file's length is identical either way.

use crate::islands::init_packet::{Flit, Slice};

/// WHICH SLICE COLUMN — one per unit, eight per flit (`dip.cpp:100`: `for slice_id = 0..8`).
///
/// ⛔ THE COLUMN IS AN IDENTITY, NOT AN INDEX TO COMPUTE. `fillGlobalBuffer` takes `slice_id` from its caller and
/// keeps a cursor per one, so a wrong column silently writes one unit's program into another's stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SliceColumn(usize);

impl SliceColumn {
    /// How many columns a flit has — the flit's own width, not a second constant.
    pub const COUNT: usize = Flit::SLICES;

    /// The column at `index`.
    ///
    /// ⛔ BOUNDED BY THE FLIT'S WIDTH, so a ninth column cannot be named. `const`, so a literal out of range is a
    /// build error.
    pub const fn of(index: usize) -> Self {
        assert!(
            index < Self::COUNT,
            "a flit has eight slice columns, one per unit"
        );
        Self(index)
    }

    pub const fn get(self) -> usize {
        self.0
    }
}

/// WHICH CORES A FLIT IS ADDRESSED TO — `core_mask`, word 3 of a null slice (`dip.cpp:4341`).
///
/// ⛔ ITS OWN TYPE BECAUSE ZERO IS A MEANINGFUL WRONG ANSWER. A null slice with a zero mask is addressed to no
/// core, so padding written as a plain `0` is padding the hardware ignores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreMask(u32);

impl CoreMask {
    pub const fn of(mask: u32) -> Self {
        Self(mask)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    /// The null slice this mask addresses: zero words but for the mask itself.
    pub const fn null_slice(self) -> Slice {
        Slice([0, 0, 0, self.0])
    }
}

/// THE RAGGED GRID the units fill — one row cursor per column, as `rowPtrPerSlice` is.
#[derive(Debug, Clone)]
pub struct Grid {
    rows: Vec<[Slice; SliceColumn::COUNT]>,
    /// `rowPtrPerSlice` — where the next slice of each column goes.
    cursors: [usize; SliceColumn::COUNT],
    mask: CoreMask,
}

impl Grid {
    /// An empty grid whose padding is addressed to `mask`.
    pub fn new(mask: CoreMask) -> Self {
        Self {
            rows: Vec::new(),
            cursors: [0; SliceColumn::COUNT],
            mask,
        }
    }

    /// Appends one unit's slices down its own column, growing the grid when that column outruns it.
    ///
    /// ⭐ THIS IS `fillGlobalBuffer` (`dip.cpp:4344-4350`): the loop advances only `rowPtrPerSlice[slice_id]`, so
    /// two units' streams interleave by ROW without either knowing the other's length.
    pub fn append(&mut self, at: SliceColumn, slices: &[Slice]) {
        let needed = self.cursors[at.get()] + slices.len();
        if needed > self.rows.len() {
            self.rows
                .resize(needed, [self.mask.null_slice(); SliceColumn::COUNT]);
        }
        for slice in slices {
            let row = self.cursors[at.get()];
            self.rows[row][at.get()] = *slice;
            self.cursors[at.get()] += 1;
        }
    }

    /// How deep the grid is — the deepest column, which is what the flit count follows.
    pub fn flit_count(&self) -> usize {
        self.rows.len()
    }

    /// The flits, in order.
    pub fn flits(&self) -> Vec<Flit> {
        self.rows.iter().map(|row| Flit(*row)).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WHAT THE GRID MUST NOT SMOOTH OVER
// ─────────────────────────────────────────────────────────────────────────────

/// ⛔ A NULL SLICE IS NOT ZERO: word 3 carries the core mask.
const _: () = assert!(CoreMask::of(0x3).null_slice().0[3] == 0x3);
const _: () = assert!(CoreMask::of(0x3).null_slice().0[0] == 0);
/// The eight columns are the flit's own width, so the two cannot disagree.
const _: () = assert!(SliceColumn::COUNT == Flit::SLICES);
const _: () = assert!(SliceColumn::of(SliceColumn::COUNT - 1).get() == 7);

use crate::isa::regfile::Component;
use crate::islands::progir::Bound;
use crate::islands::init_packet::{Packet, Packets, QgiHeader};
use crate::packet::block::UnitBlock;
use crate::packet::header::{
    CoreMask as HeaderCoreMask, CoreletUnit, Corelets, Header, InstrFlitCount, L3Unit, Target,
};
use crate::packet::ibuff::{Ibuff, IbuffFlitCount, InstrWidth, Word32};
use crate::packet::lrf::LrfRegion;
use crate::reginit::RegisterInits;

/// ONE INSTRUCTION STREAM — what a block is built from.
///
/// ⛔⛔ THE PT HAS ONE PER ROW AND EVERY OTHER UNIT ONE IN TOTAL. `constructPTInitPacket` loops the rows, names
/// each one's component `"pt_row" + p`, and calls `finalizeInitFlit(p, …)` with the ROW where the other units pass
/// a slice id (`dip.cpp:2043`, `:2124-2142`, `:4335-4351`). So iterating `Component::ALL` gives the PT ONE program
/// containing every row's instructions, which is a block that would be delivered to all of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stream {
    /// A component with one program — every unit but the PT.
    Whole(Component),
    /// One row of the systolic array.
    PtRow(u8),
}

// ⛔⛔ EVERY UNIT MUST APPEAR IN `Stream::ALL`, AND ONE DOES NOT. `Component::ALL` puts `Pt` at index 0, so the
// first loop below — which writes at the COMPONENT index rather than a compacted one — lands the eight whole units
// at indices 1..=8, and the row loop then starts writing at `Component::ALL.len() - 1 + 0` = index 8 and OVERWRITES
// the last of them. A unit absent from this list is a unit `terminate_programs` never gives a `RETURN`.
const _: () = {
    let mut comp = 0;
    while comp < Component::ALL.len() {
        // The PT has no whole-component program; its rows stand in for it.
        if !matches!(Component::ALL[comp], Component::Pt) {
            let mut found = false;
            let mut at = 0;
            while at < Stream::ALL.len() {
                if matches!(Stream::ALL[at], Stream::Whole(this) if this as u8 == Component::ALL[comp] as u8)
                {
                    found = true;
                }
                at += 1;
            }
            assert!(
                found,
                "a component is missing from `Stream::ALL`, so `terminate_programs` never appends its RETURN and \
                 its sequencer runs past the end of its IBUFF — an instruction fetch past the segment-7 XLAT \
                 window, which is a master with no translation behind it"
            );
        }
        comp += 1;
    }
    // ⛔ AND EVERY PT ROW EXACTLY ONCE. The same off-by-one that dropped a whole unit put `PtRow(0)` in the list
    // TWICE — so that row's program would have been terminated twice and another unit not at all. A list this is
    // iterated over to decide "which programs exist" must be a partition, not merely a cover.
    let mut row = 0;
    while row < crate::isa::unit::NUM_PT_ROWS {
        let mut seen = 0;
        let mut at = 0;
        while at < Stream::ALL.len() {
            if matches!(Stream::ALL[at], Stream::PtRow(this) if this == row) {
                seen += 1;
            }
            at += 1;
        }
        assert!(
            seen == 1,
            "a PT row appears in `Stream::ALL` other than exactly once — as a duplicate it is terminated twice, \
             and its slot came at the cost of a unit that is then terminated not at all"
        );
        row += 1;
    }
};

impl Stream {
    /// Every stream a packet can carry, in the order the blocks are built.
    ///
    /// ⭐ THE ROW COUNT IS THE ARCH'S — `numPTRows` (`sys-arch-spec/sysdef.cpp:199`), which is
    /// [`crate::isa::unit::NUM_PT_ROWS`] — and not the eight rows a flit can ADDRESS.
    pub(crate) const ALL: [Self; 8 + crate::isa::unit::NUM_PT_ROWS as usize] = {
        let mut all = [Self::PtRow(0); 8 + crate::isa::unit::NUM_PT_ROWS as usize];
        // ⛔⛔ THE WRITE INDEX IS COMPACTED, NOT THE COMPONENT INDEX, AND CONFLATING THEM DROPPED A WHOLE UNIT.
        // `Component::ALL` puts `Pt` FIRST and it has no whole-component program, so indexing this array by the
        // component's own position leaves a hole at 0 and pushes the eight whole units into 1..=8 — and the row
        // loop below, starting at `8`, then OVERWROTE the last of them. `Whole(L3su)` was in this list nowhere and
        // `PtRow(0)` was in it twice.
        //
        // ⛔ WHAT THAT COST: `terminate_programs` walks `Stream::ALL`, so the L3 STORE unit's program never got its
        // `RETURN`. Its sequencer runs past the end of its IBUFF, and an instruction fetch comes from segment 7 —
        // whose XLAT window flex-rs bounds to the program allocation, deliberately, "never the 16GB SEGMENT_SIZE".
        // A fetch past that window is a master with no translation behind it, which is `RAS::PCI::BusFence`.
        let mut next = 0;
        let mut comp = 0;
        while comp < Component::ALL.len() {
            match Component::ALL[comp] {
                // The PT has no whole-component program; its ROWS stand in for it, appended below.
                Component::Pt => {}
                whole => {
                    all[next] = Self::Whole(whole);
                    next += 1;
                }
            }
            comp += 1;
        }
        let mut row = 0;
        while row < crate::isa::unit::NUM_PT_ROWS {
            all[next + row as usize] = Self::PtRow(row);
            row += 1;
        }
        // ⭐ THE LENGTH IS THE PROOF THE TWO HALVES MEET: eight whole units then every PT row, with no hole left
        // by the skip and nothing overwritten.
        assert!(
            next == 8,
            "eight of the nine components have a whole-component program"
        );
        all
    };

    /// Which component's field table and opcode values this stream uses.
    pub(crate) const fn component(self) -> Component {
        match self {
            Self::Whole(comp) => comp,
            Self::PtRow(_) => Component::Pt,
        }
    }

    /// THE EXECUTOR THIS STREAM'S OWN INSTRUCTIONS NAME — what an instruction appended FOR this stream belongs to.
    ///
    /// ⛔⛔ A TERMINATOR MAY NOT INHERIT THE LAST INSTRUCTION'S `on`. A PT op is broadcast — its `unit=` is a row
    /// SPAN — so a `RETURN` carrying that span belongs to every row, and the eight row streams each appending one
    /// gave every row program EIGHT `RETURN`s and twenty-four hazard `NOP`s. `createFinalSenProg` appends exactly
    /// one of each per program (`dcgbeCodegen.cpp:2609-2624`), and the program it appends to is the STREAM's.
    pub(crate) const fn executor(self) -> crate::isa::unit::Executor {
        match self {
            Self::Whole(comp) => crate::isa::unit::Executor::whole(comp),
            Self::PtRow(row) => crate::isa::unit::Executor::PtRows {
                first: crate::isa::unit::PtRowIdx::from_row(row),
                last: crate::isa::unit::PtRowIdx::from_row(row),
            },
        }
    }

    /// WHETHER AN ISLAND-4 PROGRAM BELONGS TO THIS STREAM — the map from `SenUnit` to a slice column.
    ///
    /// ⭐ CORELET 0's PROGRAM ONLY, FOR A UNIT THAT HAS TWO. A senprog names the corelet in every header
    /// (`dpc.cpp:737-743`) and so states one program per corelet, while a packet addresses BOTH corelets from one
    /// block — dxp's own merged case, *"if the initpacket for both corelets are identical, then send it only once"*
    /// (`dip.cpp:2072-2074`). Taking both would double every compute program's instructions.
    pub(crate) fn is_unit(self, unit: crate::islands::senprog::SenUnit) -> bool {
        use crate::islands::senprog::SenUnit;
        let first_corelet = |corelet: crate::packet::header::Corelet| {
            matches!(corelet, crate::packet::header::Corelet::Zero)
        };
        match (self, unit) {
            (Self::PtRow(row), SenUnit::PtRow { row: held, corelet }) => {
                held.get() == row && first_corelet(corelet)
            }
            (Self::Whole(comp), held) => {
                held.component() as u8 == comp as u8
                    && match held {
                        SenUnit::L3(_) => true,
                        SenUnit::OnCorelet { corelet, .. }
                        | SenUnit::Compute { corelet, .. }
                        | SenUnit::PtRow { corelet, .. } => first_corelet(corelet),
                    }
            }
            (Self::PtRow(_), _) => false,
        }
    }

    /// Whether an op belonging to `on` runs in this stream.
    /// HOW MANY `NOP`s PRECEDE THIS STREAM'S `RETURN` — three for a PT row, none for anyone else.
    ///
    /// ⛔ THE REASON IS THE MACHINE'S, and `createFinalSenProg` states it in its own comment: *"add extra NOP for
    /// PTrow: issue: XRF WB is missed in senulator"* (`dcgbeCodegen.cpp:2617-2621`). The PT's write-back has not
    /// landed when the `RETURN` retires.
    ///
    /// ⛔ AND IT IS PER ROW, not once for the PT. The C++ tests `reNameUnitType(unitType) == PT` (`:2613-2618`),
    /// which every `pt_row` unit renames to — so each row's program takes its own three.
    const fn trailing_nops(self) -> crate::isa::unit::PadCount {
        match self {
            Self::PtRow(_) => crate::isa::unit::PadCount::of(3),
            Self::Whole(_) => crate::isa::unit::PadCount::of(0),
        }
    }

    /// WHICH STREAM A SENPROG UNIT'S PROGRAM IS — the map from island 4's unit to this partition.
    ///
    /// ⛔⛔ THE CORELET IS NOT PART OF IT, AND THAT IS THE REFERENCE'S OWN SHAPE. A component's program goes to
    /// BOTH corelets — a packet block addresses `Corelets::Both` and only corelet 0's senprog is laid
    /// (`dip.cpp:2072-2074`) — so two `SenUnit`s differing only in corelet run the SAME ops and share one entry.
    pub(crate) const fn of_unit(unit: crate::islands::senprog::SenUnit) -> Self {
        use crate::islands::senprog::SenUnit;
        match unit {
            SenUnit::PtRow { row, .. } => Self::PtRow(row.get()),
            // 🔑 `unit.component()` is `Pt` for no other variant, so no other variant can collide with a row.
            SenUnit::L3(_) | SenUnit::OnCorelet { .. } | SenUnit::Compute { .. } => {
                Self::Whole(unit.component())
            }
        }
    }

    /// Its position in [`Self::ALL`] — the key [`StreamIndex`] is built on.
    pub(crate) const fn which(self) -> usize {
        let mut at = 0;
        while at < Self::ALL.len() {
            let held = Self::ALL[at];
            let same = match (held, self) {
                (Self::Whole(a), Self::Whole(b)) => a as u8 == b as u8,
                (Self::PtRow(a), Self::PtRow(b)) => a == b,
                _ => false,
            };
            if same {
                return at;
            }
            at += 1;
        }
        panic!(
            "a stream that is not in `Stream::ALL` — the const assertion beside that list proves every \
             component has an entry, so this is a row index the array was not built for"
        )
    }

    pub(crate) const fn runs(self, on: crate::isa::unit::Executor) -> bool {
        match self {
            Self::Whole(comp) => matches!(
                on,
                crate::isa::unit::Executor::Whole(this) if this as u8 == comp as u8
            ),
            Self::PtRow(row) => on.covers_pt_row(row),
        }
    }
}

/// EVERY PROGRAM'S BLOCKS CLOSED AND ITS `RETURN` APPENDED — the tape as a set of finished programs.
///
/// ⛔⛔ THIS BELONGS TO BRIDGE 3, NOT TO THE LAYOUT. A `be` is a fact about an instruction's position in its
/// program and a `RETURN` is the program's terminator, so both are settled the moment the tape becomes PROGRAMS —
/// which is island 4. Running them inside [`lower`] left island 4 reading the tape one pass too early: its
/// senprog had no `RETURN` in any program, and `dip_standalone -s` laid that out happily, because dip terminates
/// nothing itself. A sequencer with no `RETURN` runs past the end of its IBUFF.
///
/// ⭐ THE ORDER IS `createSenProgramForPcfgs`'s OWN: every node lowered, every block end placed, THEN
/// `createFinalSenProg` appends the `RETURN` (`dcgbeCodegen.cpp:2609-2624`).
pub fn finished(bound: &Bound, inits: &RegisterInits) -> Terminated {
    let bound = &form_blocks(bound);
    Terminated::of(terminate_programs(bound), inits)
}

/// Island 4 → island 5: one packet per program, its flits laid out by unit column.
///
/// Each component that has instructions contributes one block — a header, its SPR slice, its LRF region and its
/// IBUFF — and `lay_blocks` places those down their own slice columns (`fillGlobalBuffer`, `dip.cpp:4334-4351`).
// ⛔⛔ AT `debug`, AND THE NUMBER IS WHY: this runs PER INSTRUCTION — 1,068,153 span-closes in one granite
// bake, 95% of a 214 MB stderr, all formatting and IO. `lib.rs`'s subscriber note states the rule this
// violated: "THE HOT SPANS ARE AT `debug`" — `FmtSpan::CLOSE` logs a line per span close at the default
// level. `RUST_LOG=deeptools=debug` still gets the spans when they are wanted.
#[tracing::instrument(level = "debug", skip_all)]
/// ONE ISLAND-4 INSTRUCTION AS THE MACHINE WORD — the bridge from `SenProg` to the bytes.
///
/// ⛔⛔⛔ THE IMAGE MUST COME FROM ISLAND 4, NOT FROM THE TAPE. Until this existed, `lower` encoded the island-3
/// tape directly and island 4 was built BESIDE it, serialized only for `dip_standalone`. Two independent paths from
/// one tape: so a senprog dip accepted and an image the card rejected could diverge with nothing to notice, which
/// makes every byte-equality result a statement about a path the card never runs.
///
/// ⭐ THE OPCODE VALUE AND THE BIT POSITIONS BOTH COME FROM THE ISA, so this is `dpc.cpp:1485-1516`'s own encoder:
/// walk the fields the instruction SET and OR each into the word at `typeToFieldBitShift`'s position — which is
/// exactly what `crate::isa::bit_of` answers.
fn word_of(
    instr: &crate::islands::senprog::Instruction,
    comp: Component,
) -> crate::generated::InstrWord {
    let opcode = instr.opcode;
    let mut word = crate::isa::values::OpUnit::of_component(comp)
        .value_of(opcode)
        .get() as u64;
    // WHICH BITS OF THE WORD SOME TERM HAS ALREADY CLAIMED.
    //
    // ⛔⛔ `|=` CANNOT REPORT A COLLISION, AND A COLLISION IS EXACTLY WHAT AN OVER-WIDE FIELD CAUSES. The crate
    // already names the failure for immediates — a value placed by `as u64` alone "sets every bit above the field,
    // overwriting whatever fields sit there" (`crate::isa::encode::imm_bits`) — but nothing was watching for it
    // once the terms reached the word. Two fields that write one bit merge silently and the instruction decodes as
    // neither.
    //
    // ⭐ IT INVENTS NO WIDTH. `defineField` states a position and, for an immediate, a width; it never states how
    // wide an encode-list field is. So the law here is not "this value fits N bits" — which would be a guess — but
    // "two fields of one instruction do not write the same bit", which is true by construction of a field table.
    let mut claimed = word;
    for (field, value) in crate::islands::senprog::stated_fields(instr) {
        // ⛔ A VIRTUAL FIELD IS NOT IN THE WORD, AND IT IS ITS NAME THAT SAYS SO. `isFieldVirtual` tests the
        // operand — `operandName == InstOperand::datatype_virtual` (`isa.hpp:315-317`) — and `defineField` then
        // asserts the consequence, `DT_CHECK(bitPos >= 100)` (`isa.cpp:179-181`). So this skips on the NAME; the
        // position is a fact about a virtual field rather than the test for one, and reading it the other way
        // round would make any field that happened to sit high vanish from the word.
        //
        // ⭐ SKIPPED, NOT REFUSED. `datatype_virtual` is a field an L3 type-2300 instruction legitimately STATES —
        // a senprog prints it, at the position the table gives — and only the encoded word has no room for it.
        if field.is_virtual() {
            continue;
        }
        // ⛔ A FIELD THIS OPCODE HAS NO POSITION FOR IS A REFUSAL, NOT A SKIP. `fields` yields what the instruction
        // holds and `bit_of` where this component puts it; a miss means the two disagree about the instruction, and
        // silently dropping the term is how a loop's trip count once encoded as zero.
        let bit = crate::isa::bit_of(comp, opcode, field).unwrap_or_else(|| {
            panic!(
                "{opcode:?} on the {comp:?} states no position for {field:?}, which island 4 says it holds",
            )
        });
        let at = bit.get();
        let shifted = value.get() << at;
        // ⛔ THE VALUE MUST SURVIVE ITS OWN SHIFT. A term wide enough to run off the top of the word loses those
        // bits here rather than in the stream, where the only symptom is an instruction that decodes as something
        // else.
        assert!(
            shifted >> at == value.get(),
            "{opcode:?} on the {comp:?} states {field:?} = {}, which does not fit in the word above bit {at}",
            value.get()
        );
        assert!(
            claimed & shifted == 0,
            "{opcode:?} on the {comp:?} writes {field:?} = {} at bit {at}, and that overlaps a field this instruction \
             has already written — `|=` would merge the two and the instruction would decode as neither",
            value.get()
        );
        claimed |= shifted;
        word |= shifted;
    }
    crate::generated::InstrWord::of(word)
}

/// WHICH CORES — the group's, and one op's, as two types that cannot be swapped.
///
/// ⛔⛔⛔ THIS IS A MODULE SO THAT `candidates` IS UNREACHABLE FROM THE EMITTER. Making the two newtypes and
/// leaving them beside the emission loop was NOT a lock: `GroupCores::candidates` was file-private, the loop is
/// in the same file, and re-introducing the original bug still compiled. Checked, and it did. The boundary is
/// what enforces it — outside this module the ONLY way to obtain a list of cores to emit for is
/// [`OpCores::of`], which cannot be called without a `GroupPos` and the register inits.
mod cores {
    use super::{HeaderCoreMask, RegisterInits};
    use crate::expansion_tape::GroupPos;
    use crate::packet::header::CoreIdx;

    /// EVERY CORE ANY OP OF THIS GROUP USES — the group's `numCoresUsed_`.
    ///
    /// ⛔ NOT THE CORES TO EMIT BLOCKS FOR. This was a bare [`HeaderCoreMask`] whose `.cores()` was read in two
    /// places to decide which cores a PACKET initialises — a `Vec<CoreIdx>` exactly like the one an OP needs, so
    /// the wrong quantity type-checked. MEASURED cost on `prefill_mq7/e6c7cdd8605ed357/group_0` op 2: its
    /// senprog names SEVEN cores while the emitter produced blocks for 28.
    #[derive(Clone, Copy)]
    pub struct GroupCores(HeaderCoreMask);

    impl GroupCores {
        #[must_use]
        pub const fn of(mask: HeaderCoreMask) -> Self {
            Self(mask)
        }

        /// The cores an op MIGHT run on. Private to this module, so the emitter cannot reach it.
        fn candidates(self) -> Vec<CoreIdx> {
            self.0.cores()
        }

        /// HOW MANY CORES THE IMAGE IS FOR — `initcores.size()`.
        ///
        /// ⛔ A COUNT, AND DELIBERATELY NOT THE LIST. [`Self::candidates`] stays private for the reason above;
        /// this hands out only the SIZE, which is all `updateLxAddress` uses it for — its loop is
        /// `for (int coreIdx = 0; coreIdx < initcores.size(); coreIdx++)` (`patchinit.cpp:25`), so the count is
        /// the whole of what the reader needs and the list would re-open what `OpCores` closed.
        #[must_use]
        pub fn count(self) -> CoreCount {
            CoreCount(u32::try_from(self.candidates().len()).expect("a group has at most 32 cores"))
        }
    }

    /// `initcores.size()` — HOW MANY CORES AN IMAGE IS FOR, and the bound on every core-mask walk over it.
    ///
    /// ⛔ A NEWTYPE BECAUSE THE ALTERNATIVE WAS `u32::BITS`. A core mask is a `u32` and it is tempting to walk
    /// all 32 bits of it; `updateLxAddress` walks `initcores.size()` of them and IGNORES the rest
    /// (`patchinit.cpp:25-27`). The two coincide only for a 32-core group — which is the group the bake runs, so
    /// the difference is invisible here and would appear on the first narrower one.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CoreCount(u32);

    impl CoreCount {
        /// The count, for the walk it bounds.
        #[must_use]
        pub const fn get(self) -> u32 {
            self.0
        }
    }

    /// THE CORES ONE OP ACTUALLY RUNS ON — `islands::superdsc::Cores`, recorded per op by `reginit::addresses`.
    ///
    /// ⛔ A DIFFERENT TYPE FROM [`GroupCores`] ON PURPOSE. Emitting a block for a core an op does not run on
    /// writes that op's program into that core's IBUFF, and `convertIr2Senprog` DT_ERRORs on a core absent from
    /// `progstateinfo_` (`sys-arch-spec/dpc/dpc.cpp:627-628`).
    pub struct OpCores(Vec<CoreIdx>);

    impl OpCores {
        /// ⭐ THE ONE PREDICATE. Every reader — the island-4 build, the model loop and the block census — comes
        /// through here, so no second reading of the group's mask can disagree with it.
        pub fn of(group: GroupCores, at: GroupPos, inits: &RegisterInits) -> Self {
            Self(
                group
                    .candidates()
                    .into_iter()
                    .filter(|core| inits.runs_on(at, crate::reginit::Core::of(core.get())))
                    .collect(),
            )
        }

        /// The core island 4 is built for: its register VALUES are this one's.
        ///
        /// ⛔ THE OP'S FIRST, NOT THE GROUP'S. This was computed once outside the op loop, so an op running on
        /// cores 5..11 built island 4 with core 0's register values.
        pub fn first(&self) -> CoreIdx {
            *self.0.first().expect(
                "an op runs on at least one core — `reginit::runs_on_cores` refuses a count of zero",
            )
        }

        pub fn each(&self) -> impl Iterator<Item = CoreIdx> + '_ {
            self.0.iter().copied()
        }

        /// Every core but the one island 4 was built for — the model's own re-runs.
        pub fn others(&self) -> impl Iterator<Item = CoreIdx> + '_ {
            self.0.iter().skip(1).copied()
        }
    }
}

pub use cores::{CoreCount, GroupCores, OpCores};

/// Cumulative wall per part of the stream loop, nanoseconds — `words`, `lrf_named`, `blocks`.
pub(crate) static I5_PART_NANOS: [core::sync::atomic::AtomicU64; 3] = [
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
];


fn i5_timed(slot: usize, at: std::time::Instant) {
    I5_PART_NANOS[slot].fetch_add(
        u64::try_from(at.elapsed().as_nanos()).unwrap_or(u64::MAX),
        core::sync::atomic::Ordering::Relaxed,
    );
}
///
/// ⭐ SPECIALISED ON THE WHOLE MODEL AND THE MACHINE, LIKE EVERY OTHER BRIDGE. `M` gathers every shape a
/// `config.json` declares and `A` every fact the architecture fixes, each as a TYPE per constant — so a model fact
/// never has to become a runtime argument, which is the whole failure this threading exists to prevent.
pub fn lower<A: crate::consts::Arch>(
    terminated: &Terminated,
    inits: &RegisterInits,
    precisions: &dyn Fn(crate::expansion_tape::GroupPos) -> crate::packet::spr::Precision,
    // The work gate's per-op denominator — see `ModelRun::expected_macs`. A closure like `precisions`,
    // because the answer lives with the GROUP the caller holds.
    expected_macs: &dyn Fn(
        crate::expansion_tape::GroupPos,
    ) -> Option<crate::bridges::progir_to_senprog::DeclaredVolume>,
    cores: GroupCores,
    // Passed through to `progir_to_senprog::lower`, which is where the model is asked — see its own note.
    execute_too: bool,
    // The memory those programs address — see `progir_to_senprog::lower`.
    segments: Option<&[crate::islands::progir::SegmentSpan]>,
) -> Packets {
    let bound = terminated.programs();
    // ⛔ PARTITIONED ONCE FOR THE WHOLE GROUP, then read per (op, core) — see `bridges::progir_to_senprog::ProgramTape`. The
    // per-core loop below calls `progir_to_senprog::lower` `ops × cores` times, and it used to hand each call the whole tape
    // to filter.
    let tape = crate::bridges::progir_to_senprog::ProgramTape::of(bound);
    // ⛔⛔ AND A VALUE FOR A CORE OUTSIDE THE MASK IS UNUSED, NOT WRONG — this is where I asserted the opposite and
    // it was my assumption rather than dxp's rule. The register map's cores come from the TENSOR'S HBM FOLD and the
    // mask from the group's `numCoresUsed_`: two different quantities, and a fold may name more cores than a
    // program runs on. dxp loops `initcores` and looks each core's `regState_` up WITHIN it
    // (`dip.cpp:1327`, `if (psInfo.regState_.count(sencomp))`), so it never asks about a core outside the set and
    // tolerates a core inside it having nothing.
    //
    // ⭐ SO THE MASK DRIVES THE EMISSION AND THE MAP ANSWERS PER CORE, which is what the loop below does. The
    // refusal that used to stand here failed a 4-core program whose tensor fold covered more.
    // ⛔⛔ ONE PROGRAM PER OP, WHICH IS WHAT THE IMAGE IS A CONCATENATION OF. `autopilot` walks `sdscNodes` and
    // merges each node's `prog_frame_ptr_` in order, checking that each begins where the last ended and setting the
    // sentinel bit on the LAST program alone (`dxp.cpp:836-870`) — and a node is one `sdsc_N.json`, which is one op.
    // So a group of N ops is N programs, each with its own header, SPR flit and register-init region.
    //
    // ⛔ AND THE SPR FLIT IS WHY IT CANNOT BE ONE PROGRAM PER GROUP: `fillSPRInfoPsinfo` reads one `precisionOfOp`
    // per program (`dip.cpp:1467`), and a group mixing an fp8 matmul with fp16 pointwise ops states two.
    let mut boundaries: Vec<crate::expansion_tape::GroupPos> = Vec::new();
    for op in bound {
        if boundaries.last() != Some(&op.at) {
            boundaries.push(op.at);
        }
    }
    // ⭐⭐ ONE INDEX FOR THE WHOLE LAYOUT. Each program used to be built by FILTERING THE ENTIRE TAPE, inside a
    // loop over programs and streams — `programs x streams x tape`, and it encoded every instruction it visited.
    // On a 512-op group that is ~53M filter-and-encode evaluations, which is where `i3::to_senprog`'s 897 s
    // (worst group 288 s) went. `StreamIndex` already holds exactly this partition, built in one pass.
    let index = StreamIndex::of(bound);
    let positions = |at: crate::expansion_tape::GroupPos, which: usize| -> &[usize] {
        index
            .per
            .get(&(at, which))
            .map_or(&[][..], |found| found.as_slice())
    };
    // ⛔⛔⛔ THE CORES OF AN **OP**, NOT OF THE GROUP — and ONE predicate answers it, so no second reading of
    // `cores` can disagree. A group's mask is every core any of its ops uses; an op runs on its OWN cores
    // (`islands::superdsc::Cores`, recorded per op by `reginit::addresses` and refused when absent). A block emitted
    // for a core an op does not run on writes that op's program into that core's IBUFF, and `convertIr2Senprog`
    // DT_ERRORs on a core absent from `progstateinfo_` (`sys-arch-spec/dpc/dpc.cpp:627-628`).
    //
    // ⭐ MEASURED WITH IBM'S OWN TOOLS on `prefill_mq7/e6c7cdd8605ed357/group_0` op 2: its senprog names SEVEN
    // cores and `dip_standalone` compiles it to 57 flits / 7 cores with programs, where this emitted 225 flits /
    // 28 cores — twenty-one cores initialised for an op that never runs on them.
    let mut packets: Packets = Packets::with_capacity(boundaries.len());
    for (position, at) in boundaries.iter().enumerate() {
        let running = OpCores::of(cores, *at, inits);
        // The core island 4 is built for: its instructions serve every core THIS OP runs on, and its register
        // VALUES are that op's first core's — not the group's, which the op may not run on at all.
        let first_core = running.first();
        let first_placed = crate::reginit::Core::of(first_core.get());
        let mut blocks: Vec<UnitBlock> = Vec::new();
        // ⛔⛔ EVERY PT ROW HOLDS THE SAME NUMBER OF INSTRUCTIONS, and it is the LONGEST row's.
        // `dip.cpp:1222-1257` pads every row to a multiple of four, takes `maxInstrCountC0` across the rows, and
        // then appends `OpCodeT::RETURN` instructions to each until they match — its own comment is "To make all
        // PT rows have same number of instructions". The PT is a systolic array whose rows step together, so rows
        // of different lengths give it a ragged grid where dxp gives a rectangular one.
        //
        // ⭐ THE MAX IS PER CORELET (`maxInstrCountC0`/`C1`, one per `getSenComponent(ptStr, cl)`), and this crate
        // emits one PT program per row addressed to BOTH corelets — dxp's own merged case, "if the initpacket for
        // both corelets are identical, then send it only once" (`:2072-2074`). So the max across rows IS the max
        // across a corelet's rows.
        // ⛔⛔⛔ THE IMAGE IS BUILT FROM ISLAND 4. This used to encode the island-3 tape directly while island 4 was
        // built beside it and serialized only for `dip_standalone` — two independent paths from one tape, so a
        // senprog dip accepted and an image the card rejected could diverge with nothing to notice. One path now:
        // `SenProg` in, flits out.
        //
        // ⭐ ONE CORE'S PROGRAMS SUFFICE FOR THE INSTRUCTIONS. Every core of a program runs the SAME instructions
        // and differs only in what its registers hold (`convertIr2Senprog`'s outer loop is over cores,
        // `dpc.cpp:625-631`), which is why the IBUFF is built once and the block repeated per core below.
        let senprog = crate::bridges::progir_to_senprog::lower::<A>(
            &tape,
            inits,
            first_core,
            first_placed,
            *at,
            crate::bridges::progir_to_senprog::ModelRun {
                execute_too,
                segments,
                precision: precisions(*at),
                expected_macs: expected_macs(*at),
            },
        );
        // ⛔⛔⛔ MEASURED OFF THE PROGRAMS BEING PADDED, NOT OFF THE TAPE.
        //
        // This counted TAPE POSITIONS — `positions(*at, which).len()` — and handed the result to
        // `Vec::resize` on ISLAND 4's instruction list. The two are different quantities and island
        // 4's is legitimately the larger: `islands::senprog::insert_exposed_pipeline_nops` inserts the
        // exposed-pipeline hazard covers (`SentientOps.cpp:40-59`), and `terminate_programs`
        // appends a RETURN. `resize` to the smaller number TRUNCATED every one of them back off.
        // MEASURED, granite-3.1-2b fp8: island 4 gave 55 instructions for `PtRow(0)` at
        // `GroupPos(0)` and the tape count was 37 — a third of the row replaced by RETURN padding,
        // in the first op of the first group, on every PT row of every group.
        //
        // ⛔ INVISIBLE TO EVERY GATE THIS CRATE OWNS, because they all read the SENPROG and the
        // senprog is correct — island 4 has the instructions. The loss is in the PACKET, and only
        // the dip differential (`DT_DIP_STANDALONE`) sees that far: dip builds 36 flits per op from
        // our own senprog where we built 32.
        let pt_row_instrs = Stream::ALL
            .iter()
            .filter(|stream| matches!(stream, Stream::PtRow(_)))
            .map(|stream| {
                senprog
                    .programs()
                    .iter()
                    .filter(|(unit, _)| stream.is_unit(*unit))
                    .map(|(_, held)| held.instructions.len())
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0);
        for (which, stream) in Stream::ALL.into_iter().enumerate() {
            let comp = stream.component();
            // ⛔⛔⛔ AN L3SU `STZ` WOULD MAKE THE QGI HEADER'S BIT 23 A DERIVED VALUE, AND IT IS EMITTED AS
            // `false`. `STZ_data_decision_reqd` has exactly one writer — `dip.cpp:3347-3354`, an L3SU
            // instruction whose opcode is `STZ` and whose `ibr` reads 0 or `"none"` — and what it tells the
            // queue generator is *"redirect program fetch from DDR to boot_strap + STZ data_message(29:0);
            // load-state-machine paused until STZ data message is received by QGI"*. A program that sends such
            // a message without setting the bit leaves the QGI fetching from the wrong place.
            //
            // ⭐ `STZ` IS A REAL VARIANT OF `InstOpCode` — one of 89 — so the type system does NOT forbid it;
            // only the vendored templates happen not to use it (it appears in none of the 36 `.smc` or 32
            // `.ddl`). That makes this a REFUSAL rather than a comment: the day a template carries one, the
            // bake stops here instead of emitting a header that lies about it.
            for (unit, held) in senprog.programs() {
                if !stream.is_unit(*unit) || comp != crate::isa::regfile::Component::L3su {
                    continue;
                }
                assert!(
                    !held
                        .instructions
                        .iter()
                        .any(|instr| instr.opcode == crate::isa::InstOpCode::STZ),
                    "the L3SU program at {at:?} contains an STZ, whose `ibr` decides the QGI header's \
                     `STZ_data_decision_reqd` (`dip.cpp:3347-3354`) — this crate emits that bit as false and \
                     has not ported the test, so the queue generator would be told to fetch from DDR while the \
                     program redirects it"
                );
            }
            let t_i5 = std::time::Instant::now();
            let program: Vec<crate::generated::InstrWord> = senprog
                .programs()
                .iter()
                .filter(|(unit, _)| stream.is_unit(*unit))
                .flat_map(|(_, held)| held.instructions.iter())
                .map(|instr| word_of(instr, comp))
                .collect();
            // ⛔ NOTHING MAY BE LOST CROSSING THE BRIDGE. The tape's own partition for this (program, stream) is
            // what island 4 was built from, so a different count means the two disagree about which ops belong to
            // this unit — and a dropped instruction is invisible in a well-formed packet.
            assert!(
                program.len() >= positions(*at, which).len(),
                "island 4 gives {} instruction(s) for {:?} on {:?} where the tape has {} — the bridge is losing \
                 instructions",
                program.len(),
                at,
                stream,
                positions(*at, which).len()
            );
            if program.is_empty() {
                continue;
            }
            // ⭐ THE EQUALIZATION IS ON THE INSTRUCTION LIST, BEFORE ANY PACKET SPLIT — which is where the C++ does
            // it (`dip.cpp:1239-1257`, long before `generatePTInitPacket`). Padding each CHUNK instead would pad it
            // to the whole program's maximum, and a chunk is bounded by `IbuffFlitCount::MAX` because that count is
            // a SEVEN-BIT header field (`initpacket.cpp:224`) — so the pad would overflow it.
            let program = match stream {
                Stream::PtRow(_) => {
                    let mut equalized = program;
                    // `inst.instn_ = OpCodeT::RETURN; inst.instFields_.clear();` (`dip.cpp:1252-1254`) — the same
                    // fieldless RETURN `terminate_programs` appends, encoded for this unit.
                    //
                    // ⛔ THROUGH `word_of`, NOT THROUGH A SECOND ENCODER. `PhysicalOp::encode` assembles the same
                    // word by the same rule, which is precisely the problem this file warns about above: two
                    // independent paths from one tape can disagree with nothing to notice. The pad is built as an
                    // island-4 instruction and encoded like every other one, so the position, overlap and virtual
                    // -field refusals apply to it too.
                    let pad = word_of(
                        &crate::islands::senprog::instruction(
                            comp,
                            crate::isa::InstOpCode::RETURN,
                            &crate::generated::PhysicalOp::RETURN(crate::generated::PhysRETURN {})
                                .fields(comp),
                        ),
                        comp,
                    );
                    // ⛔⛔⛔ EQUALIZATION PADS. IT MAY NOT TRUNCATE.
                    //
                    // `Vec::resize` shortens as readily as it lengthens, and the bound handed to it
                    // is a count of TAPE POSITIONS while the thing being resized is ISLAND 4's
                    // program — which is legitimately LONGER, because
                    // `islands::senprog::insert_exposed_pipeline_nops` inserts the exposed-pipeline
                    // hazard covers the reference inserts too. Every one of those covers was then
                    // cut back off here and replaced by RETURN padding, which is exactly the loss
                    // the assertion thirty lines above forbids in words ("NOTHING MAY BE LOST
                    // CROSSING THE BRIDGE ... a dropped instruction is invisible in a well-formed
                    // packet") and which nothing checked, because every gate this crate owns reads
                    // the SENPROG and the senprog is correct — the loss is in the PACKET.
                    assert!(
                        equalized.len() <= pt_row_instrs,
                        "the PT row equalization would TRUNCATE: island 4 gives {} instructions for \
                         {stream:?} at {at:?} and the pad length is {pt_row_instrs}. Equalization \
                         exists to make every row the same length by APPENDING RETURNs \
                         (`dip.cpp:1239-1257`); a shorter target means the length was measured off \
                         a different quantity than the one being padded",
                        equalized.len()
                    );
                    equalized.resize(pt_row_instrs, pad);
                    equalized
                }
                Stream::Whole(_) => program,
            };
            let width = InstrWidth::of_unit(comp);
            // ⛔⛔ EVERY UNIT'S SPR0 IS WRITTEN, because an all-zero SPR flit configures NOTHING: the low byte is
            // the MASK (`dip.cpp:1409-1411`), so a zero word leaves the unit exactly as it was.
            let spr = crate::packet::spr::of_unit(comp, precisions(*at)).slice();
            // ⛔⛔ ONE BLOCK PER CORE, ADDRESSED TO THAT CORE'S MASK, AND THE CORES ARE THE PROGRAM'S.
            // `initcores` is the key set of `progstateinfo` — the cores that have a PROGRAM (`dip.cpp:134-136`,
            // again at `:1153-1155`) — and every per-core loop in `dip` walks it, including the register-init
            // census (`:1327`) and the packet generation (`:1665`). A register value is one thing a core's program
            // HAS, so taking the set from the register inits inverts the dependency: a program with no address
            // register then has no cores, and emits no blocks at all.
            //
            // ⭐ AND THE REGISTER VALUES REALLY DO DIFFER PER CORE — measured, tensor `t0` at `0xc00000000` on one
            // core and `0xc00000080` on another, one stick apart, which is the work split. So the IBUFF is built
            // once per chunk and the block repeated per core with that core's own LRF region.
            // ⭐⭐ THE IBUFF DOES NOT DEPEND ON THE CORE, SO IT IS BUILT ONCE. `ibuff_of` takes only the width,
            // the component and the instruction words — every core of a program runs the SAME instructions, and
            // only the LRF region differs (the per-core HBM fold). Rebuilding it inside the core loop did the
            // same work 32 times per program per stream.
            let by_field = IbuffFlitCount::MAX * width.instrs_per_slice().get();
            let by_buffer = usize::from(crate::isa::regfile::max_ibuff_entries(comp));
            let chunked: Vec<Ibuff> = program
                .chunks(by_field.min(by_buffer))
                .map(|chunk| ibuff_of(width, comp, chunk))
                .collect();
            // ⛔⛔ THE UNIQUE-VALUE CENSUS IS TAKEN ONCE, OUTSIDE THE CORE LOOP, because the LX/L0 header's index
            // ORDER is a fact about ALL the cores at once (`LXL0LRFInitCompression`, `dip.cpp:1314-1400`). Taking
            // it per core would make every count one and leave the list merely ascending.
            let emitted: Vec<crate::reginit::Core> = running
                .each()
                .map(|core| crate::reginit::Core::of(core.get()))
                .collect();
            // ⛔ THROUGH BRIDGE 3, NOT STRAIGHT OUT OF `inits`. The set of registers the packet names must be the
            // set the senprog states, and the second arm's explicit zeros are part of it —
            // `crate::bridges::progir_to_senprog::lrf_named` is the one place that question is answered.
            i5_timed(0, t_i5);
            let t_i5 = std::time::Instant::now();
            let unique = crate::bridges::progir_to_senprog::lrf_named(bound, inits, comp, *at, &emitted);
            i5_timed(1, t_i5);
            let t_i5 = std::time::Instant::now();
            // ⛔⛔⛔ ONE BLOCK PER CORE. This loop used to group cores by LRF-region equality and address ONE
            // block to all of them, on the stated grounds that `target_core_mask` is a SET and that this was
            // `patchCoreGroups`' own test. Both halves were wrong: `Dip::isPatchInit` reads a mask naming more
            // than one core as a PATCH-INIT image BY DEFINITION (`patchinit.cpp:1268-1269`), so the shared block
            // did not make the image smaller — it made it a patch-init image with no patch headers, which IBM's
            // own reader refuses (`updateLxAddress`'s `DT_CHECK(check.size() == 1)`, `patchinit.cpp:31`) and the
            // card answers with a PCIe bus-master fence. See [`crate::packet::header::CoreMask::of_one`].
            //
            // ⭐ THE SIZE WIN IS REAL AND IS A DIFFERENT ENCODING. dxp emits 23,933 flits for the group this
            // emits ~272,960 for, and it gets there with `constructPatchInit` (`patchinit.cpp:200-941`) — patch
            // headers plus LX bookkeeping. Porting that is what buys the compression back; a multi-core mask on
            // a regular block never could.
            for core in &emitted {
                let lrf = &lrf_region(*at, comp, *core, inits, &unique);
                let addressed = crate::packet::header::CoreIdx::of_index(
                    crate::packet::header::Bits::exactly(core.get()),
                );
                // ⛔⛔ A BLOCK IS BOUNDED BY THE UNIT'S IBUFF, NOT ONLY BY THE HEADER'S FIELD.
                // `IbuffFlitCount::MAX` is 127 because the flit count is seven bits (`initpacket.cpp:224`) — 508
                // instructions for a 32-bit unit — while the unit's buffer holds `maxIBuffEntriesPerUnit`: 256 for
                // an L3 unit and 128 for every other on this arch (`sysdef.cpp:425-475`). Chunking on the field
                // alone writes a 129-instruction compute program into a 128-entry buffer, which nothing in the
                // packet can report — dxp's own check is
                // `DT_CHECK_MSG(progInsts.size() < maxIBuffEntriesPerUnit.at(comp), "Progstitch exceeded max
                // ibuff")` (`progtailor.cpp:784-786`).
                //
                // ⭐ SO THE BLOCK SIZE IS THE SMALLER OF THE TWO, and a program longer than one block becomes
                // several — which is what this loop already does for the field's bound (`09b561c8`).
                for ibuff in &chunked {
                    let header = Header {
                        registers: lrf.index_list(),
                        lrf_flits: lrf.flit_count(),
                        // ⭐ THE PADDED COUNT, because the row equalization happens BEFORE
                        // `ptHeader.pktIbuff_flit = pktIbuffFlits` is read (`dip.cpp:1239-1257` then `:4665`) —
                        // so every PT row declares the same length, and every other unit declares its own.
                        instr_flits: InstrFlitCount::of_flits(
                            crate::packet::header::Bits::exactly(
                                u32::try_from(ibuff.flit_count().get())
                                    .expect("a flit count that fits"),
                            ),
                        ),
                        target: target_of(comp),
                        core: addressed,
                    };
                    blocks.push(match stream {
                        Stream::Whole(_) => UnitBlock::of_unit(header, spr, lrf, ibuff),
                        Stream::PtRow(row) => UnitBlock::of_pt_row(
                            crate::packet::block::PtRow::of(row as usize),
                            header,
                            spr,
                            lrf,
                            ibuff,
                        ),
                    });
                }
            }
            i5_timed(2, t_i5);
        }
        // ⛔⛔ A PROGRAM WITH NO BLOCKS IS NOT A PROGRAM, AND IT IS INDISTINGUISHABLE FROM ONE. Nothing said so, and
        // an image of 79 QGI headers with EMPTY BODIES round-tripped through the packet chain, the sentinel check
        // and `serialise`'s walk — every one of them agreeing, because a header whose `myflits` is 0 is perfectly
        // consistent with a body of no flits. The card reported it as `0x7b1b`, a rejected control block, which is
        // three layers away from the cause: block emission had been put inside a loop over the cores that hold an
        // ADDRESS REGISTER, so a program with none emitted nothing at all.
        //
        // ⭐ THE BOUNDARY IS ALREADY PROOF THAT WORK EXISTS. `boundaries` comes from the ops themselves, so this
        // position exists BECAUSE an op names it — and every op belongs to exactly one unit, so at least one
        // stream must have produced a block. This asserts the consequence of a fact the loop already had.
        assert!(
            !blocks.is_empty(),
            "the program at {at:?} carries no unit block, and an op boundary exists only because an op names it — \
             so some stream's instructions were dropped between island 3 and here"
        );
        // ⛔⛔⛔ PATCH INIT, WHICH IS WHAT dxp EMITS. `lay_blocks` gives the REGULAR init — one block per core
        // — and `Dip::doPatchInit` (`patchinit.cpp:186-198`) is the transform the card is given: a shared base
        // under the group's whole core mask plus one patch flit per core that disagrees with it. MEASURED:
        // `dip_standalone -p` on this crate's own senprog emits 35 flits where the regular init is 225, and
        // dxp's own golden for a 1-op group is 40 flits carrying a patch header and `0xffffffff` masks.
        let flits = crate::packet::patch::patch_init(&blocks);
        packets.push(Packet {
            header: QgiHeader {
                myflits: crate::islands::init_packet::FlitCount::of(0),
                // ⛔⛔⛔ THESE FOUR ARE THE C++'s OWN DECLARED VALUES, and each was checked to its source
                // rather than copied from a dump. `USM_pause = true` and `LSM_pause = false` (`dip.h:219-220`)
                // are NEVER ASSIGNED ANYWHERE ELSE in dip — grep is the proof, not the default. `hoststall`
                // starts at 0 (`:229`) and moves only for the `DIP_HOSTSTALL` env var (`:314-318`).
                usm_pause: true,
                lsm_pause: false,
                hoststall: false,
                // ⛔⛔ `STZ_data_decision_reqd` STARTS AT 0 (`dip.h:223`) AND HAS EXACTLY ONE WRITER:
                // `dip.cpp:3352`, when an L3SU instruction is an `STZ` whose `ibr` reads 0 or "none"
                // (`:3347-3354`). Its meaning is the QGI's — *"redirect program fetch from DDR to boot_strap +
                // STZ data_message(29:0); load-state-machine paused until STZ data message is received by
                // QGI"* — so a program that sends one and does not say so leaves the queue generator fetching
                // from the wrong place.
                //
                // ⭐ IT IS FALSE HERE BECAUSE THE OPCODE IS UNREACHABLE: `STZ` appears in NONE of the 36 `.smc`
                // or 32 `.ddl` vendored templates, so this crate cannot emit one. ⛔ IF A TEMPLATE EVER
                // CARRIES AN L3SU `STZ`, THIS MUST BE DERIVED FROM THE INSTRUCTION STREAM AND NOT LEFT FALSE.
                stz: false,
                // ⛔⛔ THE SENTINEL IS THE LAST PROGRAM'S ALONE — `isSentinel = lastProg` (`dxp.cpp:855-862`). It is
                // what ends the compute control block, so setting it on every program would end the block after the
                // first op, and setting it on none would never end it.
                sentinel_cb: position + 1 == boundaries.len(),
            },
            flits,
        });
    }
    packets
}

/// ONE UNIT'S REGISTER-INIT REGION — which of the three LRF layouts this unit's block carries.
///
/// ⭐ THE CALLER PICKS THE LAYOUT, WHICH IS WHY THE DISPATCH IS HERE AND NOT IN `packet::lrf`. That module's three
/// packers differ in LANE WIDTH, and a lane width is not a parameter of a bit layout — so the one place that knows
/// which unit a block belongs to is the one place that may choose between them.
///
/// ⛔⛔ AND A TEMPLATE CONSTANT IS NOT IN HERE, WHICH IS THE WHOLE POINT OF SAYING SO. A constant reaches a compute
/// unit's LRF by a TRANSFER: `%src_const_46DC = ddl.unit(%const_46DC) {unit="constant"}` and
/// `ddl.data_transfer(%src_const_46DC, [%dst_expconst1_sfp])` (`unary_parallel.ddl:180-182`), lowered by
/// `SNTransferLowering.cpp:2154-2164` into a `ConstantBitstreamOp` of ONE value plus a splat `ShuffleOp`
/// (`GenerateConstantBitStreamAndShuffle`, `:2444-2484`) whose result is stored into the register. So the value
/// arrives as INSTRUCTIONS at its point of use — which is also why bridge 2 may reuse one register for two
/// constants with disjoint live ranges.
///
/// ⛔ WHAT DOES BELONG HERE IS THE ADDRESSES AND THE SPR, and no pass computes them yet: an LX or L0 load reads its
/// address out of a register whose initial value is the allocation's placed base plus the affine constant term
/// (`AgenToSentient/Helper.cpp:762-822`), and the L3 units' LAR/EAR pairs address LX and HBM.
fn lrf_region(
    at: crate::expansion_tape::GroupPos,
    comp: Component,
    core: crate::reginit::Core,
    inits: &RegisterInits,
    // ⛔ THE INDEX ORDER IS A CROSS-CORE FACT, so the census cannot be taken inside a per-core call — see
    // `crate::reginit::RegisterInits::lrf_unique_counts`.
    unique: &std::collections::BTreeMap<crate::islands::progir::RegIndex, usize>,
) -> LrfRegion {
    use crate::packet::lrf::{L0LrfValue, LxLrfValue, l0_lrf_region, lx_lrf_region};
    let held = inits.addresses_of(at, core, comp);
    let staged = inits.l3_addresses_of(at, core, comp);
    match comp {
        // ⛔ AN LX ADDRESS REGISTER IS 24 BITS OF BYTES — three bytes of the flit slice, and 21 of them reachable
        // by `IMMCOPY`'s `lrfimm`, which is the width of the whole 2 MiB part.
        Component::Lxlu | Component::Lxsu => {
            let values: Vec<(crate::islands::progir::RegIndex, LxLrfValue)> = held
                .iter()
                .map(|(reg, base)| {
                    (
                        *reg,
                        LxLrfValue::of(u32::try_from(base.get()).unwrap_or_else(|_| {
                            panic!(
                                "an LX base of {} bytes does not fit an address register (`dip.cpp:3342-3394`)",
                                base.get()
                            )
                        })),
                    )
                })
                .collect();
            // ⛔ NO MVR VALUES, AND THE MAP'S SHAPE IS WHY — not a convention that could quietly lapse.
            // `RegisterInits::addresses` is keyed `(GroupPos, Core, Component, RegIndex)` with NO `RegType`
            // (`reginit.rs:217`), unlike the `l3` map beside it which carries one; so it holds ADDRESS
            // registers alone and an `Mvr` cannot be stored there to be dropped here.
            //
            // ⭐ AND THE EMPTY SLICE IS WHAT THE C++ COMPUTES TOO. `minLrfForMvrPacking` starts at 0 and is
            // raised only for LXLU0/LXLU1 from `maxMvrIdxUsed` (`dip.cpp:1365-1382`); with no MVR init that
            // index stays -1, `reqdMvrFlits = ceil(-0.5) = 0`, and the floor is 0 — so
            // `max(minLrfForMvrPacking, minLrfForLrfPacking)` is the LRF term alone, which is what
            // `lx_l0_index_list` computes from a zero `mvr_indices`.
            lx_lrf_region(&values, &[], unique)
        }
        // ⛔ AN L0 ADDRESS REGISTER IS TEN BITS on this arch (`dip.cpp:3676-3679`), which is why the base is
        // divided by that unit's granularity before it gets here — `numPTRows` for the L0SU (`sysdef.cpp:514`).
        Component::L0lu | Component::L0su => {
            let values: Vec<(crate::islands::progir::RegIndex, L0LrfValue)> = held
                .iter()
                .map(|(reg, base)| {
                    (
                        *reg,
                        L0LrfValue::of(u16::try_from(base.get()).unwrap_or_else(|_| {
                            panic!(
                                "an L0 base of {} address units does not fit a ten-bit register \
                                 (`dip.cpp:3676-3679`)",
                                base.get()
                            )
                        })),
                    )
                })
                .collect();
            l0_lrf_region(&values, unique)
        }
        // ⛔⛔ THE L3'S ADDRESS FILES, IN ITS OWN TEN-PACKET LAYOUT (`dip.cpp:2672-2858`). Left empty, the staging
        // transfer this unit runs addresses HBM 0 and LX 0 — a real instruction that moves nothing, which is what
        // the card reported as a launch that never completes.
        //
        // ⭐ THE VALUE'S FILE IS THE VARIANT, so a bound cannot be filed as an address: `L3AddrInit::Ear` takes an
        // `EarValue` and `Ebr` an `EbrValue`, and the two differ in the width of their high lane.
        Component::L3lu | Component::L3su => {
            use crate::isa::regfile::RegType;
            use crate::packet::lrf::{Addr14, EarValue, EbrValue, L3AddrInit, L3Lrf};
            let mut inits = L3Lrf::default();
            for (file, reg, value) in staged {
                let units = u32::try_from(value.get()).unwrap_or_else(|_| {
                    panic!(
                        "an L3 address of {} units does not fit any of its registers",
                        value.get()
                    )
                });
                let narrow = || {
                    u16::try_from(units).unwrap_or_else(|_| {
                        panic!(
                            "{file:?}[{}] holds 14 bits and this address is {units} of the L3's 128-byte units \
                             (`dip.cpp:2672-2858`, `sysdef.cpp:506`)",
                            reg.get()
                        )
                    })
                };
                let init = match file {
                    RegType::Lar => L3AddrInit::Lar(Addr14::of(narrow())),
                    RegType::Lbr => L3AddrInit::Lbr(Addr14::of(narrow())),
                    RegType::Gtr => L3AddrInit::Gtr(Addr14::of(narrow())),
                    RegType::Ear => L3AddrInit::Ear(EarValue::of(units)),
                    RegType::Ebr => L3AddrInit::Ebr(EbrValue::of(units)),
                    other => panic!(
                        "{other:?} is not one of the L3's address files: only LAR, LBR, GTR, EAR and EBR have a \
                         place in its init packets (`dip.cpp:2672-2858`)"
                    ),
                };
                inits.set(reg, init);
            }
            inits.build()
        }
        // ⛔ THE COMPUTE UNITS' LRF HOLDS CONSTANTS, NOT ADDRESSES, and a constant arrives as an `IMMCOPY`
        // instruction rather than in the packet (`ConstructSplatPadInstr`, `ConstructProgIRHelper.cpp:3818-3828`).
        Component::Sfp | Component::Pe | Component::Pt => LrfRegion::default(),
    }
}

/// The encoded `RETURN` a unit pads its IBUFF with. Its opcode value differs per component
/// (`isaSystemc.h`), so the pad is looked up rather than shared.
fn return_word(comp: Component) -> u32 {
    use crate::isa::values::OpUnit;
    u32::from(
        OpUnit::of_component(comp)
            .value_of(crate::isa::InstOpCode::RETURN)
            .get(),
    )
}

/// ONE UNIT'S IBUFF, PACKED AT THAT UNIT'S OWN INSTRUCTION WIDTH.
///
/// ⛔⛔ A PE OR SFP INSTRUCTION IS 64 BITS AND OCCUPIES TWO WORDS — `initslicepart[j]` holds its LOW half and
/// `[j+1]` its HIGH half, which is exactly how dxp reads them back for `slice_id == 4 || slice_id == 5`:
/// `instr.binary_ = ((tmp_instr << 32) & 0xFFFFFFFF00000000) | initslicepart.at(j)`, stepping `j += 2`
/// (`dip.cpp:651-661`). Truncating the encoded word to 32 bits drops every wide instruction's upper half AND
/// puts two instructions in the space of one, which is why dxp's own disassembler reads our SFP program as
/// `be:260 dyn_loop:16` on one-bit and two-state fields. A stream whose block end lands inside another
/// instruction never terminates: that is the `0x1acc RbTimeOut`.
///
/// ⭐ AND THE RETURN PAD IS THE UNIT'S TOO. A 32-bit pad in a wide tail fills only the low half of each padded
/// instruction, leaving the high half holding whatever it held.
fn ibuff_of(width: InstrWidth, comp: Component, chunk: &[crate::generated::InstrWord]) -> Ibuff {
    use crate::packet::ibuff::{ReturnPad64, Word64, ibuff_slices_64};
    match width {
        InstrWidth::Bits64 => {
            let program: Vec<Word64> = chunk.iter().map(|instr| Word64::new(instr.get())).collect();
            ibuff_slices_64(&program, ReturnPad64::of_unit(comp))
        }
        InstrWidth::Bits32 => {
            let program: Vec<Word32> = chunk
                .iter()
                .map(|instr| {
                    let bits = instr.get();
                    // ⛔ THE SAME DEFECT IN THE OTHER DIRECTION. One `InstrWord` serves every unit, so a value
                    // wider than the stream it lands in would be silently halved rather than refused.
                    assert!(
                        bits >> 32 == 0,
                        "{comp:?} holds 32-bit instructions (`dip.cpp:611-613`) and this one encodes to \
                         {bits:#018x}: the upper half has nowhere to go"
                    );
                    Word32::new(bits as u32)
                })
                .collect();
            Ibuff::of_words(width, &program, Word32::new(return_word(comp)))
        }
    }
}

/// Which unit a block's header names. A corelet-resident unit names both corelets; an L3 unit is a bare id.
fn target_of(comp: Component) -> Target {
    match comp {
        Component::Pt => Target::OnCorelets {
            unit: CoreletUnit::Pt,
            corelets: Corelets::Both,
        },
        Component::Pe => Target::OnCorelets {
            unit: CoreletUnit::Pe,
            corelets: Corelets::Both,
        },
        Component::Sfp => Target::OnCorelets {
            unit: CoreletUnit::Sfp,
            corelets: Corelets::Both,
        },
        Component::L0lu => Target::OnCorelets {
            unit: CoreletUnit::L0Lu,
            corelets: Corelets::Both,
        },
        Component::L0su => Target::OnCorelets {
            unit: CoreletUnit::L0Su,
            corelets: Corelets::Both,
        },
        Component::Lxlu => Target::OnCorelets {
            unit: CoreletUnit::LxLu,
            corelets: Corelets::Both,
        },
        Component::Lxsu => Target::OnCorelets {
            unit: CoreletUnit::LxSu,
            corelets: Corelets::Both,
        },
        Component::L3lu => Target::L3(L3Unit::L3Lu),
        Component::L3su => Target::L3(L3Unit::L3Su),
    }
}

/// FORM THE BLOCKS — set `be` on the last instruction of every unit's program.
///
/// ⭐ THIS IS `updateLabelAndAddToCodeGraph`'s FUSION, PORTED (`LowerSentientHelper.cpp:46-85`). The C++ emits a
/// branch exit as a `NOP` carrying `be` (`ConstructBranchExitInstr`, `ConstructProgIRHelper.cpp:307-315`), and then
/// MERGES that flag onto the instruction before it — "This reduces code size and also cycle count" — when three
/// things hold: the exit carries no tag, the previous instruction has no `be` of its own, and that instruction is
/// not `MVLOOPCNT`.
///
/// ⛔ SO THE FLAG GOES ON THE LAST INSTRUCTION, AND ONLY WHERE THE MERGE IS LEGAL. `MVLOOPCNT` is excluded by name
/// in the C++ (`:75-76`) — it sets a loop count and the hardware reads its `be` position differently — so a program
/// ending in one cannot absorb the flag and needs the `NOP` the C++ would then have emitted.
///
/// ⛔ AND IT IS PER UNIT, because a block is a unit's. Each component's program is a FILTER over the tape, so the
/// last instruction OF THAT FILTER is the one that ends its block; taking the tape's last instruction would end one
/// block and leave every other unit hanging.
/// EVERY (PROGRAM, STREAM)'s INSTRUCTION POSITIONS, BUILT IN ONE PASS.
///
/// ⛔⛔⛔ WITHOUT THIS, EVERY PASS OVER THE TAPE IS `programs x streams x tape` — AND THE BUILD SCRIPT IS
/// UNOPTIMISED. Four passes each rebuilt a program list with `programs.contains(&op.at)` (itself `O(n·p)`) and
/// then re-scanned the WHOLE tape once or twice per (program, stream). On a 174-program group that is ~2800
/// tape walks per pass, in a debug-built `build.rs`, for each of 577 groups. Measured twice in `top`: 22 minutes
/// at 99% CPU, then 19 minutes at 199%.
///
/// ⭐ SO THE SHAPE IS: index once, then every pass is linear in what it actually touches. A pass that needs "the
/// last instruction of this program on this stream" reads it from here; one that needs a count folds over the
/// positions. Nothing re-derives the partition, so no two passes can disagree about it either.
pub(crate) struct StreamIndex {
    /// Tape positions per (program, stream index into [`Stream::ALL`]), ascending — so `.last()` IS the
    /// program's last instruction on that stream.
    pub(crate) per:
        std::collections::BTreeMap<(crate::expansion_tape::GroupPos, usize), Vec<usize>>,
}

impl StreamIndex {
    pub(crate) fn of(bound: &[crate::islands::OnUnit<crate::generated::PhysicalOp>]) -> Self {
        let mut per: std::collections::BTreeMap<
            (crate::expansion_tape::GroupPos, usize),
            Vec<usize>,
        > = std::collections::BTreeMap::new();
        for (at, op) in bound.iter().enumerate() {
            for (which, stream) in Stream::ALL.into_iter().enumerate() {
                if stream.runs(op.on) {
                    per.entry((op.at, which)).or_default().push(at);
                }
            }
        }
        Self { per }
    }
}

#[tracing::instrument(skip_all)]
fn form_blocks(bound: &Bound) -> Bound {
    // ⭐ EVERY DECISION IS TAKEN AGAINST THE ORIGINAL TAPE, THEN APPLIED ONCE. Each (program, stream) gets at
    // most one close, and a close for one program cannot change another's decision — so nothing has to be
    // recomputed as positions shift, and the whole pass is linear.
    let index = StreamIndex::of(bound);
    let mut stamp: Vec<usize> = Vec::new();
    let mut appended: Vec<(usize, crate::islands::OnUnit<crate::generated::PhysicalOp>)> =
        Vec::new();
    for ((at, _stream), positions) in &index.per {
        let Some(&last) = positions.last() else {
            continue;
        };
        // ⛔⛔⛔ THE COUNT IS THE PROGRAM'S LOOP REGIONS, AND A PROGRAM WITH NONE GETS NO BLOCK END AT ALL.
        // One block end per loop region is what `LowerYieldOperation` produces — a close per `YieldOp`
        // (`LowerSentientHelper.cpp:782-840`) — so a block end is a LOOP's close and never a program's
        // terminator. What terminates a program is the `RETURN` [`terminate_programs`] appends.
        //
        // ⛔⛔ MEASURED AGAINST dxp: across the 1564 unit-programs `CODEGEN_DUMP_IRS=1` dumps for this model,
        // `#MVLOOPCNT == #be` holds in every single one — and **1322 of them carry NO block end whatsoever**,
        // because they contain no loop. Forcing one onto a loopless program is a `be` with no counter to pop:
        // `MVLOOPCNT` has no LCCR field, the nesting IS an implicit hardware counter stack
        // (`isa.cpp:1074-1077`), and popping an empty stack sends the unit's PC somewhere nothing wrote. On an
        // L3 unit that is a DMA with an arbitrary EAR.
        //
        // ⭐ SO THE CONDITION IS THE BALANCE, NOT "IS THERE ONE YET". `bridges::a_superdsc_to_dataflow_ir`'s loop closes already stamp
        // a close per region, so a balanced program is FINISHED here; a loopless one wants nothing; and only a
        // program short of closes is completed.
        let (counters, closes) = positions.iter().fold((0usize, 0usize), |(c, b), &i| {
            (
                c + usize::from(crate::validate::is_loop_counter(&bound[i].kind)),
                b + usize::from(bound[i].kind.ends_block()),
            )
        });
        if closes >= counters {
            continue;
        }
        // ⭐⭐ MERGE WHERE THE C++ MERGES, AND EMIT THE `NOP` WHERE IT DOES NOT.
        // `updateLabelAndAddToCodeGraph` fuses the branch exit's `be` onto the previous instruction only when
        // that instruction has no `be` of its own and is not `MVLOOPCNT` (`LowerSentientHelper.cpp:70-81`);
        // otherwise the exit stays its own instruction — a `NOP` carrying `be`, which is what
        // `ConstructBranchExitInstr` builds (`ConstructProgIRHelper.cpp:307-315`).
        match bound[last].kind.can_end_block() {
            true => stamp.push(last),
            false => appended.push((
                last,
                crate::islands::OnUnit {
                    // ⭐ THE EXIT BELONGS TO THE OP IT CLOSES and runs on the SAME unit, since a block is a
                    // unit's.
                    at: *at,
                    on: bound[last].on,
                    kind: crate::generated::PhysicalOp::NOP(crate::generated::PhysNOP {
                        be: crate::isa::unit::BeOrUnstated::On,
                    }),
                },
            )),
        }
    }
    let mut out = bound.clone();
    for at in stamp {
        out[at].kind.end_block();
    }
    splice_after(out, appended)
}

/// INSERT EACH `(position, op)` DIRECTLY AFTER `position`, IN ONE PASS.
///
/// ⛔ `Vec::insert` IN A LOOP IS `O(n)` EACH AND SHIFTS EVERY LATER INDEX, so a caller then has to recompute the
/// positions it already knew. Building the new tape once keeps every position meaningful and the pass linear.
fn splice_after(
    tape: Bound,
    mut appended: Vec<(usize, crate::islands::OnUnit<crate::generated::PhysicalOp>)>,
) -> Bound {
    if appended.is_empty() {
        return tape;
    }
    appended.sort_by_key(|(at, _)| *at);
    let mut out: Bound = Bound::with_capacity(tape.len() + appended.len());
    let mut next = appended.into_iter().peekable();
    for (at, op) in tape.into_iter().enumerate() {
        out.push(op);
        while next.peek().is_some_and(|(after, _)| *after == at) {
            out.push(next.next().expect("peeked").1);
        }
    }
    // Anything addressed past the end would silently vanish, so it is a defect rather than a no-op.
    assert!(
        next.next().is_none(),
        "an appended instruction names a tape position that does not exist"
    );
    out
}

/// PORT OF `createFinalSenProg`'s TERMINATOR (`dcgbeCodegen.cpp:2613-2624`) — what every program ends with.
///
/// ⛔⛔ A RETURN PAD IS NOT A PROGRAM'S RETURN. `padInstructionsOfSenComp` appends pad RETURNs only when
/// `instrCount % 4 != 0` (`dip.cpp:318-328`), over a program `createFinalSenProg` has already terminated. A crate
/// that leaves the terminator to the pad has none at all whenever the instruction count is already a multiple of
/// four — the program runs past its last instruction into whatever its IBUFF held.
///
/// ⛔⛔ AND IT RUNS AFTER [`form_blocks`], NOT BEFORE. Nothing after a `RETURN` executes, and the C++ states that
/// directly: when `createFinalSenProg` appends code to a program that already ends in one, it REWRITES that
/// `RETURN` into a `NOP` first (`needToRemoveReturn`, `:2557-2571`). So a block end placed after the terminator is
/// a block that never ends — which is what terminating before block formation would produce, since a `RETURN`
/// states no `be` and [`form_blocks`] would append its exit `NOP` past it.
///
/// ⛔ THE PT TAKES THREE NOPs FIRST (`:2617-2621`), for the reason its own comment gives: *"add extra NOP for
/// PTrow: issue: XRF WB is missed in senulator"*. The PT's write-back has not landed when the RETURN retires, so
/// the pad is the machine's, not a style choice.
#[tracing::instrument(skip_all)]
fn terminate_programs(bound: &Bound) -> Terminated {
    // ⭐ ONE INDEX, ONE SPLICE. Each (program, stream) appends its pad NOPs and its `RETURN` after that
    // program's last instruction, and every position is taken from the ORIGINAL tape — so nothing shifts under
    // the walk and the pass is linear rather than `programs x streams x tape`.
    let index = StreamIndex::of(bound);
    let mut appended: Vec<(usize, crate::islands::OnUnit<crate::generated::PhysicalOp>)> =
        Vec::new();
    for ((at, which), positions) in &index.per {
        let Some(&last) = positions.last() else {
            continue;
        };
        let on = Stream::ALL[*which].executor();
        // `if (unitType == SenComponents::PT) for (int idx = 0; idx < 3; idx++)` (`:2617-2621`).
        for _ in 0..Stream::ALL[*which].trailing_nops().get() {
            appended.push((
                last,
                crate::islands::OnUnit {
                    at: *at,
                    on,
                    // `createInstr` clears every field (`:2618`), so the pad NOP closes no block.
                    kind: crate::generated::PhysicalOp::NOP(crate::generated::PhysNOP {
                        be: crate::isa::unit::BeOrUnstated::Unstated,
                    }),
                },
            ));
        }
        appended.push((
            last,
            crate::islands::OnUnit {
                at: *at,
                on,
                kind: crate::generated::PhysicalOp::RETURN(crate::generated::PhysRETURN {}),
            },
        ));
    }
    // ⭐ THE PADS PRECEDE THE `RETURN` BECAUSE THEY WERE PUSHED FIRST, and `splice_after` is stable for equal
    // positions — nothing may follow a `RETURN`, which is the whole point of [`Terminated`].
    Terminated(splice_after(bound.clone(), appended))
}

/// A TAPE WHOSE EVERY PROGRAM ENDS IN ITS `RETURN` — and therefore a tape nothing may be appended to.
///
/// ⛔⛔ THIS IS A TYPE AND NOT A CHECK BECAUSE A CHECK ALREADY PASSED ONCE WHILE THE TAPE WAS WRONG. A law over
/// the tape's CONTENTS cannot see WHERE a pass put something: a tape ending `… RETURN, NOP(be=1)` has a perfectly
/// balanced [`blocks_balance_loop_counters`] and still hangs the machine, because nothing after a `RETURN`
/// executes (`needToRemoveReturn`, `dcgbeCodegen.cpp:2557-2571`). Any pass that runs after termination
/// reintroduces exactly that shape.
///
/// ⭐ SO THE ONLY WAY PAST THIS POINT IS [`Self::programs`], which reads and never appends. A new pass must take
/// `&Bound` and be sequenced BEFORE [`terminate_programs`], or it does not compile.
pub struct Terminated(Bound);

impl Terminated {
    /// ⛔⛔⛔ THE TAPE'S LAWS ARE THE CONDITIONS OF THIS TYPE, NOT A PASS AFTER IT.
    ///
    /// [`crate::validate::tape`] is every condition decidable over island 3's finished tape, and it used to run
    /// in [`finished`] AFTER [`terminate_programs`] had already produced a `Terminated`. That left the validated
    /// tape and the merely-terminated one the SAME TYPE, so the checks were a step a caller performs rather than
    /// a property the value has — and a second minting site would have skipped them silently.
    ///
    /// ⭐ THE CHECKS BELONG HERE AND NOT EARLIER, which is what makes termination the right place to put them:
    /// block ends are placed and every program carries its `RETURN`, so the conditions that are about POSITION
    /// and LENGTH are meaningful for the first time.
    fn of(terminated: Terminated, inits: &RegisterInits) -> Terminated {
        crate::validate::tape(terminated.programs(), inits);
        terminated
    }

    /// The ops of one (program, stream), in order — a read, so no pass can grow the tape through it.
    pub fn programs(&self) -> &[crate::islands::OnUnit<crate::generated::PhysicalOp>] {
        &self.0
    }
}

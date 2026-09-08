//! THE BLOCKS AND THE REGIONS — `UniformInstrBlock` and `UniformInstrBlocks`: per-unit
//! instruction lists, the uniformised lists, region indices and the flattened index.
//!
//! ⛔ A REGION INDEX OF -1 IS "NOT PRESENT", and a REGULAR block always answers 0
//! (`UniformInstrAndBlock.hpp:208`). A sentinel is an `Option` here, never a negative number.
//!
//! 24 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e023_getUnitInstrList` | 0 | 12 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:266` |
//! | `e024_insertInstruction` | 0 | 5 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:322` |
//! | `e025_setCurrentRegion` | 0 | 4 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:327` |
//! | `e026_getMaxInstrSize` | 0 | 8 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:331` |
//! | `e027_getCurrentRegionInstrSize` | 0 | 5 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:360` |
//! | `e028_getInstr` | 0 | 8 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:365` |
//! | `e029_getLastInstr` | 0 | 3 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:374` |
//! | `e030_appendEmptyUniformRegion` | 0 | 4 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:378` |
//! | `e031_getMaxInstrSize` | 0 | 7 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:426` |
//! | `e032_appendRegularBlock` | 0 | 3 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:442` |
//! | `e033_appendUniformBlock` | 0 | 3 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:445` |
//! | `e034_getUniformInstr` | 0 | 6 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:449` |
//! | `e035_doesInstrWithThisIndexExist` | 0 | 7 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:456` |
//! | `e036_appendEmptyUniformRegion` | 0 | 4 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:497` |
//! | `e037_getUnitRegionIndex` | 0 | 6 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.hpp:208` |
//! | `e074_getMaxInstrRegionIndex` | 1 | 12 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:340` |
//! | `e075_getRegionInstrSize` | 1 | 7 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:353` |
//! | `e076_getUnitUniformInstrList` | 1 | 9 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:383` |
//! | `e077_getLastBlockCurrentInstrSize` | 1 | 3 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:433` |
//! | `e078_addInstructionToLastBlock` | 1 | 6 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:436` |
//! | `e079_getNextInstrIndex` | 1 | 19 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:464` |
//! | `e096_getUniformizedUnitInstrList` | 2 | 42 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:279` |
//! | `e097_flattenIndex` | 2 | 12 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:484` |
//! | `e121_getUniformizedUnitUniformInstrList` | 3 | 32 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:393` |

use crate::bridges::sentient_to_progir::lower::control::RegionIndex;
use crate::bridges::sentient_to_progir::state::UnitKey;
use crate::bridges::sentient_to_progir::uniform::instr::{
    UniformInstrInfo, UniformLabel, create_jmp_instr, create_nop_instr,
};
use crate::islands::progir::ty::{Operand, OperandValue};
use crate::islands::progir::{OpCode, OperandField};

/// WHERE ONE INSTRUCTION IS — `InstrIndex`, a `std::tuple<int, int, int>`
/// (`UniformInstrAndBlock.hpp:24`).
///
/// ⛔ NAMED FIELDS, NOT A TRIPLE. Three `int`s in a row is the transposition this crate's newtype
/// rule exists to prevent, and the reference steps one of them by position: `std::get<2>(index)++`
/// (`SentientToProgIR.cpp:344`), where the value stepped is `next_index`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct InstrIndex {
    /// Which block of the unit's program.
    pub block: usize,
    /// Which region of that block.
    pub region: RegionIndex,
    /// Which instruction of that region.
    pub instr: usize,
}

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

/// ONE BLOCK OF A PROGRAM UNIT'S INSTRUCTIONS — `UniformInstrBlock`
/// (`UniformInstrAndBlock.hpp:175-230`).
///
/// ⛔⛔ THE TWO BLOCK TYPES ARE TWO VARIANTS, AND THAT IS WHERE THIS CLASS' DT_CHECKs WENT.
/// `block_type_` decides what the other three fields mean, and each check restates that decision: a
/// REGULAR block holds ONE list (`:269`), its `current_region_` never leaves 0 (`:329`), and only a
/// UNIFORM block may append a region (`:379`). None of the three has a spelling here.
#[derive(Debug, Clone, PartialEq)]
pub enum UniformInstrBlock {
    /// `Type::REGULAR` — a continuous run of SentientIR ops with no uniform region, whose
    /// `instr_lists_` is one list.
    Regular(Vec<UniformInstrInfo>),
    /// `Type::UNIFORM` — one `uniformizeRegionsOp` or `equalizePatternOp`, one list per region.
    Uniform(UniformBlock),
}

/// THE UNIFORM SHAPE — the state only a `Type::UNIFORM` block has (`hpp:218-223`).
#[derive(Debug, Clone, PartialEq)]
pub struct UniformBlock {
    /// `instr_lists_` — one instruction list per uniform region, grown as regions are reached.
    pub regions: Vec<Vec<UniformInstrInfo>>,
    /// `current_region_` — which region an inserted instruction joins.
    pub current: RegionIndex,
    /// `unit_to_region_idx_map_` — which region each unit reads, as `fill_unit_to_id_map` builds it.
    pub unit_to_region: Vec<(UnitKey, RegionIndex)>,
}

impl Default for UniformBlock {
    /// `UniformInstrBlock(Type::UNIFORM)` — no region yet, region 0 current.
    fn default() -> UniformBlock {
        UniformBlock {
            regions: Vec::new(),
            current: RegionIndex(0),
            unit_to_region: Vec::new(),
        }
    }
}

impl UniformInstrBlock {
    /// `getInstrLists()` (`hpp:194`, an excluded field read) as one view over both shapes: a REGULAR
    /// block is its single list.
    ///
    /// ⚠️ A REGULAR BLOCK THAT NEVER TOOK AN INSTRUCTION SHOWS ONE EMPTY LIST WHERE THE REFERENCE HAS
    /// NONE. Every reader below answers the same either way; `empty()` (`hpp:179`) is the one that
    /// does not, so it reads the variant.
    #[must_use]
    pub fn instr_lists(&self) -> &[Vec<UniformInstrInfo>] {
        match self {
            UniformInstrBlock::Regular(instrs) => std::slice::from_ref(instrs),
            UniformInstrBlock::Uniform(block) => &block.regions,
        }
    }

    /// The same, mutably.
    #[must_use]
    pub fn instr_lists_mut(&mut self) -> &mut [Vec<UniformInstrInfo>] {
        match self {
            UniformInstrBlock::Regular(instrs) => std::slice::from_mut(instrs),
            UniformInstrBlock::Uniform(block) => &mut block.regions,
        }
    }

    /// `empty()` (`hpp:179`) — no instruction list at all, which is what makes
    /// `LowerUniformOperations` pop a leftover block (`LowerSentientHelper.cpp:1004-1006`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            UniformInstrBlock::Regular(instrs) => instrs.is_empty(),
            UniformInstrBlock::Uniform(block) => block.regions.is_empty(),
        }
    }

    /// `getCurrentRegion()` (`hpp:197`) — ⛔ A REGULAR BLOCK IS ALWAYS IN REGION 0.
    #[must_use]
    pub fn current_region(&self) -> RegionIndex {
        match self {
            UniformInstrBlock::Regular(_) => RegionIndex(0),
            UniformInstrBlock::Uniform(block) => block.current,
        }
    }

    /// Replaces: e026_getMaxInstrSize
    ///
    /// The longest region's instruction count — the length uniformization pads every other region of
    /// this block up to.
    #[must_use]
    pub fn max_instr_size(&self) -> usize {
        self.instr_lists().iter().map(Vec::len).max().unwrap_or(0)
    }

    /// Replaces: e027_getCurrentRegionInstrSize
    ///
    /// How many instructions the current region holds — 0 while that region is still one past the
    /// end, which is the state `insertInstruction` grows the list from.
    #[must_use]
    pub fn current_region_instr_size(&self) -> usize {
        let region = self.current_region().0 as usize;
        self.instr_lists().get(region).map_or(0, Vec::len)
    }

    /// Replaces: e028_getInstr
    ///
    /// The `idx`-th instruction of the region this unit reads.
    ///
    /// ⛔ AN UNMAPPED UNIT READS REGION 0, not its max region — the reference's own
    /// `TODO need to change to use MAX region` (`:368`), and NOT what `getUnitRegionIndex` answers.
    pub fn instr_mut(&mut self, idx: usize, unit: UnitKey) -> Option<&mut UniformInstrInfo> {
        let region = match self {
            UniformInstrBlock::Regular(_) => RegionIndex(0),
            UniformInstrBlock::Uniform(block) => block.region_of(unit).unwrap_or(RegionIndex(0)),
        };
        self.instr_lists_mut()
            .get_mut(region.0 as usize)?
            .get_mut(idx)
    }

    /// The UNIFORM shape, or `None` for a REGULAR block — the `Type::UNIFORM` DT_CHECK
    /// (`:379`, `:498`) as a pattern the caller cannot skip.
    #[must_use]
    pub fn uniform_mut(&mut self) -> Option<&mut UniformBlock> {
        match self {
            UniformInstrBlock::Regular(_) => None,
            UniformInstrBlock::Uniform(block) => Some(block),
        }
    }

    /// Replaces: e037_getUnitRegionIndex
    ///
    /// Which region of this block a unit reads.
    ///
    /// ⛔ `None` IS THE REFERENCE'S `return -1` (`hpp:210-211`), NOT REGION 0 — and a REGULAR block
    /// answers 0 for every unit, because its one list is everyone's.
    /// ⛔ NOT WHAT [`Self::instr_mut`] USES: that one falls back to region 0 for an unmapped unit.
    #[must_use]
    pub fn unit_region_index(&self, unit: UnitKey) -> Option<RegionIndex> {
        match self {
            UniformInstrBlock::Regular(_) => Some(RegionIndex(0)),
            UniformInstrBlock::Uniform(block) => block.region_of(unit),
        }
    }

    /// Replaces: e029_getLastInstr
    ///
    /// The instruction the current region ended on — how a caller learns whether the instruction it
    /// just inserted already carries a label (`LowerSentientHelper.cpp:242-246`, `:1092-1096`).
    pub fn last_instr_mut(&mut self) -> Option<&mut UniformInstrInfo> {
        let region = self.current_region().0 as usize;
        self.instr_lists_mut().get_mut(region)?.last_mut()
    }

    /// Replaces: e023_getUnitInstrList
    ///
    /// The instructions one unit runs in this block — the whole list when the block is REGULAR, its
    /// own region's when it is UNIFORM.
    ///
    /// ⛔ AN UNMAPPED UNIT RUNS NOTHING HERE, unlike `getInstr`, which reads region 0 for it
    /// (`:369`) — the two disagree in the reference and this keeps both.
    /// ⭐ THE PRODUCER LEAVES NO REPEAT TO PREFER: `setUnitRegionIndex` ASSIGNS
    /// (`UniformInstrAndBlock.hpp:205-207`), so [`UniformBlock::region_of`]'s last-wins is slack here.
    #[must_use]
    pub fn unit_instr_list(&self, unit: UnitKey) -> &[UniformInstrInfo] {
        match self {
            UniformInstrBlock::Regular(instrs) => instrs,
            UniformInstrBlock::Uniform(block) => block
                .region_of(unit)
                .and_then(|region| block.regions.get(region.0 as usize))
                .map_or(&[][..], Vec::as_slice),
        }
    }

    /// `insertInstruction` (`cpp:322`) over both shapes — ⛔ THE REGULAR HALF, which
    /// [`UniformBlock::insert_instruction`] cannot reach: a REGULAR block's one list IS its only
    /// region, and its `current_region_` never leaves 0 (`:329`).
    pub fn insert_instruction(&mut self, instr: UniformInstrInfo) {
        match self {
            UniformInstrBlock::Regular(instrs) => instrs.push(instr),
            UniformInstrBlock::Uniform(block) => block.insert_instruction(instr),
        }
    }

    /// Replaces: e074_getMaxInstrRegionIndex
    ///
    /// The FIRST region as long as the longest — the region every other one is padded up to, and
    /// whose instructions the padding copies as dead code (`cpp:311-314`).
    ///
    /// ⛔ ONE LIST OR NONE ANSWERS REGION 0 WITHOUT LOOKING (`:343`), which is also every REGULAR
    /// block — and this is a REGION INDEX, the value `getUniformizedUnitInstrList` compares against
    /// `unit_to_region_idx_map_` (`:288`), never an instruction count.
    #[must_use]
    pub fn max_instr_region_index(&self) -> RegionIndex {
        let lists = self.instr_lists();
        if lists.len() <= 1 {
            return RegionIndex(0);
        }
        let longest = self.max_instr_size();
        // The fallback is unreachable, some region holding the maximum, and it is the reference's own
        // answer either way: its `index` stays at the 0 it was initialised to (`:341`).
        lists
            .iter()
            .zip(0u32..)
            .find(|(list, _)| list.len() == longest)
            .map_or(RegionIndex(0), |(_, at)| RegionIndex(at))
    }

    /// Replaces: e075_getRegionInstrSize
    ///
    /// How many instructions one unit runs in this block.
    ///
    /// ⛔ AN UNMAPPED UNIT ANSWERS THE LONGEST REGION, NOT 0 — uniformization pads its region up to
    /// that, so that is what the block costs it. ⛔ NOT [`Self::unit_region_index`]: the reference
    /// reads `unit_to_region_idx_map_` itself here, so a REGULAR block falls through to the same
    /// maximum, which for one list is that list.
    #[must_use]
    pub fn region_instr_size(&self, unit: UnitKey) -> usize {
        let region = match self {
            UniformInstrBlock::Regular(_) => None,
            UniformInstrBlock::Uniform(block) => block.region_of(unit),
        };
        region
            .and_then(|region| self.instr_lists().get(region.0 as usize))
            .map_or_else(|| self.max_instr_size(), Vec::len)
    }
}

impl UniformBlock {
    /// `unit_to_region_idx_map_.at(unit)` (`hpp:212`, `cpp:276`, `:356`, `:371`) — the one lookup all
    /// four of this class' map readers make.
    ///
    /// ⛔ THE LAST ENTRY FOR A UNIT WINS: the map is filled by `map_[unit] = idx` (`hpp:205-207`), so
    /// a repeated unit keeps the newest region, which a `Vec` of pairs only reproduces from the back.
    #[must_use]
    pub fn region_of(&self, unit: UnitKey) -> Option<RegionIndex> {
        self.unit_to_region
            .iter()
            .rev()
            .find(|(at, _)| *at == unit)
            .map(|(_, region)| *region)
    }

    /// Replaces: e025_setCurrentRegion
    ///
    /// Which region the instructions lowered next belong to — `LowerUniformOperations` sets it once
    /// per region of the uniform op (`LowerSentientHelper.cpp:1023`).
    ///
    /// ⛔ THE RECEIVER IS THE DT_CHECK: `current_region_ > 0` demands `Type::UNIFORM` (`:329`), and a
    /// REGULAR block has no `UniformBlock` to set it on.
    pub fn set_current_region(&mut self, region: RegionIndex) {
        self.current = region;
    }

    /// Replaces: e024_insertInstruction
    ///
    /// Appends one instruction to the current region, opening that region if the lists have not
    /// reached it yet.
    ///
    /// ⚠️ A REGION PAST ONE-BEYOND-THE-END IS OPENED EMPTY, where the reference's
    /// `int(current_region_) - int(instr_lists_.size()) <= 0` aborts: `setCurrentRegion` takes any
    /// index, so the two-regions-ahead state is reachable and this fills the gap rather than losing
    /// the instruction.
    pub fn insert_instruction(&mut self, instr: UniformInstrInfo) {
        let region = self.current.0 as usize;
        if let Some(list) = self.regions.get_mut(region) {
            list.push(instr);
        } else {
            self.regions.resize_with(region, Vec::new);
            self.regions.push(vec![instr]);
        }
    }

    /// Replaces: e030_appendEmptyUniformRegion
    ///
    /// One more region, holding nothing — what a region whose only op is a `uniform.yield`
    /// contributes (`LowerSentientHelper.cpp:1106-1109`).
    pub fn append_empty_region(&mut self) {
        self.regions.push(Vec::new());
    }
}

/// EVERY BLOCK OF ONE `program_unitOp`, IN ORDER — `UniformInstrBlocks` (`hpp:237-278`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UniformInstrBlocks {
    /// `blocks_` — `getBlocks()` (`hpp:267`) is a field read here.
    pub blocks: Vec<UniformInstrBlock>,
}

impl UniformInstrBlocks {
    /// Replaces: e031_getMaxInstrSize
    ///
    /// How many instructions this unit's program takes.
    ///
    /// ⛔ A SUM OF PER-BLOCK MAXIMA, NOT A MAXIMUM: every unit executes every block, and within a
    /// block uniformization pads the short regions up to the longest, so the longest region is what
    /// each block costs.
    #[must_use]
    pub fn max_instr_size(&self) -> usize {
        self.blocks
            .iter()
            .map(UniformInstrBlock::max_instr_size)
            .sum()
    }

    /// Replaces: e032_appendRegularBlock
    ///
    /// Opens a REGULAR block for the run of ops that follows a uniform one
    /// (`LowerSentientHelper.cpp:1113-1114`); an unused one is popped again when the next op is also
    /// uniform (`:1004-1006`).
    ///
    /// ⭐ ITS ONLY CALLER `(void)`-CASTS THE REFERENCE IT RETURNS (`:1114`), so this answers nothing
    /// rather than a handle no caller reads.
    pub fn append_regular_block(&mut self) {
        self.blocks.push(UniformInstrBlock::Regular(Vec::new()));
    }

    /// Replaces: e033_appendUniformBlock
    ///
    /// Opens a UNIFORM block for one `uniformizeRegionsOp` or `equalizePatternOp` and hands it back,
    /// because its caller fills the unit-to-region map and walks the regions through it
    /// (`LowerSentientHelper.cpp:1007-1023`).
    pub fn append_uniform_block(&mut self) -> &mut UniformBlock {
        self.blocks
            .push(UniformInstrBlock::Uniform(UniformBlock::default()));
        self.blocks
            .last_mut()
            .and_then(UniformInstrBlock::uniform_mut)
            .expect("just pushed a uniform block")
    }

    /// Replaces: e036_appendEmptyUniformRegion
    ///
    /// One more region on the block currently open — what a region whose only op is a
    /// `uniform.yield` contributes (`LowerSentientHelper.cpp:1109`).
    ///
    /// ⛔ THE PATTERN IS THE DT_CHECK (`:498`): only a UNIFORM last block takes a region, and a
    /// REGULAR one — or none at all, where the reference's `back()` is undefined — takes nothing.
    pub fn append_empty_uniform_region(&mut self) {
        if let Some(block) = self
            .blocks
            .last_mut()
            .and_then(UniformInstrBlock::uniform_mut)
        {
            block.append_empty_region();
        }
    }

    /// Replaces: e034_getUniformInstr
    ///
    /// The instruction at one flat index, for reading its tag and for retagging it — both of which
    /// its caller does through the same reference (`SentientToProgIR.cpp:342-366`).
    ///
    /// ⛔ `None` WHERE THE REFERENCE'S THREE `.at()`s THROW; [`Self::does_instr_exist`] is the test
    /// its caller uses before stepping the index.
    pub fn uniform_instr_mut(&mut self, index: InstrIndex) -> Option<&mut UniformInstrInfo> {
        self.blocks
            .get_mut(index.block)?
            .instr_lists_mut()
            .get_mut(index.region.0 as usize)?
            .get_mut(index.instr)
    }

    /// Replaces: e035_doesInstrWithThisIndexExist
    ///
    /// Whether the index names an instruction — the guard on stepping to the NEXT one, which is how
    /// a label finds the instruction it must land on (`SentientToProgIR.cpp:343-346`).
    ///
    /// ⛔ THE REFERENCE ONLY BOUNDS THE INSTRUCTION: its block and region `.at()`s throw rather than
    /// answering false, so this is false for all three where that one is only false for the last.
    #[must_use]
    pub fn does_instr_exist(&self, index: InstrIndex) -> bool {
        self.blocks
            .get(index.block)
            .and_then(|block| block.instr_lists().get(index.region.0 as usize))
            .is_some_and(|region| index.instr < region.len())
    }

    /// Replaces: e076_getUnitUniformInstrList
    ///
    /// Every instruction one unit runs, across every block, in block order.
    ///
    /// ⛔ NOT UNIFORMIZED DESPITE THE NAME: no padding NOP and no JCMP, and a UNIFORM block that maps
    /// the unit nowhere contributes nothing — so this is SHORTER than the unit's program wherever
    /// [`Self::max_instr_size`] counts a region the unit does not read.
    #[must_use]
    pub fn unit_instr_list(&self, unit: UnitKey) -> Vec<UniformInstrInfo> {
        self.blocks
            .iter()
            .flat_map(|block| block.unit_instr_list(unit).to_vec())
            .collect()
    }

    /// Replaces: e077_getLastBlockCurrentInstrSize
    ///
    /// How many instructions the block currently open has taken in its current region — where the
    /// next one lands.
    ///
    /// ⛔ 0 WITH NO BLOCK AT ALL, where the reference's `blocks_.back()` reads off the end.
    #[must_use]
    pub fn last_block_current_instr_size(&self) -> usize {
        self.blocks
            .last()
            .map_or(0, UniformInstrBlock::current_region_instr_size)
    }

    /// Replaces: e078_addInstructionToLastBlock
    ///
    /// Appends one instruction to the block currently open, opening a REGULAR one first when the
    /// unit's program has no block yet.
    pub fn add_instruction_to_last_block(&mut self, instr: UniformInstrInfo) {
        if self.blocks.is_empty() {
            self.append_regular_block();
        }
        if let Some(block) = self.blocks.last_mut() {
            block.insert_instruction(instr);
        }
    }

    /// Replaces: e079_getNextInstrIndex
    ///
    /// Where the next instruction lands IF it joins the block and region currently open — ⛔ THE
    /// CALLER OWNS THE OTHER CASE: a following uniform op opens block `blocks_.len()` instead, and
    /// only the caller knows which op comes next — the reference's own four cases (`cpp:465-477`),
    /// read by `AddToLabelsMap` and the NOP-for-label path (`LowerSentientHelper.cpp:37`, `:558`).
    ///
    /// ⛔ `None` WITH NO BLOCK, where the reference answers block `-1` and then reads `back()`.
    #[must_use]
    pub fn next_instr_index(&self) -> Option<InstrIndex> {
        let block = self.blocks.len().checked_sub(1)?;
        let open = self.blocks.last()?;
        Some(InstrIndex {
            block,
            region: open.current_region(),
            instr: open.current_region_instr_size(),
        })
    }
}

impl UniformInstrBlock {
    /// Replaces: e096_getUniformizedUnitInstrList
    ///
    /// One unit's list for this block padded to the longest region: its own instructions, then a JCMP
    /// over a copy of the longest region's tail marked dead — or a single NOP where that tail is
    /// empty. The `usize` is where the jump landed, and `None` is the reference's `-1`.
    ///
    /// ⛔ ONLY A SHORTER OR UNMAPPED UNIT IS PADDED: one already on the longest region, or on a region
    /// no shorter than it, is handed back untouched. ⛔ AND `num_padded_instrs` IS AN `int` OVER A
    /// `size_t` DIFFERENCE (`:303`) — an empty longest region gives -1, not 0, so it takes the JCMP.
    #[must_use]
    pub fn uniformized_unit_instr_list(
        &self,
        unit: UnitKey,
        label: UniformLabel,
    ) -> (Vec<UniformInstrInfo>, Option<usize>) {
        let block = match self {
            UniformInstrBlock::Regular(instrs) => return (instrs.clone(), None),
            UniformInstrBlock::Uniform(block) => block,
        };
        let max_region_idx = self.max_instr_region_index();
        let max_region_list = block
            .regions
            .get(max_region_idx.0 as usize)
            .map_or(&[][..], Vec::as_slice);
        let region_of = block.region_of(unit);
        let elsewhere = region_of.filter(|at| *at != max_region_idx);
        let own = elsewhere.and_then(|region| block.regions.get(region.0 as usize));
        let pads = region_of.is_none()
            || (elsewhere.is_some() && own.map_or(0, Vec::len) < max_region_list.len());
        if !pads {
            let settled = region_of.and_then(|region| block.regions.get(region.0 as usize));
            return (settled.cloned().unwrap_or_default(), None);
        }
        let mut ret_list = own.cloned().unwrap_or_default();
        let jmp_pos = ret_list.len();
        let num_padded_instrs = max_region_list.len() as i64 - jmp_pos as i64 - 1;
        // An optimization to use NOP instruction instead of JCMP instruction to save the cycles.
        if num_padded_instrs == 0 {
            ret_list.push(create_nop_instr(label));
        } else {
            // Jump over the padding, then carry the padding itself as dead code.
            ret_list.push(create_jmp_instr(label));
            ret_list.extend(max_region_list.iter().skip(jmp_pos + 1).map(|instr| {
                let mut padding = instr.clone();
                padding.dead = true;
                padding
            }));
        }
        (ret_list, Some(jmp_pos))
    }
}

impl UniformInstrBlocks {
    /// Replaces: e097_flattenIndex
    ///
    /// Where an index lands in one unit's own flat program: what every earlier block costs that unit,
    /// plus the offset inside its region.
    ///
    /// ⛔ THE REGION HALF OF THE INDEX IS NEVER READ — the reference unpacks it into an unused local
    /// (`:488`), because [`UniformInstrBlock::region_instr_size`] resolves the unit's region itself.
    /// ⛔ AND A BLOCK COUNT PAST THE END CONTRIBUTES NOTHING, where `blocks_.at()` throws.
    #[must_use]
    pub fn flatten_index(&self, index: InstrIndex, unit: UnitKey) -> usize {
        self.blocks
            .iter()
            .take(index.block)
            .map(|block| block.region_instr_size(unit))
            .sum::<usize>()
            + index.instr
    }
}

impl UniformInstrBlocks {
    /// Replaces: e121_getUniformizedUnitUniformInstrList
    ///
    /// One unit's whole program: every block's uniformised list in order, with each padding jump
    /// aimed at the instruction that follows it.
    ///
    /// ⛔ THE JUMP AND ITS TARGET MEET HALFWAY — an UNTAGGED first instruction takes the jump's
    /// `pc_target` as its tag, and a TAGGED one replaces that `pc_target` instead.
    /// ⛔ A JUMP POSITION IS RELATIVE TO ITS OWN BLOCK, so it is offset by everything emitted before
    /// it; and only a `JCMP` there is padding — a block whose own tail is a jump is not retargeted.
    #[must_use]
    pub fn uniformized_unit_uniform_instr_list(
        &self,
        unit: UnitKey,
        first_label: UniformLabel,
    ) -> Vec<UniformInstrInfo> {
        let mut ret: Vec<UniformInstrInfo> = Vec::new();
        let mut prev_jmp_pos: Option<usize> = None;
        let mut label = first_label;
        for block in &self.blocks {
            let (mut instr_list, new_jmp_pos) = block.uniformized_unit_instr_list(unit, label);
            label = label.bump();
            if let Some(jmp) = prev_jmp_pos.and_then(|at| ret.get_mut(at))
                && jmp.opcode == OpCode::JCMP
            {
                match instr_list.first().and_then(|first| first.tag.clone()) {
                    Some(tag) => {
                        jmp.set_common_field(
                            OperandField::PcTarget,
                            Operand::every(OperandValue::InstrTag(tag)),
                        );
                    }
                    None => {
                        let target = jmp
                            .common_field(OperandField::PcTarget)
                            .and_then(|target| target.as_string(None))
                            .map(str::to_owned);
                        if let Some(first) = instr_list.first_mut() {
                            first.tag = target;
                        }
                    }
                }
            }
            prev_jmp_pos = new_jmp_pos.map(|at| at + ret.len());
            ret.append(&mut instr_list);
        }
        ret
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{InstrIndex, UniformBlock, UniformInstrBlock, UniformInstrBlocks};
    use crate::bridges::sentient_to_progir::lower::control::RegionIndex;
    use crate::bridges::sentient_to_progir::state::UnitKey;
    use crate::bridges::sentient_to_progir::uniform::instr::{UniformInstrInfo, UniformLabel};
    use crate::islands::progir::{OpCode, OperandField};
    use crate::units::{Core, DfirUnit, Residency};

    fn nop() -> UniformInstrInfo {
        UniformInstrInfo::of(OpCode::NOP)
    }

    fn ret() -> UniformInstrInfo {
        UniformInstrInfo::of(OpCode::RETURN)
    }

    fn unit(unit: DfirUnit) -> UnitKey {
        let core = Core::checked(0).expect("core 0");
        UnitKey::of(unit, Residency::CoreWide { core }).expect("core 0 keys")
    }

    /// e025 + e027 + e029: the current region selects both what is counted and what is read back,
    /// and a region the lists have not reached yet holds nothing rather than crashing.
    #[test]
    fn the_current_region_selects_what_is_counted_and_read_back() {
        let mut block = UniformBlock {
            regions: vec![vec![nop(), nop()], vec![ret()]],
            ..UniformBlock::default()
        };
        assert_eq!(
            UniformInstrBlock::Uniform(block.clone()).current_region_instr_size(),
            2
        );
        block.set_current_region(RegionIndex(1));
        let mut second = UniformInstrBlock::Uniform(block.clone());
        assert_eq!(second.current_region_instr_size(), 1);
        assert_eq!(
            second.last_instr_mut().map(|instr| instr.opcode),
            Some(OpCode::RETURN)
        );
        // One past the end — where `insertInstruction` grows the list from.
        block.set_current_region(RegionIndex(2));
        let mut past = UniformInstrBlock::Uniform(block);
        assert_eq!(past.current_region_instr_size(), 0);
        assert_eq!(past.last_instr_mut(), None);
    }

    /// e026: the longest region, including the no-region and REGULAR shapes.
    #[test]
    fn the_max_instr_size_is_the_longest_region() {
        assert_eq!(
            UniformInstrBlock::Uniform(UniformBlock::default()).max_instr_size(),
            0
        );
        let block = UniformInstrBlock::Uniform(UniformBlock {
            regions: vec![vec![nop()], vec![nop(), nop(), ret()], Vec::new()],
            ..UniformBlock::default()
        });
        assert_eq!(block.max_instr_size(), 3);
        assert_eq!(
            UniformInstrBlock::Regular(vec![nop(), ret()]).max_instr_size(),
            2
        );
    }

    /// e028: a mapped unit reads its own region; an unmapped one reads region 0.
    #[test]
    fn an_unmapped_unit_reads_region_zero() {
        let lxlu = unit(DfirUnit::Lxlu);
        let mut block = UniformInstrBlock::Uniform(UniformBlock {
            regions: vec![vec![nop()], vec![ret()]],
            unit_to_region: vec![(lxlu, RegionIndex(1))],
            ..UniformBlock::default()
        });
        assert_eq!(
            block.instr_mut(0, lxlu).map(|instr| instr.opcode),
            Some(OpCode::RETURN)
        );
        assert_eq!(
            block
                .instr_mut(0, unit(DfirUnit::L3su))
                .map(|instr| instr.opcode),
            Some(OpCode::NOP)
        );
        assert_eq!(block.instr_mut(1, lxlu), None);
    }

    /// e030: an appended region exists and is empty, so it lengthens nothing.
    #[test]
    fn an_appended_uniform_region_is_empty() {
        let mut block = UniformBlock::default();
        block.append_empty_region();
        block.append_empty_region();
        assert_eq!(block.regions.len(), 2);
        let block = UniformInstrBlock::Uniform(block);
        assert!(!block.is_empty());
        assert_eq!(block.max_instr_size(), 0);
    }

    /// e031 + e032: the appended block is REGULAR, and the unit's length sums each block's longest
    /// region rather than taking the longest block.
    #[test]
    fn the_unit_length_sums_each_blocks_longest_region() {
        let mut blocks = UniformInstrBlocks::default();
        blocks.append_regular_block();
        assert!(matches!(
            blocks.blocks.last(),
            Some(UniformInstrBlock::Regular(_))
        ));
        if let Some(UniformInstrBlock::Regular(instrs)) = blocks.blocks.last_mut() {
            instrs.push(nop());
            instrs.push(nop());
        }
        blocks.blocks.push(UniformInstrBlock::Uniform(UniformBlock {
            regions: vec![vec![nop()], vec![nop(), nop(), ret()]],
            ..UniformBlock::default()
        }));
        assert_eq!(blocks.max_instr_size(), 2 + 3);
    }

    /// e023: a mapped unit runs its own region, an unmapped one runs nothing, and a REGULAR block is
    /// one list every unit runs.
    #[test]
    fn an_unmapped_unit_runs_nothing_and_a_regular_block_is_one_list() {
        let lxlu = unit(DfirUnit::Lxlu);
        let block = UniformInstrBlock::Uniform(UniformBlock {
            regions: vec![vec![nop()], vec![ret(), nop()]],
            unit_to_region: vec![(lxlu, RegionIndex(1))],
            ..UniformBlock::default()
        });
        assert_eq!(
            block
                .unit_instr_list(lxlu)
                .iter()
                .map(|instr| instr.opcode)
                .collect::<Vec<_>>(),
            vec![OpCode::RETURN, OpCode::NOP]
        );
        assert!(block.unit_instr_list(unit(DfirUnit::L3su)).is_empty());
        let regular = UniformInstrBlock::Regular(vec![nop(), ret()]);
        assert_eq!(regular.unit_instr_list(unit(DfirUnit::Pe)).len(), 2);
    }

    /// e024: the instruction joins the current region, which is opened when the lists stop short of
    /// it — including the skipped-region case the reference refuses.
    #[test]
    fn an_inserted_instruction_opens_the_current_region() {
        let mut block = UniformBlock::default();
        block.insert_instruction(nop());
        block.insert_instruction(ret());
        assert_eq!(block.regions.len(), 1);
        assert_eq!(
            block.regions[0]
                .iter()
                .map(|i| i.opcode)
                .collect::<Vec<_>>(),
            vec![OpCode::NOP, OpCode::RETURN]
        );
        block.set_current_region(RegionIndex(2));
        block.insert_instruction(nop());
        assert_eq!(block.regions.len(), 3, "the skipped region is opened empty");
        assert!(block.regions[1].is_empty());
        assert_eq!(block.regions[2].len(), 1);
    }

    /// e033 + e036: the appended block is the UNIFORM shape, and the empty region lands in it —
    /// ⛔ A REGULAR LAST BLOCK TAKES NO REGION, which is the reference's `Type::UNIFORM` DT_CHECK.
    #[test]
    fn an_empty_uniform_region_lands_in_a_uniform_last_block_only() {
        let mut blocks = UniformInstrBlocks::default();
        blocks.append_uniform_block().unit_to_region = vec![(unit(DfirUnit::Lxlu), RegionIndex(0))];
        blocks.append_empty_uniform_region();
        blocks.append_empty_uniform_region();
        assert_eq!(blocks.blocks[0].instr_lists().len(), 2);
        blocks.append_regular_block();
        blocks.append_empty_uniform_region();
        assert!(matches!(blocks.blocks[1], UniformInstrBlock::Regular(_)));
        // ⛔ AND IT DID NOT FALL BACK TO THE UNIFORM BLOCK BEHIND IT either.
        assert_eq!(blocks.blocks[0].instr_lists().len(), 2);
    }

    /// e034 + e035: an index reads back the instruction it names, and every index past a block, a
    /// region or a region's length answers absent instead of throwing.
    #[test]
    fn only_an_index_the_lists_reach_holds_an_instruction() {
        let mut blocks = UniformInstrBlocks::default();
        blocks.append_uniform_block().regions = vec![vec![nop()], vec![nop(), ret()]];
        let at = |block, region, instr| InstrIndex {
            block,
            region: RegionIndex(region),
            instr,
        };
        assert!(blocks.does_instr_exist(at(0, 1, 1)));
        assert_eq!(
            blocks.uniform_instr_mut(at(0, 1, 1)).map(|i| i.opcode),
            Some(OpCode::RETURN)
        );
        for index in [at(0, 0, 1), at(0, 2, 0), at(1, 0, 0)] {
            assert!(!blocks.does_instr_exist(index));
            assert_eq!(blocks.uniform_instr_mut(index), None);
        }
    }

    /// e037: a mapped unit's own region, `None` for an unmapped one — ⛔ AND 0 FOR EVERY UNIT OF A
    /// REGULAR BLOCK, whose one list is everyone's. Also pins the shared lookup's last-entry-wins.
    #[test]
    fn an_unmapped_unit_has_no_region_index() {
        let lxlu = unit(DfirUnit::Lxlu);
        let block = UniformInstrBlock::Uniform(UniformBlock {
            regions: vec![vec![nop()], vec![ret()]],
            unit_to_region: vec![(lxlu, RegionIndex(1))],
            ..UniformBlock::default()
        });
        assert_eq!(block.unit_region_index(lxlu), Some(RegionIndex(1)));
        assert_eq!(block.unit_region_index(unit(DfirUnit::L3su)), None);
        assert_eq!(
            UniformInstrBlock::Regular(vec![nop()]).unit_region_index(unit(DfirUnit::L3su)),
            Some(RegionIndex(0))
        );
        let rewritten = UniformBlock {
            regions: vec![vec![nop()], vec![ret()]],
            unit_to_region: vec![(lxlu, RegionIndex(1)), (lxlu, RegionIndex(0))],
            ..UniformBlock::default()
        };
        assert_eq!(
            rewritten.region_of(lxlu),
            Some(RegionIndex(0)),
            "`map[unit] = idx` left the second entry"
        );
        let mut rewritten = UniformInstrBlock::Uniform(rewritten);
        assert_eq!(rewritten.unit_region_index(lxlu), Some(RegionIndex(0)));
        assert_eq!(
            rewritten.instr_mut(0, lxlu).map(|instr| instr.opcode),
            Some(OpCode::NOP),
            "`getInstr` reads the same entry the region index answers"
        );
    }

    /// e074: the FIRST region as long as the longest, and 0 for the one-list and no-region shapes.
    #[test]
    fn the_padding_target_is_the_first_longest_region() {
        let block = UniformInstrBlock::Uniform(UniformBlock {
            regions: vec![vec![nop()], vec![nop(), ret()], vec![ret(), nop()]],
            ..UniformBlock::default()
        });
        assert_eq!(
            block.max_instr_region_index(),
            RegionIndex(1),
            "the first of the two"
        );
        assert_eq!(
            UniformInstrBlock::Regular(vec![nop(), ret()]).max_instr_region_index(),
            RegionIndex(0)
        );
        assert_eq!(
            UniformInstrBlock::Uniform(UniformBlock::default()).max_instr_region_index(),
            RegionIndex(0)
        );
    }

    /// e075: a mapped unit's own region — ⛔ AND AN UNMAPPED ONE COSTS THE LONGEST, not 0.
    #[test]
    fn an_unmapped_unit_costs_the_longest_region() {
        let lxlu = unit(DfirUnit::Lxlu);
        let block = UniformInstrBlock::Uniform(UniformBlock {
            regions: vec![vec![nop(), ret(), nop()], vec![ret()]],
            unit_to_region: vec![(lxlu, RegionIndex(1))],
            ..UniformBlock::default()
        });
        assert_eq!(block.region_instr_size(lxlu), 1);
        assert_eq!(
            block.region_instr_size(unit(DfirUnit::L3su)),
            3,
            "uniformization pads the unmapped unit up to the longest region"
        );
        assert_eq!(
            UniformInstrBlock::Regular(vec![nop(), ret()]).region_instr_size(lxlu),
            2
        );
    }

    /// e076: every block's contribution in order — ⛔ AND A BLOCK THAT MAPS THE UNIT NOWHERE ADDS
    /// NOTHING.
    #[test]
    fn a_units_instructions_run_together_across_the_blocks() {
        let lxlu = unit(DfirUnit::Lxlu);
        let mut blocks = UniformInstrBlocks::default();
        blocks.append_regular_block();
        blocks.add_instruction_to_last_block(ret());
        blocks.blocks.push(UniformInstrBlock::Uniform(UniformBlock {
            regions: vec![vec![ret()], vec![nop(), nop()]],
            unit_to_region: vec![(lxlu, RegionIndex(1))],
            ..UniformBlock::default()
        }));
        assert_eq!(
            blocks.unit_instr_list(lxlu),
            vec![ret(), nop(), nop()],
            "the regular block's one list, then the unit's own region"
        );
        assert_eq!(
            blocks.unit_instr_list(unit(DfirUnit::L3su)),
            vec![ret()],
            "the uniform block maps this unit nowhere"
        );
    }

    /// e077: the current region of the LAST block — ⛔ AND 0 WITH NO BLOCK AT ALL.
    #[test]
    fn the_last_blocks_current_region_is_where_the_next_instruction_lands() {
        let mut blocks = UniformInstrBlocks::default();
        assert_eq!(blocks.last_block_current_instr_size(), 0);
        blocks.blocks.push(UniformInstrBlock::Uniform(UniformBlock {
            regions: vec![vec![nop(), nop()], vec![ret()]],
            current: RegionIndex(1),
            ..UniformBlock::default()
        }));
        assert_eq!(blocks.last_block_current_instr_size(), 1);
    }

    /// e078: the first instruction opens a REGULAR block, and a later one joins whatever block is
    /// open, in its current region.
    #[test]
    fn the_first_instruction_opens_a_regular_block() {
        let mut blocks = UniformInstrBlocks::default();
        blocks.add_instruction_to_last_block(nop());
        assert!(matches!(
            blocks.blocks.as_slice(),
            [UniformInstrBlock::Regular(_)]
        ));
        blocks.add_instruction_to_last_block(ret());
        assert_eq!(
            blocks.blocks[0].instr_lists().to_vec(),
            vec![vec![nop(), ret()]]
        );
        blocks
            .append_uniform_block()
            .set_current_region(RegionIndex(1));
        blocks.add_instruction_to_last_block(ret());
        assert_eq!(
            blocks.blocks[1].instr_lists().to_vec(),
            vec![Vec::new(), vec![ret()]],
            "the uniform block's current region takes it"
        );
    }

    /// e079: the index names the block and region currently open — ⛔ AND `None` with no block, where
    /// the reference answers block -1.
    #[test]
    fn the_next_index_names_the_open_block_and_region() {
        let mut blocks = UniformInstrBlocks::default();
        assert_eq!(blocks.next_instr_index(), None);
        blocks.add_instruction_to_last_block(nop());
        assert_eq!(
            blocks.next_instr_index(),
            Some(InstrIndex {
                block: 0,
                region: RegionIndex(0),
                instr: 1
            })
        );
        let uniform = blocks.append_uniform_block();
        uniform.regions = vec![vec![nop()], vec![nop(), ret()]];
        uniform.set_current_region(RegionIndex(1));
        assert_eq!(
            blocks.next_instr_index(),
            Some(InstrIndex {
                block: 1,
                region: RegionIndex(1),
                instr: 2
            })
        );
    }
    /// e096: a unit on the shorter region gets its own instructions, a JCMP and the longest region's
    /// tail as dead code — and one already on the longest region is not padded at all.
    #[test]
    fn a_shorter_region_is_padded_with_a_jump_over_dead_code() {
        let short = unit(DfirUnit::Lxlu);
        let long = unit(DfirUnit::L3su);
        let block = UniformInstrBlock::Uniform(UniformBlock {
            regions: vec![vec![nop()], vec![nop(), nop(), ret()]],
            unit_to_region: vec![(short, RegionIndex(0)), (long, RegionIndex(1))],
            ..UniformBlock::default()
        });
        let (padded, jmp_pos) = block.uniformized_unit_instr_list(short, UniformLabel(7));
        assert_eq!(jmp_pos, Some(1));
        assert_eq!(
            padded.len(),
            3,
            "its own NOP, the JCMP and one padding instr"
        );
        assert_eq!(padded[1].opcode, OpCode::JCMP);
        assert_eq!(padded[2].opcode, OpCode::RETURN);
        assert!(padded[2].dead && !padded[1].dead);
        assert_eq!(
            block.uniformized_unit_instr_list(long, UniformLabel(7)),
            (vec![nop(), nop(), ret()], None)
        );
        // An unmapped unit is padded from nothing, and a tail of exactly one instruction is a NOP.
        let one_short = UniformInstrBlock::Uniform(UniformBlock {
            regions: vec![vec![nop()], vec![nop(), ret()]],
            unit_to_region: vec![(short, RegionIndex(0))],
            ..UniformBlock::default()
        });
        let (nopped, jmp_pos) = one_short.uniformized_unit_instr_list(short, UniformLabel(7));
        assert_eq!((nopped.len(), jmp_pos), (2, Some(1)));
        assert_eq!(nopped[1].opcode, OpCode::NOP);
    }

    /// e097: the flat index sums the earlier blocks' cost for THAT unit — an unmapped unit pays the
    /// longest region of each, a mapped one pays its own.
    #[test]
    fn the_flat_index_sums_the_earlier_blocks_for_that_unit() {
        let lxlu = unit(DfirUnit::Lxlu);
        let mut blocks = UniformInstrBlocks::default();
        blocks
            .blocks
            .push(UniformInstrBlock::Regular(vec![nop(), ret()]));
        blocks.blocks.push(UniformInstrBlock::Uniform(UniformBlock {
            regions: vec![vec![nop()], vec![nop(), nop(), ret()]],
            unit_to_region: vec![(lxlu, RegionIndex(0))],
            ..UniformBlock::default()
        }));
        blocks.blocks.push(UniformInstrBlock::Regular(vec![ret()]));
        let at = |block, instr| InstrIndex {
            block,
            region: RegionIndex(1),
            instr,
        };
        assert_eq!(blocks.flatten_index(at(0, 1), lxlu), 1);
        assert_eq!(
            blocks.flatten_index(at(2, 0), lxlu),
            3,
            "2 + its own region"
        );
        assert_eq!(
            blocks.flatten_index(at(2, 0), unit(DfirUnit::L3su)),
            5,
            "an unmapped unit pays the longest region"
        );
        assert_eq!(
            blocks.flatten_index(at(9, 4), lxlu),
            8,
            "a block count past the end sums only the blocks that exist"
        );
    }

    /// e121: the padding jump of one block finds its target in the next — an untagged first
    /// instruction TAKES the jump's `pc_target`, and a tagged one REPLACES it.
    #[test]
    fn a_padding_jump_and_the_next_blocks_first_instruction_agree_on_one_tag() {
        let short = unit(DfirUnit::Lxlu);
        let long = unit(DfirUnit::L3su);
        let padded = || {
            UniformInstrBlock::Uniform(UniformBlock {
                regions: vec![vec![nop()], vec![nop(), nop(), ret()]],
                unit_to_region: vec![(short, RegionIndex(0)), (long, RegionIndex(1))],
                ..UniformBlock::default()
            })
        };
        let mut blocks = UniformInstrBlocks::default();
        blocks.blocks.push(padded());
        blocks.blocks.push(UniformInstrBlock::Regular(vec![ret()]));
        let list = blocks.uniformized_unit_uniform_instr_list(short, UniformLabel(7));
        assert_eq!(
            list.len(),
            4,
            "its NOP, the JCMP, one dead instr, then block 1"
        );
        assert_eq!(list[1].opcode, OpCode::JCMP);
        assert_eq!(
            list[3].tag.as_deref(),
            Some("uniform_tgt_7"),
            "the jump's own target became the next block's tag"
        );
        // A first instruction that already has a tag keeps it, and the jump is retargeted instead.
        let mut tagged = ret();
        tagged.tag = Some("if-label-0-end".to_owned());
        let mut blocks = UniformInstrBlocks::default();
        blocks.blocks.push(padded());
        blocks.blocks.push(UniformInstrBlock::Regular(vec![tagged]));
        let list = blocks.uniformized_unit_uniform_instr_list(short, UniformLabel(7));
        assert_eq!(list[3].tag.as_deref(), Some("if-label-0-end"));
        assert_eq!(
            list[1]
                .common_field(OperandField::PcTarget)
                .and_then(|target| target.as_string(None)),
            Some("if-label-0-end")
        );
        // ⛔ THE UNIT ON THE LONGEST REGION IS NEVER PADDED, so nothing is retagged for it.
        let list = blocks.uniformized_unit_uniform_instr_list(long, UniformLabel(7));
        assert!(list.iter().all(|instr| instr.opcode != OpCode::JCMP));
        assert_eq!(list[3].tag.as_deref(), Some("if-label-0-end"));
    }
}

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

use crate::bridges::sentient_to_progir::lower::control::{RegionIndex, UnitKey};
use crate::bridges::sentient_to_progir::uniform::instr::UniformInstrInfo;

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

/// ONE BLOCK OF A PROGRAM UNIT'S INSTRUCTIONS — `UniformInstrBlock`
/// (`UniformInstrAndBlock.hpp:172-230`).
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

/// THE UNIFORM SHAPE — the state only a `Type::UNIFORM` block has (`hpp:216-228`).
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
    /// `LowerUniformOperations` pop a leftover block (`LowerSentientHelper.cpp:1006-1008`).
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
    /// `TODO need to change to use MAX region` (`:367`), and NOT what `getUnitRegionIndex` answers.
    pub fn instr_mut(&mut self, idx: usize, unit: UnitKey) -> Option<&mut UniformInstrInfo> {
        let region = match self {
            UniformInstrBlock::Regular(_) => RegionIndex(0),
            UniformInstrBlock::Uniform(block) => block
                .unit_to_region
                .iter()
                .find(|(at, _)| *at == unit)
                .map_or(RegionIndex(0), |(_, region)| *region),
        };
        self.instr_lists_mut()
            .get_mut(region.0 as usize)?
            .get_mut(idx)
    }

    /// Replaces: e029_getLastInstr
    ///
    /// The instruction the current region ended on — how a caller retags what it just inserted
    /// (`LowerSentientHelper.cpp:1092-1095`).
    pub fn last_instr_mut(&mut self) -> Option<&mut UniformInstrInfo> {
        let region = self.current_region().0 as usize;
        self.instr_lists_mut().get_mut(region)?.last_mut()
    }
}

impl UniformBlock {
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
    /// Opens a REGULAR block for the run of ops that follows a uniform one; an unused one is popped
    /// again when the next op is also uniform (`LowerSentientHelper.cpp:1006-1008`).
    ///
    /// ⭐ ITS ONLY CALLER `(void)`-CASTS THE REFERENCE IT RETURNS (`:1114`), so this answers nothing
    /// rather than a handle no caller reads.
    pub fn append_regular_block(&mut self) {
        self.blocks.push(UniformInstrBlock::Regular(Vec::new()));
    }
}

// crustify:todo: e023_getUnitInstrList
// crustify:todo: e024_insertInstruction
// crustify:todo: e033_appendUniformBlock
// crustify:todo: e034_getUniformInstr
// crustify:todo: e035_doesInstrWithThisIndexExist
// crustify:todo: e036_appendEmptyUniformRegion
// crustify:todo: e037_getUnitRegionIndex
// crustify:todo: e074_getMaxInstrRegionIndex
// crustify:todo: e075_getRegionInstrSize
// crustify:todo: e076_getUnitUniformInstrList
// crustify:todo: e077_getLastBlockCurrentInstrSize
// crustify:todo: e078_addInstructionToLastBlock
// crustify:todo: e079_getNextInstrIndex
// crustify:todo: e096_getUniformizedUnitInstrList
// crustify:todo: e097_flattenIndex
// crustify:todo: e121_getUniformizedUnitUniformInstrList

#[cfg(test)]
mod unit_tests {
    use super::{UniformBlock, UniformInstrBlock, UniformInstrBlocks};
    use crate::bridges::sentient_to_progir::lower::control::{RegionIndex, UnitKey};
    use crate::bridges::sentient_to_progir::uniform::instr::UniformInstrInfo;
    use crate::islands::progir::OpCode;
    use crate::units::{Core, DfirUnit, Residency};

    fn nop() -> UniformInstrInfo {
        UniformInstrInfo::of(OpCode::NOP)
    }

    fn ret() -> UniformInstrInfo {
        UniformInstrInfo::of(OpCode::RETURN)
    }

    fn unit(unit: DfirUnit) -> UnitKey {
        UnitKey {
            unit,
            residency: Residency::CoreWide {
                core: Core::checked(0).expect("core 0"),
            },
        }
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
}

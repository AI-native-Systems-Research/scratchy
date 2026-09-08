//! THE CONTROL FLOW AND THE UNIFORM REGIONS — for, yield, return, sync, nop, set-send-dest, and
//! the uniform-operation pair.
//!
//! ⛔ `LowerUniformOperations` AND `GenerateProgIR` ARE MUTUALLY RECURSIVE (level 5, one
//! component). Neither can be ported as if the other were a leaf.
//! ⛔ THE SYNC ORDER IS dxp's AND CITED: reordering syncs times the card out (CB state=TimedOut,
//! sync rc=-1).
//!
//! 9 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e011_fillUnitToIdMap` | 0 | 17 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:967` |
//! | `e105_LowerSyncOperation` | 3 | 37 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:353` |
//! | `e106_LowerNOPOperation` | 3 | 14 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:391` |
//! | `e111_LowerForOperation` | 3 | 66 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:591` |
//! | `e115_LowerReturnOperation` | 3 | 7 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:900` |
//! | `e116_LowerSetSendDestinationOperation` | 3 | 18 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:908` |
//! | `e123_LowerYieldOperation` | 4 | 117 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:782` |
//! | `e125_LowerUniformYieldOperation` | 4 | 61 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:1056` |
//! | `e126_LowerUniformOperations` | 5 | 70 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:985` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

use crate::islands::sentient::dialects::{Op, Val, dataflow, defining_op};
use crate::units::{DfirUnit, Residency};

/// WHICH UNIT — `getUnitName`'s `type + "core" + N + "corelet" + M` (`dcc/src/Utils/Utils.cpp:264`)
/// as the pair it stringifies.
///
/// ⛔ THE PARTS, NOT THE SPELLING. The reference keys its unit-to-region map by that string, so two
/// units collide exactly when their names do and nothing can be read back out of one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnitKey {
    /// `type=` — which unit.
    pub unit: DfirUnit,
    /// Which core and corelet it lives on.
    pub residency: Residency,
}

/// WHICH REGION OF A UNIFORMIZED OP A UNIT TAKES ITS INSTRUCTIONS FROM — the value of
/// `unit_to_region_idx_map_` (`UniformInstrAndBlock.hpp:187-205`).
///
/// ⛔ ABSENCE IS AN `Option<RegionIndex>`, NEVER `(size_t)-1`, which is what `getUnitRegionIndex`
/// answers for a unit the map does not hold. A REGULAR block always answers 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionIndex(pub u32);

/// Replaces: e011_fillUnitToIdMap
///
/// Which region each unit of a uniformized op draws its instructions from; a group contributes every
/// unit it holds, under the same region.
/// ⛔ THE REFERENCE DEREFERENCES A GROUP MEMBER'S `GetUnitOp` UNCHECKED — `getDefiningOp<GetUnitOp>()`
/// with no null test — so a group holding anything else crashes there and is skipped here.
/// ⭐ `getNumRegions`/`getRegionUnitList` ARE THE MECHANISM, not the decision: the unit lists arrive
/// already partitioned, one per region, because `list_sizes` is what partitions them.
#[must_use]
pub fn fill_unit_to_id_map(region_units: &[Vec<Val>], scope: &[Op]) -> Vec<(UnitKey, RegionIndex)> {
    let mut map = Vec::new();
    let mut index = RegionIndex(0);
    for units in region_units {
        for unit in units {
            match defining_op(*unit, scope) {
                Some(Op::Dataflow(dataflow::Op::GetUnit {
                    residency, unit, ..
                })) => map.push((
                    UnitKey {
                        unit: *unit,
                        residency: *residency,
                    },
                    index,
                )),
                Some(Op::Dataflow(dataflow::Op::CreateGroup { unit_ids, .. })) => {
                    for member in unit_ids {
                        if let Some(Op::Dataflow(dataflow::Op::GetUnit {
                            residency, unit, ..
                        })) = defining_op(*member, scope)
                        {
                            map.push((
                                UnitKey {
                                    unit: *unit,
                                    residency: *residency,
                                },
                                index,
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
        index = RegionIndex(index.0 + 1);
    }
    map
}

// crustify:todo: e105_LowerSyncOperation
// crustify:todo: e106_LowerNOPOperation
// crustify:todo: e111_LowerForOperation
// crustify:todo: e115_LowerReturnOperation
// crustify:todo: e116_LowerSetSendDestinationOperation
// crustify:todo: e123_LowerYieldOperation
// crustify:todo: e125_LowerUniformYieldOperation
// crustify:todo: e126_LowerUniformOperations

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::units::Core;

    #[test]
    fn a_group_files_every_unit_it_holds_under_one_region() {
        let core = Core::checked(0).expect("core 0");
        let l3su = dataflow::Op::GetUnit {
            result: Val(0),
            residency: Residency::CoreWide { core },
            unit: DfirUnit::L3su,
            num_folds: None,
        };
        let lxlu = dataflow::Op::GetUnit {
            result: Val(1),
            residency: Residency::CoreWide { core },
            unit: DfirUnit::Lxlu,
            num_folds: None,
        };
        let group = dataflow::Op::CreateGroup {
            result: Val(2),
            unit_ids: vec![Val(0), Val(1)],
        };
        let scope = vec![Op::Dataflow(l3su), Op::Dataflow(lxlu), Op::Dataflow(group)];
        // Region 0 lists the group; region 1 lists one unit directly.
        let map = fill_unit_to_id_map(&[vec![Val(2)], vec![Val(1)]], &scope);
        assert_eq!(
            map,
            vec![
                (
                    UnitKey {
                        unit: DfirUnit::L3su,
                        residency: Residency::CoreWide { core }
                    },
                    RegionIndex(0)
                ),
                (
                    UnitKey {
                        unit: DfirUnit::Lxlu,
                        residency: Residency::CoreWide { core }
                    },
                    RegionIndex(0)
                ),
                (
                    UnitKey {
                        unit: DfirUnit::Lxlu,
                        residency: Residency::CoreWide { core }
                    },
                    RegionIndex(1)
                ),
            ]
        );
    }
}

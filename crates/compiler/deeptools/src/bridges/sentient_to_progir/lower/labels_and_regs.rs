//! THE LABELS, THE CODE GRAPH AND THE REGISTER IMMEDIATES — AddToLabelsMap,
//! updateLabelAndAddToCodeGraph, addToRegInit, GetAddressScale, GetOpCodePrefix, getRegImmVals.
//!
//! 6 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e008_addToRegInit` | 0 | 41 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:96` |
//! | `e009_GetAddressScale` | 0 | 16 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:502` |
//! | `e010_GetOpCodePrefix` | 0 | 18 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:519` |
//! | `e068_getRegImmVals` | 1 | 102 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:1140` |
//! | `e093_AddToLabelsMap` | 2 | 8 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:32` |
//! | `e094_updateLabelAndAddToCodeGraph` | 2 | 40 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:47` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

use crate::arch::Arch;
use crate::islands::sentient::dialects::sentient::RegType;
use sys_arch_spec::regfile::Component;
use sys_arch_spec::values::OpUnit;

// crustify:todo: e008_addToRegInit

/// HOW MUCH ONE ADDRESS UNIT IS WORTH — an entry of `addressGranularityScalePerUnit`
/// (`sysdef.cpp:531-556`), which `getAddressGranularityScale` looks up by `{component, storage}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AddressScale(u32);

impl AddressScale {
    /// The multiplier, for the one place an address is divided by it.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Replaces: e009_GetAddressScale
///
/// The granularity an address operand of `comp` counts in, keyed by the file it came out of.
/// ⛔ THE L0 STORE UNIT SCALES BY `numPTRows` AND THE LOAD UNIT BY 1 (`sysdef.cpp:544-545`) — the one
/// asymmetry in the table, and an arch fact rather than a constant.
/// ⛔ EVERY OTHER COMPONENT ANSWERS 1 THROUGH HERE. The PT/PE/SFP LRF-family entries are 128, but the
/// reference's fallthrough `return 1` reaches them first, so this function never reads them.
#[must_use]
pub fn address_scale<A: Arch>(comp: Component, locale: RegType) -> AddressScale {
    match comp {
        Component::L0lu => AddressScale(1),
        Component::L0su => AddressScale(A::PT_ROWS),
        Component::Lxlu | Component::Lxsu => AddressScale(1),
        Component::L3lu | Component::L3su => match locale {
            // ⭐ BOTH KEYS ANSWER 128: `{L3*, HBM}` and `{L3*, LX}` carry the same scale, so the
            // locale decides which entry is read rather than what comes back.
            RegType::Ear | RegType::Ebr | RegType::Jcr => AddressScale(128),
            RegType::Lar | RegType::Lbr => AddressScale(128),
            // ⛔ `llvm_unreachable` THERE, ITS OWN MESSAGE HERE.
            _ => panic!("expected locale info to calculate scale in L3"),
        },
        Component::Pt | Component::Pe | Component::Sfp => AddressScale(1),
    }
}

/// Replaces: e010_GetOpCodePrefix
///
/// Which unit family names a component's opcodes — the `PTOP`/`SFP`/`PE`/`L0`/`LX`/`L3` prefix the
/// reference writes into an opcode string, as the vendored enum that prefix selected.
/// ⛔ THE UNRECOGNISED-UNIT BRANCH IS UNREPRESENTABLE, and it was worse than a refusal: it printed to
/// `llvm::errs` and left `op_code` UNTOUCHED, so the caller went on with whatever string it held.
/// ⭐ AND THE MAPPING IS THE ONE ALREADY VENDORED — a second copy is where the L0/LX split drifts.
#[must_use]
pub fn op_code_prefix(comp: Component) -> OpUnit {
    OpUnit::of_component(comp)
}

// crustify:todo: e068_getRegImmVals
// crustify:todo: e093_AddToLabelsMap
// crustify:todo: e094_updateLabelAndAddToCodeGraph

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Target;

    #[test]
    fn the_l0_store_unit_scales_by_the_pt_row_count() {
        assert_eq!(
            address_scale::<Target>(Component::L0su, RegType::Lrf).get(),
            Target::PT_ROWS,
            "{{L0SU, L0}} is numPTRows (sysdef.cpp:544)"
        );
        assert_eq!(
            address_scale::<Target>(Component::L0lu, RegType::Lrf).get(),
            1
        );
        assert_eq!(
            address_scale::<Target>(Component::L3su, RegType::Ear).get(),
            128
        );
        assert_eq!(
            address_scale::<Target>(Component::L3lu, RegType::Lbr).get(),
            128
        );
        assert_eq!(
            address_scale::<Target>(Component::Lxsu, RegType::Lar).get(),
            1
        );
        // The fallthrough: a component the lookup is never reached for.
        assert_eq!(
            address_scale::<Target>(Component::Pt, RegType::Lrf).get(),
            1
        );
    }

    #[test]
    fn the_nine_components_name_six_opcode_families() {
        assert_eq!(op_code_prefix(Component::Pt), OpUnit::Ptop);
        assert_eq!(op_code_prefix(Component::Sfp), OpUnit::Sfp);
        assert_eq!(op_code_prefix(Component::Pe), OpUnit::Pe);
        assert_eq!(op_code_prefix(Component::L0lu), OpUnit::L0);
        assert_eq!(op_code_prefix(Component::L0su), OpUnit::L0);
        assert_eq!(op_code_prefix(Component::Lxlu), OpUnit::Lx);
        assert_eq!(op_code_prefix(Component::Lxsu), OpUnit::Lx);
        assert_eq!(op_code_prefix(Component::L3lu), OpUnit::L3);
        assert_eq!(op_code_prefix(Component::L3su), OpUnit::L3);
    }
}

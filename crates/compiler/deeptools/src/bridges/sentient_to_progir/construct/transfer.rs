//! THE TRANSFER INSTRUCTIONS — the L3 and non-L3 loads and stores, the burst normalisation and
//! the load/store fusion.
//!
//! ⛔ TWO BURST DERIVATIONS EXIST IN OUR TREE AND dxp HAS ONE. Port the reference's.
//! ⛔ BOTH STORE-SIDE SYNCS WERE FOUND INVERTED ON AN EBR-MATCHED OP, and that writes zeros.
//!
//! 10 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e003_normalizeBurstSize` | 0 | 13 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2562` |
//! | `e054_ConstructL3LoadInstr` | 1 | 95 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2576` |
//! | `e055_ConstructL3StoreInstr` | 1 | 150 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2672` |
//! | `e056_ConstructZRAssignInstr` | 1 | 14 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2828` |
//! | `e057_ConstructL3LoadAndStoreInstr` | 1 | 179 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2843` |
//! | `e058_ConstructLoadComputeInstr` | 1 | 99 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3287` |
//! | `e059_ConstructLRFCopyInstr` | 1 | 49 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3509` |
//! | `e089_ConstructLoadInstr` | 2 | 201 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3024` |
//! | `e090_ConstructLoadInstr` | 2 | 60 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3226` |
//! | `e091_ConstructStoreInstr` | 2 | 121 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3387` |

use crate::arch::{Arch, Elements};
use sys_arch_spec::regfile::Component;

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

/// THE UNIT'S BURST SIZE, WHERE IT HAS ONE — `getMaxBurstSize` (`DccExtContext.cpp:338-348`).
///
/// ⛔ `-1` FOR EVERY COMPUTE UNIT, which is the reference's spelling for "no burst size at all"
/// rather than a size. As an `Option` no arithmetic can reach it.
#[must_use]
pub const fn max_burst_size<A: Arch>(comp: Component) -> Option<Elements> {
    match comp {
        Component::L3lu | Component::L3su => Some(Elements(A::L3_BURST as u64)),
        Component::Lxlu | Component::Lxsu => Some(Elements(A::LX_BURST as u64)),
        Component::L0lu | Component::L0su => Some(Elements(A::L0_BURST as u64)),
        Component::Pt | Component::Pe | Component::Sfp => None,
    }
}

/// Replaces: e003_normalizeBurstSize
///
/// The burst size as the ISA field spells it: the maximum wraps to 0 and 0 means one beat.
///
/// ⛔ THE REFERENCE'S RANGE CHECK IS A TAUTOLOGY — `(burst_size >= 0 || burst_size <= max)`
/// (`:2565-2566`) is an OR, so it holds for every `int`, and no burst size has ever been refused
/// there. [`Elements`] is unsigned, so the half it meant to test cannot be written down.
///
/// ⛔ A COMPUTE UNIT HAS NO MAXIMUM, so the wrap-to-zero arm cannot fire — matching the reference,
/// where `max_threshold` is `-1` and `burst_size == -1` is unreachable for a real size.
#[must_use]
pub const fn normalize_burst_size<A: Arch>(burst_size: Elements, comp: Component) -> Elements {
    if let Some(max) = max_burst_size::<A>(comp) {
        if burst_size.0 == max.0 {
            return Elements(0);
        }
    }
    if burst_size.0 == 0 {
        Elements(1)
    } else {
        burst_size
    }
}

// crustify:todo: e054_ConstructL3LoadInstr
// crustify:todo: e055_ConstructL3StoreInstr
// crustify:todo: e056_ConstructZRAssignInstr
// crustify:todo: e057_ConstructL3LoadAndStoreInstr
// crustify:todo: e058_ConstructLoadComputeInstr
// crustify:todo: e059_ConstructLRFCopyInstr
// crustify:todo: e089_ConstructLoadInstr
// crustify:todo: e090_ConstructLoadInstr
// crustify:todo: e091_ConstructStoreInstr

#[cfg(test)]
mod unit_tests {
    use super::{Component, Elements, normalize_burst_size};
    use crate::arch::{Arch, Target};

    /// THE THREE ARMS AND THE UNIT THAT HAS NO MAXIMUM.
    ///
    /// ⛔ THE WRAP IS PER COMPONENT: 32 is the L3's maximum and wraps to 0, but on the LX the same 32
    /// is an ordinary size that passes through — one number, two answers.
    #[test]
    fn the_maximum_wraps_to_zero_and_zero_means_one() {
        let l3 = |n| normalize_burst_size::<Target>(Elements(n), Component::L3lu);
        assert_eq!(l3(u64::from(Target::L3_BURST)), Elements(0));
        assert_eq!(l3(0), Elements(1));
        assert_eq!(l3(7), Elements(7));
        assert_eq!(
            normalize_burst_size::<Target>(Elements(u64::from(Target::L3_BURST)), Component::Lxlu),
            Elements(u64::from(Target::L3_BURST))
        );
        // A compute unit has no maximum, so only the zero arm applies.
        assert_eq!(
            normalize_burst_size::<Target>(Elements(0), Component::Pt),
            Elements(1)
        );
    }
}

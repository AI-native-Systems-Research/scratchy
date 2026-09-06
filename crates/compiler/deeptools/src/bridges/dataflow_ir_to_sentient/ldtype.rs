// SPDX-License-Identifier: Apache-2.0
//! HOW AN LX LOAD ARRANGES A SUB-STICK TRANSFER — the `ldtype` / `shuffle_mode` decision.
//!
//! Ported from `AgenToSentientLoweringPass::setldtype`
//! (`dcc/src/Conversion/AgenToSentient/Helper.cpp:1647-1711`).
//!
//! # 🛑 THE WHOLE FUNCTION IS ABOUT ONE THING: A TRANSFER NARROWER THAN A STICK
//!
//! ⭐⭐ THE HARDWARE MOVES A STICK. When a send carries less than that, the load has to say how the
//! bytes sit inside it — splatted across the stick, or written once and the rest zero-padded — and that
//! choice is the `shuffle_mode` attribute. At stick granularity there is nothing to decide and the mode
//! stays default.
//!
//! ⛔ **AND `total_elements` IS REWRITTEN TO THE FULL STICK WHENEVER A MODE IS CHOSEN.** Both branches
//! that set a mode also set `total_elements = bytesPerStick * 8 / element_width` (`:1665`, `:1707`) —
//! its own comment says *"total_elements should reflect full stick"*. So the count the op carries stops
//! being the count of useful elements and becomes the count the stick holds. Emitting the useful count
//! beside a non-default mode describes a transfer the hardware will not perform.
//!
//! # 🛑 TWO ROUTES TO A MODE, AND THEY DISAGREE ABOUT WHAT IS SUPPORTED
//!
//! ⭐ **EXPLICIT** — a `vectorchain.shuffle` feeding the send states the arrangement, and its
//! `repetition` count plus its splat/pad shape selects one of three modes (`:1667-1687`).
//!
//! ⭐ **IMPLICIT** — no shuffle, but the send is not at stick granularity, so the reference picks a mode
//! from the byte count alone (`:1690-1706`), under a comment worth quoting: *"If a shuffleOp wasn't used
//! to explicitly state a non-default ldtype, assume user doesn't care how data is arranged (pad versus
//! splat). In this case for 16B data transfers, use zpad16b (mode 2). 2B data transfers only support
//! splat."*
//!
//! ⛔⛔ SO THE IMPLICIT ROUTE ADMITS ONLY 2 AND 16 BYTES, WHILE THE EXPLICIT ROUTE ADMITS
//! `splat16b` AS WELL. The two are not the same set, and the difference is not arbitrary: 2-byte
//! transfers *only* support splat, so a 2-byte implicit transfer gets `splat2b` rather than the
//! zero-pad the 16-byte case gets.
//!
//! ⛔ EVERY OTHER SIZE IS `"unsupported ldtype"` in the reference — an `emitOpError`. Here the
//! byte count that selects a mode is a closed set ([`SubStick`]) so there is no arm to refuse in.

use crate::arch::{Arch, Bytes};
use crate::islands::sentient::dialects::sentient::ShuffleMode;
use crate::units::DfirUnit;

/// WHETHER A UNIT HAS NON-DEFAULT LOAD TYPES AT ALL.
///
/// ⛔⛔ LX ONLY, AND THE REFERENCE OPENS WITH IT: `if (!is_any_of(comp, LXLU, LXSU)) return success()`
/// (`:1653`), above the comment *"non-default ldtypes currently support for LX only"*. So this decision
/// does not exist for the L0, L3, PT, PE or SFP — asking it of them is not a different answer, it is a
/// question that does not apply.
///
/// ⛔ EXHAUSTIVE, NO WILDCARD: a new unit kind must say whether it is an LX half.
#[must_use]
pub const fn has_load_types(unit: DfirUnit) -> bool {
    matches!(unit, DfirUnit::Lxlu | DfirUnit::Lxsu)
}

/// A TRANSFER WIDTH NARROWER THAN A STICK, as the implicit route admits it.
///
/// ⛔ TWO SIZES ONLY — the reference's implicit branch tests `num_bytes == 2` and `num_bytes == 16` and
/// calls everything else `"unsupported ldtype"` (`:1697-1704`). A closed set here is that refusal made
/// unreachable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubStick {
    /// Two bytes. ⛔ SPLAT ONLY — *"2B data transfers only support splat."*
    TwoBytes,
    /// Sixteen bytes.
    SixteenBytes,
}

/// WHAT A `vectorchain.shuffle` FEEDING A SEND STATES — the explicit route's three shapes.
///
/// ⛔ THE SHAPE **AND** THE REPETITION COUNT BOTH MATTER, and the pairs are fixed (`:1667-1685`):
/// a 2-byte splat with 64 repetitions, a 16-byte right-zero-pad with 1, a 16-byte splat with 8. A
/// matching shape with the wrong count is `"unsupported ldtype"`, so the count is part of the identity
/// of the mode rather than a parameter of it — which is why these are variants and not fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Explicit {
    /// `isSplatFromFirstElem(shuffle, 2)` with `repetition == 64` — mode 1.
    Splat2BAcross64,
    /// `isRightZeroPadFromFirstElem(shuffle, 16)` with `repetition == 1` — mode 2.
    ///
    /// ⛔⛔ THIS ONE CARRIES AN EXTRA LEGALITY CHECK IN THE REFERENCE: the shuffle's *result* must be a
    /// whole stick, or it is *"LX loads involving explicit padding should be at stick granularity"*
    /// (`:1677-1681`). See [`ZeroPad16`], which is where that check goes.
    ZeroPad16BOnce,
    /// `isSplatFromFirstElem(shuffle, 16)` with `repetition == 8` — mode 3.
    Splat16BAcross8,
}

impl Explicit {
    /// The mode this shape selects.
    #[must_use]
    pub const fn mode(self) -> ShuffleMode {
        match self {
            Explicit::Splat2BAcross64 => ShuffleMode::Splat2B,
            Explicit::ZeroPad16BOnce => ShuffleMode::ZeroPad16B,
            Explicit::Splat16BAcross8 => ShuffleMode::Splat16B,
        }
    }
}

/// A ZERO-PAD SHUFFLE WHOSE RESULT IS A WHOLE STICK — the reference's extra check, as a witness.
///
/// ⛔⛔ THE CHECK IS ON THE SHUFFLE'S **RESULT**, NOT ITS INPUT. `result_bitwidth * num_elems / 8 !=
/// bytesPerStick` is an `emitOpError` (`:1677-1681`), and its own comment is *"Return type of shuffle
/// should reflect a stick"*. So an explicit zero-pad is only legal when what it produces fills a stick —
/// the padding is what makes it a stick, and a pad that leaves it short is the error.
///
/// ⭐ A WITNESS RATHER THAN A CHECK: [`ZeroPad16::of_whole_stick`] is the only constructor and it takes
/// the arch, so the mode cannot be selected without the stick width agreeing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroPad16(());

impl ZeroPad16 {
    /// A zero-pad whose result fills a stick on this arch, or `None` if it does not.
    ///
    /// ⛔ `None` IS "NOT THIS MODE", NOT A FAILURE. A shuffle whose result is short simply is not an
    /// explicit zero-pad; the caller has the other two shapes and the implicit route to consider.
    #[must_use]
    pub fn of_whole_stick<A: Arch>(result: Bytes) -> Option<ZeroPad16> {
        (result == Bytes(A::BYTES_PER_STICK.get())).then_some(ZeroPad16(()))
    }
}

/// THE MODE AN LX LOAD TAKES, AND THE ELEMENT COUNT THAT MUST GO WITH IT.
///
/// ⛔⛔ THE PAIR IS ONE VALUE BECAUSE THE REFERENCE SETS BOTH TOGETHER, TWICE. Every branch that
/// chooses a mode also rewrites `total_elements` to the full stick; returning the mode alone would let a
/// caller emit it beside the useful-element count, which describes a transfer the hardware does not
/// perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadType {
    /// `shuffle_mode`.
    pub mode: ShuffleMode,
    /// `total_elements`, rewritten to what the STICK holds — not what the transfer needs.
    pub total_elements: u32,
}

impl LoadType {
    /// How many elements of `element_width` bits fill one stick — `bytesPerStick * 8 / element_width`.
    #[must_use]
    pub const fn elements_per_stick<A: Arch>(element_bits: u32) -> u32 {
        (A::BYTES_PER_STICK.get() as u32) * 8 / element_bits
    }

    /// THE EXPLICIT ROUTE — a shuffle feeding the send states the arrangement.
    #[must_use]
    pub const fn explicit<A: Arch>(shape: Explicit, element_bits: u32) -> LoadType {
        LoadType {
            mode: shape.mode(),
            total_elements: Self::elements_per_stick::<A>(element_bits),
        }
    }

    /// THE IMPLICIT ROUTE — no shuffle, and the send is narrower than a stick.
    ///
    /// ⛔ TWO WIDTHS, TWO MODES, AND THEY ARE NOT SYMMETRIC: two bytes splats because that is all a
    /// 2-byte transfer supports; sixteen zero-pads because the reference assumes a caller with no
    /// shuffle *"doesn't care how data is arranged"* and picks the pad.
    #[must_use]
    pub const fn implicit<A: Arch>(width: SubStick, element_bits: u32) -> LoadType {
        LoadType {
            mode: match width {
                SubStick::TwoBytes => ShuffleMode::Splat2B,
                SubStick::SixteenBytes => ShuffleMode::ZeroPad16B,
            },
            total_elements: Self::elements_per_stick::<A>(element_bits),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::Dd2;

    /// 🎯 THE DECISION DOES NOT APPLY OUTSIDE THE LX.
    #[test]
    fn only_the_lx_halves_have_load_types() {
        assert!(has_load_types(DfirUnit::Lxlu));
        assert!(has_load_types(DfirUnit::Lxsu));
        for unit in [
            DfirUnit::L3lu,
            DfirUnit::L3su,
            DfirUnit::L0lu,
            DfirUnit::L0su,
            DfirUnit::Pe,
            DfirUnit::Sfp,
        ] {
            assert!(!has_load_types(unit), "{unit:?} has no non-default ldtype");
        }
    }

    /// 🎯 THE THREE EXPLICIT SHAPES SELECT THREE DISTINCT MODES.
    #[test]
    fn the_three_explicit_shapes() {
        assert_eq!(Explicit::Splat2BAcross64.mode(), ShuffleMode::Splat2B);
        assert_eq!(Explicit::ZeroPad16BOnce.mode(), ShuffleMode::ZeroPad16B);
        assert_eq!(Explicit::Splat16BAcross8.mode(), ShuffleMode::Splat16B);
    }

    /// 🎯 CHOOSING A MODE REWRITES `total_elements` TO THE FULL STICK.
    ///
    /// ⭐ THE WHOLE POINT OF PAIRING THEM. At 16 bits an element and a 128-byte stick that is 64
    /// elements, whatever the transfer actually needed.
    #[test]
    fn a_mode_carries_the_full_stick_count() {
        let fp16 = 16;
        assert_eq!(LoadType::elements_per_stick::<Dd2>(fp16), 64);
        assert_eq!(
            LoadType::explicit::<Dd2>(Explicit::Splat2BAcross64, fp16).total_elements,
            64
        );
        assert_eq!(
            LoadType::implicit::<Dd2>(SubStick::TwoBytes, fp16).total_elements,
            64
        );
        // ⭐ AND AT 8 BITS IT IS 128 — the count follows the element width, not the transfer.
        assert_eq!(LoadType::elements_per_stick::<Dd2>(8), 128);
    }

    /// 🎯 THE IMPLICIT ROUTE IS ASYMMETRIC: 2B splats, 16B pads.
    ///
    /// ⛔ NOT A PREFERENCE. A 2-byte transfer only supports splat, so the two widths genuinely differ
    /// rather than one being a default for both.
    #[test]
    fn the_implicit_route_splats_two_bytes_and_pads_sixteen() {
        assert_eq!(
            LoadType::implicit::<Dd2>(SubStick::TwoBytes, 16).mode,
            ShuffleMode::Splat2B
        );
        assert_eq!(
            LoadType::implicit::<Dd2>(SubStick::SixteenBytes, 16).mode,
            ShuffleMode::ZeroPad16B
        );
    }

    /// 🎯 AN EXPLICIT ZERO-PAD IS ONLY LEGAL WHEN ITS RESULT FILLS A STICK.
    #[test]
    fn a_short_zero_pad_is_not_that_mode() {
        assert!(ZeroPad16::of_whole_stick::<Dd2>(Bytes(128)).is_some());
        assert!(ZeroPad16::of_whole_stick::<Dd2>(Bytes(16)).is_none());
    }
}

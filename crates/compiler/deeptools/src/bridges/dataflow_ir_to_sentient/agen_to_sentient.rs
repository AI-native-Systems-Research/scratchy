// SPDX-License-Identifier: Apache-2.0
//! `AgenToSentientLoweringPass` — the transfer statements, ported function by function.
//!
//! Each item names its entry in `docs/bridge2-porting-order.md` and cites the C++ it came from.

use crate::arch::{Arch, Elements};
use crate::islands::dataflow_ir::dialects::Val;
use crate::islands::sentient::dialects::sentient as sen;
use crate::units::DfirUnit;

/// WHETHER A UNIT TAKES THE L3'S ADDRESS RULES — `is_any_of(comp, L3LU, L3SU)`.
#[must_use]
pub const fn is_l3(unit: DfirUnit) -> bool {
    matches!(unit, DfirUnit::L3lu | DfirUnit::L3su)
}

/// WHERE A TRANSFER'S IMMUTABLE ADDRESS COMES FROM.
///
/// ⛔ TWO SOURCES, AND THEY ARE NOT INTERCHANGEABLE: the memory view's own start address (an SSA value
/// the program already bound) or a freshly minted constant. A `Val` alone could not say which, and the
/// caller has to mint the constant into the preamble before it has one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImmutableAddr {
    /// The memory view's start address, unchanged.
    MemoryStart,
    /// A `sentient.scalar_constant` of this value, minted before the program unit.
    Constant(i64),
}

/// WHAT ONE TRANSFER'S ADDRESSING RESOLVES TO — ⛔ BOTH FIELDS, ALWAYS SET TOGETHER.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Addressing {
    /// `immutable_addr`.
    pub immutable_addr: ImmutableAddr,
    /// `increment`, as the value of the constant to mint.
    pub increment: i64,
}

/// **316/490** `setImmutableAddrAndIncrements` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1581` (43L).
///
/// ⚠️ NOT A PORTED UNIT. Entry 214/384's home is `agen_helper.rs` (UNITS.tsv) and it carries no
/// `/// Replaces:` anchor; this is the pre-campaign predicate, with the emission and the newtypes
/// still missing. The PORT/AUDIT ticks for 214 were removed on that ground.
///
/// # 🛑 FOUR CASES, NOT TWO, AND THE TWO AXES ARE INDEPENDENT
///
/// ⛔⛔ THE FORK IS `is_l3` **CROSSED WITH** `perform_burst_or_groups`, and reading it as one fork gets
/// two of the four wrong:
///
/// | | burst or group | neither |
/// |---|---|---|
/// | **L3** | immutable = mem-view start; increment = `total_elements * burst_size` | immutable = mem-view start; increment = **0** |
/// | **other** | immutable = `stride_size`; increment = **`stride_size` TOO** | immutable = **0**; increment = **0** |
///
/// ⛔⛔ THE NON-L3 BURST ARM SETS **BOTH** FIELDS TO `stride_size` (`:1604-1610`) — two separate
/// `sentient::ConstantOp`s of the same value. An earlier reading of this function recorded only the
/// immutable address and left the increment alone, which leaves a bursting LX transfer advancing by
/// whatever the default was.
///
/// ⭐ AND THE L3's IMMUTABLE ADDRESS IS SET AT THE TOP, BEFORE EITHER BRANCH (`:1586-1589`):
/// *"In case of L3, immutable_addr is always mem_view_start_addr."* So the L3 never mints an immutable
/// address at all, in either column.
///
/// ⭐⭐ WHY THE CONSTANTS APPEAR ABOVE THE PROGRAM UNITS: this function moves the builder to
/// `setInsertionPoint(unit_op)` before creating them (`:1593-1594`) and restores it after. That is the
/// mechanism behind the hoisted `sentient.scalar_constant`s in the reference's output, and it is why
/// they are the caller's to mint here rather than pushed into a unit's body.
///
/// ⛔ THE ZERO INCREMENT IS DELIBERATE AND DOCUMENTED, not a missing case: *"If there is no burst size
/// or IL groups, the increment is always zero in the AgenToSentient lowering. The later passes will
/// modify increment value to non-zero values depending on possibilities. This is a statement
/// irrespective of L3/LX/L0 LU/SU units."* (`:1613-1616`).
#[must_use]
pub const fn set_immutable_addr_and_increments(
    unit: DfirUnit,
    perform_burst_or_groups: bool,
    stride_size: i64,
    burst_size: i64,
    total_elements: i64,
) -> Addressing {
    if perform_burst_or_groups {
        if is_l3(unit) {
            // ⭐ THE REFERENCE LEAVES AN UNPROVEN ASSUMPTION HERE, as a comment:
            // *"TODO: assert that stride_size x precision = stick size"* (`:1598`).
            Addressing {
                immutable_addr: ImmutableAddr::MemoryStart,
                increment: total_elements * burst_size,
            }
        } else {
            Addressing {
                immutable_addr: ImmutableAddr::Constant(stride_size),
                // ⛔ THE SAME VALUE, INTO THE OTHER FIELD.
                increment: stride_size,
            }
        }
    } else if is_l3(unit) {
        Addressing {
            immutable_addr: ImmutableAddr::MemoryStart,
            increment: 0,
        }
    } else {
        Addressing {
            immutable_addr: ImmutableAddr::Constant(0),
            increment: 0,
        }
    }
}

/// WHAT A LOAD'S DATA REACHES, AND THROUGH WHAT.
///
/// ⛔ THE REFERENCE RETURNS A **PAIR** — the consuming op and the op defining the unit it sends to —
/// and both must be non-null or the caller errors with *"can not extract the loadOp's consumer!"*
/// (`constructLoadAndSendStmt`, `:1938-1941`). So the consumer's UNIT is as much a result as the send.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadConsumer {
    /// The unit the send names — `send_op.getToUnit().getDefiningOp()`.
    pub to_unit: Val,
    /// Whether a rearrangement sits between the load and the send.
    pub via: Option<Rearrangement>,
}

/// THE THREE OPS THAT MAY SIT BETWEEN A LOAD AND ITS SEND.
///
/// ⛔ **INCLUDING `rotate`**, which the composite-region check does NOT admit
/// (`checkCompositeRegion` allows select and shuffle only, `Helper.cpp:216-220`). Two different sets,
/// fifty lines apart; reading one for the other admits a rotate where the reference refuses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rearrangement {
    /// `vectorchain.select`.
    Select,
    /// `vectorchain.shuffle`.
    Shuffle,
    /// `vectorchain.rotate`.
    Rotate,
}

/// **133/490** `getLoadConsumer` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1242` (36L).
///
/// ⛔⛔ **THE LOAD MUST HAVE EXACTLY ONE USE**: *"Assume single consumer per loadOp"* — `hasOneUse()`
/// or the error *"loadOp should have one consumer"* (`:1255-1259`). And where a rearrangement
/// intervenes it must ALSO have one use (`:1269`), so the chain is single-use at every hop.
///
/// ⛔ THREE SHAPES, AND THE THIRD RETURNS NOTHING: a send; a select/shuffle/rotate then a send; or a
/// store. A store yields no consumer — the reference falls through to *"unsupported loadOp
/// consumer!"* for anything it does not recognise, and a store is handled by a different lowering
/// entirely.
///
/// ⭐ WHICH ROOT IS FOLLOWED DEPENDS ON THE LOAD (`:1245-1251`): a `vector_load`, indirect load or
/// symbolic load is followed from its RESULT, while a composite load is followed from its **load
/// induction variable** — the block argument its region binds. Following the result of a composite
/// load would follow the wrong value.
#[must_use]
pub const fn load_consumer(to_unit: Val, via: Option<Rearrangement>) -> LoadConsumer {
    LoadConsumer { to_unit, via }
}

/// **130/490** `setldtype` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1647` (60L).
///
/// # 🛑 IT SETS THREE THINGS, AND `total_elements` IS ONE OF THEM
///
/// ⛔⛔ WHENEVER A MODE IS CHOSEN, `total_elements` IS REWRITTEN TO THE FULL STICK —
/// `bytesPerStick * 8 / element_width`, in both branches (`:1648`, `:1690`), under the comment
/// *"total_elements should reflect full stick. element_width is in bits."* So the count the op carries
/// stops being the useful-element count. Emitting the useful count beside a non-default mode describes
/// a transfer the hardware will not perform.
///
/// ⛔ **LX ONLY.** The function returns immediately for anything else: *"non-default ldtypes currently
/// support for LX only"*.
///
/// ⛔⛔ TWO ROUTES WITH DIFFERENT ADMISSIBLE SETS. **Explicit** — a `vectorchain.shuffle` feeding the
/// send states the arrangement, and the shape AND its repetition count together select one of three
/// modes: splat-2B with 64 repetitions, right-zero-pad-16B with 1, splat-16B with 8. A matching shape
/// with the wrong count is *"unsupported ldtype"*, so the count is part of the mode's identity.
/// **Implicit** — no shuffle, and the send is not at stick granularity, so the mode comes from the byte
/// count alone: 2 bytes splats, 16 bytes zero-pads, everything else is *"unsupported ldtype"*.
///
/// ⛔ THE ASYMMETRY IS NOT A PREFERENCE. The reference's own comment: *"for 16B data transfers, use
/// zpad16b (mode 2). 2B data transfers only support splat."*
///
/// ⛔ AND THE EXPLICIT ZERO-PAD CARRIES AN EXTRA CHECK ON THE SHUFFLE'S **RESULT** (`:1655-1663`): its
/// width must be a whole stick, or *"LX loads involving explicit padding should be at stick
/// granularity"*. On the result, not the input — the padding is what makes it a stick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadType {
    /// `shuffle_mode`.
    pub mode: sen::ShuffleMode,
    /// `total_elements`, ⛔ REWRITTEN TO WHAT THE STICK HOLDS whenever `mode` is not the default.
    pub total_elements: Elements,
}

/// The explicit route's three shapes — ⛔ SHAPE AND REPETITION TOGETHER.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Explicit {
    /// `isSplatFromFirstElem(shuffle, 2)` with `repetition == 64` — mode 1.
    Splat2BAcross64,
    /// `isRightZeroPadFromFirstElem(shuffle, 16)` with `repetition == 1` — mode 2, and its result must
    /// be a whole stick.
    ZeroPad16BOnce,
    /// `isSplatFromFirstElem(shuffle, 16)` with `repetition == 8` — mode 3.
    Splat16BAcross8,
}

/// The implicit route's two widths — ⛔ THE ONLY TWO the reference admits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubStick {
    /// Two bytes. ⛔ SPLAT ONLY.
    TwoBytes,
    /// Sixteen bytes.
    SixteenBytes,
}

impl LoadType {
    /// How many elements of `element_width` BITS fill one stick.
    #[must_use]
    pub const fn elements_per_stick<A: Arch>(element_width: u32) -> Elements {
        Elements((A::BYTES_PER_STICK.get() * 8) / element_width as u64)
    }

    /// THE DEFAULT — not an LX unit, or already at stick granularity.
    ///
    /// ⛔ `total_elements` IS LEFT ALONE HERE, which is the whole difference from the other two.
    #[must_use]
    pub const fn default_mode(total_elements: Elements) -> LoadType {
        LoadType {
            mode: sen::ShuffleMode::NoShuffle,
            total_elements,
        }
    }

    /// THE EXPLICIT ROUTE.
    #[must_use]
    pub const fn explicit<A: Arch>(shape: Explicit, element_width: u32) -> LoadType {
        LoadType {
            mode: match shape {
                Explicit::Splat2BAcross64 => sen::ShuffleMode::Splat2B,
                Explicit::ZeroPad16BOnce => sen::ShuffleMode::ZeroPad16B,
                Explicit::Splat16BAcross8 => sen::ShuffleMode::Splat16B,
            },
            total_elements: Self::elements_per_stick::<A>(element_width),
        }
    }

    /// THE IMPLICIT ROUTE.
    #[must_use]
    pub const fn implicit<A: Arch>(width: SubStick, element_width: u32) -> LoadType {
        LoadType {
            mode: match width {
                SubStick::TwoBytes => sen::ShuffleMode::Splat2B,
                SubStick::SixteenBytes => sen::ShuffleMode::ZeroPad16B,
            },
            total_elements: Self::elements_per_stick::<A>(element_width),
        }
    }

    /// WHETHER A TRANSFER IS ALREADY A WHOLE STICK — the test that gates the implicit route
    /// (`:1674`): `element_width * total_elements / 8 != bytesPerStick`.
    #[must_use]
    pub const fn is_stick_granular<A: Arch>(element_width: u32, total_elements: Elements) -> bool {
        (element_width as u64 * total_elements.0) / 8 == A::BYTES_PER_STICK.get()
    }
}

/// **131/490** `generateSetSendDestinationStmts` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2731` (49L).
///
/// ⛔⛔ **LXLU ONLY, AND IT RETURNS SUCCESS OTHERWISE** (`:2735-2738`) — so `sentient.set_send_dst` is
/// not a statement any unit may carry. ⛔ NOT `Lxsu`: the store half is excluded even though it is the
/// LX's other half, which is easy to lose when the rule is remembered as "the LX does this".
#[must_use]
pub const fn sets_send_destination(unit: DfirUnit) -> bool {
    matches!(unit, DfirUnit::Lxlu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::Dd2;

    /// 🎯 316 — ALL FOUR CASES, because the two axes are independent.
    #[test]
    fn the_four_addressing_cases() {
        // L3 + burst: the increment is total_elements * burst; the address stays the view's.
        assert_eq!(
            set_immutable_addr_and_increments(DfirUnit::L3lu, true, 64, 32, 64),
            Addressing { immutable_addr: ImmutableAddr::MemoryStart, increment: 2048 }
        );
        // ⛔ non-L3 + burst: BOTH fields take stride_size. This is the case an earlier reading missed.
        assert_eq!(
            set_immutable_addr_and_increments(DfirUnit::Lxlu, true, 64, 32, 64),
            Addressing { immutable_addr: ImmutableAddr::Constant(64), increment: 64 }
        );
        // L3, no burst: the view's address, zero increment.
        assert_eq!(
            set_immutable_addr_and_increments(DfirUnit::L3su, false, 64, 0, 64),
            Addressing { immutable_addr: ImmutableAddr::MemoryStart, increment: 0 }
        );
        // non-L3, no burst: both zero.
        assert_eq!(
            set_immutable_addr_and_increments(DfirUnit::Lxlu, false, 64, 0, 64),
            Addressing { immutable_addr: ImmutableAddr::Constant(0), increment: 0 }
        );
    }

    /// 🎯 316 — THE L3 NEVER MINTS AN IMMUTABLE ADDRESS, in either column.
    #[test]
    fn the_l3_always_keeps_the_view_start() {
        for burst in [true, false] {
            for unit in [DfirUnit::L3lu, DfirUnit::L3su] {
                assert_eq!(
                    set_immutable_addr_and_increments(unit, burst, 7, 4, 64).immutable_addr,
                    ImmutableAddr::MemoryStart
                );
            }
        }
    }

    /// 🎯 316 — AND THE GOLDEN'S OWN NUMBER. `group_0__g0_0_mul`'s l3lu transfer prints
    /// `src_inc(%2)` where `%2 = 2048`, with `total_elements = 64` and `burst_size = 32`.
    #[test]
    fn the_l3_increment_matches_the_reference_output() {
        assert_eq!(
            set_immutable_addr_and_increments(DfirUnit::L3lu, true, 64, 32, 64).increment,
            2048,
            "64 * 32, as the reference's SentientIR for that program carries"
        );
    }

    /// 🎯 130 — A MODE REWRITES `total_elements` TO THE FULL STICK; the default leaves it.
    #[test]
    fn choosing_a_mode_rewrites_the_element_count() {
        assert_eq!(LoadType::default_mode(Elements(4)).total_elements, Elements(4));
        assert_eq!(
            LoadType::explicit::<Dd2>(Explicit::Splat2BAcross64, 16).total_elements,
            Elements(64),
            "a 128-byte stick holds 64 sixteen-bit elements"
        );
        assert_eq!(LoadType::implicit::<Dd2>(SubStick::TwoBytes, 8).total_elements, Elements(128));
    }

    /// 🎯 130 — THE IMPLICIT ROUTE IS ASYMMETRIC: 2B splats, 16B pads.
    #[test]
    fn the_implicit_route_is_asymmetric() {
        assert_eq!(
            LoadType::implicit::<Dd2>(SubStick::TwoBytes, 16).mode,
            sen::ShuffleMode::Splat2B
        );
        assert_eq!(
            LoadType::implicit::<Dd2>(SubStick::SixteenBytes, 16).mode,
            sen::ShuffleMode::ZeroPad16B
        );
    }

    /// 🎯 130 — THE THREE EXPLICIT SHAPES PICK THREE DISTINCT MODES.
    #[test]
    fn the_explicit_shapes_are_distinct() {
        let m = |s| LoadType::explicit::<Dd2>(s, 16).mode;
        assert_eq!(m(Explicit::Splat2BAcross64), sen::ShuffleMode::Splat2B);
        assert_eq!(m(Explicit::ZeroPad16BOnce), sen::ShuffleMode::ZeroPad16B);
        assert_eq!(m(Explicit::Splat16BAcross8), sen::ShuffleMode::Splat16B);
    }

    /// 🎯 130 — AND STICK GRANULARITY IS WHAT GATES THE IMPLICIT ROUTE AT ALL.
    ///
    /// ⭐ `group_0__g0_0_mul`'s lxlu load is 64 f16 elements = 128 bytes = one stick, so it takes the
    /// default mode — which is why that golden says `shuffle_mode = noshuffle`.
    #[test]
    fn a_whole_stick_transfer_takes_no_mode() {
        assert!(LoadType::is_stick_granular::<Dd2>(16, Elements(64)));
        assert!(!LoadType::is_stick_granular::<Dd2>(16, Elements(1)));
        assert!(!LoadType::is_stick_granular::<Dd2>(16, Elements(8)));
    }

    /// 🎯 131 — ONLY THE LX **LOAD** UNIT SETS A SEND DESTINATION.
    #[test]
    fn only_lxlu_sets_send_destinations() {
        assert!(sets_send_destination(DfirUnit::Lxlu));
        assert!(!sets_send_destination(DfirUnit::Lxsu), "the store half is excluded");
        for unit in [DfirUnit::L3lu, DfirUnit::L0lu, DfirUnit::Pe, DfirUnit::Sfp] {
            assert!(!sets_send_destination(unit));
        }
    }

    /// 🎯 133 — A LOAD MAY REACH ITS SEND DIRECTLY OR THROUGH A REARRANGEMENT, INCLUDING A ROTATE.
    #[test]
    fn the_load_consumer_shapes() {
        assert_eq!(load_consumer(Val(5), None).to_unit, Val(5));
        assert_eq!(
            load_consumer(Val(5), Some(Rearrangement::Rotate)).via,
            Some(Rearrangement::Rotate),
            "rotate is admitted here, unlike inside a composite region"
        );
    }
}

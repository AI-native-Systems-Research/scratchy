// SPDX-License-Identifier: Apache-2.0
//! ⭐ A PAGE PART NEVER SHARES A GROUP — the type that carries the law.
//!
//! | the wall (what a test had to pin) | the type |
//! |---|---|
//! | "a group holding ops of both parts would write one part's rows onto the other's page — a cache write nothing downstream can detect" — pinned by a runtime offset comparison in the fusion walk *plus* an `assert!` over every op of every group, re-reading the owner op's field | [`PagePart`] — an opaque newtype the cutting law stamps, compared as a type in the fusion walk, and carried to the launch from the walk's own trip. The assert is deleted: with one source there is nothing left to assert against, and a runtime assert is a refusal the compiler never sees. |
//!
//! A group is one launch, and a launch shifts the KV segment base ONCE, to the page holding the
//! group's cursor plus its part. The law lives at three doors, all typed:
//!
//! * **the CUT** — [`PagedKvPool::write_parts`](crate::sdsc_abstract::PagedKvPool::write_parts)
//!   stamps every part it mints with its [`PagePart`], so the emitter's op stamp COPIES the cut's
//!   own value instead of re-deriving it (the `if rows_are_requests { 0 } else { part.row0 }`
//!   that used to live at two stamp sites WAS the re-derivation);
//! * **the BREAK** — a trip carries its op's [`PagePart`] and the fusion walk compares parts
//!   beside requests, over an opaque type that cannot be confused with a slot index, a page
//!   count or a `rep` (every one of those is also a small integer — the exact mistake
//!   `TripRequest`'s own doc records);
//! * **the SHIFT** — the launch index reads the group's part off the group's own first trip,
//!   never off the owner op, so "the members disagree with the shift" is not a state the emitter
//!   can construct. It was an `assert!` (lower_subtile_tape_to_superdsc's "one page part per
//!   group, of ANY kind" guard); the type deleted it.
//!
//! NO RAW SCALARS CROSS THESE DOORS: the part's quantity is [`Slots`], a newtype, and the one
//! arithmetic that combines it with anything — advancing a write-cursor position by a part — is
//! `SlotPos::advanced_by`, so no caller ever adds a bare number to a position. `at_cursor()` and
//! a cut part whose row 0 is the cursor are the same value, deliberately: an untagged op must
//! stay fusable with a part-0 neighbour, and over-breaking costs a launch where under-breaking
//! costs correctness — the same argument `TripRequest` makes for request 0.
//!
//! [`PagePart::at_slots`] is the RELAY mint: the proc-macro that renders a baked bundle into
//! generated code rebuilds a part from its slot count, because a literal is the only thing a
//! token stream can carry. It is the one machine-written hop; hand-written code takes parts from
//! the cut.

/// SLOTS PAST THE LAUNCH'S WRITE CURSOR — a COUNT, as a newtype.
///
/// Not a position (`SlotPos` is that, on the runtime side), not a page
/// index, not a byte offset: every one of those is also a small integer, and this file's history
/// is a list of exactly that substitution. The number enters ONCE, at [`Slots::of`], and leaves
/// only through [`Slots::get`] — consumed by the runtime's typed position arithmetic
/// (`SlotPos::advanced_by`) and the macro's literal relay, never by caller-side arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Slots(u32);

impl Slots {
    /// The empty count — the cursor's own page.
    pub const ZERO: Slots = Slots(0);

    /// Where the number enters. The newtype's one wide door, as every newtype here has one.
    pub const fn of(slots: u32) -> Slots {
        Slots(slots)
    }

    /// The number, for the two readers that cannot hold the type: the macro's literal relay and
    /// the runtime's position arithmetic.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// WHICH PAGE OF A MULTI-PAGE WINDOW AN OP ADDRESSES — slots past the launch's write cursor where
/// the op's KV starts.
///
/// ⛔ NO RAW SCALAR ENTERS OR LEAVES. The part's quantity is the [`Slots`] newtype, the field is
/// private, and every consumer — the op's stamp, the trip the fusion walk compares, the launch's
/// shift, the runtime's page resolution — carries the SAME value by construction instead of
/// re-deriving it. Before this type each of those four re-derived a bare `u32`, and the
/// disagreement between them was guarded by an `assert!` that fired at bake, after the wrong
/// bundle was already built.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PagePart {
    slots: Slots,
}

impl PagePart {
    /// THE LAUNCH'S OWN CURSOR PAGE — the empty part. Every op that is not a cut part: a window
    /// that fits one page, every request-major part, every ordinary op of the body.
    pub const fn at_cursor() -> PagePart {
        PagePart { slots: Slots::ZERO }
    }

    /// THE RELAY MINT — rebuild a part the bake already cut, from its slot count. The
    /// proc-macro's generated-code path (a token stream carries a literal, not a private field)
    /// and `write_parts`' own stamp; the runtime reads [`PagePart::slots`] for the same reason —
    /// it consumes what the bake decided.
    pub const fn at_slots(slots: Slots) -> PagePart {
        PagePart { slots }
    }

    /// The part's own quantity, for the runtime's typed position arithmetic
    /// (`SlotPos::advanced_by`) and the macro's relay. Not for caller-side arithmetic — a count
    /// is combined with a position by the ONE method that knows what both are.
    pub const fn slots(self) -> Slots {
        self.slots
    }

    /// Whether this is the launch's own cursor page — the empty part every untagged op carries.
    pub const fn is_cursor(self) -> bool {
        self.slots.0 == 0
    }
}

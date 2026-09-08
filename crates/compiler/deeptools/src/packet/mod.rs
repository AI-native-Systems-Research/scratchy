// SPDX-License-Identifier: Apache-2.0
//! BRIDGE 3's SLICE BUILDERS — the four kinds of 16-byte slice a packet's flits are made of.
//!
//! A flit is eight slices, one per unit SLICE COLUMN (`0=L0SU … 7=L3LU`), and a unit's own slices come in four
//! kinds: its HEADER, its SPR configuration words, its LRF register initialisations, and its IBUFF instruction
//! words. Bridge 3 lays island 3's bound instructions out into those; this module is one file per kind, so each
//! bit layout sits beside the C++ that packs it.
//!
//! ⛔ EACH FILE IS A PORT, NOT A DERIVATION. Every layout here is `dip.cpp`'s, and where a value is not in the
//! C++ it is REFUSED rather than invented — the crate has reverted five invented tile sizes already. A captured
//! bundle may CONFIRM a layout; it may never supply one, because a bundle shows what one program contained and
//! not which of two readings the compiler follows.
//!
//! ⭐ AND THE INVARIANTS ARE `const`, NOT TESTS. A bit layout is a compile-time fact: `const _: () = assert!(…)`
//! makes a wrong shift a build error, and pinning the round trip at the LARGEST value a field holds is what found
//! the flit-count/flag-bit collision that every captured fixture was too small to show.

/// The UNIT BLOCK: one unit's slices as a run of flits, several units' blocks laid in by slice COLUMN.
pub mod block;
/// The per-unit HEADER slice: which unit, which regions follow, how many of each.
pub mod header;
/// The IBUFF slices: a unit's instruction words, padded into 16-byte slices.
pub mod ibuff;
/// The LRF slices: a unit's register initialisations.
pub mod lrf;
pub mod patch;
/// The SPR slices: a unit's 16-bit special-purpose configuration words.
pub mod spr;

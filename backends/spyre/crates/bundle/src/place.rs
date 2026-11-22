// SPDX-License-Identifier: Apache-2.0
//! OPERAND IDENTITY — what a placed operand IS, and the one site that spells it.
//!
//! `superdsc` references an operand by NAME, so something must render one; [`PlaceId`]'s
//! [`Display`](std::fmt::Display) is that one site. Nothing parses the rendering back — the
//! identity is the value.

use std::fmt;

/// A placed tensor's IDENTITY — the join key between what the bake placed and what the host binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PlaceId {
    /// A buffer the CALLER numbered.
    ///
    /// The number is the caller's, carried in on a program's bindings as a
    /// [`BufferId`](crate::program::BufferId); this crate only names it and looks it up in the
    /// caller's [`BundleLayout`](crate::BundleLayout). Whatever the number MEANS to the producer — a
    /// graph tensor index, a Triton buffer slot — is a fact of the producer and not of this type.
    Act(u32),
}

impl PlaceId {
    /// The caller-numbered buffer this names.
    pub fn tid(self) -> u32 {
        match self {
            Self::Act(t) => t,
        }
    }
}

impl fmt::Display for PlaceId {
    /// THE ONE SPELLING SITE. For diagnostics and the emitted program's own operand references;
    /// nothing parses this back, so it is a rendering, not an encoding.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Act(t) => write!(f, "t{t}"),
        }
    }
}

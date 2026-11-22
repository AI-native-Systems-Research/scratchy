// SPDX-License-Identifier: Apache-2.0
//! A KTIR program as the bake carries it: the function, what it computes, and which buffer each of
//! its parameters addresses.

/// One buffer of a bundle, as the producer numbers it: the key the bake's layout places it under.
/// The lowering never asks what a buffer holds — extents, strides and format come from the
/// program's own memory views.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct BufferId(u32);

impl BufferId {
    /// Number a buffer: distinct per buffer within one bundle.
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// The producer's number.
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for BufferId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The KV extent a decode-attention bundle sweeps — one rung of the decode ladder.
///
/// The KV storage is always the full `cap`; a bundle baked at a smaller `ActiveCap` sweeps only its
/// first [`resolve`](ActiveCap::resolve) slots of the same resident cache, so the runtime picks the
/// smallest rung covering the live KV length.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ActiveCap(u32);

impl ActiveCap {
    /// Sweep the full storage cap — the ceiling rung.
    pub const FULL: ActiveCap = ActiveCap(0);

    /// Sweep no resident prefix: a chunk starting at position 0 attends only its own rows.
    pub const NONE: ActiveCap = ActiveCap(u32::MAX);

    /// A rung sweeping `slots` KV positions.
    pub const fn new(slots: u32) -> ActiveCap {
        ActiveCap(slots)
    }

    /// The requested slot count (0 for [`FULL`](ActiveCap::FULL)).
    pub const fn get(self) -> u32 {
        self.0
    }

    /// The swept extent under storage cap `cap` with KV rows in units of `stick`:
    /// [`NONE`](ActiveCap::NONE) sweeps nothing, [`FULL`](ActiveCap::FULL) and any request that is
    /// not a whole number of sticks within `cap` sweep the whole cap.
    pub fn resolve(self, cap: u32, stick: u32) -> u32 {
        if self == ActiveCap::NONE {
            return 0;
        }
        if self.0 > 0 && self.0 <= cap && self.0.is_multiple_of(stick) {
            self.0
        } else {
            cap
        }
    }

    /// The interior decode rungs strictly below `cap`, ascending: a geometric ladder, each rung
    /// bounding the sweep to about twice the live length. The caller adds `cap` as the ceiling.
    pub fn decode_ladder(cap: u32) -> Vec<ActiveCap> {
        const RUNGS: [u32; 6] = [64, 128, 256, 512, 1024, 2048];
        RUNGS
            .into_iter()
            .map(|r| r.next_multiple_of(crate::rung::FP16_LANES))
            .filter(|&r| r < cap)
            .map(ActiveCap::new)
            .collect()
    }
}

/// One node's program: the function, its kind, and the buffer each parameter addresses —
/// `bindings[i]` is the buffer the function's `i`-th parameter points at.
#[derive(Clone, Debug)]
pub struct KtirNode {
    pub func: ktir_core::ir::IRFunction<'static>,
    pub program: ktir_core::program::Program,
    pub bindings: Vec<BufferId>,
}

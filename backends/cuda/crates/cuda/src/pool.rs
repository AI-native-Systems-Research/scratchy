// SPDX-License-Identifier: Apache-2.0
//! THE WORKER'S POOLS, AS THE HOST NAMES THEM — the KV and GDN pool sizes a worker is built with,
//! and the block and slot ids of them a forward takes. No toolchain: the step plan makes the ids,
//! each checked once against its pool, and the forward reads them as they are.

use crate::kernel_consts::NO_SLOT;
use crate::step::CUDA_KV_BLOCK_SIZE;

/// The KV pool a worker allocates when its tapes attend: how many blocks each layer's K and V
/// buffers hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPoolConfig {
    pub num_blocks: u32,
}

/// The GDN state pool a worker allocates when its tapes have Gated-DeltaNet layers: how many
/// sequences' state each layer holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnPoolConfig {
    pub num_slots: u32,
}

/// A block of the KV pool a worker is built with ([`KvPoolConfig`]): below its `num_blocks`, and
/// below the last block whose slots fit a `u32` short of [`NO_SLOT`] (a pool that large cannot be
/// allocated on an H100). Only [`Self::of`] makes one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockId(u32);

impl BlockId {
    /// `block` of `pool`, or `None` when it is outside it.
    pub const fn of(block: usize, pool: KvPoolConfig) -> Option<Self> {
        let num_blocks = pool.blocks();
        if block < num_blocks as usize {
            Some(Self(block as u32))
        } else {
            None
        }
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

impl KvPoolConfig {
    /// The blocks a [`BlockId`] of this pool is below: its `num_blocks`, at most the last block
    /// whose slots fit a `u32` short of [`NO_SLOT`].
    pub const fn blocks(self) -> u32 {
        let fit = NO_SLOT / CUDA_KV_BLOCK_SIZE;
        if self.num_blocks < fit {
            self.num_blocks
        } else {
            fit
        }
    }
}

/// A sequence's slot in the GDN state pool a worker is built with ([`GdnPoolConfig`]): below its
/// `num_slots`. Only [`Self::of`] makes one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotId(u32);

impl SlotId {
    /// `slot` of `pool`, or `None` when it is outside it.
    pub const fn of(slot: usize, pool: GdnPoolConfig) -> Option<Self> {
        if slot < pool.num_slots as usize {
            Some(Self(slot as u32))
        } else {
            None
        }
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

// SPDX-License-Identifier: Apache-2.0
//
// Multi-head latent attention's constants (`MlaConstants`), every MLA kernel's the same slots —
// `mla.metal`'s header lists them — shared by `mla.metal` and the matrix unit's tiled attention
// (`mla_nax.metal`).
//
// No include guard: it declares the bake's per-kernel constants, so each kernel of a bake batch
// re-reads it in its own namespace (`baked.h`).

#include <metal_stdlib>
#include "baked.h"

SCRATCHY_CONSTANT(uint, MLA_HEADS, 0);
SCRATCHY_CONSTANT(uint, MLA_LATENT, 1);
SCRATCHY_CONSTANT(uint, MLA_NOPE, 2);
SCRATCHY_CONSTANT(uint, MLA_ROPE, 3);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_BLOCK_SIZE, 4);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_MAX_BLOCKS, 5);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_BLOCKS_PER_CHUNK, 6);
SCRATCHY_CONSTANT_OPTIONAL(float, MLA_SCALE, 7);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_HEADS_PER_TG, 8);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_SIMDGROUPS, 9);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_PARTS, 10);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_PARTIAL_STRIDE, 11);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_MERGE_THREADS, 12);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_ROW_THREADS, 13);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_HALF_ROW, 14);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_PAIRING, 15);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_WARP_LANES, 16);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_KEYS, 17);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_BLOCKS, 18);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_FRAGMENT, 19);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_BLOCK_ROWS, 20);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_SLICE, 21);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_TG_BYTES, 22);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_BLOCK_MASK, 23);

constant constexpr uint MLA_Q_HEAD = MLA_NOPE + MLA_ROPE;
constant constexpr uint MLA_ROW = MLA_LATENT + MLA_ROPE;
constant constexpr uint MLA_HALF = MLA_HALF_ROW;

// The tiled attentions' (`mla_attend_tiles`, `mla_attend_tiles_nax`) shape: BLOCKS blocks of
// BLOCK_ROWS queries a threadgroup, each LATENT / WARP_LANES warps. Unset (another kernel's
// compile of its file): one block of one warp, never dispatched.
constant constexpr uint MLA_TILE_BLOCKS = MLA_BLOCKS_SET ? MLA_BLOCKS : 1u;
constant constexpr uint MLA_TILE_WN = MLA_WARP_LANES_SET ? MLA_LATENT / MLA_WARP_LANES : 1u;
constant constexpr uint MLA_TILE_THREADS = MLA_TILE_BLOCKS * 32u * MLA_TILE_WN;

// The rows of the page group from cached row `key` on in `store`, of a sequence whose block-table
// row is `blocks` over `pages` pages: its page's chunk and block, then its row of HALF lanes. A key
// past the last page reads the last page (a tiled attention never uses what it reads there: its
// scores mask, its values load as 0).
template <typename T>
inline device T* mla_rows_at(device const uint64_t* store, device const uint* blocks, int pages,
                             int key) {
    const int page = metal::min(key / int(MLA_BLOCK_SIZE), pages - 1);
    const uint block = blocks[page] & MLA_BLOCK_MASK;
    const uint chunk = MLA_BLOCKS_PER_CHUNK == 0u ? 0u : block / MLA_BLOCKS_PER_CHUNK;
    const uint blk = MLA_BLOCKS_PER_CHUNK == 0u ? block : block % MLA_BLOCKS_PER_CHUNK;
    return (device T*)store[chunk] + blk * (MLA_BLOCK_SIZE * MLA_HALF)
        + uint(key % int(MLA_BLOCK_SIZE)) * MLA_HALF;
}

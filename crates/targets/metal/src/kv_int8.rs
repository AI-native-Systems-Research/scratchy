// SPDX-License-Identifier: Apache-2.0
//! The int8 KV store's provisioning for the Metal tape ([`build_int8_provision`]): each coded
//! layer's codes and row scales (the canonical cache, `scratchy_layers::kv_int8`'s layout) and the
//! one-layer fp16 scratch prefill attention reads them staged into. The KV writers encode into
//! the store (`kv_int8.h`, `MetalFusion::KvEncoded`); decode attention reads it directly, prefill
//! through `kv_stage` (`attention.metal`).

use objc2_metal::MTLDevice;
pub use scratchy_layers::kv_int8::{Int8BlockBytes, kv_bytes_per_token, scratch_bytes_per_token};

use crate::argmax::Buffer;
use crate::interpreter::metal::runtime::{Int8KvBuffers, KvScratch};

/// The KV pool an int8 provision serves, at its GLOBAL (full-context) geometry.
#[derive(Clone, Copy)]
pub struct Int8Pool<'a> {
    pub is_global: &'a [bool],
    pub num_blocks: usize,
    pub block_size: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub blocks_per_chunk: usize,
}

/// Build the int8 provisioning for one worker at the GLOBAL (full-context) geometry.
/// `num_blocks` is the shared KV pool capacity (the int8 store is the canonical cache of that
/// size); the scratch is ONE layer's fp16 (`num_blocks` blocks) reused across the global layers.
///
/// `is_global[L]` flags whether layer L is in the GLOBAL group (group 0). Only global layers get a
/// real store; sliding layers get a tiny placeholder buffer that the tape never binds (the KV codec
/// pass codes only the global KV writers). Uniform arches pass all-true.
///
/// `hosted` is the scratch of the pool whose arena this one runs in (a head's target's,
/// [`crate::interpreter::metal::HostArena::kv_scratch`]): of this geometry, the provision stages
/// through it and allocates none of its own.
pub fn build_int8_provision(
    allocator: &crate::MetalAllocator,
    pool: Int8Pool<'_>,
    hosted: Option<&KvScratch>,
) -> Int8KvBuffers {
    let Int8Pool {
        is_global,
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        blocks_per_chunk,
    } = pool;
    use objc2_metal::{MTLBuffer, MTLResourceOptions};
    let device = allocator.device();
    let num_layers = is_global.len();
    let n_global = is_global.iter().filter(|&&g| g).count();
    tracing::info!(
        "int8 KV: head_dim={head_dim}, num_kv_heads={num_kv_heads}, block_size={block_size}, \
         {num_blocks} blocks, {n_global}/{num_layers} global layers coded"
    );

    // Per-layer codes + scales (canonical cache), at the GLOBAL geometry, then the per-worker fp16
    // scratch: one layer (num_blocks blocks), reused across all global layers. All are
    // placement-sparse, mapped as group 0's blocks are reached (`SparseRegistry`). The SCALES read
    // zero until written: decode reads every key of the active context, and a block the step's
    // scheduler handed out but no writer reached yet decodes to 0. So does the scratch: the paged
    // attention can stage blocks just past the active context. Codes need no zeroing (a garbage
    // code times scale 0 is 0).
    let bytes = Int8BlockBytes::of(block_size, num_kv_heads, head_dim);
    let (codes, scales, scratch) = (
        (bytes.codes, false),
        (bytes.scales, true),
        (bytes.scratch, true),
    );
    // A hosted scratch of this geometry is staged through, and none of this pool's own is laid
    // out: its stores are its host's, mapped as far as the same blocks are reached.
    let chunk_bytes = blocks_per_chunk * scratch.0;
    let scratch_bytes = num_blocks * scratch.0;
    let hosted = hosted.filter(|s| s.bytes == scratch_bytes && s.chunk_bytes == chunk_bytes);
    let own_scratch = if hosted.is_some() { 0 } else { 2 };
    let layout: Vec<(usize, bool)> = [codes, codes, scales, scales]
        .into_iter()
        .flat_map(|store| std::iter::repeat_n(store, n_global))
        .chain(std::iter::repeat_n(scratch, own_scratch))
        .collect();
    let stores = crate::sparse_kv::SparseStores::new(
        device,
        allocator.residency(),
        num_blocks,
        blocks_per_chunk,
        &layout,
    );
    let stores = stores.expect("int8 KV stores");
    // The KvCodesK{layer} binding is layer-indexed, so a global layer L's store must sit at index
    // L; sliding layers (is_global[L]==false) get a 16-byte placeholder that the tape never binds.
    let mut next = (0..layout.len()).map(|i| stores.buffer(i));
    let mut by_layer = || -> Vec<Buffer> {
        let shared = MTLResourceOptions::StorageModeShared;
        let mut one = |&g: &bool| match g {
            true => next.next().expect("a store per global layer"),
            false => device
                .newBufferWithLength_options(16, shared)
                .expect("int8 KV placeholder"),
        };
        is_global.iter().map(&mut one).collect()
    };
    let (codes_k, codes_v, scales_k, scales_v) = (by_layer(), by_layer(), by_layer(), by_layer());
    // Contiguous scratch data + a chunk-table of gpuAddresses at chunk offsets (the kernels
    // deref it).
    let n_chunks = num_blocks.div_ceil(blocks_per_chunk.max(1));
    let table = |data: &Buffer| {
        let base = data.gpuAddress();
        let vals: Vec<u64> = (0..n_chunks)
            .map(|c| base + (c * chunk_bytes) as u64)
            .collect();
        crate::argmax::upload_shared_buffer(device, &vals)
    };
    let scratch = match hosted {
        Some(s) => s.clone(),
        None => {
            let k_data = next.next().expect("K scratch");
            let v_data = next.next().expect("V scratch");
            KvScratch {
                k_table: table(&k_data),
                v_table: table(&v_data),
                k_data,
                v_data,
                bytes: scratch_bytes,
                chunk_bytes,
            }
        }
    };
    let mapped = stores.mapped();
    let registry = allocator.sparse_kv.lock();
    let registered = registry.expect("sparse KV registry").register(stores);
    registered.expect("int8 KV stores mapped as far as the pool reaches");

    Int8KvBuffers {
        codes_k,
        codes_v,
        scales_k,
        scales_v,
        scratch,
        mapped,
    }
}

/// The floats one lane of the int8 decode holds of its Q and O fragments, at most. A hand-picked
/// register budget, measured on a base M5. In the kernel alone (an 8192-key context, 16 query
/// heads over 2, µs a call) 128 floats spill — head_dim 256 at a key group of one ran 1498
/// against two's 293, 512 at two 2890 against four's 404 — and 64 run: 128 at one ran 93 against
/// two's 146. Serving (8192-token prompts, batch 1, TPOT, two interleaved runs) sets it below
/// that: Gemma 4 26B-A4B (head_dim 512) ran 3-4% slower at four (64 floats) than at eight in both
/// runs, gpt-oss-20b (64) 1.5% faster at one (32 floats) than at two, and Nemotron-3-Nano (128)
/// and Qwen3.6-35B-A3B (256) the same at one or two and two or four.
pub const DECODE_LANE_FRAGMENT_FLOATS: u32 = 32;

/// The simdgroups of the int8 decode (`attention_decode_gqa_int8`) that share a key block, each a
/// slice of the head dims (`ATTN_DECODE_SG`): the fewest whose slice's Q and O fragments
/// (`head_dim / (8 · key group)` 8x8 fragments each, two floats a lane a fragment) fit
/// [`DECODE_LANE_FRAGMENT_FLOATS`]. The rest of a threadgroup's `head_dim / 32` simdgroups take
/// other key blocks.
pub const fn decode_key_group(head_dim: u32) -> u32 {
    let mut group = 1;
    while 4 * head_dim / (8 * group) > DECODE_LANE_FRAGMENT_FLOATS {
        group *= 2;
    }
    group
}

/// The most simdgroups the decode's combine (`attention_via_cache_v2_combine`) sums a query
/// head's partials over. A hand-picked constant.
pub const COMBINE_MAX_SIMDGROUPS: u32 = 4;

/// The simdgroups of the decode's combine (`ATTN_COMBINE_GROUPS`): one a lane slice of the head
/// dims, at most [`COMBINE_MAX_SIMDGROUPS`].
pub const fn combine_groups(head_dim: u32) -> u32 {
    let slices = head_dim / 32;
    if slices < COMBINE_MAX_SIMDGROUPS {
        slices
    } else {
        COMBINE_MAX_SIMDGROUPS
    }
}

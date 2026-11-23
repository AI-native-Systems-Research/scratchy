// SPDX-License-Identifier: Apache-2.0
//! The int8 KV store's provisioning for the Metal tape ([`build_int8_provision`]): each coded
//! layer's codes and row scales (the canonical cache, `scratchy_layers::kv_int8`'s layout) and the
//! one-layer fp16 scratch prefill attention reads them staged into. The KV writers encode into
//! the store (`kv_int8.h`, `MetalFusion::KvEncoded`); decode attention reads it directly, prefill
//! through `kv_stage` (`attention.metal`).

use objc2_metal::MTLDevice;
use scratchy_layers::kv_int8::{CodedHeads, KvCodec};

use crate::kv_int8_heads::{MAX_CODED_HEAD_DIM, MIN_CODED_HEAD_DIM};

/// The head dims Metal's int8 KV kernels code (`kv_int8_heads`).
pub const METAL_CODED_HEADS: CodedHeads = CodedHeads::new(MIN_CODED_HEAD_DIM, MAX_CODED_HEAD_DIM);

use scratchy_ir::model_facts::HeadGeometry;

use crate::argmax::Buffer;
use crate::interpreter::metal::runtime::{Int8KvBuffers, KvScratch};

/// The bytes a row's scale takes: one `f32`.
const SCALE_BYTES: usize = 4;

/// The width of the fp16 scratch prefill attention stages int8 K/V into.
const SCRATCH_ELEM_BYTES: usize = 2;

/// Bytes one token's K and V cost in a KV pool of `num_layers` layers of `num_kv_heads ×
/// head_dim` stored as `codec`: dense, every layer's `dense_elem_bytes`-wide row; int8, every
/// layer's codes and scales plus the one-layer fp16 scratch — what
/// [`build_int8_provision`] allocates per token.
pub fn kv_bytes_per_token(
    codec: KvCodec,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dense_elem_bytes: usize,
) -> usize {
    match codec {
        KvCodec::Dense => num_layers * 2 * num_kv_heads * head_dim * dense_elem_bytes,
        KvCodec::Int8 => {
            num_layers * 2 * num_kv_heads * (head_dim + SCALE_BYTES)
                + scratch_bytes_per_token(num_kv_heads, head_dim)
        }
    }
}

/// What one block of `block_size` tokens takes in each store an int8 pool lays out: a layer's
/// codes and its scales (one K or V each), and the one-layer fp16 scratch — as
/// [`build_int8_provision`] lays them out, one source for it
/// and for whoever must know each store's bytes (the runs a shared KV budget maps apart).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Int8BlockBytes {
    codes: usize,
    scales: usize,
    scratch: usize,
}

impl Int8BlockBytes {
    pub fn of(block_size: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            codes: block_size * num_kv_heads * head_dim,
            scales: block_size * num_kv_heads * SCALE_BYTES,
            scratch: block_size * num_kv_heads * head_dim * SCRATCH_ELEM_BYTES,
        }
    }

    /// A layer's codes of K, or of V.
    pub fn codes(self) -> usize {
        self.codes
    }

    /// A layer's scales of K, or of V.
    pub fn scales(self) -> usize {
        self.scales
    }

    /// The one-layer fp16 scratch's K, or V.
    pub fn scratch(self) -> usize {
        self.scratch
    }
}

/// Bytes one token costs in an int8 pool's one-layer fp16 scratch, K and V: what a model staging
/// through another pool's scratch does not pay.
pub fn scratch_bytes_per_token(num_kv_heads: usize, head_dim: usize) -> usize {
    2 * num_kv_heads * head_dim * SCRATCH_ELEM_BYTES
}

/// What one block of a layer of `class` takes in each store.
fn block_bytes(class: &HeadGeometry) -> Int8BlockBytes {
    Int8BlockBytes::of(
        class.page().get() as usize,
        class.kv_heads().get() as usize,
        class.head_dim().get() as usize,
    )
}

/// The KV pool an int8 provision serves: each layer's attention class (`None` for a layer that
/// holds no KV), the pool's blocks and their chunking.
#[derive(Clone, Copy)]
pub struct Int8Pool<'a> {
    pub layers: &'a [Option<HeadGeometry>],
    pub num_blocks: usize,
    pub blocks_per_chunk: usize,
}

/// Build the int8 provisioning for one worker: each KV layer's codes and scales at its class's
/// geometry, over the pool's `num_blocks` blocks (each class numbers its own), and the scratch
/// prefill attention reads them staged into — one layer's fp16 rows of the widest class's block,
/// reused by every layer. A layer that holds no KV gets a 16-byte placeholder the tape never binds.
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
        layers,
        num_blocks,
        blocks_per_chunk,
    } = pool;
    use objc2_metal::{MTLBuffer, MTLResourceOptions};
    let device = allocator.device();
    let coded: Vec<HeadGeometry> = layers.iter().flatten().copied().collect();
    tracing::info!(
        "int8 KV: {num_blocks} blocks, {}/{} layers coded",
        coded.len(),
        layers.len()
    );

    // Per-layer codes + scales (the canonical cache), then the per-worker fp16 scratch. All are
    // placement-sparse, mapped as the pool's blocks are reached (`SparseRegistry`). The SCALES
    // read zero until written: decode reads every key of the active context, and a block the
    // step's scheduler handed out but no writer reached yet decodes to 0. So does the scratch: the
    // paged attention can stage blocks just past the active context. Codes need no zeroing (a
    // garbage code times scale 0 is 0).
    let scratch_block = coded
        .iter()
        .map(|l| block_bytes(l).scratch())
        .max()
        .unwrap_or(0);
    // A hosted scratch of this geometry is staged through, and none of this pool's own is laid
    // out: its stores are its host's, mapped as far as the same blocks are reached.
    let chunk_bytes = blocks_per_chunk * scratch_block;
    let scratch_bytes = num_blocks * scratch_block;
    let hosted = hosted.filter(|s| s.bytes == scratch_bytes && s.chunk_bytes == chunk_bytes);
    let own_scratch = if hosted.is_some() { 0 } else { 2 };
    let codes = |l: &HeadGeometry| (block_bytes(l).codes(), false);
    let scales = |l: &HeadGeometry| (block_bytes(l).scales(), true);
    let layout: Vec<(usize, bool)> = (coded.iter().map(codes))
        .chain(coded.iter().map(codes))
        .chain(coded.iter().map(scales))
        .chain(coded.iter().map(scales))
        .chain(std::iter::repeat_n((scratch_block, true), own_scratch))
        .collect();
    let stores = crate::sparse_kv::SparseStores::new(
        device,
        allocator.residency(),
        num_blocks,
        blocks_per_chunk,
        &layout,
    );
    let stores = stores.expect("int8 KV stores");
    // The KvCodesK{layer} binding is layer-indexed, so layer L's store sits at index L; a layer
    // that holds no KV gets a 16-byte placeholder that the tape never binds. The layout holds
    // `coded` stores of each kind in turn, a coded layer's at its rank among them.
    let ranks: Vec<Option<usize>> = (layers.iter())
        .scan(0, |rank, l| {
            Some(l.map(|_| {
                *rank += 1;
                *rank - 1
            }))
        })
        .collect();
    let by_layer = |kind: usize| -> Vec<Buffer> {
        let shared = MTLResourceOptions::StorageModeShared;
        let one = |rank: &Option<usize>| match rank {
            Some(r) => stores.buffer(kind * coded.len() + r),
            None => device
                .newBufferWithLength_options(16, shared)
                .expect("int8 KV placeholder"),
        };
        ranks.iter().map(one).collect()
    };
    let (codes_k, codes_v, scales_k, scales_v) =
        (by_layer(0), by_layer(1), by_layer(2), by_layer(3));
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
            let k_data = stores.buffer(4 * coded.len());
            let v_data = stores.buffer(4 * coded.len() + 1);
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

/// The keys of the rung's KV cap one threadgroup of the int8 decode takes: a hand-picked constant.
pub const DECODE_SPLIT_KEYS: u32 = 256;

/// At most this many threadgroups a KV head's keys spread over: a hand-picked constant.
pub const DECODE_MAX_SPLITS: u32 = 16;

/// The rows of the matrix units' fragments the int8 decode serves its query heads in.
pub const DECODE_FRAGMENT_ROWS: u32 = 8;

/// How the int8 decode (`attention_decode_gqa_int8`) and its combine
/// (`attention_via_cache_v2_combine`) cut one class's work, every part a law of the class's
/// geometry and the keys a query reads at most: [`Self::of`] is the one way to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Int8DecodeShape {
    heads: u32,
    splits: u32,
    key_group: u32,
    combine_groups: u32,
}

impl Int8DecodeShape {
    /// The shape of a class of `head_dim` and GQA group `group` whose queries read at most `keys`
    /// keys: a threadgroup serves the largest divisor of the group that fits
    /// [`DECODE_FRAGMENT_ROWS`]; the keys spread over one threadgroup per [`DECODE_SPLIT_KEYS`],
    /// at most [`DECODE_MAX_SPLITS`]; the key group is [`decode_key_group`]'s and the combine's
    /// simdgroups [`combine_groups`]'.
    pub const fn of(head_dim: u32, group: u32, keys: u32) -> Self {
        let mut heads = 1;
        let mut r = 2;
        while r <= group && r <= DECODE_FRAGMENT_ROWS {
            if group.is_multiple_of(r) {
                heads = r;
            }
            r += 1;
        }
        let splits = keys.div_ceil(DECODE_SPLIT_KEYS);
        let splits = if splits < 1 {
            1
        } else if splits > DECODE_MAX_SPLITS {
            DECODE_MAX_SPLITS
        } else {
            splits
        };
        Self {
            heads,
            splits,
            key_group: decode_key_group(head_dim),
            combine_groups: combine_groups(head_dim),
        }
    }

    /// The query heads one threadgroup serves (`ATTN_DECODE_HEADS`).
    pub const fn heads(self) -> u32 {
        self.heads
    }

    /// The threadgroups a KV head's keys spread over (`ATTN_SPLITS`).
    pub const fn splits(self) -> u32 {
        self.splits
    }

    /// The simdgroups sharing a key block (`ATTN_DECODE_SG`).
    pub const fn key_group(self) -> u32 {
        self.key_group
    }

    /// The combine's simdgroups (`ATTN_COMBINE_GROUPS`).
    pub const fn combine_groups(self) -> u32 {
        self.combine_groups
    }
}

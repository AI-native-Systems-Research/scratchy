// SPDX-License-Identifier: Apache-2.0
//! THE TAPE PLAYER — one arena, one stream, and a forward that launches a static tape's steps.
//!
//! The counterpart of metal's `MetalWorker`: the arena is allocated once, at construction, sized
//! to the largest of the worker's tapes (metal's `arena_layout`, one arena for every bucket, since
//! one bucket runs per forward). A forward enqueues the input copy, every step's launch in tape
//! order and the result copy on the worker's stream, then waits once: one queue of work and one
//! host wait per forward, as metal's one command buffer, one commit and one wait.
//!
//! No barrier: on one in-order stream each kernel starts after the previous one has finished, and
//! sees all of its global writes (TMA stores included). A second stream, or programmatic
//! dependent launch, would need fences, and is a later issue.
//!
//! A worker whose tapes attend also owns the paged KV pool: one device allocation, sized at
//! construction from the tapes' KV geometry and the pool's block count, laid out per layer as
//! `PagedKvLayout` (`scratchy-subtile`) says. A forward's sequence data (the block table, the
//! prefix and query lengths) is checked on the host and uploaded into the tape's regions, with
//! the slot of each row derived from it, before any step runs.
//!
//! A worker whose tapes have Gated-DeltaNet layers also owns the GDN state pool: one more device
//! allocation, sized at construction from the tapes' GDN geometry and the pool's slot count, laid
//! out per layer as `GdnStateLayout` (`scratchy-subtile`) says, f32. It is never cleared. A
//! forward's GDN sequence (its slot, its real rows, whether it is the sequence's first) is
//! checked on the host and uploaded into the tape's GDN sequence region before any step runs; a
//! first forward's kernels start from zero without reading the slot, then overwrite it in full.

use std::ffi::c_void;
use std::fmt;
use std::marker::PhantomData;

use super::buffer::DeviceBuffer;
use super::device::CudaDevice;
use super::ffi;
use super::{CudaError, check};
use scratchy_subtile::gdn_state_layout::GdnStateLayout;
use scratchy_subtile::paged_kv_layout::PagedKvLayout;

use crate::kernels::{self, KernelError};
use crate::rotary::fill_rotary_rows;
use crate::step::{
    AddBf16, AttnPrefillBf16, CudaStep, EmbedBf16, ExpertCombineBf16, ExpertGemmBf16,
    ExpertGemvBf16, ExpertProjection, ExpertSortBf16, GateApplyBf16, GateScaleBf16, GateSplitBf16,
    GatedDeltaNetBf16, GemmBf16, KV_APPEND_BF16_HEAD_DIM, LayerScaleBf16, MOE_ROUTE_EXPERTS,
    MatmulKernel, MoeRouteBf16, MoeWeight, PairRows, RmsNormBf16, RmsNormUnitBf16, RopeAppendBf16,
    RopeBf16, RouteProgram, RouterLogitsBf16, ScaleBf16, SiluMulBf16, SoftCapBf16,
};
use crate::tape::{
    ArenaOffset, ArenaSpan, CUDA_ACT_BYTES, CUDA_ARENA_ALIGN, CudaTape, GdnTape, KvTape,
    RotaryTable, TapeError, TapeInput, WeightIx,
};

/// A forward's input, of the kind its tape takes ([`TapeInput`]).
#[derive(Clone, Copy, Debug)]
pub enum Input<'a> {
    /// The embedded hidden, `[bucket_m, hidden]` bf16 bits.
    Hidden(&'a [u16]),
    /// One token id per bucket row, padding rows included (the caller pads).
    TokenIds(&'a [u32]),
}

impl Input<'_> {
    /// The input's name, for [`ForwardError::InputKind`].
    fn kind(&self) -> &'static str {
        match self {
            Self::Hidden(_) => "hidden rows",
            Self::TokenIds(_) => "token ids",
        }
    }

    /// The input's bytes, as the forward uploads them.
    fn bytes(&self) -> u64 {
        match self {
            Self::Hidden(h) => std::mem::size_of_val(*h) as u64,
            Self::TokenIds(t) => std::mem::size_of_val(*t) as u64,
        }
    }

    /// The host pointer the input copy reads.
    fn ptr(&self) -> *const c_void {
        match self {
            Self::Hidden(h) => h.as_ptr().cast(),
            Self::TokenIds(t) => t.as_ptr().cast(),
        }
    }
}

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

/// One forward's GDN sequence: its state's `slot` in the pool, its `q_len` real rows, and whether
/// this is the sequence's first forward (`fresh`), whose kernels start from zero without reading
/// the slot.
///
/// The caller passes `fresh: true` on a sequence's first forward (whenever the slot is handed to a
/// new sequence); the worker cannot tell sequences apart, so `fresh: false` for a new sequence
/// continues from whatever the slot holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnSequence {
    pub slot: u32,
    pub q_len: u32,
    pub fresh: bool,
}

/// One forward's sequence: the block table of its cached and new tokens, `prefix_len` tokens
/// already in the cache, and `q_len` new ones at positions `prefix_len .. prefix_len + q_len`.
#[derive(Clone, Copy, Debug)]
pub struct KvSequence<'s> {
    pub block_table: &'s [u32],
    pub prefix_len: u32,
    pub q_len: u32,
}

/// Which of a layer's two cache buffers [`CudaWorker::read_kv`] reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvHalf {
    K,
    V,
}

/// A slot of the slot mapping or the block table that is not one: a padding row, or an unused
/// block-table entry.
const NO_SLOT: u32 = u32::MAX;

/// Why a worker could not be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkerError {
    /// The worker was given no tape.
    NoTapes,
    /// The tape at `bucket` would read or write outside its arena or its weights.
    Tape { bucket: usize, error: TapeError },
    /// A tape needs a weight at `ix` and the table has no buffer there.
    MissingWeight { ix: usize, need: u64 },
    /// The weight buffer at `ix` holds `have` bytes; a tape needs `need`.
    WeightTooSmall { ix: usize, need: u64, have: u64 },
    /// A tape has a KV cache and the worker was given no pool, or a pool of no blocks.
    KvPoolMissing,
    /// No tape has a KV cache and the worker was given a pool.
    KvPoolUnused,
    /// The tape at `bucket`'s KV layers, kv heads, head dim or block size are not the first KV
    /// tape's: one pool cannot hold both.
    KvGeometry { bucket: usize },
    /// A tape has GDN layers and the worker was given no GDN pool, or a pool of no slots.
    GdnPoolMissing,
    /// No tape has GDN layers and the worker was given a GDN pool.
    GdnPoolUnused,
    /// The tape at `bucket`'s GDN layers, heads, head dims or conv kernel are not the first GDN
    /// tape's: one pool cannot hold both.
    GdnGeometry { bucket: usize },
    /// The KV pool's bytes (`layers · 2 · num_blocks` blocks of `kv_heads · block_size ·
    /// head_dim` bf16) overflow `u64`: `num_blocks` is too large for the tapes' geometry.
    KvPoolTooLarge { num_blocks: u32 },
    /// The GDN pool's bytes (per GDN layer, `num_slots` slots of conv and ssm state) overflow
    /// `u64`: `num_slots` is too large for the tapes' geometry.
    GdnPoolTooLarge { num_slots: u32 },
    /// Creating the stream or allocating the arena or a pool failed.
    Cuda(CudaError),
}

impl fmt::Display for WorkerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoTapes => write!(f, "a CUDA worker needs at least one tape"),
            Self::Tape { bucket, error } => write!(f, "the tape of bucket {bucket}: {error}"),
            Self::MissingWeight { ix, need } => {
                write!(
                    f,
                    "weight {ix} needs {need} bytes and the worker was given none"
                )
            }
            Self::WeightTooSmall { ix, need, have } => {
                write!(f, "weight {ix} holds {have} bytes; the tape reads {need}")
            }
            Self::KvPoolMissing => write!(
                f,
                "a tape attends over the KV cache and the worker was given no pool of at least \
                 one block"
            ),
            Self::KvPoolUnused => write!(
                f,
                "the worker was given a KV pool and no tape has a KV cache"
            ),
            Self::KvGeometry { bucket } => write!(
                f,
                "the tape of bucket {bucket} has another KV geometry than the first KV tape's"
            ),
            Self::GdnPoolMissing => write!(
                f,
                "a tape has GDN layers and the worker was given no GDN pool of at least one slot"
            ),
            Self::GdnPoolUnused => write!(
                f,
                "the worker was given a GDN pool and no tape has GDN layers"
            ),
            Self::GdnGeometry { bucket } => write!(
                f,
                "the tape of bucket {bucket} has another GDN geometry than the first GDN tape's"
            ),
            Self::KvPoolTooLarge { num_blocks } => write!(
                f,
                "a KV pool of {num_blocks} blocks overflows u64 bytes for the tapes' KV geometry"
            ),
            Self::GdnPoolTooLarge { num_slots } => write!(
                f,
                "a GDN pool of {num_slots} slots overflows u64 bytes for the tapes' GDN geometry"
            ),
            Self::Cuda(e) => write!(f, "cannot build the CUDA worker: {e}"),
        }
    }
}

impl std::error::Error for WorkerError {}

/// Why a forward, or an arena read, failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ForwardError {
    /// No tape at this bucket index; nothing was enqueued.
    Bucket { bucket: usize, buckets: usize },
    /// The input is not the bucket's input span, in bytes; nothing was enqueued.
    InputSize { expected: u64, got: u64 },
    /// The input is `got` and the bucket's tape takes `expected` (hidden rows or token ids);
    /// nothing was enqueued.
    InputKind {
        expected: &'static str,
        got: &'static str,
    },
    /// Row `row`'s token id is `id`, outside the tape's `vocab`-row embedding table; nothing was
    /// enqueued.
    TokenOutOfVocab { row: usize, id: u32, vocab: u32 },
    /// The result buffer is not the bucket's result span, in bytes; nothing was enqueued.
    ResultSize { expected: u64, got: u64 },
    /// The forward was given `got` positions; the bucket's tape takes `expected` (its `bucket_m`
    /// when it has a rotary table, else 0). Nothing was enqueued.
    Positions { expected: usize, got: usize },
    /// The host buffer of an arena read is not the span's size, in bytes; nothing was enqueued.
    ReadSize { expected: u64, got: u64 },
    /// The tape has a KV cache and the forward was given no sequence; nothing was enqueued.
    KvMissing,
    /// The tape has no KV cache and the forward was given a sequence; nothing was enqueued.
    KvUnexpected,
    /// `q_len` is not in `1 ..= bucket_m`; nothing was enqueued.
    KvQueryLength { q_len: u32, bucket_m: u32 },
    /// `prefix_len + q_len` is above the `max` tokens a block table holds; nothing was enqueued.
    KvTooLong { tokens: u64, max: u64 },
    /// The block table has `got` entries; the sequence needs at least `need`, and the tape holds
    /// at most `max`. Nothing was enqueued.
    KvBlockTable { got: usize, need: usize, max: usize },
    /// Block-table entry `entry` is block `block`, outside the pool's `num_blocks`; nothing was
    /// enqueued.
    KvBlockOutsidePool {
        entry: usize,
        block: u32,
        num_blocks: u32,
    },
    /// Block `block` is used by entries `first` and `second` of the block table; nothing was
    /// enqueued.
    KvBlockRepeated {
        block: u32,
        first: usize,
        second: usize,
    },
    /// Row `row`'s position is `got`; the sequence puts it at `prefix_len + row`, `want`. Nothing
    /// was enqueued.
    PositionsDisagree { row: usize, got: u32, want: u64 },
    /// A KV read names a layer index or a block outside the pool, or its host buffer is not one
    /// block's size; nothing was enqueued.
    KvRead {
        layer: usize,
        block: u32,
        expected: usize,
        got: usize,
    },
    /// The tape has GDN layers and the forward was given no GDN sequence; nothing was enqueued.
    GdnMissing,
    /// The tape has no GDN layers and the forward was given a GDN sequence; nothing was enqueued.
    GdnUnexpected,
    /// The GDN sequence's slot is not one of the pool's `num_slots`; nothing was enqueued.
    GdnSlotOutsidePool { slot: u32, num_slots: u32 },
    /// The GDN sequence's `q_len` is not in `1 ..= bucket_m`; nothing was enqueued.
    GdnQueryLength { q_len: u32, bucket_m: u32 },
    /// The forward has both a KV sequence and a GDN sequence, and they disagree on whether it is
    /// the sequence's first forward: the KV sequence has `prefix_len` cached tokens and the GDN
    /// sequence says `fresh`. A fresh sequence has no prefix and a continued one has one, so one
    /// of the two is wrong, and the KV and GDN layers would see different histories. Nothing was
    /// enqueued.
    FreshDisagrees { prefix_len: u32, fresh: bool },
    /// A GDN state read names a layer index or a slot outside the pool, or a host buffer that is
    /// not one slot's size; nothing was enqueued.
    GdnRead {
        layer: usize,
        slot: u32,
        conv: (usize, usize),
        ssm: (usize, usize),
    },
    /// An arena read's span reaches past the arena; nothing was enqueued.
    SpanOutsideArena { span: ArenaSpan, arena_bytes: u64 },
    /// The `step`-th launch failed; nothing after it was enqueued, and the stream was drained.
    Launch { step: usize, error: KernelError },
    /// The `step`-th step is `op`, a step the lowering emits for Gemma 4 26B-A4B and no kernel
    /// launches yet (#885's 13b-1 to 13b-4 remove this piece by piece; 13b-4 deletes it). Checked over
    /// the whole tape before anything is uploaded or enqueued.
    NotLaunched { step: usize, op: &'static str },
    /// A copy or the stream wait failed; the stream was drained.
    Cuda(CudaError),
}

impl fmt::Display for ForwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bucket { bucket, buckets } => {
                write!(
                    f,
                    "bucket {bucket} does not exist; the worker has {buckets}"
                )
            }
            Self::InputSize { expected, got } => {
                write!(f, "the input is {got} bytes; the bucket takes {expected}")
            }
            Self::InputKind { expected, got } => {
                write!(f, "the input is {got}; the bucket takes {expected}")
            }
            Self::TokenOutOfVocab { row, id, vocab } => write!(
                f,
                "row {row}'s token id is {id}; the embedding table has {vocab} rows"
            ),
            Self::ResultSize { expected, got } => {
                write!(
                    f,
                    "the result buffer is {got} bytes; the bucket writes {expected}"
                )
            }
            Self::Positions { expected, got } => {
                write!(
                    f,
                    "the forward was given {got} positions; the bucket takes {expected}"
                )
            }
            Self::ReadSize { expected, got } => {
                write!(f, "the read buffer is {got} bytes; the span is {expected}")
            }
            Self::KvMissing => write!(
                f,
                "the bucket attends over the KV cache and the forward was given no sequence"
            ),
            Self::KvUnexpected => write!(
                f,
                "the bucket has no KV cache and the forward was given a sequence"
            ),
            Self::KvQueryLength { q_len, bucket_m } => write!(
                f,
                "the sequence adds {q_len} tokens; the bucket takes 1 to {bucket_m}"
            ),
            Self::KvTooLong { tokens, max } => write!(
                f,
                "the sequence is {tokens} tokens; a block table holds at most {max}"
            ),
            Self::KvBlockTable { got, need, max } => write!(
                f,
                "the block table has {got} entries; the sequence needs {need}, and the tape \
                 holds at most {max}"
            ),
            Self::KvBlockOutsidePool {
                entry,
                block,
                num_blocks,
            } => write!(
                f,
                "block-table entry {entry} is block {block}, outside the pool's {num_blocks} \
                 blocks"
            ),
            Self::KvBlockRepeated {
                block,
                first,
                second,
            } => write!(
                f,
                "block {block} is block-table entries {first} and {second}; a block holds one \
                 range of tokens"
            ),
            Self::PositionsDisagree { row, got, want } => write!(
                f,
                "row {row} is at position {got}; the sequence puts it at {want}"
            ),
            Self::KvRead {
                layer,
                block,
                expected,
                got,
            } => write!(
                f,
                "cannot read layer {layer}'s block {block} into {got} elements: the pool has no \
                 such layer or block, or a block is {expected} elements"
            ),
            Self::GdnMissing => write!(
                f,
                "the bucket has GDN layers and the forward was given no GDN sequence"
            ),
            Self::GdnUnexpected => write!(
                f,
                "the bucket has no GDN layers and the forward was given a GDN sequence"
            ),
            Self::GdnSlotOutsidePool { slot, num_slots } => write!(
                f,
                "the GDN sequence's slot is {slot}, outside the pool's {num_slots} slots"
            ),
            Self::GdnQueryLength { q_len, bucket_m } => write!(
                f,
                "the GDN sequence has {q_len} rows; the bucket takes 1 to {bucket_m}"
            ),
            Self::FreshDisagrees { prefix_len, fresh } => write!(
                f,
                "the KV sequence has {prefix_len} cached tokens and the GDN sequence says \
                 fresh={fresh}: a fresh sequence has no prefix, a continued one has one"
            ),
            Self::GdnRead {
                layer,
                slot,
                conv,
                ssm,
            } => write!(
                f,
                "cannot read GDN layer {layer}'s slot {slot} into {} conv and {} ssm elements: \
                 the pool has no such layer or slot, or a slot is {} and {}",
                conv.1, ssm.1, conv.0, ssm.0
            ),
            Self::SpanOutsideArena { span, arena_bytes } => write!(
                f,
                "the span of {} bytes at {} reaches past the {arena_bytes}-byte arena",
                span.bytes, span.offset.0
            ),
            Self::Launch { step, error } => write!(f, "step {step} did not launch: {error}"),
            Self::NotLaunched { step, op } => {
                write!(f, "step {step} is {op}, which no kernel launches yet")
            }
            Self::Cuda(e) => write!(f, "the forward failed: {e}"),
        }
    }
}

impl std::error::Error for ForwardError {}

/// A non-blocking stream, destroyed on drop.
struct Stream(ffi::CudaStreamT);

impl Stream {
    fn new() -> Result<Self, CudaError> {
        let mut stream = std::ptr::null_mut();
        // SAFETY: `stream` is a valid out-pointer.
        check(
            unsafe { ffi::cudaStreamCreateWithFlags(&mut stream, ffi::STREAM_NON_BLOCKING) },
            "cudaStreamCreateWithFlags",
        )?;
        Ok(Self(stream))
    }

    /// Wait for everything enqueued on the stream.
    fn synchronize(&self) -> Result<(), CudaError> {
        // SAFETY: the stream is live until drop.
        check(
            unsafe { ffi::cudaStreamSynchronize(self.0) },
            "cudaStreamSynchronize",
        )
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: the stream came from cudaStreamCreateWithFlags and is destroyed once; every
        // forward drained it before returning. A failure cannot be reported from a destructor.
        let _ = unsafe { ffi::cudaStreamDestroy(self.0) };
    }
}

/// The worker's paged KV pool: one allocation of every layer's K and V buffer, and the host side
/// of the forward's KV regions.
struct KvPool {
    buffer: DeviceBuffer,
    /// One layer's K (or V) buffer's layout; every layer's is the same.
    layout: PagedKvLayout,
    /// The KV layers, as the tapes' [`KvTape::layers`].
    layers: usize,
    /// The host side of the block table, sized once: `max_blocks_per_seq` entries.
    block_table_host: Vec<u32>,
    /// The host side of the slot mapping, sized once for the largest bucket.
    slot_mapping_host: Vec<u32>,
}

impl KvPool {
    /// Element offset of layer index `layer`'s `half` buffer in the pool.
    fn half_elems(&self, layer: usize, half: KvHalf) -> usize {
        let second = match half {
            KvHalf::K => 0,
            KvHalf::V => 1,
        };
        (2 * layer + second) * self.layout.buffer_elems()
    }

    /// The device address of layer index `layer`'s `half` buffer. Computed, not dereferenced.
    fn half_ptr(&self, layer: usize, half: KvHalf) -> *mut c_void {
        let bytes = self.half_elems(layer, half) * std::mem::size_of::<u16>();
        self.buffer.ptr().wrapping_byte_add(bytes)
    }
}

/// The worker's GDN state pool: one allocation of every GDN layer's conv and ssm buffers (f32),
/// and the host side of the forward's GDN sequence region.
struct GdnPool {
    buffer: DeviceBuffer,
    /// One layer's layout; every layer's is the same.
    layout: GdnStateLayout,
    /// The GDN layers, as the tapes' [`GdnTape::layers`].
    layers: usize,
    /// Per layer, in `layers` order: the byte offsets of its conv buffer and its ssm buffer in
    /// `buffer`, each a multiple of [`CUDA_ARENA_ALIGN`].
    offsets: Vec<(u64, u64)>,
    /// The host side of the sequence region, `[slot, q_len, start, 0]`, sized once.
    sequence_host: [u32; 4],
}

impl GdnPool {
    /// The device address of layer index `layer`'s conv buffer, and of its ssm buffer. Computed,
    /// not dereferenced.
    fn layer_ptrs(&self, layer: usize) -> (*mut c_void, *mut c_void) {
        let (conv, ssm) = self.offsets[layer];
        let at = |b: u64| {
            let b = usize::try_from(b).expect("a pool offset fits the host's usize");
            self.buffer.ptr().wrapping_byte_add(b)
        };
        (at(conv), at(ssm))
    }
}

/// The GDN pool of `layers` layers of `layout`: per layer, the conv buffer then the ssm buffer,
/// each at a multiple of [`CUDA_ARENA_ALIGN`]. Returns each layer's two offsets and the total
/// bytes, or `None` on overflow.
fn gdn_pool_layout(layout: &GdnStateLayout, layers: usize) -> Option<(Vec<(u64, u64)>, u64)> {
    // The buffers' bytes from the layout's u32 fields, every product checked: the layout's own
    // element counts multiply unchecked in `usize`.
    let f32_buffer = |factors: &[u32]| {
        (factors.iter())
            .try_fold(std::mem::size_of::<f32>() as u64, |acc, &f| {
                acc.checked_mul(u64::from(f))
            })?
            .checked_next_multiple_of(CUDA_ARENA_ALIGN)
    };
    let l = layout;
    let conv = f32_buffer(&[l.num_slots, l.conv_dim, l.conv_kernel.checked_sub(1)?])?;
    let ssm = f32_buffer(&[l.num_slots, l.num_v_heads, l.head_v_dim, l.head_k_dim])?;
    let per_layer = conv.checked_add(ssm)?;
    let offsets = (0..layers as u64)
        .map(|l| {
            let at = l.checked_mul(per_layer)?;
            Some((at, at.checked_add(conv)?))
        })
        .collect::<Option<Vec<_>>>()?;
    Some((offsets, per_layer.checked_mul(layers as u64)?))
}

/// The name of `s` when it is a step the lowering emits and no kernel launches yet (#885's 13b-3
/// and 13b-4): the 64-column GEMM, the router GEMM on it, a rope with a half-head pairing, a rope
/// append or attention of a class that is not the KV pool's own at head dim 256 (Gemma's global
/// layers, head dim 512, a class the pool does not hold yet), and an expert step at another count
/// than 256, on 64-column tiles or reading Gemma's own tensors. A class of the pool's geometry at
/// head dim 256 launches with its window or without one (every Qwen layer, Gemma's sliding ones). Also the Qwen routing at another count
/// than [`MOE_ROUTE_EXPERTS`]: the routing kernel's Qwen program is instantiated at 256 only (its
/// launcher refuses any other count), so such a tape is refused here, before anything is enqueued,
/// rather than at its routing step. `None` for every step a Qwen tape holds.
fn not_launched(tape: &CudaTape, s: &CudaStep) -> Option<&'static str> {
    use crate::step::RopePairing;
    // The pool is laid out at the tape's `kv_heads` and `head_dim` (class 0's), so only a class
    // of that geometry is one its layers' K and V fit.
    let pool_class = |layer: u32| {
        tape.kv.zip(tape.kv_class(layer)).is_some_and(|(kv, c)| {
            c.head_dim == KV_APPEND_BF16_HEAD_DIM
                && (c.kv_heads, c.head_dim) == (kv.kv_heads, kv.head_dim)
        })
    };
    match *s {
        CudaStep::GemmCol64(_) => Some("GemmCol64"),
        CudaStep::Rope(RopeBf16 { pairing, .. }) if pairing != RopePairing::HalfRotated => {
            Some("Rope")
        }
        CudaStep::RopeAppend(RopeAppendBf16 { layer, pairing, .. })
            if pairing != RopePairing::HalfRotated || !pool_class(layer) =>
        {
            Some("RopeAppend")
        }
        CudaStep::AttnPrefill(AttnPrefillBf16 { layer, .. }) if !pool_class(layer) => {
            Some("AttnPrefill")
        }
        CudaStep::AttnDecode(AttnPrefillBf16 { layer, .. }) if !pool_class(layer) => {
            Some("AttnDecode")
        }
        CudaStep::RouterLogits(r) if r.kernel == MatmulKernel::Col64 => Some("RouterLogits"),
        CudaStep::MoeRoute(MoeRouteBf16 {
            program: RouteProgram::QwenSoftmax,
            experts,
            ..
        }) if experts != MOE_ROUTE_EXPERTS => Some("MoeRoute"),
        CudaStep::ExpertSort(e) if e.experts != MOE_ROUTE_EXPERTS => Some("ExpertSort"),
        CudaStep::ExpertGemm(e)
            if e.experts != MOE_ROUTE_EXPERTS
                || e.n_tile != 256
                || e.weight != MoeWeight::Bundle =>
        {
            Some("ExpertGemm")
        }
        CudaStep::ExpertGemv(e)
            if e.experts != MOE_ROUTE_EXPERTS || e.weight != MoeWeight::Bundle =>
        {
            Some("ExpertGemv")
        }
        _ => None,
    }
}

/// The KV pool's bytes: `layers` layers of a K and a V buffer of `num_blocks` blocks of `kv_heads ·
/// block_size · head_dim` bf16 each, or `None` when the count overflows `u64`. Every factor is
/// widened and multiplied checked, so no step wraps.
fn kv_pool_bytes(layers: usize, layout: &PagedKvLayout) -> Option<u64> {
    [
        layout.num_blocks,
        layout.num_kv_heads,
        layout.block_size,
        layout.head_dim,
    ]
    .into_iter()
    .try_fold(u64::try_from(layers).ok()?.checked_mul(2)?, |acc, f| {
        acc.checked_mul(u64::from(f))
    })?
    .checked_mul(std::mem::size_of::<u16>() as u64)
}

/// The one GDN geometry of `tapes`: the first GDN tape's, which every other GDN tape must share.
fn gdn_geometry(tapes: &[CudaTape]) -> Result<Option<GdnTape>, WorkerError> {
    let mut first: Option<GdnTape> = None;
    for (bucket, tape) in tapes.iter().enumerate() {
        let Some(g) = tape.gdn else { continue };
        match first {
            None => first = Some(g),
            Some(f) => {
                if (
                    f.layers,
                    f.num_k_heads,
                    f.num_v_heads,
                    f.head_k,
                    f.head_v,
                    f.conv_kernel,
                ) != (
                    g.layers,
                    g.num_k_heads,
                    g.num_v_heads,
                    g.head_k,
                    g.head_v,
                    g.conv_kernel,
                ) {
                    return Err(WorkerError::GdnGeometry { bucket });
                }
            }
        }
    }
    Ok(first)
}

/// Plays a set of tapes (one per bucket) over one arena on one stream.
///
/// `W` holds the weight buffers: borrowed by default ([`Self::new`]), or owned, a
/// `Vec<DeviceBuffer>` ([`CudaWorker::owning`]), for a caller that keeps the worker and its
/// weights together.
///
/// Neither `Send` nor `Sync`: it holds raw CUDA handles. Sharing is the worker pool's decision.
pub struct CudaWorker<'a, W: AsRef<[DeviceBuffer]> = &'a [DeviceBuffer]> {
    tapes: &'static [CudaTape],
    /// Indexed by each step's `WeightIx`.
    weights: W,
    /// The borrow a borrowed `W` holds.
    borrow: PhantomData<&'a [DeviceBuffer]>,
    stream: Stream,
    arena: DeviceBuffer,
    /// The host side of the rotary tables, sized once for the largest tape's two: each forward
    /// fills each of its tables' rows from the positions (the global table first, then the local
    /// one after it) and uploads them, so a forward allocates nothing.
    rotary_host: Vec<u16>,
    /// The KV pool, when a tape has a KV cache.
    kv: Option<KvPool>,
    /// The GDN state pool, when a tape has GDN layers.
    gdn: Option<GdnPool>,
}

/// The one KV geometry of `tapes`: the first KV tape's, which every other KV tape must share.
fn kv_geometry(tapes: &[CudaTape]) -> Result<Option<KvTape>, WorkerError> {
    let mut first: Option<KvTape> = None;
    for (bucket, tape) in tapes.iter().enumerate() {
        let Some(kv) = tape.kv else { continue };
        match first {
            None => first = Some(kv),
            Some(f) => {
                if (
                    f.layers,
                    f.kv_heads,
                    f.head_dim,
                    f.block_size,
                    f.max_blocks_per_seq,
                ) != (
                    kv.layers,
                    kv.kv_heads,
                    kv.head_dim,
                    kv.block_size,
                    kv.max_blocks_per_seq,
                ) {
                    return Err(WorkerError::KvGeometry { bucket });
                }
            }
        }
    }
    Ok(first)
}

impl<'a> CudaWorker<'a> {
    /// A worker for `tapes` reading the borrowed `weights`: validates every tape
    /// ([`CudaTape::validate`]) and checks once that every weight a tape declares is large
    /// enough, before it allocates anything; then makes the stream and the one arena allocation,
    /// of the largest tape's `total_bytes`.
    ///
    /// Together the two checks are what makes [`Self::forward`] safe to call with any tape: every
    /// span a step touches lies inside its tape's arena, which is at most the worker's, and every
    /// weight a step reads is at most its declared `weight_bytes`, which is at most the buffer.
    ///
    /// When a tape has a KV cache, `kv_pool` must be given with at least one block, and every KV
    /// tape must share one geometry; the pool is then one more allocation, of `layers · 2 ·
    /// PagedKvLayout::buffer_elems()` bf16, never cleared: an unwritten slot holds anything, NaN
    /// included. The attention kernel reads unwritten slots only in the partial last tile of a
    /// sequence, masks their scores, and zeroes their V rows in shared memory before P·V
    /// (`kernels/attn_prefill_bf16.cu`), so they reach no output. When no tape has one,
    /// `kv_pool` must be `None`.
    ///
    /// When a tape has GDN layers, `gdn_pool` must be given with at least one slot, and every GDN
    /// tape must share one geometry; the pool is then one more allocation: per GDN layer, in
    /// [`GdnTape::layers`] order, the conv buffer (`GdnStateLayout::conv_buffer_elems` f32) then
    /// the ssm buffer (`ssm_buffer_elems` f32), each at a multiple of 256. It is never cleared: a
    /// sequence's first forward starts from zero without reading its slot (the `fresh` flag of
    /// [`GdnSequence`]), and overwrites it in full. When no tape has GDN layers, `gdn_pool` must
    /// be `None`.
    pub fn new(
        device: &CudaDevice,
        tapes: &'static [CudaTape],
        weights: &'a [DeviceBuffer],
        kv_pool: Option<KvPoolConfig>,
        gdn_pool: Option<GdnPoolConfig>,
    ) -> Result<Self, WorkerError> {
        Self::build(device, tapes, weights, kv_pool, gdn_pool)
    }
}

impl CudaWorker<'static, Vec<DeviceBuffer>> {
    /// A worker for `tapes` that owns `weights`, the buffers `load_weights` made: the same
    /// checks and allocations as [`CudaWorker::new`].
    pub fn owning(
        device: &CudaDevice,
        tapes: &'static [CudaTape],
        weights: Vec<DeviceBuffer>,
        kv_pool: Option<KvPoolConfig>,
        gdn_pool: Option<GdnPoolConfig>,
    ) -> Result<Self, WorkerError> {
        Self::build(device, tapes, weights, kv_pool, gdn_pool)
    }
}

impl<'a, W: AsRef<[DeviceBuffer]>> CudaWorker<'a, W> {
    /// The one constructor behind [`CudaWorker::new`] and [`CudaWorker::owning`], which
    /// differ only in how they hold `weights`.
    fn build(
        device: &CudaDevice,
        tapes: &'static [CudaTape],
        weights: W,
        kv_pool: Option<KvPoolConfig>,
        gdn_pool: Option<GdnPoolConfig>,
    ) -> Result<Self, WorkerError> {
        for (bucket, tape) in tapes.iter().enumerate() {
            tape.validate()
                .map_err(|error| WorkerError::Tape { bucket, error })?;
        }
        for tape in tapes {
            for (ix, &need) in tape.weight_bytes.iter().enumerate() {
                if need == 0 {
                    continue;
                }
                let have = (weights.as_ref())
                    .get(ix)
                    .ok_or(WorkerError::MissingWeight { ix, need })?
                    .bytes();
                if have < need {
                    return Err(WorkerError::WeightTooSmall { ix, need, have });
                }
            }
        }
        let geometry = kv_geometry(tapes)?;
        let pool = match (geometry, kv_pool) {
            (Some(kv), Some(KvPoolConfig { num_blocks })) if num_blocks >= 1 => {
                let layout = PagedKvLayout {
                    num_blocks,
                    num_kv_heads: kv.kv_heads,
                    block_size: kv.block_size,
                    head_dim: kv.head_dim,
                };
                let bytes = kv_pool_bytes(kv.layers.len(), &layout)
                    .ok_or(WorkerError::KvPoolTooLarge { num_blocks })?;
                Some((kv, layout, bytes))
            }
            (Some(_), _) => return Err(WorkerError::KvPoolMissing),
            (None, Some(_)) => return Err(WorkerError::KvPoolUnused),
            (None, None) => None,
        };
        let gdn_layout = match (gdn_geometry(tapes)?, gdn_pool) {
            (Some(g), Some(GdnPoolConfig { num_slots })) if num_slots >= 1 => {
                let layout = GdnStateLayout::new(
                    num_slots,
                    g.num_k_heads,
                    g.num_v_heads,
                    g.head_k,
                    g.head_v,
                    g.conv_kernel,
                );
                let (offsets, bytes) = gdn_pool_layout(&layout, g.layers.len())
                    .ok_or(WorkerError::GdnPoolTooLarge { num_slots })?;
                Some((layout, g.layers.len(), offsets, bytes))
            }
            (Some(_), _) => return Err(WorkerError::GdnPoolMissing),
            (None, Some(_)) => return Err(WorkerError::GdnPoolUnused),
            (None, None) => None,
        };
        let arena_bytes =
            (tapes.iter().map(|t| t.arena.total_bytes).max()).ok_or(WorkerError::NoTapes)?;
        // Each table's span lies inside its arena (`validate`), so its byte count fits a usize; a
        // tape's two tables are filled side by side.
        let rotary_elements = (tapes.iter())
            .map(|t| {
                (t.rotary.iter().chain(&t.rotary_local))
                    .map(|r| r.span.bytes as usize / std::mem::size_of::<u16>())
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0);
        let stream = Stream::new().map_err(WorkerError::Cuda)?;
        let arena = DeviceBuffer::alloc(device, arena_bytes).map_err(WorkerError::Cuda)?;
        let kv = match pool {
            Some((kv, layout, bytes)) => {
                let buffer = DeviceBuffer::alloc(device, bytes).map_err(WorkerError::Cuda)?;
                let slots = (tapes.iter())
                    .filter(|t| t.kv.is_some())
                    .map(|t| t.bucket_m);
                Some(KvPool {
                    buffer,
                    layout,
                    layers: kv.layers.len(),
                    block_table_host: vec![NO_SLOT; kv.max_blocks_per_seq as usize],
                    slot_mapping_host: vec![NO_SLOT; slots.max().unwrap_or(0) as usize],
                })
            }
            None => None,
        };
        let gdn = match gdn_layout {
            Some((layout, layers, offsets, bytes)) => Some(GdnPool {
                buffer: DeviceBuffer::alloc(device, bytes).map_err(WorkerError::Cuda)?,
                layout,
                layers,
                offsets,
                sequence_host: [0; 4],
            }),
            None => None,
        };
        Ok(Self {
            tapes,
            weights,
            borrow: PhantomData,
            stream,
            arena,
            rotary_host: vec![0; rotary_elements],
            kv,
            gdn,
        })
    }

    /// The GDN state pool's bytes, when the worker has one: its one allocation.
    pub fn gdn_pool_bytes(&self) -> Option<u64> {
        self.gdn.as_ref().map(|p| p.buffer.bytes())
    }

    /// Copy slot `slot` of GDN layer index `layer`'s state into `conv` (`conv_elems_per_slot`
    /// f32, `[conv_dim, conv_kernel − 1]`) and `ssm` (`ssm_elems_per_slot` f32, `[nv, head_v,
    /// head_k]`), synchronously: the pool's bytes, for a check after a forward.
    pub fn read_gdn_state(
        &mut self,
        layer: usize,
        slot: u32,
        conv: &mut [f32],
        ssm: &mut [f32],
    ) -> Result<(), ForwardError> {
        let refuse = |want: (usize, usize)| ForwardError::GdnRead {
            layer,
            slot,
            conv: (want.0, conv.len()),
            ssm: (want.1, ssm.len()),
        };
        let Some(pool) = &self.gdn else {
            return Err(refuse((0, 0)));
        };
        let l = pool.layout;
        let want = (l.conv_elems_per_slot(), l.ssm_elems_per_slot());
        if layer >= pool.layers || slot >= l.num_slots || (conv.len(), ssm.len()) != want {
            return Err(refuse(want));
        }
        let (conv_buf, ssm_buf) = pool.layer_ptrs(layer);
        let src = |base: *mut c_void, elems: usize| {
            base.wrapping_byte_add(elems * std::mem::size_of::<f32>())
        };
        let copies = [
            (
                conv.as_mut_ptr().cast::<c_void>(),
                src(conv_buf, l.conv_slot_offset(slot)),
                std::mem::size_of_val(conv),
            ),
            (
                ssm.as_mut_ptr().cast::<c_void>(),
                src(ssm_buf, l.ssm_slot_offset(slot)),
                std::mem::size_of_val(ssm),
            ),
        ];
        let mut copied = Ok(());
        for (dst, src, bytes) in copies {
            // SAFETY: the layer and slot lie inside the pool, `dst` holds exactly one slot's
            // bytes of its buffer, and both host buffers stay borrowed until the stream wait
            // below has returned.
            copied = copied.and_then(|()| {
                check(
                    unsafe {
                        ffi::cudaMemcpyAsync(
                            dst,
                            src,
                            bytes,
                            ffi::MEMCPY_DEVICE_TO_HOST,
                            self.stream.0,
                        )
                    },
                    "cudaMemcpyAsync(GDN state, device to host)",
                )
                .map_err(ForwardError::Cuda)
            });
        }
        let drained = self.stream.synchronize().map_err(ForwardError::Cuda);
        copied.and(drained)
    }

    /// The KV pool's bytes, when the worker has one: its one allocation.
    pub fn kv_pool_bytes(&self) -> Option<u64> {
        self.kv.as_ref().map(|p| p.buffer.bytes())
    }

    /// The arena's bytes: its one allocation.
    pub fn arena_bytes(&self) -> u64 {
        self.arena.bytes()
    }

    /// Run bucket `bucket`'s tape: upload `input` into its input span (and, when the tape has a
    /// rotary table, that table's rows for `positions`, one per token), launch every step in
    /// order, download its result span into `result`, and wait once. Returns once the GPU is done,
    /// error or not, so nothing the forward touched is still in use.
    ///
    /// `input` is of the kind the tape takes ([`TapeInput`]), else [`ForwardError::InputKind`],
    /// before any copy: the embedded hidden rows, exactly the input span's bytes, or one token id
    /// per bucket row, each below the tape's `vocab` ([`ForwardError::TokenOutOfVocab`]); a count
    /// other than `bucket_m` is [`ForwardError::InputSize`].
    ///
    /// `positions` holds one position per token when the tape has a rotary table, and is empty
    /// when it has none; anything else is [`ForwardError::Positions`], before any copy.
    ///
    /// `kv` is the forward's sequence, given exactly when the tape has a KV cache. Before any copy
    /// it is checked to be one the kernels can run safely: `1 ≤ q_len ≤ bucket_m`, the tokens fit
    /// a block table, the table has the entries the tokens need and no more than the tape holds,
    /// every used entry is a distinct block inside the pool, and the real rows' positions are
    /// `prefix_len, prefix_len + 1, …`. Then the used entries (the rest `u32::MAX`), each row's
    /// slot (`u32::MAX` for a padding row) and `[prefix_len, q_len]` are uploaded with the input.
    ///
    /// `gdn` is the forward's GDN sequence, given exactly when the tape has GDN layers. Before any
    /// copy it is checked: its slot is one of the pool's, and `1 ≤ q_len ≤ bucket_m`. Then
    /// `[slot, q_len, fresh as u32, 0]` is uploaded into the tape's GDN sequence region with the
    /// input.
    ///
    /// Given both, they must agree on whether this is the sequence's first forward: `fresh`
    /// exactly when `prefix_len` is 0, else [`ForwardError::FreshDisagrees`], before any copy.
    pub fn forward(
        &mut self,
        bucket: usize,
        input: Input<'_>,
        positions: &[u32],
        kv: Option<&KvSequence<'_>>,
        gdn: Option<&GdnSequence>,
        result: &mut [u16],
    ) -> Result<(), ForwardError> {
        let tape = self.tapes.get(bucket).ok_or(ForwardError::Bucket {
            bucket,
            buckets: self.tapes.len(),
        })?;
        if let Some((step, op)) = (tape.steps.iter().enumerate())
            .find_map(|(step, s)| not_launched(tape, s).map(|op| (step, op)))
        {
            return Err(ForwardError::NotLaunched { step, op });
        }
        let ids = match (tape.input, input) {
            (TapeInput::Hidden(_), Input::Hidden(_)) => None,
            (TapeInput::TokenIds { vocab, .. }, Input::TokenIds(ids)) => Some((ids, vocab)),
            (TapeInput::Hidden(_), got) => {
                return Err(ForwardError::InputKind {
                    expected: "hidden rows",
                    got: got.kind(),
                });
            }
            (TapeInput::TokenIds { .. }, got) => {
                return Err(ForwardError::InputKind {
                    expected: "token ids",
                    got: got.kind(),
                });
            }
        };
        let (got, expected) = (input.bytes(), tape.input.span().bytes);
        if got != expected {
            return Err(ForwardError::InputSize { expected, got });
        }
        if let Some((ids, vocab)) = ids
            && let Some((row, &id)) = ids.iter().enumerate().find(|&(_, &id)| id >= vocab)
        {
            return Err(ForwardError::TokenOutOfVocab { row, id, vocab });
        }
        let got = std::mem::size_of_val(result) as u64;
        if got != tape.result.bytes {
            return Err(ForwardError::ResultSize {
                expected: tape.result.bytes,
                got,
            });
        }
        let expected = tape.rotary.map_or(0, |_| tape.bucket_m as usize);
        if positions.len() != expected {
            return Err(ForwardError::Positions {
                expected,
                got: positions.len(),
            });
        }
        if let (Some(kv), Some(gdn)) = (kv, gdn)
            && (kv.prefix_len == 0) != gdn.fresh
        {
            return Err(ForwardError::FreshDisagrees {
                prefix_len: kv.prefix_len,
                fresh: gdn.fresh,
            });
        }
        let sequence = match (tape.kv, kv) {
            (Some(kv_tape), Some(seq)) => {
                let pool = self.kv.as_mut().expect("`new` made a pool for a KV tape");
                let data = fill_sequence(tape, &kv_tape, seq, positions, pool)?;
                Some((kv_tape, data))
            }
            (Some(_), None) => return Err(ForwardError::KvMissing),
            (None, Some(_)) => return Err(ForwardError::KvUnexpected),
            (None, None) => None,
        };
        let gdn_sequence = match (tape.gdn, gdn) {
            (Some(gdn_tape), Some(seq)) => {
                let pool = self.gdn.as_mut().expect("`new` made a pool for a GDN tape");
                let num_slots = pool.layout.num_slots;
                if seq.slot >= num_slots {
                    return Err(ForwardError::GdnSlotOutsidePool {
                        slot: seq.slot,
                        num_slots,
                    });
                }
                let bucket_m = tape.bucket_m;
                if seq.q_len == 0 || seq.q_len > bucket_m {
                    return Err(ForwardError::GdnQueryLength {
                        q_len: seq.q_len,
                        bucket_m,
                    });
                }
                pool.sequence_host = [seq.slot, seq.q_len, u32::from(seq.fresh), 0];
                Some(gdn_tape.sequence)
            }
            (Some(_), None) => return Err(ForwardError::GdnMissing),
            (None, Some(_)) => return Err(ForwardError::GdnUnexpected),
            (None, None) => None,
        };
        // The global table's rows, then the local table's after them; each table is a plain rope
        // at its own `rot_dim` and theta (the local table's `rot_dim` is its head dim, a full
        // rope). `validate` refuses a local table without a global one.
        let (global_host, local_host) = self.rotary_host.split_at_mut(
            tape.rotary
                .map_or(0, |r| positions.len() * r.rot_dim as usize),
        );
        let tables = [
            fill_table(tape.rotary, positions, global_host),
            fill_table(tape.rotary_local, positions, local_host),
        ];
        let kv_upload = match (&sequence, &self.kv) {
            (Some((kv_tape, data)), Some(pool)) => Some(KvUpload {
                tape: *kv_tape,
                block_table: &pool.block_table_host,
                slot_mapping: &pool.slot_mapping_host[..tape.bucket_m as usize],
                sequence: *data,
            }),
            _ => None,
        };
        let gdn_upload = match (gdn_sequence, &self.gdn) {
            (Some(span), Some(pool)) => Some((span, &pool.sequence_host)),
            _ => None,
        };
        let enqueued = Self::enqueue(
            &self.stream,
            &self.arena,
            self.weights.as_ref(),
            (self.kv.as_ref(), self.gdn.as_ref()),
            tape,
            Uploads {
                input,
                tables,
                kv: kv_upload,
                gdn: gdn_upload,
            },
            result,
        );
        let drained = self.stream.synchronize().map_err(ForwardError::Cuda);
        enqueued.and(drained)
    }

    /// Copy the arena span `span` into `out` after a forward: the counterpart of reading a metal
    /// worker's arena slot. Synchronous.
    pub fn read_arena(&mut self, span: ArenaSpan, out: &mut [u16]) -> Result<(), ForwardError> {
        let got = std::mem::size_of_val(out) as u64;
        if got != span.bytes {
            return Err(ForwardError::ReadSize {
                expected: span.bytes,
                got,
            });
        }
        if span
            .offset
            .0
            .checked_add(span.bytes)
            .is_none_or(|end| end > self.arena.bytes())
        {
            return Err(ForwardError::SpanOutsideArena {
                span,
                arena_bytes: self.arena.bytes(),
            });
        }
        // SAFETY: the span lies inside the arena, `out` holds exactly its bytes, and `out` stays
        // borrowed until the stream wait below has returned.
        let copied = check(
            unsafe {
                ffi::cudaMemcpyAsync(
                    out.as_mut_ptr().cast(),
                    self.slot(span.offset),
                    std::mem::size_of_val(out),
                    ffi::MEMCPY_DEVICE_TO_HOST,
                    self.stream.0,
                )
            },
            "cudaMemcpyAsync(device to host)",
        )
        .map_err(ForwardError::Cuda);
        let drained = self.stream.synchronize().map_err(ForwardError::Cuda);
        copied.and(drained)
    }

    /// Copy block `block` of KV layer index `layer`'s `half` into `out` (`kv_heads · block_size ·
    /// head_dim` bf16), synchronously: the cache's bytes, for a check after a forward.
    pub fn read_kv(
        &mut self,
        layer: usize,
        half: KvHalf,
        block: u32,
        out: &mut [u16],
    ) -> Result<(), ForwardError> {
        let refuse = |expected| ForwardError::KvRead {
            layer,
            block,
            expected,
            got: out.len(),
        };
        let Some(pool) = &self.kv else {
            return Err(refuse(0));
        };
        let block_elems = pool.layout.kv_blk_stride();
        if layer >= pool.layers || block >= pool.layout.num_blocks || out.len() != block_elems {
            return Err(refuse(block_elems));
        }
        let at = pool.half_elems(layer, half) + pool.layout.elem_offset(block, 0, 0);
        let src = pool
            .buffer
            .ptr()
            .wrapping_byte_add(at * std::mem::size_of::<u16>());
        // SAFETY: the layer and block lie inside the pool, `out` holds exactly one block's bytes,
        // and `out` stays borrowed until the stream wait below has returned.
        let copied = check(
            unsafe {
                ffi::cudaMemcpyAsync(
                    out.as_mut_ptr().cast(),
                    src,
                    std::mem::size_of_val(out),
                    ffi::MEMCPY_DEVICE_TO_HOST,
                    self.stream.0,
                )
            },
            "cudaMemcpyAsync(KV block, device to host)",
        )
        .map_err(ForwardError::Cuda);
        let drained = self.stream.synchronize().map_err(ForwardError::Cuda);
        copied.and(drained)
    }

    /// Everything one forward enqueues, stopping at the first failure: the input copy, the rotary
    /// table's rows when the tape has one (`table`: its span and the rows, which hold exactly its
    /// bytes), the KV regions when it has a KV cache, every launch and the result copy. The
    /// caller drains the stream whatever this returns.
    fn enqueue(
        stream: &Stream,
        arena: &DeviceBuffer,
        weights: &[DeviceBuffer],
        (kv, gdn): (Option<&KvPool>, Option<&GdnPool>),
        tape: &CudaTape,
        uploads: Uploads<'_>,
        result: &mut [u16],
    ) -> Result<(), ForwardError> {
        let Uploads {
            input,
            tables,
            kv: kv_upload,
            gdn: gdn_upload,
        } = uploads;
        let slot = |offset| Self::slot_of(arena, offset);
        // SAFETY: the input span lies inside the tape's arena (`new` validated it), which is at
        // most `arena_bytes`; `input` holds exactly its bytes (`forward` checked), and `input`
        // stays borrowed until the caller's stream wait has returned.
        check(
            unsafe {
                ffi::cudaMemcpyAsync(
                    slot(tape.input.span().offset),
                    input.ptr(),
                    input.bytes() as usize,
                    ffi::MEMCPY_HOST_TO_DEVICE,
                    stream.0,
                )
            },
            "cudaMemcpyAsync(host to device)",
        )
        .map_err(ForwardError::Cuda)?;
        for (span, rows) in tables.into_iter().flatten() {
            debug_assert_eq!(std::mem::size_of_val(rows) as u64, span.bytes);
            // SAFETY: the rotary span lies inside the tape's arena (`new` validated it), which is
            // at most `arena_bytes`; `rows` is `bucket_m · rot_dim` elements (`forward` checked
            // `positions.len() == bucket_m`), exactly the span's bytes, and it is the worker's own
            // buffer, which no forward touches until the caller's stream wait has returned.
            check(
                unsafe {
                    ffi::cudaMemcpyAsync(
                        slot(span.offset),
                        rows.as_ptr().cast(),
                        std::mem::size_of_val(rows),
                        ffi::MEMCPY_HOST_TO_DEVICE,
                        stream.0,
                    )
                },
                "cudaMemcpyAsync(rotary table, host to device)",
            )
            .map_err(ForwardError::Cuda)?;
        }
        if let Some(up) = &kv_upload {
            for (span, data, call) in [
                (
                    up.tape.block_table,
                    up.block_table,
                    "cudaMemcpyAsync(block table, host to device)",
                ),
                (
                    up.tape.slot_mapping,
                    up.slot_mapping,
                    "cudaMemcpyAsync(slot mapping, host to device)",
                ),
                (
                    up.tape.sequence,
                    &up.sequence[..],
                    "cudaMemcpyAsync(sequence, host to device)",
                ),
            ] {
                debug_assert_eq!(std::mem::size_of_val(data) as u64, span.bytes);
                // SAFETY: each region lies inside the tape's arena (`new` validated it), which is
                // at most `arena_bytes`; `data` holds exactly its bytes (`max_blocks_per_seq`,
                // `bucket_m` and 2 u32, which `validate` checked), and it is the worker's own
                // buffer or `kv_upload`'s, which no forward touches until the caller's stream
                // wait has returned.
                check(
                    unsafe {
                        ffi::cudaMemcpyAsync(
                            slot(span.offset),
                            data.as_ptr().cast(),
                            std::mem::size_of_val(data),
                            ffi::MEMCPY_HOST_TO_DEVICE,
                            stream.0,
                        )
                    },
                    call,
                )
                .map_err(ForwardError::Cuda)?;
            }
        }
        if let Some((span, data)) = gdn_upload {
            debug_assert_eq!(std::mem::size_of_val(data) as u64, span.bytes);
            // SAFETY: the GDN sequence region lies inside the tape's arena (`new` validated it),
            // which is at most `arena_bytes`; `data` is its 16 bytes (`validate` checked the
            // region's size), the worker's own buffer, which no forward touches until the
            // caller's stream wait has returned.
            check(
                unsafe {
                    ffi::cudaMemcpyAsync(
                        slot(span.offset),
                        data.as_ptr().cast(),
                        std::mem::size_of_val(data),
                        ffi::MEMCPY_HOST_TO_DEVICE,
                        stream.0,
                    )
                },
                "cudaMemcpyAsync(GDN sequence, host to device)",
            )
            .map_err(ForwardError::Cuda)?;
        }
        // SAFETY (every arm): `new` validated the tape (`CudaTape::validate`): each span a step
        // reads or writes (`CudaStep::extents`, the kernel's own operand sizes) is 256-byte
        // aligned and inside the tape's arena, which is at most this worker's; each span it
        // writes overlaps none it reads, nor its other write; each weight it reads has an index
        // the tape declares, declared at least as large as the step reads, and `new` checked every
        // declared size against its buffer. cudaMalloc pointers are 256-byte aligned. The stream
        // is live. A KV step's layer is one of the tape's (`validate`), so of the pool's (`new`
        // checked every KV tape has the pool's geometry), and every block a kernel touches is one
        // `forward` checked is inside the pool.
        for (step, s) in tape.steps.iter().enumerate() {
            let launched = match *s {
                // Refused above, before anything was enqueued (`not_launched`).
                CudaStep::GemmCol64(_) => {
                    let op = not_launched(tape, s).unwrap_or("a Gemma step");
                    return Err(ForwardError::NotLaunched { step, op });
                }
                CudaStep::Scale(ScaleBf16 {
                    x,
                    out,
                    rows,
                    cols,
                    scale_bits,
                }) => {
                    // SAFETY: as above, for `x` and `out` (`rows*cols` each).
                    unsafe {
                        kernels::scale_bf16(
                            slot(x),
                            slot(out),
                            (rows, cols),
                            f32::from_bits(scale_bits),
                            stream.0,
                        )
                    }
                }
                CudaStep::LayerScale(LayerScaleBf16 {
                    x,
                    scale,
                    out,
                    rows,
                    cols,
                }) => {
                    let w = weights[scale.0 as usize].ptr();
                    // SAFETY: as above, for `x` and `out` (`rows*cols` each) and `scale` (one
                    // bf16).
                    unsafe {
                        kernels::layer_scale_bf16(slot(x), w, slot(out), (rows, cols), stream.0)
                    }
                }
                CudaStep::SoftCap(SoftCapBf16 {
                    x,
                    out,
                    rows,
                    cols,
                    cap_bits,
                }) => {
                    // SAFETY: as above, for `x` and `out` (`rows*cols` each).
                    unsafe {
                        kernels::soft_cap_bf16(
                            slot(x),
                            slot(out),
                            (rows, cols),
                            f32::from_bits(cap_bits),
                            stream.0,
                        )
                    }
                }
                CudaStep::RmsNormUnit(RmsNormUnitBf16 {
                    x,
                    y,
                    rows,
                    cols,
                    eps_bits,
                }) => {
                    // SAFETY: as above, for `x` and `y` (`rows*cols` each).
                    unsafe {
                        kernels::rms_norm_unit_bf16(
                            slot(x),
                            slot(y),
                            (rows, cols),
                            f32::from_bits(eps_bits),
                            stream.0,
                        )
                    }
                }
                CudaStep::GeluMul(SiluMulBf16 {
                    gate,
                    up,
                    out,
                    rows,
                    cols,
                }) => {
                    // SAFETY: as above, for `gate`, `up` and `out` (`rows*cols` each).
                    unsafe {
                        kernels::gelu_mul_bf16(
                            slot(gate),
                            slot(up),
                            slot(out),
                            (rows, cols),
                            stream.0,
                        )
                    }
                }
                CudaStep::Embed(EmbedBf16 {
                    tokens,
                    table,
                    out,
                    rows,
                    hidden,
                    vocab,
                }) => {
                    let table = weights[table.0 as usize].ptr();
                    // SAFETY: as above, for `tokens` (`rows` u32, the token ids uploaded above,
                    // every one below `vocab`), `table` (`vocab*hidden`) and `out`
                    // (`rows*hidden`).
                    unsafe {
                        kernels::embed_bf16(
                            slot(tokens),
                            table,
                            slot(out),
                            (rows, hidden, vocab),
                            stream.0,
                        )
                    }
                }
                CudaStep::Gemm(GemmBf16 { a, b, c, m, n, k }) => {
                    let b = weights[b.0 as usize].ptr();
                    // SAFETY: as above, for `a` (`m*k`), `b` (`n*k`) and `c` (`m*n`).
                    unsafe { kernels::gemm_bf16_abt(slot(a), b, slot(c), m, n, k, stream.0) }
                }
                CudaStep::GemmNarrow(GemmBf16 { a, b, c, m, n, k }) => {
                    let b = weights[b.0 as usize].ptr();
                    // SAFETY: as above, for `a` (`m*k`), `b` (`n*k`) and `c` (`m*n`).
                    unsafe {
                        kernels::gemm_narrow_bf16_abt(slot(a), b, slot(c), (m, n, k), stream.0)
                    }
                }
                CudaStep::Gemv(GemmBf16 { a, b, c, n, k, .. }) => {
                    let b = weights[b.0 as usize].ptr();
                    // SAFETY: as above, for `a` (`k`, its `m` is 1: `validate`), `b` (`n*k`) and
                    // `c` (`n`).
                    unsafe { kernels::gemv_bf16_abt(slot(a), b, slot(c), (n, k), stream.0) }
                }
                CudaStep::RouterLogits(RouterLogitsBf16 {
                    x,
                    bundle,
                    logits,
                    m,
                    experts,
                    hidden,
                    kernel,
                    ..
                }) => {
                    // Gemma's 64-column router GEMM: refused above, before anything was enqueued
                    // (`not_launched`).
                    if kernel == MatmulKernel::Col64 {
                        let op = not_launched(tape, s).unwrap_or("RouterLogits");
                        return Err(ForwardError::NotLaunched { step, op });
                    }
                    // The router is the bundle's first tensor, at byte 0 (`moe_bundle`), or Gemma's
                    // own projection, whole (`MoeWeight::router_bytes`).
                    let b = weights[bundle.0 as usize].ptr();
                    // SAFETY: as above, for `x` (`m*hidden`), the router (`experts*hidden`, the
                    // bundle's first bytes) and `logits` (`m*experts`).
                    unsafe {
                        match kernel {
                            MatmulKernel::Tiled => kernels::gemm_bf16_abt(
                                slot(x),
                                b,
                                slot(logits),
                                m,
                                experts,
                                hidden,
                                stream.0,
                            ),
                            MatmulKernel::Gemv => kernels::gemv_bf16_abt(
                                slot(x),
                                b,
                                slot(logits),
                                (experts, hidden),
                                stream.0,
                            ),
                            MatmulKernel::Col64 => unreachable!("returned NotLaunched above"),
                        }
                    }
                }
                CudaStep::MoeRoute(MoeRouteBf16 {
                    logits,
                    topk_idx,
                    scores,
                    m,
                    experts,
                    top_k,
                    renorm,
                    program,
                    expert_scale,
                }) => match (program, expert_scale) {
                    // SAFETY: as above, for `logits` (`m*experts`), `topk_idx` and `scores`
                    // (`m*top_k` each).
                    (RouteProgram::QwenSoftmax, _) => unsafe {
                        kernels::moe_route_bf16(
                            slot(logits),
                            (slot(topk_idx), slot(scores)),
                            (m, experts, top_k),
                            renorm,
                            stream.0,
                        )
                    },
                    (RouteProgram::GemmaScaledSoftmax { scale_bits }, pes) => {
                        // A Gemma routing with no per-expert scale is the launcher's refusal (a
                        // null pointer), before it touches any pointer.
                        let pes = pes.map_or(std::ptr::null(), |w| weights[w.0 as usize].ptr());
                        // SAFETY: as above, and `expert_scale` (`experts`).
                        unsafe {
                            kernels::gemma_route_bf16(
                                (slot(logits), pes),
                                (slot(topk_idx), slot(scores)),
                                (m, experts, top_k),
                                f32::from_bits(scale_bits),
                                stream.0,
                            )
                        }
                    }
                },
                CudaStep::ExpertSort(sort) => Self::expert_sort(&sort, slot, stream),
                CudaStep::ExpertGemm(g) => Self::expert_gemm(&g, slot, weights, stream),
                CudaStep::ExpertGemv(g) => Self::expert_gemv(&g, slot, weights, stream),
                CudaStep::ExpertCombine(ExpertCombineBf16 {
                    down,
                    pos,
                    scores,
                    out,
                    m,
                    top_k,
                    hidden,
                    down_rows,
                }) => {
                    // SAFETY: as above, for `down` (`down_rows*hidden`), `pos` when sorted and
                    // `scores` (`m*top_k` each), and `out` (`m*hidden`).
                    unsafe {
                        kernels::moe_combine_bf16(
                            (slot(down), pos.map(|p| slot(p).cast_const()), slot(scores)),
                            slot(out),
                            (m, top_k, hidden, down_rows),
                            stream.0,
                        )
                    }
                }
                CudaStep::GateScale(GateScaleBf16 {
                    routed,
                    shared,
                    g,
                    out,
                    rows,
                    cols,
                }) => {
                    // SAFETY: as above, for `routed`, `shared` and `out` (`rows*cols` each) and
                    // `g` (`rows`).
                    unsafe {
                        kernels::gate_scale_bf16(
                            (slot(routed), slot(shared), slot(g)),
                            slot(out),
                            (rows, cols),
                            stream.0,
                        )
                    }
                }
                CudaStep::GatedDeltaNet(g) => {
                    let pool = gdn.expect("`new` made a pool for a GDN tape");
                    Self::gated_delta_net(&g, pool, slot, weights, stream)
                }
                CudaStep::RmsNorm(
                    norm @ RmsNormBf16 {
                        x,
                        gain,
                        y,
                        rows,
                        cols,
                        convention,
                        ..
                    },
                )
                | CudaStep::RouterNorm(
                    norm @ RmsNormBf16 {
                        x,
                        gain,
                        y,
                        rows,
                        cols,
                        convention,
                        ..
                    },
                ) => {
                    let gain = weights[gain.0 as usize].ptr();
                    // SAFETY: as above, for `x` and `y` (`rows*cols`) and `gain` (`cols`).
                    unsafe {
                        kernels::rms_norm_bf16(
                            slot(x),
                            gain,
                            slot(y),
                            (rows, cols),
                            norm.eps(),
                            convention.offset(),
                            stream.0,
                        )
                    }
                }
                CudaStep::Add(AddBf16 {
                    a,
                    b,
                    out,
                    rows,
                    cols,
                }) => {
                    // SAFETY: as above, for `a`, `b` and `out` (`rows*cols` each).
                    unsafe {
                        kernels::add_bf16(slot(a), slot(b), slot(out), (rows, cols), stream.0)
                    }
                }
                CudaStep::SiluMul(SiluMulBf16 {
                    gate,
                    up,
                    out,
                    rows,
                    cols,
                }) => {
                    // SAFETY: as above, for `gate`, `up` and `out` (`rows*cols` each).
                    unsafe {
                        kernels::silu_mul_bf16(
                            slot(gate),
                            slot(up),
                            slot(out),
                            (rows, cols),
                            stream.0,
                        )
                    }
                }
                CudaStep::Rope(RopeBf16 {
                    x,
                    table,
                    out,
                    rows,
                    cols,
                    head_dim,
                    rot_dim,
                    ..
                }) => {
                    // SAFETY: as above, for `x` and `out` (`rows*cols`) and `table`
                    // (`rows*rot_dim`), whose rows this forward uploaded above.
                    unsafe {
                        kernels::rope_bf16(
                            slot(x),
                            slot(table),
                            slot(out),
                            (rows, cols),
                            (head_dim, rot_dim),
                            stream.0,
                        )
                    }
                }
                CudaStep::GateSplit(GateSplitBf16 {
                    qg,
                    q,
                    gate,
                    rows,
                    half_cols,
                    head_dim,
                }) => {
                    // SAFETY: as above, for `qg` (`rows*2*half_cols`), and `q` and `gate`
                    // (`rows*half_cols` each), its two writes, which overlap neither each other
                    // nor `qg`.
                    unsafe {
                        kernels::gate_split_bf16(
                            slot(qg),
                            slot(q),
                            slot(gate),
                            (rows, half_cols),
                            head_dim,
                            stream.0,
                        )
                    }
                }
                CudaStep::GateApply(GateApplyBf16 {
                    attn,
                    gate,
                    out,
                    rows,
                    cols,
                }) => {
                    // SAFETY: as above, for `attn`, `gate` and `out` (`rows*cols` each).
                    unsafe {
                        kernels::gate_apply_bf16(
                            slot(attn),
                            slot(gate),
                            slot(out),
                            (rows, cols),
                            stream.0,
                        )
                    }
                }
                CudaStep::RopeAppend(RopeAppendBf16 {
                    k,
                    v,
                    table,
                    slot_mapping,
                    k_out,
                    layer,
                    rows,
                    cols,
                    head_dim,
                    rot_dim,
                    ..
                }) => {
                    let pool = kv.expect("`new` made a pool for a KV tape");
                    let l = layer as usize;
                    let layout = pool.layout;
                    // SAFETY: as above, for `k`, `v` and `k_out` (`rows*cols`), `table`
                    // (`rows*rot_dim`) and `slot_mapping` (`rows` u32), uploaded above; the cache
                    // halves are layer `l`'s, `num_blocks · kv_heads · 64 · head_dim` each, and
                    // every slot uploaded is in a checked block or `u32::MAX`.
                    unsafe {
                        kernels::rope_bf16(
                            slot(k),
                            slot(table),
                            slot(k_out),
                            (rows, cols),
                            (head_dim, rot_dim),
                            stream.0,
                        )
                        .and_then(|()| {
                            kernels::kv_append_bf16(
                                (slot(k_out), slot(v)),
                                slot(slot_mapping),
                                (pool.half_ptr(l, KvHalf::K), pool.half_ptr(l, KvHalf::V)),
                                (rows, layout.num_kv_heads, layout.head_dim),
                                (layout.block_size, layout.num_blocks),
                                stream.0,
                            )
                        })
                    }
                }
                CudaStep::AttnPrefill(
                    attn @ AttnPrefillBf16 {
                        q,
                        block_table,
                        sequence,
                        out,
                        layer,
                        rows,
                        q_heads,
                        kv_heads,
                        head_dim,
                        window,
                        ..
                    },
                ) => {
                    let pool = kv.expect("`new` made a pool for a KV tape");
                    let l = layer as usize;
                    let layout = pool.layout;
                    // SAFETY: as above, for `q` and `out` (`rows*q_heads*head_dim`), and the
                    // block table and sequence uploaded above, which `forward` checked: every
                    // block the keys `0 .. P + L` need is a table entry inside the pool, and
                    // `1 ≤ L ≤ rows`. The cache halves are layer `l`'s.
                    unsafe {
                        kernels::attn_prefill_windowed_bf16(
                            (slot(q), slot(out)),
                            (pool.half_ptr(l, KvHalf::K), pool.half_ptr(l, KvHalf::V)),
                            (slot(block_table), slot(sequence)),
                            (rows, q_heads, kv_heads, head_dim),
                            (layout.block_size, layout.num_blocks),
                            (attn.scale(), window.unwrap_or(0)),
                            stream.0,
                        )
                    }
                }
                CudaStep::AttnDecode(
                    attn @ AttnPrefillBf16 {
                        q,
                        block_table,
                        sequence,
                        out,
                        layer,
                        q_heads,
                        kv_heads,
                        head_dim,
                        window,
                        ..
                    },
                ) => {
                    let pool = kv.expect("`new` made a pool for a KV tape");
                    let l = layer as usize;
                    let layout = pool.layout;
                    // SAFETY: as the prefill attention's, for `q` and `out` (`q_heads*head_dim`,
                    // its rows are 1: `validate`), and the block table and sequence uploaded
                    // above, which `forward` checked: every block the keys `0 .. P + L` need is a
                    // table entry inside the pool, and `L ≥ 1`. The cache halves are layer `l`'s.
                    unsafe {
                        kernels::attn_decode_windowed_bf16(
                            (slot(q), slot(out)),
                            (pool.half_ptr(l, KvHalf::K), pool.half_ptr(l, KvHalf::V)),
                            (slot(block_table), slot(sequence)),
                            (q_heads, kv_heads, head_dim),
                            (layout.block_size, layout.num_blocks),
                            (attn.scale(), window.unwrap_or(0)),
                            stream.0,
                        )
                    }
                }
            };
            launched.map_err(|error| ForwardError::Launch { step, error })?;
        }
        // SAFETY: as the input copy, for the result span and `result`.
        check(
            unsafe {
                ffi::cudaMemcpyAsync(
                    result.as_mut_ptr().cast(),
                    slot(tape.result.offset),
                    std::mem::size_of_val(result),
                    ffi::MEMCPY_DEVICE_TO_HOST,
                    stream.0,
                )
            },
            "cudaMemcpyAsync(device to host)",
        )
        .map_err(ForwardError::Cuda)
    }

    /// A GDN step's four launches, in order on the stream: conv1d, gating, scan, gated norm. Each
    /// reads the GDN sequence region this forward uploaded; the first that fails stops the rest.
    fn gated_delta_net(
        g: &GatedDeltaNetBf16,
        pool: &GdnPool,
        slot: impl Fn(ArenaOffset) -> *mut c_void,
        weights: &[DeviceBuffer],
        stream: &Stream,
    ) -> Result<(), KernelError> {
        // `validate` checked that the step's geometry is the tape's and its scratch and bundle
        // sizes fit u64 (`extents`), and `new` that the tape's geometry is the pool's; were one
        // to overflow, the conv launcher refuses the shape and nothing launches.
        let (Some(conv_dim), Some(scratch), Some(bundle)) =
            (g.conv_dim(), g.scratch_layout(), g.bundle_layout())
        else {
            return Err(KernelError::GdnConvShape {
                rows: g.rows,
                conv_dim: 0,
                kernel: g.conv_kernel,
            });
        };
        let at = |base: ArenaOffset, off: u64| slot(ArenaOffset(base.0 + off));
        let w = weights[g.bundle.0 as usize].ptr();
        let wat = |off: u64| {
            let off = usize::try_from(off).expect("a bundle offset fits the host's usize");
            w.wrapping_byte_add(off).cast_const()
        };
        let (conv_state, ssm_state) = pool.layer_ptrs(g.layer as usize);
        let layout = pool.layout;
        let seq = slot(g.sequence).cast_const();
        let (conv_out, gate, beta, o) = (
            at(g.scratch, scratch.conv_out),
            at(g.scratch, scratch.g),
            at(g.scratch, scratch.beta),
            at(g.scratch, scratch.o),
        );
        // SAFETY: as `enqueue`'s, for the step's arena operands (`extents`: `qkv`, `z`, `a`, `b`,
        // `out` and the scratch, whose four regions lie inside its span) and its bundle
        // (`gdn_bundle`'s bytes, which `new` checked the buffer holds). The state buffers are
        // layer `g.layer`'s, which `validate` checked is one of the tape's GDN layers and `new`
        // allocated with this geometry; the uploaded slot is below the pool's `num_slots` and
        // `q_len` in `1 ..= rows` (`forward` checked both), and the strides passed are the
        // pool's own layout's.
        unsafe {
            kernels::gdn_conv1d(
                (slot(g.qkv).cast_const(), wat(bundle.conv1d)),
                (conv_state, layout.conv_elems_per_slot() as u64),
                seq,
                conv_out,
                (g.rows, conv_dim, g.conv_kernel),
                stream.0,
            )?;
            kernels::gdn_gating(
                (slot(g.a).cast_const(), slot(g.b).cast_const()),
                (wat(bundle.a_log), wat(bundle.dt_bias)),
                seq,
                (gate, beta),
                (g.rows, g.num_v_heads),
                stream.0,
            )?;
            kernels::gdn_scan(
                (conv_out.cast_const(), gate.cast_const(), beta.cast_const()),
                (ssm_state, layout.ssm_elems_per_slot() as u64),
                seq,
                o,
                (g.rows, g.num_k_heads, g.num_v_heads),
                (g.head_k, g.head_v, g.scale()),
                stream.0,
            )?;
            kernels::gdn_rms_norm_gated(
                (o.cast_const(), slot(g.z).cast_const(), wat(bundle.norm)),
                seq,
                slot(g.out),
                (g.rows, g.num_v_heads, g.head_v),
                g.eps(),
                stream.0,
            )
        }
    }

    /// Enqueue the MoE sort's three launches (`kernels::moe_sort`) on the stream, its index
    /// regions placed by the step's own [`crate::step::SortLayout`].
    fn expert_sort(
        s: &ExpertSortBf16,
        slot: impl Fn(ArenaOffset) -> *mut c_void,
        stream: &Stream,
    ) -> Result<(), KernelError> {
        // `validate` checked the step's layout fits u64 (`extents`); were it to overflow, the
        // sort refuses its shape and nothing launches.
        let refused = KernelError::MoeShape {
            kernel: "moe_sort",
            dims: [s.m, s.top_k, s.hidden, s.max_tiles],
        };
        if s.layout().is_none() {
            return Err(refused);
        }
        let regions = kernels::SortRegions {
            chunk_hist: slot(s.chunk_hist),
            counts: slot(s.counts),
            offsets: slot(s.offsets),
            tile_expert: slot(s.tile_expert),
            pos: slot(s.pos),
        };
        // SAFETY: as `enqueue`'s, for `x` (`m*hidden`), `topk_idx` (`m*top_k`), `rows` (the
        // layout's `rows_bytes`) and the index regions, each at its own offset inside the span
        // `extents` gives from `chunk_hist` (`MoeTape::validate` checked every region's offset).
        unsafe {
            kernels::moe_sort(
                (slot(s.x).cast_const(), slot(s.topk_idx).cast_const()),
                regions,
                slot(s.rows),
                (s.m, s.experts, s.top_k, s.hidden, s.max_tiles),
                stream.0,
            )
        }
    }

    /// Expert 0's `[n, k]` of `proj` in the MoE bundle `bundle` of `weights`, and the rows of each
    /// expert's block (the expert stride in rows of `k`), or `None` on overflow.
    fn expert_slab(
        weights: &[DeviceBuffer],
        (bundle, weight): (WeightIx, MoeWeight),
        proj: ExpertProjection,
        (experts, hidden, inter): (u32, u32, u32),
    ) -> Option<(*const c_void, u32)> {
        let layout = weight.experts(experts, hidden, inter)?;
        let (part, stride) = proj.slab(layout, hidden, inter)?;
        let (_, k) = proj.n_k(hidden, inter);
        let row_bytes = u64::from(k).checked_mul(CUDA_ACT_BYTES)?;
        let rows = u32::try_from(stride.checked_div(row_bytes)?).ok()?;
        let part = usize::try_from(part).ok()?;
        let b = weights[bundle.0 as usize].ptr().wrapping_byte_add(part);
        Some((b.cast_const(), rows))
    }

    /// Enqueue the grouped expert GEMM (`kernels::expert_gemm_bf16`) over the sorted rows, B at
    /// [`ExpertProjection::slab`]'s part of the bundle.
    fn expert_gemm(
        g: &ExpertGemmBf16,
        slot: impl Fn(ArenaOffset) -> *mut c_void,
        weights: &[DeviceBuffer],
        stream: &Stream,
    ) -> Result<(), KernelError> {
        let (n, k) = g.n_k();
        let Some((b, expert_rows)) = Self::expert_slab(
            weights,
            (g.bundle, g.weight),
            g.proj,
            (g.experts, g.hidden, g.inter),
        ) else {
            return Err(KernelError::MoeShape {
                kernel: "expert_gemm_bf16",
                dims: [g.max_tiles, n, k, 0],
            });
        };
        // SAFETY: as `enqueue`'s, for `a` (`max_tiles*128*k`), `offsets` (`experts + 1`),
        // `tile_expert` (`max_tiles`), `c` (`max_tiles*128*n`) and the bundle (`moe_bundle`'s
        // bytes, which `new` checked the buffer holds; every expert's `[n, k]` lies inside its
        // part).
        unsafe {
            kernels::expert_gemm_bf16(
                (slot(g.a).cast_const(), b),
                (
                    slot(g.offsets).cast_const(),
                    slot(g.tile_expert).cast_const(),
                ),
                slot(g.c),
                (g.max_tiles, n, k, g.experts, expert_rows),
                stream.0,
            )
        }
    }

    /// Enqueue the gathered expert GEMV (`kernels::expert_gemv_bf16`) over every pair, B at
    /// [`ExpertProjection::slab`]'s part of the bundle.
    fn expert_gemv(
        g: &ExpertGemvBf16,
        slot: impl Fn(ArenaOffset) -> *mut c_void,
        weights: &[DeviceBuffer],
        stream: &Stream,
    ) -> Result<(), KernelError> {
        let (n, k) = g.n_k();
        let Some((b, expert_rows)) = Self::expert_slab(
            weights,
            (g.bundle, g.weight),
            g.proj,
            (g.experts, g.hidden, g.inter),
        ) else {
            return Err(KernelError::MoeShape {
                kernel: "expert_gemv_bf16",
                dims: [g.m, g.top_k, n, k],
            });
        };
        // SAFETY: as `enqueue`'s, for `a` (its `PairRows` rows of `k`), `topk_idx` (`m*top_k`),
        // `c` (`m*top_k*n`) and the bundle (`moe_bundle`'s bytes, which `new` checked the buffer
        // holds; every expert's `[n, k]` lies inside its part).
        unsafe {
            kernels::expert_gemv_bf16(
                (slot(g.a).cast_const(), b, slot(g.topk_idx).cast_const()),
                slot(g.c),
                (g.m, g.top_k, n, k),
                (g.experts, expert_rows),
                g.rows == PairRows::Pair,
                stream.0,
            )
        }
    }

    /// The device address of an arena offset. The arena is device memory Rust never reads, so the
    /// address is computed, not dereferenced.
    fn slot(&self, offset: ArenaOffset) -> *mut c_void {
        Self::slot_of(&self.arena, offset)
    }

    /// [`Self::slot`] of `arena`, for [`Self::enqueue`], which borrows the worker by its parts.
    fn slot_of(arena: &DeviceBuffer, offset: ArenaOffset) -> *mut c_void {
        let offset = usize::try_from(offset.0).expect("an arena offset fits the host's usize");
        arena.ptr().wrapping_byte_add(offset)
    }
}

/// What one forward uploads before its steps: the input, the rotary table's span and rows when the
/// tape has one, the KV regions when it has a KV cache, and the GDN sequence region and its 4 u32
/// when it has GDN layers.
struct Uploads<'h> {
    input: Input<'h>,
    /// The global rotary table's rows, then the local one's.
    tables: [Option<(ArenaSpan, &'h [u16])>; 2],
    kv: Option<KvUpload<'h>>,
    gdn: Option<(ArenaSpan, &'h [u32; 4])>,
}

/// `table`'s rows for `positions` filled into the front of `host`, with its span: what the forward
/// uploads for it. `None` for a tape without that table.
fn fill_table<'h>(
    table: Option<RotaryTable>,
    positions: &[u32],
    host: &'h mut [u16],
) -> Option<(ArenaSpan, &'h [u16])> {
    table.map(|r| {
        let rows = &mut host[..positions.len() * r.rot_dim as usize];
        fill_rotary_rows(positions, r.rot_dim, r.theta(), rows);
        (r.span, &*rows)
    })
}

/// One forward's KV regions, filled on the host: the tape's spans and the bytes for each.
struct KvUpload<'h> {
    tape: KvTape,
    block_table: &'h [u32],
    slot_mapping: &'h [u32],
    sequence: [u32; 2],
}

/// Check `seq` against `tape` and the pool, and fill the pool's host block table and slot
/// mapping from it. Returns `[prefix_len, q_len]`. Nothing is copied: every refusal comes before
/// the forward enqueues anything.
fn fill_sequence(
    tape: &CudaTape,
    kv: &KvTape,
    seq: &KvSequence<'_>,
    positions: &[u32],
    pool: &mut KvPool,
) -> Result<[u32; 2], ForwardError> {
    let KvSequence {
        block_table,
        prefix_len,
        q_len,
    } = *seq;
    let bucket_m = tape.bucket_m;
    if q_len == 0 || q_len > bucket_m {
        return Err(ForwardError::KvQueryLength { q_len, bucket_m });
    }
    let tokens = u64::from(prefix_len) + u64::from(q_len);
    let max = u64::from(kv.max_blocks_per_seq) * u64::from(kv.block_size);
    if tokens > max {
        return Err(ForwardError::KvTooLong { tokens, max });
    }
    let need = tokens.div_ceil(u64::from(kv.block_size)) as usize;
    let max_entries = kv.max_blocks_per_seq as usize;
    if block_table.len() < need || block_table.len() > max_entries {
        return Err(ForwardError::KvBlockTable {
            got: block_table.len(),
            need,
            max: max_entries,
        });
    }
    let used = &block_table[..need];
    // A slot is a u32 below `u32::MAX` (the padding sentinel), so a block past the last one whose
    // slots fit is outside the pool too. A pool that large cannot be allocated on an H100.
    let num_blocks = (pool.layout.num_blocks).min(NO_SLOT / kv.block_size);
    for (entry, &block) in used.iter().enumerate() {
        if block >= num_blocks {
            return Err(ForwardError::KvBlockOutsidePool {
                entry,
                block,
                num_blocks,
            });
        }
        if let Some(first) = used[..entry].iter().position(|&b| b == block) {
            return Err(ForwardError::KvBlockRepeated {
                block,
                first,
                second: entry,
            });
        }
    }
    for (row, &got) in positions.iter().enumerate().take(q_len as usize) {
        let want = u64::from(prefix_len) + row as u64;
        if u64::from(got) != want {
            return Err(ForwardError::PositionsDisagree { row, got, want });
        }
    }
    let table = &mut pool.block_table_host;
    table.fill(NO_SLOT);
    table[..need].copy_from_slice(used);
    let block_size = kv.block_size;
    let slots = &mut pool.slot_mapping_host[..bucket_m as usize];
    for (row, slot) in slots.iter_mut().enumerate() {
        *slot = if row < q_len as usize {
            // `pos < tokens`, so `pos / block_size < need`; the block is below `NO_SLOT /
            // block_size`, so the slot is below `NO_SLOT`.
            let pos = u64::from(prefix_len) + row as u64;
            let block = used[(pos / u64::from(block_size)) as usize];
            block * block_size + (pos % u64::from(block_size)) as u32
        } else {
            NO_SLOT
        };
    }
    Ok([prefix_len, q_len])
}

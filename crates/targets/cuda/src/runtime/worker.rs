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
use crate::rotary::fill_rotary_rows_with_freq_dim;
use crate::step::{
    ATTN_D512_BF16_HEAD_DIM, ATTN_DECODE_BF16_HEAD_DIM, ATTN_PREFILL_BF16_HEAD_DIM, AddBf16,
    AttnPrefillBf16, CudaStep, EmbedBf16, ExpertCombineBf16, ExpertGateUpActBf16, ExpertGemmBf16,
    ExpertGemvBf16, ExpertProjection, ExpertSortBf16, GateApplyBf16, GateScaleBf16, GateSplitBf16,
    GateUpKernel, GatedDeltaNetBf16, GemmBf16, GemvEnds, GroupMember, LayerScaleBf16,
    MOE_ROUTE_EXPERTS, MatmulKernel, MoeRouteBf16, MoeWeight, NormAddBf16, PairRows, QuantRowsFp8,
    RmsNormBf16, RmsNormUnitBf16, RopeAppendBf16, RopeBf16, RouteProgram, RouterLogitsBf16,
    ScaleBf16, SiluMulBf16, SoftCapBf16, fp8_moe_layout, fp8_weight, gemv_splits,
};
use crate::tape::{
    ArenaOffset, ArenaSpan, CUDA_ACT_BYTES, CUDA_ARENA_ALIGN, CudaTape, GdnTape, KV_ELEM_BYTES,
    KvTape, RotaryTable, TapeError, TapeInput, WeightIx,
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
    /// Creating the stream or allocating the arena or a pool failed, or recording a tape's
    /// graph did.
    Cuda(CudaError),
    /// Recording the tape at `bucket` into its graph, the launcher of its step `step` refused
    /// (`error`): the shape `validate` passed is one the kernel does not take. Nothing ran.
    Launch {
        bucket: usize,
        step: usize,
        error: KernelError,
    },
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
            Self::Launch {
                bucket,
                step,
                error,
            } => write!(
                f,
                "recording the tape of bucket {bucket} into its graph, step {step} did not \
                 launch: {error}"
            ),
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
    /// The `step`-th step is `op`, a step no kernel launches: a Qwen routing at another count than
    /// [`MOE_ROUTE_EXPERTS`], the Qwen program's one instance (#921). Checked over the whole tape
    /// before anything is uploaded or enqueued.
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
                write!(f, "step {step} is {op}, which no kernel launches")
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

/// One bucket's tape as a CUDA Graph: the play loop's launches, recorded once at construction
/// ([`CudaWorker::capture`]) and replayed by every forward with one `cudaGraphLaunch`. Each
/// recorded launch holds its argument struct by value, its TMA tensor maps included, so a replay
/// encodes nothing and calls no launcher.
struct Graph(ffi::CudaGraphExecT);

impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: the exec came from cudaGraphInstantiateWithFlags and is destroyed once; no
        // replay is in flight, since every forward ends in its own stream sync (`forward_as`),
        // not because the stream's drop (which runs after this one) syncs. A failure cannot be
        // reported from a destructor.
        let _ = unsafe { ffi::cudaGraphExecDestroy(self.0) };
    }
}

/// How a forward runs its steps: the bucket's graph, or each launch as the graph recorded it.
#[derive(Clone, Copy)]
enum Steps<'g> {
    Graph(&'g Graph),
    Launches,
}

/// The worker's paged KV pool: one allocation of every layer's K and V buffer, and the host side
/// of the forward's KV regions.
struct KvPool {
    buffer: DeviceBuffer,
    /// Class 0's K (or V) buffer's layout: the tapes' `kv_heads` and `head_dim`, and the pool's
    /// block size and count, which every class shares.
    layout: PagedKvLayout,
    /// Per KV layer, in [`KvTape::layers`] order: its class's layout and the element offset of
    /// its K buffer in `buffer`, its V buffer right after. Computed once by [`KvTape::pool_layout`]
    /// from the tapes' compile-time `classes` and `layer_class`.
    per_layer: Vec<(PagedKvLayout, usize)>,
    /// The KV layers, as the tapes' [`KvTape::layers`].
    layers: usize,
    /// The host side of the block table, sized once: `max_blocks_per_seq` entries.
    block_table_host: Vec<u32>,
    /// The host side of the slot mapping, sized once for the largest bucket.
    slot_mapping_host: Vec<u32>,
}

impl KvPool {
    /// Element offset of layer index `layer`'s `half` buffer in the pool. For a tape of one class
    /// (every Qwen tape) this is `(2 · layer + half) · buffer_elems`, as before the classes.
    fn half_elems(&self, layer: usize, half: KvHalf) -> usize {
        let (layout, at) = self.per_layer[layer];
        match half {
            KvHalf::K => at,
            KvHalf::V => at + layout.buffer_elems(),
        }
    }

    /// Layer index `layer`'s layout: its class's `kv_heads` and `head_dim`.
    fn layer_layout(&self, layer: usize) -> PagedKvLayout {
        self.per_layer[layer].0
    }

    /// The device address of layer index `layer`'s `half` buffer. Computed, not dereferenced.
    fn half_ptr(&self, layer: usize, half: KvHalf) -> *mut c_void {
        let bytes = self.half_elems(layer, half) * KV_ELEM_BYTES as usize;
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

/// The name of `s` when no kernel launches it: the Qwen routing at another count than
/// [`MOE_ROUTE_EXPERTS`]. The routing kernel's Qwen program is instantiated at 256 only (its
/// launcher refuses any other count), so such a tape is refused before anything is enqueued,
/// rather than at its routing step. `None` for every other step.
fn not_launched(s: &CudaStep) -> Option<&'static str> {
    if let CudaStep::MoeRoute(MoeRouteBf16 {
        program: RouteProgram::QwenSoftmax,
        experts,
        ..
    }) = *s
        && experts != MOE_ROUTE_EXPERTS
    {
        return Some("MoeRoute");
    }
    // shortcut: a fused norm or residual add (FP8 S8 W4a) is refused until its kernel lands in
    // the next commits, so no launch ever drops one; upgrade by launching it.
    if s.gemv_ends().is_some_and(|e| e != GemvEnds::NONE) {
        return Some("GemvEnds");
    }
    None
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
    /// Per tape, its CUDA Graph, recorded at construction; `None` for a tape a step of which
    /// `not_launched` names, whose forward refuses before any copy. Declared before `stream`, so
    /// the graphs drop first.
    graphs: Vec<Option<Graph>>,
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

/// A prefill attention launcher over the e4m3 cache: [`kernels::attn_prefill_windowed_fp8kv`]'s
/// signature, which the head-dim-512 one shares.
type PrefillLaunch = unsafe fn(
    (*const c_void, *mut c_void),
    (*const c_void, *const c_void),
    (*const c_void, *const c_void),
    (u32, u32, u32, u32),
    (u32, u32),
    (f32, u32),
    *mut c_void,
) -> Result<(), KernelError>;

/// A decode attention launcher over the e4m3 cache: [`kernels::attn_decode_windowed_fp8kv`]'s
/// signature, which the head-dim-512 one shares.
type DecodeLaunch = unsafe fn(
    (*const c_void, *mut c_void),
    (*const c_void, *const c_void),
    (*const c_void, *const c_void),
    (u32, u32, u32),
    (u32, u32),
    (f32, u32),
    *mut c_void,
) -> Result<(), KernelError>;

/// The one KV geometry of `tapes`: the first KV tape's, which every other KV tape must share.
fn kv_geometry(tapes: &[CudaTape]) -> Result<Option<KvTape>, WorkerError> {
    let mut first: Option<KvTape> = None;
    for (bucket, tape) in tapes.iter().enumerate() {
        let Some(kv) = tape.kv else { continue };
        match first {
            None => first = Some(kv),
            Some(f) => {
                if !f.same_pool(&kv) {
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
    ///
    /// The graph capture assumes worker build is single-threaded per process: its `ThreadLocal`
    /// capture does not see another thread's `cudaMalloc` or legacy-stream sync.
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
                let (per_layer, bytes) = kv
                    .pool_layout(num_blocks)
                    .ok_or(WorkerError::KvPoolTooLarge { num_blocks })?;
                Some((kv, layout, per_layer, bytes))
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
            Some((kv, layout, per_layer, bytes)) => {
                let buffer = DeviceBuffer::alloc(device, bytes).map_err(WorkerError::Cuda)?;
                let slots = (tapes.iter())
                    .filter(|t| t.kv.is_some())
                    .map(|t| t.bucket_m);
                Some(KvPool {
                    buffer,
                    layout,
                    per_layer,
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
        let graphs = (tapes.iter().enumerate())
            .map(|(bucket, tape)| {
                if tape.steps.iter().any(|s| not_launched(s).is_some()) {
                    return Ok(None);
                }
                Self::capture(
                    &stream,
                    &arena,
                    weights.as_ref(),
                    (kv.as_ref(), gdn.as_ref()),
                    (bucket, tape),
                )
                .map(Some)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            tapes,
            weights,
            borrow: PhantomData,
            graphs,
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
    /// rotary table, that table's rows for `positions`, one per token), run every step in
    /// order (the bucket's CUDA Graph, recorded once when the worker was built), download its result span into `result`, and wait once. Returns once the GPU is done,
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
        self.forward_as(None, (bucket, input, positions), (kv, gdn), result)
    }

    /// [`Self::forward`] with each step launched one by one, the launches the graph recorded,
    /// instead of replaying it: the same kernels with the same arguments in the same order, and the
    /// same checks and copies, for the tests that check the two give the same bits.
    pub fn forward_launched(
        &mut self,
        bucket: usize,
        input: Input<'_>,
        positions: &[u32],
        kv: Option<&KvSequence<'_>>,
        gdn: Option<&GdnSequence>,
        result: &mut [u16],
    ) -> Result<(), ForwardError> {
        self.forward_as(
            Some(Steps::Launches),
            (bucket, input, positions),
            (kv, gdn),
            result,
        )
    }

    /// [`Self::forward`] (`steps` `None`: the bucket's graph) or [`Self::forward_launched`].
    fn forward_as(
        &mut self,
        steps: Option<Steps<'_>>,
        (bucket, input, positions): (usize, Input<'_>, &[u32]),
        (kv, gdn): (Option<&KvSequence<'_>>, Option<&GdnSequence>),
        result: &mut [u16],
    ) -> Result<(), ForwardError> {
        let tape = self.tapes.get(bucket).ok_or(ForwardError::Bucket {
            bucket,
            buckets: self.tapes.len(),
        })?;
        if let Some((step, op)) = (tape.steps.iter().enumerate())
            .find_map(|(step, s)| not_launched(s).map(|op| (step, op)))
        {
            return Err(ForwardError::NotLaunched { step, op });
        }
        // A tape with no step `not_launched` names was captured at construction.
        let graph = self.graphs[bucket]
            .as_ref()
            .expect("`new` captured every tape that launches");
        let steps = steps.unwrap_or(Steps::Graph(graph));
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
            (tape, steps),
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
    /// head_dim` e4m3 bytes, one an element), synchronously: the cache's bytes, for a check after
    /// a forward.
    pub fn read_kv(
        &mut self,
        layer: usize,
        half: KvHalf,
        block: u32,
        out: &mut [u8],
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
        if layer >= pool.layers {
            return Err(refuse(pool.layout.kv_blk_stride()));
        }
        let layout = pool.layer_layout(layer);
        let block_elems = layout.kv_blk_stride();
        if block >= layout.num_blocks || out.len() != block_elems {
            return Err(refuse(block_elems));
        }
        let at = pool.half_elems(layer, half) + layout.elem_offset(block, 0, 0);
        let src = pool
            .buffer
            .ptr()
            .wrapping_byte_add(at * KV_ELEM_BYTES as usize);
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

    /// Record `tape`'s steps ([`Self::play`]) on `stream` into a CUDA Graph and instantiate it:
    /// one capture, at construction. Capture runs no kernel. It is `ThreadLocal`, so a call of this
    /// thread that is unsafe during a capture (a synchronous copy, an allocation, a first load of a
    /// kernel's module) ends it with an error, which is returned, never ignored: every kernel was
    /// loaded when the device opened (`kernels::prepare_all`), so none loads here. A launcher that
    /// refuses its step's shape ends the capture and is [`WorkerError::Launch`]; that error is the
    /// one returned, and `cudaStreamEndCapture`'s (the capture it invalidated) is dropped.
    fn capture(
        stream: &Stream,
        arena: &DeviceBuffer,
        weights: &[DeviceBuffer],
        pools: (Option<&KvPool>, Option<&GdnPool>),
        (bucket, tape): (usize, &CudaTape),
    ) -> Result<Graph, WorkerError> {
        // SAFETY: the stream is live and idle (nothing was enqueued on it yet).
        check(
            unsafe { ffi::cudaStreamBeginCapture(stream.0, ffi::STREAM_CAPTURE_MODE_THREAD_LOCAL) },
            "cudaStreamBeginCapture",
        )
        .map_err(WorkerError::Cuda)?;
        let played = Self::play(stream, arena, weights, pools, tape);
        let mut graph: ffi::CudaGraphT = std::ptr::null_mut();
        // SAFETY: the stream is capturing; the capture ends here, whether `play` succeeded or not,
        // so the stream is never left capturing.
        let ended = check(
            unsafe { ffi::cudaStreamEndCapture(stream.0, &mut graph) },
            "cudaStreamEndCapture",
        );
        let instantiated = played
            .map_err(|(step, error)| WorkerError::Launch {
                bucket,
                step,
                error,
            })
            .and(ended.map_err(WorkerError::Cuda))
            .and_then(|()| {
                let mut exec: ffi::CudaGraphExecT = std::ptr::null_mut();
                // SAFETY: `graph` is the capture's graph; `exec` is a valid out-pointer.
                check(
                    unsafe { ffi::cudaGraphInstantiateWithFlags(&mut exec, graph, 0) },
                    "cudaGraphInstantiateWithFlags",
                )
                .map(|()| Graph(exec))
                .map_err(WorkerError::Cuda)
            });
        if !graph.is_null() {
            // SAFETY: the instantiated exec copies the graph; the graph itself is not used again.
            let _ = unsafe { ffi::cudaGraphDestroy(graph) };
        }
        instantiated
    }

    /// Every step of `tape`, launched in tape order on `stream`, stopping at the first launcher
    /// that refuses: the work [`Self::capture`] records into the bucket's graph. It reads only
    /// fixed device addresses (the arena's spans, the weights, the pools' bases) and each step's
    /// constant shape, so the recording is the forward's for every input.
    fn play(
        stream: &Stream,
        arena: &DeviceBuffer,
        weights: &[DeviceBuffer],
        (kv, gdn): (Option<&KvPool>, Option<&GdnPool>),
        tape: &CudaTape,
    ) -> Result<(), (usize, KernelError)> {
        let slot = |offset| Self::slot_of(arena, offset);
        // The decode GEMVs' split-K scratch, null when no step splits (a launcher refuses a null
        // scratch above one split).
        let partials = tape
            .gemv_scratch
            .map_or(std::ptr::null_mut(), |s| slot(s.offset));
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
                CudaStep::GemvFp8(GemmBf16 { a, b, c, n, k, .. }) => {
                    let b = Self::fp8_weight(weights, b, n, k);
                    // SAFETY: as the `Gemv` arm's, for `a` (`k` bf16, which the kernel quantizes
                    // in its prologue), the fp8 weight (`n*k` bytes then `n` f32,
                    // `step::fp8_weight`) and `c` (`n`).
                    unsafe {
                        kernels::gemv_fp8_abt(
                            (slot(a).cast_const(), b, slot(c)),
                            partials,
                            (n, k, gemv_splits(n, k, 1)),
                            stream.0,
                        )
                    }
                }
                CudaStep::GemvGroupFp8(g) => {
                    let present: Vec<(usize, GroupMember)> = (g.members().iter().enumerate())
                        .filter_map(|(i, m)| m.map(|m| (i, m)))
                        .collect();
                    let members: Vec<(kernels::Fp8Operand, *mut c_void, u32)> = (present.iter())
                        .map(|(_, m)| (Self::fp8_weight(weights, m.b, m.n, g.k), slot(m.c), m.n))
                        .collect();
                    let splits: Vec<u32> = (present.iter())
                        .map(|(_, m)| gemv_splits(m.n, g.k, 1))
                        .collect();
                    // Member `i`'s partials follow the members' before it, as the bf16 group's.
                    let parts: Vec<*mut c_void> = (present.iter().zip(&splits))
                        .map(|(&(i, _), &sp)| {
                            (g.gemv_scratch_before(i))
                                .filter(|_| sp > 1 && !partials.is_null())
                                .map_or(std::ptr::null_mut(), |before| {
                                    partials.cast::<u8>().wrapping_add(before as usize).cast()
                                })
                        })
                        .collect();
                    // SAFETY: as the `GemvGroup` arm's, each member's weight an fp8 buffer.
                    unsafe {
                        kernels::gemv_group_fp8_abt(
                            slot(g.a),
                            &members,
                            &parts,
                            (g.k, &splits),
                            stream.0,
                        )
                    }
                }
                CudaStep::QuantRowsFp8(QuantRowsFp8 { x, q, rows, cols }) => {
                    let (q, scale) = Self::fp8_rows(slot, q, rows, cols);
                    // SAFETY: as above, for `x` (`rows*cols` bf16) and the quant scratch, which
                    // holds `rows*cols` bytes then `rows` f32 at `scale_at` (`validate`).
                    unsafe {
                        kernels::quant_rows_fp8(
                            slot(x),
                            (q.cast_mut(), scale.cast_mut()),
                            (rows, cols),
                            stream.0,
                        )
                    }
                }
                CudaStep::GemmFp8(GemmBf16 {
                    a, b, c, m, n, k, ..
                })
                | CudaStep::GemmCol64Fp8(GemmBf16 {
                    a, b, c, m, n, k, ..
                }) => {
                    let col64 = matches!(s, CudaStep::GemmCol64Fp8(_));
                    let (a, b) = (
                        Self::fp8_rows(slot, a, m, k),
                        Self::fp8_weight(weights, b, n, k),
                    );
                    // SAFETY: as above, for the quantized rows and their scales, the fp8 weight
                    // (`n*k` bytes then `n` f32, `step::fp8_weight`) and `c` (`m*n`).
                    unsafe { kernels::gemm_fp8_abt(a, b, slot(c), (m, n, k), col64, stream.0) }
                }
                CudaStep::GemmNarrowFp8(GemmBf16 {
                    a, b, c, m, n, k, ..
                }) => {
                    let (a, b) = (
                        Self::fp8_rows(slot, a, m, k),
                        Self::fp8_weight(weights, b, n, k),
                    );
                    // SAFETY: as the `GemmFp8` arm's.
                    unsafe { kernels::gemm_narrow_fp8_abt(a, b, slot(c), (m, n, k), stream.0) }
                }
                CudaStep::GemmGroupFp8(g) | CudaStep::GemmNarrowGroupFp8(g) => {
                    let members: Vec<(kernels::Fp8Operand, *mut c_void, u32)> =
                        (g.members().iter())
                            .flatten()
                            .map(|m| (Self::fp8_weight(weights, m.b, m.n, g.k), slot(m.c), m.n))
                            .collect();
                    let a = Self::fp8_rows(slot, g.a, g.m, g.k);
                    // SAFETY: as the `GemmFp8` arm's, for every member, whose outputs overlap
                    // neither the input nor each other (`validate`).
                    unsafe {
                        if matches!(s, CudaStep::GemmGroupFp8(_)) {
                            kernels::gemm_group_fp8_abt(a, &members, (g.m, g.k), stream.0)
                        } else {
                            kernels::gemm_narrow_group_fp8_abt(a, &members, (g.m, g.k), stream.0)
                        }
                    }
                }
                CudaStep::GemmCol64(GemmBf16 {
                    a, b, c, m, n, k, ..
                }) => {
                    let b = weights[b.0 as usize].ptr();
                    // SAFETY: as above, for `a` (`m*k`), `b` (`n*k`) and `c` (`m*n`).
                    unsafe {
                        kernels::gemm_col64_bf16_abt(slot(a), b, slot(c), (m, n, k), stream.0)
                    }
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
                CudaStep::Gemm(GemmBf16 {
                    a, b, c, m, n, k, ..
                }) => {
                    let b = weights[b.0 as usize].ptr();
                    // SAFETY: as above, for `a` (`m*k`), `b` (`n*k`) and `c` (`m*n`).
                    unsafe { kernels::gemm_bf16_abt(slot(a), b, slot(c), m, n, k, stream.0) }
                }
                CudaStep::GemmNarrow(GemmBf16 {
                    a, b, c, m, n, k, ..
                }) => {
                    let b = weights[b.0 as usize].ptr();
                    // SAFETY: as above, for `a` (`m*k`), `b` (`n*k`) and `c` (`m*n`).
                    unsafe {
                        kernels::gemm_narrow_bf16_abt(slot(a), b, slot(c), (m, n, k), stream.0)
                    }
                }
                CudaStep::Gemv(GemmBf16 { a, b, c, n, k, .. }) => {
                    let b = weights[b.0 as usize].ptr();
                    let splits = gemv_splits(n, k, 1);
                    // SAFETY: as above, for `a` (`k`, its `m` is 1: `validate`), `b` (`n*k`) and
                    // `c` (`n`); the scratch is the tape's GEMV scratch, at least this step's
                    // partials' bytes, which no other step touches (`validate`).
                    unsafe {
                        // One split is the unsplit launcher, the same kernel and bits, so its
                        // refusal stays `GemvShape` (#1048's `tape_player` pin).
                        if splits == 1 {
                            kernels::gemv_bf16_abt(slot(a), b, slot(c), (n, k), stream.0)
                        } else {
                            kernels::gemv_split_bf16_abt(
                                (slot(a), b, slot(c)),
                                partials,
                                (n, k, splits),
                                stream.0,
                            )
                        }
                    }
                }
                CudaStep::GemmGroup(g) => {
                    let members: Vec<(*const c_void, *mut c_void, u32)> = (g.members().iter())
                        .flatten()
                        .map(|m| (weights[m.b.0 as usize].ptr().cast_const(), slot(m.c), m.n))
                        .collect();
                    // SAFETY: as above, for `a` (`m*k`) and each member's `b` (`n*k`) and `c`
                    // (`m*n`), which overlap neither `a` nor each other (`validate`).
                    unsafe {
                        kernels::gemm_group_bf16_abt(slot(g.a), &members, (g.m, g.k), stream.0)
                    }
                }
                CudaStep::GemmNarrowGroup(g) => {
                    let members: Vec<(*const c_void, *mut c_void, u32)> = (g.members().iter())
                        .flatten()
                        .map(|m| (weights[m.b.0 as usize].ptr().cast_const(), slot(m.c), m.n))
                        .collect();
                    // SAFETY: as the `GemmGroup` arm's.
                    unsafe {
                        kernels::gemm_narrow_group_bf16_abt(
                            slot(g.a),
                            &members,
                            (g.m, g.k),
                            stream.0,
                        )
                    }
                }
                CudaStep::GemvGroup(g) => {
                    let present: Vec<(usize, GroupMember)> = (g.members().iter().enumerate())
                        .filter_map(|(i, m)| m.map(|m| (i, m)))
                        .collect();
                    let members: Vec<(*const c_void, *mut c_void, u32)> = (present.iter())
                        .map(|(_, m)| (weights[m.b.0 as usize].ptr().cast_const(), slot(m.c), m.n))
                        .collect();
                    let splits: Vec<u32> = (present.iter())
                        .map(|(_, m)| gemv_splits(m.n, g.k, 1))
                        .collect();
                    // Member `i`'s partials follow the members' before it
                    // (`GemmGroupBf16::gemv_scratch_before`), null at one split.
                    let parts: Vec<*mut c_void> = (present.iter().zip(&splits))
                        .map(|(&(i, _), &sp)| {
                            (g.gemv_scratch_before(i))
                                .filter(|_| sp > 1 && !partials.is_null())
                                .map_or(std::ptr::null_mut(), |before| {
                                    partials.cast::<u8>().wrapping_add(before as usize).cast()
                                })
                        })
                        .collect();
                    // SAFETY: as the `Gemv` arm, for `a` (`k`) and each member's `b`, `c` and
                    // partials: the tape's GEMV scratch holds every member's partials one after
                    // another (`CudaStep::gemv_scratch`, which `validate` checked), so no two
                    // members' overlap.
                    unsafe {
                        kernels::gemv_group_bf16_abt(
                            slot(g.a),
                            &members,
                            &parts,
                            (g.k, &splits),
                            stream.0,
                        )
                    }
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
                    // The router is the bundle's first tensor, at byte 0 (`moe_bundle`), or Gemma's
                    // own projection, whole (`MoeWeight::router_bytes`).
                    let b = weights[bundle.0 as usize].ptr();
                    // SAFETY: as above, for `x` (`m*hidden`), the router (`experts*hidden`, the
                    // weight's first bytes) and `logits` (`m*experts`).
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
                            MatmulKernel::Gemv => kernels::gemv_split_bf16_abt(
                                (slot(x), b, slot(logits)),
                                partials,
                                (experts, hidden, gemv_splits(experts, hidden, 1)),
                                stream.0,
                            ),
                            // Gemma's 128-expert router at m ≥ 2, its own projection.
                            MatmulKernel::Col64 => kernels::gemm_col64_bf16_abt(
                                slot(x),
                                b,
                                slot(logits),
                                (m, experts, hidden),
                                stream.0,
                            ),
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
                CudaStep::ExpertGemv(g) => Self::expert_gemv(&g, slot, partials, weights, stream),
                CudaStep::GateUpAct(g) => {
                    let (gate, up) = (
                        weights[g.gate.b.0 as usize].ptr(),
                        weights[g.up.b.0 as usize].ptr(),
                    );
                    let (GemmBf16 { a, m, n, k, .. }, act) = (g.gate, g.act.code());
                    // SAFETY: as above, for `a` (`m*k`), `gate` and `up` (`n*k` each) and `out`
                    // (`m*n`); at one row the scratch holds both projections' partials
                    // (`GateUpActBf16::gemv_scratch`, which `validate` checked).
                    unsafe {
                        match g.kernel {
                            GateUpKernel::Gemv => kernels::gate_up_gemv_bf16(
                                (slot(a), gate, up),
                                (slot(g.out), partials),
                                (n, k, gemv_splits(n, k, 1)),
                                act,
                                stream.0,
                            ),
                            GateUpKernel::GemvFp8 => kernels::gate_up_gemv_fp8(
                                (
                                    slot(a).cast_const(),
                                    Self::fp8_weight(weights, g.gate.b, n, k),
                                    Self::fp8_weight(weights, g.up.b, n, k),
                                ),
                                (slot(g.out), partials),
                                (n, k, gemv_splits(n, k, 1)),
                                act,
                                stream.0,
                            ),
                            GateUpKernel::GemmFp8 | GateUpKernel::GemmCol64Fp8 => {
                                kernels::gate_up_gemm_fp8(
                                    (
                                        Self::fp8_rows(slot, a, m, k),
                                        Self::fp8_weight(weights, g.gate.b, n, k),
                                        Self::fp8_weight(weights, g.up.b, n, k),
                                    ),
                                    slot(g.out),
                                    (m, n, k),
                                    (act, g.kernel == GateUpKernel::GemmCol64Fp8),
                                    stream.0,
                                )
                            }
                            GateUpKernel::Gemm | GateUpKernel::GemmCol64 => {
                                kernels::gate_up_gemm_bf16(
                                    (slot(a), gate, up),
                                    slot(g.out),
                                    (m, n, k),
                                    (act, g.kernel == GateUpKernel::GemmCol64),
                                    stream.0,
                                )
                            }
                        }
                    }
                }
                CudaStep::ExpertGateUpAct(g) => {
                    Self::expert_gate_up(&g, slot, partials, weights, stream)
                }
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
                CudaStep::GdnDecode(g) => {
                    let pool = gdn.expect("`new` made a pool for a GDN tape");
                    Self::gdn_decode(&g, pool, slot, weights, stream)
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
                CudaStep::AddRmsNorm(f) => {
                    let n = f.norm;
                    let gain = weights[n.gain.0 as usize].ptr();
                    // SAFETY: as above, for `a`, `b`, `sum` and `y` (`rows*cols` each) and `gain`
                    // (`cols`); `sum` and `y` overlap neither each other nor an input (`validate`).
                    unsafe {
                        kernels::add_rms_norm_bf16(
                            (slot(f.a), slot(f.b)),
                            gain,
                            (slot(f.sum), slot(n.y)),
                            (n.rows, n.cols),
                            (n.eps(), n.convention.offset()),
                            stream.0,
                        )
                    }
                }
                CudaStep::NormAdd(NormAddBf16 { norm: n, residual }) => {
                    let gain = weights[n.gain.0 as usize].ptr();
                    // SAFETY: as above, for `x`, `residual` and `y` (`rows*cols` each) and `gain`
                    // (`cols`).
                    unsafe {
                        kernels::norm_add_bf16(
                            (slot(n.x), slot(residual)),
                            gain,
                            slot(n.y),
                            (n.rows, n.cols),
                            (n.eps(), n.convention.offset()),
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
                    pairing,
                }) => {
                    // SAFETY: as above, for `x` and `out` (`rows*cols`) and `table`
                    // (`rows*rot_dim`), whose rows this forward uploaded above.
                    unsafe {
                        kernels::rope_paired_bf16(
                            slot(x),
                            slot(table),
                            slot(out),
                            (rows, cols),
                            (head_dim, rot_dim, pairing),
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
                    pairing,
                }) => {
                    let pool = kv.expect("`new` made a pool for a KV tape");
                    let l = layer as usize;
                    // Layer `l`'s class layout: its `kv_heads` and `head_dim` (`validate`
                    // checked the step's are its class's).
                    let layout = pool.layer_layout(l);
                    // SAFETY: as above, for `k`, `v` and `k_out` (`rows*cols`), `table`
                    // (`rows*rot_dim`) and `slot_mapping` (`rows` u32), uploaded above; the cache
                    // halves are layer `l`'s, `num_blocks · kv_heads · 64 · head_dim` of its
                    // class each, and every slot uploaded is in a checked block or `u32::MAX`.
                    unsafe {
                        kernels::rope_paired_bf16(
                            slot(k),
                            slot(table),
                            slot(k_out),
                            (rows, cols),
                            (head_dim, rot_dim, pairing),
                            stream.0,
                        )
                        .and_then(|()| {
                            kernels::kv_append_fp8(
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
                    let layout = pool.layer_layout(l);
                    // The column-split kernel at head dim 512, the head-dim-256 one otherwise
                    // (whose launcher refuses any other head dim).
                    let launch: PrefillLaunch = match head_dim {
                        ATTN_PREFILL_BF16_HEAD_DIM => kernels::attn_prefill_windowed_fp8kv,
                        ATTN_D512_BF16_HEAD_DIM => kernels::attn_prefill_d512_fp8kv,
                        // No prefill attention kernel at another head dim: refused, nothing
                        // launched (`validate` already refuses it: `ATTN_BF16_HEAD_DIMS`).
                        other => {
                            let error = KernelError::AttnShape {
                                rows,
                                q_heads,
                                kv_heads,
                                head_dim: other,
                                block_size: layout.block_size,
                                num_blocks: layout.num_blocks,
                            };
                            return Err((step, error));
                        }
                    };
                    // SAFETY: as above, for `q` and `out` (`rows*q_heads*head_dim`), and the
                    // block table and sequence uploaded above, which `forward` checked: every
                    // block the keys `0 .. P + L` need is a table entry inside the pool, and
                    // `1 ≤ L ≤ rows`. The cache halves are layer `l`'s, of its class's layout.
                    unsafe {
                        launch(
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
                    let layout = pool.layer_layout(l);
                    let launch: DecodeLaunch = match head_dim {
                        ATTN_DECODE_BF16_HEAD_DIM => kernels::attn_decode_windowed_fp8kv,
                        ATTN_D512_BF16_HEAD_DIM => kernels::attn_decode_d512_fp8kv,
                        // No decode attention kernel at another head dim: refused, nothing
                        // launched (`validate` already refuses it: `ATTN_BF16_HEAD_DIMS`).
                        other => {
                            let error = KernelError::AttnDecodeShape {
                                q_heads,
                                kv_heads,
                                head_dim: other,
                                block_size: layout.block_size,
                                num_blocks: layout.num_blocks,
                            };
                            return Err((step, error));
                        }
                    };
                    // SAFETY: as the prefill attention's, for `q` and `out` (`q_heads*head_dim`,
                    // its rows are 1: `validate`), and the block table and sequence uploaded
                    // above, which `forward` checked: every block the keys `0 .. P + L` need is a
                    // table entry inside the pool, and `L ≥ 1`. The cache halves are layer `l`'s,
                    // of its class's layout.
                    unsafe {
                        launch(
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
            launched.map_err(|error| (step, error))?;
        }
        Ok(())
    }

    /// Everything one forward enqueues, stopping at the first failure: the input copy, the rotary
    /// table's rows when the tape has one (`table`: its span and the rows, which hold exactly its
    /// bytes), the KV regions when it has a KV cache, the bucket's graph (every launch) and the
    /// result copy. The caller drains the stream whatever this returns.
    fn enqueue(
        stream: &Stream,
        arena: &DeviceBuffer,
        weights: &[DeviceBuffer],
        (kv, gdn): (Option<&KvPool>, Option<&GdnPool>),
        (tape, steps): (&CudaTape, Steps<'_>),
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
        // Every step's launch, in tape order: the bucket's graph, captured from [`Self::play`] at
        // construction, or (`Steps::Launches`, for the tests that compare the two) `play` itself.
        // The uploads above and the result copy below stay outside the graph.
        match steps {
            // SAFETY: the graph was instantiated from this tape's play on this worker's stream,
            // arena, weights and pools, which live as long as the worker; the uploads it reads are
            // enqueued before it on the same stream.
            Steps::Graph(graph) => check(
                unsafe { ffi::cudaGraphLaunch(graph.0, stream.0) },
                "cudaGraphLaunch",
            )
            .map_err(ForwardError::Cuda)?,
            Steps::Launches => Self::play(stream, arena, weights, (kv, gdn), tape)
                .map_err(|(step, error)| ForwardError::Launch { step, error })?,
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
    /// Enqueue the one-launch GDN decode (#1049, `kernels::gdn_decode`) of a one-row GDN step:
    /// [`Self::gated_delta_net`]'s operands, bundle parts and state buffers, no scratch.
    fn gdn_decode(
        g: &GatedDeltaNetBf16,
        pool: &GdnPool,
        slot: impl Fn(ArenaOffset) -> *mut c_void,
        weights: &[DeviceBuffer],
        stream: &Stream,
    ) -> Result<(), KernelError> {
        let Some(bundle) = g.bundle_layout() else {
            return Err(KernelError::GdnDecodeShape {
                nk: g.num_k_heads,
                nv: g.num_v_heads,
                head_v: g.head_v,
                kernel: g.conv_kernel,
            });
        };
        let w = weights[g.bundle.0 as usize].ptr();
        let wat = |off: u64| {
            let off = usize::try_from(off).expect("a bundle offset fits the host's usize");
            w.wrapping_byte_add(off).cast_const()
        };
        let (conv_state, ssm_state) = pool.layer_ptrs(g.layer as usize);
        let layout = pool.layout;
        let operands = kernels::GdnDecodeOperands {
            qkv: slot(g.qkv).cast_const(),
            a: slot(g.a).cast_const(),
            b: slot(g.b).cast_const(),
            z: slot(g.z).cast_const(),
            w: wat(bundle.conv1d),
            a_log: wat(bundle.a_log),
            dt_bias: wat(bundle.dt_bias),
            norm: wat(bundle.norm),
            out: slot(g.out),
        };
        // SAFETY: as `gated_delta_net`'s, for the step's arena operands (`extents`) and bundle,
        // and the layer's state buffers at the pool's own strides.
        unsafe {
            kernels::gdn_decode(
                &operands,
                (conv_state, layout.conv_elems_per_slot() as u64),
                (ssm_state, layout.ssm_elems_per_slot() as u64),
                slot(g.sequence).cast_const(),
                (
                    g.num_k_heads,
                    g.num_v_heads,
                    g.head_k,
                    g.head_v,
                    g.conv_kernel,
                ),
                (g.scale(), g.eps()),
                stream.0,
            )
        }
    }

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

    /// Expert 0's `[n, k]` of `proj` in the MoE bundle or expert bank `bundle` of `weights` (its
    /// layout `weight`'s: [`MoeWeight::experts`]), and the rows of each expert's block (the expert
    /// stride in rows of `k`), or `None` on overflow.
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

    /// Expert 0's e4m3 `[n, k]` of `proj` in the fp8 bank `bundle` and its scale row, and the
    /// rows of each expert's block (`fp8_moe_layout`: the fused gate and up `2·inter`, the down
    /// `hidden`), or `None` on overflow or a bf16 bank.
    fn fp8_expert_slab(
        weights: &[DeviceBuffer],
        (bundle, weight): (WeightIx, MoeWeight),
        proj: ExpertProjection,
        (experts, hidden, inter): (u32, u32, u32),
    ) -> Option<(kernels::Fp8Operand, u32)> {
        let router = match weight {
            MoeWeight::Fp8Bundle => true,
            MoeWeight::Fp8Own => false,
            MoeWeight::Bundle | MoeWeight::Own => return None,
        };
        let l = fp8_moe_layout(experts, hidden, inter, router)?;
        let (h, i) = (u64::from(hidden), u64::from(inter));
        let (part, scale, rows) = match proj {
            ExpertProjection::Gate => (l.experts.gate_up, l.gate_up_scale, 2 * inter),
            ExpertProjection::Up => (
                l.experts.gate_up + i * h,
                l.gate_up_scale + i * 4,
                2 * inter,
            ),
            ExpertProjection::Down => (l.experts.down, l.down_scale, hidden),
        };
        let w = weights[bundle.0 as usize].ptr();
        let at = |o: u64| Some(w.wrapping_byte_add(usize::try_from(o).ok()?).cast_const());
        Some(((at(part)?, at(scale)?), rows))
    }

    /// Enqueue the grouped expert GEMM (`kernels::expert_gemm_tiled_bf16`) over the sorted rows at
    /// the step's expert count and column tile, B at [`ExpertProjection::slab`]'s part of the
    /// bundle or bank.
    fn expert_gemm(
        g: &ExpertGemmBf16,
        slot: impl Fn(ArenaOffset) -> *mut c_void,
        weights: &[DeviceBuffer],
        stream: &Stream,
    ) -> Result<(), KernelError> {
        let (n, k) = g.n_k();
        if let Some((a, rows, k)) = CudaStep::ExpertGemm(*g).quantized_input() {
            let Some((b, expert_rows)) = Self::fp8_expert_slab(
                weights,
                (g.bundle, g.weight),
                g.proj,
                (g.experts, g.hidden, g.inter),
            ) else {
                return Err(KernelError::MoeShape {
                    kernel: "expert_gemm_fp8",
                    dims: [g.max_tiles, n, k, 0],
                });
            };
            // SAFETY: as the bf16 arm's below, with `a` the quant scratch's `rows*k` e4m3 bytes
            // and `rows` f32 (`validate`), and the bank `fp8_moe_layout`'s bytes, expert 0's
            // e4m3 block and scale row inside it.
            return unsafe {
                kernels::expert_gemm_fp8(
                    (Self::fp8_rows(&slot, a, rows, k), b),
                    (
                        slot(g.offsets).cast_const(),
                        slot(g.tile_expert).cast_const(),
                    ),
                    slot(g.c),
                    (g.max_tiles, n, k, g.experts, expert_rows),
                    g.n_tile,
                    stream.0,
                )
            };
        }
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
            kernels::expert_gemm_tiled_bf16(
                (slot(g.a).cast_const(), b),
                (
                    slot(g.offsets).cast_const(),
                    slot(g.tile_expert).cast_const(),
                ),
                slot(g.c),
                (g.max_tiles, n, k, g.experts, expert_rows),
                g.n_tile,
                stream.0,
            )
        }
    }

    /// Enqueue the gathered expert GEMV (`kernels::expert_gemv_split_bf16`) over every pair at the
    /// step's expert count and K split (`step::gemv_splits`), B at [`ExpertProjection::slab`]'s
    /// part of the bundle or bank, its partials in the tape's GEMV scratch `partials`.
    fn expert_gemv(
        g: &ExpertGemvBf16,
        slot: impl Fn(ArenaOffset) -> *mut c_void,
        partials: *mut c_void,
        weights: &[DeviceBuffer],
        stream: &Stream,
    ) -> Result<(), KernelError> {
        let (n, k) = g.n_k();
        let splits = gemv_splits(n, k, g.m.saturating_mul(g.top_k));
        if matches!(g.weight, MoeWeight::Fp8Bundle | MoeWeight::Fp8Own) {
            let Some((b, expert_rows)) = Self::fp8_expert_slab(
                weights,
                (g.bundle, g.weight),
                g.proj,
                (g.experts, g.hidden, g.inter),
            ) else {
                return Err(KernelError::MoeShape {
                    kernel: "expert_gemv_fp8",
                    dims: [g.m, g.top_k, n, k],
                });
            };
            // SAFETY: as the bf16 launch's below, with the bank `fp8_moe_layout`'s bytes, expert
            // 0's e4m3 block and scale row inside it; `a` is bf16, quantized in the prologue.
            return unsafe {
                kernels::expert_gemv_fp8(
                    (slot(g.a).cast_const(), b, slot(g.topk_idx).cast_const()),
                    (slot(g.c), partials),
                    (g.m, g.top_k, n, k),
                    (g.experts, expert_rows),
                    (g.rows == PairRows::Pair, splits),
                    stream.0,
                )
            };
        }
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
        // `c` (`m*top_k*n`), the bundle (`moe_bundle`'s bytes, which `new` checked the buffer
        // holds; every expert's `[n, k]` lies inside its part) and the scratch (the tape's GEMV
        // scratch, at least this step's partials' bytes, which `validate` checked).
        unsafe {
            kernels::expert_gemv_split_bf16(
                (slot(g.a).cast_const(), b, slot(g.topk_idx).cast_const()),
                (slot(g.c), partials),
                (g.m, g.top_k, n, k),
                (g.experts, expert_rows),
                (g.rows == PairRows::Pair, splits),
                stream.0,
            )
        }
    }

    /// Enqueue the expert gate/up GEMV (#1049, `kernels::expert_gate_up_gemv_bf16`): the gate's
    /// and the up's [`Self::expert_slab`]s of one bundle, at the split each would take on its own.
    fn expert_gate_up(
        g: &ExpertGateUpActBf16,
        slot: impl Fn(ArenaOffset) -> *mut c_void,
        partials: *mut c_void,
        weights: &[DeviceBuffer],
        stream: &Stream,
    ) -> Result<(), KernelError> {
        let (n, k) = g.gate.n_k();
        let pairs = g.gate.m.saturating_mul(g.gate.top_k);
        let refused = KernelError::GateUpShape {
            kernel: "expert_gate_up_gemv_bf16",
            dims: [pairs, n, k, 0],
        };
        let splits = gemv_splits(n, k, pairs);
        if matches!(g.gate.weight, MoeWeight::Fp8Bundle | MoeWeight::Fp8Own) {
            let slab = |e: &ExpertGemvBf16| {
                Self::fp8_expert_slab(
                    weights,
                    (e.bundle, e.weight),
                    e.proj,
                    (e.experts, e.hidden, e.inter),
                )
            };
            let (Some((gate, rows)), Some((up, up_rows))) = (slab(&g.gate), slab(&g.up)) else {
                return Err(refused);
            };
            if rows != up_rows {
                return Err(refused);
            }
            // SAFETY: as the bf16 launch's below, for both fp8 slabs and their scale rows, each
            // inside the bank's gate/up part (`fp8_moe_layout`).
            return unsafe {
                kernels::expert_gate_up_gemv_fp8(
                    (
                        slot(g.gate.a).cast_const(),
                        gate,
                        up,
                        slot(g.gate.topk_idx).cast_const(),
                    ),
                    (slot(g.out), partials),
                    (g.gate.m, g.gate.top_k, n, k),
                    (g.gate.experts, rows, splits),
                    g.act.code(),
                    stream.0,
                )
            };
        }
        let slab = |e: &ExpertGemvBf16| {
            Self::expert_slab(
                weights,
                (e.bundle, e.weight),
                e.proj,
                (e.experts, e.hidden, e.inter),
            )
        };
        let (Some((gate, rows)), Some((up, up_rows))) = (slab(&g.gate), slab(&g.up)) else {
            return Err(refused);
        };
        if rows != up_rows {
            return Err(refused);
        }
        // SAFETY: as `expert_gemv`'s, for both slabs (each inside the bundle's gate/up part) and
        // the scratch (both projections' partials, `ExpertGateUpActBf16::gemv_scratch`, which
        // `validate` checked).
        unsafe {
            kernels::expert_gate_up_gemv_bf16(
                (
                    slot(g.gate.a).cast_const(),
                    gate,
                    up,
                    slot(g.gate.topk_idx).cast_const(),
                ),
                (slot(g.out), partials),
                (g.gate.m, g.gate.top_k, n, k),
                (g.gate.experts, rows, splits),
                g.act.code(),
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
    /// The quantized rows at `q` in the quant scratch, `[rows, cols]` e4m3, and their `[rows]` f32
    /// scales after them (`QuantRowsFp8::scale_at`).
    fn fp8_rows(
        slot: impl Fn(ArenaOffset) -> *mut c_void,
        q: ArenaOffset,
        rows: u32,
        cols: u32,
    ) -> kernels::Fp8Operand {
        let at = QuantRowsFp8 {
            x: q,
            q,
            rows,
            cols,
        }
        .scale_at()
        .expect("`validate` checked the quant scratch's extent");
        (
            slot(q).cast_const(),
            slot(ArenaOffset(q.0 + at)).cast_const(),
        )
    }

    /// Weight `b`'s fp8 buffer as an operand: its `[n, k]` e4m3 bytes, then its `[n]` f32 scales
    /// (`step::fp8_weight`).
    fn fp8_weight(weights: &[DeviceBuffer], b: WeightIx, n: u32, k: u32) -> kernels::Fp8Operand {
        let w = weights[b.0 as usize].ptr();
        let at = fp8_weight(n, k).expect("`validate` checked the weight's extent");
        (
            w.cast_const(),
            w.wrapping_byte_add(at.scale as usize).cast_const(),
        )
    }

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
        fill_rotary_rows_with_freq_dim(positions, r.rot_dim, r.freq_dim, r.theta(), rows);
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

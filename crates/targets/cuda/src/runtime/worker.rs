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

use std::ffi::c_void;
use std::fmt;

use super::buffer::DeviceBuffer;
use super::device::CudaDevice;
use super::ffi;
use super::{CudaError, check};
use scratchy_subtile::paged_kv_layout::PagedKvLayout;

use crate::kernels::{self, KernelError};
use crate::rotary::fill_rotary_rows;
use crate::step::{
    AddBf16, AttnPrefillBf16, CudaStep, GateApplyBf16, GateSplitBf16, GemmBf16, RmsNormBf16,
    RopeAppendBf16, RopeBf16, SiluMulBf16,
};
use crate::tape::{ArenaOffset, ArenaSpan, CudaTape, KvTape, TapeError};

/// The KV pool a worker allocates when its tapes attend: how many blocks each layer's K and V
/// buffers hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPoolConfig {
    pub num_blocks: u32,
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
    /// Creating the stream or allocating the arena or the pool failed.
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
    /// An arena read's span reaches past the arena; nothing was enqueued.
    SpanOutsideArena { span: ArenaSpan, arena_bytes: u64 },
    /// The `step`-th launch failed; nothing after it was enqueued, and the stream was drained.
    Launch { step: usize, error: KernelError },
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
            Self::SpanOutsideArena { span, arena_bytes } => write!(
                f,
                "the span of {} bytes at {} reaches past the {arena_bytes}-byte arena",
                span.bytes, span.offset.0
            ),
            Self::Launch { step, error } => write!(f, "step {step} did not launch: {error}"),
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

/// Plays a set of tapes (one per bucket) over one arena on one stream.
///
/// Neither `Send` nor `Sync`: it holds raw CUDA handles. Sharing is the worker pool's decision.
pub struct CudaWorker<'a> {
    tapes: &'static [CudaTape],
    /// Indexed by each step's `WeightIx`.
    weights: &'a [DeviceBuffer],
    stream: Stream,
    arena: DeviceBuffer,
    /// The host side of the rotary table, sized once for the largest tape's: each forward with a
    /// table fills its rows from the positions and uploads them, so a forward allocates nothing.
    rotary_host: Vec<u16>,
    /// The KV pool, when a tape has a KV cache.
    kv: Option<KvPool>,
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
    /// A worker for `tapes` reading `weights`: validates every tape ([`CudaTape::validate`]) and
    /// checks once that every weight a tape declares is large enough, before it allocates
    /// anything; then makes the stream and the one arena allocation, of the largest tape's
    /// `total_bytes`.
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
    pub fn new(
        device: &CudaDevice,
        tapes: &'static [CudaTape],
        weights: &'a [DeviceBuffer],
        kv_pool: Option<KvPoolConfig>,
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
                let have = weights
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
            (Some(kv), Some(KvPoolConfig { num_blocks })) if num_blocks >= 1 => Some((
                kv,
                PagedKvLayout {
                    num_blocks,
                    num_kv_heads: kv.kv_heads,
                    block_size: kv.block_size,
                    head_dim: kv.head_dim,
                },
            )),
            (Some(_), _) => return Err(WorkerError::KvPoolMissing),
            (None, Some(_)) => return Err(WorkerError::KvPoolUnused),
            (None, None) => None,
        };
        let arena_bytes =
            (tapes.iter().map(|t| t.arena.total_bytes).max()).ok_or(WorkerError::NoTapes)?;
        // Each table's span lies inside its arena (`validate`), so its byte count fits a usize.
        let rotary_elements = (tapes.iter())
            .filter_map(|t| t.rotary)
            .map(|r| r.span.bytes as usize / std::mem::size_of::<u16>())
            .max()
            .unwrap_or(0);
        let stream = Stream::new().map_err(WorkerError::Cuda)?;
        let arena = DeviceBuffer::alloc(device, arena_bytes).map_err(WorkerError::Cuda)?;
        let kv = match pool {
            Some((kv, layout)) => {
                let bytes = (kv.layers.len() as u64)
                    .checked_mul(2 * layout.buffer_elems() as u64)
                    .and_then(|e| e.checked_mul(std::mem::size_of::<u16>() as u64))
                    .expect("the KV pool's bytes fit u64");
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
        Ok(Self {
            tapes,
            weights,
            stream,
            arena,
            rotary_host: vec![0; rotary_elements],
            kv,
        })
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
    /// `positions` holds one position per token when the tape has a rotary table, and is empty
    /// when it has none; anything else is [`ForwardError::Positions`], before any copy.
    ///
    /// `kv` is the forward's sequence, given exactly when the tape has a KV cache. Before any copy
    /// it is checked to be one the kernels can run safely: `1 ≤ q_len ≤ bucket_m`, the tokens fit
    /// a block table, the table has the entries the tokens need and no more than the tape holds,
    /// every used entry is a distinct block inside the pool, and the real rows' positions are
    /// `prefix_len, prefix_len + 1, …`. Then the used entries (the rest `u32::MAX`), each row's
    /// slot (`u32::MAX` for a padding row) and `[prefix_len, q_len]` are uploaded with the input.
    pub fn forward(
        &mut self,
        bucket: usize,
        input: &[u16],
        positions: &[u32],
        kv: Option<&KvSequence<'_>>,
        result: &mut [u16],
    ) -> Result<(), ForwardError> {
        let tape = self.tapes.get(bucket).ok_or(ForwardError::Bucket {
            bucket,
            buckets: self.tapes.len(),
        })?;
        let got = std::mem::size_of_val(input) as u64;
        if got != tape.input.bytes {
            return Err(ForwardError::InputSize {
                expected: tape.input.bytes,
                got,
            });
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
        let table = match tape.rotary {
            Some(r) => {
                let rows = &mut self.rotary_host[..positions.len() * r.rot_dim as usize];
                fill_rotary_rows(positions, r.rot_dim, r.theta(), rows);
                Some((r.span, &*rows))
            }
            None => None,
        };
        let kv_upload = match (&sequence, &self.kv) {
            (Some((kv_tape, data)), Some(pool)) => Some(KvUpload {
                tape: *kv_tape,
                block_table: &pool.block_table_host,
                slot_mapping: &pool.slot_mapping_host[..tape.bucket_m as usize],
                sequence: *data,
            }),
            _ => None,
        };
        let enqueued = Self::enqueue(
            &self.stream,
            &self.arena,
            self.weights,
            self.kv.as_ref(),
            tape,
            Uploads {
                input,
                table,
                kv: kv_upload,
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
        kv: Option<&KvPool>,
        tape: &CudaTape,
        uploads: Uploads<'_>,
        result: &mut [u16],
    ) -> Result<(), ForwardError> {
        let Uploads {
            input,
            table,
            kv: kv_upload,
        } = uploads;
        let slot = |offset| Self::slot_of(arena, offset);
        // SAFETY: the input span lies inside the tape's arena (`new` validated it), which is at
        // most `arena_bytes`; `input` holds exactly its bytes, and `input` stays borrowed until the
        // caller's stream wait has returned.
        check(
            unsafe {
                ffi::cudaMemcpyAsync(
                    slot(tape.input.offset),
                    input.as_ptr().cast(),
                    std::mem::size_of_val(input),
                    ffi::MEMCPY_HOST_TO_DEVICE,
                    stream.0,
                )
            },
            "cudaMemcpyAsync(host to device)",
        )
        .map_err(ForwardError::Cuda)?;
        if let Some((span, rows)) = table {
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
                CudaStep::Gemm(GemmBf16 { a, b, c, m, n, k }) => {
                    let b = weights[b.0 as usize].ptr();
                    // SAFETY: as above, for `a` (`m*k`), `b` (`n*k`) and `c` (`m*n`).
                    unsafe { kernels::gemm_bf16_abt(slot(a), b, slot(c), m, n, k, stream.0) }
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
                        kernels::attn_prefill_bf16(
                            (slot(q), slot(out)),
                            (pool.half_ptr(l, KvHalf::K), pool.half_ptr(l, KvHalf::V)),
                            (slot(block_table), slot(sequence)),
                            (rows, q_heads, kv_heads, head_dim),
                            (layout.block_size, layout.num_blocks),
                            attn.scale(),
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
/// tape has one, and the KV regions when it has a KV cache.
struct Uploads<'h> {
    input: &'h [u16],
    table: Option<(ArenaSpan, &'h [u16])>,
    kv: Option<KvUpload<'h>>,
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

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

use std::ffi::c_void;
use std::fmt;

use super::buffer::DeviceBuffer;
use super::device::CudaDevice;
use super::ffi;
use super::{CudaError, check};
use crate::kernels::{self, KernelError};
use crate::step::{AddBf16, CudaStep, GemmBf16, RmsNormBf16, SiluMulBf16};
use crate::tape::{ArenaOffset, ArenaSpan, CudaTape, TapeError};

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
    /// Creating the stream or allocating the arena failed.
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
    /// The host buffer of an arena read is not the span's size, in bytes; nothing was enqueued.
    ReadSize { expected: u64, got: u64 },
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
            Self::ReadSize { expected, got } => {
                write!(f, "the read buffer is {got} bytes; the span is {expected}")
            }
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

/// Plays a set of tapes (one per bucket) over one arena on one stream.
///
/// Neither `Send` nor `Sync`: it holds raw CUDA handles. Sharing is the worker pool's decision.
pub struct CudaWorker<'a> {
    tapes: &'static [CudaTape],
    /// Indexed by each step's `WeightIx`.
    weights: &'a [DeviceBuffer],
    stream: Stream,
    arena: DeviceBuffer,
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
    pub fn new(
        device: &CudaDevice,
        tapes: &'static [CudaTape],
        weights: &'a [DeviceBuffer],
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
        let arena_bytes =
            (tapes.iter().map(|t| t.arena.total_bytes).max()).ok_or(WorkerError::NoTapes)?;
        let stream = Stream::new().map_err(WorkerError::Cuda)?;
        let arena = DeviceBuffer::alloc(device, arena_bytes).map_err(WorkerError::Cuda)?;
        Ok(Self {
            tapes,
            weights,
            stream,
            arena,
        })
    }

    /// The arena's bytes: its one allocation.
    pub fn arena_bytes(&self) -> u64 {
        self.arena.bytes()
    }

    /// Run bucket `bucket`'s tape: upload `input` into its input span, launch every step in
    /// order, download its result span into `result`, and wait once. Returns once the GPU is done,
    /// error or not, so nothing the forward touched is still in use.
    pub fn forward(
        &mut self,
        bucket: usize,
        input: &[u16],
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
        let enqueued = self.enqueue(tape, input, result);
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

    /// Everything one forward enqueues, stopping at the first failure. The caller drains the
    /// stream whatever this returns.
    fn enqueue(
        &self,
        tape: &CudaTape,
        input: &[u16],
        result: &mut [u16],
    ) -> Result<(), ForwardError> {
        // SAFETY: the input span lies inside the tape's arena (`new` validated it), which is at
        // most `arena_bytes`; `input` holds exactly its bytes, and `input` stays borrowed until the
        // caller's stream wait has returned.
        check(
            unsafe {
                ffi::cudaMemcpyAsync(
                    self.slot(tape.input.offset),
                    input.as_ptr().cast(),
                    std::mem::size_of_val(input),
                    ffi::MEMCPY_HOST_TO_DEVICE,
                    self.stream.0,
                )
            },
            "cudaMemcpyAsync(host to device)",
        )
        .map_err(ForwardError::Cuda)?;
        // SAFETY (every arm): `new` validated the tape (`CudaTape::validate`): each span a step
        // reads or writes (`CudaStep::extents`, the kernel's own operand sizes) is 256-byte
        // aligned and inside the tape's arena, which is at most this worker's; the span it writes
        // overlaps none it reads; each weight it reads has an index the tape declares, declared at
        // least as large as the step reads, and `new` checked every declared size against its
        // buffer. cudaMalloc pointers are 256-byte aligned. The stream is live.
        for (step, s) in tape.steps.iter().enumerate() {
            let launched = match *s {
                CudaStep::Gemm(GemmBf16 { a, b, c, m, n, k }) => {
                    let b = self.weights[b.0 as usize].ptr();
                    // SAFETY: as above, for `a` (`m*k`), `b` (`n*k`) and `c` (`m*n`).
                    unsafe {
                        kernels::gemm_bf16_abt(
                            self.slot(a),
                            b,
                            self.slot(c),
                            m,
                            n,
                            k,
                            self.stream.0,
                        )
                    }
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
                    let gain = self.weights[gain.0 as usize].ptr();
                    // SAFETY: as above, for `x` and `y` (`rows*cols`) and `gain` (`cols`).
                    unsafe {
                        kernels::rms_norm_bf16(
                            self.slot(x),
                            gain,
                            self.slot(y),
                            (rows, cols),
                            norm.eps(),
                            convention.offset(),
                            self.stream.0,
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
                        kernels::add_bf16(
                            self.slot(a),
                            self.slot(b),
                            self.slot(out),
                            (rows, cols),
                            self.stream.0,
                        )
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
                            self.slot(gate),
                            self.slot(up),
                            self.slot(out),
                            (rows, cols),
                            self.stream.0,
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
                    self.slot(tape.result.offset),
                    std::mem::size_of_val(result),
                    ffi::MEMCPY_DEVICE_TO_HOST,
                    self.stream.0,
                )
            },
            "cudaMemcpyAsync(device to host)",
        )
        .map_err(ForwardError::Cuda)
    }

    /// The device address of an arena offset. The arena is device memory Rust never reads, so the
    /// address is computed, not dereferenced.
    fn slot(&self, offset: ArenaOffset) -> *mut c_void {
        let offset = usize::try_from(offset.0).expect("an arena offset fits the host's usize");
        self.arena.ptr().wrapping_byte_add(offset)
    }
}

// SPDX-License-Identifier: Apache-2.0
//! ONE BUCKET'S TAPE, AS STATIC DATA — the counterpart of metal's `LoweredMetalTape`.
//!
//! A tape is what `scratchy-target-cuda-compiler` lowers one canonical (one bucket) to: the
//! arena's layout, where the forward's input and result live in it, how large each weight must
//! be, and the steps in launch order with every operand already resolved to a byte offset or a
//! weight index. Every field is a constant built at expansion time; the runtime reads it and
//! analyses nothing. Plain `Copy` data with no pointer, so it builds in a `const` context and on a
//! host with no CUDA toolchain.

use crate::step::{AttnPrefillBf16, CUDA_KV_BLOCK_SIZE, CudaStep, RopeAppendBf16, RopeBf16};

/// Bytes per activation element: CUDA's activations are bf16.
pub const CUDA_ACT_BYTES: u64 = 2;

/// Every colour's arena offset is a multiple of this. TMA needs 16-byte-aligned global addresses
/// and `cudaMalloc` returns 256-byte-aligned memory, so every slot pointer meets the launchers'
/// alignment contract with margin.
pub const CUDA_ARENA_ALIGN: u64 = 256;

/// A byte offset into the worker's arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaOffset(pub u64);

/// A weight, by its source's index in the canonical: the index into the worker's weight buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightIx(pub u32);

/// One colour's bytes in the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaSpan {
    pub offset: ArenaOffset,
    pub bytes: u64,
}

/// Where every colour lives in one tape's arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaLayout {
    /// Per colour: its value's `rows × cols × CUDA_ACT_BYTES`.
    pub colour_bytes: &'static [u64],
    /// Per colour: the sum of the [`CUDA_ARENA_ALIGN`]-aligned sizes of the colours before it.
    pub colour_offsets: &'static [u64],
    /// The sum of every colour's aligned size: the bytes this tape addresses.
    pub total_bytes: u64,
}

/// The forward's rotary table: one row per token, `[bucket_m, rot_dim]` bf16, row `t` laid out
/// `[cos θ_t,0..R/2 | sin θ_t,0..R/2]` with `θ_t,i = p_t · theta^(−2i/R)`.
///
/// Its place, size and constants are fixed at expansion; its contents are the forward's runtime
/// data, built on the host from the token positions (`crate::rotary::rotary_rows`) and uploaded
/// with the input, as token ids are. It is not a colour, and not a weight: every `Cos`/`Sin`
/// source the canonical binds reads this one table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RotaryTable {
    /// Where the table lives in the arena, after every colour.
    pub span: ArenaSpan,
    /// `R`, the rotated width of each head.
    pub rot_dim: u32,
    /// `rope_theta` as its bits (`f64::to_bits`), so the table is `Eq`; [`Self::theta`] reads it.
    pub theta_bits: u64,
}

impl RotaryTable {
    /// The table's `rope_theta`.
    pub const fn theta(&self) -> f64 {
        f64::from_bits(self.theta_bits)
    }
}

/// The tape's paged KV cache: which layers' K and V it writes or reads, their geometry, and the
/// three arena regions the forward fills with the sequence's data.
///
/// Every field is fixed at expansion. The pool's block count is not here: it is the worker's
/// (`runtime::KvPoolConfig`), sized once at construction. What the forward writes into the
/// regions (the block table, the slot of each token row, the prefix and query lengths) is its
/// runtime data, uploaded as the rotary table's rows are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvTape {
    /// Model layer ids whose K/V this tape reads or writes, in pool order (ascending); a step's
    /// `layer` is an index into this list.
    pub layers: &'static [u32],
    pub kv_heads: u32,
    pub head_dim: u32,
    /// Token slots per block: [`CUDA_KV_BLOCK_SIZE`].
    pub block_size: u32,
    /// Block-table entries per sequence: `crate::step::CUDA_MAX_BLOCKS_PER_SEQ`.
    pub max_blocks_per_seq: u32,
    /// `max_blocks_per_seq` u32, the unused entries `u32::MAX`.
    pub block_table: ArenaSpan,
    /// `bucket_m` u32, a padding row's `u32::MAX`.
    pub slot_mapping: ArenaSpan,
    /// `[prefix_len, q_len]` u32: 8 bytes at an aligned offset of its own.
    pub sequence: ArenaSpan,
}

/// Bytes of one u32 entry of a KV region.
const KV_ENTRY_BYTES: u64 = 4;

/// One bucket's forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CudaTape {
    /// The token count `M` the tape is compiled for.
    pub bucket_m: u32,
    pub arena: ArenaLayout,
    /// Colour 0: the embedded hidden, `[bucket_m, hidden]` bf16, which the forward uploads.
    pub input: ArenaSpan,
    /// The colourer's result colour, which the forward downloads.
    pub result: ArenaSpan,
    /// Per source index: the bytes its weight buffer must hold; 0 for a source that is not a
    /// dense weight.
    pub weight_bytes: &'static [u64],
    /// One step per launch, in launch order.
    pub steps: &'static [CudaStep],
    /// The rotary table the forward fills, when a step reads one; `None` for a tape with no rope.
    pub rotary: Option<RotaryTable>,
    /// The paged KV cache and its regions, when the canonical binds a `PrefixK` source; `None`
    /// for a tape with no attention.
    pub kv: Option<KvTape>,
}

/// Which of the tape's reserved regions a [`TapeError`] names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Region {
    Rotary,
    BlockTable,
    SlotMapping,
    Sequence,
}

impl std::fmt::Display for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Rotary => "the rotary table",
            Self::BlockTable => "the block table",
            Self::SlotMapping => "the slot mapping",
            Self::Sequence => "the sequence region",
        })
    }
}

/// Why a tape cannot be played: it would make a kernel read or write outside the arena, or read
/// past a weight. No tape `scratchy-target-cuda-compiler` lowers is; [`CudaTape::validate`] makes
/// that a checked fact rather than a promise, since every field of a tape is public.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TapeError {
    /// The layout has a different number of colour sizes and colour offsets.
    ColourCount { bytes: usize, offsets: usize },
    /// A colour's offset is not a multiple of [`CUDA_ARENA_ALIGN`].
    ColourMisaligned { colour: usize, offset: u64 },
    /// A colour reaches past the arena's `total` bytes, or its end overflows `u64`.
    ColourOutside {
        colour: usize,
        span: ArenaSpan,
        total: u64,
    },
    /// The input span reaches past the arena.
    InputOutside { span: ArenaSpan, total: u64 },
    /// The result span reaches past the arena.
    ResultOutside { span: ArenaSpan, total: u64 },
    /// A byte count of the `step`-th step overflows `u64`.
    StepOverflow { step: usize },
    /// A span the `step`-th step reads or writes is not at a multiple of [`CUDA_ARENA_ALIGN`].
    StepMisaligned { step: usize, offset: u64 },
    /// A span the `step`-th step reads or writes reaches past the arena.
    StepOutside {
        step: usize,
        span: ArenaSpan,
        total: u64,
    },
    /// A span the `step`-th step writes overlaps one it reads.
    WriteOverlapsRead {
        step: usize,
        write: ArenaSpan,
        read: ArenaSpan,
    },
    /// The two spans the `step`-th step writes overlap each other.
    WritesOverlap { step: usize },
    /// The rotary table is not at a multiple of [`CUDA_ARENA_ALIGN`], or reaches past the arena.
    RotaryOutside { span: ArenaSpan, total: u64 },
    /// The rotary table's span is not `bucket_m` rows of `rot_dim` bf16, or `rot_dim` is not a
    /// non-zero even number: the forward would fill a different number of bytes than it holds.
    RotaryShape {
        span: ArenaSpan,
        bucket_m: u32,
        rot_dim: u32,
    },
    /// The `step`-th step reads weight `ix`, and the tape declares `weights` weights.
    WeightIndex {
        step: usize,
        ix: u32,
        weights: usize,
    },
    /// The `step`-th step reads `need` bytes of weight `ix`; the tape declares `declared`.
    WeightShort {
        step: usize,
        ix: u32,
        need: u64,
        declared: u64,
    },
    /// A KV region is not at a multiple of [`CUDA_ARENA_ALIGN`], or reaches past the arena.
    KvRegionOutside {
        region: Region,
        span: ArenaSpan,
        total: u64,
    },
    /// A KV region is not the bytes the forward fills: the block table `max_blocks_per_seq · 4`,
    /// the slot mapping `bucket_m · 4`, the sequence exactly 8.
    KvRegionShape {
        region: Region,
        span: ArenaSpan,
        want: u64,
    },
    /// The tape's block size is not [`CUDA_KV_BLOCK_SIZE`], the one the kernels take.
    KvBlockSize { block_size: u32 },
    /// The `step`-th step uses KV layer index `layer`; the tape has `layers` (none when it has no
    /// [`KvTape`]).
    KvLayer {
        step: usize,
        layer: u32,
        layers: usize,
    },
    /// The `step`-th step's `region` operand is not the tape's region of that name (`None` when
    /// the tape has none).
    KvRegionMismatch {
        step: usize,
        region: Region,
        span: ArenaSpan,
        tape: Option<ArenaSpan>,
    },
    /// The `step`-th step's kv heads or head dim are not the tape's KV geometry, so its kernel
    /// would address the pool with other strides than the worker allocated it with.
    KvStepGeometry { step: usize },
    /// The `step`-th step's rope reads `table`, which is not the tape's rotary table (`None`
    /// when the tape has none).
    RotaryMismatch {
        step: usize,
        table: ArenaOffset,
        tape: Option<ArenaOffset>,
    },
    /// The `step`-th step writes `write` over `region`, which the forward fills before the steps
    /// run and they only read.
    WriteOverRegion {
        step: usize,
        write: ArenaSpan,
        region: Region,
    },
}

impl std::fmt::Display for TapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let at = |s: &ArenaSpan| (s.bytes, s.offset.0);
        match self {
            Self::ColourCount { bytes, offsets } => write!(
                f,
                "the arena layout has {bytes} colour sizes and {offsets} colour offsets"
            ),
            Self::ColourMisaligned { colour, offset } => write!(
                f,
                "colour {colour} is at offset {offset}, not a multiple of {CUDA_ARENA_ALIGN}"
            ),
            Self::ColourOutside {
                colour,
                span,
                total,
            } => {
                let (bytes, offset) = at(span);
                write!(
                    f,
                    "colour {colour} ({bytes} bytes at {offset}) reaches past the {total}-byte \
                     arena"
                )
            }
            Self::InputOutside { span, total } => {
                let (bytes, offset) = at(span);
                write!(
                    f,
                    "the input ({bytes} bytes at {offset}) reaches past the {total}-byte arena"
                )
            }
            Self::ResultOutside { span, total } => {
                let (bytes, offset) = at(span);
                write!(
                    f,
                    "the result ({bytes} bytes at {offset}) reaches past the {total}-byte arena"
                )
            }
            Self::StepOverflow { step } => write!(f, "step {step}'s byte counts overflow u64"),
            Self::StepMisaligned { step, offset } => write!(
                f,
                "step {step} touches the arena at {offset}, not a multiple of {CUDA_ARENA_ALIGN}"
            ),
            Self::StepOutside { step, span, total } => {
                let (bytes, offset) = at(span);
                write!(
                    f,
                    "step {step} touches {bytes} bytes at {offset}, past the {total}-byte arena"
                )
            }
            Self::WriteOverlapsRead { step, write, read } => {
                let ((wb, wo), (rb, ro)) = (at(write), at(read));
                write!(
                    f,
                    "step {step} writes {wb} bytes at {wo} over the {rb} bytes at {ro} it reads"
                )
            }
            Self::WritesOverlap { step } => {
                write!(f, "the two spans step {step} writes overlap each other")
            }
            Self::RotaryOutside { span, total } => {
                let (bytes, offset) = at(span);
                write!(
                    f,
                    "the rotary table ({bytes} bytes at {offset}) is misaligned or reaches past \
                     the {total}-byte arena"
                )
            }
            Self::RotaryShape {
                span,
                bucket_m,
                rot_dim,
            } => write!(
                f,
                "the rotary table is {} bytes; {bucket_m} rows of rot_dim = {rot_dim} bf16 (a \
                 non-zero even width) are {}",
                span.bytes,
                u64::from(*bucket_m) * u64::from(*rot_dim) * CUDA_ACT_BYTES
            ),
            Self::WeightIndex { step, ix, weights } => write!(
                f,
                "step {step} reads weight {ix}, and the tape declares {weights} weights"
            ),
            Self::WeightShort {
                step,
                ix,
                need,
                declared,
            } => write!(
                f,
                "step {step} reads {need} bytes of weight {ix}; the tape declares {declared}"
            ),
            Self::KvRegionOutside {
                region,
                span,
                total,
            } => {
                let (bytes, offset) = at(span);
                write!(
                    f,
                    "{region} ({bytes} bytes at {offset}) is misaligned or reaches past the \
                     {total}-byte arena"
                )
            }
            Self::KvRegionShape { region, span, want } => write!(
                f,
                "{region} is {} bytes; the forward fills {want}",
                span.bytes
            ),
            Self::KvBlockSize { block_size } => write!(
                f,
                "the KV block size is {block_size}; the CUDA kernels take {CUDA_KV_BLOCK_SIZE}"
            ),
            Self::KvLayer {
                step,
                layer,
                layers,
            } => write!(
                f,
                "step {step} uses KV layer index {layer}, and the tape has {layers} KV layers"
            ),
            Self::KvRegionMismatch {
                step,
                region,
                span,
                tape,
            } => {
                let (bytes, offset) = at(span);
                match tape {
                    Some(t) => {
                        let (tb, to) = at(t);
                        write!(
                            f,
                            "step {step} reads {region} as {bytes} bytes at {offset}; the tape's is \
                             {tb} bytes at {to}"
                        )
                    }
                    None => write!(
                        f,
                        "step {step} reads {region} as {bytes} bytes at {offset}; the tape has no \
                         KV cache"
                    ),
                }
            }
            Self::KvStepGeometry { step } => write!(
                f,
                "step {step}'s kv heads or head dim are not the tape's KV cache geometry"
            ),
            Self::RotaryMismatch { step, table, tape } => match tape {
                Some(t) => write!(
                    f,
                    "step {step} reads a rotary table at {}; the tape's is at {}",
                    table.0, t.0
                ),
                None => write!(
                    f,
                    "step {step} reads a rotary table at {}; the tape has none",
                    table.0
                ),
            },
            Self::WriteOverRegion {
                step,
                write,
                region,
            } => {
                let (bytes, offset) = at(write);
                write!(
                    f,
                    "step {step} writes {bytes} bytes at {offset} over {region}, which the \
                     forward fills"
                )
            }
        }
    }
}

impl std::error::Error for TapeError {}

impl CudaTape {
    /// Check that playing this tape stays inside its arena, its declared weights and its KV pool.
    ///
    /// Every colour is aligned and inside `total_bytes`, and so are the input and result spans,
    /// the rotary table and the KV regions, each region the size the forward fills. Every step's
    /// spans ([`CudaStep::extents`]) are aligned and inside `total_bytes`, each span it writes
    /// overlaps none of its reads, nor its other write, nor a region the forward fills, and each
    /// weight it reads is one the tape declares, declared at least as large as the step reads. A
    /// rope reads the tape's rotary table, a KV step the tape's regions, at a KV layer the tape
    /// has, with the tape's KV geometry. All arithmetic is checked. The worker runs this on every
    /// tape before it allocates anything; its own weight and pool checks then close the chain
    /// from each step to the buffer it reads.
    pub fn validate(&self) -> Result<(), TapeError> {
        let total = self.arena.total_bytes;
        let (bytes, offsets) = (self.arena.colour_bytes, self.arena.colour_offsets);
        if bytes.len() != offsets.len() {
            return Err(TapeError::ColourCount {
                bytes: bytes.len(),
                offsets: offsets.len(),
            });
        }
        for (colour, (&bytes, &offset)) in bytes.iter().zip(offsets).enumerate() {
            if !offset.is_multiple_of(CUDA_ARENA_ALIGN) {
                return Err(TapeError::ColourMisaligned { colour, offset });
            }
            let span = ArenaSpan {
                offset: ArenaOffset(offset),
                bytes,
            };
            if !inside(span, total) {
                return Err(TapeError::ColourOutside {
                    colour,
                    span,
                    total,
                });
            }
        }
        if !inside(self.input, total) {
            return Err(TapeError::InputOutside {
                span: self.input,
                total,
            });
        }
        if !inside(self.result, total) {
            return Err(TapeError::ResultOutside {
                span: self.result,
                total,
            });
        }
        if let Some(t) = self.rotary {
            if !(t.span.offset.0.is_multiple_of(CUDA_ARENA_ALIGN) && inside(t.span, total)) {
                return Err(TapeError::RotaryOutside {
                    span: t.span,
                    total,
                });
            }
            // The forward fills exactly `bucket_m · rot_dim` bf16 into the span, in rows of
            // `rot_dim/2` cos and `rot_dim/2` sin.
            let rows_bytes =
                (u64::from(self.bucket_m) * u64::from(t.rot_dim)).checked_mul(CUDA_ACT_BYTES);
            if t.rot_dim == 0 || t.rot_dim % 2 != 0 || rows_bytes != Some(t.span.bytes) {
                return Err(TapeError::RotaryShape {
                    span: t.span,
                    bucket_m: self.bucket_m,
                    rot_dim: t.rot_dim,
                });
            }
        }
        let regions = self.kv_regions(total)?;
        for (step, s) in self.steps.iter().enumerate() {
            let e = s.extents().ok_or(TapeError::StepOverflow { step })?;
            let reads = e.reads.iter().flatten();
            let writes = [Some(e.writes), e.second_write];
            let writes = writes.iter().flatten();
            for &span in reads.clone().chain(writes.clone()) {
                if !span.offset.0.is_multiple_of(CUDA_ARENA_ALIGN) {
                    return Err(TapeError::StepMisaligned {
                        step,
                        offset: span.offset.0,
                    });
                }
                if !inside(span, total) {
                    return Err(TapeError::StepOutside { step, span, total });
                }
            }
            for &write in writes {
                if let Some(&read) = reads.clone().find(|&&r| overlap(r, write)) {
                    return Err(TapeError::WriteOverlapsRead { step, write, read });
                }
            }
            if e.second_write.is_some_and(|w| overlap(w, e.writes)) {
                return Err(TapeError::WritesOverlap { step });
            }
            for &write in [Some(e.writes), e.second_write].iter().flatten() {
                if let Some(&(region, _)) = regions.iter().find(|(_, r)| overlap(*r, write)) {
                    return Err(TapeError::WriteOverRegion {
                        step,
                        write,
                        region,
                    });
                }
            }
            self.check_step_operands(step, s)?;
            if let Some((layer, _)) = e.kv {
                let layers = self.kv.map_or(0, |kv| kv.layers.len());
                if layer as usize >= layers {
                    return Err(TapeError::KvLayer {
                        step,
                        layer,
                        layers,
                    });
                }
            }
            if let Some(w) = e.weight {
                let weights = self.weight_bytes.len();
                let ix = w.ix.0;
                let &declared = (self.weight_bytes.get(ix as usize))
                    .ok_or(TapeError::WeightIndex { step, ix, weights })?;
                if declared < w.bytes {
                    return Err(TapeError::WeightShort {
                        step,
                        ix,
                        need: w.bytes,
                        declared,
                    });
                }
            }
        }
        Ok(())
    }

    /// The regions the forward fills before the steps run: the rotary table and the KV regions,
    /// each checked to be aligned, inside the arena, and the size the forward writes. The rotary
    /// table's own checks ran before this.
    fn kv_regions(&self, total: u64) -> Result<Vec<(Region, ArenaSpan)>, TapeError> {
        let mut regions: Vec<(Region, ArenaSpan)> = self
            .rotary
            .map(|t| (Region::Rotary, t.span))
            .into_iter()
            .collect();
        let Some(kv) = self.kv else {
            return Ok(regions);
        };
        if kv.block_size != CUDA_KV_BLOCK_SIZE {
            return Err(TapeError::KvBlockSize {
                block_size: kv.block_size,
            });
        }
        for (region, span, want) in [
            (
                Region::BlockTable,
                kv.block_table,
                u64::from(kv.max_blocks_per_seq) * KV_ENTRY_BYTES,
            ),
            (
                Region::SlotMapping,
                kv.slot_mapping,
                u64::from(self.bucket_m) * KV_ENTRY_BYTES,
            ),
            (Region::Sequence, kv.sequence, 2 * KV_ENTRY_BYTES),
        ] {
            if !(span.offset.0.is_multiple_of(CUDA_ARENA_ALIGN) && inside(span, total)) {
                return Err(TapeError::KvRegionOutside {
                    region,
                    span,
                    total,
                });
            }
            if span.bytes != want {
                return Err(TapeError::KvRegionShape { region, span, want });
            }
            regions.push((region, span));
        }
        Ok(regions)
    }

    /// A rope's or rope-append's table is the tape's rotary table, and a KV step's regions and
    /// geometry are the tape's.
    fn check_step_operands(&self, step: usize, s: &CudaStep) -> Result<(), TapeError> {
        let rotary = |table: ArenaOffset| {
            let tape = self.rotary.map(|t| t.span.offset);
            if tape == Some(table) {
                Ok(())
            } else {
                Err(TapeError::RotaryMismatch { step, table, tape })
            }
        };
        let region = |region: Region, span: ArenaSpan, tape: Option<ArenaSpan>| {
            if tape == Some(span) {
                Ok(())
            } else {
                Err(TapeError::KvRegionMismatch {
                    step,
                    region,
                    span,
                    tape,
                })
            }
        };
        match *s {
            CudaStep::Rope(RopeBf16 { table, .. }) => rotary(table),
            CudaStep::RopeAppend(RopeAppendBf16 {
                table,
                slot_mapping,
                rows,
                cols,
                head_dim,
                ..
            }) => {
                rotary(table)?;
                let span = ArenaSpan {
                    offset: slot_mapping,
                    bytes: u64::from(rows) * KV_ENTRY_BYTES,
                };
                region(Region::SlotMapping, span, self.kv.map(|k| k.slot_mapping))?;
                match self.kv {
                    Some(kv)
                        if head_dim == kv.head_dim
                            && Some(cols) == kv.kv_heads.checked_mul(kv.head_dim) =>
                    {
                        Ok(())
                    }
                    _ => Err(TapeError::KvStepGeometry { step }),
                }
            }
            CudaStep::AttnPrefill(AttnPrefillBf16 {
                block_table,
                sequence,
                kv_heads,
                head_dim,
                ..
            }) => {
                let kv = self.kv;
                let table = ArenaSpan {
                    offset: block_table,
                    bytes: kv.map_or(0, |k| u64::from(k.max_blocks_per_seq)) * KV_ENTRY_BYTES,
                };
                region(Region::BlockTable, table, kv.map(|k| k.block_table))?;
                let seq = ArenaSpan {
                    offset: sequence,
                    bytes: 2 * KV_ENTRY_BYTES,
                };
                region(Region::Sequence, seq, kv.map(|k| k.sequence))?;
                match kv {
                    Some(kv) if (kv_heads, head_dim) == (kv.kv_heads, kv.head_dim) => Ok(()),
                    _ => Err(TapeError::KvStepGeometry { step }),
                }
            }
            CudaStep::Gemm(_)
            | CudaStep::RmsNorm(_)
            | CudaStep::Add(_)
            | CudaStep::SiluMul(_)
            | CudaStep::GateSplit(_)
            | CudaStep::GateApply(_) => Ok(()),
        }
    }
}

/// Whether `span` ends at or before `total`, its end not overflowing.
fn inside(span: ArenaSpan, total: u64) -> bool {
    (span.offset.0.checked_add(span.bytes)).is_some_and(|end| end <= total)
}

/// Whether two spans share a byte. Both lie inside the arena, so neither end overflows.
fn overlap(a: ArenaSpan, b: ArenaSpan) -> bool {
    let (a0, b0) = (a.offset.0, b.offset.0);
    a.bytes > 0 && b.bytes > 0 && a0 < b0 + b.bytes && b0 < a0 + a.bytes
}

#[cfg(test)]
mod tests {
    use scratchy_subtile::subtile_ir::GainConvention;

    use super::*;
    use crate::step::{
        AddBf16, AttnPrefillBf16, CUDA_MAX_BLOCKS_PER_SEQ, GateSplitBf16, GemmBf16, KvUse,
        RmsNormBf16, RopeAppendBf16, RopeBf16,
    };

    const fn span(offset: u64, bytes: u64) -> ArenaSpan {
        ArenaSpan {
            offset: ArenaOffset(offset),
            bytes,
        }
    }

    /// `x [256, 512] → W1 (N = 1024) → y [256, 1024] → RmsNorm(y, g) → n [256, 1024]`.
    ///
    /// Colours: 0 `x` [256, 512] at 0, 1 `y` [256, 1024] at 262144, 2 `n` [256, 1024] at 786432.
    /// Weights: 1 `W1` [1024, 512], 2 `g` [1024].
    const GOOD_STEPS: &[CudaStep] = &[
        CudaStep::Gemm(GemmBf16 {
            a: ArenaOffset(0),
            b: WeightIx(1),
            c: ArenaOffset(262144),
            m: 256,
            n: 1024,
            k: 512,
        }),
        CudaStep::RmsNorm(RmsNormBf16 {
            x: ArenaOffset(262144),
            gain: WeightIx(2),
            y: ArenaOffset(786432),
            rows: 256,
            cols: 1024,
            eps_bits: 1e-6f32.to_bits(),
            convention: GainConvention::OnePlusScale,
        }),
    ];

    const GOOD: CudaTape = CudaTape {
        bucket_m: 256,
        arena: ArenaLayout {
            colour_bytes: &[262144, 524288, 524288],
            colour_offsets: &[0, 262144, 786432],
            total_bytes: 1310720,
        },
        input: span(0, 262144),
        result: span(786432, 524288),
        weight_bytes: &[0, 1048576, 2048],
        steps: GOOD_STEPS,
        rotary: None,
        kv: None,
    };

    /// `GOOD` with its steps replaced by `step` alone.
    fn with_step(step: CudaStep) -> CudaTape {
        CudaTape {
            steps: Box::leak(Box::new([step])),
            ..GOOD
        }
    }

    fn gemm(a: u64, b: u32, c: u64, (m, n, k): (u32, u32, u32)) -> CudaStep {
        CudaStep::Gemm(GemmBf16 {
            a: ArenaOffset(a),
            b: WeightIx(b),
            c: ArenaOffset(c),
            m,
            n,
            k,
        })
    }

    #[test]
    fn a_well_formed_tape_validates() {
        assert_eq!(GOOD.validate(), Ok(()));
    }

    #[test]
    fn colour_count() {
        let t = CudaTape {
            arena: ArenaLayout {
                colour_offsets: &[0, 262144],
                ..GOOD.arena
            },
            ..GOOD
        };
        assert_eq!(
            t.validate(),
            Err(TapeError::ColourCount {
                bytes: 3,
                offsets: 2
            })
        );
    }

    #[test]
    fn colour_misaligned() {
        let t = CudaTape {
            arena: ArenaLayout {
                colour_offsets: &[0, 262144, 786560],
                ..GOOD.arena
            },
            ..GOOD
        };
        assert_eq!(
            t.validate(),
            Err(TapeError::ColourMisaligned {
                colour: 2,
                offset: 786560
            })
        );
    }

    #[test]
    fn colour_outside() {
        let t = CudaTape {
            arena: ArenaLayout {
                total_bytes: 1048576,
                ..GOOD.arena
            },
            ..GOOD
        };
        assert_eq!(
            t.validate(),
            Err(TapeError::ColourOutside {
                colour: 2,
                span: span(786432, 524288),
                total: 1048576
            })
        );
        let t = CudaTape {
            arena: ArenaLayout {
                colour_offsets: &[0, 262144, u64::MAX - 255],
                ..GOOD.arena
            },
            ..GOOD
        };
        assert_eq!(
            t.validate(),
            Err(TapeError::ColourOutside {
                colour: 2,
                span: span(u64::MAX - 255, 524288),
                total: 1310720
            })
        );
    }

    #[test]
    fn input_outside() {
        let t = CudaTape {
            input: span(1310720, 256),
            ..GOOD
        };
        assert_eq!(
            t.validate(),
            Err(TapeError::InputOutside {
                span: span(1310720, 256),
                total: 1310720
            })
        );
    }

    #[test]
    fn result_outside() {
        let t = CudaTape {
            result: span(786432, 524544),
            ..GOOD
        };
        assert_eq!(
            t.validate(),
            Err(TapeError::ResultOutside {
                span: span(786432, 524544),
                total: 1310720
            })
        );
    }

    #[test]
    fn step_overflow() {
        let t = with_step(gemm(0, 1, 262144, (u32::MAX, u32::MAX, 4)));
        assert_eq!(t.validate(), Err(TapeError::StepOverflow { step: 0 }));
    }

    #[test]
    fn step_misaligned() {
        let t = with_step(gemm(128, 1, 262144, (256, 1024, 512)));
        assert_eq!(
            t.validate(),
            Err(TapeError::StepMisaligned {
                step: 0,
                offset: 128
            })
        );
    }

    #[test]
    fn step_outside() {
        let t = with_step(gemm(0, 1, 1048576, (256, 1024, 512)));
        assert_eq!(
            t.validate(),
            Err(TapeError::StepOutside {
                step: 0,
                span: span(1048576, 524288),
                total: 1310720
            })
        );
    }

    #[test]
    fn write_overlaps_read() {
        // `out` starts inside `b`'s span: [262144, 786432) and [524288, 1048576).
        let t = with_step(CudaStep::Add(AddBf16 {
            a: ArenaOffset(0),
            b: ArenaOffset(262144),
            out: ArenaOffset(524288),
            rows: 256,
            cols: 1024,
        }));
        assert_eq!(
            t.validate(),
            Err(TapeError::WriteOverlapsRead {
                step: 0,
                write: span(524288, 524288),
                read: span(262144, 524288)
            })
        );
    }

    #[test]
    fn weight_index() {
        let t = with_step(gemm(0, 9, 262144, (256, 1024, 512)));
        assert_eq!(
            t.validate(),
            Err(TapeError::WeightIndex {
                step: 0,
                ix: 9,
                weights: 3
            })
        );
    }

    #[test]
    fn weight_short() {
        // The norm's gain, read as the GEMM's B: 1048576 bytes needed, 2048 declared.
        let t = with_step(gemm(0, 2, 262144, (256, 1024, 512)));
        assert_eq!(
            t.validate(),
            Err(TapeError::WeightShort {
                step: 0,
                ix: 2,
                need: 1048576,
                declared: 2048
            })
        );
    }

    /// `GOOD` with its arena grown by [`TABLE`] bytes and a rotary table of `bytes` at `offset`.
    fn with_rotary(offset: u64, bytes: u64) -> CudaTape {
        CudaTape {
            arena: ArenaLayout {
                total_bytes: ROTARY_TOTAL,
                ..GOOD.arena
            },
            rotary: Some(RotaryTable {
                span: span(offset, bytes),
                rot_dim: 64,
                theta_bits: 1e7f64.to_bits(),
            }),
            ..GOOD
        }
    }

    /// `GOOD`'s 256 rows of 64 bf16: the table's bytes.
    const TABLE: u64 = 256 * 64 * 2;
    /// After `GOOD`'s colours: a table placed over a colour would be written by the step that
    /// writes that colour.
    const TABLE_AT: u64 = 1310720;
    /// [`with_rotary`]'s arena: `GOOD`'s and the table.
    const ROTARY_TOTAL: u64 = TABLE_AT + TABLE;

    #[test]
    fn a_rotary_table_inside_the_arena_validates() {
        let t = with_rotary(TABLE_AT, TABLE);
        assert_eq!(t.validate(), Ok(()));
        assert_eq!(t.rotary.map(|r| r.theta()), Some(1e7));
    }

    #[test]
    fn rotary_outside() {
        for (offset, bytes) in [
            (TABLE_AT + 256, TABLE),
            (TABLE_AT + 128, TABLE),
            (u64::MAX - 255, TABLE),
        ] {
            assert_eq!(
                with_rotary(offset, bytes).validate(),
                Err(TapeError::RotaryOutside {
                    span: span(offset, bytes),
                    total: ROTARY_TOTAL
                }),
                "{offset}, {bytes}"
            );
        }
    }

    /// A table of another size than `bucket_m · rot_dim` bf16, or of an odd or zero width.
    #[test]
    fn rotary_shape() {
        let t = with_rotary(TABLE_AT, TABLE - 256);
        assert_eq!(
            t.validate(),
            Err(TapeError::RotaryShape {
                span: span(TABLE_AT, TABLE - 256),
                bucket_m: 256,
                rot_dim: 64
            })
        );
        for rot_dim in [0, 63] {
            let mut t = with_rotary(TABLE_AT, TABLE);
            if let Some(r) = t.rotary.as_mut() {
                r.rot_dim = rot_dim;
            }
            assert!(
                matches!(t.validate(), Err(TapeError::RotaryShape { rot_dim: r, .. }) if r == rot_dim),
                "rot_dim {rot_dim}"
            );
        }
    }

    /// A 64-token rope over `[64, 2048]` reading `x` from colour 1, the table from the arena's
    /// last 8192 bytes, and writing colour 2.
    fn rope(table: u64) -> CudaStep {
        CudaStep::Rope(RopeBf16 {
            x: ArenaOffset(262144),
            table: ArenaOffset(table),
            out: ArenaOffset(786432),
            rows: 64,
            cols: 2048,
            head_dim: 256,
            rot_dim: 64,
        })
    }

    /// A rope reading the tape's table validates; one reaching past the arena does not.
    #[test]
    fn a_rope_reads_its_table() {
        let t = CudaTape {
            steps: Box::leak(Box::new([rope(TABLE_AT)])),
            ..with_rotary(TABLE_AT, TABLE)
        };
        assert_eq!(t.validate(), Ok(()));
        assert_eq!(
            with_step(rope(1306624)).validate(),
            Err(TapeError::StepOutside {
                step: 0,
                span: span(1306624, 8192),
                total: 1310720
            })
        );
    }

    /// A rope whose table is not the tape's rotary table, or on a tape with none: the forward
    /// fills only the tape's.
    #[test]
    fn rotary_mismatch() {
        let t = CudaTape {
            steps: Box::leak(Box::new([rope(TABLE_AT - 256)])),
            ..with_rotary(TABLE_AT, TABLE)
        };
        assert_eq!(
            t.validate(),
            Err(TapeError::RotaryMismatch {
                step: 0,
                table: ArenaOffset(TABLE_AT - 256),
                tape: Some(ArenaOffset(TABLE_AT)),
            })
        );
        assert_eq!(
            with_step(rope(1302528)).validate(),
            Err(TapeError::RotaryMismatch {
                step: 0,
                table: ArenaOffset(1302528),
                tape: None,
            })
        );
    }

    // ── The KV cache's regions ─────────────────────────────────────

    /// [`KV_GOOD`]'s regions, after `GOOD`'s colours: the rotary table (`256 · 64` bf16), the
    /// block table (`128` u32), the slot mapping (`256` u32), the sequence (`2` u32, in a 256-byte
    /// slot).
    const KV_ROTARY: u64 = 1310720;
    const KV_TABLE: u64 = KV_ROTARY + TABLE;
    const KV_SLOTS: u64 = KV_TABLE + 512;
    const KV_SEQ: u64 = KV_SLOTS + 1024;
    const KV_TOTAL: u64 = KV_SEQ + 256;

    /// `k [256, 512]` at 262144 and `v` at 524288 (colour 1's halves), rotated K into colour 2,
    /// written to KV layer `layer`.
    fn rope_append(layer: u32, slot_mapping: u64) -> CudaStep {
        CudaStep::RopeAppend(RopeAppendBf16 {
            k: ArenaOffset(262144),
            v: ArenaOffset(524288),
            table: ArenaOffset(KV_ROTARY),
            slot_mapping: ArenaOffset(slot_mapping),
            k_out: ArenaOffset(786432),
            layer,
            rows: 256,
            cols: 512,
            head_dim: 256,
            rot_dim: 64,
        })
    }

    /// `q [256, 2 · 256]` at 262144 over KV layer 0, into colour 2.
    fn attn(block_table: u64, sequence: u64, kv_heads: u32) -> CudaStep {
        CudaStep::AttnPrefill(AttnPrefillBf16 {
            q: ArenaOffset(262144),
            block_table: ArenaOffset(block_table),
            sequence: ArenaOffset(sequence),
            out: ArenaOffset(786432),
            layer: 0,
            rows: 256,
            q_heads: 2,
            kv_heads,
            head_dim: 256,
            scale_bits: 0.0625f32.to_bits(),
        })
    }

    const KV: KvTape = KvTape {
        layers: &[3],
        kv_heads: 2,
        head_dim: 256,
        block_size: CUDA_KV_BLOCK_SIZE,
        max_blocks_per_seq: CUDA_MAX_BLOCKS_PER_SEQ,
        block_table: span(KV_TABLE, 512),
        slot_mapping: span(KV_SLOTS, 1024),
        sequence: span(KV_SEQ, 8),
    };

    /// `GOOD`'s colours with the rotary table and the three KV regions after them, and `steps`.
    fn kv_tape(kv: KvTape, steps: &[CudaStep]) -> CudaTape {
        CudaTape {
            arena: ArenaLayout {
                total_bytes: KV_TOTAL,
                ..GOOD.arena
            },
            steps: Box::leak(steps.to_vec().into_boxed_slice()),
            rotary: Some(RotaryTable {
                span: span(KV_ROTARY, TABLE),
                rot_dim: 64,
                theta_bits: 1e7f64.to_bits(),
            }),
            kv: Some(kv),
            ..GOOD
        }
    }

    #[test]
    fn a_kv_tape_validates() {
        let steps = [rope_append(0, KV_SLOTS), attn(KV_TABLE, KV_SEQ, 2)];
        assert_eq!(kv_tape(KV, &steps).validate(), Ok(()));
        let [Some(e0), Some(e1)] = steps.map(|s| s.extents()) else {
            panic!("both steps' extents fit u64");
        };
        assert_eq!(e0.kv, Some((0, KvUse::Write)));
        assert_eq!(e1.kv, Some((0, KvUse::Read)));
        assert_eq!(e0.reads[3], Some(span(KV_SLOTS, 1024)), "the slot mapping");
        assert_eq!(e1.reads[1], Some(span(KV_TABLE, 512)), "the block table");
        assert_eq!(e1.reads[2], Some(span(KV_SEQ, 8)), "the sequence");
    }

    #[test]
    fn kv_region_outside() {
        for (kv, region, at) in [
            (
                KvTape {
                    block_table: span(KV_TABLE + 128, 512),
                    ..KV
                },
                Region::BlockTable,
                span(KV_TABLE + 128, 512),
            ),
            (
                KvTape {
                    slot_mapping: span(KV_TOTAL, 1024),
                    ..KV
                },
                Region::SlotMapping,
                span(KV_TOTAL, 1024),
            ),
            (
                KvTape {
                    sequence: span(u64::MAX - 255, 8),
                    ..KV
                },
                Region::Sequence,
                span(u64::MAX - 255, 8),
            ),
        ] {
            assert_eq!(
                kv_tape(kv, &[]).validate(),
                Err(TapeError::KvRegionOutside {
                    region,
                    span: at,
                    total: KV_TOTAL
                }),
                "{region}"
            );
        }
    }

    #[test]
    fn kv_region_shape() {
        for (kv, region, at, want) in [
            (
                KvTape {
                    block_table: span(KV_TABLE, 256),
                    ..KV
                },
                Region::BlockTable,
                span(KV_TABLE, 256),
                512,
            ),
            (
                KvTape {
                    slot_mapping: span(KV_SLOTS, 512),
                    ..KV
                },
                Region::SlotMapping,
                span(KV_SLOTS, 512),
                1024,
            ),
            (
                KvTape {
                    sequence: span(KV_SEQ, 4),
                    ..KV
                },
                Region::Sequence,
                span(KV_SEQ, 4),
                8,
            ),
        ] {
            assert_eq!(
                kv_tape(kv, &[]).validate(),
                Err(TapeError::KvRegionShape {
                    region,
                    span: at,
                    want
                }),
                "{region}"
            );
        }
    }

    #[test]
    fn kv_block_size() {
        let kv = KvTape {
            block_size: 16,
            ..KV
        };
        assert_eq!(
            kv_tape(kv, &[]).validate(),
            Err(TapeError::KvBlockSize { block_size: 16 })
        );
    }

    #[test]
    fn kv_layer() {
        assert_eq!(
            kv_tape(KV, &[rope_append(1, KV_SLOTS)]).validate(),
            Err(TapeError::KvLayer {
                step: 0,
                layer: 1,
                layers: 1
            })
        );
    }

    /// A rope-append's slot mapping, and an attention's block table and sequence, that are not
    /// the tape's; and a KV step on a tape with no KV cache.
    #[test]
    fn kv_region_mismatch() {
        let elsewhere = KV_TABLE;
        assert_eq!(
            kv_tape(KV, &[rope_append(0, elsewhere)]).validate(),
            Err(TapeError::KvRegionMismatch {
                step: 0,
                region: Region::SlotMapping,
                span: span(elsewhere, 1024),
                tape: Some(KV.slot_mapping),
            })
        );
        assert_eq!(
            kv_tape(KV, &[attn(KV_SLOTS, KV_SEQ, 2)]).validate(),
            Err(TapeError::KvRegionMismatch {
                step: 0,
                region: Region::BlockTable,
                span: span(KV_SLOTS, 512),
                tape: Some(KV.block_table),
            })
        );
        assert_eq!(
            kv_tape(KV, &[attn(KV_TABLE, KV_SLOTS, 2)]).validate(),
            Err(TapeError::KvRegionMismatch {
                step: 0,
                region: Region::Sequence,
                span: span(KV_SLOTS, 8),
                tape: Some(KV.sequence),
            })
        );
        let t = CudaTape {
            kv: None,
            ..kv_tape(KV, &[attn(KV_TABLE, KV_SEQ, 2)])
        };
        assert_eq!(
            t.validate(),
            Err(TapeError::KvRegionMismatch {
                step: 0,
                region: Region::BlockTable,
                span: span(KV_TABLE, 0),
                tape: None,
            })
        );
    }

    /// An attention whose kv heads are not the tape's would read the pool with other strides.
    #[test]
    fn kv_step_geometry() {
        assert_eq!(
            kv_tape(KV, &[attn(KV_TABLE, KV_SEQ, 1)]).validate(),
            Err(TapeError::KvStepGeometry { step: 0 })
        );
        let kv = KvTape { kv_heads: 4, ..KV };
        assert_eq!(
            kv_tape(kv, &[rope_append(0, KV_SLOTS)]).validate(),
            Err(TapeError::KvStepGeometry { step: 0 })
        );
    }

    /// A step writing over the rotary table or a KV region, which the forward fills before the
    /// steps run.
    #[test]
    fn write_over_region() {
        for (c, region) in [
            (KV_ROTARY, Region::Rotary),
            (KV_TABLE, Region::BlockTable),
            (KV_SLOTS, Region::SlotMapping),
        ] {
            let t = kv_tape(KV, &[gemm(0, 1, c, (1, 256, 512))]);
            assert_eq!(
                t.validate(),
                Err(TapeError::WriteOverRegion {
                    step: 0,
                    write: span(c, 512),
                    region
                }),
                "{region}"
            );
        }
    }

    /// A split of `qg [64, 2048]` at 0 into `q [64, 1024]` at 262144 and the gate at `gate`.
    fn gate_split(gate: u64) -> CudaStep {
        CudaStep::GateSplit(GateSplitBf16 {
            qg: ArenaOffset(0),
            q: ArenaOffset(262144),
            gate: ArenaOffset(gate),
            rows: 64,
            half_cols: 1024,
            head_dim: 256,
        })
    }

    #[test]
    fn a_gate_split_checks_both_writes() {
        assert_eq!(with_step(gate_split(786432)).validate(), Ok(()));
        // The gate starts inside `q`: [262144, 393216) and [327680, 458752).
        assert_eq!(
            with_step(gate_split(327680)).validate(),
            Err(TapeError::WritesOverlap { step: 0 })
        );
        // The gate over `qg`, which the split reads.
        assert_eq!(
            with_step(gate_split(65536)).validate(),
            Err(TapeError::WriteOverlapsRead {
                step: 0,
                write: span(65536, 131072),
                read: span(0, 262144)
            })
        );
        // The gate past the arena.
        assert_eq!(
            with_step(gate_split(1245184)).validate(),
            Err(TapeError::StepOutside {
                step: 0,
                span: span(1245184, 131072),
                total: 1310720
            })
        );
    }
}

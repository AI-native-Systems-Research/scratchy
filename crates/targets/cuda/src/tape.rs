// SPDX-License-Identifier: Apache-2.0
//! ONE BUCKET'S TAPE, AS STATIC DATA — the counterpart of metal's `LoweredMetalTape`.
//!
//! A tape is what `scratchy-target-cuda-compiler` lowers one canonical (one bucket) to: the
//! arena's layout, where the forward's input and result live in it, how large each weight must
//! be, and the steps in launch order with every operand already resolved to a byte offset or a
//! weight index. Every field is a constant built at expansion time; the runtime reads it and
//! analyses nothing. Plain `Copy` data with no pointer, so it builds in a `const` context and on a
//! host with no CUDA toolchain.

use crate::step::CudaStep;

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
        }
    }
}

impl std::error::Error for TapeError {}

impl CudaTape {
    /// Check that playing this tape stays inside its arena and its declared weights.
    ///
    /// Every colour is aligned and inside `total_bytes`, and so are the input and result spans and
    /// the rotary table. Every step's spans ([`CudaStep::extents`]) are aligned and inside
    /// `total_bytes`, each span it writes overlaps none of its reads nor its other write, and each
    /// weight it reads is one the tape declares, declared at least as large as the step reads. All arithmetic is checked. The worker runs this on every
    /// tape before it allocates anything; its own weight checks then close the chain from each
    /// step to the buffer it reads.
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
    use crate::step::{AddBf16, GateSplitBf16, GemmBf16, RmsNormBf16, RopeBf16};

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

    /// `GOOD` with a rotary table of `bytes` at `offset`.
    fn with_rotary(offset: u64, bytes: u64) -> CudaTape {
        CudaTape {
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
    /// The arena's last `TABLE` bytes.
    const TABLE_AT: u64 = 1310720 - TABLE;

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
                    total: 1310720
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

    #[test]
    fn a_rope_reads_its_table() {
        assert_eq!(with_step(rope(1302528)).validate(), Ok(()));
        assert_eq!(
            with_step(rope(1306624)).validate(),
            Err(TapeError::StepOutside {
                step: 0,
                span: span(1306624, 8192),
                total: 1310720
            })
        );
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

// SPDX-License-Identifier: Apache-2.0
//! ONE MODEL'S CUDA TAPES, EMITTED — this crate's side of the macro's target-emitter seam.
//!
//! The `#[forward]` macro hands [`emit_model`] a [`TargetModelInput`]: the canonical the shared
//! front end lowered at every workload point, and the config facts beside it. It does not know
//! which target compiler it called; Cargo linked this crate under the macro's `target-emitter`
//! dependency key because the backend is `cuda`. Everything CUDA decides about a model is here:
//!
//! - the model facts ([`model_facts`]): the neutral facts as [`CudaModelFacts`];
//! - the lowering: [`crate::canonical::lower_canonical`] at every point, one tape per `m`, after
//!   checking that every `sk` of that `m` lowers to the same tape (CUDA's tape reads no `sk`);
//! - the refusal policy: a bucket whose first GEMM CUDA has no kernel for — the decode GEMV
//!   (`m = 1`) or an `m` off the GEMM's 128-row tile — is left out and named in
//!   `CUDA_UNLOWERED`; any other refusal panics the build with its text;
//! - the emission: each tape as the tokens of a `static` `scratchy_target_cuda::tape::CudaTape`,
//!   through the shared serializer (`scratchy-const-tokens`) under [`CUDA_MODULES`].
//!
//! The emitted items, in the model's module:
//!
//! ```text
//! pub static CUDA_TAPES: &[::scratchy_target_cuda::tape::CudaTape];   // ascending bucket_m
//! pub static CUDA_UNLOWERED: &[(u32, &str)];                          // (m, why)
//! ```
//!
//! `CUDA_TAPES` is what `CudaWorker::new` takes.

use std::collections::BTreeMap;

use proc_macro2::TokenStream;
use quote::quote;
use scratchy_const_tokens::{ModuleRow, ModuleTable, const_tokens};
use scratchy_subtile::handoff::{TargetModelFacts, TargetModelInput, TargetPoint};
use scratchy_target_cuda::tape::{ArenaLayout, CudaTape, GdnTape, KvTape, TapeError};

use crate::canonical::{
    CanonicalAt, CanonicalGdn, CanonicalKv, CanonicalRefusal, CudaCanonical, CudaGdnFacts,
    CudaModelFacts, GemmShapeRule, lower_canonical,
};

/// The modules of the CUDA tape types, and the aliases the emitted tapes name them by: the one
/// fact the shared serializer needs from CUDA. A tape type missing here is an expansion-time
/// error naming it.
pub const CUDA_MODULES: ModuleTable = ModuleTable {
    rows: &[
        ModuleRow {
            alias: "__ct",
            path: "::scratchy_target_cuda::tape",
            types: &[
                "CudaTape",
                "ArenaLayout",
                "ArenaSpan",
                "ArenaOffset",
                "WeightIx",
                "RotaryTable",
                "KvTape",
                "GdnTape",
            ],
        },
        ModuleRow {
            alias: "__cs",
            path: "::scratchy_target_cuda::step",
            types: &[
                "CudaStep",
                "GemmBf16",
                "RmsNormBf16",
                "AddBf16",
                "SiluMulBf16",
                "RopeBf16",
                "GateSplitBf16",
                "GateApplyBf16",
                "RopeAppendBf16",
                "AttnPrefillBf16",
                "GatedDeltaNetBf16",
                "GainConvention",
            ],
        },
    ],
};

/// The model facts CUDA's lowering reads, from the neutral ones the macro resolved.
///
/// A config that declares no `rope_theta` gets a NaN here, never a default: a tape with a rotary
/// table is then refused ([`ModelRefusal::NoRopeTheta`]), and a tape with none reads no theta.
pub fn model_facts(f: &TargetModelFacts) -> CudaModelFacts {
    CudaModelFacts {
        head_dim: f.head_dim,
        rot_dim: f.rot_dim,
        rope_theta: f.rope_theta.unwrap_or(f64::NAN),
        gdn: f.gdn.map(|g| CudaGdnFacts {
            num_k_heads: g.num_k_heads,
            head_k: g.head_k,
            conv_kernel: g.conv_kernel,
            eps: f.rms_norm_eps,
        }),
    }
}

/// One model as CUDA lowers it: a tape per prefill bucket, and the buckets it leaves out.
#[derive(Debug)]
pub struct CudaModel {
    /// One per lowered `m`, ascending `bucket_m`.
    pub tapes: Vec<CudaCanonical>,
    /// Each `m` CUDA has no kernel for yet, ascending, with the refusal's text.
    pub unlowered: Vec<(u32, String)>,
}

/// Why a model's CUDA tapes cannot be emitted. Every one panics the build: a refusal is never
/// dropped silently.
#[derive(Debug)]
pub enum ModelRefusal {
    /// The shared front end produced no canonical at this point.
    FrontEnd { point: TargetPoint, why: String },
    /// The canonical at this point does not lower, for a reason other than a bucket CUDA has no
    /// GEMM kernel for.
    Lowering {
        point: TargetPoint,
        refusal: CanonicalRefusal,
    },
    /// Two `sk` points of one `m` lowered to different tapes; CUDA emits one tape per `m`.
    SkDiffers { m: u32, first: u32, other: u32 },
    /// One `sk` point of `m` lowered and another was left out.
    Mixed { m: u32, lowered: u32, left_out: u32 },
    /// The tape lowered at `m` says it is a bucket of `bucket_m` rows.
    BucketRows { m: u32, bucket_m: u32 },
    /// The tape at `m` builds a rotary table, and the config declares no `rope_theta`.
    NoRopeTheta { m: u32 },
    /// No bucket lowered: the model would have no tape.
    NothingLowered,
    /// The `static` tape at `m` does not validate.
    Tape { m: u32, error: TapeError },
}

impl std::fmt::Display for ModelRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FrontEnd { point, why } => write!(
                f,
                "the shared front end refused m={} sk={}: {why}",
                point.m, point.sk
            ),
            Self::Lowering { point, refusal } => {
                write!(f, "m={} sk={}: {refusal}", point.m, point.sk)
            }
            Self::SkDiffers { m, first, other } => write!(
                f,
                "m={m} lowers to one tape at sk={first} and another at sk={other}; CUDA's tape \
                 reads no sk, so one per m is emitted"
            ),
            Self::Mixed {
                m,
                lowered,
                left_out,
            } => write!(
                f,
                "m={m} lowers at sk={lowered} and is left out at sk={left_out}"
            ),
            Self::BucketRows { m, bucket_m } => {
                write!(
                    f,
                    "the tape lowered at m={m} is a bucket of {bucket_m} rows"
                )
            }
            Self::NoRopeTheta { m } => write!(
                f,
                "the tape at m={m} builds a rotary table and the config declares no rope_theta"
            ),
            Self::NothingLowered => write!(f, "no bucket lowers to CUDA"),
            Self::Tape { m, error } => write!(f, "the static tape at m={m}: {error}"),
        }
    }
}

impl std::error::Error for ModelRefusal {}

/// A refusal that leaves its bucket out: the bucket's first GEMM is the decode GEMV (`m = 1`) or
/// has an `m` off the bf16 GEMM's tile, which CUDA has no kernel for yet. Every other refusal is
/// an error.
pub fn leaves_bucket_out(r: &CanonicalRefusal) -> bool {
    matches!(
        r,
        CanonicalRefusal::GemmShape {
            rule: GemmShapeRule::Decode | GemmShapeRule::OffTile { dim: "m", .. },
            ..
        }
    )
}

/// One workload point's canonical, lowered or refused.
type Lowered = Result<CudaCanonical, CanonicalRefusal>;

/// Lower every canonical of `input`: one tape per `m`, the left-out `m`s named.
pub fn lower_model(input: &TargetModelInput) -> Result<CudaModel, ModelRefusal> {
    if let Some((point, why)) = input.unlowered.first() {
        return Err(ModelRefusal::FrontEnd {
            point: *point,
            why: why.clone(),
        });
    }
    let facts = model_facts(&input.facts);
    let mut by_m: BTreeMap<u32, Vec<(u32, Lowered)>> = BTreeMap::new();
    for b in &input.buckets {
        let at = CanonicalAt {
            stem: &input.stem,
            m: u64::from(b.point.m),
        };
        let lowered = lower_canonical(&b.lowered, facts, at);
        by_m.entry(b.point.m)
            .or_default()
            .push((b.point.sk, lowered));
    }
    let mut model = CudaModel {
        tapes: Vec::new(),
        unlowered: Vec::new(),
    };
    for (m, points) in by_m {
        let mut lowered: Option<(u32, CudaCanonical)> = None;
        let mut left_out: Option<(u32, CanonicalRefusal)> = None;
        for (sk, result) in points {
            match result {
                Ok(c) => match &lowered {
                    None => lowered = Some((sk, c)),
                    Some((first, tape)) if *tape != c => {
                        return Err(ModelRefusal::SkDiffers {
                            m,
                            first: *first,
                            other: sk,
                        });
                    }
                    Some(_) => {}
                },
                Err(refusal) if leaves_bucket_out(&refusal) => {
                    left_out.get_or_insert((sk, refusal));
                }
                Err(refusal) => {
                    return Err(ModelRefusal::Lowering {
                        point: TargetPoint { m, sk },
                        refusal,
                    });
                }
            }
        }
        match (lowered, left_out) {
            (Some((lowered, _)), Some((left_out, _))) => {
                return Err(ModelRefusal::Mixed {
                    m,
                    lowered,
                    left_out,
                });
            }
            (Some((_, tape)), None) => {
                if tape.bucket_m != m {
                    return Err(ModelRefusal::BucketRows {
                        m,
                        bucket_m: tape.bucket_m,
                    });
                }
                if tape.rotary.is_some() && input.facts.rope_theta.is_none() {
                    return Err(ModelRefusal::NoRopeTheta { m });
                }
                model.tapes.push(tape);
            }
            (None, Some((_, refusal))) => model.unlowered.push((m, refusal.to_string())),
            (None, None) => {}
        }
    }
    if model.tapes.is_empty() {
        return Err(ModelRefusal::NothingLowered);
    }
    Ok(model)
}

/// `v` as a `'static` slice, for a `CudaTape` built at expansion time only to be rendered as
/// tokens. The build-script process exits after the expansion.
fn baked<T: Clone>(v: &[T]) -> &'static [T] {
    Box::leak(v.to_vec().into_boxed_slice())
}

/// `c` as the `CudaTape` its static spells.
pub fn static_tape(c: &CudaCanonical) -> CudaTape {
    let CudaCanonical {
        bucket_m,
        colour_bytes,
        colour_offsets,
        total_bytes,
        input,
        result,
        weight_bytes,
        steps,
        rotary,
        kv,
        gdn,
    } = c;
    CudaTape {
        bucket_m: *bucket_m,
        arena: ArenaLayout {
            colour_bytes: baked(colour_bytes),
            colour_offsets: baked(colour_offsets),
            total_bytes: *total_bytes,
        },
        input: *input,
        result: *result,
        weight_bytes: baked(weight_bytes),
        steps: baked(steps),
        rotary: *rotary,
        kv: kv.as_ref().map(
            |CanonicalKv {
                 layers,
                 kv_heads,
                 head_dim,
                 block_size,
                 max_blocks_per_seq,
                 block_table,
                 slot_mapping,
                 sequence,
             }| KvTape {
                layers: baked(layers),
                kv_heads: *kv_heads,
                head_dim: *head_dim,
                block_size: *block_size,
                max_blocks_per_seq: *max_blocks_per_seq,
                block_table: *block_table,
                slot_mapping: *slot_mapping,
                sequence: *sequence,
            },
        ),
        gdn: gdn.as_ref().map(
            |CanonicalGdn {
                 layers,
                 num_k_heads,
                 num_v_heads,
                 head_k,
                 head_v,
                 conv_kernel,
                 sequence,
                 scratch,
             }| GdnTape {
                layers: baked(layers),
                num_k_heads: *num_k_heads,
                num_v_heads: *num_v_heads,
                head_k: *head_k,
                head_v: *head_v,
                conv_kernel: *conv_kernel,
                sequence: *sequence,
                scratch: *scratch,
            },
        ),
    }
}

/// The seam: `input`'s CUDA tapes as the tokens of `CUDA_TAPES` and `CUDA_UNLOWERED`, spliced
/// into the model's module by the macro. Panics, naming the model, on any refusal
/// ([`lower_model`]) and on a tape that does not validate.
pub fn emit_model(input: &TargetModelInput) -> TokenStream {
    let stem = &input.stem;
    let model = lower_model(input).unwrap_or_else(|e| panic!("[cuda emit] {stem}: {e}"));
    let tapes: Vec<TokenStream> = (model.tapes.iter())
        .map(|c| {
            let tape = static_tape(c);
            tape.validate().unwrap_or_else(|error| {
                let e = ModelRefusal::Tape {
                    m: c.bucket_m,
                    error,
                };
                panic!("[cuda emit] {stem}: {e}")
            });
            const_tokens(&CUDA_MODULES, &tape)
                .unwrap_or_else(|e| panic!("[cuda emit] {stem}: m={}: {e}", c.bucket_m))
        })
        .collect();
    for c in &model.tapes {
        eprintln!(
            "[cuda emit] {stem}: m={}: {} steps, arena {} bytes",
            c.bucket_m,
            c.steps.len(),
            c.total_bytes
        );
    }
    for (m, why) in &model.unlowered {
        eprintln!("[cuda emit] {stem}: m={m} left out: {why}");
    }
    let preamble = CUDA_MODULES.preamble();
    let unlowered = (model.unlowered.iter()).map(|(m, why)| quote! { (#m, #why) });
    quote! {
        /// This model's CUDA prefill tapes, one per bucket, ascending `bucket_m`: what
        /// `CudaWorker::new` takes. Every field a constant lowered at expansion.
        pub static CUDA_TAPES: &[::scratchy_target_cuda::tape::CudaTape] = {
            #preamble
            &[ #(#tapes),* ]
        };
        /// The buckets CUDA did not lower, and why: today the decode bucket and the buckets below
        /// the GEMM's 128-row tile.
        pub static CUDA_UNLOWERED: &[(u32, &str)] = &[ #(#unlowered),* ];
    }
}

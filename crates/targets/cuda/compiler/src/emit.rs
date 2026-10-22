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
//! - the refusal policy: a bucket whose first GEMM CUDA has no kernel for — an `m` above 1 and
//!   off the GEMM's 128-row tile — is left out and named in `CUDA_UNLOWERED`; any other refusal
//!   panics the build with its text. A one-row bucket (`m = 1`, decode) lowers: its GEMMs are
//!   GEMVs and its attention the decode attention;
//! - the weight table ([`weight_table`]): per weight index the tapes read, the checkpoint tensor
//!   that fills it, its shape and its conversion, named from the neutral `weight_prefixes`; two
//!   indices naming one tensor (a tied `lm_head` and the embedding table) become one buffer;
//! - the emission: each tape as the tokens of a `static` `scratchy_target_cuda::tape::CudaTape`,
//!   and the table as `scratchy_target_cuda::weights::CudaWeight`s, through the shared serializer
//!   (`scratchy-const-tokens`) under [`CUDA_MODULES`].
//!
//! The emitted items, in the model's module:
//!
//! ```text
//! pub static CUDA_TAPES: &[::scratchy_target_cuda::tape::CudaTape];   // ascending bucket_m
//! pub static CUDA_WEIGHTS: &[Option<::scratchy_target_cuda::weights::CudaWeight>]; // per WeightIx
//! pub static CUDA_UNLOWERED: &[(u32, &str)];                          // (m, why)
//! ```
//!
//! `CUDA_TAPES` is what `CudaWorker::new` takes, and `CUDA_WEIGHTS` what
//! `runtime::load_weights` plays against a checkpoint to make its weight buffers.

use std::collections::{BTreeMap, HashMap};

use proc_macro2::TokenStream;
use quote::quote;
use scratchy_const_tokens::{ModuleRow, ModuleTable, const_tokens};
use scratchy_subtile::handoff::{
    SourceBinding, TargetModelFacts, TargetModelInput, TargetPoint, UnrollIndex,
};
use scratchy_target_cuda::step::{
    CudaStep, EmbedBf16, GatedDeltaNetBf16, GemmBf16, RmsNormBf16, gdn_conv_dim,
};
use scratchy_target_cuda::tape::{ArenaLayout, CudaTape, GdnTape, KvTape, TapeError, WeightIx};
use scratchy_target_cuda::weights::CudaWeight;

use crate::canonical::{
    CanonicalAt, CanonicalGdn, CanonicalKv, CanonicalRefusal, CudaCanonical, CudaEmbedFacts,
    CudaGdnFacts, CudaModelFacts, GemmShapeRule, lower_canonical,
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
                "TapeInput",
            ],
        },
        ModuleRow {
            alias: "__cs",
            path: "::scratchy_target_cuda::step",
            types: &[
                "CudaStep",
                "EmbedBf16",
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
        ModuleRow {
            alias: "__cw",
            path: "::scratchy_target_cuda::weights",
            types: &["CudaWeight"],
        },
    ],
};

/// The model facts CUDA's lowering reads, from the neutral ones the macro resolved.
///
/// A config that declares no `rope_theta` gets a NaN here, never a default: a tape with a rotary
/// table is then refused ([`ModelRefusal::NoRopeTheta`]), and a tape with none reads no theta. A
/// model's input is its token ids, embedded from a table of `vocab_size` rows; a config that
/// declares none gets no embedding table here, and [`lower_model`] refuses it
/// ([`ModelRefusal::NoVocab`]).
pub fn model_facts(f: &TargetModelFacts) -> CudaModelFacts {
    CudaModelFacts {
        embed: (f.vocab_size > 0).then_some(CudaEmbedFacts {
            vocab: f.vocab_size,
        }),
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
    /// Per weight index, parallel to every tape's `weight_bytes`: how the checkpoint fills it.
    /// `None` for an index no step reads (`weight_bytes` 0).
    pub weights: Vec<Option<CudaWeight>>,
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
    /// The config declares no `vocab_size`, so the forward has no embedding table to gather its
    /// token ids from.
    NoVocab,
    /// The tape at `m` binds another source than the first tape's at index `ix`: the weight table
    /// is shared by every bucket.
    Bindings { m: u32, ix: usize },
    /// Two steps read weight `ix` as different things (`first`, `other`): another role, or
    /// another shape.
    WeightRole {
        ix: usize,
        first: String,
        other: String,
    },
    /// Weight `ix` declares `bytes` bytes and no step reads it.
    UnreadWeight { ix: usize, bytes: u64 },
    /// Weight `ix` is read as a model weight the handoff gives no checkpoint prefix for, or as the
    /// embedding table when the handoff names none.
    NoPrefix { ix: usize, binding: SourceBinding },
    /// Weight `ix` packs to `packed` bytes; the tapes declare `declared`.
    WeightBytes {
        ix: usize,
        packed: u64,
        declared: u64,
    },
    /// Weights `first` and `ix` name the same checkpoint tensor `tensor` as different entries.
    SharedTensor {
        ix: usize,
        first: usize,
        tensor: String,
    },
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
            Self::NoVocab => write!(
                f,
                "the config declares no vocab_size, so the forward has no embedding table for its \
                 token ids"
            ),
            Self::Bindings { m, ix } => write!(
                f,
                "the tape at m={m} binds another source at index {ix} than the first tape; one \
                 weight table serves every bucket"
            ),
            Self::WeightRole { ix, first, other } => {
                write!(f, "weight {ix} is read as {first} and as {other}")
            }
            Self::UnreadWeight { ix, bytes } => {
                write!(f, "weight {ix} declares {bytes} bytes and no step reads it")
            }
            Self::NoPrefix { ix, binding } => write!(
                f,
                "weight {ix} is source {binding:?}, which the handoff names no checkpoint prefix \
                 for"
            ),
            Self::WeightBytes {
                ix,
                packed,
                declared,
            } => write!(
                f,
                "weight {ix} packs to {packed} bytes; the tapes declare {declared}"
            ),
            Self::SharedTensor { ix, first, tensor } => write!(
                f,
                "weights {first} and {ix} name the checkpoint tensor {tensor} as different entries"
            ),
        }
    }
}

impl std::error::Error for ModelRefusal {}

/// A refusal that leaves its bucket out: the bucket's first GEMM has an `m` off the bf16 GEMM's
/// tile (above 1, which the GEMV takes), which CUDA has no kernel for yet. Every other refusal is
/// an error.
pub fn leaves_bucket_out(r: &CanonicalRefusal) -> bool {
    matches!(
        r,
        CanonicalRefusal::GemmShape {
            rule: GemmShapeRule::OffTile { dim: "m", .. },
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
    if input.facts.vocab_size == 0 {
        return Err(ModelRefusal::NoVocab);
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
        weights: Vec::new(),
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
    model.weights = weight_table(input, &mut model.tapes)?;
    Ok(model)
}

/// What a step reads a weight as: the entry that fills it, before its tensor is named.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    /// One bf16 tensor of this shape.
    Bf16([u64; 2], usize),
    /// One GDN layer's packed bundle of this geometry: `conv_dim`, `conv_kernel`, `nv`, `head_v`.
    Bundle([u32; 4]),
}

impl Role {
    fn shape(&self) -> &[u64] {
        match self {
            Self::Bf16(dims, rank) => &dims[..*rank],
            Self::Bundle(_) => &[],
        }
    }
}

/// What `step` reads its weight as, and that weight's index.
fn role(step: &CudaStep) -> Option<(WeightIx, Role)> {
    match *step {
        CudaStep::Embed(EmbedBf16 {
            table,
            hidden,
            vocab,
            ..
        }) => Some((table, Role::Bf16([u64::from(vocab), u64::from(hidden)], 2))),
        CudaStep::Gemm(GemmBf16 { b, n, k, .. })
        | CudaStep::GemmNarrow(GemmBf16 { b, n, k, .. })
        | CudaStep::Gemv(GemmBf16 { b, n, k, .. }) => {
            Some((b, Role::Bf16([u64::from(n), u64::from(k)], 2)))
        }
        CudaStep::RmsNorm(RmsNormBf16 { gain, cols, .. }) => {
            Some((gain, Role::Bf16([u64::from(cols), 0], 1)))
        }
        CudaStep::GatedDeltaNet(GatedDeltaNetBf16 {
            bundle,
            num_k_heads,
            num_v_heads,
            head_k,
            head_v,
            conv_kernel,
            ..
        }) => {
            // `validate` sized the bundle, so its conv width fits; `u32::MAX` never matches a
            // checkpoint's.
            let conv_dim =
                gdn_conv_dim(num_k_heads, head_k, num_v_heads, head_v).unwrap_or(u32::MAX);
            Some((
                bundle,
                Role::Bundle([conv_dim, conv_kernel, num_v_heads, head_v]),
            ))
        }
        CudaStep::Add(_)
        | CudaStep::SiluMul(_)
        | CudaStep::Rope(_)
        | CudaStep::GateSplit(_)
        | CudaStep::GateApply(_)
        | CudaStep::RopeAppend(_)
        | CudaStep::AttnPrefill(_)
        | CudaStep::AttnDecode(_) => None,
    }
}

/// `s` as a `'static` str, for a [`CudaWeight`] built at expansion time only to be rendered as
/// tokens. The build-script process exits after the expansion.
fn baked_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// The weight table of `input`'s lowered `tapes`, one entry per weight index; the tapes rewritten
/// so that two indices naming one checkpoint tensor read the earlier one.
///
/// Every bucket binds the same source at every index (one table serves them all). Each index a
/// step reads is read as one thing ([`role`]): a GEMM's `[n, k]`, a norm's `[cols]`, the embed's
/// `[vocab, hidden]` (all bf16 tensors `{prefix}.weight`), or a GDN bundle (`{prefix}.conv1d.weight`,
/// `{prefix}.A_log`, `{prefix}.dt_bias`, `{prefix}.norm.weight`). The prefix is the handoff's: a
/// model weight's `weight_prefixes` entry, the embedding table's `embed_prefix`. An index whose
/// declared bytes no step reads, or whose entry packs to other bytes than it declares, is refused.
/// Then the dedup: a later index whose tensor an earlier index already names is read through the
/// earlier one, its `weight_bytes` 0 and its entry `None`.
pub fn weight_table(
    input: &TargetModelInput,
    tapes: &mut [CudaCanonical],
) -> Result<Vec<Option<CudaWeight>>, ModelRefusal> {
    // Every canonical a tape was lowered from, every `sk` of its `m`.
    let lowered: Vec<_> = (input.buckets.iter())
        .filter(|b| tapes.iter().any(|t| t.bucket_m == b.point.m))
        .collect();
    let Some(first) = lowered.first() else {
        return Ok(Vec::new());
    };
    let bindings = &first.lowered.bindings;
    for b in &lowered {
        let theirs = &b.lowered.bindings;
        if let Some(ix) =
            (0..bindings.len().max(theirs.len())).find(|&i| bindings.get(i) != theirs.get(i))
        {
            return Err(ModelRefusal::Bindings { m: b.point.m, ix });
        }
    }
    let n = bindings.len();
    let mut roles: Vec<Option<Role>> = vec![None; n];
    for tape in tapes.iter() {
        for step in &tape.steps {
            let Some((WeightIx(ix), role)) = role(step) else {
                continue;
            };
            let ix = ix as usize;
            match roles[ix] {
                Some(first) if first != role => {
                    return Err(ModelRefusal::WeightRole {
                        ix,
                        first: format!("{first:?}"),
                        other: format!("{role:?}"),
                    });
                }
                _ => roles[ix] = Some(role),
            }
        }
    }
    let declared = |ix: usize| {
        (tapes.iter())
            .map(|t| t.weight_bytes[ix])
            .max()
            .unwrap_or(0)
    };
    let prefixes: HashMap<(u32, Option<UnrollIndex>), &str> = (input.weight_prefixes.iter())
        .map(|(id, index, prefix)| ((*id, *index), prefix.as_str()))
        .collect();
    let mut table: Vec<Option<CudaWeight>> = vec![None; n];
    for ix in 0..n {
        let Some(role) = roles[ix] else {
            let bytes = declared(ix);
            if bytes > 0 {
                return Err(ModelRefusal::UnreadWeight { ix, bytes });
            }
            continue;
        };
        let binding = &bindings[ix];
        let prefix = match *binding {
            SourceBinding::Weight { id, index } => prefixes.get(&(id, index)).copied(),
            SourceBinding::EmbeddedHidden => input.embed_prefix.as_deref(),
            _ => None,
        }
        .ok_or_else(|| ModelRefusal::NoPrefix {
            ix,
            binding: binding.clone(),
        })?;
        let entry = match role {
            Role::Bf16(..) => CudaWeight::Bf16 {
                tensor: baked_str(format!("{prefix}.weight")),
                shape: baked(role.shape()),
            },
            Role::Bundle([conv_dim, conv_kernel, nv, head_v]) => CudaWeight::GdnBundle {
                conv1d: baked_str(format!("{prefix}.conv1d.weight")),
                a_log: baked_str(format!("{prefix}.A_log")),
                dt_bias: baked_str(format!("{prefix}.dt_bias")),
                norm: baked_str(format!("{prefix}.norm.weight")),
                conv_dim,
                conv_kernel,
                nv,
                head_v,
            },
        };
        let (packed, declared) = (entry.packed_bytes().unwrap_or(u64::MAX), declared(ix));
        if packed != declared {
            return Err(ModelRefusal::WeightBytes {
                ix,
                packed,
                declared,
            });
        }
        table[ix] = Some(entry);
    }
    // One buffer per checkpoint tensor: a later index naming an earlier index's tensor reads it.
    let mut first_of: HashMap<&'static str, usize> = HashMap::new();
    let mut remap: Vec<usize> = (0..n).collect();
    for ix in 0..n {
        let Some(entry) = table[ix] else { continue };
        let tensor = match entry {
            CudaWeight::Bf16 { tensor, .. } => tensor,
            CudaWeight::GdnBundle { conv1d, .. } => conv1d,
        };
        match first_of.get(tensor) {
            None => {
                first_of.insert(tensor, ix);
            }
            Some(&first) if table[first] == Some(entry) => {
                remap[ix] = first;
                table[ix] = None;
            }
            Some(&first) => {
                return Err(ModelRefusal::SharedTensor {
                    ix,
                    first,
                    tensor: tensor.to_string(),
                });
            }
        }
    }
    for tape in tapes.iter_mut() {
        for step in &mut tape.steps {
            if let Some(w) = step.weight_mut() {
                w.0 = remap[w.0 as usize] as u32;
            }
        }
        for (ix, bytes) in tape.weight_bytes.iter_mut().enumerate() {
            if remap[ix] != ix {
                *bytes = 0;
            }
        }
    }
    Ok(table)
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
    let weights = const_tokens(&CUDA_MODULES, &model.weights)
        .unwrap_or_else(|e| panic!("[cuda emit] {stem}: the weight table: {e}"));
    eprintln!(
        "[cuda emit] {stem}: {} weights from the checkpoint",
        model.weights.iter().flatten().count()
    );
    let preamble = CUDA_MODULES.preamble();
    let unlowered = (model.unlowered.iter()).map(|(m, why)| quote! { (#m, #why) });
    quote! {
        /// This model's CUDA tapes, one per bucket, ascending `bucket_m`: the one-row decode
        /// bucket, then the prefill buckets. What `CudaWorker::new` takes. Every field a
        /// constant lowered at expansion.
        pub static CUDA_TAPES: &[::scratchy_target_cuda::tape::CudaTape] = {
            #preamble
            &[ #(#tapes),* ]
        };
        /// How the checkpoint fills each weight index the tapes read, parallel to every tape's
        /// `weight_bytes`: what `runtime::load_weights` plays. `None` for an index no step reads.
        pub static CUDA_WEIGHTS: &[Option<::scratchy_target_cuda::weights::CudaWeight>] = {
            #preamble
            #weights
        };
        /// The buckets CUDA did not lower, and why: today the buckets above one row and below
        /// the GEMM's 128-row tile.
        pub static CUDA_UNLOWERED: &[(u32, &str)] = &[ #(#unlowered),* ];
    }
}

// SPDX-License-Identifier: Apache-2.0
//! THE KV CODEC'S STEPS, INSERTED FROM A TARGET'S DECLARED FACTS.
//!
//! A target that stores its KV cache compressed runs extra steps around every KV writer and every
//! attention of a coded class: the writer's new K and V are encoded into a packed store, and the
//! attention either reads that store directly (its packed twin) or reads it staged back into the
//! codec's rotated domain, its query rotated in and its output rotated back. WHICH steps, in WHICH
//! order, under WHICH runtime guard, and which classes and head dims are coded, are the target's
//! facts ([`KvCodecFacts`]); inserting them is this one pass over the front end's op list. A
//! target without a codec never calls it.
//!
//! The inserted ops are expansion ops (`expansion_ops!`): a writer and its encodes are one
//! expansion, an attention and the steps around it another, so a target fences each as one. Every
//! later reader of a coded attention reads the end of its chain — the op its output became.

use std::collections::HashMap;

use crate::handoff::{Expansion, ExpansionId, LoweredDecode};
use crate::lower::{ArchOp, InputRef, LoweringInput, OpDesc, kv_operand_bias};
use crate::subtile_ir::{AttnMask, KvOperand, RotatedRows, SubOp};

/// When a codec step runs: a runtime condition the target's worker evaluates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CodecGuard {
    /// The cache is coded.
    Codec,
    /// The cache is coded and every sequence contributes one token: a decode step.
    CodecDecode,
    /// The cache is coded and the step is not a decode step.
    CodecNotDecode,
    /// The cache is not coded, or the step is not a decode step.
    UnlessCodecDecode,
}

impl CodecGuard {
    /// Whether a step under this guard runs on a decode step of a coded model. A dense model's
    /// steps carry no guard.
    pub const fn at_decode(self) -> bool {
        matches!(self, Self::Codec | Self::CodecDecode)
    }

    /// The guard that holds exactly when this one does not, where the vocabulary has one.
    pub const fn negation(self) -> Option<Self> {
        match self {
            Self::CodecDecode => Some(Self::UnlessCodecDecode),
            Self::UnlessCodecDecode => Some(Self::CodecDecode),
            Self::Codec | Self::CodecNotDecode => None,
        }
    }
}

/// A target's realization of each guard, one field per guard: a new guard is a missing field in
/// every target's table, not a silent default.
pub struct GuardGates<G> {
    pub codec: G,
    pub codec_decode: G,
    pub codec_not_decode: G,
    pub unless_codec_decode: G,
}

impl<G: Copy> GuardGates<G> {
    pub const fn gate(&self, guard: CodecGuard) -> G {
        match guard {
            CodecGuard::Codec => self.codec,
            CodecGuard::CodecDecode => self.codec_decode,
            CodecGuard::CodecNotDecode => self.codec_not_decode,
            CodecGuard::UnlessCodecDecode => self.unless_codec_decode,
        }
    }
}

/// A codec step and the guard it runs under.
#[derive(Clone, Copy, Debug)]
pub struct Guarded<S> {
    pub step: S,
    pub guard: CodecGuard,
}

/// A step ahead of a coded attention: it prepares what the attention reads.
#[derive(Clone, Copy, Debug)]
pub enum Before {
    /// The layer's `operand`, staged out of its packed store.
    Stage(KvOperand),
    /// The query, rotated into the codebook's domain: the attention reads the rotated rows.
    RotateQuery,
}

/// A step after a coded attention: it takes over the attention's output.
#[derive(Clone, Copy, Debug)]
pub enum After {
    /// The output, rotated back out of the codebook's domain.
    RotateOutput,
    /// The attention read straight off the packed store, into the same output.
    PackedTwin,
}

/// What surrounds a coded attention of one form, and the guard the attention itself runs under.
pub struct CodecAround {
    pub before: &'static [Guarded<Before>],
    pub anchor: CodecGuard,
    pub after: &'static [Guarded<After>],
}

/// The head dims a codec's kernels take.
#[derive(Clone, Copy, Debug)]
pub enum CodecHeadDims {
    PowerOfTwoAtMost(u32),
}

impl CodecHeadDims {
    pub const fn admits(self, head_dim: u32) -> bool {
        match self {
            Self::PowerOfTwoAtMost(max) => head_dim.is_power_of_two() && head_dim <= max,
        }
    }
}

/// A target's KV codec, as data.
pub struct KvCodecFacts {
    /// Every head dim the KV writers and attentions run at must be one of these, or nothing is
    /// coded.
    pub head_dims: CodecHeadDims,
    /// On an arch whose writers and attentions run at more than one head dim, the one class
    /// coded; an arch with one head dim codes every class.
    pub hybrid_class: AttnMask,
    /// After each coded KV writer — whether or not an attention reads its cache.
    pub after_writer: &'static [Guarded<KvOperand>],
    /// Around each coded attention of a decode form (one query row per sequence) ...
    pub decode: CodecAround,
    /// ... and of every other form.
    pub prefill: CodecAround,
}

/// Why a canonical's codec steps could not be inserted, naming the op at fault.
#[derive(Debug)]
pub enum KvCodecError {
    /// A coded attention whose new K is not a coded writer's: no packed store is named for it.
    NoWriter { op: usize, name: &'static str },
    /// A coded writer lacks a packed half the facts decode (no encode of it runs after it).
    Unencoded {
        op: usize,
        name: &'static str,
        operand: KvOperand,
    },
    /// A coded op lacks an operand the codec reads.
    MissingOperand {
        op: usize,
        name: &'static str,
        operand: u8,
    },
    /// A writer operand whose additive offset cannot be traced to its producer.
    Offset { op: usize, why: String },
    /// A coded op that another construct already expanded to.
    AlreadyExpanded { op: usize, name: &'static str },
}

impl std::fmt::Display for KvCodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoWriter { op, name } => write!(
                f,
                "op {op} ({name}): a coded attention whose new K no coded KV writer produced"
            ),
            Self::Unencoded { op, name, operand } => write!(
                f,
                "op {op} ({name}): its writer's {operand:?} is never encoded, and it is decoded"
            ),
            Self::MissingOperand { op, name, operand } => {
                write!(f, "op {op} ({name}): no operand {operand}")
            }
            Self::Offset { op, why } => write!(f, "op {op}: {why}"),
            Self::AlreadyExpanded { op, name } => {
                write!(f, "op {op} ({name}): already part of another expansion")
            }
        }
    }
}

/// Insert `facts`' codec steps around every coded KV writer and attention of `l`.
///
/// ⛔ A GEOMETRY THE KERNELS DO NOT TAKE CODES NOTHING: the op list comes back as it was.
pub fn expand_kv_codec(
    l: &LoweredDecode,
    facts: &KvCodecFacts,
) -> Result<LoweredDecode, KvCodecError> {
    let ops = &l.input.ops;
    // A writer states the model's base head dim, an attention its class's.
    let mut dims: Vec<u32> = ops
        .iter()
        .filter_map(|od| match od.op {
            SubOp::RopeAppend { head_dim, .. } => Some(head_dim.get()),
            SubOp::AttnDecode { geom, .. } => Some(geom.hd().get()),
            _ => None,
        })
        .collect();
    dims.sort_unstable();
    dims.dedup();
    if dims.is_empty() || !dims.iter().all(|hd| facts.head_dims.admits(*hd)) {
        return Ok(l.clone());
    }
    let hybrid = dims.len() > 1;
    let coded = |class: AttnMask| !hybrid || class == facts.hybrid_class;
    let first_id = l.op_expansion.iter().flatten().map(|e| e.id.0 + 1).max();
    let mut x = Expander {
        l,
        ops: Vec::with_capacity(ops.len()),
        tiles: Vec::with_capacity(ops.len()),
        expansion: Vec::with_capacity(ops.len()),
        at: vec![0; ops.len()],
        read_as: vec![0; ops.len()],
        packed: HashMap::new(),
        unnamed_reads: Vec::new(),
        next: first_id.unwrap_or(0),
    };
    for (i, od) in ops.iter().enumerate() {
        match od.op {
            SubOp::RopeAppend { attn, .. } if coded(attn) => x.writer(i, facts)?,
            SubOp::AttnDecode { mask, .. } if coded(mask) => {
                let around = if od.m == 1 {
                    &facts.decode
                } else {
                    &facts.prefill
                };
                x.attention(i, around)?
            }
            _ => {
                let at = x.push_own(i, l.op_expansion[i]);
                x.read_as[i] = at;
            }
        }
    }
    let at = &x.at;
    Ok(LoweredDecode {
        input: LoweringInput {
            sources: l.input.sources.clone(),
            ops: x.ops,
            result: x.read_as[l.input.result],
        },
        bindings: l.bindings.clone(),
        op_tiles: x.tiles,
        norm_gain_add_tiles: l
            .norm_gain_add_tiles
            .iter()
            .map(|(op, tile)| (at[*op], *tile))
            .collect(),
        op_expansion: x.expansion,
        unnamed_reads: (l.unnamed_reads.iter())
            .map(|(w, r)| (at[*w], at[*r]))
            .chain(x.unnamed_reads)
            .collect(),
    })
}

/// The expanded op list under construction.
struct Expander<'a> {
    l: &'a LoweredDecode,
    ops: Vec<OpDesc>,
    tiles: Vec<Option<(u32, u8)>>,
    expansion: Vec<Option<Expansion>>,
    /// Each source op's own index in the expanded list.
    at: Vec<usize>,
    /// The expanded op each source op's readers read: itself, or the end of its codec chain.
    read_as: Vec<usize>,
    /// Each coded writer's encodes, by the operand they pack.
    packed: HashMap<usize, Vec<(KvOperand, usize)>>,
    /// Each encode, after the writer whose cache half it packs; each coded attention, after the
    /// stage steps whose staged K/V it reads.
    unnamed_reads: Vec<(usize, usize)>,
    next: u32,
}

impl Expander<'_> {
    fn name(&self, i: usize) -> &'static str {
        self.l.input.ops[i].op.name()
    }

    /// A source op's operand as the expanded list reads it.
    fn input(&self, r: InputRef) -> InputRef {
        match r {
            InputRef::Op(j) => InputRef::Op(self.read_as[j]),
            InputRef::Ext(e) => InputRef::Ext(e),
        }
    }

    fn operand(&self, i: usize, k: u8) -> Result<InputRef, KvCodecError> {
        let r = self.l.input.ops[i].inputs.get(k as usize).copied();
        r.ok_or(KvCodecError::MissingOperand {
            op: i,
            name: self.name(i),
            operand: k,
        })
    }

    /// The tile an expanded op's output realizes, when it realizes one.
    fn tile(&self, r: InputRef) -> Option<(u32, u8)> {
        match r {
            InputRef::Op(j) => self.tiles[j],
            InputRef::Ext(_) => None,
        }
    }

    fn push(
        &mut self,
        op: ArchOp,
        m: u32,
        inputs: Vec<InputRef>,
        tile: Option<(u32, u8)>,
        expansion: Option<Expansion>,
    ) -> usize {
        self.ops.push(OpDesc { op, m, inputs });
        self.tiles.push(tile);
        self.expansion.push(expansion);
        self.ops.len() - 1
    }

    /// Source op `i` itself, its operands read through the chains before it.
    fn push_own(&mut self, i: usize, expansion: Option<Expansion>) -> usize {
        let od = &self.l.input.ops[i];
        let inputs = od.inputs.iter().map(|r| self.input(*r)).collect();
        let at = self.push(od.op, od.m, inputs, self.l.op_tiles[i], expansion);
        self.at[i] = at;
        at
    }

    /// A fresh expansion for coded op `i`, which no construct may have expanded already.
    fn open(&mut self, i: usize) -> Result<ExpansionId, KvCodecError> {
        if self.l.op_expansion[i].is_some() {
            let name = self.name(i);
            return Err(KvCodecError::AlreadyExpanded { op: i, name });
        }
        self.next += 1;
        Ok(ExpansionId(self.next - 1))
    }

    /// A coded KV writer, then its encodes: each the cache half it filled, its rotary row, and
    /// the projection whose bias the operand carries (`kv_operand_bias`).
    fn writer(&mut self, i: usize, facts: &KvCodecFacts) -> Result<(), KvCodecError> {
        let id = self.open(i)?;
        let own = Some(Expansion { id, guard: None });
        let at = self.push_own(i, own);
        self.read_as[i] = at;
        let ops = &self.l.input.ops;
        let m = ops[i].m;
        let mut packed = Vec::with_capacity(facts.after_writer.len());
        for g in facts.after_writer {
            // `[K, cos, sin, V, K_cache, V_cache]`.
            let (value, cache) = match g.step {
                KvOperand::K => (0, 4),
                KvOperand::V => (3, 5),
            };
            let mut inputs = vec![self.operand(i, cache)?, self.operand(i, 1)?];
            if let InputRef::Op(producer) = self.operand(i, value)? {
                let bias = kv_operand_bias(ops, producer);
                let bias = bias.map_err(|why| KvCodecError::Offset { op: i, why })?;
                if let Some((gemm, _)) = bias {
                    inputs.push(self.operand(gemm, 1)?);
                }
            }
            let inputs = inputs.into_iter().map(|r| self.input(r)).collect();
            let encode = SubOp::KvEncode { operand: g.step };
            let guard = Some(g.guard);
            let e = self.push(encode, m, inputs, None, Some(Expansion { id, guard }));
            // It reads the cache half through the cache, not the writer.
            self.unnamed_reads.push((at, e));
            packed.push((g.step, e));
        }
        self.packed.insert(i, packed);
        Ok(())
    }

    /// A coded attention: the steps before it, itself under its guard, the steps after it. Its
    /// packed store is its new K's writer's (operand 3).
    fn attention(&mut self, i: usize, around: &CodecAround) -> Result<(), KvCodecError> {
        let name = self.name(i);
        let writer = match self.operand(i, 3)? {
            InputRef::Op(w) if self.packed.contains_key(&w) => w,
            _ => return Err(KvCodecError::NoWriter { op: i, name }),
        };
        let packed = self.packed[&writer].clone();
        let packed_of = |operand: KvOperand| {
            let p = packed.iter().find(|(k, _)| *k == operand);
            p.map(|(_, e)| InputRef::Op(*e))
                .ok_or(KvCodecError::Unencoded {
                    op: i,
                    name,
                    operand,
                })
        };
        let (pk, pv) = (packed_of(KvOperand::K)?, packed_of(KvOperand::V)?);
        let id = self.open(i)?;
        let m = self.l.input.ops[i].m;
        let mut q = self.input(self.operand(i, 0)?);
        let mut staged = Vec::new();
        for g in around.before {
            let expansion = Some(Expansion {
                id,
                guard: Some(g.guard),
            });
            match g.step {
                Before::Stage(operand) => {
                    let stage = SubOp::KvStage { operand };
                    staged.push(self.push(stage, m, vec![packed_of(operand)?], None, expansion));
                }
                Before::RotateQuery => {
                    let rows = RotatedRows::Query;
                    let tile = self.tile(q);
                    q = InputRef::Op(self.push(
                        SubOp::RotateRows { rows },
                        m,
                        vec![q],
                        tile,
                        expansion,
                    ));
                }
            }
        }
        let anchor = Some(Expansion {
            id,
            guard: Some(around.anchor),
        });
        let at = self.push_own(i, anchor);
        self.ops[at].inputs[0] = q;
        self.unnamed_reads
            .extend(staged.into_iter().map(|s| (s, at)));
        let mut out = at;
        for g in around.after {
            let expansion = Some(Expansion {
                id,
                guard: Some(g.guard),
            });
            let (op, inputs) = match g.step {
                After::RotateOutput => {
                    let rows = RotatedRows::Output;
                    (SubOp::RotateRows { rows }, vec![InputRef::Op(out)])
                }
                After::PackedTwin => (SubOp::AttnPackedKv, vec![q, InputRef::Op(out), pk, pv]),
            };
            out = self.push(op, m, inputs, self.tiles[out], expansion);
        }
        self.read_as[i] = out;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{
        TURBOQUANT_SHAPED_CODEC as FACTS, front_end_lowered as lowered, one_layer_input_shaped,
    };
    use crate::lower::GemmWeight;
    use crate::subtile_ir::{EwKind, SourceShape};
    use CodecGuard::{Codec, CodecDecode, CodecNotDecode, UnlessCodecDecode};

    /// The one-layer fixture (llama-shaped, head dim `hd`) with every op `m` rows.
    fn layer(hd: u32, m: u32) -> LoweredDecode {
        let mut input = one_layer_input_shaped(256, 64, 512, hd);
        input.ops.iter_mut().for_each(|od| od.m = m);
        lowered(input)
    }

    /// Each expanded op's name and guard.
    fn shape(l: &LoweredDecode) -> Vec<(&'static str, Option<CodecGuard>)> {
        let guard = |i: usize| l.op_expansion[i].and_then(|e| e.guard);
        let ops = l.input.ops.iter().enumerate();
        ops.map(|(i, od)| (od.op.name(), guard(i))).collect()
    }

    /// Decode: the writer's two encodes after it; the attention under `UnlessCodecDecode`, its
    /// packed twin after it under `CodecDecode`; the output projection reads the twin, which
    /// takes over the attention's output and reads the writer's two encodes.
    #[test]
    fn decode_encodes_after_the_writer_and_twins_the_attention() {
        let l = expand_kv_codec(&layer(64, 1), &FACTS).expect("expands");
        let s = shape(&l);
        assert_eq!(
            s[4..11],
            [
                ("RopeRotate", None),
                ("RopeAppend", None),
                ("KvEncode", Some(Codec)),
                ("KvEncode", Some(Codec)),
                ("AttnDecode", Some(UnlessCodecDecode)),
                ("AttnPackedKv", Some(CodecDecode)),
                ("MatmulTile", None),
            ]
        );
        let id = |i: usize| l.op_expansion[i].map(|e| e.id);
        assert_eq!(id(5), id(6));
        assert_eq!(id(5), id(7));
        assert_eq!(id(8), id(9));
        assert_ne!(id(5), id(8));
        use InputRef::{Ext, Op};
        let ins = |i: usize| l.input.ops[i].inputs.clone();
        // K: its cache half (source 7) and the writer's cos; V: source 8. No bias: bare projections.
        assert_eq!(ins(6), [Ext(7), Ext(5)]);
        assert_eq!(ins(7), [Ext(8), Ext(5)]);
        assert_eq!(ins(8)[0], Op(4));
        assert_eq!(ins(9), [Op(4), Op(8), Op(6), Op(7)]);
        assert_eq!(ins(10)[0], Op(9));
        // The twin realizes the attention's tile; the encodes realize none.
        assert_eq!(l.op_tiles[9], l.op_tiles[8]);
        assert_eq!((l.op_tiles[6], l.op_tiles[7]), (None, None));
        assert_eq!(l.input.ops.len(), layer(64, 1).input.ops.len() + 3);
    }

    /// Every other form: the layer's K and V staged and the query rotated in ahead of the
    /// attention, which reads the rotated rows; its output rotated back, then the packed twin.
    #[test]
    fn prefill_stages_rotates_and_twins_the_attention() {
        let l = expand_kv_codec(&layer(64, 2), &FACTS).expect("expands");
        let s = shape(&l);
        assert_eq!(
            s[5..15],
            [
                ("RopeAppend", None),
                ("KvEncode", Some(Codec)),
                ("KvEncode", Some(Codec)),
                ("KvStage", Some(CodecNotDecode)),
                ("KvStage", Some(CodecNotDecode)),
                ("RotateRows", Some(CodecNotDecode)),
                ("AttnDecode", Some(UnlessCodecDecode)),
                ("RotateRows", Some(CodecNotDecode)),
                ("AttnPackedKv", Some(CodecDecode)),
                ("MatmulTile", None),
            ]
        );
        use InputRef::Op;
        let ins = |i: usize| l.input.ops[i].inputs.clone();
        assert_eq!((ins(8), ins(9)), (vec![Op(6)], vec![Op(7)]));
        assert_eq!(ins(10), [Op(4)]);
        assert_eq!(ins(11)[0], Op(10));
        assert_eq!(ins(12), [Op(11)]);
        assert_eq!(ins(13), [Op(10), Op(12), Op(6), Op(7)]);
        assert_eq!(ins(14)[0], Op(13));
        assert_eq!(l.op_tiles[10], l.op_tiles[4]);
        assert_eq!(l.op_tiles[12], l.op_tiles[11]);
        let ids: Vec<_> = (8..14).map(|i| l.op_expansion[i].map(|e| e.id)).collect();
        assert!(ids.iter().all(|id| id.is_some() && *id == ids[0]));
    }

    /// A writer no attention reads through the cache (an encoder's rope, attended by
    /// `EncoderAttn`) is still coded: its encodes follow it, and nothing is twinned.
    #[test]
    fn a_writer_without_a_cache_reader_is_still_encoded() {
        let mut l = layer(64, 1);
        let SubOp::AttnDecode { geom, scale, .. } = l.input.ops[6].op else {
            panic!("op 6 is the fixture's attention");
        };
        l.input.ops[6] = OpDesc {
            op: SubOp::EncoderAttn { geom, scale },
            m: 1,
            inputs: vec![InputRef::Op(4), InputRef::Op(5), InputRef::Op(3)],
        };
        let x = expand_kv_codec(&l, &FACTS).expect("expands");
        assert_eq!(
            shape(&x)[5..9],
            [
                ("RopeAppend", None),
                ("KvEncode", Some(Codec)),
                ("KvEncode", Some(Codec)),
                ("EncoderAttn", None),
            ]
        );
        assert_eq!(x.input.ops.len(), l.input.ops.len() + 2);
    }

    /// A head dim the codec's kernels do not take codes nothing: the op list comes back as it was.
    #[test]
    fn an_unsupported_head_dim_codes_nothing() {
        for (h, hd) in [(384, 96), (2048, 1024)] {
            let l = lowered(one_layer_input_shaped(h, hd, 512, hd));
            let x = expand_kv_codec(&l, &FACTS).expect("passes through");
            assert_eq!(shape(&x), shape(&l), "head dim {hd}");
        }
    }

    /// Two attention classes of different head dims (a hybrid arch): only the declared class is
    /// coded — its writer and its attention; the other class's are left as they were.
    #[test]
    fn a_hybrid_arch_codes_its_declared_class_only() {
        let mut l = layer(64, 1);
        let mut second = one_layer_input_shaped(256, 128, 512, 128);
        let (s0, o0) = (l.input.sources.len(), l.input.ops.len());
        let shift = |r: &mut InputRef| match r {
            InputRef::Op(j) => *j += o0,
            InputRef::Ext(e) => *e += s0,
        };
        for od in &mut second.ops {
            od.inputs.iter_mut().for_each(shift);
            if let SubOp::RopeAppend { attn, .. } | SubOp::AttnDecode { mask: attn, .. } =
                &mut od.op
            {
                *attn = AttnMask::SlidingWindow;
            }
        }
        l.input.sources.extend(second.sources);
        l.input.ops.extend(second.ops);
        let l = lowered(l.input);
        let x = expand_kv_codec(&l, &FACTS).expect("expands");
        let count = |name| x.input.ops.iter().filter(|od| od.op.name() == name).count();
        assert_eq!((count("KvEncode"), count("AttnPackedKv")), (2, 1));
        let coded = x.input.ops.iter().zip(&x.op_expansion);
        for (od, e) in coded.filter(|(_, e)| e.is_some()) {
            if let SubOp::RopeAppend { attn, .. } | SubOp::AttnDecode { mask: attn, .. } = od.op {
                assert_eq!(attn, AttnMask::Causal, "{e:?}");
            }
        }
    }

    /// `od` inserted at `at`, every reference to a later op moved with it.
    fn insert(input: &mut LoweringInput, at: usize, od: OpDesc) {
        for r in input.ops.iter_mut().flat_map(|o| o.inputs.iter_mut()) {
            if let InputRef::Op(j) = r
                && *j >= at
            {
                *j += 1;
            }
        }
        input.result += usize::from(input.result >= at);
        input.ops.insert(at, od);
    }

    /// A biased projection's weight rides the encode of the operand that carries it.
    #[test]
    fn an_encode_reads_its_operands_bias_projection() {
        let mut input = one_layer_input_shaped(256, 64, 512, 64);
        let bias = input.sources.len();
        input.sources.push(SourceShape { rows: 1, cols: 64 });
        // K = k_proj (op 2, weight source 3) + its bias; the rope (now op 6) appends it.
        let biased = OpDesc {
            op: SubOp::Elementwise(EwKind::BiasAdd),
            m: 1,
            inputs: vec![InputRef::Op(2), InputRef::Ext(bias)],
        };
        insert(&mut input, 4, biased);
        input.ops[6].inputs[0] = InputRef::Op(4);
        let x = expand_kv_codec(&lowered(input), &FACTS).expect("expands");
        let encodes: Vec<_> = x
            .input
            .ops
            .iter()
            .filter(|od| od.op.name() == "KvEncode")
            .collect();
        use InputRef::Ext;
        assert_eq!(encodes[0].inputs, [Ext(7), Ext(5), Ext(3)]);
        assert_eq!(encodes[1].inputs, [Ext(8), Ext(5)]);
        assert!(matches!(
            x.input.ops[2].op,
            SubOp::MatmulTile {
                weight: GemmWeight::Dense,
                ..
            }
        ));
    }

    /// A coded attention whose new K no coded writer produced has no packed store to name, and
    /// does not expand: the refusal names the attention.
    #[test]
    fn an_attention_without_its_writer_does_not_expand() {
        let mut l = layer(64, 1);
        l.input.ops[6].inputs[3] = InputRef::Op(2);
        match expand_kv_codec(&l, &FACTS) {
            Err(KvCodecError::NoWriter { op: 6, name }) => assert_eq!(name, "AttnDecode"),
            other => panic!("expected NoWriter at op 6, got {other:?}"),
        }
    }
}

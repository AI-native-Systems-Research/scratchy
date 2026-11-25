// SPDX-License-Identifier: Apache-2.0
//! THE KV CODEC'S STEPS, INSERTED FROM A TARGET'S DECLARED FACTS.
//!
//! A target that stores its KV cache compressed runs extra steps around every KV writer and every
//! attention: the writer's new K and V are encoded into a packed store, and the
//! attention either reads that store directly (its packed twin) or reads it staged back into a
//! dense cache. WHICH steps, in WHICH
//! order and under WHICH runtime guard are the target's facts ([`KvCodecFacts`]); inserting them
//! is this one pass over the front end's op list. A target without a codec never calls it.
//!
//! The inserted ops are expansion ops (`expansion_ops!`): a writer and its encodes are one
//! expansion, an attention and the steps around it another, so a target fences each as one. Every
//! later reader of a coded attention reads the end of its chain — the op its output became.

use std::collections::HashMap;

use crate::handoff::{EmbedState, Expansion, ExpansionId, LoweredDecode};
use crate::lower::{ArchOp, InputRef, LoweringInput, OpDesc, kv_operand_bias};
use crate::subtile_ir::{KvOperand, KvWriterLayout, RopeFormTag, SubOp};

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    /// The layer's `operand`, staged out of its packed store into the dense cache the attention
    /// reads.
    Stage(KvOperand),
}

/// A step after a coded attention: it takes over the attention's output.
#[derive(Clone, Copy, Debug)]
pub enum After {
    /// The attention read straight off the packed store, into the same output.
    PackedTwin,
}

/// What surrounds a coded attention of one form, and the guard the attention itself runs under.
pub struct CodecAround {
    pub before: &'static [Guarded<Before>],
    pub anchor: CodecGuard,
    pub after: &'static [Guarded<After>],
}

/// A target's KV codec, as data.
pub struct KvCodecFacts {
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
        operand: usize,
    },
    /// A writer operand whose additive offset cannot be traced to its producer.
    Offset { op: usize, why: String },
    /// A coded op that another construct already expanded to.
    AlreadyExpanded { op: usize, name: &'static str },
    /// A float8 attention whose new-token V is not the value its writer stored: it reads a V
    /// the cache never held.
    AttentionValueMismatch { op: usize, name: &'static str },
    /// A float8 KV writer whose checkpoint declares a rotation pairing the graph's form does not
    /// state: `RopeRotate` would drop it.
    InterleavedRotation { op: usize, name: &'static str },
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
            Self::InterleavedRotation { op, name } => write!(
                f,
                "op {op} ({name}): a float8 KV writer rotates through `RopeRotate`, in the graph's \
                 NeoX pairing, and this writer's checkpoint declares the interleaved one"
            ),
            Self::AttentionValueMismatch { op, name } => write!(
                f,
                "op {op} ({name}): its new-token V is not the value its KV writer stored"
            ),
        }
    }
}

/// Insert `facts`' codec steps around every coded KV writer and attention of `l`.
///
/// `attention` is the model's attention facts — a codec codes an attention's keys and values,
/// so a body it runs on always has them; a body without attention holds no coded writer and
/// comes back as it was.
pub fn expand_kv_codec(
    l: &LoweredDecode,
    facts: &KvCodecFacts,
) -> Result<LoweredDecode, KvCodecError> {
    let ops = &l.input.ops;
    // A body with no KV writer or attention (a vision tower, a recurrent-only body) comes back as
    // it was.
    let codes = ops.iter().any(|od| {
        matches!(
            od.op,
            SubOp::RopeAppend { .. } | SubOp::KvAppend { .. } | SubOp::AttnDecode { .. }
        )
    });
    if !codes {
        return Ok(l.clone());
    }
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
        // A cache writer is its own layout (`SubOp::kv_writer`), read off the op.
        if let Some(writer) = od.op.kv_writer() {
            x.writer(i, writer, facts)?;
            continue;
        }
        match od.op {
            SubOp::AttnDecode { operands, .. } => {
                let around = if od.m == 1 {
                    &facts.decode
                } else {
                    &facts.prefill
                };
                x.attention(i, operands, around)?
            }
            _ => {
                let at = x.push_own(i, l.op_expansion[i]);
                x.read_as[i] = at;
            }
        }
    }
    let at = &x.at;
    Ok(l.with_ops(
        LoweringInput {
            sources: l.input.sources.clone(),
            ops: x.ops,
            result: x.read_as[l.input.result],
            exported: l.input.exported.map(|e| x.read_as[e]),
        },
        x.tiles,
        l.norm_gain_add_tiles
            .iter()
            .map(|(op, tile)| (at[*op], *tile))
            .collect(),
        x.expansion,
        (l.unnamed_reads.iter())
            .map(|(w, r)| (at[*w], at[*r]))
            .chain(x.unnamed_reads)
            .collect(),
    ))
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

    fn operand(&self, i: usize, k: usize) -> Result<InputRef, KvCodecError> {
        let r = self.l.input.ops[i].inputs.get(k).copied();
        r.ok_or(KvCodecError::MissingOperand {
            op: i,
            name: self.name(i),
            operand: k,
        })
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

    /// A coded KV writer, then its encodes: each the cache half it filled, its rotary row (a
    /// rotating writer's), and the projection whose bias the operand carries
    /// (`kv_operand_bias`).
    fn writer(
        &mut self,
        i: usize,
        layout: crate::subtile_ir::KvWriterLayout,
        facts: &KvCodecFacts,
    ) -> Result<(), KvCodecError> {
        let id = self.open(i)?;
        let own = Some(Expansion { id, guard: None });
        let at = self.push_own(i, own);
        self.read_as[i] = at;
        let ops = &self.l.input.ops;
        let m = ops[i].m;
        let mut packed = Vec::with_capacity(facts.after_writer.len());
        for g in facts.after_writer {
            // The operand's value and the cache half it fills — the writer's own layout.
            let (value, cache) = match g.step {
                KvOperand::K => (layout.k(), layout.k_cache()),
                KvOperand::V => (layout.v(), layout.v_cache()),
            };
            let mut inputs = vec![self.operand(i, cache)?];
            // A rotating writer's encode also reads its rotation's cos row.
            if let Some((cos, _)) = layout.rotation() {
                inputs.push(self.operand(i, cos)?);
            }
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
    fn attention(
        &mut self,
        i: usize,
        layout: crate::subtile_ir::AttnLayout,
        around: &CodecAround,
    ) -> Result<(), KvCodecError> {
        let name = self.name(i);
        // The attention's slots by its own layout — minted with its inputs, so every slot it
        // names is one of them.
        let l = self.l;
        let slot = |s: usize| l.input.ops[i].inputs[s];
        // An attention over a cache it appends nothing to (its one segment the resident
        // prefix: a drafter reading its target's KV) reads a packed store no writer of this
        // forward encodes: its prefix sources stand for it. Otherwise its step segment's K is
        // its writer's.
        let packed = match layout.step_segment() {
            None => vec![
                (KvOperand::K, self.input(slot(layout.k(0)))),
                (KvOperand::V, self.input(slot(layout.v(0)))),
            ],
            Some(step) => {
                let writer = match slot(layout.k(step)) {
                    InputRef::Op(w) if self.packed.contains_key(&w) => w,
                    _ => return Err(KvCodecError::NoWriter { op: i, name }),
                };
                (self.packed[&writer].iter())
                    .map(|&(k, e)| (k, InputRef::Op(e)))
                    .collect()
            }
        };
        let packed_of = |operand: KvOperand| {
            let p = packed.iter().find(|(k, _)| *k == operand);
            p.map(|(_, e)| *e).ok_or(KvCodecError::Unencoded {
                op: i,
                name,
                operand,
            })
        };
        let (pk, pv) = (packed_of(KvOperand::K)?, packed_of(KvOperand::V)?);
        let id = self.open(i)?;
        let m = self.l.input.ops[i].m;
        let q = self.input(self.l.input.ops[i].inputs[layout.q()]);
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
            }
        }
        let anchor = Some(Expansion {
            id,
            guard: Some(around.anchor),
        });
        let at = self.push_own(i, anchor);
        self.ops[at].inputs[layout.q()] = q;
        self.unnamed_reads
            .extend(staged.into_iter().map(|s| (s, at)));
        let mut out = at;
        for g in around.after {
            let expansion = Some(Expansion {
                id,
                guard: Some(g.guard),
            });
            let (op, inputs) = match g.step {
                After::PackedTwin => {
                    // The twin IS the attention under the codec, so the attention's sinks
                    // weight (gpt-oss, the layout's trailing input) rides along. Without it
                    // the twin's weight row loses its site and the lowered decode dispatch
                    // silently drops the sinks column its dense twin still carries.
                    let mut inputs = vec![q, InputRef::Op(out), pk, pv];
                    inputs.extend(layout.sinks().map(|s| self.input(slot(s))));
                    (SubOp::AttnPackedKv, inputs)
                }
            };
            out = self.push(op, m, inputs, self.tiles[out], expansion);
        }
        self.read_as[i] = out;
        Ok(())
    }
}

/// ⭐⭐⭐ THE FLOAT8 KV CACHE'S STEPS — `KvCodec::Fp8E4m3`, the reference's `kv_cache_dtype=fp8`
/// (spyre-inference 4a7d21d, `ops/reshape_and_cache.py:20-29, 47-52`): the rope, then ONE
/// whole-tensor `quantize_kv` of the new K and of the new V, then the per-head `index_copy_` of the
/// float8 rows into the float8 pools. A rotating writer `RopeAppend(k, cos, sin, v, k_cache,
/// v_cache)` becomes `RopeRotate(k, cos, sin)`, `KvQuantize { K }` of the rotated rows,
/// `KvQuantize { V }` of `v`, and `KvAppend(kq, vq, k_cache, v_cache)`; an un-rotated writer
/// `KvAppend(k, v, ..)` gains the two quantizes ahead of it. The writer's steps are one expansion,
/// unguarded, and every reader of the writer reads the `KvAppend` it became.
///
/// `scales` is the codec fact's (`KvCodec::Fp8E4m3(scales)`); every quantize carries it.
///
/// ⭐ THE ATTENTION'S NEW-TOKEN V READS THE V THE WRITER STORED. Its new K reaches the
/// quantized rows by riding the writer (`read_as` carries every reader of the writer to the
/// `KvAppend`); its new V reaches the V projection directly — the rope's un-roped V slot is an
/// ALIAS of the projection's output, so no reader-rewiring touches it. The re-point happens
/// HERE, the one place the float8 law lives: an `AttnDecode` whose new K reads this expansion's
/// writer reads, as its new V, the V quantize that writer stored. One whose new V is any other
/// value reads a V the cache never held — refused by name, never silently half-rewired.
///
/// ⛔ NOT `KvEncode`. An encode runs AFTER its writer, under a runtime guard, packing the cache half
/// the writer filled into a SEPARATE store the attention reads instead (int8). The float8
/// cache IS the pool: its quantize runs BEFORE the writer, unguarded, over the writer's own value
/// operand, and the writer stores what the quantize produced.
///
/// ⛔ A ROTATION THE GRAPH'S FORM DOES NOT STATE IS REFUSED: `RopeRotate` rotates in the graph's own
/// pairing, so a writer whose checkpoint declares the interleaved one would lose it.
pub fn expand_kv_quantize<E: EmbedState>(
    l: &LoweredDecode<E>,
    scales: scratchy_ir::KvCacheScales,
) -> Result<LoweredDecode<E>, KvCodecError> {
    let ops = &l.input.ops;
    let mut next = (l.op_expansion.iter().flatten())
        .map(|e| e.id.0 + 1)
        .max()
        .unwrap_or(0);
    let (mut out, mut tiles, mut expansion) = (Vec::new(), Vec::new(), Vec::new());
    let mut at = vec![0; ops.len()];
    let mut read_as = vec![0; ops.len()];
    // Per ORIGINAL writer op index: the value operand it stores (remapped) and the V quantize
    // that operand became — the attention's new-token V reads the same quantized rows the
    // writer stored into the pool.
    let mut stored_v: std::collections::HashMap<usize, (InputRef, InputRef)> =
        std::collections::HashMap::new();
    for (i, od) in ops.iter().enumerate() {
        let name = od.op.name();
        let mut inputs: Vec<InputRef> = (od.inputs.iter())
            .map(|r| match *r {
                InputRef::Op(j) => InputRef::Op(read_as[j]),
                InputRef::Ext(e) => InputRef::Ext(e),
            })
            .collect();
        let operand = |k: usize| {
            inputs.get(k).copied().ok_or(KvCodecError::MissingOperand {
                op: i,
                name,
                operand: k,
            })
        };
        let (writer, [k, v, k_cache, v_cache], rotation) = match od.op {
            SubOp::RopeAppend {
                attention,
                rope,
                layer,
                attn,
                pairing,
                layout,
                ..
            } => {
                if pairing != RopeFormTag::NeoX {
                    return Err(KvCodecError::InterleavedRotation { op: i, name });
                }
                let head_dim = rope.head_dim(attention);
                let (w, (cos, sin)) = (KvWriterLayout::ROPE_APPEND, KvWriterLayout::ROPE_ROTATION);
                let rotated = out.len();
                out.push(OpDesc {
                    op: SubOp::rope_rotate(head_dim),
                    m: od.m,
                    inputs: vec![operand(w.k())?, operand(cos)?, operand(sin)?],
                });
                // The rotated K is the tile the writer's output realized.
                tiles.push(l.op_tiles[i]);
                expansion.push(None);
                let writer = SubOp::KvAppend {
                    attention,
                    head_dim,
                    layer,
                    attn,
                    layout,
                };
                // The rotated K, and the RopeAppend's own V and caches.
                let values = [
                    InputRef::Op(rotated),
                    operand(w.v())?,
                    operand(w.k_cache())?,
                    operand(w.v_cache())?,
                ];
                (writer, values, Some(rotated))
            }
            SubOp::KvAppend { .. } => {
                let w = KvWriterLayout::KV_APPEND;
                let values = [
                    operand(w.k())?,
                    operand(w.v())?,
                    operand(w.k_cache())?,
                    operand(w.v_cache())?,
                ];
                (od.op, values, None)
            }
            _ => {
                // ⭐ THE NEW-TOKEN V READS THE WRITER'S STORED ROWS (the doc above states the
                // law). The new K rides the writer through `read_as`; the new V reaches the V
                // projection directly, so it is re-pointed here — the attention whose new K
                // reads this expansion's writer reads, as its new V, the V quantize that
                // writer stored. Any other V is one the cache never held: refused by name. The
                // step segment is the attention's own layout's (an attention reading only a
                // resident cache has none, and nothing to re-point).
                if let SubOp::AttnDecode { operands, .. } = od.op
                    && let Some(step) = operands.step_segment()
                    && let InputRef::Op(w) = od.inputs[operands.k(step)]
                    && let Some(&(stored, vq)) = stored_v.get(&w)
                {
                    if inputs[operands.v(step)] == stored {
                        inputs[operands.v(step)] = vq;
                    } else {
                        return Err(KvCodecError::AttentionValueMismatch { op: i, name });
                    }
                }
                out.push(OpDesc {
                    op: od.op,
                    m: od.m,
                    inputs,
                });
                tiles.push(l.op_tiles[i]);
                expansion.push(l.op_expansion[i]);
                at[i] = out.len() - 1;
                read_as[i] = out.len() - 1;
                continue;
            }
        };
        if l.op_expansion[i].is_some() {
            return Err(KvCodecError::AlreadyExpanded { op: i, name });
        }
        let own = Some(Expansion {
            id: ExpansionId(next),
            guard: None,
        });
        next += 1;
        if let Some(r) = rotation {
            expansion[r] = own;
        }
        let mut quantize = |operand: KvOperand, x: InputRef| {
            out.push(OpDesc {
                op: SubOp::KvQuantize { operand, scales },
                m: od.m,
                inputs: vec![x],
            });
            tiles.push(None);
            expansion.push(own);
            InputRef::Op(out.len() - 1)
        };
        let (kq, vq) = (quantize(KvOperand::K, k), quantize(KvOperand::V, v));
        stored_v.insert(i, (v, vq));
        out.push(OpDesc {
            op: writer,
            m: od.m,
            inputs: vec![kq, vq, k_cache, v_cache],
        });
        tiles.push(rotation.map_or(l.op_tiles[i], |_| None));
        expansion.push(own);
        at[i] = out.len() - 1;
        read_as[i] = out.len() - 1;
    }
    Ok(l.with_ops(
        LoweringInput {
            sources: l.input.sources.clone(),
            ops: out,
            result: read_as[l.input.result],
            exported: l.input.exported.map(|e| read_as[e]),
        },
        tiles,
        (l.norm_gain_add_tiles.iter())
            .map(|(op, tile)| (at[*op], *tile))
            .collect(),
        expansion,
        (l.unnamed_reads.iter())
            .map(|(w, r)| (at[*w], at[*r]))
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{
        INT8_SHAPED_CODEC as FACTS, front_end_lowered as lowered, one_layer_attention,
        one_layer_input_shaped,
    };
    use crate::lower::GemmWeight;
    use crate::subtile_ir::{AttnMask, EwKind, SourceShape};
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

    /// An un-rotated writer (`kv_append`, Nemotron-H) codes like the rope's: its two encodes
    /// after it, each reading only the cache half it filled (no rotary row to read), the decode
    /// attention twinned onto them; off decode steps its K and V stage for the attention.
    #[test]
    fn an_unrotated_writer_encodes_its_cache_halves_alone() {
        use scratchy_ir::model_facts::HeadDim;
        let attention = one_layer_attention(256, 64, 64);
        let unrotated = |m: u32| {
            let mut input = one_layer_input_shaped(256, 64, 512, 64);
            // The query as projected; K and V written as projected.
            input.ops[4] = OpDesc {
                op: SubOp::ScalarMul { scale: 1.0 },
                m,
                inputs: vec![InputRef::Op(1)],
            };
            input.ops[5] = OpDesc {
                op: SubOp::kv_append(attention, HeadDim::new(64), 0, AttnMask::Causal),
                m,
                inputs: vec![
                    InputRef::Op(2),
                    InputRef::Op(3),
                    InputRef::Ext(7),
                    InputRef::Ext(8),
                ],
            };
            input.ops.iter_mut().for_each(|od| od.m = m);
            expand_kv_codec(&lowered(input), &FACTS).expect("expands")
        };
        use InputRef::{Ext, Op};
        let l = unrotated(1);
        assert_eq!(
            shape(&l)[4..11],
            [
                ("ScalarMul", None),
                ("KvAppend", None),
                ("KvEncode", Some(Codec)),
                ("KvEncode", Some(Codec)),
                ("AttnDecode", Some(UnlessCodecDecode)),
                ("AttnPackedKv", Some(CodecDecode)),
                ("MatmulTile", None),
            ]
        );
        let ins = |l: &LoweredDecode, i: usize| l.input.ops[i].inputs.clone();
        assert_eq!(ins(&l, 6), [Ext(7)]);
        assert_eq!(ins(&l, 7), [Ext(8)]);
        // The op registry admits the one-operand encode, or the IR validator refuses the tape.
        let bad = (l.input.ops.iter()).find(|od| !od.op.arity_ok(od.inputs.len()));
        assert!(bad.is_none(), "{bad:?}");
        assert_eq!(ins(&l, 9), [Op(4), Op(8), Op(6), Op(7)]);
        let l = unrotated(2);
        assert_eq!(
            shape(&l)[8..11],
            [
                ("KvStage", Some(CodecNotDecode)),
                ("KvStage", Some(CodecNotDecode)),
                ("AttnDecode", Some(UnlessCodecDecode)),
            ]
        );
        assert_eq!((ins(&l, 8), ins(&l, 9)), (vec![Op(6)], vec![Op(7)]));
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

    /// A sinks attention (gpt-oss: a 6th, trailing `[num_heads]` weight operand) twins with its
    /// sinks weight: the twin carries it as its 5th operand, and the encodes do NOT take it (it
    /// is the attention's, not a projection's bias). Without the carry the twin's weight row
    /// loses the sink site and the lowered decode dispatch silently drops the sink column its
    /// dense twin still carries — decode diverges from prefill and the model echoes.
    #[test]
    fn a_sinks_attention_twins_with_its_trailing_weight() {
        let mut input = one_layer_input_shaped(256, 64, 512, 64);
        input.ops.iter_mut().for_each(|od| od.m = 1);
        let sinks = input.sources.len();
        input.sources.push(SourceShape { rows: 1, cols: 4 });
        // The attention re-laid with its sinks weight, by the one assembly — its layout and its
        // inputs one statement.
        let od = &mut input.ops[6];
        let SubOp::AttnDecode { operands, .. } = &mut od.op else {
            panic!("op 6 is the fixture's attention");
        };
        let slot = |s: usize| od.inputs[s];
        let (laid, inputs) = crate::subtile_ir::AttnLayout::assemble(
            slot(operands.q()),
            (0..operands.segments()).map(|s| (slot(operands.k(s)), slot(operands.v(s)))),
            Some(InputRef::Ext(sinks)),
        )
        .expect("the two-segment attention takes a sinks weight");
        *operands = laid;
        od.inputs = inputs;
        let x = expand_kv_codec(&lowered(input), &FACTS).expect("expands");
        use InputRef::{Ext, Op};
        // The dense anchor keeps all six of its inputs under `UnlessCodecDecode`…
        assert_eq!(
            x.input.ops[8].inputs,
            [Op(4), Ext(7), Ext(8), Op(5), Op(3), Ext(sinks)]
        );
        // …and the twin carries the trailing sinks weight as its 5th operand.
        assert_eq!(
            x.input.ops[9].inputs,
            [Op(4), Op(8), Op(6), Op(7), Ext(sinks)]
        );
        // The encodes stay untouched: the sink column is not a projection bias.
        assert_eq!(x.input.ops[6].inputs, [Ext(7), Ext(5)]);
        assert_eq!(x.input.ops[7].inputs, [Ext(8), Ext(5)]);
    }

    /// Every other form: the layer's K and V staged ahead of the attention, which reads the
    /// dense cache they fill; then the packed twin.
    #[test]
    fn prefill_stages_and_twins_the_attention() {
        let l = expand_kv_codec(&layer(64, 2), &FACTS).expect("expands");
        let s = shape(&l);
        assert_eq!(
            s[5..13],
            [
                ("RopeAppend", None),
                ("KvEncode", Some(Codec)),
                ("KvEncode", Some(Codec)),
                ("KvStage", Some(CodecNotDecode)),
                ("KvStage", Some(CodecNotDecode)),
                ("AttnDecode", Some(UnlessCodecDecode)),
                ("AttnPackedKv", Some(CodecDecode)),
                ("MatmulTile", None),
            ]
        );
        use InputRef::Op;
        let ins = |i: usize| l.input.ops[i].inputs.clone();
        assert_eq!((ins(8), ins(9)), (vec![Op(6)], vec![Op(7)]));
        assert_eq!(ins(10)[0], Op(4));
        assert_eq!(ins(11), [Op(4), Op(10), Op(6), Op(7)]);
        assert_eq!(ins(12)[0], Op(11));
        assert_eq!(l.op_tiles[11], l.op_tiles[10]);
        let ids: Vec<_> = (8..12).map(|i| l.op_expansion[i].map(|e| e.id)).collect();
        assert!(ids.iter().all(|id| id.is_some() && *id == ids[0]));
    }

    /// A writer no attention reads through the cache (an encoder's rope, attended by
    /// `EncoderAttn`) is still coded: its encodes follow it, and nothing is twinned.
    #[test]
    fn a_writer_without_a_cache_reader_is_still_encoded() {
        let mut l = layer(64, 1);
        let SubOp::AttnDecode { .. } = l.input.ops[6].op else {
            panic!("op 6 is the fixture's attention");
        };
        l.input.ops[6] = OpDesc {
            op: SubOp::EncoderAttn {
                attention: one_layer_attention(256, 64, 64),
            },
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

    /// Two attention classes of different head dims (a hybrid arch): both are coded — each
    /// writer and each attention, at its own class.
    #[test]
    fn a_hybrid_arch_codes_every_class() {
        let mut l = layer(64, 1);
        let mut second = one_layer_input_shaped(512, 128, 512, 128);
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
                *attn = AttnMask::SlidingWindow {
                    window: std::num::NonZeroU32::new(1024).expect("1024 is nonzero"),
                };
            }
        }
        l.input.sources.extend(second.sources);
        l.input.ops.extend(second.ops);
        // The two classes' facts ride their own ops now: the 64-head layer's
        // writer and attention carry its uniform trio, the 128-head sliding
        // layer's its own — each op reads its class's geometry off itself, so
        // the codec still sees the two head dims (64 Causal, 128 sliding) the
        // hybrid's single threaded value used to state.
        let l = lowered(l.input);
        let x = expand_kv_codec(&l, &FACTS).expect("expands");
        let count = |name| x.input.ops.iter().filter(|od| od.op.name() == name).count();
        assert_eq!((count("KvEncode"), count("AttnPackedKv")), (4, 2));
        let coded = x.input.ops.iter().zip(&x.op_expansion);
        let classes: Vec<AttnMask> = (coded.filter(|(_, e)| e.is_some()))
            .filter_map(|(od, _)| match od.op {
                SubOp::AttnDecode { mask, .. } => Some(mask),
                _ => None,
            })
            .collect();
        assert_eq!(classes.len(), 2);
        assert!(classes.contains(&AttnMask::Causal));
        assert!(
            classes
                .iter()
                .any(|c| matches!(c, AttnMask::SlidingWindow { .. }))
        );
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

    /// ⭐ THE FLOAT8 CACHE'S STEPS — MATCH. A rotating writer becomes its rotation, the two
    /// whole-tensor quantizes and the un-rotated writer of their rows: one expansion, unguarded; the
    /// attention reads the writer it became, and the registry admits every op.
    #[test]
    fn a_float8_cache_quantizes_ahead_of_its_writer() {
        use InputRef::Op;
        let l = layer(64, 1);
        let writer = l.input.ops[5].clone();
        let x = expand_kv_quantize(&l, scratchy_ir::KvCacheScales::Unscaled).expect("expands");
        assert_eq!(
            shape(&x)[4..11],
            [
                ("RopeRotate", None),
                ("RopeRotate", None),
                ("KvQuantize", None),
                ("KvQuantize", None),
                ("KvAppend", None),
                ("AttnDecode", None),
                ("MatmulTile", None),
            ]
        );
        let ins = |i: usize| x.input.ops[i].inputs.clone();
        assert_eq!(ins(5), writer.inputs[..3].to_vec());
        assert_eq!(ins(6), [Op(5)]);
        assert_eq!(ins(7), [writer.inputs[3]]);
        assert_eq!(ins(8), [Op(6), Op(7), writer.inputs[4], writer.inputs[5]]);
        assert_eq!(ins(9)[3], Op(8), "the attention reads the writer");
        assert_eq!(
            ins(9)[4], Op(7),
            "the attention's new V reads the V quantize the writer stored"
        );
        assert!(
            (6..=7).all(|i| matches!(
                x.input.ops[i].op,
                SubOp::KvQuantize {
                    scales: scratchy_ir::KvCacheScales::Unscaled,
                    ..
                }
            )),
            "every quantize carries the codec's scales"
        );
        let id = |i: usize| x.op_expansion[i].map(|e| e.id);
        assert!(id(5).is_some() && (6..=8).all(|i| id(i) == id(5)));
        assert_eq!(id(9), None);
        let bad = (x.input.ops.iter()).find(|od| !od.op.arity_ok(od.inputs.len()));
        assert!(bad.is_none(), "{bad:?}");
    }

    /// An un-rotated writer (`kv_append`, Nemotron-H) gains the two quantizes ahead of it, and
    /// stores their rows.
    #[test]
    fn a_float8_cache_quantizes_an_unrotated_writers_rows() {
        use crate::subtile_ir::AttnMask;
        use InputRef::{Ext, Op};
        use scratchy_ir::model_facts::HeadDim;
        let attention = one_layer_attention(256, 64, 64);
        let mut input = one_layer_input_shaped(256, 64, 512, 64);
        input.ops[4] = OpDesc {
            op: SubOp::ScalarMul { scale: 1.0 },
            m: 1,
            inputs: vec![Op(1)],
        };
        input.ops[5] = OpDesc {
            op: SubOp::kv_append(attention, HeadDim::new(64), 0, AttnMask::Causal),
            m: 1,
            inputs: vec![Op(2), Op(3), Ext(7), Ext(8)],
        };
        let x = expand_kv_quantize(&lowered(input), scratchy_ir::KvCacheScales::Unscaled)
            .expect("expands");
        assert_eq!(
            shape(&x)[4..10],
            [
                ("ScalarMul", None),
                ("KvQuantize", None),
                ("KvQuantize", None),
                ("KvAppend", None),
                ("AttnDecode", None),
                ("MatmulTile", None),
            ]
        );
        let ins = |i: usize| x.input.ops[i].inputs.clone();
        assert_eq!((ins(5), ins(6)), (vec![Op(2)], vec![Op(3)]));
        assert_eq!(ins(7), [Op(5), Op(6), Ext(7), Ext(8)]);
        assert_eq!(ins(8)[3], Op(7));
        assert_eq!(
            ins(8)[4], Op(6),
            "the attention's new V reads the V quantize the writer stored"
        );
    }

    /// NEAR-MISS — a float8 attention whose new V is not its writer's value operand reads a V
    /// the cache never held: the expansion refuses it by name rather than re-point it blindly.
    #[test]
    fn a_float8_cache_refuses_an_attention_reading_a_v_its_writer_never_stored() {
        let mut l = layer(64, 1);
        // The attention's new V is the writer's operand (op 3); point it at the Q rope instead.
        l.input.ops[6].inputs[4] = InputRef::Op(4);
        match expand_kv_quantize(&l, scratchy_ir::KvCacheScales::Unscaled) {
            Err(KvCodecError::AttentionValueMismatch { op: 6, name }) => {
                assert_eq!(name, "AttnDecode")
            }
            other => panic!("expected AttentionValueMismatch at op 6, got {other:?}"),
        }
    }

    /// NEAR-MISS — a writer whose checkpoint declares the interleaved pairing: `RopeRotate` rotates
    /// in the graph's own form, so the expansion refuses it by name rather than drop the pairing.
    #[test]
    fn a_float8_cache_refuses_an_interleaved_rotation() {
        let mut l = layer(64, 1);
        if let SubOp::RopeAppend { pairing, .. } = &mut l.input.ops[5].op {
            *pairing = crate::subtile_ir::RopeFormTag::Interleaved;
        }
        match expand_kv_quantize(&l, scratchy_ir::KvCacheScales::Unscaled) {
            Err(KvCodecError::InterleavedRotation { op: 5, name }) => {
                assert_eq!(name, "RopeAppend")
            }
            other => panic!("expected InterleavedRotation at op 5, got {other:?}"),
        }
    }
}

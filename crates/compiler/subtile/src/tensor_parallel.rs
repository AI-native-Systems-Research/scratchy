// SPDX-License-Identifier: Apache-2.0
//! TENSOR PARALLELISM BY ANALYSIS — where a forward's collectives go, derived from its dataflow.
//!
//! A front end states the single-rank forward. [`shard`] rewrites it into the forward EVERY rank
//! of a `W`-rank group runs (one program, SPMD: ranks differ only in the bytes their sources
//! hold), plus the slice law each source is loaded by. It reads two things and nothing else:
//! the dataflow (`InputRef` edges) and each op's sharding rule ([`TpRule`], one registry column
//! in [`crate::ops`]). There is no weight-name matching and no per-arch table, and no collective
//! is placed by hand.
//!
//! # The lattice
//!
//! Every value is in one [`Placement`]:
//! - [`Placement::Whole`] — every rank holds all of it, bit-identical;
//! - [`Placement::Split`] — rank `r` holds block `r` of `W` equal blocks of one axis;
//! - [`Placement::Partial`] — every rank holds a same-shaped addend; the value is their sum.
//!
//! Each op's rule maps its operands' placements to its output's, and DEMANDS a placement where
//! it cannot take what it was given. A demand that is not met is exactly where a collective
//! goes: `Partial → Whole` is an [`SubOp::AllReduce`], `Split(cols) → Whole` an
//! [`SubOp::AllGather`]. A `Whole → Split` move needs no collective, so it never appears.
//!
//! # Why this produces Megatron without being told
//!
//! A contraction over a whole activation splits its weight on N (each rank computes its block
//! of output channels); over a column-split activation it splits its weight on K (each rank
//! contracts its block of the reduced axis, so the result is partial). Channel-local ops carry a
//! split through; head-local ops carry it when it falls on head boundaries. So q/k/v and
//! gate/up come out N-split, attention and the MLP activation run on local heads and channels,
//! o/down come out K-split, and the first consumer that needs the whole value — the residual add
//! — gets the all-reduce. The vocabulary projection N-splits and the forward's result demands a
//! whole value, so the lm head gets the all-gather.
//!
//! # Invariants the output carries
//!
//! - At `W = 1` [`shard`] is the identity: same ops, same sources, no collective.
//! - The forward is SPMD: every rank runs `ShardedForward::local` verbatim.
//! - Every [`Placement::Whole`] value is bit-identical on every rank: an all-reduce sums in
//!   ABSOLUTE RANK ORDER ([`eval_ranks`] is the reference every target's lowering answers to).
//! - What the analysis cannot shard is replicated, never refused. The two refusals
//!   ([`ShardRefusal`]) are inputs no forward should contain.

use std::num::NonZeroU32;

use crate::lower::{GemmWeight, InputRef, LoweringInput, OpDesc};
use crate::model_geometry::{HeadDim, KvHeads, ModelAttnGeometry, QueryHeads};
use crate::subtile_ir::{RopeForm, RowScale, SourceShape, SubOp, SubtileIR, TensorShape};

/// The most ranks a group can name: the width of the rank bitmask the card runtime tracks a
/// group's members in (`flex_rs::multi_device::BitmaskHelper`, 64 bits).
pub const MAX_RANKS: u32 = 64;

/// The number of ranks in a tensor-parallel group.
///
/// ⛔ MINTED FROM A CONST GENERIC, NEVER A RUNTIME INTEGER. A world size is a property of the
/// compiled forward (the shapes the bake lowered are `1/W` of the model's), so it is a type, and
/// a group too big for the runtime's rank mask is a build failure rather than a launch failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorldSize(NonZeroU32);

impl WorldSize {
    /// The world of `W` ranks. `W = 0` and `W > MAX_RANKS` do not compile.
    pub const fn of<const W: u32>() -> Self {
        const {
            assert!(W >= 1, "a tensor-parallel world has at least one rank");
            assert!(
                W <= MAX_RANKS,
                "a world wider than the runtime's 64-bit rank mask"
            );
        }
        match NonZeroU32::new(W) {
            Some(w) => Self(w),
            None => unreachable!(),
        }
    }

    pub const fn get(self) -> u32 {
        self.0.get()
    }

    /// Every rank of this world, in absolute order.
    pub fn ranks(self) -> impl Iterator<Item = RankIndex> {
        (0..self.get()).map(RankIndex)
    }
}

/// A rank of some [`WorldSize`] — minted only by [`WorldSize::ranks`], so it is in range by
/// construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RankIndex(u32);

impl RankIndex {
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// The smallest column block a target can split a tensor at — a TARGET FACT the shared pass takes
/// as input (spyre: one 64-element stick; a target with no layout granule: 1). A split whose
/// per-rank extent is not a whole number of granules is not taken: the op is replicated instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitGranule(NonZeroU32);

impl SplitGranule {
    pub const fn of<const G: u32>() -> Self {
        const { assert!(G >= 1, "a split granule is at least one element") };
        match NonZeroU32::new(G) {
            Some(g) => Self(g),
            None => unreachable!(),
        }
    }

    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

/// An op's sharding rule — one registry column ([`SubOp::tp_rule`]). See [`crate::ops`] for the
/// class each variant is spelled as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TpRule {
    /// Operand 0 contracted against the gemm weight's K rows.
    Contract,
    /// Channel-local and linear in every activation operand.
    Linear,
    /// Channel-local, not linear.
    Pointwise,
    /// Reduces over each row's whole column extent.
    Rowwise,
    /// Local to an attention head.
    Heads,
    /// Re-lays rows against columns (the per-head view).
    View,
    /// Inserted by [`shard`]; never stated by a front end.
    Collective,
    /// No rule: whole operands, whole output, replicated weights.
    Whole,
}

/// Which axis of a `[rows, cols]` value a [`Placement::Split`] cuts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    /// `W` contiguous column blocks.
    Cols,
    /// A per-head view's rows, `heads` per token: rank `r` holds head block `r` of EVERY token.
    /// The head count rides along because only a view back over the same `heads` turns this
    /// into a column split again.
    HeadRows { heads: NonZeroU32 },
}

/// Where a value lives across the group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Placement {
    Whole,
    Split(Axis),
    Partial,
}

/// How a SOURCE (a weight, a cache, a staged input) is loaded on each rank — the slice law
/// [`slice_source`] applies, decided by the analysis from the source's consumers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SourceShard {
    /// Every rank loads all of it.
    Whole,
    /// Rank `r` loads row block `r` — a K-split gemm weight, a per-channel scale of an N-split one.
    Rows,
    /// Rank `r` loads column block `r` — an N-split gemm weight, a head-split KV cache.
    Cols,
    /// Rank `r` loads head block `r` of every token's `heads` rows.
    HeadRows { heads: NonZeroU32 },
}

impl SourceShard {
    fn of_split(axis: Axis) -> Self {
        match axis {
            Axis::Cols => Self::Cols,
            Axis::HeadRows { heads } => Self::HeadRows { heads },
        }
    }

    /// The source's per-rank shape.
    pub fn local_shape(self, s: SourceShape, w: WorldSize) -> SourceShape {
        let w = w.get();
        match self {
            Self::Whole => s,
            Self::Rows | Self::HeadRows { .. } => SourceShape {
                rows: s.rows / w,
                cols: s.cols,
            },
            Self::Cols => SourceShape {
                rows: s.rows,
                cols: s.cols / w,
            },
        }
    }
}

/// Which collective a [`CollectiveSite`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollectiveKind {
    AllReduce,
    AllGather,
}

/// One collective the analysis wove in: the local op it is, and the local op whose output it
/// reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollectiveSite {
    pub op: usize,
    pub reads: usize,
    pub kind: CollectiveKind,
}

/// The two inputs [`shard`] refuses. Both are defects of the INPUT, not shapes the analysis
/// could not shard (those are replicated).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShardRefusal {
    /// The front end stated a collective. Only the analysis places them; a stated one would be a
    /// second, unchecked authority on where they go.
    StatedCollective { op: usize },
    /// A head-row split reaches a consumer that needs the whole value without passing back
    /// through its view. Gathering it would need a strided all-gather no target lowers, and no
    /// arch states this shape.
    HeadRowsEscape { op: usize },
}

/// The forward every rank runs, and how each rank loads its sources.
#[derive(Clone, Debug)]
pub struct ShardedForward {
    pub world: WorldSize,
    /// The rank-local forward. Identical on every rank.
    pub local: LoweringInput,
    /// Per source of the input forward (same indices): its slice law.
    pub sources: Vec<SourceShard>,
    /// `op_map[i]` = the local op computing the input forward's op `i` (in its placement).
    pub op_map: Vec<usize>,
    /// Every collective woven in, in local op order.
    pub collectives: Vec<CollectiveSite>,
    /// Per op of the input forward: the placement its local op produces.
    pub placements: Vec<Placement>,
}

/// The analysis's working state. Everything it decides is written exactly once.
struct Weave<'a> {
    input: &'a LoweringInput,
    w: u32,
    world: WorldSize,
    granule: u32,
    /// Global `(rows, cols)` of every input op's output.
    shape: Vec<(u32, u32)>,
    place: Vec<Placement>,
    src: Vec<Option<SourceShard>>,
    ops: Vec<OpDesc>,
    op_map: Vec<usize>,
    /// The whole-placed local op standing for input op `j`, once a collective made one.
    whole_of: Vec<Option<usize>>,
    collectives: Vec<CollectiveSite>,
}

/// What one op's rule decided: the local op to emit, its output placement, its operands, and
/// the source slices it commits.
struct Decision {
    op: SubOp<crate::subtile_ir::NeoX, crate::subtile_ir::Arch>,
    out: Placement,
    inputs: Vec<InputRef>,
    adopt: Vec<(usize, SourceShard)>,
}

/// How a non-voting operand of a channel-local op takes the placement its voters agreed on.
enum Follow {
    /// An op output as wide as the output: it is one of the voters.
    Votes,
    /// A source, loaded by this slice.
    Adopts(usize, SourceShard),
    /// Cannot take the placement: the op is replicated instead.
    Blocks,
}

impl Weave<'_> {
    fn src_shape(&self, e: usize) -> (u32, u32) {
        let s = self.input.sources[e];
        (s.rows, s.cols)
    }

    fn operand_shape(&self, r: InputRef) -> (u32, u32) {
        match r {
            InputRef::Op(j) => self.shape[j],
            InputRef::Ext(e) => self.src_shape(e),
        }
    }

    /// Whether `extent` splits into `W` blocks that are each a whole number of granules.
    fn splits(&self, extent: u32) -> bool {
        extent.is_multiple_of(self.w * self.granule)
    }

    /// Whether source `e` can be loaded as `want` — unset, or already that.
    fn can_adopt(&self, e: usize, want: SourceShard) -> bool {
        let fits = match want {
            SourceShard::Whole => true,
            SourceShard::Rows => self.input.sources[e].rows.is_multiple_of(self.w),
            SourceShard::Cols => self.input.sources[e].cols.is_multiple_of(self.w),
            SourceShard::HeadRows { heads } => {
                heads.get().is_multiple_of(self.w)
                    && self.input.sources[e].rows.is_multiple_of(heads.get())
            }
        };
        fits && self.src[e].is_none_or(|s| s == want)
    }

    /// The local operand standing for `r` in a WHOLE placement — the edge where the analysis
    /// weaves a collective in, once per value however many consumers demand it.
    fn whole(&mut self, r: InputRef) -> Result<InputRef, ShardRefusal> {
        let InputRef::Op(j) = r else {
            return Ok(r);
        };
        let (kind, op) = match self.place[j] {
            Placement::Whole => return Ok(InputRef::Op(self.op_map[j])),
            Placement::Partial => (CollectiveKind::AllReduce, SubOp::AllReduce),
            Placement::Split(Axis::Cols) => (
                CollectiveKind::AllGather,
                SubOp::AllGather {
                    cols: self.shape[j].1,
                },
            ),
            Placement::Split(Axis::HeadRows { .. }) => {
                return Err(ShardRefusal::HeadRowsEscape { op: j });
            }
        };
        if let Some(k) = self.whole_of[j] {
            return Ok(InputRef::Op(k));
        }
        let reads = self.op_map[j];
        let at = self.ops.len();
        self.ops.push(OpDesc {
            op,
            m: self.input.ops[j].m,
            inputs: vec![InputRef::Op(reads)],
        });
        self.collectives.push(CollectiveSite {
            op: at,
            reads,
            kind,
        });
        self.whole_of[j] = Some(at);
        Ok(InputRef::Op(at))
    }

    /// The local operand standing for `r` as it already is (its producer's placement).
    fn as_is(&self, r: InputRef) -> InputRef {
        match r {
            InputRef::Op(j) => InputRef::Op(self.op_map[j]),
            ext => ext,
        }
    }

    /// The decision that replicates op `i`: whole operands, whole sources, whole output.
    fn replicate(&mut self, i: usize) -> Result<Decision, ShardRefusal> {
        let desc = &self.input.ops[i];
        let (op, refs) = (desc.op, desc.inputs.clone());
        let mut inputs = Vec::with_capacity(refs.len());
        let mut adopt = Vec::new();
        for r in refs {
            if let InputRef::Ext(e) = r {
                adopt.push((e, SourceShard::Whole));
            }
            inputs.push(self.whole(r)?);
        }
        Ok(Decision {
            op,
            out: Placement::Whole,
            inputs,
            adopt,
        })
    }

    /// Whole sources adopted by everything but the listed operand positions.
    fn ext_whole_except(&self, i: usize, keep: &[usize]) -> Vec<(usize, SourceShard)> {
        self.input.ops[i]
            .inputs
            .iter()
            .enumerate()
            .filter(|(k, _)| !keep.contains(k))
            .filter_map(|(_, r)| match r {
                InputRef::Ext(e) => Some((*e, SourceShard::Whole)),
                InputRef::Op(_) => None,
            })
            .collect()
    }

    fn decide(&mut self, i: usize) -> Result<Decision, ShardRefusal> {
        let desc = self.input.ops[i].clone();
        match desc.op.tp_rule() {
            TpRule::Collective => Err(ShardRefusal::StatedCollective { op: i }),
            TpRule::Whole => self.replicate(i),
            TpRule::Contract => self.contract(i, &desc),
            TpRule::Linear => self.channel_local(i, &desc, true),
            TpRule::Pointwise => self.channel_local(i, &desc, false),
            TpRule::Rowwise => self.rowwise(i, &desc),
            TpRule::Heads => self.heads(i, &desc),
            TpRule::View => self.view(i, &desc),
        }
    }

    /// `[act, W]` or `[act, W, w_scale]`, `W` a `[K, N]` source.
    fn contract(&mut self, i: usize, desc: &OpDesc) -> Result<Decision, ShardRefusal> {
        let SubOp::MatmulTile { n, weight } = desc.op else {
            return self.replicate(i);
        };
        let (act, InputRef::Ext(we)) = (desc.inputs[0], desc.inputs[1]) else {
            return self.replicate(i);
        };
        let scale = desc.inputs.get(2).copied();
        let k = self.operand_shape(act).1;
        let act_place = match act {
            InputRef::Op(j) => self.place[j],
            InputRef::Ext(_) => Placement::Whole,
        };
        // The per-channel scale follows N along whichever of its axes is N long.
        let scale_on_n = |me: &Self| -> Option<(usize, SourceShard)> {
            let InputRef::Ext(se) = scale? else {
                return None;
            };
            let (sr, sc) = me.src_shape(se);
            let shard = if sr == n && sc != n {
                SourceShard::Rows
            } else {
                SourceShard::Cols
            };
            Some((se, shard))
        };
        let group_ok = |kk: u32| match weight {
            GemmWeight::Affine { affine } => kk.is_multiple_of(affine.group().get()),
            GemmWeight::Dense | GemmWeight::Fp8Dynamic => true,
        };
        match act_place {
            // A split reduced axis: each rank contracts its K block — the result is partial.
            Placement::Split(Axis::Cols)
                if self.can_adopt(we, SourceShard::Rows)
                    && group_ok(k / self.w)
                    && scale.is_none_or(|s| match s {
                        InputRef::Ext(se) => self.can_adopt(se, SourceShard::Whole),
                        InputRef::Op(_) => false,
                    }) =>
            {
                let mut adopt = vec![(we, SourceShard::Rows)];
                if let Some(InputRef::Ext(se)) = scale {
                    adopt.push((se, SourceShard::Whole));
                }
                Ok(Decision {
                    op: desc.op,
                    out: Placement::Partial,
                    inputs: desc.inputs.iter().map(|r| self.as_is(*r)).collect(),
                    adopt,
                })
            }
            // Split rows contract independently against the whole weight.
            Placement::Split(Axis::HeadRows { heads })
                if self
                    .ext_whole_except(i, &[0])
                    .iter()
                    .all(|(e, s)| self.can_adopt(*e, *s)) =>
            {
                Ok(Decision {
                    op: desc.op,
                    out: Placement::Split(Axis::HeadRows { heads }),
                    inputs: desc.inputs.iter().map(|r| self.as_is(*r)).collect(),
                    adopt: self.ext_whole_except(i, &[0]),
                })
            }
            _ => {
                let act_whole = self.whole(act)?;
                let scale_shard = scale_on_n(self);
                let n_split = self.splits(n)
                    && self.can_adopt(we, SourceShard::Cols)
                    && scale_shard.is_none_or(|(se, s)| self.can_adopt(se, s));
                let mut inputs = vec![act_whole, desc.inputs[1]];
                if let Some(s) = scale {
                    inputs.push(s);
                }
                if n_split {
                    let mut adopt = vec![(we, SourceShard::Cols)];
                    adopt.extend(scale_shard);
                    Ok(Decision {
                        op: SubOp::MatmulTile {
                            n: n / self.w,
                            weight,
                        },
                        out: Placement::Split(Axis::Cols),
                        inputs,
                        adopt,
                    })
                } else {
                    Ok(Decision {
                        op: desc.op,
                        out: Placement::Whole,
                        inputs,
                        adopt: self.ext_whole_except(i, &[]),
                    })
                }
            }
        }
    }

    /// The channel-local rules. Op operands as wide as the output VOTE for its placement;
    /// every other operand must [`Follow`] it. `linear` admits a partial sum — but only when every
    /// source operand is a scalar multiplier, since a partial plus a whole addend counts the
    /// addend `W` times.
    fn channel_local(
        &mut self,
        i: usize,
        desc: &OpDesc,
        linear: bool,
    ) -> Result<Decision, ShardRefusal> {
        let out = self.shape[i];
        let voters: Vec<Placement> = desc
            .inputs
            .iter()
            .filter_map(|r| match r {
                InputRef::Op(j) if self.shape[*j] == out => Some(self.place[*j]),
                _ => None,
            })
            .collect();
        let agreed = voters
            .first()
            .copied()
            .filter(|p| voters.iter().all(|q| q == p))
            .filter(|p| match p {
                Placement::Whole => false,
                Placement::Partial => linear,
                Placement::Split(_) => true,
            });
        let carried = agreed.and_then(|p| {
            let mut adopt = Vec::new();
            for r in &desc.inputs {
                match self.follow(p, out, *r) {
                    Follow::Votes => {}
                    Follow::Adopts(e, s) if self.can_adopt(e, s) => adopt.push((e, s)),
                    Follow::Adopts(..) | Follow::Blocks => return None,
                }
            }
            Some((p, adopt))
        });
        match carried {
            Some((p, adopt)) => Ok(Decision {
                op: desc.op,
                out: p,
                inputs: desc.inputs.iter().map(|r| self.as_is(*r)).collect(),
                adopt,
            }),
            None => self.replicate(i),
        }
    }

    /// How operand `r` of a channel-local op whose output is `out` takes the placement `p`.
    fn follow(&self, p: Placement, out: (u32, u32), r: InputRef) -> Follow {
        match r {
            InputRef::Op(j) if self.shape[j] == out => Follow::Votes,
            // A narrower op operand (a broadcast row or column) would have to be whole, and no
            // non-whole placement is carried past a whole broadcast operand here.
            InputRef::Op(_) => Follow::Blocks,
            InputRef::Ext(e) => {
                let (rows, cols) = self.src_shape(e);
                match p {
                    Placement::Whole => Follow::Adopts(e, SourceShard::Whole),
                    Placement::Partial if rows == 1 && cols == 1 => {
                        Follow::Adopts(e, SourceShard::Whole)
                    }
                    Placement::Partial => Follow::Blocks,
                    Placement::Split(Axis::Cols) if cols == out.1 => {
                        Follow::Adopts(e, SourceShard::Cols)
                    }
                    Placement::Split(Axis::Cols) if cols == 1 => {
                        Follow::Adopts(e, SourceShard::Whole)
                    }
                    Placement::Split(axis @ Axis::HeadRows { .. }) if rows == out.0 => {
                        Follow::Adopts(e, SourceShard::of_split(axis))
                    }
                    Placement::Split(Axis::HeadRows { .. }) if rows == 1 => {
                        Follow::Adopts(e, SourceShard::Whole)
                    }
                    Placement::Split(_) => Follow::Blocks,
                }
            }
        }
    }

    /// A row reduction keeps a row split (rows are independent) and needs whole rows otherwise.
    fn rowwise(&mut self, i: usize, desc: &OpDesc) -> Result<Decision, ShardRefusal> {
        let p = match desc.inputs.first() {
            Some(InputRef::Op(j)) => self.place[*j],
            _ => Placement::Whole,
        };
        if let Placement::Split(axis @ Axis::HeadRows { .. }) = p {
            let rest: Vec<(usize, SourceShard)> = self.ext_whole_except(i, &[0]);
            if rest.iter().all(|(e, s)| self.can_adopt(*e, *s)) {
                return Ok(Decision {
                    op: desc.op,
                    out: Placement::Split(axis),
                    inputs: desc.inputs.iter().map(|r| self.as_is(*r)).collect(),
                    adopt: rest,
                });
            }
        }
        self.replicate(i)
    }

    /// The head-local rules. Rotary and the KV writers carry a column split that falls on head
    /// boundaries; attention carries one when BOTH its query and its kv heads split `W` ways, so
    /// each rank's query heads read exactly that rank's kv heads.
    fn heads(&mut self, i: usize, desc: &OpDesc) -> Result<Decision, ShardRefusal> {
        // The operands that split by heads (op outputs must already be head-split; sources
        // adopt a column split), the head dim the split must fall on, the local op, and the
        // attention sinks' own slice (one sink per query head) when there are sinks.
        // ⭐ A ROTARY TABLE SPLITS WITH ITS ACTIVATION WHEN IT IS AS WIDE AS IT. `cos`/`sin` are
        // either one head's `[·, head_dim]` table, broadcast over every head (whole on every
        // rank), or that table already repeated across the heads (`[·, heads · head_dim]`, as
        // wide as the rotated activation), in which case each rank's heads read their own
        // columns of it. The shape says which; a whole wide table against a split activation is
        // a rope whose views disagree about the width.
        let tables_as_wide = |me: &Self| -> Vec<usize> {
            let width = me.operand_shape(desc.inputs[0]).1;
            [1, 2]
                .into_iter()
                .filter(|k| {
                    desc.inputs
                        .get(*k)
                        .is_some_and(|r| me.operand_shape(*r).1 == width)
                })
                .collect()
        };
        let (split, hd, op, sinks): (Vec<usize>, u32, _, Option<(usize, SourceShard)>) = match desc
            .op
        {
            SubOp::RopeRotate { head_dim, .. } => {
                let mut split = vec![0];
                split.extend(tables_as_wide(self));
                (split, head_dim.get(), desc.op, None)
            }
            // `[K, cos, sin, V, K_cache, V_cache]` (a legacy fixture may stop at V).
            SubOp::RopeAppend { head_dim, .. } => {
                let mut split: Vec<usize> = [0, 3, 4, 5]
                    .into_iter()
                    .filter(|k| *k < desc.inputs.len())
                    .collect();
                split.extend(tables_as_wide(self));
                (split, head_dim.get(), desc.op, None)
            }
            // `[K, V, K_cache, V_cache]`.
            SubOp::KvAppend { head_dim, .. } => (vec![0, 1, 2, 3], head_dim.get(), desc.op, None),
            SubOp::AttnDecode {
                geom,
                scale,
                valid_len,
                mask,
                ..
            } => {
                let (nq, nkv) = (geom.nqh().get(), geom.nkvh().get());
                let local = (nq.is_multiple_of(self.w) && nkv.is_multiple_of(self.w))
                    .then(|| {
                        ModelAttnGeometry::mint(
                            QueryHeads::new(nq / self.w),
                            KvHeads::new(nkv / self.w),
                            HeadDim::new(geom.hd().get()),
                        )
                    })
                    .flatten();
                let Some(local) = local else {
                    return self.replicate(i);
                };
                // `[q, (k, v)…]`, or `[q, pk, pv, nk, nv, sinks]`.
                let n = desc.inputs.len();
                let sinks = if n == 6 {
                    let InputRef::Ext(se) = desc.inputs[5] else {
                        return self.replicate(i);
                    };
                    let (sr, sc) = self.src_shape(se);
                    let shard = match (sr == nq, sc == nq) {
                        (_, true) => SourceShard::Cols,
                        (true, false) => SourceShard::Rows,
                        (false, false) => return self.replicate(i),
                    };
                    Some((se, shard))
                } else {
                    None
                };
                let segs = if sinks.is_some() { 5 } else { n };
                let op = SubOp::attn_decode(local, scale, valid_len, mask);
                ((0..segs).collect(), geom.hd().get(), op, sinks)
            }
            _ => return self.replicate(i),
        };
        let head_split = |me: &Self, r: InputRef| -> bool {
            match r {
                InputRef::Op(j) => {
                    me.place[j] == Placement::Split(Axis::Cols)
                        && (me.shape[j].1 / me.w).is_multiple_of(hd)
                }
                InputRef::Ext(e) => {
                    me.can_adopt(e, SourceShard::Cols)
                        && (me.src_shape(e).1 / me.w).is_multiple_of(hd)
                }
            }
        };
        if !split.iter().all(|k| head_split(self, desc.inputs[*k])) {
            return self.replicate(i);
        }
        let mut adopt: Vec<(usize, SourceShard)> = sinks.into_iter().collect();
        for (k, r) in desc.inputs.iter().enumerate() {
            let InputRef::Ext(e) = *r else { continue };
            if sinks.is_some_and(|(se, _)| se == e) {
                continue;
            }
            let shard = if split.contains(&k) {
                SourceShard::Cols
            } else {
                SourceShard::Whole
            };
            adopt.push((e, shard));
        }
        if !adopt.iter().all(|(e, s)| self.can_adopt(*e, *s)) {
            return self.replicate(i);
        }
        Ok(Decision {
            op,
            out: Placement::Split(Axis::Cols),
            inputs: desc.inputs.iter().map(|r| self.as_is(*r)).collect(),
            adopt,
        })
    }

    /// The per-head view: `[m, heads·c] → [m·heads, c]` turns a head-aligned column split into a
    /// head-row split, and the view back over the same `heads` turns it into a column split again.
    /// A view is linear, so a partial passes through unchanged.
    fn view(&mut self, i: usize, desc: &OpDesc) -> Result<Decision, ShardRefusal> {
        let SubOp::Reshape { rows, cols } = desc.op else {
            return self.replicate(i);
        };
        let Some(InputRef::Op(j)) = desc.inputs.first().copied() else {
            return self.replicate(i);
        };
        let in_cols = self.shape[j].1;
        let w = self.w;
        let local = |rows, cols| SubOp::Reshape { rows, cols };
        let decided = match (self.place[j], rows) {
            (Placement::Partial, _) => Some((desc.op, Placement::Partial)),
            (Placement::Split(Axis::Cols), RowScale::Times(heads))
                if in_cols == heads.get() * cols && heads.get().is_multiple_of(w) =>
            {
                NonZeroU32::new(heads.get() / w).map(|lh| {
                    (
                        local(RowScale::Times(lh), cols),
                        Placement::Split(Axis::HeadRows { heads }),
                    )
                })
            }
            (Placement::Split(Axis::HeadRows { heads }), RowScale::Times(one))
                if one.get() == 1 && cols == heads.get() * in_cols =>
            {
                Some((local(rows, cols / w), Placement::Split(Axis::Cols)))
            }
            _ => None,
        };
        match decided {
            Some((op, out)) => Ok(Decision {
                op,
                out,
                inputs: vec![self.as_is(desc.inputs[0])],
                adopt: vec![],
            }),
            None => self.replicate(i),
        }
    }
}

/// Every op's output `(rows, cols)`, through the registry exactly as `lower_region` derives them —
/// the shapes the analysis reasons over (on the input) and the per-rank shapes a target sizes its
/// collectives by (on [`ShardedForward::local`]).
pub fn op_shapes(input: &LoweringInput) -> Vec<(u32, u32)> {
    let mut shape: Vec<(u32, u32)> = Vec::with_capacity(input.ops.len());
    for desc in &input.ops {
        let dims = |r: InputRef, shape: &[(u32, u32)]| match r {
            InputRef::Op(j) => shape[j],
            InputRef::Ext(e) => (input.sources[e].rows, input.sources[e].cols),
        };
        let cols = desc.op.out_cols(|k| dims(desc.inputs[k], &shape).1);
        let rows = desc.op.out_rows(desc.m, |k| dims(desc.inputs[k], &shape).0);
        shape.push((rows, cols));
    }
    shape
}

/// Weave tensor parallelism into `input`: the forward every rank of `world` runs, each source's
/// slice law, and every collective the dataflow demands. See the module docs.
pub fn shard(
    input: &LoweringInput,
    world: WorldSize,
    granule: SplitGranule,
) -> Result<ShardedForward, ShardRefusal> {
    let n = input.ops.len();
    let shape = op_shapes(input);
    let mut wv = Weave {
        input,
        w: world.get(),
        world,
        granule: granule.get(),
        shape,
        place: Vec::with_capacity(n),
        src: vec![None; input.sources.len()],
        ops: Vec::with_capacity(n),
        op_map: Vec::with_capacity(n),
        whole_of: vec![None; n],
        collectives: Vec::new(),
    };
    for i in 0..n {
        let d = if wv.w == 1 {
            // ⭐ THE IDENTITY AT W = 1, by construction rather than by every rule happening to
            // agree: a one-rank world has nothing to split and nothing to collect.
            if input.ops[i].op.tp_rule() == TpRule::Collective {
                return Err(ShardRefusal::StatedCollective { op: i });
            }
            Decision {
                op: input.ops[i].op,
                out: Placement::Whole,
                inputs: input.ops[i].inputs.clone(),
                adopt: vec![],
            }
        } else {
            wv.decide(i)?
        };
        for (e, s) in d.adopt {
            // `can_adopt` was every rule's gate, so a commit never overwrites a different slice.
            debug_assert!(
                wv.src[e].is_none_or(|p| p == s),
                "source {e} sliced two ways"
            );
            wv.src[e] = Some(s);
        }
        wv.op_map.push(wv.ops.len());
        wv.ops.push(OpDesc {
            op: d.op,
            m: input.ops[i].m,
            inputs: d.inputs,
        });
        wv.place.push(d.out);
    }
    // The forward's result and exported hidden state are read by the host: whole.
    let result = match wv.whole(InputRef::Op(input.result))? {
        InputRef::Op(k) => k,
        InputRef::Ext(_) => unreachable!(),
    };
    let exported = match input.exported {
        Some(e) => match wv.whole(InputRef::Op(e))? {
            InputRef::Op(k) => Some(k),
            InputRef::Ext(_) => unreachable!(),
        },
        None => None,
    };
    let sources: Vec<SourceShard> = wv
        .src
        .iter()
        .map(|s| s.unwrap_or(SourceShard::Whole))
        .collect();
    let local_sources = input
        .sources
        .iter()
        .zip(&sources)
        .map(|(s, sh)| sh.local_shape(*s, world))
        .collect();
    Ok(ShardedForward {
        world: wv.world,
        local: LoweringInput {
            sources: local_sources,
            ops: wv.ops,
            result,
            exported,
        },
        sources,
        op_map: wv.op_map,
        collectives: wv.collectives,
        placements: wv.place,
    })
}

/// [`shard`] over a whole compiler→target handoff: the sharded forward, plus every fact the handoff
/// keeps parallel to the forward's ops carried onto the op that now computes it. A collective the
/// analysis inserted realizes no FUF tile and belongs to no expansion, so it carries neither.
///
/// Sources keep their indices (only their per-rank shapes change), so the source bindings carry over
/// unchanged; how each one is sliced is [`ShardedForward::sources`].
pub fn shard_lowered(
    lowered: &crate::handoff::LoweredDecode,
    world: WorldSize,
    granule: SplitGranule,
) -> Result<(crate::handoff::LoweredDecode, ShardedForward), ShardRefusal> {
    let sharded = shard(&lowered.input, world, granule)?;
    let n_local = sharded.local.ops.len();
    let at = |i: usize| sharded.op_map[i];
    let mut op_tiles = vec![None; n_local];
    let mut op_expansion = vec![None; n_local];
    for (i, (tile, exp)) in lowered
        .op_tiles
        .iter()
        .zip(&lowered.op_expansion)
        .enumerate()
    {
        op_tiles[at(i)] = *tile;
        op_expansion[at(i)] = *exp;
    }
    let out = crate::handoff::LoweredDecode {
        input: sharded.local.clone(),
        bindings: lowered.bindings.clone(),
        op_tiles,
        norm_gain_add_tiles: lowered
            .norm_gain_add_tiles
            .iter()
            .map(|(&i, &tile)| (at(i), tile))
            .collect(),
        op_expansion,
        unnamed_reads: lowered
            .unnamed_reads
            .iter()
            .map(|&(w, r)| (at(w), at(r)))
            .collect(),
    };
    Ok((out, sharded))
}

/// THE SLICE LAW: rank `rank`'s share of a source of `shape` under `shard`. One function, read
/// by the loader that stages a rank's weights and by [`eval_ranks`]' reference alike.
pub fn slice_source<T: Copy>(
    full: &[T],
    shape: SourceShape,
    shard: SourceShard,
    rank: RankIndex,
    world: WorldSize,
) -> Vec<T> {
    let (rows, cols) = (shape.rows as usize, shape.cols as usize);
    let (w, r) = (world.get() as usize, rank.get() as usize);
    debug_assert_eq!(full.len(), rows * cols, "source buffer vs its shape");
    match shard {
        SourceShard::Whole => full.to_vec(),
        SourceShard::Rows => {
            let lr = rows / w;
            full[r * lr * cols..(r + 1) * lr * cols].to_vec()
        }
        SourceShard::Cols => {
            let lc = cols / w;
            (0..rows)
                .flat_map(|i| {
                    full[i * cols + r * lc..i * cols + (r + 1) * lc]
                        .iter()
                        .copied()
                })
                .collect()
        }
        SourceShard::HeadRows { heads } => {
            let h = heads.get() as usize;
            let lh = h / w;
            (0..rows / h)
                .flat_map(|t| {
                    let first = t * h + r * lh;
                    full[first * cols..(first + lh) * cols].iter().copied()
                })
                .collect()
        }
    }
}

/// ⭐ WHAT A COLLECTIVE MEANS — its value on every rank, given every rank's buffers in ABSOLUTE
/// RANK ORDER (`ranks[r]` is rank `r`'s). An [`SubOp::AllReduce`] sums its operand over the ranks
/// in that order; an [`SubOp::AllGather`] concatenates their column blocks in that order. The one
/// law [`eval_ranks`] and any lockstep reference over a rank's own buffers read.
///
/// # Panics
/// On a node that is not a collective.
pub fn eval_collective<F: RopeForm>(
    node: &crate::subtile_ir::SubtileNode<F>,
    graph: &SubtileIR<F>,
    ranks: &[&[Vec<f32>]],
) -> Vec<f32> {
    use crate::subtile_ir::gather;
    match node.op {
        SubOp::AllReduce => {
            let parts: Vec<Vec<f32>> = ranks
                .iter()
                .map(|b| gather(&node.inputs[0], graph, b).0)
                .collect();
            let mut sum = parts[0].clone();
            for p in &parts[1..] {
                for (a, x) in sum.iter_mut().zip(p) {
                    *a += *x;
                }
            }
            sum
        }
        SubOp::AllGather { .. } => {
            let parts: Vec<(Vec<f32>, u32, u32)> = ranks
                .iter()
                .map(|b| gather(&node.inputs[0], graph, b))
                .collect();
            let (rows, lc) = (parts[0].1 as usize, parts[0].2 as usize);
            let mut all = Vec::with_capacity(rows * lc * ranks.len());
            for i in 0..rows {
                for p in &parts {
                    all.extend_from_slice(&p.0[i * lc..(i + 1) * lc]);
                }
            }
            all
        }
        _ => panic!("{} is not a collective", node.op.name()),
    }
}

/// Evaluate a sharded forward's local graph on every rank IN LOCKSTEP — the host reference for
/// what the collectives mean. `sources[r]` is rank `r`'s source buffers (already sliced). Every
/// non-collective node runs per rank through [`crate::subtile_ir::eval_node`]; an
/// [`SubOp::AllReduce`] sums its operand over ranks in ABSOLUTE RANK ORDER and an
/// [`SubOp::AllGather`] concatenates column blocks in rank order, and both write the same bytes
/// to every rank. Returns every rank's buffers.
pub fn eval_ranks<F: RopeForm>(
    graph: &SubtileIR<F>,
    world: WorldSize,
    sources: &[Vec<Vec<f32>>],
) -> Vec<Vec<Vec<f32>>> {
    use crate::subtile_ir::{eval_node, scatter};
    let w = world.get() as usize;
    assert_eq!(sources.len(), w, "one source set per rank");
    let mut bufs: Vec<Vec<Vec<f32>>> = sources
        .iter()
        .map(|srcs| {
            let mut b: Vec<Vec<f32>> = graph
                .tensors
                .iter()
                .map(|t| vec![0f32; (t.rows * t.cols) as usize])
                .collect();
            for (s, src) in srcs.iter().enumerate() {
                b[s].copy_from_slice(src);
            }
            b
        })
        .collect();
    for node in &graph.nodes {
        let shape: TensorShape = graph.shape(node.output.tensor);
        let outs: Vec<Vec<f32>> = match node.op {
            SubOp::AllReduce | SubOp::AllGather { .. } => {
                let ranks: Vec<&[Vec<f32>]> = bufs.iter().map(|b| b.as_slice()).collect();
                vec![eval_collective(node, graph, &ranks); w]
            }
            _ => bufs.iter().map(|b| eval_node(node, graph, b)).collect(),
        };
        for (b, out) in bufs.iter_mut().zip(outs) {
            scatter(b, node.output.tensor, node.output.region, &out, shape);
        }
    }
    bufs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::one_layer_input;
    use crate::subtile_ir::{eval_dag, lower_region};

    const NB: NonZeroU32 = NonZeroU32::new(128).unwrap();

    /// Deterministic, non-degenerate source data (a fixed LCG; no RNG dependency).
    fn fill(len: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 9) as f32 / (1u32 << 23) as f32 - 0.5) * 0.25
            })
            .collect()
    }

    fn sources_of(input: &LoweringInput) -> Vec<Vec<f32>> {
        input
            .sources
            .iter()
            .enumerate()
            .map(|(e, s)| fill((s.rows * s.cols) as usize, e as u32))
            .collect()
    }

    fn kinds(sf: &ShardedForward) -> Vec<CollectiveKind> {
        sf.collectives.iter().map(|c| c.kind).collect()
    }

    /// One rank has nothing to split and nothing to collect: the forward comes back as it went
    /// in, op for op, and every source loads whole.
    #[test]
    fn one_rank_is_the_identity() {
        let input = one_layer_input();
        let sf = shard(&input, WorldSize::of::<1>(), SplitGranule::of::<64>()).unwrap();
        assert!(sf.collectives.is_empty());
        assert!(sf.sources.iter().all(|s| *s == SourceShard::Whole));
        assert_eq!(sf.local.ops.len(), input.ops.len());
        assert_eq!(format!("{:?}", sf.local), format!("{input:?}"));
    }

    /// ⭐ MEGATRON, DERIVED. Nothing told the analysis which projection is which: q/k/v and
    /// gate/up come out column-split, o/down row-split, the KV caches head-split, the norms
    /// replicated — and exactly ONE collective per sublayer, an all-reduce, lands on each
    /// residual add's partial operand.
    #[test]
    fn a_decode_layer_shards_as_megatron_from_the_dataflow_alone() {
        let input = one_layer_input();
        let sf = shard(&input, WorldSize::of::<4>(), SplitGranule::of::<64>()).unwrap();
        use SourceShard::*;
        // Sources: x, rms0, q, k, v, cos, sin, kc, vc, o, rms1, gate, up, down.
        assert_eq!(
            sf.sources,
            vec![
                Whole, Whole, Cols, Cols, Cols, Whole, Whole, Cols, Cols, Rows, Whole, Cols, Cols,
                Rows
            ]
        );
        assert_eq!(
            kinds(&sf),
            vec![CollectiveKind::AllReduce, CollectiveKind::AllReduce]
        );
        // Each all-reduce reads a K-split projection and feeds a residual add.
        for c in &sf.collectives {
            assert!(matches!(sf.local.ops[c.reads].op, SubOp::MatmulTile { .. }));
            let feeds_add = sf.local.ops.iter().any(|o| {
                matches!(o.op, SubOp::Elementwise(crate::subtile_ir::EwKind::Add))
                    && o.inputs.contains(&InputRef::Op(c.op))
            });
            assert!(
                feeds_add,
                "an all-reduce at {} that no residual add reads",
                c.op
            );
        }
        assert_eq!(sf.placements[input.result], Placement::Whole);
    }

    /// A tensor-parallel forward computes the same function: every rank's result equals the
    /// unsharded forward's, to f32 reassociation — and every rank's result is BIT-IDENTICAL to
    /// every other's, because the all-reduce sums in absolute rank order everywhere.
    fn sharded_equals_unsharded(w: WorldSize) {
        let input = one_layer_input();
        let full = sources_of(&input);
        let reference = {
            let g = lower_region(&input, NB);
            let refs: Vec<&[f32]> = full.iter().map(|v| v.as_slice()).collect();
            let bufs = eval_dag(&g, &refs);
            bufs[g.result.index()].clone()
        };
        let sf = shard(&input, w, SplitGranule::of::<64>()).unwrap();
        let g = lower_region(&sf.local, NB);
        let per_rank: Vec<Vec<Vec<f32>>> = w
            .ranks()
            .map(|r| {
                full.iter()
                    .zip(&input.sources)
                    .zip(&sf.sources)
                    .map(|((buf, shape), shard)| slice_source(buf, *shape, *shard, r, w))
                    .collect()
            })
            .collect();
        let bufs = eval_ranks(&g, w, &per_rank);
        let first = &bufs[0][g.result.index()];
        assert_eq!(first.len(), reference.len());
        for (r, b) in bufs.iter().enumerate() {
            assert_eq!(
                &b[g.result.index()],
                first,
                "rank {r}'s whole result differs from rank 0's"
            );
        }
        let worst = first
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst < 1e-4, "sharded result off by {worst}");
    }

    #[test]
    fn two_ranks_compute_the_unsharded_forward() {
        sharded_equals_unsharded(WorldSize::of::<2>());
    }

    #[test]
    fn four_ranks_compute_the_unsharded_forward() {
        sharded_equals_unsharded(WorldSize::of::<4>());
    }

    /// Eight ranks: 32 query heads / 8 kv heads still split, one kv head per rank.
    #[test]
    fn eight_ranks_compute_the_unsharded_forward() {
        sharded_equals_unsharded(WorldSize::of::<8>());
    }

    /// Sixteen ranks cannot split 8 kv heads: attention is replicated instead (its q/k/v
    /// gathered), and the forward is still the unsharded one. An unsplittable op costs
    /// collectives, never correctness.
    #[test]
    fn an_unsplittable_attention_is_replicated_not_refused() {
        let input = one_layer_input();
        let sf = shard(&input, WorldSize::of::<16>(), SplitGranule::of::<1>()).unwrap();
        assert!(kinds(&sf).contains(&CollectiveKind::AllGather));
        sharded_equals_unsharded_at(&input, WorldSize::of::<16>(), SplitGranule::of::<1>());
    }

    fn sharded_equals_unsharded_at(input: &LoweringInput, w: WorldSize, g: SplitGranule) {
        let full = sources_of(input);
        let gr = lower_region(input, NB);
        let refs: Vec<&[f32]> = full.iter().map(|v| v.as_slice()).collect();
        let reference = eval_dag(&gr, &refs)[gr.result.index()].clone();
        let sf = shard(input, w, g).unwrap();
        let gl = lower_region(&sf.local, NB);
        let per_rank: Vec<Vec<Vec<f32>>> = w
            .ranks()
            .map(|r| {
                full.iter()
                    .zip(&input.sources)
                    .zip(&sf.sources)
                    .map(|((buf, shape), shard)| slice_source(buf, *shape, *shard, r, w))
                    .collect()
            })
            .collect();
        let got = &eval_ranks(&gl, w, &per_rank)[0][gl.result.index()];
        let worst = got
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst < 1e-4, "sharded result off by {worst}");
    }

    /// The handoff's op-parallel facts follow their ops: at one rank the handoff comes back as it
    /// went in; at two, each original op's tile lands on the op that now computes it, and each
    /// inserted collective realizes no tile and belongs to no expansion.
    #[test]
    fn the_handoff_facts_follow_their_ops() {
        use crate::fixtures::front_end_lowered;
        let lowered = front_end_lowered(one_layer_input());
        let (one, _) =
            shard_lowered(&lowered, WorldSize::of::<1>(), SplitGranule::of::<64>()).unwrap();
        assert_eq!(format!("{one:?}"), format!("{lowered:?}"));

        let (two, sf) =
            shard_lowered(&lowered, WorldSize::of::<2>(), SplitGranule::of::<64>()).unwrap();
        assert_eq!(two.op_tiles.len(), two.input.ops.len());
        assert_eq!(two.op_expansion.len(), two.input.ops.len());
        for (i, &k) in sf.op_map.iter().enumerate() {
            assert_eq!(
                two.op_tiles[k], lowered.op_tiles[i],
                "op {i}'s tile moved to {k}"
            );
        }
        for c in &sf.collectives {
            assert_eq!(two.op_tiles[c.op], None);
            assert_eq!(two.op_expansion[c.op], None);
        }
        assert_eq!(
            two.norm_gain_add_tiles.get(&sf.op_map[0]),
            lowered.norm_gain_add_tiles.get(&0)
        );
    }

    /// A front end that states a collective is refused: placement is the analysis's alone.
    #[test]
    fn a_stated_collective_is_refused() {
        let mut input = one_layer_input();
        let last = input.ops.len() - 1;
        input.ops.push(OpDesc {
            op: SubOp::AllReduce,
            m: 1,
            inputs: vec![InputRef::Op(last)],
        });
        input.result = last + 1;
        assert!(matches!(
            shard(&input, WorldSize::of::<2>(), SplitGranule::of::<64>()),
            Err(ShardRefusal::StatedCollective { .. })
        ));
    }
}

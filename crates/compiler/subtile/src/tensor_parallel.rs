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
//! # The padded block
//!
//! A split is even — `W` equal blocks of whole granules — everywhere but one place: the
//! vocabulary projection that computes the forward's result (through its logit epilogues), whose
//! width need not divide (granite's 49155). There each rank holds `ceil(N / W)` columns rounded up
//! to whole granules ([`split_block`]), the last rank's block zero-padded
//! ([`SourceShard::ColBlocks`]). Padding is taken only there because the result's one reader is
//! the all-gather that crops it off ([`SubOp::AllGather`]'s `cols` is the real width): no op ever
//! reads a padded column as a value.
//!
//! # Gathered rows
//!
//! A front end may state a source as ROWS GATHERED FROM ANOTHER SOURCE's columns ([`RowGather`]):
//! an embedding lookup whose indices the forward does not carry (a decode step's input row, read
//! out of the tied embedding table at the token the previous step drew). A gather is a contraction
//! with a one-hot vector over the table's columns, so the contraction rule applies to it: when the
//! table splits by columns, each rank can gather only the columns it holds — zero for an index
//! outside its block — and the row is a [`Placement::Partial`] sum. Its first consumer that needs
//! the whole row gets the all-reduce, exactly as a K-split projection's does.
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
use crate::subtile_ir::{RopeForm, RowScale, SourceShape, SubOp, SubtileIR, TensorShape};

/// ⭐ THE WORLD IS A TYPE — the one `tp/N` the build names ([`TpWorld<N>`]). Every function here
/// that a world shapes is generic over it: [`shard`]'s per-rank forward, [`slice_source`]'s slice
/// law, [`eval_ranks`]' lockstep reference. Nothing stores a world size, and no op carries one.
pub use scratchy_tp_world::{Build, PerRank, Rank, TpWorld, World, ranks};

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
    /// Every rank holds a same-shaped addend and the value is their sum — a [`RowGather`]'s rows
    /// over a column-split table. Loaded from a whole value, rank 0 holds it and every other rank
    /// zeros, so the sum is the value.
    Partial,
    /// Rank `r` loads columns `[r·block, (r+1)·block)` — zeros past the source's width: the
    /// padded block of an N-split whose width does not divide ([`split_block`]).
    ColBlocks { block: NonZeroU32 },
    /// Rank `r` loads rows `[r·block, (r+1)·block)` — zeros past the source's height: a padded
    /// split's per-channel scale, and its gemm weight's transposed (on-disk) form.
    RowBlocks { block: NonZeroU32 },
}

impl SourceShard {
    fn of_split(axis: Axis) -> Self {
        match axis {
            Axis::Cols => Self::Cols,
            Axis::HeadRows { heads } => Self::HeadRows { heads },
        }
    }

    /// The source's per-rank shape.
    pub fn local_shape<T: World>(self, s: SourceShape) -> SourceShape {
        let w = T::RANKS;
        match self {
            Self::Whole | Self::Partial => s,
            Self::Rows | Self::HeadRows { .. } => SourceShape {
                rows: s.rows / w,
                cols: s.cols,
            },
            Self::Cols => SourceShape {
                rows: s.rows,
                cols: s.cols / w,
            },
            Self::ColBlocks { block } => SourceShape {
                rows: s.rows,
                cols: block.get(),
            },
            Self::RowBlocks { block } => SourceShape {
                rows: block.get(),
                cols: s.cols,
            },
        }
    }
}

/// ⭐ THE PADDED BLOCK LAW — each rank's share of an `extent` split `W` ways when `extent` does not
/// divide into `W` blocks of whole granules: `ceil(extent / W)` rounded up to whole granules, so
/// every rank's block is a layout the target addresses and the blocks cover the extent, the last
/// one zero-padded. `None` when the law leaves nothing to split (one block covers the extent).
pub fn split_block<T: World>(extent: u32, granule: SplitGranule) -> Option<NonZeroU32> {
    let block = extent.div_ceil(T::RANKS).next_multiple_of(granule.get());
    NonZeroU32::new(block).filter(|b| b.get() < extent)
}

/// Which collective a [`CollectiveSite`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollectiveKind {
    AllReduce,
    AllGather,
}

/// One collective the analysis wove in: the local op it is, and the value it reads — a local op's
/// output, or a partial source ([`SourceShard::Partial`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollectiveSite {
    pub op: usize,
    pub reads: InputRef,
    pub kind: CollectiveKind,
}

/// ⭐ A SOURCE THAT IS ANOTHER SOURCE'S COLUMNS, GATHERED — stated by the front end, because the
/// indices are not in the forward: row `i` of `row` (`[m, k]`) is column `idx_i` of `table`
/// (`[k, n]`), for indices something outside the forward chose (the host's prompt tokens, or the
/// previous step's device pick). See the module docs: over a column-split table it is a partial
/// sum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowGather {
    pub row: usize,
    pub table: usize,
}

/// The inputs [`shard`] refuses. All are defects of the INPUT, not shapes the analysis could not
/// shard (those are replicated).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShardRefusal {
    /// The front end stated a collective. Only the analysis places them; a stated one would be a
    /// second, unchecked authority on where they go.
    StatedCollective { op: usize },
    /// A head-row split reaches a consumer that needs the whole value without passing back
    /// through its view. Gathering it would need a strided all-gather no target lowers, and no
    /// arch states this shape.
    HeadRowsEscape { op: usize },
    /// A [`RowGather`] whose sources are not a `[m, k]` row over a `[k, n]` table, or whose row
    /// is gathered twice.
    MalformedGather { gather: RowGather },
}

/// The forward every rank runs, and how each rank loads its sources.
#[derive(Clone, Debug)]
pub struct ShardedForward<T: World> {
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
    /// The stated [`RowGather`]s whose table split by columns — each one's row is
    /// [`SourceShard::Partial`], and whoever fills it on a rank gathers only that rank's columns.
    pub gathered: Vec<RowGather>,
    /// The world every rank of which runs [`Self::local`].
    pub world: core::marker::PhantomData<T>,
}

/// The analysis's working state. Everything it decides is written exactly once.
struct Weave<'a, T: World> {
    input: &'a LoweringInput,
    /// The world the forward is sharded for — every split count is `T::RANKS`.
    world: core::marker::PhantomData<T>,
    granule: u32,
    /// Global `(rows, cols)` of every input op's output.
    shape: Vec<(u32, u32)>,
    place: Vec<Placement>,
    src: Vec<Option<SourceShard>>,
    ops: Vec<OpDesc>,
    op_map: Vec<usize>,
    /// The whole-placed local op standing for input op `j`, once a collective made one.
    whole_of: Vec<Option<usize>>,
    /// The same, for a partial source.
    whole_of_src: Vec<Option<usize>>,
    collectives: Vec<CollectiveSite>,
    /// Per input op: its split is a padded block ([`SourceShard::ColBlocks`]) — readable only by
    /// the ops that carry it to the result.
    padded: Vec<bool>,
    /// The vocabulary projection computing the forward's result, when it ends with one: the one
    /// contraction whose N-split may pad.
    result_projection: Option<usize>,
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

impl<T: World> Weave<'_, T> {
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

    /// Where operand `r` lives: its producer's placement, or a source's — whole unless it is a
    /// gathered row's partial sum.
    fn place_of(&self, r: InputRef) -> Placement {
        match r {
            InputRef::Op(j) => self.place[j],
            InputRef::Ext(e) if self.src[e] == Some(SourceShard::Partial) => Placement::Partial,
            InputRef::Ext(_) => Placement::Whole,
        }
    }

    /// Whether `extent` splits into `W` blocks that are each a whole number of granules.
    fn splits(&self, extent: u32) -> bool {
        extent.is_multiple_of(T::RANKS * self.granule)
    }

    /// Whether source `e` can be loaded as `want` — unset, or already that.
    fn can_adopt(&self, e: usize, want: SourceShard) -> bool {
        let fits = match want {
            SourceShard::Whole => true,
            SourceShard::Rows => self.input.sources[e].rows.is_multiple_of(T::RANKS),
            SourceShard::Cols => self.input.sources[e].cols.is_multiple_of(T::RANKS),
            SourceShard::HeadRows { heads } => {
                heads.get().is_multiple_of(T::RANKS)
                    && self.input.sources[e].rows.is_multiple_of(heads.get())
            }
            SourceShard::Partial
            | SourceShard::ColBlocks { .. }
            | SourceShard::RowBlocks { .. } => true,
        };
        fits && self.src[e].is_none_or(|s| s == want)
    }

    /// How contraction `i`'s `n` output columns split, when they do: the local width and the
    /// weight's and a per-channel scale's slice laws (`(cols law, rows law)`). Even everywhere; the
    /// padded block ([`split_block`]) only for the result's projection.
    fn n_split(&self, i: usize, n: u32) -> Option<(u32, SourceShard, SourceShard)> {
        if self.splits(n) {
            return Some((n / T::RANKS, SourceShard::Cols, SourceShard::Rows));
        }
        if self.result_projection != Some(i) {
            return None;
        }
        let block = split_block::<T>(n, SplitGranule(NonZeroU32::new(self.granule)?))?;
        Some((
            block.get(),
            SourceShard::ColBlocks { block },
            SourceShard::RowBlocks { block },
        ))
    }

    /// The local operand standing for `r` in a WHOLE placement — the edge where the analysis
    /// weaves a collective in, once per value however many consumers demand it. A source is
    /// loaded whole by its slice law unless it is a gathered row's partial sum, which is reduced
    /// here exactly as a K-split projection's output is.
    fn whole(&mut self, r: InputRef) -> Result<InputRef, ShardRefusal> {
        match r {
            InputRef::Op(j) => self.whole_op(j).map(InputRef::Op),
            InputRef::Ext(e) if self.place_of(r) == Placement::Partial => {
                if let Some(k) = self.whole_of_src[e] {
                    return Ok(InputRef::Op(k));
                }
                let at = self.weave(
                    SubOp::AllReduce,
                    CollectiveKind::AllReduce,
                    r,
                    self.input.sources[e].rows,
                );
                self.whole_of_src[e] = Some(at);
                Ok(InputRef::Op(at))
            }
            InputRef::Ext(_) => Ok(r),
        }
    }

    /// Push collective `op` reading the local value `reads` (`m` rows) and record its site.
    fn weave(
        &mut self,
        op: SubOp<crate::subtile_ir::NeoX, crate::subtile_ir::Arch>,
        kind: CollectiveKind,
        reads: InputRef,
        m: u32,
    ) -> usize {
        let at = self.ops.len();
        self.ops.push(OpDesc {
            op,
            m,
            inputs: vec![reads],
        });
        self.collectives.push(CollectiveSite {
            op: at,
            reads,
            kind,
        });
        at
    }

    /// [`Self::whole`] of input op `j`'s output: the local op computing it whole — its own local
    /// op when it is already whole, else the collective woven in to make it so.
    fn whole_op(&mut self, j: usize) -> Result<usize, ShardRefusal> {
        let (kind, op) = match self.place[j] {
            Placement::Whole => return Ok(self.op_map[j]),
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
            return Ok(k);
        }
        let reads = InputRef::Op(self.op_map[j]);
        let at = self.weave(op, kind, reads, self.input.ops[j].m);
        self.whole_of[j] = Some(at);
        Ok(at)
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
        for r in refs {
            inputs.push(self.whole(r)?);
        }
        let adopt = Self::whole_sources(&inputs);
        Ok(Decision {
            op,
            out: Placement::Whole,
            inputs,
            adopt,
        })
    }

    /// The sources a decision reads as they are, loaded whole. A partial source never appears:
    /// [`Self::whole`] already stood its all-reduce in for it.
    fn whole_sources(inputs: &[InputRef]) -> Vec<(usize, SourceShard)> {
        inputs
            .iter()
            .filter_map(|r| match r {
                InputRef::Ext(e) => Some((*e, SourceShard::Whole)),
                InputRef::Op(_) => None,
            })
            .collect()
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
        let act_place = self.place_of(act);
        // The per-channel scale follows N along whichever of its axes is N long.
        let split = self.n_split(i, n);
        let scale_on_n = |me: &Self| -> Option<(usize, SourceShard)> {
            let InputRef::Ext(se) = scale? else {
                return None;
            };
            let (_, cols_law, rows_law) = split?;
            let (sr, sc) = me.src_shape(se);
            let shard = if sr == n && sc != n {
                rows_law
            } else {
                cols_law
            };
            Some((se, shard))
        };
        let act_padded = matches!(act, InputRef::Op(j) if self.padded[j]);
        let group_ok = |kk: u32| match weight {
            GemmWeight::Affine { affine } => kk.is_multiple_of(affine.group().get()),
            GemmWeight::Dense | GemmWeight::Fp8Dynamic => true,
        };
        match act_place {
            // A split reduced axis: each rank contracts its K block — the result is partial.
            Placement::Split(Axis::Cols)
                if !act_padded
                    && self.can_adopt(we, SourceShard::Rows)
                    && group_ok(k / T::RANKS)
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
                let n_split = split.filter(|&(_, cols_law, _)| {
                    self.can_adopt(we, cols_law)
                        && scale_shard.is_none_or(|(se, s)| self.can_adopt(se, s))
                });
                let mut inputs = vec![act_whole, desc.inputs[1]];
                if let Some(s) = scale {
                    inputs.push(s);
                }
                if let Some((local_n, cols_law, _)) = n_split {
                    let mut adopt = vec![(we, cols_law)];
                    adopt.extend(scale_shard);
                    Ok(Decision {
                        op: SubOp::MatmulTile { n: local_n, weight },
                        out: Placement::Split(Axis::Cols),
                        inputs,
                        adopt,
                    })
                } else {
                    let adopt = Self::whole_sources(&inputs);
                    Ok(Decision {
                        op: desc.op,
                        out: Placement::Whole,
                        inputs,
                        adopt,
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
        // A padded value's sources would need its padded block, and a padded column is never a
        // value: it is carried only through source-free ops (the logit epilogues).
        let padded = desc
            .inputs
            .iter()
            .any(|r| matches!(r, InputRef::Op(j) if self.padded[*j]));
        let carried = agreed.and_then(|p| {
            let mut adopt = Vec::new();
            for r in &desc.inputs {
                match self.follow(p, out, *r) {
                    Follow::Votes => {}
                    Follow::Adopts(e, s) if !padded && self.can_adopt(e, s) => adopt.push((e, s)),
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
            // A partial source votes for nothing: its whole form is an all-reduce.
            InputRef::Ext(_) if self.place_of(r) == Placement::Partial => Follow::Blocks,
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
        let (split, hd, op, sinks): (Vec<usize>, u32, _, Option<(usize, SourceShard)>) =
            match desc.op {
                SubOp::RopeRotate { head_dim, .. } => {
                    let mut split = vec![0];
                    split.extend(tables_as_wide(self));
                    (split, head_dim.get(), desc.op, None)
                }
                // `[K, cos, sin, V, K_cache, V_cache]` (a legacy fixture may stop at V). The op carries
                // the model's attention facts, so the rank-local op carries the rank's share of them.
                SubOp::RopeAppend {
                    attention,
                    rope,
                    layer,
                    attn,
                    pairing,
                    layout,
                    _form,
                } => {
                    let Some(local) = attention.rank_local::<T>() else {
                        return self.replicate(i);
                    };
                    let mut split: Vec<usize> = [0, 3, 4, 5]
                        .into_iter()
                        .filter(|k| *k < desc.inputs.len())
                        .collect();
                    split.extend(tables_as_wide(self));
                    let op = SubOp::RopeAppend {
                        attention: local,
                        rope,
                        layer,
                        attn,
                        pairing,
                        layout,
                        _form,
                    };
                    (split, rope.head_dim(attention).get(), op, None)
                }
                // `[K, V, K_cache, V_cache]`.
                SubOp::KvAppend {
                    attention,
                    head_dim,
                    layer,
                    attn,
                    layout,
                } => {
                    let Some(local) = attention.rank_local::<T>() else {
                        return self.replicate(i);
                    };
                    let op = SubOp::KvAppend {
                        attention: local,
                        head_dim,
                        layer,
                        attn,
                        layout,
                    };
                    (vec![0, 1, 2, 3], head_dim.get(), op, None)
                }
                SubOp::AttnDecode {
                    attention,
                    valid_len,
                    mask,
                    ..
                } => {
                    let geom = attention.class(mask.class());
                    let nq = geom.q_heads().get();
                    let Some(local) = attention.rank_local::<T>() else {
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
                    let op = SubOp::attn_decode(local, valid_len, mask);
                    ((0..segs).collect(), geom.head_dim().get(), op, sinks)
                }
                _ => return self.replicate(i),
            };
        let head_split = |me: &Self, r: InputRef| -> bool {
            match r {
                InputRef::Op(j) => {
                    me.place[j] == Placement::Split(Axis::Cols)
                        && !me.padded[j]
                        && (me.shape[j].1 / T::RANKS).is_multiple_of(hd)
                }
                InputRef::Ext(e) => {
                    me.can_adopt(e, SourceShard::Cols)
                        && (me.src_shape(e).1 / T::RANKS).is_multiple_of(hd)
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
        let w = T::RANKS;
        let local = |rows, cols| SubOp::Reshape { rows, cols };
        let decided = match (self.place[j], rows) {
            (Placement::Partial, _) => Some((desc.op, Placement::Partial)),
            (Placement::Split(Axis::Cols), RowScale::Times(heads))
                if !self.padded[j]
                    && in_cols == heads.get() * cols
                    && heads.get().is_multiple_of(w) =>
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

/// The vocabulary projection computing `input`'s result — an arity-2 contraction under
/// single-input logit epilogues (a scale, a softcap) — as an op index.
fn result_projection(input: &LoweringInput) -> Option<usize> {
    let table = input.vocab_table()?;
    let mut at = input.result;
    loop {
        let od = &input.ops[at];
        match (od.op, od.inputs.as_slice()) {
            (SubOp::MatmulTile { .. }, &[_, InputRef::Ext(w)]) if w == table => return Some(at),
            (SubOp::ScalarMul { .. } | SubOp::TanhSoftCap { .. }, &[InputRef::Op(j)]) => at = j,
            _ => return None,
        }
    }
}

/// Weave tensor parallelism into `input`: the forward every rank of world `T` runs, each source's
/// slice law, and every collective the dataflow demands. `gathers` are the sources the front end
/// states as gathered rows ([`RowGather`]). See the module docs.
pub fn shard<T: World>(
    input: &LoweringInput,
    granule: SplitGranule,
    gathers: &[RowGather],
) -> Result<ShardedForward<T>, ShardRefusal> {
    let () = T::NONEMPTY;
    let n = input.ops.len();
    let shape = op_shapes(input);
    let mut wv = Weave::<T> {
        input,
        world: core::marker::PhantomData,
        granule: granule.get(),
        shape,
        place: Vec::with_capacity(n),
        src: vec![None; input.sources.len()],
        ops: Vec::with_capacity(n),
        op_map: Vec::with_capacity(n),
        whole_of: vec![None; n],
        whole_of_src: vec![None; input.sources.len()],
        collectives: Vec::new(),
        padded: Vec::with_capacity(n),
        result_projection: result_projection(input),
    };
    // ⭐ THE GATHER'S RULE, DECIDED BEFORE ITS ROW IS READ. A row is read by the forward's first
    // ops, its table by the vocabulary projection at the end; the row's placement is the table's
    // column split, so that split is taken here, by the same test the projection's N-split uses
    // ([`Weave::splits`] on the table's columns), and the projection finds its weight already cut
    // the way it would cut it.
    let mut gathered = Vec::new();
    for &g in gathers {
        let (Some(row), Some(table)) = (input.sources.get(g.row), input.sources.get(g.table))
        else {
            return Err(ShardRefusal::MalformedGather { gather: g });
        };
        if row.cols != table.rows || g.row == g.table || wv.src[g.row].is_some() {
            return Err(ShardRefusal::MalformedGather { gather: g });
        }
        // The table's column law is the projection's own when the table is the result
        // projection's weight (padded where that one pads), else the even law.
        let projection = wv
            .result_projection
            .filter(|&p| input.ops[p].inputs.get(1) == Some(&InputRef::Ext(g.table)));
        let law = match projection {
            Some(p) => wv.n_split(p, table.cols).map(|(_, cols_law, _)| cols_law),
            None => wv.splits(table.cols).then_some(SourceShard::Cols),
        };
        if let Some(law) = law.filter(|l| T::SHARDED && wv.can_adopt(g.table, *l)) {
            wv.src[g.table] = Some(law);
            wv.src[g.row] = Some(SourceShard::Partial);
            gathered.push(g);
        }
    }
    for i in 0..n {
        let d = if !T::SHARDED {
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
        // Padded: a contraction whose local width is not an even share of its own, or an op
        // carrying a padded operand's split on.
        let split_cols = d.out == Placement::Split(Axis::Cols);
        let pads = match (d.op, input.ops[i].op) {
            (SubOp::MatmulTile { n: local, .. }, SubOp::MatmulTile { n, .. }) => {
                local * T::RANKS != n
            }
            _ => false,
        };
        let inherits = input.ops[i]
            .inputs
            .iter()
            .any(|r| matches!(r, InputRef::Op(j) if wv.padded[*j]));
        wv.padded.push(split_cols && (pads || inherits));
        wv.op_map.push(wv.ops.len());
        wv.ops.push(OpDesc {
            op: d.op,
            m: input.ops[i].m,
            inputs: d.inputs,
        });
        wv.place.push(d.out);
    }
    // The forward's result and exported hidden state are read by the host: whole.
    let result = wv.whole_op(input.result)?;
    let exported = input.exported.map(|e| wv.whole_op(e)).transpose()?;
    let sources: Vec<SourceShard> = wv
        .src
        .iter()
        .map(|s| s.unwrap_or(SourceShard::Whole))
        .collect();
    let local_sources = input
        .sources
        .iter()
        .zip(&sources)
        .map(|(s, sh)| sh.local_shape::<T>(*s))
        .collect();
    Ok(ShardedForward {
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
        gathered,
        world: core::marker::PhantomData,
    })
}

/// [`shard`] over a whole compiler→target handoff: the sharded forward, plus every fact the handoff
/// keeps parallel to the forward's ops carried onto the op that now computes it. A collective the
/// analysis inserted realizes no FUF tile and belongs to no expansion, so it carries neither.
///
/// Sources keep their indices (only their per-rank shapes change), so the source bindings carry over
/// unchanged; how each one is sliced is [`ShardedForward::sources`].
pub fn shard_lowered<T: World>(
    lowered: &crate::handoff::LoweredDecode,
    granule: SplitGranule,
    gathers: &[RowGather],
) -> Result<(crate::handoff::LoweredDecode, ShardedForward<T>), ShardRefusal> {
    let sharded = shard::<T>(&lowered.input, granule, gathers)?;
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

/// An element the slice law can cut: a value with a ZERO, for the ranks that hold none of a
/// [`SourceShard::Partial`] value. The host reference's `f32`, and a loader's raw element bytes
/// (`[u8; N]`, any width).
pub trait SliceElement: Copy {
    const ZERO: Self;
}

impl SliceElement for f32 {
    const ZERO: Self = 0.0;
}

impl<const N: usize> SliceElement for [u8; N] {
    const ZERO: Self = [0; N];
}

/// THE SLICE LAW: rank `rank`'s share of a source of `shape` under `shard`. One function, read
/// by the loader that stages a rank's weights and by [`eval_ranks`]' reference alike.
pub fn slice_source<E: SliceElement, T: World>(
    full: &[E],
    shape: SourceShape,
    shard: SourceShard,
    rank: Rank<T>,
) -> Vec<E> {
    let (rows, cols) = (shape.rows as usize, shape.cols as usize);
    let (w, r) = (T::RANKS as usize, rank.get() as usize);
    debug_assert_eq!(full.len(), rows * cols, "source buffer vs its shape");
    match shard {
        SourceShard::Whole => full.to_vec(),
        SourceShard::Partial if r == 0 => full.to_vec(),
        SourceShard::Partial => vec![E::ZERO; full.len()],
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
        SourceShard::ColBlocks { block } => {
            let b = block.get() as usize;
            (0..rows)
                .flat_map(|i| {
                    (r * b..(r + 1) * b).map(move |c| {
                        if c < cols {
                            full[i * cols + c]
                        } else {
                            E::ZERO
                        }
                    })
                })
                .collect()
        }
        SourceShard::RowBlocks { block } => {
            let b = block.get() as usize;
            (r * b..(r + 1) * b)
                .flat_map(|i| {
                    (0..cols).map(move |c| {
                        if i < rows {
                            full[i * cols + c]
                        } else {
                            E::ZERO
                        }
                    })
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
        SubOp::AllGather { cols } => {
            let parts: Vec<(Vec<f32>, u32, u32)> = ranks
                .iter()
                .map(|b| gather(&node.inputs[0], graph, b))
                .collect();
            let (rows, lc) = (parts[0].1 as usize, parts[0].2 as usize);
            let cols = cols as usize;
            assert!(
                lc * ranks.len() >= cols,
                "{} blocks of {lc} columns cannot restore {cols}",
                ranks.len()
            );
            let mut all = Vec::with_capacity(rows * cols);
            for i in 0..rows {
                let row_start = all.len();
                for p in &parts {
                    all.extend_from_slice(&p.0[i * lc..(i + 1) * lc]);
                }
                // A padded last block's tail past the real width is cropped off.
                all.truncate(row_start + cols);
            }
            all
        }
        _ => panic!("{} is not a collective", node.op.name()),
    }
}

/// Evaluate a sharded forward's local graph on every rank IN LOCKSTEP — the host reference for
/// what the collectives mean. `sources(r)` is rank `r`'s source buffers (already sliced). Every
/// non-collective node runs per rank through [`crate::subtile_ir::eval_node`]; an
/// [`SubOp::AllReduce`] sums its operand over ranks in ABSOLUTE RANK ORDER and an
/// [`SubOp::AllGather`] concatenates column blocks in rank order, and both write the same bytes
/// to every rank. Returns every rank's buffers, `[rank]`.
pub fn eval_ranks<F: RopeForm, T: World>(
    graph: &SubtileIR<F>,
    scale_site: crate::subtile_ir::AttnScaleSite,
    sources: impl Fn(Rank<T>) -> Vec<Vec<f32>>,
) -> Vec<Vec<Vec<f32>>> {
    use crate::subtile_ir::{eval_node, scatter};
    let w = T::RANKS as usize;
    let mut bufs: Vec<Vec<Vec<f32>>> = ranks::<T>()
        .map(|r| {
            let srcs = sources(r);
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
            _ => bufs
                .iter()
                .map(|b| eval_node(node, graph, scale_site, b))
                .collect(),
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

    fn kinds<T: World>(sf: &ShardedForward<T>) -> Vec<CollectiveKind> {
        sf.collectives.iter().map(|c| c.kind).collect()
    }

    /// One rank has nothing to split and nothing to collect: the forward comes back as it went
    /// in, op for op, and every source loads whole.
    #[test]
    fn one_rank_is_the_identity() {
        let input = one_layer_input();
        let sf = shard::<TpWorld<1>>(&input, SplitGranule::of::<64>(), &[]).unwrap();
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
        let sf = shard::<TpWorld<4>>(&input, SplitGranule::of::<64>(), &[]).unwrap();
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
            let InputRef::Op(reads) = c.reads else {
                panic!("an all-reduce reading a source in a forward that gathers no row")
            };
            assert!(matches!(sf.local.ops[reads].op, SubOp::MatmulTile { .. }));
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
    fn sharded_equals_unsharded<T: World>() {
        let input = one_layer_input();
        let full = sources_of(&input);
        let reference = {
            let g = lower_region(&input, NB);
            let refs: Vec<&[f32]> = full.iter().map(|v| v.as_slice()).collect();
            let bufs = eval_dag(&g, &refs);
            bufs[g.result.index()].clone()
        };
        let sf = shard::<T>(&input, SplitGranule::of::<64>(), &[]).unwrap();
        let g = lower_region(&sf.local, NB);
        let per_rank = |r: Rank<T>| -> Vec<Vec<f32>> {
            full.iter()
                .zip(&input.sources)
                .zip(&sf.sources)
                .map(|((buf, shape), shard)| slice_source(buf, *shape, *shard, r))
                .collect()
        };
        let bufs = eval_ranks(&g, crate::subtile_ir::AttnScaleSite::InKernel, per_rank);
        assert_eq!(bufs.len() as u32, T::RANKS);
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
        sharded_equals_unsharded::<TpWorld<2>>();
    }

    #[test]
    fn four_ranks_compute_the_unsharded_forward() {
        sharded_equals_unsharded::<TpWorld<4>>();
    }

    /// Eight ranks: 32 query heads / 8 kv heads still split, one kv head per rank.
    #[test]
    fn eight_ranks_compute_the_unsharded_forward() {
        sharded_equals_unsharded::<TpWorld<8>>();
    }

    /// Sixteen ranks cannot split 8 kv heads: attention is replicated instead (its q/k/v
    /// gathered), and the forward is still the unsharded one. An unsplittable op costs
    /// collectives, never correctness.
    #[test]
    fn an_unsplittable_attention_is_replicated_not_refused() {
        let input = one_layer_input();
        let sf = shard::<TpWorld<16>>(&input, SplitGranule::of::<1>(), &[]).unwrap();
        assert!(kinds(&sf).contains(&CollectiveKind::AllGather));
        sharded_equals_unsharded_at::<TpWorld<16>>(&input, SplitGranule::of::<1>());
    }

    fn sharded_equals_unsharded_at<T: World>(input: &LoweringInput, g: SplitGranule) {
        let full = sources_of(input);
        let gr = lower_region(input, NB);
        let refs: Vec<&[f32]> = full.iter().map(|v| v.as_slice()).collect();
        let reference = eval_dag(&gr, &refs)[gr.result.index()].clone();
        let sf = shard::<T>(input, g, &[]).unwrap();
        let gl = lower_region(&sf.local, NB);
        let per_rank = |r: Rank<T>| -> Vec<Vec<f32>> {
            full.iter()
                .zip(&input.sources)
                .zip(&sf.sources)
                .map(|((buf, shape), shard)| slice_source(buf, *shape, *shard, r))
                .collect()
        };
        let got = &eval_ranks(&gl, crate::subtile_ir::AttnScaleSite::InKernel, per_rank)[0]
            [gl.result.index()];
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
            shard_lowered::<TpWorld<1>>(&lowered, SplitGranule::of::<64>(), &[]).unwrap();
        assert_eq!(format!("{one:?}"), format!("{lowered:?}"));

        let (two, sf) =
            shard_lowered::<TpWorld<2>>(&lowered, SplitGranule::of::<64>(), &[]).unwrap();
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
            shard::<TpWorld<2>>(&input, SplitGranule::of::<64>(), &[]),
            Err(ShardRefusal::StatedCollective { .. })
        ));
    }

    /// The tied-embedding decode step: a decode layer under a final norm and a vocabulary
    /// projection whose table is also where the layer's input row came from — `x` (source 0) is
    /// stated as gathered from the table's columns.
    fn tied_head_input(vocab: u32) -> (LoweringInput, RowGather) {
        use crate::subtile_ir::GainConvention;
        let mut input = crate::fixtures::one_layer_input_shaped(256, 128, 512, 64);
        let h = input.sources[0].cols;
        let gain = input.sources.len();
        input.sources.push(SourceShape { rows: 1, cols: h });
        let table = input.sources.len();
        input.sources.push(SourceShape {
            rows: h,
            cols: vocab,
        });
        let norm = input.ops.len();
        input.ops.push(OpDesc {
            op: SubOp::RmsNorm {
                eps: 1e-5,
                gain: GainConvention::Scale,
            },
            m: 1,
            inputs: vec![InputRef::Op(input.result), InputRef::Ext(gain)],
        });
        input.ops.push(OpDesc {
            op: SubOp::MatmulTile {
                n: vocab,
                weight: GemmWeight::Dense,
            },
            m: 1,
            inputs: vec![InputRef::Op(norm), InputRef::Ext(table)],
        });
        input.result = norm + 1;
        (input, RowGather { row: 0, table })
    }

    /// ⭐ A GATHER IS A CONTRACTION. With the vocabulary table split by columns, the row gathered
    /// out of it is a partial sum — the owner's column, zero on every other rank — and the
    /// analysis reduces it exactly where the forward first reads it whole, as it reduces a
    /// K-split projection: one all-reduce of the input row, the projection's all-gather at the
    /// result, and the table cut ONCE, by columns, for both readers.
    #[test]
    fn a_row_gathered_from_a_split_table_is_reduced_where_it_is_first_read_whole() {
        type W = TpWorld<2>;
        let (input, gather) = tied_head_input(1024);
        let sf = shard::<W>(&input, SplitGranule::of::<64>(), &[gather]).unwrap();
        assert_eq!(sf.gathered, vec![gather]);
        assert_eq!(sf.sources[gather.row], SourceShard::Partial);
        assert_eq!(sf.sources[gather.table], SourceShard::Cols);
        let first = sf.collectives[0];
        assert_eq!(
            (first.kind, first.reads),
            (CollectiveKind::AllReduce, InputRef::Ext(gather.row))
        );
        let last = sf.collectives.last().unwrap();
        assert_eq!(last.kind, CollectiveKind::AllGather);
        assert_eq!(sf.local.result, last.op);
        // The row is reduced once, however many ops read it whole.
        let reduces_row = sf
            .collectives
            .iter()
            .filter(|c| c.reads == InputRef::Ext(gather.row))
            .count();
        assert_eq!(reduces_row, 1);

        // Every rank gathers the drawn token's column out of the block it holds — the column
        // itself on its owner, zero elsewhere — and the forward is the unsharded one.
        let token = 700;
        let mut full = sources_of(&input);
        let (h, vocab) = (input.sources[0].cols as usize, 1024usize);
        full[gather.row] = (0..h)
            .map(|k| full[gather.table][k * vocab + token])
            .collect();
        let gr = lower_region(&input, NB);
        let refs: Vec<&[f32]> = full.iter().map(|v| v.as_slice()).collect();
        let reference = eval_dag(&gr, &refs)[gr.result.index()].clone();
        let gl = lower_region(&sf.local, NB);
        let lc = vocab / W::RANKS as usize;
        let per_rank = |r: Rank<W>| -> Vec<Vec<f32>> {
            let mut srcs: Vec<Vec<f32>> = full
                .iter()
                .zip(&input.sources)
                .zip(&sf.sources)
                .map(|((buf, shape), shard)| slice_source(buf, *shape, *shard, r))
                .collect();
            let origin = r.get() as usize * lc;
            srcs[gather.row] = if (origin..origin + lc).contains(&token) {
                let block = &srcs[gather.table];
                (0..h).map(|k| block[k * lc + token - origin]).collect()
            } else {
                vec![0.0; h]
            };
            srcs
        };
        let bufs = eval_ranks(&gl, crate::subtile_ir::AttnScaleSite::InKernel, per_rank);
        for (r, b) in bufs.iter().enumerate() {
            let worst = b[gl.result.index()]
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(worst < 1e-4, "rank {r}'s logits off by {worst}");
        }
        assert_eq!(bufs[0][gl.result.index()], bufs[1][gl.result.index()]);
    }

    /// At one rank a stated gather changes nothing: the identity is unconditional.
    #[test]
    fn one_rank_ignores_a_stated_gather() {
        let (input, gather) = tied_head_input(1024);
        let sf = shard::<TpWorld<1>>(&input, SplitGranule::of::<64>(), &[gather]).unwrap();
        assert!(sf.gathered.is_empty() && sf.collectives.is_empty());
        assert_eq!(format!("{:?}", sf.local), format!("{input:?}"));
    }

    /// A table that does not split leaves its row whole on every rank: nothing to reduce.
    #[test]
    fn a_row_gathered_from_a_whole_table_stays_whole() {
        // One 64-column granule covers the whole vocabulary: no block law splits it.
        let (input, gather) = tied_head_input(64);
        let sf = shard::<TpWorld<2>>(&input, SplitGranule::of::<64>(), &[gather]).unwrap();
        assert!(sf.gathered.is_empty());
        assert_eq!(sf.sources[gather.row], SourceShard::Whole);
        assert!(
            sf.collectives
                .iter()
                .all(|c| c.reads != InputRef::Ext(gather.row))
        );
    }

    /// A gather whose row is not as wide as its table is tall names no lookup, and is refused.
    #[test]
    fn a_malformed_gather_is_refused() {
        let (input, _) = tied_head_input(1024);
        // Source 3 is the k projection `[256, 128]`; source 5 is `cos`, `[1, 64]`.
        let bad = RowGather { row: 5, table: 3 };
        assert_eq!(
            shard::<TpWorld<2>>(&input, SplitGranule::of::<64>(), &[bad]).unwrap_err(),
            ShardRefusal::MalformedGather { gather: bad }
        );
    }

    /// ⭐ THE CHAINED HEAD OVER THE SPLIT TABLE, END TO END. The two-rank forward is lowered and
    /// the chain minted onto it exactly as onto a one-rank decode: the gather reads this rank's
    /// column block of the table (the projection's own), the analysis's all-reduce is its rows'
    /// one reader, and the block the mint reports is the rank's columns. Each rank then stages its
    /// own block's decompose — the token's slab and lane on its owner, the sentinel's zero row on
    /// the other rank — and both ranks' logits are the unsharded forward's with the token's row
    /// staged by the host.
    fn chained_head_over_a_split_table(vocab: u32, token: u32) {
        use crate::subtile_ir::{
            ChainBlock, Range, Region, SubtileId, SubtileNode, TensorId, TensorRegion, chain_l_row,
            chain_l_seed, chain_s_row, chain_s_seed, chain_slab_entries, mint_embed_chain,
        };
        type W = TpWorld<2>;
        let (input, gather) = tied_head_input(vocab);
        let sf = shard::<W>(&input, SplitGranule::of::<64>(), &[gather]).unwrap();
        let mut g = lower_region(&sf.local, NB);
        let whole = |t: TensorId, s: TensorShape| TensorRegion {
            tensor: t,
            region: Region {
                rows: Range::new(0, s.rows),
                cols: Range::new(0, s.cols),
            },
        };
        // The draw's stand-in: an op output, one row per request, answering pick 0 (the chain's
        // own tests own the draw; here the stage needs only an answerable pick).
        let logits = g.result;
        let pick = TensorId::from_index(g.tensors.len());
        let pick_shape = g.shape(logits);
        g.tensors.push(pick_shape);
        g.nodes.push(SubtileNode {
            id: SubtileId::from_index(g.nodes.len()),
            op: SubOp::ScalarMul { scale: 0.0 },
            inputs: vec![whole(logits, pick_shape)],
            output: whole(pick, pick_shape),
        });
        g.result = pick;
        let lc = match sf.sources[gather.table] {
            SourceShard::Cols => vocab / W::RANKS,
            SourceShard::ColBlocks { block } => block.get(),
            other => panic!("the table is cut by columns, not {other:?}"),
        };
        let mint = mint_embed_chain(&mut g, TensorId::from_index(gather.row), pick, vocab);
        assert_eq!(mint.block, lc, "the gather reads this rank's block");
        let SubOp::EmbedRowGather { .. } = g.nodes[0].op else {
            panic!("the chain's gather is the forward's first op")
        };
        assert_eq!(
            g.nodes[0].inputs[2].tensor,
            TensorId::from_index(gather.table)
        );
        let rows = g.nodes[0].output.tensor;
        let readers: Vec<bool> = g
            .nodes
            .iter()
            .filter(|n| n.inputs.iter().any(|i| i.tensor == rows))
            .map(|n| matches!(n.op, SubOp::AllReduce))
            .collect();
        assert_eq!(
            readers,
            vec![true],
            "the gathered rows' one reader is the analysis's all-reduce"
        );

        // The reference: the one-rank forward with the token's column staged as its row.
        let mut full = sources_of(&input);
        let h = input.sources[0].cols as usize;
        full[gather.row] = (0..h)
            .map(|k| full[gather.table][k * vocab as usize + token as usize])
            .collect();
        let gr = lower_region(&input, NB);
        let refs: Vec<&[f32]> = full.iter().map(|v| v.as_slice()).collect();
        let reference = eval_dag(&gr, &refs)[gr.result.index()].clone();

        // Each rank: its slices, then its own block's sticks and tables.
        let k = chain_slab_entries(h as u64) as u32;
        let words = |v: Vec<u32>| -> Vec<f32> { v.into_iter().map(|x| x as f32).collect() };
        let tab_rows = g.shape(TensorId::from_index(mint.s_stick.index() + 2)).rows;
        let per_rank = |r: Rank<W>| -> Vec<Vec<f32>> {
            let block = ChainBlock::of_rank(lc, r);
            let mut srcs: Vec<Vec<f32>> = full
                .iter()
                .zip(&input.sources)
                .zip(&sf.sources)
                .map(|((buf, shape), shard)| slice_source(buf, *shape, *shard, r))
                .collect();
            srcs.push(words(chain_s_seed(k, &[token], 1, block)));
            srcs.push(words(chain_l_seed(&[token], h as u32, 1, block)));
            srcs.push(
                (0..tab_rows)
                    .flat_map(|t| words(chain_s_row(k, t, block)))
                    .collect(),
            );
            srcs.push(
                (0..tab_rows)
                    .flat_map(|t| words(chain_l_row(t, h as u32, block)))
                    .collect(),
            );
            srcs
        };
        let gathered_logits = g
            .nodes
            .iter()
            .find(|n| matches!(n.op, SubOp::ScalarMul { scale } if scale == 0.0))
            .expect("the draw's stand-in")
            .inputs[0]
            .tensor;
        let bufs = eval_ranks(&g, crate::subtile_ir::AttnScaleSite::InKernel, per_rank);
        for (r, b) in bufs.iter().enumerate() {
            let worst = b[gathered_logits.index()]
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(worst < 1e-4, "rank {r}'s logits off by {worst}");
        }
    }

    /// The chain over an even split (4096 columns, two blocks of 2048) and a padded one (4000
    /// columns, two blocks of 2048 — the last holding 1952 real ones), the token in the second
    /// rank's block both times. Each vocabulary is a whole pick group's pad (4096), the draw's own
    /// law for the extent its tables span.
    #[test]
    fn the_chained_head_over_a_split_table_is_the_unsharded_forward() {
        chained_head_over_a_split_table(4096, 3000);
    }

    #[test]
    fn the_chained_head_over_a_padded_table_is_the_unsharded_forward() {
        chained_head_over_a_split_table(4000, 3000);
    }

    /// ⭐ AN UNEVEN VOCABULARY PADS ITS LAST BLOCK. 1000 columns do not split into two blocks of
    /// whole 64-column granules; the result's projection splits anyway, each rank holding
    /// `ceil(1000 / 2)` rounded up to 512 columns (the last 24 of rank 1's zero), and the
    /// all-gather the result demands crops them off. The forward is the unsharded one.
    #[test]
    fn an_uneven_vocabulary_pads_its_last_block() {
        type W = TpWorld<2>;
        let (input, gather) = tied_head_input(1000);
        let g = SplitGranule::of::<64>();
        let sf = shard::<W>(&input, g, &[]).unwrap();
        let block = NonZeroU32::new(512).unwrap();
        assert_eq!(split_block::<W>(1000, g), Some(block));
        assert_eq!(sf.sources[gather.table], SourceShard::ColBlocks { block });
        let last = sf.collectives.last().unwrap();
        assert_eq!(sf.local.result, last.op);
        assert!(matches!(
            sf.local.ops[last.op].op,
            SubOp::AllGather { cols: 1000 }
        ));
        sharded_equals_unsharded_at::<W>(&input, g);
    }

    /// Padding is the result's alone: an MLP 500 wide does not split into whole granules, and
    /// its gate and up projections stay whole — a padded column there would be read as a value
    /// by the activation and the down projection.
    #[test]
    fn an_uneven_width_inside_the_forward_is_replicated_not_padded() {
        type W = TpWorld<2>;
        let input = crate::fixtures::one_layer_input_shaped(256, 128, 500, 64);
        let g = SplitGranule::of::<64>();
        let sf = shard::<W>(&input, g, &[]).unwrap();
        // Sources 11 and 12 are gate and up.
        assert_eq!(sf.sources[11], SourceShard::Whole);
        assert_eq!(sf.sources[12], SourceShard::Whole);
        assert!(sf.sources.iter().all(|s| !matches!(
            s,
            SourceShard::ColBlocks { .. } | SourceShard::RowBlocks { .. }
        )));
        sharded_equals_unsharded_at::<W>(&input, g);
    }

    /// The padded laws cut whole buffers into blocks and zero what lies past the source.
    #[test]
    fn the_padded_slice_laws_zero_past_the_source() {
        type W = TpWorld<2>;
        let (r0, r1) = (Rank::<W>::of::<0>(), Rank::<W>::of::<1>());
        let block = NonZeroU32::new(2).unwrap();
        // [2, 3]: 0 1 2 / 3 4 5.
        let full = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = SourceShape { rows: 2, cols: 3 };
        let cols = SourceShard::ColBlocks { block };
        assert_eq!(
            slice_source(&full, shape, cols, r0),
            vec![0.0, 1.0, 3.0, 4.0]
        );
        assert_eq!(
            slice_source(&full, shape, cols, r1),
            vec![2.0, 0.0, 5.0, 0.0]
        );
        // [3, 2]: 0 1 / 2 3 / 4 5.
        let shape = SourceShape { rows: 3, cols: 2 };
        let rows = SourceShard::RowBlocks { block };
        assert_eq!(
            slice_source(&full, shape, rows, r0),
            vec![0.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(
            slice_source(&full, shape, rows, r1),
            vec![4.0, 5.0, 0.0, 0.0]
        );
    }
}

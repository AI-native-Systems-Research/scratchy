// SPDX-License-Identifier: Apache-2.0
//! ONE BUCKET'S TAPE, AS STATIC DATA — the counterpart of metal's `LoweredMetalTape`.
//!
//! A tape is what `scratchy-target-cuda-compiler` lowers one canonical (one bucket) to: the
//! arena's layout, where the forward's input and result live in it, how large each weight must
//! be, and the steps in launch order with every operand already resolved to a byte offset or a
//! weight index. Every field is a constant built at expansion time; the runtime reads it and
//! analyses nothing. Plain `Copy` data with no pointer, so it builds in a `const` context and on a
//! host with no CUDA toolchain.

use serde::Serialize;

use crate::op_abi::CUDA_EXPERT_TILE_ROWS;
use crate::step::{
    ATTN_BF16_HEAD_DIMS, ATTN_WINDOW_MULTIPLE, AttnPrefillBf16, CUDA_KV_BLOCK_SIZE, CudaStep,
    EmbedBf16, ExpertCombineBf16, ExpertGemmBf16, ExpertGemvBf16, ExpertSortBf16,
    GDN_CONV_KERNEL_RANGE, GDN_HEAD_K, GDN_HEAD_V_MAX, GDN_HEAD_V_MULTIPLE, GDN_SEQUENCE_BYTES,
    GatedDeltaNetBf16, GemmBf16, MoeRouteBf16, RmsNormBf16, RopeAppendBf16, RopeBf16,
    RouterLogitsBf16, SiluMulBf16, gdn_conv_dim, gdn_scratch, sort_index_regions,
};

/// Bytes per activation element: CUDA's activations are bf16.
pub const CUDA_ACT_BYTES: u64 = 2;

/// Every colour's arena offset is a multiple of this. TMA needs 16-byte-aligned global addresses
/// and `cudaMalloc` returns 256-byte-aligned memory, so every slot pointer meets the launchers'
/// alignment contract with margin.
pub const CUDA_ARENA_ALIGN: u64 = 256;

/// A byte offset into the worker's arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ArenaOffset(pub u64);

/// A weight, by its source's index in the canonical: the index into the worker's weight buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct WeightIx(pub u32);

/// One colour's bytes in the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ArenaSpan {
    pub offset: ArenaOffset,
    pub bytes: u64,
}

/// Where every colour lives in one tape's arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
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
    /// The attention classes of the layers: one for a uniform model (every Qwen tape: its
    /// `kv_heads`, `head_dim`, no window), two for Gemma 4 (sliding, then global). Class 0 is
    /// `kv_heads` and `head_dim`'s.
    pub classes: &'static [KvClass],
    /// Per entry of `layers`, its class: an index into `classes`.
    pub layer_class: &'static [u8],
}

/// One attention class's KV geometry: each layer of it caches `kv_heads` heads of `head_dim`,
/// and its queries see their last `window` keys (`None`: every key).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct KvClass {
    pub kv_heads: u32,
    pub head_dim: u32,
    pub window: Option<u32>,
}

/// Bytes of one u32 entry of a KV region.
const KV_ENTRY_BYTES: u64 = 4;

/// The tape's Gated-DeltaNet layers: which layers' recurrent state it reads and writes, their
/// geometry, and the two arena regions its GDN steps use.
///
/// Every field is fixed at expansion. The pool's slot count is not here: it is the worker's
/// (`runtime::GdnPoolConfig`), sized once at construction, its per-layer buffers laid out by
/// `scratchy_subtile::gdn_state_layout::GdnStateLayout`. What the forward writes into the
/// sequence region (the slot, the real row count, the fresh flag) is its runtime data, uploaded as
/// the KV sequence is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GdnTape {
    /// Model layer ids whose GDN state this tape reads and writes, in pool order (ascending); a
    /// step's `layer` is an index into this list.
    pub layers: &'static [u32],
    pub num_k_heads: u32,
    pub num_v_heads: u32,
    pub head_k: u32,
    pub head_v: u32,
    pub conv_kernel: u32,
    /// `[slot, q_len, start, 0]` u32: exactly 16 bytes at an aligned offset of its own.
    pub sequence: ArenaSpan,
    /// The f32 scratch every GDN step passes its intermediates through:
    /// `crate::step::gdn_scratch(bucket_m, …)`'s bytes.
    pub scratch: ArenaSpan,
}

impl GdnTape {
    /// The scratch's bytes at `bucket_m` rows of this geometry, or `None` on overflow.
    fn scratch_bytes(&self, bucket_m: u32) -> Option<u64> {
        let value_dim = self.num_v_heads.checked_mul(self.head_v)?;
        let conv_dim = gdn_conv_dim(self.num_k_heads, self.head_k, self.num_v_heads, self.head_v)?;
        gdn_scratch(bucket_m, conv_dim, self.num_v_heads, value_dim).map(|s| s.bytes)
    }
}

/// The tape's MoE scratch: the regions every MoE layer's routing and experts pass their runtime
/// data through, the geometry they are sized by, and whether the bucket's pairs are sorted by
/// expert. Every MoE layer reuses it: the layers run in order on one stream.
///
/// With `P = bucket_m · top_k` pairs, `n = min(experts, P)`, `max_tiles = ⌊(P + 127·n) / 128⌋`
/// and `R = 128 · max_tiles` (the sorted rows, each expert's run padded to 128): `logits` `[m,
/// experts]` bf16, `topk_idx` `[m, top_k]` i32, `scores` `[m, top_k]` f32, the sort's regions
/// ([`MoeSortTape`]) when `sort` is `Some`, then `gate`, `up` and `act` `[R, inter]` bf16 and
/// `down` `[R, hidden]` bf16 (`P` rows each when gathered), each at a multiple of
/// [`CUDA_ARENA_ALIGN`], in that order. Nothing in it is cleared between forwards or layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct MoeTape {
    pub experts: u32,
    pub top_k: u32,
    pub hidden: u32,
    pub inter: u32,
    /// [`CUDA_EXPERT_TILE_ROWS`].
    pub tile_rows: u32,
    /// The sorted rows' tiles, `⌊(P + 127·n) / 128⌋`; 0 when gathered.
    pub max_tiles: u32,
    pub logits: ArenaSpan,
    pub topk_idx: ArenaSpan,
    pub scores: ArenaSpan,
    /// The sort's regions, exactly when the bucket's pairs are sorted by expert
    /// (`op_abi::CUDA_SORTED_PAIRS` pairs or more).
    pub sort: Option<MoeSortTape>,
    pub gate: ArenaSpan,
    pub up: ArenaSpan,
    pub act: ArenaSpan,
    pub down: ArenaSpan,
    /// `[m, hidden]` bf16, Gemma's router pre-norm, after `down`; `None` for a block whose router
    /// reads the hidden itself (every Qwen tape).
    pub router_norm: Option<ArenaSpan>,
}

/// The sort's regions of a [`MoeTape`]: `chunk_hist` `[⌈P/256⌉, experts]` u32, `counts`
/// `[experts]` u32, `offsets` `[experts + 1]` u32 (`offsets[experts]` the padded row total),
/// `tile_expert` `[max_tiles]` i32, `pos` `[P]` u32 (pair to sorted row), contiguous from
/// `chunk_hist` in that order, each 256-aligned, then `rows` `[R, hidden]` bf16.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct MoeSortTape {
    pub chunk_hist: ArenaSpan,
    pub counts: ArenaSpan,
    pub offsets: ArenaSpan,
    pub tile_expert: ArenaSpan,
    pub pos: ArenaSpan,
    pub rows: ArenaSpan,
}

/// The pairs, max tiles and padded rows of `bucket_m` tokens routed to `top_k` of `experts`:
/// `(P, ⌊(P + 127·min(experts, P)) / 128⌋, 128 · max_tiles)`. `None` on overflow.
pub const fn moe_tiles(bucket_m: u32, experts: u32, top_k: u32) -> Option<(u32, u32, u64)> {
    let Some(pairs) = bucket_m.checked_mul(top_k) else {
        return None;
    };
    let n = if experts < pairs { experts } else { pairs };
    let t = CUDA_EXPERT_TILE_ROWS as u64;
    let tiles = (pairs as u64 + (t - 1) * n as u64) / t;
    if tiles > u32::MAX as u64 {
        return None;
    }
    Some((pairs, tiles as u32, tiles * t))
}

/// The bytes of each region of a [`MoeTape`] of this geometry, in its order: `logits`,
/// `topk_idx`, `scores`, then, when `sorted`, `chunk_hist`, `counts`, `offsets`, `tile_expert`,
/// `pos` and `rows`, then `gate`, `up`, `act` and `down`. `None` on overflow.
pub fn moe_region_bytes(
    bucket_m: u32,
    (experts, top_k, hidden, inter): (u32, u32, u32, u32),
    sorted: bool,
) -> Option<Vec<u64>> {
    let (pairs, tiles, rows) = moe_tiles(bucket_m, experts, top_k)?;
    let bf16 = |r: u64, c: u32| r.checked_mul(u64::from(c))?.checked_mul(CUDA_ACT_BYTES);
    let m = u64::from(bucket_m);
    let mut out = vec![
        bf16(m, experts)?,
        u64::from(pairs) * 4,
        u64::from(pairs) * 4,
    ];
    let expert_rows = if sorted {
        let [_, counts, offsets, tile_expert, pos, end] =
            sort_index_regions(pairs, experts, tiles)?;
        out.extend([
            counts,
            u64::from(experts) * 4,
            (u64::from(experts) + 1) * 4,
            u64::from(tiles) * 4,
            u64::from(pairs) * 4,
        ]);
        // `sort_index_regions` aligns each region's end; the spans are the regions' own bytes.
        debug_assert!(offsets > counts && tile_expert > offsets && pos > tile_expert && end > pos);
        out.push(bf16(rows, hidden)?);
        rows
    } else {
        u64::from(pairs)
    };
    out.extend([
        bf16(expert_rows, inter)?,
        bf16(expert_rows, inter)?,
        bf16(expert_rows, inter)?,
        bf16(expert_rows, hidden)?,
    ]);
    Some(out)
}

impl MoeTape {
    /// Every region, in the tape's order, with its name.
    pub fn regions(&self) -> Vec<(Region, ArenaSpan)> {
        let mut out = vec![
            (Region::MoeLogits, self.logits),
            (Region::MoeTopKIdx, self.topk_idx),
            (Region::MoeScores, self.scores),
        ];
        if let Some(s) = self.sort {
            out.extend([
                (Region::MoeChunkHist, s.chunk_hist),
                (Region::MoeCounts, s.counts),
                (Region::MoeOffsets, s.offsets),
                (Region::MoeTileExpert, s.tile_expert),
                (Region::MoePos, s.pos),
                (Region::MoeRows, s.rows),
            ]);
        }
        out.extend([
            (Region::MoeGate, self.gate),
            (Region::MoeUp, self.up),
            (Region::MoeAct, self.act),
            (Region::MoeDown, self.down),
        ]);
        out.extend(self.router_norm.map(|r| (Region::MoeRouterNorm, r)));
        out
    }

    /// The span from the first region's start to the last one's end. The regions are in order
    /// (`CudaTape::validate` checks it), so it covers every one.
    pub fn span(&self) -> ArenaSpan {
        let start = self.logits.offset.0;
        let last = self.router_norm.unwrap_or(self.down);
        let end = last.offset.0.saturating_add(last.bytes);
        ArenaSpan {
            offset: self.logits.offset,
            bytes: end.saturating_sub(start),
        }
    }

    /// The regions projection `proj`'s expert step reads its rows from and writes them to: the
    /// gate and up the sorted rows (`sorted`, when the pairs are sorted; the token rows, an arena
    /// colour, when gathered), the down the activation rows.
    pub fn projection_regions(
        &self,
        proj: crate::step::ExpertProjection,
        sorted: Option<ArenaSpan>,
    ) -> ((Region, ArenaSpan), (Region, ArenaSpan)) {
        use crate::step::ExpertProjection as P;
        let rows = (Region::MoeRows, sorted.unwrap_or(self.logits));
        match proj {
            P::Gate => (rows, (Region::MoeGate, self.gate)),
            P::Up => (rows, (Region::MoeUp, self.up)),
            P::Down => ((Region::MoeAct, self.act), (Region::MoeDown, self.down)),
        }
    }

    /// The rows of the expert regions: `R` sorted, `P` gathered.
    pub fn expert_rows(&self, bucket_m: u32) -> u32 {
        match self.sort {
            Some(_) => self.max_tiles * self.tile_rows,
            None => bucket_m * self.top_k,
        }
    }
}

/// What a forward uploads as its input, and where it lands in the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum TapeInput {
    /// Colour 0, the embedded hidden, `[bucket_m, hidden]` bf16: a tape whose canonical has no
    /// embedding table to gather from (the layer-level fixtures).
    Hidden(ArenaSpan),
    /// `bucket_m` u32 token ids, one per row, at an aligned region after every other: the tape's
    /// first step is the one [`CudaStep::Embed`], which gathers them from the `vocab`-row table
    /// into colour 0. Every id is below `vocab`; the worker checks that before any copy.
    TokenIds { span: ArenaSpan, vocab: u32 },
}

impl TapeInput {
    /// The input's span in the arena.
    pub const fn span(&self) -> ArenaSpan {
        match *self {
            Self::Hidden(span) | Self::TokenIds { span, .. } => span,
        }
    }
}

/// Bytes of one token id.
const TOKEN_ID_BYTES: u64 = 4;

/// One bucket's forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct CudaTape {
    /// The token count `M` the tape is compiled for.
    pub bucket_m: u32,
    pub arena: ArenaLayout,
    /// What the forward uploads: the embedded hidden (colour 0), or the token ids its first step
    /// embeds into colour 0.
    pub input: TapeInput,
    /// The colourer's result colour, which the forward downloads.
    pub result: ArenaSpan,
    /// Per source index: the bytes its weight buffer must hold (a dense weight's bf16 bytes, a
    /// GDN layer's packed f32 bundle, the embedding table at the embedded hidden's index); 0 for a
    /// source no step reads as a weight.
    pub weight_bytes: &'static [u64],
    /// One step per launch, in launch order.
    pub steps: &'static [CudaStep],
    /// The rotary table the forward fills, when a step reads one; `None` for a tape with no rope.
    /// It is the `Cos { local: false }` table: a uniform model's only one, Gemma's global layers'.
    pub rotary: Option<RotaryTable>,
    /// The second rotary table, the `Cos { local: true }` one (Gemma's sliding layers), placed
    /// after [`Self::rotary`]; `None` for every model with one rope class.
    pub rotary_local: Option<RotaryTable>,
    /// The paged KV cache and its regions, when the canonical binds a `PrefixK` source; `None`
    /// for a tape with no attention.
    pub kv: Option<KvTape>,
    /// The Gated-DeltaNet layers and their regions, exactly when a step is a
    /// [`CudaStep::GatedDeltaNet`]; `None` for a tape with no linear attention.
    pub gdn: Option<GdnTape>,
    /// The MoE scratch, exactly when a step is a MoE step (a router, a routing, a sort, an expert
    /// projection or a combine); `None` for a tape with no mixture of experts.
    pub moe: Option<MoeTape>,
}

/// Which of the tape's reserved regions a [`TapeError`] names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Region {
    Rotary,
    TokenIds,
    BlockTable,
    SlotMapping,
    Sequence,
    GdnSequence,
    GdnScratch,
    MoeLogits,
    MoeTopKIdx,
    MoeScores,
    MoeChunkHist,
    MoeCounts,
    MoeOffsets,
    MoeTileExpert,
    MoePos,
    MoeRows,
    MoeGate,
    MoeUp,
    MoeAct,
    MoeDown,
    MoeRouterNorm,
}

impl std::fmt::Display for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Rotary => "the rotary table",
            Self::TokenIds => "the token ids",
            Self::BlockTable => "the block table",
            Self::SlotMapping => "the slot mapping",
            Self::Sequence => "the sequence region",
            Self::GdnSequence => "the GDN sequence region",
            Self::GdnScratch => "the GDN scratch",
            Self::MoeLogits => "the MoE logits",
            Self::MoeTopKIdx => "the MoE top-k indices",
            Self::MoeScores => "the MoE scores",
            Self::MoeChunkHist => "the MoE chunk histogram",
            Self::MoeCounts => "the MoE expert counts",
            Self::MoeOffsets => "the MoE expert offsets",
            Self::MoeTileExpert => "the MoE tile experts",
            Self::MoePos => "the MoE sorted positions",
            Self::MoeRows => "the MoE sorted rows",
            Self::MoeGate => "the MoE gate rows",
            Self::MoeUp => "the MoE up rows",
            Self::MoeAct => "the MoE activation rows",
            Self::MoeDown => "the MoE down rows",
            Self::MoeRouterNorm => "the MoE router-norm rows",
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
    /// The token-id input is not `bucket_m` u32 at a multiple of [`CUDA_ARENA_ALIGN`], or names a
    /// vocabulary of no rows.
    TokenIdsShape {
        span: ArenaSpan,
        bucket_m: u32,
        vocab: u32,
    },
    /// The tape's input is token ids and its first step is not an embed (`None`: it has no step).
    EmbedMissing { first: Option<usize> },
    /// The `step`-th step is an embed, and it is not the first step of a token-id tape: an embed
    /// after the first step, or in a tape whose input is the embedded hidden.
    EmbedMisplaced { step: usize },
    /// The `step`-th step, the embed, reads other token ids, rows or vocabulary than the tape's
    /// input: its `tokens` offset, `rows` or `vocab` disagree with the tape's.
    EmbedInput { step: usize },
    /// The `step`-th step, the embed, does not write exactly colour 0, the embedded hidden the
    /// tape's first ops read: its `out` is not colour 0's offset, or `rows · hidden` bf16 are not
    /// colour 0's bytes.
    EmbedOutput { step: usize },
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
    /// The KV attention classes are not a geometry the kernels take: none, class 0 not the
    /// tape's `kv_heads` and `head_dim`, a head dim or window off the rules, a layer without a
    /// class or naming one the tape lacks; or a local rotary table without a global one.
    KvClasses,
    /// The `step`-th step is a one-row kernel's (a GEMV or the decode attention) and its rows are
    /// `rows`, not 1, or the tape's bucket is not one row.
    OneRowStep { step: usize, rows: u32 },
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
    /// A GDN region (the sequence or the scratch) is not at a multiple of [`CUDA_ARENA_ALIGN`], or
    /// reaches past the arena.
    GdnRegionOutside {
        region: Region,
        span: ArenaSpan,
        total: u64,
    },
    /// A GDN region is not the bytes its users take: the sequence exactly 16, the scratch
    /// `gdn_scratch(bucket_m, …)`'s.
    GdnRegionShape {
        region: Region,
        span: ArenaSpan,
        want: u64,
    },
    /// The `step`-th step uses GDN layer index `layer`; the tape has `layers` (none when it has
    /// no [`GdnTape`]).
    GdnLayer {
        step: usize,
        layer: u32,
        layers: usize,
    },
    /// The `step`-th step's GDN `region` operand is not the tape's region of that name (`None`
    /// when the tape has no [`GdnTape`]).
    GdnRegionMismatch {
        step: usize,
        region: Region,
        span: ArenaSpan,
        tape: Option<ArenaSpan>,
    },
    /// The `step`-th step's heads, head dims or conv kernel are not the tape's GDN geometry, so
    /// its kernels would address the state pool with other strides than the worker allocated it
    /// with.
    GdnStepGeometry { step: usize },
    /// The tape's own GDN geometry is one the kernels do not take: `conv_kernel` outside
    /// [`GDN_CONV_KERNEL_RANGE`] (0 included, which `GdnStateLayout::new` asserts against), no
    /// key or value heads, value heads not a whole number per key head, `head_k` not
    /// [`GDN_HEAD_K`], `head_v` not a non-zero multiple of [`GDN_HEAD_V_MULTIPLE`] up to
    /// [`GDN_HEAD_V_MAX`], or a `conv_dim` that overflows `u32`. The worker would size its state
    /// pool from it.
    GdnGeometry {
        num_k_heads: u32,
        num_v_heads: u32,
        head_k: u32,
        head_v: u32,
        conv_kernel: u32,
    },
    /// The tape's MoE geometry is one no MoE kernel takes, or its `tile_rows`, `max_tiles` or
    /// sortedness is not the one its bucket has.
    MoeGeometry,
    /// A MoE region is not at a multiple of [`CUDA_ARENA_ALIGN`], reaches past the arena, does not
    /// follow the previous region, or overlaps another region of the tape.
    MoeRegionOutside {
        region: Region,
        span: ArenaSpan,
        total: u64,
    },
    /// A MoE region is not the bytes its geometry sizes it to.
    MoeRegionShape {
        region: Region,
        span: ArenaSpan,
        want: u64,
    },
    /// The `step`-th step is a MoE step and the tape has no MoE scratch, or its geometry is not
    /// the tape's MoE geometry.
    MoeStepGeometry { step: usize },
    /// The `step`-th step's `region` operand is not the tape's MoE region of that name.
    MoeRegionMismatch {
        step: usize,
        region: Region,
        span: ArenaSpan,
        tape: ArenaSpan,
    },
    /// The `step`-th step is not a MoE step (nor the `SiluMul` over the MoE gate and up rows) and
    /// reads or writes the MoE scratch.
    MoeScratchTouched { step: usize, span: ArenaSpan },
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
            Self::TokenIdsShape {
                span,
                bucket_m,
                vocab,
            } => {
                let (bytes, offset) = at(span);
                write!(
                    f,
                    "the token ids are {bytes} bytes at {offset}, over a vocabulary of {vocab}; \
                     the forward fills {bucket_m} u32 at a multiple of {CUDA_ARENA_ALIGN}, from \
                     a vocabulary of at least one row"
                )
            }
            Self::EmbedMissing { first: Some(step) } => write!(
                f,
                "the tape's input is token ids and its first step, step {step}, is not the embed"
            ),
            Self::EmbedMissing { first: None } => {
                write!(
                    f,
                    "the tape's input is token ids and it has no step to embed them"
                )
            }
            Self::EmbedMisplaced { step } => write!(
                f,
                "step {step} is an embed; a tape embeds only as its first step, and only when its \
                 input is token ids"
            ),
            Self::EmbedInput { step } => write!(
                f,
                "step {step}, the embed, reads other token ids, rows or vocabulary than the \
                 tape's input"
            ),
            Self::EmbedOutput { step } => write!(
                f,
                "step {step}, the embed, does not write exactly colour 0, the embedded hidden"
            ),
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
            Self::KvClasses => write!(
                f,
                "the tape's KV attention classes are not a geometry the kernels take, or a layer \
                 names no class"
            ),
            Self::OneRowStep { step, rows } => write!(
                f,
                "step {step} is a one-row kernel's (the GEMV or the decode attention) at \
                 {rows} rows, or in a bucket of more than one row"
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
            Self::GdnRegionOutside {
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
            Self::GdnRegionShape { region, span, want } => write!(
                f,
                "{region} is {} bytes; the GDN steps use {want}",
                span.bytes
            ),
            Self::GdnLayer {
                step,
                layer,
                layers,
            } => write!(
                f,
                "step {step} uses GDN layer index {layer}, and the tape has {layers} GDN layers"
            ),
            Self::GdnRegionMismatch {
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
                            "step {step} uses {region} as {bytes} bytes at {offset}; the tape's \
                             is {tb} bytes at {to}"
                        )
                    }
                    None => write!(
                        f,
                        "step {step} uses {region} as {bytes} bytes at {offset}; the tape has no \
                         GDN layers"
                    ),
                }
            }
            Self::GdnStepGeometry { step } => write!(
                f,
                "step {step}'s heads, head dims or conv kernel are not the tape's GDN geometry"
            ),
            Self::GdnGeometry {
                num_k_heads,
                num_v_heads,
                head_k,
                head_v,
                conv_kernel,
            } => write!(
                f,
                "the tape's GDN geometry ({num_v_heads} value heads of {head_v} over \
                 {num_k_heads} key heads of {head_k}, conv kernel {conv_kernel}) is not one the \
                 CUDA GDN kernels take"
            ),
            Self::MoeGeometry => write!(
                f,
                "the tape's MoE geometry is one the CUDA MoE kernels do not take, or its tiles or \
                 sortedness are not its bucket's"
            ),
            Self::MoeRegionOutside {
                region,
                span,
                total,
            } => {
                let (bytes, offset) = at(span);
                write!(
                    f,
                    "{region} ({bytes} bytes at {offset}) is misaligned, reaches past the \
                     {total}-byte arena, or overlaps another region"
                )
            }
            Self::MoeRegionShape { region, span, want } => write!(
                f,
                "{region} is {} bytes; its geometry sizes it to {want}",
                span.bytes
            ),
            Self::MoeStepGeometry { step } => write!(
                f,
                "step {step} is a MoE step whose geometry is not the tape's MoE scratch's, or the \
                 tape has none"
            ),
            Self::MoeRegionMismatch {
                step,
                region,
                span,
                tape,
            } => {
                let ((bytes, offset), (tb, to)) = (at(span), at(tape));
                write!(
                    f,
                    "step {step} uses {region} as {bytes} bytes at {offset}; the tape's is {tb} \
                     bytes at {to}"
                )
            }
            Self::MoeScratchTouched { step, span } => {
                let (bytes, offset) = at(span);
                write!(
                    f,
                    "step {step} is not a MoE step and touches {bytes} bytes at {offset} of the \
                     MoE scratch"
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
    /// the rotary table, the KV regions and the GDN regions, each region the size its users take.
    /// A token-id input is `bucket_m` u32 and its first step the one embed reading it; a hidden
    /// input has no embed. Every step's spans ([`CudaStep::extents`]) are aligned and inside
    /// `total_bytes`, each span it writes overlaps none of its reads, nor its other write, nor a
    /// region the forward fills (the token ids, the rotary table, the KV regions, the GDN
    /// sequence), and each weight it reads is one the tape declares, declared at least as large as
    /// the step reads. A rope reads the tape's rotary table, a KV step the tape's regions, at a KV
    /// layer the tape has, with the tape's KV geometry; a GDN step the tape's GDN regions, at a GDN
    /// layer the tape has, with its GDN geometry, and no other step touches the GDN scratch. All
    /// arithmetic is checked. The worker runs this on every tape before it allocates anything; its
    /// own weight and pool checks then close the chain from each step to the buffer it reads.
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
        let input = self.input.span();
        if !inside(input, total) {
            return Err(TapeError::InputOutside { span: input, total });
        }
        self.check_embed()?;
        if !inside(self.result, total) {
            return Err(TapeError::ResultOutside {
                span: self.result,
                total,
            });
        }
        if self.rotary_local.is_some() && self.rotary.is_none() {
            return Err(TapeError::KvClasses);
        }
        for t in self.rotary.into_iter().chain(self.rotary_local) {
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
        let mut regions = self.kv_regions(total)?;
        if let TapeInput::TokenIds { span, .. } = self.input {
            regions.push((Region::TokenIds, span));
        }
        let scratch = self.gdn_regions(total, &mut regions)?;
        let moe = self.moe_regions(total, &regions)?;
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
            // The GDN scratch holds one GDN step's intermediates between its launches: only a GDN
            // step reads or writes it. A write over it is refused as a write over a region, a
            // read of it as a step using it that is not the tape's GDN step.
            if let Some(scratch) = scratch
                && !matches!(s, CudaStep::GatedDeltaNet(_))
            {
                for &write in [Some(e.writes), e.second_write].iter().flatten() {
                    if overlap(scratch, write) {
                        return Err(TapeError::WriteOverRegion {
                            step,
                            write,
                            region: Region::GdnScratch,
                        });
                    }
                }
                if let Some(&read) = reads.clone().find(|&&r| overlap(r, scratch)) {
                    return Err(TapeError::GdnRegionMismatch {
                        step,
                        region: Region::GdnScratch,
                        span: read,
                        tape: Some(scratch),
                    });
                }
            }
            if let Some(moe) = moe {
                self.check_moe_scratch(step, s, &e, moe)?;
            }
            self.check_step_operands(step, s)?;
            if let Some(layer) = e.gdn {
                let layers = self.gdn.map_or(0, |g| g.layers.len());
                if layer as usize >= layers {
                    return Err(TapeError::GdnLayer {
                        step,
                        layer,
                        layers,
                    });
                }
            }
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

    /// A token-id input is `bucket_m` u32 at an aligned offset, and the tape's first step is the
    /// embed reading it, with its rows and vocabulary; no other step embeds, and a tape whose
    /// input is the embedded hidden has no embed.
    fn check_embed(&self) -> Result<(), TapeError> {
        let embeds =
            (self.steps.iter().enumerate()).filter(|(_, s)| matches!(s, CudaStep::Embed(_)));
        match self.input {
            TapeInput::Hidden(_) => match embeds.map(|(step, _)| step).next() {
                Some(step) => Err(TapeError::EmbedMisplaced { step }),
                None => Ok(()),
            },
            TapeInput::TokenIds { span, vocab } => {
                let want = u64::from(self.bucket_m) * TOKEN_ID_BYTES;
                if !span.offset.0.is_multiple_of(CUDA_ARENA_ALIGN)
                    || span.bytes != want
                    || vocab == 0
                {
                    return Err(TapeError::TokenIdsShape {
                        span,
                        bucket_m: self.bucket_m,
                        vocab,
                    });
                }
                if let Some((step, _)) = embeds.clone().find(|&(step, _)| step > 0) {
                    return Err(TapeError::EmbedMisplaced { step });
                }
                let Some(&CudaStep::Embed(EmbedBf16 {
                    tokens,
                    out,
                    rows,
                    hidden,
                    vocab: embed_vocab,
                    ..
                })) = self.steps.first()
                else {
                    let first = (!self.steps.is_empty()).then_some(0);
                    return Err(TapeError::EmbedMissing { first });
                };
                if (tokens, rows, embed_vocab) != (span.offset, self.bucket_m, vocab) {
                    return Err(TapeError::EmbedInput { step: 0 });
                }
                // The embed fills colour 0 and nothing else: its offset, and its bytes exactly.
                let colour0 =
                    (self.arena.colour_offsets.first()).zip(self.arena.colour_bytes.first());
                let writes = (u64::from(rows) * u64::from(hidden)).checked_mul(CUDA_ACT_BYTES);
                match colour0 {
                    Some((&offset, &bytes))
                        if out == ArenaOffset(offset) && writes == Some(bytes) =>
                    {
                        Ok(())
                    }
                    _ => Err(TapeError::EmbedOutput { step: 0 }),
                }
            }
        }
    }

    /// The KV classes: at least one, each a geometry the kernels take (non-zero heads, a head dim
    /// of [`ATTN_BF16_HEAD_DIMS`] or the tape's own, a window a non-zero multiple of
    /// [`ATTN_WINDOW_MULTIPLE`]) whose K (or V) row fits the pool's, `kv_heads · head_dim`, which
    /// sizes every layer's region; and one class per layer naming a class the tape has.
    fn kv_classes(&self, kv: &KvTape) -> Result<(), TapeError> {
        if kv.classes.is_empty()
            || kv.layer_class.len() != kv.layers.len()
            || kv
                .layer_class
                .iter()
                .any(|&c| usize::from(c) >= kv.classes.len())
        {
            return Err(TapeError::KvClasses);
        }
        for c in kv.classes {
            let window_ok = c
                .window
                .is_none_or(|w| w != 0 && w.is_multiple_of(ATTN_WINDOW_MULTIPLE));
            let head_ok = c.head_dim == kv.head_dim || ATTN_BF16_HEAD_DIMS.contains(&c.head_dim);
            let row = u64::from(c.kv_heads) * u64::from(c.head_dim);
            let fits = row <= u64::from(kv.kv_heads) * u64::from(kv.head_dim);
            if c.kv_heads == 0 || c.head_dim == 0 || !head_ok || !window_ok || !fits {
                return Err(TapeError::KvClasses);
            }
        }
        Ok(())
    }

    /// The class of KV layer index `layer`, when the tape has it.
    pub fn kv_class(&self, layer: u32) -> Option<KvClass> {
        let kv = self.kv?;
        let c = *kv.layer_class.get(layer as usize)?;
        kv.classes.get(usize::from(c)).copied()
    }

    /// Whether a KV step of `(kv_heads, head_dim, window)` at KV layer index `layer` has its
    /// layer's class geometry; a class-0 layer's is also the tape's `kv_heads` and `head_dim`, the
    /// pool's. A layer the tape lacks is checked against class 0 here, and refused as `KvLayer`
    /// by the extents check, which runs first.
    fn kv_step_fits(&self, layer: u32, step: (u32, u32, Option<u32>)) -> bool {
        let Some(kv) = self.kv else {
            return false;
        };
        let c = (kv.layer_class.get(layer as usize)).map_or(0, |&c| usize::from(c));
        let Some(class) = kv.classes.get(c) else {
            return false;
        };
        step == (class.kv_heads, class.head_dim, class.window)
            && (c != 0 || (step.0, step.1) == (kv.kv_heads, kv.head_dim))
    }

    /// The regions the forward fills before the steps run: the rotary table and the KV regions,
    /// each checked to be aligned, inside the arena, and the size the forward writes. The rotary
    /// table's own checks ran before this.
    fn kv_regions(&self, total: u64) -> Result<Vec<(Region, ArenaSpan)>, TapeError> {
        let mut regions: Vec<(Region, ArenaSpan)> = (self.rotary.into_iter())
            .chain(self.rotary_local)
            .map(|t| (Region::Rotary, t.span))
            .collect();
        if let [(_, a), (_, b)] = regions[..]
            && overlap(a, b)
        {
            return Err(TapeError::RotaryOutside { span: b, total });
        }
        let Some(kv) = self.kv else {
            return Ok(regions);
        };
        self.kv_classes(&kv)?;
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

    /// The GDN regions, when the tape has GDN layers: the sequence region (which the forward
    /// fills, so it joins `regions`) and the scratch, each checked to be aligned, inside the
    /// arena, and the size its users take. Returns the scratch's span.
    fn gdn_regions(
        &self,
        total: u64,
        regions: &mut Vec<(Region, ArenaSpan)>,
    ) -> Result<Option<ArenaSpan>, TapeError> {
        let Some(gdn) = self.gdn else {
            return Ok(None);
        };
        let GdnTape {
            num_k_heads,
            num_v_heads,
            head_k,
            head_v,
            conv_kernel,
            ..
        } = gdn;
        let (kmin, kmax) = GDN_CONV_KERNEL_RANGE;
        let takes = (kmin..=kmax).contains(&conv_kernel)
            && num_k_heads > 0
            && num_v_heads > 0
            && num_v_heads.is_multiple_of(num_k_heads)
            && head_k == GDN_HEAD_K
            && head_v > 0
            && head_v.is_multiple_of(GDN_HEAD_V_MULTIPLE)
            && head_v <= GDN_HEAD_V_MAX
            && gdn_conv_dim(num_k_heads, head_k, num_v_heads, head_v).is_some();
        if !takes {
            return Err(TapeError::GdnGeometry {
                num_k_heads,
                num_v_heads,
                head_k,
                head_v,
                conv_kernel,
            });
        }
        // A scratch whose size overflows is one no span can match: `want` 0 never equals a
        // non-empty scratch's bytes, and an empty scratch is refused as shape below.
        let scratch_want = gdn.scratch_bytes(self.bucket_m).unwrap_or(0);
        for (region, span, want) in [
            (Region::GdnSequence, gdn.sequence, GDN_SEQUENCE_BYTES),
            (Region::GdnScratch, gdn.scratch, scratch_want),
        ] {
            if !(span.offset.0.is_multiple_of(CUDA_ARENA_ALIGN) && inside(span, total)) {
                return Err(TapeError::GdnRegionOutside {
                    region,
                    span,
                    total,
                });
            }
            if want == 0 || span.bytes != want {
                return Err(TapeError::GdnRegionShape { region, span, want });
            }
        }
        regions.push((Region::GdnSequence, gdn.sequence));
        Ok(Some(gdn.scratch))
    }

    /// A rope's or rope-append's table is the tape's rotary table, a KV step's regions and
    /// geometry are the tape's, and a GDN step's regions and geometry are the tape's GDN ones.
    fn check_step_operands(&self, step: usize, s: &CudaStep) -> Result<(), TapeError> {
        let rotary = |table: ArenaOffset| {
            let tape = self.rotary.map(|t| t.span.offset);
            if tape == Some(table) || self.rotary_local.map(|t| t.span.offset) == Some(table) {
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
                layer,
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
                let kv_heads = cols.checked_div(head_dim).unwrap_or(0);
                let window = self.kv_class(layer).and_then(|c| c.window);
                match cols.is_multiple_of(head_dim.max(1))
                    && self.kv_step_fits(layer, (kv_heads, head_dim, window))
                {
                    true => Ok(()),
                    false => Err(TapeError::KvStepGeometry { step }),
                }
            }
            CudaStep::AttnPrefill(AttnPrefillBf16 {
                block_table,
                sequence,
                layer,
                kv_heads,
                head_dim,
                window,
                ..
            })
            | CudaStep::AttnDecode(AttnPrefillBf16 {
                block_table,
                sequence,
                layer,
                kv_heads,
                head_dim,
                window,
                ..
            }) => {
                self.one_row(step, s)?;
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
                match self.kv_step_fits(layer, (kv_heads, head_dim, window)) {
                    true => Ok(()),
                    false => Err(TapeError::KvStepGeometry { step }),
                }
            }
            CudaStep::GatedDeltaNet(
                g @ GatedDeltaNetBf16 {
                    sequence,
                    scratch,
                    num_k_heads,
                    num_v_heads,
                    head_k,
                    head_v,
                    conv_kernel,
                    ..
                },
            ) => {
                let mismatch = |region, span, tape| TapeError::GdnRegionMismatch {
                    step,
                    region,
                    span,
                    tape,
                };
                let seq = ArenaSpan {
                    offset: sequence,
                    bytes: GDN_SEQUENCE_BYTES,
                };
                let Some(gdn) = self.gdn else {
                    return Err(mismatch(Region::GdnSequence, seq, None));
                };
                if (num_k_heads, num_v_heads, head_k, head_v, conv_kernel)
                    != (
                        gdn.num_k_heads,
                        gdn.num_v_heads,
                        gdn.head_k,
                        gdn.head_v,
                        gdn.conv_kernel,
                    )
                {
                    return Err(TapeError::GdnStepGeometry { step });
                }
                if gdn.sequence != seq {
                    return Err(mismatch(Region::GdnSequence, seq, Some(gdn.sequence)));
                }
                // `extents` already sized the scratch, so its layout does not overflow.
                let sc = ArenaSpan {
                    offset: scratch,
                    bytes: g.scratch_layout().map_or(0, |l| l.bytes),
                };
                if gdn.scratch != sc {
                    return Err(mismatch(Region::GdnScratch, sc, Some(gdn.scratch)));
                }
                Ok(())
            }
            CudaStep::Gemv(_) => self.one_row(step, s),
            // Gemma's router norm writes the scratch's router-norm region, whole.
            CudaStep::RouterNorm(RmsNormBf16 { y, rows, cols, .. }) => match self.moe {
                Some(moe)
                    if Some(y) == moe.router_norm.map(|r| r.offset)
                        && (rows, cols) == (self.bucket_m, moe.hidden) =>
                {
                    Ok(())
                }
                Some(moe) => Err(TapeError::MoeRegionMismatch {
                    step,
                    region: Region::MoeRouterNorm,
                    span: ArenaSpan {
                        offset: y,
                        bytes: u64::from(rows) * u64::from(cols) * CUDA_ACT_BYTES,
                    },
                    tape: moe.router_norm.unwrap_or(moe.logits),
                }),
                None => Err(TapeError::MoeStepGeometry { step }),
            },
            CudaStep::RouterLogits(_)
            | CudaStep::MoeRoute(_)
            | CudaStep::ExpertSort(_)
            | CudaStep::ExpertGemm(_)
            | CudaStep::ExpertGemv(_)
            | CudaStep::ExpertCombine(_) => self.check_moe_step(step, s),
            CudaStep::Embed(_)
            | CudaStep::Gemm(_)
            | CudaStep::GemmNarrow(_)
            | CudaStep::RmsNorm(_)
            | CudaStep::Add(_)
            | CudaStep::SiluMul(_)
            | CudaStep::GateSplit(_)
            | CudaStep::GateApply(_)
            | CudaStep::GateScale(_)
            | CudaStep::Scale(_)
            | CudaStep::RmsNormUnit(_)
            | CudaStep::GeluMul(_)
            | CudaStep::LayerScale(_)
            | CudaStep::SoftCap(_)
            | CudaStep::GemmCol64(_) => Ok(()),
        }
    }

    /// The MoE scratch, when the tape has one: its geometry one the kernels take and its bucket's
    /// (`tile_rows` [`CUDA_EXPERT_TILE_ROWS`], `max_tiles` and sortedness the bucket's pairs'),
    /// each region aligned, inside the arena, after the previous one, sized by the geometry
    /// ([`moe_region_bytes`]), and overlapping no region the forward fills. Returns the scratch's
    /// span.
    fn moe_regions(
        &self,
        total: u64,
        regions: &[(Region, ArenaSpan)],
    ) -> Result<Option<ArenaSpan>, TapeError> {
        let Some(moe) = self.moe else {
            return Ok(None);
        };
        let sorted = self.bucket_m.checked_mul(moe.top_k) >= Some(crate::op_abi::CUDA_SORTED_PAIRS);
        let tiles = moe_tiles(self.bucket_m, moe.experts, moe.top_k).map(|(_, t, _)| t);
        let geometry = (moe.experts, moe.top_k, moe.hidden, moe.inter);
        if moe.experts == 0
            || moe.top_k == 0
            || moe.top_k > moe.experts
            || moe.tile_rows != CUDA_EXPERT_TILE_ROWS
            || moe.sort.is_some() != sorted
            || Some(moe.max_tiles) != if sorted { tiles } else { Some(0) }
        {
            return Err(TapeError::MoeGeometry);
        }
        let mut want =
            moe_region_bytes(self.bucket_m, geometry, sorted).ok_or(TapeError::MoeGeometry)?;
        if moe.router_norm.is_some() {
            let bytes = u64::from(self.bucket_m) * u64::from(moe.hidden) * CUDA_ACT_BYTES;
            want.push(bytes);
        }
        let mut next = 0u64;
        for (&(region, span), want) in moe.regions().iter().zip(want) {
            let outside = TapeError::MoeRegionOutside {
                region,
                span,
                total,
            };
            if !(span.offset.0.is_multiple_of(CUDA_ARENA_ALIGN) && inside(span, total))
                || span.offset.0 < next
            {
                return Err(outside);
            }
            if regions.iter().any(|&(_, r)| overlap(r, span)) {
                return Err(outside);
            }
            if span.bytes != want {
                return Err(TapeError::MoeRegionShape { region, span, want });
            }
            next = span.offset.0 + span.bytes;
        }
        Ok(Some(moe.span()))
    }

    /// No step but a MoE step reads or writes the MoE scratch, except the `SiluMul` over its gate
    /// and up rows into its activation rows; and no colour lies inside it.
    fn check_moe_scratch(
        &self,
        step: usize,
        s: &CudaStep,
        e: &crate::step::StepExtents,
        scratch: ArenaSpan,
    ) -> Result<(), TapeError> {
        let moe_step = matches!(
            s,
            CudaStep::RouterLogits(_)
                | CudaStep::RouterNorm(_)
                | CudaStep::MoeRoute(_)
                | CudaStep::ExpertSort(_)
                | CudaStep::ExpertGemm(_)
                | CudaStep::ExpertGemv(_)
                | CudaStep::ExpertCombine(_)
        );
        if moe_step {
            return Ok(());
        }
        let writes = [Some(e.writes), e.second_write];
        let touched = (e.reads.iter().flatten())
            .chain(writes.iter().flatten())
            .find(|&&span| overlap(span, scratch));
        let Some(&span) = touched else {
            return Ok(());
        };
        if let (
            CudaStep::SiluMul(SiluMulBf16 { gate, up, out, .. })
            | CudaStep::GeluMul(SiluMulBf16 { gate, up, out, .. }),
            Some(moe),
        ) = (s, self.moe)
        {
            let (g, u, a) = (e.reads[0], e.reads[1], e.writes);
            if (*gate, *up, *out) == (moe.gate.offset, moe.up.offset, moe.act.offset)
                && g == Some(moe.gate)
                && u == Some(moe.up)
                && a == moe.act
            {
                return Ok(());
            }
            return Err(TapeError::MoeRegionMismatch {
                step,
                region: Region::MoeAct,
                span: a,
                tape: moe.act,
            });
        }
        Err(TapeError::MoeScratchTouched { step, span })
    }

    /// A MoE step's geometry is the tape's MoE geometry, and every region it names is the tape's
    /// region of that name, whole. Its weight's bytes are checked with every other step's.
    fn check_moe_step(&self, step: usize, s: &CudaStep) -> Result<(), TapeError> {
        let geometry = TapeError::MoeStepGeometry { step };
        let moe = self.moe.ok_or(geometry)?;
        let rows = moe.expert_rows(self.bucket_m);
        let want = |region: Region, at: ArenaOffset, tape: ArenaSpan| {
            if at == tape.offset {
                Ok(())
            } else {
                Err(TapeError::MoeRegionMismatch {
                    step,
                    region,
                    span: ArenaSpan {
                        offset: at,
                        bytes: tape.bytes,
                    },
                    tape,
                })
            }
        };
        let same = |e: u32, h: u32, i: u32| (e, h, i) == (moe.experts, moe.hidden, moe.inter);
        match *s {
            CudaStep::RouterLogits(RouterLogitsBf16 {
                logits,
                m,
                experts,
                hidden,
                inter,
                ..
            }) => {
                if m != self.bucket_m || !same(experts, hidden, inter) {
                    return Err(geometry);
                }
                want(Region::MoeLogits, logits, moe.logits)
            }
            CudaStep::MoeRoute(MoeRouteBf16 {
                logits,
                topk_idx,
                scores,
                m,
                experts,
                top_k,
                ..
            }) => {
                if (m, experts, top_k) != (self.bucket_m, moe.experts, moe.top_k) {
                    return Err(geometry);
                }
                want(Region::MoeLogits, logits, moe.logits)?;
                want(Region::MoeTopKIdx, topk_idx, moe.topk_idx)?;
                want(Region::MoeScores, scores, moe.scores)
            }
            CudaStep::ExpertSort(ExpertSortBf16 {
                topk_idx,
                chunk_hist,
                counts,
                offsets,
                tile_expert,
                pos,
                rows: sorted_rows,
                m,
                experts,
                top_k,
                hidden,
                max_tiles,
                ..
            }) => {
                let sort = moe.sort.ok_or(geometry)?;
                if (m, experts, top_k, hidden, max_tiles)
                    != (
                        self.bucket_m,
                        moe.experts,
                        moe.top_k,
                        moe.hidden,
                        moe.max_tiles,
                    )
                {
                    return Err(geometry);
                }
                want(Region::MoeTopKIdx, topk_idx, moe.topk_idx)?;
                want(Region::MoeChunkHist, chunk_hist, sort.chunk_hist)?;
                want(Region::MoeCounts, counts, sort.counts)?;
                want(Region::MoeOffsets, offsets, sort.offsets)?;
                want(Region::MoeTileExpert, tile_expert, sort.tile_expert)?;
                want(Region::MoePos, pos, sort.pos)?;
                want(Region::MoeRows, sorted_rows, sort.rows)
            }
            CudaStep::ExpertGemm(
                g @ ExpertGemmBf16 {
                    a,
                    offsets,
                    tile_expert,
                    c,
                    max_tiles,
                    experts,
                    hidden,
                    inter,
                    ..
                },
            ) => {
                let sort = moe.sort.ok_or(geometry)?;
                if max_tiles != moe.max_tiles || !same(experts, hidden, inter) {
                    return Err(geometry);
                }
                want(Region::MoeOffsets, offsets, sort.offsets)?;
                want(Region::MoeTileExpert, tile_expert, sort.tile_expert)?;
                let (input, output) = moe.projection_regions(g.proj, Some(sort.rows));
                want(input.0, a, input.1)?;
                want(output.0, c, output.1)
            }
            CudaStep::ExpertGemv(
                g @ ExpertGemvBf16 {
                    a,
                    topk_idx,
                    c,
                    m,
                    top_k,
                    experts,
                    hidden,
                    inter,
                    ..
                },
            ) => {
                if moe.sort.is_some()
                    || (m, top_k) != (self.bucket_m, moe.top_k)
                    || !same(experts, hidden, inter)
                {
                    return Err(geometry);
                }
                want(Region::MoeTopKIdx, topk_idx, moe.topk_idx)?;
                let (input, output) = moe.projection_regions(g.proj, None);
                // The gate and up read the token rows, an arena colour the step's extents check.
                if g.proj.reads_pairs() {
                    want(input.0, a, input.1)?;
                }
                want(output.0, c, output.1)
            }
            CudaStep::ExpertCombine(ExpertCombineBf16 {
                down,
                pos,
                scores,
                m,
                top_k,
                hidden,
                down_rows,
                ..
            }) => {
                if (m, top_k, hidden, down_rows) != (self.bucket_m, moe.top_k, moe.hidden, rows) {
                    return Err(geometry);
                }
                want(Region::MoeDown, down, moe.down)?;
                want(Region::MoeScores, scores, moe.scores)?;
                match (pos, moe.sort) {
                    (Some(p), Some(sort)) => want(Region::MoePos, p, sort.pos),
                    (None, None) => Ok(()),
                    _ => Err(geometry),
                }
            }
            _ => Ok(()),
        }
    }

    /// A one-row kernel's step (the GEMV, the decode attention) is one row, in a one-row bucket:
    /// those kernels index no row past the first. Any other step passes.
    fn one_row(&self, step: usize, s: &CudaStep) -> Result<(), TapeError> {
        let rows = match *s {
            CudaStep::Gemv(GemmBf16 { m, .. }) => m,
            CudaStep::AttnDecode(AttnPrefillBf16 { rows, .. }) => rows,
            _ => return Ok(()),
        };
        if rows == 1 && self.bucket_m == 1 {
            Ok(())
        } else {
            Err(TapeError::OneRowStep { step, rows })
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
        AddBf16, AttnPrefillBf16, CUDA_MAX_BLOCKS_PER_SEQ, GateSplitBf16, GatedDeltaNetBf16,
        GemmBf16, KvUse, MoeRouteBf16, RmsNormBf16, RopeAppendBf16, RopeBf16, RopePairing,
        RouteProgram, WeightRead,
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
        input: TapeInput::Hidden(span(0, 262144)),
        result: span(786432, 524288),
        weight_bytes: &[0, 1048576, 2048],
        steps: GOOD_STEPS,
        rotary: None,
        kv: None,
        gdn: None,
        moe: None,
        rotary_local: None,
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

    /// [`GOOD`]'s token-id form: `x` gathered from a `[100, 512]` table (weight 0) by 256 ids at
    /// the region after the colours, then [`GOOD_STEPS`].
    const TOKENS: u64 = 1310720;
    const VOCAB: u32 = 100;

    fn embed_step(tokens: u64, rows: u32, vocab: u32) -> CudaStep {
        CudaStep::Embed(EmbedBf16 {
            tokens: ArenaOffset(tokens),
            table: WeightIx(0),
            out: ArenaOffset(0),
            rows,
            hidden: 512,
            vocab,
        })
    }

    fn token_tape(steps: &[CudaStep]) -> CudaTape {
        CudaTape {
            arena: ArenaLayout {
                total_bytes: TOKENS + 1024,
                ..GOOD.arena
            },
            input: TapeInput::TokenIds {
                span: span(TOKENS, 1024),
                vocab: VOCAB,
            },
            weight_bytes: &[102400, 1048576, 2048],
            steps: Box::leak(steps.to_vec().into_boxed_slice()),
            ..GOOD
        }
    }

    fn embedded(rest: &[CudaStep]) -> CudaTape {
        let mut steps = vec![embed_step(TOKENS, 256, VOCAB)];
        steps.extend_from_slice(rest);
        token_tape(&steps)
    }

    #[test]
    fn a_token_id_tape_validates() {
        assert_eq!(embedded(GOOD_STEPS).validate(), Ok(()));
        let e = embed_step(TOKENS, 256, VOCAB)
            .extents()
            .expect("the embed's extents");
        assert_eq!(
            e.reads[0],
            Some(span(TOKENS, 1024)),
            "the embed reads the token ids"
        );
        assert_eq!(e.writes, span(0, 262144), "the embed writes colour 0");
        assert_eq!(
            e.weight,
            Some(WeightRead {
                ix: WeightIx(0),
                bytes: 102400
            }),
            "the embed reads the whole table"
        );
    }

    /// A token-id tape whose first step is not the embed, or that has no step.
    #[test]
    fn a_token_id_tape_starts_with_its_embed() {
        assert_eq!(
            token_tape(GOOD_STEPS).validate(),
            Err(TapeError::EmbedMissing { first: Some(0) })
        );
        assert_eq!(
            token_tape(&[]).validate(),
            Err(TapeError::EmbedMissing { first: None })
        );
    }

    /// An embed in a hidden-input tape, or after the first step of a token-id tape.
    #[test]
    fn an_embed_is_only_a_token_id_tapes_first_step() {
        assert_eq!(
            with_step(embed_step(TOKENS, 256, VOCAB)).validate(),
            Err(TapeError::EmbedMisplaced { step: 0 })
        );
        let mut steps = vec![embed_step(TOKENS, 256, VOCAB)];
        steps.extend_from_slice(GOOD_STEPS);
        steps.push(embed_step(TOKENS, 256, VOCAB));
        assert_eq!(
            token_tape(&steps).validate(),
            Err(TapeError::EmbedMisplaced { step: 3 })
        );
    }

    /// An embed reading other token ids, rows or vocabulary than the tape's input.
    #[test]
    fn the_embed_reads_the_tapes_input() {
        for embed in [
            embed_step(0, 256, VOCAB),
            embed_step(TOKENS, 128, VOCAB),
            embed_step(TOKENS, 256, VOCAB + 1),
        ] {
            assert_eq!(
                token_tape(&[embed]).validate(),
                Err(TapeError::EmbedInput { step: 0 }),
                "{embed:?}"
            );
        }
    }

    /// Token ids that are not `bucket_m` u32 at an aligned offset, or a vocabulary of no rows.
    #[test]
    fn token_ids_shape() {
        for (input_span, vocab) in [
            (span(TOKENS, 512), VOCAB),
            (span(TOKENS + 4, 1024), VOCAB),
            (span(TOKENS, 1024), 0),
        ] {
            let t = CudaTape {
                arena: ArenaLayout {
                    total_bytes: TOKENS + 2048,
                    ..GOOD.arena
                },
                input: TapeInput::TokenIds {
                    span: input_span,
                    vocab,
                },
                ..embedded(GOOD_STEPS)
            };
            assert_eq!(
                t.validate(),
                Err(TapeError::TokenIdsShape {
                    span: input_span,
                    bucket_m: 256,
                    vocab
                }),
                "{input_span:?} vocab {vocab}"
            );
        }
    }

    /// A step writing over the token ids, which the forward fills before the steps run.
    #[test]
    fn nothing_writes_the_token_ids() {
        let t = embedded(&[gemm(0, 1, TOKENS, (1, 256, 512))]);
        assert_eq!(
            t.validate(),
            Err(TapeError::WriteOverRegion {
                step: 1,
                write: span(TOKENS, 512),
                region: Region::TokenIds,
            })
        );
    }

    /// An embedding table declared shorter than `vocab · hidden` bf16.
    #[test]
    fn an_embed_table_declared_short_is_refused() {
        let t = CudaTape {
            weight_bytes: &[102400 - 2, 1048576, 2048],
            ..embedded(GOOD_STEPS)
        };
        assert_eq!(
            t.validate(),
            Err(TapeError::WeightShort {
                step: 0,
                ix: 0,
                need: 102400,
                declared: 102398
            })
        );
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
            input: TapeInput::Hidden(span(1310720, 256)),
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
            pairing: RopePairing::HalfRotated,
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
            pairing: RopePairing::HalfRotated,
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
            window: None,
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
        classes: &[KvClass {
            kv_heads: 2,
            head_dim: 256,
            window: None,
        }],
        layer_class: &[0],
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

    // ── The GDN state's regions ────────────────────────────────────

    /// A 64-row GDN tape of one key head and one value head of 32: `qkv [64, 288]` at 0, `z [64,
    /// 32]` at 36864, `a` and `b [64, 1]` at 40960 and 41216, `out [64, 32]` at 41472; the GDN
    /// sequence at 45568 and the scratch (82432 bytes) at 45824. Weight 1 is the bundle.
    const GDN_SEQ: u64 = 45568;
    const GDN_SCRATCH: u64 = 45824;
    const GDN_SCRATCH_BYTES: u64 = 82432;
    const GDN_TOTAL: u64 = GDN_SCRATCH + GDN_SCRATCH_BYTES;
    const GDN_BUNDLE_BYTES: u64 = 5376;

    const GDN: GdnTape = GdnTape {
        layers: &[0],
        num_k_heads: 1,
        num_v_heads: 1,
        head_k: 128,
        head_v: 32,
        conv_kernel: 4,
        sequence: span(GDN_SEQ, 16),
        scratch: span(GDN_SCRATCH, GDN_SCRATCH_BYTES),
    };

    fn gdn_step(layer: u32, sequence: u64, scratch: u64) -> GatedDeltaNetBf16 {
        GatedDeltaNetBf16 {
            qkv: ArenaOffset(0),
            z: ArenaOffset(36864),
            a: ArenaOffset(40960),
            b: ArenaOffset(41216),
            bundle: WeightIx(1),
            out: ArenaOffset(41472),
            scratch: ArenaOffset(scratch),
            sequence: ArenaOffset(sequence),
            layer,
            rows: 64,
            num_k_heads: 1,
            num_v_heads: 1,
            head_k: 128,
            head_v: 32,
            conv_kernel: 4,
            eps_bits: 1e-6f32.to_bits(),
        }
    }

    fn good_gdn_step() -> CudaStep {
        CudaStep::GatedDeltaNet(gdn_step(0, GDN_SEQ, GDN_SCRATCH))
    }

    /// The GDN tape of `gdn` and `steps`.
    fn gdn_tape(gdn: Option<GdnTape>, steps: &[CudaStep]) -> CudaTape {
        CudaTape {
            bucket_m: 64,
            arena: ArenaLayout {
                colour_bytes: &[36864, 4096, 128, 128, 4096],
                colour_offsets: &[0, 36864, 40960, 41216, 41472],
                total_bytes: GDN_TOTAL,
            },
            input: TapeInput::Hidden(span(0, 36864)),
            result: span(41472, 4096),
            weight_bytes: &[0, GDN_BUNDLE_BYTES],
            steps: Box::leak(steps.to_vec().into_boxed_slice()),
            rotary: None,
            kv: None,
            gdn,
            moe: None,
            rotary_local: None,
        }
    }

    #[test]
    fn a_gdn_tape_validates() {
        let t = gdn_tape(Some(GDN), &[good_gdn_step()]);
        assert_eq!(t.validate(), Ok(()));
        let Some(e) = good_gdn_step().extents() else {
            panic!("the GDN step's extents fit u64");
        };
        assert_eq!(e.gdn, Some(0));
        assert_eq!(e.second_write, Some(span(GDN_SCRATCH, GDN_SCRATCH_BYTES)));
        assert_eq!(
            e.weight,
            Some(WeightRead {
                ix: WeightIx(1),
                bytes: GDN_BUNDLE_BYTES
            })
        );
        assert_eq!(
            e.reads,
            [
                Some(span(0, 36864)),
                Some(span(36864, 4096)),
                Some(span(40960, 128)),
                Some(span(41216, 128)),
            ]
        );
    }

    #[test]
    fn gdn_region_outside() {
        for (gdn, region, at) in [
            (
                GdnTape {
                    sequence: span(GDN_SEQ + 16, 16),
                    ..GDN
                },
                Region::GdnSequence,
                span(GDN_SEQ + 16, 16),
            ),
            (
                GdnTape {
                    scratch: span(GDN_SCRATCH + 256, GDN_SCRATCH_BYTES),
                    ..GDN
                },
                Region::GdnScratch,
                span(GDN_SCRATCH + 256, GDN_SCRATCH_BYTES),
            ),
        ] {
            assert_eq!(
                gdn_tape(Some(gdn), &[]).validate(),
                Err(TapeError::GdnRegionOutside {
                    region,
                    span: at,
                    total: GDN_TOTAL
                }),
                "{region}"
            );
        }
    }

    #[test]
    fn gdn_region_shape() {
        for (gdn, region, at, want) in [
            (
                GdnTape {
                    sequence: span(GDN_SEQ, 8),
                    ..GDN
                },
                Region::GdnSequence,
                span(GDN_SEQ, 8),
                16,
            ),
            (
                GdnTape {
                    scratch: span(GDN_SCRATCH, GDN_SCRATCH_BYTES - 256),
                    ..GDN
                },
                Region::GdnScratch,
                span(GDN_SCRATCH, GDN_SCRATCH_BYTES - 256),
                GDN_SCRATCH_BYTES,
            ),
        ] {
            assert_eq!(
                gdn_tape(Some(gdn), &[]).validate(),
                Err(TapeError::GdnRegionShape {
                    region,
                    span: at,
                    want
                }),
                "{region}"
            );
        }
    }

    /// A GDN step at a layer index past the tape's layers, and one on a tape with none.
    #[test]
    fn gdn_layer() {
        let step = CudaStep::GatedDeltaNet(gdn_step(1, GDN_SEQ, GDN_SCRATCH));
        assert_eq!(
            gdn_tape(Some(GDN), &[step]).validate(),
            Err(TapeError::GdnLayer {
                step: 0,
                layer: 1,
                layers: 1
            })
        );
    }

    /// A GDN step whose sequence or scratch is not the tape's, and one on a tape with no GDN.
    #[test]
    fn gdn_region_mismatch() {
        let step = CudaStep::GatedDeltaNet(gdn_step(0, GDN_SEQ - 256, GDN_SCRATCH));
        let t = CudaTape {
            arena: ArenaLayout {
                colour_bytes: &[36864, 4096, 128, 128, 4096, 0],
                colour_offsets: &[0, 36864, 40960, 41216, 41472, 45312],
                total_bytes: GDN_TOTAL,
            },
            ..gdn_tape(Some(GDN), &[step])
        };
        assert_eq!(
            t.validate(),
            Err(TapeError::GdnRegionMismatch {
                step: 0,
                region: Region::GdnSequence,
                span: span(GDN_SEQ - 256, 16),
                tape: Some(GDN.sequence),
            })
        );
        // The tape's scratch moved one slot later, the step's left where it was.
        let later = GdnTape {
            scratch: span(GDN_SCRATCH + 256, GDN_SCRATCH_BYTES),
            ..GDN
        };
        let t = CudaTape {
            arena: ArenaLayout {
                total_bytes: GDN_TOTAL + 256,
                ..gdn_tape(None, &[]).arena
            },
            ..gdn_tape(Some(later), &[good_gdn_step()])
        };
        assert_eq!(
            t.validate(),
            Err(TapeError::GdnRegionMismatch {
                step: 0,
                region: Region::GdnScratch,
                span: span(GDN_SCRATCH, GDN_SCRATCH_BYTES),
                tape: Some(later.scratch),
            })
        );
        assert_eq!(
            gdn_tape(None, &[good_gdn_step()]).validate(),
            Err(TapeError::GdnRegionMismatch {
                step: 0,
                region: Region::GdnSequence,
                span: span(GDN_SEQ, 16),
                tape: None,
            })
        );
    }

    /// A GDN step whose conv kernel or head layout is not the tape's, both valid geometries (each
    /// change keeps the scratch the size both declare, so only the step and the tape disagree).
    #[test]
    fn gdn_step_geometry() {
        let k3 = GdnTape {
            conv_kernel: 3,
            ..GDN
        };
        assert_eq!(
            gdn_tape(Some(k3), &[good_gdn_step()]).validate(),
            Err(TapeError::GdnStepGeometry { step: 0 })
        );
        // The step claims a 3-tap conv over the tape's 4: another ring stride than the pool's.
        let step = CudaStep::GatedDeltaNet(GatedDeltaNetBf16 {
            conv_kernel: 3,
            ..gdn_step(0, GDN_SEQ, GDN_SCRATCH)
        });
        assert_eq!(
            gdn_tape(Some(GDN), &[step]).validate(),
            Err(TapeError::GdnStepGeometry { step: 0 })
        );
    }

    /// A hand-written tape whose own GDN geometry the kernels do not take is refused by `validate`,
    /// before `CudaWorker::new` would size a pool from it (and `GdnStateLayout::new` assert on a
    /// conv kernel of 0).
    #[test]
    fn gdn_geometry() {
        for gdn in [
            GdnTape {
                conv_kernel: 0,
                ..GDN
            },
            GdnTape {
                conv_kernel: 1,
                ..GDN
            },
            GdnTape {
                conv_kernel: 9,
                ..GDN
            },
            GdnTape {
                num_k_heads: 0,
                ..GDN
            },
            GdnTape {
                num_v_heads: 0,
                ..GDN
            },
            GdnTape {
                num_k_heads: 2,
                num_v_heads: 3,
                ..GDN
            },
            GdnTape { head_k: 64, ..GDN },
            GdnTape { head_v: 40, ..GDN },
            GdnTape { head_v: 512, ..GDN },
            GdnTape {
                num_v_heads: u32::MAX,
                ..GDN
            },
        ] {
            assert_eq!(
                gdn_tape(Some(gdn), &[]).validate(),
                Err(TapeError::GdnGeometry {
                    num_k_heads: gdn.num_k_heads,
                    num_v_heads: gdn.num_v_heads,
                    head_k: gdn.head_k,
                    head_v: gdn.head_v,
                    conv_kernel: gdn.conv_kernel,
                }),
                "{gdn:?}"
            );
        }
    }

    /// A step writing the GDN sequence region, which the forward fills; a step that is not the
    /// GDN step writing or reading the GDN scratch.
    #[test]
    fn nothing_but_the_gdn_step_touches_its_regions() {
        let over_seq = gemm(0, 1, GDN_SEQ, (1, 8, 64));
        assert_eq!(
            gdn_tape(Some(GDN), &[over_seq]).validate(),
            Err(TapeError::WriteOverRegion {
                step: 0,
                write: span(GDN_SEQ, 16),
                region: Region::GdnSequence,
            })
        );
        let over_scratch = CudaStep::Add(AddBf16 {
            a: ArenaOffset(0),
            b: ArenaOffset(36864),
            out: ArenaOffset(GDN_SCRATCH),
            rows: 8,
            cols: 256,
        });
        assert_eq!(
            gdn_tape(Some(GDN), &[over_scratch]).validate(),
            Err(TapeError::WriteOverRegion {
                step: 0,
                write: span(GDN_SCRATCH, 4096),
                region: Region::GdnScratch,
            })
        );
        let reads_scratch = CudaStep::Add(AddBf16 {
            a: ArenaOffset(GDN_SCRATCH),
            b: ArenaOffset(36864),
            out: ArenaOffset(0),
            rows: 8,
            cols: 256,
        });
        assert_eq!(
            gdn_tape(Some(GDN), &[reads_scratch]).validate(),
            Err(TapeError::GdnRegionMismatch {
                step: 0,
                region: Region::GdnScratch,
                span: span(GDN_SCRATCH, 4096),
                tape: Some(GDN.scratch),
            })
        );
    }

    /// The GDN step's bundle read at its packed bytes: a weight declared one region short is
    /// refused.
    #[test]
    fn a_gdn_bundle_declared_short_is_refused() {
        let t = CudaTape {
            weight_bytes: &[0, GDN_BUNDLE_BYTES - 256],
            ..gdn_tape(Some(GDN), &[good_gdn_step()])
        };
        assert_eq!(
            t.validate(),
            Err(TapeError::WeightShort {
                step: 0,
                ix: 1,
                need: GDN_BUNDLE_BYTES,
                declared: GDN_BUNDLE_BYTES - 256
            })
        );
    }

    // ── Display, and the checks #830 added ─────────────────────────

    /// One of every [`TapeError`], each region named where an arm takes one. The match below has
    /// no wildcard, so a new variant does not compile until it is listed here.
    fn every_tape_error() -> Vec<TapeError> {
        let s = span(256, 512);
        let all = vec![
            TapeError::ColourCount {
                bytes: 2,
                offsets: 3,
            },
            TapeError::ColourMisaligned {
                colour: 1,
                offset: 4,
            },
            TapeError::ColourOutside {
                colour: 1,
                span: s,
                total: 8,
            },
            TapeError::InputOutside { span: s, total: 8 },
            TapeError::TokenIdsShape {
                span: s,
                bucket_m: 1,
                vocab: 0,
            },
            TapeError::EmbedMissing { first: Some(0) },
            TapeError::EmbedMissing { first: None },
            TapeError::EmbedMisplaced { step: 3 },
            TapeError::EmbedInput { step: 0 },
            TapeError::EmbedOutput { step: 0 },
            TapeError::ResultOutside { span: s, total: 8 },
            TapeError::StepOverflow { step: 1 },
            TapeError::StepMisaligned { step: 1, offset: 4 },
            TapeError::StepOutside {
                step: 1,
                span: s,
                total: 8,
            },
            TapeError::WriteOverlapsRead {
                step: 1,
                write: s,
                read: s,
            },
            TapeError::WritesOverlap { step: 1 },
            TapeError::RotaryOutside { span: s, total: 8 },
            TapeError::RotaryShape {
                span: s,
                bucket_m: 1,
                rot_dim: 64,
            },
            TapeError::WeightIndex {
                step: 1,
                ix: 9,
                weights: 3,
            },
            TapeError::WeightShort {
                step: 1,
                ix: 2,
                need: 8,
                declared: 4,
            },
            TapeError::KvRegionOutside {
                region: Region::BlockTable,
                span: s,
                total: 8,
            },
            TapeError::KvRegionShape {
                region: Region::Sequence,
                span: s,
                want: 8,
            },
            TapeError::KvBlockSize { block_size: 16 },
            TapeError::KvLayer {
                step: 1,
                layer: 2,
                layers: 1,
            },
            TapeError::KvRegionMismatch {
                step: 1,
                region: Region::SlotMapping,
                span: s,
                tape: Some(s),
            },
            TapeError::KvRegionMismatch {
                step: 1,
                region: Region::SlotMapping,
                span: s,
                tape: None,
            },
            TapeError::KvStepGeometry { step: 1 },
            TapeError::OneRowStep { step: 1, rows: 2 },
            TapeError::RotaryMismatch {
                step: 1,
                table: ArenaOffset(256),
                tape: Some(ArenaOffset(512)),
            },
            TapeError::RotaryMismatch {
                step: 1,
                table: ArenaOffset(256),
                tape: None,
            },
            TapeError::WriteOverRegion {
                step: 1,
                write: s,
                region: Region::TokenIds,
            },
            TapeError::GdnRegionOutside {
                region: Region::GdnScratch,
                span: s,
                total: 8,
            },
            TapeError::GdnRegionShape {
                region: Region::GdnSequence,
                span: s,
                want: 16,
            },
            TapeError::GdnLayer {
                step: 1,
                layer: 2,
                layers: 1,
            },
            TapeError::GdnRegionMismatch {
                step: 1,
                region: Region::GdnScratch,
                span: s,
                tape: Some(s),
            },
            TapeError::GdnRegionMismatch {
                step: 1,
                region: Region::GdnScratch,
                span: s,
                tape: None,
            },
            TapeError::GdnStepGeometry { step: 1 },
            TapeError::GdnGeometry {
                num_k_heads: 16,
                num_v_heads: 16,
                head_k: 64,
                head_v: 128,
                conv_kernel: 4,
            },
            TapeError::MoeGeometry,
            TapeError::MoeRegionOutside {
                region: Region::MoeRows,
                span: s,
                total: 8,
            },
            TapeError::MoeRegionShape {
                region: Region::MoeGate,
                span: s,
                want: 4,
            },
            TapeError::MoeStepGeometry { step: 1 },
            TapeError::MoeRegionMismatch {
                step: 1,
                region: Region::MoePos,
                span: s,
                tape: s,
            },
            TapeError::MoeScratchTouched { step: 1, span: s },
        ];
        for e in &all {
            match e {
                TapeError::ColourCount { .. }
                | TapeError::ColourMisaligned { .. }
                | TapeError::ColourOutside { .. }
                | TapeError::InputOutside { .. }
                | TapeError::TokenIdsShape { .. }
                | TapeError::EmbedMissing { .. }
                | TapeError::EmbedMisplaced { .. }
                | TapeError::EmbedInput { .. }
                | TapeError::EmbedOutput { .. }
                | TapeError::ResultOutside { .. }
                | TapeError::StepOverflow { .. }
                | TapeError::StepMisaligned { .. }
                | TapeError::StepOutside { .. }
                | TapeError::WriteOverlapsRead { .. }
                | TapeError::WritesOverlap { .. }
                | TapeError::RotaryOutside { .. }
                | TapeError::RotaryShape { .. }
                | TapeError::WeightIndex { .. }
                | TapeError::WeightShort { .. }
                | TapeError::KvRegionOutside { .. }
                | TapeError::KvRegionShape { .. }
                | TapeError::KvBlockSize { .. }
                | TapeError::KvLayer { .. }
                | TapeError::KvRegionMismatch { .. }
                | TapeError::KvStepGeometry { .. }
                | TapeError::KvClasses
                | TapeError::OneRowStep { .. }
                | TapeError::RotaryMismatch { .. }
                | TapeError::WriteOverRegion { .. }
                | TapeError::GdnRegionOutside { .. }
                | TapeError::GdnRegionShape { .. }
                | TapeError::GdnLayer { .. }
                | TapeError::GdnRegionMismatch { .. }
                | TapeError::GdnStepGeometry { .. }
                | TapeError::GdnGeometry { .. }
                | TapeError::MoeGeometry
                | TapeError::MoeRegionOutside { .. }
                | TapeError::MoeRegionShape { .. }
                | TapeError::MoeStepGeometry { .. }
                | TapeError::MoeRegionMismatch { .. }
                | TapeError::MoeScratchTouched { .. } => {}
            }
        }
        all
    }

    /// No `TapeError` text has two consecutive spaces: a string literal split across lines without
    /// a `\` continuation keeps the next line's indentation (#9's three embed messages did).
    #[test]
    fn no_display_string_has_two_consecutive_spaces() {
        for e in every_tape_error() {
            let text = e.to_string();
            assert!(!text.contains("  "), "{e:?}: {text:?}");
        }
        for region in [
            Region::Rotary,
            Region::TokenIds,
            Region::BlockTable,
            Region::SlotMapping,
            Region::Sequence,
            Region::GdnSequence,
            Region::GdnScratch,
            Region::MoeLogits,
            Region::MoeTopKIdx,
            Region::MoeScores,
            Region::MoeChunkHist,
            Region::MoeCounts,
            Region::MoeOffsets,
            Region::MoeTileExpert,
            Region::MoePos,
            Region::MoeRows,
            Region::MoeGate,
            Region::MoeUp,
            Region::MoeAct,
            Region::MoeDown,
        ] {
            assert!(!region.to_string().contains("  "), "{region:?}");
        }
    }

    /// The embed writes exactly colour 0: an `out` at another colour's offset, or a `hidden` whose
    /// rows are not colour 0's bytes, is refused.
    #[test]
    fn the_embed_writes_exactly_colour_0() {
        let elsewhere = CudaStep::Embed(EmbedBf16 {
            out: ArenaOffset(262144),
            ..match embed_step(TOKENS, 256, VOCAB) {
                CudaStep::Embed(e) => e,
                _ => unreachable!("an embed"),
            }
        });
        let narrow = CudaStep::Embed(EmbedBf16 {
            hidden: 256,
            ..match embed_step(TOKENS, 256, VOCAB) {
                CudaStep::Embed(e) => e,
                _ => unreachable!("an embed"),
            }
        });
        for embed in [elsewhere, narrow] {
            assert_eq!(
                token_tape(&[embed]).validate(),
                Err(TapeError::EmbedOutput { step: 0 }),
                "{embed:?}"
            );
        }
    }

    /// `x [1, 512] → GEMV (N = 1024) → y [1, 1024]`, the one-row tape: colours 0 `x` at 0 and 1
    /// `y` at 1024 (256-aligned), weight 1 `W [1024, 512]`.
    const ONE_ROW: CudaTape = CudaTape {
        bucket_m: 1,
        arena: ArenaLayout {
            colour_bytes: &[1024, 2048],
            colour_offsets: &[0, 1024],
            total_bytes: 3072,
        },
        input: TapeInput::Hidden(span(0, 1024)),
        result: span(1024, 2048),
        weight_bytes: &[0, 1048576],
        steps: &[CudaStep::Gemv(GemmBf16 {
            a: ArenaOffset(0),
            b: WeightIx(1),
            c: ArenaOffset(1024),
            m: 1,
            n: 1024,
            k: 512,
        })],
        rotary: None,
        kv: None,
        gdn: None,
        moe: None,
        rotary_local: None,
    };

    /// A one-row GEMV tape validates; a GEMV of more than one row, or one in a bucket of more
    /// than one row, is refused: the kernel reads one row of `a`.
    #[test]
    fn a_gemv_is_one_row_of_a_one_row_bucket() {
        assert_eq!(ONE_ROW.validate(), Ok(()));
        let two = CudaTape {
            steps: Box::leak(Box::new([CudaStep::Gemv(GemmBf16 {
                a: ArenaOffset(0),
                b: WeightIx(1),
                c: ArenaOffset(1024),
                m: 2,
                n: 512,
                k: 256,
            })])),
            ..ONE_ROW
        };
        assert_eq!(
            two.validate(),
            Err(TapeError::OneRowStep { step: 0, rows: 2 })
        );
        let bucket = CudaTape {
            bucket_m: 2,
            ..ONE_ROW
        };
        assert_eq!(
            bucket.validate(),
            Err(TapeError::OneRowStep { step: 0, rows: 1 })
        );
    }

    /// A decode attention is one row of a one-row bucket, and reads the KV regions the prefill
    /// attention reads.
    #[test]
    fn a_decode_attention_is_one_row_of_a_one_row_bucket() {
        let decode = |rows, kv_heads| match attn(KV_TABLE, KV_SEQ, kv_heads) {
            CudaStep::AttnPrefill(a) => CudaStep::AttnDecode(AttnPrefillBf16 { rows, ..a }),
            _ => unreachable!("an attention"),
        };
        // `kv_tape` is a 256-row bucket: a one-row decode in it is refused for the bucket, a
        // 256-row one for its rows. (The whole one-row qwen3.5 tape validates: `cuda_tapes`.)
        assert_eq!(
            kv_tape(KV, &[decode(1, 2)]).validate(),
            Err(TapeError::OneRowStep { step: 0, rows: 1 })
        );
        assert_eq!(
            kv_tape(KV, &[decode(256, 2)]).validate(),
            Err(TapeError::OneRowStep { step: 0, rows: 256 })
        );
        let e = decode(1, 2).extents().expect("the decode's extents");
        assert_eq!(e.kv, Some((0, KvUse::Read)));
        assert_eq!(e.reads[1], Some(span(KV_TABLE, 512)), "the block table");
        assert_eq!(e.reads[2], Some(span(KV_SEQ, 8)), "the sequence");
        assert_eq!(e.writes, span(786432, 1024), "one row of 2 · 256 bf16");
    }

    // ── The MoE scratch's refusals ─────────────────────────────────

    /// A gathered MoE scratch (`bucket_m` 1, 4 experts, top-2, hidden 64, inter 32) after two
    /// colours, `x [1, 64]` at 0 and `y [1, 64]` at 256: logits 8 bytes at 512, the picks and
    /// scores 8 each at 768 and 1024, gate, up and act 128 each at 1280, 1536 and 1792, down 256
    /// at 2048.
    const MOE: MoeTape = MoeTape {
        experts: 4,
        top_k: 2,
        hidden: 64,
        inter: 32,
        tile_rows: CUDA_EXPERT_TILE_ROWS,
        max_tiles: 0,
        logits: span(512, 8),
        topk_idx: span(768, 8),
        scores: span(1024, 8),
        sort: None,
        gate: span(1280, 128),
        up: span(1536, 128),
        act: span(1792, 128),
        down: span(2048, 256),
        router_norm: None,
    };
    const MOE_TOTAL: u64 = 2304;

    /// The routing over [`MOE`]'s regions.
    const MOE_ROUTE: MoeRouteBf16 = MoeRouteBf16 {
        logits: ArenaOffset(512),
        topk_idx: ArenaOffset(768),
        scores: ArenaOffset(1024),
        m: 1,
        experts: 4,
        top_k: 2,
        renorm: true,
        expert_scale: None,
        program: RouteProgram::QwenSoftmax,
    };

    fn moe_tape(moe: Option<MoeTape>, steps: &[CudaStep]) -> CudaTape {
        CudaTape {
            bucket_m: 1,
            arena: ArenaLayout {
                colour_bytes: &[128, 128],
                colour_offsets: &[0, 256],
                total_bytes: MOE_TOTAL,
            },
            input: TapeInput::Hidden(span(0, 128)),
            result: span(256, 128),
            weight_bytes: &[],
            steps: Box::leak(steps.to_vec().into_boxed_slice()),
            rotary: None,
            kv: None,
            gdn: None,
            moe,
            rotary_local: None,
        }
    }

    #[test]
    fn a_moe_tape_validates() {
        let t = moe_tape(Some(MOE), &[CudaStep::MoeRoute(MOE_ROUTE)]);
        assert_eq!(t.validate(), Ok(()));
    }

    #[test]
    fn moe_geometry() {
        for moe in [
            MoeTape {
                tile_rows: 64,
                ..MOE
            },
            MoeTape { top_k: 5, ..MOE },
            MoeTape {
                max_tiles: 1,
                ..MOE
            },
        ] {
            assert_eq!(
                moe_tape(Some(moe), &[]).validate(),
                Err(TapeError::MoeGeometry),
                "{moe:?}"
            );
        }
    }

    #[test]
    fn moe_region_outside() {
        for (moe, region, at) in [
            (
                MoeTape {
                    logits: span(640, 8),
                    ..MOE
                },
                Region::MoeLogits,
                span(640, 8),
            ),
            (
                MoeTape {
                    down: span(2304, 256),
                    ..MOE
                },
                Region::MoeDown,
                span(2304, 256),
            ),
            // Before the previous region's end: the picks over the logits.
            (
                MoeTape {
                    topk_idx: span(512, 8),
                    ..MOE
                },
                Region::MoeTopKIdx,
                span(512, 8),
            ),
        ] {
            let t = moe_tape(Some(moe), &[]);
            assert_eq!(
                t.validate(),
                Err(TapeError::MoeRegionOutside {
                    region,
                    span: at,
                    total: MOE_TOTAL
                }),
                "{region}"
            );
        }
    }

    #[test]
    fn moe_region_shape() {
        let t = moe_tape(
            Some(MoeTape {
                scores: span(1024, 16),
                ..MOE
            }),
            &[],
        );
        assert_eq!(
            t.validate(),
            Err(TapeError::MoeRegionShape {
                region: Region::MoeScores,
                span: span(1024, 16),
                want: 8
            })
        );
    }

    #[test]
    fn moe_step_geometry() {
        let other_k = CudaStep::MoeRoute(MoeRouteBf16 {
            top_k: 3,
            ..MOE_ROUTE
        });
        assert_eq!(
            moe_tape(Some(MOE), &[other_k]).validate(),
            Err(TapeError::MoeStepGeometry { step: 0 })
        );
        assert_eq!(
            moe_tape(None, &[CudaStep::MoeRoute(MOE_ROUTE)]).validate(),
            Err(TapeError::MoeStepGeometry { step: 0 })
        );
    }

    #[test]
    fn moe_region_mismatch() {
        let elsewhere = CudaStep::MoeRoute(MoeRouteBf16 {
            logits: ArenaOffset(1280),
            ..MOE_ROUTE
        });
        assert_eq!(
            moe_tape(Some(MOE), &[elsewhere]).validate(),
            Err(TapeError::MoeRegionMismatch {
                step: 0,
                region: Region::MoeLogits,
                span: span(1280, 8),
                tape: MOE.logits
            })
        );
    }

    #[test]
    fn moe_scratch_touched() {
        let add = CudaStep::Add(AddBf16 {
            a: ArenaOffset(2048),
            b: ArenaOffset(0),
            out: ArenaOffset(256),
            rows: 1,
            cols: 64,
        });
        assert_eq!(
            moe_tape(Some(MOE), &[add]).validate(),
            Err(TapeError::MoeScratchTouched {
                step: 0,
                span: span(2048, 128)
            })
        );
    }

    // ── Gemma 4's KV classes, second rotary table and router norm (#885) ──

    /// [`KV`] with two classes: class 0 its own (2 × 256), class 1 one KV head of 512 lanes (the
    /// same pool row) with no window, and its one layer of `layer_class`.
    const fn two_classes(layer_class: &'static [u8]) -> KvTape {
        KvTape {
            classes: &[
                KvClass {
                    kv_heads: 2,
                    head_dim: 256,
                    window: None,
                },
                KvClass {
                    kv_heads: 1,
                    head_dim: 512,
                    window: None,
                },
            ],
            layer_class,
            ..KV
        }
    }

    /// Each class of the KV tape is a geometry the kernels take, and each layer names one: no
    /// class, a layer naming a class the tape lacks, a class count off the layers, a head dim off
    /// [`ATTN_BF16_HEAD_DIMS`], a window off [`ATTN_WINDOW_MULTIPLE`] or of 0, and a class whose
    /// row is wider than the pool's are each `KvClasses`.
    #[test]
    fn kv_classes() {
        assert_eq!(kv_tape(two_classes(&[1]), &[]).validate(), Ok(()));
        let class = |kv_heads, head_dim, window| KvClass {
            kv_heads,
            head_dim,
            window,
        };
        let one = |c: KvClass| -> &'static [KvClass] { Box::leak(Box::new([c])) };
        for (what, kv) in [
            ("no class", KvTape { classes: &[], ..KV }),
            ("a missing class", two_classes(&[2])),
            (
                "no layer's class",
                KvTape {
                    layer_class: &[],
                    ..KV
                },
            ),
            (
                "head dim 128",
                KvTape {
                    classes: one(class(4, 128, None)),
                    ..KV
                },
            ),
            (
                "a window of 1000",
                KvTape {
                    classes: one(class(2, 256, Some(1000))),
                    ..KV
                },
            ),
            (
                "a window of 0",
                KvTape {
                    classes: one(class(2, 256, Some(0))),
                    ..KV
                },
            ),
            (
                "a row wider than the pool's",
                KvTape {
                    classes: Box::leak(Box::new([class(2, 256, None), class(2, 512, None)])),
                    layer_class: &[0],
                    ..KV
                },
            ),
            (
                "no heads",
                KvTape {
                    classes: one(class(0, 256, None)),
                    ..KV
                },
            ),
        ] {
            assert_eq!(
                kv_tape(kv, &[]).validate(),
                Err(TapeError::KvClasses),
                "{what}"
            );
        }
    }

    /// A KV step's geometry is its layer's class: a class-1 layer's append and attention of one
    /// 512-lane head validate; the class-0 geometry on that layer, or a window its class lacks, is
    /// `KvStepGeometry`.
    #[test]
    fn a_kv_step_has_its_layers_class() {
        let append = |cols, head_dim| match rope_append(0, KV_SLOTS) {
            CudaStep::RopeAppend(r) => CudaStep::RopeAppend(RopeAppendBf16 {
                cols,
                head_dim,
                ..r
            }),
            _ => unreachable!(),
        };
        let attend = |kv_heads, head_dim, window| match attn(KV_TABLE, KV_SEQ, kv_heads) {
            CudaStep::AttnPrefill(a) => CudaStep::AttnPrefill(AttnPrefillBf16 {
                head_dim,
                window,
                q_heads: 2,
                ..a
            }),
            _ => unreachable!(),
        };
        let global = two_classes(&[1]);
        assert_eq!(
            kv_tape(global, &[append(512, 512), attend(1, 512, None)]).validate(),
            Ok(())
        );
        for (what, step) in [
            ("the class-0 append", append(512, 256)),
            ("the class-0 attention", attend(2, 256, None)),
            ("a window", attend(1, 512, Some(1024))),
        ] {
            assert_eq!(
                kv_tape(global, &[step]).validate(),
                Err(TapeError::KvStepGeometry { step: 0 }),
                "{what}"
            );
        }
    }

    /// The local rotary table: one beside the global table validates and a rope may read it; one
    /// without a global table is `KvClasses`; one over the global table, or outside the arena, is
    /// `RotaryOutside`.
    #[test]
    fn the_local_rotary_table() {
        let local = |offset| RotaryTable {
            span: span(offset, TABLE),
            rot_dim: 64,
            theta_bits: 1e4f64.to_bits(),
        };
        let at = TABLE_AT + TABLE;
        let grown = |t: CudaTape| CudaTape {
            arena: ArenaLayout {
                total_bytes: ROTARY_TOTAL + TABLE,
                ..t.arena
            },
            ..t
        };
        let two = CudaTape {
            rotary_local: Some(local(at)),
            steps: Box::leak(Box::new([rope(at)])),
            ..grown(with_rotary(TABLE_AT, TABLE))
        };
        assert_eq!(two.validate(), Ok(()));
        let alone = CudaTape {
            rotary: None,
            ..two
        };
        assert_eq!(alone.validate(), Err(TapeError::KvClasses));
        let over = CudaTape {
            rotary_local: Some(local(TABLE_AT)),
            steps: &[],
            ..two
        };
        assert_eq!(
            over.validate(),
            Err(TapeError::RotaryOutside {
                span: span(TABLE_AT, TABLE),
                total: ROTARY_TOTAL + TABLE
            })
        );
        let outside = CudaTape {
            rotary_local: Some(local(ROTARY_TOTAL + TABLE)),
            steps: &[],
            ..two
        };
        assert!(
            matches!(outside.validate(), Err(TapeError::RotaryOutside { .. })),
            "{:?}",
            outside.validate()
        );
    }

    /// [`MOE`] with Gemma's router-norm rows (`[1, hidden]` bf16, 128 bytes at 2304) after its down.
    const MOE_ROUTER_NORM: MoeTape = MoeTape {
        router_norm: Some(span(2304, 128)),
        ..MOE
    };

    /// The router norm into the scratch's router-norm rows validates; the rows sized off
    /// `[m, hidden]` are `MoeRegionShape`; a router norm writing elsewhere, or with no rows in
    /// the scratch, is `MoeRegionMismatch`, and on a tape with no scratch `MoeStepGeometry`.
    #[test]
    fn the_router_norm_writes_its_rows() {
        let norm = |y| {
            CudaStep::RouterNorm(RmsNormBf16 {
                x: ArenaOffset(0),
                gain: WeightIx(0),
                y: ArenaOffset(y),
                rows: 1,
                cols: 64,
                eps_bits: 1e-6f32.to_bits(),
                convention: GainConvention::Scale,
            })
        };
        let tape = |moe, steps: &[CudaStep]| CudaTape {
            arena: ArenaLayout {
                total_bytes: MOE_TOTAL + 256,
                ..moe_tape(None, &[]).arena
            },
            weight_bytes: &[128],
            ..moe_tape(moe, steps)
        };
        assert_eq!(
            tape(Some(MOE_ROUTER_NORM), &[norm(2304)]).validate(),
            Ok(())
        );
        let wide = MoeTape {
            router_norm: Some(span(2304, 256)),
            ..MOE
        };
        assert_eq!(
            tape(Some(wide), &[]).validate(),
            Err(TapeError::MoeRegionShape {
                region: Region::MoeRouterNorm,
                span: span(2304, 256),
                want: 128
            })
        );
        assert!(
            matches!(
                tape(Some(MOE_ROUTER_NORM), &[norm(512)]).validate(),
                Err(TapeError::MoeRegionMismatch {
                    step: 0,
                    region: Region::MoeRouterNorm,
                    ..
                })
            ),
            "{:?}",
            tape(Some(MOE_ROUTER_NORM), &[norm(512)]).validate()
        );
        assert!(
            matches!(
                tape(Some(MOE), &[norm(2304)]).validate(),
                Err(TapeError::MoeRegionMismatch {
                    region: Region::MoeRouterNorm,
                    ..
                }) | Err(TapeError::StepOutside { .. })
            ),
            "{:?}",
            tape(Some(MOE), &[norm(2304)]).validate()
        );
        assert_eq!(
            tape(None, &[norm(2304)]).validate(),
            Err(TapeError::MoeStepGeometry { step: 0 })
        );
    }
}

// SPDX-License-Identifier: Apache-2.0
//! Splash's DFlash2 cycle (~/git/splash @ 1482f6bb): the constants its kernels are built with that
//! no model states — its design's, not a checkpoint's — and the layouts of the buffers a cycle's
//! commands share, which the lowering binds and the worker fills and reads.
//!
//! The model's own geometry (heads, head dim, window, convolution group, selector rank, block) is
//! the drafter's `CanonicalParams`; these are the rest, each where Splash fixes it.

use crate::tape::ids::BucketM;

/// The splits a draft attention deals a lane's live ring tiles to (SPLASH_DRAFT_ATTENTION_SPLITS,
/// runtime/metal/abi/ExecutionGeometry.h): fixed rather than derived from the GPU so the combine
/// order, and with it the rounding, is the same on every machine and lane count.
pub const ATTENTION_SPLITS: u32 = 4;
/// The shards a proposal row's top-16 is found over (SPLASH_DRAFT_SAMPLING_SHARDS).
pub const SAMPLING_SHARDS: u32 = 8;
/// The candidates the selector keeps per proposal position (SPLASH_DRAFT_CANDIDATES).
pub const CANDIDATES: u32 = 16;
/// The uniforms one lane draws a step with (SPLASH_SAMPLING_UNIFORMS: the initial draw, a draw
/// per proposal, an acceptance test per proposal, the correction), the stride of their counters
/// ([`draft_uniform`]): `dflash2_cycle.h`.
pub const SAMPLING_UNIFORMS: u32 = 1 + 2 * PROPOSALS + 1;
/// The first of a step's draws for its proposals, its acceptance tests and its correction
/// (`DRAFT_UNIFORM_*`).
pub const UNIFORM_PROPOSALS: u32 = 1;
pub const UNIFORM_ACCEPTANCE: u32 = UNIFORM_PROPOSALS + PROPOSALS;
pub const UNIFORM_CORRECTION: u32 = UNIFORM_ACCEPTANCE + PROPOSALS;

/// A request's seed as a lane's kernels draw its uniforms from (`dflash2_cycle.h` `seed_lo`,
/// `seed_hi`; Splash's `SamplingConfig::seed`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DraftSeed([u32; 2]);

impl DraftSeed {
    pub fn of(seed: u64) -> Self {
        Self([seed as u32, (seed >> 32) as u32])
    }

    /// Its low and high words, as the lane structs carry them.
    pub fn words(self) -> [u32; 2] {
        self.0
    }

    pub fn get(self) -> u64 {
        u64::from(self.0[0]) | (u64::from(self.0[1]) << 32)
    }
}

/// Draw `draw` of a lane's step at verify position `position`: `dflash2_cycle.h` `draft_uniform`,
/// Splash's sampling uniform (runtime/model/Runtime.cpp:834-840 `nextUniform`) of the counter
/// `position · SAMPLING_UNIFORMS + draw`.
pub fn draft_uniform(seed: DraftSeed, position: u32, draw: u32) -> f32 {
    let counter = u64::from(position) * u64::from(SAMPLING_UNIFORMS) + u64::from(draw);
    let mut value = (seed.get()).wrapping_add(counter.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value >> 40) as f32 / (1u64 << 24) as f32
}
/// A block's proposals (SPLASH_DRAFT_PROPOSAL_TOKENS) — its rows less the anchor.
pub const PROPOSALS: u32 = 7;
/// A verify-rows threadgroup: the walk and the verify rows run one thread.
pub const EDGE_THREADS: u32 = 256;

/// Splash's acceptance prior (runtime/model/VerifyBudget.hpp `kPrior`, the fit of its measured
/// targets' logged cycles): a [`VerifyRows::Budget`]'s slope and intercepts before Splash refits
/// them online.
pub const ACCEPTANCE_SLOPE: f32 = 0.7;
pub const ACCEPTANCE_INTERCEPTS: [f32; PROPOSALS as usize] =
    [0.0, -0.35, -0.8, -0.75, -0.75, -0.85, -0.45];

/// The most lanes a cycle's verify budget prices (SPLASH_MAXIMUM_BATCH_WIDTH, its
/// `VerifyBudget::kLanes`).
pub const BUDGET_LANES: usize = 4;

/// What a cycle costs by the rows it verifies in all (Splash `VerifyBudget::seed`,
/// runtime/model/VerifyBudget.cpp:14-21, 82-94): for one to [`BUDGET_LANES`] lanes, its cost at 1,
/// 2, 4 and 8 rows a lane, every row costing 1, linear between them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VerifyCost(pub [[f32; 4]; BUDGET_LANES]);

impl VerifyCost {
    /// The rows a lane at which each lane count's costs are given.
    pub const KNOTS: [f32; 4] = [1.0, 2.0, 4.0, 8.0];

    /// A cycle of `lanes` lanes verifying `rows` rows in all (`VerifyBudget::seed`); one of more
    /// lanes than the budget prices costs what one of [`BUDGET_LANES`] does.
    pub fn of(&self, lanes: u32, rows: u32) -> f32 {
        let shape = self.0[(lanes.clamp(1, BUDGET_LANES as u32) - 1) as usize];
        let per_lane = rows as f32 / lanes.max(1) as f32;
        let upper = (1..3).find(|&k| per_lane <= Self::KNOTS[k]).unwrap_or(3);
        let t = (per_lane - Self::KNOTS[upper - 1]) / (Self::KNOTS[upper] - Self::KNOTS[upper - 1]);
        shape[upper - 1] + t * (shape[upper] - shape[upper - 1])
    }
}

/// Splash's seed costs (`VerifyBudget.cpp` `kSeed`: the mean shape of its measured targets'
/// cycles), what its budget prices rows by before it has timed a cycle — and a compiler's, which
/// times none, always.
pub const VERIFY_COST_SEED: VerifyCost = VerifyCost([
    [0.60, 0.69, 0.85, 1.0],
    [0.50, 0.58, 0.74, 1.0],
    [0.45, 0.52, 0.70, 1.0],
    [0.40, 0.47, 0.67, 1.0],
]);

/// How a cycle chooses the rows of its block its target verifies (Splash `VerifyRowsModel`,
/// runtime/ops/DraftSelector.hpp; `draft_verify_rows`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VerifyRows {
    /// Each lane verifies its maximum (Splash's `everyRow`): a dense target's, which reads its
    /// weights once for all of a cycle's rows.
    EveryRow,
    /// The rows that maximize the expected tokens over the cost of the rows verified in all, a
    /// proposal at `position` accepted, once those before it are, with probability p where
    /// `logit(p) = slope · logit(confidence) + intercepts[position]`.
    Budget {
        slope: f32,
        intercepts: [f32; PROPOSALS as usize],
        cost: VerifyCost,
    },
}

impl VerifyRows {
    /// Splash's `--verify-rows auto` (runtime/model/Runtime.cpp:3453 `budgetsVerifyRows`): the
    /// verify budget's rows on a target that routes experts — each row it verifies routes its own,
    /// which a rejected row reads for nothing — every row on a dense one. The budget's is Splash's
    /// before its first refit and its first timed cycle (`kPrior`, `kSeed`), which it would learn
    /// at run time: the model a compiler bakes.
    pub fn of_target(routes_experts: bool) -> Self {
        match routes_experts {
            true => Self::Budget {
                slope: ACCEPTANCE_SLOPE,
                intercepts: ACCEPTANCE_INTERCEPTS,
                cost: VERIFY_COST_SEED,
            },
            false => Self::EveryRow,
        }
    }
}

/// One lane's policy for a cycle (`dflash2_cycle.h` `DraftLaneParams`), the request's seed and the
/// step's verify position its proposals draw their uniforms from ([`draft_uniform`]), and the
/// cycle's cost line ([`VerifyLine`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DraftLaneParams {
    pub temperature: f32,
    pub sampling: u32,
    pub maximum: u32,
    pub remaining: u32,
    pub seed: DraftSeed,
    pub position: u32,
    pub cost: VerifyLine,
}

/// A cycle's cost in the rows its target verifies in all: `fixed + per_row · rows`, the weight
/// bytes (GiB) its forwards — its target's verify and its drafter's passes — read whatever their
/// rows and each row's experts ([`crate::tape::step::WeightBytes`]). Splash's verify budget prices
/// rows by the line it fits to its timed cycles (runtime/model/VerifyBudget.cpp `line`, `costs`);
/// a compiler times none, and counts the bytes. No bytes a row (the default): the seed.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VerifyLine {
    pub fixed: f32,
    pub per_row: f32,
}

impl VerifyLine {
    /// The line of a cycle of `target`'s verify and `drafter`'s passes.
    pub fn of(
        target: crate::tape::step::WeightBytes,
        drafter: crate::tape::step::WeightBytes,
    ) -> Self {
        let gib = |bytes: u64| bytes as f32 / (1u64 << 30) as f32;
        Self {
            fixed: gib(target.fixed + drafter.fixed),
            per_row: gib(target.per_row + drafter.per_row),
        }
    }
}

/// The shards a sampled verify row's softmax denominator is summed over
/// (SPLASH_TARGET_SAMPLING_SHARDS): `dflash2_target_sampling.metal`'s `kShards`.
pub const TARGET_SAMPLING_SHARDS: u32 = 16;
/// The threads of a sampled row's search and of each of its draw's groups
/// (SPLASH_TARGET_VOCABULARY_THREADS, SPLASH_TARGET_VOCABULARY_GROUPS).
pub const VOCABULARY_THREADS: u32 = 1024;
pub const VOCABULARY_GROUPS: u32 = 8;
/// The ranges of the vocabulary a row's draw sums (SPLASH_TARGET_VOCABULARY_RANGES).
pub const VOCABULARY_RANGES: u32 = VOCABULARY_GROUPS * (VOCABULARY_THREADS / 32);

/// One lane of a step's verify (`dflash2_cycle.h` `VerifyLane`), as Splash's `AcceptBatchParams`
/// and `TargetSamplingParams` carry it: its first verify row among the step's sampled rows, the
/// drafts its verify carried (0 for a lane that decoded without), that row's position, the tokens
/// it may still emit (at least one) and its stop tokens; whether it samples, its distribution's
/// cuts — top-k (0: every token), temperature, top-p and min-p — and whether it ignores
/// end-of-sequence; and the request's seed, its acceptance's and correction's uniforms drawn from
/// it and `position` ([`draft_uniform`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VerifyLane {
    pub sampled: u32,
    pub drafts: u32,
    pub position: u32,
    pub remaining: u32,
    pub stop: [u32; 2],
    pub sampling: u32,
    pub top_k: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub exclude_stop: u32,
    pub seed: DraftSeed,
}

/// Splash's acceptance of a cycle's verify, and its drafter's next block's inputs, picked on the
/// device after the target's sampler (`dflash2_accept.metal`): each lane's retained rows — what
/// the drafter's context commit writes — and its next block, the last retained token and the mask
/// from the position after them. Baked per drafter with its block and mask token
/// ([`crate::off_tape`]).
pub struct DraftAcceptKernel {
    pub pipeline: crate::off_tape::OffTapePipeline,
}

/// Splash's target policy over a cycle's sampled verify rows (`dflash2_target_sampling.metal`):
/// each row's softmax mass, where its distribution ends, and its draw — a drafted row's correction
/// with its draft token's target probability, the last row's bonus. Baked per target with its
/// logits and its drafter's block ([`crate::off_tape`]).
pub struct TargetSamplingKernels {
    pub mass: crate::off_tape::OffTapePipeline,
    pub search: crate::off_tape::OffTapePipeline,
    pub draw: crate::off_tape::OffTapePipeline,
}

/// Where one step's cycle reads and writes, in one buffer for `lanes` lanes of `rows`-row blocks,
/// each section 256-aligned. The host writes each lane's [`VerifyLane`], the drafts its verify
/// carried (`[lanes][rows - 1]` u32), their candidates and candidate probabilities
/// (`[lanes][rows - 1][CANDIDATES]`, u32 and f32: the drafter's selection). The device writes the
/// rows each lane retained and the drafts it accepted (`[lanes]` u32), its next block's input ids
/// and positions (`[lanes][rows]` u32), and the target policy's scratch: each sampled row's
/// `TargetVocabularyRow` (16 bytes), shard masses (12 bytes a shard), draw ranges (8 bytes a
/// range) and arrival counter (`u32`, zero between dispatches). A cycle's uniforms are its
/// kernels' own ([`draft_uniform`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CycleLayout {
    pub lanes: u64,
    pub drafted: u64,
    pub draft_ids: u64,
    pub draft_probs: u64,
    pub retained: u64,
    pub accepted: u64,
    pub block_ids: u64,
    pub block_positions: u64,
    pub vocabulary_rows: u64,
    pub partial_masses: u64,
    pub ranges: u64,
    pub arrivals: u64,
    pub total: u64,
}

impl CycleLayout {
    pub fn of(lanes: u64, rows: u64) -> Self {
        let proposals = rows - 1;
        let candidates = lanes * proposals * u64::from(CANDIDATES);
        let verify_rows = lanes * rows;
        let mut at = 0;
        let mut section = |bytes: u64| {
            let start = at;
            at = align(at + bytes);
            start
        };
        let lane_bytes = std::mem::size_of::<VerifyLane>() as u64;
        Self {
            lanes: section(lanes * lane_bytes),
            drafted: section(lanes * proposals * 4),
            draft_ids: section(candidates * 4),
            draft_probs: section(candidates * 4),
            retained: section(lanes * 4),
            accepted: section(lanes * 4),
            block_ids: section(lanes * rows * 4),
            block_positions: section(lanes * rows * 4),
            vocabulary_rows: section(verify_rows * 16),
            partial_masses: section(verify_rows * u64::from(TARGET_SAMPLING_SHARDS) * 12),
            ranges: section(verify_rows * u64::from(VOCABULARY_RANGES) * 8),
            arrivals: section(verify_rows * 4),
            total: at,
        }
    }

    /// The acceptance's bindings in a buffer at `base`, the target's tokens (one a sampled row) at
    /// `tokens`.
    pub fn accept(self, base: u64, tokens: u64) -> [u64; 10] {
        [
            base + self.lanes,
            base + self.drafted,
            tokens,
            base + self.retained,
            base + self.accepted,
            base + self.block_ids,
            base + self.block_positions,
            base + self.draft_ids,
            base + self.draft_probs,
            base + self.vocabulary_rows,
        ]
    }

    /// The target policy's bindings — its mass, search and draw — over the logits at `logits`,
    /// its tokens at `tokens`.
    pub fn sampling(self, base: u64, logits: u64, tokens: u64) -> [Vec<u64>; 3] {
        let at = |section: u64| base + section;
        [
            vec![logits, at(self.lanes), at(self.partial_masses)],
            vec![
                logits,
                at(self.lanes),
                at(self.partial_masses),
                at(self.vocabulary_rows),
            ],
            vec![
                logits,
                at(self.lanes),
                at(self.vocabulary_rows),
                at(self.drafted),
                at(self.draft_ids),
                at(self.draft_probs),
                tokens,
                at(self.ranges),
                at(self.arrivals),
            ],
        ]
    }
}

/// Encode `pipeline` over `groups` threadgroups of `threads` onto `encoder` behind a barrier on
/// everything before it, its buffers bound through `arg_table`.
fn encode_after(
    pipeline: &crate::off_tape::OffTapePipeline,
    encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    arg_table: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
    groups: u32,
    threads: u32,
) {
    use objc2_metal::{
        MTL4CommandEncoder as _, MTL4ComputeCommandEncoder as _, MTL4VisibilityOptions, MTLSize,
        MTLStages,
    };
    encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
        MTLStages::Dispatch,
        MTLStages::Dispatch,
        MTL4VisibilityOptions::Device,
    );
    encoder.setComputePipelineState(pipeline);
    encoder.setArgumentTable(Some(arg_table));
    let size = |width: u32| MTLSize {
        width: width as usize,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(threads));
}

impl DraftAcceptKernel {
    /// `None` for a model that is no DFlash2 drafter.
    pub fn new(
        device: &crate::shader_cache::Device,
        kernels: &crate::off_tape::OffTapeKernels,
    ) -> Result<Option<Self>, crate::stream::MetalStreamError> {
        (kernels.dflash2_accept.as_ref())
            .map(|k| {
                Ok(Self {
                    pipeline: crate::off_tape::OffTapePipeline::new(device, k)?,
                })
            })
            .transpose()
    }

    /// Encode the acceptance of `lanes` lanes onto `encoder` ([`CycleLayout::accept`]).
    pub fn encode(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        arg_table: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        lanes: u32,
    ) {
        encode_after(&self.pipeline, encoder, arg_table, lanes, 1);
    }
}

impl TargetSamplingKernels {
    /// `None` for a model no DFlash2 drafter drafts for.
    pub fn new(
        device: &crate::shader_cache::Device,
        kernels: &crate::off_tape::OffTapeKernels,
    ) -> Result<Option<Self>, crate::stream::MetalStreamError> {
        (kernels.dflash2_sampling.as_ref())
            .map(|k| {
                Ok(Self {
                    mass: crate::off_tape::OffTapePipeline::new(device, &k.mass)?,
                    search: crate::off_tape::OffTapePipeline::new(device, &k.search)?,
                    draw: crate::off_tape::OffTapePipeline::new(device, &k.draw)?,
                })
            })
            .transpose()
    }

    /// Encode the policy over `rows` verify rows (`lanes · ROWS`) onto `encoder`, one argument
    /// table a stage ([`CycleLayout::sampling`]); a lane that does not sample returns at once.
    pub fn encode(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        tables: [&objc2::runtime::ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>; 3],
        rows: u32,
    ) {
        let [mass, search, draw] = tables;
        encode_after(
            &self.mass,
            encoder,
            mass,
            rows * TARGET_SAMPLING_SHARDS,
            256,
        );
        encode_after(&self.search, encoder, search, rows, VOCABULARY_THREADS);
        encode_after(
            &self.draw,
            encoder,
            draw,
            rows * VOCABULARY_GROUPS,
            VOCABULARY_THREADS,
        );
    }
}

/// The draft geometry the kernels index by: the block's rows and the attention's widths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftGeometry {
    pub rows: u32,
    pub query_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
}

impl DraftGeometry {
    pub fn attention_width(self) -> u64 {
        u64::from(self.query_heads * self.head_dim)
    }
    /// The grouped query rows of one KV head: its query heads' rows of the block.
    pub fn attention_rows(self) -> u64 {
        u64::from(self.query_heads / self.kv_heads * self.rows)
    }
    /// The lanes of a bucket of `m` rows.
    pub fn lanes(self, m: BucketM) -> u64 {
        u64::from(m.0 / self.rows)
    }
}

/// `n` rounded up to the 256-byte alignment a buffer offset needs.
fn align(n: u64) -> u64 {
    n.next_multiple_of(256)
}

/// Where a draft attention's intermediates sit in the op scratch at a bucket: its grouped queries,
/// its block keys and values, and its split partials (Splash's `DraftAttentionPlan::workspace`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionScratch {
    pub queries: u64,
    pub query_keys: u64,
    pub query_values: u64,
    pub partials: u64,
    pub total: u64,
}

impl AttentionScratch {
    pub fn at(g: DraftGeometry, m: BucketM, act_bytes: u64) -> Self {
        let lanes = g.lanes(m);
        let rows = lanes * u64::from(g.rows);
        let kv = rows * u64::from(g.kv_heads * g.head_dim) * act_bytes;
        let partial = g.attention_rows() * (u64::from(g.head_dim) + 2) * 4;
        let queries = 0;
        let query_keys = align(queries + rows * g.attention_width() * act_bytes);
        let query_values = align(query_keys + kv);
        let partials = align(query_values + kv);
        let total =
            align(partials + lanes * u64::from(g.kv_heads) * u64::from(ATTENTION_SPLITS) * partial);
        Self {
            queries,
            query_keys,
            query_values,
            partials,
            total,
        }
    }
}

/// Where a selection's shard partials sit in the op scratch at a bucket: their ids, then their
/// values followed by each position's 16 x 16 edge table (Splash's `DraftSelector::workspace`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectScratch {
    pub partial_ids: u64,
    pub partial_values: u64,
    pub total: u64,
}

impl SelectScratch {
    pub fn at(g: DraftGeometry, m: BucketM) -> Self {
        let candidates = g.lanes(m) * u64::from(PROPOSALS * CANDIDATES);
        let partial_ids = 0;
        let partial_values = align(candidates * u64::from(SAMPLING_SHARDS) * 4);
        let total =
            align(partial_values + candidates * u64::from(SAMPLING_SHARDS + CANDIDATES) * 4);
        Self {
            partial_ids,
            partial_values,
            total,
        }
    }
}

/// A region of a drafter's outputs (`RuntimeBindingKind::DraftOutputs`): what a selection writes
/// and the target's acceptance and the host read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum DraftOutput {
    /// `[lanes, proposals, candidates]` u32: each position's candidates, best first.
    Candidates,
    /// `[lanes, proposals, candidates]` f32: their unary scores.
    Unary,
    /// `[lanes, proposals]` u32: the proposals.
    Tokens,
    /// `[lanes, proposals, candidates]` f32: a sampling lane's candidate probabilities.
    QProbs,
    /// `[lanes, proposals]` f32: each position's confidence.
    Confidence,
    /// `[lanes]` u32: the rows of its block each lane's target verifies.
    VerifyRows,
}

impl DraftOutput {
    pub const ALL: [Self; 6] = [
        Self::Candidates,
        Self::Unary,
        Self::Tokens,
        Self::QProbs,
        Self::Confidence,
        Self::VerifyRows,
    ];

    /// The region's bytes for `lanes` lanes.
    pub fn bytes(self, lanes: u64) -> u64 {
        let per_lane = match self {
            Self::Candidates | Self::Unary | Self::QProbs => PROPOSALS * CANDIDATES,
            Self::Tokens | Self::Confidence => PROPOSALS,
            Self::VerifyRows => 1,
        };
        lanes * u64::from(per_lane) * 4
    }

    /// Where the region starts in a drafter's outputs sized for `lanes` lanes.
    pub fn offset(self, lanes: u64) -> u64 {
        (Self::ALL.iter())
            .take_while(|&&r| r != self)
            .map(|r| align(r.bytes(lanes)))
            .sum()
    }

    /// A drafter's outputs' bytes for `lanes` lanes.
    pub fn total(lanes: u64) -> u64 {
        Self::ALL.iter().map(|r| align(r.bytes(lanes))).sum()
    }
}

/// A DFlash2 drafter's rings (Splash `DFlashDraftRing`, runtime/model/DFlashDraft.hpp): every
/// layer's keys `[slot][kv head][window][head dim]` and values `[slot][kv head][head dim][window]`
/// in the model dtype, one slot a concurrently-resident sequence (the target's state slot), kept
/// across forwards. Slot assignment is the target's state-slot allocator's; this owns the storage.
pub struct DraftRingPool {
    keys: Vec<crate::PoolMem>,
    values: Vec<crate::PoolMem>,
    /// Every slot's context window (`dflash2.metal` `DRAFT_WINDOW_BYTES` a slot): what a
    /// prefix-cache hit rebuilds the slot's rings from (Splash `DFlashContextWindow`).
    windows: crate::PoolMem,
    pub num_slots: usize,
}

impl DraftRingPool {
    /// One slot's ring of one layer, in bytes: `kv_heads · window · head_dim` elements of
    /// `elem_bytes` each, for K or V alike.
    pub fn slot_bytes(kv_heads: usize, window: usize, head_dim: usize, elem_bytes: usize) -> usize {
        kv_heads * window * head_dim * elem_bytes
    }

    /// One slot's context window, in bytes ([`WindowLayout::bytes`]).
    pub fn window_bytes(window: usize, hidden: usize) -> usize {
        WindowLayout {
            window: window as u32,
            hidden,
        }
        .bytes()
    }

    /// Allocate `num_layers` layers' keys and values of `num_slots` slots, each `slot_bytes`, and
    /// their context windows, each `window_bytes`, by `alloc_buffer` (the device's pool
    /// allocator, which pins what it hands out).
    pub fn new(
        num_layers: usize,
        num_slots: usize,
        slot_bytes: usize,
        window_bytes: usize,
        mut alloc_buffer: impl FnMut(usize) -> anyhow::Result<crate::PoolMem>,
    ) -> anyhow::Result<Self> {
        let bytes = num_slots * slot_bytes;
        let mut alloc = || -> anyhow::Result<Vec<crate::PoolMem>> {
            (0..num_layers).map(|_| alloc_buffer(bytes)).collect()
        };
        let keys = alloc()?;
        let values = alloc()?;
        let windows = alloc_buffer(num_slots * window_bytes)?;
        Ok(Self {
            keys,
            values,
            windows,
            num_slots,
        })
    }

    /// Every slot's context window.
    pub fn windows(&self) -> &crate::PoolMem {
        &self.windows
    }

    /// Layer `layer`'s keys and values.
    pub fn layer(&self, layer: usize) -> (&crate::PoolMem, &crate::PoolMem) {
        (&self.keys[layer], &self.values[layer])
    }

    pub fn num_layers(&self) -> usize {
        self.keys.len()
    }
}

/// The values a context window's 4-bit codes share a scale and minimum over
/// (`dflash2.metal` `DRAFT_CONTEXT_GROUP`).
pub const CONTEXT_GROUP: usize = 64;

/// A slot's context window as `dflash2.metal` lays it out (`DRAFT_WINDOW_BYTES`; Splash
/// abi/DraftAttention.h:88-100 `draft_context_window_bytes`): `window` ring positions' context
/// rows of `hidden` values, every position's 4-bit codes (`hidden / 2` bytes) first, then every
/// position's groups' fp16 scale and minimum (4 bytes a group of [`CONTEXT_GROUP`] values). A
/// position's row sits at the position modulo `window`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowLayout {
    pub window: u32,
    pub hidden: usize,
}

impl WindowLayout {
    fn code_bytes(self) -> usize {
        self.hidden / 2
    }

    fn group_bytes(self) -> usize {
        self.hidden / CONTEXT_GROUP * 4
    }

    /// A slot's window, in bytes.
    pub fn bytes(self) -> usize {
        self.window as usize * (self.code_bytes() + self.group_bytes())
    }

    /// The byte runs `(offset, len)` of a slot's window holding positions `[from, to)`, at most a
    /// window of them: in each plane, the ring's run of them, cut where it wraps.
    pub fn runs(self, from: u32, to: u32) -> Vec<(usize, usize)> {
        assert!(
            from <= to && to - from <= self.window,
            "positions {from}..{to} are more than a window of {}",
            self.window
        );
        let start = from % self.window;
        let rows = to - from;
        let head = rows.min(self.window - start);
        let slots = [(start, head), (0, rows - head)];
        let codes = self.window as usize * self.code_bytes();
        (slots.into_iter())
            .filter(|&(_, n)| n > 0)
            .flat_map(|(slot, n)| {
                let (slot, n) = (slot as usize, n as usize);
                [
                    (slot * self.code_bytes(), n * self.code_bytes()),
                    (codes + slot * self.group_bytes(), n * self.group_bytes()),
                ]
            })
            .collect()
    }
}

/// One copy between context windows: `len` bytes of `from` at `from_at` to `to` at `to_at`.
pub struct WindowCopy<'a> {
    pub from: &'a objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    pub from_at: usize,
    pub to: &'a objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    pub to_at: usize,
    pub len: usize,
}

/// Encode `copies` on `encoder` between barriers: after every dispatch and copy before them, before
/// every one after. A prefix-cache snapshot takes its context window this way, and a restore gives
/// it back (Splash `QwenStateStorage::copyWindow`, runtime/model/QwenState.cpp:452-454).
pub fn encode_window_copies(
    encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    copies: &[WindowCopy<'_>],
) {
    use objc2_metal::{
        MTL4CommandEncoder as _, MTL4ComputeCommandEncoder as _, MTL4VisibilityOptions,
        MTLBuffer as _, MTLStages,
    };
    if copies.is_empty() {
        return;
    }
    let work = MTLStages::Dispatch | MTLStages::Blit;
    encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
        work,
        MTLStages::Blit,
        MTL4VisibilityOptions::Device,
    );
    for c in copies {
        assert!(
            c.from_at + c.len <= c.from.length() && c.to_at + c.len <= c.to.length(),
            "a window copy of {} bytes lies inside both buffers",
            c.len
        );
        // SAFETY: both ranges lie inside their buffers (checked above).
        unsafe {
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                c.from, c.from_at, c.to, c.to_at, c.len,
            );
        }
    }
    encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
        MTLStages::Blit,
        work,
        MTL4VisibilityOptions::Device,
    );
}

#[cfg(test)]
mod window_tests {
    use super::*;

    /// Where `dflash2.metal`'s `draft_window_store_group` writes a position's codes and groups.
    fn kernel_bytes(layout: WindowLayout, position: u32) -> Vec<usize> {
        let slot = (position % layout.window) as usize;
        let (h, groups) = (layout.hidden, layout.hidden / CONTEXT_GROUP);
        let codes = layout.window as usize * (h / 2);
        let code = (0..h / 2).map(|b| slot * (h / 2) + b);
        let group = (0..groups * 4).map(|b| codes + slot * groups * 4 + b);
        code.chain(group).collect()
    }

    /// A run of positions' byte runs are exactly the bytes the kernel writes for them, wrapping
    /// or not, and a whole window's are the whole slot.
    #[test]
    fn a_runs_bytes_are_the_kernels_for_its_positions() {
        let layout = WindowLayout {
            window: 16,
            hidden: 128,
        };
        for (from, to) in [(0, 16), (3, 9), (12, 20), (40, 56), (47, 48), (5, 5)] {
            let mut ours: Vec<usize> = (layout.runs(from, to).into_iter())
                .flat_map(|(at, len)| at..at + len)
                .collect();
            let mut theirs: Vec<usize> = (from..to).flat_map(|p| kernel_bytes(layout, p)).collect();
            ours.sort_unstable();
            theirs.sort_unstable();
            assert_eq!(ours, theirs, "positions {from}..{to}");
        }
        let whole: usize = layout.runs(16, 32).iter().map(|&(_, len)| len).sum();
        assert_eq!(whole, layout.bytes());
        assert_eq!(layout.bytes(), DraftRingPool::window_bytes(16, 128));
    }
}

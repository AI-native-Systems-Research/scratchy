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

/// One lane's policy for a cycle (`dflash2_cycle.h` `DraftLaneParams`), and the request's seed
/// and the step's verify position its proposals draw their uniforms from ([`draft_uniform`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DraftLaneParams {
    pub temperature: f32,
    pub sampling: u32,
    pub maximum: u32,
    pub remaining: u32,
    pub seed: DraftSeed,
    pub position: u32,
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
    /// Each penalized sequence's sampled rows' penalties, in place ([`PenalizeEntry`]).
    pub penalize: crate::off_tape::OffTapePipeline,
    vocab: u32,
}

/// The most rows a [`PenalizeEntry`] covers: a block's (`dflash2_cycle.h`).
pub const PENALIZE_ROWS: usize = 8;

/// One penalized sequence of a step (`dflash2_cycle.h` `PenalizeEntry`; an entry of Splash's
/// `SamplingPenaltyParams`): its first row in the step's logits and how many, its penalty words'
/// row, its penalties, and the step's input tokens of those rows — row r's context adds inputs
/// 1..r, its drafts.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PenalizeEntry {
    pub logits_row: u32,
    pub rows: u32,
    pub table_row: u32,
    pub repetition: f32,
    pub repetition_inverse: f32,
    pub presence: f32,
    pub frequency: f32,
    pub inputs: [u32; PENALIZE_ROWS],
}

const _: () = assert!(
    size_of::<PenalizeEntry>() == 60,
    "dflash2_cycle.h's PenalizeEntry"
);

impl PenalizeEntry {
    /// The entry of a sequence's `inputs` rows from logits row `logits_row`, its words at row
    /// `table_row`: its repetition's inverse saturated, as Splash's (`1 / 2^-149` overflows).
    pub fn new(
        logits_row: u32,
        inputs: &[u32],
        table_row: u32,
        [repetition, presence, frequency]: [f32; 3],
    ) -> Self {
        assert!(
            (1..=PENALIZE_ROWS).contains(&inputs.len()),
            "a penalized sequence has 1 to {PENALIZE_ROWS} rows, not {}",
            inputs.len()
        );
        let mut rows = [0; PENALIZE_ROWS];
        rows[..inputs.len()].copy_from_slice(inputs);
        Self {
            logits_row,
            rows: inputs.len() as u32,
            table_row,
            repetition,
            repetition_inverse: (1.0 / repetition).min(f32::MAX),
            presence,
            frequency,
            inputs: rows,
        }
    }
}

/// A sequence's penalty words (Splash `Sampling::rebuildPenaltyWords`, runtime/ops/Sampling.cpp,
/// a word a token) from its history, counted as the target's own sampler counts them
/// (sampling.metal `sample_softmax_reduce`): each token's occurrences in its prompt and its output,
/// the one count its repetition, frequency and presence penalties all read.
pub fn rebuild_penalty_words(words: &mut [u32], prompt: &[u32], generated: &[u32]) {
    words.fill(0);
    count_penalty_tokens(words, prompt);
    count_penalty_tokens(words, generated);
}

/// Splash `Sampling::countPenaltyTokens`: `selected`, emitted, counted into the words.
pub fn count_penalty_tokens(words: &mut [u32], selected: &[u32]) {
    for &token in selected {
        words[token as usize] += 1;
    }
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
                    penalize: crate::off_tape::OffTapePipeline::new(device, &k.penalize)?,
                    vocab: kernels.vocab.get(),
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

    /// Encode `entries` penalized sequences' penalties onto `encoder` (`table`: the logits, the
    /// penalty words and the entries), after what is on it: 256 tokens a threadgroup.
    pub fn encode_penalize(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        entries: u32,
    ) {
        encode_after(
            &self.penalize,
            encoder,
            table,
            entries * self.vocab.div_ceil(256),
            256,
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

/// The values a context window's group quantizes together (`dflash2.metal` `DRAFT_CONTEXT_GROUP`,
/// Splash prefill/draft_context.metal).
pub const WINDOW_GROUP: u32 = 64;

/// A slot's context window's bytes (`dflash2.metal` `DRAFT_WINDOW_*`; Splash
/// abi/DraftAttention.h:88-100): its `window` positions' 4-bit codes, `hidden / 2` bytes a
/// position, then each position's groups' fp16 scale and minimum, 4 bytes a group of
/// [`WINDOW_GROUP`] values. A position's codes and groups sit at its ring slot,
/// `position % window`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowLayout {
    window: u32,
    hidden: u32,
}

impl WindowLayout {
    /// The window of `window` positions of rows `hidden` values wide: `None` unless `hidden` is a
    /// whole number of groups, as the kernels quantize it.
    pub fn new(window: u32, hidden: u32) -> Option<Self> {
        (window > 0 && hidden > 0 && hidden.is_multiple_of(WINDOW_GROUP))
            .then_some(Self { window, hidden })
    }

    pub fn window(self) -> u32 {
        self.window
    }

    fn codes(self) -> u64 {
        u64::from(self.window) * self.code_bytes()
    }

    fn code_bytes(self) -> u64 {
        u64::from(self.hidden / 2)
    }

    fn group_bytes(self) -> u64 {
        u64::from(self.hidden / WINDOW_GROUP) * 4
    }

    /// A slot's whole window.
    pub fn bytes(self) -> u64 {
        self.codes() + u64::from(self.window) * self.group_bytes()
    }

    /// The byte ranges within a slot's window that hold positions `[from, to)`, at most a window
    /// of them: their codes and their groups, a run of each on either side of the ring's wrap.
    pub fn positions(self, from: u32, to: u32) -> Vec<std::ops::Range<u64>> {
        debug_assert!(
            from <= to && to - from <= self.window,
            "{from}..{to} in {self:?}"
        );
        let (start, len) = (from % self.window, to - from);
        let first = len.min(self.window - start);
        let slot_runs = [(start, first), (0, len - first)];
        let mut out = Vec::new();
        for (bytes, at) in [(self.code_bytes(), 0), (self.group_bytes(), self.codes())] {
            for &(slot, n) in slot_runs.iter().filter(|&&(_, n)| n > 0) {
                let begin = at + u64::from(slot) * bytes;
                out.push(begin..begin + u64::from(n) * bytes);
            }
        }
        out
    }
}

/// A copy of `len` bytes from `from` at `from_at` to `to` at `to_at`: a lane's context window and
/// a prefix-cache snapshot's, either way, or a penalized row's raw logits set aside and back.
pub struct BufferCopy<'a> {
    pub from: &'a objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    pub from_at: u64,
    pub to: &'a objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    pub to_at: u64,
    pub len: u64,
}

/// Encode `copies` on `encoder` between what comes before and after them: each copy waits for
/// every dispatch and copy before it, and every dispatch and copy after them waits for the copies.
pub fn encode_buffer_copies(
    encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    copies: &[BufferCopy<'_>],
) {
    use objc2_metal::{
        MTL4CommandEncoder as _, MTL4ComputeCommandEncoder as _, MTL4VisibilityOptions, MTLStages,
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
        // SAFETY: every range lies inside its buffer, as its caller laid it out.
        unsafe {
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                c.from,
                c.from_at as usize,
                c.to,
                c.to_at as usize,
                c.len as usize,
            );
        }
    }
    encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
        MTLStages::Blit,
        work,
        MTL4VisibilityOptions::Device,
    );
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
    window: WindowLayout,
    pub num_slots: usize,
}

impl DraftRingPool {
    /// One slot's ring of one layer, in bytes: `kv_heads · window · head_dim` elements of
    /// `elem_bytes` each, for K or V alike.
    pub fn slot_bytes(kv_heads: usize, window: usize, head_dim: usize, elem_bytes: usize) -> usize {
        kv_heads * window * head_dim * elem_bytes
    }

    /// Allocate `num_layers` layers' keys and values of `num_slots` slots, each `slot_bytes`, and
    /// their context windows, each laid out as `window`, by `alloc_buffer` (the device's pool
    /// allocator, which pins what it hands out).
    pub fn new(
        num_layers: usize,
        num_slots: usize,
        slot_bytes: usize,
        window: WindowLayout,
        mut alloc_buffer: impl FnMut(usize) -> anyhow::Result<crate::PoolMem>,
    ) -> anyhow::Result<Self> {
        let bytes = num_slots * slot_bytes;
        let mut alloc = || -> anyhow::Result<Vec<crate::PoolMem>> {
            (0..num_layers).map(|_| alloc_buffer(bytes)).collect()
        };
        let keys = alloc()?;
        let values = alloc()?;
        let windows = alloc_buffer(num_slots * window.bytes() as usize)?;
        Ok(Self {
            keys,
            values,
            windows,
            window,
            num_slots,
        })
    }

    /// Every slot's context window.
    pub fn windows(&self) -> &crate::PoolMem {
        &self.windows
    }

    /// A slot's window's layout.
    pub fn window(&self) -> WindowLayout {
        self.window
    }

    /// Layer `layer`'s keys and values.
    pub fn layer(&self, layer: usize) -> (&crate::PoolMem, &crate::PoolMem) {
        (&self.keys[layer], &self.values[layer])
    }

    pub fn num_layers(&self) -> usize {
        self.keys.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host word helpers: each token's occurrences in the prompt and the output counted, as
    /// the target's sampler counts them, and counting adds to what a rebuild made.
    #[test]
    fn penalty_words_count_the_prompt_and_the_output() {
        let mut words = vec![7u32; 8];
        rebuild_penalty_words(&mut words, &[1, 2, 2], &[2, 3, 3]);
        assert_eq!(words, [0, 1, 3, 2, 0, 0, 0, 0]);
        count_penalty_tokens(&mut words, &[3, 5]);
        assert_eq!(words, [0, 1, 3, 3, 0, 1, 0, 0]);
        let entry = PenalizeEntry::new(4, &[9, 8], 2, [f32::from_bits(1), 0.5, 0.25]);
        assert_eq!((entry.rows, entry.inputs[..3].to_vec()), (2, vec![9, 8, 0]));
        assert_eq!(entry.repetition_inverse, f32::MAX, "1 / 2^-149 saturates");
    }

    /// A window of 2048 positions of 2048-value rows: 1024 code bytes and 32 groups of 4 bytes a
    /// position, codes first. Positions 10..14 are four slots' codes and groups; positions
    /// 2046..2050 wrap, two slots at the window's end and two at its start, each side's codes and
    /// groups; a whole window starting anywhere is every byte once.
    #[test]
    fn a_windows_positions_are_their_slots_bytes() {
        let w = WindowLayout::new(2048, 2048).expect("a layout");
        let codes = 2048 * 1024;
        assert_eq!(w.bytes(), codes + 2048 * 128);
        assert_eq!(
            w.positions(10, 14),
            [10 * 1024..14 * 1024, codes + 10 * 128..codes + 14 * 128]
        );
        assert_eq!(
            w.positions(2046, 2050),
            [
                2046 * 1024..2048 * 1024,
                0..2 * 1024,
                codes + 2046 * 128..codes + 2048 * 128,
                codes..codes + 2 * 128,
            ]
        );
        for from in [0, 5, 4096 + 700] {
            let total: u64 = w
                .positions(from, from + 2048)
                .iter()
                .map(|r| r.end - r.start)
                .sum();
            assert_eq!(total, w.bytes(), "a whole window from {from}");
        }
        assert!(w.positions(7, 7).is_empty());
        assert_eq!(
            WindowLayout::new(2048, 2000),
            None,
            "2000 is no whole number of groups"
        );
    }
}

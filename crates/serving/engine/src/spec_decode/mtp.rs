// SPDX-License-Identifier: Apache-2.0

//! Multi-token-prediction (MTP) proposer: drafts with a head that ships beside its target model
//! (Qwen3.5/3.6 `mtp.*`; an MLX drafter repo carries it alone; Nemotron-3 Super's OptiQ conversion
//! publishes it as a sidecar file the checkpoint's config names, `mtp_file`) and reads the
//! target's final (post-norm) hidden states.
//!
//! Per step (prefill, decode or verify):
//!
//! 1. **Pass 1 over the step's rows** ([`PassOne`]), in the step's own command buffer after its
//!    target. Row `r`, at position `p`, feeds the head the target's hidden state at `r` and the
//!    token at `p + 1`: the request's history where the host has it (a prompt row gets the next
//!    prompt token), else the target's token at `r`, copied on the device — the step's new tokens,
//!    so an accepted draft's row gets its successor and the newest row the token the target just
//!    produced. Rows past the newest held rejected drafts; they run too, on causal attention that
//!    leaves the rows before them alone, and the head's KV at their positions is rewritten before
//!    it is read. So every row of a verify step can be the newest: the lm_head runs every row, and
//!    the host reads back each new token's row and picks the newest's argmax, draft 1. Every row
//!    writes the head's KV at the target's slot for `p`, so the head attends the whole history.
//!
//!    A row whose next token the host has — a prompt's, short of its newest — is never a
//!    sequence's newest, so nothing reads its output, only its KV. Those rows run first, as their
//!    own forward ([`HeadPasses::prime`]) that plays only the commands its KV write needs — the
//!    compiler's `cache_live`, a constant of each tape from a liveness walk back from the KV
//!    writes — skipping the q projection the rope append rotates beside K and V, the attention,
//!    MLP, norm and lm_head; the rest run whole ([`Drafting`]), attending the KV the first wrote.
//!    A head reading its target's KV writes none, so its rows no draft reads run nothing.
//! 2. **Passes 2..=k** ([`Chain`]), one row per drafting sequence: the previous draft at the next
//!    position, with the head's own final hidden state from the pass before. They run in the same
//!    command buffer: which row is a sequence's newest depends on the drafts the target accepted,
//!    so the device picks each pass's inputs — the token and hidden state at that row, and the
//!    position, slot and used KV length the host laid out for each accepted count. The host reads
//!    back the drafts alone.
//!
//! A parallel-drafting head (P-EAGLE, [`PastFirst::Parallel`]) runs no pass past the first: its
//! pass 1 drafts every token, as vLLM's does.
//!
//! The head's KV pool has the target's blocks, so it shares the target's block table. Only a
//! sequence that produced a token this step drafts; an intermediate prefill chunk still primes its
//! rows' head KV for the prompt.
//!
//! The worker that ran the step runs the passes ([`MtpDrafter`]) for the requests the engine
//! planned when it scheduled the step ([`MtpProposer`]'s `plan`), and returns their drafts with
//! the step's output — so the engine schedules asynchronously, waiting only on a step that drafts
//! or verifies. Each sequence of a step drafts as many tokens as keep the step's verify rows within
//! the head's compiled `spec_verify_rows`, at most its `spec_drafts` ([`DraftBudget`]), so a step
//! of more sequences drafts each fewer. A step of more sequences than one draft each still fits
//! ([`DraftBudget::max_seqs`]) costs what its target alone does, so it drafts nothing — but every
//! row still primes, keeping every sequence's head KV current: no request ever goes stale, and
//! drafting resumes as soon as a step is back within the budget.
//!
//! # TODO
//!
//! Measured on Qwen3.6-35B-A3B (`mlx-community/Qwen3.6-35B-A3B-4bit` with its MLX head) and on
//! Nemotron-3 Super 120B-A12B (`mlx-community/NVIDIA-Nemotron-3-Super-120B-A12B-OptiQ-2bit` with
//! its sidecar head), the two targets with head configs; "sampled" is the model's default
//! sampling, "greedy" temperature 0.
//!
//! **The budget** (`spec_drafts` and `spec_verify_rows`, the head's `arch.json`):
//!
//! - TODO: derive the budget per device, at expansion. It is 8 verify rows on every GPU, the
//!   verify-sized bucket: 2 drafts a sequence at 1-2 sequences (3 and 6 rows), 1 at 3-4 (6 and 8),
//!   none past 4. Before, a fixed gate of 2 sequences drafted 2 each or none. One request pays on
//!   both GPUs measured (base M5 output tok/s 1.36-1.38x sampled; M5 Max, at this head, TPOT
//!   1.33x on the code prompt and 1.51x on a long one, greedy) — on the base M5, before the gate
//!   came down to 1, sampled output tok/s was 1.07-1.18x at 2 sequences and 1.03-1.17x at 4,
//!   greedy 0.96-1.06x at 4 (two rounds each); the M5 Max was not measured past one request.
//!   Past 8 rows the budget does not rise until the verify-sized rungs do: a verify step of 3
//!   sequences at 2 drafts runs 9 rows, past the 8-row bucket onto the 64-row prefill-shaped tape
//!   (`METAL_VERIFY_ROWS`' TODO: derive the rungs from the budget). A verify step
//!   of `B` sequences runs `B * (k + 1)` rows, and a MoE target reads every distinct expert those
//!   rows pick (Qwen3.6: 256 experts, 8 a row; at 8 sequences the verify step's 24 rows pick about
//!   135, the plain step's 8 about 57, under uniform routing). So on the base M5 a verify step
//!   costs 1.6 / 2.0 / 2.2 / 2.65 plain steps at 1 / 2 / 4 / 8 sequences, against about 2.3 tokens
//!   a verify step. The budget wanted is, at each `B`, the drafts whose verify step and head passes
//!   cost less than the plain steps of the tokens they produce. That is a comparison of two compiled
//!   tapes' costs on a device, so it belongs at expansion: one budget per declared device profile
//!   (`MetalTargetProfile`: bandwidth, fp16 peak), the worker picking its device's as it picks
//!   kernels by GPU generation. Blocked on a metal tape cost model: a roofline from the profile
//!   alone predicts 2.0x at 8 sequences against the measured 2.65x.
//! - TODO: key the budget on the target, not the head arch. `spec_verify_rows` is the head arch's
//!   bound default, so every target of `qwen3-5-mtp` shares it, but its cost is the target's: a dense
//!   target's verify rows cost far less than a MoE's (Llama-3.2-3B: a 4-row step costs 1.27x a
//!   1-row step), so a dense Qwen3.5/3.6 target would pay at more sequences.
//! - TODO: priming costs what the head's KV write needs over every row of every step past the
//!   budget (Qwen3.6: the embedding, the two norms, `fc` and the k/v projections). The
//!   alternative — keep a stale request's target hidden rows (4 KiB a token at Qwen3.6's
//!   2048-wide 16-bit hidden) and run them through pass 1 once its steps are back within the
//!   budget — defers that cost to the requests that resume drafting; not taken.
//! - TODO: a prefix-cache hit can still reuse head KV that was never written: the head's KV
//!   lives in the target's blocks, so a request that hits cached blocks reuses their head KV
//!   too, and blocks written by a request that takes no drafts (its sampling reads its token
//!   history) hold none. A request drafting over them gets fewer drafts accepted (the target
//!   verifies every draft, so the output is unaffected). Not measured.
//!
//! **Cost:**
//!
//! - TODO: a verify step's cost past one sequence, which is what holds the budget down. Base M5, 8
//!   sequences: a plain step is 38.4 ms and a verify step yields about 2.3 tokens, so the verify
//!   step must fit in about 87 ms to break even. The experts take about 60 ms of it, the
//!   memory-bandwidth floor for the experts 24 rows pick (uniform routing), so the rest must fit in
//!   about 27 ms; it takes about 44 (dense projections 25 ms against about 13 at 8 rows; the
//!   Gated-DeltaNet scan 10-12 ms, each sequence's rows and the kept rows it replays one after
//!   another; attention 3; the rest about 6) plus the head's about 5, measured with 16- and
//!   32-row verify-sized rungs. Those rungs cost short prompts their first token, and the budget
//!   keeps verify steps within 8 rows, so they are gone (`METAL_VERIFY_ROWS`, 8): past 8 rows a
//!   verify step runs the 64-row prefill-shaped tape until the rungs derive from the budget.
//! - TODO: a build with a head runs the steps that replay or record — a verify step and the step
//!   after one — on Gated-DeltaNet kernels that cost more per row. A plain step runs what it runs
//!   without the head (Qwen3.6's geometry: a decode step #265's one-command decode, a prefill #263's
//!   pipelined scan; the drafts only stride their slots). The others run the conv, the simd scan
//!   and the norm, whose loops carry the replay: before #265, when every decode step ran them, a
//!   2-sequence decode step's 30 scans went 1.69 -> 2.20 ms with the head and its 30 convs 0.13 ->
//!   0.46 ms (base M5, a one-step kernel probe; the conv replays nothing — it starts from a
//!   checkpoint entry — and its cost is not localized). Not measured on the verify steps. The
//!   simd scan's replay is in its per-token loop, one loop body so the replay is bit-exact
//!   (`gdn_scan_simd_resumes_from_checkpoint`): two loops calling one inlined row lost the
//!   bit-exactness (the varlen scan's replay 1 ULP off), the compiler contracting each copy apart.
//!   Without a head both compile to the plain kernels (`GDN_SCAN_DRAFTS`, `GDN_CONV_DRAFTS` 0).
//! - TODO: time to first token pays the head's prime over the prompt — what its KV write needs
//!   over every prompt row — then its newest row whole and the chain. Per-dispatch GPU time on a
//!   1588-row prompt chunk (base M5): 5.2 ms (prime 3.1, newest row 1.1, chain 1.0), from 34.2 ms
//!   when every row ran the whole head; first token on a ~5.6k-token prompt +0.7% against main.
//!   The prompt's last chunk drafts, so it is not deferred, and waits for the chunk still in
//!   flight before its own inputs are written (~5 ms idle on the base M5): writing them on the
//!   device, as a deferred step does, would let it queue behind that chunk.
//! - TODO: the next step is scheduled once the host has read this one's tokens and drafts (#240:
//!   pipelined cycles). Its inputs are on the device already.
//! - TODO: choose `k` per bucket. `spec_drafts` is one constant for every bucket, because the
//!   target's verify rows a sequence and its Gated-DeltaNet record areas are baked from it; a
//!   bucket-specific `k` needs both sized to the largest. Gemma 4's assistant bakes 1, the
//!   wall-clock winner despite the lower acceptance: its first draft's prose acceptance is a
//!   coin flip (p1 0.42-0.69 across the M5 Max battery's chat rungs), so the second draft's
//!   head forward and verify row cost more than they return — k = 2 lost 1-13% tokens/s on
//!   every chat cell, its only win single-request patterned serve (+10%, on replies whose p1
//!   is 0.92). Greedy tokens a verify step at k = 1 / 2 / 3 were 1.77 / 2.20 / 2.36 against
//!   the MLX oracle's 1.80 / 2.43 / 2.82, measured before the fold fix that keeps an exported
//!   final hidden written (after a verify step that
//!   kept no draft, the second draft matched the oracle's 5 times in 23 before it, 28 in 28
//!   after); k = 3 is not re-measured since.
//! - Measured on Nemotron-3 Super 120B-A12B (OptiQ-2bit, its sidecar head; M5 Max, one request,
//!   greedy, `scr chat --bench`, a short prompt): a plain step 15.0 ms, a drafting step 25.3 ms
//!   (p10-p90 23.8-27.0), 2.31 tokens a step, output 67.7 -> 91.4 tok/s (+35%), time to first
//!   token 107.5 -> 110.8 ms. Before the shared-experts fix it was flat (66.9 tok/s, a 35.8 ms
//!   drafting step) — and the flatness was a kernel mode, not structure. The verify-sized
//!   gathered MoE bucket ran its pairs that share an expert in the first pair's threadgroup
//!   (`SharedExperts::InFirstPair`), whose every threadgroup scans the bucket's every other
//!   pair: a cost quadratic in the pair count, and this target routes 22 of 512 experts a row,
//!   so a 3-row verify's 66 live pairs blow past the mode's few-pairs precondition while its
//!   saved reads stay the ~3 pairs that share an expert. Per-dispatch GPU time
//!   (`SCRATCHY_METAL_KERNEL_TIME`, probe-inflated): the drafting step was 38.3 ms of verify
//!   plus 2.7 of head passes — the head is under 7% of it, innocent — and the verify's
//!   `MoeGateUpAct` alone ran 17.3 ms against the plain step's 3.6, 4.8x for 3x the pairs where
//!   the dense projections scaled 1.2x, while the same kernel at the prefill bucket's 594 pairs
//!   runs 1.6 µs a pair against the verify bucket's 6.6. Running a gathered bake's pairs apart
//!   past 64 baked pairs (`METAL_SHARED_EXPERTS_MAX_PAIRS`) took `MoeGateUpAct` to 4.7 ms —
//!   1.8 µs a pair, the prefill bucket's efficiency, 1.3x the plain step's cost for the ~63
//!   distinct experts 3 rows draw (uniform routing; the plain step's 22) — and the step to
//!   25.3 ms. The acceptance (2.31 tokens a step) clears it: those tokens' plain steps cost
//!   34.7 ms. The head pays where the routed experts hold a minority of the plain step's stream
//!   (Qwen3.6: 256 experts, 8 a row — 32 baked pairs at the verify bucket, under the cap, the
//!   mode unchanged there) or where KV reads dilute it; not measured past one sequence (its
//!   budget is Qwen3.6's, drafting through 4 sequences: an 8-row verify step's 176 baked pairs
//!   run apart) nor at depth of context.
//!
//! **Measurement, interface, size:**
//!
//! - TODO: measure the M5 Max at this build. Its numbers above predate worker-side drafting and the
//!   gate of 1; the PR's two-build scripts (without `spec/mtp` vs with it, every serving knob at
//!   its default) are the measurement.
//! - TODO: on a 32 GB Mac (base M5) Qwen3.6 serves one sequence at its defaults (#255), with or
//!   without a head, so the base-M5 numbers above at 2 and 8 sequences are development runs at
//!   `--max-num-seqs 8`. There the head costs its KV cache 145,936 -> 89,520 tokens (-39%): its
//!   weights (weights+overhead 19.3 -> 19.8 GiB) come out of a KV budget of about 1 GiB, and its
//!   layer's KV takes 528 bytes a token beside the target's 7,328 (its target's TurboQuant codes,
//!   staged through its target's scratch). The prefill bucket stays 2048. Winning the weights'
//!   share back on a small Mac means shrinking something else there, such as the prefill bucket.
//! - TODO: `--num-speculative-tokens 0` serves without the head; any other count given with a head
//!   is ignored (the head drafts its compiled count), and `serve`'s flag defaults to 2, so it
//!   cannot tell a given value from none. Make it optional, and refuse a value that differs from
//!   the head's.

use std::collections::{HashMap, HashSet};

use scratchy_serving_scheduler::scheduler::output::SchedulerOutput;

use super::backend::ForwardArgmaxRequest;
use super::proposer::{Proposer, ProposerStepCtx};

/// How many tokens a head drafts each sequence of a step: as many as keep the step's verify rows —
/// each sequence's token and its drafts — within `verify_rows` (the head's compiled
/// `spec_verify_rows`, the rows its target's verify-shaped bucket holds), at most `drafts` (its
/// `spec_drafts`). A step of few sequences drafts the most, a larger one fewer, and one past
/// [`Self::max_seqs`] none: its verify step would leave the verify-shaped bucket for a
/// prefill-shaped one, where a verify row costs what a prompt row does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftBudget {
    pub drafts: usize,
    pub verify_rows: usize,
}

impl DraftBudget {
    /// The drafts each of a step's `seqs` sequences gets.
    pub fn drafts_at(self, seqs: usize) -> usize {
        (self.verify_rows / seqs.max(1))
            .saturating_sub(1)
            .min(self.drafts)
    }

    /// The most sequences a step drafts for: one draft each still fits.
    pub fn max_seqs(self) -> usize {
        match self.drafts {
            0 => 0,
            _ => self.verify_rows / 2,
        }
    }

    /// The most draft rows a step of at most `width` sequences carries beside its sequences' own.
    pub fn draft_rows(self, width: usize) -> usize {
        let seqs = 1..=self.max_seqs().min(width);
        seqs.map(|c| c * self.drafts_at(c)).max().unwrap_or(0)
    }
}

/// Configuration for the MTP proposer, held by [`super::ProposerConfig::Mtp`].
#[derive(Debug, Clone)]
pub struct MtpProposerConfig {
    /// Drafts per step (`k`, the most): one pass-1 row plus `k - 1` chained passes.
    pub num_speculative_tokens: usize,
    /// The most sequences a step drafts for (`DraftBudget::max_seqs`), the engine's copy of the
    /// gate the worker's `MtpDrafter.budget` carries (both derive from the head's registration,
    /// so they cannot disagree).
    pub max_seqs: usize,
    /// Maximum model context length: no draft lands past it.
    pub max_model_len: usize,
}

/// The MTP proposer (module docs).
#[derive(Debug)]
pub struct MtpProposer {
    config: MtpProposerConfig,
}

impl MtpProposer {
    pub fn new(config: MtpProposerConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &MtpProposerConfig {
        &self.config
    }
}

/// The head's passes over one step (module docs), run by the worker that ran it.
#[derive(Clone, Copy, Debug)]
pub struct MtpDrafter {
    /// The drafts each sequence of a step gets, by the step's sequences: a step past its
    /// [`DraftBudget::max_seqs`] runs as its target alone, its pass 1 priming KV only (module
    /// docs).
    pub budget: DraftBudget,
    /// Maximum model context length: no draft lands past it.
    pub max_model_len: usize,
    /// The head reads its target's KV and writes none (Gemma 4's assistant): row `r`'s query sits
    /// at the position after it (the token it reads is there), attending the target's KV through
    /// `r`; no row writes KV; and a sequence's chained drafts keep its newest row's query position
    /// and KV. A head with KV of its own writes its row's KV at the row's position, and each
    /// chained draft is one position further.
    pub reads_target_kv: bool,
    /// How it drafts past its first draft.
    pub past_first: PastFirst,
}

/// How a head drafts past its first draft ([`MtpDrafter::past_first`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PastFirst {
    /// A pass a draft, each reading the draft and final hidden state before it.
    Chained,
    /// Pass 1 drafts every token (P-EAGLE, vLLM's parallel drafting): a drafting sequence's rows
    /// run on, a row a draft past the first, at the positions after its last row's, reading the
    /// head's placeholders — `mask_token`, and a target hidden row its worker holds. On the
    /// device, by what the target accepted, the rows through the newest read the target's tokens
    /// and hidden rows, the next ones the placeholders and draft past the newest; the rows the
    /// target rejected read placeholders too, and nothing reads theirs. Every row's position is
    /// the host's.
    Parallel { mask_token: u32 },
}

/// A step's rows as its target runs them, in sequence order: what the head's passes read.
#[derive(Clone, Copy, Debug)]
pub struct StepRows<'a> {
    pub req_ids: &'a [String],
    pub input_ids: &'a [u32],
    pub positions: &'a [u32],
    pub slot_mapping: &'a [u32],
    pub cu_seqlens_q: &'a [u32],
    pub seqused_k: &'a [u32],
    /// Each sequence's row of `block_table_stride` blocks.
    pub block_table: &'a [u32],
    pub block_table_stride: usize,
    /// Each sequence's blocks.
    pub block_ids: &'a [Vec<u32>],
    pub block_size: usize,
}

impl StepRows<'_> {
    /// Rows `is` of the step's block table.
    fn block_rows(&self, is: &[usize]) -> Vec<u32> {
        let stride = self.block_table_stride;
        (is.iter())
            .flat_map(|&i| &self.block_table[i * stride..(i + 1) * stride])
            .copied()
            .collect()
    }

    /// Sequence `i`'s KV slot for position `p`; `u32::MAX` (no write) past its blocks.
    fn slot(&self, i: usize, p: usize) -> u32 {
        let bs = self.block_size;
        (self.block_ids[i].get(p / bs)).map_or(u32::MAX, |&b| (b as usize * bs + p % bs) as u32)
    }
}

/// The head's passes over a step (module docs), run in the step's command buffer after its target.
#[derive(Debug, PartialEq)]
pub struct HeadPasses {
    /// Pass 1 over the rows no draft reads, for their KV alone: each planned sequence's rows whose
    /// next token the host has (a prompt's, short of its newest), or every row past the head's
    /// [`DraftBudget::max_seqs`]. `None` when there are none, or the head keeps no KV of its own.
    pub prime: Option<PassOne>,
    /// Pass 1 over the rows a draft can read, and the passes chained after it. `None` when no
    /// planned sequence gets a token from the step (prefill chunks short of their prompt's end),
    /// or the step is past the head's [`DraftBudget::max_seqs`]: nothing drafts.
    pub drafting: Option<Drafting>,
}

/// The passes that draft: pass 1 over each drafting sequence's rows whose next token is the
/// target's, every one of which can be its newest, then passes 2..=k.
#[derive(Debug, PartialEq)]
pub struct Drafting {
    pub one: PassOne,
    pub chain: Chain,
}

/// A head forward of pass 1 over rows of a step: the rows' host inputs, and what the device copies
/// from the target.
#[derive(Debug, PartialEq)]
pub struct PassOne {
    /// Each sequence's rows of the pass, in order. A row's input token is the one after its
    /// position: the host's, or a placeholder the target's overwrites ([`Self::token_rows`]).
    pub input_ids: Vec<u32>,
    pub positions: Vec<u32>,
    pub slot_mapping: Vec<u32>,
    pub cu_seqlens_q: Vec<u32>,
    pub seqused_k: Vec<u32>,
    pub block_table: Vec<u32>,
    pub block_table_stride: usize,
    /// Each sequence's last row: the row the lm_head samples on a step without speculative rows.
    pub last_rows: Vec<u32>,
    /// Some sequence has more than one row that can be its newest: the lm_head runs every row.
    pub has_spec_tokens: bool,
    /// `(row, step row)` of each row whose input is the target's token at its step row — one the
    /// host does not have yet. One of a sequence's is its newest; their argmax runs on the device.
    pub token_rows: Vec<(u32, u32)>,
    /// `(row, step row, rows)`: each sequence's rows, reading the target's final hidden states at
    /// its rows of the step.
    pub hidden_rows: Vec<(u32, u32, u32)>,
    /// Each sequence's index in the step: whose rows of the step's block tables it reads.
    pub steps: Vec<usize>,
}

/// The passes after pass 1, over each drafting sequence — one the step gives a token — on the
/// device after pass 1 (module docs): a chained head's passes 2..=k, one row a sequence each (a
/// parallel-drafting head runs none, [`PastFirst`]). Their host inputs, and the tables the device
/// picks the rest from by the number of drafts the target accepted.
#[derive(Debug, PartialEq)]
pub struct Chain {
    /// Per drafting sequence: its first token row in the step, its token rows, and its first token
    /// row in pass 1's rows.
    pub seqs: Vec<[u32; 3]>,
    /// Per step row, the token the row after it reads — the draft it verifies within a sequence.
    pub drafted: Vec<u32>,
    /// After each of its passes: per drafting sequence, accepted count `0..=k` (`k` the head's
    /// most drafts, [`Self::drafts`]) and row of the next pass ([`Self::rows`]), that row's
    /// position, slot and used KV length (zeros after the last pass, which has none).
    pub next: Vec<Vec<u32>>,
    /// Each chained pass's host inputs, a row per drafting sequence (a parallel-drafting head runs
    /// no chained pass). The token, position, slot and used KV length are placeholders the device
    /// overwrites — the position and used length the largest the last pass takes, the slot none.
    pub input_ids: Vec<u32>,
    pub positions: Vec<u32>,
    pub slot_mapping: Vec<u32>,
    pub cu_seqlens_q: Vec<u32>,
    pub seqused_k: Vec<u32>,
    pub block_table: Vec<u32>,
    pub block_table_stride: usize,
    pub last_rows: Vec<u32>,
    /// Each drafting sequence's index in the step.
    steps: Vec<usize>,
    drafts: usize,
    /// The rows a sequence each chained pass runs, one a draft it makes; a parallel-drafting
    /// head's rows past each sequence's newest in pass 1.
    rows: usize,
    /// The drafts the passes make each sequence.
    made: usize,
}

impl PassOne {
    /// The forward's host-side request (its target hidden states come from the device).
    pub fn request(&self) -> ForwardArgmaxRequest<'_> {
        request(
            [&self.input_ids, &self.positions, &self.slot_mapping],
            [&self.cu_seqlens_q, &self.seqused_k, &self.block_table],
            self.block_table_stride,
            Some(&self.last_rows),
            self.has_spec_tokens,
        )
    }

    fn empty(block_table_stride: usize) -> Self {
        Self {
            input_ids: Vec::new(),
            positions: Vec::new(),
            slot_mapping: Vec::new(),
            cu_seqlens_q: vec![0],
            seqused_k: Vec::new(),
            block_table: Vec::new(),
            block_table_stride,
            last_rows: Vec::new(),
            has_spec_tokens: false,
            token_rows: Vec::new(),
            hidden_rows: Vec::new(),
            steps: Vec::new(),
        }
    }

    /// Sequence `i`'s step rows `rows`, its KV used through `used` after them, appended; `tokens`
    /// is its history. A head reading its target's KV queries one past each row and writes no slot.
    fn push(
        &mut self,
        step: &StepRows<'_>,
        i: usize,
        rows: std::ops::Range<usize>,
        tokens: &[u32],
        reads_target_kv: bool,
        used: u32,
    ) {
        let start = self.input_ids.len();
        for r in rows.clone() {
            let p = step.positions[r] as usize;
            let row = self.input_ids.len() as u32;
            let token = tokens.get(p + 1).copied().unwrap_or_else(|| {
                self.token_rows.push((row, r as u32));
                0
            });
            self.input_ids.push(token);
            match reads_target_kv {
                true => {
                    self.positions.push(step.positions[r] + 1);
                    self.slot_mapping.push(u32::MAX);
                }
                false => {
                    self.positions.push(step.positions[r]);
                    self.slot_mapping.push(step.slot_mapping[r]);
                }
            }
        }
        let (row, step_row) = (start as u32, rows.start as u32);
        self.hidden_rows.push((row, step_row, rows.len() as u32));
        self.cu_seqlens_q.push(self.input_ids.len() as u32);
        self.seqused_k.push(used);
        self.last_rows.push(self.input_ids.len() as u32 - 1);
        self.steps.push(i);
    }

    /// `count` rows more for the sequence pushed last, at the positions after its last row's: a
    /// parallel-drafting head's rows past its newest ([`PastFirst::Parallel`]), reading
    /// `mask_token` and, on the device, the head's placeholder target hidden row.
    fn push_masks(&mut self, step: &StepRows<'_>, mask_token: u32, count: usize) {
        let (Some(&i), Some(&last)) = (self.steps.last(), self.positions.last()) else {
            return;
        };
        for p in (last + 1..).take(count) {
            self.input_ids.push(mask_token);
            self.positions.push(p);
            self.slot_mapping.push(step.slot(i, p as usize));
        }
        let rows = self.input_ids.len() as u32;
        let [Some(end), Some(used), Some(last_row)] = [
            self.cu_seqlens_q.last_mut(),
            self.seqused_k.last_mut(),
            self.last_rows.last_mut(),
        ] else {
            return;
        };
        (*end, *used, *last_row) = (rows, last + count as u32 + 1, rows - 1);
    }

    /// The pass, its sequences' block tables laid out from `step`'s; `None` with no row.
    fn of(mut self, step: &StepRows<'_>) -> Option<Self> {
        self.block_table = step.block_rows(&self.steps);
        (!self.input_ids.is_empty()).then_some(self)
    }
}

impl Chain {
    /// A chained pass's host-side request (its target hidden states come from the device). A pass
    /// of several rows a sequence drafts at every one: the lm_head runs every row.
    pub fn request(&self) -> ForwardArgmaxRequest<'_> {
        request(
            [&self.input_ids, &self.positions, &self.slot_mapping],
            [&self.cu_seqlens_q, &self.seqused_k, &self.block_table],
            self.block_table_stride,
            Some(&self.last_rows),
            self.rows > 1,
        )
    }

    /// The drafting sequences.
    pub fn num_seqs(&self) -> usize {
        self.steps.len()
    }

    /// The head's passes over the step's drafting sequences: pass 1 and the chained ones.
    pub fn passes(&self) -> usize {
        self.next.len()
    }

    /// The rows a sequence each chained pass runs: one for a chained head; a parallel-drafting
    /// one's pass 1 runs this many past each sequence's newest, a draft past the first each.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The drafts the passes make each sequence: pass 1's and a chained pass's one each, a
    /// parallel-drafting head's pass 1 every one.
    pub fn made(&self) -> usize {
        self.made
    }

    /// The head's most drafts a sequence, whatever this chain's passes: the room a sequence takes
    /// in [`Self::next`]'s accepted counts and in the drafts read back, as the chain kernel bakes
    /// it.
    pub fn drafts(&self) -> usize {
        self.drafts
    }

    /// Each drafting sequence's index in the step.
    pub fn steps(&self) -> &[usize] {
        &self.steps
    }
}

/// A head forward's request over `rows` (its tokens, positions and slots) and `seqs` (its
/// cumulative sequence lengths, used KV lengths and block table rows).
fn request<'a>(
    [input_ids, positions, slot_mapping]: [&'a [u32]; 3],
    [cu_seqlens_q, seqused_k, block_table]: [&'a [u32]; 3],
    block_table_stride: usize,
    last_rows: Option<&'a [u32]>,
    has_spec_tokens: bool,
) -> ForwardArgmaxRequest<'a> {
    ForwardArgmaxRequest {
        input_ids,
        positions,
        slot_mapping,
        cu_seqlens_q,
        seqused_k,
        block_table,
        span_ids: None,
        sliding_slot_mappings: &[],
        sliding_block_tables: &[],
        block_table_stride,
        max_seqlen_q: (cu_seqlens_q.windows(2))
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0),
        max_seqlen_k: seqused_k.iter().copied().max().unwrap_or(0) as usize,
        num_tokens: input_ids.len(),
        has_spec_tokens,
        last_token_indices: last_rows,
        target_hidden: None,
    }
}

impl MtpDrafter {
    /// The head's passes over the rows of `step` that the requests in `plan` run, before the step:
    /// `history` reads each request's tokens so far. `None` when none is planned.
    ///
    /// `holds_draft_state` is asked for each request that would draft: whether the target can
    /// keep what its next step's verify undoes (a Gated-DeltaNet target's draft lane,
    /// `GdnSlotAllocator::hold_lane`). A request it refuses only primes, as past the gate.
    pub fn passes(
        &self,
        step: &StepRows<'_>,
        history: &dyn Fn(&str) -> Option<Vec<u32>>,
        plan: &HashSet<String>,
        holds_draft_state: &mut dyn FnMut(&str) -> bool,
    ) -> Option<HeadPasses> {
        let stride = step.block_table_stride;
        let (mut prime, mut one) = (PassOne::empty(stride), PassOne::empty(stride));
        // Per drafting sequence: its index in the step, its token rows' span of `one.token_rows`.
        let mut drafting: Vec<(usize, std::ops::Range<usize>)> = Vec::new();
        // Past the gate every row primes; within it, a sequence's rows from the first whose next
        // token the host does not have draft — each can be its newest — and those before prime.
        // The step's sequences each draft what its verify rows leave room for.
        let passes = self.budget.drafts_at(step.req_ids.len());
        let past_gate = passes == 0;
        for i in (0..step.req_ids.len()).filter(|&i| plan.contains(&step.req_ids[i])) {
            let tokens = history(&step.req_ids[i]).unwrap_or_default();
            let rows = step.cu_seqlens_q[i] as usize..step.cu_seqlens_q[i + 1] as usize;
            let known = |&r: &usize| tokens.get(step.positions[r] as usize + 1).is_some();
            let split = match past_gate {
                true => rows.end,
                false => rows.start + rows.clone().take_while(known).count(),
            };
            let split = match split < rows.end && !holds_draft_state(&step.req_ids[i]) {
                true => rows.end,
                false => split,
            };
            // A head reading its target's KV keeps none of its own: its rows no draft reads run
            // nothing.
            if split > rows.start && !self.reads_target_kv {
                let used = step.positions[split - 1] + 1;
                prime.push(step, i, rows.start..split, &tokens, false, used);
            }
            if split < rows.end {
                let first = one.token_rows.len();
                let used = step.seqused_k[i];
                one.push(
                    step,
                    i,
                    split..rows.end,
                    &tokens,
                    self.reads_target_kv,
                    used,
                );
                drafting.push((i, first..one.token_rows.len()));
                if let PastFirst::Parallel { mask_token } = self.past_first {
                    one.push_masks(step, mask_token, passes - 1);
                }
            }
        }
        one.has_spec_tokens = (one.cu_seqlens_q.windows(2)).any(|w| w[1] - w[0] > 1);
        let (prime, one) = (prime.of(step), one.of(step));
        let drafting = one.map(|one| Drafting {
            chain: self.chain(step, &one, &drafting, passes),
            one,
        });
        (prime.is_some() || drafting.is_some()).then_some(HeadPasses { prime, drafting })
    }

    /// The passes after `one` for the `drafting` sequences (each its index in the step and its
    /// span of `one`'s token rows), `drafts` drafts each: a chained head's passes 2..=`drafts`; a
    /// parallel-drafting head's none, its pass 1 drafting every one. A sequence's accepted
    /// count — what the step's verify kept of the drafts it verified — runs to the head's most
    /// drafts.
    fn chain(
        &self,
        step: &StepRows<'_>,
        one: &PassOne,
        drafting: &[(usize, std::ops::Range<usize>)],
        drafts: usize,
    ) -> Chain {
        let k = self.budget.drafts;
        let n = drafting.len();
        let past_first = self.past_first;
        // The passes the drafts take, and the rows a sequence each chained one runs: a chained
        // head's pass a draft, of one row a sequence; a parallel-drafting head's pass 1 alone,
        // whose rows past each sequence's newest draft past the first.
        let (passes, rows) = match past_first {
            PastFirst::Chained => (drafts, 1),
            PastFirst::Parallel { .. } => (1, drafts - 1),
        };
        let steps: Vec<usize> = drafting.iter().map(|&(i, _)| i).collect();
        let seqs: Vec<[u32; 3]> = (drafting.iter())
            .map(|(_, tokens)| {
                let (row, step_row) = one.token_rows[tokens.start];
                [step_row, tokens.len() as u32, row]
            })
            .collect();
        // The step's next-row tokens: within a sequence, the draft each row verifies.
        let mut drafted = step.input_ids[1..].to_vec();
        drafted.push(0);
        // After pass `d`, a chained head's next row at accepted count `a` sits at `first + a + d`,
        // the newest row's position plus `d` — or, reading the target's KV, at `first + a + 1` for
        // every pass, attending the target's KV through `first + a`, writing none.
        let target_kv = self.reads_target_kv;
        let next = (1..=passes)
            .map(|d| {
                (drafting.iter().zip(&seqs))
                    .flat_map(|(&(i, _), &[step_row, ..])| {
                        let first = step.positions[step_row as usize] as usize;
                        (0..=k).flat_map(move |a| {
                            (0..rows).flat_map(move |_| match (d < passes, target_kv) {
                                (false, _) => [0; 3],
                                (true, false) => {
                                    let p = first + a + d;
                                    [p as u32, step.slot(i, p), p as u32 + 1]
                                }
                                (true, true) => {
                                    let p = (first + a + 1) as u32;
                                    [p, u32::MAX, p]
                                }
                            })
                        })
                    })
                    .collect()
            })
            .collect();
        // A chained pass's placeholders: its positions at the most drafts accepted. A
        // parallel-drafting head runs none.
        let chained = match past_first {
            PastFirst::Chained => n,
            PastFirst::Parallel { .. } => 0,
        };
        let positions: Vec<u32> = (seqs.iter())
            .take(chained)
            .map(|&[step_row, rows, _]| {
                let newest = step.positions[step_row as usize] + rows - 1;
                match target_kv {
                    true => newest + 1,
                    false => newest + passes as u32 - 1,
                }
            })
            .collect();
        Chain {
            seqs,
            drafted,
            next,
            input_ids: vec![0; chained],
            seqused_k: positions.iter().map(|p| p + 1).collect(),
            positions,
            slot_mapping: vec![u32::MAX; chained],
            cu_seqlens_q: (0..=chained as u32).collect(),
            block_table: step.block_rows(&steps[..chained]),
            block_table_stride: step.block_table_stride,
            last_rows: (0..chained as u32).collect(),
            steps,
            drafts: k,
            rows,
            made: drafts,
        }
    }

    /// The drafts after `chain` ran: `picked`, the head's most drafts' room per drafting sequence,
    /// read back — the chain's passes filled the first of each — for the requests of `step` that
    /// produced `produced` (per request, in step order) and whose token histories `history` reads,
    /// the step's tokens appended. No draft lands past the maximum model length.
    pub fn drafts(
        &self,
        chain: &Chain,
        picked: &[u32],
        step: &StepRows<'_>,
        produced: &[Vec<u32>],
        history: &dyn Fn(&str) -> Option<Vec<u32>>,
    ) -> HashMap<String, Vec<u32>> {
        let (k, made) = (chain.drafts(), chain.made());
        // Each sequence's newest row: the position before its newest token.
        let newest: Vec<Option<usize>> = (chain.steps.iter())
            .map(|&i| {
                let tokens = history(&step.req_ids[i]).unwrap_or_default().len();
                tokens.checked_sub(2).filter(|_| !produced[i].is_empty())
            })
            .collect();
        // Draft `d + 1`'s row runs at the newest position plus `d` (a chained head's pass `d + 1`,
        // a parallel head's row `d` past the newest), and its draft lands one past it.
        let fits = |d: usize| {
            newest
                .iter()
                .flatten()
                .all(|&p| p + d + 2 < self.max_model_len)
        };
        let depth = 1 + (1..made).take_while(|&d| fits(d)).count();
        (chain.steps.iter().zip(&newest).enumerate())
            .filter_map(|(s, (&i, newest))| {
                let d = picked[s * k..][..depth].to_vec();
                let newest = (*newest)?;
                tracing::debug!(
                    "MTP drafts for {} after position {}: {d:?}",
                    step.req_ids[i],
                    newest + 1
                );
                Some((step.req_ids[i].clone(), d))
            })
            .collect()
    }
}

impl Proposer for MtpProposer {
    fn plan(
        &mut self,
        sched: &SchedulerOutput,
        takes_drafts: &dyn Fn(&str) -> bool,
    ) -> HashSet<String> {
        // Every request `takes_drafts` admits is planned, gated or not: a step within the head's
        // budget chains for them, and a step past it runs pass 1 KV-only over their rows
        // (module docs), so head KV never has a hole.
        (sched.num_scheduled_tokens.keys())
            .filter(|r| takes_drafts(r))
            .cloned()
            .collect()
    }

    fn max_seqs(&self) -> Option<usize> {
        Some(self.config.max_seqs)
    }

    fn propose_for_step(&mut self, ctx: &mut ProposerStepCtx<'_>) -> HashMap<String, Vec<u32>> {
        // The worker drafted for the requests this step planned.
        ctx.worker_drafts.cloned().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A step of two sequences: `a` verifies three drafts at positions 10..=13 (its history holds
    /// 11 tokens, `500..511`; the drafts `21..=23`); `b` runs an intermediate prefill chunk, 0..=2
    /// of an 8-token prompt.
    struct TwoSequences {
        req_ids: Vec<String>,
        block_ids: Vec<Vec<u32>>,
        history_a: Vec<u32>,
        history_b: Vec<u32>,
    }

    impl TwoSequences {
        fn new() -> Self {
            Self {
                req_ids: vec!["a".into(), "b".into()],
                block_ids: vec![vec![3], vec![5]],
                history_a: (500..511).collect(),
                history_b: (700..708).collect(),
            }
        }

        fn rows(&self) -> StepRows<'_> {
            StepRows {
                req_ids: &self.req_ids,
                input_ids: &[510, 21, 22, 23, 700, 701, 702],
                positions: &[10, 11, 12, 13, 0, 1, 2],
                slot_mapping: &[58, 59, 60, 61, 80, 81, 82],
                cu_seqlens_q: &[0, 4, 7],
                seqused_k: &[14, 3],
                block_table: &[3, 0, 5, 0],
                block_table_stride: 2,
                block_ids: &self.block_ids,
                block_size: 16,
            }
        }

        /// The histories before the step.
        fn before(&self) -> impl Fn(&str) -> Option<Vec<u32>> + '_ {
            |id| match id {
                "a" => Some(self.history_a.clone()),
                "b" => Some(self.history_b.clone()),
                _ => None,
            }
        }

        /// The histories after it: `a` kept `kept` drafts and took its bonus token.
        fn after(&self, kept: u32) -> impl Fn(&str) -> Option<Vec<u32>> + '_ {
            move |id| match id {
                "a" => Some((500..512 + kept).collect()),
                _ => self.before()(id),
            }
        }
    }

    /// A step of three sequences: `a` and `b` as [`TwoSequences`]'s, plus `c` decoding at position
    /// 5 of its history (its block 7).
    struct ThreeSequences {
        req_ids: Vec<String>,
        block_ids: Vec<Vec<u32>>,
        history_c: Vec<u32>,
    }

    impl ThreeSequences {
        fn new() -> Self {
            Self {
                req_ids: vec!["a".into(), "b".into(), "c".into()],
                block_ids: vec![vec![3], vec![5], vec![7]],
                history_c: (800..805).collect(),
            }
        }

        fn rows(&self) -> StepRows<'_> {
            StepRows {
                req_ids: &self.req_ids,
                input_ids: &[510, 21, 22, 23, 700, 701, 702, 804],
                positions: &[10, 11, 12, 13, 0, 1, 2, 5],
                slot_mapping: &[58, 59, 60, 61, 80, 81, 82, 117],
                cu_seqlens_q: &[0, 4, 7, 8],
                seqused_k: &[14, 3, 6],
                block_table: &[3, 0, 5, 0, 7, 0],
                block_table_stride: 2,
                block_ids: &self.block_ids,
                block_size: 16,
            }
        }

        /// The histories before the step: `a`'s and `b`'s as [`TwoSequences`]'s.
        fn before(&self) -> impl Fn(&str) -> Option<Vec<u32>> + '_ {
            |id| match id {
                "a" => Some((500..511).collect()),
                "b" => Some((700..708).collect()),
                "c" => Some(self.history_c.clone()),
                _ => None,
            }
        }
    }

    fn proposer() -> MtpProposer {
        MtpProposer::new(MtpProposerConfig {
            num_speculative_tokens: 3,
            max_seqs: 2,
            max_model_len: 4096,
        })
    }

    /// Three drafts a sequence while they fit an 8-row verify step (1–2 sequences), one at 3–4:
    /// [`TwoSequences`] chains three passes, [`ThreeSequences`] one.
    const DRAFTER: MtpDrafter = MtpDrafter {
        budget: DraftBudget {
            drafts: 3,
            verify_rows: 8,
        },
        max_model_len: 4096,
        reads_target_kv: false,
        past_first: PastFirst::Chained,
    };

    /// A step's sequences each draft what its verify rows leave room for, at most the head's
    /// drafts, and nothing once one draft a sequence does not fit: Qwen3.6's head (2 drafts,
    /// 8 rows) drafts 2 at 1–2 sequences, 1 at 3–4, none past 4; Gemma 4's assistant (1 draft,
    /// 3 rows) 1 at one sequence, none past it.
    #[test]
    fn a_step_drafts_what_its_verify_rows_leave_room_for() {
        let qwen = DraftBudget {
            drafts: 2,
            verify_rows: 8,
        };
        let at = |b: DraftBudget| (1..=6).map(|c| b.drafts_at(c)).collect::<Vec<_>>();
        assert_eq!(at(qwen), [2, 2, 1, 1, 0, 0]);
        assert_eq!((qwen.max_seqs(), qwen.draft_rows(usize::MAX)), (4, 4));
        assert_eq!(qwen.draft_rows(1), 2, "a width of one drafts one sequence");
        let gemma = DraftBudget {
            drafts: 1,
            verify_rows: 3,
        };
        assert_eq!(at(gemma), [1, 0, 0, 0, 0, 0]);
        assert_eq!((gemma.max_seqs(), gemma.draft_rows(usize::MAX)), (1, 1));
        for b in [qwen, gemma] {
            for c in 1..=8 {
                assert!(c * (1 + b.drafts_at(c)) <= b.verify_rows || b.drafts_at(c) == 0);
                assert_eq!(b.drafts_at(c) > 0, c <= b.max_seqs(), "{b:?} at {c}");
            }
        }
    }

    /// Three sequences at a budget that drafts one each chain one pass: pass 1 drafts, and the
    /// drafts read back are one a sequence.
    #[test]
    fn a_step_of_more_sequences_chains_fewer_passes() {
        let step = ThreeSequences::new();
        let passes = DRAFTER
            .passes(
                &step.rows(),
                &step.before(),
                &ids(&["a", "b", "c"]),
                &mut any,
            )
            .expect("passes");
        let drafting = passes.drafting.expect("a and c draft");
        assert_eq!(drafting.chain.passes(), 1);
        assert_eq!(drafting.chain.next.len(), 1);
        let two = TwoSequences::new();
        let passes = DRAFTER
            .passes(&two.rows(), &two.before(), &ids(&["a", "b"]), &mut any)
            .expect("passes");
        assert_eq!(passes.drafting.expect("a drafts").chain.passes(), 3);
    }

    /// A parallel-drafting head (P-EAGLE) drafts every token in pass 1, as vLLM's does: `a`'s
    /// four verify rows run on two more, reading the mask token at positions 14 and 15, so its
    /// six rows sit at 10..=15 whatever the verify kept — which of them read the target's tokens
    /// and which the placeholders is the device's to lay out — and attend through 15. No pass
    /// chains after it, and the drafts read back are three.
    #[test]
    fn a_parallel_head_drafts_every_token_in_pass_one() {
        const MASK: u32 = 201_020;
        let drafter = MtpDrafter {
            past_first: PastFirst::Parallel { mask_token: MASK },
            ..DRAFTER
        };
        let step = TwoSequences::new();
        let passes = drafter
            .passes(&step.rows(), &step.before(), &ids(&["a", "b"]), &mut any)
            .expect("passes");
        let Drafting { one, chain } = passes.drafting.expect("a drafts");
        // Block 3 of 16-token blocks: position `p`'s slot is `48 + p`.
        assert_eq!(one.positions, [10, 11, 12, 13, 14, 15]);
        assert_eq!(one.slot_mapping, [58, 59, 60, 61, 62, 63]);
        assert_eq!(one.input_ids, [0, 0, 0, 0, MASK, MASK]);
        assert_eq!((one.cu_seqlens_q, one.seqused_k), (vec![0, 6], vec![16]));
        assert_eq!((one.last_rows, one.has_spec_tokens), (vec![5], true));
        assert_eq!(one.token_rows, [(0, 0), (1, 1), (2, 2), (3, 3)]);
        assert_eq!(chain.seqs, [[0, 4, 0]]);
        assert_eq!((chain.passes(), chain.rows(), chain.made()), (1, 2, 3));
        assert_eq!(chain.next, [vec![0; 4 * 2 * 3]]);
        assert!(chain.input_ids.is_empty() && chain.positions.is_empty());
        assert_eq!(chain.cu_seqlens_q, [0]);
        let picked = [7, 8, 9];
        let produced = [vec![0, 1, 2, 3], Vec::new()];
        let drafts = drafter.drafts(&chain, &picked, &step.rows(), &produced, &step.after(3));
        assert_eq!(drafts["a"], picked);
    }

    fn ids(of: &[&str]) -> HashSet<String> {
        of.iter().map(|&id| id.to_string()).collect()
    }

    /// A target that keeps every request's draft state.
    fn any(_: &str) -> bool {
        true
    }

    /// A request the target cannot keep draft state for only primes: `a`'s verify rows run
    /// KV-only beside `b`'s prompt, nothing drafts, and only `a` — the one that would draft — is
    /// asked.
    #[test]
    fn a_request_without_draft_state_only_primes() {
        let step = TwoSequences::new();
        let mut asked = Vec::new();
        let mut refuse = |r: &str| {
            asked.push(r.to_string());
            false
        };
        let passes = DRAFTER
            .passes(&step.rows(), &step.before(), &ids(&["a", "b"]), &mut refuse)
            .expect("passes");
        assert_eq!(asked, ["a"]);
        assert_eq!(passes.drafting, None);
        let prime = passes.prime.expect("both prime");
        assert_eq!(prime.positions, [10, 11, 12, 13, 0, 1, 2]);
        assert_eq!(prime.cu_seqlens_q, [0, 4, 7]);
        assert_eq!(prime.seqused_k, [14, 3]);
    }

    /// Pass 1 runs every row of each planned sequence, split by whether a draft can read it: the
    /// rows whose next token the host has (`b`'s prompt) prime, for their KV alone; the rest
    /// (`a`'s verify rows, each of which can be its newest) draft, the target's tokens at them.
    /// Each sequence's rows read the target's hidden states at its own rows of the step. Only `a`
    /// gets a token, so only `a` chains; `b` alone primes its prompt, and nothing planned runs
    /// nothing.
    #[test]
    fn pass_one_primes_the_rows_no_draft_reads() {
        let step = TwoSequences::new();
        let history = step.before();
        let passes = |plan: &[&str]| DRAFTER.passes(&step.rows(), &history, &ids(plan), &mut any);
        let both = passes(&["a", "b"]).expect("passes");
        let prime = PassOne {
            input_ids: vec![701, 702, 703],
            positions: vec![0, 1, 2],
            slot_mapping: vec![80, 81, 82],
            cu_seqlens_q: vec![0, 3],
            seqused_k: vec![3],
            block_table: vec![5, 0],
            block_table_stride: 2,
            last_rows: vec![2],
            has_spec_tokens: false,
            token_rows: Vec::new(),
            hidden_rows: vec![(0, 4, 3)],
            steps: vec![1],
        };
        assert_eq!(both.prime, Some(prime));
        let one = PassOne {
            input_ids: vec![0, 0, 0, 0],
            positions: vec![10, 11, 12, 13],
            slot_mapping: vec![58, 59, 60, 61],
            cu_seqlens_q: vec![0, 4],
            seqused_k: vec![14],
            block_table: vec![3, 0],
            block_table_stride: 2,
            last_rows: vec![3],
            has_spec_tokens: true,
            token_rows: vec![(0, 0), (1, 1), (2, 2), (3, 3)],
            hidden_rows: vec![(0, 0, 4)],
            steps: vec![0],
        };
        let drafting = both.drafting.expect("a drafts");
        assert_eq!(drafting.one, one);
        assert_eq!(drafting.chain.seqs, [[0, 4, 0]]);
        assert_eq!(drafting.chain.num_seqs(), 1);

        let b = passes(&["b"]).expect("passes");
        assert_eq!(b.prime.expect("b primes").input_ids, [701, 702, 703]);
        assert_eq!(b.drafting, None);
        assert_eq!(passes(&[]), None);
    }

    /// The chunk that ends `b`'s prompt (its history three tokens): its rows before the newest
    /// prime, and the newest — whose next token is the target's — drafts beside `a`'s verify rows,
    /// attending the KV the prime wrote.
    #[test]
    fn a_prompt_s_newest_row_drafts_and_the_rest_prime() {
        let step = TwoSequences::new();
        let ends = |id: &str| match id {
            "b" => Some((700..703).collect()),
            _ => step.before()(id),
        };
        let passes = DRAFTER.passes(&step.rows(), &ends, &ids(&["a", "b"]), &mut any);
        let passes = passes.expect("passes");
        let prime = passes.prime.expect("b primes");
        assert_eq!(
            (prime.input_ids, prime.positions, prime.seqused_k),
            (vec![701, 702], vec![0, 1], vec![2])
        );
        assert_eq!((prime.hidden_rows, prime.steps), (vec![(0, 4, 2)], vec![1]));
        let Drafting { one, chain } = passes.drafting.expect("both draft");
        assert_eq!(one.positions, [10, 11, 12, 13, 2]);
        assert_eq!(one.slot_mapping, [58, 59, 60, 61, 82]);
        assert_eq!(
            (one.cu_seqlens_q, one.seqused_k),
            (vec![0, 4, 5], vec![14, 3])
        );
        assert_eq!(one.token_rows, [(0, 0), (1, 1), (2, 2), (3, 3), (4, 6)]);
        assert_eq!(one.hidden_rows, [(0, 0, 4), (4, 6, 1)]);
        assert_eq!(
            (one.block_table, one.last_rows),
            (vec![3, 0, 5, 0], vec![3, 4])
        );
        assert_eq!(chain.seqs, [[0, 4, 0], [6, 1, 4]]);
    }

    /// After pass `d`, the next pass's row at accepted count `a` is at the newest position, `10 +
    /// a`, plus `d`: its slot in `a`'s block 3, its used length one past it. The placeholders are
    /// the last pass's at the most accepted; the device compares each row's target token with the
    /// draft the row after it reads.
    #[test]
    fn the_chain_lays_out_every_accepted_count() {
        let step = TwoSequences::new();
        let passes = DRAFTER.passes(&step.rows(), &step.before(), &ids(&["a", "b"]), &mut any);
        let chain = passes.expect("passes").drafting.expect("a drafts").chain;
        let at = |d: u32| -> Vec<u32> {
            (0..=3)
                .flat_map(|a| [10 + a + d, 3 * 16 + 10 + a + d, 11 + a + d])
                .collect()
        };
        assert_eq!(chain.next, [at(1), at(2), vec![0; 12]]);
        assert_eq!(chain.drafted, [21, 22, 23, 700, 701, 702, 0]);
        assert_eq!(
            (chain.input_ids, chain.positions, chain.seqused_k),
            (vec![0], vec![15], vec![16])
        );
        assert_eq!(chain.slot_mapping, [u32::MAX]);
        assert_eq!(chain.cu_seqlens_q, [0, 1]);
        assert_eq!((chain.block_table, chain.last_rows), (vec![3, 0], vec![0]));
    }

    /// The drafts the device picked, each drafting sequence's `k`, from its newest row: the host
    /// knows it from the tokens the step committed, and logs it. Near the maximum model length the
    /// drafts that would land past it go — for every sequence, as one chained pass runs them all.
    #[test]
    fn drafts_are_the_device_s_up_to_the_model_length() {
        let step = TwoSequences::new();
        let passes = DRAFTER.passes(&step.rows(), &step.before(), &ids(&["a", "b"]), &mut any);
        let chain = passes.expect("passes").drafting.expect("a drafts").chain;
        let picked = [901, 1000, 2000];
        for kept in 0..4u32 {
            let produced = vec![(0..=kept).collect(), Vec::new()];
            let drafts =
                DRAFTER.drafts(&chain, &picked, &step.rows(), &produced, &step.after(kept));
            assert_eq!(
                drafts,
                HashMap::from([("a".into(), picked.to_vec())]),
                "kept {kept}"
            );
        }
        // Kept all three: the newest row is at 13, so pass 2 runs at 14 and its draft lands at 16.
        let near = |max_model_len| MtpDrafter {
            max_model_len,
            ..DRAFTER
        };
        let produced = vec![vec![0, 1, 2, 3], Vec::new()];
        let drafts = |max_model_len| {
            let d = near(max_model_len).drafts(
                &chain,
                &picked,
                &step.rows(),
                &produced,
                &step.after(3),
            );
            d["a"].clone()
        };
        assert_eq!(drafts(18), picked);
        assert_eq!(drafts(17), picked[..2]);
        assert_eq!(drafts(16), picked[..1]);
        assert_eq!(drafts(15), picked[..1], "draft 1 is pass 1's, never cut");
    }

    /// Two sequences' one draft each fits five verify rows; three sequences' does not.
    const GATED: MtpDrafter = MtpDrafter {
        budget: DraftBudget {
            drafts: 3,
            verify_rows: 5,
        },
        ..DRAFTER
    };

    /// A step past the gate (three sequences at a two-sequence head) primes every planned row: the
    /// head's layer reads the target's tokens and hidden states as ever, so its KV misses no row,
    /// but nothing drafts. Back within the gate the same two sequences chain again.
    #[test]
    fn a_step_past_the_gate_primes_every_row() {
        let step = ThreeSequences::new();
        let gated = GATED
            .passes(
                &step.rows(),
                &step.before(),
                &ids(&["a", "b", "c"]),
                &mut any,
            )
            .expect("passes");
        assert_eq!(gated.drafting, None);
        assert_eq!(
            gated.prime,
            Some(PassOne {
                input_ids: vec![0, 0, 0, 0, 701, 702, 703, 0],
                positions: vec![10, 11, 12, 13, 0, 1, 2, 5],
                slot_mapping: vec![58, 59, 60, 61, 80, 81, 82, 117],
                cu_seqlens_q: vec![0, 4, 7, 8],
                seqused_k: vec![14, 3, 6],
                block_table: vec![3, 0, 5, 0, 7, 0],
                block_table_stride: 2,
                last_rows: vec![3, 6, 7],
                has_spec_tokens: false,
                token_rows: vec![(0, 0), (1, 1), (2, 2), (3, 3), (7, 7)],
                hidden_rows: vec![(0, 0, 4), (4, 4, 3), (7, 7, 1)],
                steps: vec![0, 1, 2],
            })
        );

        let two = TwoSequences::new();
        let within = GATED
            .passes(&two.rows(), &two.before(), &ids(&["a", "b"]), &mut any)
            .expect("passes");
        within.drafting.expect("a drafts");
    }

    /// A head reading its target's KV (Gemma 4's assistant) keeps no KV of its own, so a row no
    /// draft reads runs nothing: past the gate no pass runs, and within it `b`'s prompt rows drop
    /// out. Pass 1 queries one past each row it runs and writes no slot.
    #[test]
    fn a_head_reading_its_target_kv_runs_only_the_rows_a_draft_reads() {
        let drafter = MtpDrafter {
            reads_target_kv: true,
            ..GATED
        };
        let three = ThreeSequences::new();
        let gated = drafter.passes(
            &three.rows(),
            &three.before(),
            &ids(&["a", "b", "c"]),
            &mut any,
        );
        assert_eq!(gated, None);

        let two = TwoSequences::new();
        let within = (drafter.passes(&two.rows(), &two.before(), &ids(&["a", "b"]), &mut any))
            .expect("passes");
        assert_eq!(within.prime, None);
        let one = within.drafting.expect("a drafts").one;
        assert_eq!(one.positions, [11, 12, 13, 14]);
        assert!(one.slot_mapping.iter().all(|&s| s == u32::MAX));
        assert_eq!(
            drafter.passes(&two.rows(), &two.before(), &ids(&["b"]), &mut any),
            None
        );
    }

    /// A scheduled step of `(request, num_computed_tokens)`: one at 0 runs from its first position.
    fn scheduled(reqs: &[(&str, u32)]) -> SchedulerOutput {
        let mut s = SchedulerOutput::make_empty();
        for &(id, computed) in reqs {
            s.num_scheduled_tokens.insert(id.into(), 1);
            s.scheduled_cached_reqs.req_ids.push(id.into());
            s.scheduled_cached_reqs.num_computed_tokens.push(computed);
        }
        s
    }

    /// A step of more sequences than the head drafts for plans every request `takes_drafts`
    /// admits — the worker runs its pass 1 KV-only (module docs) — and once a step is back
    /// within the gate they chain again: no request goes stale. A request whose sampling reads
    /// its history takes no drafts. The proposer hands back the worker's drafts.
    #[test]
    fn a_step_past_the_head_s_sequences_plans_all_of_them() {
        let takes = |_: &str| true;
        let mut proposer = proposer();
        assert_eq!(proposer.max_seqs(), Some(2));
        assert_eq!(
            proposer.plan(&scheduled(&[("a", 10), ("b", 20), ("c", 30)]), &takes),
            ids(&["a", "b", "c"])
        );

        // Back within the gate: both planned again — neither went stale.
        assert_eq!(
            proposer.plan(&scheduled(&[("a", 11), ("b", 21)]), &takes),
            ids(&["a", "b"])
        );
        let not_b = |id: &str| id != "b";
        assert_eq!(
            proposer.plan(&scheduled(&[("a", 1), ("b", 0)]), &not_b),
            ids(&["a"])
        );

        let drafts = HashMap::from([("a".to_string(), vec![7, 8])]);
        let none = |_: &str| None;
        let mut ctx = ProposerStepCtx {
            scheduled_req_ids: &[],
            get_all_tokens: &none,
            worker_drafts: Some(&drafts),
            backend: None,
            draft_seed: None,
            sampled_token_ids: None,
            takes_drafts: &takes,
        };
        assert_eq!(proposer.propose_for_step(&mut ctx), drafts);
    }
}

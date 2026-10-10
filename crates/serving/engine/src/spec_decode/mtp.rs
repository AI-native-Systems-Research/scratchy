// SPDX-License-Identifier: Apache-2.0

//! Multi-token-prediction (MTP) proposer: drafts with a head that ships beside its target model
//! (Qwen3.5/3.6 `mtp.*`; an MLX drafter repo carries it alone) and reads the target's final
//! (post-norm) hidden states.
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
//! 2. **Passes 2..=k** ([`Chain`]), one row per drafting sequence: the previous draft at the next
//!    position, with the head's own final hidden state from the pass before. They run in the same
//!    command buffer: which row is a sequence's newest depends on the drafts the target accepted,
//!    so the device picks each pass's inputs — the token and hidden state at that row, and the
//!    position, slot and used KV length the host laid out for each accepted count. The host reads
//!    back the drafts alone.
//!
//! The head's KV pool has the target's blocks, so it shares the target's block table. Only a
//! sequence that produced a token this step drafts; an intermediate prefill chunk still runs its
//! rows through pass 1, priming the head's KV for the prompt.
//!
//! The worker that ran the step runs the passes ([`MtpDrafter`]) for the requests the engine
//! planned when it scheduled the step ([`MtpProposer`]'s `plan`), and returns their drafts with
//! the step's output — so the engine schedules asynchronously, waiting only on a step that drafts
//! or verifies. A step of more than the head's `max_seqs` sequences (its compiled
//! `spec_max_seqs`) plans none: it costs what its target alone does. Its sequences' head KV then
//! misses the step's rows, so they draft no more (`stale`) until a recompute runs them from their
//! first position.
//!
//! # TODO
//!
//! Measured on Qwen3.6-35B-A3B (`mlx-community/Qwen3.6-35B-A3B-4bit` with its MLX head), the one
//! target with a head config; "sampled" is the model's default sampling, "greedy" temperature 0.
//!
//! **The gate** (`spec_max_seqs`, the head's `arch.json`):
//!
//! - TODO: derive the gate per device, at expansion. It is 1 sequence on every GPU, the one point
//!   measured to pay on both GPUs measured: one request, base M5 output tok/s 1.36-1.38x sampled;
//!   M5 Max TPOT 1.14x sampled, 1.28x greedy (an earlier build of this head, at
//!   `--max-num-seqs 1`). Past it, where drafting stops paying depends on the GPU: on the base M5,
//!   before the gate came down to 1, sampled output tok/s was 1.07-1.18x at 2 sequences and
//!   1.03-1.17x at 4, greedy 0.96-1.06x at 4 (two rounds each); the M5 Max was not measured past
//!   one request. A verify step
//!   of `B` sequences runs `B * (k + 1)` rows, and a MoE target reads every distinct expert those
//!   rows pick (Qwen3.6: 256 experts, 8 a row; at 8 sequences the verify step's 24 rows pick about
//!   135, the plain step's 8 about 57, under uniform routing). So on the base M5 a verify step
//!   costs 1.6 / 2.0 / 2.2 / 2.65 plain steps at 1 / 2 / 4 / 8 sequences, against about 2.3 tokens
//!   a verify step. The gate wanted is the largest `B` whose verify step and head passes cost less
//!   than the plain steps of the tokens they produce. That is a comparison of two compiled tapes'
//!   costs on a device, so it belongs at expansion: one gate per declared device profile
//!   (`MetalTargetProfile`: bandwidth, fp16 peak), the worker picking its device's as it picks
//!   kernels by GPU generation. Blocked on a metal tape cost model: a roofline from the profile
//!   alone predicts 2.0x at 8 sequences against the measured 2.65x.
//! - TODO: key the gate on the target, not the head arch. `spec_max_seqs` is the head arch's bound
//!   default, so every target of `qwen3-5-mtp` shares it, but its cost is the target's: a dense
//!   target's verify rows cost far less than a MoE's (Llama-3.2-3B: a 4-row step costs 1.27x a
//!   1-row step), so a dense Qwen3.5/3.6 target would pay at more sequences.
//! - TODO: let stale requests draft again. A request that shares one step past the gate drafts no
//!   more for the rest of its life, even once it runs alone again: at a gate of 1, once a second
//!   request arrives (its prompt's step counts), neither drafts again. Two ways back: run pass 1
//!   KV-only on a gated step (the head's layer over the step's rows; no lm_head, no drafts), so no
//!   request goes stale, at the head's per-row cost on every step past the gate; or keep a stale
//!   request's target hidden rows (4 KiB a token at Qwen3.6's 2048-wide 16-bit hidden) and run them
//!   through pass 1 once its steps are back under the gate.
//! - TODO: a prefix-cache hit can reuse head KV that was never written. The head's KV lives in the
//!   target's blocks, so a request that hits cached blocks reuses their head KV too, and blocks
//!   written while their request was stale hold none: a request drafting over them gets fewer
//!   drafts accepted (the target verifies every draft, so the output is unaffected). Not measured.
//!   A KV-only pass 1 on gated steps closes it; a catch-up does not for blocks cached before it
//!   runs.
//!
//! **Cost:**
//!
//! - TODO: a verify step's cost past one sequence, which is what holds the gate down. Base M5, 8
//!   sequences: a plain step is 38.4 ms and a verify step yields about 2.3 tokens, so the verify
//!   step must fit in about 87 ms to break even. The experts take about 60 ms of it, the
//!   memory-bandwidth floor for the experts 24 rows pick (uniform routing), so the rest must fit in
//!   about 27 ms; it takes about 44 (dense projections 25 ms against about 13 at 8 rows; the
//!   Gated-DeltaNet scan 10-12 ms, each sequence's rows and the kept rows it replays one after
//!   another; attention 3; the rest about 6) plus the head's about 5, measured with 16- and
//!   32-row verify-sized rungs. Those rungs cost short prompts their first token, and the gate of
//!   1 keeps verify steps at 3 rows, so they are gone (`METAL_VERIFY_ROWS`, 8): past 2 sequences
//!   a verify step runs the 64-row prefill-shaped tape until the rungs derive from the gate.
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
//! - TODO: time to first token pays the head's pass over the prompt (about 0.19 s on a 5.4k-token
//!   prompt, base M5): pass 1 runs in the prompt step's command buffer, so its first token waits
//!   for the head's pass over every prompt row. The next step needs the drafts, the client does
//!   not: return the step's tokens before the head runs.
//! - TODO: the next step is scheduled once the host has read this one's tokens and drafts (#240:
//!   pipelined cycles). Its inputs are on the device already.
//! - TODO: choose `k` per bucket. `spec_drafts` (2) is one constant for every bucket, because the
//!   target's verify rows a sequence and its Gated-DeltaNet record areas are baked from it; a
//!   bucket-specific `k` needs both sized to the largest. Greedy tokens a verify step at
//!   k = 1 / 2 / 3 were 1.77 / 2.20 / 2.36 against the MLX oracle's 1.80 / 2.43 / 2.82, measured
//!   before the fold fix that keeps an exported final hidden written (after a verify step that
//!   kept no draft, the second draft matched the oracle's 5 times in 23 before it, 28 in 28
//!   after); k = 3 is not re-measured since.
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
//! - TODO: `--num-speculative-tokens` given with a head is ignored (the head drafts its compiled
//!   count), and `serve`'s flag defaults to 2, so it cannot tell a given value from none. Make it
//!   optional, and refuse a value that differs from the head's.
//! - TODO: the metal target grows, against CLAUDE.md's no net growth in target crates: `src/` +697
//!   lines net against main, 47 of them in `op_abi.rs`; shaders +251; the metal compiler +50.

use std::collections::{HashMap, HashSet};

use scratchy_serving_scheduler::scheduler::output::SchedulerOutput;

use super::backend::ForwardArgmaxRequest;
use super::proposer::{Proposer, ProposerStepCtx};

/// Configuration for the MTP proposer, held by [`super::ProposerConfig::Mtp`].
#[derive(Debug, Clone)]
pub struct MtpProposerConfig {
    /// Local path or HuggingFace repo ID of the head's checkpoint.
    pub model: String,
    /// Drafts per step (`k`): one pass-1 row plus `k - 1` chained passes.
    pub num_speculative_tokens: usize,
    /// The most sequences a step drafts for (module docs).
    pub max_seqs: usize,
    /// Maximum model context length: no draft lands past it.
    pub max_model_len: usize,
}

/// The MTP proposer (module docs).
#[derive(Debug)]
pub struct MtpProposer {
    config: MtpProposerConfig,
    /// Requests whose head KV missed a step's rows (module docs): they draft no more.
    stale: HashSet<String>,
}

impl MtpProposer {
    pub fn new(config: MtpProposerConfig) -> Self {
        Self {
            config,
            stale: HashSet::new(),
        }
    }

    pub fn config(&self) -> &MtpProposerConfig {
        &self.config
    }
}

/// The head's passes over one step (module docs), run by the worker that ran it.
#[derive(Clone, Copy, Debug)]
pub struct MtpDrafter {
    /// Drafts per step (`k`, the head's compiled `spec_drafts`).
    pub drafts: usize,
    /// Maximum model context length: no draft lands past it.
    pub max_model_len: usize,
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
    pub one: PassOne,
    /// `None` when no planned sequence gets a token from the step (prefill chunks short of their
    /// prompt's end): pass 1 primes their head KV, and nothing drafts.
    pub chain: Option<Chain>,
}

/// The head's pass 1 over a step: the rows' host inputs, and what the device copies from the
/// target.
#[derive(Debug, PartialEq)]
pub struct PassOne {
    /// Each planned sequence's rows of the step, in order. A row's input token is the one after its
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
}

/// Passes 2..=k, one row per drafting sequence — one the step gives a token — chained on the
/// device after pass 1 (module docs): their host inputs, and the tables the device picks the rest
/// from by the number of drafts the target accepted.
#[derive(Debug, PartialEq)]
pub struct Chain {
    /// Per drafting sequence: its first token row in the step, its token rows, and its first token
    /// row in pass 1's rows.
    pub seqs: Vec<[u32; 3]>,
    /// Per step row, the token the row after it reads — the draft it verifies within a sequence.
    pub drafted: Vec<u32>,
    /// After each pass 1..=k: per drafting sequence and accepted count `0..=k`, the next pass's
    /// position, slot and used KV length (zeros after the last pass, which has none).
    pub next: Vec<Vec<u32>>,
    /// Each chained pass's host inputs, one row per drafting sequence. The token, position, slot
    /// and used KV length are placeholders the device overwrites: the position and used length the
    /// largest the last pass takes, the slot none.
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
}

impl PassOne {
    /// The forward's host-side request (its target hidden states come from the device).
    pub fn request(&self) -> ForwardArgmaxRequest<'_> {
        request(
            [&self.input_ids, &self.positions, &self.slot_mapping],
            [&self.cu_seqlens_q, &self.seqused_k, &self.block_table],
            self.block_table_stride,
            &self.last_rows,
            self.has_spec_tokens,
        )
    }
}

impl Chain {
    /// A chained pass's host-side request (its target hidden states come from the device).
    pub fn request(&self) -> ForwardArgmaxRequest<'_> {
        request(
            [&self.input_ids, &self.positions, &self.slot_mapping],
            [&self.cu_seqlens_q, &self.seqused_k, &self.block_table],
            self.block_table_stride,
            &self.last_rows,
            false,
        )
    }

    /// The drafting sequences.
    pub fn num_seqs(&self) -> usize {
        self.steps.len()
    }
}

/// A head forward's request over `rows` (its tokens, positions and slots) and `seqs` (its
/// cumulative sequence lengths, used KV lengths and block table rows).
fn request<'a>(
    [input_ids, positions, slot_mapping]: [&'a [u32]; 3],
    [cu_seqlens_q, seqused_k, block_table]: [&'a [u32]; 3],
    block_table_stride: usize,
    last_rows: &'a [u32],
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
        last_token_indices: Some(last_rows),
        target_hidden: None,
    }
}

impl MtpDrafter {
    /// The head's passes over the rows of `step` that the requests in `plan` run, before the step:
    /// `history` reads each request's tokens so far. `None` when none is planned.
    pub fn passes(
        &self,
        step: &StepRows<'_>,
        history: &dyn Fn(&str) -> Option<Vec<u32>>,
        plan: &HashSet<String>,
    ) -> Option<HeadPasses> {
        let mut one = PassOne {
            input_ids: Vec::new(),
            positions: Vec::new(),
            slot_mapping: Vec::new(),
            cu_seqlens_q: vec![0],
            seqused_k: Vec::new(),
            block_table: Vec::new(),
            block_table_stride: step.block_table_stride,
            last_rows: Vec::new(),
            has_spec_tokens: false,
            token_rows: Vec::new(),
            hidden_rows: Vec::new(),
        };
        // Per drafting sequence: its index in the step, its token rows' span of `token_rows`.
        let mut drafting: Vec<(usize, std::ops::Range<usize>)> = Vec::new();
        let included: Vec<usize> = (0..step.req_ids.len())
            .filter(|&i| plan.contains(&step.req_ids[i]))
            .collect();
        for &i in &included {
            let tokens = history(&step.req_ids[i]).unwrap_or_default();
            let rows = step.cu_seqlens_q[i] as usize..step.cu_seqlens_q[i + 1] as usize;
            let (start, first_token) = (one.input_ids.len(), one.token_rows.len());
            for r in rows.clone() {
                let p = step.positions[r] as usize;
                let row = one.input_ids.len() as u32;
                let token = tokens.get(p + 1).copied().unwrap_or_else(|| {
                    one.token_rows.push((row, r as u32));
                    0
                });
                one.input_ids.push(token);
                one.positions.push(step.positions[r]);
                one.slot_mapping.push(step.slot_mapping[r]);
            }
            let (row, step_row) = (start as u32, rows.start as u32);
            one.hidden_rows.push((row, step_row, rows.len() as u32));
            one.cu_seqlens_q.push(one.input_ids.len() as u32);
            one.seqused_k.push(step.seqused_k[i]);
            one.last_rows.push(one.input_ids.len() as u32 - 1);
            if one.token_rows.len() > first_token {
                drafting.push((i, first_token..one.token_rows.len()));
            }
        }
        if included.is_empty() {
            return None;
        }
        one.has_spec_tokens = drafting.iter().any(|(_, tokens)| tokens.len() > 1);
        one.block_table = step.block_rows(&included);
        let chain = (!drafting.is_empty()).then(|| self.chain(step, &one, &drafting));
        Some(HeadPasses { one, chain })
    }

    /// Passes 2..=k after `one` for the `drafting` sequences (each its index in the step and its
    /// span of `one`'s token rows).
    fn chain(
        &self,
        step: &StepRows<'_>,
        one: &PassOne,
        drafting: &[(usize, std::ops::Range<usize>)],
    ) -> Chain {
        let k = self.drafts;
        let n = drafting.len();
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
        // After pass `d`, the next pass's row at accepted count `a` sits at `first + a + d`, the
        // newest row's position plus `d`.
        let next = (1..=k)
            .map(|d| {
                (drafting.iter().zip(&seqs))
                    .flat_map(|(&(i, _), &[step_row, ..])| {
                        let first = step.positions[step_row as usize] as usize;
                        (0..=k).flat_map(move |a| match d < k {
                            true => {
                                let p = first + a + d;
                                [p as u32, step.slot(i, p), p as u32 + 1]
                            }
                            false => [0; 3],
                        })
                    })
                    .collect()
            })
            .collect();
        // Placeholders: the last pass's position at the most drafts accepted.
        let last: Vec<u32> = (seqs.iter())
            .map(|&[step_row, rows, _]| step.positions[step_row as usize] + rows - 1 + k as u32 - 1)
            .collect();
        Chain {
            seqs,
            drafted,
            next,
            input_ids: vec![0; n],
            seqused_k: last.iter().map(|&p| p + 1).collect(),
            positions: last,
            slot_mapping: vec![u32::MAX; n],
            cu_seqlens_q: (0..=n as u32).collect(),
            block_table: step.block_rows(&steps),
            block_table_stride: step.block_table_stride,
            last_rows: (0..n as u32).collect(),
            steps,
        }
    }

    /// The drafts after `chain` ran: `picked`, its `k` drafts per drafting sequence, read back,
    /// for the requests of `step` that produced `produced` (per request, in step order) and whose
    /// token histories `history` reads, the step's tokens appended. No draft lands past the
    /// maximum model length.
    pub fn drafts(
        &self,
        chain: &Chain,
        picked: &[u32],
        step: &StepRows<'_>,
        produced: &[Vec<u32>],
        history: &dyn Fn(&str) -> Option<Vec<u32>>,
    ) -> HashMap<String, Vec<u32>> {
        let k = self.drafts;
        // Each sequence's newest row: the position before its newest token.
        let newest: Vec<Option<usize>> = (chain.steps.iter())
            .map(|&i| {
                let tokens = history(&step.req_ids[i]).unwrap_or_default().len();
                tokens.checked_sub(2).filter(|_| !produced[i].is_empty())
            })
            .collect();
        // Pass `d + 1` runs at the newest position plus `d`, and its draft lands one past it.
        let fits = |d: usize| {
            newest
                .iter()
                .flatten()
                .all(|&p| p + d + 2 < self.max_model_len)
        };
        let depth = 1 + (1..k).take_while(|&d| fits(d)).count();
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
        // A request gone drafts no more; one run from its first position has its head KV whole.
        self.stale.retain(|r| takes_drafts(r));
        let cached = &sched.scheduled_cached_reqs;
        let from_start = (sched.scheduled_new_reqs.iter())
            .filter(|r| r.num_computed_tokens == 0)
            .map(|r| &r.req_id)
            .chain(
                (cached.req_ids.iter().zip(&cached.num_computed_tokens))
                    .filter(|&(_, &n)| n == 0)
                    .map(|(r, _)| r),
            );
        for r in from_start {
            self.stale.remove(r);
        }
        let scheduled = sched
            .num_scheduled_tokens
            .keys()
            .filter(|r| takes_drafts(r));
        // A step of more sequences than the head drafts for runs as the target alone.
        // TODO: its requests go stale for good (module docs: two ways back).
        if sched.num_scheduled_tokens.len() > self.config.max_seqs {
            self.stale.extend(scheduled.cloned());
            return HashSet::new();
        }
        scheduled
            .filter(|r| !self.stale.contains(*r))
            .cloned()
            .collect()
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

    fn proposer(max_seqs: usize) -> MtpProposer {
        MtpProposer::new(MtpProposerConfig {
            model: "head".into(),
            num_speculative_tokens: 3,
            max_seqs,
            max_model_len: 4096,
        })
    }

    const DRAFTER: MtpDrafter = MtpDrafter {
        drafts: 3,
        max_model_len: 4096,
    };

    fn ids(of: &[&str]) -> HashSet<String> {
        of.iter().map(|&id| id.to_string()).collect()
    }

    /// Pass 1 runs every row of each planned sequence: the host's tokens where its history has
    /// them (`b`'s prompt), the target's where it does not (`a`'s verify rows); each sequence's
    /// rows read the target's hidden states at its own rows of the step. Only `a` gets a token, so
    /// only `a` chains; `b` alone primes its prompt, and nothing planned runs nothing.
    #[test]
    fn pass_one_runs_every_row_of_the_planned_sequences() {
        let step = TwoSequences::new();
        let history = step.before();
        let passes = |plan: &[&str]| DRAFTER.passes(&step.rows(), &history, &ids(plan));
        let both = passes(&["a", "b"]).expect("passes");
        let one = PassOne {
            input_ids: vec![0, 0, 0, 0, 701, 702, 703],
            positions: vec![10, 11, 12, 13, 0, 1, 2],
            slot_mapping: vec![58, 59, 60, 61, 80, 81, 82],
            cu_seqlens_q: vec![0, 4, 7],
            seqused_k: vec![14, 3],
            block_table: vec![3, 0, 5, 0],
            block_table_stride: 2,
            last_rows: vec![3, 6],
            has_spec_tokens: true,
            token_rows: vec![(0, 0), (1, 1), (2, 2), (3, 3)],
            hidden_rows: vec![(0, 0, 4), (4, 4, 3)],
        };
        assert_eq!(both.one, one);
        let chain = both.chain.expect("a chains");
        assert_eq!(chain.seqs, [[0, 4, 0]]);
        assert_eq!(chain.num_seqs(), 1);

        let b = passes(&["b"]).expect("passes");
        assert_eq!(b.one.input_ids, [701, 702, 703]);
        assert_eq!((b.one.token_rows.len(), b.one.has_spec_tokens), (0, false));
        assert_eq!(b.chain, None);
        assert_eq!(passes(&[]), None);
    }

    /// After pass `d`, the next pass's row at accepted count `a` is at the newest position, `10 +
    /// a`, plus `d`: its slot in `a`'s block 3, its used length one past it. The placeholders are
    /// the last pass's at the most accepted; the device compares each row's target token with the
    /// draft the row after it reads.
    #[test]
    fn the_chain_lays_out_every_accepted_count() {
        let step = TwoSequences::new();
        let passes = DRAFTER.passes(&step.rows(), &step.before(), &ids(&["a", "b"]));
        let chain = passes.expect("passes").chain.expect("a chains");
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
        let passes = DRAFTER.passes(&step.rows(), &step.before(), &ids(&["a", "b"]));
        let chain = passes.expect("passes").chain.expect("a chains");
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
            drafts: 3,
            max_model_len,
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

    /// A step of more sequences than the head drafts for plans none and stales them — their head
    /// KV misses its rows — until a recompute runs one from its first position. A request whose
    /// sampling reads its history takes no drafts. The proposer hands back the worker's drafts.
    #[test]
    fn a_step_past_the_head_s_sequences_plans_none_and_stales_them() {
        let takes = |_: &str| true;
        let mut proposer = proposer(1);
        assert!(
            proposer
                .plan(&scheduled(&[("a", 10), ("b", 20)]), &takes)
                .is_empty()
        );

        // Within its sequences again: both stale.
        proposer.config.max_seqs = 2;
        assert!(
            proposer
                .plan(&scheduled(&[("a", 11), ("b", 21)]), &takes)
                .is_empty()
        );

        // `a` recomputed from its first position: whole again.
        let plan = proposer.plan(&scheduled(&[("a", 0), ("b", 22)]), &takes);
        assert_eq!(plan, ids(&["a"]));
        let not_b = |id: &str| id != "b";
        let plan = proposer.plan(&scheduled(&[("a", 1), ("b", 0)]), &not_b);
        assert_eq!(plan, ids(&["a"]));

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

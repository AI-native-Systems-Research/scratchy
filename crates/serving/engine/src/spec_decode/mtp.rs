// SPDX-License-Identifier: Apache-2.0

//! Multi-token-prediction (MTP) proposer: drafts with a head that ships beside its target model
//! (Qwen3.5/3.6 `mtp.*`; an MLX drafter repo carries it alone) and reads the target's final
//! (post-norm) hidden states.
//!
//! Per step, after the target ran (prefill, decode or verify) and its tokens were appended:
//!
//! 1. **Pass 1 over the step's rows**, through each sequence's newest. Row `r`, at position `p`,
//!    feeds the head the target's hidden state at `r` and the token at `p + 1` — read from the
//!    request's history, so a prompt row gets the next prompt token, an accepted draft its
//!    successor, and the newest row the token the target just produced. The rows past it held
//!    rejected drafts: they keep no token and do not run (the head's KV at their positions is
//!    rewritten when they next do), so the newest row is its sequence's last — the row the lm_head
//!    samples. Every row writes the head's KV at the target's slot for `p`, so the head attends
//!    the whole history. The newest row's argmax is draft 1.
//! 2. **Passes 2..=k**, one row per drafting sequence: the previous draft at the next position,
//!    with the head's own final hidden state from the pass before.
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
//! - TODO: a build with a head pays its Gated-DeltaNet scan's replay at prefill. The scan's
//!   per-token loop also replays a verify step's kept rows, one loop body so the replay is
//!   bit-exact (`gdn_scan_simd_resumes_from_checkpoint`), which costs every prompt token: 0.207 ms
//!   against 0.167 ms a token across the 30 scans (base M5, 1024- and 2048-row prompt chunks).
//!   Without a head the replay compiles out (`GDN_SCAN_DRAFTS` 0). Two loops calling one inlined
//!   row lost the bit-exactness (the varlen scan's replay 1 ULP off): the compiler contracts each
//!   copy apart.
//! - TODO: time to first token pays the head's pass over the prompt (about 0.19 s on a 5.4k-token
//!   prompt, base M5): the worker returns a step's tokens with its drafts, so the first token waits
//!   for pass 1 over every prompt row. The next step needs the drafts, the client does not: return
//!   the step's tokens before the head runs, or run the head inside the target's submission (#240:
//!   a speculative step as one compiled tape).
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
//! - TODO: on a 32 GB Mac Qwen3.6 does not start at the default 128 sequences, with or without a
//!   head (the Gated-DeltaNet state pool alone is 8 GiB; main too), so base-M5 numbers here are
//!   development runs at `--max-num-seqs 8`. Sizing the state pool is its own fix. There, the
//!   head's memory (weights+overhead 19.8 -> 20.4 GiB) also prunes the prefill bucket 2048 ->
//!   1024: a 5.6k-token prompt's first token 5.1 -> 6.1 s.
//! - TODO: `--num-speculative-tokens` given with a head is ignored (the head drafts its compiled
//!   count), and `serve`'s flag defaults to 2, so it cannot tell a given value from none. Make it
//!   optional, and refuse a value that differs from the head's.
//! - TODO: the metal target grows, against CLAUDE.md's no net growth in target crates: `src/` +413
//!   lines net against main, 47 of them in `op_abi.rs`; shaders +226; the metal compiler +49.

use std::collections::{HashMap, HashSet};

use scratchy_serving_scheduler::scheduler::output::SchedulerOutput;

use super::backend::{
    BackendError, ForwardArgmaxRequest, KvPoolHandle, ModelHandle, SpecDecodeBackend,
};
use super::proposer::{DraftSeedInputs, Proposer, ProposerStepCtx};

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

/// The head is the backend's secondary model, with the secondary KV pool.
const HEAD: ModelHandle = ModelHandle(1);
const HEAD_KV: KvPoolHandle = KvPoolHandle(1);

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

impl MtpDrafter {
    /// The drafts of the step `seed` describes for the requests in `plan`, which produced
    /// `produced` (per request, in `seed` order) and whose token histories `history` reads.
    pub fn draft(
        &self,
        seed: &DraftSeedInputs,
        produced: &[Vec<u32>],
        history: &dyn Fn(&str) -> Option<Vec<u32>>,
        backend: &mut dyn SpecDecodeBackend,
        plan: &HashSet<String>,
    ) -> Result<HashMap<String, Vec<u32>>, BackendError> {
        let num_reqs = seed.req_ids.len();
        let row_bytes = seed.target_hidden.len() / seed.num_tokens.max(1);
        let included: Vec<usize> = (0..num_reqs)
            .filter(|&i| plan.contains(&seed.req_ids[i]))
            .collect();
        if included.is_empty() {
            return Ok(HashMap::new());
        }
        // Rows `is` of the step's block table.
        let stride = seed.block_table_stride;
        let block_rows = |is: &[usize]| -> Vec<u32> {
            is.iter()
                .flat_map(|&i| {
                    seed.block_table[i * stride..(i + 1) * stride]
                        .iter()
                        .copied()
                })
                .collect()
        };

        // Pass 1: each sequence's rows through its newest, each row with its successor, and the
        // position of each drafting sequence's newest row.
        let mut input_ids = Vec::with_capacity(seed.num_tokens);
        let mut positions = Vec::with_capacity(seed.num_tokens);
        let mut slot_mapping = Vec::with_capacity(seed.num_tokens);
        let mut target_hidden = Vec::with_capacity(seed.target_hidden.len());
        let (mut cu_seqlens_q, mut seqused_k) = (vec![0u32], Vec::with_capacity(num_reqs));
        let mut newest: Vec<Option<usize>> = vec![None; num_reqs];
        for &i in &included {
            let req_id = &seed.req_ids[i];
            let tokens = history(req_id).unwrap_or_default();
            let rows = seed.cu_seqlens_q[i] as usize..seed.cu_seqlens_q[i + 1] as usize;
            // The newest token sits at `tokens.len() - 1`, produced by the row at the position
            // before it.
            let newest_pos = tokens
                .len()
                .checked_sub(2)
                .filter(|_| !produced[i].is_empty());
            let found = rows
                .clone()
                .find(|&r| Some(seed.positions[r] as usize) == newest_pos);
            newest[i] = found.and(newest_pos);
            let end = found.map_or(rows.end, |r| r + 1);
            for row in rows.start..end {
                let p = seed.positions[row] as usize;
                input_ids.push(tokens.get(p + 1).copied().unwrap_or(0));
                positions.push(seed.positions[row]);
                slot_mapping.push(seed.slot_mapping[row]);
                target_hidden
                    .extend_from_slice(&seed.target_hidden[row * row_bytes..][..row_bytes]);
            }
            cu_seqlens_q.push(input_ids.len() as u32);
            seqused_k.push(seed.seqused_k[i] - (rows.end - end) as u32);
        }
        // Each included sequence's last row, in `included` order.
        let lm_rows: Vec<u32> = cu_seqlens_q[1..].iter().map(|&end| end - 1).collect();
        let drafting: Vec<usize> = included
            .iter()
            .copied()
            .filter(|&i| newest[i].is_some())
            .collect();
        let hidden_rows: Vec<u32> = (included.iter().zip(&lm_rows))
            .filter(|&(&i, _)| newest[i].is_some())
            .map(|(_, &r)| r)
            .collect();
        let pass1_blocks = block_rows(&included);
        let pass1 = ForwardArgmaxRequest {
            input_ids: &input_ids,
            positions: &positions,
            slot_mapping: &slot_mapping,
            cu_seqlens_q: &cu_seqlens_q,
            seqused_k: &seqused_k,
            block_table: &pass1_blocks,
            span_ids: None,
            sliding_slot_mappings: &[],
            sliding_block_tables: &[],
            block_table_stride: seed.block_table_stride,
            max_seqlen_q: cu_seqlens_q
                .windows(2)
                .map(|w| (w[1] - w[0]) as usize)
                .max()
                .unwrap_or(0),
            max_seqlen_k: seqused_k.iter().copied().max().unwrap_or(0) as usize,
            num_tokens: input_ids.len(),
            has_spec_tokens: false,
            last_token_indices: Some(&lm_rows),
            target_hidden: Some(&target_hidden),
        };
        let (argmax, mut hidden) =
            backend.forward_argmax_hidden_blocking(HEAD, HEAD_KV, &pass1, &hidden_rows)?;
        let mut drafts: Vec<Vec<u32>> = hidden_rows
            .iter()
            .map(|&r| vec![argmax[r as usize]])
            .collect();

        // Passes 2..=k: one row per drafting sequence, at the position after the last.
        let n = drafting.len();
        let rows: Vec<u32> = (0..n as u32).collect();
        let cu_seqlens: Vec<u32> = (0..=n as u32).collect();
        let block_table = block_rows(&drafting);
        for depth in 1..self.drafts {
            let positions: Vec<usize> = drafting
                .iter()
                .map(|&i| newest[i].expect("drafting sequences have a newest row") + depth)
                .collect();
            // The draft this pass makes lands at `position + 2`.
            if n == 0 || positions.iter().any(|&p| p + 2 >= self.max_model_len) {
                break;
            }
            let slot_mapping: Vec<u32> = drafting
                .iter()
                .zip(&positions)
                .map(|(&i, &p)| {
                    seed.block_ids[i]
                        .get(p / seed.block_size)
                        .map_or(u32::MAX, |&b| {
                            (b as usize * seed.block_size + p % seed.block_size) as u32
                        })
                })
                .collect();
            let seqused_k: Vec<u32> = positions.iter().map(|&p| p as u32 + 1).collect();
            let positions: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
            let input_ids: Vec<u32> = drafts.iter().map(|d| *d.last().expect("draft 1")).collect();
            let req = ForwardArgmaxRequest {
                input_ids: &input_ids,
                positions: &positions,
                slot_mapping: &slot_mapping,
                cu_seqlens_q: &cu_seqlens,
                seqused_k: &seqused_k,
                block_table: &block_table,
                span_ids: None,
                sliding_slot_mappings: &[],
                sliding_block_tables: &[],
                block_table_stride: stride,
                max_seqlen_q: 1,
                max_seqlen_k: seqused_k.iter().copied().max().unwrap_or(0) as usize,
                num_tokens: n,
                has_spec_tokens: false,
                last_token_indices: Some(&rows),
                target_hidden: Some(&hidden),
            };
            let (argmax, next_hidden) =
                backend.forward_argmax_hidden_blocking(HEAD, HEAD_KV, &req, &rows)?;
            for (draft, &token) in drafts.iter_mut().zip(&argmax) {
                draft.push(token);
            }
            hidden = next_hidden;
        }
        Ok(drafting
            .iter()
            .zip(drafts)
            .map(|(&i, d)| {
                tracing::debug!(
                    "MTP drafts for {} after position {}: {d:?}",
                    seed.req_ids[i],
                    newest[i].map_or(0, |p| p + 1)
                );
                (seed.req_ids[i].clone(), d)
            })
            .collect())
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
    use std::path::Path;

    use super::*;

    /// One forward the proposer issued: its rows and the hidden rows it asked for.
    #[derive(Debug, PartialEq)]
    struct Forward {
        input_ids: Vec<u32>,
        positions: Vec<u32>,
        slot_mapping: Vec<u32>,
        cu_seqlens_q: Vec<u32>,
        seqused_k: Vec<u32>,
        block_table: Vec<u32>,
        last_token_indices: Vec<u32>,
        target_hidden: Vec<u8>,
        hidden_rows: Vec<u32>,
    }

    /// Records every head forward; forward `c` argmaxes row `r` to `1000 * (c + 1) + r` and
    /// returns hidden rows of bytes `c + 100`.
    #[derive(Default)]
    struct Recorder {
        forwards: Vec<Forward>,
    }

    const ROW_BYTES: usize = 4;

    impl SpecDecodeBackend for Recorder {
        fn forward_argmax_blocking(
            &mut self,
            _: ModelHandle,
            _: KvPoolHandle,
            _: &ForwardArgmaxRequest<'_>,
        ) -> Result<Vec<u32>, BackendError> {
            Err(BackendError::NotImplemented("argmax without hidden rows"))
        }

        fn forward_argmax_hidden_blocking(
            &mut self,
            model: ModelHandle,
            kv_pool: KvPoolHandle,
            req: &ForwardArgmaxRequest<'_>,
            hidden_rows: &[u32],
        ) -> Result<(Vec<u32>, Vec<u8>), BackendError> {
            assert_eq!((model, kv_pool), (HEAD, HEAD_KV));
            assert_eq!(req.num_tokens, req.input_ids.len());
            let c = self.forwards.len() as u32;
            self.forwards.push(Forward {
                input_ids: req.input_ids.to_vec(),
                positions: req.positions.to_vec(),
                slot_mapping: req.slot_mapping.to_vec(),
                cu_seqlens_q: req.cu_seqlens_q.to_vec(),
                seqused_k: req.seqused_k.to_vec(),
                block_table: req.block_table.to_vec(),
                last_token_indices: req.last_token_indices.unwrap_or_default().to_vec(),
                target_hidden: req.target_hidden.unwrap_or_default().to_vec(),
                hidden_rows: hidden_rows.to_vec(),
            });
            let argmax = (0..req.num_tokens as u32).map(|r| 1000 * (c + 1) + r);
            let hidden = vec![c as u8 + 100; hidden_rows.len() * ROW_BYTES];
            Ok((argmax.collect(), hidden))
        }

        fn load_secondary_model(
            &mut self,
            _: &Path,
            _: Option<&str>,
        ) -> Result<ModelHandle, BackendError> {
            Err(BackendError::NotImplemented("load_secondary_model"))
        }

        fn allocate_kv_pool(
            &mut self,
            _: ModelHandle,
            _: usize,
        ) -> Result<KvPoolHandle, BackendError> {
            Err(BackendError::NotImplemented("allocate_kv_pool"))
        }

        fn kv_per_block_bytes(&self, _: ModelHandle) -> Result<usize, BackendError> {
            Err(BackendError::NotImplemented("kv_per_block_bytes"))
        }
    }

    /// A step of two sequences: `a` verified three drafts at positions 10..=13 and kept one (its
    /// newest token, the bonus, sits at 12, produced by the row at 11); `b` ran an intermediate
    /// prefill chunk, 0..=2 of an 8-token prompt. Its seed, the tokens it produced, and the
    /// requests' histories.
    fn two_sequence_step() -> (DraftSeedInputs, Vec<Vec<u32>>, Vec<u32>, Vec<u32>) {
        let seed = DraftSeedInputs {
            input_ids: Vec::new(),
            positions: vec![10, 11, 12, 13, 0, 1, 2],
            slot_mapping: vec![58, 59, 60, 61, 80, 81, 82],
            cu_seqlens_q: vec![0, 4, 7],
            seqused_k: vec![14, 3],
            block_table: vec![3, 0, 5, 0],
            block_table_stride: 2,
            max_seqlen_q: 4,
            max_seqlen_k: 14,
            num_tokens: 7,
            req_ids: vec!["a".into(), "b".into()],
            block_ids: vec![vec![3], vec![5]],
            tokens_before: vec![10, 0],
            q_lens: vec![4, 3],
            was_spec_decode: vec![true, false],
            block_size: 16,
            async_lockstep_done: false,
            speculative_seeds: Vec::new(),
            speculative_chain_drafts: Vec::new(),
            // Row `r`'s hidden state is `ROW_BYTES` bytes of `r`.
            target_hidden: (0..7u8).flat_map(|r| [r; ROW_BYTES]).collect(),
        };
        let produced = vec![vec![511, 512], Vec::new()];
        (seed, produced, (500..513).collect(), (700..708).collect())
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

    fn hidden_of(rows: &[u8]) -> Vec<u8> {
        rows.iter().flat_map(|&r| [r; ROW_BYTES]).collect()
    }

    /// Chained forward `c` of `a`: the draft before it at `p`, block 3, the hidden rows forward
    /// `c - 1` returned.
    fn chain(c: u8, input: u32, p: u32) -> Forward {
        Forward {
            input_ids: vec![input],
            positions: vec![p],
            slot_mapping: vec![3 * 16 + p],
            cu_seqlens_q: vec![0, 1],
            seqused_k: vec![p + 1],
            block_table: vec![3, 0],
            last_token_indices: vec![0],
            target_hidden: vec![c - 1 + 100; ROW_BYTES],
            hidden_rows: vec![0],
        }
    }

    /// [`two_sequence_step`], both planned: pass 1 runs `a` through its newest row and all of `b`;
    /// `a` alone drafts, chaining at 12 and 13.
    #[test]
    fn pass_one_runs_each_sequence_through_its_newest_row() {
        let (seed, produced, history_a, history_b) = two_sequence_step();
        let history = |id: &str| match id {
            "a" => Some(history_a.clone()),
            "b" => Some(history_b.clone()),
            _ => None,
        };
        let mut backend = Recorder::default();
        let drafts = DRAFTER
            .draft(&seed, &produced, &history, &mut backend, &ids(&["a", "b"]))
            .expect("drafts");
        let want = [
            Forward {
                input_ids: vec![history_a[11], history_a[12], 701, 702, 703],
                positions: vec![10, 11, 0, 1, 2],
                slot_mapping: vec![58, 59, 80, 81, 82],
                cu_seqlens_q: vec![0, 2, 5],
                seqused_k: vec![12, 3],
                block_table: vec![3, 0, 5, 0],
                last_token_indices: vec![1, 4],
                target_hidden: hidden_of(&[0, 1, 4, 5, 6]),
                hidden_rows: vec![1],
            },
            chain(1, 1001, 12),
            chain(2, 2000, 13),
        ];
        assert_eq!(backend.forwards, want);
        let want_drafts = HashMap::from([("a".to_string(), vec![1001, 2000, 3000])]);
        assert_eq!(drafts, want_drafts);
    }

    /// The head runs for the planned requests alone: `a` without `b` drafts as before from rows
    /// of its own; `b` alone only primes its prompt; nothing planned, nothing runs.
    #[test]
    fn pass_one_runs_only_the_planned_requests() {
        let (seed, produced, history_a, history_b) = two_sequence_step();
        let history = |id: &str| match id {
            "a" => Some(history_a.clone()),
            "b" => Some(history_b.clone()),
            _ => None,
        };
        let run = |plan: &[&str]| {
            let mut backend = Recorder::default();
            let drafts = DRAFTER
                .draft(&seed, &produced, &history, &mut backend, &ids(plan))
                .expect("drafts");
            (backend.forwards, drafts)
        };

        let (forwards, drafts) = run(&["a"]);
        let pass1 = Forward {
            input_ids: vec![history_a[11], history_a[12]],
            positions: vec![10, 11],
            slot_mapping: vec![58, 59],
            cu_seqlens_q: vec![0, 2],
            seqused_k: vec![12],
            block_table: vec![3, 0],
            last_token_indices: vec![1],
            target_hidden: hidden_of(&[0, 1]),
            hidden_rows: vec![1],
        };
        assert_eq!(forwards, [pass1, chain(1, 1001, 12), chain(2, 2000, 13)]);
        assert_eq!(
            drafts,
            HashMap::from([("a".to_string(), vec![1001, 2000, 3000])])
        );

        let (forwards, drafts) = run(&["b"]);
        assert_eq!(forwards.len(), 1, "pass 1 primes b's prompt");
        assert_eq!(forwards[0].positions, [0, 1, 2]);
        assert!(drafts.is_empty());

        let (forwards, drafts) = run(&[]);
        assert!(forwards.is_empty() && drafts.is_empty());
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

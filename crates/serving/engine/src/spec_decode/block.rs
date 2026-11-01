// SPDX-License-Identifier: Apache-2.0

//! A block drafter: Splash's DFlash2 drafter (~/git/splash @ 1482f6bb), whose one pass drafts all
//! of a sequence's tokens over a block of rows — its newest token and a mask row a draft — reading
//! the context of the rows its target retained, which it keeps in rings of its own: per layer, the
//! keys and values of each sequence's last `window` positions (Splash's `DFlashDraftRing`).
//!
//! Splash's cycle (runtime/model/Runtime.cpp:2923-3017) drafts a block, verifies it on its target,
//! accepts, and commits the context of the rows its target retained to the drafter's rings, one
//! command a cycle. A step here runs the same sequence from its target's verify on, in the step's
//! command buffer, and returns the drafts with its output for the engine to schedule into the next
//! step:
//!
//! 1. **The acceptance** of each sequence the step gives a token (`dflash2_accept`, Splash's
//!    `decode_accept_dflash`): the rows its target retained of the block it verified, and its next
//!    block — its last retained token, then the mask, from the position after them — picked on the
//!    device ([`BlockLane`]).
//! 2. **The commit** ([`Commit`]): the drafter's cache-writes play over the planned sequences' rows
//!    of the step, their target's captured hidden states into each layer's ring. A step whose
//!    sequences each verify a block commits the rows each retained (Splash's cycle commit); any
//!    other commits every row (a prompt's, a decode step's: all of them are kept).
//! 3. **The block draft** ([`Block`]): the drafter's result play over each lane's next block,
//!    reading its ring — its drafts, what the drafter's selector picked, as many as the rows of
//!    the block its next verify checks (Splash's `draft_verify_rows`: on a target that routes
//!    experts, the rows its verify budget chooses; [`BlockDrafter::drafts`]). Splash's verify runs
//!    a block's every row and routes the experts of those alone; the next step here verifies those
//!    rows alone, at the bucket that holds them.
//!
//! A step drafts while its sequences' blocks fit its target's verify rows (`DraftBudget`, `block`):
//! a sequence's whole block or nothing, since the drafter computes every proposal of it. Every step
//! commits, so a sequence's ring never has a hole: drafting resumes as soon as a step is back
//! within the budget.

use std::collections::{HashMap, HashSet};

use super::backend::ForwardArgmaxRequest;
use super::draft_context::{DraftContextError, DraftContextPlan};
use super::mtp::{DraftBudget, StepRows};

/// A block drafter's passes over one step (module docs), run by the worker that ran it.
#[derive(Clone, Copy, Debug)]
pub struct BlockDrafter {
    /// The drafts each sequence of a step gets: its block's, or none past the budget.
    pub budget: DraftBudget,
    /// Maximum model context length: no draft lands past it.
    pub max_model_len: usize,
    /// The positions a sequence's rings hold (Splash's `draftContextTokens`): the most of a
    /// prefill's rows before its end it commits.
    pub window: usize,
}

/// What a step knows of one sequence's prompt context: the inputs of its [`DraftContextPlan`] and
/// its part in this step's snapshots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SequenceContext<'a> {
    /// Where its prefill resumed on admission: 0, or the snapshot it restored.
    pub resume: u32,
    /// The positions whose state its prefill snapshots, ascending (its admission's).
    pub save_points: &'a [u32],
    /// This step restored its rings' context window from the snapshot it resumed at.
    pub restored: bool,
    /// The positions whose snapshot this step saves, ascending.
    pub saves: &'a [u32],
}

/// The drafter's passes over a step (module docs).
#[derive(Debug, PartialEq)]
pub struct BlockPasses {
    /// The rings restored this step, rebuilt from their context windows before any commit
    /// (Splash `addDraftRebuild`, runtime/model/Runtime.cpp:1556-1564, 1694-1701).
    pub rebuild: Option<Rebuild>,
    /// The step's commit, cut at each position whose snapshot the step saves. Each commit holds
    /// a sequence's rows up to its next cut, so a snapshot's window is copied out of its
    /// sequence's once the commit ending at it has run, before any row past it overwrites a
    /// position the snapshot keeps ([`Commit::saves`]). This is Splash's window capture of a
    /// state's rows below its boundary (`addDraftContext`, Runtime.cpp:1570-1606).
    pub commits: Vec<Commit>,
    /// The block draft, when the step drafts.
    pub block: Option<Block>,
}

/// The drafter's rebuild play over the restored sequences' windows: each one's last positions
/// before its resume, as many as its rings hold.
#[derive(Debug, Default, PartialEq)]
pub struct Rebuild {
    /// Placeholders: a rebuild reads its rows out of the window, not the target.
    pub input_ids: Vec<u32>,
    pub positions: Vec<u32>,
    /// A slot a row, every one none: the drafter writes no paged KV.
    pub slot_mapping: Vec<u32>,
    pub cu_seqlens_q: Vec<u32>,
    pub seqused_k: Vec<u32>,
    /// Each sequence's index in the step: whose rings it rebuilds.
    pub steps: Vec<usize>,
}

impl Rebuild {
    /// The forward's host-side request.
    pub fn request(&self) -> ForwardArgmaxRequest<'_> {
        request(
            [&self.input_ids, &self.positions, &self.slot_mapping],
            &self.cu_seqlens_q,
            &self.seqused_k,
        )
    }

    /// This rebuild as forwards of at most `rows` rows each, in order. A sequence's rows are cut
    /// where a forward fills, since a rebuilt row reads its window alone, no other row.
    pub fn chunks(&self, rows: usize) -> Vec<Rebuild> {
        let fresh = || Rebuild {
            cu_seqlens_q: vec![0],
            ..Rebuild::default()
        };
        let (mut chunks, mut chunk) = (Vec::new(), fresh());
        for (k, &i) in self.steps.iter().enumerate() {
            let (mut at, end) = (
                self.cu_seqlens_q[k] as usize,
                self.cu_seqlens_q[k + 1] as usize,
            );
            while at < end {
                if chunk.positions.len() == rows {
                    chunks.push(std::mem::replace(&mut chunk, fresh()));
                }
                let n = (end - at).min(rows - chunk.positions.len());
                chunk.input_ids.extend(&self.input_ids[at..at + n]);
                chunk.positions.extend(&self.positions[at..at + n]);
                chunk.slot_mapping.extend(&self.slot_mapping[at..at + n]);
                chunk.cu_seqlens_q.push(chunk.positions.len() as u32);
                chunk.seqused_k.push(self.positions[at + n - 1] + 1);
                chunk.steps.push(i);
                at += n;
            }
        }
        if !chunk.steps.is_empty() {
            chunks.push(chunk);
        }
        chunks
    }
}

/// The drafter's cache-writes play over the planned sequences' rows of a step: their host inputs,
/// and the target's captured rows the device copies in.
#[derive(Debug, PartialEq)]
pub struct Commit {
    pub input_ids: Vec<u32>,
    pub positions: Vec<u32>,
    /// A slot a row, every one none: the drafter writes no paged KV.
    pub slot_mapping: Vec<u32>,
    pub cu_seqlens_q: Vec<u32>,
    pub seqused_k: Vec<u32>,
    /// `(row, step row, rows)`: each sequence's rows, reading the target's captured rows at its
    /// rows of the step.
    pub hidden_rows: Vec<(u32, u32, u32)>,
    /// Each sequence's index in the step: whose ring it commits to.
    pub steps: Vec<usize>,
    /// Every sequence verifies a block, a lane of [`BlockPasses`]'s acceptance each in order: each
    /// commits the rows it retained.
    pub cycle: bool,
    /// `(sequence's index in the step, position)`: the snapshots whose context window is whole
    /// once this commit has run.
    pub saves: Vec<(usize, u32)>,
}

impl Commit {
    fn empty() -> Self {
        Self {
            input_ids: Vec::new(),
            positions: Vec::new(),
            slot_mapping: Vec::new(),
            cu_seqlens_q: vec![0],
            seqused_k: Vec::new(),
            hidden_rows: Vec::new(),
            steps: Vec::new(),
            cycle: false,
            saves: Vec::new(),
        }
    }

    /// Commit `positions` of sequence `i`: its rows from `step_row`, the step's.
    fn push(&mut self, step: &StepRows<'_>, i: usize, step_row: usize, positions: (u32, u32)) {
        let n = (positions.1 - positions.0) as usize;
        let rows = step_row..step_row + n;
        let start = self.input_ids.len() as u32;
        self.input_ids.extend(&step.input_ids[rows.clone()]);
        self.positions.extend(&step.positions[rows.clone()]);
        self.slot_mapping.extend(rows.map(|_| u32::MAX));
        self.cu_seqlens_q.push(self.input_ids.len() as u32);
        self.seqused_k.push(positions.1);
        self.hidden_rows.push((start, step_row as u32, n as u32));
        self.steps.push(i);
    }
}

/// One sequence the step gives a token, as the acceptance reads it
/// (`scratchy_target_metal::dflash2::VerifyLane`, less what the worker knows of its request).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockLane {
    /// Its index in the step.
    pub step: usize,
    /// Its first verify row among the step's sampled rows: the first of its block's, or its one.
    pub sampled: u32,
    /// The drafts its verify carried: none for a sequence that decoded or ended its prompt.
    pub drafts: u32,
    /// That row's position.
    pub position: u32,
}

/// The acceptance's lanes and the drafter's result play over their next blocks.
#[derive(Debug, PartialEq)]
pub struct Block {
    pub lanes: Vec<BlockLane>,
    /// `[lanes][budget drafts]`: the drafts each lane's verify carried, zero past its own.
    pub drafted: Vec<u32>,
    /// The block forward's host inputs, a block a lane. Its tokens and positions are placeholders
    /// the device overwrites: the positions those of the most rows retained.
    pub input_ids: Vec<u32>,
    pub positions: Vec<u32>,
    /// A slot a row, every one none: the drafter writes no paged KV.
    pub slot_mapping: Vec<u32>,
    pub cu_seqlens_q: Vec<u32>,
    pub seqused_k: Vec<u32>,
    /// Whether the block forward runs: the step drafts. Its acceptance runs whenever the commit
    /// is a cycle's.
    pub drafts: bool,
}

impl Commit {
    /// The forward's host-side request (its target hidden states come from the device).
    pub fn request(&self) -> ForwardArgmaxRequest<'_> {
        request(
            [&self.input_ids, &self.positions, &self.slot_mapping],
            &self.cu_seqlens_q,
            &self.seqused_k,
        )
    }
}

impl Block {
    /// The block forward's host-side request (its tokens and positions come from the device).
    pub fn request(&self) -> ForwardArgmaxRequest<'_> {
        request(
            [&self.input_ids, &self.positions, &self.slot_mapping],
            &self.cu_seqlens_q,
            &self.seqused_k,
        )
    }

    /// The lanes' block rows: each lane's anchor and its proposals.
    pub fn rows(&self) -> usize {
        self.input_ids.len() / self.lanes.len().max(1)
    }
}

/// A drafter forward over `input_ids`: no paged KV (its rings are its KV), every row sampled — its
/// selector reads every proposal row's logits.
fn request<'a>(
    [input_ids, positions, slot_mapping]: [&'a [u32]; 3],
    cu_seqlens_q: &'a [u32],
    seqused_k: &'a [u32],
) -> ForwardArgmaxRequest<'a> {
    ForwardArgmaxRequest {
        input_ids,
        positions,
        slot_mapping,
        cu_seqlens_q,
        seqused_k,
        block_table: &[],
        span_ids: None,
        sliding_slot_mappings: &[],
        sliding_block_tables: &[],
        block_table_stride: 0,
        max_seqlen_q: (cu_seqlens_q.windows(2))
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0),
        max_seqlen_k: seqused_k.iter().copied().max().unwrap_or(0) as usize,
        num_tokens: input_ids.len(),
        sampled_rows: None,
        target_hidden: None,
    }
}

impl BlockDrafter {
    /// The drafter's passes over the rows of `step` that the requests in `plan` run, each sequence's
    /// prompt context by `contexts` (by its index in the step; a sequence past them resumed at 0 and
    /// snapshots nothing). `None` when none is planned.
    pub fn passes(
        &self,
        step: &StepRows<'_>,
        plan: &HashSet<String>,
        contexts: &[SequenceContext<'_>],
    ) -> Result<Option<BlockPasses>, DraftContextError> {
        let seqs: Vec<usize> = (0..step.req_ids.len())
            .filter(|&i| plan.contains(&step.req_ids[i]))
            .collect();
        if seqs.is_empty() {
            return Ok(None);
        }
        let rows_of = |i: usize| step.cu_seqlens_q[i] as usize..step.cu_seqlens_q[i + 1] as usize;
        let window = self.window as u32;
        let mut commits = vec![Commit {
            cycle: true,
            ..Commit::empty()
        }];
        let mut rebuild = Rebuild {
            cu_seqlens_q: vec![0],
            ..Rebuild::default()
        };
        let block_rows = self.budget.drafts + 1;
        let mut lanes = Vec::new();
        for &i in &seqs {
            let rows = rows_of(i);
            let context = contexts.get(i).copied().unwrap_or_default();
            // Splash's draft context plan (runtime/model/DraftContextPlan.cpp:7-80
            // `planDraftContext`): what a sequence prefills up to the end of its history is
            // captured into its rings from the window before each boundary — every snapshot its
            // prefill saves and its end — and a chunk short of the end drafts nothing. Its rows past
            // that end, a decode step's, are all kept.
            let end = step.history_lens[i];
            let resume = context.resume.min(end);
            let boundaries: Vec<u32> = (context.save_points.iter().copied())
                .filter(|&b| b > resume && b <= end)
                .collect();
            let prompt = DraftContextPlan::of(resume, end, &boundaries, window)?;
            if context.restored && prompt.restores_draft_state {
                let n = resume.min(window);
                rebuild.input_ids.extend((0..n).map(|_| 0));
                rebuild.positions.extend(resume - n..resume);
                rebuild.slot_mapping.extend((0..n).map(|_| u32::MAX));
                rebuild.cu_seqlens_q.push(rebuild.positions.len() as u32);
                rebuild.seqused_k.push(resume);
                rebuild.steps.push(i);
            }
            let (first, last) = (step.positions[rows.start], step.positions[rows.end - 1] + 1);
            let reaches_end = last >= end;
            let mut kept: Vec<(u32, u32)> = Vec::new();
            let (from, to) = (first.max(resume), last.min(end));
            if from < to {
                kept.extend((prompt.dispatch(from, to)?.iter()).map(|c| (c.begin, c.end)));
            }
            let past = first.max(end);
            if past < last {
                match kept.last_mut() {
                    Some(k) if k.1 == past => k.1 = last,
                    _ => kept.push((past, last)),
                }
            }
            // Cut at each snapshot the step saves: a commit a cut.
            let saves: Vec<u32> = (context.saves.iter().copied())
                .filter(|&p| p > first && p <= last)
                .collect();
            for (n, &at) in saves.iter().enumerate() {
                if commits.len() <= n {
                    commits.push(Commit::empty());
                }
                commits[n].saves.push((i, at));
            }
            if kept.is_empty() {
                continue;
            }
            for (begin, end) in kept {
                let mut at = begin;
                for cut in (saves.iter().copied())
                    .filter(|&c| c > begin && c < end)
                    .chain([end])
                {
                    let k = saves.iter().filter(|&&c| c <= at).count();
                    if commits.len() <= k {
                        commits.push(Commit::empty());
                    }
                    let step_row = rows.start + (at - first) as usize;
                    commits[k].push(step, i, step_row, (at, cut));
                    at = cut;
                }
            }
            // A sequence the target samples every row of verifies them; one it samples the last
            // row of gets that row's token; one it samples none of runs a prompt chunk short of
            // its end.
            let sampled = |r: usize| step.sampled.index(i, r as u32);
            let verifies = rows.len() > 1 && rows.clone().all(|r| sampled(r).is_some());
            commits[0].cycle &= verifies && saves.is_empty();
            let first_row = match verifies {
                true => rows.start,
                false => rows.end - 1,
            };
            if let Some(at) = sampled(first_row).filter(|_| reaches_end) {
                lanes.push(BlockLane {
                    step: i,
                    sampled: at as u32,
                    drafts: (rows.end - 1 - first_row) as u32,
                    position: step.positions[first_row],
                });
            }
        }
        let rebuild = (!rebuild.steps.is_empty()).then_some(rebuild);
        let commits_anything = (commits.iter()).any(|c| !c.steps.is_empty() || !c.saves.is_empty());
        if !commits_anything && rebuild.is_none() {
            return Ok(None);
        }
        // A cycle's commit reads a retained count a lane: every sequence is one, in one commit.
        commits[0].cycle &= commits.len() == 1 && lanes.len() == commits[0].steps.len();
        let cycle = commits[0].cycle;
        let drafts = self.budget.drafts_at(step.req_ids.len()) > 0
            && (lanes.iter()).all(|l| self.fits(l, block_rows))
            && !lanes.is_empty();
        let block = (drafts || cycle).then(|| self.block(step, lanes, drafts));
        Ok(Some(BlockPasses {
            rebuild,
            commits,
            block,
        }))
    }

    /// Whether `lane`'s next block, after the most rows it can retain, lands within the context.
    fn fits(&self, lane: &BlockLane, block_rows: usize) -> bool {
        let next = (lane.position + lane.drafts + 1) as usize;
        next + block_rows <= self.max_model_len
    }

    fn block(&self, step: &StepRows<'_>, lanes: Vec<BlockLane>, drafts: bool) -> Block {
        let (k, rows) = (self.budget.drafts, self.budget.drafts + 1);
        let mut drafted = vec![0; lanes.len() * k];
        for (lane, l) in lanes.iter().enumerate() {
            let first = step.sampled.rows()[l.sampled as usize] as usize;
            for d in 0..l.drafts as usize {
                drafted[lane * k + d] = step.input_ids[first + 1 + d];
            }
        }
        let positions: Vec<u32> = (lanes.iter())
            .flat_map(|l| (0..rows as u32).map(move |r| l.position + l.drafts + 1 + r))
            .collect();
        Block {
            drafted,
            input_ids: vec![0; lanes.len() * rows],
            slot_mapping: vec![u32::MAX; lanes.len() * rows],
            seqused_k: (lanes.iter())
                .map(|l| l.position + l.drafts + 1 + rows as u32)
                .collect(),
            positions,
            cu_seqlens_q: (0..=lanes.len()).map(|l| (l * rows) as u32).collect(),
            lanes,
            drafts,
        }
    }

    /// The drafts after the block draft ran: `tokens`, its lanes' proposals read back, as many of
    /// each lane's as its next verify checks — `verify_rows`, its anchor's row and its proposals'
    /// (Splash's `draft_verify_rows`) — for the requests of `step` that produced `produced` (per
    /// request, in step order); none for a prompt chunk short of its end.
    pub fn drafts(
        &self,
        block: &Block,
        tokens: &[u32],
        verify_rows: &[u32],
        step: &StepRows<'_>,
        produced: &[Vec<u32>],
    ) -> HashMap<String, Vec<u32>> {
        let k = self.budget.drafts;
        (block.lanes.iter().enumerate())
            .filter(|(_, l)| !produced[l.step].is_empty())
            .map(|(lane, l)| {
                let verified = (verify_rows[lane] as usize).saturating_sub(1).min(k);
                let d = tokens[lane * k..lane * k + verified].to_vec();
                tracing::debug!(
                    "DFlash2 drafts for {} after position {}: {d:?}",
                    step.req_ids[l.step],
                    l.position + l.drafts
                );
                (step.req_ids[l.step].clone(), d)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec_decode::SampledRows;

    const DRAFTER: BlockDrafter = BlockDrafter {
        budget: DraftBudget {
            drafts: 7,
            verify_rows: 8,
            block: true,
        },
        max_model_len: 4096,
        window: 2048,
    };

    struct Step {
        req_ids: Vec<String>,
        input_ids: Vec<u32>,
        positions: Vec<u32>,
        cu: Vec<u32>,
        history: Vec<u32>,
        sampled: SampledRows,
    }

    impl Step {
        fn rows(&self) -> StepRows<'_> {
            StepRows {
                req_ids: &self.req_ids,
                input_ids: &self.input_ids,
                positions: &self.positions,
                slot_mapping: &[],
                cu_seqlens_q: &self.cu,
                seqused_k: &[],
                block_table: &[],
                block_table_stride: 0,
                block_ids: &[],
                block_size: 16,
                prompt_lens: &[],
                history_lens: &self.history,
                sampled: &self.sampled,
            }
        }
    }

    fn plan(ids: &[&str]) -> HashSet<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    /// A sequence verifying its block at positions 40..=47 (its anchor 500, drafts 501..=507)
    /// commits as a cycle's lane, accepts its seven drafts, and drafts its next block from the
    /// position after the most rows it can retain.
    #[test]
    fn a_verify_of_a_block_commits_its_retained_rows_and_drafts_the_next() {
        let step = Step {
            req_ids: vec!["a".into()],
            input_ids: (500..508).collect(),
            positions: (40..48).collect(),
            cu: vec![0, 8],
            history: vec![41],
            sampled: SampledRows::of(&[0, 8], |_| true).expect("rows"),
        };
        let passes = DRAFTER
            .passes(&step.rows(), &plan(&["a"]), &[])
            .expect("a valid plan")
            .expect("planned");
        assert!(passes.commits[0].cycle);
        assert_eq!(passes.commits[0].hidden_rows, [(0, 0, 8)]);
        let block = passes.block.expect("drafts");
        assert!(block.drafts);
        let lane = BlockLane {
            step: 0,
            sampled: 0,
            drafts: 7,
            position: 40,
        };
        assert_eq!(block.lanes, [lane]);
        assert_eq!(block.drafted, (501..508).collect::<Vec<_>>());
        assert_eq!(block.positions, (48..56).collect::<Vec<_>>());
        assert_eq!(block.cu_seqlens_q, [0, 8]);
        let proposals: Vec<u32> = (900..907).collect();
        let drafts = DRAFTER.drafts(&block, &proposals, &[8], &step.rows(), &[vec![1]]);
        assert_eq!(drafts["a"], proposals);
        let budgeted = DRAFTER.drafts(&block, &proposals, &[3], &step.rows(), &[vec![1]]);
        assert_eq!(
            budgeted["a"],
            [900, 901],
            "a verify of 3 rows checks 2 proposals"
        );
        let anchor = DRAFTER.drafts(&block, &proposals, &[1], &step.rows(), &[vec![1]]);
        assert!(
            anchor["a"].is_empty(),
            "a verify of its anchor alone checks none"
        );
        let chunk = DRAFTER.drafts(&block, &proposals, &[8], &step.rows(), &[vec![]]);
        assert!(
            chunk.is_empty(),
            "a step that produced no token drafts nothing"
        );
    }

    /// Splash's draft context plan, at a window of 16 positions: a 40-token prompt prefilled in
    /// chunks of 16 commits nothing of its first chunk (every row is more than a window before its
    /// end) and runs no pass at all; of its second, the rows from position 24, with no block
    /// draft (the chunk is short of the end); of its last, every row, then drafts its block.
    #[test]
    fn a_prefill_commits_the_window_before_its_end_and_drafts_at_its_end() {
        let drafter = BlockDrafter {
            window: 16,
            ..DRAFTER
        };
        let chunk = |from: u32, to: u32| Step {
            req_ids: vec!["a".into()],
            input_ids: (from + 100..to + 100).collect(),
            positions: (from..to).collect(),
            cu: vec![0, to - from],
            history: vec![40],
            sampled: SampledRows::of(&[0, to - from], |_| false).expect("rows"),
        };
        assert_eq!(
            drafter
                .passes(&chunk(0, 16).rows(), &plan(&["a"]), &[])
                .expect("a valid plan"),
            None
        );
        let middle = (drafter
            .passes(&chunk(16, 32).rows(), &plan(&["a"]), &[])
            .expect("a valid plan"))
        .expect("commits");
        assert_eq!(middle.commits[0].positions, (24..32).collect::<Vec<_>>());
        assert_eq!(middle.commits[0].input_ids, (124..132).collect::<Vec<_>>());
        assert_eq!(middle.commits[0].hidden_rows, [(0, 8, 8)]);
        assert_eq!(
            middle.block, None,
            "a chunk short of its end drafts nothing"
        );
        let last = (drafter
            .passes(&chunk(32, 40).rows(), &plan(&["a"]), &[])
            .expect("a valid plan"))
        .expect("commits");
        assert_eq!(last.commits[0].positions, (32..40).collect::<Vec<_>>());
        let block = last.block.expect("drafts at its end");
        assert_eq!(block.positions, (40..48).collect::<Vec<_>>());
    }

    /// A sequence verifying the 3 rows of its block its budget chose (its anchor 500, drafts 501
    /// and 502) commits as a cycle's lane all the same.
    #[test]
    fn a_verify_of_fewer_rows_commits_as_a_cycle() {
        let step = Step {
            req_ids: vec!["a".into()],
            input_ids: (500..503).collect(),
            positions: (40..43).collect(),
            cu: vec![0, 3],
            history: vec![41],
            sampled: SampledRows::of(&[0, 3], |_| true).expect("rows"),
        };
        let passes = DRAFTER
            .passes(&step.rows(), &plan(&["a"]), &[])
            .expect("a valid plan")
            .expect("planned");
        assert!(passes.commits[0].cycle);
        assert_eq!(passes.commits[0].hidden_rows, [(0, 0, 3)]);
        let block = passes.block.expect("drafts");
        assert_eq!(block.drafted[..2], [501, 502]);
        assert_eq!(block.positions, (43..51).collect::<Vec<_>>());
    }

    /// A prompt's last chunk commits every row and drafts from its last; two sequences decoding
    /// past the budget commit every row and draft nothing; an unplanned step runs nothing.
    #[test]
    fn a_prompt_commits_every_row_and_drafts_from_its_end() {
        let last = Step {
            req_ids: vec!["a".into()],
            input_ids: (0..5).collect(),
            positions: (0..5).collect(),
            cu: vec![0, 5],
            history: vec![5],
            sampled: SampledRows::of(&[0, 5], |_| false).expect("rows"),
        };
        let passes = DRAFTER
            .passes(&last.rows(), &plan(&["a"]), &[])
            .expect("a valid plan")
            .expect("planned");
        assert!(!passes.commits[0].cycle);
        let block = passes.block.expect("drafts");
        let lane = BlockLane {
            step: 0,
            sampled: 0,
            drafts: 0,
            position: 4,
        };
        assert_eq!(block.lanes, [lane]);
        assert_eq!(block.positions, (5..13).collect::<Vec<_>>());

        let decoding = Step {
            req_ids: vec!["a".into(), "b".into()],
            input_ids: vec![7, 8],
            positions: vec![30, 12],
            cu: vec![0, 1, 2],
            history: vec![31, 13],
            sampled: SampledRows::of(&[0, 1, 2], |_| false).expect("rows"),
        };
        let passes = (DRAFTER
            .passes(&decoding.rows(), &plan(&["a", "b"]), &[])
            .expect("a valid plan"))
        .expect("planned");
        assert!(!passes.commits[0].cycle, "a decode step commits every row");
        assert_eq!(passes.commits[0].positions, [30, 12]);
        // Each sequence is a lane of its own (up to BLOCK_LANES), its block its own 8 rows.
        let block = passes.block.expect("two lanes draft");
        assert_eq!(
            block.lanes,
            [
                BlockLane {
                    step: 0,
                    sampled: 0,
                    drafts: 0,
                    position: 30,
                },
                BlockLane {
                    step: 1,
                    sampled: 1,
                    drafts: 0,
                    position: 12,
                },
            ]
        );
        assert_eq!(block.cu_seqlens_q, [0, 8, 16], "a lane's block is its own 8 rows");
        assert_eq!(block.positions, [(31..39).collect::<Vec<_>>(), (13..21).collect()].concat());
        let five: Vec<String> = (0..5).map(|i| format!("s{i}")).collect();
        let wide = Step {
            req_ids: five.clone(),
            input_ids: vec![7; 5],
            positions: vec![30; 5],
            cu: (0..=5).collect(),
            history: vec![31; 5],
            sampled: SampledRows::of(&[0, 1, 2, 3, 4, 5], |_| false).expect("rows"),
        };
        let ids: Vec<&str> = five.iter().map(String::as_str).collect();
        assert_eq!(
            (DRAFTER.passes(&wide.rows(), &plan(&ids), &[]).expect("a valid plan"))
                .and_then(|p| p.block),
            None,
            "a step past BLOCK_LANES sequences drafts none"
        );
        assert_eq!(
            DRAFTER
                .passes(&decoding.rows(), &plan(&[]), &[])
                .expect("a valid plan"),
            None
        );
    }

    /// A sequence resumed at a snapshot whose prompt ends within a window of it rebuilds its rings
    /// from the snapshot's window, its last window of positions before the resume, and commits
    /// on from the resume; one ending a window or more past it rebuilds nothing.
    #[test]
    fn a_restore_rebuilds_its_window_before_a_suffix_within_one() {
        let drafter = BlockDrafter {
            window: 16,
            ..DRAFTER
        };
        let resumed = |end: u32| Step {
            req_ids: vec!["a".into()],
            input_ids: (132..100 + end).collect(),
            positions: (32..end).collect(),
            cu: vec![0, end - 32],
            history: vec![end],
            sampled: SampledRows::of(&[0, end - 32], |_| false).expect("rows"),
        };
        let restored = [SequenceContext {
            resume: 32,
            restored: true,
            ..SequenceContext::default()
        }];
        let short = resumed(40);
        let passes = (drafter.passes(&short.rows(), &plan(&["a"]), &restored))
            .expect("a valid plan")
            .expect("planned");
        let rebuild = passes.rebuild.expect("rebuilds");
        assert_eq!(rebuild.positions, (16..32).collect::<Vec<_>>());
        assert_eq!(
            (rebuild.cu_seqlens_q, rebuild.steps),
            (vec![0, 16], vec![0])
        );
        assert_eq!(passes.commits.len(), 1);
        assert_eq!(passes.commits[0].positions, (32..40).collect::<Vec<_>>());
        let long = resumed(48);
        let passes = (drafter.passes(&long.rows(), &plan(&["a"]), &restored))
            .expect("a valid plan")
            .expect("planned");
        assert_eq!(
            passes.rebuild, None,
            "its fresh window overwrites the restored one"
        );
        assert_eq!(passes.commits[0].positions, (32..48).collect::<Vec<_>>());
    }

    /// A rebuild of two sequences, 16 and 8 rows, in forwards of at most 10 rows: each forward
    /// takes the next rows in order, a sequence cut where one fills.
    #[test]
    fn a_rebuild_splits_into_forwards_its_bucket_holds() {
        let rebuild = Rebuild {
            input_ids: vec![0; 24],
            positions: (16..32).chain(40..48).collect(),
            slot_mapping: vec![u32::MAX; 24],
            cu_seqlens_q: vec![0, 16, 24],
            seqused_k: vec![32, 48],
            steps: vec![0, 1],
        };
        let chunks = rebuild.chunks(10);
        let at = |c: &Rebuild| (c.positions.clone(), c.cu_seqlens_q.clone(), c.steps.clone());
        assert_eq!(
            chunks.iter().map(at).collect::<Vec<_>>(),
            [
                ((16..26).collect(), vec![0, 10], vec![0]),
                ((26..32).chain(40..44).collect(), vec![0, 6, 10], vec![0, 1]),
                ((44..48).collect(), vec![0, 4], vec![1]),
            ]
        );
        assert_eq!(chunks[1].seqused_k, [32, 44]);
        assert_eq!(rebuild.chunks(24), [rebuild]);
    }

    /// A prompt that snapshots its state at 32 inside its one chunk commits through 32 first, its
    /// snapshot's window whole after that commit, then the rest; the window is its plan's,
    /// from 16 on.
    #[test]
    fn a_snapshot_inside_a_chunk_cuts_its_commit() {
        let drafter = BlockDrafter {
            window: 16,
            ..DRAFTER
        };
        let prompt = Step {
            req_ids: vec!["a".into()],
            input_ids: (100..140).collect(),
            positions: (0..40).collect(),
            cu: vec![0, 40],
            history: vec![40],
            sampled: SampledRows::of(&[0, 40], |_| false).expect("rows"),
        };
        let saves = [32];
        let context = [SequenceContext {
            save_points: &saves,
            saves: &saves,
            ..SequenceContext::default()
        }];
        let passes = (drafter.passes(&prompt.rows(), &plan(&["a"]), &context))
            .expect("a valid plan")
            .expect("planned");
        assert_eq!(passes.commits.len(), 2);
        assert_eq!(passes.commits[0].positions, (16..32).collect::<Vec<_>>());
        assert_eq!(passes.commits[0].hidden_rows, [(0, 16, 16)]);
        assert_eq!(passes.commits[0].saves, [(0, 32)]);
        assert_eq!(passes.commits[1].positions, (32..40).collect::<Vec<_>>());
        assert_eq!(passes.commits[1].hidden_rows, [(0, 32, 8)]);
        assert!(passes.commits[1].saves.is_empty());
        assert!(passes.block.expect("drafts at its end").drafts);
    }
}

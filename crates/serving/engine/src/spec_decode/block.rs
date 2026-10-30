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
use super::draft_context::{
    DraftContextError, DraftContexts, capture_spans_for_dispatch, plan_draft_context,
};
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

/// The drafter's passes over a step (module docs).
#[derive(Debug, PartialEq)]
pub struct BlockPasses {
    pub commit: Commit,
    /// The block draft, when the step drafts.
    pub block: Option<Block>,
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
    /// `(row, step row, rows)`: each run's rows, reading the target's captured rows at its rows of
    /// the step.
    pub hidden_rows: Vec<(u32, u32, u32)>,
    /// Each run's sequence, its index in the step: whose ring it commits to. A prompt's step may
    /// commit more than one run of its rows (its draft context plan's capture spans).
    pub steps: Vec<usize>,
    /// Every sequence verifies a block, a lane of [`BlockPasses`]'s acceptance each in order: each
    /// commits the rows it retained.
    pub cycle: bool,
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
    /// The drafter's passes over the rows of `step` that the requests in `plan` run, a prompt's
    /// rows those its draft context plan (`contexts`) captures, a sequence whose rings wait for a
    /// rebuild drafting nothing. `None` when none is planned.
    pub fn passes(
        &self,
        step: &StepRows<'_>,
        plan: &HashSet<String>,
        contexts: &DraftContexts,
    ) -> Result<Option<BlockPasses>, DraftContextError> {
        let seqs: Vec<usize> = (0..step.req_ids.len())
            .filter(|&i| plan.contains(&step.req_ids[i]))
            .collect();
        if seqs.is_empty() {
            return Ok(None);
        }
        let rows_of = |i: usize| step.cu_seqlens_q[i] as usize..step.cu_seqlens_q[i + 1] as usize;
        let mut commit = Commit {
            input_ids: Vec::new(),
            positions: Vec::new(),
            slot_mapping: Vec::new(),
            cu_seqlens_q: vec![0],
            seqused_k: Vec::new(),
            hidden_rows: Vec::new(),
            steps: Vec::new(),
            cycle: true,
        };
        let block_rows = self.budget.drafts + 1;
        let mut lanes = Vec::new();
        for &i in &seqs {
            let rows = rows_of(i);
            // Splash's draft context plan (runtime/model/DraftContextPlan.cpp `planDraftContext`,
            // `draftCaptureSpansForDispatch`): a prompt's rows are captured and committed where
            // its plan says — each state it leaves, the snapshots it saves and its end, holds the
            // window before it. Rows past the prompt (a decode or verify step's), or of a sequence
            // with no plan, are the active state's of its history: the window before its end, all
            // its rings hold. A chunk short of the end drafts nothing.
            let end = step.history_lens[i];
            let (first, last) = (step.positions[rows.start], step.positions[rows.end - 1] + 1);
            let reaches_end = last >= end;
            let planned = (contexts.get(&step.req_ids[i])).filter(|c| last <= c.replay_end);
            let active;
            let context = match planned {
                Some(context) => context,
                None => {
                    active = plan_draft_context(0, end.max(last), &[], self.window as u32)?;
                    &active
                }
            };
            let captures = capture_spans_for_dispatch(context, first, last)?;
            if captures.is_empty() {
                continue;
            }
            for capture in captures {
                let at = |position: u32| rows.start + (position - first) as usize;
                let kept = at(capture.begin)..at(capture.end);
                let start = commit.input_ids.len() as u32;
                commit.input_ids.extend(&step.input_ids[kept.clone()]);
                commit.positions.extend(&step.positions[kept.clone()]);
                commit.slot_mapping.extend(kept.clone().map(|_| u32::MAX));
                commit.cu_seqlens_q.push(commit.input_ids.len() as u32);
                commit.seqused_k.push(capture.end);
                commit
                    .hidden_rows
                    .push((start, kept.start as u32, kept.len() as u32));
                commit.steps.push(i);
            }
            // A sequence the target samples every row of verifies them; one it samples the last
            // row of gets that row's token; one it samples none of runs a prompt chunk short of
            // its end.
            let sampled = |r: usize| step.sampled.index(i, r as u32);
            let verifies = rows.len() > 1 && rows.clone().all(|r| sampled(r).is_some());
            commit.cycle &= verifies;
            let first = match verifies {
                true => rows.start,
                false => rows.end - 1,
            };
            let drafts = contexts.drafts(&step.req_ids[i]);
            if let Some(at) = sampled(first).filter(|_| reaches_end && drafts) {
                lanes.push(BlockLane {
                    step: i,
                    sampled: at as u32,
                    drafts: (rows.end - 1 - first) as u32,
                    position: step.positions[first],
                });
            }
        }
        if commit.steps.is_empty() {
            return Ok(None);
        }
        // A cycle's commit reads a retained count a lane: every sequence is one.
        commit.cycle &= lanes.len() == commit.steps.len();
        let drafts = self.budget.drafts_at(step.req_ids.len()) > 0
            && (lanes.iter()).all(|l| self.fits(l, block_rows))
            && !lanes.is_empty();
        let block = (drafts || commit.cycle).then(|| self.block(step, lanes, drafts));
        Ok(Some(BlockPasses { commit, block }))
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
            .passes(&step.rows(), &plan(&["a"]), &DraftContexts::new(2048))
            .expect("a plan")
            .expect("planned");
        assert!(passes.commit.cycle);
        assert_eq!(passes.commit.hidden_rows, [(0, 0, 8)]);
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
                .passes(
                    &chunk(0, 16).rows(),
                    &plan(&["a"]),
                    &DraftContexts::new(2048)
                )
                .expect("a plan"),
            None
        );
        let middle = (drafter
            .passes(
                &chunk(16, 32).rows(),
                &plan(&["a"]),
                &DraftContexts::new(2048),
            )
            .expect("a plan"))
        .expect("commits");
        assert_eq!(middle.commit.positions, (24..32).collect::<Vec<_>>());
        assert_eq!(middle.commit.input_ids, (124..132).collect::<Vec<_>>());
        assert_eq!(middle.commit.hidden_rows, [(0, 8, 8)]);
        assert_eq!(
            middle.block, None,
            "a chunk short of its end drafts nothing"
        );
        let last = (drafter
            .passes(
                &chunk(32, 40).rows(),
                &plan(&["a"]),
                &DraftContexts::new(2048),
            )
            .expect("a plan"))
        .expect("commits");
        assert_eq!(last.commit.positions, (32..40).collect::<Vec<_>>());
        let block = last.block.expect("drafts at its end");
        assert_eq!(block.positions, (40..48).collect::<Vec<_>>());
    }

    /// A prefill in chunks of 16 of a 64-token prompt that saves a snapshot at 24 (a window of
    /// 16): the snapshot's window, 8..24, and the end's, 48..64 — farther than a window from the
    /// snapshot, a fresh one — are committed, and nothing between them: the chunk 32..48 runs no
    /// pass at all.
    #[test]
    fn a_prompt_commits_each_snapshot_s_window_and_its_end_s() {
        let drafter = BlockDrafter {
            window: 16,
            ..DRAFTER
        };
        let mut contexts = DraftContexts::new(16);
        contexts.plan("a", 0, 64, &[24]).expect("a plan");
        let chunk = |from: u32, to: u32| Step {
            req_ids: vec!["a".into()],
            input_ids: (from + 100..to + 100).collect(),
            positions: (from..to).collect(),
            cu: vec![0, to - from],
            history: vec![64],
            sampled: SampledRows::of(&[0, to - from], |_| false).expect("rows"),
        };
        let commits = |from: u32, to: u32| {
            (drafter.passes(&chunk(from, to).rows(), &plan(&["a"]), &contexts))
                .expect("a plan")
                .map(|p| p.commit.positions)
        };
        assert_eq!(commits(0, 16), Some((8..16).collect()));
        assert_eq!(commits(16, 32), Some((16..24).collect()));
        assert_eq!(commits(32, 48), None);
        assert_eq!(commits(48, 64), Some((48..64).collect()));
    }

    /// A prompt resumed from a hit at 32 whose end is within a window of it continues the
    /// restored window: it commits every row it replays, from its first.
    #[test]
    fn a_hit_commits_its_whole_replay_onto_its_restored_window() {
        let drafter = BlockDrafter {
            window: 16,
            ..DRAFTER
        };
        let mut contexts = DraftContexts::new(16);
        contexts.plan("a", 32, 40, &[]).expect("a plan");
        assert!(contexts.continues_restored_window("a", 32));
        let step = Step {
            req_ids: vec!["a".into()],
            input_ids: (132..140).collect(),
            positions: (32..40).collect(),
            cu: vec![0, 8],
            history: vec![40],
            sampled: SampledRows::of(&[0, 8], |_| false).expect("rows"),
        };
        let passes = (drafter.passes(&step.rows(), &plan(&["a"]), &contexts))
            .expect("a plan")
            .expect("commits");
        assert_eq!(passes.commit.positions, (32..40).collect::<Vec<_>>());
        assert!(passes.block.is_some(), "it drafts at its end");
    }

    /// A sequence whose window was restored at 32 drafts nothing at its prompt's end (its first
    /// token does not wait for its rings), and its first step past the prompt rebuilds its rings
    /// over the restored positions still in the window, then drafts.
    #[test]
    fn a_restored_window_s_rings_are_rebuilt_past_the_prompt_then_draft() {
        let drafter = BlockDrafter {
            window: 16,
            ..DRAFTER
        };
        let mut contexts = DraftContexts::new(16);
        contexts.plan("a", 32, 40, &[]).expect("a plan");
        contexts.restore("a", 32);
        let end = Step {
            req_ids: vec!["a".into()],
            input_ids: (132..140).collect(),
            positions: (32..40).collect(),
            cu: vec![0, 8],
            history: vec![40],
            sampled: SampledRows::of(&[0, 8], |_| false).expect("rows"),
        };
        assert_eq!(
            contexts.take_rebuild("a", 32, 40),
            None,
            "not in its prompt"
        );
        let passes = (drafter.passes(&end.rows(), &plan(&["a"]), &contexts))
            .expect("a plan")
            .expect("commits");
        assert_eq!(passes.commit.positions, (32..40).collect::<Vec<_>>());
        assert_eq!(passes.block, None, "its prompt's end drafts nothing");

        let decode = Step {
            req_ids: vec!["a".into()],
            input_ids: vec![7],
            positions: vec![40],
            cu: vec![0, 1],
            history: vec![41],
            sampled: SampledRows::of(&[0, 1], |_| false).expect("rows"),
        };
        assert_eq!(contexts.take_rebuild("a", 40, 41), Some(25..32));
        assert_eq!(contexts.take_rebuild("a", 40, 41), None, "rebuilt once");
        let passes = (drafter.passes(&decode.rows(), &plan(&["a"]), &contexts))
            .expect("a plan")
            .expect("commits");
        assert!(passes.block.expect("drafts").drafts);
    }

    /// A sequence admitted at a hit at 32, preempted, and resumed from 0 over its prompt and reply
    /// (44 tokens) replays from where it resumes: its first plan refuses rows before its own
    /// replay, and planned again on resuming, it commits the window before its new end.
    #[test]
    fn a_resumed_sequence_replays_from_where_it_resumes() {
        let drafter = BlockDrafter {
            window: 16,
            ..DRAFTER
        };
        let mut contexts = DraftContexts::new(16);
        contexts.plan("a", 32, 40, &[]).expect("a plan");
        let chunk = |from: u32, to: u32| Step {
            req_ids: vec!["a".into()],
            input_ids: (from + 100..to + 100).collect(),
            positions: (from..to).collect(),
            cu: vec![0, to - from],
            history: vec![44],
            sampled: SampledRows::of(&[0, to - from], |_| false).expect("rows"),
        };
        let replay = |contexts: &DraftContexts, from, to| {
            (drafter.passes(&chunk(from, to).rows(), &plan(&["a"]), contexts))
                .map(|p| p.map(|p| p.commit.positions))
        };
        assert_eq!(
            replay(&contexts, 0, 16),
            Err(DraftContextError::DispatchOutsideReplay),
            "the plan of its first replay refuses its second"
        );
        contexts.release("a");
        contexts.plan("a", 0, 44, &[]).expect("a plan");
        assert_eq!(replay(&contexts, 0, 16), Ok(None));
        assert_eq!(replay(&contexts, 16, 32), Ok(Some((28..32).collect())));
        assert_eq!(replay(&contexts, 32, 44), Ok(Some((32..44).collect())));
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
            .passes(&step.rows(), &plan(&["a"]), &DraftContexts::new(2048))
            .expect("a plan")
            .expect("planned");
        assert!(passes.commit.cycle);
        assert_eq!(passes.commit.hidden_rows, [(0, 0, 3)]);
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
            .passes(&last.rows(), &plan(&["a"]), &DraftContexts::new(2048))
            .expect("a plan")
            .expect("planned");
        assert!(!passes.commit.cycle);
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
            .passes(
                &decoding.rows(),
                &plan(&["a", "b"]),
                &DraftContexts::new(2048),
            )
            .expect("a plan"))
        .expect("planned");
        assert!(!passes.commit.cycle, "a decode step commits every row");
        assert_eq!(passes.commit.positions, [30, 12]);
        assert_eq!(
            passes.block, None,
            "two sequences' blocks overflow 8 verify rows"
        );
        assert_eq!(
            DRAFTER
                .passes(&decoding.rows(), &plan(&[]), &DraftContexts::new(2048))
                .expect("a plan"),
            None
        );
    }
}

// SPDX-License-Identifier: Apache-2.0

//! ⭐⭐⭐ A WAVE OF LONG PROMPTS ON THE SHARED-WRITE-SLOT POOL ADMITS EVERY REQUEST — the serve-level
//! property #694 broke, driven through the real `Scheduler` with the worker's KV report modelled the way
//! the spyre worker computes it.
//!
//! In the lib, not `tests/`: CI runs this crate as `cargo test -p scratchy-serving-scheduler --lib`.
//!
//! MEASURED on the 2026-10-19 sweep (granite-3.1-2b fp16, `--max-num-seqs 4 --max-model-len 12288`, 192
//! blocks of 256 offered, 4 × ~10300-token prompts per wave): in every wave ONE request waited a whole
//! request lifetime for its blocks (ttft 136-142 s against 9-31 s for its batch-mates), its admission
//! lookup repeated 245 times, once per decode step, until a running request finished. At
//! `--max-num-seqs 8` (384 blocks) it was the same: one in eight, 254-308 s.
//!
//! THE ARITHMETIC THE TEST HOLDS: a 10300-token prompt with 256 generated tokens needs
//! `ceil(10556 / 256) = 42` blocks plus ONE for the shared write page, so four of them need at most 172 of
//! the 192 — the wave fits. It starved because a request whose prompt was still being prefilled was charged
//! a whole scheduler step's budget PAST a reported span that already reached its prompt's end (~50 blocks
//! held for life, `4 × 50 > 192`), and the request preempted for it was re-admitted from the dead
//! request's report instead of from scratch.

use scratchy_core_common::kv::{CacheableTokens, KvExtent, KvSlotSpan};
use scratchy_core_common::{Request, RequestStatus, SamplingParams};
use scratchy_core_config::SchedulerConfig;

use super::core::{KVCacheManagerOps, Scheduler, SimpleBlockTracker};
use super::interface::SchedulerInterface;

const BLOCK: usize = 256;
/// The sweep's prompt length (vllm bench's conc-10300 set; the serve recorded 10296-10301).
const PROMPT: usize = 10300;
const OUT: u32 = 256;
const ROWS: usize = 4;
/// `--max-model-len 12288` at 4 launch rows: 48 pages a row, the pool's 192 offered blocks.
const POOL_BLOCKS: usize = 192;
const MAX_MODEL_LEN: usize = 12288;
/// The token budget the spyre serve ran at (the generic default — the worker names no prefill ceiling).
const BUDGET: usize = 2048;

/// HOW THE WORKER DESCRIBES ITS KV after a step, as the scheduler receives it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WorkerReport {
    /// Every backend whose keys sit at their token positions (cuda, metal): it reports nothing.
    TokenAddressed,
    /// The spyre paged pool (`spyre_exec.rs`, `ReqState::kv_extent`): EVERY live request — the worker
    /// keeps every request it has been sent until the scheduler names it finished — is reported at the
    /// pool's future write bound, `max over live of max(history end, prompt)`, with its contiguous prefix
    /// as the cacheable run.
    SharedWriteSlot,
}

fn wave_request(id: usize) -> Request {
    // Distinct prompts, so no request is served from another's cached prefix and every one prefills in
    // full — the shape that needs the most blocks, and the shape that starved.
    let base = (id as u32 + 1) * 1_000_000;
    Request::new(
        format!("r{id}"),
        (base..base + PROMPT as u32).collect(),
        SamplingParams {
            max_tokens: Some(OUT),
            ..Default::default()
        },
        id as f64,
        0,
        0,
        None,
    )
}

/// The spyre worker's per-request history end after a step: the prompt positions it has computed while
/// prefilling, then one slot per decode step at the shared write slot — every live row writes there, so
/// a request that has started decoding sits at the pool's write cursor.
fn history_end(r: &Request, pool_cursor: u32) -> u32 {
    let computed = r.num_computed_tokens;
    if computed < r.num_prompt_tokens {
        computed
    } else {
        pool_cursor.max(computed)
    }
}

/// One wave, driven to completion: per request, the step it was first given a token and the step it
/// finished; and how many preemptions the wave took.
struct Wave {
    requests: Vec<(String, usize, usize)>,
    preemptions: usize,
}

fn run_wave(report: WorkerReport) -> Wave {
    let cfg = SchedulerConfig {
        max_num_batched_tokens: BUDGET,
        max_num_seqs: ROWS,
        enable_chunked_prefill: true,
        long_prefill_token_threshold: 0,
        ..Default::default()
    };
    let tracker = SimpleBlockTracker::with_caching(POOL_BLOCKS, BLOCK);
    let tracker: Box<dyn KVCacheManagerOps> = match report {
        WorkerReport::TokenAddressed => Box::new(tracker),
        WorkerReport::SharedWriteSlot => Box::new(tracker.sharing_one_write_slot()),
    };
    let mut sched = Scheduler::new(&cfg, MAX_MODEL_LEN, tracker);
    let ids: Vec<String> = (0..ROWS).map(|i| format!("r{i}")).collect();
    for i in 0..ROWS {
        sched.add_request(wave_request(i));
    }

    // Every request the worker has been sent and not yet told is finished — what it reports over.
    let mut worker_live: Vec<String> = Vec::new();
    let mut first_token: Vec<Option<usize>> = vec![None; ROWS];
    let mut finished: Vec<Option<usize>> = vec![None; ROWS];
    let mut pool_cursor: u32 = 0;
    let mut step = 0usize;
    let mut preemptions = 0usize;
    while sched.has_unfinished_requests() {
        step += 1;
        assert!(
            step < 4_000,
            "the wave did not drain in {step} steps — a request that can never be admitted"
        );
        let out = sched.schedule();
        preemptions += out.preempted_req_ids.as_ref().map_or(0, |p| p.len());
        for fid in &out.finished_req_ids {
            worker_live.retain(|id| id != fid);
        }
        for nr in &out.scheduled_new_reqs {
            if !worker_live.contains(&nr.req_id) {
                worker_live.push(nr.req_id.clone());
            }
        }
        // The engine's part: a request whose chunk reached the end of its tokens samples one.
        let mut decoded_any = false;
        for id in out.num_scheduled_tokens.keys() {
            let r = sched.get_request(id).expect("scheduled ⇒ known");
            if r.num_computed_tokens as usize >= r.num_tokens() {
                let i = ids.iter().position(|x| x == id).expect("a wave request");
                first_token[i].get_or_insert(step);
                if r.num_prompt_tokens as usize != r.num_tokens() {
                    decoded_any = true;
                }
                sched.append_output_tokens(id, &[7]);
                if sched.get_request(id).expect("known").num_output_tokens() >= OUT as usize {
                    sched.finish_requests(&[id.as_str()], RequestStatus::FinishedLengthCapped);
                    finished[i] = Some(step);
                }
            }
        }
        if decoded_any {
            pool_cursor += 1;
        }
        // The worker's report, over every request it still holds.
        if report == WorkerReport::SharedWriteSlot {
            let reach = worker_live
                .iter()
                .filter_map(|id| sched.get_request(id))
                .map(|r| history_end(r, pool_cursor).max(r.num_prompt_tokens))
                .max()
                .unwrap_or(0);
            pool_cursor = pool_cursor.max(reach);
            for id in &worker_live {
                let Some(r) = sched.get_request(id) else {
                    continue;
                };
                let cacheable = r.num_computed_tokens.min(r.num_prompt_tokens);
                let extent = KvExtent::new(KvSlotSpan::new(reach), CacheableTokens::new(cacheable));
                sched.set_kv_extent(id, extent);
            }
            sched.set_kv_pool_reach(KvSlotSpan::new(reach));
        }
    }
    let requests = ids
        .into_iter()
        .zip(first_token)
        .zip(finished)
        .map(|((id, first), fin)| {
            (
                id,
                first.expect("every request was given a first token"),
                fin.expect("every request finished"),
            )
        })
        .collect();
    Wave {
        requests,
        preemptions,
    }
}

/// ⭐ THE PROPERTY: four long prompts that fit the pool together all start decoding before ANY of them
/// finishes, and none is preempted. A request whose first token waits for a batch-mate's last one is the
/// starvation; a preemption in a wave that fits is a prompt recomputed for blocks nobody needed.
#[test]
fn a_wave_that_fits_the_pool_gives_every_request_its_first_token_before_any_finishes() {
    // The pool really is big enough: own pages for prompt + output, plus the one shared write page.
    let need = (PROMPT + OUT as usize).div_ceil(BLOCK) + 1;
    assert!(
        ROWS * need <= POOL_BLOCKS,
        "the fixture must be a wave that FITS: {ROWS} × {need} blocks against {POOL_BLOCKS}"
    );
    for report in [WorkerReport::TokenAddressed, WorkerReport::SharedWriteSlot] {
        let wave = run_wave(report);
        let first_finish = (wave.requests.iter())
            .map(|(_, _, fin)| *fin)
            .min()
            .expect("a wave");
        for (id, first, _) in &wave.requests {
            assert!(
                *first < first_finish,
                "{report:?}: {id} got its first token at step {first}, after a batch-mate FINISHED at step \
                 {first_finish} — it waited out a whole request for blocks the pool had (#694). Wave: {:?}",
                wave.requests
            );
        }
        assert_eq!(
            wave.preemptions, 0,
            "{report:?}: a wave that fits the pool preempted {} time(s) — each one throws away a prefilled \
             prompt to free blocks that were charged past what the request holds (#694)",
            wave.preemptions
        );
    }
}

/// ⛔ A PREEMPTED REQUEST HOLDS NO KV, so nothing the worker said about the KV it USED to hold may size its
/// re-admission. Its blocks were freed at preemption; a report naming them — the one it carried, or one
/// the worker sends after (it still holds the request: it is told only about FINISHED ones) — sizes the
/// fresh request by a span it no longer occupies.
///
/// A token-addressed tracker on purpose: the extent alone is under test, with no shared write slot in the
/// arithmetic. Three 16-token blocks; two 16-token prompts take one each, and the first decode token of
/// either, at a reported span of 40 slots, needs three.
#[test]
fn a_preempted_request_is_readmitted_as_a_request_that_holds_nothing() {
    let cfg = SchedulerConfig {
        max_num_batched_tokens: 64,
        max_num_seqs: 2,
        enable_chunked_prefill: true,
        long_prefill_token_threshold: 0,
        ..Default::default()
    };
    let mut sched = Scheduler::new(
        &cfg,
        8192,
        Box::new(SimpleBlockTracker::with_caching(3, 16)),
    );
    let req = |id: &str, base: u32| {
        Request::new(
            id.into(),
            (base..base + 16).collect(),
            SamplingParams {
                max_tokens: Some(8),
                ..Default::default()
            },
            0.0,
            0,
            0,
            None,
        )
    };
    sched.add_request(req("a", 0));
    sched.add_request(req("b", 1000));
    let first = sched.schedule();
    assert_eq!(
        first.scheduled_new_reqs.len(),
        2,
        "both prompts fit: one block each"
    );
    sched.append_output_tokens("a", &[1]);
    sched.append_output_tokens("b", &[2]);
    let wide = KvExtent::new(KvSlotSpan::new(40), CacheableTokens::new(16));
    sched.set_kv_extent("a", wide);
    sched.set_kv_extent("b", wide);
    let out = sched.schedule();
    let preempted = out
        .preempted_req_ids
        .expect("a's decode at a 40-slot span needs three blocks, so the tail is preempted");
    assert!(preempted.contains("b"), "the tail is preempted");
    // The worker still holds `b`, so its next report names it again.
    sched.set_kv_extent("b", wide);

    // `a` finishes; all three blocks are free. `b` is a fresh request of 17 tokens: two blocks.
    sched.finish_requests(&["a"], RequestStatus::FinishedStopped);
    let out = sched.schedule();
    let cr = &out.scheduled_cached_reqs;
    let at = cr.req_ids.iter().position(|id| id == "b").unwrap_or_else(|| {
        panic!(
            "b was not re-admitted into an empty three-block pool — it was sized by the 40-slot span its \
             freed blocks used to have (#694); kv_extent = {:?}",
            sched.get_request("b").and_then(|r| r.kv_extent)
        )
    });
    assert!(cr.resumed_req_ids.contains("b"), "b comes back as a resume");
    let blocks = cr.new_block_ids[at]
        .as_ref()
        .and_then(|g| g.first())
        .map_or(0, |g| g.len());
    assert_eq!(
        blocks, 2,
        "a resumed request holds no keys: 17 tokens = 2 blocks, not the span of KV that was freed"
    );
}

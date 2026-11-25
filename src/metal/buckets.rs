//! A dataset cell's requests by prompt length: the summary map's columns. Each request's prompt
//! tokens, as the server counted them, put it in one of its dataset's [`Buckets`]; a square compares scratchy's
//! median with the rival's over the requests of one bucket, as the random-token map compares
//! them at one prompt length.

use super::data::{RequestTiming, TaskCell};
use super::numbers::median;

/// A dataset's prompt-length buckets, in tokens: each its label and its upper bound, the last
/// unbounded. Each labelled by where it starts, as the random-token map labels a column by its
/// one prompt size: a range would not fit over its square.
pub type Buckets = &'static [(&'static str, u32)];

/// MuSiQue's: four, as the random-token map has four prompt sizes, at about the quartiles of its
/// prompts (most run 2k to 3.3k tokens): split at round 512s, a quarter of the squares had no
/// request; split here, about one in twenty.
pub const MUSIQUE: Buckets = &[
    ("<2.3k", 2300),
    ("2.3k+", 2550),
    ("2.55k+", 2850),
    ("2.85k+", u32::MAX),
];

/// One bucket of every prompt, for a dataset with no buckets of its own yet: a square is its
/// cell's whole median.
pub const WHOLE: Buckets = &[("all prompts", u32::MAX)];

/// The bucket of `buckets` a prompt of `tokens` falls in: its index.
pub fn bucket(buckets: Buckets, tokens: u32) -> usize {
    buckets
        .iter()
        .position(|&(_, below)| tokens < below)
        .unwrap_or(buckets.len() - 1)
}

/// A cell's requests' `timing` in bucket `at` of `buckets`, in the order sent: those with a
/// prompt count and a value; none where the cell kept no prompt counts, as cells from before it
/// did.
pub fn in_bucket(c: &TaskCell, buckets: Buckets, timing: RequestTiming, at: usize) -> Vec<f64> {
    let (Some(prompts), Some(values)) = (c.prompt_tokens.as_deref(), values(c, timing)) else {
        return Vec::new();
    };
    (prompts.iter().zip(values))
        .filter_map(|(p, v)| Some((p.filter(|&p| bucket(buckets, p) == at)?, (*v)?)))
        .map(|(_, v)| v)
        .collect()
}

/// How many of a cell's requests bucket `at` of `buckets` holds, timed or not: what tells a bucket
/// with no request from one whose requests kept no `timing` (a TPOT of a one-token answer).
pub fn sent_in(c: &TaskCell, buckets: Buckets, at: usize) -> usize {
    (c.prompt_tokens.iter().flatten().flatten())
        .filter(|&&p| bucket(buckets, p) == at)
        .count()
}

/// The median of a cell's requests' `timing` in bucket `at` of `buckets`; None with none there.
pub fn median_in(c: &TaskCell, buckets: Buckets, timing: RequestTiming, at: usize) -> Option<f64> {
    median(in_bucket(c, buckets, timing, at))
}

/// Whether a cell kept its requests' prompt counts, so it can be read by prompt length at all.
pub fn bucketed(c: &TaskCell) -> bool {
    c.prompt_tokens
        .as_ref()
        .is_some_and(|p| p.iter().any(Option::is_some))
}

/// What a right answer typically cost in bucket `at` of `buckets`, for `timing`: the median, over
/// the bucket's correctly answered requests, of each one's own time times its own energy over that
/// time's span ([`joules`]: prefill joules for TTFT, decode joules for TPOT, the whole request's
/// for end-to-end). Every number is one request's: no sum, no count of right answers divided into
/// a total. A request missing its time or its energy (a one-token answer has no TPOT) is not
/// counted. None where the cell kept no such energy, no prompt counts or no scores, or no request
/// there was right.
pub fn time_joules_per_right_in(
    c: &TaskCell,
    buckets: Buckets,
    timing: RequestTiming,
    at: usize,
) -> Option<f64> {
    let (prompts, joules, right, times) = (
        c.prompt_tokens.as_deref()?,
        joules(c, timing)?,
        c.task_right.as_deref()?,
        values(c, timing)?,
    );
    let costs: Vec<f64> = (prompts.iter().zip(joules).zip(right).zip(times))
        .filter(|(((p, _), r), _)| p.is_some_and(|p| bucket(buckets, p) == at) && **r == Some(true))
        .filter_map(|(((_, j), _), t)| Some((*t)? * (*j)?))
        .collect();
    median(costs)
}

/// How many of a cell's requests in bucket `at` of `buckets` were right: what its joules a
/// correct answer there is over.
pub fn rights_in(c: &TaskCell, buckets: Buckets, at: usize) -> usize {
    let (Some(prompts), Some(right)) = (c.prompt_tokens.as_deref(), c.task_right.as_deref()) else {
        return 0;
    };
    (prompts.iter().zip(right))
        .filter(|(p, r)| p.is_some_and(|p| bucket(buckets, p) == at) && **r == Some(true))
        .count()
}

fn values(c: &TaskCell, timing: RequestTiming) -> Option<&[Option<f64>]> {
    match timing {
        RequestTiming::Ttft => c.ttfts_ms.as_deref(),
        RequestTiming::Tpot => c.tpots_ms.as_deref(),
        RequestTiming::E2el => c.e2els_ms.as_deref(),
    }
}

/// The energy each request spent over the span `timing` times, in the order sent: its prefill
/// joules for TTFT (up to its first token), its decode joules for TPOT (after it), its whole
/// joules for end-to-end. The one pairing of a time with its energy every figure reads.
pub fn joules(c: &TaskCell, timing: RequestTiming) -> Option<&[Option<f64>]> {
    match timing {
        RequestTiming::Ttft => c.task_prefill_joules.as_deref(),
        RequestTiming::Tpot => c.task_decode_joules.as_deref(),
        RequestTiming::E2el => c.task_joules.as_deref(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a right answer typically cost in a bucket: the median, over its correctly answered
    /// requests, of each one's own time times its own joules — a wrong answer's not counted, nor a
    /// right one missing its time; none where none was right, or the cell kept no joules a request.
    #[test]
    fn a_buckets_cost_of_a_right_answer_is_the_median_of_each_ones_own_product() {
        let mut c: TaskCell = serde_json::from_value(serde_json::json!({
            "workload": "rag", "conc": 2,
            "prompt_tokens": [2000, 2000, 2000, 2000, 2000, 3000],
            "task_right": [true, false, true, true, true, false],
            "task_joules": [4.0, 6.0, 2.0, 5.0, 1.0, 3.0],
            "e2els_ms": [100.0, 300.0, 1000.0, null, 5000.0, 200.0],
        }))
        .unwrap();
        // The right answers with a time: 4×100, 2×1000, 1×5000 — the median is 2,000. The wrong
        // one's 6×300 and the untimed right one are not counted.
        assert_eq!(
            time_joules_per_right_in(&c, MUSIQUE, RequestTiming::E2el, 0),
            Some(2000.0),
            "each right answer's own product, the median of them"
        );
        assert_eq!(
            time_joules_per_right_in(&c, MUSIQUE, RequestTiming::E2el, 3),
            None,
            "none right"
        );
        assert_eq!(
            time_joules_per_right_in(&c, MUSIQUE, RequestTiming::E2el, 1),
            None,
            "no request there"
        );
        c.task_joules = None;
        assert_eq!(
            time_joules_per_right_in(&c, MUSIQUE, RequestTiming::E2el, 0),
            None,
            "no joules a request"
        );
    }

    fn cell(prompts: &[Option<u32>], ttfts: &[Option<f64>], tpots: &[Option<f64>]) -> TaskCell {
        serde_json::from_value(serde_json::json!({
            "workload": "rag", "conc": 1, "completed": prompts.len(),
            "prompt_tokens": prompts, "ttfts_ms": ttfts, "tpots_ms": tpots,
            "e2els_ms": ttfts.iter().map(|t| t.map(|t| t * 2.0)).collect::<Vec<_>>(),
        }))
        .unwrap()
    }

    /// A prompt goes in the first bucket it is under; one past the last bound in the last.
    #[test]
    fn a_prompt_falls_in_one_bucket() {
        assert_eq!(bucket(MUSIQUE, 1500), 0);
        assert_eq!(bucket(MUSIQUE, 2300), 1, "a bound starts the next bucket");
        assert_eq!(bucket(MUSIQUE, 2700), 2);
        assert_eq!(bucket(MUSIQUE, 2850), 3);
        assert_eq!(bucket(MUSIQUE, 9000), 3, "the last is unbounded");
        assert_eq!(bucket(WHOLE, 9000), 0, "one bucket holds every prompt");
    }

    /// A bucket's median is over its own requests, those with a prompt count and a value: a
    /// failed request, or one with no count, is in none.
    #[test]
    fn a_buckets_median_is_over_its_own_requests() {
        let c = cell(
            &[Some(1900), Some(2900), Some(2950), None, Some(3000)],
            &[Some(100.0), Some(300.0), Some(500.0), Some(900.0), None],
            &[Some(10.0), Some(20.0), Some(40.0), Some(90.0), Some(50.0)],
        );
        assert_eq!(median_in(&c, MUSIQUE, RequestTiming::Ttft, 0), Some(100.0));
        assert_eq!(
            median_in(&c, MUSIQUE, RequestTiming::Ttft, 3),
            Some(400.0),
            "300 and 500"
        );
        assert_eq!(
            median_in(&c, MUSIQUE, RequestTiming::Tpot, 3),
            Some(40.0),
            "20, 40 and 50"
        );
        assert_eq!(median_in(&c, MUSIQUE, RequestTiming::E2el, 0), Some(200.0));
        assert_eq!(
            median_in(&c, MUSIQUE, RequestTiming::Ttft, 1),
            None,
            "none that long"
        );
        assert!(bucketed(&c));
    }

    /// A bucket's requests sent, timed or not: what tells no request this long from requests
    /// none of whose timing was kept.
    #[test]
    fn a_bucket_counts_its_requests_sent() {
        let c = cell(
            &[Some(2600), Some(2700), Some(3000), None],
            &[Some(100.0), Some(200.0), Some(300.0), Some(400.0)],
            &[None, None, Some(30.0), None],
        );
        assert_eq!(sent_in(&c, MUSIQUE, 2), 2);
        let timed = in_bucket(&c, MUSIQUE, RequestTiming::Tpot, 2);
        assert!(timed.is_empty(), "none timed");
        assert_eq!(sent_in(&c, MUSIQUE, 3), 1);
        assert_eq!(sent_in(&c, MUSIQUE, 0), 0);
    }

    /// A cell from before cells kept prompt counts has no bucket of anything.
    #[test]
    fn a_cell_with_no_prompt_counts_has_no_buckets() {
        let old: TaskCell = serde_json::from_value(serde_json::json!({
            "workload": "rag", "conc": 1, "completed": 2, "ttfts_ms": [100.0, 200.0],
        }))
        .unwrap();
        assert!(!bucketed(&old));
        assert!(in_bucket(&old, MUSIQUE, RequestTiming::Ttft, 0).is_empty());
    }
}

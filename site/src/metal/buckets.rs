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

/// Every dataset's buckets, the same four, their bounds 500 prompt tokens apart: under 2.4k, 2.9k
/// and 3.4k, and over. They split MuSiQue's prompts from its 5th to its 95th percentile (1.88k to
/// 3.82k on its dev set) evenly, about 28%, 37%, 22% and 12% of them; at the matrix's 48 requests a
/// cell, 99% of squares rest on 3 or more a side. One set for every dataset, so a column means the
/// same prompt length on each.
pub const BUCKETS: Buckets = &[
    ("<2.4k", 2400),
    ("2.4k+", 2900),
    ("2.9k+", 3400),
    ("3.4k+", u32::MAX),
];

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

fn values(c: &TaskCell, timing: RequestTiming) -> Option<&[Option<f64>]> {
    match timing {
        RequestTiming::Ttft => c.ttfts_ms.as_deref(),
        RequestTiming::Tpot => c.tpots_ms.as_deref(),
        RequestTiming::E2el => c.e2els_ms.as_deref(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(prompts: &[Option<u32>], ttfts: &[Option<f64>], tpots: &[Option<f64>]) -> TaskCell {
        serde_json::from_value(serde_json::json!({
            "workload": "rag", "conc": 1, "completed": prompts.len(),
            "prompt_tokens": prompts, "ttfts_ms": ttfts, "tpots_ms": tpots,
            "e2els_ms": ttfts.iter().map(|t| t.map(|t| t * 2.0)).collect::<Vec<_>>(),
        }))
        .unwrap()
    }

    /// Every dataset's buckets are the same four, 500 prompt tokens apart: under 2.4k, 2.9k and
    /// 3.4k, and over. A prompt goes in the first it is under, one past the last bound in the last.
    #[test]
    fn a_prompt_falls_in_one_bucket() {
        let labels: Vec<&str> = BUCKETS.iter().map(|&(l, _)| l).collect();
        assert_eq!(labels, ["<2.4k", "2.4k+", "2.9k+", "3.4k+"]);
        let bounds: Vec<u32> = BUCKETS.iter().map(|&(_, b)| b).take(3).collect();
        let even = bounds.windows(2).all(|w| w[1] - w[0] == 500);
        assert!(even, "evenly spaced");
        assert_eq!(bucket(BUCKETS, 2399), 0);
        assert_eq!(bucket(BUCKETS, 2400), 1, "a bound starts the next bucket");
        assert_eq!(bucket(BUCKETS, 2899), 1);
        assert_eq!(bucket(BUCKETS, 2900), 2);
        assert_eq!(bucket(BUCKETS, 3399), 2);
        assert_eq!(bucket(BUCKETS, 3400), 3);
        assert_eq!(bucket(BUCKETS, 9000), 3, "the last is unbounded");
    }

    /// A bucket's requests sent, timed or not: what tells no request this long from requests
    /// none of whose timing was kept.
    #[test]
    fn a_bucket_counts_its_requests_sent() {
        let c = cell(
            &[Some(3000), Some(3100), Some(3500), None],
            &[Some(100.0), Some(200.0), Some(300.0), Some(400.0)],
            &[None, None, Some(30.0), None],
        );
        assert_eq!(sent_in(&c, BUCKETS, 2), 2);
        let timed = in_bucket(&c, BUCKETS, RequestTiming::Tpot, 2);
        assert!(timed.is_empty(), "none timed");
        assert_eq!(sent_in(&c, BUCKETS, 3), 1);
        assert_eq!(sent_in(&c, BUCKETS, 0), 0);
    }

    /// A bucket's median is over its own requests, those with a prompt count and a value: a
    /// failed request, or one with no count, is in none.
    #[test]
    fn a_buckets_median_is_over_its_own_requests() {
        let c = cell(
            &[Some(3000), Some(3100), Some(3200), None, Some(3500)],
            &[Some(100.0), Some(300.0), Some(500.0), Some(900.0), None],
            &[Some(10.0), Some(20.0), Some(40.0), Some(90.0), Some(50.0)],
        );
        assert_eq!(
            median_in(&c, BUCKETS, RequestTiming::Ttft, 2),
            Some(300.0),
            "100, 300 and 500"
        );
        assert_eq!(
            median_in(&c, BUCKETS, RequestTiming::Tpot, 2),
            Some(20.0),
            "10, 20 and 40"
        );
        let failed = median_in(&c, BUCKETS, RequestTiming::Ttft, 3);
        assert_eq!(failed, None, "its one failed");
        assert_eq!(median_in(&c, BUCKETS, RequestTiming::Tpot, 3), Some(50.0));
        assert_eq!(median_in(&c, BUCKETS, RequestTiming::E2el, 2), Some(600.0));
        assert_eq!(
            median_in(&c, BUCKETS, RequestTiming::Ttft, 1),
            None,
            "none that long"
        );
        assert!(bucketed(&c));
    }

    /// A cell from before cells kept prompt counts has no bucket of anything.
    #[test]
    fn a_cell_with_no_prompt_counts_has_no_buckets() {
        let old: TaskCell = serde_json::from_value(serde_json::json!({
            "workload": "rag", "conc": 1, "completed": 2, "ttfts_ms": [100.0, 200.0],
        }))
        .unwrap();
        assert!(!bucketed(&old));
        assert!(in_bucket(&old, BUCKETS, RequestTiming::Ttft, 0).is_empty());
    }
}

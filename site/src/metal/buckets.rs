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

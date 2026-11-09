//! A dataset cell's requests by prompt length: the summary map's columns. Each request's prompt
//! tokens, as the server counted them, put it in one of [`BUCKETS`]; a square compares scratchy's
//! median with the rival's over the requests of one bucket, as the random-token map compares
//! them at one prompt length.

use super::data::{RequestTiming, TaskCell};
use super::numbers::median;

/// Prompt-length buckets, in tokens: each its label and its upper bound, the last unbounded.
/// Two, split at MuSiQue's middle prompt (about 2.8k tokens on the published runs), so a quick
/// run's cell of 8 requests puts about 4 in each: more buckets leave most of them one or two
/// requests, or none.
pub const BUCKETS: [(&str, u32); 2] = [("<2.8k", 2800), ("2.8k+", u32::MAX)];

/// The fewest requests a bucket's median is read from: one or two are no reading of a bucket, a
/// single slow answer turning its square from blue to red.
pub const FLOOR: usize = 3;

/// The bucket a prompt of `tokens` falls in: its index in [`BUCKETS`].
pub fn bucket(tokens: u32) -> usize {
    BUCKETS
        .iter()
        .position(|&(_, below)| tokens < below)
        .unwrap_or(BUCKETS.len() - 1)
}

/// A cell's requests' `timing` in `bucket`, in the order sent: those with a prompt count and a
/// value; none where the cell kept no prompt counts, as cells from before it did.
pub fn in_bucket(c: &TaskCell, timing: RequestTiming, bucket_at: usize) -> Vec<f64> {
    let (Some(prompts), Some(values)) = (c.prompt_tokens.as_deref(), values(c, timing)) else {
        return Vec::new();
    };
    (prompts.iter().zip(values))
        .filter_map(|(p, v)| Some((p.filter(|&p| bucket(p) == bucket_at)?, (*v)?)))
        .map(|(_, v)| v)
        .collect()
}

/// The median of a cell's requests' `timing` in `bucket_at`; None with fewer than [`FLOOR`]
/// there.
pub fn median_in(c: &TaskCell, timing: RequestTiming, bucket_at: usize) -> Option<f64> {
    let vs = in_bucket(c, timing, bucket_at);
    (vs.len() >= FLOOR).then(|| median(vs)).flatten()
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

    /// Two buckets, split at MuSiQue's middle prompt: a quick run's 8 requests a cell fall
    /// about 4 and 4.
    #[test]
    fn a_prompt_falls_in_one_bucket() {
        assert_eq!(BUCKETS.len(), 2);
        assert_eq!(bucket(1500), 0);
        assert_eq!(bucket(2799), 0);
        assert_eq!(bucket(2800), 1, "a bound starts the next bucket");
        assert_eq!(bucket(9000), 1, "the last is unbounded");
    }

    /// A bucket of fewer than [`FLOOR`] requests has no median: one or two requests are no
    /// reading of a bucket.
    #[test]
    fn a_bucket_under_the_floor_has_no_median() {
        let c = cell(
            &[Some(2000), Some(2100), Some(3000), Some(3100), Some(3200)],
            &[Some(100.0), Some(200.0), Some(300.0), Some(500.0), Some(700.0)],
            &[None; 5],
        );
        assert_eq!(FLOOR, 3);
        assert_eq!(median_in(&c, RequestTiming::Ttft, 0), None, "two requests");
        assert_eq!(median_in(&c, RequestTiming::Ttft, 1), Some(500.0), "three");
        assert_eq!(in_bucket(&c, RequestTiming::Ttft, 0).len(), 2, "still counted");
    }

    /// A bucket's median is over its own requests, those with a prompt count and a value: a
    /// failed request, or one with no count, is in none.
    #[test]
    fn a_buckets_median_is_over_its_own_requests() {
        let c = cell(
            &[Some(1900), Some(2000), Some(2100), Some(2900), Some(2950), None, Some(3000), Some(3100)],
            &[Some(100.0), Some(150.0), Some(200.0), Some(300.0), Some(500.0), Some(900.0), None, Some(700.0)],
            &[Some(10.0), Some(11.0), Some(12.0), Some(20.0), Some(40.0), Some(90.0), Some(50.0), Some(60.0)],
        );
        assert_eq!(median_in(&c, RequestTiming::Ttft, 0), Some(150.0));
        assert_eq!(
            median_in(&c, RequestTiming::Ttft, 1),
            Some(500.0),
            "300, 500 and 700"
        );
        assert_eq!(
            median_in(&c, RequestTiming::Tpot, 1),
            Some(45.0),
            "20, 40, 50 and 60"
        );
        assert_eq!(median_in(&c, RequestTiming::E2el, 0), Some(300.0));
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
        assert!(in_bucket(&old, RequestTiming::Ttft, 1).is_empty());
    }
}

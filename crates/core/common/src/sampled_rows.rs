// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! The rows a step samples (Python vLLM's `logits_indices`): the one definition of a step's
//! sampled rows the serving engine schedules and a backend's forward defaults to.

/// The rows a step samples: each sequence's last, or — a sequence verifying drafts — every one of
/// its rows, each of which can be its newest. A sequence's sampled rows are its last rows, so
/// sampled row `j` of sequence `i` is step row `cu_seqlens_q[i + 1] - (count_i - j)`. The logits
/// the forward leaves, and every row read off them (argmax, sampler, grammar mask), are these
/// rows', in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampledRows {
    /// Each sampled row's step row, ascending (Python vLLM's `logits_indices`).
    rows: Vec<u32>,
    /// Cumulative sampled rows per sequence, `num_seqs + 1` long, strictly ascending: sequence
    /// `i`'s are `cu[i]..cu[i + 1]`, at least one.
    cu: Vec<u32>,
}

/// A step's sequence of no rows: it has no last row to sample, so no step holds one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmptySequence {
    pub seq: usize,
}

impl std::fmt::Display for EmptySequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sequence {} of the step holds no rows to sample",
            self.seq
        )
    }
}

impl std::error::Error for EmptySequence {}

impl SampledRows {
    /// The sampled rows of the step whose sequences' rows `cu_seqlens_q` bounds: every row of a
    /// sequence `every_row(i)` holds, the last of any other. Refused for a sequence of no rows.
    pub fn of(
        cu_seqlens_q: &[u32],
        every_row: impl Fn(usize) -> bool,
    ) -> Result<Self, EmptySequence> {
        let mut rows = Vec::new();
        let mut cu = vec![0];
        for (i, seq) in cu_seqlens_q.windows(2).enumerate() {
            let (start, end) = (seq[0], seq[1]);
            if end <= start {
                return Err(EmptySequence { seq: i });
            }
            let first = match every_row(i) {
                true => start,
                false => end - 1,
            };
            rows.extend(first..end);
            cu.push(rows.len() as u32);
        }
        Ok(Self { rows, cu })
    }

    /// Each sampled row's step row, ascending.
    pub fn rows(&self) -> &[u32] {
        &self.rows
    }

    /// How many rows the step samples.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Sequence `i`'s sampled rows, as indices into [`Self::rows`] — rows of the logits.
    pub fn of_seq(&self, i: usize) -> std::ops::Range<usize> {
        self.cu[i] as usize..self.cu[i + 1] as usize
    }

    /// Sequence `i`'s last sampled row, as an index into [`Self::rows`]: its last row's logits.
    pub fn last_of(&self, i: usize) -> usize {
        self.cu[i + 1] as usize - 1
    }

    /// The index into [`Self::rows`] of step row `row` of sequence `i`, one of its sampled rows
    /// (`None` past them).
    pub fn index(&self, i: usize, row: u32) -> Option<usize> {
        self.of_seq(i).find(|&j| self.rows[j] == row)
    }
}

#[cfg(test)]
mod tests {
    use super::{EmptySequence, SampledRows};

    /// A step of a decode (row 0), a verify of three rows (1..=3) and a prompt chunk (4..=9): the
    /// decode and the chunk sample their last rows, the verify its every one — ascending, each
    /// sequence's at its own offsets.
    #[test]
    fn a_step_samples_each_sequence_s_last_rows() {
        let sampled = SampledRows::of(&[0, 1, 4, 10], |i| i == 1).expect("no empty sequence");
        assert_eq!(sampled.rows(), [0, 1, 2, 3, 9]);
        assert_eq!(sampled.len(), 5);
        let seqs: Vec<_> = (0..3).map(|i| sampled.of_seq(i)).collect();
        assert_eq!(seqs, [0..1, 1..4, 4..5]);
        let last: Vec<usize> = (0..3).map(|i| sampled.last_of(i)).collect();
        assert_eq!(last, [0, 3, 4]);
        // A row's index among the sampled rows; a row its sequence does not sample has none.
        assert_eq!(sampled.index(1, 2), Some(2));
        assert_eq!(sampled.index(2, 9), Some(4));
        assert_eq!(sampled.index(2, 8), None);
        assert_eq!(sampled.index(0, 1), None);
    }

    /// A sequence of no rows has no last row to sample, whether or not it verifies drafts: the
    /// step is refused, naming it.
    #[test]
    fn a_sequence_of_no_rows_is_refused() {
        for every_row in [false, true] {
            let refused = SampledRows::of(&[0, 1, 1, 4], |_| every_row);
            assert_eq!(refused, Err(EmptySequence { seq: 1 }));
        }
    }
}

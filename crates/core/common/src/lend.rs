// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! ONE MODEL'S BUFFERS, LENT TO ANOTHER THAT NEVER RUNS AT THE SAME TIME.
//!
//! A forward's activation buffers (its arena slots and scratch) hold nothing between forwards, so a
//! model whose forwards never overlap another's — a multi-token-prediction head runs after its
//! target, each forward waited on — can place its own buffers in the other's instead of allocating
//! them. [`lend`] is the one assignment: the worker that borrows allocates by it, and the memory
//! budget prices the borrower by [`unlent_bytes`], so the bytes priced are the bytes allocated.

/// Which lent buffer each of `needs` takes: the largest need first, each the smallest free buffer
/// that holds it. `None` where no free buffer does, or the need is empty: that one is allocated.
pub fn lend(needs: &[u64], lent: &[u64]) -> Vec<Option<usize>> {
    let mut order: Vec<usize> = (0..needs.len()).filter(|&i| needs[i] > 0).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(needs[i]));
    let mut taken = vec![false; lent.len()];
    let mut out = vec![None; needs.len()];
    for i in order {
        let fits = (0..lent.len()).filter(|&j| !taken[j] && lent[j] >= needs[i]);
        if let Some(j) = fits.min_by_key(|&j| lent[j]) {
            taken[j] = true;
            out[i] = Some(j);
        }
    }
    out
}

/// The bytes of `needs` that no lent buffer holds ([`lend`]): what the borrower allocates.
pub fn unlent_bytes(needs: &[u64], lent: &[u64]) -> u64 {
    let unlent = lend(needs, lent).into_iter().zip(needs);
    unlent.filter(|(j, _)| j.is_none()).map(|(_, &n)| n).sum()
}

#[cfg(test)]
mod tests {
    use super::{lend, unlent_bytes};

    /// A head's buffers in its target's: the logits slot takes the target's logits buffer, the
    /// smaller slots the smallest that hold them, each buffer once; a need no free buffer holds is
    /// allocated, and so is an empty one.
    #[test]
    fn each_need_takes_the_smallest_free_buffer_that_holds_it() {
        let target = [970, 32, 16, 16];
        assert_eq!(
            lend(&[16, 256, 32, 16, 0], &target),
            [Some(2), Some(0), Some(1), Some(3), None]
        );
        assert_eq!(lend(&[16, 16, 16], &[16, 16]), [Some(0), Some(1), None]);
        assert_eq!(unlent_bytes(&[16, 16, 16], &[16, 16]), 16);
        assert_eq!(unlent_bytes(&[64, 8], &[]), 72);
    }

    /// The largest need is placed first, so a small one does not take the only buffer a large one
    /// fits.
    #[test]
    fn the_largest_need_is_placed_first() {
        assert_eq!(lend(&[10, 100], &[120, 50]), [Some(1), Some(0)]);
        assert_eq!(lend(&[10, 100], &[120]), [None, Some(0)]);
        assert_eq!(unlent_bytes(&[10, 100], &[120]), 10);
    }
}

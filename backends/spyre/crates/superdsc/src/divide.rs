// SPDX-License-Identifier: Apache-2.0
//! The work division: how many cores each axis of an op's space is split across.

use crate::qty::{Cores, Extent};

/// One axis as the division sees it: how many whole units it can be cut into (`1` for an axis
/// that is never cut — a contraction or reduction axis), outermost axis first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cuttable {
    pub units: Extent,
}

/// Split `axes` across at most `cores` cores: every split divides its axis's units exactly, the
/// product is the largest the budget admits, and among equal products the cores go to the axes
/// with the most units left per core. One entry per axis: the cores it is split across.
pub fn divide(axes: &[Cuttable], cores: Cores) -> Vec<Cores> {
    let units: Vec<u64> = axes.iter().map(|a| a.units.get()).collect();
    divide_units(&units, cores.get())
        .into_iter()
        .map(Cores::new)
        .collect()
}

fn divide_units(axes: &[u64], cores: u64) -> Vec<u64> {
    fn divisors(n: u64, cap: u64) -> Vec<u64> {
        (1..=n.min(cap)).filter(|&d| n.is_multiple_of(d)).collect()
    }
    fn search(
        axes: &[u64],
        cores: u64,
        at: usize,
        split: &mut Vec<u64>,
        best: &mut (u64, u64, Vec<u64>),
    ) {
        if at == axes.len() {
            let used: u64 = split.iter().product();
            // Balance: the smallest per-core share, larger is better.
            let share = axes
                .iter()
                .zip(split.iter())
                .filter(|&(&a, _)| a > 1)
                .map(|(&a, s)| a / s)
                .min()
                .unwrap_or(0);
            if (used, share) > (best.0, best.1) {
                *best = (used, share, split.clone());
            }
            return;
        }
        let used: u64 = split.iter().product();
        for d in divisors(axes[at], cores / used) {
            split.push(d);
            search(axes, cores, at + 1, split, best);
            split.pop();
        }
    }
    let mut best = (0, 0, vec![1; axes.len()]);
    search(
        axes,
        cores,
        0,
        &mut Vec::with_capacity(axes.len()),
        &mut best,
    );
    best.2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cut(units: &[u64], cores: u64) -> Vec<u64> {
        let axes: Vec<Cuttable> = units
            .iter()
            .map(|&u| Cuttable {
                units: Extent::new(u),
            })
            .collect();
        divide(&axes, Cores::new(cores))
            .into_iter()
            .map(Cores::get)
            .collect()
    }

    #[test]
    fn a_wide_row_splits_its_sticks_across_the_machine() {
        assert_eq!(cut(&[1, 64], 32), vec![1, 32]);
    }

    #[test]
    fn the_budget_bounds_the_split() {
        assert_eq!(cut(&[64, 2], 8).iter().product::<u64>(), 8);
    }

    #[test]
    fn an_odd_axis_takes_its_own_factors() {
        assert_eq!(cut(&[3], 32), vec![3]);
        assert_eq!(cut(&[1], 32), vec![1]);
    }
}

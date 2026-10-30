// SPDX-License-Identifier: Apache-2.0
//! THE COLLECTIVE PLAN — tensor parallelism's control-block pipeline, as baked data.
//!
//! The values (the world, the rank, the p2p payload and its stick-column fragments, the receive
//! window, the sids, the [`TpConsume`] a launch group carries) are `ktir_superdsc::collective`'s,
//! re-exported here so the bundle's readers name one path; they live in that leaf because the
//! emitter marks a consume on its `EmittedOp` before any launch group exists. What lives HERE is
//! the part that needs the launch group: the bundle's CB pipeline, derived from its own groups.

pub use ktir_superdsc::collective::*;

/// One control block of a bundle's pipeline, in the order the compute FIFO runs them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CbStep {
    /// Send this rank's fragment of the operand into the peer's window; it increments the peer's
    /// write-done counter on this rank's sid when it lands.
    Send { group: usize, consume: TpConsume },
    /// Wait for the peer's fragment to land in this rank's window (consumes one write-done).
    Wait { group: usize, consume: TpConsume },
    /// Launch group `group`'s program.
    Launch { group: usize },
}

/// THE CB PIPELINE OF A BUNDLE — its groups as control blocks, in launch order: an ordinary group is
/// one launch; a consume group is `send, wait, launch`. The ONLY derivation of the order, so a wait
/// can never queue ahead of its own send (both ranks waiting, neither sending) and every send is
/// paired with exactly one wait — the write-done counter returns to where it started after every
/// replay of the bundle.
///
/// ⭐ RANK-SELECTED CONSUMES PLAY ON THEIR RANK ALONE. An all-gather bakes one consume per rank
/// ([`TpConsume::plays_at`]); `session` launches its own — with its send and its wait — and skips
/// every other rank's whole, so each rank still sends exactly once and waits exactly once per
/// collective.
pub fn cb_pipeline<'g>(
    groups: &'g [crate::LaunchGroup<'_>],
    session: TpIdentity,
) -> impl Iterator<Item = CbStep> + 'g {
    groups
        .iter()
        .enumerate()
        .filter(move |(_, g)| {
            g.kv.tp
                .and_then(|c| c.plays_at)
                .is_none_or(|p| p.launches_on(session))
        })
        .flat_map(|(group, g)| {
            let exchange = g.kv.tp.map(|consume| {
                [
                    CbStep::Send { group, consume },
                    CbStep::Wait { group, consume },
                ]
            });
            exchange
                .into_iter()
                .flatten()
                .chain(std::iter::once(CbStep::Launch { group }))
        })
}

/// The write-done ledger over a pipeline: +1 per send landing on the peer, −1 per wait.
pub fn wdone_balance(steps: impl Iterator<Item = CbStep>) -> i64 {
    steps
        .map(|s| match s {
            CbStep::Send { .. } => 1,
            CbStep::Wait { .. } => -1,
            CbStep::Launch { .. } => 0,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pipeline law: an ordinary group is one launch; a consume group is send, wait, launch;
    /// order is group order; the write-done ledger closes at zero.
    #[test]
    fn a_consume_group_is_send_wait_launch_in_group_order() {
        let frag = fragments(1, 2048, WireBytes::CAP).unwrap()[0];
        let consume = TpConsume {
            kind: CollectiveKind::AllReduce,
            operand: crate::PlaceId::Act(7),
            fragment: frag,
            plays_at: None,
        };
        let groups = [
            crate::LaunchGroup::default(),
            crate::LaunchGroup {
                kv: crate::KvShifts {
                    tp: Some(consume),
                    ..Default::default()
                },
                ..Default::default()
            },
            crate::LaunchGroup::default(),
        ];
        let rank0 = TpWorld::of::<2>().ranks().next().unwrap();
        let steps: Vec<CbStep> = cb_pipeline(&groups, rank0).collect();
        assert_eq!(
            steps,
            vec![
                CbStep::Launch { group: 0 },
                CbStep::Send { group: 1, consume },
                CbStep::Wait { group: 1, consume },
                CbStep::Launch { group: 1 },
                CbStep::Launch { group: 2 },
            ]
        );
        assert_eq!(wdone_balance(cb_pipeline(&groups, rank0)), 0);
    }

    /// An all-gather's consume is baked once per rank; each rank plays its own — one send, one
    /// wait, one launch — and skips the other rank's whole.
    #[test]
    fn each_rank_plays_its_own_gather_consume() {
        let frag = fragments(1, 2048, WireBytes::CAP).unwrap()[0];
        let ranks: Vec<TpIdentity> = TpWorld::of::<2>().ranks().collect();
        let gather_at = |r: TpIdentity| crate::LaunchGroup {
            kv: crate::KvShifts {
                tp: Some(TpConsume {
                    kind: CollectiveKind::AllGather,
                    operand: crate::PlaceId::Act(9),
                    fragment: frag,
                    plays_at: Some(PlaysAt::of(r)),
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        let groups = [gather_at(ranks[0]), gather_at(ranks[1]), crate::LaunchGroup::default()];
        for (mine, id) in ranks.iter().enumerate() {
            let steps: Vec<CbStep> = cb_pipeline(&groups, *id).collect();
            assert_eq!(steps.len(), 4, "send, wait, launch, then the plain group");
            assert!(steps[..3].iter().all(|s| match s {
                CbStep::Send { group, .. } | CbStep::Wait { group, .. } | CbStep::Launch { group } =>
                    *group == mine,
            }));
            assert_eq!(steps[3], CbStep::Launch { group: 2 });
            assert_eq!(wdone_balance(cb_pipeline(&groups, *id)), 0);
        }
    }

    /// A consume's weight slot is pinned to the base region at shift 0 whatever layer or bank the
    /// launch is for; every other group keeps the layer's own pair.
    #[test]
    fn a_consume_reads_the_window_unshifted_and_unbanked() {
        let consume = TpConsume {
            kind: CollectiveKind::AllReduce,
            operand: crate::PlaceId::Act(7),
            fragment: fragments(1, 2048, WireBytes::CAP).unwrap()[0],
            plays_at: None,
        };
        let plain = crate::KvShifts::default();
        let consuming = crate::KvShifts {
            tp: Some(consume),
            ..Default::default()
        };
        assert_eq!(plain.weight_slot(4096, 2), (4096, 2));
        assert_eq!(consuming.weight_slot(4096, 2), (0, 0));
    }
}

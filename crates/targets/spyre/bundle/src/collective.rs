// SPDX-License-Identifier: Apache-2.0
//! THE COLLECTIVE PLAN — tensor parallelism's control-block pipeline, as baked data.
//!
//! The values (the world, the rank, the p2p payload and its stick-column fragments, the receive
//! window, the sids, the [`TpSend`] and [`TpConsume`] a launch group carries) are
//! `ktir_superdsc::collective`'s, re-exported here so the bundle's readers name one path; they live
//! in that leaf because the emitter marks a send or a consume on its `EmittedOp` before any launch
//! group exists. What lives HERE is the part that needs the launch group: the bundle's CB pipeline,
//! derived from its own groups.

pub use ktir_superdsc::collective::*;

/// One control block of a bundle's pipeline, in the order the compute FIFO runs them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CbStep {
    /// Launch send group `group`'s program with the weight segment translated to the PEER's
    /// region: the program's store lands in the peer's window — device p2p. Completion increments
    /// the peer's write-done counter on this rank's sid.
    Send { group: usize, send: TpSend },
    /// Wait for the peer's fragment to land in this rank's window (consumes one write-done).
    Wait { group: usize, consume: TpConsume },
    /// Launch group `group`'s program.
    Launch { group: usize },
}

/// THE CB PIPELINE OF A BUNDLE — its groups as control blocks, in launch order: a send group is a
/// [`CbStep::Send`] (its program launched against the peer's weight segment), a consume group is
/// `wait, launch`, and every other group is one launch. The emitter places each collective's send
/// before its consume, so a wait never queues ahead of its own send (both ranks waiting, neither
/// sending), and every send is paired with exactly one wait — the write-done counter returns to
/// where it started after every replay of the bundle.
///
/// ⭐ RANK-SELECTED CONSUMES PLAY ON THEIR RANK ALONE. An all-gather bakes one consume per rank
/// ([`TpConsume::plays_at`]); `session` waits and launches its own and skips every other rank's
/// whole, so each rank still waits exactly once per collective.
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
            let steps: Vec<CbStep> = match (g.kv.send, g.kv.tp) {
                (Some(send), _) => vec![CbStep::Send { group, send }],
                (None, Some(consume)) => vec![
                    CbStep::Wait { group, consume },
                    CbStep::Launch { group },
                ],
                (None, None) => vec![CbStep::Launch { group }],
            };
            steps
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

    fn frag() -> Fragment {
        fragments(1, 2048, WireBytes::CAP).unwrap()[0]
    }

    fn send_group(kind: CollectiveKind) -> crate::LaunchGroup<'static> {
        crate::LaunchGroup {
            kv: crate::KvShifts {
                send: Some(TpSend {
                    kind,
                    operand: crate::PlaceId::Act(7),
                    fragment: frag(),
                }),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn consume_group(consume: TpConsume) -> crate::LaunchGroup<'static> {
        crate::LaunchGroup {
            kv: crate::KvShifts {
                tp: Some(consume),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// The pipeline law: a send group is one peer-translated launch, a consume group waits then
    /// launches, an ordinary group launches; order is group order; the write-done ledger closes.
    #[test]
    fn a_collective_is_send_then_wait_then_consume_in_group_order() {
        let consume = TpConsume {
            kind: CollectiveKind::AllReduce,
            operand: crate::PlaceId::Act(7),
            fragment: frag(),
            plays_at: None,
        };
        let groups = [
            crate::LaunchGroup::default(),
            send_group(CollectiveKind::AllReduce),
            consume_group(consume),
            crate::LaunchGroup::default(),
        ];
        let rank0 = TpWorld::of::<2>().ranks().next().unwrap();
        let steps: Vec<CbStep> = cb_pipeline(&groups, rank0).collect();
        assert!(matches!(
            steps[..],
            [
                CbStep::Launch { group: 0 },
                CbStep::Send { group: 1, .. },
                CbStep::Wait { group: 2, .. },
                CbStep::Launch { group: 2 },
                CbStep::Launch { group: 3 },
            ]
        ));
        assert_eq!(wdone_balance(cb_pipeline(&groups, rank0)), 0);
    }

    /// An all-gather sends once (the same send on every rank) and bakes one consume per rank;
    /// each rank waits for and launches its own consume and skips the other rank's whole.
    #[test]
    fn each_rank_plays_its_own_gather_consume() {
        let ranks: Vec<TpIdentity> = TpWorld::of::<2>().ranks().collect();
        let consume_at = |r: TpIdentity| {
            consume_group(TpConsume {
                kind: CollectiveKind::AllGather,
                operand: crate::PlaceId::Act(9),
                fragment: frag(),
                plays_at: Some(PlaysAt::of(r)),
            })
        };
        let groups = [
            send_group(CollectiveKind::AllGather),
            consume_at(ranks[0]),
            consume_at(ranks[1]),
            crate::LaunchGroup::default(),
        ];
        for (mine, id) in ranks.iter().enumerate() {
            let steps: Vec<CbStep> = cb_pipeline(&groups, *id).collect();
            let consume = 1 + mine;
            assert!(matches!(
                steps[..],
                [
                    CbStep::Send { group: 0, .. },
                    CbStep::Wait { group: w, .. },
                    CbStep::Launch { group: l },
                    CbStep::Launch { group: 3 },
                ] if w == consume && l == consume
            ));
            assert_eq!(wdone_balance(cb_pipeline(&groups, *id)), 0);
        }
    }

    /// A consume and a send both name the window absolutely — shift 0, bank 0 — whatever layer or
    /// bank the launch is for; only a send's weight segment is the peer's.
    #[test]
    fn a_collective_launch_reads_or_writes_the_window_unshifted_and_unbanked() {
        let plain = crate::KvShifts::default();
        let consuming = consume_group(TpConsume {
            kind: CollectiveKind::AllReduce,
            operand: crate::PlaceId::Act(7),
            fragment: frag(),
            plays_at: None,
        })
        .kv;
        let sending = send_group(CollectiveKind::AllReduce).kv;
        assert_eq!(plain.weight_slot(4096, 2), (4096, 2));
        assert_eq!(consuming.weight_slot(4096, 2), (0, 0));
        assert_eq!(sending.weight_slot(4096, 2), (0, 0));
        assert!(sending.writes_the_peer() && !consuming.writes_the_peer() && !plain.writes_the_peer());
    }
}

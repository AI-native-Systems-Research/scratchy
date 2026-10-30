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
    /// The F-send in front of consume group `group`: this rank's fragment of the consume's operand,
    /// from its own placement into the peer's receive window, on this card's DMA engine — device p2p
    /// RDMA (flex-rs `ControlBlockWire::rdma_unicast_send`). Its landing increments the peer's
    /// write-done counter on this rank's sid.
    Send { group: usize, consume: TpConsume },
    /// Wait for the peer's fragment to land in this rank's window (consumes one write-done on the
    /// peer's sid; flex-rs `ControlBlockWire::rdma_wdone_wait`).
    Wait { group: usize, consume: TpConsume },
    /// Launch group `group`'s programs.
    Launch { group: usize },
}

/// THE CB PIPELINE OF A BUNDLE — its groups as control blocks, in launch order: a consume group is
/// `send, wait, launch`, every other group one launch. A collective's send is never a group of its
/// own: the consume names everything the send moves, so every consume has exactly one send in front
/// of it, a wait never queues ahead of its own rank's send (the symmetric deadlock: both ranks
/// waiting, neither sending), and the write-done counter returns to where it started after every
/// replay of the bundle.
///
/// ⭐ RANK-SELECTED CONSUMES PLAY ON THEIR RANK ALONE. An all-gather bakes one consume per rank
/// ([`TpConsume::plays_at`]); `session` sends, waits and launches its own and skips every other
/// rank's whole, so each rank still sends once and waits once per collective.
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
            let steps: Vec<CbStep> = match g.kv.tp {
                Some(consume) => vec![
                    CbStep::Send { group, consume },
                    CbStep::Wait { group, consume },
                    CbStep::Launch { group },
                ],
                None => vec![CbStep::Launch { group }],
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

    fn consume_group(consume: TpConsume) -> crate::LaunchGroup<'static> {
        crate::LaunchGroup {
            kv: crate::KvShifts {
                tp: Some(consume),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// The pipeline law: a consume group sends, waits, then launches; an ordinary group launches;
    /// order is group order; the write-done ledger closes.
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
            consume_group(consume),
            crate::LaunchGroup::default(),
        ];
        let rank0 = TpWorld::of::<2>().ranks().next().unwrap();
        let steps: Vec<CbStep> = cb_pipeline(&groups, rank0).collect();
        assert_eq!(
            steps,
            [
                CbStep::Launch { group: 0 },
                CbStep::Send { group: 1, consume },
                CbStep::Wait { group: 1, consume },
                CbStep::Launch { group: 1 },
                CbStep::Launch { group: 2 },
            ]
        );
        assert_eq!(wdone_balance(cb_pipeline(&groups, rank0)), 0);
    }

    /// An all-gather bakes one consume per rank; each rank sends, waits for and launches its own
    /// consume and skips the other rank's whole — one send and one wait per rank.
    #[test]
    fn each_rank_plays_its_own_gather_consume() {
        let ranks: Vec<TpIdentity> = TpWorld::of::<2>().ranks().collect();
        let consume_at = |r: TpIdentity| TpConsume {
            kind: CollectiveKind::AllGather,
            operand: crate::PlaceId::Act(9),
            fragment: frag(),
            plays_at: Some(PlaysAt::of(r)),
        };
        let groups = [
            consume_group(consume_at(ranks[0])),
            consume_group(consume_at(ranks[1])),
            crate::LaunchGroup::default(),
        ];
        for (mine, id) in ranks.iter().enumerate() {
            let steps: Vec<CbStep> = cb_pipeline(&groups, *id).collect();
            let consume = consume_at(*id);
            assert_eq!(
                steps,
                [
                    CbStep::Send {
                        group: mine,
                        consume
                    },
                    CbStep::Wait {
                        group: mine,
                        consume
                    },
                    CbStep::Launch { group: mine },
                    CbStep::Launch { group: 2 },
                ]
            );
            assert_eq!(wdone_balance(cb_pipeline(&groups, *id)), 0);
        }
    }

    /// A consume names the window absolutely — shift 0, bank 0 — whatever layer or bank the launch
    /// is for.
    #[test]
    fn a_consume_reads_the_window_unshifted_and_unbanked() {
        let plain = crate::KvShifts::default();
        let consuming = consume_group(TpConsume {
            kind: CollectiveKind::AllReduce,
            operand: crate::PlaceId::Act(7),
            fragment: frag(),
            plays_at: None,
        })
        .kv;
        assert_eq!(plain.weight_slot(4096, 2), (4096, 2));
        assert_eq!(consuming.weight_slot(4096, 2), (0, 0));
    }
}

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

/// Whether the session running as `session` plays a group carrying `tp` at all.
///
/// ⭐ RANK-SELECTED CONSUMES PLAY ON THEIR RANK ALONE. An all-gather bakes one consume per rank
/// ([`TpConsume::plays_at`]); every other group plays everywhere. The one law the CB pipeline and a
/// loader's launch list both read, so the steps a bundle names and the launches a session binds
/// cannot disagree.
pub fn plays_on(tp: Option<TpConsume>, session: TpIdentity) -> bool {
    tp.and_then(|c| c.plays_at)
        .is_none_or(|p| p.launches_on(session))
}

/// One group's control blocks, in order: a consume group is `send, wait, launch`, every other group
/// one launch. A collective's send is never a group of its own: the consume names everything the
/// send moves, so every consume has exactly one send in front of it and a wait never queues ahead
/// of its own rank's send (the symmetric deadlock: both ranks waiting, neither sending).
pub fn group_steps(group: usize, tp: Option<TpConsume>) -> impl Iterator<Item = CbStep> {
    let steps: Vec<CbStep> = match tp {
        Some(consume) => vec![
            CbStep::Send { group, consume },
            CbStep::Wait { group, consume },
            CbStep::Launch { group },
        ],
        None => vec![CbStep::Launch { group }],
    };
    steps.into_iter()
}

/// THE CB PIPELINE OF A BUNDLE — the groups `session` plays ([`plays_on`]), each as its
/// [`group_steps`], in launch order. Each rank sends once and waits once per collective, and the
/// write-done counter returns to where it started after every replay of the bundle.
pub fn cb_pipeline<'g>(
    groups: &'g [crate::LaunchGroup<'_>],
    session: TpIdentity,
) -> impl Iterator<Item = CbStep> + 'g {
    groups
        .iter()
        .enumerate()
        .filter(move |(_, g)| plays_on(g.kv.tp, session))
        .flat_map(|(group, g)| group_steps(group, g.kv.tp))
}

/// WHERE A CONSUME'S SEND READS AND LANDS — segment-relative, as the bundle baked it. The source is
/// the consume's fragment inside the operand's placement; the landing is this card's own receive
/// window (the layout is card-symmetric, so the peer's window is at the same place). Only the
/// segment BASES are bound at load: the launch the send is aimed from supplies them, exactly as it
/// supplies them to the consume's own program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TpSend {
    /// The program segment the operand lives in.
    pub source_segment: u8,
    /// The fragment's first byte inside that segment.
    pub source_offset: u64,
    /// The window's offset inside [`RecvWindow::SEGMENT`].
    pub window_offset: u64,
    pub bytes: Payload,
}

/// Why a consume's send names no baked bytes. The bake refuses every one of these (it resolves
/// every consume of a bundle through [`TpSend::resolve`] before baking it), so a loaded bundle meets
/// none of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SendUnplaced {
    /// The operand has no placement in the bundle.
    Operand(crate::PlaceId),
    /// The bundle carries consumes but no receive window.
    Window,
    /// The window is not bank 0 of the weight segment — where every consume reads it.
    WindowMisplaced { segment: u32, bank: u32 },
    /// The fragment is wider than the window.
    WindowTooSmall { window: u64, bytes: u64 },
    /// The operand lives in the weight segment, which a consume launch pins to layer 0's bank-0
    /// copy — a layer-shifted operand would be read at the wrong layer.
    OperandInWeightSegment(crate::PlaceId),
    /// The fragment runs past the operand's placement.
    FragmentPastOperand {
        operand: crate::PlaceId,
        end: u64,
        size: u64,
    },
    /// An address that is not a whole number of flits.
    Misaligned { offset: u64 },
}

impl TpSend {
    /// Read `consume`'s send off `layout`'s placements.
    ///
    /// # Errors
    /// [`SendUnplaced`], by name.
    pub fn resolve(
        layout: &crate::BundleLayout<'_>,
        consume: TpConsume,
    ) -> Result<Self, SendUnplaced> {
        let window = layout
            .place(crate::PlaceId::Act(
                ktir_superdsc::reserved_tids::TP_RECV_TID,
            ))
            .ok_or(SendUnplaced::Window)?;
        if window.segment != RecvWindow::SEGMENT || window.bank != 0 {
            return Err(SendUnplaced::WindowMisplaced {
                segment: window.segment,
                bank: window.bank,
            });
        }
        let bytes = consume.fragment.bytes();
        if bytes.get() > window.size {
            return Err(SendUnplaced::WindowTooSmall {
                window: window.size,
                bytes: bytes.get(),
            });
        }
        let operand = layout
            .place(consume.operand)
            .ok_or(SendUnplaced::Operand(consume.operand))?;
        if operand.segment == RecvWindow::SEGMENT {
            return Err(SendUnplaced::OperandInWeightSegment(consume.operand));
        }
        let end = consume.fragment.byte_offset() + bytes.get();
        if end > operand.size {
            return Err(SendUnplaced::FragmentPastOperand {
                operand: consume.operand,
                end,
                size: operand.size,
            });
        }
        let source_offset = operand.offset + consume.fragment.byte_offset();
        for offset in [source_offset, window.offset] {
            if !offset.is_multiple_of(FLIT_BYTES) {
                return Err(SendUnplaced::Misaligned { offset });
            }
        }
        Ok(Self {
            // `Placement::segment` is below `NUM_SEGMENTS`.
            source_segment: operand.segment as u8,
            source_offset,
            window_offset: window.offset,
            bytes,
        })
    }
}

/// ONE F-SEND of a [`TpSend`]: one message of its payload, at the matching offset on both ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TpMessage {
    pub source_offset: u64,
    pub window_offset: u64,
    pub bytes: WireBytes,
    /// The send section's own protocol bits: this message's place in the run.
    pub first: bool,
    pub last: bool,
}

impl TpSend {
    /// ⭐ THE CUT: the send as the vendor's windowed P2P loop issues it — one F-send per message
    /// ([`Payload::messages`]), each advancing the source and the landing together. Every one
    /// increments the peer's write-done counter once, so a send of N messages is waited for N
    /// times (a wait consumes exactly one count): the pipeline issues N waits for it.
    pub fn messages(self) -> impl Iterator<Item = TpMessage> {
        self.bytes.messages().map(move |m| TpMessage {
            source_offset: self.source_offset + m.offset,
            window_offset: self.window_offset + m.offset,
            bytes: m.bytes,
            first: m.first,
            last: m.last,
        })
    }
}

/// One consume group's send, read off its bundle's placements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupSend {
    pub group: usize,
    pub send: TpSend,
}

/// A consume group whose send names no baked bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnplacedGroup {
    pub group: usize,
    pub why: SendUnplaced,
}

/// Every consume group of `groups`, with its send read off `layout` — the bake's gate (a bundle
/// whose send names no baked bytes does not bake) and the loader's read, one function.
///
/// # Errors
/// The first consume whose send is unplaced, with its group.
pub fn sends(
    layout: &crate::BundleLayout<'_>,
    groups: &[crate::LaunchGroup<'_>],
) -> Result<Vec<GroupSend>, UnplacedGroup> {
    groups
        .iter()
        .enumerate()
        .filter_map(|(group, g)| g.kv.tp.map(|c| (group, c)))
        .map(|(group, c)| {
            TpSend::resolve(layout, c)
                .map(|send| GroupSend { group, send })
                .map_err(|why| UnplacedGroup { group, why })
        })
        .collect()
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
        fragments(1, 2048, Payload::from(WireBytes::CAP)).unwrap()[0]
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

    fn placed(id: crate::PlaceId, segment: u32, offset: u64, size: u64) -> crate::Placement {
        crate::Placement {
            id,
            segment,
            bank: 0,
            offset,
            size,
            is_logits: false,
        }
    }

    fn window() -> crate::Placement {
        placed(
            crate::PlaceId::Act(ktir_superdsc::reserved_tids::TP_RECV_TID),
            RecvWindow::SEGMENT,
            1 << 20,
            WireBytes::CAP.get(),
        )
    }

    fn layout(places: Vec<crate::Placement>) -> crate::BundleLayout<'static> {
        crate::BundleLayout {
            places: places.into(),
            ..Default::default()
        }
    }

    /// The send reads the consume's fragment at the operand's own placement — its segment, its
    /// offset plus the fragment's — and lands at the window's offset.
    #[test]
    fn a_send_reads_its_fragment_at_the_operands_placement() {
        let fragment = fragments(4, 256, Payload::new(1024).unwrap()).unwrap()[1];
        let consume = TpConsume {
            kind: CollectiveKind::AllReduce,
            operand: crate::PlaceId::Act(7),
            fragment,
            plays_at: None,
        };
        let operand = placed(crate::PlaceId::Act(7), 0, 8192, 4 * 256 * 2);
        let send = TpSend::resolve(&layout(vec![operand, window()]), consume).unwrap();
        assert_eq!(
            send,
            TpSend {
                source_segment: 0,
                source_offset: 8192 + fragment.byte_offset(),
                window_offset: 1 << 20,
                bytes: fragment.bytes(),
            }
        );
        assert_ne!(fragment.byte_offset(), 0);
    }

    /// A send wider than one message is the vendor loop's run: each message reads and lands at
    /// the same offset into the payload, the run tiles it, and only its ends carry first / last.
    #[test]
    fn a_wide_send_is_a_run_of_messages_at_matching_offsets() {
        let cap = WireBytes::CAP.get();
        let send = TpSend {
            source_segment: 0,
            source_offset: 8192,
            window_offset: 1 << 30,
            bytes: Payload::new(2 * cap + 256).unwrap(),
        };
        let m: Vec<TpMessage> = send.messages().collect();
        assert_eq!(m.len(), 3);
        for (k, x) in m.iter().enumerate() {
            let at = k as u64 * cap;
            assert_eq!(
                (x.source_offset, x.window_offset),
                (8192 + at, (1 << 30) + at)
            );
            assert_eq!((x.first, x.last), (k == 0, k == 2));
        }
        assert_eq!(m[2].bytes.get(), 256);
        assert_eq!(
            m.iter().map(|x| x.bytes.get()).sum::<u64>(),
            send.bytes.get()
        );
    }

    /// Every way a send can name no baked bytes refuses by name.
    #[test]
    fn an_unplaced_send_refuses_by_name() {
        let consume = TpConsume {
            kind: CollectiveKind::AllReduce,
            operand: crate::PlaceId::Act(7),
            fragment: frag(),
            plays_at: None,
        };
        let whole = frag().bytes().get();
        let operand = || placed(crate::PlaceId::Act(7), 0, 0, whole);
        let refusal = |places| TpSend::resolve(&layout(places), consume).unwrap_err();
        assert_eq!(refusal(vec![operand()]), SendUnplaced::Window);
        assert_eq!(
            refusal(vec![window()]),
            SendUnplaced::Operand(crate::PlaceId::Act(7))
        );
        let mut banked = window();
        banked.bank = 1;
        assert_eq!(
            refusal(vec![operand(), banked]),
            SendUnplaced::WindowMisplaced {
                segment: RecvWindow::SEGMENT,
                bank: 1
            }
        );
        let mut narrow = window();
        narrow.size = whole - FLIT_BYTES;
        assert_eq!(
            refusal(vec![operand(), narrow]),
            SendUnplaced::WindowTooSmall {
                window: whole - FLIT_BYTES,
                bytes: whole
            }
        );
        assert_eq!(
            refusal(vec![
                placed(crate::PlaceId::Act(7), RecvWindow::SEGMENT, 0, whole),
                window()
            ]),
            SendUnplaced::OperandInWeightSegment(crate::PlaceId::Act(7))
        );
        assert_eq!(
            refusal(vec![
                placed(crate::PlaceId::Act(7), 0, 0, whole - FLIT_BYTES),
                window()
            ]),
            SendUnplaced::FragmentPastOperand {
                operand: crate::PlaceId::Act(7),
                end: whole,
                size: whole - FLIT_BYTES
            }
        );
        assert_eq!(
            refusal(vec![placed(crate::PlaceId::Act(7), 0, 64, whole), window()]),
            SendUnplaced::Misaligned { offset: 64 }
        );
    }
}

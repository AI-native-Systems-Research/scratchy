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
pub enum CbStep<T: Tp> {
    /// The F-send in front of consume group `group` to peer `to`: this rank's fragment of the
    /// consume's operand, from its own placement into this rank's block of `to`'s receive window,
    /// on this card's DMA engine — device p2p RDMA (flex-rs `scheduler::build_rdma_send_cb`), one
    /// F-send per message. Each message's landing increments `to`'s write-done counter on this
    /// rank's sid for its landing slot.
    Send {
        group: usize,
        consume: TpConsume<T>,
        to: Rank<T>,
    },
    /// Wait for peer `from`'s fragment to land in `from`'s block of this rank's window (consumes
    /// one write-done per message on `from`'s sid for its slot; flex-rs
    /// `ControlBlockWire::p2p_rdma_wait`).
    Wait {
        group: usize,
        consume: TpConsume<T>,
        from: Rank<T>,
    },
    /// Launch group `group`'s programs.
    Launch { group: usize },
}

/// Whether the session running as `session` plays a group carrying `tp` at all.
///
/// ⭐ A CONSUME PLAYS ON ITS RANK ALONE. A consume reads its own share from its own operand and
/// every peer's from the window, so the bake carries one consume per rank
/// ([`TpConsume::plays_at`]); every other group plays everywhere. The one law the CB pipeline and a
/// loader's launch list both read, so the steps a bundle names and the launches a session binds
/// cannot disagree.
pub fn plays_on<T: Tp>(tp: Option<TpConsume<T>>, session: Rank<T>) -> bool {
    tp.is_none_or(|c| c.launches_on(session))
}

/// One group's control blocks, in order: a consume group is `send to every peer, wait for every
/// peer, launch`, every other group one launch. A collective's sends are never a group of their
/// own: the consume names everything they move, so every consume has its sends in front of it and
/// a wait never queues ahead of its own rank's sends (the symmetric deadlock: every rank waiting,
/// none sending).
pub fn group_steps<T: Tp>(
    group: usize,
    tp: Option<TpConsume<T>>,
) -> impl Iterator<Item = CbStep<T>> {
    let steps: Vec<CbStep<T>> = match tp {
        Some(consume) => {
            let me = consume.plays_at;
            me.peers()
                .map(|to| CbStep::Send { group, consume, to })
                .chain(me.peers().map(|from| CbStep::Wait {
                    group,
                    consume,
                    from,
                }))
                .chain(std::iter::once(CbStep::Launch { group }))
                .collect()
        }
        None => vec![CbStep::Launch { group }],
    };
    steps.into_iter()
}

/// THE CB PIPELINE OF A BUNDLE — the groups `session` plays ([`plays_on`]), each as its
/// [`group_steps`], in launch order. Each rank sends once to and waits once on every peer per
/// collective, and the write-done counter returns to where it started after every replay of the
/// bundle.
pub fn cb_pipeline<'g, T: Tp>(
    groups: &'g [crate::LaunchGroup<'_, T>],
    session: Rank<T>,
) -> impl Iterator<Item = CbStep<T>> + 'g {
    groups
        .iter()
        .enumerate()
        .filter(move |(_, g)| plays_on(g.kv.tp, session))
        .flat_map(|(group, g)| group_steps(group, g.kv.tp))
}

/// WHERE A CONSUME'S SENDS READ AND LAND — segment-relative, as the bundle baked it. The source is
/// the consume's fragment inside the operand's placement; the landing is the SENDING rank's block of
/// the window cut, on every peer (the layout is card-symmetric, so every peer's window is at the
/// same place, and the sender's block is the same block on every one). Only the segment BASES are
/// bound at load: the launch the sends are aimed from supplies them, exactly as it supplies them to
/// the consume's own program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TpSend<T: Tp> {
    /// The program segment the operand lives in.
    pub source_segment: u8,
    /// The fragment's first byte inside that segment.
    pub source_offset: u64,
    /// The sending rank's block of the window cut, inside [`WINDOW_SEGMENT`].
    pub window_offset: u64,
    pub bytes: CutPayload<T>,
    /// The cut it lands in — message `m` lands in slot `(cut, m)`, signalling that slot's sid.
    pub cut: WindowCut,
}

/// Why a consume's send names no baked bytes. The bake refuses every one of these (it resolves
/// every consume of a bundle through [`TpSend::resolve`] before baking it), so a loaded bundle meets
/// none of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SendUnplaced {
    /// The operand has no placement in the bundle.
    Operand(crate::PlaceId),
    /// The bundle carries consumes but no receive window.
    Window,
    /// A window cut is not bank 0 of the weight segment — where every consume reads it.
    WindowMisplaced { segment: u32, bank: u32 },
    /// The window's cuts are not equal, adjacent and in order — not one window.
    WindowCutsApart { cut: usize, offset: u64, size: u64 },
    /// The fragment is wider than one block of the window.
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

/// ⭐ THE WINDOW `layout` BAKES FOR WORLD `T` — its blocks' placements read back as the ONE window
/// they tile (every cut, every sender's block of it, adjacent and equal, in order), or `None` for a
/// bundle with no collective. The bake's send gate and the loader's link read it here, so they
/// cannot disagree about where a peer's sends land.
///
/// # Errors
/// A block missing or misplaced, blocks that are not one window, or an address that is not whole
/// flits — by name.
pub fn baked_window<T: Tp>(
    layout: &crate::BundleLayout<'_>,
) -> Result<Option<RecvWindow<T>>, SendUnplaced> {
    let tid = |cut: usize, from: Rank<T>| WindowCut::of_collective(cut).block_tid(from);
    if !T::SHARDED {
        return Ok(None);
    }
    let Some(first) = layout.place(crate::PlaceId::Act(tid(0, Rank::of::<0>()))) else {
        return Ok(None);
    };
    for (i, (cut, from)) in (0..WINDOW_CUTS)
        .flat_map(|c| ranks::<T>().map(move |r| (c, r)))
        .enumerate()
    {
        let block = layout
            .place(crate::PlaceId::Act(tid(cut, from)))
            .ok_or(SendUnplaced::Window)?;
        if block.segment != WINDOW_SEGMENT || block.bank != 0 {
            return Err(SendUnplaced::WindowMisplaced {
                segment: block.segment,
                bank: block.bank,
            });
        }
        if block.size != first.size || block.offset != first.offset + i as u64 * first.size {
            return Err(SendUnplaced::WindowCutsApart {
                cut,
                offset: block.offset,
                size: block.size,
            });
        }
    }
    Payload::new(first.size)
        .and_then(|block| RecvWindow::new(first.offset, block))
        .map(Some)
        .ok_or(SendUnplaced::Misaligned {
            offset: first.offset,
        })
}

impl<T: Tp> TpSend<T> {
    /// Read `consume`'s send — landing in its window cut — off `layout`'s placements.
    ///
    /// # Errors
    /// [`SendUnplaced`], by name.
    pub fn resolve(
        layout: &crate::BundleLayout<'_>,
        consume: TpConsume<T>,
    ) -> Result<Self, SendUnplaced> {
        let me = consume.plays_at;
        let window = baked_window::<T>(layout)?.ok_or(SendUnplaced::Window)?;
        let bytes = consume.fragment.bytes();
        if bytes.get() > window.block().get() {
            return Err(SendUnplaced::WindowTooSmall {
                window: window.block().get(),
                bytes: bytes.get(),
            });
        }
        let landing = window.block_offset(consume.cut, me);
        let operand = layout
            .place(consume.operand)
            .ok_or(SendUnplaced::Operand(consume.operand))?;
        if operand.segment == WINDOW_SEGMENT {
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
        for offset in [source_offset, landing] {
            if !offset.is_multiple_of(FLIT_BYTES) {
                return Err(SendUnplaced::Misaligned { offset });
            }
        }
        Ok(Self {
            // `Placement::segment` is below `NUM_SEGMENTS`.
            source_segment: operand.segment as u8,
            source_offset,
            window_offset: landing,
            // A fragment within one block is within the block's slots (`RecvWindow::new`).
            bytes: CutPayload::new(bytes).ok_or(SendUnplaced::WindowTooSmall {
                window: window.block().get(),
                bytes: bytes.get(),
            })?,
            cut: consume.cut,
        })
    }
}

/// ONE F-SEND of a [`TpSend`]: one message of its payload, at the matching offset on both ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TpMessage<T: Tp> {
    pub source_offset: u64,
    pub window_offset: u64,
    pub bytes: WireBytes,
    /// The send section's own protocol bits: this message's place in the run.
    pub first: bool,
    pub last: bool,
    /// Where it lands — and so the sid it signals on and every peer waits on.
    pub slot: LandingSlot<T>,
}

impl<T: Tp> TpSend<T> {
    /// ⭐ THE CUT: the send as the vendor's windowed P2P loop issues it — one F-send per message
    /// ([`Payload::messages`]), each advancing the source and the landing together, and each in a
    /// landing slot of its own ([`CutPayload::landings`]). Every one increments its slot's
    /// write-done counter on the peer it lands on once, so a send of N messages is waited for N
    /// times on that peer, once per slot (a wait consumes exactly one count).
    pub fn messages(self) -> impl Iterator<Item = TpMessage<T>> {
        self.bytes
            .landings(self.cut)
            .map(move |(slot, m)| TpMessage {
                source_offset: self.source_offset + m.offset,
                window_offset: self.window_offset + m.offset,
                bytes: m.bytes,
                first: m.first,
                last: m.last,
                slot,
            })
    }
}

/// One consume group's send, read off its bundle's placements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupSend<T: Tp> {
    pub group: usize,
    pub send: TpSend<T>,
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
pub fn sends<T: Tp>(
    layout: &crate::BundleLayout<'_>,
    groups: &[crate::LaunchGroup<'_, T>],
) -> Result<Vec<GroupSend<T>>, UnplacedGroup> {
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

/// The write-done ledger over one rank's pipeline: +1 per send landing on a peer, −1 per wait.
pub fn wdone_balance<T: Tp>(steps: impl Iterator<Item = CbStep<T>>) -> i64 {
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

    fn consume_group<T: Tp>(consume: TpConsume<T>) -> crate::LaunchGroup<'static, T> {
        crate::LaunchGroup {
            kv: crate::KvShifts {
                tp: Some(consume),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn consume_at<T: Tp>(kind: CollectiveKind, me: Rank<T>, cut: usize) -> TpConsume<T> {
        TpConsume {
            kind,
            operand: crate::PlaceId::Act(7),
            fragment: frag(),
            plays_at: me,
            cut: WindowCut::of_collective(cut),
        }
    }

    /// The pipeline law, at every world width: a rank plays its own consume group — a send to
    /// every peer, then a wait on every peer, then the launch — and skips every other rank's;
    /// ordinary groups launch; order is group order; the write-done ledger closes.
    #[test]
    fn a_collective_is_sends_then_waits_then_consume_in_group_order() {
        fn check<T: Tp>() {
            let ranks: Vec<Rank<T>> = ranks::<T>().collect();
            let mut groups = vec![crate::LaunchGroup::default()];
            groups.extend(
                ranks
                    .iter()
                    .map(|&r| consume_group(consume_at(CollectiveKind::AllReduce, r, 0))),
            );
            groups.push(crate::LaunchGroup::default());
            let last = groups.len() - 1;
            for (i, &me) in ranks.iter().enumerate() {
                let mine = 1 + i;
                let consume = consume_at(CollectiveKind::AllReduce, me, 0);
                let mut want = vec![CbStep::Launch { group: 0 }];
                want.extend(me.peers().map(|to| CbStep::Send {
                    group: mine,
                    consume,
                    to,
                }));
                want.extend(me.peers().map(|from| CbStep::Wait {
                    group: mine,
                    consume,
                    from,
                }));
                want.push(CbStep::Launch { group: mine });
                want.push(CbStep::Launch { group: last });
                let steps: Vec<CbStep<T>> = cb_pipeline(&groups, me).collect();
                assert_eq!(steps, want, "rank {} of {}", me.get(), T::RANKS);
                assert_eq!(wdone_balance(cb_pipeline(&groups, me)), 0);
            }
        }
        check::<TpWorld<1>>();
        check::<TpWorld<2>>();
        check::<TpWorld<3>>();
        check::<TpWorld<4>>();
        check::<TpWorld<8>>();
        check::<TpWorld<MAX_WORLD>>();
    }

    /// A consume names the window absolutely — shift 0, bank 0 — whatever layer or bank the launch
    /// is for.
    #[test]
    fn a_consume_reads_the_window_unshifted_and_unbanked() {
        let plain = crate::KvShifts::<TpWorld<4>>::default();
        let me = Rank::<TpWorld<4>>::of::<1>();
        let consuming = consume_group(consume_at(CollectiveKind::AllReduce, me, 0)).kv;
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

    /// The window as the bake places it for world `T`: `WINDOW_CUTS` adjacent cuts at 1 MiB, each
    /// one `block`-byte placement per sending rank.
    fn window_of<T: Tp>(block: u64) -> Vec<crate::Placement> {
        (0..WINDOW_CUTS)
            .flat_map(|c| ranks::<T>().map(move |r| (c, r)))
            .enumerate()
            .map(|(i, (c, r))| {
                placed(
                    crate::PlaceId::Act(WindowCut::of_collective(c).block_tid(r)),
                    WINDOW_SEGMENT,
                    (1 << 20) + i as u64 * block,
                    block,
                )
            })
            .collect()
    }

    fn with(mut window: Vec<crate::Placement>, more: crate::Placement) -> Vec<crate::Placement> {
        window.push(more);
        window
    }

    fn layout(places: Vec<crate::Placement>) -> crate::BundleLayout<'static> {
        crate::BundleLayout {
            places: places.into(),
            ..Default::default()
        }
    }

    /// The send reads the consume's fragment at the operand's own placement — its segment, its
    /// offset plus the fragment's — and lands at the SENDER'S block of its cut, at every rank of
    /// every world.
    #[test]
    fn a_send_lands_in_its_senders_block_of_its_cut() {
        fn check<T: Tp>() {
            let fragment = fragments(4, 256, Payload::new(1024).unwrap()).unwrap()[1];
            let block = WireBytes::CAP.get();
            let operand = placed(crate::PlaceId::Act(7), 0, 8192, 4 * 256 * 2);
            let places = layout(with(window_of::<T>(block), operand));
            for me in ranks::<T>() {
                for k in 0..WINDOW_CUTS {
                    let cut = WindowCut::of_collective(k);
                    let consume = TpConsume {
                        fragment,
                        ..consume_at(CollectiveKind::AllReduce, me, k)
                    };
                    assert_eq!(
                        TpSend::resolve(&places, consume).unwrap(),
                        TpSend {
                            source_segment: 0,
                            source_offset: 8192 + fragment.byte_offset(),
                            window_offset: (1 << 20)
                                + k as u64 * u64::from(T::RANKS) * block
                                + u64::from(me.get()) * block,
                            bytes: CutPayload::new(fragment.bytes()).unwrap(),
                            cut,
                        }
                    );
                }
            }
        }
        check::<TpWorld<2>>();
        check::<TpWorld<3>>();
        check::<TpWorld<4>>();
        check::<TpWorld<8>>();
        check::<TpWorld<MAX_WORLD>>();
    }

    /// The window reads back off the bake as the one window its cuts tile — a block per rank of
    /// its world — and a bundle with no cut placed has none.
    #[test]
    fn the_baked_window_is_its_cuts_read_back() {
        type W4 = TpWorld<4>;
        let window = baked_window::<W4>(&layout(window_of::<W4>(WireBytes::CAP.get())))
            .unwrap()
            .unwrap();
        assert_eq!(window.offset(), 1 << 20);
        assert_eq!(window.block(), Payload::from(WireBytes::CAP));
        assert_eq!(window.size(), WINDOW_CUTS as u64 * 4 * WireBytes::CAP.get());
        assert_eq!(baked_window::<W4>(&layout(vec![])), Ok(None));
        assert_eq!(
            baked_window::<TpWorld<1>>(&layout(window_of::<W4>(WireBytes::CAP.get()))),
            Ok(None),
            "a world of one has no window"
        );
    }

    /// Consecutive collectives alternate cuts, and each one's send lands in its own; the forward's
    /// third collective is back in the first's cut.
    #[test]
    fn consecutive_collectives_send_into_alternate_cuts() {
        type W4 = TpWorld<4>;
        let me = Rank::<W4>::of::<0>();
        let block = WireBytes::CAP.get();
        let operand = placed(crate::PlaceId::Act(7), 0, 0, frag().bytes().get());
        let places = layout(with(window_of::<W4>(block), operand));
        let groups = [
            consume_group(consume_at(CollectiveKind::AllReduce, me, 0)),
            crate::LaunchGroup::<W4>::default(),
            consume_group(consume_at(CollectiveKind::AllReduce, me, 1)),
            consume_group(consume_at(CollectiveKind::AllReduce, me, 2)),
        ];
        let landed: Vec<(usize, u64)> = sends(&places, &groups)
            .unwrap()
            .into_iter()
            .map(|s| (s.group, s.send.window_offset))
            .collect();
        let cut = 4 * block;
        assert_eq!(landed, [(0, 1 << 20), (2, (1 << 20) + cut), (3, 1 << 20)]);
        assert_eq!(WindowCut::baked(1), WindowCut::of_collective(1));
        assert_eq!(
            WindowCut::of_collective(1).block_tid(me),
            ktir_superdsc::reserved_tids::tp_recv_block_tid(1, 0)
        );
    }

    /// A send wider than one message is the vendor loop's run: each message reads and lands at
    /// the same offset into the payload, in a slot of its own, the run tiles it, and only its ends
    /// carry first / last; a payload past the world's slots per cut is no send.
    #[test]
    fn a_wide_send_is_a_run_of_messages_at_matching_offsets() {
        type W2 = TpWorld<2>;
        let cap = WireBytes::CAP.get();
        let cut = WindowCut::of_collective(1);
        let send = TpSend::<W2> {
            source_segment: 0,
            source_offset: 8192,
            window_offset: 1 << 30,
            bytes: CutPayload::new(Payload::new(2 * cap + 256).unwrap()).unwrap(),
            cut,
        };
        let m: Vec<TpMessage<W2>> = send.messages().collect();
        assert_eq!(m.len(), 3);
        for (k, x) in m.iter().enumerate() {
            let at = k as u64 * cap;
            assert_eq!(
                (x.source_offset, x.window_offset),
                (8192 + at, (1 << 30) + at)
            );
            assert_eq!((x.first, x.last), (k == 0, k == 2));
            assert_eq!(x.slot, LandingSlot::of(cut, k as u32).unwrap());
        }
        assert_eq!(m[2].bytes.get(), 256);
        assert_eq!(
            m.iter().map(|x| x.bytes.get()).sum::<u64>(),
            send.bytes.get().get()
        );
        fn past<T: Tp>() {
            let cap = WireBytes::CAP.get();
            let past = Payload::new(u64::from(T::MESSAGES_PER_CUT) * cap + FLIT_BYTES).unwrap();
            assert_eq!(
                CutPayload::<T>::new(past),
                None,
                "more messages than a block has slots"
            );
        }
        past::<TpWorld<2>>();
        past::<TpWorld<4>>();
        past::<TpWorld<8>>();
        past::<TpWorld<MAX_WORLD>>();
    }

    /// Every way a send can name no baked bytes refuses by name.
    #[test]
    fn an_unplaced_send_refuses_by_name() {
        type W2 = TpWorld<2>;
        let consume = consume_at(CollectiveKind::AllReduce, Rank::<W2>::of::<0>(), 0);
        let whole = frag().bytes().get();
        let window = || window_of::<W2>(WireBytes::CAP.get());
        let operand = || placed(crate::PlaceId::Act(7), 0, 0, whole);
        let refusal = |places| TpSend::resolve(&layout(places), consume).unwrap_err();
        assert_eq!(refusal(vec![operand()]), SendUnplaced::Window);
        assert_eq!(
            refusal(window()),
            SendUnplaced::Operand(crate::PlaceId::Act(7))
        );
        let mut one_cut = window();
        one_cut.pop();
        assert_eq!(refusal(with(one_cut, operand())), SendUnplaced::Window);
        let mut banked = window();
        banked[1].bank = 1;
        assert_eq!(
            refusal(with(banked, operand())),
            SendUnplaced::WindowMisplaced {
                segment: WINDOW_SEGMENT,
                bank: 1
            }
        );
        // The first block of cut 1 — sender 0's — moved off the window.
        let cut1 = <W2 as World>::RANKS as usize;
        let mut apart = window();
        apart[cut1].offset += FLIT_BYTES;
        assert_eq!(
            refusal(with(apart.clone(), operand())),
            SendUnplaced::WindowCutsApart {
                cut: 1,
                offset: apart[cut1].offset,
                size: apart[cut1].size
            }
        );
        assert_eq!(
            refusal(with(window_of::<W2>(whole - FLIT_BYTES), operand())),
            SendUnplaced::WindowTooSmall {
                window: whole - FLIT_BYTES,
                bytes: whole
            }
        );
        assert_eq!(
            refusal(with(
                window(),
                placed(crate::PlaceId::Act(7), WINDOW_SEGMENT, 0, whole)
            )),
            SendUnplaced::OperandInWeightSegment(crate::PlaceId::Act(7))
        );
        assert_eq!(
            refusal(with(
                window(),
                placed(crate::PlaceId::Act(7), 0, 0, whole - FLIT_BYTES)
            )),
            SendUnplaced::FragmentPastOperand {
                operand: crate::PlaceId::Act(7),
                end: whole,
                size: whole - FLIT_BYTES
            }
        );
        assert_eq!(
            refusal(with(window(), placed(crate::PlaceId::Act(7), 0, 64, whole))),
            SendUnplaced::Misaligned { offset: 64 }
        );
    }
}

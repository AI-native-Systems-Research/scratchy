// SPDX-License-Identifier: Apache-2.0
//! ONE PROCESS, N CARDS, AN ALL-GATHER OVER THE REAL FLEX PATH — the card-to-card transport
//! tensor parallelism runs on, proven on the card.
//!
//! Every card holds an N-slot buffer. Card `i` starts with its own pattern in slot `i` and zeroes
//! everywhere else; it then F-sends slot `i` to every peer, landing in that peer's slot `i`. When
//! the run ends, every card's buffer must hold every card's pattern, byte for byte.
//!
//! The mechanism, all of it the production path (`WorldCards::open_cards` / `build_all`, the
//! flex scheduler's `launch_p2p_rdma_send` / `launch_p2p_rdma_wait`):
//! - **link**: on every card, base-address-table entry `tid = peer` holds the peer's region-2 PCI
//!   address (`SenPci::pciaddr_for_region(peer, 2)`, the vendor's `initialize_p2p_address_table`);
//! - **landing**: on receiver `j`, `XSEG[sid = i]` points at `j`'s slot `i` — the sender's index is
//!   its sid, so every sender lands in its own slot. Every entry is programmed before any send;
//! - **send**: card `i` sends slot `i` to each peer on sid `i`, then waits once per peer on that
//!   peer's sid. The receive counter (`wdone`) counts landings whether or not the wait is posted
//!   yet, so send-then-wait on one stream cannot deadlock.
//!
//! Checked after the run, per card: every slot byte-exact; `wdone` 0 on every sid (each landing
//! consumed by exactly its wait); every `XSEG` entry unchanged (a send never re-points one); RDMA
//! error status 0.
//!
//! The slot is TWO messages wide (`RDMA_MESSAGE_CAP_BYTES` each), and the mode names the exchange's
//! shape — issue #1429's probe ladder:
//! - `vendor` (default): the vendor door per pair, one sid per sender, its own windowed loop — the
//!   shape whose second message the 8-MiB run silently drops;
//! - `run2`: the production CB shape — two per-message F-sends per pair in ONE `submit_batch`
//!   (`build_rdma_send_cb`), each message its own sid and landing slot, the run's `first`/`last`
//!   flags, then one wdone wait per message slot: the exact flush the (1024,1) rung wedges in;
//! - `alone2`: the fix candidate — the same, every message `RunPlace::ALONE`;
//! - `split2`: the production shape across TWO doorbells — the sends synchronized on their own,
//!   then the waits — deciding whether the stuck CB is the message-1 SEND (the sends never
//!   retire) or the slot-1 WAIT (the sends retire, the waits hang).
//!
//! `run2`/`alone2` take a wdone census (every watched sid, on every card) at 1/5/10/30 s after the
//! doorbell, BEFORE synchronizing: a wait that never satisfies leaves its sid's count wherever the
//! landing left it — 0 for a message that never landed, 1 for one that landed and was never
//! consumed — which is the drop-vs-wait split, read while the card is stuck.
//!
//! ```text
//! cargo run --release -p scratchy-target-spyre-card --features snt1p0 --example cards_all_gather -- [[N]] [vendor|run2|alone2|split2]
//! ```
//! N defaults to every card this host's card table (`CardTable`) enumerates, opened by PCI
//! address.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail, ensure};

use flex_rs::address::{ByteOffset, ByteSize, Chunk, CompositeAddress, LogicalAddress};
use flex_rs::allocator::AllocationDirective;
use flex_rs::card::pci_addr_for_region;
use flex_rs::control_block_wire::Flits;
use flex_rs::dma::{DmaDirection, DmaParams, DmaShaping, HostVirtualAddress};
use flex_rs::hal_sbf::rdma_xseg;
use flex_rs::scheduler::{
    DeviceAddressResolver, FlexAllocatorAddressResolver, P2PRdmaSendParams, P2PRdmaWaitParams,
    RunPlace,
};
use flex_rs::stream::{RuntimeStream, RuntimeStreamMode, RuntimeStreamPriority};
use scratchy_target_spyre::bundle_code::collective::{RDMA_MESSAGE_CAP_BYTES, TpWorld, WireBytes};
use scratchy_target_spyre_card::runtime::CardRuntime as AnyCard;
use scratchy_target_spyre_card::runtime::{
    BatchedCb, CardTable, Chunk0, DevAddr, RdmaEnd, RdmaSend, Stream, WorldCards,
};

/// A raw flex-level probe: the cards are driven below the world-typed p2p link, so they are opened
/// as one card's runtime each.
type CardRuntime<'a> = AnyCard<'a, TpWorld<1>>;

/// One slot: one card's message — TWO of them at the message cap, the width whose second message
/// the vendor loop drops (#1429).
const SLOT_BYTES: u64 = 8 << 20;
/// The wire's message cap: the chunk width of the vendor's windowed loop and of the production
/// per-message F-sends.
const CAP: u64 = RDMA_MESSAGE_CAP_BYTES;
/// The messages one slot's send is, at this width.
const MSGS: u64 = SLOT_BYTES / CAP;

/// Card `i`'s pattern: distinct per card, and distinct per 4-MiB message of the slot (the
/// `^ (j >> 22) * 0x5B` term flips every byte of the second half — the bare polynomial repeats
/// every 4 MiB as a byte, which cannot tell a dropped message from one landed on top of message
/// 0), so a send landing in another card's slot — or the wrong half of the right one — is a
/// mismatch.
fn pattern(i: usize) -> Vec<u8> {
    (0..SLOT_BYTES as usize)
        .map(|j| ((j * 7 + 3 + 31 * (i + 1)) ^ (j >> 8) ^ ((j >> 22) * 0x5B)) as u8)
        .collect()
}

/// A device byte address as the RDMA engine reads it: the segment in bits 27.. of the flit index.
fn flat_va(segment: u64, offset_bytes: u64) -> u64 {
    ((segment << 27) | (offset_bytes >> 7)) << 7
}

/// One card's side of the run: its runtime (borrowed from `main`, which owns every card), one
/// stream, and its N-slot buffer.
struct Side<'a, 'c> {
    card: &'c CardRuntime<'a>,
    stream: Arc<RuntimeStream<'a>>,
    buf: Arc<CompositeAddress>,
    /// The buffer's (segment, byte offset) on this card — what a peer's send lands against.
    segment: u64,
    offset_bytes: u64,
}

impl<'a, 'c> Side<'a, 'c> {
    fn new(card: &'c CardRuntime<'a>, n: usize) -> Result<Self> {
        let rt = card.context().clone();
        let allocator = rt.get_allocator().map_err(|e| anyhow!("allocator: {e}"))?;
        let buf = Arc::new(
            allocator
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .allocate(
                    ByteSize(SLOT_BYTES * n as u64),
                    &AllocationDirective::default(),
                )
                .map_err(|e| anyhow!("allocate: {e}"))?,
        );
        let chunk0 = buf.chunks()[0];
        let resolver = FlexAllocatorAddressResolver::new(allocator);
        let (segment, _, _) = resolver
            .resolve(&chunk0)
            .ok_or_else(|| anyhow!("the buffer's region resolved to no segment"))?;
        let offset_bytes = chunk0.addr.offset.0;
        ensure!(
            offset_bytes.is_multiple_of(Flits::BYTES),
            "buffer offset {offset_bytes} is not flit-aligned"
        );
        let id = rt
            .create_stream(
                RuntimeStreamPriority::Normal,
                RuntimeStreamMode::StrictOrdering,
                false,
            )
            .map_err(|e| anyhow!("create_stream: {e}"))?;
        let stream = rt
            .get_stream(id)
            .map_err(|e| anyhow!("get_stream: {e}"))?
            .ok_or_else(|| anyhow!("stream vanished right after creation"))?;
        Ok(Self {
            card,
            stream,
            buf,
            segment: u64::from(segment),
            offset_bytes,
        })
    }

    /// Slot `i` of this card's buffer, as its own address.
    fn slot(&self, i: usize) -> Arc<CompositeAddress> {
        let c = self.buf.chunks()[0];
        Arc::new(CompositeAddress::from_chunk(Chunk::new(
            LogicalAddress::new(
                c.addr.region_id,
                ByteOffset(c.addr.offset.0 + i as u64 * SLOT_BYTES),
            ),
            ByteSize(SLOT_BYTES),
            c.domain_id,
        )))
    }

    /// Where slot `i` sits, as a peer's send names it.
    fn slot_va(&self, i: usize) -> u64 {
        flat_va(self.segment, self.offset_bytes + i as u64 * SLOT_BYTES)
    }

    /// Slot `i`'s message `m`, as its own allocation view — the operand a per-message F-send names
    /// (`source` and `window` both: the send's window word is its own card's view, the peer's
    /// sitting at the same address).
    fn half(&self, i: usize, m: u64) -> DevAddr {
        let c = self.buf.chunks()[0];
        DevAddr::from_chunk(Chunk0 {
            region: c.addr.region_id,
            offset: ByteOffset(c.addr.offset.0 + i as u64 * SLOT_BYTES + m * CAP),
            size: ByteSize(CAP),
            domain: c.domain_id,
        })
    }

    /// Where slot `i`'s message `m` lands — the flat VA the receiver's `XSEG[sid]` entry names.
    fn half_va(&self, i: usize, m: u64) -> u64 {
        flat_va(
            self.segment,
            self.offset_bytes + i as u64 * SLOT_BYTES + m * CAP,
        )
    }

    fn dma(&self, host: *const u8, len: usize, dir: DmaDirection) -> Result<()> {
        let p = DmaParams::with_explicit_size(
            HostVirtualAddress(host as usize),
            ByteSize(len as u64),
            dir,
            Some(self.buf.clone()),
            DmaShaping::default(),
        )
        .map_err(|e| anyhow!("dma params: {e:?}"))?;
        match dir {
            DmaDirection::HostToDevice => self.stream.launch_h2d(p, None),
            DmaDirection::DeviceToHost => self.stream.launch_d2h(p, None),
        }
        .and_then(|()| self.stream.synchronize())
        .map_err(|e| anyhow!("{dir:?}: {e}"))
    }
}

/// The exchange's shape (issue #1429's probe ladder).
enum Mode {
    /// The vendor door per pair — `launch_p2p_rdma_send`'s own windowed loop, one sid per sender.
    Vendor,
    /// The production CB shape: per-message F-sends through ONE `submit_batch`, each message its
    /// own sid and landing slot, the run's `first`/`last` flags, one wdone wait per message slot.
    Run,
    /// The fix: the same, every message `RunPlace::ALONE` — each its own complete send.
    Alone,
    /// The production shape split across two doorbells: the sends synchronized on their own
    /// first, then the waits — deciding whether the stuck CB is the message-1 SEND (the sends
    /// never retire) or the slot-1 WAIT (the sends retire, the waits hang).
    Split,
}

impl Mode {
    /// Sender `i`'s message `m` signals sid `2i + m` (the vendor mode's one sid is the sender's
    /// index itself).
    fn sid(i: usize, m: u64) -> u8 {
        (2 * i as u64 + m) as u8
    }

    fn of(arg: &str) -> Result<Self> {
        Ok(match arg {
            "vendor" => Mode::Vendor,
            "run2" => Mode::Run,
            "alone2" => Mode::Alone,
            "split2" => Mode::Split,
            other => bail!("unknown mode {other:?}: vendor | run2 | alone2 | split2"),
        })
    }
}

fn main() -> Result<()> {
    // ── The cards: this host's whole card table by default — the card gate passes no N, so a
    //    host of any count runs its own exchange — each a world of one (this probe drives its own
    //    link and sends, not a tensor-parallel world). ──
    let table =
        scratchy_target_spyre_card::runtime::CardTable::enumerate().map_err(|e| anyhow!("{e}"))?;
    let have = table.cards().len();
    // `[N] mode`: N defaults to every card the host has, the mode to `vendor`.
    let mut args = std::env::args().skip(1);
    let (n, mode) = match args.next().as_deref() {
        None => (have, Mode::Vendor),
        Some(first) => match first.parse::<usize>() {
            Ok(n) => (
                n,
                match args.next().as_deref() {
                    None => Mode::Vendor,
                    Some(arg) => Mode::of(arg)?,
                },
            ),
            Err(_) => (have, Mode::of(first)?),
        },
    };
    ensure!(
        n >= 2 && n <= have,
        "N={n}: needs 2..={have} cards (senlib enumerates {have})"
    );
    let pcis: Vec<String> = table.cards()[..n]
        .iter()
        .map(|c| c.pci().to_string())
        .collect();
    let own: Vec<CardTable> = table.cards()[..n]
        .iter()
        .map(|c| CardTable::of([(c.index(), c.pci().to_string())]))
        .collect();

    // ── Phase 1: open every card, in parallel — each bring-up is ~5.7 s, and every open runs
    //    on its own thread (`open_cards` spawns, `join_opens` joins only after every open is
    //    already in flight). ──
    let t = Instant::now();
    let worlds: Vec<WorldCards<TpWorld<1>>> = own
        .into_iter()
        .map(|one| WorldCards::claim(&one).map_err(|e| anyhow!("{e}")))
        .collect::<Result<_>>()?;
    let opens: Vec<_> = worlds.iter().map(|w| w.open_cards()).collect();
    let opened: Vec<_> = opens
        .into_iter()
        .map(|o| WorldCards::join_opens(o).map_err(|e| anyhow!("{e}")))
        .collect::<Result<_>>()?;
    println!(
        "opened {n} cards in one process in {:.2}s: {}",
        t.elapsed().as_secs_f64(),
        pcis.join(" ")
    );

    // ── Phase 2 + the exchange: ONE scope hosts every card's completion threads for exactly
    //    the probe's span — the runtimes are plain locals borrowing `opened`, built inside it,
    //    used inside it, and their completion threads join at its close. Nothing is leaked. ──
    std::thread::scope(|scope| {
        let cards: Vec<CardRuntime<'_>> = worlds
            .iter()
            .zip(opened.iter())
            .map(|(w, o)| w.build_all(o, scope).map_err(|e| anyhow!("{e}")))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .map(|r| r.into_first())
            .collect();
        exchange(&cards, n, mode)
    })
}

/// The exchange over the opened runtimes: link, buffers, landing, sends, waits, judgment.
fn exchange(cards: &[CardRuntime<'_>], n: usize, mode: Mode) -> Result<()> {
    // ── Link: every card's BAT entry for every peer. ──
    let bar2: Vec<u64> = cards
        .iter()
        .map(|c| pci_addr_for_region(c.card().pci(), 2).map_err(|e| anyhow!("{e}")))
        .collect::<Result<_>>()?;
    for (me, c) in cards.iter().enumerate() {
        let st = c
            .card()
            .check_and_clear_rdma_status()
            .map_err(|e| anyhow!("{e}"))?;
        ensure!(!st, "card {me}: rdma_status latched before linking");
        for (peer, &addr) in bar2.iter().enumerate() {
            if peer != me {
                c.card()
                    .set_base_addr_table(peer as u8, addr)
                    .map_err(|e| anyhow!("card {me} BAT[{peer}]: {e}"))?;
            }
        }
    }
    println!("linked: {} base-address-table entries", n * (n - 1));

    // ── Buffers: own pattern in own slot, zeros elsewhere. ──
    let sides: Vec<Side<'_, '_>> = cards
        .iter()
        .enumerate()
        .map(|(i, c)| Side::new(c, n).with_context(|| format!("card {i}")))
        .collect::<Result<_>>()?;
    let patterns: Vec<Vec<u8>> = (0..n).map(pattern).collect();
    for (i, s) in sides.iter().enumerate() {
        let mut host = vec![0u8; SLOT_BYTES as usize * n];
        host[i * SLOT_BYTES as usize..(i + 1) * SLOT_BYTES as usize].copy_from_slice(&patterns[i]);
        s.dma(host.as_ptr(), host.len(), DmaDirection::HostToDevice)
            .with_context(|| format!("card {i} H2D"))?;
    }

    // ── Landing: on receiver j, XSEG[sid = sender i] → j's slot i (vendor), or XSEG[2i+m] → j's
    //    slot i's half m (the per-message shapes). All before any send. ──
    let mut xseg = vec![vec![0u32; 2 * n]; n];
    for (j, s) in sides.iter().enumerate() {
        for i in (0..n).filter(|&i| i != j) {
            let point = |sid: u8, va: u64| -> Result<u32> {
                let word = rdma_xseg(va + Flits::BYTES).map_err(|e| anyhow!("{e}"))?;
                s.card
                    .card()
                    .write_xseg(sid, word)
                    .map_err(|e| anyhow!("card {j} XSEG[{sid}]: {e}"))?;
                let back = s.card.card().read_xseg(sid).map_err(|e| anyhow!("{e}"))?;
                ensure!(
                    back == word,
                    "card {j} XSEG[{sid}] read back 0x{back:08X}, wrote 0x{word:08X}"
                );
                Ok(word)
            };
            match mode {
                Mode::Vendor => xseg[j][i] = point(i as u8, s.slot_va(i))?,
                Mode::Run | Mode::Alone | Mode::Split => {
                    for m in 0..MSGS {
                        xseg[j][2 * i + m as usize] = point(Mode::sid(i, m), s.half_va(i, m))?;
                    }
                }
            }
        }
    }

    // ── The exchange. `vendor`: card i's slot i to every peer on sid i through the vendor door,
    //    then one wait per peer on that peer's sid. `run2`/`alone2`: ONE submit_batch per card —
    //    card i's slot i, BOTH halves, to every peer (message m on sid 2i+m, the run's flags in
    //    `run2`, `ALONE` in `alone2`, the CTRL barrier on the batch's first send only), then one
    //    wdone wait per message slot on its sender's sid — the production flush's own shape. ──
    let t = Instant::now();
    match mode {
        Mode::Vendor => {
            for (i, s) in sides.iter().enumerate() {
                for (j, peer) in sides.iter().enumerate().filter(|&(j, _)| j != i) {
                    s.stream
                        .launch_p2p_rdma_send(
                            P2PRdmaSendParams {
                                sync_key: format!("ag-{i}-{j}"),
                                dest_ranks: vec![j as u64],
                                sid_list: vec![i as u32],
                                device_address: s.slot(i),
                                op_name: format!("all-gather {i}->{j}"),
                                dst_offset: peer.slot_va(i),
                            },
                            false,
                            None,
                        )
                        .map_err(|e| anyhow!("card {i} send to {j}: {e}"))?;
                }
            }
            // `ONE_WAIT=1`: each card waits ONCE, on every peer's sid in one wait CB (the wdone
            // barrier's list bitmask) — the question being whether one multi-sid wait consumes one
            // count per sid.
            let one_wait = std::env::var_os("ONE_WAIT").is_some();
            for (j, s) in sides.iter().enumerate() {
                let peers: Vec<u32> = (0..n).filter(|&i| i != j).map(|i| i as u32).collect();
                let lists: Vec<Vec<u32>> = if one_wait {
                    vec![peers]
                } else {
                    peers.into_iter().map(|i| vec![i]).collect()
                };
                for sid_list in lists {
                    s.stream
                        .launch_p2p_rdma_wait(P2PRdmaWaitParams { sid_list }, false, None)
                        .map_err(|e| anyhow!("card {j} wait: {e}"))?;
                }
            }
            for (j, s) in sides.iter().enumerate() {
                s.stream
                    .synchronize()
                    .map_err(|e| anyhow!("card {j}: synchronize after the exchange: {e}"))?;
            }
        }
        Mode::Run | Mode::Alone | Mode::Split => {
            let cap_msg = WireBytes::new(CAP).context("the message cap is not a legal message")?;
            let streams: Vec<Stream<'_>> = cards
                .iter()
                .map(|c| Stream::create(c).map_err(|e| anyhow!("create stream: {e}")))
                .collect::<Result<_>>()?;
            let waits = |me: usize| -> Vec<BatchedCb> {
                (0..n)
                    .filter(|&from| from != me)
                    .flat_map(|from| {
                        (0..MSGS).map(move |m| BatchedCb::Wait {
                            sids: vec![Mode::sid(from, m)],
                        })
                    })
                    .collect()
            };
            for (i, s) in sides.iter().enumerate() {
                let mut sends = Vec::new();
                let mut first_send = true;
                for k in (0..n).filter(|&k| k != i) {
                    for m in 0..MSGS {
                        sends.push(BatchedCb::Send(RdmaSend {
                            sid: Mode::sid(i, m),
                            dest_tid: k as u8,
                            bytes: cap_msg,
                            source: RdmaEnd {
                                base: s.half(i, m),
                                segment: s.segment as u8,
                                offset: 0,
                            },
                            window: RdmaEnd {
                                base: s.half(i, m),
                                segment: s.segment as u8,
                                offset: 0,
                            },
                            place: match mode {
                                Mode::Alone => RunPlace::ALONE,
                                _ => RunPlace {
                                    first: m == 0,
                                    last: m == MSGS - 1,
                                },
                            },
                            barrier: std::mem::replace(&mut first_send, false),
                        }));
                    }
                }
                match mode {
                    // `split2`: the sends are their own doorbell, synchronized on their own — a
                    // hang THERE names the message-1 send as the stuck CB; only once they retire
                    // do the waits go down.
                    Mode::Split => {
                        streams[i]
                            .submit_batch(&sends, true, None)
                            .map_err(|e| anyhow!("card {i} sends ({} items): {e}", sends.len()))?;
                    }
                    _ => {
                        let mut items = sends;
                        items.extend(waits(i));
                        streams[i]
                            .submit_batch(&items, true, None)
                            .map_err(|e| anyhow!("card {i} batch ({} items): {e}", items.len()))?;
                    }
                }
            }
            if matches!(mode, Mode::Split) {
                println!(
                    "split2: sends submitted; synchronizing the sends ALONE (a hang here = the \
                     message-1 send never retires; retiring sends = the wait is the stuck CB)"
                );
                for j in 0..n {
                    streams[j]
                        .synchronize()
                        .map_err(|e| anyhow!("card {j}: synchronize after the sends: {e}"))?;
                }
                println!("split2: every card's sends RETIRED");
                for (j, s) in sides.iter().enumerate() {
                    let sids: Vec<u8> = (0..n)
                        .filter(|&i| i != j)
                        .flat_map(|i| (0..MSGS).map(move |m| Mode::sid(i, m)))
                        .collect();
                    let counts = sids
                        .iter()
                        .map(|&sid| {
                            s.card
                                .card()
                                .read_wdone(sid)
                                .map_err(|e| anyhow!("card {j} wdone[{sid}]: {e}"))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    println!(
                        "split2 census (sends retired, no waits yet): card {j}: {}",
                        sids.iter()
                            .zip(&counts)
                            .map(|(&sid, wd)| format!("sid{sid}={wd}"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    );
                }
                println!("split2: submitting the waits (a hang here = the wait cannot consume)");
                for i in 0..n {
                    let items = waits(i);
                    streams[i]
                        .submit_batch(&items, true, None)
                        .map_err(|e| anyhow!("card {i} waits ({} items): {e}", items.len()))?;
                }
            }
            // ── The census, BEFORE synchronizing: a wait that never satisfies leaves its sid's
            //    count wherever the landing left it — 0 for a message that never landed, 1 for one
            //    that landed and was never consumed. That split is the whole probe. ──
            let watched: Vec<Vec<u8>> = (0..n)
                .map(|j| {
                    (0..n)
                        .filter(|&i| i != j)
                        .flat_map(|i| (0..MSGS).map(move |m| Mode::sid(i, m)))
                        .collect()
                })
                .collect();
            let started = Instant::now();
            for checkpoint in [1_u64, 5, 10, 30] {
                let target = std::time::Duration::from_secs(checkpoint);
                if let Some(remain) = target.checked_sub(started.elapsed()) {
                    std::thread::sleep(remain);
                }
                for (j, s) in sides.iter().enumerate() {
                    let counts = watched[j]
                        .iter()
                        .map(|&sid| {
                            s.card
                                .card()
                                .read_wdone(sid)
                                .map_err(|e| anyhow!("card {j} wdone[{sid}]: {e}"))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    println!(
                        "census t={checkpoint:>2}s card {j}: {}",
                        watched[j]
                            .iter()
                            .zip(&counts)
                            .map(|(&sid, wd)| format!("sid{sid}={wd}"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    );
                }
            }
            println!("synchronizing (a wedged wait hangs here; the census above is the verdict)");
            for j in 0..n {
                streams[j]
                    .synchronize()
                    .map_err(|e| anyhow!("card {j}: synchronize after the exchange: {e}"))?;
            }
        }
    }
    let secs = t.elapsed().as_secs_f64();
    let moved = (n * (n - 1)) as u64 * SLOT_BYTES;
    println!(
        "exchange: {} messages of {} KiB, {:.3} ms, {:.2} GB/s aggregate",
        n * (n - 1) * MSGS as usize,
        CAP / 1024,
        secs * 1e3,
        moved as f64 / secs / 1e9
    );

    // ── Judge every card. ──
    let mut bad = 0usize;
    for (j, s) in sides.iter().enumerate() {
        let mut got = vec![0u8; SLOT_BYTES as usize * n];
        s.dma(got.as_mut_ptr(), got.len(), DmaDirection::DeviceToHost)
            .with_context(|| format!("card {j} D2H"))?;
        for (i, slot) in got.chunks(SLOT_BYTES as usize).enumerate() {
            if let Some(b) = (0..slot.len()).find(|&b| slot[b] != patterns[i][b]) {
                println!(
                    "card {j} slot {i}: MISMATCH at byte {b}: got {} want {}",
                    slot[b], patterns[i][b]
                );
                bad += 1;
            }
        }
        let sids: Vec<(u8, usize)> = match mode {
            Mode::Vendor => (0..n).filter(|&i| i != j).map(|i| (i as u8, i)).collect(),
            Mode::Run | Mode::Alone | Mode::Split => (0..n)
                .filter(|&i| i != j)
                .flat_map(|i| (0..MSGS).map(move |m| (Mode::sid(i, m), 2 * i + m as usize)))
                .collect(),
        };
        for (sid, xs_idx) in sids {
            let wd = s.card.card().read_wdone(sid).map_err(|e| anyhow!("{e}"))?;
            let xs = s.card.card().read_xseg(sid).map_err(|e| anyhow!("{e}"))?;
            if wd != 0 {
                println!("card {j} sid {sid}: wdone {wd} left over after its wait");
                bad += 1;
            }
            if xs != xseg[j][xs_idx] {
                println!("card {j} XSEG[{sid}] re-pointed to 0x{xs:08X}");
                bad += 1;
            }
        }
        let st = s
            .card
            .card()
            .check_and_clear_rdma_status()
            .map_err(|e| anyhow!("{e}"))?;
        if st {
            println!("card {j}: rdma_status latched after the exchange");
            bad += 1;
        }
    }
    if bad != 0 {
        bail!("ALL-GATHER FAILED on {n} cards: {bad} finding(s) above");
    }
    println!(
        "ALL-GATHER PASS (N={n}): every card holds all {n} cards' slots byte-exact; wdone 0, XSEG \
         unchanged, rdma_status clean on every card"
    );
    Ok(())
}

// SPDX-License-Identifier: Apache-2.0
//! ONE PROCESS, N CARDS, AN ALL-GATHER OVER THE REAL FLEX PATH — the card-to-card transport
//! tensor parallelism runs on, proven on the card.
//!
//! Every card holds an N-slot buffer. Card `i` starts with its own pattern in slot `i` and zeroes
//! everywhere else; it then F-sends slot `i` to every peer, landing in that peer's slot `i`. When
//! the run ends, every card's buffer must hold every card's pattern, byte for byte.
//!
//! The mechanism, all of it the production path (`CardRuntime::open`, the flex scheduler's
//! `launch_p2p_rdma_send` / `launch_p2p_rdma_wait`):
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
//! ```text
//! cargo run --release -p scratchy-target-spyre --features spyre-hw --example cards_all_gather -- <N>
//! ```
//! N cards: senlib's cards 0..N, opened by PCI address.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail, ensure};

use flex_rs::address::{ByteOffset, ByteSize, Chunk, CompositeAddress, LogicalAddress};
use flex_rs::allocator::AllocationDirective;
use flex_rs::control_block_wire::Flits;
use flex_rs::dma::{DmaDirection, DmaParams, DmaShaping, HostVirtualAddress};
use flex_rs::hal_sbf::rdma_xseg;
use flex_rs::scheduler::{
    DeviceAddressResolver, FlexAllocatorAddressResolver, P2PRdmaSendParams, P2PRdmaWaitParams,
};
use flex_rs::senlib_ffi::pci_addr_for_region;
use flex_rs::stream::{RuntimeStream, RuntimeStreamMode, RuntimeStreamPriority};
use scratchy_target_spyre::runtime::CardRuntime as AnyCard;

/// A raw flex-level probe: the cards are driven below the world-typed p2p link, so they are opened
/// as one card's runtime each.
type CardRuntime = AnyCard<scratchy_target_spyre::bundle_code::collective::TpWorld<1>>;

/// One slot: one card's message.
const SLOT_BYTES: u64 = 64 * 1024;

/// Card `i`'s pattern: distinct per card, so a send landing in another card's slot is a mismatch.
fn pattern(i: usize) -> Vec<u8> {
    (0..SLOT_BYTES as usize)
        .map(|j| ((j * 7 + 3 + 31 * (i + 1)) ^ (j >> 8)) as u8)
        .collect()
}

/// A device byte address as the RDMA engine reads it: the segment in bits 27.. of the flit index.
fn flat_va(segment: u64, offset_bytes: u64) -> u64 {
    ((segment << 27) | (offset_bytes >> 7)) << 7
}

/// One card's side of the run: its runtime, one stream, and its N-slot buffer.
struct Side {
    card: Arc<CardRuntime>,
    stream: Arc<RuntimeStream>,
    buf: Arc<CompositeAddress>,
    /// The buffer's (segment, byte offset) on this card — what a peer's send lands against.
    segment: u64,
    offset_bytes: u64,
}

impl Side {
    fn new(card: Arc<CardRuntime>, n: usize) -> Result<Self> {
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

fn main() -> Result<()> {
    let n: usize = std::env::args()
        .nth(1)
        .context("usage: cards_all_gather <N>")?
        .parse()
        .context("N is a card count")?;
    let have = flex_rs::senlib_ffi::sen_pci_card_count();
    ensure!(
        n >= 2 && n as i64 <= have,
        "N={n}: needs 2..={have} cards (senlib enumerates {have})"
    );

    // ── Open every card, in parallel: each bring-up is ~5.7 s. ──
    let t = Instant::now();
    let cards: Vec<Arc<CardRuntime>> = (0..n as u32)
        .map(|i| std::thread::spawn(move || CardRuntime::open(i)))
        .collect::<Vec<_>>()
        .into_iter()
        .enumerate()
        .map(|(i, h)| {
            h.join()
                .map_err(|_| anyhow!("card {i}: bring-up thread panicked"))?
                .map_err(|e| anyhow!("card {i}: {e}"))
        })
        .collect::<Result<_>>()?;
    println!(
        "opened {n} cards in one process in {:.2}s: {}",
        t.elapsed().as_secs_f64(),
        cards
            .iter()
            .map(|c| c.card().pci().to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );

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
        ensure!(st == 0, "card {me}: rdma_status {st} before linking");
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
    let sides: Vec<Side> = cards
        .into_iter()
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

    // ── Landing: on receiver j, XSEG[sid = sender i] → j's slot i. All before any send. ──
    let mut xseg = vec![vec![0u32; n]; n];
    for (j, s) in sides.iter().enumerate() {
        for i in (0..n).filter(|&i| i != j) {
            let word = rdma_xseg(s.slot_va(i) + Flits::BYTES).map_err(|e| anyhow!("{e}"))?;
            s.card
                .card()
                .write_xseg(i as u8, word)
                .map_err(|e| anyhow!("card {j} XSEG[{i}]: {e}"))?;
            let back = s
                .card
                .card()
                .read_xseg(i as u8)
                .map_err(|e| anyhow!("{e}"))?;
            ensure!(
                back == word,
                "card {j} XSEG[{i}] read back 0x{back:08X}, wrote 0x{word:08X}"
            );
            xseg[j][i] = word;
        }
    }

    // ── Send then wait, every card at once: card i's slot i to every peer on sid i, then one
    //    wait per peer on that peer's sid. ──
    let t = Instant::now();
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
    for (j, s) in sides.iter().enumerate() {
        for i in (0..n).filter(|&i| i != j) {
            s.stream
                .launch_p2p_rdma_wait(
                    P2PRdmaWaitParams {
                        sid_list: vec![i as u32],
                    },
                    false,
                    None,
                )
                .map_err(|e| anyhow!("card {j} wait on {i}: {e}"))?;
        }
    }
    for (j, s) in sides.iter().enumerate() {
        s.stream
            .synchronize()
            .map_err(|e| anyhow!("card {j}: synchronize after the exchange: {e}"))?;
    }
    let secs = t.elapsed().as_secs_f64();
    let moved = (n * (n - 1)) as u64 * SLOT_BYTES;
    println!(
        "exchange: {} sends of {} KiB, {:.3} ms, {:.2} GB/s aggregate",
        n * (n - 1),
        SLOT_BYTES / 1024,
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
        for i in (0..n).filter(|&i| i != j) {
            let wd = s
                .card
                .card()
                .read_wdone(i as u8)
                .map_err(|e| anyhow!("{e}"))?;
            let xs = s
                .card
                .card()
                .read_xseg(i as u8)
                .map_err(|e| anyhow!("{e}"))?;
            if wd != 0 {
                println!("card {j} sid {i}: wdone {wd} left over after its wait");
                bad += 1;
            }
            if xs != xseg[j][i] {
                println!("card {j} XSEG[{i}] re-pointed to 0x{xs:08X}");
                bad += 1;
            }
        }
        let st = s
            .card
            .card()
            .check_and_clear_rdma_status()
            .map_err(|e| anyhow!("{e}"))?;
        if st != 0 {
            println!("card {j}: rdma_status {st} after the exchange");
            bad += 1;
        }
    }
    if bad != 0 {
        bail!("ALL-GATHER FAILED on {n} cards: {bad} finding(s) above");
    }
    println!(
        "ALL-GATHER PASS (N={n}): every card holds all {n} cards' slots byte-exact; wdone 0, XSEG \
         unchanged, rdma_status 0 on every card"
    );
    Ok(())
}

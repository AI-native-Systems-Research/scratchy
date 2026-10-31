//! tp38 — the landing-slot cell: tp37's transport, one step on. A DIAGNOSTIC, not product code,
//! gated exactly as `tp37_transport` is (it builds only under an explicit
//! `--example tp38_landing_slots` on a Spyre host where the shim really links).
//!
//! What it proves, and nothing more: several F-sends from one rank land at SEVERAL fixed
//! addresses of the peer's window, each through its OWN sid's `XSEG` entry, every entry
//! programmed ONCE before any send — the mechanism scratchy's tensor-parallel pipeline runs on
//! (`ktir_superdsc::collective::LandingSlot`): an F-send lands where its sid's entry points and
//! nowhere else (#1009, E8a), so two messages that must land apart signal on two sids.
//!
//! The slots are the pipeline's own: two window cuts × two messages, at the sids
//! `Sid::of(rank 0, slot)` names — `cut · MAX_CUT_MESSAGES + message` for rank 0 with
//! `MAX_CUT_MESSAGES = 8`, i.e. 0, 1, 8, 9 — so the cell also exercises sids that are not a
//! rank number.
//!
//! The receiver asserts, per slot:
//! - `read_xseg(sid)` reads back what was written, before AND after the sends (an F-send never
//!   re-points an entry);
//! - its wait unblocks well under the wait's ~5 s timeout, and `read_wdone(sid)` is 0 after (each
//!   send's +1 consumed by its own slot's wait, nothing left over on any sid);
//! - the slot's bytes are byte-exact its own pattern. Each slot carries a distinct pattern and
//!   starts zeroed, so a send landing in the wrong slot — e.g. every send collapsing onto slot 0's
//!   address — is a mismatch.
//!
//! Invocation (one process per card, rank 1 FIRST — tp37's env shape):
//!
//! ```text
//! export RANK=<0|1> WORLD_SIZE=2
//! export AIU_WORLD_RANK_0=<bdf0> AIU_WORLD_RANK_1=<bdf1>
//! export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
//! cargo build --release -p flex-rs --example tp38_landing_slots
//! ./target/release/examples/tp38_landing_slots
//! ```
//!
//! Coordination is tp37's handshake file, under `/work` (never /tmp): `/work/tp38-r1.json`,
//! written by rank 1 once every wait is pending. The launcher deletes a stale one before a run.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use flex_rs::address::{ByteOffset, ByteSize, Chunk, CompositeAddress, LogicalAddress};
use flex_rs::allocator::{AllocationDirective, DEVICE_ALIGNMENT, FlexAllocator, FlexAllocatorInit};
use flex_rs::control_block_wire::Flits;
use flex_rs::device_memory_allocator::DeviceMemoryAllocator;
use flex_rs::dma::{DmaDirection, DmaParams, DmaShaping, HostVirtualAddress};
use flex_rs::flex_config::{DeviceKind, FlexConfig};
use flex_rs::hal_sbf::rdma_xseg;
use flex_rs::iommu::PfIommuMapper;
use flex_rs::runtime::RuntimeContext;
use flex_rs::scheduler::{
    DeviceAddressResolver, FlexAllocatorAddressResolver, P2PRdmaSendParams, P2PRdmaWaitParams,
    SchedulerBackendConfig, SenlibHardwareSubmit, SenlibQueueCapacity, create_scheduler,
};
use flex_rs::senlib_ffi::{
    check_and_clear_rdma_status, flex_senlib_device_memory_size, init_runtime_bdf,
    pci_addr_for_region, read_wdone, read_xseg, resolved_pci, set_base_addr_table, write_xseg,
};
use flex_rs::stream::{RuntimeStream, RuntimeStreamMode, RuntimeStreamPriority};
use flex_rs::topology::populate_domain_topology;

/// One landing slot's bytes — one message each.
const SLOT_BYTES: u64 = 4096;
/// The slots, as `(cut, message)`: two cuts × two messages.
const SLOTS: [(u64, u64); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];
/// `ktir_superdsc::collective::MAX_CUT_MESSAGES` — the stride of a cut's slots in a rank's sids.
const MAX_CUT_MESSAGES: u64 = 8;
/// The device buffer: every slot, back to back.
const BUF_BYTES: u64 = 64 * 1024;
const HANDSHAKE_PATH: &str = "/work/tp38-r1.json";
/// tp37's bound: a wait that consumed a count unblocks well under its ~5 s timeout.
const WAIT_GENUINE_BOUND: Duration = Duration::from_millis(4500);

/// Slot `i`'s place in the buffer: its cut's two slots, then its message.
fn slot_offset(i: usize) -> u64 {
    i as u64 * SLOT_BYTES
}

/// The sid rank 0's message into slot `i` signals on: `Sid::of(rank 0, slot)`.
fn slot_sid(i: usize) -> u8 {
    let (cut, message) = SLOTS[i];
    (cut * MAX_CUT_MESSAGES + message) as u8
}

/// Slot `i`'s pattern: distinct per slot, so a send landing in another slot's place is a mismatch.
fn pattern_byte(i: usize, j: usize) -> u8 {
    ((j * 7 + 3 + 31 * (i + 1)) & 0xFF) as u8
}

/// A slot's flat device VA, as tp37 derives its one destination.
fn flat_va(segment: u64, offset_bytes: u64) -> u64 {
    ((segment << 27) | (offset_bytes >> 7)) << 7
}

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let rank = FlexConfig::rdma_world_rank().0;
    let size = FlexConfig::rdma_world_size().0;
    if size != 2 || !(rank == 0 || rank == 1) {
        eprintln!("[tp38 rank {rank}] REFUSED: RANK={rank} WORLD_SIZE={size} — needs 2 ranks");
        return 2;
    }
    let say = |msg: String| println!("[tp38 rank {rank}] {msg}");

    // ── tp37's bind and its proof, then tp37's symmetric BAT (L5, already proven there). ──
    let bdf = match FlexConfig::rdma_get_pcie_address(Some(rank)) {
        Ok(b) => b.0,
        Err(e) => {
            say(format!("FATAL: AIU_WORLD_RANK_{rank}: {e}"));
            return 1;
        }
    };
    if let Err(e) = init_runtime_bdf(&bdf) {
        say(format!("FATAL: init_runtime_bdf({bdf}) failed: {e}"));
        return 1;
    }
    match resolved_pci() {
        Ok(r) if r == bdf => say(format!("BIND-PROOF: asked {bdf}, senlib resolved {r}")),
        Ok(r) => {
            say(format!("FATAL (BIND-PROOF): resolved {r} ≠ asked {bdf}"));
            return 1;
        }
        Err(e) => {
            say(format!("FATAL: resolved_pci() failed: {e}"));
            return 1;
        }
    }
    let peer_rank = 1 - rank;
    let peer_bdf = match FlexConfig::rdma_get_pcie_address(Some(peer_rank)) {
        Ok(b) => b.0,
        Err(e) => {
            say(format!("FATAL: AIU_WORLD_RANK_{peer_rank}: {e}"));
            return 1;
        }
    };
    let peer_bar2 = match pci_addr_for_region(&peer_bdf, 2) {
        Ok(v) => v,
        Err(e) => {
            say(format!(
                "FATAL: pci_addr_for_region({peer_bdf}, 2) failed: {e}"
            ));
            return 1;
        }
    };
    if let Err(e) = set_base_addr_table(peer_rank as u8, peer_bar2) {
        say(format!(
            "FATAL: set_base_addr_table({peer_rank}, 0x{peer_bar2:X}) failed: {e}"
        ));
        return 1;
    }
    say(format!("BAT: tid {peer_rank} ← 0x{peer_bar2:X}"));

    // ── tp37's bring-up. ──
    let device_memory_size = unsafe { flex_senlib_device_memory_size() };
    if device_memory_size == 0 {
        say("FATAL: the card reports 0 device memory".to_string());
        return 1;
    }
    let allocator = Arc::new(Mutex::new(
        match FlexAllocator::new(
            FlexAllocatorInit::default(),
            populate_domain_topology(device_memory_size),
            DeviceMemoryAllocator::new(DEVICE_ALIGNMENT),
        ) {
            Ok(a) => a,
            Err(e) => {
                say(format!("FATAL: FlexAllocator::new failed: {e}"));
                return 1;
            }
        },
    ));
    let resolver = Arc::new(FlexAllocatorAddressResolver::new(allocator.clone()));
    let dyn_resolver: Arc<dyn DeviceAddressResolver> = resolver.clone();
    let scheduler = match create_scheduler(
        &DeviceKind::Pf,
        SchedulerBackendConfig::Real(
            Arc::new(SenlibHardwareSubmit::new()),
            Arc::new(SenlibQueueCapacity),
            dyn_resolver,
            PfIommuMapper::new(),
        ),
    ) {
        Ok(s) => s,
        Err(e) => {
            say(format!("FATAL: create_scheduler failed: {e}"));
            return 1;
        }
    };
    let context = match RuntimeContext::create(None, scheduler, allocator.clone()) {
        Ok(c) => c,
        Err(e) => {
            say(format!("FATAL: RuntimeContext::create failed: {e}"));
            return 1;
        }
    };
    let stream = match context
        .create_stream(
            RuntimeStreamPriority::Normal,
            RuntimeStreamMode::StrictOrdering,
            true,
        )
        .and_then(|id| context.get_stream(id))
    {
        Ok(Some(s)) => s,
        Ok(None) => {
            say("FATAL: get_stream returned None for a just-created stream".to_string());
            return 1;
        }
        Err(e) => {
            say(format!("FATAL: stream bring-up failed: {e}"));
            return 1;
        }
    };
    let buf = {
        let mut guard = allocator.lock().unwrap_or_else(|e| e.into_inner());
        match guard.allocate(ByteSize(BUF_BYTES), &AllocationDirective::default()) {
            Ok(a) => a,
            Err(e) => {
                say(format!("FATAL: allocate({BUF_BYTES}) failed: {e}"));
                return 1;
            }
        }
    };
    let chunk0 = buf.chunks()[0];
    let Some((segment, _, _)) = resolver.resolve(&chunk0) else {
        say("FATAL: the buffer's region did not resolve to a segment".to_string());
        return 1;
    };
    let offset_bytes = chunk0.addr.offset.0;
    if !offset_bytes.is_multiple_of(Flits::BYTES) {
        say(format!(
            "FATAL: buffer offset {offset_bytes} is not flit-aligned"
        ));
        return 1;
    }
    let slot_addr = |i: usize| {
        Arc::new(CompositeAddress::from_chunk(Chunk::new(
            LogicalAddress::new(
                chunk0.addr.region_id,
                ByteOffset(chunk0.addr.offset.0 + slot_offset(i)),
            ),
            ByteSize(SLOT_BYTES),
            chunk0.domain_id,
        )))
    };
    let all = Arc::new(CompositeAddress::from_chunk(Chunk::new(
        chunk0.addr,
        ByteSize(SLOT_BYTES * SLOTS.len() as u64),
        chunk0.domain_id,
    )));

    // ── H2D: rank 0 writes every slot's own pattern, rank 1 zeroes every slot. ──
    let mut host = vec![0u8; (SLOT_BYTES as usize) * SLOTS.len()];
    if rank == 0 {
        for (i, slot) in host.chunks_mut(SLOT_BYTES as usize).enumerate() {
            for (j, b) in slot.iter_mut().enumerate() {
                *b = pattern_byte(i, j);
            }
        }
    }
    let h2d = DmaParams::with_explicit_size(
        HostVirtualAddress(host.as_ptr() as usize),
        ByteSize(host.len() as u64),
        DmaDirection::HostToDevice,
        Some(all.clone()),
        DmaShaping::default(),
    )
    .expect("device_address is Some");
    if let Err(e) = stream
        .launch_h2d(h2d, None)
        .and_then(|_| stream.synchronize())
    {
        say(format!("FATAL: H2D failed: {e}"));
        return 1;
    }
    for i in 0..SLOTS.len() {
        print_diag(&say, slot_sid(i), "pre-op");
    }

    if rank == 1 {
        receiver_flow(&say, &stream, &all, u64::from(segment), offset_bytes)
    } else {
        sender_flow(&say, &stream, &slot_addr)
    }
}

/// `read_wdone(sid)` + `check_and_clear_rdma_status()`, labeled `when`.
fn print_diag(say: &dyn Fn(String), sid: u8, when: &str) {
    match read_wdone(sid) {
        Ok(v) => say(format!("{when} read_wdone({sid})={v}")),
        Err(e) => say(format!("{when} read_wdone({sid}) FAILED: {e}")),
    }
    match check_and_clear_rdma_status() {
        Ok(0) => {}
        Ok(v) => say(format!(
            "⚠️ {when} rdma_status={v} NONZERO — RDMA ERROR STATE"
        )),
        Err(e) => say(format!("{when} rdma_status read FAILED: {e}")),
    }
}

/// tp37's handshake parse: one field out of `{"segment": N, "offset_bytes": M}`.
fn json_field_u64(text: &str, key: &str) -> Option<u64> {
    let key_pat = format!("\"{key}\"");
    let i = text.find(&key_pat)?;
    let rest = text[i + key_pat.len()..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// Rank 1: point every slot's `XSEG` entry at the slot (once, before any send), submit one wait
/// per slot, publish the buffer, then judge every slot.
fn receiver_flow(
    say: &dyn Fn(String),
    stream: &Arc<RuntimeStream>,
    all: &Arc<CompositeAddress>,
    segment: u64,
    offset_bytes: u64,
) -> i32 {
    let mut words = Vec::with_capacity(SLOTS.len());
    for (i, slot) in SLOTS.iter().enumerate() {
        let (sid, dst) = (slot_sid(i), flat_va(segment, offset_bytes + slot_offset(i)));
        let word = match rdma_xseg(dst + Flits::BYTES) {
            Ok(w) => w,
            Err(e) => {
                say(format!("FATAL: rdma_xseg(slot {i}) refused: {e}"));
                return 1;
            }
        };
        if let Err(e) = write_xseg(sid, word) {
            say(format!(
                "FATAL: write_xseg({sid}, 0x{word:08X}) failed: {e}"
            ));
            return 1;
        }
        match read_xseg(sid) {
            Ok(v) if v == word => say(format!(
                "slot {i} {slot:?}: XSEG[{sid}] = 0x{word:08X} (dst 0x{dst:X}), read back"
            )),
            Ok(v) => {
                say(format!(
                    "FATAL: read_xseg({sid})=0x{v:08X} ≠ written 0x{word:08X}"
                ));
                return 1;
            }
            Err(e) => {
                say(format!("FATAL: read_xseg({sid}) failed: {e}"));
                return 1;
            }
        }
        words.push(word);
    }

    // One wait per slot, every one pending before the handshake that gates the sends.
    let t_submit = Instant::now();
    let done: Arc<Mutex<Vec<Option<Duration>>>> = Arc::new(Mutex::new(vec![None; SLOTS.len()]));
    for i in 0..SLOTS.len() {
        let done_cb = done.clone();
        let params = P2PRdmaWaitParams {
            sid_list: vec![u32::from(slot_sid(i))],
        };
        if let Err(e) = stream.launch_p2p_rdma_wait(
            params,
            false,
            Some(Box::new(move || {
                done_cb.lock().unwrap()[i] = Some(t_submit.elapsed());
            })),
        ) {
            say(format!("FATAL: launch_p2p_rdma_wait(slot {i}) failed: {e}"));
            return 1;
        }
    }
    if let Err(e) = std::fs::write(
        HANDSHAKE_PATH,
        format!("{{\"segment\": {segment}, \"offset_bytes\": {offset_bytes}}}"),
    ) {
        say(format!("FATAL: writing {HANDSHAKE_PATH} failed: {e}"));
        return 1;
    }
    say(format!(
        "every wait pending; handshake written (segment {segment}, offset {offset_bytes})"
    ));

    let deadline = Instant::now() + WAIT_GENUINE_BOUND * SLOTS.len() as u32;
    while Instant::now() < deadline && done.lock().unwrap().iter().any(Option::is_none) {
        std::thread::sleep(Duration::from_millis(10));
    }
    let done = done.lock().unwrap().clone();
    for (i, d) in done.iter().enumerate() {
        match d {
            Some(d) => say(format!("slot {i}: wait unblocked {d:?} after submit")),
            None => {
                say(format!(
                    "FATAL (STOP FINDING): slot {i}'s wait on sid {} never unblocked — its send's \
                     count never reached it",
                    slot_sid(i)
                ));
                return 1;
            }
        }
    }

    let mut got = vec![0u8; (SLOT_BYTES as usize) * SLOTS.len()];
    let d2h = DmaParams::with_explicit_size(
        HostVirtualAddress(got.as_mut_ptr() as usize),
        ByteSize(got.len() as u64),
        DmaDirection::DeviceToHost,
        Some(all.clone()),
        DmaShaping::default(),
    )
    .expect("device_address is Some");
    if let Err(e) = stream
        .launch_d2h(d2h, None)
        .and_then(|_| stream.synchronize())
    {
        say(format!("FATAL: D2H after the waits failed: {e}"));
        return 1;
    }

    let mut verdict = 0;
    for (i, slot) in got.chunks(SLOT_BYTES as usize).enumerate() {
        let sid = slot_sid(i);
        match read_wdone(sid) {
            Ok(0) => {}
            Ok(v) => {
                say(format!(
                    "FINDING: read_wdone({sid})={v} after its wait — a residual count"
                ));
                verdict = 1;
            }
            Err(e) => {
                say(format!("FATAL: read_wdone({sid}) failed: {e}"));
                verdict = 1;
            }
        }
        match read_xseg(sid) {
            Ok(v) if v == words[i] => {}
            Ok(v) => {
                say(format!(
                    "FINDING: XSEG[{sid}] = 0x{v:08X} after the sends, written 0x{:08X} — an \
                     F-send re-pointed an entry",
                    words[i]
                ));
                verdict = 1;
            }
            Err(e) => {
                say(format!("FATAL: read_xseg({sid}) failed: {e}"));
                verdict = 1;
            }
        }
        match (0..slot.len()).find(|&j| slot[j] != pattern_byte(i, j)) {
            None => say(format!("slot {i} {:?} (sid {sid}): MATCH", SLOTS[i])),
            Some(j) => {
                say(format!(
                    "slot {i} {:?} (sid {sid}): MISMATCH at byte {j}: got {} want {}",
                    SLOTS[i],
                    slot[j],
                    pattern_byte(i, j)
                ));
                verdict = 1;
            }
        }
    }
    print_diag(say, 0, "post-op");
    say(format!(
        "VERDICT: {}",
        if verdict == 0 {
            "every slot landed byte-exact at its own address through its own sid"
        } else {
            "FAILED — see the findings above"
        }
    ));
    verdict
}

/// Rank 0: one F-send per slot, each on its slot's sid, each naming its slot's true destination
/// (the +128 compensation is the scheduler submit's one site).
fn sender_flow(
    say: &dyn Fn(String),
    stream: &Arc<RuntimeStream>,
    slot_addr: &dyn Fn(usize) -> Arc<CompositeAddress>,
) -> i32 {
    let deadline = Instant::now() + Duration::from_secs(60);
    let text = loop {
        match std::fs::read_to_string(HANDSHAKE_PATH) {
            Ok(t) if t.contains("segment") => break t,
            _ => {
                if Instant::now() > deadline {
                    say(format!(
                        "FATAL: no handshake at {HANDSHAKE_PATH} within 60 s"
                    ));
                    return 1;
                }
                std::thread::sleep(Duration::from_secs(1));
            }
        }
    };
    let (Some(segment), Some(offset_bytes)) = (
        json_field_u64(&text, "segment"),
        json_field_u64(&text, "offset_bytes"),
    ) else {
        say(format!("FATAL: handshake JSON unparseable: {text:?}"));
        return 1;
    };
    for i in 0..SLOTS.len() {
        let params = P2PRdmaSendParams {
            sync_key: format!("tp38-slot{i}"),
            dest_ranks: vec![1],
            sid_list: vec![u32::from(slot_sid(i))],
            device_address: slot_addr(i),
            op_name: format!("tp38-send-slot{i}"),
            dst_offset: flat_va(segment, offset_bytes + slot_offset(i)),
        };
        if let Err(e) = stream.launch_p2p_rdma_send(params, false, None) {
            say(format!("FATAL: launch_p2p_rdma_send(slot {i}) failed: {e}"));
            return 1;
        }
    }
    if let Err(e) = stream.synchronize() {
        say(format!("FATAL: synchronize after the sends failed: {e}"));
        return 1;
    }
    for i in 0..SLOTS.len() {
        print_diag(say, slot_sid(i), "post-op");
    }
    say(format!("{} sends complete", SLOTS.len()));
    0
}

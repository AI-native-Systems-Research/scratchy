//! tp37 — Phase 2 Step 1's transport re-proof cell (spec
//! `.claude/scratch/tp-phase2-step1-spec.md`, argmax-opus 10-31). A
//! DIAGNOSTIC, not product code: it reads rank/BDF config and the setpci
//! BAR2 expectation straight from the environment, which no product path
//! may grow. Gating idiom: `tests/on_card_cancelled_marker.rs`'s — no
//! default gate compiles or runs examples (`cargo build` / `cargo test
//! --lib` / dependent clippy never see it); it only builds under an
//! explicit `--example tp37_transport` on a Spyre host where the shim
//! really links (no `SCRATCHY_SKIP_SENDNN_CXX`).
//!
//! What it proves, and nothing more: the #1008 wire doors plus the PF
//! backend's two new P2P submits move bytes card-to-card on the rebased
//! tree under the six recorded transport laws —
//!
//! - L1 sid = SOURCE rank: the receiver waits the SENDER's sid (rank 0's
//!   own rank, 0);
//! - L2 wdone[sid] is a COUNT: the F-send's +1 is consumed by the wait
//!   (post-wait `read_wdone(0)` == 0), and the wait unblocks well under
//!   its ~5 s timeout — a timeout-bounded unblock is NOT a consumed count;
//! - L3 the RECEIVER's XSEG[sid] table is authoritative: rank 1
//!   host-programs XSEG[0] via `write_xseg` with the same composed word
//!   the send encoder produces; the ONE send is an F-send (`xsegv: false`
//!   at the hal door — the send-params type carries the law);
//! - L4 the engine writes one flit (128 B) BELOW the xseg-resolved
//!   destination, compensated at the ONE caller site inside
//!   `PfBackend::submit_p2p_rdma_send` (`dest_remote_base_bytes + 128`);
//!   the receiver's table word mirrors the same compensation
//!   (`rdma_xseg(dst_offset + 128)`) so a table-authoritative landing is
//!   byte-exact;
//! - L5 the BAT is symmetric: BOTH ranks install the peer's BAR2
//!   (`pci_addr_for_region(peer_bdf, 2)`), asserted against the setpci
//!   value the launcher passes in `TP37_EXPECT_BAR2_<peer_rank>` —
//!   live-verified, never the recorded value;
//! - L6 no T-send: the receiver's wait is pending BEFORE the send (the
//!   handshake file gates the sender's read, and the wait is submitted
//!   before the handshake is written).
//!
//! The BIND PROOF (the auditor's second guard): each rank prints
//! `resolved_pci()` — senlib's own resolved PCI address, read back from
//! the live PciDevId — and asserts it equals the BDF the process asked
//! for, in the log, BEFORE any BAT work. The SECOND-INIT law (the first
//! guard) lives in the shim door itself (`flex_shim_init_runtime_bdf`).
//!
//! Invocation (nickm3, one process per card, rank 1 FIRST — the vendor's
//! own env shape, aiu-assign-ranks.py:480-482):
//!
//! ```text
//! export RANK=<0|1> WORLD_SIZE=2
//! export AIU_WORLD_RANK_0=<bdf0> AIU_WORLD_RANK_1=<bdf1>
//! export TP37_EXPECT_BAR2_0=0x<hex> TP37_EXPECT_BAR2_1=0x<hex>   # from LIVE setpci
//! export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
//! cargo build --release -p flex-rs --example tp37_transport
//! ./target/release/examples/tp37_transport
//! ```
//!
//! Cross-rank coordination is a handshake file under `/work` (NEVER /tmp):
//! rank 1 writes `/work/tp37-r1.json` after its H2D completes and its wait
//! is pending; rank 0 polls it (1 s interval, 60 s timeout). The launcher
//! deletes a stale handshake file before each run.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use flex_rs::address::{ByteSize, Chunk, CompositeAddress, LogicalAddress};
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

/// The per-rank device buffer. 64 KiB, 128-flit aligned by the allocator.
const BUF_BYTES: u64 = 64 * 1024;
/// The transferred payload: rank 0's first 4096 buffer bytes → rank 1's
/// first 4096 buffer bytes (one 4 MiB RDMA window chunk — ONE F-send CB).
const XFER_BYTES: u64 = 4096;
/// Rank 1's handshake file (under /work, NEVER /tmp).
const HANDSHAKE_PATH: &str = "/work/tp37-r1.json";
/// The wait's own ~5 s timeout law; a genuine count-consumption unblock
/// lands well under it (the vendor's own test bar: 0.5-1.5 s).
const WAIT_GENUINE_BOUND: Duration = Duration::from_millis(4500);

/// The payload pattern: byte i = (i*7 + 3) & 0xFF.
fn pattern_byte(i: usize) -> u8 {
    ((i * 7 + 3) & 0xFF) as u8
}

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    // ── Rank/size, refused loudly unless this is a 2-rank world. ──
    let rank = FlexConfig::rdma_world_rank().0;
    let size = FlexConfig::rdma_world_size().0;
    if size != 2 || !(rank == 0 || rank == 1) {
        eprintln!(
            "[tp37 rank {rank}] REFUSED: RANK={rank} WORLD_SIZE={size} — this cell needs \
             RANK ∈ {{0,1}} and WORLD_SIZE=2 (the torchrun-style env the vendor's \
             aiu-assign-ranks.py sets)"
        );
        return 2;
    }
    let say = |msg: String| println!("[tp37 rank {rank}] {msg}");

    // ── Bind THIS process's runtime to ITS card, by BDF, through the NEW
    // rank-scoped door (A2: logical_device_id selects nothing). ──
    let bdf = match FlexConfig::rdma_get_pcie_address(Some(rank)) {
        Ok(b) => b.0,
        Err(e) => {
            say(format!(
                "FATAL: rdma_get_pcie_address(rank {rank}) failed: {e} — set AIU_WORLD_RANK_{rank}=\"<bdf>\""
            ));
            return 1;
        }
    };
    if let Err(e) = init_runtime_bdf(&bdf) {
        say(format!("FATAL: init_runtime_bdf({bdf}) failed: {e}"));
        return 1;
    }
    say(format!("rank={rank} bdf={bdf}"));

    // ── THE BIND PROOF (guard 2): senlib's own resolved PCI address must
    // equal the BDF this process asked for — printed and asserted, BEFORE
    // any BAT work. "We passed the BDF" is not proof the bind took. ──
    let resolved = match resolved_pci() {
        Ok(r) => r,
        Err(e) => {
            say(format!(
                "FATAL: resolved_pci() failed: {e} — nothing is initialized"
            ));
            return 1;
        }
    };
    say(format!(
        "BIND-PROOF: asked for {bdf}, senlib resolved {resolved}"
    ));
    if resolved != bdf {
        say(format!(
            "FATAL (BIND-PROOF): senlib resolved {resolved} ≠ the asked BDF {bdf} — the \
             rank scoping did NOT take; refusing before any BAT work"
        ));
        return 1;
    }

    // ── L5: the SYMMETRIC BAT — both ranks install the peer's BAR2, with
    // the value from senlib's own `pci_addr_for_region(peer_bdf, 2)`,
    // ASSERTED against the live setpci value the launcher passes in
    // TP37_EXPECT_BAR2_<peer_rank>. ──
    let peer_rank = 1 - rank;
    let peer_bdf = match FlexConfig::rdma_get_pcie_address(Some(peer_rank)) {
        Ok(b) => b.0,
        Err(e) => {
            say(format!(
                "FATAL: rdma_get_pcie_address(peer {peer_rank}) failed: {e} — set AIU_WORLD_RANK_{peer_rank}=\"<bdf>\""
            ));
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
    let expect_var = format!("TP37_EXPECT_BAR2_{peer_rank}");
    let expected_bar2: u64 = match std::env::var(&expect_var) {
        Ok(s) => match u64::from_str_radix(s.trim_start_matches("0x"), 16) {
            Ok(v) => v,
            Err(e) => {
                say(format!("FATAL: {expect_var}={s:?} is not a hex u64: {e}"));
                return 1;
            }
        },
        Err(_) => {
            say(format!(
                "FATAL: {expect_var} is unset — the cell asserts senlib's BAR2 resolution \
                 against the LIVE setpci value; the launcher must pass it"
            ));
            return 1;
        }
    };
    say(format!(
        "L5: pci_addr_for_region({peer_bdf}, 2)=0x{peer_bar2:X}; setpci expectation=0x{expected_bar2:X}"
    ));
    if peer_bar2 != expected_bar2 {
        say(format!(
            "FINDING (STOP): pci_addr_for_region({peer_bdf}, 2)=0x{peer_bar2:X} DIFFERS from the \
             setpci BAR2 base 0x{expected_bar2:X} — senlib resolves the peer's BAR2 differently \
             than config space. Reported, not papered over."
        ));
        return 1;
    }
    if let Err(e) = set_base_addr_table(peer_rank as u8, peer_bar2) {
        say(format!(
            "FATAL: set_base_addr_table(tid={peer_rank}, 0x{peer_bar2:X}) failed: {e}"
        ));
        return 1;
    }
    say(format!(
        "L5: set_base_addr_table(tid={peer_rank}, 0x{peer_bar2:X}) rc=Ok(()) — symmetric install, \
         value equals the setpci BAR2 base"
    ));

    // ── The real init path's own bring-up (the exact sequence
    // tests/on_card_cancelled_marker.rs runs). ──
    let device_memory_size = unsafe { flex_senlib_device_memory_size() };
    if device_memory_size == 0 {
        say(
            "FATAL: the card reports 0 device memory — bring-up did not yield a usable device"
                .to_string(),
        );
        return 1;
    }
    let topo = populate_domain_topology(device_memory_size);
    let allocator = Arc::new(Mutex::new(
        match FlexAllocator::new(
            FlexAllocatorInit::default(),
            topo,
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
    let hw = Arc::new(SenlibHardwareSubmit::new());
    let capacity = Arc::new(SenlibQueueCapacity);
    let iommu_mapper = PfIommuMapper::new();
    let scheduler = match create_scheduler(
        &DeviceKind::Pf,
        SchedulerBackendConfig::Real(hw.clone(), capacity, dyn_resolver, iommu_mapper),
    ) {
        Ok(s) => s,
        Err(e) => {
            say(format!("FATAL: create_scheduler(Real) failed: {e}"));
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
    let stream_id = match context.create_stream(
        RuntimeStreamPriority::Normal,
        RuntimeStreamMode::StrictOrdering,
        true,
    ) {
        Ok(id) => id,
        Err(e) => {
            say(format!("FATAL: create_stream failed: {e}"));
            return 1;
        }
    };
    let stream = match context.get_stream(stream_id) {
        Ok(Some(s)) => s,
        Ok(None) => {
            say("FATAL: get_stream returned None for a just-created stream".to_string());
            return 1;
        }
        Err(e) => {
            say(format!("FATAL: get_stream failed: {e}"));
            return 1;
        }
    };

    // The 64 KiB device buffer + the 4096-byte transfer window at its start.
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
    let (segment, _base_paddr, region_size) = match resolver.resolve(&chunk0) {
        Some(r) => r,
        None => {
            say("FATAL: the buffer's region did not resolve to a segment".to_string());
            return 1;
        }
    };
    let offset_bytes = chunk0.addr.offset.0;
    say(format!(
        "buffer segment={segment} offset_bytes={offset_bytes} region_size={}",
        region_size.as_u64()
    ));
    let window = Arc::new(CompositeAddress::from_chunk(Chunk::new(
        LogicalAddress::new(chunk0.addr.region_id, chunk0.addr.offset),
        ByteSize(XFER_BYTES),
        chunk0.domain_id,
    )));

    // ── H2D: rank 0 writes the pattern, rank 1 writes zeros. ──
    let mut host = vec![0u8; XFER_BYTES as usize];
    if rank == 0 {
        for (i, b) in host.iter_mut().enumerate() {
            *b = pattern_byte(i);
        }
    }
    let h2d = DmaParams::with_explicit_size(
        HostVirtualAddress(host.as_ptr() as usize),
        ByteSize(XFER_BYTES),
        DmaDirection::HostToDevice,
        Some(window.clone()),
        DmaShaping::default(),
    )
    .expect("device_address is Some, constructed from an owned window");
    if let Err(e) = stream.launch_h2d(h2d, None) {
        say(format!("FATAL: launch_h2d failed: {e}"));
        return 1;
    }
    if let Err(e) = stream.synchronize() {
        say(format!("FATAL: synchronize after H2D failed: {e}"));
        return 1;
    }
    say("H2D complete".to_string());

    // The sid this cell speaks: the SENDER's rank (L1 — sid names the
    // source). Rank 0 sends, so sid = 0 everywhere.
    let sid: u32 = 0;

    // Pre-op facts: wdone + rdma_status, printed either way.
    print_diag(&say, sid as u8, "pre-op");

    if rank == 1 {
        receiver_flow(&say, &stream, &window, segment, offset_bytes, sid)
    } else {
        sender_flow(&say, &stream, &window, sid)
    }
}

/// `read_wdone(sid)` + `check_and_clear_rdma_status()`, labeled `when`.
fn print_diag(say: &dyn Fn(String), sid: u8, when: &str) {
    match read_wdone(sid) {
        Ok(v) => say(format!("{when} read_wdone({sid})={v}")),
        Err(e) => say(format!("{when} read_wdone({sid}) FAILED: {e}")),
    }
    match check_and_clear_rdma_status() {
        Ok(0) => say(format!("{when} rdma_status=0")),
        Ok(v) => say(format!(
            "⚠️⚠️⚠️ {when} rdma_status={v} NONZERO — RDMA ERROR STATE ON THIS CARD ⚠️⚠️⚠️"
        )),
        Err(e) => say(format!("{when} rdma_status read FAILED: {e}")),
    }
}

/// One field out of the handshake JSON (`{"segment": N, "offset_bytes": M}`)
/// — flex-rs has no JSON dependency and the cell must not add one.
fn json_field_u64(text: &str, key: &str) -> Option<u64> {
    let key_pat = format!("\"{key}\"");
    let i = text.find(&key_pat)?;
    let rest = text[i + key_pat.len()..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// Rank 1 (the receiver): host-program XSEG[sid] (L3), submit the WAIT
/// (L6: pending before the send can exist — the handshake is written after
/// the wait is submitted, and the sender gates on it), publish the buffer
/// location, then D2H + byte-compare once the wait has consumed the
/// F-send's count.
fn receiver_flow(
    say: &dyn Fn(String),
    stream: &Arc<RuntimeStream>,
    window: &Arc<CompositeAddress>,
    segment: u8,
    offset_bytes: u64,
    sid: u32,
) -> i32 {
    assert!(
        offset_bytes.is_multiple_of(128),
        "rank 1's buffer offset {offset_bytes} is not 128-byte aligned — the flat-dmva \
         derivation only holds for flit-aligned offsets"
    );
    // The flat-dmva derivation (Stage B spec, step 5): the send's xseg field
    // wants bytes_to_flits(remote_dmva) where remote_dmva is this rank's
    // FLAT byte VA — `((segment << 27) | (offset_bytes >> 7)) << 7`.
    let dst_offset = ((segment as u64) << 27 | (offset_bytes >> 7)) << 7;
    // L4's receiver-side mirror: the table word is the SAME compensated
    // compose the sender's CB carries (`rdma_xseg(dst_offset + 128)`), so a
    // table-authoritative landing (L3) is byte-exact despite the engine's
    // one-flit-below write.
    let xseg = match rdma_xseg(dst_offset + Flits::BYTES) {
        Ok(v) => v,
        Err(e) => {
            say(format!(
                "FATAL: rdma_xseg(dst_offset 0x{dst_offset:X} + 128) refused: {e}"
            ));
            return 1;
        }
    };
    say(format!(
        "L3/L4: dst_offset={dst_offset} (0x{dst_offset:X}) → xseg=0x{xseg:08X} (compensated, +{} = \
         the one-flit law)",
        Flits::BYTES
    ));
    if let Err(e) = write_xseg(sid as u8, xseg) {
        say(format!(
            "FATAL: write_xseg({sid}, 0x{xseg:08X}) FAILED: {e} — verbatim, stopping"
        ));
        return 1;
    }
    say(format!(
        "L3: write_xseg({sid}, 0x{xseg:08X}) rc=Ok(()) — the receiver's table is authoritative"
    ));
    match read_xseg(sid as u8) {
        Ok(v) if v == xseg => say(format!(
            "L3: read_xseg({sid})=0x{v:08X} — STUCK (matches the written value)"
        )),
        Ok(v) => {
            say(format!(
                "FATAL: read_xseg({sid})=0x{v:08X} ≠ the written 0x{xseg:08X} — READBACK MISMATCH"
            ));
            return 1;
        }
        Err(e) => {
            say(format!("FATAL: read_xseg({sid}) FAILED: {e}"));
            return 1;
        }
    }

    // ── L6: the WAIT, submitted BEFORE the handshake that gates the send —
    // the wait is pending ahead of the F-send. ──
    let t_submit = Instant::now();
    let done = Arc::new(Mutex::new(None::<Duration>));
    let done_cb = done.clone();
    let params = P2PRdmaWaitParams {
        sid_list: vec![sid],
    };
    if let Err(e) = stream.launch_p2p_rdma_wait(
        params,
        false,
        Some(Box::new(move || {
            *done_cb.lock().unwrap() = Some(t_submit.elapsed());
        })),
    ) {
        say(format!("FATAL: launch_p2p_rdma_wait failed: {e}"));
        return 1;
    }
    say("L6: wait submitted (pending before the send)".to_string());

    // ── Publish this buffer's location (AFTER the wait — the sender cannot
    // send before this file exists, so the ordering law is structural). ──
    if let Err(e) = std::fs::write(
        HANDSHAKE_PATH,
        format!("{{\"segment\": {segment}, \"offset_bytes\": {offset_bytes}}}"),
    ) {
        say(format!("FATAL: writing {HANDSHAKE_PATH} failed: {e}"));
        return 1;
    }
    say(format!(
        "handshake written: segment={segment} offset_bytes={offset_bytes}"
    ));

    // ── L2: the wait's unblock must be a consumed COUNT — well under the
    // ~5 s timeout — and read_wdone(sid) must be 0 after (the count was
    // consumed, not left resident). ──
    let deadline = Instant::now() + WAIT_GENUINE_BOUND;
    while Instant::now() < deadline && done.lock().unwrap().is_none() {
        std::thread::sleep(Duration::from_millis(10));
    }
    let elapsed = *done.lock().unwrap();
    let Some(elapsed) = elapsed else {
        say(
            "FATAL (L2 — STOP FINDING): the wait did NOT unblock within its ~5 s timeout — the \
             F-send's wdone count never reached a pending wait on the new doors"
                .to_string(),
        );
        return 1;
    };
    say(format!(
        "L2: wait unblocked {elapsed:?} after submit — {}",
        if elapsed < WAIT_GENUINE_BOUND {
            "a consumed COUNT, well under the ~5 s timeout"
        } else {
            "≈ the timeout itself — NOT a genuine consumption"
        }
    ));

    // The D2H readback, FIFO-behind the wait on this stream.
    let mut got = vec![0u8; XFER_BYTES as usize];
    let d2h = DmaParams::with_explicit_size(
        HostVirtualAddress(got.as_mut_ptr() as usize),
        ByteSize(XFER_BYTES),
        DmaDirection::DeviceToHost,
        Some(window.clone()),
        DmaShaping::default(),
    )
    .expect("device_address is Some, constructed from an owned window");
    if let Err(e) = stream.launch_d2h(d2h, None) {
        say(format!("FATAL: launch_d2h failed: {e}"));
        return 1;
    }
    if let Err(e) = stream.synchronize() {
        say(format!(
            "FATAL: synchronize after wait+D2H failed: {e} — the wait never completed or the \
             readback failed"
        ));
        return 1;
    }

    match read_wdone(sid as u8) {
        Ok(0) => say(format!(
            "L2: read_wdone({sid})=0 after the wait — the F-send's +1 was CONSUMED (no residual count)"
        )),
        Ok(v) => {
            say(format!(
                "FINDING (L2): read_wdone({sid})={v} after the wait — a RESIDUAL count the wait \
                 did not consume; reported, not papered over"
            ));
            return 1;
        }
        Err(e) => {
            say(format!(
                "FATAL: read_wdone({sid}) FAILED after the wait: {e}"
            ));
            return 1;
        }
    }
    print_diag(say, sid as u8, "post-op");

    // ── The verdict: byte-compare against the pattern. ──
    match (0..XFER_BYTES as usize).find(|&i| got[i] != pattern_byte(i)) {
        None => {
            say("VERDICT: MATCH — 4096 bytes byte-exact D2D through the new doors".to_string());
            0
        }
        Some(i) => {
            say(format!(
                "VERDICT: MISMATCH first_diff={i} got={} want={}",
                got[i],
                pattern_byte(i)
            ));
            1
        }
    }
}

/// Rank 0 (the sender): poll the handshake, then ONE F-send through the new
/// submit seam — the +128 compensation is INSIDE
/// `PfBackend::submit_p2p_rdma_send`, at its one site; this side passes the
/// raw flat destination.
fn sender_flow(
    say: &dyn Fn(String),
    stream: &Arc<RuntimeStream>,
    window: &Arc<CompositeAddress>,
    sid: u32,
) -> i32 {
    // ── Poll the handshake (1 s interval, 60 s timeout). ──
    let deadline = Instant::now() + Duration::from_secs(60);
    let text = loop {
        match std::fs::read_to_string(HANDSHAKE_PATH) {
            Ok(t) if t.contains("segment") => break t,
            _ => {
                if Instant::now() > deadline {
                    say(format!(
                        "FATAL: no handshake at {HANDSHAKE_PATH} within 60 s — is rank 1 running?"
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
    assert!(
        offset_bytes.is_multiple_of(128),
        "rank 1's buffer offset {offset_bytes} is not 128-byte aligned — the flat-dmva \
         derivation only holds for flit-aligned offsets"
    );
    let dst_offset = ((segment << 27) | (offset_bytes >> 7)) << 7;
    say(format!(
        "handshake read: segment={segment} offset_bytes={offset_bytes} → dst_offset={dst_offset} \
         (0x{dst_offset:X}) — the +128 compensation is the scheduler submit's one site"
    ));

    // ── The ONE F-send (L1: sid = this rank, the SOURCE). ──
    let t_submit = Instant::now();
    let done = Arc::new(Mutex::new(None::<Duration>));
    let done_cb = done.clone();
    let params = P2PRdmaSendParams {
        sync_key: "tp37".to_string(),
        dest_ranks: vec![1],
        sid_list: vec![sid],
        device_address: window.clone(),
        op_name: "tp37-send".to_string(),
        dst_offset,
    };
    if let Err(e) = stream.launch_p2p_rdma_send(
        params,
        false,
        Some(Box::new(move || {
            *done_cb.lock().unwrap() = Some(t_submit.elapsed());
        })),
    ) {
        say(format!("FATAL: launch_p2p_rdma_send failed: {e}"));
        return 1;
    }
    if let Err(e) = stream.synchronize() {
        say(format!(
            "FATAL: synchronize after send failed: {e} — the send never completed"
        ));
        return 1;
    }
    match *done.lock().unwrap() {
        Some(d) => say(format!("send callback {d:?} after submit")),
        None => {
            say("FATAL: the send's callback never fired (stream synchronized anyway)".to_string());
            return 1;
        }
    }

    print_diag(say, sid as u8, "post-op");
    0
}

// SPDX-License-Identifier: Apache-2.0

//! #546's concurrency leg: two tracked `OP_ORDERING` streams, one cross-stream RAW
//! hazard, one shared card — correct ordering with NO host-side lock, carried by the
//! hazard tracker the rest of #546 ported.
//!
//! ```bash
//! cargo run --release --example hazard_concurrency --features spyre-hw
//! ```
//!
//! Two rounds over the same 512 MiB device buffer:
//!
//! - **Round 1 (tracked)**: stream A H2D-writes a new pattern; the moment A's launch
//!   returns — its op registered with the tracker and its DMA in flight — stream B, on
//!   its own thread, D2H-reads the SAME buffer. The tracker's RAW edge parks B inside
//!   its own launch until A's completion signals, so B's readback can only ever see
//!   A's pattern, and B's launch measurably waits out A's DMA. THAT is the ordering
//!   demo: concurrent submission, tracker-carried ordering, data integrity asserted.
//! - **Round 2 (control, untracked)**: identical shapes with `track_hazards = false` —
//!   B's launch returns immediately (no edge, no park; `runtime_stream.hpp:85-86`:
//!   unregistered streams are user-owned and never edged). Ordering is then the
//!   caller's obligation; the round only REPORTS what the racing read saw (old, new,
//!   or torn) — it cannot assert it, by design.
//!
//! 512 MiB is not arbitrary: the card moves a 64 MiB transfer in ~2 ms (~35 GB/s), so
//! a small buffer leaves A's write in flight for less than the run-to-run noise of the
//! ~11 ms pin+submit that dominates every launch. 512 MiB keeps A's write outstanding
//! for ~15 ms after its launch returns — B's park then stands out of the noise as the
//! tracked-vs-control launch delta the leg prints at the end.
//!
//! Host buffers are byte slices end to end (the `Stream::h2d`/`d2h` slice-length law):
//! patterns are built as `u64` words flattened to bytes, and the readback is compared
//! word-wise through `as_chunks::<8>()` — no pointer reinterpretation anywhere.

use std::time::{Duration, Instant};

use anyhow::anyhow;
use scratchy_target_spyre::runtime::{DevAddr, HwError, Stream};

/// 512 MiB — see the module doc: big enough that A's DMA stays in flight for ~15 ms
/// after its launch returns, so B's park is unmistakable against launch noise.
const SIZE_BYTES: usize = 512 * 1024 * 1024;

fn pattern_bytes(xor: u64) -> Vec<u8> {
    (0..SIZE_BYTES / 8)
        .map(|i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ xor)
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

const OLD_XOR: u64 = 0xDEAD_BEEF_0000_0000;
const NEW_XOR: u64 = 0xC0FF_EE00_1111_1111;

fn words(bytes: &[u8]) -> impl Iterator<Item = u64> + '_ {
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| u64::from_le_bytes(*c))
}

/// The pattern word at index `i`, matching `pattern_bytes`' construction.
fn pattern_word(i: usize, xor: u64) -> u64 {
    (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ xor
}

/// One round: stage `old`, launch A's write of `new`, then race B's read on its own
/// thread. Returns (B's launch duration — the park, when tracked, B's readback).
fn run_round(track_hazards: bool, addr: &DevAddr) -> anyhow::Result<(Duration, Duration, Vec<u8>)> {
    let a = Stream::create_with(
        flex_rs::stream::RuntimeStreamPriority::Normal,
        flex_rs::stream::RuntimeStreamMode::OpOrdering,
        track_hazards,
    )
    .map_err(|e| anyhow!("stream A (track={track_hazards}): {e}"))?;
    let b = Stream::create_with(
        flex_rs::stream::RuntimeStreamPriority::Normal,
        flex_rs::stream::RuntimeStreamMode::OpOrdering,
        track_hazards,
    )
    .map_err(|e| anyhow!("stream B (track={track_hazards}): {e}"))?;

    // Start the buffer on a KNOWN old pattern so the readback can only be explained
    // by A's write landing (or not), never by stale device contents.
    let old = pattern_bytes(OLD_XOR);
    a.h2d(&old, addr).map_err(|e| anyhow!("setup h2d: {e}"))?;
    a.synchronize().map_err(|e| anyhow!("setup sync: {e}"))?;
    drop(old);

    let new = pattern_bytes(NEW_XOR);
    let mut readback = vec![0u8; SIZE_BYTES];

    // A's launch returns with the op registered and its DMA in flight; B's thread
    // then races in. Tracked: B parks on A's outstanding write. Untracked: it
    // doesn't, and the read races the write.
    let t_write = Instant::now();
    let t_launch_a = {
        a.h2d(&new, addr).map_err(|e| anyhow!("A h2d: {e}"))?;
        t_write.elapsed()
    };

    let reader = {
        let addr = addr.clone();
        let mut readback = std::mem::take(&mut readback);
        std::thread::spawn(move || -> Result<(Duration, Vec<u8>), HwError> {
            let t_b = Instant::now();
            b.d2h(&mut readback, &addr)?;
            let launch = t_b.elapsed();
            b.synchronize()?;
            Ok((launch, readback))
        })
    };

    let (b_launch, readback) = reader.join().map_err(|_| anyhow!("reader panicked"))??;
    a.synchronize().map_err(|e| anyhow!("A sync: {e}"))?;
    let t_a = t_write.elapsed();
    println!(
        "[hazard-concurrency] track={track_hazards}: A launch {t_launch_a:?}, \
         A end-to-end {t_a:?}, B launch {b_launch:?}"
    );
    Ok((b_launch, t_a, readback))
}

fn main() -> anyhow::Result<()> {
    // ⭐ WITHOUT A SUBSCRIBER, flex's COMPLETION THREAD has nowhere to write its CB-error
    // lines — the `hmi_syndrome` / `qgi_syndrome` / `job_count` that name WHY a launch was
    // rejected (same lesson as `bundle_run_dev`).
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_writer(std::io::stderr)
        .init();

    let addr = DevAddr::alloc(
        flex_rs::address::ByteSize(SIZE_BYTES as u64),
        flex_rs::memory_region::MemoryType::Tensor,
    )
    .map_err(|e| anyhow!("alloc: {e}"))?;

    // ── Round 1: tracked OP_ORDERING streams — the RAW edge must order them. ──
    let (b_launch_tracked, _t_a, readback) = run_round(true, &addr)?;
    let mismatches: usize = words(&readback)
        .enumerate()
        .map(|(i, got)| usize::from(got != pattern_word(i, NEW_XOR)))
        .sum();
    anyhow::ensure!(
        mismatches == 0,
        "RAW edge FAILED: B's readback has {mismatches} words that are not A's pattern \
         — cross-stream ordering was not carried"
    );
    println!(
        "[hazard-concurrency] round 1 (tracked): readback == A's pattern \
         (every one of {} words)",
        SIZE_BYTES / 8
    );

    // ── Round 2 (control): untracked — same shapes, no edge, no park. ──
    let (b_launch_untracked, _t_a, readback) = run_round(false, &addr)?;
    // No assertion on WHAT the racing read saw — untracked means unordered, and a torn
    // read is a possible (not promised) outcome. Report it honestly.
    let matched_new = words(&readback)
        .enumerate()
        .all(|(i, w)| w == pattern_word(i, NEW_XOR));
    let matched_old = words(&readback)
        .enumerate()
        .all(|(i, w)| w == pattern_word(i, OLD_XOR));
    let saw = if matched_new {
        "A's new pattern (write had landed)"
    } else if matched_old {
        "the old pattern (write had not landed)"
    } else {
        "a TORN mix of both (the race the tracker exists to close)"
    };
    println!(
        "[hazard-concurrency] round 2 (control, untracked): B launch {b_launch_untracked:?} \
         (no park — submission only); the racing read saw {saw}"
    );

    // The number the leg exists to print: the tracked B's launch had to park out the
    // slice of A's write that was still in flight, the control B's launch did not —
    // the difference IS the park. `saturating_sub` because at 512 MiB the park is
    // ~8x the ~3 ms launch noise (the module doc's sizing argument), but a delta
    // printed as zero would still be an honest report, never a panic.
    let park = b_launch_tracked.saturating_sub(b_launch_untracked);
    println!(
        "[hazard-concurrency] park delta: tracked B launch − control B launch = {park:?} \
         (the slice of A's in-flight write the edge closed)"
    );

    println!("[hazard-concurrency] LEG PASSED");
    Ok(())
}

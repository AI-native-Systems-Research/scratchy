// SPDX-License-Identifier: Apache-2.0

//! #1016's card leg: `Stream::fill_zero` on real hardware. The vendor's fill op
//! (`RuntimeOperationFill` / `SetHAFill`) had no caller anywhere in the tree before
//! #1016 — `zero_seg`'s card fill is its first use — so this is the proof the fill
//! control block actually writes zeros on a card, which no cardless test can show.
//! Cardless builds skip this example entirely via `required-features =
//! ["spyre-hw"]`.
//!
//! ```bash
//! cargo run --release --example fill_zero_probe --features spyre-hw
//! ```
//!
//! What it asserts, in order:
//!
//! 1. **Not-X first.** A non-zero pattern is H2D'd into a fresh region and read
//!    back — the readback must be non-zero, so a vacuous pass (a fill that
//!    no-ops against memory that happened to read zero) cannot clear the probe.
//! 2. **The fill zeroes every byte.** `fill_zero` over the region's
//!    `windows(WINDOW)` walk — the same single-chunk shapes `zero_seg` submits —
//!    then a D2H into a buffer pre-poisoned with `0xFF`: every byte must read 0.
//!    The region is one window plus a short tail (64 MiB + 4 KiB), so the walk
//!    and the fill both cross a window boundary.
//! 3. **The fill's rate is measured, not asserted** — printed for the record,
//!    since the load-time budget depends on it (the 16384 MB KV pool).

use std::time::Instant;

use anyhow::{Context, Result, anyhow};

use flex_rs::address::ByteSize;
use flex_rs::memory_region::MemoryType;
use scratchy_target_spyre::runtime::{DevAddr, Stream};

/// The window step of `zero_seg`'s walk, mirrored here so the probe exercises
/// the same single-chunk shapes the loader submits.
const WINDOW: u64 = 64 << 20;
/// A short tail past one window, so the walk and the fill cross a boundary.
const TAIL: u64 = 4 << 12;

/// Deterministic pattern (xorshift64*) — no `rand` dep, and a fixed seed makes a
/// mismatch reproducible byte for byte.
fn pattern_bytes(len: usize) -> Vec<u8> {
    let mut state = 0x1016u64;
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let word = state.wrapping_mul(0x2545F4914F6CDD1D);
        out.extend_from_slice(&word.to_le_bytes());
    }
    out.truncate(len);
    out
}

fn main() -> Result<()> {
    let stream = Stream::create().context("stream create (brings the runtime up)")?;
    let total = WINDOW + TAIL;
    let dev = DevAddr::alloc(ByteSize(total), MemoryType::Tensor)
        .context("device buffer alloc")?;

    // --- 1. not-X: the region really holds non-zero bytes before the fill ---
    let poison = pattern_bytes(total as usize);
    stream
        .h2d(&poison, &dev)
        .context("H2D of the non-zero pattern")?;
    let mut before = vec![0u8; total as usize];
    stream
        .d2h(&mut before, &dev)
        .context("D2H of the poisoned region")?;
    stream.synchronize().context("synchronize")?;
    let nonzero = before.iter().filter(|&&b| b != 0).count();
    anyhow::ensure!(
        nonzero > 0,
        "the poisoned region read back all zero BEFORE the fill — the probe's \
         not-X arm failed, so its zero arm would prove nothing"
    );
    println!("poison: {nonzero} of {total} B read back non-zero before the fill");

    // --- 2. the fill zeroes every byte, across the window boundary ---
    let mut windows = 0usize;
    let t = Instant::now();
    for w in dev.windows(std::num::NonZeroU64::new(WINDOW).expect("a nonzero step")) {
        stream.fill_zero(&w).context("fill_zero over a window")?;
        windows += 1;
    }
    stream.synchronize().context("synchronize")?;
    let fill_secs = t.elapsed().as_secs_f64();
    println!(
        "fill: {windows} window(s), {total} B in {:.1} ms ({:.2} GB/s)",
        fill_secs * 1e3,
        total as f64 / fill_secs / 1e9
    );
    anyhow::ensure!(windows >= 2, "the walk must cross a window boundary");
    let mut after = vec![0xFFu8; total as usize];
    stream
        .d2h(&mut after, &dev)
        .context("D2H of the filled region")?;
    stream.synchronize().context("synchronize")?;
    if let Some(off) = after.iter().position(|&b| b != 0) {
        return Err(anyhow!(
            "the fill left a non-zero byte at offset {off} (0x{off:x}) of {total}"
        ));
    }
    println!("fill_zero_probe: PASS");
    Ok(())
}

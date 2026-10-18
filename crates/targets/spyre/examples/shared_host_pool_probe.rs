// SPDX-License-Identifier: Apache-2.0

//! #606's card leg: the shared host pool on real hardware — the invariants
//! the cardless suite cannot reach (a real whole-region pin, a real stream,
//! a real device object). Cardless builds skip this example entirely via
//! `required-features = ["spyre-hw"]`.
//!
//! ```bash
//! cargo run --release --example shared_host_pool_probe --features spyre-hw
//! ```
//!
//! What it asserts, in order:
//!
//! 1. **Construction pins.** The pool builds and its region pinned to a
//!    NON-ZERO IOVA — the positive case of the constructor's null-IOVA
//!    refusal (IOVA 0 is the DMA backends' "not pre-mapped" sentinel).
//! 2. **Index addressing + stability.** `resolve_slot(i)`'s hmva and IOVA
//!    both advance by the aligned stride, and re-resolving a slot is
//!    byte-identical — the stable-IOVA property #603 PR3b's pre-built DMA
//!    CBs depend on (a slot's IOVA cannot move for the life of the pool).
//! 3. **The data path.** A pattern written into slot 0 through `slot_mut`
//!    rides `enqueue_copy` H2D into a device buffer — pre-mapped: the
//!    pool's IOVA rides `DmaParams.iova`, no staging buffer, no per-step
//!    pin — then D2H back into slot 1, and slot 1's bytes must equal the
//!    pattern. The transfer length is the DEVICE object's `total_size`
//!    (the vendor's law), so the buffer is sized to the slot exactly.
//! 4. **The validation set refuses before any DMA.** An over-capacity
//!    device object and a two-chunk device address are both refused with
//!    nothing submitted (the neighboring-slot overrun and
//!    `requireSingleChunk` ports).
//! 5. **Teardown.** `synchronize()` before drop (the transfer-lifetime
//!    law — the pool is borrowed by in-flight transfers), then drop, which
//!    unmaps the region before the backing buffer frees.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};

use flex_rs::address::{ByteSize, Chunk, CompositeAddress, LogicalAddress};
use flex_rs::iommu::PfIommuMapper;
use flex_rs::memory_region::MemoryType;
use flex_rs::runtime::RuntimeContext;
use flex_rs::shared_host_pool::{SharedHostPool, SharedHostPoolError};
use flex_rs::stream::{RuntimeStreamMode, RuntimeStreamPriority};
use scratchy_target_spyre::runtime::{DevAddr, init_runtime};

/// One page-multiple megabyte per slot: the aligned stride equals the
/// request, so the capacity arithmetic in the assertions below is exact.
const SLOT_BYTES: usize = 1024 * 1024;
const NUM_SLOTS: usize = 2;

/// Deterministic pattern (xorshift64*) — no `rand` dep, and a fixed seed
/// makes a mismatch reproducible byte for byte.
fn pattern_bytes(len: usize) -> Vec<u8> {
    let mut state = 0x606u64;
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

/// The device side of an `enqueue_copy` needs the `Arc<CompositeAddress>`
/// itself; `DevAddr` keeps its descriptor private, so rebuild an equal one
/// from the (public) chunk identity — same region, offset, size, domain.
fn arc_from_dev(dev: &DevAddr) -> Result<Arc<CompositeAddress>> {
    let c = dev
        .chunk0()
        .ok_or_else(|| anyhow!("device address is not single-chunk"))?;
    Ok(Arc::new(CompositeAddress::from_chunk(Chunk::new(
        LogicalAddress::new(c.region, c.offset),
        c.size,
        c.domain,
    ))))
}

fn main() -> Result<()> {
    // Real bring-up: shim init, card-derived topology/allocator, scheduler,
    // `RuntimeContext::create` (which installs the global the probe reads back).
    init_runtime().context("runtime bring-up")?;
    let rt = RuntimeContext::get_instance().context("runtime context")?;
    let stream_id = rt
        .create_stream(
            RuntimeStreamPriority::Normal,
            RuntimeStreamMode::StrictOrdering,
            false,
        )
        .context("create_stream")?;
    let stream = rt
        .get_stream(stream_id)?
        .ok_or_else(|| anyhow!("stream vanished right after creation"))?;

    // A FRESH mapper for the pool — the pool owns its pin independent of the
    // runtime's staging mapper, which is what lets a pool outlive
    // `RuntimeContext::reset()` (the mapper-retention law, carried by the
    // `Arc<PfIommuMapper>` inside the mapping).
    let mapper = PfIommuMapper::new();
    let mut pool =
        SharedHostPool::new(NUM_SLOTS, SLOT_BYTES, &mapper).context("pool construction")?;

    // --- 1. construction pins, to a non-null IOVA ---
    let r0 = pool.resolve_slot(0)?;
    anyhow::ensure!(
        r0.iova.0 != 0,
        "slot 0 pinned to IOVA 0 — null IOVA means not pre-mapped"
    );

    // --- 2. index addressing + stability ---
    let r1 = pool.resolve_slot(1)?;
    anyhow::ensure!(
        pool.slot_bytes() == SLOT_BYTES,
        "1 MiB is page-multiple; stride must equal the request, got {}",
        pool.slot_bytes()
    );
    anyhow::ensure!(
        r1.host_address.0 - r0.host_address.0 == pool.slot_bytes(),
        "hmva slots not stride apart"
    );
    anyhow::ensure!(
        r1.iova.0 - r0.iova.0 == pool.slot_bytes(),
        "IOVA slots not stride apart"
    );
    anyhow::ensure!(
        pool.resolve_slot(0)? == r0,
        "re-resolving slot 0 moved it — the pool's IOVAs are not stable"
    );
    println!(
        "pinned: {} slot(s) x {} B, hmva {:#x}, iova {:#x} (stride-spaced)",
        pool.slot_count(),
        pool.slot_bytes(),
        r0.host_address.0,
        r0.iova.0
    );

    // --- 3. the data path: slot 0 -> device -> slot 1 ---
    let dev = DevAddr::alloc(ByteSize(SLOT_BYTES as u64), MemoryType::Tensor)
        .context("device buffer alloc")?;
    let dev_composite = arc_from_dev(&dev)?;
    let pattern = pattern_bytes(SLOT_BYTES);
    pool.slot_mut(0)?.copy_from_slice(&pattern);
    pool.enqueue_copy(&stream, 0, Some(dev_composite.clone()), true)
        .context("H2D enqueue")?;
    pool.enqueue_copy(&stream, 1, Some(dev_composite.clone()), false)
        .context("D2H enqueue")?;
    stream.synchronize().context("synchronize")?;
    anyhow::ensure!(
        pool.slot(1)? == &pattern[..],
        "D2H readback mismatch: slot 1 does not hold the pattern sent from slot 0"
    );
    println!(
        "round trip: slot 0 -> device -> slot 1, {} B byte-identical",
        SLOT_BYTES
    );

    // --- 4. the validation set refuses before any DMA ---
    let big = DevAddr::alloc(ByteSize((2 * SLOT_BYTES) as u64), MemoryType::Tensor)
        .context("oversize device buffer alloc")?;
    let big_composite = arc_from_dev(&big)?;
    match pool.enqueue_copy(&stream, 0, Some(big_composite.clone()), true) {
        Err(SharedHostPoolError::DeviceSizeExceedsSlot {
            device_size,
            host_capacity,
        }) => {
            anyhow::ensure!(
                device_size == 2 * SLOT_BYTES as u64 && host_capacity == SLOT_BYTES,
                "refusal carried the wrong geometry: {device_size} vs {host_capacity}"
            );
        }
        other => return Err(anyhow!("oversize device must be refused, got {other:?}")),
    }
    // Two real single-chunk allocations folded into one two-chunk address:
    // requireSingleChunk's port must fire before anything else submits.
    let (c_dev, c_big) = (dev.chunk0().unwrap(), big.chunk0().unwrap());
    let two_chunk = Arc::new(CompositeAddress::from_chunks(vec![
        Chunk::new(
            LogicalAddress::new(c_dev.region, c_dev.offset),
            c_dev.size,
            c_dev.domain,
        ),
        Chunk::new(
            LogicalAddress::new(c_big.region, c_big.offset),
            c_big.size,
            c_big.domain,
        ),
    ]));
    match pool.enqueue_copy(&stream, 0, Some(two_chunk), true) {
        Err(SharedHostPoolError::MultiChunkDevice { chunk_count }) => {
            anyhow::ensure!(chunk_count == 2, "expected 2 chunks, got {chunk_count}");
        }
        other => return Err(anyhow!("multi-chunk device must be refused, got {other:?}")),
    }
    println!("validation: oversize and multi-chunk devices refused before any DMA");

    // --- 5. teardown: the transfer-lifetime law, then drop ---
    stream.synchronize().context("final synchronize")?;
    drop(pool);
    println!("shared_host_pool_probe: PASS");
    Ok(())
}

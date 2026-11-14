// SPDX-License-Identifier: Apache-2.0

//! ⭐ THE DISK TIER OF THE PREFIX CACHE'S RECURRENT-STATE SNAPSHOTS (`SnapshotDiskOps`).
//!
//! A snapshot slot shares the KV budget, and on a 32 GB Mac serving Qwen3.6-35B-A3B that budget is
//! 1.3 GiB: some 14 snapshots of 65 MiB beside their prompts' KV. A workload that comes back to
//! more prompts than that found none of them, because LRU evicts each one just before its prompt
//! returns, and every hit resumed cold over a full KV match. Here a snapshot a slot is taken from
//! is written to a file and read back on a hit: 65 MiB of I/O against a prefill of every token.
//!
//! The file is unlinked once open, so it goes with the process however that ends. A disk slot
//! holds one snapshot slot's bytes in every GDN state store, in store order, at
//! `disk × slot_bytes`.
//!
//! ⛔ A SPILL COSTS ITS STEP ONE COPY, NOT A WRITE. The step that evicts a snapshot is the
//! prefill of the request whose save took its slot, so a write in that step is time to its first
//! token. Its slot is copied into a staging buffer, and a thread of its own writes that to the
//! file. A fill waits only for a write still under way to the disk slot it reads.

use std::collections::VecDeque;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::Arc;
use std::thread::JoinHandle;

use objc2_metal::{MTLBuffer as _, MTLDevice as _, MTLResourceOptions};
use scratchy_target_metal::argmax::Buffer;
use scratchy_target_metal::device_metal::Device;
use scratchy_target_metal::dflash2::{WindowCopy, encode_window_copies};
use scratchy_target_metal::mtl4_dispatch::Mtl4DispatchBatch;
use scratchy_target_metal::residency::MetalResidencySet;

/// One snapshot slot's bytes in one GDN state store: `(buffer, offset, bytes)`
/// (`SparseStores::block_regions`).
pub(crate) type Region = (Buffer, usize, usize);

/// Writes under way at once before a spill waits for the oldest: each holds a staging buffer of
/// one slot's bytes. A step evicts at most a unit's slots, and a write is done well before the
/// next prefill evicts again.
const MAX_WRITES: usize = 2;

/// A spill's write to the file, from a staging buffer it holds until it is done.
struct Write {
    disk: usize,
    staging: Buffer,
    done: JoinHandle<std::io::Result<()>>,
}

pub(crate) struct SnapshotDisk {
    file: Arc<File>,
    slot_bytes: usize,
    /// Staging buffers of one slot's bytes that no write holds.
    spare: Vec<Buffer>,
    writes: VecDeque<Write>,
    /// Per disk slot: the prefix key written there and whether its DFlash2 window was saved with
    /// it. A fill of another key, or of a slot whose write failed, is refused.
    held: Vec<Option<(u64, bool)>>,
}

impl SnapshotDisk {
    /// A disk tier of `slot_bytes` snapshots: as many as half the free space where temp files go
    /// holds, as oMLX sizes its SSD cache. `None` when that is none, or no file can be made.
    pub(crate) fn new(slot_bytes: usize) -> Option<Self> {
        let dir = std::env::temp_dir();
        let free = scratchy_core_model::weight::available_space(&dir)?;
        let slots = usize::try_from(free / 2).ok()? / slot_bytes.max(1);
        if slots == 0 {
            return None;
        }
        let path = dir.join(format!("scratchy-state-snapshots-{}", std::process::id()));
        let file = (File::options().read(true).write(true).create_new(true))
            .open(&path)
            .ok()?;
        std::fs::remove_file(&path).ok()?;
        Some(Self {
            file: Arc::new(file),
            slot_bytes,
            spare: Vec::new(),
            writes: VecDeque::new(),
            held: vec![None; slots],
        })
    }

    /// How many snapshots the tier holds.
    pub(crate) fn slots(&self) -> usize {
        self.held.len()
    }

    /// Write the snapshot in `regions` to disk slot `disk` as `key`'s, with whether its window
    /// was saved. Returns once it is copied out of `regions`; the write goes on behind.
    pub(crate) fn spill(
        &mut self,
        device: &Device,
        residency: &MetalResidencySet,
        regions: &[Region],
        disk: usize,
        key: u64,
        window: bool,
    ) -> Result<(), String> {
        self.reap(|w| w.done.is_finished() || w.disk == disk);
        if self.writes.len() >= MAX_WRITES {
            self.reap(|_| true);
        }
        let staging = self.staging(device)?;
        if let Err(e) = copy(device, residency, regions, &staging, self.slot_bytes, true) {
            self.spare.push(staging);
            return Err(e);
        }
        let (file, len, at) = (
            self.file.clone(),
            self.slot_bytes,
            (disk * self.slot_bytes) as u64,
        );
        let from = staging.contents().as_ptr() as usize;
        let done = std::thread::spawn(move || {
            // SAFETY: `from` is the contents of a shared buffer of `len` bytes, which this write's
            // `Write` holds — out of `spare`, so no copy touches it — until it is joined.
            let bytes = unsafe { std::slice::from_raw_parts(from as *const u8, len) };
            file.write_all_at(bytes, at)
        });
        self.held[disk] = Some((key, window));
        self.writes.push_back(Write {
            disk,
            staging,
            done,
        });
        Ok(())
    }

    /// Read `key`'s snapshot from disk slot `disk` into `regions`. Returns whether its window was
    /// saved, once it is in place.
    pub(crate) fn fill(
        &mut self,
        device: &Device,
        residency: &MetalResidencySet,
        regions: &[Region],
        disk: usize,
        key: u64,
    ) -> Result<bool, String> {
        // Only a write to this disk slot is in the way: another spill of this step writes another.
        self.reap(|w| w.done.is_finished() || w.disk == disk);
        let Some((held, window)) = self.held.get(disk).copied().flatten() else {
            return Err(format!("disk slot {disk} holds no snapshot"));
        };
        if held != key {
            return Err(format!("disk slot {disk} holds another prefix's snapshot"));
        }
        let staging = self.staging(device)?;
        // SAFETY: `staging` is a shared buffer of `slot_bytes` that no write or copy holds.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                staging.contents().as_ptr().cast::<u8>(),
                self.slot_bytes,
            )
        };
        let at = (disk * self.slot_bytes) as u64;
        let read = self.file.read_exact_at(bytes, at);
        let copied = read
            .map_err(|e| format!("reading disk slot {disk}: {e}"))
            .and_then(|()| copy(device, residency, regions, &staging, self.slot_bytes, false));
        self.spare.push(staging);
        copied.map(|()| window)
    }

    /// A staging buffer no write holds: a spare one, else a new one — not zeroed, since every move
    /// writes all of it before it reads any. At most [`MAX_WRITES`] writes' and a fill's.
    fn staging(&mut self, device: &Device) -> Result<Buffer, String> {
        if let Some(spare) = self.spare.pop() {
            return Ok(spare);
        }
        let shared = MTLResourceOptions::StorageModeShared;
        (device.newBufferWithLength_options(self.slot_bytes, shared))
            .ok_or_else(|| format!("a staging buffer of {} bytes", self.slot_bytes))
    }

    /// Wait for the writes `which` picks and take their staging buffers back. A disk slot whose
    /// write failed holds nothing a fill may read.
    fn reap(&mut self, which: impl Fn(&Write) -> bool) {
        let (done, under_way): (VecDeque<Write>, VecDeque<Write>) =
            std::mem::take(&mut self.writes)
                .into_iter()
                .partition(|w| which(w));
        self.writes = under_way;
        for w in done {
            let written = w
                .done
                .join()
                .unwrap_or_else(|_| Err(std::io::Error::other("the write's thread panicked")));
            if let Err(e) = written {
                tracing::warn!("snapshot disk: writing disk slot {}: {e}", w.disk);
                self.held[w.disk] = None;
            }
            self.spare.push(w.staging);
        }
    }
}

impl Drop for SnapshotDisk {
    /// A write reads its staging buffer until it is joined.
    fn drop(&mut self) {
        self.reap(|_| true);
    }
}

/// Copy `regions` into `staging` (`out`) or back out of it, and wait for the copy.
fn copy(
    device: &Device,
    residency: &MetalResidencySet,
    regions: &[Region],
    staging: &Buffer,
    slot_bytes: usize,
    out: bool,
) -> Result<(), String> {
    let total: usize = regions.iter().map(|&(_, _, len)| len).sum();
    if total != slot_bytes {
        return Err(format!(
            "a snapshot of {total} bytes in a disk tier of {slot_bytes}-byte slots"
        ));
    }
    let batch = Mtl4DispatchBatch::begin(device).ok_or("no MTL4 command queue")?;
    batch.declare(residency);
    let _staging = batch.residency().pin(staging.clone());
    let mut at = 0;
    let copies: Vec<WindowCopy<'_>> = (regions.iter())
        .map(|(buffer, offset, len)| {
            let (from, from_at, to, to_at) = match out {
                true => (&**buffer, *offset, &**staging, at),
                false => (&**staging, at, &**buffer, *offset),
            };
            at += len;
            WindowCopy {
                from,
                from_at,
                to,
                to_at,
                len: *len,
            }
        })
        .collect();
    // `regions` are a mapped slot's blocks, which the declared set makes resident.
    encode_window_copies(batch.encoder(), &copies);
    batch.commit(true);
    Ok(())
}

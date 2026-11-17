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
//!
//! ⛔ A FILL IS READ WHILE ITS STEP RUNS. The request a fill restores runs in a later step, but the
//! step that issues the fill runs every other request's next token, and a read before that step's
//! forward is time they all wait. So a step that runs any is read by a thread of its own
//! ([`SnapshotDisk::read_behind`]) while the step's forward is built and committed, and copied
//! into its slot ([`SnapshotDisk::restore`]) once it is. A step that runs none has nothing to
//! overlap, and reads on the worker's thread ([`SnapshotDisk::read`]).

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

    /// Disk slot `disk` holds nothing a fill may read: its spill failed.
    pub(crate) fn forget(&mut self, disk: usize) {
        if let Some(held) = self.held.get_mut(disk) {
            *held = None;
        }
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
        if self.writes.len() >= MAX_WRITES
            && let Some(oldest) = self.writes.pop_front()
        {
            self.join(oldest);
        }
        let staging = self.staging(device)?;
        if let Err(e) = copy(device, residency, regions, &staging, self.slot_bytes, true) {
            self.spare.push(staging);
            return Err(e);
        }
        let (file, len, at) = (self.file.clone(), self.slot_bytes, self.at(disk));
        let from = staging.contents().as_ptr() as usize;
        let spawned = std::thread::Builder::new()
            .name("snapshot-spill".into())
            .spawn(move || {
                // SAFETY: `from` is the contents of a shared buffer of `len` bytes, which this
                // write's `Write` holds — out of `spare`, so no copy touches it — until it is
                // joined.
                let bytes = unsafe { std::slice::from_raw_parts(from as *const u8, len) };
                file.write_all_at(bytes, at)
            });
        let done = match spawned {
            Ok(done) => done,
            Err(e) => {
                self.spare.push(staging);
                return Err(format!("a thread to write disk slot {disk}: {e}"));
            }
        };
        self.held[disk] = Some((key, window));
        self.writes.push_back(Write {
            disk,
            staging,
            done,
        });
        Ok(())
    }

    /// Read `key`'s snapshot from disk slot `disk` into a staging buffer, on this thread.
    /// [`Self::restore`] puts it into its slot.
    pub(crate) fn read(
        &mut self,
        device: &Device,
        disk: usize,
        key: u64,
    ) -> Result<Staged, String> {
        let (window, staging) = self.take(device, disk, key)?;
        // SAFETY: `staging` is a shared buffer of `slot_bytes` that no write or copy holds.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                staging.contents().as_ptr().cast::<u8>(),
                self.slot_bytes,
            )
        };
        let read = self.file.read_exact_at(bytes, self.at(disk));
        self.staged(disk, window, staging, read)
    }

    /// [`Self::read`] on a thread of its own, while this one goes on. [`Self::finish_read`] waits
    /// for it.
    pub(crate) fn read_behind(
        &mut self,
        device: &Device,
        disk: usize,
        key: u64,
    ) -> Result<Reading, String> {
        let (window, staging) = self.take(device, disk, key)?;
        let (file, len, at) = (self.file.clone(), self.slot_bytes, self.at(disk));
        let to = staging.contents().as_ptr() as usize;
        let spawned = std::thread::Builder::new()
            .name("snapshot-fill".into())
            .spawn(move || {
                // SAFETY: `to` is the contents of a shared buffer of `len` bytes, which this read's
                // `Reading` holds — out of `spare`, so no other move touches it — until it is
                // joined.
                let bytes = unsafe { std::slice::from_raw_parts_mut(to as *mut u8, len) };
                file.read_exact_at(bytes, at)
            });
        match spawned {
            Ok(read) => Ok(Reading {
                disk,
                window,
                under_way: Some((staging, read)),
            }),
            Err(e) => {
                self.spare.push(staging);
                Err(format!("a thread to read disk slot {disk}: {e}"))
            }
        }
    }

    /// Wait for the read [`Self::read_behind`] started: its snapshot, staged.
    pub(crate) fn finish_read(&mut self, mut reading: Reading) -> Result<Staged, String> {
        let Some((staging, read)) = reading.under_way.take() else {
            return Err(format!("disk slot {}: a read finished twice", reading.disk));
        };
        let read = (read.join())
            .unwrap_or_else(|_| Err(std::io::Error::other("the read's thread panicked")));
        self.staged(reading.disk, reading.window, staging, read)
    }

    /// The window flag and a staging buffer for a fill of `key` from disk slot `disk`, once no
    /// write to that disk slot is under way.
    fn take(&mut self, device: &Device, disk: usize, key: u64) -> Result<(bool, Buffer), String> {
        // Only a write to this disk slot is in the way: another spill of this step writes another.
        self.reap(|w| w.done.is_finished() || w.disk == disk);
        let Some((held, window)) = self.held.get(disk).copied().flatten() else {
            return Err(format!("disk slot {disk} holds no snapshot"));
        };
        if held != key {
            return Err(format!("disk slot {disk} holds another prefix's snapshot"));
        }
        Ok((window, self.staging(device)?))
    }

    /// `staging`, read from disk slot `disk`, as a staged snapshot; back among the spares if the
    /// read failed.
    fn staged(
        &mut self,
        disk: usize,
        window: bool,
        staging: Buffer,
        read: std::io::Result<()>,
    ) -> Result<Staged, String> {
        match read {
            Ok(()) => Ok(Staged { window, staging }),
            Err(e) => {
                self.spare.push(staging);
                Err(format!("reading disk slot {disk}: {e}"))
            }
        }
    }

    /// Where disk slot `disk` starts in the file.
    fn at(&self, disk: usize) -> u64 {
        (disk * self.slot_bytes) as u64
    }

    /// Copy `staged` into `regions`, its slot's. Returns whether its window was saved with it.
    pub(crate) fn restore(
        &mut self,
        device: &Device,
        residency: &MetalResidencySet,
        regions: &[Region],
        staged: Staged,
    ) -> Result<bool, String> {
        let copied = copy(
            device,
            residency,
            regions,
            &staged.staging,
            self.slot_bytes,
            false,
        );
        self.spare.push(staged.staging);
        copied.map(|()| staged.window)
    }

    /// A staging buffer no write holds: a spare one, else a new one — not zeroed, since every move
    /// writes all of it before it reads any. At most [`MAX_WRITES`] writes' and a step's fills'.
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
            self.join(w);
        }
    }

    /// Wait for write `w` and take its staging buffer back. A disk slot whose write failed holds
    /// nothing a fill may read.
    fn join(&mut self, w: Write) {
        let written = (w.done.join())
            .unwrap_or_else(|_| Err(std::io::Error::other("the write's thread panicked")));
        if let Err(e) = written {
            tracing::warn!("snapshot disk: writing disk slot {}: {e}", w.disk);
            self.held[w.disk] = None;
        }
        self.spare.push(w.staging);
    }
}

/// A snapshot read back from disk into a staging buffer ([`SnapshotDisk::read`]), which
/// [`SnapshotDisk::restore`] copies into its slot.
pub(crate) struct Staged {
    /// Whether its DFlash2 window was saved with it.
    window: bool,
    staging: Buffer,
}

/// A read of a snapshot under way on a thread of its own ([`SnapshotDisk::read_behind`]), into a
/// staging buffer it holds until [`SnapshotDisk::finish_read`] joins it.
pub(crate) struct Reading {
    disk: usize,
    window: bool,
    /// The staging buffer and the read that writes it, until `finish_read` takes them.
    under_way: Option<(Buffer, JoinHandle<std::io::Result<()>>)>,
}

impl Drop for Reading {
    /// A read writes its staging buffer until it is joined, so one never finished is waited for
    /// before the buffer goes.
    fn drop(&mut self) {
        if let Some((_staging, read)) = self.under_way.take() {
            let _ = read.join();
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

#[cfg(test)]
mod tests {
    use super::*;
    use scratchy_target_metal::mtl4_dispatch::{read_slice, shared_zeroed};

    /// A snapshot across two stores goes out to disk and comes back byte for byte. A read never
    /// restored leaves the slot as it was and the disk slot to a later fill. A disk slot whose
    /// spill failed, or that holds another prefix, is no fill: the worker reports it rather than
    /// restore what the slot held.
    #[test]
    fn a_snapshot_comes_back_as_it_went_out_and_no_other_does() {
        let Some(device) = scratchy_target_metal::detect_device().map(|d| d.device) else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };
        let residency = MetalResidencySet::new(&device);
        let (a, b) = (shared_zeroed(&device, 1024), shared_zeroed(&device, 1024));
        let _pinned = (residency.pin(a.clone()), residency.pin(b.clone()));
        residency.commit();
        let regions: Vec<Region> = vec![(a.clone(), 256, 256), (b.clone(), 0, 768)];
        let fill_with = |byte: u8| {
            for buffer in [&a, &b] {
                // SAFETY: a shared buffer of 1024 bytes no GPU work touches now.
                unsafe {
                    std::ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), byte, 1024)
                };
            }
        };
        let fill = |disk: &mut SnapshotDisk, d, key| {
            let staged = disk.read(&device, d, key)?;
            disk.restore(&device, &residency, &regions, staged)
        };
        let fill_behind = |disk: &mut SnapshotDisk, d, key| {
            let reading = disk.read_behind(&device, d, key)?;
            let staged = disk.finish_read(reading)?;
            disk.restore(&device, &residency, &regions, staged)
        };
        let restored = |byte: u8| {
            let (a_bytes, b_bytes) = (read_slice::<u8>(&a, 1024), read_slice::<u8>(&b, 1024));
            a_bytes[256..512].iter().all(|&x| x == byte)
                && (a_bytes[..256].iter().chain(&a_bytes[512..])).all(|&x| x == 0)
                && b_bytes[..768].iter().all(|&x| x == byte)
                && b_bytes[768..].iter().all(|&x| x == 0)
        };
        let mut disk = SnapshotDisk::new(1024).expect("a disk tier");
        fill_with(0xab);
        (disk.spill(&device, &residency, &regions, 1, 7, true)).expect("spill");
        fill_with(0);
        let window = fill(&mut disk, 1, 7).expect("fill");
        assert!(window, "its window comes back with it");
        assert!(restored(0xab), "its bytes in its regions, and nothing else");

        fill_with(0);
        assert!(fill_behind(&mut disk, 1, 7).expect("a fill read behind"));
        assert!(
            restored(0xab),
            "read on a thread of its own, the same bytes"
        );

        fill_with(0);
        drop(disk.read(&device, 1, 7).expect("a read"));
        drop(disk.read_behind(&device, 1, 7).expect("a read behind"));
        assert!(
            restored(0),
            "a read never restored leaves the slot as it was"
        );
        fill(&mut disk, 1, 7).expect("the disk slot still holds it");
        assert!(restored(0xab));

        type Fill<'f> = &'f dyn Fn(&mut SnapshotDisk, usize, u64) -> Result<bool, String>;
        for each in [&fill as Fill<'_>, &fill_behind] {
            assert!(each(&mut disk, 1, 8).is_err(), "another prefix's");
            assert!(each(&mut disk, 0, 7).is_err(), "never written");
        }
        disk.forget(1);
        assert!(fill(&mut disk, 1, 7).is_err(), "its spill failed");
        assert!(fill_behind(&mut disk, 1, 7).is_err(), "its spill failed");
    }
}

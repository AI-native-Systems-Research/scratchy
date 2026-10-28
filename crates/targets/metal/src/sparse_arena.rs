// SPDX-License-Identifier: Apache-2.0
//
//! Placement-sparse activations ([`SparseBuffer`]): a worker's arena slots and scratch, each one
//! buffer as long as the largest bucket needs, mapped only as far as the buckets that ran reach
//! ([`reach`]). Allocated in full and wired with the working set, gemma-4-26b's took 4.4 GiB
//! (a 2.5 GiB arena, 2 GiB of it the 4096-row bucket's logits, and 1.9 GiB of scratch), zero-filled
//! on the first forward whatever its size: ~55 ms of a 20-token request's first token on a base M5.

use std::sync::{Arc, Mutex};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLHeap, MTLResourceOptions};

use crate::residency::{MetalResidencySet, Pinned};
use crate::sparse_kv::{KvMapper, MapRange, PAGE, map_step};

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

/// One placement-sparse buffer and how far it is mapped. Every worker binding it holds the same
/// one, so a guest that reaches further than its host maps for both.
pub struct SparseBuffer {
    buffer: Buffer,
    mapped: Mutex<Mapped>,
}

#[derive(Default)]
struct Mapped {
    /// Bytes from the start that read and write memory: whole tiles.
    bytes: usize,
    /// The heaps backing them, each shared with the buffers mapped in the same step.
    heaps: Vec<Arc<StepHeap>>,
}

/// One step's heap, wired for as long as any buffer it backs lives.
struct StepHeap {
    _heap: Pinned<ProtocolObject<dyn MTLHeap>>,
}

// SAFETY: a Metal heap is safe to share across threads, and nothing reads it through this handle:
// it only keeps the heap, and its residency pin, alive.
unsafe impl Send for StepHeap {}
unsafe impl Sync for StepHeap {}

impl SparseBuffer {
    /// A buffer of `bytes`, rounded up to whole tiles, that takes no memory until [`reach`] maps
    /// it. Private: the host never reads it.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, bytes: u64) -> Result<Arc<Self>, String> {
        let tile = device.sparseTileSizeInBytesForSparsePageSize(PAGE);
        let len = usize::try_from(bytes.max(1))
            .map_err(|_| format!("sparse buffer of {bytes} bytes"))?
            .next_multiple_of(tile);
        let private = MTLResourceOptions::StorageModePrivate;
        // SAFETY: a fresh allocation; nothing reads it until a step maps it and is waited out.
        let buffer = unsafe {
            device.newBufferWithLength_options_placementSparsePageSize(len, private, PAGE)
        };
        let buffer = buffer.ok_or_else(|| format!("sparse buffer of {len} bytes returned nil"))?;
        Ok(Arc::new(Self {
            buffer,
            mapped: Mutex::default(),
        }))
    }

    /// The buffer, for the dispatches that bind it by address.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Bytes mapped so far.
    pub fn mapped(&self) -> usize {
        self.mapped.lock().expect("sparse buffer mutex").bytes
    }
}

/// `buffer`'s contents, for the host to read, or null where it has none: private storage, as every
/// sparse activation is. The GPU binds such a buffer, never this pointer.
pub fn host_ptr(buffer: &ProtocolObject<dyn MTLBuffer>) -> *mut u8 {
    use objc2_metal::{MTLResource, MTLStorageMode};
    match buffer.storageMode() {
        MTLStorageMode::Private => std::ptr::null_mut(),
        _ => buffer.contents().as_ptr().cast(),
    }
}

// SAFETY: the buffer and heaps are Metal objects, safe to share across threads; the mapped length
// is behind its mutex.
unsafe impl Send for SparseBuffer {}
unsafe impl Sync for SparseBuffer {}

/// Map each buffer through its `bytes` (up to its length, in whole tiles) where it is not yet: one
/// heap for all a call adds, on `mapper`, wired in `residency` and zeroed — as a buffer allocated in
/// full reads — in place when this returns. A buffer named twice is mapped through the larger.
/// Whether it mapped anything.
pub fn reach(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    residency: &MetalResidencySet,
    mapper: &KvMapper,
    wants: &[(&SparseBuffer, usize)],
) -> Result<bool, String> {
    // One step at a time: buffers are shared between workers, and a step reads each one's mapped
    // length before it maps past it.
    static STEP: Mutex<()> = Mutex::new(());
    let _step = STEP.lock().expect("sparse step mutex");
    let tile = device.sparseTileSizeInBytesForSparsePageSize(PAGE);
    let mut grow: Vec<(&SparseBuffer, usize, usize)> = Vec::new();
    for &(buffer, bytes) in wants {
        // Clamped first: the length is whole tiles, so rounding up stays within it.
        let to = bytes.min(buffer.buffer.length()).next_multiple_of(tile);
        match grow.iter_mut().find(|g| std::ptr::eq(g.0, buffer)) {
            Some(g) => g.2 = g.2.max(to),
            None => grow.push((buffer, buffer.mapped(), to)),
        }
    }
    grow.retain(|&(_, from, to)| to > from);
    if grow.is_empty() {
        return Ok(false);
    }
    let ranges: Vec<MapRange<'_>> = (grow.iter())
        .map(|&(buffer, from, to)| MapRange {
            buffer: &buffer.buffer,
            from,
            to,
            zeroed: true,
        })
        .collect();
    let heap = Arc::new(StepHeap {
        _heap: map_step(device, residency, mapper, &ranges)?,
    });
    for (buffer, _, to) in grow {
        let mut mapped = buffer.mapped.lock().expect("sparse buffer mutex");
        mapped.bytes = to;
        mapped.heaps.push(heap.clone());
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mtl4_dispatch::{Mtl4DispatchBatch, read_slice, shared_zeroed};
    use objc2_foundation::NSRange;
    use objc2_metal::MTL4ComputeCommandEncoder;

    /// Buffers reached together share a step's heap without aliasing, reaching no further maps
    /// nothing, a newly mapped range reads zero, mapped bytes keep their writes when the buffer
    /// grows, and a buffer named twice maps through the larger.
    #[test]
    fn buffers_map_only_as_far_as_reached() {
        let Some(device) = crate::detect_device().map(|d| d.device) else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };
        let residency = MetalResidencySet::new(&device);
        let mapper = KvMapper::new(&device).expect("mapper");
        let tile = device.sparseTileSizeInBytesForSparsePageSize(PAGE);
        let a = SparseBuffer::new(&device, 4 * tile as u64).expect("a");
        let b = SparseBuffer::new(&device, 2 * tile as u64 + 1).expect("b");
        let _pins = [a.buffer().clone(), b.buffer().clone()].map(|x| residency.pin(x));
        assert_eq!(
            (a.buffer().length(), b.buffer().length()),
            (4 * tile, 3 * tile)
        );
        assert_eq!((a.mapped(), b.mapped()), (0, 0));

        let step = |wants: &[(&SparseBuffer, usize)]| reach(&device, &residency, &mapper, wants);
        assert!(step(&[(&a, 1), (&b, tile)]).expect("step 1"));
        assert_eq!((a.mapped(), b.mapped()), (tile, tile));
        assert!(!step(&[(&a, tile), (&b, 1)]).expect("nothing further"));

        let host = shared_zeroed(&device, 4 * tile);
        let gpu = |body: &dyn Fn(&ProtocolObject<dyn MTL4ComputeCommandEncoder>)| {
            let batch = Mtl4DispatchBatch::begin(&device).expect("MTL4 queue");
            batch.declare(&residency);
            let _host = batch.residency().pin(host.clone());
            body(batch.encoder());
            batch.commit(true);
        };
        // SAFETY (both): the ranges are mapped and resident, and `host` is shared.
        gpu(&|e| unsafe {
            e.fillBuffer_range_value(a.buffer(), NSRange::new(0, tile), 0xab);
            e.fillBuffer_range_value(b.buffer(), NSRange::new(0, tile), 0xcd);
        });
        assert!(step(&[(&a, 2 * tile), (&a, 3 * tile)]).expect("step 2"));
        assert_eq!((a.mapped(), b.mapped()), (3 * tile, tile));
        gpu(&|e| unsafe {
            let copy = |from, at, to, n| {
                e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    from, at, &host, to, n,
                )
            };
            copy(a.buffer(), 0, 0, 3 * tile);
            copy(b.buffer(), 0, 3 * tile, tile);
        });
        let got: Vec<u8> = read_slice(&host, 4 * tile);
        let reads = |at: usize, n: usize, v: u8| got[at..at + n].iter().all(|&x| x == v);
        assert!(
            reads(0, tile, 0xab),
            "mapped bytes keep their writes as the buffer grows"
        );
        assert!(reads(tile, 2 * tile, 0), "a newly mapped range reads zero");
        assert!(
            reads(3 * tile, tile, 0xcd),
            "buffers of one step do not alias"
        );
        assert!(step(&[(&b, usize::MAX)]).expect("to the length"));
        assert_eq!(
            b.mapped(),
            3 * tile,
            "a buffer maps no further than its length"
        );
    }
}

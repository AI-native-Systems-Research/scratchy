// SPDX-License-Identifier: Apache-2.0
//
//! Placement-sparse storage for the metal KV pools ([`SparseStores`]): mapped as the pool
//! reaches its blocks, on a worker's [`KvMapper`].

use std::ptr::NonNull;
use std::sync::{Arc, atomic::AtomicU64, atomic::Ordering};

use objc2::{Message, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSRange;
use objc2_metal::{
    MTL4CommandQueue, MTL4ComputeCommandEncoder, MTL4UpdateSparseBufferMappingOperation, MTLDevice,
    MTLHeap, MTLHeapDescriptor, MTLHeapType, MTLResourceOptions, MTLSharedEvent, MTLSparsePageSize,
    MTLSparseTextureMappingMode, MTLStorageMode,
};

use crate::metal_mem::MetalMem;
use crate::mtl4_dispatch::Mtl4DispatchBatch;
use crate::residency::{MetalResidencySet, Pinned};

const PAGE: MTLSparsePageSize = MTLSparsePageSize::Size64;

/// The queue a worker's KV mappings run on, and the event that says they have.
pub struct KvMapper {
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    signaled: AtomicU64,
}

impl KvMapper {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Arc<Self>, String> {
        let queue = device
            .newMTL4CommandQueue()
            .ok_or("newMTL4CommandQueue: nil")?;
        let event = device.newSharedEvent().ok_or("newSharedEvent: nil")?;
        Ok(Arc::new(Self {
            queue,
            event,
            signaled: AtomicU64::new(0),
        }))
    }

    /// Block until every mapping queued so far is in place.
    pub fn wait(&self) {
        let value = self.signaled.fetch_add(1, Ordering::Relaxed) + 1;
        self.queue
            .signalEvent_value(ProtocolObject::from_ref(&*self.event), value);
        crate::mtl4_dispatch::wait_drained(&self.event, value);
    }
}

/// Placement-sparse buffers that grow together, as one pool's stores do (each `(tensor, K/V)` of
/// the fp16 pool, or TurboQuant's packed codes, norms and scratch): ONE contiguous range each, so
/// readers address any block off its base, whose pages are mapped only as the pool reaches them.
/// A sparse buffer in the wired residency set takes no memory, where a plain capacity-sized one
/// is committed in full when made resident (gemma-4-26b on a 64 GB Mac: ~19 GiB of KV at
/// startup). Each step maps the next run of blocks of every buffer from ONE heap and at least
/// doubles what is mapped, so a pool reached through `n` blocks holds O(log n) heaps — not one
/// per buffer per chunk (~65k for granite-2b's TurboQuant stores on a 32 GB M5).
pub struct SparseStores {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    residency: MetalResidencySet,
    /// Each store's buffer, a block's bytes in it, and whether a new block must read zero.
    stores: Vec<(Pinned, usize, bool)>,
    /// Steps map a whole number of these blocks: whole tiles of every store.
    unit: usize,
    /// Blocks mapped so far, and all a store holds (a whole number of `unit`s).
    mapped: usize,
    capacity: usize,
    heaps: Vec<Pinned<ProtocolObject<dyn MTLHeap>>>,
}

impl SparseStores {
    /// Stores of `blocks` blocks, `(block_bytes, zeroed)` each, grown `granule` blocks or more at
    /// a time. Their buffers take no memory until a step maps them.
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        residency: &MetalResidencySet,
        blocks: usize,
        granule: usize,
        stores: &[(usize, bool)],
    ) -> Result<Self, String> {
        let tile = device.sparseTileSizeInBytesForSparsePageSize(PAGE);
        let whole = |unit: usize| stores.iter().all(|s| (unit * s.0).is_multiple_of(tile));
        let mut unit = granule.max(1);
        while !whole(unit) {
            unit *= 2;
        }
        let capacity = blocks.max(1).next_multiple_of(unit);
        let sparse = |&(bytes, zeroed): &(usize, bool)| {
            let (len, private) = (capacity * bytes, MTLResourceOptions::StorageModePrivate);
            // SAFETY: a fresh allocation; nothing reads it until a step maps it and is waited out.
            let buffer = unsafe {
                device.newBufferWithLength_options_placementSparsePageSize(len, private, PAGE)
            };
            let buffer =
                buffer.ok_or_else(|| format!("sparse buffer of {len} bytes returned nil"))?;
            Ok((residency.pin(buffer), bytes, zeroed))
        };
        Ok(Self {
            device: device.retain(),
            residency: residency.clone(),
            stores: stores.iter().map(sparse).collect::<Result<_, String>>()?,
            unit,
            mapped: 0,
            capacity,
            heaps: Vec::new(),
        })
    }

    /// Store `i`'s buffer, for readers that bind it by address.
    pub fn buffer(&self, i: usize) -> crate::argmax::Buffer {
        (*self.stores[i].0).clone()
    }

    /// `bytes` of store `i` from block `block` on, for a pool's chunk table.
    pub fn chunk(&self, i: usize, block: usize, bytes: usize) -> Result<MetalMem, String> {
        let (buffer, block_bytes, _) = &self.stores[i];
        let (offset, mapped) = (block * block_bytes, self.mapped * block_bytes);
        match offset + bytes <= mapped {
            true => Ok(MetalMem::from_buffer_with_offset(
                (**buffer).clone(),
                offset,
                bytes,
            )),
            false => Err(format!(
                "store {i}: {bytes} bytes at {offset}, {mapped} mapped"
            )),
        }
    }

    /// Map every store through `blocks` blocks, if it is not yet: one heap for the step, which at
    /// least doubles what is mapped (up to the capacity), mapped on `mapper`, wired, and zeroed
    /// where a store's blocks must read zero — in place when this returns. No forward in flight
    /// names a block past what was mapped before, so this races none. Whether it mapped a step.
    pub fn grow_to(&mut self, blocks: usize, mapper: &KvMapper) -> Result<bool, String> {
        if blocks > self.capacity {
            return Err(format!("{blocks} blocks, of {}", self.capacity));
        }
        if blocks <= self.mapped {
            return Ok(false);
        }
        let from = self.mapped;
        let to = blocks.next_multiple_of(self.unit).max(2 * from);
        let to = to.min(self.capacity);
        let tile = self.device.sparseTileSizeInBytesForSparsePageSize(PAGE);
        let desc = MTLHeapDescriptor::new();
        desc.setType(MTLHeapType::Placement);
        desc.setStorageMode(MTLStorageMode::Private);
        desc.setSize(self.stores.iter().map(|s| (to - from) * s.1).sum());
        desc.setMaxCompatiblePlacementSparsePageSize(PAGE);
        let heap = self.device.newHeapWithDescriptor(&desc);
        let heap = heap.ok_or("placement heap: nil")?;
        let mut at = 0;
        for (buffer, bytes, _) in &self.stores {
            let tiles = (to - from) * bytes / tile;
            let op = MTL4UpdateSparseBufferMappingOperation {
                mode: MTLSparseTextureMappingMode::Map,
                bufferRange: NSRange::new(from * bytes / tile, tiles),
                heapOffset: at,
            };
            // SAFETY: `op` maps whole tiles of this buffer's range from the heap's next `tiles`.
            unsafe {
                (mapper.queue).updateBufferMappings_heap_operations_count(
                    buffer,
                    Some(&heap),
                    NonNull::from(&op),
                    1,
                )
            };
            at += tiles;
        }
        self.heaps.push(self.residency.pin(heap));
        mapper.wait();
        self.residency.commit();
        let mut zeroed = self.stores.iter().filter(|s| s.2).peekable();
        if zeroed.peek().is_some() {
            let batch = Mtl4DispatchBatch::begin(&self.device).ok_or("zero fill: no MTL4 queue")?;
            batch.declare(&self.residency);
            for (buffer, bytes, _) in zeroed {
                let range = NSRange::new(from * bytes, (to - from) * bytes);
                // SAFETY: the range is mapped and resident, and nothing else touches it yet.
                unsafe { batch.encoder().fillBuffer_range_value(buffer, range, 0) };
            }
            batch.commit(true);
        }
        self.mapped = to;
        Ok(true)
    }
}

/// Every [`SparseStores`] a device's KV pool maps, and how far the pool's blocks reach: the worker
/// grows them all before each step's forward ([`Self::reach`]), and stores registered after — a
/// model's runtime is built on its first forward — are mapped as far before anything reads them.
#[derive(Default)]
pub struct SparseRegistry {
    stores: Vec<SparseStores>,
    blocks: usize,
    mapper: Option<Arc<KvMapper>>,
}

impl SparseRegistry {
    pub fn register(&mut self, mut stores: SparseStores) -> Result<(), String> {
        if let Some(mapper) = &self.mapper {
            stores.grow_to(self.blocks, mapper)?;
        }
        self.stores.push(stores);
        Ok(())
    }

    /// Map every store through `blocks` blocks, on `mapper`. Whether any mapped a step.
    pub fn reach(&mut self, blocks: usize, mapper: &Arc<KvMapper>) -> Result<bool, String> {
        self.blocks = self.blocks.max(blocks);
        self.mapper = Some(mapper.clone());
        let blocks = self.blocks;
        let mut stores = self.stores.iter_mut();
        stores.try_fold(false, |grew, s| Ok(s.grow_to(blocks, mapper)? || grew))
    }
}

unsafe impl Send for SparseStores {}
unsafe impl Sync for SparseStores {}
unsafe impl Send for KvMapper {}
unsafe impl Sync for KvMapper {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mtl4_dispatch::{read_slice, shared_zeroed};

    /// Stores grown together share each step's heap without aliasing (`heapOffset` counts
    /// tiles), a zeroed store's new blocks read zero, mapped blocks keep their writes, and each
    /// step at least doubles what is mapped.
    #[test]
    fn stores_grow_in_doubling_steps_without_aliasing() {
        let Some(device) = crate::detect_device().map(|d| d.device) else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };
        let residency = MetalResidencySet::new(&device);
        let mapper = KvMapper::new(&device).expect("mapper");
        // 128 blocks of the first store are one 64 KB tile, of the second two.
        let layout = [(512, true), (1024, false)];
        let mut stores =
            SparseStores::new(&device, &residency, 1000, 128, &layout).expect("stores");
        assert_eq!((stores.unit, stores.capacity), (128, 1024));
        let host = shared_zeroed(&device, 256 * 1024);
        let gpu = |body: &dyn Fn(&ProtocolObject<dyn MTL4ComputeCommandEncoder>)| {
            let batch = Mtl4DispatchBatch::begin(&device).expect("MTL4 queue");
            batch.declare(&residency);
            let _host = batch.residency().pin(host.clone());
            body(batch.encoder());
            batch.commit(true);
        };
        let (a, b) = (stores.buffer(0), stores.buffer(1));
        assert!(stores.grow_to(1, &mapper).expect("step 1"));
        // SAFETY (both): the ranges are mapped and resident, and `host` is shared.
        gpu(&|e| unsafe {
            e.fillBuffer_range_value(&a, NSRange::new(0, 128 * 512), 0xab);
            e.fillBuffer_range_value(&b, NSRange::new(0, 128 * 1024), 0xcd);
        });
        assert!(stores.grow_to(129, &mapper).expect("step 2"));
        assert!(!stores.grow_to(256, &mapper).expect("mapped"));
        // A's 256 mapped blocks, then B's first 128, into `host`.
        gpu(&|e| unsafe {
            let copy = |from, to, n| {
                e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(from, 0, &host, to, n)
            };
            copy(&a, 0, 256 * 512);
            copy(&b, 256 * 512, 128 * 1024);
        });
        let got: Vec<u8> = read_slice(&host, 256 * 1024);
        let reads = |at: usize, n: usize, v: u8| got[at..at + n].iter().all(|&x| x == v);
        assert!(reads(0, 128 * 512, 0xab), "a mapped block keeps its writes");
        assert!(
            reads(128 * 512, 128 * 512, 0),
            "a zeroed store's new blocks read zero"
        );
        assert!(
            reads(256 * 512, 128 * 1024, 0xcd),
            "the stores share a heap, apart"
        );
        assert!(stores.grow_to(257, &mapper).expect("step 3"));
        assert!(stores.grow_to(600, &mapper).expect("step 4"));
        assert_eq!((stores.mapped, stores.heaps.len()), (1024, 4));
        assert!(stores.grow_to(1025, &mapper).is_err(), "past the capacity");
    }
}

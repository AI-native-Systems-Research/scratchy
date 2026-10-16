// SPDX-License-Identifier: Apache-2.0
//
//! Placement-sparse storage for the metal KV pools ([`SparseStores`]): mapped as the pool
//! reaches its blocks, on a worker's [`KvMapper`]. Its mapping step ([`map_step`]) also maps a
//! worker's activations as its buckets reach them ([`crate::sparse_arena`]).

use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use objc2::{Message, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSRange;
use objc2_metal::{
    MTL4CommandQueue, MTL4ComputeCommandEncoder, MTL4UpdateSparseBufferMappingOperation, MTLBuffer,
    MTLDevice, MTLHeap, MTLHeapDescriptor, MTLHeapType, MTLResourceOptions, MTLSharedEvent,
    MTLSparsePageSize, MTLSparseTextureMappingMode, MTLStorageMode,
};

use crate::metal_mem::MetalMem;
use crate::mtl4_dispatch::Mtl4DispatchBatch;
use crate::residency::{MetalResidencySet, Pinned};

pub(crate) const PAGE: MTLSparsePageSize = MTLSparsePageSize::Size64;

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
/// the fp16 pool, TurboQuant's packed codes, norms and scratch, or a recurrent model's conv and
/// ssm state per layer): ONE contiguous range each, so readers address any block off its base,
/// whose pages are mapped only as the pool reaches them — each store through the tile its last
/// block ends in, so a block need not fill whole tiles.
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
    /// Steps map a whole number of these blocks (the granule).
    unit: usize,
    /// Blocks mapped so far (shared with the forwards that bind the stores), and all a store
    /// holds (a whole number of `unit`s).
    mapped: Arc<AtomicUsize>,
    capacity: usize,
    heaps: Vec<Pinned<ProtocolObject<dyn MTLHeap>>>,
    /// A second run of blocks past the first, mapped on its own (`Self::with_tail`).
    tail: Option<Tail>,
}

/// [`SparseStores`]' second run of blocks: `blocks` of them from block `start` — the first block
/// whose offset is a whole number of tiles in every store, so no tile is shared with the first run
/// — `mapped` of them mapped so far.
struct Tail {
    start: usize,
    blocks: usize,
    mapped: usize,
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
        let unit = granule.max(1);
        let capacity = blocks.max(1).next_multiple_of(unit);
        let sparse = |&(bytes, zeroed): &(usize, bool)| {
            let len = (capacity * bytes).next_multiple_of(tile);
            let private = MTLResourceOptions::StorageModePrivate;
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
            mapped: Arc::default(),
            capacity,
            heaps: Vec::new(),
            tail: None,
        })
    }

    /// Stores of `head` blocks grown from block 0 ([`Self::grow_to`]) and, past them, `tail` more
    /// grown on their own from [`Self::tail_start`] ([`Self::grow_tail_to`]) — a recurrent model's
    /// live state slots and its prefix-cache snapshot slots. Neither run takes memory until a step
    /// maps it.
    pub fn with_tail(
        device: &ProtocolObject<dyn MTLDevice>,
        residency: &MetalResidencySet,
        head: usize,
        tail: usize,
        stores: &[(usize, bool)],
    ) -> Result<Self, String> {
        let tile = device.sparseTileSizeInBytesForSparsePageSize(PAGE);
        let aligned = |b: usize| {
            stores
                .iter()
                .all(|&(bytes, _)| (b * bytes).is_multiple_of(tile))
        };
        let start = (head..)
            .find(|&b| aligned(b))
            .expect("a tile-aligned block");
        let mut s = Self::new(device, residency, start + tail, 1, stores)?;
        s.capacity = head;
        s.tail = Some(Tail {
            start,
            blocks: tail,
            mapped: 0,
        });
        Ok(s)
    }

    /// The first block of the second run (`Self::with_tail`).
    pub fn tail_start(&self) -> Option<usize> {
        self.tail.as_ref().map(|t| t.start)
    }

    /// Map the second run through its first `blocks` blocks, as [`Self::grow_to`] maps the first:
    /// at least doubling what it maps. Whether it mapped a step.
    pub fn grow_tail_to(&mut self, blocks: usize, mapper: &KvMapper) -> Result<bool, String> {
        let Some(tail) = self.tail.as_ref() else {
            return Err("no second run of blocks".into());
        };
        let (start, from) = (tail.start, tail.mapped);
        if blocks > tail.blocks {
            return Err(format!("{blocks} tail blocks, of {}", tail.blocks));
        }
        if blocks <= from {
            return Ok(false);
        }
        let to = blocks.max(2 * from).min(tail.blocks);
        self.map(start + from, start + to, mapper)?;
        if let Some(tail) = self.tail.as_mut() {
            tail.mapped = to;
        }
        Ok(true)
    }

    /// Map blocks `from..to` of every store: one heap, wired, and zeroed where a store's blocks must
    /// read zero.
    fn map(&mut self, from: usize, to: usize, mapper: &KvMapper) -> Result<(), String> {
        let tile = self.device.sparseTileSizeInBytesForSparsePageSize(PAGE);
        // A store's bytes through block `b`, rounded up to its tile: the last tile may be shared
        // with block `b`'s successor, which the step that reaches it then finds already mapped —
        // and zeroed, when its store must read zero, as `map_step` zeroes whole new tiles.
        let through = |b: usize, bytes: usize| (b * bytes).next_multiple_of(tile);
        let ranges: Vec<MapRange<'_>> = (self.stores.iter())
            .map(|(buffer, bytes, zeroed)| MapRange {
                buffer,
                from: through(from, *bytes),
                to: through(to, *bytes),
                zeroed: *zeroed,
            })
            .filter(|r| r.to > r.from)
            .collect();
        let heap = map_step(&self.device, &self.residency, mapper, &ranges)?;
        self.heaps.push(heap);
        Ok(())
    }

    /// Store `i`'s buffer, for readers that bind it by address.
    pub fn buffer(&self, i: usize) -> crate::argmax::Buffer {
        (*self.stores[i].0).clone()
    }

    /// The blocks mapped so far, as each step publishes them.
    pub fn mapped(&self) -> Arc<AtomicUsize> {
        self.mapped.clone()
    }

    /// `bytes` of store `i` from block `block` on, for a pool's chunk table.
    pub fn chunk(&self, i: usize, block: usize, bytes: usize) -> Result<MetalMem, String> {
        let (buffer, block_bytes, _) = &self.stores[i];
        let mapped = self.mapped.load(Ordering::Relaxed) * block_bytes;
        let offset = block * block_bytes;
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
        let from = self.mapped.load(Ordering::Relaxed);
        if blocks <= from {
            return Ok(false);
        }
        let to = blocks.next_multiple_of(self.unit).max(2 * from);
        let to = to.min(self.capacity);
        self.map(from, to, mapper)?;
        self.mapped.store(to, Ordering::Relaxed);
        Ok(true)
    }
}

/// One range of a placement-sparse buffer for [`map_step`] to map: bytes `from..to`, whole tiles
/// of [`PAGE`], and whether they must read zero.
pub(crate) struct MapRange<'a> {
    pub buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub from: usize,
    pub to: usize,
    pub zeroed: bool,
}

/// Map every range from ONE new placement heap, on `mapper`, wire the heap in `residency`, and zero
/// the ranges that must read zero: in place when this returns. The heap backs every range it
/// mapped, so it lives as long as any of them is read. No forward in flight names a byte of a range
/// not mapped before, so this races none.
pub(crate) fn map_step(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    residency: &MetalResidencySet,
    mapper: &KvMapper,
    ranges: &[MapRange<'_>],
) -> Result<Pinned<ProtocolObject<dyn MTLHeap>>, String> {
    let tile = device.sparseTileSizeInBytesForSparsePageSize(PAGE);
    let desc = MTLHeapDescriptor::new();
    desc.setType(MTLHeapType::Placement);
    desc.setStorageMode(MTLStorageMode::Private);
    desc.setSize(ranges.iter().map(|r| r.to - r.from).sum());
    desc.setMaxCompatiblePlacementSparsePageSize(PAGE);
    let heap = device.newHeapWithDescriptor(&desc);
    let heap = heap.ok_or("placement heap: nil")?;
    let mut at = 0;
    for r in ranges {
        let tiles = (r.to - r.from) / tile;
        let op = MTL4UpdateSparseBufferMappingOperation {
            mode: MTLSparseTextureMappingMode::Map,
            bufferRange: NSRange::new(r.from / tile, tiles),
            heapOffset: at,
        };
        // SAFETY: `op` maps whole tiles of this buffer's range from the heap's next `tiles`.
        unsafe {
            (mapper.queue).updateBufferMappings_heap_operations_count(
                r.buffer,
                Some(&heap),
                NonNull::from(&op),
                1,
            )
        };
        at += tiles;
    }
    let heap = residency.pin(heap);
    mapper.wait();
    residency.commit();
    let mut zeroed = ranges.iter().filter(|r| r.zeroed).peekable();
    if zeroed.peek().is_some() {
        let batch = Mtl4DispatchBatch::begin(device).ok_or("zero fill: no MTL4 queue")?;
        batch.declare(residency);
        for r in zeroed {
            let range = NSRange::new(r.from, r.to - r.from);
            // SAFETY: the range is mapped and resident, and nothing else touches it yet.
            unsafe { batch.encoder().fillBuffer_range_value(r.buffer, range, 0) };
        }
        batch.commit(true);
    }
    Ok(heap)
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
        assert_eq!(
            (stores.mapped().load(Ordering::Relaxed), stores.heaps.len()),
            (1024, 4)
        );
        assert!(stores.grow_to(1025, &mapper).is_err(), "past the capacity");
    }

    /// Blocks that do not fill whole tiles (a recurrent state slot): each step maps every store
    /// through the tile its last block ends in, one block at a time, and a block that straddles
    /// a tile boundary keeps the writes made before its second tile was mapped.
    #[test]
    fn blocks_that_straddle_tiles_grow_one_at_a_time() {
        let Some(device) = crate::detect_device().map(|d| d.device) else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };
        let residency = MetalResidencySet::new(&device);
        let mapper = KvMapper::new(&device).expect("mapper");
        // 1.5 tiles, and a store whose block is a sliver of one.
        let (big, small) = (3 * 32 * 1024, 2048);
        let mut stores =
            SparseStores::new(&device, &residency, 5, 1, &[(big, true), (small, false)])
                .expect("stores");
        assert_eq!((stores.unit, stores.capacity), (1, 5));
        let host = shared_zeroed(&device, 3 * big);
        let a = stores.buffer(0);
        let gpu = |body: &dyn Fn(&ProtocolObject<dyn MTL4ComputeCommandEncoder>)| {
            let batch = Mtl4DispatchBatch::begin(&device).expect("MTL4 queue");
            batch.declare(&residency);
            let _host = batch.residency().pin(host.clone());
            body(batch.encoder());
            batch.commit(true);
        };
        assert!(stores.grow_to(1, &mapper).expect("block 0"));
        // SAFETY (both): the ranges are mapped and resident, and `host` is shared.
        gpu(&|e| unsafe { e.fillBuffer_range_value(&a, NSRange::new(0, big), 0xab) });
        assert!(stores.grow_to(3, &mapper).expect("blocks 1 and 2"));
        gpu(&|e| unsafe {
            e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&a, 0, &host, 0, 3 * big)
        });
        let got: Vec<u8> = read_slice(&host, 3 * big);
        assert!(
            got[..big].iter().all(|&x| x == 0xab),
            "block 0 keeps its writes"
        );
        assert!(
            got[big..].iter().all(|&x| x == 0),
            "blocks 1 and 2 read zero"
        );
        assert!(stores.grow_to(5, &mapper).expect("the rest"));
        assert_eq!(stores.mapped().load(Ordering::Relaxed), 5);
    }

    /// Two runs grown apart — a recurrent model's live slots and its snapshot slots: the second
    /// starts on a whole tile of every store, past the first's last block, and maps on its own
    /// with the first untouched; each run stays within its blocks.
    #[test]
    fn a_tail_maps_apart_from_the_head() {
        let Some(device) = crate::detect_device().map(|d| d.device) else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };
        let residency = MetalResidencySet::new(&device);
        let mapper = KvMapper::new(&device).expect("mapper");
        // 1.5 tiles a block: three head blocks end mid-tile, so the tail starts at block 4.
        let big = 3 * 32 * 1024;
        let mut stores =
            SparseStores::with_tail(&device, &residency, 3, 2, &[(big, false)]).expect("stores");
        assert_eq!(stores.tail_start(), Some(4));
        let host = shared_zeroed(&device, big);
        let a = stores.buffer(0);
        let gpu = |body: &dyn Fn(&ProtocolObject<dyn MTL4ComputeCommandEncoder>)| {
            let batch = Mtl4DispatchBatch::begin(&device).expect("MTL4 queue");
            batch.declare(&residency);
            let _host = batch.residency().pin(host.clone());
            body(batch.encoder());
            batch.commit(true);
        };
        assert!(stores.grow_tail_to(1, &mapper).expect("tail block 0"));
        assert_eq!(
            stores.mapped().load(Ordering::Relaxed),
            0,
            "the head maps nothing"
        );
        // SAFETY (both): tail block 0 is mapped and resident, and `host` is shared.
        gpu(&|e| unsafe { e.fillBuffer_range_value(&a, NSRange::new(4 * big, big), 0xcd) });
        gpu(&|e| unsafe {
            e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &a,
                4 * big,
                &host,
                0,
                big,
            )
        });
        let got: Vec<u8> = read_slice(&host, big);
        assert!(
            got.iter().all(|&x| x == 0xcd),
            "the tail block holds its writes"
        );
        assert!(stores.grow_to(3, &mapper).expect("the head"));
        assert!(
            stores.grow_to(4, &mapper).is_err(),
            "past the head's blocks"
        );
        assert!(
            stores.grow_tail_to(3, &mapper).is_err(),
            "past the tail's blocks"
        );
        assert!(stores.grow_tail_to(2, &mapper).expect("the tail's rest"));
        assert!(!stores.grow_tail_to(1, &mapper).expect("already mapped"));
    }
}

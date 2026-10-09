// SPDX-License-Identifier: Apache-2.0
//
//! Per-`(tensor, K/V)` storage for the reactive (chunked) metal KV pool: ONE placement-sparse
//! MTLBuffer spanning the pool's capacity — one contiguous range, so readers address any block off
//! its base — whose pages are mapped in only as the pool grows ([`SparseKvLayer::commit_next`]).
//! A sparse buffer in the wired residency set takes no memory, where a plain capacity-sized one is
//! committed in full when made resident (gemma-4-26b on a 64 GB Mac: ~19 GiB of KV at startup).
//! Mappings are queued on the [`KvMapper`]; its owner waits them out before a forward reads them.

use std::ptr::NonNull;
use std::sync::{Arc, atomic::AtomicU64, atomic::Ordering};

use objc2::{Message, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSRange;
use objc2_metal::{
    MTL4CommandQueue, MTL4UpdateSparseBufferMappingOperation, MTLDevice, MTLHeap,
    MTLHeapDescriptor, MTLHeapType, MTLResourceOptions, MTLSharedEvent, MTLSparsePageSize,
    MTLSparseTextureMappingMode, MTLStorageMode,
};

use crate::metal_mem::MetalMem;
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

/// One per `(tensor, K/V)`, indexed `tensor * 2 + kv_idx` — the order `KvCachePool`'s
/// `new_metal_chunked` / `grow_to_cover` call their chunk allocator in.
pub struct SparseKvLayer {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    residency: MetalResidencySet,
    mapper: Arc<KvMapper>,
    buffer: Pinned,
    chunk_bytes: usize,
    max_chunks: usize,
    /// `heaps[i]` backs chunk `i`, pinned for the layer's life; the first `committed` are in use.
    heaps: Vec<Pinned<ProtocolObject<dyn MTLHeap>>>,
    committed: usize,
}

impl SparseKvLayer {
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        residency: &MetalResidencySet,
        mapper: &Arc<KvMapper>,
        chunk_bytes: usize,
        max_chunks: usize,
    ) -> Result<Self, String> {
        let tile_bytes = device.sparseTileSizeInBytesForSparsePageSize(PAGE);
        if max_chunks == 0 || chunk_bytes == 0 || !chunk_bytes.is_multiple_of(tile_bytes) {
            return Err(format!(
                "SparseKvLayer: {max_chunks} chunks of {chunk_bytes} bytes, not whole {tile_bytes}-byte tiles"
            ));
        }
        let len = chunk_bytes * max_chunks;
        // SAFETY: a fresh allocation; nothing reads it until a chunk is mapped and waited out.
        let buffer = unsafe {
            let private = MTLResourceOptions::StorageModePrivate;
            device.newBufferWithLength_options_placementSparsePageSize(len, private, PAGE)
        };
        let buffer = buffer.ok_or_else(|| format!("sparse buffer of {len} bytes returned nil"))?;
        Ok(Self {
            device: device.retain(),
            residency: residency.clone(),
            mapper: mapper.clone(),
            buffer: residency.pin(buffer),
            chunk_bytes,
            max_chunks,
            heaps: Vec::new(),
            committed: 0,
        })
    }

    /// The pool's next chunk, `bytes` of it: one a shrink released, else a new heap's pages mapped
    /// into the chunk's range (only queued — [`KvMapper::wait`] before a forward reads it).
    pub fn commit_next(&mut self, bytes: usize) -> Result<MetalMem, String> {
        if self.committed == self.max_chunks {
            return Err(format!("all {} chunks committed", self.max_chunks));
        }
        let offset = self.committed * self.chunk_bytes;
        if self.committed == self.heaps.len() {
            let desc = MTLHeapDescriptor::new();
            desc.setType(MTLHeapType::Placement);
            desc.setStorageMode(MTLStorageMode::Private);
            desc.setSize(self.chunk_bytes);
            desc.setMaxCompatiblePlacementSparsePageSize(PAGE);
            let heap = self
                .device
                .newHeapWithDescriptor(&desc)
                .ok_or("placement heap: nil")?;
            let tile_bytes = self.device.sparseTileSizeInBytesForSparsePageSize(PAGE);
            let (tile, tiles) = (offset / tile_bytes, self.chunk_bytes / tile_bytes);
            let op = MTL4UpdateSparseBufferMappingOperation {
                mode: MTLSparseTextureMappingMode::Map,
                bufferRange: NSRange::new(tile, tiles),
                heapOffset: 0,
            };
            let queue = &self.mapper.queue;
            // SAFETY: `op` maps whole tiles of this buffer's range from a heap of exactly that size.
            unsafe {
                queue.updateBufferMappings_heap_operations_count(
                    &self.buffer,
                    Some(&heap),
                    NonNull::from(&op),
                    1,
                )
            };
            self.heaps.push(self.residency.pin(heap));
        }
        self.committed += 1;
        let buffer = (*self.buffer).clone();
        Ok(MetalMem::from_buffer_with_offset(buffer, offset, bytes))
    }

    /// Hand back every chunk past the first `keep`; their pages stay mapped (an MTL4 command buffer
    /// does not retain what it binds) and `commit_next` reuses them.
    pub fn shrink_to(&mut self, keep: usize) {
        self.committed = self.committed.min(keep);
    }
}

unsafe impl Send for SparseKvLayer {}
unsafe impl Sync for SparseKvLayer {}
unsafe impl Send for KvMapper {}
unsafe impl Sync for KvMapper {}

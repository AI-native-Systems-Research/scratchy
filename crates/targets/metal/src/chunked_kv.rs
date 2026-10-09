// SPDX-License-Identifier: Apache-2.0
//
//! Per-`(tensor, K/V)` chunk storage for the reactive (chunked) metal KV pool.
//!
//! One `ChunkedKvLayer` backs one KV tensor with ONE `StorageModePrivate` MTLBuffer per chunk,
//! allocated and pinned into the wired residency set only when the pool grows into it
//! ([`ChunkedKvLayer::commit_next`]), so physical memory follows the blocks in use rather than
//! the pool's capacity. A buffer in a wired residency set is committed in full the moment it is
//! made resident, whatever its storage mode: one capacity-sized buffer per tensor committed the
//! whole KV budget at startup (gemma-4-26b on a 64 GB Mac: ~19 GiB idle).
//!
//! The chunks are not contiguous, so the pool's readers resolve every block through its
//! chunk-address table (`KvAddressing::Chunked`).

use objc2::Message;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLResourceOptions};

use crate::metal_mem::MetalMem;
use crate::residency::{MetalResidencySet, Pinned};

pub type Result<T> = std::result::Result<T, String>;

/// One per `(tensor, K/V)`, indexed `tensor * 2 + kv_idx` — the order `KvCachePool`'s
/// `new_metal_chunked` / `grow_to_cover` call their chunk allocator in.
pub struct ChunkedKvLayer {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    residency: MetalResidencySet,
    chunk_bytes: usize,
    max_chunks: usize,
    /// Every chunk this layer ever allocated, pinned for its lifetime; the first `committed` are
    /// the pool's.
    chunks: Vec<Pinned>,
    committed: usize,
}

impl ChunkedKvLayer {
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        residency: &MetalResidencySet,
        chunk_bytes: usize,
        max_chunks: usize,
    ) -> Result<Self> {
        if max_chunks == 0 || chunk_bytes == 0 {
            return Err(format!(
                "ChunkedKvLayer::new: chunk_bytes={chunk_bytes}, max_chunks={max_chunks} must be > 0"
            ));
        }
        Ok(Self {
            device: device.retain(),
            residency: residency.clone(),
            chunk_bytes,
            max_chunks,
            chunks: Vec::new(),
            committed: 0,
        })
    }

    /// The pool's next chunk, `bytes` of it: one a shrink released if there is one, else a fresh
    /// buffer pinned into the residency set. Err past `max_chunks`.
    pub fn commit_next(&mut self, bytes: usize) -> Result<MetalMem> {
        if self.committed == self.max_chunks {
            return Err(format!("all {} chunks committed", self.max_chunks));
        }
        if self.committed == self.chunks.len() {
            let buf = self
                .device
                .newBufferWithLength_options(
                    self.chunk_bytes,
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or_else(|| {
                    format!(
                        "newBufferWithLength_options(StorageModePrivate, len={}) returned nil",
                        self.chunk_bytes
                    )
                })?;
            self.chunks.push(self.residency.pin(buf));
        }
        let chunk = (*self.chunks[self.committed]).clone();
        self.committed += 1;
        Ok(MetalMem::from_buffer_with_offset(chunk, 0, bytes))
    }

    /// Hand back every chunk past the first `keep`. Their buffers stay allocated and pinned — an
    /// MTL4 command buffer does not retain what it binds, so freeing one a queued forward still
    /// reads would fault the GPU — and `commit_next` reuses them.
    pub fn shrink_to(&mut self, keep: usize) {
        self.committed = self.committed.min(keep);
    }

    pub fn committed_chunks(&self) -> usize {
        self.committed
    }
}

unsafe impl Send for ChunkedKvLayer {}
unsafe impl Sync for ChunkedKvLayer {}

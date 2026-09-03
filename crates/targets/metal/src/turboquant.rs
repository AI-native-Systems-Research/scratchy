// SPDX-License-Identifier: Apache-2.0
//! Dispatchers for the TurboQuant fused Metal kernels (`turboquant.metal`,
//! ported from arozanov's `turboquant_mlx/metal.py`): `tq_fused_quantize`
//! (vectors -> packed codes + norms) and `tq_dequant_fp16` (packed -> fp16
//! buffer). The dequant kernel is what the cache calls to fill the
//! dequant-to-buffer / incremental decode buffer.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize};

use crate::argmax::{Buffer, Device};
use crate::mtl4_dispatch::Mtl4DispatchBatch;
use crate::shader_cache::load_library_from_bytes;
use crate::stream::MetalStreamError;

/// TurboQuant codebook bit-width. 3 by default (arozanov's setting, ~4.7x); a
/// higher value trades compression for fidelity — needed for arches with
pub use crate::tape::quantized::tq_bits;

/// Build the TurboQuant provisioning for one worker at the GLOBAL (full-context)
/// geometry. `num_blocks` is the shared KV pool capacity (the packed store is the
/// canonical cache of that size); the scratch is ONE layer's fp16 (num_blocks
/// blocks) reused across the global layers.
///
/// `(block_size, num_kv_heads, head_dim)` are the GLOBAL group's geometry —
/// uniform arches pass their base geometry (every layer is "global"), gemma4
/// passes `(GLOBAL_BLOCK_SIZE, NUM_GLOBAL_KV_HEADS, GLOBAL_HEAD_DIM)`. The
/// codebook is built at `head_dim` (512 for gemma4 global, 128 for Llama).
///
/// `is_global[L]` flags whether layer L is in the GLOBAL group (group 0). Only
/// global layers get a real packed/norms store; sliding layers get a tiny
/// placeholder buffer that the tape never binds (the inject_tq pass only wraps
/// the global KV-writer). Uniform arches pass all-true → every layer real
/// (byte-identical to before, when `is_global` was implicitly all-true).
#[allow(clippy::too_many_arguments)]
pub fn build_tq_provision(
    device: &Device,
    is_global: &[bool],
    num_blocks: usize,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    blocks_per_chunk: usize,
    bits: u32,
    seed: u64,
) -> crate::interpreter::metal::runtime::TqRuntimeBuffers {
    use objc2_metal::{MTLBuffer, MTLResourceOptions};
    use scratchy_layers::turboquant::{PolarQuantizer, packed_dim};
    let num_layers = is_global.len();
    let n_global = is_global.iter().filter(|&&g| g).count();
    let pdim = packed_dim(head_dim, bits);
    tracing::info!(
        "TurboQuant KV: auto-selected {bits}-bit codebook (head_dim={head_dim}, \
         num_kv_heads={num_kv_heads}, block_size={block_size}, packed_dim={pdim}, \
         {num_blocks} blocks, {n_global}/{num_layers} global layers compressed)"
    );
    let q = PolarQuantizer::new(head_dim, bits, seed);
    let boundaries: Vec<f32> = q
        .centroids()
        .windows(2)
        .map(|w| (w[0] + w[1]) / 2.0)
        .collect();

    let alloc = |bytes: usize| {
        // NB: do NOT touch the pages here. The packed/norms/scratch are sized to
        // the full pool capacity (num_blocks) but only the ACTIVE context is ever
        // read/written, so leaving them untouched lets macOS lazily page them in
        // — RSS grows with the real context, not the capacity. That laziness IS
        // the memory win (a full eager zero-fill committed all ~3.85 GB upfront).
        // The prefill's empty-context dequant reads untouched pages, which the OS
        // zero-fills on first access (=> norm 0 => scratch 0, overwritten by rope).
        device
            .newBufferWithLength_options(bytes.max(16), MTLResourceOptions::StorageModeShared)
            .expect("tq alloc")
    };
    // Per-layer packed store + norms (canonical cache), at the GLOBAL geometry.
    // The TqPackedK{layer} binding is layer-indexed, so a global layer L's packed
    // store must sit at packed_k[L]; sliding layers (is_global[L]==false) get a
    // 16-byte placeholder that the tape never binds.
    let per_tok_words = num_kv_heads * pdim;
    let packed_bytes = num_blocks * block_size * per_tok_words * 4;
    let norms_bytes = num_blocks * block_size * num_kv_heads * 4;

    // Per-layer anonymous packed/norms buffers (one per global layer; sliding
    // layers get a 16-byte placeholder the tape never binds).
    let (packed_k, packed_v, norms_k, norms_v) = {
        // `zero_anon` zero-fills NORMS: the per-layer dequant runs BEFORE rope
        // every forward over the WHOLE active context, so a not-yet-quantized
        // block's norm slot is otherwise read UNINITIALIZED; `newBufferWithLength`
        // doesn't zero, and `K = centroid*scale*sign*norm` overflows to Inf -> NaN
        // logits -> `!!!!` on a continuation prefill. A zero norm dequants to 0,
        // which rope overwrites with the real new-token K. Packed codes need no
        // zeroing (a garbage code indexes a bounded centroid; product with norm 0
        // is 0), and zeroing the large packed buffers would defeat the lazy-page
        // memory win.
        let mk_region = |bytes: usize, zero_anon: bool| -> Vec<Buffer> {
            is_global
                .iter()
                .map(|&g| {
                    if !g {
                        return alloc(16);
                    }
                    let buf = alloc(bytes);
                    if zero_anon {
                        unsafe {
                            std::ptr::write_bytes(
                                buf.contents().as_ptr() as *mut u8,
                                0,
                                bytes.max(16),
                            );
                        }
                    }
                    buf
                })
                .collect()
        };
        (
            mk_region(packed_bytes, false),
            mk_region(packed_bytes, false),
            mk_region(norms_bytes, true),
            mk_region(norms_bytes, true),
        )
    };

    // Per-worker fp16 scratch: one layer (num_blocks blocks) at the GLOBAL
    // geometry, reused across all global layers (sequential, like the uniform
    // case — a dedicated scratch, not a pool-shared tensor). Contiguous data +
    // a chunk-table of gpuAddresses at chunk offsets (the kernels deref it).
    let per_block_elems = block_size * num_kv_heads * head_dim;
    let scratch_bytes = num_blocks * per_block_elems * 2;
    let n_chunks = num_blocks.div_ceil(blocks_per_chunk.max(1));
    let chunk_bytes = blocks_per_chunk * per_block_elems * 2;
    let build_scratch = || {
        let data = alloc(scratch_bytes);
        // Zero the scratch so blocks the rope DOESN'T write read back as 0
        // (finite) rather than uninitialized garbage (Inf/NaN) — the paged
        // attention can stage blocks just past the active context.
        unsafe {
            std::ptr::write_bytes(
                data.contents().as_ptr() as *mut u8,
                0,
                scratch_bytes.max(16),
            );
        }
        let base = data.gpuAddress();
        let table_vals: Vec<u64> = (0..n_chunks)
            .map(|c| base + (c * chunk_bytes) as u64)
            .collect();
        let table = crate::argmax::upload_shared_buffer(device, &table_vals);
        (table, data)
    };
    let (scratch_k_table, scratch_k_data) = build_scratch();
    let (scratch_v_table, scratch_v_data) = build_scratch();

    crate::interpreter::metal::runtime::TqRuntimeBuffers {
        packed_k,
        packed_v,
        norms_k,
        norms_v,
        signs: crate::argmax::upload_shared_buffer(device, q.signs()),
        boundaries: crate::argmax::upload_shared_buffer(device, &boundaries),
        centroids: crate::argmax::upload_shared_buffer(device, q.centroids()),
        scratch_k_table,
        scratch_v_table,
        scratch_k_data,
        scratch_v_data,
    }
}

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

pub struct TurboQuantKernels {
    pub quantize: Pipeline,
    pub dequant_fp16: Pipeline,
    pub compress_paged: Pipeline,
    pub compress_paged_bf16: Pipeline,
    pub dequant_paged: Pipeline,
    pub dequant_paged_bf16: Pipeline,
    _library: Library,
}

impl TurboQuantKernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library = load_library_from_bytes(device, crate::embedded_metallib!("turboquant"))
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("load turboquant.metallib: {e}"))
            })?;
        Ok(Self {
            quantize: build_pipeline(device, &library, "tq_fused_quantize")?,
            dequant_fp16: build_pipeline(device, &library, "tq_dequant_fp16")?,
            compress_paged: build_pipeline(device, &library, "tq_compress_paged")?,
            compress_paged_bf16: build_pipeline(device, &library, "tq_compress_paged_bf16")?,
            dequant_paged: build_pipeline(device, &library, "tq_dequant_paged")?,
            dequant_paged_bf16: build_pipeline(device, &library, "tq_dequant_paged_bf16")?,
            _library: library,
        })
    }
}

/// `tq_compress_paged`: in-place compress the new KV slots of one layer's paged
/// buffer — write packed codes+norms to the store, dequant back into the pool.
/// Grid = (n_slots, num_kv_heads).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_compress_paged(
    k: &TurboQuantKernels,
    device: &Device,
    chunk_table: &Buffer,
    chunk_buffers: &[&Buffer],
    slots: &Buffer,
    logical_slots: &Buffer,
    signs: &Buffer,
    boundaries: &Buffer,
    centroids: &Buffer,
    packed_store: &Buffer,
    norms_store: &Buffer,
    n_slots: u32,
    dim: u32,
    bits: u32,
    vals_per_word: u32,
    packed_dim: u32,
    n_centroids: u32,
    scale: f32,
    num_kv_heads: u32,
    block_size: u32,
    blocks_per_chunk: u32,
    bf16: bool,
) -> Result<(), MetalStreamError> {
    run(device, "tq_compress_paged", |batch| {
        encode_compress_paged(
            batch,
            k,
            chunk_table,
            chunk_buffers,
            slots,
            logical_slots,
            signs,
            boundaries,
            centroids,
            packed_store,
            norms_store,
            n_slots,
            dim,
            bits,
            vals_per_word,
            packed_dim,
            n_centroids,
            scale,
            num_kv_heads,
            block_size,
            blocks_per_chunk,
            bf16,
        );
    })
}

/// Encode a compress dispatch into an EXISTING MTL4 batch (no command buffer /
/// commit) — lets the worker batch all layers' K+V into one command buffer with
/// a single commit (the per-call commit+wait was a ~2x decode hit).
///
/// The chunk-data buffers are NOT bound in the argument table (the kernel
/// reaches the pool through the `gpuAddress`es stored in `chunk_table`); they
/// pass through `extra_resident` so MTL4 keeps them wired.
#[allow(clippy::too_many_arguments)]
pub fn encode_compress_paged(
    batch: &mut Mtl4DispatchBatch,
    k: &TurboQuantKernels,
    chunk_table: &Buffer,
    chunk_buffers: &[&Buffer],
    slots: &Buffer,
    logical_slots: &Buffer,
    signs: &Buffer,
    boundaries: &Buffer,
    centroids: &Buffer,
    packed_store: &Buffer,
    norms_store: &Buffer,
    n_slots: u32,
    dim: u32,
    bits: u32,
    vals_per_word: u32,
    packed_dim: u32,
    n_centroids: u32,
    scale: f32,
    num_kv_heads: u32,
    block_size: u32,
    blocks_per_chunk: u32,
    bf16: bool,
) {
    let pso = if bf16 {
        &k.compress_paged_bf16
    } else {
        &k.compress_paged
    };
    batch.encode(
        pso,
        &[
            (chunk_table, 0),
            (slots, 1),
            (signs, 2),
            (boundaries, 3),
            (centroids, 4),
            (packed_store, 5),
            (norms_store, 6),
            (logical_slots, 16),
        ],
        &[
            (dim, 7),
            (bits, 8),
            (vals_per_word, 9),
            (packed_dim, 10),
            (n_centroids, 11),
            (num_kv_heads, 13),
            (block_size, 14),
            (blocks_per_chunk, 15),
            (1, 17), // do_writeback: host/unit path validates the in-place dequant
        ],
        &[(scale, 12)],
        chunk_buffers,
        MTLSize {
            width: n_slots as usize,
            height: num_kv_heads as usize,
            depth: 1,
        },
        MTLSize {
            width: dim as usize,
            height: 1,
            depth: 1,
        },
    );
}

/// Run a batched compress: one MTL4 command buffer, one commit, NO host wait.
/// The `body` encodes all layers' K+V via `encode_compress_layer`. We do NOT
/// event-wait — the compress runs async and the next forward is on the SAME
/// device, so queue ordering already guarantees the next step reads the
/// compressed KV. Dropping the per-step host sync is what gets the decode
/// overhead from ~5% to <2% (the wait was pure CPU dead time). The batch's
/// scalar buffers / argument tables / residency set stay alive until GPU
/// completion via the async commit's feedback handler.
pub fn run_compress_batch(
    device: &Device,
    body: impl FnOnce(&mut Mtl4DispatchBatch),
) -> Result<(), MetalStreamError> {
    let Some(mut batch) = Mtl4DispatchBatch::begin(device) else {
        return Err(MetalStreamError::ShaderCompilationFailed(
            "tq_compress_batch: no MTL4 queue".into(),
        ));
    };
    body(&mut batch);
    batch.commit(false)
}

fn build_pipeline(
    device: &Device,
    library: &Library,
    name: &str,
) -> Result<Pipeline, MetalStreamError> {
    let f = library
        .newFunctionWithName(&NSString::from_str(name))
        .ok_or_else(|| MetalStreamError::ShaderCompilationFailed(format!("{name} fn missing")))?;
    device
        .newComputePipelineStateWithFunction_error(&f)
        .map_err(|e| MetalStreamError::ShaderCompilationFailed(format!("{name} pipeline: {e:?}")))
}

/// Drive ONE MTL4 dispatch (or several) synchronously: open a batch, let `body`
/// encode, then commit with an event-wait. The MTL4 replacement for the classic
/// command-buffer encode/commit/wait path.
fn run(
    device: &Device,
    label: &str,
    body: impl FnOnce(&mut Mtl4DispatchBatch),
) -> Result<(), MetalStreamError> {
    let Some(mut batch) = Mtl4DispatchBatch::begin(device) else {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "{label}: no MTL4 queue"
        )));
    };
    body(&mut batch);
    batch.commit(true)
}

/// `tq_fused_quantize`: `inp` [n_vecs, dim] f32 -> `packed_out` [n_vecs,
/// packed_dim] u32 + `norms_out` [n_vecs] f32.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_quantize(
    k: &TurboQuantKernels,
    device: &Device,
    inp: &Buffer,
    signs: &Buffer,
    boundaries: &Buffer,
    packed_out: &Buffer,
    norms_out: &Buffer,
    n_vecs: u32,
    dim: u32,
    bits: u32,
    vals_per_word: u32,
    packed_dim: u32,
    n_centroids: u32,
) -> Result<(), MetalStreamError> {
    run(device, "tq_fused_quantize", |batch| {
        batch.encode(
            &k.quantize,
            &[
                (inp, 0),
                (signs, 1),
                (boundaries, 2),
                (packed_out, 3),
                (norms_out, 4),
            ],
            &[
                (dim, 5),
                (bits, 6),
                (vals_per_word, 7),
                (packed_dim, 8),
                (n_centroids, 9),
            ],
            &[],
            &[],
            MTLSize {
                width: n_vecs as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: dim as usize,
                height: 1,
                depth: 1,
            },
        );
    })
}

/// `tq_dequant_fp16`: packed codes + norms -> `out` [n_vecs, dim] fp16.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_dequant_fp16(
    k: &TurboQuantKernels,
    device: &Device,
    packed: &Buffer,
    norms: &Buffer,
    centroids: &Buffer,
    signs: &Buffer,
    out: &Buffer,
    n_vecs: u32,
    dim: u32,
    bits: u32,
    vals_per_word: u32,
    packed_dim: u32,
    scale: f32,
) -> Result<(), MetalStreamError> {
    run(device, "tq_dequant_fp16", |batch| {
        batch.encode(
            &k.dequant_fp16,
            &[
                (packed, 0),
                (norms, 1),
                (centroids, 2),
                (signs, 3),
                (out, 4),
            ],
            &[(dim, 5), (bits, 6), (vals_per_word, 7), (packed_dim, 8)],
            &[(scale, 9)],
            &[],
            MTLSize {
                width: n_vecs as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: dim as usize,
                height: 1,
                depth: 1,
            },
        );
    })
}

/// `tq_dequant_paged`: dequant-on-read — fill a TARGET fp16 paged buffer from
/// the packed store for `n_slots` physical slots. Grid = (n_slots, num_kv_heads).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_dequant_paged(
    k: &TurboQuantKernels,
    device: &Device,
    chunk_table: &Buffer,
    chunk_buffers: &[&Buffer],
    slots: &Buffer,
    dst_slots: &Buffer,
    signs: &Buffer,
    centroids: &Buffer,
    packed_store: &Buffer,
    norms_store: &Buffer,
    n_slots: u32,
    dim: u32,
    bits: u32,
    vals_per_word: u32,
    packed_dim: u32,
    scale: f32,
    num_kv_heads: u32,
    block_size: u32,
    blocks_per_chunk: u32,
    bf16: bool,
) -> Result<(), MetalStreamError> {
    run(device, "tq_dequant_paged", |batch| {
        let pso = if bf16 {
            &k.dequant_paged_bf16
        } else {
            &k.dequant_paged
        };
        batch.encode(
            pso,
            &[
                (chunk_table, 0),
                (slots, 1),
                (signs, 2),
                (centroids, 3),
                (packed_store, 4),
                (norms_store, 5),
                (dst_slots, 14),
            ],
            &[
                (dim, 6),
                (bits, 7),
                (vals_per_word, 8),
                (packed_dim, 9),
                (num_kv_heads, 11),
                (block_size, 12),
                (blocks_per_chunk, 13),
            ],
            &[(scale, 10)],
            chunk_buffers,
            MTLSize {
                width: n_slots as usize,
                height: num_kv_heads as usize,
                depth: 1,
            },
            MTLSize {
                width: dim as usize,
                height: 1,
                depth: 1,
            },
        );
    })
}

/// Worker-side TurboQuant runtime: the kernels, the uploaded codebook/signs,
/// and a persistent per-layer PACKED STORE (the ~4.6x compressed cache, indexed
/// by physical slot). After the forward writes KV, the worker calls
/// `compress_layer`: it quantizes the new slots into the packed store and writes
/// the dequant back into the fp16 pool, so attention reads TurboQuant'd KV.
/// (Packed stores are StorageModeShared here — fine for a first version; move to
/// Private for the memory win + eviction follow-up.)
pub struct TurboQuantRuntime {
    kernels: TurboQuantKernels,
    signs: Buffer,
    boundaries: Buffer,
    centroids: Buffer,
    k_packed: Vec<Buffer>,
    k_norms: Vec<Buffer>,
    v_packed: Vec<Buffer>,
    v_norms: Vec<Buffer>,
    bits: u32,
    dim: u32,
    packed_dim: u32,
    vals_per_word: u32,
    n_centroids: u32,
    scale: f32,
    num_kv_heads: u32,
    block_size: u32,
    blocks_per_chunk: u32,
    /// true when the KV pool is bf16 (Llama/Qwen); picks the bf16 kernel.
    bf16: bool,
}

impl TurboQuantRuntime {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_blocks: usize,
        block_size: usize,
        blocks_per_chunk: usize,
        bits: u32,
        seed: u64,
        bf16: bool,
    ) -> Result<Self, MetalStreamError> {
        use scratchy_layers::turboquant::{PolarQuantizer, packed_dim, vals_per_word};
        let kernels = TurboQuantKernels::new(device)?;
        let q = PolarQuantizer::new(head_dim, bits, seed);
        let pdim = packed_dim(head_dim, bits);
        let boundaries: Vec<f32> = q
            .centroids()
            .windows(2)
            .map(|w| (w[0] + w[1]) / 2.0)
            .collect();
        let max_tokens = num_blocks * block_size;
        let store_words = (max_tokens * num_kv_heads * pdim).max(1);
        let store_norms = (max_tokens * num_kv_heads).max(1);
        let mk_p = || crate::argmax::upload_shared_buffer(device, &vec![0u32; store_words]);
        let mk_n = || crate::argmax::upload_shared_buffer(device, &vec![0.0f32; store_norms]);
        Ok(Self {
            kernels,
            signs: crate::argmax::upload_shared_buffer(device, q.signs()),
            boundaries: crate::argmax::upload_shared_buffer(device, &boundaries),
            centroids: crate::argmax::upload_shared_buffer(device, q.centroids()),
            k_packed: (0..num_layers).map(|_| mk_p()).collect(),
            k_norms: (0..num_layers).map(|_| mk_n()).collect(),
            v_packed: (0..num_layers).map(|_| mk_p()).collect(),
            v_norms: (0..num_layers).map(|_| mk_n()).collect(),
            bits,
            dim: head_dim as u32,
            packed_dim: pdim as u32,
            vals_per_word: vals_per_word(bits) as u32,
            n_centroids: q.centroids().len() as u32,
            scale: q.scale(),
            num_kv_heads: num_kv_heads as u32,
            block_size: block_size as u32,
            blocks_per_chunk: blocks_per_chunk as u32,
            bf16,
        })
    }

    pub fn num_layers(&self) -> usize {
        self.k_packed.len()
    }

    /// Bytes per cached KV token (K+V) in the packed store: per kv_head,
    /// `packed_dim` u32 code words + one f32 norm, for K and V.
    pub fn packed_bytes_per_token(&self) -> usize {
        2 * self.num_kv_heads as usize * (self.packed_dim as usize * 4 + 4)
    }

    /// Bytes per cached KV token (K+V) in an equivalent fp16 cache.
    pub fn fp16_bytes_per_token(&self) -> usize {
        2 * self.num_kv_heads as usize * self.dim as usize * 2
    }

    /// Realized compression ratio of the packed store vs fp16 (the arozanov
    /// `cache.nbytes` metric). ~4.0x at head_dim 64, ~4.6x at head_dim 128 —
    /// the f32 norm is a fixed per-vector tax that amortizes as head_dim grows.
    pub fn compression_ratio(&self) -> f32 {
        self.fp16_bytes_per_token() as f32 / self.packed_bytes_per_token().max(1) as f32
    }

    /// Compress one layer's newly-written KV slots in place (K then V):
    /// quantize -> packed store, dequant back into the fp16 pool.
    #[allow(clippy::too_many_arguments)]
    pub fn compress_layer(
        &self,
        device: &Device,
        layer: usize,
        k_chunk_table: &Buffer,
        k_chunk_bufs: &[&Buffer],
        v_chunk_table: &Buffer,
        v_chunk_bufs: &[&Buffer],
        slots: &Buffer,
        logical_slots: &Buffer,
        n_slots: u32,
    ) -> Result<(), MetalStreamError> {
        if n_slots == 0 {
            return Ok(());
        }
        dispatch_compress_paged(
            &self.kernels,
            device,
            k_chunk_table,
            k_chunk_bufs,
            slots,
            logical_slots,
            &self.signs,
            &self.boundaries,
            &self.centroids,
            &self.k_packed[layer],
            &self.k_norms[layer],
            n_slots,
            self.dim,
            self.bits,
            self.vals_per_word,
            self.packed_dim,
            self.n_centroids,
            self.scale,
            self.num_kv_heads,
            self.block_size,
            self.blocks_per_chunk,
            self.bf16,
        )?;
        dispatch_compress_paged(
            &self.kernels,
            device,
            v_chunk_table,
            v_chunk_bufs,
            slots,
            logical_slots,
            &self.signs,
            &self.boundaries,
            &self.centroids,
            &self.v_packed[layer],
            &self.v_norms[layer],
            n_slots,
            self.dim,
            self.bits,
            self.vals_per_word,
            self.packed_dim,
            self.n_centroids,
            self.scale,
            self.num_kv_heads,
            self.block_size,
            self.blocks_per_chunk,
            self.bf16,
        )?;
        Ok(())
    }

    /// Encode one layer's K+V compress into an EXISTING encoder (no command
    /// buffer) — the batched counterpart of `compress_layer`. The worker wraps a
    /// loop over all layers in one `run_compress_batch` (one commit + host wait)
    /// instead of 2·num_layers separate command buffers (the ~2x decode hit).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_compress_layer(
        &self,
        batch: &mut Mtl4DispatchBatch,
        layer: usize,
        k_chunk_table: &Buffer,
        k_chunk_bufs: &[&Buffer],
        v_chunk_table: &Buffer,
        v_chunk_bufs: &[&Buffer],
        slots: &Buffer,
        logical_slots: &Buffer,
        n_slots: u32,
    ) {
        if n_slots == 0 {
            return;
        }
        encode_compress_paged(
            batch,
            &self.kernels,
            k_chunk_table,
            k_chunk_bufs,
            slots,
            logical_slots,
            &self.signs,
            &self.boundaries,
            &self.centroids,
            &self.k_packed[layer],
            &self.k_norms[layer],
            n_slots,
            self.dim,
            self.bits,
            self.vals_per_word,
            self.packed_dim,
            self.n_centroids,
            self.scale,
            self.num_kv_heads,
            self.block_size,
            self.blocks_per_chunk,
            self.bf16,
        );
        encode_compress_paged(
            batch,
            &self.kernels,
            v_chunk_table,
            v_chunk_bufs,
            slots,
            logical_slots,
            &self.signs,
            &self.boundaries,
            &self.centroids,
            &self.v_packed[layer],
            &self.v_norms[layer],
            n_slots,
            self.dim,
            self.bits,
            self.vals_per_word,
            self.packed_dim,
            self.n_centroids,
            self.scale,
            self.num_kv_heads,
            self.block_size,
            self.blocks_per_chunk,
            self.bf16,
        );
    }

    /// Dequant `n_slots` logical blocks' worth of slots from the packed store
    /// into the window pool at `dst_slots` (the WindowMap-assigned slots). Used
    /// to fill misses before attention. `src_logical_slots` index the packed
    /// store; `dst_slots` address the window pool.
    #[allow(clippy::too_many_arguments)]
    pub fn dequant_into_window(
        &self,
        device: &Device,
        layer: usize,
        k_chunk_table: &Buffer,
        k_chunk_bufs: &[&Buffer],
        v_chunk_table: &Buffer,
        v_chunk_bufs: &[&Buffer],
        src_logical_slots: &Buffer,
        dst_slots: &Buffer,
        n_slots: u32,
    ) -> Result<(), MetalStreamError> {
        if n_slots == 0 {
            return Ok(());
        }
        dispatch_dequant_paged(
            &self.kernels,
            device,
            k_chunk_table,
            k_chunk_bufs,
            src_logical_slots,
            dst_slots,
            &self.signs,
            &self.centroids,
            &self.k_packed[layer],
            &self.k_norms[layer],
            n_slots,
            self.dim,
            self.bits,
            self.vals_per_word,
            self.packed_dim,
            self.scale,
            self.num_kv_heads,
            self.block_size,
            self.blocks_per_chunk,
            self.bf16,
        )?;
        dispatch_dequant_paged(
            &self.kernels,
            device,
            v_chunk_table,
            v_chunk_bufs,
            src_logical_slots,
            dst_slots,
            &self.signs,
            &self.centroids,
            &self.v_packed[layer],
            &self.v_norms[layer],
            n_slots,
            self.dim,
            self.bits,
            self.vals_per_word,
            self.packed_dim,
            self.scale,
            self.num_kv_heads,
            self.block_size,
            self.blocks_per_chunk,
            self.bf16,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::argmax::upload_shared_buffer;
    use objc2_metal::MTLBuffer;
    use scratchy_layers::turboquant::{PolarQuantizer, packed_dim, vals_per_word};

    #[test]
    fn gpu_turboquant_matches_host() {
        let Some(device) = crate::detect_device().map(|d| d.device) else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };
        let kernels = TurboQuantKernels::new(&device).expect("kernels");

        // Every (head_dim, bits) the metal path actually provisions —
        // NOT just (128, 3). `qwen2.5-0.5b` (head_dim 64) decodes
        // garbage under TurboQuant while `llama-3.2-1b` (also 64, but
        // 3-bit) is clean, and the host-side codebook is FINE at 64
        // (mean cosine 0.9849 / 0.9958, better than at 128). So the
        // untested combination is the suspect, and this loop is what
        // makes it a test rather than an argument.
        for (dim, bits) in [(64usize, 3u32), (64, 4), (128, 3), (128, 4), (256, 4)] {
            gpu_matches_host_at(&device, &kernels, dim, bits, 42);
        }
    }

    fn gpu_matches_host_at(
        device: &Device,
        kernels: &TurboQuantKernels,
        dim: usize,
        bits: u32,
        seed: u64,
    ) {
        {
            let q = PolarQuantizer::new(dim, bits, seed);
            let pdim = packed_dim(dim, bits);
            let vpw = vals_per_word(bits) as u32;
            let n_cent = q.centroids().len();
            let boundaries: Vec<f32> = q
                .centroids()
                .windows(2)
                .map(|w| (w[0] + w[1]) / 2.0)
                .collect();

            // Deterministic vectors.
            let n_vecs = 256usize;
            let mut s = 0xC0FFEEu64;
            let mut rnd = || {
                let mut a = 0.0f32;
                for _ in 0..3 {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    a += ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
                }
                a
            };
            let vecs: Vec<f32> = (0..n_vecs * dim).map(|_| rnd()).collect();

            // Upload + allocate.
            let inp = upload_shared_buffer(device, &vecs);
            let signs_buf = upload_shared_buffer(device, q.signs());
            let bound_buf = upload_shared_buffer(device, &boundaries);
            let cent_buf = upload_shared_buffer(device, q.centroids());
            let packed = upload_shared_buffer(device, &vec![0u32; n_vecs * pdim]);
            let norms = upload_shared_buffer(device, &vec![0.0f32; n_vecs]);
            let out = upload_shared_buffer(device, &vec![0u16; n_vecs * dim]); // fp16 bits

            dispatch_fused_quantize(
                kernels,
                device,
                &inp,
                &signs_buf,
                &bound_buf,
                &packed,
                &norms,
                n_vecs as u32,
                dim as u32,
                bits,
                vpw,
                pdim as u32,
                n_cent as u32,
            )
            .unwrap();
            dispatch_dequant_fp16(
                kernels,
                device,
                &packed,
                &norms,
                &cent_buf,
                &signs_buf,
                &out,
                n_vecs as u32,
                dim as u32,
                bits,
                vpw,
                pdim as u32,
                q.scale(),
            )
            .unwrap();

            // Readback.
            let gpu_packed: &[u32] = unsafe {
                std::slice::from_raw_parts(packed.contents().as_ptr() as *const u32, n_vecs * pdim)
            };
            let gpu_out_bits: &[u16] = unsafe {
                std::slice::from_raw_parts(out.contents().as_ptr() as *const u16, n_vecs * dim)
            };

            let mut min_cos = f32::MAX;
            let mut idx_mismatch = 0usize;
            for v in 0..n_vecs {
                let orig = &vecs[v * dim..(v + 1) * dim];
                // host reference
                let (host_idx, host_norm) = q.quantize(orig);
                let host_packed = scratchy_layers::turboquant::pack_indices(&host_idx, bits);
                // GPU indices == host indices
                if gpu_packed[v * pdim..(v + 1) * pdim] != host_packed[..] {
                    idx_mismatch += 1;
                }
                // GPU fp16 dequant vs host f32 round-trip
                let host_recon = q.dequantize(&host_idx, host_norm);
                let gpu_recon: Vec<f32> = (0..dim)
                    .map(|i| half::f16::from_bits(gpu_out_bits[v * dim + i]).to_f32())
                    .collect();
                let dot: f64 = host_recon
                    .iter()
                    .zip(&gpu_recon)
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum();
                let na: f64 = host_recon
                    .iter()
                    .map(|&x| (x as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let nb: f64 = gpu_recon
                    .iter()
                    .map(|&x| (x as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                min_cos = min_cos.min((dot / (na * nb).max(1e-12)) as f32);
            }
            println!(
                "GPU turboquant dim {dim} bits {bits}: min cosine {min_cos:.5}, \
             packed-index mismatches {idx_mismatch}/{n_vecs}"
            );
            assert_eq!(
                idx_mismatch, 0,
                "dim {dim} bits {bits}: GPU packed codes must equal host codes"
            );
            assert!(
                min_cos > 0.99,
                "dim {dim} bits {bits}: GPU dequant must match host (fp16 tol): {min_cos}"
            );
        }
    }

    /// PAGED round-trip at the geometry that actually ships — the flat
    /// `gpu_turboquant_matches_host` test passes bit-exact and still
    /// missed a real bug, because it never touches the paged layout
    /// (`[num_blocks, num_kv_heads, BLOCK_SIZE, head_dim]`, grid
    /// `(n_slots, num_kv_heads)`, chunk tables, and the
    /// `build_tq_provision` strides).
    ///
    /// `qwen2.5-0.5b` (head_dim 64, num_kv_heads 2) decodes garbage
    /// whenever it runs its own PREFILL under TurboQuant and is clean
    /// with `--kv-cache-dtype fp16`; `num_kv_heads = 2` is the smallest
    /// of any model exercised. This drives compress -> dequant over the
    /// paged store and checks the values survive.
    #[test]
    fn gpu_turboquant_paged_roundtrip() {
        let Some(device) = crate::detect_device().map(|d| d.device) else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };
        let kernels = TurboQuantKernels::new(&device).expect("kernels");
        for (dim, num_kv_heads) in [(64usize, 2usize), (64, 8), (128, 2), (128, 8)] {
            paged_roundtrip_at(&device, &kernels, dim, num_kv_heads);
        }
        RESULTS.with(|r| {
            let rows = r.borrow();
            for (d, kv, z, n, c) in rows.iter() {
                println!("  dim {d:<4} kv {kv:<2} zero {z:>4}/{n:<4} min_cos {c:.4}");
            }
            let bad: Vec<_> = rows.iter().filter(|(_, _, z, _, _)| *z > 0).collect();
            assert!(
                bad.is_empty(),
                "paged round-trip dropped vectors at: {:?}",
                bad.iter()
                    .map(|(d, kv, z, n, _)| (d, kv, z, n))
                    .collect::<Vec<_>>()
            );
        });
    }

    thread_local! {
        static RESULTS: std::cell::RefCell<Vec<(usize, usize, usize, usize, f32)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    fn paged_roundtrip_at(
        device: &Device,
        kernels: &TurboQuantKernels,
        dim: usize,
        num_kv_heads: usize,
    ) {
        use scratchy_layers::turboquant::{PolarQuantizer, packed_dim};
        let (bits, seed) = (4u32, 42u64);
        let block_size = 16usize;
        let blocks_per_chunk = 4usize;
        let n_blocks = 8usize;
        let q = PolarQuantizer::new(dim, bits, seed);
        let pdim = packed_dim(dim, bits);
        let vpw = vals_per_word(bits) as u32;
        let boundaries: Vec<f32> = q
            .centroids()
            .windows(2)
            .map(|w| (w[0] + w[1]) / 2.0)
            .collect();

        // Source K in the paged scratch layout the kernels address:
        // [n_blocks, num_kv_heads, block_size, dim].
        let per_block = block_size * num_kv_heads * dim;
        let mut st = 0xBEEFu64;
        let mut rnd = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((st >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        let src: Vec<f32> = (0..n_blocks * per_block).map(|_| rnd()).collect();
        let src_h: Vec<u16> = src
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();

        // Chunked scratch + gpuAddress table, exactly as `build_tq_provision`.
        let chunk_bytes = blocks_per_chunk * per_block * 2;
        let n_chunks = n_blocks.div_ceil(blocks_per_chunk);
        let scratch = upload_shared_buffer(device, &src_h);
        let base = scratch.gpuAddress();
        let table_vals: Vec<u64> = (0..n_chunks)
            .map(|c| base + (c * chunk_bytes) as u64)
            .collect();
        let chunk_table = upload_shared_buffer(device, &table_vals);

        // `slots` are TOKEN slots, not block slots: the dispatch grid is
        // (n_slots, num_kv_heads) and covers one VECTOR per (slot, head).
        // Passing n_blocks here processed only 8*kv of the 128*kv vectors
        // and looked like the store "dropping" 15/16 — a harness error,
        // not a kernel one.
        let n_slots = n_blocks * block_size;
        let slots: Vec<u32> = (0..n_slots as u32).collect();
        let slots_buf = upload_shared_buffer(device, &slots);
        let logical_buf = upload_shared_buffer(device, &slots);
        let signs_buf = upload_shared_buffer(device, q.signs());
        let bound_buf = upload_shared_buffer(device, &boundaries);
        let cent_buf = upload_shared_buffer(device, q.centroids());
        let n_tok = n_blocks * block_size;
        let packed = upload_shared_buffer(device, &vec![0u32; n_tok * num_kv_heads * pdim]);
        let norms = upload_shared_buffer(device, &vec![0.0f32; n_tok * num_kv_heads]);

        dispatch_compress_paged(
            kernels,
            device,
            &chunk_table,
            &[&scratch],
            &slots_buf,
            &logical_buf,
            &signs_buf,
            &bound_buf,
            &cent_buf,
            &packed,
            &norms,
            n_slots as u32,
            dim as u32,
            bits,
            vpw,
            pdim as u32,
            q.centroids().len() as u32,
            q.scale(),
            num_kv_heads as u32,
            block_size as u32,
            blocks_per_chunk as u32,
            false,
        )
        .unwrap();

        // Dequant back into a ZEROED scratch, then compare.
        let dst_h = vec![0u16; src_h.len()];
        let dst = upload_shared_buffer(device, &dst_h);
        let dbase = dst.gpuAddress();
        let dtable_vals: Vec<u64> = (0..n_chunks)
            .map(|c| dbase + (c * chunk_bytes) as u64)
            .collect();
        let dchunk_table = upload_shared_buffer(device, &dtable_vals);

        dispatch_dequant_paged(
            kernels,
            device,
            &dchunk_table,
            &[&dst],
            &slots_buf,
            &slots_buf,
            &signs_buf,
            &cent_buf,
            &packed,
            &norms,
            n_slots as u32,
            dim as u32,
            bits,
            vpw,
            pdim as u32,
            q.scale(),
            num_kv_heads as u32,
            block_size as u32,
            blocks_per_chunk as u32,
            false,
        )
        .unwrap();

        let got: &[u16] = unsafe {
            std::slice::from_raw_parts(dst.contents().as_ptr() as *const u16, src_h.len())
        };
        // Per-vector cosine: quantization is lossy, so compare direction.
        let n_vec = n_blocks * num_kv_heads * block_size;
        let mut min_cos = f32::MAX;
        let mut zero_vecs = 0usize;
        for v in 0..n_vec {
            let a = &src[v * dim..(v + 1) * dim];
            let b: Vec<f32> = (0..dim)
                .map(|i| half::f16::from_bits(got[v * dim + i]).to_f32())
                .collect();
            let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
            if nb == 0.0 {
                zero_vecs += 1;
                continue;
            }
            let dot: f64 = a.iter().zip(&b).map(|(&x, &y)| x as f64 * y as f64).sum();
            let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
            min_cos = min_cos.min((dot / (na * nb).max(1e-12)) as f32);
        }
        println!(
            "paged roundtrip dim {dim} kv_heads {num_kv_heads}: min cosine {min_cos:.4}, \
             all-zero vectors {zero_vecs}/{n_vec}"
        );
        // Report every geometry before asserting, so the PATTERN is
        // visible — a failure at one shape is a bug, a failure at all
        // shapes means this harness drives the dispatch wrong.
        RESULTS.with(|r| {
            r.borrow_mut()
                .push((dim, num_kv_heads, zero_vecs, n_vec, min_cos))
        });
    }

    /// Measures the overhead of arozanov's dequant-to-buffer mechanism: the
    /// quantize-on-write and the dequant passes (the attention itself is
    /// unchanged fp16 on the buffer). Decode adds one incremental dequant (the
    /// new K + V token) per step; prefill adds quantize(L) + dequant(L) once.
    /// Run: `cargo test -p scratchy-target-metal turboquant_overhead -- --nocapture --ignored`
    #[test]
    #[ignore = "GPU microbenchmark — run explicitly"]
    fn turboquant_overhead() {
        use std::time::Instant;
        let Some(device) = crate::detect_device().map(|d| d.device) else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };
        let k = TurboQuantKernels::new(&device).expect("kernels");
        let (dim, bits, seed) = (128usize, 3u32, 42u64);
        let q = PolarQuantizer::new(dim, bits, seed);
        let pdim = packed_dim(dim, bits);
        let vpw = vals_per_word(bits) as u32;
        let n_cent = q.centroids().len() as u32;
        let boundaries: Vec<f32> = q
            .centroids()
            .windows(2)
            .map(|w| (w[0] + w[1]) / 2.0)
            .collect();

        let signs = upload_shared_buffer(&device, q.signs());
        let bnd = upload_shared_buffer(&device, &boundaries);
        let cent = upload_shared_buffer(&device, q.centroids());

        let median = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let time = |iters: usize, f: &dyn Fn()| -> f64 {
            f(); // warm
            let samples: Vec<f64> = (0..iters)
                .map(|_| {
                    let t = Instant::now();
                    f();
                    t.elapsed().as_secs_f64() * 1e6 // us
                })
                .collect();
            median(samples)
        };

        println!("\n=== TurboQuant dequant-to-buffer kernel cost (M5, dim {dim}, 3-bit) ===");
        // Baseline: an isolated dispatch's submit+host-wait round-trip. EVERY
        // measurement below includes this; subtract it to isolate compute. (In
        // the real forward these kernels are encoded into the forward command
        // buffer with ONE submit per step, so this per-dispatch sync is NOT
        // paid — a microbench of isolated dispatches cannot measure the true
        // in-forward decode overhead; it measures Metal submit latency.)
        let inp1 = upload_shared_buffer(&device, &vec![0.0f32; dim]);
        let pk1 = upload_shared_buffer(&device, &vec![0u32; pdim]);
        let nrm1 = upload_shared_buffer(&device, &[0.0f32; 1]);
        let out1 = upload_shared_buffer(&device, &vec![0u16; dim]);
        dispatch_fused_quantize(
            &k,
            &device,
            &inp1,
            &signs,
            &bnd,
            &pk1,
            &nrm1,
            1,
            dim as u32,
            bits,
            vpw,
            pdim as u32,
            n_cent,
        )
        .unwrap();
        let submit_us = time(200, &|| {
            dispatch_dequant_fp16(
                &k,
                &device,
                &pk1,
                &nrm1,
                &cent,
                &signs,
                &out1,
                1,
                dim as u32,
                bits,
                vpw,
                pdim as u32,
                q.scale(),
            )
            .unwrap();
        });
        println!(
            "per-dispatch submit+wait floor (1-token dequant) = {submit_us:.1} us  <- mostly Metal submit latency, NOT compute"
        );

        for l in [2048usize, 8192, 16384] {
            let inp = upload_shared_buffer(&device, &vec![0.0f32; l * dim]);
            let pk = upload_shared_buffer(&device, &vec![0u32; l * pdim]);
            let nrm = upload_shared_buffer(&device, &vec![0.0f32; l]);
            let out = upload_shared_buffer(&device, &vec![0u16; l * dim]);
            let qz = time(50, &|| {
                dispatch_fused_quantize(
                    &k,
                    &device,
                    &inp,
                    &signs,
                    &bnd,
                    &pk,
                    &nrm,
                    l as u32,
                    dim as u32,
                    bits,
                    vpw,
                    pdim as u32,
                    n_cent,
                )
                .unwrap();
            });
            let dq = time(50, &|| {
                dispatch_dequant_fp16(
                    &k,
                    &device,
                    &pk,
                    &nrm,
                    &cent,
                    &signs,
                    &out,
                    l as u32,
                    dim as u32,
                    bits,
                    vpw,
                    pdim as u32,
                    q.scale(),
                )
                .unwrap();
            });
            // Subtract the submit floor to estimate the COMPUTE of the prefill
            // quantize/dequant over L tokens (the once-per-prefill added cost).
            let qz_c = (qz - submit_us).max(0.0);
            let dq_c = (dq - submit_us).max(0.0);
            println!(
                "L={l:>6}: prefill quantize {qz_c:>7.0} us + dequant {dq_c:>7.0} us compute (= {:.0} us added once to a {l}-token prefill)",
                qz_c + dq_c
            );
        }
        println!(
            "Prefill added compute is ~1ms at 16k tokens -> negligible vs a multi-second model prefill (<2%)."
        );
        println!(
            "Decode overhead is NOT measurable here (isolated-dispatch sync dominates); it needs the"
        );
        println!(
            "in-forward integration. The decode increment is a 1-token, 2-threadgroup dequant per step.\n"
        );
    }

    /// dim-512 round trip (gemma4 GLOBAL group geometry): compress in place with
    /// `tq_compress_paged`, then dequant the FULL context with
    /// `tq_dequant_blocktable` (the two kernels gemma4 actually uses), comparing
    /// the dequant pool to the host PolarQuantizer round-trip. Exercises the WHT
    /// at dim 512 (9 butterfly steps), the dim/2=256 norm reduction, and 4-bit
    /// packing (vals_per_word=8, packed_dim=64). Parameterized over num_kv_heads
    /// {1,2}.
    #[test]
    fn gpu_turboquant_dim512_blocktable() {
        use objc2_metal::{MTLBuffer, MTLResourceOptions};
        let Some(device) = crate::detect_device().map(|d| d.device) else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };
        let kernels = TurboQuantKernels::new(&device).expect("kernels");
        // Build the block-table dequant pipeline directly (it lives in the
        // interpreter dispatch path, not TurboQuantKernels).
        let library = crate::shader_cache::load_library_from_bytes(
            &device,
            crate::embedded_metallib!("turboquant"),
        )
        .expect("turboquant lib");
        let dequant_bt =
            build_pipeline(&device, &library, "tq_dequant_blocktable").expect("bt pipeline");

        let (dim, bits, seed) = (512usize, 4u32, 42u64);
        let q = PolarQuantizer::new(dim, bits, seed);
        let pdim = packed_dim(dim, bits); // ceil(512/8) = 64
        assert_eq!(pdim, 64, "4-bit dim 512 packs to 64 words");
        let vpw = vals_per_word(bits) as u32; // 8
        let n_cent = q.centroids().len() as u32;
        let boundaries: Vec<f32> = q
            .centroids()
            .windows(2)
            .map(|w| (w[0] + w[1]) / 2.0)
            .collect();

        for num_kv_heads in [1usize, 2usize] {
            // gemma4 global group: block_size 64. 1 chunk, bpc blocks.
            let block_size = 64usize;
            let bpc = 2usize; // blocks per chunk
            let buf_blocks = bpc;
            let n_slots = 80usize; // > 1 block: spans block 0 (64) + block 1 (16)
            assert!(n_slots <= buf_blocks * block_size);

            let kv_blk_stride = num_kv_heads * block_size * dim;
            let kv_head_stride = block_size * dim;
            let pool_halfs = buf_blocks * num_kv_heads * block_size * dim;

            // Deterministic original fp16 values per (slot, kv_head).
            let mut s = 0xABCD_1234u64 ^ (num_kv_heads as u64);
            let mut rnd = || {
                let mut a = 0.0f32;
                for _ in 0..3 {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    a += ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
                }
                a
            };
            let f16 = |x: f32| half::f16::from_f32(x);
            let mut orig = vec![0u16; pool_halfs];
            for slot in 0..n_slots {
                let block = slot / block_size;
                let tok = slot % block_size;
                for kvh in 0..num_kv_heads {
                    let base = block * kv_blk_stride + kvh * kv_head_stride + tok * dim;
                    for d in 0..dim {
                        orig[base + d] = f16(rnd()).to_bits();
                    }
                }
            }

            let pool = device
                .newBufferWithLength_options(pool_halfs * 2, MTLResourceOptions::StorageModeShared)
                .unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    orig.as_ptr() as *const u8,
                    pool.contents().as_ptr() as *mut u8,
                    pool_halfs * 2,
                );
            }
            let chunk_table = upload_shared_buffer(&device, &[pool.gpuAddress()]);
            let slots_buf = upload_shared_buffer(&device, &(0..n_slots as u32).collect::<Vec<_>>());
            let signs = upload_shared_buffer(&device, q.signs());
            let bnd = upload_shared_buffer(&device, &boundaries);
            let cent = upload_shared_buffer(&device, q.centroids());
            // Packed store sized for the WHOLE pool (indexed by physical slot).
            let store_tokens = buf_blocks * block_size;
            let packed_store =
                upload_shared_buffer(&device, &vec![0u32; store_tokens * num_kv_heads * pdim]);
            let norms_store =
                upload_shared_buffer(&device, &vec![0.0f32; store_tokens * num_kv_heads]);

            // ── compress the new slots in place (writes packed store + pool dequant) ──
            dispatch_compress_paged(
                &kernels,
                &device,
                &chunk_table,
                &[&pool],
                &slots_buf,
                &slots_buf,
                &signs,
                &bnd,
                &cent,
                &packed_store,
                &norms_store,
                n_slots as u32,
                dim as u32,
                bits,
                vpw,
                pdim as u32,
                n_cent,
                q.scale(),
                num_kv_heads as u32,
                block_size as u32,
                bpc as u32,
                false,
            )
            .unwrap();

            // ── dequant the FULL context into a FRESH scratch via block table ──
            let scratch = device
                .newBufferWithLength_options(pool_halfs * 2, MTLResourceOptions::StorageModeShared)
                .unwrap();
            let scratch_table = upload_shared_buffer(&device, &[scratch.gpuAddress()]);
            // Block table: seq 0 maps logical block i -> physical block i.
            let max_blocks = buf_blocks as u32;
            let block_table = upload_shared_buffer(&device, &(0..max_blocks).collect::<Vec<_>>());
            let seqused_k = upload_shared_buffer(&device, &[n_slots as u32]);
            run(&device, "tq_dequant_blocktable", |batch| {
                // scratch is reached via scratch_table's gpuAddresses → resident
                // but not bound (extra_resident), exactly like the worker path.
                batch.encode(
                    &dequant_bt,
                    &[
                        (&scratch_table, 0),
                        (&block_table, 1),
                        (&seqused_k, 2),
                        (&signs, 3),
                        (&cent, 4),
                        (&packed_store, 5),
                        (&norms_store, 6),
                    ],
                    &[
                        (dim as u32, 7),
                        (bits, 8),
                        (vpw, 9),
                        (pdim as u32, 10),
                        (num_kv_heads as u32, 12),
                        (block_size as u32, 13),
                        (bpc as u32, 14),
                    ],
                    &[(q.scale(), 11)],
                    &[&scratch],
                    MTLSize {
                        width: max_blocks as usize,
                        height: num_kv_heads,
                        depth: 1,
                    },
                    MTLSize {
                        width: dim,
                        height: 1,
                        depth: 1,
                    },
                );
            })
            .unwrap();

            // ── compare scratch (block-table dequant) vs host round-trip ──
            let pool_out: &[u16] = unsafe {
                std::slice::from_raw_parts(pool.contents().as_ptr() as *const u16, pool_halfs)
            };
            let scratch_out: &[u16] = unsafe {
                std::slice::from_raw_parts(scratch.contents().as_ptr() as *const u16, pool_halfs)
            };
            let gpu_packed: &[u32] = unsafe {
                std::slice::from_raw_parts(
                    packed_store.contents().as_ptr() as *const u32,
                    store_tokens * num_kv_heads * pdim,
                )
            };

            let mut min_cos_pool = f32::MAX;
            let mut min_cos_bt = f32::MAX;
            let mut idx_mismatch = 0usize;
            for slot in 0..n_slots {
                let block = slot / block_size;
                let tok = slot % block_size;
                for kvh in 0..num_kv_heads {
                    let base = block * kv_blk_stride + kvh * kv_head_stride + tok * dim;
                    let orig_v: Vec<f32> = (0..dim)
                        .map(|d| half::f16::from_bits(orig[base + d]).to_f32())
                        .collect();
                    let (hidx, hnorm) = q.quantize(&orig_v);
                    let hp = scratchy_layers::turboquant::pack_indices(&hidx, bits);
                    let phys_slot = block * block_size + tok;
                    let sb = (phys_slot * num_kv_heads + kvh) * pdim;
                    if gpu_packed[sb..sb + pdim] != hp[..] {
                        idx_mismatch += 1;
                    }
                    let hrec = q.dequantize(&hidx, hnorm);
                    let cos = |g: &[u16]| -> f32 {
                        let gpu_rec: Vec<f32> = (0..dim)
                            .map(|d| half::f16::from_bits(g[base + d]).to_f32())
                            .collect();
                        let dot: f64 = hrec
                            .iter()
                            .zip(&gpu_rec)
                            .map(|(&a, &b)| a as f64 * b as f64)
                            .sum();
                        let na: f64 = hrec.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
                        let nb: f64 = gpu_rec
                            .iter()
                            .map(|&x| (x as f64).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        (dot / (na * nb).max(1e-12)) as f32
                    };
                    min_cos_pool = min_cos_pool.min(cos(pool_out));
                    min_cos_bt = min_cos_bt.min(cos(scratch_out));
                }
            }
            println!(
                "dim512 kvh={num_kv_heads}: compress-pool min cos {min_cos_pool:.5}, \
                 blocktable min cos {min_cos_bt:.5}, code mismatches {idx_mismatch}/{}",
                n_slots * num_kv_heads
            );
            assert_eq!(
                idx_mismatch, 0,
                "dim512 packed codes must equal host (kvh={num_kv_heads})"
            );
            assert!(
                min_cos_pool > 0.99,
                "dim512 compress pool must match host (kvh={num_kv_heads}): {min_cos_pool}"
            );
            assert!(
                min_cos_bt > 0.99,
                "dim512 blocktable dequant must match host (kvh={num_kv_heads}): {min_cos_bt}"
            );
        }
    }

    #[test]
    fn gpu_compress_paged_in_place() {
        use objc2_metal::{MTLBuffer, MTLResourceOptions};
        let Some(device) = crate::detect_device().map(|d| d.device) else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };
        let kernels = TurboQuantKernels::new(&device).expect("kernels");

        let (dim, bits, seed) = (128usize, 3u32, 42u64);
        let (num_kv_heads, block_size, bpc) = (2usize, 16usize, 4usize);
        let q = PolarQuantizer::new(dim, bits, seed);
        let pdim = packed_dim(dim, bits);
        let vpw = vals_per_word(bits) as u32;
        let n_cent = q.centroids().len() as u32;
        let boundaries: Vec<f32> = q
            .centroids()
            .windows(2)
            .map(|w| (w[0] + w[1]) / 2.0)
            .collect();

        // Paged pool: 1 chunk, `bpc` blocks. Layout [block, kv_head, tok, dim].
        let buf_blocks = bpc;
        let pool_halfs = buf_blocks * num_kv_heads * block_size * dim;
        // Original fp16 values (random), stored into the pool.
        let mut s = 0xABCDu64;
        let mut rnd = || {
            let mut a = 0.0f32;
            for _ in 0..3 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                a += ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
            }
            a
        };
        // Compress the first 5 tokens of block 0 (slots 0..5), both kv_heads.
        let n_slots = 5usize;
        let _kv_blk_stride = num_kv_heads * block_size * dim;
        let kv_head_stride = block_size * dim;
        let mut orig = vec![0u16; pool_halfs];
        // fp16-round helper so the host quantizes the SAME bits the kernel reads.
        let f16 = |x: f32| half::f16::from_f32(x);
        for slot in 0..n_slots {
            for kvh in 0..num_kv_heads {
                let base = kvh * kv_head_stride + slot * dim; // block 0
                for d in 0..dim {
                    orig[base + d] = f16(rnd()).to_bits();
                }
            }
        }
        let pool = device
            .newBufferWithLength_options(pool_halfs * 2, MTLResourceOptions::StorageModeShared)
            .unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                orig.as_ptr() as *const u8,
                pool.contents().as_ptr() as *mut u8,
                pool_halfs * 2,
            );
        }
        let chunk_table = upload_shared_buffer(&device, &[pool.gpuAddress()]);
        let slots_buf = upload_shared_buffer(&device, &(0..n_slots as u32).collect::<Vec<_>>());
        let signs = upload_shared_buffer(&device, q.signs());
        let bnd = upload_shared_buffer(&device, &boundaries);
        let cent = upload_shared_buffer(&device, q.centroids());
        let packed_store =
            upload_shared_buffer(&device, &vec![0u32; n_slots * num_kv_heads * pdim]);
        let norms_store = upload_shared_buffer(&device, &vec![0.0f32; n_slots * num_kv_heads]);

        dispatch_compress_paged(
            &kernels,
            &device,
            &chunk_table,
            &[&pool],
            &slots_buf,
            &slots_buf,
            &signs,
            &bnd,
            &cent,
            &packed_store,
            &norms_store,
            n_slots as u32,
            dim as u32,
            bits,
            vpw,
            pdim as u32,
            n_cent,
            q.scale(),
            num_kv_heads as u32,
            block_size as u32,
            bpc as u32,
            false,
        )
        .unwrap();

        // Readback
        let pool_out: &[u16] = unsafe {
            std::slice::from_raw_parts(pool.contents().as_ptr() as *const u16, pool_halfs)
        };
        let gpu_packed: &[u32] = unsafe {
            std::slice::from_raw_parts(
                packed_store.contents().as_ptr() as *const u32,
                n_slots * num_kv_heads * pdim,
            )
        };

        let mut min_cos = f32::MAX;
        let mut idx_mismatch = 0;
        for slot in 0..n_slots {
            for kvh in 0..num_kv_heads {
                let base = kvh * kv_head_stride + slot * dim;
                let orig_v: Vec<f32> = (0..dim)
                    .map(|d| half::f16::from_bits(orig[base + d]).to_f32())
                    .collect();
                let (hidx, hnorm) = q.quantize(&orig_v);
                // packed-store codes == host codes
                let hp = scratchy_layers::turboquant::pack_indices(&hidx, bits);
                let sb = (slot * num_kv_heads + kvh) * pdim;
                if gpu_packed[sb..sb + pdim] != hp[..] {
                    idx_mismatch += 1;
                }
                // pool now holds host dequant
                let hrec = q.dequantize(&hidx, hnorm);
                let gpu_rec: Vec<f32> = (0..dim)
                    .map(|d| half::f16::from_bits(pool_out[base + d]).to_f32())
                    .collect();
                let dot: f64 = hrec
                    .iter()
                    .zip(&gpu_rec)
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum();
                let na: f64 = hrec.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
                let nb: f64 = gpu_rec
                    .iter()
                    .map(|&x| (x as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                min_cos = min_cos.min((dot / (na * nb).max(1e-12)) as f32);
            }
        }
        // Diagnostic: slot 0, kv_head 0 — kernel codes vs host codes.
        {
            let orig_v: Vec<f32> = (0..dim)
                .map(|d| half::f16::from_bits(orig[d]).to_f32())
                .collect();
            let (hidx, _) = q.quantize(&orig_v);
            let gpu_idx =
                scratchy_layers::turboquant::unpack_indices(&gpu_packed[0..pdim], bits, dim);
            let diffs = gpu_idx.iter().zip(&hidx).filter(|(a, b)| a != b).count();
            println!(
                "  [diag s0h0] code diffs = {diffs}/{dim}; host[..12]={:?} gpu[..12]={:?}",
                &hidx[..12],
                &gpu_idx[..12]
            );
        }
        println!(
            "tq_compress_paged: min cosine(pool, host dequant) = {min_cos:.5}, packed-code mismatches {idx_mismatch}/{}",
            n_slots * num_kv_heads
        );
        assert_eq!(idx_mismatch, 0, "packed store must equal host codes");
        assert!(
            min_cos > 0.99,
            "pool must hold host dequant after compress: {min_cos}"
        );

        // untouched slots stay zero (we only wrote 0..n_slots)
        let untouched = kv_head_stride; // slot n_slots, kv_head 0... actually check a slot we didn't compress
        let zb = n_slots * dim; // kv_head 0, slot n_slots
        assert!(
            (0..dim).all(|d| pool_out[zb + d] == 0),
            "uncompressed slots must be untouched, base {untouched}"
        );

        // Validate the dequant-on-READ primitive: dequant the packed store into
        // a FRESH fp16 buffer; it must match the in-place writeback (the pool).
        let target = device
            .newBufferWithLength_options(pool_halfs * 2, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let target_table = upload_shared_buffer(&device, &[target.gpuAddress()]);
        dispatch_dequant_paged(
            &kernels,
            &device,
            &target_table,
            &[&target],
            &slots_buf,
            &slots_buf,
            &signs,
            &cent,
            &packed_store,
            &norms_store,
            n_slots as u32,
            dim as u32,
            bits,
            vpw,
            pdim as u32,
            q.scale(),
            num_kv_heads as u32,
            block_size as u32,
            bpc as u32,
            false,
        )
        .unwrap();
        let tgt_out: &[u16] = unsafe {
            std::slice::from_raw_parts(target.contents().as_ptr() as *const u16, pool_halfs)
        };
        let mut dq_max = 0.0f32;
        for slot in 0..n_slots {
            for kvh in 0..num_kv_heads {
                let base = kvh * kv_head_stride + slot * dim;
                for d in 0..dim {
                    let a = half::f16::from_bits(pool_out[base + d]).to_f32();
                    let b = half::f16::from_bits(tgt_out[base + d]).to_f32();
                    dq_max = dq_max.max((a - b).abs());
                }
            }
        }
        println!("tq_dequant_paged: max|pool - dequant_paged| = {dq_max:.6}");
        assert!(
            dq_max < 1e-3,
            "dequant-on-read must match the in-place dequant: {dq_max}"
        );

        // Remap validation (the eviction-layer fill): dequant SOURCE logical
        // slots into DIFFERENT (reversed) window slots; target[dst] must hold
        // the dequant of src — proving a logical block can land in any window slot.
        let dst_phys: Vec<u32> = (0..n_slots as u32).rev().collect();
        let dst_buf = upload_shared_buffer(&device, &dst_phys);
        let target2 = device
            .newBufferWithLength_options(pool_halfs * 2, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let t2_table = upload_shared_buffer(&device, &[target2.gpuAddress()]);
        dispatch_dequant_paged(
            &kernels,
            &device,
            &t2_table,
            &[&target2],
            &slots_buf,
            &dst_buf,
            &signs,
            &cent,
            &packed_store,
            &norms_store,
            n_slots as u32,
            dim as u32,
            bits,
            vpw,
            pdim as u32,
            q.scale(),
            num_kv_heads as u32,
            block_size as u32,
            bpc as u32,
            false,
        )
        .unwrap();
        let t2_out: &[u16] = unsafe {
            std::slice::from_raw_parts(target2.contents().as_ptr() as *const u16, pool_halfs)
        };
        let mut remap_max = 0.0f32;
        for slot in 0..n_slots {
            for kvh in 0..num_kv_heads {
                let src_base = kvh * kv_head_stride + slot * dim;
                let dst_slot = n_slots - 1 - slot;
                let dst_base = kvh * kv_head_stride + dst_slot * dim;
                for d in 0..dim {
                    let a = half::f16::from_bits(pool_out[src_base + d]).to_f32();
                    let b = half::f16::from_bits(t2_out[dst_base + d]).to_f32();
                    remap_max = remap_max.max((a - b).abs());
                }
            }
        }
        println!("tq_dequant_paged remap: max|pool[src] - target[dst]| = {remap_max:.6}");
        assert!(
            remap_max < 1e-3,
            "remap dequant must place src's dequant at dst slot: {remap_max}"
        );
    }

    /// REGRESSION GUARD for the long-context `!!!!` collapse: the full-context
    /// `tq_dequant_blocktable` dispatch grid.x is the block-table row stride;
    /// the kernel only fills blocks `0..grid.x`. The worker once hardcoded
    /// grid.x = `W::MAX_BLOCKS_PER_SEQ` (128 for uniform arches), so any context
    /// past 128 blocks (2048 tokens) left the reused fp16 scratch's tail blocks
    /// stale → attention collapse. This test reproduces a >128-block context and
    /// asserts (a) grid.x=128 (the OLD bug) leaves blocks ≥128 UNFILLED, and
    /// (b) grid.x = the real block count fills the WHOLE context faithfully — the
    /// invariant the worker dispatch (`tq_dequant_max_blocks`) must uphold.
    #[test]
    fn gpu_turboquant_blocktable_covers_all_blocks() {
        use objc2_metal::{MTLBuffer, MTLResourceOptions};
        let Some(device) = crate::detect_device().map(|d| d.device) else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };
        let library = crate::shader_cache::load_library_from_bytes(
            &device,
            crate::embedded_metallib!("turboquant"),
        )
        .expect("turboquant lib");
        let compress = build_pipeline(&device, &library, "tq_compress_paged").expect("compress");
        let dequant_bt =
            build_pipeline(&device, &library, "tq_dequant_blocktable").expect("bt pipeline");

        // Llama-3.2-1B-ish geometry, but a context that SPANS the 128-block
        // boundary (and thus 2 chunks at the production bpc=128).
        let (dim, bits, seed) = (64usize, 3u32, 42u64);
        let (num_kv_heads, block_size, bpc) = (2usize, 16usize, 128usize);
        let n_blocks = 200usize; // > 128 → blocks 128..200 are the "tail"
        let n_slots = n_blocks * block_size; // full context, every slot used
        let q = PolarQuantizer::new(dim, bits, seed);
        let pdim = packed_dim(dim, bits);
        let vpw = vals_per_word(bits) as u32;
        let n_cent = q.centroids().len() as u32;
        let boundaries: Vec<f32> = q
            .centroids()
            .windows(2)
            .map(|w| (w[0] + w[1]) / 2.0)
            .collect();

        let kv_blk_stride = num_kv_heads * block_size * dim;
        let kv_head_stride = block_size * dim;
        let pool_halfs = n_blocks * kv_blk_stride;

        // Deterministic original fp16 values for every (slot, kv_head).
        let mut s = 0x5EED_1234u64;
        let mut rnd = || {
            let mut a = 0.0f32;
            for _ in 0..3 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                a += ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
            }
            a
        };
        let f16 = |x: f32| half::f16::from_f32(x);
        let mut orig = vec![0u16; pool_halfs];
        for slot in 0..n_slots {
            let block = slot / block_size;
            let tok = slot % block_size;
            for kvh in 0..num_kv_heads {
                let base = block * kv_blk_stride + kvh * kv_head_stride + tok * dim;
                for d in 0..dim {
                    orig[base + d] = f16(rnd()).to_bits();
                }
            }
        }

        // Contiguous pool with a 2-entry chunk table (bpc=128) — exactly the
        // production scratch layout (`build_tq_provision`).
        let pool = device
            .newBufferWithLength_options(pool_halfs * 2, MTLResourceOptions::StorageModeShared)
            .unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                orig.as_ptr() as *const u8,
                pool.contents().as_ptr() as *mut u8,
                pool_halfs * 2,
            );
        }
        let n_chunks = n_blocks.div_ceil(bpc);
        let chunk_bytes = bpc * kv_blk_stride * 2;
        let pool_base = pool.gpuAddress();
        let chunk_table = upload_shared_buffer(
            &device,
            &(0..n_chunks)
                .map(|c| pool_base + (c * chunk_bytes) as u64)
                .collect::<Vec<_>>(),
        );
        let slots_buf = upload_shared_buffer(&device, &(0..n_slots as u32).collect::<Vec<_>>());
        let signs = upload_shared_buffer(&device, q.signs());
        let bnd = upload_shared_buffer(&device, &boundaries);
        let cent = upload_shared_buffer(&device, q.centroids());
        let packed_store =
            upload_shared_buffer(&device, &vec![0u32; n_slots * num_kv_heads * pdim]);
        let norms_store = upload_shared_buffer(&device, &vec![0.0f32; n_slots * num_kv_heads]);

        // Compress the WHOLE context into the packed store (n_slots dispatched
        // over X; no per-block cap on the compress side).
        dispatch_compress_paged(
            &TurboQuantKernels {
                quantize: build_pipeline(&device, &library, "tq_fused_quantize").unwrap(),
                dequant_fp16: build_pipeline(&device, &library, "tq_dequant_fp16").unwrap(),
                compress_paged: compress.clone(),
                compress_paged_bf16: build_pipeline(&device, &library, "tq_compress_paged_bf16")
                    .unwrap(),
                dequant_paged: build_pipeline(&device, &library, "tq_dequant_paged").unwrap(),
                dequant_paged_bf16: build_pipeline(&device, &library, "tq_dequant_paged_bf16")
                    .unwrap(),
                _library: library.clone(),
            },
            &device,
            &chunk_table,
            &[&pool],
            &slots_buf,
            &slots_buf,
            &signs,
            &bnd,
            &cent,
            &packed_store,
            &norms_store,
            n_slots as u32,
            dim as u32,
            bits,
            vpw,
            pdim as u32,
            n_cent,
            q.scale(),
            num_kv_heads as u32,
            block_size as u32,
            bpc as u32,
            false,
        )
        .unwrap();

        // Block table: logical block i -> physical block i; full context used.
        let block_table = upload_shared_buffer(&device, &(0..n_blocks as u32).collect::<Vec<_>>());
        let seqused_k = upload_shared_buffer(&device, &[n_slots as u32]);

        // Dequant the full context into a FRESH scratch with a given grid.x.
        // Returns the scratch fp16 readback (host copy).
        let run_dequant = |grid_x: usize| -> Vec<u16> {
            let scratch = device
                .newBufferWithLength_options(pool_halfs * 2, MTLResourceOptions::StorageModeShared)
                .unwrap();
            let sbase = scratch.gpuAddress();
            let scratch_table = upload_shared_buffer(
                &device,
                &(0..n_chunks)
                    .map(|c| sbase + (c * chunk_bytes) as u64)
                    .collect::<Vec<_>>(),
            );
            run(&device, "tq_dequant_blocktable", |batch| {
                batch.encode(
                    &dequant_bt,
                    &[
                        (&scratch_table, 0),
                        (&block_table, 1),
                        (&seqused_k, 2),
                        (&signs, 3),
                        (&cent, 4),
                        (&packed_store, 5),
                        (&norms_store, 6),
                    ],
                    &[
                        (dim as u32, 7),
                        (bits, 8),
                        (vpw, 9),
                        (pdim as u32, 10),
                        (num_kv_heads as u32, 12),
                        (block_size as u32, 13),
                        (bpc as u32, 14),
                    ],
                    &[(q.scale(), 11)],
                    &[&scratch],
                    MTLSize {
                        width: grid_x,
                        height: num_kv_heads,
                        depth: 1,
                    },
                    MTLSize {
                        width: dim,
                        height: 1,
                        depth: 1,
                    },
                );
            })
            .unwrap();
            let out: &[u16] = unsafe {
                std::slice::from_raw_parts(scratch.contents().as_ptr() as *const u16, pool_halfs)
            };
            out.to_vec()
        };

        // Per-block min cosine vs host round-trip, over a range of blocks.
        let min_cos_over = |scratch: &[u16], blocks: std::ops::Range<usize>| -> f32 {
            let mut mc = f32::MAX;
            for block in blocks {
                for tok in 0..block_size {
                    for kvh in 0..num_kv_heads {
                        let base = block * kv_blk_stride + kvh * kv_head_stride + tok * dim;
                        let ov: Vec<f32> = (0..dim)
                            .map(|d| half::f16::from_bits(orig[base + d]).to_f32())
                            .collect();
                        let (hidx, hnorm) = q.quantize(&ov);
                        let hrec = q.dequantize(&hidx, hnorm);
                        let gr: Vec<f32> = (0..dim)
                            .map(|d| half::f16::from_bits(scratch[base + d]).to_f32())
                            .collect();
                        let dot: f64 = hrec
                            .iter()
                            .zip(&gr)
                            .map(|(&a, &b)| a as f64 * b as f64)
                            .sum();
                        let na: f64 = hrec.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
                        let nb: f64 = gr.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
                        mc = mc.min((dot / (na * nb).max(1e-12)) as f32);
                    }
                }
            }
            mc
        };

        // (a) OLD BUG: grid.x = MAX_BLOCKS_PER_SEQ (128). Blocks 0..128 fill;
        //     the tail 128..200 is NEVER dispatched → stays zero (the scratch
        //     starts zeroed here; in production it holds the prior layer's KV).
        let bug = run_dequant(crate::BLOCKS_PER_CHUNK as usize); // 128
        let bug_head = min_cos_over(&bug, 0..128);
        let tail_all_zero = (128 * kv_blk_stride..200 * kv_blk_stride).all(|i| bug[i] == 0);
        println!(
            "grid.x=128 (BUG): head[0,128) min cos {bug_head:.4}; tail[128,200) all-zero={tail_all_zero}"
        );
        assert!(
            bug_head > 0.99,
            "head blocks must dequant fine even with the bug"
        );
        assert!(
            tail_all_zero,
            "the bug MUST leave tail blocks unfilled (proves grid.x=128 truncates)"
        );

        // (b) FIX: grid.x = real block count. The WHOLE context is faithful.
        let fixed = run_dequant(n_blocks);
        let fix_all = min_cos_over(&fixed, 0..n_blocks);
        println!("grid.x={n_blocks} (FIX): all blocks min cos {fix_all:.4}");
        assert!(
            fix_all > 0.99,
            "grid.x = block count must dequant the WHOLE >128-block context: {fix_all}"
        );
    }
}

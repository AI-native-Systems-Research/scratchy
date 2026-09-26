// SPDX-License-Identifier: Apache-2.0
//! TurboQuant KV-cache provisioning for the Metal tape. [`build_tq_provision`]
//! allocates the codebook, the per-layer packed store + norms (the canonical
//! compressed cache) and the fp16 scratch that the tape's TurboQuant ops bind:
//! `tq_compress_paged[_bf16]` (quantize new KV into the packed store, in
//! `turboquant.metal`); attention reads the packed store itself
//! (`attention.metal`).

use objc2_metal::MTLDevice;

use crate::argmax::{Buffer, Device};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::argmax::{ComputePipelineState, upload_shared_buffer};
    use crate::mtl4_dispatch::{Mtl4DispatchBatch, read_slice};
    use objc2_foundation::NSString;
    use objc2_metal::{MTLBuffer, MTLLibrary, MTLSize};
    use scratchy_layers::turboquant::{PolarQuantizer, pack_indices, packed_dim, vals_per_word};

    /// The embedded `turboquant.metallib`'s pipeline for kernel `name`.
    fn pipeline(device: &Device, name: &str) -> ComputePipelineState {
        let library = crate::shader_cache::load_library_from_bytes(
            device,
            crate::embedded_metallib!("turboquant"),
        )
        .expect("turboquant lib");
        let f = library
            .newFunctionWithName(&NSString::from_str(name))
            .expect(name);
        device
            .newComputePipelineStateWithFunction_error(&f)
            .expect(name)
    }

    /// Encode with `body` into one MTL4 batch and commit it, waiting for the GPU.
    fn run(device: &Device, body: impl FnOnce(&mut Mtl4DispatchBatch)) {
        let mut batch = Mtl4DispatchBatch::begin(device).expect("no MTL4 queue");
        body(&mut batch);
        batch.commit(true).expect("MTL4 commit");
    }

    /// One (dim, bits) codebook, uploaded exactly as `build_tq_provision` does.
    struct Codebook {
        q: PolarQuantizer,
        dim: usize,
        bits: u32,
        pdim: usize,
        signs: Buffer,
        boundaries: Buffer,
        centroids: Buffer,
    }

    impl Codebook {
        fn new(device: &Device, dim: usize, bits: u32) -> Self {
            let q = PolarQuantizer::new(dim, bits, 42);
            let boundaries: Vec<f32> = q
                .centroids()
                .windows(2)
                .map(|w| (w[0] + w[1]) / 2.0)
                .collect();
            Self {
                pdim: packed_dim(dim, bits),
                signs: upload_shared_buffer(device, q.signs()),
                boundaries: upload_shared_buffer(device, &boundaries),
                centroids: upload_shared_buffer(device, q.centroids()),
                q,
                dim,
                bits,
            }
        }

        /// Host reference for one vector: its packed codes and its dequant.
        fn host(&self, v: &[f32]) -> (Vec<u32>, Vec<f32>) {
            let (idx, norm) = self.q.quantize(v);
            (pack_indices(&idx, self.bits), self.q.dequantize(&idx, norm))
        }
    }

    /// One layer's paged KV — fp16 `[block, kv_head, tok, dim]` in `data`,
    /// reached through `table`'s chunk `gpuAddress`es — and its packed store.
    #[derive(Clone, Copy)]
    struct PagedKv<'a> {
        cb: &'a Codebook,
        table: &'a Buffer,
        data: &'a Buffer,
        packed: &'a Buffer,
        norms: &'a Buffer,
        num_kv_heads: usize,
        block_size: usize,
        bpc: usize,
    }

    /// `halfs` in one shared buffer plus its table of chunk `gpuAddress`es, one
    /// per `chunk_bytes` — the `build_tq_provision` scratch layout.
    fn chunked(device: &Device, halfs: &[u16], chunk_bytes: usize) -> (Buffer, Buffer) {
        let data = upload_shared_buffer(device, halfs);
        let base = data.gpuAddress();
        let table: Vec<u64> = (0..(halfs.len() * 2).div_ceil(chunk_bytes))
            .map(|c| base + (c * chunk_bytes) as u64)
            .collect();
        let table = upload_shared_buffer(device, &table);
        (data, table)
    }

    /// `tq_compress_paged` over token slots `slots[..n_slots]`, bound as the
    /// kernel declares (0..=23, no offset): `slots` doubles as `logical_slots`
    /// (in place) and `do_writeback = 1`, so `kv.data` ends up holding the
    /// lossy dequant.
    fn compress(device: &Device, kv: PagedKv, slots: &Buffer, n_slots: usize) {
        let pso = pipeline(device, "tq_compress_paged");
        let cb = kv.cb;
        run(device, |batch| {
            // `kv.data` is reached via the chunk table: resident, not bound.
            batch.encode(
                &pso,
                &[
                    (kv.table, 0),
                    (slots, 1),
                    (&cb.signs, 2),
                    (&cb.boundaries, 3),
                    (&cb.centroids, 4),
                    (kv.packed, 5),
                    (kv.norms, 6),
                    (slots, 16),
                ],
                &[
                    (cb.dim as u32, 7),
                    (cb.bits, 8),
                    (vals_per_word(cb.bits) as u32, 9),
                    (cb.pdim as u32, 10),
                    (cb.q.centroids().len() as u32, 11),
                    (kv.num_kv_heads as u32, 13),
                    (kv.block_size as u32, 14),
                    (kv.bpc as u32, 15),
                    (1, 17),
                    (0, 21),
                    (0, 22),
                    (0, 23),
                ],
                &[(cb.q.scale(), 12)],
                &[kv.data],
                MTLSize {
                    width: n_slots,
                    height: kv.num_kv_heads,
                    depth: 1,
                },
                MTLSize {
                    width: cb.dim,
                    height: 1,
                    depth: 1,
                },
            );
        });
    }

    /// Deterministic samples: the sum of three LCG uniforms in [-1, 1).
    fn rng(mut s: u64) -> impl FnMut() -> f32 {
        move || {
            let mut a = 0.0f32;
            for _ in 0..3 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                a += ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
            }
            a
        }
    }

    fn f32s(halfs: &[u16]) -> Vec<f32> {
        halfs
            .iter()
            .map(|&h| half::f16::from_bits(h).to_f32())
            .collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        (dot / (norm(a) * norm(b)).max(1e-12)) as f32
    }

    /// `tq_compress_paged` in place at every (head_dim, bits) the metal path
    /// provisions — NOT just (128, 3). `qwen2.5-0.5b` (head_dim 64) decodes
    /// garbage under TurboQuant while `llama-3.2-1b` (also 64, but 3-bit) is
    /// clean, and the host-side codebook is FINE at 64 (mean cosine 0.9849 /
    /// 0.9958, better than at 128). So the untested combination is the suspect,
    /// and this sweep is what makes it a test rather than an argument.
    #[test]
    fn gpu_compress_paged_in_place() {
        let Some(device) = crate::detect_device().map(|d| d.device) else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };
        for (dim, bits) in [
            (64usize, 3u32),
            (64, 4),
            (128, 3),
            (128, 4),
            (256, 4),
            (512, 4),
        ] {
            compress_in_place_at(&device, dim, bits);
        }
    }

    fn compress_in_place_at(device: &Device, dim: usize, bits: u32) {
        let cb = Codebook::new(device, dim, bits);
        // Paged pool: 1 chunk of `bpc` blocks, layout [block, kv_head, tok, dim].
        let (num_kv_heads, block_size, bpc) = (2usize, 16usize, 8usize);
        let kv_blk_stride = num_kv_heads * block_size * dim;
        let pool_halfs = bpc * kv_blk_stride;
        let base = |slot: usize, kvh: usize| {
            (slot / block_size) * kv_blk_stride + (kvh * block_size + slot % block_size) * dim
        };
        // Compress every slot but the pool's last, both kv_heads: 254 vectors.
        let n_slots = bpc * block_size - 1;
        let mut rnd = rng(0xABCD);
        let mut orig = vec![0u16; pool_halfs];
        for slot in 0..n_slots {
            for kvh in 0..num_kv_heads {
                // fp16-round so the host quantizes the SAME bits the kernel reads.
                for h in &mut orig[base(slot, kvh)..base(slot, kvh) + dim] {
                    *h = half::f16::from_f32(rnd()).to_bits();
                }
            }
        }
        let (pool, table) = chunked(device, &orig, pool_halfs * 2);
        let slots = upload_shared_buffer(device, &(0..n_slots as u32).collect::<Vec<_>>());
        let packed = upload_shared_buffer(device, &vec![0u32; n_slots * num_kv_heads * cb.pdim]);
        let norms = upload_shared_buffer(device, &vec![0.0f32; n_slots * num_kv_heads]);
        let kv = PagedKv {
            cb: &cb,
            table: &table,
            data: &pool,
            packed: &packed,
            norms: &norms,
            num_kv_heads,
            block_size,
            bpc,
        };
        compress(device, kv, &slots, n_slots);

        let pool_out: Vec<u16> = read_slice(&pool, pool_halfs);
        let gpu_packed: Vec<u32> = read_slice(&packed, n_slots * num_kv_heads * cb.pdim);
        let mut min_cos = f32::MAX;
        let mut idx_mismatch = 0;
        for slot in 0..n_slots {
            for kvh in 0..num_kv_heads {
                let b = base(slot, kvh);
                let (hp, hrec) = cb.host(&f32s(&orig[b..b + dim]));
                // packed-store codes == host codes
                let sb = (slot * num_kv_heads + kvh) * cb.pdim;
                if gpu_packed[sb..sb + cb.pdim] != hp[..] {
                    idx_mismatch += 1;
                }
                // pool now holds host dequant
                min_cos = min_cos.min(cosine(&hrec, &f32s(&pool_out[b..b + dim])));
            }
        }
        println!(
            "tq_compress_paged dim {dim} bits {bits}: min cosine(pool, host dequant) = {min_cos:.5}, \
             packed-code mismatches {idx_mismatch}/{}",
            n_slots * num_kv_heads
        );
        assert_eq!(
            idx_mismatch, 0,
            "dim {dim} bits {bits}: packed store must equal host codes"
        );
        assert!(
            min_cos > 0.99,
            "dim {dim} bits {bits}: pool must hold host dequant after compress: {min_cos}"
        );
        let zb = base(n_slots, 0);
        assert!(
            pool_out[zb..zb + dim].iter().all(|&h| h == 0),
            "dim {dim} bits {bits}: uncompressed slot {n_slots} must be untouched"
        );
    }
}

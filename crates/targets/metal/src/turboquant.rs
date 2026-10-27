// SPDX-License-Identifier: Apache-2.0
//! TurboQuant KV-cache provisioning for the Metal tape. [`build_tq_provision`]
//! allocates the codebook, the per-layer packed store + norms (the canonical
//! compressed cache) and the fp16 scratch that the tape's TurboQuant ops bind:
//! `tq_compress_paged[_bf16]` (quantize new KV into the packed store, in
//! `turboquant.metal`); attention reads the packed store itself
//! (`attention.metal`).

use objc2_metal::MTLDevice;
use scratchy_layers::turboquant::TqBits;
pub use scratchy_layers::turboquant::{TqBlockBytes, kv_bytes_per_token, scratch_bytes_per_token};

use crate::argmax::Buffer;

/// Seed of the production codebook's rotation signs.
pub const TQ_CODEBOOK_SEED: u64 = 42;

/// A model's TurboQuant codebook at its GLOBAL head dim (`signs.len()`): the rotation signs, the
/// `bits`-bit centroids, the encoders' table (their midpoints, then the centroids again:
/// [`PolarQuantizer::encode_table`](scratchy_layers::turboquant::PolarQuantizer::encode_table)) and
/// the packed words per head — computed at expansion ([`TqCodebook::compute`]) and emitted as a
/// static per model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TqCodebook {
    pub bits: TqBits,
    pub packed_dim: u32,
    pub signs: &'static [f32],
    pub centroids: &'static [f32],
    pub encode_table: &'static [f32],
}

impl TqCodebook {
    pub fn compute(head_dim: usize, bits: TqBits) -> Self {
        use scratchy_layers::turboquant::{PolarQuantizer, packed_dim};
        let q = PolarQuantizer::new(head_dim, bits.get(), TQ_CODEBOOK_SEED);
        let baked = crate::tape::lowered::baked;
        Self {
            bits,
            packed_dim: packed_dim(head_dim, bits.get()) as u32,
            signs: baked(q.signs().to_vec()),
            centroids: baked(q.centroids().to_vec()),
            encode_table: baked(q.encode_table()),
        }
    }
}

/// The KV pool a TurboQuant provision serves, at its GLOBAL (full-context) geometry.
#[derive(Clone, Copy)]
pub struct TqPool<'a> {
    pub is_global: &'a [bool],
    pub num_blocks: usize,
    pub block_size: usize,
    pub num_kv_heads: usize,
    pub blocks_per_chunk: usize,
}

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
/// placeholder buffer that the tape never binds (the KV codec pass codes only
/// the global KV writers). Uniform arches pass all-true → every layer real
/// (byte-identical to before, when `is_global` was implicitly all-true).
///
/// `hosted` is the scratch of the pool whose arena this one runs in (a head's target's,
/// [`crate::interpreter::metal::HostArena::tq_scratch`]): of this geometry, the provision
/// stages through it and allocates none of its own.
pub fn build_tq_provision(
    allocator: &crate::MetalAllocator,
    pool: TqPool<'_>,
    codebook: &TqCodebook,
    hosted: Option<&crate::interpreter::metal::runtime::TqScratch>,
) -> crate::interpreter::metal::runtime::TqRuntimeBuffers {
    let TqPool {
        is_global,
        num_blocks,
        block_size,
        num_kv_heads,
        blocks_per_chunk,
    } = pool;
    use objc2_metal::{MTLBuffer, MTLResourceOptions};
    let device = allocator.device();
    let (bits, head_dim) = (codebook.bits.get(), codebook.signs.len());
    let num_layers = is_global.len();
    let n_global = is_global.iter().filter(|&&g| g).count();
    let pdim = codebook.packed_dim as usize;
    tracing::info!(
        "TurboQuant KV: {bits}-bit codebook (head_dim={head_dim}, \
         num_kv_heads={num_kv_heads}, block_size={block_size}, packed_dim={pdim}, \
         {num_blocks} blocks, {n_global}/{num_layers} global layers compressed)"
    );

    // Per-layer packed store + norms (canonical cache), at the GLOBAL geometry, then the
    // per-worker fp16 scratch: one layer (num_blocks blocks) at the GLOBAL geometry, reused
    // across all global layers (sequential, like the uniform case — a dedicated scratch, not a
    // pool-shared tensor). All are placement-sparse, mapped as group 0's blocks are reached
    // (`SparseRegistry`): sized to the capacity and wired, they took 21 GB of a 32 GB M5 on the
    // first request (granite-2b). The NORMS read zero until written: the per-layer dequant runs
    // BEFORE rope every forward over the WHOLE active context, so a not-yet-quantized block's
    // norm slot is read, and `K = centroid*scale*sign*norm` overflows to Inf -> NaN logits ->
    // `!!!!` on a continuation prefill; a zero norm dequants to 0, which rope overwrites. So does
    // the scratch: the paged attention can stage blocks just past the active context. Packed
    // codes need no zeroing (a garbage code indexes a bounded centroid; times norm 0 is 0).
    let bytes = TqBlockBytes::of(block_size, num_kv_heads, head_dim, bits);
    let (packed, norms, scratch) = (
        (bytes.packed, false),
        (bytes.norms, true),
        (bytes.scratch, true),
    );
    // A hosted scratch of this geometry is staged through, and none of this pool's own is laid
    // out: its stores are its host's, mapped as far as the same blocks are reached.
    let chunk_bytes = blocks_per_chunk * scratch.0;
    let scratch_bytes = num_blocks * scratch.0;
    let hosted = hosted.filter(|s| s.bytes == scratch_bytes && s.chunk_bytes == chunk_bytes);
    let own_scratch = if hosted.is_some() { 0 } else { 2 };
    let layout: Vec<(usize, bool)> = [packed, packed, norms, norms]
        .into_iter()
        .flat_map(|store| std::iter::repeat_n(store, n_global))
        .chain(std::iter::repeat_n(scratch, own_scratch))
        .collect();
    let stores = crate::sparse_kv::SparseStores::new(
        device,
        allocator.residency(),
        num_blocks,
        blocks_per_chunk,
        &layout,
    );
    let stores = stores.expect("TurboQuant stores");
    // The TqPackedK{layer} binding is layer-indexed, so a global layer L's store must sit at
    // index L; sliding layers (is_global[L]==false) get a 16-byte placeholder that the tape
    // never binds.
    let mut next = (0..layout.len()).map(|i| stores.buffer(i));
    let mut by_layer = || -> Vec<Buffer> {
        let shared = MTLResourceOptions::StorageModeShared;
        let mut one = |&g: &bool| match g {
            true => next.next().expect("a store per global layer"),
            false => device
                .newBufferWithLength_options(16, shared)
                .expect("tq alloc"),
        };
        is_global.iter().map(&mut one).collect()
    };
    let (packed_k, packed_v, norms_k, norms_v) = (by_layer(), by_layer(), by_layer(), by_layer());
    // Contiguous scratch data + a chunk-table of gpuAddresses at chunk offsets (the kernels
    // deref it).
    let n_chunks = num_blocks.div_ceil(blocks_per_chunk.max(1));
    let table = |data: &Buffer| {
        let base = data.gpuAddress();
        let vals: Vec<u64> = (0..n_chunks)
            .map(|c| base + (c * chunk_bytes) as u64)
            .collect();
        crate::argmax::upload_shared_buffer(device, &vals)
    };
    let scratch = match hosted {
        Some(s) => s.clone(),
        None => {
            let k_data = next.next().expect("K scratch");
            let v_data = next.next().expect("V scratch");
            crate::interpreter::metal::runtime::TqScratch {
                k_table: table(&k_data),
                v_table: table(&v_data),
                k_data,
                v_data,
                bytes: scratch_bytes,
                chunk_bytes,
            }
        }
    };
    let mapped = stores.mapped();
    let registry = allocator.sparse_kv.lock();
    let registered = registry.expect("sparse KV registry").register(stores);
    registered.expect("TurboQuant stores mapped as far as the pool reaches");

    crate::interpreter::metal::runtime::TqRuntimeBuffers {
        packed_k,
        packed_v,
        norms_k,
        norms_v,
        signs: crate::argmax::upload_shared_buffer(device, codebook.signs),
        encode_table: crate::argmax::upload_shared_buffer(device, codebook.encode_table),
        centroids: crate::argmax::upload_shared_buffer(device, codebook.centroids),
        scratch,
        mapped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::argmax::{Device, upload_shared_buffer};
    use crate::mtl4_dispatch::{Mtl4DispatchBatch, read_slice};
    use objc2_metal::{MTLBuffer, MTLSize};
    use scratchy_layers::turboquant::{
        KvCodec, PolarQuantizer, SCRATCH_ELEM_BYTES, pack_indices, packed_dim, vals_per_word,
    };

    /// `tq_compress_paged` baked for `kv` with no offset, writing its dequant back into the pool.
    fn compress_pipeline(device: &Device, kv: PagedKv) -> crate::aot::BakedPipeline {
        use crate::tape::ids::{
            BlockSize, BlocksPerChunk, HeadDim, NumKvHeads, RopePairOff, RotDim, TqCodeBits,
        };
        use crate::tape::kernel_constants::{TqCompressConstants, TqOffset, TqWriteback};
        let constants = TqCompressConstants {
            head_dim: HeadDim(kv.cb.dim as u32),
            bits: TqCodeBits(kv.cb.bits),
            num_kv_heads: NumKvHeads(kv.num_kv_heads as u32),
            block_size: BlockSize(kv.block_size as u32),
            blocks_per_chunk: BlocksPerChunk(kv.bpc as u32),
            writeback: TqWriteback::Dequantized,
            offset: TqOffset::None,
            rot_dim: RotDim(0),
            pair_off: RopePairOff(0),
        };
        crate::aot::baked_pipeline(device, "turboquant", "tq_compress_paged", constants.into())
            .expect("tq_compress_paged")
    }

    /// Encode with `body` into one MTL4 batch and commit it, waiting for the GPU.
    fn run(device: &Device, body: impl FnOnce(&mut Mtl4DispatchBatch)) {
        let mut batch = Mtl4DispatchBatch::begin(device).expect("no MTL4 queue");
        body(&mut batch);
        batch.commit(true);
    }

    /// One (dim, bits) codebook, uploaded exactly as `build_tq_provision` does.
    struct Codebook {
        q: PolarQuantizer,
        dim: usize,
        bits: u32,
        pdim: usize,
        signs: Buffer,
        encode_table: Buffer,
        centroids: Buffer,
    }

    impl Codebook {
        fn new(device: &Device, dim: usize, bits: u32) -> Self {
            let cb = TqCodebook::compute(dim, TqBits::new(bits));
            Self {
                pdim: cb.packed_dim as usize,
                signs: upload_shared_buffer(device, cb.signs),
                encode_table: upload_shared_buffer(device, cb.encode_table),
                centroids: upload_shared_buffer(device, cb.centroids),
                q: PolarQuantizer::new(dim, bits, TQ_CODEBOOK_SEED),
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
    /// kernel declares (no offset): `slots` doubles as `logical_slots`
    /// (in place) and the dequant is written back, so `kv.data` ends up holding
    /// the lossy dequant.
    fn compress(device: &Device, kv: PagedKv, slots: &Buffer, n_slots: usize) {
        let pso = compress_pipeline(device, kv);
        let cb = kv.cb;
        run(device, |batch| {
            // `kv.data` is reached via the chunk table: resident, not bound.
            batch.encode(
                &pso,
                &[
                    (kv.table, 0),
                    (slots, 1),
                    (&cb.signs, 2),
                    (&cb.encode_table, 3),
                    (&cb.centroids, 4),
                    (kv.packed, 5),
                    (kv.norms, 6),
                    (slots, 16),
                ],
                &[],
                &[],
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

    /// The KV budget sizes a TurboQuant pool with `kv_bytes_per_token`; the
    /// buffers `build_tq_provision` sizes per token — every layer's packed
    /// codes and norms, and the K and V scratch — must add up to exactly that,
    /// or the pool the engine counts is not the pool the GPU holds. (Sparse
    /// stores span whole steps of whole tiles: 128 blocks here, so exactly.)
    #[test]
    fn provision_allocates_kv_bytes_per_token() {
        let Some(device) = crate::detect_device().map(|d| d.device) else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };
        let (num_layers, num_blocks, block_size, num_kv_heads, head_dim) = (3, 256, 16, 8, 64);
        for bits in [TqBits::new(3), TqBits::new(4)] {
            let tq = build_tq_provision(
                &crate::MetalAllocator::new(device.clone()),
                TqPool {
                    is_global: &vec![true; num_layers],
                    num_blocks,
                    block_size,
                    num_kv_heads,
                    blocks_per_chunk: 128,
                },
                &TqCodebook::compute(head_dim, bits),
                None,
            );
            let len = |b: &Buffer| b.length();
            let allocated: usize = [&tq.packed_k, &tq.packed_v, &tq.norms_k, &tq.norms_v]
                .into_iter()
                .flatten()
                .map(len)
                .sum::<usize>()
                + len(&tq.scratch.k_data)
                + len(&tq.scratch.v_data);
            assert_eq!(
                allocated,
                num_blocks
                    * block_size
                    * kv_bytes_per_token(
                        KvCodec::TurboQuant(bits),
                        num_layers,
                        num_kv_heads,
                        head_dim,
                        2
                    ),
                "{bits:?}"
            );
        }
    }

    /// A provision hosted a scratch of its own geometry (a head's, its target's) stages through
    /// those very buffers and allocates no scratch of its own; one of another geometry — another
    /// block count, so other chunk offsets in the tables — is not its to use, and it builds its own.
    #[test]
    fn a_provision_stages_through_a_hosted_scratch_of_its_geometry() {
        let Some(device) = crate::detect_device().map(|d| d.device) else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };
        let codebook = TqCodebook::compute(64, TqBits::new(4));
        let is_global = [true];
        let pool = |num_blocks| TqPool {
            is_global: &is_global,
            num_blocks,
            block_size: 16,
            num_kv_heads: 2,
            blocks_per_chunk: 4,
        };
        let allocator = crate::MetalAllocator::new(device.clone());
        let target = build_tq_provision(&allocator, pool(8), &codebook, None);
        let same = |a: &Buffer, b: &Buffer| a.gpuAddress() == b.gpuAddress();
        let head = build_tq_provision(&allocator, pool(8), &codebook, Some(&target.scratch));
        for (h, t) in [
            (&head.scratch.k_table, &target.scratch.k_table),
            (&head.scratch.v_table, &target.scratch.v_table),
            (&head.scratch.k_data, &target.scratch.k_data),
            (&head.scratch.v_data, &target.scratch.v_data),
        ] {
            assert!(same(h, t), "the head stages through the target's scratch");
        }
        assert!(
            !same(&head.packed_k[0], &target.packed_k[0]),
            "its codes are its own"
        );
        let other = build_tq_provision(&allocator, pool(12), &codebook, Some(&target.scratch));
        assert!(!same(&other.scratch.k_data, &target.scratch.k_data));
        assert_eq!(other.scratch.bytes, 12 * 16 * 2 * 64 * SCRATCH_ELEM_BYTES);
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
        let vpw = vals_per_word(bits);
        let unpack = |words: &[u32]| {
            (0..dim)
                .map(|i| {
                    let w = i / vpw;
                    ((words[w] >> (((i % vpw) as u32) * bits)) & ((1u32 << bits) - 1)) as i16
                })
                .collect::<Vec<_>>()
        };
        let mut min_cos = f32::MAX;
        let mut tie_flips = 0usize;
        for slot in 0..n_slots {
            for kvh in 0..num_kv_heads {
                let b = base(slot, kvh);
                let (hp, hrec) = cb.host(&f32s(&orig[b..b + dim]));
                let sb = (slot * num_kv_heads + kvh) * cb.pdim;
                // The GPU and host quantizers are independent implementations of the same
                // math: the host norm accumulates in f64, the kernel's in an f32 tree, and
                // Metal's fast-math compiles the normalize its own way. An element whose
                // transform lands within rounding distance of a boundary can therefore
                // digitize one code apart — and which of those elements flips depends on
                // the GPU family's codegen (this passes as-is on M5, one vector flips on
                // M1 Max at dim 256). So the property is: every differing element is that
                // boundary tie (adjacent codes). A real defect — wrong table, stride or
                // sign application — moves codes arbitrarily far and fails here.
                for (i, (&g, &h)) in unpack(&gpu_packed[sb..sb + cb.pdim])
                    .iter()
                    .zip(&unpack(&hp))
                    .enumerate()
                {
                    if g != h {
                        assert_eq!(
                            (g - h).abs(),
                            1,
                            "dim {dim} bits {bits}: slot {slot} kvh {kvh} elem {i}: \
                             gpu code {g} vs host {h} — not a boundary tie"
                        );
                        tie_flips += 1;
                    }
                }
                // pool now holds host dequant
                min_cos = min_cos.min(cosine(&hrec, &f32s(&pool_out[b..b + dim])));
            }
        }
        let elements = n_slots * num_kv_heads * dim;
        println!(
            "tq_compress_paged dim {dim} bits {bits}: min cosine(pool, host dequant) = {min_cos:.5}, \
             boundary-tie code flips {tie_flips}/{elements} elements"
        );
        // Rounding noise flips only elements sitting within ~1e-5 relative of a boundary
        // (about 1 in 1e5 elements here). A rate three orders above that means the
        // kernel's arithmetic has drifted from the host math beyond rounding — a
        // systematic shift that the per-element ±1 tie check alone would let through.
        assert!(
            tie_flips * 1000 <= elements,
            "dim {dim} bits {bits}: {tie_flips}/{elements} codes differ from host — \
             beyond boundary-tie noise"
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

    /// FNV-1a over the bits of `tables`, in order.
    fn digest(tables: &[&[f32]]) -> u64 {
        let words = tables.iter().flat_map(|t| t.iter().map(|v| v.to_bits()));
        words.fold(0xcbf2_9ce4_8422_2325, |h, w| {
            (h ^ u64::from(w)).wrapping_mul(0x0100_0000_01b3)
        })
    }

    /// The baked codebook equals what `build_tq_provision` computed at load (a `PolarQuantizer` at
    /// the production seed, centroid midpoints, `packed_dim`), bit for bit, at every gate-model
    /// geometry; the digests pin those tables.
    #[test]
    fn baked_codebook_equals_load_time_tables() {
        let golden: [(usize, u32, u64); 4] = [
            (128, 3, 0xc6f921dfd2b14ed1),
            (128, 4, 0xc7badbedda5da1f3),
            (256, 4, 0x42e728170bcf8ff3),
            (512, 4, 0xcda751c0cdee0bf3),
        ];
        for (dim, bits, want) in golden {
            let cb = TqCodebook::compute(dim, TqBits::new(bits));
            let q = PolarQuantizer::new(dim, bits, TQ_CODEBOOK_SEED);
            let mids: Vec<f32> = q
                .centroids()
                .windows(2)
                .map(|w| (w[0] + w[1]) / 2.0)
                .collect();
            let bits_of = |t: &[f32]| t.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits_of(cb.signs), bits_of(q.signs()), "signs {dim}/{bits}");
            assert_eq!(
                bits_of(cb.centroids),
                bits_of(q.centroids()),
                "centroids {dim}/{bits}"
            );
            // The encoders' table: the midpoints they digitize against, then the centroids whose
            // length they correct each norm by.
            let (boundaries, levels) = cb.encode_table.split_at(mids.len());
            assert_eq!(
                bits_of(boundaries),
                bits_of(&mids),
                "boundaries {dim}/{bits}"
            );
            assert_eq!(
                bits_of(levels),
                bits_of(q.centroids()),
                "levels {dim}/{bits}"
            );
            assert_eq!(
                cb.packed_dim as usize,
                packed_dim(dim, bits),
                "packed_dim {dim}/{bits}"
            );
            let got = digest(&[cb.signs, cb.centroids, boundaries]);
            assert_eq!(got, want, "codebook digest {dim}/{bits}: {got:#018x}");
        }
    }
}

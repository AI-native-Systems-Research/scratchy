// SPDX-License-Identifier: Apache-2.0
//! Nemotron-H's standalone kernels against CPU references: `relu2`
//! (`activation.metal`, the non-gated MLP activation) and `kv_append`
//! (`kv_append.metal`, the paged-cache writer of its no-RoPE attention),
//! plain and with its folded-in TurboQuant encode.
//!
//! The baked kernels carry the constants the lowering bakes
//! (`aot::baked_build`). GPU tests — run with `--test-threads=1`
//! (standing rule).

use half::{bf16, f16};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_layers::turboquant::{PolarQuantizer, packed_dim, unpack_indices};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::Mtl4DispatchBatch;
use scratchy_target_metal::specialized_pipeline_cache::{PipelineKey, SpecializedPipelineCache};
use scratchy_target_metal::tape::ids::{
    BlockSize, BlocksPerChunk, ElementCount, HeadDim, NumKvHeads, RopePairOff, RotDim, TqCodeBits,
};
use scratchy_target_metal::tape::kernel_constants::{
    GeluConstants, KvAppendConstants, RopeTqConstants, TqCompressConstants, TqOffset, TqWriteback,
};

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;

/// A padding token's slot: the writer writes nothing for it.
const PAD: u32 = u32::MAX;
/// The spans "stored unrotated" flag a slot may carry: stripped before addressing.
const SPAN_BIT: u32 = 0x8000_0000;

#[derive(Clone, Copy, Debug)]
enum Dtype {
    F16,
    Bf16,
}

impl Dtype {
    const ALL: [Self; 2] = [Self::F16, Self::Bf16];

    fn tag(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    fn bits(self, x: f32) -> u16 {
        match self {
            Self::F16 => f16::from_f32(x).to_bits(),
            Self::Bf16 => bf16::from_f32(x).to_bits(),
        }
    }

    fn value(self, b: u16) -> f32 {
        match self {
            Self::F16 => f16::from_bits(b).to_f32(),
            Self::Bf16 => bf16::from_bits(b).to_f32(),
        }
    }
}

fn shared<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let bytes = std::mem::size_of_val(data);
    let buf = device
        .newBufferWithLength_options(bytes.max(16), MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    // SAFETY: `buf` holds at least `bytes` bytes and does not overlap `data`.
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buf.contents().as_ptr() as *mut u8,
            bytes,
        );
    }
    buf
}

fn read<T: Copy>(buf: &Buffer, n: usize) -> Vec<T> {
    // SAFETY: every `buf` read here was made by `shared` from at least `n` `T`s.
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const T, n).to_vec() }
}

fn tg(width: usize, height: usize) -> MTLSize {
    MTLSize {
        width,
        height,
        depth: 1,
    }
}

/// `relu2` at both dtypes over `n` elements, the dispatch rounded past `n` as the lowering's
/// `dispatch_1d` rounds it: every element below `n` is `relu(x)²` — EXACTLY, since a 16-bit
/// value's square is exact in f32 and the kernel rounds once, as mlx's in-dtype square does —
/// and none past `n` is written.
#[test]
fn relu2_matches_reference() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    let (n, padded) = (1000usize, 1024usize);
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.037).sin() * 6.0).collect();
    for dtype in Dtype::ALL {
        let symbol: &'static str = Box::leak(format!("relu2_{}", dtype.tag()).into_boxed_str());
        let constants = GeluConstants {
            elements: ElementCount(n as u32),
        };
        let key = PipelineKey::new("activation", symbol, constants.into());
        let pipeline = baked_build(&cache, &key).expect("relu2 pipeline");
        let bits: Vec<u16> = x.iter().map(|&v| dtype.bits(v)).collect();
        let input = shared(&device, &bits);
        let sentinel = dtype.bits(7.0);
        let out = shared(&device, &vec![sentinel; padded]);
        let Some(mut batch) = Mtl4DispatchBatch::begin(&device) else {
            eprintln!("skipping: no MTL4 queue");
            return;
        };
        batch.encode(
            &pipeline,
            &[(&out, 0), (&input, 1)],
            &[],
            &[],
            &[],
            tg(padded / 256, 1),
            tg(256, 1),
        );
        batch.commit(true);
        let got: Vec<u16> = read(&out, padded);
        for (i, &b) in bits.iter().enumerate() {
            let r = dtype.value(b).max(0.0);
            let want = dtype.bits(r * r);
            assert_eq!(got[i], want, "{symbol}[{i}]: x = {}", dtype.value(b));
        }
        assert!(
            got[n..].iter().all(|&g| g == sentinel),
            "{symbol} wrote past n"
        );
    }
}

/// One `kv_append` geometry.
struct Case {
    dtype: Dtype,
    head_dim: usize,
    num_kv_heads: usize,
    block_size: usize,
    blocks_per_chunk: usize,
    n_blocks: usize,
    /// One slot per token: [`PAD`] writes nothing; [`SPAN_BIT`] is stripped.
    slots: &'static [u32],
}

/// The pool a case writes, `[block][kv_head][token][head_dim]` with its chunk-address table
/// (`BLOCKS_PER_CHUNK` blocks per chunk), pre-filled with a marker so an unwritten row shows.
struct Pool {
    data: Buffer,
    table: Buffer,
}

impl Pool {
    const MARK: u16 = 0x7BFF;

    fn new(device: &Device, c: &Case) -> Self {
        let blk = c.num_kv_heads * c.block_size * c.head_dim;
        let data = shared(device, &vec![Self::MARK; c.n_blocks * blk]);
        let table: Vec<u64> = (0..c.n_blocks.div_ceil(c.blocks_per_chunk))
            .map(|ch| data.gpuAddress() + (ch * c.blocks_per_chunk * blk * 2) as u64)
            .collect();
        let table = shared(device, &table);
        Self { data, table }
    }
}

/// Run `c` on the GPU and compare both pools with the CPU reference: every non-padding token's
/// K and V rows, bit for bit, at `[block][kv_head][slot % block_size]`; every other row
/// untouched.
fn check(c: &Case) {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    let (hd, nkv, n_tok) = (c.head_dim, c.num_kv_heads, c.slots.len());
    let symbol: &'static str = Box::leak(format!("kv_append_{}", c.dtype.tag()).into_boxed_str());
    let constants = KvAppendConstants {
        head_dim: HeadDim(hd as u32),
        num_kv_heads: NumKvHeads(nkv as u32),
        block_size: BlockSize(c.block_size as u32),
        blocks_per_chunk: BlocksPerChunk(c.blocks_per_chunk as u32),
        tq: None,
    };
    let key = PipelineKey::new("kv_append", symbol, constants.into());
    let writer = baked_build(&cache, &key).expect("kv_append pipeline");

    let row = |t: usize, which: usize| -> Vec<u16> {
        (0..nkv * hd)
            .map(|e| {
                c.dtype
                    .bits(((t * 131 + e * 7 + which * 53) % 97) as f32 / 16.0 - 3.0)
            })
            .collect()
    };
    let k_rows: Vec<u16> = (0..n_tok).flat_map(|t| row(t, 0)).collect();
    let v_rows: Vec<u16> = (0..n_tok).flat_map(|t| row(t, 1)).collect();
    let (k, v) = (shared(&device, &k_rows), shared(&device, &v_rows));
    let slots = shared(&device, c.slots);
    let (pool_k, pool_v) = (Pool::new(&device, c), Pool::new(&device, c));
    let Some(mut batch) = Mtl4DispatchBatch::begin(&device) else {
        eprintln!("skipping: no MTL4 queue");
        return;
    };
    batch.encode(
        &writer,
        &[
            (&k, 0),
            (&v, 1),
            (&slots, 2),
            (&pool_k.table, 3),
            (&pool_v.table, 4),
        ],
        &[],
        &[],
        &[&pool_k.data, &pool_v.data],
        tg(n_tok, nkv),
        tg(hd, 1),
    );
    batch.commit(true);

    let elems = c.n_blocks * nkv * c.block_size * hd;
    let mut want_k = vec![Pool::MARK; elems];
    let mut want_v = vec![Pool::MARK; elems];
    for (t, &raw) in c.slots.iter().enumerate() {
        if raw == PAD {
            continue;
        }
        let slot = (raw & !SPAN_BIT) as usize;
        let (block, offset) = (slot / c.block_size, slot % c.block_size);
        for h in 0..nkv {
            let dst = ((block * nkv + h) * c.block_size + offset) * hd;
            let src = (t * nkv + h) * hd;
            want_k[dst..dst + hd].copy_from_slice(&k_rows[src..src + hd]);
            want_v[dst..dst + hd].copy_from_slice(&v_rows[src..src + hd]);
        }
    }
    assert_eq!(
        read::<u16>(&pool_k.data, elems),
        want_k,
        "K pool ({:?})",
        c.dtype
    );
    assert_eq!(
        read::<u16>(&pool_v.data, elems),
        want_v,
        "V pool ({:?})",
        c.dtype
    );
}

/// Nemotron-3's attention geometry (head_dim 128, 2 KV heads) at decode: one token per
/// sequence, each landing mid-block.
#[test]
fn kv_append_decode_nemotron_geometry() {
    for dtype in Dtype::ALL {
        check(&Case {
            dtype,
            head_dim: 128,
            num_kv_heads: 2,
            block_size: 16,
            blocks_per_chunk: 8,
            n_blocks: 8,
            slots: &[37, 5, 120],
        });
    }
}

/// A prefill bucket: consecutive slots crossing a block AND a chunk boundary (2 blocks per
/// chunk), a padding row that must write nothing, and a span-flagged slot that addresses as its
/// plain slot.
#[test]
fn kv_append_prefill_across_blocks_and_chunks() {
    for dtype in Dtype::ALL {
        check(&Case {
            dtype,
            head_dim: 32,
            num_kv_heads: 2,
            block_size: 16,
            blocks_per_chunk: 2,
            n_blocks: 6,
            slots: &[29, 30, 31, 32, 33, PAD, 70 | SPAN_BIT, 95],
        });
    }
}

/// The production codebook seed (`build_tq_provision(.., 42)`).
const TQ_SEED: u64 = 42;
/// A packed row neither run wrote keeps this norm.
const STALE_NORM: f32 = 7.5;

/// What a TurboQuant'd `kv_append` run leaves: both pools, both packed stores and their norms.
#[derive(PartialEq, Debug)]
struct Encoded {
    pool_k: Vec<u16>,
    pool_v: Vec<u16>,
    packed_k: Vec<u32>,
    packed_v: Vec<u32>,
    norms_k: Vec<f32>,
    norms_v: Vec<f32>,
}

/// `c` coded at `bits`: `fused`, the writer encoding the rows as it writes them (the
/// `KvEncoded` fold); else the plain writer, then `tq_compress_paged` over the rows it wrote —
/// the tape the fold replaces. Each row is the case's deterministic K/V rows.
fn run_coded(c: &Case, bits: u32, fused: bool) -> Option<(Encoded, Vec<u16>, Vec<u16>)> {
    let device = detect_device()?.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    let (hd, nkv, n_tok) = (c.head_dim, c.num_kv_heads, c.slots.len());
    let (head_dim, num_kv_heads) = (HeadDim(hd as u32), NumKvHeads(nkv as u32));
    let (block_size, blocks_per_chunk) = (
        BlockSize(c.block_size as u32),
        BlocksPerChunk(c.blocks_per_chunk as u32),
    );
    let tq = fused.then_some(RopeTqConstants {
        bits: TqCodeBits(bits),
        k_offset: TqOffset::None,
        v_offset: TqOffset::None,
    });
    let constants = KvAppendConstants {
        head_dim,
        num_kv_heads,
        block_size,
        blocks_per_chunk,
        tq,
    };
    let symbol: &'static str = Box::leak(format!("kv_append_{}", c.dtype.tag()).into_boxed_str());
    let key = PipelineKey::new("kv_append", symbol, constants.into());
    let writer = baked_build(&cache, &key).expect("coded kv_append pipeline");

    let row = |t: usize, which: usize| -> Vec<u16> {
        (0..nkv * hd)
            .map(|e| {
                let x = ((t * 131 + e * 7 + which * 53) % 97) as f32 / 16.0 - 3.0;
                c.dtype.bits(x * (1.0 + (e % 5) as f32 * 0.25))
            })
            .collect()
    };
    let k_rows: Vec<u16> = (0..n_tok).flat_map(|t| row(t, 0)).collect();
    let v_rows: Vec<u16> = (0..n_tok).flat_map(|t| row(t, 1)).collect();
    let (k, v) = (shared(&device, &k_rows), shared(&device, &v_rows));
    let slots = shared(&device, c.slots);
    let quant = PolarQuantizer::new(hd, bits, TQ_SEED);
    let boundaries: Vec<f32> = quant
        .centroids()
        .windows(2)
        .map(|w| (w[0] + w[1]) / 2.0)
        .collect();
    let (signs, centroids, bounds) = (
        shared(&device, quant.signs()),
        shared(&device, quant.centroids()),
        shared(&device, &boundaries),
    );
    let n_rows = c.n_blocks * c.block_size * nkv;
    let pdim = packed_dim(hd, bits);
    // Stale codes and norms in every row (a reused block): a row neither run writes stays as is.
    let stale: Vec<u32> = (0..n_rows * pdim)
        .map(|i| (i as u32).wrapping_mul(2654435761))
        .collect();
    let (packed_k, packed_v) = (shared(&device, &stale), shared(&device, &stale));
    let norms = vec![STALE_NORM; n_rows];
    let (norms_k, norms_v) = (shared(&device, &norms), shared(&device, &norms));
    let (pool_k, pool_v) = (Pool::new(&device, c), Pool::new(&device, c));
    // The compress's offset operands: unread under `TqOffset::None`.
    let unread = shared(&device, &[0u32; 4]);

    let mut binds = vec![
        (&k, 0),
        (&v, 1),
        (&slots, 2),
        (&pool_k.table, 3),
        (&pool_v.table, 4),
    ];
    if fused {
        // The lowering's `kv_writer_codec` set for an offset-free operand pair: 16-21.
        binds.extend([
            (&signs, 16),
            (&bounds, 17),
            (&packed_k, 18),
            (&norms_k, 19),
            (&packed_v, 20),
            (&norms_v, 21),
        ]);
    }
    let mut batch = Mtl4DispatchBatch::begin(&device)?;
    batch.encode(
        &writer,
        &binds,
        &[],
        &[],
        &[&pool_k.data, &pool_v.data],
        tg(n_tok, nkv),
        tg(hd, 1),
    );
    if !fused {
        batch.barrier();
        for (pool, packed, norms) in [
            (&pool_k, &packed_k, &norms_k),
            (&pool_v, &packed_v, &norms_v),
        ] {
            let constants = TqCompressConstants {
                head_dim,
                bits: TqCodeBits(bits),
                num_kv_heads,
                block_size,
                blocks_per_chunk,
                writeback: TqWriteback::Raw,
                offset: TqOffset::None,
                rot_dim: RotDim(0),
                pair_off: RopePairOff(0),
            };
            let compress = match c.dtype {
                Dtype::F16 => "tq_compress_paged",
                Dtype::Bf16 => "tq_compress_paged_bf16",
            };
            let key = PipelineKey::new("turboquant", compress, constants.into());
            let compress = baked_build(&cache, &key).expect("tq_compress_paged");
            batch.encode(
                &compress,
                &[
                    (&pool.table, 0),
                    (&slots, 1),
                    (&signs, 2),
                    (&bounds, 3),
                    (&centroids, 4),
                    (packed, 5),
                    (norms, 6),
                    (&slots, 16),
                    (&unread, 18),
                    (&unread, 19),
                    (&unread, 20),
                ],
                &[],
                &[],
                &[&pool.data],
                tg(n_tok, nkv),
                tg(hd, 1),
            );
        }
    }
    batch.commit(true);
    let pool_elems = c.n_blocks * nkv * c.block_size * hd;
    let written = Encoded {
        pool_k: read(&pool_k.data, pool_elems),
        pool_v: read(&pool_v.data, pool_elems),
        packed_k: read(&packed_k, n_rows * pdim),
        packed_v: read(&packed_v, n_rows * pdim),
        norms_k: read(&norms_k, n_rows),
        norms_v: read(&norms_v, n_rows),
    };
    Some((written, k_rows, v_rows))
}

/// `c` coded at `bits`: the fused encode writes the same bits as the writer then the compress,
/// pools included; and every written token's K and V rows encode as the CPU quantizer
/// (`PolarQuantizer`, the codec's reference) encodes them — the norm, and each code, a code
/// allowed to land one level over only where the f32 rotation puts it on a boundary.
fn check_coded(c: &Case, bits: u32) {
    let (Some((unfused, _, _)), Some((fused, k_rows, v_rows))) =
        (run_coded(c, bits, false), run_coded(c, bits, true))
    else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    for (what, a, b) in [
        ("packed K", &unfused.packed_k, &fused.packed_k),
        ("packed V", &unfused.packed_v, &fused.packed_v),
    ] {
        let diff = a.iter().zip(b).filter(|(x, y)| x != y).count();
        assert_eq!(diff, 0, "{what}: {diff} of {} words differ", a.len());
    }
    assert!(unfused == fused, "pools or norms differ");

    let (hd, nkv) = (c.head_dim, c.num_kv_heads);
    let quant = PolarQuantizer::new(hd, bits, TQ_SEED);
    let pdim = packed_dim(hd, bits);
    let (mut codes, mut off_by_one) = (0usize, 0usize);
    let mut encoded = vec![false; c.n_blocks * c.block_size * nkv];
    for (t, &raw) in c.slots.iter().enumerate() {
        if raw == PAD {
            continue;
        }
        let slot = (raw & !SPAN_BIT) as usize;
        for h in 0..nkv {
            let store = slot * nkv + h;
            encoded[store] = true;
            let src = (t * nkv + h) * hd;
            for (rows, packed, norms) in [
                (&k_rows, &fused.packed_k, &fused.norms_k),
                (&v_rows, &fused.packed_v, &fused.norms_v),
            ] {
                let x: Vec<f32> = rows[src..src + hd]
                    .iter()
                    .map(|&b| c.dtype.value(b))
                    .collect();
                let (want, norm) = quant.quantize(&x);
                let got = unpack_indices(&packed[store * pdim..][..pdim], bits, hd);
                assert!(
                    (norms[store] - norm).abs() <= 1e-5 * norm,
                    "token {t} head {h}: norm {} vs CPU {norm}",
                    norms[store]
                );
                for (g, w) in got.iter().zip(&want) {
                    codes += 1;
                    match g.abs_diff(*w) {
                        0 => {}
                        1 => off_by_one += 1,
                        d => panic!("token {t} head {h}: code {g} vs CPU {w} ({d} levels)"),
                    }
                }
            }
        }
    }
    assert!(codes > 0, "nothing encoded");
    assert!(
        off_by_one * 200 <= codes,
        "{off_by_one} of {codes} codes a level off the CPU quantizer's"
    );
    // A row no token writes keeps its stale norm: padding encodes nothing.
    for (store, &was_written) in encoded.iter().enumerate() {
        if !was_written {
            assert_eq!(
                fused.norms_k[store], STALE_NORM,
                "unwritten row {store} encoded"
            );
            assert_eq!(
                fused.norms_v[store], STALE_NORM,
                "unwritten row {store} encoded"
            );
        }
    }
}

/// The coded writer at Nemotron-3's geometry (head_dim 128, 2 KV heads), decode and a padded,
/// span-flagged prefill crossing blocks and chunks, at 4 and 3 bits.
#[test]
fn kv_append_turboquant_encode_matches_references() {
    for dtype in Dtype::ALL {
        for bits in [4, 3] {
            check_coded(
                &Case {
                    dtype,
                    head_dim: 128,
                    num_kv_heads: 2,
                    block_size: 16,
                    blocks_per_chunk: 8,
                    n_blocks: 8,
                    slots: &[37, 5, 120],
                },
                bits,
            );
            check_coded(
                &Case {
                    dtype,
                    head_dim: 128,
                    num_kv_heads: 2,
                    block_size: 16,
                    blocks_per_chunk: 2,
                    n_blocks: 6,
                    slots: &[29, 30, 31, 32, 33, PAD, 70 | SPAN_BIT, 95],
                },
                bits,
            );
        }
    }
}

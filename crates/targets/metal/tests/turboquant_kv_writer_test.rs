// SPDX-License-Identifier: Apache-2.0
//! The KV writer's folded-in TurboQuant encode (`MetalFusion::KvEncoded`): the writer baked with
//! `RopeTqConstants` against the tape the fold replaces — the plain writer, then
//! `tq_compress_paged` over the K and V rows it wrote. The packed codes, the norms, the pool and
//! the rotated queries must be the same bits.
//!
//! GPU tests — run with `--test-threads=1` (standing rule).

use half::{bf16, f16};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_layers::turboquant::{PolarQuantizer, packed_dim};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::Mtl4DispatchBatch;
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use scratchy_target_metal::tape::ids::{
    BlockSize, BlocksPerChunk, HeadDim, NumKvHeads, NumQHeads, RmsNormEps, RopePairOff, RotDim,
    TqCodeBits,
};
use scratchy_target_metal::tape::kernel_constants::{
    RopeAppendConstants, RopeAppendNormedConstants, RopeTqConstants, TqCompressConstants, TqOffset,
    TqWriteback,
};

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;

/// The production codebook seed (`build_tq_provision(.., 42)`).
const SEED: u64 = 42;
/// A padding token's slot: the writers write nothing for it.
const PAD: u32 = u32::MAX;
/// A span block's slot flag: its K is stored unrotated.
const SPAN_BIT: u32 = 0x8000_0000;
const BLOCK_SIZE: usize = 16;
const N_BLOCKS: usize = 8;
const MAX_POS: usize = 64;

#[derive(Clone, Copy)]
enum Dtype {
    F16,
    Bf16,
}

impl Dtype {
    fn bits(self, x: f32) -> u16 {
        match self {
            Dtype::F16 => f16::from_f32(x).to_bits(),
            Dtype::Bf16 => bf16::from_f32(x).to_bits(),
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Dtype::F16 => "f16",
            Dtype::Bf16 => "bf16",
        }
    }

    fn compress(self) -> &'static str {
        match self {
            Dtype::F16 => "tq_compress_paged",
            Dtype::Bf16 => "tq_compress_paged_bf16",
        }
    }
}

/// Which KV writer: `rope_append` (rotates K in place), or `rope_append_normed` (Gemma 4: per-head
/// RMSNorm of Q and K first; its K/V carry no offset).
#[derive(Clone, Copy)]
enum Writer {
    Plain,
    Normed,
}

#[derive(Clone, Copy)]
struct Case {
    writer: Writer,
    dtype: Dtype,
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    rot_dim: usize,
    pair_off: usize,
    bits: u32,
    blocks_per_chunk: usize,
    /// Qwen2's K/V projection biases, coded with each operand's offset removed.
    biased: bool,
    /// Rope-on-read: a slot with [`SPAN_BIT`] stores its K unrotated.
    spans: bool,
    /// Each token's slot.
    slots: &'static [u32],
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
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

/// A paged KV pool `[block][kv_head][token][head_dim]` and its chunk-address table.
struct Pool {
    data: Buffer,
    table: Buffer,
}

impl Pool {
    fn new(device: &Device, c: &Case) -> Self {
        let blk = c.num_kv_heads * BLOCK_SIZE * c.head_dim;
        let data = shared(device, &vec![0u16; N_BLOCKS * blk]);
        let table: Vec<u64> = (0..N_BLOCKS.div_ceil(c.blocks_per_chunk))
            .map(|ch| data.gpuAddress() + (ch * c.blocks_per_chunk * blk * 2) as u64)
            .collect();
        let table = shared(device, &table);
        Self { data, table }
    }
}

/// What a run leaves behind, as bits.
#[derive(PartialEq, Debug)]
struct Written {
    q: Vec<u16>,
    pool_k: Vec<u16>,
    pool_v: Vec<u16>,
    packed_k: Vec<u32>,
    packed_v: Vec<u32>,
    norms_k: Vec<u32>,
    norms_v: Vec<u32>,
}

/// Case `c`'s KV write: `fused`, the writer encoding as it writes; else the writer, then the
/// compress over its rows.
fn run(c: &Case, fused: bool) -> Option<Written> {
    let device = detect_device()?.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    let (hd, nq, nkv, n_tok) = (c.head_dim, c.num_q_heads, c.num_kv_heads, c.slots.len());
    let (k_offset, v_offset) = if c.biased {
        (TqOffset::RotatedBias, TqOffset::Bias)
    } else {
        (TqOffset::None, TqOffset::None)
    };
    let tq = fused.then_some(RopeTqConstants {
        bits: TqCodeBits(c.bits),
        k_offset,
        v_offset,
    });
    let rope_on_read = c.spans.then_some(1);
    let (head_dim, num_q_heads, num_kv_heads) = (
        HeadDim(hd as u32),
        NumQHeads(nq as u32),
        NumKvHeads(nkv as u32),
    );
    let (rot_dim, pair_off) = (RotDim(c.rot_dim as u32), RopePairOff(c.pair_off as u32));
    let (block_size, blocks_per_chunk) = (
        BlockSize(BLOCK_SIZE as u32),
        BlocksPerChunk(c.blocks_per_chunk as u32),
    );
    let (name, constants): (String, Vec<ConstantValue>) = match c.writer {
        Writer::Plain => (
            format!("rope_append_{}_specialized", c.dtype.tag()),
            RopeAppendConstants {
                head_dim,
                num_q_heads,
                num_kv_heads,
                rot_dim,
                block_size,
                blocks_per_chunk,
                pair_off,
                rope_on_read,
                tq,
            }
            .into(),
        ),
        Writer::Normed => (
            format!("rope_append_normed_{}_s_f16_specialized", c.dtype.tag()),
            RopeAppendNormedConstants {
                head_dim,
                num_q_heads,
                num_kv_heads,
                rot_dim,
                block_size,
                blocks_per_chunk,
                pair_off,
                rms_norm_eps: RmsNormEps(1e-6),
                weight_offset: 0.0,
                rope_on_read,
                tq,
            }
            .into(),
        ),
    };
    let name: &'static str = Box::leak(name.into_boxed_str());
    let writer = baked_build(&cache, &PipelineKey::new("rope", name, constants)).expect("writer");

    let mut rng = Lcg(11);
    let mut dt = |n: usize, scale: f32| -> Vec<u16> {
        (0..n).map(|_| c.dtype.bits(rng.next() * scale)).collect()
    };
    let q = shared(&device, &dt(n_tok * nq * hd, 2.0));
    let k = shared(&device, &dt(n_tok * nkv * hd, 2.0));
    let v = shared(&device, &dt(n_tok * nkv * hd, 2.0));
    let (kb, vb) = (
        shared(&device, &dt(nkv * hd, 6.0)),
        shared(&device, &dt(nkv * hd, 6.0)),
    );
    let gains = shared(
        &device,
        &(0..hd)
            .map(|d| f16::from_f32(0.5 + d as f32 / hd as f32).to_bits())
            .collect::<Vec<_>>(),
    );
    let half_rot = c.rot_dim / 2;
    let cos_sin: Vec<u16> = (0..MAX_POS)
        .flat_map(|p| {
            (0..c.rot_dim).map(move |i| {
                let theta = p as f32 * 10000f32.powf(-((i % half_rot) as f32) / half_rot as f32);
                if i < half_rot {
                    theta.cos()
                } else {
                    theta.sin()
                }
            })
        })
        .map(|x| c.dtype.bits(x))
        .collect();
    let cos_sin = shared(&device, &cos_sin);
    let positions: Vec<u32> = (0..n_tok as u32).map(|t| 5 + 7 * t).collect();
    let positions = shared(&device, &positions);
    let slots = shared(&device, c.slots);

    let quant = PolarQuantizer::new(hd, c.bits, SEED);
    let boundaries: Vec<f32> = quant
        .centroids()
        .windows(2)
        .map(|w| (w[0] + w[1]) / 2.0)
        .collect();
    let (signs, centroids) = (
        shared(&device, quant.signs()),
        shared(&device, quant.centroids()),
    );
    let bounds = shared(&device, &boundaries);
    let n_rows = N_BLOCKS * BLOCK_SIZE * nkv;
    let pdim = packed_dim(hd, c.bits);
    // Stale codes and norms in every row (a reused block): a row neither run writes stays as is.
    let stale: Vec<u32> = (0..n_rows * pdim)
        .map(|i| (i as u32).wrapping_mul(2654435761))
        .collect();
    let stale_norms = vec![7.5f32; n_rows];
    let (packed_k, packed_v) = (shared(&device, &stale), shared(&device, &stale));
    let (norms_k, norms_v) = (shared(&device, &stale_norms), shared(&device, &stale_norms));
    let (pool_k, pool_v) = (Pool::new(&device, c), Pool::new(&device, c));

    let mut binds = vec![
        (&q, 0),
        (&k, 1),
        (&v, 2),
        (&cos_sin, 3),
        (&positions, 4),
        (&slots, 5),
        (&pool_k.table, 6),
        (&pool_v.table, 7),
    ];
    if let Writer::Normed = c.writer {
        binds.extend([(&gains, 8), (&gains, 9)]);
    }
    if fused {
        binds.extend([
            (&signs, 16),
            (&bounds, 17),
            (&packed_k, 18),
            (&norms_k, 19),
            (&packed_v, 20),
            (&norms_v, 21),
            (&kb, 22),
            (&vb, 23),
            (&cos_sin, 24),
        ]);
    }
    let resident = [&pool_k.data, &pool_v.data];
    let mut batch = Mtl4DispatchBatch::begin(&device)?;
    batch.encode(
        &writer,
        &binds,
        &[],
        &[],
        &resident,
        tg(n_tok, nq),
        tg(hd, 1),
    );
    if !fused {
        batch.barrier();
        for (pool, packed, norms, bias, offset) in [
            (&pool_k, &packed_k, &norms_k, &kb, k_offset),
            (&pool_v, &packed_v, &norms_v, &vb, v_offset),
        ] {
            let constants = TqCompressConstants {
                head_dim,
                bits: TqCodeBits(c.bits),
                num_kv_heads,
                block_size,
                blocks_per_chunk,
                writeback: TqWriteback::Raw,
                offset,
                rot_dim,
                pair_off,
            };
            let key = PipelineKey::new("turboquant", c.dtype.compress(), constants.into());
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
                    (bias, 18),
                    (&cos_sin, 19),
                    (&positions, 20),
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

    let pool_elems = N_BLOCKS * BLOCK_SIZE * nkv * hd;
    Some(Written {
        q: read(&q, n_tok * nq * hd),
        pool_k: read(&pool_k.data, pool_elems),
        pool_v: read(&pool_v.data, pool_elems),
        packed_k: read(&packed_k, n_rows * pdim),
        packed_v: read(&packed_v, n_rows * pdim),
        norms_k: read(&norms_k, n_rows),
        norms_v: read(&norms_v, n_rows),
    })
}

fn check(c: Case) {
    let (Some(unfused), Some(fused)) = (run(&c, false), run(&c, true)) else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    // The run encoded something: every written token's K row has a fresh norm.
    let written = c.slots.iter().filter(|&&s| s != PAD).count();
    let fresh = unfused
        .norms_k
        .iter()
        .filter(|&&n| n != 7.5f32.to_bits())
        .count();
    assert_eq!(fresh, written * c.num_kv_heads, "rows encoded");
    for (what, a, b) in [
        ("packed K", &unfused.packed_k, &fused.packed_k),
        ("packed V", &unfused.packed_v, &fused.packed_v),
        ("K norms", &unfused.norms_k, &fused.norms_k),
        ("V norms", &unfused.norms_v, &fused.norms_v),
    ] {
        let diff = a.iter().zip(b).filter(|(x, y)| x != y).count();
        assert_eq!(diff, 0, "{what}: {diff} of {} words differ", a.len());
    }
    assert!(unfused == fused, "pool or rotated queries differ");
}

const LLAMA: Case = Case {
    writer: Writer::Plain,
    dtype: Dtype::Bf16,
    head_dim: 128,
    num_q_heads: 24,
    num_kv_heads: 8,
    rot_dim: 128,
    pair_off: 64,
    bits: 3,
    blocks_per_chunk: N_BLOCKS,
    biased: false,
    spans: false,
    slots: &[37],
};

#[test]
fn decode_llama_3b_bf16() {
    check(LLAMA);
}

#[test]
fn decode_llama_3b_f16() {
    check(Case {
        dtype: Dtype::F16,
        ..LLAMA
    });
}

#[test]
fn prefill_llama_3b_padded() {
    check(Case {
        slots: &[3, 17, 40, PAD, 99],
        ..LLAMA
    });
}

#[test]
fn head_dim_64_chunked_bits() {
    for bits in [2, 4] {
        check(Case {
            head_dim: 64,
            num_q_heads: 8,
            num_kv_heads: 2,
            rot_dim: 64,
            pair_off: 32,
            bits,
            blocks_per_chunk: 2,
            slots: &[5, 70, 127],
            ..LLAMA
        });
    }
}

#[test]
fn qwen2_7b_biased_kv() {
    check(Case {
        num_q_heads: 28,
        num_kv_heads: 4,
        biased: true,
        slots: &[9, 50],
        ..LLAMA
    });
}

#[test]
fn qwen2_7b_biased_kv_span_blocks() {
    check(Case {
        num_q_heads: 28,
        num_kv_heads: 4,
        biased: true,
        spans: true,
        slots: &[9 | SPAN_BIT, 50, PAD, 77 | SPAN_BIT],
        ..LLAMA
    });
}

const GEMMA4_GLOBAL: Case = Case {
    writer: Writer::Normed,
    dtype: Dtype::Bf16,
    head_dim: 512,
    num_q_heads: 16,
    num_kv_heads: 2,
    rot_dim: 128,
    pair_off: 256,
    bits: 3,
    blocks_per_chunk: N_BLOCKS,
    biased: false,
    spans: false,
    slots: &[21],
};

#[test]
fn decode_gemma4_global_head_dim_512() {
    check(GEMMA4_GLOBAL);
}

#[test]
fn prefill_gemma4_head_dim_256_f16_spans() {
    check(Case {
        dtype: Dtype::F16,
        head_dim: 256,
        num_q_heads: 8,
        num_kv_heads: 4,
        rot_dim: 256,
        pair_off: 128,
        spans: true,
        slots: &[2 | SPAN_BIT, 33, 64],
        ..GEMMA4_GLOBAL
    });
}

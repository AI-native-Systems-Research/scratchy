// SPDX-License-Identifier: Apache-2.0
//! A one-row decode attention that runs its KV writer (`ATTN_FOLD`, `MetalFusion::RopedAttention`)
//! leaves the same bits as the writer then the attention: the output, the roped query, the cache
//! and the packed store. The attention alone (no writer) leaves another output — the step's own
//! key matters — so the match is not vacuous. `attention_decode_gqa_tq` (a KV head's 8 query heads
//! together, its partials merged by `attention_via_cache_v2_combine`) leaves the query unroped.
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
    BlockSize, BlocksPerChunk, HeadDim, NumKvHeads, NumQHeads, RopePairOff, RotDim, TqCodeBits,
};
use scratchy_target_metal::tape::kernel_constants::{
    RopeAppendConstants, RopeTqConstants, TqOffset,
};

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;

/// The production codebook seed (`build_tq_provision(.., 42)`).
const SEED: u64 = 42;
const N_BLOCKS: usize = 12;
const BLOCK_SIZE: usize = 16;
const MAX_POS: usize = 256;
const SPAN_BIT: u32 = 0x8000_0000;

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
}

#[derive(Clone, Copy)]
struct Case {
    dtype: Dtype,
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    rot_dim: usize,
    pair_off: usize,
    /// The TurboQuant code width; `None`: a dense cache.
    bits: Option<u32>,
    /// Query heads per TurboQuant decode threadgroup.
    heads: usize,
    /// Qwen2's K/V projection biases, removed before encoding.
    biased: bool,
    /// Spans: the step's block holds K unrotated.
    spans: bool,
    /// The step's slot is the write-skip sentinel: the writer writes nothing.
    write_skipped: bool,
    /// Keys including the step's own.
    ctx: usize,
    blocks_per_chunk: usize,
    /// `attention_decode_gqa_tq` over this many threadgroups a KV head.
    gqa: Option<usize>,
    /// RoPE base (theta). GLM-4.5 uses 1e6, not the 10000 default.
    theta: f32,
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
    let bytes = std::mem::size_of_val(data).max(16);
    let b = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("buffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            b.contents().as_ptr() as *mut u8,
            std::mem::size_of_val(data),
        )
    };
    b
}

fn read<T: Copy>(buf: &Buffer, n: usize) -> Vec<T> {
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const T, n) }.to_vec()
}

fn tg(width: usize, height: usize) -> MTLSize {
    MTLSize {
        width,
        height,
        depth: 1,
    }
}

/// A paged KV pool `[block][kv_head][token][head_dim]` holding earlier keys, and its chunk table.
struct Pool {
    data: Buffer,
    table: Buffer,
}

impl Pool {
    fn new(device: &Device, c: &Case, seed: u64) -> Self {
        let blk = c.num_kv_heads * BLOCK_SIZE * c.head_dim;
        let mut rng = Lcg(seed);
        let elems: Vec<u16> = (0..N_BLOCKS * blk)
            .map(|_| c.dtype.bits(rng.next() * 2.0))
            .collect();
        let data = shared(device, &elems);
        let table: Vec<u64> = (0..N_BLOCKS.div_ceil(c.blocks_per_chunk))
            .map(|ch| data.gpuAddress() + (ch * c.blocks_per_chunk * blk * 2) as u64)
            .collect();
        let table = shared(device, &table);
        Self { data, table }
    }
}

/// What a run leaves behind, as bits.
#[derive(PartialEq, Debug)]
struct Left {
    out: Vec<u16>,
    q: Vec<u16>,
    pool_k: Vec<u16>,
    pool_v: Vec<u16>,
    packed_k: Vec<u32>,
    packed_v: Vec<u32>,
    norms_k: Vec<u32>,
    norms_v: Vec<u32>,
}

#[derive(Clone, Copy, PartialEq)]
enum Run {
    /// The writer, then the attention.
    Writer,
    /// The attention running its writer.
    Fold,
    /// The attention alone.
    Alone,
}

fn run(c: &Case, how: Run) -> Option<Left> {
    let device = detect_device()?.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    let (hd, nq, nkv) = (c.head_dim, c.num_q_heads, c.num_kv_heads);
    let max_blocks = c.ctx.div_ceil(BLOCK_SIZE);
    // Logical block l of the sequence lives in physical block (5l + 1) mod N_BLOCKS
    // (stride 5 coprime to 12 → injective for any ctx the pool holds: the fold's
    // tail-slot write must never clobber a live block).
    let phys = |l: usize| (5 * l + 1) % N_BLOCKS;
    let tail_block = (c.ctx - 1) / BLOCK_SIZE;
    let span = |l: usize| {
        if c.spans && l == tail_block {
            SPAN_BIT
        } else {
            0
        }
    };
    let block_table: Vec<u32> = (0..max_blocks).map(|l| phys(l) as u32 | span(l)).collect();
    let slot = if c.write_skipped {
        u32::MAX
    } else {
        (phys(tail_block) * BLOCK_SIZE + (c.ctx - 1) % BLOCK_SIZE) as u32 | span(tail_block)
    };

    let mut rng = Lcg(7);
    let mut dt = |n: usize, scale: f32| -> Vec<u16> {
        (0..n).map(|_| c.dtype.bits(rng.next() * scale)).collect()
    };
    let q = shared(&device, &dt(nq * hd, 2.0));
    let k = shared(&device, &dt(nkv * hd, 2.0));
    let v = shared(&device, &dt(nkv * hd, 2.0));
    let (kb, vb) = (
        shared(&device, &dt(nkv * hd, 6.0)),
        shared(&device, &dt(nkv * hd, 6.0)),
    );
    let half_rot = c.rot_dim / 2;
    let cos_sin: Vec<u16> = (0..MAX_POS)
        .flat_map(|p| {
            (0..c.rot_dim).map(move |i| {
                let theta = p as f32 * c.theta.powf(-((i % half_rot) as f32) / half_rot as f32);
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
    let positions = shared(&device, &[(c.ctx - 1) as u32]);
    let slots = shared(&device, &[slot]);
    let seq_used = shared(&device, &[c.ctx as u32]);
    let block_table = shared(&device, &block_table);
    let (pool_k, pool_v) = (Pool::new(&device, c, 21), Pool::new(&device, c, 22));

    let bits = c.bits.unwrap_or(4);
    let quant = PolarQuantizer::new(hd, bits, SEED);
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
    let n_rows = N_BLOCKS * BLOCK_SIZE * nkv;
    let pdim = packed_dim(hd, bits);
    // Earlier keys' codes and norms: any bits, the same in every run.
    let stale: Vec<u32> = (0..n_rows * pdim)
        .map(|i| (i as u32).wrapping_mul(2654435761))
        .collect();
    let stale_norms: Vec<f32> = (0..n_rows).map(|i| 0.5 + (i % 7) as f32 / 4.0).collect();
    let (packed_k, packed_v) = (shared(&device, &stale), shared(&device, &stale));
    let (norms_k, norms_v) = (shared(&device, &stale_norms), shared(&device, &stale_norms));
    let out = shared(&device, &vec![0u16; nq * hd]);

    let (k_offset, v_offset) = match c.biased {
        true => (TqOffset::RotatedBias, TqOffset::Bias),
        false => (TqOffset::None, TqOffset::None),
    };
    let writer_constants: Vec<ConstantValue> = RopeAppendConstants {
        head_dim: HeadDim(hd as u32),
        num_q_heads: NumQHeads(nq as u32),
        num_kv_heads: NumKvHeads(nkv as u32),
        rot_dim: RotDim(c.rot_dim as u32),
        block_size: BlockSize(BLOCK_SIZE as u32),
        blocks_per_chunk: BlocksPerChunk(c.blocks_per_chunk as u32),
        pair_off: RopePairOff(c.pair_off as u32),
        rope_on_read: c.spans.then_some(1),
        tq: c.bits.map(|b| RopeTqConstants {
            bits: TqCodeBits(b),
            k_offset,
            v_offset,
        }),
    }
    .into();
    let writer_name: &'static str =
        Box::leak(format!("rope_append_{}_specialized", c.dtype.tag()).into_boxed_str());
    let writer = baked_build(
        &cache,
        &PipelineKey::new("rope", writer_name, writer_constants),
    )
    .expect("writer");

    let mut attn_constants = vec![
        ConstantValue::uint(0, hd as u32),
        ConstantValue::uint(1, nq as u32),
        ConstantValue::uint(2, nkv as u32),
        ConstantValue::float(3, 1.0 / (hd as f32).sqrt()),
        ConstantValue::uint(4, BLOCK_SIZE as u32),
        ConstantValue::uint(5, max_blocks as u32),
        ConstantValue::uint(6, c.blocks_per_chunk as u32),
        ConstantValue::int(7, 0),
    ];
    if c.spans {
        attn_constants.extend([
            ConstantValue::uint(8, c.rot_dim as u32),
            ConstantValue::uint(9, c.pair_off as u32),
            ConstantValue::uint(10, 1),
        ]);
    }
    let heads = match (c.bits, c.gqa) {
        (Some(b), Some(splits)) => {
            attn_constants.extend([
                ConstantValue::uint(13, b),
                ConstantValue::uint(18, splits as u32),
            ]);
            1
        }
        (Some(b), None) => {
            attn_constants.push(ConstantValue::uint(13, b));
            if c.biased {
                attn_constants.extend([ConstantValue::uint(14, 1), ConstantValue::uint(15, 1)]);
            }
            attn_constants.push(ConstantValue::uint(16, c.heads as u32));
            c.heads
        }
        (None, _) => 1,
    };
    if how == Run::Fold {
        attn_constants.extend([
            ConstantValue::uint(19, c.rot_dim as u32),
            ConstantValue::uint(20, c.pair_off as u32),
        ]);
    }
    let kernel = match c.gqa {
        Some(_) => "attention_decode_gqa_tq",
        None => "attention_via_cache_v2",
    };
    let leak = |name: String| -> &'static str { Box::leak(name.into_boxed_str()) };
    let attn_name = leak(format!("{kernel}_{}_specialized", c.dtype.tag()));
    let combine_name = leak(format!(
        "attention_via_cache_v2_combine_{}_specialized",
        c.dtype.tag()
    ));
    let combine = baked_build(
        &cache,
        &PipelineKey::new("attention", combine_name, attn_constants.clone()),
    )
    .expect("combine");
    let attention = baked_build(
        &cache,
        &PipelineKey::new("attention", attn_name, attn_constants),
    )
    .expect("attention");
    // NaN: the combine reads only the partials the attention stored.
    let partials = shared(&device, &vec![f32::NAN; nq * c.gqa.unwrap_or(1) * (hd + 2)]);

    let resident = [&pool_k.data, &pool_v.data];
    let mut batch = Mtl4DispatchBatch::begin(&device)?;
    if how == Run::Writer {
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
        if c.bits.is_some() {
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
        batch.encode(&writer, &binds, &[], &[], &resident, tg(1, nq), tg(hd, 1));
        batch.barrier();
    }
    let mut binds = vec![
        (&out, 0),
        (&q, 1),
        (&seq_used, 2),
        (&block_table, 3),
        (&pool_k.table, 4),
        (&pool_v.table, 5),
        (&cos_sin, 6),
        (&packed_k, 7),
        (&packed_v, 8),
        (&norms_k, 9),
        (&norms_v, 10),
        (&signs, 11),
        (&centroids, 12),
        (&slots, 13),
        (&kb, 14),
        (&vb, 15),
        (&partials, 16),
    ];
    if how == Run::Fold {
        binds.extend([
            (&k, 17),
            (&v, 18),
            (&positions, 19),
            (&bounds, 20),
            (&cos_sin, 21),
            (&cos_sin, 22),
        ]);
    }
    match c.gqa {
        Some(splits) => {
            let grid = MTLSize {
                width: 1,
                height: nkv,
                depth: splits,
            };
            batch.encode(&attention, &binds, &[], &[], &resident, grid, tg(hd, 1));
            batch.barrier();
            let binds = [(&out, 0), (&signs, 11), (&partials, 16)];
            batch.encode(&combine, &binds, &[], &[], &[], tg(1, nq), tg(32, 1));
        }
        None => batch.encode(
            &attention,
            &binds,
            &[],
            &[],
            &resident,
            tg(1, nq / heads),
            tg(1024, 1),
        ),
    }
    batch.commit(true);

    let pool_elems = N_BLOCKS * BLOCK_SIZE * nkv * hd;
    Some(Left {
        out: read(&out, nq * hd),
        q: read(&q, nq * hd),
        pool_k: read(&pool_k.data, pool_elems),
        pool_v: read(&pool_v.data, pool_elems),
        packed_k: read(&packed_k, n_rows * pdim),
        packed_v: read(&packed_v, n_rows * pdim),
        norms_k: read(&norms_k, n_rows),
        norms_v: read(&norms_v, n_rows),
    })
}

fn check(name: &str, c: Case) {
    let Some(writer) = run(&c, Run::Writer) else {
        eprintln!("{name}: no Metal device, skipped");
        return;
    };
    let fold = run(&c, Run::Fold).expect("device");
    let alone = run(&c, Run::Alone).expect("device");
    assert!(
        writer.out.iter().any(|&b| b != 0),
        "{name}: an all-zero output"
    );
    assert_ne!(
        writer.out, alone.out,
        "{name}: the step's own key changes nothing"
    );
    match c.gqa {
        None => assert_eq!(fold.q, writer.q, "{name}: roped query"),
        Some(_) => assert_eq!(fold.q, alone.q, "{name}: the query left unroped"),
    }
    assert_eq!(fold.pool_k, writer.pool_k, "{name}: K cache");
    assert_eq!(fold.pool_v, writer.pool_v, "{name}: V cache");
    assert_eq!(fold.packed_k, writer.packed_k, "{name}: packed K");
    assert_eq!(fold.packed_v, writer.packed_v, "{name}: packed V");
    assert_eq!(fold.norms_k, writer.norms_k, "{name}: K norms");
    assert_eq!(fold.norms_v, writer.norms_v, "{name}: V norms");
    assert_eq!(fold.out, writer.out, "{name}: output");
}

/// Llama 3.2 3B's attention geometry.
fn llama(dtype: Dtype, bits: Option<u32>, heads: usize) -> Case {
    Case {
        dtype,
        head_dim: 128,
        num_q_heads: 24,
        num_kv_heads: 8,
        rot_dim: 128,
        pair_off: 64,
        bits,
        heads,
        biased: false,
        spans: false,
        write_skipped: false,
        ctx: 41,
        blocks_per_chunk: 4,
        gqa: None,
        theta: 10000.0,
    }
}

#[test]
fn llama_dense_bf16() {
    check("llama dense bf16", llama(Dtype::Bf16, None, 1));
}

#[test]
fn llama_dense_f16() {
    check("llama dense f16", llama(Dtype::F16, None, 1));
}

#[test]
fn llama_turboquant_every_head_count() {
    for heads in [1, 3] {
        for bits in [3, 4] {
            check(
                "llama turboquant bf16",
                llama(Dtype::Bf16, Some(bits), heads),
            );
        }
    }
}

#[test]
fn qwen2_biased_turboquant() {
    // Qwen2.5 3B: 16 query heads over 2 KV heads, K/V biased.
    let c = Case {
        num_q_heads: 16,
        num_kv_heads: 2,
        biased: true,
        ..llama(Dtype::Bf16, Some(4), 8)
    };
    check("qwen2 biased turboquant", c);
    check("qwen2 biased turboquant, 2 heads", Case { heads: 2, ..c });
}

#[test]
fn spans_tail_stored_unrotated() {
    check(
        "spans dense",
        Case {
            spans: true,
            ..llama(Dtype::F16, None, 1)
        },
    );
    check(
        "spans turboquant",
        Case {
            spans: true,
            ..llama(Dtype::Bf16, Some(4), 3)
        },
    );
}

#[test]
fn write_skipped_tail() {
    check(
        "skipped dense",
        Case {
            write_skipped: true,
            ..llama(Dtype::Bf16, None, 1)
        },
    );
    check(
        "skipped turboquant",
        Case {
            write_skipped: true,
            ..llama(Dtype::Bf16, Some(4), 3)
        },
    );
}

/// A KV head's 8 query heads together (`attention_decode_gqa_tq`), whole and over 3 threadgroups:
/// the step's own key (the 41st) falls to the last, which reads it from its own copy of the row
/// while the first writes it to the cache and the packed store.
#[test]
fn gqa_turboquant() {
    let gqa = |splits| Case {
        num_q_heads: 16,
        num_kv_heads: 2,
        gqa: Some(splits),
        ..llama(Dtype::Bf16, Some(4), 1)
    };
    for splits in [1, 3] {
        check("gqa turboquant", gqa(splits));
        check(
            "gqa spans turboquant",
            Case {
                spans: true,
                ..gqa(splits)
            },
        );
        check(
            "gqa skipped turboquant",
            Case {
                write_skipped: true,
                ..gqa(splits)
            },
        );
    }
}

#[test]
fn partial_rope_at_a_block_edge() {
    // Rotary dim 64 of 128 (pairs within the first half), the step's key opening a block.
    check(
        "partial rope",
        Case {
            rot_dim: 64,
            pair_off: 32,
            ctx: 33,
            ..llama(Dtype::Bf16, Some(4), 3)
        },
    );
}

#[test]
fn glm45_air_dense() {
    // GLM-4.5-Air decode: 96 query heads over 8 KV heads (GQA 12:1),
    // head_dim 128, partial rope (rot 64, pairs within the first half),
    // theta 1e6, dense KV. The span block's K rides unrotated and the
    // fold re-ropes it (rope_on_read=1 in the production M1 constants).
    for (ctx, name) in [
        (41usize, "glm45 air dense"),
        (129usize, "glm45 air dense, ctx 129"),
    ] {
        check(
            name,
            Case {
                num_q_heads: 96,
                num_kv_heads: 8,
                rot_dim: 64,
                pair_off: 32,
                theta: 1e6,
                spans: true,
                ctx,
                ..llama(Dtype::Bf16, None, 1)
            },
        );
    }
}

#[test]
fn glm45_air_dense_unbiased_no_spans() {
    // The same geometry with the plain write-roped cache (spans off),
    // isolating the rope-on-read leg from the GQA ratio.
    check(
        "glm45 air plain",
        Case {
            num_q_heads: 96,
            num_kv_heads: 8,
            rot_dim: 64,
            pair_off: 32,
            theta: 1e6,
            ..llama(Dtype::Bf16, None, 1)
        },
    );
}

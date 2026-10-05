// SPDX-License-Identifier: Apache-2.0
//! Parity of TurboQuant attention — which reads the packed KV store, never a
//! dequantized context — against a host f32 decode of the very same codes.
//!
//! - Decode (one query per sequence): `attention_via_cache_v2` with
//!   `ATTN_TQ_BITS`, reading the packed store in the codebook domain, each
//!   threadgroup serving the production `TqDecodeHeads` query heads; or
//!   `attention_decode_gqa_tq`, each KV head's 8 query heads together over
//!   `ATTN_SPLITS` threadgroups, merged by `attention_via_cache_v2_combine`.
//! - Prefill (a chunk of queries per sequence): `tq_stage_rotated` stages K and
//!   V in the codebook's rotated domain, `tq_rotate_rows` rotates q in and the
//!   output back, and the production paged prefill attention runs unchanged
//!   in between.
//!
//! Each case quantizes the context with the production `tq_compress_paged`
//! kernel into packed stores whose every other slot holds stale garbage (a
//! reused block), and the attention's scratch is NaN everywhere but the step's
//! own (written) keys — so reading anything but the packed store and those keys
//! poisons the output.
//!
//! Also (fp16 KV): each paged prefill attention a step with several sequences
//! runs, against f32 attention of every sequence alone.
//!
//! GPU tests — run with `--test-threads=1` (standing rule).

use half::{bf16, f16};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_layers::turboquant::{PolarQuantizer, packed_dim, unpack_indices};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::aot::baked_kernels;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::{Mtl4DispatchBatch, Pipeline};
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use scratchy_target_metal::steel_paged::{nax_paged_kernel, steel_paged_symbol};
use scratchy_target_metal::tape::ids::{
    BlockSize, BlocksPerChunk, HeadDim, NumKvHeads, NumQHeads, RopePairOff, RotDim, TqCodeBits,
    TqDecodeHeads,
};
use scratchy_target_metal::tape::kernel_constants::{
    TqCompressConstants, TqOffset, TqStagePass, TqWriteback,
};
use scratchy_target_metal::targets::is_nax_capable;

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;

/// The production codebook seed (`build_tq_provision(.., 42)`).
const SEED: u64 = 42;
const WRITE_SKIP: u32 = u32::MAX;
const SPAN_BIT: u32 = 0x8000_0000;

#[derive(Clone, Copy, PartialEq)]
enum Dtype {
    F16,
    Bf16,
}

impl Dtype {
    fn round(self, x: f32) -> f32 {
        match self {
            Dtype::F16 => f16::from_f32(x).to_f32(),
            Dtype::Bf16 => bf16::from_f32(x).to_f32(),
        }
    }
    fn bits(self, x: f32) -> u16 {
        match self {
            Dtype::F16 => f16::from_f32(x).to_bits(),
            Dtype::Bf16 => bf16::from_f32(x).to_bits(),
        }
    }
    fn value(self, b: u16) -> f32 {
        match self {
            Dtype::F16 => f16::from_bits(b).to_f32(),
            Dtype::Bf16 => bf16::from_bits(b).to_f32(),
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
    /// One output ulp at 1.0: the rounding the stored result carries.
    fn ulp(self) -> f32 {
        match self {
            Dtype::F16 => 1.0 / 1024.0,
            Dtype::Bf16 => 1.0 / 128.0,
        }
    }
}

/// NeoX rope-on-read geometry.
#[derive(Clone, Copy)]
struct Rope {
    rot_dim: usize,
    pair_off: usize,
    /// The decode kernel's co-resident lane layout (slot 12).
    coresident: bool,
}

#[derive(Clone)]
struct Case {
    name: &'static str,
    dtype: Dtype,
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    bits: u32,
    block_size: usize,
    blocks_per_chunk: usize,
    attn_scale: f32,
    window: i32,
    rope: Option<Rope>,
    /// Per sequence: (context length including the step's keys, the step's
    /// new keys = its queries). All-1 new counts is a decode step.
    seqs: Vec<(usize, usize)>,
    /// Logical blocks (all sequences) whose K is stored unrotated (spans).
    span_blocks: Vec<usize>,
    /// Each sequence's first new key sits in a reused span block: its slot is
    /// the write-skip sentinel and the key was quantized by an earlier request.
    first_new_write_skipped: bool,
    /// Qwen2's K/V projection biases, `bias` times the signal's spread on a
    /// few channels: K is cached as `signal + R_t·b_k` (unrotated `b_k` in a
    /// span block), V as `signal + b_v`.
    bias: Option<f32>,
    /// Query heads per decode threadgroup (`ATTN_TQ_HEADS`); `check` runs
    /// every count `TqDecodeHeads` can pick for the geometry.
    decode_heads: u32,
    /// A decode by `attention_decode_gqa_tq` over this many threadgroups a KV head.
    gqa: Option<u32>,
    /// gpt-oss attention sinks: `Some(scale)` gives every query head one
    /// extra softmax column — a logit drawn at that scale, UNSCALED by
    /// `attn_scale`, contributing no V, exactly as the arch defines them;
    /// `None` (every non-sink case) leaves the column off, byte-identical.
    sinks: Option<f32>,
}

/// Channels of each KV head that carry the large bias.
const BIASED_CHANNELS: [usize; 4] = [3, 17, 70, 101];

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn gauss(&mut self) -> f32 {
        self.next() + self.next() + self.next()
    }
}

fn shared<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let bytes = std::mem::size_of_val(data);
    let buf = device
        .newBufferWithLength_options(bytes.max(16), MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
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
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const T, n).to_vec() }
}

fn tg(width: usize, height: usize, depth: usize) -> MTLSize {
    MTLSize {
        width,
        height,
        depth,
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// A paged KV buffer `[block][kv_head][token][head_dim]` of the model dtype
/// plus its chunk-address table.
struct Pool {
    data: Buffer,
    table: Buffer,
}

/// Everything a case's GPU runs share.
struct Fixture {
    c: Case,
    n_blocks: usize,
    max_blocks: usize,
    /// Row per sequence, `max_blocks` wide, span bit applied.
    block_table: Vec<u32>,
    /// Physical slot of each sequence's key `t`.
    slot: Vec<Vec<usize>>,
    /// Plain K/V per sequence, `[t][kv_head][head_dim]`, dtype-rounded; span
    /// blocks hold K unrotated, as the writer leaves them.
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    /// Queries, `[token][q_head][head_dim]`, the sequences' new tokens in order.
    q: Vec<f32>,
    cu_seqlens: Vec<u32>,
    cos_sin: Vec<u16>,
    /// K / V projection biases `[kv_head][head_dim]` (zero without `c.bias`).
    kb: Vec<f32>,
    vb: Vec<f32>,
    /// Per-query-head sink logits, dtype-rounded (empty without `c.sinks`).
    sink: Vec<f32>,
}

impl Fixture {
    fn new(c: &Case) -> Self {
        let mut rng = Lcg(0x5eed ^ c.head_dim as u64 ^ ((c.bits as u64) << 20));
        let blocks_per_seq: Vec<usize> = c
            .seqs
            .iter()
            .map(|&(l, _)| l.div_ceil(c.block_size))
            .collect();
        let max_blocks = blocks_per_seq.iter().copied().max().unwrap_or(1).max(1);
        let n_blocks = blocks_per_seq.iter().sum::<usize>() + 3;
        // A non-identity physical placement: stride through the pool.
        let stride = (0..)
            .map(|s| 2 * s + 3)
            .find(|s| gcd(*s, n_blocks) == 1)
            .unwrap();
        let mut next_phys = (0..n_blocks).map(|i| (i * stride + 1) % n_blocks);
        let mut block_table = vec![0u32; c.seqs.len() * max_blocks];
        let mut slot = Vec::new();
        for (s, &nb) in blocks_per_seq.iter().enumerate() {
            let phys: Vec<usize> = (0..nb).map(|_| next_phys.next().unwrap()).collect();
            for (lb, &pb) in phys.iter().enumerate() {
                let span = if c.span_blocks.contains(&lb) {
                    SPAN_BIT
                } else {
                    0
                };
                block_table[s * max_blocks + lb] = pb as u32 | span;
            }
            slot.push(
                (0..c.seqs[s].0)
                    .map(|t| phys[t / c.block_size] * c.block_size + t % c.block_size)
                    .collect(),
            );
        }
        let per_tok = c.num_kv_heads * c.head_dim;
        let mut vecs =
            |n: usize| -> Vec<f32> { (0..n).map(|_| c.dtype.round(rng.gauss())).collect() };
        let mut k: Vec<Vec<f32>> = c.seqs.iter().map(|&(l, _)| vecs(l * per_tok)).collect();
        let mut v: Vec<Vec<f32>> = c.seqs.iter().map(|&(l, _)| vecs(l * per_tok)).collect();
        let n_q: usize = c.seqs.iter().map(|&(_, n)| n).sum();
        let q = vecs(n_q * c.num_q_heads * c.head_dim);
        let mut bias = |scale: f32| -> Vec<f32> {
            let mut b = vecs(per_tok);
            for h in 0..c.num_kv_heads {
                for &d in BIASED_CHANNELS.iter().filter(|&&d| d < c.head_dim) {
                    b[h * c.head_dim + d] = c.dtype.round(b[h * c.head_dim + d] * scale);
                }
            }
            b
        };
        let (kb, vb) = match c.bias {
            Some(scale) => (bias(scale), bias(scale)),
            None => (vec![0.0; per_tok], vec![0.0; per_tok]),
        };
        let cu_seqlens = std::iter::once(0)
            .chain(c.seqs.iter().scan(0u32, |acc, &(_, n)| {
                *acc += n as u32;
                Some(*acc)
            }))
            .collect();
        let max_len = c.seqs.iter().map(|&(l, _)| l).max().unwrap_or(0);
        let cos_sin = match c.rope {
            None => vec![0u16; 1],
            Some(r) => {
                let half = r.rot_dim / 2;
                (0..max_len)
                    .flat_map(|pos| {
                        let ang = move |d: usize| {
                            pos as f32 * 10000f32.powf(-(2.0 * d as f32) / r.rot_dim as f32)
                        };
                        let cs: Vec<f32> = (0..half)
                            .map(|d| ang(d).cos())
                            .chain((0..half).map(|d| ang(d).sin()))
                            .collect();
                        cs
                    })
                    .map(|x| c.dtype.bits(x))
                    .collect()
            }
        };
        // A second, independent stream: the K/V/Q draws above stay
        // byte-identical for every existing case.
        let sink = c.sinks.map_or(vec![], |scale| {
            let mut rng = Lcg(0x51e5u64 ^ (c.num_q_heads as u64) ^ ((c.head_dim as u64) << 24));
            (0..c.num_q_heads)
                .map(|_| c.dtype.round(scale * rng.gauss()))
                .collect()
        });
        let mut f = Self {
            c: c.clone(),
            n_blocks,
            max_blocks,
            block_table,
            slot,
            k: vec![],
            v: vec![],
            q,
            cu_seqlens,
            cos_sin,
            kb,
            vb,
            sink,
        };
        if c.bias.is_some() {
            for (s, &(len, _)) in c.seqs.iter().enumerate() {
                for t in 0..len {
                    for h in 0..c.num_kv_heads {
                        let row = (t * c.num_kv_heads + h) * c.head_dim;
                        let (kb, vb) = (f.offset(s, t, h, false), f.offset(s, t, h, true));
                        for d in 0..c.head_dim {
                            k[s][row + d] = c.dtype.round(k[s][row + d] + kb[d]);
                            v[s][row + d] = c.dtype.round(v[s][row + d] + vb[d]);
                        }
                    }
                }
            }
        }
        (f.k, f.v) = (k, v);
        f
    }

    /// The bias key `t` of sequence `s` carries in KV head `h`: V's as-is, K's
    /// rotated to `t` as the writer rotates the key — unrotated in a span
    /// block, whose K is stored unrotated.
    fn offset(&self, s: usize, t: usize, h: usize, is_v: bool) -> Vec<f32> {
        let (c, hd) = (&self.c, self.c.head_dim);
        let b = &(if is_v { &self.vb } else { &self.kb })[h * hd..][..hd];
        match c.rope {
            Some(r) if !is_v && !self.span(s, t) => {
                let (half, cs) = (r.rot_dim / 2, &self.cos_sin[t * r.rot_dim..][..r.rot_dim]);
                let mut o = b.to_vec();
                for d in 0..half {
                    let (cos, sin) = (c.dtype.value(cs[d]), c.dtype.value(cs[half + d]));
                    o[d] = b[d] * cos - b[d + r.pair_off] * sin;
                    o[d + r.pair_off] = b[d + r.pair_off] * cos + b[d] * sin;
                }
                o
            }
            _ => b.to_vec(),
        }
    }

    fn prefix(&self, s: usize) -> usize {
        self.c.seqs[s].0 - self.c.seqs[s].1
    }

    /// Key `t` of sequence `s` was written by this step's KV writer.
    fn written(&self, s: usize, t: usize) -> bool {
        t >= self.prefix(s) && !(self.c.first_new_write_skipped && t == self.prefix(s))
    }

    fn span(&self, s: usize, t: usize) -> bool {
        self.block_table[s * self.max_blocks + t / self.c.block_size] & SPAN_BIT != 0
    }

    fn slot_mapping(&self) -> Vec<u32> {
        (0..self.c.seqs.len())
            .flat_map(|s| {
                (self.prefix(s)..self.c.seqs[s].0).map(move |t| {
                    if self.written(s, t) {
                        self.slot[s][t] as u32
                    } else {
                        WRITE_SKIP
                    }
                })
            })
            .collect()
    }

    /// A pool filled with `fill`, then `tokens(s, t)` of `of` written raw at
    /// their slots — what the KV writer does.
    fn pool(
        &self,
        device: &Device,
        of: &[Vec<f32>],
        fill: u16,
        tokens: impl Fn(usize, usize) -> bool,
    ) -> Pool {
        let c = &self.c;
        let elems = vec![fill; self.n_blocks * c.num_kv_heads * c.block_size * c.head_dim];
        let data = shared(device, &elems);
        let blk_bytes = (c.num_kv_heads * c.block_size * c.head_dim * 2) as u64;
        let table: Vec<u64> = if c.blocks_per_chunk == 0 {
            vec![data.gpuAddress()]
        } else {
            (0..self.n_blocks.div_ceil(c.blocks_per_chunk))
                .map(|ch| data.gpuAddress() + (ch * c.blocks_per_chunk) as u64 * blk_bytes)
                .collect()
        };
        let base = data.contents().as_ptr() as *mut u16;
        for (s, slots) in self.slot.iter().enumerate() {
            for (t, &sl) in slots.iter().enumerate().filter(|&(t, _)| tokens(s, t)) {
                let (pb, tib) = (sl / c.block_size, sl % c.block_size);
                for h in 0..c.num_kv_heads {
                    let dst = ((pb * c.num_kv_heads + h) * c.block_size + tib) * c.head_dim;
                    let src = &of[s][(t * c.num_kv_heads + h) * c.head_dim..][..c.head_dim];
                    for (d, &x) in src.iter().enumerate() {
                        // SAFETY: `dst + d` is inside the pool (`sl` < n_blocks * block_size).
                        unsafe { *base.add(dst + d) = c.dtype.bits(x) };
                    }
                }
            }
        }
        Pool {
            data,
            table: shared(device, &table),
        }
    }

    /// Attention function constants 0..=7, the rope-on-read slots, and extras.
    fn attn_constants(&self, extra: &[ConstantValue]) -> Vec<ConstantValue> {
        let c = &self.c;
        let mut v = vec![
            ConstantValue::uint(0, c.head_dim as u32),
            ConstantValue::uint(1, c.num_q_heads as u32),
            ConstantValue::uint(2, c.num_kv_heads as u32),
            ConstantValue::float(3, c.attn_scale),
            ConstantValue::uint(4, c.block_size as u32),
            ConstantValue::uint(5, self.max_blocks as u32),
            ConstantValue::uint(6, c.blocks_per_chunk as u32),
            ConstantValue::int(7, c.window),
        ];
        v.extend(self.rope_constants());
        v.extend_from_slice(extra);
        v
    }

    fn rope_constants(&self) -> Vec<ConstantValue> {
        self.c.rope.map_or(Vec::new(), |r| {
            vec![
                ConstantValue::uint(8, r.rot_dim as u32),
                ConstantValue::uint(9, r.pair_off as u32),
                ConstantValue::uint(10, 1),
            ]
        })
    }
}

/// f32 softmax attention of every query over `kv(s, t, h, is_v)` — causal, the
/// window, span keys re-roped like `rope_on_read_*` (rounded to the dtype).
fn ideal_attention(f: &Fixture, kv: impl Fn(usize, usize, usize, bool) -> Vec<f32>) -> Vec<f32> {
    let c = &f.c;
    let (hd, nq) = (c.head_dim, c.num_q_heads);
    let group = nq / c.num_kv_heads;
    let n_q = *f.cu_seqlens.last().unwrap() as usize;
    let mut out = vec![0f32; n_q * nq * hd];
    for (s, &(len, new)) in c.seqs.iter().enumerate() {
        for j in 0..new {
            let row = f.cu_seqlens[s] as usize + j;
            let q_pos = len - new + j;
            for h in 0..nq {
                let kvh = h / group;
                let qrow = &f.q[(row * nq + h) * hd..][..hd];
                let mut scores = Vec::new();
                let mut vals = Vec::new();
                for t in 0..=q_pos {
                    if c.window > 0 && (q_pos - t) as i64 >= c.window as i64 {
                        continue;
                    }
                    let mut k = kv(s, t, kvh, false);
                    if let Some(r) = c.rope
                        && f.span(s, t)
                    {
                        let cs = &f.cos_sin[t * r.rot_dim..][..r.rot_dim];
                        let half = r.rot_dim / 2;
                        let src = k.clone();
                        for d in 0..half {
                            let (cos, sin) = (c.dtype.value(cs[d]), c.dtype.value(cs[half + d]));
                            let (x0, x1) = (src[d], src[d + r.pair_off]);
                            k[d] = c.dtype.round(x0 * cos - x1 * sin);
                            k[d + r.pair_off] = c.dtype.round(x1 * cos + x0 * sin);
                        }
                    }
                    scores
                        .push(c.attn_scale * qrow.iter().zip(&k).map(|(a, b)| a * b).sum::<f32>());
                    vals.push(kv(s, t, kvh, true));
                }
                // The sink column: UNSCALED by attn_scale, no V, one per head.
                let sink = f.c.sinks.map(|_| f.sink[h]);
                let mut m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if let Some(sink) = sink {
                    m = m.max(sink);
                }
                let w: Vec<f32> = scores.iter().map(|&x| (x - m).exp()).collect();
                let l: f32 = w.iter().sum::<f32>() + sink.map_or(0.0, |s| (s - m).exp());
                let orow = &mut out[(row * nq + h) * hd..][..hd];
                for (wi, v) in w.iter().zip(&vals) {
                    for d in 0..hd {
                        orow[d] += wi / l * v[d];
                    }
                }
            }
        }
    }
    out
}

struct Outputs {
    got: Vec<f32>,
    /// Host f32 attention over the very codes the GPU wrote.
    ideal: Vec<f32>,
    /// f32 attention over the true (unquantized) K/V — the goal.
    exact: Vec<f32>,
}

/// The TurboQuant attention of case `c`, the host f32 reference over the same
/// packed codes, and exact attention. `restore`: remove each operand's bias
/// before quantizing and restore it after (the fix); `false` codes the biased
/// vectors themselves.
fn run_case(c: &Case, restore: bool) -> Option<Outputs> {
    let Some(di) = detect_device() else {
        eprintln!("skipping {}: no Metal 4 GPU", c.name);
        return None;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    // The quantize of each operand, K then V, baked with its offset: K's bias rotated, V's as-is.
    let compress: [PipelineKey; 2] = [false, true].map(|is_v| {
        let offset = match (restore && c.bias.is_some(), is_v) {
            (false, _) => TqOffset::None,
            (true, false) => TqOffset::RotatedBias,
            (true, true) => TqOffset::Bias,
        };
        let (rot_dim, pair_off) = c.rope.map_or((0, 0), |r| (r.rot_dim, r.pair_off));
        let constants = TqCompressConstants {
            head_dim: HeadDim(c.head_dim as u32),
            bits: TqCodeBits(c.bits),
            num_kv_heads: NumKvHeads(c.num_kv_heads as u32),
            block_size: BlockSize(c.block_size as u32),
            blocks_per_chunk: BlocksPerChunk(c.blocks_per_chunk as u32),
            writeback: TqWriteback::Raw,
            offset,
            rot_dim: RotDim(rot_dim as u32),
            pair_off: RopePairOff(pair_off as u32),
        };
        PipelineKey::new("turboquant", c.dtype.compress(), constants.into())
    });
    cache.register_baked(&baked_kernels(&compress));
    let pso = |lib: &'static str, name: String, consts: Vec<ConstantValue>| {
        let name: &'static str = Box::leak(name.into_boxed_str());
        baked_build(&cache, &PipelineKey::new(lib, name, consts)).expect("pipeline")
    };
    let f = &Fixture::new(c);
    let (hd, nkv, bs) = (c.head_dim, c.num_kv_heads, c.block_size);
    let n_seqs = c.seqs.len();
    let n_q = *f.cu_seqlens.last().unwrap() as usize;
    let decode = c.seqs.iter().all(|&(_, new)| new == 1);
    let quant = PolarQuantizer::new(hd, c.bits, SEED);
    let boundaries: Vec<f32> = quant
        .centroids()
        .windows(2)
        .map(|w| (w[0] + w[1]) / 2.0)
        .collect();
    let pdim = packed_dim(hd, c.bits);
    let n_slots = f.n_blocks * bs;

    let signs = shared(&device, quant.signs());
    let centroids = shared(&device, quant.centroids());
    let bounds = shared(&device, &boundaries);
    let block_table = shared(&device, &f.block_table);
    let seq_used = shared(
        &device,
        &c.seqs.iter().map(|&(l, _)| l as u32).collect::<Vec<_>>(),
    );
    let cu_seqlens = shared(&device, &f.cu_seqlens);
    let slot_mapping = shared(&device, &f.slot_mapping());
    let cos_sin = shared(&device, &f.cos_sin);
    let span_ids = shared(&device, &vec![0u32; n_slots.max(1)]);
    let q = shared(
        &device,
        &f.q.iter().map(|&x| c.dtype.bits(x)).collect::<Vec<_>>(),
    );

    // Packed stores: stale garbage in every slot (block reuse), then the
    // production compress over every key an earlier step quantized.
    let mut rng = Lcg(7);
    let stale_codes: Vec<u32> = (0..n_slots * nkv * pdim)
        .map(|_| rng.0 as u32 ^ rng.next().to_bits())
        .collect();
    let stale_norms: Vec<f32> = (0..n_slots * nkv)
        .map(|_| 1.0 + rng.next().abs() * 4.0)
        .collect();
    let (packed_k, packed_v) = (shared(&device, &stale_codes), shared(&device, &stale_codes));
    let (norms_k, norms_v) = (shared(&device, &stale_norms), shared(&device, &stale_norms));
    let cached = |s: usize, t: usize| !f.written(s, t);
    // Each quantized key's slot — span bit set where its K is stored unrotated,
    // as the worker's slot_mapping carries it — and position.
    let (quant_slots, quant_pos): (Vec<u32>, Vec<u32>) = (0..n_seqs)
        .flat_map(|s| {
            (0..c.seqs[s].0)
                .filter(move |&t| cached(s, t))
                .map(move |t| {
                    let span = if f.span(s, t) { SPAN_BIT } else { 0 };
                    (f.slot[s][t] as u32 | span, t as u32)
                })
        })
        .unzip();
    let quant_slots_buf = shared(&device, &quant_slots);
    let quant_pos = shared(&device, &quant_pos);
    let dt_bits = |x: &[f32]| x.iter().map(|&e| c.dtype.bits(e)).collect::<Vec<_>>();
    let (kb, vb) = (
        shared(&device, &dt_bits(&f.kb)),
        shared(&device, &dt_bits(&f.vb)),
    );
    let sinks = shared(&device, &dt_bits(&f.sink));
    let offset_on = restore && c.bias.is_some();
    let src_k = f.pool(&device, &f.k, 0, cached);
    let src_v = f.pool(&device, &f.v, 0, cached);
    let mut batch = Mtl4DispatchBatch::begin(&device)?;
    for (src, packed, norms, bias, key) in [
        (&src_k, &packed_k, &norms_k, &kb, &compress[0]),
        (&src_v, &packed_v, &norms_v, &vb, &compress[1]),
    ] {
        let compress = baked_build(&cache, key).expect("tq_compress_paged");
        batch.encode(
            &compress,
            &[
                (&src.table, 0),
                (&quant_slots_buf, 1),
                (&signs, 2),
                (&bounds, 3),
                (&centroids, 4),
                (packed, 5),
                (norms, 6),
                (&quant_slots_buf, 16),
                (bias, 18),
                (&cos_sin, 19),
                (&quant_pos, 20),
            ],
            &[],
            &[],
            &[&src.data],
            tg(quant_slots.len(), nkv, 1),
            tg(hd, 1, 1),
        );
    }
    batch.commit(true);

    // The attention's scratch: NaN except the keys this step's writer wrote.
    let nan = c.dtype.bits(f32::NAN);
    let written = |s: usize, t: usize| f.written(s, t);
    let scratch_k = f.pool(&device, &f.k, nan, written);
    let scratch_v = f.pool(&device, &f.v, nan, written);
    let out = shared(&device, &vec![0u16; n_q * c.num_q_heads * hd]);
    let mut batch = Mtl4DispatchBatch::begin(&device)?;
    let resident = [&scratch_k.data, &scratch_v.data];
    let bits = ConstantValue::uint(13, c.bits);
    let (k_bias, v_bias) = (ConstantValue::uint(14, 1), ConstantValue::uint(15, 1));
    // Kept alive past the batch that binds them.
    let partials = shared(
        &device,
        &vec![f32::NAN; n_seqs * c.num_q_heads * c.gqa.unwrap_or(1) as usize * (hd + 2)],
    );
    if let (true, Some(splits)) = (decode, c.gqa) {
        let mut consts = f.attn_constants(&[bits, ConstantValue::uint(18, splits)]);
        if c.rope.is_some_and(|r| r.coresident) {
            consts.push(ConstantValue::uint(12, 1));
        }
        let binds = [
            (&q, 1),
            (&seq_used, 2),
            (&block_table, 3),
            (&scratch_k.table, 4),
            (&scratch_v.table, 5),
            (&cos_sin, 6),
            (&packed_k, 7),
            (&packed_v, 8),
            (&norms_k, 9),
            (&norms_v, 10),
            (&signs, 11),
            (&centroids, 12),
            (&slot_mapping, 13),
            (&partials, 16),
        ];
        let attention = pso(
            "attention",
            format!("attention_decode_gqa_tq_{}_specialized", c.dtype.tag()),
            consts.clone(),
        );
        let grid = tg(n_seqs, c.num_kv_heads, splits as usize);
        batch.encode(&attention, &binds, &[], &[], &resident, grid, tg(hd, 1, 1));
        batch.barrier();
        let combine = pso(
            "attention",
            format!(
                "attention_via_cache_v2_combine_{}_specialized",
                c.dtype.tag()
            ),
            consts,
        );
        let binds = [(&out, 0), (&signs, 11), (&partials, 16)];
        let grid = tg(n_seqs, c.num_q_heads, 1);
        batch.encode(&combine, &binds, &[], &[], &[], grid, tg(128, 1, 1));
    } else if decode {
        let heads = c.decode_heads;
        let mut consts = f.attn_constants(&[bits, ConstantValue::uint(16, heads)]);
        if c.rope.is_some_and(|r| r.coresident) {
            consts.push(ConstantValue::uint(12, 1));
        }
        if offset_on {
            consts.extend([k_bias, v_bias]);
        }
        if c.sinks.is_some() {
            consts.push(ConstantValue::uint(21, 1));
        }
        let mut binds = vec![
            (&out, 0),
            (&q, 1),
            (&seq_used, 2),
            (&block_table, 3),
            (&scratch_k.table, 4),
            (&scratch_v.table, 5),
            (&cos_sin, 6),
            (&packed_k, 7),
            (&packed_v, 8),
            (&norms_k, 9),
            (&norms_v, 10),
            (&signs, 11),
            (&centroids, 12),
            (&slot_mapping, 13),
        ];
        if offset_on {
            binds.extend([(&kb, 14), (&vb, 15)]);
        }
        if c.sinks.is_some() {
            binds.push((&sinks, 16));
        }
        let attention = pso(
            "attention",
            format!("attention_via_cache_v2_{}_specialized", c.dtype.tag()),
            consts,
        );
        batch.encode(
            &attention,
            &binds,
            &[],
            &[],
            &resident,
            tg(n_seqs, c.num_q_heads / heads as usize, 1),
            tg(1024, 1, 1),
        );
    } else {
        let stage_consts = |rope: bool| {
            let mut v = vec![
                ConstantValue::uint(0, hd as u32),
                ConstantValue::uint(2, nkv as u32),
                ConstantValue::uint(4, bs as u32),
                ConstantValue::uint(5, f.max_blocks as u32),
                ConstantValue::uint(6, c.blocks_per_chunk as u32),
            ];
            if rope {
                v.extend(f.rope_constants());
            }
            v.push(bits);
            if offset_on {
                v.push(if rope { k_bias } else { v_bias });
            }
            v
        };
        let stage_name = format!("tq_stage_rotated_{}", c.dtype.tag());
        // New rows, then cached rows, as the tape stages them.
        for pass in [TqStagePass::New, TqStagePass::Cached] {
            for (scratch, packed, norms, bias, is_k) in [
                (&scratch_k, &packed_k, &norms_k, &kb, true),
                (&scratch_v, &packed_v, &norms_v, &vb, false),
            ] {
                let mut consts = stage_consts(is_k);
                consts.push(ConstantValue::uint(17, pass as u32));
                let stage = pso("attention", stage_name.clone(), consts);
                batch.encode(
                    &stage,
                    &[
                        (&scratch.table, 0),
                        (&block_table, 1),
                        (&seq_used, 2),
                        (&cu_seqlens, 3),
                        (&slot_mapping, 4),
                        (packed, 5),
                        (norms, 6),
                        (&signs, 7),
                        (&centroids, 8),
                        (&cos_sin, 9),
                        (bias, 10),
                    ],
                    &[],
                    &[],
                    &resident,
                    tg(f.max_blocks, nkv, n_seqs),
                    tg(32, 1, 1),
                );
            }
            batch.barrier();
        }
        let rotate_consts = vec![
            ConstantValue::uint(0, hd as u32),
            ConstantValue::uint(1, c.num_q_heads as u32),
        ];
        let rotate = |dir: &str| {
            pso(
                "attention",
                format!("tq_{dir}_rows_{}", c.dtype.tag()),
                rotate_consts.clone(),
            )
        };
        batch.encode(
            &rotate("rotate"),
            &[(&q, 0), (&signs, 1)],
            &[],
            &[],
            &[],
            tg(n_q, c.num_q_heads, 1),
            tg(32, 1, 1),
        );
        batch.barrier();
        let mut prefill_consts = f.attn_constants(&[]);
        let mut prefill_binds = vec![
            (&out, 0),
            (&q, 1),
            (&cu_seqlens, 2),
            (&seq_used, 3),
            (&block_table, 4),
            (&scratch_k.table, 5),
            (&scratch_v.table, 6),
            (&cos_sin, 7),
            (&span_ids, 8),
        ];
        if c.sinks.is_some() {
            prefill_consts.push(ConstantValue::uint(21, 1));
            prefill_binds.push((&sinks, 9));
        }
        let attention = pso(
            "attention",
            format!(
                "attention_prefill_sdpa_v2_paged_{}_specialized",
                c.dtype.tag()
            ),
            prefill_consts,
        );
        batch.encode(
            &attention,
            &prefill_binds,
            &[],
            &[],
            &resident,
            tg(c.num_q_heads, n_q, 1),
            tg(1024, 1, 1),
        );
        batch.barrier();
        batch.encode(
            &rotate("unrotate"),
            &[(&out, 0), (&signs, 1)],
            &[],
            &[],
            &[],
            tg(n_q, c.num_q_heads, 1),
            tg(32, 1, 1),
        );
    }
    batch.commit(true);
    let got: Vec<f32> = read::<u16>(&out, n_q * c.num_q_heads * hd)
        .into_iter()
        .map(|b| c.dtype.value(b))
        .collect();

    // Host f32 decode of the very codes the GPU wrote.
    let codes_k: Vec<u32> = read(&packed_k, n_slots * nkv * pdim);
    let codes_v: Vec<u32> = read(&packed_v, n_slots * nkv * pdim);
    let nk: Vec<f32> = read(&norms_k, n_slots * nkv);
    let nv: Vec<f32> = read(&norms_v, n_slots * nkv);
    let decode_row = |codes: &[u32], norms: &[f32], slot: usize, h: usize| -> Vec<f32> {
        let row = slot * nkv + h;
        quant.dequantize(
            &unpack_indices(&codes[row * pdim..][..pdim], c.bits, hd),
            norms[row],
        )
    };
    let plain = |s: usize, t: usize, h: usize, is_v: bool| {
        let of = if is_v { &f.v[s] } else { &f.k[s] };
        of[(t * nkv + h) * hd..][..hd].to_vec()
    };
    let ideal = ideal_attention(f, |s, t, h, is_v| {
        if f.written(s, t) {
            return plain(s, t, h, is_v);
        }
        let mut x = if is_v {
            decode_row(&codes_v, &nv, f.slot[s][t], h)
        } else {
            decode_row(&codes_k, &nk, f.slot[s][t], h)
        };
        if offset_on {
            for (e, o) in x.iter_mut().zip(f.offset(s, t, h, is_v)) {
                *e += o;
            }
        }
        // Span keys are decoded, rounded to the dtype, then re-roped.
        if !is_v && f.span(s, t) {
            x.iter().map(|&e| c.dtype.round(e)).collect()
        } else {
            x
        }
    });
    Some(Outputs {
        got,
        ideal,
        exact: ideal_attention(f, plain),
    })
}

/// Every query-head count a decode threadgroup can serve for `c` (one run for
/// a prefill case).
fn head_counts(c: &Case) -> Vec<Case> {
    if c.gqa.is_some() || !c.seqs.iter().all(|&(_, new)| new == 1) {
        return vec![c.clone()];
    }
    TqDecodeHeads::candidates(
        HeadDim(c.head_dim as u32),
        NumQHeads(c.num_q_heads as u32),
        NumKvHeads(c.num_kv_heads as u32),
    )
    .map(|h| Case {
        decode_heads: h.get(),
        ..c.clone()
    })
    .collect()
}

fn check(c: Case) {
    for c in head_counts(&c) {
        check_one(c);
    }
}

fn check_one(c: Case) {
    let Some(Outputs { got, ideal, .. }) = run_case(&c, true) else {
        return;
    };
    assert!(
        got.iter().all(|x| x.is_finite()),
        "{}: output is not finite — it read an unstaged scratch key",
        c.name
    );
    let peak = ideal.iter().fold(0f32, |m, x| m.max(x.abs()));
    let err = got
        .iter()
        .zip(&ideal)
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max);
    // Decode rounds once (the output); prefill also stores q̂, K̂/V̂ and the
    // attention's output in the dtype before the final rotation.
    let rounding = if c.seqs.iter().all(|&(_, new)| new == 1) {
        1.0
    } else {
        3.0
    };
    let tol = peak * c.dtype.ulp() * rounding;
    eprintln!(
        "{} ({} heads/threadgroup): peak {peak:.3}  max err {err:.2e}  (tol {tol:.2e})",
        c.name, c.decode_heads
    );
    assert!(err <= tol, "{}: max error {err} > {tol}", c.name);
}

/// Llama-3.2-3B: the production geometry (bf16, 3-bit, GQA 3, NeoX
/// rope-on-read with the co-resident decode lane layout).
fn llama_3b(name: &'static str) -> Case {
    Case {
        name,
        dtype: Dtype::Bf16,
        head_dim: 128,
        num_q_heads: 24,
        num_kv_heads: 8,
        bits: 3,
        block_size: 16,
        blocks_per_chunk: 0,
        attn_scale: 1.0 / (128f32).sqrt(),
        window: 0,
        rope: Some(Rope {
            rot_dim: 128,
            pair_off: 64,
            coresident: true,
        }),
        seqs: vec![(1000, 1)],
        span_blocks: vec![],
        first_new_write_skipped: false,
        bias: None,
        decode_heads: 1,
        gqa: None,
        sinks: None,
    }
}

// ── decode ─────────────────────────────────────────────────────────────

#[test]
fn decode_llama_3b_bf16() {
    check(llama_3b("decode llama-3b bf16"));
}

#[test]
fn decode_llama_3b_f16() {
    check(Case {
        dtype: Dtype::F16,
        ..llama_3b("decode llama-3b f16")
    });
}

/// Span blocks (stored unrotated, re-roped on read) — including the block
/// holding the step's own key.
#[test]
fn decode_span_blocks() {
    check(Case {
        span_blocks: vec![0, 3, 4, 62],
        ..llama_3b("decode spans")
    });
}

/// The step's key sits in a reused span block: nothing was written to the
/// scratch, so it too must come from the packed store.
#[test]
fn decode_write_skipped_key() {
    check(Case {
        span_blocks: vec![62],
        first_new_write_skipped: true,
        ..llama_3b("decode write-skipped key")
    });
}

/// A decode step of eight sequences — the batched decode that runs in the
/// multi-token buckets — one of them only its own key.
#[test]
fn decode_eight_sequences() {
    check(Case {
        seqs: vec![
            (900, 1),
            (1, 1),
            (257, 1),
            (64, 1),
            (1000, 1),
            (33, 1),
            (512, 1),
            (2, 1),
        ],
        ..llama_3b("decode eight seqs")
    });
}

/// Llama-3.2-1B geometry on the contiguous lane layout, chunked addressing
/// across two chunks.
#[test]
fn decode_head_dim_64_chunked() {
    check(Case {
        head_dim: 64,
        num_q_heads: 32,
        num_kv_heads: 8,
        blocks_per_chunk: 128,
        attn_scale: 0.125,
        rope: None,
        seqs: vec![(2100, 1)],
        ..llama_3b("decode hd64 chunked")
    });
}

/// 4-bit, head_dim 256, one KV head, sliding window.
#[test]
fn decode_head_dim_256_sliding_window() {
    check(Case {
        head_dim: 256,
        num_q_heads: 8,
        num_kv_heads: 1,
        bits: 4,
        attn_scale: 1.0 / 16.0,
        window: 100,
        rope: None,
        seqs: vec![(700, 1)],
        ..llama_3b("decode hd256 window")
    });
}

/// Gemma-4 global layers: head_dim 512, 4-bit, 64-token pages, proportional
/// rope (pair offset head_dim/2).
fn gemma4_global(name: &'static str) -> Case {
    Case {
        head_dim: 512,
        num_q_heads: 16,
        num_kv_heads: 2,
        bits: 4,
        block_size: 64,
        attn_scale: 1.0 / 64.0,
        rope: Some(Rope {
            rot_dim: 128,
            pair_off: 256,
            coresident: true,
        }),
        seqs: vec![(900, 1)],
        span_blocks: vec![1, 14],
        ..llama_3b(name)
    }
}

#[test]
fn decode_gemma4_global_head_dim_512() {
    check(gemma4_global("decode gemma4 global"));
}

/// Gemma 4's global layers, each KV head's 8 query heads together (`attention_decode_gqa_tq`),
/// whole and over 4 and 16 threadgroups: span blocks re-roped, the step's own key (in a span
/// block) from the cache.
#[test]
fn decode_gemma4_global_gqa() {
    for splits in [1, 4, 16] {
        check(Case {
            gqa: Some(splits),
            ..gemma4_global("decode gemma4 global gqa")
        });
    }
}

/// The step's own key in a reused span block: packed, decoded and re-roped like the others.
#[test]
fn decode_gemma4_global_gqa_write_skipped_key() {
    check(Case {
        gqa: Some(4),
        span_blocks: vec![14],
        first_new_write_skipped: true,
        ..gemma4_global("decode gemma4 global gqa write-skipped key")
    });
}

/// Eight sequences, the short ones a partial key block or only their own key.
#[test]
fn decode_gemma4_global_gqa_eight_sequences() {
    check(Case {
        gqa: Some(4),
        span_blocks: vec![0, 3],
        seqs: vec![
            (900, 1),
            (1, 1),
            (257, 1),
            (64, 1),
            (17, 1),
            (8, 1),
            (9, 1),
            (2, 1),
        ],
        ..gemma4_global("decode gemma4 global gqa eight seqs")
    });
}

/// gpt-oss-20b: the first attention-sinks arch on a coded cache — head dim
/// 64 (the smallest the codec takes), GQA 8, 4-bit. Its K/V projections are
/// biased and stay biased past the gemm (the affine gemm dequantizes the
/// weight; the linear bias rides in as its own operand), so a coded gpt-oss
/// cache also carries the KB/VB bias-restore path (consts 14/15) — see
/// `gpt_oss_biased_rope` for the full composition. `check` runs every head
/// count a decode threadgroup can serve for the geometry.
fn gpt_oss(name: &'static str) -> Case {
    Case {
        head_dim: 64,
        num_q_heads: 64,
        num_kv_heads: 8,
        bits: 4,
        attn_scale: 0.125,
        rope: None,
        seqs: vec![(1000, 1)],
        sinks: Some(2.0),
        ..llama_3b(name)
    }
}

#[test]
fn decode_gpt_oss_sinks() {
    check(gpt_oss("decode gpt-oss sinks"));
}

/// The arch's even layers: the 128-token sliding window over a 1000-key
/// context, sinks on.
#[test]
fn decode_gpt_oss_sinks_sliding_window() {
    check(Case {
        window: 128,
        ..gpt_oss("decode gpt-oss sinks window")
    });
}

/// gpt-oss's every layer: biased K/V projections restored on read, NeoX
/// rope-on-read at rot 64 (YaRN lowers to the same table shape), and the
/// sink column — the full composition a coded gpt-oss cache carries.
fn gpt_oss_biased_rope(name: &'static str) -> Case {
    Case {
        bias: Some(100.0),
        rope: Some(Rope {
            rot_dim: 64,
            pair_off: 32,
            coresident: true,
        }),
        ..gpt_oss(name)
    }
}

#[test]
fn decode_gpt_oss_sinks_biased_rope() {
    check(gpt_oss_biased_rope("decode gpt-oss sinks biased rope"));
}

/// The arch's even layers: the same composition in the 128-token window.
#[test]
fn decode_gpt_oss_sinks_biased_rope_window() {
    check(Case {
        window: 128,
        ..gpt_oss_biased_rope("decode gpt-oss sinks biased rope window")
    });
}

/// Head dims 256 and 128: two key groups a threadgroup, and one. Qwen3.6's full attention is the
/// first: 64 rotary dims of 256, pairs 32 apart.
#[test]
fn decode_gqa_narrower_heads() {
    for (head_dim, rot_dim, pair_off) in [(256, 64, 32), (256, 64, 128), (128, 128, 64)] {
        check(Case {
            gqa: Some(4),
            head_dim,
            attn_scale: 1.0 / (head_dim as f32).sqrt(),
            block_size: 16,
            rope: Some(Rope {
                rot_dim,
                pair_off,
                coresident: true,
            }),
            span_blocks: vec![2, 56],
            ..gemma4_global("decode gqa narrower heads")
        });
    }
}

/// GPU time of Gemma 4's global decode attention: the per-head-group kernel at each head count,
/// and the per-KV-head one over each split count with its combine, at 1k and 16k keys.
/// `--ignored --nocapture`.
#[test]
#[ignore = "timing"]
fn gemma4_global_decode_timing() {
    let Some(di) = detect_device() else {
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    let pso = |name: String, consts: Vec<ConstantValue>| {
        let name: &'static str = Box::leak(name.into_boxed_str());
        baked_build(&cache, &PipelineKey::new("attention", name, consts)).expect("pipeline")
    };
    let (hd, nq, nkv, bs, bits) = (512usize, 16usize, 2usize, 16usize, 4u32);
    let quant = PolarQuantizer::new(hd, bits, SEED);
    let (signs, centroids) = (
        shared(&device, quant.signs()),
        shared(&device, quant.centroids()),
    );
    for kv_len in [1024usize, 16384] {
        let blocks = kv_len.div_ceil(bs);
        let n_rows = blocks * bs * nkv;
        let pdim = packed_dim(hd, bits);
        let mut rng = Lcg(3);
        let codes: Vec<u32> = (0..n_rows * pdim).map(|_| rng.next().to_bits()).collect();
        let norms: Vec<f32> = (0..n_rows).map(|_| 1.0 + rng.next().abs()).collect();
        let q: Vec<u16> = (0..nq * hd)
            .map(|_| Dtype::Bf16.bits(rng.gauss()))
            .collect();
        let (packed_k, packed_v) = (shared(&device, &codes), shared(&device, &codes));
        let (norms_k, norms_v) = (shared(&device, &norms), shared(&device, &norms));
        let q = shared(&device, &q);
        let seq_used = shared(&device, &[kv_len as u32]);
        let block_table = shared(&device, &(0..blocks as u32).collect::<Vec<_>>());
        let slot_mapping = shared(&device, &[WRITE_SKIP]);
        let table = shared(&device, &[0u64]);
        let out = shared(&device, &vec![0u16; nq * hd]);
        let partials = shared(&device, &vec![0f32; nq * 32 * (hd + 2)]);
        let base = vec![
            ConstantValue::uint(0, hd as u32),
            ConstantValue::uint(1, nq as u32),
            ConstantValue::uint(2, nkv as u32),
            ConstantValue::float(3, 1.0 / (hd as f32).sqrt()),
            ConstantValue::uint(4, bs as u32),
            ConstantValue::uint(5, blocks as u32),
            ConstantValue::uint(6, 0),
            ConstantValue::int(7, 0),
            ConstantValue::uint(13, bits),
        ];
        let binds = [
            (&out, 0),
            (&q, 1),
            (&seq_used, 2),
            (&block_table, 3),
            (&table, 4),
            (&table, 5),
            (&table, 6),
            (&packed_k, 7),
            (&packed_v, 8),
            (&norms_k, 9),
            (&norms_v, 10),
            (&signs, 11),
            (&centroids, 12),
            (&slot_mapping, 13),
            (&partials, 16),
        ];
        const REPS: usize = 50;
        let time = |steps: &[(&Pipeline, MTLSize, MTLSize)]| {
            let mut best = f64::MAX;
            for _ in 0..3 {
                let mut batch = Mtl4DispatchBatch::begin(&device).expect("MTL4 queue");
                for _ in 0..REPS {
                    for (pso, grid, threads) in steps {
                        batch.encode(pso, &binds, &[], &[], &[], *grid, *threads);
                        batch.barrier();
                    }
                }
                let t = std::time::Instant::now();
                batch.commit(true);
                best = best.min(t.elapsed().as_secs_f64() / REPS as f64);
            }
            best * 1e6
        };
        for heads in [1u32, 2] {
            let mut consts = base.clone();
            consts.push(ConstantValue::uint(16, heads));
            let p = pso("attention_via_cache_v2_bf16_specialized".into(), consts);
            let grid = tg(1, nq / heads as usize, 1);
            eprintln!(
                "{kv_len:>6} keys  per-head-group kernel, {heads} heads/threadgroup: {:8.1} us",
                time(&[(&p, grid, tg(1024, 1, 1))])
            );
        }
        for splits in [4u32, 8, 16, 32] {
            let mut consts = base.clone();
            consts.push(ConstantValue::uint(18, splits));
            let gqa = pso(
                "attention_decode_gqa_tq_bf16_specialized".into(),
                consts.clone(),
            );
            let combine = pso(
                "attention_via_cache_v2_combine_bf16_specialized".into(),
                consts,
            );
            let attention = (&*gqa, tg(1, nkv, splits as usize), tg(hd, 1, 1));
            eprintln!(
                "{kv_len:>6} keys  per-KV-head kernel, {splits:>2} splits: {:8.1} us, + combine {:8.1} us",
                time(&[attention]),
                time(&[attention, (&combine, tg(1, nq, 1), tg(128, 1, 1))])
            );
        }
    }
}

// ── prefill ────────────────────────────────────────────────────────────

/// A chunk continuing a cached prefix, and a fresh prompt, in one step.
#[test]
fn prefill_llama_3b_bf16() {
    check(Case {
        seqs: vec![(700, 45), (60, 60)],
        ..llama_3b("prefill llama-3b bf16")
    });
}

#[test]
fn prefill_llama_3b_f16() {
    check(Case {
        dtype: Dtype::F16,
        seqs: vec![(700, 45), (60, 60)],
        ..llama_3b("prefill llama-3b f16")
    });
}

/// Span blocks in the cached prefix and among the new keys, and a new key
/// in a reused (write-skipped) block.
#[test]
fn prefill_span_blocks() {
    check(Case {
        span_blocks: vec![0, 5, 40, 43],
        first_new_write_skipped: true,
        seqs: vec![(700, 45)],
        ..llama_3b("prefill spans")
    });
}

/// head_dim 64, chunked addressing across two chunks, mixed with a decoding
/// sequence (a mixed step).
#[test]
fn prefill_head_dim_64_chunked_mixed() {
    check(Case {
        head_dim: 64,
        num_q_heads: 32,
        num_kv_heads: 8,
        blocks_per_chunk: 128,
        attn_scale: 0.125,
        rope: None,
        seqs: vec![(2100, 30), (300, 1)],
        ..llama_3b("prefill hd64 chunked mixed")
    });
}

#[test]
fn prefill_head_dim_256_sliding_window() {
    check(Case {
        head_dim: 256,
        num_q_heads: 8,
        num_kv_heads: 1,
        bits: 4,
        attn_scale: 1.0 / 16.0,
        window: 100,
        rope: None,
        seqs: vec![(700, 40)],
        ..llama_3b("prefill hd256 window")
    });
}

/// gpt-oss prefill under the codec: the staged rotated-domain image, the
/// query rotated in and the output back, sinks on — full attention and the
/// arch's 128-token window.
#[test]
fn prefill_gpt_oss_sinks() {
    check(Case {
        seqs: vec![(500, 200)],
        ..gpt_oss("prefill gpt-oss sinks")
    });
    check(Case {
        seqs: vec![(500, 200)],
        window: 128,
        ..gpt_oss("prefill gpt-oss sinks window")
    });
}

#[test]
fn prefill_gemma4_global_head_dim_512() {
    check(Case {
        seqs: vec![(900, 20)],
        ..gemma4_global("prefill gemma4 global")
    });
}

// ── Qwen2: K/V projection biases ────────────────────────────────────────

/// Qwen2.5-7B: bf16, 4-bit, GQA 7 — and Qwen2's K/V projection biases, 100x the
/// signal on a few channels (its layer 0 measures 54x on the whole vector).
fn qwen2_7b(name: &'static str) -> Case {
    Case {
        num_q_heads: 28,
        num_kv_heads: 4,
        bits: 4,
        bias: Some(100.0),
        ..llama_3b(name)
    }
}

/// The goal, not just parity: attention over the codes must track attention
/// over the true K/V — and coding the biased vectors must not, or the case does
/// not exercise the defect. Error is measured on the output's input-dependent
/// part (the V bias is exact either way, so it would only inflate the scale).
fn check_fidelity(c: Case, max_rel_err: f32, min_gain: f32) {
    for c in head_counts(&c) {
        check_fidelity_one(c, max_rel_err, min_gain);
    }
}

fn check_fidelity_one(c: Case, max_rel_err: f32, min_gain: f32) {
    let Some(fixed) = run_case(&c, true) else {
        return;
    };
    let Some(defect) = run_case(&c, false) else {
        return;
    };
    let vb = Fixture::new(&c).vb;
    let group = c.num_q_heads / c.num_kv_heads;
    let rel = |o: &Outputs| {
        let (mut err, mut signal) = (0f64, 0f64);
        for (i, (&got, &want)) in o.got.iter().zip(&o.exact).enumerate() {
            let kv_head = (i / c.head_dim) % c.num_q_heads / group;
            let bias = vb[kv_head * c.head_dim + i % c.head_dim];
            err += ((got - want) as f64).powi(2);
            signal += ((want - bias) as f64).powi(2);
        }
        (err / signal.max(1e-30)).sqrt() as f32
    };
    let (fixed, defect) = (rel(&fixed), rel(&defect));
    eprintln!(
        "{}: output error vs exact — bias restored {fixed:.4}, coded {defect:.4}",
        c.name
    );
    assert!(
        defect >= fixed * min_gain,
        "{}: coding the biased vectors ({defect}) is not {min_gain}x worse than removing the \
         bias ({fixed}) — the case does not exercise the defect",
        c.name
    );
    assert!(
        fixed <= max_rel_err,
        "{}: attention over the bias-restored codes misses the exact output by {fixed} > \
         {max_rel_err}",
        c.name
    );
}

#[test]
fn decode_qwen2_7b_biased_kv() {
    check(qwen2_7b("decode qwen2-7b biased"));
}

/// Span blocks hold K unrotated, so their bias is restored unrotated — and the
/// step's own key in a reused span block comes from the packed store.
#[test]
fn decode_qwen2_7b_biased_kv_span_blocks() {
    check(Case {
        span_blocks: vec![0, 3, 4, 62],
        first_new_write_skipped: true,
        ..qwen2_7b("decode qwen2-7b biased spans")
    });
}

#[test]
fn prefill_qwen2_7b_biased_kv() {
    check(Case {
        seqs: vec![(700, 45), (60, 60)],
        ..qwen2_7b("prefill qwen2-7b biased")
    });
}

#[test]
fn prefill_qwen2_7b_biased_kv_span_blocks() {
    check(Case {
        span_blocks: vec![0, 5, 40, 43],
        first_new_write_skipped: true,
        seqs: vec![(700, 45)],
        ..qwen2_7b("prefill qwen2-7b biased spans")
    });
}

/// Measured: 0.140 relative error with the bias restored, 2.69 coding the
/// biased vectors (19x) — the garbage Qwen2 decoded under TurboQuant.
#[test]
fn decode_qwen2_7b_biased_kv_tracks_exact_attention() {
    check_fidelity(qwen2_7b("decode qwen2-7b biased fidelity"), 0.2, 10.0);
}

/// Measured: 0.139 with the bias restored in the staged rotated image, 1.97
/// coding the biased vectors (14x).
#[test]
fn prefill_qwen2_7b_biased_kv_tracks_exact_attention() {
    check_fidelity(
        Case {
            seqs: vec![(700, 45)],
            ..qwen2_7b("prefill qwen2-7b biased fidelity")
        },
        0.2,
        10.0,
    );
}

// ── fp16 KV: the attentions of a step with several sequences ───────────

/// A paged prefill attention that reads each sequence's K/V through its own
/// block-table row: what a step with several sequences runs.
#[derive(Clone, Copy)]
enum PerRow {
    /// sdpa-paged, re-roping span blocks as it reads.
    Sdpa,
    /// gqa_shared without the rope-once scratch, re-roping span blocks.
    GqaShared,
    /// The simdgroup steel kernel, reading the cache's roped K as is.
    Steel,
    /// The NAX (M5+) steel kernel, likewise.
    Nax,
}

/// `kernel` over every sequence of `c` in one step, from an fp16 cache holding
/// every key (NaN elsewhere), against f32 attention of each sequence alone.
/// The sequences' keys differ, so a kernel serving sequence 0's keys to all
/// would fail every other sequence's rows — shown first on the host.
fn check_per_row(c: Case, kernel: PerRow) {
    let Some(di) = detect_device() else {
        eprintln!("skipping {}: no Metal 4 GPU", c.name);
        return;
    };
    if matches!(kernel, PerRow::Nax) && !is_nax_capable(di.profile.generation) {
        eprintln!("skipping {}: no NAX", c.name);
        return;
    }
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    let f = &Fixture::new(&c);
    let (hd, nq, nkv) = (c.head_dim, c.num_q_heads, c.num_kv_heads);
    let n_q = *f.cu_seqlens.last().unwrap() as usize;
    let q_rows =
        |s: usize| f.cu_seqlens[s] as usize * nq * hd..f.cu_seqlens[s + 1] as usize * nq * hd;
    let max_err = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };
    let plain = |s: usize, t: usize, h: usize, is_v: bool| {
        let of = if is_v { &f.v[s] } else { &f.k[s] };
        of[(t * nkv + h) * hd..][..hd].to_vec()
    };
    let ideal = ideal_attention(f, plain);
    let peak = ideal.iter().fold(0f32, |m, x| m.max(x.abs()));
    // Rounded q, K and V, and the output.
    let tol = peak * c.dtype.ulp() * 3.0;
    // Sequence 0 is the longest, so its keys cover every other query position.
    let row_zero = ideal_attention(f, |_, t, h, is_v| plain(0, t, h, is_v));
    for s in 1..c.seqs.len() {
        let rows = q_rows(s);
        let err = max_err(&row_zero[rows.clone()], &ideal[rows]);
        eprintln!(
            "{} sequence {s} read as sequence 0: max err {err:.2e}",
            c.name
        );
        assert!(err > tol, "{}: sequence {s} reads like sequence 0", c.name);
    }

    let tag = c.dtype.tag();
    let q_blocks = |bq: usize| tg(n_q.div_ceil(bq), nq, c.seqs.len());
    let steel_debug = ConstantValue::uint(99, 0);
    let (library, function, extra, grid, threads) = match kernel {
        PerRow::Sdpa => (
            "attention",
            format!("attention_prefill_sdpa_v2_paged_{tag}_specialized"),
            None,
            tg(nq, n_q, 1),
            tg(1024, 1, 1),
        ),
        PerRow::GqaShared => (
            "attention",
            format!("attention_prefill_sdpa_gqa_shared_{tag}_specialized"),
            None,
            tg(nkv, n_q, 1),
            tg(32 * (nq / nkv), 1, 1),
        ),
        PerRow::Steel => (
            "attention_steel_paged",
            steel_paged_symbol(tag, hd as u32)
                .expect("steel instance")
                .to_owned(),
            Some(steel_debug),
            q_blocks(32),
            tg(128, 1, 1),
        ),
        PerRow::Nax => {
            let nax = nax_paged_kernel(tag, hd as u32, c.block_size as u32).expect("NAX instance");
            (
                "attention_steel_nax_paged",
                nax.symbol.to_owned(),
                Some(steel_debug),
                q_blocks(nax.bq as usize),
                tg(nax.threads as usize, 1, 1),
            )
        }
    };
    let function: &'static str = Box::leak(function.into_boxed_str());
    let pso = baked_build(
        &cache,
        &PipelineKey::new(library, function, f.attn_constants(extra.as_slice())),
    )
    .expect("pipeline");

    let nan = c.dtype.bits(f32::NAN);
    let k = f.pool(&device, &f.k, nan, |_, _| true);
    let v = f.pool(&device, &f.v, nan, |_, _| true);
    let seq_used: Vec<u32> = c.seqs.iter().map(|&(l, _)| l as u32).collect();
    let q: Vec<u16> = f.q.iter().map(|&x| c.dtype.bits(x)).collect();
    let (seq_used, q) = (shared(&device, &seq_used), shared(&device, &q));
    let cu_seqlens = shared(&device, &f.cu_seqlens);
    let block_table = shared(&device, &f.block_table);
    let cos_sin = shared(&device, &f.cos_sin);
    let span_ids = shared(&device, &vec![0u32; f.n_blocks * c.block_size]);
    let out = shared(&device, &vec![0u16; n_q * nq * hd]);
    let mut batch = Mtl4DispatchBatch::begin(&device).expect("MTL4 queue");
    batch.encode(
        &pso,
        &[
            (&out, 0),
            (&q, 1),
            (&cu_seqlens, 2),
            (&seq_used, 3),
            (&block_table, 4),
            (&k.table, 5),
            (&v.table, 6),
            (&cos_sin, 7),
            (&span_ids, 8),
        ],
        &[],
        &[],
        &[&k.data, &v.data],
        grid,
        threads,
    );
    batch.commit(true);
    let got: Vec<f32> = read::<u16>(&out, n_q * nq * hd)
        .into_iter()
        .map(|b| c.dtype.value(b))
        .collect();
    for s in 0..c.seqs.len() {
        let rows = q_rows(s);
        let err = max_err(&got[rows.clone()], &ideal[rows]);
        eprintln!("{} sequence {s}: max err {err:.2e} (tol {tol:.2e})", c.name);
        assert!(
            err <= tol,
            "{}: sequence {s} max error {err} > {tol}",
            c.name
        );
    }
}

/// Llama-3.2-3B, production chunking: a continuing chunk, a decoding
/// sequence, and a fresh prompt in one step, the longest first.
fn per_row_llama(name: &'static str) -> Case {
    Case {
        blocks_per_chunk: 128,
        seqs: vec![(700, 45), (300, 1), (60, 60)],
        ..llama_3b(name)
    }
}

#[test]
fn per_row_sdpa_paged_rerope_span_blocks() {
    check_per_row(
        Case {
            span_blocks: vec![0, 5, 40],
            ..per_row_llama("per-row sdpa-paged")
        },
        PerRow::Sdpa,
    );
}

#[test]
fn per_row_gqa_shared_rerope_span_blocks() {
    check_per_row(
        Case {
            blocks_per_chunk: 128,
            seqs: vec![(900, 20), (300, 1), (130, 70)],
            ..gemma4_global("per-row gqa_shared gemma4 global")
        },
        PerRow::GqaShared,
    );
}

#[test]
fn per_row_steel_reads_roped_keys() {
    check_per_row(
        Case {
            rope: None,
            ..per_row_llama("per-row steel")
        },
        PerRow::Steel,
    );
}

#[test]
fn per_row_nax_reads_roped_keys() {
    check_per_row(
        Case {
            rope: None,
            ..per_row_llama("per-row nax")
        },
        PerRow::Nax,
    );
}

// ── verify rows ────────────────────────────────────────────────────────

/// A few-row step's queries — a speculative verify step's last token and drafts — through the
/// decode kernel, one query row per token (`ATTN_ROW_QUERIES`, `cu_seqlens_q` at buffer 23): each
/// row's output is, bit for bit, the one decoding it alone gives, a step of one row per sequence
/// with that row's keys (its sequence's up to its own position) and slot. On TurboQuant every key
/// but a row's own comes from the packed store, so the store holds every key here — the step's
/// earlier rows' too, as the step's quantize leaves it; on an fp16 cache every key is in the cache.
#[test]
fn verify_rows_attend_as_each_would_decoding() {
    let Some(di) = detect_device() else {
        eprintln!("skipping verify rows: no Metal 4 GPU");
        return;
    };
    let device = di.device.clone();
    let c = Case {
        seqs: vec![(1000, 4), (300, 3)],
        ..llama_3b("verify rows llama-3b bf16")
    };
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    let compress: [PipelineKey; 2] = [false, true].map(|_| {
        let (rot_dim, pair_off) = c.rope.map_or((0, 0), |r| (r.rot_dim, r.pair_off));
        let constants = TqCompressConstants {
            head_dim: HeadDim(c.head_dim as u32),
            bits: TqCodeBits(c.bits),
            num_kv_heads: NumKvHeads(c.num_kv_heads as u32),
            block_size: BlockSize(c.block_size as u32),
            blocks_per_chunk: BlocksPerChunk(c.blocks_per_chunk as u32),
            writeback: TqWriteback::Raw,
            offset: TqOffset::None,
            rot_dim: RotDim(rot_dim as u32),
            pair_off: RopePairOff(pair_off as u32),
        };
        PipelineKey::new("turboquant", c.dtype.compress(), constants.into())
    });
    cache.register_baked(&baked_kernels(&compress));
    let pso = |consts: Vec<ConstantValue>| {
        let name = format!("attention_via_cache_v2_{}_specialized", c.dtype.tag());
        let name: &'static str = Box::leak(name.into_boxed_str());
        baked_build(&cache, &PipelineKey::new("attention", name, consts)).expect("pipeline")
    };
    let f = &Fixture::new(&c);
    let (hd, nkv) = (c.head_dim, c.num_kv_heads);
    let n_q = *f.cu_seqlens.last().unwrap() as usize;
    let quant = PolarQuantizer::new(hd, c.bits, SEED);
    let boundaries: Vec<f32> = (quant.centroids().windows(2))
        .map(|w| (w[0] + w[1]) / 2.0)
        .collect();
    let n_slots = f.n_blocks * c.block_size;
    let pdim = packed_dim(hd, c.bits);
    let (signs, centroids, bounds) = (
        shared(&device, quant.signs()),
        shared(&device, quant.centroids()),
        shared(&device, &boundaries),
    );
    let cos_sin = shared(&device, &f.cos_sin);
    let q = shared(
        &device,
        &f.q.iter().map(|&x| c.dtype.bits(x)).collect::<Vec<_>>(),
    );
    // Every key quantized, the step's too.
    let (slots, pos): (Vec<u32>, Vec<u32>) = (0..c.seqs.len())
        .flat_map(|s| {
            (0..c.seqs[s].0).map(move |t| {
                let span = if f.span(s, t) { SPAN_BIT } else { 0 };
                (f.slot[s][t] as u32 | span, t as u32)
            })
        })
        .unzip();
    let (slots_buf, pos_buf) = (shared(&device, &slots), shared(&device, &pos));
    let codes = || shared(&device, &vec![0u32; n_slots * nkv * pdim]);
    let (packed_k, packed_v) = (codes(), codes());
    let norms = || shared(&device, &vec![0f32; n_slots * nkv]);
    let (norms_k, norms_v) = (norms(), norms());
    let no_bias = shared(&device, &vec![0u16; nkv * hd]);
    let all = |_: usize, _: usize| true;
    let (src_k, src_v) = (f.pool(&device, &f.k, 0, all), f.pool(&device, &f.v, 0, all));
    let mut batch = Mtl4DispatchBatch::begin(&device).expect("batch");
    for (src, packed, nrm, key) in [
        (&src_k, &packed_k, &norms_k, &compress[0]),
        (&src_v, &packed_v, &norms_v, &compress[1]),
    ] {
        batch.encode(
            &baked_build(&cache, key).expect("tq_compress_paged"),
            &[
                (&src.table, 0),
                (&slots_buf, 1),
                (&signs, 2),
                (&bounds, 3),
                (&centroids, 4),
                (packed, 5),
                (nrm, 6),
                (&slots_buf, 16),
                (&no_bias, 18),
                (&cos_sin, 19),
                (&pos_buf, 20),
            ],
            &[],
            &[],
            &[&src.data],
            tg(slots.len(), nkv, 1),
            tg(hd, 1, 1),
        );
    }
    batch.commit(true);

    // Each row alone: its sequence's block-table row, its keys, its slot.
    let row_seq: Vec<usize> = (0..c.seqs.len())
        .flat_map(|s| std::iter::repeat_n(s, c.seqs[s].1))
        .collect();
    let row_keys: Vec<u32> = (0..c.seqs.len())
        .flat_map(|s| (f.prefix(s)..c.seqs[s].0).map(|t| t as u32 + 1))
        .collect();
    let alone_table: Vec<u32> = (row_seq.iter())
        .flat_map(|&s| f.block_table[s * f.max_blocks..][..f.max_blocks].to_vec())
        .collect();
    let seq_keys: Vec<u32> = c.seqs.iter().map(|&(l, _)| l as u32).collect();
    let (seq_used, alone_used) = (shared(&device, &seq_keys), shared(&device, &row_keys));
    let (block_table, alone_table) = (
        shared(&device, &f.block_table),
        shared(&device, &alone_table),
    );
    let cu_seqlens = shared(&device, &f.cu_seqlens);
    let slot_mapping = shared(&device, &f.slot_mapping());
    let heads = c.decode_heads;
    for coded in [true, false] {
        // TurboQuant reads a row's own key from the cache its writer filled (NaN elsewhere); an
        // fp16 cache holds every key.
        let nan = c.dtype.bits(f32::NAN);
        let written = |s: usize, t: usize| !coded || f.written(s, t);
        let scratch_k = f.pool(&device, &f.k, nan, written);
        let scratch_v = f.pool(&device, &f.v, nan, written);
        let run = |per_token: bool| -> Vec<u16> {
            let mut consts = Vec::new();
            if coded {
                consts.extend([
                    ConstantValue::uint(13, c.bits),
                    ConstantValue::uint(16, heads),
                ]);
            }
            if c.rope.is_some_and(|r| r.coresident) {
                consts.push(ConstantValue::uint(12, 1));
            }
            if per_token {
                consts.push(ConstantValue::uint(22, 1));
            }
            let (used, table) = match per_token {
                true => (&seq_used, &block_table),
                false => (&alone_used, &alone_table),
            };
            let out = shared(&device, &vec![0u16; n_q * c.num_q_heads * hd]);
            let mut binds = vec![
                (&out, 0),
                (&q, 1),
                (used, 2),
                (table, 3),
                (&scratch_k.table, 4),
                (&scratch_v.table, 5),
                (&cos_sin, 6),
            ];
            if coded {
                binds.extend([
                    (&packed_k, 7),
                    (&packed_v, 8),
                    (&norms_k, 9),
                    (&norms_v, 10),
                    (&signs, 11),
                    (&centroids, 12),
                    (&slot_mapping, 13),
                ]);
            }
            if per_token {
                binds.push((&cu_seqlens, 23));
            }
            let groups = match coded {
                true => c.num_q_heads / heads as usize,
                false => c.num_q_heads,
            };
            let mut batch = Mtl4DispatchBatch::begin(&device).expect("batch");
            batch.encode(
                &pso(f.attn_constants(&consts)),
                &binds,
                &[],
                &[],
                &[&scratch_k.data, &scratch_v.data],
                tg(n_q, groups, 1),
                tg(1024, 1, 1),
            );
            batch.commit(true);
            read::<u16>(&out, n_q * c.num_q_heads * hd)
        };
        let (rows, alone) = (run(true), run(false));
        let what = if coded { "TurboQuant" } else { "fp16" };
        assert!(
            rows.iter().all(|&b| c.dtype.value(b).is_finite()),
            "{what}: a row read a key it does not see"
        );
        assert_eq!(
            rows, alone,
            "{what}: per-token rows differ from each row decoded alone"
        );
    }
}

// SPDX-License-Identifier: Apache-2.0
//! Parity of TurboQuant attention — which reads the packed KV store, never a
//! dequantized context — against a host f32 decode of the very same codes.
//!
//! - Decode (one query per sequence): `attention_via_cache_v2` with
//!   `ATTN_TQ_BITS`, reading the packed store in the codebook domain, each
//!   threadgroup serving the production `TqDecodeHeads` query heads.
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
//! GPU tests — run with `--test-threads=1` (standing rule).

use half::{bf16, f16};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_layers::turboquant::{PolarQuantizer, packed_dim, unpack_indices};
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::Mtl4DispatchBatch;
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use scratchy_target_metal::tape::ids::{HeadDim, NumKvHeads, NumQHeads, TqDecodeHeads};

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
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let w: Vec<f32> = scores.iter().map(|&x| (x - m).exp()).collect();
                let l: f32 = w.iter().sum();
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
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");
    let pso = |lib: &'static str, name: String, consts: Vec<ConstantValue>| {
        let name: &'static str = Box::leak(name.into_boxed_str());
        cache
            .get_or_build(&PipelineKey::new(lib, name, consts))
            .expect("pipeline")
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
    let offset_on = restore && c.bias.is_some();
    let (rot_dim, pair_off) = c
        .rope
        .map_or((0, 0), |r| (r.rot_dim as u32, r.pair_off as u32));
    // `tq_offset` mode per operand: K's bias rotated (2), V's as-is (1).
    let mode = |is_v: bool| match (offset_on, is_v) {
        (false, _) => 0u32,
        (true, false) => 2,
        (true, true) => 1,
    };
    let src_k = f.pool(&device, &f.k, 0, cached);
    let src_v = f.pool(&device, &f.v, 0, cached);
    let compress = pso("turboquant", c.dtype.compress().to_owned(), vec![]);
    let mut batch = Mtl4DispatchBatch::begin(&device)?;
    for (src, packed, norms, bias, is_v) in [
        (&src_k, &packed_k, &norms_k, &kb, false),
        (&src_v, &packed_v, &norms_v, &vb, true),
    ] {
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
            &[
                (hd as u32, 7),
                (c.bits, 8),
                (32 / c.bits, 9),
                (pdim as u32, 10),
                (1 << c.bits, 11),
                (nkv as u32, 13),
                (bs as u32, 14),
                (c.blocks_per_chunk as u32, 15),
                (0, 17),
                (mode(is_v), 21),
                (rot_dim, 22),
                (pair_off, 23),
            ],
            &[(quant.scale(), 12)],
            &[&src.data],
            tg(quant_slots.len(), nkv, 1),
            tg(hd, 1, 1),
        );
    }
    batch.commit(true).expect("compress");

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
    if decode {
        let heads = c.decode_heads;
        let mut consts = f.attn_constants(&[bits, ConstantValue::uint(16, heads)]);
        if c.rope.is_some_and(|r| r.coresident) {
            consts.push(ConstantValue::uint(12, 1));
        }
        if offset_on {
            consts.extend([k_bias, v_bias]);
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
        for (scratch, packed, norms, bias, is_k) in [
            (&scratch_k, &packed_k, &norms_k, &kb, true),
            (&scratch_v, &packed_v, &norms_v, &vb, false),
        ] {
            let stage = pso("attention", stage_name.clone(), stage_consts(is_k));
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
        let attention = pso(
            "attention",
            format!(
                "attention_prefill_sdpa_v2_paged_{}_specialized",
                c.dtype.tag()
            ),
            f.attn_constants(&[]),
        );
        batch.encode(
            &attention,
            &[
                (&out, 0),
                (&q, 1),
                (&cu_seqlens, 2),
                (&seq_used, 3),
                (&block_table, 4),
                (&scratch_k.table, 5),
                (&scratch_v.table, 6),
                (&cos_sin, 7),
                (&span_ids, 8),
            ],
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
    batch.commit(true).expect("attention");
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
    if !c.seqs.iter().all(|&(_, new)| new == 1) {
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

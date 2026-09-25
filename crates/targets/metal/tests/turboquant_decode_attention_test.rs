// SPDX-License-Identifier: Apache-2.0
//! Parity of the TurboQuant decode attention (`attention_via_cache_v2` with
//! `ATTN_TQ_BITS`, which reads the packed KV store in the codebook domain)
//! against the path it replaces: `tq_dequant_blocktable` into the fp16 scratch,
//! the step's own key written raw on top, then plain `attention_via_cache_v2`.
//!
//! Each case quantizes the context with the production `tq_compress_paged`
//! kernel into packed stores whose every other slot holds stale garbage (a
//! reused block), and hands the fused kernel a scratch that is NaN everywhere
//! but the step's own slot — so reading anything but the packed store and that
//! one key poisons the output. The fused result must match the dequant path to
//! its rounding, and a host f32 decode of the same codes more tightly still.
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
    fn nan(self) -> u16 {
        self.bits(f32::NAN)
    }
    fn suffix(self) -> &'static str {
        match self {
            Dtype::F16 => "",
            Dtype::Bf16 => "_bf16",
        }
    }
    fn attention(self) -> &'static str {
        match self {
            Dtype::F16 => "attention_via_cache_v2_f16_specialized",
            Dtype::Bf16 => "attention_via_cache_v2_bf16_specialized",
        }
    }
}

/// NeoX rope-on-read geometry: `(rot_dim, pair_off, pair_coresident)`.
#[derive(Clone, Copy)]
struct Rope {
    rot_dim: usize,
    pair_off: usize,
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
    /// Per sequence: context length including the step's own key.
    kv_lens: Vec<usize>,
    /// Logical blocks (all sequences) whose K is stored unrotated (spans).
    span_blocks: Vec<usize>,
    /// The step's own key sits in a reused span block: its slot is the
    /// write-skip sentinel and the key was quantized by an earlier request.
    tail_write_skipped: bool,
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

/// A paged KV buffer `[block][kv_head][token][head_dim]` of the model dtype
/// plus its chunk-address table.
struct Pool {
    data: Buffer,
    table: Buffer,
}

impl Pool {
    fn new(device: &Device, c: &Case, n_blocks: usize, elems: &[u16]) -> Self {
        let data = shared(device, elems);
        let blk_bytes = (c.num_kv_heads * c.block_size * c.head_dim * 2) as u64;
        let table: Vec<u64> = if c.blocks_per_chunk == 0 {
            vec![data.gpuAddress()]
        } else {
            (0..n_blocks.div_ceil(c.blocks_per_chunk))
                .map(|ch| data.gpuAddress() + (ch * c.blocks_per_chunk) as u64 * blk_bytes)
                .collect()
        };
        let table = shared(device, &table);
        Self { data, table }
    }
}

/// Everything a case's GPU runs share.
struct Fixture {
    c: Case,
    n_blocks: usize,
    max_blocks: usize,
    /// Row per sequence, `max_blocks` wide, span bit applied.
    block_table: Vec<u32>,
    /// Physical slot of each sequence's token `t`.
    slot: Vec<Vec<usize>>,
    /// Plain K/V per sequence, `[t][kv_head][head_dim]`, dtype-rounded.
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    q: Vec<f32>,
    cos_sin: Vec<u16>,
}

impl Fixture {
    fn new(c: &Case) -> Self {
        let mut rng = Lcg(0x5eed ^ c.head_dim as u64 ^ ((c.bits as u64) << 20));
        let blocks_per_seq: Vec<usize> =
            c.kv_lens.iter().map(|l| l.div_ceil(c.block_size)).collect();
        let max_blocks = blocks_per_seq.iter().copied().max().unwrap_or(1).max(1);
        let n_blocks = blocks_per_seq.iter().sum::<usize>() + 3;
        // A non-identity physical placement: stride through the pool.
        let stride = (0..)
            .map(|s| 2 * s + 3)
            .find(|s| gcd(*s, n_blocks) == 1)
            .unwrap();
        let mut next_phys = (0..n_blocks).map(|i| (i * stride + 1) % n_blocks);
        let mut block_table = vec![0u32; c.kv_lens.len() * max_blocks];
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
                (0..c.kv_lens[s])
                    .map(|t| phys[t / c.block_size] * c.block_size + t % c.block_size)
                    .collect(),
            );
        }
        let per_tok = c.num_kv_heads * c.head_dim;
        let mut vecs =
            |n: usize| -> Vec<f32> { (0..n).map(|_| c.dtype.round(rng.gauss())).collect() };
        let k: Vec<Vec<f32>> = c.kv_lens.iter().map(|&l| vecs(l * per_tok)).collect();
        let v: Vec<Vec<f32>> = c.kv_lens.iter().map(|&l| vecs(l * per_tok)).collect();
        let q = vecs(c.kv_lens.len() * c.num_q_heads * c.head_dim);
        let max_len = c.kv_lens.iter().copied().max().unwrap_or(0);
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
        Self {
            c: c.clone(),
            n_blocks,
            max_blocks,
            block_table,
            slot,
            k,
            v,
            q,
            cos_sin,
        }
    }

    /// A pool filled with `fill`, then `write`.
    fn pool(
        &self,
        device: &Device,
        of: &[Vec<f32>],
        fill: u16,
        tokens: impl Fn(usize, usize) -> bool,
    ) -> Pool {
        let c = &self.c;
        let elems = vec![fill; self.n_blocks * c.num_kv_heads * c.block_size * c.head_dim];
        let pool = Pool::new(device, c, self.n_blocks, &elems);
        self.write(&pool, of, tokens);
        pool
    }

    /// Write `tokens(s, t)` of `of` raw at their slots — what the KV writer does.
    fn write(&self, pool: &Pool, of: &[Vec<f32>], tokens: impl Fn(usize, usize) -> bool) {
        let c = &self.c;
        let base = pool.data.contents().as_ptr() as *mut u16;
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
    }

    fn is_tail(&self, s: usize, t: usize) -> bool {
        t + 1 == self.c.kv_lens[s]
    }

    /// Whether token `t` of sequence `s` lives in the packed store.
    fn quantized(&self, s: usize, t: usize) -> bool {
        !self.is_tail(s, t) || self.c.tail_write_skipped
    }

    fn attn_constants(&self, tq: bool) -> Vec<ConstantValue> {
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
        if let Some(r) = c.rope {
            v.push(ConstantValue::uint(8, r.rot_dim as u32));
            v.push(ConstantValue::uint(9, r.pair_off as u32));
            v.push(ConstantValue::uint(10, 1));
            if r.coresident {
                v.push(ConstantValue::uint(12, 1));
            }
        }
        if tq {
            v.push(ConstantValue::uint(13, c.bits));
        }
        v
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

struct Outputs {
    fused: Vec<f32>,
    dequant_path: Vec<f32>,
    ideal: Vec<f32>,
}

fn run_case(c: &Case) -> Option<Outputs> {
    let Some(di) = detect_device() else {
        eprintln!("skipping {}: no Metal 4 GPU", c.name);
        return None;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");
    let pso = |lib: &'static str, name: &str, consts: Vec<ConstantValue>| {
        let name: &'static str = Box::leak(name.to_owned().into_boxed_str());
        cache
            .get_or_build(&PipelineKey::new(lib, name, consts))
            .expect("pipeline")
    };
    let f = &Fixture::new(c);
    let (hd, nkv, bs) = (c.head_dim, c.num_kv_heads, c.block_size);
    let n_seqs = c.kv_lens.len();
    let quant = PolarQuantizer::new(hd, c.bits, SEED);
    let boundaries: Vec<f32> = quant
        .centroids()
        .windows(2)
        .map(|w| (w[0] + w[1]) / 2.0)
        .collect();
    let pdim = packed_dim(hd, c.bits);
    let vpw = 32 / c.bits;
    let n_slots = f.n_blocks * bs;

    let signs = shared(&device, quant.signs());
    let centroids = shared(&device, quant.centroids());
    let bounds = shared(&device, &boundaries);
    let block_table = shared(&device, &f.block_table);
    let seq_used = shared(
        &device,
        &c.kv_lens.iter().map(|&l| l as u32).collect::<Vec<_>>(),
    );
    let cos_sin = shared(&device, &f.cos_sin);
    let q = shared(
        &device,
        &f.q.iter().map(|&x| c.dtype.bits(x)).collect::<Vec<_>>(),
    );

    // Packed stores: stale garbage in every slot (block reuse), then the
    // production compress over every quantized token.
    let mut rng = Lcg(7);
    let stale_codes: Vec<u32> = (0..n_slots * nkv * pdim)
        .map(|_| rng.0 as u32 ^ (rng.next().to_bits()))
        .collect();
    let stale_norms: Vec<f32> = (0..n_slots * nkv)
        .map(|_| 1.0 + rng.next().abs() * 4.0)
        .collect();
    let (packed_k, packed_v) = (shared(&device, &stale_codes), shared(&device, &stale_codes));
    let (norms_k, norms_v) = (shared(&device, &stale_norms), shared(&device, &stale_norms));
    let quant_slots: Vec<u32> = (0..n_seqs)
        .flat_map(|s| {
            (0..c.kv_lens[s])
                .filter(move |&t| f.quantized(s, t))
                .map(move |t| f.slot[s][t] as u32)
        })
        .collect();
    let quant_slots_buf = shared(&device, &quant_slots);
    let src_k = f.pool(&device, &f.k, 0, |s, t| f.quantized(s, t));
    let src_v = f.pool(&device, &f.v, 0, |s, t| f.quantized(s, t));
    let compress = pso(
        "turboquant",
        &format!("tq_compress_paged{}", c.dtype.suffix()),
        vec![],
    );
    let dequant = pso(
        "turboquant",
        &format!("tq_dequant_blocktable{}", c.dtype.suffix()),
        vec![],
    );
    let mut batch = Mtl4DispatchBatch::begin(&device)?;
    for (src, packed, norms) in [(&src_k, &packed_k, &norms_k), (&src_v, &packed_v, &norms_v)] {
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
            ],
            &[
                (hd as u32, 7),
                (c.bits, 8),
                (vpw, 9),
                (pdim as u32, 10),
                (1 << c.bits, 11),
                (nkv as u32, 13),
                (bs as u32, 14),
                (c.blocks_per_chunk as u32, 15),
                (0, 17),
            ],
            &[(quant.scale(), 12)],
            &[&src.data],
            tg(quant_slots.len(), nkv, 1),
            tg(hd, 1, 1),
        );
    }
    batch.commit(true).expect("compress");

    // Today's path: dequant the whole context into the scratch, the writer's
    // raw key on top (skipped on the write-skip sentinel), plain attention.
    let written = |s: usize, t: usize| f.is_tail(s, t) && !c.tail_write_skipped;
    let scratch_k = f.pool(&device, &f.k, 0, |_, _| false);
    let scratch_v = f.pool(&device, &f.v, 0, |_, _| false);
    let mut batch = Mtl4DispatchBatch::begin(&device)?;
    for (scratch, packed, norms) in [
        (&scratch_k, &packed_k, &norms_k),
        (&scratch_v, &packed_v, &norms_v),
    ] {
        batch.encode(
            &dequant,
            &[
                (&scratch.table, 0),
                (&block_table, 1),
                (&seq_used, 2),
                (&signs, 3),
                (&centroids, 4),
                (packed, 5),
                (norms, 6),
            ],
            &[
                (hd as u32, 7),
                (c.bits, 8),
                (vpw, 9),
                (pdim as u32, 10),
                (nkv as u32, 12),
                (bs as u32, 13),
                (c.blocks_per_chunk as u32, 14),
            ],
            &[(quant.scale(), 11)],
            &[&scratch.data],
            tg(f.max_blocks, nkv, n_seqs),
            tg(hd, 1, 1),
        );
    }
    batch.commit(true).expect("dequant");
    f.write(&scratch_k, &f.k, written);
    f.write(&scratch_v, &f.v, written);
    let out_len = n_seqs * c.num_q_heads * hd;
    let slot_mapping: Vec<u32> = (0..n_seqs)
        .map(|s| {
            if c.tail_write_skipped {
                WRITE_SKIP
            } else {
                f.slot[s][c.kv_lens[s] - 1] as u32
            }
        })
        .collect();
    let slot_mapping = shared(&device, &slot_mapping);
    let attend = |tq: bool, scratch_k: &Pool, scratch_v: &Pool| -> Option<Vec<f32>> {
        let out = shared(&device, &vec![0u16; out_len]);
        let pipeline = pso("attention", c.dtype.attention(), f.attn_constants(tq));
        let mut binds = vec![
            (&out, 0),
            (&q, 1),
            (&seq_used, 2),
            (&block_table, 3),
            (&scratch_k.table, 4),
            (&scratch_v.table, 5),
            (&cos_sin, 6),
        ];
        if tq {
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
        let mut batch = Mtl4DispatchBatch::begin(&device)?;
        batch.encode(
            &pipeline,
            &binds,
            &[],
            &[],
            &[&scratch_k.data, &scratch_v.data],
            tg(n_seqs, c.num_q_heads, 1),
            tg(1024, 1, 1),
        );
        batch.commit(true).expect("attention");
        Some(
            read::<u16>(&out, out_len)
                .into_iter()
                .map(|b| c.dtype.value(b))
                .collect(),
        )
    };
    let dequant_path = attend(false, &scratch_k, &scratch_v)?;

    // The fused kernel's scratch: NaN except the step's own (written) key.
    let nan = c.dtype.nan();
    let fused_k = f.pool(&device, &f.k, nan, written);
    let fused_v = f.pool(&device, &f.v, nan, written);
    let fused = attend(true, &fused_k, &fused_v)?;

    // Host f32 decode of the very codes the GPU wrote.
    let codes_k: Vec<u32> = read(&packed_k, n_slots * nkv * pdim);
    let codes_v: Vec<u32> = read(&packed_v, n_slots * nkv * pdim);
    let nk: Vec<f32> = read(&norms_k, n_slots * nkv);
    let nv: Vec<f32> = read(&norms_v, n_slots * nkv);
    let decode = |codes: &[u32], norms: &[f32], slot: usize, h: usize| -> Vec<f32> {
        let row = slot * nkv + h;
        quant.dequantize(
            &unpack_indices(&codes[row * pdim..][..pdim], c.bits, hd),
            norms[row],
        )
    };
    let ideal = ideal_attention(f, |s, t, h, is_v| {
        let plain = if is_v { &f.v[s] } else { &f.k[s] };
        if f.quantized(s, t) {
            let x = if is_v {
                decode(&codes_v, &nv, f.slot[s][t], h)
            } else {
                decode(&codes_k, &nk, f.slot[s][t], h)
            };
            // Today's dequant pass stores span keys rounded, then re-ropes them.
            if !is_v && f.block_table[s * f.max_blocks + t / bs] & SPAN_BIT != 0 {
                x.iter().map(|&e| c.dtype.round(e)).collect()
            } else {
                x
            }
        } else {
            plain[(t * nkv + h) * hd..][..hd].to_vec()
        }
    });
    Some(Outputs {
        fused,
        dequant_path,
        ideal,
    })
}

/// f32 softmax attention of each sequence's single query over `kv(s, t, h,
/// is_v)`, re-roping span keys like `rope_on_read_*` (rounded to the dtype).
fn ideal_attention(f: &Fixture, kv: impl Fn(usize, usize, usize, bool) -> Vec<f32>) -> Vec<f32> {
    let c = &f.c;
    let (hd, nq) = (c.head_dim, c.num_q_heads);
    let group = nq / c.num_kv_heads;
    let mut out = vec![0f32; c.kv_lens.len() * nq * hd];
    for (s, &len) in c.kv_lens.iter().enumerate() {
        for h in 0..nq {
            let kvh = h / group;
            let qrow = &f.q[(s * nq + h) * hd..][..hd];
            let mut scores = Vec::with_capacity(len);
            let mut vals = Vec::with_capacity(len);
            for t in 0..len {
                if c.window > 0 && (len - 1 - t) as i64 >= c.window as i64 {
                    continue;
                }
                let mut k = kv(s, t, kvh, false);
                if let Some(r) = c.rope
                    && f.block_table[s * f.max_blocks + t / c.block_size] & SPAN_BIT != 0
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
                scores.push(c.attn_scale * qrow.iter().zip(&k).map(|(a, b)| a * b).sum::<f32>());
                vals.push(kv(s, t, kvh, true));
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let w: Vec<f32> = scores.iter().map(|&x| (x - m).exp()).collect();
            let l: f32 = w.iter().sum();
            let orow = &mut out[(s * nq + h) * hd..][..hd];
            for (wi, v) in w.iter().zip(&vals) {
                for d in 0..hd {
                    orow[d] += wi / l * v[d];
                }
            }
        }
    }
    out
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn check(c: Case) {
    let Some(o) = run_case(&c) else { return };
    assert!(
        o.fused.iter().all(|x| x.is_finite()),
        "{}: fused output is not finite — it read the scratch",
        c.name
    );
    let peak = o.ideal.iter().fold(0f32, |m, x| m.max(x.abs()));
    // The output is stored in the model dtype; the dequant path additionally
    // rounds every decoded K/V element to it before attention reads them.
    let ulp = match c.dtype {
        Dtype::F16 => 1.0 / 1024.0,
        Dtype::Bf16 => 1.0 / 128.0,
    };
    let (vs_ideal, vs_dequant) = (
        max_abs(&o.fused, &o.ideal),
        max_abs(&o.fused, &o.dequant_path),
    );
    let dequant_vs_ideal = max_abs(&o.dequant_path, &o.ideal);
    eprintln!(
        "{}: peak {peak:.3}  fused-ideal {vs_ideal:.2e}  fused-dequant {vs_dequant:.2e}  dequant-ideal {dequant_vs_ideal:.2e}",
        c.name
    );
    assert!(
        vs_ideal <= peak * ulp,
        "{}: fused vs f32 decode of the same codes {vs_ideal} > {}",
        c.name,
        peak * ulp
    );
    assert!(
        vs_dequant <= peak * ulp * 2.0,
        "{}: fused vs the dequant path {vs_dequant} > {}",
        c.name,
        peak * ulp * 2.0
    );
}

/// Llama-3.2-3B decode: the production geometry (bf16, 3-bit, GQA 3, NeoX
/// rope-on-read with the co-resident lane layout).
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
        kv_lens: vec![1000],
        span_blocks: vec![],
        tail_write_skipped: false,
    }
}

#[test]
fn llama_3b_bf16() {
    check(llama_3b("llama-3b bf16"));
}

#[test]
fn llama_3b_f16() {
    check(Case {
        dtype: Dtype::F16,
        ..llama_3b("llama-3b f16")
    });
}

/// Span blocks (stored unrotated, re-roped on read) — including the block
/// holding the step's own key.
#[test]
fn llama_3b_span_blocks() {
    check(Case {
        span_blocks: vec![0, 3, 4, 62],
        ..llama_3b("llama-3b spans")
    });
}

/// The step's key sits in a reused span block: nothing was written to the
/// scratch, so it too must come from the packed store.
#[test]
fn llama_3b_tail_write_skipped() {
    check(Case {
        span_blocks: vec![62],
        tail_write_skipped: true,
        ..llama_3b("llama-3b write-skipped tail")
    });
}

/// Two sequences in one dispatch, one of them only its own key.
#[test]
fn llama_3b_two_sequences() {
    check(Case {
        kv_lens: vec![333, 1],
        ..llama_3b("llama-3b two seqs")
    });
}

/// Llama-3.2-1B geometry on the contiguous lane layout, chunked addressing
/// across two chunks.
#[test]
fn head_dim_64_chunked() {
    check(Case {
        name: "hd64 chunked",
        head_dim: 64,
        num_q_heads: 32,
        num_kv_heads: 8,
        blocks_per_chunk: 128,
        attn_scale: 0.125,
        rope: None,
        kv_lens: vec![2100],
        ..llama_3b("")
    });
}

/// 4-bit, head_dim 256, one KV head, sliding window.
#[test]
fn head_dim_256_sliding_window() {
    check(Case {
        name: "hd256 window",
        head_dim: 256,
        num_q_heads: 8,
        num_kv_heads: 1,
        bits: 4,
        attn_scale: 1.0 / 16.0,
        window: 100,
        rope: None,
        kv_lens: vec![700],
        ..llama_3b("")
    });
}

/// Gemma-4 global layers: head_dim 512, 4-bit, 64-token pages,
/// proportional rope (pair offset head_dim/2), attention scale 1.
#[test]
fn gemma4_global_head_dim_512() {
    check(Case {
        name: "gemma4 global",
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
        kv_lens: vec![900],
        span_blocks: vec![1, 14],
        ..llama_3b("")
    });
}

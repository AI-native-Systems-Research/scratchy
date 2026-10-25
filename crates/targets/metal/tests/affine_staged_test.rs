// SPDX-License-Identifier: Apache-2.0
//! Splash's staged decode tiles over MLX-affine 4-bit weights (`affine_staged.metal`): the decode tile and the
//! one-pass gate/up tile against a host f32 reference at every K split, both weight layouts, spread on and off;
//! and, as an ignored probe, both against the wide matvec and the small-M twin on a chip emulated by width.
//!
//! GPU tests (NAX hardware only; skipped elsewhere) — run with `--test-threads=1`.
//!
//! Probe: cargo test -p scratchy-target-metal --release --test affine_staged_test -- \
//!     --ignored --nocapture --test-threads=1

use half::bf16;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::Mtl4DispatchBatch;
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use scratchy_target_metal::targets::is_nax_capable;

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;
type Pipeline =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>;

const GROUP: usize = 64;
/// Columns a decode-tile threadgroup covers (`GGUF_TILE_COLUMNS`).
const TILE: usize = 64;
/// Columns a gate/up threadgroup covers (`GGUF_STAGED_COLUMNS`).
const PAIR_TILE: usize = 32;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1u64 << 24) as f32)
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

fn read_bf16(buf: &Buffer, len: usize) -> Vec<f32> {
    let p = buf.contents().as_ptr() as *const u16;
    (0..len)
        .map(|i| bf16::from_bits(unsafe { *p.add(i) }).to_f32())
        .collect()
}

fn counters_zero(buf: &Buffer, len: usize) -> bool {
    let p = buf.contents().as_ptr() as *const u32;
    (0..len).all(|t| unsafe { *p.add(t) } == 0)
}

fn size(width: usize, height: usize) -> MTLSize {
    MTLSize {
        width,
        height,
        depth: 1,
    }
}

fn bf16_rows(rng: &mut Rng, len: usize) -> Vec<u16> {
    (0..len)
        .map(|_| bf16::from_f32(2.0 * rng.unit() - 1.0).to_bits())
        .collect()
}

/// A staged kernel (`decode` or `gate_up`) at `bucket` rows (`m_8` or `m_16` tile), codes stored offset-8,
/// over MLX's planes or (`planes`) Splash's tile-major image of them.
fn staged(
    cache: &SpecializedPipelineCache,
    kernel: &str,
    (k, n): (usize, usize),
    bucket: usize,
    planes: bool,
    spread: bool,
) -> Pipeline {
    let rows = if bucket <= 8 { 8 } else { 16 };
    let layout = if planes { "_planes" } else { "" };
    let name: &'static str =
        format!("affine_staged_{kernel}{layout}_bf16_s_bf16_gs_64_b_4_m_{rows}").leak();
    let consts = vec![
        ConstantValue::int(0, k as i32),
        ConstantValue::int(1, n as i32),
        ConstantValue::int(2, bucket as i32),
        ConstantValue::boolean(3, spread),
        ConstantValue::boolean(5, true),
    ];
    baked_build(cache, &PipelineKey::new("affine_staged", name, consts)).expect(name)
}

/// MLX-affine 4-bit weights of `n` columns over `k` inputs, the codes stored offset-8: the packed words, bf16
/// scales and biases, and each weight's code.
struct Weights {
    packed: Vec<u32>,
    scales: Vec<u16>,
    biases: Vec<u16>,
    codes: Vec<u8>,
}

impl Weights {
    fn random(rng: &mut Rng, n: usize, k: usize) -> Self {
        let codes: Vec<u8> = (0..n * k).map(|_| (rng.next() & 0xf) as u8).collect();
        let packed = codes
            .chunks(8)
            .map(|c| {
                (c.iter().enumerate()).fold(0u32, |w, (i, &q)| w | ((q ^ 8) as u32) << (4 * i))
            })
            .collect();
        let groups = n * k / GROUP;
        let scales = (0..groups)
            .map(|_| bf16::from_f32(0.01 + 0.04 * rng.unit()).to_bits())
            .collect();
        let biases = (0..groups)
            .map(|_| bf16::from_f32(rng.unit() - 0.5).to_bits())
            .collect();
        Self {
            packed,
            scales,
            biases,
            codes,
        }
    }

    /// Splash's tile-major image (metal/abi/QuantFormat.h): a 64-column tile's group of 32 is its columns' 16
    /// code bytes side by side, and its 64-group meta unit their (scale, bias) pairs.
    fn planes(&self, n: usize, k: usize) -> (Vec<u32>, Vec<u32>) {
        let (steps, kg) = (k / 32, k / GROUP);
        let mut codes = vec![0u32; n * k / 8];
        let mut meta = vec![0u32; n * kg];
        for c in 0..n {
            let (tile, col) = (c / TILE, c % TILE);
            for g in 0..steps {
                let at = ((tile * steps + g) * TILE + col) * 4;
                codes[at..at + 4].copy_from_slice(&self.packed[c * k / 8 + g * 4..][..4]);
            }
            for u in 0..kg {
                meta[(tile * kg + u) * TILE + col] =
                    self.scales[c * kg + u] as u32 | (self.biases[c * kg + u] as u32) << 16;
            }
        }
        (codes, meta)
    }

    /// The buffers a kernel binds for these weights: codes, scales (the meta, with `planes`) and biases.
    fn bind(&self, device: &Device, n: usize, k: usize, planes: bool) -> [Buffer; 3] {
        match planes {
            false => [
                shared(device, &self.packed),
                shared(device, &self.scales),
                shared(device, &self.biases),
            ],
            true => {
                let (codes, meta) = self.planes(n, k);
                [
                    shared(device, &codes),
                    shared(device, &meta),
                    shared(device, &self.biases),
                ]
            }
        }
    }

    fn value(&self, k: usize, col: usize, kk: usize) -> f32 {
        let g = col * (k / GROUP) + kk / GROUP;
        let s = bf16::from_bits(self.scales[g]).to_f32();
        let b = bf16::from_bits(self.biases[g]).to_f32();
        s * self.codes[col * k + kk] as f32 + b
    }

    /// `x · Wᵀ` over the first `rows` rows, in f64.
    fn reference(&self, x: &[u16], rows: usize, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0f32; rows * n];
        for r in 0..rows {
            for c in 0..n {
                let mut acc = 0f64;
                for kk in 0..k {
                    acc += (bf16::from_bits(x[r * k + kk]).to_f32() * self.value(k, c, kk)) as f64;
                }
                out[r * n + c] = acc as f32;
            }
        }
        out
    }
}

fn worst(got: &[f32], want: &[f32]) -> f32 {
    (got.iter().zip(want))
        .map(|(g, w)| (g - w).abs() / (1.0 + w.abs()))
        .fold(0f32, f32::max)
}

fn nax() -> Option<(Device, SpecializedPipelineCache)> {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return None;
    };
    if !is_nax_capable(di.profile.generation) {
        eprintln!("skipping: no NAX matrix unit");
        return None;
    }
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("pipeline cache");
    Some((device, cache))
}

#[test]
fn staged_tile_matches_the_reference_at_every_split() {
    let Some((device, cache)) = nax() else {
        return;
    };
    for (n, k, bucket, rows) in [
        (128, 2048, 4, 3),
        (256, 4096, 8, 8),
        (64, 512, 4, 1),
        (192, 2048, 16, 11),
    ] {
        let mut rng = Rng(0x5eed ^ (n * 31 + k) as u64);
        let w = Weights::random(&mut rng, n, k);
        let x = bf16_rows(&mut rng, bucket * k);
        let want = w.reference(&x, rows, n, k);
        let xb = shared(&device, &x);
        for (planes, spread) in [(false, false), (true, false), (true, true)] {
            let pso = staged(&cache, "decode", (k, n), bucket, planes, spread);
            let [wb, sb, bb] = w.bind(&device, n, k, planes);
            for splits in [1usize, 2, 4] {
                let y = shared(&device, &vec![0u16; bucket * n]);
                let partials = shared(&device, &vec![0f32; splits * bucket * n]);
                let counters = shared(&device, &vec![0u32; n / TILE]);
                let mut batch = Mtl4DispatchBatch::begin(&device).expect("mtl4");
                let binds = [
                    (&wb, 0),
                    (&sb, 1),
                    (&bb, 2),
                    (&xb, 3),
                    (&y, 4),
                    (&partials, 5),
                    (&counters, 6),
                ];
                let grid = size(n / TILE, splits);
                batch.encode(&pso, &binds, &[], &[], &[], grid, size(64, 1));
                batch.commit(true);
                let worst = worst(&read_bf16(&y, rows * n), &want);
                assert!(
                    worst < 1.0 / 64.0,
                    "n {n} k {k} bucket {bucket} rows {rows} splits {splits} planes {planes} \
                     spread {spread}: worst relative error {worst}"
                );
                assert!(counters_zero(&counters, n / TILE), "a counter left nonzero");
            }
        }
    }
}

#[test]
fn staged_gate_up_matches_the_reference_at_every_split() {
    let Some((device, cache)) = nax() else {
        return;
    };
    for (n, k, bucket, rows) in [(128, 2048, 8, 8), (64, 1024, 4, 3)] {
        let mut rng = Rng(0x9a7e ^ (n * 31 + k) as u64);
        let (gate, up) = (
            Weights::random(&mut rng, n, k),
            Weights::random(&mut rng, n, k),
        );
        let x = bf16_rows(&mut rng, bucket * k);
        let (gr, ur) = (gate.reference(&x, rows, n, k), up.reference(&x, rows, n, k));
        let silu = |g: f32| g / (1.0 + (-g).exp());
        let round = |v: f32| bf16::from_f32(v).to_f32();
        let want: Vec<f32> = (gr.iter().zip(&ur))
            .map(|(&g, &u)| round(u) * silu(round(g)))
            .collect();
        let xb = shared(&device, &x);
        for planes in [false, true] {
            let pso = staged(&cache, "gate_up", (k, n), bucket, planes, true);
            let [gw, gs, gb] = gate.bind(&device, n, k, planes);
            let [uw, us, ub] = up.bind(&device, n, k, planes);
            for splits in [1usize, 2, 4] {
                let y = shared(&device, &vec![0u16; bucket * n]);
                let partials = shared(&device, &vec![0f32; 2 * splits * bucket * n]);
                let counters = shared(&device, &vec![0u32; n / PAIR_TILE]);
                let mut batch = Mtl4DispatchBatch::begin(&device).expect("mtl4");
                let binds = [
                    (&gw, 0),
                    (&gs, 1),
                    (&gb, 2),
                    (&uw, 3),
                    (&us, 4),
                    (&ub, 5),
                    (&xb, 6),
                    (&y, 7),
                    (&partials, 8),
                    (&counters, 9),
                ];
                let grid = size(n / PAIR_TILE, splits);
                batch.encode(&pso, &binds, &[], &[], &[], grid, size(32, 1));
                batch.commit(true);
                let worst = worst(&read_bf16(&y, rows * n), &want);
                assert!(
                    worst < 1.0 / 32.0,
                    "n {n} k {k} bucket {bucket} rows {rows} splits {splits} planes {planes}: \
                     worst relative error {worst}"
                );
                assert!(
                    counters_zero(&counters, n / PAIR_TILE),
                    "a counter left nonzero"
                );
            }
        }
    }
}

/// Splash's K split rule (runtime/ops/LinearGguf.cpp:46-54 `decodeSplits`) with its staged tile's tier
/// (`kStagedTiers`, :82: six threadgroups a core, 512 inputs a partition).
fn decode_splits(n: usize, k: usize, cores: usize) -> usize {
    let (tgs, inputs, cap) = (6, 512, 8);
    let grid = n / TILE;
    let mut splits = 1;
    while (k / 32).is_multiple_of(2 * splits)
        && splits < cap
        && grid * splits < tgs * cores
        && k / (2 * splits) >= inputs
    {
        splits *= 2;
    }
    splits
}

/// Weight bytes a timed command buffer streams at least: enough dispatches that the GPU holds its clock
/// through them (a few hundred µs of small dispatches leave it ramping).
const BATCH_BYTES: usize = 1 << 30;
/// Dispatches per timed command buffer, at least and at most.
const ITERS: std::ops::RangeInclusive<usize> = 48..=4096;
/// Weight bytes the copies span, past the system cache.
const ROTATED_BYTES: usize = 96 << 20;
/// Timed rounds of a shape's cells, after a warm-up round.
const ROUNDS: usize = 7;

/// The cores a probe emulates by width (Splash's "10-, 30- and 40-core GPUs emulated by width",
/// LinearGguf.cpp:79): every shape's columns divided by `EMULATED / device cores`, so the device's cores see the
/// threadgroups a core the emulated chip's would, and the staged plan is the emulated chip's.
const EMULATED: usize = 40;

/// Qwen3.6-35B-A3B's 4-bit dense projections a decode or verify bucket runs, and its DFlash2 drafter's,
/// `(name, n, k)`.
const SHAPES: &[(&str, usize, usize)] = &[
    ("lm_head", 248320, 2048),
    ("gdn qkv / attn q", 8192, 2048),
    ("gdn z", 4096, 2048),
    ("out_proj", 2048, 4096),
    ("shared down", 2048, 512),
    ("gdn a / b", 32, 2048),
    ("drafter qkv", 6144, 2048),
    ("drafter down", 2048, 6144),
    ("drafter conv proj", 512, 2048),
];

/// Gate and up projections that run as one pair: the target's shared expert and the drafter's MLP.
const PAIRS: &[(&str, usize, usize)] = &[
    ("shared gate|up", 512, 2048),
    ("drafter gate|up", 6144, 2048),
];

/// What a probe dispatch binds: one tensor's weights in a layout (the tensor's index), or the gate/up pair's.
#[derive(Clone, Copy)]
enum Binds {
    Mlx(usize),
    Planes(usize),
    PairMlx,
    PairPlanes,
}

/// One dispatch of a probe cell: its pipeline, grid, threads and buffers.
struct Dispatch {
    pso: Pipeline,
    grid: (usize, usize),
    threads: (usize, usize),
    binds: Binds,
}

/// A probe cell: its tag and its dispatches, with no barrier between them.
struct Cell {
    tag: String,
    dispatches: Vec<Dispatch>,
}

/// µs a dispatch (min / median / max over `ROUNDS` alternating rounds), barrier-separated and DRAM-cold, on a
/// chip of `EMULATED` cores emulated by width at buckets 4 and 8: the wide matvec (w) and the small-M twin (t)
/// against Splash's staged tile at the emulated chip's split with its spread (s on MLX planes, sp on Splash's);
/// for the shared expert's pair, the gate and up matvecs together (w2, t2) against the one-pass gate/up (gu, gup).
/// NAX only.
#[test]
#[ignore = "measurement probe, not an assertion: run with --ignored --nocapture"]
fn staged_tiles_on_a_chip_emulated_by_width() {
    use scratchy_target_metal::quantized::{
        DequantDtype, SMALL_M_SIMDGROUPS, SMALL_M_TILE_COLS, ScaleDtype, SmallMTile,
        small_m_kernel_static_name,
    };
    use scratchy_target_metal::tape::ids::{KDimI32, MDimI32, NDimI32};
    use scratchy_target_metal::tape::kernel_constants::{
        AffineCodes, AffineQmmTConstants, AffineQmvWideConstants,
    };
    use scratchy_target_metal::tape::quantized::{
        pick_qmv_kernel_wide, qmv_dispatch_shape, qmv_kernel_static_name,
    };
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    if !is_nax_capable(di.profile.generation) {
        eprintln!("skipping: no NAX matrix unit");
        return;
    }
    let device = di.device.clone();
    let cores = scratchy_target_metal::device::gpu_cores(&device)
        .expect("the device's GPU cores")
        .get() as usize;
    let width = EMULATED / cores;
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("pipeline cache");
    let codes = AffineCodes::of(Some(&di.profile), 4);
    let wide = |n: usize, k: usize, bucket: u32| {
        let kernel = pick_qmv_kernel_wide(n as u32, k as u32, 4, bucket, true);
        let name = qmv_kernel_static_name(
            kernel,
            DequantDtype::Bf16,
            ScaleDtype::Bf16,
            4,
            GROUP as u32,
        );
        let constants: Vec<ConstantValue> = AffineQmvWideConstants {
            k: KDimI32(k as i32),
            n: NDimI32(n as i32),
            m: MDimI32(bucket as i32),
            codes,
        }
        .into();
        let pso =
            baked_build(&cache, &PipelineKey::new("quantized_qmv", name, constants)).expect(name);
        let ((gx, gy, _), (tx, ty, _)) = qmv_dispatch_shape(kernel, bucket, n as u32, 1);
        (pso, (gx as usize, gy as usize), (tx as usize, ty as usize))
    };
    let twin = |n: usize, k: usize, bucket: u32| {
        let tile = SmallMTile::for_bucket(bucket).expect("a small-M bucket");
        let name =
            small_m_kernel_static_name(DequantDtype::Bf16, ScaleDtype::Bf16, GROUP as u32, tile);
        let constants: Vec<ConstantValue> = AffineQmmTConstants {
            k: KDimI32(k as i32),
            n: NDimI32(n as i32),
            m: MDimI32(bucket as i32),
            codes,
        }
        .into();
        let pso = baked_build(
            &cache,
            &PipelineKey::new("quantized_qmm_nax", name, constants),
        )
        .expect(name);
        let grid = (
            n / SMALL_M_TILE_COLS as usize,
            bucket.div_ceil(tile.rows()) as usize,
        );
        (pso, grid, (32 * SMALL_M_SIMDGROUPS as usize, 1))
    };
    println!(
        "µs a dispatch, min/median/max of {ROUNDS} rounds, a {EMULATED}-core chip emulated by width \
         on {cores} cores (columns / {width})"
    );
    let mut rng = Rng(7);
    let shapes = (SHAPES.iter().map(|&s| (s, false))).chain(PAIRS.iter().map(|&s| (s, true)));
    for ((name, full_n, k), pair) in shapes {
        // A projection narrower than a staged tile on the emulated chip runs at its own width: its few
        // threadgroups' latency is what every chip sees.
        let n = if full_n / width >= TILE {
            full_n / width
        } else {
            full_n
        };
        let tensors = if pair { 2 } else { 1 };
        let bytes = tensors * n * k / 2;
        let copies = ROTATED_BYTES / bytes + 1;
        // Per copy: each tensor's MLX and plane buffers.
        let weights: Vec<Vec<([Buffer; 3], [Buffer; 3])>> = (0..copies)
            .map(|_| {
                (0..tensors)
                    .map(|_| {
                        let w = Weights::random(&mut rng, n, k);
                        let planes = n.is_multiple_of(TILE);
                        (w.bind(&device, n, k, false), w.bind(&device, n, k, planes))
                    })
                    .collect()
            })
            .collect();
        let x = shared(&device, &bf16_rows(&mut rng, 16 * k));
        let outs = [
            shared(&device, &vec![0u16; 16 * n]),
            shared(&device, &vec![0u16; 16 * n]),
        ];
        let partials = shared(&device, &vec![0f32; 2 * 8 * 16 * n]);
        let counters = shared(&device, &vec![0u32; n / PAIR_TILE]);
        let mut cells: Vec<Cell> = Vec::new();
        for bucket in [4u32, 8] {
            let b = bucket as usize;
            let ours =
                |tag: &str, (pso, grid, threads): (Pipeline, (usize, usize), (usize, usize))| {
                    let dispatches = (0..tensors)
                        .map(|t| Dispatch {
                            pso: pso.clone(),
                            grid,
                            threads,
                            binds: Binds::Mlx(t),
                        })
                        .collect();
                    Cell {
                        tag: format!("{bucket}{tag}{}", if pair { "2" } else { "" }),
                        dispatches,
                    }
                };
            cells.push(ours("w", wide(n, k, bucket)));
            if n.is_multiple_of(SMALL_M_TILE_COLS as usize) {
                cells.push(ours("t", twin(n, k, bucket)));
            }
            for planes in [false, true] {
                if n % TILE != 0 {
                    break;
                }
                let p = if planes { "p" } else { "" };
                let cell = if pair {
                    // The pair's split, as one projection of its 2n columns (`oneGateUpPassDecode`).
                    let splits = decode_splits(2 * full_n, k, EMULATED);
                    let pso = staged(&cache, "gate_up", (k, n), b, planes, true);
                    let binds = if planes {
                        Binds::PairPlanes
                    } else {
                        Binds::PairMlx
                    };
                    Cell {
                        tag: format!("{bucket}gu{p}/{splits}"),
                        dispatches: vec![Dispatch {
                            pso,
                            grid: (n / PAIR_TILE, splits),
                            threads: (32, 1),
                            binds,
                        }],
                    }
                } else {
                    let splits = decode_splits(full_n, k, EMULATED);
                    let pso = staged(&cache, "decode", (k, n), b, planes, true);
                    let binds = if planes {
                        Binds::Planes(0)
                    } else {
                        Binds::Mlx(0)
                    };
                    Cell {
                        tag: format!("{bucket}s{p}/{splits}"),
                        dispatches: vec![Dispatch {
                            pso,
                            grid: (n / TILE, splits),
                            threads: (64, 1),
                            binds,
                        }],
                    }
                };
                cells.push(cell);
            }
        }
        let iters = (BATCH_BYTES / bytes).clamp(*ITERS.start(), *ITERS.end());
        let time = |cell: &Cell| {
            let mut batch = Mtl4DispatchBatch::begin(&device).expect("mtl4 batch");
            for i in 0..iters {
                let copy = &weights[i % copies];
                for Dispatch {
                    pso,
                    grid: (gx, gy),
                    threads: (tx, ty),
                    binds,
                } in &cell.dispatches
                {
                    let (grid, threads) = (size(*gx, *gy), size(*tx, *ty));
                    match *binds {
                        Binds::Mlx(t) | Binds::Planes(t) => {
                            let [w, s, b] = match *binds {
                                Binds::Planes(_) => &copy[t].1,
                                _ => &copy[t].0,
                            };
                            let binds = [
                                (w, 0),
                                (s, 1),
                                (b, 2),
                                (&x, 3),
                                (&outs[t], 4),
                                (&partials, 5),
                                (&counters, 6),
                            ];
                            batch.encode(pso, &binds, &[], &[], &[], grid, threads);
                        }
                        Binds::PairMlx | Binds::PairPlanes => {
                            let ([gw, gs, gb], [uw, us, ub]) = match *binds {
                                Binds::PairPlanes => (&copy[0].1, &copy[1].1),
                                _ => (&copy[0].0, &copy[1].0),
                            };
                            let binds = [
                                (gw, 0),
                                (gs, 1),
                                (gb, 2),
                                (uw, 3),
                                (us, 4),
                                (ub, 5),
                                (&x, 6),
                                (&outs[0], 7),
                                (&partials, 8),
                                (&counters, 9),
                            ];
                            batch.encode(pso, &binds, &[], &[], &[], grid, threads);
                        }
                    }
                }
                batch.barrier();
            }
            let t0 = std::time::Instant::now();
            batch.commit(true);
            t0.elapsed().as_secs_f64() * 1e6 / iters as f64
        };
        let mut samples = vec![Vec::new(); cells.len()];
        for round in 0..=ROUNDS {
            for (s, cell) in samples.iter_mut().zip(&cells) {
                let us = time(cell);
                if round > 0 {
                    s.push(us);
                }
            }
        }
        // Where Splash's 40-core plan runs the register tile (`apple10Registers`, LinearGguf.cpp:144: two
        // threadgroups a 64-column tile and lane, under three a core), the staged rows are B2's reference.
        let columns = if pair { 2 * full_n } else { full_n };
        let registers = 2 * (columns / TILE) < 3 * EMULATED;
        let plan = if registers {
            "register tile (B2)"
        } else {
            "staged"
        };
        println!(
            "{name} ({n} x {k}, {:.2} MB; the {EMULATED}-core plan: {plan}):",
            bytes as f64 / 1e6
        );
        for (cell, s) in cells.iter().zip(&mut samples) {
            s.sort_by(f64::total_cmp);
            println!(
                "    {:<10} {:8.1} {:8.1} {:8.1}",
                cell.tag,
                s[0],
                s[s.len() / 2],
                s[s.len() - 1]
            );
        }
    }
}

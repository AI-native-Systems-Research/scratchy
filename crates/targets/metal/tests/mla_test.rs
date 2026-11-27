// SPDX-License-Identifier: Apache-2.0
//! Golden tests for multi-head latent attention's kernels and the two core kernels it lowers
//! through — the column window (`slice_cols.metal`) and the per-head matmul
//! (`head_matmul.metal`) — each against reference math inlined here: the query rotation, the
//! cache write of each token's latent row into the paged pool's K and V halves, and the absorbed
//! attention over those rows.
//!
//! Each kernel is baked with its constants and dispatched standalone through
//! `SpecializedPipelineCache` (the path production takes through `pipeline_for_command`), at the
//! bf16 instantiation the models run.

#![cfg(target_os = "macos")]

mod common;

use common::Dtype;
use objc2_metal::{MTLBuffer, MTLSize};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use scratchy_target_metal::tape::kernel_constants::MlaTileUnit;
use scratchy_target_metal::tape::lowered::MetalDtype;
use scratchy_target_metal::tape::lowering::{mla_attend_tiles, mla_attend_tiling};
use scratchy_target_metal::targets::is_nax_capable;

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;

const BF16: Dtype = Dtype::Bf16;

/// A deterministic fill in `[-0.5, 0.5)`, its values bf16 as the kernels read them.
fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            BF16.round((x as f32 / u32::MAX as f32) - 0.5)
        })
        .collect()
}

fn bf16_buf(device: &Device, xs: &[f32]) -> Buffer {
    common::shared_bytes(device, &BF16.bytes(xs))
}

fn size(width: usize, height: usize, depth: usize) -> MTLSize {
    MTLSize {
        width,
        height,
        depth,
    }
}

fn setup() -> Option<(Device, SpecializedPipelineCache)> {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return None;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("pipeline cache");
    Some((device, cache))
}

fn uint(slot: u16, v: usize) -> ConstantValue {
    ConstantValue::uint(slot, v as u32)
}

fn assert_close(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= BF16.tol(*w),
            "{what}[{i}]: metal {g}, reference {w}"
        );
    }
}

// ── The column window ──────────────────────────────────────────────────────────────────────

/// Each row's `[offset, offset + cols)` window, bit for bit: the latent and the rope key of a
/// compressed projection row.
#[test]
fn slice_cols_takes_each_rows_window() {
    let Some((device, cache)) = setup() else {
        return;
    };
    let (rows, in_cols) = (5usize, 72usize);
    let x = fill(rows * in_cols, 1);
    for (offset, cols) in [(0usize, 64usize), (64, 8), (13, 30)] {
        let key = PipelineKey::new(
            "slice_cols",
            "slice_cols_bf16",
            vec![
                uint(0, in_cols),
                uint(1, offset),
                uint(2, cols),
                uint(3, rows * cols),
            ],
        );
        let pso = baked_build(&cache, &key).expect("slice_cols pipeline");
        let (input, out) = (
            bf16_buf(&device, &x),
            common::shared_zeroed(&device, rows * cols * 2),
        );
        let n = rows * cols;
        if !common::dispatch_threadgroups(
            &device,
            &pso,
            &[&input, &out],
            size(n.div_ceil(64), 1, 1),
            size(64, 1, 1),
        ) {
            return;
        }
        let want: Vec<f32> = (0..rows)
            .flat_map(|r| x[r * in_cols + offset..r * in_cols + offset + cols].to_vec())
            .collect();
        assert_eq!(BF16.decode(&out, n), want, "window ({offset}, {cols})");
    }
}

// ── The per-head matmul ────────────────────────────────────────────────────────────────────

/// One per-head matmul case: `heads` blocks of `n_head` rows of `k`, each over the first `k` of
/// its head's `stride` input lanes.
struct HeadCase {
    rows: usize,
    heads: usize,
    k: usize,
    stride: usize,
    n_head: usize,
}

const SIMDGROUPS: usize = 8;
const ROWS_PER_SG: usize = 4;

/// The reference: `y[t, h·N + j] = Σ_{i<k} x[t, h·stride + i] · w(h·N + j, i)`.
fn head_reference(c: &HeadCase, x: &[f32], w: &dyn Fn(usize, usize) -> f32) -> Vec<f32> {
    let n = c.heads * c.n_head;
    let mut y = vec![0.0; c.rows * n];
    for t in 0..c.rows {
        for row in 0..n {
            let head = row / c.n_head;
            let xh = &x[t * c.heads * c.stride + head * c.stride..];
            y[t * n + row] = (0..c.k).map(|i| xh[i] * w(row, i)).sum();
        }
    }
    y
}

fn head_dispatch(
    device: &Device,
    cache: &SpecializedPipelineCache,
    c: &HeadCase,
    function: &'static str,
    affine: Option<(usize, usize)>,
    buffers: [&Buffer; 4],
) -> Option<Vec<f32>> {
    let mut constants = vec![
        uint(0, c.heads),
        uint(1, c.k),
        uint(2, c.n_head),
        uint(3, c.stride),
        uint(6, SIMDGROUPS),
        uint(7, ROWS_PER_SG),
    ];
    if let Some((group, bits)) = affine {
        constants.extend([uint(4, group), uint(5, bits)]);
    }
    let key = PipelineKey::new("head_matmul", function, constants);
    let pso = baked_build(cache, &key).expect("head_matmul pipeline");
    let n = c.heads * c.n_head;
    let y = common::shared_zeroed(device, c.rows * n * 2);
    let [w, scales, biases, x] = buffers;
    let ran = common::dispatch_threadgroups(
        device,
        &pso,
        &[w, scales, biases, x, &y],
        size(c.rows, n.div_ceil(SIMDGROUPS * ROWS_PER_SG), 1),
        size(32, SIMDGROUPS, 1),
    );
    ran.then(|| BF16.decode(&y, c.rows * n))
}

/// A per-head matmul's MLX-affine weight: every head's block of rows, `bits`-wide codes packed
/// eight a pack, `bits` bytes a pack, little-endian, one bf16 scale and bias a group.
struct AffineBlocks {
    codes: Vec<u8>,
    code: Vec<u32>,
    scales: Vec<f32>,
    biases: Vec<f32>,
    k: usize,
    group: usize,
}

impl AffineBlocks {
    fn new(c: &HeadCase, bits: usize, group: usize) -> Self {
        let n = c.heads * c.n_head;
        let groups = c.k / group;
        let row_bytes = 4 * (c.k * bits).div_ceil(32);
        let mut codes = vec![0u8; n * row_bytes];
        let mut code = vec![0u32; n * c.k];
        let raw = fill(n * c.k, 7 + bits as u32);
        for row in 0..n {
            for i in 0..c.k {
                let q = ((raw[row * c.k + i] + 0.5) * (1 << bits) as f32) as u32 % (1 << bits);
                code[row * c.k + i] = q;
                let bit = i * bits;
                let at = row * row_bytes + bit / 8;
                let v = (q as u64) << (bit % 8);
                for b in 0..2 {
                    let byte = (v >> (8 * b)) as u8;
                    if byte != 0 {
                        codes[at + b] |= byte;
                    }
                }
            }
        }
        Self {
            codes,
            code,
            scales: fill(n * groups, 11),
            biases: fill(n * groups, 13),
            k: c.k,
            group,
        }
    }

    /// Weight row `row`'s element `i`, dequantized.
    fn weight(&self, row: usize, i: usize) -> f32 {
        let g = row * (self.k / self.group) + i / self.group;
        self.scales[g] * self.code[row * self.k + i] as f32 + self.biases[g]
    }
}

/// MLX-affine rows at each code width MLX writes: eight codes a pack, `bits` bytes a pack,
/// little-endian (the 2-, 4- and 8-bit word paths and the byte path of 3, 5 and 6), one bf16
/// scale and bias a group.
#[test]
fn head_qmv_matches_reference_at_every_code_width() {
    let Some((device, cache)) = setup() else {
        return;
    };
    let cases = [
        // MLA's query absorption: each head's first `k` of its `k + rope` lanes.
        HeadCase {
            rows: 3,
            heads: 4,
            k: 64,
            stride: 80,
            n_head: 48,
        },
        // Its value decompression: every lane of each head's part.
        HeadCase {
            rows: 2,
            heads: 3,
            k: 128,
            stride: 128,
            n_head: 40,
        },
    ];
    for c in &cases {
        for (bits, group) in [
            (4usize, 64usize),
            (4, 32),
            (3, 64),
            (2, 64),
            (5, 32),
            (6, 64),
            (8, 32),
        ] {
            let blocks = AffineBlocks::new(c, bits, group);
            let x = fill(c.rows * c.heads * c.stride, 17);
            let w = |row: usize, i: usize| blocks.weight(row, i);
            let want = head_reference(c, &x, &w);
            let bufs = [
                &common::shared_bytes(&device, &blocks.codes),
                &bf16_buf(&device, &blocks.scales),
                &bf16_buf(&device, &blocks.biases),
                &bf16_buf(&device, &x),
            ];
            let Some(got) = head_dispatch(
                &device,
                &cache,
                c,
                "head_qmv_bf16_s_bf16",
                Some((group, bits)),
                bufs,
            ) else {
                return;
            };
            assert_close(
                &format!("head_qmv bits={bits} group={group} k={}", c.k),
                &got,
                &want,
            );
        }
    }
}

/// The matmul a per-head matmul runs from its matvec/matmul boundary on: `qmm_t`'s standard
/// tile a head (QMM_HEADS), at each width it is instantiated for, over rows of whole and partial
/// 32-row tiles and heads of whole and partial 32-column ones.
#[test]
fn head_qmm_matches_reference() {
    let Some((device, cache)) = setup() else {
        return;
    };
    let cases = [
        // GLM-4.7-Flash's query absorption: each head's first 192 of its 256 lanes, into 512.
        HeadCase {
            rows: 37,
            heads: 20,
            k: 192,
            stride: 256,
            n_head: 512,
        },
        // A value decompression whose heads are partial column tiles.
        HeadCase {
            rows: 70,
            heads: 3,
            k: 128,
            stride: 128,
            n_head: 40,
        },
    ];
    // The standard tile everywhere; the matrix unit's (64-lane K steps, groups of 64 or more)
    // where the chip has one.
    let nax = detect_device().is_some_and(|di| is_nax_capable(di.profile.generation));
    // (library, symbol stem, rows and columns a tile, threads, the matrix unit's)
    let tiles = [
        ("quantized_qmm", "qmm_t", 32, size(32, 2, 2), false),
        ("quantized_qmm_nax", "qmm_t_nax", 64, size(32, 2, 2), true),
        (
            "quantized_qmm_nax",
            "qmm_t_nax_producer",
            64,
            size(32, 5, 1),
            true,
        ),
    ];
    let tiles = tiles.into_iter().filter(|t| !t.4 || nax);
    let runs = tiles.flat_map(|t| {
        [
            (4usize, 64usize),
            (4, 32),
            (4, 128),
            (3, 64),
            (2, 64),
            (8, 64),
        ]
        .map(|w| (t, w))
    });
    for ((library, stem, tile, threads, matrix), (bits, group)) in runs {
        for c in &cases {
            if c.k % group != 0 || (matrix && (c.k % 64 != 0 || group < 64)) {
                continue;
            }
            let blocks = AffineBlocks::new(c, bits, group);
            let x = fill(c.rows * c.heads * c.stride, 17);
            // The tile dequantizes each weight into threadgroup memory in the activation dtype.
            let want = head_reference(c, &x, &|row, i| BF16.round(blocks.weight(row, i)));
            let aligned = c.n_head % tile == 0;
            let symbol =
                format!("affine_{stem}_bf16_s_bf16_gs_{group}_b_{bits}_alN_{aligned}_batch_0");
            let constants = vec![
                ConstantValue::int(0, c.k as i32),
                ConstantValue::int(1, c.n_head as i32),
                ConstantValue::int(2, c.rows as i32),
                ConstantValue::int(6, c.heads as i32),
                ConstantValue::int(8, c.stride as i32),
            ];
            let symbol: &'static str = Box::leak(symbol.into_boxed_str());
            let key = PipelineKey::new(library, symbol, constants);
            let pso = baked_build(&cache, &key).expect("per-head qmm_t pipeline");
            let n = c.heads * c.n_head;
            let y = common::shared_zeroed(&device, c.rows * n * 2);
            if !common::dispatch_threadgroups(
                &device,
                &pso,
                &[
                    &common::shared_bytes(&device, &blocks.codes),
                    &bf16_buf(&device, &blocks.scales),
                    &bf16_buf(&device, &blocks.biases),
                    &bf16_buf(&device, &x),
                    &y,
                ],
                size(c.n_head.div_ceil(tile), c.rows.div_ceil(tile), c.heads),
                threads,
            ) {
                return;
            }
            assert_close(
                &format!(
                    "head {stem} bits={bits} group={group} k={} n_head={}",
                    c.k, c.n_head
                ),
                &BF16.decode(&y, c.rows * n),
                &want,
            );
        }
    }
}

/// Dense blocks in the activation dtype.
#[test]
fn head_mv_matches_reference() {
    let Some((device, cache)) = setup() else {
        return;
    };
    let c = HeadCase {
        rows: 3,
        heads: 4,
        k: 40,
        stride: 56,
        n_head: 24,
    };
    let n = c.heads * c.n_head;
    let w = fill(n * c.k, 19);
    let x = fill(c.rows * c.heads * c.stride, 23);
    let want = head_reference(&c, &x, &|row, i| w[row * c.k + i]);
    let unused = common::shared_zeroed(&device, 4);
    let bufs = [
        &bf16_buf(&device, &w),
        &unused,
        &unused,
        &bf16_buf(&device, &x),
    ];
    let Some(got) = head_dispatch(&device, &cache, &c, "head_mv_bf16", None, bufs) else {
        return;
    };
    assert_close("head_mv", &got, &want);
}

// ── The MLA kernels ────────────────────────────────────────────────────────────────────────

/// A latent attention's geometry, as the kernels' constants (slots 0 to 3) take it.
#[derive(Clone, Copy)]
struct Geom {
    heads: usize,
    latent: usize,
    nope: usize,
    rope: usize,
}

impl Geom {
    fn q_head(self) -> usize {
        self.nope + self.rope
    }
    fn row(self) -> usize {
        self.latent + self.rope
    }
    fn half(self) -> usize {
        self.row() / 2
    }
    fn constants(self) -> Vec<ConstantValue> {
        vec![
            uint(0, self.heads),
            uint(1, self.latent),
            uint(2, self.nope),
            uint(3, self.rope),
        ]
    }
}

/// A paged pool of one layer's latent rows: its K and V halves, each one buffer of every block
/// (`[block][token][half]`), addressed through a chunk table of `blocks_per_chunk` blocks.
struct Pool {
    k: Buffer,
    v: Buffer,
    k_table: Buffer,
    v_table: Buffer,
    blocks: usize,
    block_size: usize,
    blocks_per_chunk: usize,
}

impl Pool {
    fn new(
        device: &Device,
        g: Geom,
        blocks: usize,
        block_size: usize,
        bpc: usize,
        fill_v: f32,
    ) -> Self {
        let elems = blocks * block_size * g.half();
        let marker = vec![fill_v; elems];
        let (k, v) = (bf16_buf(device, &marker), bf16_buf(device, &marker));
        let block_bytes = (block_size * g.half() * 2) as u64;
        let table = |store: &Buffer| {
            let chunks: Vec<u64> = (0..blocks.div_ceil(bpc))
                .map(|ch| store.gpuAddress() + ch as u64 * bpc as u64 * block_bytes)
                .collect();
            common::shared_slice(device, &chunks)
        };
        let (k_table, v_table) = (table(&k), table(&v));
        Self {
            k,
            v,
            k_table,
            v_table,
            blocks,
            block_size,
            blocks_per_chunk: bpc,
        }
    }

    /// Lane `e` of the row at physical slot `slot`, its store and index there.
    fn at(&self, g: Geom, slot: usize, e: usize) -> (bool, usize) {
        let base = (slot / self.block_size) * self.block_size * g.half()
            + (slot % self.block_size) * g.half();
        match e < g.half() {
            true => (false, base + e),
            false => (true, base + e - g.half()),
        }
    }

    fn read(&self, g: Geom) -> (Vec<f32>, Vec<f32>) {
        let n = self.blocks * self.block_size * g.half();
        (BF16.decode(&self.k, n), BF16.decode(&self.v, n))
    }

    fn cache_constants(&self, g: Geom, max_blocks: usize) -> Vec<ConstantValue> {
        vec![
            uint(4, self.block_size),
            uint(5, max_blocks),
            uint(6, self.blocks_per_chunk),
            uint(14, g.half()),
        ]
    }
}

/// The rotary table the kernels read: per position, the `rope / 2` cosines then the sines of
/// frequency `i`'s angle `pos · theta^(-2i / rope)`.
fn rotary(positions: usize, rope: usize) -> Vec<f32> {
    let half = rope / 2;
    (0..positions)
        .flat_map(|pos| {
            let ang: Vec<f64> = (0..half)
                .map(|i| pos as f64 * 1e6f64.powf(-2.0 * i as f64 / rope as f64))
                .collect();
            let cos = ang.iter().map(|a| BF16.round(a.cos() as f32));
            let sin = ang.iter().map(|a| BF16.round(a.sin() as f32));
            cos.chain(sin).collect::<Vec<_>>()
        })
        .collect()
}

/// How a rope part's lanes pair, as slot 15 (`MLA_PAIRING`) states it.
#[derive(Clone, Copy, Debug)]
enum Pairing {
    /// Lanes `(2i, 2i + 1)`: slot value 0.
    Interleaved,
    /// Lanes `(i, i + rope / 2)`, the NeoX split: slot value 1.
    Halves,
}

impl Pairing {
    const ALL: [Self; 2] = [Self::Interleaved, Self::Halves];

    fn constant(self) -> ConstantValue {
        uint(
            15,
            match self {
                Self::Interleaved => 0,
                Self::Halves => 1,
            },
        )
    }
}

/// `x`'s rope part rotated at `pos`, its lanes paired as `pairing` states.
fn rotate(x: &[f32], table: &[f32], rope: usize, pos: usize, pairing: Pairing) -> Vec<f32> {
    let (half, row) = (rope / 2, &table[pos * rope..]);
    (0..rope)
        .map(|d| {
            let (f, first, i0, i1) = match pairing {
                Pairing::Interleaved => (d / 2, d % 2 == 0, d & !1, d | 1),
                Pairing::Halves => (d % half, d < half, d % half, d % half + half),
            };
            let (c, s, x0, x1) = (row[f], row[half + f], x[i0], x[i1]);
            match first {
                true => x0 * c - x1 * s,
                false => x1 * c + x0 * s,
            }
        })
        .collect()
}

/// Each query head's rope part rotated at its row's position; its no-rope part passes through.
#[test]
fn mla_rope_q_rotates_each_heads_rope_part() {
    let Some((device, cache)) = setup() else {
        return;
    };
    let g = Geom {
        heads: 3,
        latent: 64,
        nope: 24,
        rope: 16,
    };
    let positions: Vec<u32> = vec![0, 5, 17, 300];
    let rows = positions.len();
    let q = fill(rows * g.heads * g.q_head(), 29);
    let table = rotary(512, g.rope);
    for pairing in Pairing::ALL {
        let mut constants = g.constants();
        constants.push(uint(13, 32));
        constants.push(pairing.constant());
        let key = PipelineKey::new("mla", "mla_rope_q_bf16", constants);
        let pso = baked_build(&cache, &key).expect("mla_rope_q pipeline");
        let out = common::shared_zeroed(&device, q.len() * 2);
        if !common::dispatch_threadgroups(
            &device,
            &pso,
            &[
                &bf16_buf(&device, &q),
                &out,
                &common::shared_slice(&device, &positions),
                &bf16_buf(&device, &table),
            ],
            size(rows, g.heads, 1),
            size(32, 1, 1),
        ) {
            return;
        }
        let mut want = Vec::new();
        for (t, &pos) in positions.iter().enumerate() {
            for h in 0..g.heads {
                let head = &q[(t * g.heads + h) * g.q_head()..][..g.q_head()];
                want.extend_from_slice(&head[..g.nope]);
                want.extend(rotate(
                    &head[g.nope..],
                    &table,
                    g.rope,
                    pos as usize,
                    pairing,
                ));
            }
        }
        assert_close(
            &format!("mla_rope_q {pairing:?}"),
            &BF16.decode(&out, q.len()),
            &want,
        );
    }
}

/// Each token's `[latent | rotated key]` lands at its slot in the pool's K and V halves (slots
/// across blocks and chunks), its rotated key is the step's output, and a padding row writes
/// nothing.
#[test]
fn mla_append_writes_each_tokens_latent_row() {
    let Some((device, cache)) = setup() else {
        return;
    };
    for (g, pairing) in [
        Geom {
            heads: 2,
            latent: 64,
            nope: 16,
            rope: 16,
        },
        // GLM-4.7-Flash's row: a 512-wide latent and a 64-wide rope key.
        Geom {
            heads: 20,
            latent: 512,
            nope: 192,
            rope: 64,
        },
    ]
    .into_iter()
    .flat_map(|g| Pairing::ALL.map(|p| (g, p)))
    {
        let (block_size, bpc, blocks) = (4usize, 2usize, 6usize);
        let marker = 7.0;
        let pool = Pool::new(&device, g, blocks, block_size, bpc, marker);
        let positions: Vec<u32> = vec![3, 9, 4, 1];
        let slots: Vec<u32> = vec![2, 13, 0xFFFF_FFFF, 21];
        let rows = positions.len();
        let latent = fill(rows * g.latent, 31);
        let k_pe = fill(rows * g.rope, 37);
        let table = rotary(64, g.rope);
        // Fewer threads than lanes: each thread takes several.
        let threads = 256usize;
        let mut constants = g.constants();
        constants.extend(pool.cache_constants(g, 8));
        constants.push(uint(13, threads));
        constants.push(pairing.constant());
        let key = PipelineKey::new("mla", "mla_append_bf16", constants);
        let pso = baked_build(&cache, &key).expect("mla_append pipeline");
        let out = common::shared_zeroed(&device, rows * g.rope * 2);
        if !common::dispatch_threadgroups(
            &device,
            &pso,
            &[
                &bf16_buf(&device, &latent),
                &bf16_buf(&device, &k_pe),
                &out,
                &common::shared_slice(&device, &positions),
                &bf16_buf(&device, &table),
                &common::shared_slice(&device, &slots),
                &pool.k_table,
                &pool.v_table,
                &pool.k,
                &pool.v,
            ],
            size(rows, 1, 1),
            size(threads, 1, 1),
        ) {
            return;
        }
        let rotated: Vec<Vec<f32>> = (0..rows)
            .map(|t| {
                let pos = positions[t] as usize;
                rotate(&k_pe[t * g.rope..], &table, g.rope, pos, pairing)
            })
            .collect();
        let want_out: Vec<f32> = rotated.iter().flatten().map(|&x| BF16.round(x)).collect();
        assert_close(
            "mla_append out",
            &BF16.decode(&out, rows * g.rope),
            &want_out,
        );
        let n = blocks * block_size * g.half();
        let (mut want_k, mut want_v) = (vec![marker; n], vec![marker; n]);
        for t in 0..rows {
            if slots[t] == 0xFFFF_FFFF {
                continue;
            }
            for e in 0..g.row() {
                let val = match e < g.latent {
                    true => latent[t * g.latent + e],
                    false => BF16.round(rotated[t][e - g.latent]),
                };
                let (in_v, at) = pool.at(g, slots[t] as usize, e);
                match in_v {
                    true => want_v[at] = val,
                    false => want_k[at] = val,
                }
            }
        }
        let (k, v) = pool.read(g);
        assert_close("mla_append K half", &k, &want_k);
        assert_close("mla_append V half", &v, &want_v);
    }
}

/// Every row of each sequence over its sequence's cached latent rows up to its own position:
/// two sequences of several rows, their blocks scattered across chunks, heads served several to a
/// threadgroup over fewer simdgroups than it has rows, and more.
#[test]
fn mla_attend_matches_reference() {
    let Some((device, cache)) = setup() else {
        return;
    };
    let small = Geom {
        heads: 4,
        latent: 64,
        nope: 16,
        rope: 16,
    };
    // GLM-4.7-Flash's geometry.
    let glm = Geom {
        heads: 20,
        latent: 512,
        nope: 192,
        rope: 64,
    };
    // (geometry, heads a threadgroup serves, its simdgroups, the partitions of each sequence's
    // rows: one writes the output, more write partials the merge combines). First the tilings
    // the lowering ships, a decode-shaped (1-row) and a prefill-shaped (64-row) bucket's of each
    // geometry, then hand-picked ones it does not pick.
    let shipped = [small, glm].into_iter().flat_map(|g| {
        [1u32, 64].map(|bucket_m| {
            let t = mla_attend_tiling(g.heads as u32, g.latent as u32, bucket_m)
                .expect("a tiling for the geometry");
            let n = |x: u32| x as usize;
            (
                g,
                n(t.heads_per_tg.get()),
                n(t.simdgroups.get()),
                n(t.parts.get()),
            )
        })
    });
    let shipped: Vec<_> = shipped.collect();
    // GLM-4.7-Flash's decode tiling, the one its decode runs, stated as well as derived.
    assert!(
        shipped
            .iter()
            .any(|&(g, h, s, p)| g.latent == glm.latent && (h, s, p) == (1, 8, 8))
    );
    for (g, heads_per_tg, simdgroups, parts) in shipped.into_iter().chain([
        (small, 2usize, 32usize, 1usize),
        (small, 4, 2, 3),
        (glm, 4, 8, 16),
        (glm, 2, 4, 1),
    ]) {
        // Sequence 0: 9 cached rows, its last 3 this step's; sequence 1: 6, its last 2.
        let case = AttendCase::new(
            &device,
            g,
            (4, 2, 10, 6),
            &[9, 6],
            &[0, 3, 5],
            &[vec![7, 2, 9], vec![4, 0]],
        );
        let (rows, scale, max_blocks) = (case.rows, case.scale, case.max_blocks);
        let mut constants = g.constants();
        constants.extend(case.pool.cache_constants(g, max_blocks));
        constants.extend([
            ConstantValue::float(7, scale),
            uint(8, heads_per_tg),
            uint(9, simdgroups),
            uint(10, parts),
            uint(11, g.latent + 2),
            uint(12, 256),
        ]);
        let key = PipelineKey::new("mla", "mla_attend_bf16", constants.clone());
        let pso = baked_build(&cache, &key).expect("mla_attend pipeline");
        let out = common::shared_zeroed(&device, rows * g.heads * g.latent * 2);
        let partials = common::shared_zeroed(&device, rows * g.heads * parts * (g.latent + 2) * 4);
        if !common::dispatch_threadgroups(
            &device,
            &pso,
            &[
                &out,
                &case.q_latent,
                &case.q,
                &case.seq_used,
                &case.block_table,
                &case.k_table,
                &case.v_table,
                &case.cu,
                &partials,
                &case.k,
                &case.v,
            ],
            size(rows, g.heads / heads_per_tg, parts),
            size(32 * simdgroups, 1, 1),
        ) {
            return;
        }
        if parts > 1 {
            let key = PipelineKey::new("mla", "mla_attend_merge_bf16", constants);
            let merge = baked_build(&cache, &key).expect("mla_attend_merge pipeline");
            if !common::dispatch_threadgroups(
                &device,
                &merge,
                &[&out, &partials],
                size(rows, g.heads, 1),
                size(256, 1, 1),
            ) {
                return;
            }
        }
        assert_close(
            &format!(
                "mla_attend heads={} per_tg={heads_per_tg} simdgroups={simdgroups} parts={parts}",
                g.heads
            ),
            &BF16.decode(&out, case.want.len()),
            &case.want,
        );
    }
}

/// An absorbed attention's inputs, each sequence's cached rows written through its block table
/// into a paged pool, and the reference output: every row of each sequence over its sequence's
/// cached rows up to its own position.
struct AttendCase {
    pool: Pool,
    k: Buffer,
    v: Buffer,
    k_table: Buffer,
    v_table: Buffer,
    seq_used: Buffer,
    cu: Buffer,
    block_table: Buffer,
    q_latent: Buffer,
    q: Buffer,
    rows: usize,
    max_blocks: usize,
    scale: f32,
    want: Vec<f32>,
}

impl AttendCase {
    /// `pages` = (rows a block, blocks a chunk, blocks in the pool, the block table's stride);
    /// sequence `s` holds `seq_used[s]` cached rows in blocks `tables[s]`, its last
    /// `cu[s + 1] - cu[s]` this step's.
    fn new(
        device: &Device,
        g: Geom,
        pages: (usize, usize, usize, usize),
        seq_used: &[u32],
        cu: &[u32],
        tables: &[Vec<u32>],
    ) -> Self {
        let (block_size, bpc, blocks, max_blocks) = pages;
        let seqs = seq_used.len();
        let pool = Pool::new(device, g, blocks, block_size, bpc, 0.0);
        let mut block_table = vec![0u32; seqs * max_blocks];
        for (s, t) in tables.iter().enumerate() {
            block_table[s * max_blocks..s * max_blocks + t.len()].copy_from_slice(t);
        }
        let cached: Vec<Vec<f32>> = (0..seqs)
            .map(|s| fill(seq_used[s] as usize * g.row(), 41 + s as u32))
            .collect();
        let (mut k, mut v) = pool.read(g);
        for s in 0..seqs {
            for pos in 0..seq_used[s] as usize {
                let slot = tables[s][pos / block_size] as usize * block_size + pos % block_size;
                for e in 0..g.row() {
                    let (in_v, at) = pool.at(g, slot, e);
                    let val = cached[s][pos * g.row() + e];
                    match in_v {
                        true => v[at] = val,
                        false => k[at] = val,
                    }
                }
            }
        }
        let (pool_k, pool_v) = (bf16_buf(device, &k), bf16_buf(device, &v));
        let block_bytes = (block_size * g.half() * 2) as u64;
        let table = |store: &Buffer| {
            let chunks: Vec<u64> = (0..blocks.div_ceil(bpc))
                .map(|ch| store.gpuAddress() + ch as u64 * bpc as u64 * block_bytes)
                .collect();
            common::shared_slice(device, &chunks)
        };
        let (k_table, v_table) = (table(&pool_k), table(&pool_v));
        let rows = *cu.last().unwrap() as usize;
        let q_latent = fill(rows * g.heads * g.latent, 43);
        let q = fill(rows * g.heads * g.q_head(), 47);
        let scale = 1.0 / (g.q_head() as f32).sqrt();
        let mut want = vec![0.0f32; rows * g.heads * g.latent];
        for row in 0..rows {
            let s = (1..=seqs).find(|&s| row < cu[s] as usize).unwrap() - 1;
            let kv_len = seq_used[s] as usize - (cu[s + 1] as usize - 1 - row);
            for h in 0..g.heads {
                let ql = &q_latent[(row * g.heads + h) * g.latent..][..g.latent];
                let qr = &q[(row * g.heads + h) * g.q_head() + g.nope..][..g.rope];
                let scores: Vec<f32> = (0..kv_len)
                    .map(|j| {
                        let r = &cached[s][j * g.row()..][..g.row()];
                        let lat: f32 = (0..g.latent).map(|e| ql[e] * r[e]).sum();
                        let pe: f32 = (0..g.rope).map(|d| qr[d] * r[g.latent + d]).sum();
                        scale * (lat + pe)
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let p: Vec<f32> = scores.iter().map(|x| (x - m).exp()).collect();
                let sum: f32 = p.iter().sum();
                for e in 0..g.latent {
                    want[(row * g.heads + h) * g.latent + e] = (0..kv_len)
                        .map(|j| p[j] * cached[s][j * g.row() + e])
                        .sum::<f32>()
                        / sum;
                }
            }
        }
        Self {
            pool,
            k: pool_k,
            v: pool_v,
            k_table,
            v_table,
            seq_used: common::shared_slice(device, seq_used),
            cu: common::shared_slice(device, cu),
            block_table: common::shared_slice(device, &block_table),
            q_latent: bf16_buf(device, &q_latent),
            q: bf16_buf(device, &q),
            rows,
            max_blocks,
            scale,
            want,
        }
    }
}

/// The tiled attention a matmul-band bucket runs, at the tile the lowering ships for each
/// geometry and page: three sequences — a fresh prompt of several query tiles, a chunk after a
/// cached prefix whose last tile is partial, and one decode row — their blocks scattered across
/// chunks, every causal boundary inside a tile.
#[test]
fn mla_attend_tiles_matches_reference() {
    let Some((device, cache)) = setup() else {
        return;
    };
    let small = Geom {
        heads: 4,
        latent: 64,
        nope: 16,
        rope: 16,
    };
    let glm = Geom {
        heads: 20,
        latent: 512,
        nope: 192,
        rope: 64,
    };
    let nax = detect_device().is_some_and(|di| is_nax_capable(di.profile.generation));
    let units = [MlaTileUnit::Simdgroup, MlaTileUnit::Matrix];
    let units = units
        .into_iter()
        .filter(|u| nax || *u == MlaTileUnit::Simdgroup);
    let shapes = units.flat_map(|u| {
        [(small, 16usize), (small, 32), (glm, 16), (glm, 32)].map(|(g, page)| (u, g, page))
    });
    for (unit, g, block_size) in shapes {
        let half = (g.latent + g.rope) as u32 / 2;
        let tile = mla_attend_tiles(
            unit,
            g.latent as u32,
            g.rope as u32,
            block_size as u32,
            half,
        );
        let Some(tile) = tile else {
            // The matrix unit's 32-lane slices: the small row's 40-lane halves are none.
            assert!(unit == MlaTileUnit::Matrix && g.latent == small.latent);
            continue;
        };
        let (warp_lanes, keys) = (tile.warp_lanes.get() as usize, tile.keys.get() as usize);
        // Sequence 0: a fresh 45-row prompt; 1: 70 cached rows, the last 21 this step's; 2: one
        // decode row over 37.
        let seq_used = [45u32, 70, 37];
        let cu = [0u32, 45, 66, 67];
        let pages_of = |rows: u32| (rows as usize).div_ceil(block_size);
        let total_pages: usize = seq_used.iter().map(|&n| pages_of(n)).sum();
        // Each sequence's pages, scattered: a stride coprime to the pool's size, so no two pages
        // share a block.
        let blocks = total_pages + 3;
        let gcd = |mut a: usize, mut b: usize| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let stride = (5..).find(|&s| gcd(s, blocks) == 1).unwrap();
        let mut next = 0usize;
        let tables: Vec<Vec<u32>> = seq_used
            .iter()
            .map(|&n| {
                (0..pages_of(n))
                    .map(|_| {
                        next += 1;
                        ((next * stride) % blocks) as u32
                    })
                    .collect()
            })
            .collect();
        let max_blocks = tables.iter().map(Vec::len).max().unwrap() + 1;
        let case = AttendCase::new(
            &device,
            g,
            (block_size, 2, blocks, max_blocks),
            &seq_used,
            &cu,
            &tables,
        );
        let mut constants = g.constants();
        constants.extend(case.pool.cache_constants(g, max_blocks));
        constants.extend([
            ConstantValue::float(7, case.scale),
            uint(16, warp_lanes),
            uint(17, keys),
            uint(18, tile.blocks() as usize),
        ]);
        let (library, function) = tile.kernel(MetalDtype::Bf16);
        let key = PipelineKey::new(library, function, constants);
        let pso = baked_build(&cache, &key).expect("mla_attend_tiles pipeline");
        let out = common::shared_zeroed(&device, case.rows * g.heads * g.latent * 2);
        let most = (0..seq_used.len())
            .map(|s| (cu[s + 1] - cu[s]) as usize)
            .max()
            .unwrap();
        if !common::dispatch_threadgroups(
            &device,
            &pso,
            &[
                &out,
                &case.q_latent,
                &case.q,
                &case.seq_used,
                &case.block_table,
                &case.k_table,
                &case.v_table,
                &case.cu,
                &case.k,
                &case.v,
            ],
            size(
                (most * g.heads).div_ceil(tile.queries() as usize),
                1,
                seq_used.len(),
            ),
            size(tile.threads() as usize, 1, 1),
        ) {
            return;
        }
        assert_close(
            &format!(
                "mla_attend_tiles {unit:?} heads={} latent={} page={block_size}",
                g.heads, g.latent
            ),
            &BF16.decode(&out, case.want.len()),
            &case.want,
        );
    }
}

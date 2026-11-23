// SPDX-License-Identifier: Apache-2.0

//! The fp8 KV cache's card leg: ONE run separates the write from the read.
//!
//! ```bash
//! cargo --config patches/oco.toml run --release -p scratchy-target-spyre-card \
//!   --example kv_fp8_round_trip -- <work-dir>
//! ```
//!
//! Six bundles, each a production program — the splice's own constructors, the ones chat's bake
//! calls — lowered and placed as a one-kernel bundle and compiled by the bake's own dxp door:
//!
//! * `ref` — the device's own `qfp8ch` (the W8A8 activation quantizer's convert door) over a host
//!   `[rows, 128]` fp16 pool image: an fp8 pool image no write path of ours made;
//! * `write` — `reshape_and_cache_head_major` at an fp8 pool;
//! * `decode8` / `prefill8` — the paged attention at an fp8 pool;
//! * `decode16` / `prefill16` — the same at an fp16 pool, the not-X control.
//!
//! The checks, every one bit-exact:
//! * phase 1 — the write's pool bytes equal the reference pool's, byte for byte: the write, alone;
//! * phase 2c — the fp16-pool attention returns the pattern: the attention's own arithmetic;
//! * phase 2 — the fp8-pool attention over the REFERENCE pool returns the pattern: the read, alone;
//! * phase 3 — the fp8-pool attention over the WRITTEN pool returns the pattern: the round trip.
//!
//! The read is observed through the attention's own dot: the query is zero, so every score is 0,
//! and the mask leaves ONE slot `s` of each query row unmasked — so each query head `j`'s output is
//! `v[j / 4, s, :]`. A failing check prints the phase, the program, the slot or row, and the first
//! differing element; every phase runs, and the leg exits non-zero if any check failed.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use ktir_superdsc::collective::TpWorld;
use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::ktir_node::KtirNode;
use ktir_superdsc::place::{PlaceId, act_name};
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
use ktir_superdsc::reserved_tids::{PAGED_INT32_ZERO_TID, RESERVED_TID_FLOOR};
use ktir_superdsc::superdsc_opspec::{Df, Fp8, Fp16};
use scratchy_target_spyre::superdsc_bake::{ChunkExplore, DxpTool};
use scratchy_triton_splice::{
    HeadWidth, KvWriteGeometry, PagedAttnGeometry, PagedSequence, SchedulerBlockTokens,
    allocate_paged_cache, kv_write_program, paged_attn_program,
};

/// A one-card leg: a world of one.
type W1 = TpWorld<1>;

/// granite-3.1-8b's attention geometry: kv heads, query heads, head width.
const NKVH: u32 = 8;
const NQH: u32 = 32;
const HD: u32 = 128;
/// The scheduler's block, the page's slots.
const BLOCK: u32 = SchedulerBlockTokens::REFERENCE.get();
/// The pool: two pages; page 0 holds the pattern, page 1 stays zero.
const PAGES: u32 = 2;
/// The folded pool's token rows — `[PAGES · NKVH · BLOCK, HD]`.
const POOL_ROWS: u32 = PAGES * NKVH * BLOCK;
/// The write's token rows: one whole page (the kernel's `T >= 8` branch).
const T: u32 = BLOCK;
/// The decode slots read back — the first, a stick of index words' edges, a page's middle and end.
const SLOTS: [u32; 8] = [0, 1, 7, 8, 63, 64, 126, 127];
/// The prefill's query rows; row `p` reads slot `2p`.
const PREFILL_ROWS: u32 = 64;
/// granite-3.1-8b's `attention_multiplier`: the score scale chat's program is compiled at.
const SCALE: f32 = 0.0078125;
/// A masked slot's additive mask: `exp(−65504)` is 0 in fp16.
const MASKED: f32 = -65504.0;

/// ⭐ THE PATTERN — `±(1 + m/8)·2^e`, `m ∈ 0..8`, `e ∈ −4..=4`: an e4m3 NORMAL with a 3-bit
/// mantissa, so exact in e4m3, fp16 and DLFloat16, and inside the saturation. K is `which = 0`,
/// V `which = 1`.
fn pattern(which: u32, kv: u32, t: u32, d: u32) -> f32 {
    let sign = if (t + d + which).is_multiple_of(2) {
        1.0
    } else {
        -1.0
    };
    let m = ((t * 7 + d + which * 3) % 8) as f32;
    let e = ((kv + d + which) % 9) as i32 - 4;
    sign * (1.0 + m / 8.0) * 2f32.powi(e)
}

/// The value of an e4m3fn byte (bias 7, no infinity).
fn e4m3(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((b >> 3) & 0xF);
    let m = f32::from(b & 7);
    if e == 0 {
        sign * (m / 8.0) * 2f32.powi(-6)
    } else {
        sign * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

/// Every pattern value is its own e4m3 rounding — the card's `qfp8ch` encoder's byte, decoded.
fn check_pattern() -> Result<()> {
    for which in 0..2 {
        for kv in 0..NKVH {
            for t in 0..BLOCK {
                for d in 0..HD {
                    let v = pattern(which, kv, t, d);
                    let back = e4m3(ktir_core::codec::f32_to_e4m3(v));
                    if back != v {
                        bail!(
                            "the pattern is not e4m3-exact at which={which} kv={kv} t={t} d={d}: \
                             {v} rounds to {back}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

/// A pool image `[POOL_ROWS, HD]`, row `(page·NKVH + kv)·BLOCK + t`: page 0 the pattern, page 1 zero.
fn pool_image(which: u32) -> Vec<f32> {
    let mut x = vec![0f32; (POOL_ROWS * HD) as usize];
    for kv in 0..NKVH {
        for t in 0..BLOCK {
            for d in 0..HD {
                x[((kv * BLOCK + t) * HD + d) as usize] = pattern(which, kv, t, d);
            }
        }
    }
    x
}

fn f16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect()
}

fn u32_bytes(vals: &[u32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `bytes` written as `dir/name`; the path, as `bundle_run` takes it.
fn put(dir: &Path, name: &str, bytes: &[u8]) -> Result<String> {
    let p = dir.join(name);
    std::fs::write(&p, bytes).with_context(|| format!("write {}", p.display()))?;
    Ok(p.display().to_string())
}

/// A placement's size, by tid.
fn size_of(layout: &BundleLayout, tid: u32) -> Result<usize> {
    layout
        .placements
        .get(&tid)
        .map(|p| p.size as usize)
        .ok_or_else(|| anyhow!("t{tid} is not placed"))
}

/// `vals` as fp16, zero-padded to the placement's whole size (an fp16 bind is exactly its size).
fn f16_at(layout: &BundleLayout, tid: u32, vals: &[f32]) -> Result<Vec<u8>> {
    let mut b = f16_bytes(vals);
    b.resize(size_of(layout, tid)?, 0);
    Ok(b)
}

/// The host page `prepare` rounds a staged prefix up to (`superdsc_exec::HOST_PAGE`).
const HOST_PAGE: u64 = 4096;

/// The directory written (`write_dir` + `write_placements`, after the lowering, so the synthetics
/// are in the segment sizes) and compiled in place by the bake's own dxp door.
///
/// ⭐ EVERY SEGMENT TAKES ONE HOST PAGE OF PAD. `prepare` sends a host-owned slot's staged prefix
/// rounded UP to a page, clamped to the shadow only before the rounding, and the shadow is the
/// segment's 128-aligned size. A reserved constant staged at a segment's top (the `FP8_*448` sticks)
/// would then round past it. A staged tensor ends at or before the layout's size `S`, so a page of
/// pad past `S` holds every rounded prefix: the page-gather sweep's H2D window pad.
fn bake(dir: &Path, ops: &[EmittedOp<W1>], layout: &mut BundleLayout, dxp: &DxpTool) -> Result<()> {
    for b in layout.segment_bytes.iter_mut().filter(|b| **b > 0) {
        *b += HOST_PAGE;
    }
    let _ = std::fs::remove_dir_all(dir);
    triton_ktir_superdsc::bake::write_dir(dir, ops).map_err(|e| anyhow!("{}: {e}", dir.display()))?;
    triton_ktir_superdsc::bake::write_placements(dir, layout)
        .map_err(|e| anyhow!("{}: {e}", dir.display()))?;
    dxp.compile_dir(dir, ChunkExplore::of_group(ops.iter().map(|e| e.dsc())))
        .map_err(|e| anyhow!("{}: {e}", dir.display()))
}

/// A splice-minted program, placed by the one-kernel layout and lowered by the walk chat's bake runs.
fn lowered(node: &KtirNode) -> Result<(BundleLayout, Vec<EmittedOp<W1>>)> {
    let layout = triton_ktir_superdsc::layout::for_node(node).map_err(|e| anyhow!("{e}"))?;
    allocate_paged_cache(node, &layout).map_err(|e| anyhow!("{e}"))?;
    let mut sym = 0i64;
    let ops = ktir_superdsc::emit::whole_function::lower_walk::<W1>(
        node,
        Some(&layout),
        &mut sym,
        &mut ktir_superdsc::emit::ktir_matmul_fp8::QuantizedActivations::default(),
    )
    .map_err(|e| anyhow!("{}", e.message))?;
    Ok((layout, ops))
}

/// The reserved tensors the layout placed, staged with the values production stages them with.
/// Every synthetic constant comes from the runtime's one list, `wiring::synthetic_constants`: the
/// W8A8 clamp sticks `FP8_{POS,NEG,INV}448`, the registry slots at `scalarmul_scale_tid(i)` (here
/// this program's own `registry_constants`), and the rest. The paged step's int32 zero stick is
/// zero words, as `paged_step` stages it. Any other reserved placement is refused by name: this leg
/// would not know what to stage there. `rows` is the program's query rows, the bundle's `mq_pad`.
fn reserved_inputs(
    dir: &Path,
    node: &KtirNode,
    layout: &BundleLayout,
    rows: u32,
) -> Result<Vec<String>> {
    // `ConstantEnv::scalarmul_scales` is `'static` (the bake emits it as a `static`); this leg's are
    // computed once per program, so they are leaked to that lifetime for the run.
    let scales: &'static [f32] = Box::leak(
        triton_ktir_superdsc::layout::registry_constants(node)
            .map_err(|e| anyhow!("{e}"))?
            .into_boxed_slice(),
    );
    let synthetic = scratchy_target_spyre::wiring::synthetic_constants(
        &scratchy_target_spyre::wiring::ConstantEnv {
            hidden: (NQH * HD) as usize,
            head_dim: HD as usize,
            mq_pad: rows as usize,
            uses_ones_reduce: false,
            ones_reduce_len: None,
            uses_identity: false,
            rope_class_hds: &[],
            attn_class_hds: &[],
            rms_invcols: &[],
            rows: rows as usize,
            scalarmul_scales: scales,
        },
    );
    let mut out = Vec::new();
    for (&tid, p) in layout.placements.iter().filter(|(t, _)| **t >= RESERVED_TID_FLOOR) {
        let size = p.size as usize;
        let bytes = match synthetic.iter().find(|(t, _)| *t == tid) {
            Some((_, values)) => {
                let mut b = f16_bytes(&values.to_vec());
                if b.len() > size {
                    bail!(
                        "t{tid}: production stages {} B of constants into a {size} B placement",
                        b.len()
                    );
                }
                b.resize(size, 0);
                b
            }
            None if tid == PAGED_INT32_ZERO_TID => vec![0u8; size],
            None => bail!("t{tid}: a reserved placement this leg does not stage"),
        };
        out.push(format!("t{tid}={}", put(dir, &format!("reserved_{tid}.bin"), &bytes)?));
    }
    Ok(out)
}

/// ⭐ THE READ-BACK ROLE. The one-kernel layout marks `Logits` only a parameter a DIRECT store writes
/// (`store_views` follows `ktdp.construct_access_tile`s alone), and every output of the write and of
/// the paged attention is written through an INDIRECT tile — the pools' scatters, `out.index_copy_`.
/// `Executor::load` takes a bundle's read-back tensor from its `Logits` placement, so the leg marks
/// the tensors it reads back. The role is placement metadata only: the segment, offset, size and every
/// descriptor are the layout's.
fn read_back(layout: &mut BundleLayout, tids: &[u32]) -> Result<()> {
    for &tid in tids {
        layout
            .placements
            .get_mut(&tid)
            .ok_or_else(|| anyhow!("t{tid} is not placed"))?
            .role = SegRole::Logits;
    }
    Ok(())
}

/// The per-sequence attention's `mask_stack` position in `paged_attn_program`'s positional mint
/// (`[query, query_row_index, k_pages, v_pages, page_index_table, (kv_row_pool,) mask_stack, out]`):
/// the decode kernel's `kv_row_pool` sits at 5, so its mask sits one later.
fn mask_tid(seq: PagedSequence) -> u32 {
    if seq == PagedSequence::Decode { 6 } else { 5 }
}

/// The per-sequence attention's `out` position — the one after its mask.
fn out_tid(seq: PagedSequence) -> u32 {
    mask_tid(seq) + 1
}

fn run(dir: &Path, inputs: Vec<String>, outputs: Vec<String>) -> Result<()> {
    scratchy_target_spyre_card::bundle_run::run(dir, &inputs, &outputs)
        .with_context(|| format!("bundle_run {}", dir.display()))
}

/// ⭐ PHASE 0 — the reference pool: the device's own `qfp8ch` over the pool image (the W8A8
/// activation quantizer's convert door, `fq_afp8_op`'s call), its `[POOL_ROWS, HD]` fp8 output
/// one stick a row — the row-outermost pool image itself.
fn reference_pool(root: &Path, dxp: &DxpTool, which: u32) -> Result<Vec<u8>> {
    let dir = root.join(format!("ref{which}"));
    let (src, dst) = (0u32, 1u32);
    let seg = SegRole::Activation.segment();
    let mut l = BundleLayout::default();
    let mut off = 0u64;
    // The image is a per-launch input; the fp8 pool is the program's one output, which the launch
    // reads back — `SegRole::Logits`, the role the one-kernel layout gives every stored parameter.
    for (tid, size, role) in [
        (
            src,
            u64::from(POOL_ROWS) * u64::from(HD) * 2,
            SegRole::Activation,
        ),
        (dst, u64::from(POOL_ROWS) * u64::from(HD), SegRole::Logits),
    ] {
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role,
                segment: seg,
                bank: 0,
                offset: off,
                size,
            },
        );
        off += size;
    }
    // `bake` adds the one host page of pad every segment takes.
    l.segment_bytes[seg] = off;
    let mut sym = 0i64;
    let op = ktir_superdsc::emit::assemble_convert::<W1>(
        "qfp8ch_ref",
        "qfp8ch",
        POOL_ROWS,
        HD,
        &ktir_superdsc::emit::rb(&act_name(src), POOL_ROWS, HD),
        &ktir_superdsc::emit::rb_df(&act_name(dst), POOL_ROWS, HD, Df::Fp8),
        &mut sym,
        Some(&l),
    )
    .map_err(|e| anyhow!("{e}"))?;
    bake(&dir, &[op], &mut l, dxp)?;
    let x = put(&dir, "image.bin", &f16_bytes(&pool_image(which)))?;
    let pool = dir.join("pool.bin");
    run(
        &dir,
        vec![format!("t{src}={x}"), format!("stick:t{src}={POOL_ROWS}")],
        vec![format!("raw:t{dst}={}", pool.display())],
    )?;
    std::fs::read(&pool).with_context(|| format!("read {}", pool.display()))
}

/// ⭐ PHASE 1 — the production KV write over the K and V patterns as `[T, NKVH·HD]` rows, into zeroed
/// fp8 pools; the written pools' bytes.
fn written_pools(root: &Path, dxp: &DxpTool) -> Result<(Vec<u8>, Vec<u8>)> {
    let dir = root.join("write");
    let node = kv_write_program::<Fp8>(KvWriteGeometry {
        rows: T,
        nkvh: NKVH,
        hd: HeadWidth::of_whole_sticks(HD),
        pages: PAGES,
    })
    .map_err(|e| anyhow!(e))?;
    let (mut layout, ops) = lowered(&node)?;
    // Both pools are read back (raw, below): `[key, value, k_rows, v_rows, row_index]`.
    read_back(&mut layout, &[2, 3])?;
    bake(&dir, &ops, &mut layout, dxp)?;
    let rows = |which: u32| -> Vec<f32> {
        let mut x = vec![0f32; (T * NKVH * HD) as usize];
        for t in 0..T {
            for kv in 0..NKVH {
                for d in 0..HD {
                    x[(t * NKVH * HD + kv * HD + d) as usize] = pattern(which, kv, t, d);
                }
            }
        }
        x
    };
    // spyre-inference's `kv_write_index`: head `h` of slot `t` of page 0 is row `h·BLOCK + t`, one
    // `[T]` index per head, one after another.
    let index: Vec<u32> = (0..NKVH)
        .flat_map(|h| (0..T).map(move |t| h * BLOCK + t))
        .collect();
    let mut inputs = vec![
        format!("t0={}", put(&dir, "key.bin", &f16_bytes(&rows(0)))?),
        format!("stick:t0={T}"),
        format!("t1={}", put(&dir, "value.bin", &f16_bytes(&rows(1)))?),
        format!("stick:t1={T}"),
        format!("t2={}", put(&dir, "k_zero.bin", &vec![0u8; size_of(&layout, 2)?])?),
        "raw:t2".to_string(),
        format!("t3={}", put(&dir, "v_zero.bin", &vec![0u8; size_of(&layout, 3)?])?),
        "raw:t3".to_string(),
        format!("t4={}", put(&dir, "row_index.bin", &u32_bytes(&index))?),
        "raw:t4".to_string(),
    ];
    inputs.extend(reserved_inputs(&dir, &node, &layout, T)?);
    let (k, v) = (dir.join("k_pool.bin"), dir.join("v_pool.bin"));
    run(
        &dir,
        inputs,
        vec![
            format!("raw:t2={}", k.display()),
            format!("raw:t3={}", v.display()),
        ],
    )?;
    Ok((
        std::fs::read(&k).with_context(|| format!("read {}", k.display()))?,
        std::fs::read(&v).with_context(|| format!("read {}", v.display()))?,
    ))
}

/// The pools an attention bundle reads: fp8 pool bytes bound verbatim, or fp16 pool images staged
/// through the fp16 codec (`stick:…=1`, the identity map: a row-outermost pool is row-major).
enum Pools<'a> {
    Fp8 { k: &'a [u8], v: &'a [u8] },
    Fp16,
}

/// Which slot each query row reads, and the leg's own name for one launch's files.
struct Launch<'a> {
    slot_of: &'a dyn Fn(u32) -> u32,
    tag: &'a str,
}

/// One attention launch at `launch`'s slot law; the output row-major `[rows, NQH·HD]` fp16 words.
fn attend(
    dir: &Path,
    node: &KtirNode,
    layout: &BundleLayout,
    seq: PagedSequence,
    rows: u32,
    pools: &Pools<'_>,
    launch: Launch<'_>,
) -> Result<Vec<u16>> {
    let Launch { slot_of, tag } = launch;
    let decode = seq == PagedSequence::Decode;
    // `[query, query_row_index, k_pages, v_pages, page_index_table, (kv_row_pool,) mask_stack, out]`.
    let (mask, out) = (mask_tid(seq), out_tid(seq));
    let q = vec![0f32; (rows * NQH * HD) as usize];
    let mut m = vec![MASKED; (rows * BLOCK) as usize];
    for r in 0..rows {
        m[(r * BLOCK + slot_of(r)) as usize] = 0.0;
    }
    let qri: Vec<u32> = (0..rows).collect();
    // `page_index_table` `[NUM_BLOCKS, 32]`: block 0's page id, page 0, at column 0.
    let table = vec![0u32; 32];
    let mut inputs = vec![
        format!("t0={}", put(dir, &format!("{tag}_q.bin"), &f16_at(layout, 0, &q)?)?),
        format!("stick:t0={rows}"),
        format!("t1={}", put(dir, &format!("{tag}_qri.bin"), &u32_bytes(&qri))?),
        "raw:t1".to_string(),
        format!("t4={}", put(dir, &format!("{tag}_table.bin"), &u32_bytes(&table))?),
        "raw:t4".to_string(),
        format!("t{mask}={}", put(dir, &format!("{tag}_mask.bin"), &f16_at(layout, mask, &m)?)?),
        format!("stick:t{mask}={rows}"),
        format!("stick:t{out}={rows}"),
    ];
    if decode {
        let pool_rows: Vec<u32> = (0..PAGES * NKVH).collect();
        inputs.push(format!("t5={}", put(dir, &format!("{tag}_kvrp.bin"), &u32_bytes(&pool_rows))?));
        inputs.push("raw:t5".to_string());
    }
    match pools {
        Pools::Fp8 { k, v } => {
            inputs.push(format!("t2={}", put(dir, &format!("{tag}_k.bin"), k)?));
            inputs.push("raw:t2".to_string());
            inputs.push(format!("t3={}", put(dir, &format!("{tag}_v.bin"), v)?));
            inputs.push("raw:t3".to_string());
        }
        Pools::Fp16 => {
            inputs.push(format!("t2={}", put(dir, &format!("{tag}_k.bin"), &f16_at(layout, 2, &[])?)?));
            inputs.push("stick:t2=1".to_string());
            inputs.push(format!("t3={}", put(dir, &format!("{tag}_v.bin"), &f16_at(layout, 3, &pool_image(1))?)?));
            inputs.push("stick:t3=1".to_string());
        }
    }
    inputs.extend(reserved_inputs(dir, node, layout, rows)?);
    let o = dir.join(format!("{tag}_out.bin"));
    run(dir, inputs, vec![format!("t{out}={}", o.display())])?;
    let bytes = std::fs::read(&o).with_context(|| format!("read {}", o.display()))?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .take((rows * NQH * HD) as usize)
        .map(|&c| u16::from_le_bytes(c))
        .collect())
}

/// The first element where `got` differs from the pattern's V row `slot_of(row)` of kv head
/// `j / (NQH/NKVH)` for every query head `j`: `(index, got, want)`.
fn first_wrong(got: &[u16], rows: u32, slot_of: &dyn Fn(u32) -> u32) -> Option<(usize, u16, u16)> {
    let qpk = NQH / NKVH;
    (0..rows).find_map(|r| {
        (0..NQH * HD).find_map(|c| {
            let (j, d) = (c / HD, c % HD);
            let i = (r * NQH * HD + c) as usize;
            let want = half::f16::from_f32(pattern(1, j / qpk, slot_of(r), d)).to_bits();
            (got.get(i) != Some(&want)).then(|| (i, got.get(i).copied().unwrap_or(0), want))
        })
    })
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_writer(std::io::stderr)
        .init();
    let root = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or_else(|| anyhow!("usage: kv_fp8_round_trip <work-dir>"))?,
    );
    std::fs::create_dir_all(&root).with_context(|| format!("create {}", root.display()))?;
    let dxp = DxpTool::resolve()
        .ok_or_else(|| anyhow!("no dxp_standalone: DEEPTOOLS_PATH must name the SDK's share dir"))?;
    check_pattern()?;
    let mut failed: Vec<String> = Vec::new();

    // ── PHASE 0 and 1: the reference pools, the written pools, and the write's one check ──
    let (ref_k, ref_v) = (reference_pool(&root, &dxp, 0)?, reference_pool(&root, &dxp, 1)?);
    let (w_k, w_v) = written_pools(&root, &dxp)?;
    let image = (POOL_ROWS * HD) as usize;
    for (pool, w, r) in [("K", &w_k, &ref_k), ("V", &w_v, &ref_v)] {
        let (w, r) = (&w[..image.min(w.len())], &r[..image.min(r.len())]);
        match (0..image).find(|&i| w.get(i) != r.get(i)) {
            None => println!("PASS phase 1 (write): the {pool} pool is the device qfp8ch's, byte for byte"),
            Some(i) => failed.push(format!(
                "FAIL phase 1 (write): {pool} pool byte {i} (row {}, col {}) is {:?}, the reference's {:?}",
                i / HD as usize,
                i % HD as usize,
                w.get(i),
                r.get(i)
            )),
        }
    }

    // ── PHASES 2c, 2, 3: decode and prefill attention, at the fp16 control and at fp8 ──
    let hd = HeadWidth::of_whole_sticks(HD);
    let geometry = |seq: PagedSequence, rows: u32| PagedAttnGeometry {
        seq,
        rows,
        nqh: NQH,
        nkvh: NKVH,
        hd,
        pages: PAGES,
        blocks: 1,
        scale: SCALE,
    };
    let decode_slot = |s: u32| move |_r: u32| s;
    let prefill_slot = |r: u32| 2 * r;
    for (name, seq, rows) in [
        ("decode", PagedSequence::Decode, 1u32),
        (
            "prefill",
            PagedSequence::Prefill {
                padded_query_len: PREFILL_ROWS,
            },
            PREFILL_ROWS,
        ),
    ] {
        let at = |fmt: &str| root.join(format!("{name}{fmt}"));
        let n16 = paged_attn_program::<Fp16>(geometry(seq, rows)).map_err(|e| anyhow!(e))?;
        let n8 = paged_attn_program::<Fp8>(geometry(seq, rows)).map_err(|e| anyhow!(e))?;
        let (mut l16, ops16) = lowered(&n16)?;
        let (mut l8, ops8) = lowered(&n8)?;
        read_back(&mut l16, &[out_tid(seq)])?;
        read_back(&mut l8, &[out_tid(seq)])?;
        bake(&at("16"), &ops16, &mut l16, &dxp)?;
        bake(&at("8"), &ops8, &mut l8, &dxp)?;
        let slots: Vec<u32> = if seq == PagedSequence::Decode {
            SLOTS.to_vec()
        } else {
            vec![0]
        };
        for s in slots {
            let fixed = decode_slot(s);
            let slot_of: &dyn Fn(u32) -> u32 = if seq == PagedSequence::Decode {
                &fixed
            } else {
                &prefill_slot
            };
            for (phase, short, dir, node, layout, pools) in [
                ("2c (fp16 control)", "c", at("16"), &n16, &l16, Pools::Fp16),
                ("2 (read, reference pool)", "r", at("8"), &n8, &l8, Pools::Fp8 { k: &ref_k, v: &ref_v }),
                ("3 (round trip, written pool)", "w", at("8"), &n8, &l8, Pools::Fp8 { k: &w_k, v: &w_v }),
            ] {
                let tag = format!("{short}_s{s}");
                let got = attend(&dir, node, layout, seq, rows, &pools, Launch { slot_of, tag: &tag })?;
                match first_wrong(&got, rows, slot_of) {
                    None => println!("PASS phase {phase}: {name} slot law at s={s}"),
                    Some((i, g, w)) => failed.push(format!(
                        "FAIL phase {phase}: {name}, s={s}: element {i} (row {}, head {}, d {}) is \
                         0x{g:04x}, the pattern's 0x{w:04x}",
                        i / (NQH * HD) as usize,
                        (i % (NQH * HD) as usize) / HD as usize,
                        i % HD as usize
                    )),
                }
            }
        }
    }
    for f in &failed {
        println!("{f}");
    }
    if failed.is_empty() {
        println!("kv_fp8_round_trip: every check bit-exact");
        Ok(())
    } else {
        bail!("kv_fp8_round_trip: {} check(s) failed", failed.len())
    }
}

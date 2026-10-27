// SPDX-License-Identifier: Apache-2.0
//! The qwen3-5-moe block's kernels (AI-native-Systems-Research/scratchy#857) on the H100, against
//! f64 CPU references and exact host results, and their launchers' refusals.
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips: this test is the
//! kernels' only gate. Every case prints its line before a test fails, so one run reports all the
//! numbers. Each bound is #841's, stated with the case that uses it.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod numerics;
}

use common::numerics::{bf16_inputs, bits, mismatched_bits};
use half::bf16;
use scratchy_target_cuda::kernels::{
    KernelError, SortRegions, expert_gemm_bf16, expert_gemv_bf16, gate_scale_bf16, gemm_bf16_abt,
    gemv_bf16_abt, moe_combine_bf16, moe_route_bf16, moe_sort,
};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};
use scratchy_target_cuda::step::sort_index_regions;
use scratchy_target_cuda::tape::moe_tiles;

/// Extra output elements past the shape, which a kernel must leave untouched.
const TAIL: usize = 4096;
/// Every output element before a launch: a bf16 NaN, so an element a kernel never writes fails.
const SENTINEL: u16 = 0xFFFF;
/// qwen3-5-moe's hidden size.
const HIDDEN: usize = 2048;

fn h100() -> CudaDevice {
    CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"))
}

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

fn up(device: &CudaDevice, host: &[u16]) -> DeviceBuffer {
    DeviceBuffer::upload(device, host).expect("upload")
}

fn f(b: u16) -> f64 {
    f64::from(bf16::from_bits(b).to_f32())
}

// ── M6 GateScale ───────────────────────────────────────────────────

/// Run the gate scale over `(rows, cols)`; the output and whether the tail past it held.
fn run_gate_scale(
    device: &CudaDevice,
    (routed, shared, g): (&[u16], &[u16], &[u16]),
    (rows, cols): (usize, usize),
) -> (Vec<u16>, bool) {
    let (rb, sb, gb) = (up(device, routed), up(device, shared), up(device, g));
    let out = up(device, &vec![SENTINEL; rows * cols + TAIL]);
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `routed`, `shared` hold rows·cols bf16,
    // `g` rows, `out` rows·cols and the tail; `out` is its own allocation; null is the default
    // stream, which the download below waits for.
    unsafe {
        gate_scale_bf16(
            (rb.ptr(), sb.ptr(), gb.ptr()),
            out.ptr(),
            (rows as u32, cols as u32),
            std::ptr::null_mut(),
        )
    }
    .expect("gate_scale_bf16 launch");
    let mut o = out.download().expect("download the output");
    let tail = o[rows * cols..].iter().all(|&b| b == SENTINEL);
    o.truncate(rows * cols);
    (o, tail)
}

/// `routed + shared · σ(g)` at qwen3-5-moe's hidden size and 1, 7 and 512 rows, gates in [-8, 8]
/// plus the extremes ±100 on row 0: every element within #841's bound `2⁻⁸·|ref| +
/// 2⁻¹⁸·|shared·σ(g)| + 2⁻²⁴`, no output non-finite, the tail intact; a repeat bit-identical; and
/// the launcher refusing `cols` off its 8-wide rule and zero dimensions with `out` unchanged.
#[test]
fn gate_scale_bf16_matches_the_f64_reference() {
    let device = h100();
    let mut pass = true;
    for rows in [1usize, 7, 512] {
        let cols = HIDDEN;
        let seed = 0x6A5C_0000 ^ rows as u64;
        let routed = bf16_inputs(rows * cols, seed);
        let shared = bf16_inputs(rows * cols, seed ^ 1);
        let mut g: Vec<u16> = (bf16_inputs(rows, seed ^ 2).into_iter())
            .map(|b| bits(bf16::from_bits(b).to_f32() * 8.0))
            .collect();
        g[0] = bits(if rows == 7 { -100.0 } else { 100.0 });
        let (gpu, tail) = run_gate_scale(&device, (&routed, &shared, &g), (rows, cols));
        let (mut bad, mut worst, mut nonfinite) = (0usize, 0.0f64, 0usize);
        for (r, &gr) in g.iter().enumerate() {
            let sig = 1.0 / (1.0 + (-f(gr)).exp());
            for c in 0..cols {
                let i = r * cols + c;
                let scaled = f(shared[i]) * sig;
                let want = f(routed[i]) + scaled;
                let bound = want.abs() / 256.0 + scaled.abs() / 262_144.0 + 2f64.powi(-24);
                let got = f(gpu[i]);
                nonfinite += usize::from(!got.is_finite());
                let err = (got - want).abs();
                bad += usize::from(err.is_nan() || err > bound);
                worst = worst.max(err / bound);
            }
        }
        let repeat = run_gate_scale(&device, (&routed, &shared, &g), (rows, cols)).0;
        let rep = mismatched_bits(&gpu, &repeat);
        let ok = bad == 0 && nonfinite == 0 && tail && rep == 0;
        eprintln!(
            "gate_scale_bf16 rows={rows} cols={cols}: out_of_bound={bad} worst_ratio={worst:.4} \
             nonfinite={nonfinite} tail_intact={tail} repeat_mismatched_bits={rep} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    for (rows, cols) in [(7u32, 2044u32), (7, 0), (0, 2048), (65_536, 65_536)] {
        let len = (rows as usize * cols as usize).clamp(8, 1 << 16);
        let x = up(&device, &vec![0; len]);
        let out = up(&device, &vec![SENTINEL; len]);
        // SAFETY: the launcher refuses these shapes before it touches any pointer.
        let got = unsafe {
            gate_scale_bf16(
                (x.ptr(), x.ptr(), x.ptr()),
                out.ptr(),
                (rows, cols),
                std::ptr::null_mut(),
            )
        };
        let kept = (out.download().expect("out").iter()).all(|&b| b == SENTINEL);
        let ok = got == Err(KernelError::RowShape { rows, cols }) && kept;
        eprintln!(
            "gate_scale_bf16 refusal rows={rows} cols={cols}: {got:?} out_unchanged={kept} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "the gate scale failed: see the lines above");
}

// ── M1 MoeRoute ────────────────────────────────────────────────────

/// qwen3-5-moe's experts and picks.
const EXPERTS: usize = 256;
const TOP_K: usize = 8;

fn up_u32(device: &CudaDevice, host: &[u32]) -> DeviceBuffer {
    let halves: Vec<u16> = (host.iter())
        .flat_map(|w| [*w as u16, (*w >> 16) as u16])
        .collect();
    up(device, &halves)
}

fn down_u32(b: &DeviceBuffer) -> Vec<u32> {
    let h = b.download().expect("download");
    (h.as_chunks::<2>().0.iter())
        .map(|p| u32::from(p[0]) | u32::from(p[1]) << 16)
        .collect()
}

/// Run the routing of `logits` (`[m, 256]` bf16); its indices and scores, `[m, top_k]` each.
fn run_route(
    device: &CudaDevice,
    logits: &[u16],
    (m, top_k): (usize, usize),
    renorm: bool,
) -> (Vec<u32>, Vec<f32>) {
    let lb = up(device, logits);
    let idx = up_u32(device, &vec![u32::MAX; m * top_k]);
    let sc = up_u32(device, &vec![u32::MAX; m * top_k]);
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `logits` holds m·256 bf16, `idx` and `sc`
    // m·top_k 4-byte values each, both their own allocations; null is the default stream, which
    // the downloads below wait for.
    unsafe {
        moe_route_bf16(
            lb.ptr(),
            (idx.ptr(), sc.ptr()),
            (m as u32, EXPERTS as u32, top_k as u32),
            renorm,
            std::ptr::null_mut(),
        )
    }
    .expect("moe_route_bf16 launch");
    let scores = down_u32(&sc).into_iter().map(f32::from_bits).collect();
    (down_u32(&idx), scores)
}

/// One row's f64 routing: the picks ascending (slot `top_k − 1` the best; equal probabilities by
/// the higher index), their renormalised probabilities, and whether the `top_k`-th and next
/// probabilities are within `tie` of each other (a near-tie row).
fn route_reference(row: &[u16], top_k: usize, tie: f64) -> (Vec<u32>, Vec<f64>, bool) {
    let x: Vec<f64> = row.iter().map(|&b| f(b)).collect();
    let mx = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = x.iter().map(|v| (v - mx).exp()).collect();
    let s: f64 = e.iter().sum();
    let p: Vec<f64> = e.iter().map(|v| v / s).collect();
    let mut order: Vec<usize> = (0..EXPERTS).collect();
    // Descending probability, the higher index first on a tie.
    order.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(b.cmp(&a)));
    let picks = &order[..top_k];
    let near = (p[order[top_k - 1]] - p[order[top_k]]).abs() <= tie;
    let sum: f64 = picks.iter().map(|&i| p[i]).sum();
    let idx = picks.iter().rev().map(|&i| i as u32).collect();
    let sc = picks.iter().rev().map(|&i| p[i] / sum).collect();
    (idx, sc, near)
}

/// The route's f32 yardstick on the host: the same formula in f32, its worst absolute score error
/// against f64 over every row (the bound's measured noise).
fn route_yardstick(logits: &[u16], m: usize) -> f64 {
    let mut worst = 0.0f64;
    for t in 0..m {
        let row = &logits[t * EXPERTS..][..EXPERTS];
        let x: Vec<f32> = row.iter().map(|&b| bf16::from_bits(b).to_f32()).collect();
        let mx = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = x.iter().map(|v| (v - mx).exp()).collect();
        let s: f32 = e.iter().sum();
        let p: Vec<f32> = e.iter().map(|v| v / s).collect();
        let (idx, want, _) = route_reference(row, TOP_K, 0.0);
        let sum: f32 = idx.iter().map(|&i| p[i as usize]).sum();
        for (k, &i) in idx.iter().enumerate() {
            worst = worst.max((f64::from(p[i as usize] / sum) - want[k]).abs());
        }
    }
    worst
}

/// The routing at m ∈ {1, 7, 4096} on random logits in [-4, 4]: indices equal to the f64
/// routing's on every row that is not a near-tie (its 8th and 9th probabilities within 4× the f32
/// yardstick's max error), scores within `4 × yardstick + 2⁻³⁰`; the near-tie count printed; a
/// repeat bit-identical.
#[test]
fn moe_route_bf16_matches_the_f64_routing() {
    let device = h100();
    let mut pass = true;
    for m in [1usize, 7, 4096] {
        let logits: Vec<u16> = (bf16_inputs(m * EXPERTS, 0x5017_0000 ^ m as u64).into_iter())
            .map(|b| bits(bf16::from_bits(b).to_f32() * 4.0))
            .collect();
        let yard = route_yardstick(&logits, m);
        let tie = 4.0 * yard;
        let (idx, sc) = run_route(&device, &logits, (m, TOP_K), true);
        let (mut near, mut idx_off, mut sc_off) = (0usize, 0usize, 0usize);
        for t in 0..m {
            let (want_i, want_s, is_near) =
                route_reference(&logits[t * EXPERTS..][..EXPERTS], TOP_K, tie);
            near += usize::from(is_near);
            let got_i = &idx[t * TOP_K..][..TOP_K];
            if !is_near && got_i != want_i.as_slice() {
                idx_off += 1;
            }
            if got_i == want_i.as_slice() {
                for k in 0..TOP_K {
                    let err = (f64::from(sc[t * TOP_K + k]) - want_s[k]).abs();
                    sc_off += usize::from(err.is_nan() || err > tie + 2f64.powi(-30));
                }
            }
        }
        let again = run_route(&device, &logits, (m, TOP_K), true);
        let rep = usize::from(again.0 != idx)
            + usize::from(
                again
                    .1
                    .iter()
                    .map(|x| x.to_bits())
                    .ne(sc.iter().map(|x| x.to_bits())),
            );
        let ok = idx_off == 0 && sc_off == 0 && rep == 0;
        eprintln!(
            "moe_route_bf16 m={m} random: rows_with_other_picks={idx_off} near_tie_rows={near} \
             scores_out_of_bound={sc_off} yardstick_max_err={yard:.3e} repeat_differing={rep} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(
        pass,
        "the routing differs from the f64 routing: see the lines above"
    );
}

/// The routing's exact cases, one row each: (a) exact ties at the 8/9 boundary, experts 10, 20,
/// 30, 40 at one value and 100..=106 at a higher one, so the picks are 100..=106 and the highest
/// of the tied, 40; (b) all-equal logits: picks 248..=255, every score 1/8; (c) one dominant
/// expert 77 (the best, slot 7); (d) all-NaN logits: 8 distinct indices in `0..256`, every score
/// NaN.
#[test]
fn moe_route_bf16_takes_ties_by_the_higher_index_and_survives_nan() {
    let device = h100();
    let row =
        |set: &dyn Fn(usize) -> f32| -> Vec<u16> { (0..EXPERTS).map(|e| bits(set(e))).collect() };
    let ties = row(&|e| match e {
        100..=106 => 2.0,
        10 | 20 | 30 | 40 => 1.0,
        _ => -3.0,
    });
    let equal = row(&|_| 0.5);
    let dominant = row(&|e| {
        if e == 77 {
            6.0
        } else {
            -1.0 + (e % 7) as f32 * 0.25
        }
    });
    let nan: Vec<u16> = vec![0x7FC0; EXPERTS];
    let mut pass = true;
    let (i, _) = run_route(&device, &ties, (1, TOP_K), true);
    let want: Vec<u32> = vec![40, 100, 101, 102, 103, 104, 105, 106];
    let ok = i == want;
    eprintln!(
        "moe_route_bf16 exact ties: picks={i:?} want={want:?} {}",
        verdict(ok)
    );
    pass &= ok;
    let (i, s) = run_route(&device, &equal, (1, TOP_K), true);
    let ok = i == (248..256).collect::<Vec<u32>>() && s.iter().all(|&x| x == 0.125);
    eprintln!(
        "moe_route_bf16 all equal: picks={i:?} scores={s:?} {}",
        verdict(ok)
    );
    pass &= ok;
    let (i, _) = run_route(&device, &dominant, (1, TOP_K), true);
    let ok = i[TOP_K - 1] == 77;
    eprintln!("moe_route_bf16 dominant 77: picks={i:?} {}", verdict(ok));
    pass &= ok;
    let (i, s) = run_route(&device, &nan, (1, TOP_K), true);
    let distinct: std::collections::BTreeSet<u32> = i.iter().copied().collect();
    let ok = distinct.len() == TOP_K && i.iter().all(|&x| x < 256) && s.iter().all(|x| x.is_nan());
    eprintln!(
        "moe_route_bf16 all NaN: picks={i:?} distinct_in_range={} scores_nan={} {}",
        distinct.len() == TOP_K && i.iter().all(|&x| x < 256),
        s.iter().all(|x| x.is_nan()),
        verdict(ok)
    );
    pass &= ok;
    // Refusals: experts not 256, top_k 0 or 33, m 0; nothing written.
    for (m, e, k) in [(4u32, 128u32, 8u32), (4, 256, 0), (4, 256, 33), (0, 256, 8)] {
        let lb = up(&device, &vec![0; 4 * EXPERTS]);
        let out = up_u32(&device, &vec![u32::MAX; 4 * 33]);
        // SAFETY: the launcher refuses these shapes before it touches any pointer.
        let got = unsafe {
            moe_route_bf16(
                lb.ptr(),
                (out.ptr(), out.ptr()),
                (m, e, k),
                true,
                std::ptr::null_mut(),
            )
        };
        let kept = down_u32(&out).iter().all(|&x| x == u32::MAX);
        let want = Err(KernelError::MoeShape {
            kernel: "moe_route_bf16",
            dims: [m, e, k, 0],
        });
        let ok = got == want && kept;
        eprintln!(
            "moe_route_bf16 refusal m={m} experts={e} top_k={k}: {got:?} out_unchanged={kept} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "a routing exact case failed: see the lines above");
}

// ── M5 ExpertCombine ───────────────────────────────────────────────

/// Run the combine; the output `[m, hidden]` and whether the tail past it held.
fn run_combine(
    device: &CudaDevice,
    (down, pos, scores): (&[u16], Option<&[u32]>, &[f32]),
    (m, down_rows): (usize, usize),
) -> (Vec<u16>, bool) {
    let db = up(device, down);
    let pb = pos.map(|p| up_u32(device, p));
    let sb = up_u32(
        device,
        &scores.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
    );
    let out = up(device, &vec![SENTINEL; m * HIDDEN + TAIL]);
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `down` holds down_rows·HIDDEN bf16, `pos`
    // m·TOP_K u32 below down_rows, `scores` m·TOP_K f32, `out` m·HIDDEN and the tail, its own
    // allocation; null is the default stream, which the download below waits for.
    unsafe {
        moe_combine_bf16(
            (
                db.ptr(),
                pb.as_ref().map(|b| b.ptr().cast_const()),
                sb.ptr(),
            ),
            out.ptr(),
            (m as u32, TOP_K as u32, HIDDEN as u32, down_rows as u32),
            std::ptr::null_mut(),
        )
    }
    .expect("moe_combine_bf16 launch");
    let mut o = out.download().expect("download the output");
    let tail = o[m * HIDDEN..].iter().all(|&b| b == SENTINEL);
    o.truncate(m * HIDDEN);
    (o, tail)
}

/// The combine, gathered (m = 1, rows `8t + j`) and sorted (m = 512, rows through a shuffled
/// `pos` into 4,224 down rows): every element within #841's bound `2⁻⁸·|ref| + 2⁻²⁰·Σ_j|s_j·d_j|
/// + 2⁻²⁴`, the tail intact, a repeat bit-identical; and the launcher's refusals.
#[test]
fn moe_combine_bf16_matches_the_f64_sum() {
    let device = h100();
    let mut pass = true;
    for (m, sorted) in [(1usize, false), (512, true)] {
        let pairs = m * TOP_K;
        let down_rows = if sorted { pairs + 128 } else { pairs };
        let down = bf16_inputs(down_rows * HIDDEN, 0xC0B1_0000 ^ m as u64);
        let scores: Vec<f32> = (bf16_inputs(pairs, 0xC0B1_5C0E ^ m as u64).into_iter())
            .map(|b| bf16::from_bits(b).to_f32().abs() / 4.0)
            .collect();
        // A permutation of the down rows: pair p reads row (p · 37 + 11) mod down_rows, distinct.
        let pos: Vec<u32> = (0..pairs)
            .map(|p| ((p * 37 + 11) % down_rows) as u32)
            .collect();
        let row_of = |p: usize| if sorted { pos[p] as usize } else { p };
        let p = sorted.then_some(pos.as_slice());
        let (gpu, tail) = run_combine(&device, (&down, p, &scores), (m, down_rows));
        let (mut bad, mut worst) = (0usize, 0.0f64);
        for t in 0..m {
            for c in 0..HIDDEN {
                let (mut want, mut mag) = (0.0f64, 0.0f64);
                for j in 0..TOP_K {
                    let term = f64::from(scores[t * TOP_K + j])
                        * f(down[row_of(t * TOP_K + j) * HIDDEN + c]);
                    want += term;
                    mag += term.abs();
                }
                let bound = want.abs() / 256.0 + mag / 1_048_576.0 + 2f64.powi(-24);
                let err = (f(gpu[t * HIDDEN + c]) - want).abs();
                bad += usize::from(err.is_nan() || err > bound);
                worst = worst.max(err / bound);
            }
        }
        let rep = mismatched_bits(
            &gpu,
            &run_combine(&device, (&down, p, &scores), (m, down_rows)).0,
        );
        let ok = bad == 0 && tail && rep == 0;
        eprintln!(
            "moe_combine_bf16 m={m} sorted={sorted}: out_of_bound={bad} worst_ratio={worst:.4} \
             tail_intact={tail} repeat_mismatched_bits={rep} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    for (m, k, h, rows) in [
        (4u32, 8u32, 2000u32, 64u32),
        (4, 0, 2048, 64),
        (4, 8, 2048, 16),
        (0, 8, 2048, 64),
    ] {
        let x = up(&device, &vec![0; 64 * 2048]);
        let out = up(&device, &vec![SENTINEL; 4 * 2048]);
        // SAFETY: the launcher refuses these shapes before it touches any pointer.
        let got = unsafe {
            moe_combine_bf16(
                (x.ptr(), None, x.ptr()),
                out.ptr(),
                (m, k, h, rows),
                std::ptr::null_mut(),
            )
        };
        let kept = (out.download().expect("out").iter()).all(|&b| b == SENTINEL);
        let want = Err(KernelError::MoeShape {
            kernel: "moe_combine_bf16",
            dims: [m, k, h, rows],
        });
        let ok = got == want && kept;
        eprintln!(
            "moe_combine_bf16 refusal {:?}: {got:?} out_unchanged={kept} {}",
            [m, k, h, rows],
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "the combine failed: see the lines above");
}

// ── M2 ExpertSort ──────────────────────────────────────────────────

/// The grouped GEMM's row tile (`op_abi::CUDA_EXPERT_TILE_ROWS`).
const TILE_ROWS: usize = 128;

/// One sort's outputs, downloaded: its index regions and its sorted rows.
#[derive(PartialEq, Eq)]
struct Sorted {
    counts: Vec<u32>,
    offsets: Vec<u32>,
    tile_expert: Vec<u32>,
    pos: Vec<u32>,
    rows: Vec<u16>,
}

/// The exact host counting sort of `topk_idx` (`P` picks of `m` tokens over 256 experts): stable,
/// each run padded to 128 rows. `tile_expert` and `rows` are `[max_tiles]` and `[max_tiles · 128,
/// hidden]`, `u32::MAX` and `SENTINEL` where no live tile reaches.
fn sort_reference(topk_idx: &[u32], x: &[u16], hidden: usize, max_tiles: usize) -> Sorted {
    let mut counts = vec![0u32; EXPERTS];
    for &e in topk_idx {
        counts[e as usize] += 1;
    }
    let mut offsets = vec![0u32; EXPERTS + 1];
    for e in 0..EXPERTS {
        offsets[e + 1] = offsets[e] + ((counts[e] as usize).div_ceil(TILE_ROWS) * TILE_ROWS) as u32;
    }
    let mut tile_expert = vec![u32::MAX; max_tiles];
    for e in 0..EXPERTS {
        let tiles = offsets[e] as usize / TILE_ROWS..offsets[e + 1] as usize / TILE_ROWS;
        tile_expert[tiles].fill(e as u32);
    }
    let mut next = offsets.clone();
    let mut pos = vec![0u32; topk_idx.len()];
    let mut rows = vec![SENTINEL; max_tiles * TILE_ROWS * hidden];
    for (p, &e) in topk_idx.iter().enumerate() {
        let r = next[e as usize] as usize;
        next[e as usize] += 1;
        pos[p] = r as u32;
        let t = p / TOP_K;
        rows[r * hidden..(r + 1) * hidden].copy_from_slice(&x[t * hidden..(t + 1) * hidden]);
    }
    for e in 0..EXPERTS {
        for r in next[e] as usize..offsets[e + 1] as usize {
            rows[r * hidden..(r + 1) * hidden].fill(0);
        }
    }
    Sorted {
        counts,
        offsets,
        tile_expert,
        pos,
        rows,
    }
}

/// The device buffers of one sort's inputs and outputs, outputs at their sentinels.
struct SortBuffers {
    x: DeviceBuffer,
    topk_idx: DeviceBuffer,
    index: DeviceBuffer,
    rows: DeviceBuffer,
    regions: [usize; 6],
    max_tiles: usize,
}

/// Upload a sort's inputs (`m` tokens' rows `x` and `topk_idx`) and its sentinel outputs, at the
/// tape's `max_tiles` for `m` (`tape::moe_tiles`).
fn sort_buffers(device: &CudaDevice, x: &[u16], topk_idx: &[u32], m: usize) -> SortBuffers {
    let (pairs, max_tiles, _) = moe_tiles(m as u32, EXPERTS as u32, TOP_K as u32).expect("tiles");
    let regions = sort_index_regions(pairs, EXPERTS as u32, max_tiles)
        .expect("regions")
        .map(|b| b as usize);
    SortBuffers {
        x: up(device, x),
        topk_idx: up_u32(device, topk_idx),
        index: up_u32(device, &vec![u32::MAX; regions[5] / 4]),
        rows: up(
            device,
            &vec![SENTINEL; max_tiles as usize * TILE_ROWS * HIDDEN],
        ),
        regions,
        max_tiles: max_tiles as usize,
    }
}

/// Launch the sort on `b` (`m` tokens), asynchronously on the default stream.
fn launch_sort(b: &SortBuffers, m: usize) {
    let at = |i: usize| b.index.ptr().wrapping_byte_add(b.regions[i]);
    let regions = SortRegions {
        chunk_hist: at(0),
        counts: at(1),
        offsets: at(2),
        tile_expert: at(3),
        pos: at(4),
    };
    // SAFETY: cudaMalloc pointers are 256-byte aligned and every region of `sort_index_regions`
    // is 256-aligned inside `index`; `x` holds m·HIDDEN bf16, `topk_idx` m·TOP_K u32 below 256,
    // `rows` max_tiles·128·HIDDEN bf16; the outputs are their own allocations; null is the
    // default stream, which the downloads below wait for.
    unsafe {
        moe_sort(
            (b.x.ptr(), b.topk_idx.ptr()),
            regions,
            b.rows.ptr(),
            (
                m as u32,
                EXPERTS as u32,
                TOP_K as u32,
                HIDDEN as u32,
                b.max_tiles as u32,
            ),
            std::ptr::null_mut(),
        )
    }
    .expect("moe_sort launch");
}

/// Launch the sort on `b` (`m` tokens), wait, and download what it wrote.
fn run_sort(b: &SortBuffers, m: usize) -> Sorted {
    launch_sort(b, m);
    let index = down_u32(&b.index);
    let region = |i: usize, len: usize| index[b.regions[i] / 4..][..len].to_vec();
    Sorted {
        counts: region(1, EXPERTS),
        offsets: region(2, EXPERTS + 1),
        tile_expert: region(3, b.max_tiles),
        pos: region(4, m * TOP_K),
        rows: b.rows.download().expect("rows"),
    }
}

/// `m` tokens' picks: 8 distinct experts each, from `pick(t, j)` (deduplicated by stepping on).
fn picks(m: usize, pick: impl Fn(usize, usize) -> usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(m * TOP_K);
    for t in 0..m {
        let mut row: Vec<u32> = Vec::with_capacity(TOP_K);
        for j in 0..TOP_K {
            let mut e = pick(t, j) % EXPERTS;
            while row.contains(&(e as u32)) {
                e = (e + 1) % EXPERTS;
            }
            row.push(e as u32);
        }
        out.extend(row);
    }
    out
}

/// A splitmix64 stream, for routings.
fn mix(seed: u64) -> impl FnMut() -> u64 {
    let mut state = seed;
    move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The sort's routing cases at `m` tokens: every token to the same 8 experts (one long run each,
/// 248 empty experts), uniform random picks, and runs of exactly 128 and 129 pairs.
fn sort_cases(m: usize) -> Vec<(&'static str, Vec<u32>)> {
    let mut r = mix(0x5027_0000 ^ m as u64);
    let random: Vec<usize> = (0..m * TOP_K).map(|_| r() as usize).collect();
    vec![
        ("skewed", picks(m, |_, j| 200 + j)),
        ("uniform", picks(m, |t, j| random[t * TOP_K + j])),
        // Expert 7 gets tokens 0..128 (a run of exactly 128), expert 9 tokens 0..129; the rest
        // spread.
        (
            "runs_128_129",
            picks(m, |t, j| match j {
                0 if t < 128 => 7,
                1 if t < 129 => 9,
                _ => 16 + random[t * TOP_K + j] % 200,
            }),
        ),
    ]
}

/// The sort at P = 4,096 (m = 512) and P = 32,768 (m = 4,096), on each routing case: every index
/// region and every live row bit-exact against the host counting sort, padding rows `+0.0`, rows
/// past the live tiles never written; a repeat bit-identical; and the launcher's refusals.
#[test]
fn moe_sort_matches_the_host_counting_sort() {
    let device = h100();
    let mut pass = true;
    for m in [512usize, 4096] {
        let x = bf16_inputs(m * HIDDEN, 0x5027 ^ m as u64);
        for (name, topk_idx) in sort_cases(m) {
            let b = sort_buffers(&device, &x, &topk_idx, m);
            let got = run_sort(&b, m);
            let want = sort_reference(&topk_idx, &x, HIDDEN, b.max_tiles);
            let live = want.offsets[EXPERTS] as usize / TILE_ROWS;
            let regions_ok = got.counts == want.counts
                && got.offsets == want.offsets
                && got.tile_expert[..live] == want.tile_expert[..live]
                && got.pos == want.pos;
            let rows_bad = mismatched_bits(&got.rows, &want.rows);
            let repeat = run_sort(&b, m);
            let same = repeat == got;
            let ok = regions_ok && rows_bad == 0 && same;
            eprintln!(
                "moe_sort m={m} pairs={} {name}: live_tiles={live}/{} regions_exact={regions_ok} \
                 rows_mismatched_bits={rows_bad} repeat_identical={same} {}",
                m * TOP_K,
                b.max_tiles,
                verdict(ok)
            );
            pass &= ok;
        }
    }
    let rows = up(&device, &vec![SENTINEL; 1 << 16]);
    for (m, e, k, h, tiles) in [
        (512u32, 255u32, 8u32, 2048u32, 286u32),
        (512, 256, 0, 2048, 286),
        (512, 256, 33, 2048, 286),
        (512, 256, 8, 2044, 286),
        // One tile under `tape::moe_tiles(512, 256, 8)`'s 286.
        (512, 256, 8, 2048, 285),
        (0, 256, 8, 2048, 286),
    ] {
        let none = SortRegions {
            chunk_hist: rows.ptr(),
            counts: rows.ptr(),
            offsets: rows.ptr(),
            tile_expert: rows.ptr(),
            pos: rows.ptr(),
        };
        // SAFETY: the launcher refuses these shapes before it touches any pointer.
        let got = unsafe {
            moe_sort(
                (rows.ptr(), rows.ptr()),
                none,
                rows.ptr(),
                (m, e, k, h, tiles),
                std::ptr::null_mut(),
            )
        };
        let kept = (rows.download().expect("rows").iter()).all(|&b| b == SENTINEL);
        let want = Err(KernelError::MoeShape {
            kernel: "moe_sort",
            dims: [m, k, h, tiles],
        });
        let ok = got == want && kept;
        eprintln!(
            "moe_sort refusal {:?}: {got:?} out_unchanged={kept} {}",
            [m, e, k, h, tiles],
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "the sort failed: see the lines above");
}

// ── M3 ExpertGemm ──────────────────────────────────────────────────

/// qwen3-5-moe's expert width.
const INTER: usize = 512;

/// The grouped GEMM over the sorted `rows` (`max_tiles · 128` rows of `k`) and the sort's
/// `offsets`/`tile_expert`, B `[256, n, k]`; its output `[max_tiles · 128, n]` from a sentinel.
fn run_expert_gemm(
    device: &CudaDevice,
    (a, b): (&DeviceBuffer, &DeviceBuffer),
    (offsets, tile_expert): (&[u32], &[u32]),
    (max_tiles, n, k): (usize, usize, usize),
) -> Vec<u16> {
    let ob = up_u32(device, offsets);
    let tb = up_u32(device, tile_expert);
    let c = up(device, &vec![SENTINEL; max_tiles * TILE_ROWS * n]);
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `a` holds max_tiles·128·k bf16, `b`
    // 256·n·k, `offsets` 257 u32 and `tile_expert` max_tiles (the live ones below 256); `c` is its
    // own allocation of max_tiles·128·n; null is the default stream, which the download waits for.
    unsafe {
        expert_gemm_bf16(
            (a.ptr(), b.ptr()),
            (ob.ptr(), tb.ptr()),
            c.ptr(),
            (
                max_tiles as u32,
                n as u32,
                k as u32,
                EXPERTS as u32,
                n as u32,
            ),
            std::ptr::null_mut(),
        )
    }
    .expect("expert_gemm_bf16 launch");
    c.download().expect("download C")
}

/// The grouped GEMM on the sort's skewed and uniform outputs at m = 512 and 4,096, for the gate
/// and up's shape (`n` 512, `k` 2,048) and the down's (`n` 2,048, `k` 512): every live tile's rows
/// bit-identical to `gemm_bf16` on the same 128 rows against the same expert's `[n, k]`, within
/// the #761 GEMM bound against f64 (one tile per case, sampled), every dead tile's rows still at
/// the sentinel; and the launcher's refusals.
#[test]
fn expert_gemm_bf16_is_gemm_bf16_per_expert_tile() {
    let device = h100();
    let mut pass = true;
    for m in [512usize, 4096] {
        for (name, topk_idx) in sort_cases(m).into_iter().take(2) {
            let x = bf16_inputs(m * HIDDEN, 0x3E33 ^ m as u64);
            let sb = sort_buffers(&device, &x, &topk_idx, m);
            let sorted = run_sort(&sb, m);
            let live = sorted.offsets[EXPERTS] as usize / TILE_ROWS;
            let max_tiles = sb.max_tiles;
            for (proj, n, k) in [("gate_up", INTER, HIDDEN), ("down", HIDDEN, INTER)] {
                // The gate and up read the sorted rows; the down reads an activation of their
                // shape, here fresh inputs.
                let a_bits = if k == HIDDEN {
                    sorted.rows.clone()
                } else {
                    bf16_inputs(max_tiles * TILE_ROWS * k, 0xAC7 ^ m as u64)
                };
                let b_bits = bf16_inputs(EXPERTS * n * k, 0xB3 ^ n as u64);
                let (a, b) = (up(&device, &a_bits), up(&device, &b_bits));
                let gpu = run_expert_gemm(
                    &device,
                    (&a, &b),
                    (&sorted.offsets, &sorted.tile_expert),
                    (max_tiles, n, k),
                );
                let (mut bits_bad, mut bound_bad, mut worst) = (0usize, 0usize, 0.0f64);
                let tile_len = TILE_ROWS * n;
                let mut sampled = false;
                for t in 0..live {
                    let e = sorted.tile_expert[t] as usize;
                    let at = t * TILE_ROWS;
                    let a_t = up(&device, &a_bits[at * k..(at + TILE_ROWS) * k]);
                    let b_e = b.ptr().wrapping_byte_add(e * n * k * 2);
                    let c_t = up(&device, &vec![SENTINEL; tile_len]);
                    // SAFETY: `a_t` holds 128·k bf16, `b_e` expert e's n·k inside `b`, `c_t` its
                    // own allocation of 128·n; null is the default stream.
                    unsafe {
                        gemm_bf16_abt(
                            a_t.ptr(),
                            b_e,
                            c_t.ptr(),
                            TILE_ROWS as u32,
                            n as u32,
                            k as u32,
                            std::ptr::null_mut(),
                        )
                    }
                    .expect("gemm_bf16_abt launch");
                    let tiled = c_t.download().expect("download");
                    bits_bad += mismatched_bits(&gpu[at * n..][..tile_len], &tiled);
                    // One tile per case against f64, the first that is not all padding.
                    if !sampled && sorted.counts[e] > 0 {
                        sampled = true;
                        let cmp = common::numerics::compare(
                            &gpu[at * n..][..tile_len],
                            &a_bits[at * k..(at + TILE_ROWS) * k],
                            &b_bits[e * n * k..(e + 1) * n * k],
                            TILE_ROWS,
                            n,
                            k,
                        );
                        bound_bad += cmp.out_of_bound;
                        let (_, _, g, r, bd) = cmp.worst;
                        worst = (g - r).abs() / bd.max(f64::MIN_POSITIVE);
                    }
                }
                let dead_untouched = gpu[live * tile_len..].iter().all(|&v| v == SENTINEL);
                let ok = bits_bad == 0 && bound_bad == 0 && dead_untouched;
                eprintln!(
                    "expert_gemm_bf16 m={m} {name} {proj} n={n} k={k}: live_tiles={live}/\
                     {max_tiles} mismatched_bits={bits_bad} sampled_out_of_bound={bound_bad} \
                     sampled_worst_ratio={worst:.4} dead_tiles_untouched={dead_untouched} {}",
                    verdict(ok)
                );
                pass &= ok;
            }
        }
    }
    let c = up(&device, &vec![SENTINEL; 1 << 16]);
    for (tiles, n, k, e, rows) in [
        (0u32, 512u32, 2048u32, 256u32, 512u32),
        (4, 500, 2048, 256, 512),
        (4, 512, 2000, 256, 512),
        (4, 512, 2048, 255, 512),
        (4, 512, 2048, 256, 256),
    ] {
        // SAFETY: the launcher refuses these shapes before it touches any pointer.
        let got = unsafe {
            expert_gemm_bf16(
                (c.ptr(), c.ptr()),
                (c.ptr(), c.ptr()),
                c.ptr(),
                (tiles, n, k, e, rows),
                std::ptr::null_mut(),
            )
        };
        let kept = (c.download().expect("c").iter()).all(|&b| b == SENTINEL);
        let want = Err(KernelError::MoeShape {
            kernel: "expert_gemm_bf16",
            dims: [tiles, n, k, rows],
        });
        let ok = got == want && kept;
        eprintln!(
            "expert_gemm_bf16 refusal {:?}: {got:?} out_unchanged={kept} {}",
            [tiles, n, k, e, rows],
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "the grouped expert GEMM failed: see the lines above");
}

// ── M4 ExpertGemv ──────────────────────────────────────────────────

/// The gathered expert GEMV at m = 1 for the gate and up's shape (token rows, `n` 512, `k` 2,048)
/// and the down's (pair rows, `n` 2,048, `k` 512), with the 8 pairs on 8 distinct experts and on
/// one repeated expert: each pair's row bit-identical to `gemv_bf16` on the same row and expert
/// slice, within the #761 GEMM bound against f64, the tail intact; and the launcher's refusals.
#[test]
fn expert_gemv_bf16_is_gemv_bf16_per_pair() {
    let device = h100();
    let mut pass = true;
    let pairs = TOP_K;
    for (case, topk_idx) in [
        ("distinct", vec![3u32, 250, 17, 128, 0, 255, 64, 99]),
        ("repeated", vec![42u32; TOP_K]),
    ] {
        for (proj, n, k, pair_rows) in [
            ("gate_up", INTER, HIDDEN, false),
            ("down", HIDDEN, INTER, true),
        ] {
            let a_rows = if pair_rows { pairs } else { 1 };
            let a_bits = bf16_inputs(a_rows * k, 0x6E3 ^ n as u64);
            let b_bits = bf16_inputs(EXPERTS * n * k, 0xB4 ^ n as u64);
            let (a, b) = (up(&device, &a_bits), up(&device, &b_bits));
            let ib = up_u32(&device, &topk_idx);
            let c = up(&device, &vec![SENTINEL; pairs * n + TAIL]);
            // SAFETY: cudaMalloc pointers are 256-byte aligned; `a` holds a_rows·k bf16, `b`
            // 256·n·k, `topk_idx` 8 u32 below 256, `c` its own allocation of 8·n and the tail;
            // null is the default stream, which the download below waits for.
            unsafe {
                expert_gemv_bf16(
                    (a.ptr(), b.ptr(), ib.ptr()),
                    c.ptr(),
                    (1, TOP_K as u32, n as u32, k as u32),
                    (EXPERTS as u32, n as u32),
                    pair_rows,
                    std::ptr::null_mut(),
                )
            }
            .expect("expert_gemv_bf16 launch");
            let mut gpu = c.download().expect("download");
            let tail = gpu[pairs * n..].iter().all(|&v| v == SENTINEL);
            gpu.truncate(pairs * n);
            let (mut bits_bad, mut bound_bad, mut worst) = (0usize, 0usize, 0.0f64);
            for (p, &e) in topk_idx.iter().enumerate() {
                let (e, row) = (e as usize, if pair_rows { p } else { 0 });
                let a_p = &a_bits[row * k..(row + 1) * k];
                let a_up = up(&device, a_p);
                let b_e = b.ptr().wrapping_byte_add(e * n * k * 2);
                let c_p = up(&device, &vec![SENTINEL; n]);
                // SAFETY: `a_up` holds k bf16, `b_e` expert e's n·k inside `b`, `c_p` its own
                // allocation of n; null is the default stream.
                unsafe {
                    gemv_bf16_abt(
                        a_up.ptr(),
                        b_e,
                        c_p.ptr(),
                        (n as u32, k as u32),
                        std::ptr::null_mut(),
                    )
                }
                .expect("gemv_bf16_abt launch");
                let gemv = c_p.download().expect("download");
                bits_bad += mismatched_bits(&gpu[p * n..(p + 1) * n], &gemv);
                let cmp = common::numerics::compare(
                    &gpu[p * n..(p + 1) * n],
                    a_p,
                    &b_bits[e * n * k..(e + 1) * n * k],
                    1,
                    n,
                    k,
                );
                bound_bad += cmp.out_of_bound;
                let (_, _, g, r, bd) = cmp.worst;
                worst = worst.max((g - r).abs() / bd.max(f64::MIN_POSITIVE));
            }
            let ok = bits_bad == 0 && bound_bad == 0 && tail;
            eprintln!(
                "expert_gemv_bf16 m=1 {case} {proj} n={n} k={k}: mismatched_bits={bits_bad} \
                 out_of_bound={bound_bad} worst_ratio={worst:.4} tail_intact={tail} {}",
                verdict(ok)
            );
            pass &= ok;
        }
    }
    let c = up(&device, &vec![SENTINEL; 1 << 16]);
    for (m, k, n, kk, e, rows) in [
        (0u32, 8u32, 512u32, 2048u32, 256u32, 512u32),
        (1, 0, 512, 2048, 256, 512),
        (1, 33, 512, 2048, 256, 512),
        (1, 8, 500, 2048, 256, 512),
        (1, 8, 512, 2000, 256, 512),
        (1, 8, 512, 2048, 255, 512),
        (1, 8, 512, 2048, 256, 256),
        (8192, 8, 512, 2048, 256, 512),
    ] {
        // SAFETY: the launcher refuses these shapes before it touches any pointer.
        let got = unsafe {
            expert_gemv_bf16(
                (c.ptr(), c.ptr(), c.ptr()),
                c.ptr(),
                (m, k, n, kk),
                (e, rows),
                false,
                std::ptr::null_mut(),
            )
        };
        let kept = (c.download().expect("c").iter()).all(|&b| b == SENTINEL);
        let want = Err(KernelError::MoeShape {
            kernel: "expert_gemv_bf16",
            dims: [m, k, n, kk],
        });
        let ok = got == want && kept;
        eprintln!(
            "expert_gemv_bf16 refusal {:?}: {got:?} out_unchanged={kept} {}",
            [m, k, n, kk, e, rows],
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "the gathered expert GEMV failed: see the lines above");
}

// ── Timing ─────────────────────────────────────────────────────────

/// The median of 20 runs of `launch` after 3 warm-ups, each timed from the host to its completion
/// (`wait`, a download), in microseconds, and the minimum.
fn median_us(mut launch: impl FnMut(), wait: impl Fn()) -> (f64, f64) {
    let mut times = Vec::new();
    for i in 0..23 {
        let start = std::time::Instant::now();
        launch();
        wait();
        if i >= 3 {
            times.push(start.elapsed().as_secs_f64() * 1e6);
        }
    }
    times.sort_by(f64::total_cmp);
    (times[times.len() / 2], times[0])
}

/// The MoE kernels' time per launch at m = 1 and m = 4,096 (the sort and the grouped GEMM only
/// at 4,096, the GEMV only at 1: the buckets each runs on), printed and not asserted.
#[test]
fn the_moe_kernel_times_are_recorded() {
    let device = h100();
    for m in [1usize, 4096] {
        let topk_idx = picks(m, |t, j| t * 31 + j * 7);
        let logits = up(&device, &bf16_inputs(m * EXPERTS, 0x7E));
        let idx = up_u32(&device, &vec![0; m * TOP_K]);
        let sc = up_u32(&device, &vec![0; m * TOP_K]);
        let (med, min) = median_us(
            // SAFETY: as `run_route`'s.
            || {
                unsafe {
                    moe_route_bf16(
                        logits.ptr(),
                        (idx.ptr(), sc.ptr()),
                        (m as u32, EXPERTS as u32, TOP_K as u32),
                        true,
                        std::ptr::null_mut(),
                    )
                }
                .expect("route")
            },
            || drop(sc.download()),
        );
        eprintln!("moe_route_bf16 time m={m}: median={med:.1}us min={min:.1}us (not asserted)");
        let down = up(&device, &bf16_inputs(m * TOP_K * HIDDEN, 0x7F));
        let out = up(&device, &vec![0; m * HIDDEN]);
        let (med, min) = median_us(
            // SAFETY: as `run_combine`'s, gathered.
            || {
                unsafe {
                    moe_combine_bf16(
                        (down.ptr(), None, sc.ptr()),
                        out.ptr(),
                        (m as u32, TOP_K as u32, HIDDEN as u32, (m * TOP_K) as u32),
                        std::ptr::null_mut(),
                    )
                }
                .expect("combine")
            },
            || drop(out.download()),
        );
        eprintln!("moe_combine_bf16 time m={m}: median={med:.1}us min={min:.1}us (not asserted)");
        let g = up(&device, &bf16_inputs(m, 0x80));
        let (med, min) = median_us(
            // SAFETY: as `run_gate_scale`'s.
            || {
                unsafe {
                    gate_scale_bf16(
                        (down.ptr(), down.ptr(), g.ptr()),
                        out.ptr(),
                        (m as u32, HIDDEN as u32),
                        std::ptr::null_mut(),
                    )
                }
                .expect("gate scale")
            },
            || drop(out.download()),
        );
        eprintln!("gate_scale_bf16 time m={m}: median={med:.1}us min={min:.1}us (not asserted)");
        let b = up(&device, &bf16_inputs(EXPERTS * HIDDEN * INTER, 0x81));
        if m == 1 {
            let ib = up_u32(&device, &topk_idx);
            let x = up(&device, &bf16_inputs(HIDDEN, 0x82));
            let c = up(&device, &vec![0; TOP_K * INTER]);
            let (med, min) = median_us(
                // SAFETY: as `expert_gemv_bf16_is_gemv_bf16_per_pair`'s, the gate's shape.
                || {
                    unsafe {
                        expert_gemv_bf16(
                            (x.ptr(), b.ptr(), ib.ptr()),
                            c.ptr(),
                            (1, TOP_K as u32, INTER as u32, HIDDEN as u32),
                            (EXPERTS as u32, INTER as u32),
                            false,
                            std::ptr::null_mut(),
                        )
                    }
                    .expect("gemv")
                },
                || drop(c.download()),
            );
            eprintln!(
                "expert_gemv_bf16 time m=1 n={INTER} k={HIDDEN}: median={med:.1}us \
                 min={min:.1}us (not asserted)"
            );
            continue;
        }
        let x = bf16_inputs(m * HIDDEN, 0x83);
        let sb = sort_buffers(&device, &x, &topk_idx, m);
        let (med, min) = median_us(|| launch_sort(&sb, m), || drop(sb.index.download()));
        eprintln!("moe_sort time m={m}: median={med:.1}us min={min:.1}us (not asserted)");
        let sorted = run_sort(&sb, m);
        let ob = up_u32(&device, &sorted.offsets);
        let tb = up_u32(&device, &sorted.tile_expert);
        let c = up(&device, &vec![0; sb.max_tiles * TILE_ROWS * INTER]);
        let (med, min) = median_us(
            // SAFETY: as `run_expert_gemm`'s, the gate's shape.
            || {
                unsafe {
                    expert_gemm_bf16(
                        (sb.rows.ptr(), b.ptr()),
                        (ob.ptr(), tb.ptr()),
                        c.ptr(),
                        (
                            sb.max_tiles as u32,
                            INTER as u32,
                            HIDDEN as u32,
                            EXPERTS as u32,
                            INTER as u32,
                        ),
                        std::ptr::null_mut(),
                    )
                }
                .expect("grouped gemm")
            },
            || drop(c.download()),
        );
        eprintln!(
            "expert_gemm_bf16 time m={m} n={INTER} k={HIDDEN}: median={med:.1}us min={min:.1}us \
             (not asserted)"
        );
    }
}

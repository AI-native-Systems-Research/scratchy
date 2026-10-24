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
    KernelError, gate_scale_bf16, moe_combine_bf16, moe_route_bf16,
};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};

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

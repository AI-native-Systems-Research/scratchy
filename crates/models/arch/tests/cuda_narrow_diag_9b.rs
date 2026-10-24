// SPDX-License-Identifier: Apache-2.0
//! A diagnostic, not a gate: why does qwen3.5-9b's P1 on prompt 6 drop when the narrow GEMM moves
//! to ThunderKittens (TK-1, AI-native-Systems-Research/scratchy#846)? On the m=512 tape the only
//! kernel TK-1 changes that runs is `gemm_narrow_bf16` (the GDN `a`/`b` projections, m=512 n=32
//! k=4096); this test looks at every one of its 48 launches on the real prompt.
//!
//! It runs prompt 6 (161 real rows) of `cuda_hf_parity_9b`'s goldens on the real checkpoint, then
//! for every `GemmNarrow` step and every `GatedDeltaNet` step of the m=512 tape, one forward of a
//! copy of that tape cut just after the step, whose `result` is the step's output. So the output is
//! exactly what the full forward wrote there, and the step's input `A` is read back from the arena
//! afterwards (a narrow GEMM's `A` is the layer's normed hidden, written before it and live until
//! the GDN step reads it). Nothing in `src/` changes: the cut tapes are plain `CudaTape` data, and
//! `CudaWorker::forward` and `read_arena` are the public API.
//!
//! Per `GemmNarrow` step it prints, over the 161 real rows, the error against an f64 reference from
//! the same bf16 `A` and the same bf16 weights (downloaded from the device): `max|err|/bound`
//! (the kernel unit test's bound, `2⁻⁷·|ref| + 2⁻¹⁴·Σ|a·b|`), `mean_abs_err`, the worst element's
//! row and column, and `out_of_bound`, the count of elements outside that bound.
//!
//! It also writes each step's real rows as raw little-endian bf16 to
//! `<CARGO_TARGET_TMPDIR>/narrow_diag_9b/<kind>_<step>.bin` (the narrow outputs, and each GDN
//! layer's output). Run the same file at the base and at the change and compare the two dumps with
//! the script in the PR comment: it prints each GDN layer's output cosine, base against change,
//! and the narrow GEMM's per-step bit agreement.
//!
//! The only assertion is that every forward ran. Built only under `nvcc` and
//! `qwen3.5-9b-text-only`, and needs the H100, the checkpoint and the goldens, as
//! `cuda_hf_parity_9b`. Uses only APIs present at the base `82026acf`, so the same file builds
//! there.

use std::path::PathBuf;

use half::bf16;
use scratchy_models::qwen3_5::qwen3_5_9b::{CUDA_TAPES, CUDA_WEIGHTS};
use scratchy_target_cuda::runtime::{
    CudaDevice, CudaWorker, GdnPoolConfig, GdnSequence, Input, KvPoolConfig, KvSequence,
    load_weights,
};
use scratchy_target_cuda::step::{CudaStep, GatedDeltaNetBf16, GemmBf16};
use scratchy_target_cuda::tape::{ArenaSpan, CudaTape};

/// The goldens: `cuda_hf_reference.py --model qwen3.5-9b`'s.
const GOLDENS: &str = "tests/goldens/qwen3-5-qwen3.5-9b-hf";
/// The bucket, and the prompt whose P1 dropped.
const M: u32 = 512;
const PROMPT: u64 = 6;
const KV_POOL: KvPoolConfig = KvPoolConfig { num_blocks: 16 };
const GDN_POOL: GdnPoolConfig = GdnPoolConfig { num_slots: 2 };
const SEAT: (&[u32], u32) = (&[0, 1, 2, 3, 4, 5, 6, 7], 0);

fn goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(GOLDENS)
}

/// The goldens' checkpoint and prompt 6's ids.
fn prompt() -> (PathBuf, Vec<u32>) {
    let path = goldens_dir().join("meta.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}; make the goldens with `cuda_hf_reference.py --model qwen3.5-9b`",
            path.display()
        )
    });
    let meta: serde_json::Value = serde_json::from_str(&text).expect("meta.json parses");
    let checkpoint = PathBuf::from(meta["checkpoint"].as_str().expect("a checkpoint"));
    let entry = (meta["prompts"].as_array().expect("prompts").iter())
        .find(|e| e["prompt"].as_u64() == Some(PROMPT))
        .expect("prompt 6");
    let ids = (entry["ids"].as_array().expect("ids").iter())
        .map(|v| v.as_u64().expect("an id") as u32)
        .collect();
    (checkpoint, ids)
}

/// What one cut tape captures: the step's index on the full tape, and what it is.
#[derive(Clone, Copy)]
enum Capture {
    Narrow(usize, GemmBf16),
    Gdn(usize, GatedDeltaNetBf16),
}

/// The m=512 tape cut just after step `last`, its result the span `out`: a `'static` copy for
/// `CudaWorker`. Every span stays inside the same arena, so it validates as the full tape does.
fn cut(tape: &CudaTape, last: usize, out: ArenaSpan) -> CudaTape {
    CudaTape {
        steps: Box::leak(tape.steps[..=last].to_vec().into_boxed_slice()),
        result: out,
        ..*tape
    }
}

/// `x` as f64.
fn widen(x: &[u16]) -> Vec<f64> {
    x.iter()
        .map(|&b| f64::from(bf16::from_bits(b).to_f32()))
        .collect()
}

/// One narrow step's line over the real rows: `A [rows, k]`, `B [n, k]`, the GPU's `[rows, n]`.
fn narrow_line(step: usize, g: &GemmBf16, rows: usize, a: &[u16], b: &[u16], c: &[u16]) -> String {
    let (n, k) = (g.n as usize, g.k as usize);
    let (a, b) = (widen(a), widen(b));
    let (mut worst, mut at, mut sum, mut bad, mut nonfinite) = (0.0f64, (0, 0), 0.0f64, 0, 0);
    for i in 0..rows {
        for j in 0..n {
            let (mut dot, mut abs) = (0.0f64, 0.0f64);
            for l in 0..k {
                let p = a[i * k + l] * b[j * k + l];
                dot += p;
                abs += p.abs();
            }
            let gpu = f64::from(bf16::from_bits(c[i * n + j]).to_f32());
            if !gpu.is_finite() {
                nonfinite += 1;
                bad += 1;
                continue;
            }
            let (err, bound) = ((gpu - dot).abs(), dot.abs() / 128.0 + abs / 16384.0);
            sum += err;
            bad += usize::from(err > bound);
            let ratio = err / bound.max(f64::MIN_POSITIVE);
            if ratio > worst {
                (worst, at) = (ratio, (i, j));
            }
        }
    }
    format!(
        "narrow step {step} m={} n={n} k={k}: max_err_over_bound={worst:.4} mean_abs_err={:.6e} \
         worst=(row {}, col {}) out_of_bound={bad} nonfinite={nonfinite} ({rows} real rows)",
        g.m,
        sum / (rows * n) as f64,
        at.0,
        at.1
    )
}

fn dump(name: &str, bits: &[u16]) {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("narrow_diag_9b");
    std::fs::create_dir_all(&dir).expect("create the dump directory");
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    std::fs::write(dir.join(format!("{name}.bin")), bytes).expect("write a dump");
}

#[test]
fn every_narrow_gemm_and_gdn_output_of_prompt_6() {
    let device = CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"));
    let (checkpoint, ids) = prompt();
    let rows = ids.len();
    let weights = load_weights(&device, &checkpoint, CUDA_TAPES, CUDA_WEIGHTS)
        .unwrap_or_else(|e| panic!("load_weights: {e}"));
    let full = (CUDA_TAPES.iter())
        .find(|t| t.bucket_m == M)
        .expect("the m=512 tape");
    let captures: Vec<Capture> = (full.steps.iter().enumerate())
        .filter_map(|(i, s)| match *s {
            CudaStep::GemmNarrow(g) => Some(Capture::Narrow(i, g)),
            CudaStep::GatedDeltaNet(g) => Some(Capture::Gdn(i, g)),
            _ => None,
        })
        .collect();
    eprintln!(
        "prompt {PROMPT}: {rows} real rows on the m={M} tape; {} narrow GEMM and {} GDN steps",
        (captures.iter())
            .filter(|c| matches!(c, Capture::Narrow(..)))
            .count(),
        (captures.iter())
            .filter(|c| matches!(c, Capture::Gdn(..)))
            .count()
    );
    let mut padded = vec![0u32; M as usize];
    padded[..rows].copy_from_slice(&ids);
    let positions: Vec<u32> = (0..M).collect();
    let kv = KvSequence {
        block_table: SEAT.0,
        prefix_len: 0,
        q_len: rows as u32,
    };
    let gdn = GdnSequence {
        slot: SEAT.1,
        q_len: rows as u32,
        fresh: true,
    };
    for capture in captures {
        let (last, out, width) = match capture {
            Capture::Narrow(i, g) => (i, g.c, g.n),
            Capture::Gdn(i, g) => (i, g.out, g.num_v_heads * g.head_v),
        };
        let span = ArenaSpan {
            offset: out,
            bytes: u64::from(M) * u64::from(width) * 2,
        };
        let tape: &'static [CudaTape] = Box::leak(Box::new([cut(full, last, span)]));
        let mut worker = CudaWorker::new(&device, tape, &weights, Some(KV_POOL), Some(GDN_POOL))
            .unwrap_or_else(|e| panic!("CudaWorker::new for step {last}: {e}"));
        let mut result = vec![0u16; span.bytes as usize / 2];
        worker
            .forward(
                0,
                Input::TokenIds(&padded),
                &positions,
                Some(&kv),
                Some(&gdn),
                &mut result,
            )
            .unwrap_or_else(|e| panic!("forward to step {last}: {e}"));
        let real = &result[..rows * width as usize];
        match capture {
            Capture::Narrow(i, g) => {
                let a_span = ArenaSpan {
                    offset: g.a,
                    bytes: u64::from(M) * u64::from(g.k) * 2,
                };
                let mut a = vec![0u16; a_span.bytes as usize / 2];
                worker.read_arena(a_span, &mut a).expect("read A");
                let b: Vec<u16> = weights[g.b.0 as usize].download().expect("download B");
                let b = &b[..(g.n * g.k) as usize];
                eprintln!(
                    "{}",
                    narrow_line(i, &g, rows, &a[..rows * g.k as usize], b, real)
                );
                dump(&format!("narrow_{i:04}"), real);
            }
            Capture::Gdn(i, g) => {
                let nonfinite = (real.iter())
                    .filter(|&&x| !bf16::from_bits(x).to_f32().is_finite())
                    .count();
                eprintln!(
                    "gdn step {i} layer {}: [{rows}, {width}] dumped, nonfinite={nonfinite}",
                    g.layer
                );
                dump(&format!("gdn_{i:04}_layer{:02}", g.layer), real);
            }
        }
    }
}

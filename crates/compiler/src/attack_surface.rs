// SPDX-License-Identifier: Apache-2.0
//! cuBLAS attack-surface analysis for the cuBLAS-freedom workstream.
//!
//! Inspects every `Cublas`-emitting GEMM step the solver produced
//! (`NormalizedStep` with kind `"Gemm"`) and answers two structured
//! questions per pick:
//!
//! 1. **Margin** — by how much is cuBLAS faster than the best
//!    non-cuBLAS standalone-GEMM kernel calibrated at this exact
//!    `(M, N, K)`? Bucketed into:
//!    a) ≤ 0.25%   d) ≤ 1.00%   g) >  5.00%
//!    b) ≤ 0.50%   e) ≤ 2.00%   no_alt   (no cutlass row)
//!    c) ≤ 0.75%   f) ≤ 5.00%   no_csv_data (neither side calibrated)
//!
//! 2. **Fusion gap** — what fusion claim, if it existed, would have
//!    absorbed this Cublas pick? Walks the bucket's [`NormalizedStep`]
//!    stream forward/backward from the GEMM to identify
//!    fusion-eligible neighbors:
//!    Gemm→Add   — `CutlassGemmAdd` peer exists; missing
//!    cuBLAS-side variant means the DP can't pick a fused cuBLAS
//!    path when cutlass loses the bucket.
//!    Norm→Gemm  — no fusion exists today (would be
//!    `CutlassFusedRmsNormGemm` / `CutlassFusedLayerNormGemm`).
//!    Gemm→ScalarMul — no fusion; ScalarMul is BW-bound and would
//!    fold into a GEMM epilogue.
//!    lm_head    — the chain `Norm → Gemm[→ScalarMul]` at the
//!    final lm_head; no fusion exists. lm_head's huge
//!    `(vocab, hidden)` weight makes this dominate the
//!    `no_csv_data` margin class.
//!
//! Counts are dedup'd by `(arch, layer, N, K, grid_m)` to surface
//! LOGICAL pick decisions, not M-bucket multiplication.
//!
//! All inputs are runtime-accessible: the per-arch
//! `Vec<BucketDump>` from `BackboneDumpRegistration::dump_all`, plus
//! the raw cost CSV string from `scratchy_target_cuda::targets::ProfileDef`.
//! No GPU, no FUF reconstruction — `NormalizedStep` already encodes
//! everything we need.
//!
//! Print format: a global table by margin class, a fusion-gap
//! breakdown, and a per-arch tail when requested. See
//! [`AttackSurfaceReport::print`].

use std::collections::HashMap;
use std::collections::HashSet;
use std::io::{self, Write};

use crate::{BucketDump, NormalizedField, NormalizedStep};

/// Workload-grid M values the solver evaluates costs at. One of
/// these falls inside every `m_min..m_max_excl` bucket; we read CSV
/// rows at that grid M.
const WORKLOAD_GRID: &[u32] = &[1, 8, 64, 512, 4096];

/// Margin-bucket boundaries — the first one whose ceiling the gap
/// fits under wins. `(label, ceiling_fraction)`. `ceiling_fraction`
/// is `(cublas - best_alt) / cublas`; positive = cuBLAS faster.
const MARGIN_BUCKETS: &[(&str, f64)] = &[
    ("a_le_0.25", 0.0025),
    ("b_le_0.50", 0.0050),
    ("c_le_0.75", 0.0075),
    ("d_le_1.00", 0.0100),
    ("e_le_2.00", 0.0200),
    ("f_le_5.00", 0.0500),
    ("g_gt_5.00", f64::INFINITY),
];

const NO_ALT: &str = "no_alt";
const NO_CSV: &str = "no_csv_data";

/// Fusion-gap reasons. `&'static str` so they share Eq/Hash with
/// `&'static str` map keys cheaply.
pub mod fusion_gap {
    pub const GEMM_ADD: &str = "Gemm→Add (CutlassGemmAdd peer; no cuBLAS peer)";
    pub const NORM_GEMM: &str = "Norm→Gemm (no fusion exists)";
    pub const GEMM_SCALARMUL: &str = "Gemm→ScalarMul (no fusion exists)";
    pub const LM_HEAD: &str = "lm_head: Norm→Gemm[→ScalarMul] (no fusion)";
}

// ── CSV cost table ───────────────────────────────────────────────

/// Runtime-parseable cost table. Stores exact `(kernel, M, N, K)`
/// rows from `scratchy_target_cuda::targets::ProfileDef::cost_csv`. The
/// proc-macro builds a fancier predictor (linreg extrapolation) for
/// codegen-time cost lookup; here we want HONEST exact-row data so
/// `no_csv_data` is a separable class — extrapolation would lie
/// about the gap.
#[derive(Default)]
pub struct CostTable {
    rows: HashMap<(String, u32, u32, u32), f64>,
}

impl CostTable {
    /// Parse a `kernel,M,N,K,us` CSV body. Lines starting with `#`
    /// or the literal `kernel,` header row are skipped.
    pub fn parse(csv: &str) -> Self {
        let mut rows = HashMap::new();
        for line in csv.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("kernel,") {
                continue;
            }
            let mut it = line.split(',');
            let kernel = it.next();
            let m = it.next().and_then(|s| s.parse::<u32>().ok());
            let n = it.next().and_then(|s| s.parse::<u32>().ok());
            let k = it.next().and_then(|s| s.parse::<u32>().ok());
            let us = it.next().and_then(|s| s.parse::<f64>().ok());
            if let (Some(kernel), Some(m), Some(n), Some(k), Some(us)) = (kernel, m, n, k, us) {
                rows.insert((kernel.to_string(), m, n, k), us);
            }
        }
        Self { rows }
    }

    pub fn cost_us(&self, kernel: &str, m: u32, n: u32, k: u32) -> Option<f64> {
        self.rows.get(&(kernel.to_string(), m, n, k)).copied()
    }

    /// Min-cost non-cuBLAS row at `(m, n, k)` over the standalone
    /// CUTLASS-GEMM family — explicitly excluding `*_add`, `*_bias`,
    /// and EVT (`*_silu_mul`) kernels whose claim shape (Gemm+Add,
    /// Gemm+Bias, Gemm+Silu+Mul) differs from a standalone Gemm
    /// pick. Returning their costs would give a misleading "alt"
    /// the DP can't actually pick at a standalone-Gemm site.
    pub fn best_non_cublas(&self, m: u32, n: u32, k: u32) -> Option<(f64, &str)> {
        let mut best: Option<(f64, &str)> = None;
        for ((kernel, mm, nn, kk), us) in &self.rows {
            if *mm != m || *nn != n || *kk != k {
                continue;
            }
            if kernel == "cublas" {
                continue;
            }
            if !kernel.starts_with("cutlass_") {
                continue;
            }
            if kernel.ends_with("_add") || kernel.contains("_silu_mul") || kernel.contains("_bias_")
            {
                continue;
            }
            match best {
                None => best = Some((*us, kernel.as_str())),
                Some((cur_us, _)) if *us < cur_us => best = Some((*us, kernel.as_str())),
                _ => {}
            }
        }
        best
    }
}

// ── Report ───────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct PickSample {
    pub arch: String,
    pub layer: u32,
    pub grid_m: u32,
    pub n: u32,
    pub k: u32,
    pub cublas_us: Option<f64>,
    pub alt_us: Option<f64>,
    pub alt_kernel: Option<String>,
}

#[derive(Default)]
pub struct AttackSurfaceReport {
    /// Distinct `(arch, layer, n, k, grid_m)` pick keys we've already
    /// counted — keeps us from double-counting a pick that appears in
    /// both a backbone-loop body and the lm_head section, etc.
    seen: HashSet<(String, u32, u32, u32, u32)>,
    /// `margin_class → count`.
    pub by_margin: HashMap<&'static str, u64>,
    /// `arch → margin_class → count`.
    pub by_arch: HashMap<String, HashMap<&'static str, u64>>,
    /// `arch → total picks` (sum across margin classes).
    pub by_arch_total: HashMap<String, u64>,
    /// `fusion_gap_reason → count`. Each pick is in at most one reason.
    pub by_fusion_gap: HashMap<&'static str, u64>,
    /// `fusion_gap_reason → set of distinct (n, k) shapes` — proxy
    /// for kernel-authoring work the fusion would need to cover.
    pub fusion_gap_shapes: HashMap<&'static str, HashSet<(u32, u32)>>,
    /// `(fusion_gap_reason, margin_class) → count`. Cross-tab so
    /// readers can see e.g. "all 1095 lm_head picks are in
    /// no_csv_data" — those resolve in one shot from the fusion side
    /// without any cost-table calibration work.
    pub xtab: HashMap<(&'static str, &'static str), u64>,
    /// `shape_regime → count` for picks in the `g_gt_5.00` class.
    /// Disambiguates the >5% losses: which are tile-family gaps
    /// (Stream-K territory, SIMT-small-M territory) vs genuinely
    /// hard for the existing zoo. `gpu_cost_sweep` calibrates EVERY
    /// CUTLASS tile per swept shape, so >5% picks are not a sweep
    /// coverage problem (that lands in `no_csv_data` instead).
    pub g_by_regime: HashMap<&'static str, u64>,
    /// `shape_regime → up to N samples`.
    pub samples_g_regime: HashMap<&'static str, Vec<PickSample>>,
    /// Picks that land in BOTH a fusion-gap reason AND a tile-family
    /// regime (i.e. the >5% class). Used for the cumulative-fusion
    /// projection's overlap correction: applying both
    /// fusion-authoring AND a tile-family kernel removes each such
    /// pick once, not twice. Indexed by regime so we can credit the
    /// kernel-family work appropriately.
    pub overlap_by_regime: HashMap<&'static str, u64>,
    /// `margin_class → up to N samples`.
    pub samples_margin: HashMap<&'static str, Vec<PickSample>>,
    /// `fusion_gap_reason → up to N samples`.
    pub samples_fusion_gap: HashMap<&'static str, Vec<PickSample>>,
    pub max_samples_per_class: usize,
}

impl AttackSurfaceReport {
    pub fn new(max_samples_per_class: usize) -> Self {
        Self {
            max_samples_per_class,
            ..Default::default()
        }
    }

    /// Walk one bucket's `(backbone, lm_head)` step streams and
    /// classify every Gemm pick. `is_lm_head` is set when the bucket
    /// is the final lm_head section so a single Gemm-after-Norm there
    /// is flagged as the lm_head fusion gap (which subsumes
    /// `Norm→Gemm` for the purposes of effort-prioritization).
    pub fn analyze_bucket(&mut self, arch: &str, bucket: &BucketDump, cost_table: &CostTable) {
        let grid_m = grid_m_for(bucket.m_min, bucket.m_max_excl);
        self.scan_steps(arch, &bucket.backbone, false, grid_m, cost_table);
        self.scan_steps(arch, &bucket.lm_head, true, grid_m, cost_table);
    }

    fn scan_steps(
        &mut self,
        arch: &str,
        steps: &[NormalizedStep],
        is_lm_head: bool,
        grid_m: u32,
        cost_table: &CostTable,
    ) {
        for (i, step) in steps.iter().enumerate() {
            if step.kind != "Gemm" {
                continue;
            }
            // Extract `(n, k)` weight shape and the in/out slot pair.
            // Convention from `Instruction::Gemm` normalize(): fields
            // are [Slot(in), Slot(out), Layer, LayerKind, WeightShape].
            let mut slots: Vec<u32> = Vec::new();
            let mut layer: Option<u32> = None;
            let mut weight_shape: Option<(u32, u32)> = None;
            for f in &step.fields {
                match f {
                    NormalizedField::Slot(s) => slots.push(*s),
                    NormalizedField::Layer(l) => layer = Some(*l),
                    NormalizedField::WeightShape { n, k } => weight_shape = Some((*n, *k)),
                    _ => {}
                }
            }
            let (n, k) = match weight_shape {
                Some(p) => p,
                None => continue,
            };
            let layer = layer.unwrap_or(0);
            let key = (arch.to_string(), layer, n, k, grid_m);
            if !self.seen.insert(key) {
                continue;
            }

            // Margin classification.
            let cb = cost_table.cost_us("cublas", grid_m, n, k);
            let alt = cost_table.best_non_cublas(grid_m, n, k);
            let cls: &'static str = match (cb, alt) {
                (Some(cb_us), Some((alt_us, _))) => margin_class(cb_us, alt_us),
                (Some(_), None) => NO_ALT,
                (None, _) => NO_CSV,
            };
            *self.by_margin.entry(cls).or_default() += 1;
            *self
                .by_arch
                .entry(arch.to_string())
                .or_default()
                .entry(cls)
                .or_default() += 1;
            *self.by_arch_total.entry(arch.to_string()).or_default() += 1;

            if cls == "g_gt_5.00" {
                let regime = classify_regime(grid_m, n, k);
                *self.g_by_regime.entry(regime).or_default() += 1;
                let regime_sample = PickSample {
                    arch: arch.to_string(),
                    layer,
                    grid_m,
                    n,
                    k,
                    cublas_us: cb,
                    alt_us: alt.map(|(u, _)| u),
                    alt_kernel: alt.map(|(_, k)| k.to_string()),
                };
                self.samples_g_regime
                    .entry(regime)
                    .or_default()
                    .push_if_under(self.max_samples_per_class, regime_sample);
            }

            let sample = PickSample {
                arch: arch.to_string(),
                layer,
                grid_m,
                n,
                k,
                cublas_us: cb,
                alt_us: alt.map(|(u, _)| u),
                alt_kernel: alt.map(|(_, k)| k.to_string()),
            };
            self.samples_margin
                .entry(cls)
                .or_default()
                .push_if_under(self.max_samples_per_class, sample.clone());

            // Fusion-gap classification — uses neighbor walks in the
            // step stream.
            let in_slot = slots.first().copied();
            let out_slot = slots.get(1).copied();
            let prev_kind = in_slot.and_then(|s| producer_kind(steps, i, s));
            let next_kind = out_slot.and_then(|s| consumer_kind(steps, i, s));
            let gap = classify_fusion_gap(prev_kind, next_kind, is_lm_head);
            if let Some(reason) = gap {
                *self.by_fusion_gap.entry(reason).or_default() += 1;
                *self.xtab.entry((reason, cls)).or_default() += 1;
                self.fusion_gap_shapes
                    .entry(reason)
                    .or_default()
                    .insert((n, k));
                self.samples_fusion_gap
                    .entry(reason)
                    .or_default()
                    .push_if_under(self.max_samples_per_class, sample);
            }

            // Fusion ∩ regime overlap: picks that BOTH a fusion claim
            // would absorb AND a tile-family kernel would close.
            // Without tracking this, a naive projection of
            // (residual − fusion_count − regime_count) double-subtracts
            // these. Recorded only for >5% picks (where regime is
            // assigned).
            if cls == "g_gt_5.00" && gap.is_some() {
                let regime = classify_regime(grid_m, n, k);
                *self.overlap_by_regime.entry(regime).or_default() += 1;
            }
        }
    }

    /// Render the report to `out`. `per_arch` opens an additional
    /// per-arch breakdown after the main report.
    pub fn print<W: Write>(&self, out: &mut W, per_arch: bool) -> io::Result<()> {
        let total: u64 = self.by_margin.values().sum();
        writeln!(out)?;
        writeln!(
            out,
            "═══ cuBLAS attack surface ─ {} distinct (arch, layer, N, K, M-grid) picks",
            total
        )?;

        let cls_order: Vec<&'static str> = MARGIN_BUCKETS
            .iter()
            .map(|(k, _)| *k)
            .chain([NO_ALT, NO_CSV])
            .collect();

        // ── Margin breakdown ────────────────────────────────────────
        writeln!(out)?;
        writeln!(out, "Margin (cuBLAS faster than best non-cuBLAS by …)")?;
        writeln!(out, "  class             count    %    note")?;
        let margin_notes: HashMap<&'static str, &'static str> = [
            ("a_le_0.25", "would flip with a cost-eval ε bias"),
            ("d_le_1.00", "could flip with new tile-zoo entries"),
            ("g_gt_5.00", "cuBLAS truly faster — needs new kernel"),
            (NO_CSV, "neither side calibrated (mostly lm_head)"),
        ]
        .into_iter()
        .collect();
        for cls in &cls_order {
            let c = self.by_margin.get(*cls).copied().unwrap_or(0);
            if c == 0 {
                continue;
            }
            let pct = pct_of(c, total);
            let note = margin_notes.get(*cls).copied().unwrap_or("");
            writeln!(out, "  {:<14}  {:>6}  {:>4.1}%   {}", cls, c, pct, note)?;
        }

        // ── Fusion-gap rollup with shape-distinct counts ───────────
        let fg_total: u64 = self.by_fusion_gap.values().sum();
        // Stable order: largest absorption first so the dominant
        // opportunity is at the top.
        let fg_order_raw = [
            fusion_gap::LM_HEAD,
            fusion_gap::NORM_GEMM,
            fusion_gap::GEMM_SCALARMUL,
            fusion_gap::GEMM_ADD,
        ];
        let fg_order: Vec<&'static str> = fg_order_raw
            .into_iter()
            .filter(|r| self.by_fusion_gap.get(*r).copied().unwrap_or(0) > 0)
            .collect();

        if fg_total > 0 {
            writeln!(out)?;
            writeln!(
                out,
                "Fusion-gap — picks absorbable by adding a fusion claim"
            )?;
            writeln!(
                out,
                "  count   shapes  reason  ({} of {} picks = {:.1}%)",
                fg_total,
                total,
                pct_of(fg_total, total)
            )?;
            for reason in &fg_order {
                let c = self.by_fusion_gap.get(*reason).copied().unwrap_or(0);
                let shapes = self
                    .fusion_gap_shapes
                    .get(*reason)
                    .map(|s| s.len() as u64)
                    .unwrap_or(0);
                writeln!(out, "  {:>5}   {:>6}  {}", c, shapes, reason)?;
            }

            // Cross-tab: fusion-gap × margin class. Surfaces double
            // wins (lm_head picks that ALSO live in no_csv_data — one
            // fusion drops them, no calibration needed).
            writeln!(out)?;
            writeln!(out, "Fusion-gap × margin (where the absorbable picks live)")?;
            // Pick the margin classes with non-trivial counts to show.
            let xtab_cls: Vec<&'static str> = cls_order
                .iter()
                .copied()
                .filter(|cls| {
                    fg_order
                        .iter()
                        .any(|r| self.xtab.get(&(r, cls)).copied().unwrap_or(0) > 0)
                })
                .collect();
            if !xtab_cls.is_empty() {
                let header_cells: Vec<String> = xtab_cls
                    .iter()
                    .map(|c| short_margin_label(c).to_string())
                    .collect();
                writeln!(
                    out,
                    "  {:<28}  {}",
                    "reason",
                    header_cells
                        .iter()
                        .map(|s| format!("{:>10}", s))
                        .collect::<Vec<_>>()
                        .join(" ")
                )?;
                for reason in &fg_order {
                    let cells: Vec<String> = xtab_cls
                        .iter()
                        .map(|cls| self.xtab.get(&(reason, cls)).copied().unwrap_or(0))
                        .map(|v| {
                            if v == 0 {
                                format!("{:>10}", "·")
                            } else {
                                format!("{:>10}", v)
                            }
                        })
                        .collect();
                    writeln!(
                        out,
                        "  {:<28}  {}",
                        short_reason_label(reason),
                        cells.join(" ")
                    )?;
                }
            }

            // Cumulative-fusion projection: assume each fusion lands
            // in declared order and count residual picks. Then layer
            // tile-family additions (Stream-K, etc.) for the remaining
            // >5% picks, subtracting only the picks NOT already
            // absorbed by a fusion (overlap_by_regime tracks the
            // double-subtractions).
            writeln!(out)?;
            writeln!(
                out,
                "Cumulative landing projection (fusions, then kernel families)"
            )?;
            let mut residual = total;
            writeln!(
                out,
                "  starting residual            {:>6}  (current cuBLAS surface)",
                residual
            )?;
            for reason in &fg_order {
                let c = self.by_fusion_gap.get(*reason).copied().unwrap_or(0);
                residual = residual.saturating_sub(c);
                writeln!(
                    out,
                    "  +{:<27}  {:>6}  (-{})",
                    short_reason_tag(reason),
                    residual,
                    c
                )?;
            }
            // Then tile-family additions on the remaining >5% picks.
            // For each regime, count = g_by_regime[regime] minus the
            // overlap (already subtracted by the fusion landings).
            let stream_k_regimes = [
                "small-M long-K (Stream-K target)",
                "mid-M long-K (Stream-K target)",
            ];
            let stream_k_total: u64 = stream_k_regimes
                .iter()
                .map(|r| self.g_by_regime.get(*r).copied().unwrap_or(0))
                .sum();
            let stream_k_overlap: u64 = stream_k_regimes
                .iter()
                .map(|r| self.overlap_by_regime.get(*r).copied().unwrap_or(0))
                .sum();
            let stream_k_unique = stream_k_total.saturating_sub(stream_k_overlap);
            if stream_k_unique > 0 {
                residual = residual.saturating_sub(stream_k_unique);
                writeln!(
                    out,
                    "  +{:<27}  {:>6}  (-{} unique; {} were also fusion-absorbable)",
                    "StreamK (small/mid-M long-K)", residual, stream_k_unique, stream_k_overlap
                )?;
            }
            let simt_total = self
                .g_by_regime
                .get("small-M short-K (SIMT/tile_m=8 target)")
                .copied()
                .unwrap_or(0);
            let simt_overlap = self
                .overlap_by_regime
                .get("small-M short-K (SIMT/tile_m=8 target)")
                .copied()
                .unwrap_or(0);
            let simt_unique = simt_total.saturating_sub(simt_overlap);
            if simt_unique > 0 {
                residual = residual.saturating_sub(simt_unique);
                writeln!(
                    out,
                    "  +{:<27}  {:>6}  (-{} unique; {} were also fusion-absorbable)",
                    "SIMT (small-M short-K)", residual, simt_unique, simt_overlap
                )?;
            }
            writeln!(
                out,
                "  remaining {} picks: large-M compute-bound + mid-M short-K — accept under feature-gate, or hand-CUDA per-arch.",
                residual
            )?;
        }

        // ── g_gt_5.00 by shape regime ──────────────────────────────
        //
        // Disambiguates the >5% class: picks where cuBLAS beats the
        // best CUTLASS tile by >5%. NOT a sweep coverage gap (those
        // land in `no_csv_data`); these have all 44 cutlass tiles
        // measured and cuBLAS still wins. The regime tag identifies
        // which tile family (Stream-K, SIMT-small-M, etc.) would
        // close the gap, vs which picks are genuinely beyond the
        // current zoo's reach.
        let g_total: u64 = self.g_by_regime.values().sum();
        if g_total > 0 {
            writeln!(out)?;
            writeln!(
                out,
                ">5% picks ({}) by shape regime — *which* tile family would close each",
                g_total
            )?;
            let regime_order = [
                "M=1",
                "small-M long-K (Stream-K target)",
                "small-M short-K (SIMT/tile_m=8 target)",
                "mid-M long-K (Stream-K target)",
                "mid-M short-K",
                "large-M (compute-bound)",
            ];
            for regime in regime_order {
                let c = self.g_by_regime.get(regime).copied().unwrap_or(0);
                if c == 0 {
                    continue;
                }
                let pct = pct_of(c, g_total);
                writeln!(out, "  {:>5}  {:>4.0}%   {}", c, pct, regime)?;
                if let Some(samples) = self.samples_g_regime.get(regime)
                    && let Some(s) = samples.first()
                {
                    let cb = s.cublas_us.map(|v| format!("{:.1}", v)).unwrap_or_default();
                    let alt = s.alt_us.map(|v| format!("{:.1}", v)).unwrap_or_default();
                    let kn = s.alt_kernel.as_deref().unwrap_or("?");
                    writeln!(
                        out,
                        "             e.g. {} M={} N={} K={}  cb={}µs alt={}µs ({})",
                        short_arch(&s.arch),
                        s.grid_m,
                        s.n,
                        s.k,
                        cb,
                        alt,
                        kn
                    )?;
                }
            }
        }

        // ── Per-arch-family rollup ─────────────────────────────────
        //
        // The variant label is `<family> / <variant_stem> · tp=N` —
        // collapse on `<family>` to surface the dominant arch
        // families without drowning the report in per-variant
        // proliferation (gemma3-27b-it / -fp8 / -ct-int4 / etc are
        // all the same kernel-shape problem).
        let mut by_family: HashMap<String, u64> = HashMap::new();
        for (label, count) in &self.by_arch_total {
            let family = label.split('/').next().unwrap_or(label).trim().to_string();
            *by_family.entry(family).or_default() += count;
        }
        writeln!(out)?;
        writeln!(out, "cuBLAS picks by arch family")?;
        let mut families: Vec<(String, u64)> = by_family.into_iter().collect();
        families.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        for (family, count) in &families {
            let pct = pct_of(*count, total);
            writeln!(out, "  {:>5}  {:>5.1}%   {}", count, pct, family)?;
        }

        // ── Sample picks (compact, one per class) ──────────────────
        writeln!(out)?;
        writeln!(out, "Sample picks per margin class (one each)")?;
        for cls in &cls_order {
            let samples = match self.samples_margin.get(*cls) {
                Some(v) if !v.is_empty() => v,
                _ => continue,
            };
            let s = &samples[0];
            let arch_short = short_arch(&s.arch);
            let cb_str = match s.cublas_us {
                Some(v) => format!("{:>7.1}", v),
                None => "      ?".to_string(),
            };
            let alt_str = match s.alt_us {
                Some(v) => format!("{:>7.1}", v),
                None => "      ?".to_string(),
            };
            let alt_name = s.alt_kernel.as_deref().unwrap_or("(none)");
            writeln!(
                out,
                "  [{:<11}] {:<32} L={:<2} M={:<4} N={:<6} K={:<6} cb={} alt={} ({})",
                cls, arch_short, s.layer, s.grid_m, s.n, s.k, cb_str, alt_str, alt_name,
            )?;
        }

        if per_arch {
            writeln!(out)?;
            writeln!(out, "Per-arch breakdown")?;
            let mut arches: Vec<(&String, &HashMap<&'static str, u64>)> =
                self.by_arch.iter().collect();
            arches.sort_by(|a, b| {
                let sa: u64 = a.1.values().sum();
                let sb: u64 = b.1.values().sum();
                sb.cmp(&sa).then(a.0.cmp(b.0))
            });
            for (arch, classes) in arches {
                let total: u64 = classes.values().sum();
                if total == 0 {
                    continue;
                }
                writeln!(out)?;
                writeln!(out, "  {} ({} picks)", arch, total)?;
                for cls in &cls_order {
                    let c = classes.get(*cls).copied().unwrap_or(0);
                    if c == 0 {
                        continue;
                    }
                    writeln!(out, "    {:<14} {:>5}", cls, c)?;
                }
            }
        }

        Ok(())
    }
}

fn pct_of(num: u64, denom: u64) -> f64 {
    if denom == 0 {
        0.0
    } else {
        100.0 * (num as f64) / (denom as f64)
    }
}

fn short_margin_label(cls: &str) -> &str {
    match cls {
        "a_le_0.25" => "≤0.25%",
        "b_le_0.50" => "≤0.5%",
        "c_le_0.75" => "≤0.75%",
        "d_le_1.00" => "≤1%",
        "e_le_2.00" => "≤2%",
        "f_le_5.00" => "≤5%",
        "g_gt_5.00" => ">5%",
        NO_ALT => "no_alt",
        NO_CSV => "no_data",
        other => other,
    }
}

fn short_reason_label(reason: &str) -> &str {
    if reason == fusion_gap::LM_HEAD {
        "lm_head: Norm→Gemm[→ScalarMul]"
    } else if reason == fusion_gap::NORM_GEMM {
        "Norm→Gemm"
    } else if reason == fusion_gap::GEMM_SCALARMUL {
        "Gemm→ScalarMul"
    } else if reason == fusion_gap::GEMM_ADD {
        "Gemm→Add"
    } else {
        reason
    }
}

fn short_reason_tag(reason: &str) -> &str {
    if reason == fusion_gap::LM_HEAD {
        "FusedLmHead"
    } else if reason == fusion_gap::NORM_GEMM {
        "FusedNormGemm"
    } else if reason == fusion_gap::GEMM_SCALARMUL {
        "FusedGemmScalarMul"
    } else if reason == fusion_gap::GEMM_ADD {
        "FusedGemmAdd-cublas-peer"
    } else {
        reason
    }
}

// ── Helpers ──────────────────────────────────────────────────────

fn grid_m_for(m_min: u64, m_max_excl: u64) -> u32 {
    for &g in WORKLOAD_GRID {
        if (g as u64) >= m_min && (g as u64) < m_max_excl {
            return g;
        }
    }
    // Bucket smaller than any grid step (rare); fall back to the
    // bucket's lo so the lookup still has a chance.
    m_min.min(u32::MAX as u64) as u32
}

/// Bucket a `(M, N, K)` shape into a kernel-regime tag. Used to
/// disambiguate the `g_gt_5.00` class — picks where cuBLAS beats
/// every CUTLASS tile in the zoo by >5%. The regime tells us which
/// kernel family (if any) would close the gap:
///
///   M=1                — decode / lm_head; CutlassGemv handles it
///                        unless N is huge (vocab projection).
///   small-M long-K     — M ≤ 16, K ≥ 8192. Stream-K's sweet spot.
///   small-M other      — M ≤ 16, K < 8192. Tall-skinny needs
///                        SIMT-style kernels (tile_m=8).
///   mid-M long-K       — M ∈ [32, 512], K ≥ 8192. Stream-K helps.
///   mid-M other        — M ∈ [32, 512], K < 8192.
///   large-M            — M ≥ 1024. Compute-bound; tile zoo is
///                        already well-tuned here.
fn classify_regime(m: u32, _n: u32, k: u32) -> &'static str {
    if m == 1 {
        "M=1"
    } else if m <= 16 && k >= 8192 {
        "small-M long-K (Stream-K target)"
    } else if m <= 16 {
        "small-M short-K (SIMT/tile_m=8 target)"
    } else if m <= 512 && k >= 8192 {
        "mid-M long-K (Stream-K target)"
    } else if m <= 512 {
        "mid-M short-K"
    } else {
        "large-M (compute-bound)"
    }
}

fn margin_class(cublas_us: f64, alt_us: f64) -> &'static str {
    let gap = (alt_us - cublas_us) / cublas_us; // > 0 ⇒ cuBLAS faster
    for (label, ceiling) in MARGIN_BUCKETS {
        if gap <= *ceiling {
            return label;
        }
    }
    MARGIN_BUCKETS.last().expect("non-empty").0
}

/// Find the next step (after `from_idx`) that reads `slot`, return
/// its kind. None if nothing reads it before end-of-stream — typical
/// for the lm_head Gemm whose output flows into ScalarMul or
/// terminates the bucket entirely.
fn consumer_kind(steps: &[NormalizedStep], from_idx: usize, slot: u32) -> Option<&str> {
    for s in steps.iter().skip(from_idx + 1) {
        // Skip Loop markers — they're structural, not real ops.
        if s.kind == "Loop" {
            continue;
        }
        for f in &s.fields {
            if let NormalizedField::Slot(v) = f
                && *v == slot
            {
                return Some(s.kind);
            }
        }
    }
    None
}

/// Find the previous step (before `from_idx`) that wrote `slot` —
/// i.e. has `slot` as one of its slot fields. Returns its kind.
fn producer_kind(steps: &[NormalizedStep], from_idx: usize, slot: u32) -> Option<&str> {
    for i in (0..from_idx).rev() {
        let s = &steps[i];
        if s.kind == "Loop" {
            continue;
        }
        for f in &s.fields {
            if let NormalizedField::Slot(v) = f
                && *v == slot
            {
                return Some(s.kind);
            }
        }
    }
    None
}

fn classify_fusion_gap(
    prev: Option<&str>,
    next: Option<&str>,
    is_lm_head: bool,
) -> Option<&'static str> {
    // lm_head section: any Gemm here represents the projection to
    // vocab. The chain `Norm→Gemm[→ScalarMul]` is canonical for
    // every arch. Tag distinctly because the lm_head Gemm dominates
    // the no_csv_data class (vocab-N rows uncalibrated) and is its
    // own fusion-authoring opportunity.
    if is_lm_head {
        return Some(fusion_gap::LM_HEAD);
    }
    if matches!(next, Some("Add")) {
        return Some(fusion_gap::GEMM_ADD);
    }
    if matches!(next, Some("ScalarMul")) {
        return Some(fusion_gap::GEMM_SCALARMUL);
    }
    if matches!(
        prev,
        Some("RmsNorm")
            | Some("LayerNorm")
            | Some("FusedAddRmsNorm")
            | Some("FusedAddRmsNormWithOffset")
            | Some("ScalarOffsetRmsNorm")
    ) {
        return Some(fusion_gap::NORM_GEMM);
    }
    None
}

fn short_arch(arch: &str) -> String {
    // `gemma2 / gemma2-27b · tp=1` → `gemma2-27b`.
    arch.split('/')
        .nth(1)
        .map(|s| s.split('·').next().unwrap_or(s).trim().to_string())
        .unwrap_or_else(|| arch.to_string())
}

trait PushIfUnder {
    fn push_if_under(&mut self, cap: usize, item: PickSample);
}

impl PushIfUnder for Vec<PickSample> {
    fn push_if_under(&mut self, cap: usize, item: PickSample) {
        if self.len() < cap {
            self.push(item);
        }
    }
}

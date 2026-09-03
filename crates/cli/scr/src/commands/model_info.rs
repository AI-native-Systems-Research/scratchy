// SPDX-License-Identifier: Apache-2.0
//! `scr model info` — print per-bucket backbones for compiled
//! scratchy variants.
//!
//! Walks `inventory::iter::<scratchy_forward_compiler::BackboneDumpRegistration>`,
//! invokes each arch's `dump_all` to materialize the per-variant
//! `Vec<BucketDump>`, optionally filters by AND-substring match
//! against `<arch>/<variant_stem>`, and prints. No GPU, no weight
//! loading.
//!
//! Within a variant, buckets whose `(backbone, lm_head)` step
//! streams have the same structural shape (op sequence + slots +
//! layer + kernel class, ignoring scalar consts) collapse into one
//! cluster with a shared bucket-list header.
//!
//! Columns are aligned to per-variant widths so kernel-class
//! annotations and layer indices line up across clusters. When
//! stdout is a tty, headers print in bold and trailing metadata in
//! a dim style — set `NO_COLOR` to opt out.

#![cfg(any(feature = "cuda", feature = "metal"))]

use std::io::{self, BufWriter, IsTerminal, Write};

use scratchy_forward_compiler::{
    BackboneDumpRegistration, BucketDump, NormalizedField, NormalizedStep,
};
// `attack_surface` builds against `scratchy_target_cuda::targets` for GPU
// profile selection — kept cuda-only. The `--cublas-analysis` flag
// is hidden on metal (see `args.rs`), so the surrounding body code
// never reaches these types under `not(feature = "cuda")`.
#[cfg(feature = "cuda")]
use scratchy_forward_compiler::attack_surface::{AttackSurfaceReport, CostTable};

use crate::args::{ColorWhen, ModelInfoArgs};

// Inventory submissions for `scratchy_models` are kept alive by
// `scratchy-serving-worker::gpu_worker`'s `extern crate scratchy_models as _;`,
// which the CLI binary transitively pulls in.

pub async fn run_info(args: ModelInfoArgs) -> anyhow::Result<()> {
    // Write through a locked, buffered stdout and propagate
    // `io::Result`s, so a downstream `head` / `less` closing the pipe
    // turns into a clean `BrokenPipe` we can swallow — instead of
    // `println!` panicking inside `_print`.
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    match run_info_inner(&mut out, &args) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        Err(e) => Err(e.into()),
    }
}

fn run_info_inner<W: Write>(out: &mut W, args: &ModelInfoArgs) -> io::Result<()> {
    let needles: Vec<String> = args.filters.iter().map(|s| s.to_lowercase()).collect();
    let style = Style::resolve(args.color);

    // Resolve the active GPU profile + parse its cost CSV ONCE before
    // the per-arch walk. The `attack_surface` aggregator threads it
    // through every bucket; the normal info dump ignores it.
    // `--cublas-analysis` is cuda-only — the field doesn't exist on
    // the metal flavor of `ModelInfoArgs`, so the surrounding code
    // can't even reference it under `not(feature = "cuda")`.
    #[cfg(feature = "cuda")]
    let cost_table = if args.cublas_analysis {
        let profile = scratchy_target_cuda::targets::detect()
            .map(|p| (p.name, p.cost_csv))
            .unwrap_or((
                scratchy_target_cuda::targets::L4_SM89.name,
                scratchy_target_cuda::targets::L4_SM89.cost_csv,
            ));
        Some((profile.0, CostTable::parse(profile.1)))
    } else {
        None
    };
    #[cfg(feature = "cuda")]
    let mut surface = AttackSurfaceReport::new(3);

    let mut shown = 0usize;
    for reg in scratchy_forward_compiler::inventory::iter::<BackboneDumpRegistration> {
        let arch = reg.arch_name;
        for variant in (reg.dump_all)() {
            // Include `tp=N` in the substring-match key so callers can
            // filter on it the same way they filter on arch/variant
            // (e.g. `scr model info qwen 2.5 3b tp=2`). At
            // --features cuda (no nccl) every variant is tp=1, so
            // including the token doesn't disturb existing filters.
            let key = format!(
                "{arch}/{}/tp={}",
                variant.variant_stem, variant.tp_world_size
            )
            .to_lowercase();
            if !needles.iter().all(|n| key.contains(n)) {
                continue;
            }
            // Backbone print is suppressed in --attack-surface mode so
            // the analysis report isn't buried under tens of thousands
            // of step lines. On metal the flag doesn't exist, so the
            // print path is always taken.
            #[cfg(feature = "cuda")]
            let should_print_backbone = !args.cublas_analysis;
            #[cfg(not(feature = "cuda"))]
            let should_print_backbone = true;
            if should_print_backbone {
                print_variant(
                    out,
                    arch,
                    variant.variant_stem,
                    variant.tp_world_size,
                    &variant.buckets,
                    &style,
                )?;
            }
            #[cfg(feature = "cuda")]
            if let Some((_, cost_table)) = &cost_table {
                let arch_label = format!(
                    "{arch} / {} · tp={}",
                    variant.variant_stem, variant.tp_world_size
                );
                for bucket in &variant.buckets {
                    surface.analyze_bucket(&arch_label, bucket, cost_table);
                }
            }
            shown += 1;
        }
    }

    if shown == 0 {
        if needles.is_empty() {
            writeln!(out, "(no scratchy variants registered)")?;
        } else {
            writeln!(
                out,
                "(no compiled scratchy variant matched: {:?})",
                args.filters
            )?;
        }
    }
    #[cfg(feature = "cuda")]
    if let Some((profile_name, _)) = &cost_table {
        writeln!(
            out,
            "\n(attack surface against profile `{profile_name}` — \
             override with SCRATCHY_GPU=l4|l40s|h100)"
        )?;
        surface.print(out, args.per_arch)?;
    }
    out.flush()
}

// ── Styling ──────────────────────────────────────────────────────

/// Minimal ANSI styler. `on=false` → all helpers return their input
/// unchanged. Honors `NO_COLOR` (https://no-color.org/) and falls
/// back to off when stdout isn't a tty.
struct Style {
    on: bool,
}

impl Style {
    fn resolve(when: ColorWhen) -> Self {
        let on = match when {
            ColorWhen::Always => true,
            ColorWhen::Never => false,
            ColorWhen::Auto => {
                let no_color = std::env::var_os("NO_COLOR").is_some();
                !no_color && std::io::stdout().is_terminal()
            }
        };
        Self { on }
    }

    fn wrap(&self, code: &str, s: &str) -> String {
        if self.on {
            format!("\x1b[{code}m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    fn bold(&self, s: &str) -> String {
        self.wrap("1", s)
    }
    fn dim(&self, s: &str) -> String {
        self.wrap("2", s)
    }

    /// Color by kernel-kind family — yellow for GEMM-y kinds,
    /// magenta for attention, cyan for norms, default otherwise.
    fn kind(&self, kind_display: &str) -> String {
        if !self.on {
            return kind_display.to_string();
        }
        let code = kind_family_color(kind_display);
        match code {
            Some(c) => self.wrap(c, kind_display),
            None => kind_display.to_string(),
        }
    }
}

fn kind_family_color(kind: &str) -> Option<&'static str> {
    let k = kind.trim();
    if k.contains("RmsNorm") || k.contains("LayerNorm") {
        Some("36") // cyan
    } else if k.contains("Attention") || k.starts_with("Mla") || k == "RopeAppend" {
        Some("35") // magenta
    } else if k == "Cublas"
        || k.starts_with("Cutlass")
        || k.starts_with("Marlin")
        || k.starts_with("Bnb4")
        || k.starts_with("Fp8")
        || k.contains("GateUp")
        || k.contains("Qkv")
        || k == "FusedGemmBias"
    {
        Some("33") // yellow
    } else {
        None
    }
}

// ── Cluster pass ─────────────────────────────────────────────────

/// One cluster: bucket bounds whose `(backbone, lm_head)` step-
/// streams have the same structural shape, plus the first member's
/// actual step-streams for display.
struct Cluster<'a> {
    labels: Vec<String>,
    backbone: &'a [NormalizedStep],
    lm_head: &'a [NormalizedStep],
    /// Union of `m_min..m_max_excl` across every clustered bucket.
    /// Most clusters have a single M-range so this matches the
    /// header label exactly; clusters that merge multiple M-ranges
    /// (same structural picks at different M) widen to the union.
    /// Used by the GEMM-line shape annotation so each row reads
    /// `(M=…,N=…,K=…)` self-containedly without scanning the header.
    m_min: u64,
    m_max_excl: u64,
}

fn print_variant<W: Write>(
    out: &mut W,
    arch: &str,
    stem: &str,
    tp_world_size: u8,
    buckets: &[BucketDump],
    style: &Style,
) -> io::Result<()> {
    writeln!(out)?;
    writeln!(
        out,
        "{}",
        style.bold(&format!("══ {arch} / {stem} · tp={tp_world_size} ══"))
    )?;

    // First-occurrence-preserving cluster pass.
    let mut clusters: Vec<Cluster<'_>> = Vec::new();
    for b in buckets {
        let label = bucket_label(b);
        if let Some(c) = clusters
            .iter_mut()
            .find(|c| same_shape(c.backbone, &b.backbone) && same_shape(c.lm_head, &b.lm_head))
        {
            c.labels.push(label);
            c.m_min = c.m_min.min(b.m_min);
            c.m_max_excl = c.m_max_excl.max(b.m_max_excl);
        } else {
            clusters.push(Cluster {
                labels: vec![label],
                backbone: &b.backbone,
                lm_head: &b.lm_head,
                m_min: b.m_min,
                m_max_excl: b.m_max_excl,
            });
        }
    }

    // Compute per-variant column widths so the kernel-class annotation
    // aligns across every cluster's tree.
    let widths = compute_widths(&clusters);

    for c in &clusters {
        writeln!(out)?;
        writeln!(
            out,
            "  {} {}",
            style.bold("buckets:"),
            style.dim(&c.labels.join(", "))
        )?;
        writeln!(
            out,
            "  {}",
            style.dim(&format!(
                "({} backbone steps, {} lm_head steps)",
                c.backbone.len(),
                c.lm_head.len()
            ))
        )?;
        let m_range = Some((c.m_min, c.m_max_excl));
        writeln!(out, "  {}", style.bold("backbone"))?;
        print_tree(out, c.backbone, "  ", style, &widths, m_range)?;
        writeln!(out, "  {}", style.bold("lm_head"))?;
        print_tree(out, c.lm_head, "  ", style, &widths, m_range)?;
    }
    Ok(())
}

fn bucket_label(b: &BucketDump) -> String {
    fn fmt_u(v: u64) -> String {
        if v == u64::MAX {
            "∞".to_string()
        } else {
            v.to_string()
        }
    }
    format!(
        "m={}..{} sk={}..{}",
        b.m_min,
        fmt_u(b.m_max_excl),
        b.sk_min,
        fmt_u(b.sk_max_excl),
    )
}

/// True when `a` and `b` have the same op sequence with matching
/// slots, layer indices, kernel classes, and rope/loop markers —
/// IGNORING const fields. Two backbones share a kernel DAG when
/// `same_shape` holds; only scalar tuning differs.
fn same_shape(a: &[NormalizedStep], b: &[NormalizedStep]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.kind == y.kind && structural_fields(x) == structural_fields(y))
}

fn structural_fields(step: &NormalizedStep) -> Vec<&NormalizedField> {
    step.fields
        .iter()
        .filter(|f| {
            !matches!(
                f,
                NormalizedField::ConstU32(_)
                    | NormalizedField::ConstF32Bits(_)
                    | NormalizedField::ConstBool(_)
                    | NormalizedField::ConstU32Array(_)
                    | NormalizedField::ConstU8Array(_)
            )
        })
        .collect()
}

// ── Rendering ────────────────────────────────────────────────────

/// Fields we show per row, pre-computed for width measurement.
/// Const fields are dropped — see `format_step` history.
struct RowFields {
    kind: String,    // post-display-rename, no padding, no style
    slots: String,   // "[s0,s1]" or "" when no slots
    layer: String,   // "L=15" or "" when no Layer field
    kernels: String, // "<RmsNorm>" or "<LinearLayer+RmsNorm>" or ""
    rope: bool,
    /// "(N=128256,K=4096)" for GEMM-class steps, "" otherwise. Lives
    /// in its own column so cuBLAS-vs-CUTLASS analyses can read shape
    /// alongside slot pattern without cross-referencing model config.
    shape: String,
}

#[derive(Default)]
struct Widths {
    kind: usize,
    slots: usize,
    layer: usize,
    kernels: usize,
    shape: usize,
    /// Deepest tree nesting any non-loop row appears at. Top-level
    /// rows are depth 0; loop bodies are depth 1. Used to expand the
    /// kind column on shallow rows so the slots/layer/kernels
    /// columns line up across nesting levels.
    max_depth: usize,
}

fn compute_widths(clusters: &[Cluster<'_>]) -> Widths {
    let mut w = Widths::default();
    for c in clusters {
        let m_range = Some((c.m_min, c.m_max_excl));
        for steps in [c.backbone, c.lm_head] {
            measure_steps(steps, 0, &mut w, m_range);
        }
    }
    w
}

fn measure_steps(
    steps: &[NormalizedStep],
    depth: usize,
    w: &mut Widths,
    m_range: Option<(u64, u64)>,
) {
    let mut i = 0;
    while i < steps.len() {
        let s = &steps[i];
        if s.kind == "Loop" {
            let body_len = loop_body_len(s);
            let body_end = (i + 1 + body_len).min(steps.len());
            measure_steps(&steps[i + 1..body_end], depth + 1, w, m_range);
            i = body_end;
        } else {
            let r = render_fields(s, m_range);
            w.kind = w.kind.max(r.kind.chars().count());
            w.slots = w.slots.max(r.slots.chars().count());
            w.layer = w.layer.max(r.layer.chars().count());
            w.kernels = w.kernels.max(r.kernels.chars().count());
            w.shape = w.shape.max(r.shape.chars().count());
            w.max_depth = w.max_depth.max(depth);
            i += 1;
        }
    }
}

/// Display-rename: `Gemm` (the unfused cublas path,
/// `Instruction::Gemm`'s eval body calls `device.cublas.gemm`)
/// reads more clearly as `Cublas` next to the Cutlass / Marlin /
/// Bnb4 / Fp8 GEMM peers.
fn display_kind(kind: &str) -> &str {
    match kind {
        "Gemm" => "Cublas",
        other => other,
    }
}

/// Format the cluster's M-range as it appears in shape annotations.
/// Single-M clusters (`m_min + 1 == m_max_excl`) render as `M=N`;
/// multi-M as `M=lo..hi` with `∞` for the open upper bound. Matches
/// the bucket label convention used in the header.
fn fmt_m_range(m_min: u64, m_max_excl: u64) -> String {
    if m_min + 1 == m_max_excl {
        format!("M={m_min}")
    } else if m_max_excl == u64::MAX {
        format!("M={m_min}..∞")
    } else {
        format!("M={m_min}..{m_max_excl}")
    }
}

fn render_fields(step: &NormalizedStep, m_range: Option<(u64, u64)>) -> RowFields {
    let mut slots: Vec<String> = Vec::new();
    let mut layer: Option<u32> = None;
    let mut kernels: Vec<&'static str> = Vec::new();
    let mut has_rope = false;
    let mut weight_shape: Option<(u32, u32)> = None;
    for f in &step.fields {
        match f {
            NormalizedField::Slot(s) => slots.push(format!("s{s}")),
            NormalizedField::Layer(l) => layer = Some(*l),
            NormalizedField::LayerKind(k) => kernels.push(k),
            NormalizedField::RopeCosSin => has_rope = true,
            NormalizedField::WeightShape { n, k } => weight_shape = Some((*n, *k)),
            _ => {}
        }
    }
    let slots_str = if slots.is_empty() {
        String::new()
    } else {
        format!("[{}]", slots.join(","))
    };
    let layer_str = match layer {
        Some(l) => format!("L={l}"),
        None => String::new(),
    };
    let kernels_str = if kernels.is_empty() {
        String::new()
    } else {
        format!("<{}>", kernels.join("+"))
    };
    let shape_str = match (weight_shape, m_range) {
        (Some((n, k)), Some((m_min, m_max_excl))) => {
            format!("({},N={n},K={k})", fmt_m_range(m_min, m_max_excl))
        }
        (Some((n, k)), None) => format!("(N={n},K={k})"),
        (None, _) => String::new(),
    };
    RowFields {
        kind: display_kind(step.kind).to_string(),
        slots: slots_str,
        layer: layer_str,
        kernels: kernels_str,
        rope: has_rope,
        shape: shape_str,
    }
}

#[allow(clippy::too_many_arguments)]
fn print_tree<W: Write>(
    out: &mut W,
    steps: &[NormalizedStep],
    indent: &str,
    style: &Style,
    widths: &Widths,
    m_range: Option<(u64, u64)>,
) -> io::Result<()> {
    let tops = top_level_indices(steps);
    for (i, &idx) in tops.iter().enumerate() {
        let is_last = i + 1 == tops.len();
        render_node(out, steps, idx, indent, is_last, 0, style, widths, m_range)?;
    }
    Ok(())
}

fn top_level_indices(steps: &[NormalizedStep]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < steps.len() {
        out.push(i);
        if steps[i].kind == "Loop" {
            i += 1 + loop_body_len(&steps[i]);
        } else {
            i += 1;
        }
    }
    out
}

fn loop_body_len(step: &NormalizedStep) -> usize {
    match step.fields.get(1) {
        Some(NormalizedField::LoopBodyLen(b)) => *b as usize,
        _ => 0,
    }
}

fn loop_count(step: &NormalizedStep) -> u32 {
    match step.fields.first() {
        Some(NormalizedField::LoopCount(c)) => *c,
        _ => 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn render_node<W: Write>(
    out: &mut W,
    steps: &[NormalizedStep],
    idx: usize,
    prefix: &str,
    is_last: bool,
    depth: usize,
    style: &Style,
    widths: &Widths,
    m_range: Option<(u64, u64)>,
) -> io::Result<()> {
    let s = &steps[idx];
    let connector = if is_last { "└─ " } else { "├─ " };
    let child_prefix = format!("{prefix}{}", if is_last { "   " } else { "│  " });

    if s.kind == "Loop" {
        let count = loop_count(s);
        let body_len = loop_body_len(s);
        writeln!(
            out,
            "{prefix}{connector}{}",
            style.bold(&format!("loop ×{count}"))
        )?;
        let body_end = (idx + 1 + body_len).min(steps.len());
        let body = &steps[idx + 1..body_end];
        let body_tops = top_level_indices(body);
        for (j, &bidx) in body_tops.iter().enumerate() {
            let body_is_last = j + 1 == body_tops.len();
            render_node(
                out,
                body,
                bidx,
                &child_prefix,
                body_is_last,
                depth + 1,
                style,
                widths,
                m_range,
            )?;
        }
    } else {
        // Each level of nesting adds 3 visible chars to the prefix
        // (`│  ` or 3 spaces for the last-child branch). Pad
        // shallower rows by `(max_depth - depth) * 3` so the
        // slots / layer / kernels columns share the same absolute
        // starting column as the deepest rows.
        let extra = widths.max_depth.saturating_sub(depth) * 3;
        writeln!(
            out,
            "{prefix}{connector}{}",
            format_row_styled(s, style, widths, extra, m_range)
        )?;
    }
    Ok(())
}

/// Pad first, then style — ANSI escape codes don't count toward
/// visible width, but width-format specifiers run on the raw byte
/// length. This way every row's columns line up regardless of
/// styling.
fn format_row_styled(
    step: &NormalizedStep,
    style: &Style,
    widths: &Widths,
    extra_pad: usize,
    m_range: Option<(u64, u64)>,
) -> String {
    let r = render_fields(step, m_range);

    // Right-pad each segment to its column width with spaces, then
    // wrap the visible text in style codes. `extra_pad` extends the
    // kind column on shallow-nesting rows so slots / layer / kernels
    // start at the same absolute column as the deepest rows.
    let kind_padded = pad_right(&r.kind, widths.kind + extra_pad);
    let slots_padded = pad_right(&r.slots, widths.slots);
    let layer_padded = pad_right(&r.layer, widths.layer);
    let kernels_padded = pad_right(&r.kernels, widths.kernels);
    let shape_padded = pad_right(&r.shape, widths.shape);

    let kind_styled = style.kind(&kind_padded);
    let slots_styled = style.dim(&slots_padded);
    let layer_styled = style.dim(&layer_padded);
    let kernels_styled = style.dim(&kernels_padded);
    let shape_styled = style.dim(&shape_padded);
    let rope_styled = if r.rope {
        style.dim("rope")
    } else {
        " ".repeat(4)
    };

    format!(
        "{kind_styled}  {slots_styled}  {layer_styled}  {kernels_styled}  {shape_styled}  {rope_styled}"
    )
    .trim_end()
    .to_string()
}

fn pad_right(s: &str, width: usize) -> String {
    let cur = s.chars().count();
    if cur >= width {
        s.to_string()
    } else {
        let mut out = String::with_capacity(s.len() + (width - cur));
        out.push_str(s);
        for _ in 0..(width - cur) {
            out.push(' ');
        }
        out
    }
}

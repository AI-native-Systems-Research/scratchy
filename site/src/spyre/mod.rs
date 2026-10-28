//! spyre.html, from site/data/spyre/*.json.
//!
//! Each data file is one A/B sweep of the two Spyre stacks — torch-spyre and
//! triton-scratchy, scratchy's Spyre path — named <date>-<sha8>.json for the
//! day of the sweep and the commit scratchy's side ran. The page charts the
//! latest sweep and lists every one in its about table, the way metal.html
//! charts every run of its bench script.

mod data;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use dioxus::prelude::*;

use crate::Site;
use crate::carbon::{
    ContentPanel, ContentSwitcher, ContentSwitcherItem, Fold, Heading, Layer, Section, Stack,
    Table, TableCell, TableRow, Tile,
};
use crate::charts::{
    Axes, Axis, AxisDomain, BarPoint, Chart, ChartOptions, Colour, Enabled, RatioSquare, Side,
    SparkAxes, Step, TableAxes, chart, colour, glance_maps, glance_mini, highlight, sparkline,
};
use crate::chrome::{self, Content, Head, Library, Published, REPO, Root, Tab};
use crate::thousands_f;
use data::{Cell, Dtype, Metric, Notes, Progress, Provenance, Sweep};

/// The sweep files, relative to site/; copied to the same place under
/// _site/, where the about table links to each.
pub const DATA: &str = "data/spyre";

/// The progress files, relative to site/: one per perf PR, shipped in the PR
/// itself, named for its number. Copied to the same place under _site/. The
/// directory is absent until the first perf PR lands one; the page then has
/// no Progress section, which is not an error.
pub const PROGRESS: &str = "data/spyre/progress";

/// The two stacks, as every chart's sides: triton-scratchy is scratchy's
/// Spyre path, torch-spyre the torch backend it is measured against.
const SCRATCHY: Side = Side {
    line: "triton-scratchy",
    fill: "triton-scratchy ahead",
};
const TORCH: Side = Side {
    line: "torch-spyre",
    fill: "torch-spyre ahead",
};

/// Two numbers this close together are the same number: the run-to-run
/// spread the ratio scale's middle step holds.
const NOISE: f64 = 0.04;

const DASH: &str = "—";

// --------------------------------------------------------------------- load

/// Every sweep file, checked. Fails naming each bad file, so one build
/// reports all of them rather than the first.
fn load(site: &Site) -> Result<Vec<(PathBuf, Sweep)>, String> {
    let dir = site.root.join(DATA);
    let mut files: Vec<_> = fs::read_dir(&dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    let (mut sweeps, mut errors) = (Vec::new(), Vec::new());
    for f in files {
        let name = f
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let text = match fs::read_to_string(&f) {
            Ok(t) => t,
            Err(e) => {
                errors.push(format!("{name}: {e}"));
                continue;
            }
        };
        let sweep: Sweep = match serde_json::from_str(&text) {
            Ok(s) => s,
            Err(e) => {
                errors.push(format!("{name}: {e}"));
                continue;
            }
        };
        let want = format!("{}-{}.json", sweep.date, sweep.triton_scratchy.sha8());
        if name != want {
            errors.push(format!(
                "{name}: name it {want} (<date>-<scratchy sha8>.json)"
            ));
        }
        errors.extend(problems(&sweep).into_iter().map(|e| format!("{name}: {e}")));
        sweeps.push((f, sweep));
    }
    if errors.is_empty() {
        Ok(sweeps)
    } else {
        Err(errors
            .iter()
            .map(|e| format!("spyre page: {e}"))
            .collect::<Vec<_>>()
            .join("\n"))
    }
}

/// What is wrong inside one sweep, beyond what its types refuse: every cell
/// at a bucket the sweep has, measured on both sides, once; the target at a
/// cell that is there.
fn problems(sweep: &Sweep) -> Vec<String> {
    let mut out = Vec::new();
    if sweep.schema != 1 {
        out.push(format!("schema {} is not 1", sweep.schema));
    }
    if sweep.triton_scratchy.commit.is_none() {
        out.push("triton_scratchy.commit is missing (the commit scratchy's side ran)".to_string());
    }
    let mut seen = BTreeSet::new();
    for c in &sweep.cells {
        let at = format!(
            "{} {} {} conc {} at {}",
            c.model,
            c.dtype.label(),
            c.metric.short(),
            c.conc,
            c.context.as_deref().unwrap_or(DASH)
        );
        if let Some(bucket) = c
            .context
            .as_deref()
            .filter(|k| !sweep.contexts.contains_key(*k))
        {
            out.push(format!("cell context {bucket} is not one of the sweep's"));
        }
        if !(c.torch.is_finite() && c.torch > 0.0 && c.scratchy.is_finite() && c.scratchy > 0.0) {
            out.push(format!("{at}: both values must be measured and above 0"));
        }
        if !seen.insert((&c.model, c.dtype, c.metric, c.conc, &c.context)) {
            out.push(format!("{at} is in the sweep twice"));
        }
    }
    if let Some(t) = &sweep.target
        && cell(sweep, &t.model, t.dtype, t.metric, t.conc, &t.context).is_none()
    {
        out.push(format!(
            "target {}: no cell at {} {} {} conc {} at {}",
            t.label,
            t.model,
            t.dtype.label(),
            t.metric.short(),
            t.conc,
            t.context
        ));
    }
    out
}

/// Every progress file, checked the way sweeps are. Fails naming each bad
/// file, so one build reports all of them rather than the first. Ordered by
/// merge date, then PR number: the order their values supersede each other.
fn load_progress(site: &Site) -> Result<Vec<(PathBuf, Progress)>, String> {
    let dir = site.root.join(PROGRESS);
    let entries = match fs::read_dir(&dir) {
        Ok(e) => e,
        // No directory yet: no perf PR has shipped a progress file, so the
        // page has no Progress section. Not an error.
        Err(_) => return Ok(Vec::new()),
    };
    let mut files: Vec<_> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    let (mut out, mut errors) = (Vec::new(), Vec::new());
    for f in files {
        let name = f
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let text = match fs::read_to_string(&f) {
            Ok(t) => t,
            Err(e) => {
                errors.push(format!("{name}: {e}"));
                continue;
            }
        };
        let progress: Progress = match serde_json::from_str(&text) {
            Ok(p) => p,
            Err(e) => {
                errors.push(format!("{name}: {e}"));
                continue;
            }
        };
        let want = format!("{}.json", progress.pr);
        if name != want {
            errors.push(format!("{name}: name it {want} (the PR's number)"));
        }
        errors.extend(
            progress_problems(&progress)
                .into_iter()
                .map(|e| format!("{name}: {e}")),
        );
        out.push((f, progress));
    }
    if errors.is_empty() {
        out.sort_by(|a, b| (&a.1.merged, a.1.pr).cmp(&(&b.1.merged, b.1.pr)));
        Ok(out)
    } else {
        Err(errors
            .iter()
            .map(|e| format!("spyre page: {e}"))
            .collect::<Vec<_>>()
            .join("\n"))
    }
}

/// What is wrong inside one progress entry, beyond what its types refuse:
/// schema 1, full-sha head and base, a date merged, text where a line of it
/// is owed, at least two runs a side of equal count whose last run is each
/// headline (so a headline can never cherry-pick a rep), every value
/// measured and above 0, and no cell twice.
fn progress_problems(p: &Progress) -> Vec<String> {
    let mut out = Vec::new();
    if p.schema != 1 {
        out.push(format!("schema {} is not 1", p.schema));
    }
    for (what, s) in [("head", &p.head), ("base", &p.base)] {
        if !(s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit())) {
            out.push(format!("{what} {s} is not a full commit sha"));
        }
    }
    if !is_date(&p.merged) {
        out.push(format!("merged {} is not a date (YYYY-MM-DD)", p.merged));
    }
    for (what, s) in [
        ("title", &p.title),
        ("pod", &p.pod),
        ("harness", &p.harness),
    ] {
        if s.trim().is_empty() {
            out.push(format!("{what} is empty"));
        }
    }
    if p.cells.is_empty() {
        out.push("no cells: the entry measures nothing".to_string());
    }
    let mut seen = BTreeSet::new();
    for c in &p.cells {
        let at = format!(
            "{} {} {} conc {} at {}",
            c.model,
            c.dtype.label(),
            c.metric.short(),
            c.conc,
            c.context
        );
        if !seen.insert((&c.model, c.dtype, c.metric, c.conc, &c.context)) {
            out.push(format!("{at} is in the entry twice"));
        }
        for (side, headline, runs) in [("main", c.main, &c.reps.main), ("pr", c.pr, &c.reps.pr)] {
            if !(headline.is_finite() && headline > 0.0) {
                out.push(format!("{at}: {side} must be measured and above 0"));
            }
            if runs.len() < 2 {
                out.push(format!(
                    "{at}: {side} has {} run(s); a cell needs at least two a side",
                    runs.len()
                ));
            }
            if let Some(last) = runs.last()
                && *last != headline
            {
                out.push(format!(
                    "{at}: {side} headline {headline} is not its last run ({last})"
                ));
            }
            for v in runs {
                if !(v.is_finite() && *v > 0.0) {
                    out.push(format!("{at}: a {side} run is {v}, not a measurement"));
                }
            }
        }
        if c.reps.main.len() != c.reps.pr.len() {
            out.push(format!(
                "{at}: {} main run(s) against {} pr run(s); the sides must match",
                c.reps.main.len(),
                c.reps.pr.len()
            ));
        }
    }
    out
}

/// "2026-10-21": a day, as the page orders entries by.
fn is_date(s: &str) -> bool {
    s.len() == 10
        && s.as_bytes()[4] == b'-'
        && s.as_bytes()[7] == b'-'
        && s[..4].bytes().all(|b| b.is_ascii_digit())
        && s[5..7].bytes().all(|b| b.is_ascii_digit())
        && s[8..].bytes().all(|b| b.is_ascii_digit())
}

// ---------------------------------------------------------------- accessors

/// The models the sweep measured, in name order (granite's sizes sort with
/// their names).
fn models(sweep: &Sweep) -> Vec<String> {
    let mut ms: Vec<String> = sweep.cells.iter().map(|c| c.model.clone()).collect();
    ms.sort();
    ms.dedup();
    ms
}

/// The sweep's context buckets, shortest first.
fn contexts(sweep: &Sweep) -> Vec<String> {
    let mut cs: Vec<(u32, String)> = sweep
        .contexts
        .iter()
        .map(|(k, &v)| (v, k.clone()))
        .collect();
    cs.sort_by_key(|a| a.0);
    cs.into_iter().map(|(_, k)| k).collect()
}

/// A bucket's place on an axis: the prompt tokens it means.
fn tokens(sweep: &Sweep, bucket: &str) -> f64 {
    f64::from(sweep.contexts.get(bucket).copied().unwrap_or(0))
}

/// The cells at one model, dtype, metric and concurrency, shortest context
/// first; a concurrency sweep's context-less cells keep file order.
fn cells<'a>(
    sweep: &'a Sweep,
    model: &str,
    dtype: Dtype,
    metric: Metric,
    conc: u32,
) -> Vec<&'a Cell> {
    let mut cs: Vec<&Cell> = sweep
        .cells
        .iter()
        .filter(|c| c.model == model && c.dtype == dtype && c.metric == metric && c.conc == conc)
        .collect();
    let at = |c: &Cell| {
        c.context
            .as_deref()
            .map(|k| tokens(sweep, k))
            .unwrap_or(0.0)
    };
    cs.sort_by(|a, b| at(a).total_cmp(&at(b)));
    cs
}

/// The cell at one point of the sweep, if it has one.
fn cell<'a>(
    sweep: &'a Sweep,
    model: &str,
    dtype: Dtype,
    metric: Metric,
    conc: u32,
    context: &str,
) -> Option<&'a Cell> {
    sweep.cells.iter().find(|c| {
        c.model == model
            && c.dtype == dtype
            && c.metric == metric
            && c.conc == conc
            && c.context.as_deref() == Some(context)
    })
}

/// A model's parameter size as its name gives it: "granite-3.1-2b" → "2b".
fn size_name(stem: &str) -> &str {
    stem.split('-')
        .find(|part| {
            part.strip_suffix('b')
                .is_some_and(|n| !n.is_empty() && n.parse::<f64>().is_ok())
        })
        .unwrap_or(stem)
}

/// A cell's value as the tables and tooltips write it.
fn fmt_value(metric: Metric, v: f64) -> String {
    match metric {
        Metric::Itl | Metric::TokS => thousands_f(v, 1),
        Metric::Ttft if v < 1000.0 => format!("{} ms", thousands_f(v, 0)),
        Metric::Ttft => {
            let s = v / 1000.0;
            format!("{} s", thousands_f(s, if s >= 10.0 { 1 } else { 2 }))
        }
    }
}

/// A value with its unit, as a headline writes it: "143.4 ms", "85 ms",
/// "77.2 tok/s" (a whole number without its ".0").
fn fmt_unit(metric: Metric, v: f64) -> String {
    let n = thousands_f(v, if v.fract() == 0.0 { 0 } else { 1 });
    match metric {
        Metric::Itl => format!("{n} ms"),
        Metric::TokS => format!("{n} tok/s"),
        // Already in ms or s.
        Metric::Ttft => fmt_value(metric, v),
    }
}

/// How a set of cells reads: who leads where. Ratios inside the noise band
/// count as ties.
fn lead_text(cells: &[&Cell]) -> String {
    let (ahead, behind): (Vec<&Cell>, Vec<&Cell>) = cells.iter().partition(|c| {
        let r = c.ratio();
        r.is_finite() && r > 1.0 + NOISE
    });
    let behind: Vec<&Cell> = behind
        .into_iter()
        .filter(|c| {
            let r = c.ratio();
            r.is_finite() && r < 1.0 - NOISE
        })
        .collect();
    let ties = cells.len() - ahead.len() - behind.len();
    match (ahead.len(), behind.len()) {
        (0, 0) => "the two stacks are within noise at every context".to_string(),
        (n, 0) if n == cells.len() => "triton-scratchy leads at every context".to_string(),
        (0, n) if n == cells.len() => "torch-spyre leads at every context".to_string(),
        (a, 0) => format!(
            "triton-scratchy leads at {a} of {} contexts, {ties} a tie",
            cells.len()
        ),
        (0, b) => format!(
            "torch-spyre leads at {b} of {} contexts, {ties} a tie",
            cells.len()
        ),
        (a, b) => format!(
            "mixed: triton-scratchy ahead at {a}, torch-spyre at {b}, {ties} a tie of {} \
             contexts",
            cells.len()
        ),
    }
}

/// scratchy's own scaling at a context: how many times slower than perfect
/// its value is there, against the base cell the metric's ideal is set by —
/// short-context ITL held flat (decode), the first real prefill cell's TTFT
/// scaled by prompt length (prefill). None before the base: the ideal says
/// nothing there.
fn perfect(sweep: &Sweep, cells: &[&Cell], at: &Cell) -> Option<f64> {
    let metric = cells.first()?.metric;
    let base = match metric {
        Metric::Itl => cells.first(),
        Metric::Ttft => cells.get(1),
        Metric::TokS => None,
    }?;
    let (bt, bv) = (tokens(sweep, base.context.as_deref()?), base.scratchy);
    let (t, v) = (tokens(sweep, at.context.as_deref()?), at.scratchy);
    if t < bt {
        return None;
    }
    Some(match metric {
        Metric::Itl => v / bv,
        _ => v / (bv * t / bt),
    })
}

fn slug(text: &str) -> String {
    let lower = text.to_lowercase();
    let parts: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_lowercase() && !c.is_ascii_digit())
        .filter(|p| !p.is_empty())
        .collect();
    parts.join("-")
}

fn headers<const N: usize>(names: [&str; N]) -> Vec<String> {
    names.map(String::from).to_vec()
}

/// What a section ends with: the notes on its cells, each once, then the
/// sweep's own notes for it.
fn closing_notes<'a>(
    cells: impl IntoIterator<Item = &'a Cell>,
    sweep_notes: &'a [String],
) -> Element {
    let mut seen = BTreeSet::new();
    let notes: Vec<&str> = cells
        .into_iter()
        .filter_map(|c| c.note.as_deref())
        .chain(sweep_notes.iter().map(String::as_str))
        .filter(|n| seen.insert(*n))
        .collect();
    rsx! {
        for n in notes { p { "{n}" } }
    }
}

/// A part of a section: its heading, then what explains it, then it.
fn part(heading: &str, intro: Element, body: Element) -> Element {
    rsx! {
        Section { level: 3,
            Stack { gap: 3,
                Heading { "{heading}" }
                p { {intro} }
                {body}
            }
        }
    }
}

// ------------------------------------------------------------------ charts

/// One bar chart: a bar per stack at each group, scratchy's blue first.
fn bars(
    groups: &[(String, Option<f64>, Option<f64>)],
    (title, xlabel, ylabel): (&str, &str, &str),
    height: &str,
    label: &str,
) -> Element {
    // Carbon prints values as given: round to what the tables show.
    let round = |v: Option<f64>| v.map(|v| (v * 10.0).round() / 10.0);
    let pts = |name: &'static str, pick: usize| -> Vec<BarPoint> {
        groups
            .iter()
            .map(|(k, torch, scratchy)| BarPoint {
                group: name,
                key: k.clone(),
                value: round(if pick == 0 { *torch } else { *scratchy }),
            })
            .collect()
    };
    let spec = Chart::GroupedBar {
        data: [pts(SCRATCHY.line, 1), pts(TORCH.line, 0)].concat(),
        options: ChartOptions {
            title: title.to_string(),
            axes: Axes {
                bottom: Axis {
                    maps_to: "key",
                    title: xlabel,
                    scale_type: Some("labels"),
                    ticks: None,
                    include_zero: None,
                    visible: None,
                    domain: Some(AxisDomain::Labels(
                        groups.iter().map(|(k, _, _)| k.clone()).collect(),
                    )),
                },
                left: Axis {
                    maps_to: "value",
                    title: ylabel,
                    scale_type: None,
                    ticks: None,
                    include_zero: Some(true),
                    visible: None,
                    domain: None,
                },
            },
            color: Some(Colour::Scale {
                scale: [(SCRATCHY.line, colour("s1")), (TORCH.line, colour("s2"))]
                    .into_iter()
                    .collect(),
            }),
            heatmap: None,
            height: height.to_string(),
            width: None,
            legend: None,
            locale: None,
            combo: None,
            bounds: None,
            points: None,
            toolbar: Enabled { enabled: false },
        },
    };
    chart(&spec, label, "width: 100%", None)
}

/// One model's sparkline panel at one batch: its cells as steps along the
/// context axis.
struct Panel<'a> {
    model: String,
    cells: Vec<&'a Cell>,
    steps: Vec<Step>,
}

/// One model's cells at one batch, as a sparkline's steps.
fn panel<'a>(sweep: &'a Sweep, model: &str, dtype: Dtype, metric: Metric, conc: u32) -> Panel<'a> {
    let cs = cells(sweep, model, dtype, metric, conc);
    let steps = cs
        .iter()
        .map(|c| Step {
            x: tokens(sweep, c.context.as_deref().unwrap_or("")),
            scratchy: Some(c.scratchy),
            rival: Some(c.torch),
            highlight: false,
        })
        .collect();
    Panel {
        model: model.to_string(),
        cells: cs,
        steps,
    }
}

/// The sparkline small multiple: every batch rung's models along the context
/// buckets, scratchy against torch-spyre — a row per batch, a column per
/// model, the gap filled blue ahead and red behind. The columns are headed
/// as the heat maps head theirs; a rung the sweep left unmeasured says so.
fn spark_panel(sweep: &Sweep, dtype: Dtype, metric: Metric) -> Element {
    let value = match metric {
        Metric::Itl => "ITL ms",
        Metric::Ttft => "TTFT ms",
        Metric::TokS => "tok/s",
    };
    let axes = SparkAxes {
        better: metric.better(),
        along: "context tokens",
        value,
        carry: false,
    };
    let models = models(sweep);
    let rungs: Vec<(u32, Vec<Panel>)> = BATCHES
        .iter()
        .map(|&b| {
            let mut ps: Vec<Panel> = models
                .iter()
                .map(|m| panel(sweep, m, dtype, metric, b))
                .collect();
            for p in &mut ps {
                highlight(p.steps.iter_mut().map(|st| (metric.better(), st)));
            }
            (b, ps)
        })
        .collect();
    let grid = format!(
        "display: grid; grid-template-columns: 5.5rem repeat({}, minmax(0, 16rem)); \
         column-gap: 1.5rem; row-gap: 1.25rem; align-items: start",
        models.len()
    );
    rsx! {
        div { style: "{grid}",
            div {}
            for m in &models {
                div { style: "text-align: center", "{m}" }
            }
            for (rung, ps) in &rungs {
                div { "batch {rung}" }
                if ps.iter().all(|p| p.cells.is_empty()) {
                    div { style: "grid-column: 2 / -1; font-size: 0.875rem; color: var(--cds-text-secondary)",
                        "not yet measured at this batch"
                    }
                } else {
                    for p in ps {
                        if p.steps.len() >= 2 {
                            {sparkline(
                                &p.steps,
                                axes,
                                SCRATCHY,
                                TORCH,
                                Some(p.cells.iter().map(|c| c.context.clone().unwrap_or_default()).collect()),
                                "5rem",
                                &format!(
                                    "{}, {} by context at batch {rung}, triton-scratchy against torch-spyre",
                                    p.model, metric.label()
                                ),
                            )}
                        } else if let Some(only) = p.cells.first() {
                            p { style: "font-size: 0.875rem; color: var(--cds-text-secondary)",
                                "One cell only: {fmt_value(metric, only.scratchy)} against \
                                 {fmt_value(metric, only.torch)} at \
                                 {only.context.as_deref().unwrap_or(DASH)} — \
                                 ×{only.ratio():.2}. Rest of the curve pending re-measure."
                            }
                        } else {
                            p { style: "font-size: 0.875rem; color: var(--cds-text-secondary)",
                                "not yet measured at this batch"
                            }
                        }
                    }
                }
            }
        }
    }
}

/// One metric's table for one model: both stacks' values by context, the
/// ratio, and scratchy's own ÷ perfect scaling.
fn metric_table(sweep: &Sweep, cs: &[&Cell], first_header: &str) -> Element {
    rsx! {
        Table { headers: headers([first_header, "torch-spyre", "triton-scratchy", "÷ perfect", "× vs torch-spyre"]),
            for c in cs {
                TableRow {
                    TableCell { "{c.context.as_deref().unwrap_or(DASH)}" }
                    TableCell { "{fmt_value(c.metric, c.torch)}" }
                    TableCell { "{fmt_value(c.metric, c.scratchy)}" }
                    TableCell {
                        {
                            perfect(sweep, cs, c)
                                .map_or_else(|| DASH.to_string(), |p| format!("×{p:.2}"))
                        }
                    }
                    TableCell { "×{c.ratio():.2}" }
                }
            }
        }
    }
}

// ----------------------------------------------------------------- sections

/// The glance's first panel's name.
const HEAT_MAPS: &str = "Heat maps";
const DECODE_SPARKS: &str = "Decode by context";
const PREFILL_SPARKS: &str = "Prefill by context";

/// The batch rungs the grids and sparklines row by: the requests a cell
/// decodes at once. Sixteen and up corrupt on this stack, so eight is the
/// cap; a rung a sweep leaves unmeasured draws striped squares (heat maps)
/// or a pending line (sparklines).
const BATCHES: [u32; 3] = [1, 4, 8];

/// The glance's heat maps as metal.html draws its own: a row per dtype, a
/// column per model, each cell the model's prefill grid then its decode
/// grid — the batch rungs down, the context buckets across.
fn glance(sweep: &Sweep) -> Element {
    let (models, contexts) = (models(sweep), contexts(sweep));
    let dtypes: BTreeSet<Dtype> = sweep.cells.iter().map(|c| c.dtype).collect();
    let stems: Vec<&str> = models.iter().map(String::as_str).collect();
    let rows: Vec<&str> = dtypes.iter().map(|d| d.label()).collect();
    let metrics = [(Metric::Ttft, "Prefill"), (Metric::Itl, "Decode")];
    // glance_maps hands a row back as its label; the dtypes' labels are the
    // only labels a row can be.
    let dtype_of = |row: &str| {
        if row == Dtype::Fp8.label() {
            Dtype::Fp8
        } else {
            Dtype::Fp16
        }
    };
    let mini = |model: &str, dtype: Dtype, metric: Metric, name: &str| -> Element {
        let grid = TableAxes {
            across: ("context", contexts.clone()),
            down: ("batch", BATCHES.iter().map(|b| b.to_string()).collect()),
        };
        let squares = BATCHES
            .iter()
            .flat_map(|&b| {
                contexts.iter().map(move |c| {
                    // Batched decode is measured as throughput, so a decode
                    // grid lights a square the sweep has no ITL cell for
                    // from its tok/s cell at the same batch and context.
                    let found = cell(sweep, model, dtype, metric, b, c).or_else(|| {
                        if metric == Metric::Itl {
                            cell(sweep, model, dtype, Metric::TokS, b, c)
                        } else {
                            None
                        }
                    });
                    RatioSquare {
                        across: c.clone(),
                        down: b.to_string(),
                        ratio: found.map(Cell::ratio).filter(|r| r.is_finite()),
                        highlight: false,
                    }
                })
            })
            .collect();
        glance_mini(
            grid,
            squares,
            format!("scratchy/{}", TORCH.line),
            name,
            "no ratios",
            &format!("{model} {} {}", dtype.label(), metric.label()),
        )
    };
    rsx! {
        Section { id: "glance", level: 2,
            Stack { gap: 4,
                Heading { "At a glance" }
                Tile { style: "width: fit-content; max-width: 100%",
                    Stack { gap: 5,
                        ContentSwitcher { selected: slug(HEAT_MAPS),
                            ContentSwitcherItem { target: slug(HEAT_MAPS), selected: true, "{HEAT_MAPS}" }
                            ContentSwitcherItem { target: slug(PREFILL_SPARKS), selected: false, "{PREFILL_SPARKS}" }
                            ContentSwitcherItem { target: slug(DECODE_SPARKS), selected: false, "{DECODE_SPARKS}" }
                        }
                        div {
                            ContentPanel { id: slug(HEAT_MAPS), shown: true,
                                Stack { gap: 5,
                                    p { style: "contain: inline-size",
                                        "Each model and dtype block holds the model's prefill (time to \
                                         first token) grid, then its decode (inter-token latency) grid — \
                                         a row per dtype, a column per model — on one IBM Spyre card. \
                                         Inside a grid, rows are batch, the requests decoded at once (1, \
                                         4, 8), and columns context, short to long. Colour is how \
                                         many times faster triton-scratchy is than torch-spyre. Blue \
                                         faster, red slower, the middle step within noise. Hover a \
                                         square for its ratio; a striped square is a cell the sweep did \
                                         not measure. The batch rows past one are new and still filling \
                                         in; a decode grid's batch-8 square comes from the sweep's \
                                         throughput (tok/s) cell there, the unit batched decode measures."
                                    }
                                    {glance_maps(&rows, &stems, "minmax(0, 11.5rem)", true, |row, model| rsx! {
                                        for (metric, name) in metrics {
                                            {mini(model, dtype_of(row), metric, name)}
                                        }
                                    })}
                                }
                            }
                            ContentPanel { id: slug(PREFILL_SPARKS), shown: false,
                                div { style: "contain: inline-size",
                                    Stack { gap: 4,
                                        p { "Time to first token as prompts grow, fp16: a row per \
                                             batch, the requests decoded at once (1, 4, 8). Blue \
                                             fill where triton-scratchy is ahead, red where it is \
                                             behind; each fill carries its × factor. Batches past one \
                                             are not yet measured." }
                                        {spark_panel(sweep, Dtype::Fp16, Metric::Ttft)}
                                    }
                                }
                            }
                            ContentPanel { id: slug(DECODE_SPARKS), shown: false,
                                div { style: "contain: inline-size",
                                    Stack { gap: 4,
                                        p { "triton-scratchy against torch-spyre as context grows, \
                                             fp16: a row per batch, the requests decoded at once (1, \
                                             4, 8). Blue fill where triton-scratchy is ahead, red \
                                             where it is behind; each fill carries its × factor. \
                                             Batches past one are not yet measured." }
                                        {spark_panel(sweep, Dtype::Fp16, Metric::Itl)}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// One metric's section shape: its anchor, heading, axis labels, chart
/// titles, and which of the sweep's notes it ends with.
struct SectionSpec {
    id: &'static str,
    heading: &'static str,
    xlabel: &'static str,
    ylabel: &'static str,
    title: fn(&str) -> String,
    notes: fn(&Notes) -> &[String],
}

/// One metric's section, per model: a bar chart by context, then the table.
fn metric_section(
    sweep: &Sweep,
    spec: &SectionSpec,
    intro: Element,
    dtype: Dtype,
    metric: Metric,
) -> Element {
    let models = models(sweep);
    let blocks: Vec<Element> = models
        .iter()
        .filter_map(|m| {
            let cs = cells(sweep, m, dtype, metric, 1);
            if cs.is_empty() {
                return None;
            }
            let groups: Vec<(String, Option<f64>, Option<f64>)> = cs
                .iter()
                .map(|c| {
                    (
                        c.context.clone().unwrap_or_default(),
                        Some(c.torch),
                        Some(c.scratchy),
                    )
                })
                .collect();
            let lead = lead_text(&cs);
            Some(part(
                &format!("{}, {}, one request at a time", size_name(m), dtype.label()),
                rsx! {
                    "{metric.label()} by context bucket (p50 where a median). {lead}. Lower is \
                     better; × vs torch-spyre and ÷ perfect are in the table."
                },
                rsx! {
                    Stack { gap: 3,
                        {bars(
                            &groups,
                            (&(spec.title)(size_name(m)), spec.xlabel, spec.ylabel),
                            "220px",
                            &format!(
                                "{}, {} by context, triton-scratchy against torch-spyre",
                                m, metric.label()
                            ),
                        )}
                        Fold { title: "Table view",
                            {metric_table(sweep, &cs, "context")}
                        }
                    }
                },
            ))
        })
        .collect();
    if blocks.is_empty() {
        return rsx! {};
    }
    rsx! {
        Section { id: "{spec.id}", level: 2,
            Stack { gap: 4,
                Heading { "{spec.heading}" }
                p { {intro} }
                Tile {
                  Layer { level: 1,
                    Stack { gap: 6,
                        for b in &blocks { {b} }
                        {closing_notes(
                            models.iter().flat_map(|m| cells(sweep, m, dtype, metric, 1)),
                            (spec.notes)(&sweep.notes),
                        )}
                    }
                  }
                }
            }
        }
    }
}

/// The decode section's shape: inter-token latency, fp16.
const DECODE: SectionSpec = SectionSpec {
    id: "decode",
    heading: "Decode",
    xlabel: "context bucket",
    ylabel: "ITL ms",
    title: |size| format!("Decode ITL, {size} (ms, lower is better)"),
    notes: |n| &n.decode,
};

/// The prefill section's shape: time to first token, fp16.
const PREFILL: SectionSpec = SectionSpec {
    id: "prefill",
    heading: "Prefill",
    xlabel: "prompt bucket",
    ylabel: "TTFT ms",
    title: |size| format!("Prefill TTFT, {size} (ms, lower is better)"),
    notes: |n| &n.prefill,
};

fn decode_section(sweep: &Sweep) -> Element {
    metric_section(
        sweep,
        &DECODE,
        rsx! {
            "Inter-token latency, fp16, one request at a time. " strong { "÷ perfect" } " is \
             triton-scratchy's own scaling: how many times slower than holding its \
             short-context latency flat."
        },
        Dtype::Fp16,
        Metric::Itl,
    )
}

fn prefill_section(sweep: &Sweep) -> Element {
    metric_section(
        sweep,
        &PREFILL,
        rsx! {
            "Time to first token, fp16, one request at a time. " strong { "÷ perfect" } " \
             scales each model's second-shortest cell's time by prompt length; its shortest \
             sits below that base (" {DASH} ")."
        },
        Dtype::Fp16,
        Metric::Ttft,
    )
}

/// The concurrency sweeps: tokens per second at more than one request at
/// once, one block per concurrency.
fn conc_section(sweep: &Sweep) -> Element {
    let concs: Vec<u32> = {
        let cs: BTreeSet<u32> = sweep
            .cells
            .iter()
            .map(|c| c.conc)
            .filter(|c| *c > 1)
            .collect();
        cs.into_iter().collect()
    };
    if concs.is_empty() {
        return rsx! {};
    }
    let models = models(sweep);
    let blocks: Vec<Element> = concs
        .iter()
        .filter_map(|&conc| {
            let rows: Vec<(String, Option<f64>, Option<f64>, &Cell)> = models
                .iter()
                .filter_map(|m| {
                    let cs = cells(sweep, m, Dtype::Fp16, Metric::TokS, conc);
                    let c = cs.first().copied()?;
                    Some((size_name(m).to_string(), Some(c.torch), Some(c.scratchy), c))
                })
                .collect();
            if rows.is_empty() {
                return None;
            }
            let groups: Vec<(String, Option<f64>, Option<f64>)> = rows
                .iter()
                .map(|(k, t, s, _)| (k.clone(), *t, *s))
                .collect();
            let lead = lead_text(&rows.iter().map(|r| r.3).collect::<Vec<_>>());
            Some(part(
                &format!("conc {conc}, fp16"),
                rsx! {
                    "Decoded tokens per second with {conc} requests at once. {lead}. Higher is \
                     better."
                },
                rsx! {
                    Stack { gap: 3,
                        {bars(
                            &groups,
                            (
                                &format!("Throughput at conc {conc} (tok/s, higher is better)"),
                                "model",
                                "tok/s",
                            ),
                            "220px",
                            &format!("throughput at conc {conc}, triton-scratchy against torch-spyre"),
                        )}
                        Fold { title: "Table view",
                            Table { headers: headers(["model", "torch-spyre", "triton-scratchy", "× vs torch-spyre"]),
                                for (name, _, _, c) in &rows {
                                    TableRow {
                                        TableCell { "{name}" }
                                        TableCell { "{fmt_value(Metric::TokS, c.torch)}" }
                                        TableCell { "{fmt_value(Metric::TokS, c.scratchy)}" }
                                        TableCell { "×{c.ratio():.2}" }
                                    }
                                }
                            }
                        }
                    }
                },
            ))
        })
        .collect();
    if blocks.is_empty() {
        return rsx! {};
    }
    rsx! {
        Section { id: "conc", level: 2,
            Stack { gap: 4,
                Heading { "Concurrency" }
                p { "How the two stacks hold up as requests share the card." }
                Tile {
                  Layer { level: 1,
                    Stack { gap: 6,
                        for b in &blocks { {b} }
                        {closing_notes(
                            sweep.cells.iter().filter(|c| {
                                c.conc > 1 && c.dtype == Dtype::Fp16 && c.metric == Metric::TokS
                            }),
                            &sweep.notes.conc,
                        )}
                    }
                  }
                }
            }
        }
    }
}

/// fp8: every fp8 cell, the chart at each model's shortest context.
fn fp8_section(sweep: &Sweep) -> Element {
    let models = models(sweep);
    let all: Vec<&Cell> = sweep
        .cells
        .iter()
        .filter(|c| c.dtype == Dtype::Fp8)
        .collect();
    if all.is_empty() {
        return rsx! {};
    }
    // The chart: decode ITL at short context, one group per model.
    let groups: Vec<(String, Option<f64>, Option<f64>)> = models
        .iter()
        .filter_map(|m| {
            let cs = cells(sweep, m, Dtype::Fp8, Metric::Itl, 1);
            let c = cs.first().copied()?;
            Some((size_name(m).to_string(), Some(c.torch), Some(c.scratchy)))
        })
        .collect();
    let itl: Vec<&Cell> = all
        .iter()
        .copied()
        .filter(|c| c.metric == Metric::Itl)
        .collect();
    let lead = lead_text(&itl);
    rsx! {
        Section { id: "fp8", level: 2,
            Stack { gap: 4,
                Heading { "fp8" }
                p {
                    "fp8 weights, one request at a time. {lead}. Lower is better."
                }
                Tile {
                  Layer { level: 1,
                    Stack { gap: 5,
                        if !groups.is_empty() {
                            {bars(
                                &groups,
                                ("fp8 decode ITL, shortest context (ms, lower is better)", "model", "ITL ms"),
                                "220px",
                                "fp8 decode ITL at short context, triton-scratchy against torch-spyre",
                            )}
                        }
                        Fold { title: "Table view",
                            Table { headers: headers(["cell", "torch-spyre", "triton-scratchy", "× vs torch-spyre"]),
                                for c in &all {
                                    TableRow {
                                        TableCell {
                                            "{size_name(&c.model)} · {c.metric.short()} · \
                                             {c.context.as_deref().unwrap_or(DASH)}"
                                        }
                                        TableCell { "{fmt_value(c.metric, c.torch)}" }
                                        TableCell { "{fmt_value(c.metric, c.scratchy)}" }
                                        TableCell { "×{c.ratio():.2}" }
                                    }
                                }
                            }
                        }
                        {closing_notes(all.iter().copied(), &sweep.notes.fp8)}
                    }
                  }
                }
            }
        }
    }
}

/// Whether the sweep says what moved since the one before, or what is left
/// to do: the "What moved" section is there only then.
fn says_what_moved(sweep: &Sweep) -> bool {
    !sweep.moved.is_empty() || !sweep.levers.is_empty()
}

// ----------------------------------------------------------------- progress

/// One row of the Progress table: the newest measurement of a cell the perf
/// PRs are moving, whichever side measured it, and torch's newest sweep
/// value at that cell.
struct ProgressRow {
    /// triton-scratchy's newest value at the cell.
    latest: f64,
    /// Where it came from: "PR #682, 2026-10-21" or "sweep 2026-10-19".
    source: String,
    /// The source's place in order: its date, whether it is a PR (a PR on
    /// the same day as a sweep counts as newer), and its number.
    order: (String, bool, u32),
    /// torch-spyre's value at the cell in the newest sweep that has it, with
    /// that sweep's date: torch is never re-measured for a PR.
    torch: Option<(f64, String)>,
    /// The note on the PR cell the row's value came from; a sweep-sourced
    /// row carries none, because a sweep's cell notes already print under
    /// that sweep's sections.
    note: Option<String>,
}

/// A cell's place on the sweep grid: model, dtype, metric, concurrency and
/// context bucket — the key of every progress table row and timeline.
type CellKey = (String, Dtype, Metric, u32, String);

/// The cells the perf PRs are moving: every key any progress entry measured.
fn moved_keys(progress: &[(PathBuf, Progress)]) -> BTreeSet<CellKey> {
    progress
        .iter()
        .flat_map(|(_, p)| {
            p.cells.iter().map(|c| {
                (
                    c.model.clone(),
                    c.dtype,
                    c.metric,
                    c.conc,
                    c.context.clone(),
                )
            })
        })
        .collect()
}

/// The Progress table's rows: one per cell any progress entry measured — the
/// cells the perf PRs are moving, nothing else. The newest value wins — a
/// PR's over a sweep's when it merged later, a sweep's over an older PR's —
/// and torch's value comes from the newest sweep that has the cell, never a
/// PR (no PR re-measures torch).
fn progress_rows(
    progress: &[(PathBuf, Progress)],
    sweeps: &[(PathBuf, Sweep)],
) -> BTreeMap<CellKey, ProgressRow> {
    let moved = moved_keys(progress);
    let mut rows: BTreeMap<CellKey, ProgressRow> = BTreeMap::new();
    // Sweeps load oldest first: each one with the cell supersedes the last.
    for (_, s) in sweeps {
        for c in &s.cells {
            let Some(context) = &c.context else { continue };
            let key = (c.model.clone(), c.dtype, c.metric, c.conc, context.clone());
            if !moved.contains(&key) {
                continue;
            }
            rows.insert(
                key,
                ProgressRow {
                    latest: c.scratchy,
                    source: format!("sweep {}", s.date),
                    order: (s.date.clone(), false, 0),
                    torch: Some((c.torch, s.date.clone())),
                    note: None,
                },
            );
        }
    }
    // Progress entries load in merge order; each only supersedes what it is
    // newer than, and carries torch's sweep value forward untouched.
    for (_, p) in progress {
        for c in &p.cells {
            let key = cell_key(c);
            let order = (p.merged.clone(), true, p.pr);
            let newer = rows.get(&key).is_none_or(|row| order > row.order);
            if newer {
                let torch = rows.get(&key).and_then(|r| r.torch.clone());
                rows.insert(
                    key,
                    ProgressRow {
                        latest: c.pr,
                        source: format!("PR #{}, {}", p.pr, p.merged),
                        order,
                        torch,
                        note: c.note.clone(),
                    },
                );
            }
        }
    }
    rows
}

/// One source that measured a moved cell: its place in order, its tick's
/// name on a timeline's axis, scratchy's value there, and torch's — a
/// sweep's, never a PR's.
struct TimelineSource {
    order: (String, bool, u32),
    name: String,
    scratchy: f64,
    torch: Option<f64>,
}

/// Each moved cell's timeline: every source that measured it, oldest first —
/// a sweep with both stacks' values, a PR with its own — one evenly spaced
/// step per source on the sparkline's log axis. Where a PR sits past
/// torch's newest sweep, the chart carries torch's line forward from that
/// sweep, dotted, and the fill follows it. A cell only one source measured
/// has no timeline: one point is not a line, and the table already carries it.
fn progress_timelines(
    progress: &[(PathBuf, Progress)],
    sweeps: &[(PathBuf, Sweep)],
) -> BTreeMap<CellKey, (Vec<Step>, Vec<String>)> {
    let mut out = BTreeMap::new();
    for key in moved_keys(progress) {
        let mut sources: Vec<TimelineSource> = Vec::new();
        for (_, s) in sweeps {
            for c in &s.cells {
                let Some(context) = &c.context else { continue };
                if (c.model.clone(), c.dtype, c.metric, c.conc, context.clone()) == key {
                    sources.push(TimelineSource {
                        order: (s.date.clone(), false, 0),
                        name: s.date[5..].to_string(),
                        scratchy: c.scratchy,
                        torch: Some(c.torch),
                    });
                }
            }
        }
        for (_, p) in progress {
            for c in &p.cells {
                if cell_key(c) == key {
                    sources.push(TimelineSource {
                        order: (p.merged.clone(), true, p.pr),
                        name: format!("#{}", p.pr),
                        scratchy: c.pr,
                        torch: None,
                    });
                }
            }
        }
        sources.sort_by(|a, b| a.order.cmp(&b.order));
        if sources.len() >= 2 {
            let mut steps: Vec<Step> = sources
                .iter()
                .enumerate()
                .map(|(k, s)| Step {
                    x: 2f64.powi(k as i32),
                    scratchy: Some(s.scratchy),
                    rival: s.torch,
                    highlight: false,
                })
                .collect();
            // The fill between the lines carries its ×-factor at the first
            // and the last source where scratchy was measured. At the first,
            // both stacks were measured; at the last — a PR's step, past
            // torch's newest sweep — the chart carries torch's line dotted
            // from that sweep, and the factor pairs the PR's value with that
            // carried level: the same pairing the table above and the heat
            // maps make.
            let measured = |s: &Step| s.scratchy.is_some();
            if let Some(first) = steps.iter().position(measured) {
                steps[first].highlight = true;
            }
            if let Some(last) = steps.iter().rposition(measured) {
                steps[last].highlight = true;
            }
            let names = sources.into_iter().map(|s| s.name).collect();
            out.insert(key, (steps, names));
        }
    }
    out
}

/// The rendered sweep, freshened: a cell any perf PR re-measured since the
/// sweep shows the PR's value — in the heat maps, the grids, the tables and
/// the tiles, everything the page draws from the sweep — with a note naming
/// the PR and both numbers. Torch's value stays the sweep's own (no PR
/// re-measures torch), and the sweep files stay the record; only this
/// rendered view carries the newer numbers. Returns the sources it applied.
fn apply_progress(sweep: &mut Sweep, rows: &BTreeMap<CellKey, ProgressRow>) -> Vec<String> {
    let mut sources = BTreeSet::new();
    for c in &mut sweep.cells {
        let Some(context) = c.context.clone() else {
            continue;
        };
        let key = (c.model.clone(), c.dtype, c.metric, c.conc, context);
        let Some(row) = rows.get(&key) else { continue };
        // Only a PR-sourced row is newer than this sweep's own value: a
        // sweep newer than every PR leaves the cell as it measured it.
        if !row.order.1 {
            continue;
        }
        let from = c.scratchy;
        c.scratchy = row.latest;
        let line = format!(
            "{}: {} → {}",
            row.source,
            fmt_unit(c.metric, from),
            fmt_unit(c.metric, row.latest)
        );
        sources.insert(row.source.clone());
        c.note = Some(match c.note.take() {
            Some(n) if n.ends_with('.') => format!("{n} {line}."),
            Some(n) => format!("{n}. {line}."),
            None => format!("{line}."),
        });
    }
    sources.into_iter().collect()
}

/// A progress cell must land on the page's grid: its model and its context
/// have to be ones the sweeps measured, or its values cannot join anything
/// the page shows.
fn progress_vocabulary(
    progress: &[(PathBuf, Progress)],
    sweeps: &[(PathBuf, Sweep)],
) -> Vec<String> {
    let models: BTreeSet<String> = sweeps.iter().flat_map(|(_, s)| models(s)).collect();
    let contexts: BTreeSet<String> = sweeps
        .iter()
        .flat_map(|(_, s)| s.contexts.keys().cloned())
        .collect();
    let mut out = Vec::new();
    for (_, p) in progress {
        for c in &p.cells {
            if !models.contains(c.model.as_str()) {
                out.push(format!(
                    "progress {}: model {} is not one the sweeps measured",
                    p.pr, c.model
                ));
            }
            if !contexts.contains(c.context.as_str()) {
                out.push(format!(
                    "progress {}: context {} is not one the sweeps measured",
                    p.pr, c.context
                ));
            }
        }
    }
    out
}

/// A progress cell's place on the sweep grid.
fn cell_key(c: &data::ProgressCell) -> CellKey {
    (
        c.model.clone(),
        c.dtype,
        c.metric,
        c.conc,
        c.context.clone(),
    )
}

/// A cell's name in the Progress section's tables and timelines: model size,
/// dtype, metric, concurrency and context bucket.
fn cell_name(key: &CellKey) -> String {
    let (model, dtype, metric, conc, context) = key;
    let base = format!(
        "{} · {} · {}",
        size_name(model),
        dtype.label(),
        metric.short()
    );
    if *conc > 1 {
        format!("{base} · conc {conc} · {context}")
    } else {
        format!("{base} · {context}")
    }
}

/// One cell's timeline as a sparkline: its steps and tick names against
/// torch, the axis titles from the cell's metric.
fn timeline_chart(key: &CellKey, steps: &[Step], names: &[String]) -> Element {
    let value = match key.2 {
        Metric::Itl => "ITL ms",
        Metric::Ttft => "TTFT ms",
        Metric::TokS => "tok/s",
    };
    sparkline(
        steps,
        SparkAxes {
            better: key.2.better(),
            along: "measured",
            value,
            carry: true,
        },
        SCRATCHY,
        TORCH,
        Some(names.to_vec()),
        "5rem",
        &format!(
            "{}, over time, triton-scratchy against torch-spyre",
            cell_name(key)
        ),
    )
}

/// The Progress section: every perf PR's gate cells on the page. The sweeps
/// stay the record; this is the newest measurement of each cell the PRs are
/// moving, with its source, beside torch's newest sweep value there — one
/// row per cell, so the next perf PR lands with its data visible. Under the
/// table: each cell over time, then every entry's cells against their main.
fn progress_section(progress: &[(PathBuf, Progress)], sweeps: &[(PathBuf, Sweep)]) -> Element {
    if progress.is_empty() {
        return rsx! {};
    }
    let rows = progress_rows(progress, sweeps);
    let timelines = progress_timelines(progress, sweeps);
    let grid = "display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); \
                column-gap: 1.5rem; row-gap: 1.25rem; align-items: start";
    rsx! {
        Section { id: "progress", level: 2,
            Stack { gap: 4,
                Heading { "Progress" }
                p {
                    "Every perf PR's merge gate, on the page: the cells its run measured, \
                     main against the PR on one pod, back to back. Torch-spyre is not \
                     re-measured for a PR — each cell pairs with the newest sweep's torch \
                     value, labelled with that sweep's date. The sweeps stay the record; \
                     these rows are the newest measurement of each cell the PRs are \
                     moving, from a PR or a sweep, each naming its source."
                }
                Tile {
                  Layer { level: 1,
                    Table { headers: headers([
                            "cell", "latest triton-scratchy", "source", "torch-spyre", "× vs torch-spyre",
                        ]),
                        for (key, row) in &rows {
                            TableRow {
                                TableCell { "{cell_name(key)}" }
                                TableCell { "{fmt_value(key.2, row.latest)}" }
                                TableCell { "{row.source}" }
                                TableCell {
                                    {
                                        row.torch.as_ref().map_or_else(
                                            || DASH.to_string(),
                                            |(v, d)| format!("{} (sweep {d})", fmt_value(key.2, *v)),
                                        )
                                    }
                                }
                                TableCell {
                                    {
                                        row.torch.as_ref()
                                            .and_then(|(t, _)| key.2.better().faster(Some(row.latest), Some(*t)))
                                            .map_or_else(|| DASH.to_string(), |r| format!("×{r:.2}"))
                                    }
                                }
                            }
                        }
                    }
                  }
                }
                for (key, row) in &rows {
                    if let Some(note) = &row.note {
                        p { "{cell_name(key)}: {note}" }
                    }
                }
                if !timelines.is_empty() {
                    {part(
                        "Each cell, over time",
                        rsx! {
                            "Every source that measured a moved cell, oldest first — a \
                             sweep with both stacks' values, a PR with its own — so the \
                             line shows what the PRs are closing. Where a PR sits past \
                             torch-spyre's newest sweep, torch's line continues dotted, \
                             carried flat from that sweep's measurement — the level the \
                             PR's value is paired with everywhere on this page — and the \
                             fill between the lines follows it. That fill is labelled \
                             with its ×-factor at the first source and the newest: where \
                             the cell started against torch, and where it stands now."
                        },
                        rsx! {
                            div { style: "{grid}",
                                for (key, (steps, names)) in &timelines {
                                    div {
                                        p { style: "font-size: 0.875rem; margin: 0 0 0.25rem",
                                            "{cell_name(key)}"
                                        }
                                        {timeline_chart(key, steps, names)}
                                    }
                                }
                            }
                        },
                    )}
                }
                {part(
                    "What each PR moved",
                    rsx! {
                        "Every progress entry's cells, the PR against the main it ran on, \
                         newest first — the file's own sides, main and pr. Δ is how far \
                         the PR moved the cell in the better direction: above ×1 improved \
                         it, below ×1 regressed it. A gate's no-regression cells land here \
                         beside its wins."
                    },
                    rsx! {
                        Tile {
                          Layer { level: 1,
                            Table { headers: headers([
                                    "PR · merged", "title", "cell", "main", "PR", "Δ",
                                ]),
                                for (_, p) in progress.iter().rev() {
                                    for c in &p.cells {
                                        TableRow {
                                            TableCell { {format!("#{} · {}", p.pr, p.merged)} }
                                            TableCell { "{p.title}" }
                                            TableCell { "{cell_name(&cell_key(c))}" }
                                            TableCell { "{fmt_value(c.metric, c.main)}" }
                                            TableCell { "{fmt_value(c.metric, c.pr)}" }
                                            TableCell {
                                                {
                                                    c.metric.better().faster(Some(c.pr), Some(c.main))
                                                        .map_or_else(|| DASH.to_string(), |r| format!("×{r:.2}"))
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                          }
                        }
                        for (_, p) in progress.iter().rev() {
                            p { style: "font-size: 0.875rem",
                                {format!("#{} measured on {} — {}", p.pr, p.pod, p.harness)}
                            }
                        }
                    },
                )}
            }
        }
    }
}

fn moved_section(sweep: &Sweep) -> Element {
    if !says_what_moved(sweep) {
        return rsx! {};
    }
    let commit = sweep.triton_scratchy.commit.as_deref().unwrap_or(DASH);
    rsx! {
        Section { id: "moved", level: 2,
            Stack { gap: 4,
                Heading { "What moved" }
                p {
                    "triton-scratchy at " code { "{commit}" }
                    if let Some(since) = &sweep.since { ", {since} merged since the previous sweep" }
                    "."
                }
                if !sweep.moved.is_empty() {
                    Table { headers: headers(["change", "what it did"]),
                        for m in &sweep.moved {
                            TableRow { TableCell { "{m.pr}" } TableCell { "{m.what}" } }
                        }
                    }
                }
                if !sweep.levers.is_empty() {
                    Section { level: 3,
                        Stack { gap: 3,
                            Heading { "Open levers" }
                            Table { headers: headers(["lever", "open work"]),
                                for l in &sweep.levers {
                                    TableRow { TableCell { "{l.lever}" } TableCell { "{l.work}" } }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------- tiles, about

/// A headline tile's tone.
#[derive(Clone, Copy)]
enum Tone {
    Good,
    Bad,
    Plain,
}

fn tile(label: String, value: String, main: String, sub: Option<String>, tone: Tone) -> Element {
    let colour = match tone {
        Tone::Good => "var(--cds-support-success)",
        Tone::Bad => "var(--cds-support-error)",
        Tone::Plain => "var(--cds-text-secondary)",
    };
    rsx! {
        Tile { style: "flex: 1 1 13rem; min-width: 13rem",
            Stack { gap: 2,
                div { style: "font-size: 0.75rem; color: var(--cds-text-secondary)", "{label}" }
                div { style: "font-size: 1.875rem; font-weight: 600; letter-spacing: -0.01em", "{value}" }
                div { style: "font-size: 0.75rem; line-height: 1.45; color: {colour}", "{main}" }
                if let Some(s) = sub {
                    div { style: "font-size: 0.75rem; line-height: 1.45; color: var(--cds-text-secondary)", "{s}" }
                }
            }
        }
    }
}

/// The headline tiles under the intro: the sweep's best and worst cells, and
/// its target.
fn tiles(sweep: &Sweep) -> Element {
    let models = models(sweep);
    // The first (smallest) and last (largest) models with cells of a kind.
    let with = |metric: Metric, dtype: Dtype, conc: u32, first: bool| {
        let mut ms = models
            .iter()
            .filter(|m| !cells(sweep, m, dtype, metric, conc).is_empty());
        (if first { ms.next() } else { ms.next_back() })
            .map(|m| (m.clone(), cells(sweep, m, dtype, metric, conc)))
    };
    let mut out: Vec<Element> = Vec::new();
    // fp8 decode: the smallest model's short-context cell.
    if let Some((m, cs)) = with(Metric::Itl, Dtype::Fp8, 1, true) {
        let c = cs[0];
        out.push(tile(
            format!(
                "fp8 decode · {} · {}",
                size_name(&m),
                c.context.as_deref().unwrap_or("short context")
            ),
            format!("×{:.1}", c.ratio()),
            format!(
                "faster than torch-spyre — {} vs {} ms",
                fmt_value(Metric::Itl, c.scratchy),
                fmt_value(Metric::Itl, c.torch)
            ),
            c.note.clone(),
            Tone::Good,
        ));
    }
    // The sweep's target, beside where its cell is today.
    if let Some(t) = &sweep.target
        && let Some(c) = cell(sweep, &t.model, t.dtype, t.metric, t.conc, &t.context)
    {
        let now = format!(
            "today {} (torch-spyre {})",
            fmt_unit(t.metric, c.scratchy),
            fmt_unit(t.metric, c.torch)
        );
        // How many times faster scratchy has to get there.
        let to_go = t
            .metric
            .better()
            .faster(Some(t.value), Some(c.scratchy))
            .unwrap_or(f64::NAN);
        out.push(tile(
            t.label.clone(),
            format!("~{}", fmt_unit(t.metric, t.value)),
            t.why.clone(),
            Some(if to_go > 1.0 {
                format!("{now}: ×{to_go:.2} to go")
            } else {
                format!("{now}: met")
            }),
            Tone::Plain,
        ));
    }
    // fp16 decode: the smallest model's lead, first to last context.
    if let Some((m, cs)) = with(Metric::Itl, Dtype::Fp16, 1, true) {
        let (first, last) = (cs[0], cs[cs.len() - 1]);
        let ahead = first.ratio() > 1.0;
        let sub = (cs.len() > 1 && (last.ratio() - first.ratio()).abs() > NOISE).then(|| {
            if last.ratio() < first.ratio() {
                format!(
                    "lead narrows to ×{:.2} at {}",
                    last.ratio(),
                    last.context.as_deref().unwrap_or(DASH)
                )
            } else {
                format!(
                    "gap widens to ×{:.2} at {}",
                    last.ratio(),
                    last.context.as_deref().unwrap_or(DASH)
                )
            }
        });
        out.push(tile(
            format!("fp16 decode · {} · conc-1", size_name(&m)),
            format!("×{:.2}", first.ratio()),
            format!(
                "{} torch-spyre — {} vs {} ms at {}",
                if ahead { "ahead of" } else { "behind" },
                fmt_value(Metric::Itl, first.scratchy),
                fmt_value(Metric::Itl, first.torch),
                first.context.as_deref().unwrap_or(DASH)
            ),
            sub,
            if ahead { Tone::Good } else { Tone::Bad },
        ));
    }
    // Concurrency: the smallest model's throughput cell at its highest conc.
    let conc = sweep.cells.iter().map(|c| c.conc).max().unwrap_or(1);
    let best = (conc > 1)
        .then(|| with(Metric::TokS, Dtype::Fp16, conc, true))
        .flatten();
    if let Some((m, cs)) = best {
        let c = cs[0];
        let ahead = c.ratio() > 1.0;
        out.push(tile(
            format!("fp16 throughput · {} · conc-{conc}", size_name(&m)),
            format!("×{:.2}", c.ratio()),
            format!(
                "{} torch-spyre — {} vs {} tok/s",
                if ahead { "ahead of" } else { "behind" },
                fmt_value(Metric::TokS, c.scratchy),
                fmt_value(Metric::TokS, c.torch)
            ),
            c.note.clone(),
            if ahead { Tone::Good } else { Tone::Bad },
        ));
    }
    rsx! {
        div { style: "display: flex; flex-wrap: wrap; gap: 1rem",
            for t in out { {t} }
        }
    }
}

/// One side's provenance as the about table writes it.
fn side(p: &Provenance) -> String {
    let where_ = p
        .host
        .as_deref()
        .map_or_else(String::new, |h| format!(", {h}"));
    format!("{}{where_}", p.measured)
}

/// The about fold: how the numbers were measured, what the words mean, and
/// every sweep the page has had.
fn about(sweeps: &[(PathBuf, Sweep)]) -> Element {
    rsx! {
        Fold { title: "About these numbers",
            Stack { gap: 4,
                p {
                    "torch-spyre and triton-scratchy are measured by their own harnesses (each \
                     sweep's are in the table below) at slightly different prompt lengths, so \
                     cells sit in context buckets and the page compares ratios, not absolutes. \
                     Each sweep's JSON, with both sides' provenance, is linked below. Perf PRs \
                     also land their gate cells as progress entries, one JSON per PR under \
                     " code { "{PROGRESS}/" } ", which the Progress section pairs with the \
                     newest sweep's torch value."
                }
                p {
                    code { "ITL" } " inter-token latency · " code { "TTFT" } " time to first \
                     token · " code { "conc" } " concurrent requests, the batch the heat maps \
                     row by · " code { "tok/s" } " decoded tokens per second · " code { "÷ perfect" } " \
                     triton-scratchy's ratio to its own ideal: short-context ITL held flat (decode), the \
                     second-shortest cell's TTFT scaled by prompt length (prefill)."
                }
                Table { headers: headers(["sweep", "triton-scratchy", "torch-spyre", "harnesses", "data"]),
                    for (file, s) in sweeps.iter().rev() {
                        TableRow {
                            TableCell { "{s.date}" }
                            TableCell {
                                a { href: "{REPO}/commit/{s.triton_scratchy.commit.clone().unwrap_or_default()}",
                                    code { "{s.triton_scratchy.sha8()}" }
                                }
                            }
                            TableCell { "{side(&s.torch_spyre)}" }
                            TableCell {
                                "{s.torch_spyre.harness} · {s.triton_scratchy.harness}"
                            }
                            TableCell {
                                if let Some(n) = file.file_name() {
                                    a { href: "{DATA}/{n.to_string_lossy()}", "json" }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// -------------------------------------------------------------------- build

pub fn build(site: &Site, assets: &Published) -> Result<String, String> {
    let sweeps = load(site)?;
    let progress = load_progress(site)?;
    let off_grid = progress_vocabulary(&progress, &sweeps);
    if !off_grid.is_empty() {
        return Err(off_grid
            .iter()
            .map(|e| format!("spyre page: {e}"))
            .collect::<Vec<_>>()
            .join("\n"));
    }
    // The page charts the latest sweep, freshened by the progress entries:
    // a cell a perf PR re-measured since shows the PR's value. The about
    // table lists every sweep; the sweep files stay the record.
    let rows = progress_rows(&progress, &sweeps);
    let latest = sweeps.last().map(|(_, s)| {
        let mut fresh = s.clone();
        let sources = apply_progress(&mut fresh, &rows);
        (fresh, sources)
    });
    let mut sections: Vec<(String, String)> = Vec::new();
    let (intro, tiles, body) = match &latest {
        None => (
            rsx! {
                "scratchy's Spyre path against torch-spyre on one Spyre card, per A/B sweep."
            },
            rsx! {},
            rsx! {
                p {
                    "No sweeps published yet. Copy an A/B sweep's JSON into "
                    code { "site/{DATA}/" } "."
                }
            },
        ),
        Some((sweep, freshened)) => {
            sections.push(("glance".to_string(), "At a glance".to_string()));
            sections.push(("decode".to_string(), "Decode".to_string()));
            sections.push(("prefill".to_string(), "Prefill".to_string()));
            if sweep.cells.iter().any(|c| c.conc > 1) {
                sections.push(("conc".to_string(), "Concurrency".to_string()));
            }
            if sweep.cells.iter().any(|c| c.dtype == Dtype::Fp8) {
                sections.push(("fp8".to_string(), "fp8".to_string()));
            }
            if says_what_moved(sweep) {
                sections.push(("moved".to_string(), "What moved".to_string()));
            }
            if !progress.is_empty() {
                sections.push(("progress".to_string(), "Progress".to_string()));
            }
            let names = models(sweep).join(" and ");
            let torch = &sweep.torch_spyre;
            let ours = &sweep.triton_scratchy;
            (
                rsx! {
                    p {
                        "scratchy's Spyre path against torch-spyre on one Spyre card: {names}, \
                         decode, prefill, concurrency and fp8, from the A/B sweep of \
                         {sweep.date} — triton-scratchy at " code { "{ours.sha8()}" }
                        if let Some(since) = &sweep.since { " ({since} in)" }
                        ", torch-spyre measured {torch.measured} on \
                         {torch.host.as_deref().unwrap_or(\"the card\")}. Two harnesses, one \
                         card: compare ratios, not absolutes."
                    }
                    if !freshened.is_empty() {
                        p {
                            "Fresh since that sweep: " code { "{freshened.join(\", \")}" }
                            " — every cell those PRs re-measured shows the PR's value \
                             through this page, torch's value staying the sweep's; each \
                             cell's note names its PR and both numbers, and the Progress \
                             section holds every run."
                        }
                    }
                },
                tiles(sweep),
                rsx! {
                    {glance(sweep)}
                    {decode_section(sweep)}
                    {prefill_section(sweep)}
                    {conc_section(sweep)}
                    {fp8_section(sweep)}
                    {moved_section(sweep)}
                    {progress_section(&progress, &sweeps)}
                },
            )
        }
    };
    let page = rsx! {
        {chrome::header(Root(0), Some(Tab::Performance))}
        {chrome::perf_side_nav("spyre.html", &sections)}
        Content {
            Stack { gap: 6,
                Stack { gap: 4,
                    Heading { "Spyre performance" }
                    {intro}
                    {tiles}
                }
                {body}
                {about(&sweeps)}
            }
        }
        script { dangerous_inner_html: crate::charts::CHARTS_JS }
    };
    let html = chrome::document(
        Head {
            assets,
            title: "Spyre performance — scratchy",
            description: Some(
                "scratchy's Spyre path against torch-spyre on one Spyre card: decode, prefill, \
                 concurrency and fp8, per A/B sweep.",
            ),
            og: None,
            root: Root(0),
            libraries: &[Library::CarbonCharts],
        },
        page,
    )?;
    let out = site.out.join("spyre.html");
    fs::write(&out, html).map_err(|e| format!("{}: {e}", out.display()))?;
    let dest = site.out.join(DATA);
    fs::create_dir_all(&dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    for (file, _) in &sweeps {
        fs::copy(file, dest.join(file.file_name().unwrap_or_default()))
            .map_err(|e| format!("{}: {e}", file.display()))?;
    }
    if !progress.is_empty() {
        let pdest = site.out.join(PROGRESS);
        fs::create_dir_all(&pdest).map_err(|e| format!("{}: {e}", pdest.display()))?;
        for (file, _) in &progress {
            fs::copy(file, pdest.join(file.file_name().unwrap_or_default()))
                .map_err(|e| format!("{}: {e}", file.display()))?;
        }
    }
    Ok(format!(
        "spyre page: {} sweeps -> {}",
        sweeps.len(),
        out.display()
    ))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        CellKey, Dtype, Metric, Progress, Sweep, apply_progress, problems, progress_problems,
        progress_rows, progress_timelines, progress_vocabulary,
    };

    /// A sweep of `cells`, with `extra` fields after them.
    fn sweep(cells: &str, extra: &str) -> serde_json::Result<Sweep> {
        serde_json::from_str(&format!(
            r#"{{
                "schema": 1, "date": "2026-10-17", "contexts": {{ "~1k": 1051 }},
                "torch_spyre": {{ "measured": "2026-10-14", "harness": "h" }},
                "triton_scratchy": {{ "measured": "2026-10-17", "commit": "b7c4c8896", "harness": "h" }},
                "cells": [{cells}]{extra}
            }}"#
        ))
    }

    const CELL: &str = r#"{ "model": "granite-3.1-2b", "dtype": "fp16", "metric": "itl_ms",
                            "context": "~1k", "torch": 63.0, "scratchy": 46.6 }"#;

    #[test]
    fn a_clean_sweep_has_no_problems() {
        let s = sweep(CELL, "").expect("parses");
        assert!(problems(&s).is_empty(), "{:?}", problems(&s));
    }

    #[test]
    fn a_misspelt_key_fails_rather_than_dropping_out() {
        let err = sweep(CELL, r#", "lever": []"#).err().expect("refused");
        assert!(err.to_string().contains("unknown field `lever`"), "{err}");
    }

    #[test]
    fn problems_name_a_twice_measured_cell_a_zero_and_a_stray_target() {
        let zero = CELL.replace("46.6", "0.0").replace("itl_ms", "ttft_ms");
        let s = sweep(
            &format!("{CELL}, {CELL}, {zero}"),
            r#", "target": { "label": "t", "model": "granite-3.1-8b", "dtype": "fp16",
                             "metric": "itl_ms", "context": "~1k", "value": 85.0, "why": "w" }"#,
        )
        .expect("parses");
        let p = problems(&s);
        assert_eq!(p.len(), 3, "{p:?}");
        assert!(
            p[0].contains("ITL conc 1 at ~1k is in the sweep twice"),
            "{p:?}"
        );
        assert!(p[1].contains("both values must be measured"), "{p:?}");
        assert!(p[2].starts_with("target t: no cell"), "{p:?}");
    }

    /// A progress entry of `cells`, with `extra` fields after them.
    fn progress(cells: &str, extra: &str) -> serde_json::Result<Progress> {
        serde_json::from_str(&format!(
            r#"{{
                "schema": 1, "pr": 682, "title": "perf(spyre): t",
                "head": "{}", "base": "{}",
                "merged": "2026-10-21", "pod": "nickm2", "harness": "h",
                "cells": [{cells}]{extra}
            }}"#,
            "1".repeat(40),
            "2".repeat(40),
        ))
    }

    /// A progress cell whose headlines are its sides' last runs.
    const PCELL: &str = r#"{ "model": "granite-3.1-2b", "dtype": "fp16", "metric": "tok_s",
                            "conc": 8, "context": "~1k",
                            "main": 84.5, "pr": 94.62,
                            "reps": { "main": [84.12, 84.5], "pr": [93.62, 94.62] } }"#;

    #[test]
    fn a_clean_progress_entry_has_no_problems() {
        let p = progress(PCELL, "").expect("parses");
        let problems = progress_problems(&p);
        assert!(problems.is_empty(), "{problems:?}");
    }

    #[test]
    fn a_misspelt_progress_key_fails_rather_than_dropping_out() {
        let err = progress(PCELL, r#", "gate": "x""#).err().expect("refused");
        assert!(err.to_string().contains("unknown field `gate`"), "{err}");
    }

    #[test]
    fn progress_problems_name_a_cherry_picked_headline_a_single_run_and_a_short_sha() {
        // A headline that is not its side's last run: a cherry-picked rep.
        let picked = PCELL.replace(r#""main": 84.5"#, r#""main": 84.9"#);
        // One run a side, against two on the other.
        let single = PCELL.replace(
            r#""reps": { "main": [84.12, 84.5], "pr": [93.62, 94.62] } }"#,
            r#""reps": { "main": [84.5], "pr": [93.62, 94.62] } }"#,
        );
        let picked = progress_problems(&progress(&picked, "").expect("parses"));
        assert_eq!(picked.len(), 1, "{picked:?}");
        assert!(
            picked[0].contains("main headline 84.9 is not its last run (84.5)"),
            "{picked:?}"
        );
        let single = progress_problems(&progress(&single, "").expect("parses"));
        assert_eq!(single.len(), 2, "{single:?}");
        assert!(single[0].contains("main has 1 run(s)"), "{single:?}");
        assert!(
            single[1].contains("1 main run(s) against 2 pr run(s); the sides must match"),
            "{single:?}"
        );
        let mut short = progress(PCELL, "").expect("parses");
        short.head = "abc".to_string();
        let short = progress_problems(&short);
        assert_eq!(short.len(), 1, "{short:?}");
        assert!(
            short[0].contains("head abc is not a full commit sha"),
            "{short:?}"
        );
    }

    #[test]
    fn progress_vocabulary_refuses_a_model_or_context_the_sweeps_never_measured() {
        let sweeps = vec![(PathBuf::new(), sweep(CELL, "").expect("parses"))];
        let on_grid = progress(PCELL, "").expect("parses");
        assert!(progress_vocabulary(&[(PathBuf::new(), on_grid)], &sweeps).is_empty(),);
        let away = progress(
            &PCELL
                .replace("granite-3.1-2b", "granite-3.1-3b")
                .replace("~1k", "~5k"),
            "",
        )
        .expect("parses");
        let off = progress_vocabulary(&[(PathBuf::new(), away)], &sweeps);
        assert_eq!(off.len(), 2, "{off:?}");
        assert!(
            off[0].contains("model granite-3.1-3b is not one the sweeps measured"),
            "{off:?}"
        );
        assert!(
            off[1].contains("context ~5k is not one the sweeps measured"),
            "{off:?}"
        );
    }

    #[test]
    fn progress_rows_take_the_newest_source_and_never_torch_from_a_pr() {
        // The sweep (2026-10-17) holds the cell: scratchy 84.5, torch 45.0 —
        // and an ITL cell no progress entry measures, which must not grow a
        // row: the table is the cells the PRs are moving, nothing else.
        let sweeps = vec![(
            PathBuf::new(),
            sweep(
                &format!(
                    r#"{{ "model": "granite-3.1-2b", "dtype": "fp16", "metric": "tok_s",
                        "conc": 8, "context": "~1k", "torch": 45.0, "scratchy": 84.5 }}, {CELL}"#
                ),
                "",
            )
            .expect("parses"),
        )];
        // One PR merged after the sweep, one before it: only the newer one's
        // value may supersede the sweep's. Its cell carries a note, which
        // lands on the row with its value.
        let after = progress(
            &PCELL.replace(
                r#""reps": { "main": [84.12, 84.5], "pr": [93.62, 94.62] } }"#,
                r#""reps": { "main": [84.12, 84.5], "pr": [93.62, 94.62] },
                   "note": "rep 2 of a back-to-back boot" }"#,
            ),
            "",
        )
        .expect("parses");
        let mut before = progress(PCELL, "").expect("parses");
        before.merged = "2026-10-16".to_string();
        let rows = progress_rows(
            &[(PathBuf::new(), before), (PathBuf::new(), after)],
            &sweeps,
        );
        let key = (
            "granite-3.1-2b".to_string(),
            Dtype::Fp16,
            Metric::TokS,
            8,
            "~1k".to_string(),
        );
        assert_eq!(rows.len(), 1, "an unmoved sweep cell has no row");
        let row = rows.get(&key).expect("the cell has a row");
        assert_eq!(row.latest, 94.62);
        assert_eq!(row.source, "PR #682, 2026-10-21");
        // Torch comes from the sweep, whichever PR is newer.
        assert_eq!(row.torch, Some((45.0, "2026-10-17".to_string())));
        assert_eq!(row.note.as_deref(), Some("rep 2 of a back-to-back boot"));
    }

    #[test]
    fn progress_timelines_step_the_sources_and_break_torch_at_prs() {
        // Two sweeps measured the cell, then a PR after both; a second cell
        // only the PR measured grows no timeline — one point is not a line.
        let older = sweep(
            r#"{ "model": "granite-3.1-2b", "dtype": "fp16", "metric": "tok_s",
                "conc": 8, "context": "~1k", "torch": 45.0, "scratchy": 80.0 }"#,
            "",
        )
        .expect("parses");
        let mut newer = sweep(
            r#"{ "model": "granite-3.1-2b", "dtype": "fp16", "metric": "tok_s",
                "conc": 8, "context": "~1k", "torch": 45.0, "scratchy": 84.5 }"#,
            "",
        )
        .expect("parses");
        newer.date = "2026-10-18".to_string();
        // Sweeps oldest first, as load() returns them.
        let sweeps = vec![(PathBuf::new(), older), (PathBuf::new(), newer)];
        let p = progress(
            &format!("{PCELL}, {}", PCELL.replace(r#""conc": 8"#, r#""conc": 4"#)),
            "",
        )
        .expect("parses");
        let timelines = progress_timelines(&[(PathBuf::new(), p)], &sweeps);
        assert_eq!(timelines.len(), 1, "a one-source cell has no timeline");
        let key = (
            "granite-3.1-2b".to_string(),
            Dtype::Fp16,
            Metric::TokS,
            8,
            "~1k".to_string(),
        );
        let (steps, names) = timelines.get(&key).expect("the moved cell has a timeline");
        assert_eq!(*names, ["10-17", "10-18", "#682"]);
        let xs: Vec<f64> = steps.iter().map(|s| s.x).collect();
        assert_eq!(xs, vec![1.0, 2.0, 4.0]);
        let scratchy: Vec<f64> = steps.iter().filter_map(|s| s.scratchy).collect();
        assert_eq!(scratchy, vec![80.0, 84.5, 94.62]);
        // Torch's own value stops at the PR: no PR re-measures torch — the
        // chart carries its line forward from the newest sweep, dotted.
        let torch: Vec<Option<f64>> = steps.iter().map(|s| s.rival).collect();
        assert_eq!(torch, vec![Some(45.0), Some(45.0), None]);
        // The ×-factor labels ride the first and the last steps where
        // scratchy was measured — the newest being the PR's step, whose
        // factor pairs its value with the torch level carried to it.
        let highlights: Vec<bool> = steps.iter().map(|s| s.highlight).collect();
        assert_eq!(highlights, vec![true, false, true]);
    }

    #[test]
    fn apply_progress_freshens_only_cells_a_pr_re_measured() {
        // A sweep holding the cell (torch 63.0, scratchy 46.6, with a note),
        // and a second cell no progress entry touches.
        let mut s = sweep(
            &format!(
                r#"{{ "model": "granite-3.1-2b", "dtype": "fp16", "metric": "tok_s",
                    "conc": 8, "context": "~1k", "torch": 45.0, "scratchy": 84.5,
                    "note": "rep 2 of a boot." }}, {CELL}"#
            ),
            "",
        )
        .expect("parses");
        let key: CellKey = (
            "granite-3.1-2b".to_string(),
            Dtype::Fp16,
            Metric::TokS,
            8,
            "~1k".to_string(),
        );
        let row = |is_pr: bool| super::ProgressRow {
            latest: if is_pr { 94.62 } else { 90.0 },
            source: if is_pr {
                "PR #692, 2026-10-19".to_string()
            } else {
                "sweep 2026-10-19".to_string()
            },
            order: ("2026-10-19".to_string(), is_pr, if is_pr { 692 } else { 0 }),
            torch: Some((45.0, "2026-10-17".to_string())),
            note: None,
        };
        let mut rows = std::collections::BTreeMap::new();
        // A sweep-sourced row is not newer than the sweep it renders: the
        // cell stays as measured.
        rows.insert(key.clone(), row(false));
        assert!(apply_progress(&mut s, &rows).is_empty());
        assert_eq!(s.cells[0].scratchy, 84.5);
        // A PR-sourced row freshens the cell, names the PR with both
        // numbers, and leaves torch and every other cell alone.
        rows.insert(key, row(true));
        let applied = apply_progress(&mut s, &rows);
        assert_eq!(applied, ["PR #692, 2026-10-19".to_string()]);
        assert_eq!(s.cells[0].scratchy, 94.62);
        assert_eq!(s.cells[0].torch, 45.0);
        assert_eq!(
            s.cells[0].note.as_deref(),
            Some("rep 2 of a boot. PR #692, 2026-10-19: 84.5 tok/s → 94.6 tok/s.")
        );
        assert_eq!(s.cells[1].scratchy, 46.6);
        assert_eq!(s.cells[1].note, None);
    }
}

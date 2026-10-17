//! spyre.html, from site/data/spyre/*.json.
//!
//! Each data file is one A/B sweep of the two Spyre stacks — torch-spyre and
//! triton-scratchy, scratchy's Spyre path — named <date>-<sha8>.json for the
//! day of the sweep and the commit scratchy's side ran. The page charts the
//! latest sweep and lists every one in its about table, the way metal.html
//! charts every run of its bench script.

mod data;

use std::collections::BTreeSet;
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
use crate::chrome::{self, Content, Head, Library, Published, REPO, Root, Tab, Theme};
use crate::thousands_f;
use data::{Cell, Dtype, Metric, Provenance, Sweep};

/// The sweep files, relative to site/; copied to the same place under
/// _site/, where the about table links to each.
pub const DATA: &str = "data/spyre";

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

/// The largest model's short-context story: the tie both stacks sit on is a
/// floor, not a result. Its numbers are the charter's own (16.4 GB of fp16
/// weights per token; the cost model peaks at 204.8 GB/s).
const FLOOR_NOTE: &str = "Short-context parity is a shared floor: both stacks stream 16.4 GB of \
                          fp16 weights per token at ~115 GB/s. The cost model's peak is 204.8 GB/s \
                          — ≈80–90 ms if reachable — so the 8b gets the moonshot treatment, not \
                          parity.";

/// A 15 Oct fp8 run, kept as a footnote until the conc re-sweep lands.
const FP8_CONC_NOTE: &str = "2b fp8 conc-8, 15 Oct run: 56.4 vs 8.25 tok/s — 6.8× (6.8–9.9× \
                             across conc 1–8).";

/// What the latest sweep's scratchy brings over the one before it, and the
/// levers still open: the charter's state, kept with the page that reports
/// it.
const MOVED: [(&str, &str); 5] = [
    (
        "#566–#568",
        "granite coherent on main at 2b and 8b, fp16 and fp8 — logits ≥ 8192 are now written, \
         the first decode step no longer panics, and 8b fp8 stops reading unwritten o_proj \
         copies.",
    ),
    (
        "#573",
        "dxp chunk explorer restored (#299 had disabled it for every group): 2b 10k TTFT \
         18.1 → 8.9 s; 8b 10k ITL 305 → 233 ms.",
    ),
    ("#569", "K/V kept in LX for decode: 2b ITL −14–15% at 10k."),
    (
        "#583",
        "hd128 decode group attention: 8b 10k ITL 233.8 → 192.3 ms, past #502's 204–225 ms \
         band.",
    ),
    (
        "#563 + #570",
        "2b fp8 bake now 407 s cold (was 1,185 s), with a guard against regression.",
    ),
];

const LEVERS: [(&str, &str); 5] = [
    (
        "8b moonshot",
        "break the shared short-context floor: measure the card's real HBM ceiling and the 8b \
         step's per-matmul bandwidth; testing whether 512-row prefill chunks halve 8b's \
         per-token MLP cost.",
    ),
    (
        "8b 10k decode",
        "torch-spyre 1.17× ahead — hd128 K/V gather kept in LX, projected 192 → ~165–170 ms.",
    ),
    (
        "2b ~1k TTFT",
        "torch-spyre 1.48× ahead — a 27-row tail chunk pays a whole 256-row program.",
    ),
    (
        "8b 10k TTFT",
        "torch-spyre 1.30× ahead — attention plus the MLP's pointwise glue.",
    ),
    (
        "conc-8",
        "not re-measured since the fixes; a full matched re-sweep is pending a free pod.",
    ),
];

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
        if sweep.schema != 1 {
            errors.push(format!("{name}: schema {} is not 1", sweep.schema));
        }
        for c in &sweep.cells {
            if let Some(bucket) = c
                .context
                .as_deref()
                .filter(|k| !sweep.contexts.contains_key(*k))
            {
                errors.push(format!(
                    "{name}: cell context {bucket} is not one of the sweep's"
                ));
            }
        }
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
            theme: "white",
            toolbar: Enabled { enabled: false },
        },
    };
    chart(&spec, label, "width: 100%", None)
}

/// One model's sparkline panel: its cells as steps along the context axis.
struct Panel<'a> {
    model: String,
    cells: Vec<&'a Cell>,
    steps: Vec<Step>,
}

/// Every model's values along the context buckets, scratchy against
/// torch-spyre, the gap filled blue ahead and red behind.
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
    };
    let mut panels: Vec<Panel> = models(sweep)
        .into_iter()
        .map(|m| {
            let cs = cells(sweep, &m, dtype, metric, 1);
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
                model: m,
                cells: cs,
                steps,
            }
        })
        .collect();
    for p in &mut panels {
        highlight(p.steps.iter_mut().map(|st| (metric.better(), st)));
    }
    rsx! {
        div { style: "display: flex; flex-wrap: wrap; gap: 1rem 1.5rem",
            for p in &panels {
                div { style: "flex: 1 1 16rem; min-width: 16rem",
                    Stack { gap: 1,
                        strong { "{size_name(&p.model)}" }
                        if p.steps.len() >= 2 {
                            {sparkline(
                                &p.steps,
                                axes,
                                SCRATCHY,
                                TORCH,
                                Some(p.cells.iter().map(|c| c.context.clone().unwrap_or_default()).collect()),
                                "5rem",
                                &format!(
                                    "{}, {} by context, triton-scratchy against torch-spyre",
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

/// The glance's one row: the card every sweep ran on, as metal.html's rows
/// are its machines.
const CARD: &str = "IBM Spyre";

/// The glance's heat maps as metal.html draws its own: a cell per model, its
/// prefill grid then its decode grid, each its dtypes down and the context
/// buckets across, one request at a time.
fn glance(sweep: &Sweep) -> Element {
    let (models, contexts) = (models(sweep), contexts(sweep));
    let dtypes: BTreeSet<Dtype> = sweep
        .cells
        .iter()
        .filter(|c| c.conc == 1)
        .map(|c| c.dtype)
        .collect();
    let stems: Vec<&str> = models.iter().map(String::as_str).collect();
    let metrics = [(Metric::Ttft, "Prefill"), (Metric::Itl, "Decode")];
    let mini = |model: &str, metric: Metric, name: &str| -> Element {
        let grid = TableAxes {
            across: ("context", contexts.clone()),
            down: (
                "dtype",
                dtypes.iter().map(|d| d.label().to_string()).collect(),
            ),
        };
        let squares = dtypes
            .iter()
            .flat_map(|&dtype| {
                contexts.iter().map(move |c| RatioSquare {
                    across: c.clone(),
                    down: dtype.label().to_string(),
                    ratio: cell(sweep, model, dtype, metric, 1, c)
                        .map(Cell::ratio)
                        .filter(|r| r.is_finite()),
                    highlight: false,
                })
            })
            .collect();
        glance_mini(
            grid,
            squares,
            format!("scratchy/{}", TORCH.line),
            name,
            &format!("{model} {}", metric.label()),
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
                            ContentSwitcherItem { target: slug(DECODE_SPARKS), selected: false, "{DECODE_SPARKS}" }
                            ContentSwitcherItem { target: slug(PREFILL_SPARKS), selected: false, "{PREFILL_SPARKS}" }
                        }
                        div {
                            ContentPanel { id: slug(HEAT_MAPS), shown: true,
                                Stack { gap: 5,
                                    p { style: "contain: inline-size",
                                        "Every model's dtype × context grid: in each cell, the model's prefill \
                                         (time to first token) grid, then its decode (inter-token latency) \
                                         grid (rows: fp16 over fp8; columns: context, short to long), one \
                                         request at a time. Colour is how many times faster triton-scratchy is \
                                         than torch-spyre. Blue faster, red slower, the middle step within \
                                         noise. Hover a square for its ratio; a striped square is a cell the \
                                         sweep did not measure (fp8 past short context: torch-spyre's path is \
                                         broken there)."
                                    }
                                    {glance_maps(&[CARD], &stems, |_, model| rsx! {
                                        for (metric, name) in metrics { {mini(model, metric, name)} }
                                    })}
                                }
                            }
                            ContentPanel { id: slug(DECODE_SPARKS), shown: false,
                                div { style: "contain: inline-size",
                                    Stack { gap: 4,
                                        p { "triton-scratchy against torch-spyre as context grows, fp16, one \
                                             request at a time. Blue fill where triton-scratchy is ahead, red \
                                             where it is behind; each fill carries its × factor." }
                                        {spark_panel(sweep, Dtype::Fp16, Metric::Itl)}
                                    }
                                }
                            }
                            ContentPanel { id: slug(PREFILL_SPARKS), shown: false,
                                div { style: "contain: inline-size",
                                    Stack { gap: 4,
                                        p { "Time to first token as prompts grow, fp16, one request at a \
                                             time. Blue fill where triton-scratchy is ahead, red where it is \
                                             behind; each fill carries its × factor." }
                                        {spark_panel(sweep, Dtype::Fp16, Metric::Ttft)}
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
/// titles, and the notes it ends with.
struct SectionSpec {
    id: &'static str,
    heading: &'static str,
    xlabel: &'static str,
    ylabel: &'static str,
    title: fn(&str) -> String,
    notes: &'static [&'static str],
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
                        for n in spec.notes { p { "{n}" } }
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
    notes: &[FLOOR_NOTE],
};

/// The prefill section's shape: time to first token, fp16.
const PREFILL: SectionSpec = SectionSpec {
    id: "prefill",
    heading: "Prefill",
    xlabel: "prompt bucket",
    ylabel: "TTFT ms",
    title: |size| format!("Prefill TTFT, {size} (ms, lower is better)"),
    notes: &[],
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
             scales the ~1k cell's time by prompt length; the ~123 cell sits below that base \
             (" {DASH} ")."
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
            let notes: Vec<&str> = rows
                .iter()
                .filter_map(|(_, _, _, c)| c.note.as_deref())
                .collect();
            let notes: Vec<&str> = {
                let mut seen: BTreeSet<&str> = BTreeSet::new();
                notes.into_iter().filter(|n| seen.insert(*n)).collect()
            };
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
                        for n in notes { p { "{n}" } }
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
                    }
                  }
                }
            }
        }
    }
}

/// fp8: short context only, where torch-spyre's path works.
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
    let notes: Vec<&str> = {
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        all.iter()
            .filter_map(|c| c.note.as_deref())
            .filter(|n| seen.insert(*n))
            .collect()
    };
    rsx! {
        Section { id: "fp8", level: 2,
            Stack { gap: 4,
                Heading { "fp8" }
                p {
                    "fp8 weights, one request at a time, short context — torch-spyre's fp8 path \
                     is broken past it (~10k hangs, TTFT floor 32–55 s), so short context is the \
                     only fair cell. {lead}. Lower is better."
                }
                Tile {
                  Layer { level: 1,
                    Stack { gap: 5,
                        if !groups.is_empty() {
                            {bars(
                                &groups,
                                ("fp8 decode ITL, short context (ms, lower is better)", "model", "ITL ms"),
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
                        for n in notes { p { "{n}" } }
                        p { "{FP8_CONC_NOTE}" }
                    }
                  }
                }
            }
        }
    }
}

fn moved_section(sweep: &Sweep) -> Element {
    let commit = sweep.triton_scratchy.commit.as_deref().unwrap_or("main");
    rsx! {
        Section { id: "moved", level: 2,
            Stack { gap: 4,
                Heading { "What moved" }
                p { "triton-scratchy at main " code { "{commit}" } ", #563–#583 merged, since the \
                     previous sweep." }
                Table { headers: headers(["change", "what it did"]),
                    for (pr, what) in MOVED {
                        TableRow { TableCell { "{pr}" } TableCell { "{what}" } }
                    }
                }
                Section { level: 3,
                    Stack { gap: 3,
                        Heading { "Open levers" }
                        Table { headers: headers(["lever", "open work"]),
                            for (lever, work) in LEVERS {
                                TableRow { TableCell { "{lever}" } TableCell { "{work}" } }
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
/// the floor the largest model sits on.
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
    // The moonshot: the largest model's shared short-context floor.
    if let Some((m, cs)) = with(Metric::Itl, Dtype::Fp16, 1, false) {
        let c = cs[0];
        out.push(tile(
            format!("{} moonshot · short-context ITL", size_name(&m)),
            "~85 ms".to_string(),
            "cost-model peak 204.8 GB/s — both stacks stream 16.4 GB/token at ~115 GB/s"
                .to_string(),
            Some(format!(
                "today's {} tie is a shared floor, not a target",
                fmt_value(Metric::Itl, c.scratchy)
            )),
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
                    "torch-spyre and triton-scratchy are measured by their own harnesses — vllm \
                     bench serve for torch-spyre, scr serve for triton-scratchy — at slightly \
                     different prompt lengths, so cells sit in context buckets and the page \
                     compares ratios, not absolutes. Each sweep's JSON, with both sides' \
                     provenance, is linked below."
                }
                p {
                    code { "ITL" } " inter-token latency · " code { "TTFT" } " time to first \
                     token · " code { "conc" } " concurrent requests · " code { "tok/s" } " \
                     decoded tokens per second · " code { "÷ perfect" } " triton-scratchy's \
                     ratio to its own ideal: short-context ITL held flat (decode), the ~1k \
                     cell's TTFT scaled by prompt length (prefill)."
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
    // The page charts the latest sweep; the about table lists every one.
    let latest = sweeps.last().map(|(_, s)| s);
    let mut sections: Vec<(String, String)> = Vec::new();
    let (intro, tiles, body) = match latest {
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
        Some(sweep) => {
            sections.push(("glance".to_string(), "At a glance".to_string()));
            sections.push(("decode".to_string(), "Decode".to_string()));
            sections.push(("prefill".to_string(), "Prefill".to_string()));
            if sweep.cells.iter().any(|c| c.conc > 1) {
                sections.push(("conc".to_string(), "Concurrency".to_string()));
            }
            if sweep.cells.iter().any(|c| c.dtype == Dtype::Fp8) {
                sections.push(("fp8".to_string(), "fp8".to_string()));
            }
            sections.push(("moved".to_string(), "What moved".to_string()));
            let names = models(sweep).join(" and ");
            let torch = &sweep.torch_spyre;
            let ours = &sweep.triton_scratchy;
            (
                rsx! {
                    p {
                        "scratchy's Spyre path against torch-spyre on one Spyre card: {names}, \
                         decode, prefill, concurrency and fp8, from the A/B sweep of \
                         {sweep.date} — triton-scratchy at " code { "{ours.sha8()}" } " \
                         (#563–#583 in), torch-spyre measured {torch.measured} on \
                         {torch.host.as_deref().unwrap_or(\"the card\")}. Two harnesses, one \
                         card: compare ratios, not absolutes."
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
                 concurrency and fp8 on granite-3.1 2b and 8b, per A/B sweep.",
            ),
            og: None,
            root: Root(0),
            libraries: &[Library::CarbonCharts],
            theme: Theme::AlwaysLight,
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
    Ok(format!(
        "spyre page: {} sweeps -> {}",
        sweeps.len(),
        out.display()
    ))
}

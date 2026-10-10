//! metal.html, from site/data/metal/*.json.
//!
//! Each data file is one run of scripts/bench_metal_matrix.sh, copied in
//! unedited and named <machine>-<YYYY-MM-DDTHHMMSSZ>-<sha8>.json (older runs:
//! <machine>-<YYYY-MM-DD>-<sha8>.json). The page reads them at site-build
//! time, so it cannot drift from what the runner measured.
//!
//! Per machine, each model shows its newest run; older runs of the same model
//! stay visible in that model's history table, so a re-run after a Metal
//! change reads as a before/after.

mod data;
mod numbers;

use std::collections::BTreeMap;
use std::fs;

use dioxus::prelude::*;
use serde::Serialize;

use crate::Site;
use crate::carbon::{
    Alignment, Column, Definition, Fold, Grid, Heading, Layer, Section, Span, Stack, Table,
    TableCell, TableRow, Tag, TagKind, Tile, Toggletip,
};
use crate::chrome::{self, Content, Head, Library, Published, REPO, Root, Tab, Theme};
use data::{At, Cell, Model, Run, Scenario};
use numbers::{
    Engine, Metric, NO_STREAM, NOISE, PARTIAL, Startup, cell, cfmt, fmt, lfmt, med, partial,
    served, timing, untimed_note,
};

/// The run files, relative to site/; copied to the same place under _site/,
/// where each run table links to its own.
pub const DATA: &str = "data/metal";

/// Draws the line charts from their rendered specs.
const CHARTS_JS: &str = include_str!("charts.js");

// --------------------------------------------------------------------- load

/// Every run file, checked. Fails naming each bad file, so one build reports
/// all of them rather than the first.
fn load(site: &Site) -> Result<Vec<Run>, String> {
    let dir = site.root.join(DATA);
    let mut files: Vec<_> = fs::read_dir(&dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    let (mut runs, mut errors) = (Vec::new(), Vec::new());
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
        let mut run: Run = match serde_json::from_str(&text) {
            Ok(r) => r,
            Err(e) => {
                errors.push(format!("{name}: {e}"));
                continue;
            }
        };
        // The start time keeps two runs of one commit on one day apart; names
        // from before it carry only the date.
        let (machine, sha8) = (slug(run.machine.chip.as_str()), run.repo.sha8());
        let want = format!("{machine}-{}-{sha8}.json", run.generated_utc.stamp());
        if name != want && name != format!("{machine}-{}-{sha8}.json", run.generated_utc.date()) {
            errors.push(format!(
                "{name}: name it {want} (<machine>-<start time>-<sha8>.json)"
            ));
        }
        run.file = f;
        runs.push(run);
    }
    if errors.is_empty() {
        Ok(runs)
    } else {
        Err(errors
            .iter()
            .map(|e| format!("metal page: {e}"))
            .collect::<Vec<_>>()
            .join("\n"))
    }
}

fn slug(text: &str) -> String {
    let lower = text.to_lowercase();
    let parts: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_lowercase() && !c.is_ascii_digit())
        .filter(|p| !p.is_empty())
        .collect();
    parts.join("-")
}

// ------------------------------------------------------------------ indexing

/// One model's result within one run.
#[derive(Clone, Copy)]
struct Entry<'a> {
    run: &'a Run,
    m: &'a Model,
}

/// One machine's runs, oldest first, and per model (in order of first
/// appearance) every entry, oldest first: the last is the one shown.
struct MachineRuns<'a> {
    runs: Vec<&'a Run>,
    models: Vec<(&'a str, Vec<Entry<'a>>)>,
}

impl<'a> MachineRuns<'a> {
    fn latest(&self, stem: &str) -> Option<Entry<'a>> {
        self.models
            .iter()
            .find(|(s, _)| *s == stem)
            .and_then(|(_, es)| es.last().copied())
    }
}

/// Per machine, by chip name. Newer runs win, so a model shows its newest numbers.
fn index(runs: &[Run]) -> BTreeMap<&str, MachineRuns<'_>> {
    let mut by_time: Vec<&Run> = runs.iter().collect();
    by_time.sort_by(|a, b| a.generated_utc.cmp(&b.generated_utc));
    let mut machines: BTreeMap<&str, MachineRuns> = BTreeMap::new();
    for run in by_time {
        let mr = machines
            .entry(run.machine.chip.as_str())
            .or_insert_with(|| MachineRuns {
                runs: Vec::new(),
                models: Vec::new(),
            });
        mr.runs.push(run);
        for m in &run.models {
            let e = Entry { run, m };
            match mr.models.iter_mut().find(|(s, _)| *s == m.stem) {
                Some((_, es)) => es.push(e),
                None => mr.models.push((&m.stem, vec![e])),
            }
        }
    }
    machines
}

// ------------------------------------------------------------------- render

/// A change badge against the previous run: a Carbon tag, green better, red
/// worse, grey inside the noise band. The arrow and the tag's title say it
/// too, so colour is never the only signal; nothing when either side is
/// missing.
fn delta(new: Option<f64>, old: Option<f64>, higher_better: bool) -> Element {
    let (Some(new), Some(old)) = (new, old.filter(|o| *o != 0.0)) else {
        return rsx! {};
    };
    let ch = (new - old) / old;
    if ch.abs() < NOISE {
        return rsx! { Tag { kind: TagKind::Gray, title: "within 3% of the previous run", "≈" } };
    }
    let better = (ch > 0.0) == higher_better;
    let (kind, word) = if better {
        (TagKind::Green, "better")
    } else {
        (TagKind::Red, "worse")
    };
    let arrow = if ch > 0.0 { "▲" } else { "▼" };
    let pct = format!("{:.0}", ch.abs() * 100.0);
    rsx! { Tag { kind, title: "{word} than the previous run", "{arrow} {pct}%" } }
}

/// The summary row's numbers, unformatted, for comparing two runs, each with
/// whether higher is better (only tok/s is; every time and memory is lower).
fn summary_raw(m: &Model) -> [(Option<f64>, bool); 8] {
    let ladder = Engine::Scratchy.ladder(m);
    let one = cell(Engine::Scratchy.cells(m), At::Conc(1));
    [
        (med(ladder, Scenario::Frozen, Startup::Ready), false),
        (med(ladder, Scenario::Cold, Startup::Ready), false),
        (med(ladder, Scenario::Cold, Startup::FirstToken), false),
        (med(ladder, Scenario::Warm, Startup::FirstToken), false),
        (timing(one, Metric::Ttft), false),
        (timing(one, Metric::Tpot), false),
        (timing(one, Metric::Throughput), true),
        (med(ladder, Scenario::Cold, Startup::PeakRss), false),
    ]
}

/// The run settings that change scratchy's numbers on their own: its serve
/// flags and KV cache setting, as the runner recorded them.
fn settings(run: &Run) -> [(&'static str, &str); 2] {
    fn or<'a>(v: &'a Option<String>, unset: &'static str) -> &'a str {
        v.as_deref().filter(|s| !s.is_empty()).unwrap_or(unset)
    }
    let c = &run.config;
    [
        ("serve flags", or(&c.scratchy_serve_args, "none")),
        ("KV cache", or(&c.kv_cache_dtype, "default")),
    ]
}

/// Human-readable differences in settings between two runs, or "".
fn settings_changed(run: &Run, prev: &Run) -> String {
    settings(run)
        .iter()
        .zip(settings(prev))
        .filter(|((_, now), (_, then))| now != then)
        .map(|((k, now), (_, then))| format!("{k} {then} then, {now} now"))
        .collect::<Vec<_>>()
        .join("; ")
}

fn info(label: &str, body: Element) -> Element {
    rsx! { Toggletip { label, alignment: Alignment::Bottom, {body} } }
}

fn startup_info() -> Element {
    info(
        "",
        rsx! {
            strong { "frozen" } ": first launch after memory and scratchy's weights cache are cleared. "
            strong { "cold" } ": a normal relaunch. Both run until the server is ready, no request included. "
            strong { "1st req" } ": the cold server's first request, send to first token. "
            strong { "warm" } ": the median over many requests once it is running. ollama reports ready before \
                          loading the model, so its load time lands in 1st req."
        },
    )
}

fn timing_info() -> Element {
    info(
        "",
        rsx! {
            strong { "no stream" } ": the answer arrived in one piece, so TTFT and TPOT could not be timed. "
            strong { "†" } ": some answers did; the timing comes from the rest (the table views say how many). mlx-lm and \
                       ollama can stop answers early, which flatters tok/s."
        },
    )
}

fn rss_info() -> Element {
    info(
        "",
        rsx! { "The server process at its peak. For ollama, " code { "ollama serve" } " only; its runner process is not counted." },
    )
}

fn users_info() -> Element {
    info(
        "",
        rsx! { "Requests in flight at once. An engine may batch fewer than it is offered." },
    )
}

fn about() -> Element {
    info(
        "About these numbers",
        rsx! {
            "Prompts are made up, a fixed size, and unique per request so no cache can answer them, with \
             greedy decoding (temperature 0) on every engine. mlx-lm runs the same MLX checkpoint as scratchy; \
             ollama runs its own GGUF quantization, named on its row. scratchy's build time is shown per model \
             and is never part of startup. The (i) buttons next to a column explain it."
        },
    )
}

fn opt(v: Option<u32>) -> String {
    v.map_or("?".to_string(), |v| v.to_string())
}

fn headers<const N: usize>(names: [&str; N]) -> Vec<String> {
    names.map(String::from).to_vec()
}

/// A part of a model's tile: its heading, then what explains it, then it.
fn part(heading: &str, intro: Element, body: Element) -> Element {
    rsx! {
        Section { level: 4,
            Stack { gap: 3,
                Heading { "{heading}" }
                p { {intro} }
                {body}
            }
        }
    }
}

fn summary_table(m: &Model, run: &Run, prev: Option<Entry>) -> Element {
    let shape = run
        .config
        .base()
        .map(|b| format!("{} in / {} out", opt(b.input), opt(b.output)));
    let rows = Engine::ALL.into_iter().filter_map(|e| {
        let (ladder, one) = (e.ladder(m), cell(e.cells(m), At::Conc(1)));
        if e == Engine::Scratchy && !served(m) {
            let why = if m.built {
                "the server never served"
            } else {
                "build failed"
            };
            return Some(rsx! {
                TableRow {
                    TableCell { "{e.label()}" }
                    TableCell { colspan: 8, "No numbers: {why}. See the run's raw logs." }
                }
            });
        }
        if e != Engine::Scratchy && ladder.is_empty() && e.cells(m).is_empty() {
            return None; // that engine was not part of this run
        }
        let vals = [
            lfmt(ladder, Scenario::Frozen, Startup::Ready, 2, 1.0),
            lfmt(ladder, Scenario::Cold, Startup::Ready, 2, 1.0),
            lfmt(ladder, Scenario::Cold, Startup::FirstToken, 2, 1.0),
            lfmt(ladder, Scenario::Warm, Startup::FirstToken, 0, 1000.0),
            cfmt(one, Metric::Ttft),
            cfmt(one, Metric::Tpot),
            fmt(timing(one, Metric::Throughput), 1),
            fmt(med(ladder, Scenario::Cold, Startup::PeakRss), 0),
        ];
        let now = summary_raw(m);
        let then = prev
            .filter(|_| e == Engine::Scratchy)
            .map(|p| summary_raw(p.m));
        // ollama runs its own GGUF: name it on its row.
        let name = match m.ollama.as_ref().filter(|_| e == Engine::Ollama) {
            None => e.label().to_string(),
            Some(o) => {
                let tag = o.tag.as_deref().unwrap_or_default();
                match o.quantization.as_deref().filter(|q| !q.is_empty()) {
                    Some(q) => format!("{} ({tag}, {q})", e.label()),
                    None => format!("{} ({tag})", e.label()),
                }
            }
        };
        Some(rsx! {
            TableRow {
                TableCell { "{name}" }
                for (i, v) in vals.iter().enumerate() {
                    TableCell { "{v} "
                        if let Some(then) = then { {delta(now[i].0, then[i].0, now[i].1)} }
                    }
                }
            }
        })
    });
    let changed = prev
        .map(|p| settings_changed(run, p.run))
        .unwrap_or_default();
    part(
        "Startup and single-user speed",
        rsx! {
            "One row per engine. Startup in seconds (warm in ms); one user at {shape.as_deref().unwrap_or(\"the base shape\")}."
            if let Some(p) = prev {
                " Under scratchy: change since its previous run here ({p.run.generated_utc.date()}, "
                code { "{p.run.repo.sha8()}" }
                "); ≈ is within 3%."
                if !changed.is_empty() {
                    " " strong { "Settings differ from that run" } " ({changed}), so a change is not the code alone."
                }
            }
            " What the columns mean: startup " {startup_info()} " one user " {timing_info()} " peak RSS " {rss_info()}
        },
        rsx! {
            Table {
                headers: headers(["engine", "frozen s", "cold s", "1st req s", "warm ms", "TTFT ms", "TPOT ms", "tok/s", "peak RSS MiB"]),
                {rows}
            }
        },
    )
}

/// One line of a chart: its label, series colour, and (x, value) points; a
/// point present but None was not measured.
struct Series {
    label: String,
    class: &'static str,
    pts: Vec<(u32, Option<f64>)>,
}

/// A series' colour, fixed per engine: blue, orange, aqua from the data-viz
/// reference palette, validated as a set (all pairs, CVD and normal vision)
/// against Carbon's white and g100 surfaces. Aqua is under 3:1 on white, so
/// every chart has a legend and a table view. The previous run is a light
/// grey so it reads as behind.
fn colour(class: &str) -> &'static str {
    match class {
        "s1" => "#2a78d6",
        "s2" => "#d95926",
        "s3" => "#1baf7a",
        _ => "#a8a8a8",
    }
}

/// What charts.js hands Carbon Charts: which chart, its points, and its
/// options, in Carbon Charts' own field names.
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum Chart<'a> {
    Line {
        data: Vec<LinePoint<'a>>,
        options: ChartOptions<'a>,
    },
    Heatmap {
        data: Vec<HeatPoint>,
        options: ChartOptions<'a>,
        /// The values are log2(ratio): charts.js shows them as "×1.23".
        ratio: bool,
    },
}

/// One measured point of a line. A None value (present but not measured: no
/// stream) leaves a gap.
#[derive(Serialize)]
struct LinePoint<'a> {
    group: &'a str,
    key: u32,
    value: Option<f64>,
}

/// One cell of a heatmap: x is answer tokens, y prompt tokens.
#[derive(Serialize)]
struct HeatPoint {
    x: String,
    y: String,
    value: Option<f64>,
}

#[derive(Serialize)]
struct ChartOptions<'a> {
    title: String,
    axes: Axes<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    color: Option<Colour<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    heatmap: Option<HeatmapOptions>,
    height: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    width: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    legend: Option<Enabled>,
    #[serde(skip_serializing_if = "Option::is_none")]
    locale: Option<Locale>,
    // The page is always Carbon's light theme.
    theme: &'static str,
    toolbar: Enabled,
}

#[derive(Serialize)]
struct Axes<'a> {
    bottom: Axis<'a>,
    left: Axis<'a>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Axis<'a> {
    maps_to: &'static str,
    title: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    scale_type: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ticks: Option<Ticks<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include_zero: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    visible: Option<bool>,
    /// A label axis's labels in order; Carbon puts the first at the left of
    /// a horizontal axis and at the bottom of a vertical one.
    #[serde(skip_serializing_if = "Option::is_none")]
    domain: Option<Vec<String>>,
}

/// A grid's heatmap axes, read like a table: answer tokens across, prompt
/// tokens down, each smallest first, so the smallest prompt is on top.
fn grid_axes<'a>(ins: &[u32], outs: &[u32], visible: Option<bool>) -> Axes<'a> {
    let labels = |maps_to, title, domain: Vec<String>| Axis {
        maps_to,
        title,
        scale_type: Some("labels"),
        ticks: None,
        include_zero: None,
        visible,
        domain: Some(domain),
    };
    Axes {
        bottom: labels(
            "x",
            "answer tokens",
            outs.iter().map(u32::to_string).collect(),
        ),
        // Carbon draws a vertical axis's first label at the bottom.
        left: labels(
            "y",
            "prompt tokens",
            ins.iter().rev().map(u32::to_string).collect(),
        ),
    }
}

#[derive(Serialize)]
struct Ticks<'a> {
    values: &'a [u32],
}

/// Carbon Charts' colour options: a colour per series, or the steps of a
/// heatmap's quantized scale.
#[derive(Serialize)]
#[serde(untagged)]
enum Colour<'a> {
    Scale {
        scale: BTreeMap<&'a str, &'static str>,
    },
    Gradient {
        gradient: Gradient,
    },
}

#[derive(Serialize)]
struct Gradient {
    colors: &'static [&'static str],
}

/// The widest speedup the ratio heatmaps tell apart, either way: ×4 faster
/// and ×0.25 (4× slower). Anything beyond takes the end colour.
const RATIO_SPAN: f64 = 4.0;

/// The ratio heatmaps plot log2 of the ratio, so a factor of two is the same
/// distance from ×1 whichever way it goes: ±log2(RATIO_SPAN).
const RATIO_RANGE: Domain = Domain {
    min: -2.0,
    max: 2.0,
};

/// The ratio heatmaps' 33 steps, slowest to fastest, from Carbon's red, gray
/// and cyan. 33 equal steps over ±2 put ×0.96 to ×1.04 (about the run-to-run
/// noise band) in the gray 40 middle step. Carbon's own diverging palette
/// fades to white in the middle, down to 1.0:1 against the page and tiles, so
/// the cells near ×1, most of them, vanish; here the steps past the middle
/// start at red/cyan 40 and darken outward to 100, every one at least 2.37:1
/// on white and 2.15:1 on the grey tile (#f4f4f4).
const RATIO_STEPS: [&str; 33] = [
    "#2d0709", // red 100: ×0.25 or slower
    "#520408", // red 90
    "#520408", // red 90
    "#750e13", // red 80
    "#750e13", // red 80
    "#a2191f", // red 70
    "#a2191f", // red 70
    "#a2191f", // red 70
    "#da1e28", // red 60
    "#da1e28", // red 60
    "#da1e28", // red 60
    "#fa4d56", // red 50
    "#fa4d56", // red 50
    "#fa4d56", // red 50
    "#ff8389", // red 40
    "#ff8389", // red 40
    "#a8a8a8", // gray 40: ×0.96 to ×1.04, within noise
    "#33b1ff", // cyan 40
    "#33b1ff", // cyan 40
    "#1192e8", // cyan 50
    "#1192e8", // cyan 50
    "#1192e8", // cyan 50
    "#0072c3", // cyan 60
    "#0072c3", // cyan 60
    "#0072c3", // cyan 60
    "#00539a", // cyan 70
    "#00539a", // cyan 70
    "#00539a", // cyan 70
    "#003a6d", // cyan 80
    "#003a6d", // cyan 80
    "#012749", // cyan 90
    "#012749", // cyan 90
    "#061727", // cyan 100: ×4 or faster
];

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct HeatmapOptions {
    color_legend: ColorLegend,
    #[serde(skip_serializing_if = "Option::is_none")]
    color_domain: Option<Domain>,
}

#[derive(Serialize)]
struct ColorLegend {
    title: String,
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Serialize)]
struct Domain {
    min: f64,
    max: f64,
}

/// Carbon's heatmap tooltip lists the axis values, then the cell's value
/// under `locale.translations.total` (Carbon's default: "Total"; it outranks
/// `tooltip.totalLabel`).
#[derive(Serialize)]
struct Locale {
    translations: Translations,
}

#[derive(Serialize)]
struct Translations {
    total: String,
}

impl Locale {
    fn total(label: String) -> Option<Self> {
        Some(Locale {
            translations: Translations { total: label },
        })
    }
}

#[derive(Serialize)]
struct Enabled {
    enabled: bool,
}

/// A fixed-size chart's box: wider than the screen, it scrolls sideways.
const FIXED_MOUNT: &str = "max-width: 100%; overflow-x: auto";

/// A chart's mount point, styled by `mount`: charts.js draws `spec` into it.
/// `scale` is the id of the ratio scale a heatmap's squares highlight on hover.
fn chart(spec: &Chart, label: &str, mount: &str, scale: Option<&str>) -> Element {
    rsx! {
        div { role: "img", "aria-label": label, style: "{mount}", "data-scale": scale,
            "data-chart": serde_json::to_string(spec).unwrap_or_default(),
        }
    }
}

/// One line chart: offered users on x, one line per engine.
fn line_chart(series: &[Series], xs: &[u32], title: &str, ylabel: &str, label: &str) -> Element {
    let spec = Chart::Line {
        data: series
            .iter()
            .flat_map(|s| {
                s.pts.iter().map(|&(x, v)| LinePoint {
                    group: &s.label,
                    key: x,
                    // Carbon prints values as given: round to what the tables show.
                    value: v.map(|v| (v * 10.0).round() / 10.0),
                })
            })
            .collect(),
        options: ChartOptions {
            title: title.to_string(),
            axes: Axes {
                bottom: Axis {
                    maps_to: "key",
                    title: "offered concurrent users",
                    scale_type: Some("linear"),
                    ticks: Some(Ticks { values: xs }),
                    include_zero: None,
                    visible: None,
                    domain: None,
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
                scale: series
                    .iter()
                    .map(|s| (s.label.as_str(), colour(s.class)))
                    .collect(),
            }),
            heatmap: None,
            height: "280px".to_string(),
            width: None,
            legend: None,
            locale: None,
            theme: "white",
            toolbar: Enabled { enabled: false },
        },
    };
    chart(&spec, label, FIXED_MOUNT, None)
}

/// A model's full grid as a heatmap. With a rival, values are how many times
/// faster scratchy is (as log2), coloured over RATIO_RANGE by RATIO_STEPS. Without one,
/// scratchy's own values on Carbon's sequential palette.
fn grid_heatmap(
    title: String,
    (ins, outs): (&[u32], &[u32]),
    data: Vec<HeatPoint>,
    total_label: String,
    ratio: bool,
    label: &str,
    scale: Option<&str>,
) -> Element {
    let spec = Chart::Heatmap {
        data,
        options: ChartOptions {
            title,
            axes: grid_axes(ins, outs, None),
            color: ratio.then_some(Colour::Gradient {
                gradient: Gradient {
                    colors: &RATIO_STEPS,
                },
            }),
            heatmap: Some(HeatmapOptions {
                color_legend: ColorLegend {
                    // Shown only without a rival; ratio grids use ratio_scale.
                    title: total_label.clone(),
                    kind: "quantize",
                },
                color_domain: ratio.then_some(RATIO_RANGE),
            }),
            // Room for the axes (and Carbon's legend when it shows), then
            // about 64px a cell.
            height: format!("{}px", if ratio { 100 } else { 160 } + 56 * ins.len()),
            width: Some(format!("{}px", 140 + 72 * outs.len())),
            legend: ratio.then_some(Enabled { enabled: false }),
            locale: Locale::total(total_label),
            theme: "white",
            toolbar: Enabled { enabled: false },
        },
        ratio,
    };
    chart(&spec, label, FIXED_MOUNT, scale)
}

/// One small multiple: a model's whole grid as a tiny heatmap, colour only;
/// no title, axes or legend, the table around it names it.
fn mini_heatmap(
    (ins, outs): (&[u32], &[u32]),
    data: Vec<HeatPoint>,
    total_label: String,
    label: &str,
) -> Element {
    let spec = Chart::Heatmap {
        data,
        options: ChartOptions {
            title: String::new(),
            axes: grid_axes(ins, outs, Some(false)),
            color: Some(Colour::Gradient {
                gradient: Gradient {
                    colors: &RATIO_STEPS,
                },
            }),
            heatmap: Some(HeatmapOptions {
                color_legend: ColorLegend {
                    title: String::new(),
                    kind: "quantize",
                },
                color_domain: Some(RATIO_RANGE),
            }),
            // Its box decides: square, half its cell.
            height: "100%".to_string(),
            width: Some("100%".to_string()),
            legend: Some(Enabled { enabled: false }),
            locale: Locale::total(total_label),
            theme: "white",
            toolbar: Enabled { enabled: false },
        },
        ratio: true,
    };
    chart(
        &spec,
        label,
        "width: 100%; aspect-ratio: 1",
        Some(GLANCE_SCALE),
    )
}

/// The ratio heatmaps' colour scale, upright: RATIO_STEPS fastest at the top,
/// labelled at the ends and the middle. Carbon's heatmap only draws its legend
/// as a bar underneath.
/// The overview's ratio scale.
const GLANCE_SCALE: &str = "scale-glance";

fn ratio_scale(fit: ScaleFit, id: &str) -> Element {
    let rows = RATIO_STEPS.len();
    let middle = rows / 2 + 1;
    let (place, track) = match fit {
        ScaleFit::Fixed => (String::new(), "0.5rem"),
        ScaleFit::Beside => (
            // Stretched to the rows beside it: from the top of the first
            // row's grids (below its model names, 1.5rem) to the bottom of the
            // last's (above its two label lines, 2.25rem). Wrapped onto its
            // own line on a narrow screen, it keeps a readable height.
            "align-self: stretch; margin-block: 1.5rem 2.25rem; min-height: 12rem; ".to_string(),
            "1fr",
        ),
    };
    rsx! {
        div { id: "{id}", style: "{place}display: grid; grid-template-columns: 0.75rem max-content; grid-template-rows: repeat({rows}, {track}); column-gap: 0.5rem; font-size: 0.75rem; line-height: 1",
            // Fastest at the top; `data-step` counts from the slowest, the way
            // the heatmap's colours do, so charts.js can find a square's step.
            for (row, colour) in RATIO_STEPS.iter().rev().enumerate() {
                div { style: "grid-column: 1; grid-row: {row + 1}; background: {colour}",
                    "data-step": "{rows - 1 - row}",
                }
            }
            div { style: "grid-column: 2; grid-row: 1; align-self: start",
                Definition { alignment: Alignment::Left,
                    definition: "scratchy at least {RATIO_SPAN} times as fast as mlx-lm (ollama where a run has no mlx-lm); anything faster takes this colour too.",
                    strong { "×{RATIO_SPAN}" }
                }
            }
            div { style: "grid-column: 2; grid-row: {middle}; align-self: center",
                Definition { alignment: Alignment::Left,
                    definition: "The same speed, within run-to-run noise: ×0.96 to ×1.04.",
                    strong { "×1" }
                }
            }
            div { style: "grid-column: 2; grid-row: {rows}; align-self: end",
                Definition { alignment: Alignment::Left,
                    definition: "scratchy at most a quarter as fast as mlx-lm (ollama where a run has no mlx-lm), {RATIO_SPAN} times slower; anything slower takes this colour too.",
                    strong { "×{1.0 / RATIO_SPAN}" }
                }
            }
        }
    }
}

/// Where the ratio scale goes: fixed-size steps beside a model's grids, or
/// stretched to the height of the overview's rows beside it.
#[derive(Clone, Copy, PartialEq)]
enum ScaleFit {
    Fixed,
    Beside,
}

/// A ratio as a heatmap value: log2, so a factor of two either way is the
/// same distance from ×1.
fn log2_ratio(r: f64) -> f64 {
    (r.log2() * 1000.0).round() / 1000.0
}

/// A cell's value for a table view, with how many requests went untimed when
/// the timing is partial.
fn table_value(c: Option<&Cell>, metric: Metric) -> String {
    if partial(c, metric) {
        format!("{} ({})", cfmt(c, metric), untimed_note(c))
    } else {
        cfmt(c, metric)
    }
}

fn conc_cells(cells: &[Cell]) -> BTreeMap<u32, &Cell> {
    cells
        .iter()
        .filter_map(|c| {
            if let At::Conc(n) = c.at {
                Some((n, c))
            } else {
                None
            }
        })
        .collect()
}

/// Throughput and time per output token against offered users, side by side:
/// two measures, so two charts on one x axis rather than two y axes.
fn conc_chart(m: &Model, run: &Run, prev: Option<Entry>) -> Element {
    let cells: Vec<(Engine, BTreeMap<u32, &Cell>)> = Engine::ALL
        .iter()
        .map(|&e| (e, conc_cells(e.cells(m))))
        .collect();
    let mut xs: Vec<u32> = cells
        .iter()
        .flat_map(|(_, cs)| cs.keys().copied())
        .collect();
    xs.sort_unstable();
    xs.dedup();
    let Some(&last) = xs.last() else {
        return rsx! {};
    };

    // The previous run's scratchy line, drawn first so it sits behind.
    let pcs = prev
        .map(|p| conc_cells(Engine::Scratchy.cells(p.m)))
        .unwrap_or_default();
    let plabel = prev.map_or(String::new(), |p| {
        let differ = if settings_changed(run, p.run).is_empty() {
            ")"
        } else {
            ", different settings)"
        };
        format!(
            "scratchy, previous ({}{differ}",
            p.run.generated_utc.month_day()
        )
    });
    let series = |metric: Metric| -> Vec<Series> {
        let mut out = Vec::new();
        if !pcs.is_empty() {
            out.push(Series {
                label: plabel.clone(),
                class: "prev",
                pts: xs
                    .iter()
                    .filter_map(|x| Some((*x, timing(Some(pcs.get(x)?), metric))))
                    .collect(),
            });
        }
        for (e, cs) in &cells {
            if !cs.is_empty() {
                out.push(Series {
                    label: e.label().to_string(),
                    class: e.series(),
                    pts: xs
                        .iter()
                        .filter_map(|x| Some((*x, timing(Some(cs.get(x)?), metric))))
                        .collect(),
                });
            }
        }
        out.retain(|s| s.pts.iter().any(|p| p.1.is_some()));
        out
    };
    let (tput, tpot) = (series(Metric::Throughput), series(Metric::Tpot));

    let mut charts: Vec<Element> = Vec::new();
    if !tput.is_empty() {
        charts.push(line_chart(
            &tput,
            &xs,
            "Throughput (tok/s, higher is better)",
            "tok/s",
            "Output tokens per second against offered concurrent users, per engine",
        ));
    }
    if !tpot.is_empty() {
        // An engine that never streamed has no TPOT; say so rather than let
        // its line silently vanish from this chart.
        let gone: Vec<&str> = tput
            .iter()
            .filter(|s| s.class != "prev" && !tpot.iter().any(|t| t.label == s.label))
            .map(|s| s.label.as_str())
            .collect();
        charts.push(rsx! {
            {line_chart(
                &tpot,
                &xs,
                "Time per output token (ms, lower is better)",
                "TPOT ms",
                "Median time per output token against offered concurrent users, per engine",
            )}
            if !gone.is_empty() {
                p { "{gone.join(\", \")}: {NO_STREAM}, not plotted." }
            }
        });
    }
    let users = match xs.split_last() {
        Some((l, rest)) if !rest.is_empty() => format!(
            "{} and {l}",
            rest.iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => last.to_string(),
    };
    let base = run.config.base();
    let (input, output) = (
        opt(base.and_then(|b| b.input)),
        opt(base.and_then(|b| b.output)),
    );
    let mut table_headers = vec!["offered users".to_string()];
    table_headers.extend(xs.iter().map(u32::to_string));
    part(
        "As users are added",
        rsx! {
            "Throughput and time per output token with {users} users at once " {users_info()}
            "; {input}-token prompts, {output}-token answers. A line that rises in the right-hand chart means each user's answer slows down as more share the engine."
        },
        rsx! {
            Grid {
                for c in charts { Column { span: Span::HALF, {c} } }
            }
            Fold { title: "Table view",
                Table { headers: table_headers,
                    for (e, cs) in cells.iter().filter(|(_, cs)| !cs.is_empty()) {
                        for (metric, name) in [(Metric::Throughput, "tok/s"), (Metric::Ttft, "TTFT ms"), (Metric::Tpot, "TPOT ms")] {
                            TableRow {
                                TableCell { "{e.label()} · {name}" }
                                for x in &xs { TableCell { {table_value(cs.get(x).copied(), metric)} } }
                            }
                        }
                    }
                }
            }
        },
    )
}

/// One engine's grid cells, by (prompt, answer) tokens.
type ByShape<'a> = BTreeMap<(u32, u32), &'a Cell>;

/// A model's prompt-by-answer grid: the axes, each engine's cells, and the
/// engines scratchy is compared against.
struct ShapeGrid<'a> {
    ins: &'a [u32],
    outs: &'a [u32],
    have: Vec<(Engine, ByShape<'a>)>,
    /// mlx-lm, which runs the same MLX checkpoint; failing that ollama.
    rival: Option<Engine>,
    /// The other of the two, when the run has both.
    other: Option<Engine>,
}

impl<'a> ShapeGrid<'a> {
    /// None when scratchy has no grid.
    fn of(m: &'a Model, run: &'a Run) -> Option<Self> {
        let sc = run.config.scaling.as_ref()?;
        let (ins, outs) = (
            sc.grid_input.as_deref().unwrap_or_default(),
            sc.grid_output.as_deref().unwrap_or_default(),
        );
        let have: Vec<_> = Engine::ALL
            .iter()
            .map(|&e| {
                let cs = e
                    .cells(m)
                    .iter()
                    .filter_map(|c| {
                        if let At::Grid { input, output } = c.at {
                            Some(((input, output), c))
                        } else {
                            None
                        }
                    })
                    .collect();
                (e, cs)
            })
            .collect();
        let g = ShapeGrid {
            ins,
            outs,
            have,
            rival: None,
            other: None,
        };
        if g.cells(Engine::Scratchy).is_empty() || ins.is_empty() || outs.is_empty() {
            return None;
        }
        let present: Vec<Engine> = Engine::RIVALS
            .into_iter()
            .filter(|&e| !g.cells(e).is_empty())
            .collect();
        Some(ShapeGrid {
            rival: present.first().copied(),
            other: present.get(1).copied(),
            ..g
        })
    }

    fn cells(&self, e: Engine) -> &ByShape<'a> {
        &self
            .have
            .iter()
            .find(|(h, _)| *h == e)
            .expect("every engine has an entry")
            .1
    }

    fn at(&self, e: Option<Engine>, rung: (u32, u32)) -> Option<&'a Cell> {
        e.and_then(|e| self.cells(e).get(&rung).copied())
    }

    /// scratchy against `rival` at one cell: the ratio, and how it reads.
    fn versus(&self, metric: Metric, rung: (u32, u32), rival: Engine) -> (Option<f64>, String) {
        let (cs, cr) = (
            self.at(Some(Engine::Scratchy), rung),
            self.at(Some(rival), rung),
        );
        let s = timing(cs, metric);
        match metric.faster(s, timing(cr, metric)) {
            Some(r) => (Some(r), format!("×{r:.2}{}", mark(metric, &[cs, cr]))),
            None if s.is_none() => (None, why(cs, "scratchy")),
            None => (None, why(cr, rival.label())),
        }
    }
}

const GRID_METRICS: [(Metric, &str); 2] = [
    (Metric::Throughput, "throughput"),
    (Metric::Ttft, "time to first token"),
];

fn mark(metric: Metric, cells: &[Option<&Cell>]) -> &'static str {
    if cells.iter().any(|c| partial(*c, metric)) {
        PARTIAL
    } else {
        ""
    }
}

/// Why a cell has no ratio.
fn why(c: Option<&Cell>, name: &str) -> String {
    if c.is_some() {
        format!("{name}: {NO_STREAM}")
    } else {
        format!("no {name}")
    }
}

/// One model's grid for one metric, coloured by scratchy against its rival.
fn heat_map(g: &ShapeGrid, metric: Metric, title: &str, scale: &str) -> Element {
    let mut data = Vec::new();
    for &i in g.ins {
        for &o in g.outs {
            let value = match g.rival {
                Some(rival) => g.versus(metric, (i, o), rival).0.map(log2_ratio),
                // Nothing to compare against: scratchy's own value.
                None => timing(g.at(Some(Engine::Scratchy), (i, o)), metric),
            };
            data.push(HeatPoint {
                x: o.to_string(),
                y: i.to_string(),
                value,
            });
        }
    }
    let (heading, total_label) = match g.rival {
        Some(r) => (
            format!("{title}, scratchy vs {}", r.label()),
            format!("scratchy/{}", r.label()),
        ),
        None => (
            format!("{title} ({})", metric.unit()),
            format!("scratchy, {}", metric.unit()),
        ),
    };
    grid_heatmap(
        heading,
        (g.ins, g.outs),
        data,
        total_label,
        g.rival.is_some(),
        title,
        g.rival.is_some().then_some(scale),
    )
}

/// The grid's numbers for one metric: per cell the ratio against the rival,
/// then against the other engine; just scratchy's value with neither.
fn heat_table(g: &ShapeGrid, metric: Metric, title: &str) -> Element {
    let mut cols = vec![format!("{title}: prompt ↓ / answer →")];
    cols.extend(g.outs.iter().map(u32::to_string));
    rsx! {
        Table { headers: cols,
            for &i in g.ins {
                TableRow {
                    TableCell { "{i}" }
                    for &o in g.outs {
                        {
                            let text = match g.rival {
                                None => cfmt(g.at(Some(Engine::Scratchy), (i, o)), metric),
                                Some(rival) => {
                                    let mut t = g.versus(metric, (i, o), rival).1;
                                    if let Some(other) = g.other {
                                        t += &format!(" (vs {} {})", other.label(), g.versus(metric, (i, o), other).1);
                                    }
                                    t
                                }
                            };
                            rsx! { TableCell { "{text}" } }
                        }
                    }
                }
            }
        }
    }
}

fn grid_maps(m: &Model, mid: &str, run: &Run) -> Element {
    let scale = format!("scale-{mid}");
    let Some(g) = ShapeGrid::of(m, run) else {
        return rsx! {};
    };
    let conc = opt(run.config.base().and_then(|b| b.conc));
    let note = match g.rival {
        None => {
            "scratchy's own values; this run has no mlx-lm or ollama to compare against".to_string()
        }
        Some(rival) => {
            let missing = Engine::RIVALS.into_iter().find(|&e| g.cells(e).is_empty());
            format!(
                "prompt size down the side, answer size across the bottom. Colour is how many times faster scratchy is than {}{}: blue is faster, red is slower, the middle step is within noise. Hover a cell for its ratio; the table view has every cell's ratio{}",
                rival.label(),
                missing.map_or(String::new(), |e| format!(
                    " (this run has no {})",
                    e.label()
                )),
                g.other.map_or(String::new(), |o| format!(
                    " and the same against {}",
                    o.label()
                )),
            )
        }
    };
    part(
        "Prompt size × answer size",
        rsx! { "{conc} users at once; {note}." },
        rsx! {
            // Side by side while they fit; wrapped onto new lines on a
            // narrow screen.
            div { style: "display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: flex-start",
                for (metric, title) in GRID_METRICS { {heat_map(&g, metric, title, &scale)} }
                if g.rival.is_some() { {ratio_scale(ScaleFit::Fixed, &scale)} }
            }
            Fold { title: "Table view",
                Stack { gap: 4,
                    for (metric, title) in GRID_METRICS { {heat_table(&g, metric, title)} }
                }
            }
        },
    )
}

fn commit(run: &Run) -> Element {
    rsx! { a { href: "{REPO}/commit/{run.repo.sha}", code { "{run.repo.sha8()}" } } }
}

fn history(entries: &[Entry]) -> Element {
    if entries.len() < 2 {
        return rsx! {};
    }
    rsx! {
        Fold { title: "Runs of this model ({entries.len()})",
            Table {
                headers: headers(["run", "scratchy", "quant", "cold s", "1st req s", "warm ms", "tok/s, 1 user", "tok/s, most users", "TPOT ms, most users"]),
                for Entry { run, m } in entries.iter().rev() {
                    TableRow {
                        TableCell { "{run.generated_utc.date()}" }
                        TableCell { {commit(run)} }
                        TableCell { "{m.quant.as_deref().unwrap_or_default()}" }
                        if served(m) {
                            {
                                let (ladder, cells) = (Engine::Scratchy.ladder(m), Engine::Scratchy.cells(m));
                                let one = cell(cells, At::Conc(1));
                                let top = cells.iter().filter_map(|c| if let At::Conc(n) = c.at { Some(n) } else { None }).max().filter(|&n| n != 0);
                                let many = top.and_then(|n| cell(cells, At::Conc(n)));
                                let vals = [
                                    fmt(med(ladder, Scenario::Cold, Startup::Ready), 2),
                                    fmt(med(ladder, Scenario::Cold, Startup::FirstToken), 2),
                                    fmt(med(ladder, Scenario::Warm, Startup::FirstToken).map(|v| v * 1000.0), 0),
                                    fmt(timing(one, Metric::Throughput), 1),
                                    fmt(timing(many, Metric::Throughput), 1),
                                    cfmt(many, Metric::Tpot),
                                ];
                                rsx! { for v in vals { TableCell { "{v}" } } }
                            }
                        } else {
                            TableCell { colspan: 6, "no numbers" }
                        }
                    }
                }
            }
        }
    }
}

fn runs_table(runs: &[&Run]) -> Element {
    rsx! {
        Fold { title: "Runs on this machine ({runs.len()})",
            Table {
                headers: headers(["when", "scratchy", "steps", "cold priming launches", "scratchy KV cache", "scratchy serve flags", "data"]),
                for run in runs.iter().rev() {
                    {
                        let [(_, flags), (_, kv)] = settings(run);
                        let file = run.file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                        rsx! {
                            TableRow {
                                TableCell { "{run.generated_utc.minute()} UTC" }
                                TableCell { {commit(run)} }
                                TableCell { "{run.config.scenarios.join(\", \")}" }
                                TableCell { {run.config.cold_priming_launches.map_or("not recorded".to_string(), |n| n.to_string())} }
                                TableCell { "{kv}" }
                                TableCell { code { "{flags}" } }
                                TableCell { a { href: "{DATA}/{file}", "json" } }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Every model's prompt-by-answer grid on every machine as small multiples,
/// one table per metric: a machine per row, a model per column, each cell that
/// model's grid as a tiny heatmap coloured like the full ones.
fn glance(machines: &BTreeMap<&str, MachineRuns>) -> Element {
    let mut stems: Vec<&str> = Vec::new();
    for mr in machines.values() {
        for (stem, _) in &mr.models {
            if !stems.contains(stem) {
                stems.push(stem);
            }
        }
    }
    if stems.is_empty() {
        return rsx! {};
    }
    let mini = |chip: &str, stem: &str, metric: Metric, name: &str| -> Element {
        let Some(Entry { run, m }) = machines[chip].latest(stem) else {
            return rsx! { "not run" };
        };
        let Some(g) = ShapeGrid::of(m, run) else {
            return rsx! { "no grid" };
        };
        let Some(rival) = g.rival else {
            return rsx! { "nothing to compare" };
        };
        let mut ratios = Vec::new();
        let mut data = Vec::new();
        for &i in g.ins {
            for &o in g.outs {
                let r = g.versus(metric, (i, o), rival).0;
                ratios.extend(r);
                data.push(HeatPoint {
                    x: o.to_string(),
                    y: i.to_string(),
                    value: r.map(log2_ratio),
                });
            }
        }
        let range = match (
            ratios.iter().copied().reduce(f64::min),
            ratios.iter().copied().reduce(f64::max),
        ) {
            (Some(lo), Some(hi)) => format!("×{lo:.2}–×{hi:.2}"),
            _ => "no ratios".to_string(),
        };
        rsx! {
            // Half its cell, however wide the cell grows.
            div { style: "flex: 1 1 0; min-width: 0",
                Stack { gap: 2,
                    {mini_heatmap((g.ins, g.outs), data, format!("scratchy/{}", rival.label()), &format!("{stem} on {chip}: scratchy {range}"))}
                    // Name over range, centred under the grid.
                    div { style: "text-align: center",
                        div { strong { "{name}" } }
                        div { "{range}" }
                    }
                }
            }
        }
    };
    // Each machine a wrapping row: its name in a fixed-width label, then a cell
    // per model (its name over its prefill and decode grids), 11.5rem wide or
    // more. On a wide screen the rows line up as columns at 11.5rem; on a
    // narrow one the cells wrap, each growing to fill its line, the grids in
    // it with it, so nothing scrolls sideways and nothing sits in a corner. The scale stands beside the rows, or
    // wraps below them.
    let metrics = [(Metric::Ttft, "Prefill"), (Metric::Throughput, "Decode")];
    rsx! {
        Section { id: "glance", level: 2,
            Stack { gap: 4,
                Heading { "At a glance" }
                p {
                    "Every model's prompt size × answer size grid on every machine: in each cell, the model's \
                     prefill (time to first token) grid, then its decode (throughput) grid (rows: prompt, short to \
                     long; columns: answer, short to long). Colour is how many times faster scratchy is than mlx-lm \
                     (ollama where a run has no mlx-lm): blue faster, red slower, the middle step within noise. \
                     Hover a square for its ratio."
                }
                // The tile shrink-wraps the set instead of spanning the page.
                Tile { style: "width: fit-content; max-width: 100%",
                    div { style: "display: flex; flex-wrap: wrap; gap: 1.5rem 2rem",
                        div { style: "display: flex; flex-direction: column; gap: 1.5rem; min-width: 0",
                            for chip in machines.keys() {
                                div { style: "display: flex; flex-wrap: wrap; gap: 1rem 1.5rem; align-items: flex-start",
                                    div { style: "flex: none; width: 7rem", "{chip}" }
                                    for stem in &stems {
                                        // At least two 88px grids and the 8px between them.
                                        div { style: "flex: 1 1 11.5rem; min-width: 11.5rem; display: flex; flex-direction: column; gap: 0.5rem",
                                            div { style: "text-align: center", "{stem}" }
                                            div { style: "display: flex; gap: 0.5rem; align-items: flex-start",
                                                for (metric, name) in metrics { {mini(chip, stem, metric, name)} }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        {ratio_scale(ScaleFit::Beside, GLANCE_SCALE)}
                    }
                }
            }
        }
    }
}

fn machine_section(chip: &str, mr: &MachineRuns) -> Element {
    let mslug = slug(chip);
    let mc = &mr.runs[mr.runs.len() - 1].machine;
    let cores = match mc.cores_performance.filter(|&p| p != 0) {
        Some(p) => format!(
            "{} cores ({p}P + {}E)",
            mc.cores_total,
            mc.cores_efficiency.unwrap_or(0)
        ),
        None => format!("{} cores", mc.cores_total),
    };
    rsx! {
        Section { id: "{mslug}", level: 2,
            Stack { gap: 4,
                Heading { "{chip}" }
                p { "{cores} · {mc.memory_gb} GB unified memory · macOS {mc.macos}" }
                Tile {
                  Layer { level: 1,
                    Stack { gap: 5,
                {runs_table(&mr.runs)}
                for (stem, seen) in &mr.models {
                    {
                        let Entry { run, m } = seen[seen.len() - 1];
                        // Before/after: the newest earlier run of this model, here, that scratchy served.
                        let prev = seen[..seen.len() - 1].iter().rev().find(|e| served(e.m)).copied();
                        let mid = format!("{mslug}-{}", slug(stem));
                        let f = m.footprint.as_ref();
                        let build = match f.and_then(|f| f.build_seconds) {
                            None => String::new(),
                            Some(s) => {
                                let binary = f.and_then(|f| f.binary_bytes).filter(|&b| b != 0)
                                    .map_or(String::new(), |b| format!(", {} MiB binary", (b as f64 / 1_048_576.0).round_ties_even()));
                                format!(" · scratchy build {s} s{binary}")
                            }
                        };
                        rsx! {
                            Tile { id: mid.clone(),
                              Section { level: 3,
                              Layer { level: 2,
                                Stack { gap: 6,
                                    Stack { gap: 2,
                                        Heading { "{stem} on {chip}" }
                                        p {
                                            a { href: "https://huggingface.co/{m.model_id}", "{m.model_id}" }
                                            " · {m.quant.as_deref().unwrap_or(\"default\")}{build} · run {run.generated_utc.date()}, "
                                            code { "{run.repo.sha8()}" }
                                        }
                                    }
                                    {summary_table(m, run, prev)}
                                    {conc_chart(m, run, prev)}
                                    {grid_maps(m, &mid, run)}
                                    {history(seen)}
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
            }
        }
    }
}

pub fn build(site: &Site, assets: &Published) -> Result<String, String> {
    let runs = load(site)?;
    let machines = index(&runs);
    let mut sections: Vec<(String, String)> = Vec::new();
    let body = if machines.is_empty() {
        rsx! {
            p {
                "No runs published yet. Run " code { "scripts/bench_metal_matrix.sh" } " and copy its JSON into "
                code { "site/data/metal/" } "."
            }
        }
    } else {
        sections.push(("glance".to_string(), "At a glance".to_string()));
        sections.extend(machines.keys().map(|chip| (slug(chip), chip.to_string())));
        rsx! {
            {glance(&machines)}
            for (chip, mr) in &machines { {machine_section(chip, mr)} }
        }
    };
    let (count, n_machines) = (runs.len(), machines.len());
    let page = rsx! {
        {chrome::header(Root(0), Some(Tab::Performance))}
        {chrome::perf_side_nav("metal.html", &sections)}
        Content {
            div {
              Stack { gap: 6,
                Stack { gap: 4,
                    Heading { "Metal performance" }
                    p {
                        "scratchy against mlx-lm and ollama on Apple silicon: how long each takes to start, how \
                         fast it answers one user, and how it holds up as prompts, answers and users grow. \
                         {count} runs on {n_machines} machines, every number measured by "
                        a { href: "{REPO}/blob/main/scripts/bench_metal_matrix.sh", code { "scripts/bench_metal_matrix.sh" } }
                        "."
                    }
                    div { {about()} }
                }
                {body}
              }
            }
        }
        script { dangerous_inner_html: CHARTS_JS }
    };
    let html = chrome::document(
        Head {
            assets,
            title: "Metal performance — scratchy",
            description: Some(
                "scratchy against mlx-lm and ollama on Apple silicon: startup, single-user speed, and scaling, per machine.",
            ),
            og: None,
            root: Root(0),
            libraries: &[Library::CarbonCharts],
            theme: Theme::AlwaysLight,
        },
        page,
    )?;
    let out = site.out.join("metal.html");
    fs::write(&out, html).map_err(|e| format!("{}: {e}", out.display()))?;
    let dest = site.out.join(DATA);
    fs::create_dir_all(&dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    for run in &runs {
        if let Some(name) = run.file.file_name() {
            fs::copy(&run.file, dest.join(name))
                .map_err(|e| format!("{}: {e}", run.file.display()))?;
        }
    }
    Ok(format!(
        "metal page: {count} runs on {n_machines} machines -> {}",
        out.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::{RATIO_RANGE, RATIO_SPAN, RATIO_STEPS};

    #[test]
    fn ratio_range_is_the_span_either_way() {
        assert_eq!(RATIO_RANGE.max, RATIO_SPAN.log2());
        assert_eq!(RATIO_RANGE.min, -RATIO_SPAN.log2());
    }

    #[test]
    fn ratio_steps_mirror_around_a_gray_middle() {
        let n = RATIO_STEPS.len();
        assert_eq!(n % 2, 1);
        assert_eq!(RATIO_STEPS[n / 2], "#a8a8a8");
    }
}

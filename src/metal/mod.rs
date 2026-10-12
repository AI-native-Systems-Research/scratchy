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

use std::collections::{BTreeMap, BTreeSet};
use std::fs;

use dioxus::prelude::*;
use serde::Serialize;

use crate::Site;
use crate::carbon::{
    Alignment, Column, ContentPanel, ContentSwitcher, ContentSwitcherItem, Definition, Fold, Grid,
    Heading, Layer, Section, Span, Stack, Table, TableCell, TableRow, Tag, TagKind, Tile,
    Toggletip,
};
use crate::chrome::{self, Content, Head, Library, Published, REPO, Root, Tab, Theme};
use data::{At, Bytes, Cell, Model, Run, Scenario};
use numbers::{
    Better, DASH, Engine, Metric, NO_STREAM, NOISE, PARTIAL, Startup, cell, cfmt, fmt, lfmt, med,
    partial, served, timing, untimed_note,
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
        for m in run.models.iter().filter(|m| m.weights().is_none()) {
            errors.push(format!(
                "{name}: no weights size for {}; add it to WEIGHTS in data.rs",
                m.model_id
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
    /// Carbon's GroupedBarChart: a bar per group, side by side, at each key.
    #[serde(rename = "groupedbar")]
    GroupedBar {
        data: Vec<BarPoint<'a>>,
        options: ChartOptions<'a>,
    },
    Heatmap {
        data: Vec<HeatPoint>,
        options: ChartOptions<'a>,
        /// The values are log2(ratio): charts.js shows them as "×1.23".
        ratio: bool,
    },
    /// A sparkline: Carbon's ComboChart, two lines and the fills between them.
    Combo {
        data: Vec<SparkPoint>,
        options: ChartOptions<'a>,
    },
}

/// One point of a sparkline. On the two lines, an engine's value; on the
/// fills, also the band from the lower line to the upper. A None value breaks
/// the line or fill there.
#[derive(Serialize)]
struct SparkPoint {
    group: &'static str,
    key: f64,
    value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    upper: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lower: Option<f64>,
    /// On a highlighted step's fill: how many times faster scratchy is there,
    /// which charts.js writes in the fill.
    #[serde(skip_serializing_if = "Option::is_none")]
    factor: Option<f64>,
}

/// One measured bar. A None value (present but not measured: no stream)
/// leaves no bar.
#[derive(Serialize)]
struct BarPoint<'a> {
    group: &'a str,
    key: String,
    value: Option<f64>,
}

/// One cell of a heatmap: x is answer tokens, y prompt tokens. A highlight
/// has its factor written in it (charts.js).
#[derive(Serialize)]
struct HeatPoint {
    x: String,
    y: String,
    value: Option<f64>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    highlight: bool,
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
    #[serde(rename = "comboChartTypes", skip_serializing_if = "Option::is_none")]
    combo: Option<Vec<ComboLayer>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bounds: Option<Bounds>,
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
    ticks: Option<Ticks>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include_zero: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    visible: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    domain: Option<AxisDomain>,
}

/// An axis's fixed extent.
#[derive(Serialize)]
#[serde(untagged)]
enum AxisDomain {
    /// A label axis's labels in order; Carbon puts the first at the left of a
    /// horizontal axis and at the bottom of a vertical one.
    Labels(Vec<String>),
    /// A numeric axis's [from, to]. Given, it is not rounded out: Carbon's
    /// own log range goes to powers of ten (1 to 16 users became 1 to 100,
    /// the line stopping 60% of the way across).
    Values([f64; 2]),
}

/// One layer of a combo chart: a chart type and the series it draws.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ComboLayer {
    #[serde(rename = "type")]
    kind: &'static str,
    corresponding_datasets: Vec<&'static str>,
}

/// Which point fields a bounded area fills between. Carbon draws bounds only
/// for an area layer holding a single series.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Bounds {
    upper_bound_maps_to: &'static str,
    lower_bound_maps_to: &'static str,
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
        domain: Some(AxisDomain::Labels(domain)),
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
struct Ticks {
    values: Vec<f64>,
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

/// A chart's mount point, styled by `mount`: charts.js draws `spec` into it.
/// `scale` is the id of the ratio scale a heatmap's squares highlight on hover.
fn chart(spec: &Chart, label: &str, mount: &str, scale: Option<&str>) -> Element {
    rsx! {
        div { role: "img", "aria-label": label, style: "{mount}", "data-scale": scale,
            "data-chart": serde_json::to_string(spec).unwrap_or_default(),
        }
    }
}

/// One bar chart: offered users on x, a bar per engine at each.
fn bar_chart(series: &[Series], xs: &[u32], title: &str, ylabel: &str, label: &str) -> Element {
    let spec = Chart::GroupedBar {
        data: series
            .iter()
            .flat_map(|s| {
                s.pts.iter().map(|&(x, v)| BarPoint {
                    group: &s.label,
                    key: x.to_string(),
                    // Carbon prints values as given: round to what the tables show.
                    value: v.map(|v| (v * 10.0).round() / 10.0),
                })
            })
            .collect(),
        options: ChartOptions {
            title: title.to_string(),
            axes: Axes {
                // Each user count a category, in the order measured.
                bottom: Axis {
                    maps_to: "key",
                    title: "offered concurrent users",
                    scale_type: Some("labels"),
                    ticks: None,
                    include_zero: None,
                    visible: None,
                    domain: Some(AxisDomain::Labels(xs.iter().map(u32::to_string).collect())),
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
            combo: None,
            bounds: None,
            theme: "white",
            toolbar: Enabled { enabled: false },
        },
    };
    chart(&spec, label, "width: 100%", None)
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
            // about 64px a cell. Carbon sets its box to this width.
            height: format!("{}px", if ratio { 100 } else { 160 } + 56 * ins.len()),
            width: Some(format!("{}px", 140 + 72 * outs.len())),
            legend: ratio.then_some(Enabled { enabled: false }),
            locale: Locale::total(total_label),
            combo: None,
            bounds: None,
            theme: "white",
            toolbar: Enabled { enabled: false },
        },
        ratio,
    };
    // Narrower on a narrow screen. Never a scroll box: scrolling one way
    // makes the other scroll too, and Carbon's tooltip, drawn inside the box,
    // then grows a scrollbar that resizes the chart under the pointer, over
    // and over.
    chart(&spec, label, "max-width: 100%", scale)
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
            combo: None,
            bounds: None,
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

/// The RATIO_STEPS colour a ratio heatmap gives a ratio, as Carbon's quantize
/// scale picks it: RATIO_RANGE in equal slices, beyond it the end colours.
fn ratio_step(r: f64) -> usize {
    let n = RATIO_STEPS.len();
    let at = (log2_ratio(r) - RATIO_RANGE.min) / (RATIO_RANGE.max - RATIO_RANGE.min);
    // Truncation is the floor here: clamped to 0 first.
    ((at.max(0.0) * n as f64) as usize).min(n - 1)
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
        charts.push(bar_chart(
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
            {bar_chart(
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

/// A dimension a sparkline follows, with the metric it mostly drives.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Trend {
    UsersDecode,
    UsersPrefill,
    PromptPrefill,
    AnswerDecode,
}

impl Trend {
    fn metric(self) -> Metric {
        match self {
            Trend::UsersDecode | Trend::AnswerDecode => Metric::Throughput,
            Trend::UsersPrefill | Trend::PromptPrefill => Metric::Ttft,
        }
    }

    fn title(self) -> &'static str {
        match self {
            Trend::UsersDecode => "Concurrency scaling — Decode",
            Trend::UsersPrefill => "Concurrency scaling — Prefill",
            Trend::PromptPrefill => "Input length scaling — Prefill",
            Trend::AnswerDecode => "Output length scaling — Decode",
        }
    }

    /// The x axis's title: not printed (styles.css), but Carbon's tooltip
    /// label for the x value.
    fn along(self) -> &'static str {
        match self {
            Trend::UsersDecode | Trend::UsersPrefill => "users at once",
            Trend::PromptPrefill => "prompt tokens",
            Trend::AnswerDecode => "answer tokens",
        }
    }

    fn axes(self) -> SparkAxes {
        let metric = self.metric();
        SparkAxes {
            better: metric.better(),
            along: self.along(),
            value: if metric == Metric::Ttft {
                "TTFT ms"
            } else {
                metric.unit()
            },
        }
    }

    /// At each step of the dimension, scratchy's value and its rival's, and
    /// the rival; None with nothing to compare. Prompt and answer lengths are
    /// each measured at three lengths of the other: a step's value is the
    /// geometric mean over those.
    fn points(self, m: &Model, run: &Run) -> Option<(Engine, Vec<Step>)> {
        let metric = self.metric();
        match self {
            Trend::UsersDecode | Trend::UsersPrefill => {
                let mine = conc_cells(Engine::Scratchy.cells(m));
                let rival = Engine::RIVALS
                    .into_iter()
                    .find(|e| !conc_cells(e.cells(m)).is_empty())?;
                let theirs = conc_cells(rival.cells(m));
                let pts = mine
                    .iter()
                    .map(|(n, c)| Step {
                        x: f64::from(*n),
                        scratchy: timing(Some(c), metric),
                        rival: timing(theirs.get(n).copied(), metric),
                        highlight: false,
                    })
                    .collect();
                Some((rival, pts))
            }
            Trend::PromptPrefill | Trend::AnswerDecode => {
                let g = ShapeGrid::of(m, run)?;
                let rival = g.rival?;
                let prompt = self == Trend::PromptPrefill;
                let (along, across) = if prompt {
                    (g.ins, g.outs)
                } else {
                    (g.outs, g.ins)
                };
                let pts = along
                    .iter()
                    .map(|&a| {
                        let at = |e: Engine| {
                            geometric_mean(across.iter().filter_map(|&b| {
                                timing(g.at(Some(e), if prompt { (a, b) } else { (b, a) }), metric)
                            }))
                        };
                        Step {
                            x: f64::from(a),
                            scratchy: at(Engine::Scratchy),
                            rival: at(rival),
                            highlight: false,
                        }
                    })
                    .collect();
                Some((rival, pts))
            }
        }
    }
}

/// One step along a sparkline's dimension: where, each engine's value, and
/// whether it is one of its panel's highlights (its factor written in its
/// fill).
#[derive(Clone, Copy)]
struct Step {
    x: f64,
    scratchy: Option<f64>,
    rival: Option<f64>,
    highlight: bool,
}

/// The geometric mean, the right average for rates and times; None of nothing.
fn geometric_mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let (sum, n) = values.fold((0.0, 0u32), |(s, n), v| (s + v.ln(), n + 1));
    (n > 0).then(|| (sum / f64::from(n)).exp())
}

/// The lines' points, and the two fills' points: each fill present where its
/// engine is ahead, the two meeting where the lines cross (found on the log
/// x axis, on which the chart draws straight segments).
fn spark_points(pts: &[Step], better: Better, rival: Engine) -> Vec<SparkPoint> {
    let (scratchy, scratchy_ahead) = (Engine::Scratchy.label(), Engine::Scratchy.ahead());
    let (rival, rival_ahead) = (rival.label(), rival.ahead());
    let line = |group, key, value| SparkPoint {
        group,
        key,
        value,
        upper: None,
        lower: None,
        factor: None,
    };
    let band = |group, key, a: f64, b: f64| SparkPoint {
        group,
        key,
        value: Some(a.max(b)),
        upper: Some(a.max(b)),
        lower: Some(a.min(b)),
        factor: None,
    };
    let gap = |group, key| SparkPoint {
        group,
        key,
        value: None,
        upper: None,
        lower: None,
        factor: None,
    };
    // Positive where scratchy is ahead: more tokens a second, or less time.
    let lead = |s: f64, r: f64| match better {
        Better::Higher => s - r,
        Better::Lower => r - s,
    };
    let mut out: Vec<SparkPoint> = Vec::new();
    for &Step {
        x,
        scratchy: sv,
        rival: rv,
        ..
    } in pts
    {
        out.push(line(scratchy, x, sv));
        out.push(line(rival, x, rv));
    }
    let mut fills: Vec<SparkPoint> = Vec::new();
    for (
        k,
        &Step {
            x,
            scratchy: sv,
            rival: rv,
            highlight,
        },
    ) in pts.iter().enumerate()
    {
        match (sv, rv) {
            (Some(sv), Some(rv)) => {
                let d = lead(sv, rv);
                let factor = if highlight {
                    better.faster(Some(sv), Some(rv))
                } else {
                    None
                };
                fills.push(if d > 0.0 {
                    SparkPoint {
                        factor,
                        ..band(scratchy_ahead, x, sv, rv)
                    }
                } else {
                    gap(scratchy_ahead, x)
                });
                fills.push(if d < 0.0 {
                    SparkPoint {
                        factor,
                        ..band(rival_ahead, x, sv, rv)
                    }
                } else {
                    gap(rival_ahead, x)
                });
                // Where the next step's lead has the other sign, the lines
                // cross between: both fills end there, at no width.
                if let Some(&Step {
                    x: x1,
                    scratchy: Some(s1),
                    rival: Some(r1),
                    ..
                }) = pts.get(k + 1)
                {
                    let d1 = lead(s1, r1);
                    if d * d1 < 0.0 {
                        let t = d / (d - d1);
                        let xc = x * (x1 / x).powf(t);
                        let yc = sv + t * (s1 - sv);
                        fills.push(band(scratchy_ahead, xc, yc, yc));
                        fills.push(band(rival_ahead, xc, yc, yc));
                    }
                }
            }
            _ => {
                fills.push(gap(scratchy_ahead, x));
                fills.push(gap(rival_ahead, x));
            }
        }
    }
    out.extend(fills);
    out
}

/// What a sparkline follows: which way its values are better, and its axes'
/// titles, which are the tooltip's labels.
#[derive(Clone, Copy)]
struct SparkAxes {
    better: Better,
    along: &'static str,
    value: &'static str,
}

/// A sparkline: scratchy's and its rival's values along one dimension, the gap
/// between them filled blue where scratchy is ahead and red where it is behind.
fn sparkline(pts: &[Step], axes: SparkAxes, rival: Engine, height: &str, label: &str) -> Element {
    let xs: Vec<f64> = pts.iter().map(|p| p.x).collect();
    // From the power of two at or below the first step to the one at or above
    // the last: users and lengths are powers of two already, so they span the
    // chart; model sizes sit inside (Carbon misdraws a log axis given the
    // sizes' own ends, 1.4 to 15.3 GB, and draws 1 to 16 right).
    let range = match (xs.first(), xs.last()) {
        (Some(&a), Some(&b)) => Some(AxisDomain::Values([
            a.log2().floor().exp2(),
            b.log2().ceil().exp2(),
        ])),
        _ => None,
    };
    // The axes' titles are the tooltip's labels; the hidden one's too.
    let bottom = Axis {
        maps_to: "key",
        title: axes.along,
        scale_type: Some("log"),
        ticks: Some(Ticks { values: xs }),
        include_zero: Some(false),
        visible: None,
        domain: range,
    };
    let left = Axis {
        maps_to: "value",
        title: axes.value,
        scale_type: None,
        ticks: None,
        include_zero: Some(false),
        visible: Some(false),
        domain: None,
    };
    let spec = Chart::Combo {
        data: spark_points(pts, axes.better, rival),
        options: ChartOptions {
            title: String::new(),
            // Users, both lengths and model sizes grow by a factor per step:
            // on a log axis the steps sit evenly.
            axes: Axes { bottom, left },
            color: Some(Colour::Scale {
                scale: [
                    (Engine::Scratchy.label(), colour(Engine::Scratchy.series())),
                    (rival.label(), colour(rival.series())),
                    // The heat maps' cyan and red 50.
                    (Engine::Scratchy.ahead(), "#1192e8"),
                    (rival.ahead(), "#fa4d56"),
                ]
                .into_iter()
                .collect(),
            }),
            heatmap: None,
            height: height.to_string(),
            width: None,
            legend: Some(Enabled { enabled: false }),
            locale: None,
            // Carbon draws an area's bounds only when the layer holds one series.
            combo: Some(vec![
                ComboLayer {
                    kind: "area",
                    corresponding_datasets: vec![Engine::Scratchy.ahead()],
                },
                ComboLayer {
                    kind: "area",
                    corresponding_datasets: vec![rival.ahead()],
                },
                ComboLayer {
                    kind: "line",
                    corresponding_datasets: vec![Engine::Scratchy.label(), rival.label()],
                },
            ]),
            bounds: Some(Bounds {
                upper_bound_maps_to: "upper",
                lower_bound_maps_to: "lower",
            }),
            theme: "white",
            toolbar: Enabled { enabled: false },
        },
    };
    chart(&spec, label, "width: 100%", None)
}

/// Which overview sparkline: machine, model and trend.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SparkAt<'a> {
    chip: &'a str,
    stem: &'a str,
    trend: Trend,
}

/// One overview sparkline's steps, against its rival.
struct Spark {
    rival: Engine,
    steps: Vec<Step>,
}

impl Spark {
    /// A model's trend on a machine's latest run, or why there is none.
    fn of(mr: &MachineRuns, stem: &str, trend: Trend) -> Result<Self, &'static str> {
        let Entry { run, m } = mr.latest(stem).ok_or("not run")?;
        let (rival, steps) = trend.points(m, run).ok_or("nothing to compare")?;
        Ok(Spark { rival, steps })
    }
}

/// One model's sparkline for one trend on one machine, or why there is none.
fn trend_spark(at: SparkAt, spark: &Result<Spark, &str>, height: &str) -> Element {
    let spark = match spark {
        Ok(s) => s,
        Err(why) => return rsx! { "{why}" },
    };
    let label = format!(
        "{} on {}: {}, scratchy against {}",
        at.stem,
        at.chip,
        at.trend.title(),
        spark.rival.label()
    );
    sparkline(&spark.steps, at.trend.axes(), spark.rival, height, &label)
}

/// Prefill first, then decode, as in the overview.
const GRID_METRICS: [(Metric, &str); 2] = [
    (Metric::Ttft, "time to first token"),
    (Metric::Throughput, "throughput"),
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
                // A ratio grid has room for every square's factor.
                highlight: g.rival.is_some(),
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

/// How many of each sparkline panel's steps, at each end, have their factor
/// written in them: the ones scratchy is furthest ahead at, and the ones it
/// is furthest behind at (or least ahead), over every machine, model and
/// metric.
const HIGHLIGHTS: usize = 3;

/// Marks one sparkline panel's highlights, from all its steps in order: the
/// HIGHLIGHTS where scratchy is furthest ahead and those where it is furthest
/// behind (or least ahead).
fn highlight<'s>(steps: impl Iterator<Item = (Better, &'s mut Step)>) {
    let mut ranked: Vec<(f64, &mut Step)> = steps
        .filter_map(|(better, st)| Some((better.faster(st.scratchy, st.rival)?, st)))
        .collect();
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
    let n = ranked.len();
    for (k, (_, st)) in ranked.iter_mut().enumerate() {
        if k < HIGHLIGHTS || k + HIGHLIGHTS >= n {
            st.highlight = true;
        }
    }
}

/// How many of the overview's heat map squares have their factor written in
/// them: at most one per colour step, spread over the steps the squares use.
/// A heat map left without one then gets one of its own.
const HEAT_LABELS: usize = 12;

/// Where an overview heat map square is: machine, model, metric, and its
/// prompt and answer lengths.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Square<'a> {
    chip: &'a str,
    stem: &'a str,
    metric: &'a str,
    prompt: u32,
    answer: u32,
}

/// One overview heat map's ratios: its grid's lengths, the rival, and each
/// square's (prompt, answer, how many times faster scratchy is).
struct GlanceGrid<'a> {
    ins: &'a [u32],
    outs: &'a [u32],
    rival: Engine,
    cells: Vec<(u32, u32, Option<f64>)>,
}

impl<'a> GlanceGrid<'a> {
    /// A model's grid at a metric on a machine's latest run, or why none.
    fn of(mr: &'a MachineRuns, stem: &str, metric: Metric) -> Result<Self, &'static str> {
        let Entry { run, m } = mr.latest(stem).ok_or("not run")?;
        let g = ShapeGrid::of(m, run).ok_or("no grid")?;
        let rival = g.rival.ok_or("nothing to compare")?;
        let cells = g
            .ins
            .iter()
            .flat_map(|&i| g.outs.iter().map(move |&o| (i, o)))
            .map(|(i, o)| (i, o, g.versus(metric, (i, o), rival).0))
            .collect();
        Ok(GlanceGrid {
            ins: g.ins,
            outs: g.outs,
            rival,
            cells,
        })
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
    let metrics = [(Metric::Ttft, "Prefill"), (Metric::Throughput, "Decode")];
    // Every grid's ratios first, keyed by machine, model and metric, or why
    // the grid has none: the highlights are picked from all of them.
    let mut grids: BTreeMap<(&str, &str, &str), Result<GlanceGrid, &str>> = BTreeMap::new();
    for &chip in machines.keys() {
        for &stem in &stems {
            for (metric, name) in metrics {
                grids.insert(
                    (chip, stem, name),
                    GlanceGrid::of(&machines[chip], stem, metric),
                );
            }
        }
    }
    let squares: BTreeMap<Square, f64> = grids
        .iter()
        .filter_map(|(&(chip, stem, name), g)| Some((chip, stem, name, g.as_ref().ok()?)))
        .flat_map(|(chip, stem, name, g)| {
            g.cells.iter().filter_map(move |&(i, o, r)| {
                let at = Square {
                    chip,
                    stem,
                    metric: name,
                    prompt: i,
                    answer: o,
                };
                Some((at, r?))
            })
        })
        .collect();
    // Each colour step's square furthest from ×1; then HEAT_LABELS of those
    // steps, evenly spread from the slowest to the fastest, both always in.
    let mut by_step: BTreeMap<usize, (Square, f64)> = BTreeMap::new();
    for (&at, &r) in &squares {
        let kept = by_step.entry(ratio_step(r)).or_insert((at, r));
        if r.log2().abs() > kept.1.log2().abs() {
            *kept = (at, r);
        }
    }
    let used: Vec<Square> = by_step.into_values().map(|(at, _)| at).collect();
    let mut highlights: BTreeSet<Square> = if used.len() <= HEAT_LABELS {
        used.into_iter().collect()
    } else {
        (0..HEAT_LABELS)
            .map(|k| used[k * (used.len() - 1) / (HEAT_LABELS - 1)])
            .collect()
    };
    // Then each heat map still without a label gets one: its square furthest
    // from ×1 in a colour step not yet labelled, or, every step it has taken,
    // its square furthest from ×1.
    let mut taken: BTreeSet<usize> = highlights
        .iter()
        .map(|at| ratio_step(squares[at]))
        .collect();
    for &(chip, stem, metric) in grids.keys() {
        let mine = || {
            squares
                .iter()
                .filter(move |(at, _)| (at.chip, at.stem, at.metric) == (chip, stem, metric))
        };
        if mine().any(|(at, _)| highlights.contains(at)) {
            continue;
        }
        let pick = mine().max_by(|&(_, &a), &(_, &b)| {
            let fresh = |r: f64| !taken.contains(&ratio_step(r));
            fresh(a)
                .cmp(&fresh(b))
                .then(a.log2().abs().total_cmp(&b.log2().abs()))
        });
        if let Some((&at, &r)) = pick {
            highlights.insert(at);
            taken.insert(ratio_step(r));
        }
    }
    // Likewise every sparkline's steps, each panel's highlights picked from
    // its own: the steps scratchy is furthest ahead and furthest behind at.
    let mut sparks: BTreeMap<SparkAt, Result<Spark, &str>> = BTreeMap::new();
    for &chip in machines.keys() {
        for &stem in &stems {
            for (_, _, trends) in SCALING {
                for trend in trends {
                    let at = SparkAt { chip, stem, trend };
                    sparks.insert(at, Spark::of(&machines[chip], stem, trend));
                }
            }
        }
    }
    for (_, _, trends) in SCALING {
        highlight(
            sparks
                .iter_mut()
                .filter(|(at, _)| trends.contains(&at.trend))
                .filter_map(|(at, spark)| Some((at.trend.metric().better(), spark.as_mut().ok()?)))
                .flat_map(|(better, spark)| spark.steps.iter_mut().map(move |st| (better, st))),
        );
    }
    let spark = |chip: &str, stem: &str, trend: Trend| -> Element {
        let at = SparkAt { chip, stem, trend };
        trend_spark(at, &sparks[&at], "4.5rem")
    };
    let mini = |chip: &str, stem: &str, name: &str| -> Element {
        let g = match &grids[&(chip, stem, name)] {
            Ok(g) => g,
            Err(why) => return rsx! { "{why}" },
        };
        let ratios: Vec<f64> = g.cells.iter().filter_map(|&(_, _, r)| r).collect();
        let data = g
            .cells
            .iter()
            .map(|&(i, o, r)| HeatPoint {
                x: o.to_string(),
                y: i.to_string(),
                value: r.map(log2_ratio),
                highlight: highlights.contains(&Square {
                    chip,
                    stem,
                    metric: name,
                    prompt: i,
                    answer: o,
                }),
            })
            .collect();
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
                    {mini_heatmap((g.ins, g.outs), data, format!("scratchy/{}", g.rival.label()), &format!("{stem} on {chip}: scratchy {range}"))}
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
    rsx! {
        Section { id: "glance", level: 2,
            Stack { gap: 4,
                Heading { "At a glance" }
                // One tile, a content switcher choosing its panel: the heat maps,
                // then one panel of sparklines per dimension. The tile shrink-wraps
                // the heat maps instead of spanning the page; the intros and the
                // sparkline panels add no width of their own (an intro would
                // otherwise run on one line as wide as it likes) and stretch to the
                // heat maps'.
                Tile { style: "width: fit-content; max-width: 100%",
                    Stack { gap: 5,
                        ContentSwitcher { selected: slug(HEAT_MAPS),
                            ContentSwitcherItem { target: slug(HEAT_MAPS), selected: true, "{HEAT_MAPS}" }
                            for (title, _, _) in SCALING {
                                ContentSwitcherItem { target: slug(title), selected: false, "{title}" }
                            }
                            ContentSwitcherItem { target: slug(STARTUP), selected: false, "{STARTUP}" }
                        }
                        div {
                            ContentPanel { id: slug(HEAT_MAPS), shown: true,
                                Stack { gap: 5,
                                    p { style: "contain: inline-size",
                                        "Every model's prompt size × answer size grid on every machine: in each cell, the model's \
                                         prefill (time to first token) grid, then its decode (throughput) grid (rows: prompt, short to \
                                         long; columns: answer, short to long). Colour is how many times faster scratchy is than mlx-lm \
                                         (ollama where a run has no mlx-lm): blue faster, red slower, the middle step within noise. \
                                         Hover a square for its ratio."
                                    }
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
                                                                for (_, name) in metrics { {mini(chip, stem, name)} }
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
                            // Each a machine × model matrix, every cell its prefill
                            // sparkline over its decode. The 2rem between models
                            // holds the end tick labels Carbon centres on each
                            // chart's edges.
                            for (title, along, [prefill, decode]) in SCALING {
                                ContentPanel { id: slug(title), shown: false,
                                    div { style: "contain: inline-size",
                                        Stack { gap: 5,
                                            p { "scratchy against mlx-lm (ollama where a run has no mlx-lm) along {along}. Blue fill where scratchy is ahead, red where it is behind." }
                                            div { style: spark_grid(stems.len()),
                                                div {}
                                                div {}
                                                for stem in &stems { div { style: "text-align: center", "{stem}" } }
                                                for chip in machines.keys() {
                                                    div { style: "grid-row: span 2; align-self: start", "{chip}" }
                                                    div { "Prefill" }
                                                    for stem in &stems { div { {spark(chip, stem, prefill)} } }
                                                    div { "Decode" }
                                                    for stem in &stems { div { {spark(chip, stem, decode)} } }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            ContentPanel { id: slug(STARTUP), shown: false,
                                div { style: "contain: inline-size", {startup_panel(machines, stems.len())} }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The glance's sparkline grids, every panel's, so every sparkline is one
/// size: the machine, a row label, then `columns` equal columns. The label
/// column has a width of its own rather than its labels', so a panel without
/// row labels gives its sparklines the same width as one with them.
fn spark_grid(columns: usize) -> String {
    format!(
        "display: grid; grid-template-columns: 7rem 4rem repeat({columns}, minmax(0, 1fr)); \
         gap: 0.25rem 2rem; align-items: center"
    )
}

/// The glance's startup panel's name.
const STARTUP: &str = "Startup by model size";

/// The engines the startup panel compares: scratchy, and mlx-lm, which loads
/// the same MLX checkpoint.
const STARTUP_ENGINES: [Engine; 2] = [Engine::Scratchy, Engine::MlxLm];

/// The unit a startup time is shown in.
#[derive(Clone, Copy)]
enum TimeUnit {
    Seconds,
    Millis,
}

impl TimeUnit {
    fn label(self) -> &'static str {
        match self {
            TimeUnit::Seconds => "s",
            TimeUnit::Millis => "ms",
        }
    }

    fn per_second(self) -> f64 {
        match self {
            TimeUnit::Seconds => 1.0,
            TimeUnit::Millis => 1000.0,
        }
    }

    /// Decimals it is written with in the table.
    fn decimals(self) -> usize {
        match self {
            TimeUnit::Seconds => 2,
            TimeUnit::Millis => 0,
        }
    }
}

/// One column of the startup panel: its scenario and title, what it times and
/// in what unit, and its sparklines' axes.
struct StartupColumn {
    scenario: Scenario,
    title: &'static str,
    what: Startup,
    unit: TimeUnit,
    axes: SparkAxes,
}

/// Frozen and cold time a launch, to the first token rather than to ready;
/// warm, a request to a server already running.
const STARTUP_COLUMNS: [StartupColumn; 3] = [
    StartupColumn {
        scenario: Scenario::Frozen,
        title: "Frozen",
        what: Startup::Launch,
        unit: TimeUnit::Seconds,
        axes: SparkAxes {
            better: Better::Lower,
            along: "model size, GB",
            value: "launch to first token, s",
        },
    },
    StartupColumn {
        scenario: Scenario::Cold,
        title: "Cold",
        what: Startup::Launch,
        unit: TimeUnit::Seconds,
        axes: SparkAxes {
            better: Better::Lower,
            along: "model size, GB",
            value: "launch to first token, s",
        },
    },
    StartupColumn {
        scenario: Scenario::Warm,
        title: "Warm",
        what: Startup::FirstToken,
        unit: TimeUnit::Millis,
        axes: SparkAxes {
            better: Better::Lower,
            along: "model size, GB",
            value: "request to first token, ms",
        },
    },
];

impl StartupColumn {
    /// An engine's median time for a model, in the column's unit.
    fn time(&self, e: Engine, m: &Model) -> Option<f64> {
        med(e.ladder(m), self.scenario, self.what).map(|v| v * self.unit.per_second())
    }

    /// An engine's time for a model as the table writes it, or why there is
    /// none; a launch also split into ready, then the first request.
    fn written(&self, e: Engine, m: &Model) -> String {
        let (ladder, unit) = (e.ladder(m), self.unit);
        let write = |what| {
            lfmt(
                ladder,
                self.scenario,
                what,
                unit.decimals(),
                unit.per_second(),
            )
        };
        match self.what {
            Startup::Launch => format!(
                "{} ({} + {})",
                write(Startup::Launch),
                write(Startup::Ready),
                write(Startup::FirstToken)
            ),
            what => write(what),
        }
    }
}

/// A size as a sparkline's step and tooltip show it: GB to one decimal.
fn gb1(b: Bytes) -> f64 {
    (b.gb() * 10.0).round() / 10.0
}

/// One machine's startup sparkline for one column: one run's models smallest
/// first, and at each one scratchy's and mlx-lm's median time.
struct StartupSpark<'a> {
    models: Vec<(&'a Model, Bytes)>,
    steps: Vec<Step>,
}

impl<'a> StartupSpark<'a> {
    /// From the newest of the machine's runs that timed both engines for the
    /// most models, so a sparkline's steps share one commit and one set of
    /// engine versions; or why there is none.
    fn of(mr: &MachineRuns<'a>, col: &StartupColumn) -> Result<Self, &'static str> {
        let both = |run: &Run| {
            run.models
                .iter()
                .filter(|m| STARTUP_ENGINES.iter().all(|&e| col.time(e, m).is_some()))
                .count()
        };
        let run = mr
            .runs
            .iter()
            .copied()
            .filter(|r| both(r) > 0)
            .max_by(|a, b| {
                both(a)
                    .cmp(&both(b))
                    .then(a.generated_utc.cmp(&b.generated_utc))
            })
            .ok_or_else(|| {
                let timed = mr.runs.iter().any(|r| {
                    r.models
                        .iter()
                        .any(|m| col.time(Engine::Scratchy, m).is_some())
                });
                if timed {
                    "nothing to compare"
                } else {
                    "not measured"
                }
            })?;
        let mut models: Vec<(&Model, Bytes)> = run
            .models
            .iter()
            .filter_map(|m| Some((m, m.weights()?)))
            .collect();
        models.sort_by_key(|&(_, b)| b);
        let steps = models
            .iter()
            .map(|&(m, b)| Step {
                x: gb1(b),
                scratchy: col.time(Engine::Scratchy, m),
                rival: col.time(Engine::MlxLm, m),
                highlight: false,
            })
            .collect();
        Ok(StartupSpark { models, steps })
    }
}

/// Every machine's startup against model size, scratchy against mlx-lm, in
/// the scaling panels' grid (`columns` is theirs): a row per machine, frozen,
/// cold and warm side by side, each sparkline in a cell the size of theirs;
/// then the numbers behind them as a table.
fn startup_panel(machines: &BTreeMap<&str, MachineRuns>, columns: usize) -> Element {
    let mut sparks: BTreeMap<(&str, usize), Result<StartupSpark, &str>> = machines
        .iter()
        .flat_map(|(&chip, mr)| {
            STARTUP_COLUMNS
                .iter()
                .enumerate()
                .map(move |(k, col)| ((chip, k), StartupSpark::of(mr, col)))
        })
        .collect();
    highlight(
        sparks
            .iter_mut()
            .filter_map(|(&(_, k), sp)| Some((k, sp.as_mut().ok()?)))
            .flat_map(|(k, sp)| {
                sp.steps
                    .iter_mut()
                    .map(move |st| (STARTUP_COLUMNS[k].axes.better, st))
            }),
    );
    // Every model charted, smallest first, for the intro and the table.
    let mut models: Vec<(&str, Bytes)> = Vec::new();
    for sp in sparks.values().filter_map(|sp| sp.as_ref().ok()) {
        for &(m, b) in &sp.models {
            if !models.iter().any(|(s, _)| *s == m.stem) {
                models.push((&m.stem, b));
            }
        }
    }
    models.sort_by_key(|&(_, b)| b);
    let sizes = models
        .iter()
        .map(|&(stem, b)| format!("{stem} {:.1} GB", b.gb()))
        .collect::<Vec<_>>()
        .join(", ");
    let mut table_headers = headers(["machine", "engine", "startup"]);
    table_headers.extend(
        models
            .iter()
            .map(|&(stem, b)| format!("{stem} ({:.1} GB)", b.gb())),
    );
    rsx! {
        Stack { gap: 5,
            p {
                "scratchy against mlx-lm as the model grows ({sizes}, the MLX checkpoint both load). Blue fill \
                 where scratchy is ahead, red where it is behind. Frozen is the first launch after memory and \
                 scratchy's weights cache are cleared, cold a normal relaunch: both time launch to the first \
                 token, not to ready, because mlx-lm reports ready at about the same time whatever the model's \
                 size and its loading lands in the first request. Warm is a request to a server already running, \
                 send to first token. Each sparkline is one run: the newest that timed both engines for the most models."
            }
            div { style: spark_grid(columns.max(STARTUP_COLUMNS.len())),
                div {}
                div {}
                for col in &STARTUP_COLUMNS { div { style: "text-align: center", "{col.title}" } }
                for &chip in machines.keys() {
                    // Column 1 starts a row, past any columns this panel leaves empty.
                    div { style: "grid-column: 1; align-self: start", "{chip}" }
                    div {}
                    for (k, col) in STARTUP_COLUMNS.iter().enumerate() {
                        div {
                            match &sparks[&(chip, k)] {
                                Ok(sp) => sparkline(&sp.steps, col.axes, Engine::MlxLm, "4.5rem", &format!("{chip}, {}: {} by model size, scratchy against mlx-lm", col.title, col.axes.value)),
                                Err(why) => rsx! { "{why}" },
                            }
                        }
                    }
                }
            }
            Fold { title: "Table view",
                p { "Frozen and cold: seconds from launch to first token, then in brackets to ready and from there to the first token. Warm: milliseconds from sending a request to its first token." }
                Table { headers: table_headers,
                    for (&(chip, k), sp) in &sparks {
                        if let Ok(sp) = sp {
                            for e in STARTUP_ENGINES {
                                TableRow {
                                    TableCell { "{chip}" }
                                    TableCell { "{e.label()}" }
                                    TableCell { "{STARTUP_COLUMNS[k].title.to_lowercase()}, {STARTUP_COLUMNS[k].unit.label()}" }
                                    for &(stem, _) in &models {
                                        TableCell {
                                            {
                                                match sp.models.iter().find(|(m, _)| m.stem == stem) {
                                                    None => DASH.to_string(),
                                                    Some((m, _)) => STARTUP_COLUMNS[k].written(e, m),
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

/// The glance's first panel's name.
const HEAT_MAPS: &str = "Heat maps";

/// The sparkline panels: a dimension, what runs along it, and its prefill and
/// decode trends.
const SCALING: [(&str, &str, [Trend; 2]); 2] = [
    (
        "Concurrency scaling",
        "1, 4 and 16 users at once, with 512-token prompts and 128-token answers",
        [Trend::UsersPrefill, Trend::UsersDecode],
    ),
    (
        "Length scaling",
        "prompt length (128, 1024 and 4096 tokens) for prefill and answer length (16, 128 and 512 tokens) for decode, with 8 users at once. Each prefill point is the geometric mean over the three answer lengths, and each decode point over the three prompt lengths",
        [Trend::PromptPrefill, Trend::AnswerDecode],
    ),
];

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
    use super::{
        Engine, Metric, RATIO_RANGE, RATIO_SPAN, RATIO_STEPS, Step, ratio_step, spark_points,
    };

    #[test]
    fn the_fills_switch_where_the_lines_cross() {
        // Ahead at 1 user (10 tok/s against 5), behind at 4 (5 against 10).
        let steps = [
            Step {
                x: 1.0,
                scratchy: Some(10.0),
                rival: Some(5.0),
                highlight: false,
            },
            Step {
                x: 4.0,
                scratchy: Some(5.0),
                rival: Some(10.0),
                highlight: false,
            },
        ];
        let pts = spark_points(&steps, Metric::Throughput.better(), Engine::MlxLm);
        let fill = |group| -> Vec<(f64, Option<(f64, f64)>)> {
            pts.iter()
                .filter(|p| p.group == group)
                .map(|p| (p.key, p.lower.zip(p.upper)))
                .collect()
        };
        // Halfway along the lead, halfway along the log axis: x = 2, y = 7.5.
        assert_eq!(
            fill(Engine::Scratchy.ahead()),
            [
                (1.0, Some((5.0, 10.0))),
                (2.0, Some((7.5, 7.5))),
                (4.0, None)
            ]
        );
        assert_eq!(
            fill(Engine::MlxLm.ahead()),
            [
                (1.0, None),
                (2.0, Some((7.5, 7.5))),
                (4.0, Some((5.0, 10.0)))
            ]
        );
        // For time, less is ahead: the same numbers swap the fills.
        let ttft = spark_points(&steps, Metric::Ttft.better(), Engine::MlxLm);
        assert!(
            ttft.iter()
                .any(|p| p.group == Engine::MlxLm.ahead() && p.key == 1.0 && p.upper.is_some())
        );
    }

    #[test]
    fn ratio_steps_split_the_range_evenly_and_clamp() {
        let last = RATIO_STEPS.len() - 1;
        assert_eq!(ratio_step(1.0), last / 2);
        assert_eq!(ratio_step(RATIO_SPAN), last);
        assert_eq!(ratio_step(1.0 / RATIO_SPAN), 0);
        assert_eq!(ratio_step(100.0), last);
        assert_eq!(ratio_step(0.01), 0);
        // The gray middle step holds the noise band, ×0.96 to ×1.04.
        assert_eq!(ratio_step(0.96), last / 2);
        assert_eq!(ratio_step(1.04), last / 2);
        assert_eq!(ratio_step(1.1), last / 2 + 1);
    }

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

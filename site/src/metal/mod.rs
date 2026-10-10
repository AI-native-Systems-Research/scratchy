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
use crate::carbon::{Alignment, Fold, Module, Toggletip, Tooltip};
use crate::chrome::{self, Head, Library, REPO, Root, Tab, Theme};
use data::{At, Cell, Model, Run, Scenario};
use numbers::{
    Engine, HEAT_LABELS, Metric, NO_STREAM, NOISE, PARTIAL, Startup, cell, cfmt, fmt, heat_class,
    lfmt, med, partial, served, timing, untimed_note,
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

/// One engine's line in a tooltip.
struct TipRow {
    value: String,
    label: String,
    engine: Engine,
}

/// A tooltip's body: what the mark is, then one line per engine.
fn tip_body(head: &str, rows: &[TipRow]) -> Element {
    rsx! {
        div { class: "mtiphead", "{head}" }
        for r in rows {
            div {
                span { class: "lkey {r.engine.series()}" }
                b { "{r.value}" }
                span { class: "mtipdim", " {r.label}" }
            }
        }
    }
}

/// A change badge against the previous run: arrow plus percent, the arrow
/// coloured by better/worse so colour is never the only signal; "≈" inside the
/// noise band; nothing when either side is missing.
fn delta(new: Option<f64>, old: Option<f64>, higher_better: bool) -> Element {
    let (Some(new), Some(old)) = (new, old.filter(|o| *o != 0.0)) else {
        return rsx! {};
    };
    let ch = (new - old) / old;
    if ch.abs() < NOISE {
        return rsx! { span { class: "delta", title: "within 3% of the previous run", "≈" } };
    }
    let word = if (ch > 0.0) == higher_better {
        "better"
    } else {
        "worse"
    };
    let arrow = if ch > 0.0 { "▲" } else { "▼" };
    let pct = format!("{:.0}", ch.abs() * 100.0);
    rsx! { span { class: "delta", title: "{word} than the previous run", i { class: word, "{arrow}" } "{pct}%" } }
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
            b { "frozen" } ": first launch after memory and scratchy's weights cache are cleared. "
            b { "cold" } ": a normal relaunch. Both run until the server is ready, no request included. "
            b { "1st req" } ": the cold server's first request, send to first token. "
            b { "warm" } ": the median over many requests once it is running. ollama reports ready before \
                          loading the model, so its load time lands in 1st req."
        },
    )
}

fn timing_info() -> Element {
    info(
        "",
        rsx! {
            b { "no stream" } ": the answer arrived in one piece, so TTFT and TPOT could not be timed. "
            b { "†" } ": some answers did; the timing comes from the rest (hover for how many). mlx-lm and \
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

fn summary_table(m: &Model, run: &Run, prev: Option<Entry>) -> Element {
    let shape = run
        .config
        .base()
        .map(|b| format!("{} in / {} out", opt(b.input), opt(b.output)));
    let rows = Engine::ALL.into_iter().filter_map(|e| {
        let (ladder, one) = (e.ladder(m), cell(e.cells(m), At::Conc(1)));
        let key = rsx! { span { class: "key {e.series()}" } };
        if e == Engine::Scratchy && !served(m) {
            let why = if m.built { "the server never served" } else { "build failed" };
            return Some(rsx! {
                tr {
                    th { scope: "row", {key} "{e.label()}" }
                    td { colspan: "8", class: "gap", "No numbers: {why}. See the run's raw logs." }
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
        let then = prev.filter(|_| e == Engine::Scratchy).map(|p| summary_raw(p.m));
        let ollama = m.ollama.as_ref().filter(|_| e == Engine::Ollama);
        Some(rsx! {
            tr {
                th { scope: "row", {key} "{e.label()}"
                    if let Some(o) = ollama {
                        " "
                        span { class: "dim",
                            "{o.tag.as_deref().unwrap_or_default()}"
                            if let Some(q) = o.quantization.as_deref().filter(|q| !q.is_empty()) { ", {q}" }
                        }
                    }
                }
                for (i, v) in vals.iter().enumerate() {
                    td { "{v}"
                        if let Some(then) = then { {delta(now[i].0, then[i].0, now[i].1)} }
                    }
                }
            }
        })
    });
    let changed = prev
        .map(|p| settings_changed(run, p.run))
        .unwrap_or_default();
    let one_user = shape
        .as_ref()
        .map_or("one user".to_string(), |s| format!("one user, {s}"));
    rsx! {
        div { class: "mpart",
            h4 { "Startup and single-user speed" }
            p { class: "msub",
                "One row per engine. Startup in seconds (warm in ms); one user at {shape.as_deref().unwrap_or(\"the base shape\")}."
                if let Some(p) = prev {
                    " Under scratchy: change since its previous run here ({p.run.generated_utc.date()}, "
                    code { "{p.run.repo.sha8()}" }
                    "); ≈ is within 3%."
                    if !changed.is_empty() {
                        " " b { "Settings differ from that run" } " ({changed}), so a change is not the code alone."
                    }
                }
                " What the columns mean: startup " {startup_info()} " one user " {timing_info()} " peak RSS " {rss_info()}
            }
            div { class: "mtable",
                table {
                    thead {
                        tr {
                            th { rowspan: "2", scope: "col", class: "first", "engine" }
                            th { colspan: "4", scope: "colgroup", "startup" }
                            th { colspan: "3", scope: "colgroup", "{one_user}" }
                            th { rowspan: "2", scope: "col", "peak RSS" br {} span { class: "dim", "MiB" } }
                        }
                        tr {
                            for (h, u) in [("frozen", "s"), ("cold", "s"), ("1st req", "s"), ("warm", "ms"), ("TTFT", "ms"), ("TPOT", "ms")] {
                                th { scope: "col", "{h}" br {} span { class: "dim", "{u}" } }
                            }
                            th { scope: "col", "tok/s" }
                        }
                    }
                    tbody { {rows} }
                }
            }
        }
    }
}

/// One line of a chart: its label, series colour, and (x, value) points; a
/// point present but None was not measured.
struct Series {
    label: String,
    class: &'static str,
    pts: Vec<(u32, Option<f64>)>,
}

/// A series' colour, as Carbon Charts needs it: a value, not a class. The
/// same hues as `--s1` to `--s3` in styles.css; the previous run is a light
/// grey so it reads as behind.
fn colour(class: &str) -> &'static str {
    match class {
        "s1" => "#2a78d6",
        "s2" => "#d95926",
        "s3" => "#1baf7a",
        _ => "#a8a8a8",
    }
}

/// What charts.js hands `new Charts.LineChart`: the points and the options,
/// in Carbon Charts' own field names.
#[derive(Serialize)]
struct LineChart<'a> {
    data: Vec<Point<'a>>,
    options: ChartOptions<'a>,
}

/// One measured point. A None value (present but not measured: no stream)
/// leaves a gap in its line; `note` is added to its tooltip.
#[derive(Serialize)]
struct Point<'a> {
    group: &'a str,
    key: u32,
    value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    note: Option<String>,
}

#[derive(Serialize)]
struct ChartOptions<'a> {
    axes: Axes<'a>,
    color: ColourScale<'a>,
    height: &'static str,
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
}

#[derive(Serialize)]
struct Ticks<'a> {
    values: &'a [u32],
}

#[derive(Serialize)]
struct ColourScale<'a> {
    scale: BTreeMap<&'a str, &'static str>,
}

#[derive(Serialize)]
struct Enabled {
    enabled: bool,
}

/// One line chart: offered users on x, one line per engine, drawn in the
/// browser by Carbon Charts (see charts.js). `notes` maps (engine label, x)
/// to a suffix for that point's tooltip.
fn line_chart(
    series: &[Series],
    xs: &[u32],
    ylabel: &str,
    title: &str,
    notes: &BTreeMap<(String, u32), String>,
) -> Element {
    let spec = LineChart {
        data: series
            .iter()
            .flat_map(|s| {
                s.pts.iter().map(|&(x, v)| Point {
                    group: &s.label,
                    key: x,
                    // Carbon prints values as given: round to what the tables show.
                    value: v.map(|v| (v * 10.0).round() / 10.0),
                    note: notes.get(&(s.label.clone(), x)).cloned(),
                })
            })
            .collect(),
        options: ChartOptions {
            axes: Axes {
                bottom: Axis {
                    maps_to: "key",
                    title: "offered concurrent users",
                    scale_type: Some("linear"),
                    ticks: Some(Ticks { values: xs }),
                    include_zero: None,
                },
                left: Axis {
                    maps_to: "value",
                    title: ylabel,
                    scale_type: None,
                    ticks: None,
                    include_zero: Some(true),
                },
            },
            color: ColourScale {
                scale: series
                    .iter()
                    .map(|s| (s.label.as_str(), colour(s.class)))
                    .collect(),
            },
            height: "260px",
            // The page is always Carbon's light theme.
            theme: "white",
            toolbar: Enabled { enabled: false },
        },
    };
    rsx! {
        div { class: "mchart", role: "img", "aria-label": title,
            "data-chart": serde_json::to_string(&spec).unwrap_or_default(),
        }
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

    let mut charts: Vec<(&str, &str, Element, Element)> = Vec::new();
    if !tput.is_empty() {
        charts.push((
            "Throughput",
            "tok/s, higher is better",
            line_chart(
                &tput,
                &xs,
                "tok/s",
                "Output tokens per second against offered concurrent users, per engine",
                &BTreeMap::new(),
            ),
            rsx! {},
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
        let note = if gone.is_empty() {
            rsx! {}
        } else {
            rsx! { p { class: "msub", "{gone.join(\", \")}: {NO_STREAM}, not plotted." } }
        };
        let marks: BTreeMap<(String, u32), String> = cells
            .iter()
            .flat_map(|(e, cs)| {
                cs.iter()
                    .filter(|(_, c)| partial(Some(c), Metric::Tpot))
                    .map(move |(x, c)| ((e.label().to_string(), *x), untimed_note(Some(c))))
            })
            .collect();
        charts.push((
            "Time per output token",
            "ms, lower is better",
            line_chart(
                &tpot,
                &xs,
                "TPOT ms",
                "Median time per output token against offered concurrent users, per engine",
                &marks,
            ),
            note,
        ));
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
    rsx! {
        figure { class: "mfig",
            figcaption {
                h4 { "As users are added" }
                p { class: "msub",
                    "Throughput and time per output token with {users} users at once " {users_info()}
                    "; {input}-token prompts, {output}-token answers. A line that rises in the right-hand chart means each user's answer slows down as more share the engine."
                }
            }
            div { class: "chartrow",
                for (title, unit, chart, note) in charts {
                    div { class: "chartpane",
                        div { class: "heatttl", "{title} " span { class: "dim", "({unit})" } }
                        {chart}
                        {note}
                    }
                }
            }
            Fold { title: "Table view",
                div { class: "mtable",
                    table {
                        thead { tr { th { scope: "col", "offered users" } for x in &xs { th { scope: "col", "{x}" } } } }
                        tbody {
                            for (e, cs) in cells.iter().filter(|(_, cs)| !cs.is_empty()) {
                                for (metric, name) in [(Metric::Throughput, "tok/s"), (Metric::Ttft, "TTFT ms"), (Metric::Tpot, "TPOT ms")] {
                                    tr {
                                        th { scope: "row", "{e.label()} · {name}" }
                                        for x in &xs { td { {cfmt(cs.get(x).copied(), metric)} } }
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

/// One engine's grid cells, by (prompt, answer) tokens.
type ByShape<'a> = BTreeMap<(u32, u32), &'a Cell>;

/// A model's prompt-by-answer grid: the axes, each engine's cells, and the
/// engines scratchy is compared against.
struct Grid<'a> {
    ins: &'a [u32],
    outs: &'a [u32],
    have: Vec<(Engine, ByShape<'a>)>,
    /// mlx-lm, which runs the same MLX checkpoint; failing that ollama.
    rival: Option<Engine>,
    /// The other of the two, when the run has both.
    other: Option<Engine>,
}

impl<'a> Grid<'a> {
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
        let g = Grid {
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
        Some(Grid {
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
}

const GRID_METRICS: [(Metric, &str); 2] = [
    (Metric::Throughput, "throughput"),
    (Metric::Ttft, "time to first token"),
];

fn heat_legend() -> Element {
    rsx! {
        for (i, label) in HEAT_LABELS.iter().enumerate() {
            span { i { class: "sw hm{i}" } "{label}" }
        }
    }
}

fn mark(metric: Metric, cells: &[Option<&Cell>]) -> &'static str {
    if cells.iter().any(|c| partial(*c, metric)) {
        PARTIAL
    } else {
        ""
    }
}

fn heat_table(g: &Grid, mid: &str, metric: Metric, title: &str) -> Element {
    let why = |c: Option<&Cell>, name: &str| {
        if c.is_some() {
            format!("{name}: {NO_STREAM}")
        } else {
            format!("no {name}")
        }
    };
    let versus = g.rival.map_or(" · scratchy".to_string(), |r| {
        format!(" · scratchy vs {}", r.label())
    });
    rsx! {
        div { class: "heat",
            div { class: "heatttl", b { "{title}" } "{versus}" }
            div { class: "mtable",
                table {
                    thead { tr { th { scope: "col", span { class: "dim", "prompt ↓ / answer →" } } for o in g.outs { th { scope: "col", "{o}" } } } }
                    tbody {
                        for &i in g.ins {
                            tr {
                                th { scope: "row", "{i}" }
                                for &o in g.outs {
                                    {
                                        let rung = (i, o);
                                        let cs = g.at(Some(Engine::Scratchy), rung);
                                        let (cr, co) = (g.at(g.rival, rung), g.at(g.other, rung));
                                        let s = timing(cs, metric);
                                        let (r_r, r_o) = (metric.faster(s, timing(cr, metric)), metric.faster(s, timing(co, metric)));
                                        let tiprows: Vec<TipRow> = g.have.iter().filter(|(_, h)| !h.is_empty()).map(|&(engine, ref h)| {
                                            let c = h.get(&rung).copied();
                                            let label = if partial(c, metric) { format!("{} ({})", engine.label(), untimed_note(c)) } else { engine.label().to_string() };
                                            TipRow { value: cfmt(c, metric), label, engine }
                                        }).collect();
                                        let (big, small) = match g.rival {
                                            // Nothing to compare against.
                                            None => (cfmt(cs, metric), if s.is_some() { metric.unit().to_string() } else { String::new() }),
                                            Some(rival) => (
                                                match r_r {
                                                    Some(r) => format!("×{r:.2}{}", mark(metric, &[cs, cr])),
                                                    None if s.is_none() => why(cs, "scratchy"),
                                                    None => why(cr, rival.label()),
                                                },
                                                match (r_o, g.other) {
                                                    (Some(r), Some(other)) => format!("vs {} ×{r:.2}{}", other.label(), mark(metric, &[cs, co])),
                                                    (_, Some(other)) if s.is_some() => why(co, other.label()),
                                                    _ => String::new(),
                                                },
                                            ),
                                        };
                                        rsx! {
                                            td { class: r_r.map(heat_class).unwrap_or_default(),
                                                Tooltip { id: "tip-{mid}-{metric.id()}-{i}x{o}", class: "mcell", focusable: true,
                                                    trigger: rsx! { b { "{big}" } span { class: "msmall", "{small}" } },
                                                    {tip_body(&format!("{i} in × {o} out · {title}"), &tiprows)}
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

fn grid_maps(m: &Model, mid: &str, run: &Run) -> Element {
    let Some(g) = Grid::of(m, run) else {
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
                "prompt size down the side, answer size across the top. Each large number is how many times faster scratchy is than {}{}: ×1.20 is 20% faster, below ×1 is slower{}. Hover for every engine's value",
                rival.label(),
                missing.map_or(String::new(), |e| format!(
                    " (this run has no {})",
                    e.label()
                )),
                g.other.map_or(String::new(), |o| format!(
                    "; the small line is the same against {}",
                    o.label()
                )),
            )
        }
    };
    rsx! {
        figure { class: "mfig",
            figcaption {
                h4 { "Prompt size × answer size" }
                p { class: "msub", "{conc} users at once; {note}." }
            }
            div { class: "heatrow",
                for (metric, title) in GRID_METRICS { {heat_table(&g, mid, metric, title)} }
            }
            if let Some(rival) = g.rival {
                div { class: "scale",
                    span { class: "scalekey", "Colour and large number, scratchy vs {rival.label()}:" }
                    {heat_legend()}
                }
            }
        }
    }
}

fn history(entries: &[Entry]) -> Element {
    if entries.len() < 2 {
        return rsx! {};
    }
    rsx! {
        Fold { title: "Runs of this model ({entries.len()})",
            div { class: "mtable",
                table {
                    thead {
                        tr {
                            for h in ["run", "scratchy", "quant", "cold s", "1st req s", "warm ms", "tok/s, 1 user", "tok/s, most users", "TPOT ms, most users"] {
                                th { scope: "col", "{h}" }
                            }
                        }
                    }
                    tbody {
                        for Entry { run, m } in entries.iter().rev() {
                            tr {
                                th { scope: "row", "{run.generated_utc.date()}" }
                                td { a { href: "{REPO}/commit/{run.repo.sha}", code { "{run.repo.sha8()}" } } }
                                td { "{m.quant.as_deref().unwrap_or_default()}" }
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
                                        rsx! { for v in vals { td { "{v}" } } }
                                    }
                                } else {
                                    td { colspan: "6", class: "gap", "no numbers" }
                                }
                            }
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
            div { class: "mtable",
                table {
                    thead {
                        tr {
                            for h in ["when", "scratchy", "steps", "cold priming launches", "scratchy KV cache", "scratchy serve flags", "data"] {
                                th { scope: "col", "{h}" }
                            }
                        }
                    }
                    tbody {
                        for run in runs.iter().rev() {
                            {
                                let [(_, flags), (_, kv)] = settings(run);
                                let file = run.file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                                rsx! {
                                    tr {
                                        th { scope: "row", "{run.generated_utc.minute()} UTC" }
                                        td { a { href: "{REPO}/commit/{run.repo.sha}", code { "{run.repo.sha8()}" } } }
                                        td { "{run.config.scenarios.join(\", \")}" }
                                        td { {run.config.cold_priming_launches.map_or("not recorded".to_string(), |n| n.to_string())} }
                                        td { "{kv}" }
                                        td { code { "{flags}" } }
                                        td { a { href: "{DATA}/{file}", "json" } }
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

/// Every model's prompt-by-answer grid on every machine as one small multiple
/// per metric: the same colours as the full grids, no numbers in the squares,
/// each linking to its model's full section.
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
    let mini = |chip: &str, stem: &str, metric: Metric| -> Element {
        let Some(Entry { run, m }) = machines[chip].latest(stem) else {
            return rsx! { td { class: "gnone", "not run" } };
        };
        let Some(g) = Grid::of(m, run) else {
            return rsx! { td { class: "gnone", "no grid" } };
        };
        let Some(rival) = g.rival else {
            return rsx! { td { class: "gnone", "nothing to compare" } };
        };
        let mut ratios = Vec::new();
        let mut squares = Vec::new();
        for &i in g.ins {
            for &o in g.outs {
                let (cs, cr) = (
                    g.at(Some(Engine::Scratchy), (i, o)),
                    g.at(Some(rival), (i, o)),
                );
                let r = metric.faster(timing(cs, metric), timing(cr, metric));
                ratios.extend(r);
                let value = r.map_or("no comparison".to_string(), |r| {
                    format!("×{r:.2}{}", mark(metric, &[cs, cr]))
                });
                // A square with nothing to compare is hollow, so it never reads
                // as "about the same".
                let class = r.map_or("gnil".to_string(), heat_class);
                // Not focusable: the grid is one link, whose label sums it up.
                let row = TipRow {
                    value,
                    label: format!("vs {}", rival.label()),
                    engine: Engine::Scratchy,
                };
                squares.push(rsx! {
                    Tooltip { id: "tip-glance-{metric.id()}-{slug(chip)}-{slug(stem)}-{i}x{o}", class: "gsq {class}",
                        trigger: rsx! {},
                        {tip_body(&format!("{chip} · {stem} · {i} in × {o} out"), &[row])}
                    }
                });
            }
        }
        let span = match (
            ratios.iter().copied().reduce(f64::min),
            ratios.iter().copied().reduce(f64::max),
        ) {
            (Some(lo), Some(hi)) => format!("×{lo:.2} to ×{hi:.2}"),
            _ => "no ratios".to_string(),
        };
        rsx! {
            td {
                a { class: "gmini", href: "#{slug(chip)}-{slug(stem)}",
                    "aria-label": "{stem} on {chip}: scratchy {span} against {rival.label()}; open the full grid",
                    style: "grid-template-columns: repeat({g.outs.len()}, 1fr)",
                    {squares.into_iter()}
                }
                span { class: "grange", "{span} vs {rival.label()}" }
            }
        }
    };
    rsx! {
        section { class: "msection", id: "glance",
            h2 { "At a glance" }
            p { class: "msub",
                "Every model's prompt size × answer size grid on every machine. Each square is one cell of the \
                 full grid (rows: prompt, short to long; columns: answer, short to long), coloured by how many \
                 times faster scratchy is than mlx-lm (ollama where a run has no mlx-lm). Hover a square for \
                 its ratio; click a grid for its model."
            }
            div { class: "heatrow",
                for (metric, title) in GRID_METRICS {
                    div { class: "heat",
                        div { class: "heatttl", "{title}" }
                        div { class: "mtable",
                            table { class: "gtable",
                                thead { tr { th { scope: "col" } for stem in &stems { th { scope: "col", "{stem}" } } } }
                                tbody {
                                    for chip in machines.keys() {
                                        tr {
                                            th { scope: "row", "{chip}" }
                                            for stem in &stems { {mini(chip, stem, metric)} }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            div { class: "scale",
                span { class: "scalekey", "scratchy is:" }
                {heat_legend()}
                span { i { class: "sw nil" } "no comparison" }
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
        section { class: "msection", id: "{mslug}",
            h2 { "{chip}" }
            p { class: "mmeta", "{cores} · {mc.memory_gb} GB unified memory · macOS {mc.macos}" }
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
                        article { class: "mmodel cds--tile", id: "{mid}",
                            h3 { "{stem} " span { class: "onmachine", "on {chip}" } }
                            p { class: "mmeta",
                                a { href: "https://huggingface.co/{m.model_id}", "{m.model_id}" }
                                " · {m.quant.as_deref().unwrap_or(\"default\")}{build} · run {run.generated_utc.date()}, "
                                code { "{run.repo.sha8()}" }
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

pub fn build(site: &Site) -> Result<String, String> {
    let runs = load(site)?;
    let machines = index(&runs);
    let mut sections: Vec<(String, String)> = Vec::new();
    let body = if machines.is_empty() {
        rsx! {
            p { class: "empty",
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
        div { class: "docs-layout",
            main { class: "docs-content metalpage",
                div { class: "hero-body",
                    h1 { "Metal performance" }
                    p { class: "lede",
                        "scratchy against mlx-lm and ollama on Apple silicon: how long each takes to start, how \
                         fast it answers one user, and how it holds up as prompts, answers and users grow. \
                         {count} runs on {n_machines} machines, every number measured by "
                        a { href: "{REPO}/blob/main/scripts/bench_metal_matrix.sh", code { "scripts/bench_metal_matrix.sh" } }
                        "."
                    }
                    div { class: "mabout", {about()} }
                }
                {body}
            }
        }
        script { dangerous_inner_html: CHARTS_JS }
    };
    let html = chrome::document(
        Head {
            title: "Metal performance — scratchy",
            description: Some(
                "scratchy against mlx-lm and ollama on Apple silicon: startup, single-user speed, and scaling, per machine.",
            ),
            og: None,
            root: Root(0),
            modules: &[
                Module::UiShell,
                Module::ToggleTip,
                Module::Tooltip,
                Module::Accordion,
            ],
            libraries: &[Library::CarbonCharts],
            theme: Theme::AlwaysLight,
        },
        page,
    );
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

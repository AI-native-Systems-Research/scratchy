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
    Alignment, Definition, Fold, Heading, Layer, Section, Stack, Table, TableCell, TableRow, Tag,
    TagKind, Tile, Toggletip,
};
use crate::chrome::{self, Content, Head, Library, Published, REPO, Root, Tab, Theme};
use data::{At, Cell, Model, Run, Scenario};
use numbers::{
    Engine, FAILED, Metric, NO_STREAM, NOISE, Startup, cfmt, failed, fmt, lfmt, marks, marks_note,
    med, served, timing,
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

/// The models the runner measured as current when this run was made: its
/// full set, or the default list the first runs to record one kept; None for
/// runs older than that.
fn current_models(run: &Run) -> Option<&Vec<String>> {
    run.config
        .current_models
        .as_ref()
        .or(run.config.default_models.as_ref())
        .filter(|v| !v.is_empty())
}

/// Per machine, by chip name. Newer runs win, so a model shows its newest
/// numbers. A model is retired once the newest run that recorded the
/// runner's current models leaves it out and ran it no later, so a --models
/// subset hides nothing; runs before that record keep every model.
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
    for mr in machines.values_mut() {
        let Some(newest) = mr.runs.iter().rev().find(|r| current_models(r).is_some()) else {
            continue;
        };
        let keep = current_models(newest).cloned().unwrap_or_default();
        let when = &newest.generated_utc;
        mr.models.retain(|(stem, es)| {
            keep.iter().any(|k| k == stem) || es.last().is_some_and(|e| &e.run.generated_utc > when)
        });
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
fn summary_raw(m: &Model, run: &Run) -> [(Option<f64>, bool); 8] {
    let ladder = Engine::Scratchy.ladder(m);
    let one = by_users(m, run, Engine::Scratchy).get(&1).copied();
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

/// A model's base shape: the run's, with the users count the runner lowered
/// for a model whose KV cache would not fit more (GLM-4.5-Air).
fn base_of(run: &Run, m: &Model) -> (Option<u32>, Option<u32>, Option<u32>) {
    let b = run.config.base();
    let conc = m
        .scaling_limits
        .as_ref()
        .and_then(|l| l.base_conc)
        .or(b.and_then(|b| b.conc));
    (b.and_then(|b| b.input), b.and_then(|b| b.output), conc)
}

/// Where a model's users numbers come from: the separate users sweep at the
/// base shape (older runs, or `--scale-axes conc`), else the grid's own users
/// dimension at its middle cell (#349).
#[derive(Clone, Copy, PartialEq)]
enum UsersFrom {
    Sweep,
    Grid { input: u32, output: u32 },
}

fn users_from(m: &Model, run: &Run) -> UsersFrom {
    if Engine::Scratchy
        .cells(m)
        .iter()
        .any(|c| matches!(c.at, At::Conc(_)))
    {
        return UsersFrom::Sweep;
    }
    let sc = run.config.scaling.as_ref();
    let mid = |v: Option<&Vec<u32>>| v.filter(|v| !v.is_empty()).map(|v| v[v.len() / 2]);
    match (
        mid(sc.and_then(|s| s.grid_input.as_ref())),
        mid(sc.and_then(|s| s.grid_output.as_ref())),
    ) {
        (Some(input), Some(output)) => UsersFrom::Grid { input, output },
        _ => UsersFrom::Sweep,
    }
}

/// One engine's cells by users count, from wherever this model's come from.
fn by_users<'a>(m: &'a Model, run: &Run, e: Engine) -> BTreeMap<u32, &'a Cell> {
    let base = base_of(run, m).2.unwrap_or(0);
    match users_from(m, run) {
        UsersFrom::Sweep => e
            .cells(m)
            .iter()
            .filter_map(|c| match c.at {
                At::Conc(n) => Some((n, c)),
                _ => None,
            })
            .collect(),
        UsersFrom::Grid { input, output } => e
            .cells(m)
            .iter()
            .filter(|c| c.at == At::Grid { input, output })
            .map(|c| (c.users.unwrap_or(base), c))
            .collect(),
    }
}

/// The prompt and answer sizes a model's users numbers were measured at.
fn users_shape(m: &Model, run: &Run) -> (Option<u32>, Option<u32>) {
    match users_from(m, run) {
        UsersFrom::Sweep => {
            let (i, o, _) = base_of(run, m);
            (i, o)
        }
        UsersFrom::Grid { input, output } => (Some(input), Some(output)),
    }
}

/// The comparison engines' releases as the runner recorded them: every engine
/// from `config.engines`, else the mlx-lm/ollama pair older runs kept.
fn engine_versions(run: &Run) -> String {
    let c = &run.config;
    let parts: Vec<String> = match &c.engines {
        Some(es) => es
            .iter()
            .filter(|e| e.key != "scratchy" && e.installed == Some(true))
            .filter_map(|e| {
                Some(format!(
                    "{} {}",
                    e.label,
                    e.version.as_deref().filter(|v| !v.is_empty())?
                ))
            })
            .collect(),
        None => c
            .engine_versions
            .iter()
            .flatten()
            .filter_map(|(k, v)| {
                let label = Engine::ALL.into_iter().find(|e| e.key() == k)?.label();
                Some(format!("{label} {}", v.as_deref()?))
            })
            .collect(),
    };
    if parts.is_empty() {
        "not recorded".to_string()
    } else {
        parts.join(" · ")
    }
}

/// Whether the run was an agreed, fully pinned one.
fn pinned(run: &Run) -> String {
    match run.config.pinned {
        None => "not recorded".to_string(),
        Some(true) => "yes".to_string(),
        Some(false) => match run.config.pin_problems.as_deref() {
            Some(ps) if !ps.is_empty() => format!("no: {}", ps.join("; ")),
            _ => "no".to_string(),
        },
    }
}

/// Everything else that can move a run-to-run number besides scratchy's own
/// code: the base shape, the other engines' releases, whether the run was
/// pinned, macOS, and the model files. Unrecorded on either side never counts
/// as a change, so runs from before a field existed do not all warn.
fn conditions(run: &Run, m: &Model) -> [(&'static str, String); 5] {
    let (i, o, c) = base_of(run, m);
    let shape = if run.config.base().is_some() {
        format!("{}/{}/{}", opt(i), opt(o), opt(c))
    } else {
        "not recorded".to_string()
    };
    let files = m
        .model_files
        .as_ref()
        .and_then(|f| f.revision.as_deref())
        .map_or("not recorded".to_string(), |r| {
            r[..r.len().min(8)].to_string()
        });
    let pin = pinned(run);
    [
        ("base shape (in/out/users)", shape),
        ("engine versions", engine_versions(run)),
        (
            "pinned",
            pin.split(':').next().unwrap_or_default().to_string(),
        ),
        ("macOS", run.machine.macos.to_string()),
        ("model files", files),
    ]
}

fn conditions_changed(run: &Run, m: &Model, prev: Entry) -> String {
    conditions(run, m)
        .into_iter()
        .zip(conditions(prev.run, prev.m))
        .filter(|((_, now), (_, then))| {
            now != then && now != "not recorded" && then != "not recorded"
        })
        .map(|((k, now), (_, then))| format!("{k} {then} then, {now} now"))
        .collect::<Vec<_>>()
        .join("; ")
}

/// What an engine served for this model, for its row: the ollama tag or GGUF
/// file it loaded, with its quantization when it reported one. Runs from
/// before `engine_models` kept only ollama's.
fn served_as(m: &Model, e: Engine) -> Option<String> {
    let (model, quant) = match m.engine_models.as_ref().and_then(|em| em.get(e.key())) {
        Some(s) => (s.model.clone(), s.quantization.clone()),
        None if e == Engine::Ollama => {
            let o = m.ollama.as_ref()?;
            (o.tag.clone(), o.quantization.clone())
        }
        None => return None,
    };
    // The MLX engines serve scratchy's own checkpoint: nothing to add.
    let model = match e {
        Engine::Ollama => model,
        Engine::LlamaCpp | Engine::Mistralrs => {
            model.map(|m| m.rsplit(':').next().unwrap_or_default().to_string())
        }
        _ => None,
    };
    let parts: Vec<String> = [model, quant]
        .into_iter()
        .flatten()
        .filter(|s| !s.is_empty())
        .collect();
    (!parts.is_empty()).then(|| parts.join(", "))
}

/// The parity gate's verdict for the model line: scratchy and mlx-lm must
/// give the same greedy answers before anything is timed. None for runs from
/// before the gate.
fn parity(m: &Model) -> Option<&'static str> {
    m.parity_mlx_lm.map(|p| match p {
        Some(true) => "output matches mlx-lm",
        Some(false) => "output does not match mlx-lm",
        None => "output not checked (no mlx-lm)",
    })
}

fn parity_info() -> Element {
    info(
        "",
        rsx! {
            "Before anything is timed, scratchy and mlx-lm answer the same short prompts with greedy decoding, \
             and the answers must match exactly. A mismatch points at a wrong load path (quant preset, group \
             size, dequant), so that model is not timed at all."
        },
    )
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
            strong { "†" } ": some answers did; the timing comes from the rest (the table views say how many). "
            strong { "{FAILED}" } ": some requests failed (an error or a timeout, usually the engine running out of \
                       memory), so every number in that cell covers only the rest. mlx-lm and ollama can stop \
                       answers early, which flatters tok/s."
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
            "Prompts are made up, a fixed size, and unique per request so no cache can answer them (warmups use \
             prompts of their own), with greedy decoding (temperature 0) on every engine. oMLX (as shipped, and \
             with 4-bit TurboQuant KV), mlx-lm and vllm-metal run the same MLX checkpoint as scratchy; ollama, \
             llama.cpp and mistral.rs run their own quantizations, named on their rows. Engine versions, and \
             whether a run was pinned, are in each machine's runs table. scratchy's build time is shown per model \
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
    let (si, so) = users_shape(m, run);
    let shape = (si.is_some() || so.is_some()).then(|| format!("{} in / {} out", opt(si), opt(so)));
    // Badges only against a run whose one-user numbers are at the same shape:
    // a change of shape is not a speedup.
    let same_shape = prev.is_some_and(|p| users_shape(p.m, p.run) == (si, so));
    let rows = Engine::ALL.into_iter().filter_map(|e| {
        let (ladder, one) = (e.ladder(m), by_users(m, run, e).get(&1).copied());
        if e == Engine::Scratchy && !served(m) {
            let why = if !m.built {
                "build failed"
            } else if m.parity_mlx_lm == Some(Some(false)) {
                "its answers did not match mlx-lm's, so no engine was timed"
            } else {
                "the server never served"
            };
            return Some(rsx! {
                TableRow {
                    TableCell { "{e.label()}" }
                    TableCell { colspan: 8, "No numbers: {why}. See the run's raw logs." }
                }
            });
        }
        if e != Engine::Scratchy && !e.ran(m) {
            return None; // that engine was not part of this run
        }
        let vals = [
            lfmt(ladder, Scenario::Frozen, Startup::Ready, 2, 1.0),
            lfmt(ladder, Scenario::Cold, Startup::Ready, 2, 1.0),
            lfmt(ladder, Scenario::Cold, Startup::FirstToken, 2, 1.0),
            lfmt(ladder, Scenario::Warm, Startup::FirstToken, 0, 1000.0),
            cfmt(one, Metric::Ttft),
            cfmt(one, Metric::Tpot),
            fmt(timing(one, Metric::Throughput), 1) + if failed(one) > 0 { FAILED } else { "" },
            fmt(med(ladder, Scenario::Cold, Startup::PeakRss), 0),
        ];
        let now = summary_raw(m, run);
        let then = prev
            .filter(|_| e == Engine::Scratchy && same_shape)
            .map(|p| summary_raw(p.m, p.run));
        // An engine with its own quantization says which on its row.
        let name = match served_as(m, e) {
            None => e.label().to_string(),
            Some(s) => format!("{} ({s})", e.label()),
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
    let around = prev
        .map(|p| conditions_changed(run, m, p))
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
                if !around.is_empty() {
                    " " strong { "Also different" } ": {around}."
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
    key: Key,
    value: Option<f64>,
}

/// A line point's x: a number on a linear axis, or its label on a labels axis
/// (Carbon's log axis draws no lines).
#[derive(Serialize)]
#[serde(untagged)]
enum Key {
    Number(u32),
    Label(String),
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
fn chart(spec: &Chart, label: &str, mount: &str) -> Element {
    rsx! {
        div { role: "img", "aria-label": label, style: "{mount}",
            "data-chart": serde_json::to_string(spec).unwrap_or_default(),
        }
    }
}

/// What a sweep's line charts put on x.
#[derive(Clone, Copy, PartialEq)]
enum Sweep {
    /// Offered concurrent users, at the base shape.
    Users,
    /// Prompt tokens, at the base answer size and users.
    Prompt,
}

impl Sweep {
    fn at(self, c: &Cell) -> Option<u32> {
        match (self, c.at) {
            (Sweep::Users, At::Conc(n)) | (Sweep::Prompt, At::Input(n)) => Some(n),
            _ => None,
        }
    }
    fn xlabel(self) -> &'static str {
        match self {
            Sweep::Users => "offered concurrent users",
            Sweep::Prompt => "prompt tokens",
        }
    }
    /// Prompt sizes grow by factors of four, so they go on a labels axis,
    /// evenly spaced; users on a linear one.
    fn scale(self) -> &'static str {
        match self {
            Sweep::Users => "linear",
            Sweep::Prompt => "labels",
        }
    }
    fn key(self, x: u32) -> Key {
        match self {
            Sweep::Users => Key::Number(x),
            Sweep::Prompt => Key::Label(x.to_string()),
        }
    }
    fn metrics(self) -> [(Metric, &'static str, &'static str); 2] {
        match self {
            Sweep::Users => [
                (
                    Metric::Throughput,
                    "Throughput (tok/s, higher is better)",
                    "tok/s",
                ),
                (
                    Metric::Tpot,
                    "Time per output token (ms, lower is better)",
                    "TPOT ms",
                ),
            ],
            Sweep::Prompt => [
                (
                    Metric::Ttft,
                    "Time to first token (ms, lower is better)",
                    "TTFT ms",
                ),
                (
                    Metric::PeakMem,
                    "Peak memory (MiB, lower is better)",
                    "peak MiB",
                ),
            ],
        }
    }
}

/// One line chart: the sweep's x, one line per series.
fn line_chart(
    series: &[Series],
    xs: &[u32],
    (title, ylabel): (&str, &str),
    sweep: Sweep,
    height: &str,
    label: &str,
) -> Element {
    let spec = Chart::Line {
        data: series
            .iter()
            .flat_map(|s| {
                s.pts.iter().map(|&(x, v)| LinePoint {
                    group: &s.label,
                    key: sweep.key(x),
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
                    title: sweep.xlabel(),
                    scale_type: Some(sweep.scale()),
                    ticks: (sweep == Sweep::Users).then_some(Ticks { values: xs }),
                    include_zero: None,
                    visible: None,
                    domain: (sweep == Sweep::Prompt)
                        .then(|| xs.iter().map(u32::to_string).collect()),
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
            height: height.to_string(),
            width: None,
            legend: None,
            locale: None,
            theme: "white",
            toolbar: Enabled { enabled: false },
        },
    };
    chart(&spec, label, FIXED_MOUNT)
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
    chart(&spec, label, FIXED_MOUNT)
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
    chart(&spec, label, "width: 100%; aspect-ratio: 1")
}

/// The ratio heatmaps' colour scale, upright: RATIO_STEPS fastest at the top,
/// labelled at the ends and the middle. Carbon's heatmap only draws its legend
/// as a bar underneath.
fn ratio_scale(fit: ScaleFit) -> Element {
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
        div { style: "{place}display: grid; grid-template-columns: 0.75rem max-content; grid-template-rows: repeat({rows}, {track}); column-gap: 0.5rem; font-size: 0.75rem; line-height: 1",
            for (row, colour) in RATIO_STEPS.iter().rev().enumerate() {
                div { style: "grid-column: 1; grid-row: {row + 1}; background: {colour}" }
            }
            div { style: "grid-column: 2; grid-row: 1; align-self: start",
                Definition { alignment: Alignment::Left,
                    definition: "scratchy at least {RATIO_SPAN} times as fast as the engine the map compares it with; anything faster takes this colour too.",
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
                    definition: "scratchy at most a quarter as fast as the engine the map compares it with, {RATIO_SPAN} times slower; anything slower takes this colour too.",
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

/// A cell's value for a table view, with what its marks mean: how many
/// requests went untimed, or failed.
fn table_value(c: Option<&Cell>, metric: Metric) -> String {
    let note = marks_note(c, metric);
    if note.is_empty() {
        cfmt(c, metric)
    } else {
        format!("{} ({note})", cfmt(c, metric))
    }
}

fn sweep_cells(cells: &[Cell], sweep: Sweep) -> BTreeMap<u32, &Cell> {
    cells
        .iter()
        .filter_map(|c| Some((sweep.at(c)?, c)))
        .collect()
}

/// A sweep as small multiples: per metric, one small chart per engine scratchy
/// is compared with, each scratchy (blue) against that one engine (orange)
/// with scratchy's previous run in grey behind, so however many engines a run
/// has, no chart holds more than three lines.
fn sweep_part<'a>(m: &'a Model, run: &Run, prev: Option<Entry<'a>>, sweep: Sweep) -> Element {
    let map_of = |mm: &'a Model, rr: &Run, e: Engine| -> BTreeMap<u32, &'a Cell> {
        match sweep {
            Sweep::Users => by_users(mm, rr, e),
            Sweep::Prompt => sweep_cells(e.cells(mm), sweep),
        }
    };
    let cells: Vec<(Engine, BTreeMap<u32, &Cell>)> = Engine::ALL
        .iter()
        .map(|&e| (e, map_of(m, run, e)))
        .filter(|(_, cs)| !cs.is_empty())
        .collect();
    let mine = cells
        .iter()
        .find(|(e, _)| *e == Engine::Scratchy)
        .map(|(_, cs)| cs);
    let Some(mine) = mine else {
        return rsx! {};
    };
    let mut xs: Vec<u32> = cells
        .iter()
        .flat_map(|(_, cs)| cs.keys().copied())
        .collect();
    xs.sort_unstable();
    xs.dedup();

    // The previous run's scratchy line, drawn first so it sits behind.
    let pcs = prev
        .filter(|p| sweep == Sweep::Prompt || users_shape(p.m, p.run) == users_shape(m, run))
        .map(|p| map_of(p.m, p.run, Engine::Scratchy))
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
    let line =
        |label: &str, class: &'static str, cs: &BTreeMap<u32, &Cell>, metric: Metric| Series {
            label: label.to_string(),
            class,
            pts: xs
                .iter()
                .filter_map(|x| Some((*x, timing(Some(cs.get(x)?), metric))))
                .collect(),
        };
    let rivals: Vec<&(Engine, BTreeMap<u32, &Cell>)> = cells
        .iter()
        .filter(|(e, _)| *e != Engine::Scratchy)
        .collect();
    let row = |metric: Metric, title: &str, ylabel: &str| -> Element {
        let charts: Vec<(String, Vec<Series>)> = if rivals.is_empty() {
            vec![(
                "scratchy".to_string(),
                vec![line("scratchy", "s1", mine, metric)],
            )]
        } else {
            rivals
                .iter()
                .map(|(e, cs)| {
                    let mut ss = Vec::new();
                    if !pcs.is_empty() {
                        ss.push(line(&plabel, "prev", &pcs, metric));
                    }
                    ss.push(line("scratchy", "s1", mine, metric));
                    ss.push(line(e.label(), "s2", cs, metric));
                    (format!("vs {}", e.label()), ss)
                })
                .collect()
        };
        let charts: Vec<(String, Vec<Series>)> = charts
            .into_iter()
            .map(|(t, mut ss)| {
                ss.retain(|s| s.pts.iter().any(|p| p.1.is_some()));
                (t, ss)
            })
            .filter(|(_, ss)| ss.iter().any(|s| s.class == "s1"))
            .collect();
        // An engine whose line is missing never measured this (no stream).
        let gone: Vec<&str> = rivals
            .iter()
            .filter(|(e, _)| {
                !charts
                    .iter()
                    .any(|(_, ss)| ss.iter().any(|s| s.label == e.label()))
            })
            .map(|(e, _)| e.label())
            .collect();
        rsx! {
            Stack { gap: 2,
                strong { "{title}" }
                div { style: "display: flex; flex-wrap: wrap; gap: 1rem 1.5rem",
                    for (t, ss) in &charts {
                        div { style: "flex: 1 1 16rem; min-width: 16rem; max-width: 24rem",
                            {line_chart(ss, &xs, (t.as_str(), ylabel), sweep, "200px", &format!("{title}, scratchy {t}"))}
                        }
                    }
                }
                if !gone.is_empty() {
                    p { "{gone.join(\", \")}: {NO_STREAM}, not plotted." }
                }
            }
        }
    };
    let xs_text = match xs.split_last() {
        Some((l, rest)) if !rest.is_empty() => format!(
            "{} and {l}",
            rest.iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Some((l, _)) => l.to_string(),
        None => String::new(),
    };
    let (_, output, conc) = base_of(run, m);
    let (ui, uo) = users_shape(m, run);
    let from_grid = if matches!(users_from(m, run), UsersFrom::Grid { .. }) {
        ", the grid's middle cell"
    } else {
        ""
    };
    let (heading, intro) = match sweep {
        Sweep::Users => (
            "As users are added",
            rsx! {
                "Throughput and time per output token with {xs_text} users at once " {users_info()}
                "; {opt(ui)}-token prompts, {opt(uo)}-token answers{from_grid}. One small chart per engine, each scratchy \
                 against that engine, scratchy's previous run in grey. A line that rises in the second row means \
                 each user's answer slows down as more share the engine."
            },
        ),
        Sweep::Prompt => (
            "As prompts get longer",
            rsx! {
                "Time to first token and the server's peak memory with {xs_text}-token prompts; {opt(output)}-token \
                 answers, {opt(conc)} users at once. One small chart per engine, each scratchy against that engine. \
                 Prompts past a model's own context are left out."
            },
        ),
    };
    let mut table_headers = vec![sweep.xlabel().to_string()];
    table_headers.extend(xs.iter().map(u32::to_string));
    let table_metrics: &[(Metric, &str)] = match sweep {
        Sweep::Users => &[
            (Metric::Throughput, "tok/s"),
            (Metric::Ttft, "TTFT ms"),
            (Metric::Tpot, "TPOT ms"),
        ],
        Sweep::Prompt => &[
            (Metric::Ttft, "TTFT ms"),
            (Metric::Throughput, "tok/s"),
            (Metric::PeakMem, "peak MiB"),
        ],
    };
    let [(m1, t1, y1), (m2, t2, y2)] = sweep.metrics();
    part(
        heading,
        intro,
        rsx! {
            Stack { gap: 5,
                {row(m1, t1, y1)}
                {row(m2, t2, y2)}
            }
            Fold { title: "Table view",
                Table { headers: table_headers,
                    for (e, cs) in &cells {
                        for (metric, name) in table_metrics {
                            TableRow {
                                TableCell { "{e.label()} · {name}" }
                                for x in &xs { TableCell { {table_value(cs.get(x).copied(), *metric)} } }
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

/// A model's prompt-by-answer grid at one users count: the axes, each
/// engine's cells, and the engines scratchy is compared against.
struct ShapeGrid<'a> {
    ins: &'a [u32],
    outs: &'a [u32],
    users: u32,
    have: Vec<(Engine, ByShape<'a>)>,
    /// The headline comparison: the first of Engine::RIVALS the run has.
    rival: Option<Engine>,
    /// Every engine the run has a grid for, the rival first.
    present: Vec<Engine>,
}

/// The users counts a model's grid ran at, fewest first. Cells from before
/// the grid had its own count ran at the base count.
fn grid_users(m: &Model, run: &Run) -> Vec<u32> {
    let base = base_of(run, m).2.unwrap_or(0);
    let mut us: Vec<u32> = Engine::Scratchy
        .cells(m)
        .iter()
        .filter(|c| matches!(c.at, At::Grid { .. }))
        .map(|c| c.users.unwrap_or(base))
        .collect();
    us.sort_unstable();
    us.dedup();
    us
}

fn users_text(n: u32) -> String {
    if n == 1 {
        "1 user at a time".to_string()
    } else {
        format!("{n} users at once")
    }
}

impl<'a> ShapeGrid<'a> {
    /// None when scratchy has no grid at that users count.
    fn of(m: &'a Model, run: &'a Run, users: u32) -> Option<Self> {
        let base = base_of(run, m).2.unwrap_or(0);
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
                    .filter(|c| c.users.unwrap_or(base) == users)
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
            users,
            have,
            rival: None,
            present: Vec::new(),
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
            present,
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

/// The marks any of these cells carries, each once.
fn mark(metric: Metric, cells: &[Option<&Cell>]) -> String {
    let mut out = String::new();
    for ch in cells
        .iter()
        .flat_map(|c| marks(*c, metric).chars().collect::<Vec<_>>())
    {
        if !out.contains(ch) {
            out.push(ch);
        }
    }
    out
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
fn heat_map(g: &ShapeGrid, metric: Metric, title: &str) -> Element {
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
    )
}

/// The grid's numbers for one metric, dense: per prompt size one row per
/// engine scratchy is compared with, each cell the ratio and its marks; just
/// scratchy's values with none.
fn heat_table(g: &ShapeGrid, metric: Metric, title: &str) -> Element {
    let mut cols = vec![format!("{title}: prompt ↓ / answer →")];
    cols.extend(g.outs.iter().map(u32::to_string));
    rsx! {
        Table { headers: cols,
            for &i in g.ins {
                if g.present.is_empty() {
                    TableRow {
                        TableCell { "{i}" }
                        for &o in g.outs { TableCell { {table_value(g.at(Some(Engine::Scratchy), (i, o)), metric)} } }
                    }
                }
                for &e in &g.present {
                    TableRow {
                        TableCell { "{i} · vs {e.label()}" }
                        for &o in g.outs {
                            {
                                // The ratio, then what its marks mean, side by side.
                                let notes: Vec<String> = [(Engine::Scratchy, "scratchy"), (e, e.label())]
                                    .into_iter()
                                    .filter_map(|(who, name)| {
                                        let n = marks_note(g.at(Some(who), (i, o)), metric);
                                        (!n.is_empty()).then(|| format!("{name}: {n}"))
                                    })
                                    .collect();
                                let text = g.versus(metric, (i, o), e).1;
                                rsx! { TableCell { if notes.is_empty() { "{text}" } else { "{text} ({notes.join(\"; \")})" } } }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// scratchy against one engine as a tiny ratio map, named under it: the
/// small multiples beside the full maps, one per other engine.
fn versus_mini(g: &ShapeGrid, metric: Metric, e: Engine, label: &str) -> Element {
    let mut data = Vec::new();
    for &i in g.ins {
        for &o in g.outs {
            data.push(HeatPoint {
                x: o.to_string(),
                y: i.to_string(),
                value: g.versus(metric, (i, o), e).0.map(log2_ratio),
            });
        }
    }
    rsx! {
        div { style: "flex: 0 0 7rem; width: 7rem",
            Stack { gap: 1,
                {mini_heatmap((g.ins, g.outs), data, format!("scratchy/{}", e.label()), label)}
                div { style: "text-align: center; font-size: 0.75rem", "vs {e.label()}" }
            }
        }
    }
}

fn grid_maps(m: &Model, run: &Run) -> Element {
    let grids: Vec<ShapeGrid> = grid_users(m, run)
        .into_iter()
        .filter_map(|u| ShapeGrid::of(m, run, u))
        .collect();
    if grids.is_empty() {
        return rsx! {};
    }
    let counts = grids
        .iter()
        .map(|g| users_text(g.users))
        .collect::<Vec<_>>()
        .join(", then ");
    let one = |g: &ShapeGrid| -> Element {
        let note = match g.rival {
            None => {
                "scratchy's own values; this run has no other engine to compare against".to_string()
            }
            Some(rival) => format!(
                "Colour is how many times faster scratchy is than {}: blue is faster, red is slower, the middle \
                 step is within noise. Hover a cell for its ratio{}; the table view has every cell against every engine",
                rival.label(),
                if g.present.len() > 1 {
                    ". The small maps below compare with each other engine"
                } else {
                    ""
                },
            ),
        };
        let others: Vec<Engine> = g.present.iter().copied().skip(1).collect();
        rsx! {
            Stack { gap: 3,
                strong { "{users_text(g.users)}" }
                p { "{note}." }
                div { style: "display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: flex-start",
                    for (metric, title) in GRID_METRICS { {heat_map(g, metric, title)} }
                    if g.rival.is_some() { {ratio_scale(ScaleFit::Fixed)} }
                }
                if !others.is_empty() {
                    for (metric, title) in GRID_METRICS {
                        Stack { gap: 1,
                            span { "{title}, vs every engine" }
                            div { style: "display: flex; flex-wrap: wrap; gap: 1rem",
                                for e in &others { {versus_mini(g, metric, *e, &format!("{title}, scratchy vs {}, {}", e.label(), users_text(g.users)))} }
                            }
                        }
                    }
                }
                Fold { title: "Table view, {users_text(g.users)}",
                    Stack { gap: 4,
                        for (metric, title) in GRID_METRICS { {heat_table(g, metric, title)} }
                    }
                }
            }
        }
    };
    part(
        "Prompt size × answer size",
        rsx! { "Prompt size down the side, answer size across the bottom, measured at {counts}." },
        rsx! {
            Stack { gap: 6,
                for g in &grids { {one(g)} }
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
                                let ladder = Engine::Scratchy.ladder(m);
                                let us = by_users(m, run, Engine::Scratchy);
                                let one = us.get(&1).copied();
                                let many = us.keys().max().filter(|&&n| n != 0).and_then(|n| us.get(n).copied());
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
                headers: headers(["when", "scratchy", "steps", "cold priming launches", "scratchy KV cache", "scratchy serve flags", "engine versions", "pinned", "data"]),
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
                                TableCell { "{engine_versions(run)}" }
                                TableCell { "{pinned(run)}" }
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
    let mini = |chip: &str, stem: &str, metric: Metric, name: &str, users: u32| -> Element {
        let Some(Entry { run, m }) = machines[chip].latest(stem) else {
            return rsx! { "not run" };
        };
        let Some(g) = ShapeGrid::of(m, run, users) else {
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
                    {mini_heatmap((g.ins, g.outs), data, format!("scratchy/{}", rival.label()), &format!("{stem} on {chip}: scratchy {range} vs {}", rival.label()))}
                    // Name over range, centred under the grid, and who it is against.
                    div { style: "text-align: center",
                        div { strong { "{name}" } }
                        div { "{range}" }
                        div { style: "font-size: 0.75rem", "vs {rival.label()}" }
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
    // One block per users count: 1, and each model's base count, of those its
    // grid ran at. The full set per users count is in each model's own section.
    let mut counts: Vec<u32> = machines
        .values()
        .flat_map(|mr| mr.models.iter().filter_map(|(stem, _)| mr.latest(stem)))
        .flat_map(|Entry { run, m }| {
            let base = base_of(run, m).2.unwrap_or(0);
            grid_users(m, run)
                .into_iter()
                .filter(move |&u| u == 1 || u == base)
        })
        .collect();
    counts.sort_unstable();
    counts.dedup();
    rsx! {
        Section { id: "glance", level: 2,
            Stack { gap: 4,
                Heading { "At a glance" }
                p {
                    "Every model's prompt size × answer size grid on every machine, once per users count: in each \
                     cell, the model's prefill (time to first token) grid, then its decode (throughput) grid (rows: \
                     prompt, short to long; columns: answer, short to long). Colour is how many times faster scratchy \
                     is than the engine named under the grid: oMLX TurboQuant where the run has it, else the next \
                     of oMLX, mlx-lm, vllm-metal, ollama, llama.cpp, mistral.rs. Blue faster, red slower, the middle \
                     step within noise. Hover a square for its ratio."
                }
                for users in counts.iter().copied() {
                Heading { "{users_text(users)}" }
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
                                                for (metric, name) in metrics { {mini(chip, stem, metric, name, users)} }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        {ratio_scale(ScaleFit::Beside)}
                    }
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
                {
                    // Models the newest run left out here, and why: memory, so far.
                    let newest = mr.runs[mr.runs.len() - 1];
                    let skipped: Vec<String> = newest.skipped_models.iter().flatten()
                        .filter(|x| mr.latest(&x.stem).is_none())
                        .map(|x| format!("{} ({})", x.stem, x.reason.as_deref().unwrap_or("skipped")))
                        .collect();
                    rsx! { if !skipped.is_empty() { p { "Not run on this machine: {skipped.join(\"; \")}." } } }
                }
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
                            Tile { id: mid,
                              Section { level: 3,
                              Layer { level: 2,
                                Stack { gap: 6,
                                    Stack { gap: 2,
                                        Heading { "{stem} on {chip}" }
                                        p {
                                            a { href: "https://huggingface.co/{m.model_id}", "{m.model_id}" }
                                            " · {m.quant.as_deref().unwrap_or(\"default\")}"
                                            if let Some(rev) = m.model_files.as_ref().and_then(|f| f.revision.as_deref()) {
                                                " · files "
                                                a { href: "https://huggingface.co/{m.model_id}/tree/{rev}", code { "{&rev[..rev.len().min(8)]}" } }
                                            }
                                            "{build} · run {run.generated_utc.date()}, "
                                            code { "{run.repo.sha8()}" }
                                            if let Some(verdict) = parity(m) {
                                                " · {verdict} " {parity_info()}
                                            }
                                        }
                                    }
                                    {summary_table(m, run, prev)}
                                    {sweep_part(m, run, prev, Sweep::Users)}
                                    {grid_maps(m, run)}
                                    {sweep_part(m, run, prev, Sweep::Prompt)}
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
                        "scratchy against oMLX, mlx-lm, vllm-metal, ollama, llama.cpp and mistral.rs on Apple \
                         silicon: how long each takes to start, how fast it answers one user, and how it holds up as \
                         prompts, answers and users grow. \
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
                "scratchy against oMLX, mlx-lm, vllm-metal, ollama, llama.cpp and mistral.rs on Apple silicon: startup, single-user speed, and scaling, per machine.",
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

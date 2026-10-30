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

use crate::Site;
use crate::carbon::{
    Alignment, ContentPanel, ContentSwitcher, ContentSwitcherItem, Definition, Fold, Heading,
    Layer, Section, Stack, TabItem, Table, TableCell, TableRow, Tabs, TabsKind, Tag, TagKind,
    Tile, TipHeading, Toggletip,
};
use crate::charts::{
    Axes, Axis, AxisDomain, BarPoint, Better, Bounds, BubblePoint, Bubbles, Chart, ChartOptions,
    ColorLegend, Colour, ComboLayer, Enabled, Gradient, HeatPoint, HeatmapOptions, Hover,
    LineOptions, Locale, RATIO_RANGE, RATIO_STEPS, Ramp, ScaleFit, SparkAxes, SparkPoint, Step,
    TableAxes, chart, colour, glance_maps, highlight, log2_ratio,
    mini_heatmap, ratio_scale, ratio_step, sparkline,
};
use crate::chrome::{self, Content, Head, Library, Published, REPO, Root, Tab};
use data::{At, Bytes, Cell, Model, RequestTiming, Run, Scenario};
use numbers::{
    BELOW_WEIGHTS, DASH, Engine, FAILED, Metric, NO_STREAM, NOISE, SHORT, Startup, cfmt,
    failed_note, fmt, lfmt, marks, marks_note, med, peak_rss, rss_fmt, served, timing,
};

/// The run files, relative to site/; copied to the same place under _site/,
/// where each run table links to its own.
pub const DATA: &str = "data/metal";
/// The runs a newer one replaced, relative to site/: only Over time reads them.
pub const HISTORY: &str = "data/metal/history";

// --------------------------------------------------------------------- load

/// Every run file, checked. Fails naming each bad file, so one build reports
/// all of them rather than the first.
fn load(site: &Site) -> Result<Vec<Run>, String> {
    load_from(site, &[DATA])
}

/// [`load`], over every run file in `dirs` (relative to the site): the page's own runs, or those
/// and the ones a newer run replaced ([`HISTORY`]), for Over time.
fn load_from(site: &Site, dirs: &[&str]) -> Result<Vec<Run>, String> {
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    for dir in dirs {
        let dir = site.root.join(dir);
        // history/ may not exist yet; the page's own folder must.
        let Ok(entries) = fs::read_dir(&dir) else {
            if *dir == *site.root.join(DATA) {
                return Err(format!("{}: not found", dir.display()));
            }
            continue;
        };
        files.extend(
            entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "json")),
        );
    }
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
        run.mark_dead();
        run.mark_short();
        run.mark_mem_below_weights();
        run.file = f;
        runs.push(run);
    }
    if errors.is_empty() {
        borrow_earlier(&mut runs);
        Ok(runs)
    } else {
        Err(errors
            .iter()
            .map(|e| format!("metal page: {e}"))
            .collect::<Vec<_>>()
            .join("\n"))
    }
}

/// Prompts, users counts and answer lengths of a run's users × prompt maps, and its base cell.
type Compared<'a> = (Vec<u32>, Vec<u32>, Vec<u32>, Option<&'a data::Base>);

/// The cells two runs' heat maps and users curve compare scratchy at: the users × prompt map at
/// each answer length run above 1 user, and the base cell. Equal for two runs whose numbers line
/// up, whatever else either swept.
fn compared_cells(run: &Run) -> Option<Compared<'_>> {
    let sc = run.config.scaling.as_ref()?;
    let users = sc.grid_conc.clone().unwrap_or_default();
    // Runs from before `grid_users_output` ran every answer length at every users count.
    let answers = (sc.grid_users_output.clone())
        .or_else(|| sc.grid_output.clone())
        .unwrap_or_default();
    Some((
        sc.grid_input.clone().unwrap_or_default(),
        users,
        answers,
        sc.base.as_ref(),
    ))
}

/// Fill each model's newest numbers on a machine from its earlier runs there, where a run left
/// them out: a run of scratchy alone keeps the other engines' last numbers, and a run without the
/// frozen rung keeps scratchy's last frozen launches. Each one filled in is recorded in
/// [`Model::borrowed`] with the run it came from, which the page shows beside it.
///
/// Every run keeps what it measured. A later run that left an engine out shows that engine's
/// newest earlier numbers, when that earlier run measured the cells this one compares
/// ([`compared_cells`]): a run of another shape (other prompts, users counts or answer lengths
/// across users) measured other cells, and its numbers stay with it. A full run's 512- and
/// 2048-token answers at 1 user are not compared by a quick run, so they do not stop it
/// borrowing. scratchy's own scaling cells and its cold and warm launches are never
/// filled: they are always the run's own, so scratchy is only ever compared with itself across
/// runs by the previous-run badges. Its frozen launches have no shape and are filled whatever the
/// grid.
fn borrow_earlier(runs: &mut [Run]) {
    let mut order: Vec<usize> = (0..runs.len()).collect();
    order.sort_by(|&a, &b| runs[a].generated_utc.cmp(&runs[b].generated_utc));
    let shapes: Vec<_> = runs.iter().map(compared_cells).collect();
    let shape_of: Vec<usize> = (0..runs.len())
        .map(|i| (0..=i).find(|&j| shapes[j] == shapes[i]).unwrap_or(i))
        .collect();
    // Per (machine, model): each engine's numbers, with when and the shape they were measured at,
    // and scratchy's frozen launches, with when, as last measured.
    type Measured = (
        Option<Vec<data::Rep>>,
        Option<Vec<data::Cell>>,
        data::Utc,
        usize,
    );
    let mut engines: BTreeMap<(String, String, &str), Measured> = BTreeMap::new();
    let mut frozen: BTreeMap<(String, String), (Vec<data::Rep>, data::Utc)> = BTreeMap::new();
    for i in order {
        let run = &mut runs[i];
        let (chip, when) = (
            run.machine.chip.as_str().to_string(),
            run.generated_utc.clone(),
        );
        for m in &mut run.models {
            let at = (chip.clone(), m.stem.clone());
            for e in Engine::RIVALS {
                let key = (at.0.clone(), at.1.clone(), e.key());
                let (ladder, cells) = e.fields_mut(m);
                let measured = ladder.as_ref().is_some_and(|l| !l.is_empty())
                    || cells.as_ref().is_some_and(|c| !c.is_empty());
                if measured {
                    let mine = (ladder.clone(), cells.clone(), when.clone(), shape_of[i]);
                    engines.insert(key, mine);
                } else if let Some((l, c, from, shape)) = engines.get(&key)
                    && *shape == shape_of[i]
                {
                    (*ladder, *cells) = (l.clone(), c.clone());
                    m.borrowed.insert(e.key().to_string(), from.clone());
                }
            }
            let own: Vec<data::Rep> = (Engine::Scratchy.ladder(m).iter())
                .filter(|r| r.scenario == Scenario::Frozen)
                .cloned()
                .collect();
            if !own.is_empty() {
                frozen.insert(at, (own, when.clone()));
            } else if let Some((reps, from)) = frozen.get(&at).filter(|_| served(m)) {
                (m.cache_ladder.get_or_insert_with(Vec::new)).extend(reps.iter().cloned());
                m.borrowed
                    .insert(data::BORROWED_FROZEN.to_string(), from.clone());
            }
        }
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

/// The machines in page order: smallest memory first, then by chip name, so a row reads up from
/// the small set's Macs to the large set's.
fn in_order<'a>(machines: &BTreeMap<&'a str, MachineRuns<'a>>) -> Vec<&'a str> {
    let memory = |chip: &str| {
        machines[chip].runs[machines[chip].runs.len() - 1]
            .machine
            .memory_gb
    };
    let mut chips: Vec<&str> = machines.keys().copied().collect();
    chips.sort_by_key(|&c| (memory(c), c));
    chips
}

// ------------------------------------------------------------------- render

/// A grey tag on a number an earlier run measured, saying when ([`Model::borrowed`]).
fn from_run(when: &data::Utc) -> Element {
    let day = when.month_day();
    rsx! { Tag { kind: TagKind::Gray, title: "measured by an earlier run, on {day}: this run did not measure it", "from {day}" } }
}

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
        (peak_rss(m, Engine::Scratchy, Scenario::Cold).mib(), false),
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
/// dimension (#349), at the base prompt where the grid ran it (its middle one
/// otherwise) and the answer length it ran at every users count.
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
    let ins = sc.and_then(|s| s.grid_input.as_ref());
    let base = base_of(run, m)
        .0
        .filter(|b| ins.is_some_and(|i| i.contains(b)));
    let output = match sc.and_then(|s| s.grid_users_output.as_ref()) {
        Some(_) => Some(glance_answer(run)),
        None => mid(sc.and_then(|s| s.grid_output.as_ref())),
    };
    match (base.or_else(|| mid(ins)), output) {
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
    // The MLX engines serve scratchy's own checkpoint (`Engine::serves_mlx_checkpoint`): nothing to
    // add. The others serve their own build, named here.
    let model = match e {
        _ if e.serves_mlx_checkpoint() => None,
        Engine::Ollama => model,
        _ => model.map(|m| m.rsplit(':').next().unwrap_or_default().to_string()),
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
fn parity(m: &Model) -> Option<String> {
    m.parity_mlx_lm.map(|p| match (p, &m.parity_skipped) {
        (None, Some(why)) => format!("output not checked against mlx-lm: {why}"),
        (Some(true), _) => match &m.parity_ties {
            Some(ties) if !ties.is_empty() => {
                let (n, max) = (ties.values().flatten())
                    .fold((0, 0.0_f64), |(n, max), &g| (n + 1, max.max(g)));
                format!(
                    "output matches mlx-lm up to {n} near-tie token{} (at most {max:.2} logits \
                     behind mlx-lm's top)",
                    if n == 1 { "" } else { "s" }
                )
            }
            _ => "output matches mlx-lm".to_string(),
        },
        (Some(false), _) => "output does not match mlx-lm".to_string(),
        (None, None) => "output not checked (no mlx-lm)".to_string(),
    })
}

fn parity_info() -> Element {
    info(
        "",
        rsx! {
            "Before anything is timed, scratchy and mlx-lm answer the same short prompts with greedy decoding, \
             and the answers must match. Where they part, mlx-lm is run over scratchy's own answer and judges \
             each token scratchy picked: one within a logit of mlx-lm's top is a near-tie two correct stacks \
             break differently, and passes. A token further behind points at a wrong load path (quant preset, \
             group size, dequant), so that model is not timed at all."
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
                       memory), so every number in that cell covers only the rest. "
            strong { "{SHORT}" } ": the answers came back short of the tokens asked for. Every request asks the \
                       engine to ignore EOS and answer to length; oMLX, ollama and sometimes mlx-lm stop at EOS \
                       anyway, so the cell measured less decode than its label says, which flatters tok/s (the \
                       table views say what share)."
        },
    )
}

fn rss_info() -> Element {
    info(
        "",
        rsx! {
            "The server process at its peak. For ollama, " code { "ollama serve" } " only; its runner process is not counted. "
            strong { "{BELOW_WEIGHTS}" } ": less than the model's own weights. A process serving the model holds at \
                       least those, so the reading can't be all of its memory, and is left out (#414). The same rule \
                       leaves out a peak-memory reading on the prompt-length chart."
        },
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
             with 4-bit TurboQuant KV), mlx-lm, vllm-metal and Splash run the same MLX checkpoint as scratchy \
             (Splash on the Qwen models only, with the draft model it ships); ollama, llama.cpp and mistral.rs run \
             their own quantizations, named on their rows. Engine versions, and \
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
            cfmt(one, Metric::Throughput),
            rss_fmt(m, e, Scenario::Cold),
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
        // Numbers an earlier run measured say when; a borrowed value has no change to show.
        let from = m.borrowed.get(e.key());
        let frozen_from = (e == Engine::Scratchy)
            .then(|| m.borrowed.get(data::BORROWED_FROZEN))
            .flatten();
        Some(rsx! {
            TableRow {
                TableCell { "{name}" if let Some(w) = from { " " {from_run(w)} } }
                for (i, v) in vals.iter().enumerate() {
                    TableCell { "{v} "
                        if i == 0 && let Some(w) = frozen_from {
                            {from_run(w)}
                        } else if let Some(then) = then {
                            {delta(now[i].0, then[i].0, now[i].1)}
                        }
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

/// What a sweep's charts put on x.
#[derive(Clone, Copy, PartialEq)]
enum Sweep {
    /// Offered concurrent users, at the base shape (or the grid's middle cell).
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
    fn metrics(self) -> [(Metric, &'static str, &'static str); 2] {
        match self {
            Sweep::Users => [
                (
                    Metric::Throughput,
                    "Throughput (tok/s, higher is better)",
                    "tok/s",
                ),
                (
                    Metric::E2el,
                    "End-to-end time (ms, lower is better)",
                    "E2EL ms",
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

/// One bar chart: the sweep's points on x, each a category in the order
/// measured, a bar per series at each.
fn bar_chart(
    series: &[Series],
    xs: &[u32],
    (title, ylabel): (&str, &str),
    sweep: Sweep,
    height: &str,
    label: &str,
) -> Element {
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
                    title: sweep.xlabel(),
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
            height: height.to_string(),
            width: None,
            legend: None,
            locale: None,
            combo: None,
            bounds: None,
            bubble: None,
            line: None,
            hover: None,
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
    grid: TableAxes,
    data: Vec<HeatPoint>,
    total_label: String,
    ratio: bool,
    label: &str,
    scale: Option<&str>,
) -> Element {
    let (rows, columns) = (grid.down.1.len(), grid.across.1.len());
    let spec = Chart::Heatmap {
        data,
        options: ChartOptions {
            title,
            axes: grid.axes(None),
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
            height: format!("{}px", if ratio { 100 } else { 160 } + 56 * rows),
            width: Some(format!("{}px", 140 + 72 * columns)),
            legend: ratio.then_some(Enabled { enabled: false }),
            locale: Locale::total(total_label),
            combo: None,
            bounds: None,
            bubble: None,
            line: None,
            hover: None,
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
    // Every engine that ran this model, scratchy first, with its cells at
    // this sweep's source (empty when it has none there).
    let ran: Vec<(Engine, BTreeMap<u32, &Cell>)> = Engine::ALL
        .iter()
        .filter(|&&e| e.ran(m))
        .map(|&e| (e, map_of(m, run, e)))
        .collect();
    // A sweep no engine has a cell for was not run: no section. Otherwise the
    // section stays, and says who is missing, rather than vanish.
    if ran.iter().all(|(_, cs)| cs.is_empty()) {
        return rsx! {};
    }
    let cells: Vec<&(Engine, BTreeMap<u32, &Cell>)> =
        ran.iter().filter(|(_, cs)| !cs.is_empty()).collect();
    let empty = BTreeMap::new();
    let mine = ran
        .iter()
        .find(|(e, _)| *e == Engine::Scratchy)
        .map_or(&empty, |(_, cs)| cs);
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
    let rivals: Vec<&(Engine, BTreeMap<u32, &Cell>)> =
        ran.iter().filter(|(e, _)| *e != Engine::Scratchy).collect();
    let row = |metric: Metric, title: &str, ylabel: &str| -> Element {
        let plotted =
            |cs: &BTreeMap<u32, &Cell>| cs.values().any(|c| timing(Some(c), metric).is_some());
        // One chart per engine scratchy is compared with, kept while either
        // side has a point to draw; scratchy alone when there is none.
        let pairs: Vec<(String, Vec<Series>)> = if rivals.is_empty() {
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
        let charts: Vec<(String, Vec<Series>)> = pairs
            .into_iter()
            .map(|(t, mut ss)| {
                ss.retain(|s| s.pts.iter().any(|p| p.1.is_some()));
                (t, ss)
            })
            .filter(|(_, ss)| ss.iter().any(|s| s.class != "prev"))
            .collect();
        // Each engine not plotted, by why: no cells at all, every cell from a server that had
        // stopped answering, or cells whose answers never streamed.
        let (mut no_cells, mut dead, mut no_stream) = (Vec::new(), Vec::new(), Vec::new());
        for (e, cs) in ran.iter().filter(|(_, cs)| !plotted(cs)) {
            if cs.is_empty() {
                no_cells.push(e.label())
            } else if cs.values().all(|c| c.dead) {
                dead.push(e.label())
            } else {
                no_stream.push(e.label())
            }
        }
        rsx! {
            Stack { gap: 2,
                strong { "{title}" }
                div { style: "display: flex; flex-wrap: wrap; gap: 1rem 1.5rem",
                    for (t, ss) in &charts {
                        div { style: "flex: 1 1 16rem; min-width: 16rem; max-width: 24rem",
                            {bar_chart(ss, &xs, (t.as_str(), ylabel), sweep, "200px", &format!("{title}, scratchy {t}"))}
                        }
                    }
                }
                if !dead.is_empty() {
                    p { "{dead.join(\", \")}: the server had stopped answering, so every request failed; not plotted." }
                }
                if !no_stream.is_empty() {
                    p { "{no_stream.join(\", \")}: {NO_STREAM}, not plotted." }
                }
                if !no_cells.is_empty() {
                    p { "{no_cells.join(\", \")}: no numbers for this sweep (its requests failed or it was not measured; see the run's logs), not plotted." }
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
                "Throughput and end-to-end time with {xs_text} users at once " {users_info()}
                "; {opt(ui)}-token prompts, {opt(uo)}-token answers{from_grid}. One small chart per engine, each scratchy \
                 against that engine, scratchy's previous run in grey. A line that rises in the second row means \
                 each request takes longer as more share the engine; once the GPU is busy, scheduling only moves \
                 that wait between the first token and the rest, so the end-to-end time is what is compared."
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
            (Metric::E2el, "E2EL ms"),
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

/// One engine's cells in one slice of the grid, by (row, column).
type ByShape<'a> = BTreeMap<(u32, u32), &'a Cell>;

/// One slice of a model's grid (users × prompt × answer tokens): one
/// dimension held at one value, the other two read like a table.
#[derive(Clone, Copy, PartialEq)]
enum Slice {
    /// Users at once down, prompt tokens across, at one answer length: the
    /// heat maps, read like spyre.html's batch by context.
    Answer(u32),
    /// Prompt tokens down, answer tokens across, at one users count: what the
    /// prompt and answer sparklines average over.
    Users(u32),
}

impl Slice {
    /// Where a cell sits in this slice, (row, column), if it is in it. Cells
    /// from before the grid had its own users count ran at `base`.
    fn place(self, c: &Cell, base: u32) -> Option<(u32, u32)> {
        let At::Grid { input, output } = c.at else {
            return None;
        };
        let users = c.users.unwrap_or(base);
        match self {
            Slice::Answer(a) => (output == a).then_some((users, input)),
            Slice::Users(u) => (users == u).then_some((input, output)),
        }
    }

    /// Its rows and columns, each smallest first: the smallest on top.
    fn axes(self, down: &[u32], across: &[u32]) -> TableAxes<'static> {
        let (d, a) = match self {
            Slice::Answer(_) => ("users at once", "prompt tokens"),
            Slice::Users(_) => ("prompt tokens", "answer tokens"),
        };
        TableAxes {
            across: (a, across.iter().map(u32::to_string).collect()),
            down: (d, down.iter().map(u32::to_string).collect()),
        }
    }
}

/// A model's grid at one slice: its rows and columns, each engine's cells,
/// and the engines scratchy is compared against.
#[derive(Clone)]
struct ShapeGrid<'a> {
    slice: Slice,
    down: Vec<u32>,
    across: Vec<u32>,
    have: Vec<(Engine, ByShape<'a>)>,
    /// The headline comparison: the first of Engine::RIVALS with a live cell here.
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

/// The users count the prompt and answer sparklines follow a model's grid
/// at: one user at a time when its grid ran at 1 (#349), else the count it
/// ran at (the base count, in runs from before the grid had its own).
fn glance_users(m: &Model, run: &Run) -> u32 {
    let us = grid_users(m, run);
    if us.contains(&1) {
        1
    } else {
        us.first().copied().unwrap_or(0)
    }
}

/// The answer length the overview's heat maps show a run's grid at: the one it ran at every users
/// count (`grid_users_output`, 64 tokens since that field), else the 128 of the runs before it.
fn glance_answer(run: &Run) -> u32 {
    (run.config.scaling.as_ref())
        .and_then(|s| s.grid_users_output.as_ref())
        .and_then(|os| os.first().copied())
        .unwrap_or(128)
}

/// A run's grid answer lengths across its users counts, each a users × prompt
/// map: the overview's first, then the rest, shortest first. A run that ran
/// some lengths only at 1 user (`grid_users_output`) has no map for those.
fn grid_answers(run: &Run) -> Vec<u32> {
    let sc = run.config.scaling.as_ref();
    let mut os = (sc.and_then(|s| s.grid_users_output.clone()))
        .or_else(|| sc.and_then(|s| s.grid_output.clone()))
        .unwrap_or_default();
    let glance = glance_answer(run);
    os.sort_unstable_by_key(|&o| (o != glance, o));
    os.dedup();
    os
}

impl<'a> ShapeGrid<'a> {
    /// None when scratchy has no cells in that slice.
    fn of(m: &'a Model, run: &'a Run, slice: Slice) -> Option<Self> {
        let base = base_of(run, m).2.unwrap_or(0);
        let sc = run.config.scaling.as_ref()?;
        let (ins, outs) = (
            (sc.grid_input.iter().flatten().copied())
                .filter(|&i| sc.max_prompt.is_none_or(|max| i <= max))
                .collect::<Vec<_>>(),
            sc.grid_output.clone().unwrap_or_default(),
        );
        let (down, across) = match slice {
            Slice::Answer(_) => (grid_users(m, run), ins),
            Slice::Users(_) => (ins, outs),
        };
        let have: Vec<_> = Engine::ALL
            .iter()
            .map(|&e| {
                let cs = e
                    .cells(m)
                    .iter()
                    .filter_map(|c| Some((slice.place(c, base)?, c)))
                    .collect();
                (e, cs)
            })
            .collect();
        let g = ShapeGrid {
            slice,
            down,
            across,
            have,
            rival: None,
            present: Vec::new(),
        };
        if g.cells(Engine::Scratchy).is_empty() || g.down.is_empty() || g.across.is_empty() {
            return None;
        }
        let present: Vec<Engine> = Engine::RIVALS
            .into_iter()
            .filter(|&e| !g.cells(e).is_empty())
            .collect();
        // The headline comparison is the first rival with a live cell: one whose server stopped
        // answering for the whole slice has nothing to compare against, while a later one does.
        let rival = (present.iter().copied())
            .find(|&e| g.cells(e).values().any(|c| !c.dead))
            .or(present.first().copied());
        Some(ShapeGrid {
            rival,
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

    fn axes(&self) -> TableAxes<'static> {
        self.slice.axes(&self.down, &self.across)
    }

    /// Only the users counts `load` scores ([`Load`]); a slice at one users count is kept whole
    /// where `load` scores that count, and emptied where it doesn't.
    fn under(&self, load: Load) -> Self {
        let down = match self.slice {
            Slice::Answer(_) => load.users(&self.down),
            Slice::Users(u) if Load::of(u) != load => Vec::new(),
            Slice::Users(_) => self.down.clone(),
        };
        ShapeGrid {
            down,
            ..self.clone()
        }
    }

    /// Every (row, column), row by row.
    fn rungs(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.down
            .iter()
            .flat_map(|&d| self.across.iter().map(move |&a| (d, a)))
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
    UsersEndToEnd,
    PromptPrefill,
    AnswerDecode,
}

impl Trend {
    fn metric(self) -> Metric {
        match self {
            Trend::UsersDecode | Trend::AnswerDecode => Metric::Throughput,
            Trend::UsersEndToEnd => Metric::E2el,
            Trend::PromptPrefill => Metric::Ttft,
        }
    }

    fn title(self) -> &'static str {
        match self {
            Trend::UsersDecode => "Concurrency scaling — Decode",
            Trend::UsersEndToEnd => "Concurrency scaling — End-to-end",
            Trend::PromptPrefill => "Input length scaling — Prefill",
            Trend::AnswerDecode => "Output length scaling — Decode",
        }
    }

    /// Its name over its sparkline in the overview.
    fn label(self) -> &'static str {
        match self {
            Trend::UsersDecode | Trend::AnswerDecode => "Decode",
            Trend::UsersEndToEnd => "End-to-end",
            Trend::PromptPrefill => "Prefill",
        }
    }

    /// The x axis's title: not printed (styles.css), but Carbon's tooltip
    /// label for the x value.
    fn along(self) -> &'static str {
        match self {
            Trend::UsersDecode | Trend::UsersEndToEnd => "users at once",
            Trend::PromptPrefill => "prompt tokens",
            Trend::AnswerDecode => "answer tokens",
        }
    }

    fn axes(self) -> SparkAxes {
        let metric = self.metric();
        SparkAxes {
            better: metric.better(),
            along: self.along(),
            value: match metric {
                Metric::Ttft => "TTFT ms",
                Metric::E2el => "E2EL ms",
                _ => metric.unit(),
            },
            carry: false,
        }
    }

    /// At each step of the dimension, scratchy's value and its rival's, and
    /// the rival; None with nothing to compare. Prompt and answer lengths are
    /// each measured at three lengths of the other: a step's value is the
    /// geometric mean over those.
    fn points(self, m: &Model, run: &Run) -> Option<(Engine, Vec<Step>)> {
        let metric = self.metric();
        match self {
            Trend::UsersDecode | Trend::UsersEndToEnd => {
                let mine = by_users(m, run, Engine::Scratchy);
                // The first rival with a live cell along users, as `ShapeGrid::of` picks.
                let rival = Engine::RIVALS
                    .into_iter()
                    .find(|&e| by_users(m, run, e).values().any(|c| !c.dead))?;
                let theirs = by_users(m, run, rival);
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
                let g = ShapeGrid::of(m, run, Slice::Users(glance_users(m, run)))?;
                let rival = g.rival?;
                let prompt = self == Trend::PromptPrefill;
                let (along, across) = if prompt {
                    (&g.down, &g.across)
                } else {
                    (&g.across, &g.down)
                };
                let pts = along
                    .iter()
                    .map(|&a| {
                        // Only the rungs both engines measured: averaging each over its own would
                        // compare different cells, and a dead or unstreamed cell left out of one
                        // side would move that side's mean alone.
                        let both: Vec<(f64, f64)> = (across.iter())
                            .filter_map(|&b| {
                                let at = if prompt { (a, b) } else { (b, a) };
                                let value = |e: Engine| timing(g.at(Some(e), at), metric);
                                Some((value(Engine::Scratchy)?, value(rival)?))
                            })
                            .collect();
                        Step {
                            x: f64::from(a),
                            scratchy: geometric_mean(both.iter().map(|&(s, _)| s)),
                            rival: geometric_mean(both.iter().map(|&(_, r)| r)),
                            highlight: false,
                        }
                    })
                    .collect();
                Some((rival, pts))
            }
        }
    }
}

/// The geometric mean, the right average for rates and times; None of nothing.
fn geometric_mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let (sum, n) = values.fold((0.0, 0u32), |(s, n), v| (s + v.ln(), n + 1));
    (n > 0).then(|| (sum / f64::from(n)).exp())
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
    sparkline(
        &spark.steps,
        at.trend.axes(),
        Engine::Scratchy.side(),
        spark.rival.side(),
        None,
        height,
        &label,
    )
}

/// What a users × prompt grid scores, at the users counts it scores it at ([`Load`]).
#[derive(Clone, Copy)]
struct GridMetric {
    metric: Metric,
    load: Load,
    /// Its name: the head of its row's tooltip in the overview, and in each square's.
    name: &'static str,
    /// What it measures, under its name in its row's tooltip; under load, after the users count.
    about: &'static str,
    /// A model section's title for its full map.
    title: &'static str,
    /// The one map's row label: on its own at 1 user, after the users count under load ("2u").
    letter: &'static str,
}

/// Prefill and decode at 1 user, then the end-to-end time under load: the overview and every
/// model section alike.
const GRID_METRICS: [GridMetric; 3] = [
    GridMetric {
        metric: Metric::Ttft,
        load: Load::Alone,
        name: "Prefill",
        about: "Time to first token with 1 user: how long the prompt takes before the answer starts.",
        title: "time to first token, 1 user",
        letter: "P",
    },
    GridMetric {
        metric: Metric::Tpot,
        load: Load::Alone,
        name: "Decode",
        about: "Time per output token with 1 user: how fast the answer streams once it has started.",
        title: "time per output token, 1 user",
        letter: "D",
    },
    GridMetric {
        metric: Metric::E2el,
        load: Load::Shared,
        name: "End-to-end Latency",
        about: "From sending a request to its last token. Once the GPU is busy, scheduling only moves \
                each request's wait between its first token and the rest, so their sum is what compares.",
        title: "end-to-end time, 2 or more users",
        letter: "u",
    },
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

/// Why a cell has no ratio: not measured, measured a server that had stopped answering, or never
/// streamed.
fn why(c: Option<&Cell>, name: &str) -> String {
    match c {
        None => format!("no {name}"),
        Some(c) if c.dead => format!("{name}: {}", failed_note(Some(c))),
        Some(_) => format!("{name}: {NO_STREAM}"),
    }
}

/// One model's grid for one metric, coloured by scratchy against its rival.
fn heat_map(g: &ShapeGrid, metric: Metric, title: &str, scale: &str) -> Element {
    let data = g
        .rungs()
        .map(|(d, a)| HeatPoint {
            x: a.to_string(),
            y: d.to_string(),
            value: match g.rival {
                Some(rival) => g.versus(metric, (d, a), rival).0.map(log2_ratio),
                // Nothing to compare against: scratchy's own value.
                None => timing(g.at(Some(Engine::Scratchy), (d, a)), metric),
            },
            // A ratio grid has room for every square's factor.
            highlight: g.rival.is_some(),
        })
        .collect();
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
        g.axes(),
        data,
        total_label,
        g.rival.is_some(),
        title,
        g.rival.is_some().then_some(scale),
    )
}

/// The grid's numbers for one metric, dense: per row one row per engine
/// scratchy is compared with, each cell the ratio and its marks; just
/// scratchy's values with none.
fn heat_table(g: &ShapeGrid, metric: Metric, title: &str) -> Element {
    let axes = g.axes();
    let mut cols = vec![format!("{title}: {} ↓ / {} →", axes.down.0, axes.across.0)];
    cols.extend(axes.across.1);
    rsx! {
        Table { headers: cols,
            for &d in &g.down {
                if g.present.is_empty() {
                    TableRow {
                        TableCell { "{d}" }
                        for &a in &g.across { TableCell { {table_value(g.at(Some(Engine::Scratchy), (d, a)), metric)} } }
                    }
                }
                for &e in &g.present {
                    TableRow {
                        TableCell { "{d} · vs {e.label()}" }
                        for &a in &g.across {
                            {
                                // The ratio, then what its marks mean, side by side.
                                let notes: Vec<String> = [(Engine::Scratchy, "scratchy"), (e, e.label())]
                                    .into_iter()
                                    .filter_map(|(who, name)| {
                                        let n = marks_note(g.at(Some(who), (d, a)), metric);
                                        (!n.is_empty()).then(|| format!("{name}: {n}"))
                                    })
                                    .collect();
                                let text = g.versus(metric, (d, a), e).1;
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
    let data = g
        .rungs()
        .map(|(d, a)| HeatPoint {
            x: a.to_string(),
            y: d.to_string(),
            value: g.versus(metric, (d, a), e).0.map(log2_ratio),
            highlight: false,
        })
        .collect();
    rsx! {
        div { style: "flex: 0 0 7rem; width: 7rem",
            Stack { gap: 1,
                {mini_heatmap(g.axes(), data, format!("scratchy/{}", e.label()), label)}
                div { style: "text-align: center; font-size: 0.75rem", "vs {e.label()}" }
            }
        }
    }
}

fn grid_maps(m: &Model, mid: &str, run: &Run) -> Element {
    let grids: Vec<(u32, ShapeGrid)> = grid_answers(run)
        .into_iter()
        .filter_map(|a| Some((a, ShapeGrid::of(m, run, Slice::Answer(a))?)))
        .collect();
    if grids.is_empty() {
        return rsx! {};
    }
    let answers = grids
        .iter()
        .map(|(a, _)| a.to_string())
        .collect::<Vec<_>>()
        .join(", then ");
    let one = |a: u32, g: &ShapeGrid| -> Element {
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
        // Each metric's map over the users counts it scores, where the grid ran any.
        let parts: Vec<(GridMetric, ShapeGrid)> = (GRID_METRICS.iter())
            .map(|&gm| (gm, g.under(gm.load)))
            .filter(|(_, s)| !s.down.is_empty())
            .collect();
        rsx! {
            Stack { gap: 3,
                strong { "{a}-token answers" }
                p { "{note}." }
                div { style: "display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: flex-start",
                    for (gm, s) in &parts { {heat_map(s, gm.metric, gm.title, &format!("scale-{mid}-{a}"))} }
                    if g.rival.is_some() { {ratio_scale(ScaleFit::Fixed, &format!("scale-{mid}-{a}"))} }
                }
                if !others.is_empty() {
                    for (gm, s) in &parts {
                        Stack { gap: 1,
                            span { "{gm.title}, vs every engine" }
                            div { style: "display: flex; flex-wrap: wrap; gap: 1rem",
                                for e in &others { {versus_mini(s, gm.metric, *e, &format!("{}, scratchy vs {}, {a}-token answers", gm.title, e.label()))} }
                            }
                        }
                    }
                }
                Fold { title: "Table view, {a}-token answers",
                    Stack { gap: 4,
                        for (gm, s) in &parts { {heat_table(s, gm.metric, gm.title)} }
                    }
                }
            }
        }
    };
    part(
        "Users × prompt size",
        rsx! {
            "Users at once down the side, fewest first; prompt size across the bottom, shortest first. One grid \
             per answer length: {answers} tokens."
            if let Some(max) = run.config.scaling.as_ref().and_then(|s| s.max_prompt) {
                " Prompts longer than {max} tokens were not run: this Mac's memory does not hold them."
            }
            if let Some(max) = run.config.scaling.as_ref().and_then(|s| s.max_prompt_users) {
                " Above 1 user, prompts longer than {max} tokens were not run either: their blank squares were \
                 left out for memory, not measured."
            }
            if let Some(u) = run.config.scratchy_memory_util {
                " scratchy ran at --gpu-memory-utilization {u} for this Mac's memory."
            }
        },
        rsx! {
            Stack { gap: 6,
                for (a, g) in &grids { {one(*a, g)} }
            }
        },
    )
}

/// A grid cell's place in the request plots: users at once, then answer tokens, then prompt
/// tokens, each fewest first.
type CellKey = (u32, u32, u32);

fn cell_key(c: &Cell, base: u32) -> Option<CellKey> {
    match c.at {
        At::Grid { input, output } => Some((c.users.unwrap_or(base), output, input)),
        At::Conc(_) | At::Input(_) | At::Output(_) => None,
    }
}

/// One model's requests in one run: its grid cells that kept every request, scratchy's and the
/// first rival's that did too.
struct Latency<'a> {
    m: &'a Model,
    base: u32,
    engines: Vec<Engine>,
    keys: Vec<CellKey>,
}

/// A latency chart's extent along each timing, [from, to] in ms: (TTFT, TPOT).
type Span = ([f64; 2], [f64; 2]);

impl<'a> Latency<'a> {
    /// None where scratchy kept no request (runs from before the run file kept them).
    fn of(m: &'a Model, run: &Run) -> Option<Self> {
        let base = base_of(run, m).2.unwrap_or(0);
        let mut keys: Vec<CellKey> = (Engine::Scratchy.cells(m).iter())
            .filter(|c| c.requests.is_some())
            .filter_map(|c| cell_key(c, base))
            .collect();
        keys.sort_unstable();
        keys.dedup();
        let rival =
            (Engine::RIVALS.into_iter()).find(|e| e.cells(m).iter().any(|c| c.requests.is_some()));
        (!keys.is_empty()).then(|| Latency {
            m,
            base,
            engines: std::iter::once(Engine::Scratchy).chain(rival).collect(),
            keys,
        })
    }

    fn cell(&self, e: Engine, k: CellKey) -> Option<&'a Cell> {
        (e.cells(self.m).iter()).find(|c| cell_key(c, self.base) == Some(k))
    }

    /// Each engine's requests at one cell that have both timings: (engine, TTFT, TPOT).
    fn requests(&self, k: CellKey) -> Vec<(Engine, f64, f64)> {
        (self.engines.iter())
            .filter_map(|&e| self.cell(e, k)?.requests.as_ref().map(|rs| (e, rs)))
            .flat_map(|(e, rs)| {
                (rs.of(RequestTiming::Ttft).zip(rs.of(RequestTiming::Tpot)))
                    .filter_map(move |((_, t), (_, p))| Some((e, t?, p?)))
            })
            .collect()
    }

    /// The extent of these cells' requests along each timing, padded a tenth either side.
    fn span(&self, keys: &[CellKey]) -> Option<Span> {
        let rs: Vec<(Engine, f64, f64)> = keys.iter().flat_map(|&k| self.requests(k)).collect();
        let range = |vs: &mut dyn Iterator<Item = f64>| {
            vs.fold(None, |r: Option<[f64; 2]>, v| {
                Some(r.map_or([v, v], |[lo, hi]| [lo.min(v), hi.max(v)]))
            })
        };
        let [t0, t1] = range(&mut rs.iter().map(|r| r.1))?;
        let [p0, p1] = range(&mut rs.iter().map(|r| r.2))?;
        let padded = |lo: f64, hi: f64| {
            let pad = (hi - lo).max(hi * 0.05) * 0.1;
            [(lo - pad).max(0.0), hi + pad]
        };
        Some((padded(t0, t1), padded(p0, p1)))
    }

    /// One cell's requests as a square bubble chart, TTFT up and TPOT across, both linear: on the
    /// cell's own ranges ([`Axes2::Own`]) or on `span`, its small multiple's ([`Axes2::Same`]).
    /// Requests an engine landed within one dot of each other are one bubble, its area their
    /// count, so a batch's identical TTFTs show as the crowd they are, not as one dot.
    fn chart(&self, k: CellKey, span: Span, axes: Axes2, side: Px, label: &str) -> Element {
        let Some(own) = self.span(&[k]) else {
            return rsx! {};
        };
        let (ttft, tpot) = match axes {
            Axes2::Own => own,
            Axes2::Same => span,
        };
        let unit_radius = 3.5;
        let data = bubbles(&self.requests(k), (ttft, tpot), side, unit_radius);
        let axis = |maps_to, title, scale_type, domain| Axis {
            maps_to,
            title,
            scale_type,
            ticks: None,
            include_zero: Some(false),
            visible: None,
            domain: Some(AxisDomain::Values(domain)),
        };
        let spec = Chart::Bubble {
            data,
            unit_radius,
            options: ChartOptions {
                title: String::new(),
                axes: Axes {
                    bottom: axis("key", "TPOT (ms)", None, tpot),
                    left: axis("value", "TTFT (ms)", None, ttft),
                },
                color: Some(Colour::Scale {
                    scale: (self.engines.iter().zip(["s1", "s2"]))
                        .map(|(&e, class)| (e.label(), colour(class)))
                        .collect(),
                }),
                heatmap: None,
                height: side.to_string(),
                width: Some(side.to_string()),
                legend: Some(Enabled { enabled: false }),
                locale: None,
                combo: None,
                bounds: None,
                bubble: Some(Bubbles {
                    radius_maps_to: "radius",
                    radius_label: "Requests",
                    fill_opacity: 0.6,
                }),
                line: None,
                hover: None,
                toolbar: Enabled { enabled: false },
            },
        };
        // Its title inside its plot, top left, letting the pointer through to the dots under it.
        let (u, o, i) = k;
        rsx! {
            div { style: "position: relative; width: {side}",
                {chart(&spec, label, &format!("width: {side}"), None)}
                span { style: "position: absolute; top: 0; left: 0; padding: 4px 0 0 3.5rem; font-size: 0.75rem; pointer-events: none",
                    "{i} in, {o} out, {u} at once"
                }
            }
        }
    }
}

impl Latency<'_> {
    /// One cell's `t` as each engine's empirical CDF: the share of its requests (up) done within
    /// each time (across), a step at every request, each curve run on to the range's end. Between
    /// scratchy's curve and its rival's, the sparklines' fills: cyan where more of scratchy's
    /// requests are done by then, red where fewer; across the middle, a line from one median to
    /// the other, cyan or red as scratchy's is the sooner or later, its factor written over it.
    /// Time runs on a log axis [`ECDF_SPAN`] wide, centred on the engines' medians: a ratio, the
    /// factor's included, is one width on every chart. Bare: no axis titles or tick labels,
    /// `tag`'s lines in its corner; as wide as its column and `height` high.
    fn ecdf(&self, k: CellKey, t: RequestTiming, tag: &[&str], label: &str, height: &str) -> Element {
        let curves: Vec<(Engine, Vec<f64>)> = (self.engines.iter())
            .filter_map(|&e| {
                let rs = self.cell(e, k)?.requests.as_ref()?;
                let mut vs: Vec<f64> = (rs.answered(t, k.1).filter_map(|(_, v)| v))
                    .filter(|&v| v > 0.0)
                    .collect();
                vs.sort_by(f64::total_cmp);
                (!vs.is_empty()).then_some((e, vs))
            })
            .collect();
        // Centred on the engines' medians (their geometric mean), so the factor between them sits
        // in the middle; a request past either edge is drawn at it.
        let medians: Vec<f64> = curves.iter().filter_map(|(_, vs)| quantile(vs, 50.0)).collect();
        if medians.is_empty() {
            return rsx! {};
        }
        let centre = (medians.iter().map(|m| m.ln()).sum::<f64>() / medians.len() as f64).exp();
        let half = ECDF_SPAN.sqrt();
        let (start, end) = (centre / half, centre * half);
        // Plotted as log10(ms) on a linear axis: Carbon rounds a log axis out to whole decades,
        // which would undo the common span.
        let point = |group, key: f64, value: f64| SparkPoint {
            group,
            key: key.clamp(start, end).log10(),
            value: Some(value),
            upper: None,
            lower: None,
            factor: None,
        };
        let mut data: Vec<SparkPoint> = (curves.iter())
            .flat_map(|(e, vs)| {
                let n = vs.len() as f64;
                std::iter::once(point(e.label(), vs[0], 0.0))
                    .chain(
                        (vs.iter().enumerate())
                            .map(move |(i, &v)| point(e.label(), v, (i + 1) as f64 / n)),
                    )
                    .chain(std::iter::once(point(e.label(), end, 1.0)))
            })
            .collect();
        let mut lines: Vec<&'static str> = curves.iter().map(|(e, _)| e.label()).collect();
        let mut dashed: Vec<&'static str> = Vec::new();
        if let [(_, mine), (_, theirs)] = curves.as_slice() {
            // Each curve's share done by `x`, at every step either takes: the fills' bounds.
            let share =
                |vs: &[f64], x: f64| vs.partition_point(|&v| v <= x) as f64 / vs.len() as f64;
            let mut xs: Vec<f64> = mine.iter().chain(theirs).copied().chain([end]).collect();
            xs.sort_by(f64::total_cmp);
            xs.dedup();
            for x in xs {
                let (a, b) = (share(mine, x), share(theirs, x));
                // Each fill spans the curves where its side is ahead, and nothing elsewhere.
                for (group, shown) in [(ECDF_AHEAD, a > b), (ECDF_BEHIND, a < b)] {
                    let (low, high) = if shown { (a.min(b), a.max(b)) } else { (a, a) };
                    data.push(SparkPoint {
                        upper: Some(high),
                        lower: Some(low),
                        ..point(group, x, high)
                    });
                }
            }
            // Both are times, lower better: how many times sooner scratchy's median is.
            if let (Some(m), Some(r)) = (quantile(mine, 50.0), quantile(theirs, 50.0)) {
                let factor = r / m;
                let group = if factor >= 1.0 {
                    ECDF_SOONER
                } else {
                    ECDF_LATER
                };
                // The factor sits a little over the line, at its middle on the log axis.
                let over = 0.5 + 0.12;
                data.extend([
                    point(group, m, 0.5),
                    SparkPoint {
                        upper: Some(over),
                        lower: Some(over),
                        factor: Some(factor),
                        ..point(group, (m * r).sqrt(), 0.5)
                    },
                    point(group, r, 0.5),
                ]);
                lines.push(group);
                dashed.push(group);
            }
        }
        let axis = |maps_to, title, scale_type, domain| Axis {
            maps_to,
            // Kept on the hidden axes: Carbon's tooltip names each value by it.
            title,
            scale_type,
            ticks: None,
            include_zero: Some(false),
            visible: Some(false),
            domain: Some(AxisDomain::Values(domain)),
        };
        let fills = curves.len() == 2;
        let spec = Chart::Combo {
            data,
            dashed,
            options: ChartOptions {
                title: String::new(),
                axes: Axes {
                    bottom: axis("key", "log10 ms", None, [start.log10(), end.log10()]),
                    left: axis("value", "share of requests", None, [0.0, 1.0]),
                },
                color: Some(Colour::Scale {
                    scale: (self.engines.iter().zip(["s1", "s2"]))
                        .map(|(&e, class)| (e.label(), colour(class)))
                        .chain([
                            // The heat maps' cyan and red: 50 to fill, 70 for the median line.
                            (ECDF_AHEAD, Ramp::Cyan.at::<50>()),
                            (ECDF_BEHIND, Ramp::Red.at::<50>()),
                            (ECDF_SOONER, Ramp::Cyan.at::<70>()),
                            (ECDF_LATER, Ramp::Red.at::<70>()),
                        ])
                        .collect(),
                }),
                heatmap: None,
                height: height.to_string(),
                width: Some("100%".to_string()),
                legend: Some(Enabled { enabled: false }),
                locale: None,
                // Carbon draws an area's bounds only when the layer holds one series.
                combo: Some(
                    (fills
                        .then_some([ECDF_AHEAD, ECDF_BEHIND])
                        .into_iter()
                        .flatten())
                    .map(|fill| ComboLayer {
                        kind: "area",
                        corresponding_datasets: vec![fill],
                    })
                    .chain([ComboLayer {
                        kind: "line",
                        corresponding_datasets: lines,
                    }])
                    .collect(),
                ),
                bounds: Some(Bounds {
                    upper_bound_maps_to: "upper",
                    lower_bound_maps_to: "lower",
                }),
                bubble: None,
                line: Some(LineOptions {
                    curve: "curveStepAfter",
                    points: Enabled { enabled: false },
                }),
                // A step per request, each curve its own: no one step lists every series.
                hover: Some(Hover::NONE),
                toolbar: Enabled { enabled: false },
            },
        };
        rsx! {
            div { style: "position: relative; width: 100%",
                {chart(&spec, label, "width: 100%", None)}
                div { style: "position: absolute; top: 0; left: 0; padding: 2px 4px; font-size: 0.625rem; font-weight: 600; pointer-events: none",
                    for line in tag { div { "{line}" } }
                }
            }
        }
    }

    /// Every cell at answer length `o` as a small multiple, users at once down and prompt size
    /// across, each column `column` wide (a grid track), under `title`: `cell` draws each, and
    /// names its own users count where `users` says the rows don't.
    fn multiple(
        &self,
        o: u32,
        column: &str,
        title: &str,
        users: UsersNamed,
        cell: impl Fn(CellKey) -> Element,
    ) -> Element {
        let ks: Vec<CellKey> = self.keys.iter().copied().filter(|k| k.1 == o).collect();
        let mut rows: Vec<u32> = ks.iter().map(|k| k.0).collect();
        let mut prompts: Vec<u32> = ks.iter().map(|k| k.2).collect();
        rows.dedup();
        prompts.sort_unstable();
        prompts.dedup();
        let beside = users == UsersNamed::Beside;
        let columns = format!(
            "display: grid; grid-template-columns: {}repeat({}, {column}); gap: 0.5rem; align-items: center",
            if beside { "max-content " } else { "" },
            prompts.len()
        );
        // Each column named with its unit, "128 input tokens", and each row, "1 user", beside it.
        rsx! {
            Stack { gap: 3,
                strong { "{title}" }
                div { style: "{columns}",
                    if beside { div {} }
                    for i in &prompts { div { style: "text-align: center", "{i} input tokens" } }
                    for &u in &rows {
                        if beside { div { "{users_at(u)}" } }
                        for &i in &prompts {
                            div { style: "min-width: 0",
                                if ks.contains(&(u, o, i)) {
                                    {cell((u, o, i))}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// A users count with its unit: "1 user", "4 users".
fn users_at(u: u32) -> String {
    if u == 1 {
        "1 user".to_string()
    } else {
        format!("{u} users")
    }
}

/// Where a small multiple ([`Latency::multiple`]) names each cell's users count: beside its row,
/// or in the cell itself.
#[derive(Clone, Copy, PartialEq)]
enum UsersNamed {
    Beside,
    InCells,
}

/// A glance sparkline's height; each is as wide as its column.
const GLANCE_CHART_HEIGHT: &str = "4.5rem";

/// The latency tab's ECDFs' height: about 1.8 wide for 1 high in its four prompt columns, so a
/// quantile reads off the share axis (each of a cell's 50 requests some 3px), not a sparkline's.
const ECDF_HEIGHT: &str = "9rem";

/// The ratio every latency ECDF spans edge to edge, so a ratio is one width on all of them
/// ([`Latency::ecdf`]): ×1.8 a little over half. Over the runs published by 2026-10-30, it holds
/// the middle 90% (p5 to p95) of 90% of the charts' requests and every gap between medians (the
/// widest ×2.7); the tails of the rest are drawn at the edges.
const ECDF_SPAN: f64 = 3.0;

/// An ECDF's fills ([`Latency::ecdf`]): where more of scratchy's requests are done than its
/// rival's, and where fewer.
const ECDF_AHEAD: &str = "scratchy ahead";
const ECDF_BEHIND: &str = "scratchy behind";
/// An ECDF's median line: scratchy's median the sooner, or the later.
const ECDF_SOONER: &str = "median, scratchy sooner";
const ECDF_LATER: &str = "median, scratchy later";

/// A small multiple's panel: square, so neither timing's spread looks wider by the chart's shape.
const PANEL_SIDE: Px = Px(214);

/// A length in CSS pixels; written as one ("80px").
#[derive(Clone, Copy)]
struct Px(u32);

impl std::fmt::Display for Px {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}px", self.0)
    }
}

/// An engine's requests as bubbles on a `side`-square plot of `span`: the requests within one
/// `unit_radius` dot's width of each other, along both timings, are one bubble at their mean,
/// its radius the square root of their count.
fn bubbles(
    requests: &[(Engine, f64, f64)],
    ([t0, t1], [p0, p1]): Span,
    side: Px,
    unit_radius: f64,
) -> Vec<BubblePoint<'static>> {
    let dots = f64::from(side.0) / (2.0 * unit_radius);
    let at = |v: f64, lo: f64, hi: f64| ((v - lo) / (hi - lo) * dots).floor() as i64;
    let mut crowds: Vec<Crowd> = Vec::new();
    for &(engine, t, p) in requests {
        let dot = (at(t, t0, t1), at(p, p0, p1));
        match crowds
            .iter_mut()
            .find(|c| c.engine == engine && c.dot == dot)
        {
            Some(c) => {
                c.requests += 1;
                c.ttft += t;
                c.tpot += p;
            }
            None => crowds.push(Crowd {
                engine,
                dot,
                requests: 1,
                ttft: t,
                tpot: p,
            }),
        }
    }
    (crowds.into_iter())
        .map(|c| {
            let n = f64::from(c.requests);
            BubblePoint {
                group: c.engine.label(),
                key: c.tpot / n,
                value: c.ttft / n,
                radius: n.sqrt(),
            }
        })
        .collect()
}

/// The requests one engine landed on one dot of a plot: how many, and their timings' sums.
struct Crowd {
    engine: Engine,
    /// The dot, in dot widths up (TTFT) and across (TPOT).
    dot: (i64, i64),
    requests: u32,
    ttft: f64,
    tpot: f64,
}

/// The legend a model's latency charts share, in the engines' colours.
fn latency_legend(engines: &[Engine]) -> Element {
    rsx! {
        div { style: "display: flex; gap: 1.5rem; align-items: center",
            for (&e, class) in engines.iter().zip(["s1", "s2"]) { {engine_key(e, class)} }
        }
    }
}

/// An engine's name after a dot of its series colour `class`: one entry of a legend, or a key in
/// running text.
fn engine_key(e: Engine, class: &str) -> Element {
    rsx! {
        span { style: "display: inline-flex; gap: 0.4rem; align-items: center",
            span { style: "width: 10px; height: 10px; border-radius: 50%; background: {colour(class)}" }
            "{e.label()}"
        }
    }
}

/// Which ranges a small multiple's panels plot on.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Axes2 {
    /// Each panel its own, TTFT and TPOT: the shape of each cell's spread, which a shared range
    /// flattens (TTFT spans a hundredfold across a grid, TPOT's two clouds sit 1 ms apart).
    Own,
    /// One range for every panel: where the cells sit against each other.
    Same,
}

impl Axes2 {
    const ALL: [Axes2; 2] = [Axes2::Own, Axes2::Same];

    fn title(self) -> &'static str {
        match self {
            Axes2::Own => "Each its own axes",
            Axes2::Same => "Same axes",
        }
    }
}

/// Every request of every grid cell, time to first token against time per output token, a dot
/// each, scratchy against its rival: per answer length a small multiple, users at once down,
/// prompt size across, each panel on its own axes or every one on the same (a switch).
fn request_plots(m: &Model, mid: &str, run: &Run) -> Element {
    let Some(l) = Latency::of(m, run) else {
        return rsx! {};
    };
    // The same axes span the bubble charts only: the 1-user cells ([`Load`]).
    let alone: Vec<CellKey> = (l.keys.iter().copied())
        .filter(|&(u, _, _)| Load::of(u) == Load::Alone)
        .collect();
    let Some(span) = l.span(&alone).or_else(|| l.span(&l.keys)) else {
        return rsx! {};
    };
    let mut answers: Vec<u32> = l.keys.iter().map(|k| k.1).collect();
    answers.dedup();
    answers.sort_unstable();
    answers.dedup();
    // A 1-user cell's bubble chart, on its own axes or on every cell's across every answer length;
    // a cell under load its end-to-end time's distribution ([`Load`]).
    let multiple = |o: u32, axes: Axes2| {
        l.multiple(
            o,
            &PANEL_SIDE.to_string(),
            &format!("{o}-token answers"),
            UsersNamed::Beside,
            |(u, o, i)| match Load::of(u) {
                Load::Alone => {
                    let about =
                        format!("TTFT against TPOT, every request, {i} in {o} out {u} at once, {mid}");
                    l.chart((u, o, i), span, axes, PANEL_SIDE, &about)
                }
                Load::Shared => {
                    let about = format!(
                        "{} of every request, {i} in {o} out {u} at once, {mid}",
                        RequestTiming::E2el.title()
                    );
                    // As wide as the bubble charts, about 1.8 to 1.
                    let height = Px(PANEL_SIDE.0 * 5 / 9).to_string();
                    l.ecdf((u, o, i), RequestTiming::E2el, &["E2EL"], &about, &height)
                }
            },
        )
    };
    let table = |t: RequestTiming| -> Element {
        rsx! {
            Table { headers: headers([t.title(), "engine", "requests", "min", "p25", "median", "p75", "max"]),
                for &(u, o, i) in &l.keys {
                    for &e in &l.engines {
                        if let Some(rs) = l.cell(e, (u, o, i)).and_then(|c| c.requests.as_ref()) {
                            {
                                let mut vs: Vec<f64> = rs.answered(t, o).filter_map(|(_, v)| v).collect();
                                vs.sort_by(f64::total_cmp);
                                let q = |p: f64| fmt(quantile(&vs, p), 1);
                                rsx! {
                                    TableRow {
                                        TableCell { "{i}→{o} ×{u}" }
                                        TableCell { "{e.label()}" }
                                        TableCell { "{vs.len()}" }
                                        TableCell { {q(0.0)} }
                                        TableCell { {q(25.0)} }
                                        TableCell { {q(50.0)} }
                                        TableCell { {q(75.0)} }
                                        TableCell { {q(100.0)} }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };
    part(
        "Latency distributions",
        rsx! {
            "With 1 user, every request a dot: time to first token up, time per output token across, \
             both linear, each panel on its own ranges (its ticks say them) or every panel on the same: \
             the switch below. Requests that failed, or whose answer arrived in one piece, have neither \
             timing to plot. With more users, the share of requests (up) done within each end-to-end \
             time (across): once the GPU is busy, scheduling only moves each request's wait between \
             its first token and the rest, so their sum is what compares. Its time axis is a log one \
             ×3 wide on every chart, centred on the two medians, so a factor is one width everywhere."
        },
        rsx! {
            Stack { gap: 6,
                {latency_legend(&l.engines)}
                // Own axes first and shown; the same axes a switch away (Carbon draws a chart
                // in a hidden panel when its panel is shown).
                ContentSwitcher { selected: format!("{mid}-axes-{}", slug(Axes2::Own.title())),
                    for axes in Axes2::ALL {
                        ContentSwitcherItem { target: format!("{mid}-axes-{}", slug(axes.title())), selected: axes == Axes2::Own, "{axes.title()}" }
                    }
                }
                div {
                    for axes in Axes2::ALL {
                        ContentPanel { id: format!("{mid}-axes-{}", slug(axes.title())), shown: axes == Axes2::Own,
                            Stack { gap: 6,
                                for &o in &answers { {multiple(o, axes)} }
                            }
                        }
                    }
                }
                Fold { title: "Table view, every request",
                    Stack { gap: 4,
                        for t in RequestTiming::ALL { {table(t)} }
                    }
                }
            }
        },
    )
}

/// numpy.percentile(method='linear') of sorted values, as the bench figures its own.
fn quantile(sorted: &[f64], p: f64) -> Option<f64> {
    let last = sorted.len().checked_sub(1)?;
    let k = last as f64 * p / 100.0;
    let (f, c) = (k.floor() as usize, (k.ceil() as usize).min(last));
    Some(sorted[f] + (sorted[c] - sorted[f]) * (k - k.floor()))
}

fn commit(run: &Run) -> Element {
    rsx! { a { href: "{REPO}/commit/{run.repo.sha}", code { "{run.repo.sha8()}" } } }
}

/// The numbers a model's timeline follows across its runs on a machine: its one-user throughput
/// and time to first token at the base cell, each run's own.
const TIMELINE_METRICS: [(Metric, &str); 2] = [
    (Metric::Throughput, "tok/s, 1 user"),
    (Metric::Ttft, "TTFT ms, 1 user"),
];

/// A timeline's steps at one metric: a step a run, scratchy's value and the rival's where that
/// run measured each itself.
fn timeline_steps(kept: &[&Entry], rival: Option<Engine>, metric: Metric) -> Vec<Step> {
    (kept.iter().enumerate())
        // A power of two a run: the sparkline's x axis is a log one, so runs at 1, 2, 3, ...
        // would crowd the newest to the right edge. The spyre timeline spaces its sweeps so too.
        .map(|(i, e)| Step {
            x: 2f64.powi(i as i32),
            scratchy: timing(own_cell(e, Engine::Scratchy), metric),
            rival: rival.and_then(|r| timing(own_cell(e, r), metric)),
            highlight: false,
        })
        .collect()
}

/// A timeline's two labelled ×-factors. Where scratchy started: the first run that measured
/// both, since a quick run's borrowed rival is no point of its own and the carry runs only
/// forward. Where it stands: the newest run scratchy measured, against the rival carried to it.
fn timeline_highlights(steps: &mut [Step]) {
    let measured = |s: &Step| s.scratchy.is_some();
    let both = |s: &Step| s.scratchy.is_some() && s.rival.is_some();
    if let Some(first) = steps.iter().position(both) {
        steps[first].highlight = true;
    }
    if let Some(last) = steps.iter().rposition(measured) {
        steps[last].highlight = true;
    }
}

/// An engine's one-user cell in a run, where that run measured it itself (not borrowed).
fn own_cell<'a>(e: &Entry<'a>, engine: Engine) -> Option<&'a Cell> {
    (engine == Engine::Scratchy || !e.m.borrowed.contains_key(engine.key()))
        .then(|| by_users(e.m, e.run, engine).get(&1).copied())
        .flatten()
}

/// A model's timelines over its runs on a machine, oldest first, one a metric
/// ([`TIMELINE_METRICS`]): scratchy's line and the first rival any run measured, each a run's own
/// measurement. A rival a run borrowed ([`Model::borrowed`]) is no measurement of that run's, so
/// its line continues dotted there, carried from the last run that measured it. Runs whose
/// one-user cell is at another shape (another answer length) measured something else, so a
/// timeline keeps the newest run's shape. The fill carries its ×-factor at the first and the last
/// run, as the spyre page's do: where scratchy started, and where it stands. Empty with fewer than
/// two runs: one point is not a line.
fn model_timelines(entries: &[Entry]) -> Vec<(&'static str, Element)> {
    let Some(newest) = entries.last() else {
        return Vec::new();
    };
    let shape = users_shape(newest.m, newest.run);
    let kept: Vec<&Entry> = (entries.iter())
        .filter(|e| users_shape(e.m, e.run) == shape)
        .collect();
    if kept.len() < 2 {
        return Vec::new();
    }
    let rival = Engine::RIVALS
        .into_iter()
        .find(|&r| kept.iter().any(|e| own_cell(e, r).is_some()));
    let names: Vec<String> = (kept.iter())
        .map(|e| e.run.generated_utc.month_day().to_string())
        .collect();
    let rival_side = rival.unwrap_or(Engine::Scratchy).side();
    let against = rival.map_or("alone".to_string(), |r| format!("against {}", r.label()));
    TIMELINE_METRICS
        .iter()
        .map(|&(metric, value)| {
            let mut steps = timeline_steps(&kept, rival, metric);
            timeline_highlights(&mut steps);
            let chart = sparkline(
                &steps,
                SparkAxes {
                    better: metric.better(),
                    along: "run",
                    value,
                    carry: true,
                },
                Engine::Scratchy.side(),
                rival_side,
                Some(names.clone()),
                "5rem",
                &format!("{value} over its runs, scratchy {against}"),
            );
            (value, chart)
        })
        .collect()
}

/// The Over time section's charts, titled: none until some model of the current lineup has two
/// runs of one shape on one machine. Only the current lineup, as in the overview ([`lineup`]): a
/// model an older round ran keeps its runs in its machine's own section.
fn over_time_charts(machines: &BTreeMap<&str, MachineRuns>, chip: &str) -> Vec<(String, Element)> {
    let classes = lineup(machines).map_or(&[][..], |l| l.classes);
    let current = |stem: &str| classes.iter().any(|c| c.small == stem || c.large == stem);
    let mut charts = Vec::new();
    let Some(mr) = machines.get(chip) else {
        return charts;
    };
    for (stem, entries) in mr.models.iter().filter(|(s, _)| current(s)) {
        for (value, chart) in model_timelines(entries) {
            charts.push((format!("{stem} · {value}"), chart));
        }
    }
    charts
}

/// The Over time section, one block a Mac: each current model's one-user throughput and time to
/// first token against its rival, over every run the Mac has published (history/ included) of the
/// model's newest shape. Nothing where no model has two such runs.
fn over_time(machines: &BTreeMap<&str, MachineRuns>) -> Option<Element> {
    let blocks: Vec<(&str, Vec<(String, Element)>)> = (in_order(machines).into_iter())
        .map(|chip| (chip, over_time_charts(machines, chip)))
        .filter(|(_, charts)| !charts.is_empty())
        .collect();
    if blocks.is_empty() {
        return None;
    }
    let grid = "display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); \
                column-gap: 1.5rem; row-gap: 1.25rem; align-items: start";
    Some(rsx! {
        Section { id: "over-time", level: 2,
            Stack { gap: 4,
                Heading { "Over time" }
                p {
                    "Each model's one-user throughput and time to first token at its base cell, a \
                     point a run on its Mac, oldest first, over every run the Mac has published \
                     (the older ones are kept in "
                    code { "site/data/metal/history/" }
                    "), scratchy against the first engine any run there measured. A run of \
                     scratchy alone (a quick run) takes the other engine's line dotted, carried from \
                     the last run that measured it, and the fill between the lines follows it. The \
                     fill is labelled with its ×-factor at the first run and the newest: where \
                     scratchy started, and where it stands now."
                }
                for (chip, charts) in blocks {
                    Stack { gap: 3,
                        strong { "{chip}" }
                        div { style: "{grid}",
                            for (title, chart) in charts {
                                div {
                                    p { style: "font-size: 0.875rem; margin: 0 0 0.25rem", "{title}" }
                                    {chart}
                                }
                            }
                        }
                    }
                }
            }
        }
    })
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
                                    // Every cell's number through cfmt, so a dead or failed
                                    // one carries its mark like the TPOT beside it.
                                    cfmt(one, Metric::Throughput),
                                    cfmt(many, Metric::Throughput),
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

/// How many of the overview's heat map squares have their factor written in
/// them: at most one per colour step, spread over the steps the squares use.
/// A heat map left without one then gets one of its own.
const HEAT_LABELS: usize = 12;

/// Where an overview heat map square is: machine, model, metric, and its
/// users count and prompt length.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Square<'a> {
    chip: &'a str,
    stem: &'a str,
    metric: &'a str,
    users: u32,
    prompt: u32,
}

/// One overview heat map's ratios, at its run's [`glance_answer`]: the rival, and each square's
/// (users, prompt, how many times faster scratchy is).
struct GlanceGrid {
    rival: Engine,
    cells: Vec<(u32, u32, Option<f64>)>,
}

impl GlanceGrid {
    /// A model's grid at a metric on a machine's latest run, on the overview's users counts by
    /// its prompts ([`Lineup`]), or why none.
    fn of(
        mr: &MachineRuns,
        stem: &str,
        metric: Metric,
        users: &[u32],
        prompts: &[u32],
    ) -> Result<Self, &'static str> {
        let Entry { run, m } = mr.latest(stem).ok_or("not run")?;
        let g = ShapeGrid::of(m, run, Slice::Answer(glance_answer(run))).ok_or("no grid")?;
        let rival = g
            .rival
            .ok_or("scratchy alone so far: no engine to compare")?;
        let cells = (users.iter())
            .flat_map(|&u| prompts.iter().map(move |&i| (u, i)))
            .map(|(u, i)| (u, i, g.versus(metric, (u, i), rival).0))
            .collect();
        Ok(GlanceGrid {
            rival,
            cells,
        })
    }
}

/// The overview's two sets of Macs, by memory ([`Lineup`]): each runs its own models and its own
/// users × prompt grid, so each gets heat maps of its own.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Set {
    Small,
    Large,
}

impl Set {
    const ALL: [Set; 2] = [Set::Small, Set::Large];

    fn of(memory_gb: u32, large_from_gb: u32) -> Set {
        if memory_gb >= large_from_gb {
            Set::Large
        } else {
            Set::Small
        }
    }
}

/// Which of a heat map's users counts it scores. One user leaves the GPU room, so prefill and
/// decode speed are each a fact of their own; with more, scheduling only moves each request's
/// wait between its TTFT and its TPOT, so only their sum, the end-to-end time, is a score.
#[derive(Clone, Copy, PartialEq)]
enum Load {
    Alone,
    Shared,
}

impl Load {
    fn of(users: u32) -> Load {
        if users == 1 { Load::Alone } else { Load::Shared }
    }

    fn users(self, all: &[u32]) -> Vec<u32> {
        (all.iter().copied())
            .filter(|&u| Load::of(u) == self)
            .collect()
    }
}

/// A set's heat map shape: users at once down the side, prompt tokens across.
#[derive(Default)]
struct Shape {
    users: Vec<u32>,
    prompts: Vec<u32>,
}

impl Shape {
    /// A run's users × prompt grid (`grid_conc`, `grid_input`), None for a run without one (a
    /// `--no-scaling` run, or one from before the grid had its own users counts).
    fn of(run: &Run) -> Option<Shape> {
        let sc = run.config.scaling.as_ref()?;
        Some(Shape {
            users: sc.grid_conc.clone()?,
            prompts: sc.grid_input.clone()?,
        })
    }
}

/// The overview's shape: a column per class of the lineup (`config.classes`), each Mac on the
/// class's small model (every Mac, where the lineup's runs took the small set everywhere:
/// `small_everywhere`) or, in a lineup from before that, its large one from `large_from_gb`, and
/// each [`Set`] its own map shape. A square its run did not measure (a run of an older shape, or a prompt the Mac's memory
/// did not hold) is blank.
struct Lineup<'a> {
    classes: &'a [data::Class],
    large_from_gb: u32,
    /// Every Mac runs the small set ([`data::Config::default_set`]), so a column is one model on
    /// every chip; false for a lineup from before it, which ran the large set from 64 GB.
    small_everywhere: bool,
    shapes: BTreeMap<Set, Shape>,
}

impl Lineup<'_> {
    fn model(&self, class: &str, memory_gb: u32) -> Option<&str> {
        (self.classes.iter())
            .find(|c| c.class == class)
            .map(|c| c.model(memory_gb, self.large_from_gb, self.small_everywhere))
    }

    fn set(&self, memory_gb: u32) -> Set {
        Set::of(memory_gb, self.large_from_gb)
    }
}

/// The overview's [`Lineup`]: the classes and cut-off of the newest run that records a lineup, and
/// per set the grid of the set's newest run that records a lineup too, else of its newest run with
/// a grid at all, so a `--no-scaling` run does not empty the set's maps. None before any run
/// records a lineup.
fn lineup<'a>(machines: &BTreeMap<&str, MachineRuns<'a>>) -> Option<Lineup<'a>> {
    let mut runs: Vec<&Run> = (machines.values())
        .flat_map(|mr| mr.runs.iter().copied())
        .collect();
    runs.sort_by(|a, b| b.generated_utc.cmp(&a.generated_utc));
    let newest = runs.iter().find(|r| r.config.classes.is_some())?;
    // Runs from before `large_from_gb` split the sets at 64 GB.
    let large_from_gb = newest.config.large_from_gb.unwrap_or(64);
    let shapes = (Set::ALL.into_iter())
        .map(|set| {
            let mine = || {
                (runs.iter()).filter(move |r| Set::of(r.machine.memory_gb, large_from_gb) == set)
            };
            let shape = (mine().filter(|r| r.config.classes.is_some()))
                .find_map(|r| Shape::of(r))
                .or_else(|| mine().find_map(|r| Shape::of(r)))
                .unwrap_or_default();
            (set, shape)
        })
        .collect();
    Some(Lineup {
        classes: newest.config.classes.as_deref()?,
        large_from_gb,
        // From the newest run that took a default set: a run that named its models (`--models large`,
        // `full`, or by name) says nothing about it, so it must not flip every Mac's column.
        small_everywhere: (runs.iter())
            .find(|r| r.config.classes.is_some() && !r.config.models_given)
            .and_then(|r| r.config.default_set.as_deref())
            == Some("small"),
        shapes,
    })
}

/// Every model's users-by-prompt grid on every machine as small multiples,
/// one table per metric: a machine per row, a model per column, each cell that
/// model's grid as a tiny heatmap coloured like the full ones.
fn glance(machines: &BTreeMap<&str, MachineRuns>) -> Element {
    // A column per architecture class ([`Lineup`]); each machine's row shows the class's model:
    // the small one on every Mac where the lineup's newest default run took the small set
    // everywhere, else the small set below the lineup's `large_from_gb` and the large one from it,
    // run or not yet. A model an older round ran stays in its machine's own section, not here.
    let Some(lineup) = lineup(machines) else {
        return rsx! {};
    };
    let memory = |chip: &str| {
        machines[chip].runs[machines[chip].runs.len() - 1]
            .machine
            .memory_gb
    };
    let model_of = |chip: &str, class: &str| lineup.model(class, memory(chip));
    let classes: Vec<&str> = lineup.classes.iter().map(|c| c.class.as_str()).collect();
    // Each machine's own row: (machine, model) for its set only, so another set's model it ran in
    // an earlier round sets neither the intro's answer lengths nor the highlights.
    let shown: Vec<(&str, &str)> = (machines.keys())
        .flat_map(|&chip| {
            (classes.iter()).filter_map(move |&class| Some((chip, model_of(chip, class)?)))
        })
        .collect();
    if shown.is_empty() {
        return rsx! {};
    }
    // The scaling and startup panels are a column per model: every model any row shows.
    let mut stems: Vec<&str> = Vec::new();
    for &(_, stem) in &shown {
        if !stems.contains(&stem) {
            stems.push(stem);
        }
    }
    let shape = |chip: &str| &lineup.shapes[&lineup.set(memory(chip))];
    // The answer lengths the maps are at: one, unless machines are on runs of different shapes.
    let answers: BTreeSet<u32> = (shown.iter())
        .filter_map(|&(chip, stem)| machines[chip].latest(stem))
        .map(|Entry { run, .. }| glance_answer(run))
        .collect();
    let answers = (answers.iter())
        .map(|a| a.to_string())
        .collect::<Vec<_>>()
        .join("- and ");
    let rivals = Engine::RIVALS.map(Engine::label).join(", ");
    // Every grid's ratios first, keyed by machine, model and metric, or why
    // the grid has none: the highlights are picked from all of them.
    let mut grids: BTreeMap<(&str, &str, &str), Result<GlanceGrid, &str>> = BTreeMap::new();
    for &(chip, stem) in &shown {
        for gm in GRID_METRICS {
            grids.insert(
                (chip, stem, gm.name),
                GlanceGrid::of(
                    &machines[chip],
                    stem,
                    gm.metric,
                    &gm.load.users(&shape(chip).users),
                    &shape(chip).prompts,
                ),
            );
        }
    }
    let squares: BTreeMap<Square, f64> = grids
        .iter()
        .filter_map(|(&(chip, stem, name), g)| Some((chip, stem, name, g.as_ref().ok()?)))
        .flat_map(|(chip, stem, name, g)| {
            g.cells.iter().filter_map(move |&(u, i, r)| {
                let at = Square {
                    chip,
                    stem,
                    metric: name,
                    users: u,
                    prompt: i,
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
        trend_spark(at, &sparks[&at], GLANCE_CHART_HEIGHT)
    };
    let chips: Vec<&str> = in_order(machines);
    // Every machine a row, the small set's first.
    let rows: Vec<&str> = (Set::ALL.into_iter())
        .flat_map(|set| {
            let lineup = &lineup;
            (chips.iter().copied()).filter(move |&chip| lineup.set(memory(chip)) == set)
        })
        .collect();
    // The one map: every machine, model, measure and users count in a single grid, a square a
    // cell, on the one ratio scale, under one header. Under each model (its class) a column per
    // prompt size, shortest first; where the sets of Macs ran different sizes there, its header
    // names each set's, small set first. Under each machine its rows: prefill and decode with 1
    // user, then end-to-end time with each users count ([`GRID_METRICS`]).
    let one_map = {
        enum Line<'a> {
            Classes,
            Models,
            Prompts,
            Measure(&'a str, GridMetric, u32),
        }
        let most = (lineup.shapes.values()).map(|s| s.prompts.len()).max().unwrap_or(0);
        let col = |k: usize, p: usize| 3 + k * most + p;
        let mut lines: Vec<Line> = vec![Line::Classes, Line::Models, Line::Prompts];
        // Each machine's label spans its rows: (machine, first line, last line).
        let mut spans: Vec<(&str, usize, usize)> = Vec::new();
        for &chip in &rows {
            let start = lines.len();
            let users = &shape(chip).users;
            for &gm in &GRID_METRICS {
                lines.extend(gm.load.users(users).into_iter().map(|u| Line::Measure(chip, gm, u)));
            }
            spans.push((chip, start, lines.len() - 1));
        }
        // A model per class, or each model the Macs run there where sets differ.
        let models = |class: &str| {
            let mut names: Vec<&str> = Vec::new();
            for &chip in &rows {
                if let Some(stem) = model_of(chip, class).filter(|s| !names.contains(s)) {
                    names.push(stem);
                }
            }
            names.join(" / ")
        };
        let template_columns =
            format!("max-content max-content repeat({}, 2.25rem)", classes.len() * most);
        let k_prompt = |i: u32| {
            if i >= 1024 && i.is_multiple_of(1024) {
                format!("{}k", i / 1024)
            } else {
                i.to_string()
            }
        };
        // Each column's prompt size, or each set's where they differ: "512/1k".
        let header = |p: usize| {
            let mut sizes: Vec<u32> = Vec::new();
            for set in Set::ALL {
                if let Some(&i) = lineup.shapes.get(&set).and_then(|s| s.prompts.get(p))
                    && !sizes.contains(&i)
                    && rows.iter().any(|&chip| lineup.set(memory(chip)) == set)
                {
                    sizes.push(i);
                }
            }
            sizes.into_iter().map(k_prompt).collect::<Vec<_>>().join("/")
        };
        // A row's slowest and fastest ratio, over every model and prompt size in it, printed at
        // its right.
        let range = |chip: &str, gm: GridMetric, u: u32| -> Option<(f64, f64)> {
            (classes.iter())
                .filter_map(|class| grids[&(chip, model_of(chip, class)?, gm.name)].as_ref().ok())
                .flat_map(|g| g.cells.iter().filter(|c| c.0 == u).filter_map(|c| c.2))
                .fold(None, |acc, r| match acc {
                    None => Some((r, r)),
                    Some((lo, hi)) => Some((f64::min(lo, r), f64::max(hi, r))),
                })
        };
        // Over the grid's first and last two columns the tooltip reaches inward, not off its edge.
        let align = |k: usize, p: usize| match (k, p) {
            (0, p) if p < 2 => "top-left",
            (k, p) if k + 1 == classes.len() && p + 2 >= most => "top-right",
            _ => "top",
        };
        // Each square with Carbon's tooltip: the square is its trigger, focusable so a keyboard
        // opens it too; the tooltip gives the ratio (or why there is none), the model on its
        // machine, and the measure, prompt size and users.
        let square = |chip: &str, stem: &str, gm: GridMetric, u: u32, i: u32, align: &str, place: String| -> Element {
            let ratio = match &grids[&(chip, stem, gm.name)] {
                Ok(g) => match g.cells.iter().find(|&&(cu, ci, _)| (cu, ci) == (u, i)) {
                    Some(&(_, _, Some(r))) => Ok((r, g.rival)),
                    _ => Err("no ratio"),
                },
                Err(why) => Err(*why),
            };
            let (fill, label, verdict) = match ratio {
                Ok((r, rival)) => {
                    let step = ratio_step(r);
                    let shown = highlights.contains(&Square { chip, stem, metric: gm.name, users: u, prompt: i });
                    // The steps far from ×1 are dark in light mode and light in dark mode.
                    let ink = if step.abs_diff(RATIO_STEPS.len() / 2) >= 6 {
                        "light-dark(#ffffff, #161616)"
                    } else {
                        "light-dark(#161616, #ffffff)"
                    };
                    (
                        format!("background: {}; color: {ink}", RATIO_STEPS[step]),
                        shown.then(|| format!("×{r:.1}")),
                        format!("×{r:.2} vs {}", rival.label()),
                    )
                }
                Err(why) => (
                    "background: repeating-linear-gradient(135deg, transparent 0 4px, \
                     light-dark(#c6c6c6, #525252) 4px 5px)"
                        .to_string(),
                    None,
                    why.to_string(),
                ),
            };
            rsx! {
                // The grid's own cell, and a flex box: Carbon puts the trigger in an inline box in its
                // shadow DOM, whose line would add room for text under every square.
                cds-tooltip { "align": align, style: "{place}; display: flex",
                    // Its own width: Carbon wraps the trigger in an inline box, so an empty square would
                    // shrink to nothing.
                    div { tabindex: "0",
                        style: "width: 2.25rem; height: 2.25rem; {fill}; display: flex; align-items: center; justify-content: center; font-size: 0.625rem; font-weight: 600",
                        if let Some(label) = label { "{label}" }
                    }
                    cds-tooltip-content { style: "text-align: left",
                        TipHeading { heading: verdict }
                        div { "{stem} on {chip}" }
                        div { "{gm.name} · {i} prompt tokens · {users_at(u)}" }
                    }
                }
            }
        };
        rsx! {
            div { style: "display: grid; grid-template-columns: {template_columns} max-content max-content; column-gap: 2px; row-gap: 2px; align-items: center",
                for (n, line) in lines.iter().enumerate() {
                    match *line {
                        Line::Classes => rsx! {
                            for (k, class) in classes.iter().enumerate() {
                                div { style: "grid-row: {n + 1}; grid-column: {col(k, 0)} / span {most}; text-align: center; font-weight: 600", "{class}" }
                            }
                        },
                        Line::Models => rsx! {
                            for (k, class) in classes.iter().enumerate() {
                                div { style: "grid-row: {n + 1}; grid-column: {col(k, 0)} / span {most}; text-align: center; font-size: 0.75rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap",
                                    {models(class)}
                                }
                            }
                        },
                        Line::Prompts => rsx! {
                            for k in 0..classes.len() {
                                for p in 0..most {
                                    div { style: "grid-row: {n + 1}; grid-column: {col(k, p)}; text-align: center; font-size: 0.625rem; white-space: nowrap", {header(p)} }
                                }
                            }
                        },
                        Line::Measure(chip, gm, u) => {
                            let (label, about) = match gm.load {
                                Load::Alone => (gm.letter.to_string(), gm.about.to_string()),
                                Load::Shared => (
                                    format!("{u}{}", gm.letter),
                                    format!("With {u} users at once. {}", gm.about),
                                ),
                            };
                            rsx! {
                                div { style: "grid-row: {n + 1}; grid-column: 2; font-size: 0.875rem; font-weight: 600; text-align: right; padding-right: 0.5rem; white-space: nowrap",
                                    Definition { alignment: Alignment::BottomLeft, heading: gm.name.to_string(), definition: about, strong { "{label}" } }
                                }
                                for (k, class) in classes.iter().enumerate() {
                                    if let Some(stem) = model_of(chip, class) {
                                        for (p, &i) in shape(chip).prompts.iter().enumerate() {
                                            {square(chip, stem, gm, u, i, align(k, p), format!("grid-row: {}; grid-column: {}", n + 1, col(k, p)))}
                                        }
                                    }
                                }
                                if let Some((lo, hi)) = range(chip, gm, u) {
                                    div { style: "grid-row: {n + 1}; grid-column: {col(classes.len(), 0) + 1}; font-size: 0.75rem; white-space: nowrap; padding-left: 1.5rem",
                                        "×{lo:.2}–×{hi:.2}"
                                    }
                                }
                            }
                        }
                    }
                }
                for &(chip, start, end) in &spans {
                    // A margin, not padding: the rotation would turn padding onto the label's far side.
                    div { style: "grid-row: {start + 1} / {end + 2}; grid-column: 1; writing-mode: vertical-rl; transform: rotate(180deg); text-align: center; font-weight: 600; font-size: 0.875rem; margin-right: 1rem",
                        "{chip}"
                    }
                }
                // The ratio scale right beside the squares, as tall as their rows: one cell over
                // all of them, so each row's range, a column further on, stays on its row.
                if let (Some(&(_, first, _)), Some(&(_, _, last))) = (spans.first(), spans.last()) {
                    div { style: "grid-row: {first + 1} / {last + 2}; grid-column: {col(classes.len(), 0)}; align-self: stretch; display: flex; padding-left: 0.5rem",
                        {ratio_scale(ScaleFit::Fill, "scale-one-map")}
                    }
                }
            }
        }
    };
    rsx! {
        Section { id: "glance", level: 2,
            Stack { gap: 4,
                Heading { "At a glance" }
                // One tile, a content switcher choosing its panel: the heat maps,
                // then one panel of sparklines per dimension. The tile shrink-wraps
                // the heat maps' grid instead of spanning the page; the intros and
                // the sparkline panels add no width of their own (an intro would
                // otherwise run on one line as wide as it likes) and stretch to the
                // heat maps'.
                // The tabs flush on their panels, so the latency panel's own two rows of tabs
                // stack right under them; every other panel starts a step down.
                Tile { style: "width: fit-content; max-width: 100%",
                    div {
                        Tabs { selected: slug(ONE_MAP), kind: TabsKind::Contained,
                            TabItem { value: slug(ONE_MAP), target: Some(slug(ONE_MAP)), selected: true, "{ONE_MAP}" }
                            TabItem { value: slug(LATENCY), target: Some(slug(LATENCY)), selected: false, "{LATENCY}" }
                            if GLANCE_TREND_TABS {
                                for (title, _, _) in SCALING {
                                    TabItem { value: slug(title), target: Some(slug(title)), selected: false, "{title}" }
                                }
                                TabItem { value: slug(STARTUP), target: Some(slug(STARTUP)), selected: false, "{STARTUP}" }
                            }
                        }
                        div {
                            ContentPanel { id: slug(ONE_MAP), shown: true,
                                div { style: "padding-top: var(--cds-spacing-05)",
                                Stack { gap: 5,
                                    p { style: "contain: inline-size",
                                        "Every Mac, model and measure in one map at {answers}-token answers: a column per \
                                         prompt size under each model, shortest first (where the 32–36 GB and 64 GB Macs ran \
                                         different sizes, the header gives both, smaller Macs' first; hover a square for its \
                                         own), and under each Mac, with 1 user, P its prefill (time to \
                                         first token) and D its decode (time per output token), then 2u and 4u its end-to-end \
                                         time with 2 and 4 users, since once the GPU is busy scheduling only moves each \
                                         request's wait between those two. Colour is how many times faster scratchy is than the run's \
                                         first rival, in the order {rivals}: blue faster, red slower, grey within noise; right \
                                         of each row, its slowest and fastest. Hover a square for its ratio and engine."
                                    }
                                    {one_map}
                                }
                                }
                            }
                            // Each a machine × model matrix, every cell its panel's two
                            // sparklines, one over the other. The 2rem between models
                            // holds the end tick labels Carbon centres on each
                            // chart's edges.
                            for (title, along, [upper, lower]) in SCALING.into_iter().filter(|_| GLANCE_TREND_TABS) {
                                ContentPanel { id: slug(title), shown: false,
                                    div { style: "contain: inline-size; padding-top: var(--cds-spacing-05)",
                                        Stack { gap: 5,
                                            p { "scratchy against oMLX TurboQuant (or the next engine a run has, in the order of the heat maps) along {along}. Blue fill where scratchy is ahead, red where it is behind." }
                                            {glance_maps(&rows, &classes, "minmax(0, 1fr)", false, |chip: &str, class: &str| {
                                                        let Some(stem) = model_of(chip, class) else { return rsx! {} };
                                                        rsx! {
                                                            div { style: "flex: 1 1 0; min-width: 0",
                                                                Stack { gap: 1,
                                                                    span { style: "font-size: 0.75rem; font-weight: 600", "{upper.label()}" }
                                                                    {spark(chip, stem, upper)}
                                                                    span { style: "font-size: 0.75rem; font-weight: 600", "{lower.label()}" }
                                                                    {spark(chip, stem, lower)}
                                                                    div { style: "font-size: 0.75rem; text-align: center; overflow-wrap: anywhere", "{stem}" }
                                                                }
                                                            }
                                                        }
                                                    })}
                                        }
                                    }
                                }
                            }
                            if GLANCE_TREND_TABS {
                                ContentPanel { id: slug(STARTUP), shown: false,
                                    div { style: "contain: inline-size; padding-top: var(--cds-spacing-05)", {startup_panel(machines, stems.len())} }
                                }
                            }
                            ContentPanel { id: slug(LATENCY), shown: false,
                                div { style: "contain: inline-size", {latency_panel(machines, &rows, &classes, model_of)} }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The glance's sparkline grids: `columns` equal columns, each machine's name
/// a line of its own over its row's upper left rather than a column.
fn spark_grid(columns: usize) -> String {
    format!(
        "display: grid; grid-template-columns: repeat({columns}, minmax(0, 1fr)); \
         gap: 0.25rem 2rem; align-items: center"
    )
}

/// The glance's latency panel's name.
const LATENCY: &str = "Latency Distributions";

/// The overview's latency panel: under the glance's tabs, a row of line tabs choosing a
/// machine and under it one choosing a model class, stacked flush; then the chosen machine's
/// model's every request at the heat maps' answer length, each chart on its own axes
/// ([`Latency::multiple`]), its legend to its right; a dash for a run from before the run file
/// kept requests. The two rows choose one panel of every pair between them, so the panels are one
/// flat list, LATENCY_JS showing the pair both rows name.
fn latency_panel<'l>(
    machines: &BTreeMap<&str, MachineRuns>,
    rows: &[&str],
    classes: &[&str],
    model_of: impl Fn(&str, &str) -> Option<&'l str>,
) -> Element {
    let machine = |chip: &str| format!("latency-machine-{}", slug(chip));
    let model = |class: &str| format!("latency-model-{}", slug(class));
    let pair = |chip: &str, class: &str| -> Element {
        let Some(stem) = model_of(chip, class) else {
            return rsx! { div { "not run" } };
        };
        let Some(Entry { run, m }) = machines[chip].latest(stem) else {
            return rsx! { div { "not run" } };
        };
        let Some(l) = Latency::of(m, run) else {
            return rsx! { div { title: "this run did not keep its requests", "{DASH} {stem}" } };
        };
        let o = glance_answer(run);
        let about = format!("{stem} on {chip}");
        // A 1-user cell its TTFT's and TPOT's distributions, one over the other; a cell under load
        // its end-to-end time's ([`Load`]). Each chart names its own users count, under its timing.
        l.multiple(o, "minmax(0, 1fr)", &format!("{stem} · {o}-token answers"), UsersNamed::InCells, |(u, o, i)| {
            let label = |t: RequestTiming| format!("{} of every request, {i} in {o} out {u} at once, {about}", t.title());
            let users = users_at(u);
            let ecdf = |t: RequestTiming, tag: &str| l.ecdf((u, o, i), t, &[tag, &users], &label(t), ECDF_HEIGHT);
            rsx! {
                Stack { gap: 1,
                    match Load::of(u) {
                        Load::Alone => rsx! {
                            {ecdf(RequestTiming::Ttft, "TTFT")}
                            {ecdf(RequestTiming::Tpot, "TPOT")}
                        },
                        Load::Shared => ecdf(RequestTiming::E2el, "E2EL"),
                    }
                }
            }
        })
    };
    let (first_chip, first_class) = (
        rows.first().copied().unwrap_or_default(),
        classes.first().copied().unwrap_or_default(),
    );
    rsx! {
        Tabs { selected: machine(first_chip), id: "latency-machines",
            for &chip in rows {
                TabItem { value: machine(chip), selected: chip == first_chip, "{chip}" }
            }
        }
        Tabs { selected: model(first_class), id: "latency-models",
            for &class in classes {
                TabItem { value: model(class), selected: class == first_class, "{class}" }
            }
        }
        script { dangerous_inner_html: LATENCY_JS }
        div { style: "padding-top: var(--cds-spacing-05)",
            Stack { gap: 5,
                // The engines' colours keyed in the text itself: "● scratchy versus ● oMLX".
                p {
                    for (n, (&e, class)) in latency_engines(machines).iter().zip(["s1", "s2"]).enumerate() {
                        if n > 0 { " versus " }
                        {engine_key(e, class)}
                    }
                    ": in each cell at the heat maps' answer length, users at once down and prompt size \
                     across, the share of each one's requests (up) done within each time (across): with 1 user, \
                     time to first token over time per output token; with more, end-to-end time, since once the \
                     GPU is busy scheduling only moves each request's wait between those two. Time runs on a log \
                     axis ×3 wide on every chart, centred on its two medians, so a factor is one width everywhere; \
                     a request past an edge is drawn at it. Further left is faster; a steeper rise is steadier."
                }
                div {
                    for &chip in rows {
                        for &class in classes {
                            div { id: format!("latency-{}-{}", slug(chip), slug(class)), "data-panel": "", "data-latency-pair": "", "data-machine": machine(chip), "data-model": model(class),
                                hidden: (chip != first_chip || class != first_class).then_some(true),
                                {pair(chip, class)}
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Shows the latency pair the machine and model tabs both name, hiding the rest, and keeps it in
/// the address: choosing either tab puts the pair's id there; an address naming a pair, a machine
/// or a model (on load, or when it changes) opens the glance's latency tab on it.
const LATENCY_JS: &str = "(function () {\
  var machine = document.getElementById('latency-machines'), model = document.getElementById('latency-models');\
  var panel = machine.closest('[data-panel]');\
  function show() {\
    var shown;\
    document.querySelectorAll('[data-latency-pair]').forEach(function (p) {\
      p.hidden = p.dataset.machine !== machine.value || p.dataset.model !== model.value;\
      if (!p.hidden) shown = p;\
    });\
    return shown;\
  }\
  function chose() {\
    var p = show();\
    if (p) history.replaceState(null, '', '#' + p.id);\
  }\
  function fromHash() {\
    var id = decodeURIComponent(location.hash.slice(1)), has = function (tabs) {\
      return id && tabs.querySelector('cds-tab[value=\"' + CSS.escape(id) + '\"]');\
    };\
    var p = id && document.getElementById(id);\
    if (p && p.hasAttribute('data-latency-pair')) { machine.value = p.dataset.machine; model.value = p.dataset.model; }\
    else if (has(machine)) machine.value = id;\
    else if (has(model)) model.value = id;\
    else { show(); return; }\
    var tab = document.querySelector('cds-tab[target=\"' + CSS.escape(panel.id) + '\"]');\
    if (tab) tab.closest('cds-tabs').value = panel.id;\
    show();\
  }\
  machine.addEventListener('cds-tabs-selected', chose);\
  model.addEventListener('cds-tabs-selected', chose);\
  customElements.whenDefined('cds-tabs')\
    .then(function () { return Promise.all([machine.updateComplete, model.updateComplete]); })\
    .then(function () {\
      fromHash();\
      window.addEventListener('hashchange', fromHash);\
    });\
})();";

/// The engines the overview's latency panel draws: scratchy, and the first rival any machine's
/// newest run kept requests for.
fn latency_engines(machines: &BTreeMap<&str, MachineRuns>) -> Vec<Engine> {
    std::iter::once(Engine::Scratchy)
        .chain((Engine::RIVALS.into_iter()).find(|e| {
            (machines.values()).any(|mr| {
                (mr.models.iter()).any(|(_, es)| {
                    es.last()
                        .is_some_and(|en| e.cells(en.m).iter().any(|c| c.requests.is_some()))
                })
            })
        }))
        .collect()
}

/// The glance's startup panel's name.
const STARTUP: &str = "Startup by model size";

// The startup panel compares scratchy with the first of Engine::RIVALS a run
// timed (StartupSpark::rival): oMLX TurboQuant where it has it, mlx-lm (which
// loads the same MLX checkpoint) in runs from before oMLX.

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
            carry: false,
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
            carry: false,
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
            carry: false,
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

/// A model's parameter size as its name gives it: the first `-`-separated
/// part that is a number then `b` ("granite-3.3-2b-instruct": "2b"; not
/// gemma's active "a4b").
fn size_name(stem: &str) -> Option<String> {
    stem.split('-')
        .find(|part| {
            part.strip_suffix('b')
                .is_some_and(|n| !n.is_empty() && n.parse::<f64>().is_ok())
        })
        .map(str::to_string)
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
    /// The engine scratchy is compared with: the first of Engine::RIVALS the
    /// chosen run timed.
    rival: Engine,
}

impl<'a> StartupSpark<'a> {
    /// Its steps' tick labels: each model's parameter size, as its name gives
    /// it ("2b"), else its size in GB. The steps stay where the GB put them.
    fn names(&self) -> Vec<String> {
        self.models
            .iter()
            .map(|&(m, b)| size_name(&m.stem).unwrap_or_else(|| gb1(b).to_string()))
            .collect()
    }

    /// From the newest of the machine's runs that timed both engines for the
    /// most models, so a sparkline's steps share one commit and one set of
    /// engine versions; or why there is none.
    fn of(mr: &MachineRuns<'a>, col: &StartupColumn) -> Result<Self, &'static str> {
        // A run's rival: the first engine it timed alongside scratchy.
        let rival_of = |run: &Run| {
            Engine::RIVALS.into_iter().find(|&e| {
                run.models
                    .iter()
                    .any(|m| col.time(Engine::Scratchy, m).is_some() && col.time(e, m).is_some())
            })
        };
        let both = |run: &Run| {
            rival_of(run).map_or(0, |e| {
                run.models
                    .iter()
                    .filter(|m| col.time(Engine::Scratchy, m).is_some() && col.time(e, m).is_some())
                    .count()
            })
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
        let rival = rival_of(run).ok_or("nothing to compare")?;
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
                rival: col.time(rival, m),
                highlight: false,
            })
            .collect();
        Ok(StartupSpark {
            models,
            steps,
            rival,
        })
    }
}

/// Every machine's startup against model size, scratchy against its rival, in
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
                "scratchy against oMLX TurboQuant (or the next engine a run timed, in the order of the heat maps; \
                 mlx-lm in runs from before oMLX) as the model grows ({sizes}, the MLX checkpoint they load). Blue fill \
                 where scratchy is ahead, red where it is behind. Frozen is the first launch after memory and \
                 scratchy's weights cache are cleared, cold a normal relaunch: both time launch to the first \
                 token, not to ready, because mlx-lm reports ready at about the same time whatever the model's \
                 size and its loading lands in the first request. Warm is a request to a server already running, \
                 send to first token. Each sparkline is one run: the newest that timed both engines for the most models."
            }
            div { style: spark_grid(columns.max(STARTUP_COLUMNS.len())),
                for col in &STARTUP_COLUMNS { div { style: "text-align: center", "{col.title}" } }
                for chip in in_order(machines) {
                    div { style: "grid-column: 1 / -1; margin-top: 0.5rem", strong { "{chip}" } }
                    for (k, col) in STARTUP_COLUMNS.iter().enumerate() {
                        div {
                            match &sparks[&(chip, k)] {
                                Ok(sp) => sparkline(&sp.steps, col.axes, Engine::Scratchy.side(), sp.rival.side(), Some(sp.names()), GLANCE_CHART_HEIGHT, &format!("{chip}, {}: {} by model size, scratchy against {}", col.title, col.axes.value, sp.rival.label())),
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
                            for e in [Engine::Scratchy, sp.rival] {
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

/// The glance's first panel's name: every heat map as one.
const ONE_MAP: &str = "Heat map";

/// Whether the overview shows its sparkline ([`SCALING`]) and startup tabs: not for now, so its
/// heat maps and latency distributions carry it; the panels stay built.
const GLANCE_TREND_TABS: bool = false;
/// The sparkline panels: a dimension, what runs along it, and its two trends.
const SCALING: [(&str, &str, [Trend; 2]); 2] = [
    (
        "Concurrency scaling",
        "users at once, at the grid's one-user summary cell (1024-token prompts, and the answer length the heat maps use); runs from before the grid had its own users (#349) at 1, 4 and 16 users, 512-token prompts and 128-token answers",
        [Trend::UsersEndToEnd, Trend::UsersDecode],
    ),
    (
        "Length scaling",
        "prompt length for prefill and answer length for decode, along the grid's prompt and answer lengths, one user at a time where the grid ran at 1 (8 users at once in runs from before). Each prefill point is the geometric mean over the answer lengths, and each decode point over the prompt lengths",
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
                            Tile { id: mid.clone(),
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
                                    {
                                        // An engine whose own memory guard refused the model
                                        // outright: its columns are dashes because it never
                                        // served. This says why, in the engine's words.
                                        let declined: Vec<String> = Engine::ALL.iter()
                                            .filter_map(|e| m.engine_declined.as_ref()
                                                .and_then(|d| d.get(e.key()))
                                                .map(|r| format!("{} — {}", e.label(), r.as_str())))
                                            .collect();
                                        rsx! { if !declined.is_empty() {
                                            p { "Declined before serving: {declined.join(\"; \")}." }
                                        } }
                                    }
                                    {summary_table(m, run, prev)}
                                    {sweep_part(m, run, prev, Sweep::Users)}
                                    {grid_maps(m, &mid, run)}
                                    {request_plots(m, &mid, run)}
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

/// The engines the page compares scratchy with, in [`Engine::RIVALS`] order: those any machine's
/// shown models measured (oMLX TurboQuant under oMLX), written as a list, or None for none.
fn rivals_named(machines: &BTreeMap<&str, MachineRuns>) -> Option<String> {
    let shown: Vec<&Model> = (machines.values())
        .flat_map(|mr| mr.models.iter().filter_map(|(_, es)| es.last()))
        .map(|e| e.m)
        .collect();
    let mut names: Vec<&str> = Vec::new();
    for e in Engine::RIVALS {
        let name = match e {
            Engine::OmlxTq => Engine::Omlx.label(),
            _ => e.label(),
        };
        if shown.iter().any(|m| e.ran(m)) && !names.contains(&name) {
            names.push(name);
        }
    }
    match names.as_slice() {
        [] => None,
        [one] => Some(one.to_string()),
        [rest @ .., last] => Some(format!("{} and {last}", rest.join(", "))),
    }
}

pub fn build(site: &Site, assets: &Published) -> Result<String, String> {
    let runs = load(site)?;
    // Over time reads every published run, the ones a newer run replaced too; the rest of the
    // page shows each Mac's two newest.
    let all_runs = load_from(site, &[DATA, HISTORY])?;
    let all_machines = index(&all_runs);
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
        // No link to a section that is not there.
        let over = over_time(&all_machines);
        if over.is_some() {
            sections.push(("over-time".to_string(), "Over time".to_string()));
        }
        let order = in_order(&machines);
        sections.extend(order.iter().map(|chip| (slug(chip), chip.to_string())));
        rsx! {
            {glance(&machines)}
            if let Some(over) = over { {over} }
            for chip in &order { {machine_section(chip, &machines[chip])} }
        }
    };
    let (count, n_machines) = (runs.len(), machines.len());
    let against =
        rivals_named(&machines).map_or("scratchy".to_string(), |r| format!("scratchy against {r}"));
    let page = rsx! {
        {chrome::header(Root(0), Some(Tab::Performance))}
        {chrome::perf_side_nav("metal.html", &sections)}
        Content {
            div {
              Stack { gap: 6,
                Stack { gap: 4,
                    Heading { "Metal performance" }
                    p {
                        "{against} on Apple \
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
        script { dangerous_inner_html: crate::charts::CHARTS_JS }
    };
    let description = format!(
        "{against} on Apple silicon: startup, single-user speed, and scaling, per machine."
    );
    let html = chrome::document(
        Head {
            assets,
            title: "Metal performance — scratchy",
            description: Some(&description),
            og: None,
            root: Root(0),
            libraries: &[Library::CarbonCharts],
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
    #[test]
    fn a_cells_requests_read_per_repeat_and_untimed_ones_drop_out() {
        use super::data::{Cell, RequestTiming};
        let c: Cell = serde_json::from_value(serde_json::json!({
            "axis": "grid", "rung": "1024x64", "conc": 2,
            "median_ttft_ms": 900.0, "median_tpot_ms": 20.0, "output_throughput": 50.0,
            "completed": 4, "unstreamed_requests": 1,
            "requests": {"rep": [0, 0, 1, 1], "ttft_ms": [100.0, 200.0, 300.0, null],
                         "tpot_ms": [15.0, 16.0, 17.0, null], "e2el_ms": [1.0, 2.0, 3.0, 4.0],
                         "output_tokens": [64, 64, 64, 64]},
        }))
        .unwrap();
        let rs = c.requests.as_ref().unwrap();
        let ttft: Vec<_> = rs.of(RequestTiming::Ttft).collect();
        assert_eq!(
            ttft,
            [
                (0, Some(100.0)),
                (0, Some(200.0)),
                (1, Some(300.0)),
                (1, None)
            ]
        );
        let mut timed: Vec<f64> = ttft.iter().filter_map(|(_, v)| *v).collect();
        timed.sort_by(f64::total_cmp);
        assert_eq!(super::quantile(&timed, 50.0), Some(200.0));
        assert_eq!(super::quantile(&timed, 25.0), Some(150.0));
        assert_eq!(super::quantile(&[], 50.0), None);
        // A run from before the run file kept requests reads as none.
        let old: Cell = serde_json::from_value(serde_json::json!({
            "axis": "grid", "rung": "1024x64", "conc": 2, "median_ttft_ms": 900.0,
            "median_tpot_ms": 20.0, "output_throughput": 50.0, "completed": 4,
            "unstreamed_requests": 0,
        }))
        .unwrap();
        assert!(old.requests.is_none());
    }

    use super::{Run, size_name, slug};

    /// The runs the tests read: copies of published runs, trimmed, that no data PR touches.
    pub(super) const FIXTURES: &str = "tests/fixtures/metal";

    #[test]
    fn a_models_name_gives_its_parameter_size() {
        assert_eq!(size_name("granite-3.3-2b-instruct").as_deref(), Some("2b"));
        assert_eq!(size_name("qwen2.5-7b").as_deref(), Some("7b"));
        assert_eq!(size_name("gemma-4-26b-a4b-it").as_deref(), Some("26b"));
        assert_eq!(size_name("smollm2-135m"), None);
        // The one-model-per-class round.
        assert_eq!(size_name("gpt-oss-20b").as_deref(), Some("20b"));
        assert_eq!(size_name("gpt-oss-120b-3bit").as_deref(), Some("120b"));
        assert_eq!(size_name("nemotron-3-nano-30b-a3b").as_deref(), Some("30b"));
        assert_eq!(
            size_name("nemotron-3-super-120b-a12b").as_deref(),
            Some("120b")
        );
        assert_eq!(size_name("qwen3.5-122b-a10b").as_deref(), Some("122b"));
        assert_eq!(size_name("granite-4.1-8b").as_deref(), Some("8b"));
    }

    /// A later run of scratchy alone, without the frozen rung, shows the engines and the frozen
    /// launches of the run before it, each marked with that run's time, and only its own scratchy
    /// cells and cold and warm launches.
    #[test]
    fn a_run_of_scratchy_alone_keeps_the_last_engines_and_frozen_launches() {
        use super::data::{BORROWED_FROZEN, Run, Scenario};
        use super::numbers::Engine;
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/metal/apple-m5-max-2026-10-14T111352Z-987b06a2.json"
        );
        let text = std::fs::read_to_string(path).expect("the published M5 Max run");
        let mut later: serde_json::Value = serde_json::from_str(&text).unwrap();
        later["generated_utc"] = "2026-10-21T12:00:00Z".into();
        for m in later["models"].as_array_mut().unwrap() {
            let m = m.as_object_mut().unwrap();
            m.retain(|k, _| !(k.starts_with("scaling_omlx") || k.starts_with("cache_ladder_omlx")));
            m["cache_ladder"]
                .as_array_mut()
                .unwrap()
                .retain(|r| r["scenario"] != "frozen");
            m["scaling"].as_array_mut().unwrap().truncate(3);
        }
        let earlier: Run = serde_json::from_str(&text).unwrap();
        let later: Run = serde_json::from_value(later).unwrap();
        let mut runs = vec![later, earlier];
        super::borrow_earlier(&mut runs);
        let [later, earlier] = &runs[..] else {
            unreachable!()
        };
        for (new, old) in later.models.iter().zip(&earlier.models) {
            let from = old.borrowed.is_empty().then_some(&earlier.generated_utc);
            for e in [Engine::OmlxTq, Engine::Omlx] {
                assert_eq!(
                    e.cells(new).len(),
                    e.cells(old).len(),
                    "{} {}",
                    new.stem,
                    e.key()
                );
                assert_eq!(new.borrowed.get(e.key()), from);
            }
            let frozen = |m: &super::data::Model| {
                (Engine::Scratchy.ladder(m).iter())
                    .filter(|r| r.scenario == Scenario::Frozen)
                    .count()
            };
            assert_eq!(frozen(new), frozen(old));
            assert_eq!(new.borrowed.get(BORROWED_FROZEN), from);
            // scratchy's own cells are the later run's three, never the earlier run's.
            assert_eq!(Engine::Scratchy.cells(new).len(), 3);
            assert!(
                old.borrowed.is_empty(),
                "{}: the earlier run measured its own",
                old.stem
            );
        }
    }

    /// An engine measured at another shape (here, a run with other prompt lengths) is not
    /// carried into a later run: its cells are other cells. scratchy's frozen launches still are.
    #[test]
    fn an_engine_measured_at_another_shape_is_not_carried() {
        use super::data::{BORROWED_FROZEN, Run};
        use super::numbers::Engine;
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/metal/apple-m5-max-2026-10-14T111352Z-987b06a2.json"
        );
        let text = std::fs::read_to_string(path).expect("the published M5 Max run");
        let mut later: serde_json::Value = serde_json::from_str(&text).unwrap();
        later["generated_utc"] = "2026-10-21T12:00:00Z".into();
        later["config"]["scaling"]["grid_input"] = serde_json::json!([128, 1024, 4096, 32768]);
        for m in later["models"].as_array_mut().unwrap() {
            let m = m.as_object_mut().unwrap();
            m.retain(|k, _| !(k.starts_with("scaling_omlx") || k.starts_with("cache_ladder_omlx")));
            m["cache_ladder"]
                .as_array_mut()
                .unwrap()
                .retain(|r| r["scenario"] != "frozen");
        }
        let earlier: Run = serde_json::from_str(&text).unwrap();
        let later: Run = serde_json::from_value(later).unwrap();
        let mut runs = vec![later, earlier];
        super::borrow_earlier(&mut runs);
        for m in &runs[0].models {
            for e in [Engine::OmlxTq, Engine::Omlx] {
                assert!(e.cells(m).is_empty(), "{} {}", m.stem, e.key());
                assert!(!m.borrowed.contains_key(e.key()));
            }
            assert!(m.borrowed.contains_key(BORROWED_FROZEN), "{}", m.stem);
        }
    }

    /// A quick run after a full run borrows the full run's engines: the quick run's 64-token
    /// answers are the full run's users × prompt map, and the full run's 512- and 2048-token
    /// answers ran at 1 user only, so they are not cells the quick run compares.
    #[test]
    fn a_quick_run_after_a_full_run_borrows_its_engines() {
        use super::data::Run;
        use super::numbers::Engine;
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/metal/apple-m3-pro-2026-10-21T145041Z-a7820fbc.json"
        );
        let text = std::fs::read_to_string(path).expect("the M3 Pro's full run");
        let mut quick: serde_json::Value = serde_json::from_str(&text).unwrap();
        quick["generated_utc"] = "2026-10-22T12:00:00Z".into();
        quick["config"]["scaling"]["grid_output"] = serde_json::json!([64]);
        for m in quick["models"].as_array_mut().unwrap() {
            let m = m.as_object_mut().unwrap();
            m.retain(|k, _| !(k.starts_with("scaling_omlx") || k.starts_with("cache_ladder_omlx")));
        }
        let full: Run = serde_json::from_str(&text).unwrap();
        let quick: Run = serde_json::from_value(quick).unwrap();
        let mut runs = vec![quick, full];
        super::borrow_earlier(&mut runs);
        let [quick, full] = &runs[..] else {
            unreachable!()
        };
        let mut borrowed = 0;
        for (q, f) in quick.models.iter().zip(&full.models) {
            for e in [Engine::OmlxTq, Engine::Omlx] {
                assert_eq!(e.cells(q).len(), e.cells(f).len(), "{} {}", q.stem, e.key());
                if !e.cells(f).is_empty() {
                    assert_eq!(q.borrowed.get(e.key()), Some(&full.generated_utc));
                    borrowed += 1;
                }
            }
        }
        assert!(borrowed > 0, "the full run measured oMLX on some model");
    }

    /// A run in the quick-run shape: a 4x4 users x prompt grid at 64-token answers, out to 32k,
    /// with 512- and 2048-token answers at 1 user only. The users curve reads all four users
    /// counts at the base cell (1024 in / 64 out), the overview reads the 64-token map, and the
    /// length curve has every answer length at 1 user.
    #[test]
    fn a_run_in_the_quick_shape_reads_every_view_from_its_own_grid() {
        use super::data::{At, Run};
        use super::numbers::Engine;
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/metal/apple-m5-max-2026-10-14T111352Z-987b06a2.json"
        );
        let mut v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let (ins, users) = ([128u32, 1024, 4096, 32768], [1u32, 4, 8, 16]);
        v["config"]["scaling"] = serde_json::json!({
            "axes": ["grid"], "grid_input": ins, "grid_output": [64, 512, 2048],
            "grid_conc": users, "grid_users_output": [64],
            "base": {"input": 1024, "output": 64, "conc": 8},
        });
        let cell = |i: u32, o: u32, c: u32| {
            serde_json::json!({"axis": "grid", "rung": format!("{i}x{o}"), "conc": c,
                "median_ttft_ms": 100.0, "median_tpot_ms": 10.0, "output_throughput": 50.0,
                "completed": 12, "total_output_tokens": 12 * o})
        };
        let mut cells = Vec::new();
        for &c in &users {
            for &i in &ins {
                for o in [64u32, 512, 2048] {
                    if c == 1 || o == 64 {
                        cells.push(cell(i, o, c));
                    }
                }
            }
        }
        let m = &mut v["models"][0];
        m["scaling"] = cells.clone().into();
        m["scaling_omlx"] = cells.into();
        let v_capped = v.clone();
        let run: Run = serde_json::from_value(v).unwrap();
        let m = &run.models[0];
        // The users curve: every users count, at the base cell.
        assert!(matches!(
            super::users_from(m, &run),
            super::UsersFrom::Grid {
                input: 1024,
                output: 64
            }
        ));
        let us = super::by_users(m, &run, Engine::Scratchy);
        assert_eq!(us.keys().copied().collect::<Vec<_>>(), users);
        // The overview's map: 64-token answers, 4 users counts x 4 prompts.
        assert_eq!(super::glance_answer(&run), 64);
        let map = super::ShapeGrid::of(m, &run, super::Slice::Answer(64)).expect("a map");
        assert_eq!((map.down.len(), map.across.len()), (4, 4));
        // The length curve: every answer length, at 1 user.
        let one = super::ShapeGrid::of(m, &run, super::Slice::Users(1)).expect("the 1-user grid");
        assert_eq!(one.across, vec![64, 512, 2048]);
        assert!(
            Engine::Scratchy
                .cells(m)
                .iter()
                .all(|c| matches!(c.at, At::Grid { .. }))
        );
        // A Mac that held prompts up to 4096 shows a 4x3 map: no 32k column.
        let mut capped = v_capped;
        capped["config"]["scaling"]["max_prompt"] = 4096.into();
        let run: Run = serde_json::from_value(capped).unwrap();
        let map = super::ShapeGrid::of(&run.models[0], &run, super::Slice::Answer(64)).unwrap();
        assert_eq!(map.across, vec![128, 1024, 4096]);
    }

    /// Splash serves the Qwen models only: its cells are read on the one model that has them, and
    /// never make it the headline rival over oMLX.
    #[test]
    fn splash_cells_are_read_where_it_ran_and_omlx_stays_the_rival() {
        use super::data::Run;
        use super::numbers::Engine;
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/metal/apple-m3-pro-2026-10-21T145041Z-a7820fbc.json"
        );
        let mut v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let qwen = (v["models"].as_array().unwrap().iter())
            .position(|m| m["stem"] == "qwen3.6-35b-a3b")
            .unwrap();
        let cells = v["models"][qwen]["scaling_omlx"].clone();
        v["models"][qwen]["scaling_splash"] = cells;
        let run: Run = serde_json::from_value(v).unwrap();
        for m in &run.models {
            assert_eq!(
                Engine::Splash.ran(m),
                m.stem == "qwen3.6-35b-a3b",
                "{}",
                m.stem
            );
        }
        let m = &run.models[qwen];
        assert!(!Engine::Splash.cells(m).is_empty());
        assert_eq!(Engine::RIVALS.last().map(|e| e.key()), Some("splash"));
        assert_eq!(Engine::Splash.key(), "splash");
    }

    /// Over time takes every run of a Mac it is given.
    #[test]
    fn over_time_takes_every_run_it_is_given() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let site = crate::Site {
            repo: root.join(".."),
            out: root.join("_site"),
            root,
        };
        let runs = super::load_from(&site, &[FIXTURES]).expect("the fixtures");
        let machines = super::index(&runs);
        // The M3 Pro fixtures hold two runs of the current round's gpt-oss-20b, a week apart:
        // Over time draws them, not just the newest.
        let m3 = &machines["Apple M3 Pro"];
        let gpt = (m3.models.iter())
            .find(|(s, _)| *s == "gpt-oss-20b")
            .map(|(_, es)| es.len());
        assert_eq!(gpt, Some(2), "gpt-oss-20b runs on the M3 Pro");
        assert!(super::over_time(&machines).is_some());
    }

    /// Machines read up from the smallest: a 32 GB base M5 before the 36 GB M3 Pro, the 64 GB
    /// M4 Max and M5 Max after. Below 64 GB a Mac's overview row is the small set, at 64 GB the
    /// large one.
    #[test]
    fn machines_order_by_memory_and_take_their_set_by_it() {
        use super::data::Run;
        let read = |name: &str| -> serde_json::Value {
            let p = format!("{}/tests/fixtures/metal/{name}", env!("CARGO_MANIFEST_DIR"));
            serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
        };
        let mut base = read("apple-m3-pro-2026-10-21T145041Z-a7820fbc.json");
        base["machine"]["chip"] = "Apple M5".into();
        base["machine"]["memory_gb"] = 32.into();
        // Below 64 GB the grid runs 1, 2 and 4 users; the 36 GB M3 Pro's run is from before that.
        base["config"]["scaling"]["grid_conc"] = serde_json::json!([1, 2, 4]);
        base["generated_utc"] = "2026-10-26T20:05:50Z".into();
        let runs: Vec<Run> = [
            base,
            read("apple-m3-pro-2026-10-21T145041Z-a7820fbc.json"),
            read("apple-m4-max-2026-10-13T221900Z-2683d8c3.json"),
            read("apple-m5-max-2026-10-14T111352Z-987b06a2.json"),
        ]
        .into_iter()
        .map(|v| serde_json::from_value(v).unwrap())
        .collect();
        let machines = super::index(&runs);
        assert_eq!(
            super::in_order(&machines),
            ["Apple M5", "Apple M3 Pro", "Apple M4 Max", "Apple M5 Max"]
        );
        // The overview's columns are the full run's recorded lineup, each Mac on its set, and each
        // set's maps the grid of its own newest run: the small set the M5's, not the older M3
        // Pro's, and the large set, with no lineup run of its own, the M5 Max's.
        let lineup = super::lineup(&machines).expect("the M3 Pro's run records its lineup");
        let small = &lineup.shapes[&super::Set::Small];
        assert_eq!(
            (&small.users[..], &small.prompts[..]),
            (&[1, 2, 4][..], &[128, 1024, 4096, 32768][..])
        );
        let large = &lineup.shapes[&super::Set::Large];
        assert_eq!(
            (&large.users[..], &large.prompts[..]),
            (&[1, 2, 4, 8, 16][..], &[128, 1024, 4096][..])
        );
        assert!(lineup.set(36) == super::Set::Small && lineup.set(64) == super::Set::Large);
        // A lineup from before `default_set`: the large set from 64 GB.
        assert!(!lineup.small_everywhere);
        assert_eq!(lineup.model("Dense", 32), Some("granite-4.2-8b"));
        assert_eq!(lineup.model("Dense", 36), Some("granite-4.2-8b"));
        assert_eq!(lineup.model("Dense", 64), Some("granite-4.2-30b"));
    }

    /// A lineup whose newest run records `default_set: "small"`: every Mac's column is the small
    /// model, a 64 GB Mac's too, so the overview compares one model across every chip.
    #[test]
    fn a_small_set_lineup_puts_every_mac_on_the_small_models() {
        use super::data::Run;
        // The M3 Pro's 2026-10-27 run at f9739363, a newer one replaced.
        let read = |name: &str| -> serde_json::Value {
            let p = format!("{}/tests/fixtures/metal/{name}", env!("CARGO_MANIFEST_DIR"));
            serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
        };
        let mut small = read("apple-m3-pro-2026-10-27T031941Z-f9739363.json");
        small["config"]["default_set"] = "small".into();
        let mut big = small.clone();
        big["machine"]["chip"] = "Apple M5 Max".into();
        big["machine"]["memory_gb"] = 128.into();
        let runs: Vec<Run> = [small, big]
            .into_iter()
            .map(|v| serde_json::from_value(v).unwrap())
            .collect();
        let machines = super::index(&runs);
        let lineup = super::lineup(&machines).expect("a lineup");
        assert!(lineup.small_everywhere);
        // This run's class table names them in lower case.
        for gb in [36, 64, 128] {
            assert_eq!(lineup.model("dense", gb), Some("granite-4.2-3b"), "{gb} GB");
            assert_eq!(
                lineup.model("sliding window", gb),
                Some("gpt-oss-20b"),
                "{gb} GB"
            );
        }
    }

    /// A run that named its models (`--models large`) says nothing about the default set: published
    /// after a default small-set run, it leaves every Mac's column on the small model, rather than
    /// flipping the overview back to the large set from 64 GB.
    #[test]
    fn a_run_that_named_its_models_does_not_flip_the_lineup() {
        use super::data::Run;
        // The M3 Pro's 2026-10-27 run at f9739363, a newer one replaced.
        let read = |name: &str| -> serde_json::Value {
            let p = format!("{}/tests/fixtures/metal/{name}", env!("CARGO_MANIFEST_DIR"));
            serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
        };
        let mut small = read("apple-m3-pro-2026-10-27T031941Z-f9739363.json");
        small["config"]["default_set"] = "small".into();
        // The 64 GB Mac's 120B round, run by name, a day later.
        let mut large = small.clone();
        large["machine"]["chip"] = "Apple M5 Max".into();
        large["machine"]["memory_gb"] = 128.into();
        large["generated_utc"] = "2026-10-28T12:00:00Z".into();
        large["config"]["default_set"] = serde_json::Value::Null;
        large["config"]["models_given"] = true.into();
        let runs: Vec<Run> = [small, large]
            .into_iter()
            .map(|v| serde_json::from_value(v).unwrap())
            .collect();
        let machines = super::index(&runs);
        let lineup = super::lineup(&machines).expect("a lineup");
        assert!(lineup.small_everywhere);
        assert_eq!(lineup.model("dense", 128), Some("granite-4.2-3b"));
    }

    /// Over every published run: a cell whose answers came back short of the tokens asked for
    /// carries § and says what share it delivered, and a dead cell never does (#479). Whether
    /// scratchy answers to length is the matrix script's to judge as it measures (`run_cell`);
    /// the page marks whatever it is given.
    #[test]
    fn published_cells_that_stopped_answers_early_are_marked_short() {
        use super::numbers::{Engine, Metric, SHORT, cfmt, marks_note, short};
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut by_engine = std::collections::BTreeMap::<&str, usize>::new();
        let mut files: Vec<_> = (std::fs::read_dir(root.join(FIXTURES)).expect(FIXTURES))
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect();
        files.sort();
        for f in files {
            let mut run: Run = serde_json::from_str(&std::fs::read_to_string(&f).unwrap())
                .unwrap_or_else(|e| panic!("{}: {e}", f.display()));
            run.mark_dead();
            run.mark_short();
            for m in &run.models {
                for e in Engine::ALL {
                    for c in e.cells(m) {
                        let Some(share) = short(Some(c)) else {
                            assert!(!cfmt(Some(c), Metric::Tpot).contains(SHORT));
                            continue;
                        };
                        assert!(!c.dead, "a dead cell is never also short");
                        assert!(share < 0.99, "{share}");
                        assert!(
                            cfmt(Some(c), Metric::Tpot).contains(SHORT)
                                || c.median_tpot_ms.is_none()
                        );
                        assert!(
                            marks_note(Some(c), Metric::Tpot).contains("% of the answer tokens")
                        );
                        *by_engine.entry(e.key()).or_default() += 1;
                    }
                }
            }
        }
        eprintln!("short cells by engine: {by_engine:?}");
        assert!(
            by_engine.get("omlx").is_some_and(|&n| n > 0),
            "#479's oMLX cells stop short: {by_engine:?}"
        );
    }

    /// The published runs' cells a dead server measured (#413) are marked, and no other: the
    /// M3 Pro's gemma 4096-token cells from the one its OOM struck in (#412), 4096x16, which still streamed a
    /// few tokens, on. Every engine that stops at EOS early (ollama, oMLX) keeps its numbers, and
    /// so does scratchy's f378699e gemma cell at 16 users, live at 99% of its tokens. A marked
    /// cell counts every request failed, says why, and shows no number.
    #[test]
    fn the_published_cells_of_a_dead_server_and_only_those_are_marked() {
        use super::numbers::{DASH, Engine, FAILED, Metric, cfmt, failed, failed_note, partial};
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut runs: Vec<Run> = Vec::new();
        for dir in [FIXTURES] {
            let mut files: Vec<_> = (std::fs::read_dir(root.join(dir)).expect(dir))
                .map(|e| e.unwrap().path())
                .filter(|p| p.extension().is_some_and(|x| x == "json"))
                .collect();
            files.sort();
            for f in files {
                let mut run: Run = serde_json::from_str(&std::fs::read_to_string(&f).unwrap())
                    .unwrap_or_else(|e| panic!("{}: {e}", f.display()));
                run.mark_dead();
                runs.push(run);
            }
        }
        let mut dead = std::collections::BTreeMap::<(String, String, &str, &str), usize>::new();
        for r in &runs {
            for m in &r.models {
                for e in Engine::ALL {
                    for c in e.cells(m).iter().filter(|c| c.dead) {
                        let all = c.completed.unwrap_or(0) + c.failed.unwrap_or(0);
                        assert_eq!(failed(Some(c)), all, "every request of a dead cell failed");
                        assert!(failed_note(Some(c)).contains("stopped answering"));
                        assert_eq!(cfmt(Some(c), Metric::Ttft), format!("{DASH}{FAILED}"));
                        assert!(!partial(Some(c), Metric::Ttft));
                        let machine = slug(r.machine.chip.as_str());
                        let key = (machine, r.repo.sha8().to_string(), m.stem.as_str(), e.key());
                        *dead.entry(key).or_default() += 1;
                    }
                }
            }
        }
        let want = [
            // The M3 Pro's full run: gpt-oss-20b's server ran out of memory at 4096x2048 (#775),
            // then answered its next two cells, 128x64 and 1024x64 at 4 users, almost empty.
            ("apple-m3-pro", "a7820fbc", "gpt-oss-20b", "scratchy", 3),
            (
                "apple-m3-pro",
                "649ebce2",
                "gemma-4-26b-a4b-it",
                "scratchy",
                3,
            ),
            (
                "apple-m3-pro",
                "cc51e0d8",
                "gemma-4-26b-a4b-it",
                "scratchy",
                3,
            ),
        ];
        let want: std::collections::BTreeMap<_, _> = (want.iter())
            .map(|&(mc, sha, stem, e, n)| ((mc.to_string(), sha.to_string(), stem, e), n))
            .collect();
        assert_eq!(dead, want);
    }

    /// The bench's own verdict on a cell is the one the page takes: a scratchy cell it recorded as
    /// a dead server's (`dead_server`, under half its answer tokens) shows no number, even in a
    /// run that counted its failures itself, where the page judges no cell by its tokens. Without
    /// it the dead server's ~0 tok/s went on the Decode heat map, and as the square furthest from
    /// x1 it took a highlight. A cell the client never finished says so.
    #[test]
    fn the_benchs_dead_server_and_timeout_verdicts_reach_the_page() {
        use super::numbers::{Metric, failed_note, timing};
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/metal/apple-m3-pro-2026-10-21T145041Z-a7820fbc.json"
        );
        let mut json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        json["config"]["failures_counted"] = true.into();
        let cells = json["models"][0]["scaling"].as_array_mut().unwrap();
        // As the bench writes them: a dead server's cell, and one that timed out.
        cells[0]["dead_server"] = true.into();
        cells[0]["failed"] = cells[0]["completed"].clone();
        cells[0]["completed"] = 0.into();
        cells[1] = serde_json::json!({"axis": "grid", "rung": cells[1]["rung"].clone(),
            "conc": cells[1]["conc"].clone(), "completed": 0, "failed": 12, "timed_out": true});
        let mut run: Run = serde_json::from_value(json).unwrap();
        run.mark_dead();
        let scaling = run.models[0].scaling.as_deref().unwrap();
        let (dead, slow, live) = (&scaling[0], &scaling[1], &scaling[2]);
        assert!(dead.dead, "the bench's dead-server verdict is taken");
        assert_eq!(timing(Some(dead), Metric::Throughput), None);
        assert!(failed_note(Some(dead)).contains("stopped answering"));
        assert!(failed_note(Some(slow)).contains("did not finish"));
        assert!(!live.dead && timing(Some(live), Metric::Throughput).is_some());
    }

    /// A run whose bench counted aborted and errored requests itself (#510) keeps what it
    /// recorded: its cells are never re-judged by their answer tokens, however few.
    #[test]
    fn a_run_that_counted_its_failures_is_not_re_judged() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let file = root.join("tests/fixtures/metal/apple-m3-pro-2026-10-13T010519Z-cc51e0d8.json");
        let text = std::fs::read_to_string(&file).expect("the published M3 Pro run");
        let mut json: serde_json::Value = serde_json::from_str(&text).expect("json");
        let judged = |json: &serde_json::Value| {
            let mut run: Run = serde_json::from_value(json.clone()).expect("a run");
            run.mark_dead();
            let m = run
                .models
                .iter()
                .find(|m| m.stem == "gemma-4-26b-a4b-it")
                .expect("gemma");
            m.scaling.iter().flatten().filter(|c| c.dead).count()
        };
        assert_eq!(
            judged(&json),
            3,
            "an older run is judged by its answer tokens"
        );
        json["config"]["failures_counted"] = true.into();
        assert_eq!(judged(&json), 0, "a run that counted its failures is not");
    }

    /// A timeline steps through a machine's runs of one shape: scratchy at every run, the rival
    /// only at the run that measured it, so a quick run's carried oMLX is no point of its own.
    #[test]
    fn a_timeline_takes_the_rival_only_from_the_runs_that_measured_it() {
        use super::data::Run;
        use super::numbers::{Engine, Metric};
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/metal/apple-m5-max-2026-10-14T111352Z-987b06a2.json"
        );
        let full: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let quick = |when: &str| {
            let mut v = full.clone();
            v["generated_utc"] = when.into();
            for m in v["models"].as_array_mut().unwrap() {
                m.as_object_mut()
                    .unwrap()
                    .retain(|k, _| !(k.starts_with("scaling_") || k.starts_with("cache_ladder_")));
            }
            v
        };
        let mut runs: Vec<Run> = [
            quick("2026-10-13T12:00:00Z"),
            full.clone(),
            quick("2026-10-15T12:00:00Z"),
        ]
        .into_iter()
        .map(|v| serde_json::from_value(v).unwrap())
        .collect();
        super::borrow_earlier(&mut runs);
        let machines = super::index(&runs);
        let mr = &machines["Apple M5 Max"];
        let (stem, entries) = &mr.models[0];
        let kept: Vec<&super::Entry> = entries.iter().collect();
        assert_eq!(kept.len(), 3, "{stem}");
        let steps = super::timeline_steps(&kept, Some(Engine::OmlxTq), Metric::Throughput);
        assert!(
            steps.iter().all(|s| s.scratchy.is_some()),
            "scratchy at every run"
        );
        let rival: Vec<bool> = steps.iter().map(|s| s.rival.is_some()).collect();
        assert_eq!(
            rival,
            [false, true, false],
            "oMLX only where it was measured"
        );
        // Evenly spaced on the sparkline's log axis.
        let xs: Vec<f64> = steps.iter().map(|s| s.x).collect();
        assert_eq!(xs, [1.0, 2.0, 4.0]);
        // Where scratchy started is the first run with both engines, not the quick run before it.
        let mut steps = steps;
        super::timeline_highlights(&mut steps);
        let labelled: Vec<bool> = steps.iter().map(|s| s.highlight).collect();
        assert_eq!(labelled, [false, true, true]);
        assert_eq!(super::model_timelines(entries).len(), 2);
    }

    /// A published run with one model's memory readings set to `rss` (every launch's peak RSS)
    /// and `mem` (every scaling cell's peak memory), in MiB, for the rule tests below: the run's
    /// other fields stay real, so it loads as one does. Returns the run and the model's index.
    fn with_memory(rss: f64, mem: f64) -> (Run, usize) {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let file = root.join("tests/fixtures/metal/apple-m3-pro-2026-10-21T145041Z-a7820fbc.json");
        let mut json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&file).expect("a published run"))
                .expect("json");
        let at = (json["models"].as_array().expect("models").iter())
            .position(|m| m["model_id"] == "mlx-community/Qwen3.6-35B-A3B-4bit")
            .expect("qwen3.6-35b-a3b in the run");
        let m = &mut json["models"][at];
        for key in ["cache_ladder", "cache_ladder_omlx"] {
            for rep in m[key].as_array_mut().into_iter().flatten() {
                rep["peak_rss_mib"] = rss.into();
            }
        }
        for key in ["scaling", "scaling_omlx"] {
            for c in m[key].as_array_mut().into_iter().flatten() {
                c["server_mem_peak_mib"] = mem.into();
            }
        }
        let mut run: Run = serde_json::from_value(json).expect("a run");
        run.mark_mem_below_weights();
        (run, at)
    }

    /// A memory reading below the model's own weights is left out, and one at or above them is
    /// kept: the comparison and its MiB-to-bytes conversion, on either side of the line (#414).
    #[test]
    fn a_memory_reading_below_the_weights_is_left_out_and_one_above_is_kept() {
        use super::Scenario;
        use super::numbers::{Engine, Memory, Metric, peak_rss};
        let w = {
            let (run, at) = with_memory(0.0, 0.0);
            run.models[at].weights().expect("qwen's weights").mib()
        };
        for (mib, below) in [(w - 1.0, true), (w + 1.0, false)] {
            let (run, at) = with_memory(mib, mib);
            let m = &run.models[at];
            let rss = peak_rss(m, Engine::Scratchy, Scenario::Cold);
            assert_eq!(
                rss == Memory::BelowWeights,
                below,
                "peak RSS {mib} MiB, weights {w}: {rss:?}"
            );
            for c in m.scaling.iter().flatten() {
                let shown = super::numbers::timing(Some(c), Metric::PeakMem);
                assert_eq!(
                    shown.is_none(),
                    below,
                    "peak memory {mib} MiB, weights {w}: {shown:?}"
                );
            }
        }
    }

    /// Only an engine that serves the MLX checkpoint is judged against its weights: ollama,
    /// llama.cpp and mistral.rs serve their own builds, so a low reading of theirs is still one.
    #[test]
    fn only_an_engine_serving_the_mlx_checkpoint_is_judged_by_its_weights() {
        use super::numbers::{Engine, below_weights};
        let (run, at) = with_memory(0.0, 0.0);
        let m = &run.models[at];
        let low = m.weights().expect("qwen's weights").mib() - 1.0;
        for e in Engine::ALL {
            assert_eq!(
                below_weights(m, e, low),
                e.serves_mlx_checkpoint(),
                "{}",
                e.label()
            );
        }
    }
}

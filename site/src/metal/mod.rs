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
    Alignment, ContentPanel, ContentSwitcher, ContentSwitcherItem, Fold, Heading, Layer, Section,
    Stack, Table, TableCell, TableRow, Tag, TagKind, Tile, Toggletip,
};
use crate::charts::{
    Axes, Axis, AxisDomain, BarPoint, Better, Chart, ChartOptions, ColorLegend, Colour, Enabled,
    Gradient, HeatPoint, HeatmapOptions, Locale, RATIO_RANGE, RATIO_STEPS, RatioSquare, ScaleFit,
    SparkAxes, Step, TableAxes, chart, colour, glance_maps, glance_mini, highlight, log2_ratio,
    mini_heatmap, ratio_scale, ratio_step, sparkline,
};
use crate::chrome::{self, Content, Head, Library, Published, REPO, Root, Tab};
use data::{At, Bytes, Cell, Model, Run, Scenario};
use numbers::{
    BELOW_WEIGHTS, DASH, Engine, FAILED, Metric, NO_STREAM, NOISE, Startup, cfmt, failed_note, fmt,
    lfmt, marks, marks_note, med, peak_rss, rss_fmt, served, timing,
};

/// The run files, relative to site/; copied to the same place under _site/,
/// where each run table links to its own.
pub const DATA: &str = "data/metal";
/// Runs only the tests read: earlier rounds the page no longer shows, kept for the cases they
/// hold (a dead server's cells, a run before #510 counted failures).
#[cfg(test)]
const TESTDATA: &str = "testdata/metal";

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
        run.mark_dead();
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
                       memory), so every number in that cell covers only the rest. mlx-lm and ollama can stop \
                       answers early, which flatters tok/s."
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
            Trend::UsersDecode | Trend::UsersPrefill => {
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

/// Prefill first, then decode, as in the overview.
const GRID_METRICS: [(Metric, &str); 2] = [
    (Metric::Ttft, "time to first token"),
    (Metric::Throughput, "throughput"),
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
        rsx! {
            Stack { gap: 3,
                strong { "{a}-token answers" }
                p { "{note}." }
                div { style: "display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: flex-start",
                    for (metric, title) in GRID_METRICS { {heat_map(g, metric, title, &format!("scale-{mid}-{a}"))} }
                    if g.rival.is_some() { {ratio_scale(ScaleFit::Fixed, &format!("scale-{mid}-{a}"))} }
                }
                if !others.is_empty() {
                    for (metric, title) in GRID_METRICS {
                        Stack { gap: 1,
                            span { "{title}, vs every engine" }
                            div { style: "display: flex; flex-wrap: wrap; gap: 1rem",
                                for e in &others { {versus_mini(g, metric, *e, &format!("{title}, scratchy vs {}, {a}-token answers", e.label()))} }
                            }
                        }
                    }
                }
                Fold { title: "Table view, {a}-token answers",
                    Stack { gap: 4,
                        for (metric, title) in GRID_METRICS { {heat_table(g, metric, title)} }
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
fn over_time_charts(machines: &BTreeMap<&str, MachineRuns>) -> Vec<(String, Element)> {
    let classes = lineup(machines).map_or(&[][..], |l| l.classes);
    let current = |stem: &str| classes.iter().any(|c| c.small == stem || c.large == stem);
    let mut charts = Vec::new();
    for chip in in_order(machines) {
        for (stem, entries) in machines[chip].models.iter().filter(|(s, _)| current(s)) {
            for (value, chart) in model_timelines(entries) {
                charts.push((format!("{chip} · {stem} · {value}"), chart));
            }
        }
    }
    charts
}

/// The Over time section: every machine's models over their runs, a small timeline a machine,
/// model and metric, as the spyre page shows its cells. Nothing until some model has two runs of
/// one shape on one machine.
fn over_time(charts: Vec<(String, Element)>) -> Element {
    if charts.is_empty() {
        return rsx! {};
    }
    let grid = "display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); \
                column-gap: 1.5rem; row-gap: 1.25rem; align-items: start";
    rsx! {
        Section { id: "over-time", level: 2,
            Stack { gap: 4,
                Heading { "Over time" }
                p {
                    "Each model's one-user throughput and time to first token at its base cell, a \
                     point a run on its Mac, oldest first, scratchy against the first engine any \
                     run there measured. A run of scratchy alone (a quick run) takes the other \
                     engine's line dotted, carried from the last run that measured it, and the fill \
                     between the lines follows it. The fill is labelled with its ×-factor at the \
                     first run and the newest: where scratchy started, and where it stands now."
                }
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

/// One overview heat map's ratios, at its run's [`glance_answer`]: its prompt lengths, the
/// rival, and each square's (users, prompt, how many times faster scratchy
/// is).
struct GlanceGrid {
    prompts: Vec<u32>,
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
            prompts: prompts.to_vec(),
            rival,
            cells,
        })
    }
}

/// The overview's shape: a column per class of the lineup (`config.classes`), each Mac on the
/// class's small model or, from `large_from_gb`, its large one, and each map a users count by a
/// prompt length (`grid_conc`, `grid_input`), one shape for every model on every machine: users at
/// once down the side, prompt tokens across. A square its run did not measure (a run of another
/// shape, or a prompt the Mac's memory did not hold) is blank.
struct Lineup<'a> {
    classes: &'a [data::Class],
    large_from_gb: u32,
    users: Vec<u32>,
    prompts: Vec<u32>,
}

impl Lineup<'_> {
    fn model(&self, class: &str, memory_gb: u32) -> Option<&str> {
        (self.classes.iter())
            .find(|c| c.class == class)
            .map(|c| c.model(memory_gb, self.large_from_gb))
    }
}

/// The overview's [`Lineup`]: the classes and cut-off of the newest run that records a lineup, and
/// the grid of the newest such run that has one, so a `--no-scaling` run (no grid) after it does
/// not empty every map. None before any run records a lineup.
fn lineup<'a>(machines: &BTreeMap<&str, MachineRuns<'a>>) -> Option<Lineup<'a>> {
    let mut with_lineup: Vec<&Run> = (machines.values())
        .flat_map(|mr| mr.runs.iter().copied())
        .filter(|r| r.config.classes.is_some())
        .collect();
    with_lineup.sort_by(|a, b| b.generated_utc.cmp(&a.generated_utc));
    let newest = with_lineup.first()?;
    let grid = (with_lineup.iter())
        .find_map(|r| {
            let sc = r.config.scaling.as_ref()?;
            Some((sc.grid_conc.clone()?, sc.grid_input.clone()?))
        })
        .unwrap_or_default();
    Some(Lineup {
        classes: newest.config.classes.as_deref()?,
        // Runs from before `large_from_gb` split the sets at 64 GB.
        large_from_gb: newest.config.large_from_gb.unwrap_or(64),
        users: grid.0,
        prompts: grid.1,
    })
}

/// Every model's users-by-prompt grid on every machine as small multiples,
/// one table per metric: a machine per row, a model per column, each cell that
/// model's grid as a tiny heatmap coloured like the full ones.
fn glance(machines: &BTreeMap<&str, MachineRuns>) -> Element {
    // A column per architecture class ([`Lineup`]); each machine's row shows the class's model
    // for its memory, the small set below the lineup's `large_from_gb` and the large one from it,
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
    let metrics = [(Metric::Ttft, "Prefill"), (Metric::Throughput, "Decode")];
    let users: Vec<u32> = lineup.users.clone();
    // The answer lengths the maps are at: one, unless machines are on runs of different shapes.
    let answers: BTreeSet<u32> = (shown.iter())
        .filter_map(|&(chip, stem)| machines[chip].latest(stem))
        .map(|Entry { run, .. }| glance_answer(run))
        .collect();
    let answers = (answers.iter())
        .map(|a| a.to_string())
        .collect::<Vec<_>>()
        .join("- and ");
    // Every grid's ratios first, keyed by machine, model and metric, or why
    // the grid has none: the highlights are picked from all of them.
    let mut grids: BTreeMap<(&str, &str, &str), Result<GlanceGrid, &str>> = BTreeMap::new();
    for &(chip, stem) in &shown {
        for (metric, name) in metrics {
            grids.insert(
                (chip, stem, name),
                GlanceGrid::of(
                    &machines[chip],
                    stem,
                    metric,
                    &lineup.users,
                    &lineup.prompts,
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
        trend_spark(at, &sparks[&at], "4.5rem")
    };
    // Every prompt length any shown grid has: a grid with none of its own
    // (the model not run on that machine, say) keeps its place as a map of
    // blank squares this wide, over why it has nothing to show.
    let prompts: Vec<u32> = lineup.prompts.clone();
    let mini = |chip: &str, stem: &str, name: &str| -> Element {
        let (ins, cells, total_label, none) = match &grids[&(chip, stem, name)] {
            Ok(g) => (
                &g.prompts,
                g.cells.clone(),
                format!("scratchy/{}", g.rival.label()),
                "no ratios",
            ),
            Err(why) => (
                &prompts,
                users
                    .iter()
                    .flat_map(|&u| prompts.iter().map(move |&i| (u, i, None)))
                    .collect(),
                "scratchy".to_string(),
                *why,
            ),
        };
        let squares = cells
            .into_iter()
            .map(|(u, i, r)| RatioSquare {
                across: i.to_string(),
                down: u.to_string(),
                ratio: r,
                highlight: highlights.contains(&Square {
                    chip,
                    stem,
                    metric: name,
                    users: u,
                    prompt: i,
                }),
            })
            .collect();
        glance_mini(
            Slice::Answer(0).axes(&users, ins),
            squares,
            total_label,
            name,
            none,
            &format!("{stem} on {chip}"),
        )
    };
    let chips: Vec<&str> = in_order(machines);
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
                                        "Every model's users × prompt size grid on every machine, at {answers}-token answers: in \
                                         each cell, the model's prefill (time to first token) grid, then its decode (throughput) grid \
                                         (rows: users at once, fewest first; columns: \
                                         prompt, short to long). A blank square has no ratio: scratchy or the engine it is compared with \
                                         was not measured there, or never streamed an answer (the model's own grids below say which); runs \
                                         from before the grid had its own users counts measured it at the base count only. \
                                         Colour is how many times faster scratchy is than the engine named under the \
                                         grid: oMLX TurboQuant where the run has it, else the next of oMLX, mlx-lm, vllm-metal, ollama, \
                                         llama.cpp, mistral.rs. Blue faster, red slower, the middle step within noise. \
                                         Hover a square for its ratio."
                                    }
                                    {glance_maps(&chips, &classes, |chip, class| {
                                        let Some(stem) = model_of(chip, class) else { return rsx! {} };
                                        rsx! {
                                            Stack { gap: 1,
                                                div { style: "display: flex; gap: 0.5rem; align-items: flex-start",
                                                    for (_, name) in metrics { {mini(chip, stem, name)} }
                                                }
                                                div { style: "text-align: center; font-size: 0.75rem", "{stem}" }
                                            }
                                        }
                                    })}
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
                                            p { "scratchy against oMLX TurboQuant (or the next engine a run has, in the order of the heat maps) along {along}. Blue fill where scratchy is ahead, red where it is behind." }
                                            div { style: spark_grid(stems.len()),
                                                div {}
                                                div {}
                                                for stem in &stems { div { style: "text-align: center", "{stem}" } }
                                                for chip in &chips {
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
                div {}
                div {}
                for col in &STARTUP_COLUMNS { div { style: "text-align: center", "{col.title}" } }
                for chip in in_order(machines) {
                    // Column 1 starts a row, past any columns this panel leaves empty.
                    div { style: "grid-column: 1; align-self: start", "{chip}" }
                    div {}
                    for (k, col) in STARTUP_COLUMNS.iter().enumerate() {
                        div {
                            match &sparks[&(chip, k)] {
                                Ok(sp) => sparkline(&sp.steps, col.axes, Engine::Scratchy.side(), sp.rival.side(), Some(sp.names()), "4.5rem", &format!("{chip}, {}: {} by model size, scratchy against {}", col.title, col.axes.value, sp.rival.label())),
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

/// The glance's first panel's name.
const HEAT_MAPS: &str = "Heat maps";

/// The sparkline panels: a dimension, what runs along it, and its prefill and
/// decode trends.
const SCALING: [(&str, &str, [Trend; 2]); 2] = [
    (
        "Concurrency scaling",
        "users at once, at the grid's one-user summary cell (1024-token prompts, and the answer length the heat maps use); runs from before the grid had its own users (#349) at 1, 4 and 16 users, 512-token prompts and 128-token answers",
        [Trend::UsersPrefill, Trend::UsersDecode],
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
        // No link to a section that is not there.
        let timelines = over_time_charts(&machines);
        if !timelines.is_empty() {
            sections.push(("over-time".to_string(), "Over time".to_string()));
        }
        let order = in_order(&machines);
        sections.extend(order.iter().map(|chip| (slug(chip), chip.to_string())));
        rsx! {
            {glance(&machines)}
            {over_time(timelines)}
            for chip in &order { {machine_section(chip, &machines[chip])} }
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
        script { dangerous_inner_html: crate::charts::CHARTS_JS }
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
    use super::{DATA, Run, TESTDATA, size_name, slug};

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
            "/data/metal/apple-m5-max-2026-10-14T111352Z-987b06a2.json"
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
            "/data/metal/apple-m5-max-2026-10-14T111352Z-987b06a2.json"
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
            "/data/metal/apple-m3-pro-2026-10-21T145041Z-a7820fbc.json"
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
            "/data/metal/apple-m5-max-2026-10-14T111352Z-987b06a2.json"
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

    /// Machines read up from the smallest: a 32 GB base M5 before the 36 GB M3 Pro, the 64 GB
    /// M4 Max and M5 Max after. Below 64 GB a Mac's overview row is the small set, at 64 GB the
    /// large one.
    #[test]
    fn machines_order_by_memory_and_take_their_set_by_it() {
        use super::data::Run;
        let read = |name: &str| -> serde_json::Value {
            let p = format!("{}/data/metal/{name}", env!("CARGO_MANIFEST_DIR"));
            serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
        };
        let mut base = read("apple-m3-pro-2026-10-21T145041Z-a7820fbc.json");
        base["machine"]["chip"] = "Apple M5".into();
        base["machine"]["memory_gb"] = 32.into();
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
        // The overview's columns are the full run's recorded lineup, each Mac on its set.
        let lineup = super::lineup(&machines).expect("the M3 Pro's run records its lineup");
        assert_eq!(lineup.users, [1, 4, 8, 16]);
        assert_eq!(lineup.prompts, [128, 1024, 4096, 32768]);
        let dense = lineup.classes.iter().find(|c| c.class == "dense").unwrap();
        assert_eq!(dense.model(32, lineup.large_from_gb), "granite-4.2-8b");
        assert_eq!(dense.model(36, lineup.large_from_gb), "granite-4.2-8b");
        assert_eq!(dense.model(64, lineup.large_from_gb), "granite-4.2-30b");
    }

    /// The published runs' cells a dead server measured (#413) are marked, and no other: the
    /// glm-4.5-air and qwen3.6-35b-a3b cells of the M1 Max run in #472, and the M3 Pro's gemma
    /// 4096-token cells from the one its OOM struck in (#412), 4096x16, which still streamed a
    /// few tokens, on. Every engine that stops at EOS early (ollama, oMLX) keeps its numbers, and
    /// so does scratchy's f378699e gemma cell at 16 users, live at 99% of its tokens. A marked
    /// cell counts every request failed, says why, and shows no number.
    #[test]
    fn the_published_cells_of_a_dead_server_and_only_those_are_marked() {
        use super::numbers::{DASH, Engine, FAILED, Metric, cfmt, failed, failed_note, partial};
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut runs: Vec<Run> = Vec::new();
        for dir in [DATA, TESTDATA] {
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
            ("apple-m1-max", "c8860b1d", "glm-4.5-air", "scratchy", 22),
            (
                "apple-m1-max",
                "c8860b1d",
                "qwen3.6-35b-a3b",
                "scratchy",
                10,
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

    /// A sparkline step compares the two engines on the rungs both measured. The M1 Max run in
    /// #472 has glm-4.5-air's scratchy cells dead at 1024 and 4096 prompt tokens with 16-token
    /// answers: averaged over each engine's own cells, scratchy read 1.86x oMLX TurboQuant's
    /// throughput; on the one rung both have, it is 0.95x.
    #[test]
    fn a_sparkline_step_averages_only_the_rungs_both_engines_measured() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let file = root
            .join(TESTDATA)
            .join("apple-m1-max-2026-10-14T191550Z-c8860b1d.json");
        let mut run: Run =
            serde_json::from_str(&std::fs::read_to_string(&file).expect("the M1 Max run"))
                .expect("a run");
        run.mark_dead();
        let m = run
            .models
            .iter()
            .find(|m| m.stem == "glm-4.5-air")
            .expect("glm-4.5-air");
        let (rival, steps) = super::Trend::AnswerDecode
            .points(m, &run)
            .expect("a sparkline");
        assert_eq!(rival.key(), "omlx_tq");
        let step = steps
            .iter()
            .find(|s| s.x == 16.0)
            .expect("the 16-token step");
        let ratio = step.scratchy.expect("scratchy") / step.rival.expect("the rival");
        assert!(
            (ratio - 0.95).abs() < 0.01,
            "scratchy against oMLX TurboQuant: {ratio:.3}x"
        );
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
            "/data/metal/apple-m3-pro-2026-10-21T145041Z-a7820fbc.json"
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
        let file = root
            .join(TESTDATA)
            .join("apple-m3-pro-2026-10-13T010519Z-cc51e0d8.json");
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
            "/data/metal/apple-m5-max-2026-10-14T111352Z-987b06a2.json"
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
        let file = root
            .join(DATA)
            .join("apple-m3-pro-2026-10-21T145041Z-a7820fbc.json");
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

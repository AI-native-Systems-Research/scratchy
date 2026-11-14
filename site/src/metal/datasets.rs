//! The page by dataset: which datasets the shown runs measured, and one section a dataset.

use std::collections::{BTreeMap, BTreeSet};

use dioxus::prelude::*;

use crate::carbon::{
    Alignment, ContentPanel, Definition, Heading, Section, Stack, TabItem, Tabs, TabsKind, Tile,
    TipHeading,
};
use crate::charts::{
    RATIO_STEPS, STRIPED, ScaleFit, ratio_ink, ratio_labels, ratio_scale, ratio_step,
};

use super::buckets::{self, Buckets};
use super::data::{At, Model, RequestTiming, Run, TaskCell};
use super::numbers::{Engine, served};
use super::{
    ECDF_COLUMN, Entry, HEAT_LABELS, MachineRuns, Series, Sweep, bar_chart, ecdf_chart, engine_key,
    grid_glance, grid_maps, helper, helper_info, in_order, kept, lineup, marks_key,
    over_time_charts, parity, parity_info, previous_run, request_plots, runs_table, served_as,
    slug, sweep_part, tasks, users_at,
};

/// A summary square's side: the grid's, so one dataset's map is as small as the grid's are.
const SQUARE: &str = "2.25rem";

/// The random-token grid, read as one more dataset: its rows are the grid's prompt lengths.
pub(super) const RANDOM: &str = "random";

/// What the page knows of a dataset, one row each in [`DATASETS`].
pub(super) struct Dataset {
    /// Its key: its workload in the run file (`--dataset-name`), or [`RANDOM`].
    pub key: &'static str,
    /// Its name on the page.
    pub label: &'static str,
    /// What it measures.
    pub about: &'static str,
    /// The workload of its scored cell, a row of its table and not a section of its own:
    /// MuSiQue's, over more tasks for its accuracy alone.
    pub scored: Option<&'static str>,
    /// Whether its tasks are MuSiQue's, drawn as the run's `task_sample` says: a change of
    /// sampling is a change of its questions, and of no other dataset's.
    pub sampled: bool,
    /// The turns of each of its tasks: a conversation's, 1 for a single question.
    pub turns: u32,
    /// Its summary map's prompt-length columns.
    pub buckets: Buckets,
}

/// A dataset of nothing but its name: one question a task, its prompts in one bucket.
const fn plain(key: &'static str, label: &'static str, about: &'static str) -> Dataset {
    Dataset {
        key,
        label,
        about,
        scored: None,
        sampled: false,
        turns: 1,
        buckets: buckets::WHOLE,
    }
}

/// Every dataset the page knows, in its order: the quick run's workloads first, the random-token
/// grid last.
const DATASETS: [Dataset; 9] = [
    plain(
        "sharegpt",
        "ShareGPT chat",
        "real chat prompts, one turn each",
    ),
    Dataset {
        scored: Some("rag-accuracy"),
        sampled: true,
        buckets: buckets::MUSIQUE,
        ..plain(
            "rag",
            "MuSiQue",
            "multi-document QA, MuSiQue: a question over its passages",
        )
    },
    Dataset {
        sampled: true,
        buckets: buckets::MUSIQUE,
        ..plain(
            "rag-spans",
            "RAG, passage reuse",
            "the RAG tasks through scratchy's span cache",
        )
    },
    Dataset {
        turns: 4,
        ..plain(
            "sharegpt-multi-turn",
            "ShareGPT conversations",
            "each turn with the turns before it",
        )
    },
    plain(
        "longbench",
        "LongBench",
        "LongBench v2 short: long documents",
    ),
    plain(
        "shared-prefix",
        "Shared prompt",
        "short questions sharing one long system prompt",
    ),
    plain(
        "shared-prefix-control",
        "Shared prompt, no reuse",
        "the same, every prompt its own",
    ),
    Dataset {
        sampled: true,
        buckets: buckets::MUSIQUE,
        ..plain(
            "rag-accuracy",
            "MuSiQue, scored",
            "RAG over more tasks, for its accuracy alone: 100 on a quick run, 400 or the whole dev \
             set on a full one",
        )
    },
    plain(
        RANDOM,
        "Random tokens",
        "random-token prompts at fixed lengths, the matrix's diagnostic grid (--random)",
    ),
];

/// A dataset's row; None for one the page does not know yet.
pub(super) fn of(key: &str) -> Option<&'static Dataset> {
    DATASETS.iter().find(|d| d.key == key)
}

/// A dataset's place in the page's order; one the page does not know goes last.
pub(super) fn order(key: &str) -> usize {
    (DATASETS.iter().position(|d| d.key == key)).unwrap_or(DATASETS.len())
}

/// Whether `workload` is another dataset's scored cell ([`Dataset::scored`]).
pub(super) fn is_scored(workload: &str) -> bool {
    DATASETS.iter().any(|d| d.scored == Some(workload))
}

/// Whether `workload`'s tasks are drawn as the run's `task_sample` says ([`Dataset::sampled`]).
pub(super) fn sampled(workload: &str) -> bool {
    of(workload).is_some_and(|d| d.sampled)
}

/// Every dataset a shown model measured, in the page's order, then [`RANDOM`] if any shown model
/// measured the random-token grid. A model counts by every run of it on each machine, in `all`,
/// the runs a newer one replaced too: a run of one dataset leaves another's section standing.
pub(super) fn shown<'a>(
    machines: &BTreeMap<&str, MachineRuns<'_>>,
    all: &BTreeMap<&str, MachineRuns<'a>>,
) -> Vec<&'a str> {
    let every = || {
        (machines.iter())
            .filter_map(|(chip, mr)| Some((mr, all.get(chip)?)))
            .flat_map(|(mr, every)| {
                (every.models.iter())
                    .filter(|(stem, _)| mr.latest(stem).is_some())
                    .flat_map(|(_, es)| es.iter().copied())
            })
    };
    let mut shown: Vec<&'a str> = Vec::new();
    for e in every() {
        // A scored cell is its dataset's, a row of its detail table, not a section of its own.
        for w in tasks::workloads(e.m).into_iter().filter(|&w| !is_scored(w)) {
            if !shown.contains(&w) {
                shown.push(w);
            }
        }
    }
    if every().any(|e| ran_grid(e.m)) {
        shown.push(RANDOM);
    }
    shown.sort_by_key(|w| order(w));
    shown
}

/// A dataset's name on the page: its label, or for a workload the page does not know yet its own
/// name, so it keeps a link and section of its own.
pub(super) fn title(dataset: &str) -> String {
    of(dataset).map_or(dataset, |d| d.label).to_string()
}

/// Whether a model's run measured the random-token grid.
fn ran_grid(m: &Model) -> bool {
    (m.scaling.iter().flatten()).any(|c| matches!(c.at, At::Grid { .. }))
}

/// Whether a model's run measured `dataset`: the grid for [`RANDOM`], else scratchy's cells of it.
fn measures(dataset: &str, m: &Model) -> bool {
    match dataset {
        RANDOM => ran_grid(m),
        w => tasks::workloads(m).contains(&w),
    }
}

/// Each shown machine's runs as `dataset`'s section reads them: from `all`, the runs a newer one
/// replaced too, each shown model's entries only those that measured the dataset, so its newest
/// is the newest run that did. The machine's runs are the page's, and any older one a model's
/// numbers come from.
fn measured<'a>(
    dataset: &str,
    machines: &BTreeMap<&str, MachineRuns<'_>>,
    all: &BTreeMap<&'a str, MachineRuns<'a>>,
) -> BTreeMap<&'a str, MachineRuns<'a>> {
    (all.iter())
        .filter_map(|(&chip, every)| {
            let page = machines.get(chip)?;
            let models: Vec<(&'a str, Vec<Entry<'a>>)> = (every.models.iter())
                .filter(|(stem, _)| page.latest(stem).is_some())
                .map(|(stem, es)| {
                    let kept = es.iter().copied().filter(|e| measures(dataset, e.m));
                    (*stem, kept.collect())
                })
                .collect();
            let on_page = |r: &Run| page.runs.iter().any(|p| p.generated_utc == r.generated_utc);
            let drawn = |r: &Run| {
                let when = &r.generated_utc;
                (models.iter())
                    .any(|(_, es)| es.last().is_some_and(|e| &e.run.generated_utc == when))
            };
            let runs = (every.runs.iter().copied())
                .filter(|r| on_page(r) || drawn(r))
                .collect();
            Some((chip, MachineRuns { runs, models }))
        })
        .collect()
}

/// The machines that ran `dataset`, fewest GB first, each with every model it ran that measured
/// it, in the order it first ran them: every model, not only its lineup class's current one, so a
/// model an earlier round ran keeps its numbers on the page.
fn ran<'a>(
    dataset: &str,
    machines: &'a BTreeMap<&str, MachineRuns<'a>>,
) -> Vec<(&'a str, Vec<&'a str>)> {
    (in_order(machines).into_iter())
        .filter_map(|chip| {
            let mr = &machines[chip];
            let stems: Vec<&str> = (mr.models.iter())
                .filter(|(stem, _)| mr.latest(stem).is_some_and(|e| measures(dataset, e.m)))
                .map(|(stem, _)| *stem)
                .collect();
            (!stems.is_empty()).then_some((chip, stems))
        })
        .collect()
}

/// One dataset's section: its summary map, its timelines where two or more runs measured it, then
/// one tab strip of machine and model, each pair its latency distributions, what its numbers came
/// from, its numbers and its users curves. The random-token grid's is laid out the same way, from
/// its own cells: its summary map, timelines, request plots and grid maps. Each pair is its
/// model's newest run that measured the dataset ([`measured`]). Nothing when no shown machine ran
/// it.
pub(super) fn section(
    dataset: &str,
    machines: &BTreeMap<&str, MachineRuns>,
    all: &BTreeMap<&str, MachineRuns>,
) -> Element {
    let view = measured(dataset, machines, all);
    let machines = &view;
    let ran = ran(dataset, machines);
    if ran.is_empty() {
        return rsx! {};
    }
    let id = slug(dataset);
    let random = dataset == RANDOM;
    let name = title(dataset);
    let about = of(dataset).map_or("", |d| d.about);
    let timelines: Vec<(&str, Vec<(String, Element)>)> = (ran.iter())
        .map(|&(chip, _)| match random {
            true => (chip, over_time_charts(all, chip)),
            false => (chip, dataset_timelines(dataset, all, chip, &ran)),
        })
        .filter(|(_, cs)| !cs.is_empty())
        .collect();
    let summary = match random {
        true => grid_glance(&id, machines),
        false => dataset_summary(dataset, &id, &ran, machines),
    };
    rsx! {
        Section { id: "{id}", level: 2,
            Stack { gap: 5,
                Stack { gap: 2,
                    Heading { "{name}" }
                    p { "{about}" }
                }
                {tabbed(&id, &Pairs::of(&ran, machines), summary, over_time_panel(&timelines),
                    |chip, stem| match random {
                        true => grid_latency(&id, &machines[chip], stem, chip),
                        false => pair_latency(dataset, &machines[chip], stem, chip),
                    },
                    |chip, stem| match random {
                        true => grid_pair(&id, &machines[chip], stem, chip),
                        false => pair_detail(dataset, &machines[chip], stem, chip),
                    })}
            }
        }
    }
}

/// A dataset's Summary tab: what its map shows, the map, then its accuracy where it is scored.
fn dataset_summary(
    dataset: &str,
    id: &str,
    ran: &[(&str, Vec<&str>)],
    machines: &BTreeMap<&str, MachineRuns>,
) -> Element {
    // The summary's columns: the lineup's classes, where the runs record one; else every model
    // shown, a column each.
    let summary_columns = summary_grid(ran, machines);
    // Each machine's models as the section reads them, for the accuracy table: in page order.
    let scored_pairs: Vec<(&str, &Model)> = (ran.iter())
        .flat_map(|(chip, stems)| {
            (stems.iter()).filter_map(move |stem| Some((*chip, machines[chip].latest(stem)?.m)))
        })
        .collect();
    rsx! {
        Stack { gap: 4,
            p { style: "contain: inline-size",
                "Per machine, its rows: prefill (P) and decode (D) with 1 user, then end-to-end time \
                 with each users count above it (2u, 4u). A column per model, under its family; the \
                 same tasks to every engine. Colour is how many times faster scratchy is than the \
                 engine it is set against, its median over the cell's requests against the rival's: \
                 blue faster, red slower, grey within noise. A striped square has no ratio: one engine \
                 did not run it, or most of its requests failed. Hover or focus a square for its \
                 ratio, the engine, and a borrowed rival's run."
            }
            {summary_map(dataset, id, ran, machines, &summary_columns)}
            {tasks::accuracy_table(dataset, &scored_pairs)}
        }
    }
}

/// The summary map's columns: the lineup's classes, each machine's model in a class its tile, when
/// the runs record a lineup and it covers every model shown; else one column a model shown.
struct SummaryGrid<'a> {
    columns: Vec<&'a str>,
    cells: Vec<(&'a str, &'a str, &'a str)>,
}

impl<'a> SummaryGrid<'a> {
    fn stem(&self, chip: &str, column: &str) -> Option<&'a str> {
        (self.cells.iter())
            .find(|&&(c, k, _)| c == chip && k == column)
            .map(|&(_, _, s)| s)
    }

    /// The models a class column shows, once each, a line each where Macs run different ones.
    /// Empty where the column is a model itself.
    fn models_of(&self, column: &str) -> Vec<&'a str> {
        let mut names: Vec<&'a str> = Vec::new();
        for &(_, k, stem) in &self.cells {
            if k == column && stem != column && !names.contains(&stem) {
                names.push(stem);
            }
        }
        names
    }
}

fn summary_grid<'a>(
    ran: &[(&'a str, Vec<&'a str>)],
    machines: &'a BTreeMap<&str, MachineRuns<'a>>,
) -> SummaryGrid<'a> {
    let lineup = lineup(machines);
    let classes = lineup.as_ref().map(|l| l.classes).unwrap_or_default();
    let class_of = |stem: &str| {
        (classes.iter())
            .find(|c| c.small == stem || c.large == stem)
            .map(|c| c.class.as_str())
    };
    // The model a Mac shows in a class, as the random-token map has it ([`Lineup::model`]): one,
    // the small set's where every Mac runs it, though a Mac that ran both sets measured both.
    let shows = |chip: &str, stem: &str| {
        let memory = machines[chip].runs[machines[chip].runs.len() - 1]
            .machine
            .memory_gb;
        match (lineup.as_ref(), class_of(stem)) {
            (Some(l), Some(class)) => l.model(class, memory) == Some(stem),
            _ => true,
        }
    };
    let all_classed = ran
        .iter()
        .all(|(_, stems)| stems.iter().all(|s| class_of(s).is_some()));
    let mut columns: Vec<&str> = Vec::new();
    let mut cells = Vec::new();
    for (chip, stems) in ran {
        for &stem in stems.iter().filter(|s| !all_classed || shows(chip, s)) {
            let column = match all_classed {
                true => class_of(stem).unwrap_or(stem),
                false => stem,
            };
            if !columns.contains(&column) {
                columns.push(column);
            }
            cells.push((*chip, column, stem));
        }
    }
    SummaryGrid { columns, cells }
}

/// A machine's dataset timelines, one a model it shows that two or more runs measured, titled by
/// the model.
fn dataset_timelines(
    dataset: &str,
    all: &BTreeMap<&str, MachineRuns>,
    chip: &str,
    ran: &[(&str, Vec<&str>)],
) -> Vec<(String, Element)> {
    let Some(mr) = all.get(chip) else {
        return Vec::new();
    };
    let shown = (ran.iter())
        .find(|(c, _)| *c == chip)
        .map_or(&[][..], |(_, s)| s.as_slice());
    (mr.models.iter())
        .filter(|(stem, _)| shown.contains(stem))
        .filter_map(|(stem, entries)| {
            Some((stem.to_string(), super::task_timeline(entries, dataset)?))
        })
        .collect()
}

/// The Over time tab: a heading a machine, its charts under it, each titled; a line saying why
/// when no model has two runs here yet.
fn over_time_panel(timelines: &[(&str, Vec<(String, Element)>)]) -> Element {
    if timelines.is_empty() {
        return rsx! { p { "A timeline starts with a model's second run of this on a machine." } };
    }
    rsx! {
        Stack { gap: 4,
            for (chip, charts) in timelines {
                Stack { gap: 2,
                    strong { "{chip}" }
                    div { style: "display: flex; flex-wrap: wrap; gap: 1rem 1.5rem",
                        for (title, chart) in charts {
                            div { style: "flex: 1 1 14rem; min-width: 14rem; max-width: 22rem",
                                Stack { gap: 1, span { style: "font-size: 0.75rem", "{title}" } {chart} }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Every machine of the page and every model class of its lineup, and the model each machine runs
/// in each class, where it ran this dataset: so a section's buttons are every machine and class,
/// as main's latency tab has them, and a pair that ran nothing yet says so. Without a lineup, every
/// model a machine ran is a class of its own.
pub(super) struct Pairs<'a> {
    pub(super) chips: Vec<&'a str>,
    pub(super) classes: Vec<&'a str>,
    stems: BTreeMap<(&'a str, &'a str), &'a str>,
    /// Each lineup class's models, small then large, for a button no machine ran a model of.
    models: BTreeMap<&'a str, String>,
}

impl<'a> Pairs<'a> {
    fn of(ran: &[(&'a str, Vec<&'a str>)], machines: &'a BTreeMap<&str, MachineRuns<'a>>) -> Self {
        let chips = in_order(machines);
        let mut stems = BTreeMap::new();
        let mut models = BTreeMap::new();
        let classes: Vec<&str> = match lineup(machines) {
            Some(l) => {
                for c in l.classes {
                    let name = match c.small == c.large {
                        true => c.small.clone(),
                        false => format!("{} / {}", c.small, c.large),
                    };
                    models.insert(c.class.as_str(), name);
                }
                let mut classes: Vec<&str> = l.classes.iter().map(|c| c.class.as_str()).collect();
                let lineup_classes = classes.len();
                for &(chip, ref ran_stems) in ran {
                    let memory = machines[chip].runs[machines[chip].runs.len() - 1]
                        .machine
                        .memory_gb;
                    for &class in &classes[..lineup_classes] {
                        // The run's own name for the class's model: the lineup's is borrowed from it.
                        let wanted = l.model(class, memory);
                        if let Some(&stem) = ran_stems.iter().find(|&&s| Some(s) == wanted) {
                            stems.insert((chip, class), stem);
                        }
                    }
                }
                // A model a machine ran that is not its class's current one (an earlier round's,
                // or the other set's) keeps a button of its own, so its numbers stay on the page.
                for &(chip, ref ran_stems) in ran {
                    for &stem in ran_stems {
                        let shown = stems.iter().any(|(&(c, _), &s)| c == chip && s == stem);
                        if !shown {
                            if !classes.contains(&stem) {
                                classes.push(stem);
                            }
                            stems.insert((chip, stem), stem);
                        }
                    }
                }
                classes
            }
            None => {
                let mut classes = Vec::new();
                for &(chip, ref ran_stems) in ran {
                    for &stem in ran_stems {
                        if !classes.contains(&stem) {
                            classes.push(stem);
                        }
                        stems.insert((chip, stem), stem);
                    }
                }
                classes
            }
        };
        Pairs {
            chips,
            classes,
            stems,
            models,
        }
    }

    fn stem(&self, chip: &str, class: &str) -> Option<&'a str> {
        self.stems.get(&(chip, class)).copied()
    }

    /// A class's button: the model it shows, not the class, or each model where machines run
    /// different ones in it ("gpt-oss-20b / gpt-oss-120b-3bit"), as the summary map's columns
    /// read; the class itself where no machine ran one.
    fn label(&self, class: &str) -> String {
        let mut names: Vec<&str> = Vec::new();
        for &chip in &self.chips {
            if let Some(stem) = self.stem(chip, class).filter(|s| !names.contains(s)) {
                names.push(stem);
            }
        }
        match names.is_empty() {
            true => self
                .models
                .get(class)
                .cloned()
                .unwrap_or_else(|| class.to_string()),
            false => names.join(" / "),
        }
    }

    /// The first machine and class that ran it: the pair the page opens on.
    fn first(&self) -> (&'a str, &'a str) {
        (self.chips.iter())
            .flat_map(|&c| self.classes.iter().map(move |&k| (c, k)))
            .find(|&(c, k)| self.stem(c, k).is_some())
            .unwrap_or((
                self.chips[0],
                self.classes.first().copied().unwrap_or_default(),
            ))
    }
}

/// A section's three tabs (Summary and Over time, every machine at once; Latency distributions, the
/// pair its own buttons name), then the detail of the pair its own buttons below the tabs name.
/// Each set of buttons is two rows, as main's latency tab has: every machine, then every model
/// class, and a pair that ran nothing yet says so. An address naming a pair opens it.
fn tabbed(
    id: &str,
    pairs: &Pairs,
    summary: Element,
    over_time: Element,
    latency: impl Fn(&str, &str) -> Element,
    detail: impl Fn(&str, &str) -> Element,
) -> Element {
    let panel = |t: &str| format!("{id}-{}", slug(t));
    let (first_chip, first_class) = pairs.first();
    // One set of buttons and its pairs: `which` is latency or detail.
    let chosen = |which: &str, draw: &dyn Fn(&str, &str) -> Element| {
        let machine = |chip: &str| format!("{id}-{which}-machine-{}", slug(chip));
        let model = |class: &str| format!("{id}-{which}-model-{}", slug(class));
        rsx! {
            Tabs { selected: machine(first_chip), id: "{id}-{which}-machines",
                for &chip in &pairs.chips {
                    TabItem { value: machine(chip), selected: chip == first_chip, "{chip}" }
                }
            }
            Tabs { selected: model(first_class), id: "{id}-{which}-models",
                for &class in &pairs.classes {
                    TabItem { value: model(class), selected: class == first_class, {pairs.label(class)} }
                }
            }
            div { style: "padding-top: var(--cds-spacing-05)",
                for &chip in &pairs.chips {
                    for &class in &pairs.classes {
                        // An attribute's name is no format string: the set it belongs to is
                        // its value, `{id}-{which}`.
                        div { id: "{id}-{which}-{slug(chip)}-{slug(class)}",
                            "data-pair": "{id}-{which}", "data-machine": machine(chip),
                            "data-model": model(class),
                            hidden: (chip != first_chip || class != first_class).then_some(true),
                            match pairs.stem(chip, class) {
                                Some(stem) => draw(chip, stem),
                                None => rsx! { p { "{class} on {chip}: not run yet." } },
                            }
                        }
                    }
                }
            }
            script { dangerous_inner_html: pair_js(id, which) }
        }
    };
    rsx! {
        Stack { gap: 5,
            Tile { style: "max-width: 100%",
                div {
                    Tabs { selected: panel(TABS[0]), kind: TabsKind::Contained,
                        for (n, t) in TABS.iter().enumerate() {
                            TabItem { value: panel(t), target: Some(panel(t)), selected: n == 0, "{t}" }
                        }
                    }
                    div { style: "padding-top: var(--cds-spacing-05)",
                        ContentPanel { id: panel(TABS[0]), shown: true, {summary} }
                        ContentPanel { id: panel(TABS[1]), shown: false, {chosen("latency", &latency)} }
                        ContentPanel { id: panel(TABS[2]), shown: false, {over_time} }
                    }
                }
            }
            div { {chosen("detail", &detail)} }
        }
    }
}

/// A section's tabs, in order.
const TABS: [&str; 3] = ["Summary", "Latency distributions", "Over time"];

/// One machine and model's random-token detail: what its numbers came from, then the grid's maps,
/// its request plots and its users and prompt curves, as its machine section drew them.
fn grid_pair(id: &str, mr: &MachineRuns, stem: &str, chip: &str) -> Element {
    let Some(Entry { run, m }) = mr.latest(stem) else {
        return rsx! { div { "not run" } };
    };
    let mid = format!("{id}-{}-{}", slug(chip), slug(stem));
    let seen: &[Entry] = (mr.models.iter())
        .find(|(s, _)| *s == stem)
        .map_or(&[], |(_, es)| es.as_slice());
    rsx! {
        Stack { gap: 6,
            {provenance(mr, stem, chip)}
            if let Some(why) = no_numbers(m) {
                p { "No numbers: {why}. See the run's raw logs." }
            }
            {borrowed_note(&(m.borrowed.iter()).filter_map(|(k, when)| Some((Engine::ALL.into_iter().find(|e| e.key() == k)?, when.month_day().to_string()))).collect::<Vec<_>>())}
            if let Some(note) = previous_run(seen) { {note} }
            p { style: "font-size: 0.875rem", "What the marks on its numbers mean " {marks_key()} }
            {sweep_part(m, run, Sweep::Users)}
            {grid_maps(m, &mid, run)}
            {sweep_part(m, run, Sweep::Prompt)}
        }
    }
}

/// What a pair's numbers came from: the machine (its cores, memory and macOS), the model's
/// checkpoint, quantization and files, scratchy's build, the run and its parity check, the engines
/// that declined it, the models the machine's newest run skipped, and every run of the machine,
/// with its commit, steps, settings, engine versions and pins.
fn provenance(mr: &MachineRuns, stem: &str, chip: &str) -> Element {
    let Some(Entry { run, m }) = mr.latest(stem) else {
        return rsx! {};
    };
    let mc = &mr.runs[mr.runs.len() - 1].machine;
    let cores = match mc.cores_performance.filter(|&p| p != 0) {
        Some(p) => format!(
            "{} cores ({p}P + {}E)",
            mc.cores_total,
            mc.cores_efficiency.unwrap_or(0)
        ),
        None => format!("{} cores", mc.cores_total),
    };
    let build = (m.footprint.as_ref().and_then(|f| f.build_seconds))
        .map(|s| {
            let binary = (m.footprint.as_ref().and_then(|f| f.binary_bytes))
                .filter(|&b| b != 0)
                .map_or(String::new(), |b| {
                    format!(
                        ", {} MiB binary",
                        (b as f64 / 1_048_576.0).round_ties_even()
                    )
                });
            format!(" · scratchy build {s} s{binary}")
        })
        .unwrap_or_default();
    // An engine whose own memory guard refused the model: its columns are empty because it
    // never served. This says why, in the engine's words.
    let declined: Vec<String> = (Engine::ALL.iter())
        .filter_map(|e| {
            (m.engine_declined.as_ref())
                .and_then(|d| d.get(e.key()))
                .map(|r| format!("{} — {}", e.label(), r.as_str()))
        })
        .collect();
    // What each engine that serves its own build loaded for this model: an ollama tag, a GGUF file.
    let served: Vec<String> = (Engine::ALL.iter())
        .filter_map(|&e| served_as(m, e).map(|s| format!("{} {s}", e.label())))
        .collect();
    let newest = mr.runs[mr.runs.len() - 1];
    let skipped: Vec<String> = (newest.skipped_models.iter().flatten())
        .filter(|x| mr.latest(&x.stem).is_none())
        .map(|x| format!("{} ({})", x.stem, x.reason.as_deref().unwrap_or("skipped")))
        .collect();
    // The machine, the model's files and its build: folded with the runs, out of the way.
    let about = rsx! {
        p { "{cores} · {mc.memory_gb} GB unified memory · macOS {mc.macos}" }
        p {
            a { href: "https://huggingface.co/{m.model_id}", "{m.model_id}" }
            " · {m.quant.as_deref().unwrap_or(\"default\")}"
            if let Some(rev) = m.model_files.as_ref().and_then(|f| f.revision.as_deref()) {
                " · files "
                a { href: "https://huggingface.co/{m.model_id}/tree/{rev}", code { "{&rev[..rev.len().min(8)]}" } }
            }
            "{build}"
        }
    };
    rsx! {
        Stack { gap: 2,
            strong { "{stem} on {chip}" }
            // What a reader needs first: which run, and whether its answers matched mlx-lm.
            p {
                "Run {run.generated_utc.date()}, "
                code { "{run.repo.sha8()}" }
                if let Some(verdict) = parity(m) {
                    " · {verdict} " {parity_info()}
                }
                if let Some(helper) = helper(m) {
                    " · {helper} " {helper_info(m)}
                }
            }
            if !served.is_empty() {
                p { "Served as: {served.join(\"; \")}." }
            }
            if !declined.is_empty() {
                p { "Declined before serving: {declined.join(\"; \")}." }
            }
            if !skipped.is_empty() {
                p { "Not run on this machine: {skipped.join(\"; \")}." }
            }
            {runs_table(&mr.runs, about)}
        }
    }
}

/// One machine and model's random-token latency: its request plots, every request's timing in
/// each grid cell, as the overview's latency panel drew them.
fn grid_latency(id: &str, mr: &MachineRuns, stem: &str, chip: &str) -> Element {
    let Some(Entry { run, m }) = mr.latest(stem) else {
        return rsx! { div { "not run" } };
    };
    let mid = format!("{id}-latency-{}-{}", slug(chip), slug(stem));
    rsx! {
        Stack { gap: 2,
            strong { "{stem} on {chip}" }
            {request_plots(m, &mid, run)}
        }
    }
}

/// A summary map's square: by its machine, row, column group and column, each counted from 0.
pub(super) type MapAt = (usize, usize, usize, usize);

/// One square of a summary map: how many times faster scratchy is and the engine it is set
/// against ("oMLX (from Oct 27)"), or why it has no ratio; then its tooltip's lines.
pub(super) struct MapSquare {
    pub ratio: Result<(f64, String), String>,
    pub tip: Vec<String>,
}

/// One row of a summary map: its label (P, D, 2u), its measure's name, and what it measures.
pub(super) struct MapRow {
    pub label: String,
    pub name: String,
    pub about: String,
}

/// A summary map, the random-token grid's and every dataset's: per machine its rows, per column
/// group (a class, or a model) its columns (prompt sizes, or prompt-length buckets), a square
/// each on the one ratio scale, which stands beside them, and each row's slowest and fastest
/// ratio right of it.
pub(super) struct SummaryMap<'a> {
    /// Each column group's name, and under it the models it shows, a line each.
    pub groups: Vec<(&'a str, Vec<String>)>,
    /// The columns under every group.
    pub columns: Vec<String>,
    /// Each machine and its rows, top to bottom.
    pub machines: Vec<(&'a str, Vec<MapRow>)>,
    /// Each square drawn; a place with none is left empty.
    pub squares: BTreeMap<MapAt, MapSquare>,
    /// The squares with their factor written in them ([`SummaryMap::ratios`], [`ratio_labels`]).
    pub labelled: BTreeSet<MapAt>,
    /// A line under the map, where its columns need one.
    pub legend: Option<String>,
}

impl SummaryMap<'_> {
    /// Each square's ratio, where it has one.
    pub fn ratios(&self) -> BTreeMap<MapAt, f64> {
        (self.squares.iter())
            .filter_map(|(&at, s)| Some((at, s.ratio.as_ref().ok()?.0)))
            .collect()
    }

    /// The map, its ratio scale's id `scale`.
    pub fn render(&self, scale: &str) -> Element {
        let (groups, n) = (self.groups.len(), self.columns.len());
        let col = |k: usize, c: usize| 3 + k * n + c;
        // The grid's lines: three header lines (group, models, columns), then each machine's rows.
        let mut lines: Vec<(usize, usize, &MapRow)> = Vec::new();
        let mut spans: Vec<(&str, usize, usize)> = Vec::new();
        for (m, (chip, rows)) in self.machines.iter().enumerate() {
            let start = 3 + lines.len();
            lines.extend(rows.iter().enumerate().map(|(r, row)| (m, r, row)));
            if 3 + lines.len() > start {
                spans.push((chip, start, 3 + lines.len() - 1));
            }
        }
        let ratios = self.ratios();
        let range = |m: usize, r: usize| -> Option<(f64, f64)> {
            (ratios.iter())
                .filter(|&(at, _)| (at.0, at.1) == (m, r))
                .map(|(_, &v)| v)
                .fold(None, |acc, v| match acc {
                    None => Some((v, v)),
                    Some((lo, hi)) => Some((f64::min(lo, v), f64::max(hi, v))),
                })
        };
        // Over the map's first and last two columns the tooltip reaches inward, not off its edge.
        let align = |k: usize, c: usize| match (k, c) {
            (0, c) if c < 2 => "top-left",
            (k, c) if k + 1 == groups && c + 2 >= n => "top-right",
            _ => "top",
        };
        // Each square with Carbon's tooltip: the square is its trigger, focusable so a keyboard
        // opens it too; the tooltip gives the ratio (or why there is none), then its lines.
        let square = |at: MapAt, sq: &MapSquare, place: String| -> Element {
            let (fill, label, verdict) = match &sq.ratio {
                Ok((r, vs)) => {
                    let step = ratio_step(*r);
                    (
                        format!(
                            "background: {}; color: {}",
                            RATIO_STEPS[step],
                            ratio_ink(step)
                        ),
                        self.labelled.contains(&at).then(|| format!("×{r:.1}")),
                        format!("×{r:.2} vs {vs}"),
                    )
                }
                Err(why) => (STRIPED.to_string(), None, why.clone()),
            };
            rsx! {
                // The grid's own cell, and a flex box: Carbon puts the trigger in an inline box in its
                // shadow DOM, whose line would add room for text under every square.
                cds-tooltip { "align": align(at.2, at.3), style: "{place}; display: flex",
                    // Its own width: Carbon wraps the trigger in an inline box, so an empty square would
                    // shrink to nothing.
                    div { tabindex: "0", style: "width: {SQUARE}; height: {SQUARE}; {fill}; display: flex; align-items: center; justify-content: center; font-size: 0.625rem; font-weight: 600",
                        if let Some(label) = label { "{label}" }
                    }
                    cds-tooltip-content { style: "text-align: left",
                        TipHeading { heading: verdict }
                        for line in &sq.tip { div { "{line}" } }
                    }
                }
            }
        };
        let template = format!("max-content max-content repeat({}, {SQUARE})", groups * n);
        rsx! {
            div { style: "display: flex; gap: 1.5rem; align-items: flex-start; overflow-x: auto",
                div { style: "display: grid; grid-template-columns: {template} max-content max-content; column-gap: 2px; row-gap: 2px; align-items: center",
                    for (k, (group, models)) in self.groups.iter().enumerate() {
                        div { style: "grid-row: 1; grid-column: {col(k, 0)} / span {n}; text-align: center; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0",
                            "{group}"
                        }
                        div { style: "grid-row: 2; grid-column: {col(k, 0)} / span {n}; text-align: center; font-size: 0.75rem; min-width: 0",
                            for name in models {
                                div { style: "overflow: hidden; text-overflow: ellipsis; white-space: nowrap", title: "{name}", "{name}" }
                            }
                        }
                        for (c, label) in self.columns.iter().enumerate() {
                            div { style: "grid-row: 3; grid-column: {col(k, c)}; text-align: center; font-size: 0.625rem; white-space: nowrap",
                                "{label}"
                            }
                        }
                    }
                    for (i, &(m, r, row)) in lines.iter().enumerate() {
                        div { style: "grid-row: {i + 4}; grid-column: 2; font-size: 0.875rem; font-weight: 600; text-align: right; padding-right: 0.5rem; white-space: nowrap",
                            Definition { alignment: Alignment::BottomLeft, heading: row.name.clone(), definition: row.about.clone(), strong { "{row.label}" } }
                        }
                        for (&at, sq) in self.squares.range((m, r, 0, 0)..=(m, r, usize::MAX, usize::MAX)) {
                            {square(at, sq, format!("grid-row: {}; grid-column: {}", i + 4, col(at.2, at.3)))}
                        }
                        if let Some((lo, hi)) = range(m, r) {
                            div { style: "grid-row: {i + 4}; grid-column: {col(groups, 0) + 1}; font-size: 0.75rem; white-space: nowrap; padding-left: 1.5rem",
                                "×{lo:.2}–×{hi:.2}"
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
                        div { style: "grid-row: {first + 1} / {last + 2}; grid-column: {col(groups, 0)}; align-self: stretch; display: flex; padding-left: 0.5rem",
                            {ratio_scale(ScaleFit::Fill, scale)}
                        }
                    }
                }
            }
            if let Some(legend) = &self.legend {
                p { style: "font-size: 0.75rem", "{legend}" }
            }
        }
    }
}

/// A dataset's summary map ([`SummaryMap`]): per machine its rows (P and D, prefill and decode at
/// 1 user, then end-to-end time at each users count above it, 2u, 4u), per summary column (a
/// class, or a model) a column each for its prompt-length buckets ([`Dataset::buckets`]), each
/// square scratchy's median against the pair's rival's over the requests of that bucket. Runs
/// from before cells kept each request's prompt count have none to draw, and say so.
fn summary_map(
    dataset: &str,
    id: &str,
    ran: &[(&str, Vec<&str>)],
    machines: &BTreeMap<&str, MachineRuns>,
    grid: &SummaryGrid,
) -> Element {
    // Each pair's scratchy and rival cells of this dataset, by users count.
    let pair = |chip: &str, column: &str| -> Option<Pair> {
        let stem = grid.stem(chip, column)?;
        let Entry { m, .. } = machines[chip].latest(stem)?;
        let rival = tasks::rival_of(m, dataset);
        let by = |e: Engine| -> BTreeMap<u32, &TaskCell> {
            cells(m, e, dataset)
                .into_iter()
                .map(|c| (c.conc, c))
                .collect()
        };
        Some(Pair {
            stem: m.stem.as_str(),
            rival,
            ours: by(Engine::Scratchy),
            theirs: rival.map(by).unwrap_or_default(),
        })
    };
    let any_bucketed = ran.iter().any(|&(chip, _)| {
        (grid.columns.iter()).any(|&col| {
            pair(chip, col).is_some_and(|p| p.ours.values().any(|c| buckets::bucketed(c)))
        })
    });
    if !any_bucketed {
        return rsx! {
            p { "This dataset's summary is drawn by prompt length from runs that kept each request's \
                 prompt count; none of the runs shown did. The detail below has each pair's numbers." }
        };
    }
    // Each machine's rows: P and D at 1 user, then end-to-end time at each users count above it
    // that any pair of the machine ran.
    let rows_of = |chip: &str| -> Vec<Row> {
        let mut users: Vec<u32> = (grid.columns.iter())
            .filter_map(|&col| pair(chip, col))
            .flat_map(|p| p.ours.keys().copied().collect::<Vec<_>>())
            .collect();
        users.sort_unstable();
        users.dedup();
        let mut rows = Vec::new();
        if users.contains(&1) {
            rows.push(Row { label: "P".into(), name: "Prefill", users: 1, timing: RequestTiming::Ttft,
                about: "Time to first token with 1 user: how long the prompt takes before the answer starts." });
            rows.push(Row { label: "D".into(), name: "Decode", users: 1, timing: RequestTiming::Tpot,
                about: "Time per output token with 1 user: how fast the answer streams once it has started." });
        }
        for &u in users.iter().filter(|&&u| u > 1) {
            rows.push(Row {
                label: format!("{u}u"),
                name: "End-to-end Latency",
                users: u,
                timing: RequestTiming::E2el,
                about: "From sending a request to its last token, with that many users at once.",
            });
        }
        rows
    };
    let bs = of(dataset).map_or(buckets::WHOLE, |d| d.buckets);
    // A square's ratio, how many times faster scratchy is, or why it has none.
    let ratio = |p: Option<&Pair>, row: &Row, b: usize| -> Result<(f64, String), &'static str> {
        let p = p.ok_or("not run")?;
        let rival = p.rival.ok_or("scratchy alone so far")?;
        let (ours, theirs) = (p.ours.get(&row.users), p.theirs.get(&row.users));
        let (Some(ours), Some(theirs)) = (ours, theirs) else {
            return Err("not run at this load");
        };
        if tasks::mostly_failed(ours) || tasks::mostly_failed(theirs) {
            return Err("most requests failed");
        }
        let a = buckets::median_in(ours, bs, row.timing, b);
        let t = buckets::median_in(theirs, bs, row.timing, b);
        let r = tasks::times(a, t, true).ok_or("no timing on one side")?;
        let from = (theirs.borrowed_from.as_ref())
            .map_or(String::new(), |w| format!(" (from {})", w.month_day()));
        Ok((r, format!("{}{from}", rival.label())))
    };
    let mut squares = BTreeMap::new();
    let mut map_machines = Vec::new();
    for (m, &(chip, _)) in ran.iter().enumerate() {
        let rows = rows_of(chip);
        for (r, row) in rows.iter().enumerate() {
            for (k, &column) in grid.columns.iter().enumerate() {
                let p = pair(chip, column);
                for (b, &(bucket, _)) in bs.iter().enumerate() {
                    // How many requests each side's median is over: a cell sends few, so a square
                    // of one or two can turn on a single answer, and says so.
                    let over = p.as_ref().and_then(|p| {
                        let n = |c: Option<&&TaskCell>| {
                            c.map_or(0, |c| buckets::in_bucket(c, bs, row.timing, b).len())
                        };
                        let (a, t) = (n(p.ours.get(&row.users)), n(p.theirs.get(&row.users)));
                        (a + t > 0).then(|| format!("over {a} and {t} requests"))
                    });
                    let stem = p.as_ref().map_or("", |p| p.stem);
                    // scratchy's guesses kept over the whole cell: its helper's, not a bucket's.
                    let helped = (p.as_ref())
                        .and_then(|p| kept(p.ours.get(&row.users)?.spec_decode.as_ref()))
                        .map(|k| format!("helper kept {k} guessed tokens, over the cell"));
                    let tip = [
                        Some(format!("{stem} on {chip}")),
                        Some(format!(
                            "{} · {bucket} prompt tokens · {}",
                            row.name,
                            users_at(row.users)
                        )),
                        over,
                        helped,
                    ];
                    squares.insert(
                        (m, r, k, b),
                        MapSquare {
                            ratio: ratio(p.as_ref(), row, b).map_err(str::to_string),
                            tip: tip.into_iter().flatten().collect(),
                        },
                    );
                }
            }
        }
        if !rows.is_empty() {
            let rows = (rows.into_iter())
                .map(|r| MapRow {
                    label: r.label,
                    name: r.name.to_string(),
                    about: r.about.to_string(),
                })
                .collect();
            map_machines.push((chip, rows));
        }
    }
    let mut map = SummaryMap {
        groups: (grid.columns.iter())
            .map(|&c| {
                (
                    c,
                    grid.models_of(c).into_iter().map(str::to_string).collect(),
                )
            })
            .collect(),
        columns: bs.iter().map(|&(label, _)| label.to_string()).collect(),
        machines: map_machines,
        squares,
        labelled: BTreeSet::new(),
        legend: None,
    };
    // The squares to write their factor in, as the random-token map does: each colour step's
    // square furthest from ×1, so its shades read apart.
    map.labelled = ratio_labels(&map.ratios(), HEAT_LABELS);
    map.render(&format!("{id}-summary-scale"))
}

/// One machine and model's cells of a dataset, scratchy's and its rival's, by users count.
struct Pair<'a> {
    stem: &'a str,
    rival: Option<Engine>,
    ours: BTreeMap<u32, &'a TaskCell>,
    theirs: BTreeMap<u32, &'a TaskCell>,
}

/// One row of the summary map: its label (P, D, 2u), its measure's name and what it measures,
/// the users count it is read at, and the request timing it compares.
struct Row {
    label: String,
    name: &'static str,
    users: u32,
    timing: RequestTiming,
    about: &'static str,
}

/// The engines a pair of `dataset` is drawn for, scratchy then its rival, and every users count
/// either measured: one scratchy failed still shows its rival's.
fn pair_shape(m: &Model, dataset: &str) -> (Vec<Engine>, Vec<u32>) {
    let engines: Vec<Engine> = std::iter::once(Engine::Scratchy)
        .chain(tasks::rival_of(m, dataset))
        .collect();
    let mut users: Vec<u32> = (engines.iter())
        .flat_map(|&e| cells(m, e, dataset).into_iter().map(|c| c.conc))
        .collect();
    users.sort_unstable();
    users.dedup();
    (engines, users)
}

/// One machine and model's latency distributions for a dataset: the end-to-end time of every
/// request at each users count, scratchy against its rival.
fn pair_latency(dataset: &str, mr: &MachineRuns, stem: &str, chip: &str) -> Element {
    let Some(Entry { m, .. }) = mr.latest(stem) else {
        return rsx! { div { "not run" } };
    };
    let (engines, users) = pair_shape(m, dataset);
    let curves = |u: u32| -> Vec<(Engine, Vec<f64>)> {
        (engines.iter())
            .filter_map(|&e| {
                let c = cells(m, e, dataset).into_iter().find(|c| c.conc == u)?;
                let mut vs = tasks::request_ms(c, RequestTiming::E2el);
                vs.retain(|&v| v > 0.0);
                vs.sort_by(f64::total_cmp);
                (!vs.is_empty()).then_some((e, vs))
            })
            .collect()
    };
    if !users.iter().any(|&u| !curves(u).is_empty()) {
        return rsx! { p { "{stem} on {chip}: this run did not keep its requests." } };
    }
    rsx! {
        Stack { gap: 2,
            strong { "{stem} on {chip}" }
            p { style: "contain: inline-size",
                for (n, (&e, class)) in engines.iter().zip(["s1", "s2"]).enumerate() {
                    if n > 0 { " versus " }
                    {engine_key(e, class)}
                }
                ": at each users count, the share of each one's requests (up) done within each time \
                 (across), end to end. Time runs on a log axis, centred on the two medians; further \
                 left is faster, a steeper rise steadier."
            }
            div { style: "display: grid; gap: var(--cds-spacing-05); grid-template-columns: repeat(auto-fit, {ECDF_COLUMN})",
                for &u in &users {
                    {ecdf_chart(&curves(u), &engines, &["E2EL", &users_at(u)],
                        &format!("end-to-end time of every request, {} at once, {stem} on {chip}", users_at(u)))}
                }
            }
            {borrowed_note(&lent(m, dataset))}
        }
    }
}

/// One machine and model's detail for a dataset: what its numbers came from, its numbers, and
/// its users curves.
fn pair_detail(dataset: &str, mr: &MachineRuns, stem: &str, chip: &str) -> Element {
    let Some(Entry { m, .. }) = mr.latest(stem) else {
        return rsx! { div { "not run" } };
    };
    let (engines, users) = pair_shape(m, dataset);
    rsx! {
        Stack { gap: 5,
            {provenance(mr, stem, chip)}
            if let Some(why) = no_numbers(m) {
                p { "No numbers: {why}. See the run's raw logs." }
            }
            {tasks::detail_table(m, dataset)}
            {users_curves(m, dataset, &engines, &users, stem, chip)}
            {borrowed_note(&lent(m, dataset))}
        }
    }
}

/// Why scratchy has no numbers for a model, when it has none: its build failed, its answers did
/// not match mlx-lm's (so no engine was timed), or its server never served. None when it served.
fn no_numbers(m: &Model) -> Option<&'static str> {
    if served(m) {
        return None;
    }
    Some(if !m.built {
        "build failed"
    } else if m.parity_mlx_lm == Some(Some(false)) {
        "its answers did not match mlx-lm's, so no engine was timed"
    } else {
        "the server never served"
    })
}

/// The engines whose `dataset` cells this model borrowed, each with the run's day.
fn lent(m: &Model, dataset: &str) -> Vec<(Engine, String)> {
    (Engine::RIVALS.into_iter())
        .filter_map(|e| {
            let when = (e.tasks(m).iter())
                .filter(|c| c.workload == dataset)
                .find_map(|c| c.borrowed_from.as_ref())?;
            Some((e, when.month_day().to_string()))
        })
        .collect()
}

/// Says which engines' numbers in a pair an earlier run measured: this run did not.
fn borrowed_note(lent: &[(Engine, String)]) -> Element {
    rsx! {
        for (e, day) in lent {
            p { style: "font-size: 0.875rem",
                "{e.label()}'s numbers here are from its {day} run: this run measured scratchy alone."
            }
        }
    }
}

/// An engine's measured cells of `dataset`, fewest users first.
fn cells<'m>(m: &'m Model, e: Engine, dataset: &str) -> Vec<&'m TaskCell> {
    let mut cs: Vec<&TaskCell> = (e.tasks(m).iter())
        .filter(|c| c.workload == dataset && c.measured())
        .collect();
    cs.sort_by_key(|c| c.conc);
    cs
}

/// A dataset's users curves for one model: time to first token, time per output token,
/// end-to-end time and output tokens a second, at each users count, scratchy against its rival.
fn users_curves(
    m: &Model,
    dataset: &str,
    engines: &[Engine],
    users: &[u32],
    stem: &str,
    chip: &str,
) -> Element {
    type Pick = fn(&TaskCell) -> Option<f64>;
    let metrics: [(&str, &str, Pick); 4] = [
        ("TTFT p50", "ms", |c| c.median_ttft_ms),
        ("TPOT p50", "ms", |c| c.median_tpot_ms),
        ("End-to-end p50", "ms", |c| c.median_e2el_ms),
        ("Output tokens a second", "tok/s", |c| c.output_throughput),
    ];
    let series = |pick: Pick| -> Vec<Series> {
        (engines.iter().zip(["s1", "s2"]))
            .map(|(&e, class)| Series {
                label: e.label().to_string(),
                class,
                pts: (cells(m, e, dataset).into_iter())
                    .map(|c| (c.conc, pick(c)))
                    .collect(),
            })
            .filter(|s| s.pts.iter().any(|p| p.1.is_some()))
            .collect()
    };
    rsx! {
        Stack { gap: 2,
            strong { "Users" }
            // Two a row across the panel, so four bars each read at a glance.
            div { style: "display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 20rem), 1fr)); gap: 1rem 1.5rem",
                for (title, unit, pick) in metrics {
                    {
                        let ss = series(pick);
                        rsx! {
                            if !ss.is_empty() {
                                div { style: "min-width: 0",
                                    {bar_chart(&ss, users, (title, unit), Sweep::Users, "280px",
                                        &format!("{title}, {stem} on {chip}"))}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Shows the pair one set of buttons names (`which`: the latency tab's, or the detail's), hiding
/// the rest, as main's latency tab does: its machine row and its model row choose one pair between
/// them. Keeps the pair in the address, and opens the tab and pair an address names.
fn pair_js(id: &str, which: &str) -> String {
    format!(
        "(function () {{\
  var machine = document.getElementById('{id}-{which}-machines'), model = document.getElementById('{id}-{which}-models');\
  var pairs = document.querySelectorAll('[data-pair=\"{id}-{which}\"]');\
  function show() {{\
    var shown;\
    pairs.forEach(function (p) {{\
      p.hidden = p.dataset.machine !== machine.value || p.dataset.model !== model.value;\
      if (!p.hidden) shown = p;\
    }});\
    return shown;\
  }}\
  function chose() {{ var p = show(); if (p) history.replaceState(null, '', '#' + p.id); }}\
  function fromHash() {{\
    var p = document.getElementById(decodeURIComponent(location.hash.slice(1)));\
    if (p && p.dataset.pair === '{id}-{which}') {{\
      machine.value = p.dataset.machine; model.value = p.dataset.model;\
      var panel = p.closest('[data-panel]');\
      var tab = panel && document.querySelector('cds-tab[target=\"' + CSS.escape(panel.id) + '\"]');\
      if (tab) tab.closest('cds-tabs').value = panel.id;\
    }}\
    show();\
  }}\
  machine.addEventListener('cds-tabs-selected', chose);\
  model.addEventListener('cds-tabs-selected', chose);\
  customElements.whenDefined('cds-tabs')\
    .then(function () {{ return Promise.all([machine.updateComplete, model.updateComplete]); }})\
    .then(function () {{ fromHash(); window.addEventListener('hashchange', fromHash); }});\
}})();"
    )
}

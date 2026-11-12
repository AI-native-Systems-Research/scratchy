//! The real-task view: each dataset workload a model ran, scratchy against the engine it is
//! compared with, on the same tasks (`scr bench serve --dataset-name <workload>`).

use dioxus::prelude::*;

use crate::carbon::{Fold, Stack, Table, TableCell, TableRow};
use crate::charts::{RangeGroup, RangeMark, range_dots};

use super::data::{Model, RequestTiming, TaskCell};
use super::numbers::Engine;

/// A workload's name on the page, and what it measures.
pub fn workload_label(w: &str) -> (&'static str, &'static str) {
    match w {
        "sharegpt" => ("ShareGPT chat", "real chat prompts, one turn each"),
        "rag" => (
            "MuSiQue, timed",
            "multi-document QA, MuSiQue: a question over its passages",
        ),
        "rag-spans" => (
            "RAG, passage reuse",
            "the RAG tasks through scratchy's span cache",
        ),
        "sharegpt-multi-turn" => (
            "ShareGPT conversations",
            "each turn with the turns before it",
        ),
        "longbench" => ("LongBench", "LongBench v2 short: long documents"),
        "shared-prefix" => (
            "Shared prompt",
            "short questions sharing one long system prompt",
        ),
        "shared-prefix-control" => ("Shared prompt, no reuse", "the same, every prompt its own"),
        "rag-accuracy" => (
            "MuSiQue, scored",
            "RAG over more tasks, for its accuracy alone: 100 on a quick run, 400 or the whole dev set on a full one",
        ),
        _ => ("dataset", ""),
    }
}

/// The page's order: the quick run's workloads first.
const ORDER: [&str; 8] = [
    "sharegpt",
    "rag",
    "rag-spans",
    "sharegpt-multi-turn",
    "longbench",
    "shared-prefix",
    "shared-prefix-control",
    "rag-accuracy",
];

/// The engine a model's `dataset` cells are set against: the first rival that measured any of
/// them. None for scratchy alone.
pub fn rival_of(m: &Model, dataset: &str) -> Option<Engine> {
    Engine::RIVALS
        .into_iter()
        .find(|&e| (e.tasks(m).iter()).any(|c| c.workload == dataset && c.measured()))
}

/// A cell's requests' times to first token or end-to-end times, failed ones left out. Empty for
/// the time per output token, which a dataset cell keeps for none of its requests, and for a cell
/// that kept no requests.
pub fn request_ms(c: &TaskCell, t: RequestTiming) -> Vec<f64> {
    let kept = match t {
        RequestTiming::Ttft => c.ttfts_ms.as_deref(),
        RequestTiming::E2el => c.e2els_ms.as_deref(),
        RequestTiming::Tpot => None,
    };
    kept.unwrap_or_default().iter().flatten().copied().collect()
}

/// A workload's place in the page's order; one the page does not know goes last.
pub fn order(w: &str) -> usize {
    ORDER.iter().position(|o| *o == w).unwrap_or(ORDER.len())
}

/// Fewer tasks than this and a cell's accuracy is indicative: at 8 a task is 12.5 points of it.
const FIRM_ACCURACY_TASKS: u32 = 100;

/// The engine scratchy's task numbers are compared with: the first, in the heat maps' order,
/// that ran this workload at this users count.
pub fn rival<'a>(m: &'a Model, workload: &str, conc: u32) -> Option<(Engine, &'a TaskCell)> {
    Engine::RIVALS.into_iter().find_map(|e| {
        (e.tasks(m).iter())
            .find(|c| c.workload == workload && c.conc == conc && c.measured())
            .map(|c| (e, c))
    })
}

/// A cell's value, as the table shows it.
fn ms(v: Option<f64>) -> String {
    v.map_or_else(
        || "—".to_string(),
        |v| {
            if v >= 10_000.0 {
                format!("{:.1} s", v / 1000.0)
            } else {
                format!("{v:.0} ms")
            }
        },
    )
}

fn n(v: Option<f64>) -> String {
    v.map_or_else(|| "—".to_string(), |v| format!("{v:.0}"))
}

/// Whether most of a cell's answers ran to their cap.
fn mostly_capped(c: &TaskCell) -> bool {
    c.completed
        .is_some_and(|n| n > 0 && 2 * c.capped_answers.unwrap_or(0) > n)
}

/// Whether most of a cell's requests failed: its times are the few that answered, its accuracy
/// counts the failures wrong, and its energy spreads the time it ran over requests never served.
pub fn mostly_failed(c: &TaskCell) -> bool {
    c.failed.unwrap_or(0) > c.completed.unwrap_or(0)
}

/// Whether a cell's answers stand for the engine's: neither mostly capped nor mostly failed.
fn usable(c: &TaskCell) -> bool {
    !mostly_capped(c) && !mostly_failed(c)
}

/// Whether most of a cell's answers opened as reasoning: their accuracy and tokens count it too.
fn mostly_thought(c: &TaskCell) -> bool {
    c.completed
        .is_some_and(|n| n > 0 && 2 * c.thinking_answers.unwrap_or(0) > n)
}

/// Accuracy, where it is like for like: none for a cell most of whose answers opened as
/// reasoning, ran to their cap (which read as wrong without being answers), or failed.
fn accuracy(c: &TaskCell) -> Option<f64> {
    c.accuracy.filter(|_| !mostly_thought(c) && usable(c))
}

/// Why a cell shows no accuracy, in its cell: "not scored" for a timed cell, its answers forced
/// to one length; else what held it back, with how many.
fn no_accuracy(c: &TaskCell) -> String {
    let done = c.completed.unwrap_or(0) as usize;
    let of = |k: Option<u32>| {
        format!(
            "{} of {}",
            crate::thousands(k.unwrap_or(0) as usize),
            crate::thousands(done)
        )
    };
    if c.accuracy.is_none() {
        "not scored".to_string()
    } else if mostly_capped(c) {
        format!(
            "cut off: {} answers hit the length limit",
            of(c.capped_answers)
        )
    } else if mostly_thought(c) {
        format!("answered in reasoning: {}", of(c.thinking_answers))
    } else {
        "—".to_string()
    }
}

/// How many times faster scratchy is to its first token than the rival: none where either mostly
/// ran its answers to their cap, whose requests were not the tasks' answers, or mostly failed.
fn ttft_times(c: &TaskCell, r: &TaskCell) -> Option<f64> {
    (usable(c) && usable(r))
        .then(|| times(c.median_ttft_ms, r.median_ttft_ms, true))
        .flatten()
}

/// How far apart two cells' mean answers may be, as a share of the longer, for their whole
/// answers' times to be compared: an engine that stops at EOS where the other runs on (oMLX does,
/// despite `ignore_eos`) answers in fewer tokens, and its time is no faster for it.
const SAME_LENGTH: f64 = 0.10;

/// How many times faster scratchy is to its whole answer than the rival: only where both answered
/// at about the same length ([`SAME_LENGTH`]), neither mostly ran to its cap, and both say how
/// long their answers were.
fn total_times(c: &TaskCell, r: &TaskCell) -> Option<f64> {
    let (a, b) = (c.answer_tokens()?, r.answer_tokens()?);
    let close = (a - b).abs() <= SAME_LENGTH * a.max(b);
    (close && usable(c) && usable(r))
        .then(|| times(c.median_e2el_ms, r.median_e2el_ms, true))
        .flatten()
}

/// What the energy columns cover: the counters, the idle floor, and the sampling.
const ENERGY_COVERS: &str = "From the CPU, GPU and ANE energy counters, not DRAM, with the Mac's \
     idle floor of about 1.6 W in it; read once a second, so a cell under about 20 s is coarse.";

/// Each column's header and what it means, shown where it is read ([`Table`]'s definitions).
fn columns() -> (Vec<String>, Vec<String>) {
    let cols = [
        (
            "workload",
            "Timed: every answer forced to one length, so times compare across engines. Scored: \
            the full run's larger sample, answers ending on their own; its accuracy is the \
            accuracy table's.",
        ),
        (
            "load",
            "Users sending at once, each its next request when its last answer returns.",
        ),
        (
            "TTFT",
            "Time to first token, the median: how long before the answer starts.",
        ),
        (
            "total",
            "From sending a request to its last token, the median.",
        ),
        ("answer tok", "Output tokens an answer took, on average."),
        (
            "cached tok",
            "Prompt tokens the engine's cache served, on average.",
        ),
        (
            "J/request",
            &format!("Energy a request, over every request sent. {ENERGY_COVERS}"),
        ),
        (
            "J/token",
            &format!("Energy an output token. {ENERGY_COVERS}"),
        ),
        (
            "vs",
            "The engine scratchy is set against on these same tasks.",
        ),
        (
            "TTFT",
            "How many times faster scratchy is to its first token.",
        ),
        (
            "total",
            "How many times faster scratchy is end to end; shown only where both \
            engines' answers were about one length.",
        ),
        (
            "J/request",
            "How many times less energy scratchy spent a request; shown only where \
            both engines' answers were about one length.",
        ),
    ];
    cols.iter()
        .map(|(h, a)| (h.to_string(), a.to_string()))
        .unzip()
}

/// A cell's joules a request, an output token, and a right answer where it is scored and shows
/// its accuracy; None where its run kept no energy, or most of its requests failed.
fn per_request(c: &TaskCell) -> Option<f64> {
    let e = c.energy.as_ref().filter(|_| !mostly_failed(c))?;
    e.j_per_request
}

fn per_token(c: &TaskCell) -> Option<f64> {
    let e = c.energy.as_ref().filter(|_| !mostly_failed(c))?;
    e.j_per_output_token
}

fn per_right(c: &TaskCell) -> Option<f64> {
    accuracy(c).filter(|&a| a > 0.0)?;
    c.energy.as_ref().and_then(|e| e.j_per_right_answer)
}

/// Joules, to two decimals under 1, one under 100, whole above.
fn joules(v: Option<f64>) -> String {
    v.map_or_else(
        || "—".to_string(),
        |v| match v {
            v if v < 1.0 => format!("{v:.2}"),
            v if v < 100.0 => format!("{v:.1}"),
            v => format!("{v:.0}"),
        },
    )
}

/// How many times less energy scratchy spent a request than the rival: as the total time's
/// ratio is, only where both answered about one length and neither mostly ran to its cap, an
/// engine that stops early spending less for answering less.
fn energy_times(c: &TaskCell, r: &TaskCell) -> Option<f64> {
    let (a, b) = (c.answer_tokens()?, r.answer_tokens()?);
    let close = (a - b).abs() <= SAME_LENGTH * a.max(b);
    (close && usable(c) && usable(r))
        .then(|| times(per_request(c), per_request(r), true))
        .flatten()
}

/// The mark of a cell some of whose requests failed, as the random cells carry it.
fn failed(c: &TaskCell) -> &'static str {
    if c.failed.unwrap_or(0) > 0 {
        super::numbers::FAILED
    } else {
        ""
    }
}

fn pct(v: Option<f64>) -> String {
    v.map_or_else(|| "—".to_string(), |v| format!("{:.0}%", v * 100.0))
}

/// A cell's accuracy as the table shows it: with its task count, and marked `~` where too few
/// tasks make it indicative ([`FIRM_ACCURACY_TASKS`]); for a cell most of whose requests failed,
/// how many answered instead.
fn right(c: &TaskCell) -> String {
    if mostly_failed(c) {
        let done = c.completed.unwrap_or(0) as usize;
        let sent = done + c.failed.unwrap_or(0) as usize;
        return format!(
            "{} of {} answered",
            crate::thousands(done),
            crate::thousands(sent)
        );
    }
    let Some(a) = accuracy(c) else {
        return no_accuracy(c);
    };
    let whole = match c.num_tasks {
        Some(n) if n >= FIRM_ACCURACY_TASKS => format!("{} of {n}", pct(Some(a))),
        Some(n) => format!("~{} of {n}", pct(Some(a))),
        None => format!("~{}", pct(Some(a))),
    };
    match by_hops(c) {
        Some(hops) => format!("{whole} ({hops})"),
        None => whole,
    }
}

/// A MuSiQue cell's accuracy by hop count, where it recorded one: "2-hop 52%, 3-hop 43%, 4-hop 38%".
fn by_hops(c: &TaskCell) -> Option<String> {
    let by = c.accuracy_by_group.as_ref().filter(|b| !b.is_empty())?;
    let mut hops: Vec<(u32, Option<f64>)> = (by.iter())
        .filter_map(|(g, &a)| Some((g.parse().ok()?, a)))
        .collect();
    hops.sort_unstable_by_key(|&(h, _)| h);
    Some(
        (hops.into_iter())
            .map(|(h, a)| format!("{h}-hop {}", pct(a)))
            .collect::<Vec<_>>()
            .join(", "),
    )
}

/// How many times better scratchy is: `rival / scratchy` for a time (lower is better), `scratchy
/// / rival` for cached tokens. None where either side has no number.
pub fn times(scratchy: Option<f64>, rival: Option<f64>, lower_better: bool) -> Option<f64> {
    let (s, r) = (scratchy?, rival?);
    (s > 0.0 && r > 0.0).then(|| if lower_better { r / s } else { s / r })
}

fn x(v: Option<f64>) -> String {
    v.map_or_else(String::new, |v| format!("×{v:.2}"))
}

/// Why a pair's cells have no ratio, where it is the rival's doing: most of its requests failed.
fn no_ratio(theirs: &TaskCell) -> Option<String> {
    mostly_failed(theirs).then(|| {
        let done = theirs.completed.unwrap_or(0) as usize;
        let sent = done + theirs.failed.unwrap_or(0) as usize;
        format!(
            "rival: {} of {} answered",
            crate::thousands(done),
            crate::thousands(sent)
        )
    })
}

/// An accuracy's 95% range, in points: the binomial interval over its tasks,
/// ±1.96·√(p(1−p)/n), to one decimal. None where it shows no accuracy or kept no task count.
fn range(c: &TaskCell) -> Option<f64> {
    let (p, n) = (accuracy(c)?, f64::from(c.num_tasks.filter(|&n| n > 0)?));
    Some((1960.0 * (p * (1.0 - p) / n).sqrt()).round() / 10.0)
}

/// A cell's accuracy with its range, "25% ±4.2", as the accuracy table shows it; or why it has
/// none, or how many requests answered where most failed ([`right`]).
fn scored(c: &TaskCell) -> String {
    match (accuracy(c), range(c)) {
        (Some(a), Some(r)) => format!("{} ±{r:.1}", pct(Some(a))),
        (Some(a), None) => format!("~{}", pct(Some(a))),
        _ => right(c),
    }
}

/// A dataset's scored cells, a row each a machine, model and engine: scratchy's and its rival's
/// side by side, each accuracy with its range, by hops, and joules a right answer; a held-back
/// one says why. `pairs` is each machine and its models as the section reads them. Nothing where
/// none of them was scored.
pub fn accuracy_table(dataset: &str, pairs: &[(&str, &Model)]) -> Element {
    let rows: Vec<(String, &str, &TaskCell)> = (pairs.iter())
        .flat_map(|&(chip, m)| {
            let mut engines = vec![Engine::Scratchy];
            engines.extend(rival_of(m, dataset));
            engines.into_iter().flat_map(move |e| {
                (e.tasks(m).iter())
                    .filter(|c| of_dataset(c, dataset) && !c.timed_answers && c.accuracy.is_some())
                    .map(move |c| (format!("{} on {chip}", m.stem), e.label(), c))
            })
        })
        .collect();
    if rows.is_empty() {
        return rsx! {};
    }
    let cols = [
        ("model", "The model and the Mac it ran on."),
        (
            "engine",
            "scratchy, or the engine it is set against on these same tasks.",
        ),
        ("load", "Users sending at once."),
        (
            "tasks",
            "Tasks asked: the full run's sample, every engine the same ones.",
        ),
        (
            "accuracy",
            "The share of tasks answered right, a failed or cut-off answer counted \
            wrong, ± its 95% range over the tasks. Not shown where most answers were cut off at the length limit, \
            opened in reasoning, or failed: the cell says which.",
        ),
        (
            "by hops",
            "Accuracy over each hop count's questions: 2-hop, 3-hop and 4-hop.",
        ),
        (
            "cut off",
            "Answers cut off at the length limit before they ended on their own.",
        ),
        ("J/right answer", "Energy over the right answers alone."),
    ];
    let (headers, abouts): (Vec<String>, Vec<String>) = cols
        .iter()
        .map(|(h, a)| (h.to_string(), a.to_string()))
        .unzip();
    let groups = accuracy_dots(dataset, pairs);
    rsx! {
        Stack { gap: 2,
            // The dataset's own name, "MuSiQue accuracy": its scored cell's name is "MuSiQue, scored".
            strong { "{workload_label(dataset).0.split(',').next().unwrap_or_default()} accuracy" }
            p { style: "font-size: 0.875rem",
                "Each dot is an engine's accuracy on the same tasks, its bar the 95% range. Where two \
                 engines' bars overlap they are within noise of each other; where they do not, the gap \
                 is real. Hover a dot for its numbers; the table view has every one."
            }
            {range_dots(&groups)}
            Fold { title: "Table view",
            Table { headers, abouts,
                for (model, engine, c) in rows {
                    TableRow {
                        TableCell { "{model}" }
                        TableCell { "{engine}" }
                        TableCell { "{load_label(c)}" }
                        TableCell { {c.num_tasks.map_or("—".to_string(), |n| crate::thousands(n as usize))} }
                        TableCell { "{scored(c)}" }
                        TableCell { "{by_hops(c).filter(|_| accuracy(c).is_some()).unwrap_or_default()}" }
                        TableCell { {c.capped_answers.map_or("—".to_string(), |k| crate::thousands(k as usize))} }
                        TableCell { "{joules(per_right(c))}" }
                    }
                }
            }
            }
        }
    }
}

/// Why a cell has no accuracy, short enough for its line on the plot: "331 of 400 cut off",
/// "268 of 2,415 answered"; the table view words it in full ([`right`]).
fn short_reason(c: &TaskCell) -> String {
    let done = c.completed.unwrap_or(0) as usize;
    let of = |k: Option<u32>| {
        format!(
            "{} of {}",
            crate::thousands(k.unwrap_or(0) as usize),
            crate::thousands(done)
        )
    };
    if mostly_failed(c) {
        let sent = done + c.failed.unwrap_or(0) as usize;
        format!(
            "{} of {} answered",
            crate::thousands(done),
            crate::thousands(sent)
        )
    } else if mostly_capped(c) {
        format!("{} cut off", of(c.capped_answers))
    } else if mostly_thought(c) {
        format!("reasoning {}", of(c.thinking_answers))
    } else {
        right(c)
    }
}

/// [`accuracy_table`]'s plot: a group a model, in the order the section reads them, a row a Mac
/// it ran on, a mark an engine, scratchy first: its accuracy and 95% range, or why it has none.
fn accuracy_dots(dataset: &str, pairs: &[(&str, &Model)]) -> Vec<RangeGroup> {
    let mut groups: Vec<RangeGroup> = Vec::new();
    for &(chip, m) in pairs {
        let mut engines = vec![Engine::Scratchy];
        engines.extend(rival_of(m, dataset));
        let marks: Vec<RangeMark> = (engines.iter().zip(["s1", "s2"]))
            .filter_map(|(&e, class)| {
                let c = (e.tasks(m).iter())
                    .find(|c| of_dataset(c, dataset) && !c.timed_answers && c.accuracy.is_some())?;
                let at = match (accuracy(c), range(c)) {
                    (Some(a), Some(r)) => Ok((a, a - r / 100.0, a + r / 100.0)),
                    (Some(a), None) => Ok((a, a, a)),
                    _ => Err(short_reason(c)),
                };
                let mut tip = vec![
                    format!("{} on {chip}", m.stem),
                    format!("accuracy {}", scored(c)),
                ];
                tip.extend(by_hops(c).filter(|_| accuracy(c).is_some()));
                if let Some(n) = c.num_tasks {
                    tip.push(format!("over {} tasks", crate::thousands(n as usize)));
                }
                Some(RangeMark {
                    class,
                    name: e.label().to_string(),
                    at,
                    tip,
                })
            })
            .collect();
        if marks.is_empty() {
            continue;
        }
        let row = (chip.to_string(), marks);
        match groups.iter_mut().find(|(stem, _)| stem == &m.stem) {
            Some((_, rows)) => rows.push(row),
            None => groups.push((m.stem.clone(), vec![row])),
        }
    }
    groups
}

/// [`tasks_table`]'s rows of one workload: its detail in a dataset's section. Nothing for a model
/// that did not run it.
pub fn detail_table(m: &Model, dataset: &str) -> Element {
    table(m, Some(dataset))
}

/// Whether a cell belongs in `dataset`'s table: its own, and for MuSiQue its accuracy cell too,
/// MuSiQue over more tasks.
fn of_dataset(c: &TaskCell, dataset: &str) -> bool {
    c.workload == dataset || (dataset == "rag" && c.workload == "rag-accuracy")
}

fn table(m: &Model, only: Option<&str>) -> Element {
    let ours: Vec<&TaskCell> = (Engine::Scratchy.tasks(m).iter())
        .filter(|c| only.is_none_or(|w| of_dataset(c, w)))
        .collect();
    if ours.is_empty() {
        return rsx! {};
    }
    let mut cells = ours;
    let rank = |w: &str| ORDER.iter().position(|o| *o == w).unwrap_or(ORDER.len());
    cells.sort_by_key(|c| (rank(&c.workload), c.conc));
    // The open-loop note, only where the run timed its cells open loop (`--open-loop`).
    let open_loop = cells.iter().any(|c| c.load.as_deref() == Some("rate"));
    let (headers, abouts) = columns();
    rsx! {
        Table {
            headers,
            abouts,
            for c in cells {
                {
                    let (name, _) = workload_label(&c.workload);
                    let them = rival(m, &c.workload, c.conc);
                    let ttft = them.and_then(|(_, r)| ttft_times(c, r));
                    let total = them.and_then(|(_, r)| total_times(c, r));
                    let spent = them.and_then(|(_, r)| energy_times(c, r));
                    // A borrowed rival says which run it is from, as the grid's borrowed engines do.
                    let vs = them.map_or_else(
                        || "—".to_string(),
                        |(e, r)| match &r.borrowed_from {
                            Some(when) => format!("{} (from {})", e.label(), when.month_day()),
                            None => e.label().to_string(),
                        },
                    );
                    rsx! {
                        TableRow {
                            TableCell { "{name}" }
                            TableCell { "{load_label(c)}" }
                            if c.measured() {
                                // A cell where some requests failed covers only the others: marked,
                                // as the random cells are.
                                TableCell { "{ms(c.median_ttft_ms)}{failed(c)}" }
                                TableCell { "{ms(c.median_e2el_ms)}" }
                                TableCell { "{n(c.answer_tokens())}" }
                                TableCell { "{n(c.mean_cached_tokens)}" }
                                TableCell { "{joules(per_request(c))}" }
                                TableCell { "{joules(per_token(c))}" }
                            } else {
                                TableCell { colspan: 6, "no answers" }
                            }
                            TableCell { "{vs}" }
                            // A rival that mostly failed says so, rather than leave its ratios blank.
                            if let Some(why) = them.and_then(|(_, r)| no_ratio(r)) {
                                TableCell { colspan: 3, "{why}" }
                            } else {
                                TableCell { "{x(ttft)}" }
                                TableCell { "{x(total)}" }
                                TableCell { "{x(spent)}" }
                            }
                        }
                    }
                }
            }
        }
        if open_loop {
            p { style: "font-size: 0.875rem",
                "Load: an open-loop cell offers its requests at a fixed rate, a fraction of scratchy's \
                 own one-user rate on that model and workload (1 user (base)), and every engine is \
                 offered the same rate. Engines are compared at equal traffic: one slower than \
                 scratchy at one user is offered more than it gets through alone, so it queues, and \
                 its first token comes later."
            }
        }
    }
}

/// The workloads a model's dataset cells cover, in the page's order: the overview's columns.
pub fn workloads(m: &Model) -> Vec<&str> {
    let mut ws: Vec<&str> = Engine::Scratchy
        .tasks(m)
        .iter()
        .map(|c| c.workload.as_str())
        .collect();
    ws.sort_by_key(|w| ORDER.iter().position(|o| o == w).unwrap_or(ORDER.len()));
    ws.dedup();
    ws
}

/// A cell's load as the page names it: "1 user (base)" for the closed-loop cell the rates are
/// fractions of, "0.42/s" for an open-loop one, "N users" for a closed-loop one (older runs, and
/// the accuracy cell).
pub fn load_label(c: &TaskCell) -> String {
    match (c.load.as_deref(), c.request_rate) {
        (Some("base"), _) => "1 user (base)".to_string(),
        (Some("rate"), Some(r)) => format!("{r:.2}/s"),
        _ if c.conc == 1 => "1 user".to_string(),
        _ => format!("{} users", c.conc),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Accuracy carries its task count, and under a hundred tasks reads as indicative.
    #[test]
    fn accuracy_says_how_many_tasks_it_is_over() {
        let mut c: TaskCell = serde_json::from_value(serde_json::json!({
            "workload": "rag", "conc": 1, "completed": 8, "accuracy": 0.375,
            "thinking_answers": 0, "capped_answers": 0, "num_tasks": 8,
        }))
        .unwrap();
        assert_eq!(right(&c), "~38% of 8");
        c.num_tasks = Some(100);
        assert_eq!(right(&c), "38% of 100");
        c.num_tasks = None;
        assert_eq!(right(&c), "~38%", "a run from before the count");
    }

    /// A dataset cell's energy, where its run kept it: joules a request, an output token and, on
    /// a scored cell that shows its accuracy, a right answer; and how many times less scratchy
    /// spent a request than the rival, only at about one answer length, as the total time is.
    #[test]
    fn a_dataset_cells_energy_is_a_request_a_token_and_a_right_answer() {
        let cell = |tokens: f64, j: f64, accuracy: Option<f64>| -> TaskCell {
            serde_json::from_value(serde_json::json!({
                "workload": "rag", "conc": 1, "completed": 8, "num_tasks": 8,
                "accuracy": accuracy, "thinking_answers": 0, "capped_answers": 0,
                "mean_output_tokens": tokens,
                "energy": {"soc_j": j * 8.0, "window_s": 20.0, "j_per_request": j,
                    "j_per_output_token": j / tokens,
                    "j_per_right_answer": accuracy.filter(|&a| a > 0.0).map(|a| j / a)},
            }))
            .unwrap()
        };
        let ours = cell(100.0, 40.0, Some(0.5));
        assert_eq!(joules(per_request(&ours)), "40.0");
        assert_eq!(joules(per_token(&ours)), "0.40");
        assert_eq!(joules(per_right(&ours)), "80.0");
        // A third less energy a request than the rival's: x1.50.
        assert_eq!(energy_times(&ours, &cell(100.0, 60.0, Some(0.5))), Some(1.5));
        // Answers of different lengths: no ratio, as for the total time.
        assert_eq!(energy_times(&ours, &cell(200.0, 60.0, Some(0.5))), None);
        // Not scored, or none right: no figure a right answer.
        assert_eq!(joules(per_right(&cell(100.0, 40.0, None))), "—");
        assert_eq!(joules(per_right(&cell(100.0, 40.0, Some(0.0)))), "—");
        // A run from before cells kept energy.
        let old: TaskCell = serde_json::from_value(serde_json::json!({
            "workload": "rag", "conc": 1, "completed": 8,
        }))
        .unwrap();
        assert_eq!(joules(per_request(&old)), "—");
    }

    /// A cell most of whose requests failed shows no accuracy, energy or ratio, on either side of
    /// a pair, and says how many of its requests answered.
    #[test]
    fn a_mostly_failed_cell_shows_no_accuracy_energy_or_ratio() {
        let cell = |completed: u32, failed: u32| -> TaskCell {
            serde_json::from_value(serde_json::json!({
                "workload": "rag-accuracy", "conc": 4, "completed": completed, "failed": failed,
                "num_tasks": completed + failed, "accuracy": 0.06, "thinking_answers": 0,
                "capped_answers": 0, "mean_output_tokens": 30.0,
                "median_ttft_ms": 100.0, "median_e2el_ms": 1000.0,
                "energy": {"soc_j": 6500.0, "window_s": 144.0, "j_per_request": 2.7,
                    "j_per_output_token": 0.09, "j_per_right_answer": 48.4},
            }))
            .unwrap()
        };
        let broken = cell(268, 2147);
        let whole = cell(2415, 0);
        assert_eq!(right(&broken), "268 of 2,415 answered");
        assert_eq!(joules(per_request(&broken)), "—");
        assert_eq!(joules(per_token(&broken)), "—");
        assert_eq!(joules(per_right(&broken)), "—");
        for (a, b) in [(&broken, &whole), (&whole, &broken)] {
            assert_eq!(ttft_times(a, b), None);
            assert_eq!(total_times(a, b), None);
            assert_eq!(energy_times(a, b), None);
        }
        // A few failures leave the rest standing.
        let few = cell(2400, 15);
        assert_eq!(right(&few), "6% of 2415");
        assert_eq!(ttft_times(&few, &whole), Some(1.0));
    }

    /// A MuSiQue cell that recorded its accuracy by hop count shows each beside the whole, in hop
    /// order; one from before shows the whole alone.
    #[test]
    fn a_cells_accuracy_shows_by_hops() {
        let mut c: TaskCell = serde_json::from_value(serde_json::json!({
            "workload": "rag-accuracy", "conc": 4, "completed": 400, "accuracy": 0.47,
            "thinking_answers": 0, "capped_answers": 0, "num_tasks": 400,
            "accuracy_by_group": {"4": 0.38, "2": 0.52, "3": 0.43},
        }))
        .unwrap();
        assert_eq!(right(&c), "47% of 400 (2-hop 52%, 3-hop 43%, 4-hop 38%)");
        c.accuracy_by_group = None;
        assert_eq!(right(&c), "47% of 400");
    }

    /// The whole answer's time is set against the rival's only at about the same answer length,
    /// and neither mostly capped; the first token's always is.
    #[test]
    fn a_total_ratio_needs_answers_of_about_one_length() {
        let cell = |tokens: f64, capped: u32| TaskCell {
            workload: "rag".into(),
            conc: 1,
            completed: Some(8),
            failed: Some(0),
            median_ttft_ms: Some(100.0),
            median_e2el_ms: Some(1000.0),
            mean_output_tokens: Some(tokens),
            total_output_tokens: None,
            mean_cached_tokens: None,
            accuracy: None,
            thinking_answers: Some(0),
            capped_answers: Some(capped),
            num_tasks: Some(8),
            load: None,
            request_rate: None,
            timed_answers: false,
            timed_out: false,
            median_tpot_ms: None,
            output_throughput: None,
            ttfts_ms: None,
            e2els_ms: None,
            prompt_tokens: None,
            tpots_ms: None,
            accuracy_by_group: None,
            borrowed_from: None,
            energy: None,
        };
        assert_eq!(total_times(&cell(100.0, 0), &cell(105.0, 0)), Some(1.0));
        assert_eq!(total_times(&cell(100.0, 0), &cell(150.0, 0)), None, "longer");
        assert_eq!(total_times(&cell(100.0, 0), &cell(100.0, 5)), None, "capped");
        assert_eq!(ttft_times(&cell(100.0, 0), &cell(50.0, 0)), Some(1.0), "any length");
        assert_eq!(ttft_times(&cell(100.0, 0), &cell(100.0, 5)), None, "a capped rival");
        let mut unknown = cell(100.0, 0);
        unknown.mean_output_tokens = None;
        assert_eq!(total_times(&cell(100.0, 0), &unknown), None, "no length");
    }

    /// A run's workloads, in the order it ran them.
    #[test]
    fn a_runs_workloads_are_read_in_its_order() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/metal/apple-m3-pro-2026-10-26T043309Z-2a9661c3.json"
        );
        let run: super::super::data::Run =
            serde_json::from_str(&std::fs::read_to_string(path).expect("the oMLX run")).unwrap();
        let m = &run.models[0];
        assert_eq!(
            workloads(m),
            ["sharegpt", "rag", "rag-spans", "sharegpt-multi-turn"]
        );
    }

    /// A cell is named by its load: the closed-loop base, an open-loop rate, or (older runs, and
    /// the accuracy cell) a users count.
    #[test]
    fn a_cell_is_named_by_its_load() {
        let cell = |load: Option<&str>, conc: u32, rate: Option<f64>| -> TaskCell {
            serde_json::from_value(serde_json::json!({
                "workload": "rag", "conc": conc, "load": load, "request_rate": rate,
            }))
            .unwrap()
        };
        assert_eq!(load_label(&cell(Some("base"), 0, None)), "1 user (base)");
        assert_eq!(load_label(&cell(Some("rate"), 2, Some(0.4152))), "0.42/s");
        assert_eq!(load_label(&cell(None, 4, None)), "4 users", "a run from before rates");
        assert_eq!(load_label(&cell(None, 1, None)), "1 user");
    }

    /// An accuracy's 95% range: the binomial interval over its tasks, in points.
    #[test]
    fn an_accuracy_carries_its_range() {
        let cell = |acc: f64, n: u32| -> TaskCell {
            serde_json::from_value(serde_json::json!({"workload": "rag-accuracy", "conc": 4,
                "completed": n, "num_tasks": n, "accuracy": acc, "thinking_answers": 0,
                "capped_answers": 0}))
            .unwrap()
        };
        // 1.96 * sqrt(0.25 * 0.75 / 400) = 4.2 points; over 2,415 tasks, 1.7.
        assert_eq!(range(&cell(0.25, 400)), Some(4.2));
        assert_eq!(range(&cell(0.25, 2415)), Some(1.7));
        assert_eq!(scored(&cell(0.25, 400)), "25% ±4.2");
        // One decimal always: 1.96 * sqrt(0.5 * 0.5 / 2401) = 2.0 points.
        assert_eq!(scored(&cell(0.5, 2401)), "50% ±2.0");
    }

    /// A held-back accuracy says why in its cell, and one reasoning answer among many does not
    /// hide a score: only most of them do, as for capped answers.
    #[test]
    fn a_held_back_accuracy_says_why() {
        let cell = |capped: u32, thinking: u32| -> TaskCell {
            serde_json::from_value(serde_json::json!({
                "workload": "rag-accuracy", "conc": 4, "completed": 400, "num_tasks": 400,
                "accuracy": 0.47, "capped_answers": capped, "thinking_answers": thinking,
            }))
            .unwrap()
        };
        assert_eq!(
            right(&cell(331, 0)),
            "cut off: 331 of 400 answers hit the length limit"
        );
        assert_eq!(right(&cell(0, 378)), "answered in reasoning: 378 of 400");
        assert_eq!(
            right(&cell(0, 1)),
            "47% of 400",
            "one reasoning answer in 400"
        );
        let timed: TaskCell = serde_json::from_value(serde_json::json!({
            "workload": "rag", "conc": 1, "completed": 8,
        }))
        .unwrap();
        assert_eq!(right(&timed), "not scored");
    }

    #[test]
    fn times_faster_is_rival_over_scratchy_for_a_time() {
        assert_eq!(times(Some(500.0), Some(1000.0), true), Some(2.0));
        assert_eq!(times(Some(200.0), Some(100.0), false), Some(2.0));
        assert_eq!(times(None, Some(1.0), true), None);
        assert_eq!(times(Some(0.0), Some(1.0), true), None);
    }

    #[test]
    fn a_completions_cell_takes_its_answer_tokens_from_its_total() {
        let c: TaskCell = serde_json::from_value(serde_json::json!({
            "workload": "sharegpt", "conc": 1, "completed": 8, "median_ttft_ms": 165.0,
            "total_output_tokens": 2432,
        }))
        .unwrap();
        assert_eq!(c.answer_tokens(), Some(304.0));
    }

    #[test]
    fn a_thinking_answer_has_no_comparable_accuracy() {
        let cell = |thinking: u32| TaskCell {
            workload: "rag-spans".into(),
            conc: 1,
            completed: Some(4),
            failed: Some(0),
            median_ttft_ms: Some(1.0),
            median_e2el_ms: Some(2.0),
            mean_output_tokens: Some(10.0),
            total_output_tokens: Some(40),
            mean_cached_tokens: None,
            accuracy: Some(0.5),
            thinking_answers: Some(thinking),
            capped_answers: Some(0),
            num_tasks: Some(4),
            load: None,
            request_rate: None,
            timed_answers: false,
            timed_out: false,
            median_tpot_ms: None,
            output_throughput: None,
            ttfts_ms: None,
            e2els_ms: None,
            prompt_tokens: None,
            tpots_ms: None,
            accuracy_by_group: None,
            borrowed_from: None,
            energy: None,
        };
        assert_eq!(accuracy(&cell(0)), Some(0.5));
        assert_eq!(accuracy(&cell(4)), None);
        let mut capped = cell(0);
        capped.capped_answers = Some(4);
        assert_eq!(accuracy(&capped), None, "every answer ran to its cap");
        capped.capped_answers = Some(3);
        assert_eq!(accuracy(&capped), None, "most ran to their cap");
        capped.capped_answers = Some(2);
        assert_eq!(accuracy(&capped), Some(0.5), "half ended on their own");
    }
}

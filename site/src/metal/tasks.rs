//! The real-task view: each dataset workload a model ran, scratchy against the engine it is
//! compared with, on the same tasks (`scr bench serve --dataset-name <workload>`).

use dioxus::prelude::*;

use crate::carbon::{Table, TableCell, TableRow};

use super::data::{Model, RequestTiming, TaskCell};
use super::numbers::Engine;

/// A workload's name on the page, and what it measures.
pub fn workload_label(w: &str) -> (&'static str, &'static str) {
    match w {
        "sharegpt" => ("ShareGPT chat", "real chat prompts, one turn each"),
        "rag" => (
            "MuSiQue",
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
            "MuSiQue accuracy",
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

/// Whether most of a cell's answers opened as reasoning, scored as answers they are not.
fn mostly_thinking(c: &TaskCell) -> bool {
    c.completed
        .is_some_and(|n| n > 0 && 2 * c.thinking_answers.unwrap_or(0) > n)
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

/// Accuracy, where it is like for like: none for a cell most of whose answers opened as
/// reasoning, or ran to their cap, which read as wrong without being answers, or most of whose
/// requests failed.
fn accuracy(c: &TaskCell) -> Option<f64> {
    c.accuracy.filter(|_| !mostly_thinking(c) && usable(c))
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

/// What a cell's energy figures cover, said wherever they are shown.
pub const ENERGY_NOTE: &str = "J/request and J/token: the energy of the cell's requests, over \
     every request sent and every output token, from the CPU, GPU and ANE counters, not DRAM, \
     the Mac's idle floor of about 1.6 W in it; read once a second, so a cell of under about \
     20 s is coarse.";

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
        return held_back(c).unwrap_or_else(|| "—".to_string());
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

/// Why a cell shows no accuracy, where the page knows: a timed cell's answers are forced to one
/// length and never scored; a scored cell most of whose answers ran to their cap, or opened as
/// reasoning, is held back, with how many.
fn held_back(c: &TaskCell) -> Option<String> {
    if c.timed_answers {
        return Some("timed, not scored".to_string());
    }
    c.accuracy?;
    let of = c.num_tasks.or(c.completed).unwrap_or(0);
    if mostly_capped(c) {
        return Some(format!("ran to its cap: {} of {of}", c.capped_answers.unwrap_or(0)));
    }
    let thinking = c.thinking_answers.unwrap_or(0);
    mostly_thinking(c).then(|| format!("opened as reasoning: {thinking} of {of}"))
}

/// Why a pair's cells have no ratio, where it is the rival's doing: most of its requests failed.
/// None where the ratios stand or are left out for the reasons the notes give.
fn no_ratio(_ours: &TaskCell, theirs: &TaskCell) -> Option<String> {
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
    rsx! {
        Table {
            headers: [
                "workload", "load", "TTFT", "total", "answer tok", "cached tok", "accuracy",
                "J/request", "J/token", "J/right answer", "vs", "TTFT", "total", "J/request",
            ]
            .map(String::from)
            .to_vec(),
            for c in cells {
                {
                    let (name, _) = workload_label(&c.workload);
                    let them = rival(m, &c.workload, c.conc);
                    let ttft = them.and_then(|(_, r)| ttft_times(c, r));
                    let total = them.and_then(|(_, r)| total_times(c, r));
                    let spent = them.and_then(|(_, r)| energy_times(c, r));
                    let why = them.and_then(|(_, r)| no_ratio(c, r));
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
                                TableCell { "{right(c)}" }
                                TableCell { "{joules(per_request(c))}" }
                                TableCell { "{joules(per_token(c))}" }
                                TableCell { "{joules(per_right(c))}" }
                            } else {
                                TableCell { colspan: 8, "no answers" }
                            }
                            TableCell { "{vs}" }
                            if let Some(why) = why {
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
    }
}

/// What a dataset's table columns mean, said once under the section's summary, not under every
/// pair's table: accuracy and why it is held back, energy, and, where any cell shown was timed
/// open loop (`--open-loop`), its load.
pub fn table_notes(open_loop: bool) -> Element {
    rsx! {
        p { style: "font-size: 0.875rem",
            "Accuracy: the share of tasks answered right, a failed request or an answer that ran to \
             its cap counted wrong; none on a cell whose answers are forced to one length, so its \
             times compare across engines, as every cell at a users count is; left out where most answers ran to their cap or opened as reasoning, the cell saying which \
             and how many. A cell most of whose requests failed shows how many answered, and no accuracy, energy or ratio on either side. ~ marks one over fewer \
             than {FIRM_ACCURACY_TASKS} tasks, where one task moves it by 100/n points, so it is \
             indicative only. The second total ratio is shown only where both engines' answers were \
             about one length."
        }
        p { style: "font-size: 0.875rem", {ENERGY_NOTE} " J/right answer: over the right answers, \
             where the cell is scored. Its ratio is how many times less energy scratchy spent a request, \
             shown only where both engines' answers were about one length, an engine that stops \
             early spending less for answering less."
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

/// Whether any of a model's `dataset` cells was timed open loop: the load note's condition.
pub fn open_loop(m: &Model, dataset: &str) -> bool {
    (Engine::Scratchy.tasks(m).iter()).any(|c| of_dataset(c, dataset) && c.load.as_deref() == Some("rate"))
}

/// One dataset overview square: the users count, the workload, and how many times faster
/// scratchy is to its first token (None where either side has no number).
pub type GlanceSquare = (u32, String, Option<f64>);

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

/// The label of a load key on the overview: the load of scratchy's own cell at that key.
pub fn key_label(m: &Model, key: u32) -> String {
    (Engine::Scratchy.tasks(m).iter())
        .find(|c| c.conc == key && compared(c))
        .map_or_else(|| key.to_string(), load_label)
}

/// Whether a cell's numbers are set against a rival's: every open-loop or closed-loop users
/// cell, not the base, which only scratchy runs.
fn compared(c: &TaskCell) -> bool {
    c.load.as_deref() != Some("base")
}

/// One model's dataset overview map: a square a (load, workload), how many times faster
/// scratchy is to its first token than the rival, None where either side has no number, and the
/// rival it is set against (the first that ran any of these cells). The map is shown only with a
/// rival: scratchy alone has nothing to be faster than.
pub fn glance_squares(m: &Model) -> Option<(Engine, Vec<GlanceSquare>)> {
    squares(m, None)
}

fn squares(m: &Model, only: Option<&str>) -> Option<(Engine, Vec<GlanceSquare>)> {
    let ours = Engine::Scratchy.tasks(m);
    let ws: Vec<&str> = (workloads(m).into_iter())
        .filter(|w| only.is_none_or(|o| o == *w))
        .collect();
    let mut users: Vec<u32> = (ours.iter())
        .filter(|c| compared(c) && only.is_none_or(|o| c.workload == o))
        .map(|c| c.conc)
        .collect();
    users.sort_unstable();
    users.dedup();
    // One dataset's squares are set against its own rival, as its pair is ([`rival_of`]); the
    // whole map against the first that measured any.
    let rival = match only {
        Some(dataset) => rival_of(m, dataset)?,
        None => Engine::RIVALS
            .into_iter()
            .find(|&e| e.tasks(m).iter().any(TaskCell::measured))?,
    };
    let squares = (users.iter())
        .flat_map(|&u| ws.iter().map(move |&w| (u, w)))
        .map(|(u, w)| {
            let mine = ours
                .iter()
                .find(|c| c.workload == w && c.conc == u && c.measured());
            let theirs =
                (rival.tasks(m).iter()).find(|c| c.workload == w && c.conc == u && c.measured());
            let r = mine
                .zip(theirs)
                .and_then(|(a, b)| ttft_times(a, b));
            (u, w.to_string(), r)
        })
        .collect();
    Some((rival, squares))
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

    /// A cell with no accuracy says why in its place: a timed cell is never scored, and a scored
    /// one held back says how many answers ran to their cap or opened as reasoning.
    #[test]
    fn a_held_back_accuracy_says_why() {
        let cell = |v: serde_json::Value| -> TaskCell { serde_json::from_value(v).unwrap() };
        let timed = cell(serde_json::json!({"workload": "rag", "conc": 1, "completed": 8,
            "timed_answers": true, "thinking_answers": 8}));
        assert_eq!(right(&timed), "timed, not scored");
        let capped = cell(serde_json::json!({"workload": "rag-accuracy", "conc": 4,
            "completed": 400, "num_tasks": 400, "accuracy": 0.1, "capped_answers": 331,
            "thinking_answers": 0}));
        assert_eq!(right(&capped), "ran to its cap: 331 of 400");
        let thought = cell(serde_json::json!({"workload": "rag-accuracy", "conc": 4,
            "completed": 400, "num_tasks": 400, "accuracy": 0.1, "capped_answers": 10,
            "thinking_answers": 378}));
        assert_eq!(right(&thought), "opened as reasoning: 378 of 400");
        // Both: the cap first, the one a reader can act on.
        let both = cell(serde_json::json!({"workload": "rag-accuracy", "conc": 4,
            "completed": 400, "num_tasks": 400, "accuracy": 0.1, "capped_answers": 331,
            "thinking_answers": 378}));
        assert_eq!(right(&both), "ran to its cap: 331 of 400");
        // One answer of many opening as reasoning holds nothing back.
        let one = cell(serde_json::json!({"workload": "rag-accuracy", "conc": 4,
            "completed": 2415, "num_tasks": 2415, "accuracy": 0.4766, "capped_answers": 43,
            "thinking_answers": 1}));
        assert_eq!(right(&one), "48% of 2415");
        // A cell from before timed cells said so, with no accuracy: a dash, as before.
        let old = cell(serde_json::json!({"workload": "rag", "conc": 1, "completed": 8}));
        assert_eq!(right(&old), "—");
    }

    /// The ratio against a rival most of whose requests failed says so, in place of a ratio.
    #[test]
    fn a_ratio_against_a_failed_rival_says_why() {
        let cell = |done: u32, failed: u32| -> TaskCell {
            serde_json::from_value(serde_json::json!({"workload": "rag-accuracy", "conc": 4,
                "completed": done, "failed": failed, "median_ttft_ms": 100.0}))
            .unwrap()
        };
        assert_eq!(
            no_ratio(&cell(2415, 0), &cell(9, 2406)).as_deref(),
            Some("rival: 9 of 2,415 answered")
        );
        assert_eq!(no_ratio(&cell(2415, 0), &cell(2415, 0)), None);
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

    /// The overview's map of a run that timed scratchy and oMLX TurboQuant on the same tasks: a
    /// square a users count and workload, how many times faster scratchy is to its first token;
    /// rag-spans, scratchy's own, has no rival and so no ratio.
    #[test]
    fn the_overview_sets_scratchy_against_the_rival_on_each_workload() {
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
        let (rival, squares) = glance_squares(m).expect("a rival ran");
        assert_eq!(rival.key(), "omlx_tq");
        let at = |u: u32, w: &str| {
            squares
                .iter()
                .find(|(su, sw, _)| *su == u && sw == w)
                .unwrap()
                .2
        };
        // ShareGPT at 1 user: 167 ms against 280 ms.
        assert!((at(1, "sharegpt").unwrap() - 280.0 / 167.0).abs() < 0.05);
        assert!(at(1, "rag-spans").is_none(), "scratchy's own: no rival");
        assert_eq!(squares.len(), 2 * 4, "two users counts by four workloads");
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
        assert!(!compared(&cell(Some("base"), 0, None)), "the base is scratchy's only");
        assert!(compared(&cell(Some("rate"), 1, Some(0.2))));
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

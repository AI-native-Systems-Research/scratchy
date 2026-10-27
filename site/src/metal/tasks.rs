//! The real-task view: each dataset workload a model ran, scratchy against the engine it is
//! compared with, on the same tasks (`scr bench serve --dataset-name <workload>`).

use dioxus::prelude::*;

use crate::carbon::{Table, TableCell, TableRow};

use super::data::{Model, TaskCell};
use super::numbers::Engine;

/// A workload's name on the page, and what it measures.
pub fn workload_label(w: &str) -> (&'static str, &'static str) {
    match w {
        "sharegpt" => ("ShareGPT chat", "real chat prompts, one turn each"),
        "rag" => (
            "RAG",
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
        _ => ("dataset", ""),
    }
}

/// The page's order: the quick run's workloads first.
const ORDER: [&str; 7] = [
    "sharegpt",
    "rag",
    "rag-spans",
    "sharegpt-multi-turn",
    "longbench",
    "shared-prefix",
    "shared-prefix-control",
];

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

/// Accuracy, where it is like for like: none for a cell whose answers opened as reasoning, or one
/// most of whose answers ran to their cap, which read as wrong without being answers.
fn accuracy(c: &TaskCell) -> Option<f64> {
    c.accuracy
        .filter(|_| c.thinking_answers.unwrap_or(0) == 0 && !mostly_capped(c))
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
    (close && !mostly_capped(c) && !mostly_capped(r))
        .then(|| times(c.median_e2el_ms, r.median_e2el_ms, true))
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

/// How many times better scratchy is: `rival / scratchy` for a time (lower is better), `scratchy
/// / rival` for cached tokens. None where either side has no number.
pub fn times(scratchy: Option<f64>, rival: Option<f64>, lower_better: bool) -> Option<f64> {
    let (s, r) = (scratchy?, rival?);
    (s > 0.0 && r > 0.0).then(|| if lower_better { r / s } else { s / r })
}

fn x(v: Option<f64>) -> String {
    v.map_or_else(String::new, |v| format!("×{v:.2}"))
}

/// A model's real-task table, one row a workload and users count: scratchy's numbers, then the
/// rival's, and how many times faster scratchy is to its first token and to its whole answer.
/// Nothing for a run without dataset workloads.
pub fn tasks_table(m: &Model) -> Element {
    let ours = Engine::Scratchy.tasks(m);
    if ours.is_empty() {
        return rsx! {};
    }
    let mut cells: Vec<&TaskCell> = ours.iter().collect();
    let rank = |w: &str| ORDER.iter().position(|o| *o == w).unwrap_or(ORDER.len());
    cells.sort_by_key(|c| (rank(&c.workload), c.conc));
    rsx! {
        Table {
            headers: [
                "workload", "users", "TTFT", "total", "answer tok", "cached tok", "right",
                "vs", "TTFT", "total",
            ]
            .map(String::from)
            .to_vec(),
            for c in cells {
                {
                    let (name, _) = workload_label(&c.workload);
                    let them = rival(m, &c.workload, c.conc);
                    let ttft = them.and_then(|(_, r)| times(c.median_ttft_ms, r.median_ttft_ms, true));
                    let total = them.and_then(|(_, r)| total_times(c, r));
                    let vs = them.map_or_else(|| "—".to_string(), |(e, _)| e.label().to_string());
                    rsx! {
                        TableRow {
                            TableCell { "{name}" }
                            TableCell { "{c.conc}" }
                            if c.measured() {
                                // A cell where some requests failed covers only the others: marked,
                                // as the random cells are.
                                TableCell { "{ms(c.median_ttft_ms)}{failed(c)}" }
                                TableCell { "{ms(c.median_e2el_ms)}" }
                                TableCell { "{n(c.answer_tokens())}" }
                                TableCell { "{n(c.mean_cached_tokens)}" }
                                TableCell { "{pct(accuracy(c))}" }
                            } else {
                                TableCell { colspan: 5, "no answers" }
                            }
                            TableCell { "{vs}" }
                            TableCell { "{x(ttft)}" }
                            TableCell { "{x(total)}" }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            timed_out: false,
        };
        assert_eq!(total_times(&cell(100.0, 0), &cell(105.0, 0)), Some(1.0));
        assert_eq!(total_times(&cell(100.0, 0), &cell(150.0, 0)), None, "longer");
        assert_eq!(total_times(&cell(100.0, 0), &cell(100.0, 5)), None, "capped");
        let mut unknown = cell(100.0, 0);
        unknown.mean_output_tokens = None;
        assert_eq!(total_times(&cell(100.0, 0), &unknown), None, "no length");
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
            timed_out: false,
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

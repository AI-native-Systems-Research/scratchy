//! What the page computes from a run: per-engine accessors, medians, ratios,
//! and how each number is written.

use super::data::{At, Cell, Model, Rep, Scenario};
use crate::thousands_f;

/// The engines a run compares, in page order. Every chart pairs scratchy
/// (its own colour) with one other engine, so the colours never run out
/// however many engines a run includes.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Engine {
    Scratchy,
    OmlxTq,
    Omlx,
    MlxLm,
    VllmMetal,
    Ollama,
    LlamaCpp,
    Mistralrs,
}

impl Engine {
    pub const ALL: [Engine; 8] = [
        Engine::Scratchy,
        Engine::OmlxTq,
        Engine::Omlx,
        Engine::MlxLm,
        Engine::VllmMetal,
        Engine::Ollama,
        Engine::LlamaCpp,
        Engine::Mistralrs,
    ];
    /// The engines scratchy is compared against, preferred first for the
    /// headline comparison: oMLX TurboQuant (4-bit KV, as scratchy's
    /// default), then oMLX as shipped, then the others. Runs from before oMLX
    /// fall back to mlx-lm, then ollama, as they always did.
    pub const RIVALS: [Engine; 7] = [
        Engine::OmlxTq,
        Engine::Omlx,
        Engine::MlxLm,
        Engine::VllmMetal,
        Engine::Ollama,
        Engine::LlamaCpp,
        Engine::Mistralrs,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Engine::Scratchy => "scratchy",
            Engine::OmlxTq => "oMLX TurboQuant",
            Engine::Omlx => "oMLX",
            Engine::MlxLm => "mlx-lm",
            Engine::VllmMetal => "vllm-metal",
            Engine::Ollama => "ollama",
            Engine::LlamaCpp => "llama.cpp",
            Engine::Mistralrs => "mistral.rs",
        }
    }

    /// Its key in the run file (`cache_ladder_<key>`, `engine_models`).
    pub fn key(self) -> &'static str {
        match self {
            Engine::Scratchy => "scratchy",
            Engine::OmlxTq => "omlx_tq",
            Engine::Omlx => "omlx",
            Engine::MlxLm => "mlx_lm",
            Engine::VllmMetal => "vllm_metal",
            Engine::Ollama => "ollama",
            Engine::LlamaCpp => "llama_cpp",
            Engine::Mistralrs => "mistralrs",
        }
    }

    pub fn ladder(self, m: &Model) -> &[Rep] {
        match self {
            Engine::Scratchy => &m.cache_ladder,
            Engine::OmlxTq => &m.cache_ladder_omlx_tq,
            Engine::Omlx => &m.cache_ladder_omlx,
            Engine::MlxLm => &m.cache_ladder_mlx_lm,
            Engine::VllmMetal => &m.cache_ladder_vllm_metal,
            Engine::Ollama => &m.cache_ladder_ollama,
            Engine::LlamaCpp => &m.cache_ladder_llama_cpp,
            Engine::Mistralrs => &m.cache_ladder_mistralrs,
        }
        .as_deref()
        .unwrap_or_default()
    }

    pub fn cells(self, m: &Model) -> &[Cell] {
        match self {
            Engine::Scratchy => &m.scaling,
            Engine::OmlxTq => &m.scaling_omlx_tq,
            Engine::Omlx => &m.scaling_omlx,
            Engine::MlxLm => &m.scaling_mlx_lm,
            Engine::VllmMetal => &m.scaling_vllm_metal,
            Engine::Ollama => &m.scaling_ollama,
            Engine::LlamaCpp => &m.scaling_llama_cpp,
            Engine::Mistralrs => &m.scaling_mistralrs,
        }
        .as_deref()
        .unwrap_or_default()
    }

    /// Whether this run measured the engine for this model at all.
    pub fn ran(self, m: &Model) -> bool {
        !self.ladder(m).is_empty() || !self.cells(m).is_empty()
    }
}

/// A startup measurement of one launch.
#[derive(Clone, Copy, PartialEq)]
pub enum Startup {
    /// Launch until the server reports ready.
    Ready,
    /// Request sent until its first token.
    FirstToken,
    PeakRss,
}

impl Startup {
    fn of(self, r: &Rep) -> Option<f64> {
        match self {
            Startup::Ready => r.t_ready_s,
            Startup::FirstToken => r.ttft_from_send_s,
            Startup::PeakRss => r.peak_rss_mib,
        }
    }
}

/// A serving measurement of one cell.
#[derive(Clone, Copy, PartialEq)]
pub enum Metric {
    Ttft,
    Tpot,
    Throughput,
    /// The server's peak memory footprint during the cell.
    PeakMem,
}

impl Metric {
    /// TTFT and TPOT need a streamed answer to mean anything; tok/s doesn't.
    fn is_timing(self) -> bool {
        matches!(self, Metric::Ttft | Metric::Tpot)
    }

    fn raw(self, c: &Cell) -> Option<f64> {
        match self {
            Metric::Ttft => c.median_ttft_ms,
            Metric::Tpot => c.median_tpot_ms,
            Metric::Throughput => c.output_throughput,
            Metric::PeakMem => c.server_mem_peak_mib,
        }
    }

    /// Decimals it is written with.
    pub fn decimals(self) -> usize {
        match self {
            Metric::Ttft | Metric::PeakMem => 0,
            Metric::Tpot | Metric::Throughput => 1,
        }
    }

    pub fn unit(self) -> &'static str {
        match self {
            Metric::Throughput => "tok/s",
            Metric::Ttft | Metric::Tpot => "ms",
            Metric::PeakMem => "MiB",
        }
    }

    /// How many times faster scratchy is than another engine at this metric:
    /// throughput is better higher, time better lower.
    pub fn faster(self, scratchy: Option<f64>, other: Option<f64>) -> Option<f64> {
        let (s, o) = (
            scratchy.filter(|v| *v != 0.0)?,
            other.filter(|v| *v != 0.0)?,
        );
        Some(match self {
            Metric::Throughput => s / o,
            Metric::Ttft | Metric::Tpot | Metric::PeakMem => o / s,
        })
    }
}

pub fn median(mut vals: Vec<f64>) -> Option<f64> {
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(f64::total_cmp);
    let n = vals.len();
    Some(if n % 2 == 1 {
        vals[n / 2]
    } else {
        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
    })
}

/// The median of one startup measurement over a ladder's launches of one scenario.
pub fn med(ladder: &[Rep], scenario: Scenario, what: Startup) -> Option<f64> {
    median(
        ladder
            .iter()
            .filter(|r| r.scenario == scenario)
            .filter_map(|r| what.of(r))
            .collect(),
    )
}

pub fn cell(cells: &[Cell], at: At) -> Option<&Cell> {
    cells.iter().find(|c| c.at == at)
}

pub const DASH: &str = "—";
pub const NO_STREAM: &str = "no stream";
/// Marks a timing taken from only some of a cell's requests.
pub const PARTIAL: &str = "†";
/// Marks any number from a cell where some requests failed.
pub const FAILED: &str = "‡";

pub fn fmt(v: Option<f64>, decimals: usize) -> String {
    v.map_or_else(|| DASH.to_string(), |v| thousands_f(v, decimals))
}

/// A startup median, or why there is none: launches that reached ready but
/// never streamed a token say so instead of showing a dash.
pub fn lfmt(
    ladder: &[Rep],
    scenario: Scenario,
    what: Startup,
    decimals: usize,
    scale: f64,
) -> String {
    let v = med(ladder, scenario, what);
    if v.is_none() && what != Startup::Ready && med(ladder, scenario, Startup::Ready).is_some() {
        return "no first token".to_string();
    }
    fmt(v.map(|v| v * scale), decimals)
}

/// Whether scratchy produced any number for this model.
pub fn served(m: &Model) -> bool {
    m.built
        && (!Engine::Scratchy.ladder(m).is_empty()
            || m.warm_serving.as_ref().is_some_and(|w| !w.is_empty())
            || !Engine::Scratchy.cells(m).is_empty())
}

/// Whether a cell's answer actually streamed. A zero median TPOT means the
/// tokens arrived in one piece (or there was only one), so neither timing was
/// measured: its TTFT is the whole answer's time, not the first token's.
/// ollama's gemma4 does this.
pub fn streamed(c: Option<&Cell>) -> bool {
    c.is_some_and(|c| c.median_tpot_ms.unwrap_or(0.0) > 0.0)
}

/// A cell's value, or None where it was not really measured.
pub fn timing(c: Option<&Cell>, metric: Metric) -> Option<f64> {
    let c = c?;
    if metric.is_timing() && !streamed(Some(c)) {
        return None;
    }
    metric.raw(c)
}

fn untimed(c: Option<&Cell>) -> u32 {
    c.and_then(|c| c.unstreamed_requests).unwrap_or(0)
}

/// Whether a timing is real but taken from only some of the cell's requests.
pub fn partial(c: Option<&Cell>, metric: Metric) -> bool {
    metric.is_timing() && streamed(c) && untimed(c) > 0
}

/// Requests in a cell that errored or timed out.
pub fn failed(c: Option<&Cell>) -> u32 {
    c.and_then(|c| c.failed).unwrap_or(0)
}

pub fn failed_note(c: Option<&Cell>) -> String {
    let n = failed(c);
    format!(
        "{n} of {} requests failed",
        n + c.and_then(|c| c.completed).unwrap_or(0)
    )
}

/// The marks a cell's value carries: † for a timing from only some of its
/// requests, ‡ for any number from a cell where requests failed.
pub fn marks(c: Option<&Cell>, metric: Metric) -> String {
    let mut m = String::new();
    if partial(c, metric) {
        m += PARTIAL;
    }
    if failed(c) > 0 {
        m += FAILED;
    }
    m
}

/// What a cell's marks mean, for a table view or a tooltip; "" without any.
pub fn marks_note(c: Option<&Cell>, metric: Metric) -> String {
    let mut parts = Vec::new();
    if partial(c, metric) {
        parts.push(untimed_note(c));
    }
    if failed(c) > 0 {
        parts.push(failed_note(c));
    }
    parts.join("; ")
}

pub fn untimed_note(c: Option<&Cell>) -> String {
    let completed = c
        .and_then(|c| c.completed)
        .map_or("?".to_string(), |n| n.to_string());
    format!("{} of {completed} untimed", untimed(c))
}

/// A cell's value as text: says "no stream" rather than show a timing the
/// engine never produced, and marks one taken from only some requests, or
/// from a cell where requests failed.
pub fn cfmt(c: Option<&Cell>, metric: Metric) -> String {
    if c.is_some() && metric.is_timing() && !streamed(c) {
        return NO_STREAM.to_string() + if failed(c) > 0 { FAILED } else { "" };
    }
    fmt(timing(c, metric), metric.decimals()) + &marks(c, metric)
}

/// Run-to-run spread on one machine at one commit: two M1 Max runs of af009bf1
/// agreed within about 2%, so a smaller change is noise, not a result.
pub const NOISE: f64 = 0.03;

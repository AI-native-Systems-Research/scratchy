//! What the page computes from a run: per-engine accessors, medians, ratios,
//! and how each number is written.

use super::data::{Cell, Model, Rep, Scenario};
use crate::charts::{Better, Side};
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
    Splash,
}

impl Engine {
    pub const ALL: [Engine; 9] = [
        Engine::Scratchy,
        Engine::OmlxTq,
        Engine::Omlx,
        Engine::MlxLm,
        Engine::VllmMetal,
        Engine::Ollama,
        Engine::LlamaCpp,
        Engine::Mistralrs,
        Engine::Splash,
    ];
    /// The engines scratchy is compared against, preferred first for the
    /// headline comparison: oMLX TurboQuant (4-bit KV, as scratchy's
    /// default), then oMLX as shipped, then the others. Runs from before oMLX
    /// fall back to mlx-lm, then ollama, as they always did. Splash comes last:
    /// it serves only the Qwen family, so it is never the headline rival.
    pub const RIVALS: [Engine; 8] = [
        Engine::OmlxTq,
        Engine::Omlx,
        Engine::MlxLm,
        Engine::VllmMetal,
        Engine::Ollama,
        Engine::LlamaCpp,
        Engine::Mistralrs,
        Engine::Splash,
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
            Engine::Splash => "Splash",
        }
    }

    /// Its sparkline fill's name, where it is the faster of two.
    pub fn ahead(self) -> &'static str {
        match self {
            Engine::Scratchy => "scratchy ahead",
            Engine::OmlxTq => "oMLX TurboQuant ahead",
            Engine::Omlx => "oMLX ahead",
            Engine::MlxLm => "mlx-lm ahead",
            Engine::VllmMetal => "vllm-metal ahead",
            Engine::Ollama => "ollama ahead",
            Engine::LlamaCpp => "llama.cpp ahead",
            Engine::Mistralrs => "mistral.rs ahead",
            Engine::Splash => "Splash ahead",
        }
    }

    /// Its side of a comparison chart: its line's name, and its fill's where
    /// it is the faster of two.
    pub fn side(self) -> Side {
        Side {
            line: self.label(),
            fill: self.ahead(),
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
            Engine::Splash => "splash",
        }
    }

    /// Whether it serves an `ignore_eos` request to its full length, so a live server delivers
    /// every answer token asked for ([`super::data::Run::mark_dead`]). In every published run,
    /// scratchy's live cells deliver 99% or more of them. mlx_lm.server has no `ignore_eos` and
    /// stops at EOS: granite-3.3-2b's 128x512 cell delivers 89% on it, 100% on scratchy. oMLX stops
    /// at EOS regardless (#479), as does ollama, at as little as 36% and 15%; vllm-metal, llama.cpp
    /// and mistral.rs have no published cells to show either, so they are not judged. Splash takes
    /// `ignore_eos` by its docs, but has no published cells yet, so it is not judged either.
    pub fn honours_ignore_eos(self) -> bool {
        match self {
            Engine::Scratchy => true,
            Engine::MlxLm
            | Engine::OmlxTq
            | Engine::Omlx
            | Engine::VllmMetal
            | Engine::Ollama
            | Engine::LlamaCpp
            | Engine::Mistralrs
            | Engine::Splash => false,
        }
    }

    /// Its scaling cells, to mark as a run loads ([`super::data::Run::mark_dead`]).
    pub fn cells_mut(self, m: &mut Model) -> &mut [Cell] {
        self.fields_mut(m).1.as_deref_mut().unwrap_or_default()
    }

    /// Its startup launches and its scaling cells in a run's model: the one table of which fields
    /// are whose.
    fn fields(self, m: &Model) -> (&Option<Vec<Rep>>, &Option<Vec<Cell>>) {
        match self {
            Engine::Scratchy => (&m.cache_ladder, &m.scaling),
            Engine::OmlxTq => (&m.cache_ladder_omlx_tq, &m.scaling_omlx_tq),
            Engine::Omlx => (&m.cache_ladder_omlx, &m.scaling_omlx),
            Engine::MlxLm => (&m.cache_ladder_mlx_lm, &m.scaling_mlx_lm),
            Engine::VllmMetal => (&m.cache_ladder_vllm_metal, &m.scaling_vllm_metal),
            Engine::Ollama => (&m.cache_ladder_ollama, &m.scaling_ollama),
            Engine::LlamaCpp => (&m.cache_ladder_llama_cpp, &m.scaling_llama_cpp),
            Engine::Mistralrs => (&m.cache_ladder_mistralrs, &m.scaling_mistralrs),
            Engine::Splash => (&m.cache_ladder_splash, &m.scaling_splash),
        }
    }

    /// [`Engine::fields`], to fill them from an earlier run ([`super::borrow_earlier`]).
    pub fn fields_mut(self, m: &mut Model) -> (&mut Option<Vec<Rep>>, &mut Option<Vec<Cell>>) {
        match self {
            Engine::Scratchy => (&mut m.cache_ladder, &mut m.scaling),
            Engine::OmlxTq => (&mut m.cache_ladder_omlx_tq, &mut m.scaling_omlx_tq),
            Engine::Omlx => (&mut m.cache_ladder_omlx, &mut m.scaling_omlx),
            Engine::MlxLm => (&mut m.cache_ladder_mlx_lm, &mut m.scaling_mlx_lm),
            Engine::VllmMetal => (&mut m.cache_ladder_vllm_metal, &mut m.scaling_vllm_metal),
            Engine::Ollama => (&mut m.cache_ladder_ollama, &mut m.scaling_ollama),
            Engine::LlamaCpp => (&mut m.cache_ladder_llama_cpp, &mut m.scaling_llama_cpp),
            Engine::Mistralrs => (&mut m.cache_ladder_mistralrs, &mut m.scaling_mistralrs),
            Engine::Splash => (&mut m.cache_ladder_splash, &mut m.scaling_splash),
        }
    }

    /// Whether it serves scratchy's own MLX checkpoint, the files [`Model::weights`] measures.
    /// ollama, llama.cpp and mistral.rs serve their own builds of the model (`served_as` names
    /// them), so its weights say nothing about their memory. Splash loads scratchy's checkpoint,
    /// with a draft model of its own beside it.
    pub fn serves_mlx_checkpoint(self) -> bool {
        match self {
            Engine::Scratchy
            | Engine::OmlxTq
            | Engine::Omlx
            | Engine::MlxLm
            | Engine::VllmMetal
            | Engine::Splash => true,
            Engine::Ollama | Engine::LlamaCpp | Engine::Mistralrs => false,
        }
    }

    pub fn ladder(self, m: &Model) -> &[Rep] {
        self.fields(m).0.as_deref().unwrap_or_default()
    }

    /// Its dataset workloads' cells in a run's model; empty for an engine the tasks axis does not
    /// record (only scratchy, mlx-lm and the oMLX pair run it).
    pub fn tasks(self, m: &Model) -> &[super::data::TaskCell] {
        match self {
            Engine::Scratchy => &m.tasks,
            Engine::OmlxTq => &m.tasks_omlx_tq,
            Engine::Omlx => &m.tasks_omlx,
            Engine::MlxLm => &m.tasks_mlx_lm,
            Engine::Splash => &m.tasks_splash,
            Engine::VllmMetal | Engine::Ollama | Engine::LlamaCpp | Engine::Mistralrs => &None,
        }
        .as_deref()
        .unwrap_or_default()
    }

    /// [`Engine::tasks`], to fill from an earlier run ([`super::borrow_earlier`]). None for an
    /// engine the tasks axis does not record.
    pub fn tasks_mut(self, m: &mut Model) -> Option<&mut Option<Vec<super::data::TaskCell>>> {
        match self {
            Engine::Scratchy => Some(&mut m.tasks),
            Engine::OmlxTq => Some(&mut m.tasks_omlx_tq),
            Engine::Omlx => Some(&mut m.tasks_omlx),
            Engine::MlxLm => Some(&mut m.tasks_mlx_lm),
            Engine::Splash => Some(&mut m.tasks_splash),
            Engine::VllmMetal | Engine::Ollama | Engine::LlamaCpp | Engine::Mistralrs => None,
        }
    }

    pub fn cells(self, m: &Model) -> &[Cell] {
        self.fields(m).1.as_deref().unwrap_or_default()
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
    /// Launch until the first token: Ready, then FirstToken.
    Launch,
    PeakRss,
}

impl Startup {
    fn of(self, r: &Rep) -> Option<f64> {
        match self {
            Startup::Ready => r.t_ready_s,
            Startup::FirstToken => r.ttft_from_send_s,
            Startup::Launch => r.ttft_exec_s,
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
            Metric::PeakMem => c.server_mem_peak_mib.filter(|_| !c.mem_below_weights),
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

    /// Throughput is better higher, time better lower.
    pub fn better(self) -> Better {
        match self {
            Metric::Throughput => Better::Higher,
            Metric::Ttft | Metric::Tpot | Metric::PeakMem => Better::Lower,
        }
    }

    /// How many times faster scratchy is than another engine at this metric.
    pub fn faster(self, scratchy: Option<f64>, other: Option<f64>) -> Option<f64> {
        self.better().faster(scratchy, other)
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

pub const DASH: &str = "—";
pub const NO_STREAM: &str = "no stream";
/// Marks a timing taken from only some of a cell's requests.
pub const PARTIAL: &str = "†";
/// Marks any number from a cell where some requests failed.
pub const FAILED: &str = "‡";
/// Marks any number from a cell whose answers came back short of the tokens asked for.
pub const SHORT: &str = "§";
/// Stands in for a peak RSS below the model's own weights ([`peak_rss`]).
pub const BELOW_WEIGHTS: &str = "below weights";

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

/// A memory reading of an engine serving a model: its MiB, below the model's own weights (so it
/// can't be all the process holds), or not measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Memory {
    Reading(f64),
    BelowWeights,
    NotMeasured,
}

impl Memory {
    /// `mib` as read for engine `e` serving `m`: below its weights, for an engine that loads the
    /// checkpoint they measure ([`Engine::serves_mlx_checkpoint`]), is not a reading.
    pub fn of(m: &Model, e: Engine, mib: Option<f64>) -> Self {
        match mib {
            None => Memory::NotMeasured,
            Some(mib) if below_weights(m, e, mib) => Memory::BelowWeights,
            Some(mib) => Memory::Reading(mib),
        }
    }

    /// The MiB, where it is a reading.
    pub fn mib(self) -> Option<f64> {
        match self {
            Memory::Reading(mib) => Some(mib),
            Memory::BelowWeights | Memory::NotMeasured => None,
        }
    }
}

/// Whether `mib` reads below `m`'s weights for an engine that loads them.
pub fn below_weights(m: &Model, e: Engine, mib: f64) -> bool {
    e.serves_mlx_checkpoint() && m.weights().is_some_and(|w| mib < w.mib())
}

/// An engine's median peak RSS over a model's launches of one scenario (#414).
pub fn peak_rss(m: &Model, e: Engine, scenario: Scenario) -> Memory {
    Memory::of(m, e, med(e.ladder(m), scenario, Startup::PeakRss))
}

/// [`peak_rss`] as text: says it reads below the weights rather than show the number.
pub fn rss_fmt(m: &Model, e: Engine, scenario: Scenario) -> String {
    match peak_rss(m, e, scenario) {
        Memory::BelowWeights => BELOW_WEIGHTS.to_string(),
        r => fmt(r.mib(), 0),
    }
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

/// A cell's value, or None where it was not really measured: a timing from an answer that never
/// streamed, or any number from a server that had stopped answering ([`Cell::dead`]).
pub fn timing(c: Option<&Cell>, metric: Metric) -> Option<f64> {
    let c = c.filter(|c| !c.dead)?;
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
    metric.is_timing() && c.is_some_and(|c| !c.dead) && streamed(c) && untimed(c) > 0
}

/// Requests in a cell that errored or timed out; every one, where its server had stopped
/// answering ([`Cell::dead`]).
pub fn failed(c: Option<&Cell>) -> u32 {
    match c {
        Some(c) if c.dead => c.failed.unwrap_or(0) + c.completed.unwrap_or(0),
        _ => c.and_then(|c| c.failed).unwrap_or(0),
    }
}

pub fn failed_note(c: Option<&Cell>) -> String {
    if let Some(c) = c.filter(|c| c.dead) {
        return format!(
            "the server had stopped answering: {} answer tokens in all",
            c.total_output_tokens.unwrap_or(0)
        );
    }
    if c.is_some_and(|c| c.timed_out) {
        return "it did not finish within the run's cell timeout".to_string();
    }
    let n = failed(c);
    format!(
        "{n} of {} requests failed",
        n + c.and_then(|c| c.completed).unwrap_or(0)
    )
}

/// The marks a cell's value carries: † for a timing from only some of its
/// requests, ‡ for any number from a cell where requests failed, § for any
/// number from a cell whose answers came back short.
pub fn marks(c: Option<&Cell>, metric: Metric) -> String {
    let mut m = String::new();
    if partial(c, metric) {
        m += PARTIAL;
    }
    if failed(c) > 0 {
        m += FAILED;
    }
    if short(c).is_some() {
        m += SHORT;
    }
    m
}

/// The share of the answer tokens asked for that a cell delivered, where its answers came back
/// short ([`Cell::short`]).
pub fn short(c: Option<&Cell>) -> Option<f64> {
    c.and_then(|c| c.short)
}

pub fn short_note(c: Option<&Cell>) -> String {
    short(c).map_or_else(String::new, |share| {
        format!(
            "answers stopped early: {:.0}% of the answer tokens asked for",
            share * 100.0
        )
    })
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
    if short(c).is_some() {
        parts.push(short_note(c));
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
    if c.is_some_and(|c| c.dead) {
        return DASH.to_string() + FAILED;
    }
    if c.is_some() && metric.is_timing() && !streamed(c) {
        return NO_STREAM.to_string() + if failed(c) > 0 { FAILED } else { "" };
    }
    fmt(timing(c, metric), metric.decimals()) + &marks(c, metric)
}

/// Run-to-run spread on one machine at one commit: two M1 Max runs of af009bf1
/// agreed within about 2%, so a smaller change is noise, not a result.
pub const NOISE: f64 = 0.03;

//! What the page computes from a run: per-engine accessors, medians, ratios,
//! and how each number is written.

use super::data::{At, Cell, Model, Rep, Scenario};
use crate::thousands_f;

/// The engines a run compares, in a fixed order, so an engine keeps its colour
/// whichever engines a run happens to include.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Engine {
    Scratchy,
    MlxLm,
    Ollama,
}

impl Engine {
    pub const ALL: [Engine; 3] = [Engine::Scratchy, Engine::MlxLm, Engine::Ollama];
    /// The engines scratchy is compared against, preferred first: mlx-lm runs
    /// the same MLX checkpoint; ollama its own GGUF.
    pub const RIVALS: [Engine; 2] = [Engine::MlxLm, Engine::Ollama];

    pub fn label(self) -> &'static str {
        match self {
            Engine::Scratchy => "scratchy",
            Engine::MlxLm => "mlx-lm",
            Engine::Ollama => "ollama",
        }
    }

    /// Its series colour (.s1 to .s3 in styles.css).
    pub fn series(self) -> &'static str {
        match self {
            Engine::Scratchy => "s1",
            Engine::MlxLm => "s2",
            Engine::Ollama => "s3",
        }
    }

    pub fn ladder(self, m: &Model) -> &[Rep] {
        match self {
            Engine::Scratchy => &m.cache_ladder,
            Engine::MlxLm => &m.cache_ladder_mlx_lm,
            Engine::Ollama => &m.cache_ladder_ollama,
        }
        .as_deref()
        .unwrap_or_default()
    }

    pub fn cells(self, m: &Model) -> &[Cell] {
        match self {
            Engine::Scratchy => &m.scaling,
            Engine::MlxLm => &m.scaling_mlx_lm,
            Engine::Ollama => &m.scaling_ollama,
        }
        .as_deref()
        .unwrap_or_default()
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
        }
    }

    /// A short name for element ids.
    pub fn id(self) -> &'static str {
        match self {
            Metric::Ttft => "ttft",
            Metric::Tpot => "tpot",
            Metric::Throughput => "tput",
        }
    }

    /// Decimals it is written with.
    pub fn decimals(self) -> usize {
        match self {
            Metric::Ttft => 0,
            Metric::Tpot | Metric::Throughput => 1,
        }
    }

    pub fn unit(self) -> &'static str {
        match self {
            Metric::Throughput => "tok/s",
            Metric::Ttft | Metric::Tpot => "ms",
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
            Metric::Ttft | Metric::Tpot => o / s,
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

pub fn untimed_note(c: Option<&Cell>) -> String {
    let completed = c
        .and_then(|c| c.completed)
        .map_or("?".to_string(), |n| n.to_string());
    format!("{} of {completed} untimed", untimed(c))
}

/// A cell's value as text: says "no stream" rather than show a timing the
/// engine never produced, and marks one taken from only some requests.
pub fn cfmt(c: Option<&Cell>, metric: Metric) -> String {
    if c.is_some() && metric.is_timing() && !streamed(c) {
        return NO_STREAM.to_string();
    }
    fmt(timing(c, metric), metric.decimals()) + if partial(c, metric) { PARTIAL } else { "" }
}

/// Run-to-run spread on one machine at one commit: two M1 Max runs of af009bf1
/// agreed within about 2%, so a smaller change is noise, not a result.
pub const NOISE: f64 = 0.03;

/// ColorBrewer RdBu, 7 classes: diverging, colour-blind safe, stepped so "a bit
/// faster" and "much faster" read differently. Bounds are symmetric in log
/// space (×0.8 is as far from ×1 as ×1.25) and the middle class is the noise
/// band. The colours live in styles.css as .hm0 to .hm6.
const HEAT_BOUNDS: [f64; 6] = [0.5, 0.8, 1.0 - NOISE, 1.0 + NOISE, 1.25, 2.0];
pub const HEAT_LABELS: [&str; 7] = [
    "much slower (x0.5 or less)",
    "slower",
    "a bit slower",
    "about the same",
    "a bit faster",
    "faster",
    "much faster (x2 or more)",
];

/// The heat class for a "how many times faster is scratchy" ratio: hm0 (much
/// slower) to hm6 (much faster), hm3 inside the noise band.
pub fn heat_class(r: f64) -> String {
    format!("hm{}", HEAT_BOUNDS.iter().filter(|&&b| r >= b).count())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_band_is_the_middle_class() {
        assert_eq!(heat_class(1.0), "hm3");
        assert_eq!(heat_class(0.5), "hm1");
        assert_eq!(heat_class(0.49), "hm0");
        assert_eq!(heat_class(2.0), "hm6");
    }
}

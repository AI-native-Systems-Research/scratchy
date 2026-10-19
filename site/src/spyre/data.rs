//! site/data/spyre/*.json — the A/B sweeps the page charts.
//!
//! One file per sweep, named `<date>-<sha8>.json`: the day of the sweep and
//! the commit scratchy's side ran. A sweep holds both stacks' measurements:
//! torch-spyre and triton-scratchy are measured by their own harnesses (the
//! file says which), so a cell is a bucket, not a matched prompt — compare
//! ratios, not absolutes. It also holds everything the page says about that
//! sweep alone (what moved since the last one, the levers left, the notes
//! under a section, a target): a new sweep is one new file, and nothing on
//! the page speaks for an older one. `.claude/skills/spyre-bench` is how to
//! write one.
//!
//! Every struct refuses fields it does not know, so a misspelt key fails the
//! build rather than dropping out of the page.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::charts::Better;

/// A whole A/B sweep.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sweep {
    pub schema: u32,
    /// The day of the sweep, "2026-10-17": the file is named for it.
    pub date: String,
    /// The context buckets the cells sit in, and the prompt tokens each
    /// bucket means on scratchy's side ("~1k": 1051 — the lengths the two
    /// harnesses used differ, so a bucket is the honest unit).
    #[serde(default)]
    pub contexts: BTreeMap<String, u32>,
    pub torch_spyre: Provenance,
    pub triton_scratchy: Provenance,
    pub cells: Vec<Cell>,
    /// The PRs merged on scratchy's side since the previous sweep,
    /// "#563–#583"; None for a first sweep.
    #[serde(default)]
    pub since: Option<String>,
    /// What those PRs did, one row each.
    #[serde(default)]
    pub moved: Vec<Moved>,
    /// The work still open where torch-spyre leads (or a target is unmet).
    #[serde(default)]
    pub levers: Vec<Lever>,
    /// Paragraphs under a section, after its charts.
    #[serde(default)]
    pub notes: Notes,
    /// A cell's target, beyond parity: a headline tile of its own.
    #[serde(default)]
    pub target: Option<Target>,
}

/// One row of "What moved": a change and what it did, in numbers.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Moved {
    /// The PR, or PRs: "#566–#568".
    pub pr: String,
    pub what: String,
}

/// One row of "Open levers": where, and the work.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lever {
    /// The cell or theme: "8b 10k decode".
    pub lever: String,
    pub work: String,
}

/// Each section's closing paragraphs.
#[derive(Default, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Notes {
    #[serde(default)]
    pub decode: Vec<String>,
    #[serde(default)]
    pub prefill: Vec<String>,
    #[serde(default)]
    pub conc: Vec<String>,
    #[serde(default)]
    pub fp8: Vec<String>,
}

/// What one cell should reach, and why: the page sets it beside today's
/// value there.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    /// The tile's heading: "8b moonshot · short-context ITL".
    pub label: String,
    pub model: String,
    pub dtype: Dtype,
    pub metric: Metric,
    #[serde(default = "one")]
    pub conc: u32,
    pub context: String,
    /// In the metric's unit (ms, or tok/s).
    pub value: f64,
    /// Where the target comes from: "cost-model peak 204.8 GB/s — …".
    pub why: String,
}

/// One side's provenance: when, where and how its half of the sweep ran.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub measured: String,
    #[serde(default)]
    pub host: Option<String>,
    #[serde(default)]
    pub commit: Option<String>,
    pub harness: String,
}

impl Provenance {
    /// The commit's first eight characters, as the sweep's file is named.
    pub fn sha8(&self) -> &str {
        self.commit.as_deref().map_or("", |c| &c[..c.len().min(8)])
    }
}

/// What a cell measures.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub enum Metric {
    /// Inter-token latency, ms.
    #[serde(rename = "itl_ms")]
    Itl,
    /// Time to first token, ms.
    #[serde(rename = "ttft_ms")]
    Ttft,
    /// Decoded tokens per second.
    #[serde(rename = "tok_s")]
    TokS,
}

impl Metric {
    pub fn label(self) -> &'static str {
        match self {
            Metric::Itl => "decode ITL",
            Metric::Ttft => "prefill TTFT",
            Metric::TokS => "throughput",
        }
    }

    /// The short name a table column or tile carries it under.
    pub fn short(self) -> &'static str {
        match self {
            Metric::Itl => "ITL",
            Metric::Ttft => "TTFT",
            Metric::TokS => "tok/s",
        }
    }

    pub fn better(self) -> Better {
        match self {
            Metric::TokS => Better::Higher,
            Metric::Itl | Metric::Ttft => Better::Lower,
        }
    }
}

/// The weights a cell ran.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    Fp16,
    Fp8,
}

impl Dtype {
    pub fn label(self) -> &'static str {
        match self {
            Dtype::Fp16 => "fp16",
            Dtype::Fp8 => "fp8",
        }
    }
}

/// One measured cell: both stacks' value at one model, dtype, metric,
/// concurrency and context bucket.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cell {
    pub model: String,
    pub dtype: Dtype,
    pub metric: Metric,
    #[serde(default = "one")]
    pub conc: u32,
    /// The context bucket the cell sits in ("~1k"); a concurrency sweep's
    /// cells carry none.
    #[serde(default)]
    pub context: Option<String>,
    pub torch: f64,
    pub scratchy: f64,
    #[serde(default)]
    pub note: Option<String>,
}

fn one() -> u32 {
    1
}

impl Cell {
    /// How many times faster scratchy is than torch-spyre at this cell.
    pub fn ratio(&self) -> f64 {
        self.metric
            .better()
            .faster(Some(self.scratchy), Some(self.torch))
            .unwrap_or(f64::NAN)
    }
}

/// One perf PR's progress entry, `progress/<pr>.json`, shipped in the PR
/// itself: the cells its merge gate measured, main against the PR, on one
/// pod, back to back. The sweep files stay the record; a progress entry
/// layers on top of them, and the page pairs each of its values with the
/// newest sweep's torch cell — torch-spyre is never re-measured for one.
///
/// Every struct refuses fields it does not know, so a misspelt key fails the
/// build rather than dropping out of the page.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Progress {
    pub schema: u32,
    /// The PR the file is named for: `progress/<pr>.json`.
    pub pr: u32,
    /// The PR's title as merged.
    pub title: String,
    /// The PR head commit that was measured, full sha.
    pub head: String,
    /// The main commit it was measured against, full sha.
    pub base: String,
    /// The day the PR merged, "2026-10-21": the page orders entries by it.
    pub merged: String,
    /// The one pod both sides ran on, back to back.
    pub pod: String,
    /// How the cells were measured: the script, its cell set, its gate.
    pub harness: String,
    /// Exactly the cells the PR measured — no filler.
    pub cells: Vec<ProgressCell>,
}

/// One cell a progress entry measured: main's and the PR's value at one
/// point of the sweep grid.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgressCell {
    pub model: String,
    pub dtype: Dtype,
    pub metric: Metric,
    #[serde(default = "one")]
    pub conc: u32,
    /// The context bucket the cell sits in ("~1k").
    pub context: String,
    /// main's value at the cell.
    pub main: f64,
    /// The PR's value at the cell.
    pub pr: f64,
    /// Every run behind each headline, in run order. Each headline is its
    /// side's last run — the steady-state law the sweep cells follow — and
    /// the build checks it, so a headline can never cherry-pick a rep.
    pub reps: Reps,
    #[serde(default)]
    pub note: Option<String>,
}

/// The runs behind a progress cell's headlines, per side.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reps {
    pub main: Vec<f64>,
    pub pr: Vec<f64>,
}

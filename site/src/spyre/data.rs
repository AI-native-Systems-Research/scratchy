//! site/data/spyre/*.json — the A/B sweeps the page charts.
//!
//! One file per sweep, named `<date>-<sha8>.json`: the day of the sweep and
//! the commit scratchy's side ran. A sweep holds both stacks' measurements:
//! torch-spyre and triton-scratchy are measured by their own harnesses (the
//! file says which), so a cell is a bucket, not a matched prompt — compare
//! ratios, not absolutes.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::charts::Better;

/// A whole A/B sweep.
#[derive(Deserialize)]
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
}

/// One side's provenance: when, where and how its half of the sweep ran.
#[derive(Deserialize)]
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
#[derive(Deserialize)]
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

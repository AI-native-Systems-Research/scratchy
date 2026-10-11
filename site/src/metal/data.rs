//! One run of scripts/bench_metal_matrix.sh, as the page reads it (schema 2).
//! Every field the page needs is typed here, so a file missing one fails the
//! build naming it, rather than rendering a blank.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Deserializer};

#[derive(Deserialize)]
pub struct Run {
    #[serde(rename = "schema")]
    _schema: Schema,
    pub generated_utc: Utc,
    #[serde(rename = "generator")]
    _generator: NonEmpty,
    pub machine: Machine,
    pub repo: Repo,
    pub config: Config,
    pub models: Vec<Model>,
    /// Models the runner left out on this machine, and why (memory).
    #[serde(default)]
    pub skipped_models: Option<Vec<Skipped>>,
    /// The file it was read from, for the link to it.
    #[serde(skip)]
    pub file: PathBuf,
}

/// The only schema this page reads.
#[derive(Deserialize)]
#[serde(try_from = "u32")]
pub struct Schema;

impl TryFrom<u32> for Schema {
    type Error = String;
    fn try_from(v: u32) -> Result<Self, String> {
        if v == 2 {
            Ok(Schema)
        } else {
            Err(format!("schema {v}, this page reads schema 2"))
        }
    }
}

/// A string the runner must have filled in.
#[derive(Deserialize, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[serde(try_from = "String")]
pub struct NonEmpty(String);

impl TryFrom<String> for NonEmpty {
    type Error = &'static str;
    fn try_from(s: String) -> Result<Self, &'static str> {
        if s.is_empty() {
            Err("empty")
        } else {
            Ok(NonEmpty(s))
        }
    }
}

impl NonEmpty {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for NonEmpty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// When a run started, `YYYY-MM-DDTHH:MM:SSZ`.
#[derive(Deserialize, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[serde(try_from = "String")]
pub struct Utc(String);

impl TryFrom<String> for Utc {
    type Error = String;
    fn try_from(s: String) -> Result<Self, String> {
        let shape = "dddd-dd-ddTdd:dd:ddZ";
        let ok = s.len() == shape.len()
            && s.bytes().zip(shape.bytes()).all(|(c, p)| {
                if p == b'd' {
                    c.is_ascii_digit()
                } else {
                    c == p
                }
            });
        if ok {
            Ok(Utc(s))
        } else {
            Err(format!("generated_utc {s:?} is not YYYY-MM-DDTHH:MM:SSZ"))
        }
    }
}

impl Utc {
    /// `YYYY-MM-DD`.
    pub fn date(&self) -> &str {
        &self.0[..10]
    }
    /// `MM-DD`.
    pub fn month_day(&self) -> &str {
        &self.0[5..10]
    }
    /// `YYYY-MM-DD HH:MM`.
    pub fn minute(&self) -> String {
        format!("{} {}", &self.0[..10], &self.0[11..16])
    }
    /// `YYYY-MM-DDTHHMMSSZ`, as file names carry it.
    pub fn stamp(&self) -> String {
        format!("{}T{}Z", &self.0[..10], self.0[11..19].replace(':', ""))
    }
}

#[derive(Deserialize)]
pub struct Machine {
    pub chip: NonEmpty,
    pub cores_total: u32,
    pub cores_performance: Option<u32>,
    pub cores_efficiency: Option<u32>,
    pub memory_gb: u32,
    pub macos: NonEmpty,
}

#[derive(Deserialize)]
pub struct Repo {
    pub sha: NonEmpty,
}

impl Repo {
    pub fn sha8(&self) -> &str {
        let s = self.sha.as_str();
        &s[..s.len().min(8)]
    }
}

#[derive(Deserialize)]
pub struct Skipped {
    pub stem: String,
    pub reason: Option<String>,
}

#[derive(Deserialize)]
pub struct Config {
    pub scenarios: Vec<String>,
    pub cold_priming_launches: Option<u32>,
    pub kv_cache_dtype: Option<String>,
    pub scratchy_serve_args: Option<String>,
    pub scaling: Option<Scaling>,
    /// Every engine with its version and pin; runs before pins have only
    /// `engine_versions` (mlx-lm and ollama).
    #[serde(default)]
    pub engines: Option<Vec<EngineInfo>>,
    #[serde(default)]
    pub engine_versions: Option<BTreeMap<String, Option<String>>>,
    /// Whether every engine matched its pin; absent before pins.
    #[serde(default)]
    pub pinned: Option<bool>,
    #[serde(default)]
    pub pin_problems: Option<Vec<String>>,
    /// Every model the runner measures (`--models full`); the first runs to
    /// record a list kept only the default round, as `default_models`.
    #[serde(default)]
    pub current_models: Option<Vec<String>>,
    #[serde(default)]
    pub default_models: Option<Vec<String>>,
}

#[derive(Deserialize)]
pub struct EngineInfo {
    pub key: String,
    pub label: String,
    pub installed: Option<bool>,
    pub version: Option<String>,
}

impl Config {
    pub fn base(&self) -> Option<&Base> {
        self.scaling.as_ref().and_then(|s| s.base.as_ref())
    }
}

#[derive(Deserialize)]
pub struct Scaling {
    pub base: Option<Base>,
    pub grid_input: Option<Vec<u32>>,
    pub grid_output: Option<Vec<u32>>,
}

/// What a model ran at when a limit lowered it (GLM-4.5-Air: 4 users).
#[derive(Deserialize)]
pub struct Limits {
    pub base_conc: Option<u32>,
}

/// What one engine served for a model.
#[derive(Deserialize)]
pub struct Served {
    pub model: Option<String>,
    pub quantization: Option<String>,
}

/// Which version of a model's files a run used.
#[derive(Deserialize)]
pub struct Files {
    pub revision: Option<String>,
}

/// Some(value) when the field is present, even as null; None when absent.
fn present<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Option<bool>>, D::Error> {
    Ok(Some(Option::<bool>::deserialize(d)?))
}

/// The shape every non-grid measurement uses.
#[derive(Deserialize)]
pub struct Base {
    pub input: Option<u32>,
    pub output: Option<u32>,
    pub conc: Option<u32>,
}

#[derive(Deserialize)]
pub struct Model {
    pub stem: String,
    pub model_id: String,
    pub built: bool,
    pub quant: Option<String>,
    pub footprint: Option<Footprint>,
    pub ollama: Option<Ollama>,
    pub cache_ladder: Option<Vec<Rep>>,
    pub cache_ladder_mlx_lm: Option<Vec<Rep>>,
    pub cache_ladder_ollama: Option<Vec<Rep>>,
    #[serde(default)]
    pub cache_ladder_omlx: Option<Vec<Rep>>,
    #[serde(default)]
    pub cache_ladder_omlx_tq: Option<Vec<Rep>>,
    #[serde(default)]
    pub cache_ladder_vllm_metal: Option<Vec<Rep>>,
    #[serde(default)]
    pub cache_ladder_llama_cpp: Option<Vec<Rep>>,
    #[serde(default)]
    pub cache_ladder_mistralrs: Option<Vec<Rep>>,
    pub scaling: Option<Vec<Cell>>,
    pub scaling_mlx_lm: Option<Vec<Cell>>,
    pub scaling_ollama: Option<Vec<Cell>>,
    #[serde(default)]
    pub scaling_omlx: Option<Vec<Cell>>,
    #[serde(default)]
    pub scaling_omlx_tq: Option<Vec<Cell>>,
    #[serde(default)]
    pub scaling_vllm_metal: Option<Vec<Cell>>,
    #[serde(default)]
    pub scaling_llama_cpp: Option<Vec<Cell>>,
    #[serde(default)]
    pub scaling_mistralrs: Option<Vec<Cell>>,
    /// Per engine key, what it served: the ollama tag or GGUF file and its
    /// quantization, and the files' revision.
    #[serde(default)]
    pub engine_models: Option<BTreeMap<String, Served>>,
    /// The revision of the MLX checkpoint scratchy (and the MLX engines) ran.
    #[serde(default)]
    pub model_files: Option<Files>,
    /// The mlx-lm parity gate: Some(Some(true)) passed, Some(Some(false))
    /// failed (nothing timed), Some(None) not run, None from before the gate.
    #[serde(default, deserialize_with = "present")]
    pub parity_mlx_lm: Option<Option<bool>>,
    #[serde(default)]
    pub scaling_limits: Option<Limits>,
    /// Only its presence matters: whether scratchy served anything warm.
    pub warm_serving: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Deserialize)]
pub struct Footprint {
    pub build_seconds: Option<u32>,
    pub binary_bytes: Option<u64>,
}

#[derive(Deserialize)]
pub struct Ollama {
    pub tag: Option<String>,
    pub quantization: Option<String>,
}

/// The startup scenarios a cache ladder steps through.
#[derive(Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Scenario {
    /// First launch after memory and scratchy's weights cache are cleared.
    Frozen,
    /// A normal relaunch.
    Cold,
    /// Requests to a server already running.
    Warm,
}

/// One launch (or request) of a cache ladder.
#[derive(Deserialize)]
pub struct Rep {
    pub scenario: Scenario,
    pub t_ready_s: Option<f64>,
    pub ttft_from_send_s: Option<f64>,
    pub peak_rss_mib: Option<f64>,
}

/// Where in the scaling sweep a cell sits.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum At {
    /// Offered concurrent users, at the base shape.
    Conc(u32),
    /// Prompt tokens, at the base answer size and users.
    Input(u32),
    /// Answer tokens, at the base prompt size and users.
    Output(u32),
    /// Prompt × answer tokens, at the cell's own users (`Cell::users`).
    Grid { input: u32, output: u32 },
}

/// One `bench serve` cell of the scaling sweep.
#[derive(Deserialize)]
#[serde(try_from = "RawCell")]
pub struct Cell {
    pub at: At,
    pub median_ttft_ms: Option<f64>,
    pub median_tpot_ms: Option<f64>,
    pub output_throughput: Option<f64>,
    pub completed: Option<u32>,
    /// Requests that arrived in one piece and so were left out of TTFT/TPOT;
    /// absent (so 0) in runs from before it existed.
    pub unstreamed_requests: Option<u32>,
    /// Requests that errored or timed out: every number here then covers only
    /// the completed ones. Absent (so 0) in runs from before it existed.
    pub failed: Option<u32>,
    /// The server's peak memory footprint during the cell.
    pub server_mem_peak_mib: Option<f64>,
    /// A grid cell's users at once; None in runs from before the grid had its
    /// own, which ran it at the base users count.
    pub users: Option<u32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Axis {
    Conc,
    Input,
    Output,
    Grid,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawRung {
    Users(u32),
    Shape(String),
}

#[derive(Deserialize)]
struct RawCell {
    axis: Axis,
    rung: RawRung,
    median_ttft_ms: Option<f64>,
    median_tpot_ms: Option<f64>,
    output_throughput: Option<f64>,
    completed: Option<u32>,
    unstreamed_requests: Option<u32>,
    #[serde(default)]
    failed: Option<u32>,
    #[serde(default)]
    server_mem_peak_mib: Option<f64>,
    #[serde(default)]
    conc: Option<u32>,
}

impl TryFrom<RawCell> for Cell {
    type Error = String;
    fn try_from(c: RawCell) -> Result<Self, String> {
        let at = match (c.axis, c.rung) {
            (Axis::Conc, RawRung::Users(n)) => At::Conc(n),
            (Axis::Input, RawRung::Users(n)) => At::Input(n),
            (Axis::Output, RawRung::Users(n)) => At::Output(n),
            (Axis::Grid, RawRung::Shape(s)) => {
                let shape = s
                    .split_once('x')
                    .and_then(|(i, o)| Some((i.parse().ok()?, o.parse().ok()?)));
                let (input, output) =
                    shape.ok_or_else(|| format!("grid rung {s:?} is not <input>x<output>"))?;
                At::Grid { input, output }
            }
            (Axis::Conc | Axis::Input | Axis::Output, RawRung::Shape(s)) => {
                return Err(format!("rung {s:?} is not a count"));
            }
            (Axis::Grid, RawRung::Users(n)) => {
                return Err(format!("grid rung {n} is not <input>x<output>"));
            }
        };
        Ok(Cell {
            at,
            median_ttft_ms: c.median_ttft_ms,
            median_tpot_ms: c.median_tpot_ms,
            output_throughput: c.output_throughput,
            completed: c.completed,
            unstreamed_requests: c.unstreamed_requests,
            failed: c.failed,
            server_mem_peak_mib: c.server_mem_peak_mib,
            users: c.conc,
        })
    }
}

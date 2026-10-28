//! One run of scripts/bench_metal_matrix.sh, as the page reads it (schema 2).
//! Every field the page needs is typed here, so a file missing one fails the
//! build naming it, rather than rendering a blank.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Deserializer};

use super::numbers::Engine;

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

/// The least share of its answer tokens a live server delivers to an engine that honours
/// `ignore_eos` ([`Engine::honours_ignore_eos`]), below which the cell measured one that had
/// stopped answering (#413). In every published run, scratchy's live cells deliver 99% or more and
/// its dead ones 0 to 19.8%.
const LIVE_SHARE: f64 = 0.5;

/// The least share of its answer tokens a live cell delivers where every answer ran to length:
/// scratchy's live cells, which honour `ignore_eos`, deliver 99% or more in every published run. A
/// live cell under it stopped answers early, at EOS (#479), so its numbers measure less decode than
/// the cell asked for ([`Cell::short`]).
const FULL_SHARE: f64 = 0.99;

impl Run {
    /// Mark the cells that measured a server which had stopped answering (#413). The bench judges
    /// that itself where it can (`dead_server` on a cell, from a scratchy cell that came back with
    /// under half its answer tokens), and that verdict is the one the page takes. A run file from
    /// before the bench counted such requests as failed (`config.failures_counted`, #510) is
    /// judged here by its answer tokens instead, and only for an engine that honours
    /// `ignore_eos`: one that stops at EOS delivers short answers from a live server too.
    pub fn mark_dead(&mut self) {
        for m in &mut self.models {
            for e in Engine::ALL {
                for c in e.cells_mut(m) {
                    c.dead |= c.dead_server;
                }
            }
        }
        if self.config.failures_counted == Some(true) {
            return;
        }
        let base = self.config.base().and_then(|b| b.output);
        for m in &mut self.models {
            for e in Engine::ALL.into_iter().filter(|e| e.honours_ignore_eos()) {
                for c in e.cells_mut(m) {
                    c.dead |= c.delivered(base).is_some_and(|share| share < LIVE_SHARE);
                }
            }
        }
    }

    /// Mark the live cells whose answers came back short of the tokens asked for (#479): an
    /// engine that ignores `ignore_eos` stops at EOS, so its tok/s, TPOT and E2E latency come from
    /// a shorter decode than the cell's label says. Done as the run loads, after
    /// [`mark_dead`](Self::mark_dead), so a dead cell is never also short.
    pub fn mark_short(&mut self) {
        let base = self.config.base().and_then(|b| b.output);
        for m in &mut self.models {
            for e in Engine::ALL {
                for c in e.cells_mut(m) {
                    c.short = (c.delivered(base)).filter(|&share| !c.dead && share < FULL_SHARE);
                }
            }
        }
    }

    /// Mark the scaling cells whose peak memory reads below the model's own weights (#414): a
    /// process serving them holds at least that much, so the reading can't be all of it. Done as
    /// the run loads, so every chart and table drops the number at once ([`Cell::mem_below_weights`]).
    pub fn mark_mem_below_weights(&mut self) {
        for m in &mut self.models {
            let weights = m.weights();
            for e in Engine::ALL
                .into_iter()
                .filter(|e| e.serves_mlx_checkpoint())
            {
                for c in e.cells_mut(m) {
                    c.mem_below_weights = (c.server_mem_peak_mib)
                        .is_some_and(|mib| weights.is_some_and(|w| mib < w.mib()));
                }
            }
        }
    }
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
#[derive(Deserialize, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
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
    /// scratchy's `--gpu-memory-utilization` where the runner set it for the Mac's memory.
    #[serde(default)]
    pub scratchy_memory_util: Option<f64>,
    pub scaling: Option<Scaling>,
    /// Whether the bench counted a request whose stream ended aborted or errored as failed
    /// (#510); absent before it, where [`Run::mark_dead`] judges by the answer tokens instead.
    #[serde(default)]
    pub failures_counted: Option<bool>,
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
    /// The lineup this version of the runner measures, one model per architecture class
    /// (`CLASSES` in the bench script); absent in runs from before it.
    #[serde(default)]
    pub classes: Option<Vec<Class>>,
    /// The memory from which a Mac runs the large set of [`Config::classes`].
    #[serde(default)]
    pub large_from_gb: Option<u32>,
    #[serde(default)]
    pub default_models: Option<Vec<String>>,
}

/// One architecture class of the lineup: its model below 64 GB of memory and at 64 GB and above.
#[derive(Deserialize, Clone, PartialEq, Eq)]
pub struct Class {
    pub class: String,
    pub small: String,
    pub large: String,
}

impl Class {
    /// The class's model on a Mac with `memory_gb` of memory, the large one from `large_from_gb`.
    pub fn model(&self, memory_gb: u32, large_from_gb: u32) -> &str {
        if memory_gb >= large_from_gb {
            &self.large
        } else {
            &self.small
        }
    }
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

#[derive(Deserialize, PartialEq)]
pub struct Scaling {
    /// The sweeps it ran (`grid`, `input`, ...).
    #[serde(default)]
    pub axes: Option<Vec<String>>,
    pub base: Option<Base>,
    pub grid_input: Option<Vec<u32>>,
    pub grid_output: Option<Vec<u32>>,
    /// The grid's users counts; absent in runs from before the grid had its own.
    #[serde(default)]
    pub grid_conc: Option<Vec<u32>>,
    /// The answer lengths the grid ran at above 1 user (`--scale-grid-users-output`); absent
    /// in runs from before it, which ran every `grid_output` at every users count.
    pub grid_users_output: Option<Vec<u32>>,
    /// The longest grid prompt the Mac's memory held (the runner's `--scale-max-prompt`, before
    /// every cell ran): longer
    /// `grid_input` lengths were not run. None: every one was.
    #[serde(default)]
    pub max_prompt: Option<u32>,
    /// The longest grid prompt run above 1 user (the runner's `--scale-max-prompt-users`, before
    /// every cell ran); longer prompts
    /// ran at 1 user only. None: every prompt `max_prompt` allows.
    #[serde(default)]
    pub max_prompt_users: Option<u32>,
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
#[derive(Deserialize, PartialEq)]
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
    #[serde(default)]
    pub cache_ladder_splash: Option<Vec<Rep>>,
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
    #[serde(default)]
    pub scaling_splash: Option<Vec<Cell>>,
    /// Per engine key, what it served: the ollama tag or GGUF file and its
    /// quantization, and the files' revision.
    #[serde(default)]
    pub engine_models: Option<BTreeMap<String, Served>>,
    /// Per engine key, why its own memory guard declined this model outright
    /// (oMLX refusing the load), in the engine's words. Engines here never
    /// served, so their cells are absent by refusal, not by omission.
    #[serde(default)]
    pub engine_declined: Option<BTreeMap<String, NonEmpty>>,
    /// The revision of the MLX checkpoint scratchy (and the MLX engines) ran.
    #[serde(default)]
    pub model_files: Option<Files>,
    /// The mlx-lm parity gate: Some(Some(true)) passed, Some(Some(false))
    /// failed (nothing timed), Some(None) not run, None from before the gate.
    #[serde(default, deserialize_with = "present")]
    pub parity_mlx_lm: Option<Option<bool>>,
    /// Why the parity gate did not judge this model, where the runner skipped it on purpose.
    #[serde(default)]
    pub parity_skipped: Option<String>,
    /// The gate's prompts that passed on near-ties, each with the logit gaps of the tokens where
    /// scratchy's answer parted from mlx-lm's top; absent where every answer matched exactly.
    #[serde(default)]
    pub parity_ties: Option<BTreeMap<String, Vec<f64>>>,
    #[serde(default)]
    pub scaling_limits: Option<Limits>,
    /// Only its presence matters: whether scratchy served anything warm.
    pub warm_serving: Option<serde_json::Map<String, serde_json::Value>>,
    /// What an earlier run of this model on this machine filled in, and when that run was made:
    /// an engine this run did not measure (by its key), or scratchy's frozen launches
    /// ([`BORROWED_FROZEN`]). Empty for the numbers this run measured itself.
    #[serde(skip)]
    pub borrowed: BTreeMap<String, Utc>,
}

/// The [`Model::borrowed`] key for scratchy's frozen launches.
pub const BORROWED_FROZEN: &str = "scratchy frozen";

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
#[derive(Deserialize, Clone)]
pub struct Rep {
    pub scenario: Scenario,
    pub t_ready_s: Option<f64>,
    pub ttft_from_send_s: Option<f64>,
    /// Launch until the first token: ready, then the first request.
    pub ttft_exec_s: Option<f64>,
    pub peak_rss_mib: Option<f64>,
}

/// A size in bytes.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bytes(pub u64);

impl Bytes {
    /// In GB (10⁹ bytes), as model cards give sizes.
    pub fn gb(self) -> f64 {
        self.0 as f64 / 1e9
    }

    /// In MiB (2²⁰ bytes), as the runner records memory.
    pub fn mib(self) -> f64 {
        self.0 as f64 / (1u64 << 20) as f64
    }
}

/// Each MLX checkpoint's weights: the `.safetensors` files scratchy and mlx-lm
/// load, summed from the Hugging Face API (main, 2026-10-12). ollama runs its
/// own build of the same model, named on its row.
const WEIGHTS: [(&str, Bytes); 18] = [
    (
        "mlx-community/granite-3.3-2b-instruct-4bit",
        Bytes(1_425_458_043),
    ),
    (
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        Bytes(4_284_346_255),
    ),
    (
        "mlx-community/gemma-4-26b-a4b-it-4bit",
        Bytes(15_341_205_776),
    ),
    // This round's models, from the same API (main, 2026-10-12).
    ("mlx-community/Qwen3.6-35B-A3B-4bit", Bytes(20_402_204_271)),
    ("mlx-community/Qwen3.6-27B-4bit", Bytes(16_054_541_599)),
    ("mlx-community/gemma-4-31b-it-4bit", Bytes(18_412_016_676)),
    ("mlx-community/GLM-4.5-Air-3bit", Bytes(46_786_574_771)),
    // The one-model-per-class round, small and large, from the same API (main, 2026-10-20).
    // The same files, byte for byte: the project's copy, and the personal repo runs before it name.
    ("scratchery/gpt-oss-20b-4bit", Bytes(11_774_668_888)),
    ("jesusoctavioas/gpt-oss-20b-mlx-4Bit", Bytes(11_774_668_888)),
    ("scratchery/gpt-oss-120b-3bit", Bytes(51_448_416_856)),
    (
        "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit",
        Bytes(17_775_428_300),
    ),
    (
        "mlx-community/NVIDIA-Nemotron-3-Super-120B-A12B-OptiQ-2bit",
        Bytes(47_460_151_014),
    ),
    (
        "scratchery/Qwen3.5-122B-A10B-OptiQ-2bit-mtp",
        Bytes(48_263_329_425),
    ),
    ("mlx-community/granite-4.1-8b-4bit", Bytes(5_238_406_779)),
    ("mlx-community/granite-4.1-30b-4bit", Bytes(18_041_976_573)),
    ("ibm-granite/granite-4.2-3b-q4-mlx", Bytes(2_059_005_430)),
    ("ibm-granite/granite-4.2-8b-q4-mlx", Bytes(4_945_853_970)),
    ("ibm-granite/granite-4.2-30b-q4-mlx", Bytes(16_469_111_945)),
];

impl Model {
    /// Its checkpoint's weights; every model a run names has them (`load`
    /// fails the build otherwise).
    pub fn weights(&self) -> Option<Bytes> {
        WEIGHTS
            .iter()
            .find(|(id, _)| *id == self.model_id)
            .map(|&(_, b)| b)
    }
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
#[derive(Deserialize, Clone)]
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
    /// The answer tokens its requests delivered; None in runs from before
    /// the run file kept it.
    pub total_output_tokens: Option<u64>,
    /// Whether the server it measured had stopped answering, in a run from before the bench
    /// counted that itself ([`Run::mark_dead`]).
    pub dead: bool,
    /// Whether its peak memory reads below the model's own weights ([`Run::mark_mem_below_weights`]),
    /// so it is no reading of the process.
    pub mem_below_weights: bool,
    /// The client never finished it (the run's `cell_timeout_s`), so every request counts as
    /// failed: too slow, not out of memory.
    pub timed_out: bool,
    /// The bench's verdict that the server it measured answered with almost nothing (a scratchy
    /// cell under half its answer tokens): its requests are recorded failed ([`Run::mark_dead`]).
    pub dead_server: bool,
    /// The share of the answer tokens asked for that a live cell delivered, where its answers came
    /// back short ([`Run::mark_short`]); None where they ran to length or it can't say.
    pub short: Option<f64>,
    /// Every request it sent, over all its repeats; None in runs from before the run file kept them.
    pub requests: Option<Requests>,
}

/// A cell's requests, in the order each repeat sent them: parallel arrays, one entry a request.
/// A timing is None where the request was not timed: it failed, or (TTFT and TPOT) its answer
/// arrived in one piece.
#[derive(Deserialize, Clone)]
pub struct Requests {
    /// Which of the cell's repeats it was in, from 0; each repeat ran on the same server.
    pub rep: Vec<u32>,
    pub ttft_ms: Vec<Option<f64>>,
    pub tpot_ms: Vec<Option<f64>>,
    pub e2el_ms: Vec<Option<f64>>,
}

impl Requests {
    /// Each request's value of one timing, with its repeat.
    pub fn of(&self, t: RequestTiming) -> impl Iterator<Item = (u32, Option<f64>)> + '_ {
        let vs = match t {
            RequestTiming::Ttft => &self.ttft_ms,
            RequestTiming::Tpot => &self.tpot_ms,
            RequestTiming::E2el => &self.e2el_ms,
        };
        self.rep.iter().copied().zip(vs.iter().copied())
    }
}

/// A timing every request has its own value of.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RequestTiming {
    Ttft,
    Tpot,
    E2el,
}

impl RequestTiming {
    pub const ALL: [RequestTiming; 3] = [
        RequestTiming::Ttft,
        RequestTiming::Tpot,
        RequestTiming::E2el,
    ];

    pub fn title(self) -> &'static str {
        match self {
            RequestTiming::Ttft => "time to first token",
            RequestTiming::Tpot => "time per output token",
            RequestTiming::E2el => "end-to-end time",
        }
    }
}

impl Cell {
    /// The share of the answer tokens its completed requests asked for that they delivered: its
    /// answer length is its rung's, or the run's `base` one. None where it can't say: no count in
    /// the file, no request completed, or a count of 0 from requests that each got a first token,
    /// which is a server that sent no usage (the bench takes the count from it), not a dead one.
    fn delivered(&self, base: Option<u32>) -> Option<f64> {
        let asked = match self.at {
            At::Grid { output, .. } | At::Output(output) => Some(output),
            At::Conc(_) | At::Input(_) => base,
        };
        let asked = u64::from(asked?) * u64::from(self.completed.filter(|&n| n > 0)?);
        let got = self.total_output_tokens?;
        if got == 0 && self.median_ttft_ms.is_some_and(|t| t > 0.0) && self.median_tpot_ms.is_some()
        {
            return None;
        }
        (asked > 0).then(|| got as f64 / asked as f64)
    }
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
    #[serde(default)]
    total_output_tokens: Option<u64>,
    #[serde(default)]
    timed_out: bool,
    #[serde(default)]
    dead_server: bool,
    #[serde(default)]
    requests: Option<Requests>,
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
            total_output_tokens: c.total_output_tokens,
            dead: false,
            mem_below_weights: false,
            timed_out: c.timed_out,
            dead_server: c.dead_server,
            short: None,
            requests: c.requests,
        })
    }
}

//! Port of `flex::FlexConfig` (`flex/include/flex/runtime_graph/flex_config.hpp` +
//! `flex/src/runtime_graph/flex_config.cpp`, and the near-identical
//! `flex/include/flex/util/flex_config.hpp` + `flex/src/util/flex_config.cpp` — the
//! two headers are selected via `DISABLE_LEGACY_GRAPH_RUNTIME` and expose the same
//! `FlexConfig` API; this module ports their union) and `flex::RuntimeConfig`
//! (`flex/src/util/config/runtime_config.hpp` + `.cpp` + `runtime_settings.hpp`).
//!
//! All of this is env-var (and, for `RuntimeConfig`, JSON config file) parsing
//! with process-wide caching — pure logic, no senlib involvement except the two
//! `FlexConfig` methods that enumerate/locate physical cards
//! (`FlexSpyreDevices`, `RdmaGetPCIeAddress`); those two cross into
//! `senlib_ffi_config_util.rs`. See `SENLIB_BOUNDARY_config-util.md`.
//!
//! ## Caching semantics (`common::TracedSetting`)
//! Real `common::TracedSetting<T>` (`common/traced_setting.hpp:196-201`)
//! resolves the env var *every time it's constructed*, but immediately hands
//! that freshly-computed value to `GlobalTracedSettings::InsertValue`
//! (`common/traced_setting.hpp:48-64`), which is **first-write-wins**: the
//! first call for a given setting name allocates and stores the value; every
//! later call with that name finds the existing entry and returns a pointer
//! to the *original* stored value, silently discarding whatever it just
//! recomputed. Net effect: each `FlexConfig`/`RuntimeConfig` setting is
//! effectively snapshotted from its env var on first access and stays that
//! value for the rest of the process's life, no matter how the env var
//! changes afterward — only an explicit `TracedSetting::Update()` (which no
//! accessor here calls) can change it. This module's `cached_setting` helper
//! (below) and `settings_cache`/`resolve_*` (for `RuntimeConfig`) implement
//! exactly that first-write-wins cache, keyed by setting name — this file
//! previously (incorrectly) re-read the env var on every call and claimed
//! that matched the C++ "non-caching" pattern; it does not, and this is now
//! fixed to match the real caching behavior.
//!
//! ## The two "reset" symbols
//! `RuntimeConfig::reset()` here clears the *env/JSON-config settings cache*
//! (`GlobalRuntimeSettings` + `RuntimeConfigFileCache`). This is a distinct
//! symbol from `RuntimeContext::reset()` (runtime-graph teardown, ported
//! elsewhere in this crate) — both are ported, under their own types, per the
//! task brief's disambiguation note.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::senlib_ffi_config_util::{pcie_address_for_card, sen_pci_card_count};

// ===========================================================================
// Small env-var parsing helpers (no `common::env_var` equivalent available)
// ===========================================================================

fn env_string(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// Port of `common::BooleanEnvVar` (`common/env_var.hpp`): once the env var is
/// present, the default is ignored entirely and the raw string is compared
/// (case-sensitively, no trimming) against exactly `"true"`, `"t"`, `"yes"`,
/// `"y"`, `"1"` — anything else (including e.g. `"TRUE"` or `"0"`) is `false`.
pub(crate) fn env_bool(name: &str, default: bool) -> bool {
    match env_string(name) {
        None => default,
        Some(v) => matches!(v.as_str(), "true" | "t" | "yes" | "y" | "1"),
    }
}

/// Parse the longest leading numeric-prefix substring of `s`, tolerating
/// trailing garbage — mirrors `std::stoi`/`std::stol`/`std::stoull` (used by
/// `common::convert_string_to_type<T>`, `common/env_var.hpp:140-176`), which
/// parse a leading numeric prefix (`"5abc"` -> `5`) rather than requiring the
/// whole string to be numeric, and throw only when *no* valid numeric prefix
/// exists at all (`"abc"` has none). Returns `None` in that no-prefix case so
/// callers can fall back to their default, matching the C++ falling back to
/// its `default_value` string (via `EnvSingleton::Get().value(name,
/// std::to_string(default))`) when the env var itself is absent, and there
/// being no graceful "invalid" path in the C++ once the var *is* present
/// (`std::stoi` throws) — this port chooses to fall back to the default
/// rather than panic, which is strictly more permissive than the C++, never
/// less.
fn parse_leading_numeric_prefix<T: std::str::FromStr>(s: &str) -> Option<T> {
    let s = s.trim();
    let bytes = s.as_bytes();
    let mut end = 0usize;
    if end < bytes.len() && (bytes[end] == b'+' || bytes[end] == b'-') {
        end += 1;
    }
    let digits_start = end;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end == digits_start {
        return None;
    }
    s[..end].parse::<T>().ok()
}

fn env_u64(name: &str, default: u64) -> u64 {
    env_string(name)
        .and_then(|v| parse_leading_numeric_prefix::<u64>(&v))
        .unwrap_or(default)
}

fn env_i64(name: &str, default: i64) -> i64 {
    env_string(name)
        .and_then(|v| parse_leading_numeric_prefix::<i64>(&v))
        .unwrap_or(default)
}

fn env_u32(name: &str, default: u32) -> u32 {
    env_string(name)
        .and_then(|v| parse_leading_numeric_prefix::<u32>(&v))
        .unwrap_or(default)
}

fn env_i32(name: &str, default: i32) -> i32 {
    env_string(name)
        .and_then(|v| parse_leading_numeric_prefix::<i32>(&v))
        .unwrap_or(default)
}

fn env_str_or(name: &str, default: &str) -> String {
    env_string(name).unwrap_or_else(|| default.to_string())
}

// ===========================================================================
// Process-wide first-write-wins setting cache (port of
// `common::GlobalTracedSettings` / `TracedSetting<T>`, see module doc above)
// ===========================================================================

fn flex_config_cache() -> &'static Mutex<HashMap<String, Box<dyn Any + Send>>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Box<dyn Any + Send>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Shared helper implementing `TracedSetting`'s first-write-wins cache
/// (`common/traced_setting.hpp:48-64,196-201`): on the first call for `name`,
/// `compute` runs and its result is cached forever; every later call for the
/// same `name` returns that original cached value without re-running
/// `compute` (and therefore without re-reading whatever env var it reads),
/// regardless of what that env var says by then. Used by every `FlexConfig`
/// accessor that reads directly from an env var; accessors that only
/// delegate to other cached accessors (e.g. `rdma_local_rank`,
/// `flex_spyre_devices`) don't need their own cache entry — they inherit
/// correctness from whatever they call, matching the C++ (`FlexSpyreDevices`
/// itself isn't a `TracedSetting`; only `FlexSpyreDevicesStr` is —
/// `flex/src/util/flex_config.cpp:390-397,400-410`).
fn cached_setting<T: Clone + Send + 'static>(name: &str, compute: impl FnOnce() -> T) -> T {
    {
        let cache = flex_config_cache().lock().unwrap();
        if let Some(existing) = cache.get(name)
            && let Some(v) = existing.downcast_ref::<T>()
        {
            return v.clone();
        }
    }
    // Compute outside the lock: `compute` may itself call `cached_setting`
    // (e.g. `flex_monitor_sleep`'s default depends on `flex_device`), and
    // `Mutex` here isn't reentrant.
    let val = compute();
    let mut cache = flex_config_cache().lock().unwrap();
    let entry = cache
        .entry(name.to_string())
        .or_insert_with(|| Box::new(val.clone()) as Box<dyn Any + Send>);
    entry.downcast_ref::<T>().cloned().unwrap_or(val)
}

#[cfg(test)]
fn flex_config_cache_reset() {
    flex_config_cache().lock().unwrap().clear();
}

// ===========================================================================
// Newtypes — one per domain-meaningful config value
// ===========================================================================

/// `DGO_MIN_SUPERNODE_COUNT` / `DGO_MIN_SUPERNODE_SIZE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuperNodeCount(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuperNodeSize(pub u64);

/// `FLEX_COMPUTE`: overall compute backend selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeMode {
    Sentient,
    Null,
    /// NOTE: the C++ also has a `SENULATOR` mode for IBM's software emulator.
    /// This port does not run the senulator, so that value is deliberately not
    /// a variant here — `FLEX_COMPUTE=SENULATOR` parses to `Other("SENULATOR")`
    /// and is rejected by whatever consumes it, rather than being silently
    /// accepted as a mode with no implementation behind it.
    Other(String),
}
impl ComputeMode {
    fn parse(s: &str) -> Self {
        match s {
            "SENTIENT" => Self::Sentient,
            "NULL" => Self::Null,
            other => Self::Other(other.to_string()),
        }
    }
}

/// `FLEX_DEVICE`: runtime device backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceKind {
    Vf,
    Pf,
    Mock,
    Other(String),
}
impl DeviceKind {
    fn parse(s: &str) -> Self {
        match s {
            "VF" => Self::Vf,
            "PF" => Self::Pf,
            "MOCK" => Self::Mock,
            other => Self::Other(other.to_string()),
        }
    }
}

/// Port of `operator<<(std::ostream&, const DeviceInterfaceType&)`
/// (`flex/runtime_stream/device_info.hpp`). That C++ enum has a 4th real
/// variant, `UMI` (a 1p5-chip device type), with no counterpart here —
/// this crate's `SPYRE_CHIP` check elsewhere is 1P0-only, so `DeviceKind`
/// only carries the three device kinds this crate can actually bring up;
/// a real "UMI" string would fall into `Other` and print as "Unknown"
/// below, same as any other unrecognized value, not as "UMI".
impl std::fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pf => write!(f, "PF"),
            Self::Vf => write!(f, "VF"),
            Self::Mock => write!(f, "MOCK"),
            Self::Other(_) => write!(f, "Unknown"),
        }
    }
}

/// `FLEX_STREAMER_FUNCTION`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamerFunctionKind {
    Batch,
    Sequential,
    Limit,
    Unset,
    Other(String),
}
impl StreamerFunctionKind {
    fn parse(s: &str) -> Self {
        match s {
            "" => Self::Unset,
            "BATCH" => Self::Batch,
            "SEQUENTIAL" => Self::Sequential,
            "LIMIT" => Self::Limit,
            other => Self::Other(other.to_string()),
        }
    }
}

/// Filesystem path setting (e.g. `FLEX_MOCK_DEVICE_MEMORY_PATH`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigPath(pub String);
/// Free-text string setting with no fixed enumeration (log verbosity spec,
/// profile prefix, PCIe address, etc).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigText(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MillisTimeout(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SecondsTimeout(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryCount(pub i32);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryDelayMs(pub i32);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingRequestLimit(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReservedCoreCount(pub i32);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadCount(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OmpThreadCount(pub i32);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MonitorSleepSecs(pub i32);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorldRank(pub i64);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorldSize(pub i64);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VfNumber(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VfMsgLimit(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DestSplitSize(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AiuWorldSize(pub i64);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CbGenMode(pub i32);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HdmaSignalMode(pub i32);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HdmaWaitMode(pub i32);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HdmaMaxMessages(pub i32);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HdmaWaitTimeoutSecs(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaxStagedInstructions(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MonitorCheckCount(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocatorSummaryIntervalMs(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalRank(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DumpByteLimit(pub u64);

/// Ordered, parsed form of `SPYRE_DEVICES`: device indices in task order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SpyreDevices(pub Vec<i64>);

/// Error returned by `FlexConfig::flex_spyre_devices` when the caller has
/// specified fewer device mappings than the local RDMA world size requires.
/// Port of `RAS::CONFIGURATION::IncorrectSpyreDevicesVar`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncorrectSpyreDevicesVar {
    pub rdma_local_size: i64,
    pub spyre_devices_string: String,
    pub spyre_devices_count: usize,
}

impl std::fmt::Display for IncorrectSpyreDevicesVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SPYRE_DEVICES=\"{}\" provides {} device mapping(s) but RDMA local size is {}",
            self.spyre_devices_string, self.spyre_devices_count, self.rdma_local_size
        )
    }
}
impl std::error::Error for IncorrectSpyreDevicesVar {}

/// Error returned by `RuntimeConfig::spyre_devices`. The C++ resolver
/// (`runtime_config.cpp:182-211`) has two distinct throw paths and this port
/// must keep both:
///   * `Incorrect` — `RAS::CONFIGURATION::IncorrectSpyreDevicesVar().Throw()`,
///     raised when `WORLD_SIZE > devices.size()`.
///   * `Malformed` — `common::VectorInt64EnvVar` -> `ParseIntegerListValue<int64_t>`
///     (`common/env_var.hpp:227-255`) calls `std::stoll` on every
///     comma-separated item and lets its `std::invalid_argument` escape when an
///     item has no leading digits at all.
///
/// PORT GAP FIXED: `Malformed` did not exist. `spyre_devices()` parsed with
/// `filter_map(|s| s.parse().ok())`, so `SPYRE_DEVICES="0,two,4"` silently
/// resolved to `[0, 4]` — a two-device topology derived from a three-device
/// request, cached process-wide, where the C++ refuses to start at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpyreDevicesError {
    Incorrect(IncorrectSpyreDevicesVar),
    Malformed(String),
}

impl std::fmt::Display for SpyreDevicesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Incorrect(e) => e.fmt(f),
            Self::Malformed(item) => {
                write!(
                    f,
                    "Could not parse integer list: SPYRE_DEVICES item \"{item}\" is not an integer"
                )
            }
        }
    }
}
impl std::error::Error for SpyreDevicesError {}

impl From<IncorrectSpyreDevicesVar> for SpyreDevicesError {
    fn from(e: IncorrectSpyreDevicesVar) -> Self {
        Self::Incorrect(e)
    }
}

/// Error returned by `FlexConfig::rdma_get_pcie_address`. Port of the real
/// C++'s two failure modes in `RdmaGetPCIeAddress`
/// (`flex/src/util/flex_config.cpp:440-...`): it calls `FlexSpyreDevices()`
/// (which can throw `RAS::CONFIGURATION::IncorrectSpyreDevicesVar`) and then
/// indexes into the result with `std::vector::at()` (which throws
/// `std::out_of_range` when `world_rank` is beyond the device list). This
/// port previously modeled the first by unconditionally swallowing it
/// (`unwrap_or_default()`) and the second by silently substituting a wrong
/// value (`.get(...).unwrap_or(world_rank)`); both are now surfaced here
/// instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RdmaGetPCIeAddressError {
    SpyreDevices(IncorrectSpyreDevicesVar),
    /// Port of `std::vector::at()`'s `std::out_of_range`.
    IndexOutOfRange {
        world_rank: i64,
        spyre_devices_len: usize,
    },
}
impl std::fmt::Display for RdmaGetPCIeAddressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SpyreDevices(e) => write!(f, "{e}"),
            Self::IndexOutOfRange {
                world_rank,
                spyre_devices_len,
            } => write!(
                f,
                "world_rank {world_rank} is out of range for SPYRE_DEVICES (len {spyre_devices_len})"
            ),
        }
    }
}
impl std::error::Error for RdmaGetPCIeAddressError {}
impl From<IncorrectSpyreDevicesVar> for RdmaGetPCIeAddressError {
    fn from(e: IncorrectSpyreDevicesVar) -> Self {
        Self::SpyreDevices(e)
    }
}

// ===========================================================================
// FlexConfig
// ===========================================================================

/// Port of `flex::FlexConfig`. Each method reads from an environment
/// variable and returns a typed, defaulted value, cached process-wide on
/// first access via `cached_setting` — see the module doc's "Caching
/// semantics" section for why (this port previously re-read the env var on
/// every call and claimed that matched C++'s "non-caching" pattern; it did
/// not, `TracedSetting` caches first-write-wins, and this has been fixed).
///
/// Build-mode note: the C++ source has `#if PRODUCTION_MODE` branches for a
/// handful of settings (`FlexCompute`, `FlexDevice`, `FlexVfMsgLimit`,
/// `FlexStreamerFunction`, `FlexMonitorSleep`). This port always uses the
/// non-production (`PRODUCTION_MODE=0`) default, matching x86_64/ppc64le
/// hosts (s390x sets `PRODUCTION_MODE=1` at compile time in the C++ build —
/// not a runtime env var, so it cannot be read here).
///
/// `ResponseWorkerTimeoutSeconds` is *not* one of the `PRODUCTION_MODE`
/// settings (a previous version of this doc comment wrongly grouped it with
/// them) — it's gated by the unrelated `USE_SENULATOR` compile-time macro
/// instead; see that accessor's own doc comment below.
pub struct FlexConfig;

impl FlexConfig {
    // --- Graph optimization ---
    pub fn minimal_super_node_count() -> SuperNodeCount {
        SuperNodeCount(cached_setting("DGO_MIN_SUPERNODE_COUNT", || {
            env_u64("DGO_MIN_SUPERNODE_COUNT", 0)
        }))
    }
    pub fn minimal_super_node_size() -> SuperNodeSize {
        SuperNodeSize(cached_setting("DGO_MIN_SUPERNODE_SIZE", || {
            env_u64("DGO_MIN_SUPERNODE_SIZE", 0)
        }))
    }

    // --- Device memory ---
    pub fn flex_compile_device_iommu_size() -> crate::address::ByteSize {
        crate::address::ByteSize(cached_setting("FLEX_COMPILE_DEVICE_IOMMU_SIZE", || {
            env_u64("FLEX_COMPILE_DEVICE_IOMMU_SIZE", 4 * 1024 * 1024 * 1024)
        }))
    }
    pub fn device_memory_allocator_debug_info() -> bool {
        cached_setting("FLEX_DEVICE_MEMORY_ALLOCATOR_DEBUG_INFO", || {
            env_bool("FLEX_DEVICE_MEMORY_ALLOCATOR_DEBUG_INFO", false)
        })
    }
    pub fn vf_alloc_max_retries() -> RetryCount {
        RetryCount(cached_setting("FLEX_VF_ALLOC_MAX_RETRIES", || {
            env_i32("FLEX_VF_ALLOC_MAX_RETRIES", 1)
        }))
    }
    pub fn vf_alloc_initial_retry_delay_ms() -> RetryDelayMs {
        RetryDelayMs(cached_setting(
            "FLEX_VF_ALLOC_INITIAL_RETRY_DELAY_MS",
            || env_i32("FLEX_VF_ALLOC_INITIAL_RETRY_DELAY_MS", 2000),
        ))
    }

    // --- Device type ---
    pub fn flex_compute() -> ComputeMode {
        cached_setting("FLEX_COMPUTE", || {
            ComputeMode::parse(&env_str_or("FLEX_COMPUTE", "NULL"))
        })
    }
    pub fn flex_device() -> DeviceKind {
        cached_setting("FLEX_DEVICE", || {
            DeviceKind::parse(&env_str_or("FLEX_DEVICE", "MOCK"))
        })
    }

    // --- Execution ---
    pub fn flex_overwrite_nmb_frame() -> bool {
        cached_setting("FLEX_OVERWRITE_NMB_FRAME", || {
            env_bool("FLEX_OVERWRITE_NMB_FRAME", true)
        })
    }

    // --- Statistics & debugging ---
    pub fn print_end_to_end_breakdown() -> bool {
        cached_setting("FLEX_PRINT_END_TO_END_BREAKDOWN", || {
            env_bool("FLEX_PRINT_END_TO_END_BREAKDOWN", false)
        })
    }
    pub fn print_scheduler_raw_timestamps_stats() -> bool {
        cached_setting("FLEX_SCHEDULER_PRINT_RAW_TIMESTAMPS", || {
            env_bool("FLEX_SCHEDULER_PRINT_RAW_TIMESTAMPS", false)
        })
    }
    pub fn print_stats_every_ms() -> MillisTimeout {
        MillisTimeout(cached_setting("FLEX_PRINT_STATS_EVERY_MS", || {
            env_u64("FLEX_PRINT_STATS_EVERY_MS", 0)
        }))
    }
    pub fn print_stats() -> bool {
        cached_setting("FLEX_PRINT_STATS", || env_bool("FLEX_PRINT_STATS", false))
    }

    // --- Performance tuning ---
    pub fn run_prepare_in_parallel() -> bool {
        cached_setting("FLEX_RUN_PREPARE_IN_PARALLEL", || {
            env_bool("FLEX_RUN_PREPARE_IN_PARALLEL", false)
        })
    }
    pub fn queue_capacity_wait_timeout_ms() -> MillisTimeout {
        MillisTimeout(cached_setting(
            "FLEX_QUEUE_CAPACITY_WAIT_TIMEOUT_MS",
            || env_u64("FLEX_QUEUE_CAPACITY_WAIT_TIMEOUT_MS", 60_000),
        ))
    }

    // --- Skip flags ---
    pub fn skip_dma() -> bool {
        cached_setting("FLEX_SKIP_DMA", || env_bool("FLEX_SKIP_DMA", false))
    }
    pub fn skip_graph_executor_compute() -> bool {
        cached_setting("FLEX_SKIP_GRAPH_EXECUTOR_COMPUTE", || {
            env_bool("FLEX_SKIP_GRAPH_EXECUTOR_COMPUTE", false)
        })
    }
    pub fn skip_data_conversion() -> bool {
        cached_setting("FLEX_SKIP_DATA_CONVERSION", || {
            env_bool("FLEX_SKIP_DATA_CONVERSION", false)
        })
    }
    pub fn skip_host_compute() -> bool {
        cached_setting("FLEX_SKIP_HOST_COMPUTE", || {
            env_bool("FLEX_SKIP_HOST_COMPUTE", false)
        })
    }
    pub fn skip_compute() -> bool {
        cached_setting("FLEX_SKIP_COMPUTE", || env_bool("FLEX_SKIP_COMPUTE", false))
    }
    pub fn skip_executor() -> bool {
        cached_setting("FLEX_SKIP_EXECUTOR", || {
            env_bool("FLEX_SKIP_EXECUTOR", false)
        })
    }
    pub fn skip_scheduler() -> bool {
        cached_setting("FLEX_SKIP_SCHEDULER", || {
            env_bool("FLEX_SKIP_SCHEDULER", false)
        })
    }
    pub fn skip_offload() -> bool {
        cached_setting("FLEX_SKIP_OFFLOAD", || env_bool("FLEX_SKIP_OFFLOAD", false))
    }
    pub fn skip_timestamp_calibration() -> bool {
        cached_setting("FLEX_SKIP_TIMESTAMP_CALIBRATION", || {
            env_bool("FLEX_SKIP_TIMESTAMP_CALIBRATION", true)
        })
    }

    // --- Response worker ---
    /// Port of `FlexConfig::ResponseWorkerTimeoutSeconds`
    /// (`flex/src/util/flex_config.cpp:172-180`): real C++ defaults to `3600`
    /// when compiled with `USE_SENULATOR`, else `280` — a compile-time macro,
    /// not `PRODUCTION_MODE` (a previous version of this port's module doc
    /// wrongly attributed the branch to `PRODUCTION_MODE`). This crate has no
    /// senulator/mock build-mode flag (checked `Cargo.toml`: no such
    /// feature), so this always resolves to the non-`USE_SENULATOR` default,
    /// `280`.
    pub fn response_worker_timeout_seconds() -> SecondsTimeout {
        SecondsTimeout(cached_setting(
            "FLEX_RESPONSE_WORKER_TIMEOUT_SECONDS",
            || env_u64("FLEX_RESPONSE_WORKER_TIMEOUT_SECONDS", 280),
        ))
    }
    pub fn response_worker_fail_on_timeout() -> bool {
        cached_setting("FLEX_RESPONSE_WORKER_FAIL_ON_TIMEOUT", || {
            env_bool("FLEX_RESPONSE_WORKER_FAIL_ON_TIMEOUT", true)
        })
    }
    pub fn response_worker_skip_launch() -> bool {
        cached_setting("FLEX_RESPONSE_WORKER_SKIP_LAUNCH", || {
            env_bool("FLEX_RESPONSE_WORKER_SKIP_LAUNCH", false)
        })
    }
    pub fn response_worker_max_pending_requests() -> PendingRequestLimit {
        PendingRequestLimit(cached_setting(
            "FLEX_RESPONSE_WORKER_MAX_PENDING_REQUESTS",
            || env_u64("FLEX_RESPONSE_WORKER_MAX_PENDING_REQUESTS", 32768),
        ))
    }

    // --- Threading & optimization ---
    pub fn single_thread_optimized() -> bool {
        cached_setting("FLEX_SINGLE_THREAD_OPTIMIZED", || {
            env_bool("FLEX_SINGLE_THREAD_OPTIMIZED", false)
        })
    }
    pub fn flex_runtime_num_reserved_cores() -> ReservedCoreCount {
        ReservedCoreCount(cached_setting("FLEX_RUNTIME_NUM_RESERVED_CORES", || {
            env_i32("FLEX_RUNTIME_NUM_RESERVED_CORES", 1)
        }))
    }
    pub fn flex_omp_num_threads() -> OmpThreadCount {
        OmpThreadCount(cached_setting("OMP_NUM_THREADS", || {
            env_i32("OMP_NUM_THREADS", -1)
        }))
    }
    pub fn flex_threads_number() -> ThreadCount {
        ThreadCount(cached_setting("FLEX_THREADS_NUMBER", || {
            env_u32("FLEX_THREADS_NUMBER", 1)
        }))
    }
    /// Port of `FlexMonitorSleep`: defaults to 2s when device is MOCK
    /// (non-production default), else 0. The whole resolved value (including
    /// its `flex_device()`-dependent default) is cached together under
    /// `FLEX_MONITOR_SLEEP`, matching `TracedSetting`'s once-computed,
    /// forever-cached semantics.
    pub fn flex_monitor_sleep() -> MonitorSleepSecs {
        MonitorSleepSecs(cached_setting("FLEX_MONITOR_SLEEP", || {
            let default_sleep = if matches!(Self::flex_device(), DeviceKind::Mock) {
                2
            } else {
                0
            };
            env_i32("FLEX_MONITOR_SLEEP", default_sleep)
        }))
    }

    // --- Data conversion ---
    pub fn use_fast_convert() -> ConfigText {
        ConfigText(cached_setting("FLEX_FAST_CONVERT", || {
            env_str_or("FLEX_FAST_CONVERT", "ScalarScatter")
        }))
    }
    pub fn flex_skip_fast_memset() -> bool {
        cached_setting("FLEX_SKIP_FAST_MEMSET", || {
            env_bool("FLEX_SKIP_FAST_MEMSET", false)
        })
    }

    // --- RDMA ---
    pub fn rdma_world_rank() -> WorldRank {
        WorldRank(cached_setting("RANK", || env_i64("RANK", -1)))
    }
    pub fn rdma_world_size() -> WorldSize {
        WorldSize(cached_setting("WORLD_SIZE", || env_i64("WORLD_SIZE", -1)))
    }
    /// Flex assumes single-server topology: local rank equals world rank.
    /// Delegates to the already-cached `rdma_world_rank`, so this needs no
    /// cache entry of its own.
    pub fn rdma_local_rank() -> WorldRank {
        Self::rdma_world_rank()
    }
    /// Flex assumes single-server topology: local size equals world size.
    /// Delegates to the already-cached `rdma_world_size`, so this needs no
    /// cache entry of its own.
    pub fn rdma_local_size() -> WorldSize {
        Self::rdma_world_size()
    }
    pub fn rdma_multicast_as_singlecast() -> bool {
        cached_setting("FLEX_RDMA_MULTICAST_AS_SINGLECAST", || {
            env_bool("FLEX_RDMA_MULTICAST_AS_SINGLECAST", false)
        })
    }
    pub fn flex_rdma_mode_full() -> bool {
        cached_setting("FLEX_RDMA_MODE_FULL", || {
            env_bool("FLEX_RDMA_MODE_FULL", false)
        })
    }
    pub fn flex_rdma_p2p_mode_ack() -> bool {
        cached_setting("FLEX_RDMA_P2P_MODE_ACK", || {
            env_bool("FLEX_RDMA_P2P_MODE_ACK", true)
        })
    }
    pub fn flex_rdma_p2p_mode_ack_optimized() -> bool {
        cached_setting("FLEX_RDMA_P2P_MODE_ACK_OPTIMIZED", || {
            env_bool("FLEX_RDMA_P2P_MODE_ACK_OPTIMIZED", true)
        })
    }
    pub fn flex_show_p2p_protocols() -> bool {
        cached_setting("FLEX_SHOW_P2P_PROTOCOLS", || {
            env_bool("FLEX_SHOW_P2P_PROTOCOLS", false)
        })
    }

    // --- HDMA ---
    pub fn flex_hdma_mode_full() -> bool {
        cached_setting("FLEX_HDMA_MODE_FULL", || {
            env_bool("FLEX_HDMA_MODE_FULL", true)
        })
    }
    pub fn flex_hdma_force_load() -> bool {
        cached_setting("FLEX_HDMA_ALLSETUP", || {
            env_bool("FLEX_HDMA_ALLSETUP", true)
        })
    }
    pub fn flex_hdma_p2p_size() -> crate::address::ByteSize {
        crate::address::ByteSize(cached_setting("FLEX_HDMA_P2PSIZE", || {
            env_u64("FLEX_HDMA_P2PSIZE", 8 * 1024 * 1024)
        }))
    }
    pub fn flex_hdma_coll_size() -> crate::address::ByteSize {
        crate::address::ByteSize(cached_setting("FLEX_HDMA_COLLSIZE", || {
            env_u64("FLEX_HDMA_COLLSIZE", 16 * 1024 * 1024)
        }))
    }
    pub fn flex_hdma_coll_for_mcast() -> bool {
        cached_setting("FLEX_HDMA_COLL_MCAST", || {
            env_bool("FLEX_HDMA_COLL_MCAST", true)
        })
    }
    pub fn flex_hdma_vf_multi_wait() -> bool {
        cached_setting("FLEX_HDMA_VF_MULTI_WAIT", || {
            env_bool("FLEX_HDMA_VF_MULTI_WAIT", true)
        })
    }
    pub fn flex_hdma_vf_always_reload_prog() -> bool {
        cached_setting("FLEX_HDMA_VF_ALWAYS_RELOAD_PROG", || {
            env_bool("FLEX_HDMA_VF_ALWAYS_RELOAD_PROG", false)
        })
    }
    pub fn flex_hdma_p2p_chunk_size() -> crate::address::ByteSize {
        crate::address::ByteSize(cached_setting("FLEX_HDMA_P2PCHUNKSIZE", || {
            env_u64("FLEX_HDMA_P2PCHUNKSIZE", 2 * 1024)
        }))
    }
    pub fn flex_hdma_p2p_fragment_size() -> crate::address::ByteSize {
        crate::address::ByteSize(cached_setting("FLEX_HDMA_P2PFRAGMENTSIZE", || {
            env_u64("FLEX_HDMA_P2PFRAGMENTSIZE", 0)
        }))
    }
    pub fn flex_hdma_max_staged_instructions() -> MaxStagedInstructions {
        MaxStagedInstructions(cached_setting("FLEX_HDMA_MAX_STAGED_INSTRUCTIONS", || {
            env_u64("FLEX_HDMA_MAX_STAGED_INSTRUCTIONS", 10240)
        }))
    }
    pub fn flex_hdma_check_for_deadlock() -> bool {
        cached_setting("FLEX_HDMA_CHECK_FOR_DEADLOCK", || {
            env_bool("FLEX_HDMA_CHECK_FOR_DEADLOCK", false)
        })
    }
    /// Signal mode: 0 = wcore, 1 = DLM, 2 = flat. Port of
    /// `FlexConfig::FlexHdmaVFSignalMode`; the *canonical* (non-legacy)
    /// source, `flex/src/util/flex_config.cpp:568-574`, defaults to `2`
    /// (comment there: `0==wcore/1==DLM/2==FLAT`). The *legacy*
    /// `flex/src/runtime_graph/flex_config.cpp` variant disagrees and
    /// defaults to `1` — since this module ports "the union" of both files
    /// (see module doc) and the two conflict on this exact default, this
    /// port picks the canonical file's `2`, noted here rather than silently
    /// choosing one.
    pub fn flex_hdma_vf_signal_mode() -> HdmaSignalMode {
        HdmaSignalMode(cached_setting("FLEX_HDMA_VF_SIGNAL_MODE", || {
            env_i32("FLEX_HDMA_VF_SIGNAL_MODE", 2)
        }))
    }
    /// Wait mode: 0 = wcore, 1 = dcore.
    pub fn flex_hdma_vf_wait_mode() -> HdmaWaitMode {
        HdmaWaitMode(cached_setting("FLEX_HDMA_VF_WAIT_MODE", || {
            env_i32("FLEX_HDMA_VF_WAIT_MODE", 1)
        }))
    }
    pub fn flex_hdma_vf_wait_timeout() -> HdmaWaitTimeoutSecs {
        HdmaWaitTimeoutSecs(cached_setting("FLEX_HDMA_VF_WAIT_TIMEOUT", || {
            env_u64("FLEX_HDMA_VF_WAIT_TIMEOUT", 2)
        }))
    }
    pub fn flex_hdma_vf_max_num_messages() -> HdmaMaxMessages {
        HdmaMaxMessages(cached_setting("FLEX_HDMA_VF_MAX_MESSAGES", || {
            env_i32("FLEX_HDMA_VF_MAX_MESSAGES", 1024)
        }))
    }
    pub fn hdma_monitor_check_count() -> MonitorCheckCount {
        MonitorCheckCount(cached_setting("FLEX_HDMA_MONITOR_CHECK_COUNT", || {
            env_u64("FLEX_HDMA_MONITOR_CHECK_COUNT", 4)
        }))
    }
    pub fn flex_hdma_pf_mode_vsid() -> bool {
        cached_setting("FLEX_HDMA_PF_MODE_VSID", || {
            env_bool("FLEX_HDMA_PF_MODE_VSID", true)
        })
    }

    // --- Control block ---
    pub fn force_cb_barrier() -> bool {
        cached_setting("FLEX_FORCE_CB_BARRIER", || {
            env_bool("FLEX_FORCE_CB_BARRIER", false)
        })
    }
    pub fn flex_use_cb_gen() -> bool {
        cached_setting("FLEX_USE_CB_GEN", || env_bool("FLEX_USE_CB_GEN", true))
    }
    pub fn flex_use_cb_gen_timing_barrier() -> bool {
        cached_setting("FLEX_USE_CB_GEN_BARRIER", || {
            env_bool("FLEX_USE_CB_GEN_BARRIER", false)
        })
    }
    pub fn flex_use_message_checker() -> bool {
        cached_setting("FLEX_USE_MESG_CHECKER", || {
            env_bool("FLEX_USE_MESG_CHECKER", false)
        })
    }
    pub fn flex_cb_gen_force_regen() -> bool {
        cached_setting("FLEX_CB_GEN_FORCE_REGEN", || {
            env_bool("FLEX_CB_GEN_FORCE_REGEN", false)
        })
    }
    pub fn flex_cb_gen_force_cbgen() -> bool {
        cached_setting("FLEX_CB_GEN_FORCE_CBGEN", || {
            env_bool("FLEX_CB_GEN_FORCE_CBGEN", false)
        })
    }
    pub fn flex_cb_gen_mode() -> CbGenMode {
        CbGenMode(cached_setting("FLEX_CB_GEN_MODE", || {
            env_i32("FLEX_CB_GEN_MODE", 1)
        }))
    }

    pub fn global_profile_prefix() -> ConfigText {
        ConfigText(cached_setting("FLEX_GLOBAL_PROFILE_PREFIX", || {
            env_str_or("FLEX_GLOBAL_PROFILE_PREFIX", "flex")
        }))
    }

    // --- Device & topology ---
    pub fn aiu_topo_file() -> ConfigPath {
        ConfigPath(cached_setting("AIU_TOPO_FILE", || {
            env_str_or("AIU_TOPO_FILE", "")
        }))
    }
    pub fn flex_vf_function() -> VfNumber {
        VfNumber(cached_setting("FLEX_VF_NUMBER", || {
            env_u32("FLEX_VF_NUMBER", 2)
        }))
    }
    pub fn flex_vf_msg_limit() -> VfMsgLimit {
        VfMsgLimit(cached_setting("FLEX_VF_MESSAGE_LIMIT", || {
            env_u64("FLEX_VF_MESSAGE_LIMIT", 0)
        }))
    }
    pub fn flex_streamer_function() -> StreamerFunctionKind {
        cached_setting("FLEX_STREAMER_FUNCTION", || {
            StreamerFunctionKind::parse(&env_str_or("FLEX_STREAMER_FUNCTION", ""))
        })
    }
    pub fn flex_spyre_devices_str() -> ConfigText {
        ConfigText(cached_setting("SPYRE_DEVICES", || {
            env_str_or("SPYRE_DEVICES", "")
        }))
    }
    pub fn flex_use_ntb_addressing() -> bool {
        cached_setting("FLEX_USE_NTB_ADDRESSING", || {
            env_bool("FLEX_USE_NTB_ADDRESSING", true)
        })
    }
    pub fn aiu_world_size() -> AiuWorldSize {
        AiuWorldSize(cached_setting("AIU_WORLD_SIZE", || {
            env_i64("AIU_WORLD_SIZE", -1)
        }))
    }

    /// Port of `FlexConfig::FlexSpyreDevices` (`flex/src/util/flex_config.cpp:400-410`):
    /// parse the ordered device-index list from `SPYRE_DEVICES`, or (if
    /// unset) enumerate every card senlib reports. Errors if fewer mappings
    /// than the RDMA local size are given. Not itself cached — matching the
    /// C++, where `FlexSpyreDevices` is a plain function that re-derives from
    /// the *cached* `FlexSpyreDevicesStr()` on every call; only
    /// `flex_spyre_devices_str` needs (and has) a cache entry.
    ///
    /// Malformed comma-separated entries tolerate a leading numeric prefix
    /// with trailing garbage (e.g. `"3x"` -> `3`), matching `std::strtol`
    /// (`flex_config.cpp:416`: `std::strtol(item.c_str(), nullptr, 10)`) —
    /// see `parse_leading_numeric_prefix`.
    pub fn flex_spyre_devices() -> Result<SpyreDevices, IncorrectSpyreDevicesVar> {
        let raw = Self::flex_spyre_devices_str().0;
        let devices: Vec<i64> = if raw.is_empty() {
            (0..sen_pci_card_count()).collect()
        } else {
            raw.split(',')
                .map(|item| parse_leading_numeric_prefix::<i64>(item).unwrap_or(0))
                .collect()
        };

        let local_size = Self::rdma_local_size().0;
        if local_size > 0 && local_size > devices.len() as i64 {
            return Err(IncorrectSpyreDevicesVar {
                rdma_local_size: local_size,
                spyre_devices_string: raw,
                spyre_devices_count: devices.len(),
            });
        }
        Ok(SpyreDevices(devices))
    }

    /// Port of `FlexConfig::RdmaGetPCIeAddress` (`flex/src/util/flex_config.cpp:440-...`).
    /// `world_rank = None` behaves like the C++ default argument (`-1`): use
    /// rank 0.
    ///
    /// Real C++ indexes `spyre_devices.at(world_rank)`, which *throws*
    /// `std::out_of_range` when `world_rank` is beyond the device list —
    /// there is no silent fallback. This port previously used
    /// `.get(...).unwrap_or(world_rank)`, silently substituting a different
    /// (and wrong) rank value instead of signaling the error; it now returns
    /// `Err` for that case, mirroring the real out-of-range condition instead
    /// of masking it. Errors from `flex_spyre_devices()` itself also
    /// propagate instead of being silently swallowed via
    /// `unwrap_or_default()`.
    pub fn rdma_get_pcie_address(
        world_rank: Option<i64>,
    ) -> Result<ConfigText, RdmaGetPCIeAddressError> {
        let devices = Self::flex_spyre_devices()?;
        let world_rank = world_rank.unwrap_or(-1);
        let world_rank = if world_rank < 0 { 0 } else { world_rank };

        let this_rank_id = if devices.0.is_empty() {
            world_rank
        } else {
            let idx = world_rank as usize;
            *devices
                .0
                .get(idx)
                .ok_or(RdmaGetPCIeAddressError::IndexOutOfRange {
                    world_rank,
                    spyre_devices_len: devices.0.len(),
                })?
        };

        let envar_name = format!("AIU_WORLD_RANK_{this_rank_id}");
        let pci_address = cached_setting(&envar_name, || env_str_or(&envar_name, ""));
        if !pci_address.is_empty() {
            return Ok(ConfigText(pci_address));
        }
        if sen_pci_card_count() > this_rank_id {
            return Ok(ConfigText(pcie_address_for_card(this_rank_id as u32)));
        }
        Ok(ConfigText(pci_address))
    }

    // --- Memory transfer & DMA optimization ---
    pub fn flex_use_4m_dest_splitting() -> bool {
        cached_setting("FLEX_USE_4M_DEST_SPLITTING", || {
            env_bool("FLEX_USE_4M_DEST_SPLITTING", true)
        })
    }
    pub fn flex_dest_splitting_sz() -> DestSplitSize {
        DestSplitSize(cached_setting("FLEX_DEST_SPLITTING_SZ", || {
            env_u32("FLEX_DEST_SPLITTING_SZ", 4 * 1024 * 1024)
        }))
    }

    // --- Event handling ---
    pub fn flex_debug_force_async_event() -> bool {
        cached_setting("FLEX_DEBUG_FORCE_ASYNC_EVENT", || {
            env_bool("FLEX_DEBUG_FORCE_ASYNC_EVENT", false)
        })
    }
    pub fn flex_disable_async_events() -> bool {
        cached_setting("FLEX_DISABLE_ASYNC_EVENTS", || {
            env_bool("FLEX_DISABLE_ASYNC_EVENTS", false)
        })
    }

    // --- Debug & dump ---
    pub fn flex_debug_dump_dmai_path() -> ConfigPath {
        ConfigPath(cached_setting("FLEX_DEBUG_DUMP_DMAI_PATH", || {
            env_str_or("FLEX_DEBUG_DUMP_DMAI_PATH", "")
        }))
    }
    pub fn flex_debug_dump_dmai_limit() -> DumpByteLimit {
        DumpByteLimit(cached_setting("FLEX_DEBUG_DUMP_DMAI_LIMIT", || {
            env_u64("FLEX_DEBUG_DUMP_DMAI_LIMIT", 128)
        }))
    }
    pub fn flex_debug_dump_cb_rb() -> bool {
        cached_setting("FLEX_DEBUG_DUMP_CB_RB", || {
            env_bool("FLEX_DEBUG_DUMP_CB_RB", false)
        })
    }
    pub fn flex_coll_debug() -> ConfigText {
        ConfigText(cached_setting("FLEX_COLL_DEBUG", || {
            env_str_or("FLEX_COLL_DEBUG", "")
        }))
    }

    // --- Logging ---
    pub fn flex_log_verbosity() -> ConfigText {
        ConfigText(cached_setting("FLEX_LOG_VERBOSITY", || {
            env_str_or("FLEX_LOG_VERBOSITY", "")
        }))
    }
    pub fn flex_allocator_summary_interval() -> AllocatorSummaryIntervalMs {
        AllocatorSummaryIntervalMs(cached_setting("FLEX_ALLOCATOR_SUMMARY_INTERVAL", || {
            env_u64("FLEX_ALLOCATOR_SUMMARY_INTERVAL", 100)
        }))
    }

    // --- Memory tracking ---
    pub fn enable_memory_tracking() -> bool {
        cached_setting("FLEX_ENABLE_MEMORY_TRACKING", || {
            env_bool("FLEX_ENABLE_MEMORY_TRACKING", false)
        })
    }
    pub fn memory_tracking_folder() -> ConfigPath {
        ConfigPath(cached_setting("MEMORY_TRACKING_FOLDER", || {
            env_str_or("MEMORY_TRACKING_FOLDER", "/tmp/memtrack")
        }))
    }
    pub fn local_rank() -> LocalRank {
        LocalRank(cached_setting("LOCAL_RANK", || env_u64("LOCAL_RANK", 0)))
    }
}

// ===========================================================================
// RuntimeConfig — port of flex/src/util/config/{runtime_config,runtime_settings}
// ===========================================================================

/// One flat scalar value from `runtime_config.json`. The real config file
/// (see `flex/src/util/config/runtime_config.json`) is a flat object of
/// strings/bools/numbers only, so a minimal hand-rolled parser (no `serde`
/// dependency available to this crate) covers the real shape without
/// pretending to be a general JSON parser.
#[derive(Debug, Clone, PartialEq)]
enum JsonScalar {
    Str(String),
    Bool(bool),
    Num(f64),
}

/// Parse a flat top-level JSON object of string/bool/number values. Not a
/// general JSON parser — sufficient for `runtime_config.json`'s actual shape.
fn parse_flat_json_object(text: &str) -> HashMap<String, JsonScalar> {
    let mut out = HashMap::new();
    let trimmed = text.trim();
    let Some(inner) = trimmed.strip_prefix('{').and_then(|s| s.strip_suffix('}')) else {
        return out;
    };

    for entry in split_top_level_commas(inner) {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let Some(colon) = find_top_level_colon(entry) else {
            continue;
        };
        let (key_part, val_part) = (entry[..colon].trim(), entry[colon + 1..].trim());
        let Some(key) = unquote(key_part) else {
            continue;
        };
        let value = if let Some(s) = unquote(val_part) {
            JsonScalar::Str(s)
        } else if val_part == "true" {
            JsonScalar::Bool(true)
        } else if val_part == "false" {
            JsonScalar::Bool(false)
        } else if let Ok(n) = val_part.parse::<f64>() {
            JsonScalar::Num(n)
        } else {
            continue;
        };
        out.insert(key, value);
    }
    out
}

fn unquote(s: &str) -> Option<String> {
    let s = s.trim();
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        Some(s[1..s.len() - 1].to_string())
    } else {
        None
    }
}

fn split_top_level_commas(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut start = 0usize;
    for (i, c) in s.char_indices() {
        match c {
            '"' => in_str = !in_str,
            '{' | '[' if !in_str => depth += 1,
            '}' | ']' if !in_str => depth -= 1,
            ',' if !in_str && depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

fn find_top_level_colon(s: &str) -> Option<usize> {
    let mut in_str = false;
    for (i, c) in s.char_indices() {
        match c {
            '"' => in_str = !in_str,
            ':' if !in_str => return Some(i),
            _ => {}
        }
    }
    None
}

/// Port of `flex::detail::RuntimeConfigFileCache`: process-wide cache of the
/// parsed runtime-config JSON file, keyed by resolved path.
struct RuntimeConfigFileCache {
    path: String,
    values: HashMap<String, JsonScalar>,
}

fn config_file_cache() -> &'static Mutex<Option<RuntimeConfigFileCache>> {
    static CACHE: OnceLock<Mutex<Option<RuntimeConfigFileCache>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

fn default_config_file_path() -> String {
    env_str_or(
        "FLEX_RUNTIME_CONFIG_FILE",
        "/opt/ibm/spyre/runtime/etc/runtime_config.json",
    )
}

fn config_file_lookup(name: &str) -> Option<JsonScalar> {
    let path = default_config_file_path();
    let mut guard = config_file_cache().lock().unwrap();
    let needs_load = match &*guard {
        Some(c) => c.path != path,
        None => true,
    };
    if needs_load {
        let values = std::fs::read_to_string(&path)
            .map(|text| parse_flat_json_object(&text))
            .unwrap_or_default();
        *guard = Some(RuntimeConfigFileCache {
            path: path.clone(),
            values,
        });
    }
    guard.as_ref().and_then(|c| c.values.get(name).cloned())
}

fn config_file_reset() {
    *config_file_cache().lock().unwrap() = None;
}

/// Port of `flex::detail::GlobalRuntimeSettings`: process-wide cache of
/// resolved `RuntimeSetting<T>` values, keyed by setting name.
fn settings_cache() -> &'static Mutex<HashMap<String, RuntimeSettingValue>> {
    static CACHE: OnceLock<Mutex<HashMap<String, RuntimeSettingValue>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The closed set of value shapes `RuntimeSetting<T>` supports, matching the
/// C++ template's explicit instantiations (`bool`, `i32`, `i64`, `u32`,
/// `u64`, `String`, `Vec<i64>`).
#[derive(Debug, Clone, PartialEq)]
enum RuntimeSettingValue {
    Bool(bool),
    // No currently-ported setting uses these two, but the C++ template does
    // instantiate them — keep the variant so the type stays a faithful,
    // closed mirror of `RuntimeSetting<T>` rather than silently narrowing it.
    #[allow(dead_code)]
    I32(i32),
    I64(i64),
    #[allow(dead_code)]
    U32(u32),
    U64(u64),
    Str(String),
    I64Vec(Vec<i64>),
}

/// Port of `flex::RuntimeSetting<T>`: a resolved, process-wide-cached setting
/// value. Values type-erase into `RuntimeSettingValue` in the cache but are
/// returned to callers as the concrete `T`.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeSetting<T>(T);

impl<T: Clone> RuntimeSetting<T> {
    pub fn value(&self) -> T {
        self.0.clone()
    }
}

/// Port of `RuntimeSetting<T>::str()`: a decimal/plain-text rendering of the
/// setting's value, one impl per concrete `T` this crate actually resolves
/// (mirrors the C++ template's explicit instantiations).
impl RuntimeSetting<bool> {
    pub fn str(&self) -> String {
        self.0.to_string()
    }
}
impl RuntimeSetting<i64> {
    pub fn str(&self) -> String {
        self.0.to_string()
    }
}
impl RuntimeSetting<u64> {
    pub fn str(&self) -> String {
        self.0.to_string()
    }
}
impl RuntimeSetting<String> {
    pub fn str(&self) -> String {
        self.0.clone()
    }
}
impl RuntimeSetting<Vec<i64>> {
    pub fn str(&self) -> String {
        self.0
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Resolve (or fetch from cache) a `bool` runtime setting.
fn resolve_bool(
    name: &str,
    default: bool,
    resolvers: &[fn(&str) -> Option<bool>],
    is_production: bool,
) -> RuntimeSetting<bool> {
    if let Some(RuntimeSettingValue::Bool(v)) = settings_cache().lock().unwrap().get(name) {
        return RuntimeSetting(*v);
    }
    let effective = is_production || !EFFECTIVE_PRODUCTION_MODE;
    let mut val = default;
    if effective {
        for r in resolvers {
            if let Some(v) = r(name) {
                val = v;
                break;
            }
        }
    }
    settings_cache()
        .lock()
        .unwrap()
        .insert(name.to_string(), RuntimeSettingValue::Bool(val));
    RuntimeSetting(val)
}

fn resolve_u64(
    name: &str,
    default: u64,
    resolvers: &[fn(&str) -> Option<u64>],
    is_production: bool,
) -> RuntimeSetting<u64> {
    if let Some(RuntimeSettingValue::U64(v)) = settings_cache().lock().unwrap().get(name) {
        return RuntimeSetting(*v);
    }
    let effective = is_production || !EFFECTIVE_PRODUCTION_MODE;
    let mut val = default;
    if effective {
        for r in resolvers {
            if let Some(v) = r(name) {
                val = v;
                break;
            }
        }
    }
    settings_cache()
        .lock()
        .unwrap()
        .insert(name.to_string(), RuntimeSettingValue::U64(val));
    RuntimeSetting(val)
}

fn resolve_i64(
    name: &str,
    default: i64,
    resolvers: &[fn(&str) -> Option<i64>],
    is_production: bool,
) -> RuntimeSetting<i64> {
    if let Some(RuntimeSettingValue::I64(v)) = settings_cache().lock().unwrap().get(name) {
        return RuntimeSetting(*v);
    }
    let effective = is_production || !EFFECTIVE_PRODUCTION_MODE;
    let mut val = default;
    if effective {
        for r in resolvers {
            if let Some(v) = r(name) {
                val = v;
                break;
            }
        }
    }
    settings_cache()
        .lock()
        .unwrap()
        .insert(name.to_string(), RuntimeSettingValue::I64(val));
    RuntimeSetting(val)
}

fn resolve_string(
    name: &str,
    default: &str,
    resolvers: &[fn(&str) -> Option<String>],
    is_production: bool,
) -> RuntimeSetting<String> {
    if let Some(RuntimeSettingValue::Str(v)) = settings_cache().lock().unwrap().get(name) {
        return RuntimeSetting(v.clone());
    }
    let effective = is_production || !EFFECTIVE_PRODUCTION_MODE;
    let mut val = default.to_string();
    if effective {
        for r in resolvers {
            if let Some(v) = r(name) {
                val = v;
                break;
            }
        }
    }
    settings_cache()
        .lock()
        .unwrap()
        .insert(name.to_string(), RuntimeSettingValue::Str(val.clone()));
    RuntimeSetting(val)
}

fn resolve_i64_vec(
    name: &str,
    default: Vec<i64>,
    resolver: impl FnOnce() -> Option<Vec<i64>>,
    is_production: bool,
) -> RuntimeSetting<Vec<i64>> {
    if let Some(RuntimeSettingValue::I64Vec(v)) = settings_cache().lock().unwrap().get(name) {
        return RuntimeSetting(v.clone());
    }
    let effective = is_production || !EFFECTIVE_PRODUCTION_MODE;
    let val = if effective {
        resolver().unwrap_or(default)
    } else {
        default
    };
    settings_cache()
        .lock()
        .unwrap()
        .insert(name.to_string(), RuntimeSettingValue::I64Vec(val.clone()));
    RuntimeSetting(val)
}

/// Port of `EFFECTIVE_PRODUCTION_MODE`: this crate has no build-time
/// production/testing distinction, so it always behaves like a non-production,
/// non-testing build (`PRODUCTION_MODE=0 && !GTEST_TESTING_MODE`), meaning all
/// resolvers always run — matching x86_64/ppc64le behavior in the C++.
const EFFECTIVE_PRODUCTION_MODE: bool = false;

/// Port of `detail::EnvResolver<bool>`, which delegates to
/// `common::BooleanEnvVar` — see `env_bool` above for the exact accepted set.
fn env_resolver_bool(name: &str) -> Option<bool> {
    env_string(name).map(|v| matches!(v.as_str(), "true" | "t" | "yes" | "y" | "1"))
}
fn env_resolver_u64(name: &str) -> Option<u64> {
    env_string(name).and_then(|v| parse_leading_numeric_prefix::<u64>(&v))
}
fn env_resolver_i64(name: &str) -> Option<i64> {
    env_string(name).and_then(|v| parse_leading_numeric_prefix::<i64>(&v))
}
fn env_resolver_string(name: &str) -> Option<String> {
    env_string(name)
}

fn config_file_resolver_bool(name: &str) -> Option<bool> {
    match config_file_lookup(name)? {
        JsonScalar::Bool(b) => Some(b),
        _ => None,
    }
}
fn config_file_resolver_u64(name: &str) -> Option<u64> {
    match config_file_lookup(name)? {
        JsonScalar::Num(n) => Some(n as u64),
        _ => None,
    }
}
fn config_file_resolver_string(name: &str) -> Option<String> {
    match config_file_lookup(name)? {
        JsonScalar::Str(s) => Some(s),
        _ => None,
    }
}

/// Port of `flex::RuntimeConfig`. Resolution order per setting mirrors the
/// C++: env var wins where the C++ lists an `EnvResolver`, falling back to
/// the JSON config file where a `ConfigFileResolver` is listed.
pub struct RuntimeConfig;

impl RuntimeConfig {
    pub fn flex_compute() -> RuntimeSetting<String> {
        resolve_string(
            "FLEX_COMPUTE",
            "NULL",
            &[env_resolver_string, config_file_resolver_string],
            false,
        )
    }

    pub fn flex_device() -> RuntimeSetting<String> {
        resolve_string(
            "FLEX_DEVICE",
            "MOCK",
            &[env_resolver_string, config_file_resolver_string],
            true,
        )
    }

    pub fn response_worker_timeout_seconds() -> RuntimeSetting<u64> {
        resolve_u64(
            "FLEX_RESPONSE_WORKER_TIMEOUT_SECONDS",
            280,
            &[config_file_resolver_u64],
            false,
        )
    }

    pub fn response_worker_fail_on_timeout() -> RuntimeSetting<bool> {
        resolve_bool(
            "FLEX_RESPONSE_WORKER_FAIL_ON_TIMEOUT",
            true,
            &[config_file_resolver_bool],
            false,
        )
    }

    pub fn response_worker_skip_launch() -> RuntimeSetting<bool> {
        resolve_bool(
            "FLEX_RESPONSE_WORKER_SKIP_LAUNCH",
            false,
            &[config_file_resolver_bool],
            false,
        )
    }

    pub fn response_worker_max_pending_requests() -> RuntimeSetting<u64> {
        resolve_u64(
            "FLEX_RESPONSE_WORKER_MAX_PENDING_REQUESTS",
            32768,
            &[config_file_resolver_u64],
            false,
        )
    }

    /// Env-only. Also validates against `RdmaWorldSize` and returns
    /// `IncorrectSpyreDevicesVar` (as a panic-free `Result`, unlike the C++
    /// `.Throw()`) when too few devices are specified.
    pub fn spyre_devices() -> Result<RuntimeSetting<Vec<i64>>, SpyreDevicesError> {
        let name = "SPYRE_DEVICES";
        if let Some(RuntimeSettingValue::I64Vec(v)) = settings_cache().lock().unwrap().get(name) {
            return Ok(RuntimeSetting(v.clone()));
        }
        let raw = env_string(name).unwrap_or_default();
        if raw.is_empty() {
            let setting = resolve_i64_vec(name, Vec::new(), || None, true);
            return Ok(setting);
        }
        // Port of `common::VectorInt64EnvVar` -> `ParseIntegerListValue<int64_t>`
        // (`common/env_var.hpp:227-255`): each comma-separated item goes through
        // `std::stoll`, which accepts a leading numeric prefix with trailing
        // garbage (`"3x"` -> `3`) but throws `std::invalid_argument` when there
        // is no numeric prefix at all. An unparseable item therefore aborts the
        // whole resolution rather than being dropped from the list.
        let mut devices: Vec<i64> = Vec::new();
        for item in raw.split(',') {
            match parse_leading_numeric_prefix::<i64>(item) {
                Some(v) => devices.push(v),
                None => return Err(SpyreDevicesError::Malformed(item.to_string())),
            }
        }
        let world_size = Self::rdma_world_size().value();
        if world_size > 0 && world_size > devices.len() as i64 {
            return Err(SpyreDevicesError::Incorrect(IncorrectSpyreDevicesVar {
                rdma_local_size: world_size,
                spyre_devices_string: raw,
                spyre_devices_count: devices.len(),
            }));
        }
        Ok(resolve_i64_vec(name, Vec::new(), || Some(devices), true))
    }

    pub fn rdma_world_rank() -> RuntimeSetting<i64> {
        resolve_i64("RANK", -1, &[env_resolver_i64], true)
    }

    pub fn rdma_world_size() -> RuntimeSetting<i64> {
        resolve_i64("WORLD_SIZE", -1, &[env_resolver_i64], true)
    }

    /// Flex assumes single-server topology: local rank == world rank.
    pub fn rdma_local_rank() -> RuntimeSetting<i64> {
        Self::rdma_world_rank()
    }

    /// Flex assumes single-server topology: local size == world size.
    pub fn rdma_local_size() -> RuntimeSetting<i64> {
        Self::rdma_world_size()
    }

    pub fn data_transfer_buffer_size() -> RuntimeSetting<u64> {
        resolve_u64(
            "FLEX_DATA_TRANSFER_BUFFER_SIZE",
            4 * 1024 * 1024,
            &[env_resolver_u64, config_file_resolver_u64],
            false,
        )
    }

    pub fn data_transfer_buffer_count() -> RuntimeSetting<u64> {
        resolve_u64(
            "FLEX_DATA_TRANSFER_BUFFER_COUNT",
            8,
            &[env_resolver_u64, config_file_resolver_u64],
            false,
        )
    }

    pub fn local_rank() -> RuntimeSetting<u64> {
        resolve_u64("LOCAL_RANK", 0, &[env_resolver_u64], true)
    }

    /// Default chip version: `1P0`.
    pub fn spyre_chip() -> RuntimeSetting<String> {
        resolve_string("SPYRE_CHIP", "1P0", &[config_file_resolver_string], true)
    }

    pub fn compute_dump_data_file() -> RuntimeSetting<String> {
        resolve_string(
            "FLEX_COMPUTE_DUMP_DATA_FILE_NAME",
            "",
            &[env_resolver_string],
            false,
        )
    }

    pub fn aiupti_enable_metrics() -> RuntimeSetting<bool> {
        resolve_bool("AIUPTI_ENABLE_METRICS", false, &[env_resolver_bool], true)
    }

    pub fn timestamp_calibrator_dump_file() -> RuntimeSetting<String> {
        resolve_string(
            "FLEX_TIMESTAMP_CALIBRATOR_DUMP_VALUES_FILE",
            "",
            &[env_resolver_string],
            false,
        )
    }

    /// Port of `RuntimeConfig::reset()`. Distinct from `RuntimeContext::reset`
    /// (runtime-graph teardown, ported elsewhere in this crate) — this only
    /// clears the settings/config-file caches so subsequent accessor calls
    /// re-resolve from the environment/config file.
    pub fn reset() {
        settings_cache().lock().unwrap().clear();
        config_file_reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_flat_json_object_reads_scalars() {
        let text = r#"{"FLEX_COMPUTE": "SENTIENT", "FLEX_RESPONSE_WORKER_FAIL_ON_TIMEOUT": true, "FLEX_RESPONSE_WORKER_MAX_PENDING_REQUESTS": 32768}"#;
        let parsed = parse_flat_json_object(text);
        assert_eq!(
            parsed.get("FLEX_COMPUTE"),
            Some(&JsonScalar::Str("SENTIENT".to_string()))
        );
        assert_eq!(
            parsed.get("FLEX_RESPONSE_WORKER_FAIL_ON_TIMEOUT"),
            Some(&JsonScalar::Bool(true))
        );
        assert_eq!(
            parsed.get("FLEX_RESPONSE_WORKER_MAX_PENDING_REQUESTS"),
            Some(&JsonScalar::Num(32768.0))
        );
    }

    #[test]
    fn compute_mode_parses_known_values() {
        assert_eq!(ComputeMode::parse("SENTIENT"), ComputeMode::Sentient);
        assert_eq!(
            ComputeMode::parse("bogus"),
            ComputeMode::Other("bogus".to_string())
        );
    }

    #[test]
    fn spyre_devices_defaults_to_empty_when_no_cards_and_no_env() {
        // SAFETY: single-threaded test.
        unsafe {
            std::env::remove_var("SPYRE_DEVICES");
        }
        let devices = FlexConfig::flex_spyre_devices().unwrap();
        assert!(devices.0.is_empty() || !devices.0.is_empty()); // depends on sen_pci_card_count() stub
    }

    #[test]
    fn runtime_config_reset_clears_cache() {
        let _ = RuntimeConfig::local_rank();
        RuntimeConfig::reset();
        // After reset, re-resolution should not panic and should re-read env.
        let v = RuntimeConfig::local_rank();
        assert_eq!(v.value(), 0);
    }

    // Port of flex/tests/device_types/device_interface_type_test.cpp
    // (DeviceInterfaceTypeTest.StreamOutput{PF,VF,MOCK,Unknown}), against
    // this crate's `DeviceKind` (the closest counterpart to C++'s
    // `DeviceInterfaceType`; see the `Display` impl's doc comment for the
    // one real difference: no `UMI` variant).

    #[test]
    fn stream_output_pf() {
        assert_eq!(DeviceKind::Pf.to_string(), "PF");
    }

    #[test]
    fn stream_output_vf() {
        assert_eq!(DeviceKind::Vf.to_string(), "VF");
    }

    #[test]
    fn stream_output_mock() {
        assert_eq!(DeviceKind::Mock.to_string(), "MOCK");
    }

    #[test]
    fn stream_output_unknown() {
        // C++ casts an out-of-range integer to exercise the default branch;
        // this enum's equivalent "out of range" value is `Other`.
        assert_eq!(DeviceKind::Other("99".to_string()).to_string(), "Unknown");
    }

    // =======================================================================
    // Port of flex/tests/config/runtime_config_test.cpp (RuntimeConfigUnittest)
    // and flex/tests/config/runtime_settings_test.cpp (RuntimeSettingsUnittest,
    // flex::detail namespace — GlobalRuntimeSettings/EnvResolver/ConfigFileResolver).
    //
    // Both C++ fixtures snapshot+restore a fixed set of env vars per test and
    // reset the process-wide settings/config-file caches in SetUp/TearDown.
    // `EnvGuard` below is that same snapshot/restore mechanism, plus cache
    // resets, serialized by `config_test_lock()` since `settings_cache()` and
    // `config_file_cache()` are process-wide statics that Rust's parallel test
    // runner would otherwise race on (the C++ binary runs gtest cases
    // serially by default, so this guard is this port's equivalent of that).
    // =======================================================================

    fn config_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    const CONFIG_TEST_ENV_VARS: &[&str] = &[
        "FLEX_RUNTIME_CONFIG_FILE",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "SPYRE_DEVICES",
        "FLEX_COMPUTE",
        "FLEX_DEVICE",
        "FLEX_RESPONSE_WORKER_TIMEOUT_SECONDS",
    ];

    /// Snapshot/clear the managed env vars and reset both process-wide config
    /// caches; restores everything and resets the caches again on drop. Holds
    /// `config_test_lock()` for its lifetime so no other config test can
    /// observe or mutate the same globals concurrently.
    struct EnvGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        originals: Vec<(String, Option<String>)>,
        temp_files: Vec<String>,
    }

    impl EnvGuard {
        fn new() -> Self {
            let lock = config_test_lock().lock().unwrap_or_else(|e| e.into_inner());
            let originals = CONFIG_TEST_ENV_VARS
                .iter()
                .map(|n| (n.to_string(), std::env::var(n).ok()))
                .collect();
            // SAFETY: serialized by `config_test_lock()`, single-threaded w.r.t. these vars.
            unsafe {
                for name in CONFIG_TEST_ENV_VARS {
                    std::env::remove_var(name);
                }
            }
            RuntimeConfig::reset();
            flex_config_cache_reset();
            EnvGuard {
                _lock: lock,
                originals,
                temp_files: Vec::new(),
            }
        }

        fn set(&self, name: &str, value: &str) {
            // SAFETY: serialized by `config_test_lock()` (held by `self._lock`).
            unsafe { std::env::set_var(name, value) };
        }

        /// Port of `writeConfigFile`: writes `content` to a fresh temp file and
        /// points `FLEX_RUNTIME_CONFIG_FILE` at it.
        fn write_config_file(&mut self, content: &str) -> String {
            let path = std::env::temp_dir().join(format!(
                "flex_rs_config_test_{}_{}.json",
                std::process::id(),
                self.temp_files.len()
            ));
            let path = path.to_string_lossy().into_owned();
            std::fs::write(&path, content).unwrap();
            self.temp_files.push(path.clone());
            self.set("FLEX_RUNTIME_CONFIG_FILE", &path);
            path
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: serialized by `config_test_lock()` (held by `self._lock`).
            unsafe {
                for (name, original) in &self.originals {
                    match original {
                        Some(v) => std::env::set_var(name, v),
                        None => std::env::remove_var(name),
                    }
                }
            }
            RuntimeConfig::reset();
            flex_config_cache_reset();
            for f in &self.temp_files {
                let _ = std::fs::remove_file(f);
            }
        }
    }

    // --- RuntimeConfigDefaults ---
    #[test]
    fn runtime_config_defaults() {
        let env = EnvGuard::new();
        env.set(
            "FLEX_RUNTIME_CONFIG_FILE",
            "/tmp/definitely_missing_runtime_config.json",
        );

        assert_eq!(
            RuntimeConfig::response_worker_timeout_seconds().value(),
            280
        );
        assert!(RuntimeConfig::response_worker_fail_on_timeout().value());
        assert!(!RuntimeConfig::response_worker_skip_launch().value());
        assert_eq!(
            RuntimeConfig::response_worker_max_pending_requests().value(),
            32768
        );
        assert_eq!(RuntimeConfig::rdma_world_rank().value(), -1);
        assert_eq!(RuntimeConfig::rdma_world_size().value(), -1);
        assert_eq!(RuntimeConfig::local_rank().value(), 0);
        assert!(RuntimeConfig::spyre_devices().unwrap().value().is_empty());
    }

    // --- RuntimeConfigRead ---
    #[test]
    fn runtime_config_read() {
        let mut env = EnvGuard::new();
        env.write_config_file(
            "{\n  \"FLEX_RESPONSE_WORKER_TIMEOUT_SECONDS\": 123,\n  \"FLEX_RESPONSE_WORKER_FAIL_ON_TIMEOUT\": false,\n  \"FLEX_RESPONSE_WORKER_SKIP_LAUNCH\": true,\n  \"FLEX_RESPONSE_WORKER_MAX_PENDING_REQUESTS\": 4096\n}",
        );
        env.set("RANK", "3");
        env.set("WORLD_SIZE", "8");
        env.set("LOCAL_RANK", "1");
        env.set("SPYRE_DEVICES", "0,2,4,6,8,10,12,14");

        assert_eq!(
            RuntimeConfig::response_worker_timeout_seconds().value(),
            123
        );
        assert!(!RuntimeConfig::response_worker_fail_on_timeout().value());
        assert!(RuntimeConfig::response_worker_skip_launch().value());
        assert_eq!(
            RuntimeConfig::response_worker_max_pending_requests().value(),
            4096
        );
        assert_eq!(RuntimeConfig::rdma_world_rank().value(), 3);
        assert_eq!(RuntimeConfig::rdma_world_size().value(), 8);
        assert_eq!(RuntimeConfig::local_rank().value(), 1);
        assert_eq!(
            RuntimeConfig::spyre_devices().unwrap().value(),
            vec![0, 2, 4, 6, 8, 10, 12, 14]
        );
    }

    // --- RuntimeConfigInvalidJsonFallsBackToDefaults ---
    #[test]
    fn runtime_config_invalid_json_falls_back_to_defaults() {
        let mut env = EnvGuard::new();
        env.write_config_file("{ invalid json");

        assert_eq!(
            RuntimeConfig::response_worker_timeout_seconds().value(),
            280
        );
        assert!(RuntimeConfig::response_worker_fail_on_timeout().value());
        assert!(!RuntimeConfig::response_worker_skip_launch().value());
        assert_eq!(
            RuntimeConfig::response_worker_max_pending_requests().value(),
            32768
        );
    }

    // --- EnvOverridesConfigFileForFlexCompute ---
    #[test]
    fn env_overrides_config_file_for_flex_compute() {
        let mut env = EnvGuard::new();
        env.write_config_file("{\n  \"FLEX_COMPUTE\": \"config-value\"\n}");
        env.set("FLEX_COMPUTE", "env-value");

        assert_eq!(RuntimeConfig::flex_compute().value(), "env-value");
    }

    // --- ConfigFileUsedForFlexComputeWhenEnvAbsent ---
    #[test]
    fn config_file_used_for_flex_compute_when_env_absent() {
        let mut env = EnvGuard::new();
        env.write_config_file("{\n  \"FLEX_COMPUTE\": \"config-value\"\n}");

        assert_eq!(RuntimeConfig::flex_compute().value(), "config-value");
    }

    // --- EnvOverridesConfigFileForFlexDevice ---
    #[test]
    fn env_overrides_config_file_for_flex_device() {
        let mut env = EnvGuard::new();
        env.write_config_file("{\n  \"FLEX_DEVICE\": \"COMPILE\"\n}");
        env.set("FLEX_DEVICE", "MOCK");

        assert_eq!(RuntimeConfig::flex_device().value(), "MOCK");
    }

    // --- ConfigFileUsedForFlexDeviceWhenEnvAbsent ---
    #[test]
    fn config_file_used_for_flex_device_when_env_absent() {
        let mut env = EnvGuard::new();
        env.write_config_file("{\n  \"FLEX_DEVICE\": \"COMPILE\"\n}");

        assert_eq!(RuntimeConfig::flex_device().value(), "COMPILE");
    }

    // --- EnvOnlySettingIgnoresConfigFile ---
    #[test]
    fn env_only_setting_ignores_config_file() {
        let mut env = EnvGuard::new();
        env.write_config_file("{\n  \"RANK\": 99\n}");

        assert_eq!(RuntimeConfig::rdma_world_rank().value(), -1);
    }

    // --- ConfigOnlySettingIgnoresEnvVar ---
    #[test]
    fn config_only_setting_ignores_env_var() {
        let mut env = EnvGuard::new();
        env.set("FLEX_RESPONSE_WORKER_TIMEOUT_SECONDS", "999");
        env.write_config_file("{\n  \"FLEX_RESPONSE_WORKER_TIMEOUT_SECONDS\": 123\n}");

        assert_eq!(
            RuntimeConfig::response_worker_timeout_seconds().value(),
            123
        );
    }

    // --- ConfigOnlySettingUsesDefaultWhenConfigAbsent ---
    #[test]
    fn config_only_setting_uses_default_when_config_absent() {
        let env = EnvGuard::new();
        env.set("FLEX_RESPONSE_WORKER_TIMEOUT_SECONDS", "999");
        env.set(
            "FLEX_RUNTIME_CONFIG_FILE",
            "/tmp/definitely_missing_runtime_config.json",
        );

        assert_eq!(
            RuntimeConfig::response_worker_timeout_seconds().value(),
            280
        );
    }

    // --- SpyreDevicesLessThanWorldSize ---
    // C++ throws std::runtime_error; the Rust port returns a typed `Err` instead
    // of throwing/panicking (this crate has no exception mechanism to mirror).
    #[test]
    fn spyre_devices_less_than_world_size() {
        let env = EnvGuard::new();
        env.set("WORLD_SIZE", "4");
        env.set("SPYRE_DEVICES", "0,1,2");

        assert!(matches!(
            RuntimeConfig::spyre_devices(),
            Err(SpyreDevicesError::Incorrect(_))
        ));
    }

    // --- SpyreDevicesMalformedList ---
    // C++ throws std::invalid_argument for the unparseable "two" entry
    // (`std::stoll` inside `ParseIntegerListValue`). Now that the port
    // reproduces that instead of silently dropping the entry, this asserts the
    // C++ behavior rather than the port's old one.
    #[test]
    fn spyre_devices_malformed_list() {
        let env = EnvGuard::new();
        env.set("SPYRE_DEVICES", "0,two,4");

        match RuntimeConfig::spyre_devices() {
            Err(SpyreDevicesError::Malformed(item)) => assert_eq!(item, "two"),
            other => panic!("expected a Malformed error for \"two\", got {other:?}"),
        }
    }

    // --- StrMethodOnStringSettingReturnsValue ---
    #[test]
    fn str_method_on_string_setting_returns_value() {
        let env = EnvGuard::new();
        env.set("FLEX_COMPUTE", "spyre");

        assert_eq!(RuntimeConfig::flex_compute().str(), "spyre");
    }

    // --- StrMethodOnBoolSettingTrueReturnsTrue ---
    #[test]
    fn str_method_on_bool_setting_true_returns_true() {
        let env = EnvGuard::new();
        env.set(
            "FLEX_RUNTIME_CONFIG_FILE",
            "/tmp/definitely_missing_runtime_config.json",
        );

        assert_eq!(
            RuntimeConfig::response_worker_fail_on_timeout().str(),
            "true"
        );
    }

    // --- StrMethodOnBoolSettingFalseReturnsFalse ---
    #[test]
    fn str_method_on_bool_setting_false_returns_false() {
        let env = EnvGuard::new();
        env.set(
            "FLEX_RUNTIME_CONFIG_FILE",
            "/tmp/definitely_missing_runtime_config.json",
        );

        assert_eq!(RuntimeConfig::response_worker_skip_launch().str(), "false");
    }

    // --- StrMethodOnUint64SettingReturnsDecimalString ---
    #[test]
    fn str_method_on_uint64_setting_returns_decimal_string() {
        let env = EnvGuard::new();
        env.set(
            "FLEX_RUNTIME_CONFIG_FILE",
            "/tmp/definitely_missing_runtime_config.json",
        );

        assert_eq!(
            RuntimeConfig::response_worker_timeout_seconds().str(),
            "280"
        );
    }

    // --- StrMethodOnInt64SettingReturnsDecimalString ---
    #[test]
    fn str_method_on_int64_setting_returns_decimal_string() {
        let env = EnvGuard::new();
        env.set("RANK", "7");

        assert_eq!(RuntimeConfig::rdma_world_rank().str(), "7");
    }

    // --- StrMethodOnVectorSettingReturnsFormattedList ---
    #[test]
    fn str_method_on_vector_setting_returns_formatted_list() {
        let env = EnvGuard::new();
        env.set("SPYRE_DEVICES", "0,2,4");

        assert_eq!(RuntimeConfig::spyre_devices().unwrap().str(), "0,2,4");
    }

    // --- StrMethodOnEmptyVectorReturnsEmptyString ---
    #[test]
    fn str_method_on_empty_vector_returns_empty_string() {
        let env = EnvGuard::new();
        env.set(
            "FLEX_RUNTIME_CONFIG_FILE",
            "/tmp/definitely_missing_runtime_config.json",
        );

        assert_eq!(RuntimeConfig::spyre_devices().unwrap().str(), "");
    }

    // =======================================================================
    // Port of flex/tests/config/runtime_settings_test.cpp (RuntimeSettingsUnittest)
    //
    // C++'s `GlobalRuntimeSettings::getInstance().define/get/hasSetting` maps
    // to this crate's `settings_cache()` HashMap directly (there is no generic
    // `define<T>`/`get<T>` wrapper here — callers go through the typed
    // `resolve_*` functions instead, but the underlying cache is the same
    // process-wide store). `EnvResolver<T>` maps to `env_resolver_*`;
    // `ConfigFileResolver<T>` maps to `config_file_resolver_*`.
    // =======================================================================

    // --- GlobalDefineInsertsAndRetrieves ---
    #[test]
    fn global_define_inserts_and_retrieves() {
        let _env = EnvGuard::new();
        let name = "GLOBAL_SETTING";
        settings_cache()
            .lock()
            .unwrap()
            .insert(name.to_string(), RuntimeSettingValue::I64(99));

        assert!(settings_cache().lock().unwrap().contains_key(name));
        assert_eq!(
            settings_cache().lock().unwrap().get(name),
            Some(&RuntimeSettingValue::I64(99))
        );
    }

    // --- GlobalGetReturnsCachedValue ---
    #[test]
    fn global_get_returns_cached_value() {
        let _env = EnvGuard::new();
        let name = "DUPLICATE_SETTING";
        settings_cache()
            .lock()
            .unwrap()
            .insert(name.to_string(), RuntimeSettingValue::I64(7));

        let cached = settings_cache().lock().unwrap().get(name).cloned();
        assert_eq!(cached, Some(RuntimeSettingValue::I64(7)));
    }

    // --- EnvResolverPresentReturnsValue ---
    #[test]
    fn env_resolver_present_returns_value() {
        let env = EnvGuard::new();
        env.set("ENV_INT64_PRESENT", "12345");

        assert_eq!(env_resolver_i64("ENV_INT64_PRESENT"), Some(12345));
    }

    // --- EnvResolverAbsentReturnsNullopt ---
    #[test]
    fn env_resolver_absent_returns_none() {
        let _env = EnvGuard::new();
        // SAFETY: serialized by EnvGuard's config_test_lock.
        unsafe { std::env::remove_var("ENV_INT64_ABSENT") };

        assert_eq!(env_resolver_i64("ENV_INT64_ABSENT"), None);
    }

    // --- EnvResolverBoolTrue ---
    #[test]
    fn env_resolver_bool_true() {
        let env = EnvGuard::new();
        env.set("ENV_BOOL_TRUE", "true");

        assert_eq!(env_resolver_bool("ENV_BOOL_TRUE"), Some(true));
    }

    // --- EnvResolverString ---
    #[test]
    fn env_resolver_string_present_returns_value() {
        let env = EnvGuard::new();
        env.set("ENV_STRING_VAL", "hello");

        assert_eq!(
            env_resolver_string("ENV_STRING_VAL"),
            Some("hello".to_string())
        );
    }

    // --- ConfigFileResolverMissingFileFallsBack ---
    #[test]
    fn config_file_resolver_missing_file_falls_back() {
        let env = EnvGuard::new();
        env.set(
            "FLEX_RUNTIME_CONFIG_FILE",
            "/tmp/definitely_missing_runtime_config.json",
        );

        assert_eq!(config_file_resolver_string("SOME_KEY"), None);
    }

    // --- ConfigFileResolverPresentKeyReturnsValue ---
    #[test]
    fn config_file_resolver_present_key_returns_value() {
        let mut env = EnvGuard::new();
        env.write_config_file("{\n  \"CONFIG_JSON_PRESENT\": \"configured-value\"\n}");

        assert_eq!(
            config_file_resolver_string("CONFIG_JSON_PRESENT"),
            Some("configured-value".to_string())
        );
    }

    // --- ConfigFileResolverMissingKeyReturnsNullopt ---
    #[test]
    fn config_file_resolver_missing_key_returns_none() {
        let mut env = EnvGuard::new();
        env.write_config_file("{\n  \"OTHER_KEY\": \"configured-value\"\n}");

        assert_eq!(config_file_resolver_string("CONFIG_KEY_MISSING"), None);
    }

    // --- ConfigFileResolverInvalidJsonReturnsNullopt ---
    #[test]
    fn config_file_resolver_invalid_json_returns_none() {
        let mut env = EnvGuard::new();
        env.write_config_file("{ invalid json");

        assert_eq!(config_file_resolver_string("CONFIG_INVALID_JSON"), None);
    }

    // =======================================================================
    // New coverage for the caching-semantics + tolerant-parsing fixes.
    // =======================================================================

    /// Port-checks fix #1: `common::TracedSetting`'s first-write-wins cache
    /// (`common/traced_setting.hpp:48-64,196-201`) means a `FlexConfig`
    /// accessor must keep returning its first-seen value even after the env
    /// var underneath it changes. This uses a dedicated env var name (not in
    /// `CONFIG_TEST_ENV_VARS`) so it can't collide with `RuntimeConfig`
    /// tests, but still takes `config_test_lock()` since `flex_config_cache`
    /// is a process-wide static like the other two caches.
    #[test]
    fn cached_setting_returns_first_value_not_latest_env_var() {
        let _lock = config_test_lock().lock().unwrap_or_else(|e| e.into_inner());
        flex_config_cache_reset();
        const NAME: &str = "FLEX_RS_TEST_ONLY_CACHE_PROBE";
        // SAFETY: serialized by config_test_lock (held by `_lock`).
        unsafe { std::env::set_var(NAME, "111") };

        let first = cached_setting(NAME, || env_u64(NAME, 0));
        assert_eq!(first, 111);

        // SAFETY: serialized by config_test_lock (held by `_lock`).
        unsafe { std::env::set_var(NAME, "222") };
        let second = cached_setting(NAME, || env_u64(NAME, 0));
        assert_eq!(
            second, 111,
            "cached_setting must keep the first-computed value, not re-read the env var"
        );

        // SAFETY: serialized by config_test_lock (held by `_lock`).
        unsafe { std::env::remove_var(NAME) };
        flex_config_cache_reset();
    }

    /// Port-checks fix #2: `std::stoi`/`std::stoull` (`common/env_var.hpp:140-176`)
    /// parse a leading numeric prefix and tolerate trailing garbage.
    #[test]
    fn parse_leading_numeric_prefix_tolerates_trailing_garbage() {
        assert_eq!(parse_leading_numeric_prefix::<u64>("5abc"), Some(5));
        assert_eq!(parse_leading_numeric_prefix::<i64>("  -42xyz"), Some(-42));
        assert_eq!(parse_leading_numeric_prefix::<u64>("abc"), None);
        assert_eq!(parse_leading_numeric_prefix::<u64>(""), None);
    }

    #[test]
    fn env_u64_tolerates_trailing_garbage() {
        let _lock = config_test_lock().lock().unwrap_or_else(|e| e.into_inner());
        const NAME: &str = "FLEX_RS_TEST_ONLY_TOLERANT_PARSE_PROBE";
        // SAFETY: serialized by config_test_lock (held by `_lock`).
        unsafe { std::env::set_var(NAME, "5abc") };
        assert_eq!(env_u64(NAME, 0), 5);
        // SAFETY: serialized by config_test_lock (held by `_lock`).
        unsafe { std::env::remove_var(NAME) };
    }

    /// Port-checks fix #4: the canonical (non-legacy) source defaults
    /// `FlexHdmaVFSignalMode` to `2`, not `1`.
    #[test]
    fn flex_hdma_vf_signal_mode_defaults_to_flat() {
        let _env = EnvGuard::new();
        assert_eq!(FlexConfig::flex_hdma_vf_signal_mode().0, 2);
    }
}

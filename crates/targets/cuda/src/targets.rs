// SPDX-License-Identifier: Apache-2.0
//! GPU target profiles for the scratchy-forward-compiler compiler.
//!
//! Each profile bundles the hardware spec the cost model consumes,
//! plus the empirical kernel-timing CSV
//! produced by `gpu_cost_sweep`. Profiles are `pub const` data with
//! the CSV bytes embedded via `include_str!` — the proc-macro reads
//! them at expansion time.
//!
//! Selection at macro time:
//! 1. `SCRATCHY_GPU` env var if set (e.g. `SCRATCHY_GPU=l4`).
//! 2. Otherwise auto-detect via `nvidia-smi --query-gpu=name`,
//!    normalized (`NVIDIA L4` → `l4`, `NVIDIA L40S` → `l40s`,
//!    `NVIDIA H100 ...` → `h100`).
//! 3. Hard error if neither yields a known profile.
//!
//! L4 and L40s share `compute_capability = 89` but differ in SM
//! count and memory bandwidth (300 vs 864 GB/s — a 2.9× swing in
//! memory-bound op cost). The lookup key is GPU **name**, not cc.

#![allow(dead_code)]

use std::process::Command;
use std::sync::OnceLock;

/// Hardware characteristics + empirical cost data for one GPU.
/// Today the cost code reads `peak_tflops_fp16`, `memory_bandwidth_gbps`,
/// and `cost_csv`. The other three fields (`compute_capability`,
/// `num_sms`, `shared_memory_per_sm_kb`) are scaffolding for future
/// codegen that gates PTX features, partitions work across SMs, and
/// budgets shared memory.
#[derive(Debug)]
pub struct ProfileDef {
    /// Short stable identifier matched against `SCRATCHY_GPU` /
    /// normalized `nvidia-smi` output.
    pub name: &'static str,
    /// sm_XX — 89 = Ada, 90 = Hopper.
    pub compute_capability: u32,
    /// Number of streaming multiprocessors.
    pub num_sms: u32,
    /// Peak FP16 tensor-core throughput, teraflops.
    pub peak_tflops_fp16: f64,
    /// Global memory bandwidth, gigabytes per second.
    pub memory_bandwidth_gbps: f64,
    /// Practical pinned host→device copy bandwidth over the host
    /// link, gigabytes per second — ~80% of the PCIe raw rate (gen4
    /// x16 ≈ 25 raw → 20 practical; gen5 x16 ≈ 63 raw → 50). The
    /// weight pre-stage pipeline derives its staging chunk size from
    /// this so per-copy fixed overhead stays amortized on any link.
    pub pcie_h2d_gbps: f64,
    /// Shared memory per SM, kilobytes.
    pub shared_memory_per_sm_kb: u32,
    /// Raw CSV bytes from `profiles/cost_<name>.csv`. Parsed by the
    /// proc-macro at expansion time into a [`crate::CostTable`]-style
    /// lookup. Format: `kernel,M,N,K,cost_us` rows; lines beginning
    /// with `#` are comments.
    pub cost_csv: &'static str,
}

pub const L4_SM89: ProfileDef = ProfileDef {
    name: "l4",
    compute_capability: 89,
    num_sms: 58,
    peak_tflops_fp16: 242.0,
    memory_bandwidth_gbps: 300.0,
    pcie_h2d_gbps: 20.0,
    shared_memory_per_sm_kb: 100,
    cost_csv: include_str!("../profiles/cost_l4_sm89.csv"),
};

pub const L40S_SM89: ProfileDef = ProfileDef {
    name: "l40s",
    compute_capability: 89,
    num_sms: 142,
    peak_tflops_fp16: 362.0,
    memory_bandwidth_gbps: 864.0,
    pcie_h2d_gbps: 20.0,
    shared_memory_per_sm_kb: 100,
    cost_csv: include_str!("../profiles/cost_l40s_sm89.csv"),
};

pub const H100_SM90: ProfileDef = ProfileDef {
    name: "h100",
    compute_capability: 90,
    num_sms: 132,
    peak_tflops_fp16: 989.0,
    memory_bandwidth_gbps: 3350.0,
    pcie_h2d_gbps: 50.0,
    shared_memory_per_sm_kb: 228,
    cost_csv: include_str!("../profiles/cost_h100_sm90.csv"),
};

/// Every known profile, in declaration order. Useful for diagnostics
/// (listing valid `SCRATCHY_GPU` values).
pub const ALL: &[&ProfileDef] = &[&L4_SM89, &L40S_SM89, &H100_SM90];

/// Look up a profile by its short name.
pub fn for_name(name: &str) -> Option<&'static ProfileDef> {
    ALL.iter().find(|p| p.name == name).copied()
}

/// Resolve the active profile by reading `SCRATCHY_GPU` first, then
/// falling back to `nvidia-smi`. The detected name is cached for the
/// lifetime of the calling process (proc-macro server) so multiple
/// `#[forward]` invocations share one detection.
pub fn detect() -> Result<&'static ProfileDef, String> {
    static CACHED: OnceLock<Result<String, String>> = OnceLock::new();
    let name = CACHED
        .get_or_init(|| {
            if let Ok(v) = std::env::var("SCRATCHY_GPU") {
                let v = v.trim().to_string();
                if !v.is_empty() {
                    return Ok(v);
                }
            }
            detect_via_nvidia_smi()
        })
        .as_ref()
        .map_err(|e| e.clone())?;

    for_name(name).ok_or_else(|| {
        let known: Vec<&str> = ALL.iter().map(|p| p.name).collect();
        format!(
            "no scratchy-target-cuda profile for `{name}`. \
             Set SCRATCHY_GPU=<one of {known:?}>."
        )
    })
}

/// Run `nvidia-smi --query-gpu=name --format=csv,noheader` and
/// normalize the first GPU's name into our short identifier.
/// Public so an integration test can exercise the subprocess path
/// independently of [`detect`]'s `OnceLock` cache.
pub fn detect_via_nvidia_smi() -> Result<String, String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .map_err(|e| {
            format!(
                "SCRATCHY_GPU not set and `nvidia-smi` failed: {e}. \
                 Set SCRATCHY_GPU=<one of {:?}>.",
                ALL.iter().map(|p| p.name).collect::<Vec<_>>(),
            )
        })?;
    if !output.status.success() {
        return Err(format!(
            "nvidia-smi exited {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim(),
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line = stdout
        .lines()
        .next()
        .ok_or_else(|| "nvidia-smi returned no GPUs".to_string())?
        .trim();
    Ok(normalize_gpu_name(first_line))
}

/// Strip the `NVIDIA ` prefix and any trailing memory/variant
/// qualifiers, lowercase. Examples:
///   `NVIDIA L4`              → `l4`
///   `NVIDIA L40S`            → `l40s`
///   `NVIDIA H100 80GB HBM3`  → `h100`
///   `NVIDIA H100 PCIe`       → `h100`
fn normalize_gpu_name(raw: &str) -> String {
    let s = raw.trim();
    let s = s.strip_prefix("NVIDIA ").unwrap_or(s);
    // Take the first whitespace-delimited token — handles
    // "H100 80GB HBM3" → "H100" without us having to enumerate
    // every memory-config suffix NVIDIA ships.
    let first = s.split_whitespace().next().unwrap_or(s);
    first.to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_profiles_round_trip_through_for_name() {
        for p in ALL {
            let looked_up = for_name(p.name).expect("known profile");
            assert_eq!(looked_up.name, p.name);
        }
    }

    #[test]
    fn unknown_profile_returns_none() {
        assert!(for_name("rtx_5090").is_none());
    }

    #[test]
    fn name_normalization_handles_common_nvidia_smi_outputs() {
        assert_eq!(normalize_gpu_name("NVIDIA L4"), "l4");
        assert_eq!(normalize_gpu_name("NVIDIA L40S"), "l40s");
        assert_eq!(normalize_gpu_name("NVIDIA H100 80GB HBM3"), "h100");
        assert_eq!(normalize_gpu_name("NVIDIA H100 PCIe"), "h100");
        // Defensive: missing prefix.
        assert_eq!(normalize_gpu_name("L4"), "l4");
    }

    #[test]
    fn cost_csv_is_nonempty_for_every_profile() {
        for p in ALL {
            assert!(!p.cost_csv.is_empty(), "{} cost_csv empty", p.name);
            assert!(p.cost_csv.contains(','), "{} cost_csv unparseable", p.name);
        }
    }
}

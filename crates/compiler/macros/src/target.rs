// SPDX-License-Identifier: Apache-2.0
//! Target profiles — hardware metadata the cost model consumes.
//!
//! Profile data lives in the `scratchy-target-cuda` crate as `pub
//! const ProfileDef` values; this module bridges from those to the
//! `TargetProfile` shape `impl_lib.rs` consumes (`CostTable` parsed
//! from the embedded CSV, plus the analytic spec fields).

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

#[cfg(feature = "cuda")]
use scratchy_target_cuda::targets::ProfileDef;

/// Empirical GPU cost table: `(kernel_name, M, N, K) -> cost_us`.
///
/// Populated from `target_profiles/cost_<profile>.csv` when present.
/// Keyed by `kernel` column value (`cublas`, `cutlass_128x128_s4`,
/// `cutlass_gemv`, …) to support both the cuBLAS reference line and
/// each cutlass tile variant.
///
/// `kernel_set` is a denormalized cache of the distinct kernel names
/// seen in `entries` — populated lazily on first access. Callers like
/// `Implementation::target_compatible` hit this in their inner loop
/// (per tile × per impl × per workload), so a hot-path `contains()`
/// must be O(1), not O(entries × clone_each_String).
#[derive(Clone, Debug, Default)]
pub struct CostTable {
    entries: HashMap<(String, u32, u32, u32), f64>,
    kernel_set: HashSet<String>,
    /// When true, [`has_kernel`](Self::has_kernel) returns true for ANY name —
    /// so a backend with no empirical cost CSV (spyre/KTIR) passes every impl's
    /// `target_compatible` gate; cost falls back to the analytic formula.
    universal: bool,
}

impl CostTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// A cost table that claims every kernel (`has_kernel` always true) with no
    /// cost entries — analytic cost fallback. The spyre/KTIR target uses this:
    /// it has no GPU cost sweep, and the solver only needs *a* valid impl per op
    /// to produce the standard decode SubtileIR the KTIR emitter lowers.
    pub fn universal() -> Self {
        Self {
            universal: true,
            ..Self::default()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn insert(&mut self, kernel: impl Into<String>, m: u32, n: u32, k: u32, cost_us: f64) {
        let kernel = kernel.into();
        self.kernel_set.insert(kernel.clone());
        self.entries.insert((kernel, m, n, k), cost_us);
    }

    pub fn get(&self, kernel: &str, m: u32, n: u32, k: u32) -> Option<f64> {
        // Use a borrowed-key tuple to skip the per-call String allocation.
        // HashMap's Borrow impl on tuples doesn't quite let us borrow the
        // String directly, so we still allocate here — but this is the
        // cold path (called once the candidate is being evaluated, not
        // per-impl-per-tile-per-workload).
        self.entries.get(&(kernel.to_string(), m, n, k)).copied()
    }

    /// O(1) membership check on kernel names. Used by hot-path
    /// `target_compatible` to gate impls without scanning the cost
    /// table per call.
    pub fn has_kernel(&self, kernel: &str) -> bool {
        self.universal || self.kernel_set.contains(kernel)
    }

    /// Every distinct kernel name observed in the CSV. Useful for
    /// debugging and tests; NOT for hot paths — use [`has_kernel`] for
    /// existence checks.
    pub fn kernel_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.kernel_set.iter().cloned().collect();
        names.sort();
        names
    }
}

/// Backend type discriminator for target profiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cuda,
    Metal,
    /// IBM Spyre / KTIR host emulator — a host-orchestrated backend with no GPU
    /// cost model. The solver only needs a profile to cover the FUF with
    /// spyre's own claim impls, which the KTIR emitter then lowers.
    Spyre,
}

/// CUDA-specific hardware characteristics.
#[derive(Clone, Debug)]
pub struct CudaSpec {
    /// sm_XX — 89 = Ada, 90 = Hopper.
    pub compute_capability: u32,
    /// Number of streaming multiprocessors.
    pub num_sms: u32,
    /// Shared memory per SM, kilobytes.
    pub shared_memory_per_sm_kb: u32,
}

/// Metal-specific hardware characteristics.
#[derive(Clone, Debug)]
pub struct MetalSpec {
    /// Apple Silicon generation (M1, M2, M3, M4).
    pub generation: String,
    /// Number of GPU cores.
    pub gpu_cores: u32,
    /// Threadgroup memory per threadgroup, kilobytes.
    pub threadgroup_memory_kb: u32,
}

/// Hardware characteristics a cost model uses to estimate kernel
/// timing. All units are explicit. Backend-agnostic to support
/// both CUDA and Metal targets.
#[derive(Clone, Debug)]
pub struct TargetProfile {
    pub name: String,
    pub source_path: PathBuf,
    pub backend: Backend,
    /// Peak FP16 tensor-core throughput, teraflops.
    pub peak_tflops_fp16: f64,
    /// Global memory bandwidth, gigabytes per second.
    pub memory_bandwidth_gbps: f64,
    /// Backend-specific hardware details.
    pub backend_spec: BackendSpec,
    /// Empirical cost table loaded from `cost_<name>.csv` alongside
    /// the JSON, when present. Populated with the GPU-swept
    /// measurements from prior scratchy (cublas + every cutlass tile
    /// variant across a grid of `(M, N, K)`). Empty when no CSV is
    /// present — cost impls fall back to their analytic formula.
    pub cost_table: CostTable,
}

/// Backend-specific hardware specification.
#[derive(Clone, Debug)]
pub enum BackendSpec {
    Cuda(CudaSpec),
    Metal(MetalSpec),
    /// Spyre/KTIR host backend — no GPU hardware spec (analytic cost only).
    Spyre,
}

impl TargetProfile {
    /// Look up an empirical cost. Returns `None` when either the
    /// profile has no CSV, or the (kernel, M, N, K) shape isn't in
    /// the swept grid.
    pub fn cost_us_for(&self, kernel: &str, m: u32, n: u32, k: u32) -> Option<f64> {
        self.cost_table.get(kernel, m, n, k)
    }
}

/// Build a `TargetProfile` from a `scratchy-target-cuda` profile
/// const, parsing the embedded CSV bytes into a `CostTable`. The
/// proc-macro calls this once per `#[forward]` invocation after
/// resolving the active GPU (`scratchy_target_cuda::targets::detect()`).
#[cfg(feature = "cuda")]
pub fn from_profile_def(def: &ProfileDef) -> TargetProfile {
    TargetProfile {
        name: def.name.to_string(),
        source_path: PathBuf::new(),
        backend: Backend::Cuda,
        peak_tflops_fp16: def.peak_tflops_fp16,
        memory_bandwidth_gbps: def.memory_bandwidth_gbps,
        backend_spec: BackendSpec::Cuda(CudaSpec {
            compute_capability: def.compute_capability,
            num_sms: def.num_sms,
            shared_memory_per_sm_kb: def.shared_memory_per_sm_kb,
        }),
        cost_table: parse_cost_csv(def.cost_csv),
    }
}

/// Parse the embedded `cost_<gpu>.csv` bytes into a `CostTable`.
/// Format: `kernel,M,N,K,cost_us` rows. Comment lines (`#`-prefixed)
/// and the header are skipped; malformed rows are silently dropped
/// rather than aborting the whole parse.
pub fn parse_cost_csv(csv: &str) -> CostTable {
    let mut table = CostTable::new();
    for raw in csv.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with("kernel,") {
            continue;
        }
        let mut parts = line.splitn(5, ',');
        let (Some(kernel), Some(m), Some(n), Some(k), Some(cost)) = (
            parts.next(),
            parts.next(),
            parts.next(),
            parts.next(),
            parts.next(),
        ) else {
            continue;
        };
        let (Ok(m), Ok(n), Ok(k), Ok(cost_us)) = (
            m.trim().parse::<u32>(),
            n.trim().parse::<u32>(),
            k.trim().parse::<u32>(),
            cost.trim().parse::<f64>(),
        ) else {
            continue;
        };
        table.insert(kernel.trim(), m, n, k, cost_us);
    }
    table
}

// ⛔ CUDA ONLY, AND THE GATE USED TO SAY OTHERWISE. This module names
// `scratchy_target_cuda` and `crate::solver` (instruction selection, itself
// cuda-gated), so under `spyre` the cfg ADMITTED it and it failed to compile —
// taking every other test in this crate down with it. A predicate that is WRONG
// rather than absent, so the crate's whole suite was unrunnable off cuda.
#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use scratchy_target_cuda::targets::{H100_SM90, L4_SM89};

    #[test]
    fn from_profile_def_carries_spec_fields() {
        let l4 = from_profile_def(&L4_SM89);
        assert_eq!(l4.backend, Backend::Cuda);
        if let BackendSpec::Cuda(spec) = &l4.backend_spec {
            assert_eq!(spec.compute_capability, 89);
            assert_eq!(spec.num_sms, 58);
        } else {
            panic!("Expected CUDA backend spec");
        }
        assert!(l4.peak_tflops_fp16 > 100.0);

        let h100 = from_profile_def(&H100_SM90);
        assert_eq!(h100.backend, Backend::Cuda);
        if let BackendSpec::Cuda(spec) = &h100.backend_spec {
            assert_eq!(spec.compute_capability, 90);
        } else {
            panic!("Expected CUDA backend spec");
        }
        assert!(h100.peak_tflops_fp16 > l4.peak_tflops_fp16);
    }

    #[test]
    fn cost_csv_parses_into_table() {
        let l4 = from_profile_def(&L4_SM89);
        // gpu_cost_sweep produces thousands of rows across kernel ×
        // (M, N, K). The embedded CSV survives the parse.
        assert!(!l4.cost_table.is_empty(), "expected cost_l4_sm89 rows");
        assert!(l4.cost_table.len() > 1000);
        let kinds = l4.cost_table.kernel_names();
        assert!(kinds.iter().any(|k| k == "cublas"), "cublas baseline");
        assert!(
            kinds.iter().any(|k| k.starts_with("cutlass_")),
            "cutlass tile variants"
        );
        assert!(
            kinds.iter().any(|k| k == "cutlass_gemv"),
            "cutlass_gemv for M=1"
        );
    }

    #[test]
    fn cost_lookup_round_trips() {
        let l4 = from_profile_def(&L4_SM89);
        // The first data row in cost_l4_sm89.csv is
        // `cublas,1,2048,2048,9.7`. Use it as a canary.
        let c = l4.cost_us_for("cublas", 1, 2048, 2048);
        assert!(c.is_some(), "cublas 1x2048x2048 missing from table");
        let us = c.unwrap();
        assert!(us > 0.0 && us.is_finite());
    }

    #[test]
    fn parse_cost_csv_skips_garbage() {
        let csv = "\
            # comment line\n\
            kernel,M,N,K,cost_us\n\
            \n\
            cublas,1,2,3,4.5\n\
            malformed_row\n\
            cutlass_128x128_s3,8,16,32,7.0\n\
            cublas,not_a_number,2,3,4.5\n";
        let table = parse_cost_csv(csv);
        assert_eq!(table.len(), 2);
        assert_eq!(table.get("cublas", 1, 2, 3), Some(4.5));
        assert_eq!(table.get("cutlass_128x128_s3", 8, 16, 32), Some(7.0));
    }
}

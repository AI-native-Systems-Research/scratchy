// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr collect-env` subcommand — print environment information for bug reports.
//!
//! Mirrors Python vLLM's `vllm collect-env` output structure, substituting
//! Rust toolchain info for PyTorch/Python sections.

use std::process::Command;

use anyhow::Result;

use crate::args::CollectEnvArgs;

const NA: &str = "N/A";

fn run_cmd(program: &str, args: &[&str]) -> Option<String> {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

fn first_line(cmd: &str, args: &[&str]) -> String {
    run_cmd(cmd, args)
        .and_then(|out| out.lines().next().map(|l| l.to_string()))
        .unwrap_or_else(|| NA.to_string())
}

fn or_na(opt: Option<String>) -> String {
    opt.unwrap_or_else(|| NA.to_string())
}

// ---------------------------------------------------------------------------
// System info
// ---------------------------------------------------------------------------

fn get_os_info() -> String {
    #[cfg(target_os = "macos")]
    {
        let version = run_cmd("sw_vers", &["-productVersion"]).unwrap_or_default();
        format!("macOS {} ({})", version, std::env::consts::ARCH)
    }
    #[cfg(target_os = "linux")]
    {
        let pretty = run_cmd("sh", &["-c", "cat /etc/*-release"]).and_then(|out| {
            out.lines()
                .find(|l| l.starts_with("PRETTY_NAME="))
                .map(|l| {
                    l.trim_start_matches("PRETTY_NAME=")
                        .trim_matches('"')
                        .to_string()
                })
        });
        let arch = std::env::consts::ARCH;
        match pretty {
            Some(name) => format!("{name} ({arch})"),
            None => format!("Linux ({arch})"),
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        format!("{} ({})", std::env::consts::OS, std::env::consts::ARCH)
    }
}

fn get_libc_version() -> String {
    #[cfg(target_os = "linux")]
    {
        // Try ldd --version to get glibc version.
        run_cmd("sh", &["-c", "ldd --version 2>&1"])
            .and_then(|out| out.lines().next().map(|l| l.to_string()))
            .unwrap_or_else(|| NA.to_string())
    }
    #[cfg(not(target_os = "linux"))]
    {
        NA.to_string()
    }
}

// ---------------------------------------------------------------------------
// CPU info
// ---------------------------------------------------------------------------

fn get_cpu_info() -> String {
    #[cfg(target_os = "macos")]
    {
        run_cmd("sysctl", &["-n", "machdep.cpu.brand_string"])
            .unwrap_or_else(|| "Unknown".to_string())
    }
    #[cfg(target_os = "linux")]
    {
        run_cmd("lscpu", &[]).unwrap_or_else(|| "Unknown".to_string())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        "Unknown".to_string()
    }
}

// ---------------------------------------------------------------------------
// CUDA / GPU info
// ---------------------------------------------------------------------------

fn get_gpu_info() -> String {
    if let Some(out) = run_cmd("nvidia-smi", &["-L"]) {
        // Strip UUIDs like Python does.
        let lines: Vec<String> = out
            .lines()
            .map(|l| {
                if let Some(idx) = l.find(" (UUID:") {
                    l[..idx].to_string()
                } else {
                    l.to_string()
                }
            })
            .collect();
        let joined = lines.join("\n");
        return if lines.len() > 1 {
            format!("\n{joined}")
        } else {
            joined
        };
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(out) = run_cmd(
            "system_profiler",
            &["SPDisplaysDataType", "-detailLevel", "mini"],
        ) {
            let chips: Vec<&str> = out
                .lines()
                .filter(|l| l.contains("Chipset Model:") || l.contains("Chip:"))
                .collect();
            if !chips.is_empty() {
                return chips
                    .iter()
                    .map(|l| l.trim())
                    .collect::<Vec<_>>()
                    .join("\n");
            }
        }
    }
    "No GPU detected".to_string()
}

fn get_nvidia_driver_version() -> String {
    #[cfg(target_os = "macos")]
    {
        NA.to_string()
    }
    #[cfg(not(target_os = "macos"))]
    {
        run_cmd("nvidia-smi", &[])
            .and_then(|out| {
                out.lines()
                    .find(|l| l.contains("Driver Version:"))
                    .and_then(|l| {
                        l.split("Driver Version:")
                            .nth(1)
                            .and_then(|rest| rest.split_whitespace().next())
                            .map(|v| v.to_string())
                    })
            })
            .unwrap_or_else(|| NA.to_string())
    }
}

fn get_cuda_runtime_version() -> String {
    or_na(run_cmd("nvcc", &["--version"]).and_then(|out| {
        out.lines()
            .find(|l| l.contains("release"))
            .and_then(|l| l.split('V').next_back())
            .map(|v| v.trim().to_string())
    }))
}

fn is_cuda_available() -> &'static str {
    if run_cmd("nvidia-smi", &["-L"]).is_some() {
        "Yes"
    } else {
        "No"
    }
}

fn get_cudnn_version() -> String {
    #[cfg(target_os = "linux")]
    {
        or_na(
            run_cmd(
                "sh",
                &[
                    "-c",
                    "ldconfig -p | grep libcudnn | rev | cut -d' ' -f1 | rev",
                ],
            )
            .map(|out| {
                let mut files: Vec<String> = out
                    .lines()
                    .filter_map(|l| {
                        let path = l.trim();
                        if std::path::Path::new(path).exists() {
                            std::fs::canonicalize(path)
                                .ok()
                                .map(|p| p.display().to_string())
                        } else {
                            None
                        }
                    })
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .collect();
                files.sort();
                files.join("\n")
            })
            .filter(|s| !s.is_empty()),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        NA.to_string()
    }
}

fn get_gpu_topo() -> String {
    #[cfg(target_os = "linux")]
    {
        run_cmd("nvidia-smi", &["topo", "-m"])
            .or_else(|| run_cmd("rocm-smi", &["--showtopo"]))
            .unwrap_or_else(|| NA.to_string())
    }
    #[cfg(not(target_os = "linux"))]
    {
        NA.to_string()
    }
}

// ---------------------------------------------------------------------------
// Build features
// ---------------------------------------------------------------------------

fn get_build_features() -> String {
    let mut features = Vec::new();
    if cfg!(feature = "cuda") {
        features.push("cuda");
    }
    if cfg!(feature = "metal") {
        features.push("metal");
    }
    if cfg!(feature = "nccl") {
        features.push("nccl");
    }
    if cfg!(feature = "guided-decoding") {
        features.push("guided-decoding");
    }
    // chat-template is always enabled (built into scratchy-serving-api).
    features.push("chat-template");
    if cfg!(feature = "metrics") {
        features.push("metrics");
    }
    if cfg!(feature = "tls") {
        features.push("tls");
    }
    if cfg!(feature = "bench") {
        features.push("bench");
    }
    if features.is_empty() {
        "none".to_string()
    } else {
        features.join(", ")
    }
}

// ---------------------------------------------------------------------------
// Environment variables
// ---------------------------------------------------------------------------

fn get_env_vars() -> String {
    let env_prefixes = [
        "VLLM_",
        "TORCH",
        "NCCL",
        "PYTORCH",
        "CUDA",
        "CUBLAS",
        "CUDNN",
        "OMP_",
        "MKL_",
        "NVIDIA",
        "HF_",
        "HUGGING_FACE",
        "RUST",
        "CARGO",
    ];
    let secret_terms = ["secret", "token", "api", "access", "password", "key"];

    let mut env_vars = Vec::new();
    for (k, v) in std::env::vars() {
        if secret_terms.iter().any(|t| k.to_lowercase().contains(t)) {
            continue;
        }
        if env_prefixes.iter().any(|p| k.starts_with(p)) {
            env_vars.push(format!("{k}={v}"));
        }
    }
    env_vars.sort();
    if env_vars.is_empty() {
        "(none)".to_string()
    } else {
        env_vars.join("\n")
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub async fn run_collect_env(_args: CollectEnvArgs) -> Result<()> {
    println!("Collecting environment information...");

    let os_info = get_os_info();
    let gcc = first_line("gcc", &["--version"]);
    let clang = first_line("clang", &["--version"]);
    let cmake = first_line("cmake", &["--version"]);
    let libc = get_libc_version();

    let rustc = or_na(run_cmd("rustc", &["--version"]));
    let cargo = or_na(run_cmd("cargo", &["--version"]));

    let is_cuda = is_cuda_available();
    let cuda_runtime = get_cuda_runtime_version();
    let gpu_models = get_gpu_info();
    let nvidia_driver = get_nvidia_driver_version();
    let cudnn = get_cudnn_version();

    let cpu_info = get_cpu_info();
    let gpu_topo = get_gpu_topo();

    let vllm_version = env!("CARGO_PKG_VERSION");
    let build_features = get_build_features();
    let env_vars = get_env_vars();

    println!(
        "\
==============================
        System Info
==============================
OS                           : {os_info}
GCC version                  : {gcc}
Clang version                : {clang}
CMake version                : {cmake}
Libc version                 : {libc}

==============================
       Rust Toolchain
==============================
Rustc version                : {rustc}
Cargo version                : {cargo}

==============================
       CUDA / GPU Info
==============================
Is CUDA available            : {is_cuda}
CUDA runtime version         : {cuda_runtime}
GPU models and configuration : {gpu_models}
Nvidia driver version        : {nvidia_driver}
cuDNN version                : {cudnn}

==============================
          CPU Info
==============================
{cpu_info}

==============================
         vLLM-rs Info
==============================
vLLM-rs version              : {vllm_version}
Build features               : {build_features}
GPU Topology:
  {gpu_topo}

==============================
     Environment Variables
==============================
{env_vars}"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_or_na() {
        assert_eq!(or_na(Some("hello".to_string())), "hello");
        assert_eq!(or_na(None), "N/A");
    }

    #[test]
    fn test_run_cmd_success() {
        let out = run_cmd("echo", &["hello"]);
        assert_eq!(out, Some("hello".to_string()));
    }

    #[test]
    fn test_run_cmd_failure() {
        let out = run_cmd("__nonexistent_command_12345__", &[]);
        assert!(out.is_none());
    }

    #[test]
    fn test_first_line() {
        // echo prints a single line.
        let line = first_line("echo", &["first\nsecond"]);
        assert_eq!(line, "first");
    }

    #[test]
    fn test_first_line_missing_cmd() {
        let line = first_line("__nonexistent__", &[]);
        assert_eq!(line, "N/A");
    }

    #[test]
    fn test_get_os_info_not_empty() {
        let info = get_os_info();
        assert!(!info.is_empty());
    }

    #[test]
    fn test_get_cpu_info_not_empty() {
        let info = get_cpu_info();
        assert!(!info.is_empty());
    }

    #[test]
    fn test_get_build_features_contains_expected() {
        let features = get_build_features();
        // Default features include chat-template and guided-decoding.
        assert!(features.contains("chat-template"));
        assert!(features.contains("guided-decoding"));
    }

    #[test]
    fn test_get_env_vars_filters_secrets() {
        // Set a var that should be filtered out.
        unsafe {
            std::env::set_var("VLLM_TEST_SECRET_KEY", "sensitive");
            std::env::set_var("VLLM_TEST_VISIBLE", "visible");
        }
        let vars = get_env_vars();
        assert!(!vars.contains("VLLM_TEST_SECRET_KEY"));
        assert!(vars.contains("VLLM_TEST_VISIBLE"));
        unsafe {
            std::env::remove_var("VLLM_TEST_SECRET_KEY");
            std::env::remove_var("VLLM_TEST_VISIBLE");
        }
    }
}

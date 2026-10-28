// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr launch claude --best-model` (`-A`) — pick the model to serve from this
//! Mac's memory class instead of naming one. Metal only.
//!
//! The pick is the biggest quantized checkpoint in [`TIERS`] whose memory floor
//! the machine clears AND whose config this binary compiled in: a model the
//! build cannot load would fail minutes later with "no compiled model variant
//! matches this checkpoint". When nothing that fits is compiled in, launch
//! stops with the build line that compiles it.
//!
//! The context window is deliberately not part of the pick. Launch asks for
//! none, and the server serves the model's native context capped to what this
//! machine's KV cache holds (see `serve_argv`) — already the biggest window the
//! GPU funds, where a per-tier guess would either waste it or ask for more than
//! fits and be refused. The compressed TurboQuant KV cache a `metal` build
//! compiles in is what stretches that window.
//!
//! `-a <MODEL>` keeps the `--best-model` settings but serves `<MODEL>` instead
//! of the pick.

use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::args::LaunchClaudeArgs;

const GIB: u64 = 1 << 30;

/// One rung of the ladder: the smallest Mac that serves `model` with room left
/// for a Claude Code-sized KV cache.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Tier {
    /// Total unified RAM floor in GiB.
    min_gib: u64,
    /// HuggingFace id served.
    pub(crate) model: &'static str,
    /// The `configs/<arch>/<stem>.json` that has to be compiled in to load it.
    pub(crate) stem: &'static str,
    /// The quant preset that checkpoint loads under.
    pub(crate) quant: &'static str,
    /// What the weights are, for the one line launch prints.
    pub(crate) weights: &'static str,
}

impl Tier {
    /// The name the build gives this tier's compiled variant: a quant preset
    /// emits `<stem>-<preset>` (`qwen3.6-35b-a3b-mlx-affine-b4-g64-qembed`), the
    /// name `scr model info` prints. The bare stem is the dense variant, which
    /// cannot load this tier's quantized checkpoint, so it does not count.
    fn variant_stem(&self) -> String {
        format!("{}-{}", self.stem, self.quant)
    }
}

/// Biggest first. Every one is an MLX checkpoint with quantized embeddings,
/// hence the one `-qembed` preset. Weights on disk: 43.6 / 19.0 / 5.6 / 1.6
/// GiB, against a Metal working set of ~75% of RAM.
pub(crate) const TIERS: &[Tier] = &[
    Tier {
        min_gib: 96,
        model: "mlx-community/Qwen3.5-122B-A10B-OptiQ-2bit",
        stem: "qwen3.5-122b-a10b",
        quant: "mlx-affine-b4-g64-qembed",
        weights: "MLX 2/4-bit mixed",
    },
    Tier {
        min_gib: 36,
        model: "mlx-community/Qwen3.6-35B-A3B-4bit",
        stem: "qwen3.6-35b-a3b",
        quant: "mlx-affine-b4-g64-qembed",
        weights: "MLX 4-bit",
    },
    Tier {
        min_gib: 16,
        model: "mlx-community/Qwen3.5-9B-4bit",
        stem: "qwen3.5-9b",
        quant: "mlx-affine-b4-g64-qembed",
        weights: "MLX 4-bit",
    },
    Tier {
        min_gib: 0,
        model: "mlx-community/Qwen2.5-3B-Instruct-4bit",
        stem: "qwen2.5-3b",
        quant: "mlx-affine-b4-g64-qembed",
        weights: "MLX 4-bit",
    },
];

/// What auto mode detected about this machine.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Gpu {
    pub(crate) name: String,
    pub(crate) mem_bytes: u64,
}

/// The tier auto mode serves, and whether this binary can load it.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Pick {
    pub(crate) tier: &'static Tier,
    /// A bigger rung the machine fits but this binary did not compile.
    pub(crate) bigger_uncompiled: Option<&'static Tier>,
}

/// The biggest tier the machine fits whose stem is compiled in. `Err` carries
/// the biggest tier that fits when none of the fitting ones is compiled — the
/// one to tell the user to build.
pub(crate) fn pick(mem_bytes: u64, compiled: &[String]) -> Result<Pick, &'static Tier> {
    let mut fitting = TIERS
        .iter()
        .filter(|t| t.min_gib * GIB <= mem_bytes)
        .peekable();
    // The last rung's floor is 0, so something always fits.
    let biggest = *fitting.peek().expect("the tier ladder ends at 0 GiB");
    let tier = fitting
        .find(|t| compiled.iter().any(|s| *s == t.variant_stem()))
        .ok_or(biggest)?;
    Ok(Pick {
        tier,
        bigger_uncompiled: (tier != biggest).then_some(biggest),
    })
}

/// The build line that compiles `tier` into a launch-capable binary.
pub(crate) fn build_line(tier: &Tier) -> String {
    format!(
        "cargo build --release -p scratchy-cli --features metal,claude,model/{},quant/{}",
        tier.stem, tier.quant
    )
}

/// `sysctl -n hw.memsize machdep.cpu.brand_string` output: the RAM byte count,
/// then the chip name.
fn parse_sysctl(out: &str) -> Option<Gpu> {
    let mut lines = out.lines();
    let mem_bytes = lines.next()?.trim().parse().ok()?;
    let name = lines.next().map(str::trim).unwrap_or("Apple Silicon");
    Some(Gpu {
        name: name.to_string(),
        mem_bytes,
    })
}

fn detect() -> Result<Gpu> {
    let out = Command::new("sysctl")
        .args(["-n", "hw.memsize", "machdep.cpu.brand_string"])
        .output()
        .context("--best-model could not run `sysctl` to size the GPU")?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    parse_sysctl(&stdout).with_context(|| {
        format!(
            "--best-model could not read this Mac's memory from `sysctl` (got {:?}); name a \
             model instead: `scr launch claude <MODEL>`",
            stdout.trim()
        )
    })
}

/// Resolve `--best-model` for `args`: `None` when neither `-A` nor `-a` was
/// given, otherwise the model to serve.
pub(crate) fn resolve(args: &LaunchClaudeArgs) -> Result<Option<String>> {
    if !args.best_model && args.best_model_override.is_none() {
        return Ok(None);
    }
    if !cfg!(feature = "metal") {
        bail!(
            "--best-model picks a model for this Mac's GPU, and this build has no metal backend. \
             Rebuild with `--features metal`."
        );
    }
    if args.device == "cpu" {
        bail!("--best-model sizes the model to the GPU; it has nothing to size with --device cpu");
    }
    let gpu = detect()?;
    let mem_gib = gpu.mem_bytes as f64 / GIB as f64;

    if let Some(model) = &args.best_model_override {
        eprintln!(
            "scratchy best model: {} ({mem_gib:.0} GiB) → {model} (from -a; context sized by the \
             server to what the KV cache holds)",
            gpu.name
        );
        return Ok(Some(model.clone()));
    }

    let compiled = super::model::compiled_variant_stems();
    let pick = pick(gpu.mem_bytes, &compiled).map_err(|tier| {
        anyhow::anyhow!(
            "--best-model picked {} for {} ({mem_gib:.0} GiB), but this binary does not compile \
             it in. Build it with:\n  {}\nor name a model this build has: `scr launch claude \
             -a <MODEL>`.",
            tier.model,
            gpu.name,
            build_line(tier),
        )
    })?;
    if let Some(bigger) = pick.bigger_uncompiled {
        eprintln!(
            "Note: this machine also fits {}, which this binary does not compile in:\n  {}",
            bigger.model,
            build_line(bigger),
        );
    }
    eprintln!(
        "scratchy best model: {} ({mem_gib:.0} GiB) → {} ({} weights; context sized by the server \
         to what the KV cache holds)",
        gpu.name, pick.tier.model, pick.tier.weights
    );
    Ok(Some(pick.tier.model.to_string()))
}

#[cfg(test)]
mod tests {
    use super::{GIB, TIERS, build_line, parse_sysctl, pick};

    fn stems(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    /// The ladder is biggest-first and ends at a 0 GiB floor, so `pick` always
    /// has a rung to fall back to.
    #[test]
    fn ladder_descends_to_zero() {
        assert!(TIERS.windows(2).all(|w| w[0].min_gib > w[1].min_gib));
        assert_eq!(TIERS.last().unwrap().min_gib, 0);
    }

    #[test]
    fn picks_the_biggest_compiled_tier_that_fits() {
        let all: Vec<String> = TIERS.iter().map(|t| t.variant_stem()).collect();
        let at = |gib: u64| pick(gib * GIB, &all).unwrap().tier.model;
        assert_eq!(at(128), "mlx-community/Qwen3.5-122B-A10B-OptiQ-2bit");
        assert_eq!(at(64), "mlx-community/Qwen3.6-35B-A3B-4bit");
        assert_eq!(at(36), "mlx-community/Qwen3.6-35B-A3B-4bit");
        assert_eq!(at(24), "mlx-community/Qwen3.5-9B-4bit");
        assert_eq!(at(8), "mlx-community/Qwen2.5-3B-Instruct-4bit");
    }

    /// A tier counts as compiled in by the name `scr model info` prints for it:
    /// the stem with the quant preset appended, which is what a `quant/mlx`
    /// build emits. Matching the bare stem refused a binary that had the 35B.
    #[test]
    fn matches_the_preset_suffixed_variant_a_quant_build_emits() {
        let built = stems(&["qwen3.6-35b-a3b-mlx-affine-b4-g64-qembed"]);
        let p = pick(64 * GIB, &built).unwrap();
        assert_eq!(p.tier.model, "mlx-community/Qwen3.6-35B-A3B-4bit");
        assert_eq!(p.bigger_uncompiled, None);
    }

    /// The bare stem is the dense variant, which cannot load the tier's
    /// quantized checkpoint, so it does not make the tier compiled in.
    #[test]
    fn a_dense_only_build_does_not_count() {
        let err = pick(64 * GIB, &stems(&["qwen3.6-35b-a3b"])).unwrap_err();
        assert_eq!(err.stem, "qwen3.6-35b-a3b");
    }

    /// A bigger rung that fits but is not compiled in falls through to the
    /// biggest one that is, and names the one it skipped.
    #[test]
    fn falls_back_to_a_compiled_tier_and_names_the_skipped_one() {
        let built = stems(&[
            "qwen3.5-9b-mlx-affine-b4-g64-qembed",
            "llama-3.2-1b-mlx-affine-b4-g64",
        ]);
        let p = pick(64 * GIB, &built).unwrap();
        assert_eq!(p.tier.stem, "qwen3.5-9b");
        assert_eq!(p.bigger_uncompiled.unwrap().stem, "qwen3.6-35b-a3b");
    }

    /// Only a tier that FITS may be picked: a compiled 35B on a 16 GiB Mac is
    /// not an option, and nothing fitting compiled in is an error naming the
    /// biggest rung that fits.
    #[test]
    fn never_picks_a_tier_that_does_not_fit() {
        let built = stems(&["qwen3.6-35b-a3b-mlx-affine-b4-g64-qembed"]);
        let err = pick(16 * GIB, &built).unwrap_err();
        assert_eq!(err.stem, "qwen3.5-9b");
    }

    #[test]
    fn build_line_names_model_and_quant() {
        assert_eq!(
            build_line(&TIERS[1]),
            "cargo build --release -p scratchy-cli --features \
             metal,claude,model/qwen3.6-35b-a3b,quant/mlx-affine-b4-g64-qembed"
        );
    }

    #[test]
    fn parses_sysctl() {
        let gpu = parse_sysctl("68719476736\nApple M5 Max\n").unwrap();
        assert_eq!(gpu.mem_bytes, 64 * GIB);
        assert_eq!(gpu.name, "Apple M5 Max");
        assert_eq!(parse_sysctl(""), None);
    }
}

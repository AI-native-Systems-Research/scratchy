// SPDX-License-Identifier: Apache-2.0
//! The surveyed templates, pinned.
//!
//! ⛔ PINNED BY HASH, AND THE HASH IS THE POINT. Upstream templates are edited in
//! place: `google/gemma-4-12b-it` ships no `chat_template` at all while the
//! `unsloth` mirror of the same model ships 18,924 B, and the published figure for
//! that template elsewhere is 18,683 B — i.e. revisions of "the same" template
//! differ. Re-fetching whatever upstream serves today would silently redefine the
//! sample under the conclusions in ../SPEC.md, so a changed hash is reported as
//! DRIFT rather than absorbed.
//!
//! This is the same idea as the spec's `validated_against` directive, applied to
//! the survey itself.

use std::path::{Path, PathBuf};

/// One surveyed template.
pub struct Fixture {
    /// Short name used in reports and as the cache filename.
    pub name: &'static str,
    /// Architecture directory in `crates/models/arch/configs/` this stands for.
    pub arch: &'static str,
    /// HuggingFace repo, in fetch order. Later entries are ungated mirrors for
    /// gated repos; the template bytes are the mirror's, which is why they are
    /// hashed.
    pub repos: &'static [&'static str],
    /// Expected SHA-256 of the template, lowercase hex. Empty means "not yet
    /// pinned" — the fetcher prints what it got so it can be filled in.
    pub sha256: &'static str,
    /// Expected byte length, as a second, human-checkable pin.
    pub bytes: usize,
}

/// ⛔ ARCH COVERAGE IS A CLAIM IN SPEC.md ("23 of the 25 architecture families").
/// `arch` fields here are what makes that checkable: `cargo run -- coverage`
/// diffs this table against the real `configs/` directory listing, so the claim
/// cannot quietly rot as architectures are added.
pub const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "commandr",
        arch: "commandr",
        bytes: 0,
        sha256: "",
        repos: &[
            "CohereLabs/c4ai-command-r-v01",
            "CohereForAI/c4ai-command-r-v01",
            "CohereLabs/c4ai-command-r-08-2024",
            "alpindale/c4ai-command-r-v01",
        ],
    },
    Fixture {
        name: "deepseek-v2",
        arch: "deepseek-v2",
        bytes: 459,
        sha256: "8aeba567270fa9a8d5372caf4b04affea947c4421d96adc01e3ff94021b0ae8e",
        repos: &["deepseek-ai/DeepSeek-V2-Lite-Chat"],
    },
    Fixture {
        name: "deepseek-v3",
        arch: "deepseek-v3",
        bytes: 2301,
        sha256: "3b8267e54b67df6500f231ffa969955cc9d26b3444a291ced4eb96ca7d3ca1ea",
        repos: &["deepseek-ai/DeepSeek-V3"],
    },
    Fixture {
        name: "deepseek-v32",
        arch: "deepseek-v3-flat",
        bytes: 3243,
        sha256: "6e50812c1ed791d81930e675021f2524b075f5f23558305b7eef5c030b4c61f2",
        repos: &["deepseek-ai/DeepSeek-V3.2-Exp"],
    },
    Fixture {
        name: "gemma2",
        arch: "gemma2",
        bytes: 591,
        sha256: "ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6",
        repos: &["unsloth/gemma-2-9b-it", "google/gemma-2-9b-it"],
    },
    Fixture {
        name: "gemma3",
        arch: "gemma3",
        bytes: 1532,
        sha256: "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4",
        repos: &["unsloth/gemma-3-4b-it", "google/gemma-3-4b-it"],
    },
    Fixture {
        name: "gemma4",
        arch: "gemma4",
        bytes: 18924,
        sha256: "845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b",
        repos: &["unsloth/gemma-4-12b-it", "google/gemma-4-12b-it"],
    },
    Fixture {
        name: "granite",
        arch: "granite",
        bytes: 4566,
        sha256: "0bea7711316005a1c37c9941a7ee2a26311d223c979abee416aab0a402fc5b56",
        repos: &["ibm-granite/granite-3.3-8b-instruct"],
    },
    Fixture {
        name: "llama32",
        arch: "llama",
        bytes: 3827,
        sha256: "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4",
        repos: &[
            "unsloth/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
        ],
    },
    Fixture {
        name: "smollm2",
        arch: "llama",
        bytes: 368,
        sha256: "872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788",
        repos: &["HuggingFaceTB/SmolLM2-135M-Instruct"],
    },
    Fixture {
        name: "tinyllama",
        arch: "llama",
        bytes: 410,
        sha256: "66291cf0045c2425a3a667cf3cbb7af2b11f09e025c02f97245323ab79119362",
        repos: &["TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
    },
    Fixture {
        name: "mistral",
        arch: "mistral",
        bytes: 3959,
        sha256: "e16746b40344d6c5b5265988e0328a0bf7277be86f1c335156eae07e29c82826",
        repos: &["mistralai/Mistral-7B-Instruct-v0.3"],
    },
    Fixture {
        name: "mixtral",
        arch: "mixtral",
        bytes: 1058,
        sha256: "796853172235ab3aaa4810013ddfe2eab481bda3306dad4f700c00d436eed596",
        repos: &["mistralai/Mixtral-8x7B-Instruct-v0.1"],
    },
    Fixture {
        name: "moonlight",
        arch: "deepseek-v3",
        bytes: 527,
        sha256: "0af7f5845651f3ba85a82d70d8af968a1da0a60f4c7bb662a9c305cfc9cc224c",
        repos: &["moonshotai/Moonlight-16B-A3B-Instruct"],
    },
    Fixture {
        name: "phi3",
        arch: "phi3",
        bytes: 407,
        sha256: "dcaee66df77bfbb789ec541fe713218da30225a526f1db97138fc39ee6dc56aa",
        repos: &["microsoft/Phi-3-mini-4k-instruct"],
    },
    Fixture {
        name: "phi4",
        arch: "phi3",
        bytes: 423,
        sha256: "febf589225c9728ab791f52e8897d7607a823d45368f0a4c92fa68997b40cce9",
        repos: &["microsoft/Phi-4-mini-instruct"],
    },
    Fixture {
        name: "qwen2",
        arch: "qwen2",
        bytes: 328,
        sha256: "793202280c0910ab26bc6eb57c8212c2417324c6d8c4efd0d10d59092ab3e3eb",
        repos: &["Qwen/Qwen2-1.5B-Instruct"],
    },
    Fixture {
        name: "qwen25",
        arch: "qwen2",
        bytes: 2507,
        sha256: "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f",
        repos: &["Qwen/Qwen2.5-7B-Instruct"],
    },
    Fixture {
        name: "qwen2vl",
        arch: "qwen2-vl",
        bytes: 1017,
        sha256: "a0bc6f6fc7a29a80017a433e8f03a1cc1236e838a944a2d034295a60c4f2fddb",
        repos: &["Qwen/Qwen2-VL-2B-Instruct"],
    },
    Fixture {
        name: "qwen25vl",
        arch: "qwen2-5-vl",
        bytes: 1017,
        sha256: "a0bc6f6fc7a29a80017a433e8f03a1cc1236e838a944a2d034295a60c4f2fddb",
        repos: &["Qwen/Qwen2.5-VL-3B-Instruct"],
    },
    Fixture {
        name: "qwen3",
        arch: "qwen3",
        bytes: 4168,
        sha256: "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        repos: &["Qwen/Qwen3-8B"],
    },
    Fixture {
        name: "qwen35",
        arch: "qwen3-5",
        bytes: 7756,
        sha256: "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
        repos: &["Qwen/Qwen3.5-9B"],
    },
    Fixture {
        name: "qwen3moe",
        arch: "qwen3-moe",
        bytes: 2630,
        sha256: "64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326",
        repos: &["Qwen/Qwen3-30B-A3B-Instruct-2507"],
    },
    Fixture {
        name: "qwen3next",
        arch: "qwen3-moe",
        bytes: 2630,
        sha256: "64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326",
        repos: &["Qwen/Qwen3-Next-80B-A3B-Instruct"],
    },
];

/// Architectures with no chat template to survey, and why. Named explicitly so
/// `coverage` can distinguish "not applicable" from "we forgot".
pub const NOT_CHAT_MODELS: &[(&str, &str)] = &[
    ("modernbert", "encoder, not a chat model"),
    ("locateanything", "detection model, no chat template"),
];

/// Architectures intentionally covered by another fixture's template lineage.
pub const SHARED_LINEAGE: &[(&str, &str)] = &[
    ("gemma3-mm", "same template as gemma3"),
    ("gemma4-moe", "same template as gemma4"),
    ("qwen2-moe", "same template as qwen2"),
    ("qwen3-5-moe", "same template as qwen3.5"),
    ("qwen3-5-vl", "same template as qwen3.5"),
];

pub fn cache_dir() -> PathBuf {
    // Under target/ so it is already gitignored: these are third-party templates
    // and vendoring 64 KB of them to serve a dev tool is not worth the review.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../../target/chat-survey-cache")
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Pull `chat_template` out of a `tokenizer_config.json` body.
fn extract_template(body: &str) -> Option<String> {
    let cfg: serde_json::Value = serde_json::from_str(body).ok()?;
    match cfg.get("chat_template")? {
        // A few repos ship a LIST of named templates rather than one string.
        serde_json::Value::Array(a) => a
            .first()
            .and_then(|e| e.get("template"))
            .and_then(|t| t.as_str())
            .map(str::to_owned),
        serde_json::Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

fn http_get(url: &str) -> Result<String, String> {
    let mut res = ureq::get(url).call().map_err(|e| e.to_string())?;
    res.body_mut().read_to_string().map_err(|e| e.to_string())
}

/// What happened for one fixture.
pub enum Outcome {
    /// Already cached and the hash matches its pin.
    Cached,
    /// Fetched now. Carries the repo it came from and the hash observed.
    Fetched { repo: String, sha: String },
    /// Hash or length differs from the pin — upstream edited the template.
    Drift { repo: String, got: String },
    /// Hash matches but the recorded byte count does not. Means the `bytes` pin in
    /// this table was mistyped, not that upstream changed: fix the table.
    BadLengthPin { sha: String, got: usize },
    /// Not obtainable from any listed repo.
    Unavailable,
}

/// Fetch every fixture that is not already cached, verifying pins.
pub fn sync() -> Vec<(&'static Fixture, Outcome)> {
    let dir = cache_dir();
    let _ = std::fs::create_dir_all(&dir);
    let mut out = Vec::new();

    for f in FIXTURES {
        let path = dir.join(format!("{}.jinja", f.name));
        if let Ok(existing) = std::fs::read(&path) {
            let got = sha256_hex(&existing);
            if !f.sha256.is_empty() && got != f.sha256 {
                out.push((
                    f,
                    Outcome::Drift {
                        repo: "cache".into(),
                        got,
                    },
                ));
            } else if f.bytes != 0 && existing.len() != f.bytes {
                out.push((
                    f,
                    Outcome::BadLengthPin {
                        sha: got,
                        got: existing.len(),
                    },
                ));
            } else {
                out.push((f, Outcome::Cached));
            }
            continue;
        }

        let mut done = false;
        for repo in f.repos {
            let url = format!("https://huggingface.co/{repo}/resolve/main/tokenizer_config.json");
            let Ok(body) = http_get(&url) else { continue };
            let Some(tpl) = extract_template(&body) else {
                continue;
            };
            if tpl.trim().is_empty() {
                continue;
            }
            let sha = sha256_hex(tpl.as_bytes());
            if !f.sha256.is_empty() && sha != f.sha256 {
                out.push((
                    f,
                    Outcome::Drift {
                        repo: (*repo).into(),
                        got: sha,
                    },
                ));
                done = true;
                break;
            }
            let _ = std::fs::write(&path, &tpl);
            if f.bytes != 0 && tpl.len() != f.bytes {
                out.push((
                    f,
                    Outcome::BadLengthPin {
                        sha,
                        got: tpl.len(),
                    },
                ));
            } else {
                out.push((
                    f,
                    Outcome::Fetched {
                        repo: (*repo).into(),
                        sha,
                    },
                ));
            }
            done = true;
            break;
        }
        if !done {
            out.push((f, Outcome::Unavailable));
        }
    }
    out
}

/// Templates available on disk, as (name, source). Only these are analysed.
pub fn loaded() -> Vec<(&'static str, String)> {
    let dir = cache_dir();
    FIXTURES
        .iter()
        .filter_map(|f| {
            std::fs::read_to_string(dir.join(format!("{}.jinja", f.name)))
                .ok()
                .map(|s| (f.name, s))
        })
        .collect()
}

/// Compare the fixture table's `arch` fields against the real configs tree.
pub fn configs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../models/arch/configs")
}

pub fn arch_families(dir: &Path) -> Vec<String> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut v: Vec<String> = rd
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().to_str().map(str::to_owned))
        .collect();
    v.sort();
    v
}

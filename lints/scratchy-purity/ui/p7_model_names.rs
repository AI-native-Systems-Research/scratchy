//! A model name in logic. Clean: `qwen3` and `gemma` in this doc comment decide nothing.

// Flagged: a `match` arm pattern, both sides of an or-pattern.
fn heads(arch: &str) -> u32 {
    match arch {
        "qwen3" => 8,
        "llama" | "mistral" => 4,
        _ => 1,
    }
}

// Flagged: an `==` and a `!=` operand, either side.
fn soft_cap(arch: &str) -> bool {
    arch == "gemma" || "gemma2" == arch
}

fn plain(arch: &str) -> bool {
    arch != "granite"
}

// Flagged: an allowlist's members, used for membership.
const ALLOW: &[&str] = &["llama", "qwen2", "not-a-model"];

fn allowed(arch: &str) -> bool {
    ALLOW.contains(&arch)
}

static FAMILIES: [&str; 2] = ["phi3", "commandr"];

fn family(arch: &str) -> bool {
    FAMILIES.iter().any(|f| *f == arch)
}

// Flagged: an inline list, and a text match on an arch or stem value.
fn inline(arch: &str) -> bool {
    ["mixtral", "qwen3-moe"].contains(&arch)
}

fn stem_match(stem: &str) -> bool {
    stem.starts_with("smollm2-360m") || stem.contains("LlamaForCausalLM")
}

// Flagged: a `matches!` pattern, and an `if let` pattern.
fn moe(arch: &str) -> bool {
    matches!(arch, "glm4-moe" | "gpt-oss")
}

fn tied(arch: Option<&str>) -> bool {
    if let Some("arch-qwen3") = arch { true } else { false }
}

// Flagged: a local bound from an arch or stem value, and a config's model type.
struct Config {
    model_type: Option<String>,
}

fn lowered(arch: &str) -> bool {
    let a = arch.to_lowercase();
    a.contains("nemotron-h")
}

fn typed(cfg: &Config) -> bool {
    let base = cfg.model_type.as_deref().unwrap_or("");
    base == "qwen2-moe"
}

fn each(hf_arches: &[&str]) -> bool {
    hf_arches.iter().any(|a| *a == "modernbert")
}

// Clean: a checkpoint header's own field, whose values are spelled like a model (a tokenizer
// kind), decides on what the checkpoint carries.
fn tokenizer(header: &std::collections::HashMap<String, String>) -> u32 {
    let model = header.get("tokenizer.ggml.model").map(String::as_str);
    let pre = header.get("tokenizer.ggml.pre").map_or("default", String::as_str);
    match (model, pre) {
        (Some("llama"), _) => 1,
        (_, "qwen2") => 2,
        _ if model == Some("llama") => 3,
        _ => 0,
    }
}

// Clean: a parser registry keyed by the names a user passes on the command line.
fn parser(name: &str) -> u32 {
    match name {
        "granite" => 1,
        _ => 0,
    }
}

// Clean: a list never used for membership, names in a log string, names that are no model.
const LABELS: &[&str] = &["llama", "qwen3"];

fn log(arch: &str, rope: &str) -> bool {
    println!("loading llama, qwen3 or {arch}");
    eprintln!("{}", LABELS.len());
    let label = "gemma2";
    let _ = label.len();
    rope == "llama3" && arch != "default"
}

#[cfg(test)]
mod tests {
    // Clean: test code names its models.
    #[test]
    fn heads() {
        assert!(super::heads("qwen3") == 8);
        assert_eq!(match "llama" { "llama" => 1, _ => 0 }, 1);
    }
}

#[test]
fn a_test_fn() {
    assert!("qwen3" == "qwen3");
}

fn main() {
    let _ = (
        heads("x"),
        soft_cap("x"),
        plain("x"),
        allowed("x"),
        family("x"),
        inline("x"),
        stem_match("x"),
        moe("x"),
        tied(None),
        log("x", "y"),
        lowered("x"),
        typed(&Config { model_type: None }),
        each(&[]),
        tokenizer(&Default::default()),
        parser("x"),
    );
}

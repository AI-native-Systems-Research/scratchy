// SPDX-License-Identifier: Apache-2.0
//! The DSL carrier declares MATH ONLY; an arch's other facts live in
//! `configs/<arch>/arch.json`.
//!
//! These guard the boundary that replaced the `mod`-carrier const surface
//! (see `arch_spec`'s module docs). They are integration tests rather than
//! unit tests in `config.rs` because the reader is a private module — what
//! is checkable from outside is the carrier contract plus the shipped
//! declaration files, which is also what a future regression would break.

use std::path::{Path, PathBuf};

fn arch_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/arch")
}

/// Every key `load_arch_json` accepts. Duplicated deliberately: this test
/// fails if the reader gains a key without the suite acknowledging it, which
/// is the moment to decide whether the new key belongs in JSON at all.
const KNOWN_KEYS: &[&str] = &[
    "_comment",
    "vision_safetensors_layout",
    "vision_d_model_fingerprint",
    "vision_patch_embed_flatten",
    "vision_pos_embed_key",
    "vision_rope_style",
    "vision_pos_emb_interp",
    "vision_norm_eps",
    "decoder_safetensors_prefix",
    "scale_dtype",
    "tie_default",
    "bound_defaults",
    "scalar_defaults",
    "config_aliases",
    "weight_leaf_renames",
    "params",
];

fn each_arch_json(mut f: impl FnMut(&Path, &serde_json::Value)) {
    let configs = arch_dir().join("configs");
    let mut seen = 0;
    for e in std::fs::read_dir(&configs).expect("configs/").flatten() {
        let p = e.path().join("arch.json");
        if !p.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&p).unwrap();
        let json: serde_json::Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("{} is not valid JSON: {e}", p.display()));
        f(&p, &json);
        seen += 1;
    }
    assert!(seen > 0, "no arch.json found under {}", configs.display());
}

/// A typo'd key must not silently no-op — the property the token surface
/// enforced by erroring on unknown const NAMES, and the reason
/// `load_arch_json` rejects rather than ignores.
#[test]
fn shipped_arch_json_uses_only_known_keys() {
    each_arch_json(|path, json| {
        let obj = json
            .as_object()
            .unwrap_or_else(|| panic!("{}: top level must be an object", path.display()));
        for k in obj.keys() {
            assert!(
                KNOWN_KEYS.contains(&k.as_str()),
                "{}: unknown key `{k}` — load_arch_json would reject this",
                path.display()
            );
        }
    });
}

/// `params` is an ARRAY, not an object, because `expr` entries read bounds
/// that earlier entries defined and `eval_params` walks in declaration
/// order. A JSON object would not preserve it. Each entry carries exactly
/// one source.
#[test]
fn params_entries_are_ordered_and_single_source() {
    each_arch_json(|path, json| {
        let Some(params) = json.get("params") else {
            return;
        };
        let arr = params
            .as_array()
            .unwrap_or_else(|| panic!("{}: `params` must be an array", path.display()));
        let mut defined: Vec<String> = Vec::new();
        for (i, f) in arr.iter().enumerate() {
            let name = f
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| panic!("{}: params[{i}] has no `name`", path.display()));
            let sources = ["from", "expr", "value"]
                .iter()
                .filter(|k| f.get(**k).is_some())
                .count();
            assert_eq!(
                sources,
                1,
                "{}: params[{i}] ({name}) needs exactly one of from/expr/value",
                path.display()
            );
            // Order is load-bearing: an `expr` may only reference bounds
            // already defined ABOVE it. Bounds from the flat config harvest
            // are not known here, so only check names this file defines —
            // a forward reference to a sibling field is the real bug shape.
            if let Some(e) = f.get("expr").and_then(|v| v.as_str()) {
                for later in arr.iter().skip(i + 1) {
                    let ln = later.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let referenced = e
                        .split(|c: char| !c.is_alphanumeric() && c != '_')
                        .any(|tok| tok == ln);
                    assert!(
                        !referenced,
                        "{}: params[{i}] ({name}) expr `{e}` references `{ln}`, \
                         which is declared LATER — eval_params walks in order",
                        path.display()
                    );
                }
            }
            defined.push(name.to_string());
        }
        let mut dedup = defined.clone();
        dedup.sort();
        dedup.dedup();
        assert_eq!(
            dedup.len(),
            defined.len(),
            "{}: duplicate params names",
            path.display()
        );
    });
}

/// The carrier declares math only. A declaration const next to the `def`
/// is what the deleted const surface looked like, so it must fail — and say
/// where declarations go.
#[test]
fn declaration_beside_the_carrier_is_rejected_with_a_pointer_to_arch_json() {
    let src = "SCALE_DTYPE = ScaleDtype.Bf16

@forward
def qwen3():
    x = embed(input_ids, embed_tokens)
";
    // `PythonCarrier` is deliberately not Debug, so match rather than expect_err.
    let msg = match scratchy_forward_compiler_macro::parse_python_file(src) {
        Ok(_) => panic!("a declaration beside the carrier must be rejected"),
        Err(e) => e,
    };
    assert!(
        msg.contains("arch.json"),
        "error should point at configs/<arch>/arch.json, got: {msg}"
    );
}

/// The bare form is accepted and takes the arch name from the `def`.
#[test]
fn bare_def_carrier_is_accepted() {
    let carrier = scratchy_forward_compiler_macro::parse_python_file(
        "@forward
def llama():
    hidden_states = embed(input_ids, embed_tokens)
",
    );
    assert_eq!(
        carrier.map(|c| c.arch_name).as_deref(),
        Ok("llama"),
        "bare def carrier is THE form"
    );
}

/// No DSL file may reintroduce a `mod` carrier or declaration consts —
/// the regression this whole change exists to prevent.
#[test]
fn no_dsl_file_declares_anything_but_math() {
    let dsl = arch_dir().join("dsl");
    let mut seen = 0;
    for e in std::fs::read_dir(&dsl).expect("dsl/").flatten() {
        let p = e.path();
        if p.extension().is_none_or(|x| x != "py") {
            continue;
        }
        // The carrier parser admits only imports + one decorated `def` at
        // the top level, so a declaration const fails here.
        let text = std::fs::read_to_string(&p).unwrap();
        if let Err(e) = scratchy_forward_compiler_macro::parse_python_file(&text) {
            panic!(
                "{}: {e} — DSL files contain exactly one `@forward` def and nothing else; \
                 declarations belong in configs/<arch>/arch.json",
                p.display()
            );
        }
        seen += 1;
    }
    assert_eq!(seen, 25, "expected 25 DSL files, found {seen}");
}

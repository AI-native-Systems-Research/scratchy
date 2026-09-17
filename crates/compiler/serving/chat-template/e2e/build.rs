// SPDX-License-Identifier: Apache-2.0
//! Compile the vendored chat templates into $OUT_DIR.
//!
//! This is a stand-in for what `crates/models/arch/scratchy-forwards.rs` will
//! do per enabled model. Kept deliberately close to that shape so the harness
//! validates the real integration and not a convenient approximation.

use std::path::PathBuf;

/// (model stem, path to the vendored template relative to the repo root)
const TEMPLATES: &[(&str, &str)] = &[(
    "smollm2_135m",
    "crates/models/arch/configs/llama/smollm2-135m.chat.jinja",
)];

fn main() {
    // e2e -> chat-template -> serving -> compiler -> crates -> repo root
    let root: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(5)
        .expect("repo root")
        .to_path_buf();
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let mut modules = String::new();
    for (stem, rel) in TEMPLATES {
        let path = root.join(rel);
        println!("cargo:rerun-if-changed={}", path.display());
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

        // ⛔ AN UNSUPPORTED CONSTRUCT IS A BUILD FAILURE, NOT A FALLBACK. The
        // error names template, line, column and construct, so the build tells
        // you exactly what to implement next.
        let code = scratchy_chat_template_compiler::compile::compile(&src, rel, "render")
            .unwrap_or_else(|e| panic!("chat template did not compile:\n  {e}"));

        let file = out.join(format!("{stem}_chat.rs"));
        std::fs::write(&file, &code).unwrap();
        modules.push_str(&format!(
            "pub mod {stem} {{\n    use scratchy_chat_template_compiler::*;\n    include!(concat!(env!(\"OUT_DIR\"), \"/{stem}_chat.rs\"));\n}}\n"
        ));
    }
    std::fs::write(out.join("templates.rs"), modules).unwrap();
}

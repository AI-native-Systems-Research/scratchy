// SPDX-License-Identifier: Apache-2.0
//! Real-config fixtures: an arch's checked-in configs and DSL, one synthesized `(stem, preset)`
//! variant, and the program and fused graph the real passes build from them. Shared by this
//! crate's tests and a target compiler's (`test-support`), so a target's emission is tested on
//! the same models the pipeline is.

use std::path::PathBuf;

use crate::classified::Program;
use crate::config::ModelParams;
use crate::fuf::Fuf;

/// Per-arch config dir in the `crates/models/arch/configs/<arch>/` layout. Pass the arch slug
/// using hyphens, e.g. `"deepseek-v3"`, `"qwen3"`.
pub fn arch_configs(arch: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/arch")
        .join("configs")
        .join(arch)
}

/// Load ONE checked-in config by stem.
pub fn config_of(arch: &str, stem: &str) -> ModelParams {
    let path = arch_configs(arch).join(format!("{stem}.json"));
    crate::config::load_file(&path).unwrap_or_else(|e| panic!("load {}: {e}", path.display()))
}

/// Load ONE synthesized `(stem, preset)` quant variant — the same deep merge `load_dir`'s overlay
/// loop performs, for one pair — under the load-OR-refuse law: a variant whose merged config
/// ships a mixed-bit affine quantization refuses under a backend that can't lower it (asserted by
/// name inside the helper) and yields `None` — the caller skips it; any other error still fails
/// loudly.
pub fn variant_of(arch: &str, stem: &str, preset: &str) -> Option<ModelParams> {
    let config = arch_configs(arch).join(format!("{stem}.json"));
    let preset_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/quantization/presets")
        .join(format!("{preset}.json"));
    crate::config::assert_load_or_refuse_by_name(
        // The synthesized variant's own stem, as the merge names it.
        &format!("{stem}-{preset}"),
        crate::config::load_file_with_preset(&config, &preset_path),
    )
}

/// An arch's DSL carrier, parsed.
pub fn dsl_of(arch: &str) -> crate::ast::Ast {
    let dsl = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/arch/dsl")
        .join(format!("{arch}.py"));
    crate::parse_python::parse_python_file(&std::fs::read_to_string(dsl).unwrap())
        .expect("carrier parses")
        .ast
}

/// The program and fused graph of `model` under its arch's DSL, with the compiler's drafter
/// capture inserted as the build inserts it.
pub fn fuf_of(arch: &str, model: &ModelParams) -> (Program, Fuf) {
    fuf_of_ast(arch, model, &crate::draft_capture::insert(&dsl_of(arch)))
}

/// The program and fused graph of `model` under `ast`.
pub fn fuf_of_ast(arch: &str, model: &ModelParams, ast: &crate::ast::Ast) -> (Program, Fuf) {
    let mut program =
        crate::classify::classify_with(ast, crate::classified::Prelude::Decoder).expect("classify");
    program.weight_leaf_renames = model.arch.weight_leaf_renames.clone();
    let inferred = crate::typed_bodies::trace_carrier_program(arch, &program);
    let cfg = crate::cfg::build_cfg(&program, model).expect("cfg");
    let mut fuf = crate::fuf::unroll(&cfg, &inferred).expect("unroll");
    fuf.annotate_storage_formats(&program, model);
    (program, fuf)
}

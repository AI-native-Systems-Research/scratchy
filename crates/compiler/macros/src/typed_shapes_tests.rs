// SPDX-License-Identifier: Apache-2.0
//! The typed-Rust emit pass and its signatures, against the carriers and configs on disk: the
//! checks that name an arch, kept out of the two modules that must name none.

/// Every classifier local of the typed llama but its loop variable has a traced shape, and every
/// config's program unrolls from them.
#[test]
fn the_typed_llama_shapes_every_local_and_unrolls_on_every_config() {
    let (program, shaped) = typed_carrier_unrolls("llama", 13);
    assert_eq!(shaped, program.locals.ids().count() - 1);
}

/// Gemma 4 (per-class attention, its per-head norms as explicit reshapes) unrolls from its trace
/// on every config.
#[test]
fn the_typed_gemma4_unrolls_on_every_config() {
    typed_carrier_unrolls("gemma4", 4);
}

/// The typed carrier `stem` (`configs` of them) as the bake lowers it: its trace shapes every local
/// the classifier binds but a loop variable, and every config's program (its identity regroupings
/// erased) unrolls from that trace, every nested value found. The classified program and how many
/// locals the trace shapes.
fn typed_carrier_unrolls(stem: &str, configs: usize) -> (crate::classified::Program, usize) {
    let arch = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/arch");
    let source = std::fs::read_to_string(arch.join(format!("dsl/{stem}.py"))).expect("carrier");
    let carrier = crate::parse_python::parse_python_file(&source).expect("parse");
    // The program the bake lowers.
    let program = crate::classify::carrier_program(&carrier).unwrap();
    let inferred = crate::typed_shapes::traced(&carrier.arch_name)
        .unwrap_or_else(|| panic!("{stem} has a typed program"));
    assert!(
        inferred.locals_of(&local_names(&program)),
        "{stem}: the trace is of the program the bake lowers"
    );
    let loop_vars = loop_vars(&program.statements);
    let unshaped: Vec<String> = (program.locals.ids())
        .filter(|id| !inferred.locals.contains_key(id) && !loop_vars.contains(id))
        .map(|id| program.locals.name(id).to_string())
        .collect();
    assert!(
        unshaped.is_empty(),
        "{stem}: no traced shape for {unshaped:?}"
    );
    let models = carrier_models(&arch, stem, carrier.vision);
    assert_eq!(models.len(), configs, "{stem}'s configs");
    for (cfg, _, model) in models {
        let model = model.unwrap_or_else(|e| panic!("{cfg}: {e:?}"));
        let mut prog = program.clone();
        if !carrier.vision {
            prog.decoder_safetensors_prefix = model.arch.decoder_prefix.clone();
        }
        prog.weight_leaf_renames = model.arch.weight_leaf_renames.clone();
        crate::shape::erase_identity_regroupings(&mut prog, &inferred, &model.bounds);
        let cfg_ = crate::cfg::build_cfg(&prog, &model).unwrap_or_else(|e| panic!("{cfg}: {e}"));
        let fuf = crate::fuf::unroll(&cfg_, &inferred).unwrap_or_else(|e| panic!("{cfg}: {e}"));
        assert!(!fuf.nodes.is_empty(), "{cfg}: nothing unrolled");
    }
    (program, inferred.locals.len())
}

/// Each of `program`'s locals' names, by `LocalId`.
fn local_names(program: &crate::classified::Program) -> Vec<String> {
    (program.locals.ids())
        .map(|id| program.locals.name(id).to_string())
        .collect()
}

/// The trace knows the program it is of: llama's is of the program the bake lowers (the drafter
/// capture inserted), and not of the program its DSL alone classifies to, whose locals it would
/// misname.
#[test]
fn a_trace_is_of_its_own_program_only() {
    let arch = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/arch");
    let source = std::fs::read_to_string(arch.join("dsl/llama.py")).expect("llama.py");
    let carrier = crate::parse_python::parse_python_file(&source).expect("parse");
    let inferred = crate::typed_shapes::traced(&carrier.arch_name).expect("typed");
    let lowered = crate::classify::carrier_program(&carrier).unwrap();
    assert!(inferred.locals_of(&local_names(&lowered)));
    let written = crate::classify::classify(&carrier.ast).unwrap();
    assert_ne!(local_names(&written), local_names(&lowered));
    assert!(!inferred.locals_of(&local_names(&written)));
}

/// Every loop variable the program binds.
fn loop_vars(
    stmts: &[crate::classified::Stmt],
) -> std::collections::HashSet<crate::classified::LocalId> {
    use crate::classified::Stmt;
    let mut out = std::collections::HashSet::new();
    for s in stmts {
        match s {
            Stmt::For { ivar, body, .. } => {
                out.insert(*ivar);
                out.extend(loop_vars(body));
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                out.extend(loop_vars(then_body));
                out.extend(loop_vars(else_body));
            }
            _ => {}
        }
    }
    out
}

/// Every name a carrier or a model config goes by: the `dsl/` file stems and the config stems.
fn arch_and_model_names(arch: &std::path::Path) -> Vec<String> {
    let stems = |dir: std::path::PathBuf, ext: &str| -> Vec<String> {
        (std::fs::read_dir(&dir).expect("dir").flatten())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some(ext))
            .filter_map(|p| p.file_stem().and_then(|s| s.to_str()).map(str::to_string))
            .collect()
    };
    let mut names = stems(arch.join("dsl"), "py");
    for dir in (std::fs::read_dir(arch.join("configs"))
        .expect("configs")
        .flatten())
    .map(|e| e.path())
    .filter(|p| p.is_dir())
    {
        names.extend(
            stems(dir, "json")
                .into_iter()
                .filter(|s| !matches!(s.as_str(), "weights" | "quantizations" | "arch")),
        );
    }
    names
}

/// The emit pass and the signature module are a compiler's: neither names an arch or a model, so
/// nothing in them is one arch's. Each name is matched as a whole word.
#[test]
fn the_emit_pass_and_its_signatures_name_no_arch_or_model() {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let arch = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/arch");
    let names = arch_and_model_names(&arch);
    assert!(
        names.iter().any(|n| n == "llama"),
        "the scan sees the carriers"
    );
    for file in ["../../frontend/src/typed_emit.rs", "typed_shapes.rs"] {
        let text = std::fs::read_to_string(src.join(file)).expect(file);
        let words: std::collections::HashSet<&str> = text
            .split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.')))
            .collect();
        let named: Vec<&String> = names
            .iter()
            .filter(|n| words.contains(n.as_str()))
            .collect();
        assert!(named.is_empty(), "{file} names {named:?}");
    }
}

/// The reshapes a program still carries, nested blocks included.
fn reshapes(stmts: &[crate::classified::Stmt]) -> usize {
    use crate::classified::{Expr, OpKind, Stmt};
    stmts
        .iter()
        .map(|s| match s {
            Stmt::Assign {
                value: Expr::Call { op, .. },
                ..
            } if *op == OpKind::Reshape => 1,
            Stmt::For { body, .. } => reshapes(body),
            Stmt::If {
                then_body,
                else_body,
                ..
            } => reshapes(then_body) + reshapes(else_body),
            _ => 0,
        })
        .sum()
}

/// Gemma 4's program after `shape::erase_identity_regroupings` for one of its configs.
fn gemma4_reshapes_left(config: &str) -> (usize, usize) {
    let arch = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/arch");
    let source = std::fs::read_to_string(arch.join("dsl/gemma4.py")).expect("gemma4.py");
    let carrier = crate::parse_python::parse_python_file(&source).expect("parse");
    let mut program = crate::classify::carrier_program(&carrier).unwrap();
    let configs_dir = arch.join("configs/gemma4");
    let model = crate::config::load_file_declared(&configs_dir.join(config)).unwrap();
    let inferred = crate::typed_shapes::traced(&carrier.arch_name).expect("typed");
    let before = reshapes(&program.statements);
    crate::shape::erase_identity_regroupings(&mut program, &inferred, &model.bounds);
    (before, reshapes(&program.statements))
}

/// A regrouping by a factor of 1 is the identity for that model and erased: Gemma 4 12B's single
/// global KV head makes its global K split, its K merge and its V merge identities (3 of the 9).
#[test]
fn a_split_by_one_and_its_merge_are_erased() {
    assert_eq!(gemma4_reshapes_left("gemma-4-12b-it.json"), (9, 6));
}

/// A real split (factor above 1) is kept: Gemma 4 31B's four global KV heads keep every regrouping.
#[test]
fn a_real_split_is_kept() {
    assert_eq!(gemma4_reshapes_left("gemma-4-31b-it.json"), (9, 9));
}

/// gemma4-moe (Gemma 4's attention plus the fused sparse block) unrolls from its trace.
#[test]
fn the_typed_gemma4_moe_unrolls_on_every_config() {
    typed_carrier_unrolls("gemma4-moe", 6);
}

/// gemma4-assistant (concat, the shared attentions, the target's rows) unrolls from its trace.
#[test]
fn the_typed_gemma4_assistant_unrolls_on_every_config() {
    typed_carrier_unrolls("gemma4-assistant", 2);
}

/// Every model config of the carrier `stem` (`vision` for a `@vision_forward` one), as the config
/// loader loads it, with its file: its own configs, or, for a head that ships inside its targets'
/// checkpoints, each target config that carries it.
fn carrier_models(
    arch: &std::path::Path,
    stem: &str,
    vision: bool,
) -> Vec<(
    String,
    std::path::PathBuf,
    Result<crate::config::ModelParams, crate::config::ConfigError>,
)> {
    let model_configs = |dir: std::path::PathBuf| {
        let mut paths: Vec<_> = (std::fs::read_dir(dir).into_iter().flatten().flatten())
            .map(|e| e.path())
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                name.ends_with(".json")
                    && !name.ends_with(".overrides.json")
                    && !matches!(name, "weights.json" | "quantizations.json" | "arch.json")
            })
            .collect();
        paths.sort();
        paths
    };
    let configs_dir = arch.join("configs").join(stem);
    let label = |p: &std::path::Path| p.file_stem().unwrap().to_str().unwrap().to_string();
    let own = model_configs(configs_dir.clone());
    if !own.is_empty() {
        return (own.into_iter())
            .map(|p| {
                let model = match vision {
                    true => crate::config::load_file_declared_vision(&p),
                    false => crate::config::load_file_declared(&p),
                };
                (label(&p), p, model)
            })
            .collect();
    }
    let Ok(spec) = crate::config::load_arch_json(&configs_dir) else {
        return Vec::new();
    };
    let mut models = Vec::new();
    for target in &spec.checkpoint_of {
        for p in model_configs(arch.join("configs").join(target)) {
            let json: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
            if crate::config::carries_head(&spec, &json) {
                let model = crate::config::load_head_file(&p, &configs_dir);
                models.push((format!("{target}/{}", label(&p)), p, model));
            }
        }
    }
    models
}

/// Every carrier under `dsl/`: its file stem and whether it is a `@vision_forward` one.
fn carriers(arch: &std::path::Path) -> Vec<(String, bool)> {
    let mut paths: Vec<_> = (std::fs::read_dir(arch.join("dsl")).unwrap().flatten())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("py"))
        .collect();
    paths.sort();
    (paths.iter())
        .map(|p| {
            let carrier =
                crate::parse_python::parse_python_file(&std::fs::read_to_string(p).unwrap())
                    .unwrap();
            (
                p.file_stem().unwrap().to_str().unwrap().to_string(),
                carrier.vision,
            )
        })
        .collect()
}

/// Every config's bounds as the config loader derives them, one `carrier/config name=value` line
/// each: a snapshot to compare across a change to how bounds are derived. A report, not run by
/// default: `cargo test -p scratchy-forward-compiler-macro --features metal --lib bounds_snapshot
/// -- --ignored --nocapture`.
#[test]
#[ignore]
fn bounds_snapshot() {
    let arch = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/arch");
    for (stem, vision) in carriers(&arch) {
        for (cfg, _, model) in carrier_models(&arch, &stem, vision) {
            match model {
                Ok(m) => {
                    for (k, v) in &m.bounds {
                        println!("{stem}/{cfg} {k}={v}");
                    }
                }
                Err(e) => println!("{stem}/{cfg} ERROR {e:?}"),
            }
        }
    }
}

/// A gated derived dim's flag is one rule (`hf_config::flag_value`): for every config of every
/// carrier, the flag the macros crate's build script resolves from the raw config and its arch.json
/// is the one the config loader derived, and so is every gated dim's value.
#[test]
fn a_gated_flag_resolves_as_the_config_loader_derives_it() {
    let arch = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/arch");
    let mut checked = 0;
    for (stem, vision) in carriers(&arch) {
        let arch_json = arch.join("configs").join(&stem).join("arch.json");
        let facts = match arch_json.exists() {
            true => crate::hf_config::arch_facts(
                &serde_json::from_str(&std::fs::read_to_string(&arch_json).unwrap()).unwrap(),
            )
            .unwrap(),
            false => crate::hf_config::ArchFacts::default(),
        };
        for (cfg, path, model) in carrier_models(&arch, &stem, vision) {
            let model = model.unwrap_or_else(|e| panic!("{stem}/{cfg}: {e:?}"));
            let raw: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            for d in crate::dim::DERIVED_DIMS.iter() {
                let Some(flag) = d.rule.gate_flag() else {
                    continue;
                };
                // The loader reads a gated flag on the decoder path only: a vision tower derives no
                // gated dim, so there is nothing to agree on (and the build script resolves none).
                if vision {
                    assert!(
                        !model.bounds.contains_key(d.name),
                        "{stem}/{cfg}: `{}`",
                        d.name
                    );
                    continue;
                }
                let resolved = crate::hf_config::flag_value(
                    &raw,
                    &facts.config_aliases,
                    &facts.bound_defaults,
                    flag,
                );
                assert_eq!(
                    resolved,
                    model.bounds.get(flag).copied().unwrap_or(0),
                    "{stem}/{cfg}: `{flag}`"
                );
                if let Some(&loaded) = model.bounds.get(d.name) {
                    let mut bounds = model.bounds.clone();
                    bounds.insert(flag.to_string(), resolved);
                    assert_eq!(
                        d.rule.eval(&bounds),
                        Some(loaded),
                        "{stem}/{cfg}: `{}`",
                        d.name
                    );
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 0);
}

/// Every derived dim a typed carrier's module spells as a type equals, under each of its configs'
/// bounds, the value the config loader derived for it: the types and the loader read one table
/// (and one arch.json grammar).
#[test]
fn every_derived_alias_equals_the_config_loaders_value() {
    let arch = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/arch");
    let mut checked = 0;
    for (stem, vision) in carriers(&arch) {
        let source = std::fs::read_to_string(arch.join("dsl").join(format!("{stem}.py"))).unwrap();
        let carrier = crate::parse_python::parse_python_file(&source).unwrap();
        let Some(derived) = crate::typed_shapes::derived_dims_of(&carrier.arch_name) else {
            continue;
        };
        for (cfg, _, model) in carrier_models(&arch, &stem, vision) {
            let model = model.unwrap_or_else(|e| panic!("{stem}/{cfg}: {e:?}"));
            for (name, dim) in &derived {
                assert_eq!(
                    crate::shape::eval_closed_dim(dim, &model.bounds),
                    model.bounds.get(*name).copied(),
                    "{stem}/{cfg}: `{name}`"
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "no typed carrier names a derived dim");
}

/// The qwen3.5 MTP heads (the gate, the shared expert, and the target's rows at the target's
/// width) unroll from their traces on their targets' configs that carry them.
#[test]
fn the_typed_qwen3_5_mtp_heads_unroll_on_every_config() {
    typed_carrier_unrolls("qwen3-5-mtp", 4);
    typed_carrier_unrolls("qwen3-5-dense-mtp", 5);
}

/// qwen2-vl's vision tower (the 2-D rope, per-image attention, and the merge's structural
/// regrouping, `group_rows`) unrolls from its trace on every vision config.
#[test]
fn the_typed_qwen2_vl_unrolls_on_every_config() {
    typed_carrier_unrolls("qwen2-vl", 5);
}

/// Every carrier under `dsl/` (the build script's scan, so a new carrier is covered unnamed) is
/// typed, and unrolls from its trace on every config it compiles for.
#[test]
fn every_typed_carrier_unrolls_on_every_config() {
    let arch = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/arch");
    let carriers = carriers(&arch);
    assert_eq!(carriers.len(), 34, "the carriers under dsl/");
    let mut configs = 0;
    let mut failed = Vec::new();
    for (stem, vision) in &carriers {
        let n = carrier_models(&arch, stem, *vision).len();
        assert!(n > 0, "{stem}: no config compiles it");
        // Every carrier's verdict, not just the first failure's.
        if std::panic::catch_unwind(|| typed_carrier_unrolls(stem, n)).is_err() {
            failed.push(stem.clone());
        }
        configs += n;
    }
    assert!(failed.is_empty(), "unrolling fails on {failed:?}");
    eprintln!("{} carriers, {configs} configs", carriers.len());
}

/// The qwen3.5 MoE MTP head's program, its trace, and one target config that carries it.
fn mtp_head_on_its_target() -> (
    crate::classified::Program,
    crate::shape::Inferred,
    crate::config::ModelParams,
) {
    let arch = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/arch");
    let source = std::fs::read_to_string(arch.join("dsl/qwen3-5-mtp.py")).expect("carrier");
    let carrier = crate::parse_python::parse_python_file(&source).expect("parse");
    let program = crate::classify::carrier_program(&carrier).unwrap();
    let (_, _, model) = carrier_models(&arch, "qwen3-5-mtp", false)
        .into_iter()
        .next()
        .expect("a target config carries the head");
    let model = model.unwrap();
    let inferred = crate::typed_shapes::traced(&carrier.arch_name).expect("typed");
    (program, inferred, model)
}

/// The head's facts, as the fill states them for its bake.
fn head_facts(
    program: &crate::classified::Program,
    inferred: &crate::shape::Inferred,
    model: &crate::config::ModelParams,
) -> scratchy_ir::model_facts::ModelFacts {
    let width = crate::codegen::target_width(program, inferred);
    crate::codegen::fill_canonical_params(model, 1, false, &Default::default(), Some(width)).0
}

/// An in-checkpoint head reads its target's lent lm_head through `as_dim`: its own width stated as
/// its target's, which its facts state as a lent target (`Target::lent`), and the equal widths of
/// every config it ships in hold `ModelFacts`' law.
#[test]
fn an_as_dim_whose_widths_agree_compiles() {
    let (program, inferred, model) = mtp_head_on_its_target();
    assert_eq!(program.dim_casts.len(), 1, "the head's one lend");
    let facts = head_facts(&program, &inferred, &model);
    let target = facts.target().expect("a head states its target");
    assert_eq!(target.width(), scratchy_ir::model_facts::TargetWidth::Lent);
    assert_eq!(target.hidden(), facts.hidden());
}

/// A model whose two widths differ breaks `ModelFacts`' law: its facts are refused (in the bake,
/// at the const evaluation of its `FACTS`).
#[test]
fn an_as_dim_whose_widths_differ_is_refused() {
    let (program, inferred, mut model) = mtp_head_on_its_target();
    let hidden = model.bounds["hidden_size"];
    model
        .bounds
        .insert("backbone_hidden_size".to_string(), hidden + 1);
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        head_facts(&program, &inferred, &model)
    }))
    .expect_err("the law refuses the facts");
    let why = (refused.downcast_ref::<&str>().map(|s| s.to_string()))
        .or_else(|| refused.downcast_ref::<String>().cloned())
        .unwrap_or_default();
    assert!(
        why.contains("a head lent its target's width is that width"),
        "{why}"
    );
}

/// Llama's carrier (its text edited by `edit`) and its `weights.json` (edited by `declare`), through
/// the emit pass and rustc as the build script and this crate's compile take it: the trace, or
/// rustc's errors.
fn typed_llama(
    edit: impl Fn(&str) -> String,
    declare: impl Fn(&mut crate::weights_manifest::WeightsManifest),
) -> Result<crate::shape::Inferred, String> {
    let arch = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/arch");
    let written = std::fs::read_to_string(arch.join("dsl/llama.py")).expect("llama.py");
    let source = edit(&written);
    let carrier = crate::parse_python::parse_python_file(&source).expect("parse");
    let program = crate::classify::carrier_program(&carrier).unwrap();
    let mut manifest =
        crate::weights_manifest::load_or_empty(&arch.join("configs/llama")).expect("manifest");
    declare(&mut manifest);
    let facts = scratchy_forward_compiler_frontend::carrier_facts::carrier_facts(
        &arch,
        "llama",
        false,
        &mut Vec::new(),
    );
    crate::typed_bodies::typed(
        &program,
        &source,
        "crates/models/arch/dsl/llama.py",
        &manifest,
        &facts,
    )
}

/// The rustc errors of a program that must not compile, with what each must name.
fn refused(typed: Result<crate::shape::Inferred, String>, names: &[&str]) {
    let errors = typed.expect_err("the typed compile refuses it");
    assert!(
        errors.contains("error[E0308]: mismatched types"),
        "{errors}"
    );
    for name in names {
        assert!(errors.contains(name), "the error names `{name}`:\n{errors}");
    }
}

/// The types reject a wrong carrier: `q_proj` written where `k_proj` belongs (`llama.py:15`) fails
/// the typed compile, and rustc names the carrier's line where the two widths meet (the attention,
/// `llama.py:18`, the first that reads `k`) and both widths. Llama as written compiles through the
/// same path, so the refusal is the edit's.
#[test]
fn a_wrong_dsl_line_fails_the_typed_compile_and_names_the_line() {
    let k_proj = "k = gemm(normed, self_attn.k_proj[layer])";
    assert!(
        typed_llama(str::to_string, |_| {}).is_ok(),
        "llama as written types"
    );
    refused(
        typed_llama(
            |s| {
                assert!(s.contains(k_proj));
                s.replace(k_proj, "k = gemm(normed, self_attn.q_proj[layer])")
            },
            |_| {},
        ),
        &[
            "crates/models/arch/dsl/llama.py:18: attn = attention(",
            "NumKeyValueHeads",
            "NumAttentionHeads",
        ],
    );
}

/// The types reject a wrong `weights.json`: `mlp.down_proj` declared `[hidden_size, hidden_size]`
/// (its rows are the MLP's width) fails the typed compile at the carrier's line that reads it
/// (`llama.py:25`), naming both widths.
#[test]
fn a_misdeclared_weight_width_fails_the_typed_compile_and_names_the_line() {
    use crate::dim::Dim;
    let hidden = Dim::Bound("hidden_size".to_string());
    refused(
        typed_llama(str::to_string, |m| {
            let down = m.entries.get_mut("mlp.down_proj").expect("declared");
            assert_eq!(down[0], Dim::Bound("intermediate_size".to_string()));
            *down = vec![hidden.clone(), hidden.clone()];
        }),
        &[
            "crates/models/arch/dsl/llama.py:25: down = gemm(gate * up, mlp.down_proj[layer])",
            "IntermediateSize",
            "HiddenSize",
        ],
    );
}

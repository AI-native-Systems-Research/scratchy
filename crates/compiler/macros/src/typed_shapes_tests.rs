// SPDX-License-Identifier: Apache-2.0
//! The typed-Rust emit pass and its signatures, against the carriers and configs on disk: the
//! checks that name an arch, kept out of the two modules that must name none.

use crate::typed_shapes::check_parity;

/// Every model config of a carrier's arch dir, each loaded with its arch's declarations.
fn arch_configs(arch_dir: &std::path::Path) -> Vec<(String, crate::config::ModelParams)> {
    let mut paths: Vec<_> = (std::fs::read_dir(arch_dir).expect("arch dir").flatten())
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name.ends_with(".json")
                && !name.ends_with(".overrides.json")
                && !matches!(name, "weights.json" | "quantizations.json" | "arch.json")
        })
        .collect();
    paths.sort();
    (paths.iter())
        .map(|p| {
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap().to_string();
            let model = crate::config::load_file_declared(p)
                .unwrap_or_else(|e| panic!("{}: {e:?}", p.display()));
            (stem, model)
        })
        .collect()
}

/// Phase A's gate 1, on every config of the arch the emit pass typed (the bake checks it on
/// the configs a build selects; this, on all of them): the classifier's every local has the
/// shape `shape::infer` closes for that config, and reshape recovery adds none.
#[test]
fn the_typed_llama_traces_what_shape_infer_closes_for_every_config() {
    let arch = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/arch");
    let source = std::fs::read_to_string(arch.join("dsl/llama.py")).expect("llama.py");
    let carrier = crate::parse_python::parse_python_file(&source).expect("parse");
    let ast = crate::draft_capture::insert(&carrier.ast);
    let program =
        crate::classify::classify_with(&ast, crate::classified::Prelude::Decoder).unwrap();
    let configs_dir = arch.join("configs/llama");
    let manifest = crate::weights_manifest::load_or_empty(&configs_dir).expect("manifest");
    let configs = arch_configs(&configs_dir);
    assert_eq!(configs.len(), 13, "llama's configs");
    for (stem, model) in configs {
        let mut prog = program.clone();
        prog.decoder_safetensors_prefix = model.arch.decoder_prefix.clone();
        prog.weight_leaf_renames = model.arch.weight_leaf_renames.clone();
        let inferred = crate::shape::infer(&prog, &manifest, &model.bounds)
            .unwrap_or_else(|e| panic!("{stem}: shape::infer, reshape recovery included: {e}"));
        let parity = check_parity(
            &carrier.arch_name,
            &program,
            &inferred.locals,
            &model.bounds,
        )
        .expect("llama has a typed program");
        assert!(
            parity.mismatches.is_empty(),
            "{stem}: {:#?}",
            parity.mismatches
        );
        assert!(
            parity.recovered.is_empty(),
            "{stem}: {:?}",
            parity.recovered
        );
        // Every local the classifier bound but the loop variable is shaped by both sides.
        assert_eq!(parity.compared, program.locals.ids().count() - 1, "{stem}");
        eprintln!(
            "{stem}: {} locals, 0 reshape-recovery hints",
            parity.compared
        );
    }
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
    for file in ["typed_emit.rs", "typed_shapes.rs"] {
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

/// The parity check sees a difference: one local's inferred shape changed is a mismatch naming it.
#[test]
fn a_changed_shape_is_a_parity_mismatch() {
    let arch = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/arch");
    let source = std::fs::read_to_string(arch.join("dsl/llama.py")).expect("llama.py");
    let carrier = crate::parse_python::parse_python_file(&source).expect("parse");
    let ast = crate::draft_capture::insert(&carrier.ast);
    let program =
        crate::classify::classify_with(&ast, crate::classified::Prelude::Decoder).unwrap();
    let configs_dir = arch.join("configs/llama");
    let manifest = crate::weights_manifest::load_or_empty(&configs_dir).expect("manifest");
    let model = crate::config::load_file_declared(&configs_dir.join("llama-3.2-1b.json")).unwrap();
    let mut inferred = crate::shape::infer(&program, &manifest, &model.bounds)
        .unwrap()
        .locals;
    // The first rank-2 local's features become 1, which no feature width of the model is.
    let changed = *(inferred.iter())
        .filter(|(_, s)| s.len() == 2)
        .map(|(id, _)| id)
        .min_by_key(|id| id.0)
        .expect("a rank-2 local");
    inferred.get_mut(&changed).unwrap()[1] = crate::dim::Dim::Lit(1);
    let parity = check_parity(&carrier.arch_name, &program, &inferred, &model.bounds).unwrap();
    let mismatched: Vec<_> = parity.mismatches.iter().map(|m| m.local).collect();
    assert_eq!(mismatched, vec![changed]);
}

// SPDX-License-Identifier: Apache-2.0
//! The shared pipeline driven through metal's emitter, on checked-in models.

use scratchy_forward_compiler_macro::Scope;
use scratchy_forward_compiler_macro::config::{load_arch_json, load_dir};
use scratchy_target_metal_compiler::Metal;
use std::path::PathBuf;

/// The repo root, from this crate's manifest dir.
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../..")
        .canonicalize()
        .unwrap()
}

/// `qwen3.py` drives the full parse → classify → shape → CFG → unroll → schedule → codegen chain
/// to a NON-EMPTY, complete metal emission, without pinning bytes. Numerical equivalence of the
/// emission is covered by the metal models crate's `qwen3_py_parity` e2e gate. The scope names
/// the one model the test means — the emission cannot be empty by accident, so no non-vacuity
/// guard is needed (nor an env var hand-set into the invocation).
#[test]
fn python_costume_emits_the_qwen3_tape() {
    let root = root();
    let arch = "qwen3";
    let configs_dir = root.join("crates/models/arch/configs").join(arch);

    let py_text = std::fs::read_to_string(root.join("crates/models/arch/dsl/qwen3.py")).unwrap();
    let py_carrier =
        scratchy_forward_compiler_macro::parse_python_file(&py_text).expect("python parse");
    let py_tokens = scratchy_forward_compiler_macro::compile_carrier(
        py_carrier,
        &configs_dir,
        None,
        &Scope::stems(&[("qwen3", "qwen3-0.6b")]),
        &Metal { turboquant: false },
    )
    .expect("python-costume pipeline");

    let mut py_rendered = String::with_capacity(1 << 20);
    scratchy_forward_compiler_macro::render_tokens(py_tokens, &mut py_rendered);

    // The emission must be COMPLETE, not just big: the flip pinned per-bucket tape symbols + the
    // load/forward entry points. Check the shape, not the bytes — byte-pinning every codegen
    // change would force regenerating a ~67k-line fixture with no extra correctness signal over
    // the numeric e2e gate.
    for marker in [
        "FORWARD_TABLE",
        "fn load",
        "fn forward",
        "q_proj",
        "rope",
        "lm_head",
    ] {
        assert!(
            py_rendered.contains(marker),
            "qwen3 emission lacks `{marker}` — the pipeline emitted an incomplete tape"
        );
    }
}

/// Metal lowers the MLX affine presets and NVFP4 alone: a build of every config of an arch with
/// every preset forwarded loads each base's variants exactly `<base>-<preset>` for those presets —
/// a selected preset replaces the dense variant, it doesn't add to it.
#[test]
fn metal_lowers_the_mlx_affine_and_nvfp4_presets_only() {
    let metal = Metal { turboquant: false };
    for (arch, presets) in [
        (
            "llama",
            &["mlx-affine-b4-g64", "mlx-affine-b4-g64-qembed", "nvfp4"][..],
        ),
        (
            "qwen2",
            &["mlx-affine-b4-g64", "mlx-affine-b4-g64-qembed"][..],
        ),
    ] {
        let dir = root().join("crates/models/arch/configs").join(arch);
        let spec = load_arch_json(&dir).unwrap_or_else(|e| panic!("{arch} arch.json: {e}"));
        let dense = load_dir(
            &dir,
            &spec,
            &Scope::all().with_quants(Vec::<String>::new()),
            &metal,
        )
        .unwrap_or_else(|e| panic!("load {arch} dense: {e}"));
        let quantized = load_dir(&dir, &spec, &Scope::all(), &metal)
            .unwrap_or_else(|e| panic!("load {arch} with every preset: {e}"));
        let mut want: Vec<String> = (dense.iter())
            .flat_map(|b| {
                presets
                    .iter()
                    .map(move |p| format!("{}-{p}", b.source_stem))
            })
            .collect();
        want.sort();
        let mut got: Vec<String> = quantized.iter().map(|c| c.source_stem.clone()).collect();
        got.sort();
        assert_eq!(got, want, "{arch}: every base × every preset metal lowers");
    }
}

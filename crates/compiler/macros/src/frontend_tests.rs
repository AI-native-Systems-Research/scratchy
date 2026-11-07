// SPDX-License-Identifier: Apache-2.0
//! Tests of the front end (`scratchy-forward-compiler-frontend`) that run it through this crate's
//! passes, which the front end cannot name.

mod draft_capture {
    use crate::ast::{Ast, Stmt};
    use crate::cfg::build_cfg;
    use crate::classified::OpKind;
    use crate::config::{self, LayerSet, ModelParams};
    use crate::draft_capture::{DRAFT_CAPTURE_LAYERS, insert};
    use crate::fuf::{Fuf, FufInput, unroll};
    use crate::parse_python::parse_body;
    use std::path::PathBuf;

    /// A decoder: an embedding, a layer loop carrying `hidden_states`, the final norm and head.
    const DECODER: &str = "
        hidden_states = embed(input_ids, embed_tokens)
        for layer in range(num_hidden_layers):
            normed = rmsnorm(hidden_states, input_layernorm[layer])
            hidden_states = add(normed, hidden_states)
        normed = rmsnorm(hidden_states, norm)
        logits = gemm(normed, lm_head)
        ";

    fn llama(captured: &[u64]) -> ModelParams {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/arch/configs/llama/llama-3.2-1b.json");
        let mut params = config::load_file(&path).expect("load config");
        let set = LayerSet::of(captured.iter().copied());
        params
            .layer_sets
            .insert(DRAFT_CAPTURE_LAYERS.to_string(), set);
        params
    }

    fn unroll_ast(ast: &Ast, params: &ModelParams) -> Fuf {
        let manifest = crate::weights_manifest::WeightsManifest::llama_test_conventions();
        let (program, inferred) = crate::typed_bodies::trace_ast(ast, DECODER, &manifest);
        let cfg = build_cfg(&program, params).expect("build cfg");
        unroll(&cfg, &inferred).expect("unroll")
    }

    fn ops(fuf: &Fuf) -> Vec<OpKind> {
        fuf.nodes.iter().map(|n| n.op).collect()
    }

    /// A target with no drafter beside it (an empty `draft_capture_layers`) unrolls to exactly the
    /// tiles it did before the capture was inserted: every layer takes the alias arm.
    #[test]
    fn a_target_with_no_drafter_unrolls_as_before() {
        let ast = parse_body(DECODER).expect("parse");
        let params = llama(&[]);
        let (before, after) = (
            unroll_ast(&ast, &params),
            unroll_ast(&insert(&ast), &params),
        );
        assert_eq!(ops(&before), ops(&after));
    }

    /// With a drafter reading layers 1 and 3, the target captures the residual stream each of them
    /// carries out — the add that ends the layer — and nothing else.
    #[test]
    fn a_target_captures_the_layers_its_drafter_reads() {
        let ast = parse_body(DECODER).expect("parse");
        let fuf = unroll_ast(&insert(&ast), &llama(&[1, 3]));
        let adds: Vec<u32> = (fuf.nodes.iter())
            .filter(|n| n.op == OpKind::Add)
            .map(|n| n.id.0)
            .collect();
        let captured: Vec<u32> = (fuf.nodes.iter())
            .filter(|n| n.op == OpKind::CaptureHidden)
            .map(|n| match n.inputs[0] {
                FufInput::Tile { id, .. } => id.0,
                ref other => panic!("a capture of {other:?}"),
            })
            .collect();
        assert_eq!(captured, [adds[1], adds[3]]);
    }

    /// A layer loop whose writes nothing after it reads carries nothing out: no capture.
    #[test]
    fn a_loop_carrying_nothing_out_gets_no_capture() {
        let src = "
            hidden_states = embed(input_ids, embed_tokens)
            for layer in range(num_hidden_layers):
                scratch = rmsnorm(hidden_states, input_layernorm[layer])
            logits = gemm(hidden_states, lm_head)
            ";
        let ast = parse_body(src).expect("parse");
        let Stmt::For { body, .. } = &insert(&ast).statements[1] else {
            panic!("the loop moved");
        };
        assert_eq!(body.len(), 1);
    }
}

mod parse_python {
    /// `qwen3.py` drives the full parse → classify → shape → CFG →
    /// unroll → schedule → codegen chain to a NON-EMPTY, complete
    /// emission, without pinning bytes. Numerical equivalence of the
    /// emission is covered by the scratchy-models `qwen3_py_parity`
    /// e2e gate.
    #[test]
    fn python_costume_emits_the_qwen3_tape() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .canonicalize()
            .unwrap();
        let arch = "qwen3";
        let configs_dir = root.join("crates/models/arch/configs").join(arch);

        let py_text =
            std::fs::read_to_string(root.join("crates/models/arch/dsl/qwen3.py")).unwrap();
        let py_carrier = crate::parse_python::parse_python_file(&py_text).expect("python parse");
        // ONE model, named: the scope a `-Fmodel/qwen3-0.6b` build selects. The pipeline's subject
        // here is the emission, not the selection.
        let scope = crate::config::BuildScope::of_features(["qwen3-0.6b"]);
        let py_tokens = crate::compile_carrier_in(py_carrier, &configs_dir, &scope)
            .expect("python-costume pipeline");

        let mut py_rendered = String::with_capacity(1 << 20);
        crate::render_tokens(py_tokens, &mut py_rendered);

        // Non-vacuity guard: a scope that selects nothing emits an EMPTY token stream, and the
        // test would pass having verified nothing.
        assert!(
            py_rendered.len() > 100_000,
            "qwen3 emission is {} bytes — the scope selected no model",
            py_rendered.len(),
        );

        // The emission must be COMPLETE, not just big: the flip pinned
        // per-bucket tape symbols + the load/forward entry points. Check
        // the shape, not the bytes — byte-pinning every codegen change
        // would force regenerating a ~67k-line fixture with no extra
        // correctness signal over the numeric e2e gate.
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
}

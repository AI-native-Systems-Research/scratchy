// SPDX-License-Identifier: Apache-2.0
// The model-independent body of the `cuda_tapes` tests, shared by every qwen3.5 model's binary
// (`cuda_tapes.rs` for qwen3.5-0.8b, `cuda_tapes_9b.rs` for qwen3.5-9b). Included with
// `include!` at the binary's root, not as a module, so every test keeps its name (a module would
// prefix it). The including binary supplies the model: its `CUDA_TAPES`, `CUDA_UNLOWERED` and
// `CUDA_WEIGHTS`, the `chains` module (`targets/cuda/tests/common/chains.rs`), and the constants
// these tests read: `STEM`, `FEATURE`, `VOCAB`, `HIDDEN`, `HEADS`, `LAYERS`, `KV_LAYERS`,
// `GDN_HEADS`, `STEPS`, `DECODE_STEPS`, `DECODE_COLOUR_BYTES`, `DECODE_TOTAL`, `DECODE_RESULT`,
// `DECODE_IDS`, `DECODE_ROTARY`, `DECODE_KV`, `DECODE_GDN`, `PREFILL_STEPS`, `COLOURS`, `ARENAS`,
// `TOTALS`, `WEIGHTS`, `WEIGHT_BYTES`, `BUNDLES`, `TENSORS`, `SHAPES`, `LAYER_0_BUNDLE`,
// `BUNDLE_BYTES`, `BUNDLE_REGIONS` and `UNTIED_LM_HEAD`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use scratchy_subtile::handoff::TargetModelInput;
use scratchy_target_cuda::step::{AttnPrefillBf16, CudaStep, EmbedBf16, GemmBf16};
use scratchy_target_cuda::tape::{ArenaOffset, ArenaSpan, CudaTape, TapeInput, WeightIx};
use scratchy_target_cuda::weights::CudaWeight;
use scratchy_target_cuda_compiler::emit::{lower_model, model_facts};

/// The model's canonicals, as `#[forward]` handed them to the seam.
///
/// The pipeline selects models the way the build script that expanded `CUDA_TAPES` did: from
/// Cargo's `CARGO_FEATURE_<STEM>` and the quant scope in `SCRATCHY_QUANTS`, which Cargo and the
/// build script set for the build script only. This build's scope is the one model and no quant
/// preset, so the same two are set here, once, before the pipeline reads them; no other test in
/// this binary reads the environment.
fn input() -> TargetModelInput {
    static INPUT: std::sync::OnceLock<TargetModelInput> = std::sync::OnceLock::new();
    INPUT.get_or_init(capture).clone()
}

fn capture() -> TargetModelInput {
    // SAFETY: set before the pipeline's own threads start, and no other thread of this binary
    // reads or writes the environment.
    unsafe {
        std::env::set_var(FEATURE, "1");
        std::env::set_var("SCRATCHY_QUANTS", "");
    }
    let models = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dsl = std::fs::read_to_string(models.join("dsl/qwen3-5.py")).expect("read the DSL");
    let carrier = scratchy_forward_compiler_macro::parse_python_file(&dsl).expect("parse the DSL");
    let inputs = scratchy_forward_compiler_macro::target_model_inputs(
        carrier,
        &models.join("configs/qwen3-5"),
    )
    .expect("the forward pipeline");
    let mut inputs: Vec<_> = (inputs.into_iter()).filter(|i| i.stem == STEM).collect();
    assert_eq!(inputs.len(), 1, "one {STEM} model handed to the seam");
    inputs.remove(0)
}

/// The prefill buckets.
const BUCKETS: [u32; 4] = [512, 1024, 2048, 4096];
/// Every lowered bucket, ascending: decode's one row, then the prefill buckets.
const TAPE_BUCKETS: [u32; 5] = [1, 512, 1024, 2048, 4096];

const fn span(offset: u64, bytes: u64) -> ArenaSpan {
    ArenaSpan {
        offset: ArenaOffset(offset),
        bytes,
    }
}

/// The lowered tape of bucket `m`, looked up by its `bucket_m`, never by position.
fn tape_of(m: u32) -> &'static CudaTape {
    (CUDA_TAPES.iter())
        .find(|t| t.bucket_m == m)
        .unwrap_or_else(|| panic!("no m={m} tape"))
}

/// One tape's steps, by kind.
fn histogram(tape: &CudaTape) -> BTreeMap<&'static str, usize> {
    let mut h = BTreeMap::new();
    for step in tape.steps {
        let kind = match step {
            CudaStep::Embed(_) => "Embed",
            CudaStep::Gemm(_) => "Gemm",
            CudaStep::GemmNarrow(_) => "GemmNarrow",
            CudaStep::Gemv(_) => "Gemv",
            CudaStep::RmsNorm(_) => "RmsNorm",
            CudaStep::Add(_) => "Add",
            CudaStep::SiluMul(_) => "SiluMul",
            CudaStep::Rope(_) => "Rope",
            CudaStep::GateSplit(_) => "GateSplit",
            CudaStep::GateApply(_) => "GateApply",
            CudaStep::RopeAppend(_) => "RopeAppend",
            CudaStep::AttnPrefill(_) => "AttnPrefill",
            CudaStep::AttnDecode(_) => "AttnDecode",
            CudaStep::GatedDeltaNet(_) => "GatedDeltaNet",
            CudaStep::RouterLogits(_) => "RouterLogits",
            CudaStep::MoeRoute(_) => "MoeRoute",
            CudaStep::ExpertSort(_) => "ExpertSort",
            CudaStep::ExpertGemm(_) => "ExpertGemm",
            CudaStep::ExpertGemv(_) => "ExpertGemv",
            CudaStep::ExpertCombine(_) => "ExpertCombine",
            CudaStep::GateScale(_) => "GateScale",
        };
        *h.entry(kind).or_default() += 1;
    }
    h
}

#[test]
fn every_emitted_tape_is_what_cuda_lowers_from_the_same_canonical() {
    let input = input();
    let facts = model_facts(&input.facts);
    assert_eq!(
        (facts.head_dim, facts.rot_dim, facts.rope_theta),
        (256, 64, 1e7),
        "{STEM}'s head_dim, rot_dim and rope_theta"
    );
    let gdn = facts.gdn.expect("qwen3.5 has GDN layers");
    assert_eq!(
        (gdn.num_k_heads, gdn.head_k, gdn.conv_kernel, gdn.eps),
        (16, 128, 4, 1e-6),
        "{STEM}'s GDN facts"
    );
    assert_eq!(
        facts.embed.map(|e| e.vocab),
        Some(VOCAB),
        "{STEM}'s embedding table"
    );
    // The whole model as the seam lowers it: every bucket, every sk (one tape per m, which
    // `lower_model` checks), then the weight table.
    let model = lower_model(&input).unwrap_or_else(|e| panic!("lower_model: {e}"));
    assert_eq!(model.tapes.len(), CUDA_TAPES.len(), "tapes");
    for (tape, lowered) in CUDA_TAPES.iter().zip(&model.tapes) {
        let name = format!("{STEM} m={}", tape.bucket_m);
        chains::assert_tape_is(&name, tape, lowered);
    }
    assert_eq!(CUDA_WEIGHTS, model.weights.as_slice(), "CUDA_WEIGHTS");
}

#[test]
fn the_tapes_are_the_decode_and_prefill_buckets_and_validate() {
    let ms: Vec<u32> = CUDA_TAPES.iter().map(|t| t.bucket_m).collect();
    assert_eq!(ms, TAPE_BUCKETS, "CUDA_TAPES' buckets");
    for tape in CUDA_TAPES {
        tape.validate()
            .unwrap_or_else(|e| panic!("m={}: {e}", tape.bucket_m));
    }
}

/// The buckets above one row and below the GEMM's tile are left out, each by its first GEMM's
/// `m`; the one-row bucket is not among them, it lowers (the GEMV).
#[test]
fn the_buckets_below_the_gemm_tile_are_left_out_and_named() {
    let ms: Vec<u32> = CUDA_UNLOWERED.iter().map(|&(m, _)| m).collect();
    assert_eq!(ms, [2, 4, 8, 64], "CUDA_UNLOWERED's buckets");
    for &(m, why) in CUDA_UNLOWERED {
        let want = format!("m = {m} must be a non-zero multiple of 128, the bf16 GEMM's tile");
        assert!(why.contains(&want), "m={m}: {why}");
    }
}

/// The one-row decode tape: the layout the shared colouring and arena passes give it. Every GEMM
/// whose `n` the narrow GEMM does not take is a GEMV, the attentions are the decode kernel's, and
/// the steps, colours and regions sit where the prefill buckets' rules put them.
#[test]
fn the_decode_tape_has_the_predicted_layout() {
    let tape = tape_of(1);
    let want_steps: BTreeMap<&str, usize> = DECODE_STEPS.into_iter().collect();
    assert_eq!(tape.steps.len(), STEPS, "steps");
    assert_eq!(histogram(tape), want_steps, "step histogram");
    assert_eq!(tape.arena.colour_bytes, DECODE_COLOUR_BYTES, "colour bytes");
    assert_eq!(tape.arena.total_bytes, DECODE_TOTAL, "arena total_bytes");
    assert_eq!(tape.result, DECODE_RESULT, "the logits, [1, {VOCAB}] bf16");
    let ids = DECODE_IDS;
    assert_eq!(
        tape.input,
        TapeInput::TokenIds {
            span: ids,
            vocab: VOCAB
        },
        "input"
    );
    assert_eq!(
        tape.steps[0],
        CudaStep::Embed(EmbedBf16 {
            tokens: ids.offset,
            table: WeightIx(0),
            out: ArenaOffset(0),
            rows: 1,
            hidden: HIDDEN,
            vocab: VOCAB,
        }),
        "the embed"
    );
    let rotary = tape.rotary.expect("a rotary table");
    assert_eq!((rotary.span, rotary.rot_dim), (DECODE_ROTARY, 64), "rotary");
    let kv = tape.kv.expect("a KV cache");
    assert_eq!(
        [kv.block_table, kv.slot_mapping, kv.sequence],
        DECODE_KV,
        "KV regions"
    );
    let gdn = tape.gdn.expect("GDN layers");
    assert_eq!([gdn.sequence, gdn.scratch], DECODE_GDN, "GDN regions");
    for (i, step) in tape.steps.iter().enumerate() {
        match *step {
            CudaStep::Gemv(GemmBf16 { m, .. }) | CudaStep::GemmNarrow(GemmBf16 { m, .. }) => {
                assert_eq!(m, 1, "step {i}");
            }
            CudaStep::AttnDecode(AttnPrefillBf16 {
                rows,
                q_heads,
                kv_heads,
                head_dim,
                block_table,
                sequence,
                ..
            }) => {
                assert_eq!(
                    (rows, q_heads, kv_heads, head_dim),
                    (1, HEADS.0, HEADS.1, 256),
                    "step {i}"
                );
                assert_eq!(
                    (block_table, sequence),
                    (kv.block_table.offset, kv.sequence.offset)
                );
            }
            CudaStep::Gemm(_) | CudaStep::AttnPrefill(_) => {
                panic!("step {i}: a prefill kernel in the one-row tape: {step:?}")
            }
            _ => {}
        }
    }
}

/// The four prefill tapes, each looked up by its bucket.
#[test]
fn every_prefill_tape_has_the_measured_layout() {
    let want_steps: BTreeMap<&str, usize> = PREFILL_STEPS.into_iter().collect();
    let gdn_layers: Vec<u32> = (0..LAYERS).filter(|l| !KV_LAYERS.contains(l)).collect();
    for &(m, total, input, result) in &ARENAS {
        let tape = tape_of(m);
        assert_eq!(tape.steps.len(), STEPS, "m={m}: steps");
        assert_eq!(histogram(tape), want_steps, "m={m}: step histogram");
        assert_eq!(tape.arena.colour_bytes.len(), COLOURS, "m={m}: colours");
        let with_ids = TOTALS[BUCKETS.iter().position(|&b| b == m).expect("a bucket")];
        assert_eq!(tape.arena.total_bytes, with_ids, "m={m}: arena total_bytes");
        assert_eq!(
            with_ids,
            total + 4 * u64::from(m),
            "m={m}: the arena grows by 4·m"
        );
        let ids = span(total, 4 * u64::from(m));
        assert_eq!(
            tape.input,
            TapeInput::TokenIds {
                span: ids,
                vocab: VOCAB
            },
            "m={m}: input"
        );
        assert_eq!(
            tape.steps[0],
            CudaStep::Embed(EmbedBf16 {
                tokens: ids.offset,
                table: WeightIx(0),
                out: input.offset,
                rows: m,
                hidden: HIDDEN,
                vocab: VOCAB,
            }),
            "m={m}: the embed writes colour 0, {input:?}"
        );
        assert_eq!(tape.arena.colour_bytes[0], input.bytes, "m={m}: colour 0");
        assert_eq!(tape.result, result, "m={m}: result span");
        // The result is the full logits, `[m, 248320]` bf16.
        assert_eq!(result.bytes, u64::from(m) * 248_320 * 2, "m={m}: logits");
        let weights: Vec<u64> = (tape.weight_bytes.iter().copied())
            .filter(|&b| b > 0)
            .collect();
        assert_eq!(weights.len(), WEIGHTS, "m={m}: nonzero weights");
        assert_eq!(
            weights.iter().sum::<u64>(),
            WEIGHT_BYTES,
            "m={m}: weight bytes"
        );
        let kv = tape.kv.expect("a KV cache");
        assert_eq!(kv.layers, KV_LAYERS, "m={m}: KV layers");
        assert_eq!(
            (kv.kv_heads, kv.head_dim),
            (HEADS.1, 256),
            "m={m}: KV geometry"
        );
        let gdn = tape.gdn.expect("GDN layers");
        assert_eq!(gdn.layers, gdn_layers.as_slice(), "m={m}: GDN layers");
        assert_eq!(
            (
                gdn.num_k_heads,
                gdn.num_v_heads,
                gdn.head_k,
                gdn.head_v,
                gdn.conv_kernel
            ),
            (GDN_HEADS.0, GDN_HEADS.1, 128, 128, 4),
            "m={m}: GDN geometry"
        );
    }
}

/// The weight table: one entry per weight index the tapes read, `WEIGHTS` of them, from the
/// checkpoint's `TENSORS` decoder tensors (the bf16 tensors, `BUNDLES` GDN bundles of four, the
/// embedding table); none named twice. The lm_head reads the embedding table's index when
/// `UNTIED_LM_HEAD` is `None` (tied), and its own tensor at its own index when it names one.
#[test]
fn the_weight_table_names_the_checkpoints_tensors() {
    assert_eq!(
        CUDA_WEIGHTS.len(),
        CUDA_TAPES[0].weight_bytes.len(),
        "one entry per index"
    );
    let entries: Vec<(usize, CudaWeight)> = (CUDA_WEIGHTS.iter().enumerate())
        .filter_map(|(ix, w)| w.map(|w| (ix, w)))
        .collect();
    assert_eq!(entries.len(), WEIGHTS, "Some entries");
    let bundles = (entries.iter())
        .filter(|(_, w)| matches!(w, CudaWeight::GdnBundle { .. }))
        .count();
    assert_eq!(bundles, BUNDLES, "GDN bundles");
    let mut names = std::collections::BTreeSet::new();
    let mut leads = std::collections::BTreeSet::new();
    for (ix, w) in &entries {
        let tensors = match *w {
            CudaWeight::Bf16 { tensor, .. } => vec![tensor],
            CudaWeight::GdnBundle {
                conv1d,
                a_log,
                dt_bias,
                norm,
                ..
            } => vec![conv1d, a_log, dt_bias, norm],
            CudaWeight::MoeBundle {
                router,
                gate_up,
                down,
                ..
            } => vec![router, gate_up, down],
        };
        assert!(
            leads.insert(tensors[0]),
            "weight {ix}: {} named twice",
            tensors[0]
        );
        names.extend(tensors);
        for tape in CUDA_TAPES {
            assert_eq!(
                Some(tape.weight_bytes[*ix]),
                w.packed_bytes(),
                "m={}: weight {ix}'s declared bytes",
                tape.bucket_m
            );
        }
    }
    assert_eq!(names.len(), TENSORS, "distinct tensors");
    for (ix, w) in CUDA_WEIGHTS.iter().enumerate() {
        if w.is_none() {
            for tape in CUDA_TAPES {
                assert_eq!(tape.weight_bytes[ix], 0, "m={}: weight {ix}", tape.bucket_m);
            }
        }
    }
    let bytes: u64 = (entries.iter())
        .map(|(_, w)| w.packed_bytes().expect("fits"))
        .sum();
    assert_eq!(bytes, WEIGHT_BYTES, "the table's bytes");
    let by_name = |name: &str| {
        (entries.iter())
            .find(|(_, w)| matches!(w, CudaWeight::Bf16 { tensor, .. } if *tensor == name))
            .copied()
            .unwrap_or_else(|| panic!("no entry for {name}"))
    };
    let shape = |w: CudaWeight| match w {
        CudaWeight::Bf16 { shape, .. } => shape,
        CudaWeight::GdnBundle { .. } | CudaWeight::MoeBundle { .. } => panic!("a bundle"),
    };
    let (e, embed) = by_name("language_model.model.embed_tokens.weight");
    assert_eq!(e, 0, "the embedding table is the embedded hidden's index");
    assert_eq!(
        shape(embed),
        [u64::from(VOCAB), u64::from(HIDDEN)],
        "the embedding table"
    );
    for (name, want) in SHAPES {
        assert_eq!(shape(by_name(name).1), want, "{name}");
    }
    let bundle = (entries.iter())
        .find_map(|(_, w)| match *w {
            b @ CudaWeight::GdnBundle {
                conv1d: "language_model.model.layers.0.linear_attn.conv1d.weight",
                ..
            } => Some(b),
            _ => None,
        })
        .expect("layer 0's bundle");
    assert_eq!(bundle, LAYER_0_BUNDLE);
    assert_eq!(
        bundle.packed_bytes(),
        Some(BUNDLE_BYTES),
        "a bundle's bytes"
    );
    let CudaWeight::GdnBundle {
        conv_dim,
        conv_kernel,
        nv,
        head_v,
        ..
    } = LAYER_0_BUNDLE
    else {
        unreachable!("LAYER_0_BUNDLE is a bundle");
    };
    let layout =
        scratchy_target_cuda::step::gdn_bundle(conv_dim, conv_kernel, nv, head_v).expect("fits");
    assert_eq!(
        (
            layout.conv1d,
            layout.a_log,
            layout.dt_bias,
            layout.norm,
            layout.bytes
        ),
        BUNDLE_REGIONS,
        "the bundle's regions"
    );
    let lm_head = match UNTIED_LM_HEAD {
        // The tied lm_head: the last step reads the embedding table's index, and its own is empty.
        None => WeightIx(e as u32),
        // The untied lm_head: its own `[vocab, hidden]` tensor at its own index, and no step but
        // each tape's first, the embed, reads the embedding table.
        Some((ix, tensor)) => {
            let (at, w) = by_name(tensor);
            assert_eq!(at, ix as usize, "the lm_head's index");
            assert_ne!(at, e, "the lm_head's index is not the embedding table's");
            assert_eq!(
                shape(w),
                [u64::from(VOCAB), u64::from(HIDDEN)],
                "the lm_head, {tensor}"
            );
            for tape in CUDA_TAPES {
                for (i, step) in tape.steps.iter().enumerate().skip(1) {
                    let mut step = *step;
                    assert_ne!(
                        step.weight_mut().copied(),
                        Some(WeightIx(e as u32)),
                        "m={}: step {i} reads the embedding table",
                        tape.bucket_m
                    );
                }
            }
            WeightIx(ix)
        }
    };
    // The decode tape's lm_head is the GEMV, every prefill tape's the tiled GEMM.
    for tape in CUDA_TAPES {
        let m = tape.bucket_m;
        let (Some(&CudaStep::Gemm(GemmBf16 { b, n, k, .. }))
        | Some(&CudaStep::Gemv(GemmBf16 { b, n, k, .. }))) = tape.steps.last()
        else {
            panic!("m={m}: the tape ends with the lm_head GEMM");
        };
        assert_eq!(
            matches!(tape.steps.last(), Some(CudaStep::Gemv(_))),
            m == 1,
            "m={m}: the lm_head's kernel"
        );
        assert_eq!((b, n, k), (lm_head, VOCAB, HIDDEN), "m={m}: the lm_head");
    }
}

/// The model is in the CUDA registry exactly once, under its stem, holding the emitted tapes and
/// weight table themselves (the same statics, not copies): what the serving worker finds it by.
#[test]
fn the_model_is_registered_under_its_stem() {
    let entries: Vec<_> = scratchy_target_cuda::registry::entries().collect();
    let stems: Vec<&str> = entries.iter().map(|e| e.stem).collect();
    assert_eq!(stems, [STEM], "the registry");
    let entry = entries[0];
    assert!(std::ptr::eq(entry.tapes, CUDA_TAPES), "the entry's tapes");
    assert!(
        std::ptr::eq(entry.weights, CUDA_WEIGHTS),
        "the entry's weight table"
    );
}

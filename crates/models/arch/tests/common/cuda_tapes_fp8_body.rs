// SPDX-License-Identifier: Apache-2.0
// The model-independent body of the FP8 `cuda_tapes_*_fp8` tests (FP8 S2), shared by the 27B, the
// 35B and Gemma 4 26B-A4B under `fp8-dynamic-per-channel`. Included with `include!` at the
// binary's root, as `cuda_tapes_body.rs` is, so every test keeps its name. The including binary
// supplies the model: its `CUDA_TAPES`, `CUDA_UNLOWERED` and `CUDA_WEIGHTS`, the `chains` module
// (`targets/cuda/tests/common/chains.rs`), and the constants these tests read: `STEM`, `FEATURE`,
// `ARCH`, `DECODE_STEPS`, `PREFILL_STEPS`, `QUANTS`, `QUANT_COLS`, `QUANT_SCRATCH`,
// `EXPERT_WEIGHT`, `WEIGHT_KINDS`, `TABLE_LEN` and `HEADERS`.
//
// No toolchain and no GPU: the tapes are checked as emitted, and the weight table against the
// checkpoint's real safetensors headers (a fixture of names, dtypes and shapes, written out as a
// sparse file of that header, so no weight byte is read or stored).

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use scratchy_subtile::handoff::TargetModelInput;
use scratchy_target_cuda::step::{
    CudaStep, ExpertGateUpActBf16, ExpertGemmBf16, ExpertGemvBf16, GateUpActBf16, GateUpKernel,
    MoeWeight, QuantChainError, QuantRowsFp8, quant_chain,
};
use scratchy_target_cuda::tape::{CUDA_ARENA_ALIGN, CudaTape, TapeError};
use scratchy_target_cuda::weights::{CudaWeight, check, open_checkpoint};
use scratchy_target_cuda_compiler::emit::lower_model;

/// The quant preset every tape of this binary is lowered under.
const PRESET: &str = "fp8-dynamic-per-channel";
/// Every lowered bucket, ascending: decode's one row, then the prefill buckets.
const TAPE_BUCKETS: [u32; 5] = [1, 512, 1024, 2048, 4096];

/// The stem the seam hands the fp8 variant over under, and the registry holds it by.
fn variant() -> String {
    format!("{STEM}-{PRESET}")
}

/// The model's canonicals, as `#[forward]` handed them to the seam: selected, as the build script
/// that expanded `CUDA_TAPES` did, by `CARGO_FEATURE_<STEM>` and `SCRATCHY_QUANTS` naming the one
/// preset, set once here, before the pipeline reads them; no other test in this binary reads the
/// environment.
fn input() -> TargetModelInput {
    static INPUT: std::sync::OnceLock<TargetModelInput> = std::sync::OnceLock::new();
    INPUT.get_or_init(capture).clone()
}

fn capture() -> TargetModelInput {
    // SAFETY: set before the pipeline's own threads start, and no other thread of this binary
    // reads or writes the environment.
    unsafe {
        std::env::set_var(FEATURE, "1");
        std::env::set_var("SCRATCHY_QUANTS", PRESET);
    }
    let models = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dsl = std::fs::read_to_string(models.join(format!("dsl/{ARCH}.py"))).expect("the DSL");
    let carrier = scratchy_forward_compiler_macro::parse_python_file(&dsl).expect("parse the DSL");
    let inputs = scratchy_forward_compiler_macro::target_model_inputs(
        carrier,
        &models.join(format!("configs/{ARCH}")),
    )
    .expect("the forward pipeline");
    let stem = variant();
    let mut inputs: Vec<_> = (inputs.into_iter()).filter(|i| i.stem == stem).collect();
    assert_eq!(inputs.len(), 1, "one {stem} model handed to the seam");
    inputs.remove(0)
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
            CudaStep::Scale(_) => "Scale",
            CudaStep::RmsNormUnit(_) => "RmsNormUnit",
            CudaStep::GeluMul(_) => "GeluMul",
            CudaStep::LayerScale(_) => "LayerScale",
            CudaStep::SoftCap(_) => "SoftCap",
            CudaStep::RouterNorm(_) => "RouterNorm",
            CudaStep::GemmCol64(_) => "GemmCol64",
            CudaStep::GemmGroup(_) => "GemmGroup",
            CudaStep::GemmNarrowGroup(_) => "GemmNarrowGroup",
            CudaStep::GemvGroup(_) => "GemvGroup",
            CudaStep::AddRmsNorm(_) => "AddRmsNorm",
            CudaStep::NormAdd(_) => "NormAdd",
            CudaStep::GateUpAct(_) => "GateUpAct",
            CudaStep::ExpertGateUpAct(_) => "ExpertGateUpAct",
            CudaStep::GdnDecode(_) => "GdnDecode",
            CudaStep::QuantRowsFp8(_) => "QuantRowsFp8",
            CudaStep::GemmFp8(_) => "GemmFp8",
            CudaStep::GemmCol64Fp8(_) => "GemmCol64Fp8",
            CudaStep::GemmNarrowFp8(_) => "GemmNarrowFp8",
            CudaStep::GemvFp8(_) => "GemvFp8",
            CudaStep::GemmGroupFp8(_) => "GemmGroupFp8",
            CudaStep::GemmNarrowGroupFp8(_) => "GemmNarrowGroupFp8",
            CudaStep::GemvGroupFp8(_) => "GemvGroupFp8",
        };
        *h.entry(kind).or_default() += 1;
    }
    h
}

/// Every quant write of `tape`, with its position.
fn quants(tape: &CudaTape) -> Vec<(usize, QuantRowsFp8)> {
    (tape.steps.iter().enumerate())
        .filter_map(|(i, s)| match *s {
            CudaStep::QuantRowsFp8(q) => Some((i, q)),
            _ => None,
        })
        .collect()
}

#[test]
fn every_emitted_tape_is_what_cuda_lowers_from_the_same_canonical() {
    let input = input();
    let model = lower_model(&input).unwrap_or_else(|e| panic!("lower_model: {e}"));
    assert_eq!(model.tapes.len(), CUDA_TAPES.len(), "tapes");
    for (tape, lowered) in CUDA_TAPES.iter().zip(&model.tapes) {
        let name = format!("{} m={}", variant(), tape.bucket_m);
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
    let left: Vec<u32> = CUDA_UNLOWERED.iter().map(|&(m, _)| m).collect();
    assert_eq!(left, [2, 4, 8, 16, 32, 64], "CUDA_UNLOWERED's buckets");
}

/// The step histogram of every bucket: the decode tape's projections are fp8 GEMVs and it has no
/// quant write (the GEMV quantizes its one row in its prologue); every prefill tape's are fp8
/// GEMMs, each (a group or a gate/up pair once) after its own quant write. What stays bf16 is
/// what the checkpoint leaves unquantized (the lm_head, the routers, Qwen3.6's GDN projections).
#[test]
fn every_tape_has_the_fp8_steps() {
    for tape in CUDA_TAPES {
        let m = tape.bucket_m;
        let want: BTreeMap<&str, usize> = match m {
            1 => DECODE_STEPS.iter().copied().collect(),
            _ => PREFILL_STEPS.iter().copied().collect(),
        };
        assert_eq!(histogram(tape), want, "m={m}: step histogram");
        let quantized = (tape.steps.iter())
            .filter(|s| s.quantized_input().is_some())
            .count();
        assert_eq!(quantized, if m == 1 { 0 } else { QUANTS }, "m={m}");
        for (i, s) in tape.steps.iter().enumerate() {
            // Every gate/up pair of these checkpoints is quantized.
            if let CudaStep::GateUpAct(GateUpActBf16 { kernel, .. }) = *s {
                let want: &[GateUpKernel] = match m {
                    1 => &[GateUpKernel::GemvFp8],
                    _ => &[GateUpKernel::GemmFp8, GateUpKernel::GemmCol64Fp8],
                };
                assert!(want.contains(&kernel), "m={m} step {i}: {kernel:?}");
            }
            let expert = match *s {
                CudaStep::ExpertGemm(ExpertGemmBf16 { weight, .. })
                | CudaStep::ExpertGemv(ExpertGemvBf16 { weight, .. })
                | CudaStep::ExpertGateUpAct(ExpertGateUpActBf16 {
                    gate: ExpertGemvBf16 { weight, .. },
                    ..
                }) => Some(weight),
                _ => None,
            };
            if let Some(weight) = expert {
                assert_eq!(Some(weight), EXPERT_WEIGHT, "m={m} step {i}: the bank");
            }
        }
    }
}

/// The quant writes of every prefill tape: each sits right before the one fp8 GEMM that reads it,
/// quantizes all `m` rows of that GEMM's `k` columns from a bf16 colour, and writes the one quant
/// scratch, the arena's last region: every write starts at its offset, the largest is its size,
/// and it ends the arena.
#[test]
fn every_quant_write_sits_before_its_fp8_gemm() {
    assert!(quants(tape_of(1)).is_empty(), "m=1: no quant write");
    for &(m, at, bytes) in &QUANT_SCRATCH {
        let tape = tape_of(m);
        let qs = quants(tape);
        assert_eq!(qs.len(), QUANTS, "m={m}: quant writes");
        let mut cols = std::collections::BTreeSet::new();
        let mut most = 0;
        for &(i, q) in &qs {
            let reader = tape.steps.get(i + 1).and_then(CudaStep::quantized_input);
            assert_eq!(
                reader,
                Some((q.q, q.rows, q.cols)),
                "m={m} step {i}: the next step reads what it wrote"
            );
            assert_eq!((q.q.0, q.rows), (at, m), "m={m} step {i}: its region, rows");
            assert!(
                q.x.0 < at && q.x.0.is_multiple_of(CUDA_ARENA_ALIGN),
                "m={m} step {i}: x {} is an aligned colour before the scratch",
                q.x.0
            );
            cols.insert(q.cols);
            most = most.max(q.bytes().expect("fits"));
        }
        assert_eq!(cols.into_iter().collect::<Vec<_>>(), QUANT_COLS, "m={m}");
        assert_eq!(most, bytes, "m={m}: the scratch is the largest write");
        assert!(at.is_multiple_of(CUDA_ARENA_ALIGN), "m={m}: aligned");
        let end = at + bytes.next_multiple_of(CUDA_ARENA_ALIGN);
        assert_eq!(end, tape.arena.total_bytes, "m={m}: the arena's last region");
    }
}

/// The quant scratch's liveness check, on the real tapes: each passes, and each loses it when a
/// quant write is dropped (its GEMM then reads rows nobody wrote for it) or doubled (the first
/// copy is overwritten unread), through `quant_chain` and through `CudaTape::validate`.
#[test]
fn the_quant_scratch_liveness_check_holds_and_bites() {
    for &(m, ..) in &QUANT_SCRATCH {
        let tape = tape_of(m);
        assert_eq!(quant_chain(tape.steps), Ok(()), "m={m}");
        let (i, q) = quants(tape)[QUANTS / 2];
        let mut dropped = tape.steps.to_vec();
        dropped.remove(i);
        assert_eq!(
            quant_chain(&dropped),
            Err(QuantChainError::Unquantized { step: i }),
            "m={m}: write {i} dropped"
        );
        let mut doubled = tape.steps.to_vec();
        doubled.insert(i, CudaStep::QuantRowsFp8(q));
        assert_eq!(
            quant_chain(&doubled),
            Err(QuantChainError::Overwritten { step: i }),
            "m={m}: write {i} doubled"
        );
        for (what, steps, error) in [
            ("dropped", dropped, QuantChainError::Unquantized { step: i }),
            ("doubled", doubled, QuantChainError::Overwritten { step: i }),
        ] {
            let bad = CudaTape {
                steps: Box::leak(steps.into_boxed_slice()),
                ..*tape
            };
            assert_eq!(
                bad.validate(),
                Err(TapeError::QuantChain { error }),
                "m={m}: validate, write {i} {what}"
            );
        }
    }
}

/// The weight table: its length and its entries by kind, every fp8 projection's buffer its e4m3
/// bytes then its f32 scales, and every index a tape reads with an entry.
#[test]
fn the_weight_table_has_the_fp8_entries() {
    assert_eq!(CUDA_WEIGHTS.len(), TABLE_LEN, "the table's indices");
    let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
    for w in CUDA_WEIGHTS.iter().flatten() {
        let kind = match w {
            CudaWeight::Bf16 { .. } => "Bf16",
            CudaWeight::GdnBundle { .. } => "GdnBundle",
            CudaWeight::MoeBundle { .. } => "MoeBundle",
            CudaWeight::ExpertBank { .. } => "ExpertBank",
            CudaWeight::Fp8 { .. } => "Fp8",
            CudaWeight::Fp8MoeBundle { .. } => "Fp8MoeBundle",
            CudaWeight::Fp8ExpertBank { .. } => "Fp8ExpertBank",
        };
        *kinds.entry(kind).or_default() += 1;
        if let CudaWeight::Fp8 {
            weight,
            scale,
            shape,
        } = *w
        {
            assert_eq!(scale, format!("{weight}_scale"), "{weight}'s scale");
            let &[n, k] = shape else {
                panic!("{weight}: {shape:?}")
            };
            let at = (n * k).next_multiple_of(CUDA_ARENA_ALIGN);
            assert_eq!(w.packed_bytes(), Some(at + 4 * n), "{weight}'s buffer");
        }
    }
    assert_eq!(kinds, WEIGHT_KINDS.iter().copied().collect(), "entries");
    for tape in CUDA_TAPES {
        for (ix, &bytes) in tape.weight_bytes.iter().enumerate() {
            let entry = CUDA_WEIGHTS[ix];
            assert_eq!(
                bytes == 0,
                entry.is_none(),
                "m={} weight {ix}: {bytes} bytes, entry {entry:?}",
                tape.bucket_m
            );
        }
    }
}

/// One checkpoint's header fixture: its tensors as (name, safetensors dtype, shape), every expert
/// written out from expert 0's, in the fixture's order.
fn header_tensors() -> Vec<(String, String, Vec<u64>)> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(HEADERS);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let doc: serde_json::Value = serde_json::from_str(&text).expect("the fixture parses");
    let experts = doc["experts"].as_u64().expect("experts");
    let mut out = Vec::new();
    for row in doc["tensors"].as_array().expect("tensors") {
        let name = row[0].as_str().expect("a name");
        let dtype = row[1].as_str().expect("a dtype").to_string();
        let shape: Vec<u64> = (row[2].as_array().expect("a shape").iter())
            .map(|d| d.as_u64().expect("a dim"))
            .collect();
        match name.split_once(".experts.0.") {
            Some((at, leaf)) => out.extend((0..experts).map(|e| {
                (format!("{at}.experts.{e}.{leaf}"), dtype.clone(), shape.clone())
            })),
            None => out.push((name.to_string(), dtype, shape)),
        }
    }
    out
}

/// `tensors` as one `model.safetensors` in `dir`: the header as safetensors writes it, then a
/// sparse hole of the data's size, so every offset is in the file and no byte is stored.
fn write_header_checkpoint(dir: &Path, tensors: &[(String, String, Vec<u64>)]) {
    let mut header = serde_json::Map::new();
    let mut end = 0u64;
    for (name, dtype, shape) in tensors {
        let size = match dtype.as_str() {
            "F8_E4M3" => 1,
            "BF16" => 2,
            "F32" => 4,
            other => panic!("{name}: dtype {other}"),
        };
        let bytes = shape.iter().product::<u64>() * size;
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [end, end + bytes]}),
        );
        end += bytes;
    }
    let mut json = serde_json::Value::Object(header).to_string().into_bytes();
    json.resize(json.len().next_multiple_of(8), b' ');
    let path = dir.join("model.safetensors");
    let mut f = std::fs::File::create(&path).expect("create the checkpoint");
    f.write_all(&(json.len() as u64).to_le_bytes()).expect("its length");
    f.write_all(&json).expect("its header");
    f.set_len(8 + json.len() as u64 + end).expect("its data, a hole");
}

/// The weight table against the checkpoint's real headers (`tests/fixtures/{HEADERS}`): `check`
/// finds every tensor every entry reads, of the dtype it takes (F8_E4M3 for an fp8 weight, BF16
/// for its scale and for every dense tensor) and the shape it needs, and every packed size is the
/// one the tapes declare; and every F8_E4M3 tensor of the decoder is one an fp8 entry reads.
#[test]
fn the_weight_table_checks_against_the_checkpoints_headers() {
    let tensors = header_tensors();
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("fp8_headers_{STEM}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("make the checkpoint dir");
    write_header_checkpoint(&dir, &tensors);
    let w = open_checkpoint(&dir).expect("open the header checkpoint");
    check(CUDA_WEIGHTS, CUDA_TAPES, &w).unwrap_or_else(|e| panic!("{}: {e}", variant()));
    let fp8_on_disk = tensors.iter().filter(|(_, d, _)| d == "F8_E4M3").count();
    let fp8_read: usize = (CUDA_WEIGHTS.iter().flatten())
        .map(|w| match *w {
            CudaWeight::Fp8 { .. } => 1,
            CudaWeight::Fp8MoeBundle { experts, .. } | CudaWeight::Fp8ExpertBank { experts, .. } => {
                3 * experts as usize
            }
            _ => 0,
        })
        .sum();
    assert_eq!(fp8_read, fp8_on_disk, "the fp8 tensors the table reads");
}

/// The fp8 variant is in the CUDA registry exactly once, under its own stem, holding the emitted
/// tapes and weight table themselves: the preset replaced the bf16 model rather than adding one.
#[test]
fn the_model_is_registered_under_its_fp8_stem() {
    let entries: Vec<_> = scratchy_target_cuda::registry::entries().collect();
    let stems: Vec<&str> = entries.iter().map(|e| e.stem).collect();
    assert_eq!(stems, [variant().as_str()], "the registry");
    assert!(std::ptr::eq(entries[0].tapes, CUDA_TAPES), "the tapes");
    assert!(std::ptr::eq(entries[0].weights, CUDA_WEIGHTS), "the table");
}

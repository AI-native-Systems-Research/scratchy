// SPDX-License-Identifier: Apache-2.0
// The model-independent body of the FP8 `cuda_tapes_*_fp8` tests (FP8 S2), shared by the 27B, the
// 35B and Gemma 4 26B-A4B under `fp8-dynamic-per-channel`. Included with `include!` at the
// binary's root, so every test keeps its name. The including binary
// supplies the model: its `CUDA_TAPES`, `CUDA_UNLOWERED` and `CUDA_WEIGHTS`, the `chains` module
// (`targets/cuda/tests/common/chains.rs`), and the constants these tests read: `STEM`, `FEATURE`,
// `ARCH`, `GREEDY_WIDTH`, `DECODE_STEPS`, `PREFILL_STEPS`, `QUANTS`, `QUANT_READERS`, `QUANT_COLS`,
// `QUANT_SCRATCH`, `EXPERT_WEIGHT`, `WEIGHT_KINDS`, `TABLE_LEN` and `HEADERS`.
//
// No toolchain and no GPU: the tapes are checked as emitted, and the weight table against the
// checkpoint's real safetensors headers (a fixture of names, dtypes and shapes, written out as a
// sparse file of that header, so no weight byte is read or stored).

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use scratchy_subtile::handoff::TargetModelInput;
use scratchy_target_cuda::step::{
    CudaStep, ExpertGateUpActBf16, ExpertGateUpTcFp8, ExpertGemmBf16, ExpertGemvBf16,
    GateUpActBf16, GateUpKernel, GateUpTcFp8, MoeWeight, QuantChainError, QuantRowsFp8,
    gemv_tc_takes, quant_chain,
};
use scratchy_target_cuda::tape::{
    ArenaSpan, CUDA_ACT_BYTES, CUDA_ARENA_ALIGN, CudaTape, TapeError, overlap,
};
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
/// that expanded `CUDA_TAPES` did, by a scope of the stem's feature (`FEATURE`, cargo's env name
/// for it without the `CARGO_FEATURE_` prefix) and the one preset.
fn input() -> TargetModelInput {
    static INPUT: std::sync::OnceLock<TargetModelInput> = std::sync::OnceLock::new();
    INPUT.get_or_init(capture).clone()
}

fn capture() -> TargetModelInput {
    let scope = scratchy_forward_compiler_macro::Scope::from_feature_flags([FEATURE], false, false)
        .with_quants([PRESET]);
    let models = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dsl = std::fs::read_to_string(models.join(format!("dsl/{ARCH}.py"))).expect("the DSL");
    let carrier = scratchy_forward_compiler_macro::parse_python_file(&dsl).expect("parse the DSL");
    let inputs = scratchy_forward_compiler_macro::target_model_inputs(
        carrier,
        &models.join(format!("configs/{ARCH}")),
        &scope,
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
            CudaStep::ExpertSortFp8(_) => "ExpertSortFp8",
            CudaStep::ExpertActQuant(_) => "ExpertActQuant",
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
            CudaStep::GemvTcFp8(_) => "GemvTcFp8",
            CudaStep::GemvGroupTcFp8(_) => "GemvGroupTcFp8",
            CudaStep::GateUpTcFp8(_) => "GateUpTcFp8",
            CudaStep::ExpertGemvTcFp8(_) => "ExpertGemvTcFp8",
            CudaStep::ExpertGateUpTcFp8(_) => "ExpertGateUpTcFp8",
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

/// The step histogram of every bucket: the decode tape's fp8 GEMVs the shape rule takes run on the
/// tensor cores, each after a one-row quant write (or reading one a step before it wrote); every
/// prefill tape's are fp8 GEMMs, each (a group or a gate/up pair once) after a quant write of its
/// rows (its own, or one still live: `every_quant_write_sits_before_its_fp8_gemm`). What stays
/// bf16 is what the checkpoint leaves unquantized (the lm_head, the routers, Qwen3.6's GDN
/// projections).
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
        assert_eq!(quantized, if m == 1 { DECODE_QUANT_READERS } else { QUANT_READERS }, "m={m}");
        for (i, s) in tape.steps.iter().enumerate() {
            // Every gate/up pair of these checkpoints is quantized; a tensor-core one folds the
            // CUDA-core one-row gate/up's kernel.
            if let CudaStep::GateUpAct(GateUpActBf16 { kernel, .. })
            | CudaStep::GateUpTcFp8(GateUpTcFp8 {
                fold: GateUpActBf16 { kernel, .. },
                ..
            }) = *s
            {
                let want: &[GateUpKernel] = match m {
                    1 => &[GateUpKernel::GemvFp8],
                    _ => &[GateUpKernel::GemmFp8, GateUpKernel::GemmCol64Fp8],
                };
                assert!(want.contains(&kernel), "m={m} step {i}: {kernel:?}");
            }
            let expert = match *s {
                CudaStep::ExpertGemm(ExpertGemmBf16 { weight, .. })
                | CudaStep::ExpertGemv(ExpertGemvBf16 { weight, .. })
                | CudaStep::ExpertGemvTcFp8(ExpertGemvBf16 { weight, .. })
                | CudaStep::ExpertGateUpAct(ExpertGateUpActBf16 {
                    gate: ExpertGemvBf16 { weight, .. },
                    ..
                })
                | CudaStep::ExpertGateUpTcFp8(ExpertGateUpTcFp8 {
                    fold:
                        ExpertGateUpActBf16 {
                            gate: ExpertGemvBf16 { weight, .. },
                            ..
                        },
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

/// The quant writes of every prefill tape: each sits right before the fp8 GEMM that reads it,
/// quantizes all `m` rows of that GEMM's `k` columns (an expert GEMM's: all `max_tiles·128`) from a
/// bf16 colour or the MoE scratch's rows, and writes the one quant scratch, the arena's last
/// region: every write starts at its offset, the largest is its size, and it ends the arena. A
/// reader with no write right before it reads the quantization still live in the scratch: the last
/// write's, its bf16 rows unwritten since (an expert up GEMM after its gate; the 27B's GDN a/b
/// group after its qkvz group).
#[test]
fn every_quant_write_sits_before_its_fp8_gemm() {
    assert_eq!(quants(tape_of(1)).len(), DECODE_QUANTS, "m=1: quant writes");
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
            let rows = match tape.steps.get(i + 1) {
                Some(CudaStep::ExpertGemm(g)) => g.max_tiles * 128,
                _ => m,
            };
            assert_eq!((q.q.0, q.rows), (at, rows), "m={m} step {i}: its region, rows");
            assert!(
                q.x.0 < at && q.x.0.is_multiple_of(CUDA_ARENA_ALIGN),
                "m={m} step {i}: x {} is an aligned colour or the MoE scratch's rows before the \
                 scratch",
                q.x.0
            );
            cols.insert(q.cols);
            most = most.max(q.bytes().expect("fits"));
        }
        // The last write, and whether a step wrote its bf16 rows since (a step reads, then writes).
        let mut live: Option<(QuantRowsFp8, bool)> = None;
        for (i, s) in tape.steps.iter().enumerate() {
            if let CudaStep::QuantRowsFp8(q) = *s {
                live = Some((q, false));
                continue;
            }
            if let Some(read) = s.quantized_input() {
                assert!(
                    matches!(live, Some((q, false)) if (q.q, q.rows, q.cols) == read),
                    "m={m} step {i}: a reader with no write before it reads the live quantization"
                );
            }
            if let Some((q, stale)) = &mut live {
                let rows = ArenaSpan {
                    offset: q.x,
                    bytes: u64::from(q.rows) * u64::from(q.cols) * CUDA_ACT_BYTES,
                };
                *stale |= (s.parts().into_iter().flatten()).any(|p| {
                    p.extents().is_none_or(|e| {
                        overlap(e.writes, rows) || e.second_write.is_some_and(|w| overlap(w, rows))
                    })
                });
            }
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

/// The decode tape's fp8 GEMVs on the tensor cores (FP8 S8 T2, /tmp/a-tc-plan.md §1, §2): every
/// CUDA-core fp8 GEMV left is one `gemv_tc_takes` refuses; every quant write quantizes the one
/// token row; every tensor-core gate/up's down, the step right after it, reads its act + quantize's
/// write; and the quant scratch's liveness check holds.
#[test]
fn the_decode_tape_runs_its_fp8_gemvs_on_the_tensor_cores() {
    let tape = tape_of(1);
    assert_eq!(quant_chain(tape.steps), Ok(()), "m=1");
    for (i, s) in tape.steps.iter().enumerate() {
        match *s {
            CudaStep::QuantRowsFp8(q) => assert_eq!(q.rows, 1, "m=1 step {i}: one row"),
            CudaStep::GemvFp8(g) => {
                assert!(!gemv_tc_takes([g.n, 0, 0], g.k, 1), "m=1 step {i}: {g:?}")
            }
            CudaStep::GemvGroupFp8(g) => {
                let ns = [g.first.n, g.second.n, g.third.map_or(0, |t| t.n)];
                assert!(!gemv_tc_takes(ns, g.k, 1), "m=1 step {i}: {ns:?}");
            }
            CudaStep::GateUpAct(g) => panic!("m=1 step {i}: a CUDA-core gate/up {:?}", g.kernel),
            CudaStep::ExpertGateUpAct(_) | CudaStep::ExpertGemv(_) => {
                panic!("m=1 step {i}: a CUDA-core expert GEMV")
            }
            _ => {}
        }
        if let (false, Some(w)) = (matches!(s, CudaStep::QuantRowsFp8(_)), s.quant_write()) {
            let down = tape.steps.get(i + 1).and_then(CudaStep::quantized_input);
            assert_eq!(down, Some((w.q, w.rows, w.cols)), "m=1 step {i}: its down reads it");
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

/// W7: the greedy argmax is a compile-time fact of the decode tape alone. The one-row bucket runs
/// it over its whole logits row (`GREEDY_WIDTH`, the model's vocab); no prefill bucket does.
#[test]
fn only_the_decode_tape_runs_the_greedy_argmax() {
    let got: Vec<(u32, Option<u32>)> = (CUDA_TAPES.iter())
        .map(|t| (t.bucket_m, t.greedy.map(|g| g.width)))
        .collect();
    let want: Vec<(u32, Option<u32>)> = (TAPE_BUCKETS.iter())
        .map(|&m| (m, (m == 1).then_some(GREEDY_WIDTH)))
        .collect();
    assert_eq!(got, want, "{STEM}: the buckets with a greedy argmax");
}

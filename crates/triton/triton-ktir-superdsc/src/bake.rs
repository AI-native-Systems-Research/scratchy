// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! `Vec<EmittedOp>` -> the DIRECTORY `dxp_standalone -d <dir>` reads.
//!
//! # WHY THIS EXISTS
//!
//! `emit_node` returns descriptors as VALUES. SuperDSC is a JSON format and the validation gate
//! reads it off disk, so the only thing between an `EMITTED ops=6` and a verdict from the scheduler
//! is a serializer. `SdscOp` is `ktir-superdsc`'s own `#[derive(Serialize)]` wire type, so nothing
//! here builds a descriptor — it writes the ones the vendored emitter already built.
//!
//! # THE CONTRACT, COPIED FROM THE SHIPPED ARTIFACTS AND FROM THE PRODUCER THAT PASSES ON SILICON
//!
//! Not invented. Two sources, and every rule below is cited to one of them:
//!
//! * `deeptools/dxp/test/test_scatter_1core/` — the whole all-concrete form, in nine lines:
//!   ```text
//!   module {
//!       func.func @sdsc_bundle() {
//!           sdscbundle.sdsc_execute () {sdsc_filename="sdsc_0.json"}
//!           sdscbundle.sdsc_execute () {sdsc_filename="sdsc_1.json"}
//!           return
//!       }
//!   }
//!   ```
//!   No signature, no operands, NO `symbol_ids` attribute at all — and its `sdsc_N.json` carry
//!   absolute addresses (`startAddressCoreCorelet_.data_["[0, 0, 0]"] = "51539607552"`) with no
//!   `isStartAddrSymbolic_` and no `symbolDefinitions_`. That is exactly the shape our emission has:
//!   `resolve_seg_base` resolves every operand to `SEGMENT_OFFSETS[seg] + intra`, a number, so there
//!   is nothing for the host to bind and nothing to give a symbol id to.
//! * `scratchy`'s `lower_subtile_tape_to_superdsc.rs` (`bundle_mlir`, `emit_bundle_mlir`,
//!   `concrete_trips`, `render_dxp_input`) — the same form, from the producer whose bundles are
//!   proven on card. [`dxp_input`] is that function's core with the group walk removed, because one
//!   Triton kernel is one node.
//!
//! ⛔ THE TOP-LEVEL JSON KEY IS `{idx}_{op_name}`, AND THE INDEX IS LOAD-BEARING. scratchy's comment
//! at that site records what dropping it costs: "so dxp's ModuleStitcher can order it — without it:
//! `DtException: expecting a valid entry in core schedule` (ModuleStitcher.cpp:216)". The shipped
//! fixtures agree (`0_identity`, `1_identity`). `idx` is the position in `bundle.mlir`'s execute
//! list, so the key and the filename cannot disagree about which program this is.
//!
//! ⛔ AND A `time > 1` OP IS PRE-UNROLLED, NOT WRAPPED IN AN `scf.for`. [`concrete_trips`] is
//! scratchy's, and its reason is a measured defect: dxp's always-on `LoopUnroll` clones an in-loop
//! `sdsc_execute` with IDENTICAL `symbol_ids` and double-reserves them (`DtException "Symbol already
//! reserved"`, `VariableDefinition.cpp:629`). So trip `t` is a copy with every tiled tensor's start
//! addresses bumped by `t · affine_strides[ti]["out"]`, and the bundle lists N flat executes.
//!
//! ⛔ NO `manifest.json`. The shipped fixtures have none and are accepted; our other bake writes one,
//! and copying that would be cargo-culting a file this consumer never reads.
//!
//! # WHAT A PASS LOOKS LIKE, AND WHY THE SMALL PLAN IS THE POINT
//!
//! MEASURED on the pod (`dxp_standalone -d <dir> -b sentient`, `SENARCH=MPW4`), 2026-09-16:
//!
//! ```text
//!   test_gather_1core      exit 0   spyreCodeDir/   spyrecode.json 11864 B   (has symbols)
//!   test_scatter_1core     exit 0   spyreCodeDir/   spyrecode.json   479 B   (all-concrete)
//!   rmsnorm_granite (this) exit 0   spyreCodeDir/   spyrecode.json   481 B   init_binary 114688 B
//!   test_softmax_1core     exit 1   (none)   "symbol -19 … cannot read back as a definition"
//!   test_loop_exec_1core   exit 1   (none)   "symbol -4 … cannot read back as a definition"
//! ```
//!
//! ⭐ THE 481-BYTE PLAN IS NOT A SHORT COMPILE. An all-concrete bundle has nothing runtime-supplied,
//! so its `JobExecPlan` is one `ComputeOnDevice` with no `DataTransfer … progCorr` step — and
//! `test_scatter_1core`, shipped and accepted, produces the same shape at 479 B. The 11864-byte
//! gather plan is large because it HAS a symbol to correct. Reading our small plan as a failure, or
//! the exit 0 as vacuous, is the mistake this table exists to prevent.
//!
//! ⭐ AND THE GATE DISCRIMINATES, which is the other half. Two shipped fixtures FAIL it, so exit 0 is
//! a verdict rather than a default. Two gap tests on our own output confirm the compile is really
//! reading it: deleting `sdsc_5.json` (still named by `bundle.mlir`) fails with
//! `DtException: std::filesystem::exists(fullPath)` at `SDSCLoading.cpp:61`, so every program is
//! loaded; and perturbing ONE start address in `sdsc_5.json` changes `init_binary.bin`'s md5 at the
//! same byte length, so the job binary is a function of our descriptors and not a canned artifact.

use std::collections::BTreeMap;

use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::wire::SdscOp;

use crate::Error;

/// The files `dxp_standalone -d <dir>` reads, as `(filename, contents)` — `sdsc_N.json` per trip
/// then `bundle.mlir`, flat, in the order the bundle lists them.
pub fn dxp_input(ops: &[EmittedOp]) -> Result<Vec<(String, String)>, Error> {
    // ⛔ REFUSE A KTIR-ONLY OP RATHER THAN REACH ITS DESCRIPTOR. `EmittedOp::op` is `None` for one
    // and `dsc()` PANICS on it ("a KTIR op has no SuperDSC descriptor"). Nothing our bodies emit is
    // KTIR-only today, and a bundle short a program is the failure to see, not to skip past.
    if let Some(e) = ops.iter().find(|e| e.op.is_none()) {
        return Err(Error {
            stage: "bake",
            message: format!(
                "{}: this op carries only its KTIR program and no SuperDSC descriptor, so there is \
                 nothing to write as `sdsc_N.json`. A bundle missing one program is not a bundle \
                 with fewer steps — every `sdsc_execute` the scheduler orders has to exist.",
                e.op_name
            ),
        });
    }
    // Flat, in execute order: every op's trips in sequence. The index is the position here, which is
    // both the filename's number and the json key's prefix, so the two cannot disagree.
    // ⛔ THE TRIP AXIS, STATED PER OP WHEN ASKED. `trips` bumps by the `"out"` stride only, so an op
    // whose tiling advances a DIFFERENT axis would emit N descriptors all pointing at tile 0 — which
    // reads on the card as "the first tile is exact and the rest is wrong". `KTIR_TRIPS=1` prints each
    // op's `time` and the AXES its stride maps actually name, so that is checked rather than assumed.
    if std::env::var_os("KTIR_TRIPS").is_some() {
        for e in ops.iter() {
            let axes: Vec<String> = e
                .affine_strides
                .iter()
                .enumerate()
                .filter(|(_, m)| !m.is_empty())
                .map(|(ti, m)| {
                    let kv: Vec<String> = m.iter().map(|(k, v)| format!("{k}={v}")).collect();
                    format!("t{ti}:[{}]", kv.join(","))
                })
                .collect();
            eprintln!(
                "TRIPS   {:24} time={} tiled={}",
                e.op_name,
                e.time,
                if axes.is_empty() { "-".to_string() } else { axes.join(" ") }
            );
        }
    }
    let flat: Vec<(&EmittedOp, SdscOp)> = ops
        .iter()
        .flat_map(|e| trips(e).into_iter().map(move |t| (e, t)))
        .collect();
    let mut files: Vec<(String, String)> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    for (idx, (e, trip)) in flat.iter().enumerate() {
        let name = format!("sdsc_{idx}.json");
        // `{idx}_{op_name}`, and `idx` is the position in the execute list — see the module doc on
        // ModuleStitcher.cpp:216.
        let key = format!("{idx}_{}", e.op_name);
        let one: BTreeMap<&str, &SdscOp> = BTreeMap::from([(key.as_str(), trip)]);
        // COMPACT. Nothing reads these by eye and indentation was ~2/3 of every byte in the
        // producer's own measurement; the consumer parses either.
        let json = serde_json::to_string(&one).map_err(|err| Error {
            stage: "bake",
            message: format!("{}: cannot serialize its descriptor: {err}", e.op_name),
        })?;
        files.push((name.clone(), json));
        names.push(name);
    }
    if names.is_empty() {
        return Err(Error {
            stage: "bake",
            message: "no descriptors to bake — a bundle with no `sdsc_execute` has nothing for the \
                      scheduler to read, and an empty directory would look like a pass"
                .to_string(),
        });
    }
    files.push(("bundle.mlir".to_string(), bundle_mlir(&names)));
    Ok(files)
}

/// Write [`dxp_input`] into `dir`, creating it. FLAT — the contract is two kinds of file and no
/// subdirectory.
pub fn write_dir(dir: &std::path::Path, ops: &[EmittedOp]) -> Result<Vec<String>, Error> {
    let files = dxp_input(ops)?;
    std::fs::create_dir_all(dir).map_err(|e| Error {
        stage: "bake",
        message: format!("cannot create {}: {e}", dir.display()),
    })?;
    for (name, contents) in &files {
        std::fs::write(dir.join(name), contents).map_err(|e| Error {
            stage: "bake",
            message: format!("cannot write {}: {e}", dir.join(name).display()),
        })?;
    }
    Ok(files.into_iter().map(|(n, _)| n).collect())
}

/// One flat `sdsc_execute` per program, no signature, no operands, no `symbol_ids` — byte-identical
/// to `test_scatter_1core/bundle.mlir` and to scratchy's `bundle_mlir`, tabs included.
fn bundle_mlir(sdsc_filenames: &[String]) -> String {
    let mut body = String::new();
    for f in sdsc_filenames {
        body.push_str(&format!(
            "    sdscbundle.sdsc_execute () {{sdsc_filename=\"{f}\"}}\n"
        ));
    }
    format!("module {{\n  func.func @sdsc_bundle() {{\n{body}    return\n  }}\n}}\n")
}

/// One [`EmittedOp`]'s CONCRETE per-trip descriptors — scratchy's `concrete_trips`.
///
/// A `time == 1` op yields its descriptor. A `time == N` op yields N copies, trip `t` with every
/// TILED tensor's start addresses bumped by `t · affine_strides[ti]["out"]` (the per-trip HBM
/// advance) and every non-tiled tensor's left alone. Addresses stay concrete, which is what keeps
/// the bundle free of the `scf.for` that would make dxp's `LoopUnroll` double-reserve symbols.
fn trips(e: &EmittedOp) -> Vec<SdscOp> {
    if e.time <= 1 {
        return vec![e.dsc().clone()];
    }
    (0..e.time)
        .map(|t| {
            let mut op = e.dsc().clone();
            for dsc_map in op.dscs_.iter_mut() {
                for dsc in dsc_map.values_mut() {
                    for node in dsc.scheduleTree_.iter_mut() {
                        let ti = node.ldsIdx_ as usize;
                        let Some(stride) = e.affine_strides.get(ti).and_then(|m| m.get("out"))
                        else {
                            continue; // non-tiled tensor — base address unchanged.
                        };
                        let bump = t as i64 * *stride;
                        for v in node.startAddressCoreCorelet_.data_.values_mut() {
                            // The address is a STRING on the wire (the torch-spyre frontend's form).
                            // A value that does not parse is a descriptor this bump cannot express,
                            // and silently treating it as 0 would move the tensor to segment 0 — so
                            // it is left ALONE rather than zeroed.
                            if let Ok(base) = v.parse::<i64>() {
                                *v = (base + bump).to_string();
                            }
                        }
                    }
                }
            }
            op
        })
        .collect()
}

/// ⭐⭐⭐ THE ONE FILE A LAUNCH NEEDS AND `dxp_standalone` DOES NOT PRODUCE — each tensor's
/// `(segment, offset, size)`.
///
/// ⛔ WHY IT CANNOT BE DERIVED FROM THE DIRECTORY. `superdsc_exec::Executor::prepare` allocates ONE
/// device region per tensor segment and keeps `tensor_allocs` POSITIONAL — entry *i* backs the
/// program's logical segment *i*, with a 128 B placeholder even for an empty segment — so a launcher
/// has to know which segment each tensor lives in and at what offset inside it. The descriptors in
/// the directory carry ADDRESSES whose segment bases are symbolic until the launch's correction flits
/// patch them, so the map is not recoverable from them without re-deriving the emitter's own
/// addressing scheme. It IS, however, already computed here: `layout::for_node` builds it before any
/// op is emitted, and this writes that same value out rather than a second derivation of it.
///
/// The `spyrecode.json` beside it carries the rest (the program allocation, the init image, the
/// corrections and `job_bin_ptr`), so directory + this file is a complete launch input.
pub fn write_placements(
    dir: &std::path::Path,
    layout: &ktir_superdsc::placement::BundleLayout,
) -> Result<(), Error> {
    use std::fmt::Write as _;
    // Order-independent: `write_dir` also creates the directory, and which of the two runs first is
    // not a property this should depend on.
    std::fs::create_dir_all(dir).map_err(|e| Error {
        stage: "bake",
        message: format!("cannot create {}: {e}", dir.display()),
    })?;
    let mut s = String::new();
    // ⛔⛔⛔ THE SYNTHETICS MUST BE IN THE SEGMENT SIZE, and `layout.segment_bytes` DELIBERATELY
    // EXCLUDES THEM — its own comment says so: they are bump-allocated during lowering and "do not
    // exist yet" when the placement high-water is computed. The rmsnorm/silu decompositions mint
    // intermediates (`t{id}_sq`, `t{id}_silu`, ...) into the Intermediate segment DURING emission, so
    // a launch sized from the placements alone gives that segment a region the program stores PAST.
    //
    // MEASURED on the card, `rmsnorm_granite` with seg3 at the placement high-water: the launch is
    // accepted and the device raises a memory-access error on a store —
    // `HMI=STORE addr=0xd00 (26 flits) syndrome=0x1` — which is visible only once a tracing subscriber
    // exists for flex's completion thread. Without one it is a bare `sync rc=-1`.
    let synth = layout.synth.borrow();
    let mut seg_bytes = layout.segment_bytes;
    let intermediate = ktir_superdsc::placement::SegRole::Intermediate.segment();
    if intermediate < seg_bytes.len() {
        seg_bytes[intermediate] = seg_bytes[intermediate].max(synth.next);
    }
    s.push_str("{\n  \"segment_bytes\": [");
    for (i, b) in seg_bytes.iter().enumerate() {
        let _ = write!(s, "{}{}", if i == 0 { "" } else { ", " }, b);
    }
    s.push_str("],\n  \"places\": [\n");
    for (i, (tid, p)) in layout.placements.iter().enumerate() {
        let _ = writeln!(
            s,
            "    {{\"tid\": {}, \"role\": \"{:?}\", \"segment\": {}, \"bank\": {}, \"offset\": {}, \"size\": {}}}{}",
            tid,
            p.role,
            p.segment,
            p.bank,
            p.offset,
            p.size,
            if i + 1 == layout.placements.len() { "" } else { "," }
        );
    }
    s.push_str("  ],\n  \"synth\": [\n");
    for (i, (name, off)) in synth.map.iter().enumerate() {
        let _ = writeln!(
            s,
            "    {{\"name\": \"{}\", \"segment\": {}, \"offset\": {}, \"size\": {}}}{}",
            name,
            intermediate,
            off,
            synth.sizes.get(name).copied().unwrap_or(0),
            if i + 1 == synth.map.len() { "" } else { "," }
        );
    }
    // ⛔⛔⛔ THE SCALARMUL SCALES, AND WITHOUT THEM A DECODER SILENTLY COMPUTES ITS OWN RESIDUAL.
    //
    // `Program::ScalarMul` does not put its constant in the descriptor: the `i`-th DISTINCT scale is a
    // `[1, 1]` fp16 at the reserved tid `scalarmul_scale_tid(i)` that the HOST binds, and the
    // pointwise `mul` reads it broadcast. A `places` row exists for it — so a launcher sees its
    // segment, offset and 2-byte size — but the row cannot say WHAT GOES IN IT, and the value is a
    // kernel constexpr the directory records nowhere else. Every unbound scale is therefore whatever
    // the segment held, and a zeroed segment makes it 0.
    //
    // MEASURED on card, `decoder_layer_one_flat` with all four scales unbound: the readback is the
    // INPUT `x` at `within_2pct = 1.0000`, max|err| 0.00195 — because `x + attn·RM` and `+ mlp·RM` both
    // scale by `RM = 0` and only the residual survives. It is byte-IDENTICAL with every one of the
    // seven weight tensors set to zero, and the weights themselves read back off the card correctly to
    // one DL16 ULP. So the bundle was right, the binding was right, and the answer was the identity —
    // a clean exit 0 with a plausible dynamic range, which is exactly the failure this file's own
    // header says a launch map exists to prevent.
    //
    // The tid is computed by the SAME function the emitter used (`scalarmul_scale_tid`) rather than
    // restated as `BASE - i`, so the two cannot drift apart.
    s.push_str("  ],\n  \"scalarmul_scales\": [\n");
    let scales = &layout.scalarmul_scales;
    for (i, v) in scales.iter().enumerate() {
        let _ = writeln!(
            s,
            "    {{\"tid\": {}, \"index\": {}, \"value\": {:?}}}{}",
            ktir_superdsc::reserved_tids::scalarmul_scale_tid(i),
            i,
            v,
            if i + 1 == scales.len() { "" } else { "," }
        );
    }
    s.push_str("  ]\n}\n");
    // ⛔⛔⛔ THE fp8 W8A8 CLAMP CONSTS, same gap as the scalarmul scales above. The fp8
    // activation-quantize chain reads three `[1, stick]` fp16 reserved consts — ±448 (E4M3
    // clamp bounds) and 1/448 (the amax reciprocal) — placed by `layout::for_regions` iff
    // a region states `is_fp8`. A `places` row says WHERE they live but not WHAT GOES IN
    // THEM, and scratchy's own producer records the cost of leaving them unbound: "clamp
    // consts never bound ⇒ garbage quant" (every clamp collapses to `min(x, 0)` and the
    // reciprocal to `recip(0) = inf`). The VALUES are the device's own E4M3 constants,
    // fixed by the format, not by any model or launch — so they are written out here for
    // the launcher to bind, the same treatment `scalarmul_scales` gets.
    if layout
        .placements
        .contains_key(&ktir_superdsc::reserved_tids::FP8_INV448_TID)
    {
        use ktir_superdsc::reserved_tids::{
            FP8_NEG448_TID, FP8_POS448_TID, FP8_INV448_TID,
        };
        // Rewrite the closing brace to append the block, keeping the JSON valid.
        s.truncate(s.len() - "  ]\n}\n".len());
        s.push_str("  ],\n  \"fp8_consts\": [\n");
        let rows = [
            (FP8_POS448_TID, 448.0),
            (FP8_NEG448_TID, -448.0),
            (FP8_INV448_TID, 1.0 / 448.0),
        ];
        for (i, (tid, v)) in rows.iter().enumerate() {
            let _ = writeln!(
                s,
                "    {{\"tid\": {}, \"value\": {}}}{}",
                tid,
                v,
                if i + 1 == rows.len() { "" } else { "," }
            );
        }
        s.push_str("  ]\n}\n");
    }
    std::fs::write(dir.join("placements.json"), s).map_err(|e| Error {
        stage: "bake",
        message: format!(
            "cannot write {}: {e}",
            dir.join("placements.json").display()
        ),
    })
}

// SPDX-License-Identifier: Apache-2.0
//! THE COMPOSITE ARGMAX CHAIN'S FIRST EMISSION — the probe matrix for #603 PR3a
//! (`emit::argmax`'s module doc states the laws; this file pins the descriptor's own side
//! of them and stages the probe GROUP — `bundle.mlir` + one `sdsc_N.json` per link — for
//! the pod's cardless `dxp_standalone -d <dir>` rc=0 run, the real bake driver's own
//! contract).
//!
//! What is asserted HERE (the descriptor's side):
//!   • the chain assembles at granite-3.3-2b's own geometry (vocab 49152, hidden 2048) as
//!     EIGHT ops in dependency order, with the opFuncNames the vendor table resolves
//!     (`max`/`equal`/`dl16tofp32`/`mul`/`max`/`sub`/`fp32toint32`/`identity`);
//!   • the FIRST op reads the staging form — the logits operand sticked on `out` (vocab),
//!     the lm_head's own output axis — NOT any Kᵀ-restuck token-stick form;
//!   • the gather's index operand is SENUINT32 at wordLength 4 (the ingestion law);
//!   • DETERMINISM, which is the emission-side half of the content-key guard: the mint's
//!     signature cannot express a cap rung (no rung parameter exists), so two assemblies
//!     at the same geometry render byte-identical descriptors — a per-rung offset cannot
//!     enter this chain through this door. The ladder-side assert lands with the wiring.
//!   • the refusals: an UNTIED config is refused with the named reason (llama cannot take
//!     this path), and a rows>1 geometry is refused as second-wave work.
//!
//! What is NOT asserted here: anything the compiler decides. The staged group goes to the
//! pod; `dxp_standalone`'s selection (the table entries) and the scheduler are the gates
//! that close there. (`ddc_standalone -s` is NOT that gate: it demands the hand-written
//! FIXTURE form — chip descriptors the real pipeline fills in code — so even the real
//! dump files abort in it; reproduced during this leg.)
use ktir_superdsc::emit::argmax::{
    ArgmaxGeom, ArgmaxSlots, TiedLmHeadEmbedding, assemble_argmax, try_assemble_argmax,
};

/// granite-3.3-2b's own vocab/hidden — the PR3a gate model's geometry. granite-3.1's
/// 49155 (`PAD_VOCAB` below) is the pad-lane leg's: three lanes past the last stick.
const VOCAB: u32 = 49152;
const HIDDEN: u32 = 2048;
const PAD_VOCAB: u32 = 49155;

fn slots() -> ArgmaxSlots<'static> {
    ArgmaxSlots {
        logits: "s550_logits",
        biased: "s559_biased",
        bias: "c_argmax_bias",
        m: "s551_m",
        mask: "s552_mask",
        mask32: "s553_mask32",
        ramp: "c_argmax_ramp",
        v_const: "c_argmax_v",
        w: "s554_w",
        j: "s555_j",
        idx32: "s556_idx32",
        idx_u32: "s557_idx",
        table: "w_lm_head_emb",
        row: "s558_row",
    }
}

fn chain() -> Vec<serde_json::Value> {
    let mut sid = -1i64;
    let tied = TiedLmHeadEmbedding::from_tied_config(true).expect("granite ties");
    assemble_argmax(
        ArgmaxGeom::k1(VOCAB, HIDDEN),
        &slots(),
        tied,
        &mut sid,
        None,
    )
    .into_iter()
    .map(|e| serde_json::to_value(&e.op).expect("serializable"))
    .collect()
}

/// The pad-lane form's chain — the same assembly at granite-3.1's own vocab.
fn pad_chain() -> Vec<serde_json::Value> {
    let mut sid = -1i64;
    let tied = TiedLmHeadEmbedding::from_tied_config(true).expect("granite ties");
    assemble_argmax(
        ArgmaxGeom::k1(PAD_VOCAB, HIDDEN),
        &slots(),
        tied,
        &mut sid,
        None,
    )
    .into_iter()
    .map(|e| serde_json::to_value(&e.op).expect("serializable"))
    .collect()
}

/// The emitted dsc body (`dscs_[0]`'s first value — the op's own name is its key) — the
/// level `N_`, the layouts and the computeOp row live at.
fn body(d: &serde_json::Value) -> &serde_json::Value {
    let dscs = d
        .get("dscs_")
        .and_then(|v| v.as_array())
        .expect("dscs_ array");
    let dsc = dscs
        .first()
        .expect("one dsc")
        .as_object()
        .expect("dsc object");
    dsc.values().next().expect("the op's own body")
}

/// The opFuncName row of one emitted op — the field the vendor's selection table keys on.
fn op_func_name(d: &serde_json::Value) -> &str {
    body(d)
        .pointer("/computeOp_/0/opFuncName")
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| panic!("no opFuncName in {}", body(d)))
}

#[test]
fn the_chain_is_eight_ops_in_dependency_order() {
    let chain = chain();
    let names: Vec<&str> = chain.iter().map(op_func_name).collect();
    assert_eq!(
        names,
        vec![
            "max",
            "equal",
            "dl16tofp32",
            "mul",
            "max",
            "sub",
            "fp32toint32",
            "identity",
        ],
        "the composite argmax is exactly opus's chain; any other order or member is a bug"
    );
}

/// The PAD-LANE form at granite-3.1's own vocab (49155 = 768·64 + 3): NINE ops, the add
/// first, every later link at the stick-padded staging width (49216), the whole-stick
/// chain's kernels otherwise unchanged.
#[test]
fn the_pad_lane_chain_is_nine_ops_with_the_add_first() {
    let chain = pad_chain();
    let names: Vec<&str> = chain.iter().map(op_func_name).collect();
    assert_eq!(
        names,
        vec![
            "add",
            "max",
            "equal",
            "dl16tofp32",
            "mul",
            "max",
            "sub",
            "fp32toint32",
            "identity",
        ],
        "the pad-lane composite is the landed chain with the bias-add as link 0"
    );
    // The whole chain runs at the stick-padded staging width — the reduce's own N_
    // states 49216, never the real 49155 (the dxp scheduler rejects sub-stick tiles).
    let out = body(&chain[1])
        .get("N_")
        .and_then(|n| n.get("out_"))
        .and_then(|v| v.as_u64())
        .expect("out_ in N_ of the pad-lane max");
    assert_eq!(
        out,
        u64::from(PAD_VOCAB.div_ceil(64) * 64),
        "the pad-lane links run at the stick-padded staging width"
    );
    // And the add's own width: fp16 at the padded staging width — the mb-broadcast
    // operand form the mul's ramp rides, at the link's own stick extent.
    let add_out = body(&chain[0])
        .get("N_")
        .and_then(|n| n.get("out_"))
        .and_then(|v| v.as_u64())
        .expect("out_ in N_ of the add");
    assert_eq!(
        add_out,
        u64::from(PAD_VOCAB.div_ceil(64) * 64),
        "the bias-add runs at the stick-padded staging width too"
    );
}

/// The two chains differ ONLY by the pad-lane links: the pad-lane form's links 1..8
/// resolve the SAME vendor kernels as the whole-stick chain's eight (the landed chain,
/// kernel for kernel — only the widths move to the padded staging), and the whole-stick
/// chain's own widths are the real vocab (the `N_` asserts above pin each side's).
#[test]
fn the_whole_stick_chain_is_the_pad_lane_one_minus_the_add() {
    let plain = chain();
    let pad = pad_chain();
    assert_eq!(
        plain.len() + 1,
        pad.len(),
        "the pad-lane form adds exactly link 0"
    );
    let kernels = |c: &[serde_json::Value]| {
        c.iter()
            .map(|d| op_func_name(d).to_string())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        kernels(&plain),
        kernels(&pad)[1..],
        "the landed kernels, verbatim, in both forms — the pad-lane form adds the add and \
         widens the links; it changes no other member"
    );
}

#[test]
fn the_first_max_reads_the_staging_form_not_the_kt_form() {
    let chain = chain();
    let max16 = &chain[0];
    // The logits operand is sticked on `out` — the vocab axis — matching the lm_head's own
    // output (`matmul/dims.rs`: out = N, the stick). The Kᵀ-restuck form would stick `y`
    // (tokens) instead; that form is the padding this chain exists to avoid.
    let text = serde_json::to_string(max16).unwrap();
    assert!(
        text.contains("\"out\"") && !text.contains("\"stickDimOrder_\":[\"y\"]"),
        "the max's operands must carry the vocab-sticked staging form"
    );
    // vocab rides the stick at the full width: N_ carries out_ = 49152 (the stick extent).
    let out = body(max16)
        .get("N_")
        .and_then(|n| n.get("out_"))
        .and_then(|v| v.as_u64())
        .expect("out_ in N_");
    assert_eq!(
        out, VOCAB as u64,
        "the vocab axis is the reduce's stick axis"
    );
}

#[test]
fn the_gathers_index_is_senuint32_at_four_bytes() {
    let chain = chain();
    let gather = &chain[7];
    let text = serde_json::to_string(gather).unwrap();
    assert!(
        text.contains("SENUINT32"),
        "the embedding gather's index must be SENUINT32 (the ingestion law)"
    );
    assert!(
        text.contains("\"wordLength\":4") || text.contains("\"wordLength_\":4"),
        "the index words are 4 bytes"
    );
}

#[test]
fn two_assemblies_render_byte_identical_descriptors() {
    // The content-key guard's emission-side half: the mint's signature has no rung
    // parameter, so nothing rung-dependent can enter a descriptor through this door, and
    // the rendering is deterministic — the same geometry twice is byte-identical, which is
    // what the bake's content key hashes. A per-rung offset would break this equality
    // only by first breaking the signature's own shape.
    let a = chain();
    let b = chain();
    let sa: Vec<String> = a.iter().map(|v| v.to_string()).collect();
    let sb: Vec<String> = b.iter().map(|v| v.to_string()).collect();
    assert_eq!(sa, sb, "the chain must render deterministically");
}

#[test]
fn an_untied_model_is_refused_with_the_named_reason() {
    let err = TiedLmHeadEmbedding::from_tied_config(false)
        .expect_err("untied must refuse")
        .to_string();
    assert!(
        err.contains("untied") && err.contains("staging fallback"),
        "the refusal names the cause: {err}"
    );
}

#[test]
fn a_wide_geometry_is_refused_as_second_wave() {
    let mut sid = -1i64;
    let tied = TiedLmHeadEmbedding::from_tied_config(true).unwrap();
    let err = match try_assemble_argmax(
        ArgmaxGeom {
            rows: 2,
            vocab: VOCAB,
            hidden: HIDDEN,
        },
        &slots(),
        tied,
        &mut sid,
        None,
    ) {
        Ok(_) => panic!("rows>1 must refuse in the first cut"),
        Err(e) => e,
    };
    assert!(
        err.contains("w>1") || err.contains("rows = 2"),
        "the refusal names the second-wave boundary: {err}"
    );
}

#[test]
fn stage_the_probe_jsons_for_the_pod() {
    stage_group("argmax-probes", &chain());
    // The pad-lane group too — the NINE-link form at granite-3.1's own vocab. The pod
    // leg runs `dxp_standalone -d <dir>` over both groups; rc=0 on the pad-lane one is
    // the new link's own gate (the add is the pointwise family's fp16 mb-broadcast form,
    // the mul's shape at fp16 precision — the one descriptor this form adds).
    stage_group("argmax-probes-padlane", &pad_chain());
}

/// Stage one probe GROUP — the dump's own dir contract, which is what `dxp_standalone
/// -d <dir>` (the real bake's compile driver) reads: `bundle.mlir`, one
/// `sdscbundle.sdsc_execute` per link in dependency order, plus one `sdsc_N.json` per
/// link in the dump's own wire form — the sdsc WRAPPED IN ITS DSC'S NAME
/// (`importJsonObj` demands every top-level key's value carry `dscs_`; the bare sdsc
/// object is refused as "No dsc in sdsc input") and WITHOUT `target_` (the driver's
/// `-b`/default supplies the target; none of the real dump files carry it). The pod leg
/// runs `dxp_standalone -d <dir>` on the whole group — rc=0 is the gate, which is
/// strictly stronger than `ddc_standalone -s`'s selection: ddc_standalone demands the
/// hand-written FIXTURE form (chip descriptors `B_`/`ChipD_`/`CoreD_`/… that the real
/// pipeline fills in code), so even the real dump files abort there with a `map::at`
/// — reproduced and pinned during this leg. The run logs the pod's compile load first
/// (the quiet-pod law).
fn stage_group(dir_name: &str, chain: &[serde_json::Value]) {
    let dir = std::env::temp_dir().join(dir_name);
    std::fs::create_dir_all(&dir).expect("probe dir");
    let mut bundle = String::from("module {\n  func.func @sdsc_bundle() {\n");
    for (i, op) in chain.iter().enumerate() {
        let sdsc = op;
        let dsc_name = sdsc["dscs_"][0]
            .as_object()
            .expect("one named dsc")
            .keys()
            .next()
            .expect("the dsc's own name")
            .clone();
        let mut wrapped = serde_json::Map::new();
        wrapped.insert(dsc_name, sdsc.clone());
        let path = dir.join(format!("sdsc_{i}.json"));
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&serde_json::Value::Object(wrapped)).expect("serializable"),
        )
        .expect("write probe json");
        bundle.push_str(&format!(
            "    sdscbundle.sdsc_execute () {{sdsc_filename=\"sdsc_{i}.json\"}}\n"
        ));
    }
    // `return` — the func body's terminator (mlir refuses a block without one; the real
    // bundles close with it).
    bundle.push_str("    return\n  }\n}\n");
    std::fs::write(dir.join("bundle.mlir"), bundle).expect("write bundle.mlir");
    eprintln!("argmax probe group staged at {}", dir.display());
}

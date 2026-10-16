// SPDX-License-Identifier: Apache-2.0
//! The split-KV chunk body's emission, from the shipped `assemble_attn` at granite-3.1-2b's geometry.

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::assemble_attn;
use ktir_superdsc::sdsc_abstract::{AttnGeometry, ChunkScratch, PagedKvPool, attn_bundle_rows};

const NQH: u32 = 32;
const NKVH: u32 = 8;
const HD: u32 = 64;
const CAP: u32 = 256;

fn emit_chunk(pages: u32) -> Vec<(String, serde_json::Value)> {
    named(emit_chunk_ops(pages))
}

fn named(ops: Vec<ktir_superdsc::emit::EmittedOp>) -> Vec<(String, serde_json::Value)> {
    ops.iter()
        .map(|e| {
            let v = serde_json::to_value(&e.op).unwrap();
            let name = v["dscs_"][0]
                .as_object()
                .and_then(|m| m.keys().next().cloned())
                .expect("one named dsc per emitted op");
            (name, v)
        })
        .collect()
}

fn emit_chunk_ops(pages: u32) -> Vec<ktir_superdsc::emit::EmittedOp> {
    emit_chunk_ops_at(pages, 1)
}

/// A `pages`-page chunk at `mq` query rows per head: 1 is the one-request decode rung, more is a
/// prompt chunk over its resident prefix (rows are one sequence's positions, not requests).
fn emit_chunk_ops_at(pages: u32, mq: u32) -> Vec<ktir_superdsc::emit::EmittedOp> {
    let geom = AttnGeometry::<NQH, NKVH, HD>::minted();
    let rows = if mq == 1 {
        attn_bundle_rows(geom, 1, true).expect("the one-request decode rung")
    } else {
        attn_bundle_rows(geom, mq, false).expect("a prompt chunk's rows")
    };
    let chunk = ChunkScratch::of_chunk(PagedKvPool::new(NKVH as usize, HD as usize), pages)
        .expect("a chunk granite-2b admits");
    let mut sym = 0i64;
    assemble_attn(
        0,
        geom,
        rows,
        CAP,
        CAP,
        "t_qs",
        "t_new_k",
        "t_new_v",
        "t_kct",
        "t_vc",
        "t_pmask",
        "t_cmask",
        Some("t_kvbi"),
        Some(chunk),
        ktir_superdsc::place::PlaceId::Act(900),
        false,
        false,
        &mut sym,
        None,
    )
    .unwrap_or_else(|e| panic!("pages={pages}: assemble_attn refused: {}", e.0))
}

/// ⭐ THE WHOLE POINT: one launch attends every page, with the SAME compute ops at every chunk width —
/// only the gather's copies (one per index stick) grow with it.
#[test]
fn the_chunk_op_count_is_independent_of_its_width() {
    let counts: Vec<(u32, usize)> = ChunkScratch::LADDER
        .iter()
        .map(|&j| {
            let compute = emit_chunk(j)
                .iter()
                .filter(|(n, _)| !n.starts_with("attn_cgkt_") && !n.starts_with("attn_cgv_"))
                .count();
            (j, compute)
        })
        .collect();
    assert!(
        counts.windows(2).all(|w| w[0].1 == w[1].1),
        "chunk compute ops vary with the chunk width: {counts:?}"
    );
}

/// ⛔ THE CHUNK'S TWO REDUCTIONS ARE PER-ROW, ON STICK-BLOCKED OPERANDS, AT EVERY WIDTH: each row's max
/// (the stabilizer — a group max underflows `o`/`l` in fp16) and each row's denominator are stick-major
/// `[gqa, slots]` reduces, one per kv head. Both data operands walk `-1` (dxp reconstructs the
/// stick-blocked layout from the global shape) — every per-lane page-axis form with a DECLARED row-major
/// walk mis-walked part of each core's rows on card.
#[test]
fn the_chunk_reductions_are_per_row_on_stick_blocked_operands() {
    for &j in &ChunkScratch::LADDER {
        let slots = u64::from(j * 256);
        for (stem, rows, cols) in [("attn_cbmax_g", 4, slots), ("attn_cbsum_g", 4, slots)] {
            let ops: Vec<(String, serde_json::Value)> = emit_chunk(j)
                .into_iter()
                .filter(|(n, _)| n.starts_with(stem))
                .collect();
            assert_eq!(ops.len(), 8, "{stem}* at {j} page(s): one per kv head");
            for (name, v) in &ops {
                let body = &v["dscs_"][0][name];
                let n = &body["N_"];
                assert_eq!(
                    (n["mb_"].as_u64(), n["out_"].as_u64()),
                    (Some(rows), Some(cols)),
                    "{name} at {j} page(s): N_ = {n}"
                );
                let walk: Vec<i64> = body["scheduleTree_"][0]["maxDimSizes_"]
                    .as_array()
                    .map(|a| a.iter().map(|d| d.as_i64().unwrap_or(0)).collect())
                    .unwrap_or_default();
                assert!(
                    walk.iter().all(|&d| d == -1),
                    "{name} at {j} page(s): the data walks {walk:?}, not the reconstructed layout"
                );
            }
        }
    }
}

/// ⭐ AT DECODE THE VALUE LEG's CONTRACTION IS DIVIDED ACROSS EVERY CORE, AND A PROMPT CHUNK's IS NOT. The
/// one-request value matmul's output is one stick of `gqa` rows, so the planner alone would split only
/// those 4 rows — 4 cores, each contracting the whole chunk and reading the kv head's whole V. The
/// contraction-first split hands the cores whole-stick runs of the chunk instead (dxp PSUM-accumulates
/// the partials into the shared output) — as many cores as whole sticks allow: the largest divisor of the
/// chunk's `4·pages` sticks within the 32-core budget (8 at two pages, 32 from eight). The rows stay whole.
/// A prompt chunk's value leg has `gqa·mq` rows to divide and keeps the planner's split, contraction whole.
#[test]
fn the_decode_value_leg_divides_its_contraction_across_the_cores() {
    let splits = |v: &serde_json::Value| {
        let s = &v["numWkSlicesPerDim_"];
        (
            s["in"].as_u64().unwrap_or(1),
            s["mb"].as_u64().unwrap_or(1),
            v["numCoresUsed_"].as_u64(),
        )
    };
    for &j in &ChunkScratch::LADDER {
        let sticks = u64::from(j * 256 / 64);
        let most = (1..=32u64.min(sticks))
            .rev()
            .find(|s| sticks.is_multiple_of(*s))
            .unwrap_or(1);
        for (name, v) in emit_chunk(j)
            .iter()
            .filter(|(n, _)| n.starts_with("attn_cov_g"))
        {
            assert_eq!(
                splits(v),
                (most, 1, Some(most)),
                "{name} at {j} page(s): split {}",
                v["numWkSlicesPerDim_"]
            );
        }
        for (name, v) in named(emit_chunk_ops_at(
            j,
            PagedKvPool::PREFILL_CHUNK_SLOTS as u32,
        ))
        .iter()
        .filter(|(n, _)| n.starts_with("attn_cov_g"))
        {
            assert_eq!(
                splits(v).0,
                1,
                "{name} at {j} page(s), prompt chunk: split {}",
                v["numWkSlicesPerDim_"]
            );
        }
    }
}

/// ⭐⭐⭐ A PROMPT CHUNK'S PREFIX IS THE SAME CHUNK OVER EVERY ROW: at the shipped prompt-chunk width
/// (`PREFILL_CHUNK_SLOTS` rows per head, 256 → `gqa·256 = 1024` query rows per kv head) each kv head's
/// score, row max and denominator span all of its rows over the whole chunk, the op count outside the
/// gather does not grow with the chunk, and nothing is a fold op — every row of every head in one
/// launch, nothing per page.
#[test]
fn a_prompt_chunk_attends_every_row_of_every_head_in_one_pass() {
    const MQ: u32 = PagedKvPool::PREFILL_CHUNK_SLOTS as u32;
    let gqa_rows = u64::from(NQH / NKVH * MQ);
    // Four op families grow with the chunk by design, and each is asserted on its own: the gather's
    // copies (one per index stick), the value leg's K-parts (one PV per `VALUE_PART_PAGES` pages, below),
    // the probabilities' slot parts (`a_wide_prompt_chunk_cuts_its_probabilities_into_untiled_parts`) and
    // the row reductions' parts with their combines (the per-kv-head coverage below). Nothing else may.
    let non_gather = |ops: &[(String, serde_json::Value)]| {
        ops.iter()
            .filter(|(n, _)| {
                !n.starts_with("attn_cgkt_")
                    && !n.starts_with("attn_cgv_")
                    && !n.starts_with("attn_cov_g")
                    && !n.starts_with("attn_covsum_")
                    && !n.starts_with("attn_cesub_g")
                    && !n.starts_with("attn_cexp_g")
                    && !n.starts_with("attn_cbmax_")
                    && !n.starts_with("attn_cbsum_")
            })
            .count()
    };
    let narrowest = non_gather(&named(emit_chunk_ops_at(ChunkScratch::LADDER[0], MQ)));
    for &j in &ChunkScratch::LADDER {
        let raw = emit_chunk_ops_at(j, MQ);
        let trips: std::collections::BTreeMap<String, u64> = raw
            .iter()
            .map(|e| (e.op_name.clone(), u64::from(e.time)))
            .collect();
        let ops = named(raw);
        let slots = u64::from(j * 256);
        // Each kv head's row max and denominator cover ALL of its rows over the WHOLE chunk, in ONE trip
        // each (the emitter refuses a time-tiled chunk op) — as one op, or as the equal parts it is cut
        // into on a wide chunk at 256 rows.
        for stem in ["attn_cbmax_g", "attn_cbsum_g"] {
            for kvh in 0..NKVH {
                let mine = format!("{stem}{kvh}_");
                let reds: Vec<&(String, serde_json::Value)> =
                    ops.iter().filter(|(n, _)| n.starts_with(&mine)).collect();
                assert!(!reds.is_empty(), "{mine}* at {j} page(s)");
                let mut covered = 0u64;
                for (name, v) in reds {
                    let n = &v["dscs_"][0][name]["N_"];
                    assert_eq!(trips.get(name).copied(), Some(1), "{name}: one trip");
                    assert_eq!(n["mb_"].as_u64(), Some(gqa_rows), "{name}: N_ = {n}");
                    covered += n["out_"].as_u64().unwrap_or(0);
                }
                assert_eq!(
                    covered, slots,
                    "{mine}* at {j} page(s) cover {covered} slots"
                );
            }
        }
        for (name, v) in ops.iter().filter(|(n, _)| n.starts_with("attn_csc_g")) {
            assert_eq!(
                v["dscs_"][0][name]["N_"]["mb_"].as_u64(),
                Some(gqa_rows),
                "{name} at {j} page(s): the score covers every query row of its kv head"
            );
        }
        assert_eq!(
            non_gather(&ops),
            narrowest,
            "{j} page(s): the prompt chunk's ops outside the gather grow with the chunk"
        );
        // ⛔ THE VALUE LEG IS CUT ALONG K, EVERY PART CONTRACTS AT MOST `VALUE_PART_PAGES` PAGES — the
        // whole-chunk PV at 384 rows is what dxp refused to verify — and each kv head's parts cover its
        // chunk exactly once. (A prompt chunk's value parts are its probability parts, which stay in LX.)
        let mut parts = 0u32;
        for kvh in 0..NKVH {
            let mine = format!("attn_cov_g{kvh}_");
            let mut covered = 0u64;
            let mut n = 0u32;
            for (name, v) in ops.iter().filter(|(n, _)| n.starts_with(&mine)) {
                let k = v["dscs_"][0][name]["N_"]["in_"].as_u64().unwrap_or(0);
                assert!(
                    k > 0 && k <= u64::from(ChunkScratch::VALUE_PART_PAGES * 256),
                    "{name} at {j} page(s): contracts {k} slots"
                );
                covered += k;
                n += 1;
            }
            assert_eq!(
                covered, slots,
                "{mine}* at {j} page(s) contract {covered} slots"
            );
            parts = n;
        }
        assert_eq!(
            ops.iter().any(|(n, _)| n.starts_with("attn_covsum_")),
            parts.is_multiple_of(2),
            "{j} page(s): the copy home exists exactly when the part count is even"
        );
        for e in emit_chunk_ops_at(j, MQ) {
            assert!(
                !e.kv_page_fold,
                "{} at {j} page(s) is a fold op — the chunk launches once",
                e.op_name
            );
        }
    }
}

/// ⛔ A 256-ROW PROMPT CHUNK EMITS AT EVERY WIDTH, its probabilities in slot parts. At 1024 query rows per
/// kv head a 64-page chunk's `esub` is time-tiled whole, and a time-tiled op with a broadcast operand (the
/// row max) is the builder's own refusal. Cut into equal parts that each run untiled, every width emits,
/// and the parts cover each kv head's chunk exactly once, one `exp` per `esub`.
#[test]
fn a_wide_prompt_chunk_cuts_its_probabilities_into_untiled_parts() {
    const MQ: u32 = 256;
    for &j in &ChunkScratch::LADDER {
        let ops = named(emit_chunk_ops_at(j, MQ));
        let esub: Vec<&(String, serde_json::Value)> = ops
            .iter()
            .filter(|(n, _)| n.starts_with("attn_cesub_g"))
            .collect();
        let covered: u64 = esub
            .iter()
            .map(|(name, v)| v["dscs_"][0][name]["N_"]["out_"].as_u64().unwrap_or(0))
            .sum();
        assert_eq!(
            covered,
            u64::from(NKVH * j * 256),
            "{j} page(s): the esub parts cover {covered} slots across the kv heads"
        );
        assert_eq!(
            ops.iter()
                .filter(|(n, _)| n.starts_with("attn_cexp_g"))
                .count(),
            esub.len(),
            "{j} page(s): one exp per esub part"
        );
    }
}

/// The chunk at `pages` × `mq`, emitted against a REAL layout — the residency the body declares
/// (`BundleLayout::synth_lx_core_local`) only reaches the descriptors through one. Every caller-bound
/// operand is placed (an undeclared name is a build error); `assemble_attn` declares its own scratches.
fn emit_chunk_in_layout(pages: u32, mq: u32) -> Vec<(String, serde_json::Value)> {
    emit_chunk_in_layout_at::<HD>(pages, mq)
}

/// [`emit_chunk_in_layout`] at head dim `D` — 128 is granite-3.1-8b's geometry (same heads, two slabs).
fn emit_chunk_in_layout_at<const D: u32>(pages: u32, mq: u32) -> Vec<(String, serde_json::Value)> {
    use ktir_superdsc::place::PlaceId;
    use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
    let mut layout = BundleLayout::default();
    let mut names = Vec::new();
    for (i, tid) in (1..=8u32).chain([900]).enumerate() {
        let name = PlaceId::Act(tid).to_string();
        layout
            .ids
            .borrow_mut()
            .insert(name.clone(), PlaceId::Act(tid));
        names.push(name);
        layout.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: 3,
                bank: 0,
                offset: (i as u64) << 30,
                size: 1 << 30,
            },
        );
    }
    let geom = AttnGeometry::<NQH, NKVH, D>::minted();
    let rows = if mq == 1 {
        attn_bundle_rows(geom, 1, true).expect("the one-request decode rung")
    } else {
        attn_bundle_rows(geom, mq, false).expect("a prompt chunk's rows")
    };
    let chunk = ChunkScratch::of_chunk(PagedKvPool::new(NKVH as usize, D as usize), pages)
        .expect("a chunk granite admits");
    let mut sym = 0i64;
    let ops = assemble_attn(
        0,
        geom,
        rows,
        CAP,
        CAP,
        &names[0],
        &names[1],
        &names[2],
        &names[3],
        &names[4],
        &names[5],
        &names[6],
        Some(&names[7]),
        Some(chunk),
        PlaceId::Act(900),
        false,
        false,
        &mut sym,
        Some(&layout),
    )
    .unwrap_or_else(|e| panic!("pages={pages} mq={mq}: assemble_attn refused: {}", e.0));
    named(ops)
}

/// ⭐ THE CHUNK's `esub` OUTPUT NEVER LEAVES LX, AT DECODE AND AT THE SHIPPED PROMPT WIDTH, ON EVERY RUNG.
/// Each `esub` writes the `cesub` scratch LX-only and each `exp` reads it LX-only, at ONE start address on
/// every core (core-local: each core holds its own slice, the reference's form for its softmax
/// intermediates). That the whole ladder emits at all is the build-time proof that every `exp` divides
/// the work exactly as its `esub` did — the emitter refuses any op that does not.
#[test]
fn the_chunk_probabilities_stay_in_lx_between_esub_and_exp() {
    for mq in [1, PagedKvPool::PREFILL_CHUNK_SLOTS as u32] {
        for &j in &ChunkScratch::LADDER {
            let ops = emit_chunk_in_layout(j, mq);
            // A prompt chunk keeps its probabilities in LX all the way: exp writes them there and the
            // row sum and the PV read them there. Decode writes exp's output to HBM as before.
            let mut checks = vec![("attn_cesub_g", "out"), ("attn_cexp_g", "in")];
            if mq > 1 {
                checks.extend([
                    ("attn_cexp_g", "out"),
                    ("attn_cbsum_g", "in"),
                    ("attn_cov_g", "in"),
                ]);
            }
            for (stem, at) in checks {
                assert_core_local_lx(&ops, stem, at, &format!("{j} page(s), mq {mq}"));
            }
        }
    }
}

/// Every op named `stem*` reads (`at == "in"`, its first operand) or writes (`"out"`, its last) an
/// LX-only operand at ONE start address on every core — a core-local pin.
fn assert_core_local_lx(ops: &[(String, serde_json::Value)], stem: &str, at: &str, ctx: &str) {
    let mine: Vec<&(String, serde_json::Value)> =
        ops.iter().filter(|(n, _)| n.starts_with(stem)).collect();
    assert!(!mine.is_empty(), "{stem}* at {ctx}");
    for (name, v) in mine {
        let body = &v["dscs_"][0][name];
        let lds = body["labeledDs_"].as_array().expect("labeled operands");
        let lds = if at == "out" { lds.last() } else { lds.first() }.expect("an operand");
        assert!(
            lds["memOrg_"]
                .as_object()
                .is_some_and(|m| m.len() == 1 && m.contains_key("lx")),
            "{name} at {ctx}: its {at} operand is {}",
            lds["memOrg_"]
        );
        let idx = lds["ldsIdx_"].as_u64();
        let alloc = body["scheduleTree_"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|n| n["nodeType_"] == "allocate" && n["ldsIdx_"].as_u64() == idx)
            .expect("the operand's allocate node");
        assert_eq!(alloc["component_"], "lx", "{name}");
        let starts: std::collections::BTreeSet<String> = alloc["startAddressCoreCorelet_"]["data_"]
            .as_object()
            .into_iter()
            .flatten()
            .map(|(_, a)| a.to_string())
            .collect();
        assert_eq!(
            starts.len(),
            1,
            "{name} at {ctx}: starts at {starts:?} — one pin on every core"
        );
    }
}

/// ⭐⭐⭐ AT HEAD DIM 128 (granite-3.1-8b) A WHOLE-STICK PROMPT CHUNK IS CUT PER QUERY HEAD — the causal
/// block and the resident chunk alike — and keeps its probabilities in LX. Above one slab only one head's
/// `[mq, hd]` rows are a contiguous operand (`ChunkRowBlocks`), so every block is a query head: one score /
/// max / esub / exp / sum per head over its whole `[mq, mq_pad]` causal block, and one PV per (head, slab).
/// That every rung of the ladder emits at both prompt widths is the build-time proof that each PV divides
/// its work as the `exp` it reads from LX did.
#[test]
fn an_hd128_prompt_chunk_is_cut_per_query_head_and_keeps_its_probabilities_in_lx() {
    const HD128: u32 = 128;
    let slabs = (HD128 / 64) as usize;
    for mq in [128, PagedKvPool::PREFILL_CHUNK_SLOTS as u32] {
        for &j in &ChunkScratch::LADDER {
            let ops = emit_chunk_in_layout_at::<HD128>(j, mq);
            let ctx = format!("hd 128, {j} page(s), mq {mq}");
            let count = |stem: &str| ops.iter().filter(|(n, _)| n.starts_with(stem)).count();
            for stem in [
                "attn_ncsc_h",
                "attn_ncbmax_h",
                "attn_ncesub_h",
                "attn_ncexp_h",
                "attn_ncbsum_h",
            ] {
                assert_eq!(
                    count(stem),
                    NQH as usize,
                    "{stem}* at {ctx}: one per query head"
                );
            }
            assert_eq!(
                count("attn_ncov_h"),
                NQH as usize * slabs,
                "attn_ncov_h* at {ctx}: one per (query head, slab)"
            );
            assert_eq!(count("attn_ncsc_g"), 0, "{ctx}: no per-kv-head block");
            assert_eq!(
                count("attn_nsc_h"),
                0,
                "{ctx}: no sub-blocked new-block score"
            );
            for (stem, at) in [
                ("attn_ncesub_h", "out"),
                ("attn_cesub_h", "out"),
                ("attn_cexp_h", "in"),
                ("attn_cexp_h", "out"),
                ("attn_cbsum_h", "in"),
                ("attn_cov_h", "in"),
            ] {
                assert_core_local_lx(&ops, stem, at, &ctx);
            }
        }
    }
}

/// ⭐⭐⭐ A WHOLE-STICK PROMPT CHUNK'S CAUSAL BLOCK IS ONE FIXED SET PER KV HEAD, and every other width keeps
/// the sub-blocks. At 128 rows each kv head runs one score / max / esub / exp / sum / PV over its whole
/// `[gqa·mq, mq_pad]` block — no per-query-head op — with `esub`'s output LX-only; at 96 rows (not a whole
/// number of sticks, so its new-V block is shorter than `mq_pad`) none of that form appears.
#[test]
fn a_whole_stick_prompt_chunk_forms_its_causal_block_per_kv_head() {
    let stems = [
        "attn_ncsc_g",
        "attn_ncbmax_g",
        "attn_ncesub_g",
        "attn_ncexp_g",
        "attn_ncbsum_g",
        "attn_ncov_g",
    ];
    let ops = emit_chunk_in_layout(2, 128);
    for stem in stems {
        assert_eq!(
            ops.iter().filter(|(n, _)| n.starts_with(stem)).count(),
            NKVH as usize,
            "{stem}* at 128 rows: one per kv head"
        );
    }
    assert!(
        !ops.iter().any(|(n, _)| n.starts_with("attn_nsc_h")),
        "128 rows: the per-query-head new-block scores are gone"
    );
    for (name, v) in ops.iter().filter(|(n, _)| n.starts_with("attn_ncsc_g")) {
        let n = &v["dscs_"][0][name]["N_"];
        assert_eq!(
            (n["mb_"].as_u64(), n["out_"].as_u64()),
            (Some(u64::from(NQH / NKVH * 128)), Some(128)),
            "{name}: one kv head's rows over the whole block, N_ = {n}"
        );
    }
    for (name, v) in ops.iter().filter(|(n, _)| n.starts_with("attn_ncesub_g")) {
        let out = v["dscs_"][0][name]["labeledDs_"]
            .as_array()
            .and_then(|a| a.last())
            .expect("an output");
        assert_eq!(
            out["memOrg_"]
                .as_object()
                .map(|m| m.keys().cloned().collect::<Vec<_>>()),
            Some(vec!["lx".to_string()]),
            "{name}: the causal block's esub output stays in LX"
        );
    }
    let narrow = emit_chunk_in_layout(2, 96);
    for stem in stems {
        assert!(
            !narrow.iter().any(|(n, _)| n.starts_with(stem)),
            "{stem}* at 96 rows: a width that is not whole sticks keeps the sub-blocks"
        );
    }
}

/// ⛔ A CORE-LOCAL TENSOR TAKES ONE WORK DIVISION: the first op's is recorded, the same one again stands,
/// and any other is refused — the build-time half of "each core holds only what its own producer wrote".
#[test]
fn a_core_local_tensor_refuses_a_second_division() {
    use ktir_superdsc::place::PlaceId;
    use ktir_superdsc::placement::BundleLayout;
    let l = BundleLayout::default();
    l.synth_lx_core_local(PlaceId::Act(7), 4096);
    let name = PlaceId::Act(7).to_string();
    assert!(l.lx_core_local(&name));
    assert_eq!(l.lx_pin(&name).map(|(_, fp)| fp), Some(4096));
    assert_eq!(
        l.record_lx_division(&name, "esub", "split {out: 32}".into()),
        Ok(())
    );
    assert_eq!(
        l.record_lx_division(&name, "exp", "split {out: 32}".into()),
        Ok(())
    );
    assert!(
        l.record_lx_division(&name, "sum", "split {mb: 32}".into())
            .is_err()
    );
}

/// ⛔ EXACTLY THE GATHER COPIES ARE DECLARED GATHERED, AND NOTHING IS A FOLD OP. A gathered op groups on
/// its own (`GroupKind::Gathered`): its launch takes no KV page shift — the entries are absolute — and its
/// body is the one the host stages the index for. An unmarked copy lands in an ordinary group, whose
/// write-page base offsets every absolute entry; on card that was a refusal to run at all ("KvBlockIndex
/// step on a forward with no block table"). And the chunk body launches ONCE, so no op may carry the
/// per-page relaunch.
#[test]
fn only_the_gather_copies_are_declared_gathered_and_nothing_folds() {
    for &j in &ChunkScratch::LADDER {
        for e in emit_chunk_ops(j) {
            let copy = e.op_name.starts_with("attn_cgkt_") || e.op_name.starts_with("attn_cgv_");
            assert_eq!(e.kv_gathered, copy, "{} at {j} page(s)", e.op_name);
            assert!(
                !e.kv_page_fold,
                "{} at {j} page(s) is a fold op — the chunk body launches once",
                e.op_name
            );
        }
    }
}

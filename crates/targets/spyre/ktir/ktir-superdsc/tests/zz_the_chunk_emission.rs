// SPDX-License-Identifier: Apache-2.0
//! The split-KV chunk body's emission, from the shipped `assemble_attn` at granite-3.1-2b's geometry.

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{AttnFacts, assemble_attn};
use ktir_superdsc::sdsc_abstract::{
    AttnGeometry, ChunkRowKind, ChunkScratch, KvExtent, PagedKvPool, SlotCount, attn_bundle_rows,
};

const NQH: u32 = 32;
const NKVH: u32 = 8;
const HD: u32 = 64;
const CAP: u32 = 256;
/// The storage/sweep pair on the boundary's one carrier — the sweep is the full cap (no rung).
const EXTENT: KvExtent = KvExtent::new(SlotCount::new(CAP), SlotCount::new(CAP));

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
    let chunk = ChunkScratch::of_chunk(PagedKvPool::new(NKVH, HD), pages)
        .expect("a chunk granite-2b admits");
    let mut sym = 0i64;
    assemble_attn(
        AttnFacts {
            t: 0,
            geom,
            bundle_rows: rows,
            extent: EXTENT,
            qs: "t_qs",
            new_k_scaled: "t_new_k",
            new_v: "t_new_v",
            kct: "t_kct",
            vc: "t_vc",
            pmask: "t_pmask",
            cmask: "t_cmask",
            kv_block_index: Some("t_kvbi"),
            chunk: Some(chunk),
            out_id: ktir_superdsc::place::PlaceId::Act(900),
            rows_are_requests: false,
            headmajor_handoff: false,
        },
        &mut sym,
        None,
    )
    .unwrap_or_else(|e| panic!("pages={pages}: assemble_attn refused: {}", e.0))
}

/// ⭐ THE WHOLE POINT: one launch attends every page, with the SAME compute ops at every chunk width.
/// Only three things grow with it:
///   * the gather's copies, one per index stick;
///   * the decode score, one op per (kv head, part);
///   * the decode value leg, one op per (kv head, part). An even part count adds one copy home
///     (`attn_covsum`).
///
/// Past 8 pages both legs' parts are the LX gather's (`ChunkLx`): 8 pages each plus a 4-page tail.
#[test]
fn the_chunk_op_count_is_independent_of_its_width() {
    let pool = PagedKvPool::new(NKVH, HD);
    let value_leg = |n: &str| n.starts_with("attn_cov_") || n.starts_with("attn_covsum_");
    let score = |n: &str| n.starts_with("attn_csc_");
    let counts: Vec<(u32, usize)> = ChunkScratch::LADDER
        .iter()
        .map(|&j| {
            let ops = emit_chunk(j);
            let lx = ChunkScratch::of_chunk(pool, j).unwrap().lx_form(
                ChunkRowKind::OneRequest,
                /* no op-local pins: geometry only */ 0,
                /* no probability-part pins: geometry only */ 0,
            );
            let parts = lx.map_or(1, |v| v.lx_parts() + u32::from(v.tail_pages() > 0));
            let values = ops.iter().filter(|(n, _)| value_leg(n)).count();
            assert_eq!(
                values as u32,
                NKVH * parts + u32::from(parts.is_multiple_of(2)),
                "{j} page(s): {parts} value part(s)"
            );
            // The score is cut into parts only where Kᵗ stays in LX (a chunk with no tail).
            let score_parts = if lx.is_some_and(|v| v.keys().is_some()) {
                parts
            } else {
                1
            };
            let scores = ops.iter().filter(|(n, _)| score(n)).count();
            assert_eq!(
                scores as u32,
                NKVH * score_parts,
                "{j} page(s): {score_parts} score part(s)"
            );
            let compute = ops
                .iter()
                .filter(|(n, _)| {
                    !n.starts_with("attn_cgkt_")
                        && !n.starts_with("attn_cgv_")
                        && !value_leg(n)
                        && !score(n)
                })
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
/// Past 8 pages the value leg is cut into the LX value leg's parts. Each 8-page part is 32 sticks on
/// 32 cores, one per copy entry. A 4-page tail is 16 sticks on 16 cores.
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
    let most = |sticks: u64| {
        (1..=32u64.min(sticks))
            .rev()
            .find(|s| sticks.is_multiple_of(*s))
            .unwrap_or(1)
    };
    for &j in &ChunkScratch::LADDER {
        // The slots of value part `p` (parts named `_k{p}_`, one part named without a suffix).
        let part_sticks = |name: &str| -> u64 {
            let p = name
                .split("_k")
                .nth(1)
                .and_then(|s| s.split('_').next())
                .map_or(0, |s| s.parse::<u64>().unwrap());
            let per = if j >= 8 { 8 } else { u64::from(j) };
            let pages = per.min(u64::from(j) - p * per);
            pages * 256 / 64
        };
        for (name, v) in emit_chunk(j)
            .iter()
            .filter(|(n, _)| n.starts_with("attn_cov_g"))
        {
            let m = most(part_sticks(name));
            assert_eq!(
                splits(v),
                (m, 1, Some(m)),
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

/// ⛔ A DECODE SCORE PART DIVIDES ITS COLUMNS ALONE. Where Kᵗ stays in LX (a whole number of 8-page
/// parts, `ChunkLx::keys`) the one-request score is one op per (kv head, part), each writing its
/// columns of the wider score block. Every part keeps its rows whole and hands each core one whole
/// stick of its columns, 32 cores per 8-page part: one per Kᵗ copy entry.
#[test]
fn the_decode_score_parts_divide_their_columns_alone() {
    let pool = PagedKvPool::new(NKVH, HD);
    let keys_in_lx = |j: u32| {
        ChunkScratch::of_chunk(pool, j)
            .unwrap()
            .lx_form(ChunkRowKind::OneRequest, 0, 0)
            .is_some_and(|v| v.keys().is_some())
    };
    for &j in ChunkScratch::LADDER.iter().filter(|&&j| keys_in_lx(j)) {
        for (name, v) in emit_chunk(j)
            .iter()
            .filter(|(n, _)| n.starts_with("attn_csc_g"))
        {
            let p: u32 = name
                .split("_k")
                .nth(1)
                .and_then(|s| s.split('_').next())
                .unwrap()
                .parse()
                .unwrap();
            let sticks = 8u32.min(j - 8 * p) * 256 / 64;
            let s = &v["numWkSlicesPerDim_"];
            assert_eq!(
                (
                    s["out"].as_u64().unwrap_or(1),
                    s["mb"].as_u64().unwrap_or(1),
                    s["in"].as_u64().unwrap_or(1),
                    v["numCoresUsed_"].as_u64(),
                ),
                (u64::from(sticks), 1, 1, Some(u64::from(sticks))),
                "{name} at {j} page(s): split {s}"
            );
        }
    }
}

/// ⭐⭐⭐ A PROMPT CHUNK'S PREFIX IS THE SAME CHUNK OVER EVERY ROW: at the shipped prompt-chunk width
/// (`PREFILL_CHUNK_SLOTS` rows per head, 512 → `gqa·512 = 2048` query rows per kv head) each kv head's
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
                cover: ktir_superdsc::placement::BindCover::Full,
            },
        );
    }
    let geom = AttnGeometry::<NQH, NKVH, D>::minted();
    let rows = if mq == 1 {
        attn_bundle_rows(geom, 1, true).expect("the one-request decode rung")
    } else {
        attn_bundle_rows(geom, mq, false).expect("a prompt chunk's rows")
    };
    let chunk =
        ChunkScratch::of_chunk(PagedKvPool::new(NKVH, D), pages).expect("a chunk granite admits");
    let mut sym = 0i64;
    let ops = assemble_attn(
        AttnFacts {
            t: 0,
            geom,
            bundle_rows: rows,
            extent: EXTENT,
            qs: &names[0],
            new_k_scaled: &names[1],
            new_v: &names[2],
            kct: &names[3],
            vc: &names[4],
            pmask: &names[5],
            cmask: &names[6],
            kv_block_index: Some(&names[7]),
            chunk: Some(chunk),
            out_id: PlaceId::Act(900),
            rows_are_requests: false,
            headmajor_handoff: false,
        },
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
                ("attn_ncsc_h", "out"),
                ("attn_ncbmax_h", "in"),
                ("attn_ncesub_h", "in"),
                ("attn_ncesub_h", "out"),
                ("attn_ncexp_h", "in"),
                ("attn_ncexp_h", "out"),
                ("attn_ncbsum_h", "in"),
                ("attn_ncov_h", "in"),
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
    // ⭐ THE WHOLE BLOCK STAYS IN LX: the score, `esub` and `exp` are each written to and read from a
    // core-local pin by every op on them — none touches HBM.
    for (stem, at) in [
        ("attn_ncsc_g", "out"),
        ("attn_ncbmax_g", "in"),
        ("attn_ncesub_g", "in"),
        ("attn_ncesub_g", "out"),
        ("attn_ncexp_g", "in"),
        ("attn_ncexp_g", "out"),
        ("attn_ncbsum_g", "in"),
        ("attn_ncov_g", "in"),
    ] {
        assert_core_local_lx(&ops, stem, at, "the 128-row causal block");
    }
    // ⭐ AND THE SCORE DIVIDES ITS ROWS ALONE, as every consumer of its core-local slice does: the planner's
    // `{mb, out}` split would hand core `c` columns no consumer on core `c` reads.
    for (name, v) in ops.iter().filter(|(n, _)| n.starts_with("attn_ncsc_g")) {
        let split = &v["numWkSlicesPerDim_"];
        assert_eq!(
            (split["out"].as_u64(), split["in"].as_u64()),
            (Some(1), Some(1)),
            "{name}: the causal score divides rows only, split = {split}"
        );
        assert!(
            split["mb"].as_u64().is_some_and(|m| m > 1),
            "{name}: the causal score's rows are divided, split = {split}"
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

/// ⭐⭐⭐ THE ONE-REQUEST DECODE's GATHERED Kᵗ AND V NEVER TOUCH HBM, PART BY PART, AT EVERY RUNG FROM 8
/// PAGES.
///
/// Both legs are cut into 8-page parts, one copy per (part, kv head) per leg: one entry per core. The
/// part's score divides its slots, and its PV its contraction, so that core `c` gets exactly entry
/// `c`'s 64 slots. So each such copy writes its own core-local LX tensor, and that part's score or PV
/// reads its KERNEL from the same pin. Both are LX-only, with one start address on every core.
///
/// A chunk with a 4-page tail keeps V's tail in HBM: one kv head's tail is 16 entries, and dxp refuses
/// an index shorter than one stick, so its PV is an HBM part. Its Kᵗ stays whole in HBM, one score over
/// the whole chunk (`ChunkLx::keys`).
///
/// That the ladder emits at all is the build-time proof the divisions agree. The read law compares
/// each core's slice in the TENSOR's device layout, across the copy's `[mb, out]`, PV's `[in, out]`,
/// and the score's stick-major Kᵗ.
///
/// Below 8 pages, and every prompt chunk, keeps the HBM scratches.
#[test]
fn the_decode_gather_legs_read_each_part_kernel_from_lx() {
    let lx_only = |v: &serde_json::Value| {
        v["memOrg_"]
            .as_object()
            .is_some_and(|m| m.len() == 1 && m.contains_key("lx"))
    };
    // The one start address an operand's allocate node gives every core, or `None` if they differ.
    let pin = |body: &serde_json::Value, lds: &serde_json::Value| -> Option<String> {
        let idx = lds["ldsIdx_"].as_u64();
        let alloc = body["scheduleTree_"]
            .as_array()?
            .iter()
            .find(|n| n["nodeType_"] == "allocate" && n["ldsIdx_"].as_u64() == idx)?;
        assert_eq!(alloc["component_"], "lx");
        let starts: std::collections::BTreeSet<String> = alloc["startAddressCoreCorelet_"]["data_"]
            .as_object()?
            .values()
            .map(|a| a.to_string())
            .collect();
        (starts.len() == 1).then(|| starts.into_iter().next().unwrap())
    };
    let pool = PagedKvPool::new(NKVH, HD);
    for mq in [1, PagedKvPool::PREFILL_CHUNK_SLOTS as u32] {
        let rows = if mq == 1 {
            ChunkRowKind::OneRequest
        } else {
            ChunkRowKind::Prompt
        };
        for &j in &ChunkScratch::LADDER {
            let vlx = ChunkScratch::of_chunk(pool, j).unwrap().lx_form(rows, 0, 0);
            assert_eq!(vlx.is_some(), mq == 1 && j >= 8, "{j} page(s), mq {mq}");
            let (parts, tail) = vlx.map_or((0, 0), |v| (v.lx_parts(), v.tail_pages()));
            if let Some(v) = vlx {
                assert_eq!((v.part_pages(), parts * 8 + tail), (8, j));
                assert!(tail == 0 || tail == 4, "{j} page(s): tail {tail}");
                assert!(v.lx_bytes_per_core() <= 512 * 1024);
            }
            let ops = emit_chunk_in_layout(j, mq);
            let mut all_pins = std::collections::BTreeSet::new();
            for (copy, consumer) in [("attn_cgkt_", "attn_csc_"), ("attn_cgv_", "attn_cov_")] {
                // Every LX copy and every LX part's consumer, keyed by the pin they meet at.
                let mut copy_pins = Vec::new();
                let mut hbm_copies = 0;
                for (name, v) in ops.iter().filter(|(n, _)| n.starts_with(copy)) {
                    let body = &v["dscs_"][0][name];
                    let out = body["labeledDs_"].as_array().unwrap().last().unwrap();
                    if lx_only(out) {
                        copy_pins.push(pin(body, out).expect("one pin on every core"));
                    } else {
                        hbm_copies += 1;
                    }
                }
                let mut read_pins = Vec::new();
                let mut hbm_parts = 0;
                for (name, v) in ops.iter().filter(|(n, _)| n.starts_with(consumer)) {
                    let body = &v["dscs_"][0][name];
                    let kernel = body["labeledDs_"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|l| l["dsType_"] == "KERNEL")
                        .expect("the consumer's kernel");
                    if lx_only(kernel) {
                        read_pins.push(pin(body, kernel).expect("one pin on every core"));
                    } else {
                        hbm_parts += 1;
                    }
                }
                // Kᵗ stays in LX only on a chunk with no tail (`ChunkLx::keys`); V wherever there is
                // an LX form at all.
                let keys = copy == "attn_cgkt_";
                let leg_in_lx = vlx.is_some_and(|v| !keys || v.keys().is_some());
                let lx_tensors = if leg_in_lx {
                    (parts * NKVH) as usize
                } else {
                    0
                };
                assert_eq!(
                    copy_pins.len(),
                    lx_tensors,
                    "{copy} at {j} page(s), mq {mq}"
                );
                assert_eq!(
                    read_pins.len(),
                    lx_tensors,
                    "{consumer} at {j} page(s), mq {mq}"
                );
                // Each pin is written by exactly one copy and read by exactly one consumer.
                let (mut c, mut r) = (copy_pins.clone(), read_pins.clone());
                c.sort();
                r.sort();
                assert_eq!(
                    c, r,
                    "{copy} at {j} page(s): copy and consumer meet at one pin"
                );
                for p in &copy_pins {
                    assert!(all_pins.insert(p.clone()), "{j} page(s): pin {p} shared");
                }
                if leg_in_lx {
                    // The tail: the remaining copies are HBM, flat across kv heads (32 entries each),
                    // and each kv head's tail consumer is one HBM part.
                    assert_eq!(hbm_copies, (tail * 4 * NKVH / 32) as usize, "{copy} at {j}");
                    let tails = if tail > 0 { NKVH as usize } else { 0 };
                    assert_eq!(hbm_parts, tails, "{consumer} at {j} page(s)");
                } else if vlx.is_some() {
                    // Kᵗ whole in HBM beside an LX V: every copy HBM, one score per kv head.
                    assert_eq!(hbm_copies, j as usize, "{copy} at {j}");
                    assert_eq!(hbm_parts, NKVH as usize, "{consumer} at {j} page(s)");
                }
            }
            let legs = vlx.map_or(0, |v| 1 + u32::from(v.keys().is_some()));
            assert_eq!(
                all_pins.len(),
                (legs * parts * NKVH) as usize,
                "{j} page(s)"
            );
        }
    }
    // granite-8b's LX form is pinned rung by rung in `zz_the_hd128_chunk_in_lx`.
}

/// ⭐ THE RUN's ORDER IS ONE LAW FOR BOTH LEGS' COPIES AND THE HOST. Word `c` of a whole part's stick
/// names slot run `c` of that part, the run its score and PV both hand core `c`. The tail is the flat
/// chunk order over the tail's pages, and the run covers every (kv head, page, block) of the chunk
/// exactly once. The host's table is this one run, and each copy reads exactly its own stick of it.
#[test]
fn the_lx_form_run_covers_the_chunk_once_in_the_order_each_part_pv_reads() {
    let pool = PagedKvPool::new(NKVH, HD);
    for &j in ChunkScratch::LADDER.iter().filter(|&&j| j >= 8) {
        let chunk = ChunkScratch::of_chunk(pool, j).unwrap();
        let vlx = chunk.lx_form(ChunkRowKind::OneRequest, 0, 0).unwrap();
        let eph = chunk.entries_per_head_page();
        let mut seen = std::collections::BTreeSet::new();
        for e in 0..chunk.entries() {
            let (kvh, page, block) = vlx.entry_coord(e);
            assert!(
                kvh < NKVH && page < j && block < eph,
                "{j} page(s), entry {e}"
            );
            assert!(
                seen.insert((kvh, page, block)),
                "{j} page(s): entry {e} repeats"
            );
            if e < vlx.lx_tensors() * 32 {
                let (op, c) = (e / 32, e % 32);
                let (part, h) = (op / NKVH, op % NKVH);
                assert_eq!(vlx.lx_tensor(part, h, 0), op);
                assert_eq!(
                    (kvh, page * eph + block),
                    (h, part * 8 * eph + c),
                    "{j} page(s)"
                );
            }
        }
        assert_eq!(seen.len(), chunk.entries() as usize);
        // Each LX copy reads its own stick of the LX run: the ONE run both legs share when Kᵗ stays in
        // LX, else V's run after Kᵗ's own head-major one.
        let base = if vlx.keys().is_some() {
            0
        } else {
            chunk.index_sticks()
        };
        let sticks: Vec<u32> = vlx.copies().map(|(_, cp)| cp.stick()).collect();
        let want: Vec<u32> = (base..base + chunk.index_sticks()).collect();
        assert_eq!(sticks, want, "{j} page(s)");
        // Kᵗ's own copies (with its witness) read the same sticks and write tensor `stick`.
        if let Some(keys) = vlx.keys() {
            let k: Vec<(u32, u32)> = keys.copies().map(|(i, cp)| (i, cp.stick())).collect();
            let want: Vec<(u32, u32)> = (0..chunk.index_sticks()).map(|j| (j, j)).collect();
            assert_eq!(k, want, "{j} page(s)");
        }
        // And the host's table is that run (after Kᵗ's own when Kᵗ is in HBM): the prompt chunk's
        // same entries, part-major.
        let pages: Vec<i64> = (0..j as i64).collect();
        // A one-layer pool's page is `nkvh * PAGE_SLOTS / POOL_STICK` stick blocks; the ceiling
        // carries the ladder's own top (PAGES_MAX = 64) as the page budget.
        let ceiling = ktir_superdsc::sdsc_abstract::gather_entry_ceiling(
            64,
            32 * pool.stick_block_bytes(),
            pool,
        )
        .unwrap();
        let one = ktir_superdsc::sdsc_abstract::chunk_gather_index_table(
            chunk,
            ChunkRowKind::OneRequest,
            &pages,
            ceiling,
            0,
            0,
        )
        .unwrap();
        let prompt = ktir_superdsc::sdsc_abstract::chunk_gather_index_table(
            chunk,
            ChunkRowKind::Prompt,
            &pages,
            ceiling,
            0,
            0,
        )
        .unwrap();
        assert_eq!(prompt.len() as u32, chunk.entries(), "{j} page(s)");
        let n = chunk.entries() as usize;
        let one = if vlx.keys().is_some() {
            assert_eq!(one.len(), n, "{j} page(s): one run");
            one
        } else {
            assert_eq!(one.len(), 2 * n, "{j} page(s): Kᵗ's run, then V's");
            assert_eq!(
                &one[..n],
                &prompt[..],
                "{j} page(s): Kᵗ's run is head-major"
            );
            one[n..].to_vec()
        };
        let (mut a, mut b) = (one.clone(), prompt.clone());
        a.sort();
        b.sort();
        assert_eq!(a, b, "{j} page(s): the same entries, reordered");
        // One whole part and no tail is the one width where part-major IS head-major.
        if vlx.lx_parts() > 1 || vlx.tail_pages() > 0 {
            assert_ne!(one, prompt, "{j} page(s): part-major, not head-major");
        } else {
            assert_eq!(one, prompt, "{j} page(s): one part is the head-major order");
        }
    }
}

/// ⛔ A Kᵗ LX FORM WITH A TAIL IS UNREPRESENTABLE — the bake-time lockout of the measured card fault.
/// granite-2b's 12- and 20-page rungs (a 4-page tail) faulted with Kᵗ in LX (`HMI=STORE`, see
/// `ChunkLx::keys`, which also says why that measurement is confounded by the session's allocation and
/// not yet re-run). The only door to the Kᵗ LX form (`ChunkKeysLx`) refuses
/// exactly the tail rungs, at every rung of both shipped granite pools. The bundles those rungs bake
/// gather and score Kᵗ in HBM: no Kᵗ copy writes LX, and the score is one op per kv head.
#[test]
fn a_tail_rung_cannot_mint_keys_in_lx() {
    let pool = PagedKvPool::new(NKVH, HD);
    for &j in &ChunkScratch::LADDER {
        let Some(lx) =
            ChunkScratch::of_chunk(pool, j)
                .unwrap()
                .lx_form(ChunkRowKind::OneRequest, 0, 0)
        else {
            continue;
        };
        assert_eq!(
            lx.keys().is_some(),
            lx.tail_pages() == 0,
            "{j} page(s): tail {}",
            lx.tail_pages()
        );
        if lx.tail_pages() == 0 {
            continue;
        }
        let ops = emit_chunk_in_layout(j, 1);
        for (name, v) in ops.iter().filter(|(n, _)| n.starts_with("attn_cgkt_")) {
            let out = v["dscs_"][0][name]["labeledDs_"]
                .as_array()
                .unwrap()
                .last()
                .unwrap()
                .clone();
            assert!(
                out["memOrg_"].as_object().unwrap().contains_key("hbm"),
                "{name} at {j} page(s): a tail rung's Kᵗ copy writes LX"
            );
        }
        assert_eq!(
            ops.iter()
                .filter(|(n, _)| n.starts_with("attn_csc_"))
                .count(),
            NKVH as usize,
            "{j} page(s): one whole-chunk score per kv head"
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

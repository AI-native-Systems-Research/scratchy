//! ⛔ THE BATCHED-DECODE RUNG WIDTHS AT THE DOOR — the width gate the golden's
//! own routing never drives.
//!
//! `spliced_attn_lowers_through_the_door` states `rows_are_requests` only at
//! `mq == 1` (both kinds) and `false` for every `mq > 1` shape — the batched
//! decode rungs production bakes (`rows_are_requests = !is_prefill &&
//! decode_rows > 1`, MQ = the width) have NEVER been driven through the door
//! by a test, which is exactly how the batch widths could refuse at every
//! bake (the rebase2 cold-bake stderr: widths 2–16 by the elementwise
//! operand-mode extent law, width 32 by the `TensorExpandShape` walk arm)
//! without a test going red.
//!
//! This is that test: every width 2/4/8/16/32, granite 2b geometry, the full
//! one-page rope-appended cache, `rows_are_requests = true` — the routing law
//! production states (`r_ar = true` at `MQ` = the width) — and the program
//! MUST lower green. Born red on the current refusals; the lowering fix turns
//! it green. Every width's survivor set (`mq·nkvh·gqa` = `32·width` lanes) is
//! at or above one whole fp16 stick, so NONE of these cells is the sub-stick
//! class the golden pins at `mq == 1` — a refusal here is a door defect, not
//! wall 13.
//!
//! STEP 1 (localization): run with `-- --nocapture` to read the per-width
//! dumps — every KTIR op with its operands and result type, every parameter
//! region, and the door's verdict verbatim.

use std::collections::HashSet;

use scratchy_target_spyre::ktir_superdsc_door::lower as door_lower;
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::compute_bundle_layout;

/// The batch widths the bake re-rolls decode rungs for — every width that
/// divides the 32-entry id stick (`ENTRIES = NUM_SEQS · BLOCKS_PER_CHUNK`,
/// `CopyDims::ENTRIES_PER_OP`).
const WIDTHS: &[u32] = &[2, 4, 8, 16, 32];

/// One attention shape (copied from the golden's private harness — the same
/// fields, the same laws per field).
#[derive(Clone, Copy)]
struct AttnShape {
    /// (nqh, nkvh, hd) — granite 2b is 32/8/64.
    geom: (u32, u32, u32),
    /// Query rows: at a batched-decode rung, the WIDTH (one row per request).
    mq: u32,
    /// The resident cache TENSOR's row extent.
    cap: u32,
    /// The swept rung (0 = FULL).
    rung: u32,
}

impl AttnShape {
    fn active_cap(self) -> ktir_superdsc::ktir_node::ActiveCap {
        ktir_superdsc::ktir_node::ActiveCap::new(self.rung)
    }
}

/// The attention fixture's decode-position model (the golden's own value).
const ATTN_VALID_LEN: u32 = 33;

/// The golden's attention fixture, copied verbatim in construction: rope
/// append then attention over the appended one-page cache. See
/// `triton_splice_golden.rs::attn_ir` — every law stated there holds here.
fn attn_ir(s: AttnShape, scale: f32) -> scratchy_subtile::subtile_ir::SubtileIR {
    let (nqh, nkvh, hd) = s.geom;
    let q_width = nqh * hd;
    let kv_width = nkvh * hd;
    let mq = s.mq;
    let valid_len = ATTN_VALID_LEN;
    // The new block's rows: one new row per request at a batched-decode rung.
    let new_len = mq;
    let sources = |cache_rows: u32| {
        vec![
            scratchy_subtile::subtile_ir::SourceShape {
                rows: mq,
                cols: q_width,
            }, // 0: q
            scratchy_subtile::subtile_ir::SourceShape {
                rows: cache_rows,
                cols: kv_width,
            }, // 1: prefix_k
            scratchy_subtile::subtile_ir::SourceShape {
                rows: cache_rows,
                cols: kv_width,
            }, // 2: prefix_v
            scratchy_subtile::subtile_ir::SourceShape {
                rows: new_len,
                cols: kv_width,
            }, // 3: new_k
            scratchy_subtile::subtile_ir::SourceShape {
                rows: new_len,
                cols: kv_width,
            }, // 4: new_v
            scratchy_subtile::subtile_ir::SourceShape {
                rows: new_len,
                cols: kv_width,
            }, // 5: cos
            scratchy_subtile::subtile_ir::SourceShape {
                rows: new_len,
                cols: kv_width,
            }, // 6: sin
        ]
    };
    let attn_op = scratchy_subtile::lower::OpDesc {
        op: scratchy_subtile::subtile_ir::SubOp::attn_decode(
            attn_geometry(s.geom).expect("granite's geometry mints"),
            scale,
            valid_len,
            scratchy_subtile::subtile_ir::AttnMask::Causal,
        ),
        m: mq,
        inputs: vec![
            scratchy_subtile::lower::InputRef::Ext(0),
            scratchy_subtile::lower::InputRef::Ext(1),
            scratchy_subtile::lower::InputRef::Ext(2),
            scratchy_subtile::lower::InputRef::Ext(3),
            scratchy_subtile::lower::InputRef::Ext(4),
        ],
    };
    let nb = std::num::NonZeroU32::new(8192).expect("8192 != 0");
    let input = scratchy_subtile::lower::LoweringInput {
        sources: sources(s.cap),
        ops: vec![
            // 0: k' = rope_append(k, cos, sin, v, prefix_k, prefix_v)
            scratchy_subtile::lower::OpDesc {
                op: scratchy_subtile::subtile_ir::SubOp::rope_append(
                    scratchy_subtile::model_geometry::HeadDim::new(hd),
                    0,
                    scratchy_subtile::subtile_ir::AttnMask::Causal,
                    scratchy_subtile::subtile_ir::RopeFormTag::NeoX,
                ),
                m: new_len,
                inputs: vec![
                    scratchy_subtile::lower::InputRef::Ext(3),
                    scratchy_subtile::lower::InputRef::Ext(5),
                    scratchy_subtile::lower::InputRef::Ext(6),
                    scratchy_subtile::lower::InputRef::Ext(4),
                    scratchy_subtile::lower::InputRef::Ext(1),
                    scratchy_subtile::lower::InputRef::Ext(2),
                ],
            },
            // 1: attn = AttnDecode(q, prefix_k, prefix_v, new_k, new_v)
            attn_op,
        ],
        result: 1,
        exported: None,
    };
    scratchy_subtile::subtile_ir::lower_region(&input, nb)
}

/// granite's geometry, through the same mint the config path uses.
fn attn_geometry(
    geom: (u32, u32, u32),
) -> Option<scratchy_subtile::model_geometry::ModelAttnGeometry> {
    scratchy_subtile::model_geometry::ModelAttnGeometry::mint(
        scratchy_subtile::model_geometry::QueryHeads::new(geom.0),
        scratchy_subtile::model_geometry::KvHeads::new(geom.1),
        scratchy_subtile::model_geometry::HeadDim::new(geom.2),
    )
}

#[test]
fn the_batched_decode_rung_widths_lower_through_the_door() {
    // granite 2b: 32 query heads, 8 KV heads, head dim 64 → GQA 4.
    let geom = (32, 8, 64);
    let (nqh, nkvh, _hd) = geom;
    let gqa = nqh / nkvh;
    let stick = ktir_superdsc::superdsc_opspec::Df::Fp16.elems_per_stick();
    // granite's attention_multiplier at hd 64 (the config value, f16-exact).
    let scale = 0.015625f32;
    let mut failures: Vec<(u32, String)> = Vec::new();

    for &width in WIDTHS {
        let s = AttnShape {
            geom,
            mq: width,
            cap: 256,
            rung: 0,
        };
        // THE TEST'S OWN PREMISE: every batch width's merge survivor set is at
        // or above one whole stick, so none of these cells may refuse under
        // wall 13's sub-stick law (the golden pins that law at `mq == 1`).
        let survivors = width * nkvh * gqa;
        assert!(
            survivors >= stick,
            "width {width}: {survivors} survivor lanes is sub-stick at {stick} — this shape \
             belongs to the golden's sub-stick pin, not this width gate"
        );

        let ir = attn_ir(s, scale);
        let weight_ids: HashSet<u32> = HashSet::new();
        let mut layout = compute_bundle_layout(&ir, &weight_ids, true, true, &[])
            .unwrap_or_else(|e| panic!("layout minted width={width}: {e}"));
        let node = ir.nodes.last().unwrap();
        let cap = ir.tensors[node.inputs[1].tensor.index()].rows;

        // The splice, at the bake's own routing fact: `rows_are_requests =
        // true` at `MQ` = the width.
        let spliced_ops = scratchy_triton_splice::lower_attn(
            node,
            &ir,
            cap,
            s.active_cap(),
            true,
            None,
            scratchy_triton_splice::AttnBundleFacts {
                scale_law: Some(ktir_superdsc::score_scale::ScoreScaleLaw::StatedByProgram),
                out_stays_head_rows: false,
            },
        )
        .unwrap_or_else(|e| panic!("splice compiled width={width}: {e}"));
        assert_eq!(
            spliced_ops.len(),
            1,
            "a foldless `lower_attn` returns exactly the one-shot program (width={width})"
        );
        let spliced = &spliced_ops[0];
        assert_eq!(
            spliced.op_name,
            format!("attn_s{}", node.id.index()),
            "op_name (width={width})"
        );

        println!(
            "\n=== width {width} (r_ar=true, rung FULL, {survivors} survivors, stick {stick}) \
             — {} ===",
            spliced.op_name
        );

        let ktir = spliced
            .ktir
            .as_ref()
            .expect("spliced op carries its program");

        // THE PARAMETER REGIONS — the buffers the program addresses, with the
        // windows it takes of them.
        match ktir_superdsc::emit::lower_ktir_to_superdsc::regions(ktir) {
            Ok(regs) => {
                for r in &regs {
                    println!(
                        "[region] tid={} view=[{}x{}] window=({},{})+{}x{} r_cover={:?} \
                         c_cover={:?} out={} fp8={} df={}",
                        r.tid,
                        r.v_rows,
                        r.v_cols,
                        r.r_start,
                        r.c_start,
                        r.r_len,
                        r.c_len,
                        r.r_cover,
                        r.c_cover,
                        r.is_out,
                        r.is_fp8,
                        r.df.name()
                    );
                }
            }
            Err(e) => println!("[region] REGIONS WALK REFUSED: {}", e.message),
        }

        // THE OP CENSUS — every op in program order, with its operands, its
        // result, and its result type (the shape a refusal names).
        for (i, op) in ktir.func.ops_deep().iter().enumerate() {
            let res = match op.result {
                Some(v) => format!("t{}", v.0),
                None => "-".to_string(),
            };
            let ty = match &op.result_type {
                Some(t) => format!(" : {t:?}"),
                None => String::new(),
            };
            let operands = op
                .operands
                .iter()
                .map(|v| format!("t{}", v.0))
                .collect::<Vec<_>>()
                .join(", ");
            println!("[op {i:3}] {:?} {res}{ty} <- [{operands}]", op.op_type);
        }
        println!("[bindings] {:?}", ktir.bindings);
        println!("[mask] {:?}", ktir.mask);

        // The placement and scale passes the walk runs before the door.
        scratchy_target_spyre::lower_subtile_tape_to_superdsc::place_step_inputs(
            &mut layout,
            &[std::slice::from_ref(spliced)],
            ir.tensors.len() as u32,
        )
        .unwrap_or_else(|e| panic!("step inputs placed width={width}: {e}"));
        scratchy_target_spyre::lower_subtile_tape_to_ktir::complete_program_scales(
            &mut layout,
            &[std::slice::from_ref(spliced)],
        );
        // The walk's own TAIL placements, in its own order: the KV block index, then the batched
        // arm's Q repetition table, which `place_step_inputs` defers because it is the one step
        // input whose presence differs between a session's bodies.
        layout.place_kv_block_index_last();
        layout.place_rep_table_last();

        let attn_params = scratchy_target_spyre::ktir_superdsc_door::BundleAttnParams {
            rows_are_requests: true,
            gathers_kv: true,
            chunk_pages: None,
            scale_law: ktir_superdsc::score_scale::ScoreScaleLaw::StatedByProgram,
            headmajor_handoff: false,
            oproj_wtid: None,
        };
        let mut sym = 0i64;
        let lowered = door_lower(ktir, &mut sym, Some(&layout), Some(attn_params));
        match &lowered {
            Ok(_) => println!("[door] width {width}: LOWERED (green)"),
            Err(e) => {
                println!("[door] width {width}: REFUSED: {}", e.message);
                failures.push((width, e.message.clone()));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "every batch width must lower green through the door (r_ar=true at MQ = the width); \
         refused: {failures:#?}"
    );
}

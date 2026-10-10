// SPDX-License-Identifier: Apache-2.0
//! THE HEAD-MAJOR o_proj HANDOFF, END TO END — both halves of the restructure minted by
//! ONE walk over a granite-3.1-8b-shaped graph, and the contract between them proven at
//! the descriptor level.
//!
//! The restructure has two emission halves — the attention finalize's slab-major form
//! (one op per feature slab instead of one per head×slab) and the o matmul's B-operand
//! swap onto a second, k-axis-block-shuffled o_proj copy — joined by ONE bundle fact
//! ([`BundleAttnParams::headmajor_handoff`] + `oproj_wtid`). A half-applied restructure
//! is not a slowdown, it is GARBAGE: slab-major activation bytes read against an
//! unpermuted weight (or the reverse) contract the wrong elements. So this file pins
//! the WHOLE chain on the 8b geometry (`hd = 128`, two feature slabs; the o matmul fp8
//! W8A8; `mq == 1` with rows-as-requests):
//!
//! 1. **the bundle fact** — the walk mints `headmajor_handoff` naming the o matmul's
//!    weight tid, found by DATAFLOW (its A operand is an attention output), never by a
//!    shape (`[4096, 4096]` is square; a shape key would re-point every square
//!    projection at the permuted copy);
//! 2. **the layout** — every layer's o_proj gains a second copy under
//!    [`oproj_headmajor_tid`], placed one 128-aligned block after its companion in the
//!    weight segment, so each layer's block grows by exactly one copy and the per-layer
//!    stride stays uniform; both copy families register in `per_layer`; the walk's
//!    retile manifest carries the six-axis permuted staging walk;
//! 3. **the mint is bundle-invariant** — the copies exist for the PREFILL row kind too,
//!    because every bundle of a model shares ONE staged weight segment (a rows-gated
//!    mint would hand the single-row body addresses no other bundle staged);
//! 4. **the door A/B** — with the handoff, the o matmul's `matmulfp8` reads its weight
//!    operand at the PERMUTED copy's placement and the finalize is 2 ops (one per slab);
//!    without it, the same program reads the companion's placement and the finalize is
//!    the shipped 64 ops (32 heads × 2 slabs). Same walk, same layout, one boolean;
//! 5. **the desyncs refuse** — a handoff naming no weight, and a handoff naming a DENSE
//!    projection's weight, are build errors naming the bundle's own statement, never a
//!    half-restructured emission.
//!
//! The fixture is the compiler's own `one_layer_input_shaped` at the 8b geometry with
//! the o projection converted to fp8 W8A8 (the ONLY fp8 matmul, so the program carrying
//! the `matmulfp8` identifies the o matmul by DATA), replicated per-layer with distinct
//! weights the `n_layer_input_distinct` way — a baked-layer-0 regression must diverge.

use std::collections::{BTreeMap, HashSet};

use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::emit::lower_ktir_to_superdsc::Error as DoorError;
use ktir_superdsc::placement::SegRole;
use ktir_superdsc::reserved_tids::oproj_headmajor_tid;
use ktir_superdsc::wire::SEGMENT_OFFSETS;
use scratchy_subtile::fixtures;
use scratchy_subtile::lower::{GemmWeight, InputRef, LoweringInput, OpDesc};
use scratchy_subtile::subtile_ir::{
    NeoX, SourceShape, SubOp, SubtileIR, ValidatedGraph, lower_region,
};
use scratchy_subtile::subtile_tape::{lower_dag_to_tape, reroll_subtile_tape};
use scratchy_subtile::superdsc_opspec::{DataFormat, Fp8};
use scratchy_target_spyre::ktir_superdsc_door::{BundleAttnParams, lower as door_lower};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::{
    ActiveCap, RolledSuperDsc, compute_bundle_layout, lower_subtile_tape_to_superdsc,
};

/// granite-3.1-8b's attention block: hidden 4096, kv_dim 1024 (8 kv-heads), head_dim 128
/// (32 q-heads → `nslab = hd / POOL_STICK = 2`, the multi-slab fact the restructure needs).
/// The intermediate is sized 4096 — any %64 width carries the layout law, and the o_proj
/// geometry under test (`[k = 4096, n = 4096]`, square) does not depend on it.
const H: u32 = 4096;
const KV: u32 = 1024;
const I: u32 = 4096;
const HD: u32 = 128;
/// Two layers: the reroll needs a repeating body, and `per_layer` needs a second entry to
/// prove the copy FAMILY (not just one placement) — the minimum that exercises both.
const LAYERS: u32 = 2;

/// The one-layer decode fixture at the 8b geometry, with the o projection (op 7, reading
/// Ext(9)) converted to fp8 W8A8: `GemmWeight::Fp8Dynamic` plus the arity-3 scale operand
/// (source 16, `[1, n]` — one fp16 scale per output channel). Every other projection stays
/// dense.
///
/// ⛔ THE ROTARY TABLE IS PRE-TILED PER ROPE SITE — the production law
/// (`to_wavefront::cos_sin`: one source pair per FULL column width, memoized; the worker
/// tiles the per-position row across heads and the door's rope reads the table WHOLE).
/// The shared fixture's `[1, hd]` row suits the subtile-level tests, but this walk runs
/// the door, whose `rope_xc` reads `[1, heads·hd]` — so the q site's pair (sources 5, 6)
/// is widened to `[1, H]`, and the K-side append — a DIFFERENT width, `kv_heads·hd` —
/// gets its own `[1, KV]` pair (sources 14, 15) the way production memoizes per width.
fn fp8_oproj_one_layer() -> LoweringInput {
    let mut input = fixtures::one_layer_input_shaped(H, KV, I, HD);
    input.sources[5] = SourceShape { rows: 1, cols: H }; // 5: cos, q-site tiled [1, 32·128]
    input.sources[6] = SourceShape { rows: 1, cols: H }; // 6: sin, q-site tiled
    input.sources.push(SourceShape { rows: 1, cols: KV }); // 14: cos, k-site tiled [1, 8·128]
    input.sources.push(SourceShape { rows: 1, cols: KV }); // 15: sin, k-site tiled
    // The K-side append (op 5) reads its own-width pair — inputs 1, 2 are its cos, sin.
    input.ops[5].inputs[1] = InputRef::Ext(14);
    input.ops[5].inputs[2] = InputRef::Ext(15);
    input.sources.push(SourceShape { rows: 1, cols: H }); // 16: o_scale [1, n]
    let o = &mut input.ops[7];
    // Op 7 is the o projection — its first input is op 6, the AttnDecode — and the scale
    // is the arity-3 matmul's THIRD input, after [activation, weight].
    let SubOp::MatmulTile { n, .. } = o.op else {
        panic!("fixture op 7 is the o projection");
    };
    o.op = SubOp::MatmulTile {
        n,
        weight: GemmWeight::Fp8Dynamic,
    };
    o.inputs.push(InputRef::Ext(16));
    input
}

/// `LAYERS` copies of the fp8 layer, each with its OWN weights/KV/scales (the
/// `n_layer_input_distinct` law), under a one-op PROLOGUE standing in for the
/// embedding: `x₀ = scalarmul(x, 12.0)` — granite-3.1's own `embedding_multiplier`,
/// the production fact this op stands in for. The prologue is LOAD-BEARING for the
/// reroll, not set dressing: with x a raw external, layer 0's rmsnorm reads an
/// `External` while every later layer reads the previous layer's `Computed` residual,
/// so layer 0 stands alone outside the repeating run and two layers leave that run one
/// copy long — no loop, and the walk refuses ("no repeating body"). Under the prologue
/// every layer's input is Computed (layer 0 reads the prologue's output the way layer
/// L reads layer L−1's), which is the production shape: the macro's own reroll runs on
/// graphs whose embedding op feeds layer 0 exactly so, and `per_layer` families are one
/// entry per LOOP ITERATION by contract — every layer must be in the run.
///
/// ⛔ The scale is 12.0, never 1.0: the door reads the multiplier off the program as
/// ONE splat feeding every `arith.mulf` (`scalarmul.py`'s own contract), and a multiply
/// by 1.0 is exactly what constant folding erases — the splat vanishes, and the walk
/// refuses ("the program states no single multiplier").
///
/// Shared across layers: x and the four rotary-table sources. Returns the input with
/// the staged-WEIGHT source id set — every per-layer source except the two KV caches
/// (seg2's own role): the gammas, the seven projections, and the fp8 scale, the
/// production classification of what the checkpoint ships as a layer weight.
///
/// Source ids: 0=x, 1=cos_q, 2=sin_q, 3=cos_k, 4=sin_k (all five SHARED — the rotary
/// tables are layer-invariant), then `n` blocks of twelve per-layer sources. Layer 0's
/// block is `5..=16` (o_w = 11, gate_w = 13, o_scale = 16); layer `L`'s starts at
/// `5 + L·12` (layer 1: o_w = 23, gate_w = 25, o_scale = 28).
fn n_layer_fp8_oproj(n: u32) -> (LoweringInput, HashSet<u32>) {
    let base = fp8_oproj_one_layer();
    let body = base.ops.clone();
    let ops_per_layer = body.len(); // 15
    // Base source ids: 0=x, 1=rms_w0, 2=q_w, 3=k_w, 4=v_w, 5=cos_q, 6=sin_q, 7=prefix_k,
    // 8=prefix_v, 9=o_w, 10=rms_w1, 11=gate_w, 12=up_w, 13=down_w, 14=cos_k, 15=sin_k,
    // 16=o_scale. SHARED across layers: 0 (x — global input) and the four rotary-table
    // ids (5, 6, 14, 15 — layer-invariant). PER-LAYER: the other twelve.
    let per_layer_src_ids: Vec<usize> = (1..base.sources.len())
        .filter(|e| !matches!(e, 5 | 6 | 14 | 15))
        .collect();
    let n_per = per_layer_src_ids.len(); // 12
    let mut sources = vec![
        base.sources[0],
        base.sources[5],
        base.sources[6],
        base.sources[14],
        base.sources[15],
    ];
    for _ in 0..n {
        for &e in &per_layer_src_ids {
            sources.push(base.sources[e]);
        }
    }
    let map_ext = |e: usize, layer: usize| -> usize {
        match e {
            5 => 1,  // cos_q (shared)
            6 => 2,  // sin_q (shared)
            14 => 3, // cos_k (shared)
            15 => 4, // sin_k (shared)
            _ => {
                5 + layer * n_per
                    + per_layer_src_ids
                        .iter()
                        .position(|&x| x == e)
                        .expect("per-layer src")
            }
        }
    };
    let mut ops: Vec<OpDesc> = Vec::with_capacity(1 + n as usize * ops_per_layer);
    // Op 0 — the embedding stand-in (see the doc above). It reads NO weight, so the
    // staged-weight set below is untouched by it.
    ops.push(OpDesc {
        op: SubOp::ScalarMul { scale: 12.0 },
        m: 1,
        inputs: vec![InputRef::Ext(0)],
    });
    let mut weight_ids = HashSet::new();
    for layer in 0..n as usize {
        // +1: every layer op sits one node after the prologue.
        let base_op = 1 + layer * ops_per_layer;
        for op in body.iter() {
            let inputs = op
                .inputs
                .iter()
                .map(|inp| match inp {
                    InputRef::Op(j) => InputRef::Op(base_op + *j),
                    // Ext(0) = the layer-input x: the prologue's output for layer 0,
                    // the previous layer's residual output for L>0 — ONE rule, which
                    // is what makes every layer a copy of one repeating body.
                    InputRef::Ext(0) => InputRef::Op(base_op - 1),
                    InputRef::Ext(e) => InputRef::Ext(map_ext(*e, layer)),
                })
                .collect();
            ops.push(OpDesc {
                op: op.op,
                m: op.m,
                inputs,
            });
        }
        // This layer's staged weights: the twelve per-layer sources minus the KV caches.
        for &e in &per_layer_src_ids {
            if e != 7 && e != 8 {
                weight_ids.insert(map_ext(e, layer) as u32);
            }
        }
    }
    (
        LoweringInput {
            sources,
            result: ops.len() - 1,
            ops,
        },
        weight_ids,
    )
}

/// The production chain over the fixture — the codegen's own four steps (region →
/// validate → tape → reroll) and the rolled walk, at the ONE-REQUEST rung's own argument
/// values: `rows_are_requests = false` (the single-row bake keeps the shipped emission —
/// that arg turns true only for batched rungs, two rows and wider) and
/// `one_request_decode = true` (the handoff's own fact). Returns the graph and weight set
/// beside the roll so the layout mint can be re-run for the prefill row kind.
fn walk_at_8b() -> (RolledSuperDsc, SubtileIR<NeoX>, HashSet<u32>) {
    let (input, weight_ids) = n_layer_fp8_oproj(LAYERS);
    let rg = lower_region(&input, std::num::NonZeroU32::new(8192).unwrap());
    let valid = ValidatedGraph::new(&rg).expect("the fp8 o_proj graph validates");
    let tape = lower_dag_to_tape(&valid);
    let rolled_tape = reroll_subtile_tape(&tape, &rg);
    let rolled = lower_subtile_tape_to_superdsc(
        &rolled_tape,
        &rg,
        &weight_ids,
        ActiveCap::FULL,
        false,
        true,
    )
    .expect("the 8b-shaped two-layer graph lowers with the head-major handoff");
    (rolled, rg, weight_ids)
}

/// Door-lower ONE body op's program — the consumer pass's per-op call (`sym` threads the
/// descriptor id counter, `quantized` the per-bundle activation-quantize dedup set; both
/// fresh per call, which cannot move an address — placements, not symbols, place tensors).
fn door_lower_op(
    rolled: &RolledSuperDsc,
    e: &EmittedOp,
    params: Option<BundleAttnParams>,
) -> Result<Vec<EmittedOp>, DoorError> {
    let k = e.ktir.as_ref().expect("a body op carries its program");
    let mut sym = 0i64;
    let mut quantized = HashSet::new();
    door_lower(k, &mut sym, Some(&rolled.layout), &mut quantized, params)
}

/// The body's matmul programs, in walk order (one canonical layer's worth: q, k, v, o,
/// gate, up, down — node ids 2, 3, 4, 8, 11, 12, 14 of the fixture, one past the
/// prologue).
fn matmul_body_ops(rolled: &RolledSuperDsc) -> Vec<&EmittedOp> {
    let ops: Vec<&EmittedOp> = rolled
        .body
        .iter()
        .filter(|e| e.op_name.starts_with("matmul_s"))
        .collect();
    assert_eq!(
        ops.len(),
        7,
        "one canonical layer's projections: q, k, v, o, gate, up, down"
    );
    ops
}

/// Every declared operand's device address, in `labeledDs_` order (the
/// `zz_the_bound_scale_is_a_launch_binding` idiom): the descriptor names operands only
/// positionally, so which address appears is which buffer the op reads.
fn operand_addresses(d: &ktir_superdsc::wire::Dsc) -> Vec<u64> {
    d.labeledDs_
        .iter()
        .filter_map(|l| {
            d.scheduleTree_
                .iter()
                .find(|n| n.nodeType_ == "allocate" && n.ldsIdx_ == l.ldsIdx_)
                .and_then(|n| {
                    n.startAddressCoreCorelet_
                        .data_
                        .get("[0, 0, 0]")
                        .and_then(|v| v.parse().ok())
                })
        })
        .collect()
}

/// THE WHOLE CHAIN: the bundle fact, the layout law, the manifest, the family, the
/// bundle-invariant mint, and the door A/B — one walk, every half present.
#[test]
fn the_headmajor_oproj_handoff_runs_the_whole_walk() {
    let (rolled, rg, weight_ids) = walk_at_8b();
    let params = rolled
        .attn_params
        .expect("an attention graph states its bundle facts");

    // ── 1. THE BUNDLE FACT ────────────────────────────────────────────────────────────────────
    // The handoff is minted (not defaulted) and names the o matmul's weight BY DATAFLOW:
    // layer 0's o_w is source 11. `k == n` here — 4096 × 4096 — so this tid is the only
    // thing standing between the o matmul and every other square projection.
    // The one-request rung's OWN value — false. The arg turns true only for batched rungs
    // (two rows and wider); the single-row bake keeps the shipped emission for everything
    // downstream of it, which is exactly why the handoff is stated by its own fact instead.
    assert!(!params.rows_are_requests);
    assert!(
        params.headmajor_handoff,
        "the one-request decode rung over a multi-slab fp8 o_proj mints the handoff"
    );
    assert_eq!(
        params.oproj_wtid,
        Some(11),
        "the o matmul's weight, by dataflow"
    );

    // ── 2. THE LAYOUT: both layers' copies, after their companions ────────────────────────────
    let perm0 = oproj_headmajor_tid(11);
    let perm1 = oproj_headmajor_tid(23);
    assert_eq!(
        rolled.layout.weight_copies,
        vec![(11, perm0), (23, perm1)],
        "one permuted copy per layer's o_proj, in source-tid order"
    );
    let comp = &rolled.layout.placements[&11];
    let perm = &rolled.layout.placements[&perm0];
    assert_eq!(
        perm.segment, comp.segment,
        "the copy lives in the weight segment"
    );
    assert!(matches!(perm.role, SegRole::Weight));
    assert_eq!(
        perm.bank, comp.bank,
        "the copy follows its companion's bank"
    );
    assert_eq!(
        perm.size, comp.size,
        "the permuted copy is the same byte count as the weight it permutes"
    );
    // The insert's own law: one 128-aligned block AFTER the companion (the align128 of
    // `offset + size`), so each layer's block grows by exactly one copy.
    assert_eq!(
        perm.offset,
        (comp.offset + comp.size + 127) & !127,
        "the copy is placed one aligned block after its companion"
    );

    // ── 3. THE MANIFEST: the six-axis permuted staging walk ───────────────────────────────────
    // `[n/64, nslab, nqh, POOL_STICK/2, 64, 2]` × `[64k, POOL_STICK, hd, 2, k, 1]` at
    // k = n = 4096, hd = 128, nqh = 32, nslab = 2 — the walk the worker re-tiles the copy
    // with, at fp8 stick granularity like any fp8 weight.
    for p in [perm0, perm1] {
        let d = rolled
            .layout
            .kernel_weights
            .get(&p)
            .unwrap_or_else(|| panic!("the permuted copy t{p} has a retile manifest entry"));
        assert_eq!(
            d.device_size,
            vec![64, 2, 32, 32, 64, 2],
            "t{p}: device_size"
        );
        assert_eq!(
            d.stride_map,
            vec![64 * 4096, 64, 128, 2, 4096, 1],
            "t{p}: stride_map"
        );
        assert_eq!(
            d.stick_size,
            Fp8::ELEMS_PER_STICK,
            "t{p}: fp8 stick granularity"
        );
        assert_eq!(d.word_length, Fp8::WORD_LENGTH, "t{p}: fp8 element width");
    }

    // ── 4. THE FAMILY AND THE STRIDE ──────────────────────────────────────────────────────────
    // The copy registers as a per-layer family keyed on the CANONICAL copy tid, and its
    // per-layer advance is the SAME weight_stride the companion family advances by — the
    // uniformity the executor's one-base-per-segment bind needs (and the stride guard
    // proves; this is the walk's own statement of it).
    assert_eq!(rolled.iters, LAYERS);
    assert_eq!(
        rolled.layers_per_bank, rolled.iters,
        "unbanked: every layer in bank 0"
    );
    assert_eq!(
        rolled.per_layer.get(&perm0).map(|v| v.as_slice()),
        Some(&[perm0, perm1][..]),
        "the permuted copies form their own per-layer family"
    );
    assert_eq!(
        rolled.per_layer.get(&11).map(|v| v.as_slice()),
        Some(&[11, 23][..]),
        "the companions form theirs"
    );
    assert!(rolled.weight_stride > 0);
    assert_eq!(
        rolled.layout.placements[&perm1].offset - rolled.layout.placements[&perm0].offset,
        rolled.weight_stride,
        "the copy family advances by the weight stride"
    );
    assert_eq!(
        rolled.layout.placements[&23].offset - rolled.layout.placements[&11].offset,
        rolled.weight_stride,
        "…exactly as the companion family does"
    );

    // ── 5. THE MINT IS BUNDLE-INVARIANT ───────────────────────────────────────────────────────
    // The copies exist for the PREFILL row kind too: the gate is the GEOMETRY, never this
    // bundle's row count, because every bundle of a model shares ONE staged weight
    // segment. (The empty per-layer map is the wiring pass's own production call shape —
    // `graph_wiring` mints unbanked.)
    let layout_prefill = compute_bundle_layout(&rg, &weight_ids, false, &BTreeMap::new())
        .expect("the mint runs for the prefill row kind");
    assert_eq!(
        layout_prefill.weight_copies,
        vec![(11, perm0), (23, perm1)],
        "the permuted copies are staged for every bundle of the model, read or not"
    );

    // ── 6. THE DOOR A/B — same walk, same layout, one boolean ─────────────────────────────────
    // WITH the handoff: the finalize is one op per feature slab (2, not the shipped 64 =
    // 32 heads × 2 slabs) and the o matmul's matmulfp8 reads its weight at the PERMUTED
    // copy's placement. WITHOUT: the shipped 64-op finalize and the companion placement.
    // The names are the door's own (the fp8 chain ends `…_fq_mm`; the finalize forms carry
    // `attn_o_s{slab}` / `attn_o_h{head}s{slab}`), never a node index.
    let attn_body = rolled
        .body
        .iter()
        .find(|e| e.op_name.starts_with("attn_s"))
        .expect("the body's attention program");
    let with = door_lower_op(&rolled, attn_body, Some(params))
        .expect("the attention lowers under the handoff");
    let slab_finalizes = with
        .iter()
        .filter(|o| o.op_name.contains("attn_o_s"))
        .count();
    let per_head = with
        .iter()
        .filter(|o| o.op_name.contains("attn_o_h"))
        .count();
    assert_eq!(
        slab_finalizes, 2,
        "one finalize op per feature slab (hd/POOL_STICK)"
    );
    assert_eq!(per_head, 0, "the per-head finalize loop is gone");

    let mut shipped = params;
    shipped.headmajor_handoff = false;
    let without = door_lower_op(&rolled, attn_body, Some(shipped))
        .expect("the attention lowers without the handoff (the shipped emission)");
    assert_eq!(
        without
            .iter()
            .filter(|o| o.op_name.contains("attn_o_h"))
            .count(),
        64,
        "the shipped per-head finalize: 32 heads × 2 slabs"
    );
    assert_eq!(
        without
            .iter()
            .filter(|o| o.op_name.contains("attn_o_s"))
            .count(),
        0,
        "the slab-major form is the handoff's alone"
    );

    // The o matmul's program — identified by DATA (the only fp8 matmul in the graph, so
    // the only door-lowered group carrying a `matmulfp8`), not by node index.
    let o_group_with: Vec<EmittedOp> = matmul_body_ops(&rolled)
        .into_iter()
        .map(|e| {
            door_lower_op(&rolled, e, Some(params))
                .unwrap_or_else(|err| panic!("{}: {}", e.op_name, err.message))
        })
        .find(|ops| ops.iter().any(|o| o.op_name.ends_with("fq_mm")))
        .expect("the fp8 o matmul lowers and emits its matmulfp8");
    let o_group_without: Vec<EmittedOp> = matmul_body_ops(&rolled)
        .into_iter()
        .map(|e| {
            door_lower_op(&rolled, e, Some(shipped))
                .unwrap_or_else(|err| panic!("{}: {}", e.op_name, err.message))
        })
        .find(|ops| ops.iter().any(|o| o.op_name.ends_with("fq_mm")))
        .expect("the fp8 o matmul lowers without the handoff too");
    let mm_addresses = |ops: &[EmittedOp]| {
        let mm = ops
            .iter()
            .find(|o| o.op_name.ends_with("fq_mm"))
            .expect("the matmulfp8");
        let dsc = mm
            .dsc()
            .dscs_
            .first()
            .and_then(|m| m.values().next())
            .expect("the matmulfp8 has a descriptor");
        operand_addresses(dsc)
    };
    let with_addrs = mm_addresses(&o_group_with);
    let without_addrs = mm_addresses(&o_group_without);
    // The swap moves the WEIGHT operand's address — and nothing else: every other operand
    // (activation, scale, output) is placed identically both ways. The address may be
    // core-spread, so the law is the SPAN: every address that appears only under the
    // handoff lies inside the permuted copy's placement, and every address that appears
    // only without it lies inside the companion's.
    let perm_addr = SEGMENT_OFFSETS[perm.segment] + perm.offset;
    let comp_addr = SEGMENT_OFFSETS[comp.segment] + comp.offset;
    let moved_in: Vec<u64> = with_addrs
        .iter()
        .copied()
        .filter(|a| !without_addrs.contains(a))
        .collect();
    let moved_out: Vec<u64> = without_addrs
        .iter()
        .copied()
        .filter(|a| !with_addrs.contains(a))
        .collect();
    assert!(
        !moved_in.is_empty(),
        "the handoff must move the weight operand's address"
    );
    assert!(
        moved_in
            .iter()
            .all(|a| (perm_addr..perm_addr + perm.size).contains(a)),
        "every address gained under the handoff lies in the permuted copy's span \
         ({moved_in:?} vs {perm_addr}..{})",
        perm_addr + perm.size
    );
    assert!(
        !moved_out.is_empty(),
        "dropping the handoff must move the weight operand's address back"
    );
    assert!(
        moved_out
            .iter()
            .all(|a| (comp_addr..comp_addr + comp.size).contains(a)),
        "every address lost under the handoff lies in the companion's span \
         ({moved_out:?} vs {comp_addr}..{})",
        comp_addr + comp.size
    );
}

/// A handoff that names no o_proj weight is a MALFORMED BUNDLE STATEMENT — the handoff is
/// minted only where the dataflow walk found the o matmul — and the door refuses it for
/// EVERY matmul program of the bundle, before any body runs: a slab-major finalize with no
/// weight to meet it is exactly the half-restructure this file exists to make impossible.
#[test]
fn a_handoff_without_an_oproj_weight_is_refused() {
    let (rolled, ..) = walk_at_8b();
    let mut params = rolled.attn_params.expect("the bundle states its facts");
    params.oproj_wtid = None; // the handoff itself stays stated
    for e in matmul_body_ops(&rolled) {
        // (`.err()` and not `expect_err`: the Ok type is `Vec<EmittedOp>`, which carries no
        // `Debug`, and the refusal — not the success — is what this pass is about.)
        let err = door_lower_op(&rolled, e, Some(params))
            .err()
            .unwrap_or_else(|| panic!("{}: a handoff without a weight must refuse", e.op_name));
        assert!(
            err.message.contains("names no o_proj weight tid"),
            "{}: {}",
            e.op_name,
            err.message
        );
    }
}

/// A handoff naming a DENSE projection's weight refuses at exactly that projection — the
/// width and precision are the handoff's own (minted for the single-row fp8 W8A8 o
/// matmul), and a swap over an f16 weight reads slab-major bytes against an unpermuted
/// one. Naming layer 0's gate weight (source 13, dense) must refuse the gate projection
/// alone; the o matmul (whose weight tid is 11, not 13) and every other projection lower
/// normally.
#[test]
fn the_handoff_refuses_a_dense_projection_named_as_the_o_proj() {
    let (rolled, ..) = walk_at_8b();
    let mut params = rolled.attn_params.expect("the bundle states its facts");
    params.oproj_wtid = Some(13); // the gate weight — DENSE
    let mut refused = Vec::new();
    for e in matmul_body_ops(&rolled) {
        match door_lower_op(&rolled, e, Some(params)) {
            Ok(_) => {}
            Err(err) => refused.push((e.op_name.clone(), err)),
        }
    }
    assert_eq!(
        refused.len(),
        1,
        "exactly the named weight's matmul refuses"
    );
    let (name, err) = &refused[0];
    assert_eq!(
        name, "matmul_s11",
        "the gate projection (fixture op 10, node 11 under the prologue)"
    );
    assert!(
        err.message.contains("for weight t13,"),
        "the refusal names the bundle's own statement: {}",
        err.message
    );
    assert!(
        err.message.contains("over an f16 weight"),
        "the refusal names the precision mismatch: {}",
        err.message
    );
}

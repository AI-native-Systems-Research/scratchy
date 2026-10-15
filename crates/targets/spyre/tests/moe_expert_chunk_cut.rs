// SPDX-License-Identifier: Apache-2.0
//! THE EXPERT DOOR'S CHUNK CUT, AT THE 26b's OWN GEOMETRY — the LX wall's fix.
//!
//! One gather entry stages its WHOLE page into per-core LX, and the copy's work
//! division splits `mb` by whole ENTRIES only, so the only lever on the
//! residency is the ENTRY'S OWN SIZE. A whole gemma-4-26b gate/up slab is
//! 704·2816 = 1,982,464 fp8 elements ≈ 3.96 MB per core against the
//! 1,677,721-B `USABLE_LX_BYTES` budget — the bake panic this file exists to
//! keep refusals ahead of. The fix cuts the slab into 64-column N-stick CHUNKS
//! and stages the bank CHUNK-MAJOR: expert `e`'s chunk `χ` at
//! `(χ·E + e)·chunk_elems`, so the RAW expert-id index (`skip_addr = chunk`)
//! and the per-chunk source base (`χ·E·chunk`) compose exactly. That cut
//! cleared the RESIDENCY wall; the vendor's LX SCHEDULER is tighter still:
//! `reserveFrontendLx` (ProgramLayout.cpp:81-98) holds back `memCapacity × 0.2`
//! of the 64-KiB-truncated LX, and `allocAllMem` (L3DlOpsScheduler.cpp:6455)
//! charges a gathered op 2×value + 2×output capacity at the one-page minimum
//! chunk — 4 × the page — against the 406,323-B free pool. The 26b gate/up
//! page (1408 sub-rows = 180,224 B) charges 720,896 B: the pod's live
//! `Expect valid chunk size that fits in LX` refusal (L3DlOpsScheduler.cpp:2242,
//! group_15/sdsc_79). The SUB-PAGE cut answers it: when `4·page_w·128`
//! exceeds the budget the door cuts the page into
//! `CopyDims::ENTRIES_PER_OP` = 32 sub-pages, gathering through an `[E, 32]`
//! u32 bank (`embi_`, one op per pair — the replica-run heads are strided
//! `rep·m_pad` elements apart, so no contiguous run covers two) that maps
//! the raw router id to the fine index `e·32 + s`, so each weight gather
//! moves `page_w/32` sub-rows per entry and the charge falls 32×. This file pins that arithmetic at the
//! real geometry:
//!
//! * **the cut**: 11 groups (704/64, prime) ⇒ `chunk_groups = 1`, `chunks = 11`,
//!   `chunk_n = 64`, `page_w = 1408` sub-rows; the vendor charge 4·1408·128 B
//!   = 720,896 B exceeds 406,323 B ⇒ `sub = 32`, `page_sub = 44` sub-rows;
//! * **the gather count and names**: 11 chunks × 8 pairs = 88 weight-gather
//!   ops plus ONE bank gather PER PAIR (`embi_p{p}` — the replica-run heads
//!   are strided `rep·m_pad` u32 elements apart, so no contiguous run covers
//!   two of them and each pair's head entry is its own op); chunk 0 keeps
//!   the per-pair `emwg_p{p}` name, every later chunk carrying
//!   `emwg_c{χ}_p{p}`;
//! * **the source law survives the cut UNCHANGED**: the fine index carries
//!   the pair, not the source — a chunk's eight gathers read the SAME base,
//!   advancing one chunk PLANE (`E·chunk_elems`) per chunk;
//! * **the per-(pair, chunk) matmul windows**: 8 pairs × 11 chunks = 88 matmuls,
//!   pair `p`'s chunk `χ` reading scratch `(χ·pairs + p)·chunk_elems` and
//!   writing output `p·n + χ·chunk_n`;
//! * **the byte-identity control**: at a geometry whose page fits the VENDOR
//!   budget (`page_w ≤ 793` sub-rows) the door emits ONE weight gather, ONE
//!   matmul per pair, no chunk renaming, and NO bank gather — `chunks == 1
//!   && sub == 1` is the expert-major bank, unchanged. The shipped tiny26
//!   proportions (page_w = 2816) now fire the sub-page cut too — the budget
//!   binds there as well, and the cut is what keeps that build vendor-legal.
//!
//! ⛔ WHY THE DOOR PATH AND NOT THE EMULATOR: same law as the sort/combine
//! gates — the emulator runs the KTIR ops directly, so an emu-side test cannot
//! see the descriptors the door bakes. The NUMERIC gate (tiny26 EMU-vs-card
//! tensordump parity) is the port's own and runs on the pod; what belongs HERE
//! is the descriptor-level contract: op count, baked offsets, window shape.

use scratchy_subtile::subtile_ir::{
    ExpertBundle, ExpertProj, NeoX, SubOp, SubtileIR, SubtileId, SubtileNode, TensorId,
    TensorRegion, TensorShape, TopK,
};

/// The 26b gate/up door's own numbers: one token row, eight routed experts,
/// a 128-expert bank of `[704, 2816]` slabs.
const M: u32 = 1;
const K: u32 = 8;
const E: u32 = 128;
const N_OUT: u32 = 704;
const IN: u32 = 2816;

/// The expert-matmul door at the given geometry, wired the way the wavefront
/// wires it: an honest router chain — `RouteArgsort` over the raw logits,
/// `RouteTopK` inverting the ranks into the `[m, k]` indices (P1's
/// provenance law — the sort's ids must be a top-k output), a
/// `RouteGatherScores` closing the chain so the pair's binding walk binds
/// (the argsort and the gather both FOLD into the fused pair, so only the
/// topk program mints) — then an `ExpertSort` produces the pair rows (which
/// ARE the routing tile the matmul's builder binds), and the `ExpertMatmul`
/// reads `[pairs, pairs, w, s]`. Tensors: `t0` x `[m, in]`, `t1` the router
/// logits `[m, e]`, `t2` the fp8 bank `[E·n, in]`, `t3` the scales
/// `[E·n, 1]`, `t4` the sort's pair rows `[m, k·in]`, `t5` the matmul out
/// `[m, k·n]`, `t6` the top-k indices `[m, k]`, `t7` the argsort's rank
/// vector `[m, e]`, `t8` the gather's scores `[m, k]` (t6–t8 appended after
/// the result so every `_o{tid}` pin keeps its id).
fn expert_matmul_ir(
    m: u32,
    k: u32,
    e: u32,
    n_out: u32,
    in_dim: u32,
    proj: ExpertProj,
) -> SubtileIR<NeoX> {
    let nz = |n: u32| std::num::NonZeroU32::new(n).unwrap();
    let tensors = vec![
        TensorShape {
            rows: m,
            cols: in_dim,
        }, // 0: x (source)
        TensorShape { rows: m, cols: e }, // 1: the router logits (source)
        TensorShape {
            rows: e * n_out,
            cols: in_dim,
        }, // 2: the fp8 bank (source)
        TensorShape {
            rows: e * n_out,
            cols: 1,
        }, // 3: the scale bank (source)
        TensorShape {
            rows: m,
            cols: in_dim * k,
        }, // 4: the sort's pair rows
        TensorShape {
            rows: m,
            cols: n_out * k,
        }, // 5: the matmul's out
        TensorShape { rows: m, cols: k }, // 6: the top-k indices (the sort's routing)
        TensorShape { rows: m, cols: e }, // 7: the argsort's rank vector (folds)
        TensorShape { rows: m, cols: k }, // 8: the gather's scores (folds)
    ];
    let tr = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    // The honest router chain's two folding legs: the argsort (its only
    // consumer is the topk) and the gather (its ids come from the topk over
    // the same logits). Both fold into the fused pair, so the pair's binding
    // walk binds and only the topk program mints.
    let argsort = SubtileNode {
        id: SubtileId::from_index(3),
        op: SubOp::RouteArgsort,
        inputs: vec![tr(1)],
        output: tr(7),
    };
    let topk = SubtileNode {
        id: SubtileId::from_index(0),
        op: SubOp::RouteTopK {
            k: TopK::new(nz(k)),
        },
        inputs: vec![tr(7)],
        output: tr(6),
    };
    let gather = SubtileNode {
        id: SubtileId::from_index(4),
        op: SubOp::RouteGatherScores,
        inputs: vec![tr(1), tr(6)],
        output: tr(8),
    };
    let sort = SubtileNode {
        id: SubtileId::from_index(1),
        op: SubOp::ExpertSort {
            experts: scratchy_subtile::subtile_ir::NumExperts::new(nz(e)),
            k: TopK::new(nz(k)),
            bundle: ExpertBundle::SwitchGlu,
        },
        inputs: vec![tr(0), tr(6)],
        output: tr(4),
    };
    let matmul = SubtileNode {
        id: SubtileId::from_index(2),
        op: SubOp::ExpertMatmul {
            proj,
            n: n_out,
            k: TopK::new(nz(k)),
            quant: scratchy_subtile::lower::ExpertQuant::declared(64, 8),
            bundle: ExpertBundle::SwitchGlu,
        },
        inputs: vec![tr(4), tr(4), tr(2), tr(3)],
        output: tr(5),
    };
    SubtileIR {
        result: TensorId::from_index(5),
        tensors,
        num_sources: 4,
        nodes: vec![argsort, topk, gather, sort, matmul],
        op_output: Vec::new(),
    }
}

/// Lower through the REAL door path and return the emitted ops.
fn through_the_door(ir: &SubtileIR<NeoX>) -> Vec<ktir_superdsc::emit::EmittedOp> {
    let weight_ids = std::collections::HashSet::new();
    let (ops, _layout) =
        scratchy_target_spyre::lower_subtile_tape_to_ktir::lower_graph_to_superdsc(
            ir,
            &weight_ids,
            ktir_superdsc::ktir_node::ActiveCap::FULL,
            false,
        )
        .unwrap_or_else(|e| panic!("lower through the door: {e:?}"));
    ops
}

/// The emitted op names, in emission order.
fn op_names(ops: &[ktir_superdsc::emit::EmittedOp]) -> Vec<&str> {
    ops.iter().map(|o| o.op_name.as_str()).collect()
}

/// ⭐ THE CUT ITSELF: at the 26b gate/up geometry the door emits EIGHTY-EIGHT
/// weight-gather ops — 11 chunks × 8 pairs, every gather moving a
/// `page_w/32` = 44-sub-row slice per entry — plus ONE bank gather PER PAIR
/// (`embi_p{p}`, k of them): the replica-run heads the walks address are
/// strided `rep·m_pad` u32 elements apart, so no contiguous run covers two
/// heads and each pair's head entry is its own one-entry op. The whole-page
/// form this replaces was the pod's live `Expect valid chunk size that fits
/// in LX` refusal: the vendor's `allocAllMem` charges 4× the page (720,896 B
/// at page_w=1408) against the 406,323-B post-reserve pool.
#[test]
fn the_26b_gate_slab_cuts_into_eleven_chunks() {
    let ops = through_the_door(&expert_matmul_ir(M, K, E, N_OUT, IN, ExpertProj::Gate));
    let names = op_names(&ops);
    let banks: Vec<&str> = names
        .iter()
        .copied()
        .filter(|n| n.starts_with("embi_"))
        .collect();
    assert_eq!(
        banks.len(),
        K as usize,
        "one bank gather per pair — each pair's head entry is its own op: {banks:?}"
    );
    for (p, name) in banks.iter().enumerate() {
        assert_eq!(
            *name,
            format!("embi_p{p}_o6").as_str(),
            "the bank gather's name"
        );
    }
    let gathers: Vec<&str> = names
        .iter()
        .copied()
        .filter(|n| n.starts_with("emwg_"))
        .collect();
    assert_eq!(
        gathers.len(),
        88,
        "11 chunks × 8 pairs under the sub-page cut: {gathers:?}"
    );
    for (p, name) in gathers.iter().take(K as usize).enumerate() {
        assert_eq!(
            *name,
            format!("emwg_p{p}_o5").as_str(),
            "chunk 0 keeps the per-pair name"
        );
    }
    for c in 1..11usize {
        for (p, name) in gathers
            .iter()
            .skip(c * K as usize)
            .take(K as usize)
            .enumerate()
        {
            assert_eq!(
                *name,
                format!("emwg_c{c}_p{p}_o5").as_str(),
                "every later chunk carries its chunk AND pair index"
            );
        }
    }
}

/// ⭐ THE WINDOW COUNT: eight pairs × eleven chunks = EIGHTY-EIGHT per-pair
/// matmuls. The whole-slab form emitted eight.
#[test]
fn the_26b_matmuls_are_per_pair_per_chunk() {
    let ops = through_the_door(&expert_matmul_ir(M, K, E, N_OUT, IN, ExpertProj::Gate));
    let names = op_names(&ops);
    let matmuls: Vec<&str> = names
        .iter()
        .copied()
        .filter(|n| n.starts_with("emmm_"))
        .collect();
    assert_eq!(
        matmuls.len(),
        (K * 11) as usize,
        "8 pairs × 11 chunks: {matmuls:?}"
    );
    // ⭐ THE MATMULS NAME UNIFORMLY: every per-(pair, chunk) matmul carries
    // BOTH indices (the gather keeps the shipped name at chunk 0; the matmuls
    // do not — the chunk cut is their shape, not an offset rename).
    assert_eq!(matmuls[0], "emmm_p0_c0_o5", "pair 0, chunk 0");
    assert_eq!(matmuls[1], "emmm_p0_c1_o5", "pair 0's chunk 1");
    assert_eq!(matmuls[11], "emmm_p1_c0_o5", "pair 1, chunk 0");
}

/// ⭐ THE GATHER'S BAKED OFFSETS, read off the emitted descriptors as DELTAS:
/// chunk χ's source operand carries the chunk-major base `χ·E·chunk_elems`
/// (the source half of `e·chunk + χ·E·chunk = (χ·E + e)·chunk`), folded into
/// the operand's HBM start by `resolve_seg_base` — so the law is visible in
/// the emitted addresses as the DIFFERENCE between consecutive chunks' gather
/// sources: exactly one chunk PLANE (`E·chunk_elems` fp8 elements, one byte
/// each) per chunk, independent of where the bank itself was placed. The
/// sub-page cut must not disturb it: the PAIR lives in the fine index the
/// `embi_` gather mints, so all eight gathers of one chunk read the SAME
/// source base.
#[test]
fn the_26b_chunk_sources_advance_by_a_chunk_plane() {
    let ops = through_the_door(&expert_matmul_ir(M, K, E, N_OUT, IN, ExpertProj::Gate));
    let chunk_elems = 64 * IN; // chunk_n(64) × in(2816)
    let gathers: Vec<&ktir_superdsc::emit::EmittedOp> = ops
        .iter()
        .filter(|o| o.op_name.starts_with("emwg_"))
        .collect();
    assert_eq!(gathers.len(), 88);
    let base = |c: usize, p: usize| gather_source_base_bytes(gathers[c * K as usize + p]);
    // The chunk plane term is E·chunk_elems = the whole bank's per-chunk
    // slice; chunk χ's source base is χ times that.
    let plane = E as u64 * chunk_elems as u64;
    let start0 = base(0, 0);
    for c in 0..11usize {
        for p in 0..K as usize {
            let delta = base(c, p) - start0;
            assert_eq!(
                delta,
                c as u64 * plane,
                "chunk {c}'s source is {c}·E·chunk_elems ({c}·{plane} B) past chunk 0's"
            );
            assert_eq!(
                base(c, p),
                base(c, 0),
                "pair {p}'s gather reads pair 0's source — the pair is in the index"
            );
        }
    }
}

/// The gathered SOURCE operand's own start, in BYTES — core 0's HBM start for
/// the first input operand. `resolve_seg_base` folds the door's
/// `offset_elems` into the segment base the allocator hands out, so this is
/// the base PLUS the placement's own segment start; the LAWS read the deltas
/// between ops' starts (the placement term cancels).
fn gather_source_base_bytes(o: &ktir_superdsc::emit::EmittedOp) -> u64 {
    let dsc = o.dsc();
    let d = dsc
        .dscs_
        .first()
        .and_then(|m| m.values().next())
        .expect("an emitted op has at least one dsc");
    let in_lds = d
        .computeOp_
        .first()
        .and_then(|c| c.inputLabeledDs.first())
        .cloned()
        .unwrap_or_default();
    let idx: u32 = in_lds
        .rsplit("idx")
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(u32::MAX);
    let alloc = d
        .scheduleTree_
        .iter()
        .find(|n| n.nodeType_ == "allocate" && n.component_ == "hbm" && n.ldsIdx_ == idx)
        .unwrap_or_else(|| panic!("the source's hbm allocate (lds {idx}, from {in_lds:?})"));
    alloc
        .startAddressCoreCorelet_
        .data_
        .get("[0, 0, 0]")
        .and_then(|s| s.parse().ok())
        .expect("core 0's start address")
}

/// ⭐ THE BYTE-IDENTITY CONTROL: at a geometry whose page fits the VENDOR
/// sched budget (`page_w ≤ 793` sub-rows — 4·page_w·128 B ≤ 406,323 B) the
/// door emits ONE weight gather PER PAIR (the head-entry law — the uncut
/// walk is per-pair exactly like the cut one), ONE matmul per pair, NO
/// chunk renaming, and NO bank gather. The 26b fix must be invisible to
/// every model whose pages fit — the names and counts this pins are the
/// per-pair law at the uncut page unit.
#[test]
fn a_slab_that_fits_whole_stays_whole() {
    // n=64 (one N-group) at in=1408: page_w = 1·64·1408/128 = 704 sub-rows,
    // 4·704·128 B = 360,448 B under the 406,323-B budget ⇒ no sub-page cut.
    let (e, k, n_out, in_dim) = (4u32, 2u32, 64u32, 1408u32);
    let ops = through_the_door(&expert_matmul_ir(M, k, e, n_out, in_dim, ExpertProj::Gate));
    let names = op_names(&ops);
    assert_eq!(
        names.iter().filter(|n| n.starts_with("embi_")).count(),
        0,
        "no bank gather when the page fits the vendor budget"
    );
    let gathers: Vec<&str> = names
        .iter()
        .copied()
        .filter(|n| n.starts_with("emwg_"))
        .collect();
    assert_eq!(
        gathers.len(),
        k as usize,
        "one gather per pair: {gathers:?}"
    );
    for (p, g) in gathers.iter().enumerate() {
        assert_eq!(*g, format!("emwg_p{p}_o5").as_str(), "the per-pair name");
    }
    let matmuls: Vec<&str> = names
        .iter()
        .copied()
        .filter(|n| n.starts_with("emmm_"))
        .collect();
    assert_eq!(matmuls.len(), k as usize, "one matmul per pair");
    for (p, m) in matmuls.iter().enumerate() {
        assert_eq!(*m, format!("emmm_p{p}_o5").as_str(), "no chunk renaming");
    }
}

/// ⭐ THE CONTROL'S INVERSE, at the SHIPPED tiny26 proportions: page_w = 2816
/// sub-rows ⇒ 4·page_w·128 B = 1,441,792 B over the 406,323-B vendor budget,
/// so the sub-page cut fires there too — one chunk (2 N-groups), TWO per-pair
/// gathers plus the bank. The cut is what keeps that build vendor-legal; this
/// pins the emission the pod's tiny26 gate now produces so `embi_`/`emwg_p`
/// names in its bundle are expected, not a regression.
#[test]
fn the_tiny26_proportions_fire_the_subpage_cut_too() {
    // tiny26 gate/up proportions: n=128 (2 groups), in=2816, E=4, k=2.
    let (e, k, n_out, in_dim) = (4u32, 2u32, 128u32, 2816u32);
    let ops = through_the_door(&expert_matmul_ir(M, k, e, n_out, in_dim, ExpertProj::Gate));
    let names = op_names(&ops);
    assert_eq!(
        names.iter().filter(|n| n.starts_with("embi_")).count(),
        k as usize,
        "one bank gather per pair — the head entries: {names:?}"
    );
    let gathers: Vec<&str> = names
        .iter()
        .copied()
        .filter(|n| n.starts_with("emwg_"))
        .collect();
    assert_eq!(
        gathers.len(),
        k as usize,
        "one chunk × one gather per pair: {gathers:?}"
    );
    for (p, g) in gathers.iter().enumerate() {
        assert_eq!(*g, format!("emwg_p{p}_o5").as_str());
    }
}

/// An INPUT operand's own core-0 HBM start, in BYTES — the same law
/// `gather_source_base_bytes` states (an arg's `offset_elems` folds into the
/// segment base), read here for the per-pair matmul's A operand (the chain's
/// quantized `afp8`), so the per-pair tail's window laws are visible in the
/// emitted addresses.
fn input_operand_start_bytes(o: &ktir_superdsc::emit::EmittedOp, needle: &str) -> u64 {
    let dsc = o.dsc();
    let d = dsc
        .dscs_
        .first()
        .and_then(|m| m.values().next())
        .expect("an emitted op has at least one dsc");
    for node in d
        .scheduleTree_
        .iter()
        .filter(|n| n.nodeType_ == "allocate" && n.component_ == "hbm")
    {
        let binding = &o.arg_bindings[node.ldsIdx_ as usize];
        if binding.is_input && binding.buffer.contains(needle) {
            return node
                .startAddressCoreCorelet_
                .data_
                .get("[0, 0, 0]")
                .and_then(|s| s.parse().ok())
                .expect("core 0's start address");
        }
    }
    panic!("{}: no input AllocNode matching {}", o.op_name, needle);
}

/// ⭐⭐⭐⭐⭐ THE 26b DOWN GEOMETRY (in=704, the half-stick pair stride): the door's own
/// numbers for the projection the in-pad exists for — inn=704 = 5.5 fp8 sticks, out=2816,
/// E=128, k=8. Before the in-pad this door REFUSED at the stick guard (the pod's live
/// `expertmatmul_s34` refusal: "in=704 is not a whole 64-fp16/128-fp8 stick"); with the
/// pad it emits, and every law the pad creates is visible in the emission:
///   · the CHUNK CUT at the padded width — 44 N-groups cut into 4 chunks of 11 groups
///     (chunk_n=704 at inn_pad=768), so 4 chunks × 8 pairs = 32 SUB-PAGE weight gathers
///     (page_w=4224 ⇒ sub=32) plus the bank gather, and 8×4 = 32 per-(pair, chunk)
///     matmuls;
///   · the PER-PAIR TAIL — EIGHT `qfp8ch` converts plus ONE zero-fill (the quantize
///     chain's per-pair padded law);
///   · the A operands start on whole 128-B sticks and advance by the PADDED 768-B
///     stride — the half-stick start this layout exists to make unexpressible;
///   · the gather sources advance by a whole CHUNK PLANE `E·chunk_elems` per chunk,
///     the same law the gate/up cut states at its own geometry.
#[test]
fn the_26b_down_in704_pads_the_pairs_and_cuts_four_chunks() {
    let (n_out, in_dim) = (2816u32, 704u32);
    let ops = through_the_door(&expert_matmul_ir(M, K, E, n_out, in_dim, ExpertProj::Down));
    let names = op_names(&ops);
    // THIRTY-TWO gathers — 4 chunks × 8 pairs under the sub-page cut.
    let gathers: Vec<&str> = names
        .iter()
        .copied()
        .filter(|n| n.starts_with("emwg_"))
        .collect();
    assert_eq!(
        gathers.len(),
        32,
        "44 groups cut into 4 chunks of 11 at the padded in, × 8 pairs: {gathers:?}"
    );
    for (p, name) in gathers.iter().take(K as usize).enumerate() {
        assert_eq!(
            *name,
            format!("emwg_p{p}_o5").as_str(),
            "chunk 0 keeps the per-pair name"
        );
    }
    for c in 1..4usize {
        for (p, name) in gathers
            .iter()
            .skip(c * K as usize)
            .take(K as usize)
            .enumerate()
        {
            assert_eq!(*name, format!("emwg_c{c}_p{p}_o5").as_str());
        }
    }
    // The sub-page bank gathers: ONE op PER PAIR — the replica-run heads are
    // strided `rep·m_pad` u32 elements apart, so no contiguous run covers two.
    assert_eq!(
        names.iter().filter(|n| n.starts_with("embi_")).count(),
        K as usize,
        "the [E, 32] fine-index bank is one one-entry gather per pair"
    );
    // THIRTY-TWO per-(pair, chunk) matmuls, both indices in every name.
    let matmuls: Vec<&str> = names
        .iter()
        .copied()
        .filter(|n| n.starts_with("emmm_"))
        .collect();
    assert_eq!(matmuls.len(), (K * 4) as usize, "8 pairs × 4 chunks");
    assert_eq!(matmuls[0], "emmm_p0_c0_o5");
    assert_eq!(matmuls[31], "emmm_p7_c3_o5");
    // The scale gathers: ONE op PER PAIR — the same head-entry law.
    assert_eq!(
        names.iter().filter(|n| n.starts_with("emsg_")).count(),
        K as usize,
        "the scale bank is expert-major, uncut, one head-entry gather per pair"
    );
    // The per-pair tail: EIGHT converts (pair 0 shipped, 1..7 indexed) and ONE fill.
    let converts = ops
        .iter()
        .filter(|o| o.op_name.ends_with("fq_afp8_op") || o.op_name.contains("fq_afp8_p"))
        .count();
    assert_eq!(
        converts, K as usize,
        "one qfp8ch per pair (the padded tail)"
    );
    assert_eq!(
        ops.iter()
            .filter(|o| o.op_name.ends_with("fq_zfpad_op"))
            .count(),
        1,
        "exactly one zero-fill — the last pair's overhang"
    );
    // The A windows: every start a whole 128-B stick, every step the PADDED stride.
    let a_starts: Vec<u64> = (0..K)
        .map(|p| {
            let mmn = format!("emmm_p{p}_c0_o5");
            let op = ops
                .iter()
                .find(|o| o.op_name == mmn)
                .unwrap_or_else(|| panic!("the matmul {mmn} exists"));
            input_operand_start_bytes(op, "fq_afp8")
        })
        .collect();
    for s in &a_starts {
        assert_eq!(
            s % 128,
            0,
            "every A window starts on a whole 128-B fp8 stick"
        );
    }
    for p in 1..a_starts.len() {
        assert_eq!(
            a_starts[p] - a_starts[p - 1],
            768,
            "pair {p}'s A window advances by the PADDED stride (768 B), not the logical 704"
        );
    }
    // The gather sources advance by one CHUNK PLANE per chunk at the padded
    // chunk size, and the PAIR is carried by the fine index, not the source.
    let chunk_elems = 704u64 * 768; // chunk_n(704) × inn_pad(768)
    let plane = E as u64 * chunk_elems;
    let gather_ops: Vec<&ktir_superdsc::emit::EmittedOp> = ops
        .iter()
        .filter(|o| o.op_name.starts_with("emwg_"))
        .collect();
    let base = |c: usize, p: usize| gather_source_base_bytes(gather_ops[c * K as usize + p]);
    let start0 = base(0, 0);
    for c in 0..4usize {
        for p in 0..K as usize {
            assert_eq!(
                base(c, p) - start0,
                c as u64 * plane,
                "chunk {c}'s source is {c}·E·chunk_elems ({c}·{plane} B) past chunk 0's"
            );
            assert_eq!(
                base(c, p),
                base(c, 0),
                "pair {p}'s gather reads pair 0's source — the index, not the base, is per-pair"
            );
        }
    }
}

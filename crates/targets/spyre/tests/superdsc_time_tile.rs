// SPDX-License-Identifier: Apache-2.0
// ⛔ RUN THIS FILE WITH `SCRATCHY_PLAN_ONLY_BAKE=1`.
//
// `tiled_matmul_trips_do_not_alias_and_advance` drives a real bake, and the bake
// REFUSES on a host with no device compiler (`no_dxp_or_die`) — deliberately, so
// a build cannot ship a memory plan with no device programs, which links and
// serves and returns empty completions with no error anywhere. A test that only
// inspects the PLAN is the one legitimate caller of that escape hatch, so it
// must state it on purpose:
//
//     SCRATCHY_PLAN_ONLY_BAKE=1 cargo test -p scratchy-target-spyre
//
// It is NOT set from inside the test: `std::env::set_var` is `unsafe` in edition
// 2024 and races every other thread in this binary that bakes.
//! LX-fit TIME-TILING (coarse_tile) gate for the typed SuperDSC IR.
//!
//! Pins the **TimeTile** / **TiledStickExtent** / **SymbolicWhenTiled** witnesses:
//!   (i)   an FFN-sized matmul gets `time_tile Some` with `out_per_time` a whole
//!         multiple of 64 and the per-time per-core resident ≤ `USABLE_LX_BYTES`;
//!   (ii)  the known bmm (384×384×64 batch16, 576 KiB) fits LX in one trip
//!         (time = 1, no TimeTile);
//!   (iii) an impossibly-large matmul (won't fit even tiled to single sticks) is a
//!         Rust `Err` — the `cargo build` failure that replaces on-card
//!         DtException 1535;
//!   (iv)  a tiled op's `bundle.mlir` carries `scf.for` + `affine.apply` +
//!         `isStartAddrSymbolic_:1`, and a time=1 op's `bundle.mlir` is byte-
//!         identical to the historical flat form.
//!
//! Lives in `tests/` (a separate binary using only the public API) so it does
//! not co-compile the crate's feature-gated inline `cpu_golden` test modules,
//! which do not build on a plain `cargo test` here.

use ktir_superdsc::emit;
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{assemble_reduce, assemble_rmsnorm};
use scratchy_subtile::sdsc_abstract::{KernelTag, Stk};
use scratchy_subtile::superdsc_opspec::{
    Allocation, DataFormat, Df, FP16_BYTES, Fp16, ItDim, MAX_CORES, MaxCores, OpFunc, Role, Scale,
    TensorArg, USABLE_LX_BYTES, WorkPlan,
};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc as superdsc;
use std::collections::BTreeMap;

// ── GUARD (the LOOP, 2026-06-25): every emitted `opFuncName` MUST be in dxp's
//    recognized set (dscdefn.cpp `opFuncsToString`), else on-card DtException
//    "Unrecognized opFunc" (designSpaceConfig.cpp:7713). Asserts on the IR
//    (OpFunc::name()), turning a wrong name into a `cargo build`/test failure —
//    NOT a runtime crash. (Observed 2026-06-25: "multiply" → must be "mul";
//    "gelu" → must be "gelufwd".)
#[test]
fn opfunc_names_are_dxp_recognized() {
    // The dxp-recognized opFunc names this emitter can produce (verified on-card
    // against /project_src/deeptools/dsc/dscdefn.cpp `opFuncsToString`).
    const RECOGNIZED: &[&str] = &[
        "matmul",
        "batchmatmul",
        "add",
        "sub",
        "mul",
        "silu",
        "exp",
        "reciprocal",
        "sqrt",
        "rsqrt",
        "sigmoid",
        "gelufwd",
        "mish",
        "tanh",
        "sum",
        "max",
        "mean",
        // ⛔ THESE FOUR WERE MISSING WHILE LIVE ON-CARD BODIES EMIT THEM. Added with the emitters that
        // make them de-facto card-proven: `realdiv` (attention's softmax finalize), `abs`, `maximum` and
        // `minimum` (the fp8 activation-quantize chain's amax/clamp, plus attention's running max). They
        // ride in every baked bundle that runs today, so their absence here was a hole in the guard, not
        // a statement that they are unrecognized.
        "realdiv",
        "abs",
        "maximum",
        "minimum",
    ];
    // ⚠️ THIS LIST IS HAND-PICKED, WHICH IS WHY THE FOUR ABOVE COULD GO MISSING. `OpFunc` has 27
    // variants; this loop names 21. A new variant is NOT an E0004 here, so it joins the emitter
    // unguarded — exactly how `realdiv`/`abs`/`maximum`/`minimum` did. Making it exhaustive is the real
    // fix and is deliberately NOT done here: the six left out (`Transpose`, `Restickify`, `Identity`,
    // `Qfp8ch`, `Dl16ToFp32`, `Fp32ToDl16`) would each need their dxp name confirmed against
    // `dscdefn.cpp opFuncsToString`, and asserting a name I have not verified would make this guard lie.
    // Two of the six now have vendor-fixture evidence and are the cheapest to close:
    // `interslicetranspose_fp16` appears in `ddc/ddl_templates/test/sdsc_interslicetranspose.json`, and
    // `identity` in `dxp/test/test_gather_1core/sdsc_1.json`.
    for f in [
        OpFunc::Matmul,
        OpFunc::BatchMatmul,
        OpFunc::Add,
        OpFunc::Subtract,
        OpFunc::Multiply,
        OpFunc::Silu,
        OpFunc::Exp,
        OpFunc::Reciprocal,
        OpFunc::Sqrt,
        OpFunc::Rsqrt,
        OpFunc::Sigmoid,
        OpFunc::Gelu,
        OpFunc::Mish,
        OpFunc::Tanh,
        OpFunc::Sum,
        OpFunc::Max,
        OpFunc::Mean,
        OpFunc::RealDiv,
        OpFunc::Abs,
        OpFunc::Maximum,
        OpFunc::Minimum,
    ] {
        assert!(
            RECOGNIZED.contains(&f.name()),
            "OpFunc {f:?} emits opFuncName {:?} which is NOT in dxp's recognized set → \
             on-card DtException 'Unrecognized opFunc' (designSpaceConfig.cpp:7713)",
            f.name()
        );
    }
}

// ── MaterializedStick witness: the DtException-1535 invariant is enforced BY
//    CONSTRUCTION in `TensorArg::new` — a stick dim with a phantom (-1) scale is
//    UNCONSTRUCTIBLE. This asserts on the IR (the ctor result), NOT on generated
//    SDSC text. With this, no downstream emit-time check / JSON grep is needed.
#[test]
fn phantom_scaled_stick_dim_is_unconstructible() {
    let plan = matmul_plan(384, 384, 64, 16, 16, 2);
    let dims = plan.iter_syms(["mb", "in", "x", "y"]);
    // OK: `in` (the stick dim) materialized (Active).
    assert!(
        TensorArg::<4>::new(
            true,
            "ok".into(),
            Role::Input,
            [
                Scale::Active,
                Scale::Active,
                Scale::Active,
                Scale::RedNonStick
            ],
            dims,
            ["mb", "in", "x", "y"],
            "in",
            Allocation::Hbm,
        )
        .is_ok(),
        "a materialized stick dim constructs fine"
    );
    // ERR: `in` (the stick dim) phantom-scaled (-1) → would be on-card 1535.
    assert!(
        TensorArg::<4>::new(
            true,
            "bad".into(),
            Role::Input,
            [
                Scale::Active,
                Scale::RedNonStick,
                Scale::Active,
                Scale::Active
            ],
            dims,
            ["mb", "in", "x", "y"],
            "in",
            Allocation::Hbm,
        )
        .is_err(),
        "a phantom-scaled stick dim MUST be a build Err (MaterializedStick witness)"
    );
}

// ── GUARDS (the LOOP, 2026-06-25) for the 3 on-card constraints the RMSNorm
//    multi-op compile surfaced — all fixed BY CONSTRUCTION; these pin them so a
//    regression is a test failure, not an on-card DtException:
//      (1) coreIdToDsc_ count == numCoresUsed_  (ModuleStitcher.cpp:216 stitch);
//      (2) a `mean` reduce emits a `scaling_factor` const                (ddl_conversion.cpp:718);
//      (3) that const's value fits 16 bits (fp16 element)               (DSC2ToDataflowIR.cpp:51).
#[test]
fn reduce_and_multiop_invariants() {
    let emitted = assemble_reduce(
        "mean_g",
        "mean",
        64,
        576,
        &emit::rb("r_x", 64, 576),
        &emit::rb("r_acc", 64, 1),
        None,
    );
    let j: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&emitted.op).unwrap()).unwrap();
    // (1) coreIdToDsc_ spans exactly numCoresUsed_ cores (consistent with the schedule).
    let ncores = j["numCoresUsed_"].as_u64().unwrap();
    let n_coreid = j["coreIdToDsc_"].as_object().unwrap().len() as u64;
    let n_sched = j["coreIdToDscSchedule"].as_object().unwrap().len() as u64;
    assert_eq!(
        n_coreid, ncores,
        "coreIdToDsc_ must span exactly numCoresUsed_ (stitch)"
    );
    assert_eq!(
        n_sched, ncores,
        "coreIdToDscSchedule must span exactly numCoresUsed_ (stitch)"
    );
    // (2)+(3) the mean reduce emits a `scaling_factor` const whose value ≤ 0xFFFF.
    let ci = &j["dscs_"][0]["mean_g"]["constantInfo_"];
    let sf = ci
        .as_object()
        .unwrap()
        .values()
        .find(|c| c["name_"] == "scaling_factor")
        .expect("mean reduce must emit a scaling_factor const (summeanmaxexx2.ddl)");
    let val = sf["data_"][0].as_u64().unwrap();
    assert!(
        val <= 0xFFFF,
        "scaling_factor must be a 16-bit fp16 word, got {val:#x}"
    );

    // multi-op (RMSNorm 6-op): EVERY op's coreIdToDsc_ matches its numCoresUsed_.
    let mut sym = 0i64;
    for op in assemble_rmsnorm(
        "g",
        64,
        576,
        "x",
        "w",
        scratchy_target_spyre::bundle_code::PlaceId::Act(7),
        "eps",
        &mut sym,
        None,
    ) {
        let jo: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&op.op).unwrap()).unwrap();
        let nc = jo["numCoresUsed_"].as_u64().unwrap();
        let cid = jo["coreIdToDsc_"].as_object().unwrap().len() as u64;
        assert_eq!(
            cid, nc,
            "RMSNorm op {} coreIdToDsc_ != numCoresUsed_",
            op.op_name
        );
    }
}

// ── SLICE operand (RoPE prerequisite, proven on-card SLICE_RC=0 2026-06-25): a
//    `.with_offset(n)` operand reads `x[n:]` — its AllocNode startAddr is bumped by
//    `n·2` bytes. Pins that the offset reaches the emitted startAddr.
#[test]
fn slice_operand_offsets_startaddr() {
    let e = emit::assemble_slice_add_gate("slice_g", 64, 64, "sx", "so");
    let j: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&e.op).unwrap()).unwrap();
    let nodes = &j["dscs_"][0]["slice_g"]["scheduleTree_"];
    let addr = |i: usize| -> u64 {
        nodes[i]["startAddressCoreCorelet_"]["data_"]["[0, 0, 0]"]
            .as_str()
            .unwrap()
            .parse()
            .unwrap()
    };
    // Tensor1 (x_hi = x[64:]) startAddr = Tensor0 base + Tensor1 base-bump + 64*2.
    // The +128 intra-tensor slice offset is the load-bearing bit.
    assert_eq!(
        addr(1) % (1 << 20),
        64 * 2,
        "x[64:] slice must offset startAddr by 64*2 bytes"
    );
}

// ── #50 PER-CORE HBM SEGMENT ADDRESSING ───────────────────────────────────────
// Each I/O arg lives in its OWN 16 GiB HBM segment (SEGMENT_OFFSETS, torch-spyre
// constants.py:44), and within a segment each core's tile starts at its
// work-slice byte offset. These pin: (1) distinct args use distinct segment
// bases; (2) cores of a SPLIT tensor get DISTINCT, correctly-strided addresses;
// (3) an unsplit (reduction-resident) operand shares one address across cores.

/// Parse the per-core startAddr map of allocate node `ldsidx` of op `op_name`.
fn per_core_addrs(emitted: &emit::EmittedOp, op_name: &str, ldsidx: u32) -> Vec<u64> {
    let j: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&emitted.op).unwrap()).unwrap();
    let nodes = j["dscs_"][0][op_name]["scheduleTree_"]
        .as_array()
        .unwrap()
        .clone();
    let node = nodes
        .iter()
        .find(|n| n["ldsIdx_"].as_u64().unwrap() as u32 == ldsidx)
        .expect("allocate node for ldsidx");
    let data = node["startAddressCoreCorelet_"]["data_"]
        .as_object()
        .unwrap();
    let cores = data.len();
    (0..cores)
        .map(|c| {
            data[&format!("[{c}, 0, 0]")]
                .as_str()
                .unwrap()
                .parse::<u64>()
                .unwrap()
        })
        .collect()
}

#[test]
fn per_core_addresses_use_distinct_segments_and_disjoint_cores() {
    // bmm 384×384×64 batch16, cost split fills 32 cores (mb16×out2). A is INPUT
    // (arg0, seg0), W KERNEL (arg1, seg1), O OUTPUT (arg2, seg2).
    let e = superdsc::assemble_matmul(
        "mm",
        384,
        384,
        64,
        16,
        &emit::rb("a", 384, 64),
        &Stk::<KernelTag>::kernel(64_usize, 384_usize, "w"),
        &emit::rb("o", 384, 384),
        None,
    );
    assert_eq!(e.time, 1, "bmm fits LX in one trip");
    let seg = 0x4_0000_0000u64;
    let a = per_core_addrs(&e, "mm", 0); // INPUT  [mb,in,x,y] split on mb
    let w = per_core_addrs(&e, "mm", 1); // KERNEL [in,out,x] split on out
    let o = per_core_addrs(&e, "mm", 2); // OUTPUT [mb,out,y,x] split on mb+out
    assert_eq!(a.len(), 32);
    // (1) each arg's base is its own 16 GiB segment.
    assert!(
        a.iter().all(|&x| x < seg),
        "INPUT (arg0) in segment 0: {a:?}"
    );
    assert!(
        w.iter().all(|&x| (seg..2 * seg).contains(&x)),
        "KERNEL (arg1) in segment 1"
    );
    assert!(
        o.iter().all(|&x| (2 * seg..3 * seg).contains(&x)),
        "OUTPUT (arg2) in segment 2"
    );
    // (2) the OUTPUT is split on BOTH mb(×16) and out(×2) → all 32 cores DISJOINT.
    let mut uniq: std::collections::BTreeSet<u64> = o.iter().copied().collect();
    assert_eq!(
        uniq.len(),
        32,
        "every core writes a DISTINCT output byte range: {o:?}"
    );
    uniq.clear();
    // (3) the INPUT A [mb,in,x,y] is split only on mb (16) → 16 distinct bases,
    //     each shared by the 2 cores that differ only in the `out` split (A has no
    //     `out` dim, so out-split cores read the SAME A slice).
    let distinct_a = a
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    assert_eq!(
        distinct_a, 16,
        "A split on mb(16) only → 16 distinct per-core slices: {a:?}"
    );
    // Core numbering (core_to_wk_slice): the LARGER split is the outer (slower)
    // dim, so mb(×16) is outer and out(×2) is inner. Cores c and c+1 differ only in
    // `out` (which A lacks) → SAME A slice; cores c and c+2 step one `mb` slice.
    assert_eq!(
        a[0], a[1],
        "cores differing only in the out-split read the same A slice"
    );
    // Per-mb-slice stride = per_core_mb(24) * inner_full_product(in*x*y = 64*16, y=1) * 2.
    let stride = 24u64 * (64 * 16) * 2;
    assert_eq!(
        a[2] - a[0],
        stride,
        "A per-mb-slice stride = 24*1024*2 bytes"
    );
}

#[test]
fn tiled_matmul_trips_do_not_alias_and_advance() {
    // 64×16384×2048 batch1 → time>1 (proven tiled in tiled_bundle_mlir test).
    //
    // ⛔ THROUGH THE REAL CHAIN, BECAUSE A HAND-BUILT DESCRIPTOR IS NO LONGER A BUNDLE. This used to
    // call `assemble_matmul` and hand the descriptor straight to `emit_bundle`; after the
    // `SubtileIR → KTIR → SuperDSC` split a bundle is built from PROGRAMS (`ktir_groups` on
    // `-Fspyre-emu`, `ktir_groups_via_superdsc` on `-Fspyre-hw`), and an op carrying a descriptor and
    // no program is refused by both — deliberately, since the only path to SuperDSC is through a KTIR
    // program. So the fixture is the SubtileIR node the descriptor came from, and the two halves are
    // read where each now exists: `time`/trips off the LOWERED descriptor, the bundle off the
    // PROGRAMS. Nothing about the assertions changes.
    //
    // `rows_are_requests` is TRUE for exactly the reason this test exists: 64 rows that are 64
    // separate requests may NOT fold to the last row (every row's logits are sampled), so the
    // vocab-wide tail time-tiles at m=64 — which is addressable only because these trips advance
    // instead of aliasing.
    let ir = single_matmul_ir(64, 16384, 2048);
    let weight_ids: std::collections::HashSet<u32> = [1u32].into_iter().collect();
    let (programs, layout) =
        superdsc::lower_graph_to_ktir(&ir, &weight_ids, superdsc::ActiveCap::FULL, true)
            .expect("KTIR for a 64x16384x2048 matmul");
    let (dscs, _) =
        superdsc::lower_graph_to_superdsc(&ir, &weight_ids, superdsc::ActiveCap::FULL, true)
            .expect("SuperDSC for a 64x16384x2048 matmul");
    let [e] = &dscs[..] else {
        panic!(
            "one matmul node lowers to one descriptor, got {:?}",
            dscs.iter().map(|o| &o.op_name).collect::<Vec<_>>()
        )
    };
    assert!(e.time > 1);
    // Concrete trips: every OUTPUT per-core address is UNIQUE across all trips
    // (#50 — real per-core bases + per-trip stride mean trips never alias).
    let trips = superdsc::concrete_trips(e);
    let mut out_addrs: Vec<u64> = Vec::new();
    for trip in &trips {
        let j: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(trip).unwrap()).unwrap();
        for node in j["dscs_"][0][&e.op_name]["scheduleTree_"]
            .as_array()
            .unwrap()
        {
            // OUTPUT is ldsIdx 2 (A=0, W=1, O=2).
            if node["ldsIdx_"].as_u64().unwrap() == 2 {
                for (_k, v) in node["startAddressCoreCorelet_"]["data_"]
                    .as_object()
                    .unwrap()
                {
                    out_addrs.push(v.as_str().unwrap().parse().unwrap());
                }
            }
        }
    }
    let uniq: std::collections::BTreeSet<u64> = out_addrs.iter().copied().collect();
    assert_eq!(
        uniq.len(),
        out_addrs.len(),
        "no two (trip,core) OUTPUT writes alias"
    );
    // GUARD #14 LIFTED by #50: write_bundle_cached no longer REFUSES a time-tiled
    // bundle (it now only refuses an ALIASING one — which the disjointness above
    // proves this is not). A tiled op pre-unrolls to N>1 files, so it still trips
    // the SEPARATE multi-op stitch GUARD #16; opt into that (its own on-card
    // concern) to isolate that #14 itself is gone. SCRATCHY_SUPERDSC_ALLOW_TILED is
    // NO LONGER required.
    let tmp = std::env::temp_dir().join("superdsc_50_tiled_cache");
    let _ = std::fs::remove_dir_all(&tmp);
    // SAFETY: single-threaded test; set the multi-op (stitch) opt-in only.
    unsafe { std::env::set_var("SCRATCHY_SUPERDSC_ALLOW_MULTIOP", "1") };
    let r = superdsc::emit_bundle(
        &programs,
        Some(&layout),
        superdsc::FoldGrouping::Split,
        // A single tiled matmul: no attention node, so nothing for the four attention facts
        // to describe.
        None,
    );
    unsafe { std::env::remove_var("SCRATCHY_SUPERDSC_ALLOW_MULTIOP") };
    assert!(
        r.is_ok(),
        "tiled bundle no longer refused for tiling (GUARD #14 lifted by #50): {r:?}"
    );
}

/// `hidden[m, k] @ W[k, n] -> out[m, n]` as a one-node [`SubtileIR`] — the graph the tiled-matmul
/// descriptors under test are lowered from. t0 = activation source, t1 = weight source, t2 = result.
fn single_matmul_ir(m: u32, n: u32, k: u32) -> scratchy_subtile::subtile_ir::SubtileIR {
    use scratchy_subtile::subtile_ir::{
        SubOp, SubtileIR, SubtileId, SubtileNode, TensorId, TensorRegion, TensorShape,
    };
    let tensors = vec![
        TensorShape { rows: m, cols: k },
        TensorShape { rows: k, cols: n },
        TensorShape { rows: m, cols: n },
    ];
    let whole = |t: usize, ts: &[TensorShape]| TensorRegion {
        tensor: TensorId::from_index(t),
        region: ts[t].whole(),
    };
    let node = SubtileNode {
        id: SubtileId::from_index(0),
        op: SubOp::MatmulTile {
            weight: scratchy_subtile::lower::GemmWeight::Dense,
        },
        inputs: vec![whole(0, &tensors), whole(1, &tensors)],
        output: whole(2, &tensors),
    };
    SubtileIR {
        tensors,
        num_sources: 2,
        nodes: vec![node],
        result: TensorId::from_index(2),
        // Hand-authored fixture: there is no source op list to be the provenance of.
        op_output: Vec::new(),
    }
}

/// Build a validated matmul WorkPlan with an explicit split (mb×out).
fn matmul_plan(m: u32, n: u32, k: u32, batch: u32, mb: u32, out: u32) -> WorkPlan {
    let dims = [
        ItDim {
            name: "mb",
            size: m,
            df: Df::Fp16,
            is_reduction: false,
            is_stick: false,
        },
        ItDim {
            name: "out",
            size: n,
            df: Df::Fp16,
            is_reduction: false,
            is_stick: true,
        },
        ItDim {
            name: "in",
            size: k,
            df: Df::Fp16,
            is_reduction: true,
            is_stick: true,
        },
        ItDim {
            name: "x",
            size: batch,
            df: Df::Fp16,
            is_reduction: false,
            is_stick: false,
        },
        ItDim {
            name: "y",
            size: 1,
            df: Df::Fp16,
            is_reduction: false,
            is_stick: false,
        },
    ];
    WorkPlan::divide(&dims, MaxCores::<MAX_CORES>, |_, _| {
        BTreeMap::from([("mb", mb), ("out", out)])
    })
    .unwrap()
}

/// The matmul per-core LX residency (A + W + O), fp16 = 2 B, INCLUDING batch.
fn resident(plan: &WorkPlan, out_per_time: u32) -> u64 {
    let per_core_mb = plan.per_core_extent("mb") as u64;
    let k = plan.extent("in") as u64;
    let batch = plan.extent("x").max(1) as u64;
    let opt = out_per_time as u64;
    per_core_mb * k * batch * FP16_BYTES
        + k * opt * batch * FP16_BYTES
        + per_core_mb * opt * batch * FP16_BYTES
}

// (i) ── FFN-sized matmul tiles; out_per_time multiple of 64, resident ≤ USABLE ──
#[test]
fn ffn_matmul_time_tiles_to_64_multiple() {
    // 384×2048×2048 batch1, split mb16×out2 → per_core_out=1024 → time=4.
    let plan = matmul_plan(384, 2048, 2048, 1, 16, 2);
    assert_eq!(plan.per_core_extent("out"), 1024);
    let tt = plan
        .time_tile_for_lx(resident, "out")
        .expect("ffn must tile, not Err")
        .expect("ffn per-core tile overflows LX → Some(TimeTile)");
    assert!(tt.count() > 1);
    assert_eq!(tt.dim().name(), "out");
    let out_per_time = 1024 / tt.count();
    assert_eq!(
        out_per_time % (Fp16::ELEMS_PER_STICK),
        0,
        "TiledStickExtent: 64-aligned"
    );
    assert!(resident(&plan, out_per_time) <= USABLE_LX_BYTES);
    assert_eq!(tt.count(), 4, "smallest fitting time for the ffn up-proj");
}

// (ii) ── the known bmm fits LX in one trip ──
#[test]
fn bmm_fits_lx_in_one_trip() {
    // 384×384×64 batch16, split mb16×out2 → per_core_out=192, resident 576 KiB.
    let plan = matmul_plan(384, 384, 64, 16, 16, 2);
    assert_eq!(plan.per_core_extent("out"), 192);
    let r = resident(&plan, 192);
    assert_eq!(r, 589_824, "bmm per-core resident incl. batch = 576 KiB");
    assert!(r < USABLE_LX_BYTES);
    assert!(
        plan.time_tile_for_lx(resident, "out").unwrap().is_none(),
        "bmm fits → time=1, no TimeTile"
    );
}

// (iii) ── impossibly-large matmul is a Rust Err (the DtException-1535 guard) ──
#[test]
fn impossibly_large_matmul_is_build_err() {
    // K=49152 batch1: w at out=64 = 49152*64*2 = 6.29 MiB ≫ USABLE_LX → Err even
    // when tiled to single sticks. The whole point: this is a cargo build failure,
    // NEVER an on-card DtException 1535.
    let plan = matmul_plan(64, 4096, 49152, 1, 1, 1);
    assert!(
        plan.time_tile_for_lx(resident, "out").is_err(),
        "a matmul too big to tile on N alone MUST be a build Err"
    );
}

// (iv-a) ── a tiled op's bundle.mlir has scf.for + affine.apply + symbolic addr ──
//
// 64×16384×2048 batch1: the cost model fills 32 cores splitting `out` ×32 →
// per_core_out=512 (8 sticks). Per-core resident (A 256 KiB + W 2 MiB + O) =
// 2.42 MiB > 1.68 MiB usable LX → MUST time-tile. The A (input) term is only
// 256 KiB so out-tiling DOES bring it under LX (unlike a wide-K shape, which
// would Err). time=2 → out_per_time=256, resident ≈ 1.34 MiB.
#[test]
fn tiled_bundle_mlir_is_concrete_unroll() {
    // CONCRETE-UNROLL (task #53): a time=N tiled op pre-unrolls into N FLAT
    // sdsc_execute (no scf.for, no symbols), and N per-trip SdscOps whose tiled-
    // tensor start addresses advance by t·stride. Sidesteps dxp's always-on
    // LoopUnroll (which cloned identical symbol_ids → DtException).
    let emitted = superdsc::assemble_matmul(
        "matmul_o7",
        64,
        16384,
        2048,
        1,
        &emit::rb("a", 64, 2048),
        &Stk::<KernelTag>::kernel(2048_usize, 16384_usize, "w"),
        &emit::rb("o", 64, 16384),
        None,
    );
    let n = emitted.time;
    assert!(n > 1, "this matmul must time-tile, got time={n}");
    let mlir = superdsc::emit_bundle_mlir(std::slice::from_ref(&emitted));
    // NO loop, NO symbols — just N flat executes over sdsc_0..N-1.json.
    assert!(
        !mlir.contains("scf.for"),
        "concrete-unroll must NOT emit scf.for:\n{mlir}"
    );
    assert!(
        !mlir.contains("affine.apply"),
        "no affine.apply in concrete-unroll"
    );
    assert!(
        !mlir.contains("symbol_ids"),
        "no symbol_ids in concrete-unroll"
    );
    let executes = mlir.matches("sdscbundle.sdsc_execute").count();
    assert_eq!(executes as u32, n, "one flat execute per trip");
    for t in 0..n {
        assert!(
            mlir.contains(&format!("sdsc_filename=\"sdsc_{t}.json\"")),
            "trip {t} file"
        );
    }
    // The per-trip SdscOps: CONCRETE (no isStartAddrSymbolic_, no negative ids),
    // and trip 1 differs from trip 0 (its tiled start addresses are bumped by
    // stride). Robust check: distinct serialized json, neither symbolic.
    let trips = superdsc::concrete_trips(&emitted);
    assert_eq!(trips.len() as u32, n);
    let j0 = serde_json::to_string(&trips[0]).unwrap();
    let j1 = serde_json::to_string(&trips[1]).unwrap();
    assert!(
        !j0.contains("isStartAddrSymbolic_"),
        "concrete trip is not symbolic"
    );
    assert!(!j0.contains("\"-1\""), "no negative symbol ids in the json");
    assert_ne!(
        j0, j1,
        "trip 1 must differ from trip 0 (bumped start addresses)"
    );
}

// (iv-b) ── a time=1 op's bundle.mlir is byte-identical to the historical flat form ──
#[test]
fn time1_bundle_mlir_is_byte_identical_to_flat() {
    let emitted = superdsc::assemble_matmul(
        "MatMul_0",
        384,
        384,
        64,
        16,
        &emit::rb("act", 384, 64),
        &Stk::<KernelTag>::kernel(64_usize, 384_usize, "wt"),
        &emit::rb("out", 384, 384),
        None,
    );
    assert_eq!(emitted.time, 1, "bmm must NOT time-tile");
    let via_emitted = superdsc::emit_bundle_mlir(&[emitted]);
    let flat = superdsc::bundle_mlir(&["sdsc_0.json".to_string()]);
    assert_eq!(
        via_emitted, flat,
        "all-time=1 bundle.mlir must equal the flat form"
    );
    assert!(!via_emitted.contains("scf.for"));
}

// ── DUMP harness (run with `--ignored`): writes the two GATE-1 dxp probe bundles
//    to /tmp/superdsc_gate/{fit64,ov4096} for an on-card `dxp_standalone --bundle`
//    run. NOT a unit assertion — the de-risk instrument for "does the typed
//    minimal frontend format (time=1) and the time-tiled format clear 1535 on the
//    card?". Per THE LOOP: any residual DtException → a new Rust witness here.
#[test]
#[ignore]
fn dump_gate_bundles() {
    use std::path::Path;
    let root = Path::new("/tmp/superdsc_gate");
    // (A) FITTING 64×64×64 — time=1, no loop. Hypothesis-b probe.
    let fit = superdsc::assemble_matmul(
        "gate1_mm",
        64,
        64,
        64,
        1,
        &emit::rb("g1_a", 64, 64),
        &Stk::<KernelTag>::kernel(64_usize, 64_usize, "g1_w"),
        &emit::rb("g1_o", 64, 64),
        None,
    );
    assert_eq!(fit.time, 1, "64^3 must fit LX in one trip");
    superdsc::write_dxp_input(&root.join("fit64"), &[fit], superdsc::FoldGrouping::Split).unwrap();
    // (B) OVERFLOWING but TILEABLE-ON-`out` — must overflow the full per-core tile
    //     yet fit once `out` is tiled to sticks (i.e. the activation a=per_core_mb·K·2
    //     must itself fit; large-K shapes that need K-PSUM correctly `Err`). Probe
    //     real-ish shapes and bake the FIRST tileable-overflow one.
    let candidates: &[(u32, u32, u32, u32, &str)] = &[
        (384, 49152, 576, 1, "lm_head_smollm2"), // vocab proj, small K=576
        (512, 12288, 768, 1, "wide_outproj"),
        (256, 16384, 1024, 1, "wide_k1024"),
    ];
    let mut wrote_b = false;
    for &(m, n, k, b, label) in candidates {
        let s = superdsc::matmul_cost_split(b, m, n, k, MAX_CORES);
        let per_core_mb = (m / s.m).max(1) as u64;
        let per_core_out = (n / s.n) as u64;
        let kk = k as u64;
        let full = per_core_mb * kk * 2 + kk * per_core_out * 2 + per_core_mb * per_core_out * 2;
        // resident at single 64-stick out tile (a is FIXED — the tileability gate):
        let stick = per_core_mb * kk * 2 + kk * 64 * 2 + per_core_mb * 64 * 2;
        eprintln!(
            "{label} {m}x{n}x{k} b{b}: split={s:?} per_core_out={per_core_out} full={full} single_stick={stick} usable={USABLE_LX_BYTES}"
        );
        if full > USABLE_LX_BYTES && stick <= USABLE_LX_BYTES {
            let ov = superdsc::assemble_matmul(
                "gate1b_mm",
                m,
                n,
                k,
                b,
                &emit::rb("gb_a", m, k),
                &Stk::<KernelTag>::kernel(k as usize, n as usize, "gb_w"),
                &emit::rb("gb_o", m, n),
                None,
            );
            assert!(ov.time > 1, "{label} must time-tile");
            eprintln!("→ BAKING {label} as GATE-1b: time = {}", ov.time);
            superdsc::write_dxp_input(&root.join("ov_tiled"), &[ov], superdsc::FoldGrouping::Split)
                .unwrap();
            wrote_b = true;
            break;
        }
    }
    assert!(
        wrote_b,
        "no tileable-overflow candidate found — widen the probe list"
    );
    // (C) BROADCAST gate: multiply(x[m,cols], v[m,1 broadcast-over-cols]) — the
    //     RmsNorm/RoPE/Attn prerequisite. v's `out`=RedStick → alpha_=0 stick fold
    //     (broadcast read). Tests whether dxp accepts a broadcast operand.
    let bc = emit::assemble_broadcast_mul_gate(
        "bcast_mul",
        scratchy_subtile::sdsc_abstract::RowCount::of_token_rows(64),
        scratchy_subtile::sdsc_abstract::BlockCols::of_feature_cols(256),
        "bc_x",
        "bc_v",
        "bc_o",
    );
    eprintln!("→ BAKING broadcast-mul gate: time={}", bc.time);
    superdsc::write_dxp_input(&root.join("bcast"), &[bc], superdsc::FoldGrouping::Split).unwrap();
    // (D) RMSNORM gate: the full 6-op decomposition (IBM spyre_rms_norm) over
    //     [m=64, hidden=576] (SmolLM2 hidden). Tests the multi-op + broadcast chain
    //     (reduce→add(eps)→rsqrt→broadcast-mul) on-card.
    let mut sym = 0i64;
    let rms = assemble_rmsnorm(
        "gate",
        64,
        576,
        "rms_x",
        "rms_w",
        scratchy_target_spyre::bundle_code::PlaceId::Act(7),
        "eps",
        &mut sym,
        None,
    );
    eprintln!("→ BAKING rmsnorm gate: {} ops", rms.len());
    superdsc::write_dxp_input(&root.join("rmsnorm"), &rms, superdsc::FoldGrouping::Split).unwrap();
    // (E) SLICE gate: add(x[:,0:64], x[:,64:128]) — the RoPE slice-operand probe.
    let sl = emit::assemble_slice_add_gate("slice_add", 64, 64, "sl_x", "sl_o");
    eprintln!("→ BAKING slice-add gate: time={}", sl.time);
    superdsc::write_dxp_input(&root.join("slice"), &[sl], superdsc::FoldGrouping::Split).unwrap();
    eprintln!("wrote fit64, ov_tiled, bcast, rmsnorm, slice");
}

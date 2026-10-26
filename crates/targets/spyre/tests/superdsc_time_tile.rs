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
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{
    RmsnormFacts, assemble_reduce, assemble_rmsnorm,
};
use scratchy_subtile::sdsc_abstract::{BlockCols, KernelTag, RowCount, Stk};
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
        // The fused exx2 reduce head (rmsnorm). Evidence: registered in the dd2 sysconfig's op
        // array (multi_sentient_dd2_sysconfig.json — the same list that carries sum/max/mean and
        // our card-proven quantscalepertokenfp8), plus the vendor's own compiled fixture
        // ddc/ddl_templates/test/sdsc_exx2.json. NOTE dscdefn.cpp's computeTypeToString is the
        // INTERNAL ComputeOpType namespace (fmul/reduce/exx2_32_p1/…) — "mean"/"sum"/"max" are not
        // in it either; the sysconfig array is the opFuncName registry.
        "exx2",
        // The compare/select family of broadcast_ops.ddl (:36-42) — the gemma-4
        // MoE port's rank-vector/one-hot vocabulary. Each name is the
        // `operation_bind`'s own `opFuncName` string in the vendor template,
        // which is the same evidence class as the on-card names above.
        "lesserthan",
        "equal",
        "notequal",
        "where3",
    ];
    // ⚠️ THIS LIST IS HAND-PICKED, WHICH IS WHY OPS COULD GO MISSING. `OpFunc` has 31
    // variants; this loop names 25. A new variant is NOT an E0004 here, so it joins the emitter
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
        OpFunc::Exx2,
        OpFunc::LesserThan,
        OpFunc::Equal,
        OpFunc::NotEqual,
        OpFunc::Where3,
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
        &emit::rb(
            "r_x",
            RowCount::of_token_rows(64),
            BlockCols::of_feature_cols(576),
        ),
        &emit::rb(
            "r_acc",
            RowCount::of_token_rows(64),
            BlockCols::of_feature_cols(1),
        ),
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
    // ⛔ #197's fold-manager const format: `data_` is the fold-manager object whose own `data_`
    // keys fold triples and whose word is a DECIMAL STRING, not a flat u64 array. Reading
    // `sf["data_"][0].as_u64()` was silently `None` the moment the wrapper landed.
    let val = sf["data_"]["data_"]["[0, 0, 0]"][0]
        .as_str()
        .and_then(|s| s.parse::<u64>().ok())
        .expect("scaling_factor's fold-manager const word (decimal string at [0, 0, 0])");
    assert!(
        val <= 0xFFFF,
        "scaling_factor must be a 16-bit fp16 word, got {val:#x}"
    );

    // multi-op (RMSNorm 5-op): EVERY op's coreIdToDsc_ matches its numCoresUsed_. The chain's HEAD
    // is now the device's own fused `exx2` reduce (the vendor layernorm head borrowed for RMS —
    // rmsnorm.rs's module doc): the former `rmsq`+`rmmean` pair is ONE op, 6 → 5.
    let mut sym = 0i64;
    let ops = assemble_rmsnorm(
        RmsnormFacts {
            prefix: "g",
            rows: 64,
            cols: 576,
            x: "x",
            gamma: "w",
            out_id: scratchy_target_spyre::bundle_code::PlaceId::Act(7),
            eps_const: "eps",
            ones_const: None,
        },
        &mut sym,
        None,
    );
    assert_eq!(ops.len(), 5, "the exx2 head fuses rmsq+rmmean: 6 ops → 5");
    assert!(
        ops.iter().all(|o| o.op_name != "rmsq_g"),
        "the x² pointwise is GONE — the square happens inside the reduce"
    );
    for op in &ops {
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
    // The head's WIRE FORM. The vendor's own compiled fixture
    // (ddc/ddl_templates/test/sdsc_exx2.json) carries useZeroMean=1 — but it is compile-tested
    // only (its RUN line is `ddl_standalone`, no execution), and 1 is NOT a runnable value:
    //   * opFuncName "exx2" is the ONLY runtime-supported spelling — it is what
    //     updatelibrary.cpp's libMapping resolves to exx2_fwd.smc. A useZeroMean of 1 makes ddc
    //     rewrite the name to `exx2_zeromean` (ddcv1.cpp:2072), a DISTINCT OpFuncs enum with NO
    //     libMapping entry — the runtime's `libMapping.at()` throws on the miss and the op
    //     executes as garbage (green bake, degenerate text; measured on card 2026-10-10). The
    //     vendor's production lowering never emits 1 (decompositions.py:1038 passes `False`);
    //   * constantInfo_ carries BOTH named consts: `exx2scale` (SEN169 bits of 1/cols — the fixture's
    //     own word for a 4096-wide input is 9728 = sen169(2⁻¹²), and summeanmaxexx2.ddl requires the
    //     external const by name or ddc DT_ERRORs "Missing external constant in DSC") and
    //     `useZeroMean` (the RAW integer 0 — NOT a sen169-encoded value, because ddcv1 compares
    //     == 1; the vendor's production spelling).
    let head = ops.first().expect("the exx2 head is the chain's first op");
    assert_eq!(head.op_name, "rmmean_g");
    let jh: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&head.op).unwrap()).unwrap();
    assert_eq!(
        jh["dscs_"][0]["rmmean_g"]["computeOp_"][0]["opFuncName"], "exx2",
        "the head must emit the plain spelling — the only name the runtime smc table resolves"
    );
    let ci = jh["dscs_"][0]["rmmean_g"]["constantInfo_"]
        .as_object()
        .expect("the exx2 head carries its two named consts (summeanmaxexx2.ddl)");
    let const_word = |name: &str| -> u64 {
        ci.values()
            .find(|c| c["name_"] == name)
            .unwrap_or_else(|| panic!("exx2 constantInfo_ must carry {name}"))
            ["data_"]["data_"]["[0, 0, 0]"][0]
            .as_str()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or_else(|| panic!("{name}'s fold-manager word (decimal string at [0, 0, 0])"))
    };
    assert!(
        const_word("exx2scale") <= 0xFFFF,
        "exx2scale must be a single 16-bit sen169 word"
    );
    assert_eq!(
        const_word("useZeroMean"),
        0,
        "useZeroMean must stay the RAW integer 0 — at 1 ddc renames the op to exx2_zeromean, \
         which the runtime's libMapping.at() cannot resolve (card-measured garbage)"
    );
    // The E[x²] LANE READ — the address half of the wire contract. The exx2 output's two
    // half-stick values sit at elements [0..8) (E[x]; computed and unread) and [8..16)
    // (E[x²]) — the vendor's own write AND cross-chunk read-back of the second value both carry
    // `stick_replicated_dim_offset_elements=8` (summeanmaxexx2.ddl:625/:192), and their
    // layernormscale reads it at the same offset (layernormscale_32.ddl:75). So the tail must
    // read rinv at ELEMENT 8: rmxn's rinv operand startAddr = rinv's own base + 8·2B. The first
    // card run of this chain read element 1 instead — the E[x] half, ≈0 for a signed activation
    // row, so the norm multiplied by rsqrt(E[x]+eps) and the model emitted degenerate text. This
    // delta is the build-time pin on that failure class.
    let tensor1_addr = |op_idx: usize| -> u64 {
        let jo: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&ops[op_idx].op).unwrap()).unwrap();
        let name = &ops[op_idx].op_name;
        jo["dscs_"][0][name.as_str()]["scheduleTree_"][1]["startAddressCoreCorelet_"]["data_"]
            ["[0, 0, 0]"]
            .as_str()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or_else(|| panic!("{name}'s Tensor1 startAddr (decimal string)"))
    };
    // rmrsqrt's Tensor1 is rinv at its own base; rmxn's Tensor1 is the SAME synth read at the
    // lane offset. ops: [rmmean, rmeps, rmrsqrt, rmxn, rmg].
    assert_eq!(
        tensor1_addr(3) - tensor1_addr(2),
        8 * FP16_BYTES,
        "rmxn must read rinv at element 8 — the exx2 output's E[x²] half-stick"
    );
}

// ── PINNED-LX rinv — the residency half of the lane-read contract. The `+16B` element-8 read
//    only survives the compile chain when rinv is LX-pinned (`BundleLayout::synth_lx_op_local`): on HBM
//    the L3 EBR floors a sub-stick offset to the stick base and the reader's 2bsplat reads
//    element 0 (rsqrt(E[x]+eps), a saturated constant — the green-bake garbage of the first
//    three card runs). An LX allocate is byte-granular end-to-end, so the offset lands
//    BYTE-exact. This pins the wire shape that mechanism emits.
#[test]
fn rmsnorm_rinv_is_pinned_lx() {
    use ktir_superdsc::placement::{BundleLayout, LX_PIN_END, SegRole, TensorPlacement};
    use scratchy_target_spyre::bundle_code::PlaceId;

    // A minimal layout holding the chain's real operands (x / gamma / eps / out), so
    // `resolve_seg_base` finds placements for them and the chain's own synth declarations
    // (Mean/Meps/Xn in seg3; rinv in the pinned-LX region) fill the rest.
    fn layout_for(rows: u32, cols: u32) -> BundleLayout {
        let mut l = BundleLayout::default();
        let place = |l: &mut BundleLayout, tid: u32, size: u64| {
            l.ids
                .borrow_mut()
                .insert(format!("t{tid}"), PlaceId::Act(tid));
            l.placements.insert(
                tid,
                TensorPlacement {
                    tid,
                    role: SegRole::Intermediate,
                    segment: 0,
                    bank: 0,
                    offset: 0,
                    size,
                    cover: ktir_superdsc::placement::BindCover::Full,
                },
            );
        };
        let xy = rows as u64 * cols as u64 * FP16_BYTES;
        place(&mut l, 1, xy); // x
        place(&mut l, 2, cols as u64 * FP16_BYTES); // gamma [1, cols]
        place(&mut l, 3, 64 * FP16_BYTES); // eps [1, stick]
        place(&mut l, 7, xy); // out
        l
    }
    let run = |rows: u32| -> Vec<emit::EmittedOp> {
        let l = layout_for(rows, 576);
        let mut sym = 0i64;
        assemble_rmsnorm(
            RmsnormFacts {
                prefix: "g",
                rows,
                cols: 576,
                x: "t1",
                gamma: "t2",
                out_id: PlaceId::Act(7),
                eps_const: "t3",
                // The Scale convention — no `γ+1` row, byte-identical to the emission before
                // the OnePlusScale row existed. The oneplus law has its own golden
                // (triton-numeric's `rmsnorm_oneplus`).
                ones_const: None,
            },
            &mut sym,
            Some(&l),
        )
    };
    let json_of = |op: &emit::EmittedOp| -> (String, serde_json::Value) {
        (
            op.op_name.clone(),
            serde_json::from_str(&serde_json::to_string(&op.op).unwrap()).unwrap(),
        )
    };
    let alloc = |j: &serde_json::Value, dsc: &str, idx: usize| {
        j["dscs_"][0][dsc]["scheduleTree_"][idx].clone()
    };

    // ── m>1 (rows=64): the producer REPLICATES and the reader's slice corner folds ──
    let ops = run(64);
    assert_eq!(ops.len(), 5, "same 5-op chain under a layout");
    let (w_name, wj) = json_of(&ops[2]); // rmrsqrt — rinv's producer
    assert_eq!(w_name, "rmrsqrt_g");
    assert_eq!(
        wj["numCoresUsed_"].as_u64(),
        Some(MAX_CORES as u64),
        "an LX-resident output's producer REPLICATES across all cores"
    );
    assert_eq!(
        wj["coreIdToDsc_"].as_object().unwrap().len(),
        MAX_CORES as usize
    );
    assert!(
        wj["numWkSlicesPerDim_"]
            .as_object()
            .unwrap()
            .values()
            .all(|v| v.as_u64() == Some(1)),
        "a replicated plan carries NO work-division splits"
    );
    // The allocate: an `lx` component at the PIN, same byte on every core. The pin is a
    // TOP-DOWN op-local slot (#645): the rinv's footprint class reserves
    // [LX_PIN_END − fp, LX_PIN_END), so every norm of the class shares this one address.
    let node = alloc(&wj, "rmrsqrt_g", 1);
    assert_eq!(node["name_"], "allocate-Tensor1_lx");
    assert_eq!(node["component_"], "lx");
    let rinv_fp = 64u64 * 64 * FP16_BYTES; // [rows=64, stick=64] f16 — the class footprint
    for (k, v) in node["startAddressCoreCorelet_"]["data_"]
        .as_object()
        .unwrap()
    {
        assert_eq!(
            v.as_str(),
            Some((LX_PIN_END - rinv_fp).to_string().as_str()),
            "rmrsqrt's rinv pin must be the top-down op-local slot's start, below {LX_PIN_END}, \
             on every core ({k})"
        );
    }
    // The lds: lx-ONLY residency — no HBM copy exists, so no L3 movement is generated.
    let rinv_lds = wj["dscs_"][0]["rmrsqrt_g"]["labeledDs_"][1]["memOrg_"].clone();
    assert!(rinv_lds.get("lx").is_some(), "rinv memOrg_ carries lx");
    assert!(
        rinv_lds.get("hbm").is_none(),
        "rinv memOrg_ must NOT carry hbm — an HBM residency would stage it through the \
         stick-granular L3 path that floors the lane read away"
    );
    // The reader (rmxn, mb-split 32-way at rows=64): rinv at pin + 16, and its per-core
    // slice corner advances by one row-group — the same fold HBM addressing uses, on the
    // per-core private copy the replicated producer wrote.
    let (r_name, rj) = json_of(&ops[3]);
    assert_eq!(r_name, "rmxn_g");
    let rnode = alloc(&rj, "rmxn_g", 1);
    assert_eq!(rnode["component_"], "lx");
    let pin_str = |s: u64| s.to_string();
    let pin = LX_PIN_END - rinv_fp; // the class slot's start (top-down, below LX_PIN_END)
    assert_eq!(
        rnode["startAddressCoreCorelet_"]["data_"]["[0, 0, 0]"].as_str(),
        Some(pin_str(pin + 8 * FP16_BYTES).as_str()),
        "rmxn reads rinv at pin + element-8 byte offset (the E[x²] lane)"
    );
    assert_eq!(
        rnode["startAddressCoreCorelet_"]["data_"]["[1, 0, 0]"].as_str(),
        Some(pin_str(pin + 8 * FP16_BYTES + 2 * 64 * FP16_BYTES).as_str()),
        "core 1's rinv corner = its 2-row work slice (2 rows × one stick) past the pin"
    );

    // ── m=1 (decode, the lever's bucket): the reader is out-split, its rinv corner is 0 —
    //    every reader core reads pin+16 out of its OWN private copy ──
    let ops1 = run(1);
    let (_, rj1) = json_of(&ops1[3]);
    let rnode1 = alloc(&rj1, "rmxn_g", 1);
    // The m=1 run lowers under its OWN layout: the [1,64] class slots at its own top-down
    // address, one stick below the boundary.
    let pin1 = LX_PIN_END - 64 * FP16_BYTES;
    for (k, v) in rnode1["startAddressCoreCorelet_"]["data_"]
        .as_object()
        .unwrap()
    {
        assert_eq!(
            v.as_str(),
            Some(pin_str(pin1 + 8 * FP16_BYTES).as_str()),
            "at m=1 every reader core reads the SAME pinned lane ({k})"
        );
    }
    let (_, wj1) = json_of(&ops1[2]);
    assert_eq!(
        wj1["numCoresUsed_"].as_u64(),
        Some(MAX_CORES as u64),
        "the m=1 producer replicates too (a 1-core producer would leave cores 1..31 \
         reading an empty scratchpad)"
    );
}

// ── #645 fix (a), STEP 3: THE OP-LOCAL SLOT ALTERNATION WALK ───────────────────
//    Two norms of ONE footprint class share ONE top-down slot (pinned above: every
//    same-class rinv resolves to the same address below LX_PIN_END), so a name's bytes
//    survive exactly until the next same-class name's program runs. The walk proves the
//    touching names form CONTIGUOUS RUNS over the bundle's emitted order — per launch
//    group over its programs, with the run state carried ACROSS groups (a bundle's
//    launches are ordered on one stream). Every test here drives it the way
//    `ktir_groups_via_superdsc` does: real `assemble_rmsnorm` chains against one
//    layout, `per_prog` one (program, descriptor-count) entry per chain, the SAME
//    partition (`trip_kinds_for` + `group_ranges`) the launch index uses.

/// The walk tests' fixture: one layout holding both norms' operands — x/gamma/eps shared
/// (placed tensors; the pins are keyed on the OUT), outs `t7` and `t17` — so two real
/// chains mint two DIFFERENT rinv pins of ONE footprint class ([64, 64] f16 = 8192 B)
/// into ONE shared slot.
fn two_norm_layout() -> ktir_superdsc::placement::BundleLayout {
    use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
    use scratchy_target_spyre::bundle_code::PlaceId;

    let mut l = BundleLayout::default();
    let place = |l: &mut BundleLayout, tid: u32, size: u64| {
        l.ids
            .borrow_mut()
            .insert(format!("t{tid}"), PlaceId::Act(tid));
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Intermediate,
                segment: 0,
                bank: 0,
                offset: 0,
                size,
                cover: ktir_superdsc::placement::BindCover::Full,
            },
        );
    };
    let xy = 64u64 * 576 * FP16_BYTES;
    place(&mut l, 1, xy); // x
    place(&mut l, 2, 576 * FP16_BYTES); // gamma [1, cols]
    place(&mut l, 3, 64 * FP16_BYTES); // eps [1, stick]
    place(&mut l, 7, xy); // norm A's out
    place(&mut l, 17, xy); // norm B's out
    l
}

/// One real rmsnorm chain against the fixture's layout: `out` selects the pin (A on `t7`,
/// B on `t17`), and the SAME out again (with a fresh prefix) is the RETURNING name of the
/// interleave tests — `synth_lx_op_local` is idempotent per name, so A′'s chain touches
/// A's pin.
fn norm(
    l: &ktir_superdsc::placement::BundleLayout,
    prefix: &str,
    out: u32,
    sym: &mut i64,
) -> Vec<emit::EmittedOp> {
    assemble_rmsnorm(
        RmsnormFacts {
            prefix,
            rows: 64,
            cols: 576,
            x: "t1",
            gamma: "t2",
            out_id: scratchy_target_spyre::bundle_code::PlaceId::Act(out),
            eps_const: "t3",
            ones_const: None,
        },
        sym,
        Some(l),
    )
}

#[test]
fn two_norms_of_one_class_alternate_the_one_shared_slot() {
    use ktir_superdsc::placement::LX_PIN_END;

    let l = two_norm_layout();
    let mut sym = 0i64;
    let a = norm(&l, "a", 7, &mut sym);
    let b = norm(&l, "b", 17, &mut sym);
    assert_eq!((a.len(), b.len()), (5, 5), "two five-op chains");
    // The fixture's own promise: two pin names, ONE footprint class, ONE slot address.
    let pins = l.lx_op_local_pins();
    assert_eq!(pins.len(), 2);
    let pin_of = |n: &str| pins.iter().find(|p| p.0 == n).unwrap();
    assert_eq!(pin_of("t7_rinv").1, 8192, "the [64, 64] f16 class");
    assert_eq!(
        pin_of("t17_rinv").1,
        pin_of("t7_rinv").1,
        "one footprint class"
    );
    assert_eq!(
        pin_of("t17_rinv").2,
        pin_of("t7_rinv").2,
        "one shared top-down slot"
    );
    assert_eq!(pin_of("t7_rinv").2, LX_PIN_END - 8192);
    // The walk: two programs in order, one name per class each — the accepting shape.
    let mut ops = a.clone();
    ops.extend(b.iter().cloned());
    let per_prog = vec![("rms_a".to_string(), 5), ("rms_b".to_string(), 5)];
    let ledger = superdsc::walk_lx_slot_alternation(
        &ops,
        &per_prog,
        superdsc::FoldGrouping::Fused,
        Some(&l),
    )
    .unwrap();
    assert_eq!(ledger.len(), 2, "one ledger row per pin");
    let row_of = |n: &str| ledger.iter().find(|r| r.name == n).unwrap();
    for (pin, prog) in [("t7_rinv", "rms_a"), ("t17_rinv", "rms_b")] {
        let row = row_of(pin);
        assert_eq!(row.class_bytes, 8192);
        assert_eq!(row.slot, pin_of(pin).2);
        assert_eq!(
            row.touched_by,
            vec![(prog.to_string(), false), (prog.to_string(), true)],
            "{pin}: its own program writes it (rmrsqrt) then reads it (rmxn) — no other \
             program touches it"
        );
    }
    // The walk keys strictly off the layout's pin table: the same ops against no layout (a
    // layout-less emission mints no pins) and against a layout that declared none both walk
    // clean and empty.
    assert!(
        superdsc::walk_lx_slot_alternation(&ops, &per_prog, superdsc::FoldGrouping::Fused, None)
            .unwrap()
            .is_empty()
    );
    assert!(
        superdsc::walk_lx_slot_alternation(
            &ops,
            &per_prog,
            superdsc::FoldGrouping::Fused,
            Some(&two_norm_layout())
        )
        .unwrap()
        .is_empty()
    );
}

#[test]
fn the_walk_refuses_one_program_touching_two_same_class_pins() {
    let l = two_norm_layout();
    let mut sym = 0i64;
    let mut ops = norm(&l, "a", 7, &mut sym);
    ops.extend(norm(&l, "b", 17, &mut sym));
    // ONE program owning both chains' ops — the shape a collapsed lowering would produce.
    // One program's descriptors are one SDSC phase (the citations schedule per node, one
    // phase each), so both pins live in the SAME phase with no phase boundary between
    // their touches for a shared slot to alternate across: refused outright, both named.
    let per_prog = vec![("rms_pair".to_string(), 10)];
    let err = superdsc::walk_lx_slot_alternation(
        &ops,
        &per_prog,
        superdsc::FoldGrouping::Fused,
        Some(&l),
    )
    .unwrap_err();
    assert!(err.contains("same-class op-local LX pins"), "{err}");
    assert!(err.contains("'rms_pair'"), "{err}");
    assert!(
        err.contains("t7_rinv") && err.contains("t17_rinv"),
        "names both pins of the class: {err}"
    );
}

#[test]
fn the_walk_refuses_a_name_returning_after_its_sibling_took_the_slot() {
    let l = two_norm_layout();
    let mut sym = 0i64;
    let mut ops = norm(&l, "a", 7, &mut sym);
    ops.extend(norm(&l, "b", 17, &mut sym));
    ops.extend(norm(&l, "a2", 7, &mut sym)); // A′: A's pin again, a THIRD program
    let per_prog = vec![
        ("rms_a".to_string(), 5),
        ("rms_b".to_string(), 5),
        ("rms_a2".to_string(), 5),
    ];
    let err = superdsc::walk_lx_slot_alternation(
        &ops,
        &per_prog,
        superdsc::FoldGrouping::Fused,
        Some(&l),
    )
    .unwrap_err();
    assert!(err.contains("touched again"), "{err}");
    assert!(err.contains("'rms_a2'"), "{err}");
    assert!(err.contains("t7_rinv"), "names the RETURNING pin: {err}");
    assert!(
        err.contains("t17_rinv"),
        "names the sibling that took the slot: {err}"
    );
}

#[test]
fn the_walk_carries_run_state_across_launch_groups() {
    let l = two_norm_layout();
    let mut sym = 0i64;
    let a = norm(&l, "a", 7, &mut sym);
    let b = norm(&l, "b", 17, &mut sym);
    let a2 = norm(&l, "a2", 7, &mut sym);
    // A trip of ANOTHER kind between the runs — a distinct-slot KV cachewr
    // (`slot_stride_bytes` = 128, the decode write's head_dim·2) — cuts the partition:
    // [A's 5 Pure] [the Slot singleton] [B's and A′'s 10 Pure] are THREE launch groups.
    // The clone binds no op-local pin (A's reduce reads x and writes its mean synth), so it
    // contributes no touch — only its trip KIND matters.
    let mut slot = a[0].clone();
    slot.op_name = "cachewr_slot".to_string();
    slot.slot_stride_bytes = 128;
    let mut ops = a.clone();
    ops.push(slot);
    ops.extend(b.iter().cloned());
    let per_prog = vec![
        ("rms_a".to_string(), 5),
        ("cachewr".to_string(), 1),
        ("rms_b".to_string(), 5),
    ];
    // Without the returning name, the group boundary alone is NOT a refusal: A's run simply
    // stays open across the Slot's group and B closes it in the third.
    let ledger = superdsc::walk_lx_slot_alternation(
        &ops,
        &per_prog,
        superdsc::FoldGrouping::Fused,
        Some(&l),
    )
    .unwrap();
    assert_eq!(ledger.len(), 2, "A | Slot | B walks clean");
    // With A′ following B in the LAST group, the refusal fires ONLY because A's run — opened
    // in the FIRST group — was closed and remembered ACROSS the Slot boundary: were the run
    // state reset per group, the third group would start empty and A′'s touch would merely
    // close B's run, with no refusal.
    ops.extend(a2);
    let per_prog = vec![
        ("rms_a".to_string(), 5),
        ("cachewr".to_string(), 1),
        ("rms_b".to_string(), 5),
        ("rms_a2".to_string(), 5),
    ];
    let err = superdsc::walk_lx_slot_alternation(
        &ops,
        &per_prog,
        superdsc::FoldGrouping::Fused,
        Some(&l),
    )
    .unwrap_err();
    assert!(err.contains("touched again"), "{err}");
    assert!(err.contains("'rms_a2'"), "{err}");
    assert!(err.contains("t7_rinv"), "names the returning pin: {err}");
}

// ── THE DIRECTION GUARD: every op-local pin this bundle TOUCHES is touched in BOTH directions —
//    ≥1 write and ≥1 read, both through NAMED arg bindings (the only consult the walk has).
//    A read with no write is a producer the emission lost, or one addressing the pin through a
//    RAW OFFSET no walk over arg_bindings can see; a write with no read is dead scratch minted
//    against the window budget. Both are build failures here, never card-time stale bytes.
#[test]
fn the_walk_refuses_a_pin_no_program_writes() {
    let l = two_norm_layout();
    let mut sym = 0i64;
    // A's chain with its WRITER dropped — `rmrsqrt`, the op whose OUTPUT binding is the pin's
    // one write. The pin is still declared (the chain's own mint), and `rmxn` still READS it
    // by name, so the walk sees a read-only touch.
    let a: Vec<emit::EmittedOp> = norm(&l, "a", 7, &mut sym)
        .into_iter()
        .filter(|e| e.op_name != "rmrsqrt_a")
        .collect();
    assert_eq!(a.len(), 4, "the writer dropped, the reader kept");
    let b = norm(&l, "b", 17, &mut sym);
    let mut ops = a.clone();
    ops.extend(b.iter().cloned());
    let per_prog = vec![
        ("rms_a".to_string(), a.len()),
        ("rms_b".to_string(), b.len()),
    ];
    let err = superdsc::walk_lx_slot_alternation(
        &ops,
        &per_prog,
        superdsc::FoldGrouping::Fused,
        Some(&l),
    )
    .unwrap_err();
    assert!(err.contains("no program WRITES it"), "{err}");
    assert!(err.contains("t7_rinv"), "names the orphaned pin: {err}");
    assert!(
        !err.contains("t17_rinv"),
        "B's intact chain is not the refusal's subject: {err}"
    );
}

#[test]
fn the_walk_refuses_a_pin_no_program_reads() {
    let l = two_norm_layout();
    let mut sym = 0i64;
    // A's chain with its READER dropped — `rmxn`, the op whose INPUT binding (the `col_at`
    // view of the pin) is the pin's one read. The writer still declares it, so the walk sees
    // a write-only touch: dead scratch against the LX window.
    let a: Vec<emit::EmittedOp> = norm(&l, "a", 7, &mut sym)
        .into_iter()
        .filter(|e| e.op_name != "rmxn_a")
        .collect();
    assert_eq!(a.len(), 4, "the reader dropped, the writer kept");
    let b = norm(&l, "b", 17, &mut sym);
    let mut ops = a.clone();
    ops.extend(b.iter().cloned());
    let per_prog = vec![
        ("rms_a".to_string(), a.len()),
        ("rms_b".to_string(), b.len()),
    ];
    let err = superdsc::walk_lx_slot_alternation(
        &ops,
        &per_prog,
        superdsc::FoldGrouping::Fused,
        Some(&l),
    )
    .unwrap_err();
    assert!(err.contains("no program READS it"), "{err}");
    assert!(err.contains("t7_rinv"), "names the dead pin: {err}");
    assert!(
        !err.contains("t17_rinv"),
        "B's intact chain is not the refusal's subject: {err}"
    );
}

// ── THE CARVE-OUT the guard's scoping owes its safety to: the layout is RUNG-shared (the
//    declare pass pre-declares prefix, bodies and suffix alike), so ANOTHER bundle's pins sit
//    untouched in this bundle's table. They carry empty ledger rows and are not this walk's
//    to judge — a guard over every pin in the table would refuse every bundle that shares a
//    rung with a norm it never runs.
#[test]
fn the_walk_leaves_a_pin_this_bundle_never_touches_alone() {
    use ktir_superdsc::place::SynthRole;
    use scratchy_target_spyre::bundle_code::PlaceId;

    let l = two_norm_layout();
    // A third pin of the SAME class, minted straight into the shared layout the way the
    // declare pass mints another bundle's norm — declared there, never touched here.
    l.synth_lx_op_local(PlaceId::Act(27).synth(SynthRole::Rinv), &[64, 64]);
    let mut sym = 0i64;
    let mut ops = norm(&l, "a", 7, &mut sym);
    ops.extend(norm(&l, "b", 17, &mut sym));
    let per_prog = vec![("rms_a".to_string(), 5), ("rms_b".to_string(), 5)];
    let ledger = superdsc::walk_lx_slot_alternation(
        &ops,
        &per_prog,
        superdsc::FoldGrouping::Fused,
        Some(&l),
    )
    .unwrap();
    assert_eq!(
        ledger.len(),
        3,
        "one ledger row per pin in the table, touched or not"
    );
    let orphan = ledger
        .iter()
        .find(|r| r.name == PlaceId::Act(27).synth(SynthRole::Rinv).to_string())
        .unwrap();
    assert_eq!(
        orphan.touched_by,
        Vec::<(String, bool)>::new(),
        "the untouched pin carries an empty row — not a refusal"
    );
    assert_eq!(
        orphan.class_bytes, 8192,
        "the same class as the touched pins, sharing their slot"
    );
}

// ── THE REAL BUNDLE SHAPE the walk guards: an attention chunk's program followed by a
//    trailing norm's — the #645 fix (a) emission order (declare pass first, chunk second,
//    norm's real pass third). The chunk's pins are CORE-LOCAL (per-core scratch), never
//    op-local, so the op-local ledger carries exactly the trailing norm's rinv; the walk's
//    acceptance here is the green test that the guard does not refuse the very bundle the
//    fix exists to enable. The ADMISSION side of the same bundle — the chunk's `lx_form`
//    budgeting against the declared trailing norm — is pinned in ktir-superdsc's
//    `zz_the_hd128_chunk_in_lx.rs` (`a_trailing_norms_slot_still_bounds_the_chunk_that_precedes_it`).
#[test]
fn the_walk_accepts_a_chunk_program_followed_by_its_trailing_norm() {
    use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{AttnFacts, assemble_attn};
    use ktir_superdsc::place::{PlaceId, SynthRole};
    use ktir_superdsc::placement::{SegRole, TensorPlacement};
    use ktir_superdsc::sdsc_abstract::{
        AttnGeometry, ChunkScratch, KvExtent, PagedKvPool, SlotCount, attn_bundle_rows,
    };

    const NQH: u32 = 32;
    const NKVH: u32 = 8;
    const HD: u32 = 128;
    // The hd128 one-request decode layout (zz_the_hd128_chunk_in_lx's shape): eight placed
    // activation names for the attention operands, plus the trailing norm's out and eps.
    let mut layout = ktir_superdsc::placement::BundleLayout {
        attn_class_hds: vec![HD],
        ..ktir_superdsc::placement::BundleLayout::default()
    };
    let mut names = Vec::new();
    let ident = ktir_superdsc::reserved_tids::identity_class_tid(0);
    for (i, tid) in (1..=8u32).chain([900, ident]).enumerate() {
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
    let place = |l: &mut ktir_superdsc::placement::BundleLayout, tid: u32, size: u64| {
        l.ids
            .borrow_mut()
            .insert(PlaceId::Act(tid).to_string(), PlaceId::Act(tid));
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: 3,
                bank: 0,
                offset: 40 << 30,
                size,
                cover: ktir_superdsc::placement::BindCover::Full,
            },
        );
    };
    place(&mut layout, 901, 64 * 576 * FP16_BYTES); // the trailing norm's out
    place(&mut layout, 902, 64 * FP16_BYTES); // its eps const
    // ⭐ THE DECLARE PASS'S ACTION, one program EARLY: the trailing norm's whole public
    //    lowering runs against the shared layout first — throwaway symbols, ops dropped —
    //    so its rinv pin is in the table before the chunk's admission consults the slot sum.
    let mut declare_syms = 0i64;
    let _ = assemble_rmsnorm(
        RmsnormFacts {
            prefix: "tail",
            rows: 64,
            cols: 576,
            x: &names[0],
            gamma: &names[1],
            out_id: PlaceId::Act(901),
            eps_const: "t902",
            ones_const: None,
        },
        &mut declare_syms,
        Some(&layout),
    );
    // The chunk: 40 pages, the widest rung the hd128 one-request form admits.
    let geom = AttnGeometry::<NQH, NKVH, HD>::minted();
    let rows = attn_bundle_rows(geom, 1, true).expect("the one-request decode rung");
    let chunk = ChunkScratch::of_chunk(PagedKvPool::new(NKVH, HD), 40).expect("a chunk");
    let mut sym = 0i64;
    let attn_ops = assemble_attn(
        AttnFacts {
            t: 0,
            geom,
            bundle_rows: rows,
            // The storage/sweep pair on the boundary's one carrier — the sweep is the full 256-slot
            // cap (no rung).
            extent: KvExtent::new(SlotCount::new(256), SlotCount::new(256)),
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
    .unwrap_or_else(|e| panic!("assemble_attn refused: {}", e.0));
    assert!(
        attn_ops.iter().any(|e| e.op_name.starts_with("attn_cgkt_")),
        "the chunk's Kᵗ gather copies are present"
    );
    // The trailing norm's REAL pass: the same public lowering, its pin already declared.
    let tail = assemble_rmsnorm(
        RmsnormFacts {
            prefix: "tail",
            rows: 64,
            cols: 576,
            x: &names[0],
            gamma: &names[1],
            out_id: PlaceId::Act(901),
            eps_const: "t902",
            ones_const: None,
        },
        &mut sym,
        Some(&layout),
    );
    // THE BUNDLE: the chunk's program, then the norm's — the norm FOLLOWS the chunk in
    // program order, exactly as the rung lowers.
    let mut ops = attn_ops.clone();
    ops.extend(tail.iter().cloned());
    let per_prog = vec![
        ("attn_chunk".to_string(), attn_ops.len()),
        ("rmsnorm_tail".to_string(), tail.len()),
    ];
    let ledger = superdsc::walk_lx_slot_alternation(
        &ops,
        &per_prog,
        superdsc::FoldGrouping::Fused,
        Some(&layout),
    )
    .unwrap();
    assert_eq!(
        ledger.len(),
        1,
        "the chunk's pins are core-local, not op-local"
    );
    assert_eq!(
        ledger[0].name,
        PlaceId::Act(901).synth(SynthRole::Rinv).to_string(),
        "the one op-local pin is the trailing norm's rinv"
    );
    assert_eq!(
        ledger[0].touched_by,
        vec![
            ("rmsnorm_tail".to_string(), false),
            ("rmsnorm_tail".to_string(), true)
        ],
        "touched only by the norm's own program — the chunk's program never appears"
    );
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
        &emit::rb(
            "a",
            RowCount::of_token_rows(384),
            BlockCols::of_feature_cols(64),
        ),
        &Stk::<KernelTag>::kernel(64_usize, 384_usize, "w"),
        &emit::rb(
            "o",
            RowCount::of_token_rows(384),
            BlockCols::of_feature_cols(384),
        ),
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

/// ⭐ THROUGH THE REAL CHAIN, A DENSE MATMUL NODE IS ONE UNTILED DESCRIPTOR — dxp stages it.
///
/// `rows_are_requests` is TRUE: 64 rows that are 64 separate requests may NOT fold to the last row
/// (every row's logits are sampled), so this vocab-wide tail is the widest unbatched matmul a decode
/// batch emits. Every dense matmul is handed to dxp whole, and the bundle still builds.
#[test]
fn an_unbatched_fp16_matmul_is_one_descriptor() {
    // ⛔ THROUGH THE REAL CHAIN, BECAUSE A HAND-BUILT DESCRIPTOR IS NO LONGER A BUNDLE: after the
    // `SubtileIR → KTIR → SuperDSC` split a bundle is built from PROGRAMS (`ktir_groups` on
    // `-Fspyre-emu`, `ktir_groups_via_superdsc` on `-Fspyre-hw`), and an op carrying a descriptor and
    // no program is refused by both. So the fixture is the SubtileIR node, `time` is read off the
    // LOWERED descriptor, and the bundle off the PROGRAMS.
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
    assert_eq!(e.time, 1, "an unbatched fp16 matmul is staged by dxp");
    assert_eq!(superdsc::concrete_trips(e).len(), 1);
    // SAFETY: single-threaded test; the multi-op stitch opt-in only.
    unsafe { std::env::set_var("SCRATCHY_SUPERDSC_ALLOW_MULTIOP", "1") };
    let r = superdsc::emit_bundle(
        &programs,
        Some(&layout),
        superdsc::FoldGrouping::Split,
        None,
    );
    unsafe { std::env::remove_var("SCRATCHY_SUPERDSC_ALLOW_MULTIOP") };
    assert!(r.is_ok(), "the untiled matmul's bundle builds: {r:?}");
}

/// The LX-overflowing `y`-batched matmul the trip tests drive: the card-validated batched form (a
/// ONE-stick contraction, `in = 64`, at 16 kernel batches: the (ii) bmm) widened in N until its per-core
/// kernel slab overflows LX — at most 32 cores leave every core ≥ 1024 `out` columns, 2 MiB of kernel.
/// A `y`-batched matmul keeps the frontend's LX trips, and a one-stick
/// contraction is the only `y`-batched form the build admits.
fn batched_lx_overflowing_matmul() -> emit::EmittedOp {
    let (m, n, k, batch) = (64, 32768, 64, 16);
    superdsc::assemble_matmul(
        "matmul_o7",
        m,
        n,
        k,
        batch,
        &emit::rb(
            "a",
            RowCount::of_token_rows(m),
            BlockCols::of_feature_cols(k),
        ),
        &Stk::<KernelTag>::kernel(k as usize, n as usize, "w"),
        &emit::rb(
            "o",
            RowCount::of_token_rows(m),
            BlockCols::of_feature_cols(n),
        ),
        None,
    )
}

/// The `y`-batched shape the trip tests drove before the one-stick refusal, 64×16384×2048 at two kernel
/// batches: a 32-stick contraction under a `y` batch, the form measured incoherent on granite-3.1-8b.
/// It must stay a build error, and the error must name why.
fn assert_the_multi_stick_batched_matmul_is_refused() {
    let refused = ktir_superdsc::ir::bridge::tiled_op_sdsc_op::try_assemble_matmul_seeded(
        "matmul_o7",
        64,
        16384,
        2048,
        2,
        &emit::rb(
            "a",
            RowCount::of_token_rows(64),
            BlockCols::of_feature_cols(2048),
        ),
        &Stk::<KernelTag>::kernel(2048_usize, 16384_usize, "w"),
        &emit::rb(
            "o",
            RowCount::of_token_rows(64),
            BlockCols::of_feature_cols(16384),
        ),
        &mut 0,
        None,
    )
    .err()
    .expect("a y-batched multi-stick contraction must be refused");
    assert!(
        refused.contains(
            "a y-batched (batch 2) shared-kernel matmul contracts 2048 elements, 32 sticks"
        ) && refused.contains("card-validated only at a one-stick contraction"),
        "the refusal names the shape and the reason: {refused}"
    );
}

#[test]
fn tiled_matmul_trips_do_not_alias_and_advance() {
    // The LX-overflowing batched matmul is time>1 (proven tiled in the tiled_bundle_mlir test). Its
    // trips must be addressable — advancing instead of aliasing.
    assert_the_multi_stick_batched_matmul_is_refused();
    let e = batched_lx_overflowing_matmul();
    let e = &e;
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
            n,
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

// (iv-a) ── a tiled op's bundle.mlir is a concrete unroll ──
//
// 64×32768×64 at 16 kernel batches (`batched_lx_overflowing_matmul`): a `y`-batched
// matmul keeps the frontend's LX trips, and its per-core resident set (the batch
// factor included) overflows the 1.68 MiB usable LX → MUST time-tile. Its
// contraction is one stick, the only `y`-batched form the build admits. An
// unbatched fp16 matmul is handed to dxp whole
// (`an_unbatched_fp16_matmul_is_one_descriptor`).
#[test]
fn tiled_bundle_mlir_is_concrete_unroll() {
    // CONCRETE-UNROLL (task #53): a time=N tiled op pre-unrolls into N FLAT
    // sdsc_execute (no scf.for, no symbols), and N per-trip SdscOps whose tiled-
    // tensor start addresses advance by t·stride. Sidesteps dxp's always-on
    // LoopUnroll (which cloned identical symbol_ids → DtException).
    assert_the_multi_stick_batched_matmul_is_refused();
    let emitted = batched_lx_overflowing_matmul();
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
        &emit::rb(
            "act",
            RowCount::of_token_rows(384),
            BlockCols::of_feature_cols(64),
        ),
        &Stk::<KernelTag>::kernel(64_usize, 384_usize, "wt"),
        &emit::rb(
            "out",
            RowCount::of_token_rows(384),
            BlockCols::of_feature_cols(384),
        ),
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
        &emit::rb(
            "g1_a",
            RowCount::of_token_rows(64),
            BlockCols::of_feature_cols(64),
        ),
        &Stk::<KernelTag>::kernel(64_usize, 64_usize, "g1_w"),
        &emit::rb(
            "g1_o",
            RowCount::of_token_rows(64),
            BlockCols::of_feature_cols(64),
        ),
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
                &emit::rb(
                    "gb_a",
                    RowCount::of_token_rows(m),
                    BlockCols::of_feature_cols(k),
                ),
                &Stk::<KernelTag>::kernel(k as usize, n as usize, "gb_w"),
                &emit::rb(
                    "gb_o",
                    RowCount::of_token_rows(m),
                    BlockCols::of_feature_cols(n),
                ),
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
    // (D) RMSNORM gate: the full 5-op decomposition (the vendor's spyre_rms_norm formula with the
    //     device's own fused `exx2` head) over [m=64, hidden=576] (SmolLM2 hidden). Tests the
    //     multi-op + broadcast chain (exx2-reduce→add(eps)→rsqrt→broadcast-mul) on-card.
    let mut sym = 0i64;
    let rms = assemble_rmsnorm(
        RmsnormFacts {
            prefix: "gate",
            rows: 64,
            cols: 576,
            x: "rms_x",
            gamma: "rms_w",
            out_id: scratchy_target_spyre::bundle_code::PlaceId::Act(7),
            eps_const: "eps",
            ones_const: None,
        },
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

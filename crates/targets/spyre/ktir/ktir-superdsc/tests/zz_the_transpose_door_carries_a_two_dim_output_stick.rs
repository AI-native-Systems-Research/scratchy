// SPDX-License-Identifier: Apache-2.0
//! THE ONE OP THE DECODER CONFIGURATIONS DIE ON, AND THE ONE PROPERTY THAT KILLS IT.
//!
//! # WHAT WAS MEASURED (dxp_standalone, deeptools `d3a9bf4`-era tree, 2026-09-18)
//!
//! `decoder_layer_one_flat` and `decoder_two_layers_flat` both BAKE (`ops=52`/`ops=104`) and both
//! refuse in `dxp_standalone` at `sdsc_22`, which is `transpose_o31` — a `[mb 64, out 64]` fp16
//! `interslicetranspose_fp16`. The refusal is arch-dependent and MOVES rather than clearing:
//!
//! ```text
//!   SENARCH=mpw4        exit 1  ddc/ddcv1.cpp:3414                "Implicit syncs not available
//!                                                                  for architectures prior to
//!                                                                  RCUDD1A"           (on sdsc_22)
//!   default = rcudd1a   exit 1  dsc-based-utils/DSC2ToDataflowIR/  "Translator currently supports
//!                               V3/SNComputeLowering.cpp:74         translating only 1-D dynamic
//!                                                                   masking"
//! ```
//!
//! (`DEFAULT_ISA = RCUDD1A_ISA`, `sys-arch-spec/isa/isa.hpp:31`, so unset ≡ `SENARCH=rcudd1a`;
//! `SENARCH=mpw4` → `MPW4_ISA`, `isa.cpp:1239-1240`.)
//!
//! ⭐ THE MECHANISM, FROM THE DEFINITIONS. `Ddc::transformForInterSliceRestickify`
//! (`ddc/ddc_transformation.cpp:2398`, called unconditionally from `Ddc::run_v1`,
//! `ddc/ddcv1.cpp:3764`) fires on any PT compute node whose INPUT stick order differs from its
//! OUTPUT stick order — every relayout — and then writes one mask offset per dim of
//! `compNode->getParentDimLoop(outputStickDimOrder[0])`
//! (`ddc_transformation.cpp:2383-2397`). `SNComputeLowering::constructDynamicMasking` accepts
//! exactly ONE loop node (`SNComputeLowering.cpp:68`) carrying exactly ONE dim (`:74`).
//!
//! ⛔⛔⛔ SO THE KILLING PROPERTY IS THE **TWO-DIM OUTPUT STICK**, not the shape and not the core
//! division. `interslicetranspose_fp16`'s output stick is the `8×8` inter-slice block over
//! (`out`, `mb`) — `emit_sdsc`'s `is_transpose_out` arm — so `getStickSizes(OUTPUT)` has two
//! entries, DDC takes `.at(0)` = `out`, and the innermost `out`-carrying loop of a 2-D
//! (`mb`, `out`) tile in this path is the FUSED `loop_dsX_dsY_out_mb` (dims `[mb, out]`, read out
//! of `debug/sdsc_*/sdsc.json` for every 2-D program of the same bundle). Two dims ⇒ `:74`.
//!
//! The restickify door does the SAME transposition with a ONE-dim output stick, DDC picks its
//! single-dim `loop_ds2_ds3_y`, and the same masked-MACC path is accepted — measured
//! `computeMaskLoopOffsets_ = {"0": {"loop_ds2_ds3_y": {"y": 1}}}` on eight
//! `compute_ptrowN_fma16_masked` nodes, exit 0, a 1456-line DataflowIR with
//! `dataflow.implicit_sync_on_streaming_buffer`.
//!
//! # WHAT THIS FILE ASSERTS, AND WHY THAT IS THE RIGHT ORACLE
//!
//! `cargo test` cannot run `dxp_standalone`. What it CAN pin is the one descriptor property the
//! refusal is a function of — the output stick's dim count — on both doors, so that a future change
//! to either one is a test failure that says "re-run the dxp leg" instead of a silent
//! re-introduction. The shape sweep is here because it was the discriminating control: all four
//! shapes refuse identically, including two whose core division splits BOTH axes like the golden
//! `ddc/ddl_templates/test/sdsc_interslicetranspose.json` (`{"out": 8, "mb": 4}`), which is what
//! ruled the division out.
//!
//! ⚖️ THAT SWEEP WAS MEASURED UNDER A DIFFERENT DIVIDER, and the shapes no longer all reach a
//! descriptor. `distribute_cores` now splits `mb` FIRST and to the hilt (`work.rs`'s flat-activation
//! row-mixing fix), so at `mb == 64` each of 32 cores gets 2 rows and `transpose_opspec`'s `[8, 8]`
//! block guard refuses before emission. The dxp measurement above STANDS as the reason the division
//! was ruled out — it was taken when those shapes did divide like the golden — but it is history, not
//! something this file can still reproduce. The sweep therefore records each shape as
//! refused-by-division or emitted-with-a-two-dim-stick, and asserts that at least one still emits so
//! the discriminating assertion cannot pass vacuously.
//!
//! # ⭐⭐⭐ WHAT THE REROUTE MEASURED, END TO END (2026-09-24)
//!
//! Both decoder configurations now BAKE and both COMPILE THROUGH dxp at the default RCUDD1A arch —
//! the leg the transpose door could not pass:
//!
//! ```text
//!   decoder_layer_one_flat    KTIR_WHOLE=1 bake  ops=52   files=53
//!   decoder_two_layers_flat   KTIR_WHOLE=1 bake  ops=104  files=105
//!   dxp_standalone --bundle -d <dir> -b sentient        RC=0 for BOTH, spyreCodeDir/ written
//! ```
//!
//! The two relayouts are `sdsc_22.json`/`sdsc_24.json` (`22_transpose_o31`, `24_transpose_o33`), and
//! the descriptor dxp accepted is the one [`the_rerouted_ktir_door_transposes_with_a_one_dim_output_stick_at_every_shape`]
//! pins:
//!
//! ```text
//!   INPUT   layoutDimOrder_ ["y","out"]  stickDimOrder_ ["out"]  stickSize_ [64]
//!   OUTPUT  layoutDimOrder_ ["out","y"]  stickDimOrder_ ["y"]    stickSize_ [64]
//!   computeOp_  exUnit "sfp"  opFuncName "ReStickifyOpHBM"       numCoresUsed_ 1
//! ```
//!
//! ⛔ TWO SEPARATE GAPS BLOCKED THESE CONFIGURATIONS, and the transpose was only the first. Once it
//! was rerouted the walk refused on the SOFTMAX instead: `Elementwise(Sub)` over a `[64, 1]` per-row
//! max, which `pointwise_extents_agree` correctly refuses because the seeded whole-tensor builder
//! addresses every operand at the OUTPUT's extent. That is now emitted through the `EwOperand`
//! builders the refusal itself names — see `lower_ktir_to_superdsc::elementwise_broadcast`, whose own
//! probes (the accepting `Col`/`Mb` arms and the three refusals around them) live beside it in that
//! module's test block.
//!
//! ⛔ NOTHING HERE LOOSENS A GUARD. `try_assemble_transpose` is called in its fallible form, so a
//! shape its core-division law refuses is RECORDED as refused, not forced through.
//!
//! # RE-RUNNING THE dxp LEG
//!
//! This crate has no bundle-dir writer, so the `sdsc_*.json` + `bundle.mlir` input dxp needs comes
//! from the downstream KTIR driver's bake (`KTIR_WHOLE=1 bake_py decoder_layer_one_flat`). The
//! descriptor it writes for `transpose_o31` is the one this file's assertions pin, so the two cannot
//! drift.
//!
//! ```bash
//! export DEEPTOOLS_PATH=<a deeptools checkout>   # $DEEPTOOLS_PATH/ddc/ddl_templates/ is read
//! for d in <bake dir>/*/; do
//!   rm -rf /tmp/r && cp -R "$d" /tmp/r            # dxp WRITES into its input dir
//!   dxp_standalone -d /tmp/r -b sentient; echo "$d -> $?"
//! done
//! ```
//!
//! ⛔ `inter_slice_transpose.ddl` IS NOT IN THE INSTALLED TOOLCHAIN. It and
//! `inter_slice_transpose_with_bottomdatastage.ddl` are the only two `.ddl` in
//! `ddc/ddl_templates/` that `DDL_TEMPLATES` (that directory's `CMakeLists.txt:8-32`) does not
//! list, and the install/obfuscate rules iterate that list alone (`:59-72`) — so a run against an
//! INSTALL rather than a source checkout dies earlier, on a missing template.

use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::emit::{assemble_restickify_kt_2d, try_assemble_transpose};

/// The shapes, and why each one is in the list.
///
/// `out` is the STICK axis, so `cap(out) = out / 64` for fp16 — that is why the decoder's own shape
/// cannot split `out` at all, and why the first two entries pin `numWkSlicesPerDim_["out"] == 1`.
const SHAPES: &[(u32, u32, &str)] = &[
    // THE DECODER'S OWN SHAPE. `transpose_o31`/`transpose_o33` of `decoder_layer_one_flat`,
    // byte-for-byte the `N_ {out_ 64, mb_ 64, y_ 1}` those two programs carry.
    (64, 64, "decoder"),
    // ONE STICK WIDE, MORE ROWS: `out` still unsplittable, so it isolates "is it the ROW count?".
    (256, 64, "one_stick_wide"),
    // EIGHT STICKS WIDE: the first shape where `out` CAN be split, so it isolates "is it the
    // single `out` slice?" — the one difference from the golden our divider cannot remove at 64.
    (64, 512, "out_splittable"),
    // BOTH AXES WIDE, the golden's own proportions shrunk to fit.
    (128, 1024, "both_wide"),
];

/// The output stick's `(dim order, sizes)` — the property `Ddc::transformForInterSliceRestickify`
/// reads and the V3 translator's 1-D limit is a function of.
fn output_stick(op: &EmittedOp) -> (Vec<&'static str>, Vec<u32>) {
    let dsc = op
        .op
        .as_ref()
        .expect("a relayout carries a SuperDSC descriptor");
    let one = dsc.dscs_.first().expect("one dsc");
    let body = one.values().next().expect("one op in the dsc");
    let out = body.primaryDsInfo_.get("OUTPUT").expect("an OUTPUT layout");
    (out.stickDimOrder_.clone(), out.stickSize_.clone())
}

/// ⛔ THE TRANSPOSE DOOR CARRIES THE TWO-DIM BLOCK STICK WHEREVER IT EMITS AT ALL, and that — not the
/// shape — is what `SNComputeLowering.cpp:74` refuses.
///
/// ⚖️ THE SWEEP NOW RECORDS **TWO** REFUSAL MODES, AND ONLY ONE OF THEM IS THE DXP ONE. When this
/// list was written `distribute_cores` split both axes and all four shapes reached a descriptor, which
/// is what let the sweep rule the core division out: shapes dividing like the golden
/// `sdsc_interslicetranspose.json` refused in dxp exactly as the decoder's own did. The divider has
/// since been changed to split `mb` FIRST and to the hilt (`work.rs`'s row-mixing fix), so at
/// `mb == 64` each of 32 cores gets 2 rows and [`transpose_opspec`]'s own block guard refuses BEFORE a
/// descriptor exists. That guard is CORRECT and is not touched here — a per-core extent that is not a
/// whole multiple of 8 really would put two cores inside one `[8, 8]` output stick.
///
/// So each shape is recorded as whichever it is, and the assertion that carries the discriminating
/// power still runs: EVERY shape that does emit carries `["out","mb"]`/`[8, 8]`. The count is asserted
/// too, so a change that silently stopped emitting anything at all could not pass this vacuously.
#[test]
fn every_transpose_shape_carries_the_two_dim_block_stick_that_dxp_refuses() {
    let mut emitted = 0;
    for &(mb, out, label) in SHAPES {
        let mut sid = -1i64;
        let op = match try_assemble_transpose("tr", mb, out, "t_in", "t_out", &mut sid, None) {
            // THE CORE-DIVISION GUARD, not the dxp one. It is named so a reader cannot mistake this
            // line for the two-dim-stick refusal the module is about, and so a DIFFERENT refusal
            // appearing here is a failure rather than a silent pass.
            Err(e) => {
                assert!(
                    e.contains("the core division gives each core"),
                    "transpose [{mb}, {out}] ({label}) is refused by something OTHER than the \
                     block-division guard, so this sweep no longer measures what it says: {e}"
                );
                println!("REFUSED  {label:16} [mb {mb}, out {out}]  core division: {e}");
                continue;
            }
            Ok(op) => op,
        };
        emitted += 1;
        let dsc = op
            .op
            .as_ref()
            .expect("a transpose carries a SuperDSC descriptor");
        let (order, sizes) = output_stick(&op);
        println!(
            "BAKED    {label:16} [mb {mb}, out {out}]  numWkSlicesPerDim_={:?}  OUTPUT stick {order:?} {sizes:?}",
            dsc.numWkSlicesPerDim_,
        );
        assert_eq!(
            (order.as_slice(), sizes.as_slice()),
            (["out", "mb"].as_slice(), [8u32, 8].as_slice()),
            "{label} [mb {mb}, out {out}]: the transpose's OUTPUT stick is what DDC turns into a \
             dynamic mask, and a two-dim stick is what the V3 translator refuses at \
             SNComputeLowering.cpp:74. If this is now ONE dim, the refusal may be gone — re-run \
             the dxp leg (see the module docs) before changing this assertion",
        );
    }
    assert!(
        emitted > 0,
        "NO shape in the sweep reached a descriptor, so the two-dim-stick assertion never ran and \
         this test proved nothing. `OpFunc::Transpose`'s output stick is the property the reroute is \
         justified by; if every shape is now refused before emission, add one whose per-core extents \
         are both multiples of 8 rather than leaving the sweep vacuous"
    );
}

/// ⭐ THE OTHER DOOR TO THE SAME RELAYOUT, and the one that BAKES.
///
/// [`assemble_restickify_kt_2d`] realizes the same `64 × 64` transposition by moving the STICK AXIS
/// instead of by an inter-slice block move: `OpFunc::Restickify` ⇒ `OpFuncs::ReStickifyOpHBM`,
/// which maps to `restickify.ddl` at both `RCUDD1A_ISA` and `MPW4_ISA`
/// (`ddc/ddl/ddl_conversion.h:260-266`) against `interslicetranspose`'s single ungated,
/// never-installed `inter_slice_transpose.ddl` (`:252`), and which flips `datastageBasedElemOff`
/// (`ddc/ddcv1.cpp:3711-3715`, true for exactly these two op funcs).
///
/// MEASURED through `dxp_standalone` at the default arch: **exit 0**, a `spyreCodeDir/` with a
/// 477-byte all-concrete plan and a 3200-byte `init_binary.bin` whose md5 changes when either
/// operand's start address is perturbed by one 128-B granule. At `SENARCH=mpw4` it refuses at the
/// SAME `ddcv1.cpp:3414` implicit-sync gate — `restickify.ddl:80` carries an unconditional
/// `ddl.implicit_sync` too, and its MPW4 entry names the same file — so MPW4 cannot do ANY
/// relayout, which is the correction this measurement makes to the recorded diagnosis.
///
/// ⚖️ THIS IS A DOOR TEST, NOT A CLAIM THAT THE SUBSTITUTION IS CORRECT. The two doors have
/// different contracts: the restickify's INPUT must already be the slab-major
/// `dev_off([cap,hd],1,·)` store its own doc names, its extents arrive as the paged-KV newtypes
/// [`KtTileSlots`]/[`KtTileFeats`] whose only 64-valued constructors are `of_row_window`/
/// `of_head_dim` (so there is no door for an arbitrary extent), and it rides `distribute_cores`
/// rather than the transpose's block-aware divider. Substituting it at
/// `emit::lower_ktir_to_superdsc.rs:4006` is a vendor change with those three questions open.
#[test]
fn the_restickify_door_carries_a_one_dim_output_stick_for_the_same_relayout() {
    use ktir_superdsc::addr::DevOff;
    use ktir_superdsc::sdsc_abstract::{KtTileFeats, KtTileSlots, WindowRows};

    let slots = KtTileSlots::of_row_window(WindowRows);
    let feats = KtTileFeats::of_head_dim(64);
    assert_eq!(
        (slots.extent(), feats.extent()),
        (64, 64),
        "this test compares the two doors AT THE DECODER'S OWN 64x64 TILE; if `WindowRows` no \
         longer spans 64 slots the comparison is between different relayouts",
    );

    let mut sid = -1i64;
    let op = assemble_restickify_kt_2d(
        "rk",
        slots,
        feats,
        "k_nat",
        DevOff::ZERO,
        "k_t",
        DevOff::ZERO,
        &mut sid,
        None,
    );
    let dsc = op
        .op
        .as_ref()
        .expect("a restickify carries a SuperDSC descriptor");
    let (order, sizes) = output_stick(&op);
    println!(
        "BAKED    restickify_kt    [slots 64, feats 64]  numWkSlicesPerDim_={:?}  OUTPUT stick {order:?} {sizes:?}",
        dsc.numWkSlicesPerDim_,
    );
    assert_eq!(
        (order.len(), sizes.as_slice()),
        (1, [64u32].as_slice()),
        "the restickify door bakes through dxp BECAUSE its output stick is one-dim, which lets \
         `Ddc::transformForInterSliceRestickify` land on a single-dim loop. A two-dim stick here \
         would put this door on the same refusal as the transpose",
    );
}

// ══════════════════════════════════════════════════════════════════════════════════════════════════
//  ⭐⭐⭐ THE DOOR AFTER THE REROUTE — `lower_ktir_to_superdsc::transpose` now calls
//  `try_assemble_restickify_transpose_2d`, so these assert THE THING THE DECODERS ACTUALLY EMIT.
//
//  The two tests above are KEPT and neither is loosened: the first still pins that `OpFunc::Transpose`
//  carries the two-dim block stick at every shape (so a future re-route back to it is a test failure
//  that says "re-run the dxp leg"), and the second still pins the paged-KV door it shares a builder
//  with.
// ══════════════════════════════════════════════════════════════════════════════════════════════════

/// ⭐ THE REROUTED DOOR CARRIES THE ONE-DIM STICK AT EVERY SHAPE THE TRANSPOSE DOOR REFUSED — and the
/// descriptor's own `layoutDimOrder_` says it is a TRANSPOSITION and not an identity copy.
///
/// The `INPUT`/`OUTPUT` dim orders are `["y","out"]`/`["out","y"]` — swapped — and the OUTPUT's stick
/// is the single `y` axis. That pair is the whole contract: the swap is the relayout, and the one-dim
/// stick is what `Ddc::transformForInterSliceRestickify` can land on a single-dim loop for
/// `SNComputeLowering.cpp:74`. If the two orders were EQUAL the descriptor would be a re-stick with no
/// transposition — the `restickify_v_opspec_2d` contract, which is a different op — and it would carry
/// exactly the one-dim stick the first assertion accepts. So both are needed.
#[test]
fn the_rerouted_ktir_door_transposes_with_a_one_dim_output_stick_at_every_shape() {
    for &(mb, out, label) in SHAPES {
        let mut sid = -1i64;
        let op = ktir_superdsc::emit::try_assemble_restickify_transpose_2d(
            "tr", mb, out, "t_in", "t_out", &mut sid, None,
        )
        .unwrap_or_else(|e| {
            panic!(
                "the KTIR transpose door must accept [{mb}, {out}] ({label}) — this is the shape set \
                 the transpose door was measured refusing through dxp, so a refusal here would mean \
                 the reroute solves nothing: {e}"
            )
        });
        let dsc = op
            .op
            .as_ref()
            .expect("a relayout carries a SuperDSC descriptor");
        let body = dsc.dscs_[0].values().next().expect("one op in the dsc");
        let (inp, outp) = (
            &body.primaryDsInfo_["INPUT"],
            &body.primaryDsInfo_["OUTPUT"],
        );
        let (order, sizes) = output_stick(&op);
        println!(
            "REROUTED {label:16} [mb {mb}, out {out}]  numWkSlicesPerDim_={:?}  \
             INPUT {:?} stick {:?}  OUTPUT {:?} stick {order:?} {sizes:?}",
            dsc.numWkSlicesPerDim_, inp.layoutDimOrder_, inp.stickDimOrder_, outp.layoutDimOrder_,
        );
        assert_eq!(
            order.len(),
            1,
            "{label} [mb {mb}, out {out}]: the rerouted door's OUTPUT stick must be ONE dim — that is \
             the only property `SNComputeLowering.cpp:74` is a function of, and the reason the \
             transpose door could not be translated at the default RCUDD1A arch",
        );
        assert_eq!(
            (
                inp.layoutDimOrder_.as_slice(),
                outp.layoutDimOrder_.as_slice()
            ),
            (["y", "out"].as_slice(), ["out", "y"].as_slice()),
            "{label} [mb {mb}, out {out}]: the two roles' dim orders must be SWAPPED — that swap IS \
             the transposition",
        );
        assert_eq!(
            (inp.stickDimOrder_.as_slice(), order.as_slice()),
            (["out"].as_slice(), ["y"].as_slice()),
            "{label} [mb {mb}, out {out}]: the INPUT sticks on `out` (the feature axis, its LAST dim — \
             the slab-major read a rank-2 pointwise producer writes) and the OUTPUT on `y` (the slot \
             axis, ITS last dim — the residency `StickLayout::kernel` reads). Either one moving \
             re-points the descriptor at bytes nobody wrote",
        );
    }
}

/// ⛔⛔⛔ AND IT FAILS CLOSED ON LX, WITH A CONTROL ONE VARIABLE AWAY.
///
/// `restickify_kt_opspec_2d` carries `time_tile: None` and cannot be time-tiled — its output stick IS
/// the whole `y` axis, so there is no stick dim left to divide in time without splitting an output
/// stick. `transpose_opspec` DOES time-tile (via `TileOp::tile`), so the reroute genuinely narrows what
/// this door accepts, and the narrowing has to be a NAMED REFUSAL rather than a descriptor that
/// addresses past the 1_677_721-B scratchpad.
///
/// `[4096, 4096]`: `distribute_cores` splits `y` by its 64 sticks to all 32 cores and has none left for
/// `out`, so each core holds (y 128, out 4096) = 2 × 1_048_576 B = 2_097_152 B. REFUSED.
/// `[4096, 2048]` is the SAME division on the SAME axis with half the columns — (y 128, out 2048),
/// 1_048_576 B — and is ACCEPTED. One variable, and it is the one the guard reads.
#[test]
fn an_over_lx_transpose_is_refused_and_the_half_size_control_is_accepted() {
    let mut sid = -1i64;
    // `EmittedOp` is not `Debug`, so the Ok arm is named by hand rather than through `expect_err`.
    let e = match ktir_superdsc::emit::try_assemble_restickify_transpose_2d(
        "big", 4096, 4096, "t_in", "t_out", &mut sid, None,
    ) {
        Ok(_) => panic!(
            "[4096, 4096] EMITTED: its per-core tile is 2_097_152 B against the 1_677_721-B LX \
             scratchpad, so this door is now handing dxp a descriptor that addresses past LX"
        ),
        Err(e) => e,
    };
    println!("REFUSED  [4096, 4096]: {e}");
    assert!(
        e.contains("1677721") && e.contains("time_tile"),
        "the refusal must name the budget AND say why this relayout cannot be time-tiled instead of \
         refused — otherwise it reads as a shape complaint; got: {e}"
    );

    let mut sid = -1i64;
    let op = ktir_superdsc::emit::try_assemble_restickify_transpose_2d(
        "ctl", 4096, 2048, "t_in", "t_out", &mut sid, None,
    )
    .expect("the half-width control fits LX and must still be emitted");
    let (order, _) = output_stick(&op);
    println!(
        "ACCEPTED [4096, 2048] control: numWkSlicesPerDim_={:?}  OUTPUT stick {order:?}",
        op.op.as_ref().expect("a descriptor").numWkSlicesPerDim_,
    );
    assert_eq!(
        order.len(),
        1,
        "the control must be a real emission with the same one-dim stick, not merely 'not an error'",
    );
}

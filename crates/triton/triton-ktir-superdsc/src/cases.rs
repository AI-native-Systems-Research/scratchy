// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! THE TWENTY-FIVE CONFIGURATIONS, STATED ONCE.
//!
//! A "configuration" is a `.py` fixture plus the `tl.constexpr` values and the launch grid it is
//! compiled at. Three consumers need that tuple: `examples/bake_py` (the dxp bake),
//! `triton-ktir/tests/pure_rust_ktir.rs` (the KTIR diff), and `triton-numeric` (the emulator's
//! NUMERIC check). It lived in `examples/bake_py.rs` and is now here, `pub`, because a
//! configuration restated in two places is a configuration that drifts in one of them -- the
//! failure this crate has already recorded twice (`rope_q32`'s grid became `vec![4, 32]` in one
//! statement of it and 252 of 256 token positions were never written, and it BAKED).
//!
//! ⛔ THE GRID IS PART OF THE CONFIGURATION, NOT A DETAIL. See the `rope_q32` comment below.
//!
//! Nothing here reads a file but the fixture `.py` named in the `KernelSpec`.

use std::collections::HashMap;
use std::path::PathBuf;

use ktir_superdsc::ktir_node::{Elementwise, Program};
use triton_frontend::codegen::{ArgSpec, KernelSpec};
use triton_frontend::semantic::Val;

/// Every configuration name `case` answers to, so a consumer can sweep them rather than
/// restating the list. Order is the table's own.
pub const ALL: &[&str] = &[
    "rmsnorm_granite",
    "rmsnorm_granite_m32",
    "swiglu_mlp_small",
    "swiglu_mlp_flat",
    "swiglu_mlp_granite_flat",
    "swiglu_mlp_tiled_k",
    "swiglu_mlp_granite",
    "swiglu_mlp_granite_tiled_k",
    "rope_q32",
    "rope_kv8",
    "embedding_granite",
    "embedding_granite_bm128",
    "embedding_granite_m32",
    "decoder_layer_one",
    "decoder_layer_one_flat",
    "decoder_two_layers",
    "decoder_two_layers_flat",
    // Attention, through `attn_fwd_noncausal` (fixture delta 11 -- the four-pointer entry point;
    // see the `case` arm for why a five-pointer noncausal launch is a handoff refusal), and the
    // causal two-stage split through `attn_fwd` itself.
    "attention_flash_noncausal",
    "attention_flash_noncausal_1tile",
    "attention_flash_causal",
    // ⭐ fp8 W8A8, the WEIGHT-QUANTIZED matmul -- scratchy's card-proven chain reached
    // through the arity-3 `matmul_oriented` door (A, W_fp8, w_scale over a view whose
    // `Dtype` is `Fp8E4m3`). `Program::Matmul` is the per-`Program` door, which hands
    // the node's FULL region list, so the 4-parameter kernel reaches the fp8 arm with
    // zero vendored changes. The small twin mirrors `swiglu_mlp_small`'s shape family.
    "matmul_fp8_small",
    // ⭐ RUNG 3 of the address-provenance ladder: the same one-block multiply as `mul`,
    // with the scale a RUNTIME fp32 ARGUMENT instead of a second loaded tensor or a
    // `tl.constexpr`. The measurement twin for the launcher-bound scale registry -- the
    // signature entry is `ArgSpec::Scalar`, which the frontend already lowers to a block
    // arg, so what this row names is where the chain refuses a splat OF an argument.
    "mul_scale_small",
    // ⭐ RUNG 4, the gathered CONTRACTION (see the `paged_score_small` case row).
    "paged_score_small",
    // ⭐ TASK 27: rung 4 AROUND A FULL ONE-BLOCK FLASH STEP (see the `paged_attention_small`
    // case row).
    "paged_attention_small",
    // ⭐ TASK 28: the multi-block sweep — block table on device, flash recurrence carried
    // across blocks (see the `paged_attention_multiblock` case row).
    "paged_attention_multiblock",
];

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate sits at crates/triton/triton-ktir-superdsc")
        .to_path_buf()
}

/// A kernel's launch contract, in the same shape `pure_rust_ktir.rs` states it -- so the two
/// cannot drift about what a configuration IS.
pub fn spec(fixture: &str, kernel: &str, ptrs: &[(&str, &str)], ces: &[(&str, Val)]) -> KernelSpec {
    let mut signature = HashMap::new();
    for (k, v) in ptrs {
        signature.insert((*k).to_string(), ArgSpec::parse(v).expect("signature"));
    }
    let mut constexprs = HashMap::new();
    for (k, v) in ces {
        signature.insert((*k).to_string(), ArgSpec::parse("constexpr").expect("constexpr"));
        constexprs.insert((*k).to_string(), v.clone());
    }
    KernelSpec {
        kernel: kernel.to_string(),
        signature,
        constexprs,
        file: root()
            .join("test-fixtures")
            .join(format!("{fixture}.py"))
            .to_string_lossy()
            .to_string(),
    }
}

/// One configuration: which `.py`, which kernel, the constexprs, the grid, and the node kind.
///
/// The constexprs and grid are the SAME VALUES as `test/experiment1/index.json` and
/// `pure_rust_ktir.rs`'s case table. `Program` is stated, not inferred -- see the crate docs.
pub fn case(name: &str) -> Option<(KernelSpec, Vec<i64>, Program)> {
    Some(match name {
        "rmsnorm_granite" => (
            spec(
                "rmsnorm",
                "rmsnorm_fwd",
                &[("desc_x", "*fp16"), ("desc_w", "*fp16"), ("desc_o", "*fp16")],
                &[
                    ("M", Val::Int(64)),
                    ("D_MODEL", Val::Int(4096)),
                    ("BLOCK_M", Val::Int(64)),
                    ("EPS", Val::Float(1e-05)),
                    ("INV_D", Val::Float(1.0 / 4096.0)),
                ],
            ),
            vec![1],
            Program::RmsNorm,
        ),
        // THE SAME NORM AT 32 ROWS -- the control for the `mb` FOLD.
        //
        // `rmsnorm_granite` is 64 rows, and 64 rows makes the emitted `mb` axis FOLD by 2 (the fold
        // splits `mb` into chunks of <=32). Measured on card at 64 rows, `rmg`'s mb-broadcast gamma
        // operand is read ROW-INDEXED -- row r fetches gamma's stick r and never advances along the
        // stick axis -- so `out = xn * gamma[r*64 + c%64]` instead of `xn * gamma[c]`. Every other
        // field of that descriptor is identical to granite's own bake (`/work/dump-ktp`), whose norms
        // run at mb<=32 with fold 1, so 32 rows is the ONE configuration that separates "the fold
        // breaks the mb broadcast" from "the mb broadcast is broken at rows>1 regardless of folding".
        // Same D_MODEL, so the stick-major predicate (`cols > 64`) is unchanged and only `mb` moves.
        "rmsnorm_granite_m32" => (
            spec(
                "rmsnorm",
                "rmsnorm_fwd",
                &[("desc_x", "*fp16"), ("desc_w", "*fp16"), ("desc_o", "*fp16")],
                &[
                    ("M", Val::Int(32)),
                    ("D_MODEL", Val::Int(4096)),
                    ("BLOCK_M", Val::Int(32)),
                    ("EPS", Val::Float(1e-05)),
                    ("INV_D", Val::Float(1.0 / 4096.0)),
                ],
            ),
            vec![1],
            Program::RmsNorm,
        ),
        // SWIGLU AT TWO BLOCKINGS, to separate two problems that look like one.
        //
        // The kernel has two loops: `for n in tl.range(0, D_FF, BLOCK_N)` and, inside it,
        // `for k in tl.range(0, D_MODEL, BLOCK_K)`. Delta 12 of the fixture records that
        // `BLOCK_K == D_MODEL` makes the inner one single-trip and `canonicalize` deletes it.
        // `swiglu_mlp_small` already sets that, so the loop the nested-window guard refuses is
        // the OUTER one -- 4 trips at D_FF=256 / BLOCK_N=64.
        //
        // `_flat` sets BOTH to their full extent, so both loops are single-trip and the program
        // is straight-line. It is a MEASUREMENT, not a proposed fixture: if the refusal moves,
        // the loops were the whole of that blocker; if it does not, they were not.
        "swiglu_mlp_small" => (
            spec("swiglu_mlp", "swiglu_mlp_fwd", SWIGLU, &swiglu_ce(128, 256, 64, 128)),
            vec![1],
            Program::Matmul,
        ),
        "swiglu_mlp_flat" => (
            spec("swiglu_mlp", "swiglu_mlp_fwd", SWIGLU, &swiglu_ce(128, 256, 256, 128)),
            vec![1],
            Program::Matmul,
        ),
        // GRANITE WIDTH, FLAT. D_FF=12800, so this states a [64, 12800] f16 extent = 1.56 MiB --
        // more than an LX tile. THE POINT IS THAT WE DO NOT MATERIALISE IT: their
        // `WorkPlan::time_tile_for_lx` is the only minter of `TimeTile` and turns "does not fit LX"
        // into a build error, so stating the full extent and letting THEIR tiler slice it is the
        // division of labour their own types describe. This measures whether that actually happens
        // or whether the extent is refused.
        "swiglu_mlp_granite_flat" => (
            spec("swiglu_mlp", "swiglu_mlp_fwd", SWIGLU, &swiglu_ce(4096, 12800, 12800, 4096)),
            vec![1],
            Program::Matmul,
        ),
        // SWIGLU AS `pure_rust_ktir.rs` STATES IT -- the BLOCKED forms, kept beside the flat ones
        // so "does un-blocking clear this?" is a question one binary answers twice.
        "swiglu_mlp_tiled_k" => (
            spec("swiglu_mlp", "swiglu_mlp_fwd", SWIGLU, &swiglu_ce_m(16, 128, 256, 128, 64)),
            vec![1],
            Program::Matmul,
        ),
        "swiglu_mlp_granite" => (
            spec("swiglu_mlp", "swiglu_mlp_fwd", SWIGLU, &swiglu_ce(4096, 12800, 64, 4096)),
            vec![1],
            Program::Matmul,
        ),
        // ⭐⭐ GRANITE WIDTH AT THE BLOCKING WHOSE TILES ACTUALLY FIT A 2 MiB LX, which is the
        // configuration `triton-numeric` gets a NUMBER from. The two above do not: `_flat` refuses
        // at `TensorSplat 1638400 + 1638400 > 2097152` charging a `[64, 12800]` tile and
        // `swiglu_mlp_granite` at `ScfFor 1589248 + 524288 > 2097152` charging a `[64, 4096]` one.
        //
        // ⛔ AND BOTH KNOBS ARE MEASURED, NOT PICKED. Swept with the KTIR executor over zero-filled
        // buffers of the real extents (the LX charge is a function of the tile EXTENTS, so the
        // values are irrelevant to it):
        //
        //   BLOCK_N = 64 is FORCED, by the DOWN-PROJECTION WEIGHT TILE `[D_MODEL, BLOCK_N]` and not
        //   by the activation width. At 128 the charge is `1130496 + 1048576` for a `[4096, 128]`
        //   tile and it refuses at BLOCK_K 2048, 1024 AND 512 -- the same two numbers all three
        //   times, which is what shows the overflow is BLOCK_K-INDEPENDENT. 256 asks for the whole
        //   2097152 B and 320 for 2621440 B. 64 charges 524288 B, and with the `[64, 4096]`
        //   loop-carried `acc` beside it that is 1 MiB of the 2.
        //
        //   BLOCK_K = 2048 is then the LARGEST DIVISOR OF D_MODEL THAT FITS: 4096 refuses (it is
        //   `swiglu_mlp_granite`, above), 2048 / 1024 / 512 all execute. The largest is chosen
        //   deliberately -- it is the FEWEST extra f16 roundings on the gate/up path, i.e. the
        //   blocking that deviates least from the flat one, and `bounds::swiglu` charges the
        //   envelope for every trip so a smaller BLOCK_K would only loosen the gate.
        //
        // ⭐ AND T_k = 2 IS THE SAME TRIP COUNT THE SuperDSC SIDE MEASURED INDEPENDENTLY. Patch
        // `4bff6e74f` ("ONE NODE PER K TRIP", `vendor/PROVENANCE.md`) reports `64x4096x12800` needs
        // T=2 from `plan_k_trips`' own tiler. Two different residency models, the same answer.
        //
        // ⛔ WHAT IT IS NOT: a new kernel. `BLOCK_N`/`BLOCK_K` are TILING knobs of `swiglu_mlp.py`
        // (its delta 12), the computed function is identical, and `gen_numeric_data.py` draws
        // byte-identical bytes for this config and `swiglu_mlp_granite_flat` because
        // `_swiglu_inputs` reads only M/D_MODEL/D_FF.
        //
        // ⛔⛔⛔ AND IT DOES **NOT** BAKE, WHICH IS THE OTHER HALF OF THE MEASUREMENT AND MUST NOT BE
        // READ AS A DEFECT IN EITHER CONSUMER. The two consumers of this kernel want OPPOSITE
        // blockings, and both refusals are about the SAME structural fact from different sides:
        //
        //   * the KTIR EMULATOR has no tiler and runs the program as written against a 2 MiB LX, so
        //     at Granite width it needs the loops to SURVIVE and be MULTI-trip. It refuses `_flat`.
        //   * the SuperDSC WHOLE-FUNCTION DOOR walks the function's TOP LEVEL, so it needs the loops
        //     to be GONE -- `whole_function.rs` refuses an `ktdp.construct_access_tile` window stated
        //     inside an `scf.for` body by name ("A time-tiled program has to be split into one node
        //     per trip, or its windows hoisted"). It refuses every blocked form.
        //
        // MEASURED, and the refusal is BLOCKING-driven and not WIDTH-driven, which is what makes it
        // a structural statement: `KTIR_WHOLE=1 bake_py` gives `swiglu_mlp_flat` ops=5 and
        // `swiglu_mlp_granite_flat` ops=7, while `swiglu_mlp_tiled_k` (128/256!), `swiglu_mlp_granite`
        // and `swiglu_mlp_granite_tiled_k` all REFUSE at `regions` with that identical message.
        //
        // ⭐ AND THE SPLIT IS ALREADY DONE ON THE OTHER SIDE, WHICH IS WHY THIS IS A DIVISION OF
        // LABOUR RATHER THAN A HOLE: `vendor/PROVENANCE.md`'s `4bff6e74f` "ONE NODE PER K TRIP" takes
        // the FLAT Granite program and splits the contraction into T=2 whole matmuls plus explicit
        // adds INSIDE the SuperDSC lowering -- the same trip count this config states in Triton.
        //
        // ⚖️ SO THE NUMERIC NUMBER AND THE dxp NUMBER ARE AT DIFFERENT BLOCKINGS OF ONE KERNEL, AND
        // NEITHER SUBSTITUTES FOR THE OTHER. `dxp_standalone` executes no arithmetic, so a green bake
        // of `_flat` says nothing about its values; the emulator is layout-agnostic, so a green
        // comparison here says nothing about descriptors. What the pair DOES establish is that the
        // FUNCTION `swiglu_mlp.py` computes is right at Granite width, which is a property of the
        // `.py` and not of a blocking -- and that is exactly the claim `_flat` could not make.
        "swiglu_mlp_granite_tiled_k" => (
            spec("swiglu_mlp", "swiglu_mlp_fwd", SWIGLU, &swiglu_ce(4096, 12800, 64, 2048)),
            vec![1],
            Program::Matmul,
        ),
        // ROPE. THE GRID IS ONE WORK ITEM PER TOKEN POSITION -- `vec![N_TOK]` -- and it is ONE
        // DIMENSION, not two.
        //
        // ⛔⛔⛔ IT USED TO BE `vec![4, 32]` / `vec![4, 8]`, AND THAT WAS A SILENT WRONG ANSWER THE
        // MOMENT THE KERNEL BECAME TOKEN-MAJOR. The old kernel took `start_m = program_id(0)` over
        // `N_TOK / BLOCK_M` blocks and `off_h = program_id(1)` over the heads, so a 2-D grid was its
        // launch contract. The token-major kernel reads `pos = tl.program_id(0)` ALONE and puts all
        // `H` heads of a position in one `[H, HALF]` tile, so there is no head axis to launch: with
        // `vec![4, 32]` the head axis became 32 REDUNDANT LAUNCHES of the same work and `pos` only
        // ever reached 0..4 of 256 positions -- 252 positions never written, and it BAKED, because
        // nothing downstream compares a grid against the extent a kernel indexes.
        //
        // `H` still separates the two configurations, but now only through the plane width
        // (`H * HEAD_DIM`), which is exactly what `drive_rope` cross-checks against the output view.
        "rope_q32" => (spec("rope", "rope_fwd", ROPE, &rope_ce(32)), vec![256], Program::Rope),
        "rope_kv8" => (spec("rope", "rope_fwd", ROPE, &rope_ce(8)), vec![256], Program::Rope),
        // EMBEDDING. `Program::ScalarMul` is the kind of the one COMPUTE op (`rows * EMB_SCALE`);
        // the gather itself is addressing, and whether this crate has a door for an INDIRECT access
        // tile is what these three measure. BLOCK_M 64 is the Granite launch; the 128 case is the
        // control that isolated the one-stick index tile (`pure_rust_ktir.rs`'s note).
        //
        // ⭐⭐⭐⭐⭐ AND THE TWO GRANITE-WIDTH ONES ARE **EXACT ON CARD**, EACH AS EIGHT OPS. Both bake
        // the SAME descriptor -- `KTIR_WHOLE=1` and the per-`Program` door agree byte-for-byte, and
        // `embedding_granite` and `_bm128` do too, because the emission spans the whole node
        // (`node_rows` = the output view's 256 rows) and BLOCK_M only decides the grid the fold
        // absorbs. That node's index is 256 entries against a 32-entry index STICK, and dxp fills the
        // L3LU IBR with ONE stick transfer -- so `assemble_pointwise_broadcast_gather` CUTS it into one
        // op per stick, each reading index words `[32k, 32k+32)` and writing rows `[32k, 32k+32)` of
        // the output IN PLACE. Card, pod `nickm3-5999dffbdf-t2wcl`, both configurations, all eight
        // row-blocks against the fixture's own `ref_out.bin`:
        //
        //   rows[0:32] .. rows[224:256]   within_2pct_strict = 1.000000   (131072 elements each)
        //   rows[0:256]                   within_2pct_strict = 1.000000   max|err| 0.0625 = the f16 ulp
        //
        // with seven collapsing controls (table staged stick-major 0.000752, output not de-scattered
        // 0.006652, ids staged fp16 0.000751, EMB_SCALE bound 1.0 0.000731, ids REVERSED 0.006532, ids
        // ROTATED by one stick 0.006373, ids reversed WITHIN each stick 0.006518 -- the last two are
        // the cut's own, and they are what prove each leg reads its OWN entry stick).
        //
        // ⛔ BEFORE THE CUT THE ONE-OP FORM WAS A SILENT WRONG ANSWER, which is why the guard stays:
        // every output row `r` held the table row `ids[r mod 32]` names -- rows 0..31 EXACT and the
        // whole `[256, 4096]` at 0.130597 ~= 32/256, with the right rms because a vocabulary's rows are
        // independent `randn` draws. It baked clean, dxp exited 0 and the launch returned rc=0.
        "embedding_granite" => (
            spec("embedding", "embedding_fwd", EMBEDDING, &embedding_ce(256, 64)),
            vec![4],
            Program::ScalarMul,
        ),
        "embedding_granite_bm128" => (
            spec("embedding", "embedding_fwd", EMBEDDING, &embedding_ce(256, 128)),
            vec![2],
            Program::ScalarMul,
        ),
        // ⭐⭐⭐ THE ACCEPTING SIDE, AND THE ONE CONFIGURATION THAT MAKES THE GUARD A GUARD RATHER THAN
        // A WITHDRAWAL. N_TOK = 32 is exactly ONE index stick (`CopyDims::ENTRIES_PER_OP`, the
        // `SenUint32` 32-entry stick), so the descriptor's entries fit the IBR and nothing wraps. Same
        // fixture, same vocabulary, same `EMB_SCALE`, same gather -- the ONLY thing moved is the extent
        // the ceiling is about, which is what makes a pass here attributable.
        //
        // ⛔ AND `BLOCK_M` IS 32 TOO, i.e. grid 1, because the entry count is the NODE's row count and
        // not the block's: a `[32, 4096]` node at BLOCK_M 16 would state 16 entries on its indirect
        // tile and still emit a 32-entry index, so moving BLOCK_M here would measure nothing.
        "embedding_granite_m32" => (
            spec("embedding", "embedding_fwd", EMBEDDING, &embedding_ce(32, 32)),
            vec![1],
            Program::ScalarMul,
        ),
        // THE DECODER BLOCK -- 12 matmuls in one function for one layer, 24 for two, so it is the
        // whole-function door's real case. `Program` is stated only because the signature demands
        // one; `KTIR_WHOLE=1` walks the ops and never reads it.
        //
        // `_flat` sets BLOCK_N = D_FF so the MLP's `for n in tl.range(0, D_FF, BLOCK_N)` is
        // single-trip and `canonicalize` deletes it -- the same measurement `swiglu_mlp_flat` is.
        "decoder_layer_one" => (
            spec("decoder_block", "decoder_layer_fwd", &dec_ptrs_one(), &dec_ce(64)),
            vec![1],
            Program::Matmul,
        ),
        "decoder_layer_one_flat" => (
            spec("decoder_block", "decoder_layer_fwd", &dec_ptrs_one(), &dec_ce(256)),
            vec![1],
            Program::Matmul,
        ),
        "decoder_two_layers" => (
            spec("decoder_block", "decoder_two_layers_fwd", &dec_ptrs_two(), &dec_ce(64)),
            vec![1],
            Program::Matmul,
        ),
        "decoder_two_layers_flat" => (
            spec("decoder_block", "decoder_two_layers_fwd", &dec_ptrs_two(), &dec_ce(256)),
            vec![1],
            Program::Matmul,
        ),
        // ⭐ ATTENTION, THE ONE KERNEL THIS TABLE HAD NO ROW FOR — so no attention descriptor has
        // ever gone down the from-source ladder (`bake_py`), only the ttir-golden path that
        // `drive`/`bake` take. The constexprs and grid are `test/experiment1/index.json`'s own for
        // these two configurations, values included rather than defaulted: `sm_scale` is delta 9
        // (a `tl.constexpr`, because a runtime scale still refuses by name) and `STAGE` is what
        // separates them — 1 is non-causal, 3 is the causal two-stage split.
        //
        // ⛔ AND THE TWO ROWS NAME DIFFERENT KERNELS, WHICH IS NOT A TIDY-UP. The non-causal
        // configuration never loads the mask (the inner gets `4 - STAGE == 3`), so a five-pointer
        // signature leaves parameter 4 addressed by nothing and `every_parameter_states_its_width`
        // red-stops — correctly, because the consumer numbers buffers BY PARAMETER POSITION and a
        // trailing hole would silently renumber every buffer after it. `attn_fwd_noncausal` is the
        // fixture's FOUR-pointer entry point (delta 11): a `@triton.jit` wrapper that `make_ttir`'s
        // `inline_calls` inlines, so the ops that reach the emitter are `attn_fwd`'s own.
        "attention_flash_noncausal" => (
            spec("attention_flash", "attn_fwd_noncausal", ATTN_NONCAUSAL, &attn_ce(1)),
            vec![4, 4],
            Program::Attn,
        ),
        // ⭐ THE ONE-TILE FORM OF THE SAME KERNEL — the whole recurrence at the smallest launch that
        // states it. `attn_ce(1)` is untouched (N_CTX=256, BLOCK_M=BLOCK_N=64), so this is ONE query
        // block sweeping ALL FOUR KV blocks: the flash recurrence with real per-trip windows, live
        // accumulator threading (`decompose_matmul_accumulators`' exact case), and `desc_q`/`desc_o`
        // windows at the buffer base. The grid is the ONLY difference from the row above.
        //
        // ⛔ AND THAT DIFFERENCE IS A MEASURED WALL, NOT A TUNING CHOICE. At grid [4,4]=16 the
        // Q/O window corners root at `KtdpGetComputeTileId` (`remui(ctid,4)` → `start_m`), and the
        // whole-function door resolves window corners through `index_constants`, which sees only
        // `arith.constant` (and casts of one). A grid-derived corner has no descriptor spelling
        // through this door — that is the next attention item, and it is OURS. This row exists so
        // everything ELSE attention needs is proven first: it bakes what the 16-tile launch cannot,
        // and the 16-tile launch is one `vec![4,4]` edit away once the corner wall falls.
        "attention_flash_noncausal_1tile" => (
            spec("attention_flash", "attn_fwd_noncausal", ATTN_NONCAUSAL, &attn_ce(1)),
            vec![1, 1],
            Program::Attn,
        ),
        "attention_flash_causal" => (
            spec("attention_flash", "attn_fwd", ATTN, &attn_ce(3)),
            vec![4, 4],
            Program::Attn,
        ),
        // ⭐ fp8 W8A8 AT THE SMALL TWIN'S SHAPES -- the whole contract is the four
        // buffers (x f16 [M,K], w PACKED fp8 [N,K], ws f16 [1,N], o f16 [M,N]) and the
        // kernel body's spelled dequant (see the fixture's own delta list). The
        // descriptors block at the swiglu small twin's extents; `BLOCK_K == K` so the
        // contraction is single-tile, which is the whole-kernel contract the fp8 arm of
        // `verify_canonical_matmul_kernel` admits (a K-looped form is a follow-on).
        "matmul_fp8_small" => (
            spec(
                "matmul_fp8",
                "matmul_fp8_fwd",
                &[
                    ("desc_x", "*fp16"),
                    ("desc_w", "*fp8e4nv"),
                    ("desc_ws", "*fp16"),
                    ("desc_o", "*fp16"),
                ],
                &[
                    ("M", Val::Int(64)),
                    ("K", Val::Int(128)),
                    // N == BLOCK_N: the single-tile contract (the verifier refuses
                    // `BLOCK_N < N` -- the emitter emits ONE M x N program and discards
                    // the block size, so a multi-block N would be silently collapsed).
                    ("N", Val::Int(128)),
                    ("BLOCK_M", Val::Int(64)),
                    ("BLOCK_K", Val::Int(128)),
                    ("BLOCK_N", Val::Int(128)),
                ],
            ),
            vec![1],
            Program::Matmul,
        ),
        // ⭐ RUNG 3, THE MEASUREMENT. The kernel is `mul` with the second pointer
        // replaced by ONE runtime fp32 scalar (`a * scale`), so the whole case is the
        // scale's provenance: `spec` puts `scale` in the signature as `ArgSpec::Scalar`,
        // which the frontend makes a block arg, and the frontend's `broadcast_pair`
        // splats it to the tile -- a `tt.splat` of a NON-constant, which no fixture
        // before this one ever produced. What this row measures is the FIRST refusal
        // the chain raises for that (expected at `from_ttir`, whose op set is the census
        // of the sixteen post-make_ttir goldens and contains no `tt.splat` of an
        // argument); it exists so the rung-3 work has a named, fail-closed before-state.
        // `Program::Elementwise(Mul)` is stated per the constructor's demand; the
        // whole-function door walks the ops and never reads it.
        "mul_scale_small" => (
            spec(
                "mul_scale",
                "mul_scale_kernel",
                &[("a_ptr", "*fp16"), ("c_ptr", "*fp16"), ("scale", "fp16")],
                &[("n", Val::Int(1024)), ("BLOCK", Val::Int(64))],
            ),
            vec![16],
            Program::Elementwise(Elementwise::Mul),
        ),
        // ⭐⭐⭐ RUNG 4 OF THE ADDRESS-PROVENANCE LADDER, THE MEASUREMENT: `q @ gather(k,
        // ids).T` -- a row index read from DEVICE MEMORY, the gathered rows feeding a
        // CONTRACTION. Paged attention's inner step in miniature (the page table chooses
        // the K rows; the score tile contracts them with Q). `embedding.py` covers the
        // gather-reaches-scalarmul half, card-verified through both doors; this row
        // measures the gather-reaches-a-CONTRACTION half. THE V-LEG (SCRATCHY) FORM:
        // `tl.dot(p, v_rows)` with the rows gathered directly and contracted WHERE THEY
        // LIE (row-blocked gathered rows ARE the kernel slot's `[k, n]` residency), no
        // transpose anywhere -- the transposed K leg is refused by `dot_to_linalg`'s
        // paged contract with the arch reason (dxp refuses relayout ops on MPW4), and
        // scratchy's answer to it is the host-presented Kt plane, not a kernel op.
        // K=64 meets BOTH floors: the frontend's own `tl.dot` K>=16 and the vendor matmul
        // emitter's whole-stick K (64 fp16 elements, `assemble_matmul` emits no
        // `coordinateMasking_`); M=8 and HEAD_DIM=64 are the gather's own floors (index
        // vector >= 8, f16 columns >= 16). `Program::Matmul`
        // is stated per the constructor's demand; the whole-function door walks the ops
        // and never reads it.
        "paged_score_small" => (
            spec(
                "paged_score",
                "paged_vmatmul_fwd",
                &[("desc_p", "*fp16"), ("desc_v", "*fp16"), ("desc_ids", "*i32"),
                  ("desc_o", "*fp16")],
                &[("M", Val::Int(8)), ("K", Val::Int(64)), ("V", Val::Int(128)),
                  ("BLOCK_M", Val::Int(8)), ("HEAD_DIM", Val::Int(64))],
            ),
            vec![1],
            Program::Matmul,
        ),
        // ⭐⭐⭐ TASK 27 — PAGED FLASH ATTENTION, THE ONE-BLOCK STEP: the full flash body
        // around rung 4's gathered V-leg contraction. The block table (`desc_ids`) picks
        // the V rows at runtime; the score matmul runs over a HOST-PRESENTED Kt plane
        // (the shipped attention's third physical plane, because a transposed gather is
        // arch-refused and two gathers in one program are refused by `gather_of`); both
        // softmax reductions sit on the tutorial's [M, N] tile at one stick of KV; the
        // value contraction is `tl.dot(p, v_rows)` verbatim. ONE launch = ONE KV block =
        // 64 rows, so the block recurrence is the caller's loop. Multi-dot, so
        // `dot_to_linalg` leaves every dot UNTAGGED (no canonical verifier runs — that
        // gate is single-dot-only) and the whole-function door (`KTIR_WHOLE=1`) walks the
        // ops. `Program::Matmul` is stated per the constructor's demand; the walk never
        // reads it. Geometry: the paged_score floors (K=64 gathered rows, M=8+,
        // HEAD_DIM=64) plus the attention BLOCK_M floor of 64 (a reduce at rows==1 has
        // no per-row split to refuse; BLOCK_M=64 keeps the score tile square at one
        // stick per axis, the smallest shape every floor admits at once).
        "paged_attention_small" => (
            spec(
                "paged_attention",
                "paged_attn_fwd",
                &[("desc_q", "*fp16"), ("desc_kt", "*fp16"), ("desc_v", "*fp16"),
                  ("desc_ids", "*i32"), ("desc_o", "*fp16")],
                &[("N_CTX", Val::Int(64)), ("HEAD_DIM", Val::Int(64)),
                  ("V", Val::Int(1024)), ("BLOCK_M", Val::Int(64))],
            ),
            vec![1],
            Program::Matmul,
        ),
        // ⭐ TASK 28: the paged form PROPER — the KV sweep over a DEVICE-RESIDENT block
        // table with the flash recurrence (m, l, acc) carried across blocks in ONE
        // program. `unroll_constant_trip_loops` turns the sweep into straight-line
        // trips (the same mechanism the dense 16-tile form card-verified), each trip
        // loading its block-table row at a CONSTANT corner (`[b*BLOCK_N]`) and each
        // carrying its OWN gather of the V rows through those indices — so what is new
        // relative to the dense form is a gather per unrolled trip, and what is new
        // relative to `paged_attention_small` is the recurrence and the per-block index
        // slices. Two blocks of 64: the smallest shape that has an INTERIOR join (trip
        // 0's yield feeding trip 1's carry), the property the one-block form cannot
        // test. The Kt plane is `[HEAD_DIM, NUM_BLOCKS*BLOCK_N]`, block t's K^T in
        // column block t, host-pre-transposed at page fill.
        "paged_attention_multiblock" => (
            spec(
                "paged_attention_multiblock",
                "paged_attn_multiblock_fwd",
                &[("desc_q", "*fp16"), ("desc_kt", "*fp16"), ("desc_v", "*fp16"),
                  ("desc_table", "*i32"), ("desc_o", "*fp16")],
                &[("NUM_BLOCKS", Val::Int(2)), ("BLOCK_N", Val::Int(64)),
                  ("HEAD_DIM", Val::Int(64)), ("V", Val::Int(1024)),
                  ("BLOCK_M", Val::Int(64))],
            ),
            vec![1],
            Program::Matmul,
        ),
        _ => return None,
    })
}

/// Attention's buffers for the CAUSAL configuration. `desc_mask` is delta 10 — the PREPARED
/// additive causal mask, because there is no lane-wise integer compare on this device.
const ATTN: &[(&str, &str)] = &[
    ("desc_q", "*fp16"),
    ("desc_k", "*fp16"),
    ("desc_v", "*fp16"),
    ("desc_o", "*fp16"),
    ("desc_mask", "*fp16"),
];

/// Attention's buffers for the NON-CAUSAL configuration — the same four, WITHOUT the mask, because
/// STAGE 1 never loads it and an unaddressed parameter is a handoff refusal (see the case row).
const ATTN_NONCAUSAL: &[(&str, &str)] = &[
    ("desc_q", "*fp16"),
    ("desc_k", "*fp16"),
    ("desc_v", "*fp16"),
    ("desc_o", "*fp16"),
];

/// Attention's constexprs at `index.json`'s geometry, with `STAGE` the only difference between the
/// two configurations.
fn attn_ce(stage: i128) -> Vec<(&'static str, Val)> {
    vec![
        ("sm_scale", Val::Float(1.0)),
        ("Z", Val::Int(1)),
        ("H", Val::Int(4)),
        ("N_CTX", Val::Int(256)),
        ("HEAD_DIM", Val::Int(128)),
        ("BLOCK_M", Val::Int(64)),
        ("BLOCK_N", Val::Int(64)),
        ("GQA", Val::Int(2)),
        ("STAGE", Val::Int(stage)),
    ]
}

const SWIGLU: &[(&str, &str)] = &[
    ("desc_x", "*fp16"),
    ("desc_wg", "*fp16"),
    ("desc_wu", "*fp16"),
    ("desc_wd", "*fp16"),
    ("desc_o", "*fp16"),
];

/// SwiGLU's constexprs. `block_n`/`block_k` are explicit because they are the knob under test.
fn swiglu_ce(d_model: i128, d_ff: i128, block_n: i128, block_k: i128) -> Vec<(&'static str, Val)> {
    swiglu_ce_m(64, d_model, d_ff, block_n, block_k)
}

/// [`swiglu_ce`] with the row count stated too -- the debug twin's knob for staying under the
/// matmul util floor (`M × n_dev × k ≥ 2²⁰` bumps a narrow output's sticks).
fn swiglu_ce_m(
    m: i128,
    d_model: i128,
    d_ff: i128,
    block_n: i128,
    block_k: i128,
) -> Vec<(&'static str, Val)> {
    vec![
        ("M", Val::Int(m)),
        ("D_MODEL", Val::Int(d_model)),
        ("D_FF", Val::Int(d_ff)),
        ("BLOCK_M", Val::Int(m)),
        ("BLOCK_N", Val::Int(block_n)),
        ("BLOCK_K", Val::Int(block_k)),
    ]
}

const ROPE: &[(&str, &str)] = &[
    ("desc_x", "*fp16"),
    ("desc_cos", "*fp16"),
    ("desc_sin", "*fp16"),
    ("desc_o", "*fp16"),
];

/// RoPE's constexprs at head count `h` -- Granite-3.3 8B's head_dim 128, so `HALF` is one f16
/// stick.
///
/// NO `BLOCK_M`, and its absence is the token-major kernel's contract rather than an omission:
/// a work item is ONE POSITION and its tile is `[H, HALF]`, so there is no row-block size left to
/// choose. `rope_fwd`'s parameter list is `H, N_TOK, HEAD_DIM, HALF` and nothing else. An extra
/// constexpr would be silently ignored by the front end, which is exactly why it is not passed --
/// a spec that states a knob the kernel does not have is a spec a reader cannot trust.
fn rope_ce(h: i128) -> Vec<(&'static str, Val)> {
    vec![
        ("H", Val::Int(h)),
        ("N_TOK", Val::Int(256)),
        ("HEAD_DIM", Val::Int(128)),
        ("HALF", Val::Int(64)),
    ]
}

const EMBEDDING: &[(&str, &str)] =
    &[("desc_ids", "*i32"), ("desc_table", "*fp16"), ("desc_o", "*fp16")];

/// The embedding's constexprs at token count `n_tok` and block height `block_m`. Granite's own
/// vocabulary extent (49159, NOT a multiple of 64 -- embedding.py delta 3) and hidden size, from
/// `pure_rust_ktir.rs`.
///
/// ⭐ `n_tok` IS A KNOB BECAUSE THE GATHER'S INDEX HAS A CEILING, and it is the ONE extent that moves
/// the entry count. The emitted descriptor spans the whole node, so its entries are `N_TOK` (one per
/// gathered row) whatever `BLOCK_M` is -- see the `_m32` configuration.
fn embedding_ce(n_tok: i128, block_m: i128) -> Vec<(&'static str, Val)> {
    vec![
        ("N_TOK", Val::Int(n_tok)),
        ("V", Val::Int(49159)),
        ("D_MODEL", Val::Int(4096)),
        ("BLOCK_M", Val::Int(block_m)),
        ("EMB_SCALE", Val::Float(12.0)),
    ]
}

/// The decoder block's constexprs -- `pure_rust_ktir.rs`'s `dec_ce` with BLOCK_N as the knob,
/// because BLOCK_N == D_FF is what makes the MLP's one loop single-trip.
fn dec_ce(block_n: i128) -> Vec<(&'static str, Val)> {
    vec![
        ("M", Val::Int(64)),
        ("D_MODEL", Val::Int(128)),
        ("D_FF", Val::Int(256)),
        ("BLOCK_N", Val::Int(block_n)),
        ("HALF", Val::Int(64)),
        ("EPS", Val::Float(1e-05)),
        ("INV_D", Val::Float(1.0 / 128.0)),
        ("QK_SCALE", Val::Float(0.011271055)), // 0.0078125 * 1.44269504, folded (Python's literal, bit-identical; do NOT substitute std's LOG2_E — its f64 spelling differs)
        ("RM", Val::Float(0.22)),
    ]
}

fn dec_ptrs(names: &[&'static str]) -> Vec<(&'static str, &'static str)> {
    names.iter().map(|p| (*p, "*fp16")).collect()
}

/// One decoder layer's pointer list, in `decoder_layer_fwd`'s own parameter order.
fn dec_ptrs_one() -> Vec<(&'static str, &'static str)> {
    dec_ptrs(&[
        "desc_x", "desc_o", "desc_n1", "desc_wq", "desc_wk", "desc_wv", "desc_wo", "desc_mask",
        "desc_cos", "desc_sin", "desc_n2", "desc_wg", "desc_wu", "desc_wd",
    ])
}

/// Two layers' pointer list, in `decoder_two_layers_fwd`'s own parameter order.
fn dec_ptrs_two() -> Vec<(&'static str, &'static str)> {
    dec_ptrs(&[
        "desc_x", "desc_o", "desc_n1a", "desc_wqa", "desc_wka", "desc_wva", "desc_woa", "desc_n2a",
        "desc_wga", "desc_wua", "desc_wda", "desc_n1b", "desc_wqb", "desc_wkb", "desc_wvb",
        "desc_wob", "desc_n2b", "desc_wgb", "desc_wub", "desc_wdb", "desc_mask", "desc_cos",
        "desc_sin",
    ])
}

#[cfg(test)]
mod tests {
    /// `ALL` AND `case` MUST AGREE, or a sweep silently covers less than its name. The forward
    /// direction (every name resolves) catches a typo in the list; there is no mechanical reverse
    /// direction -- `case` is a `match` and Rust cannot enumerate its arms -- so the list's
    /// completeness is asserted by COUNT, which fails the moment an arm is added without the name.
    #[test]
    fn every_name_in_all_resolves() {
        for name in super::ALL {
            assert!(super::case(name).is_some(), "`{name}` is in ALL but `case` has no arm");
        }
        assert_eq!(
            super::ALL.len(),
            25,
            "a configuration was added or removed: update ALL (and the module title) deliberately"
        );
    }

    /// A name that is NOT a configuration comes back `None` rather than resolving to a neighbour.
    /// The `_ => return None` arm is the fail-closed edge every consumer relies on to tell
    /// "no such config" from "this config refused".
    #[test]
    fn an_unknown_name_is_none() {
        assert!(super::case("rmsnorm").is_none());
        assert!(super::case("").is_none());
        assert!(super::case("decoder_layer_one_FLAT").is_none());
    }

    /// THE FIXTURE A CONFIGURATION NAMES MUST EXIST ON DISK. `spec` builds the path from
    /// `CARGO_MANIFEST_DIR` and nothing reads it until `codegen::compile` does, so a moved or
    /// renamed fixture surfaces as a compile refusal three legs downstream. Checked here instead.
    #[test]
    fn every_configuration_names_a_fixture_that_exists() {
        for name in super::ALL {
            let (spec, grid, _) = super::case(name).expect("a name from ALL");
            assert!(
                std::path::Path::new(&spec.file).is_file(),
                "`{name}` names `{}`, which is not a file",
                spec.file
            );
            assert!(!grid.is_empty(), "`{name}` has an empty grid");
            assert!(grid.iter().all(|&e| e > 0), "`{name}` has a non-positive grid extent");
        }
    }
}

// SPDX-License-Identifier: Apache-2.0

//! ⛔⛔⛔ THE W8A8 GATE/UP SLAB GEMMS AT THE WHOLE-FUNCTION DOOR — CLAIMED ONLY WHEN PROVEN.
//!
//! # WHAT THIS MEASURES
//!
//! `program_w8a8_gemm_chains` claims a fused-MoE kernel's gate/up projections —
//! ONE per-token quantize of the expert input (`abs → amax → ·1/448 → max(1/448²) →
//! min(448)` = `sx`; the codes `clip(x·(1/sx), ±448)` truncated to e4m3 and widened
//! back = `qx`), per pair a gathered fp8 slab gemm `qx @ wᵀ`, and the two-multiply
//! dequant `raw · sx · the pair's router scale` — and the walk lowers each
//! projection as ONE `expert_matmul_fused_m1`: the card's own
//! `quantscalepertokenfp8` + gathered `matmulfp8` + dequant pair, not eleven
//! pointwise ops and a PLAIN f16 matmul per pair (a different algorithm, to which
//! no derived tolerance applies — the gelu reader's own law, and the reason the
//! m=1 expert door exists at all).
//!
//! The kernel driven here is moe.py's expert half — the four helpers VERBATIM
//! (the file the splice will splice, not a paraphrase), ids as an input pointer,
//! the gelu·up products accumulated in registers, one corner-0 store — because
//! the FULL kernel's down projection (the reload of the act scratch the program
//! round-trips) is not this recogniser's to claim: its spine roots at a STORED
//! parameter, the provenance the down discriminator keys on, and moe_door's rung
//! pin owns where the full kernel's door stops. This file isolates the
//! recogniser's own contract.
//!
//! # WHY EACH CONTROL IS A CORRECTNESS FACT AND NOT A STYLE RULE
//!
//! * the POSITIVE control — the half-kernel must BAKE as exactly 80 ops: 31 for
//!   the gate projection (8 slab gathers + 8 scale gathers + the 5-op shared
//!   quantize chain + 8 `matmulfp8`s + the 2-op dequant), 26 for up (the same
//!   legs with the quantize DEDUPED — gate and up are two projections over ONE
//!   expert input, so the second emits no second chain), 16 for the eight
//!   gelu·up gated acts, and 7 for the left-deep add chain. A bake with ten
//!   pointwise ops per pair would be the recogniser not firing; one with two
//!   quantize chains would be the shared-spine law broken; one with a plain
//!   matmul would be the W8A8 numerics silently swapped for f16 — each is a
//!   different-algorithm answer, not a style regression;
//! * a planted wrong **INV_QMAX** — the amax scaling is READ off the splat and
//!   refused BY NAME: the card's `quantscalepertokenfp8` constant family is the
//!   f16 images of `1/448` and `1/448²`, and any other values are a different
//!   quantize law with the same op kinds. A recogniser that ASSUMED the
//!   constants would bake this kernel silently wrong;
//! * a planted wrong **SCALE_FLOOR** — the same law on the floor, the other
//!   subnormal constant the verification owns. ⛔ THE MUTATION MUST CLEAR THE
//!   SUBNORMAL ROUNDING BOUNDARY: `1/448²` is f16-SUBNORMAL (ulp ≈ 5.96e-8;
//!   the image is 84 ulps, the next image 85, so the boundary sits at
//!   ×1.012), and a mutant inside that rounds to the SAME image and
//!   legitimately bakes — the control here moves 5%, past the boundary;
//! * a planted **second token row** (M=2) — moe.py's per-token spelling keeps
//!   ONE `[M, H]` view of the expert input (the per-token loads are windows
//!   into it), so the first claimed chain's region walk sees a 2-row expert
//!   input and the door refuses AT THE ROW COUNT, by name: this fused W8A8
//!   is the M=1 (decode) form, and M>1 is the per-`Program` door's gathered
//!   prefill body, not the nearest thing to lower here. (The one-spine law
//!   behind it stays for a two-spine spelling; the observed M=2 form never
//!   reaches it — the row count refuses first.)
//!
//! # THE DRIVER
//!
//! Every case compiles a Triton kernel through the full ladder (`codegen::compile`
//! → `make_ttir` → `from_ttir` → `make_ktir` → `to_ktir`) and into `emit_whole`
//! with a REAL `BundleLayout` — the W8A8 walk mints the projections' output
//! intermediates, which `layout: None` cannot place (the unit-test arm's own
//! refusal), and the emission resolves every parameter and clamp constant by
//! name through the layout's placements, so the drive places them the way the
//! real bake does. The node's `program` label is INERT under this door
//! (`lower_function` derives every op's kind from its `OpKind` and never reads
//! `node.program`), so it is minted with `Program::ExpertMatmul`, the variant
//! the fused legs emit under — moe_door's own law.

use ktir_superdsc::ktir_node::{KtirNode, Program};
use ktir_superdsc::place::{PlaceId, act_name};
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
use ktir_superdsc::reserved_tids::{FP8_INV448_TID, FP8_NEG448_TID, FP8_POS448_TID};
use std::collections::HashMap;
use triton_frontend::codegen::{self, ArgSpec, KernelSpec};
use triton_frontend::opt;
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

/// The golden geometry: the ladder's own (`triton-numeric`'s `moe.rs`,
/// `triton-frontend`'s `common::moe`) — one token, 64 experts, top-8, `H=128`,
/// `I=64`. `M=1` is the door's own law; the M=2 control drives the same source.
const H: usize = 128;
const I: usize = 64;
const E: usize = 64;
const K: usize = 8;

/// The canonical half-kernel: moe.py's four helpers VERBATIM (docstrings
/// included — they state the f16-tile and island laws the recogniser's checks
/// enforce), the expert loop's own spellings (`e = tl.load(desc_ids + …)`, the
/// fp8 slab loads reshaped to `[I, H]`, the per-pair `_w8a8` at the shared
/// codes/scale), and a register accumulation in place of the act round-trip —
/// `if j == 0: acc = act` (a `static_range` induction is a compile-time
/// constant, so the branch folds and no control flow reaches the KTIR) rather
/// than a zeros-splat seed the walk would refuse as a constant input.
const SRC: &str = r#"
import triton
import triton.language as tl


@triton.jit
def _fp8_scale(amax, INV_QMAX: tl.constexpr, SCALE_FLOOR: tl.constexpr):
    """The per-token activation scale — `quantscalepertokenfp8`'s
    `FMIN(FMAX(amax · mulConst, clipMin), clipMax)` (`OpFunc::op_consts`), for one token
    row (`amax` is `[1]`). The bounds are f16 tiles from birth: `tl.minimum`/`tl.maximum`
    turn a bare Python float into an f32 tensor, which would widen the scale to f32."""
    lo = tl.full([1], SCALE_FLOOR, tl.float16)
    hi = tl.full([1], 448.0, tl.float16)
    return tl.minimum(tl.maximum(amax * INV_QMAX, lo), hi)


@triton.jit
def _fp8_codes(a, inv, N: tl.constexpr):
    """`clip(a · (1/s), ±448)` rounded to e4m3 — the card's `fq_sc`/`fq_clip`/`qfp8ch` legs,
    for one `[1, N]` row. Widened back to f16 for the dot: the codes are the values."""
    lo = tl.full([1, N], -448.0, tl.float16)
    hi = tl.full([1, N], 448.0, tl.float16)
    q = tl.minimum(tl.maximum(a * inv[:, None], lo), hi)
    return q.to(tl.float8e4nv).to(tl.float16)


@triton.jit
def _w8a8(q, s, w_codes, w_scale, ROWS: tl.constexpr, N: tl.constexpr):
    """`(q @ W.T · s) · w_scale` — the matmulfp8 and its two dequant legs."""
    raw = tl.dot(q, w_codes.T, tl.zeros([ROWS, N], dtype=tl.float16))
    return (raw * s[:, None]) * w_scale


@triton.jit
def _gelu_tanh(g, N: tl.constexpr):
    """`gatedact.py`'s gelu: the tanh form through the sigmoid identity, the cubic in the
    f32 island (an f16 x³ overflows at |x| ≥ 41). The `1` is an f16 tile, `swiglu_mlp.py`'s
    spelling: `1.0 + e` would make the divide an f16-by-f32 `fdiv`, which upstream refuses."""
    g32 = g.to(tl.float32)
    v = 1.5957691216057308 * (g32 + 0.044715 * g32 * g32 * g32)
    e = tl.exp(-v).to(tl.float16)
    return tl.fdiv(g, tl.full([1, N], 1.0, tl.float16) + e)


@triton.jit
def w8a8_fwd(desc_xe, desc_wg, desc_sg, desc_wu, desc_su,  #
             desc_o, desc_ids,  #
             M: tl.constexpr, H: tl.constexpr, I: tl.constexpr,  #
             E: tl.constexpr, K: tl.constexpr,  #
             INV_QMAX: tl.constexpr, SCALE_FLOOR: tl.constexpr):
    xe_desc = tl.make_tensor_descriptor(desc_xe, shape=[M, H], strides=[H, 1],
                                        block_shape=[1, H])
    wg_desc = tl.make_tensor_descriptor(desc_wg, shape=[E, I, H], strides=[I * H, H, 1],
                                        block_shape=[1, I, H])
    sg_desc = tl.make_tensor_descriptor(desc_sg, shape=[E, I], strides=[I, 1],
                                        block_shape=[1, I])
    wu_desc = tl.make_tensor_descriptor(desc_wu, shape=[E, I, H], strides=[I * H, H, 1],
                                        block_shape=[1, I, H])
    su_desc = tl.make_tensor_descriptor(desc_su, shape=[E, I], strides=[I, 1],
                                        block_shape=[1, I])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, I], strides=[I, 1],
                                       block_shape=[1, I])
    for m in tl.static_range(0, M):
        # The gate/up activation: the token's input, one scale for all K pairs.
        xe = xe_desc.load([m, 0])
        sx = _fp8_scale(tl.max(tl.abs(xe), 1).to(tl.float16), INV_QMAX, SCALE_FLOOR)
        qx = _fp8_codes(xe, tl.fdiv(tl.full([1], 1.0, tl.float16), sx), H)
        for j in tl.static_range(0, K):
            e = tl.load(desc_ids + (m * K + j))
            wg = tl.reshape(wg_desc.load([e, 0, 0]), [I, H]).to(tl.float16)
            wu = tl.reshape(wu_desc.load([e, 0, 0]), [I, H]).to(tl.float16)
            gate = _w8a8(qx, sx, wg, sg_desc.load([e, 0]), 1, I)
            up = _w8a8(qx, sx, wu, su_desc.load([e, 0]), 1, I)
            act = _gelu_tanh(gate, I) * up
            if j == 0:
                acc = act
            else:
                acc = acc + act
        o_desc.store([m, 0], acc)
"#;

/// A layout that places what the REAL bake places for this kernel: the seven
/// parameters, at their true footprints and distinct offsets in the Activation
/// segment, and the fp8 quantize chain's three clamp constants
/// (`lower_subtile_tape_to_superdsc`'s own reserved-constant insert, which
/// fires whenever a matmul-fp8 is emitted). Everything else the door touches —
/// the projections' output intermediates, the packed-weight and quantize
/// scratch, the gelu's minted rows — is minted through `BundleLayout::synth`
/// by the walk and the emission themselves, which is the law the
/// undeclared-synthetic guard states: a name with no placement and no synth
/// entry is a build error, not a buffer to serve.
fn layout_for(node: &KtirNode, m: u64) -> BundleLayout {
    let mut l = BundleLayout::default();
    // Parameter bytes by binding order — xe [m,H] f16, wg/wu [E,I,H] fp8,
    // sg/su [E,I] f16, o [m,I] f16, ids [m,K] i32.
    let bytes = [
        m * H as u64 * 2,
        (E * I * H) as u64,
        (E * I * 2) as u64,
        (E * I * H) as u64,
        (E * I * 2) as u64,
        m * I as u64 * 2,
        m * K as u64 * 4,
    ];
    for (i, b) in node.bindings.iter().enumerate() {
        let tid = b.get();
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        // Distinct 1 MiB offsets: the fp8 slabs are 512 KiB each, so a tighter
        // spacing would overlap them.
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: 3,
                bank: 0,
                offset: i as u64 * 0x10_0000,
                size: bytes[i],
            },
        );
    }
    // The clamp constants: one [1, 64] f16 stick each, past the parameters.
    for (i, tid) in [FP8_POS448_TID, FP8_NEG448_TID, FP8_INV448_TID]
        .into_iter()
        .enumerate()
    {
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: 3,
                bank: 0,
                offset: 0x70_0000 + i as u64 * 0x1000,
                size: 128,
            },
        );
    }
    l
}

/// Compile the half-kernel through the whole ladder and into `emit_whole` — the
/// door the recognisers live behind — at the given token count and quantize
/// constants. `Err` carries the first stage's refusal text, so a ladder-stage
/// break and a door refusal are distinguishable in the assertion message.
fn drive_src(m: i128, inv_qmax: f64, scale_floor: f64) -> Result<Vec<String>, String> {
    let mut signature = HashMap::new();
    for (k, v) in [
        ("desc_xe", "*fp16"),
        ("desc_wg", "*fp8e4nv"),
        ("desc_sg", "*fp16"),
        ("desc_wu", "*fp8e4nv"),
        ("desc_su", "*fp16"),
        ("desc_o", "*fp16"),
        ("desc_ids", "*i32"),
    ] {
        signature.insert(k.into(), ArgSpec::parse(v).expect("pointer spec"));
    }
    let mut constexprs = HashMap::new();
    for (k, v) in [
        ("M", Val::Int(m)),
        ("H", Val::Int(H as i128)),
        ("I", Val::Int(I as i128)),
        ("E", Val::Int(E as i128)),
        ("K", Val::Int(K as i128)),
        ("INV_QMAX", Val::Float(inv_qmax)),
        ("SCALE_FLOOR", Val::Float(scale_floor)),
    ] {
        signature.insert(k.into(), ArgSpec::parse("constexpr").expect("constexpr"));
        constexprs.insert(k.into(), v);
    }
    let kspec = KernelSpec {
        kernel: "w8a8_fwd".into(),
        signature,
        constexprs,
        file: "w8a8_door.py".into(),
    };
    let mut tt =
        codegen::compile(SRC, &kspec, Target::spyre()).map_err(|e| format!("codegen: {e}"))?;
    opt::make_ttir(&mut tt).map_err(|e| format!("make_ttir: {e}"))?;
    let mut mm = triton_ktir::from_ttir::convert(&tt).map_err(|e| format!("from_ttir: {e}"))?;
    triton_ktir::make_ktir(&mut mm, &[1]).map_err(|e| format!("make_ktir: {e}"))?;
    triton_ktir::passes::to_ktir::run(&mut mm, &[1]).map_err(|e| format!("to_ktir: {e}"))?;
    let node = triton_ktir_superdsc::node_for(&mm, Program::ExpertMatmul)
        .map_err(|e| format!("handoff: {}", e.message))?;
    // A REAL layout: the W8A8 walk mints the projections' output intermediates,
    // which `layout: None` (the unit-test arm) cannot place — the mint's own law.
    let layout = layout_for(&node, m as u64);
    let ops = triton_ktir_superdsc::emit_whole(&node, Some(&layout))
        .map_err(|e| format!("door: {}", e.message))?;
    Ok(ops.into_iter().map(|o| o.op_name.clone()).collect())
}

/// The canonical drive: one token, the card's own constants.
fn drive() -> Result<Vec<String>, String> {
    drive_src(1, 1.0 / 448.0, 1.0 / 448.0 / 448.0)
}

/// THE POSITIVE CONTROL: the half-kernel bakes, and the op list IS the two fused
/// projections plus their consumers — 80 ops, counted by name. Every count is a
/// correctness fact: the gathers and `matmulfp8`s are the W8A8 numerics (a plain
/// matmul in their place is a different algorithm), the ONE `_fq_` quantize chain
/// is the shared-spine law (two chains would quantize the same row twice at two
/// scales), the gated acts are the gelu rung's own two-op form, and the seven
/// adds are the register accumulation the act round-trip was replaced by.
#[test]
fn the_half_kernel_bakes_as_two_fused_projections_and_their_consumers() {
    let names = drive().expect("the canonical half-kernel bakes");
    assert_eq!(
        names.len(),
        80,
        "31 gate + 26 up + 16 gated acts + 7 adds; got: {names:?}"
    );
    let count = |p: &str| names.iter().filter(|n| n.starts_with(p)).count();
    // The two projections' legs, 8 pairs each: slab gathers, scale gathers,
    // fp8 gemms — and the dequant pair per projection.
    assert_eq!(count("emwg_p"), 16, "slab gathers: {names:?}");
    assert_eq!(count("emsg_p"), 16, "scale gathers: {names:?}");
    assert_eq!(count("emmm_p"), 16, "matmulfp8s: {names:?}");
    assert_eq!(count("ems_o"), 2, "dequant scale legs: {names:?}");
    assert_eq!(count("emo_o"), 2, "dequant out legs: {names:?}");
    // ⭐ THE SHARED QUANTIZE, exactly once: gate and up are two projections over
    // ONE expert input, so the second emission's `quantized` dedup skips its
    // chain — 5 ops (`fq_qspt`/`fq_invs`/`fq_sc`/`fq_clip`/`fq_afp8`), not 10.
    assert_eq!(
        names.iter().filter(|n| n.contains("_fq_")).count(),
        5,
        "ONE shared quantize chain, not one per projection: {names:?}"
    );
    // The consumers: eight gelu·up gated acts (the gelu rung's own two-op form)
    // and the seven adds that accumulate them.
    assert_eq!(count("gact_o"), 8, "gated act legs: {names:?}");
    assert_eq!(count("mulgact_o"), 8, "gated act multiplies: {names:?}");
    assert_eq!(count("add_o"), 7, "the register accumulation: {names:?}");
    // ORDER: the gate projection leads (its pair-0 terminal is the first chain
    // terminal in program order, where the walk lowers the whole projection),
    // and the stored final add ends it.
    assert!(
        names[0].starts_with("emwg_p0_"),
        "the gate projection's first slab gather leads: {names:?}"
    );
    assert!(
        names.last().is_some_and(|n| n.starts_with("add_o")),
        "the stored accumulation ends it: {names:?}"
    );
    // The silu body's names must NOT appear — the gelu routes through
    // `gated_act`, and a drift onto `silumul` would emit the silu primitive for
    // a gelu (the exact misrecognition the door's fail-closed laws exist for).
    assert!(
        !names.iter().any(|n| n.contains("silu")),
        "`silu_o`/`mulsilu_o` are the silu body's names and must not be emitted here: {names:?}"
    );
}

/// A PLANTED WRONG INV_QMAX — the amax scaling mutated one part in ~450. The
/// f16 grid rounds it to a DIFFERENT image (checked: `1/447` and `1/448` sit two
/// mantissa steps apart), so the recogniser must refuse it BY NAME, naming the
/// constant family — the check that proves the splat was READ rather than
/// assumed (an assumed constant would bake this kernel silently wrong).
#[test]
fn a_planted_wrong_inv_qmax_is_refused_naming_the_constant_family() {
    let err = match drive_src(1, 1.0 / 447.0, 1.0 / 448.0 / 448.0) {
        Ok(names) => panic!(
            "BAKED ({} ops) — a W8A8 spine with a mutated INV_QMAX lowered through the \
             door; the constant check did not read the splat",
            names.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.starts_with("door:")
            && err.contains("the W8A8 quantize spine's amax scaling (1/448)")
            && err.contains("refused rather than fused"),
        "refused, but not for the reason this test pins. Expected the door to name the \
         amax scaling's constant family and refuse, got: {err:?}"
    );
}

/// A PLANTED WRONG SCALE_FLOOR — the floor moved 5%, past the subnormal
/// rounding boundary. `1/448²` is f16-SUBNORMAL (ulp ≈ 5.96e-8): its image is
/// 84 ulps and the next image is 85, so the boundary sits at ×1.012 — a ×1.01
/// mutant rounds to the SAME image and legitimately bakes (the first draft of
/// this control moved 1% and did exactly that). Same law, same message shape:
/// the floor is the other subnormal constant the verification owns.
#[test]
fn a_planted_wrong_scale_floor_is_refused_naming_the_constant_family() {
    let err = match drive_src(1, 1.0 / 448.0, 1.0 / 448.0 / 448.0 * 1.05) {
        Ok(names) => panic!(
            "BAKED ({} ops) — a W8A8 spine with a mutated SCALE_FLOOR lowered through \
             the door; the constant check did not read the splat",
            names.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.starts_with("door:")
            && err.contains("the W8A8 quantize spine's scale floor (1/448²)")
            && err.contains("refused rather than fused"),
        "refused, but not for the reason this test pins. Expected the door to name the \
         scale floor's constant family and refuse, got: {err:?}"
    );
}

/// A PLANTED SECOND TOKEN ROW — at M=2 the per-token descriptor loads are
/// windows into ONE `[M, H]` view of the expert input, so the first claimed
/// chain's region walk sees a 2-row input and the door refuses AT THE ROW
/// COUNT, by name: this fused W8A8 is the M=1 (decode) form, and M>1 is the
/// per-`Program` door's gathered prefill body, not the nearest thing to lower
/// here. (The one-spine law behind it stays for a two-spine spelling; the
/// observed M=2 form never reaches it — the row count refuses first.)
#[test]
fn a_second_token_row_is_refused_naming_the_decode_form_law() {
    let err = match drive_src(2, 1.0 / 448.0, 1.0 / 448.0 / 448.0) {
        Ok(names) => panic!(
            "BAKED ({} ops) — a two-token W8A8 half-kernel lowered through the door; \
             the M=1 law did not fire",
            names.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.starts_with("door:")
            && err.contains("is 2 rows wide")
            && err.contains("M=1 (decode) form"),
        "refused, but not for the reason this test pins. Expected the expert input's row \
         count to refuse M=2 as off-form, got: {err:?}"
    );
}

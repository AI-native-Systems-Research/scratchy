// SPDX-License-Identifier: Apache-2.0

//! ⛔⛔⛔ THE W8A8 DOWN SLAB GEMMS AT THE WHOLE-FUNCTION DOOR — CLAIMED ONLY WHEN
//! PROVEN, EMITTED ONLY AT THEIR OWN POSITION.
//!
//! # WHAT THIS MEASURES
//!
//! `program_w8a8_down_chains` claims a fused-MoE kernel's DOWN projections —
//! each pair's OWN quantize of its OWN round-tripped act window (`gelu(gate)·up`
//! stored to the act scratch at window `j·I` and reloaded by pair `j`'s spine:
//! `abs → amax → ·1/448 → max(1/448²) → min(448)` = `sa_j`; the codes
//! `clip(a·(1/sa_j), ±448)` truncated to e4m3 and widened back), the gathered
//! fp8 slab gemm `qa_j @ wdᵀ`, and the two-multiply dequant — and the walk
//! lowers EACH PAIR's ten-leg set at THAT PAIR'S OWN TERMINAL through
//! `expert_down_fused_m1`: the card's own `quantscalepertokenfp8` + gathered
//! `matmulfp8` + dequant pair per pair, not eleven pointwise ops and a PLAIN
//! f16 matmul per pair (a different algorithm, to which no derived tolerance
//! applies).
//!
//! # WHY PER-POSITION, AND WHY THE ORDER IS A CORRECTNESS FACT
//!
//! Program order is tape order is execution order, and the unrolled kernel
//! interleaves `gelu_j → down_j → combine_j` per pair. A single fire at the
//! FIRST terminal would read act windows whose gelu writers have not run
//! (pairs 1..k−1 — garbage on card, not a slower answer) or write out windows
//! the combine has already read; the shared clip/convert has no legal position
//! at all. So the positive control asserts the interleaving PER PAIR: pair
//! `j`'s quantize (`emqs_p{j}`) starts after exactly `j` gelu chains and `j`
//! combine multiplies, and between pair `j`'s dequant (`emdo_p{j}`) and pair
//! `j+1`'s quantize sit exactly pair `j`'s combine legs and pair `j+1`'s gelu.
//! A single-fire bake would put all eighty down legs before the second gelu —
//! the exact defect the per-position law exists to prevent, and observable in
//! nothing but the order.
//!
//! The kernel driven here is moe.py's down half — the four helpers VERBATIM
//! (the file the splice will splice, not a paraphrase), the act round-trip
//! VERBATIM (`act_desc.store([m, j*I], act)` then `act_desc.load([m, j*I])`),
//! ids as an input pointer — with two substitutions the door's own 1:1 law
//! forces: the gate/up producers are PLAIN loads (this file measures the DOWN
//! reader in isolation; the gate/up projections are w8a8_door's own contract),
//! and the consumer is a per-pair weighted store of each pair's down row (the
//! walk's chunk law gives one 1:1 op ONE column window shared by every
//! operand, and the down terminal is a window) rather than moe.py's one-hot
//! `tl.where(koffs == j, wrow, zeros)` extraction and running sum — `tl.where`
//! has no 1:1 `Program` and the sum crosses windows, so a positive control
//! carrying the real combine would refuse at the combine, not measure the
//! down door. The real combine is the combine recognizer's future rung;
//! moe_door's rung pin owns where the full kernel's door stops.
//!
//! # WHY EACH CONTROL IS A CORRECTNESS FACT AND NOT A STYLE RULE
//!
//! * the POSITIVE control — the half-kernel must BAKE as exactly 104 ops: 16
//!   for the eight gelu·up gated acts (the gelu rung's own two-op form), 80
//!   for the eight down pairs (8 each of the ten legs: `emwd`/`emsd` the
//!   gathers, `emqs`/`emiv`/`emsc`/`emcl`/`emcv` the pair's OWN quantize —
//!   NOT one shared chain, the per-pair scale law — `emmm` the fp8 gemm,
//!   `emds`/`emdo` the dequant), and 8 for the substituted consumer (each
//!   pair's weighted down row, stored to its own window). A bake with ten
//!   pointwise ops per pair would be the recogniser not firing; one with ONE
//!   shared quantize chain would be the per-pair scale law broken — pair
//!   `p`'s scale quantizing pair `q`'s activation;
//! * a planted wrong **INV_QMAX** — the amax scaling is READ off the splat and
//!   refused BY NAME: the card's `quantscalepertokenfp8` constant family is
//!   the f16 images of `1/448` and `1/448²`, and any other values are a
//!   different quantize law with the same op kinds. A recogniser that ASSUMED
//!   the constants would bake this kernel silently wrong;
//! * a planted wrong **SCALE_FLOOR** — the same law on the floor, the other
//!   subnormal constant the verification owns. ⛔ THE MUTATION MUST CLEAR THE
//!   SUBNORMAL ROUNDING BOUNDARY: `1/448²` is f16-SUBNORMAL (ulp ≈ 5.96e-8;
//!   the image is 84 ulps, the next image 85, so the boundary sits at
//!   ×1.012), and a mutant inside that rounds to the SAME image and
//!   legitimately bakes — the control here moves 5%, past the boundary;
//! * a planted **second token row** (M=2) — the act scratch is ONE `[M, K·I]`
//!   view (the per-pair windows are tiles into it), so the first claimed
//!   chain's region walk sees a 2-row act buffer and the door refuses AT THE
//!   ROW COUNT, by name: this fused W8A8 is the M=1 (decode) form, and M>1 is
//!   the per-`Program` door's gathered prefill body, not the nearest thing to
//!   lower here;
//! * a planted **half-stick `I`** (I=64, H=128) — the RECOGNITION passes (the
//!   per-pair windows are whole fp16 sticks) and the EMISSION refuses at the
//!   staged-bank law: `fp8_k_pad(64) = 128 ≠ 64`, the padded K rows are the
//!   per-`Program` door's worker-manifest machinery, and no whole-function
//!   program states that bank. This is the 26b's real DOWN geometry shape
//!   (`I=704 → pad 768`) at a testable size — the address-arithmetic arm.
//!
//! # THE DRIVER
//!
//! Every case compiles a Triton kernel through the full ladder (`codegen::compile`
//! → `make_ttir` → `from_ttir` → `make_ktir` → `to_ktir`) and into `emit_whole`
//! with a REAL `BundleLayout` — the down walk mints the projection's output
//! intermediate, which `layout: None` cannot place (the unit-test arm's own
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

/// The golden geometry: one token, 64 experts, top-8, `H=128` (the projection's
/// `n`), `I=128` (the gated act's width, the contraction's `inn`). `I=128` is
/// the ALIGNED arm — `fp8_k_pad(128) = 128`, the padded layout IS the logical
/// one — so the door bakes; the I=64 control drives the half-stick arm. `M=1`
/// is the door's own law; the M=2 control drives the same source.
const H: usize = 128;
const E: usize = 64;
const K: usize = 8;

/// The canonical half-kernel: moe.py's four helpers VERBATIM (docstrings
/// included — they state the f16-tile and island laws the recogniser's checks
/// enforce), the down loop's own spellings (the act round-trip
/// `store([m, j*I]) → load([m, j*I])`, `sa = _fp8_scale(tl.max(tl.abs(a), 1)…)`
/// per pair, the fp8 slab load reshaped to `[H, I]`, the per-pair `_w8a8` at
/// that pair's OWN codes/scale), and the substituted consumer — each pair's
/// down row weighted by its own `[1, H]` weight-row window and stored to its
/// own window of the output. That store-at-the-pair's-own-window form is the
/// one consumer the 1:1 walk can express: its chunk law gives one op ONE
/// column window shared by every operand, and the down terminal IS a window
/// (pair `j`'s `[1, n]` row of the projection's output at column `j·n`, with
/// `n == H` so the weight row's `j·H` window is the same column). The REAL
/// combine — moe.py's one-hot `tl.where(koffs == j, wrow, zeros)` extraction
/// and its running sum — has no 1:1 `Program` and sums across windows, so it
/// is the combine recognizer's future rung, not this file's; moe_door's rung
/// pin owns where the full kernel's door stops.
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
def down_fwd(desc_gate, desc_up, desc_act, desc_wd, desc_sd,  #
             desc_o, desc_ids, desc_w,  #
             M: tl.constexpr, H: tl.constexpr, I: tl.constexpr,  #
             E: tl.constexpr, K: tl.constexpr,  #
             INV_QMAX: tl.constexpr, SCALE_FLOOR: tl.constexpr):
    gate_desc = tl.make_tensor_descriptor(desc_gate, shape=[1, M * K * I], strides=[M * K * I, 1],
                                          block_shape=[1, I])
    up_desc = tl.make_tensor_descriptor(desc_up, shape=[1, M * K * I], strides=[M * K * I, 1],
                                        block_shape=[1, I])
    act_desc = tl.make_tensor_descriptor(desc_act, shape=[M, K * I], strides=[K * I, 1],
                                         block_shape=[1, I])
    wd_desc = tl.make_tensor_descriptor(desc_wd, shape=[E, H, I], strides=[H * I, I, 1],
                                        block_shape=[1, H, I])
    sd_desc = tl.make_tensor_descriptor(desc_sd, shape=[E, H], strides=[H, 1],
                                        block_shape=[1, H])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[1, K * H], strides=[K * H, 1],
                                       block_shape=[1, H])
    w_desc = tl.make_tensor_descriptor(desc_w, shape=[1, K * H], strides=[K * H, 1],
                                       block_shape=[1, H])
    for m in tl.static_range(0, M):
        for j in tl.static_range(0, K):
            e = tl.load(desc_ids + (m * K + j))
            # The gated act, round-tripped through the act scratch: pair j's
            # window [m, j·I) is written here and reloaded below — the
            # provenance the down reader's spine keys on.
            gate = gate_desc.load([0, (m * K + j) * I])
            up = up_desc.load([0, (m * K + j) * I])
            act = _gelu_tanh(gate, I) * up
            act_desc.store([m, j * I], act)
            a = act_desc.load([m, j * I])
            sa = _fp8_scale(tl.max(tl.abs(a), 1).to(tl.float16), INV_QMAX, SCALE_FLOOR)
            qa = _fp8_codes(a, tl.fdiv(tl.full([1], 1.0, tl.float16), sa), I)
            wd = tl.reshape(wd_desc.load([e, 0, 0]), [H, I]).to(tl.float16)
            down = _w8a8(qa, sa, wd, sd_desc.load([e, 0]), 1, H)
            wj = w_desc.load([0, j * H])
            term = down * wj
            o_desc.store([0, j * H], term)
"#;

/// A layout that places what the REAL bake places for this kernel: the eight
/// parameters, at their true footprints and distinct offsets in the Activation
/// segment, and the fp8 quantize chain's three clamp constants
/// (`lower_subtile_tape_to_superdsc`'s own reserved-constant insert, which
/// fires whenever a matmul-fp8 is emitted). Everything else the door touches —
/// the projection's output intermediate, the packed-weight and quantize
/// scratch, the per-pair scale sticks — is minted through `BundleLayout::synth`
/// by the walk and the emission themselves, which is the law the
/// undeclared-synthetic guard states: a name with no placement and no synth
/// entry is a build error, not a buffer to serve.
fn layout_for(node: &KtirNode, m: u64, i: u64) -> BundleLayout {
    let mut l = BundleLayout::default();
    // Parameter bytes by binding order — gate/up [1, M·K·I] f16, act [M, K·I]
    // f16, wd [E, H, I] fp8, sd [E, H] f16, o [1, K·H] f16, ids [M·K] i32,
    // w [1, K·H] f16.
    let bytes = [
        m * K as u64 * i * 2,
        m * K as u64 * i * 2,
        m * K as u64 * i * 2,
        (E as u64 * H as u64 * i),
        (E * H * 2) as u64,
        (K * H * 2) as u64,
        m * K as u64 * 4,
        (K * H * 2) as u64,
    ];
    let mut slot: u64 = bytes.iter().copied().max().unwrap_or(0).next_power_of_two();
    slot = slot.max(0x20_0000);
    for (i, b) in node.bindings.iter().enumerate() {
        let tid = b.get();
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        // Distinct slots, sized to the widest parameter: 2 MiB at the golden
        // geometry (the fp8 slab is 1 MiB there), 8 MiB at the 26b's I=704
        // (the slab is 5.5 MiB) — a fixed 2 MiB would overlap it.
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: 3,
                bank: 0,
                offset: i as u64 * slot,
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
                offset: 8 * slot + i as u64 * 0x1000,
                size: 128,
            },
        );
    }
    l
}

/// Compile the half-kernel through the whole ladder and into `emit_whole` — the
/// door the recognisers live behind — at the given token count, gated-act
/// width, and quantize constants. `Err` carries the first stage's refusal
/// text, so a ladder-stage break and a door refusal are distinguishable in the
/// assertion message.
fn drive_src(m: i128, i: i128, inv_qmax: f64, scale_floor: f64) -> Result<Vec<String>, String> {
    let mut signature = HashMap::new();
    for (k, v) in [
        ("desc_gate", "*fp16"),
        ("desc_up", "*fp16"),
        ("desc_act", "*fp16"),
        ("desc_wd", "*fp8e4nv"),
        ("desc_sd", "*fp16"),
        ("desc_o", "*fp16"),
        ("desc_ids", "*i32"),
        ("desc_w", "*fp16"),
    ] {
        signature.insert(k.into(), ArgSpec::parse(v).expect("pointer spec"));
    }
    let mut constexprs = HashMap::new();
    for (k, v) in [
        ("M", Val::Int(m)),
        ("H", Val::Int(H as i128)),
        ("I", Val::Int(i)),
        ("E", Val::Int(E as i128)),
        ("K", Val::Int(K as i128)),
        ("INV_QMAX", Val::Float(inv_qmax)),
        ("SCALE_FLOOR", Val::Float(scale_floor)),
    ] {
        signature.insert(k.into(), ArgSpec::parse("constexpr").expect("constexpr"));
        constexprs.insert(k.into(), v);
    }
    let kspec = KernelSpec {
        kernel: "down_fwd".into(),
        signature,
        constexprs,
        file: "down_door.py".into(),
    };
    let mut tt =
        codegen::compile(SRC, &kspec, Target::spyre()).map_err(|e| format!("codegen: {e}"))?;
    opt::make_ttir(&mut tt).map_err(|e| format!("make_ttir: {e}"))?;
    let mut mm = triton_ktir::from_ttir::convert(&tt).map_err(|e| format!("from_ttir: {e}"))?;
    triton_ktir::make_ktir(&mut mm, &[1]).map_err(|e| format!("make_ktir: {e}"))?;
    triton_ktir::passes::to_ktir::run(&mut mm, &[1]).map_err(|e| format!("to_ktir: {e}"))?;
    let node = triton_ktir_superdsc::node_for(&mm, Program::ExpertMatmul)
        .map_err(|e| format!("handoff: {}", e.message))?;
    // A REAL layout: the down walk mints the projection's output intermediate,
    // which `layout: None` (the unit-test arm) cannot place — the mint's own law.
    let layout = layout_for(&node, m as u64, i as u64);
    let ops = triton_ktir_superdsc::emit_whole(&node, Some(&layout))
        .map_err(|e| format!("door: {}", e.message))?;
    Ok(ops.into_iter().map(|o| o.op_name.clone()).collect())
}

/// The canonical drive: one token, the golden width, the card's own constants.
fn drive() -> Result<Vec<String>, String> {
    drive_src(1, 128, 1.0 / 448.0, 1.0 / 448.0 / 448.0)
}

/// THE POSITIVE CONTROL: the half-kernel bakes, and the op list IS the eight
/// per-position down pairs plus their producers and consumers — 104 ops,
/// counted by name. Every count is a correctness fact: the gathers and
/// `matmulfp8`s are the W8A8 numerics (a plain matmul in their place is a
/// different algorithm), the EIGHT quantize chains are the per-pair scale law
/// (one shared chain would quantize every pair's window at pair 0's scale),
/// the gated acts are the gelu rung's own two-op form, and the eight stored
/// weighted rows are the substituted consumer.
#[test]
fn the_half_kernel_bakes_as_eight_per_position_down_pairs() {
    let names = drive().expect("the canonical half-kernel bakes");
    assert_eq!(
        names.len(),
        104,
        "16 gated acts + 80 down legs + 8 stored weighted rows; got: {names:?}"
    );
    let count = |p: &str| names.iter().filter(|n| n.starts_with(p)).count();
    // The eight pairs' ten legs each — the per-pair form: ONE quantize chain
    // per pair (emqs/emiv/emsc/emcl/emcv), not one shared chain.
    for leg in [
        "emwd_p", "emsd_p", "emqs_p", "emiv_p", "emsc_p", "emcl_p", "emcv_p", "emmm_p", "emds_p",
        "emdo_p",
    ] {
        assert_eq!(count(leg), 8, "{leg} legs, one per pair: {names:?}");
    }
    // The shared-spine names must NOT appear — the down form has no shared
    // quantize, and a `_fq_` chain here would be the gate/up reader's emission
    // (or a second, wrong, dedup) quantizing every pair at one scale.
    assert!(
        !names.iter().any(|n| n.contains("_fq_")),
        "the down form states one whole quantize per pair; a shared `_fq_` chain would \
         quantize every pair's window at one scale: {names:?}"
    );
    // The producers: eight gelu·up gated acts, each stored to its own act
    // window (the round trip the down reader's spine roots at).
    assert_eq!(count("gact_o"), 8, "gated act legs: {names:?}");
    assert_eq!(count("mulgact_o"), 8, "gated act multiplies: {names:?}");
    // The consumers: the substituted form — eight weighted rows, each stored
    // to its own window of the output. `Elementwise(Mul)` lowers through the
    // `multiply` op func, so the name is `multiply_o{tid}`.
    assert_eq!(count("multiply_o"), 8, "weighted-row multiplies: {names:?}");
    assert_eq!(
        count("add_o"),
        0,
        "the real combine's running sum is a future rung: {names:?}"
    );
    // ORDER: the program's first compute op is pair 0's gelu, and pair 7's
    // stored weighted row ends it.
    assert!(
        names[0].starts_with("gact_o"),
        "pair 0's gated act leads: {names:?}"
    );
    assert!(
        names.last().is_some_and(|n| n.starts_with("multiply_o")),
        "the last pair's stored weighted row ends it: {names:?}"
    );
    // ⭐⭐⭐ THE PER-POSITION LAW, asserted per pair. Pair j's quantize starts
    // after its own gelu and every earlier pair's (2(j+1) legs) and exactly j
    // weighted-row multiplies — pair j's own row lands after its dequant;
    // between pair j's dequant and pair j+1's quantize sit exactly pair j's
    // weighted row (one mul) and pair j+1's gelu (two legs). A single-fire
    // bake — all eighty down legs at pair 0's terminal — fails every one of
    // these: the whole projection would precede the second gelu.
    let pos = |p: &str| {
        names
            .iter()
            .position(|n| n.starts_with(p))
            .unwrap_or_else(|| {
                panic!("no {p} op in the bake: {names:?}");
            })
    };
    for j in 0..K {
        let qs = pos(&format!("emqs_p{j}_"));
        let gelu_before = names[..qs]
            .iter()
            .filter(|n| n.starts_with("gact_o") || n.starts_with("mulgact_o"))
            .count();
        assert_eq!(
            gelu_before,
            2 * (j + 1),
            "pair {j}'s quantize starts after its own gelu and every earlier pair's — the \
             per-position law: {names:?}"
        );
        let mul_before = names[..qs]
            .iter()
            .filter(|n| n.starts_with("multiply_o"))
            .count();
        assert_eq!(
            mul_before, j,
            "pair {j}'s quantize starts after exactly {j} weighted-row multiplies: {names:?}"
        );
        let do_pos = pos(&format!("emdo_p{j}_"));
        if j + 1 < K {
            let j1 = j + 1;
            let next_qs = pos(&format!("emqs_p{j1}_"));
            assert!(
                qs < do_pos && do_pos < next_qs,
                "pair {j}'s legs are emitted at pair {j}'s own position: {names:?}"
            );
            let between = &names[do_pos + 1..next_qs];
            assert_eq!(
                between.len(),
                5,
                "between pair {j}'s dequant and pair {}'s quantize sit exactly pair {j}'s \
                 weighted row, pair {}'s gelu, and pair {}'s weight and scale gathers: {names:?}",
                j + 1,
                j + 1,
                j + 1
            );
            assert_eq!(
                between
                    .iter()
                    .filter(|n| n.starts_with("gact_o") || n.starts_with("mulgact_o"))
                    .count(),
                2,
                "pair {}'s gelu sits between the pairs' down legs: {names:?}",
                j + 1
            );
            assert_eq!(
                between
                    .iter()
                    .filter(|n| n.starts_with("emwd_p") || n.starts_with("emsd_p"))
                    .count(),
                2,
                "pair {}'s weight and scale gathers sit after its gelu, before its quantize: \
                 {names:?}",
                j + 1
            );
            assert_eq!(
                between
                    .iter()
                    .filter(|n| n.starts_with("multiply_o"))
                    .count(),
                1,
                "pair {j}'s weighted row sits after its dequant: {names:?}"
            );
        }
    }
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
    let err = match drive_src(1, 128, 1.0 / 447.0, 1.0 / 448.0 / 448.0) {
        Ok(names) => panic!(
            "BAKED ({} ops) — a W8A8 DOWN spine with a mutated INV_QMAX lowered through the \
             door; the constant check did not read the splat",
            names.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.starts_with("door:")
            && err.contains("the W8A8 DOWN quantize spine's amax scaling (1/448)")
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
    let err = match drive_src(1, 128, 1.0 / 448.0, 1.0 / 448.0 / 448.0 * 1.05) {
        Ok(names) => panic!(
            "BAKED ({} ops) — a W8A8 DOWN spine with a mutated SCALE_FLOOR lowered through \
             the door; the constant check did not read the splat",
            names.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.starts_with("door:")
            && err.contains("the W8A8 DOWN quantize spine's scale floor (1/448²)")
            && err.contains("refused rather than fused"),
        "refused, but not for the reason this test pins. Expected the door to name the \
         scale floor's constant family and refuse, got: {err:?}"
    );
}

/// A PLANTED SECOND TOKEN ROW — at M=2 the act scratch is ONE `[M, K·I]` view
/// (the per-pair windows are tiles into it), so the first claimed chain's
/// region walk sees a 2-row act buffer and the door refuses AT THE ROW COUNT,
/// by name: this fused W8A8 is the M=1 (decode) form, and M>1 is the
/// per-`Program` door's gathered prefill body, not the nearest thing to lower
/// here.
#[test]
fn a_second_token_row_is_refused_naming_the_decode_form_law() {
    let err = match drive_src(2, 128, 1.0 / 448.0, 1.0 / 448.0 / 448.0) {
        Ok(names) => panic!(
            "BAKED ({} ops) — a two-token W8A8 down half-kernel lowered through the door; \
             the M=1 law did not fire",
            names.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.starts_with("door:")
            && err.contains("is 2 rows wide")
            && err.contains("M=1 (decode) form"),
        "refused, but not for the reason this test pins. Expected the act buffer's row \
         count to refuse M=2 as off-form, got: {err:?}"
    );
}

/// A PLANTED HALF-STICK WIDTH — at I=64 (H=128) the RECOGNITION passes: the
/// per-pair act windows are whole fp16 sticks, the anchors are the whole
/// top-k, the round trip is proven. The EMISSION refuses at the staged-bank
/// law: `fp8_k_pad(64) = 128 ≠ 64` — the padded K rows are the per-`Program`
/// door's worker-manifest machinery, and no whole-function program states
/// that bank. This is the 26b's real DOWN geometry (`I=704 → pad 768`) at a
/// testable size: the address arithmetic the per-pair windows would need, and
/// the refusal that fires before any of it is emitted.
#[test]
fn a_half_stick_width_is_refused_at_the_staged_bank_law() {
    let err = match drive_src(1, 64, 1.0 / 448.0, 1.0 / 448.0 / 448.0) {
        Ok(names) => panic!(
            "BAKED ({} ops) — a half-stick-width W8A8 down half-kernel lowered through the \
             door; the staged-bank law did not fire",
            names.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.starts_with("door:")
            && err.contains("in=64 pads to 128 fp8 elements")
            && err.contains("staged-bank law"),
        "refused, but not for the reason this test pins. Expected the emission to name the \
         fp8 K-pad and refuse at the staged-bank law, got: {err:?}"
    );
}

/// THE 26B'S OWN WIDTH — `I=704` is the model's real DOWN gated-act width, and
/// this arm runs the address arithmetic at that exact size: the act windows
/// sit at corners `704·j` of a `[1, 5632]` row, the fp8 slab is 5.5 MiB (the
/// layout widens its slots for it), and `fp8_k_pad(704) = 768 ≠ 704`. The
/// staged-bank refusal is the same law the half-stick arm pins, stated at the
/// width the shipped model actually presents — it fires before any of the ten
/// per-pair legs is emitted.
#[test]
fn the_26b_width_704_is_refused_at_the_staged_bank_law() {
    let err = match drive_src(1, 704, 1.0 / 448.0, 1.0 / 448.0 / 448.0) {
        Ok(names) => panic!(
            "BAKED ({} ops) — the 26b's 704-wide W8A8 down half-kernel lowered through the \
             door; the staged-bank law did not fire at the model's own width",
            names.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.starts_with("door:")
            && err.contains("in=704 pads to 768 fp8 elements")
            && err.contains("staged-bank law"),
        "refused, but not for the reason this test pins. Expected the emission to name the \
         26b's fp8 K-pad (704 → 768) and refuse at the staged-bank law, got: {err:?}"
    );
}

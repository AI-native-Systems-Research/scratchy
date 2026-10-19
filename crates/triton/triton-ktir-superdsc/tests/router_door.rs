// SPDX-License-Identifier: Apache-2.0

//! ⛔⛔⛔ THE ROUTER AT THE WHOLE-FUNCTION DOOR — RUNG FIVE OF THE RECOGNISER LADDER.
//!
//! # WHAT THIS DRIVES
//!
//! `kernels/moe.py`'s ROUTER SECTION, verbatim — the rmsnorm over the un-normed residual,
//! the `[E, H]` projection dot, the K-round top-k (`max → lane-select → argmax-reduce →
//! one-hot accumulate → mask-out`), the temperature softmax, and the two scratch stores —
//! COMPOSED WITH `w8a8_door.rs`'s expert half (that file's four helpers and m-loop,
//! themselves moe.py's own): the full fused kernel minus the down projection, the act
//! round-trip, and the combine. The full kernel's door pin (`moe_door.rs`) stops at the
//! SCRATCH CENSUS's refusal of the combine's read — the census runs before the walk — so
//! the walk past the router's fire point has never run there; this file is the rung's own
//! contract, driven where nothing else refuses first.
//!
//! # WHY THE EXPERT HALF IS THERE — THE LADDER'S DOT-COUNT LAW
//!
//! The router-only kernel this rung wants is NOT spellable through the ladder:
//! `triton-ktir`'s `spyre-dot-to-linalg` discriminates on DOT COUNT. Exactly ONE `tt.dot`
//! puts the whole kernel under the canonical single-dot contract ("A and B must be direct
//! `tt.descriptor_load` results" — the f16 arm's own refusal), and the router's A is the
//! norm's computed output, so the ladder RED-stops the router-only form at `make_ktir`,
//! before the door ever sees it. MORE than one dot and each dot is lowered locally,
//! untagged — the law the full fused kernel itself rides (its router dot sits among
//! 8×2 expert gemms and 8 downs). The honest minimal multi-dot form that still drives
//! the router rung is therefore the router prefixed to the expert half: 17 dots, both
//! sections moe.py itself splices, every rung the campaign has landed claimed (the router
//! chain, the top-k, the W8A8 gate/up projections, the gated act), and — the reason this
//! composition is a rung of its own — the FIRST drive where the router's minted flat
//! index actually feeds the expert gathers: `route_topk` mints the `XFl/XF32/XIdx` trio
//! keyed off the ids parameter's tid, and the m=1 expert door reads the door-minted
//! `XIdx` through the same PlaceId (the one-landing law both doors state — the
//! reservation is idempotent, so the two doors share ONE landing with zero drift).
//!
//! # WHY EACH CONTROL IS A CORRECTNESS FACT AND NOT A STYLE RULE
//!
//! * the POSITIVE control — the composed kernel must BAKE as BOTH sections' own forms at
//!   once: the router's four per-`Program` emissions (`RouteTopK`, `ScalarMul`,
//!   `RouteSoftmax`, `RouteExpertScale`) chained over the walk's five minted products,
//!   AND the expert half's own census (`w8a8_door.rs`'s: the slab and scale gathers, the
//!   `matmulfp8`s, the ONE shared quantize, the gated acts, the register accumulation) —
//!   neither section's op set disturbed by the other's presence. A bake with pointwise
//!   ops per round would be the recogniser not firing; one with a second top-k or a
//!   second quantize chain would be the shared-scratch law broken;
//! * the NO-UNREAD-OUTPUT law — every buffer an emitted op writes, some emitted op reads,
//!   with exactly two exemptions, both reads that live OUTSIDE this kernel: the
//!   route-weights product (the combine rung reads it) and the final accumulated act
//!   (the walk FOLDS the terminal `o` store into the last add — its output lands AT the
//!   output parameter, the buffer the CALLER reads). The flat index convert's `XIdx` is
//!   NOT exempt: the expert gathers read it, the handoff this drive is the first to
//!   exercise. A product nothing reads is a chain that fired without a consumer — the
//!   recognition campaign's own definition of an unsound claim;
//! * the WRITE-DEAD law, made mechanical — the two scratch parameters appear in NO
//!   emitted binding: the program's stores into them lower to nothing, the expert doors
//!   read the door-minted index through its own synth landing (never the parameter), and
//!   the census (the pre-read `moe_door.rs` pins on the full kernel) proves every read of
//!   them is a claimed chain's;
//! * a planted **K′ ≠ K** — the k-offset generate widened past the round count (16 lanes,
//!   8 rounds, a 16-column landing): the walk's three-way k cross-check refuses BY NAME,
//!   because a walk that proved a different selection than the program stored is a
//!   recognition, not a lowering;
//! * a planted **zero lane mask** — the lane select's mask-out moved off −1.0: the
//!   tie-break law refuses it BY NAME (`reduce_max` over `select(cmp, offs, −1)` takes
//!   the LAST lane of the maximum; any other mask-out selects a different expert on
//!   ties). A recogniser that ASSUMED the mask would bake this kernel silently wrong;
//! * a planted **unregistered temperature** — the softmax's temperature moved off
//!   `H^-0.5` while the layout's registry keeps the canonical value: the `scale_slot`
//!   law refuses the registry desync BY NAME, because the descriptor multiplies by a
//!   bound `[1,1]` const and the value the program uses must have a slot;
//! * a planted **unfolded round 0** — the three accumulators seeded at 1.0 instead of
//!   zeros, so round 0's updates keep their adds and the lockstep walk descends past
//!   round 0: the round count refuses the ninth round against the landing's eight slots
//!   BY NAME — the round form is `acc + x·onehot` with round 0 FOLDED to `x·onehot`, and
//!   the descent that stops at a fold is the form being read;
//! * a planted **unclaimed scratch reader** — a copy of the route-weights scratch
//!   through a fresh parameter, spelled exactly as the combine's read (a direct load
//!   feeding a lane select): the scratch census refuses it BY NAME, the same refusal the
//!   full kernel's door pin holds the combine rung at.
//!
//! # THE DRIVER
//!
//! Every case compiles a Triton kernel through the full ladder (`codegen::compile` →
//! `make_ttir` → `from_ttir` → `make_ktir` → `to_ktir`) and into `emit_whole` with a REAL
//! `BundleLayout`: the five products are minted intermediates, which `layout: None`
//! cannot place. The layout places the twelve parameters at their true byte sizes (the
//! router's four, the expert half's six, the two scratch — moe.py's own order minus the
//! down and act parameters), the four router consts and the three fp8 clamp consts the
//! emissions read by reserved tid, and BOTH `scalarmul_scales` slots (the norm's epsilon
//! and the softmax's temperature — the registry law). The node's `program` label is INERT
//! under this door (`lower_function` derives every op's kind from its `OpKind` and never
//! reads `node.program`), so it is minted with `Program::ExpertMatmul`, the variant the
//! fused legs emit under — moe_door's own law.

use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::ktir_node::{KtirNode, Program};
use ktir_superdsc::place::{PlaceId, act_name};
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
use ktir_superdsc::reserved_tids::{
    FP8_INV448_TID, FP8_NEG448_TID, FP8_POS448_TID, router_identity_tid, router_sm_mask_tid,
    router_topk_iota_tid, router_topk_pad_tid, scalarmul_scale_tid,
};
use std::collections::HashMap;
use triton_frontend::codegen::{self, ArgSpec, KernelSpec};
use triton_frontend::opt;
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

/// The golden geometry: the ladder's own (`triton-numeric`'s `moe.rs`,
/// `triton-frontend`'s `common::moe`) — one token, 64 experts, top-8, `H=128`,
/// `I=64`. `M=1` is the door's own law (the m>1 forms are the per-`Program` door's).
const M: usize = 1;
const H: usize = 128;
const I: usize = 64;
const E: usize = 64;
const K: usize = 8;

/// The softmax temperature, `H^-0.5` — the value the registry slot holds in every drive
/// (the unregistered-temperature mutant moves the KERNEL's splat, never the registry).
const TEMP: f64 = 0.08838834764831845;

/// The twelve pointers, in signature order — moe.py's own order with the down and act
/// parameters deleted (the sections this kernel does not carry).
const PTRS: &[(&str, &str)] = &[
    ("desc_xr", "*fp16"),
    ("desc_rg", "*fp16"),
    ("desc_rw", "*fp16"),
    ("desc_pes", "*fp16"),
    ("desc_xe", "*fp16"),
    ("desc_wg", "*fp8e4nv"),
    ("desc_sg", "*fp16"),
    ("desc_wu", "*fp8e4nv"),
    ("desc_su", "*fp16"),
    ("desc_o", "*fp16"),
    ("desc_ids", "*i32"),
    ("desc_rs", "*fp16"),
];

/// The composed kernel — moe.py's ROUTER SECTION VERBATIM (the norm, the dot, the
/// round-form top-k, the temperature softmax, the two stores), then `w8a8_door.rs`'s
/// EXPERT HALF VERBATIM (the four helpers and the m-loop that file drives, themselves
/// moe.py's own spellings): the full fused kernel with the down projection, the act
/// round-trip, and the combine deleted. The helpers' and sections' comments are the
/// source files' own — they state the f16-tile laws the recognisers' checks enforce.
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
def router_fwd(desc_xr, desc_rg, desc_rw, desc_pes,  #
               desc_xe, desc_wg, desc_sg, desc_wu, desc_su,  #
               desc_o,  #
               desc_ids, desc_rs,  #
               M: tl.constexpr, H: tl.constexpr, I: tl.constexpr,  #
               E: tl.constexpr, K: tl.constexpr,  #
               EPS: tl.constexpr, INV_H: tl.constexpr, TEMP: tl.constexpr,  #
               INV_QMAX: tl.constexpr, SCALE_FLOOR: tl.constexpr):
    # ── the router (moe.py's section, verbatim) ────────────────────────────────
    xr_desc = tl.make_tensor_descriptor(desc_xr, shape=[M, H], strides=[H, 1],
                                        block_shape=[M, H])
    rg_desc = tl.make_tensor_descriptor(desc_rg, shape=[H], strides=[1], block_shape=[H])
    rw_desc = tl.make_tensor_descriptor(desc_rw, shape=[E, H], strides=[H, 1],
                                        block_shape=[E, H])
    pes_desc = tl.make_tensor_descriptor(desc_pes, shape=[E], strides=[1], block_shape=[E])
    xr = xr_desc.load([0, 0])
    ms = tl.sum(xr * xr, 1) * INV_H
    r = tl.rsqrt((ms + EPS).to(tl.float32)).to(tl.float16)
    xn = xr * r[:, None] * rg_desc.load([0])[None, :]
    lg = tl.dot(xn, rw_desc.load([0, 0]).T, tl.zeros([M, E], dtype=tl.float16))

    # THE TOP-K, K rounds over the whole `[M, E]` row: each round takes the row maximum
    # and the HIGHEST expert id among the lanes equal to it, then masks that lane out.
    # This is `topk.ddl`'s ripple order exactly — descending by value, and among equal
    # values the later scan position first (the ripple's strict `>` lets the newcomer take
    # the slot). The lane ids ride f16 (exact to 2048; the real router's `E = 128`).
    offs = tl.arange(0, E).to(tl.float16)
    koffs = tl.arange(0, K).to(tl.float16)
    pes = pes_desc.load([0])
    vals = tl.zeros([M, K], dtype=tl.float16)
    idsf = tl.zeros([M, K], dtype=tl.float16)
    psel = tl.zeros([M, K], dtype=tl.float16)
    # Every select is tile-on-tile f16 (`routetopk.py`'s law): a bare Python float in a
    # `tl.where` becomes an f32 tensor and widens the whole select.
    neg_inf = tl.full([M, E], float("-inf"), tl.float16)
    no_lane = tl.full([M, E], -1.0, tl.float16)
    zero_e = tl.zeros([M, E], dtype=tl.float16)
    one_k = tl.full([K], 1.0, tl.float16)
    zero_k = tl.zeros([K], dtype=tl.float16)
    for j in tl.static_range(0, K):
        # `tl.max` reduces an f16 tile in f32 (upstream's own promotion); each reduce is
        # narrowed straight back, the widening island LegalizeTypes collapses.
        mx = tl.max(lg, 1).to(tl.float16)
        lanes = tl.where(lg == mx[:, None], offs[None, :] + zero_e, no_lane)
        idx = tl.max(lanes, 1).to(tl.float16)
        sel = offs[None, :] == idx[:, None]
        # The selected expert's per-expert scale, by its id (a one-hot over the E lanes).
        pj = tl.sum(tl.where(sel, pes[None, :] + zero_e, zero_e), 1)
        # Slot j of the three `[M, K]` results: each slot is written exactly once, so
        # accumulating a one-hot column into zeros places it exactly. The one-hot is a
        # select of f16 tiles, not a bool cast (`.to` of an i1 is `arith.uitofp`).
        oh = tl.where(koffs == j, one_k, zero_k)[None, :]
        vals = vals + mx[:, None] * oh
        idsf = idsf + idx[:, None] * oh
        psel = psel + pj[:, None] * oh
        lg = tl.where(sel, neg_inf, lg)

    # The route weights: the K selected logits at the `H^-0.5` temperature, a softmax over
    # the K (`routesoftmax.py`'s chain), × each selected expert's scale. The ids land as i32:
    # the slab gather reads them back through a pointer (its index view).
    ids_desc = tl.make_tensor_descriptor(desc_ids, shape=[M, K], strides=[K, 1],
                                         block_shape=[M, K])
    rs_desc = tl.make_tensor_descriptor(desc_rs, shape=[M, K], strides=[K, 1],
                                        block_shape=[M, K])
    ids_desc.store([0, 0], idsf.to(tl.int32))
    sv = (vals * TEMP).to(tl.float32)
    ex = tl.exp(sv - tl.max(sv, 1)[:, None])
    rs_desc.store([0, 0], tl.fdiv(ex, tl.sum(ex, 1)[:, None]).to(tl.float16) * psel)

    # ── the experts (`w8a8_door.rs`'s half, verbatim: moe.py's loop minus the down
    # projection, the act round-trip, and the combine) ─────────────────────────
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

/// A mutant source: `SRC` with each anchor replaced, refusing loudly if an anchor
/// drifted (a silent miss would test nothing).
fn src_with(repls: &[(&str, &str)]) -> String {
    let mut s = SRC.to_string();
    for (from, to) in repls {
        assert!(s.contains(from), "a mutant anchor drifted: {from:?}");
        s = s.replace(from, to);
    }
    s
}

/// The kernel spec at the given temperature, extra pointers (the leak mutant's), and
/// extra constexprs (the K′ mutant's `KW`).
fn spec(temp: f64, extra_ptrs: &[(&str, &str)], extra_ces: &[(&str, Val)]) -> KernelSpec {
    let mut signature = HashMap::new();
    for (k, v) in PTRS.iter().chain(extra_ptrs) {
        signature.insert(k.to_string(), ArgSpec::parse(v).expect("pointer spec"));
    }
    let mut constexprs = HashMap::new();
    let ces: Vec<(&str, Val)> = [
        ("M", Val::Int(M as i128)),
        ("H", Val::Int(H as i128)),
        ("I", Val::Int(I as i128)),
        ("E", Val::Int(E as i128)),
        ("K", Val::Int(K as i128)),
        ("EPS", Val::Float(1e-6)),
        ("INV_H", Val::Float(1.0 / H as f64)),
        ("TEMP", Val::Float(temp)),
        ("INV_QMAX", Val::Float(1.0 / 448.0)),
        ("SCALE_FLOOR", Val::Float(1.0 / 448.0 / 448.0)),
    ]
    .into_iter()
    .chain(extra_ces.iter().cloned())
    .collect();
    for (k, v) in ces {
        signature.insert(
            k.to_string(),
            ArgSpec::parse("constexpr").expect("constexpr"),
        );
        constexprs.insert(k.to_string(), v);
    }
    KernelSpec {
        kernel: "router_fwd".to_string(),
        signature,
        constexprs,
        file: "router_door.py".to_string(),
    }
}

/// The whole ladder, value-level: `.py` → TTIR → KTIR → the node the door takes. `Err`
/// carries the first stage's refusal text, so a ladder-stage break and a door refusal
/// are distinguishable in the assertion message.
fn router_node(
    src: &str,
    temp: f64,
    extra_ptrs: &[(&str, &str)],
    extra_ces: &[(&str, Val)],
) -> Result<KtirNode, String> {
    let mut tt = codegen::compile(src, &spec(temp, extra_ptrs, extra_ces), Target::spyre())
        .map_err(|e| format!("codegen: {e}"))?;
    opt::make_ttir(&mut tt).map_err(|e| format!("make_ttir: {e}"))?;
    let mut m = triton_ktir::from_ttir::convert(&tt).map_err(|e| format!("from_ttir: {e}"))?;
    triton_ktir::make_ktir(&mut m, &[1]).map_err(|e| format!("make_ktir: {e}"))?;
    triton_ktir::passes::to_ktir::run(&mut m, &[1]).map_err(|e| format!("to_ktir: {e}"))?;
    triton_ktir_superdsc::node_for(&m, Program::ExpertMatmul)
        .map_err(|e| format!("handoff: {}", e.message))
}

/// The parameters' true bytes in binding order — moe.py's own footprints with the down
/// and act parameters deleted. `k_store` is the width the two scratch descriptors state
/// (K for every kernel but the K′ mutant, whose landing is KW-wide). Extra pointers
/// size themselves the same way and land AFTER the twelve (the leak mutant threads its
/// parameter past the scratch line, last in the signature).
fn drive_bytes(k_store: u64, extra_ptrs: &[(&str, &str)]) -> Vec<u64> {
    let mut b = vec![
        (M * H * 2) as u64,     // desc_xr  [M, H] f16
        (H * 2) as u64,         // desc_rg  [H] f16
        (E * H * 2) as u64,     // desc_rw  [E, H] f16
        (E * 2) as u64,         // desc_pes [E] f16
        (M * H * 2) as u64,     // desc_xe  [M, H] f16
        (E * I * H) as u64,     // desc_wg  [E, I, H] fp8 — one byte per element
        (E * I * 2) as u64,     // desc_sg  [E, I] f16
        (E * I * H) as u64,     // desc_wu  [E, I, H] fp8
        (E * I * 2) as u64,     // desc_su  [E, I] f16
        (M * I * 2) as u64,     // desc_o   [M, I] f16
        M as u64 * k_store * 4, // desc_ids [M, K] i32
        M as u64 * k_store * 2, // desc_rs  [M, K] f16
    ];
    for (name, _) in extra_ptrs {
        match *name {
            "desc_leak" => b.push(M as u64 * k_store * 2),
            _ => panic!("the router door places an unknown pointer: {name}"),
        }
    }
    b
}

/// A REAL layout: every parameter at its true byte size (bindings enumerate the
/// signature's pointers in order, so the bytes ride positionally), the four router
/// consts and the three fp8 clamp consts the emissions read by reserved tid (the
/// sanitize mask `[1, 64]`, the iota row `[1, 64]`, the pad row `[1, 64]` — unread at
/// `W == E` but placed as every other layout places it — the identity table `[64, 64]`,
/// and the quantize chain's `[1, 64]` clamps), and BOTH registry slots
/// (`scalarmul_scales` holds the norm's epsilon and the CANONICAL temperature — the
/// unregistered-temperature mutant moves the kernel's splat, never this registry).
/// Everything else the door touches is minted through `BundleLayout::synth` by the walk
/// and the emissions themselves.
fn router_layout(node: &KtirNode, bytes: &[u64]) -> BundleLayout {
    assert_eq!(
        node.bindings.len(),
        bytes.len(),
        "one byte size per parameter binding"
    );
    let mut l = BundleLayout::default();
    for (i, b) in node.bindings.iter().enumerate() {
        let tid = b.get();
        // Distinct 2 MiB slots — the widest parameter (the 512 KiB fp8 slabs) fits with
        // room to spare.
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: SegRole::Activation.segment(),
                bank: 0,
                offset: i as u64 * 0x20_0000,
                size: bytes[i],
            },
        );
    }
    // The reserved consts, past the twelve parameters: the identity table is `[64, 64]`
    // f16 (8 KiB), the router rows and the fp8 clamps one stick each.
    let base = 12u64 * 0x20_0000;
    for (i, (tid, size)) in [
        (router_sm_mask_tid(), 128u64),
        (router_topk_iota_tid(), 128),
        (router_topk_pad_tid(), 128),
        (router_identity_tid(), 8192),
        (FP8_POS448_TID, 128),
        (FP8_NEG448_TID, 128),
        (FP8_INV448_TID, 128),
    ]
    .into_iter()
    .enumerate()
    {
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: SegRole::Activation.segment(),
                bank: 0,
                offset: base + i as u64 * 0x4000,
                size,
            },
        );
    }
    // The registry slots: the epsilon and the temperature, one `[1,1]` f16 each.
    l.scalarmul_scales = vec![1e-6, TEMP as f32];
    for i in 0..2usize {
        let tid = scalarmul_scale_tid(i);
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: SegRole::Activation.segment(),
                bank: 0,
                offset: base + 7 * 0x4000 + i as u64 * 0x1000,
                size: 2,
            },
        );
    }
    l
}

/// Any drive: source, temperature, the scratch descriptors' width, extra pointers and
/// constexprs. `Err` carries the first stage's refusal text.
fn drive_src(
    src: &str,
    temp: f64,
    k_store: u64,
    extra_ptrs: &[(&str, &str)],
    extra_ces: &[(&str, Val)],
) -> Result<Vec<EmittedOp>, String> {
    let node = router_node(src, temp, extra_ptrs, extra_ces)?;
    let layout = router_layout(&node, &drive_bytes(k_store, extra_ptrs));
    triton_ktir_superdsc::emit_whole(&node, Some(&layout))
        .map_err(|e| format!("door: {}", e.message))
}

/// THE POSITIVE CONTROL: the composed kernel bakes as BOTH sections' own forms — the
/// router's four per-`Program` emissions chained over the walk's five minted products,
/// and the expert half's `w8a8_door.rs` census — every product is read (the two
/// exemptions are reads that live outside this kernel), and the two scratch parameters
/// appear in no binding at all — the write-dead law, made mechanical.
#[test]
fn the_composed_kernel_bakes_as_both_sections_own_forms() {
    let node = router_node(SRC, TEMP, &[], &[]).expect("the ladder drive is green");
    let layout = router_layout(&node, &drive_bytes(K as u64, &[]));
    let ops = triton_ktir_superdsc::emit_whole(&node, Some(&layout))
        .map_err(|e| format!("door: {}", e.message))
        .expect("the composed kernel bakes");
    let names: Vec<&str> = ops.iter().map(|o| o.op_name.as_str()).collect();
    let count = |p: &str| names.iter().filter(|n| n.starts_with(p)).count();
    // THE WALK HEAD — the rmsnorm rung's own five-op form over the un-normed residual
    // (`rmmean`/`rmeps`/`rmrsqrt`/`rmxn`/`rmg`) and the projection dot as the one
    // generic `matmul`: the router chain roots at the logits, so these lower as the
    // walked-generic head, claimed by no fused emission.
    for leg in ["rmmean_", "rmeps_", "rmrsqrt_", "rmxn_", "rmg_"] {
        assert_eq!(count(leg), 1, "one {leg} leg of the router norm: {names:?}");
    }
    assert_eq!(count("matmul_"), 1, "one projection matmul: {names:?}");
    // 150 ops: 6 head + 9 top-k legs + 55 router emission + 80 expert half.
    assert_eq!(
        names.len(),
        150,
        "6 head + 9 top-k + 55 router + 80 experts; got: {names:?}"
    );
    // The top-k's own legs — route_topk's, keyed off the ids parameter: the stage (one
    // group at W=64), the Kᵀ restickify, the index/value pair, the flat convert trio,
    // and the two landing transposes. ONE of each: the shared-scratch law (a second
    // top-k would key a second scratch).
    for leg in [
        "rtkstg_g", "rtkkt_", "rtkix_", "rtkvl_", "rtkfl_", "rtkxf_", "rtkxi_", "rtkti_", "rtktv_",
    ] {
        assert_eq!(count(leg), 1, "one {leg} leg: {names:?}");
    }
    // The temperature multiply, the sanitize, and the row softmax's five-op chain —
    // `scalarmul_scaled`'s and `route_softmax`'s own forms.
    assert_eq!(count("rttemp_"), 1, "one temperature leg: {names:?}");
    assert_eq!(count("rssan_"), 1, "one sanitize leg: {names:?}");
    for leg in ["rsmax_", "rssub_", "rsexp_", "rssum_", "rsdiv_"] {
        assert_eq!(count(leg), 1, "one {leg} leg of the row softmax: {names:?}");
    }
    // The per-expert scale chain — `route_expert_scale`'s own form: eight rounds of the
    // five compare/select/reduce legs, the j=0 seed, and the seven adds that follow.
    for leg in ["remtc_", "resel_", "reprd_", "reval_", "reout_"] {
        assert_eq!(count(leg), K, "eight {leg} legs, one per round: {names:?}");
    }
    assert_eq!(count("reseed_"), 1, "one accumulator seed: {names:?}");
    assert_eq!(count("reacc_"), K - 1, "seven accumulating adds: {names:?}");
    // The expert half's legs — `w8a8_door.rs`'s own census, undisturbed by the router's
    // presence: the slab and scale gathers, the `matmulfp8`s and their dequant pairs,
    // the ONE shared quantize, the gated acts, and the register accumulation.
    assert_eq!(count("emwg_p"), 16, "slab gathers: {names:?}");
    assert_eq!(count("emsg_p"), 16, "scale gathers: {names:?}");
    assert_eq!(count("emmm_p"), 16, "matmulfp8s: {names:?}");
    assert_eq!(count("ems_o"), 2, "dequant scale legs: {names:?}");
    assert_eq!(count("emo_o"), 2, "dequant out legs: {names:?}");
    assert_eq!(
        names.iter().filter(|n| n.contains("_fq_")).count(),
        5,
        "ONE shared quantize chain, not one per projection: {names:?}"
    );
    assert_eq!(count("gact_o"), 8, "gated act legs: {names:?}");
    assert_eq!(count("mulgact_o"), 8, "gated act multiplies: {names:?}");
    assert_eq!(count("add_o"), 7, "the register accumulation: {names:?}");
    // ORDER: the norm's first leg leads (the router section precedes the experts in
    // program order) and the final register accumulate ends it — the walk FOLDS the
    // terminal `o` store into that last add, whose output lands AT the output
    // parameter itself (`add_o9` — binding 9 is `desc_o`), so no store op exists.
    assert!(
        names.first().is_some_and(|n| n.starts_with("rmmean_")),
        "the router norm's first leg leads: {names:?}"
    );
    assert!(
        names.last().is_some_and(|n| n.starts_with("add_o")),
        "the folded final accumulation ends it: {names:?}"
    );
    // The silu body's names must NOT appear — the gelu routes through `gated_act`, and
    // a drift onto `silumul` would emit the silu primitive for a gelu (the exact
    // misrecognition the door's fail-closed laws exist for).
    assert!(
        !names.iter().any(|n| n.contains("silu")),
        "`silu_o`/`mulsilu_o` are activation-body names and must not be emitted here: {names:?}"
    );
    // ⭐ THE NO-UNREAD-OUTPUT LAW: every buffer an emitted op writes, some emitted op
    // reads — with exactly two exemptions, both reads that live OUTSIDE this kernel: the
    // route-weights product (the combine rung's read) and the final accumulated act
    // (the walk folds the terminal `o` store into the last add, so its output IS the
    // output parameter — the buffer the CALLER reads). The flat index convert's `XIdx`
    // is NOT exempt: the expert gathers read it, the handoff this drive is the first to
    // exercise. A product nothing reads is a chain that fired without a consumer — an
    // unsound claim.
    let output_of = |prefix: &str| {
        ops.iter()
            .find(|o| o.op_name.starts_with(prefix))
            .unwrap_or_else(|| panic!("no {prefix} op in the bake"))
            .arg_bindings
            .iter()
            .find(|b| !b.is_input)
            .unwrap_or_else(|| panic!("the {prefix} op states no output"))
            .buffer
            .clone()
    };
    let final_output_of = |prefix: &str| {
        ops.iter()
            .rev()
            .find(|o| o.op_name.starts_with(prefix))
            .unwrap_or_else(|| panic!("no {prefix} op in the bake"))
            .arg_bindings
            .iter()
            .find(|b| !b.is_input)
            .unwrap_or_else(|| panic!("the {prefix} op states no output"))
            .buffer
            .clone()
    };
    let exempt = [
        output_of("reacc_j7_"),   // the route weights — the combine rung reads them
        final_output_of("add_o"), // the final act, AT the o parameter — the caller reads it
    ];
    let read: std::collections::HashSet<&str> = ops
        .iter()
        .flat_map(|o| o.arg_bindings.iter().filter(|b| b.is_input))
        .map(|b| b.buffer.as_str())
        .collect();
    for o in &ops {
        for b in &o.arg_bindings {
            if !b.is_input {
                assert!(
                    read.contains(b.buffer.as_str()) || exempt.contains(&b.buffer),
                    "{} wrote {}, which no emitted op reads — a product with no consumer",
                    o.op_name,
                    b.buffer
                );
            }
        }
    }
    // ⭐ THE WRITE-DEAD LAW, MECHANICAL: the two scratch parameters appear in NO
    // binding — the program's stores into them lowered to nothing, the expert doors
    // read the door-minted index through its own synth landing, and nothing the
    // emission reads addresses the parameters themselves.
    let idx_of = |n: &str| PTRS.iter().position(|(p, _)| *p == n).unwrap();
    let scratch = [
        act_name(
            node.bindings
                .get(idx_of("desc_ids"))
                .map(|b| b.get())
                .unwrap(),
        ),
        act_name(
            node.bindings
                .get(idx_of("desc_rs"))
                .map(|b| b.get())
                .unwrap(),
        ),
    ];
    for o in &ops {
        for b in &o.arg_bindings {
            assert!(
                !scratch.contains(&b.buffer),
                "{} binds the write-dead scratch {} — the fused router must never touch it",
                o.op_name,
                b.buffer
            );
        }
    }
}

/// ⭐ TOP-1: at k=1 the seed's product IS the route weights — the chain writes its
/// output at the weights-out mint, and no accumulating add follows. The unfixed
/// chain seeded the accumulator and never wrote the output at all (top-1 routing
/// is a real geometry — Llama-4-class and Switch-style MoEs — and 1 divides the
/// landing's 64 slots, so the shape is spellable). The seed's name carries the
/// weights-out tid (`reseed_o{tid}`); its output binding must BE that tensor, not
/// the accumulator synth.
#[test]
fn the_top_1_route_writes_its_weights_at_the_output() {
    let ops = drive_src(SRC, TEMP, 1, &[], &[("K", Val::Int(1))]).expect("the top-1 kernel bakes");
    let names: Vec<&str> = ops.iter().map(|o| o.op_name.as_str()).collect();
    assert_eq!(
        names.iter().filter(|n| n.starts_with("reseed_")).count(),
        1,
        "one seed leg: {names:?}"
    );
    assert!(
        !names.iter().any(|n| n.starts_with("reacc_")),
        "no accumulating adds at top-1: {names:?}"
    );
    let seed = ops
        .iter()
        .find(|o| o.op_name.starts_with("reseed_"))
        .expect("the seed leg");
    let tid: u32 = seed
        .op_name
        .strip_prefix("reseed_o")
        .and_then(|s| s.parse().ok())
        .expect("the seed's name carries the weights-out tid");
    let out_binding = seed
        .arg_bindings
        .iter()
        .find(|b| !b.is_input)
        .expect("the seed states an output");
    assert_eq!(
        out_binding.buffer,
        act_name(tid),
        "the top-1 seed writes the weights-out mint itself, not the accumulator synth"
    );
}

/// The layout-less arm — the unit-test arm's own limit: the walk's first intermediate
/// mint — the walk head's leading `ArithMulf` (the norm's `xr · xr`) — refuses without
/// a `BundleLayout` to hold its buffer, before any fused emission fires. Pinned so the
/// arm's refusal stays the MINT's and never silently becomes a walk refusal.
#[test]
fn the_router_kernel_without_a_layout_refuses_at_the_first_mint() {
    let err = match router_node(SRC, TEMP, &[], &[]) {
        Err(stage) => stage,
        Ok(node) => match triton_ktir_superdsc::emit_whole(&node, None) {
            Ok(ops) => panic!(
                "BAKED ({} ops) — the composed kernel lowered with no layout; an \
                 intermediate mint refused nothing",
                ops.len()
            ),
            Err(e) => format!("door: {}", e.message),
        },
    };
    assert!(
        err.starts_with("door:"),
        "the refusal came from a ladder stage, not the door: {err:?}"
    );
    assert!(
        err.contains("`ArithMulf`'s result is an intermediate")
            && err.contains("which needs a `BundleLayout` to hold its buffer")
            && err.contains("cannot place one"),
        "the layout-less arm's first refusal is not the walk head's first mint: {err:?}"
    );
}

/// A PLANTED K′ ≠ K — the k-offset generate widened to 16 lanes while the loop runs
/// its 8 rounds and the landing states 16 columns. The walk's k is cross-checked THREE
/// ways (the accumulation walk, the k-offset generate's width, the ids landing's own
/// columns); a walk that proved a different selection than the program stored is a
/// recognition, not a lowering, and the door refuses it BY NAME.
#[test]
fn a_wider_k_offset_vector_than_the_rounds_is_refused_by_the_three_way_k_cross_check() {
    let src = src_with(&[
        (
            "K: tl.constexpr,  #",
            "K: tl.constexpr, KW: tl.constexpr,  #",
        ),
        ("tl.arange(0, K)", "tl.arange(0, KW)"),
        ("[M, K]", "[M, KW]"),
        ("strides=[K, 1]", "strides=[KW, 1]"),
        ("tl.full([K],", "tl.full([KW],"),
        ("tl.zeros([K],", "tl.zeros([KW],"),
    ]);
    let err = match drive_src(&src, TEMP, 16, &[], &[("KW", Val::Int(16))]) {
        Ok(ops) => panic!(
            "BAKED ({} ops) — a router whose k-offset vector states 16 lanes over 8 rounds \
             lowered through the door; the three-way k cross-check did not fire",
            ops.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.starts_with("door:")
            && err.contains("by the accumulation walk")
            && err.contains("by the k-offset generate's width")
            && err.contains("by the ids landing's own columns"),
        "refused, but not for the reason this test pins. Expected the three-way k \
         cross-check to name the disagreement, got: {err:?}"
    );
}

/// A PLANTED ZERO LANE MASK — the lane select's mask-out moved off −1.0. The
/// tie-break law: `reduce_max` over `select(cmp, offs, −1)` takes the LAST lane
/// holding the maximum (scan order), so any other mask-out is a different selection
/// rule on ties, and the recogniser must refuse it BY NAME — the check that proves the
/// mask was READ rather than assumed (an assumed constant would bake this kernel
/// silently wrong).
#[test]
fn a_zero_lane_mask_is_refused_naming_the_tie_break_law() {
    let src = src_with(&[(
        "tl.full([M, E], -1.0, tl.float16)",
        "tl.full([M, E], 0.0, tl.float16)",
    )]);
    let err = match drive_src(&src, TEMP, K as u64, &[], &[]) {
        Ok(ops) => panic!(
            "BAKED ({} ops) — a router with a zero lane mask lowered through the door; \
             the tie-break check did not read the splat",
            ops.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.starts_with("door:")
            && err.contains("masks the unselected lanes with a splat other than −1.0")
            && err.contains("the tie-break law"),
        "refused, but not for the reason this test pins. Expected the lane select's \
         mask-out to be refused by the tie-break law, got: {err:?}"
    );
}

/// A PLANTED UNREGISTERED TEMPERATURE — the softmax's temperature moved off `H^-0.5`
/// (to 0.09, a value the f16 grid rounds to a DIFFERENT image) while the layout's
/// registry keeps the canonical slot. The descriptor multiplies by a bound `[1,1]`
/// const, so the value the program uses must have a registry slot — a kernel and a
/// registry that disagree are two programs, and the `scale_slot` law refuses the
/// desync BY NAME.
#[test]
fn an_unregistered_temperature_is_refused_naming_the_registry_desync() {
    let err = match drive_src(SRC, 0.09, K as u64, &[], &[]) {
        Ok(ops) => panic!(
            "BAKED ({} ops) — a router at an unregistered temperature lowered through the \
             door; the scale-slot law did not consult the registry",
            ops.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.starts_with("door:")
            && err.contains("is absent from `BundleLayout::scalarmul_scales`")
            && err.contains("registry desync"),
        "refused, but not for the reason this test pins. Expected the temperature to be \
         refused as a registry desync, got: {err:?}"
    );
}

/// A PLANTED UNFOLDED ROUND 0 — the three accumulators seeded at 1.0 instead of zeros,
/// so round 0's updates keep their adds and the lockstep walk descends PAST round 0
/// looking for the fold. The walk counts one round per top-k slot; the descent past
/// round 0 finds a NINTH round against the landing's eight slots, and the walk refuses
/// BY NAME — the descent itself is the form being read (a folded round 0 stops it at
/// eight), so an assumed form would have baked this kernel silently wrong.
#[test]
fn an_unfolded_round_zero_is_refused_naming_the_round_form() {
    let src = src_with(&[(
        "tl.zeros([M, K], dtype=tl.float16)",
        "tl.full([M, K], 1.0, tl.float16)",
    )]);
    let err = match drive_src(&src, TEMP, K as u64, &[], &[]) {
        Ok(ops) => panic!(
            "BAKED ({} ops) — a router whose round 0 keeps its add lowered through the \
             door; the accumulator walk did not read the form",
            ops.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.starts_with("door:")
            && err.contains(
                "accumulation walk found 9 rounds against an ids landing of \
                             8 slots"
            )
            && err.contains("counts one round per top-k slot"),
        "refused, but not for the reason this test pins. Expected the unfolded round 0 to \
         be refused at the accumulation walk's round count, got: {err:?}"
    );
}

/// A PLANTED PARTIAL COMBINE READ — a copy of the route-weights scratch through a
/// fresh parameter, spelled exactly as the combine's read (a direct load over a
/// plain tile feeding a lane select) but with ONE select, not one per pair. The
/// combine rung roots at every route-weights read and claims it ahead of the walk:
/// the read must be the combine's own form, and this one holds a single one-hot
/// select against the router's proved k, so the recognizer refuses it BY NAME at
/// the pair-count cross-check. The write-dead hazard this mutant first exercised
/// is unchanged — the fused router writes its own minted landings and never the
/// scratch parameter — but the read is now the combine recognizer's to judge, and
/// its refusal is the stronger one: no compute reader of the route weights falls
/// through to the census anymore.
#[test]
fn a_partial_combine_read_of_the_route_weights_is_refused_naming_the_pair_count_law() {
    let src = src_with(&[
        ("desc_ids, desc_rs,  #", "desc_ids, desc_rs, desc_leak,  #"),
        (
            "    rs_desc.store([0, 0], tl.fdiv(ex, tl.sum(ex, 1)[:, None]).to(tl.float16) * psel)",
            "    rs_desc.store([0, 0], tl.fdiv(ex, tl.sum(ex, 1)[:, None]).to(tl.float16) * psel)\n\
             \x20   leak_desc = tl.make_tensor_descriptor(desc_leak, shape=[M, K], strides=[K, 1],\n\
             \x20                                        block_shape=[M, K])\n\
             \x20   leak_cond = koffs[None, :] == tl.full([1, K], 0.0, tl.float16)\n\
             \x20   leak_desc.store([0, 0], tl.where(leak_cond, rs_desc.load([0, 0]),\n\
             \x20                                      tl.zeros([1, K], tl.float16)))",
        ),
    ]);
    let err = match drive_src(&src, TEMP, K as u64, &[("desc_leak", "*fp16")], &[]) {
        Ok(ops) => panic!(
            "BAKED ({} ops) — a partial combine read of the route-weights scratch lowered \
             through the door; the recognizer did not hold the pair-count law",
            ops.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.starts_with("door:")
            && err.contains("has 1 one-hot selects")
            && err.contains("the router proved k=8")
            && err.contains("one select per pair is the combine's own form"),
        "refused, but not for the reason this test pins. Expected the partial combine read \
         to be refused at the pair-count cross-check, got: {err:?}"
    );
}

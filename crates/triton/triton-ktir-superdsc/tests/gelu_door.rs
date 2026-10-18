// SPDX-License-Identifier: Apache-2.0

//! ⛔⛔⛔ THE GELU-TANH LONGHAND AT THE WHOLE-FUNCTION DOOR — CLAIMED ONLY WHEN PROVEN.
//!
//! # WHAT THIS MEASURES
//!
//! `program_gelu_mul_chains` claims a `g/(1+exp(-c₁·(g+c₂·g³)))` chain and the walk
//! lowers its terminal `arith.mulf` as ONE `Program::ExpertGatedAct(Gelu)` —
//! `gated_act`'s two-op form (`gact_o*` then `mulgact_o*`), the same body the
//! per-`Program` door emits, so the two doors cannot drift about what a gelu is.
//! The recognition is a CORRECTNESS matter on both edges: without it, the longhand is
//! a silu NEAR-MISS (the same five op kinds, a different numerator), the silu reader's
//! fail-closed discriminator refuses it, and — because the pre-reads gate the whole
//! per-op walk — NOTHING else in a fused kernel holding one can lower; with a sloppy
//! one, a misread constant or association is a DIFFERENT function lowered silently.
//!
//! # WHY EACH CONTROL IS A CORRECTNESS FACT AND NOT A STYLE RULE
//!
//! * the POSITIVE control — the canonical kernel (moe.py's `_gelu_tanh` verbatim, on
//!   plain loads) must BAKE as exactly the two gated-act ops. A bake that emits the
//!   silu names (`silu_o*`/`mulsilu_o*`) would be the gelu misrouted through the silu
//!   body; one that emits ten pointwise ops would be the recogniser not firing; either
//!   is a silent wrong answer, not a style regression;
//! * a planted wrong **c₂** — the cubic's constant is READ off the splat and refused
//!   BY NAME: the tanh gelu is `2·√(2/π)` and `0.044715`, and any other constants are a
//!   different function with the same op kinds. A recogniser that ASSUMED the constant
//!   would bake this kernel, so this control is what makes the read load-bearing;
//! * a planted wrong **c₁** — the same law on the e-argument's scale, the other splat
//!   the verification owns;
//! * a planted **silu** — `swiglu_mlp.py`'s open-coded sigmoid through the same door:
//!   it must stay `silu_o*`/`mulsilu_o*`, NOT the gated-act names. This is the two
//!   readers' division of the negfs made executable: the gelu reader returns the silu
//!   chain unclaimed at the numerator discriminator, and the silu reader skips only
//!   CLAIMED gelus — a drift in either direction misroutes the other's chains.
//!
//! # THE DRIVER
//!
//! Every case compiles a Triton kernel through the full ladder (`codegen::compile` →
//! `make_ttir` → `from_ttir` → `make_ktir` → `to_ktir`) and into `emit_whole` — the
//! door the recognisers live behind — because a test that starts from hand-written
//! KTIR cannot refuse at the chain reader. The canonical source carries moe.py's
//! `_gelu_tanh` VERBATIM (the file the splice will splice, not a paraphrase of it);
//! the controls are one `.replace()` away from it, the matmul_fp8 contract's own law.
//! The full fused kernel — the eight chains in their real W8A8 context — is moe_door's
//! pin, which this recogniser moved past the pre-reads; this file isolates the
//! recogniser's own contract.

use ktir_superdsc::ktir_node::{GatedAct, Program};
use std::collections::HashMap;
use triton_frontend::codegen::{self, ArgSpec, KernelSpec};
use triton_frontend::opt;
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

/// The canonical gelu kernel: moe.py's `_gelu_tanh` copied VERBATIM (docstring
/// included — it states the f32-island law the recogniser's narrowing check enforces),
/// on the plainest loads that reach the door. `N` is the moe geometry's intermediate
/// width `I` — one f16 stick, the pointwise column law's own minimum.
const SRC: &str = r#"
import triton
import triton.language as tl


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
def gelu_fwd(desc_g, desc_u, desc_o, N: tl.constexpr):
    g_desc = tl.make_tensor_descriptor(desc_g, shape=[1, N], strides=[N, 1],
                                       block_shape=[1, N])
    u_desc = tl.make_tensor_descriptor(desc_u, shape=[1, N], strides=[N, 1],
                                        block_shape=[1, N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[1, N], strides=[N, 1],
                                       block_shape=[1, N])
    g = g_desc.load([0, 0])
    u = u_desc.load([0, 0])
    o_desc.store([0, 0], _gelu_tanh(g, N) * u)
"#;

/// The silu control: `swiglu_mlp.py`'s open-coded sigmoid (its deltas 5 and 6 — the
/// widen/truncate island `LegalizeTypes` collapses, `tl.fdiv` because `/` upcasts f16
/// to f32), on the same loads. The silu longhand is the gelu's five op kinds with the
/// negated value as the divide's numerator — the discriminator that must send it to
/// the silu reader, not this one.
const SILU_SRC: &str = r#"
import triton
import triton.language as tl


@triton.jit
def silu_fwd(desc_g, desc_u, desc_o, N: tl.constexpr):
    g_desc = tl.make_tensor_descriptor(desc_g, shape=[1, N], strides=[N, 1],
                                       block_shape=[1, N])
    u_desc = tl.make_tensor_descriptor(desc_u, shape=[1, N], strides=[N, 1],
                                        block_shape=[1, N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[1, N], strides=[N, 1],
                                       block_shape=[1, N])
    g = g_desc.load([0, 0])
    u = u_desc.load([0, 0])
    e = tl.exp((-g).to(tl.float32)).to(tl.float16)
    s = tl.fdiv(g, tl.full([1, N], 1.0, tl.float16) + e)
    o_desc.store([0, 0], s * u)
"#;

/// The three-pointer signature both kernels share: gate f16, up f16, out f16.
fn sig() -> HashMap<String, ArgSpec> {
    HashMap::from([
        ("desc_g".into(), ArgSpec::parse("*fp16").unwrap()),
        ("desc_u".into(), ArgSpec::parse("*fp16").unwrap()),
        ("desc_o".into(), ArgSpec::parse("*fp16").unwrap()),
    ])
}

/// Compile a kernel through the whole ladder and into `emit_whole` — the door the
/// recognisers live behind. `Err` carries the first stage's refusal text, so a
/// ladder-stage break and a door refusal are distinguishable in the assertion message.
/// The node's `program` label is INERT under this door (`lower_function` derives every
/// op's kind from its `OpKind` and never reads `node.program`), so it is minted with
/// the fused kind the kernel's chain lowers as.
fn drive_src(
    src: &str,
    kernel: &str,
    program: Program,
) -> Result<Vec<String>, String> {
    let mut signature = sig();
    signature.insert("N".into(), ArgSpec::parse("constexpr").expect("constexpr"));
    let kspec = KernelSpec {
        kernel: kernel.into(),
        signature,
        constexprs: HashMap::from([("N".into(), Val::Int(64))]),
        file: format!("{kernel}.py"),
    };
    let mut tt =
        codegen::compile(src, &kspec, Target::spyre()).map_err(|e| format!("codegen: {e}"))?;
    opt::make_ttir(&mut tt).map_err(|e| format!("make_ttir: {e}"))?;
    let mut m = triton_ktir::from_ttir::convert(&tt).map_err(|e| format!("from_ttir: {e}"))?;
    triton_ktir::make_ktir(&mut m, &[1]).map_err(|e| format!("make_ktir: {e}"))?;
    triton_ktir::passes::to_ktir::run(&mut m, &[1]).map_err(|e| format!("to_ktir: {e}"))?;
    let node = triton_ktir_superdsc::node_for(&m, program).map_err(|e| format!("handoff: {}", e.message))?;
    let ops = triton_ktir_superdsc::emit_whole(&node, None)
        .map_err(|e| format!("door: {}", e.message))?;
    Ok(ops.into_iter().map(|o| o.op_name.clone()).collect())
}

/// The gelu kernel — the canonical source, unmodified.
fn drive_gelu() -> Result<Vec<String>, String> {
    drive_src(SRC, "gelu_fwd", Program::ExpertGatedAct(GatedAct::Gelu))
}

/// THE POSITIVE CONTROL: the canonical kernel bakes, and the op list IS the gated act —
/// the act leg then the multiply leg, two ops and no more. The ten-op longhand
/// (three cubic products, the sum, the c₁ scale, negate, exp, the 1+, the divide, the
/// terminal multiply) is subsumed by the two descriptors, so a recogniser that silently
/// stopped claiming would show up as an op-count change, and a misrouting to the silu
/// body as the silu names — both asserted against, not just the happy path.
#[test]
fn the_canonical_gelu_kernel_bakes_and_emits_the_gated_act() {
    let names = drive_gelu().expect("the canonical gelu kernel bakes");
    assert!(
        names.len() == 2,
        "the gated act is 2 ops (gact→mulgact); got {}: {names:?}",
        names.len()
    );
    // The chain IN ORDER: the act leg strictly before the multiply leg. The names carry
    // the out tid (`gact_o2`), so the match is on the prefix.
    let mut at = 0;
    for want in ["gact_o", "mulgact_o"] {
        match names[at..].iter().position(|n| n.starts_with(want)) {
            Some(i) => at += i + 1,
            None => panic!(
                "the gated act is 2 ops (gact→mulgact), in order; `{want}` is missing or \
                 out of order: {names:?}"
            ),
        }
    }
    // The silu body's names must NOT appear — the gelu routes through `gated_act`, and
    // a drift onto `silumul` would emit the silu primitive for a gelu (the exact
    // misrecognition the door's fail-closed laws exist for).
    assert!(
        !names.iter().any(|n| n.contains("silu")),
        "`silu_o`/`mulsilu_o` are the silu body's names and must not be emitted for a \
         gelu: {names:?}"
    );
}

/// A PLANTED WRONG c₂ — the cubic's constant mutated one decimal away. The tanh gelu
/// is `2·√(2/π)·(g + 0.044715·g³)`; any other constant is a different function with
/// the same op kinds, so the recogniser must refuse it BY NAME — and the refusal naming
/// the expected constant is what proves the check READ the splat rather than assuming
/// it (an assumed constant would bake this kernel silently wrong).
#[test]
fn a_planted_wrong_c2_is_refused_naming_the_constant() {
    let src = SRC.replace("0.044715", "0.0449");
    let err = match drive_src(&src, "gelu_fwd", Program::ExpertGatedAct(GatedAct::Gelu)) {
        Ok(names) => panic!(
            "BAKED ({} ops) — a gelu longhand with a mutated c₂ lowered through the door; \
             the constant check did not read the splat",
            names.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.starts_with("door:")
            && err.contains("0.044715")
            && err.contains("refused rather than lowered as `OpFunc::Gelu`"),
        "refused, but not for the reason this test pins. Expected the door to name the \
         tanh gelu's c₂ and refuse as `OpFunc::Gelu`, got: {err:?}"
    );
}

/// A PLANTED WRONG c₁ — the e-argument's scale mutated, the other splat the
/// verification owns. Same law, same message shape: the sigmoid identity's argument is
/// exactly `2·√(2/π)·(g + c₂·g³)`, and a different scale is a different function.
#[test]
fn a_planted_wrong_c1_is_refused_naming_the_constant() {
    let src = SRC.replace("1.5957691216057308", "1.6");
    let err = match drive_src(&src, "gelu_fwd", Program::ExpertGatedAct(GatedAct::Gelu)) {
        Ok(names) => panic!(
            "BAKED ({} ops) — a gelu longhand with a mutated c₁ lowered through the door; \
             the constant check did not read the splat",
            names.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.starts_with("door:")
            && err.contains("1.5957691216057308")
            && err.contains("refused rather than lowered as `OpFunc::Gelu`"),
        "refused, but not for the reason this test pins. Expected the door to name the \
         tanh gelu's c₁ and refuse as `OpFunc::Gelu`, got: {err:?}"
    );
}

/// A PLANTED SILU — `swiglu_mlp.py`'s open-coded sigmoid through the same door. It must
/// STAY a silu: the silu names, not the gated-act names. This pins the two readers'
/// division of the negfs in both directions at once — the gelu reader returning the
/// silu chain unclaimed at the numerator discriminator, and (with the positive control
/// above) the silu reader skipping only claimed gelus. A drift in either direction
/// misroutes the other reader's chains and computes the wrong activation.
#[test]
fn a_planted_silu_stays_silu() {
    let names =
        drive_src(SILU_SRC, "silu_fwd", Program::SiluMul).expect("the silu kernel bakes");
    assert!(
        names.len() == 2,
        "the silu multiply is 2 ops (silu→mulsilu); got {}: {names:?}",
        names.len()
    );
    let mut at = 0;
    for want in ["silu_o", "mulsilu_o"] {
        match names[at..].iter().position(|n| n.starts_with(want)) {
            Some(i) => at += i + 1,
            None => panic!(
                "the silu multiply is 2 ops (silu→mulsilu), in order; `{want}` is missing \
                 or out of order: {names:?}"
            ),
        }
    }
    assert!(
        !names.iter().any(|n| n.contains("gact")),
        "`gact_o`/`mulgact_o` are the gated act's names and must not be emitted for a \
         silu: {names:?}"
    );
}

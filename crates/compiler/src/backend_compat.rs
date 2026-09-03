// SPDX-License-Identifier: Apache-2.0
//! Compile-time backend-capability check.
//!
//! Each backend (`Cuda`, `Metal`, …) carries a capability matrix —
//! the set of DSL tile kinds whose `Impl::matches` actually claims a
//! tile on that backend. When an arch's DSL emits a tile no Impl
//! claims, the solver historically failed at macro-expansion time
//! and was silently swallowed (or claimed by a stub that didn't
//! actually run the op), producing garbage output at inference time.
//! Most recent instance: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
//! on metal — Qwen2 emits `bias_add(q, q_proj.bias)` tiles after the
//! Q/K/V projections; on metal-int4 every claimant rejects the tile
//! (`MetalBiasAddImpl::matches() → None`, `FusedGemmBiasImpl` gates
//! out Affine storage, `MetalSynthPreAttnImpl` doesn't include the
//! biased pattern), so the bias is silently dropped layer-by-layer
//! → "valid-tokens-in-random-order" output.
//!
//! `BackendCompat<B>` makes that failure a *compile* error. The
//! trait's `COMPAT_CHECK` associated const carries a `const_assert!`
//! over the arch's `HAS_*` capability flags ([`CanonicalParams`]
//! emits these from DSL inspection). Every macro-emitted per-arch
//! metal codepath references the const so monomorphization triggers
//! the assertion.
//!
//! When a new backend Impl lands that claims a previously-unclaimed
//! tile, relax the corresponding `assert!` here.

use crate::CanonicalParams;

/// Marker type for the CUDA backend.
pub struct Cuda;
/// Marker type for the Metal backend.
pub struct Metal;
/// Marker type for the WGPU backend (placeholder — relax bounds as
/// WGPU Impls land).
pub struct Wgpu;

/// Per-backend compatibility marker. The blanket impl below sits on
/// every `CanonicalParams`; the `COMPAT_CHECK` const-eval `assert!`s
/// the arch's capability flags against the backend's supported set.
/// A failing assertion is a compile-time error at the first use site
/// that monomorphizes `<Weights as BackendCompat<Backend>>::COMPAT_CHECK`.
pub trait BackendCompat<B>: CanonicalParams {
    /// Const-eval assertion. Read by the macro-emitted per-arch
    /// metal/cuda codepath as `const _: () = <Weights as
    /// BackendCompat<Metal>>::COMPAT_CHECK;` to force monomorphization.
    const COMPAT_CHECK: ();
}

// ── Cuda — currently supports the full matrix ─────────────────────

impl<W: CanonicalParams> BackendCompat<Cuda> for W {
    const COMPAT_CHECK: () = ();
}

// ── Metal — supports a subset; expands as Impls land ─────────────

impl<W: CanonicalParams> BackendCompat<Metal> for W {
    const COMPAT_CHECK: () = {
        // QKV `bias_add` tiles are claimed by `MetalBiasAddImpl`
        // (singleton path) — landed P1 to unblock Qwen2 4bit on
        // Metal. The synth-pre-attn megakernel (P2, in flight) will
        // absorb the same tiles into one dispatch at M=1 decode; at
        // M≥2 prefill the cost CSV picks the unfused singleton
        // chain (`MetalAffineQmmImpl` + `MetalBiasAddImpl` +
        // `MetalRopeAppendImpl`). No `HAS_BIAS_ADD` assertion needed
        // — both decode and prefill routes claim end-to-end.
        //
        // MoE-on-Metal — router prereqs are in flight but
        // `gather_qmm_rhs` + the pure-ICB SwitchGLU lowering
        // haven't landed. Every MoE
        // arch (Mixtral, Qwen2-MoE, Qwen3-MoE, DeepSeek V2/V3)
        // blocks on this until the SwitchGLU Impl is wired.
        assert!(
            !W::HAS_MOE,
            "BackendCompat<Metal>: this arch is mixture-of-experts but \
             MoE-on-Metal isn't landed yet (router softmax/argpartition + \
             gather_qmm_rhs ICB lowering open). Relax this assert once \
             SwitchGLU is wired through to a metal Impl.",
        );
    };
}

// ── Wgpu — placeholder ────────────────────────────────────────────

impl<W: CanonicalParams> BackendCompat<Wgpu> for W {
    const COMPAT_CHECK: () = {
        panic!(
            "BackendCompat<Wgpu>: no arches are wired to the WGPU backend yet. \
             Land per-arch WGPU Impls before instantiating this constraint."
        );
    };
}

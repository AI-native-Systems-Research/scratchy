// SPDX-License-Identifier: Apache-2.0
//! [`ConcurrencyModel`] — what overlaps on the target hardware.
//!
//! Consulted by the post-scheduler cost aggregator (see
//! [`crate::cost`]) when two implementations land in the same wave.
//! Without it the cost model would happily produce a multi-stream
//! schedule that doesn't actually run faster (e.g. two compute-bound
//! GEMMs on different streams that both hit the saturated tensor
//! core).
//!
//! Ported verbatim from `scratchy-target-metal/src/lowering/concurrency.rs`.
//! The rules are target-agnostic — they consume only the
//! `launch_kind()` and `is_compute_bound()` declarations on each
//! Impl, both of which are uniform across architectures.
//!
//! ## First-cut rules (sm_89, L4)
//!
//! Encoded as four hand-written rules:
//!
//! 1. **CooperativeLaunch is exclusive.** Only one cooperative grid
//!    can be resident on the device. Two CooperativeLaunch impls
//!    scheduled at the same wave ⇒ infinite contention factor.
//!
//! 2. **Compute-bound impls don't overlap each other.** Two impls
//!    that are both `is_compute_bound() == true` and run on the
//!    same target's tensor cores get a contention factor of 1.0
//!    (no speedup from concurrency — they serialize on the
//!    saturated MMA pipeline).
//!
//! 3. **Memory-bound impls can overlap each other and a
//!    compute-bound impl.** A memory-bound op (norm, rope,
//!    silu_mul, residual_add) running concurrently with a
//!    compute-bound GEMM on a different stream actually overlaps
//!    on the target — the LSU and tensor cores are independent
//!    units. Contention factor for the memory-bound op is ~0.5
//!    (it runs roughly in the shadow of the compute-bound op).
//!
//! 4. **DeviceCallable inside a persistent megakernel** can
//!    overlap with other DeviceCallable impls in the same kernel
//!    if they're assigned to disjoint CTA subsets, modulo the
//!    resource-union constraint. The amortization factor is
//!    proportional to the CTA-subset coverage. Not yet exercised
//!    by the starter library (all HostCallback); returns 1.0 as a
//!    stub — extend when the first DeviceCallable+DeviceCallable
//!    case appears.
//!
//! These rules are minimal but enough to get the solver started.
//! Extension shape is documented inline.

use crate::impl_lib::{Implementation, LaunchKind};
use crate::target::TargetProfile;

/// Multiplicative factor applied to an implementation's `cost_us`
/// when it runs concurrently with another. `1.0` means "no
/// speedup, no slowdown"; `< 1.0` means "this impl runs faster
/// because it overlaps with the other"; `f64::INFINITY` means
/// "this concurrent schedule is infeasible (e.g. two cooperative
/// grids)."
pub type ContentionFactor = f64;

/// First-cut concurrency model. Hand-written rules; extend as
/// measurements arrive.
pub struct ConcurrencyModel<'p> {
    pub profile: &'p TargetProfile,
}

impl<'p> ConcurrencyModel<'p> {
    pub fn new(profile: &'p TargetProfile) -> Self {
        Self { profile }
    }

    /// Contention factor for `target_impl` running concurrently
    /// with the set of `other_impls` already scheduled in the same
    /// wave.
    ///
    /// Returns `f64::INFINITY` if the schedule is infeasible (e.g.
    /// `target_impl` is `CooperativeLaunch` and the others include
    /// any other launch).
    pub fn contention_factor(
        &self,
        target_impl: &dyn Implementation,
        other_impls: &[&dyn Implementation],
    ) -> ContentionFactor {
        // Rule 1: CooperativeLaunch exclusivity.
        if matches!(target_impl.launch_kind(), LaunchKind::CooperativeLaunch)
            && !other_impls.is_empty()
        {
            return f64::INFINITY;
        }
        for other in other_impls {
            if matches!(other.launch_kind(), LaunchKind::CooperativeLaunch) {
                return f64::INFINITY;
            }
        }

        // Rule 2: compute-bound + compute-bound = no overlap.
        if target_impl.is_compute_bound() {
            let any_other_compute_bound = other_impls.iter().any(|o| o.is_compute_bound());
            if any_other_compute_bound {
                // Both compete for the tensor cores. No speedup.
                return 1.0;
            }
        }

        // Rule 3: memory-bound impl shadows behind a compute-bound
        // impl. The memory-bound impl gets a 0.5 contention factor
        // (effectively half the wall-clock cost) when there's a
        // compute-bound impl running concurrently. Compute-bound
        // impls are unaffected by memory-bound co-runners.
        if !target_impl.is_compute_bound() {
            let any_compute_bound = other_impls.iter().any(|o| o.is_compute_bound());
            if any_compute_bound {
                return 0.5;
            }
        }

        // Rule 4: DeviceCallable in same persistent kernel — not yet
        // exercised by the L4 sm_89 starter library, which is all
        // HostCallback. Stub for now; returns 1.0 (no concurrency
        // benefit by default; add the per-CTA-subset rule when the
        // first DeviceCallable+DeviceCallable case shows up).
        let _ = self.profile;
        1.0
    }
}

#[cfg(all(test, any(feature = "cuda", feature = "spyre")))]
mod tests {
    use super::*;
    use crate::classified::OpKind;
    use crate::fuf::{Fuf, TileId};
    use crate::impl_lib::{CostCtx, Handoff, Layout, MatchInfo, Resources};

    fn l4_target() -> TargetProfile {
        crate::target::from_profile_def(&scratchy_target_cuda::targets::L4_SM89)
    }

    /// Minimal Impl for exercising ConcurrencyModel rules in isolation.
    #[derive(Debug)]
    struct TestImpl {
        name: &'static str,
        launch: LaunchKind,
        compute_bound: bool,
    }

    impl Implementation for TestImpl {
        fn name(&self) -> &'static str {
            self.name
        }
        fn target_compatible(&self, _p: &TargetProfile) -> bool {
            true
        }
        fn matches(&self, _f: &Fuf, _s: TileId, _p: &TargetProfile) -> Option<MatchInfo> {
            None
        }
        fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
            1.0
        }
        fn resources(&self, _m: &MatchInfo) -> Resources {
            Resources::ZERO
        }
        fn launch_kind(&self) -> LaunchKind {
            self.launch
        }
        fn supported_input_handoffs(&self) -> &[Handoff] {
            &[]
        }
        fn supported_output_handoffs(&self) -> &[Handoff] {
            &[]
        }
        fn input_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
            vec![]
        }
        fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
            vec![]
        }
        fn is_compute_bound(&self) -> bool {
            self.compute_bound
        }
    }

    fn compute_bound(name: &'static str) -> TestImpl {
        TestImpl {
            name,
            launch: LaunchKind::HostCallback,
            compute_bound: true,
        }
    }
    fn memory_bound(name: &'static str) -> TestImpl {
        TestImpl {
            name,
            launch: LaunchKind::HostCallback,
            compute_bound: false,
        }
    }

    fn _unused_opkind_placeholder() {
        // Pin OpKind as used — silences dead-code warnings in tests
        // that only exercise the concurrency layer.
        let _ = OpKind::Gemm;
    }

    #[test]
    fn empty_wave_is_factor_one() {
        let target = l4_target();
        let cm = ConcurrencyModel::new(&target);
        let g = compute_bound("gemm");
        let factor = cm.contention_factor(&g, &[]);
        assert_eq!(factor, 1.0, "single impl in a wave has no contention");
    }

    #[test]
    fn two_cooperative_launches_are_infeasible() {
        let target = l4_target();
        let cm = ConcurrencyModel::new(&target);
        let a = TestImpl {
            name: "coop_a",
            launch: LaunchKind::CooperativeLaunch,
            compute_bound: true,
        };
        let b = TestImpl {
            name: "coop_b",
            launch: LaunchKind::CooperativeLaunch,
            compute_bound: true,
        };
        let factor = cm.contention_factor(&a, &[&b]);
        assert!(factor.is_infinite(), "two coop grids is infeasible");
    }

    #[test]
    fn cooperative_blocks_any_coresident() {
        let target = l4_target();
        let cm = ConcurrencyModel::new(&target);
        let coop = TestImpl {
            name: "coop",
            launch: LaunchKind::CooperativeLaunch,
            compute_bound: true,
        };
        let host = compute_bound("host");
        // Coop + any other → infeasible, regardless of which is
        // "target".
        assert!(cm.contention_factor(&coop, &[&host]).is_infinite());
        assert!(cm.contention_factor(&host, &[&coop]).is_infinite());
    }

    #[test]
    fn compute_plus_compute_in_same_wave_no_speedup() {
        let target = l4_target();
        let cm = ConcurrencyModel::new(&target);
        let a = compute_bound("gemm_a");
        let b = compute_bound("gemm_b");
        let factor = cm.contention_factor(&a, &[&b]);
        assert_eq!(
            factor, 1.0,
            "two compute-bound impls serialize on tensor cores"
        );
    }

    #[test]
    fn memory_shadows_behind_compute() {
        let target = l4_target();
        let cm = ConcurrencyModel::new(&target);
        let gemm = compute_bound("gemm");
        let norm = memory_bound("norm");
        // Memory-bound impl with a compute-bound co-runner: ~0.5 factor.
        assert_eq!(cm.contention_factor(&norm, &[&gemm]), 0.5);
        // Compute-bound impl with a memory-bound co-runner: unaffected.
        assert_eq!(cm.contention_factor(&gemm, &[&norm]), 1.0);
    }

    #[test]
    fn memory_plus_memory_is_neutral() {
        let target = l4_target();
        let cm = ConcurrencyModel::new(&target);
        let a = memory_bound("norm_a");
        let b = memory_bound("norm_b");
        // No compute-bound co-runner → default 1.0 (no shadow).
        assert_eq!(cm.contention_factor(&a, &[&b]), 1.0);
    }
}

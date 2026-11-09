// SPDX-License-Identifier: Apache-2.0
//! How a launched group relaunches and where it reads — the bake's walk law, applied per launch.
//!
//! Every relaunched group on the card is a walk's body: its trip count and per-trip step are bake
//! facts ([`scratchy_spyre_bundle::Walk`]) carried on the group, so the launch reads them and decides
//! nothing. Every other group launches once, where the bake placed it.

/// The KV cache's segment index.
pub const SEG_KV: usize = 2;
/// The activation segment's index — where a walk's step tables (page table, mask stack) sit, and so
/// the segment a walk's body shifts per trip.
pub const SEG_MASK: usize = 3;

/// A byte quantity — an offset, a stride, or a base.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct Bytes(pub u64);

impl std::ops::Add for Bytes {
    type Output = Bytes;
    fn add(self, o: Bytes) -> Bytes {
        Bytes(self.0.wrapping_add(o.0))
    }
}

/// The per-op launch-group facts the launch arithmetic reads, as the bake stated them.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpKv {
    /// `kv_page_fold`: this group is a walk's body, relaunched [`Self::walk`]'s trip count.
    pub page_fold: bool,
    /// ⭐ THIS GROUP READS THROUGH A **GATHERED** SCRATCH — each page came from the index the host
    /// stages, so the group takes NO KV segment shift: a gathered read computes
    /// `addr = idx * skip_addr + base`, and a page shift on top of an absolute entry composes two bases
    /// and lands inside neither.
    pub gathered: bool,
    /// ⭐ A WALK'S BLOCK BODY (`bundle::KvShifts::walk`): launched the walk's own trip count, the
    /// activation segment shifted one step per trip and nothing else — the bake's facts.
    pub walk: Option<scratchy_spyre_bundle::Walk>,
}

/// ⛔⛔⛔ THE RUNG A CONTEXT PAST EVERY SWEEP TAKES — the widest one that FOLDS, and the top only when
/// none does. `folds[i]` says whether ladder rung `i` (ascending) carries a page-fold group.
///
/// A folding body re-launches once per resident page, so it serves any length the pool holds. A
/// split-KV chunk body does not fold: it attends exactly its `pages * POOL_PAGE_SLOTS`, and the one-request
/// ladder's TOP is its widest chunk. Taking the top past it handed a 21,945-token granite-2b decode the
/// 64-page chunk, and the worker refused the step ("the chunk ladder's top rung (64 pages) is below
/// this context") where the per-page fold had served it. Every ladder without a chunk folds on every
/// rung, so there this is the top, as before. Applied ONCE, at bake, by
/// [`fallback_rung`]; the runtime reads the index it baked (`RerollMeta::fallback_rung`).
pub fn past_every_sweep(folds: &[bool]) -> usize {
    folds
        .iter()
        .rposition(|&f| f)
        .unwrap_or(folds.len().saturating_sub(1))
}

/// ⭐ ONE VALUE PER ATTENTION CLASS, AND ONLY EVERY CLASS — the fold-fused twins of one body.
///
/// ⛔⛔⛔ A SET WITH A CLASS MISSING IS UNREPRESENTABLE, AND THAT IS THE POINT. The twins come off the
/// meta as one `SiblingFp` per class, `none()` for a class that sits out, and the executor's loader
/// FILTERED the sitting-out ones away before collecting. Two defects followed from that one filter:
///   * a rung whose classes all sat out became an EMPTY list, the selector still handed it out, and
///     class 0's lookup indexed past its end. MEASURED on granite-3.1-2b fp16 (pod nickm, post-#299
///     main + #558): the first decode step panicked `index out of bounds: the len is 0 but the
///     index is 0` in `superdsc_exec`'s `list()`, `RungFused` arm;
///   * a rung where class 0 sat out and class 1 did not became a ONE-entry list, so class 0's
///     lookup ran CLASS 1's ops: a different attention body, silently.
///
/// [`PerClass::of_every_class`] keeps a class's position or refuses the set. It is `Some` only when
/// there is exactly one twin per class, so `class(c)` always names class `c`'s own twin.
#[derive(Debug)]
pub struct PerClass<T>(Vec<T>);

impl<T> PerClass<T> {
    /// The set of every class's value, in class order. `None` unless exactly `classes` values are
    /// given and each is present.
    pub fn of_every_class(
        classes: usize,
        each: impl IntoIterator<Item = Option<T>>,
    ) -> Option<Self> {
        let v: Vec<T> = each.into_iter().collect::<Option<Vec<T>>>()?;
        (classes > 0 && v.len() == classes).then_some(PerClass(v))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Class `c`'s own value.
    pub fn class(&self, c: usize) -> &T {
        &self.0[c]
    }

    pub fn class_mut(&mut self, c: usize) -> &mut T {
        &mut self.0[c]
    }
}

/// The ladder's fold-fused rungs. Each rung, given as `(swept extent, class count, one optional twin
/// per class)`, is kept with its twins in class order exactly when EVERY class has one
/// ([`PerClass`]). A rung with any class sitting out is dropped, so the selector never hands it out:
/// that context runs the split ladder, which can serve it (one launch per layer dearer).
pub fn fused_rungs<C, T>(
    rungs: impl IntoIterator<Item = (C, usize, Vec<Option<T>>)>,
) -> Vec<(C, PerClass<T>)> {
    rungs
        .into_iter()
        .filter_map(|(cap, classes, twins)| {
            PerClass::of_every_class(classes, twins).map(|p| (cap, p))
        })
        .collect()
}

/// What a baked decode-ladder rung can serve past its own sweep — the two bake facts
/// [`fallback_rung`] needs, read off the rung's emitted ops by the macro.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RungReach {
    /// The body carries a page-fold group: it re-launches per resident page, so it serves any length.
    /// The same fact the runtime reads (`KvShifts::page_fold`, stamped from each op's `kv_page_fold`).
    pub folds: bool,
    /// The rung baked a fold-fused twin — the body `superdsc_exec::select_body_paged` runs whenever the
    /// fold would make ONE pass, i.e. at most one resident page for one request.
    pub fused_twin: bool,
}

/// The bake's half of [`past_every_sweep`]: the rung a context past every sweep runs, `rungs`
/// ascending exactly as the runtime holds them. Baked as `RerollMeta::fallback_rung`, which the
/// runtime reads instead of choosing — so it cannot run another.
pub fn fallback_rung(rungs: &[RungReach]) -> usize {
    let folds: Vec<bool> = rungs.iter().map(|r| r.folds).collect();
    past_every_sweep(&folds)
}

/// How many times this op's group is launched: a walk's body its baked trip count, every other group
/// once.
pub fn reps(op: &OpKv) -> i64 {
    // A WALK'S BODY runs its walk's trip count, baked with the rung: the blocks past a sequence's own
    // are fully masked, so the count is the walk's, never the session's pages. Every other group
    // launches once.
    op.walk.map_or(1, |w| i64::from(w.trips))
}

/// Everything the launch loop adds to the segment bases for one op on one relaunch.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct SegDeltas {
    pub kv: Bytes,
    pub mask: Bytes,
    /// The shift into the INTERMEDIATE segment (seg0). Zero for every group: a walk's body keeps its
    /// state in the one carried buffer.
    pub intermediate: Bytes,
}

/// THE segment shifts for op `op` on relaunch `rep`. Only a walk's body moves: trip `rep` reads the
/// step's page-table row and mask tile `rep` steps into the activation segment, and everything else
/// where the bake placed it — its gathers carry their own pages (the entries are absolute), so no KV
/// shift, and its state is the one carried buffer. Every other group launches once, unshifted.
pub fn seg_deltas(op: &OpKv, rep: i64) -> SegDeltas {
    match op.walk {
        Some(w) => SegDeltas {
            kv: Bytes(0),
            mask: Bytes((rep as u64).wrapping_mul(u64::from(w.step_bytes))),
            intermediate: Bytes(0),
        },
        None => SegDeltas::default(),
    }
}

#[cfg(test)]
mod tests {
    // ⛔ THIS MODULE, NOT `cursor_proofs`. That one is `#[cfg(kani)]` and NEVER compiles under `cargo test` — this
    // file's own history records ~100 commits of "the proofs hold" that were VACUOUS for exactly that reason, and
    // a test appended into it a moment ago reported "0 passed; 19 filtered out" rather than failing. A test that
    // cannot run is worse than no test: it reads as coverage.

    /// ⛔ A RUNG WHOSE CLASSES ALL SIT OUT HAS NO FUSED BODY: it is dropped, so the selector never
    /// returns it and the step runs the split ladder. The measured panic was the old loader making it
    /// an empty list the selector still handed out.
    #[test]
    fn a_rung_whose_classes_all_sit_out_has_no_fused_body() {
        use super::fused_rungs;
        let r = fused_rungs([(256u32, 1, vec![None::<&str>]), (512, 2, vec![None, None])]);
        assert!(r.is_empty(), "no rung with a sitting-out class survives");
    }

    /// ⛔ A CLASS NEVER RUNS ANOTHER CLASS's TWIN. With class 0 sitting out and class 1 not, the old
    /// filter left `[class 1's ops]`, so class 0's lookup ran class 1's body. The rung must be dropped
    /// instead, and a rung where every class has a twin keeps each at its own index.
    #[test]
    fn a_class_never_runs_another_class_s_fused_twin() {
        use super::fused_rungs;
        let r = fused_rungs([
            (256u32, 2, vec![None, Some("class1@256")]),
            (512, 2, vec![Some("class0@512"), Some("class1@512")]),
        ]);
        assert_eq!(
            r.len(),
            1,
            "only the rung where every class has a twin is kept"
        );
        let (cap, twins) = &r[0];
        assert_eq!(*cap, 512);
        assert_eq!(twins.len(), 2);
        assert_eq!(*twins.class(0), "class0@512");
        assert_eq!(*twins.class(1), "class1@512");
    }

    /// A set with the wrong number of classes is no set: a meta stating fewer twins than bodies is
    /// refused rather than indexed short.
    #[test]
    fn a_twin_set_must_name_every_class() {
        use super::PerClass;
        assert!(PerClass::of_every_class(2, [Some(1)]).is_none());
        assert!(PerClass::of_every_class(0, Vec::<Option<u8>>::new()).is_none());
        assert_eq!(
            PerClass::of_every_class(1, [Some(7)]).map(|p| *p.class(0)),
            Some(7)
        );
    }

    /// ⛔ A CONTEXT PAST EVERY SWEEP NEVER LANDS ON A CHUNK. The one-request ladder is the per-page sweep
    /// rungs (64, 128, the 256 page body — all folding) then the chunks (2..64 pages, none folding): past
    /// the widest chunk the step must take the 256 page body, which re-launches per page. A ladder that
    /// folds on every rung (head dim 128, the batched bakes) keeps its top.
    #[test]
    fn a_context_past_every_sweep_takes_the_widest_folding_rung() {
        use super::past_every_sweep;
        let one_request = [true, true, true, false, false, false, false, false, false];
        assert_eq!(past_every_sweep(&one_request), 2);
        assert_eq!(past_every_sweep(&[true, true, true]), 2);
        assert_eq!(past_every_sweep(&[false, false]), 1);
    }

    /// ⭐ ONLY A WALK'S BODY RELAUNCHES, AND ONLY BY ITS OWN WALK: its trip count from the bake, its
    /// mask tile `rep` steps of the baked step into the activation segment, and no KV shift — its gathers
    /// carry their own pages. Every other group launches once, unshifted.
    #[test]
    fn only_a_walk_body_relaunches_and_it_steps_by_its_own_walk() {
        use super::{Bytes, OpKv, SegDeltas, reps, seg_deltas};
        let once = OpKv::default();
        assert_eq!(reps(&once), 1);
        assert_eq!(seg_deltas(&once, 0), SegDeltas::default());
        let walk = OpKv {
            page_fold: true,
            gathered: true,
            walk: Some(scratchy_spyre_bundle::Walk {
                trips: 7,
                step_bytes: 4096,
            }),
        };
        assert_eq!(reps(&walk), 7);
        for rep in 0..7 {
            assert_eq!(
                seg_deltas(&walk, rep),
                SegDeltas {
                    kv: Bytes(0),
                    mask: Bytes(rep as u64 * 4096),
                    intermediate: Bytes(0),
                }
            );
        }
    }
}

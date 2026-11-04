// SPDX-License-Identifier: Apache-2.0

//! Splash's draft context plan (~/git/splash @ 1482f6bb, runtime/model/DraftContextPlan.cpp):
//! which of a prefill's rows a block drafter captures into its rings. The goal is that at every
//! boundary the prefill passes, the rings hold that boundary's last window of rows and nothing
//! more. The boundaries are each position whose recurrent state the prefix cache snapshots, and
//! the prefill's end.
//!
//! A boundary's window is captured from the boundary before it when that one is within a window,
//! so its rows continue the rings. Otherwise it is captured whole as a fresh window, and the rows
//! between the two boundaries are never captured. A prefill resumed from a snapshot continues the
//! rings the snapshot's context window rebuilds, when its first boundary is within a window of
//! the resume. Past that the rebuild is skipped, since a fresh window overwrites every row of it.

/// Why a boundary is planned (Splash `DraftBoundaryPurpose`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryPurpose {
    /// A position whose state the prefix cache snapshots, context window and all.
    Materialization,
    /// The prefill's end, where its request drafts.
    Active,
}

/// A boundary and the first row of the window captured for it (Splash `DraftBoundaryPlan`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundaryPlan {
    pub boundary: u32,
    pub purpose: BoundaryPurpose,
    pub capture_begin: u32,
}

/// Rows `[begin, end)`, captured. `reset`: they start a fresh window rather than continuing the
/// rows before them (Splash `DraftCaptureSpan`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureSpan {
    pub begin: u32,
    pub end: u32,
    pub reset: bool,
}

/// One dispatch's rows of a capture span (Splash `DispatchDraftCaptureSpan`): `[begin, end)`, at
/// `compact_row` among the rows the dispatch captures, and how many of them each purpose's windows
/// take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchCapture {
    pub begin: u32,
    pub end: u32,
    pub compact_row: u32,
    /// The rows start their span's fresh window.
    pub reset: bool,
    pub active_rows: u32,
    pub materialization_rows: u32,
}

/// What a prefill over rows `[replay_begin, replay_end)` captures (Splash `DraftContextPlan`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftContextPlan {
    pub replay_begin: u32,
    pub replay_end: u32,
    pub spans: Vec<CaptureSpan>,
    pub boundaries: Vec<BoundaryPlan>,
    /// The first capture continues the rings a restore rebuilt: the replay starts past 0 and its
    /// first boundary is within a window of it. Otherwise the restore leaves the rings alone.
    pub restores_draft_state: bool,
}

/// A plan's inputs Splash refuses (`planDraftContext`, `draftCaptureSpansForDispatch`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DraftContextError {
    #[error("draft replay range {begin}..{end} is reversed")]
    Reversed { begin: u32, end: u32 },
    #[error("draft materialization boundary {boundary} is outside replay range ({begin}, {end}]")]
    Outside { boundary: u32, begin: u32, end: u32 },
    #[error("draft materialization boundary {boundary} after {previous}: not sorted and unique")]
    Unsorted { previous: u32, boundary: u32 },
    #[error("dispatch {begin}..{end} is outside replay range {replay_begin}..{replay_end}")]
    Dispatch {
        begin: u32,
        end: u32,
        replay_begin: u32,
        replay_end: u32,
    },
}

impl DraftContextPlan {
    /// The plan for a prefill over `[replay_begin, replay_end)` that snapshots its state at each of
    /// `materializations` (ascending, each inside `(replay_begin, replay_end]`), for rings of
    /// `window` positions (Splash `planDraftContext`, DraftContextPlan.cpp:7-80).
    pub fn of(
        replay_begin: u32,
        replay_end: u32,
        materializations: &[u32],
        window: u32,
    ) -> Result<Self, DraftContextError> {
        if replay_end < replay_begin {
            return Err(DraftContextError::Reversed {
                begin: replay_begin,
                end: replay_end,
            });
        }
        let mut previous: Option<u32> = None;
        for &boundary in materializations {
            if boundary <= replay_begin || boundary > replay_end {
                return Err(DraftContextError::Outside {
                    boundary,
                    begin: replay_begin,
                    end: replay_end,
                });
            }
            if let Some(previous) = previous.filter(|&p| boundary <= p) {
                return Err(DraftContextError::Unsorted { previous, boundary });
            }
            previous = Some(boundary);
        }
        let mut plan = Self {
            replay_begin,
            replay_end,
            spans: Vec::new(),
            boundaries: Vec::new(),
            restores_draft_state: false,
        };
        let mut state = replay_begin;
        let mut have_state = replay_begin != 0;
        // A snapshot at the prefill's end is its active state itself: planned once, as that.
        let boundaries = (materializations.iter())
            .filter(|&&b| b < replay_end)
            .map(|&b| (b, BoundaryPurpose::Materialization))
            .chain([(replay_end, BoundaryPurpose::Active)]);
        for (boundary, purpose) in boundaries {
            let distance = boundary - state;
            let mut capture_begin = boundary;
            if !(have_state && distance == 0) {
                let continues = have_state && distance < window;
                capture_begin = match continues {
                    true => state,
                    false => boundary - boundary.min(window),
                };
                let rows = boundary - capture_begin;
                match plan.spans.last_mut() {
                    Some(last) if rows != 0 && continues && last.end == capture_begin => {
                        last.end = boundary;
                    }
                    _ if rows != 0 => plan.spans.push(CaptureSpan {
                        begin: capture_begin,
                        end: boundary,
                        reset: !continues,
                    }),
                    _ => {}
                }
            }
            plan.boundaries.push(BoundaryPlan {
                boundary,
                purpose,
                capture_begin,
            });
            state = boundary;
            have_state = true;
        }
        let first = (materializations.first().copied())
            .filter(|&b| b < replay_end)
            .unwrap_or(replay_end);
        plan.restores_draft_state = replay_begin != 0 && first - replay_begin < window;
        Ok(plan)
    }

    /// The rows the plan captures in all (Splash `draftContextRows`).
    pub fn captured_rows(&self) -> u32 {
        self.spans.iter().map(|s| s.end - s.begin).sum()
    }

    /// The captures of a dispatch over rows `[begin, end)` (Splash `draftCaptureSpansForDispatch`,
    /// DraftContextPlan.cpp:82-121), in row order.
    pub fn dispatch(
        &self,
        begin: u32,
        end: u32,
    ) -> Result<Vec<DispatchCapture>, DraftContextError> {
        if end < begin || begin < self.replay_begin || end > self.replay_end {
            return Err(DraftContextError::Dispatch {
                begin,
                end,
                replay_begin: self.replay_begin,
                replay_end: self.replay_end,
            });
        }
        let mut captures = Vec::new();
        let mut compact_row = 0;
        for span in &self.spans {
            let (from, to) = (span.begin.max(begin), span.end.min(end));
            if from >= to {
                continue;
            }
            let mut capture = DispatchCapture {
                begin: from,
                end: to,
                compact_row,
                reset: span.reset && from == span.begin,
                active_rows: 0,
                materialization_rows: 0,
            };
            for boundary in &self.boundaries {
                let (b, e) = (from.max(boundary.capture_begin), to.min(boundary.boundary));
                if b >= e {
                    continue;
                }
                let rows = match boundary.purpose {
                    BoundaryPurpose::Active => &mut capture.active_rows,
                    BoundaryPurpose::Materialization => &mut capture.materialization_rows,
                };
                *rows += e - b;
            }
            debug_assert_eq!(
                capture.active_rows + capture.materialization_rows,
                to - from,
                "every captured row is some boundary's"
            );
            captures.push(capture);
            compact_row += to - from;
        }
        Ok(captures)
    }
}

/// Splash's draft_context_plan_test (dev/tests/engine/draft_context_plan_test.cpp), at its window
/// of 2048 positions (`ExecutionLimits::draftContextTokens`).
#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 2048;

    fn active(begin: u32, end: u32) -> DraftContextPlan {
        DraftContextPlan::of(begin, end, &[], W).expect("a valid plan")
    }

    fn cached(begin: u32, end: u32, boundaries: &[u32]) -> DraftContextPlan {
        DraftContextPlan::of(begin, end, boundaries, W).expect("a valid plan")
    }

    fn purpose_rows(plan: &DraftContextPlan, purpose: BoundaryPurpose) -> u32 {
        (plan.boundaries.iter())
            .filter(|b| b.purpose == purpose)
            .map(|b| b.boundary - b.capture_begin)
            .sum()
    }

    fn resets(plan: &DraftContextPlan) -> usize {
        plan.spans.iter().filter(|s| s.reset).count()
    }

    fn span(begin: u32, end: u32, reset: bool) -> CaptureSpan {
        CaptureSpan { begin, end, reset }
    }

    /// testColdLengths.
    #[test]
    fn a_cold_prefill_captures_its_last_window() {
        for length in [0, 1, 7, 8, 31, 32, 33, 2047, 2048, 2049, 4096, 10000] {
            let plan = active(0, length);
            assert_eq!(plan.captured_rows(), length.min(W), "length {length}");
            assert_eq!(resets(&plan), usize::from(length != 0), "length {length}");
        }
    }

    /// testPartialHit.
    #[test]
    fn a_short_suffix_continues_the_restored_rings_and_a_long_one_does_not() {
        let short = active(4096, 5000);
        assert_eq!(short.spans, [span(4096, 5000, false)]);
        assert_eq!(short.captured_rows(), 904);
        assert!(short.restores_draft_state);
        let long = active(4096, 8192);
        assert_eq!(long.spans, [span(6144, 8192, true)]);
        assert_eq!(long.captured_rows(), 2048);
        assert!(!long.restores_draft_state);
    }

    /// testFinalFullBlockBound.
    #[test]
    fn nearby_boundaries_share_one_span() {
        let plan = cached(0, 10000, &[9984]);
        assert_eq!(plan.spans.len(), 1);
        assert_eq!((plan.spans[0].begin, plan.spans[0].end), (7936, 10000));
        assert_eq!(plan.captured_rows(), 2064);
        // A prompt end on a block boundary backs its snapshot off a whole block: keeping both
        // states takes a window and a block.
        assert_eq!(cached(0, 4096, &[4064]).captured_rows(), 2080);
    }

    /// testJunctionDistancesAndReservation.
    #[test]
    fn a_junction_continues_within_a_window_and_starts_fresh_past_one() {
        assert!(cached(4096, 7000, &[5000]).restores_draft_state);
        let near = cached(0, 6000, &[4096]);
        assert_eq!(near.spans, [span(2048, 6000, true)]);
        assert_eq!(purpose_rows(&near, BoundaryPurpose::Materialization), 2048);
        assert_eq!(purpose_rows(&near, BoundaryPurpose::Active), 1904);
        let far = cached(0, 7000, &[4096]);
        assert_eq!(far.spans, [span(2048, 4096, true), span(4952, 7000, true)]);
        let denied = active(0, 7000);
        assert_eq!(denied.spans, [span(4952, 7000, true)]);
        assert_eq!(purpose_rows(&denied, BoundaryPurpose::Materialization), 0);
    }

    /// testJunctionAndLatestReplayAreBothMaterialized.
    #[test]
    fn a_junction_and_the_prompt_end_are_both_materialized() {
        let plan = cached(0, 10000, &[4096, 9984]);
        let planned: Vec<(u32, BoundaryPurpose)> = (plan.boundaries.iter())
            .map(|b| (b.boundary, b.purpose))
            .collect();
        assert_eq!(
            planned,
            [
                (4096, BoundaryPurpose::Materialization),
                (9984, BoundaryPurpose::Materialization),
                (10000, BoundaryPurpose::Active),
            ]
        );
        assert_eq!(
            plan.spans,
            [span(2048, 4096, true), span(7936, 10000, true)]
        );
        assert_eq!(purpose_rows(&plan, BoundaryPurpose::Materialization), 4096);
        assert_eq!(purpose_rows(&plan, BoundaryPurpose::Active), 16);
        assert_eq!(plan.captured_rows(), 4112);
    }

    /// testCompactDispatchCaptures.
    #[test]
    fn a_dispatch_takes_each_span_it_meets_with_its_accounting() {
        let plan = cached(0, 7000, &[4096]);
        let first = plan.dispatch(2000, 4000).expect("inside");
        let capture =
            |begin, end, compact_row, reset, active_rows, materialization_rows| DispatchCapture {
                begin,
                end,
                compact_row,
                reset,
                active_rows,
                materialization_rows,
            };
        assert_eq!(first, [capture(2048, 4000, 0, true, 0, 1952)]);
        let middle = plan.dispatch(4000, 6048).expect("inside");
        assert_eq!(
            middle,
            [
                capture(4000, 4096, 0, false, 0, 96),
                capture(4952, 6048, 96, true, 1096, 0),
            ]
        );
    }

    /// testFourRaggedLanes.
    #[test]
    fn ragged_lanes_capture_their_own_rows() {
        let rows: u32 = ([33, 2047, 2049, 10000].iter())
            .map(|&l| active(0, l).captured_rows())
            .sum();
        assert_eq!(rows, 33 + 2047 + 2048 + 2048);
    }

    /// testSparseCheckpointWindowsAndShortResume.
    #[test]
    fn sparse_checkpoints_capture_only_their_windows() {
        let plan = cached(0, 33001, &[8192, 16384, 24576, 32768, 32992]);
        assert_eq!((plan.spans.len(), plan.boundaries.len()), (4, 6));
        assert_eq!(plan.captured_rows(), 4 * 2048 + 233);
        assert_eq!(
            purpose_rows(&plan, BoundaryPurpose::Materialization),
            4 * 2048 + 224
        );
        assert_eq!(purpose_rows(&plan, BoundaryPurpose::Active), 9);
        let dispatched: u32 = (0..33001)
            .step_by(1376)
            .flat_map(|b| plan.dispatch(b, (b + 1376).min(33001)).expect("inside"))
            .map(|c| c.end - c.begin)
            .sum();
        assert_eq!(dispatched, plan.captured_rows());
        for suffix in [1, 31, 32, 2047, 2048, 2049] {
            let resumed = active(16384, 16384 + suffix);
            assert_eq!(resumed.captured_rows(), suffix.min(W));
            assert_eq!(resumed.restores_draft_state, suffix < W);
        }
    }

    /// testCheckpointWindowsAcrossSmallDispatches: replaying a plan's dispatches leaves every
    /// boundary's window of rows, in order, at the end of the rows captured.
    #[test]
    fn every_boundary_holds_its_window_across_small_dispatches() {
        let plans = [
            cached(0, 12321, &[4096, 8192, 12288, 12320]),
            cached(8192, 14337, &[8224, 10272, 14336]),
            cached(0, 65, &[32, 64]),
        ];
        for plan in &plans {
            for budget in [64, 256, 1376, 2048] {
                let mut state: Vec<u32> =
                    (plan.replay_begin - plan.replay_begin.min(W)..plan.replay_begin).collect();
                let mut begin = plan.replay_begin;
                for boundary in &plan.boundaries {
                    while begin < boundary.boundary {
                        let end = (begin + budget).min(boundary.boundary);
                        for capture in plan.dispatch(begin, end).expect("inside") {
                            if capture.reset {
                                state.clear();
                            }
                            state.extend(capture.begin..capture.end);
                        }
                        begin = end;
                    }
                    let rows = boundary.boundary.min(W) as usize;
                    assert!(state.len() >= rows, "budget {budget}: rows dropped");
                    let expected: Vec<u32> =
                        (boundary.boundary - rows as u32..boundary.boundary).collect();
                    assert_eq!(state[state.len() - rows..], expected, "budget {budget}");
                }
            }
        }
    }

    /// testRestoresDraftStateFlag.
    #[test]
    fn a_restore_continues_its_rings_exactly_within_a_window() {
        for (plan, restores) in [
            (active(4096, 4096 + W - 1), true),
            (active(4096, 4096 + W), false),
            (cached(4096, 9000, &[4096 + W - 1]), true),
            (cached(4096, 9000, &[4096 + W]), false),
        ] {
            assert_eq!(plan.restores_draft_state, restores);
            assert_eq!(plan.spans[0].reset, !restores);
        }
        for plan in [active(0, 1), active(0, W - 1), cached(0, 3 * W, &[W])] {
            assert!(!plan.restores_draft_state, "a cold plan restores nothing");
        }
    }

    /// testInvalidInputs.
    #[test]
    fn invalid_inputs_are_refused() {
        let plan = active(0, 4096);
        assert!(matches!(
            plan.dispatch(0, 4097),
            Err(DraftContextError::Dispatch { .. })
        ));
        assert_eq!(
            DraftContextPlan::of(0, 4096, &[2048, 1024], W),
            Err(DraftContextError::Unsorted {
                previous: 2048,
                boundary: 1024
            })
        );
    }
}

// SPDX-License-Identifier: Apache-2.0

//! Splash's draft context plan (~/git/splash @ 1482f6bb runtime/model/DraftContextPlan.cpp,
//! `planDraftContext` and `draftCaptureSpansForDispatch`; its structs in runtime/model/Model.hpp):
//! which of a prompt's rows its block drafter captures — reads its target's hidden states of, and
//! commits into its rings and context window — so that every state the prompt leaves holds the
//! drafter's context of the `window` rows before it, and no more rows are captured than that.
//!
//! A prompt replays its rows `[replay_begin, replay_end)` past a prefix-cache hit (`replay_begin`
//! 0 when cold). Its states are each boundary it saves a snapshot at (a materialization) and its
//! end (the active state its decode drafts from). A boundary's window is the `window` rows before
//! it. A state restored at `replay_begin` carries its own window: a boundary within `window` of the
//! last state continues that state's window, capturing from it; one farther captures its own
//! window afresh. `restores_draft_state` says whether the first boundary continues the restored
//! state's window, which the worker then rebuilds the rings from.
//!
//! Scratchy's one difference: a step's rows are not bounded by Splash's prefill token budget (one
//! window), so a step may cross more than two capture spans, and Splash's two-span array is a
//! vector.

/// Why a boundary has the drafter's context of the rows before it (Splash `DraftBoundaryPurpose`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryPurpose {
    /// A snapshot the prefix cache saves.
    Materialization,
    /// The prompt's end, which the sequence drafts from.
    Active,
}

/// A run of rows the drafter captures (Splash `DraftCaptureSpan`): `[begin, end)`, starting a fresh
/// window (`reset`) or continuing the one before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureSpan {
    pub begin: u32,
    pub end: u32,
    pub reset: bool,
}

/// A boundary and the first row its window captures (Splash `DraftBoundaryPlan`): `boundary` itself
/// when it is the restored state, which captures nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundaryPlan {
    pub boundary: u32,
    pub purpose: BoundaryPurpose,
    pub capture_begin: u32,
}

/// A prompt's draft context plan (Splash `DraftContextPlan`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DraftContextPlan {
    pub replay_begin: u32,
    pub replay_end: u32,
    pub capture_spans: Vec<CaptureSpan>,
    pub boundaries: Vec<BoundaryPlan>,
    /// The first capture continues the restored state's window: a replay from a hit whose first
    /// boundary is within one window of it. Otherwise the restore leaves the drafter's state alone.
    pub restores_draft_state: bool,
}

/// The rows of a capture span one step dispatches (Splash `DispatchDraftCaptureSpan`):
/// `[begin, end)`, the first at `compact_row` of the step's captured rows, whether it starts a
/// fresh window (only where its span does), and how many of them each purpose's windows take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchCapture {
    pub begin: u32,
    pub end: u32,
    pub compact_row: u32,
    pub reset: bool,
    pub active_rows: u32,
    pub materialization_rows: u32,
}

/// A plan or dispatch Splash refuses (its `std::invalid_argument`s).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftContextError {
    /// The replay ends before it begins.
    ReplayReversed,
    /// A materialization boundary lies outside `(replay_begin, replay_end]`.
    BoundaryOutsideReplay,
    /// The materialization boundaries are not strictly increasing.
    BoundariesUnsorted,
    /// The dispatch is reversed or reaches outside the replay.
    DispatchOutsideReplay,
}

/// Splash `planDraftContext`: the plan of a replay `[replay_begin, replay_end)` whose snapshots
/// save at `materializations` (strictly increasing, each within the replay), for a drafter whose
/// rings hold `window` positions.
pub fn plan_draft_context(
    replay_begin: u32,
    replay_end: u32,
    materializations: &[u32],
    window: u32,
) -> Result<DraftContextPlan, DraftContextError> {
    if replay_end < replay_begin {
        return Err(DraftContextError::ReplayReversed);
    }
    let mut previous = None;
    for &boundary in materializations {
        if boundary <= replay_begin || boundary > replay_end {
            return Err(DraftContextError::BoundaryOutsideReplay);
        }
        if previous.is_some_and(|p| boundary <= p) {
            return Err(DraftContextError::BoundariesUnsorted);
        }
        previous = Some(boundary);
    }

    let mut plan = DraftContextPlan {
        replay_begin,
        replay_end,
        ..DraftContextPlan::default()
    };
    let mut state_boundary = replay_begin;
    let mut have_state = replay_begin != 0;
    let mut add_boundary = |plan: &mut DraftContextPlan, boundary: u32, purpose| {
        let distance = boundary - state_boundary;
        let use_restored = have_state && distance == 0;
        let mut capture_begin = boundary;
        if !use_restored {
            let continue_state = have_state && distance < window;
            capture_begin = match continue_state {
                true => state_boundary,
                false => boundary - boundary.min(window),
            };
            let rows = boundary - capture_begin;
            match plan.capture_spans.last_mut() {
                Some(last) if rows != 0 && continue_state && last.end == capture_begin => {
                    last.end = boundary;
                }
                _ if rows != 0 => plan.capture_spans.push(CaptureSpan {
                    begin: capture_begin,
                    end: boundary,
                    reset: !continue_state,
                }),
                _ => {}
            }
        }
        plan.boundaries.push(BoundaryPlan {
            boundary,
            purpose,
            capture_begin,
        });
        state_boundary = boundary;
        have_state = true;
    };

    // A snapshot at the prompt's end is the active state itself: it is not planned twice.
    for &boundary in materializations.iter().filter(|&&b| b < replay_end) {
        add_boundary(&mut plan, boundary, BoundaryPurpose::Materialization);
    }
    add_boundary(&mut plan, replay_end, BoundaryPurpose::Active);

    let first_boundary = (materializations.first().copied())
        .filter(|&b| b < replay_end)
        .unwrap_or(replay_end);
    plan.restores_draft_state = replay_begin != 0 && first_boundary - replay_begin < window;
    Ok(plan)
}

/// Splash `draftCaptureSpansForDispatch`: the capture spans a step dispatching the plan's rows
/// `[begin, end)` runs, in order, each with how many of its rows each purpose's windows take.
pub fn capture_spans_for_dispatch(
    plan: &DraftContextPlan,
    begin: u32,
    end: u32,
) -> Result<Vec<DispatchCapture>, DraftContextError> {
    if end < begin || begin < plan.replay_begin || end > plan.replay_end {
        return Err(DraftContextError::DispatchOutsideReplay);
    }
    let mut out = Vec::new();
    let mut compact_row = 0;
    for span in &plan.capture_spans {
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
        for boundary in &plan.boundaries {
            let (segment_begin, segment_end) =
                (from.max(boundary.capture_begin), to.min(boundary.boundary));
            if segment_begin >= segment_end {
                continue;
            }
            let rows = match boundary.purpose {
                BoundaryPurpose::Active => &mut capture.active_rows,
                BoundaryPurpose::Materialization => &mut capture.materialization_rows,
            };
            *rows += segment_end - segment_begin;
        }
        debug_assert_eq!(
            capture.active_rows + capture.materialization_rows,
            to - from,
            "draft capture accounting is incomplete"
        );
        out.push(capture);
        compact_row += to - from;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    //! Splash's dev/tests/engine/draft_context_plan_test.cpp, transcribed, with a window of 2048
    //! (Splash's `draftContextTokens`). Its `testBenchmarkWorkIncludesRecoveryPoints` is not: it
    //! tests Splash's rolling prefill checkpoints (engine `kPrefillCheckpointTokens`), a policy of
    //! its engine scratchy's prefix cache does not have.
    use super::*;

    const WINDOW: u32 = 2048;

    fn active_plan(replay_begin: u32, replay_end: u32) -> DraftContextPlan {
        plan_draft_context(replay_begin, replay_end, &[], WINDOW).expect("a plan")
    }

    fn cached_plan(replay_begin: u32, replay_end: u32, boundaries: &[u32]) -> DraftContextPlan {
        plan_draft_context(replay_begin, replay_end, boundaries, WINDOW).expect("a plan")
    }

    /// Splash benchmarks' `draftContextRows`: every row the plan captures.
    fn draft_context_rows(plan: &DraftContextPlan) -> u32 {
        (plan.capture_spans.iter()).map(|s| s.end - s.begin).sum()
    }

    /// The rows a plan captures for the boundaries of one purpose.
    fn purpose_rows(plan: &DraftContextPlan, purpose: BoundaryPurpose) -> u32 {
        (plan.boundaries.iter())
            .filter(|b| b.purpose == purpose)
            .map(|b| b.boundary - b.capture_begin)
            .sum()
    }

    /// The capture spans that start a fresh window.
    fn resets(plan: &DraftContextPlan) -> usize {
        plan.capture_spans.iter().filter(|s| s.reset).count()
    }

    fn span(begin: u32, end: u32, reset: bool) -> CaptureSpan {
        CaptureSpan { begin, end, reset }
    }

    #[test]
    fn cold_lengths() {
        for length in [0, 1, 7, 8, 31, 32, 33, 2047, 2048, 2049, 4096, 10000] {
            let plan = active_plan(0, length);
            assert_eq!(
                draft_context_rows(&plan),
                length.min(2048),
                "length {length}"
            );
            assert_eq!(resets(&plan), usize::from(length != 0), "length {length}");
        }
    }

    #[test]
    fn partial_hit() {
        let short = active_plan(4096, 5000);
        assert_eq!(short.capture_spans, [span(4096, 5000, false)]);
        assert_eq!(draft_context_rows(&short), 904);
        assert!(
            short.restores_draft_state,
            "a short suffix continues its restored state"
        );

        let long = active_plan(4096, 8192);
        assert_eq!(long.capture_spans, [span(6144, 8192, true)]);
        assert_eq!(draft_context_rows(&long), 2048);
        assert!(
            !long.restores_draft_state,
            "a long suffix overwrites the state it would copy"
        );
    }

    #[test]
    fn final_full_block_bound() {
        let plan = cached_plan(0, 10000, &[9984]);
        assert_eq!(
            plan.capture_spans.len(),
            1,
            "nearby boundaries share one span"
        );
        assert_eq!(
            (plan.capture_spans[0].begin, plan.capture_spans[0].end),
            (7936, 10000)
        );
        assert_eq!(draft_context_rows(&plan), 2064);
        assert!(
            draft_context_rows(&plan) <= 2079,
            "at most a window and 31 rows"
        );

        // A prompt end on a 32-row page backs its snapshot off one whole block: keeping both that
        // state and the active one takes a window and 32 rows.
        assert_eq!(draft_context_rows(&cached_plan(0, 4096, &[4064])), 2080);
    }

    #[test]
    fn junction_distances_and_reservation() {
        assert!(
            cached_plan(4096, 7000, &[5000]).restores_draft_state,
            "a near first junction continues its restored state"
        );

        let near = cached_plan(0, 6000, &[4096]);
        assert_eq!(near.capture_spans, [span(2048, 6000, true)]);
        assert_eq!(purpose_rows(&near, BoundaryPurpose::Materialization), 2048);
        assert_eq!(purpose_rows(&near, BoundaryPurpose::Active), 1904);

        let far = cached_plan(0, 7000, &[4096]);
        assert_eq!(
            far.capture_spans,
            [span(2048, 4096, true), span(4952, 7000, true)],
            "a far junction's end captures a fresh window"
        );

        let denied = active_plan(0, 7000);
        assert_eq!(denied.capture_spans, [span(4952, 7000, true)]);
        assert_eq!(purpose_rows(&denied, BoundaryPurpose::Materialization), 0);
    }

    #[test]
    fn junction_and_latest_replay_are_both_materialized() {
        let plan = cached_plan(0, 10000, &[4096, 9984]);
        let purposes: Vec<_> = (plan.boundaries.iter())
            .map(|b| (b.boundary, b.purpose))
            .collect();
        assert_eq!(
            purposes,
            [
                (4096, BoundaryPurpose::Materialization),
                (9984, BoundaryPurpose::Materialization),
                (10000, BoundaryPurpose::Active),
            ]
        );
        assert_eq!(
            plan.capture_spans,
            [span(2048, 4096, true), span(7936, 10000, true)]
        );
        assert_eq!(purpose_rows(&plan, BoundaryPurpose::Materialization), 4096);
        assert_eq!(purpose_rows(&plan, BoundaryPurpose::Active), 16);
        assert_eq!(draft_context_rows(&plan), 4112);
    }

    #[test]
    fn compact_dispatch_captures() {
        let plan = cached_plan(0, 7000, &[4096]);
        let first = capture_spans_for_dispatch(&plan, 2000, 4000).expect("a dispatch");
        assert_eq!(
            first,
            [DispatchCapture {
                begin: 2048,
                end: 4000,
                compact_row: 0,
                reset: true,
                active_rows: 0,
                materialization_rows: 1952,
            }]
        );
        let middle = capture_spans_for_dispatch(&plan, 4000, 6048).expect("a dispatch");
        assert_eq!(
            middle,
            [
                DispatchCapture {
                    begin: 4000,
                    end: 4096,
                    compact_row: 0,
                    reset: false,
                    active_rows: 0,
                    materialization_rows: 96,
                },
                DispatchCapture {
                    begin: 4952,
                    end: 6048,
                    compact_row: 96,
                    reset: true,
                    active_rows: 1096,
                    materialization_rows: 0,
                },
            ]
        );
    }

    #[test]
    fn four_ragged_lanes() {
        let rows: u32 = [33, 2047, 2049, 10000]
            .map(|length| draft_context_rows(&active_plan(0, length)))
            .iter()
            .sum();
        assert_eq!(rows, 33 + 2047 + 2048 + 2048, "lanes share or pad no rows");
    }

    #[test]
    fn sparse_checkpoint_windows_and_short_resume() {
        let plan = cached_plan(0, 33001, &[8192, 16384, 24576, 32768, 32992]);
        assert_eq!((plan.capture_spans.len(), plan.boundaries.len()), (4, 6));
        assert_eq!(draft_context_rows(&plan), 4 * 2048 + 233);
        assert_eq!(
            purpose_rows(&plan, BoundaryPurpose::Materialization),
            4 * 2048 + 224
        );
        assert_eq!(purpose_rows(&plan, BoundaryPurpose::Active), 9);
        let mut dispatched = 0;
        for begin in (0..33001).step_by(1376) {
            let end = (begin + 1376).min(33001);
            for capture in capture_spans_for_dispatch(&plan, begin, end).expect("a dispatch") {
                dispatched += capture.end - capture.begin;
            }
        }
        assert_eq!(
            dispatched,
            draft_context_rows(&plan),
            "no capture lost or repeated"
        );

        for suffix in [1, 31, 32, 2047, 2048, 2049] {
            let resumed = active_plan(16384, 16384 + suffix);
            assert_eq!(
                draft_context_rows(&resumed),
                suffix.min(2048),
                "suffix {suffix}"
            );
            assert_eq!(
                resumed.restores_draft_state,
                suffix < 2048,
                "suffix {suffix}"
            );
        }
    }

    #[test]
    fn checkpoint_windows_across_small_dispatches() {
        let plans = [
            cached_plan(0, 12321, &[4096, 8192, 12288, 12320]),
            cached_plan(8192, 14337, &[8224, 10272, 14336]),
            cached_plan(0, 65, &[32, 64]),
        ];
        for plan in &plans {
            for budget in [64, 256, 1376, 2048] {
                // The rows the drafter's state holds, oldest first: the restored window, then what
                // each dispatch captures.
                let mut state: Vec<u32> = (plan.replay_begin - plan.replay_begin.min(WINDOW)
                    ..plan.replay_begin)
                    .collect();
                let mut begin = plan.replay_begin;
                for boundary in &plan.boundaries {
                    while begin < boundary.boundary {
                        let end = (begin + budget).min(boundary.boundary);
                        for capture in
                            capture_spans_for_dispatch(plan, begin, end).expect("a dispatch")
                        {
                            if capture.reset {
                                state.clear();
                            }
                            state.extend(capture.begin..capture.end);
                        }
                        begin = end;
                    }
                    let rows = boundary.boundary.min(WINDOW) as usize;
                    assert!(state.len() >= rows, "a checkpoint's window lost rows");
                    let expected: Vec<u32> =
                        (boundary.boundary - rows as u32..boundary.boundary).collect();
                    assert_eq!(
                        state[state.len() - rows..],
                        expected[..],
                        "a checkpoint's window holds stale or reordered rows"
                    );
                }
            }
        }
    }

    /// A restored plan continues the restored window, its first capture no reset, exactly when its
    /// first boundary is within one window of the restore; a cold plan restores none.
    #[test]
    fn restores_draft_state_flag() {
        for (plan, restores) in [
            (active_plan(4096, 4096 + WINDOW - 1), true),
            (active_plan(4096, 4096 + WINDOW), false),
            (cached_plan(4096, 9000, &[4096 + WINDOW - 1]), true),
            (cached_plan(4096, 9000, &[4096 + WINDOW]), false),
        ] {
            assert_eq!(plan.restores_draft_state, restores, "{plan:?}");
            assert_eq!(plan.capture_spans[0].reset, !restores, "{plan:?}");
        }
        for plan in [
            active_plan(0, 1),
            active_plan(0, WINDOW - 1),
            cached_plan(0, 3 * WINDOW, &[WINDOW]),
        ] {
            assert!(!plan.restores_draft_state, "{plan:?}");
        }
    }

    #[test]
    fn invalid_inputs() {
        let plan = active_plan(0, 4096);
        assert_eq!(
            capture_spans_for_dispatch(&plan, 0, 4097),
            Err(DraftContextError::DispatchOutsideReplay)
        );
        assert_eq!(
            plan_draft_context(0, 4096, &[2048, 1024], WINDOW),
            Err(DraftContextError::BoundariesUnsorted)
        );
        assert_eq!(
            plan_draft_context(4096, 2048, &[], WINDOW),
            Err(DraftContextError::ReplayReversed)
        );
        assert_eq!(
            plan_draft_context(1024, 4096, &[1024], WINDOW),
            Err(DraftContextError::BoundaryOutsideReplay)
        );
    }
}

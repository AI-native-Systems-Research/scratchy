// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Scheduler interface trait and pause state, ported from
//! `vllm/v1/core/sched/interface.py`.

use scratchy_core_common::{Request, RequestStatus};

use super::output::SchedulerOutput;

// ---------------------------------------------------------------------------
// PauseState
// ---------------------------------------------------------------------------

/// Scheduler pause state.
///
/// * `Unpaused` -- Normal operation; all requests are scheduled.
/// * `PausedNew` -- No new requests are scheduled; requests already in the
///   running state continue to be scheduled.
/// * `PausedAll` -- No requests are scheduled at all.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PauseState {
    #[default]
    Unpaused = 0,
    PausedNew = 1,
    PausedAll = 2,
}

// ---------------------------------------------------------------------------
// SchedulerInterface
// ---------------------------------------------------------------------------

/// The public interface that any scheduler implementation must satisfy.
///
/// Ported from the Python `SchedulerInterface(ABC)` in
/// `vllm/v1/core/sched/interface.py`.
pub trait SchedulerInterface {
    /// Schedule the requests to process in this scheduling step.
    ///
    /// The scheduling decision is made at the iteration level. Each
    /// scheduling step corresponds to a single forward pass of the model.
    /// The scheduler produces a mapping of `{req_id: num_tokens}` that
    /// specifies how many tokens to process for each request.
    fn schedule(&mut self) -> SchedulerOutput;

    /// Add a new request to the scheduler's internal queue.
    fn add_request(&mut self, request: Request);

    /// Finish (abort or stop) the given requests.
    ///
    /// Returns a list of `(request_id, client_index)` pairs for requests
    /// that were actually aborted (i.e., were not already finished).
    fn finish_requests(
        &mut self,
        request_ids: &[&str],
        finished_status: RequestStatus,
    ) -> Vec<(String, u32)>;

    /// Number of unfinished requests in the scheduler's internal queue.
    fn get_num_unfinished_requests(&self) -> usize;

    /// Collect the IDs of all unfinished requests (running + waiting).
    fn get_unfinished_request_ids(&self) -> Vec<String>;

    /// Returns `true` if there are unfinished requests.
    fn has_unfinished_requests(&self) -> bool {
        self.get_num_unfinished_requests() > 0
    }

    /// Returns `true` if there are finished requests that need to be cleared.
    ///
    /// The scheduler maintains an internal set of request IDs finished in the
    /// previous step. This set is returned from the next call to `schedule()`
    /// so that workers can free cached states for those requests.
    fn has_finished_requests(&self) -> bool;

    /// Returns `true` if there are either unfinished or pending-finished
    /// requests.
    fn has_requests(&self) -> bool {
        self.has_unfinished_requests() || self.has_finished_requests()
    }

    /// Current pause state of the scheduler.
    fn pause_state(&self) -> PauseState;

    /// Set the pause state of the scheduler.
    fn set_pause_state(&mut self, state: PauseState);

    /// Reset the prefix cache for KV cache.
    ///
    /// Returns `true` if the reset was successful (no running requests
    /// blocking the reset).
    fn reset_prefix_cache(&mut self) -> bool;

    /// Returns `(num_running_reqs, num_waiting_reqs)`.
    fn get_request_counts(&self) -> (usize, usize);

    /// KV cache usage as a fraction in `[0.0, 1.0]`.
    fn kv_cache_usage(&self) -> f64;

    /// Total number of GPU KV cache blocks.
    fn num_total_blocks(&self) -> usize;

    /// Number of GPU KV cache blocks currently in use.
    fn num_used_blocks(&self) -> usize;

    /// Shut down the scheduler.
    fn shutdown(&mut self);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pause_state_default() {
        assert_eq!(PauseState::default(), PauseState::Unpaused);
    }

    #[test]
    fn test_pause_state_repr() {
        assert_eq!(PauseState::Unpaused as u8, 0);
        assert_eq!(PauseState::PausedNew as u8, 1);
        assert_eq!(PauseState::PausedAll as u8, 2);
    }
}

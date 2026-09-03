// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Engine core client: the interface between the API server and the engine
//! core.
//!
//! The client abstracts over different communication modes:
//! - **In-process**: Direct function calls to `EngineCore` (no IPC).
//! - **Multi-process**: ZMQ sockets to a background `EngineCore` process.
//!
//! Port of: `vllm/v1/engine/core_client.py`

use std::collections::HashMap;

use scratchy_core_common::{EngineCoreOutputs, EngineCoreRequest, Request};
use scratchy_serving_scheduler::scheduler::output::SchedulerOutput;
use scratchy_serving_transport::messages::PauseMode;
use tracing::info;

use crate::engine_core::{EngineCore, EngineCoreConfig, StepOutcome, StepOutputs};
use crate::error::{EngineError, EngineResult};
use crate::executor::{Executor, ModelRunnerOutput};

// ---------------------------------------------------------------------------
// EngineCoreClient trait
// ---------------------------------------------------------------------------

/// The interface between the API server front-end and the engine core.
///
/// Subclasses handle different communication modes (in-process, ZMQ, etc.).
///
/// Port of: `vllm/v1/engine/core_client.py::EngineCoreClient`
pub trait EngineCoreClient {
    /// Get the output from the latest engine step.
    ///
    /// For the in-process client, this runs a step and returns the outputs.
    /// For the multi-process client, this receives outputs from ZMQ.
    ///
    /// See [`StepOutcome`] — a stall is a VARIANT, not a flag, so a step loop cannot reach the
    /// outputs without deciding what it does when nothing happened.
    fn get_output(&mut self) -> EngineResult<StepOutcome<EngineCoreOutputs>>;

    /// Add a new inference request.
    fn add_request(&mut self, request: EngineCoreRequest) -> EngineResult<()>;

    /// Abort requests by ID.
    fn abort_requests(&mut self, request_ids: &[String]) -> EngineResult<()>;

    /// Abort all currently running requests.
    ///
    /// Used when a persistent executor error (OOM, shape mismatch) makes it
    /// impossible to continue processing the current batch. Drains the
    /// scheduler's running queue and frees their KV blocks.
    fn abort_running_requests(&mut self);

    /// Shut down the engine.
    fn shutdown(&mut self) -> EngineResult<()>;

    /// Reset the prefix cache.
    fn reset_prefix_cache(&mut self) -> EngineResult<bool>;

    /// Pause the scheduler.
    fn pause_scheduler(&mut self, mode: PauseMode) -> EngineResult<()>;

    /// Resume the scheduler.
    fn resume_scheduler(&mut self) -> EngineResult<()>;

    /// Whether the scheduler is paused.
    fn is_scheduler_paused(&self) -> bool;

    /// Put the engine to sleep, freeing GPU memory.
    fn sleep(&mut self, level: u32) -> EngineResult<()>;

    /// Wake the engine from sleep.
    fn wake_up(&mut self, tags: Option<&[String]>) -> EngineResult<()>;

    /// Whether the engine is currently sleeping.
    fn is_sleeping(&self) -> bool;

    /// Compute embeddings for the given token ID sequences.
    ///
    /// Bypasses the scheduler — embedding is a single prefill pass with no
    /// KV cache or decode loop.
    fn embed(&mut self, _token_id_seqs: Vec<Vec<u32>>) -> EngineResult<Vec<Vec<f32>>> {
        Err(crate::error::EngineError::Executor(
            "embedding not supported".into(),
        ))
    }

    // -------------------------------------------------------------------
    // Async scheduling split ops (default: unsupported)
    // -------------------------------------------------------------------

    /// Take the executor out for use on a dedicated thread.
    fn take_executor(&mut self) -> Option<Box<dyn Executor>> {
        None
    }

    /// Post-execution processing: update state from model output.
    fn finalize_step(
        &mut self,
        _sched: &SchedulerOutput,
        _model: &ModelRunnerOutput,
    ) -> EngineResult<StepOutputs> {
        Err(crate::error::EngineError::Executor(
            "finalize_step not supported".into(),
        ))
    }

    /// Run scheduling if there is work to do.
    fn schedule_next(&mut self) -> EngineResult<Option<SchedulerOutput>> {
        Err(crate::error::EngineError::Executor(
            "schedule_next not supported".into(),
        ))
    }

    /// Whether async scheduling is enabled on the underlying engine.
    fn async_scheduling(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// InprocClient
// ---------------------------------------------------------------------------

/// In-process engine core client.
///
/// Directly calls methods on an `EngineCore` instance. When async scheduling
/// is enabled, spawns a background executor thread with a 2-batch pipeline
/// to overlap CPU scheduling with GPU execution (matching Python's
/// `EngineCore.step_with_batch_queue()`).
///
/// Port of: `vllm/v1/engine/core_client.py::InprocClient`
pub struct InprocClient {
    engine: EngineCore,
    /// Pipeline state for async scheduling. `None` when async scheduling is
    /// disabled (falls back to synchronous `step()`).
    pipeline: Option<PipelineState>,
}

/// Background executor thread state for pipelined execution.
struct PipelineState {
    sched_tx: std::sync::mpsc::SyncSender<PipelineMsg>,
    result_rx: std::sync::mpsc::Receiver<PipelineResult>,
    /// Deferred (sched, model_output) from the previous step, to be finalized
    /// at the start of the next `get_output()` call.
    deferred: Option<(SchedulerOutput, ModelRunnerOutput)>,
    /// Number of batches currently in-flight on the executor thread.
    gpu_in_flight: u32,
    _thread: std::thread::JoinHandle<()>,
}

/// A cloneable, Send+Sync handle for computing embeddings via the
/// background executor pipeline. Obtained from [`InprocClient::embed_sender`].
#[derive(Clone)]
pub struct EmbedSender {
    tx: std::sync::mpsc::SyncSender<PipelineMsg>,
}

impl EmbedSender {
    /// Compute embeddings for the given token ID sequences.
    /// Blocks until the executor thread returns the result.
    pub fn embed(&self, token_id_seqs: Vec<Vec<u32>>) -> EngineResult<Vec<Vec<f32>>> {
        let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
        self.tx
            .send(PipelineMsg::Embed(token_id_seqs, reply_tx))
            .map_err(|_| EngineError::Executor("executor thread exited".into()))?;
        reply_rx
            .recv()
            .map_err(|_| EngineError::Executor("executor thread exited".into()))?
    }
}

/// Messages sent to the background executor thread.
enum PipelineMsg {
    /// Execute a model step.
    Step(Box<SchedulerOutput>),
    /// Compute embeddings (bypasses scheduler).
    Embed(
        Vec<Vec<u32>>,
        std::sync::mpsc::SyncSender<EngineResult<Vec<Vec<f32>>>>,
    ),
}

/// Results from the background executor thread.
enum PipelineResult {
    Step(Box<SchedulerOutput>, EngineResult<ModelRunnerOutput>),
}

/// Background executor thread loop: receives scheduler outputs, runs
/// `execute_model`, and sends back results.
fn executor_bg_loop(
    mut executor: Box<dyn Executor>,
    rx: std::sync::mpsc::Receiver<PipelineMsg>,
    tx: std::sync::mpsc::SyncSender<PipelineResult>,
) {
    while let Ok(msg) = rx.recv() {
        match msg {
            PipelineMsg::Step(sched) => {
                let result = executor.execute_model(&sched);
                if tx.send(PipelineResult::Step(sched, result)).is_err() {
                    break;
                }
            }
            PipelineMsg::Embed(token_id_seqs, reply) => {
                let _ = reply.send(executor.embed(token_id_seqs));
            }
        }
    }
    executor.shutdown();
}

impl Drop for InprocClient {
    fn drop(&mut self) {
        // Deterministic pipeline teardown. The background executor thread
        // runs `executor.shutdown()` (→ worker teardown, which flushes the
        // one-time aligned-sidecar cache) only when its scheduler channel
        // closes. The thread handle was previously dropped without a join,
        // so a short-lived process (`scr chat -q …`) exited the instant
        // the blocking task returned and KILLED that teardown mid-write —
        // the cache never persisted, so every launch re-paid the cold
        // realign-copy. Drop the sender to end the loop, then JOIN so the
        // caller actually waits for shutdown to finish. (regression: c54955a3)
        if let Some(pipeline) = self.pipeline.take() {
            let PipelineState {
                sched_tx, _thread, ..
            } = pipeline;
            drop(sched_tx);
            let _ = _thread.join();
        }
    }
}

impl InprocClient {
    /// Create a new in-process client, initializing the engine core.
    ///
    /// When `async_scheduling` is enabled, the pipeline thread is started
    /// lazily on the first `get_output()` call. This allows the server path
    /// to call `take_executor()` first for its own background thread.
    pub fn new(config: EngineCoreConfig, executor: Box<dyn Executor>) -> Self {
        Self {
            engine: EngineCore::new(config, executor),
            pipeline: None,
        }
    }

    /// Start the background executor pipeline for overlapping CPU scheduling
    /// with GPU execution. Called explicitly by the LLM path.
    ///
    /// Does nothing if the pipeline is already started or the executor has
    /// been taken by the server path.
    pub fn start_pipeline(&mut self) {
        if self.pipeline.is_some() {
            return;
        }
        if let Some(executor) = self.engine.take_executor() {
            info!("InprocClient: spawning background executor thread for pipelined execution");
            let (sched_tx, sched_rx) = std::sync::mpsc::sync_channel::<PipelineMsg>(2);
            let (result_tx, result_rx) = std::sync::mpsc::sync_channel(2);
            let thread = std::thread::Builder::new()
                .name("scratchy-serving-worker".into())
                .spawn(move || executor_bg_loop(executor, sched_rx, result_tx))
                .expect("failed to spawn executor thread");
            self.pipeline = Some(PipelineState {
                sched_tx,
                result_rx,
                deferred: None,
                gpu_in_flight: 0,
                _thread: thread,
            });
        }
    }

    /// Get an [`EmbedSender`] for computing embeddings via the pipeline.
    /// Returns `None` if the pipeline has not been started.
    pub fn embed_sender(&self) -> Option<EmbedSender> {
        self.pipeline.as_ref().map(|p| EmbedSender {
            tx: p.sched_tx.clone(),
        })
    }

    /// Get a reference to the inner engine core.
    pub fn engine(&self) -> &EngineCore {
        &self.engine
    }

    /// Whether there are unfinished requests (including pipelined in-flight).
    pub fn has_unfinished_requests(&self) -> bool {
        if self.engine.has_unfinished_requests() {
            return true;
        }
        if let Some(ref pipeline) = self.pipeline
            && (pipeline.gpu_in_flight > 0 || pipeline.deferred.is_some())
        {
            return true;
        }
        false
    }

    /// Get a mutable reference to the inner engine core.
    pub fn engine_mut(&mut self) -> &mut EngineCore {
        &mut self.engine
    }

    /// Convert an `EngineCoreRequest` to a `Request` and add it.
    fn convert_and_add(&mut self, ec_request: EngineCoreRequest) -> EngineResult<()> {
        let params = ec_request.sampling_params.unwrap_or_default();
        let prompt_token_ids = ec_request.prompt_token_ids.unwrap_or_default();

        let mut request = Request::new(
            ec_request.request_id,
            prompt_token_ids,
            params,
            ec_request.arrival_time,
            ec_request.client_index,
            ec_request.priority,
            ec_request.cache_salt,
        );
        request.is_pooling = ec_request.is_pooling;
        request.mm_data = ec_request.mm_data;
        request.block_annotations = ec_request.block_annotations;
        request.seal = ec_request.seal;
        request.volatile = ec_request.volatile;

        self.engine.add_request(request);
        Ok(())
    }

    /// Pipelined get_output matching Python's `step_with_batch_queue`.
    ///
    /// 1. Finalize the deferred (previous) step's output.
    /// 2. Pre-schedule: fill pipeline up to 2 in-flight batches.
    /// 3. Block on the oldest GPU result, store as deferred.
    fn get_output_pipelined(
        engine: &mut EngineCore,
        pipeline: &mut PipelineState,
    ) -> EngineResult<StepOutcome<StepOutputs>> {
        // 1. Finalize previous deferred result. Resolve deferred D2H first
        //    (syncs the CUDA event and populates token IDs from pinned buffer).
        let mut prev_outputs: StepOutputs = HashMap::new();
        let mut had_prev = false;
        if let Some((prev_sched, mut prev_output)) = pipeline.deferred.take() {
            prev_output.resolve();
            prev_outputs = engine.finalize_step(&prev_sched, &prev_output);
            had_prev = true;
        }

        // 2. Pre-schedule: fill pipeline up to 2 in-flight batches.
        while pipeline.gpu_in_flight < 2 {
            if let Some(sched) = engine.schedule_next() {
                if pipeline
                    .sched_tx
                    .send(PipelineMsg::Step(Box::new(sched)))
                    .is_err()
                {
                    return Err(EngineError::Executor("executor thread exited".into()));
                }
                pipeline.gpu_in_flight += 1;
            } else {
                break;
            }
        }

        // 3. Block on oldest GPU result.
        if pipeline.gpu_in_flight > 0 {
            let PipelineResult::Step(sched_box, result) = pipeline
                .result_rx
                .recv()
                .map_err(|_| EngineError::Executor("executor thread exited".into()))?;
            let sched = *sched_box;
            let model_output = result?;
            pipeline.deferred = Some((sched, model_output));
            pipeline.gpu_in_flight -= 1;
        }

        // PROGRESS ON THIS PATH IS "the pipeline is turning": a batch was finalized, one is on the
        // device, or one is waiting to be finalized. None of those ⇒ the pipeline is empty while
        // requests are outstanding, i.e. `schedule_next` placed nothing — the same stall the
        // synchronous path names, reached by a different route.
        if had_prev || pipeline.gpu_in_flight > 0 || pipeline.deferred.is_some() {
            Ok(StepOutcome::Progressed {
                outputs: prev_outputs,
                model_executed: true,
            })
        } else if engine.has_unfinished_requests() {
            Ok(StepOutcome::Stalled)
        } else {
            Ok(StepOutcome::Idle)
        }
    }
}

impl EngineCoreClient for InprocClient {
    fn get_output(&mut self) -> EngineResult<StepOutcome<EngineCoreOutputs>> {
        // TTFT ATTRIBUTION (SCRATCHY_TTFT_PHASE): say WHICH path ran. The `[ttft-step]` breakdown
        // lives in `EngineCore::step`, so on the pipelined path it never prints and the absence of it
        // would otherwise look like the probe failing rather than a different code path being taken.
        let _ttft_phase = std::env::var_os("SCRATCHY_TTFT_PHASE").is_some();
        let _t_go = std::time::Instant::now();
        let _pipelined = self.pipeline.is_some();
        let outcome = if let Some(ref mut pipeline) = self.pipeline {
            // Pipelined path: overlap CPU scheduling with GPU execution.
            Self::get_output_pipelined(&mut self.engine, pipeline)?
        } else {
            // Synchronous path.
            self.engine.step()?
        };
        // A stall or an idle step carries no outputs to merge — forward the variant unchanged so
        // the CALLER decides, which is the whole reason this is an enum and not a flag.
        let StepOutcome::Progressed {
            outputs,
            model_executed,
        } = outcome
        else {
            return Ok(outcome.map(|_| EngineCoreOutputs::default()));
        };
        if _ttft_phase && model_executed && _t_go.elapsed().as_secs_f64() > 0.005 {
            eprintln!(
                "[ttft-step] get_output={:.2} ms  path={}",
                _t_go.elapsed().as_secs_f64() * 1e3,
                if _pipelined { "PIPELINED" } else { "sync" },
            );
        }

        // Take this client's outputs. Empty is not a fallback for a missing answer: it IS the
        // answer for a step that PROGRESSED without producing client output — one that retired a
        // finished request. `Progressed` already carries that distinction, so the empty set needs
        // no second encoding, and the `is_empty` pre-check this replaces was the same test twice.
        Ok(StepOutcome::Progressed {
            outputs: outputs.into_values().next().unwrap_or_default(),
            model_executed,
        })
    }

    fn add_request(&mut self, request: EngineCoreRequest) -> EngineResult<()> {
        self.convert_and_add(request)
    }

    fn abort_requests(&mut self, request_ids: &[String]) -> EngineResult<()> {
        self.engine.abort_requests(request_ids);
        Ok(())
    }

    fn abort_running_requests(&mut self) {
        self.engine.abort_running_requests();
    }

    fn shutdown(&mut self) -> EngineResult<()> {
        self.engine.shutdown();
        Ok(())
    }

    fn reset_prefix_cache(&mut self) -> EngineResult<bool> {
        Ok(self.engine.reset_prefix_cache())
    }

    fn pause_scheduler(&mut self, mode: PauseMode) -> EngineResult<()> {
        self.engine.pause_scheduler(mode)?;
        Ok(())
    }

    fn resume_scheduler(&mut self) -> EngineResult<()> {
        self.engine.resume_scheduler();
        Ok(())
    }

    fn is_scheduler_paused(&self) -> bool {
        self.engine.is_scheduler_paused()
    }

    fn sleep(&mut self, level: u32) -> EngineResult<()> {
        self.engine.sleep(level)
    }

    fn wake_up(&mut self, tags: Option<&[String]>) -> EngineResult<()> {
        self.engine.wake_up(tags)
    }

    fn is_sleeping(&self) -> bool {
        self.engine.is_sleeping()
    }

    fn embed(&mut self, token_id_seqs: Vec<Vec<u32>>) -> EngineResult<Vec<Vec<f32>>> {
        if let Some(ref pipeline) = self.pipeline {
            // Route through the background executor thread.
            let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
            pipeline
                .sched_tx
                .send(PipelineMsg::Embed(token_id_seqs, reply_tx))
                .map_err(|_| EngineError::Executor("executor thread exited".into()))?;
            reply_rx
                .recv()
                .map_err(|_| EngineError::Executor("executor thread exited".into()))?
        } else {
            self.engine.embed(token_id_seqs)
        }
    }

    fn take_executor(&mut self) -> Option<Box<dyn Executor>> {
        self.engine.take_executor()
    }

    fn finalize_step(
        &mut self,
        sched: &SchedulerOutput,
        model: &ModelRunnerOutput,
    ) -> EngineResult<StepOutputs> {
        Ok(self.engine.finalize_step(sched, model))
    }

    fn schedule_next(&mut self) -> EngineResult<Option<SchedulerOutput>> {
        Ok(self.engine.schedule_next())
    }

    fn async_scheduling(&self) -> bool {
        self.engine.async_scheduling()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::NoopExecutor;
    use scratchy_core_common::SamplingParams;
    use scratchy_core_config::{SchedulerConfig, SchedulerPolicy};

    fn make_test_config() -> EngineCoreConfig {
        EngineCoreConfig {
            scheduler_config: SchedulerConfig {
                max_num_batched_tokens: 8192,
                max_num_seqs: 256,
                max_num_scheduled_tokens: None,
                policy: SchedulerPolicy::Fcfs,
                enable_chunked_prefill: true,
                long_prefill_token_threshold: 0,
                ..Default::default()
            },
            max_model_len: 4096,
            num_gpu_blocks: 1024,
            block_size: 16,
            engine_index: 0,
            async_scheduling: false,
            use_spec_decode: false,
            proposer_config: None,
            eos_token_ids: vec![],
            is_pooling: false,
            enable_prefix_caching: false,
            hybrid_kv: None,
        }
    }

    fn make_ec_request(id: &str, num_tokens: usize) -> EngineCoreRequest {
        EngineCoreRequest {
            request_id: id.to_string(),
            prompt_token_ids: Some((0..num_tokens as u32).collect()),
            sampling_params: Some(SamplingParams {
                max_tokens: Some(16),
                ..Default::default()
            }),
            arrival_time: 0.0,
            client_index: 0,
            priority: 0,
            cache_salt: None,
            data_parallel_rank: None,
            is_pooling: false,
            mm_data: None,
            block_annotations: None,
            seal: false,
            volatile: false,
        }
    }

    #[test]
    fn test_inproc_client_new() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let client = InprocClient::new(config, executor);
        assert!(!client.engine().is_shutdown());
    }

    #[test]
    fn test_inproc_add_and_get_output() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut client = InprocClient::new(config, executor);

        // Add a request.
        client.add_request(make_ec_request("req-1", 10)).unwrap();

        // Get output (triggers a step).
        let (outputs, model_executed) = client.get_output().unwrap().into_tuple();
        assert!(!outputs.outputs.is_empty());
        assert!(model_executed);
    }

    #[test]
    fn test_inproc_empty_output() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut client = InprocClient::new(config, executor);

        // No requests, should return empty outputs.
        let (outputs, model_executed) = client.get_output().unwrap().into_tuple();
        assert!(outputs.outputs.is_empty());
        assert!(!model_executed);
    }

    #[test]
    fn test_inproc_abort() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut client = InprocClient::new(config, executor);

        client.add_request(make_ec_request("req-1", 10)).unwrap();
        client.add_request(make_ec_request("req-2", 10)).unwrap();

        client.abort_requests(&["req-1".to_string()]).unwrap();

        assert_eq!(client.engine().num_unfinished_requests(), 1);
    }

    #[test]
    fn test_inproc_pause_resume() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut client = InprocClient::new(config, executor);

        assert!(!client.is_scheduler_paused());

        client.pause_scheduler(PauseMode::Keep).unwrap();
        assert!(client.is_scheduler_paused());

        client.resume_scheduler().unwrap();
        assert!(!client.is_scheduler_paused());
    }

    #[test]
    fn test_inproc_reset_prefix_cache() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut client = InprocClient::new(config, executor);

        assert!(client.reset_prefix_cache().unwrap());
    }

    #[test]
    fn test_inproc_shutdown() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut client = InprocClient::new(config, executor);

        client.shutdown().unwrap();
        assert!(client.engine().is_shutdown());
    }

    #[test]
    fn test_inproc_multiple_steps() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut client = InprocClient::new(config, executor);

        client.add_request(make_ec_request("req-1", 10)).unwrap();

        // First step should produce output.
        let (outputs, _) = client.get_output().unwrap().into_tuple();
        assert!(!outputs.outputs.is_empty());

        // Subsequent steps should not error (request may or may not produce
        // output depending on scheduler state).
        for _ in 0..4 {
            let _ = client.get_output().unwrap();
        }
    }

    #[test]
    fn test_inproc_multiple_requests() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut client = InprocClient::new(config, executor);

        for i in 0..10 {
            client
                .add_request(make_ec_request(&format!("req-{i}"), 10))
                .unwrap();
        }

        let (outputs, _) = client.get_output().unwrap().into_tuple();
        assert!(outputs.outputs.len() >= 10);
    }

    #[test]
    fn test_trait_object() {
        // Ensure InprocClient can be used as a trait object.
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut client: Box<dyn EngineCoreClient> = Box::new(InprocClient::new(config, executor));

        client.add_request(make_ec_request("req-1", 10)).unwrap();
        let (outputs, _) = client.get_output().unwrap().into_tuple();
        assert!(!outputs.outputs.is_empty());
        client.shutdown().unwrap();
    }
}

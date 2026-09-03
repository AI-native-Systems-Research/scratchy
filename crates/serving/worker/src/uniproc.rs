// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Single-process executor.
//!
//! The `UniProcExecutor` runs a single worker in the same process as the
//! engine core. Method calls are dispatched directly to the worker without
//! any IPC overhead.
//!
//! Port of: `vllm/v1/executor/uniproc_executor.py::UniProcExecutor`

use scratchy_serving_engine::error::EngineResult;
use scratchy_serving_engine::executor::{Executor, ModelRunnerOutput};
use scratchy_serving_scheduler::scheduler::output::SchedulerOutput;
use tracing::info;

use crate::error::ExecutorResult;
use crate::worker::Worker;

// ---------------------------------------------------------------------------
// UniProcExecutor
// ---------------------------------------------------------------------------

/// Single-process executor that wraps a single worker.
///
/// All method calls are dispatched directly to the underlying worker.
/// This is the simplest executor, suitable for single-GPU inference.
///
/// Port of: `vllm/v1/executor/uniproc_executor.py::UniProcExecutor`
pub struct UniProcExecutor {
    /// The underlying worker.
    worker: Box<dyn Worker>,
    /// Whether the executor is sleeping.
    is_sleeping: bool,
    /// Sleeping resource tags.
    sleeping_tags: Vec<&'static str>,
    /// Whether the executor has been shut down.
    is_shutdown: bool,
}

impl UniProcExecutor {
    /// Create a new `UniProcExecutor` from a worker.
    ///
    /// The worker should already be constructed but may not yet be
    /// initialized. Call `initialize()` to run the full initialization
    /// sequence (init_device → load_model → compile_or_warm_up).
    pub fn new(worker: Box<dyn Worker>) -> Self {
        Self {
            worker,
            is_sleeping: false,
            sleeping_tags: Vec::new(),
            is_shutdown: false,
        }
    }

    /// Run the full worker initialization sequence.
    ///
    /// 1. Initialize device
    /// 2. Load model
    /// 3. Compile/warm up model
    pub fn initialize(&mut self) -> ExecutorResult<()> {
        info!(
            "UniProcExecutor: initializing worker (rank {})",
            self.worker.rank()
        );

        self.worker.init_device()?;
        self.worker.load_model()?;
        self.worker.compile_or_warm_up_model()?;

        info!("UniProcExecutor: worker initialized successfully");
        Ok(())
    }

    /// Create a `UniProcExecutor` wrapping a worker that has already been
    /// initialized (init_device + load_model completed). Skips the
    /// `initialize()` sequence.
    pub fn new_pre_initialized(worker: Box<dyn Worker>) -> Self {
        Self {
            worker,
            is_sleeping: false,
            sleeping_tags: Vec::new(),
            is_shutdown: false,
        }
    }

    /// Get a reference to the underlying worker.
    pub fn worker(&self) -> &dyn Worker {
        &*self.worker
    }

    /// Get a mutable reference to the underlying worker.
    pub fn worker_mut(&mut self) -> &mut dyn Worker {
        &mut *self.worker
    }
}

impl Executor for UniProcExecutor {
    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> EngineResult<ModelRunnerOutput> {
        self.worker
            .execute_model(scheduler_output)
            .map_err(|e| scratchy_serving_engine::error::EngineError::Executor(e.to_string()))
    }

    fn spec_decode_backend(
        &mut self,
    ) -> Option<&mut dyn scratchy_serving_engine::spec_decode::SpecDecodeBackend> {
        self.worker.spec_decode_backend()
    }

    /// Straight through to the worker — it is the thing that knows whether its batched step appends every
    /// row at one shared slot, and the scheduler needs the answer before the first allocation.
    fn kv_addressing(&self) -> scratchy_core_common::KvAddressing {
        self.worker.kv_addressing()
    }

    fn initialize_cache(
        &mut self,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
    ) -> EngineResult<()> {
        self.worker
            .initialize_cache(num_gpu_blocks, num_cpu_blocks)
            .map_err(|e| scratchy_serving_engine::error::EngineError::Executor(e.to_string()))
    }

    fn determine_available_memory(&mut self) -> EngineResult<Vec<usize>> {
        let memory = self
            .worker
            .determine_available_memory()
            .map_err(|e| scratchy_serving_engine::error::EngineError::Executor(e.to_string()))?;
        Ok(vec![memory])
    }

    fn check_health(&self) -> EngineResult<()> {
        self.worker
            .check_health()
            .map_err(|e| scratchy_serving_engine::error::EngineError::Executor(e.to_string()))
    }

    fn sleep(&mut self, level: u32) -> EngineResult<()> {
        if self.is_sleeping {
            return Ok(());
        }
        self.worker
            .sleep(level)
            .map_err(|e| scratchy_serving_engine::error::EngineError::Executor(e.to_string()))?;
        self.sleeping_tags = vec!["weights", "kv_cache"];
        self.is_sleeping = true;
        Ok(())
    }

    fn wake_up(&mut self, tags: Option<&[String]>) -> EngineResult<()> {
        if !self.is_sleeping {
            return Ok(());
        }
        self.worker
            .wake_up(tags)
            .map_err(|e| scratchy_serving_engine::error::EngineError::Executor(e.to_string()))?;

        if let Some(tags) = tags {
            self.sleeping_tags
                .retain(|t| !tags.iter().any(|s| s.as_str() == *t));
        } else {
            self.sleeping_tags.clear();
        }
        if self.sleeping_tags.is_empty() {
            self.is_sleeping = false;
        }
        Ok(())
    }

    fn is_sleeping(&self) -> bool {
        self.is_sleeping
    }

    fn embed(&mut self, token_id_seqs: Vec<Vec<u32>>) -> EngineResult<Vec<Vec<f32>>> {
        let refs: Vec<&[u32]> = token_id_seqs.iter().map(|v| v.as_slice()).collect();
        self.worker
            .embed(&refs)
            .map_err(|e| scratchy_serving_engine::error::EngineError::Executor(e.to_string()))
    }

    fn shutdown(&mut self) {
        if self.is_shutdown {
            return;
        }
        info!("UniProcExecutor: shutting down");
        self.worker.shutdown();
        self.is_shutdown = true;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::NoopWorker;
    use std::collections::HashMap;

    fn make_test_executor() -> UniProcExecutor {
        let worker = Box::new(NoopWorker::with_defaults(16 * 1024 * 1024));
        UniProcExecutor::new(worker)
    }

    #[test]
    fn test_uniproc_new() {
        let exec = make_test_executor();
        assert!(!exec.is_sleeping());
        assert!(!exec.is_shutdown);
        assert_eq!(exec.worker().rank(), 0);
        assert!(exec.worker().is_driver_worker());
    }

    #[test]
    fn test_uniproc_initialize() {
        let mut exec = make_test_executor();
        exec.initialize().unwrap();
    }

    #[test]
    fn test_uniproc_execute_model() {
        let mut exec = make_test_executor();
        exec.initialize().unwrap();
        exec.initialize_cache(512, 0).unwrap();

        let mut num_scheduled = HashMap::new();
        num_scheduled.insert("req-1".to_string(), 100);
        num_scheduled.insert("req-2".to_string(), 50);

        let sched_output = SchedulerOutput {
            num_scheduled_tokens: num_scheduled,
            total_num_scheduled_tokens: 150,
            ..SchedulerOutput::make_empty()
        };

        let output = exec.execute_model(&sched_output).unwrap();
        assert_eq!(output.num_requests(), 2);
        assert!(output.get_tokens("req-1").is_some());
        assert!(output.get_tokens("req-2").is_some());
    }

    #[test]
    fn test_uniproc_determine_memory() {
        let mut exec = make_test_executor();
        exec.initialize().unwrap();

        let memory = exec.determine_available_memory().unwrap();
        assert_eq!(memory.len(), 1);
        assert_eq!(memory[0], 16 * 1024 * 1024);
    }

    #[test]
    fn test_uniproc_health_check() {
        let exec = make_test_executor();
        exec.check_health().unwrap();
    }

    #[test]
    fn test_uniproc_sleep_wake() {
        let mut exec = make_test_executor();
        assert!(!exec.is_sleeping());

        exec.sleep(1).unwrap();
        assert!(exec.is_sleeping());

        // Double sleep is a no-op.
        exec.sleep(1).unwrap();
        assert!(exec.is_sleeping());

        // Wake up specific tags.
        exec.wake_up(Some(&["weights".to_string()])).unwrap();
        assert!(exec.is_sleeping()); // Still sleeping (kv_cache tag remains).

        exec.wake_up(Some(&["kv_cache".to_string()])).unwrap();
        assert!(!exec.is_sleeping()); // Fully awake now.

        // Wake when not sleeping is a no-op.
        exec.wake_up(None).unwrap();
    }

    #[test]
    fn test_uniproc_sleep_wake_all() {
        let mut exec = make_test_executor();

        exec.sleep(1).unwrap();
        assert!(exec.is_sleeping());

        // Wake all at once.
        exec.wake_up(None).unwrap();
        assert!(!exec.is_sleeping());
    }

    #[test]
    fn test_uniproc_shutdown() {
        let mut exec = make_test_executor();
        exec.shutdown();
        assert!(exec.is_shutdown);

        // Double shutdown is safe.
        exec.shutdown();
        assert!(exec.is_shutdown);
    }

    #[test]
    fn test_uniproc_sequential_execution() {
        let mut exec = make_test_executor();
        exec.initialize().unwrap();

        let mut num_scheduled = HashMap::new();
        num_scheduled.insert("req-1".to_string(), 10);

        let sched_output = SchedulerOutput {
            num_scheduled_tokens: num_scheduled,
            total_num_scheduled_tokens: 10,
            ..SchedulerOutput::make_empty()
        };

        let out1 = exec.execute_model(&sched_output).unwrap();
        let out2 = exec.execute_model(&sched_output).unwrap();

        let t1 = out1.get_tokens("req-1").unwrap()[0];
        let t2 = out2.get_tokens("req-1").unwrap()[0];
        assert_ne!(t1, t2);
        assert_eq!(t2, t1 + 1);
    }

    #[test]
    fn test_uniproc_with_engine_core() {
        // Verify UniProcExecutor works as a drop-in for EngineCore.
        use scratchy_core_config::{SchedulerConfig, SchedulerPolicy};
        use scratchy_serving_engine::engine_core::{EngineCore, EngineCoreConfig, StepOutcome};

        let worker = Box::new(NoopWorker::with_defaults(16 * 1024 * 1024));
        let mut exec = UniProcExecutor::new(worker);
        exec.initialize().unwrap();

        let config = EngineCoreConfig {
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
        };

        let mut engine = EngineCore::new(config, Box::new(exec));

        // Add a request and step.
        let req = scratchy_core_common::Request::new(
            "test-1".to_string(),
            (0..10u32).collect(),
            scratchy_core_common::SamplingParams {
                max_tokens: Some(16),
                ..Default::default()
            },
            0.0,
            0,
            0,
            None,
        );
        engine.add_request(req);

        // A stall here would BE the failure, so assert the variant rather than unwrap a flag.
        match engine.step().unwrap() {
            StepOutcome::Progressed {
                outputs,
                model_executed,
            } => {
                assert!(model_executed);
                assert!(!outputs.is_empty());
            }
            other => panic!("expected the step to run the model, got {other:?}"),
        }

        engine.shutdown();
    }
}

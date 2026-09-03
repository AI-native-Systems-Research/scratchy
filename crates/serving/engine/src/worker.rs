// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Worker trait defining the interface for device-level execution.
//!
//! A worker is responsible for a single device (GPU). It handles:
//! - Device initialization
//! - Model loading
//! - KV cache initialization
//! - Model forward pass execution
//! - Memory profiling
//!
//! Port of: `vllm/v1/worker/worker_base.py::WorkerBase`

use std::collections::HashMap;

use crate::executor::ModelRunnerOutput;
use scratchy_serving_scheduler::scheduler::output::SchedulerOutput;

use crate::error::ExecutorResult;

// ---------------------------------------------------------------------------
// Worker trait
// ---------------------------------------------------------------------------

/// Configuration for initializing a worker.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Local device index (e.g., GPU 0, 1, 2, ...).
    pub local_rank: usize,
    /// Global rank in the distributed group.
    pub rank: usize,
    /// Whether this is the driver worker (rank 0 of TP group).
    pub is_driver_worker: bool,
    /// Distributed initialization method (e.g., "tcp://host:port").
    pub distributed_init_method: String,
    /// Optional HF token for gated repos, forwarded to `resolve_model_path`.
    pub hf_token: Option<String>,
    /// Optional GGUF file name for HF Hub download, forwarded to `resolve_model_path`.
    pub gguf_file: Option<String>,
}

/// Trait defining the worker interface.
///
/// Workers are the device-level execution units. Each worker manages one
/// device (GPU) and handles model execution, KV cache operations, and
/// memory management.
///
/// Port of: `vllm/v1/worker/worker_base.py::WorkerBase`
pub trait Worker: Send {
    /// Initialize the device (e.g., set CUDA device, init distributed).
    fn init_device(&mut self) -> ExecutorResult<()>;

    /// Load the model onto the device.
    fn load_model(&mut self) -> ExecutorResult<()>;

    /// Initialize KV cache from configuration.
    ///
    /// Called after memory profiling to set up the actual KV cache
    /// with the determined number of blocks.
    fn initialize_cache(
        &mut self,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
    ) -> ExecutorResult<()>;

    /// Determine available memory for KV cache (in bytes).
    ///
    /// This typically involves a memory profiling step where a dummy
    /// forward pass is run and the remaining memory is measured.
    fn determine_available_memory(&mut self) -> ExecutorResult<usize>;

    /// Execute the model for one step.
    ///
    /// Takes the scheduler output describing which requests to process
    /// and returns the generated tokens and optional logprobs.
    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput>;

    /// Compile or warm up the model for inference.
    ///
    /// This may include CUDA graph capture, JIT compilation, etc.
    fn compile_or_warm_up_model(&mut self) -> ExecutorResult<()> {
        Ok(())
    }

    /// Largest prefill bucket the device can afford, if the backend prunes a
    /// compiled bucket ladder target-reactively (see `select_prefill_bucket`).
    /// The engine clamps `SchedulerConfig::max_num_batched_tokens` to this so a
    /// single forward never exceeds the largest resident bucket (which would
    /// otherwise hit `NoBucketFits` mid-run). `None` = no cap (keep the
    /// configured value); valid only after `determine_available_memory`.
    fn prefill_bucket_max_m(&self) -> Option<u32> {
        None
    }

    /// Maximum sequence length (tokens) this backend's paged-attention kernels
    /// can ADDRESS, if smaller than the model's `max_position_embeddings`. The
    /// per-sequence block table is baked at a fixed row stride
    /// (`MAX_BLOCKS_PER_SEQ`); the smallest-block-size KV group (the sliding
    /// class on a hybrid SWA arch) binds the limit at
    /// `MAX_BLOCKS_PER_SEQ × block_size`. The engine caps `max_model_len` to
    /// this (so an over-long prompt is rejected, not silently truncated by the
    /// kernel) AND sizes the KV pool to it instead of the full memory budget —
    /// the hybrid pool is lazy + window-bounded, so reserving the whole budget
    /// of residency-attached VA just starves the activation arena. `None` = no
    /// limit (the model's full context fits the kernel). Valid after
    /// `load_model`.
    fn kv_max_addressable_tokens(&self) -> Option<usize> {
        None
    }

    /// HARD CAP on the scheduler's total `num_gpu_blocks`, if the backend's KV
    /// pool is a FIXED-SIZE resident allocation rather than a memory-budget-derived
    /// one. The uniform `compute_num_blocks` sizes `num_gpu_blocks` from the memory
    /// budget (`determine_available_memory`), which can exceed a backend whose pool
    /// is a baked, fixed number of physical blocks. The spyre/sendnn PAGED path has
    /// a resident on-card KV pool of exactly `nblk` blocks of `block_size`=64 slots;
    /// the scheduler's block allocator MUST NOT hand out a block id ≥ `nblk` (it
    /// would index past the pool → silent KV corruption / on-card fault). When this
    /// returns `Some(n)` the engine sets `num_gpu_blocks = min(computed, n)`. `None`
    /// = no cap (cuda/metal size the pool from the memory budget). Valid after
    /// `load_model`.
    fn kv_cache_num_blocks_override(&self) -> Option<usize> {
        None
    }

    /// HARD CAP on the scheduler's `max_num_seqs`, if the backend can only DECODE a bounded number of
    /// sequences in one batched step.
    ///
    /// ⛔ IT EXISTS BECAUSE cuda's LADDER IS DERIVED AND SPYRE'S IS BAKED. Both backends pick a decode
    /// width from a discrete ladder (cuda: captured graph sizes, spyre: `BATCH_RUNGS`), both take the
    /// smallest entry `>= live`, and both fall back to a slower correct path when nothing is wide enough.
    /// cuda holds the invariant *"the fast path always covers the configured concurrency"* by building its
    /// ladder FROM `max_num_seqs` (`auto_capture_sizes`), so no configured width can miss it. A backend
    /// whose ladder is fixed at BAKE time cannot grow it to meet `max_num_seqs`, so it holds the same
    /// invariant from the other end: it reports the widest width it can batch and the scheduler admits no
    /// more than that.
    ///
    /// ⭐ CAPPING, NOT REFUSING — cuda refuses no `max_num_seqs` and neither should anyone else. Exceeding
    /// the ladder is a THROUGHPUT cliff, not a correctness bound: spyre's own fallback is one launch per
    /// request (~93 µs each) and stays correct. The cap keeps a run on the batched path instead of
    /// silently dropping it onto the slow one; it is not protecting against corruption.
    ///
    /// `None` = no cap (cuda/metal, whose ladders already track `max_num_seqs`). Valid after `load_model`.
    fn max_num_seqs_override(&self) -> Option<usize> {
        None
    }

    /// ⭐⭐⭐ HOW THIS WORKER ADDRESSES ITS KV — a CAPABILITY, declared before it runs a step.
    ///
    /// `ByToken` (the default) is every backend whose key `n` sits at slot `n`: cuda, metal. A worker whose
    /// batched step appends EVERY row at one shared slot returns `OneSharedWriteSlot`, and the scheduler then
    /// sizes each request by the step's reach rather than by its own token count — which is what lets the host
    /// own every page the launch writes instead of the worker covering them from a reserve of its own.
    ///
    /// ⛔ IT LIVES HERE, BESIDE `kv_cache_num_blocks_override`, BECAUSE IT IS THE SAME KIND OF FACT: a
    /// property of the backend, known at `load_model` time. It was previously inferred from whether the
    /// worker had ever REPORTED a pool reach — a value that first exists after a step has been scheduled and
    /// allocated, so the answer arrived one step late and every allocation before it used the wrong rule.
    fn kv_addressing(&self) -> scratchy_core_common::KvAddressing {
        scratchy_core_common::KvAddressing::ByToken
    }

    /// Whether this worker's DECODE path implements the hybrid sliding-window
    /// KV layout — separate paged cache groups per sliding window, addressed by
    /// the `sliding_groups` per-request block tables. Metal wires this; the CUDA
    /// decode currently reads a single block table for EVERY layer, so it must
    /// run on the uniform single-pool layout instead (correct for any context
    /// up to the sliding window). When this returns false the scheduler keeps
    /// gemma4 (and other hybrid-SWA arches) on the uniform pool so the worker's
    /// single-block-table decode never reads a sliding layer's KV from the
    /// global group's blocks. Default true (metal + uniform arches).
    fn supports_hybrid_swa_kv(&self) -> bool {
        true
    }

    /// Check whether the worker is healthy.
    fn check_health(&self) -> ExecutorResult<()> {
        Ok(())
    }

    /// Put the worker to sleep, freeing device memory.
    ///
    /// `level` controls aggressiveness: 1 = free KV cache, 2 = free model weights.
    fn sleep(&mut self, _level: u32) -> ExecutorResult<()> {
        Ok(())
    }

    /// Wake the worker from sleep, restoring device memory.
    ///
    /// `tags` specifies which resources to restore. `None` = restore all.
    fn wake_up(&mut self, _tags: Option<&[String]>) -> ExecutorResult<()> {
        Ok(())
    }

    /// Compute embeddings for the given token ID sequences.
    ///
    /// Each inner slice is a single input to embed. Returns one embedding
    /// vector (as `Vec<f32>`) per input.
    ///
    /// Default implementation returns an error — override in workers that
    /// support embedding.
    fn embed(&mut self, _token_id_seqs: &[&[u32]]) -> ExecutorResult<Vec<Vec<f32>>> {
        Err(crate::error::ExecutorError::WorkerExecution(
            "embedding not supported".into(),
        ))
    }

    /// Take the pre-loaded tokenizer, if one was loaded during `load_model()`.
    ///
    /// Workers that support parallel tokenizer loading will parse `tokenizer.json`
    /// on a background thread during weight loading. This method retrieves (and
    /// consumes) that tokenizer so the caller can avoid a redundant load.
    fn take_preloaded_tokenizer(&mut self) -> Option<tokenizers::Tokenizer> {
        None
    }

    /// Return the resolved model architecture name (e.g. "LlamaForCausalLM").
    ///
    /// Available after `load_model()` has been called.
    fn architecture(&self) -> Option<String> {
        None
    }

    /// Mutable access to the worker's spec-decode backend. Default
    /// returns `None`; metal `MetalWorker` overrides to return
    /// `Some(&mut *self as &mut dyn SpecDecodeBackend)` so the
    /// engine-side `DraftModelProposer` can issue lockstep prefill +
    /// K-step decode calls against it.
    fn spec_decode_backend(&mut self) -> Option<&mut dyn crate::spec_decode::SpecDecodeBackend> {
        None
    }

    /// Shut down the worker and release all resources.
    fn shutdown(&mut self);

    /// The worker's global rank.
    fn rank(&self) -> usize;

    /// The worker's local device rank.
    fn local_rank(&self) -> usize;

    /// Whether this is the driver worker.
    fn is_driver_worker(&self) -> bool;
}

// ---------------------------------------------------------------------------
// NoopWorker -- a minimal worker for testing
// ---------------------------------------------------------------------------

/// A no-op worker for testing executor logic without a real GPU.
///
/// Generates sequential dummy token IDs and reports fake available memory.
pub struct NoopWorker {
    config: WorkerConfig,
    /// Fake available memory in bytes.
    available_memory: usize,
    /// Next token ID to generate (increments per request per step).
    next_token_id: u32,
    /// Whether the worker is initialized.
    initialized: bool,
    /// Whether the worker has been shut down.
    is_shutdown: bool,
}

impl NoopWorker {
    /// Create a new no-op worker.
    pub fn new(config: WorkerConfig, available_memory: usize) -> Self {
        Self {
            config,
            available_memory,
            next_token_id: 1000,
            initialized: false,
            is_shutdown: false,
        }
    }

    /// Create a no-op worker with default config (rank 0, driver).
    pub fn with_defaults(available_memory: usize) -> Self {
        Self::new(
            WorkerConfig {
                local_rank: 0,
                rank: 0,
                is_driver_worker: true,
                distributed_init_method: "tcp://localhost:0".to_string(),
                hf_token: None,
                gguf_file: None,
            },
            available_memory,
        )
    }
}

impl Worker for NoopWorker {
    fn init_device(&mut self) -> ExecutorResult<()> {
        self.initialized = true;
        Ok(())
    }

    fn load_model(&mut self) -> ExecutorResult<()> {
        if !self.initialized {
            return Err(crate::error::ExecutorError::WorkerInit(
                "device not initialized".to_string(),
            ));
        }
        Ok(())
    }

    fn initialize_cache(
        &mut self,
        _num_gpu_blocks: usize,
        _num_cpu_blocks: usize,
    ) -> ExecutorResult<()> {
        Ok(())
    }

    fn determine_available_memory(&mut self) -> ExecutorResult<usize> {
        Ok(self.available_memory)
    }

    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput> {
        let mut token_map = HashMap::new();

        for req_id in scheduler_output.num_scheduled_tokens.keys() {
            token_map.insert(req_id.clone(), vec![self.next_token_id]);
            self.next_token_id += 1;
        }

        Ok(ModelRunnerOutput::from_token_map(token_map))
    }

    fn shutdown(&mut self) {
        self.is_shutdown = true;
    }

    fn rank(&self) -> usize {
        self.config.rank
    }

    fn local_rank(&self) -> usize {
        self.config.local_rank
    }

    fn is_driver_worker(&self) -> bool {
        self.config.is_driver_worker
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_config() {
        let config = WorkerConfig {
            local_rank: 0,
            rank: 0,
            is_driver_worker: true,
            distributed_init_method: "tcp://localhost:29500".to_string(),
            hf_token: None,
            gguf_file: None,
        };
        assert_eq!(config.rank, 0);
        assert!(config.is_driver_worker);
    }

    #[test]
    fn test_noop_worker_lifecycle() {
        let mut worker = NoopWorker::with_defaults(1024 * 1024 * 1024);

        assert_eq!(worker.rank(), 0);
        assert_eq!(worker.local_rank(), 0);
        assert!(worker.is_driver_worker());

        // Init device.
        worker.init_device().unwrap();

        // Load model.
        worker.load_model().unwrap();

        // Profile memory.
        let memory = worker.determine_available_memory().unwrap();
        assert_eq!(memory, 1024 * 1024 * 1024);

        // Initialize cache.
        worker.initialize_cache(512, 0).unwrap();

        // Health check.
        worker.check_health().unwrap();

        // Shutdown.
        worker.shutdown();
        assert!(worker.is_shutdown);
    }

    #[test]
    fn test_noop_worker_load_before_init() {
        let mut worker = NoopWorker::with_defaults(1024);
        // Load model before init_device should fail.
        let result = worker.load_model();
        assert!(result.is_err());
    }

    #[test]
    fn test_noop_worker_execute() {
        let mut worker = NoopWorker::with_defaults(1024);
        worker.init_device().unwrap();

        let mut num_scheduled = HashMap::new();
        num_scheduled.insert("req-1".to_string(), 50);
        num_scheduled.insert("req-2".to_string(), 30);

        let sched_output = SchedulerOutput {
            num_scheduled_tokens: num_scheduled,
            total_num_scheduled_tokens: 80,
            ..SchedulerOutput::make_empty()
        };

        let output = worker.execute_model(&sched_output).unwrap();
        assert_eq!(output.num_requests(), 2);

        let t1 = output.get_tokens("req-1").unwrap();
        let t2 = output.get_tokens("req-2").unwrap();
        assert_eq!(t1.len(), 1);
        assert_eq!(t2.len(), 1);
        // Tokens should be different.
        assert_ne!(t1[0], t2[0]);
    }

    #[test]
    fn test_noop_worker_sequential_tokens() {
        let mut worker = NoopWorker::with_defaults(1024);
        worker.init_device().unwrap();

        let mut num_scheduled = HashMap::new();
        num_scheduled.insert("req-1".to_string(), 10);

        let sched_output = SchedulerOutput {
            num_scheduled_tokens: num_scheduled,
            total_num_scheduled_tokens: 10,
            ..SchedulerOutput::make_empty()
        };

        let out1 = worker.execute_model(&sched_output).unwrap();
        let out2 = worker.execute_model(&sched_output).unwrap();

        // Token IDs should increment.
        let t1 = out1.get_tokens("req-1").unwrap()[0];
        let t2 = out2.get_tokens("req-1").unwrap()[0];
        assert_eq!(t2, t1 + 1);
    }

    #[test]
    fn test_noop_worker_sleep_wake() {
        let mut worker = NoopWorker::with_defaults(1024);
        worker.sleep(1).unwrap();
        worker.wake_up(None).unwrap();
    }

    #[test]
    fn test_noop_worker_compile() {
        let mut worker = NoopWorker::with_defaults(1024);
        worker.compile_or_warm_up_model().unwrap();
    }
}

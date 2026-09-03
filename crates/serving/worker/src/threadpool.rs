// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Thread-pool executor for multi-GPU tensor parallelism.
//!
//! Thread-pool executor giving each worker a dedicated OS thread. This is
//! essential for NCCL: collectives like all-reduce block until all ranks
//! participate, so each rank must run on its own OS thread to avoid deadlock.
//!
//! # Architecture
//!
//! ```text
//!   ThreadPoolExecutor
//!     ├── OS Thread (rank 0) ──► Worker + NcclProcessGroup
//!     └── OS Thread (rank 1) ──► Worker + NcclProcessGroup
//! ```
//!
//! The executor broadcasts scheduler outputs to all workers via channels,
//! and workers signal completion via response channels. Only the output
//! rank's result is returned to the engine.

use std::sync::Arc;

use scratchy_serving_engine::error::EngineResult;
use scratchy_serving_engine::executor::{Executor, ModelRunnerOutput};
use scratchy_serving_scheduler::scheduler::output::SchedulerOutput;
use tracing::info;

use crate::error::ExecutorResult;
use crate::parallel::ResolvedParallelConfig;
use crate::worker::Worker;

// ---------------------------------------------------------------------------
// Worker handle (OS thread + channels)
// ---------------------------------------------------------------------------

/// Message sent from executor to worker thread.
enum Request {
    ExecuteModel(Arc<SchedulerOutput>),
    InitializeCache {
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
    },
    DetermineAvailableMemory,
    CheckHealth,
    /// NCCL broadcast bytes on the worker thread (correct CUDA context).
    /// `(data, root)` — root sends, others receive.
    NcclBroadcast(Vec<u8>, usize),
    Shutdown,
}

/// Response from worker thread to executor.
enum Response {
    ModelOutput(Box<ExecutorResult<ModelRunnerOutput>>),
    CacheInitialized(ExecutorResult<()>),
    AvailableMemory(ExecutorResult<usize>),
    HealthOk(ExecutorResult<()>),
    /// Result of NCCL broadcast (received bytes or error).
    NcclBroadcastResult(ExecutorResult<Vec<u8>>),
    ShutdownAck,
}

struct WorkerThread {
    #[allow(dead_code)]
    rank: usize,
    is_output_rank: bool,
    tx: std::sync::mpsc::Sender<Request>,
    rx: std::sync::mpsc::Receiver<Response>,
    _handle: std::thread::JoinHandle<()>,
}

fn worker_loop(
    mut worker: Box<dyn Worker>,
    pg: Option<std::sync::Arc<dyn scratchy_core_model::process_group::ProcessGroup>>,
    rx: std::sync::mpsc::Receiver<Request>,
    tx: std::sync::mpsc::Sender<Response>,
) {
    while let Ok(request) = rx.recv() {
        let response = match request {
            Request::ExecuteModel(sched_output) => {
                Response::ModelOutput(Box::new(worker.execute_model(&sched_output)))
            }
            Request::InitializeCache {
                num_gpu_blocks,
                num_cpu_blocks,
            } => {
                Response::CacheInitialized(worker.initialize_cache(num_gpu_blocks, num_cpu_blocks))
            }
            Request::DetermineAvailableMemory => {
                Response::AvailableMemory(worker.determine_available_memory())
            }
            Request::CheckHealth => Response::HealthOk(worker.check_health()),
            Request::NcclBroadcast(data, root) => {
                let result = if let Some(ref pg) = pg {
                    pg.broadcast_bytes(&data, root)
                        .map_err(|e| crate::error::ExecutorError::WorkerExecution(e.to_string()))
                } else {
                    Err(crate::error::ExecutorError::Config(
                        "NCCL broadcast requires process group".to_string(),
                    ))
                };
                Response::NcclBroadcastResult(result)
            }
            Request::Shutdown => {
                worker.shutdown();
                let _ = tx.send(Response::ShutdownAck);
                return;
            }
        };
        if tx.send(response).is_err() {
            return; // Executor dropped
        }
    }
}

// ---------------------------------------------------------------------------
// ThreadPoolExecutor
// ---------------------------------------------------------------------------

/// Multi-GPU executor using dedicated OS threads per worker.
///
/// Each worker runs on its own OS thread, ensuring NCCL collectives can
/// execute concurrently across ranks without tokio scheduling issues.
pub struct ThreadPoolExecutor {
    workers: Vec<WorkerThread>,
    #[allow(dead_code)]
    parallel_config: ResolvedParallelConfig,
    is_shutdown: bool,
}

impl ThreadPoolExecutor {
    /// Create a new thread-pool executor from pre-initialized workers.
    ///
    /// Workers are moved to dedicated OS threads. They must already have
    /// device init, model load, and NCCL injection completed.
    /// Create a new thread-pool executor.
    ///
    /// `process_groups` is an optional list of per-worker process groups for
    /// NCCL broadcast (multi-node control channel). Pass `None` for
    /// single-node setups.
    pub fn new(workers: Vec<Box<dyn Worker>>, parallel_config: ResolvedParallelConfig) -> Self {
        Self::with_process_groups(workers, parallel_config, None)
    }

    /// Create with explicit per-worker NCCL process groups for multi-node broadcast.
    pub fn with_process_groups(
        workers: Vec<Box<dyn Worker>>,
        parallel_config: ResolvedParallelConfig,
        process_groups: Option<Vec<Arc<dyn scratchy_core_model::process_group::ProcessGroup>>>,
    ) -> Self {
        let output_rank = parallel_config.output_rank();

        let handles: Vec<WorkerThread> = workers
            .into_iter()
            .enumerate()
            .map(|(rank, worker)| {
                let (req_tx, req_rx) = std::sync::mpsc::channel();
                let (resp_tx, resp_rx) = std::sync::mpsc::channel();
                let pg = process_groups
                    .as_ref()
                    .and_then(|pgs| pgs.get(rank).cloned());

                let handle = std::thread::Builder::new()
                    .name(format!("vllm-worker-{rank}"))
                    .spawn(move || worker_loop(worker, pg, req_rx, resp_tx))
                    .expect("failed to spawn worker thread");

                WorkerThread {
                    rank,
                    is_output_rank: rank == output_rank,
                    tx: req_tx,
                    rx: resp_rx,
                    _handle: handle,
                }
            })
            .collect();

        info!(
            "ThreadPoolExecutor: spawned {} worker threads, output_rank={}",
            handles.len(),
            output_rank
        );

        Self {
            workers: handles,
            parallel_config,
            is_shutdown: false,
        }
    }

    /// Broadcast a request to all workers and collect responses.
    fn broadcast(&self, make_request: impl Fn() -> Request) -> Vec<Response> {
        // Send to all workers.
        for w in &self.workers {
            let _ = w.tx.send(make_request());
        }
        // Collect responses (blocking — each worker responds in order).
        self.workers.iter().map(|w| w.rx.recv().unwrap()).collect()
    }

    /// NCCL broadcast bytes on the worker thread (correct CUDA context).
    ///
    /// Dispatches to the rank-0 worker thread. For multi-node: rank 0 sends,
    /// remote ranks receive. All ranks must call simultaneously (collective).
    pub fn nccl_broadcast(&self, data: &[u8], root: usize) -> ExecutorResult<Vec<u8>> {
        // Only dispatch to the first (rank 0) worker — it has the NCCL comm.
        let w = &self.workers[0];
        let _ = w.tx.send(Request::NcclBroadcast(data.to_vec(), root));
        match w.rx.recv().unwrap() {
            Response::NcclBroadcastResult(result) => result,
            _ => Err(crate::error::ExecutorError::WorkerExecution(
                "unexpected response from NCCL broadcast".to_string(),
            )),
        }
    }
}

impl Executor for ThreadPoolExecutor {
    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> EngineResult<ModelRunnerOutput> {
        let shared = Arc::new(scheduler_output.clone());
        let responses = self.broadcast(|| Request::ExecuteModel(Arc::clone(&shared)));

        // Return output rank's result.
        for (w, resp) in self.workers.iter().zip(responses) {
            if w.is_output_rank {
                return match resp {
                    Response::ModelOutput(result) => (*result).map_err(|e| {
                        scratchy_serving_engine::error::EngineError::Executor(e.to_string())
                    }),
                    _ => Err(scratchy_serving_engine::error::EngineError::Executor(
                        "unexpected response".to_string(),
                    )),
                };
            }
        }
        Err(scratchy_serving_engine::error::EngineError::Executor(
            "no output rank".to_string(),
        ))
    }

    fn max_concurrent_batches(&self) -> usize {
        self.parallel_config.pp_group.world_size
    }

    fn initialize_cache(
        &mut self,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
    ) -> EngineResult<()> {
        let responses = self.broadcast(|| Request::InitializeCache {
            num_gpu_blocks,
            num_cpu_blocks,
        });
        for resp in responses {
            if let Response::CacheInitialized(Err(e)) = resp {
                return Err(scratchy_serving_engine::error::EngineError::Executor(
                    e.to_string(),
                ));
            }
        }
        Ok(())
    }

    fn determine_available_memory(&mut self) -> EngineResult<Vec<usize>> {
        let responses = self.broadcast(|| Request::DetermineAvailableMemory);
        let mut memories = Vec::with_capacity(responses.len());
        for resp in responses {
            match resp {
                Response::AvailableMemory(Ok(mem)) => memories.push(mem),
                Response::AvailableMemory(Err(e)) => {
                    return Err(scratchy_serving_engine::error::EngineError::Executor(
                        e.to_string(),
                    ));
                }
                _ => {
                    return Err(scratchy_serving_engine::error::EngineError::Executor(
                        "unexpected response".to_string(),
                    ));
                }
            }
        }
        Ok(memories)
    }

    fn check_health(&self) -> EngineResult<()> {
        // Send to all workers.
        for w in &self.workers {
            let _ = w.tx.send(Request::CheckHealth);
        }
        for w in &self.workers {
            match w.rx.recv().unwrap() {
                Response::HealthOk(Ok(())) => {}
                Response::HealthOk(Err(e)) => {
                    return Err(scratchy_serving_engine::error::EngineError::Executor(
                        e.to_string(),
                    ));
                }
                _ => {
                    return Err(scratchy_serving_engine::error::EngineError::Executor(
                        "unexpected response".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn sleep(&mut self, _level: u32) -> EngineResult<()> {
        Ok(()) // Not implemented for thread pool
    }

    fn wake_up(&mut self, _tags: Option<&[String]>) -> EngineResult<()> {
        Ok(()) // Not implemented for thread pool
    }

    fn is_sleeping(&self) -> bool {
        false
    }

    fn shutdown(&mut self) {
        if self.is_shutdown {
            return;
        }
        info!(
            "ThreadPoolExecutor: shutting down {} workers",
            self.workers.len()
        );
        for w in &self.workers {
            let _ = w.tx.send(Request::Shutdown);
        }
        // Wait for acks.
        for w in &self.workers {
            let _ = w.rx.recv();
        }
        self.is_shutdown = true;
    }
}

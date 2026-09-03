// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Multi-node executor: wraps a local executor and broadcasts
//! `SchedulerOutput` to all remote follower nodes via TCP before each
//! forward step.
//!
//! TCP is used for the control plane (scheduler output broadcast).
//! NCCL is used only for the data plane (model forward-pass collectives).
//! This matches Python vLLM's `mp` backend architecture.

use scratchy_serving_engine::error::{EngineError, EngineResult};
use scratchy_serving_engine::executor::{Executor, ModelRunnerOutput};
use scratchy_serving_scheduler::scheduler::output::SchedulerOutput;
use scratchy_target_cuda::TcpControlChannel;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Control protocol (serialized via bincode, broadcast via TCP)
// ---------------------------------------------------------------------------

/// Messages broadcast from rank 0 to all follower nodes via TCP.
#[derive(Serialize, Deserialize)]
pub enum ControlMessage {
    /// Execute a forward pass with this scheduler output.
    ExecuteModel(Box<SchedulerOutput>),
    /// Initialize KV cache on all workers.
    InitCache {
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
    },
    /// Warm up / compile the model (CUDA graph capture).
    Warmup,
    /// Shut down the remote worker loop.
    Shutdown,
}

// ---------------------------------------------------------------------------
// MultiNodeExecutor
// ---------------------------------------------------------------------------

/// Wraps a local executor and broadcasts scheduler outputs to all remote
/// follower nodes via TCP before each local `execute_model` call.
///
/// The TCP control channel carries serialized `ControlMessage`s. NCCL
/// collectives in the model forward pass synchronize the actual tensor
/// data between ranks.
pub struct MultiNodeExecutor {
    inner: Box<dyn Executor>,
    channel: TcpControlChannel,
}

impl MultiNodeExecutor {
    pub fn new(inner: Box<dyn Executor>, channel: TcpControlChannel) -> Self {
        Self { inner, channel }
    }

    /// Serialize and broadcast a control message via TCP (rank 0 sends).
    fn broadcast_msg(&mut self, msg: &ControlMessage) -> EngineResult<()> {
        let data = bincode::serialize(msg).map_err(|e| {
            EngineError::Executor(format!("failed to serialize control message: {e}"))
        })?;
        self.channel
            .broadcast(&data)
            .map_err(|e| EngineError::Executor(format!("TCP broadcast failed: {e}")))?;
        Ok(())
    }
}

impl Executor for MultiNodeExecutor {
    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> EngineResult<ModelRunnerOutput> {
        // Broadcast to all followers so they start their forward pass.
        let msg = ControlMessage::ExecuteModel(Box::new(scheduler_output.clone()));
        self.broadcast_msg(&msg)?;
        // Execute locally — NCCL collectives in the model synchronize ranks.
        self.inner.execute_model(scheduler_output)
    }

    fn max_concurrent_batches(&self) -> usize {
        self.inner.max_concurrent_batches()
    }

    fn initialize_cache(
        &mut self,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
    ) -> EngineResult<()> {
        let msg = ControlMessage::InitCache {
            num_gpu_blocks,
            num_cpu_blocks,
        };
        self.broadcast_msg(&msg)?;
        self.inner.initialize_cache(num_gpu_blocks, num_cpu_blocks)
    }

    fn determine_available_memory(&mut self) -> EngineResult<Vec<usize>> {
        self.inner.determine_available_memory()
    }

    fn check_health(&self) -> EngineResult<()> {
        self.inner.check_health()
    }

    fn sleep(&mut self, level: u32) -> EngineResult<()> {
        self.inner.sleep(level)
    }

    fn wake_up(&mut self, tags: Option<&[String]>) -> EngineResult<()> {
        self.inner.wake_up(tags)
    }

    fn is_sleeping(&self) -> bool {
        self.inner.is_sleeping()
    }

    fn shutdown(&mut self) {
        let _ = self.broadcast_msg(&ControlMessage::Shutdown);
        self.inner.shutdown();
    }
}

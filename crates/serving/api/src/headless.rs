// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Headless worker loop for remote (non-master) nodes in multi-node TP.
//!
//! After NCCL init, node_rank > 0 enters this blocking loop instead of
//! starting an engine/scheduler/HTTP server. Rank 0 broadcasts
//! `ControlMessage`s via TCP; this loop receives them and dispatches to
//! the local `UniProcExecutor`.
//!
//! NCCL collectives in the model forward pass synchronize tensor data
//! between ranks automatically — no explicit NCCL calls needed here.
//! On spyre each rank is a process driving one card, and its collectives
//! are the card's own p2p RDMA control blocks, in the same stream as the
//! compute — this loop only hands each rank the step to run.

#[cfg(any(feature = "nccl", feature = "spyre-hw"))]
use scratchy_serving_engine::executor::Executor;
#[cfg(any(feature = "nccl", feature = "spyre-hw"))]
use scratchy_serving_transport::TcpControlChannel;
#[cfg(any(feature = "nccl", feature = "spyre-hw"))]
use scratchy_serving_worker::multinode::ControlMessage;
#[cfg(any(feature = "nccl", feature = "spyre-hw"))]
use scratchy_serving_worker::uniproc::UniProcExecutor;
#[cfg(any(feature = "nccl", feature = "spyre-hw"))]
use tracing::{error, info};

/// Run the headless worker loop. Blocks forever (or until shutdown).
///
/// Loops on TCP control channel: receive `ControlMessage` from rank 0,
/// dispatch to local executor, discard output, repeat.
#[cfg(any(feature = "nccl", feature = "spyre-hw"))]
pub fn run_headless(
    mut executor: UniProcExecutor,
    mut channel: TcpControlChannel,
) -> anyhow::Result<()> {
    info!("Headless worker: entering TCP control channel loop");

    loop {
        // Receive broadcast from rank 0 via TCP.
        match ControlMessage::recv(&mut channel)? {
            ControlMessage::ExecuteModel(scheduler_output) => {
                if let Err(e) = executor.execute_model(&scheduler_output) {
                    error!("Headless worker: execute_model failed: {e}");
                }
            }
            ControlMessage::InitCache {
                num_gpu_blocks,
                num_cpu_blocks,
            } => {
                info!(
                    "Headless worker: initializing cache (gpu_blocks={}, cpu_blocks={})",
                    num_gpu_blocks, num_cpu_blocks
                );
                if let Err(e) = executor.initialize_cache(num_gpu_blocks, num_cpu_blocks) {
                    error!("Headless worker: initialize_cache failed: {e}");
                }
            }
            ControlMessage::Warmup => {
                info!("Headless worker: warming up model");
                if let Err(e) = executor.worker_mut().compile_or_warm_up_model() {
                    error!("Headless worker: warmup failed: {e}");
                }
            }
            ControlMessage::Shutdown => {
                info!("Headless worker: received shutdown, exiting");
                break;
            }
        }
    }

    executor.shutdown();
    Ok(())
}

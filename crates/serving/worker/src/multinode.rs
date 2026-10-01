// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Multi-node executor: wraps a local executor and broadcasts
//! `SchedulerOutput` to all remote follower nodes via TCP before each
//! forward step.
//!
//! TCP is used for the control plane (scheduler output broadcast).
//! NCCL is used only for the data plane (model forward-pass collectives).
//! This matches Python vLLM's `mp` backend architecture.

use anyhow::Context;
use scratchy_serving_engine::error::{EngineError, EngineResult};
use scratchy_serving_engine::executor::{Executor, ModelRunnerOutput};
use scratchy_serving_scheduler::scheduler::output::SchedulerOutput;
use scratchy_serving_transport::codec;
use scratchy_target_cuda::TcpControlChannel;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Control protocol (msgpack, broadcast via TCP)
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

impl ControlMessage {
    /// Send this message to every follower (rank 0 only).
    ///
    /// Encoded with the transport codec (a msgpack map with named fields),
    /// never positionally: `NewRequestData` omits its `None` optionals on the
    /// wire, and `GuidedGrammar::JsonSchema` carries a `serde_json::Value`,
    /// which only a self-describing format can decode.
    pub fn broadcast(&self, channel: &mut TcpControlChannel) -> anyhow::Result<()> {
        let data = codec::encode(self).context("failed to encode control message")?;
        channel.broadcast(&data)
    }

    /// Block until the next message from rank 0 arrives (followers only).
    pub fn recv(channel: &mut TcpControlChannel) -> anyhow::Result<Self> {
        let data = channel.recv()?;
        codec::decode(&data)
            .with_context(|| format!("failed to decode control message ({} bytes)", data.len()))
    }
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
        msg.broadcast(&mut self.channel)
            .map_err(|e| EngineError::Executor(format!("control broadcast failed: {e:#}")))
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

#[cfg(test)]
mod tests {
    use super::*;
    use scratchy_core_common::SamplingParams;
    use scratchy_core_common::sampling::GuidedGrammar;
    use scratchy_serving_scheduler::scheduler::output::NewRequestData;

    fn new_request(req_id: &str, guided_grammar: Option<GuidedGrammar>) -> NewRequestData {
        NewRequestData::new(
            req_id.into(),
            Some(vec![1, 2, 3]),
            vec![vec![0]],
            0,
            Some(SamplingParams {
                guided_grammar,
                ..SamplingParams::default()
            }),
            None,
            None,
        )
    }

    /// A step scheduling two new requests — `None` span fields (skipped on
    /// the wire) and a JSON-schema grammar — reaches the follower intact.
    #[test]
    fn execute_model_reaches_follower_intact() {
        // The channel binds `master_port + 2`: probe THAT port, not the base.
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
            - 2;
        let schema = serde_json::json!({"type": "object", "required": ["a"]});

        let mut step = SchedulerOutput::make_empty();
        step.scheduled_new_reqs = vec![
            new_request("plain", None),
            new_request(
                "json",
                Some(GuidedGrammar::JsonSchema {
                    schema: schema.clone(),
                }),
            ),
        ];
        step.total_num_scheduled_tokens = 6;

        let leader = std::thread::spawn(move || {
            let mut ch = TcpControlChannel::establish(0, 2, "127.0.0.1", port).unwrap();
            ControlMessage::ExecuteModel(Box::new(step))
                .broadcast(&mut ch)
                .unwrap();
        });
        let mut ch = TcpControlChannel::establish(1, 2, "127.0.0.1", port).unwrap();
        let received = ControlMessage::recv(&mut ch).unwrap();
        leader.join().unwrap();

        let ControlMessage::ExecuteModel(step) = received else {
            panic!("follower received a message other than ExecuteModel");
        };
        assert_eq!(step.total_num_scheduled_tokens, 6);
        let [plain, json] = step.scheduled_new_reqs.as_slice() else {
            panic!(
                "expected 2 new requests, got {}",
                step.scheduled_new_reqs.len()
            );
        };
        assert_eq!(plain.req_id, "plain");
        assert_eq!(json.req_id, "json");
        assert_eq!(json.prompt_token_ids.as_deref(), Some(&[1, 2, 3][..]));
        assert!(json.block_annotations.is_none() && json.reused_block_idxs.is_none());
        let Some(GuidedGrammar::JsonSchema { schema: received }) = json
            .sampling_params
            .as_ref()
            .and_then(|p| p.guided_grammar.as_ref())
        else {
            panic!("JSON-schema grammar lost in transit");
        };
        assert_eq!(*received, schema);
    }
}

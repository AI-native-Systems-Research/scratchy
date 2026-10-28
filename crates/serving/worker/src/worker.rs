// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! The `Worker` trait + `NoopWorker` moved down into `scratchy-serving-engine`
//! (below `compiler` and the target crates) so backend worker impls in
//! `targets/*` can implement the trait without a dependency cycle.

pub use scratchy_serving_engine::worker::*;

#[cfg(test)]
use scratchy_serving_engine::error::ExecutorResult;
#[cfg(test)]
use scratchy_serving_engine::executor::ModelRunnerOutput;
#[cfg(test)]
use scratchy_serving_scheduler::scheduler::output::SchedulerOutput;

/// A [`NoopWorker`] that keeps the stop tokens it is handed.
#[cfg(test)]
pub(crate) struct StopRecordingWorker(
    pub(crate) NoopWorker,
    pub(crate) std::sync::Arc<std::sync::Mutex<Vec<u32>>>,
);

#[cfg(test)]
impl Worker for StopRecordingWorker {
    fn init_device(&mut self) -> ExecutorResult<()> {
        self.0.init_device()
    }
    fn load_model(&mut self) -> ExecutorResult<()> {
        self.0.load_model()
    }
    fn initialize_cache(&mut self, gpu: usize, cpu: usize) -> ExecutorResult<()> {
        self.0.initialize_cache(gpu, cpu)
    }
    fn determine_available_memory(&mut self) -> ExecutorResult<usize> {
        self.0.determine_available_memory()
    }
    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput> {
        self.0.execute_model(scheduler_output)
    }
    fn set_stop_token_ids(&mut self, stop_token_ids: &[u32]) {
        *self.1.lock().unwrap() = stop_token_ids.to_vec();
    }
    fn shutdown(&mut self) {
        self.0.shutdown()
    }
    fn rank(&self) -> usize {
        self.0.rank()
    }
    fn local_rank(&self) -> usize {
        self.0.local_rank()
    }
    fn is_driver_worker(&self) -> bool {
        self.0.is_driver_worker()
    }
}

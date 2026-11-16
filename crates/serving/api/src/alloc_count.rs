// SPDX-License-Identifier: Apache-2.0
//! The heap allocations a piece of the serving stack makes, counted, so a test can state what a
//! steady decode step costs (#1269): this crate's tests run on a system allocator that counts every
//! allocation and reallocation on the thread making it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    /// This thread's allocations so far: `const`, with no destructor, so counting never allocates.
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

/// The system allocator, counting.
struct Counting;

fn count() {
    // A thread past its thread-locals' teardown has no count to add to.
    let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
}

// SAFETY: every method is `System`'s, called with the caller's own arguments.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: the caller upholds `alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: the caller upholds `alloc_zeroed`'s contract.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        // SAFETY: the caller upholds `realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller upholds `dealloc`'s contract.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// `f`'s result, and the allocations it made on this thread.
pub(crate) fn allocations<R>(f: impl FnOnce() -> R) -> (R, u64) {
    let before = ALLOCATIONS.with(Cell::get);
    let r = f();
    (r, ALLOCATIONS.with(Cell::get) - before)
}

#[cfg(test)]
mod tests {
    use super::allocations;
    use scratchy_core_common::{KvBlockTokens, Request, SamplingParams};
    use scratchy_core_config::{SchedulerConfig, SchedulerPolicy};
    use scratchy_serving_engine::engine_core::{EngineCore, EngineCoreConfig};
    use scratchy_serving_engine::executor::NoopExecutor;

    /// The requests decoding side by side.
    const BATCH: usize = 8;

    fn engine() -> EngineCore {
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
            block_size: KvBlockTokens::new(16).expect("16 > 0"),
            engine_index: 0,
            async_scheduling: false,
            use_spec_decode: false,
            proposer_config: None,
            eos_token_ids: vec![],
            is_pooling: false,
            enable_prefix_caching: false,
            hybrid_kv: None,
        };
        let mut engine = EngineCore::new(config, Box::new(NoopExecutor::new(1024)));
        for i in 0..BATCH {
            let params = SamplingParams {
                max_tokens: Some(1000),
                ..Default::default()
            };
            let prompt = (0..10).collect();
            engine.add_request(Request::new(
                format!("req-{i}"),
                prompt,
                params,
                0.0,
                0,
                0,
                None,
            ));
        }
        engine
    }

    /// A steady decode step's `finalize_step` allocates the outputs it hands its client — one
    /// list, and each request's id and new tokens — and nothing else but its requests' own growing
    /// histories, which double: on the steps where they do, every request of the batch grows at
    /// once (#1269).
    #[test]
    fn a_decode_step_finalizes_into_its_outputs_alone() {
        let mut engine = engine();
        let mut executor = engine.take_executor().expect("an executor");
        let mut step = |engine: &mut EngineCore| {
            let sched = engine.schedule_next().expect("decoding");
            let out = executor.execute_model(&sched).expect("the noop executes");
            allocations(|| engine.finalize_step(&sched, &out))
        };
        // Through the prefill, into steady decode.
        for _ in 0..4 {
            step(&mut engine);
        }
        let outputs = (1 + 2 * BATCH) as u64;
        let steps: Vec<u64> = (0..64)
            .map(|_| {
                let (step_outputs, n) = step(&mut engine);
                let step_outputs = step_outputs.expect("every request decodes a token");
                assert_eq!(step_outputs.outputs.len(), BATCH);
                n
            })
            .collect();
        assert_eq!(steps.iter().min(), Some(&outputs), "per step: {steps:?}");
        for n in &steps {
            assert_eq!((n - outputs) % BATCH as u64, 0, "per step: {steps:?}");
        }
    }
}

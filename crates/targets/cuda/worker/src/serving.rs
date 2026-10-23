// SPDX-License-Identifier: Apache-2.0
//! THE CUDA `Worker` — one sequence at a time, greedy, over the emitted tapes. Only under `nvcc`.
//!
//! `load_model` opens the checkpoint once and checks it against every model in the CUDA registry;
//! the first whose weight table it passes is the model, and the shared `load_weights` makes its
//! buffers. Its pools are fixed facts of the tapes, reported to the engine rather than sized from
//! the memory budget: one block table's worth of KV blocks (plus the scheduler's null block), one
//! GDN slot, one running sequence, and the largest prefill bucket as the step budget.
//!
//! `execute_model` keeps each request's tokens and block table in the shared `InputBatch`, as
//! spyre's worker does, runs the step's one request in the smallest bucket that holds its tokens
//! (`plan`), and returns the greedy token of its last row when the step reaches the end of its
//! history. The token is in the `InputBatch` before the call returns, so a step the engine
//! scheduled behind this one finds it there.

use std::path::PathBuf;
use std::time::Instant;

use scratchy_core_common::SamplingParams;
use scratchy_core_model::weight::HfModelConfig;
use scratchy_forward_compiler::gdn_slot_allocator::{GdnRow, GdnSlotAllocator};
use scratchy_serving_engine::error::{ExecutorError, ExecutorResult};
use scratchy_serving_engine::executor::ModelRunnerOutput;
use scratchy_serving_engine::gpu_budget::gdn_slot_key;
use scratchy_serving_engine::input_batch::InputBatch;
use scratchy_serving_engine::worker::Worker;
use scratchy_serving_engine::worker_factory::{
    WorkerCreateConfig, WorkerCreationResult, resolve_model_path,
};
use scratchy_serving_scheduler::scheduler::output::SchedulerOutput;
use scratchy_target_cuda::registry::{self, CudaModelEntry};
use scratchy_target_cuda::runtime::{
    CudaDevice, CudaWorker, DeviceBuffer, GdnPoolConfig, GdnSequence, Input, KvPoolConfig,
    KvSequence, load_weights,
};
use scratchy_target_cuda::tape::KvTape;
use scratchy_target_cuda::weights::{check, open_checkpoint};
use tracing::info;

use crate::plan::{
    block_table_u32, check_sampling, choose_bucket, emits_token, fill_inputs, gdn_fresh,
    greedy_token, kv_block_bytes, logits_row, scheduled_request,
};

/// The GDN slots, and the sequences running at once: one.
const SEQUENCES: usize = 1;

/// Construct → init → load → extract, for [`crate::CudaWorkerFactory`].
pub(crate) fn create(cfg: WorkerCreateConfig) -> anyhow::Result<WorkerCreationResult> {
    use anyhow::Context as _;
    refuse_unsupported(&cfg)?;
    let mut worker = CudaServingWorker::new(cfg);
    worker.init_device().context("CUDA init_device")?;
    worker.load_model().context("CUDA load_model")?;
    let model = worker.model.as_ref().context("the CUDA model after load")?;
    let (hf_config, model_dir) = (model.hf_config.clone(), Some(model.dir.clone()));
    // The KV cache is bf16.
    Ok((Box::new(worker), hf_config, model_dir, 2))
}

/// Refuse, before anything loads, what this worker does not do.
fn refuse_unsupported(cfg: &WorkerCreateConfig) -> anyhow::Result<()> {
    let refuse = |what: &str| anyhow::bail!("the CUDA backend does not support {what} yet");
    if cfg.draft.is_some() {
        return refuse("speculative decoding: drop --speculative-model");
    }
    if cfg.is_pooling {
        return refuse("pooling (--runner pooling)");
    }
    if cfg.lora_adapter.is_some() {
        return refuse("LoRA adapters");
    }
    if cfg.kv_cache_dtype != "auto" {
        return refuse(&format!(
            "a KV cache of {}: its KV cache is bf16 (--kv-cache-dtype auto)",
            cfg.kv_cache_dtype
        ));
    }
    if !matches!(cfg.dtype.as_str(), "auto" | "bfloat16" | "bf16") {
        return refuse(&format!(
            "{} weights: its tapes read bf16 (--dtype auto)",
            cfg.dtype
        ));
    }
    Ok(())
}

/// The model `load_model` found: its registry entry, checkpoint and config, its weights until
/// `initialize_cache` hands them to the tape player, and the host buffers a step reuses.
struct Loaded {
    entry: &'static CudaModelEntry,
    dir: PathBuf,
    hf_config: HfModelConfig,
    /// Taken by `initialize_cache`.
    weights: Option<Vec<DeviceBuffer>>,
    /// The first KV tape's geometry, which every KV tape shares (`CudaWorker::new` checks).
    kv: Option<KvTape>,
    /// The host result buffer, sized once for the largest tape's result.
    result: Vec<u16>,
}

/// What `initialize_cache` built.
struct Running {
    tapes: CudaWorker<'static, Vec<DeviceBuffer>>,
    gdn: GdnSlotAllocator,
}

/// The CUDA serving worker. See the module docs.
pub(crate) struct CudaServingWorker {
    cfg: WorkerCreateConfig,
    device: Option<CudaDevice>,
    model: Option<Loaded>,
    running: Option<Running>,
    input_batch: InputBatch,
    /// A step's token ids, positions and block table, reused.
    ids: Vec<u32>,
    positions: Vec<u32>,
    table: Vec<u32>,
    chunk: Vec<u32>,
}

// SAFETY: the worker holds raw CUDA handles (the tape player's stream and device pointers), which
// are not tied to the thread that made them: the CUDA runtime's primary context is per process,
// and device 0 is current by default on every thread. The engine uses a worker from one thread at
// a time (`Worker: Send`, not `Sync`), the same reasoning as metal's `MetalWorker`.
unsafe impl Send for CudaServingWorker {}

fn exec(e: impl std::fmt::Display) -> ExecutorError {
    ExecutorError::WorkerExecution(e.to_string())
}

fn init(e: impl std::fmt::Display) -> ExecutorError {
    ExecutorError::WorkerInit(e.to_string())
}

impl CudaServingWorker {
    fn new(cfg: WorkerCreateConfig) -> Self {
        Self {
            cfg,
            device: None,
            model: None,
            running: None,
            input_batch: InputBatch::new(),
            ids: Vec::new(),
            positions: Vec::new(),
            table: Vec::new(),
            chunk: Vec::new(),
        }
    }

    fn device(&self) -> ExecutorResult<&CudaDevice> {
        (self.device.as_ref()).ok_or_else(|| init("the CUDA device is not open (init_device)"))
    }

    /// The tapes' KV geometry, after `load_model`.
    fn kv(&self) -> Option<KvTape> {
        self.model.as_ref().and_then(|m| m.kv)
    }

    /// Admit a new request: its sampling refused unless greedy, then its whole prompt and block
    /// table into the `InputBatch`.
    fn admit(
        &mut self,
        req_id: &str,
        prompt: &[u32],
        params: Option<&SamplingParams>,
        blocks: Vec<usize>,
        num_computed_tokens: u32,
    ) -> ExecutorResult<()> {
        check_sampling(params).map_err(exec)?;
        (self.input_batch).add_request(req_id.to_string(), prompt, blocks, num_computed_tokens);
        Ok(())
    }

    /// The scheduler's `num_computed_tokens` for `req_id` this step.
    fn computed(sched: &SchedulerOutput, req_id: &str) -> Option<u32> {
        let new = (sched.scheduled_new_reqs.iter()).find(|r| r.req_id == req_id);
        if let Some(r) = new {
            return Some(r.num_computed_tokens);
        }
        let cached = &sched.scheduled_cached_reqs;
        (cached.req_ids.iter().position(|id| id == req_id))
            .and_then(|i| cached.num_computed_tokens.get(i).copied())
    }

    /// Run `req_id`'s `n` tokens from `start`: the forward, and its greedy token when the step
    /// reaches the end of the request's history.
    fn step(&mut self, req_id: &str, start: u32, n: usize) -> ExecutorResult<Option<u32>> {
        let batch = &self.input_batch;
        if !batch.contains(req_id) {
            return Err(exec(format!(
                "request {req_id} is scheduled and the CUDA worker does not hold it: the step \
                 that admitted it was refused"
            )));
        }
        let in_pool = batch.tokens_in_pool_for(req_id);
        if in_pool != start as usize {
            return Err(exec(format!(
                "request {req_id} is scheduled from token {start} and its KV holds {in_pool}"
            )));
        }
        self.chunk.clear();
        for pos in start as usize..start as usize + n {
            // ⛔ A STEP NEVER RUNS ON A TOKEN THE HOST DOES NOT HAVE. Every token this worker
            // samples is in the `InputBatch` before `execute_model` returns, so a step the
            // engine scheduled behind it finds it; a missing one is refused, not guessed.
            let token = batch.token_at(req_id, pos).ok_or_else(|| {
                exec(format!(
                    "request {req_id} has no token at position {pos} (it has {}): a step was \
                     scheduled on a token this worker has not produced",
                    batch.token_count(req_id)
                ))
            })?;
            self.chunk.push(token);
        }
        let history = batch.token_count(req_id);
        let blocks = batch.block_table(req_id).unwrap_or(&[]);
        block_table_u32(blocks, &mut self.table).map_err(exec)?;

        let model = (self.model.as_mut()).ok_or_else(|| exec("no CUDA model is loaded"))?;
        let running = (self.running.as_mut())
            .ok_or_else(|| exec("the CUDA KV cache is not built (initialize_cache)"))?;
        let tapes = model.entry.tapes;
        let bucket_ms: Vec<u32> = tapes.iter().map(|t| t.bucket_m).collect();
        let (bucket, bucket_m) = choose_bucket(&bucket_ms, n).map_err(exec)?;
        let tape = &tapes[bucket];
        fill_inputs(
            &self.chunk,
            start,
            bucket_m,
            &mut self.ids,
            &mut self.positions,
        )
        .map_err(exec)?;
        let q_len = n as u32;
        let kv = tape.kv.map(|_| KvSequence {
            block_table: &self.table,
            prefix_len: start,
            q_len,
        });
        let gdn = match tape.gdn {
            None => None,
            Some(_) => {
                let row = GdnRow::new(gdn_slot_key(req_id), 0, start);
                let (slots, codes) = running.gdn.step([row]).map_err(exec)?;
                let fresh = gdn_fresh(codes[0], start)
                    .map_err(|e| exec(format!("request {req_id}: {e}")))?;
                Some(GdnSequence {
                    slot: slots[0].slot().index() as u32,
                    q_len,
                    fresh,
                })
            }
        };
        let positions = match tape.rotary {
            Some(_) => &self.positions[..],
            None => &[],
        };
        let result = &mut model.result[..tape.result.bytes as usize / 2];
        (running.tapes)
            .forward(
                bucket,
                Input::TokenIds(&self.ids),
                positions,
                kv.as_ref(),
                gdn.as_ref(),
                result,
            )
            .map_err(exec)?;
        let token = if emits_token(start as usize, n, history) {
            let row = logits_row(result, bucket_m, n - 1).map_err(exec)?;
            Some(greedy_token(row).map_err(exec)?)
        } else {
            None
        };
        if let Some(t) = token {
            self.input_batch.push_generated(req_id, t);
        }
        (self.input_batch).set_tokens_in_pool(req_id, start as usize + n);
        Ok(token)
    }
}

impl Worker for CudaServingWorker {
    fn init_device(&mut self) -> ExecutorResult<()> {
        self.device = Some(CudaDevice::open().map_err(init)?);
        Ok(())
    }

    fn load_model(&mut self) -> ExecutorResult<()> {
        let device = self.device()?;
        let cfg = &self.cfg;
        let dir = resolve_model_path(
            &cfg.model_path,
            cfg.hf_token.as_deref(),
            cfg.gguf_file.as_deref(),
            None,
        )?;
        if !dir.is_dir() {
            return Err(init(format!(
                "{} is not a safetensors checkpoint directory; the CUDA backend loads no GGUF",
                dir.display()
            )));
        }
        let hf_config = HfModelConfig::from_dir(&dir).map_err(init)?;
        let stems: Vec<&str> = registry::entries().map(|e| e.stem).collect();
        info!("CUDA model registry: {} stems {stems:?}", stems.len());
        if stems.is_empty() {
            return Err(init(
                "the CUDA model registry is empty: this scr names no model; build it with \
                 `--features nvcc,model/<stem>`",
            ));
        }
        // The checkpoint's header against every registered weight table: each tensor's name,
        // dtype and shape. The first table it passes is the model.
        let header = open_checkpoint(&dir).map_err(init)?;
        let mut refusals = Vec::new();
        let mut found = None;
        for entry in registry::entries() {
            match check(entry.weights, entry.tapes, &header) {
                Ok(()) => {
                    found = Some(entry);
                    break;
                }
                Err(e) => refusals.push(format!("{}: {e}", entry.stem)),
            }
        }
        drop(header);
        let entry = found.ok_or_else(|| {
            init(format!(
                "no CUDA model in this build matches the checkpoint at {}:\n  {}",
                dir.display(),
                refusals.join("\n  ")
            ))
        })?;
        let t = Instant::now();
        let before = device.allocations();
        let weights = load_weights(device, &dir, entry.tapes, entry.weights).map_err(init)?;
        info!(
            "CUDA model {}: {} weight buffers loaded in {:.1} s; device allocations {} (was \
             {before})",
            entry.stem,
            weights.len(),
            t.elapsed().as_secs_f64(),
            device.allocations(),
        );
        let kv = entry.tapes.iter().find_map(|t| t.kv);
        let result_elems = (entry.tapes.iter())
            .map(|t| t.result.bytes as usize / 2)
            .max()
            .unwrap_or(0);
        let rows = (entry.tapes.iter()).map(|t| t.bucket_m as usize).max();
        let rows = rows.unwrap_or(0);
        self.ids.reserve(rows);
        self.positions.reserve(rows);
        self.chunk.reserve(rows);
        self.table
            .reserve(kv.map_or(0, |k| k.max_blocks_per_seq as usize));
        self.model = Some(Loaded {
            entry,
            dir,
            hf_config,
            weights: Some(weights),
            kv,
            result: vec![0; result_elems],
        });
        Ok(())
    }

    fn initialize_cache(
        &mut self,
        num_gpu_blocks: usize,
        _num_cpu_blocks: usize,
    ) -> ExecutorResult<()> {
        // Block 0 is the scheduler's null block: a pool of fewer than two has no usable block.
        if num_gpu_blocks < 2 {
            return Err(init(format!(
                "a KV pool of {num_gpu_blocks} blocks has no usable block: block 0 is the \
                 scheduler's null block"
            )));
        }
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| init("the CUDA device is not open"))?;
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| init("no CUDA model is loaded"))?;
        let weights =
            (model.weights.take()).ok_or_else(|| init("the CUDA KV cache is already built"))?;
        let tapes = model.entry.tapes;
        let num_blocks = u32::try_from(num_gpu_blocks)
            .map_err(|_| init(format!("{num_gpu_blocks} KV blocks do not fit u32")))?;
        let kv_pool = (model.kv.is_some()).then_some(KvPoolConfig { num_blocks });
        let gdn_pool = (tapes.iter().any(|t| t.gdn.is_some())).then_some(GdnPoolConfig {
            num_slots: SEQUENCES as u32,
        });
        let before = device.allocations();
        let player = CudaWorker::owning(device, tapes, weights, kv_pool, gdn_pool).map_err(init)?;
        info!(
            "CUDA worker: {num_gpu_blocks} KV blocks ({} bytes), GDN pool {} bytes, arena {} \
             bytes; device allocations {} (was {before})",
            player.kv_pool_bytes().unwrap_or(0),
            player.gdn_pool_bytes().unwrap_or(0),
            player.arena_bytes(),
            device.allocations(),
        );
        self.running = Some(Running {
            tapes: player,
            gdn: GdnSlotAllocator::new(SEQUENCES, None),
        });
        Ok(())
    }

    fn determine_available_memory(&mut self) -> ExecutorResult<usize> {
        let (free, _total) = self.device()?.mem_info().map_err(init)?;
        Ok(free)
    }

    fn execute_model(&mut self, sched: &SchedulerOutput) -> ExecutorResult<ModelRunnerOutput> {
        // 1. Requests that finished in the previous step: their slot and their history. Their KV
        //    blocks are the scheduler's to free.
        for id in &sched.finished_req_ids {
            self.input_batch.remove_request(id);
            if let Some(r) = &mut self.running {
                r.gdn.release(gdn_slot_key(id));
            }
            if let Some(d) = &self.device {
                info!(
                    "CUDA worker: request {id} finished; device allocations {}",
                    d.allocations()
                );
            }
        }
        // A preempted request resumes by recomputing from the scheduler's cursor, so its GDN state
        // restarts from zero like a new request's, as metal's worker does.
        if let (Some(r), Some(preempted)) = (&mut self.running, &sched.preempted_req_ids) {
            for id in preempted {
                r.gdn.release(gdn_slot_key(id));
            }
        }
        // 2. New requests: greedy sampling only, then their prompt and blocks (one KV cache group).
        for nr in &sched.scheduled_new_reqs {
            self.admit(
                &nr.req_id,
                nr.prompt_token_ids.as_deref().unwrap_or(&[]),
                nr.sampling_params.as_ref(),
                nr.block_ids.first().cloned().unwrap_or_default(),
                nr.num_computed_tokens,
            )?;
        }
        // Cached requests: the scheduler ships each one's whole block table, which replaces the
        // old; a resumed request starts again from its `num_computed_tokens`.
        let cached = &sched.scheduled_cached_reqs;
        for (i, id) in cached.req_ids.iter().enumerate() {
            let blocks = (cached.new_block_ids.get(i))
                .and_then(|g| g.as_ref())
                .and_then(|groups| groups.first());
            if let Some(blocks) = blocks {
                self.input_batch.update_blocks(id, blocks.clone());
            }
            if cached.resumed_req_ids.contains(id)
                && let Some(&computed) = cached.num_computed_tokens.get(i)
            {
                self.input_batch.set_tokens_in_pool(id, computed as usize);
            }
        }
        // 3. The one request this step runs.
        let Some((req_id, n)) = scheduled_request(&sched.num_scheduled_tokens).map_err(exec)?
        else {
            return Ok(ModelRunnerOutput::empty());
        };
        let start = Self::computed(sched, req_id).ok_or_else(|| {
            exec(format!(
                "request {req_id} is scheduled and the step lists it as neither new nor cached"
            ))
        })?;
        let token = self.step(req_id, start, n)?;
        let mut out =
            ModelRunnerOutput::from_ordered(vec![req_id.to_string()], vec![token.unwrap_or(0)]);
        if token.is_none() {
            out.sampled_token_ids[0].clear();
        }
        Ok(out)
    }

    /// The prefill buckets' largest: no step may run more tokens than one forward holds.
    fn prefill_bucket_max_m(&self) -> Option<u32> {
        let model = self.model.as_ref()?;
        model.entry.tapes.iter().map(|t| t.bucket_m).max()
    }

    /// One block table of tokens: what the attention kernels address for a sequence.
    fn kv_max_addressable_tokens(&self) -> Option<usize> {
        self.kv()
            .map(|k| k.max_blocks_per_seq as usize * k.block_size as usize)
    }

    /// One sequence's whole block table, plus the scheduler's null block 0.
    fn kv_cache_num_blocks_override(&self) -> Option<usize> {
        self.kv().map(|k| k.max_blocks_per_seq as usize + 1)
    }

    /// Per KV layer, a K and a V block of `kv_heads · block_size · head_dim` bf16.
    fn kv_block_bytes(&self, block_size: usize) -> Option<usize> {
        self.kv().and_then(|k| kv_block_bytes(&k, block_size))
    }

    /// One sequence at a time: the GDN pool holds one slot and every tape runs one sequence.
    fn max_num_seqs_override(&self) -> Option<usize> {
        Some(SEQUENCES)
    }

    /// One block table for every attention layer: no sliding-window groups.
    fn supports_hybrid_swa_kv(&self) -> bool {
        false
    }

    fn architecture(&self) -> Option<String> {
        let model = self.model.as_ref()?;
        model.hf_config.architectures.first().cloned()
    }

    fn shutdown(&mut self) {
        self.running = None;
        self.model = None;
    }

    fn rank(&self) -> usize {
        self.cfg.tp_rank
    }

    fn local_rank(&self) -> usize {
        self.cfg.tp_rank
    }

    fn is_driver_worker(&self) -> bool {
        self.cfg.tp_rank == 0
    }
}

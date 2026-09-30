// SPDX-License-Identifier: Apache-2.0
//! O2 — the runtime bit-exactness oracle for the baked metal tapes.
//!
//! For one cached checkpoint per case, this loads the model through the same public entry points
//! the metal worker uses (`GpuWeights::from_dir` → `scratchy_forward_compiler::try_load`), builds
//! the KV pool the way `MetalWorker::initialize_cache` does (uniform, or vLLM's group-shared hybrid
//! layout; a Gated-DeltaNet state pool when the arch has one; TurboQuant where the build compiled
//! the model's KV codec as it — `--features turboquant`),
//! then runs ONE sequence at a time — a prefill over each fixed prompt, then greedy decode — and
//! prints, per step, the greedy token and a SHA-256 of the full logits row the step produced (the
//! terminal arena slot). Decoders hash the sampled row; encoders hash every row.
//!
//! The printed `O2` lines are the oracle: `/tmp/metal-unify/gate-o2.sh` records them once from a
//! baseline tree and requires every later tree to print exactly the same lines.
//!
//! `#[ignore]`d: a case needs a Metal 4 GPU and its checkpoint in the local Hub cache. Run one case
//! per process — the wired KV pool and weights are never torn down between cases:
//!
//! ```text
//! cargo test --release -p scratchy-models --features metal,<stem> --test metal_o2_logits \
//!     -- --ignored --exact --nocapture <case>
//! ```
#![cfg(all(feature = "metal", target_os = "macos"))]

// Nothing here names a model module — they register through `inventory` — so the crate has to be
// linked explicitly or `try_load` sees no registrations at all.
extern crate scratchy_models as _;

use std::path::PathBuf;
use std::sync::Arc;

use scratchy_core_config::{LayerKvGeometry, compute_hybrid_kv_layout};
use scratchy_core_model::weight::HfModelConfig;
use scratchy_forward_compiler::{HfFingerprint, ScratchyWeights, hash_json_value, try_load};
use scratchy_target_metal::gdn_state::GdnStatePool;
use scratchy_target_metal::interpreter::metal::{BLOCKS_PER_CHUNK, MetalDtype, TapePlay};
use scratchy_target_metal::kv_cache::KvCachePool;
use scratchy_target_metal::single_buffer_kv::SingleBufferKvLayer;
use scratchy_target_metal::weights::GpuWeights;
use scratchy_target_metal::{
    DType, ForwardCtx, ForwardCtxHandle, ForwardDeviceHandle, GpuDevice, GpuTensor, MetalAllocator,
    MetalMem, PoolMem, TensorView, detect_device,
};
use sha2::{Digest, Sha256};

/// How the baked tapes are played for one step. The tape itself is fixed at expansion; this picks
/// the realization (`GpuDevice::metal_tape_play`, read by the forward). A decoder case runs both
/// over the same prompts, and the gate requires identical bytes.
#[derive(Clone, Copy, Debug)]
enum ExecPath {
    /// One MTL4 dispatch per tape command.
    Dispatch,
    /// The decode megakernel runs the bucket-1 tape carries, every other command dispatched.
    Megakernel,
}

impl ExecPath {
    fn name(self) -> &'static str {
        match self {
            Self::Dispatch => "dispatch",
            Self::Megakernel => "megakernel",
        }
    }

    fn play(self) -> TapePlay {
        match self {
            Self::Dispatch => TapePlay::Dispatch,
            Self::Megakernel => TapePlay::Megakernel,
        }
    }

    /// Every path a case of `rows` runs: an encoder has no decode step to play differently.
    fn all_for(rows: Rows) -> &'static [ExecPath] {
        match rows {
            Rows::Sampled => &[Self::Dispatch, Self::Megakernel],
            Rows::All => &[Self::Dispatch],
        }
    }
}

/// Which rows of the step's terminal slot are the oracle.
#[derive(Clone, Copy)]
enum Rows {
    /// Decoder: the one row the lm_head slice samples (the last token of the step).
    Sampled,
    /// Encoder: the terminal slot is `[n, hidden]` hidden states — every row.
    All,
}

struct Case {
    name: &'static str,
    /// Hub repo id, resolved in the local cache only (never downloaded here).
    repo: &'static str,
    rows: Rows,
    prompts: &'static [&'static str],
    /// Largest prefill bucket kept in the pool's ladder.
    bucket_cap: u32,
}

/// Fixed prompts, sized to land in different prefill buckets: 1-2, 5-8 and 25-35 tokens depending
/// on the tokenizer (bucket 1 or 2, then 8, then 64).
const PROMPTS: &[&str] = &[
    "Hi",
    "The capital of France is",
    "Explain in three short sentences why the sky looks blue during the day but red at sunset, \
     and what role Rayleigh scattering plays in both.",
];

/// One prompt of 65-512 tokens: with [`LONG_BUCKET_CAP`] its prefill runs in bucket 512, the
/// smallest bucket whose baked M5 tapes carry the lm_head last-token slice
/// (`GatherLastToken` → qmv → `ScatterFirstToLastRow`); buckets ≤ 64 run without it on M5.
/// Compiled with the one case that reads it.
#[cfg(feature = "llama-3.2-3b")]
const LONG_PROMPTS: &[&str] = &[
    "A lighthouse keeper on a remote northern island keeps a careful journal of every ship that \
     passes, the weather at dawn and dusk, the colour of the sea, and the birds that nest on the \
     cliffs below the lamp. One winter a storm cuts the island off for six weeks, the supply boat \
     cannot land, and the keeper must ration oil, food and paper while still lighting the lamp \
     every night. Write the journal entry for the first calm morning after the storm, describing \
     what the keeper sees, what has been lost, and what the keeper decides to do next.",
];

/// The ladder cap of [`LONG_PROMPTS`] cases (keeps bucket 512, drops 1024+).
#[cfg(feature = "llama-3.2-3b")]
const LONG_BUCKET_CAP: u32 = 512;

/// Greedy decode steps after each prefill (decoders only).
const DECODE_STEPS: usize = 16;

/// Paged-KV block size — the engine default (`CacheConfig::block_size`).
const BLOCK_SIZE: usize = 16;

/// One chunk's worth of blocks: every KV tensor is exactly one chunk, so nothing ever grows.
const NUM_BLOCKS: usize = BLOCKS_PER_CHUNK as usize;

/// Largest prefill bucket kept in the pool's ladder — the largest a prompt above needs.
const BUCKET_CAP: u32 = 64;

/// Snapshot directory of `repo` in the local Hub cache.
fn snapshot_dir(repo: &str) -> PathBuf {
    let root = hf_hub_downloader::cache::default_root();
    hf_hub_downloader::cache::cached_path(&root, repo, "main", "config.json")
        .and_then(|p| p.parent().map(PathBuf::from))
        .unwrap_or_else(|| panic!("{repo}: not in the local Hub cache (`scr model pull {repo}`)"))
}

/// A loaded model plus the runtime state one sequence needs.
struct Loaded {
    model: Box<dyn ScratchyWeights>,
    gpu: GpuDevice,
    kv: KvCachePool,
    /// Backing buffers of `kv`'s chunks; must outlive the pool.
    _kv_layers: Vec<SingleBufferKvLayer>,
    gdn: Option<GdnStatePool<PoolMem>>,
    /// Group 0 (full attention) block size — page-unified on hybrid layouts.
    full_block_size: usize,
    /// KV-cache groups (1 = uniform; 1 + sliding groups on vLLM's hybrid layout).
    num_groups: usize,
    tokenizer: tokenizers::Tokenizer,
}

fn load(repo: &str, bucket_cap: u32) -> Loaded {
    let dir = snapshot_dir(repo);
    let metal = detect_device().expect("no Metal 4 device");
    let device = Arc::new(metal.device.clone());
    let allocator = MetalAllocator::new((*device).clone());
    let mut gpu = GpuDevice::new(device.clone(), Arc::new(allocator.clone()));
    let mut weights = GpuWeights::from_dir(&dir, allocator).expect("GpuWeights::from_dir");
    weights.set_target_dtype(DType::BF16);

    let hf = HfModelConfig::from_path(&dir).expect("config.json");
    let arch = hf.architectures.first().cloned().unwrap_or_default();
    let rope_scaling = hf.extra.get("rope_scaling");
    let fingerprint = HfFingerprint {
        rope_scaling_type: rope_scaling
            .and_then(|rs| rs.get("rope_type").or_else(|| rs.get("type")))
            .and_then(|v| v.as_str()),
        rope_scaling_hash: rope_scaling.map(hash_json_value),
        rope_theta: hf.rope_theta,
    };
    let max_model_len = hf.max_position_embeddings().unwrap_or(4096);
    // The tensors every generated `fingerprint_matches` keys on first — named in the refusal so a
    // checkpoint/variant mismatch (quantized embed, group size) reads off the panic.
    let probed: Vec<String> = [
        "model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.scales",
        "language_model.model.layers.0.self_attn.q_proj.scales",
    ]
    .iter()
    .filter_map(|p| weights.tensor_shape_any(p).map(|s| format!("{p}={s:?}")))
    .collect();
    let model = try_load(&mut weights, (), &arch, 1, 0, max_model_len, fingerprint)
        .expect("try_load")
        .unwrap_or_else(|| {
            let linked: Vec<&str> = scratchy_forward_compiler::inventory::iter::<
                scratchy_forward_compiler::ScratchyArchRegistration,
            >()
            .map(|r| r.arch_name)
            .collect();
            panic!(
                "{repo} ({arch}): no compiled variant claims this checkpoint \
                 ({fingerprint:?}; {probed:?}); linked arch registrations: {linked:?}"
            )
        });

    let head_dim = model.head_dim() as usize;
    let kv_heads = model.num_key_value_heads() as usize;
    let layers = model.num_hidden_layers() as usize;

    let cache_dtype = match model.metal_dtype() {
        MetalDtype::Bf16 => DType::BF16,
        MetalDtype::F16 => DType::F16,
        MetalDtype::Int4 => panic!("{repo}: an int4 KV cache does not exist"),
    };
    let elem_bytes = cache_dtype.size_bytes();
    let blocks_per_chunk = BLOCKS_PER_CHUNK as usize;
    let per_layer_block_elems: Option<Vec<usize>> = model
        .per_layer_kv_token_elems()
        .map(|v| v.iter().map(|e| e * BLOCK_SIZE).collect());
    let chunk_bytes = blocks_per_chunk * kv_heads * BLOCK_SIZE * head_dim * elem_bytes;
    // The page-differentiated (gemma-4) case only — the same trigger the scheduler uses.
    let hybrid = model.per_layer_kv_token_elems().and_then(|elems| {
        let max_e = *elems.iter().max()?;
        let geom: Vec<LayerKvGeometry> = elems
            .iter()
            .map(|&e| LayerKvGeometry {
                is_sliding: e == max_e,
                num_kv_heads: e,
                head_size: 1,
                head_size_v: None,
                sliding_window: (e == max_e).then_some(1),
            })
            .collect();
        compute_hybrid_kv_layout(&geom, BLOCK_SIZE, usize::MAX / 2, elem_bytes)
    });
    let num_tensors = hybrid.as_ref().map_or(layers, |h| h.group_size);
    let full_block_size = hybrid.as_ref().map_or(BLOCK_SIZE, |h| h.full_block_size());

    let residency = gpu.allocator.residency().clone();
    let mut kv_layers: Vec<SingleBufferKvLayer> = (0..num_tensors * 2)
        .map(|slot| {
            let bytes = match (&hybrid, &per_layer_block_elems) {
                (None, Some(v)) => blocks_per_chunk * v[slot / 2] * elem_bytes,
                _ => chunk_bytes,
            };
            SingleBufferKvLayer::new(&device, &residency, bytes, 1).expect("SingleBufferKvLayer")
        })
        .collect();
    let block_cap = max_model_len.div_ceil(BLOCK_SIZE).clamp(1, NUM_BLOCKS);
    let n_slots = num_tensors * 2;
    let calls = std::cell::Cell::new(0usize);
    let mut kv = unsafe {
        let layers_ref = &mut kv_layers;
        KvCachePool::new_metal_chunked(
            layers,
            NUM_BLOCKS,
            BLOCK_SIZE,
            kv_heads,
            head_dim,
            block_cap,
            if hybrid.is_some() {
                None
            } else {
                per_layer_block_elems.clone()
            },
            hybrid.as_ref().map(|h| h.layer_to_tensor.clone()),
            cache_dtype,
            blocks_per_chunk,
            1,
            |bytes| {
                let c = calls.get();
                calls.set(c + 1);
                let layer = &mut layers_ref[c % n_slots];
                let chunk = layer.committed_chunks();
                layer
                    .commit_through(chunk + 1)
                    .map_err(|e| anyhow::anyhow!("KV chunk commit: {e}"))?;
                Ok(MetalMem::from_buffer_with_offset(
                    layer.buffer_clone(),
                    chunk * layer.chunk_bytes(),
                    bytes,
                ))
            },
            |bytes| Ok(MetalMem::new_pinned(&device, &residency, bytes)),
        )
    }
    .expect("KvCachePool::new_metal_chunked");
    if let Some(h) = hybrid.as_ref() {
        kv.set_kv_group_layout(h.num_groups(), h.layer_to_group_u32());
    }
    residency.commit();
    kv.fill_chunk_tables(|m| m.gpu_address());

    let gdn = model.gdn_runtime_config().map(|cfg| {
        unsafe {
            GdnStatePool::new(
                layers,
                &cfg.linear_layers,
                1,
                cfg.conv_dim as usize,
                cfg.conv_kernel as usize,
                cfg.num_v_heads as usize,
                cfg.head_v_dim as usize,
                cfg.head_k_dim as usize,
                |bytes| Ok(MetalMem::new_pinned(&device, &residency, bytes)),
            )
        }
        .expect("GdnStatePool::new")
    });

    gpu.metal_bucket_max_m = Some(bucket_cap);
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .unwrap_or_else(|e| panic!("{repo}: tokenizer.json: {e}"));
    Loaded {
        model,
        gpu,
        num_groups: kv.num_kv_groups(),
        kv,
        _kv_layers: kv_layers,
        gdn,
        full_block_size,
        tokenizer,
    }
}

/// Block ids for one sequence: every group draws from the one shared pool, group after group
/// (block 0 is the null block), so no two groups ever share a block.
fn sequence_blocks(num_groups: usize, block_sizes: &[usize], max_tokens: usize) -> Vec<Vec<u32>> {
    let mut next = 1u32;
    (0..num_groups)
        .map(|g| {
            let n = max_tokens.div_ceil(block_sizes[g]) as u32;
            let ids: Vec<u32> = (next..next + n).collect();
            next += n;
            ids
        })
        .collect()
}

/// One step's host inputs, laid out as `MetalWorker::execute_model` lays them out for a batch of
/// one sequence.
struct StepInputs {
    ids: Vec<u32>,
    positions: Vec<u32>,
    /// Per KV group: slot mapping (`[n]`) and block-table row (`[block_cap]`).
    slot_mappings: Vec<Vec<u32>>,
    block_tables: Vec<Vec<u32>>,
    cu_seqlens_q: Vec<u32>,
    seqused_k: Vec<u32>,
    last_token_indices: Vec<u32>,
    gdn_indices: Vec<i32>,
    gdn_fresh: Vec<u32>,
}

fn step_inputs(
    l: &Loaded,
    blocks: &[Vec<u32>],
    ids: &[u32],
    start: usize,
    fresh: bool,
) -> StepInputs {
    let n = ids.len();
    let ctx_len = start + n;
    let stride = l.kv.max_blocks_per_seq;
    let mut slot_mappings = Vec::with_capacity(l.num_groups);
    let mut block_tables = Vec::with_capacity(l.num_groups);
    for (g, group_blocks) in blocks.iter().enumerate() {
        let bs = if g == 0 {
            l.full_block_size
        } else {
            BLOCK_SIZE
        };
        slot_mappings.push(
            (start..ctx_len)
                .map(|p| group_blocks[p / bs] * bs as u32 + (p % bs) as u32)
                .collect(),
        );
        let used = ctx_len.div_ceil(bs);
        let mut row = vec![0u32; stride];
        row[..used].copy_from_slice(&group_blocks[..used]);
        block_tables.push(row);
    }
    StepInputs {
        ids: ids.to_vec(),
        positions: (start as u32..ctx_len as u32).collect(),
        slot_mappings,
        block_tables,
        cu_seqlens_q: vec![0, n as u32],
        seqused_k: vec![ctx_len as u32],
        last_token_indices: vec![n as u32 - 1],
        gdn_indices: vec![0],
        gdn_fresh: vec![u32::from(fresh)],
    }
}

fn view<'a, T>(v: &'a [T], shape: &[usize], dtype: DType) -> TensorView<'a> {
    unsafe { TensorView::from_raw(GpuTensor::new(v.as_ptr() as *mut u8, shape, dtype)) }
}

/// Run one step; return the terminal slot's `[n, width]` bytes (bf16/f16 raw) and `width`.
fn run_step(l: &mut Loaded, path: ExecPath, s: &StepInputs) -> (Vec<u8>, usize) {
    let n = s.ids.len();
    let stride = l.kv.max_blocks_per_seq;
    let u32t = DType::U32;
    let ctx = ForwardCtx {
        input_ids: view(&s.ids, &[n], u32t),
        positions: view(&s.positions, &[n], u32t),
        slot_mapping: view(&s.slot_mappings[0], &[n], u32t),
        cu_seqlens_q: view(&s.cu_seqlens_q, &[2], u32t),
        seqused_k: view(&s.seqused_k, &[1], u32t),
        span_ids: None,
        block_table: view(&s.block_tables[0], &[1, stride], u32t),
        sliding_slot_mappings: s.slot_mappings[1..]
            .iter()
            .map(|v| view(v, &[n], u32t))
            .collect(),
        sliding_block_tables: s.block_tables[1..]
            .iter()
            .map(|v| view(v, &[1, stride], u32t))
            .collect(),
        max_seqlen_q: n,
        max_seqlen_k: s.seqused_k[0] as usize,
        kv_cache: &l.kv,
        mm_embeds: None,
        embed_patches: &[],
        vision_rope_cos: None,
        vision_rope_sin: None,
        vision_rope_freqs: None,
        pixels: None,
        pos_embeds: None,
        vision_cu_seqlens_full: None,
        vision_cu_seqlens_window: None,
        vision_max_seqlen_full: None,
        vision_max_seqlen_window: None,
        vision_window_index: None,
        vision_reverse_indices: None,
        vision_position_ids: None,
        gdn_state: l.gdn.as_ref(),
        gdn_state_indices: l
            .gdn
            .as_ref()
            .map(|_| view(&s.gdn_indices, &[1], DType::I32)),
        gdn_is_fresh: l.gdn.as_ref().map(|_| view(&s.gdn_fresh, &[1], u32t)),
        has_spec_tokens: false,
        last_token_indices: Some(view(&s.last_token_indices, &[1], u32t)),
    };
    l.gpu.metal_tape_play = path.play();
    let out = unsafe {
        l.model.forward_with_metal_followup(
            ForwardCtxHandle::new(&ctx),
            ForwardDeviceHandle::new(&mut l.gpu),
            n as u64,
            None,
        )
    };
    let t = out.as_gpu_tensor();
    let shape = t.shape().to_vec();
    assert_eq!(shape.len(), 2, "terminal slot is not [n, width]: {shape:?}");
    assert_eq!(
        shape[0] as usize, n,
        "terminal slot rows != step tokens: {shape:?}"
    );
    let width = shape[1] as usize;
    let bytes = unsafe { std::slice::from_raw_parts(t.raw_ptr() as *const u8, n * width * 2) };
    (bytes.to_vec(), width)
}

fn to_f32(bits: u16, dtype: MetalDtype) -> f32 {
    match dtype {
        MetalDtype::Bf16 => f32::from_bits(u32::from(bits) << 16),
        MetalDtype::F16 => half_to_f32(bits),
        MetalDtype::Int4 => unreachable!("int4 logits"),
    }
}

/// IEEE binary16 → f32.
fn half_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let frac = u32::from(h & 0x3ff);
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, f) => {
            let shift = f.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | (((f << shift) & 0x3ff) << 13)
        }
        (0x1f, f) => sign | 0x7f80_0000 | (f << 13),
        (e, f) => sign | ((e + 112) << 23) | (f << 13),
    };
    f32::from_bits(bits)
}

/// First index of the largest non-NaN value, and the NaN count.
fn argmax(row: &[u8], dtype: MetalDtype) -> (u32, usize) {
    let mut best = (0u32, f32::NEG_INFINITY);
    let mut nans = 0;
    for (i, c) in row.as_chunks::<2>().0.iter().enumerate() {
        let v = to_f32(u16::from_le_bytes(*c), dtype);
        if v.is_nan() {
            nans += 1;
        } else if v > best.1 {
            best = (i as u32, v);
        }
    }
    (best.0, nans)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn run_case(case: Case, paths: &[ExecPath]) {
    let mut l = load(case.repo, case.bucket_cap);
    let dtype = l.model.metal_dtype();
    let steps = match case.rows {
        Rows::Sampled => DECODE_STEPS,
        Rows::All => 0,
    };
    let block_sizes: Vec<usize> = (0..l.num_groups)
        .map(|g| {
            if g == 0 {
                l.full_block_size
            } else {
                BLOCK_SIZE
            }
        })
        .collect();
    println!(
        // Leading newline: libtest has already printed `test <name> ... ` without one.
        "\nO2META {} repo={} arch={} vocab={} kv={} kv_groups={} full_block={} gdn={} dtype={:?}",
        case.name,
        case.repo,
        l.model.arch_name(),
        l.model.vocab_size(),
        l.model.kv_codec(),
        l.num_groups,
        l.full_block_size,
        l.gdn.is_some(),
        dtype,
    );
    for &path in paths {
        for (pi, prompt) in case.prompts.iter().enumerate() {
            let prompt_ids: Vec<u32> = l
                .tokenizer
                .encode(*prompt, true)
                .unwrap_or_else(|e| panic!("tokenize: {e}"))
                .get_ids()
                .to_vec();
            let blocks = sequence_blocks(l.num_groups, &block_sizes, prompt_ids.len() + steps);
            let mut generated: Vec<u32> = Vec::with_capacity(steps + 1);
            let mut input = prompt_ids.clone();
            let mut start = 0usize;
            for step in 0..=steps {
                let s = step_inputs(&l, &blocks, &input, start, step == 0);
                let (bytes, width) = run_step(&mut l, path, &s);
                let n = input.len();
                let row_bytes = width * 2;
                let sampled = &bytes[(n - 1) * row_bytes..n * row_bytes];
                let hashed = match case.rows {
                    Rows::Sampled => sampled,
                    Rows::All => &bytes[..],
                };
                let (tok, nans) = argmax(sampled, dtype);
                println!(
                    "O2 {} path={} prompt={pi} step={step} n={n} width={width} tok={tok} nan={nans} sha256={}",
                    case.name,
                    path.name(),
                    sha256_hex(hashed),
                );
                generated.push(tok);
                start += n;
                input = vec![tok];
            }
            let ids: Vec<String> = generated.iter().map(u32::to_string).collect();
            println!(
                "O2 {} path={} prompt={pi} prompt_len={} tokens={}",
                case.name,
                path.name(),
                prompt_ids.len(),
                ids.join(",")
            );
            if matches!(case.rows, Rows::Sampled) {
                let text = l.tokenizer.decode(&generated, false).unwrap_or_default();
                println!(
                    "O2TEXT {} path={} prompt={pi} {text:?}",
                    case.name,
                    path.name()
                );
            }
        }
    }
}

macro_rules! o2_cases {
    (@or $default:ident) => { $default };
    (@or $default:ident, $given:ident) => { $given };
    ($( $feat:literal => $case:ident ($repo:literal, $rows:ident $(, $prompts:ident @ $cap:ident)?); )*) => {$(
        #[cfg(feature = $feat)]
        #[test]
        #[ignore = "needs a Metal 4 GPU and the checkpoint in the local Hub cache"]
        fn $case() {
            run_case(
                Case {
                    name: stringify!($case),
                    repo: $repo,
                    rows: Rows::$rows,
                    prompts: o2_cases!(@or PROMPTS $(, $prompts)?),
                    bucket_cap: o2_cases!(@or BUCKET_CAP $(, $cap)?),
                },
                ExecPath::all_for(Rows::$rows),
            );
        }
    )*};
}

// The runtime gate. Each case needs the compiled variant that claims its checkpoint, so each runs
// only under the gate build whose features (and quant preset) select it — see
// `/tmp/metal-unify/gate-builds.sh`. The suffix names the variant family the checkpoint matches.
o2_cases! {
    // Dense (no quant preset).
    "smollm2-135m" => o2_smollm2_135m("HuggingFaceTB/SmolLM2-135M-Instruct", Sampled);
    "llama-3.2-1b" => o2_llama_3_2_1b("unsloth/Llama-3.2-1B-Instruct", Sampled);
    "granite-3.1-2b-instruct" => o2_granite_3_1_2b_instruct("ibm-granite/granite-3.1-2b-instruct", Sampled);
    "qwen2.5-0.5b" => o2_qwen2_5_0_5b("Qwen/Qwen2.5-0.5B-Instruct", Sampled);
    "qwen3-0.6b" => o2_qwen3_0_6b("Qwen/Qwen3-0.6B", Sampled);
    "modernbert-base" => o2_modernbert_base("answerdotai/ModernBERT-base", All);
    // mlx-affine-b4-g64 (+ gate8-qembed, the only preset qwen3-5-moe declares).
    "llama-3.2-3b" => o2_llama_3_2_3b_mlx("mlx-community/Llama-3.2-3B-Instruct-4bit", Sampled);
    "llama-3.2-3b" => o2_llama_3_2_3b_mlx_long("mlx-community/Llama-3.2-3B-Instruct-4bit", Sampled, LONG_PROMPTS @ LONG_BUCKET_CAP);
    "llama-3.2-1b" => o2_llama_3_2_1b_mlx("mlx-community/Llama-3.2-1B-Instruct-4bit", Sampled);
    "gemma-4-26b-a4b-it" => o2_gemma_4_26b_a4b_mlx("mlx-community/gemma-4-26b-a4b-it-4bit", Sampled);
    "gemma-3-1b-it" => o2_gemma_3_1b_mlx("mlx-community/gemma-3-1b-it-4bit", Sampled);
    "granite-3.3-2b-instruct" => o2_granite_3_3_2b_mlx("mlx-community/granite-3.3-2b-instruct-4bit", Sampled);
    "qwen2.5-0.5b" => o2_qwen2_5_0_5b_mlx("mlx-community/Qwen2.5-0.5B-Instruct-4bit", Sampled);
    "qwen2.5-3b" => o2_qwen2_5_3b_mlx("mlx-community/Qwen2.5-3B-Instruct-4bit", Sampled);
    "qwen3-0.6b" => o2_qwen3_0_6b_mlx("mlx-community/Qwen3-0.6B-4bit", Sampled);
    "qwen3.5-0.8b" => o2_qwen3_5_0_8b_mlx("mlx-community/Qwen3.5-0.8B-4bit", Sampled);
    "qwen3.5-35b-a3b" => o2_qwen3_5_35b_a3b_mlx("mlx-community/Qwen3.6-35B-A3B-4bit", Sampled);
    // mlx-affine-b4-g64-qembed (untied, quantized embedding) and mlx-affine-b4-g32.
    "qwen2.5-7b" => o2_qwen2_5_7b_mlx_qembed("mlx-community/Qwen2.5-7B-Instruct-4bit", Sampled);
    "qwen2-vl-2b-mlx-text-only" => o2_qwen2_vl_2b_text_mlx_qembed("mlx-community/Qwen2-VL-2B-Instruct-4bit", Sampled);
    "granite-4.1-3b" => o2_granite_4_1_3b_mlx_g32("mlx-community/granite-4.1-3b-4bit", Sampled);
}

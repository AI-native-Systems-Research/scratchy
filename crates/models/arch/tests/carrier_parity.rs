// SPDX-License-Identifier: Apache-2.0
//! UNIVERSAL CARRIER PARITY GATE: the SAME `dsl/<arch>.py` text, run as
//! torch-Python by the oracle (`tests/golden_gen_carrier.py`) vs compiled
//! through scratchy onto Metal, compared numerically.
//!
//! One generic test walks `tests/goldens/<arch>-<stem>/` — no per-arch
//! code. Everything model-specific is DECLARED data the repo already has:
//!   - the synthetic checkpoint (`checkpoint/model.safetensors` +
//!     `config.json`) the oracle writes from the same tensors it executed
//!   - `harness.json` — inputs geometry (num_tokens, bounds, kv heads,
//!     head_dim, layers, rms_eps)
//!   - the trait accessors on `ScratchyWeights` — pool sizing comes from
//!     the loaded model, not from this file
//!
//! Load goes through the inventory (`try_load`, arch_hint =
//! `config.json`'s `architectures[0]`) — the same dispatch the worker
//! uses — so the gate exercises the fingerprint sniff too: a checkpoint
//! that fails `fingerprint_matches` for EVERY compiled variant fails
//! here, not just in production.
//!
//! A text carrier runs as the worker runs a sequence: one prefill over the
//! leading `prefill_tokens`, then one decode step per remaining token over
//! the same KV pool and recurrent state. Each step's LAST logits row (the
//! lm_head scatter target — the same convention as the qwen3 gate) must
//! reach cosine > 0.99 against the oracle's row at that position, mirroring
//! the qwen3.5-VL green gate's threshold: bf16 accumulation over the whole
//! stack means a lower value is a real divergence between carrier text and
//! compiled tape, not noise.
//!
//! Regenerate goldens + checkpoints (torch venv):
//!   python tests/golden_gen_carrier.py --arch qwen3 --stem qwen3-0.6b --tiny
//!
//! Run (NO quant features — a preset replaces dense emission):
//!   cargo test -p scratchy-models --features metal,qwen3-0.6b-parity-tiny \
//!     --test carrier_parity -- --nocapture
//!
//! A `(stem, preset)` variant's golden (`golden_gen_carrier.py --preset P`)
//! runs in a build selecting that preset
//! (`--features ...,scratchy-quantizations/P`); every other build skips it.

#[cfg(all(target_os = "macos", feature = "metal"))]
mod imp {
    // The gate walks every goldens dir, but only arches whose stem feature
    // this build names have a compiled variant. Those variants register via
    // `inventory::submit!` ctors, which the linker dead-strips unless the
    // crate is force-linked — per the crate's own lib docs.
    use scratchy_models as _;

    use half::bf16;
    use objc2_metal::MTLCreateSystemDefaultDevice;
    use scratchy_forward_compiler::{ArchLoad, HfFingerprint, hash_json_value, try_load};
    use scratchy_target_metal::device::Device;
    use scratchy_target_metal::kv_cache::KvCachePool;
    use scratchy_target_metal::{
        DType, ForwardCtx, GpuDevice, GpuTensor, GpuWeights, MetalAllocator, MetalMem, TensorView,
    };
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    const GOLDENS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/goldens");
    const CONFIGS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/configs");
    const BLOCK_SIZE: usize = 16; // engine block size (metal worker default)
    const NUM_BLOCKS: usize = 128; // pool size the oracle built its identity layout over

    // ── raw little-endian dumps. The oracle's dump() converts EVERY
    // tensor to f32 before writing (`a.float().numpy()` — bf16 has no
    // numpy dtype), so integer-valued fixtures (input_ids, positions,
    // block_table, slot_mapping) are f32 bytes holding exact small
    // integers. goldens.json records the shapes. ────────────────
    fn load_bin_f32(path: &Path) -> Vec<f32> {
        std::fs::read(path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    fn load_bin_u32(path: &Path) -> Vec<u32> {
        load_bin_f32(path).into_iter().map(|v| v as u32).collect()
    }

    /// A ctx input over host memory — what the worker hands the forward,
    /// which reads every input on the host into its own runtime buffers.
    fn host_view<'a, T>(v: &'a [T], shape: &[usize], dtype: DType) -> TensorView<'a> {
        unsafe { TensorView::from_raw(GpuTensor::new(v.as_ptr() as *mut u8, shape, dtype)) }
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for i in 0..a.len() {
            let (x, y) = (a[i] as f64, b[i] as f64);
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    struct Case {
        dir: PathBuf,
        arch: String,
        stem: String,
        /// `harness.json`'s `vision_grid` object, present only for vision
        /// goldens (`@vision_forward` carriers): grid t/h/w, patch/merge
        /// sizes, n_merged. Its presence routes the case to the vision arm.
        #[cfg_attr(not(feature = "vision"), allow(dead_code))]
        vision_grid: Option<serde_json::Value>,
        /// `harness.json`'s `dense_moe`: whether the oracle EXECUTED a
        /// dense MoE (a hybrid schedule may declare experts no layer runs).
        dense_moe: bool,
        /// `harness.json`'s `oracle_quant`: whether the oracle EXECUTED the
        /// checkpoint's quantization — the dequantized MLX-affine weights
        /// the checkpoint stores.
        oracle_quant: bool,
        /// `harness.json`'s `prefill_tokens`: the leading tokens the gate
        /// prefills in one step; every later token is one decode step.
        prefill_tokens: usize,
        /// `harness.json`'s `preset`: the quant preset whose `(stem, preset)`
        /// variant the oracle generated this golden for (`--preset`), or
        /// `None` for the dense base.
        preset: Option<String>,
    }

    impl Case {
        /// Whether this build emits the `(stem, preset)` variant the golden
        /// was generated for. A selected preset REPLACES the dense emission
        /// of every arch whose `quantizations.json` declares it, so a dense
        /// golden belongs to a build selecting none of its arch's presets,
        /// and a preset golden to a build selecting that preset.
        fn emitted_by_this_build(&self) -> bool {
            let selected = scratchy_quantizations::enabled_presets();
            match &self.preset {
                Some(p) => selected.contains(&p.as_str()),
                None => {
                    let path = Path::new(CONFIGS)
                        .join(&self.arch)
                        .join("quantizations.json");
                    let Ok(text) = std::fs::read_to_string(&path) else {
                        return true;
                    };
                    let declared: serde_json::Value =
                        serde_json::from_str(&text).expect("quantizations.json parses");
                    // Each entry is a bare preset name or a one-key object.
                    !declared["quantizations"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(|e| {
                            e.as_str()
                                .or_else(|| e.as_object()?.keys().next().map(String::as_str))
                        })
                        .any(|p| selected.contains(&p))
                }
            }
        }
    }

    /// Every `goldens/<arch>-<stem>/` dir carrying a checkpoint — the
    /// oracle writes one for every carrier (text AND vision), so a
    /// missing dir means "regenerate", not "skip silently".
    fn discover() -> Vec<Case> {
        let mut out = Vec::new();
        let root = Path::new(GOLDENS);
        let rd = match std::fs::read_dir(root) {
            Ok(rd) => rd,
            Err(_) => return out,
        };
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if !name.contains('-') {
                continue;
            }
            let dir = e.path();
            if !dir.join("checkpoint").join("model.safetensors").is_file() {
                continue;
            }
            if !dir.join("harness.json").is_file() {
                continue;
            }
            let harness: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(dir.join("harness.json"))
                    .expect("harness.json present (checked above) but unreadable"),
            )
            .expect("harness.json parses");
            out.push(Case {
                arch: harness["arch"].as_str().expect("harness arch").into(),
                stem: harness["stem"].as_str().expect("harness stem").into(),
                vision_grid: harness.get("vision_grid").cloned(),
                dense_moe: harness["dense_moe"]
                    .as_bool()
                    .expect("harness dense_moe (regenerate the goldens)"),
                oracle_quant: harness["oracle_quant"]
                    .as_bool()
                    .expect("harness oracle_quant (regenerate the goldens)"),
                prefill_tokens: harness["prefill_tokens"]
                    .as_u64()
                    .expect("harness prefill_tokens (regenerate the goldens)")
                    as usize,
                preset: harness["preset"].as_str().map(String::from),
                dir,
            });
        }
        out.sort_by(|a, b| a.dir.cmp(&b.dir));
        out
    }

    fn run_case(case: &Case, device_arc: &Arc<Device>) {
        let dir = &case.dir;
        // ── weights: the oracle's synthetic checkpoint, through the
        // worker's load path (fingerprint sniff included) ──
        let ckpt = dir.join("checkpoint");
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(ckpt.join("config.json")).unwrap())
                .expect("checkpoint config.json parses");
        let arch_hint = cfg["architectures"][0]
            .as_str()
            .expect("config architectures[0]")
            .to_string();

        // Vision carriers (`@vision_forward`) route to the vision arm: the
        // oracle executed the tower standalone, so parity is the projected
        // `[n_merged, d_model]` output vs the compiled `vision_forward` —
        // the production MM entry point (`try_load_mm`), same dispatch the
        // worker uses.
        #[cfg(feature = "vision")]
        if case.vision_grid.is_some() {
            run_vision_case(case, &cfg, &arch_hint, device_arc);
            return;
        }
        #[cfg(not(feature = "vision"))]
        if case.vision_grid.is_some() {
            panic!(
                "{}/{}: vision golden but the compiler's vision registry is \
                 not compiled in (enable an arch-<X>-vl feature)",
                case.arch, case.stem
            );
        }

        let ids: Vec<u32> = load_bin_u32(&dir.join("input_ids.bin"));
        let pos: Vec<u32> = load_bin_u32(&dir.join("positions.bin"));
        let block_table: Vec<u32> = load_bin_u32(&dir.join("block_table.bin"));
        let slot_mapping: Vec<u32> = load_bin_u32(&dir.join("slot_mapping.bin"));
        let golden = load_bin_f32(&dir.join("logits.bin"));
        let n = ids.len();
        let prefill = case.prefill_tokens;
        assert!(!ids.is_empty(), "{}: empty input_ids", case.stem);
        assert!(
            (1..=n).contains(&prefill),
            "{}: prefill_tokens {prefill} outside 1..={n}",
            case.stem
        );
        assert_eq!(pos.len(), n, "{}: positions len", case.stem);
        assert_eq!(slot_mapping.len(), n, "{}: slot_mapping len", case.stem);
        assert_eq!(
            block_table.len(),
            NUM_BLOCKS,
            "{}: block_table row len",
            case.stem
        );
        assert_eq!(golden.len() % n, 0, "{}: logits divisible by n", case.stem);
        let vocab = golden.len() / n;
        let rope_theta = cfg["rope_theta"].as_f64();
        let rope_scaling = cfg.get("rope_scaling");
        // Mirror `extract_rope_scaling` (config.rs): only llama3 / longrope
        // / su / yarn / mrope types count as a scaling. Other types the
        // manifest ignores ("linear" on gemma3) must read as None here or
        // the baked fingerprint — which encoded the manifest's ignored
        // value — rejects its own checkpoint.
        let recognized = ["llama3", "longrope", "su", "yarn", "mrope"];
        let hf = HfFingerprint {
            rope_scaling_type: rope_scaling
                .and_then(|rs| rs.get("rope_type").or_else(|| rs.get("type")))
                .and_then(|v| v.as_str())
                .filter(|t| recognized.contains(t)),
            rope_scaling_hash: rope_scaling.map(hash_json_value),
            rope_theta,
        };
        let max_model_len = cfg["max_position_embeddings"].as_u64().unwrap_or(4096) as usize;

        // A config carrying a quantization_config names a *quant* variant
        // (bnb nf4, fp8-block, ...). The oracle executes a quantization only
        // when it stores one it computes with exactly — MLX-affine weights,
        // executed dequantized (`oracle_quant`); any other it runs dense, and
        // running that here would test the dense variant again under a quant
        // name: a false green.
        if cfg.get("quantization_config").is_some() && !case.oracle_quant {
            eprintln!(
                "skip {}/{}: config carries a quantization_config the oracle did not execute",
                case.arch, case.stem
            );
            return;
        }

        // Dense MoE: the metal backend has no dense-MoE realization — the
        // MoE kernels require affine-quantized expert storage
        // (`to_wavefront.rs`: "MoE without affine expert storage has no
        // metal realization"), so a dense-MoE checkpoint refuses at pool
        // construction with `NoRung`. The same refusal fires for the
        // oracle's synthetic checkpoint. Comparing torch vs metal here
        // would compare an execution against a refusal: not parity, a
        // mechanism gap. Named loudly, not silently passed — such an arch
        // runs through the golden of its quantized variant instead
        // (`golden_gen_carrier.py --preset`). Read off what the oracle ran
        // (a dense MoE shim executed), not an arch list — nor the config's
        // expert-count keys, which a hybrid schedule may declare for no layer.
        if case.dense_moe {
            eprintln!(
                "DEFERRED {}/{}: dense MoE — metal requires affine expert storage; \
                 the oracle is dense-only (same refusal both sides)",
                case.arch, case.stem
            );
            return;
        }

        let allocator = MetalAllocator::new((**device_arc).clone());
        let mut device = GpuDevice::new(device_arc.clone(), Arc::new(allocator.clone()));
        let mut gw = GpuWeights::from_dir(&ckpt, allocator.clone()).expect("GpuWeights::from_dir");
        gw.set_target_dtype(DType::BF16);
        let model =
            match try_load(&mut gw, (), &arch_hint, 1, 0, max_model_len, hf).expect("try_load") {
                ArchLoad::Loaded(m) => m,
                ArchLoad::ArchNotCompiled => {
                    // No compiled variant claims this arch string — the build's
                    // feature scope names other arches. A skip, not a failure:
                    // the same goldens dir runs green in a build that scopes
                    // the arch in.
                    eprintln!(
                        "skip {}/{}: {arch_hint} not compiled into this build",
                        case.arch, case.stem
                    );
                    return;
                }
                // The arch IS compiled and every variant declined. The build's
                // quant scope explains that only when the golden was generated
                // for another (stem, preset) emission than this build makes — a
                // selected preset REPLACES its arch's dense emission.
                ArchLoad::NoVariantMatched if !case.emitted_by_this_build() => {
                    eprintln!(
                        "skip {}/{}: generated for preset {:?}, which is not the emission \
                         this build's quant scope makes for {}",
                        case.arch, case.stem, case.preset, case.arch
                    );
                    return;
                }
                // Otherwise no feature scope explains it — the synthetic
                // checkpoint and the compiled variant disagree about shapes or
                // quantization layout. That is a real failure, not a scope skip.
                ArchLoad::NoVariantMatched => panic!(
                    "{}/{}: {arch_hint} is compiled, but no compiled variant matches this \
                     checkpoint — its shapes or quantization layout differ from every \
                     (stem, preset) pair in this build",
                    case.arch, case.stem,
                ),
            };

        // ── pools, sized from the loaded model (no per-arch constants) ──
        let layers = model.num_hidden_layers() as usize;
        let kv_heads = model.num_key_value_heads() as usize;
        let head_dim = model.head_dim() as usize;
        let per_layer = model.per_layer_kv_token_elems();
        // Hybrid KV (gemma4 SWA): the SAME shared layout the worker builds —
        // `compute_hybrid_kv_layout` over per-layer page proxies
        // (`per_layer_kv_token_elems`, head_size 1 — only the ratio drives
        // grouping), then the pool with `layer_to_tensor` sharing and the
        // group layout attached. Uniform models (`per_layer == None`) keep
        // the one-tensor-per-layer pool, byte-identical to before.
        let hybrid: Option<scratchy_core_config::HybridKvLayout> =
            per_layer.as_ref().and_then(|elems| {
                let max_e = *elems.iter().max()?;
                let geom: Vec<scratchy_core_config::LayerKvGeometry> = elems
                    .iter()
                    .map(|&e| scratchy_core_config::LayerKvGeometry {
                        is_sliding: e == max_e,
                        // Page proxy: head_size 1 keeps the per-class page RATIO.
                        num_kv_heads: e,
                        head_size: 1,
                        head_size_v: None,
                        sliding_window: if e == max_e { Some(1) } else { None },
                    })
                    .collect();
                scratchy_core_config::compute_hybrid_kv_layout(
                    &geom,
                    BLOCK_SIZE,
                    usize::MAX / 2,
                    2, // bf16 elem bytes — only the (ignored) num_blocks read
                )
            });
        let mut kv = unsafe {
            KvCachePool::new_metal_chunked(
                layers,
                NUM_BLOCKS,
                BLOCK_SIZE,
                kv_heads,
                head_dim,
                NUM_BLOCKS.max(1),
                // Page-unified elems on the hybrid path (uniform size, so
                // the pool's default is correct); `None` on the uniform
                // path too (no per-layer override there).
                None,
                // The worker's tensors (`kv_pool_tensors`): the hybrid's
                // group-shared positions, else a tensor per layer that holds
                // KV (a recurrent hybrid's attention layers), else per layer.
                match (hybrid.as_ref(), model.kv_cache_layers()) {
                    (Some(l), _) => Some(l.layer_to_tensor.clone()),
                    (None, Some(kv)) => {
                        Some(scratchy_core_config::compact_kv_layer_to_tensor(layers, kv))
                    }
                    (None, None) => None,
                },
                DType::BF16,
                128,        // BLOCKS_PER_CHUNK (metal target const)
                usize::MAX, // eager: one chunk covers our tokens
                |bytes| {
                    // new_pinned: allocates + holds residency in the set —
                    // a non-resident chunk reads structured garbage.
                    Ok(MetalMem::new_pinned(
                        device_arc,
                        allocator.residency(),
                        bytes,
                    ))
                },
                |bytes| {
                    Ok(MetalMem::new_pinned(
                        device_arc,
                        allocator.residency(),
                        bytes,
                    ))
                },
            )
            .expect("KvCachePool::new_metal_chunked")
        };
        if let Some(layout) = hybrid.as_ref() {
            kv.set_kv_group_layout(layout.num_groups(), layout.layer_to_group_u32());
        }
        kv.fill_chunk_tables(|m| m.gpu_address());

        // GDN / Mamba-2 state pool (recurrent hybrid arches). One slot for
        // our single sequence, marked fresh on the prefill step — the oracle
        // zero-inits state — and carried into every decode step after it.
        let gdn_pool = model.gdn_runtime_config().map(|gdn_cfg| unsafe {
            scratchy_target_metal::gdn_state::GdnStatePool::new(
                layers,
                &gdn_cfg.linear_layers,
                1,
                // A lane for a model that drafts: its tapes bind one, though this one never does.
                scratchy_target_metal::gdn_state::DraftLanes::new(gdn_cfg.checkpoint_rows(), 1),
                gdn_cfg.state_dims(),
                |bytes| {
                    // f32 conv/ssm state; MUST be StorageModeShared.
                    Ok(MetalMem::new_pinned(
                        device_arc,
                        allocator.residency(),
                        bytes,
                    ))
                },
            )
            .expect("GdnStatePool::new")
        });

        // Recurrent-state slot 0 for our one sequence: its `state_indices` record.
        let gdn_idx = scratchy_target_metal::gdn_state::GdnSeqSlots::plain(
            scratchy_target_metal::gdn_state::GdnSlot::new(0),
        )
        .words();

        // Sliding KV-cache groups (gemma4 SWA), the worker's encoding: group
        // 0 (full) keeps the dumped slot_mapping/block_table (identity block
        // ids; the page-unified full_block_size encoding of identity ids
        // agrees with the oracle's base-BLOCK_SIZE identity slots — both are
        // just abs_pos); each sliding group gets its OWN block-id range past
        // the full group's (the groups SHARE the pool's physical tensors, so
        // two groups writing the same block id would clobber each other),
        // encoded with the base BLOCK_SIZE. Built over every absolute
        // position once; each step slices its own tokens.
        let mut sliding_sm: Vec<Vec<u32>> = Vec::new();
        let mut sliding_bt: Vec<Vec<u32>> = Vec::new();
        if let Some(layout) = hybrid.as_ref() {
            let full_bs = layout.full_block_size().max(1);
            let full_blocks = n.div_ceil(full_bs);
            let slide_blocks = n.div_ceil(BLOCK_SIZE);
            for s in 0..(layout.num_groups() - 1) {
                // This group's block ids: [full_blocks + s*slide_blocks, …).
                let base = full_blocks + s * slide_blocks;
                // slot_mapping[t] = block_ids[abs/bs]*bs + abs%bs.
                sliding_sm.push(
                    (0..n)
                        .map(|t| ((base + t / BLOCK_SIZE) * BLOCK_SIZE + t % BLOCK_SIZE) as u32)
                        .collect(),
                );
                // block_table[logical_b] = physical id, same row stride as
                // the full table (the kernel's MAX_BLOCKS_PER_SEQ).
                sliding_bt.push((0..NUM_BLOCKS).map(|b| (base + b) as u32).collect());
            }
        }

        // ── steps: ONE prefill over the first `prefill` tokens, then one
        // decode step per remaining token, each fed the oracle's next id
        // (teacher forcing). The KV pool and the recurrent state persist
        // across steps; the oracle ran the whole sequence as one causal
        // forward, so step (start, len)'s last row must reproduce the
        // oracle's row at absolute position start + len - 1 — the KV append,
        // the paged decode attention and the conv/ssm state carry are all in
        // that comparison.
        eprintln!(
            "──────── {}/{} CARRIER PARITY ────────",
            case.arch, case.stem
        );
        let steps = std::iter::once((0, prefill)).chain((prefill..n).map(|p| (p, 1)));
        let mut diverged = Vec::new();
        for (start, len) in steps {
            let end = start + len;
            let (cu, su) = ([0, len as u32], [end as u32]);
            // The worker ALWAYS passes per-sequence sample rows — `None`
            // means 0 and the lm_head slice trio's gather/scatter become
            // no-ops, leaving only stale arena at the last row.
            let lti = [len as u32 - 1];
            let fresh = [u32::from(start == 0)];
            let u32t = DType::U32;
            let ctx = ForwardCtx {
                input_ids: host_view(&ids[start..end], &[len], u32t),
                positions: host_view(&pos[start..end], &[len], u32t),
                slot_mapping: host_view(&slot_mapping[start..end], &[len], u32t),
                cu_seqlens_q: host_view(&cu, &[2], u32t),
                seqused_k: host_view(&su, &[1], u32t),
                span_ids: None,
                // Row stride = NUM_BLOCKS: the pool is built with
                // `max_blocks_per_seq = NUM_BLOCKS.max(1)`, and the macro
                // bakes the kernel's MAX_BLOCKS_PER_SEQ from that same pool
                // value — so host stride == kernel stride by construction
                // (the worker's own rule).
                block_table: host_view(&block_table, &[1, NUM_BLOCKS], u32t),
                sliding_slot_mappings: sliding_sm
                    .iter()
                    .map(|sm| host_view(&sm[start..end], &[len], u32t))
                    .collect(),
                sliding_block_tables: sliding_bt
                    .iter()
                    .map(|bt| host_view(bt, &[1, NUM_BLOCKS], u32t))
                    .collect(),
                max_seqlen_q: len,
                max_seqlen_k: end,
                kv_cache: &kv,
                gdn_state: gdn_pool.as_ref(),
                gdn_state_indices: gdn_pool
                    .as_ref()
                    .map(|_| host_view(&gdn_idx, &[gdn_idx.len()], DType::I32)),
                gdn_is_fresh: gdn_pool.as_ref().map(|_| host_view(&fresh, &[1], u32t)),
                has_spec_tokens: false,
                mm_embeds: None,
                embed_patches: &[],
                vision_rope_cos: None,
                vision_rope_sin: None,
                vision_rope_freqs: None,
                pixels: None,
                pos_embeds: None,
                target_hidden: None,
                hidden_out: None,
                vision_cu_seqlens_full: None,
                vision_cu_seqlens_window: None,
                vision_max_seqlen_full: None,
                vision_max_seqlen_window: None,
                vision_window_index: None,
                vision_reverse_indices: None,
                vision_position_ids: None,
                last_token_indices: Some(host_view(&lti, &[1], u32t)),
                device_inputs: &[],
                deferred: None,
                after: None,
                reads: scratchy_target_metal::ForwardReads::Outputs,
            };

            // ── RUN the compiled forward (the tape from the .py carrier) ──
            let ctx_h = scratchy_tensors::ForwardCtxHandle::new(&ctx);
            let dev_h = scratchy_tensors::ForwardDeviceHandle::new(&mut device);
            let out = unsafe { model.forward(ctx_h, dev_h, len as u64) };

            // ── readback: logits bf16 [len, vocab]; compare the LAST row ──
            let t = out.as_gpu_tensor();
            assert_eq!(t.dim(0), len, "{}: logits rows", case.stem);
            assert_eq!(t.dim(1), vocab, "{}: logits cols", case.stem);
            let raw = t.raw_ptr() as *const u16;
            let got_row = (len - 1) * vocab;
            let got: Vec<f32> = (0..vocab)
                .map(|i| bf16::from_bits(unsafe { *raw.add(got_row + i) }).to_f32())
                .collect();
            let gold_row = &golden[(end - 1) * vocab..end * vocab];

            let cos = cosine(&got, gold_row);
            let max_diff = got
                .iter()
                .zip(gold_row)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            let step = if start == 0 {
                format!("prefill {len} tokens")
            } else {
                format!("decode @ position {start}")
            };
            eprintln!("  {step}: cosine(logits[last]) = {cos:.6}, max_abs_diff = {max_diff:.4}");

            let nan = got.iter().filter(|x| x.is_nan() || x.is_infinite()).count();
            if nan > 0 {
                diverged.push(format!("{step}: {nan} NaN/Inf logits"));
            } else if cos.is_nan() || cos <= 0.99 {
                diverged.push(format!("{step}: cosine {cos}"));
            }
        }
        // Every step runs before the verdict, so one report shows whether a
        // divergence starts at the prefill or only once state carries.
        assert!(
            diverged.is_empty(),
            "{}: cosine <= 0.99 (or non-finite logits) — the compiled tape diverged \
             from the carrier-as-Python oracle (a real .py↔tape divergence) at: {}",
            case.stem,
            diverged.join("; ")
        );
    }

    /// Vision arm: the oracle executed the SAME `dsl/<arch>.py` text as a
    /// standalone torch tower over the dumped raw image; parity is the
    /// production `MultimodalForward::vision_forward` output (loaded via
    /// `try_load_mm` — the worker's own MM dispatch, fingerprint sniff
    /// included) against `logits.bin`'s `[n_merged, d_model]` rows, ALL of
    /// them (every row is an independent projection — there is no last-row
    /// convention on this side).
    #[cfg(feature = "vision")]
    fn run_vision_case(
        case: &Case,
        cfg: &serde_json::Value,
        arch_hint: &str,
        device_arc: &Arc<Device>,
    ) {
        use scratchy_forward_compiler::{EmbedPatch, PixelInput, try_load_mm};

        let dir = &case.dir;
        let golden = load_bin_f32(&dir.join("logits.bin"));
        let image = load_bin_f32(&dir.join("image.bin"));
        let vg = case
            .vision_grid
            .as_ref()
            .expect("vision_grid (dispatched on)");
        let num = |k: &str| vg[k].as_u64().expect("vision_grid {k}") as usize;
        let patch = num("patch_size");
        let merge = num("spatial_merge_size");
        let gt = num("t");
        let gh = num("h");
        let gw = num("w");
        let in_chans = num("in_chans");
        let pool_factor = num("pool_factor");
        let n_merged = num("n_merged");
        let height = gh * patch;
        let width = gw * patch;
        assert_eq!(
            image.len(),
            in_chans * height * width,
            "{}: image.bin size",
            case.stem
        );
        // The oracle's own grid asserts, restated: the geometry the tower
        // was shrunk to must tile exactly.
        assert_eq!(gt * gh * gw % (merge * merge).max(1), 0);

        let max_model_len = cfg["max_position_embeddings"].as_u64().unwrap_or(4096) as usize;
        // No rope scaling on the vision side; the checkpoint's (text-side)
        // value, if any, is what the emitted fingerprint baked.
        let rope_scaling = cfg.get("rope_scaling");
        let recognized = ["llama3", "longrope", "su", "yarn", "mrope"];
        let hf = HfFingerprint {
            rope_scaling_type: rope_scaling
                .and_then(|rs| rs.get("rope_type").or_else(|| rs.get("type")))
                .and_then(|v| v.as_str())
                .filter(|t| recognized.contains(t)),
            rope_scaling_hash: rope_scaling.map(hash_json_value),
            rope_theta: cfg["rope_theta"].as_f64(),
        };

        let allocator = MetalAllocator::new((**device_arc).clone());
        let mut device = GpuDevice::new(device_arc.clone(), Arc::new(allocator.clone()));
        let mut weights = GpuWeights::from_dir(dir.join("checkpoint"), allocator.clone())
            .expect("GpuWeights::from_dir");
        weights.set_target_dtype(DType::BF16);
        let model = match try_load_mm(&mut weights, (), arch_hint, 1, 0, max_model_len, hf)
            .expect("try_load_mm")
        {
            Some(m) => m,
            None => {
                // Same convention as the text arm: no compiled MM variant
                // claims this arch string — the build's feature scope names
                // other arches. The same goldens dir runs green in a build
                // that scopes the arch in.
                eprintln!(
                    "skip {}/{}: {arch_hint} not compiled into this build",
                    case.arch, case.stem
                );
                return;
            }
        };

        // One still image; `vision_forward` runs the production pixel pack
        // (patches_from_normalized_chw) itself — the gate feeds it the SAME
        // raw CHW f32 the oracle packed from. The placeholder carries the
        // merged token count (the worker's convention — `vision_forward`
        // fills only the grid fields on return).
        let pixels_in = [PixelInput {
            pixels: &image,
            height: height as u32,
            width: width as u32,
        }];
        let placeholders = [EmbedPatch {
            length: n_merged as u32,
            ..EmbedPatch::default()
        }];
        let dev_h = scratchy_tensors::ForwardDeviceHandle::new(&mut device);
        let (out, _patches) = unsafe { model.vision_forward(&pixels_in, &placeholders, dev_h) };

        // readback: [n_merged, d_model] bf16, every row compared
        let t = out.as_gpu_tensor();
        let d_model = t.dim(1);
        assert_eq!(t.dim(0), n_merged, "{}: vision output rows", case.stem);
        assert_eq!(
            golden.len(),
            n_merged * d_model,
            "{}: golden size",
            case.stem
        );
        assert_eq!(
            n_merged,
            gt * gh * gw / (merge * merge).max(1) / pool_factor.max(1),
            "{}: n_merged vs grid",
            case.stem
        );
        let raw = t.raw_ptr() as *const u16;
        let got: Vec<f32> = (0..n_merged * d_model)
            .map(|i| bf16::from_bits(unsafe { *raw.add(i) }).to_f32())
            .collect();

        let cos = cosine(&got, &golden);
        let max_diff = got
            .iter()
            .zip(&golden)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!(
            "──────── {}/{} VISION CARRIER PARITY ────────",
            case.arch, case.stem
        );
        eprintln!(
            "  grid {gt}x{gh}x{gw} (patch {patch}, merge {merge}) → {n_merged} merged rows x {d_model}"
        );
        eprintln!("  cosine(vision_out)   = {cos:.6}");
        eprintln!("  max_abs_diff         = {max_diff:.4}");

        let nan = got.iter().filter(|x| x.is_nan() || x.is_infinite()).count();
        assert_eq!(
            nan, 0,
            "{}: scratchy vision output contains NaN/Inf",
            case.stem
        );
        assert!(
            cos > 0.99,
            "{}: cosine {cos} <= 0.99 — the compiled vision tape diverged \
             from the carrier-as-Python oracle",
            case.stem
        );
    }

    #[test]
    fn carrier_py_vs_scratchy_parity() {
        let Some(raw_device) = MTLCreateSystemDefaultDevice() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let device_arc = Arc::new(raw_device);
        let cases = discover();
        // Goldens are local-only (gitignored *.bin) — same convention as
        // the qwen3 py-parity fixture: absent fixtures mean "regenerate",
        // not "fail". The sweep that emitted them is green locally.
        if cases.is_empty() {
            eprintln!(
                "skipping: no goldens under {GOLDENS} — regenerate with \
                 tests/golden_gen_carrier.py --tiny (see its header)"
            );
            return;
        }
        let mut failed = Vec::new();
        for case in &cases {
            // Each case gets its own device + allocator: the residency
            // set / arena must not leak across models.
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_case(case, &device_arc)
            })) {
                Ok(()) => {}
                Err(_) => failed.push(format!("{}/{}", case.arch, case.stem)),
            }
        }
        assert!(
            failed.is_empty(),
            "carrier parity FAILED for: {}",
            failed.join(", ")
        );
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
mod imp {}

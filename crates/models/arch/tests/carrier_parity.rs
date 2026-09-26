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
//! Cosine > 0.99 on the LAST logits row (the lm_head scatter target —
//! the same convention as the qwen3 gate), mirroring the qwen3.5-VL
//! green gate's threshold: bf16 accumulation over the whole stack means
//! a lower value is a real divergence between carrier text and compiled
//! tape, not noise.
//!
//! Regenerate goldens + checkpoints (torch venv):
//!   python tests/golden_gen_carrier.py --arch qwen3 --stem qwen3-0.6b --tiny
//!
//! Run (NO quant features — a preset replaces dense emission):
//!   cargo test -p scratchy-models --features metal,qwen3-0.6b-parity-tiny \
//!     --test carrier_parity -- --nocapture

#[cfg(target_os = "macos")]
mod imp {
    // The gate walks every goldens dir, but only arches whose stem feature
    // this build names have a compiled variant. Those variants register via
    // `inventory::submit!` ctors, which the linker dead-strips unless the
    // crate is force-linked — per the crate's own lib docs.
    use scratchy_models as _;

    use half::bf16;
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice as _, MTLResourceOptions};
    use scratchy_forward_compiler::{HfFingerprint, hash_json_value, try_load};
    use scratchy_target_metal::device::Device;
    use scratchy_target_metal::kv_cache::KvCachePool;
    use scratchy_target_metal::{
        DType, ForwardCtx, GpuDevice, GpuWeights, MetalAllocator, MetalMem,
    };
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    const GOLDENS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/goldens");
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
        assert!(!ids.is_empty(), "{}: empty input_ids", case.stem);
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
        // (bnb nf4, fp8-block, ...), but the oracle executes the carrier
        // dense — it has no bitsandbytes emulation. Running it here would
        // just test the dense variant again under a quant name: a false
        // green. Quant parity is a separate mechanism, not this gate.
        if cfg.get("quantization_config").is_some() {
            eprintln!(
                "skip {}/{}: config carries quantization_config — the oracle is dense-only",
                case.arch, case.stem
            );
            return;
        }

        // Dense MoE: the metal backend has no dense-MoE realization — the
        // MoE kernels require affine-quantized expert storage
        // (`to_wavefront.rs`: "MoE without affine expert storage has no
        // metal realization"), so a dense-MoE checkpoint refuses at pool
        // construction with `BucketLower`. The same refusal fires for the
        // oracle's synthetic checkpoint. Comparing torch vs metal here
        // would compare an execution against a refusal: not parity, a
        // mechanism gap. Named loudly, not silently passed — the unlock is
        // either a dense-MoE metal kernel or a quant-emulating oracle,
        // whichever lands first. Sniffed off config keys (num_local_experts
        // / n_routed_experts / num_experts — the three vocabularies the
        // bridge itself reads), not an arch list.
        let moe_keys = ["num_local_experts", "n_routed_experts", "num_experts"];
        if moe_keys.iter().any(|k| cfg.get(k).is_some()) {
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
                Some(m) => m,
                None => {
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
                hybrid.as_ref().map(|l| l.layer_to_tensor.clone()),
                DType::BF16,
                128,        // BLOCKS_PER_CHUNK (metal target const)
                usize::MAX, // eager: one chunk covers our tokens
                |bytes| {
                    let buf = device_arc
                        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
                        .expect("kv chunk alloc");
                    // residency BEFORE the first forward's lazy commit —
                    // a non-resident chunk reads structured garbage.
                    allocator.residency().insert(&buf);
                    Ok(MetalMem::from_buffer(buf))
                },
                |bytes| {
                    let buf = device_arc
                        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
                        .expect("kv table alloc");
                    allocator.residency().insert(&buf);
                    Ok(MetalMem::from_buffer(buf))
                },
            )
            .expect("KvCachePool::new_metal_chunked")
        };
        if let Some(layout) = hybrid.as_ref() {
            kv.set_kv_group_layout(layout.num_groups(), layout.layer_to_group_u32());
        }
        kv.fill_chunk_tables(|m| m.gpu_address());

        // GDN state pool (qwen3-5-style hybrid arches). One slot for our
        // single sequence, marked fresh — the oracle zero-inits state.
        let gdn_pool;
        let mut gdn_indices_view = None;
        let mut gdn_fresh_view = None;
        if let Some(gdn_cfg) = model.gdn_runtime_config() {
            let num_slots = 1usize;
            gdn_pool = Some(unsafe {
                scratchy_target_metal::gdn_state::GdnStatePool::new(
                    layers,
                    &gdn_cfg.linear_layers,
                    num_slots,
                    gdn_cfg.conv_dim as usize,
                    gdn_cfg.conv_kernel as usize,
                    gdn_cfg.num_v_heads as usize,
                    gdn_cfg.head_v_dim as usize,
                    gdn_cfg.head_k_dim as usize,
                    |bytes| {
                        // f32 conv/ssm state; MUST be StorageModeShared.
                        let buf = device_arc
                            .newBufferWithLength_options(
                                bytes,
                                MTLResourceOptions::StorageModeShared,
                            )
                            .expect("GDN state buffer alloc");
                        allocator.residency().insert(&buf);
                        Ok(MetalMem::from_buffer(buf))
                    },
                )
                .expect("GdnStatePool::new")
            });
            // slot 0 for our one sequence, fresh (state zero-init).
            let idx: [i32; 1] = [0];
            let fresh: [u32; 1] = [1];
            let idx_buf = device.alloc_gpu_tensor_from_host(&[1], DType::I32, unsafe {
                std::slice::from_raw_parts(idx.as_ptr() as *const u8, 4)
            });
            let fresh_buf = device.alloc_gpu_tensor_from_host(&[1], DType::U32, unsafe {
                std::slice::from_raw_parts(fresh.as_ptr() as *const u8, 4)
            });
            gdn_indices_view = Some(unsafe { idx_buf.as_view() });
            gdn_fresh_view = Some(unsafe { fresh_buf.as_view() });
        } else {
            gdn_pool = None;
        }

        // ── inputs (identity layout, the oracle's convention) ────
        let cu_seqlens: Vec<u32> = vec![0, n as u32];
        let seqused_k: Vec<u32> = vec![n as u32];
        let ids_buf = device.alloc_gpu_tensor_from_host(&[n], DType::U32, unsafe {
            std::slice::from_raw_parts(ids.as_ptr() as *const u8, n * 4)
        });
        let pos_buf = device.alloc_gpu_tensor_from_host(&[n], DType::U32, unsafe {
            std::slice::from_raw_parts(pos.as_ptr() as *const u8, n * 4)
        });
        let sm_buf = device.alloc_gpu_tensor_from_host(&[n], DType::U32, unsafe {
            std::slice::from_raw_parts(slot_mapping.as_ptr() as *const u8, n * 4)
        });
        let cu_buf = device.alloc_gpu_tensor_from_host(&[cu_seqlens.len()], DType::U32, unsafe {
            std::slice::from_raw_parts(cu_seqlens.as_ptr() as *const u8, cu_seqlens.len() * 4)
        });
        let su_buf = device.alloc_gpu_tensor_from_host(&[seqused_k.len()], DType::U32, unsafe {
            std::slice::from_raw_parts(seqused_k.as_ptr() as *const u8, seqused_k.len() * 4)
        });
        // Row stride = NUM_BLOCKS: the pool is built with
        // `max_blocks_per_seq = NUM_BLOCKS.max(1)`, and the macro bakes the
        // kernel's MAX_BLOCKS_PER_SEQ from that same pool value — so host
        // stride == kernel stride by construction (the worker's own rule).
        let bt_buf = device.alloc_gpu_tensor_from_host(&[1, NUM_BLOCKS], DType::U32, unsafe {
            std::slice::from_raw_parts(block_table.as_ptr() as *const u8, NUM_BLOCKS * 4)
        });
        // The worker ALWAYS passes per-sequence sample rows for prefill —
        // `None` means 0 and the lm_head slice trio's gather/scatter become
        // no-ops, leaving only stale arena at the last row.
        let lti: Vec<u32> = vec![n as u32 - 1];
        let lti_buf = device.alloc_gpu_tensor_from_host(&[1], DType::U32, unsafe {
            std::slice::from_raw_parts(lti.as_ptr() as *const u8, 4)
        });

        // Sliding KV-cache groups (gemma4 SWA), the worker's encoding: group
        // 0 (full) keeps the dumped slot_mapping/block_table (identity block
        // ids; the page-unified full_block_size encoding of identity ids
        // agrees with the oracle's base-BLOCK_SIZE identity slots — both are
        // just abs_pos); each sliding group gets its OWN block-id range past
        // the full group's (the groups SHARE the pool's physical tensors, so
        // two groups writing the same block id would clobber each other),
        // encoded with the base BLOCK_SIZE.
        let mut sliding_sm_views: Vec<scratchy_tensors::TensorView> = Vec::new();
        let mut sliding_bt_views: Vec<scratchy_tensors::TensorView> = Vec::new();
        if let Some(layout) = hybrid.as_ref() {
            let full_bs = layout.full_block_size().max(1);
            let full_blocks = n.div_ceil(full_bs);
            let slide_blocks = n.div_ceil(BLOCK_SIZE);
            for s in 0..(layout.num_groups() - 1) {
                // This group's block ids: [full_blocks + s*slide_blocks, …).
                let base = full_blocks + s * slide_blocks;
                // slot_mapping[t] = block_ids[abs/bs]*bs + abs%bs.
                let sm: Vec<u32> = (0..n)
                    .map(|t| ((base + t / BLOCK_SIZE) * BLOCK_SIZE + t % BLOCK_SIZE) as u32)
                    .collect();
                let sm_buf = device.alloc_gpu_tensor_from_host(&[n], DType::U32, unsafe {
                    std::slice::from_raw_parts(sm.as_ptr() as *const u8, n * 4)
                });
                sliding_sm_views.push(unsafe { sm_buf.as_view() });
                // block_table[logical_b] = physical id, same row stride as
                // the full table (the kernel's MAX_BLOCKS_PER_SEQ).
                let bt: Vec<u32> = (0..NUM_BLOCKS).map(|b| (base + b) as u32).collect();
                let bt_buf =
                    device.alloc_gpu_tensor_from_host(&[1, NUM_BLOCKS], DType::U32, unsafe {
                        std::slice::from_raw_parts(bt.as_ptr() as *const u8, NUM_BLOCKS * 4)
                    });
                sliding_bt_views.push(unsafe { bt_buf.as_view() });
            }
        }

        let ctx = ForwardCtx {
            input_ids: unsafe { ids_buf.as_view() },
            positions: unsafe { pos_buf.as_view() },
            slot_mapping: unsafe { sm_buf.as_view() },
            cu_seqlens_q: unsafe { cu_buf.as_view() },
            seqused_k: unsafe { su_buf.as_view() },
            span_ids: None,
            block_table: unsafe { bt_buf.as_view() },
            sliding_slot_mappings: sliding_sm_views,
            sliding_block_tables: sliding_bt_views,
            max_seqlen_q: n,
            max_seqlen_k: n,
            kv_cache: &kv,
            gdn_state: gdn_pool.as_ref(),
            gdn_state_indices: gdn_indices_view,
            gdn_is_fresh: gdn_fresh_view,
            has_spec_tokens: false,
            kv_turboquant: false,
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
            last_token_indices: Some(unsafe { lti_buf.as_view() }),
        };

        // ── RUN the compiled forward (the tape from the .py carrier) ──
        let ctx_h = scratchy_tensors::ForwardCtxHandle::new(&ctx);
        let dev_h = scratchy_tensors::ForwardDeviceHandle::new(&mut device);
        let out = unsafe { model.forward(ctx_h, dev_h, n as u64) };

        // ── readback: logits bf16 [n, vocab]; compare the LAST row ──
        let t = out.as_gpu_tensor();
        assert_eq!(t.dim(0), n, "{}: logits rows", case.stem);
        assert_eq!(t.dim(1), vocab, "{}: logits cols", case.stem);
        let raw = t.raw_ptr() as *const u16;
        let got_row = (n - 1) * vocab;
        let got: Vec<f32> = (0..vocab)
            .map(|i| bf16::from_bits(unsafe { *raw.add(got_row + i) }).to_f32())
            .collect();
        let gold_row: Vec<f32> = golden[(n - 1) * vocab..n * vocab].to_vec();

        let cos = cosine(&got, &gold_row);
        let max_diff = got
            .iter()
            .zip(&gold_row)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!(
            "──────── {}/{} CARRIER PARITY ────────",
            case.arch, case.stem
        );
        eprintln!("  n_tokens = {n} (comparing the last-token logits row)");
        eprintln!("  cosine(logits[last]) = {cos:.6}");
        eprintln!("  max_abs_diff         = {max_diff:.4}");

        let nan = got.iter().filter(|x| x.is_nan() || x.is_infinite()).count();
        assert_eq!(nan, 0, "{}: scratchy logits contain NaN/Inf", case.stem);
        assert!(
            cos > 0.99,
            "{}: cosine {cos} <= 0.99 — the compiled tape diverged from \
             the carrier-as-Python oracle (a real .py↔tape divergence)",
            case.stem
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

#[cfg(not(target_os = "macos"))]
mod imp {}

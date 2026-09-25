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
    use objc2_metal::{MTLDevice as _, MTLResourceOptions, MTLCreateSystemDefaultDevice};
    use scratchy_forward_compiler::{hash_json_value, try_load, HfFingerprint};
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
    }

    /// Every `goldens/<arch>-<stem>/` dir carrying a checkpoint — the
    /// oracle only writes one for text (`@forward`) arches, so vision
    /// goldens are skipped by construction, and a missing dir means
    /// "regenerate", not "skip silently".
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
            let harness: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(dir.join("harness.json")).expect(
                    "harness.json present (checked above) but unreadable",
                ))
                .expect("harness.json parses");
            out.push(Case {
                arch: harness["arch"].as_str().expect("harness arch").into(),
                stem: harness["stem"].as_str().expect("harness stem").into(),
                dir,
            });
        }
        out.sort_by(|a, b| a.dir.cmp(&b.dir));
        out
    }

    fn run_case(case: &Case, device_arc: &Arc<Device>) {
        let dir = &case.dir;
        let ids: Vec<u32> = load_bin_u32(&dir.join("input_ids.bin"));
        let pos: Vec<u32> = load_bin_u32(&dir.join("positions.bin"));
        let block_table: Vec<u32> = load_bin_u32(&dir.join("block_table.bin"));
        let slot_mapping: Vec<u32> = load_bin_u32(&dir.join("slot_mapping.bin"));
        let golden = load_bin_f32(&dir.join("logits.bin"));
        let n = ids.len();
        assert!(!ids.is_empty(), "{}: empty input_ids", case.stem);
        assert_eq!(pos.len(), n, "{}: positions len", case.stem);
        assert_eq!(slot_mapping.len(), n, "{}: slot_mapping len", case.stem);
        assert_eq!(block_table.len(), NUM_BLOCKS, "{}: block_table row len", case.stem);
        assert_eq!(golden.len() % n, 0, "{}: logits divisible by n", case.stem);
        let vocab = golden.len() / n;

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
                case.arch,
                case.stem
            );
            return;
        }

        let allocator = MetalAllocator::new((**device_arc).clone());
        let mut device = GpuDevice::new(device_arc.clone(), Arc::new(allocator.clone()));
        let mut gw = GpuWeights::from_dir(&ckpt, allocator.clone()).expect("GpuWeights::from_dir");
        gw.set_target_dtype(DType::BF16);
        let model = match try_load(&mut gw, (), &arch_hint, 1, 0, max_model_len, hf).expect("try_load") {
            Some(m) => m,
            None => {
                // No compiled variant claims this arch string — the build's
                // feature scope names other arches. A skip, not a failure:
                // the same goldens dir runs green in a build that scopes
                // the arch in.
                eprintln!("skip {}/{}: {arch_hint} not compiled into this build", case.arch, case.stem);
                return;
            }
        };

        // ── pools, sized from the loaded model (no per-arch constants) ──
        let layers = model.num_hidden_layers() as usize;
        let kv_heads = model.num_key_value_heads() as usize;
        let head_dim = model.head_dim() as usize;
        let max_blocks = model.max_blocks_per_seq();
        let per_layer = model.per_layer_kv_token_elems();
        if let Some(elems) = &per_layer {
            // Hybrid geometry (gemma4-style mixed full+sliding layers):
            // building the pool needs the per-layer is_sliding mask +
            // page-unified block sizes + per-sliding-group tables — facts
            // the trait doesn't carry yet. Deferred, NOT silently passed:
            // named loudly so the deferral is visible in every run.
            eprintln!(
                "DEFERRED {}/{}: hybrid KV geometry ({:?} distinct \
                 per-layer token elems) needs the sliding-group tables \
                 this gate doesn't build yet",
                case.arch,
                case.stem,
                elems.iter().collect::<std::collections::BTreeSet<_>>()
            );
            return;
        }
        let kv = unsafe {
            KvCachePool::new_metal_chunked(
                layers,
                NUM_BLOCKS,
                BLOCK_SIZE,
                kv_heads,
                head_dim,
                NUM_BLOCKS.max(1),
                None, // uniform geometry (hybrid deferred above)
                None, // one tensor per layer
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
                            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
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
            let idx_buf = device.alloc_gpu_tensor_from_host(
                &[1],
                DType::I32,
                unsafe { std::slice::from_raw_parts(idx.as_ptr() as *const u8, 4) },
            );
            let fresh_buf = device.alloc_gpu_tensor_from_host(
                &[1],
                DType::U32,
                unsafe { std::slice::from_raw_parts(fresh.as_ptr() as *const u8, 4) },
            );
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
        let bt_buf = device.alloc_gpu_tensor_from_host(
            &[1, max_blocks],
            DType::U32,
            unsafe {
                std::slice::from_raw_parts(
                    block_table.as_ptr() as *const u8,
                    NUM_BLOCKS * 4,
                )
            },
        );
        // The worker ALWAYS passes per-sequence sample rows for prefill —
        // `None` means 0 and the lm_head slice trio's gather/scatter become
        // no-ops, leaving only stale arena at the last row.
        let lti: Vec<u32> = vec![n as u32 - 1];
        let lti_buf =
            device.alloc_gpu_tensor_from_host(&[1], DType::U32, unsafe {
                std::slice::from_raw_parts(lti.as_ptr() as *const u8, 4)
            });

        let ctx = ForwardCtx {
            input_ids: unsafe { ids_buf.as_view() },
            positions: unsafe { pos_buf.as_view() },
            slot_mapping: unsafe { sm_buf.as_view() },
            cu_seqlens_q: unsafe { cu_buf.as_view() },
            seqused_k: unsafe { su_buf.as_view() },
            span_ids: None,
            block_table: unsafe { bt_buf.as_view() },
            sliding_slot_mappings: Vec::new(),
            sliding_block_tables: Vec::new(),
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
        eprintln!("──────── {}/{} CARRIER PARITY ────────", case.arch, case.stem);
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

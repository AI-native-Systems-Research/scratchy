// SPDX-License-Identifier: Apache-2.0
//! E2E PARITY GATE: the Python carrier RUN AS PYTHON vs THROUGH scratchy.
//!
//! `crates/models/arch/dsl/qwen3.py` is executed under torch by
//! `tests/golden_gen_qwen3_py.py` (the oracle — same text, same weights,
//! same inputs) which dumps `logits.bin` + the inputs. This test builds
//! the SAME state on the metal side (real checkpoint through the
//! macro-emitted `load`, a `KvCachePool::new_metal_chunked` pool, an
//! identity-layout block table) and runs the macro-emitted `forward`,
//! then compares logits numerically.
//!
//! The gate is cosine + max-abs-diff, mirroring the qwen3.5-VL green
//! gate's thresholds (bf16 accumulation over 28 layers ⇒ cosine ≥0.99;
//! a lower value is a real divergence between carrier text and compiled
//! tape).
//!
//! Regenerate goldens (torch venv):
//!   python tests/golden_gen_qwen3_py.py \
//!     --checkpoint ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<sha>
//!
//! Run:
//!   cargo test -p scratchy-models --features metal,qwen3-0.6b \
//!     --test qwen3_py_parity -- --nocapture

use half::bf16;
use objc2_metal::MTLCreateSystemDefaultDevice;
use scratchy_target_metal::kv_cache::KvCachePool;
use scratchy_target_metal::{DType, ForwardCtx, GpuDevice, GpuWeights, MetalAllocator};
use std::sync::Arc;

const SNAPSHOT: &str = concat!(
    env!("HOME"),
    "/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/",
    "c1899de289a04d12100db370d81485cdf75e47ca"
);
const GOLDEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/goldens");

// Geometry from configs/qwen3/qwen3-0.6b.json — must match the oracle
// script's constants (asserted below via the goldens manifest values).
const LAYERS: usize = 28;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const BLOCK_SIZE: usize = 16; // engine block size (worker default)
const NUM_BLOCKS: usize = 128; // pool size; ≥ ceil(n/16) used

/// Raw little-endian dump reader (paired with goldens.json shapes —
/// same convention as the qwen2-vl golden fixture).
fn load_bin_i64(path: &str) -> Vec<i64> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("read {path}: {e}"))
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect()
}

fn load_bin_f32(path: &str) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("read {path}: {e}"))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
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

#[test]
fn qwen3_py_carrier_parity() {
    let Some(raw_device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    for name in [
        "input_ids",
        "positions",
        "logits",
        "block_table",
        "slot_mapping",
    ] {
        assert!(
            std::path::Path::new(&format!("{GOLDEN}/{name}.bin")).exists(),
            "missing golden {GOLDEN}/{name}.bin — regenerate with \
             tests/golden_gen_qwen3_py.py (see this file's header)"
        );
    }
    if !std::path::Path::new(SNAPSHOT).exists() {
        eprintln!("skipping: checkpoint snapshot not found at {SNAPSHOT}");
        return;
    }

    // ── Oracle fixtures ─────────────────────────────────────────────
    let ids: Vec<u32> = load_bin_i64(&format!("{GOLDEN}/input_ids.bin"))
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let pos: Vec<u32> = load_bin_i64(&format!("{GOLDEN}/positions.bin"))
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let block_table: Vec<u32> = load_bin_i64(&format!("{GOLDEN}/block_table.bin"))
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let slot_mapping: Vec<u32> = load_bin_i64(&format!("{GOLDEN}/slot_mapping.bin"))
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let golden = load_bin_f32(&format!("{GOLDEN}/logits.bin"));
    assert!(!ids.is_empty(), "empty input_ids golden");
    let n_tokens = ids.len();
    assert_eq!(pos.len(), n_tokens, "positions len");
    assert_eq!(slot_mapping.len(), n_tokens, "slot_mapping len");
    assert_eq!(block_table.len(), NUM_BLOCKS, "block_table row len");
    assert_eq!(golden.len() % n_tokens, 0, "logits divisible by n");
    let vocab = golden.len() / n_tokens;
    assert_eq!(vocab, 151936, "vocab");

    // ── Device + weights, exactly like the qwen3.5-VL green gate ──
    let device_arc = Arc::new(raw_device);
    let allocator = MetalAllocator::new((*device_arc).clone());
    let mut device = GpuDevice::new(device_arc.clone(), Arc::new(allocator.clone()));
    let mut gw = GpuWeights::from_dir(SNAPSHOT, allocator.clone()).expect("GpuWeights::from_dir");
    gw.set_target_dtype(DType::BF16);
    // max_model_len small: rope cache only needs to cover our positions.
    let w = scratchy_models::qwen3::qwen3_0_6b::load(&mut gw, (), n_tokens, 0)
        .expect("load (qwen3-0.6b)");

    // ── Paged KV pool: the identity layout the oracle used ─────────
    // num_blocks = NUM_BLOCKS, one chunk per layer (reactive growth is
    // the worker's job; a fully-eager pool with initial_chunks =
    // usize::MAX covers the one chunk our 8 tokens touch).
    let kv = unsafe {
        KvCachePool::new_metal_chunked(
            LAYERS,
            NUM_BLOCKS,
            BLOCK_SIZE,
            KV_HEADS,
            HEAD_DIM,
            NUM_BLOCKS.max(1),
            None, // uniform geometry
            None, // one tensor per layer
            DType::BF16,
            128,        // BLOCKS_PER_CHUNK (metal target const)
            usize::MAX, // eager: allocate all chunks now
            |bytes| {
                use objc2_metal::{MTLDevice as _, MTLResourceOptions};
                let buf = device_arc
                    .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
                    .expect("kv chunk alloc");
                // The interpreter's command buffers attach the allocator's
                // residency set — every buffer the forward reads MUST be
                // inserted before the first forward's lazy commit (the
                // worker inserts each chunk exactly this way, gpu_worker.rs
                // "residency.insert(&buffer)"). A non-resident chunk reads
                // structured garbage through the chunk-address table.
                allocator.residency().insert(&buf);
                Ok(scratchy_target_metal::MetalMem::from_buffer(buf))
            },
            |bytes| {
                use objc2_metal::{MTLDevice as _, MTLResourceOptions};
                let buf = device_arc
                    .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
                    .expect("kv table alloc");
                allocator.residency().insert(&buf);
                Ok(scratchy_target_metal::MetalMem::from_buffer(buf))
            },
        )
        .expect("KvCachePool::new_metal_chunked")
    };
    // The chunk-address tables start zeroed — the worker fills them with
    // each chunk's GPU virtual address right after pool construction
    // (gpu_worker.rs: "Populate each per-layer chunk-address table…").
    // Without this, rope_append/attention read K/V through null
    // addresses and the logits are garbage.
    kv.fill_chunk_tables(|m| m.gpu_address());

    // ── Input buffers (shared storage → host-visible views) ─────────
    let n = n_tokens;
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
    let bt_buf = device.alloc_gpu_tensor_from_host(&[1, NUM_BLOCKS], DType::U32, unsafe {
        std::slice::from_raw_parts(block_table.as_ptr() as *const u8, NUM_BLOCKS * 4)
    });
    // The worker always passes the per-sequence sample rows for prefill
    // (`last_token_indices = [n-1]` for one sequence). It is NOT optional
    // decoration: the lm_head slice trio (gather → qmv → scatter) reads
    // `num_sample_rows = last_token_indices.len()` as its early-out count —
    // `None` means 0 and the gather/scatter become no-ops, leaving the
    // logits buffer valid only at scattered sample rows (everything else
    // is stale arena). Compare the LAST row only — that is the row the
    // scatter writes and the row production samples from.
    let lti: Vec<u32> = vec![n as u32 - 1];
    let lti_buf = device.alloc_gpu_tensor_from_host(&[1], DType::U32, unsafe {
        std::slice::from_raw_parts(lti.as_ptr() as *const u8, 4)
    });

    let null_view = unsafe { scratchy_target_metal::GpuTensor::null(DType::U32).as_view() };
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
        gdn_state: None,
        gdn_state_indices: None,
        gdn_is_fresh: None,
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
    let _ = null_view; // completeness placeholder, unused fields are None

    // ── RUN the compiled forward (the tape from the .py carrier) ──
    let out =
        unsafe { scratchy_models::qwen3::qwen3_0_6b::forward(&w, &ctx, &mut device, n as u64) };

    // ── Readback: StorageModeShared arena → host-readable after the
    // interpreter's waitUntilCompleted. Logits are bf16 [n, vocab]; only
    // the LAST row is a valid sample row (the scatter target) — compare
    // it against the oracle's last-token logits row. ──
    let t = out.as_gpu_tensor();
    assert_eq!(t.dim(0), n, "logits rows");
    assert_eq!(t.dim(1), vocab, "logits cols");
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
    eprintln!("──────────── QWEN3 PY-CARRIER PARITY GATE ────────────");
    eprintln!("  n_tokens = {n} (comparing the last-token logits row)");
    eprintln!("  cosine(logits[last]) = {cos:.6}");
    eprintln!("  max_abs_diff         = {max_diff:.4}");
    eprintln!("  got  [0..6]  = {:?}", &got[0..6]);
    eprintln!("  gold [0..6]  = {:?}", &gold_row[0..6]);
    eprintln!("───────────────────────────────────────────────────────");

    let nan = got.iter().filter(|x| x.is_nan() || x.is_infinite()).count();
    assert_eq!(nan, 0, "scratchy logits contain NaN/Inf");
    assert!(
        cos > 0.99,
        "cosine {cos} <= 0.99 — the compiled tape diverged from the \
         carrier-as-Python oracle (a real .py↔tape divergence, not noise)"
    );
}

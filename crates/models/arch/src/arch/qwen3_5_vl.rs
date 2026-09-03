include!(concat!(env!("OUT_DIR"), "/qwen3_5_vl.rs"));

/// Qwen3.5-VL CPU preprocessing. patch 16 × spatial_merge 2 → smart-resize
/// factor 32, **symmetric ±1 normalization** via `preprocess_symmetric_unit`
/// (`image_mean = image_std = 0.5` → `(px/255 − 0.5)/0.5`, matches the HF
/// `Qwen3VLImageProcessor` / mlx-vlm; CLIP mean/std is WRONG for this arch
/// and corrupts the ViT input — verified pixel_values cosine 0.99997 vs
/// mlx-vlm with this fn, vs 0.9946 with CLIP), `<|image_pad|>` placeholder,
/// per-image grid. `default_min_pixels` / `default_max_pixels` mirror the
/// processor's `size.shortest_edge` / `size.longest_edge` (65536 /
/// 16777216) so a small image (e.g. 224×224) smart-resizes UP to the
/// 256×256 / 16×16-patch grid the model expects.
pub const PROCESSOR: scratchy_vision::MmMetadata = scratchy_vision::MmMetadata {
    hf_token_id_key: "image_token_id",
    hf_token_id_default: 151655,
    size_policy: scratchy_vision::SizePolicy::SmartResize {
        factor: 32,
        default_min_pixels: 65536,
        default_max_pixels: 16_777_216,
    },
    tokens_per_image: scratchy_vision::TokensPerImage::PerImageGrid {
        spatial_merge_default: 2,
    },
    preprocess: scratchy_vision::preprocess::preprocess_symmetric_unit,
    default_image_size: 384,
    chat_template_image_part_type: "image",
    placeholder_policy: scratchy_vision::PlaceholderPolicy::RepeatMarker,
    mrope_positions: true,
    numbered_image_tag_marker: None,
};

// ───────────────────────────── GREEN GATE ──────────────────────────────
//
// Step 4 of the VL-on-metal bring-up: the first end-to-end RUN of the
// metal vision tower (loader + lowering have only ever *compiled*).
// Drives the macro-emitted `Weights` exactly like the production path
// (mirror `try_load_mm` + `VisionWrapper::vision_forward`'s ctx build),
// but feeds the *already-packed* golden `pixel_values` directly as the
// `pixels` tensor so the GPU tower is isolated from CPU pixel-packing —
// a clean parity check against the mlx-vlm `merger_out` golden.
//
// The gate is EXACT (cosine ~0.9997 vs the mlx-vlm `merger_out` golden,
// = bf16 accumulation over 27 blocks). Two fixes got it there:
//   (1) Step 6 — the learned `pos_embeds` is added after patch_embed
//       (host `fast_pos_embed_interpolate`; fed here from the golden).
//   (2) the patch_embed conv weight is permuted channels-first before
//       flattening (`flatten_conv_weight_channels_last`) — it ships
//       channels-LAST `[out,kt,kh,kw,in]`, and the old plain reshape
//       mispaired elements, corrupting only the high-variance patches
//       (the flat background patches stayed correct, masking the bug).
//
// Run: `cargo test -p scratchy-models \
//        --features metal,qwen3.5-9b --release green_gate -- --nocapture`
// Also gated on the `qwen3.5-9b` model feature specifically (model
// selection is fully opt-in, no default) — this test hardcodes that one
// model's emitted module path.
#[cfg(all(test, feature = "metal", feature = "qwen3.5-9b"))]
mod green_gate {
    use half::bf16;
    use objc2_metal::MTLCreateSystemDefaultDevice;
    use scratchy_target_metal::kv_cache::KvCachePool;
    use scratchy_target_metal::{DType, GpuDevice, GpuTensor, GpuWeights, MetalAllocator};
    use scratchy_target_metal::{ForwardCtx, VisionArchWeights};
    use scratchy_vision::{
        bf16_slice_as_bytes, build_cu_seqlens_i32, f32_slice_as_bytes, i32_slice_as_bytes,
    };
    use std::sync::Arc;

    const SNAPSHOT: &str = concat!(
        env!("HOME"),
        "/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit",
        "/snapshots/938d8919941c6e7efd3c7150eff7fe9d12afa631"
    );
    const GOLDEN: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tools/vision_parity/golden"
    );

    /// Minimal NumPy v1.0 reader: all our golden fixtures are
    /// `{'descr':'<f4', 'fortran_order':False, ...}`. The header is
    /// `6 magic + 2 version + 2 LE u16 header_len`; the payload follows.
    /// We only need the flat f32 values (shape is asserted by callers).
    fn load_npy_f32(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        assert_eq!(&bytes[0..6], b"\x93NUMPY", "{path}: not an npy file");
        assert_eq!(bytes[6], 1, "{path}: expected npy v1.x");
        let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let data = &bytes[10 + hlen..];
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
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
    fn metal_vision_tower_green_gate() {
        let Some(raw_device) = MTLCreateSystemDefaultDevice() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        if !std::path::Path::new(SNAPSHOT).exists() {
            eprintln!("skipping: checkpoint snapshot not found at {SNAPSHOT}");
            return;
        }

        // Single allocator shared between GpuWeights (loader arena) and
        // GpuDevice (worker-side buffer_for), exactly as the metal worker
        // does (gpu_worker.rs): MetalAllocator::clone shares the
        // arena vec via Arc<Mutex<…>>, so a weight uploaded through the
        // GpuWeights clone is reachable via the GpuDevice clone.
        let device_arc = Arc::new(raw_device);
        let allocator = MetalAllocator::new((*device_arc).clone());
        let mut device = GpuDevice::new(device_arc.clone(), Arc::new(allocator.clone()));

        let mut gw = GpuWeights::from_dir(SNAPSHOT, allocator).expect("GpuWeights::from_dir");
        gw.set_target_dtype(DType::BF16);
        // Conv3d patch-embed weight ships 5D `[1152, 2, 16, 16, 3]`
        // (channels-LAST). It must be permuted to `[out, in, kt, kh, kw]`
        // then flattened to `[1152, 1536]` so it pairs with the
        // channels-FIRST `[in, kt, kh, kw]` patch packing — a plain
        // reshape mispairs every element and silently corrupts the
        // high-variance patches (cos 0.83 not 1.0). The prelude
        // `try_load_mm` runs this but the bare `load()` does not.
        gw.flatten_conv_weight_channels_last("vision_tower.patch_embed.proj.weight", 0)
            .expect("flatten patch_embed.proj.weight");
        // `stream` is `()` on metal; max_model_len/tp_rank mirror the worker.
        // The macro emits `load`/`Weights` in a per-variant module.
        let w = super::qwen3_5_9b::load(&mut gw, (), 4096, 0).expect("load (vision weights)");

        // ── Inputs from the mlx-vlm golden (red_circle_224, grid 1×16×16) ──
        let cfg = w.vision_config();
        let grid_thw: Vec<(u32, u32, u32)> = vec![(1, 16, 16)];
        let total_l = 256usize;
        let feat = (cfg.in_chans as usize)
            * (cfg.temporal_patch_size as usize)
            * (cfg.patch_size as usize)
            * (cfg.patch_size as usize);
        assert_eq!(feat, 1536, "vision_in_features");
        let half_rot = cfg.half_rot();
        let merge2 = (cfg.spatial_merge_size as usize).pow(2);
        let out_rows = total_l / merge2; // 64
        let out_cols = cfg.d_model as usize; // 4096
        let n_out = out_rows * out_cols;

        // pixels: golden f32 [256,1536] → bf16 (already-packed patches).
        let pixels_f32 = load_npy_f32(&format!("{GOLDEN}/pixel_values.npy"));
        assert_eq!(pixels_f32.len(), total_l * feat);
        let pixels_bf16: Vec<u16> = pixels_f32
            .iter()
            .map(|&x| bf16::from_f32(x).to_bits())
            .collect();
        let pixels = device.alloc_gpu_tensor_from_host(
            &[total_l, feat],
            DType::BF16,
            bf16_slice_as_bytes(&pixels_bf16),
        );

        // cos/sin (bf16) — present for ctx completeness; the metal
        // vision_rope_2d kernel actually consumes the raw f32 `freqs`.
        let (cos_host, sin_host) = cfg.build_rope_cos_sin_bf16(&grid_thw, total_l);
        let cos = device.alloc_gpu_tensor_from_host(
            &[total_l, half_rot],
            DType::BF16,
            bf16_slice_as_bytes(&cos_host),
        );
        let sin = device.alloc_gpu_tensor_from_host(
            &[total_l, half_rot],
            DType::BF16,
            bf16_slice_as_bytes(&sin_host),
        );

        // raw f32 rope freqs (what vision_rope_2d.metal reads). Verified
        // == the mlx `rope_block0_q_freqs` golden (cos 1.0, maxdiff 0).
        let freqs_host = cfg.build_rope_freqs_f32(&grid_thw, total_l);
        assert_eq!(freqs_host.len(), total_l * half_rot);
        let freqs = device.alloc_gpu_tensor_from_host(
            &[total_l, half_rot],
            DType::F32,
            f32_slice_as_bytes(&freqs_host),
        );

        // pos_embeds: computed HOST-SIDE by `fast_pos_embed_interpolate`
        // from the learned `vision_tower.pos_embed.weight` table — the
        // exact production path. We validate it against the mlx-vlm
        // `pos_embeds.npy` golden, then feed the COMPUTED values into the
        // tower (so the gate exercises the real host interp end-to-end).
        let embed_dim = cfg.embed_dim as usize;
        let pe_table = gw
            .tensor_to_f32("vision_tower.pos_embed.weight")
            .expect("pos_embed table");
        // num_grid_per_side = sqrt(num_position_embeddings) = sqrt(table_rows).
        let num_grid = (((pe_table.len() / embed_dim) as f64).sqrt()).round() as usize;
        let mut pos_f32 = cfg.fast_pos_embed_interpolate(&grid_thw, num_grid, &pe_table, total_l);
        assert_eq!(pos_f32.len(), total_l * embed_dim, "pos_embeds shape");
        let pos_golden = load_npy_f32(&format!("{GOLDEN}/pos_embeds.npy"));
        let interp_cos = cosine(&pos_f32, &pos_golden);
        eprintln!(
            "  [pos_embed interp] cos(host fast_pos_embed_interpolate, golden) = {interp_cos:.6}"
        );
        assert!(
            interp_cos > 0.999,
            "host fast_pos_embed_interpolate cos {interp_cos} != golden — bilinear/merge-order bug"
        );
        let pos_bf16: Vec<u16> = pos_f32
            .iter()
            .map(|&x| bf16::from_f32(x).to_bits())
            .collect();
        let pos_embeds = device.alloc_gpu_tensor_from_host(
            &[total_l, embed_dim],
            DType::BF16,
            bf16_slice_as_bytes(&pos_bf16),
        );

        // cu_seqlens [0, 256], one segment (the single image).
        let (cu_host, max_seqlen) = build_cu_seqlens_i32(&grid_thw);
        let cu = device.alloc_gpu_tensor_from_host(
            &[cu_host.len()],
            DType::I32,
            i32_slice_as_bytes(&cu_host),
        );

        // kv_cache placeholder — vision body never reads it.
        let kv = KvCachePool::empty_for_vision();

        // ── ForwardCtx — mirrors VisionWrapper::vision_forward (vision_arch.rs
        // :302-346); non-windowed, no learned pos-embed → those fields None. ──
        let null_view = unsafe { GpuTensor::null(DType::U32).as_view() };
        let pixels_view = unsafe { pixels.as_view() };
        let pos_view = unsafe { pos_embeds.as_view() };
        let cu_view = unsafe { cu.as_view() };
        let cos_view = unsafe { cos.as_view() };
        let sin_view = unsafe { sin.as_view() };
        let freqs_view = unsafe { freqs.as_view() };
        let ctx = ForwardCtx {
            input_ids: null_view,
            positions: null_view,
            slot_mapping: null_view,
            cu_seqlens_q: cu_view,
            seqused_k: null_view,
            block_table: null_view,
            sliding_slot_mapping: null_view,
            sliding_block_table: null_view,
            max_seqlen_q: max_seqlen,
            max_seqlen_k: 0,
            kv_cache: &kv,
            mm_embeds: None,
            embed_patches: &[],
            vision_rope_cos: Some(cos_view),
            vision_rope_sin: Some(sin_view),
            vision_rope_freqs: Some(freqs_view),
            pixels: Some(pixels_view),
            pos_embeds: Some(pos_view),
            vision_cu_seqlens_full: None,
            vision_cu_seqlens_window: None,
            vision_max_seqlen_full: None,
            vision_max_seqlen_window: None,
            vision_window_index: None,
            vision_reverse_indices: None,
            vision_position_ids: None,
            last_token_indices: None,
            #[cfg(feature = "metal")]
            has_spec_tokens: false,
            #[cfg(feature = "metal")]
            gdn_state: None,
            #[cfg(feature = "metal")]
            gdn_state_indices: None,
            #[cfg(feature = "metal")]
            gdn_is_fresh: None,
            #[cfg(feature = "nccl")]
            tp_group: None,
        };

        // ── RUN the metal vision tower (forward_m_256 workload bucket) ──
        let projected = unsafe { w.vision_forward(&ctx, &mut device, total_l as u64) };

        // ── Readback: StorageModeShared arena ptr is host-readable after
        // the interpreter's waitUntilCompleted. ──
        let probe = "merger_out";
        let golden = load_npy_f32(&format!("{GOLDEN}/{probe}.npy"));
        let n_real = golden.len();
        let (rows, cols) = if probe == "merger_out" {
            (out_rows, out_cols)
        } else {
            (total_l, n_real / total_l) // [256, cols] goldens
        };
        let _ = (n_out,);
        let t = projected.as_gpu_tensor();
        let raw = t.raw_ptr() as *const u16;
        let out: Vec<f32> = (0..n_real)
            .map(|i| bf16::from_bits(unsafe { *raw.add(i) }).to_f32())
            .collect();

        // ── Coherence diagnostics ──
        let (out_rows, out_cols) = (rows, cols);
        let nan = out.iter().filter(|x| x.is_nan() || x.is_infinite()).count();
        let zero_rows = (0..out_rows)
            .filter(|&r| {
                out[r * out_cols..(r + 1) * out_cols]
                    .iter()
                    .all(|&x| x == 0.0)
            })
            .count();
        let mean = out.iter().map(|&x| x as f64).sum::<f64>() / out.len() as f64;
        let absmax = out.iter().fold(0f32, |m, &x| m.max(x.abs()));
        assert_eq!(golden.len(), out.len(), "golden shape");
        let cos_sim = cosine(&out, &golden);

        eprintln!("──────────────── METAL VISION GREEN GATE ────────────────");
        eprintln!("  cosine({probe}) = {cos_sim:.6}");
        eprintln!("  nan/inf = {nan}   zero_rows = {zero_rows}/{out_rows}");
        eprintln!("  mean = {mean:.5}   absmax = {absmax:.4}");
        eprintln!("  out [0..6]  = {:?}", &out[0..6]);
        eprintln!("  gold[0..6]  = {:?}", &golden[0..6]);
        eprintln!("──────────────────────────────────────────────────────────");

        assert_eq!(nan, 0, "vision output has NaN/Inf values");
        assert_eq!(
            zero_rows, 0,
            "vision output has all-zero rows (a stage produced nothing)"
        );
        assert!(
            absmax > 1e-3 && absmax < 1e4,
            "vision output magnitude {absmax} out of sane range"
        );
        // EXACT gate. Both bugs are fixed: (1) the learned pos_embed is
        // added (Step 6); (2) the patch_embed conv weight is permuted
        // channels-first before flattening (it ships channels-LAST, and a
        // plain reshape silently corrupted the high-variance patches —
        // see `flatten_conv_weight_channels_last`). The metal tower now
        // matches the mlx-vlm `merger_out` golden to bf16 accumulation
        // over 27 blocks (~0.9997).
        let threshold = 0.99;
        assert!(
            cos_sim > threshold,
            "cosine {cos_sim} too low (threshold {threshold}) — the metal ViT should \
             match merger_out to bf16 rounding (~1.0). A drop is a real regression in \
             the tower, the pos_embed wiring, or the patch_embed weight permute"
        );
    }
}

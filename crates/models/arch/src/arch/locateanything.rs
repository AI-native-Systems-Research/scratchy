include!(concat!(env!("OUT_DIR"), "/locateanything.rs"));

/// LocateAnything CPU preprocessing — mirrors mlx-vlm
/// `image_processing_locateanything.py` + `processing_locateanything.py`:
/// patch-cap resize (`in_token_limit` 25600 raw 14×14 patches, downscale
/// only) then ceil-pad both dims to a multiple of 28 (= patch · merge);
/// symmetric ±1 normalization (`image_mean = image_std = 0.5`);
/// `<IMG_CONTEXT>` (151665) placeholder bracketed by `<img>`/`</img>`
/// (151666/151667) with `(h/28)·(w/28)` context tokens per image; the
/// chat template renders literal `<image-N>` text that serve replaces
/// with the marker pre-tokenization (`numbered_image_tag_marker`).
/// Plain 1D positions — NO mrope.
pub const PROCESSOR: scratchy_vision::MmMetadata = scratchy_vision::MmMetadata {
    hf_token_id_key: "image_token_index",
    hf_token_id_default: 151665,
    size_policy: scratchy_vision::SizePolicy::PatchCapCeilPad {
        patch: 14,
        pad_multiple: 28,
        default_max_patches: 25600,
    },
    tokens_per_image: scratchy_vision::TokensPerImage::PerImageGrid {
        spatial_merge_default: 2,
    },
    preprocess: scratchy_vision::preprocess::preprocess_symmetric_unit,
    default_image_size: 224,
    chat_template_image_part_type: "image",
    placeholder_policy: scratchy_vision::PlaceholderPolicy::BracketRepeat {
        start_token_id: 151666,
        end_token_id: 151667,
    },
    mrope_positions: false,
    numbered_image_tag_marker: Some("<IMG_CONTEXT>"),
};

// ───────────────────────────── GREEN GATE ──────────────────────────────
//
// First end-to-end RUN of the MoonViT metal tower against the mlx-vlm
// per-stage goldens (tools/vision_parity/dump_golden_locateanything.py,
// red_circle_224, grid 1×16×16). Mirrors the qwen3-5-vl gate; the one
// LocateAnything-specific twist is patch ORDER: the goldens are dumped
// in mlx's ROW-major patch order, our tape runs WINDOW-major (spatial-
// merge packing), so patch-level tensors (`pixel_values`, `pos_embeds`,
// per-block probes) get the window-major permutation applied before
// upload/compare. Merged-token tensors (`projector_out`, the default
// gate) share the order in both frameworks — no permutation.
//
// Run: `cargo test -p scratchy-models \
//        --features metal,locateanything-3b --release \
//        green_gate -- --nocapture`
// Also gated on the `locateanything-3b` model feature specifically (model
// selection is fully opt-in, no default) — this test hardcodes that one
// model's emitted module path.
#[cfg(all(test, feature = "metal", feature = "locateanything-3b"))]
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
        "/.cache/huggingface/hub/models--mlx-community--LocateAnything-3B-4bit",
        "/snapshots/e4517cd171dfa6c5a376a50c5ef8b73949aea0b7"
    );
    const GOLDEN: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tools/vision_parity/golden_locateanything"
    );

    /// Minimal NumPy v1.0 reader (f32 little-endian payload).
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

    /// Row-major → window-major permutation for the 16×16 / merge-2 test
    /// grid: `perm[window_idx] = row_major_idx`. Same `(hb, wb, sh, sw)`
    /// walk as `patches_from_normalized_chw` / the rope+pos-emb builders.
    fn window_major_perm(grid_h: usize, grid_w: usize, s: usize) -> Vec<usize> {
        let mut perm = Vec::with_capacity(grid_h * grid_w);
        for hb in 0..grid_h / s {
            for wb in 0..grid_w / s {
                for sh in 0..s {
                    for sw in 0..s {
                        perm.push((hb * s + sh) * grid_w + (wb * s + sw));
                    }
                }
            }
        }
        perm
    }

    fn permute_rows_f32(src: &[f32], perm: &[usize], width: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(src.len());
        for &rm in perm {
            out.extend_from_slice(&src[rm * width..(rm + 1) * width]);
        }
        out
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

        let device_arc = Arc::new(raw_device);
        let allocator = MetalAllocator::new((*device_arc).clone());
        let mut device = GpuDevice::new(device_arc.clone(), Arc::new(allocator.clone()));

        let mut gw = GpuWeights::from_dir(SNAPSHOT, allocator).expect("GpuWeights::from_dir");
        gw.set_target_dtype(DType::BF16);
        // Conv2d patch-embed weight ships 4D channels-LAST `[1152,14,14,3]`
        // (mlx layout); permute channels first then flatten → `[1152, 588]`
        // in (c,h,w) element order, pairing with the channels-first packing.
        gw.flatten_conv_weight_channels_last("vision_tower.patch_embed.proj.weight", 0)
            .expect("flatten patch_embed.proj.weight");
        let w =
            super::locateanything_3b::load(&mut gw, (), 4096, 0).expect("load (vision weights)");

        // ── Inputs from the mlx-vlm golden (red_circle_224, grid 1×16×16) ──
        let cfg = w.vision_config();
        let grid_thw: Vec<(u32, u32, u32)> = vec![(1, 16, 16)];
        let total_l = 256usize;
        let feat = (cfg.in_chans as usize)
            * (cfg.temporal_patch_size as usize)
            * (cfg.patch_size as usize)
            * (cfg.patch_size as usize);
        assert_eq!(feat, 588, "vision_in_features");
        let half_rot = cfg.half_rot();
        let s = cfg.spatial_merge_size as usize;
        let merge2 = s.pow(2);
        let out_rows = total_l / merge2; // 64
        let out_cols = cfg.d_model as usize; // 2048
        let perm = window_major_perm(16, 16, s);

        // pixels: golden f32 [256,3,14,14] (ROW-major patches, (c,h,w)
        // elements) → window-major rows → bf16.
        let pixels_rm = load_npy_f32(&format!("{GOLDEN}/pixel_values.npy"));
        assert_eq!(pixels_rm.len(), total_l * feat);
        let pixels_f32 = permute_rows_f32(&pixels_rm, &perm, feat);
        let pixels_bf16: Vec<u16> = pixels_f32
            .iter()
            .map(|&x| bf16::from_f32(x).to_bits())
            .collect();
        let pixels = device.alloc_gpu_tensor_from_host(
            &[total_l, feat],
            DType::BF16,
            bf16_slice_as_bytes(&pixels_bf16),
        );

        // cos/sin (bf16, interleaved layout) — ctx completeness; the metal
        // kernel consumes the raw f32 `freqs`.
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

        // raw f32 rope freqs (interleaved x/y per pair, window-major token
        // order). Validate against the golden freqs_cis (ROW-major →
        // permute): cos/sin of our angles must match real/imag.
        let freqs_host = cfg.build_rope_freqs_f32(&grid_thw, total_l);
        assert_eq!(freqs_host.len(), total_l * half_rot);
        {
            let gr = permute_rows_f32(
                &load_npy_f32(&format!("{GOLDEN}/rope_block0_freqs_cis_real.npy")),
                &perm,
                half_rot,
            );
            let gi = permute_rows_f32(
                &load_npy_f32(&format!("{GOLDEN}/rope_block0_freqs_cis_imag.npy")),
                &perm,
                half_rot,
            );
            let max_err = freqs_host
                .iter()
                .zip(gr.iter().zip(&gi))
                .map(|(&a, (&cr, &ci))| (a.cos() - cr).abs().max((a.sin() - ci).abs()))
                .fold(0f32, f32::max);
            eprintln!("  [rope freqs] max |cos/sin - golden| = {max_err:.2e}");
            assert!(max_err < 1e-4, "interleaved rope freqs layout mismatch");
        }
        let freqs = device.alloc_gpu_tensor_from_host(
            &[total_l, half_rot],
            DType::F32,
            f32_slice_as_bytes(&freqs_host),
        );

        // pos_embeds: HOST-side bicubic interp from the learned 64×64
        // table — the exact production path — validated vs the golden
        // (row-major → permute) before upload.
        let embed_dim = cfg.embed_dim as usize;
        let pe_table = gw
            .tensor_to_f32("vision_tower.patch_embed.pos_emb.weight")
            .expect("pos_emb table");
        let num_grid = (((pe_table.len() / embed_dim) as f64).sqrt()).round() as usize;
        assert_eq!(num_grid, 64, "MoonViT learned pos-emb grid is 64×64");
        let mut pos_f32 =
            cfg.bicubic_pos_embed_interpolate(&grid_thw, num_grid, &pe_table, total_l);
        assert_eq!(pos_f32.len(), total_l * embed_dim, "pos_embeds shape");
        let pos_golden = permute_rows_f32(
            &load_npy_f32(&format!("{GOLDEN}/pos_embeds.npy")),
            &perm,
            embed_dim,
        );
        let interp_cos = cosine(&pos_f32, &pos_golden);
        eprintln!("  [pos_embed interp] cos(host bicubic, golden) = {interp_cos:.6}");
        assert!(
            interp_cos > 0.999,
            "host bicubic_pos_embed_interpolate cos {interp_cos} != golden — bicubic/merge-order bug"
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

        let kv = KvCachePool::empty_for_vision();

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

        // ── Readback + compare. Default gate = `projector_out` [64, 2048]
        // (merged-token order — shared with mlx, no permutation). Probes
        // with 256 rows are patch-level → permute the golden. ──
        let probe = "projector_out";
        let mut golden = load_npy_f32(&format!("{GOLDEN}/{probe}.npy"));
        let n_real = golden.len();
        let (rows, cols) = if n_real % total_l == 0 && n_real / total_l >= embed_dim {
            // patch-level probe [256, cols] — permute to window-major
            let cols = n_real / total_l;
            golden = permute_rows_f32(&golden, &perm, cols);
            (total_l, cols)
        } else {
            (out_rows, out_cols)
        };
        let t = projected.as_gpu_tensor();
        let raw = t.raw_ptr() as *const u16;
        let out: Vec<f32> = (0..n_real)
            .map(|i| bf16::from_bits(unsafe { *raw.add(i) }).to_f32())
            .collect();

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

        eprintln!("──────────── LOCATEANYTHING METAL VISION GREEN GATE ────────────");
        eprintln!("  cosine({probe}) = {cos_sim:.6}");
        eprintln!("  nan/inf = {nan}   zero_rows = {zero_rows}/{out_rows}");
        eprintln!("  mean = {mean:.5}   absmax = {absmax:.4}");
        eprintln!("  out [0..6]  = {:?}", &out[0..6]);
        eprintln!("  gold[0..6]  = {:?}", &golden[0..6]);
        eprintln!("──────────────────────────────────────────────────────────────");

        assert_eq!(nan, 0, "vision output has NaN/Inf values");
        assert_eq!(
            zero_rows, 0,
            "vision output has all-zero rows (a stage produced nothing)"
        );
        assert!(
            absmax > 1e-3 && absmax < 1e4,
            "vision output magnitude {absmax} out of sane range"
        );
        // EXACT gate vs `projector_out` (bf16 accumulation over 27 blocks
        // + projector).
        let threshold = 0.99;
        assert!(
            cos_sim > threshold,
            "cosine {cos_sim} too low (threshold {threshold}) — the metal MoonViT should \
             match projector_out to bf16 rounding (~1.0). A drop is a real regression in \
             the tower, the interleaved rope, the bicubic pos-emb, or the packing order"
        );
    }
}

# P-1 ORACLE — Qwen3.5-VL-9B vision-tower parity reference

**Status: RESOLVED.** The numeric oracle is **mlx-vlm**, which runs the real
Qwen3.5-VL-9B vision tower natively on this Mac (no torch). `dump_golden.py`
loads it and captures per-stage tensors as golden fixtures for the
scratchy-target-metal vision port.

## Oracle
- Repo: `~/git/mlx-vlm` HEAD `549a75a` (installed editable into `~/.venv` via `uv pip install -e .`, `mlx_vlm==0.6.0`).
- Model: `mlx-community/Qwen3.5-9B-MLX-4bit` (cached, 5.6 GB). Vision tower is **dense/unquantized** (`vision_tower.*`).
- Source of truth: `mlx_vlm/models/qwen3_vl/vision.py` (the 439-line ViT, re-exported by `qwen3_5/vision.py`); `qwen3_5/qwen3_5.py` (merge + mrope); `qwen3_5/config.py`.
- torch / transformers-models / mlx-vlm are NOT otherwise importable — mlx-vlm is the ONLY runnable vision reference on this box. (mlx-**lm** is text-only; every `*_vl.py` there pops `vision_tower`.)

## Harness
`~/.venv/bin/python vllm-rs/tools/vision_parity/dump_golden.py` → `golden/*.npy` (gitignored, regenerable). Deterministic input = a red circle on white, 224×224 → processor → 256 patches.

## Pinned dims / layouts (verified from the run)
- ViT: depth **27**, hidden **1152**, heads **16** → **head_dim 72**, half_rot **36**, intermediate 4304, patch 16, spatial_merge 2, out_hidden **4096**, `gelu_pytorch_tanh`, pos rows 2304, **no deepstack, no window**.
- Norms = **LayerNorm-with-bias**, eps **1e-6**.
- Stage shapes (red_circle, 256 patches): `pixel_values (256,1536)` → `patch_embed (256,1152)` → `+pos_embed (256,1152)` → 27×block `(256,1152)` → merger **`(64,4096)`** (256 → 2×2 merge → 64 tokens × text-hidden 4096).
- **Rope (`apply_rotary_pos_emb_vision`, NeoX):** input `freqs[seq,36]`; `cos=cos(freqs)`, `sin=sin(freqs)`, each **tiled by concat** to 72 (`[f0..f35, f0..f35]`); `rotate_half(x)=concat(-x[36:72], x[0:36])`; `out[i] = x[i]*cos(freqs[i%36]) + rotate_half(x)[i]*sin(freqs[i%36])`. Applied to q AND k (MHA, every head). Golden: `rope_block0_{q,k}_{in,freqs,out}` with q/k shaped `(1,1,256,16,72)`.
- **PosEmbed = `fast_pos_embed_interpolate`** (4-corner bilinear over a 48×48 learned grid) → compute host-side + add (no kernel). Golden: `pos_embeds`.
- **Attention** = per-segment SDPA over `cu_seqlens` (bidirectional, no causal mask). For a single image `cu_seqlens=[0,256]` (one segment) — **a ≥2-image fixture is still needed to golden-test cross-segment isolation (P2).**
- **mrope_section = [11,11,10]**; merge = `masked_scatter` at `input_ids == image_token_index` (from `qwen3_5.py`; not yet dumped — add for P4).

## Golden fixtures produced
`pixel_values, grid_thw, post_patch_embed, pos_embeds, rot_pos_emb_table (256,36), post_block_{0,13,26}, merger_out, vision_output, rope_block0_{q,k}_{in,freqs,out}`.

## TODO for fuller coverage
- 2-image fixture (cu_seqlens with ≥2 segments) for the P2 attention leakage test.
- Dump `cu_seqlens`, the per-block attention output, the mrope `position_ids` (`get_rope_index`), and the post-`masked_scatter` text embeds for P4.

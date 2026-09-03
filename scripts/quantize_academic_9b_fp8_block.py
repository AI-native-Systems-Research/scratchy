#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Quantize ByteDance-Seed/academic-ds-9B to FP8 block-128x128 (DeepSeek V3 / Kimi K2 layout).

Output format matches what `DeepSeekV2Fp8BlockMoELayer::load` expects:
  - FP8 E4M3 weights for every expert + shared expert + dense Linear
  - `weight_scale_inv` block scales stored as F32, shape [N/128, K/128]
  - Gate router stays in BF16 (matches V3/K2 official checkpoints)

Usage:
    /home/moosevan/vllm/.venv/bin/python scripts/quantize_academic_9b_fp8_block.py
"""
import json
import os
from pathlib import Path

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "ByteDance-Seed/academic-ds-9B"
OUT_DIR = Path("/tmp/academic-ds-9b-fp8-block")

# FP8 block-128x128. Match DeepSeek V3 / Kimi K2 official:
#   - Quantize all Linear except `lm_head` and the MoE `gate` (router stays BF16)
#   - Stored as `weight` (FP8 E4M3) + `weight_scale_inv` ([N/128, K/128] f32)
recipe = QuantizationModifier(
    targets=["Linear"],
    scheme="FP8_BLOCK",
    ignore=["lm_head", "re:.*\\.gate$", "re:.*\\.mlp\\.gate$", "re:.*moe\\.gate$"],
)

print(f"Loading {MODEL_ID} (BF16) ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="cuda:0",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print(f"Quantizing → {OUT_DIR}")
oneshot(
    model=model,
    recipe=recipe,
    output_dir=str(OUT_DIR),
)
tokenizer.save_pretrained(str(OUT_DIR))

# Post-process: broaden compressed-tensors `targets` so Python vLLM's
# `find_matched_target` can resolve V3's fused `fused_qkv_a_proj` layer.
# llmcompressor emits `targets=["Linear"]`, which is the right hint for
# its own walk over `nn.Linear` modules at quant time, but Python vLLM
# uses the same field at load time to look up the quant scheme for the
# fused module name (constructed at runtime from `q_a_proj` +
# `kv_a_proj_with_mqa`). Plain `"Linear"` only substring-matches the
# module's `__class__.__name__` (`DeepSeekV2FusedQkvAProj`), which
# misses. Adding `re:.*` (still gated by `ignore`) catches it.
cfg_path = OUT_DIR / "config.json"
cfg = json.loads(cfg_path.read_text())
qc = cfg.get("quantization_config")
if qc:
    for group in qc.get("config_groups", {}).values():
        targets = group.setdefault("targets", [])
        if "re:.*" not in targets:
            targets.append("re:.*")
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"Patched {cfg_path}: targets now {targets}")

print(f"DONE. Output dir size:")
os.system(f"du -sh {OUT_DIR}")

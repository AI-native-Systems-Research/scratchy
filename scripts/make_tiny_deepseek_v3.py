#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project
"""
Create a tiny synthetic DeepSeek-V3-style model for E2E testing.

Architecture:
  - DeepseekV3ForCausalLM: MLA attention (q_lora_rank enabled) + MoE
  - 4 layers: layer 0 dense SwiGLU, layers 1-3 MoE (8 routed + 1 shared)
  - Sigmoid routing with e_score_correction_bias (topk_method=noaux_tc)
  - BF16 weights, vocab_size=102400 (matches DeepSeek-V2-Lite tokenizer)

Run on a pod (or locally with transformers + safetensors installed):
    cd /root/vllm/vllm-rs
    python3 scripts/make_tiny_deepseek_v3.py [--out /tmp/deepseek-v3-tiny]

Output: a self-contained HuggingFace-compatible directory with:
  config.json, tokenizer files (from deepseek-ai/DeepSeek-V2-Lite), model.safetensors
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Model dimensions
# ---------------------------------------------------------------------------
HIDDEN          = 512
INTERMEDIATE    = 1024       # dense MLP only (layer 0)
MOE_INTER       = 128        # per expert intermediate size
# Attention dimensions match DeepSeek-V2-Lite MLA shape so Python vLLM's
# TritonMLA backend (the only MLA backend on SM89 / Ada) accepts the model.
# TritonMLA requires head_size = KV_LORA_RANK + QK_ROPE = 576, matching V2-Lite.
# FlashInfer MLA requires QK_NOPE = 128.
NUM_HEADS       = 4          # 4 * V_HEAD(128) = 512 = HIDDEN
NUM_LAYERS      = 4
VOCAB_SIZE      = 102400
QK_NOPE        = 128         # FlashInfer MLA requirement
QK_ROPE        = 64          # matches V2-Lite ratio
QK_HEAD        = QK_NOPE + QK_ROPE          # 192
V_HEAD         = 128         # NUM_HEADS * V_HEAD = 4*128 = 512 = HIDDEN
Q_LORA_RANK    = 64
KV_LORA_RANK   = 512         # gives head_size=576 which TritonMLA supports
KV_A_PROJ_OUT  = KV_LORA_RANK + QK_ROPE    # 576
KV_LORA_OUT    = NUM_HEADS * (QK_NOPE + V_HEAD)   # 4 * 256 = 1024
Q_PROJ_OUT     = NUM_HEADS * QK_HEAD        # 4 * 192 = 768
N_ROUTED       = 8
N_SHARED       = 1
TOP_K          = 2
FIRST_DENSE    = 1   # layers < 1 are dense


def randbf16(*shape):
    """Small-magnitude BF16 tensor (matches HF initializer_range≈0.02)."""
    return (torch.randn(*shape) * 0.02).to(torch.bfloat16)


def make_weights():
    sd = {}

    # Embeddings + final norm + lm_head
    sd["model.embed_tokens.weight"]  = randbf16(VOCAB_SIZE, HIDDEN)
    sd["model.norm.weight"]          = torch.ones(HIDDEN, dtype=torch.bfloat16)
    sd["lm_head.weight"]             = randbf16(VOCAB_SIZE, HIDDEN)

    for layer in range(NUM_LAYERS):
        pfx = f"model.layers.{layer}"

        # Norms
        sd[f"{pfx}.input_layernorm.weight"]           = torch.ones(HIDDEN, dtype=torch.bfloat16)
        sd[f"{pfx}.post_attention_layernorm.weight"]  = torch.ones(HIDDEN, dtype=torch.bfloat16)

        # ── MLA attention ────────────────────────────────────────────────────
        # q path: q_a_proj → q_a_layernorm → q_b_proj
        sd[f"{pfx}.self_attn.q_a_proj.weight"]        = randbf16(Q_LORA_RANK, HIDDEN)
        sd[f"{pfx}.self_attn.q_a_layernorm.weight"]   = torch.ones(Q_LORA_RANK, dtype=torch.bfloat16)
        sd[f"{pfx}.self_attn.q_b_proj.weight"]        = randbf16(Q_PROJ_OUT, Q_LORA_RANK)

        # kv path
        sd[f"{pfx}.self_attn.kv_a_proj_with_mqa.weight"] = randbf16(KV_A_PROJ_OUT, HIDDEN)
        sd[f"{pfx}.self_attn.kv_a_layernorm.weight"]     = torch.ones(KV_LORA_RANK, dtype=torch.bfloat16)
        sd[f"{pfx}.self_attn.kv_b_proj.weight"]          = randbf16(KV_LORA_OUT, KV_LORA_RANK)

        sd[f"{pfx}.self_attn.o_proj.weight"]          = randbf16(HIDDEN, NUM_HEADS * V_HEAD)

        # ── MLP / MoE ────────────────────────────────────────────────────────
        if layer < FIRST_DENSE:
            # Dense SwiGLU
            sd[f"{pfx}.mlp.gate_proj.weight"] = randbf16(INTERMEDIATE, HIDDEN)
            sd[f"{pfx}.mlp.up_proj.weight"]   = randbf16(INTERMEDIATE, HIDDEN)
            sd[f"{pfx}.mlp.down_proj.weight"] = randbf16(HIDDEN, INTERMEDIATE)
        else:
            # MoE gate (router) + e_score_correction_bias (noaux_tc)
            sd[f"{pfx}.mlp.gate.weight"]                    = randbf16(N_ROUTED, HIDDEN)
            sd[f"{pfx}.mlp.gate.e_score_correction_bias"]   = torch.zeros(N_ROUTED, dtype=torch.float32)

            # Routed experts
            for e in range(N_ROUTED):
                sd[f"{pfx}.mlp.experts.{e}.gate_proj.weight"] = randbf16(MOE_INTER, HIDDEN)
                sd[f"{pfx}.mlp.experts.{e}.up_proj.weight"]   = randbf16(MOE_INTER, HIDDEN)
                sd[f"{pfx}.mlp.experts.{e}.down_proj.weight"] = randbf16(HIDDEN, MOE_INTER)

            # Shared expert (N_SHARED merged into one Linear of size N_SHARED*MOE_INTER)
            shared_inter = N_SHARED * MOE_INTER
            sd[f"{pfx}.mlp.shared_experts.gate_proj.weight"] = randbf16(shared_inter, HIDDEN)
            sd[f"{pfx}.mlp.shared_experts.up_proj.weight"]   = randbf16(shared_inter, HIDDEN)
            sd[f"{pfx}.mlp.shared_experts.down_proj.weight"] = randbf16(HIDDEN, shared_inter)

    return sd


def make_config():
    return {
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v3",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "moe_intermediate_size": MOE_INTER,
        "num_attention_heads": NUM_HEADS,
        "num_hidden_layers": NUM_LAYERS,
        "num_key_value_heads": NUM_HEADS,
        "vocab_size": VOCAB_SIZE,
        "rms_norm_eps": 1e-6,
        "rope_theta": 50000.0,
        "rope_scaling": {
            "beta_fast": 1.0,
            "beta_slow": 1.0,
            "factor": 8.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        },
        "tie_word_embeddings": False,
        "head_dim": QK_HEAD,
        "qk_nope_head_dim": QK_NOPE,
        "qk_rope_head_dim": QK_ROPE,
        "v_head_dim": V_HEAD,
        "q_lora_rank": Q_LORA_RANK,
        "kv_lora_rank": KV_LORA_RANK,
        "n_routed_experts": N_ROUTED,
        "n_shared_experts": N_SHARED,
        "num_experts_per_tok": TOP_K,
        "first_k_dense_replace": FIRST_DENSE,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "torch_dtype": "bfloat16",
        "max_position_embeddings": 4096,
        # Point to V2-Lite tokenizer so vLLM can load it.
        # The tokenizer should already be cached on the pod from the
        # deepseek_v2_lite golden run.
        "tokenizer_name_or_path": "deepseek-ai/DeepSeek-V2-Lite",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/deepseek-v3-tiny",
                        help="Output directory for the tiny model checkpoint")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Building weights...")
    sd = make_weights()
    total_params = sum(t.numel() for t in sd.values())
    total_mb = sum(t.numel() * t.element_size() for t in sd.values()) / 1e6
    print(f"  {total_params:,} params, {total_mb:.1f} MB")

    print(f"Saving to {out}/model.safetensors ...")
    # safetensors requires contiguous float tensors; e_score_correction_bias is f32
    save_file({k: v.contiguous() for k, v in sd.items()}, out / "model.safetensors")

    print("Writing config.json ...")
    with open(out / "config.json", "w") as f:
        json.dump(make_config(), f, indent=2)

    # Copy tokenizer files from the cached V2-Lite model so the output
    # dir is fully self-contained (no HF download needed at inference time).
    try:
        from huggingface_hub import snapshot_download
        import shutil
        tok_dir = snapshot_download(
            "deepseek-ai/DeepSeek-V2-Lite",
            ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf"],
        )
        # Only copy tokenizer files — never copy safetensors shards, index
        # files, or modeling code that would conflict with our checkpoint.
        TOKENIZER_SUFFIXES = (
            "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "vocab.txt", "merges.txt",
        )
        for fname in os.listdir(tok_dir):
            if not any(fname == s or fname.startswith("tokenizer") for s in TOKENIZER_SUFFIXES):
                continue
            src = os.path.join(tok_dir, fname)
            dst = out / fname
            if os.path.isfile(src) and not dst.exists():
                shutil.copy2(src, dst)
                print(f"  Copied {fname}")
        print(f"Tokenizer files copied from {tok_dir}")
        # Remove tokenizer_name_or_path so vLLM doesn't try to fetch it
        cfg = json.loads((out / "config.json").read_text())
        cfg.pop("tokenizer_name_or_path", None)
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
    except Exception as e:
        print(f"  Tokenizer copy skipped ({e}); vLLM will fetch deepseek-ai/DeepSeek-V2-Lite")

    print(f"Done — tiny DeepSeek-V3 model at {out}")
    print("  Next: python3 scripts/generate_golden_refs.py deepseek_v3_tiny")


if __name__ == "__main__":
    main()

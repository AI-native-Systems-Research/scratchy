#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project
"""
Create a tiny synthetic Kimi-K2-style model for E2E testing.

Architecture:
  - DeepseekV3ForCausalLM (Kimi K2 declares this); model_type=kimi_k2
  - 4 layers: layer 0 dense SwiGLU, layers 1-3 MoE (8 routed + 1 shared)
  - **Flat** sigmoid+noaux_tc routing (n_group=1, topk_group=1) — Kimi K2's
    distinguishing routing config vs DeepSeek V3 (which uses n_group=8).
  - first_k_dense_replace=1 (vs V3's 3); routed_scaling_factor=2.827.
  - BF16 weights, vocab_size=102400 (matches DeepSeek-V2-Lite tokenizer).

Run on a pod (or locally with transformers + safetensors installed):
    cd /root/vllm/vllm-rs
    python3 scripts/make_tiny_kimi_k2.py [--out /tmp/kimi-k2-tiny]

Output: a self-contained HF-compatible directory.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Model dimensions — match deepseek_v3 tiny (so MLA shape matches TritonMLA).
# ---------------------------------------------------------------------------
HIDDEN          = 512
INTERMEDIATE    = 1024       # dense MLP only (layer 0)
MOE_INTER       = 128
NUM_HEADS       = 4
NUM_LAYERS      = 4
VOCAB_SIZE      = 102400
QK_NOPE         = 128
QK_ROPE         = 64
QK_HEAD         = QK_NOPE + QK_ROPE          # 192
V_HEAD          = 128
Q_LORA_RANK     = 64
KV_LORA_RANK    = 512
KV_A_PROJ_OUT   = KV_LORA_RANK + QK_ROPE     # 576
KV_LORA_OUT     = NUM_HEADS * (QK_NOPE + V_HEAD)   # 1024
Q_PROJ_OUT      = NUM_HEADS * QK_HEAD        # 768
N_ROUTED        = 8
N_SHARED        = 1
TOP_K           = 2
FIRST_DENSE     = 1   # K2: only layer 0 is dense (vs V3's 3)
ROUTED_SCALE    = 2.827  # K2's distinguishing scale (vs V3's 2.5)


def randbf16(*shape):
    return (torch.randn(*shape) * 0.02).to(torch.bfloat16)


# Seed once at import time so re-invoking the script produces identical weights.
# Without this the Python golden and scratchy test loaded different random
# checkpoints across runs even though they pointed at the same /tmp path.
torch.manual_seed(0)


def make_weights():
    sd = {}

    sd["model.embed_tokens.weight"] = randbf16(VOCAB_SIZE, HIDDEN)
    sd["model.norm.weight"] = torch.ones(HIDDEN, dtype=torch.bfloat16)
    sd["lm_head.weight"] = randbf16(VOCAB_SIZE, HIDDEN)

    for layer in range(NUM_LAYERS):
        pfx = f"model.layers.{layer}"

        sd[f"{pfx}.input_layernorm.weight"] = torch.ones(HIDDEN, dtype=torch.bfloat16)
        sd[f"{pfx}.post_attention_layernorm.weight"] = torch.ones(HIDDEN, dtype=torch.bfloat16)

        # MLA attention (q-LoRA path).
        sd[f"{pfx}.self_attn.q_a_proj.weight"] = randbf16(Q_LORA_RANK, HIDDEN)
        sd[f"{pfx}.self_attn.q_a_layernorm.weight"] = torch.ones(Q_LORA_RANK, dtype=torch.bfloat16)
        sd[f"{pfx}.self_attn.q_b_proj.weight"] = randbf16(Q_PROJ_OUT, Q_LORA_RANK)
        sd[f"{pfx}.self_attn.kv_a_proj_with_mqa.weight"] = randbf16(KV_A_PROJ_OUT, HIDDEN)
        sd[f"{pfx}.self_attn.kv_a_layernorm.weight"] = torch.ones(KV_LORA_RANK, dtype=torch.bfloat16)
        sd[f"{pfx}.self_attn.kv_b_proj.weight"] = randbf16(KV_LORA_OUT, KV_LORA_RANK)
        sd[f"{pfx}.self_attn.o_proj.weight"] = randbf16(HIDDEN, NUM_HEADS * V_HEAD)

        if layer < FIRST_DENSE:
            sd[f"{pfx}.mlp.gate_proj.weight"] = randbf16(INTERMEDIATE, HIDDEN)
            sd[f"{pfx}.mlp.up_proj.weight"] = randbf16(INTERMEDIATE, HIDDEN)
            sd[f"{pfx}.mlp.down_proj.weight"] = randbf16(HIDDEN, INTERMEDIATE)
        else:
            sd[f"{pfx}.mlp.gate.weight"] = randbf16(N_ROUTED, HIDDEN)
            sd[f"{pfx}.mlp.gate.e_score_correction_bias"] = torch.zeros(N_ROUTED, dtype=torch.float32)

            for e in range(N_ROUTED):
                sd[f"{pfx}.mlp.experts.{e}.gate_proj.weight"] = randbf16(MOE_INTER, HIDDEN)
                sd[f"{pfx}.mlp.experts.{e}.up_proj.weight"] = randbf16(MOE_INTER, HIDDEN)
                sd[f"{pfx}.mlp.experts.{e}.down_proj.weight"] = randbf16(HIDDEN, MOE_INTER)

            shared_inter = N_SHARED * MOE_INTER
            sd[f"{pfx}.mlp.shared_experts.gate_proj.weight"] = randbf16(shared_inter, HIDDEN)
            sd[f"{pfx}.mlp.shared_experts.up_proj.weight"] = randbf16(shared_inter, HIDDEN)
            sd[f"{pfx}.mlp.shared_experts.down_proj.weight"] = randbf16(HIDDEN, shared_inter)

    return sd


def make_config():
    return {
        "architectures": ["DeepseekV3ForCausalLM"],
        # Transformers does not yet ship a `kimi_k2` model_type entry, but the
        # K2 prod checkpoint declares architectures=["DeepseekV3ForCausalLM"]
        # — same code path. Pin model_type to "deepseek_v3" so Python vLLM
        # loads the synthetic tiny without --trust-remote-code.
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
            # factor=4.0 picks a HfFingerprint distinct from
            # deepseek-v3-tiny.json (factor=8.0) so scratchy's runtime
            # variant sniff loads the K2 arm, not the V3-tiny arm.
            "beta_fast": 1.0,
            "beta_slow": 1.0,
            "factor": 4.0,
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
        "routed_scaling_factor": ROUTED_SCALE,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        # Kimi K2: flat top-k (no group selection).
        "n_group": 1,
        "topk_group": 1,
        "torch_dtype": "bfloat16",
        "max_position_embeddings": 4096,
        "tokenizer_name_or_path": "deepseek-ai/DeepSeek-V2-Lite",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/kimi-k2-tiny")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Building weights...")
    sd = make_weights()
    total_params = sum(t.numel() for t in sd.values())
    total_mb = sum(t.numel() * t.element_size() for t in sd.values()) / 1e6
    print(f"  {total_params:,} params, {total_mb:.1f} MB")

    print(f"Saving to {out}/model.safetensors ...")
    save_file({k: v.contiguous() for k, v in sd.items()}, out / "model.safetensors")

    print("Writing config.json ...")
    with open(out / "config.json", "w") as f:
        json.dump(make_config(), f, indent=2)

    try:
        from huggingface_hub import snapshot_download
        import shutil
        tok_dir = snapshot_download(
            "deepseek-ai/DeepSeek-V2-Lite",
            ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf"],
        )
        for fname in os.listdir(tok_dir):
            if not (fname.startswith("tokenizer") or fname == "special_tokens_map.json"):
                continue
            src = os.path.join(tok_dir, fname)
            dst = out / fname
            if os.path.isfile(src) and not dst.exists():
                shutil.copy2(src, dst)
                print(f"  Copied {fname}")
        cfg = json.loads((out / "config.json").read_text())
        cfg.pop("tokenizer_name_or_path", None)
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
    except Exception as e:
        print(f"  Tokenizer copy skipped ({e}); vLLM will fetch deepseek-ai/DeepSeek-V2-Lite")

    print(f"Done — tiny Kimi K2 model at {out}")
    print("  Next: python3 scripts/generate_golden_refs.py kimi_k2_tiny")


if __name__ == "__main__":
    main()

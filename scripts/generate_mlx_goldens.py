#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

"""Generate golden reference token streams for scratchy-metal int4
parity testing using `mlx_lm.generate` (greedy decode at temp=0).

Run from the vllm-rs root on Apple Silicon with `mlx-lm` installed
in `~/.venv`:

    source ~/.venv/bin/activate
    cd /path/to/vllm-rs
    python3 scripts/generate_mlx_goldens.py

Output: `crates/e2e/testdata/golden/<model_key>_mlx_4bit.json`
matching the `GoldenReference` schema in `e2e/src/assertions.rs`
(model, max_tokens, num_logprobs, results: [{prompt, output_tokens,
output_text, logprobs}]).

Unlike `generate_golden_refs.py` (Python vLLM on GPU, used for the
CUDA path), this script captures from MLX — the reference for the
scratchy-metal int4 backend — so the goldens reflect the same FP
arithmetic scratchy is targeting parity against (P10 of
INT4_PARITY_PLAN.md).
"""

import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


# Match testdata/prompts.txt + generate_golden_refs.py PROMPTS exactly.
PROMPTS = [
    "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.",
    "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020.",
    "Compare and contrast artificial intelligence with human intelligence in terms of processing information.",
    "Describe the basic components of a neural network and how it can be trained.",
    "Write a short story about a robot that dreams for the first time.",
    "Analyze the impact of the COVID-19 pandemic on global economic structures and future business models.",
    "Explain the cultural significance of the Mona Lisa painting, and how its perception might vary in Western versus Eastern societies.",
    "Translate the following English sentence into Japanese, French, and Swahili: 'The early bird catches the worm.'",
]

# MLX-only mlx-community 4bit checkpoints. Keys mirror the
# `llama_3_2_1b_*` family in `crates/e2e/testdata/golden/`.
# Stick to 1B for now to fit the 24Gi dev machine; 3B added when
# we validate that path separately (see P10).
MODELS = {
    "llama_3_2_1b_mlx_4bit": "mlx-community/Llama-3.2-1B-Instruct-4bit",
}

MAX_TOKENS = 64
# Number of top tokens to capture per position. Existing CUDA goldens
# use 20; match for downstream `check_logprobs_close` compatibility.
NUM_LOGPROBS = 20

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "crates" / "e2e" / "testdata" / "golden"


def generate_golden_for_model(repo: str, model_key: str) -> dict:
    """Generate one model's full golden across all prompts.

    Greedy decode (temp=0 argmax) for MAX_TOKENS positions per
    prompt. Captures per-step token id + decoded text + top-N
    logprobs (decoded via the tokenizer for downstream
    string-keyed comparison).
    """
    print(f"\n=== {model_key} ({repo}) ===")
    model, tokenizer = load(repo)
    # mlx_lm's `stream_generate` accepts a `sampler` callable on the
    # underlying `generate_step`. Temp=0 → argmax.
    sampler = make_sampler(temp=0.0)

    results = []
    for prompt_idx, prompt in enumerate(PROMPTS):
        print(f"  [{prompt_idx}] {prompt[:60]}...", flush=True)

        # Token ids in the order MLX emits them, plus the decoded
        # text segment per step (BPE merges may make this an empty
        # string for some positions, which the existing
        # `extract_engine_output` already handles).
        output_token_ids: list[int] = []
        output_segments: list[str] = []
        logprobs_per_step: list[dict[str, float]] = []

        for resp in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=MAX_TOKENS,
            sampler=sampler,
        ):
            output_token_ids.append(int(resp.token))
            output_segments.append(resp.text)

            # `logprobs` is the full vocab vector for this step.
            # argpartition gives unsorted top-N; sort that slice
            # descending and decode via the tokenizer.
            lp_vec = resp.logprobs
            k = min(NUM_LOGPROBS, lp_vec.size)
            top_idx_unsorted = mx.argpartition(-lp_vec, k - 1)[:k]
            # Sort the top-k slice descending by logprob.
            top_idx_unsorted = top_idx_unsorted[
                mx.argsort(-lp_vec[top_idx_unsorted])
            ]
            top_ids = [int(t) for t in top_idx_unsorted]
            top_lps = [float(lp_vec[t]) for t in top_idx_unsorted]
            # Decode each candidate id to its token string.
            top_map: dict[str, float] = {}
            for tid, lp in zip(top_ids, top_lps):
                # `decode([tid])` gives the rendered text for the
                # token. Most HF tokenizers strip BPE prefixes; the
                # existing CUDA goldens use whatever the tokenizer
                # returns, which works because the engine side uses
                # the same tokenizer.
                tok_str = tokenizer.decode([tid])
                # In the very rare collision case (two ids mapping
                # to the same surface string), keep the higher
                # logprob to mirror Python vLLM's behavior.
                if tok_str not in top_map or lp > top_map[tok_str]:
                    top_map[tok_str] = lp
            logprobs_per_step.append(top_map)

        # Per-step decoded token string. The completions API on the
        # engine side returns these as `Vec<String>` so the golden
        # comparison is token-string-level (matches the CUDA
        # goldens' shape).
        per_step_strings = [tokenizer.decode([t]) for t in output_token_ids]
        full_text = "".join(output_segments)

        results.append({
            "prompt": prompt,
            "output_tokens": per_step_strings,
            "output_text": full_text,
            "logprobs": logprobs_per_step,
        })

    return {
        "model": repo,
        "max_tokens": MAX_TOKENS,
        "num_logprobs": NUM_LOGPROBS,
        "results": results,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for model_key, repo in MODELS.items():
        golden = generate_golden_for_model(repo, model_key)
        out_path = OUTPUT_DIR / f"{model_key}.json"
        out_path.write_text(json.dumps(golden, indent=2, ensure_ascii=False))
        print(f"  → wrote {out_path}")


if __name__ == "__main__":
    main()

"""A minimal valid board, for the tests to break one rule at a time.

Deliberately small: each test mutates one field and asserts the one problem that
comes back, so a failure names the rule that broke rather than a diff.
"""

import copy
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent
if str(SITE) not in sys.path:
    sys.path.insert(0, str(SITE))

PROV = {
    "machine": "h1",
    "rung": "R1",
    "scratchy_sha": "abc1234",
    "claude_version": "2.1.285",
    "ollama_version": "0.34.2",
    "context_length_tokens": 4096,
    "kv_cache_dtype": {"scratchy": "turboquant-3bit", "ollama": "f16"},
    "quantization": {"scratchy": "mlx-affine-b4-g64", "ollama": "Q4_K_M"},
    "checkpoint_revision": {"scratchy": "x@1", "ollama": "y@2"},
    "sampling": {"temperature": 0.0},
    "disclosed": ["kv-dtype-unmatched"],
}


def _stat(m, lo, hi):
    return {"median": m, "p10": lo, "p90": hi}


def lane_a():
    return {
        "schema": 1,
        "run_id": "r-a",
        "lane": "A",
        "status": "published",
        "provenance": copy.deepcopy(PROV),
        "measurements": [
            {"lane": "A", "metric": "ttft_ms", "engine": "scratchy", "model": "m1",
             "n": 12, "stat": _stat(100, 90, 110),
             "series": {"x": "turn_index", "points": [[1, 500], [2, 100], [3, 100]]}},
            {"lane": "A", "metric": "ttft_ms", "engine": "ollama", "model": "m1",
             "n": 12, "stat": _stat(300, 280, 320),
             "series": {"x": "turn_index", "points": [[1, 520], [2, 300], [3, 300]]}},
        ],
        "gaps": [],
        "ablations": [
            {"optimization": "o1", "model": "m1", "metric": "ttft_ms",
             "delta_pct": -40.0, "ci_pct": [-44.0, -36.0], "n": 12,
             "attribution": "optimization", "quality_check": "parity gate passed"},
        ],
    }


def lane_b():
    return {
        "schema": 1,
        "run_id": "r-b",
        "lane": "B",
        "status": "published",
        "provenance": copy.deepcopy(PROV),
        "measurements": [
            {"lane": "B", "metric": "task_pass", "engine": "scratchy", "model": "m1",
             "n": 5, "stat": _stat(1.0, 1.0, 1.0)},
            {"lane": "B", "metric": "task_pass", "engine": "ollama", "model": "m1",
             "n": 5, "stat": _stat(0.8, 0.6, 1.0)},
            {"lane": "B", "metric": "tool_call_parse_failure_rate",
             "engine": "scratchy", "model": "m1", "n": 5, "stat": _stat(0.0, 0.0, 0.0)},
            {"lane": "B", "metric": "tool_call_parse_failure_rate",
             "engine": "ollama", "model": "m1", "n": 5, "stat": _stat(0.0, 0.0, 0.0)},
        ],
        "gaps": [],
        "ablations": [],
    }


def board():
    return {
        "schema": 1,
        "title": "Test board",
        "tagline": "t",
        "epic": 158,
        "method_issue": 161,
        "measurement_issue": 166,
        "repo": "https://example.invalid/repo",
        "unaffiliated": "u",
        "entry_rule": "e",
        "workload": {"client": "claude", "endpoint": "/v1/messages", "why": "w"},
        "correctness_gate": "tool_call_parse_failure_rate",
        "no_composite_score": "n",
        "lanes": {
            "A": {"name": "trace replay", "min_n": 3, "summary": "s",
                  "owns": ["ttft_ms"]},
            "B": {"name": "live Claude Code", "min_n": 5, "summary": "s",
                  "owns": ["task_pass", "tool_call_parse_failure_rate"]},
        },
        "metrics": [
            {"id": "ttft_ms", "label": "TTFT", "unit": "ms",
             "lower_is_better": True, "digits": 1},
            {"id": "task_pass", "label": "task pass", "unit": "",
             "lower_is_better": False, "digits": 2},
            {"id": "tool_call_parse_failure_rate", "label": "tool-call failures",
             "unit": "", "lower_is_better": True, "digits": 3},
        ],
        "engines": [
            {"id": "scratchy", "label": "scratchy", "series": 1, "launch": "a"},
            {"id": "ollama", "label": "ollama", "series": 2, "launch": "b"},
        ],
        "models": [{"id": "m1", "label": "m1", "family": "dense", "note": "n"}],
        "machines": [{"id": "h1", "label": "Host one", "os": "o", "power": "p"}],
        "rungs": [{"id": "R1", "label": "what a user gets", "detail": "d"}],
        "optimizations": [
            {"id": "o1", "label": "prefix caching", "switch": "--no-prefix-caching",
             "expected": "e"},
        ],
        "applicability": [
            {"optimization": "o1", "model": "m1", "state": "live",
             "reason": "caches", "evidence": "init.rs:1"},
        ],
        "expects": {
            "metrics": ["ttft_ms", "task_pass", "tool_call_parse_failure_rate"],
            "engines": ["scratchy", "ollama"],
            "models": ["m1"],
            "machines": ["h1"],
            "rungs": ["R1"],
        },
        "kpis": [
            {"id": "k1", "label": "TTFT", "hero": True, "metric": "ttft_ms",
             "model": "m1", "numerator": "ollama", "denominator": "scratchy",
             "spark": "scratchy"},
        ],
        "disclosures": [
            {"id": "kv-dtype-unmatched", "text": "t", "source": "s"},
        ],
        "status": {"pending": {"issue": 166, "text": "nothing yet"}},
        "runs": [lane_a(), lane_b()],
    }


def indexed(b=None):
    """A board with its cell index attached, as `load()` would return it."""
    import build_leaderboard

    return build_leaderboard.index(b if b is not None else board())

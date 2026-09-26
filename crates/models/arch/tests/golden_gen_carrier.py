#!/usr/bin/env python3
"""UNIVERSAL carrier-as-Python oracle: runs ANY `dsl/<arch>.py` under torch
against synthetic weights and dumps the logits scratchy must reproduce.

One generic mechanism works for every arch because all per-model structure
is DECLARED data the repo already has — there is no per-arch code here:
  - bounds:      `configs/<arch>/<stem>.json` (verbatim HF config), through
                 the same derivation table config.rs applies
  - leaf shapes: `configs/<arch>/weights.json` — every weight's dims are
                 formula strings over those bounds; evaluate them
  - ambient names: the carrier's free variables ARE the manifest's dotted
                 paths (layered leaves × depth, scalars bound from the
                 bounds table / config by NAME), `.bias` siblings scanned
                 from the carrier text, and the identity-layout runtime
                 tensors the qwen3 gate established (kv_cache, rotary,
                 block_table, positions, …)
  - op semantics: the shim library below — one torch function per DSL op
                 (classified.rs `from_name`), each mirroring its kernel's
                 reference math (cpu_golden.rs / shaders/*.metal)

The SAME carrier text is the reference implementation and the spec at
once: this script executes it, scratchy compiles it, and the parity gate
compares logits. `--tiny` shrinks a fixed allowlist of config bounds
(re-running the derivation on the shrunk config) and writes a matching
synthetic checkpoint (model.safetensors + config.json in HF layout,
honoring safetensors_prefix's renames / prefixes / tie / digit-suffix /
packed-split rules) so the Rust gate can load the SAME weights through
the macro-emitted `load`.

Usage:
    python tests/golden_gen_carrier.py --arch qwen3 --stem qwen3-0.6b \
        [--tiny] [--num-tokens 8] [--out goldens/<arch>-<stem>]
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

ARCH_DIR = Path(__file__).parent.parent / "configs"
DSL_DIR = Path(__file__).parent.parent / "dsl"

INIT_SCALE = 0.02  # randn scale — deep stacks stay finite, logits non-degenerate
BLOCK_SIZE = 16    # engine block size (metal worker default)
NUM_BLOCKS = 128   # pool size the Rust gate builds
SEED = 0xC0FFEE

# `--tiny` bound shrink: raw-config keys → target, applied BEFORE the
# derivation pass so every derived bound (head_dim, attn_q_dim, …)
# recomputes consistently. Absent keys are no-ops. head_dim itself is
# NOT shrunk (it derives from hidden/heads when absent, and shrinking
# it independently would break head-count × head_dim == q width).
TINY_BOUNDS = {
    "vocab_size": 512,
    "hidden_size": 128,
    "intermediate_size": 256,
    "moe_intermediate_size": 128,
    "shared_expert_intermediate_size": 128,
    "max_position_embeddings": 64,
    "num_hidden_layers": 2,
    "sliding_window": 8,
    "head_dim": 32,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 16,
    "v_head_dim": 16,
    "kv_lora_rank": 32,
    "q_lora_rank": 32,
    "linear_value_head_dim": 16,
    "linear_key_head_dim": 16,
    # head counts: shrink so hidden stays divisible AND heads ×
    # head_dim == attn width. 4 heads × 32 head_dim = 128 = hidden.
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "num_global_key_value_heads": 1,
    # gemma4's global-attention class: 2× the sliding head_dim, preserving
    # the full-scale class relation (512 vs 256). Shrinking to head_dim
    # itself would COLLAPSE the classes: codegen's rope guard then reads
    # uniform geometry and demands GLOBAL_ROT_DIM == ROT_DIM, but the
    # classes' rope widths still differ (global proportional 0.25×, sliding
    # full). 64 also keeps the page relation real gemma4 has (global page
    # 1×64 = 64 < sliding page 4×32 = 128 → the supported scale-up branch).
    "global_head_dim": 64,
    # expert counts
    "num_local_experts": 8,
    "num_experts": 8,
    "n_routed_experts": 8,
    "n_group": 2,
    "topk_group": 1,
    "num_experts_per_tok": 2,
    "top_k_experts": 2,
    "n_shared_experts": 1,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 2,
    # gemma3-mm hides the text side under text_config — the projector's
    # d_model reads it, so it shrinks too.
    "text_config.hidden_size": 64,
    "vision_config.hidden_size": 64,
    "vision_config.embed_dim": 64,
    "vision_config.intermediate_size": 128,
    "vision_config.image_size": 56,
    "vision_config.depth": 2,
    "vision_config.num_heads": 4,
    "vision_config.patch_size": 14,
    "vision_config.in_chans": 3,
    "vision_config.in_channels": 3,
    "vision_config.temporal_patch_size": 1,
    "vision_config.spatial_merge_size": 2,
    "vision_config.out_hidden_size": 128,
    # gemma3-mm's SigLIP spells depth `num_hidden_layers`
    "vision_config.num_hidden_layers": 2,
    "vision_config.num_attention_heads": 4,
    "vision_config.mlp_ratio": 2,
    # Qwen2.5-VL window attention: 56/merge2/patch14 = win_cells 2 — a
    # REAL window tiling (not whole-grid) at the 16×16-patch grid this
    # arch's goldens use (--num-tokens 256), so the window permutation
    # is non-identity and the oracle transcribes the dispatch.
    "vision_config.window_size": 56,
    # gemma3-mm's pool chain: mm_tokens_per_image (top-level) pairs
    # with vision_num_positions — 16 positions / 4 pooled tokens =
    # pool factor 4 → kernel 2, the same >1 pooling class as the real
    # config (4096/256 = 16 → kernel 4).
    "mm_tokens_per_image": 4,
}


# ───────────────────────── config → bounds (config.rs replica) ─────────────────────────


def json_path(obj, dotted):
    cur = obj
    for seg in dotted.split("."):
        if isinstance(cur, dict):
            if seg not in cur:
                return None
            cur = cur[seg]
        elif isinstance(cur, list):
            try:
                cur = cur[int(seg)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur


def set_json_path(obj, dotted, value):
    cur = obj
    segs = dotted.split(".")
    for seg in segs[:-1]:
        cur = cur.setdefault(seg, {})
    cur[segs[-1]] = value


def load_config(arch, stem):
    cfg = json.loads((ARCH_DIR / arch / f"{stem}.json").read_text())
    # normalize_hf_config: hoist text_config + rope_parameters; explicit
    # top-level wins. vision_config stays nested (vision glue owns it).
    flat = dict(cfg)
    for nested in ("text_config", "rope_parameters"):
        for k, v in (cfg.get(nested) or {}).items():
            flat.setdefault(k, v)
    # arch.json: config_aliases (post-hoist, dotted paths allowed) and
    # bound_defaults / scalar_defaults (declared defaults — explicit
    # config values still win).
    aj_path = ARCH_DIR / arch / "arch.json"
    if aj_path.exists():
        aj = json.loads(aj_path.read_text())
        for std, alt in (aj.get("config_aliases") or {}).items():
            if std in flat:
                continue
            v = json_path(flat, alt)
            if v is not None:
                flat[std] = v
        for k, v in (aj.get("bound_defaults") or {}).items():
            flat.setdefault(k, v)
        for k, v in (aj.get("scalar_defaults") or {}).items():
            flat.setdefault(k, v)
    # Drop nulls: configs carry `"final_logit_softcapping": null` and
    # `"q_lora_rank": null` for "absent" — a null must not shadow a
    # default or a derived value.
    return {k: v for k, v in flat.items() if v is not None}


def derive_bounds(cfg):
    """derive_implicit_bounds + apply_generic_bound_defaults +
    derive_sliding_window_pattern, in config.rs order."""
    b = {k: int(v) for k, v in cfg.items() if isinstance(v, int) and not isinstance(v, bool)}
    has = b.__contains__
    put = b.setdefault

    if not has("head_dim") and has("qk_nope_head_dim") and has("qk_rope_head_dim"):
        put("head_dim", b["qk_nope_head_dim"] + b["qk_rope_head_dim"])
    if not has("head_dim") and has("hidden_size") and has("num_attention_heads"):
        h, n = b["hidden_size"], b["num_attention_heads"]
        if n and h % n == 0:
            put("head_dim", h // n)
    if all(has(k) for k in ("num_attention_heads", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")):
        heads, nope, rope, vhd = (b[k] for k in ("num_attention_heads", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"))
        put("q_proj_out", heads * (nope + rope))
        put("kv_lora_out", heads * (nope + vhd))
        put("attn_out", heads * vhd)
        if has("kv_lora_rank"):
            put("kv_a_proj_out", b["kv_lora_rank"] + rope)
    if not has("num_key_value_heads") and has("num_attention_heads"):
        put("num_key_value_heads", b["num_attention_heads"])
    if not has("sliding_window_global_remainder") and b.get("sliding_window_pattern", 0) > 0:
        put("sliding_window_global_remainder", b["sliding_window_pattern"] - 1)
    if not has("intermediate_size") and has("moe_intermediate_size"):
        put("intermediate_size", b.get("shared_expert_intermediate_size", 0))
    if not has("attn_q_dim") and has("num_attention_heads") and has("head_dim"):
        put("attn_q_dim", b["num_attention_heads"] * b["head_dim"])
    if not has("q_gate_dim") and has("attn_q_dim"):
        put("q_gate_dim", 2 * b["attn_q_dim"] if b.get("attn_output_gate", 0) != 0 else b["attn_q_dim"])
    if not has("kv_dim") and has("num_key_value_heads") and has("head_dim"):
        put("kv_dim", b["num_key_value_heads"] * b["head_dim"])
    if not has("gdn_value_dim") and has("linear_num_value_heads") and has("linear_value_head_dim"):
        put("gdn_value_dim", b["linear_num_value_heads"] * b["linear_value_head_dim"])
    if not has("gdn_conv_dim") and has("linear_num_key_heads") and has("linear_key_head_dim") and has("gdn_value_dim"):
        put("gdn_conv_dim", 2 * b["linear_num_key_heads"] * b["linear_key_head_dim"] + b["gdn_value_dim"])
    # apply_generic_bound_defaults
    if not has("global_head_dim") and has("head_dim"):
        put("global_head_dim", b["head_dim"])
    if not has("num_global_key_value_heads") and has("num_key_value_heads"):
        put("num_global_key_value_heads", b["num_key_value_heads"])
    if not has("q_global_dim") and has("num_attention_heads") and has("global_head_dim"):
        put("q_global_dim", b["num_attention_heads"] * b["global_head_dim"])
    if not has("k_global_dim") and has("num_global_key_value_heads") and has("global_head_dim"):
        put("k_global_dim", b["num_global_key_value_heads"] * b["global_head_dim"])
    # derive_sliding_window_pattern (from verbatim layer_types)
    if not has("sliding_window_pattern"):
        types = cfg.get("layer_types")
        if isinstance(types, list) and types:
            for i, t in enumerate(types):
                if t != types[0]:
                    put("sliding_window_pattern", i + 1)
                    break
    return b


def eval_params(arch, cfg, bounds):
    """Evaluate the arch.json `params` table (config.rs `eval_params`):
    each entry is {name, from} (dotted config path, optional default) or
    {name, expr} (formula over earlier entries) or {name, value}. Order
    matters — exprs read bounds earlier entries defined. This is where
    the vision_* / d_model bounds come from; there is no other source."""
    aj_path = ARCH_DIR / arch / "arch.json"
    if not aj_path.exists():
        return
    aj = json.loads(aj_path.read_text())
    for entry in aj.get("params") or []:
        name = entry["name"]
        if "value" in entry:
            v = entry["value"]
        elif "from" in entry:
            v = json_path(cfg, entry["from"])
            if v is None:
                v = entry.get("default")
            if v is None:
                continue
        else:
            expr = entry["expr"]
            for tok in set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", expr)):
                if tok == "sqrt":
                    continue
                if tok not in bounds:
                    raise KeyError(f"params expr {expr!r}: bound {tok!r} missing")
            v = eval(expr, {"sqrt": math.sqrt}, dict(bounds))
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        bounds[name] = v
        # eval_params also mirrors derived ints into cfg so later scalar
        # binding sees them (config.rs mirrors into `scalars`).
        cfg[name] = v


def eval_formula(expr, bounds):
    if re.fullmatch(r"\d+", expr):
        return int(expr)
    for tok in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", expr):
        if tok not in bounds:
            raise KeyError(f"bound {tok!r} (formula {expr!r}) missing from config")
    return int(eval(expr, {}, dict(bounds)))  # declarative data, not input


# ───────────────────────── synthetic weights ─────────────────────────


def rand_tensor(shape):
    return (torch.randn(*shape) * INIT_SCALE).float()


class Tree:
    """Attribute-style weight tree: `self_attn.q_proj[layer]` works."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class Layered(list):
    """A per-layer list that DISTRIBUTES attribute access over its
    elements — the carrier's two reference forms both work:
    `ns.leaf[layer]` (ns.leaf → Layered of leaves, then index) and
    `ns[layer].leaf` (index → element, then attribute)."""

    def __getattr__(self, name):
        return Layered(getattr(el, name) for el in self)


# Bundle internals for manifest `["<marker>"]` entries. The leaves are the
# runtime layer-struct fields each loader reads (crates/layers/src/), with
# the STORAGE layouts those structs declare. The `moe_block`/`gemma_moe`/
# `gated_delta_net` shims consume them — they never flow through the DSL
# `gemm` checker, so they use each layer struct's own orientation.
BUNDLE_LEAVES = {
    # Mixtral block_sparse_moe → FusedMoELayer::Dense (DenseFusedMoELayer):
    # gate Linear [E, hidden→E? no — [hidden, E] gemm], w1 stacked
    # gate+up [E, 2*inter, hidden], w2 [E, hidden, inter].
    "num_local_experts": [
        ("gate", lambda b: [b["hidden_size"], b["num_local_experts"]]),
        ("w1", lambda b: [b["num_local_experts"], 2 * b["intermediate_size"], b["hidden_size"]]),
        ("w2", lambda b: [b["num_local_experts"], b["hidden_size"], b["intermediate_size"]]),
    ],
    # Qwen-MoE mlp → SharedFusedMoELayer::Dense: moe {gate, w1, w2} +
    # optional shared_gate_up [2*inter, hidden] + shared_down +
    # shared_expert_gate [1, hidden] (Linear layouts; shared omitted
    # when shared_expert_intermediate_size is 0/absent — the layer
    # struct's Option fields). This entry is a callable so the leaf list
    # itself can depend on the bounds.
    "num_experts": lambda b: [
        ("gate", [b["hidden_size"], b["num_experts"]]),
        ("w1", [b["num_experts"], 2 * b["moe_intermediate_size"], b["hidden_size"]]),
        ("w2", [b["num_experts"], b["hidden_size"], b["moe_intermediate_size"]]),
    ] + ([
        ("shared_gate_up", [2 * b["shared_expert_intermediate_size"], b["hidden_size"]]),
        ("shared_down", [b["hidden_size"], b["shared_expert_intermediate_size"]]),
        ("shared_expert_gate", [1, b["hidden_size"]]),
    ] if b.get("shared_expert_intermediate_size", 0) else []),
    # DeepSeek moe → DeepSeekV2MoELayer: FusedMoELayer {gate, w1, w2} +
    # e_score_correction_bias [E] + shared_gate_up [2*inter, hidden] +
    # shared_down [hidden, inter] (Linear layouts).
    "n_routed_experts": [
        ("gate", lambda b: [b["hidden_size"], b["n_routed_experts"]]),
        ("w1", lambda b: [b["n_routed_experts"], 2 * b["moe_intermediate_size"], b["hidden_size"]]),
        ("w2", lambda b: [b["n_routed_experts"], b["hidden_size"], b["moe_intermediate_size"]]),
        ("e_score_correction_bias", lambda b: [b["n_routed_experts"]]),
        ("shared_gate_up", lambda b: [2 * b["moe_intermediate_size"], b["hidden_size"]]),
        ("shared_down", lambda b: [b["hidden_size"], b["moe_intermediate_size"]]),
    ],
    # Gemma-4 router (GemmaRouterLayer) — gemma_moe input 2: gate [E, hidden]
    # (Gemm orientation), scale [hidden], per_expert_scale [E].
    "router": [
        ("gate", lambda b: [b["num_experts"], b["hidden_size"]]),
        ("scale", lambda b: [b["hidden_size"]]),
        ("per_expert_scale", lambda b: [b["num_experts"]]),
    ],
    # Gemma-4 SwitchGLU experts (SwitchGluExpertsLayer) — gemma_moe input 3.
    # Affine storage on disk; mathematically per-expert SwiGLU/GeGLU.
    "switch_glu": [
        ("gate_proj", lambda b: [b["num_experts"], b["moe_intermediate_size"], b["hidden_size"]]),
        ("down_proj", lambda b: [b["num_experts"], b["hidden_size"], b["moe_intermediate_size"]]),
        ("up_proj", lambda b: [b["num_experts"], b["moe_intermediate_size"], b["hidden_size"]]),
    ],
    # GDN linear_attn (GatedDeltaNetLayer) — gated_delta_net input 4.
    # conv1d is [conv_dim, 1, 4] on disk; the singleton dim drops at use.
    "linear_attn": [
        ("conv1d", lambda b: [b["gdn_conv_dim"], 1, 4]),
        ("a_log", lambda b: [b["linear_num_value_heads"]]),
        ("dt_bias", lambda b: [b["linear_num_value_heads"]]),
        ("norm", lambda b: [b["gdn_value_dim"]]),
    ],
}


def bundle_key_for(manifest_key, dims, bounds):
    """Resolve which BUNDLE_LEAVES entry a manifest marker names — the
    impl_lib.rs discriminators: which expert-count bound exists
    (num_local_experts=Mixtral / num_experts=Qwen / n_routed_experts=
    DeepSeek), else the key's leaf name (router/switch_glu/linear_attn).
    A 1-D `["<bound>"]` entry where <bound> is NOT hidden_size is a
    genuine weight; `["hidden_size"]` on a NON-linear leaf (moe, mlp,
    block_sparse_moe) is a structural bundle marker."""
    if len(dims) != 1 or not isinstance(dims[0], str):
        return None
    leaf = manifest_key.split(".")[-1]
    if leaf in ("router", "switch_glu", "linear_attn"):
        return leaf
    marker = dims[0]
    if marker != "hidden_size":
        return None
    # `["hidden_size"]`: a bundle marker iff an expert-count bound
    # exists (impl_lib.rs applies_to discriminators) — a lone hidden
    # dim on a genuinely-1-D weight only occurs on non-MoE arches.
    for k in ("num_local_experts", "num_experts", "n_routed_experts"):
        if k in bounds:
            return k
    return None


def synth_bundle(marker, bounds):
    entry = BUNDLE_LEAVES[marker]
    leaves_spec = entry(bounds) if callable(entry) else entry
    leaves = {}
    for name, shp in leaves_spec:
        shape = shp(bounds) if callable(shp) else shp
        leaves[name] = rand_tensor(shape)
    return Tree(**leaves)


def scan_bias_refs(carrier_text, layered_keys):
    """`name.bias` refs in the carrier text the manifest doesn't declare.
    Returns [(dotted_base, layered)] — base excludes `.bias`. (Kept for
    the harness dump; the trees attach biases as weight attributes.)"""
    refs = set()
    for m in re.finditer(r"([A-Za-z_][A-Za-z_0-9.]*)\.bias\b", carrier_text):
        refs.add(m.group(1))
    return sorted(refs)


# Ops whose weight arguments are layer-struct BUNDLES, not tensors —
# shape.rs `weight_arg_ranks` skips the tensor-rank assertion for
# exactly these (Moe/GemmaMoe/GatedDeltaNet). A manifest key is a
# bundle iff the carrier passes it to one of them.
BUNDLE_OPS = ("moe_block", "gemma_moe", "gated_delta_net")


def bundle_call_args(carrier_text):
    """Concatenated argument text of every BUNDLE_OPS call in the
    carrier (balanced-paren extraction) — the membership test for
    bundle keys."""
    out = []
    for op in BUNDLE_OPS:
        start = 0
        while (i := carrier_text.find(op + "(", start)) != -1:
            depth, j = 0, i + len(op)
            while j < len(carrier_text):
                c = carrier_text[j]
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            out.append(carrier_text[i:j])
            start = j
    return " ".join(out)


def synth_trees(manifest, bounds, depth, carrier_text):
    """Build the ambient weight trees from the manifest's dotted paths.

    Layered-ness comes from the CARRIER's own `[...]` indexing — the same
    signal the DSL solver reads — not from dottedness: a manifest key is
    per-layer iff the carrier references `key[...]` (vision manifests use
    top-level names the carrier still indexes). Bundle-ness likewise
    comes from the carrier: a key is a bundle iff it is passed to a
    BUNDLE_OPS call (shape.rs skips the tensor-rank assertion for
    exactly those ops' weight args). Bundle-marker keys with dotted
    siblings (`linear_attn` + `linear_attn.in_proj_qkv`) merge into one
    Tree per layer: the bundle leaves and the dotted leaves both live on
    it, and the Layered wrapper distributes attribute access so
    `ns.leaf[layer]`, `ns[layer].leaf`, and `ns.leaf.child[layer]` all
    resolve."""
    normalized = {}
    for key, dims in manifest.items():
        if key.startswith("__"):
            continue
        if isinstance(dims, dict):
            dims = dims["shape"]
        normalized[key] = dims

    bundle_args = bundle_call_args(carrier_text)
    # word-boundary match: bare `norm[` must not match inside
    # `input_layernorm[` — a plain substring check would layer `norm`
    layered_refs = {m.group(1) for m in re.finditer(
        r"([A-Za-z_][A-Za-z_0-9.]*)\[", carrier_text)}

    def values_for(key, dims):
        if f"{key}[" in bundle_args \
                and (m := bundle_key_for(key, dims, bounds)) is not None:
            return [synth_bundle(m, bounds) for _ in range(depth)]
        shape = [eval_formula(d, bounds) for d in dims]
        return [rand_tensor(shape) for _ in range(depth)]

    def build_ns(flat):
        """Nested Tree from a {dotted.key: value} dict. A dotted key whose
        parent path lands on a tensor (e.g. `norm1.bias` where `norm1` is
        a tensor) attaches as an attribute on that tensor."""
        root = Tree()
        for key, v in flat.items():
            parts = key.split(".")
            node = root
            for p in parts[:-1]:
                if not hasattr(node, p):
                    setattr(node, p, Tree())
                node = getattr(node, p)
            setattr(node, parts[-1], v)
        return root

    shared, layers = {}, [{} for _ in range(depth)]
    for key, dims in normalized.items():
        vals = values_for(key, dims)
        if key in layered_refs:
            for l in range(depth):
                layers[l][key] = vals[l]
        else:
            shared[key] = vals[0]

    # Merge a bundle key's dotted siblings INTO the bundle Tree
    # (qwen3-5: `linear_attn` marker + `linear_attn.in_proj_qkv` leaves
    # are one object per layer — the carrier writes both
    # `linear_attn[layer]` and `linear_attn.in_proj_qkv[layer]`;
    # qwen3-5-moe: `mlp` marker + `mlp.shared_expert.gate_proj` leaves).
    # When the manifest declares its own shared-expert leaves the
    # CARRIER computes the shared expert — drop the bundle's synthetic
    # shared leaves so moe_block doesn't run them too (qwen2-moe, whose
    # manifest has none, keeps the bundle's: there the layer struct
    # owns the shared expert).
    bundle_keys = {k for k in normalized
                   if f"{k}[" in bundle_args
                   and bundle_key_for(k, normalized[k], bounds) is not None}
    manifest_shared = {k for k in normalized
                       if re.match(r"^[A-Za-z_]+\.shared_expert", k)}
    for l in range(depth):
        for bk in bundle_keys:
            if any(k.startswith(bk + ".") for k in manifest_shared) \
                    and isinstance(layers[l].get(bk), Tree):
                for leaf in ("shared_gate_up", "shared_down", "shared_expert_gate"):
                    if hasattr(layers[l][bk], leaf):
                        delattr(layers[l][bk], leaf)
            for key in [k for k in layers[l] if k.startswith(bk + ".")]:
                leaf = key[len(bk) + 1:]
                parts = leaf.split(".")
                node = layers[l][bk]
                for p in parts[:-1]:
                    if not hasattr(node, p):
                        setattr(node, p, Tree())
                    node = getattr(node, p)
                setattr(node, parts[-1], layers[l].pop(key))

    root = build_ns(shared)
    layer_trees = [build_ns(l) for l in layers]
    for p in {p for lt in layer_trees for p in vars(lt)}:
        if hasattr(root, p):
            raise ValueError(
                f"namespace collision at {p!r}: both layered and unlayered "
                f"manifest keys live under it"
            )
        setattr(root, p, Layered(getattr(lt, p) for lt in layer_trees))

    # `.bias` siblings the carrier references but the manifest doesn't
    # declare (linear/LayerNorm biases — the loader pulls `<prefix>.bias`
    # alongside the weight). Attach each as an ATTRIBUTE on every
    # per-layer weight tensor — Layered distributes the attribute walk,
    # so `ns.leaf.bias[layer]` resolves to the layer's bias — sized to
    # the weight's output dim (manifest orientation, dim 1): what
    # bias_add adds to.
    for m in re.finditer(r"([A-Za-z_][A-Za-z_0-9.]*)\.bias\b", carrier_text):
        base = m.group(1)
        if base not in normalized:
            raise KeyError(
                f"carrier references {base}.bias but {base} is not in "
                f"weights.json — no way to size the bias"
            )
        w = root
        for p in base.split("."):
            w = getattr(w, p)
        dims = normalized[base]
        # bias width: the weight's LAST dim (output N for [K, N] gemm
        # weights; D itself for rank-1 [D] norm weights)
        width = eval_formula(dims[-1], bounds)
        targets = list(w) if isinstance(w, Layered) else [w]
        if all(hasattr(t, "bias") for t in targets):
            continue  # the manifest declares it (build_ns attached it)
        for t in targets:
            t.bias = rand_tensor([width])
    return root


# ───────────────────────── rope / attn geometry helpers ─────────────────────────


def build_rotary(head_dim, rotary_dim, max_pos, theta, proportional=False):
    """RotaryCache replica (rotary.rs): inv_freq[i] =
    theta^(-2i/denominator), rows [cos(half) | sin(half)] of width
    rotary_dim. `proportional` (Gemma4 global class) uses head_dim as
    the exponent denominator while rotating only rotary_dim dims."""
    denom = head_dim if proportional else rotary_dim
    half = rotary_dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float64) * 2.0 / denom))
    ang = torch.arange(max_pos, dtype=torch.float64).unsqueeze(1) * inv.unsqueeze(0)
    table = torch.cat([ang.cos(), ang.sin()], dim=1)
    return table[:, :half].float(), table[:, half:].float()


def resolve_rotary_config(cfg, bounds):
    """The rotary facts the macro bakes (codegen.rs rotary_load*): theta,
    partial factor, Gemma4 proportional-global, local theta. Returns a
    dict of {name: (head_dim, rotary_dim, max_pos, theta, proportional)}."""
    scalars = {k: v for k, v in cfg.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    max_pos = bounds.get("max_position_embeddings", 64)
    head_dim = bounds.get("head_dim", 0)

    def theta_of(key):
        v = scalars.get(key, cfg.get(key))
        if isinstance(v, (int, float)):
            return float(v)
        # rope_parameters.full_attention.rope_theta (hoisted)
        v = json_path(cfg, f"rope_parameters.{key.split('_', 1)[0]}_attention.{key}" if key.startswith("rope_") else key)
        return float(v) if isinstance(v, (int, float)) else 10000.0

    out = {}
    partial = scalars.get("partial_rotary_factor", cfg.get("partial_rotary_factor"))
    gpf = scalars.get("global_partial_rotary_factor", cfg.get("global_partial_rotary_factor"))
    if isinstance(gpf, (int, float)) and abs(gpf - 1.0) > 1e-9 and "global_head_dim" in bounds:
        g_hd = bounds["global_head_dim"]
        out["rotary"] = (g_hd, round(gpf * g_hd), max_pos, theta_of("rope_theta"), True)
    elif isinstance(partial, (int, float)) and abs(partial - 1.0) > 1e-9:
        out["rotary"] = (head_dim, round(partial * head_dim), max_pos, theta_of("rope_theta"), False)
    else:
        out["rotary"] = (head_dim, head_dim, max_pos, theta_of("rope_theta"), False)
    if "rope_local_base_freq" in scalars or "rope_local_base_freq" in cfg or "local_rope_theta" in cfg:
        out["rotary_local"] = (head_dim, head_dim, max_pos, theta_of("rope_local_base_freq"), False)
    return out


def attn_scale_for(cfg, bounds, head_dim):
    if (m := cfg.get("attention_multiplier")) is not None and isinstance(m, (int, float)):
        return float(m)
    if (q := cfg.get("query_pre_attn_scalar")) is not None and isinstance(q, (int, float)):
        return float(q) ** -0.5
    return 1.0 / math.sqrt(head_dim)


# ───────────────────────── op shims (torch) ─────────────────────────
# One function per DSL op (classified.rs `from_name`), mirroring its
# kernel's reference math. STATE is set by main() before exec.

STATE = {}


def embed(ids, table):
    return table[ids]


def gemm(x, w):
    """DSL gemm: w is in the manifest's [K, N] orientation (shape.rs
    `sig_gemm` unifies x's last dim with w[0]). The [out, in] HF layout
    is a load-time transpose the Rust side owns — the synthetic
    checkpoint writer mirrors it there, not here."""
    return x @ w


def rmsnorm(x, w):
    eps = STATE["rms_eps"]
    off = STATE.get("norm_gain_offset", 0.0)
    if w.dim() == 1 and x.shape[-1] % w.shape[0] == 0 and w.shape[0] != x.shape[-1]:
        heads = x.shape[-1] // w.shape[0]
        xv = x.view(*x.shape[:-1], heads, w.shape[0])
        var = xv.pow(2).mean(-1, keepdim=True)
        return (xv * torch.rsqrt(var + eps) * (w + off)).view(*x.shape)
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * (w + off)


def rmsnorm_unit(x):
    """Gemma4 `v_norm`: per-head unit RMSNorm (RMSNormNoScale(head_dim);
    metal rope_append_normed norms V over head_dim inside each head, NOT
    the flat kv_heads*head_dim row). The V row width uniquely identifies
    the attention class (sliding kv_dim vs global kv_dim), so key off it;
    anything that doesn't match a class geometry falls back to whole-row.
    """
    hd = None
    for kvh_key, hd_key in (
        ("num_key_value_heads", "head_dim"),
        ("num_global_key_value_heads", "global_head_dim"),
    ):
        kvh = STATE["bounds"].get(kvh_key)
        h = STATE["bounds"].get(hd_key)
        if isinstance(kvh, int) and isinstance(h, int) and kvh * h == x.shape[-1]:
            hd = h
            break
    if hd is None or hd == x.shape[-1]:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + STATE["rms_eps"])
    xv = x.view(*x.shape[:-1], -1, hd)
    var = xv.pow(2).mean(-1, keepdim=True)
    return (xv * torch.rsqrt(var + STATE["rms_eps"])).view(*x.shape)


def mean(x):
    return x.mean(-1, keepdim=True)


def sub(a, b):
    return a - b


def add(a, b):
    return a + b


def bias_add(x, b):
    return x + b


def silu(x):
    return torch.nn.functional.silu(x)


def gelu(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


def gelu_erf(x):
    return torch.nn.functional.gelu(x, approximate="none")


def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


def tanh_softcap(x):
    cap = STATE["final_logit_softcapping"]
    return cap * torch.tanh(x / cap)


def scalar(name):
    """DSL `scalar(<config key>)`: parse_python.rs Expr::ConfigScalar —
    the value is the config field by NAME. The env binds every numeric
    config field as an ambient name, so here the argument arrives as the
    value itself."""
    if isinstance(name, str):
        return STATE["scalars"][name]
    return name


def sqrt(x):
    return math.sqrt(x)


def scalar_weight_mul(x, w):
    return x * w


def recip_scalar(v):
    return 1.0 / v


def reshape(x, shape):
    return x.view(*[int(s) for s in shape])


# ── rope_append family (rope.metal: NeoX half-split, paged V write) ──


def _rope_one(t, heads, cos, sin, half, pair_off):
    """NeoX half-split rotation with PAIR_OFF (standard: rot_dim/2 —
    pairs live inside the rot window; proportional: head_dim/2 — pairs
    span the full head). Lanes ≥ half pass through."""
    n = t.shape[0]
    tv = t.view(n, heads, -1)  # [n, heads, head_dim]
    out = tv.clone()
    x0, x1 = tv[:, :, :half], tv[:, :, pair_off:pair_off + half]
    r0 = x0 * cos.unsqueeze(1) - x1 * sin.unsqueeze(1)
    r1 = x1 * cos.unsqueeze(1) + x0 * sin.unsqueeze(1)
    out[:, :, :half] = r0
    out[:, :, pair_off:pair_off + half] = r1
    return out.reshape(n, -1)


def _paged_write(k, v, layer, kv_heads, head_dim):
    kcache, vcache = STATE["kv_cache"][layer]
    sm = STATE["slot_mapping"]
    for t in range(k.shape[0]):
        slot = sm[t]
        blk, off = slot // BLOCK_SIZE, slot % BLOCK_SIZE
        kcache[blk, :, off, :] = k[t].reshape(kv_heads, head_dim)
        vcache[blk, :, off, :] = v[t].reshape(kv_heads, head_dim)


def rope_append(q, k, v, positions, rotary, layer_cache):
    """(q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]).
    NeoX half-split; Q/K rotate, V passes; K/V appended to the paged cache.
    `layer_cache` is `kv_cache[layer]` — the layer index is recovered from
    the object identity via STATE's per-layer list."""
    layer = STATE["layer_of"][id(layer_cache)]
    rot = STATE["rotary_of"][id(rotary)]
    (head_dim, rotary_dim, _, _, proportional), (cos_t, sin_t) = rot["spec"], rot["tables"]
    pos = torch.as_tensor(positions, dtype=torch.long)
    cos, sin = cos_t[pos], sin_t[pos]  # [n, half]
    half = rotary_dim // 2
    pair_off = head_dim // 2 if proportional else half
    qh, kvh = rot["q_heads"], rot["kv_heads"]
    out_q = _rope_one(q, qh, cos, sin, half, pair_off)
    out_k = _rope_one(k, kvh, cos, sin, half, pair_off)
    _paged_write(out_k, v, layer, kvh, head_dim)
    return out_q, out_k, v


def rope_append_interleaved(q, k, v, positions, rotary, layer_cache):
    """Cohere/GPT-J style: pair (2i, 2i+1) within the rot window."""
    layer = STATE["layer_of"][id(layer_cache)]
    rot = STATE["rotary_of"][id(rotary)]
    (head_dim, rotary_dim, _, _, _), (cos_t, sin_t) = rot["spec"], rot["tables"]
    pos = torch.as_tensor(positions, dtype=torch.long)
    cos, sin = cos_t[pos], sin_t[pos]
    half = rotary_dim // 2
    qh, kvh = rot["q_heads"], rot["kv_heads"]

    def rot_inter(t, heads):
        n = t.shape[0]
        tv = t.view(n, heads, head_dim)
        x0 = tv[:, :, 0::2][:, :, :half]
        x1 = tv[:, :, 1::2][:, :, :half]
        out = tv.clone()
        out[:, :, 0::2][:, :, :half] = x0 * cos.unsqueeze(1) - x1 * sin.unsqueeze(1)
        out[:, :, 1::2][:, :, :half] = x1 * cos.unsqueeze(1) + x0 * sin.unsqueeze(1)
        return out.reshape(n, -1)

    out_q = rot_inter(q, qh)
    out_k = rot_inter(k, kvh)
    _paged_write(out_k, v, layer, kvh, head_dim)
    return out_q, out_k, v


# ── attention family ──


def attention(q, k, v, layer_cache=None, block_table=None):
    """Causal paged prefill (cpu_golden.rs attention_prefill_paged): token
    i attends cache positions [0, positions[i]] through block_table; GQA
    broadcast; scale from STATE. The 3-arg form (modernbert) is the
    EncoderAttentionImpl path: bidirectional, no KV cache read — every
    token attends every token."""
    if layer_cache is None:
        n = q.shape[0]
        st = STATE["attn"][0]
        qh, kvh, hd, scale = st["q_heads"], st["kv_heads"], st["head_dim"], st["scale"]
        qv = q.view(n, qh, hd)
        kv = k.view(n, kvh, hd).transpose(0, 1)  # [kvh, n, hd]
        vv = v.view(n, kvh, hd).transpose(0, 1)
        if kvh != qh:
            kv = kv.repeat_interleave(qh // kvh, dim=0)
            vv = vv.repeat_interleave(qh // kvh, dim=0)
        scores = torch.einsum("ihd,hjd->hij", qv, kv) * scale
        probs = torch.softmax(scores, dim=-1)  # bidirectional: no mask
        return torch.einsum("hij,hjd->ihd", probs, vv).reshape(n, qh * hd)
    layer = STATE["layer_of"][id(layer_cache)]
    st = STATE["attn"][layer]
    qh, kvh, hd, scale = st["q_heads"], st["kv_heads"], st["head_dim"], st["scale"]
    kcache, vcache = layer_cache
    n = q.shape[0]
    qv = q.view(n, qh, hd)
    out = torch.empty_like(qv)
    bt = STATE["block_table"]
    for t in range(n):
        pos = STATE["positions"][t]
        attend = pos + 1
        ks = torch.empty(attend, kvh, hd)
        vs = torch.empty_like(ks)
        for j in range(attend):
            blk, off = j // BLOCK_SIZE, j % BLOCK_SIZE
            pb = bt[blk]
            ks[j] = kcache[pb, :, off, :]
            vs[j] = vcache[pb, :, off, :]
        for h in range(qh):
            kv_h = h // (qh // kvh)
            scores = (ks[:, kv_h, :] @ qv[t, h]) * scale
            probs = torch.softmax(scores, dim=0)
            out[t, h] = probs @ vs[:, kv_h, :]
    return out.reshape(n, qh * hd)


def sliding_attention(q, k, v, layer_cache, block_table):
    """attention.metal: query at absolute position `q` attends keys k with
    0 <= q - k < window (self + window-1 prior)."""
    layer = STATE["layer_of"][id(layer_cache)]
    st = STATE["attn"][layer]
    qh, kvh, hd, scale, window = st["q_heads"], st["kv_heads"], st["head_dim"], st["scale"], st["window"]
    kcache, vcache = layer_cache
    n = q.shape[0]
    qv = q.view(n, qh, hd)
    out = torch.empty_like(qv)
    bt = STATE["block_table"]
    for t in range(n):
        pos = STATE["positions"][t]
        lo = max(0, pos - window + 1)
        ks = torch.empty(pos - lo + 1, kvh, hd)
        vs = torch.empty_like(ks)
        for jj, j in enumerate(range(lo, pos + 1)):
            blk, off = j // BLOCK_SIZE, j % BLOCK_SIZE
            pb = bt[blk]
            ks[jj] = kcache[pb, :, off, :]
            vs[jj] = vcache[pb, :, off, :]
        for h in range(qh):
            kv_h = h // (qh // kvh)
            scores = (ks[:, kv_h, :] @ qv[t, h]) * scale
            probs = torch.softmax(scores, dim=0)
            out[t, h] = probs @ vs[:, kv_h, :]
    return out.reshape(n, qh * hd)


def varlen_attention(q, k, v, cu_seqlens, max_seqlen):
    """vision_varlen_attn.metal: cacheless, NON-causal, per-segment SDPA,
    scale = head_dim**-0.5, MHA (no GQA)."""
    hd = STATE["vision_head_dim"]
    scale = 1.0 / math.sqrt(hd)
    n = q.shape[0]
    heads = q.shape[1] // hd
    qv = q.view(n, heads, hd)
    kv = k.view(n, heads, hd)
    vv = v.view(n, heads, hd)
    out = torch.empty_like(qv)
    for s in range(len(cu_seqlens) - 1):
        b, e = int(cu_seqlens[s]), int(cu_seqlens[s + 1])
        if e <= b:
            continue
        for h in range(heads):
            scores = (qv[b:e, h] @ kv[b:e, h].T) * scale
            probs = torch.softmax(scores, dim=-1)
            out[b:e, h] = probs @ vv[b:e, h]
    return out.reshape(n, heads * hd)


def vision_rope(q, k, cos, sin):
    """vision_rope_2d.metal: rotate_half NeoX per (t,h), freqs row tiled
    by concat — out = x*cos(f) + rotate_half(x)*sin(f), f per-token."""
    hd = STATE["vision_head_dim"]
    n = q.shape[0]
    heads = q.shape[1] // hd
    half = hd // 2
    freqs = STATE["vision_freqs"]  # [n, half] f32

    def rot2d(t):
        tv = t.view(n, heads, hd)
        f = freqs.unsqueeze(1)  # [n, 1, half]
        c, s = f.cos(), f.sin()
        x0, x1 = tv[:, :, :half], tv[:, :, half:]
        out = tv.clone()
        out[:, :, :half] = x0 * c - x1 * s
        out[:, :, half:] = x1 * c + x0 * s
        return out.reshape(n, -1)

    return rot2d(q), rot2d(k)


# ── MLA (cuda mla_attention_eval) ──


def mla_split(kv_a):
    rank = STATE["bounds"]["kv_lora_rank"]
    return kv_a[:, :rank], kv_a[:, rank:]


def mla_attention(q, kv_b, k_pe, positions, rotary, layer_cache, block_table):
    """cuda eval: extract q_pe (per-head rope tail), interleaved-rope q_pe
    and k_pe, write K/V to the cache, causal attention with MLA scale,
    slice output to heads*v_head_dim."""
    b = STATE["bounds"]
    heads, nope, rope_d, vhd = b["num_attention_heads"], b["qk_nope_head_dim"], b["qk_rope_head_dim"], b["v_head_dim"]
    qk_hd = nope + rope_d
    layer = STATE["layer_of"][id(layer_cache)]
    kcache, vcache = layer_cache
    n = q.shape[0]

    # Per-head layout: q [n, heads, nope+rope] — rope tail is the LAST
    # rope_d of each head.
    qv = q.view(n, heads, qk_hd)
    q_pe = qv[:, :, nope:]  # [n, heads, rope_d]

    # Interleaved rope on q_pe / k_pe (pair (2i, 2i+1)).
    theta = STATE["mla_theta"]
    pos = torch.as_tensor(positions, dtype=torch.long)
    half = rope_d // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float64) * 2.0 / rope_d))
    ang = pos.to(torch.float64).unsqueeze(1) * inv.unsqueeze(0)
    cos, sin = ang.cos().float(), ang.sin().float()  # [n, half]

    def rot_i(t, hdim):
        tv = t.view(n, -1, hdim)
        x0 = tv[:, :, 0::2]
        x1 = tv[:, :, 1::2]
        out = tv.clone()
        out[:, :, 0::2] = x0 * cos.unsqueeze(1) - x1 * sin.unsqueeze(1)
        out[:, :, 1::2] = x1 * cos.unsqueeze(1) + x0 * sin.unsqueeze(1)
        return out.reshape(n, -1)

    q_pe_r = rot_i(q_pe.reshape(n, heads * rope_d), rope_d)
    k_pe_r = rot_i(k_pe, rope_d)

    # Assemble K: per head [kv_b_head | k_pe], V: kv_b per head padded…
    # mla_assemble_k/v: K[t, h] = [kv_b[t, h*(nope+vhd) .. +nope] | k_pe[t]],
    # V[t, h] = kv_b[t, h*(nope+vhd)+nope .. +vhd]. k_pe is SINGLE-HEAD
    # ([T, rope_dim], impl_lib MlaAttention doc) — broadcast across heads.
    kv_bv = kv_b.view(n, heads, nope + vhd)
    k = torch.cat([kv_bv[:, :, :nope], k_pe_r.view(n, 1, rope_d).expand(n, heads, rope_d)], dim=-1)  # [n, heads, qk_hd]
    v = kv_bv[:, :, nope:]  # [n, heads, vhd]

    # Write paged cache at this step's slots (cache geometry: kv heads =
    # q heads for MLA, head_dim = qk_hd for K... K and V widths differ, so
    # the cache holds [nope+rope | pad] and [vhd] — the eval writes K at
    # qk_head_dim and V zero-padded to qk_head_dim.
    q_pe_final = q_pe_r.view(n, heads * rope_d)
    qv = qv.clone()
    qv[:, :, nope:] = q_pe_r.view(n, heads, rope_d)
    q = qv.reshape(n, heads * qk_hd)

    # attention: causal over cache positions, MLA scale, per-head over
    # qk_hd; output sliced to vhd.
    scale = STATE["mla_scale"]
    k_full = torch.zeros(n, heads, qk_hd)
    k_full[:, :, :] = k
    v_full = torch.zeros(n, heads, qk_hd)
    v_full[:, :, :vhd] = v
    _paged_write(k.reshape(n, heads * qk_hd), v_full.reshape(n, heads * qk_hd), layer, heads, qk_hd)
    bt = STATE["block_table"]
    out = torch.empty(n, heads, vhd)
    for t in range(n):
        p = STATE["positions"][t]
        attend = p + 1
        ks = torch.empty(attend, heads, qk_hd)
        vs = torch.empty(attend, heads, qk_hd)
        for j in range(attend):
            blk, off = j // BLOCK_SIZE, j % BLOCK_SIZE
            pb = bt[blk]
            ks[j] = kcache[pb, :, off, :]
            vs[j] = vcache[pb, :, off, :]
        for h in range(heads):
            scores = (ks[:, h, :] @ q.view(n, heads, qk_hd)[t, h]) * scale
            probs = torch.softmax(scores, dim=0)
            out[t, h] = (probs @ vs[:, h, :])[:vhd]
    return out.reshape(n, heads * vhd)


# ── GDN (cpu_golden.rs gdn_* pipeline, exact order) ──


def gated_delta_net(qkv, z, a, b, layer_w):
    bnds = STATE["bounds"]
    nk = bnds["linear_num_key_heads"]
    nv = bnds["linear_num_value_heads"]
    hk = bnds["linear_key_head_dim"]
    hv = bnds["linear_value_head_dim"]
    key_dim, value_dim = nk * hk, nv * hv
    conv_dim = 2 * key_dim + value_dim
    t = qkv.shape[0]

    # 1. causal conv1d + SiLU (kernel 4, weight [conv_dim, 1, 4])
    w = layer_w.conv1d
    kk = w.shape[-1]
    conv = torch.zeros(t, conv_dim)
    xpad = torch.cat([torch.zeros(kk - 1, conv_dim), qkv])
    for ti in range(t):
        for c in range(conv_dim):
            acc = 0.0
            for j in range(kk):
                acc += float(w[c, 0, j]) * float(xpad[ti + j, c])
            conv[ti, c] = torch.nn.functional.silu(torch.tensor(acc))
    # 2. split -> q, k, v
    q = conv[:, 0:key_dim]
    k = conv[:, key_dim:2 * key_dim]
    v = conv[:, 2 * key_dim:]
    # 3. gating
    a_log, dt_bias = layer_w.a_log, layer_w.dt_bias
    softplus = lambda x: torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)  # noqa: E731
    g = -(a_log.exp().unsqueeze(0)) * softplus(a + dt_bias.unsqueeze(0))
    beta = torch.sigmoid(b)
    # 4. recurrent delta-rule scan (zero init state)
    scale = 1.0 / math.sqrt(hk)
    groups = nv // nk
    state = torch.zeros(nv, hv, hk)
    o = torch.zeros(t, value_dim)
    for ti in range(t):
        for h in range(nv):
            ki = h // groups
            qs = q[ti, ki * hk:(ki + 1) * hk]
            ksr = k[ti, ki * hk:(ki + 1) * hk]
            qinv = scale / math.sqrt(float(qs.pow(2).sum()) + 1e-6)
            kinv = 1.0 / math.sqrt(float(ksr.pow(2).sum()) + 1e-6)
            qn = qs * qinv
            kn = ksr * kinv
            sh = state[h]
            sh *= math.exp(float(g[ti, h]))
            gt = float(beta[ti, h])
            vs_ = v[ti, h * hv:(h + 1) * hv]
            for vd in range(hv):
                sk = float(sh[vd] @ kn)
                u = gt * (float(vs_[vd]) - sk)
                sh[vd] += u * kn
                o[ti, h * hv + vd] = float(sh[vd] @ qn)
    # 5. gated RMSNorm: rmsnorm(x) * w * silu(z)
    var = o.pow(2).mean(-1, keepdim=True)
    xn = o * torch.rsqrt(var + 1e-6)
    return xn * layer_w.norm * torch.nn.functional.silu(z)


# ── gates (gate_split / gate_apply / gate_scale kernels) ──


def gate_split(qg):
    """Per-head deinterleave: each head's 2*head_dim block is [q | gate]."""
    heads = STATE["bounds"]["num_attention_heads"]
    hd = STATE["bounds"]["head_dim"]
    n = qg.shape[0]
    tv = qg.view(n, heads, 2 * hd)
    return tv[:, :, :hd].reshape(n, heads * hd), tv[:, :, hd:].reshape(n, heads * hd)


def gate_apply(attn, gate):
    return attn * torch.sigmoid(gate)


def gate_scale(routed, shared_y, g):
    """out = routed + shared_y * sigmoid(g); g is [T, 1] row-broadcast."""
    return routed + shared_y * torch.sigmoid(g)


# ── MoE (cuda kernels.rs topk_softmax / topk_noaux_tc + layers_moe.rs
#    forward conventions; gemma from lower_gemma_moe) ──


def _swiglu_stacked(x, w1, w2, weight):
    """One expert step: w1 is stacked gate+up [2*inter, hidden] (Linear
    orientation [out, in]), w2 [hidden, inter]. `weight` = routing
    score applied to the down output."""
    inter = w2.shape[1]
    gu = x @ w1.T  # [n, 2*inter]
    gate, up = gu[:, :inter], gu[:, inter:]
    return weight * ((torch.nn.functional.silu(gate) * up) @ w2.T)


def moe_block(x, layer_w):
    b = STATE["bounds"]
    cfg = STATE["cfg"]
    top_k = b.get("num_experts_per_tok") or b.get("top_k_experts") or b.get("top_k")
    n = x.shape[0]
    if hasattr(layer_w, "w1") and "n_routed_experts" in b:
        # DeepSeek noaux_tc (grouped_topk_noaux.cu): sigmoid scores;
        # group_score = sum of top-2 (sigmoid+bias) per group; select
        # topk_group groups; within them top topk by (sigmoid+bias);
        # weights = unbiased sigmoid, × routed_scaling_factor (renorm
        # only if norm_topk_prob); shared expert is a PLAIN add.
        n_group = b.get("n_group", 0)
        topk_group = b.get("topk_group", 0)
        logits = x @ layer_w.gate  # gate [hidden, E] gemm orientation
        sig = torch.sigmoid(logits)
        e = logits.shape[1]
        bias = layer_w.e_score_correction_bias
        biased = sig + bias
        if n_group and topk_group:
            per = e // n_group
            group_scores = torch.stack([
                biased[:, g * per:(g + 1) * per].topk(2, dim=-1).values.sum(-1)
                for g in range(n_group)
            ], dim=1)
            sel_groups = group_scores.topk(topk_group, dim=-1).indices
            mask = torch.zeros_like(biased)
            for g in sel_groups[0].tolist():
                mask[:, g * per:(g + 1) * per] = 1.0
            cand = biased * mask - 1e9 * (1 - mask)
        else:
            cand = biased
        top_scores, idx = torch.topk(cand, top_k, dim=-1)
        weights = sig.gather(1, idx)
        if b.get("norm_topk_prob"):
            weights = weights / weights.sum(-1, keepdim=True)
        weights = weights * cfg.get("routed_scaling_factor", 1.0)
        out = torch.zeros_like(x)
        for t in range(n):
            for j in range(top_k):
                ex = int(idx[t, j])
                out[t] += _swiglu_stacked(x[t:t+1], layer_w.w1[ex], layer_w.w2[ex], 1.0)[0] * float(weights[t, j])
        # shared expert: plain add (no sigmoid gate)
        out = out + _swiglu_stacked(x, layer_w.shared_gate_up, layer_w.shared_down, 1.0)
        return out
    if hasattr(layer_w, "w1") and "num_local_experts" in b:
        # Mixtral: softmax over the FULL logits first, then top-k over
        # the softmaxed values (topk_softmax: softmax → moeTopK).
        logits = x @ layer_w.gate
        probs = torch.softmax(logits, dim=-1)
        scores, idx = torch.topk(probs, top_k, dim=-1)
        out = torch.zeros_like(x)
        for t in range(n):
            for j in range(top_k):
                ex = int(idx[t, j])
                out[t] += _swiglu_stacked(x[t:t+1], layer_w.w1[ex], layer_w.w2[ex], 1.0)[0] * float(scores[t, j])
        return out
    # Qwen-MoE (SharedFusedMoE): same softmax-first routing, optional
    # renorm (norm_topk_prob), shared expert with sigmoid gate when
    # shared_expert_intermediate_size > 0.
    norm_topk = bool(b.get("norm_topk_prob", 0))
    logits = x @ layer_w.gate
    probs = torch.softmax(logits, dim=-1)
    scores, idx = torch.topk(probs, top_k, dim=-1)
    if norm_topk:
        scores = scores / scores.sum(-1, keepdim=True)
    out = torch.zeros_like(x)
    for t in range(n):
        for j in range(top_k):
            ex = int(idx[t, j])
            out[t] += _swiglu_stacked(x[t:t+1], layer_w.w1[ex], layer_w.w2[ex], 1.0)[0] * float(scores[t, j])
    if b.get("shared_expert_intermediate_size", 0) and hasattr(layer_w, "shared_gate_up"):
        shared = _swiglu_stacked(x, layer_w.shared_gate_up, layer_w.shared_down, 1.0)
        shared = shared * torch.sigmoid(x @ layer_w.shared_expert_gate.T)
        out = out + shared
    return out


def gemma_moe(router_in, moe_in, router_w, experts_w):
    """lower_gemma_moe: xr = rmsnorm(router_in, router.scale); logits =
    xr @ router.gate.T; topk; × hidden^-0.5; softmax; × per_expert_scale;
    GeGLU experts; weighted sum."""
    b = STATE["bounds"]
    top_k = b.get("top_k_experts") or b.get("num_experts_per_tok")
    hidden = b["hidden_size"]
    n = router_in.shape[0]
    xr = router_in * torch.rsqrt(router_in.pow(2).mean(-1, keepdim=True) + STATE["rms_eps"]) * router_w.scale
    logits = xr @ router_w.gate.T
    scores, idx = torch.topk(logits, top_k, dim=-1)
    scores = torch.softmax(scores * (hidden ** -0.5), dim=-1)
    scores = scores * router_w.per_expert_scale[idx]
    out = torch.zeros_like(moe_in)
    for t in range(n):
        for j in range(top_k):
            ex = int(idx[t, j])
            gu = moe_in[t:t+1] @ experts_w.gate_proj[ex].T
            up = moe_in[t:t+1] @ experts_w.up_proj[ex].T
            act = torch.nn.functional.gelu(gu, approximate="tanh") * up
            out[t] += (act @ experts_w.down_proj[ex].T)[0] * float(scores[t, j])
    return out


# ── vision extras ──


def pos_embed(position_ids, table):
    return table[torch.as_tensor(position_ids, dtype=torch.long)]


def embedding_gather(x, indices):
    idx = torch.as_tensor(indices, dtype=torch.long)
    return x[idx]


def avg_pool_2d(x):
    """[L, e] → [L/pool_factor, e]: the patch grid is L = g*g rows,
    pooled with kernel k = sqrt(factor) — average each k×k block of ROWS
    (the kernel treats the row axis as the grid axis, channel-last)."""
    f = STATE["bounds"]["vision_pool_factor"]
    k = int(round(math.sqrt(f)))
    l, e = x.shape
    g = int(round(math.sqrt(l)))
    grid = x.view(g, g, e)
    # k×k pooling over the (g, g) grid
    pooled = grid.view(g // k, k, g // k, k, e).mean(dim=(1, 3))
    return pooled.reshape(l // f, e)


def embed_keys_of(carrier_text):
    """Manifest keys passed as `embed(input_ids, <key>)` — the Embedding
    leaves (verbatim [vocab, hidden] on disk; also the tie source)."""
    return {m.group(1) for m in re.finditer(
        r"embed\(\s*input_ids\s*,\s*([A-Za-z_][A-Za-z_0-9.]*)\s*\)", carrier_text)}


def is_tied(arch, cfg):
    """config.rs tie resolution: explicit config flag, else the arch's
    tie_default (gemma2/gemma3)."""
    aj_path = ARCH_DIR / arch / "arch.json"
    aj = json.loads(aj_path.read_text()) if aj_path.exists() else {}
    return bool(cfg.get("tie_word_embeddings", aj.get("tie_default", False)))


# ───────────────────────── synthetic checkpoint writer ─────────────────────────
# Emits `model.safetensors` + `config.json` in HF layout from the SAME
# tree tensors the oracle just executed, replicating codegen.rs
# `safetensors_prefix` (digit-suffix translation, weight_leaf_renames,
# layered `model.layers.{l}.{path}`, unlayered `model.{path}`, lm_head
# top-level, decoder-prefix wrap/replace, packed-split row-wise fusion,
# tie omission) — so the Rust gate loads byte-identical weights through
# the macro-emitted `load`. One mechanism for every arch; per-arch
# facts are the arch.json / weights.json declarations already read.


def _ckpt_key(joined, index, dec_prefix):
    """safetensors_prefix for a non-vision path: layered/unlayered/lm_head
    + decoder_safetensors_prefix wrap/replace."""
    if joined == "lm_head":
        return "lm_head"
    key = f"model.layers.{index}.{joined}" if index is not None else f"model.{joined}"
    if dec_prefix is None:
        return key
    prefix = dec_prefix.rstrip(".")
    if key == "lm_head":
        return key
    if key.startswith("model") and prefix.startswith("model"):
        return f"{prefix}{key[len('model'):]}"
    return f"{prefix}.{key}"


def write_checkpoint(out_dir, arch, cfg, manifest, bounds, trees, depth, carrier_text):
    from safetensors.torch import save_file

    # Dual-class layer split (gemma4): pattern/remainder decide which
    # attention class layer l runs — write_checkpoint routes each layer's
    # shared disk leaf through its OWN class's tree (see the
    # class_shared arm below).
    pattern = bounds.get("sliding_window_pattern", 0)
    remainder = bounds.get(
        "sliding_window_global_remainder", pattern - 1 if pattern else 0)

    aj = {}
    aj_path = ARCH_DIR / arch / "arch.json"
    if aj_path.exists():
        aj = json.loads(aj_path.read_text())
    dec_prefix = aj.get("decoder_safetensors_prefix")
    renames = aj.get("weight_leaf_renames") or {}
    packed = manifest.get("__packed_splits__") or {}
    packed_targets = {t for targets in packed.values() for t in targets}
    tie = bool(cfg.get("tie_word_embeddings", aj.get("tie_default", False)))

    normalized = {}
    for key, dims in manifest.items():
        if key.startswith("__"):
            continue
        if isinstance(dims, dict):
            dims = dims["shape"]
        normalized[key] = dims

    # Which keys are layered: the SAME `[...]`-ref census synth_trees
    # applies (the DSL solver's signal — a manifest key is per-layer
    # iff the carrier indexes it).
    layered_refs = {m.group(1) for m in re.finditer(
        r"([A-Za-z_][A-Za-z_0-9.]*)\[", carrier_text)}
    bundle_args = bundle_call_args(carrier_text)
    # The embed() call's weight argument — an Embedding ([vocab, hidden]
    # on disk verbatim, NOT a transposed Linear), discovered from the
    # carrier text exactly as the macro's embed matcher does.
    embed_keys = embed_keys_of(carrier_text)

    def resolve(key):
        """Tree walk: `a.b.c` → trees.a.b.c (Layered distributes)."""
        node = trees
        for p in key.split("."):
            node = getattr(node, p)
        return node

    def dsl_to_disk(key):
        """digit-suffix translation + weight_leaf_renames (longest-suffix
        match — codegen applies first hit; renames are ordered by
        descending suffix length in arch.json authoring)."""
        segs = [re.sub(r"_(\d+)$", r".\1", s) for s in key.split(".")]
        joined = ".".join(segs)
        for dsl_leaf, disk_leaf in renames.items():
            if joined == dsl_leaf:
                return disk_leaf
            if (head := joined[: -len(dsl_leaf) - 1] if joined.endswith("." + dsl_leaf) else None) is not None:
                return f"{head}.{disk_leaf}"
        return joined

    def disk_prefix(key, index):
        return _ckpt_key(dsl_to_disk(key), index, dec_prefix)

    # The GDN bundle: GatedDeltaNetLayer::load reads
    # `{prefix}.conv1d.weight` (capital-A) `A_log` / `dt_bias` /
    # `norm.weight` under the bundle's OWN disk prefix; the oracle leaf
    # is lowercase `a_log`. in_proj/out_proj siblings are ordinary
    # gemms under their own manifest keys.
    def is_gdn_bundle(key):
        return f"{key}[" in bundle_args and key.split(".")[-1] == "linear_attn"

    out = {}

    def put(key, tensor):
        # bf16 on disk — the gate's GpuWeights runs set_target_dtype(BF16).
        out[key] = tensor.detach().to(torch.bfloat16).contiguous()

    def put_f32(key, tensor):
        out[key] = tensor.detach().to(torch.float32).contiguous()

    for key, dims in normalized.items():
        if key in packed_targets:
            continue  # written fused into its packed parent below
        if key == "lm_head" and tie:
            # tie_word_embeddings: NO lm_head on disk — the macro reuses
            # embed_tokens (FieldLoad::LinearTiedToEmbedding).
            continue
        node = resolve(key)
        is_layered = key in layered_refs
        vals = list(node) if is_layered else [node]
        if is_gdn_bundle(key):
            for l, bundle in enumerate(vals):
                pre = disk_prefix(key, l if is_layered else None)
                put(f"{pre}.conv1d.weight", bundle.conv1d)
                put_f32(f"{pre}.A_log", bundle.a_log)
                put(f"{pre}.dt_bias", bundle.dt_bias)
                put_f32(f"{pre}.norm.weight", bundle.norm)
            continue
        # Dual-class per-layer leaves (gemma4): `weight_leaf_renames` maps
        # `q_proj_global` onto the SAME disk leaf as `q_proj`, but in a
        # real checkpoint each layer's leaf carries that LAYER's class
        # width. Writing both trees naively to the same disk key makes
        # the later put() clobber the earlier — the compiled side then
        # reads the wrong class's tensor while the oracle keeps its own
        # tree. Route each layer's disk write through the tree of THAT
        # layer's attention class: the renamed-to key's "base" name
        # (rename stripped) walks the class branch.
        class_shared = None
        for dsl_leaf, disk_leaf in renames.items():
            # The rename key is the GLOBAL-class name (e.g.
            # `self_attn.q_proj_global` → `self_attn.q_proj`); the shared
            # pair is (base key, f"{base}_global"). Either member of the
            # pair enters this arm.
            base_of_rename = (
                dsl_leaf[:-len("_global")] if dsl_leaf.endswith("_global")
                else dsl_leaf)
            if key == dsl_leaf or key == base_of_rename:
                class_shared = base_of_rename
                break
        if class_shared is not None:
            base_name = class_shared
            global_name = f"{base_name}_global"
            base_tree = trees
            for p in base_name.split("."):
                base_tree = getattr(base_tree, p)
            # The global-class tree may not exist in the manifest for
            # this arch (renames can be non-class, e.g. LocateAnything
            # `mm.proj_in`): fall back to whole-list base values.
            try:
                global_tree = trees
                for p in global_name.split("."):
                    global_tree = getattr(global_tree, p)
            except AttributeError:
                global_tree = base_tree
            base_list = list(base_tree) if is_layered else [base_tree]
            glob_list = list(global_tree) if is_layered else [global_tree]
            for l in range(len(vals)):
                is_class_l = (l % pattern == remainder) if pattern else False
                w = (glob_list[l] if is_class_l else base_list[l])
                pre = disk_prefix(key, l if is_layered else None)
                if hasattr(w, "bias") and isinstance(w.bias, torch.Tensor):
                    put(f"{pre}.bias", w.bias)
                if w.ndim == 2 and key not in embed_keys:
                    put(f"{pre}.weight", w.T)
                else:
                    put(f"{pre}.weight", w)
            continue
        for l, t in enumerate(vals):
            pre = disk_prefix(key, l if is_layered else None)
            # `dotted.sibling` keys resolve to sub-Trees via `resolve`;
            # a leaf whose parent chain hits a bundle marker merges into
            # the bundle Tree (synth_trees moved it there) — ordinary
            # attr walk handles both.
            if isinstance(t, Tree):
                continue  # non-GDN bundle markers are not gate phase-1
            w = t
            if hasattr(w, "bias") and isinstance(w.bias, torch.Tensor):
                # `.bias` sibling (synth_trees attached it, or the
                # manifest declared it via the dotted namespace).
                put(f"{pre}.bias", w.bias)
            if w.ndim == 2 and key not in embed_keys:
                # gemm [K, N] → on-disk Linear [N, K]
                put(f"{pre}.weight", w.T)
            elif w.ndim == 3 and key.split(".")[-1] == "conv1d":
                put(f"{pre}.weight", w)
            else:
                # Embedding [vocab, hidden] verbatim, norm gains [D],
                # moe stacks
                put(f"{pre}.weight", w)

    # Packed splits: fuse each target group ROW-WISE into
    # `{grandparent}.{parent}.weight` (weights.rs
    # synthesize_packed_row_split_sizes carves them back out in
    # listed order). Targets are written TRANSPOSED first, so the
    # packed rows are output rows. The parent is NOT a manifest key
    # (synth_trees never built it) — the fused tensor is assembled
    # from its targets here.
    for parent, targets in packed.items():
        is_layered = any(t in layered_refs for t in targets)
        l_indices = range(depth) if is_layered else [None]
        for l in l_indices:
            packed_rows = []
            for t in targets:
                w = resolve(t)
                w = w[l] if t in layered_refs else w
                packed_rows.append(w.T)
            fused = torch.cat(packed_rows, dim=0)
            pre = disk_prefix(parent, l)
            put(f"{pre}.weight", fused)
            # `.bias` siblings carve identically — fuse those too.
            bias_rows = []
            for t in targets:
                w = resolve(t)
                w = w[l] if t in layered_refs else w
                if hasattr(w, "bias") and isinstance(w.bias, torch.Tensor):
                    bias_rows.append(w.bias)
            if bias_rows and len(bias_rows) == len(targets):
                put(f"{pre}.bias", torch.cat(bias_rows, dim=0))

    # tie_word_embeddings: NO lm_head on disk — the macro reuses
    # embed_tokens (FieldLoad::LinearTiedToEmbedding). An untied
    # lm_head was already written above as an ordinary gemm.

    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(ckpt_dir / "model.safetensors"))
    # config.json: the shrunk config VERBATIM — it is the config the
    # tiny stem was checked in as, so the gate's fingerprint (embed
    # shape, layer count, theta) matches the compiled variant.
    (ckpt_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    return ckpt_dir


def pos_embed_keys_of(carrier_text):
    """Manifest keys passed as `pos_embed(position_ids, <key>)` —
    learned lookup tables, verbatim [N, D] on disk (NOT transposed;
    the macro's pos-embed matcher reads them like embeddings)."""
    return {m.group(1) for m in re.finditer(
        r"pos_embed\(\s*position_ids\s*,\s*([A-Za-z_][A-Za-z_0-9.]*)\s*\)", carrier_text)}


def write_vision_checkpoint(out_dir, arch, cfg, manifest, bounds, trees, carrier_text):
    """Vision-tower counterpart of write_checkpoint: same manifest →
    HF-layout safetensors, but through the VISION path rules
    (codegen.rs safetensors_prefix's is_vision arm):
      - layered keys → `<default_root>.<layered_subpath>.{l}.<leaf>`
      - unlayered keys → `<default_root>.<leaf>`
      - a first segment mapped in `subtrees` replaces the whole prefix
        and is unindexed (e.g. `mm.*` → `multi_modal_projector.*`)
      - digit-suffix translation + weight_leaf_renames, applied BEFORE
        subtree resolution (same order as codegen)
    The patch-embed conv weight ships RANK-2 `[E, in_features]` with
    the k-axis in the pixel pack's (c, t, ph, pw) order — the
    post-flatten form every `try_load_mm` sniff converges to (the
    emitted flatten only fires at rank > 2). `raw_linear` manifest
    kinds (gemma3-mm's matmul-natural `nn.Parameter`) and
    `pos_embed()` lookup tables are written VERBATIM, no transpose.
    """
    from safetensors.torch import save_file

    aj = json.loads((ARCH_DIR / arch / "arch.json").read_text())
    layout = aj["vision_safetensors_layout"]
    renames = aj.get("weight_leaf_renames") or {}
    packed = manifest.get("__packed_splits__") or {}
    packed_targets = {t for targets in packed.values() for t in targets}

    normalized = {}
    kinds = {}
    for key, dims in manifest.items():
        if key.startswith("__"):
            continue
        if isinstance(dims, dict):
            kinds[key] = dims.get("kind")
            dims = dims["shape"]
        normalized[key] = dims

    layered_refs = {m.group(1) for m in re.finditer(
        r"([A-Za-z_][A-Za-z_0-9.]*)\[", carrier_text)}
    embed_keys = embed_keys_of(carrier_text)
    posembed_keys = pos_embed_keys_of(carrier_text)

    def resolve(key):
        node = trees
        for p in key.split("."):
            node = getattr(node, p)
        return node

    def dsl_to_disk(key):
        segs = [re.sub(r"_(\d+)$", r".\1", s) for s in key.split(".")]
        joined = ".".join(segs)
        for dsl_leaf, disk_leaf in renames.items():
            if joined == dsl_leaf:
                return disk_leaf
            if (head := joined[: -len(dsl_leaf) - 1] if joined.endswith("." + dsl_leaf) else None) is not None:
                return f"{head}.{disk_leaf}"
        return joined

    def disk_prefix(key, index):
        joined = dsl_to_disk(key)
        segs = joined.split(".")
        if segs[0] in layout.get("subtrees", {}):
            rest = segs[1:]
            disk = layout["subtrees"][segs[0]]
            return disk if not rest else f"{disk}.{'.'.join(rest)}"
        if index is not None:
            return f"{layout['default_root']}.{layout['layered_subpath']}.{index}.{joined}"
        return f"{layout['default_root']}.{joined}"

    out = {}

    def put(key, tensor):
        out[key] = tensor.detach().to(torch.bfloat16).contiguous()

    for key, dims in normalized.items():
        if key in packed_targets:
            continue
        node = resolve(key)
        is_layered = key in layered_refs
        vals = list(node) if is_layered else [node]
        for l, w in enumerate(vals):
            pre = disk_prefix(key, l if is_layered else None)
            if hasattr(w, "bias") and isinstance(w.bias, torch.Tensor):
                put(f"{pre}.bias", w.bias)
            if kinds.get(key) == "raw_linear":
                # nn.Parameter: `load_raw` reads the disk key VERBATIM —
                # no `.weight` suffix — matmul-natural, no transpose.
                put(pre, w)
            elif w.ndim == 2 and key not in embed_keys \
                    and key not in posembed_keys:
                put(f"{pre}.weight", w.T)
            else:
                put(f"{pre}.weight", w)

    # Learned pos-embed table (arches declaring `vision_pos_embed_key`):
    # not part of the manifest — the wrapper captures it host-side via
    # `gw.tensor_to_f32(key)`. Written VERBATIM (bf16 on disk, exactly
    # what `put` does) under the declared disk key.
    pe_table = STATE.get("vision_pos_table")
    if pe_table is not None:
        key, table, _ng = pe_table
        assert key not in out, f"pos-embed table {key} collides with a manifest entry"
        out[key] = table.detach().to(torch.bfloat16).contiguous()

    # Packed splits (attn.qkv / attn.wqkv): fuse target groups ROW-WISE
    # into `{prefix}.{parent}.weight` + `.bias` — the loader's
    # synthesize_packed_row_split_sizes carves them back out.
    for parent, targets in packed.items():
        is_layered = any(t in layered_refs for t in targets)
        l_indices = range(len(list(resolve(targets[0])))) if is_layered else [None]
        for l in l_indices:
            packed_rows = []
            for t in targets:
                w = resolve(t)
                w = w[l] if t in layered_refs else w
                packed_rows.append(w.T)
            put(f"{disk_prefix(parent, l)}.weight", torch.cat(packed_rows, dim=0))
            bias_rows = []
            for t in targets:
                w = resolve(t)
                w = w[l] if t in layered_refs else w
                if hasattr(w, "bias") and isinstance(w.bias, torch.Tensor):
                    bias_rows.append(w.bias)
            if bias_rows and len(bias_rows) == len(targets):
                put(f"{disk_prefix(parent, l)}.bias", torch.cat(bias_rows, dim=0))

    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(ckpt_dir / "model.safetensors"))
    (ckpt_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    return ckpt_dir


def forward(fn):
    """`@forward` under torch is the identity — the decorator carries
    compile-time metadata only."""
    return fn


def vision_forward(*, workloads=None, sk_buckets=None, processor=None, pixel_pack=None):
    def deco(fn):
        return fn
    return deco


# ───────────────────────── main ─────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--num-tokens", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()

    arch, stem = args.arch, args.stem
    out_dir = args.out or (Path(__file__).parent / "goldens" / f"{arch}-{stem}")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = args.num_tokens

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    # Vision classification is the CARRIER DECORATOR (parse_python.rs
    # `PythonCarrier::vision`): `@vision_forward` → vision prelude,
    # `@forward` → decoder. NOT config-based — qwen3-5 is a @forward
    # text decoder whose configs nest a `vision_config` for the VL
    # sibling, and the macro compiles those configs through the
    # decoder path (text_config hoisted, vision_config ignored).
    # Read BEFORE the shrink: the ForConditionalGeneration rewrite
    # below must not fire for vision arches — the MM registry
    # (`ScratchyMmRegistration`) claims the CondGen string verbatim.
    carrier = (DSL_DIR / f"{arch}.py").read_text()
    is_vision = re.search(r"@vision_forward\s*(\(|\n)", carrier) is not None

    cfg = load_config(arch, stem)
    if args.tiny:
        # Derived keys the derivation below recomputes: drop them before
        # shrinking so a shrunk input (e.g. qk_nope_head_dim) doesn't pair
        # with a stale explicit product (q_proj_out 3072). config.rs never
        # recomputes an explicit key — but a shrunk config is OUR artifact,
        # and consistency of the synthetic model is what matters.
        REDERIVABLE = {
            "head_dim", "q_proj_out", "kv_lora_out", "attn_out", "kv_a_proj_out",
            "num_key_value_heads", "sliding_window_global_remainder",
            "attn_q_dim", "q_gate_dim", "kv_dim",
            "gdn_value_dim", "gdn_conv_dim", "global_head_dim",
            "num_global_key_value_heads", "q_global_dim", "k_global_dim",
            # NOT sliding_window_pattern: derive_bounds computes it only
            # when absent, and it must stay consistent with layer_types
            # (which we truncate below) — so let the derive recompute it
            # by popping... it can't: derive skips explicit keys. Instead
            # we recompute it from the truncated layer_types below.
        }
        # `global_head_dim` is REDERIVABLE because uniform arches derive it
        # = head_dim — but a DISTINCT explicit value (gemma4: 512 vs 256) IS
        # the dual-class geometry. Popping it would let the derive collapse
        # it onto the shrunk head_dim, fusing the two attention classes
        # (codegen's rope guard then reads uniform geometry and refuses:
        # GLOBAL_ROT_DIM != ROT_DIM). Capture it before the pop and keep it
        # explicit so the TINY_BOUNDS shrink below can pin it at 2× head_dim,
        # preserving the class relation; uniform arches (value == head_dim)
        # still pop.
        ghd_full, hd_full = cfg.get("global_head_dim"), cfg.get("head_dim")
        for k in REDERIVABLE:
            cfg.pop(k, None)
        if isinstance(ghd_full, int) and isinstance(hd_full, int) \
                and ghd_full != hd_full:
            cfg["global_head_dim"] = ghd_full
        for k, v in TINY_BOUNDS.items():
            if (cur := json_path(cfg, k)) is not None and isinstance(cur, int):
                set_json_path(cfg, k, v)
        # layer_types: keep 2 entries, sliding then the first differing
        # type, so the (sliding | global) alternation survives the shrink
        # and sliding_window_pattern = 2 matches the truncated list.
        if isinstance(cfg.get("layer_types"), list):
            lt = cfg["layer_types"]
            first = lt[0]
            other = next((t for t in lt if t != first), None)
            if other is None:
                cfg["layer_types"] = [first, first]
                cfg.pop("sliding_window_pattern", None)  # uniform: no pattern
            else:
                cfg["layer_types"] = [first, other]
                cfg["sliding_window_pattern"] = 2
        # `sliding_window_pattern` WITHOUT `layer_types` (gemma2/gemma3):
        # the pattern indexes layer position mod pattern to pick the
        # sliding vs global attention class. A 2-layer shrink under a
        # pattern > 2 lands both layers on the SAME class — for gemma3
        # (pattern 6, all-sliding) that erases the global-rope ops from
        # one bucket but not the other, and the emitted Weights struct
        # and its accessor disagree (no field `rotary`). Clamp the
        # pattern to 2 so the shrink keeps exactly one layer of each
        # class — the smallest geometry that still exercises both.
        if not isinstance(cfg.get("layer_types"), list) \
                and isinstance(cfg.get("sliding_window_pattern"), int) \
                and cfg["sliding_window_pattern"] > 2:
            cfg["sliding_window_pattern"] = 2
        # LongRoPE per-channel factors (`rope_scaling.short_factor` /
        # `long_factor`, phi3): one entry per rotary PAIR at full scale.
        # After the shrink the pair count is smaller, so interpolate
        # the table down to the new count — try_load's LongRope loader
        # refuses a length mismatch (short=48, expected 16).
        # Identity preservation: a config field EQUAL to a product of
        # others at full scale (deepseek-v2's hidden_size == heads·
        # v_head_dim — its o_proj manifest formula reads [hidden_size,
        # hidden_size] where the attention output is heads·v_head_dim)
        # must still hold after the shrink, or the manifest formulas
        # stop describing the model. Re-impose by overwriting the
        # product field with the shrunk factors' product.
        full = derive_bounds(dict(load_config(arch, stem)))
        shrunk = derive_bounds(dict(cfg))
        for target, factors in (("hidden_size", ("num_attention_heads", "v_head_dim")),):
            if target in full and all(f in full for f in factors) \
                    and full[target] == math.prod(full[f] for f in factors):
                cfg[target] = math.prod(shrunk[f] for f in factors)
        rs = cfg.get("rope_scaling")
        if isinstance(rs, dict):
            partial = json_path(cfg, "rope_parameters.partial_rotary_factor") \
                or rs.get("partial_rotary_factor")
            hd = shrunk.get("head_dim", 0)
            rot_dim = round(partial * hd) if isinstance(partial, (int, float)) \
                and abs(partial - 1.0) > 1e-9 else hd
            pair = rot_dim // 2
            for key in ("short_factor", "long_factor"):
                tbl = rs.get(key)
                if isinstance(tbl, list) and len(tbl) != pair:
                    rs[key] = [
                        tbl[min(len(tbl) - 1, round(i * (len(tbl) - 1) / max(pair - 1, 1)))]
                        for i in range(pair)
                    ]
        # Text execution of a CondGen-wrapped config (gemma3: the oracle
        # runs dsl/gemma3.py — @forward — over the mm wrapper's config):
        # the compiled side derives its safetensors key prefix from the
        # arch string + arch.json's decoder_safetensors_prefix. A
        # ForConditionalGeneration string without a declared prefix makes
        # the baked fingerprint expect `model.language_model.*` while the
        # writer emits `model.*` — a silent fingerprint miss. Arch strings
        # with a declared prefix (qwen3-5: "language_model") stay as-is;
        # both sides nest consistently. Vision arches (@vision_forward)
        # keep the CondGen string — the MM registry claims it verbatim
        # and the vision loader keys only off the tower's visual.* leaves.
        archs = cfg.get("architectures")
        aj = json.loads((ARCH_DIR / arch / "arch.json").read_text()) \
            if (ARCH_DIR / arch / "arch.json").exists() else {}
        if not is_vision \
                and isinstance(archs, list) and len(archs) == 1 \
                and isinstance(archs[0], str) \
                and archs[0].endswith("ForConditionalGeneration") \
                and not aj.get("decoder_safetensors_prefix"):
            cfg["architectures"] = [archs[0].replace(
                "ForConditionalGeneration", "ForCausalLM")]
        # Materialize the popped shrink-target keys into the config we
        # WRITE — the ones that can also live nested in a `text_config`
        # (head_dim, num_key_value_heads, num_global_key_value_heads).
        # Without a top-level entry, scratchy's text_config hoist
        # (or_insert — nested values win only when top level is absent)
        # leaks the FULL-SCALE dims from a stale nested text_config
        # (gemma3: num_key_value_heads 8, head_dim 256 under a shrunk
        # 4×32 top level — NoHeadGrouping at compile time). ONLY the
        # shrink targets, never the other derived bounds (attn_q_dim,
        # q_gate_dim, sliding_window_global_remainder, …): config.rs
        # never recomputes an explicit key, so pinning a derived bound
        # changes downstream op classification (gemma3's m=2 tape lost
        # its global-rope ops while the accessor kept `self.rotary`).
        for k in REDERIVABLE:
            if k in TINY_BOUNDS and k in shrunk:
                cfg[k] = shrunk[k]
        # gemma4's dual-class geometry: the global q/kv projection widths
        # (`q_global_dim` / `k_global_dim`) are derived (config.rs) as
        # num_attention_heads × global_head_dim / num_global_key_value_heads
        # × global_head_dim. The shrink keeps global_head_dim at 2× head_dim
        # (64 vs 32, the full-scale class relation), so the derived global
        # q width (4×64) stays distinct from the sliding one (4×32). The
        # global KV head count also collapses if left at num_attention_heads
        # — shrink it to 1 (the real gemma4 relation: 1 global kv head vs 8
        # sliding) so the class geometry stays distinct through the shrink.
        if "global_head_dim" in cfg and isinstance(cfg.get("num_attention_heads"), int):
            gkv = cfg.get("num_global_key_value_heads")
            if isinstance(gkv, int) and gkv == cfg["num_attention_heads"]:
                cfg["num_global_key_value_heads"] = 1
        # mrope_section: bands cover the ROTARY half (rotary_dim/2), so
        # a shrunk head_dim/partial factor needs the section rescaled to
        # the new pair count — the macro's expansion guard (codegen.rs
        # mrope_section_tokens) refuses a config whose section sums to
        # anything else. Text decode is unaffected (all bands share the
        # same 1D positions); only the written config must be consistent.
        partial = cfg.get("partial_rotary_factor")
        if not isinstance(partial, (int, float)):
            partial = json_path(cfg, "rope_parameters.partial_rotary_factor")
        hd = shrunk.get("head_dim", 0)
        rot_dim = hd
        if isinstance(partial, (int, float)) and abs(partial - 1.0) > 1e-9:
            rot_dim = round(partial * hd)
        pair = rot_dim // 2
        for dotted in ("rope_parameters.mrope_section", "rope_scaling.mrope_section"):
            sec = json_path(cfg, dotted)
            if isinstance(sec, list) and len(sec) == 3 and sum(sec) != pair:
                old = sum(sec)
                new = [max(1, round(s * pair / old)) for s in sec]
                # repair the sum on the largest band
                i = new.index(max(new))
                new[i] += pair - sum(new)
                set_json_path(cfg, dotted, new)
    bounds = derive_bounds(cfg)
    # Vision arches: arch.json `params` table derives the vision_* / d_model
    # bounds (config.rs eval_params — additive over the flat harvest).
    eval_params(arch, cfg, bounds)
    manifest = json.loads((ARCH_DIR / arch / "weights.json").read_text())

    # ── STATE the shims read ──
    STATE["bounds"] = bounds
    STATE["cfg"] = cfg
    STATE["arch"] = arch
    STATE["rms_eps"] = float(cfg.get("rms_norm_eps", cfg.get("layer_norm_eps", 1e-6)))
    # Vision block-norm eps (config.rs: explicit flat key → arch-declared
    # → 1e-6) — the emitted loader bakes it into every vision LayerNorm.
    if is_vision:
        aj_eps = json.loads((ARCH_DIR / arch / "arch.json").read_text()) \
            .get("vision_norm_eps") if (ARCH_DIR / arch / "arch.json").exists() else None
        STATE["rms_eps"] = float(
            cfg.get("vision_norm_eps", aj_eps if aj_eps is not None else 1e-6))
    STATE["final_logit_softcapping"] = float(cfg.get("final_logit_softcapping", 0.0) or 30.0)
    # scalar() binding: config.rs extract_scalars — EVERY numeric config
    # field, by name, plus arch scalar_defaults (already merged into cfg).
    STATE["scalars"] = {k: float(v) for k, v in cfg.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    # Gemma-style zero-centered gains: effective gain = 1 + weight.
    STATE["norm_gain_offset"] = 1.0 if bounds.get("rms_norm_zero_centered") else 0.0

    depth = bounds.get("num_hidden_layers", bounds.get("vision_depth", 2))
    kv_heads = bounds.get("num_key_value_heads", bounds.get("num_attention_heads", 1))
    head_dim = bounds.get("head_dim", 0)

    # kv pools: per-layer (k, v) zeros [NUM_BLOCKS, kv_heads, BLOCK_SIZE, head_dim].
    # gemma4's global layers carry their own (kv heads, head_dim) — the
    # per-layer class split (pattern/remainder) decides which geometry
    # each layer's pool takes.
    STATE["kv_cache"] = []
    g_kv = bounds.get("num_global_key_value_heads", kv_heads)
    g_hd = bounds.get("global_head_dim", head_dim)
    pattern = bounds.get("sliding_window_pattern", 0)
    remainder = bounds.get("sliding_window_global_remainder", pattern - 1 if pattern else 0)
    for l in range(depth):
        is_global = pattern > 0 and l % pattern == remainder
        l_kv, l_hd = (g_kv, g_hd) if is_global else (kv_heads, head_dim)
        STATE["kv_cache"].append((
            torch.zeros(NUM_BLOCKS, l_kv, BLOCK_SIZE, l_hd),
            torch.zeros(NUM_BLOCKS, l_kv, BLOCK_SIZE, l_hd),
        ))
    STATE["layer_of"] = {id(c): i for i, c in enumerate(STATE["kv_cache"])}

    rot_cfg = resolve_rotary_config(cfg, bounds)
    rot_objs = {}
    max_pos = bounds.get("max_position_embeddings", 64)
    # Which layers run the GLOBAL class (the rotary/head geometry split
    # gemma4's dual-class attention creates): the pattern/remainder the
    # carrier itself branches on. The GLOBAL rotary object pairs with the
    # GLOBAL kv head count; `rotary_local` (sliding class) with the base.
    g_kv = bounds.get("num_global_key_value_heads", kv_heads)
    for name, spec in rot_cfg.items():
        tables = build_rotary(spec[0], spec[1], min(spec[2], max_pos), spec[3], spec[4])
        is_global_rot = name == "rotary" and (
            bounds.get("global_partial_rotary_factor") is not None
            or bounds.get("global_head_dim", spec[0]) != spec[0]
            or g_kv != kv_heads)
        rot_objs[name] = {"spec": spec, "tables": tables,
                          "q_heads": bounds.get("num_attention_heads", 1),
                          "kv_heads": g_kv if is_global_rot else kv_heads}
    if not rot_objs:  # vision or GDN-only arch — rotary unused
        rot_objs["rotary"] = {"spec": (head_dim, head_dim, 8, 10000.0, False),
                              "tables": build_rotary(head_dim or 8, head_dim or 8, 8, 10000.0),
                              "q_heads": 1, "kv_heads": 1}
    STATE["rotary_of"] = {id(v): v for v in rot_objs.values()}

    STATE["attn"] = {}
    for layer in range(depth):
        # gemma4's per-layer class split (mirrors `layer_types` /
        # sliding_window_pattern): global layers run global_head_dim with
        # num_global_key_value_heads; sliding layers the base geometry.
        is_global = False
        pattern = bounds.get("sliding_window_pattern", 0)
        if pattern > 0:
            remainder = bounds.get("sliding_window_global_remainder", pattern - 1)
            is_global = layer % pattern == remainder
        g_kv = bounds.get("num_global_key_value_heads", kv_heads)
        g_hd = bounds.get("global_head_dim", head_dim)
        if is_global and (g_kv != kv_heads or g_hd != head_dim):
            STATE["attn"][layer] = {
                "q_heads": bounds.get("num_attention_heads", 1),
                "kv_heads": g_kv,
                "head_dim": g_hd,
                "scale": attn_scale_for(cfg, bounds, g_hd),
                "window": bounds.get("sliding_window", 0),
            }
        else:
            STATE["attn"][layer] = {
                "q_heads": bounds.get("num_attention_heads", 1),
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "scale": attn_scale_for(cfg, bounds, head_dim),
                "window": bounds.get("sliding_window", 0),
            }
    STATE["mla_theta"] = float(cfg.get("rope_theta", 10000.0))
    STATE["mla_scale"] = 1.0 / math.sqrt(bounds.get("qk_nope_head_dim", 1) + bounds.get("qk_rope_head_dim", 1)) \
        if "qk_nope_head_dim" in bounds else 0.0

    # inputs — identity layout (the qwen3 gate's convention)
    STATE["input_ids"] = torch.from_numpy(rng.integers(low=1, high=max(2, min(1000, bounds.get("vocab_size", 1000))), size=n).astype(np.int64))
    STATE["positions"] = list(range(n))
    STATE["slot_mapping"] = list(range(n))
    n_blocks_needed = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    STATE["block_table"] = [i for i in range(n_blocks_needed)]

    trees = synth_trees(manifest, bounds, depth, carrier)
    # tie_word_embeddings: HF ties lm_head to the embedding TABLE — the
    # compiled side reuses the embed tensor (LinearTiedToEmbedding; the
    # metal gemm computes x·W^T against the [vocab, hidden] table), so
    # the oracle must consume the SAME tensor or the golden logits are
    # computed against weights the tape never sees. The gemm shim takes
    # [K, N], so the tied operand is the table transposed.
    if is_tied(arch, cfg) and "lm_head" in manifest:
        node = trees
        for p in sorted(embed_keys_of(carrier))[0].split("."):
            node = getattr(node, p)
        trees.lm_head = node.T

    # ── ambient env the carrier body reads ──
    env = {k: v for k, v in globals().items() if not k.startswith("__")}
    env.update({
        "torch": torch, "math": math,
        # `crate.PROCESSOR` appears in vision_forward decorator kwargs —
        # the identity decorator never touches it; a bare object suffices.
        "crate": type("crate", (), {"PROCESSOR": None})(),
        "input_ids": STATE["input_ids"],
        "positions": STATE["positions"],
        "num_tokens": n,
        "block_table": STATE["block_table"],
        "kv_cache": STATE["kv_cache"],
        **{name: obj for name, obj in rot_objs.items()},
        **{k: v for k, v in vars(trees).items()},
        # bounds by NAME — loop bounds and predicates
        **bounds,
        # numeric config fields by NAME — `scalar(<config key>)` passes the
        # key as a bare identifier (Expr::ConfigScalar), so the name must
        # resolve here (commandr logit_scale, granite embedding_multiplier).
        # Bounds WIN on collision (loop bounds must stay int).
        **{k: v for k, v in STATE["scalars"].items() if k not in bounds},
        # vision ambient names (harmless when unused)
        "cos": None, "sin": None, "freqs": None,
        "cu_seqlens": [0, n], "max_seqlen": n,
        "cu_seqlens_full": [0, n], "max_seqlen_full": n,
        "cu_seqlens_window": [0, n], "max_seqlen_window": n,
        "window_index": list(range(n)),
        "reverse_indices": list(range(n)),
        "position_ids": list(range(n)),
        "pos_embeds": torch.zeros(1),
        "pixels": torch.zeros(1),
    })

    if is_vision:
        setup_vision_state(env, cfg, bounds, n, rng)
        # The carrier reads num_tokens for the merger reshape — the
        # PATCH count, which setup_vision_state just fixed (grid may
        # differ from --num-tokens under a patch_grid_side bound).
        n = env["num_tokens"]

    # ── EXECUTE THE CARRIER AS PYTHON ──
    exec(compile(carrier, f"{arch}.py", "exec"), env)
    fn_name = arch.replace("-", "_")
    logits = env[fn_name]()

    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)
    assert torch.isfinite(logits).all(), "logits contain NaN/Inf"

    manifest_out = {}

    def dump(name, arr):
        a = arr.detach().contiguous().cpu() if isinstance(arr, torch.Tensor) else torch.as_tensor(arr)
        a = a.float().numpy()  # bf16 has no numpy dtype
        a.tofile(out_dir / f"{name}.bin")
        manifest_out[name] = {"shape": list(a.shape), "dtype": str(a.dtype)}

    dump("input_ids", STATE["input_ids"].to(torch.int64))
    dump("positions", np.asarray(STATE["positions"], dtype=np.int64))
    dump("logits", logits.float())
    dump("block_table", np.asarray(STATE["block_table"] + [0] * (NUM_BLOCKS - len(STATE["block_table"])), dtype=np.int64))
    dump("slot_mapping", np.asarray(STATE["slot_mapping"], dtype=np.int64))
    if is_vision:
        dump("cos", STATE["vision_cos"])
        dump("sin", STATE["vision_sin"])
        dump("freqs", STATE["vision_freqs"])
        dump("cu_seqlens", np.asarray(STATE["vision_cu"], dtype=np.int64))
        dump("pixels", STATE["vision_pixels"])
        # The RAW image (CHW f32) + geometry — the gate feeds this to the
        # production `MultimodalForward::vision_forward`, whose own pixel
        # pack / rope build must reproduce the tables above.
        dump("image", STATE["vision_image"])

    # harness facts the Rust gate needs (no per-arch code on that side)
    harness = {
        "arch": arch, "stem": stem, "num_tokens": n,
        "bounds": {k: bounds[k] for k in sorted(bounds)},
        "rms_eps": STATE["rms_eps"],
        "attn_scale": STATE["attn"][0]["scale"] if STATE["attn"] else 0.0,
        "kv_heads": kv_heads, "head_dim": head_dim, "layers": depth,
        "vision": is_vision,
        "rotary": {k: v["spec"] for k, v in rot_objs.items()},
    }
    if is_vision:
        v = {k[7:]: val for k, val in bounds.items() if k.startswith("vision_")}
        merge = v.get("spatial_merge_size", 1)
        gt, gh, gw = STATE["vision_grid"]
        harness["vision_grid"] = {
            "t": int(gt), "h": int(gh), "w": int(gw),
            "patch_size": int(v.get("patch_size", 14)),
            "spatial_merge_size": int(merge),
            "temporal_patch_size": int(v.get("temporal_patch_size", 1)),
            "in_chans": int(v.get("in_chans", 3)),
            "pool_factor": int(v.get("pool_factor", 1)),
            # output rows after merge (× pool) — the logits row count
            "n_merged": int(gt * gh * gw // max(1, merge * merge)
                            // max(1, v.get("pool_factor", 1))),
        }
    (out_dir / "harness.json").write_text(json.dumps(harness, indent=2))
    (out_dir / "goldens.json").write_text(json.dumps(manifest_out, indent=2))
    if is_vision:
        ckpt = write_vision_checkpoint(
            out_dir, arch, cfg, manifest, bounds, trees, carrier)
        print(f"[{arch}/{stem}] vision checkpoint → {ckpt}")
    else:
        ckpt = write_checkpoint(
            out_dir, arch, cfg, manifest, bounds, trees, depth, carrier)
        print(f"[{arch}/{stem}] checkpoint → {ckpt}")
    print(f"[{arch}/{stem}] logits {tuple(logits.shape)} → {out_dir}")


def setup_vision_state(env, cfg, bounds, n, rng):
    """Vision ambient tensors, TRANSCRIBING the production
    preprocessing (VisionWrapper::vision_forward + scratchy-vision):
    a raw normalized CHW image → patches_from_normalized_chw's exact
    (gh, gw, mh, mw, ci, ti, ph, pw) pack with the bf16 round-trip,
    per-token 2D-rope freqs/cos/sin from build_rope_freqs_f32's
    block-grouped (hpos, wpos) position logic, one cu_seqlens segment
    per image. The gate runs the production path over the same raw
    image (image.bin + grid facts in harness.json), so any divergence
    between this transcription and the Rust builders IS the finding.

    Geometry: the tower's own patch/merge size, one square grid that
    yields num_tokens patch rows (`--num-tokens` must be a perfect
    square; the grid side must be divisible by spatial_merge_size)."""
    v = {k[7:]: val for k, val in bounds.items() if k.startswith("vision_")}
    in_chans = v.get("in_chans", 3)
    patch = v.get("patch_size", 14)
    temporal = v.get("temporal_patch_size", 1)
    merge = v.get("spatial_merge_size", 1)
    merge_factor = v.get("merge_factor", merge * merge)
    embed_dim = v.get("embed_dim", v.get("hidden_size", 64))
    num_heads = v.get("num_heads", v.get("num_attention_heads", 1))
    head_dim = v.get("head_dim", embed_dim // max(1, num_heads))
    STATE["vision_head_dim"] = head_dim

    # Rope style: arch.json `vision_rope_style` ("interleaved_xy" for
    # LocateAnything; NeoxHw default — vision_glue.rs rope_style_tokens).
    aj = json.loads((ARCH_DIR / STATE["arch"] / "arch.json").read_text()) \
        if (ARCH_DIR / STATE["arch"] / "arch.json").exists() else {}
    interleaved = aj.get("vision_rope_style") == "interleaved_xy"
    STATE["vision_interleaved"] = interleaved

    in_features = v.get("in_features", in_chans * temporal * patch * patch)

    # Grid: one square image, grid_side² patches == n. Gemma3-mm's
    # SigLIP (merge 1) declares patch_grid_side; rotary towers merge
    # S² patches per token row but the ENCODER runs pre-merge over all
    # grid_side² patches — L here is the pre-merge patch count.
    if v.get("patch_grid_side"):
        g = v["patch_grid_side"]
    else:
        r = math.isqrt(n)
        assert r * r == n, (
            f"--num-tokens {n} is not a perfect square — the vision "
            f"grid is square (grid_side² patches)")
        g = r
    n_tokens = g * g
    # The S-divisibility asserts patches_from_normalized_chw enforces.
    assert g % merge == 0, (
        f"grid side {g} not divisible by spatial_merge_size {merge}")
    height = width = g * patch

    # ── raw image + production patch packing ──
    image = torch.from_numpy(rng.standard_normal(
        (in_chans, height, width)).astype(np.float32))
    STATE["vision_image"] = image
    STATE["vision_grid"] = (temporal, g, g)  # grid_thw (t, grid_h, grid_w)
    p, s, t, c = patch, merge, temporal, in_chans
    g_h, g_w = g // s, g // s
    feat = c * t * p * p
    patches = torch.zeros(n_tokens * feat)
    img_flat = image.flatten()
    out_idx = 0
    for gh in range(g_h):
        for gw in range(g_w):
            for mh in range(s):
                for mw in range(s):
                    for ci in range(c):
                        for _ti in range(t):
                            img_h_base = gh * (s * p) + mh * p
                            img_w_base = gw * (s * p) + mw * p
                            for ph in range(p):
                                img_h = img_h_base + ph
                                row = ci * (height * width) + img_h * width + img_w_base
                                for pw in range(p):
                                    patches[out_idx] = img_flat[row + pw]
                                    out_idx += 1
    assert out_idx == n_tokens * feat
    patches = patches.view(n_tokens, feat)
    # bf16 round-trip (the wrapper uploads bf16 patches — the oracle
    # must consume the SAME rounding the production path does).
    pixels = patches.to(torch.bfloat16).to(torch.float32)
    STATE["vision_pixels"] = pixels
    env["pixels"] = pixels
    env["num_tokens"] = n_tokens

    # ── 2D rope tables (build_rope_freqs_f32 / build_rope_cos_sin_bf16) ──
    # half = head_dim/2 angles per token; freq_axis = half/2; the
    # first freq_axis angles rotate on the H position, the rest on W
    # (NeoxHw; InterleavedXy alternates x=column, y=row per freq).
    half = head_dim // 2
    freq_axis_dim = half // 2
    inv_freq = [1.0 / (10000.0 ** ((2 * i) / half)) for i in range(freq_axis_dim)]
    # hpos/wpos in the S²-block-grouped order the Rust builder uses.
    frame_len = g * g
    hpos = [0] * frame_len
    wpos = [0] * frame_len
    idx = 0
    for hb in range(g // s):
        for wb in range(g // s):
            for sh in range(s):
                for sw in range(s):
                    hpos[idx] = hb * s + sh
                    wpos[idx] = wb * s + sw
                    idx += 1
    freqs_rows = []
    for token in range(frame_len):
        hp, wp = float(hpos[token]), float(wpos[token])
        row = []
        for f in inv_freq:
            if interleaved:
                # InterleavedXy: x (column) then y, per freq.
                row.append(wp * f)
                row.append(hp * f)
            else:
                # NeoxHw: all H angles, then all W.
                row.append(hp * f)
        if not interleaved:
            for f in inv_freq:
                row.append(wp * f)
        freqs_rows.append(row)
    freqs = torch.tensor(freqs_rows, dtype=torch.float32)
    STATE["vision_freqs"] = freqs
    # cos/sin: computed on the f32 angle, RESULT rounded to bf16
    # (f32_to_bf16 in build_rope_cos_sin_bf16 — only the CUDA path
    # reads these; the metal rope kernel reads the raw freqs).
    STATE["vision_cos"] = freqs.cos().to(torch.bfloat16).to(torch.float32)
    STATE["vision_sin"] = freqs.sin().to(torch.bfloat16).to(torch.float32)
    env["cos"] = STATE["vision_cos"]
    env["sin"] = STATE["vision_sin"]
    env["freqs"] = freqs

    # ── Learned pos-embed table (qwen3-5-vl / locateanything) —
    # transcribes the wrapper's host-side interpolation
    # (fast_pos_embed_interpolate bilinear / bicubic_...): the table
    # is captured f32 from the checkpoint (bf16 on disk → f32 at
    # read), interpolated to this grid in merge order, and uploaded
    # bf16 as the `pos_embeds` extern. ng is a SMALLER grid than the
    # real checkpoints' (48 / 64) so the interpolation genuinely
    # resamples. ──
    pe_key = aj.get("vision_pos_embed_key")
    if pe_key is not None:
        ng = 8
        table = torch.from_numpy(
            rng.standard_normal((ng * ng, embed_dim)).astype(np.float32)) * INIT_SCALE
        # production reads the on-disk bf16 tensor back as f32
        table = table.to(torch.bfloat16).to(torch.float32)
        STATE["vision_pos_table"] = (pe_key, table, ng)

        def lin(i, n):
            return 0.0 if n <= 1 else i * (ng - 1) / (n - 1)

        def bilinear_frame():
            frame = []
            for hb in range(g // s):
                for wb in range(g // s):
                    for sh in range(s):
                        for sw in range(s):
                            row, col = hb * s + sh, wb * s + sw
                            hf, wf = lin(row, g), lin(col, g)
                            h_floor, w_floor = int(hf), int(wf)
                            h_ceil = min(h_floor + 1, ng - 1)
                            w_ceil = min(w_floor + 1, ng - 1)
                            dh, dw = hf - h_floor, wf - w_floor
                            w00 = (1 - dh) * (1 - dw); w01 = (1 - dh) * dw
                            w10 = dh * (1 - dw); w11 = dh * dw
                            t00 = table[h_floor * ng + w_floor]
                            t01 = table[h_floor * ng + w_ceil]
                            t10 = table[h_ceil * ng + w_floor]
                            t11 = table[h_ceil * ng + w_ceil]
                            frame.append(w00 * t00 + w01 * t01 + w10 * t10 + w11 * t11)
            return torch.stack(frame)  # [g*g, e] merge order

        def bicubic_frame():
            def cubic(t):
                a = -0.75
                t = abs(t)
                if t <= 1.0:
                    return (a + 2) * t**3 - (a + 3) * t**2 + 1
                if t < 2.0:
                    return a * (t**3 - 5 * t**2 + 8 * t - 4)
                return 0.0

            def src_window(dst, out_n):
                src = (dst + 0.5) * ng / out_n - 0.5
                start = max(math.floor(src - 2.0) + 1, 0)
                end = min(max(math.floor(src + 2.0) + 1, 0), ng)
                return src, start, end

            frame = []
            for hb in range(g // s):
                for wb in range(g // s):
                    for sh in range(s):
                        for sw in range(s):
                            row, col = hb * s + sh, wb * s + sw
                            y, ys, ye = src_window(row, g)
                            x, xs, xe = src_window(col, g)
                            acc = torch.zeros(embed_dim)
                            wsum = 0.0
                            for yy in range(ys, ye):
                                wy = cubic(yy - y)
                                for xx in range(xs, xe):
                                    wgt = wy * cubic(xx - x)
                                    wsum += wgt
                                    acc = acc + wgt * table[yy * ng + xx]
                            frame.append(acc / wsum)
            return torch.stack(frame)

        if aj.get("vision_pos_emb_interp") == "bicubic":
            frame = bicubic_frame()
        else:
            frame = bilinear_frame()
        # tiled per temporal frame, then the bf16 upload round-trip
        pos_embeds = torch.cat([frame] * temporal, dim=0)
        pos_embeds = pos_embeds.to(torch.bfloat16).to(torch.float32)
        env["pos_embeds"] = pos_embeds
        assert pos_embeds.shape[0] == n_tokens

    # ── cu_seqlens: one segment per image; one image here ──
    STATE["vision_cu"] = [0, n_tokens]
    env["cu_seqlens"] = STATE["vision_cu"]
    env["max_seqlen"] = n_tokens
    env["cu_seqlens_full"] = STATE["vision_cu"]
    env["max_seqlen_full"] = n_tokens
    env["cu_seqlens_window"] = STATE["vision_cu"]
    env["max_seqlen_window"] = n_tokens
    # window permute indices operate on MERGED rows (num_tokens /
    # merge_factor), matching the carrier's reshape-before-gather
    env["window_index"] = list(range(n_tokens // merge_factor))
    env["reverse_indices"] = list(range(n_tokens // merge_factor))
    env["position_ids"] = list(range(n_tokens))
    # pos_embeds: [L, embed_dim] additive table — set by the learned
    # pos-embed block above for arches declaring one; unused elsewhere
    # (the env prelude's zeros default covers carriers that never read it).

    # ── Qwen2.5-VL window dispatch — a transcription of
    # build_qwen2_5_window_dispatch (scratchy-vision) + the S²
    # block-grouped permutes the metal wrapper applies to the rope
    # tables. Only arches carrying a vision_window_size bound (the
    # codegen `windowed_attn_window_size` override) take this path;
    # the natural-order tables above are the identity default. ──
    ws = v.get("window_size")
    if ws is not None:
        win_cells = ws // s // patch
        assert win_cells > 0, "window_size/S/patch_size must be > 0 (tiny bounds)"
        llm_h, llm_w = g // s, g // s
        assert temporal * llm_h * llm_w == n_tokens // (s * s), "grid/merge mismatch"
        window_index = []
        cu = [0]
        cu_last = 0
        max_cells = 0
        # single image (gt frames): frame_base = ti*llm_h*llm_w
        for ti in range(temporal):
            frame_base = ti * llm_h * llm_w
            pad_h = (win_cells - llm_h % win_cells) % win_cells
            pad_w = (win_cells - llm_w % win_cells) % win_cells
            nh = (llm_h + pad_h) // win_cells
            nw = (llm_w + pad_w) // win_cells
            for wh in range(nh):
                for ww in range(nw):
                    seg_cells = 0
                    for ih in range(win_cells):
                        for iw in range(win_cells):
                            row = wh * win_cells + ih
                            col = ww * win_cells + iw
                            if row < llm_h and col < llm_w:
                                window_index.append(frame_base + row * llm_w + col)
                                seg_cells += 1
                    cu_last += seg_cells * (s * s)
                    cu.append(cu_last)
                    max_cells = max(max_cells, seg_cells)
        # dedup_consecutive
        cu_win = [cu[0]] + [c for i, c in enumerate(cu[1:], 1) if c != cu[i - 1]]
        # invert_permutation
        reverse = [0] * len(window_index)
        for i, p in enumerate(window_index):
            reverse[p] = i
        env["window_index"] = window_index
        env["reverse_indices"] = reverse
        env["cu_seqlens_window"] = cu_win
        env["max_seqlen_window"] = max_cells * (s * s)

        # S² block-grouped row permute of the rope tables (the metal
        # wrapper's permute_rows_block_grouped_{bf16,f32}): permutation
        # is over merged cells, each cell = s² consecutive rows moved
        # as a unit, intra-cell order preserved.
        def permute_block_grouped(table):
            rows = table.shape[0]
            inner = table.shape[1]
            assert rows == n_tokens
            out = torch.empty_like(table)
            for i, cell in enumerate(window_index):
                out[i * s * s:(i + 1) * s * s] = table[cell * s * s:(cell + 1) * s * s]
            return out

        freqs = permute_block_grouped(freqs)
        STATE["vision_freqs"] = freqs
        STATE["vision_cos"] = freqs.cos().to(torch.bfloat16).to(torch.float32)
        STATE["vision_sin"] = freqs.sin().to(torch.bfloat16).to(torch.float32)
        env["cos"] = STATE["vision_cos"]
        env["sin"] = STATE["vision_sin"]
        env["freqs"] = freqs


if __name__ == "__main__":
    main()

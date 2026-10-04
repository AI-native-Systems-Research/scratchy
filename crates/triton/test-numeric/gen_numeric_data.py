# Copyright (c) 2026 IBM Corporation. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""THE CHECKED-IN TORCH REFERENCE DATA, produced by the FIXTURES' OWN `inputs()`/`reference()`.

    python3 gen_numeric_data.py <config> <out-dir>
    python3 gen_numeric_data.py --list

Per config it writes, into `<out-dir>`:

  * `in_<argname>.bin` -- one file per KERNEL POINTER ARGUMENT, in the kernel's own dtype, raw
    little-endian, in the kernel's own parameter order. The output pointer gets an `in_desc_o.bin`
    of zeros, because the harness owes the device an HBM buffer for it.
  * `ref_out.bin` -- the fixture `reference()`'s result, `.to(torch.float16)`, raw little-endian,
    in the element order the kernel's `o_desc.store` writes.
  * `meta.json` -- machine-readable provenance; a Rust reader is written against this schema.
  * `sha256.txt` -- `sha256  <filename>` per data file, so the bytes' identity is in the tree.

THREE BUFFERS ARE NOT STORED DENSELY, and each one announces itself in the FILE NAME as well as in
`meta.json`, so a reader that ignores the metadata cannot find a file at the name it guessed and
fails closed instead of misreading a short file as a long one:

  * `in_<arg>.sparse` -- a `[rows, cols]` table of which only a few rows are ever read, stored as
    those rows plus their indices (`SPRSTBL1`, laid out in full above `write_sparse`). Granite's
    `[49159, 4096]` embedding table is 384 MiB of which the kernel gathers 256 rows: 2 MiB stored.
    AN ABSENT ROW EXPANDS TO ZERO, which is what makes it sound -- see `write_sparse`.
  * `in_<arg>.rep` -- one block of a `repeat_interleave`d buffer plus the rule. RoPE's angle tables
    are `H` consecutive copies of each distinct row, so `rope_q32`'s 2 MiB each carry 64 KiB.
  * neither, for `swiglu_mlp_granite_flat`: 300 MiB of DENSELY read weights, so its `meta.json`
    carries `"oversized": true` and `"staging"` and no bytes are checked in.

`"data_dir"` in a `meta.json` means the `file` fields resolve against THAT sibling directory --
`embedding_granite_bm128` is byte-for-byte `embedding_granite` because `BLOCK_M` never reaches
`inputs()`, so it carries one statement of the bytes rather than a second copy. `generate` VERIFIES
that identity against the sibling's `sha256.txt` and refuses to share if it ever stops holding.

WHY THIS SCRIPT EXISTS RATHER THAN A RUST GENERATOR
===================================================

The point of the harness is THE SAME BYTES, so torch's own RNG has to be the one that draws them.
Restating `randn` in Rust would be a second stimulus that agrees with this one only by accident,
and restating a `reference()` would compare the device against a paraphrase. So this script never
does arithmetic a fixture already does: it calls `inputs()` and `reference()` and dumps what comes
back. The two exceptions are stated where they occur and are visible in `meta.json`:

  1. THE WEIGHT ORIENTATION FOR THE DECODER (`_dec_inputs`). `decoder_block.layer_weights` builds
     the weights in the orientation ITS OWN `reference()` multiplies -- `h @ w`, i.e. `[in, out]`.
     Its docstring says "the [in, out] orientation the kernel reads", and THAT PHRASE IS WRONG:
     every dot in `_decoder_layer` carries `.T` on the weight operand and `wg_desc`'s shape is
     `[D_FF, D_MODEL]` while `layer_weights` returns `wg` as `[d_model, d_ff]`. Those are the same
     numbers in the other orientation, so the kernel's buffer is the TRANSPOSE. This script
     transposes on the way to the `.bin`, which is a RELAYOUT of the fixture's own values, not a
     different stimulus.
  2. SWIGLU HAS NO FIXTURE `inputs()`/`reference()` (`swiglu_mlp.py` ships `constexprs()` only) and
     nothing in `test/pod/` covers the THREE-MATMUL MLP -- `swiglu_case.py` / `swiglu_oracle.py`
     hand the device `g` and `u` as HBM activations and reference the ELEMENTWISE `silu(g)*u` alone.
     So `_swiglu_inputs`/`_swiglu_reference` are AUTHORED HERE, and say so in `meta.json`'s
     `reference_note`. The activation is `swiglu_oracle.true_swiglu`'s form (`g*sigmoid(g)*u`), the
     stimulus follows `decoder_block.layer_weights`' `randn * 1/sqrt(d_model)`, and the compute is
     f32 with one `.to(f16)` at the end -- the house style of every `reference()` in `test/fixtures`.

THE CONFIGURATION TABLE IS A COPY, AND THE ORIGINAL IS RUST
===========================================================

`rust/triton-ktir-superdsc/src/cases.rs`'s `case()` is the authority for the constexprs and the
grid -- it used to live in `examples/bake_py.rs` and moved so `triton-numeric` reads the same
statement; `CONFIGS` below restates it because this script cannot call it. VERIFIED FIELD BY FIELD
against that file, including `dec_ptrs_one`/`dec_ptrs_two`'s argument ORDER, which is also what the
`.py` kernel signatures spell. Any disagreement is a bug HERE.

TRITON IS NOT INSTALLED ON THE POD (measured: `ModuleNotFoundError: No module named 'triton'`), and
every fixture does `import triton` at module scope. `_stub_triton()` installs a placeholder so the
TORCH-SIDE helpers import; the `@triton.jit` kernels then exist as plain functions and are never
called. Nothing about the reference values depends on it -- see the docstring there.
"""

import argparse
import hashlib
import importlib
import json
import os
import struct
import sys
import types

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURES = os.path.abspath(os.path.join(_HERE, "..", "fixtures"))

SEED = 0

# ---------------------------------------------------------------------------
# importing a fixture where triton is absent


def _stub_triton():
    """Install a placeholder `triton` / `triton.language` so a fixture module imports.

    The fixtures touch triton at module scope in exactly two ways: `@triton.jit` as a decorator,
    and `tl.constexpr` in a parameter ANNOTATION (evaluated at `def` time, since no fixture does
    `from __future__ import annotations`). Everything else -- `tl.zeros`, `tl.dot`,
    `tl.make_tensor_descriptor` -- is inside a body that this script never runs.

    So the stub is: `jit` returns the function unchanged, and any other attribute is a fresh
    sentinel object. NO ARITHMETIC IS STUBBED, and none could be: the values this script dumps come
    from `inputs()` and `reference()`, which are plain torch and never mention triton. If a fixture
    ever needed the real compiler to produce a reference, this would fail at import rather than
    quietly hand back a wrong number.
    """

    class _Any(types.ModuleType):
        def __getattr__(self, name):  # pragma: no cover - trivial
            if name.startswith("__"):
                raise AttributeError(name)
            v = type(name, (), {})
            setattr(self, name, v)
            return v

    tl = _Any("triton.language")
    tl.math = _Any("triton.language.math")
    triton = _Any("triton")
    triton.jit = lambda f=None, **kw: (f if f is not None else (lambda g: g))
    triton.language = tl
    sys.modules.setdefault("triton", triton)
    sys.modules.setdefault("triton.language", tl)
    sys.modules.setdefault("triton.language.math", tl.math)


def fixture(name, fixtures_dir):
    """Import `<fixtures_dir>/<name>.py`, stubbing triton only if the real one is missing."""
    try:
        importlib.import_module("triton")
    except ModuleNotFoundError:
        _stub_triton()
    if fixtures_dir not in sys.path:
        sys.path.insert(0, fixtures_dir)
    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# the constexpr table -- a copy of `triton-ktir-superdsc/src/cases.rs`, which is the authority

LOG2E = 1.44269504  # decoder_block.py's own constant, folded into QK_SCALE there


def _rmsnorm_ce(m, d_model):
    return {"M": m, "D_MODEL": d_model, "BLOCK_M": 64,
            "EPS": 1e-05, "INV_D": 1.0 / d_model}


def _rope_ce(h):
    return {"H": h, "N_TOK": 256, "HEAD_DIM": 128, "HALF": 64}


def _embedding_ce(n_tok, block_m):
    return {"N_TOK": n_tok, "V": 49159, "D_MODEL": 4096,
            "BLOCK_M": block_m, "EMB_SCALE": 12.0}


def _swiglu_ce(d_model, d_ff, block_n, block_k):
    return {"M": 64, "D_MODEL": d_model, "D_FF": d_ff,
            "BLOCK_M": 64, "BLOCK_N": block_n, "BLOCK_K": block_k}


def _dec_ce(block_n):
    return {"M": 64, "D_MODEL": 128, "D_FF": 256, "BLOCK_N": block_n, "HALF": 64,
            "EPS": 1e-05, "INV_D": 1.0 / 128.0,
            "QK_SCALE": 0.0078125 * LOG2E, "RM": 0.22}


def _attn_ce_1tile():
    # A copy of `attn_ce` in `triton-ktir-superdsc/src/cases.rs` (the authority) at the
    # non-causal 1-tile geometry: STAGE=1, everything else shared with the 16-tile launch.
    return {"Z": 1, "H": 4, "N_CTX": 256, "HEAD_DIM": 128,
            "BLOCK_M": 64, "BLOCK_N": 64, "GQA": 2, "STAGE": 1, "sm_scale": 1.0}


CONFIGS = {
    "rmsnorm_granite": dict(fixture="rmsnorm", kernel="rmsnorm_fwd",
                            ce=_rmsnorm_ce(64, 4096), grid=[1]),
    "rope_q32": dict(fixture="rope", kernel="rope_fwd", ce=_rope_ce(32), grid=[256]),
    "rope_kv8": dict(fixture="rope", kernel="rope_fwd", ce=_rope_ce(8), grid=[256]),
    "embedding_granite": dict(fixture="embedding", kernel="embedding_fwd",
                              ce=_embedding_ce(256, 64), grid=[4]),
    # ⭐⭐⭐ THE GATHER AT **ONE INDEX STICK**, which is the only extent the hardware's index path
    # admits in one op. The emitted descriptor spans the whole node, so its index holds one entry per
    # TOKEN -- 256 for the two configurations above, against the 32-entry (128-byte) stick dxp fills the
    # L3LU IBR with in a single transfer. MEASURED on the card at 256: every output row `r` held the
    # table row `ids[r mod 32]` names, rows 0..31 exact and the whole tensor at within_2pct 0.130597
    # with the right rms. At N_TOK = 32 nothing wraps, so this is the configuration a gather can be
    # verified END TO END on, and the two above are the refusing side of the build-time guard.
    #
    # Its own bytes, NOT `embedding_granite`'s: N_TOK reaches `inputs()` (unlike BLOCK_M), so the ids,
    # the sparse table and the reference are all different. 32 gathered rows is ~262 KB stored.
    "embedding_granite_m32": dict(fixture="embedding", kernel="embedding_fwd",
                                  ce=_embedding_ce(32, 32), grid=[1]),
    # ⭐ THE SAME BYTES, TWO CONFIGURATIONS. `BLOCK_M` never reaches `inputs()` -- the fixture's
    # signature has no `block_m` at all -- so this config's stimulus is byte-for-byte
    # `embedding_granite`'s. `data_dir` says so instead of a second 4 MiB copy: the `file` fields
    # resolve against THAT directory. It keeps its own `constexprs` (BLOCK_M=128), its own `grid`
    # ([2]) and its own `sha256.txt`. `generate` VERIFIES the identity rather than assuming it -- if
    # a future fixture change made `BLOCK_M` reach the data, the sha comparison fails and the
    # sharing is refused rather than silently serving the wrong bytes.
    "embedding_granite_bm128": dict(fixture="embedding", kernel="embedding_fwd",
                                    ce=_embedding_ce(256, 128), grid=[2],
                                    data_dir="embedding_granite"),
    "swiglu_mlp_flat": dict(fixture="swiglu_mlp", kernel="swiglu_mlp_fwd",
                            ce=_swiglu_ce(128, 256, 256, 128), grid=[1]),
    "swiglu_mlp_granite_flat": dict(fixture="swiglu_mlp", kernel="swiglu_mlp_fwd",
                                    ce=_swiglu_ce(4096, 12800, 12800, 4096), grid=[1]),
    # ⭐ GRANITE WIDTH AT THE BLOCKING THAT FITS THE EXECUTOR'S LX MODEL, and it is the same bytes
    # as `swiglu_mlp_granite_flat` -- BLOCK_N/BLOCK_K are TILING knobs and `_swiglu_inputs` reads
    # only M/D_MODEL/D_FF, so the stimulus and the reference are byte-for-byte that config's. NOT
    # shared via `data_dir`, because `data_dir` resolves against a SIBLING IN-TREE directory and
    # these bytes are not in the tree at all (`oversized`); the two staged directories are instead
    # checked against each other with `sha256sum`, and each `meta.json` carries its own sha map.
    "swiglu_mlp_granite_tiled_k": dict(fixture="swiglu_mlp", kernel="swiglu_mlp_fwd",
                                       ce=_swiglu_ce(4096, 12800, 64, 2048), grid=[1]),
    # THE MIDDLE BLOCKING (BLOCK_N 64, BLOCK_K 4096), which REFUSED at this executor from Sep 19
    # (`ScfFor 1589248 + 524288`) until two Sep 28 changes met the fold that restored the
    # accumulate form: `unroll_constant_trip_loops` made the program straight-line (the loop's
    # yield no longer charges a fresh carry beside the live one), and the emulator's
    # consume-on-last-use frees each accumulator at the matmul that takes it as `outs`, so 2 MiB
    # holds it. Registered as a RUNNING config with its own staged directory -- same bytes as the
    # other two, same `sha256.txt` -- so its OWN derived envelope gates it (t_k=1, TIGHTER than
    # tiled_k's, never a shared-and-looser arrangement).
    "swiglu_mlp_granite": dict(fixture="swiglu_mlp", kernel="swiglu_mlp_fwd",
                               ce=_swiglu_ce(4096, 12800, 64, 4096), grid=[1]),
    # THE DEBUG TWIN of `swiglu_mlp_granite_tiled_k`: two N-trips, two K-trips, M=16 to stay
    # under the matmul util floor so no output-width bump fires -- every intermediate's device
    # width equals its logical width, which is the granite tiled_k bake's own regime.
    "swiglu_mlp_tiled_k": dict(fixture="swiglu_mlp", kernel="swiglu_mlp_fwd",
                               ce=_swiglu_ce(128, 256, 128, 64) | {"M": 16, "BLOCK_M": 16},
                               grid=[1]),
    "decoder_layer_one_flat": dict(fixture="decoder_block", kernel="decoder_layer_fwd",
                                   ce=_dec_ce(256), grid=[1]),
    "decoder_two_layers_flat": dict(fixture="decoder_block", kernel="decoder_two_layers_fwd",
                                    ce=_dec_ce(256), grid=[1]),
    # ⭐⭐⭐ THE ONE-TILE NON-CAUSAL FLASH BLOCK -- the first attention kernel through the ladder.
    #
    # grid [1, 1] = Z=0, head 0, query rows 0..BLOCK_M of a Z=1/H=4/GQA=2/N_CTX=256 geometry, so
    # the whole function is ONE unrolled flash block over the natural-layout buffers the kernel's
    # own descriptors declare (Q [y_dim, hd], K/V [kv_dim, hd], O [y_dim, hd]) -- the 16-tile
    # grid-corner wall is what keeps the full launch out, and this row exists so the numerics are
    # proven first. STAGE=1 is non-causal, so the reference is plain unmasked SDPA over the
    # window's keys; the reference is computed in f32 and compared in f16, the same contract every
    # other row's `reference()` states.
    "attention_flash_noncausal_1tile": dict(fixture="attention_flash",
                                            kernel="attn_fwd_noncausal",
                                            ce=_attn_ce_1tile(), grid=[1, 1]),
    # ⭐⭐⭐ RUNG 3 OF THE ADDRESS-PROVENANCE LADDER: `c = a * scale` with `scale` a RUNTIME fp16
    # scalar argument, the smallest kernel that can carry a launch-bound value end to end. The
    # reference binds SCALE = 1.5; the card run's NEGATIVE CONTROL binds a DIFFERENT value
    # (0.5) with the SAME input bytes and must DIVERGE -- the recorded failure mode this row
    # exists to catch is the unbound-scale identity trap (clean exit 0, readback = the input,
    # within_2pct 1.0000), so a control that still matches with a wrong bind is the pass that
    # proves nothing. `scale` is NOT an input file: the runner binds it as `const:t2=1.5`,
    # which is the whole point of the rung.
    "mul_scale_small": dict(fixture="mul_scale", kernel="mul_scale_kernel",
                            ce={"n": 1024, "BLOCK": 64}, grid=[16]),
    # ⭐⭐⭐ RUNG 4 OF THE ADDRESS-PROVENANCE LADDER: `o = p @ gather(v, ids)` -- the row
    # index is DATA read from device memory, and the gathered rows feed a CONTRACTION.
    # This is paged attention's V leg exactly as scratchy runs it: the rows gathered
    # directly and contracted WHERE THEY LIE (row-blocked gathered rows ARE the kernel
    # slot's `[k, n]` residency), no transpose anywhere -- the transposed K leg
    # (`tl.dot(q, k_rows.T)`) needs a relayout this arch's dxp refuses, and scratchy's
    # answer to it is the host-presented Kt plane, so the contract refuses it by name.
    # K=16 is the frontend's own `tl.dot` K floor. The ids are a PERMUTATION (a shuffle
    # of 0..15), deliberately NOT the identity: a direct read of the table's first K
    # rows is exactly the wrong answer the whole-function gather guard exists to refuse,
    # so the reference must not agree with it. The card run's negative control binds the
    # ids SORTED against the same p/v bytes -- the output must equal the identity
    # permutation's product, which DIFFERS from `ref_out.bin`, so a control that still
    # matches is the direct-read trap (descriptor read the table at its base and ignored
    # the ids) and is a FAILURE.
    "paged_score_small": dict(fixture="paged_score", kernel="paged_vmatmul_fwd",
                              ce={"M": 8, "K": 64, "V": 128,
                                  "BLOCK_M": 8, "HEAD_DIM": 64}, grid=[1]),
    # ⭐⭐⭐ TASK 27: PAGED FLASH ATTENTION, THE ONE-BLOCK STEP -- the full flash body
    # around rung 4's gathered V-leg contraction. The block table (desc_ids) picks the
    # V rows at runtime; the score matmul runs over a HOST-PRESENTED Kt plane (the
    # shipped attention's third physical plane, because a transposed gather is
    # arch-refused and two gathers in one program are refused by `gather_of`); the value
    # contraction is `tl.dot(p, v_rows)` verbatim. One launch = one KV block = 64 rows.
    # The ids are the SAME PERMUTATION as `paged_score_small`'s, for the same negative
    # control: sorted ids must read the table's first 64 rows directly and DIFFER from
    # `ref_out.bin` -- a match there is the direct-read trap and a FAILURE.
    "paged_attention_small": dict(fixture="paged_attention", kernel="paged_attn_fwd",
                                  ce={"N_CTX": 64, "HEAD_DIM": 64, "V": 1024,
                                      "BLOCK_M": 64}, grid=[1]),
    # TASK 28: the multi-block paged form. The block table is `[2, 64]` i32 on device
    # (one row per block, sliced at a constant corner per unrolled trip), kt is
    # `[64, 128]` with block t's K^T in column block t, and the reference runs the
    # kernel's own base-2 recurrence over both blocks. Controls: sorted table, block
    # SWAP, and the input corruptions -- all must differ from `ref_out.bin`.
    "paged_attention_multiblock": dict(
        fixture="paged_attention_multiblock", kernel="paged_attn_multiblock_fwd",
        ce={"NUM_BLOCKS": 2, "BLOCK_N": 64, "HEAD_DIM": 64, "V": 1024,
            "BLOCK_M": 64}, grid=[1]),
}

# Above this, the bytes are NOT checked in: `meta.json` carries `"oversized": true`, `"staging"`,
# the shapes and the sha256s, and the files stay wherever this script was pointed. Measured on the
# BYTES ACTUALLY WRITTEN for the inputs -- so a sparse or replicated encoding counts at its stored
# size, which is the whole point of having one.
OVERSIZE_INPUT_BYTES = 8 * 1024 * 1024

# Where an oversized config's bytes were staged. Named in `meta.json` so a harness can be POINTED at
# a directory and report MISSING when it is not, rather than treating an absent directory as a pass.
POD = "nickm-7db9667cdd-z2jc6"
STAGING_DIR = "/tmp/numout"


def _staging(config):
    return (f"THE BYTES ARE NOT IN THE TREE -- this configuration's inputs exceed "
            f"{OVERSIZE_INPUT_BYTES} B even at their stored size, and unlike the embedding table "
            f"and the RoPE angle tables there is nothing sparse or replicated about them to "
            f"exploit: every weight element is read. Staged on the pod `{POD}` at "
            f"`{STAGING_DIR}/{config}/`. Reproduce anywhere torch {torch.__version__} is available "
            f"with `python3 test/numeric/gen_numeric_data.py {config} <dir>`; the sha256 map in "
            f"this file is what a regenerated set must match. A harness must be POINTED at a "
            f"staged directory and must report MISSING when it has not been -- an absent directory "
            f"is not a pass.")


# ---------------------------------------------------------------------------
# per-fixture wiring: arg name -> tensor, in the kernel's own parameter order


def _f16(t):
    return t.to(torch.float16).contiguous()


def _rmsnorm_inputs(mod, ce):
    x, w = mod.inputs(seed=SEED, m=ce["M"], d_model=ce["D_MODEL"])
    ref = mod.reference(x, w, eps=ce["EPS"])
    ins = [("desc_x", x, "f16"), ("desc_w", w, "f16"),
           ("desc_o", torch.zeros_like(x), "f16")]
    return ins, ref, None


def _rope_inputs(mod, ce):
    """The angle tables are stored as ONE BLOCK and a replication rule -- see `write_replicated`.

    `rope.stage_table` builds them as `cat((angles, angles), -1).repeat_interleave(h, dim=0)`, so the
    `[N_TOK * H, HEAD_DIM]` buffer the kernel reads is `H` consecutive copies of each of `N_TOK`
    distinct rows: at H=32 that is 2 MiB of file carrying 64 KiB of distinct values. `x` and `o` are
    NOT replicated -- every (token, head) row of `x` is distinct -- so only cos/sin carry the rule.
    """
    h, n_tok, hd = ce["H"], ce["N_TOK"], ce["HEAD_DIM"]
    x, cos, sin = mod.inputs(seed=SEED, h=h, n_tok=n_tok, head_dim=hd)
    ref = mod.reference(x, cos, sin, h=h, n_tok=n_tok)
    rep = ("replicate", h)
    ins = [("desc_x", x, "f16"), ("desc_cos", cos, "f16", rep), ("desc_sin", sin, "f16", rep),
           ("desc_o", torch.zeros_like(x), "f16")]
    return ins, ref, _ROPE_NOTE


_ROPE_NOTE = (
    "desc_cos and desc_sin are stored as ONE [N_TOK, HEAD_DIM] block plus a `replicate` rule, not "
    "as the full [N_TOK * H, HEAD_DIM] buffer: rope.stage_table's repeat_interleave(h, dim=0) makes "
    "the full buffer H consecutive copies of each distinct row. The rule is repeat_interleave, NOT "
    "repeat -- full[i*factor + j] = stored[i], not stored[i] tiled factor times -- and those two "
    "differ only in their VALUES, never in their shape, which is the same token-major/head-major "
    "trap stage_table's own docstring records. So the rule is not merely asserted here: the "
    "generator expands the stored block by the recorded rule and requires it to reproduce the full "
    "buffer BIT FOR BIT before writing, and requires the `repeat` reading NOT to."
)


def _embedding_inputs(mod, ce):
    """The vocabulary table is stored SPARSELY -- only the rows `desc_ids` names. See `write_sparse`
    for the format and for why an absent row expanding to ZERO can only cause a false FAILURE."""
    n_tok, v, d_model = ce["N_TOK"], ce["V"], ce["D_MODEL"]
    ids, table = mod.inputs(seed=SEED, n_tok=n_tok, v=v, d_model=d_model)
    ref = mod.reference(ids, table, emb_scale=ce["EMB_SCALE"])
    ins = [("desc_ids", ids, "i32"), ("desc_table", table, "f16", ("sparse", ids)),
           ("desc_o", torch.zeros(n_tok, d_model, dtype=torch.float16), "f16")]
    return ins, ref, _EMBEDDING_NOTE


_EMBEDDING_NOTE = (
    "desc_table is stored SPARSELY: the kernel gathers 256 rows of a 49159-row table, so only the "
    "distinct rows desc_ids names are kept -- 2 MiB instead of 384 MiB. THE PROPERTY THAT MAKES "
    "THIS SOUND: an absent row expands to ZERO, so if the kernel ever reads a row the ids did not "
    "name it reads zeros and the comparison DIVERGES. Sparsification can therefore produce a false "
    "FAILURE but never a false PASS -- it cannot mask a wrong gather, which is the bug class this "
    "configuration exists to catch. A reader that expands to anything other than zero (uninitialised "
    "memory, a splat, a wrap-around) breaks that guarantee and must not."
)


_SWIGLU_NOTE = (
    "AUTHORED HERE, not ported. swiglu_mlp.py ships constexprs() only -- no inputs(), no "
    "reference() -- and test/pod's SwiGLU apparatus (swiglu_case.py, swiglu_oracle.py, "
    "verify_swiglu.py, gen_swiglu_artifact.py) covers the ELEMENTWISE silu(g)*u ONLY: it hands the "
    "device g and u as two HBM activation planes (swiglu-stage=3) and never forms x@Wg, x@Wu or "
    "h@Wd, so there is nothing there to port for the three-matmul MLP. WHICH PART IS PORTED AND "
    "WHICH IS NEW: ported is the ACTIVATION's form -- swiglu_oracle.py:151 true_swiglu computes "
    "(g*sigmoid(g))*u, and _swiglu_reference's `(gate * torch.sigmoid(gate)) * up` matches THAT form "
    "term for term (it is also swiglu_mlp.py's delta 8, silu(gate)*up, not upstream's s*(up+1)). NEW "
    "here are the three matmuls around it -- x@Wg.T, x@Wu.T, h@Wd.T -- and the stimulus, because "
    "nothing in the tree had either. The stimulus follows "
    "decoder_block.py:500 layer_weights -- torch.randn * (1/sqrt(d_model)), one seeded generator, "
    "drawn x, wg, wu, wd -- and the weights are drawn DIRECTLY in the kernel's own buffer "
    "orientation (wg/wu [D_FF, D_MODEL], wd [D_MODEL, D_FF]) so no host transpose can go wrong. "
    "Compute is f32 with one .to(f16) at the end, as every reference() in test/fixtures is."
)


def _swiglu_inputs(mod, ce):
    """The three-matmul MLP: `o = (silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T`.

    THE ORIENTATIONS ARE THE KERNEL'S, read off `swiglu_mlp_fwd`'s descriptors rather than assumed.
    `wg_desc`/`wu_desc` are `shape=[D_FF, D_MODEL]` and their dot operands carry `.T`, so
    `g[:, n] = sum_i x[:, i] * wg[n, i]` and the buffers are `[D_FF, D_MODEL]`. `wd_desc` is
    `shape=[D_MODEL, D_FF]` and `wd_desc.load([0, n]).T` is `[BLOCK_N, D_MODEL]`, so
    `o[:, j] = sum_n h[:, n] * wd[j, n]` and that buffer is `[D_MODEL, D_FF]`. Both are HF's own
    `gate_proj.weight` / `down_proj.weight` shapes, and each `@` below takes the matching `.T`.
    """
    m, d_model, d_ff = ce["M"], ce["D_MODEL"], ce["D_FF"]
    g = torch.Generator().manual_seed(SEED)
    scale = 1.0 / (d_model ** 0.5)
    x = torch.randn(m, d_model, generator=g, dtype=torch.float16)
    wg = torch.randn(d_ff, d_model, generator=g, dtype=torch.float16) * scale
    wu = torch.randn(d_ff, d_model, generator=g, dtype=torch.float16) * scale
    wd = torch.randn(d_model, d_ff, generator=g, dtype=torch.float16) * scale
    ref = _swiglu_reference(x, wg, wu, wd)
    ins = [("desc_x", x, "f16"), ("desc_wg", wg, "f16"), ("desc_wu", wu, "f16"),
           ("desc_wd", wd, "f16"),
           ("desc_o", torch.zeros(m, d_model, dtype=torch.float16), "f16")]
    return ins, ref, _SWIGLU_NOTE


def _swiglu_reference(x, wg, wu, wd):
    """f32 compute, one cast at the end. `wg`/`wu` are `[D_FF, D_MODEL]`, `wd` is `[D_MODEL, D_FF]`
    -- the kernel's buffers -- so each `@` takes the transpose, which is HF's `x @ W.T`."""
    xf = x.to(torch.float32)
    gate = xf @ wg.to(torch.float32).T          # [M, D_FF]
    up = xf @ wu.to(torch.float32).T            # [M, D_FF]
    h = (gate * torch.sigmoid(gate)) * up       # swiglu_oracle.true_swiglu's form
    return (h @ wd.to(torch.float32).T).to(torch.float16)


_DEC_NOTE = (
    "decoder_block.reference() verbatim, on decoder_block.inputs()'s own dict. THE WEIGHT BUFFERS "
    "ARE TRANSPOSED ON THE WAY OUT: layer_weights() builds them in the orientation reference() "
    "multiplies (h @ w, [in, out]), but every dot in _decoder_layer carries .T on the weight and "
    "wg_desc's shape is [D_FF, D_MODEL] against layer_weights' [d_model, d_ff] -- so the kernel's "
    "buffer is the transpose. layer_weights' docstring calls its orientation 'the [in, out] "
    "orientation the kernel reads', and that phrase is wrong; the code decides. n1/n2 are 1-D and "
    "mask/cos/sin are passed through untouched."
)

# `decoder_layer_fwd`'s parameter order, minus desc_x/desc_o: (arg, weight key).
_DEC_ONE = [("desc_n1", "n1"), ("desc_wq", "wq"), ("desc_wk", "wk"), ("desc_wv", "wv"),
            ("desc_wo", "wo")]
_DEC_MLP = [("desc_n2", "n2"), ("desc_wg", "wg"), ("desc_wu", "wu"), ("desc_wd", "wd")]
# Which keys are 2-D and therefore transposed into the kernel's [out, in] buffer.
_DEC_T = {"wq", "wk", "wv", "wo", "wg", "wu", "wd"}


def _dec_w(layer, key):
    t = layer[key]
    return _f16(t.T) if key in _DEC_T else _f16(t)


def _dec_inputs(mod, ce, two_layers):
    m, d_model, d_ff = ce["M"], ce["D_MODEL"], ce["D_FF"]
    n_layers = 2 if two_layers else 1
    inp = mod.inputs(seed=SEED, m=m, d_model=d_model, d_ff=d_ff, layers=n_layers)
    ref = mod.reference(inp)
    x = inp["x"]
    o = torch.zeros(m, d_model, dtype=torch.float16)
    ins = [("desc_x", x, "f16"), ("desc_o", o, "f16")]
    if not two_layers:
        # desc_x, desc_o, n1, wq, wk, wv, wo, mask, cos, sin, n2, wg, wu, wd
        L = inp["layers"][0]
        ins += [(a, _dec_w(L, k), "f16") for a, k in _DEC_ONE]
        ins += [("desc_mask", inp["mask"], "f16"), ("desc_cos", inp["cos"], "f16"),
                ("desc_sin", inp["sin"], "f16")]
        ins += [(a, _dec_w(L, k), "f16") for a, k in _DEC_MLP]
    else:
        # desc_x, desc_o, {n1,wq,wk,wv,wo,n2,wg,wu,wd}a, ...b, mask, cos, sin
        for suffix, L in zip("ab", inp["layers"]):
            ins += [(a + suffix, _dec_w(L, k), "f16") for a, k in _DEC_ONE + _DEC_MLP]
        ins += [("desc_mask", inp["mask"], "f16"), ("desc_cos", inp["cos"], "f16"),
                ("desc_sin", inp["sin"], "f16")]
    return ins, ref, _DEC_NOTE


# `attn_fwd_noncausal`'s input staging, for the ONE-TILE (grid [1, 1]) window. Authored here
# because `attention_flash.py` has no torch-side `inputs()`/`reference()` -- its bodies are
# `@triton.jit` kernels and nothing else, the same ground as the SwiGLU rows.
#
# ⭐ THE GEOMETRY IS THE CASE ROW'S OWN: Z=1, H=4, GQA=2, N_CTX=256, HEAD_DIM=128. Grid [1, 1]
# picks Z=0, head 0, query rows 0..BLOCK_M(64); the kv plane it reads is plane 0 (off_h // GQA
# at head 0). The reference is therefore SDPA over THAT window -- not over the whole tensor,
# which the other 15 (z, head) blocks would cover and this launch never touches.
#
# ⭐⭐⭐ THE REFERENCE IS THE FLASH ALGORITHM ITSELF, IN F32, not torch's `sdpa`: the kernel
# computes `exp2(qk·log2e·sm_scale − m)·V / Σ…` with the running-max rescaling, and matching
# the ALGORITHM (not a fused library's own rounding) is what isolates the lowering's numerics
# from torch's kernel choice. The block's max over all 256 keys is one value per row, so the
# single-block form is `softmax(rows) @ V` exactly.
# `mul_scale_kernel`'s input staging: (a_ptr, c_ptr, scale). THE SCALE IS NOT A FILE — it is the
# rung-3 scalar the launch binds as `const:t2=<value>`, so this row stages only a and c and the
# reference carries the value the card run must bind. SCALE = 1.5 is a DELIBERATELY non-trivial
# multiplier: not 1 (which would make the identity trap indistinguishable from a correct run at
# a glance) and not a power of two (which would only exercise mantissa passes).
MUL_SCALE = 1.5


def _mul_scale_inputs(mod, ce):
    a = mod.inputs(seed=SEED)
    ref = mod.reference(a, MUL_SCALE)
    ins = [("a_ptr", a, "f16"), ("c_ptr", torch.zeros_like(a), "f16")]
    note = (
        f"THE SCALE IS NOT A FILE. The kernel's third parameter is the rung-3 runtime scalar, "
        f"bound at launch as `const:t2={MUL_SCALE}` (one IEEE fp16, 2 B); this directory stages "
        f"only `a_ptr` and `c_ptr`. The reference multiplied by {MUL_SCALE}, so a launch that "
        f"binds anything else -- or nothing, which the device reads as whatever the segment held "
        f"-- diverges. THE NEGATIVE CONTROL IS PART OF THE CONTRACT: bind `const:t2=0.5` against "
        f"these same bytes and the output must NOT match `ref_out.bin`; a run that still matches "
        f"is the unbound-scale identity trap (readback = the input at within_2pct 1.0000) and is "
        f"a FAILURE, not a pass."
    )
    return ins, ref, note


def _paged_score_inputs(mod, ce):
    p, v = mod.inputs(seed=SEED, m=ce["M"], k=ce["K"], v=ce["V"], head_dim=ce["HEAD_DIM"])
    # THE IDS ARE A PERMUTATION, NOT THE IDENTITY. A descriptor that reads the table's
    # first K rows directly (ignoring the ids) produces `p @ v[:64]`, which is the
    # wrong answer this row's negative control catches: the sorted-ids product
    # must differ from `ref_out.bin`, so a direct-read trap cannot hide inside a pass.
    # K=64 is the vendor matmul emitter's own whole-stick floor (K must be a multiple
    # of 64 fp16 elements), met at its smallest.
    perm = torch.tensor([
        53, 29, 32, 22, 39, 40, 28, 26, 50, 42, 49, 20, 55, 43, 58, 47,
        0, 57, 37, 2, 8, 15, 24, 31, 7, 38, 36, 21, 52, 60, 27, 56,
        45, 51, 62, 10, 12, 44, 41, 54, 13, 16, 61, 11, 17, 23, 34, 33,
        14, 3, 48, 18, 35, 59, 1, 4, 5, 9, 63, 25, 46, 6, 19, 30,
    ], dtype=torch.int32)
    ref = mod.reference(p, v, perm)
    ins = [("desc_p", p, "f16"), ("desc_v", v, "f16"),
           ("desc_ids", perm, "i32"), ("desc_o", torch.zeros(8, 64), "f16")]
    note = (
        f"THE IDS ARE A PERMUTATION of 0..63 (K=64, the vendor matmul's whole-stick "
        f"floor), not the identity, so a direct read of the table's first 64 rows is a "
        f"DIFFERENT matrix product and must not match `ref_out.bin`. THE NEGATIVE "
        f"CONTROL IS PART OF THE CONTRACT: bind the ids SORTED (0..63) against these "
        f"same p/v bytes and the output must equal `p @ v[:64]` and DIFFER from "
        f"`ref_out.bin` -- a run that still matches with sorted ids is the direct-read "
        f"trap (the descriptor ignored the index operand) and is a FAILURE, not a pass."
    )
    return ins, ref, note


def _paged_attention_inputs(mod, ce):
    q, kt, table = mod.inputs(seed=SEED, m=ce["BLOCK_M"], n_ctx=ce["N_CTX"],
                              v=ce["V"], head_dim=ce["HEAD_DIM"])
    # THE SAME PERMUTATION, THE SAME NEGATIVE CONTROL as `paged_score_small`: sorted ids
    # read the table's first 64 rows directly, and that product must DIFFER from
    # `ref_out.bin` -- a run that still matches with sorted ids is the direct-read trap
    # (the gather legs ignored the index operand) and is a FAILURE, not a pass.
    perm = torch.tensor([
        53, 29, 32, 22, 39, 40, 28, 26, 50, 42, 49, 20, 55, 43, 58, 47,
        0, 57, 37, 2, 8, 15, 24, 31, 7, 38, 36, 21, 52, 60, 27, 56,
        45, 51, 62, 10, 12, 44, 41, 54, 13, 16, 61, 11, 17, 23, 34, 33,
        14, 3, 48, 18, 35, 59, 1, 4, 5, 9, 63, 25, 46, 6, 19, 30,
    ], dtype=torch.int32)
    ref = mod.reference(q, kt, table, perm)
    ins = [("desc_q", q, "f16"), ("desc_kt", kt, "f16"), ("desc_v", table, "f16"),
           ("desc_ids", perm, "i32"),
           ("desc_o", torch.zeros(ce["BLOCK_M"], ce["HEAD_DIM"]), "f16")]
    note = (
        f"THE IDS ARE A PERMUTATION of 0..63 (one KV block of 64 rows), not the "
        f"identity, so a direct read of the table's first 64 rows is a DIFFERENT "
        f"matrix product and must not match `ref_out.bin`. THE NEGATIVE CONTROL IS "
        f"PART OF THE CONTRACT: bind the ids SORTED (0..63) against these same "
        f"q/kt/v bytes and the output must equal the softmax-weighted product over "
        f"v[:64] and DIFFER from `ref_out.bin` -- a run that still matches with "
        f"sorted ids is the direct-read trap (the gather legs ignored the index "
        f"operand) and is a FAILURE, not a pass."
    )
    return ins, ref, note


def _paged_attention_multiblock_inputs(mod, ce):
    q, kt, table = mod.inputs(seed=SEED, m=ce["BLOCK_M"],
                              num_blocks=ce["NUM_BLOCKS"],
                              block_n=ce["BLOCK_N"], v=ce["V"],
                              head_dim=ce["HEAD_DIM"])
    # THE BLOCK TABLE IS A PERMUTATION of 0..127 in `[2, 64]` shape, scattered so no
    # block's rows are contiguous in the table. THREE negative controls are part of the
    # contract: (a) SORTED table must give the base-2 softmax over table[:128] in row
    # order and DIFFER from `ref_out.bin` (the direct-read trap); (b) a block SWAP
    # (rows flipped) must DIFFER (catches a trip reading the wrong table row -- the
    # one-block form cannot see block-level errors); (c) input corruptions as in the
    # one-block scorer.
    tbl = mod.PAGED_ATTN_TABLE[:ce["NUM_BLOCKS"] * ce["BLOCK_N"]].reshape(
        ce["NUM_BLOCKS"], ce["BLOCK_N"])
    ref = mod.reference(q, kt, table, tbl, block_n=ce["BLOCK_N"])
    ins = [("desc_q", q, "f16"), ("desc_kt", kt, "f16"), ("desc_v", table, "f16"),
           ("desc_table", tbl, "i32"),
           ("desc_o", torch.zeros(ce["BLOCK_M"], ce["HEAD_DIM"]), "f16")]
    note = (
        f"THE BLOCK TABLE IS A PERMUTATION of 0..{tbl.numel() - 1} in "
        f"[{ce['NUM_BLOCKS']}, {ce['BLOCK_N']}] shape, scattered across blocks. "
        f"SORTED table: output must equal the base-2 softmax over table rows in "
        f"order and DIFFER from `ref_out.bin` (a match is the direct-read trap). "
        f"BLOCK SWAP (rows flipped): must DIFFER (a trip reading the wrong table "
        f"row gives block-level wrong rows, which the one-block controls cannot "
        f"see). Both are FAILURES if they match, not passes."
    )
    return ins, ref, note


def _attn_inputs_1tile(mod, ce):
    z, h, gqa = ce["Z"], ce["H"], ce["GQA"]
    n_ctx, hd, block_m = ce["N_CTX"], ce["HEAD_DIM"], ce["BLOCK_M"]
    scale = ce["sm_scale"]
    y_dim = z * h * n_ctx
    kv_y_dim = z * (h // gqa) * n_ctx
    g = torch.Generator().manual_seed(SEED)
    q = torch.randn(y_dim, hd, generator=g)
    k = torch.randn(kv_y_dim, hd, generator=g)
    v = torch.randn(kv_y_dim, hd, generator=g)
    # The window: head 0's query rows 0..block_m against kv plane 0's all n_ctx keys.
    qw = q[:block_m].to(torch.float32)
    kw = k[:n_ctx].to(torch.float32)
    vw = v[:n_ctx].to(torch.float32)
    scores = (qw @ kw.T) * scale
    p = torch.softmax(scores, dim=-1)
    ref = (p @ vw).to(torch.float16)
    ins = [("desc_q", _f16(q), "f16"),
           ("desc_k", _f16(k), "f16"),
           ("desc_v", _f16(v), "f16"),
           ("desc_o", torch.zeros(y_dim, hd, dtype=torch.float16), "f16")]
    # The card writes the whole [y_dim, hd] buffer; only rows 0..block_m are the reference's.
    # `ref_out.bin` is the WINDOW the launch stores, so a harness comparing elementwise over the
    # full buffer must zero-fill the rest -- stated here so the reader sizes it, not guesses it.
    # ⛔ THIS REFERENCE IS 1-TILE-ONLY BY CONSTRUCTION (measured 2026-09-29: 960 of 1024 rows
    # all-zero). The 16-TILE launch's reference is NOT this file -- scoring the 16-tile card
    # output against it measures real output against zeros (measured: full-plane corr 0.25
    # while every position is actually 0.9999+). The 16-position reference is computed from
    # these same input files at score time (test/pod/attn_score_full16.py's `noncausal_ref`),
    # heads h reading kv plane h // GQA -- the checked-in file stays 1-tile because the
    # 16-tile configs share this fixture's num/ directory.
    full = torch.zeros(y_dim, hd, dtype=torch.float16)
    full[:block_m] = ref
    ins, full = _attn_card_staging_1tile(ins, full, block_m, hd, kv_y_dim, n_ctx)
    return ins, full, None


def _attn_card_staging_1tile(ins, full, block_m, hd, kv_y_dim, n_ctx):
    # ⭐⭐⭐ THE WHOLE-FUNCTION DOOR'S HOST LAYOUT, MEASURED ON CARD (2026-09-28, /work/attn/door).
    #
    # The generic door emits the flash block's four unrolled trips with operands addressed at
    # TILE extents, and each parameter's file must be staged in the layout its DESCRIPTOR's own
    # address law states -- the runner's `stick:tN=<rows>` restick spelling. Measured by reading
    # every stage off the card (`--out synth:tN@<rows>`) against a numpy model of the same stage:
    #
    #   * Q (the row-windowed ACTIVATION of the QK^T matmul): the A handle's stick-group stride
    #     derives from the TILE's m=block_m, so the descriptor reads element (r, c) at
    #     (c/64)*m*64 + r*64 + (c%64) -- the window padded to [m, hd] per 64-column stick group,
    #     i.e. a [m, m*32] stick-major buffer with the window in the leading hd columns.
    #   * K (the TRANSPOSE-B KERNEL of the QK^T matmul): the kernel reads a [in=hd, out=n_ctx]
    #     stick-major buffer -- K TRANSPOSED, host rows = hd. Staging K row-major read K[k][n]
    #     where the kernel wanted K^T[k][n].
    #   * V (the PLAIN-B KERNEL of the P·V matmul): the kernel's law (element (k,n) at
    #     (n/64)*(in_phys*64) + k*64 + (n%64), in_phys = the weight's own rows) reads the
    #     natural row-major [kv_y_dim, hd] staging EXACTLY -- measured, p@v exact given its p.
    #   * O (the WINDOWED OUTPUT): the launch writes [m, m*32] stick-major into t3's leading
    #     m*hd*2 bytes; the read-back states rows=m and the leading hd columns are the result.
    #
    # Before this staging the card returned uncorrelated garbage (max|err| 5.43, corr 0.013)
    # on a full-buffer read. ⛔ The end-to-end score previously quoted here (window corr
    # 0.999962) came from a scorer that sliced 128 of 2048 read-back columns (nuke: e78e53297);
    # the output is UNSCORED until a full-buffer scorer exists. The per-stage synth read-backs
    # are the verified evidence for this staging law.
    # The four `desc_*` names stay (the ktir-emulator path is layout-agnostic); this adds the
    # card spellings beside them.
    d = {}
    for name, t, kind in ins:
        d[name] = t
    q = d["desc_q"].reshape(-1, hd)
    k = d["desc_k"].reshape(-1, hd)
    v = d["desc_v"].reshape(-1, hd)
    # Q: window in the leading hd columns of a [m, m*32] buffer.
    qw = torch.zeros(block_m, block_m * 32, dtype=torch.float16)
    qw[:, :hd] = q[:block_m]
    # K^T: [hd, kv_y_dim] row-major.
    kt = k.t().contiguous()
    card = [("card_q", qw.contiguous(), "f16"),
            ("card_k", kt, "f16"),
            ("card_v", v.contiguous(), "f16"),
            ("card_o", torch.zeros(block_m, block_m * 32, dtype=torch.float16), "f16")]
    return ins + card, full


BUILD = {
    "rmsnorm_granite": _rmsnorm_inputs,
    "rope_q32": _rope_inputs,
    "rope_kv8": _rope_inputs,
    "embedding_granite": _embedding_inputs,
    "embedding_granite_bm128": _embedding_inputs,
    "embedding_granite_m32": _embedding_inputs,
    "swiglu_mlp_flat": _swiglu_inputs,
    "swiglu_mlp_tiled_k": _swiglu_inputs,
    "swiglu_mlp_granite_flat": _swiglu_inputs,
    "swiglu_mlp_granite_tiled_k": _swiglu_inputs,
    "swiglu_mlp_granite": _swiglu_inputs,
    "decoder_layer_one_flat": lambda mod, ce: _dec_inputs(mod, ce, two_layers=False),
    "decoder_two_layers_flat": lambda mod, ce: _dec_inputs(mod, ce, two_layers=True),
    "attention_flash_noncausal_1tile": _attn_inputs_1tile,
    "mul_scale_small": _mul_scale_inputs,
    "paged_score_small": _paged_score_inputs,
    "paged_attention_small": _paged_attention_inputs,
    "paged_attention_multiblock": _paged_attention_multiblock_inputs,
}

# Which function produced each side, for `meta.json`. The default is the FIXTURE's own
# `inputs`/`reference`; the two SwiGLU configs name THIS script's functions instead, because
# `swiglu_mlp.py` has neither -- `reference_note` says so at length.
FNS = {
    "swiglu_mlp_flat": ("_swiglu_inputs", "_swiglu_reference"),
    "swiglu_mlp_tiled_k": ("_swiglu_inputs", "_swiglu_reference"),
    "swiglu_mlp_granite_flat": ("_swiglu_inputs", "_swiglu_reference"),
    "swiglu_mlp_granite_tiled_k": ("_swiglu_inputs", "_swiglu_reference"),
    "swiglu_mlp_granite": ("_swiglu_inputs", "_swiglu_reference"),
}


# ---------------------------------------------------------------------------
# the dump


_NP = {"f16": "<f2", "i32": "<i4"}
_TORCH = {"f16": torch.float16, "i32": torch.int32}


def dump(t, dtype, path):
    """Raw little-endian bytes, in the tensor's own row-major element order."""
    a = t.to(_TORCH[dtype]).contiguous().numpy().astype(_NP[dtype], copy=False)
    with open(path, "wb") as f:
        f.write(a.tobytes())
    return a.nbytes


# ---------------------------------------------------------------------------
# THE SPARSE ROW FORMAT -- `SPRSTBL1`
#
# A `[rows, cols]` table of which only a few rows are ever read, stored as those rows and their
# indices. Every field little-endian; the whole layout is also in `meta.json`'s `sparse` block, so a
# reader has ONE statement to work from and this comment and that block are generated together.
#
#   offset  size  field       type      note
#   0       8     magic       ascii     exactly `SPRSTBL1`, no terminator
#   8       8     rows        u64le     the FULL row count (49159 for Granite's vocabulary)
#   16      8     cols        u64le     the FULL column count
#   24      4     itemsize    u32le     bytes per element (2 for f16)
#   28      4     n_present   u32le     number of DISTINCT rows stored
#   32      ...   records               `n_present` of them, ASCENDING by row_index
#
#   record: row_index u32le, then `cols * itemsize` bytes of that row, little-endian.
#           stride = 4 + cols * itemsize. Every offset stays 2-byte aligned (32 and 8196 are even),
#           so an f16 row can be read in place.
#
# ⭐ AN ABSENT ROW EXPANDS TO ZERO, AND THAT IS WHY THIS IS SOUND. If the kernel ever reads a row the
# ids did not name, it reads zeros, the product is zero where the reference is not, and the
# comparison DIVERGES. So the encoding can produce a FALSE FAILURE but never a FALSE PASS -- it
# cannot mask a wrong gather, which is the bug class the embedding configuration exists to catch. A
# reader that expands absent rows to anything else -- uninitialised memory, a splat, a wrap-around --
# destroys that property and must not.
SPARSE_MAGIC = b"SPRSTBL1"
SPARSE_HEADER = 32


def sparse_meta(rows, cols, dtype, itemsize, n_present):
    """The format, as data. Generated beside the writer so the two cannot drift."""
    return {
        "format": SPARSE_MAGIC.decode(),
        "full_shape": [rows, cols],
        "dtype": dtype,
        "itemsize": itemsize,
        "n_present": n_present,
        "absent_rows": "zero",
        "endian": "little",
        "header": [
            {"field": "magic", "offset": 0, "size": 8, "type": "ascii",
             "value": SPARSE_MAGIC.decode()},
            {"field": "rows", "offset": 8, "size": 8, "type": "u64"},
            {"field": "cols", "offset": 16, "size": 8, "type": "u64"},
            {"field": "itemsize", "offset": 24, "size": 4, "type": "u32"},
            {"field": "n_present", "offset": 28, "size": 4, "type": "u32"},
        ],
        "records": {
            "offset": SPARSE_HEADER,
            "count": n_present,
            "stride": 4 + cols * itemsize,
            "order": "row_index ascending, each row_index distinct and < rows",
            "layout": [
                {"field": "row_index", "size": 4, "type": "u32"},
                {"field": "row", "size": cols * itemsize,
                 "type": f"{cols} x {dtype} contiguous"},
            ],
        },
    }


def write_sparse(t, dtype, path, ids):
    """Write the `SPRSTBL1` file holding only the DISTINCT rows `ids` names, and prove it back.

    The self-check is not decoration. It reads the file, expands it the way a reader must (a zero
    table with the present rows filled in) and requires that `expanded[i] == dense[i]` for EVERY id
    the kernel will gather -- so a mis-sorted record, an off-by-one stride or a duplicate dropped in
    the wrong place fails HERE rather than on card.
    """
    rows, cols = int(t.shape[0]), int(t.shape[1])
    a = t.to(_TORCH[dtype]).contiguous().numpy().astype(_NP[dtype], copy=False)
    itemsize = a.dtype.itemsize
    idx = ids.reshape(-1).to(torch.int64)
    assert int(idx.min()) >= 0 and int(idx.max()) < rows, "an id is outside the table"
    present = sorted({int(i) for i in idx.tolist()})

    with open(path, "wb") as f:
        f.write(SPARSE_MAGIC)
        f.write(struct.pack("<QQII", rows, cols, itemsize, len(present)))
        for r in present:
            f.write(struct.pack("<I", r))
            f.write(a[r].tobytes())
    n = os.path.getsize(path)
    assert n == SPARSE_HEADER + len(present) * (4 + cols * itemsize), "sparse size is wrong"

    # READ IT BACK THE WAY A READER MUST, then compare on every row the kernel gathers.
    with open(path, "rb") as f:
        blob = f.read()
    assert blob[:8] == SPARSE_MAGIC
    r2, c2, it2, np2 = struct.unpack("<QQII", blob[8:SPARSE_HEADER])
    assert (r2, c2, it2, np2) == (rows, cols, itemsize, len(present))
    expanded = np.zeros((rows, cols), dtype=a.dtype)
    off, stride, last = SPARSE_HEADER, cols * itemsize, -1
    for _ in range(np2):
        (ri,) = struct.unpack("<I", blob[off:off + 4])
        assert ri > last, "records are not ascending / a row_index repeats"
        last = ri
        expanded[ri] = np.frombuffer(blob[off + 4:off + 4 + stride], dtype=a.dtype)
        off += 4 + stride
    assert off == len(blob), "trailing bytes"
    g = idx.numpy()
    assert np.array_equal(expanded[g].view(np.uint16), a[g].view(np.uint16)), (
        "the expansion does not reproduce the gathered rows bit for bit")
    absent = np.ones(rows, bool)
    absent[np.array(present)] = False
    assert not expanded[absent].any(), "an absent row is not zero"
    print(f"      sparse: {len(present)} distinct of {rows} rows, {n} B; expansion reproduces "
          f"all {g.size} gathered rows bit for bit; every absent row is zero")
    return n, sparse_meta(rows, cols, dtype, itemsize, len(present))


def write_replicated(t, dtype, path, factor):
    """Write ONE block of a `repeat_interleave`d buffer, and prove the rule reproduces the whole.

    `full[i * factor + j] = stored[i]`. The DISCRIMINATING control is that `repeat` -- the other
    reading, `stored` tiled `factor` times -- must NOT reproduce it: the two have the same shape and
    the same element count and differ only in their values, so nothing but this comparison separates
    them. That is `rope.stage_table`'s own recorded trap.
    """
    n_full, cols = int(t.shape[0]), int(t.shape[1])
    assert n_full % factor == 0, f"{n_full} rows is not a multiple of factor {factor}"
    block = n_full // factor
    stored = t[::factor].contiguous()
    assert list(stored.shape) == [block, cols]

    bits = lambda x: x.to(torch.float16).contiguous().view(torch.int16)  # noqa: E731
    ok = torch.equal(bits(stored.repeat_interleave(factor, dim=0)), bits(t))
    assert ok, "repeat_interleave of the stored block does NOT reproduce the full buffer"
    if factor > 1 and block > 1:
        tiled = torch.equal(bits(stored.repeat(factor, 1)), bits(t))
        assert not tiled, "`repeat` also reproduces it, so the rule is not discriminating"
    n = dump(stored, dtype, path)
    print(f"      replicate: stored [{block}, {cols}] = {n} B; repeat_interleave(factor={factor}, "
          f"axis=0) reproduces [{n_full}, {cols}] BIT FOR BIT; the `repeat` reading does not")
    return n, {
        "rule": "repeat_interleave",
        "axis": 0,
        "factor": factor,
        "block": block,
        "stored_shape": [block, cols],
        "stored_bytes": n,
        "full_shape": [n_full, cols],
        "expand": "full[i * factor + j] = stored[i] for i in [0, block), j in [0, factor)",
        "not": "torch.repeat / tiling -- same shape, different values",
    }


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate(config, out_dir, fixtures_dir):
    if config not in CONFIGS:
        raise SystemExit(f"no config `{config}`; --list shows them")
    cfg = CONFIGS[config]
    mod = fixture(cfg["fixture"], fixtures_dir)
    ins, ref, note = BUILD[config](mod, cfg["ce"])
    os.makedirs(out_dir, exist_ok=True)

    entries, in_bytes, data_files = [], 0, []
    for spec in ins:
        arg, t, dtype = spec[0], spec[1], spec[2]
        enc = spec[3] if len(spec) > 3 else None
        # THE SUFFIX ANNOUNCES THE ENCODING, so a reader that ignores `meta.json`'s `sparse` /
        # `replicate` block cannot find a file at the name it guessed and fails closed.
        kind = enc[0] if enc else None
        name = f"in_{arg}." + {"sparse": "sparse", "replicate": "rep"}.get(kind, "bin")
        path = os.path.join(out_dir, name)
        e = {"arg": arg, "file": name, "shape": list(t.shape), "dtype": dtype}
        if kind == "sparse":
            n, e["sparse"] = write_sparse(t, dtype, path, enc[1])
        elif kind == "replicate":
            n, e["replicate"] = write_replicated(t, dtype, path, enc[1])
        else:
            n = dump(t, dtype, path)
        in_bytes += n
        if arg == "desc_o":
            e["zeroed"] = True
        entries.append(e)
        data_files.append(name)

    ref = ref.to(torch.float16).contiguous()
    out_bytes = dump(ref, "f16", os.path.join(out_dir, "ref_out.bin"))
    data_files.append("ref_out.bin")

    data_files.sort()
    shas = {f: sha256(os.path.join(out_dir, f)) for f in data_files}
    with open(os.path.join(out_dir, "sha256.txt"), "w") as f:
        for name in data_files:
            f.write(f"{shas[name]}  {name}\n")

    inputs_fn, reference_fn = FNS.get(config, ("inputs", "reference"))
    meta = {
        "config": config,
        "fixture": cfg["fixture"],
        "kernel": cfg["kernel"],
        "inputs_fn": inputs_fn,
        "reference_fn": reference_fn,
        "seed": SEED,
        "torch": torch.__version__,
        "generated_by": "test/numeric/gen_numeric_data.py",
        "constexprs": cfg["ce"],
        "grid": cfg["grid"],
        "inputs": entries,
        "output": {"arg": "desc_o", "file": "ref_out.bin",
                   "shape": list(ref.shape), "dtype": "f16"},
    }
    if note:
        meta["reference_note"] = note

    shared = cfg.get("data_dir")
    if shared:
        # THE BYTES LIVE IN A SIBLING DIRECTORY. Verified, not assumed: the sha of everything this
        # run produced must equal the sha the sibling recorded, or the sharing is refused.
        sib = os.path.join(os.path.dirname(os.path.abspath(out_dir)), shared)
        rec = os.path.join(sib, "sha256.txt")
        if os.path.exists(rec):
            theirs = {}
            for line in open(rec):
                h, _, nm = line.strip().partition("  ")
                theirs[nm] = h
            if theirs != shas:
                raise SystemExit(
                    f"`{config}` claims `data_dir: {shared}` but the bytes DIFFER -- refusing to "
                    f"share.\n  here:  {shas}\n  there: {theirs}\n"
                    f"Something now reaches the data that did not before; give this config its own "
                    f"bytes and drop `data_dir`.")
            print(f"      data_dir: byte-identical to `{shared}` (verified, {len(shas)} files); "
                  f"this directory keeps only meta.json + sha256.txt")
        else:
            print(f"      data_dir: `{shared}` not generated yet -- identity NOT verified this run")
        meta["data_dir"] = shared
        for nm in data_files:
            os.remove(os.path.join(out_dir, nm))
        in_bytes = out_bytes = 0

    if in_bytes > OVERSIZE_INPUT_BYTES:
        # THE BYTES ARE NOT IN THE TREE. The shapes, the sha256s and where they were staged are, so
        # a regenerated set can be CHECKED against this record rather than trusted.
        meta["oversized"] = True
        meta["staging"] = _staging(config)
    if shared or meta.get("oversized"):
        meta["sha256"] = shas
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    print(f"{config}: stored inputs={in_bytes} B ref_out={out_bytes} B "
          f"total={in_bytes + out_bytes} B oversized={meta.get('oversized', False)}"
          + (f" data_dir={shared}" if shared else ""))
    for e in entries:
        extra = ""
        if "sparse" in e:
            extra = f"  SPARSE n_present={e['sparse']['n_present']}"
        elif "replicate" in e:
            extra = f"  REPLICATED factor={e['replicate']['factor']}"
        print(f"  in  {e['arg']:12} {e['dtype']:4} {tuple(e['shape'])}  <- {e['file']}{extra}")
    print(f"  out {'desc_o':12} f16  {tuple(ref.shape)}  sha256={shas['ref_out.bin']}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--fixtures-dir", default=_FIXTURES)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k in CONFIGS:
            print(k)
        return
    if not a.config or not a.out_dir:
        raise SystemExit("use: gen_numeric_data.py <config> <out-dir>")
    generate(a.config, a.out_dir, a.fixtures_dir)


if __name__ == "__main__":
    main()

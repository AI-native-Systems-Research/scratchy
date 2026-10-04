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

"""Emit the ttir -> KTIR golden chain from the C++ toolchain, per fixture config.

Driven by `regen.sh`, which owns PYTHONPATH and the audit-chain stages. This half
runs the two PYTHON stages -- make_ttir and make_ktir -- and records a refusal
verbatim when a fixture is rejected, because reproducing the refusal BY NAME is
part of the Rust port's contract.

WHY A REFUSAL IS A GOLDEN. `vector_add` and `mul` are refused by PlanCorelets (an
even two-corelet split of ONE 64-element f16 stick is the empty [0,0] plus [0,1]),
and `bias_add_f32` fails earlier still. A port that ACCEPTS one of those is wrong,
and the only way to test that is to have the message on disk.
"""

import io
import os
import sys
import contextlib

sys.path.insert(0, "third_party/spyre/test/fixtures")

from triton._C.libtriton import ir
from triton.backends import backends
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource

SPYRE = GPUTarget("spyre", "spyre", 1)

# TWO SCALES, ON PURPOSE, because they emit DIFFERENT OP SETS and each is a real
# configuration:
#
#   1/sqrt(HEAD_DIM)  what a REAL caller passes. qk_scale = sm_scale * 1.44269504 is
#                     then 1.275630e-01, which does NOT fold, so the KTIR carries two
#                     `arith.mulf` by a splat -- the ops a port must handle.
#   1/1.44269504      makes qk_scale EXACTLY 1.0, so Triton folds the identity multiply
#                     away. That is the configuration `audit_flash_layout.sh` pins for
#                     its exactly-zero numeric bound, and the FOLD is itself a
#                     behaviour worth a golden (four fewer ops than the scaled run).
FLASH_SM_SCALE = 128 ** -0.5
FLASH_SM_SCALE_UNIT = 1.0 / 1.44269504


def flash(causal, sm_scale=FLASH_SM_SCALE):
    import attention_flash as A
    ce = A.constexprs(z=1, h=4, n_ctx=128, head_dim=128, block_m=64, gqa=2,
                      causal=causal, sm_scale=sm_scale)
    grid = (ce["N_CTX"] // ce["BLOCK_M"], 1 * 4)
    return A.attn_fwd, A.SIGNATURE, ce, grid


def swiglu():
    import swiglu_mlp as S
    ce = S.constexprs(m=64, d_model=128, d_ff=256)
    grid = (ce["M"] // ce["BLOCK_M"],)
    return S.swiglu_mlp_fwd, S.SIGNATURE, ce, grid


def elementwise(mod_name, kernel_name):
    mod = __import__(mod_name)
    ce = {"BLOCK": mod.BLOCK}
    sig = {"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp16", "n": "i32",
           "BLOCK": "constexpr"}
    return getattr(mod, kernel_name), sig, ce, (mod.N // mod.BLOCK,)


def bias_add():
    import bias_add_f32 as B
    ce = {"BLOCK": B.BLOCK}
    sig = {"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp16", "n": "i32",
           "BLOCK": "constexpr"}
    return B.bias_add_f32_kernel, sig, ce, (B.N // B.BLOCK,)


CONFIGS = {
    "attention_flash_noncausal": lambda: flash(False),
    "attention_flash_causal": lambda: flash(True),
    "attention_flash_noncausal_unitscale":
        lambda: flash(False, FLASH_SM_SCALE_UNIT),
    "swiglu_mlp": swiglu,
    "vector_add": lambda: elementwise("vector_add", "vector_add_kernel"),
    "mul": lambda: elementwise("mul", "mul_kernel"),
    "bias_add_f32": bias_add,
}


def run(name, build, out):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    for stale in ("0_ttir.mlir", "1_ktir.mlir", "refusal.txt", "stage.txt",
                  "2_layout.mlir", "3_sched.mlir", "4_groups.mlir"):
        p = os.path.join(d, stale)
        if os.path.exists(p):
            os.remove(p)

    stage = "build"
    # MLIR diagnostics go to the process's stderr, NOT into the Python exception, so
    # a refusal message is only recoverable by capturing the fd. Capture at the fd
    # level (not sys.stderr) because the emitter is C++.
    err_r, err_w = os.pipe()
    saved = os.dup(2)
    os.dup2(err_w, 2)
    captured = b""
    try:
        try:
            fn, sig, ce, grid = build()
            backend = backends["spyre"].compiler(SPYRE)
            options = backend.parse_options({"grid": grid})
            src = ASTSource(fn, signature=sig, constexprs=ce)
            ctx = ir.context()
            ir.load_dialects(ctx)
            backend.load_dialects(ctx)
            stage = "make_ir"
            mod = src.make_ir(SPYRE, options,
                              backend.get_codegen_implementation(options),
                              backend.get_module_map(), ctx)
            mod.context = ctx
            md = {}
            stage = "make_ttir"
            mod = backend.make_ttir(mod, md, options)
            ttir = str(mod)
            stage = "make_ktir"
            ktir = str(backend.make_ktir(mod, md, options))
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            os.close(err_w)
            captured = os.read(err_r, 1 << 22)
            os.close(err_r)
    except BaseException as e:  # noqa: BLE001 -- a refusal is the measurement
        msg = captured.decode("utf-8", "replace")
        if not msg.strip():
            msg = f"{type(e).__name__}: {e}"
        with open(os.path.join(d, "refusal.txt"), "w") as f:
            f.write(msg)
        with open(os.path.join(d, "stage.txt"), "w") as f:
            f.write(stage + "\n")
        if stage in ("make_ktir",) and "ttir" in dir():
            with open(os.path.join(d, "0_ttir.mlir"), "w") as f:
                f.write(ttir)
        print(f"{name}: REFUSED at {stage}: {msg.strip().splitlines()[0][:120]}")
        return

    with open(os.path.join(d, "0_ttir.mlir"), "w") as f:
        f.write(ttir)
    with open(os.path.join(d, "1_ktir.mlir"), "w") as f:
        f.write(ktir)
    with open(os.path.join(d, "grid.txt"), "w") as f:
        f.write(",".join(str(g) for g in grid) + "\n")
    print(f"{name}: ttir {ttir.count(chr(10))} lines, ktir {ktir.count(chr(10))} "
          f"lines, tt.dot={ttir.count('tt.dot')} "
          f"linalg.matmul={ktir.count('linalg.matmul')}")


def main():
    out = sys.argv[1]
    only = sys.argv[2:] or list(CONFIGS)
    for name in only:
        run(name, CONFIGS[name], out)


if __name__ == "__main__":
    main()

"""Regenerate the TTIR goldens the Rust front end (bridge one) is diffed against.

THE ORACLE, AND WHICH CHECKPOINT IS THE RIGHT ONE.

`triton.compile` reaches TTIR in two steps, and only the FIRST is bridge one's job:

    ASTSource.make_ir(...)   # code_generator.py walks the Python AST -> ttir
    backend.make_ttir(...)   # inliner, canonicalizer, ttir-combine,
                             # reorder-broadcast, cse, symbol-dce

`make_ttir` is a PASS PIPELINE (third_party/spyre/backend/compiler.py:115). Diffing a
port of `code_generator.py` against its output would demand the port also reproduce
MLIR canonicalization, constant CSE and function inlining -- none of which are AST
lowering. So this writes BOTH, distinguished by suffix:

    <name>.ttir_raw.mlir   make_ir output      <- BRIDGE ONE's oracle
    <name>.ttir.mlir       make_ttir output    <- bridge two's INPUT

Run:

    PYTHONPATH=python TRITON_BACKENDS_IN_TREE=1 python3 \
        third_party/spyre/rust/triton-frontend/tools/gen_ttir_goldens.py [outdir]

Fixtures that FAIL are reported with their exception quoted and a nonzero exit, never
skipped silently: `bias_add_f32` is expected to fail here (a module global read inside
a jitted kernel), and that expectation is ASSERTED -- if it ever starts compiling this
generator says SURPRISE and exits nonzero.
"""
import json
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", ".."))
_FIX = os.path.join(_ROOT, "third_party", "spyre", "test", "fixtures")
sys.path.insert(0, _FIX)

# The fixtures import torch only for their host-side `inputs()`/`reference()` halves,
# which this generator never calls. Stub exactly the attributes read at import time so
# goldens regenerate on a machine with no torch.
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    import types as _types

    class _Dt:
        def __init__(self, n):
            self.n = n

        def __repr__(self):
            return "torch." + self.n

    _t = _types.ModuleType("torch")
    for _n in ("float16", "float32", "bfloat16", "int8", "int32"):
        setattr(_t, _n, _Dt(_n))
    sys.modules["torch"] = _t

from triton._C.libtriton import ir  # noqa: E402
from triton.backends import backends  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler.compiler import ASTSource  # noqa: E402

SPYRE = GPUTarget("spyre", "spyre", 1)

# Fixtures known NOT to reach TTIR, with the reason.
EXPECTED_FAIL = {
    "bias_add_f32": "module global read inside a @triton.jit kernel",
}


def cases():
    import attention_flash as A
    import swiglu_mlp as S
    import vector_add as V
    import mul as M
    import bias_add_f32 as B
    import embedding as E
    import rmsnorm as R
    import rope as P
    import decoder_block as DB

    out = []
    for mod, kern, name in ((V, "vector_add_kernel", "vector_add"),
                            (M, "mul_kernel", "mul"),
                            (B, "bias_add_f32_kernel", "bias_add_f32")):
        out.append(dict(
            name=name, fn=getattr(mod, kern),
            signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp16",
                       "n": "i32", "BLOCK": "constexpr"},
            constexprs={"BLOCK": mod.BLOCK},
            grid=(mod.N // mod.BLOCK,),
        ))
    for cfgname, kw in (("swiglu_mlp", dict(m=64, d_model=128, d_ff=256)),
                        ("swiglu_mlp_granite", dict(m=64, **S.GRANITE)),
                        ("swiglu_mlp_tiledk", dict(m=64, d_model=128, d_ff=256,
                                                   block_k=64))):
        ce = S.constexprs(**kw)
        out.append(dict(name=cfgname, fn=S.swiglu_mlp_fwd, signature=S.SIGNATURE,
                        constexprs=ce, grid=(ce["M"] // ce["BLOCK_M"],)))
    for cfgname, kw in (("embedding", dict(n_tok=256, v=512, d_model=128)),
                        ("embedding_granite", dict(n_tok=256, **E.GRANITE))):
        ce = E.constexprs(**kw)
        out.append(dict(name=cfgname, fn=E.embedding_fwd, signature=E.SIGNATURE,
                        constexprs=ce, grid=(ce["N_TOK"] // ce["BLOCK_M"],)))
    for cfgname, kw in (("rmsnorm", dict(m=64, d_model=128)),
                        ("rmsnorm_granite", dict(m=64, **R.GRANITE))):
        ce = R.constexprs(**kw)
        out.append(dict(name=cfgname, fn=R.rmsnorm_fwd, signature=R.SIGNATURE,
                        constexprs=ce, grid=(ce["M"] // ce["BLOCK_M"],)))
    # `rope`'s three configurations differ ONLY in the head count of the plane being
    # rotated: 4 for the small one, then Granite's 32 query heads and 8 kv heads, which are
    # two launches of the same source (its delta 6).
    #
    # ⛔ THE GRID IS ONE WORK ITEM PER TOKEN POSITION -- `(N_TOK,)` -- AND IT IS ONE
    # DIMENSION. This read `(ce["N_TOK"] // ce["BLOCK_M"], h)` and that is the OLD
    # head-major kernel's launch: `rope.constexprs()` has no `BLOCK_M` at all any more, so
    # this generator raised `KeyError: 'BLOCK_M'` inside `cases()` and produced NO GOLDEN FOR
    # ANY FIXTURE -- which is why every rope golden in `tests/goldens/` still describes the
    # pre-token-major kernel (82 ops, two `tt.get_program_id`, `start_m`) while the fixture
    # compiles to 48 ops with one, named `pos`.
    #
    # The token-major kernel reads `pos = tl.program_id(0)` ALONE and puts all `H` heads of a
    # position in one `[H, HALF]` tile, so there is no head axis to launch. The Rust side
    # states exactly this and records what the two-dimensional grid cost:
    # `triton-ktir-superdsc/src/cases.rs`'s `rope_q32` comment -- with `vec![4, 32]` the head
    # axis became 32 REDUNDANT launches and 252 of 256 token positions were never written,
    # and it BAKED.
    for cfgname, h in (("rope", 4),
                       ("rope_granite_q", P.GRANITE_Q_HEADS),
                       ("rope_granite_kv", P.GRANITE_KV_HEADS)):
        ce = P.constexprs(h=h, n_tok=256)
        out.append(dict(name=cfgname, fn=P.rope_fwd, signature=P.SIGNATURE,
                        constexprs=ce, grid=(ce["N_TOK"],)))
    # The fusion experiment: ONE layer and the SAME layer twice, from one source. `M` is both
    # the block and the sequence length -- see the fixture's finding 2.
    dce = DB.constexprs(m=64, d_model=128, d_ff=256)
    out.append(dict(name="decoder_layer", fn=DB.decoder_layer_fwd,
                    signature=DB.ONE_LAYER_SIGNATURE, constexprs=dce, grid=(1,)))
    out.append(dict(name="decoder_two_layers", fn=DB.decoder_two_layers_fwd,
                    signature=DB.TWO_LAYER_SIGNATURE, constexprs=dce, grid=(1,)))
    for cfgname, causal in (("attention_flash_noncausal", False),
                            ("attention_flash_causal", True)):
        Z, H, N_CTX = 1, 4, 256
        ce = A.constexprs(z=Z, h=H, n_ctx=N_CTX, head_dim=128, block_m=64,
                          gqa=2, causal=causal)
        out.append(dict(name=cfgname, fn=A.attn_fwd, signature=A.SIGNATURE,
                        constexprs=ce, grid=(N_CTX // ce["BLOCK_M"], Z * H)))
    return out


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "..", "tests",
                                                                "goldens")
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    backend = backends["spyre"].compiler(SPYRE)
    index, failures, surprises = [], [], []
    for c in cases():
        name = c["name"]
        try:
            options = backend.parse_options({"grid": c["grid"]})
            src = ASTSource(c["fn"], signature=c["signature"],
                            constexprs=c["constexprs"])
            ctx = ir.context()
            ir.load_dialects(ctx)
            backend.load_dialects(ctx)
            mod = src.make_ir(SPYRE, options,
                              backend.get_codegen_implementation(options),
                              backend.get_module_map(), ctx)
            mod.context = ctx
            raw = str(mod)
            with open(os.path.join(outdir, name + ".ttir_raw.mlir"), "w") as f:
                f.write(raw)
            md = {}
            mod = backend.make_ttir(mod, md, options)
            post = str(mod)
            with open(os.path.join(outdir, name + ".ttir.mlir"), "w") as f:
                f.write(post)
            index.append(dict(name=name, grid=list(c["grid"]),
                              signature=c["signature"], constexprs=c["constexprs"],
                              kernel=c["fn"].__name__,
                              raw_lines=len(raw.splitlines()),
                              post_lines=len(post.splitlines())))
            if name in EXPECTED_FAIL:
                surprises.append("%s was expected to FAIL (%s) but compiled" % (
                    name, EXPECTED_FAIL[name]))
            print("OK    %-28s raw=%4d lines  post=%4d lines" % (
                name, len(raw.splitlines()), len(post.splitlines())))
        except Exception as e:  # noqa: BLE001 -- quoting the failure IS the point
            msg = str(e).strip()
            line = "%s: %s: %s" % (name, type(e).__name__,
                                   msg.splitlines()[0] if msg else "<no message>")
            if name in EXPECTED_FAIL:
                print("XFAIL %-28s %s" % (name, line))
                index.append(dict(name=name, xfail=EXPECTED_FAIL[name], error=line))
            else:
                print("FAIL  %-28s %s" % (name, line))
                traceback.print_exc()
                failures.append(line)
    with open(os.path.join(outdir, "index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")
    for s in surprises:
        print("SURPRISE: " + s)
    if failures or surprises:
        print("\n%d unexpected failure(s), %d surprise(s)" % (len(failures),
                                                             len(surprises)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

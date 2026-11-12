"""EXPERIMENT 1 -- Python Triton -> C++ KTIR -> our Rust SuperDSC lowering -> dxp_standalone.

THE ONE DRIVER for the whole experiment, and the producer of the checked-in golden set.
Every artefact under this directory came from this file; nothing was hand-edited.

    export DEEPTOOLS_PATH=/Users/nickm/tmp/dt_src   # required by stage 3
    PYTHONPATH=python TRITON_BACKENDS_IN_TREE=1 python3 \
        third_party/spyre/test/experiment1/run_experiment1.py

Optional first argument: a substring filter over configuration names.

THE THREE STAGES, and what each one is allowed to claim
------------------------------------------------------

1. ``ktir``    Python Triton -> C++ KTIR. Runs ``triton.compile``'s OWN stages --
               ``make_ir`` -> ``make_ttir`` -> ``make_ktir`` -- and ``make_ktir`` IS the
               C++ pass list (``add_dot_to_linalg``, ``add_ktir_lowering(grid)``,
               ``add_plan_corelets``; ``third_party/spyre/backend/compiler.py:151-157``).
               Nothing here reimplements a pass. Writes ``ktir/<name>.ttir.mlir`` and
               ``ktir/<name>.ktir.mlir``. On failure the MLIR diagnostic is captured off
               fd 2 -- the Python exception is only ``PassManager::run failed`` and says
               nothing -- and recorded verbatim, naming the pass that refused.

2. ``bundle``  C++ KTIR **text** -> SuperDSC, through
               ``third_party/spyre/rust/triton-superdsc/triton-superdsc-lower``'s
               ``emit_bundles`` example. The text boundary is the real one: it is how the
               C++ side hands off. Writes ``bundles/<name>/`` (``bundle.mlir``,
               ``sdsc_0.json``..``sdsc_N.json``, ``manifest.json``) into a FRESH directory
               -- ``emit_bundles`` does not clear its output, so a stale ``sdsc_N.json``
               from a larger earlier emission would over-count a glob census.

3. ``program`` SuperDSC bundle -> device program, ``dxp_standalone -d <dir> -b sentient``.
               Run in a scratch copy so the multi-megabyte ``spyreCodeDir/`` never lands in
               the tree. **The ARTIFACT is required**: the tool prints nothing on success
               and an abort returns 0 through a pipe, so the existence of
               ``spyreCodeDir/init_binary.bin`` is the only acceptance signal.
               ``DBO_DEBUG=1`` dumps one ``debug/sdsc_<i>/`` per SCHEDULED SuperDSC, which
               is how the in-equals-out census is taken.

WHAT THIS DRIVER MAY NOT SAY
----------------------------

Nothing here executes arithmetic. Per configuration the outcome is exactly one of
"produces a program" (with a census and a byte size) or "refuses" (with the stage and the
verbatim diagnostic). No tolerance, no ``allclose``, no correctness claim.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_FIX = os.path.join(_ROOT, "third_party", "spyre", "test", "fixtures")
_RUST = os.path.join(_ROOT, "third_party", "spyre", "rust", "triton-superdsc")
_DXP = os.path.expanduser("~/tmp/dt_src/build/dxp/dxp_standalone")
_DEEPTOOLS = os.environ.get("DEEPTOOLS_PATH", os.path.expanduser("~/tmp/dt_src"))

sys.path.insert(0, _FIX)

# The fixtures import torch for their host-side `inputs()`/`reference()` halves, which
# this driver never calls. Stub exactly the attributes read at import time so the
# experiment can be re-run on a machine with no torch; anything more raises an
# AttributeError that names itself rather than silently producing a wrong constexpr.
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


def pass_lists():
    """THE PASS LISTS, READ OUT OF `backend/compiler.py`'s SOURCE -- never retyped.

    These goldens are the ORACLE for a Rust front end, and a diff against them is
    meaningless unless the reader knows which C++ pass list produced them. Recording that
    from a hand-written string would let the record drift silently away from the passes
    that actually ran, which is the whole failure mode this is here to prevent -- so the
    driver greps the real `make_ttir` / `make_ktir` bodies and records what it finds. If a
    pass is added, removed or reordered upstream, the next regeneration says so.
    """
    import re

    src_path = os.path.join(_ROOT, "third_party", "spyre", "backend", "compiler.py")
    with open(src_path) as f:
        src = f.read()
    out = {}
    for stage in ("make_ttir", "make_ktir"):
        m = re.search(r"def %s\(.*?\n(?=    @|\Z)" % stage, src, re.S)
        if m is None:
            out[stage] = ["<could not locate %s in %s>" % (stage, src_path)]
            continue
        calls = re.findall(r"^\s+(?:spyre\.)?passes\.[\w.]+\(pm[^\n]*\)", m.group(0), re.M)
        out[stage] = [c.strip() for c in calls]
    out["source"] = "third_party/spyre/backend/compiler.py"
    return out


def git_head(produced_now):
    """Which commit the ARTEFACTS came from, kept distinct from when this ran.

    A provenance-only refresh does NOT regenerate the artefacts, so stamping it with the
    current HEAD would claim the goldens came from a commit that never produced them --
    exactly the kind of quiet overstatement a golden set must not carry. So the two are
    separate fields, and the artefact commit is DERIVED (the last commit to touch `ktir/`
    and `bundles/`) rather than assumed to be HEAD.
    """
    def git(*a):
        p = subprocess.run(("git", ) + a, cwd=_ROOT, capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None

    rel = os.path.relpath(_HERE, _ROOT)
    out = {
        "recorded_at_commit": git("rev-parse", "HEAD"),
        # A golden produced from a dirty tree is not reproducible from the commit alone.
        # Say so rather than let the commit imply more than it can.
        "tree_dirty_when_recorded": bool(git("status", "--porcelain")),
    }
    if produced_now:
        out["artifacts_produced_at_commit"] = out["recorded_at_commit"]
        out["artifacts_produced_now"] = True
    else:
        out["artifacts_produced_at_commit"] = git(
            "log", "-1", "--format=%H", "--",
            os.path.join(rel, "ktir"), os.path.join(rel, "bundles"))
        out["artifacts_produced_now"] = False
        out["note"] = ("provenance refreshed without regenerating: "
                       "`artifacts_produced_at_commit` is the last commit to touch "
                       "`ktir/` or `bundles/`, NOT this run's HEAD")
    return out


def cases():
    """Every Granite kernel configuration in the experiment, in report order.

    ``grid_point`` pins ``ktdp.get_compute_tile_id`` for the Rust lowering. It is NOT
    always 0: flash-attention causal at query block 0 has an EMPTY off-band sweep, which
    would prove nothing about causal, so it is pinned at 3 of 4.
    """
    import attention_flash as A
    import swiglu_mlp as S
    import embedding as E
    import rmsnorm as R
    import rope as P
    import decoder_block as D

    out = []

    # --- flash attention, non-causal then causal. Z=1 H=4 N_CTX=256 head_dim=128 GQA=2.
    for cfgname, causal, gp in (("attention_flash_noncausal", False, 0),
                                ("attention_flash_causal", True, 3)):
        Z, H, N_CTX = 1, 4, 256
        ce = A.constexprs(z=Z, h=H, n_ctx=N_CTX, head_dim=128, block_m=64, gqa=2,
                          causal=causal)
        out.append(dict(name=cfgname, fixture="attention_flash.py", fn=A.attn_fwd,
                        signature=A.SIGNATURE, constexprs=ce,
                        grid=(N_CTX // ce["BLOCK_M"], Z * H), grid_point=gp,
                        config="Z=1 H=4 N_CTX=256 HEAD_DIM=128 BLOCK_M=64 GQA=2 causal=%s"
                        % causal))

    # --- SwiGLU MLP at all three widths: small, Granite, tiled-K.
    for cfgname, kw, desc in (
        ("swiglu_mlp_small", dict(m=64, d_model=128, d_ff=256),
         "d_model=128 d_ff=256 BLOCK_K=128"),
        ("swiglu_mlp_granite", dict(m=64, **S.GRANITE),
         "d_model=4096 d_ff=12800 BLOCK_K=4096 (Granite)"),
        ("swiglu_mlp_tiled_k", dict(m=64, d_model=128, d_ff=256, block_k=64),
         "d_model=128 d_ff=256 BLOCK_K=64 (nested scf.for)"),
    ):
        ce = S.constexprs(**kw)
        out.append(dict(name=cfgname, fixture="swiglu_mlp.py", fn=S.swiglu_mlp_fwd,
                        signature=S.SIGNATURE, constexprs=ce,
                        grid=(ce["M"] // ce["BLOCK_M"], ), grid_point=0, config=desc))

    # --- NEW: embedding, the gather. Granite's V=49159 D_MODEL=4096.
    ce = E.constexprs(n_tok=256, **E.GRANITE)
    out.append(dict(name="embedding_granite", fixture="embedding.py", fn=E.embedding_fwd,
                    signature=E.SIGNATURE, constexprs=ce,
                    grid=(ce["N_TOK"] // ce["BLOCK_M"], ), grid_point=0,
                    config="V=49159 D_MODEL=4096 N_TOK=256 BLOCK_M=64 (Granite)"))
    # The DISCRIMINATING CONTROL for the embedding refusal. Its BLOCK_M=64 i32 index
    # vector is exactly ONE 64-element stick, and PlanCorelets' elementwise `split`
    # pattern halves a stick count -- so the refusal may be about the one-stick tile
    # rather than about the gather. Doubling BLOCK_M gives that tile two sticks and
    # separates the two causes. NOT a Granite configuration; it exists only to localise.
    ce2 = dict(ce)
    ce2["BLOCK_M"] = 128
    out.append(dict(name="embedding_granite_bm128_control", fixture="embedding.py",
                    fn=E.embedding_fwd, signature=E.SIGNATURE, constexprs=ce2,
                    grid=(ce2["N_TOK"] // ce2["BLOCK_M"], ), grid_point=0,
                    config="V=49159 D_MODEL=4096 N_TOK=256 BLOCK_M=128 "
                           "(CONTROL for the one-stick index tile, not a Granite width)"))

    # --- NEW: RMSNorm at Granite's hidden size.
    ce = R.constexprs(m=64, **R.GRANITE)
    out.append(dict(name="rmsnorm_granite", fixture="rmsnorm.py", fn=R.rmsnorm_fwd,
                    signature=R.SIGNATURE, constexprs=ce, grid=(1, ), grid_point=0,
                    config="M=64 D_MODEL=4096 EPS=1e-05 (Granite)"))

    # --- NEW: RoPE, both Granite head counts (32 query heads, 8 key/value heads).
    for cfgname, h in (("rope_q32", P.GRANITE_Q_HEADS), ("rope_kv8", P.GRANITE_KV_HEADS)):
        ce = P.constexprs(h=h, n_tok=256)
        out.append(dict(name=cfgname, fixture="rope.py", fn=P.rope_fwd,
                        signature=P.SIGNATURE, constexprs=ce,
                        grid=(ce["N_TOK"] // ce["BLOCK_M"], h), grid_point=0,
                        config="H=%d N_TOK=256 HEAD_DIM=128 HALF=64 BLOCK_M=64" % h))

    # --- NEW: the decoder block, one layer then the two-layer fused kernel.
    ce = D.constexprs()
    out.append(dict(name="decoder_layer_one", fixture="decoder_block.py",
                    fn=D.decoder_layer_fwd, signature=D.ONE_LAYER_SIGNATURE,
                    constexprs=ce, grid=(1, ), grid_point=0,
                    config="M=64 D_MODEL=128 D_FF=256 BLOCK_N=64 (one layer)"))
    out.append(dict(name="decoder_two_layers", fixture="decoder_block.py",
                    fn=D.decoder_two_layers_fwd, signature=D.TWO_LAYER_SIGNATURE,
                    constexprs=ce, grid=(1, ), grid_point=0,
                    config="M=64 D_MODEL=128 D_FF=256 BLOCK_N=64 (two layers, fused)"))
    return out


class _Fd2:
    """Capture C++ fd 2 around a call.

    The MLIR diagnostics that say WHICH PASS refused go to the process's fd 2; the Python
    exception is a bare ``PassManager::run failed``. Recording only the exception would
    turn a named refusal into an anonymous one, so the real stream is captured.
    """

    def __enter__(self):
        self.tmp = tempfile.TemporaryFile(mode="w+")
        self.text = ""
        sys.stderr.flush()
        self.saved = os.dup(2)
        os.dup2(self.tmp.fileno(), 2)
        return self

    def read(self):
        """Readable from INSIDE the block -- the diagnostic has to be recovered while the
        exception is still being handled, i.e. before __exit__ has run."""
        os.fsync(2)
        pos = self.tmp.tell()
        self.tmp.seek(0)
        self.text = self.tmp.read()
        self.tmp.seek(pos)
        return self.text

    def __exit__(self, *a):
        self.read()
        os.dup2(self.saved, 2)
        os.close(self.saved)
        self.tmp.close()
        return False


def _diag(text):
    """The refusal lines out of a captured MLIR diagnostic stream, in order."""
    keep = [ln.strip() for ln in text.splitlines()
            if (": error:" in ln or ": note:" in ln) and "reproducer" not in ln]
    return keep or [ln.strip() for ln in text.splitlines() if ln.strip()][:6]


def stage_ktir(backend, c, ktir_dir):
    """Stage 1. Returns (ok, record). Writes the TTIR and KTIR goldens on success."""
    rec = {"stage": "ktir"}
    options = backend.parse_options({"grid": c["grid"]})
    src = ASTSource(c["fn"], signature=c["signature"], constexprs=c["constexprs"])
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    phase = "make_ir"
    ttir = None
    with _Fd2() as cap:
        try:
            mod = src.make_ir(SPYRE, options, backend.get_codegen_implementation(options),
                              backend.get_module_map(), ctx)
            mod.context = ctx
            md = {}
            phase = "make_ttir"
            mod = backend.make_ttir(mod, md, options)
            ttir = str(mod)
            phase = "make_ktir"
            mod = backend.make_ktir(mod, md, options)
            ktir = str(mod)
        except Exception as e:  # noqa: BLE001
            rec["refused_at"] = phase
            rec["exception"] = " ".join(str(e).split())[:400]
            rec["diagnostic"] = _diag(cap.read())
            # KEEP THE TTIR A `make_ktir` REFUSAL WOULD OTHERWISE THROW AWAY. The TTIR
            # leg is usable golden material on its own -- experiment 2 can be diffed at
            # that checkpoint even for a configuration whose C++ KTIR does not exist --
            # and discarding it would make the refusal look like nothing was produced.
            if ttir is not None:
                with open(os.path.join(ktir_dir, c["name"] + ".ttir.mlir"), "w") as f:
                    f.write(ttir)
                rec["ttir_lines"] = ttir.count("\n")
                rec["ttir_bytes"] = len(ttir)
                rec["ttir_kept"] = True
            return False, rec
    with open(os.path.join(ktir_dir, c["name"] + ".ttir.mlir"), "w") as f:
        f.write(ttir)
    with open(os.path.join(ktir_dir, c["name"] + ".ktir.mlir"), "w") as f:
        f.write(ktir)
    rec["ttir_lines"] = ttir.count("\n")
    rec["ktir_lines"] = ktir.count("\n")
    rec["ttir_bytes"] = len(ttir)
    rec["ktir_bytes"] = len(ktir)
    rec["ok"] = True
    return True, rec


def build_lowering():
    """Build `emit_bundles` ONCE, before any kernel runs.

    A BUILD FAILURE IS NOT A REFUSAL, and conflating the two is how a broken crate reads
    as "every kernel refuses". Measured: while the lowering crate was mid-migration it
    did not compile, and because the per-kernel `cargo run` reported a non-zero exit the
    driver recorded twelve refusals whose diagnostic was `could not compile
    triton-superdsc-lower (lib) due to 151 previous errors`. Every one of those was a
    lie about the kernel. So the build is a separate, up-front, fatal step.
    """
    cmd = ["cargo", "build", "--offline", "--quiet", "--example", "emit_bundles"]
    p = subprocess.run(cmd, cwd=_RUST, capture_output=True, text=True)
    if p.returncode != 0:
        return [ln.strip() for ln in p.stderr.splitlines() if ln.strip()][-15:]
    return None


def stage_bundle(c, ktir_path, bundle_dir):
    """Stage 2. Returns (ok, record).

    Emits into a FRESH SCRATCH directory and only then replaces `bundle_dir`. Two reasons,
    both measured: `emit_bundles` does not clear its output, so a stale `sdsc_N.json` from
    a larger earlier emission would over-count a glob census; and clearing `bundle_dir`
    up front DESTROYS A CHECKED-IN GOLDEN when the run then fails for a reason that has
    nothing to do with this kernel (a crate that does not build, an interrupted run). A
    golden set that a failed re-run can delete is not a golden set.
    """
    rec = {"stage": "bundle"}
    scratch = tempfile.mkdtemp(prefix="exp1_emit_" + c["name"] + "_")
    cmd = ["cargo", "run", "--offline", "--quiet", "--example", "emit_bundles", "--",
           ktir_path, str(c["grid_point"]), scratch]
    p = subprocess.run(cmd, cwd=_RUST, capture_output=True, text=True)
    rec["command"] = " ".join(cmd[:-1] + ["<out_dir>"])
    if p.returncode != 0:
        rec["refused_at"] = "triton-superdsc-lower"
        rec["diagnostic"] = [ln.strip() for ln in p.stderr.splitlines() if ln.strip()][-8:]
        # FAIL CLOSED, visibly: a refusal leaves NO directory, not an empty one that
        # could be mistaken for a bundle whose files were forgotten.
        rec["emitted_files"] = sorted(os.listdir(scratch))
        shutil.rmtree(scratch, ignore_errors=True)
        shutil.rmtree(bundle_dir, ignore_errors=True)
        return False, rec
    if os.path.isdir(bundle_dir):
        shutil.rmtree(bundle_dir)
    shutil.move(scratch, bundle_dir)
    census, files, notes = {}, [], []
    op_funcs = []
    mode = None
    for ln in p.stdout.splitlines():
        s = ln.strip()
        if ln.startswith("  ") and mode == "files":
            files.append(s)
        elif ln.startswith("  ") and mode == "op_funcs":
            op_funcs.append(s.split("] ", 1)[1] if "] " in s else s)
        elif ln.startswith("  - ") and mode == "notes":
            notes.append(s[2:].strip())
        elif ":" in ln and not ln.startswith(" "):
            k, v = [x.strip() for x in ln.split(":", 1)]
            census[k] = v
            mode = {"files": "files", "op_funcs": "op_funcs", "notes": "notes"}.get(k)
    rec["kernel"] = census.get("kernel")
    rec["sdsc_count"] = int(census.get("sdsc_count", "-1"))
    rec["op_funcs"] = op_funcs
    rec["op_func_histogram"] = {k: op_funcs.count(k) for k in sorted(set(op_funcs))}
    rec["files"] = sorted(os.listdir(bundle_dir))
    rec["file_bytes"] = {n: os.path.getsize(os.path.join(bundle_dir, n))
                         for n in rec["files"]}
    rec["notes"] = notes
    rec["ok"] = True
    return True, rec


def stage_program(c, bundle_dir, prog_dir):
    """Stage 3. Returns (ok, record). The ARTIFACT is required, not the exit code."""
    rec = {"stage": "program"}
    scratch = tempfile.mkdtemp(prefix="exp1_" + c["name"] + "_")
    for n in os.listdir(bundle_dir):
        if n == "bundle.mlir" or (n.startswith("sdsc_") and n.endswith(".json")):
            shutil.copy(os.path.join(bundle_dir, n), scratch)
    env = dict(os.environ)
    env["DEEPTOOLS_PATH"] = _DEEPTOOLS
    env["DBO_DEBUG"] = "1"
    env.pop("FLEX_COMPUTE", None)
    cmd = [_DXP, "-d", scratch, "-b", "sentient"]
    p = subprocess.run(cmd, cwd=scratch, capture_output=True, text=True, env=env)
    rec["command"] = ("DEEPTOOLS_PATH=%s DBO_DEBUG=1 %s -d <dir> -b sentient"
                      % (_DEEPTOOLS, _DXP))
    rec["exit_code"] = p.returncode
    code = os.path.join(scratch, "spyreCodeDir")
    binpath = os.path.join(code, "init_binary.bin")
    if not os.path.isfile(binpath):
        rec["refused_at"] = "dxp_standalone -b sentient"
        tail = [ln.strip() for ln in (p.stdout + "\n" + p.stderr).splitlines()
                if ln.strip()][-12:]
        rec["diagnostic"] = tail or ["no output; spyreCodeDir/init_binary.bin absent"]
        shutil.rmtree(scratch, ignore_errors=True)
        return False, rec
    rec["artifact_files"] = {}
    for n in sorted(os.listdir(code)):
        path = os.path.join(code, n)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            data = f.read()
        rec["artifact_files"][n] = {"bytes": len(data),
                                    "sha256": hashlib.sha256(data).hexdigest()}
    dbg = os.path.join(scratch, "debug")
    scheduled = 0
    if os.path.isdir(dbg):
        scheduled = len([n for n in os.listdir(dbg) if n.startswith("sdsc_")])
    rec["sdsc_scheduled"] = scheduled
    rec["init_binary_bytes"] = rec["artifact_files"]["init_binary.bin"]["bytes"]
    rec["ok"] = True
    os.makedirs(prog_dir, exist_ok=True)
    shutil.rmtree(scratch, ignore_errors=True)
    return True, rec


def provenance_block(produced_now):
    """What every configuration's golden records about how it was produced."""
    return {
        "pass_lists": pass_lists(),
        "git": git_head(produced_now),
        "note": "These KTIR goldens are the ORACLE for a Rust front end. `make_ktir` IS "
                "the C++ pass list recorded in `pass_lists`, taken from that file's "
                "source rather than retyped; nothing in this driver reimplements a pass. "
                "A diff against `ktir/<name>.ktir.mlir` is only meaningful against the "
                "same pass list.",
    }


def rewrite_provenance_only():
    """Refresh `provenance` in the existing index.json and touch NOTHING else.

    Adding provenance must not require re-running the three stages: a regeneration is a
    much bigger event than a metadata fix, and the artefacts are the deliverable. This
    rewrites exactly the one key, in place, and leaves every `.mlir`, every `sdsc_N.json`,
    every `bundle.mlir`, every `manifest.json` and every recorded census, byte size and
    sha256 untouched.
    """
    path = os.path.join(_HERE, "index.json")
    if not os.path.isfile(path):
        print("FATAL: no index.json to update at %s" % path)
        return 2
    with open(path) as f:
        index = json.load(f)
    prov = provenance_block(produced_now=False)
    for e in index:
        e["provenance"] = prov
    with open(path, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")
    print("provenance written to %d configurations in %s" % (len(index), path))
    for stage, calls in prov["pass_lists"].items():
        if stage == "source":
            continue
        print("  %s:" % stage)
        for c in calls:
            print("    " + c)
    g = prov["git"]
    print("  artifacts produced at : %s" % g["artifacts_produced_at_commit"])
    print("  provenance recorded at: %s (tree dirty=%s)"
          % (g["recorded_at_commit"], g["tree_dirty_when_recorded"]))
    return 0


def run_ktir_only(filt):
    """STAGE 1 ONLY: regenerate the C++ KTIR and touch no SuperDSC artefact.

    The SuperDSC lowering moved out of this effort's scope, and it is under active edit
    elsewhere. Running stage 2 now would bake another session's in-flight state into a
    checked-in golden set -- the same reason a provenance refresh does not regenerate.
    So this path writes `ktir/*.mlir`, merges only the `ktir` stage record into
    `index.json`, and leaves every `bundles/` file, every recorded census, every byte size
    and every program `sha256` exactly as they are.

    A configuration whose KTIR now succeeds where the set recorded a refusal gets the
    outcome `KTIR ONLY`, NOT `PROGRAM`: stages 2 and 3 were not run for it, and an outcome
    that implied they had would be the same overstatement as stamping a provenance refresh
    with the current HEAD.
    """
    path = os.path.join(_HERE, "index.json")
    ktir_dir = os.path.join(_HERE, "ktir")
    os.makedirs(ktir_dir, exist_ok=True)
    with open(path) as f:
        index = json.load(f)
    byname = {e["name"]: e for e in index}
    backend = backends["spyre"].compiler(SPYRE)
    prov = provenance_block(produced_now=True)
    rc = 0
    for c in cases():
        if filt and filt not in c["name"]:
            continue
        ok, rec = stage_ktir(backend, c, ktir_dir)
        e = byname.setdefault(c["name"], {"name": c["name"]})
        prev = e.get("outcome", "")
        e["stages"] = e.get("stages", {})
        e["stages"]["ktir"] = rec
        # Provenance is PER-CONFIGURATION, and a stage-1-only run produced only stage 1.
        # Copying a whole-run provenance block here would claim this run also produced the
        # bundle and the program, which for a configuration that kept its earlier ones is
        # false. So record which stage this run actually wrote, and keep any older block
        # that covers the stages it did not touch.
        pc = dict(prov)
        pc["git"] = dict(prov["git"])
        pc["stages_produced_by_this_run"] = ["ktir"]
        prior = e.get("provenance", {}).get("git", {})
        if prior.get("artifacts_produced_at_commit"):
            pc["git"]["earlier_stages_produced_at_commit"] = (
                prior["artifacts_produced_at_commit"])
        e["provenance"] = pc
        if not ok:
            e["outcome"] = "REFUSED at %s" % rec["refused_at"]
            print("REFUSED  %-34s %-9s %s"
                  % (c["name"], rec["refused_at"], rec["diagnostic"][0][:120]))
            rc = 1
            continue
        # Stage 1 passed. Drop any stale stage-2/3 refusal recorded for it, but never
        # invent a stage-2/3 result: say plainly that they were not run in this scope.
        if "PROGRAM" not in prev:
            for dead in ("bundle", "program"):
                e["stages"].pop(dead, None)
            e.pop("census", None)
            e["outcome"] = "KTIR ONLY"
            e["scope_note"] = (
                "stage 1 (C++ KTIR) only: the SuperDSC lowering moved out of this "
                "effort's scope, so stages 2 and 3 were NOT run for this configuration. "
                "Its `ktir/*.mlir` is a full golden; the absence of a bundle here is a "
                "scope boundary, not a refusal.")
        print("KTIR OK  %-34s ttir=%4dL ktir=%4dL   (was: %s)"
              % (c["name"], rec["ttir_lines"], rec["ktir_lines"], prev or "new"))
    order = [x["name"] for x in cases()]
    index = [byname[n] for n in order if n in byname]
    with open(path, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")
    return rc


def main():
    args = [a for a in sys.argv[1:]]
    if "--provenance-only" in args:
        return rewrite_provenance_only()
    if "--ktir-only" in args:
        rest = [a for a in args if a != "--ktir-only"]
        return run_ktir_only(rest[0] if rest else "")
    filt = args[0] if args else ""
    ktir_dir = os.path.join(_HERE, "ktir")
    bundles_dir = os.path.join(_HERE, "bundles")
    prog_dir = os.path.join(_HERE, "programs")
    for d in (ktir_dir, bundles_dir, prog_dir):
        os.makedirs(d, exist_ok=True)
    if not os.path.isfile(_DXP):
        print("FATAL: no dxp_standalone at %s" % _DXP)
        return 2
    build_err = build_lowering()
    if build_err is not None:
        print("FATAL: the lowering crate does not build. This is NOT a per-kernel "
              "refusal and no golden was touched:")
        for ln in build_err:
            print("  " + ln)
        return 2
    backend = backends["spyre"].compiler(SPYRE)
    provenance = provenance_block(produced_now=True)
    index, rc = [], 0
    for c in cases():
        if filt and filt not in c["name"]:
            continue
        entry = {"name": c["name"], "fixture": c["fixture"],
                 "kernel": c["fn"].__name__, "configuration": c["config"],
                 "grid": list(c["grid"]), "grid_point": c["grid_point"],
                 "signature": dict(c["signature"]),
                 "constexprs": {k: repr(v) if isinstance(v, float) else v
                                for k, v in c["constexprs"].items()},
                 "provenance": provenance,
                 "stages": {}}
        ok, rec = stage_ktir(backend, c, ktir_dir)
        entry["stages"]["ktir"] = rec
        if not ok:
            entry["outcome"] = "REFUSED at %s" % rec["refused_at"]
            print("REFUSED %-34s %-9s %s" % (c["name"], rec["refused_at"],
                                             rec["diagnostic"][0][:150]))
            index.append(entry)
            continue
        ktir_path = os.path.join(ktir_dir, c["name"] + ".ktir.mlir")
        ok, rec = stage_bundle(c, ktir_path, os.path.join(bundles_dir, c["name"]))
        entry["stages"]["bundle"] = rec
        if not ok:
            entry["outcome"] = "REFUSED at %s" % rec["refused_at"]
            print("REFUSED %-34s %-9s %s" % (c["name"], "lower", rec["diagnostic"][-1][:150]))
            index.append(entry)
            continue
        n_in = rec["sdsc_count"]
        ok, rec = stage_program(c, os.path.join(bundles_dir, c["name"]), prog_dir)
        entry["stages"]["program"] = rec
        if not ok:
            entry["outcome"] = "REFUSED at %s" % rec["refused_at"]
            print("REFUSED %-34s %-9s in=%d %s"
                  % (c["name"], "dxp", n_in, rec["diagnostic"][-1][:120]))
            index.append(entry)
            continue
        n_out = rec["sdsc_scheduled"]
        entry["census"] = {"sdsc_in": n_in, "sdsc_scheduled": n_out,
                           "in_equals_out": n_in == n_out}
        entry["outcome"] = "PROGRAM"
        if n_in != n_out:
            rc = 1
            entry["outcome"] = "PROGRAM but CENSUS MISMATCH"
        print("PROGRAM %-34s in=%-3d out=%-3d init_binary=%9d B  %s"
              % (c["name"], n_in, n_out, rec["init_binary_bytes"],
                 "" if n_in == n_out else "*** CENSUS MISMATCH ***"))
        index.append(entry)
    path = os.path.join(_HERE, "index.json")
    if filt:
        # A filtered run must not truncate the checked-in index: merge by name.
        old = json.load(open(path)) if os.path.isfile(path) else []
        byname = {e["name"]: e for e in old}
        for e in index:
            byname[e["name"]] = e
        order = [c["name"] for c in cases()]
        index = [byname[n] for n in order if n in byname]
    with open(path, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

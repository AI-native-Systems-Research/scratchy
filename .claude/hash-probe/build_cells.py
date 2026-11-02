#!/usr/bin/env python3
"""B1 card-run cell builder (w2048 geometry, nickm4).

Base: the REAL baked one-op w2048 cell pulled from the pod (`w2048/src_a_w2048.json`
+ its placements) — `mul` at `[mb=256, out=2048]` fp16, 32 cores, 524,288 lanes.
`dxp_standalone` accepted this schema (the 549 one-op set is the schema dxp reads),
so every mutation below is a minimal spelling change on a known-good base.

Legs, each a dir under `cells/` (sdsc_0.json + bundle.mlir + placements.json):

* control — the cell VERBATIM, run through the ORIGINAL IO path (IEEE→SEN staging,
  SEN→IEEE readback, stick map stated per tensor). Proves dxp + launch + readback
  on this exact geometry before any integer spelling is judged. Inputs are small
  integers (< 22) whose products (< 512) are EXACT in both SEN169 and IEEE fp16,
  so the host comparison is bitwise.
* mul_g1 / add_g1 — MINIMAL mutation: opFuncName + the integer spelling
  (labeledDs_ SENUINT32 at wordLength 4, stickSize 32 elements, computeOp
  attributes IEEE_FP32). Every fold and every start address left verbatim.
* mul_g2 / add_g2 — FOLDS-CONSISTENT: the minimal mutation PLUS the out
  coordinates restated for a 32-element stick (elem_arr_0 64→32, elem_arr_1
  32→64, its stride alpha 64→32; 2048 elements per core kept). The g1/g2
  divergence is the datum on whether dxp recomputes arrays from N_ + stickSize
  or takes the folds verbatim.
* i2f_g1 — int32tofp32, UNARY: Tensor0 (SENUINT32) → Tensor2 (IEEE_FP32);
  Tensor1 stays allocated but leaves the op's input list.
* log_g1 — log, UNARY: Tensor0 (IEEE_FP32) → Tensor2 (IEEE_FP32). The DDL was
  not probed for this name at compile time; a dxp refusal here is a datum, not
  a gate.

Inputs (deterministic, regenerated identically by check_cells.py via import):
u32 patterns exercising wraparound (near ±2^31, small ints, full-range random),
f32 log-domain values, and the control's exact small-integer fp16s.
"""
import copy
import json
import pathlib
import random
import struct
import sys

HERE = pathlib.Path(__file__).resolve().parent
CELLS = HERE / "cells"
W2048 = HERE / "w2048"

N_LANES = 524_288          # 32 cores x 2048 out x 8 mb
U32_BYTES = N_LANES * 4    # 2,097,152 B per tensor at wordLength 4
F16_BYTES = N_LANES * 2    # 1,048,576 B per tensor at wordLength 2

BUNDLE = (
    "module {\n"
    "  func.func @sdsc_bundle() {\n"
    '    sdscbundle.sdsc_execute () {sdsc_filename="sdsc_0.json"}\n'
    "    return\n"
    "  }\n"
    "}\n"
)


def load_base() -> dict:
    return json.loads((W2048 / "src_a_w2048.json").read_text())


def inner(doc: dict) -> dict:
    return next(iter(next(iter(doc.values()))["dscs_"][0].values()))


# ── deterministic input generators (check_cells.py imports these verbatim) ──

def gen_u32(seed: int, n: int = N_LANES) -> bytes:
    """Wraparound-exercising u32 patterns: near ±2^31, small ints, full-range."""
    r = random.Random(seed)
    out = bytearray()
    for i in range(n):
        m = i % 8
        if m == 0:
            v = 0x7FFFFFF0 + (i & 0xF)       # near INT32_MAX
        elif m == 1:
            v = 0x80000000 + (i & 0xF)       # near INT32_MIN
        elif m == 2:
            v = i & 0xFF                     # small positive
        elif m == 3:
            v = 0xFFFFFF00 + (i & 0xF)       # near -1
        elif m == 4:
            v = 0x40000000 + (i & 0xFFFF)    # ~2^30
        elif m == 5:
            v = (i & 0xF) | 0x80000000       # small negative
        else:
            v = r.getrandbits(32)            # full-range random
        out += struct.pack("<I", v)
    return bytes(out)


def gen_f32_log(seed: int, n: int = N_LANES) -> bytes:
    """Positive f32 across exponents 2^-40..2^20 — the log domain, no 0/inf/nan."""
    r = random.Random(seed)
    out = bytearray()
    for i in range(n):
        e = -40 + (i % 61)
        m = 1.0 + r.random()
        out += struct.pack("<f", m * (2.0 ** e))
    return bytes(out)


def gen_f16_control(seed: int, n: int = N_LANES) -> bytes:
    """Integers < 22: products < 512 are EXACT in SEN169 (9-bit mantissa) and fp16."""
    r = random.Random(seed)
    return b"".join(struct.pack("<e", r.randrange(22)) for _ in range(n))


# ── cell writers ──

def write_cell(name: str, doc: dict, sizes: list[int]) -> None:
    d = CELLS / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "sdsc_0.json").write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    (d / "bundle.mlir").write_text(BUNDLE)
    pl = json.loads((W2048 / "src_a_w2048_placements.json").read_text())
    for p, sz in zip(pl["places"], sizes):
        p["size"] = sz
    pl["segment_bytes"][0] = max(p["offset"] + p["size"] for p in pl["places"])
    (d / "placements.json").write_text(json.dumps(pl, indent=2) + "\n")
    print(f"{name}: wrote ({sizes[0]} B tensors, seg0 {pl['segment_bytes'][0]} B)")


def set_int_spelling(node: dict, op: str, out_fmt: str = "SENUINT32") -> None:
    """The minimal integer mutation: op name, IEEE_FP32 compute attrs, wl-4 labeledDs,
    32-element sticks. Folds and start addresses untouched."""
    cop = node["computeOp_"][0]
    assert cop["opFuncName"] == "mul", cop["opFuncName"]
    cop["opFuncName"] = op
    cop["attributes_"]["dataFormat_"] = "IEEE_FP32"
    for ds in node["labeledDs_"]:
        assert ds["wordLength"] == 2 and ds["dataFormat_"] == "SEN169_FP16", ds["dsName_"]
        ds["wordLength"] = 4
        ds["dataFormat_"] = out_fmt if ds["dsName_"] == "Tensor2" else "SENUINT32"
    pds = node["primaryDsInfo_"]["OUTPUT"]
    assert pds["stickSize_"] == [64], pds["stickSize_"]
    pds["stickSize_"] = [32]


def set_out_folds_u32(node: dict) -> None:
    """The folds-consistent restatement: out elem_arr_0 64→32 (the 32-elem stick),
    elem_arr_1 32→64 (per-core out stays 2048), elem_arr_1 stride alpha 64→32."""
    for alloc in node["scheduleTree_"]:
        ci = alloc["coordinates_"]["coordInfo"]["out"]
        fa = ci["folds"]["dim_prop_attr"]
        ff = ci["folds"]["dim_prop_func"]
        assert fa[3]["label_"] == "elem_arr_1" and fa[3]["factor_"] == 32, fa[3]
        assert fa[4]["label_"] == "elem_arr_0" and fa[4]["factor_"] == 64, fa[4]
        assert ff[3]["Affine"]["alpha_"] == 64 and ff[4]["Affine"]["alpha_"] == 1
        fa[3]["factor_"] = 64
        ff[3]["Affine"]["alpha_"] = 32
        fa[4]["factor_"] = 32


def main() -> int:
    CELLS.mkdir(exist_ok=True)

    # control — verbatim
    write_cell("control", load_base(), [F16_BYTES] * 3)

    # binary legs — g1 minimal, g2 folds-consistent; SAME inputs across the pair
    for op, base in (("muli32toi32", "mul"), ("addi32toi32", "add")):
        for geo in ("g1", "g2"):
            doc = load_base()
            node = inner(doc)
            set_int_spelling(node, op)
            if geo == "g2":
                set_out_folds_u32(node)
            write_cell(f"{base}_{geo}", doc, [U32_BYTES] * 3)

    # i2f_g1 — unary int32tofp32
    doc = load_base()
    node = inner(doc)
    set_int_spelling(node, "int32tofp32", out_fmt="IEEE_FP32")
    node["computeOp_"][0]["inputLabeledDs"] = ["Tensor0-idx0"]
    write_cell("i2f_g1", doc, [U32_BYTES] * 3)

    # log_g1 — unary log, all-IEEE_FP32 (Tensor1 allocated but unread; uniform dtype)
    doc = load_base()
    node = inner(doc)
    set_int_spelling(node, "log", out_fmt="IEEE_FP32")
    node["labeledDs_"][0]["dataFormat_"] = "IEEE_FP32"
    node["labeledDs_"][1]["dataFormat_"] = "IEEE_FP32"
    node["computeOp_"][0]["inputLabeledDs"] = ["Tensor0-idx0"]
    write_cell("log_g1", doc, [U32_BYTES] * 3)

    # ── inputs ──
    (CELLS / "in0_control.bin").write_bytes(gen_f16_control(31))
    (CELLS / "in1_control.bin").write_bytes(gen_f16_control(37))
    for base, sa, sb in (("mul", 11, 13), ("add", 17, 19)):
        a, b = gen_u32(sa), gen_u32(sb)
        for geo in ("g1", "g2"):
            (CELLS / f"in0_{base}_{geo}.bin").write_bytes(a)
            (CELLS / f"in1_{base}_{geo}.bin").write_bytes(b)
    (CELLS / "in0_i2f_g1.bin").write_bytes(gen_u32(23))
    (CELLS / "in0_log_g1.bin").write_bytes(gen_f32_log(29))
    print("inputs written")

    for name in ("control", "mul_g1", "mul_g2", "add_g1", "add_g2", "i2f_g1", "log_g1"):
        json.loads((CELLS / name / "sdsc_0.json").read_text())
    print("all seven parse")
    return 0


if __name__ == "__main__":
    sys.exit(main())

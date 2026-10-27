#!/usr/bin/env python3
"""Probe-input builder for the integer-op hardware question (Front B1).

Derives one-op sdsc inputs from the vendor's own ddc lit fixtures:

* muli32toi32 / addi32toi32: from `sdsc_add.json` (a one-computeOp add graph,
  three fp16 dataspaces), swapping the computeOp's opFuncName and restating the
  operand dataspaces as the integer spelling the vendor's own machinery uses:
  SENUINT32 at wordLength 4, stick 32 elements, and the computeOp attributes at
  IEEE_FP32 (the DDL binds both integer ops with the [%type_fp32] operation
  list — broadcast_ops.ddl:28/:30 — and the vendor's sdsc_int32tofp32.json
  carries IEEE_FP32 attributes for the same op-type list).
* int32tofp32: the vendor's own sdsc_int32tofp32.json verbatim, comment lines
  stripped (its RUN line expects clang -E preprocessing).

Each lands in its own directory as sdsc_0.json + bundle.mlir — the layout
scratchy's bake (triton-ktir-superdsc/src/bake.rs) feeds to
`dxp_standalone -d <dir> -b sentient`.
"""
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent


def strip_comments(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("//"))


def derive_binary(src: dict, op: str) -> dict:
    out = json.loads(json.dumps(src))  # deep copy
    top = next(iter(out.values()))
    dsc = top["dscs_"][0]
    inner = next(iter(dsc.values()))
    assert len(inner["computeOp_"]) == 1, "expected the one-op fixture"
    cop = inner["computeOp_"][0]
    assert cop["opFuncName"] == "add", f"unexpected base op {cop['opFuncName']}"
    cop["opFuncName"] = op
    cop["attributes_"]["dataFormat_"] = "IEEE_FP32"
    for ds in inner["labeledDs_"]:
        assert ds["wordLength"] == 2 and ds["dataFormat_"] == "SEN169_FP16", ds["dsName_"]
        ds["wordLength"] = 4
        ds["dataFormat_"] = "SENUINT32"
    pds = inner["primaryDsInfo_"]["OUTPUT"]
    assert pds["stickSize_"] == [64], pds["stickSize_"]
    pds["stickSize_"] = [32]
    return out


BUNDLE = (
    "module {\n"
    "  func.func @sdsc_bundle() {\n"
    '    sdscbundle.sdsc_execute () {sdsc_filename="sdsc_0.json"}\n'
    "    return\n"
    "  }\n"
    "}\n"
)


def main() -> int:
    add = json.loads(strip_comments((HERE / "src_add.json").read_text()))
    for op in ("muli32toi32", "addi32toi32"):
        d = HERE / op
        d.mkdir(exist_ok=True)
        (d / "sdsc_0.json").write_text(json.dumps(derive_binary(add, op), indent=2) + "\n")
        (d / "bundle.mlir").write_text(BUNDLE)
        print(f"{op}: wrote {d}/sdsc_0.json + bundle.mlir")
    i2f = HERE / "int32tofp32"
    i2f.mkdir(exist_ok=True)
    raw = strip_comments((HERE / "src_int32tofp32.json").read_text())
    json.loads(raw)  # it must parse before we ship it
    (i2f / "sdsc_0.json").write_text(raw + "\n")
    (i2f / "bundle.mlir").write_text(BUNDLE)
    print("int32tofp32: wrote verbatim (comments stripped)")
    # sanity: every emitted json parses
    for op in ("muli32toi32", "addi32toi32", "int32tofp32"):
        json.loads((HERE / op / "sdsc_0.json").read_text())
    print("all three parse")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Probe-input builder, generation 2 (Front B1).

Base: a REAL baked one-op sdsc (`mul`, from /work/549-sg-cells/a_w64 on
nickm4) — a file this card's own `dxp_standalone -b sentient` accepted
(`spyreCodeDir/` present), so the schema is exactly what the pod's dxp reads.
The vendor ddc fixtures were refused at the JSON-schema layer (scheduleTree_
fields dxp does not recognize), which said nothing about the ops.

Legs, each in its own dir as sdsc_0.json + bundle.mlir:

* control      — the cell VERBATIM. Proves the probe harness itself compiles.
* muli32toi32  — opFuncName + the integer spelling: labeledDs_ SENUINT32 at
                 wordLength 4, primaryDsInfo_ stick 32 elements, computeOp
                 attributes IEEE_FP32 (the DDL's [%type_fp32] operation list).
* addi32toi32  — same, the add spelling.
* int32tofp32  — unary: one input (SENUINT32 wl 4) -> IEEE_FP32 output,
                 attributes IEEE_FP32 — the vendor's own fixture geometry.
"""
import copy
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
BUNDLE = (
    "module {\n"
    "  func.func @sdsc_bundle() {\n"
    '    sdscbundle.sdsc_execute () {sdsc_filename="sdsc_0.json"}\n'
    "    return\n"
    "  }\n"
    "}\n"
)


def load() -> dict:
    return json.loads((HERE / "src_baked_mul.json").read_text())


def inner(doc: dict) -> dict:
    return next(iter(next(iter(doc.values()))["dscs_"][0].values()))


def write(name: str, doc: dict) -> None:
    d = HERE / name
    d.mkdir(exist_ok=True)
    (d / "sdsc_0.json").write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    (d / "bundle.mlir").write_text(BUNDLE)
    print(f"{name}: wrote")


def set_df(ds: dict, fmt: str, wl: int) -> None:
    assert ds["wordLength"] == 2 and ds["dataFormat_"] == "SEN169_FP16", ds["dsName_"]
    ds["wordLength"] = wl
    ds["dataFormat_"] = fmt


def main() -> int:
    write("control", load())

    for op in ("muli32toi32", "addi32toi32"):
        doc = load()
        node = inner(doc)
        cop = node["computeOp_"][0]
        assert cop["opFuncName"] == "mul"
        cop["opFuncName"] = op
        cop["attributes_"]["dataFormat_"] = "IEEE_FP32"
        for ds in node["labeledDs_"]:
            set_df(ds, "SENUINT32", 4)
        pds = node["primaryDsInfo_"]["OUTPUT"]
        assert pds["stickSize_"] == [64], pds["stickSize_"]
        pds["stickSize_"] = [32]
        write(op, doc)

    # int32tofp32 — UNARY: keep Tensor0 as the input (SENUINT32), Tensor2 as
    # the output (IEEE_FP32), drop Tensor1 from the op's input list.
    doc = load()
    node = inner(doc)
    cop = node["computeOp_"][0]
    cop["opFuncName"] = "int32tofp32"
    cop["attributes_"]["dataFormat_"] = "IEEE_FP32"
    cop["inputLabeledDs"] = ["Tensor0-idx0"]
    set_df(node["labeledDs_"][0], "SENUINT32", 4)
    set_df(node["labeledDs_"][1], "SENUINT32", 4)
    set_df(node["labeledDs_"][2], "IEEE_FP32", 4)
    pds = node["primaryDsInfo_"]["OUTPUT"]
    pds["stickSize_"] = [32]
    write("int32tofp32", doc)

    for name in ("control", "muli32toi32", "addi32toi32", "int32tofp32"):
        json.loads((HERE / name / "sdsc_0.json").read_text())
    print("all four parse")
    return 0


if __name__ == "__main__":
    sys.exit(main())

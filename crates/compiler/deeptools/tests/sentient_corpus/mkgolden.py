import re, sys, subprocess, pathlib, os
K=pathlib.Path("/tmp/superdsc-stage-1221857/ktir_decode_granite_3_1_2b_instruct_fp8_dynamic_per_channel")
OUT=pathlib.Path("/tmp/goldens"); OUT.mkdir(exist_ok=True)
env=dict(os.environ, PATH="/opt/ibm/spyre/deeptools/bin:"+os.environ["PATH"], SENARCH="mpw4")
def programs(text):
    """each named program module -> its inner func body, as a bare non-private func"""
    for m in re.finditer(r"^  module @(\w+) \{$", text, re.M):
        name=m.group(1); i=m.end(); depth=1; lines=text[i:].splitlines()
        body=[]
        for l in lines:
            depth += l.count("{")-l.count("}")
            body.append(l)
            if depth<=0: break
        blob="\n".join(body)
        fn=re.search(r"^(\s*)func\.func (?:private )?@\w+\(\)(.*)\{[ \t]*$", blob, re.M)
        if not fn: continue
        start=fn.end(); d=1; out=[]
        for l in blob[start:].splitlines():
            d += l.count("{")-l.count("}")
            if d<=0: break
            out.append(l[4:] if l.startswith("    ") else l)
        yield name, "func.func @dataflowProgram()"+fn.group(2)+"{\n"+"\n".join(out)+"\n}\n"
groups=sorted(K.glob("group_*/group.mlir"), key=lambda p:int(p.parent.name.split("_")[1]))
ok=parse_fail=pass_fail=0
for g in groups:
    gname=g.parent.name
    for name, src in programs(g.read_text()):
        inp=OUT/f"{gname}__{name}.dfir.mlir"; inp.write_text(src)
        r=subprocess.run(["dcc_standalone", str(inp), "-kEmitProgIR", "--mlir-disable-threading",
                          "--mlir-print-ir-after=dcc-dataflow-to-sentient"],
                         capture_output=True, text=True, env=env, timeout=300)
        blob=r.stdout+r.stderr
        if "IR Dump After DataflowToSentient" not in blob:
            (OUT/f"{gname}__{name}.FAILED.txt").write_text(blob[:4000]); 
            if "error:" in blob: parse_fail+=1
            else: pass_fail+=1
            inp.unlink(missing_ok=True); continue
        (OUT/f"{gname}__{name}.sentient.mlir").write_text(blob[blob.index("IR Dump After DataflowToSentient"):])
        ok+=1
print(f"goldens {ok}  parse-failed {parse_fail}  no-dump {pass_fail}")

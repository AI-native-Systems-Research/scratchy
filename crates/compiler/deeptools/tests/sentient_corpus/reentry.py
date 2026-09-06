import pathlib, subprocess, os, sys
G=pathlib.Path("/tmp/goldens"); env=dict(os.environ, PATH="/opt/ibm/spyre/deeptools/bin:"+os.environ["PATH"], SENARCH="mpw4")
FLAG="-kEmitProgIR=progir-format=senprog dump-progir=true"
def run(p):
    r=subprocess.run(["dcc_standalone",str(p),FLAG],capture_output=True,text=True,env=env,timeout=300)
    return r.returncode, r.stdout
same=diff=failA=failB=empty=0
diffs=[]
for dfir in sorted(G.glob("*.dfir.mlir")):
    sent=G/dfir.name.replace(".dfir.",".sentient.")
    if not sent.exists(): continue
    tmp=pathlib.Path("/tmp/_re.mlir")
    tmp.write_text("\n".join(sent.read_text().splitlines()[1:]))
    ra,a=run(dfir)
    rb,b=run(tmp)
    if ra!=0: failA+=1; continue
    if rb!=0: failB+=1; diffs.append((dfir.name,"B rc!=0")); continue
    if "START file: prog.txt" not in a: empty+=1; continue
    if a==b: same+=1
    else: diff+=1; diffs.append((dfir.name,"output differs"))
print(f"byte-identical {same}   differ {diff}   A failed {failA}   B failed {failB}   A had no program {empty}")
for n,w in diffs[:8]: print(f"   {n}: {w}")

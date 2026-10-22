import json, re, math, hashlib
import numpy as np
from collections import defaultdict, Counter
D="/tmp/claude-0/-home-user-ie105/dd09b186-5b32-5246-9add-691e2a7b029b/scratchpad/kg_prep_out/secvul_lines"
R="/home/user/ie105/experiments/exp5_line_labels/results"
rows=[json.loads(l) for l in open(f"{D}/test.jsonl")]; by={r["idx"]:r for r in rows}
def load(d): return {json.loads(l)["idx"]:np.array(json.loads(l)["line_scores"]) for l in open(f"{R}/{d}/test_scores.jsonl")}
A=load("slide_ModernBERT-base_2048_s0_conv"); B=load("slide_ModernBERT-base_2048_s0_contrast_conv")
norm=lambda s: re.sub(r"\s+","",s)
r=by[16033]; print("idx16033 project",r["project"],"n_lines",r["n_lines"],"flaw_lines",len(r["flaw_lines"]),"fixed n_lines",by[r["pair_idx"]]["n_lines"])
print("  first lines:",r["func"][:200].replace("\n","|"))
# flaw density distribution
dens=[len(r["flaw_lines"])/r["n_lines"] for r in rows if r["target"]==1 and r["flaw_lines"]]
print("flaw density: median %.3f, mean %.3f, share>0.5: %.1f%%, share flaw_lines>=20: %.1f%%"%(np.median(dens),np.mean(dens),100*np.mean(np.array(dens)>0.5),100*np.mean([len(r["flaw_lines"])>=20 for r in rows if r["target"]==1 and r["flaw_lines"]])))
print("vulnerable test fns with no flaw lines (add-only fixes):",sum(1 for r in rows if r["target"]==1 and not r["flaw_lines"]),"of",sum(r["target"] for r in rows))
# same_text recompute with exclusions
def same_text(S,excl=set(),excl_proj=set()):
    recs=[]
    for r in rows:
        if r["target"]!=1 or r["pair_idx"] not in S or not r["flaw_lines"] or r["idx"] in excl or r["project"] in excl_proj: continue
        f=by[r["pair_idx"]]; ft=defaultdict(list)
        for j,t in enumerate(f["func"].split("\n")):
            if norm(t): ft[norm(t)].append(j)
        vl=r["func"].split("\n")
        for l in r["flaw_lines"]:
            k=norm(vl[l]) if l<len(vl) else ""
            if len(k)<6 or k not in ft: continue
            j=min(ft[k],key=lambda x:abs(x-l)); a,b=S[r["idx"]][l],S[r["pair_idx"]][j]
            recs.append((r["idx"],r["project"],1.0 if a>b+1e-4 else (0.5 if abs(a-b)<=1e-4 else 0.0)))
    return recs
for name,S in (("A",A),("B",B)):
    full=same_text(S); ex=same_text(S,excl={16033}); exh=same_text(S,excl_proj={"hermes"})
    w16=[w for i,p,w in full if i==16033]
    print(f"{name}: all n={len(full)} win={100*np.mean([w for _,_,w in full]):.1f} | idx16033 n={len(w16)} win={100*np.mean(w16):.1f} | excl 16033 n={len(ex)} win={100*np.mean([w for _,_,w in ex]):.1f} | excl hermes n={len(exh)} win={100*np.mean([w for _,_,w in exh]):.1f}")
# project bootstrap of same_text excluding 16033, and function-averaged
projects=sorted({r["project"] for r in rows}); rng=np.random.default_rng(0)
def boot(recsA,recsB,n=2000,fn_avg=False):
    byp=defaultdict(list); 
    for (i,p,w),(i2,p2,w2) in zip(recsA,recsB): byp[p].append((i,w,w2))
    diffs=[];pa=[];pb=[]
    for _ in range(n):
        pick=rng.choice(projects,size=len(projects),replace=True)
        wa=[];wb=[]
        for p in pick:
            if fn_avg:
                f=defaultdict(lambda:[[],[]])
                for i,w,w2 in byp[p]: f[i][0].append(w); f[i][1].append(w2)
                wa+=[np.mean(v[0]) for v in f.values()]; wb+=[np.mean(v[1]) for v in f.values()]
            else:
                wa+=[w for _,w,_ in byp[p]]; wb+=[w2 for _,_,w2 in byp[p]]
        if wa: pa.append(np.mean(wa)); pb.append(np.mean(wb)); diffs.append(np.mean(wb)-np.mean(wa))
    return np.percentile(pa,[2.5,97.5]),np.percentile(pb,[2.5,97.5]),np.percentile(diffs,[2.5,97.5])
fa,fb=same_text(A),same_text(B)
print("boot same_text line-level (B-A):",boot(fa,fb)[2])
print("boot same_text function-averaged (B-A):",boot(fa,fb,fn_avg=True)[2])
ea,eb=same_text(A,excl={16033}),same_text(B,excl={16033})
print("boot same_text excl 16033 (B-A):",boot(ea,eb)[2], "points A=%.1f B=%.1f"%(100*np.mean([w for _,_,w in ea]),100*np.mean([w for _,_,w in eb])))
# longest-line-first control & overlap with model top1
def topk(S,scorefn):
    rec=[]
    for r in rows:
        if r["target"]!=1 or not r["flaw_lines"]: continue
        sc=scorefn(r,S[r["idx"]]); n=len(sc); fl=[l for l in r["flaw_lines"] if l<n]
        order=sorted(range(n),key=lambda i:(-sc[i],i)); rank={l:k for k,l in enumerate(order)}; first=min(rank[l] for l in fl)
        rec.append((first<1,first<5,first<10,order[0]))
    return rec
L=lambda r,sc: np.array([len(l.strip()) for l in r["func"].split("\n")],float)
ll=topk(A,L); ma=topk(A,lambda r,sc:sc); mb=topk(B,lambda r,sc:sc)
print("longest-line control: top1 %.1f top5 %.1f top10 %.1f"%tuple(100*np.mean([x[k] for x in ll]) for k in range(3)))
print("model top1 == longest line: A %.1f%% B %.1f%%"%(100*np.mean([a[3]==l[3] for a,l in zip(ma,ll)]),100*np.mean([b[3]==l[3] for b,l in zip(mb,ll)])))
# model top-5 conditional on longest-line control failing at top5
idx=[i for i,x in enumerate(ll) if not x[1]]
print("when longest-line misses top5 (n=%d): model top5 A=%.1f B=%.1f"%(len(idx),100*np.mean([ma[i][1] for i in idx]),100*np.mean([mb[i][1] for i in idx])))
idx=[i for i,x in enumerate(ll) if x[1]]
print("when longest-line hits top5 (n=%d): model top5 A=%.1f B=%.1f"%(len(idx),100*np.mean([ma[i][1] for i in idx]),100*np.mean([mb[i][1] for i in idx])))
# length-residualized ranking: rank model logit minus within-function linear fit on log line length
def resid(r,sc):
    Ln=np.log1p(L(r,sc)); ok=sc>-1e8
    if ok.sum()<3: return sc
    a,b=np.polyfit(Ln[ok],sc[ok],1); return np.where(ok,sc-(a*Ln+b),sc)
ra=topk(A,resid); rb=topk(B,resid)
print("length-residualized model: A top1/5/10 %.1f %.1f %.1f | B %.1f %.1f %.1f"%tuple([100*np.mean([x[k] for x in ra]) for k in range(3)]+[100*np.mean([x[k] for x in rb]) for k in range(3)]))
# lexical control from controls.json? check keys
c=json.load(open(f"{R}/controls.json")); print("controls keys:",list(c))
# train/test exact duplicate leakage
train_h=set()
for l in open(f"{D}/train.jsonl"):
    x=json.loads(l); train_h.add(hashlib.md5(norm(x["func"]).encode()).hexdigest())
dup=[r for r in rows if hashlib.md5(norm(r["func"]).encode()).hexdigest() in train_h]
print("test functions with identical (ws-normalised) text in train:",len(dup),"of",len(rows),"; vulnerable among them",sum(r["target"] for r in dup), Counter(r["project"] for r in dup).most_common(6))
# commit-level overlap
tc={json.loads(l)["commit_id"] for l in open(f"{D}/train.jsonl")}
print("test rows sharing commit_id with train:",sum(r["commit_id"] in tc for r in rows))

import json, re, math, numpy as np
from collections import defaultdict, Counter
D="/tmp/claude-0/-home-user-ie105/dd09b186-5b32-5246-9add-691e2a7b029b/scratchpad/kg_prep_out/secvul_lines"
R="/home/user/ie105/experiments/exp5_line_labels/results"
rows=[json.loads(l) for l in open(f"{D}/test.jsonl")]; by={r["idx"]:r for r in rows}
def load(d): return {json.loads(l)["idx"]:np.array(json.loads(l)["line_scores"]) for l in open(f"{R}/{d}/test_scores.jsonl")}
A=load("slide_ModernBERT-base_2048_s0_conv"); B=load("slide_ModernBERT-base_2048_s0_contrast_conv")
norm=lambda s: re.sub(r"\s+","",s)
print("== valid curves (selection score = top5 + same_text)")
for d in ("slide_ModernBERT-base_2048_s0_conv","slide_ModernBERT-base_2048_s0_contrast_conv"):
    r=json.load(open(f"{R}/{d}/results.json"))
    for e in r["valid_curve"]:
        v=e["valid"]; print(f"  {d[-14:]} ep{e['epoch']} top1 {v['lines_ALL']['top1']:.1f} top5 {v['lines_ALL']['top5']:.1f} top10 {v['lines_ALL']['top10']:.1f} pairs {v['pairs']['ranking_acc_ties_half']:.1f} same {v['same_text_flaw_lines']['vul_scored_higher_pct']:.1f} (n={v['same_text_flaw_lines']['n']}) score {v['lines_ALL']['top5']+v['same_text_flaw_lines']['vul_scored_higher_pct']:.1f} sp {v['function']['spearman_with_tokens']:.2f}")
print("== project overlap")
P={s:Counter(json.loads(l)["project"] for l in open(f"{D}/{s}.jsonl")) for s in ("train","valid","test")}
print("  n projects",{s:len(c) for s,c in P.items()},"train∩valid",len(set(P["train"])&set(P["valid"])),"train∩test",len(set(P["train"])&set(P["test"])),"valid∩test",len(set(P["valid"])&set(P["test"])))
print("  valid top projects",P["valid"].most_common(5)); print("  train top",P["train"].most_common(5))
print("== line 0 / signature")
V=[r for r in rows if r["target"]==1 and r["flaw_lines"]]
print("  vul fns with line0 in flaw_lines: %.1f%%"%(100*np.mean([0 in r["flaw_lines"] for r in V])))
for name,S in (("A",A),("B",B)):
    t0=[int(np.argmax(S[r["idx"]])) for r in V]
    print(f"  {name}: top1 at line0 {100*np.mean([t==0 for t in t0]):.1f}%; top1 in first 3 lines {100*np.mean([t<3 for t in t0]):.1f}%; top1 relative position median {np.median([t/len(S[r['idx']]) for t,r in zip(t0,V)]):.2f}")
print("  flaw lines relative position median %.2f"%np.median([l/r["n_lines"] for r in V for l in r["flaw_lines"]]))
# how often is top1 a flaw line AND line 0
for name,S in (("A",A),("B",B)):
    hit0=sum(1 for r in V if int(np.argmax(S[r["idx"]]))==0 and 0 in r["flaw_lines"]); hit=sum(1 for r in V if int(np.argmax(S[r["idx"]])) in r["flaw_lines"])
    print(f"  {name}: top1 hits {hit}, of which line0 {hit0} ({100*hit0/hit:.0f}%)")
print("== pairs: length in chars / tokens-proxy")
pairs=[(r,by[r["pair_idx"]]) for r in rows if r["target"]==1 and r["pair_idx"] in by]
print("  fix longer by chars %.1f%%, equal %.1f%%, shorter %.1f%%"%tuple(100*np.mean([np.sign(len(f["func"])-len(v["func"]))==s for v,f in pairs]) for s in (1,0,-1)))
def pacc(S,sel):
    w=[]
    for v,f in pairs:
        if not sel(v,f): continue
        a,b=S[v["idx"]].max(),S[f["idx"]].max(); w.append(1.0 if a>b+1e-4 else (0.5 if abs(a-b)<=1e-4 else 0.0))
    return np.array(w)
def binom_p(w):
    # sign test vs 50% ignoring ties
    n1=int((w==1).sum()); n0=int((w==0).sum()); n=n1+n0
    p=2*sum(math.comb(n,k) for k in range(0,min(n1,n0)+1))/2**n; return n1,n0,min(1,p)
for lab,sel in (("equal chars",lambda v,f:len(v["func"])==len(f["func"])),("equal lines",lambda v,f:v["n_lines"]==f["n_lines"]),("|dchars|<=20",lambda v,f:abs(len(v["func"])-len(f["func"]))<=20),("fix shorter chars",lambda v,f:len(f["func"])<len(v["func"]))):
    wa,wb=pacc(A,sel),pacc(B,sel)
    print(f"  {lab}: n={len(wa)} A={100*wa.mean():.1f} (wins/losses/p {binom_p(wa)}) B={100*wb.mean():.1f} ({binom_p(wb)})")
print("== quantisation / ties"); 
allv=np.concatenate([A[r["idx"]] for r in rows]); allv=allv[allv>-1e8]
print("  A distinct line-score values:",len(np.unique(allv)),"of",len(allv),"; typical spacing ~bf16")
print("== same_text: per-function-averaged and excl-16033 with project bootstrap (B-A)")
def st(S):
    recs=defaultdict(list)
    for r in rows:
        if r["target"]!=1 or r["pair_idx"] not in S or not r["flaw_lines"]: continue
        f=by[r["pair_idx"]]; ft=defaultdict(list)
        for j,t in enumerate(f["func"].split("\n")):
            if norm(t): ft[norm(t)].append(j)
        vl=r["func"].split("\n")
        for l in r["flaw_lines"]:
            k=norm(vl[l]) if l<len(vl) else ""
            if len(k)<6 or k not in ft: continue
            j=min(ft[k],key=lambda x:abs(x-l)); a,b=S[r["idx"]][l],S[r["pair_idx"]][j]
            recs[(r["project"],r["idx"])].append(1.0 if a>b+1e-4 else (0.5 if abs(a-b)<=1e-4 else 0.0))
    return recs
SA,SB=st(A),st(B)
projects=sorted({r["project"] for r in rows}); rng=np.random.default_rng(1)
def boot(fa,fb,keys,n=2000):
    byp=defaultdict(list)
    for k in keys: byp[k[0]].append(k)
    out=[]
    for _ in range(n):
        pick=rng.choice(projects,size=len(projects),replace=True); wa=[];wb=[]
        for p in pick:
            for k in byp.get(p,[]): wa.append(fa[k]); wb.append(fb[k])
        out.append(100*(np.mean(wb)-np.mean(wa)))
    return np.percentile(out,[2.5,50,97.5])
keys=list(SA); fa={k:np.mean(v) for k,v in SA.items()}; fb={k:np.mean(v) for k,v in SB.items()}
print("  fn-averaged: A=%.1f B=%.1f boot(B-A) %s"%(100*np.mean(list(fa.values())),100*np.mean(list(fb.values())),boot(fa,fb,keys)))
k2=[k for k in keys if k[1]!=16033]; print("  fn-averaged excl 16033: n=%d A=%.1f B=%.1f boot %s"%(len(k2),100*np.mean([fa[k] for k in k2]),100*np.mean([fb[k] for k in k2]),boot(fa,fb,k2)))
# per-function sign test fn-averaged
wb=sum(fb[k]>fa[k] for k in keys); wa=sum(fa[k]>fb[k] for k in keys); print("  fn-level sign test B>A:",wb,"A>B:",wa,"p=%.3f"%binom_p(np.array([1.0 if fb[k]>fa[k] else (0.0 if fa[k]>fb[k] else 0.5) for k in keys]))[2])
# how many of same_text lines are "flaw" only because whole function flagged (flaw density >0.5)
dense={r["idx"] for r in V if len(r["flaw_lines"])/r["n_lines"]>0.5}
print("  same_text lines coming from fns with flaw density>0.5: %d of %d"%(sum(len(v) for k,v in SA.items() if k[1] in dense),sum(len(v) for v in SA.values())))
print("== pair accuracy vs project bootstrap restricted to equal-line pairs (B-A)")
eq={(v["project"],v["idx"]):None for v,f in pairs if v["n_lines"]==f["n_lines"]}
pa={k:0 for k in eq}; pb={k:0 for k in eq}
for v,f in pairs:
    k=(v["project"],v["idx"])
    if k in eq:
        for S,dct in ((A,pa),(B,pb)):
            a,b=S[v["idx"]].max(),S[f["idx"]].max(); dct[k]=1.0 if a>b+1e-4 else (0.5 if abs(a-b)<=1e-4 else 0.0)
print("  equal-lines pairs boot(B-A):",boot(pa,pb,list(eq)))
# group sizes by n_lines for test
print("== functions by group proxy: vul fns n_lines>100: %d; flaw lines beyond line 100 only: %d"%(sum(r["n_lines"]>100 for r in V),sum(min(r["flaw_lines"])>=100 for r in V)))

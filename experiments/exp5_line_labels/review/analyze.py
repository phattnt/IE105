import json, re, math, sys
import numpy as np
from collections import defaultdict, Counter
D="/tmp/claude-0/-home-user-ie105/dd09b186-5b32-5246-9add-691e2a7b029b/scratchpad/kg_prep_out/secvul_lines"
R="/home/user/ie105/experiments/exp5_line_labels/results"
rows=[json.loads(l) for l in open(f"{D}/test.jsonl")]
by={r["idx"]:r for r in rows}
def load(d): return {json.loads(l)["idx"]:np.array(json.loads(l)["line_scores"]) for l in open(f"{R}/{d}/test_scores.jsonl")}
A=load("slide_ModernBERT-base_2048_s0_conv"); B=load("slide_ModernBERT-base_2048_s0_contrast_conv")
norm=lambda s: re.sub(r"\s+","",s)
print("n rows",len(rows),"n scores",len(A),len(B))
# ---------- per-function units
def units(S):
    out={}
    for r in rows:
        sc=S[r["idx"]]; d={"project":r["project"],"target":r["target"],"n_lines":r["n_lines"],"chars":len(r["func"])}
        d["fn"]=float(sc.max())
        if r["target"]==1 and r["flaw_lines"]:
            fl=[l for l in r["flaw_lines"] if l<len(sc)]
            order=sorted(range(len(sc)),key=lambda i:(-sc[i],i)); rank={l:k for k,l in enumerate(order)}
            first=min(rank[l] for l in fl); d.update(ifa=first,top1=first<1,top5=first<5,top10=first<10)
        if r["target"]==1 and r["pair_idx"] in S:
            fv,ff=sc.max(),S[r["pair_idx"]].max()
            d["pair"]=1.0 if fv>ff+1e-4 else (0.5 if abs(fv-ff)<=1e-4 else 0.0)
            d["tie"]=abs(fv-ff)<=1e-4
            f=by[r["pair_idx"]]; d["fix_longer"]=np.sign(f["n_lines"]-r["n_lines"])
            d["fix_chars_longer"]=np.sign(len(f["func"])-len(r["func"]))
        out[r["idx"]]=d
    return out
UA,UB=units(A),units(B)
def summ(U,name):
    v=[u for u in U.values() if "top1" in u]; p=[u for u in U.values() if "pair" in u]
    print(f"{name}: lines n={len(v)} top1={100*np.mean([u['top1'] for u in v]):.1f} top5={100*np.mean([u['top5'] for u in v]):.1f} top10={100*np.mean([u['top10'] for u in v]):.1f} | pairs n={len(p)} ties={sum(u['tie'] for u in p)} acc_half={100*np.mean([u['pair'] for u in p]):.1f}")
summ(UA,"A"); summ(UB,"B")
# ---------- length confound on pairs
print("\n== pair accuracy by relative length (fixed longer / equal / shorter, by lines)")
for name,U in (("A",UA),("B",UB)):
    p=[u for u in U.values() if "pair" in u]
    for s,lab in ((1,"fix longer"),(0,"equal"),(-1,"fix shorter")):
        q=[u["pair"] for u in p if u["fix_longer"]==s]
        print(f"  {name} {lab:12s} n={len(q):4d} acc={100*np.mean(q):.1f}")
p=[u for u in UA.values() if "pair" in u]
print("  share fix longer(lines)=%.1f%% equal=%.1f%% shorter=%.1f%%"%tuple(100*np.mean([u['fix_longer']==s for u in p]) for s in (1,0,-1)))
print("  'shorter wins' heuristic (chars) acc_half=%.1f"%(100*np.mean([1.0 if u['fix_chars_longer']>0 else (0.5 if u['fix_chars_longer']==0 else 0.0) for u in p])))
# spearman fn score vs n_lines, chars
def spearman(a,b):
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b)); return np.corrcoef(ra,rb)[0,1]
for name,U in (("A",UA),("B",UB)):
    fn=np.array([u["fn"] for u in U.values()]); nl=np.array([u["n_lines"] for u in U.values()]); ch=np.array([u["chars"] for u in U.values()])
    y=np.array([u["target"] for u in U.values()])
    print(f"  {name} spearman(fn, n_lines)={spearman(fn,nl):.3f} (fn,chars)={spearman(fn,ch):.3f}")
# within-pair: does fn score difference correlate with length difference?
for name,U in (("A",UA),("B",UB)):
    d=[];dl=[]
    for r in rows:
        if r["target"]==1 and r["pair_idx"] in U:
            d.append(U[r["idx"]]["fn"]-U[r["pair_idx"]]["fn"]); dl.append(by[r["pair_idx"]]["n_lines"]-r["n_lines"])
    d=np.array(d);dl=np.array(dl)
    print(f"  {name} spearman(fn_vul-fn_fix, lines_fix-lines_vul)={spearman(d,dl):.3f}")
# ---------- McNemar A vs B on pairs
pa=np.array([UA[i]["pair"] for i in UA if "pair" in UA[i]]); pb=np.array([UB[i]["pair"] for i in UB if "pair" in UB[i]])
b01=int(((pa<pb)).sum()); b10=int((pa>pb).sum())
print(f"\n== McNemar pairs: B better on {b01}, A better on {b10}, same {len(pa)-b01-b10}; exact-binomial p two-sided ~ {2*sum(math.comb(b01+b10,k) for k in range(0,min(b01,b10)+1))/2**(b01+b10):.4f}")
ta=np.array([UA[i]["top5"] for i in UA if "top5" in UA[i]]); tb=np.array([UB[i]["top5"] for i in UB if "top5" in UB[i]])
b01=int(((~ta)&tb).sum()); b10=int((ta&~tb).sum())
print(f"== McNemar top5: B-only {b01}, A-only {b10}; p~{2*sum(math.comb(b01+b10,k) for k in range(0,min(b01,b10)+1))/2**(b01+b10):.4f}")
ta=np.array([UA[i]["top1"] for i in UA if "top1" in UA[i]]); tb=np.array([UB[i]["top1"] for i in UB if "top1" in UB[i]])
b01=int(((~ta)&tb).sum()); b10=int((ta&~tb).sum())
print(f"== McNemar top1: B-only {b01}, A-only {b10}; p~{2*sum(math.comb(b01+b10,k) for k in range(0,min(b01,b10)+1))/2**(b01+b10):.4f}")
# ---------- project concentration
pc=Counter(u["project"] for u in UA.values() if "top1" in u)
print("\n== line-eval functions per project (top 8 of %d):"%len(pc),pc.most_common(8))
pc2=Counter(u["project"] for u in UA.values() if "pair" in u)
print("== pairs per project (top 8 of %d):"%len(pc2),pc2.most_common(8))
# ---------- line-level shortcuts: rank vs line length / position; flaw line length vs others
def line_stats(S,name):
    sp_len=[];sp_pos=[];flaw_len=[];other_len=[];top1_text=[]
    for r in rows:
        if r["target"]!=1 or not r["flaw_lines"]: continue
        sc=S[r["idx"]]; lines=r["func"].split("\n"); n=len(lines)
        if n<5: continue
        L=np.array([len(l.strip()) for l in lines]); pos=np.arange(n)
        sp_len.append(spearman(sc,L)); sp_pos.append(spearman(sc,pos))
        fl=set(r["flaw_lines"]); flaw_len+= [L[i] for i in fl if i<n]; other_len+=[L[i] for i in range(n) if i not in fl]
        top1_text.append(lines[int(np.argmax(sc))].strip())
    print(f"  {name}: median within-fn spearman(line logit, line length)={np.median(sp_len):.3f} mean={np.mean(sp_len):.3f}; (logit, position)={np.median(sp_pos):.3f} mean={np.mean(sp_pos):.3f}")
    return top1_text,flaw_len,other_len
print("\n== line-level shortcut checks")
t1A,fl_len,ot_len=line_stats(A,"A"); t1B,_,_=line_stats(B,"B")
print(f"  flaw line median length={np.median(fl_len)} mean={np.mean(fl_len):.1f} vs other lines median={np.median(ot_len)} mean={np.mean(ot_len):.1f}")
TRIV=re.compile(r"^(\{|\}|\};|return;|return [A-Za-z0-9_\-]+;|break;|continue;|else|\} else \{|\}\s*else\s*\{|/\*.*\*/|//.*|#.*)?$")
def triv(t): return len(norm(t))<6 or bool(TRIV.match(t))
print(f"  share of top-1 predicted lines that are trivial: A={100*np.mean([triv(t) for t in t1A]):.1f}% B={100*np.mean([triv(t) for t in t1B]):.1f}%")
# ---------- same_text deep dive
def same_text(S,name):
    recs=[]
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
            w=1.0 if a>b+1e-4 else (0.5 if abs(a-b)<=1e-4 else 0.0)
            recs.append({"idx":r["idx"],"project":r["project"],"win":w,"text":vl[l].strip(),"a":a,"b":b,"same_pos":j==l,"nflaw":len(r["flaw_lines"]),"n_lines":r["n_lines"],"fix_longer":f["n_lines"]-r["n_lines"]})
    W=np.array([x["win"] for x in recs])
    print(f"  {name}: n={len(recs)} win={100*W.mean():.1f} ties={(W==0.5).sum()} functions={len({x['idx'] for x in recs})} projects={len({x['project'] for x in recs})}")
    cf=Counter(x["idx"] for x in recs); print("    lines per function: median",np.median(list(cf.values())),"max",max(cf.values()),"top5 fns",cf.most_common(5))
    cp=Counter(x["project"] for x in recs); print("    lines per project top6",cp.most_common(6))
    # per-function mean, then mean over functions
    fw=defaultdict(list)
    for x in recs: fw[x["idx"]].append(x["win"])
    print(f"    function-averaged win={100*np.mean([np.mean(v) for v in fw.values()]):.1f}; project-averaged={100*np.mean([np.mean([x['win'] for x in recs if x['project']==p]) for p in cp]):.1f}")
    tr=np.array([triv(x["text"]) for x in recs]); print(f"    trivial-ish lines {tr.sum()} win={100*W[tr].mean():.1f}; non-trivial win={100*W[~tr].mean():.1f}")
    # lines whose text appears at identical position (unchanged context) vs moved
    sp=np.array([x["same_pos"] for x in recs]); print(f"    same position {sp.sum()} win={100*W[sp].mean():.1f}; shifted {(~sp).sum()} win={100*W[~sp].mean():.1f}")
    # fix longer vs shorter
    fl=np.array([x["fix_longer"] for x in recs]); print(f"    fix longer: n={(fl>0).sum()} win={100*W[fl>0].mean():.1f}; fix same/shorter n={(fl<=0).sum()} win={100*W[fl<=0].mean():.1f}")
    # cap per function at 5 lines
    capped=[np.mean(v[:50]) for v in fw.values()]
    return recs
print("\n== same_text")
SA=same_text(A,"A"); SB=same_text(B,"B")
# most common same-text line texts
ct=Counter(x["text"] for x in SA); print("  most common same-text flaw lines:",ct.most_common(12))
# what are 'flaw lines also present in fixed version'? Why are they flaw lines? -> e.g. lines that moved
# examine the distribution of |a-b| 
print("  A mean (a-b)=%.3f  B mean (a-b)=%.3f"%(np.mean([x["a"]-x["b"] for x in SA]),np.mean([x["a"]-x["b"] for x in SB])))
# ---------- group C

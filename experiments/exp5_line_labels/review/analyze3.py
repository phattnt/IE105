import json, re, math
import numpy as np
from collections import defaultdict, Counter
D="/tmp/claude-0/-home-user-ie105/dd09b186-5b32-5246-9add-691e2a7b029b/scratchpad/kg_prep_out/secvul_lines"
R="/home/user/ie105/experiments/exp5_line_labels/results"
rows=[json.loads(l) for l in open(f"{D}/test.jsonl")]; by={r["idx"]:r for r in rows}
def load(d): return {json.loads(l)["idx"]:np.array(json.loads(l)["line_scores"]) for l in open(f"{R}/{d}/test_scores.jsonl")}
A=load("slide_ModernBERT-base_2048_s0_conv"); B=load("slide_ModernBERT-base_2048_s0_contrast_conv"); P=load("slide_ModernBERT-base_2048_s0")
def exp_rand(n,f,k): return 1.0 if k>=n else 1-math.comb(n-f,k)/math.comb(n,k)
def recs(S):
    out={}
    for r in rows:
        if r["target"]!=1 or not r["flaw_lines"]: continue
        sc=S[r["idx"]]; n=len(sc); fl=[l for l in r["flaw_lines"] if l<n]
        order=sorted(range(n),key=lambda i:(-sc[i],i)); rank={l:k for k,l in enumerate(order)}; first=min(rank[l] for l in fl)
        out[r["idx"]]=dict(top1=first<1,top5=first<5,top10=first<10,r5=exp_rand(n,len(fl),5),dens=len(fl)/n,nf=len(fl),n=n,project=r["project"])
    return out
RA,RB,RP=recs(A),recs(B),recs(P)
L=lambda r: np.array([len(l.strip()) for l in r["func"].split("\n")],float)
RL=recs({r["idx"]:L(r) for r in rows if r["target"]==1})
def show(lab,sel):
    ids=[i for i in RA if sel(RA[i])]
    print(f"{lab}: n={len(ids)} rand5={100*np.mean([RA[i]['r5'] for i in ids]):.1f} longest5={100*np.mean([RL[i]['top5'] for i in ids]):.1f} A5={100*np.mean([RA[i]['top5'] for i in ids]):.1f} B5={100*np.mean([RB[i]['top5'] for i in ids]):.1f} pilotA5={100*np.mean([RP[i]['top5'] for i in ids]):.1f} | top1 A={100*np.mean([RA[i]['top1'] for i in ids]):.1f} B={100*np.mean([RB[i]['top1'] for i in ids]):.1f} longest={100*np.mean([RL[i]['top1'] for i in ids]):.1f}")
show("all",lambda x:True)
show("dens<=0.5",lambda x:x["dens"]<=0.5)
show("dens<=0.2 & nf<20",lambda x:x["dens"]<=0.2 and x["nf"]<20)
show("single flaw line",lambda x:x["nf"]==1)
show("n_lines>100",lambda x:x["n"]>100)
show("n_lines<=30",lambda x:x["n"]<=30)
show("longest misses top5",lambda x:not RL[[i for i in RA if RA[i] is x][0]]["top5"])
# per-project sign test A vs B on top5 (projects with >=5 fns)
pp=defaultdict(list)
for i in RA: pp[RA[i]["project"]].append((RA[i]["top5"],RB[i]["top5"]))
big={p:v for p,v in pp.items() if len(v)>=5}
wa=sum(np.mean([a for a,b in v])>np.mean([b for a,b in v]) for v in big.values()); wb=sum(np.mean([a for a,b in v])<np.mean([b for a,b in v]) for v in big.values())
print(f"projects with >=5 line-eval fns: {len(big)}; A better top5 in {wa}, B better in {wb}")
for p,v in sorted(big.items(),key=lambda kv:-len(kv[1]))[:8]:
    print(f"   {p:22s} n={len(v):3d} A5={100*np.mean([a for a,b in v]):.1f} B5={100*np.mean([b for a,b in v]):.1f}")
# pairs per project A vs B
def pairs(S):
    out={}
    for r in rows:
        if r["target"]==1 and r["pair_idx"] in S:
            fv,ff=S[r["idx"]].max(),S[r["pair_idx"]].max(); out[r["idx"]]=(r["project"],1.0 if fv>ff+1e-4 else (0.5 if abs(fv-ff)<=1e-4 else 0.0))
    return out
PA,PB,PP=pairs(A),pairs(B),pairs(P)
pp=defaultdict(list)
for i in PA: pp[PA[i][0]].append((PA[i][1],PB[i][1]))
big={p:v for p,v in pp.items() if len(v)>=5}
wa=sum(np.mean([a for a,b in v])>np.mean([b for a,b in v]) for v in big.values()); wb=sum(np.mean([a for a,b in v])<np.mean([b for a,b in v]) for v in big.values())
print(f"pairs: projects>=5: {len(big)}; A better in {wa}, B better in {wb}")
for p,v in sorted(big.items(),key=lambda kv:-len(kv[1]))[:8]:
    print(f"   {p:22s} n={len(v):3d} Apair={100*np.mean([a for a,b in v]):.1f} Bpair={100*np.mean([b for a,b in v]):.1f}")
# run-to-run: plain pilot vs plain conv
print("plain pilot vs plain conv: pairs %.1f vs %.1f; top5 %.1f vs %.1f; top1 %.1f vs %.1f"%(100*np.mean([v[1] for v in PP.values()]),100*np.mean([v[1] for v in PA.values()]),100*np.mean([x['top5'] for x in RP.values()]),100*np.mean([x['top5'] for x in RA.values()]),100*np.mean([x['top1'] for x in RP.values()]),100*np.mean([x['top1'] for x in RA.values()])))
# group C McNemar on SecVulEval needs group labels: approximate by n_lines? skip; use chars>~2000 as proxy
# agreement between A and pilot vs A and B on top5 (how different are two plain runs?)
ids=list(RA)
agAP=np.mean([RA[i]["top5"]==RP[i]["top5"] for i in ids]); agAB=np.mean([RA[i]["top5"]==RB[i]["top5"] for i in ids])
print(f"top5 agreement: plain-conv vs plain-pilot {100*agAP:.1f}%, plain-conv vs contrast-conv {100*agAB:.1f}%")
# contrastive pilot test numbers
cp=json.load(open(f"{R}/slide_ModernBERT-base_2048_s0_contrast/results.json"))
print("contrast pilot keys",list(cp)); t=cp.get("test",{}); 
if t: print("contrast pilot test: top1 %.1f top5 %.1f top10 %.1f pairs %.1f same %.1f sp %.2f"%(t["lines_ALL"]["top1"],t["lines_ALL"]["top5"],t["lines_ALL"]["top10"],t["pairs"]["ranking_acc_non_tied"],t["same_text_flaw_lines"]["vul_scored_higher_pct"],t["function"]["spearman_with_tokens"]))
# estimate how many training functions exceed 2 windows (>~3069 tokens ~ chars/3.3)
tr=[json.loads(l) for l in open(f"{D}/train.jsonl")]
big_fn=[r for r in tr if len(r["func"])/3.3>3069]
print("train fns with est. >2 windows:",len(big_fn),"of",len(tr),"; vulnerable among them",sum(r["target"] for r in big_fn),"; flaw lines they hold:",sum(len(r["flaw_lines"]) for r in big_fn),"of",sum(len(r["flaw_lines"]) for r in tr))

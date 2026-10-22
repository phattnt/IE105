import json, math, numpy as np
from collections import defaultdict
D="/tmp/claude-0/-home-user-ie105/dd09b186-5b32-5246-9add-691e2a7b029b/scratchpad/kg_prep_out/secvul_lines"
R="/home/user/ie105/experiments/exp5_line_labels/results"
rows=[json.loads(l) for l in open(f"{D}/test.jsonl")]; by={r["idx"]:r for r in rows}
def load(d): return {json.loads(l)["idx"]:np.array(json.loads(l)["line_scores"]) for l in open(f"{R}/{d}/test_scores.jsonl")}
A=load("slide_ModernBERT-base_2048_s0_conv"); B=load("slide_ModernBERT-base_2048_s0_contrast_conv")
def mcn(wa,wb):
    b01=int(sum(1 for a,b in zip(wa,wb) if b>a)); b10=int(sum(1 for a,b in zip(wa,wb) if a>b)); n=b01+b10
    p=2*sum(math.comb(n,k) for k in range(0,min(b01,b10)+1))/2**n if n else 1
    return b01,b10,min(1,p)
def pairs(S,agg):
    out={}
    for r in rows:
        if r["target"]==1 and r["pair_idx"] in S:
            fv,ff=agg(S[r["idx"]]),agg(S[r["pair_idx"]]); out[r["idx"]]=1.0 if fv>ff+1e-4 else (0.5 if abs(fv-ff)<=1e-4 else 0.0)
    return out
mx=lambda s:s.max()
PA,PB=pairs(A,mx),pairs(B,mx)
for lab,sel in (("fix shorter(lines)",lambda r:by[r["pair_idx"]]["n_lines"]<r["n_lines"]),("equal",lambda r:by[r["pair_idx"]]["n_lines"]==r["n_lines"]),("fix longer",lambda r:by[r["pair_idx"]]["n_lines"]>r["n_lines"])):
    ids=[r["idx"] for r in rows if r["idx"] in PA and sel(r)]
    print(lab,"n",len(ids),"A=%.1f B=%.1f"%(100*np.mean([PA[i] for i in ids]),100*np.mean([PB[i] for i in ids])),"McNemar (B>,A>,p)",mcn([PA[i] for i in ids],[PB[i] for i in ids]))
# alternative aggregations (eval-only)
for lab,agg in (("max",mx),("mean",lambda s:s[s>-1e8].mean()),("top3mean",lambda s:np.sort(s[s>-1e8])[-3:].mean()),("lse",lambda s:np.log(np.exp(s[s>-1e8]).sum()))):
    pa,pb=pairs(A,agg),pairs(B,agg)
    fa=np.array([agg(A[r["idx"]]) for r in rows]); nl=np.array([r["n_lines"] for r in rows]); 
    sp=lambda a,b: np.corrcoef(np.argsort(np.argsort(a)),np.argsort(np.argsort(b)))[0,1]
    fb=np.array([agg(B[r["idx"]]) for r in rows])
    print(f"agg={lab:8s} pairs A={100*np.mean(list(pa.values())):.1f} B={100*np.mean(list(pb.values())):.1f} | spearman(fn,n_lines) A={sp(fa,nl):.2f} B={sp(fb,nl):.2f}")
# A: how does the max logit depend on length within benign/vul? mean max logit by n_lines bin
for name,S in (("A",A),("B",B)):
    bins=[(0,20),(20,50),(50,100),(100,300),(300,10000)]
    print(name,"mean max-logit by n_lines bin (vul/fixed):",[(f"{a}-{b}",round(float(np.mean([S[r['idx']].max() for r in rows if a<=r['n_lines']<b and r['target']==1])),2),round(float(np.mean([S[r['idx']].max() for r in rows if a<=r['n_lines']<b and r['target']==0])),2)) for a,b in bins])

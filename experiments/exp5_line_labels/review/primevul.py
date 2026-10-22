import json, numpy as np, math
R="/home/user/ie105/experiments/exp5_line_labels/results"
def auc(y,s):
    # rank-based ROC AUC with ties
    order=np.argsort(s); r=np.empty(len(s)); 
    ss=s[order]; i=0; rk=np.zeros(len(s))
    while i<len(ss):
        j=i
        while j+1<len(ss) and ss[j+1]==ss[i]: j+=1
        rk[order[i:j+1]]=(i+j)/2+1; i=j+1
    n1=y.sum(); n0=len(y)-n1
    return (rk[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
for d in ["slide_ModernBERT-base_2048_s0_conv","slide_ModernBERT-base_2048_s0_contrast_conv","slide_ModernBERT-base_2048_s0"]:
    rec=[json.loads(l) for l in open(f"{R}/{d}/primevul_test_scores.jsonl")]
    y=np.array([r["target"] for r in rec]); s=np.array([r["logit"] for r in rec])
    print(d, "n",len(y),"pos",y.sum(),"AUC=%.3f"%auc(y,s), "AUC(-s)=%.3f"%auc(y,-s), "mean logit vul=%.2f benign=%.2f"%(s[y==1].mean(),s[y==0].mean()))
    # decile analysis
    q=np.quantile(s,np.linspace(0,1,11)); 
    print("   prevalence by score decile (low->high):",[round(100*y[(s>=q[i])&(s<=q[i+1])].mean(),2) for i in range(10)])
# Wilson CIs for paired aggregates
def wilson(p,n,z=1.96):
    c=p+z*z/(2*n); d=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return ((c-d)/(1+z*z/n),(c+d)/(1+z*z/n))
for lab,p,n in [("A pairs",0.6336,524),("B pairs",0.6299,535),("A same_text",0.5725,200),("B same_text",0.525,200),("A top1 ALL",0.1933,357),("B top1 ALL",0.1261,357),("A top5",0.4118,357),("B top5",0.3613,357),("A top10",0.5938,357),("B top10",0.4986,357),("A grpC top5",0.2348,132),("B grpC top5",0.1591,132),("SecVul A grpC top5",0.251,231),("SecVul B grpC top5",0.195,231)]:
    lo,hi=wilson(p,n); print(f"{lab}: {100*p:.1f} [{100*lo:.1f},{100*hi:.1f}] n={n}")
# two-proportion z for independent-ish approximations (lower bound; actual are paired so this is conservative-ish)
def z2(p1,n1,p2,n2):
    p=(p1*n1+p2*n2)/(n1+n2); se=math.sqrt(p*(1-p)*(1/n1+1/n2)); return (p1-p2)/se
print("z top10 PV A vs B (unpaired approx):",round(z2(0.5938,357,0.4986,357),2))
print("z top1 PV A vs B:",round(z2(0.1933,357,0.1261,357),2))
print("z same_text PV:",round(z2(0.5725,200,0.525,200),2))

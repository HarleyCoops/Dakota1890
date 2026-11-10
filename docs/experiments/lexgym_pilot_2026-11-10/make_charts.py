import json, sys
from decimal import Decimal, ROUND_HALF_UP
r2=lambda v: str(Decimal(str(round(v,4))).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
out=sys.argv[1] if len(sys.argv)>1 else "charts"  # run from this folder: python make_charts.py
INK,SEC,MUT,GRID,AX,SURF="#0b0b0b","#52514e","#898781","#e1e0d9","#c3c2b7","#fcfcfb"
C=["#2a78d6","#eb6834","#1baf7a"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.edgecolor":AX,"axes.labelcolor":SEC,"xtick.color":MUT,"ytick.color":MUT,"text.color":INK,"figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF})
load=lambda p:[json.loads(l) for l in open(p)]
b,r,s=load('pilot_base_eval_metrics.jsonl'),load('pilot_rl_metrics.jsonl'),load('pilot_sft_metrics.jsonl')
def ev(rows,step,task,key):
    for x in rows:
        k=f'test/env/task/{task}/{key}'
        if x.get('step')==step and k in x: return x[k]
tasks=[("syllabify","ledger/exact"),("pos","ledger/exact"),("root","ledger/exact"),("inflect","ledger/exact"),("lookup_gloss","ledger/exact"),("lookup_root_gloss","ledger/exact"),("gloss","ledger/token_f1"),("en_to_dak","ledger/exact")]
labels=["syllabify","pos","root","inflect","lookup\n(1 hop)","lookup\n(2 hops)","gloss\n(token F1)","English→\nDakota"]
series=[("Base model",[ev(b,0,t,k) for t,k in tasks]),("After supervised stage",[ev(r,0,t,k) for t,k in tasks]),("After supervised + 30 RL steps",[ev(r,30,t,k) for t,k in tasks])]
def style(ax):
    for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True,color=GRID,linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
fig,ax=plt.subplots(figsize=(11,4.6),dpi=160)
w=0.26
for i,(name,vals) in enumerate(series):
    xs=[j+(i-1)*(w+0.02) for j in range(len(tasks))]
    ax.bar(xs,vals,width=w,color=C[i],label=name,linewidth=0)
    for x,v in zip(xs,vals): ax.text(x,v+0.012,r2(v),ha="center",va="bottom",fontsize=7.5,color=SEC)
ax.set_xticks(range(len(tasks))); ax.set_xticklabels(labels,color=SEC); ax.set_ylim(0,1.0); ax.set_ylabel("Exact-match rate")
ax.axvline(5.5,color=AX,linewidth=1); ax.text(2.5,0.965,"Held-out word families (generalization)",ha="center",fontsize=9,color=SEC); ax.text(6.5,0.965,"Recall (training vocabulary)",ha="center",fontsize=9,color=SEC)
style(ax); ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.2),ncol=3,labelcolor=SEC)
ax.set_title("Pilot: evaluation by task and stage  (Qwen3.5-4B, 40 tasks per type, one seed)",loc="left",fontsize=11.5,pad=12)
fig.tight_layout(); fig.savefig(f"{out}/pilot_eval_by_stage.png"); plt.close(fig)

fig,(a1,a2)=plt.subplots(1,2,figsize=(11,3.8),dpi=160)
tr=[(x['step'],x['train_mean_nll']) for x in s if 'train_mean_nll' in x]; te=[(x['step'],x['test/nll']) for x in s if 'test/nll' in x]
k=15; sm=[(tr[i][0],sum(v for _,v in tr[max(0,i-k+1):i+1])/len(tr[max(0,i-k+1):i+1])) for i in range(len(tr))]
a1.plot(*zip(*sm),color=C[0],linewidth=2); a1.plot(*zip(*te),color=C[1],linewidth=2,marker="o",markersize=4)
a1.text(120,1.07,"train (15-step mean)",color=SEC,fontsize=8.5); a1.text(60,0.80,"held-out rows",color=SEC,fontsize=8.5)
a1.set_ylim(0.5,1.9); a1.set_xlabel("Supervised step (128 rows per step)"); a1.set_ylabel("Loss (nats per answer token)"); a1.set_title("Stage 1: supervised warm start",loc="left",fontsize=11); style(a1)
steps=[0,10,20,30]
for i,(t,lab) in enumerate([("lookup_gloss","1 lookup needed"),("lookup_root_gloss","2 lookups needed")]):
    ys=[ev(r,st,t,"ledger/exact") for st in steps]; a2.plot(steps,ys,color=C[i],linewidth=2,marker="o",markersize=6,markeredgecolor=SURF,markeredgewidth=1.5)
    a2.text(30.6,ys[-1],lab,color=SEC,fontsize=8.5,va="center")
    a2.plot([-0.5,30],[ev(b,0,t,"ledger/exact")]*2,color=C[i],linewidth=1,linestyle=(0,(4,3)))
a2.text(0.3,0.52,"dashed: base model",color=MUT,fontsize=8)
a2.set_xlim(-0.5,40); a2.set_xticks(steps); a2.set_ylim(0,1); a2.set_xlabel("RL step (32 tasks × 8 rollouts per step)"); a2.set_ylabel("Held-out exact-match rate"); a2.set_title("Stage 2: RL restores dictionary lookup",loc="left",fontsize=11); style(a2)
fig.tight_layout(); fig.savefig(f"{out}/pilot_training_curves.png"); plt.close(fig)
print([ (n,[round(v,3) for v in vals]) for n,vals in series])

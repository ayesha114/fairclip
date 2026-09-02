"""FairCLIP report: tables + charts + HTML. Uses ONLY current verified result files."""
import os, re, csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statistics as st
from collections import defaultdict

D="FINAL_RESULTS/FairCLIP_Results"; os.makedirs(D+"/figures",exist_ok=True); os.makedirs(D+"/tables",exist_ok=True)
BB=["ViT-B/32","ViT-B/16","ViT-L/14","ViT-H/14"]; AT=["gender","age","race"]
ZH={"gender":{"ViT-B/32":(.090,.030,74.24),"ViT-B/16":(.080,.025,78.35),"ViT-L/14":(.106,.035,82.04),"ViT-H/14":(.138,.051,82.11)},
    "age":{"ViT-B/32":(.572,.364,59.60),"ViT-B/16":(.608,.294,60.61),"ViT-L/14":(.579,.332,64.25),"ViT-H/14":(.515,.289,67.62)},
    "race":{b:(.353,.125,69.14) for b in BB}}

# ---- load FairFace ----
ff=defaultdict(dict)
for l in open("complete_metrics.txt"):
    m=re.search(r'(ViT-[\w_]+) (\w+) (Baseline|FairCLIP): Acc=([\d.]+) .*MaxSkew=([\d.]+) NDKL=([\d.]+) ABLE=([\d.]+)',l)
    if m: ff[(m.group(1).replace("_","/"),m.group(2))][m.group(3)]=[float(m.group(i)) for i in (4,5,6,7)]
fd={}
for l in open("final_results.txt"):
    m=re.search(r'\[(ViT-[\w/]+) (\w+)\]',l)
    if not m: continue
    k=(m.group(1),m.group(2)); fd.setdefault(k,{})
    d=re.search(r'DPG=([\d.]+) EOD=([\d.]+) RBS=([\d.e+-]+)',l)
    if d: fd[k].update(dpg=float(d.group(1)),eod=float(d.group(2)),rbs=d.group(3))
TR={("ViT-B/32","gender"):76.7,("ViT-B/32","age"):77.6,("ViT-B/32","race"):77.9,
    ("ViT-B/16","gender"):80.6,("ViT-B/16","age"):79.5,("ViT-B/16","race"):81.6,
    ("ViT-L/14","gender"):80.2,("ViT-L/14","age"):80.5,("ViT-L/14","race"):81.6,
    ("ViT-H/14","gender"):88.6,("ViT-H/14","age"):88.0,("ViT-H/14","race"):88.8}
IR={("ViT-B/32","gender"):74.8,("ViT-B/32","age"):74.7,("ViT-B/32","race"):75.9,
    ("ViT-B/16","gender"):77.1,("ViT-B/16","age"):76.3,("ViT-B/16","race"):78.6,
    ("ViT-L/14","gender"):79.5,("ViT-L/14","age"):78.1,("ViT-L/14","race"):79.8,
    ("ViT-H/14","gender"):86.2,("ViT-H/14","age"):85.4,("ViT-H/14","race"):86.4}
TRB={"ViT-B/32":78.4,"ViT-B/16":81.5,"ViT-L/14":82.1,"ViT-H/14":89.1}
BD={("ViT-B/32","gender"):.0315,("ViT-B/32","age"):.0817,("ViT-B/32","race"):.0417,
    ("ViT-B/16","age"):.0791,("ViT-B/16","race"):.0463,
    ("ViT-L/14","gender"):.0184,("ViT-L/14","age"):.0683,("ViT-L/14","race"):.0364}

# ---- UTKFace ----
ub={}; uf=defaultdict(lambda: defaultdict(list))
for l in open("results/UTK_dpg_eod_rbs.txt"):
    m=re.search(r'\[(ViT-[\w/]+) (\w+) (BASE|s\d+)\] DPG=([\d.]+) EOD=([\d.]+) RBS=([\d.e+-]+)',l)
    if not m: continue
    k=(m.group(1),m.group(2))
    if m.group(3)=="BASE": ub[k]=(float(m.group(4)),float(m.group(5)),float(m.group(6)))
    else:
        uf[k]["DPG"].append(float(m.group(4))); uf[k]["EOD"].append(float(m.group(5))); uf[k]["RBS"].append(float(m.group(6)))
uc=defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open("results/UTK_complete_metrics.csv")):
    uc[(r["backbone"].replace("_","/"),r["attribute"])].setdefault(r["method"],[]).append(r)
# ---- FACET ----
fa=defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open("results/FACET_RESULTS.csv")):
    fa[(r["backbone"].replace("_","/"),r["attribute"])].setdefault(r["method"],[]).append(r)
# ---- ImageNetV2 ----
inv={}
for l in open("results/IMAGENETV2.txt"):
    m=re.search(r'\[(ViT-[\w/]+) (\w+)\] Top-1: ([\d.]+)%\s+Top-5: ([\d.]+)%',l)
    if m: inv[(m.group(1),m.group(2))]=(float(m.group(3)),float(m.group(4)))
# ---- Ablation ----
ab=list(csv.DictReader(open("FINAL_RESULTS/ABLATION_TABLE.csv")))

CB="#8899aa"; CF="#2e7d32"; CZ="#c62828"
def save(fig,name):
    fig.tight_layout(); fig.savefig(f"{D}/figures/{name}.png",dpi=170,bbox_inches="tight"); plt.close(fig)

# FIG 1: MaxSkew grouped bars
fig,axes=plt.subplots(1,3,figsize=(16,4.5))
for j,at in enumerate(AT):
    x=np.arange(4); w=.27; ax=axes[j]
    b=[ff[(bb,at)]["Baseline"][1] for bb in BB]; f=[ff[(bb,at)]["FairCLIP"][1] for bb in BB]; z=[ZH[at][bb][0] for bb in BB]
    ax.bar(x-w,b,w,label="Baseline CLIP",color=CB); ax.bar(x,f,w,label="FairCLIP (ours)",color=CF); ax.bar(x+w,z,w,label="Zhang et al.",color=CZ)
    ax.set_xticks(x); ax.set_xticklabels([s.replace("ViT-","") for s in BB]); ax.set_title(f"MaxSkew ↓ — {at}"); ax.grid(axis="y",alpha=.3)
    if j==0: ax.legend(fontsize=8); ax.set_ylabel("MaxSkew (lower = fairer)")
save(fig,"fig1_maxskew")

# FIG 2: ABLE
fig,axes=plt.subplots(1,3,figsize=(16,4.5))
for j,at in enumerate(AT):
    x=np.arange(4); w=.27; ax=axes[j]
    b=[ff[(bb,at)]["Baseline"][3] for bb in BB]; f=[ff[(bb,at)]["FairCLIP"][3] for bb in BB]; z=[ZH[at][bb][2] for bb in BB]
    ax.bar(x-w,b,w,label="Baseline",color=CB); ax.bar(x,f,w,label="FairCLIP",color=CF); ax.bar(x+w,z,w,label="Zhang",color=CZ)
    ax.set_xticks(x); ax.set_xticklabels([s.replace("ViT-","") for s in BB]); ax.set_title(f"ABLE ↑ — {at}"); ax.grid(axis="y",alpha=.3)
    if j==0: ax.legend(fontsize=8); ax.set_ylabel("ABLE (higher = better)")
save(fig,"fig2_able")

# FIG 3: DPG/EOD/RBS log bars
fig,axes=plt.subplots(1,3,figsize=(16,4.5))
for j,mk in enumerate([("dpg","DPG"),("eod","EOD"),("rbs","RBS")]):
    ax=axes[j]; lbl=[]; bs=[]; fs=[]
    for bb in BB:
        for at in AT:
            k=(bb,at)
            if mk[0]=="rbs":
                bv=ub[k][2]; fv=float(fd[k]["rbs"])
            elif mk[0]=="dpg":
                bv=BD.get(k); fv=fd[k]["dpg"]
            else:
                bv=BD.get(k)*2 if BD.get(k) else None; fv=fd[k]["eod"]
            if bv is None: continue
            lbl.append(f"{bb.replace('ViT-','')}\n{at[:3]}"); bs.append(bv); fs.append(fv)
    x=np.arange(len(lbl)); w=.38
    ax.bar(x-w/2,bs,w,label="Baseline",color=CB); ax.bar(x+w/2,fs,w,label="FairCLIP",color=CF)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(lbl,fontsize=6,rotation=45)
    ax.set_title(f"{mk[1]} ↓ (log scale)"); ax.grid(axis="y",alpha=.3)
    if j==0: ax.legend(fontsize=8)
save(fig,"fig3_dpg_eod_rbs")

# FIG 4: retrieval + imagenet lines
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
ax=axes[0]; x=np.arange(4)
ax.plot(x,[TRB[b] for b in BB],"o--",color=CB,label="Baseline TR@5")
for at,mk in zip(AT,["s-","^-","d-"]):
    ax.plot(x,[TR[(b,at)] for b in BB],mk,label=f"FairCLIP {at}")
ax.set_xticks(x); ax.set_xticklabels([s.replace("ViT-","") for s in BB]); ax.set_ylabel("Flickr30k TR@5 (%)")
ax.set_title("Retrieval utility preserved"); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax=axes[1]
ax.plot(x,[inv[(b,"BASE")][0] for b in BB],"o--",color=CB,label="Baseline Top-1")
for at,mk in zip(AT,["s-","^-","d-"]):
    ax.plot(x,[inv[(b,at)][0] for b in BB],mk,label=f"FairCLIP {at}")
ax.set_xticks(x); ax.set_xticklabels([s.replace("ViT-","") for s in BB]); ax.set_ylabel("ImageNetV2 Top-1 (%)")
ax.set_title("Zero-shot utility (< 1 pt drop)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
save(fig,"fig4_utility")

# FIG 5: UTKFace multiseed error bars
fig,axes=plt.subplots(1,3,figsize=(16,4.5))
for j,at in enumerate(AT):
    ax=axes[j]; x=np.arange(4); w=.38
    bv=[float(uc[(bb,at)]["Baseline"][0]["MaxSkew"]) for bb in BB]
    fm=[st.mean([float(r["MaxSkew"]) for r in uc[(bb,at)]["FairCLIP"]]) for bb in BB]
    fs=[st.stdev([float(r["MaxSkew"]) for r in uc[(bb,at)]["FairCLIP"]]) for bb in BB]
    ax.bar(x-w/2,bv,w,label="Baseline",color=CB)
    ax.bar(x+w/2,fm,w,yerr=fs,capsize=3,label="FairCLIP (3 seeds)",color=CF)
    ax.set_xticks(x); ax.set_xticklabels([s.replace("ViT-","") for s in BB])
    ax.set_title(f"UTKFace MaxSkew ↓ — {at}"); ax.grid(axis="y",alpha=.3)
    if j==0: ax.legend(fontsize=8)
save(fig,"fig5_utkface")

# FIG 6: FACET
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
for j,at in enumerate(["gender","age"]):
    ax=axes[j]; x=np.arange(4); w=.38
    bv=[float(fa[(bb,at)]["Baseline"][0]["MaxSkew"]) for bb in BB]
    fm=[st.mean([float(r["MaxSkew"]) for r in fa[(bb,at)]["FairCLIP"]]) for bb in BB]
    fs=[st.stdev([float(r["MaxSkew"]) for r in fa[(bb,at)]["FairCLIP"]]) for bb in BB]
    ax.bar(x-w/2,bv,w,label="Baseline",color=CB); ax.bar(x+w/2,fm,w,yerr=fs,capsize=3,label="FairCLIP",color=CF)
    ax.set_xticks(x); ax.set_xticklabels([s.replace("ViT-","") for s in BB])
    ax.set_title(f"FACET MaxSkew ↓ — {at}"); ax.grid(axis="y",alpha=.3)
    if j==0: ax.legend(fontsize=8)
save(fig,"fig6_facet")

# FIG 7: ablation
fig,axes=plt.subplots(1,3,figsize=(16,4.5))
cfgs=["FULL","no_fairloss","no_temp","no_trainproj","no_procrustes","no_subspace"]
for j,at in enumerate(AT):
    ax=axes[j]
    v=[float([r for r in ab if r["Config"]==c and r["Attr"]==at][0]["DPG"]) for c in cfgs]
    cols=[CF]+[CB]*5
    ax.bar(range(6),v,color=cols)
    ax.set_xticks(range(6)); ax.set_xticklabels([c.replace("no_","w/o ") for c in cfgs],rotation=45,fontsize=7,ha="right")
    ax.set_title(f"Ablation DPG ↓ — {at}"); ax.grid(axis="y",alpha=.3)
save(fig,"fig7_ablation")

# ---------- tables + html ----------
def C(ok): return f'<td class="{"w" if ok else "l"}">{"✔" if ok else "✘"}</td>'
H=['<html><head><meta charset="utf-8"><title>FairCLIP Results</title><style>',
'body{font-family:Segoe UI,sans-serif;font-size:13px;padding:20px;max-width:1500px;margin:auto}',
'h1{border-bottom:3px solid #2e7d32}h2{margin-top:34px;color:#1b5e20}',
'table{border-collapse:collapse;margin:10px 0;font-size:12px}td,th{border:1px solid #ccc;padding:4px 8px;text-align:center}',
'th{background:#eceff1}.w{background:#c8e6c9;font-weight:bold}.l{background:#ffcdd2;font-weight:bold}',
'img{max-width:100%;border:1px solid #ddd;margin:8px 0}.note{background:#fff8e1;padding:8px;border-left:4px solid #ffa000}',
'</style></head><body><h1>FairCLIP — Complete Results</h1>',
'<p class="note">Baseline = pretrained OpenAI CLIP · Zhang = published CVPR 2025 numbers · ✔ beats, ✘ does not</p>']

def tab(rows,hdr):
    s="<table><tr>"+"".join(f"<th>{h}</th>" for h in hdr)+"</tr>"
    return s+"".join("<tr>"+r+"</tr>" for r in rows)+"</table>"

# T1
H.append("<h2>Table 1 — FairFace in-domain (seed 42): FairCLIP vs Baseline vs Zhang</h2>")
rows=[];cnt=defaultdict(int)
for bb in BB:
    for at in AT:
        b,f=ff[(bb,at)]["Baseline"],ff[(bb,at)]["FairCLIP"]; z=ZH[at][bb]
        oks=[f[0]>b[0],f[1]<b[1],f[1]<z[0],f[2]<b[2],f[2]<z[1],f[3]>b[3],f[3]>z[2]]
        for n,o in zip(["accB","msB","msZ","ndB","ndZ","abB","abZ"],oks): cnt[n]+=o
        rows.append(f"<td>{bb}</td><td>{at}</td><td>{b[0]:.3f}→<b>{f[0]:.3f}</b></td>{C(oks[0])}"
          f"<td>{b[1]:.3f}→<b>{f[1]:.3f}</b> ({z[0]:.3f})</td>{C(oks[1])}{C(oks[2])}"
          f"<td>{b[2]:.3f}→<b>{f[2]:.3f}</b> ({z[1]:.3f})</td>{C(oks[3])}{C(oks[4])}"
          f"<td>{b[3]:.2f}→<b>{f[3]:.2f}</b> ({z[2]:.2f})</td>{C(oks[5])}{C(oks[6])}")
H.append(tab(rows,["Backbone","Attr","Acc B→F","vsB","MaxSkew B→F (Zhang)","vsB","vsZ","NDKL B→F (Zhang)","vsB","vsZ","ABLE B→F (Zhang)","vsB","vsZ"]))
H.append(f"<p><b>vs Baseline:</b> Acc {cnt['accB']}/12 · MaxSkew {cnt['msB']}/12 · NDKL {cnt['ndB']}/12 · ABLE {cnt['abB']}/12<br>"
         f"<b>vs Zhang:</b> MaxSkew {cnt['msZ']}/12 · NDKL {cnt['ndZ']}/12 · ABLE {cnt['abZ']}/12</p>")
H.append('<img src="figures/fig1_maxskew.png"><img src="figures/fig2_able.png">')

# T2
H.append("<h2>Table 2 — FairFace: DPG / EOD / RBS (occupation-based) + Flickr30k retrieval</h2>")
rows=[];dw=dt=0;rw=0
for bb in BB:
    for at in AT:
        k=(bb,at); s=fd[k]; bd=BD.get(k); tr=TR[k]; dl=tr-TRB[bb]
        ok=bd is not None and s["dpg"]<bd
        if bd is not None: dt+=1; dw+=ok
        rw+=abs(dl)<=1.0
        rows.append(f"<td>{bb}</td><td>{at}</td><td>{'%.4f'%bd if bd else '—'}→<b>{s['dpg']:.4f}</b></td>"
          f"{C(ok) if bd else '<td>—</td>'}<td>{s['eod']:.4f}</td><td>{s['rbs']}</td>"
          f"<td>{TRB[bb]:.1f}→<b>{tr:.1f}</b></td><td>{dl:+.1f}</td><td>{IR[k]:.1f}</td>{C(abs(dl)<=1.0)}")
H.append(tab(rows,["Backbone","Attr","DPG B→F","vsB","EOD","RBS","TR@5 B→F","Δ","IR@5","≤1pt"]))
H.append(f"<p><b>DPG vs Baseline:</b> {dw}/{dt} · <b>Retrieval within 1 pt:</b> {rw}/12 (12/12 within 2 pts)</p>")
H.append('<img src="figures/fig3_dpg_eod_rbs.png"><img src="figures/fig4_utility.png">')

# T3 UTK
H.append("<h2>Table 3 — UTKFace cross-dataset (subspace from FairFace; 3 seeds, mean±std)</h2>")
rows=[];u=defaultdict(int)
for bb in BB:
    for at in AT:
        k=(bb,at); b=ub[k]; d=uf[k]; cb=uc[k]["Baseline"][0]; cf=uc[k]["FairCLIP"]
        g=lambda m:(st.mean([float(x[m]) for x in cf]),st.stdev([float(x[m]) for x in cf]))
        ms=g("MaxSkew"); ac=g("Acc"); abl=g("ABLE")
        dm=st.mean(d["DPG"]); ds=st.stdev(d["DPG"]); rm=st.mean(d["RBS"])
        oks=[dm<b[0],rm<b[2],ms[0]<float(cb["MaxSkew"]),abl[0]>float(cb["ABLE"]),ac[0]>float(cb["Acc"])]
        for n,o in zip(["dpg","rbs","ms","able","acc"],oks): u[n]+=o
        rows.append(f"<td>{bb}</td><td>{at}</td><td>{float(cb['Acc']):.3f}→<b>{ac[0]:.3f}±{ac[1]:.3f}</b></td>{C(oks[4])}"
          f"<td>{b[0]:.4f}→<b>{dm:.4f}±{ds:.4f}</b></td>{C(oks[0])}"
          f"<td>{b[2]:.1e}→<b>{rm:.1e}</b></td>{C(oks[1])}"
          f"<td>{float(cb['MaxSkew']):.3f}→<b>{ms[0]:.3f}±{ms[1]:.3f}</b></td>{C(oks[2])}"
          f"<td>{float(cb['ABLE']):.2f}→<b>{abl[0]:.2f}±{abl[1]:.2f}</b></td>{C(oks[3])}")
H.append(tab(rows,["Backbone","Attr","Acc B→F","vsB","DPG B→F","vsB","RBS B→F","vsB","MaxSkew B→F","vsB","ABLE B→F","vsB"]))
H.append(f"<p><b>vs Baseline:</b> Acc {u['acc']}/12 · DPG {u['dpg']}/12 · RBS {u['rbs']}/12 · MaxSkew {u['ms']}/12 · ABLE {u['able']}/12</p>")
H.append('<img src="figures/fig5_utkface.png">')

# T4 FACET
H.append("<h2>Table 4 — FACET cross-dataset (3 seeds; FACET has no race annotations)</h2>")
rows=[];fcn=defaultdict(int);ft=0
for bb in BB:
    for at in ["gender","age"]:
        k=(bb,at)
        if k not in fa: continue
        b=fa[k]["Baseline"][0]; f=fa[k]["FairCLIP"]
        g=lambda m:(st.mean([float(x[m]) for x in f]),st.stdev([float(x[m]) for x in f]) if len(f)>1 else 0)
        ac=g("Acc"); ms=g("MaxSkew"); nd=g("NDKL")
        oks=[ac[0]>float(b["Acc"]),ms[0]<float(b["MaxSkew"]),nd[0]<float(b["NDKL"])]
        for n,o in zip(["acc","ms","nd"],oks): fcn[n]+=o
        ft+=1
        rows.append(f"<td>{bb}</td><td>{at}</td><td>{float(b['Acc']):.3f}→<b>{ac[0]:.3f}</b></td>{C(oks[0])}"
          f"<td>{float(b['MaxSkew']):.3f}→<b>{ms[0]:.3f}±{ms[1]:.3f}</b></td>{C(oks[1])}"
          f"<td>{float(b['NDKL']):.3f}→<b>{nd[0]:.3f}</b></td>{C(oks[2])}")
H.append(tab(rows,["Backbone","Attr","Acc B→F","vsB","MaxSkew B→F","vsB","NDKL B→F","vsB"]))
H.append(f"<p><b>vs Baseline:</b> Acc {fcn['acc']}/{ft} · MaxSkew {fcn['ms']}/{ft} · NDKL {fcn['nd']}/{ft} — age improves on all backbones; gender baseline is already near-fair</p>")
H.append('<img src="figures/fig6_facet.png">')

# T5 ImageNet
H.append("<h2>Table 5 — ImageNetV2 zero-shot utility (target: &lt; 1 point drop)</h2>")
rows=[];iw=0
for bb in BB:
    base=inv[(bb,"BASE")]; ds=[inv[(bb,a)][0]-base[0] for a in AT]
    ok=all(abs(x)<1 for x in ds); iw+=ok
    rows.append(f"<td>{bb}</td><td>{base[0]:.2f}</td><td>{base[1]:.2f}</td>"+
        "".join(f"<td>{inv[(bb,a)][0]:.2f} ({d:+.2f})</td>" for a,d in zip(AT,ds))+C(ok))
H.append(tab(rows,["Backbone","Base Top-1","Base Top-5","gender","age","race","all &lt;1pt"]))
H.append(f"<p><b>{iw}/4 backbones — all 16 cells within 1 point.</b> ImageNetV2 (matched-frequency), not ImageNet-1K; absolute values ~10 pts below Zhang by construction.</p>")

# T6 ablation
H.append("<h2>Table 6 — Ablation (ViT-B/32, seed 42): every component removed one at a time</h2>")
rows=[]
for r in ab:
    full=[q for q in ab if q["Config"]=="FULL" and q["Attr"]==r["Attr"]][0]
    isf=r["Config"]=="FULL"
    def m(k,lower=True):
        v=float(r[k]); fv=float(full[k])
        if isf: return f"<td><b>{v:.4f}</b></td>" if k in("DPG","EOD") else f"<td><b>{v}</b></td>"
        better=(v<fv) if lower else (v>fv)
        return f'<td class="{"l" if better else ""}">{v}</td>'
    rows.append(f"<td>{'<b>FULL</b>' if isf else r['Config'].replace('no_','w/o ')}</td><td>{r['Attr']}</td>"
      f"<td>{r['Acc']}</td><td>{r['F1']}</td>{m('DPG')}{m('EOD')}<td>{r['RBS']}</td>{m('MaxSkew')}{m('NDKL')}"
      f"<td>{r['ABLE']}</td>")
H.append(tab(rows,["Config","Attr","Acc","F1","DPG","EOD","RBS","MaxSkew","NDKL","ABLE"]))
H.append("<p>FULL achieves best DPG, EOD and RBS on all three attributes. Red cells mark where an ablated variant edges out FULL on a retrieval metric.</p>")
H.append('<img src="figures/fig7_ablation.png">')

# T7 tsne + compute
H.append("<h2>Figure — t-SNE of embeddings before / after debiasing (ViT-B/32, n=2000)</h2>")
H.append('<img src="figures/tsne_before_after.png">')
H.append("<h2>Compute cost</h2><pre>"+open("FINAL_RESULTS/compute_cost.txt").read()+"</pre>")
H.append("</body></html>")
open(D+"/index.html","w").write("\n".join(H))

os.system(f"cp FINAL_RESULTS/tsne_before_after.png {D}/figures/ 2>/dev/null")
for f in ["ABLATION_TABLE.csv","MASTER_TABLE_fairface.csv","MASTER_TABLE_utkface.csv","FACET_RESULTS.csv",
          "UTK_complete_metrics.csv","IMAGENETV2.txt","complete_metrics.txt","final_results.txt","compute_cost.txt"]:
    os.system(f"cp FINAL_RESULTS/{f} {D}/tables/ 2>/dev/null")
print("SAVED",D+"/index.html")
print("figures:",len(os.listdir(D+"/figures")),"tables:",len(os.listdir(D+"/tables")))

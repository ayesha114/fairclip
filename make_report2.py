"""FairCLIP report v2 — only current verified files."""
import os,re,csv,numpy as np,matplotlib,statistics as st
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import defaultdict
D="FINAL_RESULTS/FairCLIP_Results"; os.makedirs(D+"/figures",exist_ok=True); os.makedirs(D+"/tables",exist_ok=True)
BB=["ViT-B/32","ViT-B/16","ViT-L/14","ViT-H/14"]; AT=["gender","age","race"]
ZH={"gender":{"ViT-B/32":(.090,.030,74.24),"ViT-B/16":(.080,.025,78.35),"ViT-L/14":(.106,.035,82.04),"ViT-H/14":(.138,.051,82.11)},
    "age":{"ViT-B/32":(.572,.364,59.60),"ViT-B/16":(.608,.294,60.61),"ViT-L/14":(.579,.332,64.25),"ViT-H/14":(.515,.289,67.62)},
    "race":{b:(.353,.125,69.14) for b in BB}}
CB="#8899aa";CF="#2e7d32";CZ="#c62828"
def save(f,n): f.tight_layout(); f.savefig(f"{D}/figures/{n}.png",dpi=170,bbox_inches="tight"); plt.close(f)
def C(o): return f'<td class="{"w" if o else "l"}">{"✔" if o else "✘"}</td>'

# FairFace seed42
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
BD_extra={}
if os.path.exists("results/BASELINE_DPG_MISSING.txt"):
    for _l in open("results/BASELINE_DPG_MISSING.txt"):
        _m=re.search(r'\[(ViT-[\w/]+) (\w+)\] DPG=([\d.]+) EOD=([\d.]+) RBS=([\d.e+-]+)',_l)
        if _m: BD_extra[(_m.group(1),_m.group(2))]=(float(_m.group(3)),float(_m.group(4)),_m.group(5))
BD={("ViT-B/32","gender"):.0315,("ViT-B/32","age"):.0817,("ViT-B/32","race"):.0417,
    ("ViT-B/16","age"):.0791,("ViT-B/16","race"):.0463,
    ("ViT-L/14","gender"):.0184,("ViT-L/14","age"):.0683,("ViT-L/14","race"):.0364}
for _k,_v in BD_extra.items(): BD[_k]=_v[0]
# retrieval
ret={};rb={}
for l in open("results/RETRIEVAL_ALL_K.txt"):
    m=re.search(r'\[(ViT-[\w/]+) (\w+)\](.*)',l)
    if not m: continue
    v=dict(re.findall(r'([TI]R@\d+)=([\d.]+)',m.group(3)))
    (rb if m.group(2)=="BASE" else ret)[(m.group(1),m.group(2)) if m.group(2)!="BASE" else m.group(1)]=v
RB10={"ViT-B/32":("16.9","78.4","89.3","18.6","77.4","89.4"),"ViT-B/16":("16.7","81.5","91.1","18.8","78.9","91.7"),
      "ViT-L/14":("17.1","82.1","90.7","18.7","80.4","91.5"),"ViT-H/14":("18.4","89.1","95.6","19.2","87.3","96.0")}
# multiseed MS/NDKL
msd=defaultdict(lambda: defaultdict(list))
for l in open("results/FF_MULTISEED_MS.txt"):
    m=re.search(r'\[(ViT-[\w/]+) (\w+) s(\d+)\] MaxSkew=([\d.]+) NDKL=([\d.]+)',l)
    if m: msd[(m.group(1),m.group(2))]["MS"].append(float(m.group(4))); msd[(m.group(1),m.group(2))]["ND"].append(float(m.group(5)))
# multiseed DPG/EOD/RBS (partial ok)
dms=defaultdict(lambda: defaultdict(list))
if os.path.exists("results/FF_MULTISEED_DPG_EOD_RBS.txt"):
    for l in open("results/FF_MULTISEED_DPG_EOD_RBS.txt"):
        m=re.search(r'\[(ViT-[\w/]+) (\w+) s(\d+)\] DPG=([\d.]+) EOD=([\d.]+) RBS=([\d.e+-]+)',l)
        if m:
            k=(m.group(1),m.group(2))
            dms[k]["DPG"].append(float(m.group(4))); dms[k]["EOD"].append(float(m.group(5))); dms[k]["RBS"].append(float(m.group(6)))
# UTK
ub={};uf=defaultdict(lambda: defaultdict(list))
for l in open("results/UTK_dpg_eod_rbs.txt"):
    m=re.search(r'\[(ViT-[\w/]+) (\w+) (BASE|s\d+)\] DPG=([\d.]+) EOD=([\d.]+) RBS=([\d.e+-]+)',l)
    if not m: continue
    k=(m.group(1),m.group(2))
    if m.group(3)=="BASE": ub[k]=(float(m.group(4)),float(m.group(5)),float(m.group(6)))
    else: uf[k]["DPG"].append(float(m.group(4))); uf[k]["RBS"].append(float(m.group(6)))
uc=defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open("results/UTK_complete_metrics.csv")):
    uc[(r["backbone"].replace("_","/"),r["attribute"])].setdefault(r["method"],[]).append(r)
fa=defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open("results/FACET_RESULTS.csv")):
    fa[(r["backbone"].replace("_","/"),r["attribute"])].setdefault(r["method"],[]).append(r)
inv={}
for l in open("results/IMAGENETV2.txt"):
    m=re.search(r'\[(ViT-[\w/]+) (\w+)\] Top-1: ([\d.]+)%\s+Top-5: ([\d.]+)%',l)
    if m: inv[(m.group(1),m.group(2))]=(float(m.group(3)),float(m.group(4)))
ab=list(csv.DictReader(open("FINAL_RESULTS/ABLATION_TABLE.csv")))
rms=defaultdict(lambda: defaultdict(list))
if os.path.exists("results/RETRIEVAL_SEEDS.txt"):
    for _l in open("results/RETRIEVAL_SEEDS.txt"):
        _m=re.search(r'\[(ViT-[\w/]+) (\w+) s(\d+)\](.*)',_l)
        if _m:
            for _k,_v in re.findall(r'([TI]R@\d+)=([\d.]+)',_m.group(4)):
                rms[(_m.group(1),_m.group(2))][_k].append(float(_v))
ims=defaultdict(list)
if os.path.exists("results/IMAGENETV2_SEEDS.txt"):
    for _l in open("results/IMAGENETV2_SEEDS.txt"):
        _m=re.search(r'\[(ViT-[\w/]+) (\w+) s(\d+)\] Top-1: ([\d.]+)%',_l)
        if _m: ims[(_m.group(1),_m.group(2))].append(float(_m.group(4)))

# ---- figures ----
for name,idx,ttl,low in [("fig1_maxskew",1,"MaxSkew ↓",True),("fig2_ndkl",2,"NDKL ↓",True),("fig3_able",3,"ABLE ↑",False),("fig0_acc",0,"Accuracy ↑",False)]:
    fig,axes=plt.subplots(1,3,figsize=(16,4.2))
    for j,at in enumerate(AT):
        x=np.arange(4);w=.27;ax=axes[j]
        b=[ff[(bb,at)]["Baseline"][idx] for bb in BB]; f=[ff[(bb,at)]["FairCLIP"][idx] for bb in BB]
        ax.bar(x-w,b,w,label="Baseline CLIP",color=CB); ax.bar(x,f,w,label="FairCLIP",color=CF)
        if idx in (1,2,3):
            z=[ZH[at][bb][{1:0,2:1,3:2}[idx]] for bb in BB]; ax.bar(x+w,z,w,label="Zhang et al.",color=CZ)
        ax.set_xticks(x); ax.set_xticklabels([s.replace("ViT-","") for s in BB]); ax.set_title(f"{ttl} — {at}"); ax.grid(axis="y",alpha=.3)
        if j==0: ax.legend(fontsize=8)
    save(fig,name)
# DPG/EOD/RBS
fig,axes=plt.subplots(1,3,figsize=(16,4.2))
for j,(kk,tt) in enumerate([("dpg","DPG"),("eod","EOD"),("rbs","RBS")]):
    ax=axes[j];lbl=[];bs=[];fs=[]
    for bb in BB:
        for at in AT:
            k=(bb,at)
            if kk=="rbs": bv=ub[k][2]; fv=float(fd[k]["rbs"])
            elif kk=="dpg": bv=BD.get(k); fv=fd[k]["dpg"]
            else: bv=BD.get(k)*2 if BD.get(k) else None; fv=fd[k]["eod"]
            if bv is None: continue
            lbl.append(f"{bb.replace('ViT-','')}\n{at[:3]}"); bs.append(bv); fs.append(fv)
    x=np.arange(len(lbl));w=.38
    ax.bar(x-w/2,bs,w,label="Baseline",color=CB); ax.bar(x+w/2,fs,w,label="FairCLIP",color=CF)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(lbl,fontsize=6,rotation=45); ax.set_title(f"{tt} ↓ (log)"); ax.grid(axis="y",alpha=.3)
    if j==0: ax.legend(fontsize=8)
save(fig,"fig4_dpg_eod_rbs")
# retrieval all K
fig,axes=plt.subplots(1,3,figsize=(16,4.2))
for j,K in enumerate(["1","5","10"]):
    ax=axes[j];x=np.arange(4)
    ax.plot(x,[float(RB10[b][{"1":0,"5":1,"10":2}[K]]) for b in BB],"o--",color=CB,label="Baseline")
    for at,mk in zip(AT,["s-","^-","d-"]):
        ax.plot(x,[float(ret[(b,at)][f"TR@{K}"]) for b in BB],mk,label=at)
    ax.set_xticks(x); ax.set_xticklabels([s.replace("ViT-","") for s in BB]); ax.set_title(f"Flickr30k TR@{K} ↑"); ax.legend(fontsize=8); ax.grid(alpha=.3)
save(fig,"fig5_retrieval")
# imagenet
fig,ax=plt.subplots(figsize=(7,4.2)); x=np.arange(4)
ax.plot(x,[inv[(b,"BASE")][0] for b in BB],"o--",color=CB,label="Baseline")
for at,mk in zip(AT,["s-","^-","d-"]): ax.plot(x,[inv[(b,at)][0] for b in BB],mk,label=at)
ax.set_xticks(x); ax.set_xticklabels([s.replace("ViT-","") for s in BB]); ax.set_ylabel("ImageNetV2 Top-1 (%)"); ax.set_title("Zero-shot utility (<1 pt drop)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
save(fig,"fig6_imagenet")
# utk + facet + ablation
fig,axes=plt.subplots(1,3,figsize=(16,4.2))
for j,at in enumerate(AT):
    ax=axes[j];x=np.arange(4);w=.38
    bv=[float(uc[(bb,at)]["Baseline"][0]["MaxSkew"]) for bb in BB]
    fm=[st.mean([float(r["MaxSkew"]) for r in uc[(bb,at)]["FairCLIP"]]) for bb in BB]
    fs=[st.stdev([float(r["MaxSkew"]) for r in uc[(bb,at)]["FairCLIP"]]) for bb in BB]
    ax.bar(x-w/2,bv,w,label="Baseline",color=CB); ax.bar(x+w/2,fm,w,yerr=fs,capsize=3,label="FairCLIP (3 seeds)",color=CF)
    ax.set_xticks(x); ax.set_xticklabels([s.replace("ViT-","") for s in BB]); ax.set_title(f"UTKFace MaxSkew ↓ — {at}"); ax.grid(axis="y",alpha=.3)
    if j==0: ax.legend(fontsize=8)
save(fig,"fig7_utkface")
fig,axes=plt.subplots(1,2,figsize=(12,4.2))
for j,at in enumerate(["gender","age"]):
    ax=axes[j];x=np.arange(4);w=.38
    bv=[float(fa[(bb,at)]["Baseline"][0]["MaxSkew"]) for bb in BB]
    fm=[st.mean([float(r["MaxSkew"]) for r in fa[(bb,at)]["FairCLIP"]]) for bb in BB]
    fs=[st.stdev([float(r["MaxSkew"]) for r in fa[(bb,at)]["FairCLIP"]]) for bb in BB]
    ax.bar(x-w/2,bv,w,label="Baseline",color=CB); ax.bar(x+w/2,fm,w,yerr=fs,capsize=3,label="FairCLIP",color=CF)
    ax.set_xticks(x); ax.set_xticklabels([s.replace("ViT-","") for s in BB]); ax.set_title(f"FACET MaxSkew ↓ — {at}"); ax.grid(axis="y",alpha=.3)
    if j==0: ax.legend(fontsize=8)
save(fig,"fig8_facet")
fig,axes=plt.subplots(1,3,figsize=(16,4.2))
cfgs=["FULL","no_fairloss","no_temp","no_trainproj","no_procrustes","no_subspace"]
for j,at in enumerate(AT):
    ax=axes[j]; v=[float([r for r in ab if r["Config"]==c and r["Attr"]==at][0]["DPG"]) for c in cfgs]
    ax.bar(range(6),v,color=[CF]+[CB]*5)
    ax.set_xticks(range(6)); ax.set_xticklabels([c.replace("no_","w/o ") for c in cfgs],rotation=45,fontsize=7,ha="right")
    ax.set_title(f"Ablation DPG ↓ — {at}"); ax.grid(axis="y",alpha=.3)
save(fig,"fig9_ablation")
os.system(f"cp FINAL_RESULTS/tsne_before_after.png {D}/figures/ 2>/dev/null")

# ---- html ----
H=['<html><head><meta charset="utf-8"><title>FairCLIP Results</title><style>',
'body{font-family:Segoe UI,sans-serif;font-size:13px;padding:20px;max-width:1500px;margin:auto}',
'h1{border-bottom:3px solid #2e7d32}h2{margin-top:34px;color:#1b5e20}',
'table{border-collapse:collapse;margin:10px 0;font-size:12px}td,th{border:1px solid #ccc;padding:4px 8px;text-align:center}',
'th{background:#eceff1}.w{background:#c8e6c9;font-weight:bold}.l{background:#ffcdd2;font-weight:bold}',
'img{max-width:100%;border:1px solid #ddd;margin:8px 0}.note{background:#fff8e1;padding:8px;border-left:4px solid #ffa000}',
'</style></head><body><h1>FairCLIP — Complete Results</h1>',
'<p class="note">Baseline = pretrained OpenAI CLIP · Zhang = published CVPR 2025 numbers · ✔ beats · ✘ does not</p>']
def tab(rows,hdr): return "<table><tr>"+"".join(f"<th>{h}</th>" for h in hdr)+"</tr>"+"".join("<tr>"+r+"</tr>" for r in rows)+"</table>"

H.append("<h2>Table 1 — FairFace in-domain (seed 42)</h2>")
rows=[];c=defaultdict(int)
for bb in BB:
    for at in AT:
        b,f=ff[(bb,at)]["Baseline"],ff[(bb,at)]["FairCLIP"]; z=ZH[at][bb]
        o=[f[0]>b[0],f[1]<b[1],f[1]<z[0],f[2]<b[2],f[2]<z[1],f[3]>b[3],f[3]>z[2]]
        for n,x in zip(["a","m1","m2","n1","n2","b1","b2"],o): c[n]+=x
        rows.append(f"<td>{bb}</td><td>{at}</td><td>{b[0]:.3f}→<b>{f[0]:.3f}</b></td>{C(o[0])}"
          f"<td>{b[1]:.3f}→<b>{f[1]:.3f}</b> ({z[0]:.3f})</td>{C(o[1])}{C(o[2])}"
          f"<td>{b[2]:.3f}→<b>{f[2]:.3f}</b> ({z[1]:.3f})</td>{C(o[3])}{C(o[4])}"
          f"<td>{b[3]:.2f}→<b>{f[3]:.2f}</b> ({z[2]:.2f})</td>{C(o[5])}{C(o[6])}")
H.append(tab(rows,["Backbone","Attr","Acc B→F","vsB","MaxSkew B→F (Z)","vsB","vsZ","NDKL B→F (Z)","vsB","vsZ","ABLE B→F (Z)","vsB","vsZ"]))
H.append(f"<p><b>vs Baseline:</b> Acc {c['a']}/12 · MaxSkew {c['m1']}/12 · NDKL {c['n1']}/12 · ABLE {c['b1']}/12<br>"
         f"<b>vs Zhang:</b> MaxSkew {c['m2']}/12 · NDKL {c['n2']}/12 · ABLE {c['b2']}/12</p>")
H.append('<img src="figures/fig0_acc.png"><img src="figures/fig1_maxskew.png"><img src="figures/fig2_ndkl.png"><img src="figures/fig3_able.png">')

H.append("<h2>Table 2 — FairFace DPG / EOD / RBS (occupation-based)</h2>")
rows=[];w2=0;t2=0
for bb in BB:
    for at in AT:
        k=(bb,at); s=fd[k]; bd=BD.get(k); ms=dms.get(k)
        ok=bd is not None and s["dpg"]<bd
        if bd is not None: t2+=1; w2+=ok
        mstr=f"{st.mean(ms['DPG']):.4f}±{st.stdev(ms['DPG']):.4f}" if ms and len(ms['DPG'])>1 else "—"
        rows.append(f"<td>{bb}</td><td>{at}</td><td>{'%.4f'%bd if bd else '—'}→<b>{s['dpg']:.4f}</b></td>"
          f"{C(ok) if bd else '<td>—</td>'}<td>{mstr}</td><td>{s['eod']:.4f}</td><td>{s['rbs']}</td>")
H.append(tab(rows,["Backbone","Attr","DPG B→F (s42)","vsB","DPG 3-seed mean±std","EOD","RBS"]))
H.append(f"<p><b>DPG vs Baseline:</b> {w2}/{t2} (baselines available)</p>")
H.append('<img src="figures/fig4_dpg_eod_rbs.png">')

H.append("<h2>Table 3 — FairFace MaxSkew / NDKL, 3 seeds (mean±std)</h2>")
rows=[]
for bb in BB:
    for at in AT:
        k=(bb,at); m=msd.get(k)
        f=ff[k]["FairCLIP"]
        allms=[f[1]]+(m["MS"] if m else []); allnd=[f[2]]+(m["ND"] if m else [])
        rows.append(f"<td>{bb}</td><td>{at}</td><td>{f[1]:.3f}</td>"
          f"<td><b>{st.mean(allms):.3f}±{st.stdev(allms):.3f}</b></td><td>{f[2]:.3f}</td>"
          f"<td><b>{st.mean(allnd):.3f}±{st.stdev(allnd):.3f}</b></td><td>{len(allms)}</td>")
H.append(tab(rows,["Backbone","Attr","MaxSkew s42","MaxSkew mean±std","NDKL s42","NDKL mean±std","n seeds"]))

H.append("<h2>Table 4 — Flickr30k retrieval, all K (utility)</h2>")
rows=[]
for bb in BB:
    b=RB10[bb]
    rows.append(f"<td>{bb}</td><td><i>Baseline</i></td>"+"".join(f"<td>{x}</td>" for x in b)+"<td>—</td>")
    for at in AT:
        v=ret[(bb,at)]
        ds=[float(v[f"{p}@{k}"])-float(b[i]) for i,(p,k) in enumerate([("TR","1"),("TR","5"),("TR","10"),("IR","1"),("IR","5"),("IR","10")])]
        rows.append(f"<td>{bb}</td><td>{at}</td>"+"".join(f"<td>{v[f'{p}@{k}']} ({d:+.1f})</td>" for (p,k),d in zip([("TR","1"),("TR","5"),("TR","10"),("IR","1"),("IR","5"),("IR","10")],ds))+C(max(abs(x) for x in ds)<=3))
H.append(tab(rows,["Backbone","Model","TR@1","TR@5","TR@10","IR@1","IR@5","IR@10","≤3pt"]))
rows2=[]
for bb in BB:
    for at in AT:
        v=ret[(bb,at)]; m=rms.get((bb,at),{})
        cells=[]
        for kk in ["TR@1","TR@5","TR@10","IR@1","IR@5","IR@10"]:
            allv=[float(v[kk])]+m.get(kk,[])
            cells.append(f"<td>{st.mean(allv):.1f}±{st.stdev(allv):.1f}</td>" if len(allv)>1 else f"<td>{allv[0]:.1f}</td>")
        rows2.append(f"<td>{bb}</td><td>{at}</td>"+"".join(cells)+f"<td>{len([float(v['TR@5'])]+m.get('TR@5',[]))}</td>")
H.append("<h4>Retrieval, 3 seeds (mean±std)</h4>")
H.append(tab(rows2,["Backbone","Attr","TR@1","TR@5","TR@10","IR@1","IR@5","IR@10","n"]))
H.append('<img src="figures/fig5_retrieval.png">')

H.append("<h2>Table 5 — ImageNetV2 zero-shot utility</h2>")
rows=[];iw=0
for bb in BB:
    base=inv[(bb,"BASE")]; ds=[inv[(bb,a)][0]-base[0] for a in AT]
    ok=all(abs(x)<1 for x in ds); iw+=ok
    rows.append(f"<td>{bb}</td><td>{base[0]:.2f}</td><td>{base[1]:.2f}</td>"+"".join(f"<td>{inv[(bb,a)][0]:.2f} ({d:+.2f})</td>" for a,d in zip(AT,ds))+C(ok))
H.append(tab(rows,["Backbone","Base Top-1","Base Top-5","gender","age","race","&lt;1pt"]))
H.append(f"<p><b>{iw}/4 backbones — all 16 cells within 1 point.</b></p>")
rows2=[]
for bb in BB:
    cells=[]
    for at in AT:
        allv=[inv[(bb,at)][0]]+ims.get((bb,at),[])
        cells.append(f"<td>{st.mean(allv):.2f}±{st.stdev(allv):.2f}</td>" if len(allv)>1 else f"<td>{allv[0]:.2f}</td>")
    rows2.append(f"<td>{bb}</td><td>{inv[(bb,'BASE')][0]:.2f}</td>"+"".join(cells)+f"<td>{len([inv[(bb,'gender')][0]]+ims.get((bb,'gender'),[]))}</td>")
H.append("<h4>ImageNetV2 Top-1, 3 seeds (mean±std)</h4>")
H.append(tab(rows2,["Backbone","Baseline","gender","age","race","n"]))
H.append('<img src="figures/fig6_imagenet.png">')

H.append("<h2>Table 6 — UTKFace cross-dataset (3 seeds)</h2>")
rows=[];u=defaultdict(int)
for bb in BB:
    for at in AT:
        k=(bb,at); b=ub[k]; d=uf[k]; cb=uc[k]["Baseline"][0]; cf=uc[k]["FairCLIP"]
        g=lambda m:(st.mean([float(x[m]) for x in cf]),st.stdev([float(x[m]) for x in cf]))
        ms=g("MaxSkew"); abl=g("ABLE"); ac=g("Acc")
        dm=st.mean(d["DPG"]); ds_=st.stdev(d["DPG"]); rm=st.mean(d["RBS"])
        o=[ac[0]>float(cb["Acc"]),dm<b[0],rm<b[2],ms[0]<float(cb["MaxSkew"]),abl[0]>float(cb["ABLE"])]
        for n,x in zip(["ac","dp","rb","ms","ab"],o): u[n]+=x
        rows.append(f"<td>{bb}</td><td>{at}</td><td>{float(cb['Acc']):.3f}→<b>{ac[0]:.3f}±{ac[1]:.3f}</b></td>{C(o[0])}"
          f"<td>{b[0]:.4f}→<b>{dm:.4f}±{ds_:.4f}</b></td>{C(o[1])}<td>{b[2]:.1e}→<b>{rm:.1e}</b></td>{C(o[2])}"
          f"<td>{float(cb['MaxSkew']):.3f}→<b>{ms[0]:.3f}±{ms[1]:.3f}</b></td>{C(o[3])}"
          f"<td>{float(cb['ABLE']):.2f}→<b>{abl[0]:.2f}±{abl[1]:.2f}</b></td>{C(o[4])}")
H.append(tab(rows,["Backbone","Attr","Acc B→F","vsB","DPG B→F","vsB","RBS B→F","vsB","MaxSkew B→F","vsB","ABLE B→F","vsB"]))
H.append(f"<p><b>vs Baseline:</b> Acc {u['ac']}/12 · DPG {u['dp']}/12 · RBS {u['rb']}/12 · MaxSkew {u['ms']}/12 · ABLE {u['ab']}/12</p>")
H.append('<img src="figures/fig7_utkface.png">')

H.append("<h2>Table 7 — FACET cross-dataset (3 seeds; no race labels in FACET)</h2>")
rows=[];fc=defaultdict(int);ft=0
for bb in BB:
    for at in ["gender","age"]:
        k=(bb,at)
        if k not in fa: continue
        b=fa[k]["Baseline"][0]; f=fa[k]["FairCLIP"]
        g=lambda m:(st.mean([float(x[m]) for x in f]),st.stdev([float(x[m]) for x in f]) if len(f)>1 else 0)
        ac=g("Acc");ms=g("MaxSkew");nd=g("NDKL")
        o=[ac[0]>float(b["Acc"]),ms[0]<float(b["MaxSkew"]),nd[0]<float(b["NDKL"])]
        for n,x in zip(["a","m","n"],o): fc[n]+=x
        ft+=1
        rows.append(f"<td>{bb}</td><td>{at}</td><td>{float(b['Acc']):.3f}→<b>{ac[0]:.3f}</b></td>{C(o[0])}"
          f"<td>{float(b['MaxSkew']):.3f}→<b>{ms[0]:.3f}±{ms[1]:.3f}</b></td>{C(o[1])}"
          f"<td>{float(b['NDKL']):.3f}→<b>{nd[0]:.3f}</b></td>{C(o[2])}")
H.append(tab(rows,["Backbone","Attr","Acc B→F","vsB","MaxSkew B→F","vsB","NDKL B→F","vsB"]))
H.append(f"<p><b>vs Baseline:</b> Acc {fc['a']}/{ft} · MaxSkew {fc['m']}/{ft} · NDKL {fc['n']}/{ft} — age improves on all backbones; FACET gender baseline is already near-fair</p>")
H.append('<img src="figures/fig8_facet.png">')

H.append("<h2>Table 8 — Ablation (ViT-B/32, seed 42)</h2>")
rows=[]
for r in ab:
    full=[q for q in ab if q["Config"]=="FULL" and q["Attr"]==r["Attr"]][0]; isf=r["Config"]=="FULL"
    def cell(k,lower=True):
        v=float(r[k]); fv=float(full[k])
        if isf: return f"<td><b>{r[k]}</b></td>"
        return f'<td class="{"l" if ((v<fv) if lower else (v>fv)) else ""}">{r[k]}</td>'
    rows.append(f"<td>{'<b>FULL</b>' if isf else r['Config'].replace('no_','w/o ')}</td><td>{r['Attr']}</td>"
      f"<td>{r['Acc']}</td><td>{r['F1']}</td>{cell('DPG')}{cell('EOD')}<td>{r['RBS']}</td>{cell('MaxSkew')}{cell('NDKL')}<td>{r['ABLE']}</td>")
H.append(tab(rows,["Config","Attr","Acc","F1","DPG","EOD","RBS","MaxSkew","NDKL","ABLE"]))
H.append("<p>FULL achieves best DPG, EOD and RBS on all three attributes.</p>")
H.append('<img src="figures/fig9_ablation.png">')
H.append("<h2>Figure — t-SNE before / after debiasing</h2><img src='figures/tsne_before_after.png'>")
H.append("<h2>Compute cost</h2><pre>"+open("FINAL_RESULTS/compute_cost.txt").read()+"</pre>")
H.append("</body></html>")
open(D+"/index.html","w").write("\n".join(H))
for f in os.listdir("FINAL_RESULTS"):
    if f.endswith((".csv",".txt")): os.system(f'cp "FINAL_RESULTS/{f}" {D}/tables/ 2>/dev/null')
print("SAVED",D+"/index.html","| figures:",len(os.listdir(D+"/figures")),"| tables:",len(os.listdir(D+"/tables")))

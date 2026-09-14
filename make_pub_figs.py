import os,re,csv,numpy as np,matplotlib,statistics as st
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import defaultdict
P="FINAL_RESULTS/FairCLIP_Results/publication"; os.makedirs(P+"/figures",exist_ok=True); os.makedirs(P+"/latex",exist_ok=True)
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.titlesize":10,"xtick.labelsize":8,
 "ytick.labelsize":8,"legend.fontsize":8,"axes.linewidth":.8,"figure.dpi":600,"savefig.dpi":600,
 "axes.spines.top":False,"axes.spines.right":False})
CB="#7f8c9b"; CF="#1b7837"; CZ="#b2182b"
BB=["ViT-B/32","ViT-B/16","ViT-L/14","ViT-H/14"]; SB=[b.replace("ViT-","") for b in BB]; AT=["gender","age","race"]
ZH={"gender":{"ViT-B/32":(.090,.030,74.24),"ViT-B/16":(.080,.025,78.35),"ViT-L/14":(.106,.035,82.04),"ViT-H/14":(.138,.051,82.11)},
    "age":{"ViT-B/32":(.572,.364,59.60),"ViT-B/16":(.608,.294,60.61),"ViT-L/14":(.579,.332,64.25),"ViT-H/14":(.515,.289,67.62)},
    "race":{b:(.353,.125,69.14) for b in BB}}
def out(fig,n):
    fig.tight_layout(pad=.4,w_pad=1.2)
    fig.tight_layout(pad=.4,w_pad=1.2)
    fig.savefig(f"{P}/figures/{n}.png",bbox_inches="tight",dpi=600); plt.close(fig)

ff=defaultdict(dict)
for l in open("complete_metrics.txt"):
    m=re.search(r'(ViT-[\w_]+) (\w+) (Baseline|FairCLIP): Acc=([\d.]+) .*MaxSkew=([\d.]+) NDKL=([\d.]+) ABLE=([\d.]+)',l)
    if m: ff[(m.group(1).replace("_","/"),m.group(2))][m.group(3)]=[float(m.group(i)) for i in (4,5,6,7)]
msd=defaultdict(lambda: defaultdict(list))
for l in open("results/FF_MULTISEED_MS.txt"):
    m=re.search(r'\[(ViT-[\w/]+) (\w+) s\d+\] MaxSkew=([\d.]+) NDKL=([\d.]+)',l)
    if m: msd[(m.group(1),m.group(2))]["MS"].append(float(m.group(3))); msd[(m.group(1),m.group(2))]["ND"].append(float(m.group(4)))
dms=defaultdict(lambda: defaultdict(list))
for l in open("results/FF_MULTISEED_DPG_EOD_RBS.txt"):
    m=re.search(r'\[(ViT-[\w/]+) (\w+) s\d+\] DPG=([\d.]+) EOD=([\d.]+) RBS=([\d.e+-]+)',l)
    if m:
        k=(m.group(1),m.group(2))
        dms[k]["DPG"].append(float(m.group(3))); dms[k]["EOD"].append(float(m.group(4))); dms[k]["RBS"].append(float(m.group(5)))
BD={("ViT-B/32","gender"):(.0315,.0631,1.68e-2),("ViT-B/32","age"):(.0817,.1635,5.94e-2),("ViT-B/32","race"):(.0417,.0834,4.11e-2),
    ("ViT-B/16","age"):(.0791,.1581,5.61e-2),("ViT-B/16","race"):(.0463,.0926,4.08e-2),
    ("ViT-L/14","gender"):(.0184,.0368,1.66e-2),("ViT-L/14","age"):(.0683,.1365,4.92e-2),("ViT-L/14","race"):(.0364,.0728,3.34e-2)}
for l in open("results/BASELINE_DPG_MISSING.txt"):
    m=re.search(r'\[(ViT-[\w/]+) (\w+)\] DPG=([\d.]+) EOD=([\d.]+) RBS=([\d.e+-]+)',l)
    if m: BD[(m.group(1),m.group(2))]=(float(m.group(3)),float(m.group(4)),float(m.group(5)))
ret={};rms=defaultdict(lambda: defaultdict(list))
for l in open("results/RETRIEVAL_ALL_K.txt"):
    m=re.search(r'\[(ViT-[\w/]+) (\w+)\](.*)',l)
    if m and m.group(2)!="BASE": ret[(m.group(1),m.group(2))]=dict(re.findall(r'([TI]R@\d+)=([\d.]+)',m.group(3)))
for l in open("results/RETRIEVAL_SEEDS.txt"):
    m=re.search(r'\[(ViT-[\w/]+) (\w+) s\d+\](.*)',l)
    if m:
        for k,v in re.findall(r'([TI]R@\d+)=([\d.]+)',m.group(3)): rms[(m.group(1),m.group(2))][k].append(float(v))
RB={"ViT-B/32":(16.9,78.4,89.3,18.6,77.4,89.4),"ViT-B/16":(16.7,81.5,91.1,18.8,78.9,91.7),
    "ViT-L/14":(17.1,82.1,90.7,18.7,80.4,91.5),"ViT-H/14":(18.4,89.1,95.6,19.2,87.3,96.0)}
inv={};ims=defaultdict(list)
for l in open("results/IMAGENETV2.txt"):
    m=re.search(r'\[(ViT-[\w/]+) (\w+)\] Top-1: ([\d.]+)%\s+Top-5: ([\d.]+)%',l)
    if m: inv[(m.group(1),m.group(2))]=(float(m.group(3)),float(m.group(4)))
for l in open("results/IMAGENETV2_SEEDS.txt"):
    m=re.search(r'\[(ViT-[\w/]+) (\w+) s\d+\] Top-1: ([\d.]+)%',l)
    if m: ims[(m.group(1),m.group(2))].append(float(m.group(3)))
uc=defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open("results/UTK_complete_metrics.csv")):
    uc[(r["backbone"].replace("_","/"),r["attribute"])].setdefault(r["method"],[]).append(r)
fa=defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open("results/FACET_RESULTS.csv")):
    fa[(r["backbone"].replace("_","/"),r["attribute"])].setdefault(r["method"],[]).append(r)
ab=list(csv.DictReader(open("FINAL_RESULTS/ABLATION_TABLE.csv")))

def trio(idx,ylab,zidx,fname,seeds=None):
    fig,axes=plt.subplots(1,3,figsize=(7.16,2.6))
    x=np.arange(4); w=.26
    for j,at in enumerate(AT):
        ax=axes[j]
        b=[ff[(bb,at)]["Baseline"][idx] for bb in BB]; er=None
        f=[ff[(bb,at)]["FairCLIP"][idx] for bb in BB]
        if seeds:
            f=[];er=[]
            for bb in BB:
                v=[ff[(bb,at)]["FairCLIP"][idx]]+msd[(bb,at)][seeds]
                f.append(st.mean(v)); er.append(st.stdev(v))
        ax.bar(x-w,b,w,label="Baseline CLIP",color=CB,edgecolor="black",linewidth=.4)
        ax.bar(x,f,w,yerr=er,capsize=2,label="FairCLIP (ours)",color=CF,edgecolor="black",linewidth=.4,error_kw=dict(lw=.7))
        if zidx is not None:
            ax.bar(x+w,[ZH[at][bb][zidx] for bb in BB],w,label="Zhang et al.",color=CZ,edgecolor="black",linewidth=.4)
        ax.set_xticks(x); ax.set_xticklabels(SB); ax.set_xlabel("CLIP backbone")
        ax.set_xlabel(f"CLIP backbone ({at})")
        ax.set_ylabel(ylab)
        ax.grid(axis="y",alpha=.25,linestyle=":")
    axes[1].legend(ncol=3,loc="upper center",bbox_to_anchor=(.5,1.34),frameon=False)
    out(fig,fname)

trio(1,"MaxSkew (lower better)",0,"fig1_maxskew",seeds="MS")
trio(2,"NDKL (lower better)",1,"fig2_ndkl",seeds="ND")
trio(3,"ABLE (higher better)",2,"fig3_able")
trio(0,"Top-1 accuracy",None,"fig4_accuracy")

fig,axes=plt.subplots(1,3,figsize=(7.16,2.6))
for j,(kk,tt,bi) in enumerate([("DPG","DPG (lower better)",0),("EOD","EOD (lower better)",1),("RBS","RBS (lower better)",2)]):
    ax=axes[j]; lbl=[];bs=[];fs=[];er=[]
    for bb in BB:
        for at in AT:
            k=(bb,at); bs.append(BD[k][bi]); v=dms[k][kk]
            fs.append(st.mean(v)); er.append(st.stdev(v)); lbl.append(f"{bb.replace('ViT-','')}/{at[0].upper()}")
    x=np.arange(len(lbl)); w=.38
    ax.bar(x-w/2,bs,w,label="Baseline",color=CB,edgecolor="black",linewidth=.3)
    ax.bar(x+w/2,fs,w,yerr=er,capsize=1.5,label="FairCLIP",color=CF,edgecolor="black",linewidth=.3,error_kw=dict(lw=.6))
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(lbl,rotation=90,fontsize=5.5); ax.set_xlabel("Backbone / attribute")
    ax.set_xlabel("CLIP backbone / attribute"); ax.set_ylabel(tt); ax.grid(axis="y",alpha=.25,linestyle=":")
    if j==0: ax.legend(frameon=False)
out(fig,"fig5_dpg_eod_rbs")

fig,axes=plt.subplots(1,3,figsize=(7.16,2.5))
for j,K in enumerate(["1","5","10"]):
    ax=axes[j]; x=np.arange(4); i={"1":0,"5":1,"10":2}[K]
    ax.plot(x,[RB[b][i] for b in BB],"o--",color=CB,label="Baseline",ms=4)
    for at,mk in zip(AT,["s-","^-","d-"]):
        y=[];e=[]
        for b in BB:
            v=[float(ret[(b,at)][f"TR@{K}"])]+rms[(b,at)][f"TR@{K}"]
            y.append(st.mean(v)); e.append(st.stdev(v))
        ax.errorbar(x,y,yerr=e,fmt=mk,ms=3.5,capsize=2,label=at,lw=1.2,elinewidth=.7)
    ax.set_xticks(x); ax.set_xticklabels(SB); ax.set_xlabel("CLIP backbone")
    ax.set_xlabel("CLIP backbone"); ax.set_ylabel(f"Flickr30k TR@{K} (%)")
    ax.grid(alpha=.25,linestyle=":")
axes[2].legend(frameon=False,fontsize=7)
out(fig,"fig6_retrieval")

fig,ax=plt.subplots(figsize=(3.6,2.6)); x=np.arange(4)
ax.plot(x,[inv[(b,"BASE")][0] for b in BB],"o--",color=CB,label="Baseline",ms=4)
for at,mk in zip(AT,["s-","^-","d-"]):
    y=[];e=[]
    for b in BB:
        v=[inv[(b,at)][0]]+ims[(b,at)]; y.append(st.mean(v)); e.append(st.stdev(v))
    ax.errorbar(x,y,yerr=e,fmt=mk,ms=3.5,capsize=2,label=at,lw=1.2,elinewidth=.7)
ax.set_xticks(x); ax.set_xticklabels(SB); ax.set_xlabel("CLIP backbone"); ax.set_xlabel("CLIP backbone"); ax.set_ylabel("ImageNetV2 Top-1 accuracy (%)")
ax.legend(frameon=False,fontsize=7); ax.grid(alpha=.25,linestyle=":")
out(fig,"fig7_imagenet")

for tag,src,attrs,fn in [("UTKFace",uc,AT,"fig8_utkface"),("FACET",fa,["gender","age"],"fig9_facet")]:
    fig,axes=plt.subplots(1,len(attrs),figsize=(7.16 if len(attrs)==3 else 5.2,2.6))
    for j,at in enumerate(attrs):
        ax=axes[j]; x=np.arange(4); w=.38
        bv=[float(src[(bb,at)]["Baseline"][0]["MaxSkew"]) for bb in BB]
        fm=[st.mean([float(r["MaxSkew"]) for r in src[(bb,at)]["FairCLIP"]]) for bb in BB]
        fe=[st.stdev([float(r["MaxSkew"]) for r in src[(bb,at)]["FairCLIP"]]) for bb in BB]
        ax.bar(x-w/2,bv,w,label="Baseline",color=CB,edgecolor="black",linewidth=.4)
        ax.bar(x+w/2,fm,w,yerr=fe,capsize=2,label="FairCLIP",color=CF,edgecolor="black",linewidth=.4,error_kw=dict(lw=.7))
        ax.set_xticks(x); ax.set_xticklabels(SB); ax.set_xlabel("CLIP backbone")
        ax.set_xlabel(f"CLIP backbone ({at})")
        ax.set_ylabel(f"{tag} MaxSkew (lower better)")
        ax.grid(axis="y",alpha=.25,linestyle=":")
        if j==0: ax.legend(frameon=False)
    out(fig,fn)

cfgs=["FULL","no_fairloss","no_temp","no_trainproj","no_procrustes","no_subspace"]
lab=["Full","w/o Lfair","w/o temp.","w/o proj.","w/o Procr.","w/o subsp."]
fig,axes=plt.subplots(1,3,figsize=(7.16,2.5))
for j,at in enumerate(AT):
    ax=axes[j]; v=[float([r for r in ab if r["Config"]==c and r["Attr"]==at][0]["DPG"]) for c in cfgs]
    ax.bar(range(6),v,color=[CF]+[CB]*5,edgecolor="black",linewidth=.4)
    ax.set_xticks(range(6)); ax.set_xticklabels(lab,rotation=45,ha="right",fontsize=6.5); ax.set_xlabel("Component removed")
    ax.set_xlabel(f"Component removed ({at})")
    ax.set_ylabel("DPG (lower better)")
    ax.grid(axis="y",alpha=.25,linestyle=":")
out(fig,"fig10_ablation")

L=[r"\begin{table*}[t]\centering\small",r"\caption{FairFace in-domain results.}",r"\label{tab:fairface}",
   r"\begin{tabular}{llcccccccccc}\toprule",
   r"& & \multicolumn{2}{c}{Acc $\uparrow$} & \multicolumn{3}{c}{MaxSkew $\downarrow$} & \multicolumn{3}{c}{NDKL $\downarrow$} & \multicolumn{2}{c}{ABLE $\uparrow$}\\",
   r"Backbone & Attr. & Base & Ours & Base & Ours & Zhang & Base & Ours & Zhang & Base & Ours\\\midrule"]
for bb in BB:
    for at in AT:
        b,f=ff[(bb,at)]["Baseline"],ff[(bb,at)]["FairCLIP"]; z=ZH[at][bb]
        L.append(f"{bb} & {at} & {b[0]:.3f} & \\textbf{{{f[0]:.3f}}} & {b[1]:.3f} & \\textbf{{{f[1]:.3f}}} & {z[0]:.3f} & {b[2]:.3f} & \\textbf{{{f[2]:.3f}}} & {z[1]:.3f} & {b[3]:.2f} & \\textbf{{{f[3]:.2f}}}\\\\")
L.append(r"\bottomrule\end{tabular}\end{table*}")
open(P+"/latex/table1_fairface.tex","w").write("\n".join(L))

L=[r"\begin{table}[t]\centering\small",r"\caption{FairFace fairness metrics, mean$\pm$std over three seeds.}",
   r"\begin{tabular}{llccc}\toprule",r"Backbone & Attr. & DPG $\downarrow$ & EOD $\downarrow$ & RBS $\downarrow$\\\midrule"]
for bb in BB:
    for at in AT:
        k=(bb,at); d=dms[k]
        L.append(f"{bb} & {at} & {BD[k][0]:.4f} $\\to$ \\textbf{{{st.mean(d['DPG']):.4f}}}$\\pm${st.stdev(d['DPG']):.4f} & {BD[k][1]:.4f} $\\to$ \\textbf{{{st.mean(d['EOD']):.4f}}} & {BD[k][2]:.1e} $\\to$ \\textbf{{{st.mean(d['RBS']):.1e}}}\\\\")
L.append(r"\bottomrule\end{tabular}\end{table}")
open(P+"/latex/table2_dpg.tex","w").write("\n".join(L))

L=[r"\begin{table}[t]\centering\small",r"\caption{Vision-language utility, mean over three seeds.}",
   r"\begin{tabular}{llcccc}\toprule",r"Backbone & Model & TR@5 & IR@5 & IN-V2 Top-1 & $\Delta$\\\midrule"]
for bb in BB:
    L.append(f"{bb} & Baseline & {RB[bb][1]:.1f} & {RB[bb][4]:.1f} & {inv[(bb,'BASE')][0]:.2f} & --\\\\")
    for at in AT:
        tr=st.mean([float(ret[(bb,at)]["TR@5"])]+rms[(bb,at)]["TR@5"])
        ir=st.mean([float(ret[(bb,at)]["IR@5"])]+rms[(bb,at)]["IR@5"])
        iv=st.mean([inv[(bb,at)][0]]+ims[(bb,at)])
        L.append(f" & {at} & {tr:.1f} & {ir:.1f} & {iv:.2f} & {iv-inv[(bb,'BASE')][0]:+.2f}\\\\")
L.append(r"\bottomrule\end{tabular}\end{table}")
open(P+"/latex/table3_utility.tex","w").write("\n".join(L))

L=[r"\begin{table}[t]\centering\small",r"\caption{Ablation study (ViT-B/32, FairFace).}",
   r"\begin{tabular}{llccccc}\toprule",r"Variant & Attr. & Acc $\uparrow$ & DPG $\downarrow$ & EOD $\downarrow$ & MaxSkew $\downarrow$ & ABLE $\uparrow$\\\midrule"]
for r in ab:
    nm="Full model" if r["Config"]=="FULL" else "w/o "+r["Config"].replace("no_","").replace("_"," ")
    L.append(f"{nm} & {r['Attr']} & {r['Acc']} & {r['DPG']} & {r['EOD']} & {r['MaxSkew']} & {r['ABLE']}\\\\")
L.append(r"\bottomrule\end{tabular}\end{table}")
open(P+"/latex/table4_ablation.tex","w").write("\n".join(L))

os.system(f"cp FINAL_RESULTS/tsne_before_after.png {P}/figures/ 2>/dev/null")
print("SAVED",P)
print("figures:",sorted(os.listdir(P+"/figures")))
print("latex:",sorted(os.listdir(P+"/latex")))

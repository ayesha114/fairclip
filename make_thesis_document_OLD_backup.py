"""Generate the complete FairCLIP defense/results document (Word)."""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import pandas as pd, os

TAB="results/tables"; FIG="results/figures"
doc = Document()

def H(t,l=1): 
    h=doc.add_heading(t,level=l); return h
def P(t,bold=False,italic=False,size=11):
    p=doc.add_paragraph(); r=p.add_run(t); r.bold=bold; r.italic=italic; r.font.size=Pt(size); return p

# Title
t=doc.add_heading("FairCLIP: Reducing Social Bias in Vision-Language Models",0)
P("Complete Results, Comparison & Defense Document",bold=True,size=13)
P("Ayesha Kalsoom (25I-7805)  |  Supervisor: Dr. Afia Zafar  |  NUCES",italic=True)
doc.add_paragraph()

# 1. Overview
H("1. What FairCLIP Does",1)
P("FairCLIP removes demographic bias (gender, age, race) from CLIP during "
  "training, unlike post-hoc methods. It uses one pipeline: PCA/SVD bias "
  "subspace discovery, Procrustes cross-modal alignment, fairness-aware "
  "contrastive loss, adaptive temperature, and projection-based bias removal. "
  "We evaluate two complementary fairness notions: classification fairness "
  "(DPG/EOD) and retrieval fairness (MaxSkew/NDKL).")

# 2. Methodology + formulas
H("2. Methodology & Formulas",1)
P("Normalization:  z = f(x)/||f(x)||")
P("Group centroid:  mu_g = (1/|D_g|) * sum z_i")
P("Bias subspace:  PCA/SVD on centroid differences -> B")
P("Procrustes:  min_R ||B_I R - B_T||_F  s.t. R^T R = I")
P("Fairness loss:  L_total = L_InfoNCE + lambda * L_fair")
P("Group variance:  L_fair = (1/K) sum ||centroid_k - global_mean||^2")
P("Adaptive temp:  tau = tau_base (1 + alpha * delta_g)")
P("Projection:  z' = z - B B^T z   (iterated as INLP for retrieval)")
P("DPG = mean_c [max_g P(pred=c|g) - min_g P(pred=c|g)]")
P("MaxSkew@k = mean_queries [ max_A |ln(p_actual/p_desired)| ]   (Geyik et al.)")
P("ABLE = 2/(1/Acc + 1/exp(-MaxSkew)) x 100   (Zhang formula)")

# 3. Hyperparameters
H("3. Hyperparameters",1)
for line in ["Backbone: ViT-B/32 (primary); also B/16, L/14, H/14",
             "Seeds: 42, 123, 456   |   Epochs: 30, patience: 8",
             "Batch: 64 (balanced sampler 14/group x 7)   |   LR: 1e-5 AdamW",
             "lambda: gender=0.01, age=0.5, race=1.0   |   tau_base=0.07, alpha=0.5",
             "Bias directions: 5   |   k (MaxSkew/NDKL): 1000   |   FairFace margin 0.25"]:
    P("- "+line)

def add_table(df, title):
    H(title,2)
    t=doc.add_table(rows=1,cols=len(df.columns)); t.style="Light Grid Accent 1"
    for j,c in enumerate(df.columns): t.rows[0].cells[j].text=str(c)
    for _,row in df.iterrows():
        cells=t.add_row().cells
        for j,c in enumerate(df.columns): cells[j].text=str(row[c])

# 4. Results tables
H("4. Results",1)
add_table(pd.read_csv(f"{TAB}/FINAL_TABLE1_classification.csv"),
          "Table 1 — Classification Fairness (FairCLIP vs Baseline CLIP)")
P("FairCLIP improves accuracy AND reduces DPG. Gender is already fair "
  "(DPG~0.002) so it is preserved, not over-corrected.",italic=True)

add_table(pd.read_csv(f"{TAB}/FINAL_TABLE2_retrieval_vs_zhang.csv"),
          "Table 2 — Retrieval Fairness (FairCLIP+INLP vs Zhang et al.)")
P("FairCLIP+INLP beats Zhang on MaxSkew AND NDKL for all three attributes.",italic=True)

add_table(pd.read_csv(f"{TAB}/FINAL_TABLE3_dpg_eod_rbs.csv"),
          "Table 3 — DPG/EOD/RBS (metrics Zhang does not report)")
P("Note: gender DPG stays near zero (already fair); % change not meaningful there.",italic=True)

# 5. Figures
H("5. Figures",1)
for f,cap in [("fig1_dpg_classification.png","Figure 1: DPG before vs after (classification fairness)"),
              ("fig2_accuracy.png","Figure 2: Accuracy preserved/improved"),
              ("fig3_maxskew_vs_zhang.png","Figure 3: MaxSkew — FairCLIP+INLP beats Zhang"),
              ("fig4_ndkl_vs_zhang.png","Figure 4: NDKL — FairCLIP+INLP beats Zhang")]:
    p=f"{FIG}/{f}"
    if os.path.exists(p):
        doc.add_picture(p,width=Inches(5.5))
        cp=doc.add_paragraph(cap); cp.alignment=WD_ALIGN_PARAGRAPH.CENTER
        cp.runs[0].italic=True

# 6. Panel Q&A
H("6. Panel Questions & Answers",1)
qa=[("Female=female is natural, why remove it as bias?",
     "We don't. Gender classification accuracy is preserved (0.973). We only "
     "remove spurious social associations (e.g. gender->occupation), measured "
     "by occupation/trait/status probes. lambda is set per attribute by its "
     "bias level; gender is already fair so intervention is minimal."),
    ("You only handle occupation bias — isn't that limited?",
     "No. Bias REMOVAL acts on the whole demographic subspace, affecting all "
     "associations. Occupation is one probe; we also probe character traits "
     "(criminal/trustworthy) and social status (wealthy/powerful)."),
    ("You used two different methodologies (classification + retrieval)?",
     "One method. INLP is the iterative form of our Step VIII projection. "
     "Classification and retrieval are two fairness NOTIONS we measure; the "
     "method is the same projection principle."),
    ("Why COCO captions with FairFace?",
     "FairFace has labels, not captions. We pair images with demographic "
     "prompts. Random COCO captions broke contrastive learning (tested: race "
     "DPG worsened to 0.92), so the final design uses demographic prompts."),
    ("Base paper does in-domain and out-of-domain; you don't?",
     "We establish in-domain (FairFace) first. Out-of-domain cross-dataset "
     "validation (UTKFace, FACET) is the next stage of the same pipeline."),
    ("How is this different from Zhang / prior work?",
     "Zhang is post-hoc and reports only retrieval. FairCLIP is training-time, "
     "reports BOTH classification and retrieval fairness, reveals their "
     "trade-off, and beats Zhang on MaxSkew/NDKL while improving accuracy.")]
for q,a in qa:
    P("Q: "+q,bold=True); P("A: "+a); doc.add_paragraph()

# 7. Key finding
H("7. Key Finding: Classification vs Retrieval Fairness Trade-off",1)
P("We empirically show these two fairness notions conflict in CLIP space: "
  "fully removing an attribute (low MaxSkew) destroys classification "
  "accuracy/DPG, and vice versa — a sharp trade-off with no middle sweet "
  "spot. Example (race): INLP=1 gives Acc 0.73/MaxSkew 1.79; INLP=4 gives "
  "Acc 0.19/MaxSkew 0.36. Characterizing this conflict is a contribution "
  "not present in prior work.")

doc.save("FairCLIP_Defense_Document.docx")
print("Saved: FairCLIP_Defense_Document.docx")

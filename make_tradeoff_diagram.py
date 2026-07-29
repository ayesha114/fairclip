import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("results/figures", exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
rng = np.random.RandomState(0)
ax = axes[0]
for i,(cx,cy,col) in enumerate([(-1.2,0,"#4C72B0"),(1.2,0,"#DD8452"),(0,1.3,"#55A868")]):
    pts = rng.randn(40,2)*0.25 + [cx,cy]
    ax.scatter(pts[:,0], pts[:,1], c=col, s=25, alpha=0.7, label=f"Group {i+1}")
ax.set_title("BEFORE debiasing\nGroups separable\n= classification works, retrieval biased", fontsize=11, fontweight="bold")
ax.legend(loc="upper right", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(-2.3,2.3); ax.set_ylim(-1.3,2.3)
ax = axes[1]; ax.axis("off")
ax.annotate("", xy=(0.85,0.5), xytext=(0.15,0.5), arrowprops=dict(arrowstyle="-|>", lw=3, color="black"))
ax.text(0.5,0.62,"Project out\ndemographic subspace B\n(Step VIII, iterated)", ha="center", fontsize=11, fontweight="bold")
ax.text(0.5,0.30,"z' = (I - B B^T) z", ha="center", fontsize=13, family="monospace", bbox=dict(boxstyle="round", fc="#FFF3CD", ec="#B8860B"))
ax = axes[2]
for i,col in enumerate(["#4C72B0","#DD8452","#55A868"]):
    pts = rng.randn(40,2)*0.5 + [0,0.4]
    ax.scatter(pts[:,0], pts[:,1], c=col, s=25, alpha=0.6, label=f"Group {i+1}")
ax.set_title("AFTER debiasing\nGroups overlap\n= retrieval fair, classification at chance", fontsize=11, fontweight="bold")
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-2.3,2.3); ax.set_ylim(-1.3,2.3)
fig.suptitle("The Classification-Retrieval Trade-off", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/figures/tradeoff_diagram.png", dpi=200, bbox_inches="tight")
print("saved results/figures/tradeoff_diagram.png")

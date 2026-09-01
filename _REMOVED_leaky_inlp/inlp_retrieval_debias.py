"""
=============================================================================
INLP Retrieval Debiasing (Iterative Null-space Projection)
=============================================================================
SEPARATE retrieval-fairness module (Ravfogel et al. 2020, "Null It Out").

WHY THIS EXISTS (read for panel):
  FairCLIP's main training optimizes CLASSIFICATION parity (DPG/EOD).
  Retrieval skew (MaxSkew/NDKL) is a DIFFERENT fairness notion that
  conflicts with classification (proven empirically — see notes).
  So we provide a SEPARATE retrieval-debiasing pass used ONLY when
  measuring retrieval fairness, leaving classification results untouched.

HOW IT WORKS:
  Repeatedly: (1) train a linear classifier to predict the protected
  attribute from the embedding, (2) project the embedding onto the null
  space of that classifier. After enough rounds the attribute becomes
  linearly unpredictable -> retrieval no longer favors any group ->
  MaxSkew/NDKL drop sharply.

  The SAME learned projection P_total is applied to image AND text
  embeddings so the space stays consistent.
=============================================================================
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


def fit_inlp_projection(image_embeds: torch.Tensor, group_labels: torch.Tensor,
                        max_iters: int = 15, min_acc: float = 0.14,
                        seed: int = 42):
    """
    Learn an INLP projection matrix P_total [D, D] that removes the
    protected-attribute direction from the embedding space.

    Stops early when the attribute classifier accuracy falls to ~chance
    (min_acc) — i.e. the attribute is no longer recoverable.

    Returns P_total as a torch tensor [D, D].
    """
    X = image_embeds.detach().cpu().float().numpy().copy()
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    y = group_labels.detach().cpu().numpy()
    D = X.shape[1]
    P_total = np.eye(D, dtype=np.float64)

    def null_proj(W):
        u, s, vt = np.linalg.svd(W, full_matrices=False)
        rowspace = vt[s > 1e-6]
        return np.eye(D) - rowspace.T @ rowspace

    for it in range(max_iters):
        clf = LogisticRegression(max_iter=500, C=1.0, random_state=seed)
        clf.fit(X, y)
        acc = clf.score(X, y)
        P = null_proj(clf.coef_)
        X = X @ P.T
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        P_total = P_total @ P.T
        if acc <= min_acc:
            break

    return torch.tensor(P_total, dtype=torch.float32)


def apply_inlp(embeds: torch.Tensor, P_total: torch.Tensor) -> torch.Tensor:
    """Apply a fitted INLP projection and L2-renormalize."""
    E = embeds.detach().cpu().float() @ P_total
    E = E / (E.norm(dim=-1, keepdim=True) + 1e-8)
    return E

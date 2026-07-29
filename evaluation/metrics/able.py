"""
ABLE Metric — Zhang et al. CVPR 2025 formula.
ABLE = 2 / (1/Acc + 1/exp(-MaxSkew))  on a 0-100 scale.
Higher ABLE = better (high accuracy AND low skew).

Both accuracy and maxskew are passed IN (already computed correctly elsewhere)
so ABLE never recomputes a blown-up skew.
"""
import numpy as np


def compute_able(image_embeddings=None, text_embeddings=None, group_labels=None,
                 k: int = 5, accuracy: float = None, maxskew: float = None) -> dict:
    """
    Zhang-style ABLE.

    accuracy: zero-shot accuracy (0-1), passed in.
    maxskew:  text-query MaxSkew (already computed, bounded), passed in.
    """
    acc = max(min(accuracy if accuracy is not None else 0.0, 1.0), 1e-6)
    ms = maxskew if maxskew is not None else 0.0

    skew_term = max(np.exp(-ms), 1e-6)          # (0,1], 1 = perfectly fair
    able = 2.0 / (1.0 / acc + 1.0 / skew_term)  # harmonic mean of acc & fairness
    able = able * 100.0                          # 0-100 scale like Zhang

    return {
        "able": round(float(able), 2),
        "accuracy_term": round(float(acc), 4),
        "maxskew": round(float(ms), 4),
        "skew_term": round(float(skew_term), 4),
    }

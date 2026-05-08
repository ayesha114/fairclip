"""
=============================================================================
Image-Text Retrieval Evaluation
=============================================================================
Measures V-L alignment preservation after debiasing.
Answers RQ4: "Can the model still perform well after bias is reduced?"

Computes:
    TR@K  — Text-to-Image Recall@K
    IR@K  — Image-to-Text Recall@K

For K = 1, 5, 10 (standard retrieval evaluation protocol)

Used on COCO Captions and Flickr30k validation sets.
=============================================================================
"""

import logging
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def compute_recall_at_k(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    k_values: list = [1, 5, 10],
) -> dict:
    """
    Compute TR@K and IR@K retrieval metrics.

    Assumes image_embeddings[i] pairs with text_embeddings[i].

    Args:
        image_embeddings: [N, D] L2-normalized
        text_embeddings: [N, D] L2-normalized
        k_values: list of K values to evaluate

    Returns:
        dict with TR@1, TR@5, TR@10, IR@1, IR@5, IR@10
    """
    n = len(image_embeddings)

    # Full similarity matrix [N, N]
    sim = image_embeddings.float() @ text_embeddings.float().T

    results = {}

    # Text-to-Image: for each text query, find the matching image
    for k in k_values:
        correct = 0
        for i in range(n):
            # Similarities of all images to text query i
            sims = sim[:, i]
            topk = sims.topk(min(k, len(sims))).indices
            if i in topk:
                correct += 1
        results[f"TR@{k}"] = round(100.0 * correct / n, 2)

    # Image-to-Text: for each image query, find the matching text
    for k in k_values:
        correct = 0
        for i in range(n):
            # Similarities of all texts to image query i
            sims = sim[i, :]
            topk = sims.topk(min(k, len(sims))).indices
            if i in topk:
                correct += 1
        results[f"IR@{k}"] = round(100.0 * correct / n, 2)

    return results


@torch.no_grad()
def evaluate_retrieval(
    model,
    config_path: str = "configs/datasets/local_paths.yaml",
    dataset: str = "coco",
    max_samples: int = 1000,
    device: str = "cuda",
) -> dict:
    """
    Run full retrieval evaluation on COCO or Flickr30k.

    Args:
        model: FairCLIP model (or baseline CLIP wrapper)
        config_path: path to local_paths.yaml
        dataset: "coco" or "flickr30k"
        max_samples: number of pairs to evaluate (1000 is standard)
        device: cuda or cpu

    Returns:
        dict with TR@1, TR@5, TR@10, IR@1, IR@5, IR@10
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from data.datasets import CaptionDataset, collate_dict

    cfg = OmegaConf.load(config_path)

    if dataset == "coco":
        manifest = cfg.coco.manifest_val
    elif dataset == "flickr30k":
        manifest = cfg.flickr30k.manifest
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    ds = CaptionDataset(manifest, max_samples=max_samples)
    loader = DataLoader(
        ds, batch_size=64, shuffle=False,
        collate_fn=collate_dict, num_workers=4
    )

    all_image_embs = []
    all_text_embs = []

    log.info(f"Extracting embeddings for {dataset} retrieval eval...")
    for batch in loader:
        images = batch["image"].to(device)
        captions = batch["caption"]

        if hasattr(model, "encode_and_debias"):
            # FairCLIP — use debiased embeddings
            img_embs, txt_embs = model.encode_and_debias(images, captions)
        else:
            # Baseline CLIP
            img_embs = model.encode_image(images)
            txt_embs = model.encode_text(captions)
            img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
            txt_embs = txt_embs / txt_embs.norm(dim=-1, keepdim=True)

        all_image_embs.append(img_embs.cpu())
        all_text_embs.append(txt_embs.cpu())

    all_image_embs = torch.cat(all_image_embs, dim=0)
    all_text_embs = torch.cat(all_text_embs, dim=0)

    log.info(f"Computing Recall@K on {len(all_image_embs)} pairs...")
    results = compute_recall_at_k(all_image_embs, all_text_embs)

    log.info("Retrieval results:")
    for k, v in results.items():
        log.info(f"  {k}: {v:.2f}%")

    return results

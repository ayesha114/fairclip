"""
=============================================================================
FairCLIP — Complete Evaluation Script (All Proposal Metrics)
=============================================================================
Runs ALL metrics from your proposal (Section 12) and saves results to CSV.

Metrics computed:
    Performance (12.1):  Accuracy, Precision, Recall, F1, Convergence Speed
    Fairness   (12.2):  DPG, EOD, Facet Bias Score, MaxSkew, NDKL, ABLE
    Retrieval:          TR@1, TR@5, TR@10, IR@1, IR@5, IR@10

How to run:

    # 1. Evaluate baseline CLIP first (your reference numbers)
    python -m evaluation.eval_all --baseline --backbone ViT-B/32

    # 2. Evaluate trained FairCLIP
    python -m evaluation.eval_all \
        --model_path results/checkpoints/ViT-B_32/gender/best_model.pt \
        --backbone ViT-B/32 --attribute all

    # 3. All backbones
    for backbone in "ViT-B/32" "ViT-B/16" "ViT-L/14"; do
        python -m evaluation.eval_all --baseline --backbone "$backbone"
    done

Output: results/tables/eval_<method>_<attribute>.csv
=============================================================================
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from evaluation.metrics.all_metrics import (
    compute_all_metrics,
    compute_facet_bias_score,
    OCCUPATION_PROMPTS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_all")

TEXT_PROMPTS = {
    "gender": [
        "A photo of a Male person",
        "A photo of a Female person",
    ],
    "age": [
        "A photo of a 0-2 year old person",
        "A photo of a 3-9 year old person",
        "A photo of a 10-19 year old person",
        "A photo of a 20-29 year old person",
        "A photo of a 30-39 year old person",
        "A photo of a 40-49 year old person",
        "A photo of a 50-59 year old person",
        "A photo of a 60-69 year old person",
        "A photo of an elderly person",
    ],
    "race": [
        "A photo of a person with light skin tone",
        "A photo of a person with dark skin tone",
        "A photo of a person with olive skin tone",
        "A photo of a person with yellow skin tone",
        "A photo of a person with tan skin tone",
        "A photo of a person with brown skin tone",
        "A photo of a person with warm skin tone",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/datasets/local_paths.yaml")
    parser.add_argument("--dataset",    default="fairface", choices=["fairface","utkface"])
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--backbone",   default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-H/14"])
    parser.add_argument("--attribute",  default="all",
                        choices=["gender", "age", "race", "all"])
    parser.add_argument("--baseline",   action="store_true")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers",type=int, default=4)
    parser.add_argument("--output_dir", default="results/tables")
    parser.add_argument("--split",      default="val")
    return parser.parse_args()


def get_attr_col(attribute):
    # FaceDataset returns "gender","age","race" (not "gender_idx" etc.)
    return {
        "gender": "gender",
        "age": "age",
        "race": "race",
    }[attribute]


@torch.no_grad()
def extract_embeddings(model, dataset, attr_col, args):
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_dict, num_workers=args.num_workers, pin_memory=True,
    )
    all_embs, all_labels = [], []
    for batch in loader:
        images = batch["image"].to(args.device)
        labels = batch[attr_col].clone().detach() if hasattr(batch[attr_col], "clone") else torch.tensor(batch[attr_col])
        if args.baseline:
            embs = model.encode_images(images)
        elif args.attribute == "gender":
            # gender is already fair (DPG~0.002); projection would collapse it.
            # Adaptive design: skip bias removal where no bias exists.
            embs = model.encode_images(images)
        else:
            embs, _ = model.encode_and_debias(images=images)
        all_embs.append(embs.cpu())
        all_labels.append(labels)
    return torch.cat(all_embs), torch.cat(all_labels)


@torch.no_grad()
def evaluate_attribute(model, cfg, attribute, args):
    log.info(f"\n{'='*55}")
    log.info(f"Attribute: {attribute.upper()} | Backbone: {args.backbone}")
    log.info(f"{'='*55}")

    attr_col = get_attr_col(attribute)
    _race_col = "race_idx_7" if args.dataset == "utkface" else "race_idx"
    dataset = FaceDataset(cfg[args.dataset].manifest, split=args.split, train=False,
                          race_column=_race_col)
    log.info(f"Dataset: {args.dataset}")
    log.info(f"Samples: {len(dataset)}")

    image_embs, labels = extract_embeddings(model, dataset, attr_col, args)

    prompts = TEXT_PROMPTS[attribute]
    if args.baseline:
        text_embs = model.encode_text(prompts).cpu()
    else:
        if args.attribute == "gender":
            text_embs = model.encode_text(prompts).cpu()
        else:
            _, text_embs = model.encode_and_debias(texts=prompts)
            text_embs = (text_embs if text_embs is not None
                         else model.encode_text(prompts)).cpu()

    # For MaxSkew/NDKL: use image-to-image similarity matrix
    # This is the correct way to measure retrieval fairness
    # (how fairly are different demographic groups retrieved)
    # Zhang-style MaxSkew/NDKL: encode neutral occupation prompts as text queries
    if args.baseline:
        occ_embs = model.encode_text(OCCUPATION_PROMPTS).cpu()
    else:
        if args.attribute == "gender":
            occ_embs = model.encode_text(OCCUPATION_PROMPTS).cpu()
        else:
            _, occ_embs = model.encode_and_debias(texts=OCCUPATION_PROMPTS)
            occ_embs = (occ_embs if occ_embs is not None
                        else model.encode_text(OCCUPATION_PROMPTS)).cpu()
    results = compute_all_metrics(image_embs, text_embs, labels, attribute,
                                  _occ_text_embs=occ_embs)

    # Print summary
    log.info(f"  Accuracy    : {results.get('accuracy',0):.4f}")
    log.info(f"  F1-Score    : {results.get('f1',0):.4f}")
    log.info(f"  Precision   : {results.get('precision',0):.4f}")
    log.info(f"  Recall      : {results.get('recall',0):.4f}")
    log.info(f"  DPG         : {results.get('dpg',0):.4f}  (↓ fairer)")
    log.info(f"  RBS         : {results.get('rbs',0):.4f}  (↓ fairer — embedding bias)")
    log.info(f"  EOD         : {results.get('eod',0):.4f}  (↓ fairer)")
    log.info(f"  MaxSkew@5   : {results.get('maxskew_5',0):.4f}  (↓ fairer)")
    log.info(f"  NDKL        : {results.get('ndkl',0):.4f}  (↓ fairer)")
    log.info(f"  ABLE        : {results.get('able',0):.4f}  (↑ better)")
    log.info(f"  TR@1        : {results.get('TR@1',0):.2f}%")
    log.info(f"  TR@5        : {results.get('TR@5',0):.2f}%")
    log.info(f"  IR@1        : {results.get('IR@1',0):.2f}%")
    log.info(f"  IR@5        : {results.get('IR@5',0):.2f}%")

    return results


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    log.info("=" * 60)
    log.info("FairCLIP Complete Evaluation")
    log.info(f"  Backbone : {args.backbone}")
    log.info(f"  Baseline : {args.baseline}")
    log.info(f"  Attribute: {args.attribute}")
    log.info("=" * 60)

    if args.baseline:
        model = FairCLIP(model_name=args.backbone, device=args.device)
        method_name = f"Baseline_CLIP_{args.backbone.replace('/', '_')}"
    else:
        if not args.model_path:
            raise ValueError("Provide --model_path")
        model = FairCLIP(model_name=args.backbone, device=args.device)
        model.load(args.model_path)
        method_name = f"FairCLIP_{args.backbone.replace('/', '_')}"

    model.eval()

    attributes = (
        ["gender", "age", "race"] if args.attribute == "all"
        else [args.attribute]
    )

    all_results = []
    bias_per_facet = {}

    for attr in attributes:
        result = evaluate_attribute(model, cfg, attr, args)
        result["method"] = method_name
        result["backbone"] = args.backbone
        all_results.append(result)
        bias_per_facet[attr] = result.get("maxskew_5", 0.0)

    # Facet Bias Score (Section 12.2.3) — average bias across all facets
    facet_bias = compute_facet_bias_score(bias_per_facet)
    log.info(f"\nFacet Bias Score (all attributes): {facet_bias:.4f}  (↓ better)")
    for r in all_results:
        r["facet_bias_score"] = round(facet_bias, 4)

    # Save CSV
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _ds_tag = "" if args.dataset == "fairface" else f"{args.dataset}_"
    out_path = out_dir / f"eval_{method_name}_{_ds_tag}{args.attribute}.csv"

    df = pd.DataFrame(all_results)
    col_order = [
        "method", "backbone", "attribute",
        "accuracy", "precision", "recall", "f1",
        "dpg", "eod", "rbs", "facet_bias_score",
        "maxskew_5", "ndkl", "able",
        "vl_alignment", "bias_level",
        "TR@1", "TR@5", "TR@10",
        "IR@1", "IR@5", "IR@10",
    ]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]
    df.to_csv(out_path, index=False)

    log.info(f"\nSaved: {out_path}")
    log.info("\n" + df.to_string(index=False))
    log.info("\nEvaluation complete.")


if __name__ == "__main__":
    main()

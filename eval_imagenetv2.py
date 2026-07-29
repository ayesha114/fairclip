"""
ImageNetV2 zero-shot evaluation for FairCLIP utility check.
NOTE: This is ImageNetV2 (10k images), NOT ImageNet-1K (Zhang uses 1K).
Report with explicit disclaimer.
Usage: python eval_imagenetv2.py --model_path <ckpt> --attribute race
"""
import argparse, os, torch
from PIL import Image
import clip as clip_lib
from models.fairclip import FairCLIP

IMAGENETV2_DIR = "datasets/imagenetv2/imagenetv2-matched-frequency-format-val"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--max_per_class", type=int, default=10)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    m = FairCLIP(model_name=args.backbone, device=dev)
    if not args.baseline and args.model_path:
        m.load(args.model_path)
    m.eval()

    # ImageNet class names for zero-shot prompts
    from imagenet_classes import IMAGENET_CLASSES
    prompts = [f"a photo of a {c}" for c in IMAGENET_CLASSES]
    with torch.no_grad():
        txt = m.encode_text(prompts)  # [1000, D]
        txt = txt / txt.norm(dim=-1, keepdim=True)

    _, preprocess = clip_lib.load(args.backbone, device=dev)

    top1 = top5 = total = 0
    class_dirs = sorted(os.listdir(IMAGENETV2_DIR), key=lambda x: int(x))
    with torch.no_grad():
        for cls_idx, d in enumerate(class_dirs):
            dpath = os.path.join(IMAGENETV2_DIR, d)
            imgs = os.listdir(dpath)[:args.max_per_class]
            for im in imgs:
                try:
                    image = preprocess(Image.open(os.path.join(dpath, im)).convert("RGB")).unsqueeze(0).to(dev)
                except Exception:
                    continue
                feat = m.encode_images(image)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                sims = (feat @ txt.T).squeeze(0)
                top5_pred = sims.topk(5).indices.tolist()
                true = int(d)
                if top5_pred[0] == true: top1 += 1
                if true in top5_pred: top5 += 1
                total += 1
            if cls_idx % 200 == 0:
                print(f"  ...{cls_idx}/1000 classes, running Top-1={top1/max(total,1):.3f}")

    print(f"\n=== ImageNetV2 Zero-Shot ({'Baseline' if args.baseline else 'FairCLIP'}) ===")
    print(f"  Top-1: {100*top1/total:.2f}%   Top-5: {100*top5/total:.2f}%   (N={total})")
    print(f"  DISCLAIMER: ImageNetV2 (matched-frequency), NOT ImageNet-1K.")

if __name__ == "__main__":
    main()

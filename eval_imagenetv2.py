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

    Bi = Bt = None
    if not args.baseline and args.model_path:
        _t = FairCLIP(model_name=args.backbone, device=dev)
        _t.load(args.model_path)
        _bi = _t.bias_discoverer.image_bias_directions
        _bt = _t.bias_discoverer.text_bias_directions
        if _bi is not None: Bi = _bi.detach().float().to(dev)
        if _bt is not None: Bt = _bt.detach().float().to(dev)
        del _t; torch.cuda.empty_cache()
    m = FairCLIP(model_name=args.backbone, device=dev)   # FROZEN baseline encoder
    m.eval()

    def _proj(e, B):
        if B is None: return e
        d = e - (e @ B) @ B.T
        return d / d.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # ImageNet class names for zero-shot prompts
    from imagenet_classes import IMAGENET_CLASSES
    prompts = [f"a photo of a {c}" for c in IMAGENET_CLASSES]
    with torch.no_grad():
        _cb = 50 if args.backbone in ("ViT-L/14","ViT-H/14") else 250
        _parts = []
        for _i in range(0, len(prompts), _cb):
            _e = m.encode_text(prompts[_i:_i+_cb])
            _parts.append(_e.float().cpu())
            torch.cuda.empty_cache()
        txt = torch.cat(_parts).to(dev)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        txt = _proj(txt.float(), Bt)
        torch.cuda.empty_cache()

    preprocess = getattr(m.backbone, "preprocess", None)
    if preprocess is None:
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
                feat = _proj(feat.float(), Bi)
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

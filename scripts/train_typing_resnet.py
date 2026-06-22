"""
Fracture-type classifier on a ResNet50 backbone — RadImageNet vs ImageNet
pretrain (clean A/B), to test whether medical pretraining beats the current
ImageNet/COCO yolov8s-cls (typing_dataset_v2 baseline ~0.66 on the 62-img test).

RadImageNet weights: Lab-Rasool/RadImageNet ResNet50.pt (loaded weights_only=True
— safe). Backbone is torchvision resnet50 children[:8]; we add a fresh 4-class head.

ponytail: small-data prototype — finetune-all, ImageNet normalisation for both
arms (RadImageNet's exact train-norm is undocumented in the port; close enough to
rank the two pretrains). Not a production trainer.

Usage
-----
    python scripts/train_typing_resnet.py --pretrain radimagenet --data typing_dataset_v2
    python scripts/train_typing_resnet.py --pretrain imagenet    --data typing_dataset_v2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader


def build(pretrain: str) -> nn.Module:
    r = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2
                           if pretrain == "imagenet" else None)
    backbone = nn.Sequential(*list(r.children())[:8])  # conv1..layer4 (idx 0-7)
    if pretrain == "radimagenet":
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id="Lab-Rasool/RadImageNet", filename="ResNet50.pt")
        sd = torch.load(p, map_location="cpu", weights_only=True)  # safe
        sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
        backbone.load_state_dict(sd, strict=True)
    return nn.Sequential(backbone, nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Dropout(0.3), nn.Linear(2048, 4))


class CaffePreprocess:
    """RadImageNet's actual preprocessing (Keras 'caffe' mode): RGB->BGR,
    subtract ImageNet BGR mean, NO [0,1] scaling (inputs stay in [0,255])."""
    mean = torch.tensor([103.939, 116.779, 123.68]).view(3, 1, 1)  # BGR

    def __call__(self, pil):
        import numpy as np
        t = torch.from_numpy(np.array(pil.convert("RGB"))).float().permute(2, 0, 1)
        return t[[2, 1, 0]] - self.mean  # RGB->BGR, mean-subtract


def loaders(root: Path, pretrain: str):
    # Each pretrain needs the normalisation it was trained with.
    if pretrain == "radimagenet":
        tail = [CaffePreprocess()]
    else:
        tail = [tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])]
    tf_tr = tv.transforms.Compose([
        tv.transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomRotation(15), *tail])
    tf_ev = tv.transforms.Compose([
        tv.transforms.Resize(256), tv.transforms.CenterCrop(224), *tail])
    ds = lambda s, t: tv.datasets.ImageFolder(root / s, t)
    tr = ds("train", tf_tr)
    return (DataLoader(tr, 32, shuffle=True, num_workers=0),
            DataLoader(ds("val", tf_ev), 32, num_workers=0),
            DataLoader(ds("test", tf_ev), 32, num_workers=0),
            tr.classes)


@torch.inference_mode()
def acc(model, dl, dev):
    model.eval(); c = n = 0
    for x, y in dl:
        p = model(x.to(dev)).argmax(1).cpu()
        c += (p == y).sum().item(); n += len(y)
    return c / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain", choices=["radimagenet", "imagenet"], required=True)
    ap.add_argument("--data", type=Path, default="typing_dataset_v2")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build(args.pretrain).to(dev)
    tr, va, te, classes = loaders(args.data, args.pretrain)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()

    best_va = best_te = 0.0
    for ep in range(args.epochs):
        model.train()
        for x, y in tr:
            opt.zero_grad()
            lossf(model(x.to(dev)), y.to(dev)).backward()
            opt.step()
        va_a = acc(model, va, dev)
        if va_a >= best_va:                      # select on val, report its test
            best_va, best_te = va_a, acc(model, te, dev)
        print(f"  ep{ep+1:02d} val={va_a:.3f}", flush=True)
    print(f"PRETRAIN={args.pretrain} | best val={best_va:.3f} | test@bestval={best_te:.3f}")


if __name__ == "__main__":
    main()

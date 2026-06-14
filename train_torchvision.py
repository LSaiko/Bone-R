"""
Fine-tune a torchvision CNN detector (Faster R-CNN / RetinaNet / FCOS) on the
SAME YOLO-format dataset tree used for YOLOv8 — so you can benchmark backbones
head-to-head without reformatting labels.

Reads dataset/images/{train,val}/*.{png,jpg} with sibling YOLO labels under
dataset/labels/{train,val}/*.txt (class cx cy w h normalized).

Usage
-----
    python train_torchvision.py --arch retinanet --data dataset \
        --epochs 20 --batch 4 --out retinanet_fracture.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from models import build_detector
from models.torchvision_backend import _device

IMG_EXTS = {".png", ".jpg", ".jpeg"}


class YoloDetectionDataset(Dataset):
    """YOLO-format dir -> (image_tensor, target dict) for torchvision models."""

    def __init__(self, root: Path, split: str):
        self.img_dir = root / "images" / split
        self.lbl_dir = root / "labels" / split
        self.images = sorted(
            p for p in self.img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img_path = self.images[i]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        tensor = torch.from_numpy(
            np.asarray(img)).permute(2, 0, 1).float().div(255.0)

        boxes, labels = [], []
        lbl_path = self.lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                if not line.strip():
                    continue
                _, cx, cy, bw, bh = map(float, line.split())
                # normalized cx,cy,w,h -> absolute x1,y1,x2,y2
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(1)  # class 1 = fracture (0 = background)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([i]),
        }
        return tensor, target


def collate(batch):
    return tuple(zip(*batch))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="retinanet",
                    choices=["fasterrcnn", "retinanet", "fcos"])
    ap.add_argument("--data", type=Path, default="dataset")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = _device()
    # build_detector returns the wrapper; we need the underlying nn.Module.
    detector = build_detector(args.arch, num_classes=1)
    model = detector.model.train()

    train_ds = YoloDetectionDataset(args.data, "train")
    # num_workers=0 avoids Windows page-file DLL-load failures (WinError 1455).
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                        collate_fn=collate, num_workers=0)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(args.epochs):
        running = 0.0
        for imgs, targets in loader:
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)  # train mode returns losses
            loss = sum(loss_dict.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        print(f"epoch {epoch + 1}/{args.epochs}  loss={running / len(loader):.4f}")

    out = args.out or f"{args.arch}_fracture.pt"
    torch.save({"model": model.state_dict()}, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

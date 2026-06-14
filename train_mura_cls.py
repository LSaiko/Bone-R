"""
Train a MURA abnormal/normal classifier and use it to harden the detector.

MURA has no boxes but ~40k labeled X-rays, so it's ideal for a study-level
abnormality classifier. That classifier acts as a *second opinion* on top of
the FracAtlas detector (see ensemble.py): if the detector fires but the
classifier strongly says "normal", we can down-weight the detection, and vice
versa — improving robustness on body parts FracAtlas under-covers.

Expects the tree built by:
    python scripts/mura_to_yolo.py --mode classify --out MURA_cls
producing MURA_cls/{train,val}/{abnormal,normal}/*.png

Usage
-----
    python train_mura_cls.py --data MURA_cls --model yolov8s-cls.pt --epochs 20
"""

import argparse
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="MURA_cls",
                    help="Classification root with train/ and val/ subdirs")
    ap.add_argument("--model", default="yolov8s-cls.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--name", default="mura_abnormality")
    ap.add_argument("--workers", type=int, default=0)  # Windows page-file safe
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        workers=args.workers,
        patience=8,
        cos_lr=True,
    )
    m = model.val()
    # top1 accuracy is the headline metric for binary abnormality.
    print(f"top-1 accuracy: {m.top1:.4f}")
    print(f"weights: runs/classify/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()

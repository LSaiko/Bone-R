"""
Train a fracture-type classifier on typing_dataset.

WHY THIS SCRIPT EXISTS
----------------------
The project's original goal included ">95% fracture-TYPE classification
accuracy" but no typed training data existed until the HUMERUS_rf dataset
was integrated. This script trains a YOLOv8-cls model on the 4
fracture-morphology classes: oblique, segmental, spiral, transverse.

The dataset is humerus-only (single anatomic region), ~1,420 images total,
built by scripts/build_typing_dataset.py from the HUMERUS_rf Roboflow export.

Usage
-----
    python train_typing.py
    python train_typing.py --epochs 40 --batch 32

Once trained, the best weights sit at:
    runs/classify/<name>/weights/best.pt
"""

import argparse
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train YOLOv8-cls fracture-type classifier"
    )
    ap.add_argument("--data", default="typing_dataset",
                    help="Classification root built by scripts/build_typing_dataset.py "
                         "(expected subdirs: train/ val/ test/ each with class folders)")
    ap.add_argument("--model", default="yolov8s-cls.pt",
                    help="Pretrained YOLOv8-cls checkpoint to fine-tune")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--name", default="fracture_type_cls")
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
        patience=10,
        cos_lr=True,
    )
    m = model.val()
    # top1 / top5 are the headline metrics for multi-class type classification.
    print(f"top-1 accuracy: {m.top1:.4f}")
    print(f"top-5 accuracy: {m.top5:.4f}")
    print(f"weights: runs/classify/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()

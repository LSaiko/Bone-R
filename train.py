"""Train a YOLOv8 fracture detector with Ultralytics."""

import argparse
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--data", default="dataset.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="fracture_yolov8s")
    # On Windows with a small page file, spawning many DataLoader workers can
    # fail loading torch DLLs (WinError 1455). 0 = load in the main process.
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        workers=args.workers,
        # Fracture-friendly defaults: keep recall high, mild augmentation.
        patience=20,
        cos_lr=True,
    )
    metrics = model.val()
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()

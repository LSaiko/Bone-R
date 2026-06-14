"""
Benchmark every CNN backend on the same test split and emit a comparison table
+ JSON (for the landing page) of mAP and screening sensitivity.

Each backend must have a fine-tuned checkpoint; pass them as name=weights pairs.

Usage
-----
    python benchmark.py --data dataset.yaml \
        --model yolov8=runs/detect/fracture_yolov8s/weights/best.pt \
        --model retinanet=retinanet_fracture.pt \
        --model fasterrcnn=fasterrcnn_fracture.pt \
        --out docs/data/benchmark.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def eval_yolo(weights: str, data: str) -> dict:
    from ultralytics import YOLO
    m = YOLO(weights).val(data=data, verbose=False)
    return {
        "map50": round(float(m.box.map50), 4),
        "map5095": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
    }


def eval_torchvision(arch: str, weights: str, data_root: Path) -> dict:
    """Lightweight mAP@0.5 for torchvision backends on dataset/labels/test."""
    import torch
    from torchmetrics.detection import MeanAveragePrecision
    from models import build_detector
    from train_torchvision import YoloDetectionDataset

    det = build_detector(arch, weights=weights, num_classes=1)
    ds = YoloDetectionDataset(data_root, "test")
    metric = MeanAveragePrecision(box_format="xyxy")
    import numpy as np
    for i in range(len(ds)):
        tensor, target = ds[i]
        img = (tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
        dets = det.predict(img, conf=0.05)
        preds = [{
            "boxes": torch.tensor([d.bbox_xyxy for d in dets]).reshape(-1, 4),
            "scores": torch.tensor([d.confidence for d in dets]),
            "labels": torch.ones(len(dets), dtype=torch.int64),
        }]
        gts = [{"boxes": target["boxes"],
                "labels": torch.ones(len(target["boxes"]), dtype=torch.int64)}]
        metric.update(preds, gts)
    res = metric.compute()
    return {
        "map50": round(float(res["map_50"]), 4),
        "map5095": round(float(res["map"]), 4),
        "precision": None,
        "recall": round(float(res.get("mar_100", float("nan"))), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.yaml", help="YOLO data yaml")
    ap.add_argument("--data-root", default="dataset", type=Path)
    ap.add_argument("--model", action="append", default=[],
                    help="name=weights (name in yolov8|fasterrcnn|retinanet|fcos)")
    ap.add_argument("--out", default="docs/data/benchmark.json", type=Path)
    args = ap.parse_args()

    results = {}
    for spec in args.model:
        name, weights = spec.split("=", 1)
        print(f"evaluating {name} ({weights}) ...")
        try:
            if name == "yolov8":
                results[name] = eval_yolo(weights, args.data)
            else:
                results[name] = eval_torchvision(name, weights, args.data_root)
        except Exception as e:
            results[name] = {"error": str(e)}
        print(f"  {results[name]}")

    # Console table
    print("\n=== Backend comparison (test split) ===")
    print(f"{'backend':12s} {'mAP@.5':>8s} {'mAP@.5:.95':>11s} "
          f"{'precision':>10s} {'recall':>8s}")
    for name, r in results.items():
        if "error" in r:
            print(f"{name:12s}  ERROR: {r['error']}")
            continue
        p = "-" if r["precision"] is None else f"{r['precision']:.4f}"
        print(f"{name:12s} {r['map50']:8.4f} {r['map5095']:11.4f} "
              f"{p:>10s} {r['recall']:8.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

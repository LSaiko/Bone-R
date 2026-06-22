"""
External validation: run a trained detector over images from a source the model
NEVER trained on, and report image-level screening stats. The honest test of
generalization — in-distribution metrics flatter the model.

Images are found recursively under --images. Ground truth per image:
  * sibling <stem>.txt present & non-empty -> positive; empty -> negative
  * no .txt and --all-positive            -> positive (e.g. an all-fractured set)

Usage
-----
    python scripts/external_val.py \
        --weights runs/detect/fracture_yolov8m_v5/weights/best.pt \
        --images pkdarabi_raw --all-positive --conf 0.25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference import read_image                       # noqa: E402
from models import build_detector                      # noqa: E402
from evaluate import image_level_stats                 # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--all-positive", action="store_true",
                    help="Treat every image as fractured (set has no negatives)")
    args = ap.parse_args()

    det = build_detector("yolov8", weights=args.weights)
    y_true, y_pred = [], []
    for p in args.images.rglob("*"):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        lbl = p.with_suffix(".txt")
        gt = 1 if args.all_positive else int(lbl.exists() and lbl.read_text().strip() != "")
        dets = det.predict(read_image(p.read_bytes(), p.name), conf=args.conf)
        y_true.append(gt)
        y_pred.append(1 if dets else 0)

    s = image_level_stats(y_true, y_pred)
    print(f"external set     : {args.images}  (n={len(y_true)}, pos={sum(y_true)})")
    print(f"sensitivity      : {s['sensitivity']:.4f}")
    print(f"specificity      : {s['specificity']:.4f}")
    print(f"PPV              : {s['PPV']:.4f}")
    print("NOTE: compare to in-distribution sensitivity — a large drop is the "
          "real generalization gap that in-set metrics hide.")


if __name__ == "__main__":
    main()

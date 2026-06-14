"""
Evaluate the detector+classifier ENSEMBLE at image level over a whole split,
and compare it head-to-head with the detector alone.

Answers the question raised in JOURNAL Entry 003: does the MURA abnormality
classifier, fused via ensemble.py, lift image-level sensitivity past the
detector-only ceiling (0.56 sensitivity) without tanking specificity (0.99)?

Reuses the pure metric helpers from evaluate.py so the math is identical.

Usage
-----
    python eval_ensemble.py \
        --detector runs/detect/fracture_yolov8m/weights/best.pt \
        --classifier runs/classify/mura_abnormality/weights/best.pt \
        --data-root dataset_v2 --split test --alpha 0.5 --conf 0.15
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluate import image_level_stats, load_csv_index, region_of
from ensemble import FractureEnsemble

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--data-root", default="dataset_v2", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--csv", default="FracAtlas/dataset.csv", type=Path)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--keep", type=float, default=0.25)
    args = ap.parse_args()

    from inference import read_image  # local import; avoids overhead if unused

    img_dir = args.data_root / "images" / args.split
    lbl_dir = args.data_root / "labels" / args.split
    csv_index = load_csv_index(args.csv) if args.csv.exists() else {}

    ens = FractureEnsemble(args.detector, args.classifier, alpha=args.alpha)

    y_true, y_pred = [], []
    by_region: dict[str, list[tuple[int, int]]] = {}

    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    for i, img_path in enumerate(imgs):
        lbl = lbl_dir / (img_path.stem + ".txt")
        gt = 1 if (lbl.exists() and lbl.read_text().strip()) else 0

        with open(img_path, "rb") as fh:
            rgb = read_image(fh.read(), img_path.name)
        res = ens.predict(rgb, conf=args.conf, keep_threshold=args.keep)
        pred = 1 if res.fracture_detected else 0

        y_true.append(gt)
        y_pred.append(pred)
        region = region_of(img_path.name, csv_index)
        by_region.setdefault(region, []).append((gt, pred))

        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(imgs)}")

    stats = image_level_stats(y_true, y_pred)
    print(f"\n=== ENSEMBLE image-level (split={args.split}, "
          f"alpha={args.alpha}, conf={args.conf}, keep={args.keep}) ===")
    print(f"  images           : {len(y_true)}")
    print(f"  positives (GT)   : {sum(y_true)}")
    print(f"  TP={int(stats['TP'])} FP={int(stats['FP'])} "
          f"TN={int(stats['TN'])} FN={int(stats['FN'])}")
    print(f"  sensitivity      : {stats['sensitivity']:.4f}")
    print(f"  specificity      : {stats['specificity']:.4f}")
    print(f"  PPV              : {stats['PPV']:.4f}")
    print(f"  NPV              : {stats['NPV']:.4f}")
    print(f"  accuracy         : {stats['accuracy']:.4f}")

    print("\n  vs detector-only baseline (Entry 003): "
          "sensitivity 0.5606 | specificity 0.9883 | PPV 0.9024")

    print("\n=== Per-region ensemble sensitivity ===")
    for region, pairs in sorted(by_region.items()):
        n = len(pairs)
        pos = sum(g for g, _ in pairs)
        tp = sum(1 for g, p in pairs if g == 1 and p == 1)
        sens = tp / pos if pos else float("nan")
        print(f"  {region:10s} n={n:4d} pos={pos:3d} "
              f"sensitivity={sens:.4f}" if pos else
              f"  {region:10s} n={n:4d} pos={pos:3d} sensitivity=  N/A")


if __name__ == "__main__":
    main()

"""
Evaluate a trained fracture detector.

Reports the standard detection metrics plus clinically-motivated ones:

1. Standard box metrics (mAP@0.5, mAP@0.5:0.95, precision, recall).
2. Sensitivity @ target recall — the confidence threshold that achieves a
   target recall (default 0.95) on the P/R curve, and the precision paid
   for it.  In fracture screening a MISS is far worse than a false alarm.
3. Image-level sensitivity & specificity — evaluated over every image in the
   chosen split.  An image is "positive" if its YOLO label file is non-empty
   (≥1 ground-truth box).  The model calls it "positive" if it predicts ≥1
   box at confidence ≥ the threshold selected in step 2.  From the resulting
   2×2 confusion table we report sensitivity (recall), specificity, PPV, NPV,
   and accuracy.  These are the screening-relevant metrics a clinician cares
   about; they are complementary to box-level mAP.
4. Per-region breakdown — FracAtlas tags every image with a body region
   (hand / leg / hip / shoulder / mixed).  We look up each test image's region
   from dataset.csv and report per-region image-level sensitivity so we can
   see where the model is weakest.

Usage
-----
    python evaluate.py --weights runs/detect/fracture_yolov8s/weights/best.pt
    python evaluate.py --weights best.pt --data dataset_v2.yaml \\
        --data-root dataset_v2 --split test --target-recall 0.90
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Pure helper functions (no model, no GPU — unit-testable)
# ---------------------------------------------------------------------------

def image_level_stats(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Compute image-level binary classification statistics.

    Parameters
    ----------
    y_true : list of int
        Ground-truth labels, 1 = fracture present, 0 = normal.
    y_pred : list of int
        Model predictions at the chosen operating-point threshold,
        1 = model predicted fracture, 0 = model predicted normal.

    Returns
    -------
    dict with keys: TP, FP, TN, FN, sensitivity, specificity, PPV, NPV, accuracy.
    All float so callers can format uniformly.

    Notes
    -----
    * sensitivity (recall) = TP / (TP + FN)  — did we catch every fracture?
    * specificity          = TN / (TN + FP)  — how often do we correctly clear a normal image?
    * PPV (precision)      = TP / (TP + FP)  — when we flag, how often are we right?
    * NPV                  = TN / (TN + FN)  — when we clear, how confident can we be?

    A denominator of zero yields NaN rather than crashing, so callers must
    handle NaN when a split contains only positives or only negatives.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")

    tp = fp = tn = fn = 0
    for gt, pr in zip(y_true, y_pred):
        if gt == 1 and pr == 1:
            tp += 1
        elif gt == 0 and pr == 1:
            fp += 1
        elif gt == 0 and pr == 0:
            tn += 1
        else:  # gt == 1, pr == 0
            fn += 1

    def _safe(num, den):
        return float(num) / float(den) if den else float("nan")

    return {
        "TP": float(tp),
        "FP": float(fp),
        "TN": float(tn),
        "FN": float(fn),
        "sensitivity": _safe(tp, tp + fn),   # recall
        "specificity": _safe(tn, tn + fp),
        "PPV":         _safe(tp, tp + fp),   # precision
        "NPV":         _safe(tn, tn + fn),
        "accuracy":    _safe(tp + tn, tp + fp + tn + fn),
    }


def load_csv_index(csv_path: Path) -> Dict[str, str]:
    """Load FracAtlas dataset.csv into a dict mapping image_id stem -> region.

    The region is determined by the one-hot columns hand/leg/hip/shoulder/mixed.
    If none is set (or the id is missing) the region is 'unknown'.

    Returns
    -------
    dict  {stem_without_extension: region_name}
    e.g.  {'IMG0001547': 'leg', 'IMG0000002': 'hand', ...}
    """
    region_cols = ["hand", "leg", "hip", "shoulder", "mixed"]
    index: Dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # image_id looks like "IMG0000002.jpg" — drop extension for keying.
            img_stem = Path(row["image_id"]).stem
            region = "unknown"
            for col in region_cols:
                if row.get(col, "0").strip() == "1":
                    region = col
                    break
            index[img_stem] = region
    return index


def region_of(filename: str, csv_index: Dict[str, str]) -> str:
    """Return the body region for *filename* using a pre-loaded csv_index.

    Test images are renamed with a numeric prefix by make_splits.py, e.g.::

        000123_IMG0001547.png  ->  original stem  IMG0001547

    We strip the leading ``NNNNNN_`` prefix (if present) to recover the
    original ``IMG*`` stem, then look it up in *csv_index*.

    Parameters
    ----------
    filename : str
        Basename of the test image (with or without extension).
    csv_index : dict
        Mapping produced by :func:`load_csv_index`.

    Returns
    -------
    str  — region name from the CSV or ``'unknown'`` if not found.
    """
    stem = Path(filename).stem  # drop extension

    # make_splits.py prefixes a zero-padded index: "000123_IMG0001547"
    # Strip the prefix if it matches the pattern <digits>_<rest>.
    parts = stem.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        stem = parts[1]

    return csv_index.get(stem, "unknown")


def source_of(filename: str) -> str:
    """Tag a test image by its origin dataset from the filename alone.

    The merged training set mixes four sources whose images carry distinctive
    name patterns. This lets evaluate report per-source sensitivity (the
    project's core honesty metric: FracAtlas-native is the real-world floor,
    wrist/humerus/hip are the easier added regions) without needing the source
    dirs present at eval time.

    ponytail: pattern-based tag, order matters (HUMERUS names contain 'Img').
    If a new source is added, extend the patterns or switch to a built manifest.
    """
    s = filename.lower()
    if "wri" in s:                                   # GRAZPEDWRI pediatric wrist
        return "wrist"
    if "anonim" in s:                                # HUMERUS shoulder set
        return "humerus"
    if any(k in s for k in ("intertrochanteric", "subtrochanteric", "neck")):
        return "hip"                                 # proximal-femur
    if "img" in s:                                   # FracAtlas IMG####
        return "fracatlas"
    return "other"


# ---------------------------------------------------------------------------
# Image-level prediction gathering (requires model — guarded separately)
# ---------------------------------------------------------------------------

def _gather_image_predictions(
    model,
    img_dir: Path,
    lbl_dir: Path,
    conf_thresh: float,
) -> tuple[List[str], List[int], List[int]]:
    """Run the model over every image in *img_dir* and return parallel lists.

    This function calls ``model.predict`` on each image individually so we can
    pair predictions with ground-truth label files.  It is intentionally kept
    separate from the pure stats functions so those remain unit-testable.

    Returns
    -------
    filenames : list[str]  — basename of each image
    y_true    : list[int]  — 1 if label file has ≥1 box, else 0
    y_pred    : list[int]  — 1 if model predicted ≥1 box at conf ≥ thresh, else 0
    """
    exts = {".png", ".jpg", ".jpeg"}
    img_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)

    filenames: List[str] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for img_path in img_paths:
        # Ground truth: label file non-empty means fracture present.
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        gt_positive = 0
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            # Make sure file has actual annotation lines (not just whitespace).
            content = lbl_path.read_text().strip()
            gt_positive = 1 if content else 0

        # Model prediction at the operating-point threshold.
        results = model.predict(str(img_path), conf=conf_thresh, verbose=False)
        pred_positive = 0
        for r in results:
            if len(r.boxes) > 0:
                pred_positive = 1
                break

        filenames.append(img_path.name)
        y_true.append(gt_positive)
        y_pred.append(pred_positive)

    return filenames, y_true, y_pred


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate a YOLOv8 fracture detector with clinical metrics."
    )
    ap.add_argument("--weights", required=True,
                    help="Path to trained weights (.pt)")
    ap.add_argument("--data", default="dataset_v2.yaml",
                    help="YOLO dataset YAML (default: dataset_v2.yaml)")
    ap.add_argument("--data-root", default="dataset_v2",
                    help="Dataset root directory containing images/ and labels/ "
                         "(default: dataset_v2)")
    ap.add_argument("--split", default="test",
                    help="Which split to evaluate image-level metrics on "
                         "(default: test)")
    ap.add_argument("--target-recall", type=float, default=0.95,
                    help="Target recall for operating-point selection "
                         "(default: 0.95)")
    ap.add_argument("--csv", default="FracAtlas/dataset.csv",
                    help="Path to FracAtlas dataset.csv for region lookup "
                         "(default: FracAtlas/dataset.csv)")
    args = ap.parse_args()

    from ultralytics import YOLO  # import deferred so test file never needs GPU

    model = YOLO(args.weights)

    # ------------------------------------------------------------------
    # 1. Standard detection metrics via YOLO val()
    # ------------------------------------------------------------------
    m = model.val(data=args.data)

    print("=== Standard detection metrics ===")
    print(f"mAP@0.5      : {m.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {m.box.map:.4f}")
    print(f"precision    : {m.box.mp:.4f}")
    print(f"recall       : {m.box.mr:.4f}")

    # ------------------------------------------------------------------
    # 2. Sensitivity @ target recall — operating-point selection
    # ------------------------------------------------------------------
    print("\n=== Sensitivity @ high recall (fracture screening) ===")
    selected_conf: float = 0.25  # fallback if curve unavailable
    try:
        r_curve = np.asarray(m.box.r_curve).reshape(-1)
        p_curve = np.asarray(m.box.p_curve).reshape(-1)
        conf = np.linspace(0, 1, len(r_curve))
        # Highest confidence threshold that still meets the target recall.
        # We want the most selective threshold that doesn't sacrifice recall —
        # i.e. the rightmost (highest-conf) point still at or above target.
        ok = np.where(r_curve >= args.target_recall)[0]
        if len(ok):
            i = ok[-1]
            selected_conf = float(conf[i])
            print(f"target recall {args.target_recall:.2f} reachable")
            print(f"  conf threshold : {selected_conf:.3f}")
            print(f"  recall         : {r_curve[i]:.4f}")
            print(f"  precision      : {p_curve[i]:.4f}")
            print(f"  -> at this operating point you miss "
                  f"{(1 - r_curve[i]) * 100:.1f}% of fractures")
        else:
            print(f"Target recall {args.target_recall:.2f} NOT achievable; "
                  f"max recall = {r_curve.max():.4f}")
            print(f"Using fallback conf threshold {selected_conf:.3f} for "
                  f"image-level metrics.")
    except Exception as e:  # curve attrs vary across ultralytics versions
        print(f"Could not compute recall-curve operating point: {e}")
        print(f"Using fallback conf threshold {selected_conf:.3f}.")

    # ------------------------------------------------------------------
    # 3. Image-level sensitivity / specificity over the chosen split
    # ------------------------------------------------------------------
    data_root = Path(args.data_root)
    img_dir = data_root / "images" / args.split
    lbl_dir = data_root / "labels" / args.split

    if not img_dir.exists():
        print(f"\nWARNING: image directory {img_dir} not found; "
              f"skipping image-level metrics.")
        return

    print(f"\n=== Image-level metrics (split={args.split}, "
          f"conf≥{selected_conf:.3f}) ===")
    print("Running per-image inference — this may take a few minutes …")

    filenames, y_true, y_pred = _gather_image_predictions(
        model, img_dir, lbl_dir, selected_conf
    )

    stats = image_level_stats(y_true, y_pred)
    print(f"  images evaluated : {len(y_true)}")
    print(f"  positives (GT)   : {int(sum(y_true))}")
    print(f"  negatives (GT)   : {int(len(y_true) - sum(y_true))}")
    print(f"  TP={int(stats['TP'])}  FP={int(stats['FP'])}  "
          f"TN={int(stats['TN'])}  FN={int(stats['FN'])}")
    print(f"  sensitivity      : {stats['sensitivity']:.4f}  "
          f"(= recall at image level)")
    print(f"  specificity      : {stats['specificity']:.4f}")
    print(f"  PPV (precision)  : {stats['PPV']:.4f}")
    print(f"  NPV              : {stats['NPV']:.4f}")
    print(f"  accuracy         : {stats['accuracy']:.4f}")

    # ------------------------------------------------------------------
    # 4. Per-region breakdown
    # ------------------------------------------------------------------
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"\nWARNING: {csv_path} not found; skipping per-region breakdown.")
        return

    print(f"\n=== Per-region image-level sensitivity ===")
    csv_index = load_csv_index(csv_path)

    # Group indices by region.
    from collections import defaultdict
    region_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, fname in enumerate(filenames):
        reg = region_of(fname, csv_index)
        region_indices[reg].append(idx)

    for region in sorted(region_indices.keys()):
        idxs = region_indices[region]
        rt = [y_true[i] for i in idxs]
        rp = [y_pred[i] for i in idxs]
        rs = image_level_stats(rt, rp)
        n_pos = int(sum(rt))
        n_total = len(rt)
        sens = rs["sensitivity"]
        sens_str = f"{sens:.4f}" if sens == sens else "  N/A "  # NaN guard
        print(f"  {region:<10s}  n={n_total:4d}  pos={n_pos:4d}  "
              f"sensitivity={sens_str}  "
              f"TP={int(rs['TP'])}  FN={int(rs['FN'])}")

    # ------------------------------------------------------------------
    # 5. Per-source breakdown (which dataset each region of skill came from)
    # ------------------------------------------------------------------
    print(f"\n=== Per-source image-level sensitivity ===")
    source_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, fname in enumerate(filenames):
        source_indices[source_of(fname)].append(idx)
    for src in sorted(source_indices.keys()):
        idxs = source_indices[src]
        ss = image_level_stats([y_true[i] for i in idxs],
                                [y_pred[i] for i in idxs])
        sens = ss["sensitivity"]
        sens_str = f"{sens:.4f}" if sens == sens else "  N/A "
        print(f"  {src:<10s}  n={len(idxs):4d}  pos={int(sum(y_true[i] for i in idxs)):4d}  "
              f"sensitivity={sens_str}")


if __name__ == "__main__":
    main()

"""
Build a YOLO detection dataset tree from FracAtlas (and optionally MURA weak
labels) by splitting images + their .txt labels into train/val/test.

Produces:
    dataset/
      images/{train,val,test}/*.jpg
      labels/{train,val,test}/*.txt   (mirrors image stems)

Run fracatlas_to_yolo.py (and optionally mura_to_yolo.py --mode weakbox) first
so that every image has a sibling .txt.

Usage
-----
    python scripts/make_splits.py --src FracAtlas/images --out dataset \
        --val 0.15 --test 0.10
    # add MURA weak-labeled positives too:
    python scripts/make_splits.py --src FracAtlas/images --src MURA --out dataset

    # region-stratified split (rare hip/shoulder appear in every split):
    python scripts/make_splits.py --src FracAtlas_proc --src GRAZ_proc \
        --out dataset_v5 --copy --stratify
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg"}
_REGION_COLS = ("hand", "leg", "hip", "shoulder", "mixed")


def collect(srcs: list[Path]) -> list[Path]:
    imgs: list[Path] = []
    for src in srcs:
        for p in src.rglob("*"):
            if p.suffix.lower() in IMG_EXTS and p.with_suffix(".txt").exists():
                imgs.append(p)
    return imgs


def load_region_index(csv_path: Path) -> dict[str, str]:
    """FracAtlas dataset.csv -> {IMG stem: body region}. Empty if file absent."""
    index: dict[str, str] = {}
    if not csv_path or not csv_path.exists():
        return index
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            stem = Path(row["image_id"]).stem
            region = "unknown"
            for col in _REGION_COLS:
                if row.get(col, "0").strip() == "1":
                    region = col
                    break
            index[stem] = region
    return index


def stratum_key(img: Path, region_index: dict[str, str]) -> str:
    """Group key for stratified splitting: '<region/source>_<pos|neg>'.

    Why: rare regions (hip/shoulder) have so few fractured images that a plain
    random split can land ZERO of them in the test set, making per-region
    metrics unevaluable (exactly what happened in v1-v3). Splitting WITHIN each
    region+fracture stratum guarantees each is represented across train/val/test.
    """
    stem = img.stem
    has_fx = img.with_suffix(".txt").read_text().strip() != ""
    tag = "pos" if has_fx else "neg"
    if stem in region_index:                      # FracAtlas image
        return f"{region_index[stem]}_{tag}"
    if "WRI" in stem.upper():                      # GRAZPEDWRI wrist
        return f"wrist_{tag}"
    return f"other_{tag}"                           # HUMERUS / misc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", type=Path, required=True,
                    help="One or more source roots (repeatable)")
    ap.add_argument("--out", type=Path, default="dataset")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true",
                    help="Copy files instead of symlinking")
    ap.add_argument("--stratify", action="store_true",
                    help="Split within region+fracture strata so rare regions "
                         "(hip/shoulder) are represented in every split")
    ap.add_argument("--stratify-csv", type=Path, default=Path("FracAtlas/dataset.csv"),
                    help="FracAtlas dataset.csv used to look up body region")
    args = ap.parse_args()

    imgs = collect(args.src)
    rng = random.Random(args.seed)

    if args.stratify:
        region_index = load_region_index(args.stratify_csv)
        # Bucket images by stratum, then split each bucket by the same ratios so
        # every stratum (incl. rare hip/shoulder positives) appears in all splits.
        buckets: dict[str, list[Path]] = defaultdict(list)
        for img in imgs:
            buckets[stratum_key(img, region_index)].append(img)
        splits = {"val": [], "test": [], "train": []}
        for key in sorted(buckets):
            group = buckets[key]
            rng.shuffle(group)
            g = len(group)
            n_val, n_test = int(g * args.val), int(g * args.test)
            splits["val"] += group[:n_val]
            splits["test"] += group[n_val:n_val + n_test]
            splits["train"] += group[n_val + n_test:]
        print(f"stratified across {len(buckets)} region/fracture strata")
    else:
        rng.shuffle(imgs)
        n = len(imgs)
        n_val, n_test = int(n * args.val), int(n * args.test)
        splits = {
            "val": imgs[:n_val],
            "test": imgs[n_val:n_val + n_test],
            "train": imgs[n_val + n_test:],
        }

    for split, files in splits.items():
        img_dir = args.out / "images" / split
        lbl_dir = args.out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(files):
            # Prefix index to avoid cross-dataset name collisions.
            stem = f"{i:06d}_{img.stem}"
            dst_img = img_dir / f"{stem}{img.suffix}"
            dst_lbl = lbl_dir / f"{stem}.txt"
            _place(img, dst_img, args.copy)
            _place(img.with_suffix(".txt"), dst_lbl, args.copy)
        print(f"{split:5s}: {len(files)} images")
    print(f"Done. Dataset at: {args.out}")


def _place(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists():
        return
    try:
        if copy:
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


if __name__ == "__main__":
    main()

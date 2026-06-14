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
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def collect(srcs: list[Path]) -> list[Path]:
    imgs: list[Path] = []
    for src in srcs:
        for p in src.rglob("*"):
            if p.suffix.lower() in IMG_EXTS and p.with_suffix(".txt").exists():
                imgs.append(p)
    return imgs


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
    args = ap.parse_args()

    imgs = collect(args.src)
    random.Random(args.seed).shuffle(imgs)
    n = len(imgs)
    n_val = int(n * args.val)
    n_test = int(n * args.test)
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

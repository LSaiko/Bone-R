"""
MURA  ->  YOLO-compatible labels.

MURA ships only *image-level* (study-level) labels: each study folder is named
`studyN_positive` (abnormal) or `studyN_negative` (normal). There are NO
bounding boxes, so MURA cannot be turned into true box-level detection labels
the way FracAtlas can. This script handles that reality in two ways:

  mode = "classify"  (default, recommended)
      Builds a YOLO *classification* dataset manifest: writes a CSV and an
      optional symlinked/`copied` train/val folder tree of the form
      <out>/<split>/<class>/<image>.png  where class is `abnormal` / `normal`.
      Use this to pre-train / co-train a classifier head for robustness.

  mode = "weakbox"
      Writes a YOLO *detection* .txt next to every POSITIVE image containing a
      single full-frame box (class 0). Negative images get an empty .txt.
      This is a WEAK label (the whole image, not the fracture) — only use it to
      augment FracAtlas detection training when you explicitly want extra
      positive/negative signal, and expect noisier boxes.

Label is parsed from the path component `..._positive` / `..._negative`, which
is reliable across the local MURA layout regardless of the CSV root prefix.

Usage
-----
    # classification manifest (no boxes)
    python scripts/mura_to_yolo.py --root MURA --mode classify --out MURA_cls

    # weak full-frame detection labels written alongside images
    python scripts/mura_to_yolo.py --root MURA --mode weakbox
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

FRACTURE_CLASS_ID = 0
IMG_EXTS = {".png", ".jpg", ".jpeg"}


def label_from_path(path: Path) -> str | None:
    """Return 'abnormal' / 'normal' from a `..._positive` / `..._negative` dir."""
    for part in path.parts:
        p = part.lower()
        if p.endswith("_positive"):
            return "abnormal"
        if p.endswith("_negative"):
            return "normal"
    return None


def split_from_path(path: Path) -> str:
    """Map MURA's own train/valid folders to YOLO train/val."""
    parts = {p.lower() for p in path.parts}
    if "valid" in parts or "val" in parts:
        return "val"
    return "train"


def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            yield p


def run_classify(root: Path, out: Path, link: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "mura_manifest.csv"
    counts: dict[tuple[str, str], int] = {}
    rows = 0
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filepath", "split", "label"])
        for img in iter_images(root):
            label = label_from_path(img)
            if label is None:
                continue
            split = split_from_path(img)
            writer.writerow([str(img), split, label])
            counts[(split, label)] = counts.get((split, label), 0) + 1
            rows += 1

            dest_dir = out / split / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            # Build a globally-unique name. MURA study folders (study1_positive)
            # and image names (image1.png) repeat across EVERY patient/region, so
            # naming by study+image alone collides massively (whole dataset
            # collapses to a handful of files). Include region + patient + study.
            parts = img.parts
            uniq = "__".join(parts[-4:])  # XR_ELBOW/patientXXXXX/studyN_x/imageM.png
            dest = dest_dir / uniq
            if dest.exists():
                continue
            try:
                if link:
                    dest.symlink_to(img.resolve())
                else:
                    shutil.copy2(img, dest)
            except OSError:
                # symlink may require privileges on Windows; fall back to copy.
                shutil.copy2(img, dest)

    print(f"MURA -> classification manifest complete ({rows} images).")
    for (split, label), n in sorted(counts.items()):
        print(f"  {split:5s} / {label:8s}: {n}")
    print(f"  manifest: {manifest}")
    print(f"  image tree: {out}/<split>/<class>/")


def run_weakbox(root: Path) -> None:
    # Full-frame box, centered, covering the whole image.
    full_box = f"{FRACTURE_CLASS_ID} 0.500000 0.500000 1.000000 1.000000"
    pos, neg = 0, 0
    for img in iter_images(root):
        label = label_from_path(img)
        if label is None:
            continue
        txt = img.with_suffix(".txt")
        if label == "abnormal":
            txt.write_text(full_box, encoding="utf-8")
            pos += 1
        else:
            txt.write_text("", encoding="utf-8")
            neg += 1
    print(f"MURA -> weak detection labels complete.")
    print(f"  positive (full-frame box): {pos}")
    print(f"  negative (empty .txt)    : {neg}")
    print("  NOTE: boxes are whole-image weak labels, not localized fractures.")


def main() -> None:
    ap = argparse.ArgumentParser(description="MURA -> YOLO-compatible labels")
    ap.add_argument("--root", default="MURA", type=Path)
    ap.add_argument("--mode", choices=["classify", "weakbox"], default="classify")
    ap.add_argument("--out", default="MURA_cls", type=Path,
                    help="Output dir for classification manifest/tree")
    ap.add_argument("--link", action="store_true",
                    help="Symlink instead of copy images into the class tree")
    args = ap.parse_args()

    if args.mode == "classify":
        run_classify(args.root, args.out, args.link)
    else:
        run_weakbox(args.root)


if __name__ == "__main__":
    main()

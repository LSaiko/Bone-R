"""
FracAtlas COCO JSON  ->  YOLO label files.

Reads the FracAtlas COCO annotation file, converts every bounding box to
normalized YOLO format (class cx cy w h, all in 0-1), and writes one .txt
label file per image *alongside* the corresponding image.

COCO bbox format is [x_min, y_min, width, height] in absolute pixels.
YOLO format is `class_id cx cy w h` normalized to [0, 1].

Fractured images that have annotations get a populated .txt.
Non-fractured images (and any fractured image without a box) get an EMPTY
.txt, which YOLO treats as a valid "background / no object" sample.

Usage
-----
    python scripts/fracatlas_to_yolo.py \
        --root FracAtlas \
        --coco "FracAtlas/Annotations/COCO JSON/COCO_fracture_masks.json"

All annotations use a single class -> id 0 ("fracture").
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# FracAtlas is single-class. We remap every COCO category to YOLO class 0.
FRACTURE_CLASS_ID = 0


def to_yolo_xywh(bbox, img_w: int, img_h: int):
    """COCO [x_min, y_min, w, h] (pixels) -> YOLO (cx, cy, w, h) normalized."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp to guard against annotation noise spilling past the image edge.
    return (
        min(max(cx, 0.0), 1.0),
        min(max(cy, 0.0), 1.0),
        min(max(nw, 0.0), 1.0),
        min(max(nh, 0.0), 1.0),
    )


def find_image(root: Path, file_name: str) -> Path | None:
    """Locate an image by name under root/images/{Fractured,Non_fractured}."""
    for sub in ("images/Fractured", "images/Non_fractured", "images"):
        candidate = root / sub / file_name
        if candidate.exists():
            return candidate
    # Fall back to a recursive search (slower, only if layout differs).
    matches = list(root.glob(f"**/{file_name}"))
    return matches[0] if matches else None


def main() -> None:
    ap = argparse.ArgumentParser(description="FracAtlas COCO -> YOLO labels")
    ap.add_argument("--root", default="FracAtlas", type=Path,
                    help="FracAtlas dataset root")
    ap.add_argument("--coco", type=Path,
                    default="FracAtlas/Annotations/COCO JSON/COCO_fracture_masks.json")
    ap.add_argument("--include-negatives", action="store_true", default=True,
                    help="Also write empty .txt files for Non_fractured images")
    args = ap.parse_args()

    coco = json.loads(args.coco.read_text(encoding="utf-8"))

    # image_id -> image metadata
    images = {img["id"]: img for img in coco["images"]}

    # image_id -> list of YOLO lines
    labels: dict[int, list[str]] = defaultdict(list)
    skipped = 0
    for ann in coco["annotations"]:
        img = images.get(ann["image_id"])
        if img is None:
            skipped += 1
            continue
        cx, cy, w, h = to_yolo_xywh(ann["bbox"], img["width"], img["height"])
        if w <= 0 or h <= 0:
            skipped += 1
            continue
        labels[ann["image_id"]].append(
            f"{FRACTURE_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        )

    written, missing = 0, 0
    # 1) Write labels for every annotated image.
    for img_id, img in images.items():
        img_path = find_image(args.root, img["file_name"])
        if img_path is None:
            missing += 1
            continue
        label_path = img_path.with_suffix(".txt")
        label_path.write_text("\n".join(labels.get(img_id, [])), encoding="utf-8")
        written += 1

    # 2) Optionally write empty labels for non-fractured images not in COCO.
    neg_written = 0
    if args.include_negatives:
        coco_names = {img["file_name"] for img in coco["images"]}
        neg_dir = args.root / "images" / "Non_fractured"
        if neg_dir.exists():
            for img_path in neg_dir.glob("*.[jp][pn]g"):
                if img_path.name in coco_names:
                    continue
                label_path = img_path.with_suffix(".txt")
                if not label_path.exists():
                    label_path.write_text("", encoding="utf-8")
                    neg_written += 1

    print(f"FracAtlas -> YOLO conversion complete.")
    print(f"  annotated images written : {written}")
    print(f"  negative empty labels    : {neg_written}")
    print(f"  images not found on disk : {missing}")
    print(f"  annotations skipped      : {skipped}")


if __name__ == "__main__":
    main()

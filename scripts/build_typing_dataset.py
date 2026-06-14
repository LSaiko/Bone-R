"""
Build a YOLO-classification-style dataset for fracture-type classification.

WHY THIS SCRIPT EXISTS
----------------------
The HUMERUS_rf dataset (Roboflow YOLOv8 detection export) contains 4
fracture-morphology classes: oblique, segmental, spiral, transverse. Each
detection label has `<class> cx cy w h` rows. We collapse each image to a
single type label — the class of its first/highest-area box — then copy
images into a YOLO-cls tree:

    typing_dataset/<split>/<typename>/<unique_image_name>

The unique-naming lesson from mura_to_yolo.py: Roboflow augmented filenames
already embed an rf.<hash> suffix, which is unique enough; we keep the full
filename to avoid any collision across splits.

OUTPUT STRUCTURE (mirrors what train_typing.py expects)
-------------------------------------------------------
typing_dataset/
    train/<classname>/<img>
    val/<classname>/<img>
    test/<classname>/<img>

Usage
-----
    python scripts/build_typing_dataset.py
    python scripts/build_typing_dataset.py --src HUMERUS_rf --out typing_dataset
"""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


def box_area(row: str) -> float:
    """Return w*h of a YOLO box row (class cx cy w h). Used to pick dominant box."""
    parts = row.strip().split()
    return float(parts[3]) * float(parts[4])


def dominant_class(label_path: Path) -> int | None:
    """
    Return the class index of the highest-area box in a YOLO label file.

    WHY highest-area rather than first-row: first row is insertion order,
    not importance order. Largest box most reliably represents the primary
    fracture pattern visible in the crop.
    """
    lines = [l for l in label_path.read_text().splitlines() if l.strip()]
    if not lines:
        return None
    return int(max(lines, key=box_area).split()[0])


def build(src: Path, out: Path, class_names: list[str]) -> None:
    split_map = {"train": "train", "valid": "val", "test": "test"}
    counts: dict[str, dict[str, int]] = {}

    for src_split, dst_split in split_map.items():
        img_dir = src / src_split / "images"
        lbl_dir = src / src_split / "labels"
        if not img_dir.exists():
            print(f"[SKIP] {img_dir} not found")
            continue

        split_counts: dict[str, int] = defaultdict(int)
        skipped = 0

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                skipped += 1
                continue

            cls_idx = dominant_class(lbl_path)
            if cls_idx is None or cls_idx >= len(class_names):
                skipped += 1
                continue

            cls_name = class_names[cls_idx]
            dst_dir = out / dst_split / cls_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dst_dir / img_path.name)
            split_counts[cls_name] += 1

        counts[dst_split] = dict(split_counts)
        if skipped:
            print(f"[{dst_split}] skipped {skipped} images (no label / empty)")

    # Report distribution
    print("\n=== Fracture-type distribution ===")
    all_totals: dict[str, int] = defaultdict(int)
    for split, cls_dict in counts.items():
        total = sum(cls_dict.values())
        print(f"\n{split} ({total} images):")
        for cls in class_names:
            n = cls_dict.get(cls, 0)
            all_totals[cls] += n
            flag = "  *** FEW ***" if n < 20 else ""
            print(f"  {cls:12s}: {n:4d}{flag}")

    print("\nGrand total per class:")
    grand = sum(all_totals.values())
    for cls in class_names:
        n = all_totals[cls]
        pct = 100 * n / grand if grand else 0
        flag = "  *** FEW (<50) ***" if n < 50 else ""
        print(f"  {cls:12s}: {n:4d}  ({pct:.1f}%){flag}")

    print(f"\nDataset written to: {out.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build typing_dataset from HUMERUS_rf for fracture-type classification"
    )
    ap.add_argument("--src", default="HUMERUS_rf",
                    help="Roboflow YOLOv8 export root (contains train/valid/test)")
    ap.add_argument("--out", default="typing_dataset",
                    help="Output classification tree root")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    yaml_path = src / "data.yaml"
    with yaml_path.open() as f:
        cfg = yaml.safe_load(f)
    class_names: list[str] = cfg["names"]
    print(f"Classes from data.yaml: {class_names}")

    build(src, out, class_names)


if __name__ == "__main__":
    main()

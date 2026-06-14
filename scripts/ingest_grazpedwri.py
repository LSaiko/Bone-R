"""
Track D prep — ingest GRAZPEDWRI-DX into the Bone-R single-class pipeline.

GRAZPEDWRI-DX (20,327 pediatric wrist X-rays, 18,090 fracture boxes) ships in
YOLOv5 format but is MULTI-CLASS (fracture, boneanomaly, bonelesion, foreignbody,
metal, periostealreaction, pronatorsign, softtissue, text). Bone-R is a single
`fracture` detector (class 0), so this adapter:

  1. Reads each GRAZPEDWRI YOLO label, KEEPS only the fracture-class rows, and
     REMAPS that class id to 0 (Bone-R's `fracture`). All other classes are
     dropped — they're not fractures and would pollute single-class training.
  2. Images with no fracture row become an EMPTY .txt (valid background/negative,
     consistent with how fracatlas_to_yolo.py treats non-fractured images).
  3. Writes images + remapped labels into --out, mirroring a flat layout so the
     result can flow straight into preprocess.py → make_splits.py alongside
     FracAtlas (multi-`--src`).

This adapter does NOT download anything. Point --src at an already-downloaded
GRAZPEDWRI-DX root (images + YOLO labels). The fracture class index defaults to
3 (the value in the dataset's published data.yaml at time of writing) but is
configurable via --fracture-class since upstream ordering can change — ALWAYS
confirm against the dataset's own data.yaml before a real run.

Usage
-----
    python scripts/ingest_grazpedwri.py \
        --images path/to/GRAZPEDWRI/images \
        --labels path/to/GRAZPEDWRI/labels \
        --out GRAZPEDWRI_bone_r --fracture-class 3

    # then harmonize + merge with FracAtlas:
    python scripts/preprocess.py --src GRAZPEDWRI_bone_r --out GRAZ_proc
    python scripts/make_splits.py --src FracAtlas_proc --src GRAZ_proc --out dataset_v3 --copy
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg"}
FRACTURE_CLASS_ID = 0  # Bone-R target id


def remap_label(text: str, fracture_class: int) -> list[str]:
    """Keep only fracture rows, remap their class id to 0. Returns YOLO lines."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        if cls != fracture_class:
            continue  # drop non-fracture classes
        # Replace class id with 0; keep the 4 normalized box coords as-is.
        out.append(" ".join([str(FRACTURE_CLASS_ID)] + parts[1:5]))
    return out


def find_label(labels_root: Path, stem: str) -> Path | None:
    hits = list(labels_root.rglob(f"{stem}.txt"))
    return hits[0] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest GRAZPEDWRI-DX → Bone-R format")
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fracture-class", type=int, default=3,
                    help="Source class id for 'fracture' in GRAZPEDWRI data.yaml")
    ap.add_argument("--copy", action="store_true",
                    help="Copy images (default symlink; copy is Windows-safe)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in args.images.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        print(f"No images found under {args.images}")
        return

    n_pos = n_neg = n_boxes = 0
    for img in imgs:
        lbl = find_label(args.labels, img.stem)
        lines = remap_label(lbl.read_text(), args.fracture_class) if lbl else []
        n_boxes += len(lines)
        if lines:
            n_pos += 1
        else:
            n_neg += 1

        dst_img = args.out / img.name
        if not dst_img.exists():
            try:
                if args.copy:
                    shutil.copy2(img, dst_img)
                else:
                    dst_img.symlink_to(img.resolve())
            except OSError:
                shutil.copy2(img, dst_img)
        (args.out / f"{img.stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    print("GRAZPEDWRI-DX ingest complete.")
    print(f"  images          : {len(imgs)}")
    print(f"  with fracture   : {n_pos}")
    print(f"  background      : {n_neg}")
    print(f"  fracture boxes  : {n_boxes}")
    print(f"  output          : {args.out}")
    print(f"  (fracture class {args.fracture_class} -> 0; other classes dropped)")
    if n_pos == 0:
        print("  WARNING: 0 fracture rows kept — check --fracture-class vs the "
              "dataset's data.yaml!")


if __name__ == "__main__":
    main()

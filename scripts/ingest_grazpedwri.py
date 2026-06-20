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


def remap_label(text: str, keep_ids: set[int] | None) -> list[str]:
    """Keep rows whose class id is in *keep_ids*, remap them to fracture id 0.

    keep_ids = None  -> keep EVERY row (all classes are fracture subtypes, e.g.
                        the HUMERUS set: oblique/transverse/segmental/spiral).
    keep_ids = {..}  -> keep only those source class ids (e.g. the hip set's
                        true-fracture classes: intertrochanteric / femoral neck /
                        subtrochanteric), DROPPING landmarks / normal classes.
    """
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
        if keep_ids is not None and cls not in keep_ids:
            continue  # drop non-fracture classes
        # Replace class id with 0; keep the 4 normalized box coords as-is.
        out.append(" ".join([str(FRACTURE_CLASS_ID)] + parts[1:5]))
    return out


def resolve_class_ids(data_yaml: Path, names: list[str]) -> set[int]:
    """Map fracture-class NAMES to their indices using a YOLO data.yaml.

    Robust against index reordering across dataset versions — match by name
    (case-insensitive, trimmed). Raises if a requested name isn't found so the
    overnight run fails loudly rather than silently dropping a fracture class.
    """
    import yaml
    spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    raw = spec.get("names", [])
    # names may be a list or an index->name dict
    items = raw.items() if isinstance(raw, dict) else enumerate(raw)
    by_name = {str(v).strip().lower(): int(k) for k, v in items}
    ids = set()
    for n in names:
        key = n.strip().lower()
        if key not in by_name:
            raise SystemExit(f"class name {n!r} not in data.yaml names {list(by_name)}")
        ids.add(by_name[key])
    return ids


def find_label(labels_root: Path, stem: str) -> Path | None:
    hits = list(labels_root.rglob(f"{stem}.txt"))
    return hits[0] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest GRAZPEDWRI-DX -> Bone-R format")
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fracture-class", type=int, default=3,
                    help="Source class id for 'fracture' in GRAZPEDWRI data.yaml")
    ap.add_argument("--copy", action="store_true",
                    help="Copy images (default symlink; copy is Windows-safe)")
    ap.add_argument("--all-fracture", action="store_true",
                    help="Treat ALL classes as fracture (datasets typed by "
                         "fracture morphology, e.g. HUMERUS oblique/transverse/...)")
    ap.add_argument("--fracture-classes", default=None,
                    help="Comma list of SOURCE class ids to keep as fracture "
                         "(e.g. '0,1,2'). Overrides --fracture-class.")
    ap.add_argument("--fracture-names", default=None,
                    help="Comma list of fracture class NAMES to keep (resolved "
                         "via --data-yaml). Use for the hip set, e.g. "
                         "'intertrochanteric,femoral neck,subtrochanteric'.")
    ap.add_argument("--data-yaml", type=Path, default=None,
                    help="data.yaml for --fracture-names resolution")
    args = ap.parse_args()

    # Resolve which source class ids count as fracture.
    if args.all_fracture:
        keep_ids = None                                   # keep every row
    elif args.fracture_names:
        if not args.data_yaml:
            raise SystemExit("--fracture-names requires --data-yaml")
        keep_ids = resolve_class_ids(args.data_yaml,
                                     args.fracture_names.split(","))
    elif args.fracture_classes:
        keep_ids = {int(x) for x in args.fracture_classes.split(",")}
    else:
        keep_ids = {args.fracture_class}
    print(f"keeping source class ids: {keep_ids if keep_ids is not None else 'ALL'}")

    args.out.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in args.images.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        print(f"No images found under {args.images}")
        return

    n_pos = n_neg = n_boxes = 0
    for img in imgs:
        lbl = find_label(args.labels, img.stem)
        lines = remap_label(lbl.read_text(), keep_ids) if lbl else []
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
    kept = "ALL" if keep_ids is None else sorted(keep_ids)
    print(f"  (kept source class ids {kept} -> 0; other classes dropped)")
    if n_pos == 0:
        print("  WARNING: 0 fracture rows kept — check the fracture class "
              "selection vs the dataset's data.yaml!")


if __name__ == "__main__":
    main()

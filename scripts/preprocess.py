"""
Phase 1 — shared preprocessing contract for FracAtlas + MURA.

Harmonizes the two datasets into one consistent training medium so a detector
trained across them isn't fighting format/contrast/scale inconsistency. Applies
the SAME deterministic pipeline to every image, regardless of source:

  1. Robust decode (PNG/JPG/DICOM), repairing truncated JPEGs instead of dropping
     them (FracAtlas ships ~37 truncated files).
  2. DICOM window/level via rescale slope/intercept so films from different
     machines land on a common intensity scale before 8-bit normalization.
  3. Grayscale normalization (X-rays are single-channel) -> 3-channel for CNNs.
  4. CLAHE (contrast-limited adaptive histogram equalization) for LOCAL contrast
     on bone edges — sharpens hairline fractures without the noise amplification
     of global sharpening. Clip-limited so it can't blow out the image.
  5. Aspect-PRESERVING downscale to a max long side. This is label-safe: YOLO
     boxes are normalized fractions, and scaling both axes by the same factor
     leaves those fractions unchanged, so sibling .txt labels are copied as-is.
     (We deliberately do NOT letterbox/pad here — padding would shift the
     normalized coords; YOLO applies letterboxing internally at train time.)

Output mirrors the source layout under --out, writing PNGs + copied .txt labels,
ready to feed straight into make_splits.py.

Usage
-----
    # harmonize FracAtlas (images carry sibling YOLO .txt from fracatlas_to_yolo)
    python scripts/preprocess.py --src FracAtlas/images --out FracAtlas_proc

    # harmonize both sources together
    python scripts/preprocess.py --src FracAtlas/images --src MURA --out proc \
        --max-size 1024

    # then rebuild the split from harmonized images
    python scripts/make_splits.py --src FracAtlas_proc --out dataset --copy
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile

# Repair rather than reject truncated JPEGs (FracAtlas has several).
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import pydicom
except Exception:  # optional
    pydicom = None

IMG_EXTS = {".png", ".jpg", ".jpeg"}
DICOM_EXTS = {".dcm", ".dicom"}


def decode(path: Path) -> np.ndarray | None:
    """Load any supported X-ray -> 8-bit grayscale 2D array, or None on failure."""
    ext = path.suffix.lower()
    try:
        if ext in DICOM_EXTS:
            if pydicom is None:
                return None
            ds = pydicom.dcmread(str(path))
            arr = ds.pixel_array.astype(np.float32)
            # Apply rescale slope/intercept (modality LUT) if present.
            slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
            intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
            arr = arr * slope + intercept
            # Window/level if specified, else min-max.
            wc = getattr(ds, "WindowCenter", None)
            ww = getattr(ds, "WindowWidth", None)
            if wc is not None and ww is not None:
                wc = float(wc[0] if isinstance(wc, (list, tuple)) else wc)
                ww = float(ww[0] if isinstance(ww, (list, tuple)) else ww)
                lo, hi = wc - ww / 2, wc + ww / 2
            else:
                lo, hi = float(arr.min()), float(arr.max())
            arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
            # MONOCHROME1 = inverted (white background); flip to MONOCHROME2.
            if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
                arr = 1.0 - arr
            return (arr * 255).astype(np.uint8)

        # PNG/JPG via PIL (handles truncation), to grayscale.
        img = Image.open(path).convert("L")
        return np.asarray(img)
    except Exception as e:
        print(f"  ! decode failed {path.name}: {e}")
        return None


def apply_clahe(gray: np.ndarray, clip: float, grid: int) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)


def resize_long_side(img: np.ndarray, max_size: int) -> np.ndarray:
    """Aspect-preserving downscale so the long side <= max_size. Label-safe."""
    h, w = img.shape[:2]
    long = max(h, w)
    if max_size <= 0 or long <= max_size:
        return img
    scale = max_size / long
    new = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(img, new, interpolation=cv2.INTER_AREA)


def iter_images(src: Path):
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() in (IMG_EXTS | DICOM_EXTS):
            yield p


def process_one(path: Path, src_root: Path, out_root: Path, args) -> str:
    gray = decode(path)
    if gray is None:
        return "skipped"

    if not args.no_clahe:
        gray = apply_clahe(gray, args.clahe_clip, args.clahe_grid)
    gray = resize_long_side(gray, args.max_size)

    # Grayscale -> 3-channel for CNN backbones.
    out_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    rel = path.relative_to(src_root)
    dst = (out_root / rel).with_suffix(".png")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), out_img)

    # Copy sibling YOLO label unchanged (normalized coords are resize-invariant).
    lbl = path.with_suffix(".txt")
    if lbl.exists():
        shutil.copy2(lbl, dst.with_suffix(".txt"))
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 preprocessing contract")
    ap.add_argument("--src", action="append", required=True, type=Path,
                    help="Source root(s) (repeatable)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-size", type=int, default=1024,
                    help="Max long side in px (aspect-preserving). 0 = no resize")
    ap.add_argument("--clahe-clip", type=float, default=2.0)
    ap.add_argument("--clahe-grid", type=int, default=8)
    ap.add_argument("--no-clahe", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only N images (smoke test). 0 = all")
    args = ap.parse_args()

    counts = {"ok": 0, "skipped": 0}
    n = 0
    for src in args.src:
        for path in iter_images(src):
            counts[process_one(path, src, args.out, args)] += 1
            n += 1
            if n % 500 == 0:
                print(f"  processed {n} ...")
            if args.limit and n >= args.limit:
                break
        if args.limit and n >= args.limit:
            break

    print("Preprocessing complete.")
    print(f"  written : {counts['ok']}")
    print(f"  skipped : {counts['skipped']}")
    print(f"  output  : {args.out}")
    print("  CLAHE   :", "off" if args.no_clahe else
          f"clip={args.clahe_clip} grid={args.clahe_grid}")
    print(f"  max long side: {args.max_size or 'unchanged'}")


if __name__ == "__main__":
    main()

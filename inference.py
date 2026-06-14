"""
Shared fracture-inference helpers: image loading (PNG/JPG/DICOM), running the
YOLO model, and heuristic "best guess" fracture type + severity.

The type/severity estimates are intentionally simple, rule-based heuristics
derived from box geometry and model confidence. They are NOT a diagnosis — the
API surfaces them as low-confidence hints and always recommends an orthopedic
consult. A real system would train a dedicated multi-class / grading head.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, asdict
from functools import lru_cache

import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import pydicom  # optional, only needed for DICOM input
except Exception:  # pragma: no cover
    pydicom = None


@dataclass
class Detection:
    bbox_xyxy: list[float]      # [x1, y1, x2, y2] in pixels
    confidence: float
    fracture_type: str          # heuristic best guess
    severity: str               # heuristic best guess
    notes: str

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=2)
def load_model(weights: str):
    from ultralytics import YOLO
    return YOLO(weights)


def read_image(data: bytes, filename: str) -> np.ndarray:
    """Decode bytes -> RGB uint8 array. Supports PNG/JPG and DICOM."""
    name = filename.lower()
    if name.endswith(".dcm") or name.endswith(".dicom"):
        if pydicom is None:
            raise RuntimeError("pydicom not installed; cannot read DICOM")
        ds = pydicom.dcmread(io.BytesIO(data))
        arr = ds.pixel_array.astype(np.float32)
        # Normalize to 8-bit grayscale, then stack to 3 channels.
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        arr = (arr * 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr
    # PNG / JPG path via PIL.
    from PIL import Image
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(img)


def guess_type_and_severity(bbox, conf, img_w, img_h):
    """Heuristic fracture type + severity from box geometry and confidence."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    area_frac = (w * h) / (img_w * img_h + 1e-9)
    aspect = max(w, h) / (min(w, h) + 1e-9)

    # Type heuristic from box shape (very rough).
    if aspect > 3.0:
        ftype = "transverse/linear (best guess)"
    elif area_frac > 0.06:
        ftype = "comminuted/complex (best guess)"
    else:
        ftype = "localized/simple (best guess)"

    # Severity from combined size + confidence.
    score = area_frac * 4 + conf
    if score > 0.9 or area_frac > 0.08:
        severity = "high (best guess)"
    elif score > 0.5:
        severity = "moderate (best guess)"
    else:
        severity = "low (best guess)"

    return ftype, severity


def run_inference(model, image: np.ndarray, conf: float = 0.25) -> list[Detection]:
    h, w = image.shape[:2]
    results = model.predict(image, conf=conf, verbose=False)
    dets: list[Detection] = []
    for r in results:
        for box in r.boxes:
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            c = float(box.conf[0])
            ftype, sev = guess_type_and_severity(xyxy, c, w, h)
            dets.append(Detection(
                bbox_xyxy=[round(v, 1) for v in xyxy],
                confidence=round(c, 4),
                fracture_type=ftype,
                severity=sev,
                notes="Heuristic estimate — not a diagnosis.",
            ))
    return dets

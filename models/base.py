"""Common detector interface + registry."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

# Allow `from inference import ...` when imported from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference import Detection, guess_type_and_severity  # noqa: E402


class Detector(ABC):
    """Backend-agnostic detector. predict() returns inference.Detection list."""

    @abstractmethod
    def predict(self, image: np.ndarray, conf: float = 0.25) -> list[Detection]:
        ...

    @staticmethod
    def _to_detections(boxes_xyxy, scores, img_w, img_h) -> list[Detection]:
        dets: list[Detection] = []
        for box, score in zip(boxes_xyxy, scores):
            xyxy = [float(v) for v in box]
            c = float(score)
            ftype, sev = guess_type_and_severity(xyxy, c, img_w, img_h)
            dets.append(Detection(
                bbox_xyxy=[round(v, 1) for v in xyxy],
                confidence=round(c, 4),
                fracture_type=ftype,
                severity=sev,
                notes="Heuristic estimate — not a diagnosis.",
            ))
        return dets


DETECTOR_REGISTRY: dict[str, type[Detector]] = {}


def register(name: str):
    def deco(cls: type[Detector]):
        DETECTOR_REGISTRY[name] = cls
        return cls
    return deco


def build_detector(name: str, **kwargs) -> Detector:
    if name not in DETECTOR_REGISTRY:
        raise ValueError(
            f"Unknown detector '{name}'. Available: {list(DETECTOR_REGISTRY)}")
    return DETECTOR_REGISTRY[name](**kwargs)


# Import concrete backends so they self-register. Kept lazy-tolerant: a missing
# optional dependency (e.g. torchvision) only disables that one backend.
from . import yolo_backend   # noqa: E402,F401
try:
    from . import torchvision_backend  # noqa: E402,F401
except Exception:  # torch/torchvision not installed
    pass

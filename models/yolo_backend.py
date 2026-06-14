"""Ultralytics YOLOv8 backend (default detector)."""

from __future__ import annotations

import numpy as np

from .base import Detector, register
from inference import run_inference, load_model


@register("yolov8")
class YOLOv8Detector(Detector):
    def __init__(self, weights: str = "yolov8s.pt", **_):
        self.model = load_model(weights)

    def predict(self, image: np.ndarray, conf: float = 0.25):
        # YOLO path already produces inference.Detection objects.
        return run_inference(self.model, image, conf=conf)

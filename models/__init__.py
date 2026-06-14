"""Pluggable fracture-detector framework.

A single `Detector` interface over the strongest object-detection CNNs so you
can train/serve any of them without touching the rest of the pipeline
(visualize.py, app/main.py, ensemble.py all consume `inference.Detection`).

Registry:
    yolov8     Ultralytics YOLOv8  (anchor-free, fast, strong default)
    fasterrcnn torchvision Faster R-CNN ResNet50-FPN (two-stage, high precision)
    retinanet  torchvision RetinaNet ResNet50-FPN  (one-stage, focal loss)
    fcos       torchvision FCOS ResNet50-FPN        (anchor-free one-stage)

    from models import build_detector
    det = build_detector("retinanet", weights="rn.pt", num_classes=1)
    detections = det.predict(rgb_image, conf=0.25)
"""

from .base import Detector, build_detector, DETECTOR_REGISTRY

__all__ = ["Detector", "build_detector", "DETECTOR_REGISTRY"]

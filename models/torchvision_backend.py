"""
torchvision CNN detector backends: Faster R-CNN, RetinaNet, FCOS.

These give you stronger two-stage precision (Faster R-CNN) and focal-loss
one-stage recall (RetinaNet) alternatives to YOLO under one interface. Each is
built with a custom head sized for `num_classes` (default 1 = fracture, plus an
implicit background class handled by torchvision).

Pass `weights` to load a fine-tuned checkpoint (a state_dict .pt). With no
weights the model uses COCO-pretrained backbone weights — useful as an
initialized starting point for fine-tuning via train_torchvision.py.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
    fcos_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    FCOS_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import ResNet50_Weights

from .base import Detector, register


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class _TorchvisionDetector(Detector):
    """Shared predict() for torchvision detection models."""

    def __init__(self, model, weights: str | None):
        self.device = _device()
        if weights:
            state = torch.load(weights, map_location=self.device)
            model.load_state_dict(state.get("model", state))
        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image: np.ndarray, conf: float = 0.25):
        h, w = image.shape[:2]
        # HWC uint8 RGB -> CHW float tensor in [0,1].
        t = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        out = self.model([t.to(self.device)])[0]
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        keep = scores >= conf
        return self._to_detections(boxes[keep], scores[keep], w, h)


@register("fasterrcnn")
class FasterRCNNDetector(_TorchvisionDetector):
    def __init__(self, weights: str | None = None, num_classes: int = 1, **_):
        # +1 for the implicit background class in two-stage detectors.
        m = fasterrcnn_resnet50_fpn_v2(
            weights=None if weights else FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        in_features = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        super().__init__(m, weights)


# One-stage detectors (RetinaNet, FCOS) have no separate background class, but
# we keep num_classes+1 so valid label ids match the two-stage convention used
# by train_torchvision.py (which assigns fracture = label 1). Critically, we
# build a FRESH detection head of the correct size on a pretrained BACKBONE for
# BOTH training and eval — never the full 91-class COCO detector. The previous
# code built the 91-class COCO head during training (weights=None branch), so
# the saved checkpoint couldn't load back into a fracture-sized model.
_BACKBONE_W = ResNet50_Weights.IMAGENET1K_V1


@register("retinanet")
class RetinaNetDetector(_TorchvisionDetector):
    def __init__(self, weights: str | None = None, num_classes: int = 1, **_):
        m = retinanet_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None if weights else _BACKBONE_W,
            num_classes=num_classes + 1)
        super().__init__(m, weights)


@register("fcos")
class FCOSDetector(_TorchvisionDetector):
    def __init__(self, weights: str | None = None, num_classes: int = 1, **_):
        m = fcos_resnet50_fpn(
            weights=None,
            weights_backbone=None if weights else _BACKBONE_W,
            num_classes=num_classes + 1)
        super().__init__(m, weights)

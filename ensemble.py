"""
Detector + MURA-classifier ensemble.

Combines the FracAtlas YOLO *detector* (localizes fractures) with the MURA
*abnormality classifier* (study-level second opinion). The classifier's
abnormal-probability gates / re-weights detector confidences:

    fused_conf = det_conf * (alpha + (1 - alpha) * p_abnormal)

so a detection survives best when both heads agree. When the classifier is
confident the film is normal, weak detections are suppressed; a confidently
abnormal classifier preserves borderline detections (favoring sensitivity).

Usage (programmatic)
--------------------
    from ensemble import FractureEnsemble
    ens = FractureEnsemble("det.pt", "cls.pt")
    result = ens.predict(rgb_image, conf=0.15)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from inference import load_model, run_inference, Detection


@dataclass
class EnsembleResult:
    p_abnormal: float
    fracture_detected: bool
    detections: list[Detection] = field(default_factory=list)


class FractureEnsemble:
    def __init__(self, detector_weights: str, classifier_weights: str,
                 alpha: float = 0.5):
        self.detector = load_model(detector_weights)
        self.classifier = load_model(classifier_weights)
        self.alpha = alpha

    def _classify(self, image) -> float:
        """Return P(abnormal) from the MURA classifier."""
        r = self.classifier.predict(image, verbose=False)[0]
        names = r.names  # idx -> class name
        probs = r.probs.data.tolist()
        # Find the 'abnormal' class index robustly.
        for idx, name in names.items():
            if str(name).lower().startswith("abnormal"):
                return float(probs[idx])
        # Fallback: assume binary, abnormal is the non-'normal' class.
        return float(max(probs))

    def predict(self, image, conf: float = 0.15,
                keep_threshold: float = 0.25) -> EnsembleResult:
        p_abnormal = self._classify(image)
        # Run the detector at a low conf to stay sensitive, then re-weight.
        dets = run_inference(self.detector, image, conf=conf)
        kept: list[Detection] = []
        for d in dets:
            fused = d.confidence * (self.alpha + (1 - self.alpha) * p_abnormal)
            if fused >= keep_threshold:
                d.confidence = round(fused, 4)
                kept.append(d)
        # If the classifier is strongly abnormal but the detector found nothing,
        # flag for review rather than silently clearing the film.
        flagged = p_abnormal >= 0.8 and not kept
        return EnsembleResult(
            p_abnormal=round(p_abnormal, 4),
            fracture_detected=bool(kept) or flagged,
            detections=kept,
        )


if __name__ == "__main__":
    import argparse
    from inference import read_image

    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--conf", type=float, default=0.15)
    args = ap.parse_args()

    ens = FractureEnsemble(args.detector, args.classifier)
    with open(args.image, "rb") as fh:
        rgb = read_image(fh.read(), args.image)
    res = ens.predict(rgb, conf=args.conf)
    print(f"P(abnormal)      : {res.p_abnormal}")
    print(f"fracture flagged : {res.fracture_detected}")
    for d in res.detections:
        print(f"  {d.bbox_xyxy}  conf={d.confidence}  {d.severity}")

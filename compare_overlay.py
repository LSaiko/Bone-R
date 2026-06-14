"""
Multi-model comparison overlay — "which is which".

Runs several CNN backends on the same X-ray and draws each model's predicted
boxes in its own color with a legend, so you can visually compare where the
models agree/disagree. Saves a showcase PNG and a JSON of per-model boxes that
the GitHub Pages viewer (docs/) consumes for an interactive overlay.

Usage
-----
    python compare_overlay.py --image xray.png --out docs/assets/compare.png \
        --json docs/data/predictions.json \
        --model yolov8=runs/detect/fracture_yolov8s/weights/best.pt \
        --model retinanet=retinanet_fracture.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from inference import read_image
from models import build_detector

# Stable BGR colors per model (and matching hex for the web legend).
MODEL_COLORS = {
    "yolov8":     ((0, 200, 0),   "#00c800"),   # green
    "fasterrcnn": ((0, 0, 255),   "#ff0000"),   # red
    "retinanet":  ((255, 128, 0), "#0080ff"),   # blue
    "fcos":       ((0, 200, 255), "#ffc800"),   # amber
}
DEFAULT = ((255, 255, 255), "#ffffff")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", action="append", required=True,
                    help="name=weights (repeatable)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="compare.png")
    ap.add_argument("--json", default=None,
                    help="Optional JSON export for the web viewer")
    args = ap.parse_args()

    with open(args.image, "rb") as fh:
        rgb = read_image(fh.read(), args.image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = rgb.shape[:2]

    export = {"image": Path(args.image).name, "width": w, "height": h,
              "models": {}}

    y_legend = 30
    for spec in args.model:
        name, weights = spec.split("=", 1)
        bgr_color, hex_color = MODEL_COLORS.get(name, DEFAULT)
        det = build_detector(name, weights=weights, num_classes=1)
        dets = det.predict(rgb, conf=args.conf)

        export["models"][name] = {
            "color": hex_color,
            "boxes": [
                {"xyxy": d.bbox_xyxy, "conf": d.confidence,
                 "type": d.fracture_type, "severity": d.severity}
                for d in dets
            ],
        }

        for d in dets:
            x1, y1, x2, y2 = map(int, d.bbox_xyxy)
            cv2.rectangle(bgr, (x1, y1), (x2, y2), bgr_color, 2)
            cv2.putText(bgr, f"{name} {d.confidence:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 1)

        # Legend entry
        cv2.rectangle(bgr, (10, y_legend - 12), (28, y_legend + 2),
                      bgr_color, -1)
        cv2.putText(bgr, f"{name} ({len(dets)})", (34, y_legend),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2)
        y_legend += 26

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, bgr)
    print(f"wrote overlay -> {args.out}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(export, indent=2), encoding="utf-8")
        print(f"wrote predictions -> {args.json}")


if __name__ == "__main__":
    main()

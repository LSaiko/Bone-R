"""
Draw predicted fracture boxes onto an X-ray and save the overlay.

Usage
-----
    python visualize.py --weights runs/detect/fracture_yolov8s/weights/best.pt \
        --image xray.png --out out.png --conf 0.25
"""

import argparse
import cv2

from inference import load_model, read_image, run_inference

# BGR colors keyed by severity.
COLORS = {
    "high (best guess)": (0, 0, 255),       # red
    "moderate (best guess)": (0, 165, 255),  # orange
    "low (best guess)": (0, 200, 0),         # green
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="prediction.png")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    with open(args.image, "rb") as fh:
        rgb = read_image(fh.read(), args.image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    model = load_model(args.weights)
    dets = run_inference(model, rgb, conf=args.conf)

    for d in dets:
        x1, y1, x2, y2 = map(int, d.bbox_xyxy)
        color = COLORS.get(d.severity, (255, 255, 255))
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{d.confidence:.2f} {d.severity.split(' ')[0]}"
        cv2.putText(bgr, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(args.out, bgr)
    print(f"Wrote {len(dets)} detection(s) -> {args.out}")


if __name__ == "__main__":
    main()

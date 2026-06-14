"""
FastAPI service for fracture detection.

POST /detect
    multipart file upload (PNG/JPG/DICOM X-ray) -> JSON with bounding boxes,
    confidence scores, heuristic type/severity, and an orthopedic-consult
    recommendation. If lat/lng query params are supplied and a Google Maps API
    key is configured, nearby orthopedic clinics are looked up.

Run:
    export FRACTURE_WEIGHTS=runs/detect/fracture_yolov8s/weights/best.pt
    export GOOGLE_MAPS_API_KEY=...        # optional
    uvicorn app.main:app --reload --port 8000

Disclaimer: research/educational tool. Not a medical device; outputs are
"best guess" estimates and must not be used for diagnosis.
"""

from __future__ import annotations

import os

import httpx
from fastapi import FastAPI, File, UploadFile, Query, HTTPException

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference import read_image, conf_for_mode  # noqa: E402
from models import build_detector  # noqa: E402

WEIGHTS = os.environ.get("FRACTURE_WEIGHTS", "yolov8s.pt")
# Backend: yolov8 (default) | fasterrcnn | retinanet | fcos
BACKEND = os.environ.get("FRACTURE_BACKEND", "yolov8")
MAPS_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = build_detector(BACKEND, weights=WEIGHTS, num_classes=1)
    return _detector

app = FastAPI(title="Bone-R Fracture Detection", version="1.0")


def find_orthopedic_clinics(lat: float, lng: float, radius_m: int = 20000):
    """Google Places Nearby Search for orthopedic clinics near the user."""
    if not MAPS_KEY:
        return {"available": False,
                "reason": "GOOGLE_MAPS_API_KEY not configured"}
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius_m,
        "keyword": "orthopedic",
        "type": "doctor",
        "key": MAPS_KEY,
    }
    try:
        r = httpx.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"available": False, "reason": str(e)}
    results = [
        {
            "name": p.get("name"),
            "address": p.get("vicinity"),
            "rating": p.get("rating"),
            "location": p.get("geometry", {}).get("location"),
            "maps_url": f"https://www.google.com/maps/place/?q=place_id:{p.get('place_id')}",
        }
        for p in data.get("results", [])[:5]
    ]
    return {"available": True, "clinics": results}


@app.get("/health")
def health():
    return {"status": "ok", "backend": BACKEND, "weights": WEIGHTS,
            "maps_enabled": bool(MAPS_KEY)}


_MODE_NOTES: dict[str, str] = {
    "screening": (
        "High-recall screening mode — favors catching fractures over avoiding "
        "false alarms; verify all positives clinically."
    ),
    "balanced": (
        "Balanced mode — default operating point; trades recall and precision "
        "roughly equally."
    ),
    "specific": (
        "High-precision mode — reduces false alarms; may miss subtle fractures; "
        "use only when additional clinical evidence is already available."
    ),
}

_CONF_SENTINEL = -1.0  # signals that no explicit conf was provided


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    # mode selects a predefined operating point (screening | balanced | specific).
    # An explicit conf= still wins as a manual override.
    mode: str = Query("balanced"),
    conf: float = Query(_CONF_SENTINEL, ge=-1.0, le=1.0),
    lat: float | None = Query(None),
    lng: float | None = Query(None),
):
    # Resolve effective confidence threshold.
    # Explicit conf= takes priority; otherwise derive from mode.
    if conf != _CONF_SENTINEL:
        # Caller provided an explicit override — honour it and note the mode as
        # "custom" so the response is transparent.
        effective_conf = conf
        effective_mode = "custom"
        mode_note = (
            f"Custom threshold override — conf={conf:.2f} was supplied "
            "directly and takes precedence over mode."
        )
    else:
        effective_conf = conf_for_mode(mode)
        effective_mode = mode if mode in _MODE_NOTES else "balanced"
        mode_note = _MODE_NOTES.get(effective_mode, _MODE_NOTES["balanced"])

    data = await file.read()
    try:
        image = read_image(data, file.filename or "upload.png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    dets = get_detector().predict(image, conf=effective_conf)

    fractured = len(dets) > 0
    response = {
        "filename": file.filename,
        "operating_mode": {
            "mode": effective_mode,
            "conf_threshold": effective_conf,
            "note": mode_note,
        },
        "image_size": {"height": image.shape[0], "width": image.shape[1]},
        "fracture_detected": fractured,
        "num_detections": len(dets),
        "detections": [d.to_dict() for d in dets],
        "recommendation": (
            "Findings suggest a possible fracture. Seek an orthopedic consult "
            "for confirmation and management."
            if fractured else
            "No fracture detected by the model. If symptoms persist, consult a "
            "clinician — model negatives are not definitive."
        ),
        "disclaimer": "Research tool. Best-guess estimates, not a diagnosis.",
    }

    if fractured and lat is not None and lng is not None:
        response["orthopedic_consult"] = find_orthopedic_clinics(lat, lng)

    return response

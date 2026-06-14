# Bone-R — Bone Fracture Detection (FracAtlas + MURA → YOLOv8)

[![Site Data Validation](https://github.com/lsaiko/Bone-R/actions/workflows/site-data.yml/badge.svg)](https://github.com/lsaiko/Bone-R/actions/workflows/site-data.yml)

End-to-end pipeline to detect, localize, and roughly characterize bone
fractures on X-rays. **FracAtlas** provides true bounding-box annotations for
detection; **MURA** provides large-scale image-level (abnormal/normal) labels
for robustness via co-training / pre-training. The two datasets can be used
separately or together.

> ⚠️ Research/educational tool only. Outputs are "best guess" estimates and are
> **not** a medical diagnosis. Always recommend a qualified orthopedic consult.

## Results (current model)

Detector **YOLOv8m**, trained on the Phase-1 harmonized FracAtlas split
(`dataset_v2`), evaluated on the held-out test split. The screening metrics
matter more than mAP here — for fracture triage a *miss* is worse than a false
alarm — so image-level sensitivity/specificity are reported alongside.

**Backbone comparison** (same test split, single `fracture` class):

| Backbone | mAP@0.5 | mAP@0.5:0.95 | Recall |
|---|---|---|---|
| **YOLOv8m** (shipped) | **0.460** | **0.198** | **0.429** |
| Faster R-CNN R50-FPNv2 | 0.354 | 0.171 | 0.288 |
| RetinaNet R50-FPNv2 | 0.240 | 0.072 | 0.188 |

**Image-level screening** (YOLOv8m, conf ≥ 0.25): sensitivity **0.561**,
specificity **0.988**, PPV **0.902** (TP 37 / FP 4 / TN 338 / FN 29).

**Honest limitations** (see [JOURNAL.md](JOURNAL.md) for the full analysis):
- Harmonization (CLAHE + truncation repair + imgsz 800 + bigger backbone)
  **nearly doubled** every metric over the v8s baseline (mAP@0.5 0.27 → 0.48).
- But the model still misses ~44% of fractures, and recall **cannot** be tuned
  past **0.797** at any threshold — a *data* limit, not a tuning one.
- Hip/shoulder are near-unevaluable (≈1 / 0 fractured test cases). The path to
  the sensitivity goal runs through **more boxed data** (GRAZPEDWRI-DX, Roboflow
  Hip/Humerus — see `analysis/data_sourcing_log.md`), not more model tweaks.
- A MURA abnormality-classifier ensemble was tested and **rejected** — it traded
  away specificity (0.99 → 0.87) without recovering missed fractures, because
  MURA's "abnormal" label ≠ fracture (Entry 004). Studied negative result.

## Layout

```
scripts/fracatlas_to_yolo.py   COCO JSON  -> YOLO .txt boxes (alongside images)
scripts/mura_to_yolo.py        MURA labels -> classification manifest OR weak boxes
scripts/make_splits.py         build dataset/{images,labels}/{train,val,test}
dataset.yaml                   YOLO dataset config (single class: fracture)
train.py                       Ultralytics YOLOv8 training
evaluate.py                    mAP@0.5, mAP@0.5:0.95 + sensitivity @ high recall
inference.py                   shared load/decode (PNG/JPG/DICOM) + heuristics
visualize.py                   cv2.rectangle overlays saved to an image
app/main.py                    FastAPI /detect endpoint (+ Google Maps consult)
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. FracAtlas: COCO -> YOLO labels written next to each image
python scripts/fracatlas_to_yolo.py

# 2. (optional) MURA classification manifest for co-training robustness
python scripts/mura_to_yolo.py --mode classify --out MURA_cls
#    ...or weak full-frame detection labels to augment FracAtlas:
# python scripts/mura_to_yolo.py --mode weakbox

# 3. Build train/val/test detection tree (FracAtlas only, or add MURA)
python scripts/make_splits.py --src FracAtlas/images --out dataset
# python scripts/make_splits.py --src FracAtlas/images --src MURA --out dataset

# 4. Train
python train.py --model yolov8s.pt --data dataset.yaml --epochs 50 --imgsz 640
#    (or: yolo detect train model=yolov8s.pt data=dataset.yaml epochs=50)

# 5. Evaluate — standard mAP + the screening-oriented sensitivity metric
python evaluate.py --weights runs/detect/fracture_yolov8s/weights/best.pt \
    --target-recall 0.95

# 6. Visualize predictions
python visualize.py --weights runs/detect/fracture_yolov8s/weights/best.pt \
    --image xray.png --out prediction.png --conf 0.25

# 7. Serve the API
export FRACTURE_WEIGHTS=runs/detect/fracture_yolov8s/weights/best.pt
export GOOGLE_MAPS_API_KEY=...        # optional, for orthopedic-consult lookup
uvicorn app.main:app --reload --port 8000
```

### API example

```bash
curl -F file=@xray.png \
  "http://localhost:8000/detect?conf=0.25&lat=37.77&lng=-122.42"
```

Returns JSON: bounding boxes (`xyxy` + confidence), heuristic fracture type and
severity, a consult recommendation, and (if a Maps key + coordinates are given)
nearby orthopedic clinics.

## Notes on the two datasets

- **FracAtlas** is single-class and box-annotated → drives true detection.
  `fracatlas_to_yolo.py` reads `Annotations/COCO JSON/COCO_fracture_masks.json`,
  converts each COCO `[x,y,w,h]` box to normalized YOLO `cx cy w h`, writes a
  populated `.txt` next to fractured images and an **empty** `.txt` next to
  non-fractured images (valid negative/background samples).
- **MURA** has **no boxes** — labels are study-level (`*_positive` /
  `*_negative`). It cannot yield true localized boxes. Use the classification
  manifest to pre-train/co-train a classifier head, or `--mode weakbox` to emit
  whole-image weak labels (noisier; use deliberately).

## Phase 1 — Dataset preprocessing (harmonization)

Before training, both datasets are harmonized through a shared preprocessing contract
applied offline:

```powershell
# Harmonize FracAtlas (and optionally MURA) into a consistent medium
python scripts\preprocess.py --src FracAtlas\images --out FracAtlas_proc --max-size 1024
# Then build the split from the harmonized images
python scripts\make_splits.py --src FracAtlas_proc --out dataset --copy
```

**What it does:**
- **Repairs truncated JPEGs** — recovers ~37 FracAtlas files that YOLO otherwise drops
- **CLAHE (contrast-limited adaptive histogram equalization)** — enhances local
  contrast on bone edges, sharpens hairline fractures without noise amplification
- **Grayscale → 3-channel RGB** — consistent representation across datasets
- **DICOM window/level** — applies radiographic rescale (slope/intercept) so
  different machines land on the same intensity scale
- **Aspect-preserving resize** — scales to a max long side (default 1024) without
  distortion, keeping YOLO labels valid without modification

Run this **after** the label converters (`fracatlas_to_yolo.py` / `mura_to_yolo.py`)
but **before** `make_splits.py`.

## MURA co-training (robustness)

MURA's 40k labeled films train a study-level **abnormality classifier** that
acts as a second opinion on the FracAtlas detector.

```bash
python scripts/mura_to_yolo.py --mode classify --out MURA_cls   # build tree
python train_mura_cls.py --data MURA_cls --model yolov8s-cls.pt --epochs 20

# Ensemble: classifier P(abnormal) re-weights detector confidences
python ensemble.py \
  --detector runs/detect/fracture_yolov8s/weights/best.pt \
  --classifier runs/classify/mura_abnormality/weights/best.pt \
  --image xray.png --conf 0.15
```

`fused_conf = det_conf * (alpha + (1-alpha)*p_abnormal)` — a detection survives
best when both heads agree; a confidently-abnormal film keeps borderline boxes
(favoring sensitivity), and a strongly-normal film suppresses weak ones.

## CNN detector framework (swappable backbones)

`models/` exposes one `Detector` interface over the strongest detection CNNs so
you can benchmark backbones on the **same** YOLO-format dataset:

| key          | model                          | trait                        |
|--------------|--------------------------------|------------------------------|
| `yolov8`     | Ultralytics YOLOv8             | fast anchor-free default     |
| `fasterrcnn` | Faster R-CNN ResNet50-FPN v2   | two-stage, high precision    |
| `retinanet`  | RetinaNet ResNet50-FPN v2      | one-stage, focal loss        |
| `fcos`       | FCOS ResNet50-FPN              | anchor-free one-stage        |

```python
from models import build_detector
det = build_detector("retinanet", weights="retinanet_fracture.pt", num_classes=1)
detections = det.predict(rgb_image, conf=0.25)   # -> inference.Detection list
```

```bash
# Fine-tune a torchvision backbone on the YOLO-format tree
python train_torchvision.py --arch retinanet --data dataset --epochs 20 --out rn.pt

# Serve any backend through the same API
export FRACTURE_BACKEND=retinanet
export FRACTURE_WEIGHTS=rn.pt
uvicorn app.main:app --port 8000
```

Every backend returns the shared `inference.Detection` type, so `visualize.py`,
`ensemble.py`, and the FastAPI app work unchanged regardless of model.

## Model comparison & live overlay

Compare backbones head-to-head on the same test split, and render a
"which-is-which" overlay where each model owns a color:

```bash
# Numeric comparison -> console table + JSON for the landing page
python benchmark.py --data dataset.yaml \
  --model yolov8=runs/detect/fracture_yolov8s/weights/best.pt \
  --model retinanet=retinanet_fracture.pt \
  --model fasterrcnn=fasterrcnn_fracture.pt \
  --out docs/data/benchmark.json

# Visual overlay -> showcase PNG + JSON the web viewer consumes
python compare_overlay.py --image xray.png \
  --out docs/assets/compare.png --json docs/data/predictions.json \
  --model yolov8=runs/detect/fracture_yolov8s/weights/best.pt \
  --model retinanet=retinanet_fracture.pt
```

## Landing page (GitHub Pages → lsaiko.github.io/Bone-R)

`docs/` is a static, dependency-free site with an interactive overlay viewer
(toggle models, confidence slider, hover-for-details) and a benchmark table. It
ships with synthetic demo data so it renders immediately; `benchmark.py` and
`compare_overlay.py` overwrite `docs/data/*.json` with real model output.

**Deploy:**

```bash
git init && git add . && git commit -m "Bone-R: fracture detection pipeline + site"
git remote add origin https://github.com/LSaiko/Bone-R.git
git branch -M main && git push -u origin main
```

Then in the repo: **Settings → Pages → Source: Deploy from a branch →
`main` / `/docs`**. The site goes live at `https://lsaiko.github.io/Bone-R/`.
(Datasets and weights are git-ignored; only code + the `docs/` site are pushed.)

## Severity / type heuristics

`inference.guess_type_and_severity` derives a rough fracture *type* (from box
aspect ratio) and *severity* (from box area + model confidence). These are
transparent rule-based hints flagged "best guess", not a learned grading model.
For production, train a dedicated multi-class detection or grading head.

## Datasets & attribution

- **FracAtlas** — Abedeen et al., *FracAtlas: A Dataset for Fracture
  Classification, Localization and Segmentation of Musculoskeletal Radiographs*
  (Scientific Data, 2023). Licensed **CC BY 4.0**. The sample radiograph shown
  on the landing page is a FracAtlas test image, used under that license.
- **MURA** — Rajpurkar et al., *MURA: Large Dataset for Abnormality Detection in
  Musculoskeletal Radiographs* (Stanford ML Group). Used for the abnormality
  classifier experiment under its research data-use terms.

Datasets are **not** redistributed in this repo (git-ignored); only code, the
`docs/` site, and a single attributed sample image are committed.

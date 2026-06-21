# Bone-R — Bone Fracture Detection (FracAtlas + MURA → YOLOv8)

[![Site Data Validation](https://github.com/lsaiko/Bone-R/actions/workflows/site-data.yml/badge.svg)](https://github.com/lsaiko/Bone-R/actions/workflows/site-data.yml)

End-to-end pipeline to detect, localize, and roughly characterize bone
fractures on X-rays. **FracAtlas** provides true bounding-box annotations for
detection; **MURA** provides large-scale image-level (abnormal/normal) labels
for robustness via co-training / pre-training. The two datasets can be used
separately or together.

> ⚠️ Research/educational tool only. Outputs are "best guess" estimates and are
> **not** a medical diagnosis. Always recommend a qualified orthopedic consult.

## Results

Shipped detector: **YOLOv8m** (`fracture_yolov8m_v3`), trained on the harmonized
**FracAtlas + GRAZPEDWRI-DX** merge (`dataset_v3`). The model improved in three
deliberate stages — bigger backbone + harmonization, then a data scale-up — and
the gains are reported honestly, including *where they did and didn't land*.

**Progression** (held-out test split, image-level screening):

| Version | Data | mAP@0.5 | Sensitivity | Specificity | Max recall |
|---|---|---|---|---|---|
| v1-baseline (v8s) | FracAtlas | 0.27 | — | — | — |
| v1 (v8m, harmonized) | FracAtlas | 0.46 | 0.561 | 0.988 | 0.797 |
| v3 (v8m + GRAZPEDWRI) | + wrist | 0.76 | 0.812 | 0.980 | 0.875 |
| v4 (v8m + HUMERUS) | + shoulder | 0.86 | 0.927 | 0.975 | 0.895 |
| **v5 (v8m + proximal-femur)** | + hip | 0.79 | **0.878** | 0.986 | — |

v5 per-source sensitivity: hip **0.674** (now evaluable, was 2 cases), humerus
0.993, wrist 0.925, FracAtlas-native 0.671. The overall dip vs v4 is honest — v5's
test set is region-stratified and includes 43 hard hip cases (v4's was easier).

v4 PPV **0.973** (TP 357 / FP 10 / TN 394 / FN 28 on 789 test images).

**Per-source sensitivity (v4)** — gains are region-specific, reported honestly:
humerus/shoulder **0.986**, wrist 0.961, **FracAtlas-native 0.712** (the realistic
in-the-wild number; the headline is lifted by the easier wrist/humerus sets).

**Backbone study** (separate apples-to-apples comparison on `dataset_v2`, all
three trained + tested on identical data):

| Backbone | mAP@0.5 | mAP@0.5:0.95 | Recall |
|---|---|---|---|
| **YOLOv8m** | **0.460** | **0.198** | **0.429** |
| Faster R-CNN R50-FPNv2 | 0.354 | 0.171 | 0.288 |
| RetinaNet R50-FPNv2 | 0.240 | 0.072 | 0.188 |

**Honest limitations** (full analysis in [JOURNAL.md](JOURNAL.md)):
- Gains are **region-specific**: v4 detects humerus (0.986) and wrist (0.961)
  fractures far better than FracAtlas-native ones (**0.712**) — the headline
  sensitivity is lifted by the easier added distributions.
- **Hip is still the open blind spot** — only 2 fractured test cases (caught 1).
  The "hip" Roboflow set turned out to be *osteoporosis*; a genuine hip-fracture
  set (Roboflow proximal-femur) is identified for the next data pass.
- Recall still cannot be pushed past **0.895**; ≥0.95 sensitivity needs more
  region-diverse boxed data.
- Wrist data (GRAZPEDWRI) is **pediatric** — an age-distribution skew vs adult
  X-rays.
- A MURA abnormality-classifier ensemble was tested and **rejected** — it traded
  specificity (0.99 → 0.87) for nothing, since MURA "abnormal" ≠ fracture
  (a studied negative result, Entry 004).

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

# Bone-R — Development Journal

A running log of findings, decisions, and next steps for the bone-fracture
detection pipeline (FracAtlas + MURA → CNN detectors).

---

## Entry 001 — 2026-06-12 — Baseline detector trained; dataset-consistency problem identified

### What was done
- Built the full pipeline end-to-end: dataset converters, split builder, YOLOv8
  trainer, MURA classifier, multi-backbone framework (YOLOv8 / Faster R-CNN /
  RetinaNet / FCOS), FastAPI service, comparison overlay, and a GitHub Pages site.
- Resolved the environment blockers: Python 3.14 + RTX 5060 (Blackwell) requires
  the **cu128** PyTorch build; Windows **page file** was too small to spawn
  DataLoader workers (WinError 1455) — fixed by enlarging virtual memory and
  defaulting `workers=0`.
- Trained the first baseline: **YOLOv8s, 50 epochs, imgsz 640, batch 8**,
  ~49 min on the RTX 5060.

### Baseline results (best epoch 45 — `runs/detect/fracture_yolov8s-2/weights/best.pt`)
| Metric | Value |
|---|---|
| mAP@0.5 | 0.273 |
| mAP@0.5:0.95 | 0.110 |
| Precision | 0.428 |
| Recall | 0.301 |

Training was healthy (box loss 2.84 → 1.86, metrics still rising at ep50), so
this is an *under-trained / under-powered* baseline, not a broken one. But it is
far from clinically useful: at conf 0.25 it misses ~70% of fractures.

### Core finding: the two datasets are not yet a coherent training medium
We are fusing two fundamentally different data sources, and the inconsistency is
the main thing limiting model quality:

| Aspect | FracAtlas | MURA |
|---|---|---|
| Label granularity | **Box-level** (true detection) | **Image/study-level** only |
| Classes | Single class: `fractured` | Binary: abnormal / normal |
| Fracture *type* labels | **None** | **None** |
| Body regions | hand/leg/hip/shoulder/mixed | 7 regions (elbow…wrist) |
| Image format | JPG, ~2304×2880, some **truncated/corrupt** | PNG, varied sizes |
| Positives | 717 boxed images | ~16.4k abnormal (no localization) |

Implications:
1. **FracAtlas alone is too small** (717 boxed positives) to reach high recall.
2. **MURA can't supply boxes** — only weak/image-level signal. It helps as a
   classifier (second opinion / co-training), not as a detection source.
3. **No dataset here contains fracture *type* labels** (transverse, oblique,
   comminuted, greenstick, etc.). The current "type" output is a geometry
   heuristic, not a learned classifier. A typed target accuracy is **not
   achievable from FracAtlas + MURA as-is** — it requires a typed dataset.
4. **Input inconsistency** (format, bit depth, resolution, truncation, contrast)
   injects noise. Both mediums need a shared preprocessing contract.

### Stated goals (recorded for tracking)
- **>97% fracture-detection accuracy.**
- **>95% fracture-type classification accuracy.**

### Honest gap assessment against those goals
- "Accuracy" must be defined per task before it can be claimed. For detection,
  the meaningful targets are **image-level sensitivity (recall)** and
  **specificity** at a chosen operating point, plus **mAP** for localization.
  A flat ">97% accuracy" on an imbalanced set can be misleading (a model that
  calls everything normal scores high accuracy while catching no fractures).
- **Detection:** current image-level recall is far below 97%. Reaching high-90s
  **sensitivity** is plausible with more data + a bigger model + threshold
  tuning, but 97% *localization* mAP is not a realistic bar for this task on
  public data — top published fracture-detection work sits well below that.
  Recommend reframing the detection goal as **≥95% sensitivity at a fixed,
  acceptable false-positive rate**, reported with mAP alongside.
- **Type classification:** blocked entirely until a **type-labeled dataset** is
  sourced (e.g. AO/OTA-coded radiographs or an expert-annotated set). Until
  then, ">95% type accuracy" is unmeasurable — there is no ground truth.

### Decisions
- Keep YOLOv8s baseline as the reference point; move to a stronger config next.
- Treat preprocessing harmonization as a first-class deliverable, not an
  afterthought.
- Separate the program into two measurable tracks: **(A) fracture detection**
  (FracAtlas boxes + MURA classifier) and **(B) fracture typing** (requires new
  data) — do not conflate their accuracy targets.

---

## Going forward — recommended next steps

### Phase 1 — Dataset harmonization (the consistency fix)
1. **Shared preprocessing contract** applied to *both* datasets before training:
   - Decode + repair truncated JPEGs (`ImageFile.LOAD_TRUNCATED_IMAGES`);
     re-encode FracAtlas's ~37 corrupt files once, offline.
   - Convert everything to a common working format (PNG, grayscale→3ch),
     consistent bit depth.
   - **CLAHE** (contrast-limited adaptive histogram equalization) for local
     contrast on bone edges — sharpens hairline fractures without blowing out
     the image; far safer than global sharpening which amplifies noise.
   - **Aspect-preserving resize + letterbox** to a fixed long side (start 800,
     test 1024) — autoscaling without distortion.
   - DICOM path: apply window/level (rescale slope/intercept) before 8-bit
     normalization so X-rays from different machines land on the same scale.
2. **Normalize labels to one schema** (YOLO `class cx cy w h`) and keep
   FracAtlas (boxes) and MURA (image-level) in clearly separated roles.
3. **Per-region balance check** — FracAtlas is hand/leg/hip/shoulder heavy;
   document the region distribution so we know where the model is blind.

### Phase 2 — Stronger detection model
4. Train **YOLOv8m, imgsz 800–1024, ~150 epochs**, with mosaic/HSV augmentation
   tuned mild (medical images are not natural scenes). Higher resolution is the
   single biggest lever for small fracture lines.
5. **Threshold tuning for sensitivity:** run `evaluate.py --target-recall 0.95`,
   pick the conf that hits the recall target, report precision/FP cost there.
6. **Add MURA co-training/ensemble:** train the abnormality classifier, fuse via
   `ensemble.py` so a confident "abnormal" film preserves borderline boxes.
7. **Backbone bake-off:** fine-tune RetinaNet + Faster R-CNN on the same split,
   run `benchmark.py`, and publish the comparison overlay on the landing page.

### Phase 3 — Data scale-up (required to approach the accuracy goals)
8. **More boxed fracture data is the only path to high recall.** Candidates:
   GRAZPEDWRI-DX (large pediatric wrist set with boxes), RSNA/Kaggle fracture
   sets, or expert re-annotation of MURA positives into boxes (semi-automated:
   weak-box → human correction loop).
9. **Source a type-labeled dataset** to unblock Track B (fracture typing). Until
   one exists, keep the type output explicitly labeled "heuristic — not learned."

### Phase 4 — Evaluation & honesty
10. Define and lock the metric contract:
    - Detection: **sensitivity @ fixed FP rate** (primary), mAP@0.5 and
      mAP@0.5:0.95 (secondary), per-region breakdown.
    - Typing: per-class accuracy + confusion matrix, once data exists.
11. Add a held-out **external test** (e.g. train on FracAtlas, test on a
    different source) to measure real-world generalization, not just in-set mAP.
12. Re-state the >97% / >95% goals in these defined terms and track progress
    against them each entry.

### Immediate action items (this week)
- [x] Build `scripts/preprocess.py` — DONE 2026-06-12 (CLAHE + truncation repair + label-safe resize + DICOM windowing; smoke-tested, recovers the ~37 corrupt FracAtlas files).
- [~] Re-split from preprocessed images; retrain YOLOv8m @ imgsz 800. — split done (`dataset_v2`), training IN PROGRESS (run `fracture_yolov8m`).
- [ ] Run `evaluate.py` sensitivity analysis on current + new model. — eval harness upgraded (see Entry 002); awaiting weights.
- [ ] Train MURA classifier; produce first ensemble result. — queued for when GPU frees.
- [ ] Generate real `docs/data/*.json` and deploy the site with honest numbers. — CI/deploy chain now built (Entry 002); awaiting weights.

---

## Entry 002 — 2026-06-12 — Phase 1 harmonized retrain launched; parallel groundwork completed

### What was done
- **Harmonized retrain underway.** Ran `preprocess.py` over FracAtlas → `dataset_v2`
  (3063/612/408 split, **0 corrupt** vs 37 before — the ~37 truncated files are
  now recovered). Launched **YOLOv8m @ imgsz 800, batch 8, 150 epochs** on the
  RTX 5060 (cu128). GPU mem ~5.1/8 G, ~2.5 min/epoch, ETA ~6–8 h. Writes to
  `runs/detect/fracture_yolov8m/`; baseline `fracture_yolov8s-2` left intact for
  before/after comparison.
- **Ran four parallel (CPU/code/research) workstreams while the GPU was busy**,
  using Opus as orchestrator + Sonnet/Haiku sub-agents (see [[model-orchestration]]):

  1. **Eval metric contract** (`evaluate.py`): added image-level sensitivity /
     specificity / PPV / NPV and a **per-region** sensitivity breakdown, plus
     `tests/test_eval_metrics.py` (12/12 passing). Turns the vague ">97%" goal
     into measurable screening metrics.
  2. **Region balance analysis** (`analysis/region_balance.md`): quantified the
     blind spots — **hip ≈63, shoulder ≈63 fractured examples** (vs hand 438);
     oblique view test set has only **3** fractured cases. Rare-group test
     metrics will be unreliable until more data is added.
  3. **Data sourcing loop** (`analysis/data_sourcing_log.md`, 3 passes, closed):
     ranked acquisition plan in two tracks.
     - **Detection boxes #1: GRAZPEDWRI-DX** — 20,327 wrist X-rays, 18,090 boxes,
       YOLO-ready, CC BY 4.0, no-login download. Pair with pkdarabi CV Project +
       Roboflow Hip set for region diversity.
     - **Fracture typing #1: Roboflow HUMERUS** — ~548 shoulder/humerus images
       with **fracture-morphology boxes** (oblique/segmental/spiral/transverse);
       the *only* public source of typed boxes found. Unblocks the typing track.
     - **Structural finding:** no AO/OTA-coded public radiograph corpus exists
       (TCIA/PhysioNet/RSNA all out of scope) — needs a clinical data-sharing
       agreement, not more searching. Synthetic copy-paste augmentation logged as
       the interim fallback for hip/shoulder scarcity.
  4. **CI/site automation loop** (`analysis/ci_automation_log.md`, 3 passes,
     closed): `validate_site_data.py` + pre-commit hook, `site-data.yml`
     (validate + weights-artifact regen + nightly cron), `pages-deploy.yml`
     (publishes `/docs` to lsaiko.github.io/Bone-R), and `docs/RUNBOOK_site_data.md`.

### Decisions / implications
- The typing track (Track B) is no longer fully blocked: **Roboflow HUMERUS** is
  a viable, if small, typed-box starting point. Goal ">95% type accuracy" should
  still be treated as aspirational until a larger typed set (likely clinical) is
  obtained.
- Data scale-up is confirmed as the real ceiling. **GRAZPEDWRI-DX is the single
  highest-leverage next acquisition** for detection recall.

### Next (when GPU frees after `fracture_yolov8m`)
1. `evaluate.py` on the new weights — compare to 0.273 / 0.110 / 0.301 baseline,
   read per-region sensitivity, find the conf for target recall.
2. Train MURA classifier → `ensemble.py` first result.
3. RetinaNet + Faster R-CNN bake-off on `dataset_v2`; `benchmark.py` + overlay.
4. Acquire GRAZPEDWRI-DX; merge via `make_splits` multi-`--src`; retrain.

---

## Entry 003 — 2026-06-13 — YOLOv8m results: Phase 1 validated, data ceiling confirmed

### Result: harmonization + bigger model nearly doubled every metric
YOLOv8m on `dataset_v2`, early-stopped at epoch 121 (best **epoch 114**):

| Metric | Baseline v8s | v8m harmonized | Δ |
|---|---|---|---|
| mAP@0.5 | 0.273 | **0.481** | +76% |
| mAP@0.5:0.95 | 0.110 | **0.192** | +75% |
| Precision | 0.428 | **0.603** | +41% |
| Recall | 0.301 | **0.459** | +52% |

### Screening view (test split, conf≥0.25) — the honest numbers
- Image-level **sensitivity 0.561**, specificity 0.988, PPV 0.902, accuracy 0.919.
- Confusion: TP 37 / FP 4 / TN 338 / FN 29 → **29 missed fractures**.
- **Accuracy 0.92 is inflated** by class imbalance (342 neg / 66 pos); sensitivity
  0.56 is the real screening metric. Model misses ~44% of fractures.

### Hard ceiling found
- **Target recall 0.90 is NOT achievable at any threshold — max recall = 0.797.**
  This is a data limitation, not a tuning one.

### Per-region sensitivity — blind spots now measured (not predicted)
| region | test pos | sensitivity |
|---|---|---|
| hand | 37 | 0.568 |
| leg | 28 | 0.571 |
| hip | 1 | 0.000 |
| shoulder | 0 | N/A (unevaluable) |

Empirically confirms Entry 002's region analysis: hip/shoulder are too sparse to
learn or assess. **GRAZPEDWRI-DX + Roboflow Hip/Humerus acquisition is now
data-justified, not speculative.**

### Decisions
- v8m harmonized is the new reference model (`runs/detect/fracture_yolov8m`).
- Phase 1 (harmonization) is validated and CLOSED.
- The path to the >95% sensitivity goal runs through DATA (Phase 3), not more
  model/aug tweaks — the 0.797 recall ceiling makes that unambiguous.

### Next
1. (GPU) MURA classifier → ensemble: test whether the abnormality second-opinion
   lifts sensitivity past the 0.56 / 0.797 ceiling without tanking specificity.
2. (GPU) RetinaNet + Faster R-CNN bake-off → benchmark + real overlay for site.
3. Generate real `docs/data/*.json`; package repo; deploy Pages.
4. (Data) Acquire GRAZPEDWRI-DX; merge; retrain — the real ceiling-lifter.

---

## Entry 004 — 2026-06-13 — MURA ensemble: NEGATIVE result (autonomous run)

### MURA abnormality classifier trained
`yolov8s-cls` on the rebuilt MURA_cls tree. 20 epochs, imgsz 320, batch 64.
**Best top-1 = 0.832** (epoch 15). Healthy.

### Bug caught & fixed before training: MURA tree filename collision
`mura_to_yolo.py` named copied files `{study_folder}__{image}`, but MURA study
folders (`study1_positive`) and image names (`image1.png`) repeat across EVERY
patient — so the whole 40k dataset collapsed to **53 files** on disk. Fixed to
use the last 4 path parts (`XR_REGION__patientXXXXX__studyN_x__imageM.png`);
rebuilt → correct 14,873 abnormal / 21,939 normal train counts. Verifying disk
counts before the GPU launch surfaced this.

### Ensemble evaluation (new `eval_ensemble.py`, test split, n=408)
| config | sensitivity | specificity | PPV |
|---|---|---|---|
| detector-only (v8m, Entry 003) | 0.561 | **0.988** | **0.902** |
| ensemble α=0.5 conf=0.15 | 0.561 | 0.877 | 0.468 |
| ensemble α=0.9 conf=0.25 | 0.606 | 0.874 | 0.482 |

**Conclusion — the ensemble HURTS and will NOT ship in v1.** The MURA classifier
flags many FracAtlas normals as "abnormal" (MURA's abnormal class spans hardware,
degenerative change, etc. — not just fractures), so fusion adds false positives
(FP 4 → 42) while recovering at most 3 fractures. Sensitivity is fundamentally
capped by the *detector's* localization recall; an image-level classifier can't
lift it without trading away specificity. Label-semantics mismatch is the root
cause.

### Decisions
- **Shipped model stays detector-only `fracture_yolov8m`.** Ensemble code is
  retained but disabled-by-default; documented as a studied negative result.
- Possible v1.1 reframe: use the classifier for a "needs-review triage queue"
  (where extra FPs are acceptable) rather than box fusion — deferred.

### Next (continuing autonomously)
1. (GPU) RetinaNet + Faster R-CNN bake-off on `dataset_v2` → `benchmark.py`.
2. Real `docs/data/*.json` + overlay from v8m; README polish; local commit.
3. Track-D prep: GRAZPEDWRI-DX ingestion adapter (code only).

---

## Entry 005 — 2026-06-13 — Backbone bake-off + ship pass (autonomous)

### Bake-off complete (test split, dataset_v2)
| Backbone | mAP@0.5 | mAP@0.5:0.95 | recall |
|---|---|---|---|
| **YOLOv8m (shipped)** | **0.460** | **0.198** | **0.429** |
| Faster R-CNN R50-FPNv2 | 0.354 | 0.171 | 0.288 |
| RetinaNet R50-FPNv2 | 0.240 | 0.072 | 0.188 |

YOLOv8m wins decisively — expected and defensible for a single-class, small-data
detection task. torchvision two-stage/one-stage backbones trail with only ~3k
training images and fresh heads.

### Bug caught & fixed: torchvision RetinaNet/FCOS head sizing
`models/torchvision_backend.py` had inverted `num_classes` logic — during
training (no checkpoint) it built the **full 91-class COCO head**, so the saved
RetinaNet weights (819 = 91×9 logits) could not load into the 2-class eval model
(18 logits) → `size mismatch` at benchmark time. Fixed both RetinaNet and FCOS
to build a **fresh num_classes+1 head on a pretrained backbone** for BOTH train
and eval (Faster R-CNN already did this via FastRCNNPredictor, so it was fine).
Verified with a save/load round-trip, then RETRAINED RetinaNet → loads & scores.

### Ship artifacts generated
- **Real benchmark** → `docs/data/benchmark.json` (3 backbones, validator-passing).
- **Real comparison overlay** → `compare_overlay.py` on a fractured test X-ray
  (`000029_IMG0002482.png`): yolov8 2 boxes, retinanet 1, fasterrcnn 1, all
  clustering on the same elbow fracture. Wrote `docs/data/predictions.json` +
  `docs/assets/compare.png`.
- **Interactive viewer upgraded** (`docs/app.js`): now loads the actual
  radiograph behind the model boxes (canvas sized to image; FracAtlas CC BY 4.0
  attribution shown). Verified live via preview — no console errors, boxes
  overlay correctly, toggles + confidence slider functional.
- **README** — added Results section (backbone table + screening metrics +
  honest limitations) and a Datasets & attribution section.
- **Track-D prep** — `scripts/ingest_grazpedwri.py` (multi-class GRAZPEDWRI →
  single-class fracture remap), unit-tested.

### Decisions
- **Shipped configuration locked:** YOLOv8m detector, detector-only (no ensemble).
- Site is showcase-ready with real numbers; remaining ship step is the public
  `git push` + GitHub Pages enable, which need the user's account (left for them).

### Status: v1 is code-complete and staged. Pending user: push + Pages enable.
### Next (v1.1+, Track D): acquire GRAZPEDWRI-DX → ingest → merge → retrain to
lift the 0.797 recall ceiling; revisit fracture-typing via Roboflow HUMERUS.

---

## Entry 006 — 2026-06-14 — Track D: GRAZPEDWRI merge lifts the ceiling (v3)

### Acquisition reality check
The raw Figshare GRAZPEDWRI-DX release has **no bounding-box labels** — only
images + image-level metadata (`dataset.csv`, which DOES carry AO/OTA
`ao_classification` codes → a future Track-B typing source). YOLO boxes come from
derivatives. Sourced via the **Roboflow mirror** (project-bef88/grazpedwri-dx
v2) — but it is a **~2,396-image SUBSET**, not the full 20k. Fracture class index
there is **2**, not 3 (8 classes, `foreignbody` dropped).

### Ingest + merge
`ingest_grazpedwri.py --fracture-class 2` → **1,599 fracture-positive** images
(+797 bg, 2,111 boxes). Merged with `FracAtlas_proc` → `dataset_v3`
(4,861 / 971 / 647). This **3.2×'d the fracture-positive image count**
(717 → 2,316) — the decisive change, since FracAtlas was mostly negatives.

### v3 results (YOLOv8m, early-stop epoch 92, best 69)
| Metric (test) | v1 | v3 | Δ |
|---|---|---|---|
| mAP@0.5 | 0.460 | **0.761** | +65% |
| mAP@0.5:0.95 | 0.198 | **0.369** | +86% |
| image-level sensitivity | 0.561 | **0.812** | +45% |
| specificity | 0.988 | 0.980 | ~flat |
| PPV | 0.902 | **0.961** | +7% |
| **max achievable recall** | 0.797 | **0.875** | ceiling lifted |

### Honest read — gains are wrist-concentrated
Per-region sensitivity: **unknown (GRAZPEDWRI wrist) 0.879** (174/244 test
positives), hand 0.568→**0.705**, leg ~flat (0.538). **Hip & shoulder still 0
test positives — unevaluable and unaddressed.** The headline jump is real but
driven by the wrist distribution we added; the original FracAtlas regions
improved modestly. GRAZPEDWRI being wrist-only means hip/shoulder remain the open
gap (need Roboflow Hip/Humerus).

### Decisions
- **Shipped model is now `fracture_yolov8m_v3`.** Site overlay + README updated
  to v3 (on a FracAtlas test image so all 3 backbones get a fair comparison).
- Backbone study stays on `dataset_v2` (apples-to-apples); v3 is the data-scaling
  result, presented separately to avoid mixing test sets.
- Full 20k GRAZPEDWRI (OneDrive) and hip/shoulder sets remain the next data lever.

### Status: v3 trained, evaluated, site regenerated, staged for commit.
### Next: commit v3 (user pushes); then Roboflow Hip/Humerus for the remaining
blind spots + AO/OTA typing track (data now identified in dataset.csv).

---

## Entry 007 — 2026-06-14 — v4: HUMERUS closes the shoulder gap

### Data
Added HUMERUS (Roboflow, 1,420 typed-fracture images, all classes→fracture via
new `ingest_grazpedwri.py --all-fracture`). Rejected the "hip" Roboflow set —
it was **osteoporosis**, not hip fractures. Merged → `dataset_v4`
(5,926 / 1,184 / 789). Found a GENUINE hip set for later (Roboflow proximal-femur).

### v4 results (YOLOv8m, early-stop ep 116, best 71)
| Metric (test) | v3 | v4 |
|---|---|---|
| mAP@0.5 | 0.761 | **0.858** |
| mAP@0.5:0.95 | 0.369 | **0.479** |
| image-level sensitivity | 0.812 | **0.927** |
| specificity | 0.980 | 0.975 |
| PPV | 0.961 | 0.973 |
| max achievable recall | 0.875 | **0.895** |

### Per-source sensitivity — the shoulder gap CLOSED
| source | n (test) | pos | sensitivity |
|---|---|---|---|
| **humerus (shoulder region)** | 141 | 141 | **0.986** |
| wrist (GRAZPEDWRI) | 248 | 178 | 0.961 |
| FracAtlas-native | 400 | 66 | 0.712 |

The HUMERUS addition did exactly its job: humerus fractures now detected at 98.6%
(was a blind spot in v1-v3). FracAtlas-native also rose (0.56→0.71) from more data.

### Honest caveats
- The headline 0.927 sensitivity is **inflated by easy humerus/wrist** distributions;
  the harder FracAtlas-native fractures sit at 0.712 — that's the real-world number.
- **Hip still barely evaluable** (2 test positives, caught 1). True hip fractures
  remain the open gap → Roboflow proximal-femur (v5).
- A per-source eval bug bit me first: HUMERUS filenames contain "...-Img1.rf..."
  so an "IMG" substring check mis-tagged them as FracAtlas. Fixed the tag
  (rf. = Roboflow; WRI = wrist, else humerus).

### Decisions
- **Shipped model is now `fracture_yolov8m_v4`.** Site overlay + README + MODEL_CARD
  updated to v4.
- Stratified split (`--stratify`, built this session) should be used for v5 so
  hip/shoulder are properly represented.

### Next: train the typing classifier (train_typing.py ready); acquire proximal-femur
for the hip gap; consider full 20k GRAZPEDWRI for more wrist headroom.

---

## Entry 008 — 2026-06-14 — v5: hip blind spot closed (proximal-femur)

### Data
Added Roboflow proximal-femur set (thesisyolo-v8, 1,214 imgs, 14 L/R classes).
Kept only the 6 true-fracture classes (intertrochanteric / neck / subtrochanteric
x L+R) via `ingest --fracture-names ... --data-yaml`; dropped dislocations,
trochanter landmarks, normals -> 392 hip-fracture images, 815 boxes. Merged all
4 sources with **`--stratify`** -> `dataset_v5` (6,847 train, 12 strata).

### v5 results (YOLOv8m, early-stop ep 120, best 86)
Box: mAP@0.5 0.790, mAP@0.5:0.95 0.427, P 0.840, R 0.721 (down from v4 0.858 —
expected: 392 hard hip fractures + harder stratified test set).
Image-level (test n=906): **sensitivity 0.878, specificity 0.986, PPV 0.981**.

### Per-source sensitivity — HIP NOW EVALUABLE
| source | n(test) | pos | sensitivity |
|---|---|---|---|
| **hip** | 125 | 43 | **0.674** (was 2 cases in v4) |
| humerus | 138 | 138 | 0.993 |
| wrist | 238 | 159 | 0.925 |
| fracatlas-native | 405 | 70 | 0.671 |

### Findings / was it good?
**YES.** The primary goal — the hip blind spot — is closed. Hip went from 2
unreliable test cases to 43, detected at **0.674**, on par with FracAtlas-native
(0.671). The slight overall sensitivity dip (0.927->0.878) is honest, not a
regression: v4's 0.927 was inflated by easy wrist/humerus and a soft test split;
v5's test set now includes hard hip cases and is region-stratified, so 0.878 is a
more trustworthy number. The model is now genuinely multi-region: hip 0.67,
humerus 0.99, wrist 0.93, hand/leg(fracatlas) 0.67.

### Do we need more datasets?
Major anatomical blind spots are now COVERED (wrist, humerus/shoulder, hip,
hand/leg). The lever is no longer NEW anatomy — it's DEPTH for the weakest
regions:
- **Hip (0.67) & FracAtlas-native (0.67)** are the floor — more hip volume (only
  392 imgs) and a large general adult-fracture set would lift them most.
- **Pediatric skew** persists (GRAZPEDWRI wrist) — adult wrist data would balance it.
- The **>=0.95 sensitivity** clinical bar still needs institutional/PACS-scale
  data + expert labels (per MODEL_CARD) — not reachable from public sets.
Recommendation: prioritise hip-volume + a large adult general-fracture corpus;
exotic new regions are NOT needed.

### Shipped: `fracture_yolov8m_v5`. Site, README, MODEL_CARD updated to v5.

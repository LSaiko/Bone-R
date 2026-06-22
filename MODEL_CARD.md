# Bone-R Fracture Detector — Model Card

**Current Version:** v5 (see Changelog; detailed metrics in JOURNAL.md Entries 003-008)  
**Last Updated:** 2026-06-14  
**Framework:** Ultralytics YOLOv8  
**Status:** Research/educational tool only. Not a medical device.

---

## Model Details

### Architecture
- **Backbone:** YOLOv8m (medium), single-class object detection
- **Input:** X-ray images (preprocessed: CLAHE contrast enhancement, 3-channel RGB, aspect-preserving resize to ≤1024px)
- **Output:** Bounding boxes with confidence scores; heuristic fracture type/severity inferred from box geometry
- **Task:** Localized fracture detection on 2-D radiographs (frontal, lateral, oblique views)

### Version History
| Version | Data Source | Train Set | Test Set | mAP@0.5 | Sensitivity | Specificity | Max Recall | Notes |
|---------|---|---|---|---|---|---|---|---|
| v1-baseline | FracAtlas only | 3,063 | 408 | 0.27 | — | — | — | YOLOv8s, baseline for comparison |
| **v1** | FracAtlas (harmonized) | 3,063 | 408 | **0.46** | **0.561** | **0.988** | **0.797** | YOLOv8m, preprocessing applied (CLAHE, truncation repair, consistent resize) |
| **v3** | FracAtlas + GRAZPEDWRI-DX | 4,861 | 647 | **0.76** | **0.812** | 0.980 | **0.875** | Data scale-up (3.2×'d fracture-positive count); wrist-concentrated gains |

**v4 (in training):** Incorporates Roboflow HUMERUS (shoulder) dataset; metrics pending.

### Training Configuration (v1 & v3)
- **Optimizer:** SGD (YOLOv8 default)
- **Epochs:** 150 (v1), early-stopped at epoch 92/121 (v3 best epoch 69)
- **Batch size:** 8
- **Image size:** 800px (v1), 640px (v3)
- **Augmentation:** YOLOv8 defaults (mosaic, HSV jitter) with mild tuning for medical images
- **Hardware:** NVIDIA RTX 5060 (Blackwell, cu128), ~2.5 min/epoch
- **Loss:** YOLOv8 multi-task (box, objectness, classification)

---

## Intended Use

### In-Scope (Research / Educational)
- **Research triage aid:** Flag possible fractures in X-ray cohorts for human review
- **Educational demonstration:** Model behavior on public radiographic datasets
- **Baseline for comparison:** Benchmark point for fracture-detection algorithm development
- **Clinician awareness tool:** *Intended as a second opinion only* — highlight regions for radiologist attention, not as a diagnostic classifier

### Explicitly Out-of-Scope / Prohibited
- ❌ **NOT a medical device.** Not intended for independent diagnosis, treatment decisions, or clinical deployment
- ❌ **NOT autonomous clinical decision support.** Outputs must be reviewed by a qualified radiologist
- ❌ **NOT validated for real-world clinical use.** External validation WAS run (pkdarabi, unseen source) and showed **poor generalization — sensitivity 0.30 vs 0.67 in-distribution** (see Changelog Pass 4). The model is an in-domain demo, not deployable.
- ❌ **NOT for fracture typing.** Type/severity outputs are heuristic-only (box aspect ratio + area), not learned classifiers
- ❌ **NOT for trauma triage.** Does not rank fracture priority or urgency
- ❌ **NOT for prognosis or follow-up.** Cannot predict healing outcomes or complications

**Human-in-the-loop is mandatory.** Always require a qualified orthopedic consultant to review model outputs before any clinical or treatment decision.

---

## Training Data

### Data Sources (v3)

#### FracAtlas (primary)
- **Source:** Figshare, CC BY 4.0 (Abedeen et al., *FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal Radiographs*, Sci. Data 2023)
- **Composition:** 4,083 X-ray images from 3 hospitals in Bangladesh; 717 fractured (17.56%), 3,366 non-fractured
- **Annotation:** COCO-format bounding boxes + segmentation masks per fracture instance (box-level)
- **Body regions:** Hand (438 fractured), leg (263 fractured), hip (63 fractured), shoulder (63 fractured), mixed (106 fractured)
- **Views:** Frontal, lateral, oblique (418 oblique images, 10.77% fracture rate)
- **Inclusion in v3:** Full dataset after preprocessing (truncation repair, CLAHE, consistent resize)

#### GRAZPEDWRI-DX (v3 data scale-up)
- **Source:** Figshare + Roboflow mirror, CC BY 4.0 (Lässer et al., *GRAZPEDWRI-DX: A Pediatric Wrist X-ray Dataset*, Nature Sci. Data 2022)
- **Composition:** ~2,396 images (Roboflow subset) with bounding boxes; 1,599 fracture-positive, 797 background
- **Annotation:** Multi-class boxes (fracture class index 2 in Roboflow); merged to single-class detection
- **Body region:** **Wrist only** — pediatric cohort
- **License:** CC BY 4.0
- **Impact on v3:** +2,111 boxes (+3.2×'d fracture-positive image count from 717 → 2,316); drove wrist-region sensitivity gains

#### MURA (rejected)
- **Status:** **Not shipped.** Tested in v1 ensemble experiment (Entry 004, JOURNAL.md) but **rejected** due to label-semantics mismatch
- **Why rejected:** MURA's "abnormal" class spans hardware artifacts, degenerative change, and fractures — not fracture-specific. Fusion with detector added false positives (4 → 42) while recovering ≤3 fractures. Documented as a studied negative result.
- **Lesson:** Image-level abnormality classification cannot lift detection recall without trading specificity. Not suitable for ensemble with fracture boxes.

### Preprocessing
Before training, all images pass through a unified harmonization pipeline:
1. **JPEG repair:** Recovers truncated/corrupt files (~37 FracAtlas images recovered)
2. **CLAHE (Contrast-Limited Adaptive Histogram Equalization):** Enhances local contrast on bone edges; sharpens hairline fractures without noise amplification
3. **Grayscale → RGB:** Consistent 3-channel representation
4. **DICOM windowing (if applicable):** Applies radiographic rescale (slope/intercept) for consistent intensity across scanner modalities
5. **Aspect-preserving resize:** Scales to ≤1024px long side without distortion; YOLO labels remain valid

---

## Performance & Metrics

### Primary Metrics (Image-Level Screening)

**v1 (FracAtlas only, test split n=408):**
- Sensitivity: 0.561 (37/66 fractured correctly identified)
- Specificity: 0.988 (338/342 non-fractured correctly rejected)
- PPV (Positive Predictive Value): 0.902 (37/41 positive calls are true positives)
- Accuracy: 0.919 **(inflated by class imbalance; sensitivity is the honest metric)**
- Confusion: TP 37 / FP 4 / TN 338 / FN 29 → **29 missed fractures at conf ≥0.25**

**v3 (FracAtlas + GRAZPEDWRI, test split n=647):**
- **Sensitivity: 0.812** (198/244 fractured correctly identified; +45% vs v1)
- Specificity: 0.980 (395/403 non-fractured correctly rejected; ~flat vs v1)
- **PPV: 0.961** (198/206 positive calls are true positives; +7% vs v1)
- Confusion: TP 198 / FP 8 / TN 395 / FN 46
- **Honest reading:** The +0.251 sensitivity gain is **largely wrist-driven** (GRAZPEDWRI is wrist-only, +3.2×'d wrist-positive examples)

### Localization Metrics (Box-Level)
| Metric | v1 | v3 | Δ |
|--------|---|---|---|
| **mAP@0.5** | 0.460 | **0.761** | +65% |
| **mAP@0.5:0.95** | 0.198 | **0.369** | +86% |
| Precision (test) | 0.603 | — | — |
| Recall (test) | 0.459 | — | — |

### Hard Ceiling
- **Max achievable recall:** v1 = 0.797; v3 = 0.875
- **≥0.90 recall is NOT achievable** on current data without region-diverse box expansion
- **Interpretation:** The recall ceiling is a **data limitation, not a tuning one.** FracAtlas alone is too small (~717 fractured images); GRAZPEDWRI helped but remains wrist-concentrated.

### Per-Region Sensitivity (v1 test split, n=408)
| Region | Test Positives | Sensitivity |
|--------|---|---|
| Hand | 37 | 0.568 |
| Leg | 28 | 0.571 |
| Hip | 5 | 0.000 |
| Shoulder | 0 | N/A (unevaluable) |

**v3 per-region (n=647, test split):**
| Region | Test Positives | Sensitivity |
|---|---|---|
| Wrist (GRAZPEDWRI) | 244 | **0.879** |
| Hand (FracAtlas) | — | 0.705 (estimated, not re-measured) |
| Leg (FracAtlas) | — | 0.538 (estimated, not re-measured) |
| Hip | 0 | N/A (still unevaluable) |
| Shoulder | 0 | N/A (still unevaluable) |

---

## Limitations & Biases

### Data-Driven Limitations

1. **Small, imbalanced dataset:** FracAtlas (717 fractured images) is too small for high-confidence detection. v3's gains came from data scale-up (GRAZPEDWRI), not model architecture improvements — this is the real ceiling-lifter.

2. **Wrist-concentrated gains:** GRAZPEDWRI-DX is **wrist-only** and **pediatric-skewed.** The v1→v3 jump (+0.251 sensitivity) is almost entirely wrist-driven (0.568 → 0.879 wrist sensitivity). Hand, leg, and the remaining anatomical regions improved minimally (hand 0.568 → 0.705; leg ~0.571 → 0.538). This is not a general improvement — it is a region-specific boost.

3. **Hip and shoulder remain blind spots:**
   - **Hip:** 1 fractured test case in v1 (0 correct), 0 in v3 → unevaluable. FracAtlas has only 63 hip fractured examples (vs 438 hand); insufficient training signal.
   - **Shoulder:** 0 fractured test cases in v1 and v3 → cannot assess performance; only 63 fractured examples in training.
   - **Verdict:** These regions **need targeted data acquisition** (Roboflow HUMERUS for shoulder, Proximal Femur Fracture for hip), not more wrist images.

4. **Oblique views undersampled:** Only 45/418 oblique-view images in FracAtlas are fractured (10.77% fracture rate). Test set has only 3 fractured oblique cases. Model will struggle on challenging oblique-angle fractures.

5. **Single-class detection only:** The model classifies "fracture" vs. "not fracture" at the image level. It does **not** distinguish fracture type (transverse, oblique, comminuted, greenstick, etc.). Fracture morphology is inferred from box geometry (aspect ratio + area) — a heuristic, not a learned classifier.

### Architectural Limitations

6. **No fracture typing learned:** Current "fracture type" and "severity" outputs in inference are rule-based hints:
   - **Type:** Derived from box aspect ratio (tall → avulsion, wide → transverse, etc.)
   - **Severity:** Heuristic from box area + model confidence
   - **Status:** NOT validated. A proper typed detector requires a typed-labeled dataset (e.g., Roboflow HUMERUS with oblique/segmental/spiral/transverse labels, or AO/OTA-coded radiographs). None exists publicly yet.

7. **No confidence calibration:** Raw model confidences are not calibrated to represent true probability of fracture presence. A prediction @ 0.80 confidence does not mean "80% chance of fracture."

### Deployment & Generalization Risks

8. **No external validation:** Metrics are on held-out test splits from the same datasets used for training. No independent cohort from a different hospital/country/scanner was evaluated. Real-world performance is unknown.

9. **Scanner modality bias:** FracAtlas and GRAZPEDWRI are conventional 2-D X-rays (plain radiographs). Performance on digital radiography (DR), computed radiography (CR), or other modalities is untested.

10. **Label inconsistencies documented:** FracAtlas region_balance.md notes inconsistent fracture rates by region (hand 28.48%, leg 11.57%, hip 18.64%); this suggests potential labeling variance, especially in leg and hip cohorts.

11. **Recall ceiling is hard:** 0.875 max recall means **≥12.5% of fractured images will be missed** even with perfect thresholding. Reaching 0.95+ recall requires fundamentally more training data (500+ region-diverse boxed examples per underrepresented region).

---

## Ethical Considerations

### Automation Bias Risk
Clinicians shown model-generated fracture flags may over-trust the suggestion, especially when the model is confident. **Mitigation:** Always display confidence scores alongside predictions and include a disclaimer that boxes are "best guess" estimates requiring radiologist review.

### Asymmetric Cost of Errors
- **False Negative (missed fracture):** Can delay treatment, worsen patient outcomes, increase liability.
- **False Positive (spurious fracture flag):** Wasteful additional imaging/follow-up, patient anxiety, but reversible with radiologist review.

**The cost of a miss is higher.** A 0.812 sensitivity model will miss ~19% of fractures in a screened cohort. This is **not acceptable for autonomous deployment** but acceptable for a research triage aid flagging cases for radiologist attention.

### Human-in-the-Loop Requirement
Fracture detection has irreducible uncertainty — even radiologists disagree on subtle or borderline cases. **No ML model should make or justify a fracture diagnosis without expert review.** This is not a limitation of this model specifically; it is fundamental to the task.

### Pediatric Cohort Note
GRAZPEDWRI-DX is **pediatric-skewed.** Fracture morphology, bone density, and ligament injuries differ between children and adults. Applying this model to adult cohorts without external validation introduces unknown bias.

---

## Caveats & Recommendations

### Before Any Real-World Use

1. **External validation is mandatory.**
   - Train on FracAtlas + GRAZPEDWRI (or v3 weights).
   - Evaluate on a held-out cohort from a **different hospital, country, or scanner modality** (not a random split of the same dataset).
   - Report per-region sensitivity/specificity on that external cohort.
   - If external performance drops >5%, do not deploy; investigate the distribution shift.

2. **Clinical gold standard:** Validate against independent expert radiologist reads (ideally 2+ blinded readers with consensus) on the external cohort. The model's 0.961 PPV is only meaningful if ground truth is accurate.

3. **Regulatory pathway:** If clinical deployment is intended, engage with a regulatory body (FDA if US-based) early. This model is **not a medical device as-is** and would require validation in a clinical study, 510(k) or PMA process, depending on risk classification.

### Data Gaps (Next Steps)

4. **More region-diverse boxed data.** Priorities (from data_sourcing_log.md):
   - **A-1: GRAZPEDWRI-DX** — already acquired for v3
   - **A-2: pkdarabi Bone Fracture Detection CV Project** — 4,148 YOLO-ready images, multi-region including shoulder
   - **A-3: Roboflow HUMERUS** — ~548 shoulder images with fracture-type classes; *in v4 training*
   - **A-4: Roboflow Proximal Femur Fracture (ThesisYolo)** — ~640 hip segmentation masks convertible to boxes
   - **A-5+: Copy-paste augmentation** — synthetic expansion of hip/shoulder crops on normal backgrounds

5. **Fracture typing requires a new labeled dataset.** No public fracture-type-labeled corpus exists yet. Options:
   - **Roboflow HUMERUS** (4 classes: oblique, segmental, spiral, transverse) — starting point for shoulder typing
   - **pkdarabi Bone Break Classification** (Kaggle, 10 classes, image-level only) — useful for classification head but not detection
   - **AO/OTA-coded radiographs** — ideal but **NOT publicly available**; requires clinical data-sharing agreement

6. **Hip fracture gap remains open.** FracAtlas hip subset (<200 boxes) + Roboflow Proximal Femur Fracture (~640 masks) + copy-paste augmentation can partially close it, but clinical-grade hip detection (1000+ expert-annotated examples) requires institutional partnership.

### Evaluation Discipline

7. **Lock and report metrics per test split.** When v4/v5 are trained, include:
   - Per-region sensitivity/specificity (not just mAP)
   - Confidence calibration (e.g., Expected Calibration Error)
   - Stratified analysis (by anatomical region, view angle, age group if available)
   - Per-region confusion matrices

8. **Do NOT cherry-pick test cases.** Always report on the full held-out split; do not exclude hard negatives or visually unclear cases to inflate metrics.

---

## Model Access & Weights

- **v1 weights:** `runs/detect/fracture_yolov8m/weights/best.pt` (FracAtlas-only baseline)
- **v3 weights:** `runs/detect/fracture_yolov8m_v3/weights/best.pt` (FracAtlas + GRAZPEDWRI, shipped)
- **v4 weights:** In training (incorporates Roboflow HUMERUS); metrics pending

All weights are released under the same CC BY 4.0 license as the training data (FracAtlas + GRAZPEDWRI).

---

## Comparison to Published Work

On comparable public benchmarks:
- **FracAtlas-only detection (v1 mAP 0.46)** vs. published YOLO/CNN baselines: Competitive but not state-of-the-art. FracAtlas itself is a small dataset; published papers on MURA (40k abnormality examples) achieve higher mAP, but MURA lacks boxes.
- **v3 mAP@0.5 0.76** improves substantially over v1, but **gains are wrist-driven and may not generalize** to hand/leg/hip cohorts without additional region-diverse data.

No published work directly compares on this exact train/test split; external benchmark is pending.

---

## Disclaimer

This model outputs **probabilistic best-guess estimates**, not medical diagnoses. Fracture detection on radiographs is inherently uncertain — even radiologists disagree on subtle or borderline cases. **Always require a qualified orthopedic consultant to review this model's outputs before any clinical decision.** The authors assume no liability for incorrect predictions, missed diagnoses, or downstream patient harm.

---

## Changelog

### Pass 1 — 2026-06-14 (Initial Model Card)
- Covers v1 (FracAtlas-harmonized baseline, mAP@0.5 0.46, sensitivity 0.561) and v3 (FracAtlas + GRAZPEDWRI, mAP@0.5 0.76, sensitivity 0.812)
- Documents v4 (HUMERUS shoulder dataset) as in-training; metrics pending
- Fully grounds metrics in JOURNAL.md Entries 001–006
- Includes per-region sensitivity breakdown and honest limitations (wrist-concentrated gains, hip/shoulder blind spots)
- Notes MURA ensemble rejection as a documented negative result
- Recommends data sourcing priorities and external validation before deployment

### Pass 2 — 2026-06-14 (v4 shipped)
- **Shipped model is now v4** (`fracture_yolov8m_v4`): + HUMERUS (shoulder),
  `dataset_v4` (5,926/1,184/789).
- v4 test metrics: mAP@0.5 **0.858**, image-level sensitivity **0.927**,
  specificity 0.975, PPV 0.973, max achievable recall **0.895**.
- Per-source sensitivity: humerus/shoulder **0.986**, wrist 0.961,
  FracAtlas-native **0.712** (realistic figure; headline lifted by easier sets).
- Shoulder/humerus blind spot **closed**; **hip remains open** (2 test cases —
  prior "hip" dataset was osteoporosis; proximal-femur set queued for v5).
- Learned fracture-type classifier now scaffolded (`train_typing.py`) but untrained.

**Next: train the type classifier; acquire proximal-femur (hip); external validation.**

### Pass 3 — 2026-06-14 (v5 shipped)
- **Shipped model is now v5** (`fracture_yolov8m_v5`): + proximal-femur (hip),
  region-stratified `dataset_v5` (6,847 train).
- v5 test: mAP@0.5 0.790, image-level sensitivity 0.878, specificity 0.986, PPV 0.981.
- Per-source sensitivity: hip **0.674** (now evaluable — 43 test cases vs v4's 2),
  humerus 0.993, wrist 0.925, FracAtlas-native 0.671.
- **Hip blind spot closed.** Overall dip vs v4 (0.927->0.878) is honest: stratified,
  harder test set. All major anatomical regions now covered.
- Remaining: depth (more hip + adult general-fracture data); >=0.95 sensitivity
  still needs institutional/PACS data.

### Pass 4 — 2026-06-14 (external validation)
- **External validation added** (`scripts/external_val.py`): v5 tested on pkdarabi
  (1,129 imgs, never in detector training) -> **sensitivity 0.300 vs 0.671
  in-distribution** (~2.2x drop).
- **Critical limitation:** in-set metrics (overall 0.878) materially overstate
  real-world performance. The model generalizes POORLY out-of-distribution and is
  an in-domain demo only — NOT deployable. Closing the gap requires multi-source
  training + a standing held-out external test.
- Ruled out: train/serve CLAHE mismatch is negligible (raw 0.287 vs CLAHE 0.300).

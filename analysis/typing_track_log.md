# Fracture-Type Classification Track — Progress Log

**Purpose:** Append-only log tracking the build-out of fracture-TYPE classification,
a goal stated in the project README (">95% fracture-TYPE classification accuracy")
that was previously blocked by the absence of typed training data.

---

## Pass 1 — 2026-06-14

### What was built

**Data-prep script:** `scripts/build_typing_dataset.py`
- Reads HUMERUS_rf (Roboflow YOLOv8 export, train/valid/test splits).
- Assigns each image a single fracture-type label = class of its highest-area box
  (highest area chosen over first-row because Roboflow label insertion order is not
  importance order; largest box most reliably names the dominant fracture pattern).
- Class index → name mapping from `HUMERUS_rf/data.yaml`:
  `0=oblique, 1=segmental, 2=spiral, 3=transverse`
- Copies images into `typing_dataset/<split>/<classname>/<filename>` — full Roboflow
  augmented filenames retained (contain `rf.<hash>` suffix) to avoid any collision.

**Training script:** `train_typing.py`
- Ultralytics YOLOv8-cls fine-tune, mirrors `train_mura_cls.py` structure.
- Defaults: `--model yolov8s-cls.pt --epochs 40 --imgsz 320 --batch 32 --workers 0`.
- Confirmed: imports cleanly and `--help` works; NOT run (GPU busy with another job).

**Dataset built:** `typing_dataset/` — 1,420 images across train/val/test.

### Typed class distribution

| split   | oblique | segmental | spiral | transverse | total |
|---------|---------|-----------|--------|------------|-------|
| train   | 403     | 289       | 333    | 299        | 1,324 |
| val     | 16      | 16        | 16     | 16         | 64    |
| test    | 8       | 8         | 8      | 8          | 32    |
| **all** | **427** | **313**   | **357**| **323**    | **1,420** |

Grand-total percentages: oblique 30.1%, segmental 22.0%, spiral 25.1%, transverse 22.7%.

### Caveats

1. **Small dataset.** 1,324 training images is workable for fine-tuning a pretrained
   backbone but tight for 4-way classification; expect noisy accuracy estimates.
2. **Humerus-only.** All images are from a single anatomic region (HUMERUS_rf). A model
   trained here will not generalise to femur/tibia/radius fracture types without
   additional typed data.
3. **Val/test splits are tiny.** 64/32 images stratified evenly across 4 classes — only
   16 / 8 images per class. Accuracy numbers on these splits will have high variance.
4. **Mild class imbalance.** Oblique (30%) vs. segmental (22%) — a ~1.4× ratio.
   Not severe enough to require resampling for a first run, but worth monitoring per-class
   recall after training.
5. **GRAZPEDWRI `ao_classification` column not yet used.** `GRAZ_raw/dataset.csv` has
   AO/OTA codes which would provide a far larger (multi-regional) typing signal, but
   mapping AO codes → the 4 morphology classes requires a schema lookup and is deferred.

### Next pass

- [ ] **Add GRAZPEDWRI AO/OTA typing source.** Map `ao_classification` codes in
  `GRAZ_raw/dataset.csv` → {oblique, segmental, spiral, transverse} (e.g. AO 12-A3 →
  transverse). This would 5–10× the typed training set and add multi-region coverage.
- [ ] **Evaluate train_typing.py** once the current GPU job finishes — run with
  default flags, report per-class top-1 on the test split.
- [ ] **Address class imbalance** if per-class recall diverges >10 pp; options: weighted
  loss, oversampling segmental/transverse, or augmenting those classes.
- [ ] **Export best weights** to `models/` and wire into `ensemble.py` as an optional
  type-tagging head on top of the existing fracture detector.

---

## Pass 2 — 2026-06-14 — First type classifier trained

Ran `train_typing.py` (yolov8s-cls, 40 epochs req, early-stop ep 19, best ep 9,
imgsz 320, batch 32). Weights: `runs/classify/fracture_type_cls/weights/best.pt`.

**Overall top-1 = 0.703** (4 classes, random baseline 0.25) — real signal, far
from the >95% project goal.

Per-class (test, ~8/class — TINY, noisy):
| type | acc | confusion |
|---|---|---|
| spiral | 1.00 (8/8) | — |
| oblique | 0.75 (6/8) | → spiral, transverse |
| transverse | 0.62 (5/8) | → segmental/spiral/oblique |
| **segmental** | **0.38 (3/8)** | → oblique (4 of 5 misses) |

**Read:** segmental↔oblique is the main confusion (clinically plausible —
both can show angled/multi-line patterns). The 32-image test set makes per-class
numbers swing ±12 pp per image, so treat as a proof-of-concept, not a grade.

**Decisions / next pass:**
- Proof-of-concept SUCCESS: typing is learnable from this data. NOT production.
- Biggest lever is DATA: add GRAZPEDWRI `ao_classification` (AO/OTA → morphology
  map) to 5-10x the set and add a real test split. Until then accuracy is noisy.
- Then: oversample/augment segmental; re-evaluate; only after a real test split
  should the ">95%" goal be assessed.
- Keep type output flagged "best guess / experimental" in any UI — do NOT replace
  the detector's role with it.

---

## Pass 3 — 2026-06-14 — AO/OTA feasibility

### Question
Can `ao_classification` codes in `GRAZ_raw/dataset.csv` be mapped to fracture
morphology classes {oblique, transverse, segmental, spiral} to grow the typing set?

### Dataset counts

| metric | count |
|---|---|
| Total rows in dataset.csv | 20,327 |
| fracture_visible == 1 | 13,550 |
| Non-empty ao_classification | 14,158 |
| Distinct AO code strings | 104 |

Many rows hold *compound* codes (e.g. `23r-M/3.1; 23u-E/7`) because pediatric
forearm fractures often break both radius and ulna simultaneously.

### Code structure (AO Pediatric PCCF)

Format: `<bone><laterality>-<location>/<child_code>`

- **Bone/segment prefix** — e.g. `23r` = radius (bone 2, segment 3),
  `23u` = ulna, `22` = distal radius, `72` = finger phalanx.
- **Location letter** — `M` = metaphysis, `E` = epiphysis, `D` = diaphysis.
- **Child code** — a numeric severity/pattern code *within* that anatomic location.

#### Child code frequency across all entries

| child code | count | AO meaning |
|---|---|---|
| 2.1 | 9,230 | Incomplete / torus / buckle fracture |
| 3.1 | 6,925 | Complete, non-displaced (greenstick) |
| 7   | 3,227 | Physeal (Salter-Harris growth-plate) injury |
| 1   |   180 | Plastic deformation |
| 4.1 |   147 | Complete, displaced |
| 1.1 |    56 | Bowing / stress deformation |
| 3   |    40 | Complete fracture (generic) |
| 5.1 |    16 | Comminuted |
| 4.2 |     8 | Displaced with angulation |

Location breakdown: M (metaphysis) 14,308 · E (epiphysis) 5,020 · D (diaphysis) 528.

### Mapping verdict: NOT MAPPABLE (cleanly or partially)

**The AO Pediatric PCCF child codes encode severity, completeness, and
displacement — NOT the line-direction morphology classes used in HUMERUS_rf.**

Specifically:
- The morphology classes we need (oblique / transverse / spiral / segmental)
  describe the **orientation and shape of the fracture line** as seen on the X-ray.
- The AO PCCF child codes (2.1, 3.1, 4.1 …) describe the **degree of cortical
  disruption** (buckle vs. greenstick vs. complete vs. comminuted) and
  **displacement**, not the line direction.
- The dominant codes `2.1` (torus/buckle) and `3.1` (greenstick) are pediatric-
  specific incomplete fracture patterns that have no direct adult-AO/OTA
  morphology equivalent. Code `7` is exclusively Salter-Harris physeal injury —
  also orthogonal to line direction.
- Even `4.1` ("complete, displaced") and `5.1` ("comminuted") tell us completion
  and displacement but not whether the fracture line is transverse, oblique, or
  spiral — those distinctions require reading the image directly.
- No sub-qualifier in the PCCF format (as used in this CSV) encodes
  oblique / transverse / spiral. Adult AO/OTA long-bone codes (e.g. 12-A1 =
  simple transverse, 12-A2 = simple oblique) *do* encode this, but GRAZPEDWRI
  uses the **Pediatric** PCCF, which replaced that axis with completeness/
  displacement severity — a different taxonomy entirely.

### What the images *could* contribute (with expert re-labeling)

GRAZPEDWRI has 13,550 fracture-visible images from a pediatric wrist/forearm
cohort. If someone added morphology labels from image review, it would 10× the
typing set and add a different anatomic region (radius/ulna vs. humerus).
However:
- Pediatric forearm fractures are dominated by torus/greenstick patterns that
  are *not* in the current 4-class taxonomy — most would need a new class or
  be excluded.
- The 16 GB full Figshare image archive is not on disk; only the ~2,400-image
  Roboflow subset has been downloaded.

### Recommended next steps

1. **Do NOT attempt an AO → morphology code mapping.** There is no reliable
   lookup — the codes answer a different question than the one the typing head
   needs.
2. **Best near-term lever:** find an adult long-bone typed dataset that uses
   AO/OTA 12-A/B/C or OTA codes (which *do* encode line direction), or a
   dataset with direct "transverse / oblique / spiral" radiologist labels.
   Candidates: MURA (no type labels), RxRx (no fracture subtype), custom
   hospital PACS pull with radiology reports.
3. **Alternative — extend HUMERUS_rf with augmentation** (flips, rotations,
   zoom, brightness) to double training set size at low cost while the typed
   data problem is unsolved; then re-evaluate Pass 2 confusion.
4. **Keep typing classifier flagged "experimental"** until training set exceeds
   ~5,000 images with a genuine held-out test split.

---

## Pass 4 — 2026-06-14 — Oversampling the weak classes (+ AO path closed)

**AO/OTA data-growth: DEAD END (Pass 3).** GRAZPEDWRI uses pediatric PCCF codes
(cortical completeness/severity), NOT line-direction. They do NOT map to
oblique/transverse/segmental/spiral. No 16GB download is worthwhile. Growing
typing data requires a NEW adult OTA-coded or radiologist-labeled morphology set
(a research task, not compute).

**Oversampling experiment.** Duplicated segmental (289->578) + transverse
(299->598) in train, retrained yolov8s-cls 60 epochs (early-stop 25, best 15).
- Overall top-1: 0.703 -> **0.734**.
- segmental 0.38 -> 0.625, transverse 0.62 -> 0.875 (targeted classes improved);
  oblique 0.75 -> 0.625, spiral 1.00 -> 0.875 (small cost). Classic rebalance.
- 32-image test => treat as directional, not definitive.

**Status: typing track at a DATA WALL.** Best model
`runs/classify/fracture_type_cls_os` (top-1 0.734, experimental). Further gains
need new typed data, not more compute.

**Next (research, anytime):** find an adult long-bone dataset with OTA 12-A/B/C
codes or direct morphology labels; only then is the >95% goal assessable. Keep
type output flagged experimental in any UI.

---

## Pass 5 — 2026-06-14 — Typed-data hunt

### Objective
Find public datasets with fracture-MORPHOLOGY labels (line direction: transverse /
oblique / spiral / comminuted / segmental etc.) to unblock the typing classifier
stuck at top-1 0.734 on 1,324 humerus images.

### Candidates evaluated

---

#### 1. Kaggle — Bone Break Classification Image Dataset (pkdarabi)
- **URL:** https://www.kaggle.com/datasets/pkdarabi/bone-break-classification-image-dataset
- **Label scheme:** CONFIRMED morphology. 10 image-level classes:
  Avulsion, Comminuted, Fracture-Dislocation, Greenstick, Hairline, Impacted,
  Longitudinal, Oblique, Pathological, Spiral.
  **NOTE: no "Transverse" class** — it is labelled "Longitudinal" (perpendicular
  axis framing may differ). "Segmental" is also absent; closest analogue is
  Comminuted. Overlap with current 4-class scheme: Oblique ✓, Spiral ✓,
  Comminuted ≈ Segmental (partial), Transverse ✗ (not present by that name).
- **Size:** ~1,000–1,200 images total across 10 classes; per-class rough counts
  from search-confirmed figures: Avulsion ~123, Comminuted ~148, Greenstick ~122,
  Hairline ~111, Impacted ~84, Longitudinal ~80, Oblique ~85, Pathological ~134,
  Spiral ~86. Total ~1,000 images (some sources cite up to ~9,000 after
  augmentation — treat augmented versions with caution for training).
- **Annotation type:** Image-level classification (folder-per-class). No bounding boxes.
- **License:** Not confirmed from search results alone; Kaggle default is
  "Data files © Original Authors" unless stated otherwise. Must check dataset
  page for explicit license before use in production.
- **Access:** Free Kaggle download (requires account). Direct download via
  `kaggle datasets download pkdarabi/bone-break-classification-image-dataset`.
- **Fit score: 3/5.** Real morphology labels confirmed. Overlaps Oblique + Spiral
  directly. Missing Transverse (uses Longitudinal instead — mappable with care)
  and Segmental. ~80–148 images/class is comparable to our current humerus
  counts, so volume gain is modest unless augmented versions are trustworthy.
  Multi-region (mixed anatomy) adds generalisation but also noise. Class name
  reconciliation needed before merging with current 4-class taxonomy.

---

#### 2. Roboflow Universe — Fracture_ID (hazar/fracture_id)
- **URL:** https://universe.roboflow.com/hazar/fracture_id
- **Label scheme:** Morphology classification dataset with classes including
  Avulsion, Comminuted, Dislocation, Greenstick, Hairline, Impacted, Longitudinal,
  Oblique, Pathological, Spiral — same or very similar taxonomy to pkdarabi
  (may be the same underlying source re-uploaded to Roboflow with bounding-box
  annotations added).
- **Size:** 309 images (small). Likely bounding-box annotated.
- **Annotation type:** Object detection (boxes), which means morphology is a box
  class — can be collapsed to image-level for classifier training.
- **License:** Roboflow Public datasets are typically CC BY 4.0; verify on page.
- **Access:** Free Roboflow download in YOLO/COCO/VOC formats.
- **Fit score: 2/5.** Morphology labels confirmed but 309 images total is too
  small to move the needle meaningfully. Worth merging if pkdarabi is acquired,
  but not a standalone solution.

---

#### 3. FracAtlas (figshare / Kaggle mirror)
- **URL:** https://figshare.com/articles/dataset/The_dataset/22363012
- **Label scheme:** CONFIRMED BINARY ONLY — fractured / not-fractured.
  CSV columns: fractured (0/1), hand/leg/hip/shoulder region, hardware present.
  NO fracture morphology or line-direction labels whatsoever.
  **DISQUALIFIED** — same recurring mistake as GRAZPEDWRI. Do not re-propose.
- **Size:** 4,083 images (717 fractured, 3,366 normal).
- **License:** CC BY 4.0.
- **Fit score: 0/5.** Binary classification only. Not usable for morphology typing.

---

#### 4. Tanzi et al. AO/OTA Femur Fracture Dataset (proximal femur, AO 31A/B)
- **URL / paper:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11422035/ (YOLOv8
  follow-up); original Tanzi 2022 in European Journal of Radiology Open.
- **Label scheme:** CONFIRMED adult AO/OTA adult codes for proximal femur:
  31-A1, 31-A2, 31-A3, 31-B1, 31-B2, 31-B3 — 6 classes plus "unbroken".
  AO 31-A1 = simple transverse-like intertrochanteric; A2/A3 = comminuted
  patterns. These DO encode fracture morphology and line characteristics (adult
  AO, not pediatric PCCF). Strong clinical validity.
- **Size:** 4,233 manually annotated images; class breakdown: Unbroken 1,976,
  A1 535, A2 473, A3 171, B1 637, B2 333, B3 108.
- **Annotation type:** Image-level (X-ray classification) with some bounding-box
  work in follow-up papers.
- **License / Access:** NOT publicly downloadable. Dataset appears to be held by
  the original authors (University of Turin / CTO Hospital Turin). Multiple papers
  use it but no public repository found. Contact: authors of Tanzi et al. 2022
  (DOI 10.1016/j.ejro.2020.100383).
- **Fit score: 4/5 if obtained; 0/5 currently.** Best clinical-grade morphology
  signal found — adult AO codes directly map to line-direction morphology. Blocked
  by non-public access. Worth a direct email request to authors.

---

#### 5. Kaggle — Bone Fracture Multi-Region X-ray Data (bmadushanirodrigo)
- **URL:** https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data
- **Label scheme:** Based on search context this dataset contains fracture /
  no-fracture labels by body region (multi-region detection). Morphology type
  labels NOT confirmed. Likely binary per-region, similar to FracAtlas pattern.
- **Size:** Unknown without direct page access.
- **Fit score: 1/5 (unconfirmed).** Do not acquire until morphology labels are
  explicitly confirmed on the Kaggle page.

---

### Typed-data shortlist (ranked)

| Rank | Dataset | Morphology confirmed? | ~Usable typed images | Public? |
|------|---------|----------------------|---------------------|---------|
| 1 | **pkdarabi Bone Break Classification (Kaggle)** | YES — 10 classes | ~900 net (excl. pathological/dislocation) | Yes |
| 2 | **Tanzi AO/OTA Femur (email request)** | YES — AO 31A/B | ~2,257 fractured | No (request) |
| 3 | **hazar fracture_id (Roboflow)** | YES — same taxonomy | ~280 usable | Yes |
| 4 | **bmadushanirodrigo multi-region** | UNCONFIRMED | unknown | Yes |
| X | FracAtlas | **NO — binary only** | 0 | Yes (irrelevant) |

### #1 Recommended acquisition: pkdarabi Bone Break Classification

**Why:** Only confirmed-public dataset with direct fracture morphology image-level
labels at a scale that could meaningfully extend the current 1,324-image humerus
set. Oblique and Spiral map 1:1 to our taxonomy. Longitudinal (~80 images) maps
approximately to Transverse (same bone-axis fracture, name differs). Comminuted
(~148 images) is the closest proxy for Segmental (both multi-fragment); label
reconciliation is needed but straightforward.

**Prep / conversion needed:**
1. Confirm license on Kaggle page before use — must be CC BY or similar.
2. Download with `kaggle datasets download pkdarabi/bone-break-classification-image-dataset`.
3. Exclude Avulsion, Hairline, Impacted, Pathological, Fracture-Dislocation,
   Greenstick — these have no clean mapping to current 4-class scheme.
4. Remap: Oblique → oblique, Spiral → spiral, Longitudinal → transverse
   (document this mapping), Comminuted → segmental (document as approximate).
5. Merge with existing `typing_dataset/train` — run de-dup check on filenames.
6. Re-train with the merged ~1,900–2,200 image set, re-evaluate on humerus test split.

**#2 if public access is blocked:** Email Tanzi et al. (University of Turin /
CTO Hospital Turin) requesting the 4,233-image AO/OTA femur dataset. This would
add 2,257 fracture-positive images with high-quality adult AO labels (A1/A2/A3
map cleanly to transverse/oblique/comminuted patterns). Longer lead time but
far higher clinical quality.

### >95% typing goal: realistically reachable from public data?

**No — not from currently available public data alone.**

Adding pkdarabi (~900 usable typed images) brings training to ~2,200 images across
a harmonised 4-class scheme. That likely moves top-1 from 0.734 toward ~0.80–0.85
(estimated), still short of >95%. To reach >95% reliably:
- Need ≥5,000 typed training images with a genuine 500+ image held-out test split.
- The Tanzi femur dataset (via email request) plus pkdarabi combined would approach
  ~3,500–4,000 usable typed images — closer, but anatomy mismatch (femur vs. humerus
  vs. radius) means the model must learn anatomy-agnostic morphology features, which
  requires even more diversity and likely a larger backbone.
- Realistic path to >95%: either (a) a hospital PACS pull with radiology report NLP
  to extract morphology labels (100s–1000s per institution), or (b) expert re-labeling
  of the 13,550-image GRAZPEDWRI set for adult-equivalent morphology patterns (high
  cost, pediatric anatomy caveat still applies).
- **Recommended near-term goal:** target top-1 ~0.85 with pkdarabi merge, flag the
  >95% goal as requiring institutional data, and keep type output marked experimental.

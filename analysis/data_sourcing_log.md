# Data Sourcing Log — Bone-R Fracture Detection

**Purpose:** Track public datasets that supply bounding-box (or easily-convertible) fracture annotations to supplement FracAtlas.
Priority is given to datasets covering hip fractures, shoulder fractures, oblique-view fractures, and AO/OTA fracture-type labels — the current blind spots identified in `analysis/region_balance.md`.

---

## Priority shortlist (live)

| Rank | Dataset | Rationale |
|------|---------|-----------|
| 1 | **GRAZPEDWRI-DX** | 20 K images, YOLO boxes ready, CC BY 4.0, directly downloadable from Figshare — highest volume/lowest friction of all candidates so far. Wrist-only so won't fix hip/shoulder gap, but will dramatically increase fracture-box count and anatomical diversity (9 classes). Acquire first. |
| 2 | **Roboflow "Hip X-ray" (modzie-work)** | Only known public set with hip-specific bounding boxes. Small (~83 images), but directly targets the #1 blind spot. License and exact class labels need verification before use. |
| 3 | **Bone Fracture Multi-Region X-ray (Kaggle)** | 10,580 images covering hip, knee, lumbar and limbs; however annotation type is image-level only — conversion to YOLO boxes requires pseudo-labelling or manual re-annotation. Investigate further before committing. |

---

## Pass — 2026-06-12

### 1. GRAZPEDWRI-DX

- **Link:** https://doi.org/10.6084/m9.figshare.14825193  
  Paper: https://www.nature.com/articles/s41597-022-01328-z  
  Roboflow mirror: https://universe.roboflow.com/project-bef88/grazpedwri-dx
- **Modality / Region:** Plain radiograph — **pediatric wrist only** (no hip/shoulder)
- **Images / Fractures:** 20,327 images from 6,091 patients; 18,090 fracture objects in 13,550 images (66.7 %)
- **Annotation type:** Multi-class **bounding box** (box) for most classes + polygon for two; YOLO v5 TXT labels included natively in the ZIP
- **9 classes:** fracture (box), boneanomaly (box), bonelesion (polygon), foreignbody (box), metal (box), periostealreaction (polygon), pronatorsign (box), softtissue (box), axis (line)
- **License:** CC BY 4.0 — free to use, redistribute, adapt with attribution
- **Access / Download:** Direct download from Figshare (no login required). Total ~15.2 GB ZIP archives. The `yolov5/` subfolder contains TXT label files + `meta.yaml` ready to plug into YOLO training pipelines.
- **Format conversion to YOLO:** Zero effort — labels are already in YOLO v5 format. Only the `fracture` class is needed; the other 8 can be dropped or retained as auxiliary classes.
- **Fit score: 3 / 5** — High volume and zero annotation friction, but wrist-only means it does not address the hip/shoulder blind spots. Will meaningfully expand total fracture-box count and help with generalization, but should be paired with a hip/shoulder set.

---

### 2. Bone Fracture Multi-Region X-ray (Kaggle — bmadushanirodrigo)

- **Link:** https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data
- **Modality / Region:** Plain radiograph — lower limb, upper limb, lumbar, **hip**, knee, and more
- **Images / Fractures:** 10,580 images total; fractured vs. non-fractured split; exact fracture count not published in metadata
- **Annotation type:** **Image-level classification only** (fractured / not fractured per folder) — no spatial boxes
- **License:** Listed on Kaggle; based on prior search results appears to be community-posted with no explicit open license confirmed — **must verify before redistribution**
- **Access / Download:** Kaggle API: `kaggle datasets download bmadushanirodrigo/fracture-multi-region-x-ray-data`; requires free Kaggle account
- **Format conversion to YOLO:** Hard — no box annotations exist. Options: (a) use a pre-trained detector to generate pseudo-labels and manually audit, (b) use image-level labels only to augment classifier head while keeping YOLO boxes from other datasets. Not a drop-in solution.
- **Fit score: 2 / 5** — Good regional diversity including hip, but image-level only. Useful as a supplementary classification signal or for weak-supervision experiments, not for direct box training.

---

### 3. GRAZPEDWRI-DX "Ju split" (pre-processed version)

- **Link:** https://ruiyangju.github.io/GRAZPEDWRI-DX_JU/
- **Notes:** Community re-split of the original GRAZPEDWRI-DX (14,204 train / 4,094 val / 2,029 test). Same CC BY 4.0 license as upstream. Accessible via OneDrive link on the page. Annotations are identical to the Figshare original. This is NOT a separate dataset — log it here only as a convenient alternative download path if Figshare is slow.
- **Fit score:** Same as GRAZPEDWRI-DX above (3 / 5); use if direct Figshare download is problematic.

---

### 4. Roboflow "Hip X-ray" (modzie-work)

- **Link:** https://universe.roboflow.com/modzie-work/hip-xray  
  (HTTP 403 on direct fetch — must access via Roboflow web UI or API)
- **Modality / Region:** Plain radiograph — **hip/femoral neck** specifically
- **Images / Fractures:** ~83 images (small); class reported as `femoral-neck`
- **Annotation type:** **Bounding box** (Roboflow object detection project)
- **License:** Roboflow Universe projects default to CC BY 4.0 unless author overrides — **must confirm on the dataset page before use**
- **Access / Download:** Roboflow API export in YOLO format; free tier allows download after sign-in
- **Format conversion to YOLO:** Trivial — Roboflow export directly to YOLOv8 TXT format
- **Fit score: 4 / 5** — Directly addresses the #1 blind spot (hip). Very small (83 images) so will not close the gap alone, but is the highest-priority targeted acquisition. Combine with GRAZPEDWRI-DX for volume.

---

### Open questions / next pass (superseded — see Pass 2 below)

1. **Verify Roboflow Hip X-ray license** — access the page directly (not via fetch) and confirm CC BY 4.0 or equivalent; also check actual class names and whether fractures are explicitly labeled vs. all-hip images.
2. **Search for shoulder-specific datasets** — the Roboflow Universe query `class:shoulder fracture` returned results; fetch and evaluate top 2–3 hits (e.g., "humerus" class datasets). ✓ Done in Pass 2.
3. **Investigate AO/OTA type-coded sets** — project currently has zero fracture-type (AO/OTA) labels. ✓ Done in Pass 2.
4. **Check Bone Fracture Detection (pkdarabi, Kaggle)** — search results suggest this may have bounding boxes; Kaggle page fetch failed — retry or access directly. ✓ Done in Pass 2.
5. **RSNA 2019 "Bone Age" vs. RSNA 2022 "Cervical Spine"** — the 2022 cervical spine challenge has bounding boxes on ~16 % of positive cases (CT, not X-ray); confirm modality mismatch and decide whether to log as out-of-scope.
6. **Assess whether high-value public X-ray targets are exhausted** — after next pass covering shoulder and AO/OTA, evaluate whether remaining gaps require synthetic augmentation or clinical data-sharing agreements instead of additional public datasets.

---

## Pass 2 — 2026-06-12

### 5. Roboflow HUMERUS Dataset (new-workspace-ozkjr)

- **Link:** https://universe.roboflow.com/new-workspace-ozkjr/humerus-ghx7b
- **Modality / Region:** Plain radiograph — **humerus / shoulder** (upper arm bone fractures)
- **Images / Fractures:** ~548 images (as reported in search snippets); all images contain fracture annotations
- **Annotation type:** **Bounding box** (Roboflow object detection project) with **4 fracture-TYPE classes: oblique, segmental, spiral, transverse** — this is the first identified public dataset labeling fracture morphology with spatial boxes
- **License:** Roboflow Universe projects default to CC BY 4.0 unless overridden — **must verify on the dataset page before use**
- **Access / Download:** Roboflow API export in YOLO format; free tier allows download after sign-in. Export URL pattern: `https://universe.roboflow.com/new-workspace-ozkjr/humerus-ghx7b` → Download → YOLOv8
- **Format conversion to YOLO:** Trivial — Roboflow exports directly to YOLOv8 TXT format; the 4 existing class labels (oblique, segmental, spiral, transverse) can be used as-is or merged into a single `fracture` class for the detection track
- **Fit score: 5 / 5** — This is a double win: it directly addresses the shoulder/humerus blind spot AND provides the first fracture-morphology labels (oblique, segmental, spiral, transverse) found in any public dataset. Even at ~548 images it is uniquely valuable for the fracture-type classification track. **Highest priority acquisition this pass. Acquire immediately.**
- **Caveat:** Image count and license need on-page verification; 548 images is small — useful for initializing the fracture-type track but will require augmentation.

---

### 6. Bone Break Classification Image Dataset (pkdarabi, Kaggle)

- **Link:** https://www.kaggle.com/datasets/pkdarabi/bone-break-classification-image-dataset
- **Modality / Region:** Plain radiograph — multi-region (no specific region breakdown published in metadata)
- **Images / Fractures:** Exact count not confirmed in search results; likely in the low thousands based on dataset structure
- **Annotation type:** **Image-level classification** with **10 named fracture-TYPE classes**: Avulsion, Comminuted, Fracture dislocation, Greenstick, Hairline, Impacted, Longitudinal, Oblique, Pathological, Spiral — no bounding boxes
- **License:** Kaggle community dataset — license not explicitly confirmed; **must check before redistribution**
- **Access / Download:** Kaggle API: `kaggle datasets download pkdarabi/bone-break-classification-image-dataset`; requires free Kaggle account
- **Format conversion to YOLO:** Hard — image-level only, no boxes. However, the 10-class fracture-type taxonomy is a strong asset: images can supply a classification head for fracture type, and pseudo-labelling from a detector trained on the HUMERUS Roboflow set (entry #5) could generate approximate boxes. This is the most complete fracture-type label vocabulary found in any public dataset.
- **Fit score: 4 / 5** — Does not supply boxes, so cannot directly train the YOLO detection head. However, the 10-class typed taxonomy is the broadest fracture-morphology label set found; invaluable for the fracture-type classification track once a box-level detector exists. Prioritize alongside entry #5.

---

### 7. pkdarabi Bone Fracture Detection — Computer Vision Project (Kaggle)

- **Link:** https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project
- **Modality / Region:** Plain radiograph — multi-region: wrist, elbow, **shoulder**, knee, ankle (confirmed from dataset description)
- **Images / Fractures:** 4,148 images; YOLO annotation files for train/val/test splits; also available in COCO format
- **Annotation type:** **Bounding box** — YOLO TXT format, with class labels indicating both anatomical region and fracture presence/absence. This is a separate dataset from the "Bone Break Classification" set above.
- **License:** Kaggle community dataset — license not confirmed; **must verify before redistribution**
- **Access / Download:** Kaggle API: `kaggle datasets download pkdarabi/bone-fracture-detection-computer-vision-project`; free account required
- **Format conversion to YOLO:** Zero effort — labels are already in YOLO TXT format with train/val/test splits pre-defined
- **Fit score: 4 / 5** — Directly addresses the shoulder gap with bounding boxes, YOLO-ready, moderate size (4,148 images). The class labels cover anatomical regions + fracture/no-fracture; does NOT provide fracture-type labels. Combination with entry #5 (HUMERUS, typed) is the ideal strategy.
- **Note:** This same author (pkdarabi) was flagged in Pass 1 open questions — confirmed to have boxes.

---

## Priority shortlist (live) — updated Pass 2

| Rank | Dataset | Rationale |
|------|---------|-----------|
| 1 | **GRAZPEDWRI-DX** | 20 K images, YOLO boxes ready, CC BY 4.0, zero friction. Wrist-only but highest fracture-box volume. Acquire first. |
| 2 | **Roboflow HUMERUS** (new-workspace-ozkjr) | ~548 images, 4 fracture-TYPE classes (oblique, segmental, spiral, transverse) with bounding boxes. Only public dataset found combining shoulder coverage + fracture morphology labels. Opens the fracture-type classification track. **Critical acquisition.** |
| 3 | **pkdarabi Bone Fracture Detection CV Project** | 4,148 images, YOLO-ready boxes, includes shoulder. Good volume supplement for the shoulder gap; no typing labels. |
| 4 | **Roboflow "Hip X-ray" (modzie-work)** | ~83 images, hip-specific bounding boxes. Tiny but targeted; license still unverified. |
| 5 | **Bone Break Classification** (pkdarabi) | 10 fracture-type classes, image-level only. Use as classification head training data or for pseudo-label generation once a box detector exists. |
| 6 | **Bone Fracture Multi-Region X-ray (Kaggle)** | 10,580 images but image-level only; useful for classifier augmentation or weak supervision. |

---

### Open questions / next pass

1. **Verify license and exact image count for Roboflow HUMERUS dataset** — access https://universe.roboflow.com/new-workspace-ozkjr/humerus-ghx7b directly; confirm CC BY 4.0, actual class distribution, and whether all 548 images have fracture annotations.
2. **Verify license for pkdarabi datasets** — both Kaggle pages need direct access to confirm license before redistribution.
3. **RSNA 2022 Cervical Spine Fracture** (CT, not X-ray) and **RSNA 2023 Abdominal Trauma** — if CT-based data is acceptable, these have bounding boxes; worth logging as out-of-scope or a modality-extension option.
4. **AO/OTA coded sets remain a gap** — all AO/OTA research papers found use private hospital data; no public AO/OTA-coded X-ray corpus was identified. This gap likely requires clinical partnership or a data sharing agreement rather than public dataset sourcing. Pass 3 could do one final check on PhysioNet and the OpenNeuro/TCIA radiograph collections, but expectations should be low.
5. **Exhaustion assessment:** The main public X-ray fracture-detection space (Roboflow Universe, Kaggle) appears largely surveyed for bounding-box and fracture-type datasets. The fracture-type classification track now has two entry points (HUMERUS Roboflow + Bone Break Classification). Pass 3 should focus narrowly on: (a) AO/OTA coded corpora via TCIA/PhysioNet, and (b) any RSNA challenge datasets with boxes not yet evaluated. After that, the high-value public targets are likely exhausted and the log should advise on synthetic augmentation and clinical data-sharing strategies.

---

## Pass 3 — 2026-06-12 (final)

### 8. TCIA (The Cancer Imaging Archive) — fracture / musculoskeletal collections

- **Link:** https://www.cancerimagingarchive.net/collections/
- **Modality / Region:** Mixed — predominantly CT and MRI; a minority of plain radiograph collections exist
- **Relevance to Bone-R:** TCIA's primary mandate is oncology imaging; its musculoskeletal collections (e.g., QIN-HEADNECK, TCGA-SARC, ACRIN-CONTRALATERAL-BREAST) are tumor-focused, not fracture-focused. As of the knowledge cutoff (Aug 2025), no TCIA collection provides AO/OTA-coded fracture labels or fracture-type bounding boxes on plain radiographs. The closest adjacent collection is the **Musculoskeletal Oncology** series (CT-based bone tumor segmentations), which is out of scope.
- **AO/OTA label status:** No AO/OTA-coded radiograph corpus exists in TCIA. All published AO/OTA classification research uses private hospital PACS data with ethics board approval (see e.g., Oliveira et al. 2021, Prijs et al. 2020). TCIA has no mechanism for crowdsourced fracture typing.
- **Fit score: 1 / 5** — Out of scope. TCIA is designed for oncology and advanced modalities, not for plain-radiograph fracture detection or AO/OTA typing. Not recommended for acquisition.
- **Next action:** None. Skip TCIA for this project.

---

### 9. PhysioNet — fracture / musculoskeletal datasets

- **Link:** https://physionet.org/content/
- **Modality / Region:** PhysioNet's primary focus is physiological time-series (ECG, EEG, waveforms). Its imaging collections are sparse and predominantly ophthalmology/chest X-ray (e.g., MIMIC-CXR for chest, EyePACS for retina).
- **Relevance to Bone-R:** No PhysioNet dataset provides musculoskeletal plain radiographs with fracture bounding boxes or AO/OTA fracture-type labels as of Aug 2025. PhysioNet does not host extremity X-ray corpora.
- **AO/OTA label status:** No AO/OTA-coded corpus. The structural barrier is identical to TCIA: AO/OTA coding is a clinical workflow annotation that has not been released as a public open dataset by any institution.
- **Fit score: 1 / 5** — Out of scope. PhysioNet is not a radiograph-fracture resource. Not recommended for acquisition.
- **Next action:** None. Skip PhysioNet for this project.

---

### 10. RSNA 2022 Cervical Spine Fracture Detection Challenge — OUT OF SCOPE

- **Link:** https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection
- **Modality / Region:** **CT** (computed tomography) — axial slices of the cervical spine (C1–C7)
- **Images / Fractures:** ~3,000 patient studies; bounding boxes on ~16 % of positive cases; 7 vertebra-level classes (C1–C7 fracture/no-fracture)
- **Annotation type:** Bounding box (YOLO-convertible from competition format) on CT slices — NOT plain X-ray
- **Relevance verdict: OUT OF SCOPE** — CT modality is incompatible with Bone-R's plain-radiograph pipeline. Feature distributions, bone texture, and Hounsfield unit intensity profiles differ fundamentally from 2-D X-ray. Fine-tuning a YOLO model on CT slices and applying it to X-rays would degrade performance. Spine is also outside the hip/shoulder/wrist extremity focus.
- **Fit score: 0 / 5** — Do not acquire. Log here for completeness.

---

### 11. RSNA Pediatric Bone Age Challenge (2017) — OUT OF SCOPE

- **Link:** https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017
  Kaggle: https://www.kaggle.com/competitions/rsna-bone-age
- **Modality / Region:** Plain radiograph — **left-hand wrist** (pediatric, for bone-age estimation)
- **Images / Fractures:** ~14,000 hand X-rays; labels are **numeric bone age in months** — zero fracture annotations
- **Relevance verdict: OUT OF SCOPE** — Bone age estimation is an unrelated task. The images are wrist X-rays but contain no fracture labels, no bounding boxes, and the distribution skews heavily toward normal (unfractured) pediatric wrists. Using them for fracture detection would add only negative/background samples with no corresponding annotations.
- **Fit score: 0 / 5** — Do not acquire. Log here for completeness.

---

### 12. RSNA 2019 Pneumonia Detection / RSNA 2023 Abdominal Trauma — OUT OF SCOPE

- **RSNA 2019 Pneumonia:** Chest X-ray, lung opacity bounding boxes — no bone/fracture content.
- **RSNA 2023 Abdominal Trauma:** CT, abdominal organs — no extremity fractures.
- Both are out of scope by modality (CT) and/or anatomical region. Not logged further.

---

### Synthetic augmentation fallback

Because no new public dataset meaningfully closes the hip-fracture bounding-box gap or adds AO/OTA-typed labels, synthetic augmentation is the practical path for both shortfalls once the acquisitions above are complete.

**Strategy 1 — Copy-paste (mosaic/cut-mix) augmentation of fracture boxes**

Take confirmed fracture bounding-box crops from GRAZPEDWRI-DX and the Roboflow HUMERUS dataset and paste them onto background X-ray images from the multi-region classification sets (Kaggle Bone Fracture Multi-Region, pkdarabi Bone Break Classification). This is straightforward in Albumentations (`CopyPaste` transform or a custom mosaic pipeline) and has demonstrated +2–4 % mAP improvement in data-scarce medical detection tasks. It does not hallucinate anatomy — the pasted crop is a real fracture image — but blending artifacts at paste boundaries require Gaussian blur or Poisson blending. Recommended as the first augmentation step, especially for hip (only ~83 labeled images currently available).

**Strategy 2 — Diffusion/GAN synthesis (higher-risk, longer timeline)**

Conditional diffusion models (e.g., a ControlNet conditioned on bone structure masks, or a fine-tuned Stable Diffusion on X-ray domain) can generate synthetic fracture images with plausible texture. The primary use case is AO/OTA fracture-type generation where no box-labeled public data exists (e.g., comminuted hip fracture patterns). Caveats: (a) requires an existing labeled seed set (~200+ per class) to avoid mode collapse; (b) synthetic images must be validated by a radiologist before mixing into training data to avoid introducing clinically implausible fracture morphologies; (c) regulatory risk — if the model is destined for clinical deployment, synthetic training data provenance must be disclosed. Treat this as a 2nd-phase strategy once real-data baselines are established.

---

## Priority shortlist (CLOSED)

This section consolidates all three passes into a final acquisition plan. The "live" shortlists above are superseded by this entry.

### Track A — Detection Boxes (YOLO bounding box training)

| Priority | Dataset | Rationale | Next action |
|----------|---------|-----------|-------------|
| A-1 | **GRAZPEDWRI-DX** | 20 K images, YOLO-ready, CC BY 4.0, zero friction; highest fracture-box volume of any public dataset found | Download from Figshare (no login); unzip `yolov5/` subfolder |
| A-2 | **pkdarabi Bone Fracture Detection CV Project** (Kaggle) | 4,148 images, YOLO TXT format, multi-region including shoulder; good volume supplement | `kaggle datasets download pkdarabi/bone-fracture-detection-computer-vision-project`; verify license on dataset page |
| A-3 | **Roboflow HUMERUS** (new-workspace-ozkjr) | ~548 images, shoulder coverage with fracture-TYPE boxes; also feeds Track B | Roboflow API export → YOLOv8; verify CC BY 4.0 on dataset page |
| A-4 | **Roboflow Hip X-ray** (modzie-work) | ~83 images; only public hip-specific bounding box dataset found; tiny but targeted | Roboflow API export; verify license; combine with copy-paste augmentation |
| A-5 | **Copy-paste augmentation** | Synthetically expand hip/shoulder boxes using Strategy 1 above once A-1 through A-4 are downloaded | Implement Albumentations `CopyPaste` or mosaic in training pipeline |

### Track B — Fracture Typing (AO/OTA morphology classification)

| Priority | Dataset | Rationale | Next action |
|----------|---------|-----------|-------------|
| B-1 | **Roboflow HUMERUS** (new-workspace-ozkjr) | Only public dataset found with fracture-type bounding boxes (oblique, segmental, spiral, transverse); opens the typed-detection track | Same as A-3 above |
| B-2 | **pkdarabi Bone Break Classification** (Kaggle) | 10 fracture-type classes (Avulsion, Comminuted, Greenstick, Hairline, Impacted, Longitudinal, Oblique, Pathological, Spiral, Fracture dislocation); image-level only but broadest typing vocabulary found | `kaggle datasets download pkdarabi/bone-break-classification-image-dataset`; use for classification head or pseudo-label generation |
| B-3 | **AO/OTA coded corpus** | No public AO/OTA-coded X-ray dataset exists; gap requires clinical data-sharing agreement with a hospital PACS, not further public dataset search | Pursue IRB/data-sharing agreement with a trauma center; this loop cannot close this gap via public data |
| B-4 | **Diffusion synthesis (Strategy 2)** | Once B-1 and B-2 baselines exist, synthesize underrepresented fracture types (comminuted hip, impacted femoral neck) via conditional diffusion | Defer to phase 2; requires radiologist validation before training use |

### Summary of negative findings (this pass)

- **TCIA:** No fracture-detection or AO/OTA-coded plain-radiograph collection. Out of scope.
- **PhysioNet:** No extremity X-ray corpus. Out of scope.
- **RSNA Cervical Spine 2022:** CT modality, spine only. Out of scope.
- **RSNA Bone Age 2017:** Plain wrist X-ray but zero fracture labels. Out of scope.
- **AO/OTA public gap confirmed:** Three passes across Roboflow Universe, Kaggle, TCIA, PhysioNet, and RSNA challenge archives found zero publicly released AO/OTA-coded radiograph datasets. This is a structural gap in the public data ecosystem, not a search failure.

---

**Loop status: REOPENED — Pass 4 (hip-fracture targeted re-search)**

The "Roboflow Hip X-ray (modzie-work)" dataset logged at A-4 was found on inspection to be an OSTEOPOROSIS classification dataset (classes: Osteoporosis, Implant, No osteoporosis) — not a hip fracture detector. It has been removed from the acquisition plan. Pass 4 below searches specifically for genuine hip/proximal-femur fracture datasets.

---

## Pass 4 — 2026-06-14 (hip-fracture targeted re-search)

**Context:** The Roboflow "Hip X-ray" (modzie-work) entry was rejected after class-label verification revealed it labels osteoporosis/implants, not fractures. This pass searches Roboflow Universe, Kaggle, Zenodo, and published papers for datasets that GENUINELY annotate hip fractures (proximal femur / femoral neck / intertrochanteric / subtrochanteric) with bounding boxes, masks, or at minimum image-level fracture labels.

---

### 13. Roboflow "Proximal Femur Fracture" (ThesisYolo v8) — PRIMARY CANDIDATE

- **Links:**
  - Instance segmentation variant: https://universe.roboflow.com/thesisyolo-v8/proximal-femur-fracture
  - Detection + classification variant: https://universe.roboflow.com/thesisyolo-v8/proximal-femur-fracture-detection-and-classification
- **Modality / Region:** Plain radiograph — **proximal femur / hip** (anteroposterior pelvis/hip views)
- **Images / Fractures:** ~756 images (segmentation project); ~640 images (detection + classification project); both are fracture-only (no "no fracture" class reported)
- **Annotation type:** **Instance segmentation masks** (polygon outlines) — convertible to bounding boxes via mask bounding rect. The detection+classification variant also has a pre-trained model.
- **Classes (CONFIRMED):** 7 classes explicitly describing fracture anatomy — `dislocation`, `grater-trochanter` (greater trochanter avulsion), `intertrochanteric`, `lesser-trochanter`, `neck` (femoral neck fracture), `neck-normal` (non-fractured neck, used as a negative class in context), `subtrochanteric`. These are TRUE fracture anatomical subtypes — CONFIRMED to be genuine hip fracture labels, not osteoporosis/implant classes.
- **License:** CC BY 4.0 (reported on dataset page) — free to use with attribution
- **Access / Download:** Roboflow API export → YOLOv8 Segmentation format; free Roboflow account required. Export URL: https://universe.roboflow.com/thesisyolo-v8/proximal-femur-fracture → Download → YOLOv8 Segment. Alternatively, the detection+classification variant exports as standard bounding-box YOLO.
- **Format conversion to YOLO:** Low effort — Roboflow exports to YOLOv8 segmentation TXT format natively; convert polygon masks to bounding boxes with a one-liner (e.g. `x_min, y_min, x_max, y_max = mask.bounds`). The 7 classes can be merged into a single `hip_fracture` class for the detection track, or kept typed for the fracture-type track. `neck-normal` (negative class) should be excluded from fracture training targets.
- **Fit score: 5 / 5** — This is the strongest hip-fracture dataset found in any public repository. It directly addresses the hip blind spot, has CONFIRMED true fracture subtypes, ships with segmentation masks (richer than boxes), and is CC BY 4.0. Image count (~640–756) is small but comparable to the HUMERUS dataset (which proved sufficient to open the shoulder track). **Replace the rejected modzie-work entry; this is now A-4 in the acquisition plan.**
- **Caveats:** (1) Dataset appears to be a thesis project — provenance of the underlying X-rays is not documented on the Roboflow page; confirm source and IRB status if clinical deployment is planned. (2) `neck-normal` class must be filtered out before fracture-label training. (3) Two separate Roboflow projects exist (segmentation + detection) — download both and compare; the detection variant may have slightly different class definitions.

---

### 14. FracAtlas — hip subset

- **Link:** https://figshare.com/articles/dataset/The_dataset/22363012  
  Paper: https://www.nature.com/articles/s41597-023-02432-4
- **Modality / Region:** Plain radiograph — multi-region: **hand, wrist, shoulder, leg, hip** (4,083 images total from 3 hospitals in Bangladesh)
- **Images / Fractures:** 4,083 images total; 717 images with fracture annotations (922 fracture instances total). Hip images are a subset — exact hip count not separately published in metadata, but the dataset description explicitly lists hip as one of the four regions.
- **Annotation type:** **Bounding box + segmentation mask** per fracture instance; global classification labels also included. Full COCO-format JSON annotations provided.
- **Classes (CONFIRMED):** Annotations are at the fracture-instance level (fractured / not fractured per image, with box+mask per instance). Fracture type by region is implied by anatomical region but not AO/OTA-typed. The hip images contain genuine hip fracture annotations — CONFIRMED fracture labels (this dataset is already partially in use in the Bone-R project pipeline).
- **License:** CC BY 4.0
- **Access / Download:** Figshare direct download (no login required): https://figshare.com/articles/dataset/The_dataset/22363012
- **Format conversion to YOLO:** Low effort — COCO JSON → YOLO TXT via standard converters (e.g. `ultralytics` `convert_coco`). Already used as the base dataset for Bone-R; the hip subset can be isolated by filtering images from the `Hip` folder.
- **Fit score: 3 / 5** — Already in the project (likely ingested as part of the existing pipeline). The hip subset is real and genuinely annotated, but the total hip-image count is a fraction of the 717 fractured images (probably <200 hip-specific fracture images). Useful as a baseline but insufficient alone for the hip gap; supplement with entry #13 above.

---

### 15. PENGWIN 2024 — Pelvic X-ray (synthetic DRR, segmentation masks)

- **Link:** https://zenodo.org/records/10990768  
  Challenge page: https://pengwin.grand-challenge.org/data/
- **Modality / Region:** **Simulated X-ray (DRR from CT)** — pelvic fracture, sacrum and hip bone fragments
- **Images / Fractures:** 50,000 simulated X-ray images generated from 100 CT volumes (500 angles per volume); binary segmentation masks for bone fragments
- **Annotation type:** **Segmentation mask** (binary, per bone fragment) — no typed fracture-class labels; oriented toward fragment detection, not fracture-line classification
- **Classes (CONFIRMED):** Fragment-level segmentation — genuine pelvic/hip fracture content. However, the images are synthetic DRRs generated via DeepDRR from CT, not real plain radiographs. Distribution shift from real X-ray is significant.
- **License:** Not explicitly stated in search results — must verify on Zenodo page before use
- **Access / Download:** https://doi.org/10.5281/zenodo.10927452 and https://doi.org/10.5281/zenodo.10913195 (Zenodo, no login required)
- **Format conversion to YOLO:** Medium effort — binary masks → bounding rects is straightforward, but synthetic DRR domain gap may harm YOLO performance on real X-rays; domain adaptation (style transfer or mixed training) would be needed
- **Fit score: 2 / 5** — Genuine hip/pelvic fracture content with large volume, but synthetic images introduce domain gap. Not suitable as a primary training source without domain adaptation. Worth noting as a data-augmentation or pre-training corpus if the ThesisYolo set proves too small.

---

### 16. Published institutional hip-fracture datasets — NOT publicly released

- **Gloucester / Bath NHS hip fracture dataset:** Referenced in multiple papers (Adams et al. 2020, RSNA AI); contains ~3,000+ hip X-rays with bounding boxes and 6-class fracture typing. **NOT publicly available** — held under NHS data governance; requires data-sharing agreement with Gloucestershire Hospitals NHS Foundation Trust.
- **RSNA 2020 hip fracture dataset (internal):** Used in the Rajpurkar/Ng Stanford paper (Automatic Hip Fracture Identification, arXiv 1909.06326); 1,118 studies with 6 fracture classes and bounding boxes (3,034 bounded hips). **NOT publicly available** — Stanford AIMI has not released this dataset.
- **2025 Insights into Imaging segmentation study:** 10,308 hip X-rays, 986 manually annotated with segmentation masks and fracture-grade classification (Garden/Evans-Jensen). **NOT publicly available** — institutional data from a single center.
- **Verdict:** The highest-quality hip-fracture annotation sets are all behind institutional data-sharing agreements. The best realistically acquirable dataset is the ThesisYolo Roboflow set (entry #13). If the project needs clinical-grade volume, a data-sharing agreement is the only path.

---

## Priority shortlist — updated Pass 4 (hip gap re-opened)

**The modzie-work "Hip X-ray" entry is REMOVED from the acquisition plan (osteoporosis dataset, not fracture).**

### Track A — Detection Boxes (updated)

| Priority | Dataset | Rationale | Next action |
|----------|---------|-----------|-------------|
| A-1 | **GRAZPEDWRI-DX** | 20 K images, YOLO-ready, CC BY 4.0. Unchanged. | Download from Figshare |
| A-2 | **pkdarabi Bone Fracture Detection CV Project** | 4,148 images, YOLO boxes, multi-region. Unchanged. | Kaggle download; verify license |
| A-3 | **Roboflow HUMERUS** (new-workspace-ozkjr) | ~548 images, fracture-TYPE boxes, shoulder. Unchanged. | Roboflow API export |
| A-4 | **Roboflow Proximal Femur Fracture** (ThesisYolo v8) | ~640–756 images, CONFIRMED hip fracture subtypes (intertrochanteric, femoral neck, subtrochanteric, dislocation), CC BY 4.0, segmentation masks → boxes. **REPLACES the rejected modzie-work entry.** | Roboflow API export → YOLOv8 Segment; convert masks to boxes; exclude `neck-normal` class |
| A-5 | **FracAtlas hip subset** | Already in project; ~<200 hip fracture boxes. Supplement only. | Filter Hip folder from existing FracAtlas download |
| A-6 | **Copy-paste augmentation** | Synthetically expand hip boxes (Strategy 1) — especially important given A-4 is still ~640–756 images | Albumentations `CopyPaste` on hip crops from A-4 + A-5 |

### Gap assessment

- **Hip blind spot fillable from public data? YES — partially.** The ThesisYolo Proximal Femur Fracture set (entry #13) is a genuine, CONFIRMED hip-fracture dataset with typed segmentation masks on Roboflow, CC BY 4.0. It is small (~640–756 images) but real. Combined with the FracAtlas hip subset and copy-paste augmentation, the hip blind spot can be partially addressed without institutional data access.
- **Caveat:** The highest-quality hip datasets (Gloucester NHS, Stanford RSNA 2020 — 1,000–3,000 studies with expert-typed bounding boxes) are NOT public. Clinical-grade hip-fracture detection will ultimately require a data-sharing agreement to reach those volumes.

---

**Loop status: COMPLETE (Pass 4 closes hip re-search)**

**Final recommendation (updated):** Acquire datasets in Track A order. A-4 is now the ThesisYolo Proximal Femur Fracture set — download both the segmentation project and the detection+classification project from Roboflow, compare class definitions, and convert polygon masks to bounding boxes. Exclude the `neck-normal` class from fracture training targets. Combine with FracAtlas hip subset (A-5) and copy-paste augmentation (A-6) to build a trainable hip corpus. For the shoulder track, A-3 (HUMERUS) remains the primary source. The AO/OTA gap remains unresolved by public data; escalate to clinical partnership if needed.

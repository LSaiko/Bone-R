# FracAtlas Class and Region Balance Analysis

## Entry — 2026-06-12

### Summary

**Dataset Overview:** The FracAtlas dataset contains 4,083 total X-ray images, of which 717 (17.56%) are fractured. The dataset exhibits significant imbalance across body regions and fracture types:

1. **Critical Blind Spots:** Hip (18.64% fracture rate, 63 fractured) and shoulder (18.05% fracture rate, 63 fractured) are severely under-represented compared to hand (438 fractured) and leg (263 fractured). Hip and shoulder together account for only 18.1% of fractured examples, while hand and leg account for 76.4%.

2. **Leg Fractures Underdetected:** Leg is the most common body region in the dataset (55.6% of all images) but has the lowest fracture rate (11.57%), indicating either fewer leg fractures in the population or labeling inconsistency. The model will likely overpredict non-fractured leg images.

3. **Train/Test Alignment:** The harmonized split (train/val/test) is broadly representative of the full dataset, with fracture rates stable across splits (17.86% / 16.99% / 16.18%). However, the test set shows concerning underrepresentation in shoulder (only 4 fractured images) and oblique view (only 3 fractured images), which limits generalization assessment.

4. **View Imbalance:** Oblique views are rare (418 / 4,083 = 10.2%) and have the lowest fracture rate (10.77%). The model will have minimal exposure to rare, challenging oblique-angle fractures, particularly in the test set.

---

### Tables

#### **Full FracAtlas Dataset Summary**

| Metric | Count | Percentage |
|--------|-------|-----------|
| **Total Images** | 4,083 | 100.0% |
| **Fractured** | 717 | 17.56% |
| **Non-Fractured** | 3,366 | 82.44% |

#### **Body Region Distribution (Full Dataset)**

| Region | Total Images | Fractured | Non-Fractured | Fracture Rate |
|--------|-------------|-----------|---------------|---------------|
| **Hand** | 1,538 | 438 | 1,100 | 28.48% |
| **Leg** | 2,273 | 263 | 2,010 | 11.57% |
| **Hip** | 338 | 63 | 275 | 18.64% |
| **Shoulder** | 349 | 63 | 286 | 18.05% |
| **Mixed** | 398 | 106 | 292 | 26.63% |

**Regional Imbalance Severity:** Hand and leg represent 89.3% of all fractured images (701/717), leaving hip and shoulder with only 126 fractured examples (17.6% of fractured cases) despite being anatomically significant injuries.

#### **View Distribution (Full Dataset)**

| View | Total Images | Fractured | Non-Fractured | Fracture Rate |
|------|-------------|-----------|---------------|---------------|
| **Frontal** | 2,503 | 438 | 2,065 | 17.50% |
| **Lateral** | 1,492 | 326 | 1,166 | 21.85% |
| **Oblique** | 418 | 45 | 373 | 10.77% |

**View Imbalance:** Oblique views comprise only 10.2% of the dataset and have the lowest fracture rate. The model will be least exposed to this challenging viewpoint.

#### **Fracture Count Distribution (Among 717 Fractured Images)**

| Number of Fractures | Count | Percentage |
|-------------------|-------|-----------|
| **1 fracture** | 546 | 76.15% |
| **2 fractures** | 146 | 20.36% |
| **3 fractures** | 17 | 2.37% |
| **4 fractures** | 7 | 0.98% |
| **5 fractures** | 1 | 0.14% |

Most fractured images contain a single fracture (76%), limiting multi-fracture learning. Complex cases (3+ fractures) are extremely rare.

---

### Harmonized Split Analysis

#### **Train Split (3,063 images, 75.0% of dataset)**

| Region | Total | Fractured | Rate |
|--------|-------|-----------|------|
| **Hand** | 1,136 | 329 | 28.96% |
| **Leg** | 1,725 | 204 | 11.83% |
| **Hip** | 240 | 49 | 20.42% |
| **Shoulder** | 260 | 50 | 19.23% |
| **Mixed** | 286 | 82 | 28.67% |

**Train Fracture Rate: 17.86%**

| View | Total | Fractured | Rate |
|------|-------|-----------|------|
| **Frontal** | 1,887 | 341 | 18.07% |
| **Lateral** | 1,110 | 235 | 21.17% |
| **Oblique** | 306 | 32 | 10.46% |

---

#### **Validation Split (612 images, 15.0% of dataset)**

| Region | Total | Fractured | Rate |
|--------|-------|-----------|------|
| **Hand** | 254 | 72 | 28.35% |
| **Leg** | 318 | 31 | 9.75% |
| **Hip** | 53 | 9 | 16.98% |
| **Shoulder** | 57 | 9 | 15.79% |
| **Mixed** | 67 | 16 | 23.88% |

**Val Fracture Rate: 16.99%**

| View | Total | Fractured | Rate |
|------|-------|-----------|------|
| **Frontal** | 361 | 56 | 15.51% |
| **Lateral** | 245 | 57 | 23.27% |
| **Oblique** | 59 | 10 | 16.95% |

---

#### **Test Split (408 images, 10.0% of dataset)**

| Region | Total | Fractured | Rate |
|--------|-------|-----------|------|
| **Hand** | 148 | 37 | 25.00% |
| **Leg** | 230 | 28 | 12.17% |
| **Hip** | 45 | 5 | 11.11% |
| **Shoulder** | 32 | 4 | 12.50% |
| **Mixed** | 45 | 8 | 17.78% |

**Test Fracture Rate: 16.18%**

| View | Total | Fractured | Rate |
|------|-------|-----------|------|
| **Frontal** | 255 | 41 | 16.08% |
| **Lateral** | 137 | 34 | 24.82% |
| **Oblique** | 53 | 3 | 5.66% |

**Test-Set Red Flags:**
- Shoulder has only 4 fractured examples (vs. 50 in train, 9 in val), limiting evaluation of shoulder-fracture performance.
- Oblique view has only 3 fractured examples (vs. 32 in train, 10 in val), making it impossible to robustly assess oblique-angle detection.
- Leg fracture rate drops to 12.17% in test vs. 11.83% in train—low denominator overall.

---

### Implications for the Model

1. **Performance Ceiling by Region:** Hand-fracture detection will be well-learned (438 training examples) and should achieve high accuracy, but hip and shoulder fractures will remain difficult due to minimal training signal (49–50 examples each). Expect 5–10% lower precision/recall on hip and shoulder fractures.

2. **Class Imbalance Optimization Needed:** With only 17.56% positive examples, the model should use weighted loss functions or oversampling of fractured images (especially hip/shoulder) to avoid learning a trivial "always negative" classifier. Threshold tuning during validation is critical.

3. **Oblique View as the Real Challenge:** Despite 418 oblique-view images in the dataset, only 45 (10.77%) contain fractures. The model will see fractured oblique images sparingly during training and validation, and almost never in test (3 examples). Consider augmenting fractured oblique examples or using domain adaptation techniques.

4. **Data Collection Priority:** Hip, shoulder, and oblique-view fractures should be prioritized for future data collection. A 2–3x increase in fractured hip/shoulder examples and fractured oblique views would significantly improve generalization. Current test set is insufficient to validate performance on these sub-groups.


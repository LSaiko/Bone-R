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

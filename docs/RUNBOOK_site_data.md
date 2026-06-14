# Site Data Regeneration Runbook

This document describes the end-to-end manual workflow to regenerate the landing page data files (`docs/data/benchmark.json` and `docs/data/predictions.json`) from trained model weights.

## Overview

The landing page displays model benchmark metrics and prediction overlays. These data files are generated offline from trained model weights via two scripts:
- `benchmark.py` — runs inference on a benchmark dataset, produces `docs/data/benchmark.json`
- `compare_overlay.py` — runs inference on a test X-ray image, produces `docs/data/predictions.json`

This runbook covers three workflows:
1. **Local regeneration** — for development and testing
2. **GitHub Actions regeneration** — for production, via artifact upload
3. **Validation** — catching stale or malformed data before commit

---

## Prerequisites

- Python 3.10+ (3.11 recommended)
- Trained model weights (`.pt` files for yolov8, retinanet, fasterrcnn)
- Test dataset and benchmark YAML (`dataset.yaml`)
- Test X-ray image (`docs/assets/demo.png` or custom path)

## Local Workflow

### Step 1: Prepare Model Weights

Ensure you have trained weights on your local machine. Place them in a known directory, e.g. `weights/`:
```
weights/
├── yolov8/
│   └── best.pt
├── retinanet.pt
└── fasterrcnn.pt
```

### Step 2: Run Benchmark Script

Generate `docs/data/benchmark.json` by running inference on your benchmark dataset:

```powershell
python benchmark.py `
  --data dataset.yaml `
  --model yolov8=weights/yolov8/best.pt `
  --model retinanet=weights/retinanet.pt `
  --model fasterrcnn=weights/fasterrcnn.pt `
  --out docs/data/benchmark.json
```

Expected output: `docs/data/benchmark.json` with structure:
```json
{
  "yolov8": { "map50": 0.62, "map5095": 0.31, "precision": 0.71, "recall": 0.66 },
  "retinanet": { "map50": 0.58, "map5095": 0.29, "precision": 0.64, "recall": 0.72 },
  "fasterrcnn": { "map50": 0.6, "map5095": 0.33, "precision": 0.78, "recall": 0.59 }
}
```

### Step 3: Run Overlay Script

Generate `docs/data/predictions.json` and a visual overlay by running inference on a test X-ray:

```powershell
python compare_overlay.py `
  --image docs/assets/demo.png `
  --model yolov8=weights/yolov8/best.pt `
  --model retinanet=weights/retinanet.pt `
  --model fasterrcnn=weights/fasterrcnn.pt `
  --json docs/data/predictions.json `
  --out docs/assets/compare-overlay.png
```

Expected output:
- `docs/data/predictions.json` — predictions from all three models on the test image
- `docs/assets/compare-overlay.png` — visual side-by-side comparison

### Step 4: Validate Locally

Run the validation script to ensure JSON is well-formed and conforms to schema:

```powershell
python scripts/validate_site_data.py --docs-dir docs
```

Expected output (exit code 0):
```
Validating C:\Users\Admin\Projects\Bone-R\docs\data\predictions.json...
  [OK] Valid JSON
  [OK] Required keys present: height, image, models, width
  ...

Validating C:\Users\Admin\Projects\Bone-R\docs\data\benchmark.json...
  [OK] Valid JSON
  [OK] 3 models found
  ...

All validations passed!
```

### Step 5: Commit and Push

Once validation passes, commit the regenerated files:

```powershell
git add docs/data/benchmark.json docs/data/predictions.json docs/assets/compare-overlay.png
git commit -m "Regenerate site data from trained weights"
git push origin main
```

The CI pipeline will re-validate on push and deploy to GitHub Pages if validation passes.

---

## GitHub Actions Workflow (Artifact-Based)

### For Release Engineers

If you want to trigger regeneration via GitHub Actions without sharing weights in the repo:

#### Step 1: Upload Weights as Artifact

In your training job or a separate CI step, upload trained weights as an artifact named `model-weights` (or any name you choose):

```yaml
- name: Upload weights artifact
  uses: actions/upload-artifact@v4
  with:
    name: model-weights
    path: weights/
    retention-days: 7
```

#### Step 2: Manually Trigger Regeneration Workflow

1. Go to **Actions** → **Validate Site Data**
2. Click **Run workflow** (top-right dropdown)
3. Set:
   - **Branch**: `main`
   - **weights_artifact**: `model-weights` (or your artifact name)
   - **test_image** (optional): `docs/assets/demo.png` (default)
4. Click **Run workflow**

#### Step 3: Wait for Workflow Completion

The workflow will:
1. Run validation on current `docs/data/` (quick sanity check)
2. Download your weights artifact
3. Run `benchmark.py` and `compare_overlay.py`
4. Validate the regenerated JSON
5. Upload regenerated files as artifact `regenerated-site-data`

#### Step 4: Download & Review Artifact

If workflow succeeds:
1. Go to workflow run summary
2. Download `regenerated-site-data` artifact
3. Inspect `benchmark.json` and `predictions.json` locally
4. If satisfied, manually commit and push (no auto-merge in CI)

#### Step 5: Commit Regenerated Files

```powershell
# Extract artifact contents
Expand-Archive regenerated-site-data.zip -DestinationPath docs-new
Move-Item docs-new\* docs\data\ -Force
rm -Recurse docs-new

# Validate again locally (optional)
python scripts/validate_site_data.py --docs-dir docs

# Commit
git add docs/data/
git commit -m "Regenerate site data from GitHub Actions weights artifact"
git push origin main
```

---

## Pre-Commit Hook (Local Validation Gate)

To prevent accidental commits of invalid JSON, set up a pre-commit hook that runs the validator:

### Step 1: Configure Git Hooks Directory

```powershell
git config core.hooksPath scripts
```

This tells git to look for hooks in the `scripts/` directory.

### Step 2: Create Pre-Commit Hook

The hook at `scripts/pre_commit_validate.py` is called automatically before each commit. It runs the validator and blocks the commit if `docs/data/*.json` is invalid.

To see the hook in action, try committing a broken JSON file:

```powershell
# Edit docs/data/benchmark.json manually, break it (e.g., remove a comma)
git add docs/data/benchmark.json
git commit -m "Test broken JSON"
```

Expected output (commit blocked):
```
[Validate site data before commit]
Validating C:\Users\Admin\Projects\Bone-R\docs\data\predictions.json...
  [OK] Valid JSON
  ...

Validating C:\Users\Admin\Projects\Bone-R\docs\data\benchmark.json...
  [FAIL] Invalid JSON: Expecting ',' delimiter: line 2 column 1
Validation failed!
```

Commit will be aborted. Fix the JSON and try again.

### Step 3: Bypass Hook (If Needed)

If you need to commit without validation (e.g., reverting a file), use:

```powershell
git commit --no-verify
```

---

## Nightly Validation Job

The `.github/workflows/site-data.yml` includes a **scheduled job** that validates `docs/data/` every day at **02:00 UTC**.

If validation fails on the nightly run, GitHub will send a notification. This catches stale or corrupted data introduced by a force-push or manual edit.

To check nightly job results:
- Go to **Actions** → **Validate Site Data**
- Filter by **Schedule** trigger
- Review failed runs

---

## Troubleshooting

### Validation Fails: Missing Keys

**Error**: `[FAIL] Missing required keys: {'map50', 'map5095'}`

**Fix**: Ensure `benchmark.py` completed and produced valid JSON. Check script logs for GPU/dependency errors.

### Validation Fails: Type Mismatch

**Error**: `[FAIL] 'width' must be positive int, got <class 'str'>`

**Fix**: Edit the JSON file by hand and ensure numeric values are not quoted. E.g., `"width": 640` not `"width": "640"`.

### Workflow_Dispatch Not Triggering Regenerate Job

**Check**: Ensure `weights_artifact` input is non-empty. If empty, the regenerate job is skipped (by design).

**Fix**: 
```powershell
# On Run workflow form:
# - weights_artifact: model-weights   <-- must not be blank
```

### Pre-Commit Hook Not Firing

**Check**: Verify hooks directory is configured:
```powershell
git config core.hooksPath
# Should output: scripts
```

**Fix** (if missing):
```powershell
git config core.hooksPath scripts
```

---

## Summary of Automation

| Trigger | Action | Output |
|---------|--------|--------|
| **Manual local run** | Developer runs `benchmark.py` + `compare_overlay.py` + validator | `docs/data/*.json` (local) |
| **Push to main** (docs/ changed) | GitHub Actions runs validator | Pass/fail check; blocks deploy if invalid |
| **Nightly (02:00 UTC)** | GitHub Actions runs validator on current main | Alert if data is stale/broken |
| **Workflow_dispatch** (+ weights artifact) | GitHub Actions runs regenerate job | `regenerated-site-data` artifact for review |
| **Pre-commit hook** | Developer `git commit` | Block commit if JSON invalid |

---

## Files Involved

- `benchmark.py` — benchmark script (trains/infers on dataset, outputs benchmark.json)
- `compare_overlay.py` — overlay script (infers on image, outputs predictions.json + PNG)
- `scripts/validate_site_data.py` — validator script (checks JSON schema, used in CI and pre-commit)
- `scripts/pre_commit_validate.py` — pre-commit hook wrapper (calls validator)
- `.github/workflows/site-data.yml` — CI workflow (validation + optional regeneration + nightly cron)
- `docs/data/benchmark.json` — landing page benchmark metrics
- `docs/data/predictions.json` — landing page predictions overlay data
- `docs/assets/demo.png` — test X-ray image for overlay
- `docs/assets/compare-overlay.png` — generated visual comparison (committed to repo)

---

**Last updated**: 2026-06-12

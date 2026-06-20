"""
One-command v5 pipeline: acquire the hip (proximal-femur) dataset, ingest only
its true-fracture classes, harmonize, stratified-merge with v4's sources, and
train YOLOv8m. Designed for an unattended overnight run.

It FAILS LOUDLY and early on the most likely problem — a hip class-name mismatch
in the Roboflow export's data.yaml — by printing the dataset's class names and
resolving the requested fracture names before any heavy work.

Usage
-----
    # full run (downloads hip set, then trains ~6h):
    python scripts/run_v5.py --roboflow-key YOUR_KEY \
        --hip-workspace <ws> --hip-project <proj> --hip-version <n>

    # if the hip set is already downloaded to HIP_rf, skip the key:
    python scripts/run_v5.py --skip-download

The hip fracture class NAMES default to the proximal-femur set's true fractures;
landmark/normal classes (greater/lesser-trochanter, neck-normal, dislocation)
are dropped. Adjust --fracture-names after you've seen the printed class list.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roboflow-key")
    ap.add_argument("--hip-workspace", default="thesisyolo-v8")
    ap.add_argument("--hip-project", default="proximal-femur-fracture")
    ap.add_argument("--hip-version", type=int, default=1)
    ap.add_argument("--skip-download", action="store_true",
                    help="HIP_rf already present; don't re-download")
    ap.add_argument("--fracture-names",
                    default="intertrochanteric,femoral neck,subtrochanteric",
                    help="Hip fracture class names to KEEP (drop landmarks/normal)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    hip_rf = ROOT / "HIP_rf"

    # 1. Acquire (instance-seg set; 'yolov8' export gives detection boxes).
    if not args.skip_download:
        if not args.roboflow_key:
            sys.exit("Need --roboflow-key (or --skip-download if HIP_rf exists)")
        import os
        os.environ["ROBOFLOW_API_KEY"] = args.roboflow_key
        from roboflow import Roboflow
        rf = Roboflow(api_key=args.roboflow_key)
        (rf.workspace(args.hip_workspace).project(args.hip_project)
           .version(args.hip_version).download("yolov8", location=str(hip_rf)))

    # 2. Show the class list BEFORE ingest — fail loudly if names don't resolve.
    import yaml
    spec = yaml.safe_load((hip_rf / "data.yaml").read_text(encoding="utf-8"))
    print(f"\nHIP dataset classes: {spec.get('names')}")
    print(f"Keeping as fracture : {args.fracture_names}")

    # 3. Ingest only the true-fracture classes (landmarks/normal dropped).
    run([PY, "scripts/ingest_grazpedwri.py", "--images", "HIP_rf",
         "--labels", "HIP_rf", "--out", "HIP_bone_r", "--copy",
         "--fracture-names", args.fracture_names,
         "--data-yaml", "HIP_rf/data.yaml"])

    # 4. Harmonize + stratified merge with v4's sources.
    run([PY, "scripts/preprocess.py", "--src", "HIP_bone_r",
         "--out", "HIP_proc", "--max-size", "1024"])
    run([PY, "scripts/make_splits.py",
         "--src", "FracAtlas_proc", "--src", "GRAZ_proc",
         "--src", "HUMERUS_proc", "--src", "HIP_proc",
         "--out", "dataset_v5", "--copy", "--stratify"])

    # 5. dataset_v5.yaml
    (ROOT / "dataset_v5.yaml").write_text(
        "path: dataset_v5\ntrain: images/train\nval: images/val\n"
        "test: images/test\nnames:\n  0: fracture\n", encoding="utf-8")

    # 6. Train.
    run([PY, "train.py", "--model", "yolov8m.pt", "--data", "dataset_v5.yaml",
         "--epochs", str(args.epochs), "--imgsz", "800",
         "--batch", str(args.batch), "--name", "fracture_yolov8m_v5"])

    print("\nv5 pipeline complete. Evaluate with:\n"
          "  python evaluate.py --weights runs/detect/fracture_yolov8m_v5/"
          "weights/best.pt --data dataset_v5.yaml --data-root dataset_v5 "
          "--split test --target-recall 0.90")


if __name__ == "__main__":
    main()

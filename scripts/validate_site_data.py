"""
Validate that landing page data files (benchmark.json and predictions.json)
are well-formed JSON and conform to the expected schema.

This script has NO external dependencies (only stdlib) and NO GPU requirements.
It can be run locally by developers or in CI to gate site deployment.

Usage
-----
    python scripts/validate_site_data.py [--docs-dir DOCS_DIR]

Exit codes:
    0 = all validations passed
    1 = validation failed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def validate_predictions_json(path: Path) -> bool:
    """
    Validate predictions.json schema.

    Expected structure:
    {
        "image": str,
        "width": int,
        "height": int,
        "_note"?: str,  # optional
        "models": {
            "<model_name>": {
                "color": str (hex, e.g. "#00c800"),
                "boxes": [
                    {
                        "xyxy": [x1, y1, x2, y2],
                        "conf": float,
                        "type": str,
                        "severity": str
                    },
                    ...
                ]
            },
            ...
        }
    }
    """
    print(f"Validating {path}...")

    # Parse JSON
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("  [OK] Valid JSON")
    except json.JSONDecodeError as e:
        print(f"  [FAIL] Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] Error reading file: {e}")
        return False

    # Validate required top-level keys
    required_keys = {"image", "width", "height", "models"}
    if not all(k in data for k in required_keys):
        missing = required_keys - set(data.keys())
        print(f"  [FAIL] Missing required keys: {missing}")
        return False
    print(f"  [OK] Required keys present: {', '.join(sorted(required_keys))}")

    # Validate types and values
    if not isinstance(data["image"], str):
        print(f"  [FAIL] 'image' must be str, got {type(data['image']).__name__}")
        return False
    print(f"  [OK] image type: str ({data['image']})")

    if not isinstance(data["width"], int) or data["width"] <= 0:
        print(f"  [FAIL] 'width' must be positive int, got {data['width']}")
        return False
    print(f"  [OK] width type: int ({data['width']})")

    if not isinstance(data["height"], int) or data["height"] <= 0:
        print(f"  [FAIL] 'height' must be positive int, got {data['height']}")
        return False
    print(f"  [OK] height type: int ({data['height']})")

    # Validate models object
    models = data["models"]
    if not isinstance(models, dict):
        print(f"  [FAIL] 'models' must be dict, got {type(models).__name__}")
        return False

    if not models:
        print("  [WARN] 'models' is empty (no predictions)")
        return True

    print(f"  [OK] {len(models)} models found")

    # Validate each model
    for model_name, model_data in models.items():
        if not isinstance(model_data, dict):
            print(f"  [FAIL] model '{model_name}' data must be dict, got {type(model_data).__name__}")
            return False

        # Check required keys in model
        if "color" not in model_data or "boxes" not in model_data:
            print(f"  [FAIL] model '{model_name}' missing 'color' or 'boxes'")
            return False

        # Validate color (hex format)
        color = model_data["color"]
        if not isinstance(color, str) or not color.startswith("#") or len(color) != 7:
            print(f"  [FAIL] model '{model_name}' color must be hex string like '#00c800', got {color}")
            return False

        # Validate boxes list
        boxes = model_data["boxes"]
        if not isinstance(boxes, list):
            print(f"  [FAIL] model '{model_name}' boxes must be list, got {type(boxes).__name__}")
            return False

        # Validate each box
        for i, box in enumerate(boxes):
            if not isinstance(box, dict):
                print(f"  [FAIL] model '{model_name}' box {i} must be dict, got {type(box).__name__}")
                return False

            # Check required keys
            box_required = {"xyxy", "conf", "type", "severity"}
            if not all(k in box for k in box_required):
                missing = box_required - set(box.keys())
                print(f"  [FAIL] model '{model_name}' box {i} missing: {missing}")
                return False

            # Validate xyxy format
            xyxy = box["xyxy"]
            if not isinstance(xyxy, list) or len(xyxy) != 4:
                print(f"  [FAIL] model '{model_name}' box {i} xyxy must be 4-element list, got {xyxy}")
                return False

            if not all(isinstance(x, (int, float)) for x in xyxy):
                print(f"  [FAIL] model '{model_name}' box {i} xyxy must contain numbers, got {xyxy}")
                return False

            # Validate conf (0-1 range, typically)
            if not isinstance(box["conf"], (int, float)):
                print(f"  [FAIL] model '{model_name}' box {i} conf must be number, got {box['conf']}")
                return False

            # Validate type and severity are strings
            if not isinstance(box["type"], str):
                print(f"  [FAIL] model '{model_name}' box {i} type must be str, got {box['type']}")
                return False

            if not isinstance(box["severity"], str):
                print(f"  [FAIL] model '{model_name}' box {i} severity must be str, got {box['severity']}")
                return False

        print(f"  [OK] {model_name}: color={color}, {len(boxes)} boxes")

    return True


def validate_benchmark_json(path: Path) -> bool:
    """
    Validate benchmark.json schema.

    Expected structure:
    {
        "_note"?: str,  # optional
        "<model_name>": {
            "map50": float | null,
            "map5095": float | null,
            "precision": float | null,
            "recall": float | null
        },
        ...
    }
    """
    print(f"Validating {path}...")

    # Parse JSON
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("  [OK] Valid JSON")
    except json.JSONDecodeError as e:
        print(f"  [FAIL] Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] Error reading file: {e}")
        return False

    if not isinstance(data, dict):
        print(f"  [FAIL] Root must be dict, got {type(data).__name__}")
        return False

    # Count models (exclude _note)
    models = {k: v for k, v in data.items() if not k.startswith("_")}

    if not models:
        print("  [WARN] No models found (only metadata)")
        return True

    print(f"  [OK] {len(models)} models found")

    # Validate each model
    for model_name, model_data in models.items():
        if not isinstance(model_data, dict):
            print(f"  [FAIL] model '{model_name}' data must be dict, got {type(model_data).__name__}")
            return False

        # Check required keys
        required = {"map50", "map5095", "precision", "recall"}
        if not all(k in model_data for k in required):
            missing = required - set(model_data.keys())
            print(f"  [FAIL] model '{model_name}' missing keys: {missing}")
            return False

        # Validate each field (must be float, int, or null)
        for key in required:
            value = model_data[key]
            if value is not None and not isinstance(value, (int, float)):
                print(f"  [FAIL] model '{model_name}' {key} must be number or null, got {type(value).__name__}")
                return False

        # Build a display string
        parts = []
        for key in ["map50", "map5095", "precision", "recall"]:
            val = model_data[key]
            if val is None:
                parts.append(f"{key}=null")
            else:
                parts.append(f"{key}={val}")

        print(f"  [OK] {model_name}: {', '.join(parts)}")

    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate landing page data files (benchmark.json and predictions.json)"
    )
    ap.add_argument(
        "--docs-dir",
        type=Path,
        default=Path(__file__).parent.parent / "docs",
        help="Path to docs directory (default: ../docs relative to scripts/)"
    )
    args = ap.parse_args()

    docs_dir = args.docs_dir
    data_dir = docs_dir / "data"

    # Check that data directory exists
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(1)

    # Validate both files
    predictions_path = data_dir / "predictions.json"
    benchmark_path = data_dir / "benchmark.json"

    all_valid = True

    if predictions_path.exists():
        if not validate_predictions_json(predictions_path):
            all_valid = False
        print()
    else:
        print(f"Warning: {predictions_path} not found, skipping")
        print()

    if benchmark_path.exists():
        if not validate_benchmark_json(benchmark_path):
            all_valid = False
        print()
    else:
        print(f"Warning: {benchmark_path} not found, skipping")
        print()

    if all_valid:
        print("All validations passed!")
        sys.exit(0)
    else:
        print("Validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

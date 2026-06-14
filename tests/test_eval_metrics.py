"""
Unit tests for the pure helper functions in evaluate.py.

Run with:
    python tests/test_eval_metrics.py

No model, no GPU, no ultralytics import required.
These tests exercise the confusion-matrix math and the region-lookup logic
so regressions in the clinical metric code are caught quickly.
"""

import sys
import math
from pathlib import Path

# Ensure the project root is on sys.path so we can import evaluate directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate import image_level_stats, region_of


# ---------------------------------------------------------------------------
# image_level_stats
# ---------------------------------------------------------------------------

def test_perfect_classifier():
    """All predictions correct: sensitivity=1, specificity=1, etc."""
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 1, 0, 0, 1, 0]
    s = image_level_stats(y_true, y_pred)
    assert s["TP"] == 3
    assert s["TN"] == 3
    assert s["FP"] == 0
    assert s["FN"] == 0
    assert s["sensitivity"] == 1.0
    assert s["specificity"] == 1.0
    assert s["PPV"] == 1.0
    assert s["NPV"] == 1.0
    assert s["accuracy"] == 1.0


def test_all_wrong():
    """Every prediction is inverted."""
    y_true = [1, 1, 0, 0]
    y_pred = [0, 0, 1, 1]
    s = image_level_stats(y_true, y_pred)
    assert s["TP"] == 0
    assert s["FN"] == 2
    assert s["FP"] == 2
    assert s["TN"] == 0
    assert s["sensitivity"] == 0.0
    assert s["specificity"] == 0.0
    assert s["accuracy"] == 0.0


def test_mixed():
    """Realistic mixed case: 3 fracture images, 2 normal."""
    # GT:   1  1  1  0  0
    # Pred: 1  0  1  0  1
    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 0, 1, 0, 1]
    s = image_level_stats(y_true, y_pred)
    assert s["TP"] == 2
    assert s["FN"] == 1
    assert s["TN"] == 1
    assert s["FP"] == 1
    # sensitivity = 2/3
    assert abs(s["sensitivity"] - 2 / 3) < 1e-9
    # specificity = 1/2
    assert abs(s["specificity"] - 1 / 2) < 1e-9
    # PPV = 2/3
    assert abs(s["PPV"] - 2 / 3) < 1e-9
    # NPV = 1/2
    assert abs(s["NPV"] - 1 / 2) < 1e-9
    # accuracy = 3/5
    assert abs(s["accuracy"] - 3 / 5) < 1e-9


def test_all_positive_gt():
    """No negatives in ground truth: specificity is NaN (no TN+FP denominator).
    NPV = TN/(TN+FN); if there are FN the denominator is non-zero, so NPV is 0."""
    y_true = [1, 1, 1]
    y_pred = [1, 1, 0]
    s = image_level_stats(y_true, y_pred)
    assert abs(s["sensitivity"] - 2 / 3) < 1e-9
    assert math.isnan(s["specificity"])   # TN+FP == 0 → NaN
    # TN=0, FN=1 → NPV = 0/1 = 0.0 (not NaN)
    assert s["NPV"] == 0.0
    assert s["PPV"] == 1.0  # 2 TP, 0 FP


def test_all_negative_gt():
    """No positives in ground truth: sensitivity and PPV are NaN."""
    y_true = [0, 0, 0]
    y_pred = [0, 1, 0]
    s = image_level_stats(y_true, y_pred)
    assert abs(s["specificity"] - 2 / 3) < 1e-9
    assert math.isnan(s["sensitivity"])  # TP+FN == 0 → NaN
    # PPV = TP/(TP+FP) = 0/1 = 0.0 (denominator is non-zero, so not NaN)
    assert s["PPV"] == 0.0
    assert s["NPV"] == 1.0  # TN=2, FN=0 → perfect negative predictive value



def test_single_true_positive():
    y_true = [1]
    y_pred = [1]
    s = image_level_stats(y_true, y_pred)
    assert s["TP"] == 1 and s["FP"] == 0 and s["TN"] == 0 and s["FN"] == 0
    assert s["sensitivity"] == 1.0
    assert s["accuracy"] == 1.0


def test_length_mismatch_raises():
    try:
        image_level_stats([1, 0], [1])
        assert False, "should have raised ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# region_of
# ---------------------------------------------------------------------------

def _make_index():
    """Minimal synthetic csv_index for testing."""
    return {
        "IMG0001547": "leg",
        "IMG0000002": "hand",
        "IMG0000010": "hip",
        "IMG0000020": "shoulder",
        "IMG0000030": "mixed",
    }


def test_region_prefixed_filename():
    """make_splits.py renames files as NNNNNN_IMGXXXXXXX.ext — strip the prefix."""
    idx = _make_index()
    assert region_of("000123_IMG0001547.png", idx) == "leg"
    assert region_of("000002_IMG0000002.jpg", idx) == "hand"


def test_region_unprefixed_filename():
    """Original filename with no numeric prefix also works."""
    idx = _make_index()
    assert region_of("IMG0000010.png", idx) == "hip"


def test_region_unknown_id():
    """Image not in CSV returns 'unknown'."""
    idx = _make_index()
    assert region_of("000999_IMG9999999.png", idx) == "unknown"


def test_region_no_extension():
    """Stem without extension is handled."""
    idx = _make_index()
    assert region_of("000002_IMG0000002", idx) == "hand"


def test_region_shoulder_and_mixed():
    idx = _make_index()
    assert region_of("IMG0000020.jpg", idx) == "shoulder"
    assert region_of("000030_IMG0000030.jpg", idx) == "mixed"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_perfect_classifier,
        test_all_wrong,
        test_mixed,
        test_all_positive_gt,
        test_all_negative_gt,
        test_single_true_positive,
        test_length_mismatch_raises,
        test_region_prefixed_filename,
        test_region_unprefixed_filename,
        test_region_unknown_id,
        test_region_no_extension,
        test_region_shoulder_and_mixed,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed.")
    if failed:
        sys.exit(1)
